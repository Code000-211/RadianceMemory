/* gemma4_bf16.cpp -- Gemma 4 dense, bf16, with its vision tower (rad_block_gemma4_vision.h) and,
 * for E2B/E4B, its audio encoder (rad_block_gemma4_audio.h); for the 12B ("unified"), its image and
 * audio embedders in their place (rad_block_gemma4_unified.h).
 *
 * Two attention classes over one residual stream, alternating by layer. A "sliding" layer is
 * sliding-window attention over head_dim 256 with a v projection and standard RoPE; a "global"
 * layer is windowless attention over head_dim 512 with no v projection (k == v) and proportional
 * RoPE (the "head" ladder) over the partial rotary width. Norms are plain RMSNorm, x_hat * w (see
 * declare), with a second norm on each projection's output before its residual, a weightless RMS
 * norm on the value, and a GELU (not SwiGLU) feed-forward. The final logits are soft-capped to
 * cap * tanh(x / cap).
 *
 * Three draft models, whichever the container carries under `draft.`: Google's assistant head (a
 * serial head over the trunk's own cache, the Gemma4Head below), z-lab's DFlash drafter (a
 * five-layer block drafter with its own cache, reading the trunk's residual stream at six layers,
 * arch/common/rad_gemma4_dflash.h; 15 tokens a pass on the 31B), or a DFlash2 drafter such as
 * RedHatAI's gemma-4-31B-it-speculator.dflash2 (arch/common/rad_block_dflash2.h; 7 a pass).
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_gemma4_audio.h>
#include <arch/rad_block_gemma4_unified.h>
#include <arch/rad_block_gemma4_vision.h>
#include <arch/rad_gemma4_chat.h>
#include <arch/rad_gemma4_dflash.h>
#include <arch/rad_block_dflash2.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace gemma4_bf16 {

using namespace rad::arch;

/* Residual junctions as one op each. A w4a8 block opens with rmsnorm -> quant_act_i8 and closes
 * with rmsnorm (post) -> add -> [scale], so between two blocks the stream runs five single-row
 * dispatches: the previous post-norm, the add, the layer scalar, this block's norm and its
 * quantiser. libr4d's `rmsnorm_quant_i8` is that chain in one byte-identical dispatch, and q, k and
 * v read one quantised pair instead of three. `qkv_norm_rope` does the same for the attention
 * prologue's five ops, `gelu_mul_quant_i8` for down's input, and at two ranks
 * `ar_rmsnorm_quant_i8` also folds in the all-reduce before each junction. A kernel set without an
 * op keeps that op's chain.
 *
 * The linears on quant_act_i8g's grid (i8a8, a4 and its five-bit form) take the `_i8g` twins of the
 * three: the same junctions with a scale per (row, 128) instead of per row. An a4 linear reads no
 * int8 pair up to its decode band (G4Linear::kA4Split rows) but an f16 copy of the bf16 norm output
 * (`cast`, then w4a16_asym or w5a16), so there its junction also writes that norm output where the
 * unfused norm would, into the linear's input row, and the cast and the skinny GEMM run as before. */

/* layer_types is optional. When present it must name every layer "full_attention" or
 * "sliding_attention"; when absent, layer l is global when (l + 1) % interval == 0. */
static int g4_parse_types(const char* lt, int64_t n, std::vector<int>& is_global) {
    RAD_ARCH_TRY(parse_layer_kinds("gemma4", lt, {"full_attention"}, "sliding_attention",
                                   &is_global));
    if ((int64_t)is_global.size() != n) {
        fprintf(stderr, "radiance: gemma4: layer_types names %zu layers and the model has %lld\n",
                is_global.size(), (long long)n);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* A config boolean, flattened as "true"/"false" (`use_double_wide_mlp`). */
static bool g4_getb(const RadModelMeta* m, const char* k, bool dflt) {
    const char* s = meta_text_gets(m, k, nullptr);
    if (!s) return dflt;
    if (!std::strcmp(s, "true") || !std::strcmp(s, "True")) return true;
    if (!std::strcmp(s, "false") || !std::strcmp(s, "False") || !std::strcmp(s, "null")) return false;
    return std::atoll(s) != 0;
}

static int g4_schedule(const RadModelMeta* m, int64_t n, int64_t interval,
                       std::vector<int>& is_global) {
    const char* lt = meta_text_gets(m, "layer_types", nullptr);
    if (lt && *lt) {
        RAD_ARCH_TRY(g4_parse_types(lt, n, is_global));
    } else {
        if (interval <= 0) {
            fprintf(stderr, "radiance: gemma4: no layer_types and invalid full_attention_interval %lld\n",
                    (long long)interval);
            return RAD_E_INVAL;
        }
        is_global.resize((size_t)n);
        for (int64_t l = 0; l < n; ++l) is_global[(size_t)l] = is_full_attention(l, interval);
    }
    return RAD_OK;
}

/* A linear's weight is its checkpoint's `X.weight`. Google's CT-INT4 (QAT) release ships
 * `X.weight_packed` and `X.weight_scale`, which the checkpoint reader imports as `X.weight`
 * (core/format/checkpoint.h), so one name map serves both releases. */

/* The two attention classes share the residual stream and the FFN but differ in head geometry,
 * the value path and the RoPE. The main Geom (from geom_from) is the slider; the global class is
 * read from the container head and folded in here. */
struct ClassGeom {
    int64_t n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
    int64_t window = 0;
    double  theta  = 1.0;
    float   scale  = 1.0f;
    bool    head_ladder = false;   /* true: proportional over head_dim (global) */
    bool    has_v       = true;    /* false: k == v (global) */
    /* Fewer KV heads than ranks: each rank keeps the one at row `kv_off` of the n_head_kv_all,
     * shared with the other world / n_head_kv_all ranks of its group. */
    bool    kv_rep      = false;
    int64_t kv_off = 0, n_head_kv_all = 0;
    int64_t q_dim()  const { return n_head * head_dim; }
    int64_t kv_dim() const { return n_head_kv * head_dim; }
    int64_t gqa()    const { return n_head / n_head_kv; }
};

/* ---- one linear, plain bf16 or w4a8. The recipe quantises GEMM weights to int4 codes with an
 * f16 scale per [1 x 128]; a weight no rule matches stays bf16. The weight's encoding picks the
 * op: `gemm_nt` for plain bf16, `gemm_nt_q` at w4a8 for codes + scale. w4a8 reads an int8
 * activation and an f32 per-row scale, written by a quant_act_i8 declared here over the bf16
 * activation, in the GEMM's fragment byte order. Not quant_act_i8g (row-major, per-(row, group)
 * scale; i8a8's producer), and not w4a16, whose row expects an f16 activation and does not check
 * the operand dtype, so bf16 bits would be misread as f16. The name map is made before the weight
 * is queried (Builder::finish() needs the map entry before the declaration). */
struct G4Linear {
    rad_weight w = 0, ws = 0;   /* the codes + scale (w4a8), or the single bf16 weight */
    rad_op     op = 0, op_q = 0;   /* the quantised GEMM and, for w4a8, its activation quantiser */
    rad_buf    q8 = 0, s8 = 0;     /* the int8 pair (i8 codes + f32 row scale) the quantiser writes */
    int64_t    N = 0, K = 0;
    bool       quant = false;   /* true: w4a8 / w8a8 over a per-row int8 activation */
    /* true: i8a8 as arch/gemma3's G3Linear: int8 codes with a bf16 scale per [1 x 128] (libquant
     * `codes=i8 group=128 scale=bf16`) over quant_act_i8g, an activation scale per (row, 128).
     * Gemma 4 E4B's activations carry a few channels 40-200 times the row's median (the down
     * projection's input most), and one int8 scale a row spends its codes on them. Its junctions
     * are the `_i8g` twins (see "Residual junctions" above). */
    bool       i8g = false;
    /* true: asymmetric int4 as arch/gemma3's G3Linear: u4 codes, a u8 zero point and an f16 scale
     * per [1 x 128] (libquant `codes=u4 zero=u8 group=128 scale=f16`), gemm_nt_q at w4a8i8_asym
     * over the same per-(row, 128) int8 activation as i8g. */
    bool       a4 = false;
    /* true with a4: the same grid at five bits, u5 codes and a zero point 0..31 (libquant
     * `codes=u5 zero=u8 group=128 scale=f16`), ParoQuant int5's grid with no rotation. Read by
     * w5a8i8 and w5a16 where a4 reads w4a8i8_asym and w4a16_asym, over the same activations. */
    bool       w5 = false;
    bool       grouped() const { return i8g || a4; }
    /* reads an int8 pair a junction can write, at either grid */
    bool       pair() const { return quant || grouped(); }
    /* the pair's scales a row: one, or one per 128 columns */
    int64_t    scale_w() const { return grouped() ? K / RAD_FP8_BLOCK : 1; }
    rad_buf    in = 0;            /* the bf16 input row `a` given at declare */
    /* a4 at decode heights, as LinearFP8's w4 (rad_fp8.h): up to kA4Split rows the bf16 input is
     * cast to f16 (`a16`) for w4a16_asym's skinny kernel; w4a8i8_asym has only a prefill kernel,
     * which at one row does 16 times the matrix work and reads its weight a quarter as fast. */
    static constexpr int64_t kA4Split = 64;
    int64_t    a4_split = 0;      /* rows up to which op_dec serves; 0: no decode band */
    rad_buf    a16 = 0;
    rad_op     op_a16 = 0, op_dec = 0;
    /* Reads another linear's int8 pair. q, k and v quantise the same norm output on the same
     * per-row grid, so with `in_from` set at declare this linear reads that linear's pair and
     * declares no quantiser of its own: one dispatch instead of three. */
    bool       shared_in = false;

    int  declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                 std::initializer_list<const char*> src,
                 int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                 rad_buf a, rad_buf y,
                 std::initializer_list<int64_t> row_parts = {},
                 const G4Linear* in_from = nullptr);
    void step(RadCtx* c, rad_buf a, rad_buf y, int64_t T) const;
    /* the GEMM alone, for a caller that wrote the int8 pair itself (a fused norm + quantiser); an
     * a4 linear's decode band reads the bf16 input row instead, which that caller then wrote */
    void gemm(RadCtx* c, rad_buf y, int64_t T) const;
    /* The same two over rows [r0, r0 + n) of the step: a slice of the prefill overlap. The
     * quantiser and the GEMM are a row at a time above their bands, so a slice's rows are the
     * whole step's (see "the prefill overlap" below). */
    void step_rows(RadCtx* c, rad_buf a, rad_buf y, int64_t r0, int64_t n) const;
    void gemm_rows(RadCtx* c, rad_buf y, int64_t r0, int64_t n) const;
    /* This rank's rows of each row part where they are not an even share (a replicated KV head),
     * on the weight and on its scale plane: every grid here keeps one scale row a weight row. */
    int  span_parts(RadBuilder* b, const int64_t* lo, const int64_t* hi, const int64_t* total,
                    int n) const {
        for (rad_weight x : { w, ws }) {
            if (!x) continue;
            RAD_ARCH_TRY(n == 1 ? rad_weight_shard_span(b, x, lo[0], hi[0])
                                : rad_weight_shard_span_parts(b, x, lo, hi, total, n));
        }
        return RAD_OK;
    }
};

/* A weight on the w4a8 / w8a8 grid: i4 or i8 codes with an f16 scale per [1 x 128], the encoding
 * gemm_nt_q reads against quant_act_i8's pair. */
static bool g4_grid_w4(const RadEncoding& enc) {
    const RadEncPlane* sc = rad_enc_plane(&enc, "scale");
    const uint32_t cdt = enc.n_planes > 0 ? enc.plane[0].dtype : 0;
    return rad_enc_is(&enc, "affine") && enc.n_planes == 2 && !enc.transform[0] &&
           (cdt == RAD_I4 || cdt == RAD_I8) && enc.plane[0].block[1] == 1 &&
           sc && sc->dtype == RAD_F16 && sc->block[0] == 1 && sc->block[1] == 128;
}

/* An empty `src` means the caller made the weight's name map (a probe asked it first). */
inline int G4Linear::declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                            std::initializer_list<const char*> src,
                            int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                            rad_buf a, rad_buf y,
                            std::initializer_list<int64_t> row_parts,
                            const G4Linear* in_from) {
    N = n_out; K = n_in; in = a;
    if (N <= 0 || K <= 0) return RAD_E_INVAL;
    const char* wn = nm.f("%s.weight", base);
    if (src.size() == 1)     RAD_ARCH_TRY(map_copy(b, wn, *src.begin()));
    else if (src.size() > 1) RAD_ARCH_TRY(map_concat(b, wn, src, 0));
    RadEncoding enc{};
    const bool known = weight_enc(b, wn, &enc);
    /* A builder with no model behind it (a test, a synthetic tool) answers RAD_E_NOTFOUND, and
     * the weight is read as plain bf16. */
    const bool plain = !known || (rad_enc_is(&enc, "plain") && enc.n_planes == 1 &&
                                  enc.plane[0].dtype == RAD_BF16);
    int64_t parts[RAD_MAX_ROW_PARTS] = {0};
    int np = 0;
    for (int64_t p : row_parts) if (np < RAD_MAX_ROW_PARTS) parts[np++] = p;

    if (plain) {
        w = decl_w(b, wn, RAD_BF16, { N, K }, RAD_ACCESS_PER_TOKEN, shard, grp, 0, row_parts);
        if (!w) return RAD_E_INVAL;
        op = rw(b, RAD_OP(b, "gemm_nt",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K), RAD_STR("dtype", g.dtype)),
                    RAD_WEIGHTS(w)),
                { a }, { y });
        return op ? RAD_OK : RAD_E_INVAL;
    }

    /* A w4 weight: i4 codes and an f16 scale per [1 x 128]. gemm_nt_q at w4a8 reads an int8
     * activation and an f32 per-row scale, so a quantiser over the bf16 plane `a` writes the int8
     * pair (q8, s8). The codes + scale layout is the same as w4a16's, so containers serve
     * unchanged. An 8-bit weight on the same grid (i8 codes, f16 scale per [1 x 128]): gemm_nt_q
     * at w8a8 reads the same int8 activation and per-row scale as w4a8, so the quantiser is
     * shared. A recipe uses it for linears int4 hurts (attention, down_proj). */
    const uint32_t cdt = known && enc.n_planes > 0 ? enc.plane[0].dtype : 0;
    i8g = known && linear_is_w8(enc);
    {
        const RadEncPlane* sp = rad_enc_plane(&enc, "scale");
        const RadEncPlane* zp = rad_enc_plane(&enc, "zero");
        a4 = known && rad_enc_is(&enc, "affine") && enc.n_planes == 3 && !enc.transform[0] &&
             (cdt == RAD_U4 || cdt == RAD_U5) && enc.plane[0].block[1] == 1 && sp &&
             sp->dtype == RAD_F16 && sp->block[0] == 1 && sp->block[1] == RAD_FP8_BLOCK && zp &&
             zp->dtype == RAD_U8 && zp->block[0] == 1 && zp->block[1] == RAD_FP8_BLOCK;
        w5 = a4 && cdt == RAD_U5;
    }
    if (i8g || a4) {
        if (N % 16 || K % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; it takes rows in 16s and K in "
                            "whole %d-column activation scale groups\n", base, (long long)N,
                    (long long)K, RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
        if (a4) {
            w  = decl_view(b, wn, nullptr, "codes", w5 ? RAD_U5 : RAD_U4, { N, K }, RAD_ACCESS_PER_TOKEN,
                           shard, grp, parts, np);
            ws = decl_view(b, nm.f("%s.scale", base), wn, "scale,zero", RAD_F16, { N, K },
                           RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
        } else {
            w  = decl_view(b, wn, nullptr, "codes", RAD_I8, { N, K }, RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
            ws = decl_view(b, nm.f("%s.scale", base), wn, "scale", RAD_BF16, { N, K / RAD_FP8_BLOCK },
                           RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
        }
        if (!w || !ws) return RAD_E_INVAL;
        /* not an a4 linear's pair: below its decode band it writes none */
        if (in_from && in_from->grouped() && !in_from->a4 && in_from->q8 && in_from->K == K) {
            q8 = in_from->q8;
            s8 = in_from->s8;
            shared_in = true;
        } else {
            q8 = decl_b(b, nm.f("%s.q8", base), RAD_I8,  { g.max_tok, K });
            s8 = decl_b(b, nm.f("%s.s8", base), RAD_F32, { g.max_tok, K / RAD_FP8_BLOCK });
            if (!q8 || !s8) return RAD_E_INVAL;
            op_q = rw(b, RAD_OP(b, "quant_act_i8g",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", K),
                                   RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
                    { a }, { q8, s8 });
            if (!op_q) return RAD_E_INVAL;
        }
        op = rw(b, RAD_OP(b, "gemm_nt_q",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K),
                               RAD_INT("group", RAD_FP8_BLOCK),
                               RAD_STR("dtype", w5 ? "w5a8i8" : a4 ? "w4a8i8_asym" : "i8a8")),
                    RAD_WEIGHTS(w, ws)),
                { q8, s8 }, { y });
        if (!op) return RAD_E_INVAL;
        if (a4) {
            a4_split = std::min<int64_t>(kA4Split, g.max_tok);
            a16 = decl_b(b, nm.f("%s.a16", base), RAD_F16, { a4_split, K });
            if (!a16) return RAD_E_INVAL;
            op_a16 = rw(b, RAD_OP(b, "cast",
                           RAD_PARAMS(RAD_RANGE("M", 1, a4_split), RAD_INT("n", K),
                                      RAD_STR("from", g.dtype), RAD_STR("to", "f16")),
                           RAD_NOWEIGHTS),
                        { a }, { a16 });
            op_dec = rw(b, RAD_OP(b, "gemm_nt_q",
                            RAD_PARAMS(RAD_RANGE("M", 1, a4_split), RAD_INT("N", N), RAD_INT("K", K),
                                       RAD_INT("group", RAD_FP8_BLOCK),
                                       RAD_STR("dtype", w5 ? "w5a16" : "w4a16_asym")),
                            RAD_WEIGHTS(w, ws)),
                        { a16 }, { y });
            if (!op_a16 || !op_dec) return RAD_E_INVAL;
        }
        return RAD_OK;
    }
    if (known && g4_grid_w4(enc)) {
        if (N % 16 || K % 128) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; a w4 weight takes rows in "
                            "16s and K in whole 128-column scale groups.\n", base,
                    (long long)N, (long long)K);
            return RAD_E_INVAL;
        }
        const char* sn = nm.f("%s.scale", base);
        w  = decl_view(b, wn, nullptr, "codes", cdt, { N, K }, RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
        ws = decl_view(b, sn, wn, "scale", RAD_F16, { N, K / 128 }, RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
        if (!w || !ws) return RAD_E_INVAL;
        if (in_from && in_from->quant && in_from->q8 && in_from->K == K) {
            q8 = in_from->q8;
            s8 = in_from->s8;
            shared_in = true;
        } else {
            q8 = decl_b(b, nm.f("%s.q8", base), RAD_I8,  { g.max_tok, K });
            s8 = decl_b(b, nm.f("%s.s8", base), RAD_F32, { g.max_tok, 1 });
        }
        if (!q8 || !s8) return RAD_E_INVAL;
        /* quant_act_i8, not quant_act_i8g: the w4a8 and w8a8 GEMMs read codes in the int8 WMMA's
         * step order with one f32 scale a row (docs/OPS.md). quant_act_i8g writes row-major codes
         * with a scale per (row, group) for i8a8; paired with these GEMMs it produces NaN logits.
         * `group` only sizes the optional code sums, which this symmetric grid does not take. */
        if (!shared_in)
            op_q = rw(b, RAD_OP(b, "quant_act_i8",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", K),
                                   RAD_INT("group", 128), RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
                    { a }, { q8, s8 });
        op = rw(b, RAD_OP(b, "gemm_nt_q",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K),
                               RAD_INT("group", 128), RAD_STR("dtype", cdt == RAD_I8 ? "w8a8" : "w4a8")),
                    RAD_WEIGHTS(w, ws)),
                { q8, s8 }, { y });
        quant = true;
        return ((op_q || shared_in) && op) ? RAD_OK : RAD_E_INVAL;
    }

    char en[256];
    rad_enc_format(&enc, en, sizeof en);
    fprintf(stderr, "radiance: '%s' is %s, and this model's linears are plain bf16, w4a8, w8a8, "
                    "int8 a 128 (i8a8) or asymmetric int4 or int5 a 128 -- none matches\n", wn, en);
    return RAD_E_DTYPE;
}

inline void G4Linear::step(RadCtx* c, rad_buf a, rad_buf y, int64_t T) const {
    if (quant) {
        /* Quantise the bf16 activation into the int8 pair, then run the GEMM. quant_act_i8 takes
         * x, q, scale, asum; `asum` (the per-group row sums asymmetric grids subtract) is absent on
         * this symmetric grid and passed as a null tensor, since operands are positional. */
        if (!shared_in)
            RAD_ISSUE_N(c, op_q, T, brows(a, T), brows(q8, T), brows(s8, T), RAD_NONE);
        gemm(c, y, T);
    } else if (a4 && T <= a4_split) {
        /* the decode band: f16 copy of the input, the skinny kernel; operands as LinearFP8's w4 */
        RAD_ISSUE_N(c, op_a16, T, brows(a, T), brows(a16, T));
        RAD_ISSUE_N(c, op_dec, T, brows(a16, T), RAD_NONE, RAD_W(w), RAD_W(ws), brows(y, T),
                    RAD_NONE, RAD_NONE);
    } else if (grouped()) {
        /* quant_act_i8g takes x, q, scale: no optional `asum`. */
        if (!shared_in)
            RAD_ISSUE_N(c, op_q, T, brows(a, T), brows(q8, T), brows(s8, T));
        gemm(c, y, T);
    } else
        RAD_ISSUE(c, op, brows(a, T), RAD_W(w), brows(y, T));
}

inline void G4Linear::gemm(RadCtx* c, rad_buf y, int64_t T) const {
    if (a4 && T <= a4_split) {
        /* the decode band reads no pair: the junction wrote the norm output into `in` instead */
        RAD_ISSUE_N(c, op_a16, T, brows(in, T), brows(a16, T));
        RAD_ISSUE_N(c, op_dec, T, brows(a16, T), RAD_NONE, RAD_W(w), RAD_W(ws), brows(y, T),
                    RAD_NONE, RAD_NONE);
        return;
    }
    RAD_ISSUE_N(c, op, T, brows(q8, T), brows(s8, T), RAD_W(w), RAD_W(ws),
                brows(y, T), RAD_NONE, RAD_NONE);
}

inline void G4Linear::step_rows(RadCtx* c, rad_buf a, rad_buf y, int64_t r0, int64_t n) const {
    if (a4 && n <= a4_split) {
        RAD_ISSUE_N(c, op_a16, n, brow_slice(a, r0, n, K), brows(a16, n));
        RAD_ISSUE_N(c, op_dec, n, brows(a16, n), RAD_NONE, RAD_W(w), RAD_W(ws),
                    brow_slice(y, r0, n, N), RAD_NONE, RAD_NONE);
        return;
    }
    if (!quant && !grouped()) {
        RAD_ISSUE_N(c, op, n, brow_slice(a, r0, n, K), RAD_W(w), brow_slice(y, r0, n, N));
        return;
    }
    if (!shared_in && grouped())
        RAD_ISSUE_N(c, op_q, n, brow_slice(a, r0, n, K), brow_slice(q8, r0, n, K),
                    brow_slice(s8, r0, n, scale_w()));
    else if (!shared_in)
        RAD_ISSUE_N(c, op_q, n, brow_slice(a, r0, n, K), brow_slice(q8, r0, n, K),
                    brow_slice(s8, r0, n, 1), RAD_NONE);
    gemm_rows(c, y, r0, n);
}

/* A slice of the prefill overlap only, which is above every decode band (kOverlapRows). */
inline void G4Linear::gemm_rows(RadCtx* c, rad_buf y, int64_t r0, int64_t n) const {
    RAD_ISSUE_N(c, op, n, brow_slice(q8, r0, n, K), brow_slice(s8, r0, n, scale_w()), RAD_W(w),
                RAD_W(ws), brow_slice(y, r0, n, N), RAD_NONE, RAD_NONE);
}

/* ---- one attention block, parameterised by class. */
struct Gemma4Attn {
    /* `qkv` is the fused projection's row, [max_tok, q | k | v], declared by the first layer of its
     * class that has one (the model's slot passed to declare). */
    struct Wire { rad_buf x = 0, h = 0, q = 0, k = 0, v = 0, attn = 0, qkv = 0; };

    Wire      w{};
    Geom      g{};
    ClassGeom cls{};
    int         layer = 0;
    rad_kvgroup kv    = 0;
    /* KV sharing (num_kv_shared_layers, Gemma 4 E2B/E4B): a shared layer has no k or v of its own
     * and attends over the cache of `kv_layer`, the last earlier layer of its class that keeps
     * one. A layer with its own cache has kv_layer == layer. */
    bool        shared   = false;
    int         kv_layer = 0;

    rad_weight w_norm = 0, w_qn = 0, w_kn = 0, w_post = 0;   /* the norms (always bf16) */
    G4Linear   ln_q{}, ln_k{}, ln_v{}, ln_o{};              /* the GEMM weights (bf16 or w4a8) */
    /* q, k and v as one linear, where the container holds them so (blk.N.attn_qkv): one GEMM a
     * layer instead of three. q, k and v are then column slices of `w.qkv`. */
    G4Linear   ln_qkv{};
    bool       fused = false;
    rad_op op_norm = 0, op_vn = 0, op_qn = 0, op_kn = 0,
           op_rq = 0, op_rk = 0, op_attn = 0, op_kv = 0, op_post = 0, op_ar = 0, op_add = 0;
    /* the five prologue ops (v norm, q norm, k norm, two rotaries) as one, where it resolves */
    rad_op op_qnr = 0;
    /* ...and that with kv_store folded in (qkv_norm_rope_kv_store), issued in place of op_qnr and
     * op_kv where it resolves; op_kv is then not declared */
    rad_op op_qnr_kv = 0;
    /* Fused only: attention reading q where it lies in the row (attn_paged_qrow), where it resolves
     * to the kernel attn_paged would take anyway -- at every height (op_attn_row), or at decode
     * heights up to dec_hi query rows a sequence (op_attn_dec); else the copy of q into its own rows
     * (`cast`) that a step of more than one row issues ahead of attn_paged. Without qkv_norm_rope
     * the copies of k and v too, ahead of the unfused prologue, which reads them dense. */
    rad_op op_attn_row = 0, op_attn_dec = 0, op_lift_q = 0, op_lift_k = 0, op_lift_v = 0;
    int64_t dec_hi = 0;
    /* Tree verification (rad_drafter_tree): attention that honours RadBatch::spec_parent over a
     * step's decode rows, reading q in the row (attn_paged_qrow_tree) or dense (attn_paged_tree,
     * after op_tree_lift copies it out). Declared only when `tree` is set before declare; 0 where
     * no library serves it, which the model reads as "no trees". */
    bool   tree = false;
    rad_op op_attn_tree = 0, op_tree_lift = 0;
    /* the feed-forward's gate_up read into the cache beside this block's all-reduce, or 0 (see
     * "Prefetch beside the reductions") */
    rad_op     op_pf = 0;
    rad_weight pf_w  = 0;
    /* the feed-forward's down_proj read into the cache beside the attention itself, or 0 */
    rad_op     op_pfd = 0;
    rad_weight pfd_w  = 0;

    /* The linear whose int8 pair a junction writes, and whether every projection reads it. */
    const G4Linear& ln_in() const { return fused ? ln_qkv : ln_q; }
    bool reads_pair() const {
        if (shared) return ln_q.pair();
        return fused ? ln_qkv.pair()
                     : ln_q.pair() && ln_k.shared_in && (!cls.has_v || ln_v.shared_in);
    }

    /* `kv_src` < 0: the layer keeps its own cache; else the layer whose cache it reads. */
    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const ClassGeom& cgeo,
                 rad_kvgroup group, const Wire& wire, const char* pre, rad_buf* qkv_slot,
                 int kv_src = -1);
    int  declare_shared(RadBuilder* b, Names& nm, const char* pre);
    /* `entry_done`: a junction op already wrote this block's quantised input (ln_q's pair).
     * `tail_fused`: the caller issues the post-norm and the add as a junction op. */
    void step(RadCtx* c, const RadBatch* batch, bool entry_done = false, bool tail_fused = false,
              bool ar_fused = false) const;
    /* step()'s parts, for the prefill overlap: the projections over rows [r0, r0 + n) from the
     * pair a junction wrote, everything between the projections and o_proj over the whole step
     * (it reads every row's keys), and o_proj over rows [r0, r0 + n). */
    void project(RadCtx* c, int64_t r0, int64_t n) const;
    void attend(RadCtx* c, const RadBatch* batch) const;
    void out_proj(RadCtx* c, int64_t r0, int64_t n) const;
    /* The prologue and the attention over rows [r0, r0 + n) of a one-sequence step alone (the
     * prefill overlap's slices), `used` the context the slice ends at (kv_len_trim for a slice
     * that ends before the step does). Only with the one-launch prologue and a prefill-height
     * attention that reads q dense or in the row (slicable). */
    bool slicable() const { return op_qnr_kv && (op_attn_row || op_attn); }
    void attend_rows(RadCtx* c, const RadBatch* batch, int64_t r0, int64_t n,
                     RadOperand used) const;
};

int Gemma4Attn::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const ClassGeom& cgeo,
                        rad_kvgroup group, const Wire& wire, const char* pre, rad_buf* qkv_slot,
                        int kv_src) {
    g = geom; cls = cgeo; layer = l; kv = group; w = wire;
    shared = kv_src >= 0;
    kv_layer = shared ? kv_src : l;
    if (shared) return declare_shared(b, nm, pre);
    const int64_t qd = cls.q_dim(), kvd = cls.kv_dim();
    const int64_t qkvd = qd + kvd + (cls.has_v ? kvd : 0);

    /* The norms: always bf16 (no rule matches them), name map first. */
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l),      nm.ckpt("%s.%lld.input_layernorm.weight",   pre, (long long)l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l),    nm.ckpt("%s.%lld.self_attn.q_norm.weight",   pre, (long long)l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", l),    nm.ckpt("%s.%lld.self_attn.k_norm.weight",   pre, (long long)l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_post_norm.weight", l), nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l)));
    /* Declared bf16, not f32: the checkpoint holds them bf16, and a widened weight cannot be
     * served from the mapped tier. Reading the gain as bf16 gives the same values. */
    w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_BF16, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_qn = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", l), RAD_BF16, {cls.head_dim},
                 RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_kn = decl_w(b, nm.f("blk.%d.attn_k_norm.weight", l), RAD_BF16, {cls.head_dim},
                 RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_post = decl_w(b, nm.f("blk.%d.attn_post_norm.weight", l), RAD_BF16, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* The GEMM weights: bf16 or w4a8, probed by G4Linear.
     *
     * q, k and v are one linear, `blk.N.attn_qkv` (q | k | v, or q | k on a k_eq_v class), where
     * the weight is on the w4a8 / w8a8 grid: a recipe that names it (rad-convert answers the
     * probe from the recipe) or a container converted with one. Each rank takes its slice of each
     * part, so the row holds this rank's q, k and v heads. The rows are q's, k's and v's rows, and
     * the grid's groups never cross a row, so the codes are the three linears' codes (libr4d's
     * skinny GEMM picks its K split from N, so on a card k and v may round differently in the last
     * bit). One GEMM a layer reads the three weights in one pass where three paid the skinny GEMM's
     * fixed cost each: on the R9700 at decode (--profile-ops, tp 4) q, k and v take 17 + 13.5 +
     * 13.5 us and a 4096-row GEMM of the same K 25 us. A container that holds the three apart, or
     * bf16 weights, keeps three linears reading q's int8 pair. */
    const char* q_src = nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l);
    const char* k_src = nm.ckpt("%s.%lld.self_attn.k_proj.weight", pre, (long long)l);
    const char* v_src = nm.ckpt("%s.%lld.self_attn.v_proj.weight", pre, (long long)l);
    const char* qkv_wn = nm.f("blk.%d.attn_qkv.weight", l);
    if (cls.has_v) RAD_ARCH_TRY(map_concat(b, qkv_wn, { q_src, k_src, v_src }));
    else           RAD_ARCH_TRY(map_concat(b, qkv_wn, { q_src, k_src }));
    {
        /* one q|k|v GEMM wherever the weight is on a quantised grid: w4a8 / w8a8, i8a8, a4 or
         * its five-bit form */
        RadEncoding e{};
        fused = weight_enc(b, qkv_wn, &e) && (g4_grid_w4(e) || linear_is_w8(e) ||
                                              (rad_enc_is(&e, "affine") && e.n_planes == 3 &&
                                               (e.plane[0].dtype == RAD_U4 ||
                                                e.plane[0].dtype == RAD_U5)));
    }
    if (fused) {
        if (!qkv_slot) return RAD_E_INVAL;
        if (!*qkv_slot)
            /* named by class, not by the value path: a global class with its own v (E2B/E4B,
             * attention_k_eq_v false) has one too, and the two rows differ in width */
            *qkv_slot = decl_b(b, nm.f(cls.head_ladder ? "attn_qkvg" : "attn_qkvs"), g.act_dtype,
                               {g.max_tok, qkvd});
        w.qkv = *qkv_slot;
        if (cls.has_v)
            RAD_ARCH_TRY(ln_qkv.declare(b, nm, g, nm.f("blk.%d.attn_qkv", l), {}, qkvd, g.n_embd,
                                        RAD_SHARD_ROW, grp_layer(l), w.h, w.qkv, {qd, kvd, kvd}));
        else
            RAD_ARCH_TRY(ln_qkv.declare(b, nm, g, nm.f("blk.%d.attn_qkv", l), {}, qkvd, g.n_embd,
                                        RAD_SHARD_ROW, grp_layer(l), w.h, w.qkv, {qd, kvd}));
    } else {
        RAD_ARCH_TRY(ln_q.declare(b, nm, g, nm.f("blk.%d.attn_q", l), { q_src },
                                  qd, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.q));
        const G4Linear* qin = &ln_q;   /* k and v read q's int8 pair */
        RAD_ARCH_TRY(ln_k.declare(b, nm, g, nm.f("blk.%d.attn_k", l), { k_src },
                                  kvd, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.k, {}, qin));
        if (cls.has_v)
            RAD_ARCH_TRY(ln_v.declare(b, nm, g, nm.f("blk.%d.attn_v", l), { v_src },
                                      kvd, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.v, {}, qin));
    }
    RAD_ARCH_TRY(ln_o.declare(b, nm, g, nm.f("blk.%d.attn_o", l),
                              { nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l) },
                              g.n_embd, qd, RAD_SHARD_COL, grp_layer(l), w.attn, w.h));

    /* A replicated KV head: this rank's k and v rows are its head's, not an even share. q stays
     * an even share; a rank's q heads all read that head. */
    if (cls.kv_rep) {
        const int64_t kv_all = cls.n_head_kv_all * cls.head_dim, q0 = (int64_t)g.rank * qd;
        if (fused) {
            const int64_t lo[3]  = { q0, cls.kv_off, cls.kv_off };
            const int64_t hi[3]  = { q0 + qd, cls.kv_off + kvd, cls.kv_off + kvd };
            const int64_t tot[3] = { qd * g.world, kv_all, kv_all };
            RAD_ARCH_TRY(ln_qkv.span_parts(b, lo, hi, tot, cls.has_v ? 3 : 2));
        } else {
            const int64_t lo = cls.kv_off, hi = cls.kv_off + kvd;
            RAD_ARCH_TRY(ln_k.span_parts(b, &lo, &hi, &kv_all, 1));
            if (cls.has_v) RAD_ARCH_TRY(ln_v.span_parts(b, &lo, &hi, &kv_all, 1));
        }
    }

    /* input norm: x_hat * w over the residual stream. */
    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                             RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});

    /* Gemma normalises the value with a weightless RMS norm (the checkpoint has no v_norm gain).
     * A sliding class normalises v in place; a k_eq_v class derives v from the raw k into its own
     * buffer, which must happen before k_norm rewrites k. */
    op_vn = rw(b, RAD_OP(b, "rmsnorm_p",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * cls.n_head_kv), RAD_INT("n", cls.head_dim),
                          RAD_F64("eps", g.eps), RAD_STR("dtype", g.dtype)),
               RAD_NOWEIGHTS),
              {cls.has_v ? w.v : w.k}, {w.v});

    op_qn = rw(b, RAD_OP(b, "rmsnorm",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * cls.n_head), RAD_INT("n", cls.head_dim),
                          RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
               RAD_WEIGHTS(w_qn)),
             {w.q}, {w.q});
    op_kn = rw(b, RAD_OP(b, "rmsnorm",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * cls.n_head_kv), RAD_INT("n", cls.head_dim),
                          RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
               RAD_WEIGHTS(w_kn)),
             {w.k}, {w.k});

    /* sliding: standard RoPE over the full width, theta_s. global: proportional over the partial
     * width (the "head" ladder), theta_g. q names its query heads; k its kv heads. */
    op_rq = rw(b, RAD_OP(b, "rope",
                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                            RAD_INT("n_head", cls.n_head), RAD_INT("n_head_kv", 0),
                            RAD_F64("theta", cls.theta), RAD_F64("scale", cls.scale),
                            RAD_STR("mode", "neox"), RAD_INT("rotary_dim", cls.n_rot),
                            RAD_STR("ladder", cls.head_ladder ? "head" : "rotary")),
                 RAD_NOWEIGHTS),
                {w.q}, {w.q});
    /* The rope kernel rotates n_head + n_head_kv heads of a fused QKV row and requires
     * n_head > 0. K is a separate buffer, so it is rotated as a Q-only form over the K heads:
     * n_head = n_head_kv and n_head_kv = 0, a row of n_head_kv*head_dim, exactly K's width. Q and K
     * share theta and positions, so this matches the reference's separate K rope. */
    op_rk = rw(b, RAD_OP(b, "rope",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                             RAD_INT("n_head", cls.n_head_kv), RAD_INT("n_head_kv", 0),
                             RAD_F64("theta", cls.theta), RAD_F64("scale", cls.scale),
                             RAD_STR("mode", "neox"), RAD_INT("rotary_dim", cls.n_rot),
                             RAD_STR("ladder", cls.head_ladder ? "head" : "rotary")),
                  RAD_NOWEIGHTS),
                 {w.k}, {w.k});

    /* Gemma 4 scores Q.K without the 1/sqrt(head_dim) factor (HF Gemma4TextAttention and vLLM
     * use scaling = 1.0; the q/k norm gains absorb it). attn_paged defaults to 1/sqrt(head_dim),
     * so 1.0 is passed explicitly. */
    auto attn = [&](const char* op, std::initializer_list<rad_buf> rd, int64_t q_hi = 0) {
        return rw(b, RAD_OP(b, op,
                     RAD_PARAMS(RAD_RANGE("q_len", 1, q_hi > 0 ? q_hi : g.max_tok),
                                RAD_INT("head_dim", cls.head_dim),
                                RAD_INT("gqa", cls.gqa()), RAD_INT("causal", 1),
                                RAD_INT("window", cls.window), RAD_INT("n_head", cls.n_head),
                                RAD_INT("block_size", RAD_KV_BLOCK),
                                RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                RAD_F64("scale", 1.0),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                  rd, {w.attn});
    };

    /* The prologue as one op: libr4d's `qkv_norm_rope` is rmsnorm_p + rmsnorm + rmsnorm + rope +
     * rope over the three projections, byte-identical, one workgroup per (token, head). Declared
     * beside the unfused five and used only where it resolves; a multi-component or YaRN rotary
     * keeps the five, as does an engine whose libraries do not declare it (CPU-only), where asking
     * for it would stop the declare. It reads q, k and v at their own row pitches, so on a fused
     * layer it works in place on the row's column slices. */
    /* What the prologue reads and writes, either form. A k_eq_v class derives v from the raw k:
     * v is written, never read. */
    auto qnr_rw = [&](rad_op op) {
        if (fused)
            return cls.has_v ? rw(b, op, {w.qkv}, {w.qkv}) : rw(b, op, {w.qkv}, {w.qkv, w.v});
        return cls.has_v ? rw(b, op, {w.q, w.k, w.v}, {w.q, w.k, w.v})
                         : rw(b, op, {w.q, w.k}, {w.q, w.k, w.v});
    };
    auto decl_qnr = [&]() {
        if (g.rope_mc || g.yarn.factor != 0.0f || !rad_op_declared(b, "qkv_norm_rope")) return;
        op_qnr = RAD_OP(b, "qkv_norm_rope",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                                 RAD_INT("n_head", cls.n_head), RAD_INT("n_head_kv", cls.n_head_kv),
                                 RAD_F64("theta", cls.theta), RAD_F64("eps", g.eps),
                                 RAD_F64("scale", cls.scale), RAD_STR("mode", "neox"),
                                 RAD_INT("rotary_dim", cls.n_rot),
                                 RAD_STR("ladder", cls.head_ladder ? "head" : "rotary"),
                                 RAD_F64("wadd", g.wadd_qk), RAD_INT("v_from_k", cls.has_v ? 0 : 1),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_qn, w_kn));
        op_qnr = qnr_rw(op_qnr);
    };

    /* A column slice of the fused row copied into its own rows (`cast`, bf16 to bf16, which every
     * library takes with pitched operands). */
    auto lift = [&](int64_t n, rad_buf dst) {
        return rw(b, RAD_OP(b, "cast",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", n),
                                RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                     RAD_NOWEIGHTS),
                  {w.qkv}, {dst});
    };

    if (!fused) {
        op_attn = attn("attn_paged", {w.q});
        decl_qnr();
    } else {
        /* The prologue first: what attention reads depends on it. Without it the unfused five read
         * q, k and v dense, so all three are copied out of the row ahead of them. */
        decl_qnr();
        if (!op_qnr) {
            op_lift_k = lift(kvd, w.k);
            if (cls.has_v) op_lift_v = lift(kvd, w.v);
        }
        /* With it attention reads q in the row where that costs nothing. libr4d's dedicated paged
         * kernels stop at head_dim 256; at 512 (the global class) attn_paged takes the windowed
         * kernel, which is the kernel attn_paged_qrow takes, so the row is read in place at every
         * height. At 256 the dedicated h256 kernels serve attn_paged in 0.26-0.71 of the windowed
         * kernel's time (libr4d/r4d_plugin.h, R4D_PRIO_ATTN_WINDOW). Their split-KV decode reads q
         * at a pitch under attn_paged_qrow too, over its band (q_len * gqa <= 64: 32 query rows a
         * sequence at the sliding class's gqa 2), so the sliding class reads the row there -- a
         * verify step included -- and keeps attn_paged above it, a prefill copying q out first. */
        const bool qrow = op_qnr && rad_op_declared(b, "attn_paged_qrow");
        if (qrow && cls.head_dim > 256) {
            op_attn_row = attn("attn_paged_qrow", {w.qkv});
        } else if (qrow) {
            dec_hi = std::min<int64_t>(g.max_tok, 64 / cls.gqa());
            op_attn_dec = attn("attn_paged_qrow", {w.qkv}, dec_hi);
            if (!op_attn_dec) dec_hi = 0;
        }
        if (!op_attn_row) {
            op_lift_q = lift(qd, w.q);
            op_attn = attn("attn_paged", {w.q, w.qkv});
        }
    }

    /* The tree form over the decode rows: a verify block of 1 + n_spec rows a sequence, so the
     * split-KV band (q_len * gqa <= 64) covers it at every class here. In the row where the
     * prologue leaves q there, else dense after a copy. */
    if (tree) {
        const int64_t thi = std::min<int64_t>(
            g.max_tok, std::max<int64_t>({ (int64_t)1, 64 / cls.gqa(), (int64_t)g.max_spec + 1 }));
        if (fused && op_qnr && rad_op_declared(b, "attn_paged_qrow_tree"))
            op_attn_tree = attn("attn_paged_qrow_tree", {w.qkv}, thi);
        if (!op_attn_tree && rad_op_declared(b, "attn_paged_tree")) {
            op_attn_tree = attn("attn_paged_tree", {w.q}, thi);
            if (op_attn_tree && fused && op_qnr) op_tree_lift = lift(qd, w.q);
        }
    }

    /* With the cache store folded into the prologue where a library serves it
     * (qkv_norm_rope_kv_store: the same prologue, then kv_store of the finished k and v from the
     * same launch, byte for byte), one launch a layer fewer. Declared after the attention ops,
     * since its block_size comes off the resolved attention kernel (rad_kv_block_size), and
     * issued in place of op_qnr and op_kv. */
    if (op_qnr && rad_op_declared(b, "qkv_norm_rope_kv_store")) {
        op_qnr_kv = RAD_OP(b, "qkv_norm_rope_kv_store",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                                 RAD_INT("n_head", cls.n_head), RAD_INT("n_head_kv", cls.n_head_kv),
                                 RAD_F64("theta", cls.theta), RAD_F64("eps", g.eps),
                                 RAD_F64("scale", cls.scale), RAD_STR("mode", "neox"),
                                 RAD_INT("rotary_dim", cls.n_rot),
                                 RAD_STR("ladder", cls.head_ladder ? "head" : "rotary"),
                                 RAD_F64("wadd", g.wadd_qk), RAD_INT("v_from_k", cls.has_v ? 0 : 1),
                                 RAD_STR("dtype", g.dtype),
                                 RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                 RAD_STR("kv_dtype", g.kv_dtype)),
                      RAD_WEIGHTS(w_qn, w_kn));
        if (op_qnr_kv) op_qnr_kv = qnr_rw(op_qnr_kv);
    }
    /* On a fused layer with the one-op prologue, k and v are stored from the row (kv_store takes
     * them at their own pitches); a k_eq_v class's v is its own buffer either way. Not declared
     * where the prologue stores them itself. */
    if (!op_qnr_kv) {
        op_kv = RAD_OP(b, "kv_store",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                               RAD_INT("n_head_kv", cls.n_head_kv),
                               RAD_INT("block_size", rad_kv_block_size(b, kv)),
                               RAD_STR("kv_dtype", g.kv_dtype)),
                    RAD_NOWEIGHTS);
        if (!(fused && op_qnr)) op_kv = rw(b, op_kv, {w.k, w.v}, {});
        else if (cls.has_v)     op_kv = rw(b, op_kv, {w.qkv}, {});
        else                    op_kv = rw(b, op_kv, {w.qkv, w.v}, {});
    }

    /* o_proj is column-sharded: the rank's partial sum is reduced first, then normed. */
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                   RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                              RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                              RAD_INT("min_bytes", g.wire_min_bytes)),
                   RAD_NOWEIGHTS),
                  {w.h}, {w.h});

    /* The second norm: the reduced attention output is normed before the residual add. */
    op_post = rw(b, RAD_OP(b, "rmsnorm",
                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                            RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                 RAD_WEIGHTS(w_post)),
                {w.h}, {w.h});

    op_add = rw(b, RAD_OP(b, "add",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
               RAD_NOWEIGHTS),
              {w.x, w.h}, {w.x});
    return RAD_OK;
}

/* A KV-shared layer (transformers' Gemma4TextAttention with is_kv_shared_layer): q_proj, the per-head
 * q norm and the class's rotary, attention over the source layer's cache, o_proj and the two norms.
 * No k or v projection, no k or v norm and no cache store: the source layer, earlier in the same
 * step, wrote the keys and values these queries read. The checkpoint's k_proj / v_proj / k_norm of
 * these layers (the "frozen shared kv" file) are unused, as in the reference. */
int Gemma4Attn::declare_shared(RadBuilder* b, Names& nm, const char* pre) {
    const int l = layer;
    const int64_t qd = cls.q_dim();
    fused = false;
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l),      nm.ckpt("%s.%lld.input_layernorm.weight",   pre, (long long)l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l),    nm.ckpt("%s.%lld.self_attn.q_norm.weight",   pre, (long long)l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_post_norm.weight", l), nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l)));
    w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_BF16, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_qn = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", l), RAD_BF16, {cls.head_dim},
                 RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_post = decl_w(b, nm.f("blk.%d.attn_post_norm.weight", l), RAD_BF16, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    RAD_ARCH_TRY(ln_q.declare(b, nm, g, nm.f("blk.%d.attn_q", l),
                              { nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l) },
                              qd, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.q));
    RAD_ARCH_TRY(ln_o.declare(b, nm, g, nm.f("blk.%d.attn_o", l),
                              { nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l) },
                              g.n_embd, qd, RAD_SHARD_COL, grp_layer(l), w.attn, w.h));
    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                             RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});
    op_qn = rw(b, RAD_OP(b, "rmsnorm",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * cls.n_head), RAD_INT("n", cls.head_dim),
                          RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
               RAD_WEIGHTS(w_qn)),
             {w.q}, {w.q});
    op_rq = rw(b, RAD_OP(b, "rope",
                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                            RAD_INT("n_head", cls.n_head), RAD_INT("n_head_kv", 0),
                            RAD_F64("theta", cls.theta), RAD_F64("scale", cls.scale),
                            RAD_STR("mode", "neox"), RAD_INT("rotary_dim", cls.n_rot),
                            RAD_STR("ladder", cls.head_ladder ? "head" : "rotary")),
                 RAD_NOWEIGHTS),
                {w.q}, {w.q});
    op_attn = rw(b, RAD_OP(b, "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok),
                                RAD_INT("head_dim", cls.head_dim),
                                RAD_INT("gqa", cls.gqa()), RAD_INT("causal", 1),
                                RAD_INT("window", cls.window), RAD_INT("n_head", cls.n_head),
                                RAD_INT("block_size", RAD_KV_BLOCK),
                                RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                RAD_F64("scale", 1.0),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                  {w.q}, {w.attn});
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                   RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                              RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                              RAD_INT("min_bytes", g.wire_min_bytes)),
                   RAD_NOWEIGHTS),
                  {w.h}, {w.h});
    op_post = rw(b, RAD_OP(b, "rmsnorm",
                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                            RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                 RAD_WEIGHTS(w_post)),
                {w.h}, {w.h});
    op_add = rw(b, RAD_OP(b, "add",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
               RAD_NOWEIGHTS),
              {w.x, w.h}, {w.x});
    return (op_norm && op_qn && op_rq && op_attn && op_post && op_add) ? RAD_OK : RAD_E_INVAL;
}

/* ---- Prefetch beside the reductions (prefetch_beside, rad_arch.h). The all-reduce in front of
 * each GEMM leaves the card's memory idle for its round trip (~20 us at a DFlash2 verify's rows on
 * 4x R9700), so the trunk reads the next GEMM's leading bytes in that window -- gate_up's after
 * o_proj, the next layer's q|k|v after down_proj -- and the GEMM finds them in the 64 MB on-die
 * cache. The attention leaves it mostly idle too at a verify's few rows and a chat's context, so
 * down_proj's weight is read beside it, after q|k|v. On four cards (a microbenchmark of the
 * two-shot and a GEMM-like stream of a cold 29 MB weight) 16 MB read beside the reduction took a
 * layer from 84.0 us to 70.1; read on the same lane it gained nothing. */

/* The height prefetch_beside judges a trunk step by: its rows on a decode step (decode and verify
 * rows only), and none at all on one that carries a prompt's rows (rad_arch.h, kPrefetchRows): the
 * row ceiling keeps long prefills out, and this keeps out the short ones it would let in. */
static int64_t prefetch_height(const RadBatch* batch) {
    return batch->phase == RAD_PHASE_DECODE ? batch->n_tok : INT64_MAX;
}

void Gemma4Attn::step(RadCtx* c, const RadBatch* batch, bool entry_done, bool tail_fused,
                      bool ar_fused) const {
    const int64_t T = batch->n_tok;

    if (shared) {
        if (entry_done) ln_q.gemm(c, w.q, T);
        else {
            RAD_ISSUE(c, op_norm, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));
            ln_q.step(c, w.h, w.q, T);
        }
    } else if (fused) {
        if (!entry_done) RAD_ISSUE(c, op_norm, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));
        if (entry_done) ln_qkv.gemm(c, w.qkv, T);
        else            ln_qkv.step(c, w.h, w.qkv, T);
    } else if (entry_done) {
        ln_q.gemm(c, w.q, T);
        ln_k.gemm(c, w.k, T);
        if (cls.has_v) ln_v.gemm(c, w.v, T);
    } else {
        RAD_ISSUE(c, op_norm, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));
        ln_q.step(c, w.h, w.q, T);
        ln_k.step(c, w.h, w.k, T);
        if (cls.has_v) ln_v.step(c, w.h, w.v, T);
    }
    prefetch_beside(c, op_pfd, pfd_w, prefetch_height(batch));
    attend(c, batch);

    ln_o.step(c, w.attn, w.h, T);
    prefetch_beside(c, op_pf, pf_w, prefetch_height(batch));
    /* The all-reduce comes before the post-norm: o_proj is column-sharded, so each rank holds a
     * partial sum, and the norm is of the whole output (sum_r rmsnorm(p_r) != rmsnorm(sum_r p_r)). */
    if (op_ar && !(tail_fused && ar_fused)) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    if (tail_fused) return;
    RAD_ISSUE_N(c, op_post, T, brows(w.h, T), RAD_W(w_post), brows(w.h, T));
    RAD_ISSUE(c, op_add,  brows(w.x, T), brows(w.h, T), brows(w.x, T));
}

void Gemma4Attn::project(RadCtx* c, int64_t r0, int64_t n) const {
    if (shared) {
        ln_q.gemm_rows(c, w.q, r0, n);
        return;
    }
    if (fused) {
        ln_qkv.gemm_rows(c, w.qkv, r0, n);
        return;
    }
    ln_q.gemm_rows(c, w.q, r0, n);
    ln_k.gemm_rows(c, w.k, r0, n);
    if (cls.has_v) ln_v.gemm_rows(c, w.v, r0, n);
}

void Gemma4Attn::out_proj(RadCtx* c, int64_t r0, int64_t n) const {
    ln_o.step_rows(c, w.attn, w.h, r0, n);
}

void Gemma4Attn::attend(RadCtx* c, const RadBatch* batch) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;
    const int64_t T = batch->n_tok;
    const int64_t qd = cls.q_dim(), kvd = cls.kv_dim();
    if (shared) {
        /* q only: its norm and rotary, then attention over the source layer's cache. */
        RAD_ISSUE_N(c, op_qn, T * cls.n_head, brows(w.q, T), RAD_W(w_qn), brows(w.q, T));
        RAD_ISSUE(c, op_rq, brows(w.q, T), rope_posmc(g, batch, T));
        RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                    brows(w.q, T), kv_cache(kv, kv_layer),
                    praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, batch->n_seq),
                    RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    brows(w.attn, T));
        return;
    }
    /* q, k and v where the prologue works on them: their own buffers, or on a fused layer with the
     * one-op prologue the row's column slices (a k_eq_v class's v is its own buffer either way). */
    const bool row = fused && op_qnr;
    const RadOperand Q = row ? bcol(w.qkv, 0, qd, T) : brows(w.q, T);
    const RadOperand K = row ? bcol(w.qkv, qd, kvd, T) : brows(w.k, T);
    const RadOperand V = row && cls.has_v ? bcol(w.qkv, qd + kvd, kvd, T) : brows(w.v, T);

    if (op_qnr_kv) {
        RAD_ISSUE(c, op_qnr_kv, Q, K, V, rope_posmc(g, batch, T), RAD_W(w_qn), RAD_W(w_kn),
                  praw(slot, RAD_I32, T), kv_cache(kv, layer));
    } else if (op_qnr) {
        RAD_ISSUE(c, op_qnr, Q, K, V, rope_posmc(g, batch, T), RAD_W(w_qn), RAD_W(w_kn));
    } else {
        if (fused) {   /* the unfused five read q, k and v dense */
            RAD_ISSUE_N(c, op_lift_q, T, bcol(w.qkv, 0, qd, T), brows(w.q, T));
            RAD_ISSUE_N(c, op_lift_k, T, bcol(w.qkv, qd, kvd, T), brows(w.k, T));
            if (cls.has_v) RAD_ISSUE_N(c, op_lift_v, T, bcol(w.qkv, qd + kvd, kvd, T), brows(w.v, T));
        }
        /* Normalise v first: on a k_eq_v class v derives from the raw k, before k_norm rewrites it. */
        RAD_ISSUE_N(c, op_vn, T * cls.n_head_kv, brows(cls.has_v ? w.v : w.k, T), brows(w.v, T));
        RAD_ISSUE_N(c, op_qn, T * cls.n_head,    brows(w.q, T), RAD_W(w_qn), brows(w.q, T));
        RAD_ISSUE_N(c, op_kn, T * cls.n_head_kv, brows(w.k, T), RAD_W(w_kn), brows(w.k, T));

        RAD_ISSUE(c, op_rq, brows(w.q, T), rope_posmc(g, batch, T));
        RAD_ISSUE(c, op_rk, brows(w.k, T), rope_posmc(g, batch, T));
    }

    if (op_kv) RAD_ISSUE(c, op_kv, K, V, praw(slot, RAD_I32, T), kv_cache(kv, layer));

    /* A tree step (rad_drafter_tree): every decode row's verify block sees, inside the block,
     * only that row's ancestors (RadBatch::spec_parent), so the decode rows go through the tree
     * form. On a step of decode rows alone that is the step's attention; beside a prefill chunk the
     * ordinary form runs over every row first and the tree form then rewrites the decode rows,
     * which come first (spec §8). */
    const bool tree_step = op_attn_tree && batch->spec_parent && batch->draft_pass == 0 &&
                           batch->n_seq_decode > 0;
    if (!tree_step || batch->n_seq_decode < batch->n_seq) {
        /* attn_paged reads q dense: one row of the row's slice is, a taller step copies q out. */
        rad_op a_op = op_attn;
        RadOperand a_q = brows(w.q, T);
        if (op_attn_row) {
            a_op = op_attn_row;
            a_q  = Q;
        } else if (op_attn_dec && batch->max_q_len <= dec_hi) {
            a_op = op_attn_dec;
            a_q  = Q;
        } else if (row) {
            if (T == 1) a_q = Q;
            else        RAD_ISSUE_N(c, op_lift_q, T, Q, brows(w.q, T));
        }
        RAD_ISSUE_N(c, a_op, batch->max_q_len,
                    a_q, kv_cache(kv, layer),
                    praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, batch->n_seq),
                    RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    brows(w.attn, T));
    }
    if (tree_step) {
        const int64_t Td = batch->n_tok_decode, Sd = batch->n_seq_decode;
        RadOperand tq = row ? bcol(w.qkv, 0, qd, Td) : brows(w.q, Td);
        if (op_tree_lift) {
            RAD_ISSUE_N(c, op_tree_lift, Td, bcol(w.qkv, 0, qd, Td), brows(w.q, Td));
            tq = brows(w.q, Td);
        }
        RAD_ISSUE_N(c, op_attn_tree, batch->max_q_len_decode,
                    tq, kv_cache(kv, layer),
                    praw2(table, RAD_I32, Sd, kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, Sd),
                    RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, Sd + 1),
                    brows(w.attn, Td),
                    praw(batch->spec_parent, RAD_I32, Td));
    }
}

/* A slice's rows see exactly the keys they see in the whole step: the cache holds the step's
 * earlier rows (the slices run in order, each storing its own keys first), `used` ends the context
 * at the slice's last row, the kernel counts each row's position back from there, and the key
 * tiles start at the sequence's first key in both cases. The prologue is a row at a time. */
void Gemma4Attn::attend_rows(RadCtx* c, const RadBatch* batch, int64_t r0, int64_t n,
                             RadOperand used) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int64_t qd = cls.q_dim(), kvd = cls.kv_dim();
    const int64_t qkvd = qd + kvd + (cls.has_v ? kvd : 0);
    const bool row = fused && op_qnr;
    const RadOperand Q = row ? bcol_at(w.qkv, r0, qkvd, 0, qd, n) : brow_slice(w.q, r0, n, qd);
    const RadOperand K = row ? bcol_at(w.qkv, r0, qkvd, qd, kvd, n) : brow_slice(w.k, r0, n, kvd);
    const RadOperand V = row && cls.has_v ? bcol_at(w.qkv, r0, qkvd, qd + kvd, kvd, n)
                                          : brow_slice(w.v, r0, n, kvd);
    RAD_ISSUE_N(c, op_qnr_kv, n, Q, K, V, praw(batch->positions + r0, RAD_I32, n), RAD_W(w_qn),
                RAD_W(w_kn), praw(slot ? slot + r0 : nullptr, RAD_I32, n), kv_cache(kv, layer));
    rad_op a_op = op_attn;
    RadOperand a_q = brow_slice(w.q, r0, n, qd);
    if (op_attn_row) {
        a_op = op_attn_row;
        a_q  = Q;
    } else if (row) {
        RAD_ISSUE_N(c, op_lift_q, n, Q, a_q);
    }
    RAD_ISSUE_N(c, a_op, n, a_q, kv_cache(kv, layer),
                praw2(table, RAD_I32, 1, kvb ? kvb->block_table_pitch : 0), used,
                RAD_NONE, RAD_NONE, RAD_NONE, brow_slice(w.attn, r0, n, qd));
}

/* ---- the feed-forward: GELU (not SwiGLU), with a second norm on the down-projection output. */
struct Gemma4Mlp {
    struct Wire { rad_buf x = 0, h = 0, gu = 0, ffn = 0; };

    Wire  w{};
    Geom  g{};              /* g.n_ff is this layer's width (double-wide layers are wider) */
    int   layer = 0;
    rad_weight w_norm = 0, w_post = 0;   /* the norms (always bf16) */
    G4Linear   ln_gu{}, ln_down{};      /* the GEMM weights (bf16 or w4a8) */
    rad_op op_norm = 0, op_act = 0, op_post = 0, op_ar = 0, op_add = 0;
    /* gelu_mul and down's quantiser as one, where it resolves (libr4d `gelu_mul_quant_i8`, or its
     * `_i8g` twin before a down_proj on quant_act_i8g's grid); not in an a4 down_proj's decode
     * band, which reads bf16 (gq_lo: the most rows that band takes, else 0) */
    rad_op  op_gq = 0;
    int64_t gq_lo = 0;
    /* the next layer's q|k|v read into the cache beside this block's all-reduce, or 0 (see
     * "Prefetch beside the reductions") */
    rad_op     op_pf = 0;
    rad_weight pf_w  = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire, const char* pre,
                 int64_t n_ff);
    void step(RadCtx* c, const RadBatch* batch, bool entry_done = false, bool tail_fused = false,
              bool ar_fused = false) const;
    /* For the prefill overlap, over rows [r0, r0 + n): the arm from the pair a junction wrote to
     * down_proj's partial in `h`, and the post-norm and add the last layer takes unfused. */
    void arm(RadCtx* c, int64_t r0, int64_t n) const;
    void tail(RadCtx* c, int64_t r0, int64_t n) const;
};

void Gemma4Mlp::arm(RadCtx* c, int64_t r0, int64_t n) const {
    const int64_t F = g.n_ff;
    ln_gu.gemm_rows(c, w.gu, r0, n);
    if (op_gq && n > gq_lo) {
        RAD_ISSUE_N(c, op_gq, n, brow_slice(w.gu, r0, n, 2 * F), brow_slice(ln_down.q8, r0, n, F),
                    brow_slice(ln_down.s8, r0, n, ln_down.scale_w()), RAD_NONE);
        ln_down.gemm_rows(c, w.h, r0, n);
    } else {
        RAD_ISSUE_N(c, op_act, n, brow_slice(w.gu, r0, n, 2 * F), brow_slice(w.ffn, r0, n, F));
        ln_down.step_rows(c, w.ffn, w.h, r0, n);
    }
}

void Gemma4Mlp::tail(RadCtx* c, int64_t r0, int64_t n) const {
    const int64_t E = g.n_embd;
    RAD_ISSUE_N(c, op_post, n, brow_slice(w.h, r0, n, E), RAD_W(w_post), brow_slice(w.h, r0, n, E));
    RAD_ISSUE_N(c, op_add, n, brow_slice(w.x, r0, n, E), brow_slice(w.h, r0, n, E),
                brow_slice(w.x, r0, n, E));
}

int Gemma4Mlp::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                       const char* pre, int64_t n_ff) {
    g = geom; layer = l; w = wire; g.n_ff = n_ff;

    /* The norms: always bf16 (no rule matches them), name map first. */
    RAD_ARCH_TRY(map_copy (b, nm.f("blk.%d.ffn_norm.weight", l), nm.ckpt("%s.%lld.pre_feedforward_layernorm.weight", pre, (long long)l)));
    RAD_ARCH_TRY(map_copy (b, nm.f("blk.%d.ffn_post_norm.weight", l), nm.ckpt("%s.%lld.post_feedforward_layernorm.weight", pre, (long long)l)));
    /* bf16, not f32, as for the attention norms: a widened weight cannot be served from the
     * mapped tier. */
    w_norm = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l), RAD_BF16, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_post = decl_w(b, nm.f("blk.%d.ffn_post_norm.weight", l), RAD_BF16, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* The GEMM weights: bf16 or w4a8, probed by G4Linear. gate_up is the gate|up fusion. */
    RAD_ARCH_TRY(ln_gu.declare(b, nm, g, nm.f("blk.%d.ffn_gate_up", l),
                               { nm.ckpt("%s.%lld.mlp.gate_proj.weight", pre, (long long)l),
                                 nm.ckpt("%s.%lld.mlp.up_proj.weight", pre, (long long)l) },
                               2 * g.n_ff, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.gu,
                               { g.n_ff, g.n_ff }));
    RAD_ARCH_TRY(ln_down.declare(b, nm, g, nm.f("blk.%d.ffn_down", l),
                                 { nm.ckpt("%s.%lld.mlp.down_proj.weight", pre, (long long)l) },
                                 g.n_embd, g.n_ff, RAD_SHARD_COL, grp_layer(l), w.ffn, w.h));

    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                           RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                RAD_WEIGHTS(w_norm)),
               {w.x}, {w.h});
    op_act = rw(b, RAD_OP(b, "gelu_mul",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff), RAD_STR("dtype", g.dtype)),
               RAD_NOWEIGHTS),
              {w.gu}, {w.ffn});
    const char* gq = ln_down.quant ? "gelu_mul_quant_i8"
                   : ln_down.grouped() ? "gelu_mul_quant_i8g" : nullptr;
    if (gq && !ln_down.shared_in && rad_op_declared(b, gq)) {
        op_gq = rw(b, RAD_OP(b, gq,
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff), RAD_STR("dtype", g.dtype)),
                     RAD_NOWEIGHTS),
                   {w.gu}, {ln_down.q8, ln_down.s8});
        gq_lo = ln_down.a4 ? ln_down.a4_split : 0;
    }
    /* down_proj is column-sharded: reduce first, then the post-norm. */
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                   RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                              RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                              RAD_INT("min_bytes", g.wire_min_bytes)),
                   RAD_NOWEIGHTS),
                  {w.h}, {w.h});
    op_post = rw(b, RAD_OP(b, "rmsnorm",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                          RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
               RAD_WEIGHTS(w_post)),
              {w.h}, {w.h});
    op_add = rw(b, RAD_OP(b, "add",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
               RAD_NOWEIGHTS),
              {w.x, w.h}, {w.x});
    return RAD_OK;
}

void Gemma4Mlp::step(RadCtx* c, const RadBatch* batch, bool entry_done, bool tail_fused,
                     bool ar_fused) const {
    const int64_t T = batch->n_tok;
    if (entry_done) {
        ln_gu.gemm(c, w.gu, T);
    } else {
        RAD_ISSUE(c, op_norm,  brows(w.x, T),   RAD_W(w_norm),  brows(w.h, T));
        ln_gu.step(c, w.h, w.gu, T);
    }
    if (op_gq && T > gq_lo) {
        RAD_ISSUE_N(c, op_gq, T, brows(w.gu, T), brows(ln_down.q8, T), brows(ln_down.s8, T), RAD_NONE);
        ln_down.gemm(c, w.h, T);
    } else {
        RAD_ISSUE(c, op_act,   brows(w.gu, T),  brows(w.ffn, T));
        ln_down.step(c, w.ffn, w.h, T);
    }
    prefetch_beside(c, op_pf, pf_w, prefetch_height(batch));
    /* The all-reduce before the post-norm, as in the attention block: down_proj is column-sharded. */
    if (op_ar && !(tail_fused && ar_fused)) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    if (tail_fused) return;
    RAD_ISSUE_N(c, op_post, T, brows(w.h, T), RAD_W(w_post), brows(w.h, T));
    RAD_ISSUE(c, op_add,   brows(w.x, T),   brows(w.h, T),  brows(w.x, T));
}

/* ---- one residual junction as one op: [post-norm of the branch `x`, the add into `res`, the
 * layer scalar,] the next block's norm and its int8 quantiser (libr4d `rmsnorm_quant_i8`, or
 * `rmsnorm_quant_i8g` before a linear on quant_act_i8g's grid). With `res` null it is the plain
 * norm of `x`, the first block's entry. Before an a4 linear the op also writes the bf16 norm output
 * into the linear's input row (`out_bf16`, `y`) in a step its decode band takes; that row is `x`
 * itself past the first block, as the unfused chain normalises into `h`. */
/* The four-rank fold's band. libr4d's four-rank row serves it in three forms, all byte-identical
 * to all_reduce + rmsnorm_quant_i8. Up to 4 rows it is the exact one-shot with the junction as its
 * epilogue: three times the message per rank and one fenced handshake a row, which wins at one row
 * (plain decode about 5% faster on 4x R9700) and loses to the standalone two-shot by 16. From 5
 * rows it is that two-shot with the junction as its tail (axis `tail`), behind a card-local count,
 * so the junction costs neither a handshake nor a launch. Above kAr4FoldRows a step issues the
 * pair. */
static constexpr int64_t kAr4FoldRows = 16;
/* From kArSpMinRows rows to kArSpMaxRows, libr4d's four-rank row takes `sp`: the all-reduce split
 * around the junction. Each rank reduces only the rows it owns (a quarter, in rank order), runs the
 * junction on them and gathers the int8 codes the next GEMM reads, so the second hop carries a byte
 * a column instead of two and the junction runs on a quarter of the rows. q and the scales are
 * the whole form's bytes; the residual stream then holds this rank's rows only, which is all the
 * next junction reads. A junction whose stream anything else reads next asks for every row (sp 2):
 * the layers the drafter taps, and the last layer's attention junction, since that layer's
 * feed-forward closes with its own all-reduce, post-norm and add over every row. Below 4 rows a
 * rank would idle (on 4x R9700, 31.7 against 18.7 us at one row); at 24-64 rows, where the pair
 * runs, the split form lost about 10% of aggregate throughput under load. */
static constexpr int64_t kArSpMinRows = 4;
static constexpr int64_t kArSpMaxRows = 16;

struct G4Junction {
    rad_op     op = 0;
    rad_op     op_ar = 0;   /* the same with the exact all-reduce of `x` in front, or 0 */
    int64_t    ar_rows = 0; /* the most rows op_ar serves: the step's at two ranks, a band at four */
    rad_op     op_sp = 0;   /* ...with the all-reduce split around it (four ranks), or 0 */
    int64_t    sp_hi = 0;   /* the most rows op_sp serves, from kArSpMinRows */
    rad_buf    x = 0, res = 0, q8 = 0, s8 = 0;
    rad_weight w = 0, wp = 0, sc = 0;
    int64_t    sw = 1;      /* the pair's scales a row (G4Linear::scale_w) */
    rad_buf    y = 0;       /* an a4 linear's input row, written in its decode band, or 0 */
    int64_t    y_hi = 0;    /* ...that band's most rows */

    /* `full_res`: something besides the next junction reads the residual stream after this one,
     * so the split form must leave it on every row. */
    int declare(RadBuilder* b, const Geom& g, rad_buf x_, rad_buf res_, rad_weight w_,
                rad_weight wp_, rad_weight sc_, const G4Linear& into, bool full_res = false) {
        x = x_; res = res_; w = w_; wp = wp_; sc = sc_; q8 = into.q8; s8 = into.s8;
        sw = into.scale_w();
        if (!into.pair() || into.shared_in || !q8 || !s8 || !w || (sc && !wp) || ((wp || sc) && !res))
            return RAD_E_INVAL;
        if (into.a4) {
            y = into.in;
            y_hi = into.a4_split;
            if (!y) return RAD_E_INVAL;
        }
        const bool grp = into.grouped();
        const char* rq  = grp ? "rmsnorm_quant_i8g" : "rmsnorm_quant_i8";
        const char* arq = grp ? "ar_rmsnorm_quant_i8g" : "ar_rmsnorm_quant_i8";
        const RadParam p[] = { RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_F64("eps", g.eps), RAD_STR("dtype", g.dtype),
                               RAD_F64("wadd", g.wadd) };
        const rad_weight ws[3] = { w, wp, sc };
        const int nw = sc ? 3 : (wp ? 2 : 1);
        /* Not declared by any loaded library (CPU-only) or not resolved: the caller keeps the
         * unfused chain. */
        if (rad_op_declared(b, rq))
            op = rad_decl_op(b, rq, p, (int)(sizeof p / sizeof p[0]), ws, nw);
        if (!op) return RAD_OK;
        const rad_buf rd[2] = { x, res };
        rad_buf wr[4] = { q8, s8, 0, 0 };
        int nwr = 2;
        if (res) wr[nwr++] = res;
        if (y)   wr[nwr++] = y;
        rad_op_reads (b, op, rd, res ? 2 : 1);
        rad_op_writes(b, op, wr, nwr);
        /* With the all-reduce in front, at two or four ranks on the exact wire: `x` is then a
         * rank's partial sum of a column-sharded linear. With the lossy wire allowed, the
         * standalone all-reduce is kept, since it may choose that wire. At four ranks the fold
         * serves steps of up to kAr4FoldRows rows; the i8a8 grid has no four-rank fold, so there the
         * standalone all-reduce stays. */
        const bool four = g.world == 4;
        const int64_t hi = four ? std::min<int64_t>(kAr4FoldRows, g.max_tok) : g.max_tok;
        if (res && g.wire_exact && (g.world == 2 || (four && !grp)) && hi >= 1 &&
            rad_op_declared(b, arq)) {
            const RadParam pa[] = { RAD_RANGE("M", 1, hi), RAD_INT("n", g.n_embd),
                                    RAD_F64("eps", g.eps), RAD_INT("world_size", g.world),
                                    RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd) };
            op_ar = rad_decl_op(b, arq, pa, (int)(sizeof pa / sizeof pa[0]), ws, nw);
            if (op_ar) {
                const rad_buf wa[5] = { x, q8, s8, res, y };
                rad_op_reads (b, op_ar, rd, 2);
                rad_op_writes(b, op_ar, wa, y ? 5 : 4);
                ar_rows = hi;
            }
            /* The split form from kArSpMinRows to kArSpMaxRows rows. It needs the post-norm, as
             * every trunk junction past the first has; `x` is declared written, as the fold's.
             * Not at quant_act_i8g's grid, whose fold has no split form. */
            const int64_t top = std::min<int64_t>(g.max_tok, kArSpMaxRows);
            if (op_ar && four && wp && !grp && top >= kArSpMinRows) {
                const RadParam ps[] = { RAD_RANGE("M", 1, top), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", g.eps), RAD_INT("world_size", g.world),
                                        RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd),
                                        RAD_INT("sp", full_res ? 2 : 1) };
                op_sp = rad_decl_op(b, "ar_rmsnorm_quant_i8", ps, (int)(sizeof ps / sizeof ps[0]),
                                    ws, nw);
                if (op_sp) {
                    const rad_buf wa[4] = { x, q8, s8, res };
                    rad_op_reads (b, op_sp, rd, 2);
                    rad_op_writes(b, op_sp, wa, 4);
                    sp_hi = top;
                }
            }
        }
        return RAD_OK;
    }
    /* The most rows a step may have for step(..., true): the fold's band, and the split form's. */
    int64_t ar_top() const { return op_sp ? std::max(ar_rows, sp_hi) : ar_rows; }
    void step(RadCtx* c, int64_t T, bool with_ar = false) const {
        const rad_op o = !with_ar ? op : (op_sp && T >= kArSpMinRows && T <= sp_hi) ? op_sp : op_ar;
        RAD_ISSUE_N(c, o, T, brows(x, T), res ? brows(res, T) : RAD_NONE, RAD_W(w),
                    brows(q8, T), brows(s8, T), y && T <= y_hi ? brows(y, T) : RAD_NONE,
                    wp ? RAD_W(wp) : RAD_NONE, sc ? RAD_W(sc) : RAD_NONE);
    }
    /* Rows [r0, r0 + n), without the all-reduce: a slice of the prefill overlap, whose reduction
     * rode the second lane. A slice is above every decode band, so no bf16 row. */
    void step_rows(RadCtx* c, int64_t E, int64_t r0, int64_t n) const {
        RAD_ISSUE_N(c, op, n, brow_slice(x, r0, n, E), res ? brow_slice(res, r0, n, E) : RAD_NONE,
                    RAD_W(w), brow_slice(q8, r0, n, E), brow_slice(s8, r0, n, sw), RAD_NONE,
                    wp ? RAD_W(wp) : RAD_NONE, sc ? RAD_W(sc) : RAD_NONE);
    }
};

/* ==================================================================== the assistant head
 *
 * Gemma 4's draft head, the "assistant" (Gemma4AssistantForCausalLM), merged under `draft.` by
 * `rad-convert --draft-model <gemma-4-*-it-assistant>`. From the checkpoint and three references:
 * transformers' modeling_gemma4_assistant.py with its candidate generator, vLLM's gemma4_mtp.py
 * with spec_decode/gemma4.py, and llama.cpp's gemma4-assistant.cpp:
 *
 *     x  = pre_projection( concat( E_trunk[tok] * sqrt(n_trunk), h_trunk ) )        2*n_trunk -> n
 *     x  = the head's layers over x (31B: sliding, sliding, sliding, full)
 *     hd = norm(x)
 *     logits = lm_head(hd)       the head's own table, tied to its embed_tokens; no soft-cap
 *     h' = post_projection(hd)                                                       n -> n_trunk
 *
 * `h_trunk` is the trunk's final-normed hidden state (the trunk lm_head's input) and `E_trunk` the
 * trunk's embedding table, scaled as the trunk scales it. The head's own embed_tokens serves only
 * as its lm_head (vLLM uses the trunk's table on the input side). The concatenation is embedding
 * first, then hidden state; all three references agree.
 *
 * A head layer is a trunk layer without key and value: q_proj, the per-head q norm, its class's
 * rotary, attention over the trunk's cache at scale 1, o_proj, the same four plain-gain norms, the
 * GELU feed-forward and the per-layer scalar. Each layer reads the last trunk layer of its class
 * (58 and 59 on the 31B; vLLM's _setup_gemma4_kv_sharing), so the head's q heads and the trunk's
 * kv heads must share geometry, which declare checks.
 *
 * The head keeps no cache of its own (rad_drafter_shared_kv): no history pass, one row per
 * drafting sequence, every round at the committed position over the committed context. Round 1
 * embeds the token the trunk chose and continues from the trunk's hidden state; round r embeds
 * round r-1's proposal and continues from its h'. Those two are the only path from one round to
 * the next, by design of the head.
 *
 * Tensor-parallel split as the trunk's: q heads row-sharded, so each rank's queries meet its share
 * of the trunk's kv heads; o_proj and the feed-forward down projection column-sharded, each
 * followed by an all-reduce before its post-norm. The two projections and every norm are
 * replicated, since they read and write the replicated stream. The lm_head is vocab-sharded like
 * the trunk's and goes through the same sampler chain, greedy. */
/* Where the head's rows sit for the rotary, relative to the committed position
 * (rad_drafter_shared_kv), and the depth `--num-speculative-tokens auto` takes. */
static constexpr int kHeadPosOffset = 0;
static constexpr int kHeadDepth     = 3;

struct Gemma4Head {
    struct HLayer {
        bool        global = false;
        int         trunk_layer = 0;   /* the trunk layer whose cache this layer reads */
        rad_kvgroup kv = 0;
        ClassGeom   cls{};
        rad_buf     q = 0, attn = 0;   /* the class's own rows: the two widths differ */
        rad_weight  w_norm = 0, w_qn = 0, w_post = 0, w_fnorm = 0, w_fpost = 0, w_scalar = 0;
        G4Linear    ln_q{}, ln_o{}, ln_gu{}, ln_down{};
        rad_op      op_norm = 0, op_qn = 0, op_rq = 0, op_attn = 0, op_ar = 0, op_post = 0,
                    op_add = 0, op_fnorm = 0, op_act = 0, op_far = 0, op_fpost = 0, op_fadd = 0,
                    op_scale = 0;
    };

    Geom    g{};            /* the trunk's, with max_tok cut to the head's rows */
    int64_t n = 0;          /* the head's width (1024 on the 31B) */
    int64_t n_trunk = 0;    /* the trunk's width, which the two projections join */
    int64_t n_ff = 0;       /* this rank's share of the feed-forward */
    int64_t rows = 0;       /* the most rows one pass carries: one per drafting sequence */
    double  softcap = 0.0;  /* 0: none, which is what the 31B's head says */
    std::vector<HLayer> layers;

    rad_buf src = 0;        /* the trunk's final-normed hidden state, [max_tok, n_trunk] */
    rad_buf logits = 0;     /* the trunk's logits plane: the sampler reads the head's from it */
    rad_buf b_c = 0, b_e = 0, b_x = 0, b_h = 0, b_gu = 0, b_ffn = 0, b_hd = 0, b_hn = 0;
    rad_weight w_tok = 0, w_norm = 0, w_lm = 0;
    G4Linear   ln_pre{}, ln_post{};
    rad_op op_hsrc = 0, op_hcopy = 0, op_embed = 0, op_embed_ar = 0, op_ecopy = 0, op_norm = 0,
           op_logits = 0, op_cap = 0;

    int  declare(RadBuilder* b, Names& nm, const RadModelMeta* meta, const Geom& trunk,
                 const ClassGeom& cs, const ClassGeom& cg, rad_kvgroup kv_s, rad_kvgroup kv_g,
                 int last_sliding, int last_full, int64_t n_main, rad_weight tok_embd,
                 rad_buf trunk_h, rad_buf trunk_logits);
    void step(RadCtx* c, const RadBatch* batch) const;
};

/* Does the container carry an assistant head at all? The 12B's is the same head under another
 * name (Gemma4UnifiedAssistantForCausalLM). */
static bool has_assistant(const RadModelMeta* m) {
    const char* a = rad_meta_gets(m, "draft.architectures", "");
    return a && (std::strstr(a, "Gemma4Assistant") || std::strstr(a, "Gemma4UnifiedAssistant"));
}

int Gemma4Head::declare(RadBuilder* b, Names& nm, const RadModelMeta* meta, const Geom& trunk,
                        const ClassGeom& cs, const ClassGeom& cg, rad_kvgroup kv_s,
                        rad_kvgroup kv_g, int last_sliding, int last_full, int64_t n_main,
                        rad_weight tok_embd, rad_buf trunk_h, rad_buf trunk_logits) {
    g = trunk;
    rows = trunk.max_seqs > 0 ? trunk.max_seqs : 1;
    g.max_tok = rows;
    src = trunk_h;
    logits = trunk_logits;
    w_tok = tok_embd;

    /* ---- the head's own geometry, and what it must share with the trunk's. */
    n       = meta_text_geti(meta, "hidden_size", 0, "draft.");
    n_trunk = rad_meta_geti(meta, "draft.backbone_hidden_size", 0);
    const long long n_l      = meta_text_geti(meta, "num_hidden_layers", 0, "draft.");
    const long long n_h      = meta_text_geti(meta, "num_attention_heads", 0, "draft.");
    const long long n_kv     = meta_text_geti(meta, "num_key_value_heads", 0, "draft.");
    const long long n_kv_g   = meta_text_geti(meta, "num_global_key_value_heads", n_kv, "draft.");
    const long long head_s   = meta_text_geti(meta, "head_dim", 0, "draft.");
    const long long head_g   = meta_text_geti(meta, "global_head_dim", head_s, "draft.");
    const long long ff       = meta_text_geti(meta, "intermediate_size", 0, "draft.");
    const long long win      = meta_text_geti(meta, "sliding_window", 0, "draft.");
    const long long vocab    = meta_text_geti(meta, "vocab_size", 0, "draft.");
    const long long ordered  = rad_meta_geti(meta, "draft.use_ordered_embeddings", 0);
    const double    eps      = meta_text_getf(meta, "rms_norm_eps", 0.0, "draft.");
    const double    theta_s  = meta_text_getf(meta, "rope_parameters.sliding_attention.rope_theta", 0.0, "draft.");
    const double    theta_g  = meta_text_getf(meta, "rope_parameters.full_attention.rope_theta", 0.0, "draft.");
    double          prf      = meta_text_getf(meta, "rope_parameters.full_attention.partial_rotary_factor", -1.0, "draft.");
    softcap = meta_text_getf(meta, "final_logit_softcapping", 0.0, "draft.");
    if (!(prf > 0.0)) prf = 1.0;

    if (n <= 0 || n_l <= 0 || n_h <= 0 || ff <= 0 || !(eps > 0.0)) {
        fprintf(stderr, "radiance: gemma4 assistant: the `draft.` config is incomplete (hidden %lld, "
                        "layers %lld, heads %lld, ffn %lld, eps %g)\n",
                (long long)n, n_l, n_h, ff, eps);
        return RAD_E_INVAL;
    }
    /* The head reads the trunk's cache, so everything the cache's shape depends on must agree:
     * the two head widths, the two kv head counts and the window. A head that disagrees was trained
     * with another trunk. The counts are the model's, not a rank's: a replicated kv head (fewer
     * than ranks) is one head on every rank. */
    if (n_trunk != trunk.n_embd || vocab != trunk.n_vocab_all || head_s != cs.head_dim ||
        head_g != cg.head_dim || n_kv != cs.n_head_kv_all || n_kv_g != cg.n_head_kv_all ||
        win != cs.window) {
        fprintf(stderr, "radiance: gemma4 assistant: this head was not made for this trunk -- it "
                        "states width %lld, vocabulary %lld, heads %lld/%lld x %lld/%lld kv over a "
                        "window of %lld; the trunk's cache is %lld/%lld kv heads of %lld/%lld over "
                        "%lld, at width %lld and vocabulary %lld\n",
                (long long)n_trunk, vocab, n_kv, n_kv_g, head_s, head_g, win,
                (long long)cs.n_head_kv_all, (long long)cg.n_head_kv_all,
                (long long)cs.head_dim, (long long)cg.head_dim, (long long)cs.window,
                (long long)trunk.n_embd, (long long)trunk.n_vocab_all);
        return RAD_E_INVAL;
    }
    /* The queries must rotate as the trunk's keys did, or q.k measures the wrong distance. */
    const int64_t rot_g = (int64_t)(head_g * prf + 0.5);
    if (theta_s != cs.theta || theta_g != cg.theta || rot_g != cg.n_rot) {
        fprintf(stderr, "radiance: gemma4 assistant: its rotary (theta %g / %g, %lld of %lld global "
                        "dims) is not the trunk's (%g / %g, %lld): its queries would meet the "
                        "trunk's keys at the wrong angle\n",
                theta_s, theta_g, (long long)rot_g, head_g, cs.theta, cg.theta,
                (long long)cg.n_rot);
        return RAD_E_INVAL;
    }
    /* Centroid masking (use_ordered_embeddings) scores a top-k of vocabulary clusters instead of
     * the whole head. The 31B's head does not use it and it is not implemented, so it is refused. */
    if (ordered) {
        fprintf(stderr, "radiance: gemma4 assistant: use_ordered_embeddings (centroid-masked logits) "
                        "is not implemented\n");
        return RAD_E_UNSUPPORTED;
    }
    int64_t nh = 0;
    if (divide_or_fail(n_h, trunk.world, "draft num_attention_heads", &nh) < 0) return RAD_E_INVAL;
    if (divide_or_fail(ff, trunk.world, "draft intermediate_size", &n_ff) < 0) return RAD_E_INVAL;
    if (nh % cs.n_head_kv || nh % cg.n_head_kv) {
        fprintf(stderr, "radiance: gemma4 assistant: %lld query heads a rank do not group over %lld "
                        "sliding / %lld global kv heads\n",
                (long long)nh, (long long)cs.n_head_kv, (long long)cg.n_head_kv);
        return RAD_E_INVAL;
    }

    /* The layer schedule: `layer_types` names every layer, as the trunk's does. */
    std::vector<int> is_global;
    {
        const char* lt = rad_meta_gets(meta, "draft.text_config.layer_types",
                                       rad_meta_gets(meta, "draft.layer_types", nullptr));
        if (!lt || !*lt || g4_parse_types(lt, n_l, is_global) < 0) {
            fprintf(stderr, "radiance: gemma4 assistant: the `draft.` config's layer_types does not "
                            "name its %lld layers\n", n_l);
            return RAD_E_INVAL;
        }
    }

    const int64_t base = n_main;   /* the head's weight groups, past the trunk's layers */
    g.eps = (float)eps;

    /* ---- the rows. The two classes keep their own query and attention rows, as in the trunk. */
    ClassGeom hs = cs; hs.n_head = nh;
    ClassGeom hgc = cg; hgc.n_head = nh;
    b_c   = decl_b(b, nm.f("mtp_c"),   g.act_dtype, {rows, 2 * n_trunk});
    b_e   = decl_b(b, nm.f("mtp_e"),   g.act_dtype, {rows, n_trunk});
    b_x   = decl_b(b, nm.f("mtp_x"),   g.act_dtype, {rows, n});
    b_h   = decl_b(b, nm.f("mtp_h"),   g.act_dtype, {rows, n});
    b_gu  = decl_b(b, nm.f("mtp_gu"),  g.act_dtype, {rows, 2 * n_ff});
    b_ffn = decl_b(b, nm.f("mtp_ffn"), g.act_dtype, {rows, n_ff});
    b_hd  = decl_b(b, nm.f("mtp_hd"),  g.act_dtype, {rows, n});
    b_hn  = decl_b(b, nm.f("mtp_hn"),  g.act_dtype, {rows, n_trunk});
    const rad_buf q_s = decl_b(b, nm.f("mtp_qs"), g.act_dtype, {rows, hs.q_dim()});
    const rad_buf a_s = decl_b(b, nm.f("mtp_as"), g.act_dtype, {rows, hs.q_dim()});
    const rad_buf q_g = decl_b(b, nm.f("mtp_qg"), g.act_dtype, {rows, hgc.q_dim()});
    const rad_buf a_g = decl_b(b, nm.f("mtp_ag"), g.act_dtype, {rows, hgc.q_dim()});

    /* ---- the entry: the trunk's hidden state into the right half of `c`, the trunk's scaled
     * embedding into the left. A history pass gathers the hidden state from the trunk's rows; a
     * round copies the head's own h' from the previous round. Both write the half in place, since
     * gather and cast take the destination's row pitch from the operand. */
    op_hsrc = rw(b, RAD_OP(b, "gather_rows",
                   RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n_trunk), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
                 {src}, {b_c});
    op_hcopy = rw(b, RAD_OP(b, "cast",
                    RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n_trunk),
                               RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                    RAD_NOWEIGHTS),
                  {b_hn}, {b_c});
    op_embed = rw(b, RAD_OP(b, "embed_lookup",
                    RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n_embd", n_trunk),
                               RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                               RAD_INT("vocab_offset", g.vocab_off),
                               RAD_F64("mult", gemma4::embed_scale(n_trunk))),
                    RAD_WEIGHTS(w_tok)),
                  {}, {b_e});
    if (g.world > 1)
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                             RAD_PARAMS(RAD_INT("world_size", g.world),
                                        RAD_RANGE("numel", n_trunk, rows * n_trunk),
                                        RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                             RAD_NOWEIGHTS),
                         {b_e}, {b_e});
    op_ecopy = rw(b, RAD_OP(b, "cast",
                    RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n_trunk),
                               RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                    RAD_NOWEIGHTS),
                  {b_e}, {b_c});
    if (!op_hsrc || !op_hcopy || !op_embed || !op_ecopy) return RAD_E_INVAL;

    /* pre_projection: the joined row to the head's width. Replicated, like the stream it reads
     * and writes. */
    RAD_ARCH_TRY(ln_pre.declare(b, nm, g, nm.f("mtp.pre_proj"), { nm.ckpt("draft.pre_projection.weight") },
                                n, 2 * n_trunk, RAD_SHARD_NONE, grp_layer((int)(base + n_l)),
                                b_c, b_x));

    /* The head's n-wide norm (plain gain), residual add and all-reduce. */
    auto norm = [&](rad_weight w, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w)),
                  {x}, {y});
    };
    auto add = [&]() {
        return rw(b, RAD_OP(b, "add",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                     RAD_NOWEIGHTS),
                  {b_x, b_h}, {b_x});
    };
    auto all_reduce = [&]() {
        return rw(b, RAD_OP(b, "all_reduce",
                     RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", n, rows * n),
                                RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                RAD_INT("min_bytes", g.wire_min_bytes)),
                     RAD_NOWEIGHTS),
                  {b_h}, {b_h});
    };

    /* ---- the layers. */
    layers.resize((size_t)n_l);
    for (int64_t l = 0; l < n_l; ++l) {
        HLayer& L = layers[(size_t)l];
        const int gl = (int)(base + l);
        L.global = is_global[(size_t)l] == 1;
        L.cls = L.global ? hgc : hs;
        L.trunk_layer = L.global ? last_full : last_sliding;
        L.kv = L.global ? kv_g : kv_s;
        L.q = L.global ? q_g : q_s;
        L.attn = L.global ? a_g : a_s;
        if (L.trunk_layer < 0) {
            fprintf(stderr, "radiance: gemma4 assistant: layer %lld reads a %s trunk layer and the "
                            "trunk has none\n", (long long)l, L.global ? "full" : "sliding");
            return RAD_E_INVAL;
        }
        const char* pre = "draft.model.layers";
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.attn_norm.weight", (long long)l),
                              nm.ckpt("%s.%lld.input_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.attn_q_norm.weight", (long long)l),
                              nm.ckpt("%s.%lld.self_attn.q_norm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.attn_post_norm.weight", (long long)l),
                              nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.ffn_norm.weight", (long long)l),
                              nm.ckpt("%s.%lld.pre_feedforward_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.ffn_post_norm.weight", (long long)l),
                              nm.ckpt("%s.%lld.post_feedforward_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.layer_scalar", (long long)l),
                              nm.ckpt("%s.%lld.layer_scalar", pre, (long long)l)));
        L.w_norm   = decl_w(b, nm.f("mtp.blk.%lld.attn_norm.weight", (long long)l), RAD_BF16, {n},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));
        L.w_qn     = decl_w(b, nm.f("mtp.blk.%lld.attn_q_norm.weight", (long long)l), RAD_BF16,
                            {L.cls.head_dim}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));
        L.w_post   = decl_w(b, nm.f("mtp.blk.%lld.attn_post_norm.weight", (long long)l), RAD_BF16, {n},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));
        L.w_fnorm  = decl_w(b, nm.f("mtp.blk.%lld.ffn_norm.weight", (long long)l), RAD_BF16, {n},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));
        L.w_fpost  = decl_w(b, nm.f("mtp.blk.%lld.ffn_post_norm.weight", (long long)l), RAD_BF16, {n},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));
        L.w_scalar = decl_w(b, nm.f("mtp.blk.%lld.layer_scalar", (long long)l), RAD_BF16, {1},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));

        RAD_ARCH_TRY(L.ln_q.declare(b, nm, g, nm.f("mtp.blk.%lld.attn_q", (long long)l),
                                    { nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l) },
                                    L.cls.q_dim(), n, RAD_SHARD_ROW, grp_layer(gl), b_h, L.q));
        RAD_ARCH_TRY(L.ln_o.declare(b, nm, g, nm.f("mtp.blk.%lld.attn_o", (long long)l),
                                    { nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l) },
                                    n, L.cls.q_dim(), RAD_SHARD_COL, grp_layer(gl), L.attn, b_h));
        RAD_ARCH_TRY(L.ln_gu.declare(b, nm, g, nm.f("mtp.blk.%lld.ffn_gate_up", (long long)l),
                                     { nm.ckpt("%s.%lld.mlp.gate_proj.weight", pre, (long long)l),
                                       nm.ckpt("%s.%lld.mlp.up_proj.weight",   pre, (long long)l) },
                                     2 * n_ff, n, RAD_SHARD_ROW, grp_layer(gl), b_h, b_gu,
                                     { n_ff, n_ff }));
        RAD_ARCH_TRY(L.ln_down.declare(b, nm, g, nm.f("mtp.blk.%lld.ffn_down", (long long)l),
                                       { nm.ckpt("%s.%lld.mlp.down_proj.weight", pre, (long long)l) },
                                       n, n_ff, RAD_SHARD_COL, grp_layer(gl), b_ffn, b_h));

        L.op_norm = norm(L.w_norm, b_x, b_h);
        L.op_qn = rw(b, RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, rows * L.cls.n_head), RAD_INT("n", L.cls.head_dim),
                                  RAD_F64("eps", eps), RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                       RAD_WEIGHTS(L.w_qn)),
                     {L.q}, {L.q});
        L.op_rq = rw(b, RAD_OP(b, "rope",
                       RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", L.cls.head_dim),
                                  RAD_INT("n_head", L.cls.n_head), RAD_INT("n_head_kv", 0),
                                  RAD_F64("theta", L.cls.theta), RAD_F64("scale", L.cls.scale),
                                  RAD_STR("mode", "neox"), RAD_INT("rotary_dim", L.cls.n_rot),
                                  RAD_STR("ladder", L.cls.head_ladder ? "head" : "rotary")),
                       RAD_NOWEIGHTS),
                     {L.q}, {L.q});
        /* One query row per sequence over the trunk's cache, at scale 1 like the trunk. The
         * context is the committed positions (rad_drafter_shared_kv), so only the decode band
         * exists. */
        L.op_attn = rw(b, RAD_OP(b, "attn_paged",
                         RAD_PARAMS(RAD_RANGE("q_len", 1, 1), RAD_INT("head_dim", L.cls.head_dim),
                                    RAD_INT("gqa", L.cls.gqa()), RAD_INT("causal", 1),
                                    RAD_INT("window", L.cls.window), RAD_INT("n_head", L.cls.n_head),
                                    RAD_INT("block_size", RAD_KV_BLOCK), RAD_INT("max_seqs", rows),
                                    RAD_INT("max_ctx", g.max_ctx), RAD_F64("scale", 1.0),
                                    RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                       {L.q}, {L.attn});
        if (g.world > 1) L.op_ar = all_reduce();
        L.op_post  = norm(L.w_post, b_h, b_h);
        L.op_add   = add();
        L.op_fnorm = norm(L.w_fnorm, b_x, b_h);
        L.op_act = rw(b, RAD_OP(b, "gelu_mul",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n_ff), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                      {b_gu}, {b_ffn});
        if (g.world > 1) L.op_far = all_reduce();
        L.op_fpost = norm(L.w_fpost, b_h, b_h);
        L.op_fadd  = add();
        L.op_scale = rw(b, RAD_OP(b, "scale",
                          RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                          RAD_WEIGHTS(L.w_scalar)),
                        {b_x}, {b_x});
        if (!L.op_norm || !L.op_qn || !L.op_rq || !L.op_attn || !L.op_post || !L.op_add ||
            !L.op_fnorm || !L.op_act || !L.op_fpost || !L.op_fadd || !L.op_scale ||
            (g.world > 1 && (!L.op_ar || !L.op_far)))
            return RAD_E_INVAL;
    }

    /* ---- the exit: the final norm, the head's own lm_head, and h' for the next round. */
    RAD_ARCH_TRY(map_copy(b, nm.f("mtp.output_norm.weight"), nm.ckpt("draft.model.norm.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("mtp.output.weight"), nm.ckpt("draft.model.embed_tokens.weight")));
    w_norm = decl_w(b, nm.f("mtp.output_norm.weight"), RAD_BF16, {n}, RAD_ACCESS_PER_TOKEN,
                    RAD_SHARD_NONE, grp_layer((int)(base + n_l)));
    /* Vocab-sharded like the trunk's: the draft's logits go through the same sampler chain, which
     * merges the ranks' candidates. Declared at its codes' dtype (bf16 as shipped, or int8 / block
     * fp8 where a recipe makes it so), selecting the matching logits_gemm kernel. It is a round's
     * largest read, and quantising it only affects proposals, which the trunk verifies. */
    w_lm = decl_w(b, nm.f("mtp.output.weight"), weight_codes_dtype(b, nm.f("mtp.output.weight"), RAD_BF16),
                  {g.n_vocab, n}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_layer((int)(base + n_l)));
    op_norm = norm(w_norm, b_x, b_hd);
    op_logits = rw(b, RAD_OP(b, "logits_gemm",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n_vocab", g.n_vocab),
                                RAD_INT("n_embd", n), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_lm)),
                   {b_hd}, {logits});
    if (softcap > 0.0)
        op_cap = rw(b, RAD_OP(b, "logit_softcap",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_vocab),
                                 RAD_F64("cap", softcap), RAD_STR("dtype", "f32")),
                      RAD_NOWEIGHTS),
                    {logits}, {logits});
    if (!op_norm || !op_logits || (softcap > 0.0 && !op_cap)) return RAD_E_INVAL;
    /* post_projection last: h' is the next round's input, read by the head's first op
     * (op_hcopy). Declared here, its buffer lives across the whole head, so nothing in a pass can
     * overwrite it before the next round reads it. */
    RAD_ARCH_TRY(ln_post.declare(b, nm, g, nm.f("mtp.post_proj"), { nm.ckpt("draft.post_projection.weight") },
                                 n_trunk, n, RAD_SHARD_NONE, grp_layer((int)(base + n_l)),
                                 b_hd, b_hn));

    rad_note(b, "the assistant head is declared: %lld layer(s) of width %lld at absolute %lld..%lld, "
                "reading the trunk's cache at layer %d (sliding) and %d (full); no cache of its own, "
                "every round at the committed position",
             (long long)n_l, (long long)n, (long long)base, (long long)(base + n_l - 1),
             last_sliding, last_full);
    return RAD_OK;
}

/* One pass, over the batch staged for a head with no cache of its own: one row per drafting
 * sequence, at the committed position, with the context ending there. */
void Gemma4Head::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    if (batch->draft_pass < 0) {
        /* Every row must be a drafting row, as rad_drafter_shared_kv requested; anything else
         * was built for a head with its own cache. */
        if (batch->n_draft_out != T || batch->n_out != T) {
            rad_step_fail(c, "the gemma4 assistant head was handed a history pass that is not one row "
                             "a drafting sequence; it keeps no cache of its own and drafts from that "
                             "shape only (rad_drafter_shared_kv)");
            return;
        }
        RAD_ISSUE_N(c, op_hsrc, T, RAD_B(src), praw(batch->out_ids, RAD_I32, T),
                    bcol(b_c, n_trunk, n_trunk, T));
    } else {
        RAD_ISSUE_N(c, op_hcopy, T, brows(b_hn, T), bcol(b_c, n_trunk, n_trunk, T));
    }
    RAD_ISSUE(c, op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(w_tok), brows(b_e, T));
    if (op_embed_ar) RAD_ISSUE_N(c, op_embed_ar, T * n_trunk, brows(b_e, T), RAD_NONE);
    RAD_ISSUE_N(c, op_ecopy, T, brows(b_e, T), bcol(b_c, 0, n_trunk, T));
    ln_pre.step(c, b_c, b_x, T);

    for (const HLayer& L : layers) {
        const RadKVGroupBatch* kvb = kv_batch(batch, L.kv);
        RAD_ISSUE(c, L.op_norm, brows(b_x, T), RAD_W(L.w_norm), brows(b_h, T));
        L.ln_q.step(c, b_h, L.q, T);
        RAD_ISSUE_N(c, L.op_qn, T * L.cls.n_head, brows(L.q, T), RAD_W(L.w_qn), brows(L.q, T));
        RAD_ISSUE(c, L.op_rq, brows(L.q, T), rope_posmc(g, batch, T));
        RAD_ISSUE_N(c, L.op_attn, batch->max_q_len,
                    brows(L.q, T), kv_cache(L.kv, L.trunk_layer),
                    praw2(kvb ? kvb->block_table : nullptr, RAD_I32, batch->n_seq,
                          kvb ? kvb->block_table_pitch : 0),
                    praw(kvb ? kvb->seqused : nullptr, RAD_I32, batch->n_seq),
                    RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    brows(L.attn, T));
        L.ln_o.step(c, L.attn, b_h, T);
        if (L.op_ar) RAD_ISSUE_N(c, L.op_ar, T * n, brows(b_h, T), RAD_NONE);
        RAD_ISSUE_N(c, L.op_post, T, brows(b_h, T), RAD_W(L.w_post), brows(b_h, T));
        RAD_ISSUE(c, L.op_add, brows(b_x, T), brows(b_h, T), brows(b_x, T));

        RAD_ISSUE(c, L.op_fnorm, brows(b_x, T), RAD_W(L.w_fnorm), brows(b_h, T));
        L.ln_gu.step(c, b_h, b_gu, T);
        RAD_ISSUE(c, L.op_act, brows(b_gu, T), brows(b_ffn, T));
        L.ln_down.step(c, b_ffn, b_h, T);
        if (L.op_far) RAD_ISSUE_N(c, L.op_far, T * n, brows(b_h, T), RAD_NONE);
        RAD_ISSUE_N(c, L.op_fpost, T, brows(b_h, T), RAD_W(L.w_fpost), brows(b_h, T));
        RAD_ISSUE(c, L.op_fadd, brows(b_x, T), brows(b_h, T), brows(b_x, T));
        RAD_ISSUE(c, L.op_scale, brows(b_x, T), RAD_W(L.w_scalar), brows(b_x, T));
    }

    RAD_ISSUE(c, op_norm, brows(b_x, T), RAD_W(w_norm), brows(b_hd, T));
    RAD_ISSUE_N(c, op_logits, T, brows(b_hd, T), RAD_W(w_lm), brows(logits, T));
    if (op_cap) RAD_ISSUE_N(c, op_cap, T, brows(logits, T), brows(logits, T));
    ln_post.step(c, b_hd, b_hn, T);
}

/* The prefill overlap's smallest slice (trunk_overlap). It is above every row band the sliced
 * kernels have, with room to spare: the int8 GEMMs' skinny band (64) and r4d_act_threads' last
 * width change at these widths (120 rows), which fixes the norms' and the quantisers'
 * sum-of-squares partition. And it is large enough that a slice's reduction is worth a second
 * lane. */
static constexpr int64_t kOverlapRows = 512;
/* ...and above an a4 linear's decode band, so a slice always reads the int8 pair a junction's
 * slice writes (G4Linear::gemm_rows, G4Junction::step_rows). */
static_assert(kOverlapRows > G4Linear::kA4Split, "a prefill slice inside an a4 decode band");

struct Layer {
    bool       global = false;
    Gemma4Attn attn{};
    Gemma4Mlp  mlp{};
    rad_weight w_scalar = 0;   /* the per-layer [1] residual scalar, `cur = cur * scalar` */
    rad_op     op_scale = 0;
    /* Per-layer embeddings (Gemma 4 E2B/E4B), after the feed-forward's residual add and before the
     * scalar: x += norm(proj(gelu(gate(x)) * pli[l])). `ple_post` is the norm's gain. */
    G4Linear   ln_pg{}, ln_pp{};
    rad_weight w_ple_post = 0;
    rad_op     op_ple_act = 0, op_ple_post = 0, op_ple_add = 0;
    /* The fused junctions (see "Residual junctions" above): attention -> this layer's
     * feed-forward, and this layer's feed-forward -> the next layer's attention, scalar included.
     * The last layer has no j_ma: its stream goes to the final norm, which feeds the lm_head, not
     * a quantiser. */
    G4Junction j_am{}, j_ma{};
    /* the DFlash drafter's tap: this layer's output into its column slice of `dflash_src` */
    rad_op     op_tap = 0;
    /* The prefill overlap's reductions (trunk_overlap): attn.op_ar and mlp.op_ar at `overlap` 1,
     * the narrow launch that leaves the CUs to the other slice's compute; 0 where not declared. */
    rad_op     op_ar_att_ov = 0, op_ar_ff_ov = 0;
};

/* The prefill overlap's attention per slice (trunk_overlap): slice A's context end, one
 * kv_len_trim a KV group the attention reads. */
struct SliceTrim {
    rad_kvgroup kv = 0;
    rad_op      op = 0;
    rad_buf     used = 0;   /* [max_seqs] int32 */
};

struct Model {
    Names      nm{""};
    Geom       g{};
    ClassGeom  cs{};   /* sliding class */
    ClassGeom  cg{};   /* global class  */
    int64_t    n_main = 0;

    rad_kvgroup kv_s = 0, kv_g = 0;

    rad_buf b_x = 0, b_h = 0, b_hout = 0, b_logits = 0;
    rad_buf b_qs = 0, b_ks = 0, b_vs = 0, b_attns = 0;   /* sliding rows */
    rad_buf b_qg = 0, b_kg = 0, b_vg = 0, b_atting = 0;  /* global rows  */
    rad_buf b_qkvs = 0, b_qkvg = 0;   /* the fused projections' rows, where a layer has one */
    rad_buf b_gu = 0, b_ffn = 0;
    /* ...and the double-wide layers' own pair, since a GEMM reads its activation packed (a narrow
     * layer in the wide buffer would read a padded row). Declared only for use_double_wide_mlp. */
    rad_buf b_gu_w = 0, b_ffn_w = 0;

    rad_weight w_tok = 0, w_out = 0, w_lm = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_final = 0, op_gather = 0, op_logits = 0, op_cap = 0;

    /* Per-layer embeddings (hidden_size_per_layer_input > 0): the token-identity table
     * (embed_tokens_per_layer, n_layer x n_ple wide), the context projection of the scaled
     * embedding and its per-(token, layer) norm, combined into `b_ple` [max_tok, n_layer * n_ple]
     * that every layer reads its slice of. At two ranks the table is vocab-sharded like the token
     * table, and its lookup is reduced the same way (op_ple_ar). The rest of the branch is
     * replicated, so no layer reduces it. */
    int64_t    n_ple = 0;
    rad_weight w_ple_tok = 0, w_ple_norm = 0;
    G4Linear   ln_ple_proj{};
    rad_buf    b_ple = 0, b_ple_tok = 0, b_pg = 0;
    rad_op     op_ple_tok = 0, op_ple_ar = 0, op_ple_norm = 0, op_ple_add = 0;
    /* An image's placeholder positions take the token-identity row of pad_token_id, as the
     * reference looks them up (Gemma4Model.forward: llm_input_ids = pad where multimodal): that row
     * times sqrt(n_ple), made as fill(sqrt(n_ple)) * the row, scattered over the image rows. */
    rad_weight w_ple_pad = 0;
    rad_buf    b_ple_pad = 0;
    rad_op     op_ple_pfill = 0, op_ple_pmul = 0, op_ple_pscat = 0;

    std::vector<Layer> layers;

    /* The assistant head, declared only when the container carries one and the deployment
     * speculates (`max_spec` > 0). */
    Gemma4Head head{};
    bool       have_head = false;

    /* The vision tower and the audio encoder, each declared when the container carries it and
     * the deployment serves that medium; `op_mm` writes their rows over the placeholders'
     * embeddings. */
    Gemma4Vision vis{};
    Gemma4Audio  aud{};
    Gemma4UnifiedVision uvis{};   /* the 12B's embedders, in place of the tower and encoder */
    Gemma4UnifiedAudio  uaud{};
    rad_op       op_mm = 0;

    G4Junction j0{};       /* the first block's entry: norm + quantiser of the embedding stream */
    bool       fuse = false;
    bool       fuse_ar = false;   /* every junction also carries the all-reduce in front of it */
    int64_t    ar_rows = 0;       /* ...in a step of at most this many rows (G4Junction::ar_rows) */
    /* A prefill step of 2 * kOverlapRows rows or more runs as two slices, each slice's
     * all-reduces on the second lane beside the other's compute (trunk_overlap). Set at declare. */
    bool       overlap = false;

    /* z-lab's DFlash drafter (arch/common/rad_gemma4_dflash.h), declared only when the container
     * carries one and the deployment speculates: the trunk layers it taps, the [max_tok, taps x
     * n_embd] plane the taps fill, the block and its own KV group. */
    enum { MAX_TAPS = gemma4::Dflash1::MAX_TAPS };
    int             n_taps = 0;
    int             tap_layer[MAX_TAPS] = {};
    rad_buf         b_dfsrc = 0;
    gemma4::Dflash1 df{};
    rad_kvgroup     kv_draft = 0;
    bool            have_df = false;
    /* Or a DFlash2 drafter (arch/common/rad_block_dflash2.h): RedHatAI's speculators-format one
     * for the 31B. Same taps and plane; its own block, buffers and KV group. */
    Dflash2         df2{};
    Dflash2::Wire   df2w{};
    bool            have_df2 = false;
    /* Tree verification (RadBuildCtx::draft_tree): every layer's attention has its tree form, and
     * the drafter proposes trees once it has declared dflash_tree too. */
    bool            tree = false;
    /* The prefill overlap runs attention per slice: a trim a KV group (sliding, global). */
    SliceTrim       trim[2]{};
    int             n_trim = 0;
    bool            slice_attn = false;
};

static Model g_model[MAX_RANKS];

/* A DFlash2 drafter's config key, from vLLM's speculators layout (the speculator's keys at the
 * top of `draft.`) or z-lab's (under `draft.dflash_config.`), whichever the checkpoint has. */
static long long df2_geti(const RadModelMeta* meta, const char* key, long long dflt) {
    char k[160];
    std::snprintf(k, sizeof k, "draft.%s", key);
    const long long v = rad_meta_geti(meta, k, LLONG_MIN);
    if (v != LLONG_MIN) return v;
    std::snprintf(k, sizeof k, "draft.dflash_config.%s", key);
    return rad_meta_geti(meta, k, dflt);
}

/* The DFlash2 drafter: RedHatAI/gemma-4-31B-it-speculator.dflash2, a speculators-format DFlash2
 * (z-lab's architecture, trained by Red Hat on gemma-4-31B-it's outputs) in place of z-lab's
 * DFlash. rad_block_dflash2.h is the block; this reads its config and wires it to the trunk.
 *
 * From the checkpoint and speculators' code (models/dflash2, models/dflash/core.py):
 *
 *   - five Qwen3-shaped layers at the trunk's width, every one sliding (2048) and non-causal;
 *   - the target's table as a plain nn.Embedding: no Gemma sqrt(hidden) scale (z-lab's DFlash
 *     goes through the target's scaled module; this one does not), so the block's unscaled
 *     lookup is right;
 *   - the target's head as a plain linear, no soft-cap, which the block's head is;
 *   - taps counted as vLLM captures them, the residual entering a layer (gemma4::dflash2_taps);
 *   - block 8: the anchor predicts nothing, seven drafts (sample_from_anchor false).
 *
 * The head reads the 8-bit copy the -df2 recipe makes (rad_block_dflash2.h, head_i8), so the
 * selector ranks what an 8-bit head proposes. Nothing here is defaulted quietly: a config this
 * block does not implement is refused by name. */
static int declare_dflash2(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                           Model& m) {
    Names& nm = m.nm;
    Geom   dg{};
    RAD_ARCH_TRY(drafter_geom(dg, m.g, meta, "gemma4 DFlash2"));
    const char* lp = drafter_layer_prefix(meta);
    char key[160];
    std::snprintf(key, sizeof key, "%shidden_size", lp);
    if (rad_meta_geti(meta, key, dg.n_embd) != dg.n_embd) {
        fprintf(stderr, "radiance: gemma4: the DFlash2 drafter is %lld wide and the trunk %lld; its "
                        "fc and residual stream are the trunk's width\n",
                rad_meta_geti(meta, key, 0), (long long)dg.n_embd);
        return RAD_E_UNSUPPORTED;
    }
    std::snprintf(key, sizeof key, "%shidden_act", lp);
    const char* act = rad_meta_gets(meta, key, "silu");
    if (std::strcmp(act, "silu")) {
        fprintf(stderr, "radiance: gemma4: the DFlash2 drafter's MLP uses %s; the block's is silu\n",
                act);
        return RAD_E_UNSUPPORTED;
    }
    if (df2_geti(meta, "sample_from_anchor", 0)) {
        fprintf(stderr, "radiance: gemma4: this DFlash2 drafter samples from its anchor row; the "
                        "block's anchor predicts nothing\n");
        return RAD_E_UNSUPPORTED;
    }

    Dflash2::Config dc{};
    dc.n_taps   = m.n_taps;
    dc.block    = ctx->max_spec + 1;
    /* A tree (m.tree, asked before the trunk) is `max_spec` nodes grown over the lattice of the
     * block the drafter was trained at, whatever its width: the block is the trained one and
     * --num-speculative-tokens counts nodes. */
    if (m.tree && gemma4::dflash_block(meta) > 1) {
        dc.block      = gemma4::dflash_block(meta);
        dc.tree_nodes = ctx->max_spec;
    }
    dc.taps     = df2_geti(meta, "conv_kernel_size", 2);
    dc.group    = df2_geti(meta, "conv_group_size", 16);
    dc.sel_rank = df2_geti(meta, "selector_rank", 0);
    dc.sel_topk = df2_geti(meta, "selector_top_k", 0);
    dc.prefetch = ctx->weight_prefetch_off ? 0 : gemma4::kPrefetchBytes;
    if (dc.sel_rank <= 0 || dc.sel_topk <= 0) {
        fprintf(stderr, "radiance: gemma4: the DFlash2 drafter states no selector geometry (rank "
                        "%lld, top_k %lld)\n", (long long)dc.sel_rank, (long long)dc.sel_topk);
        return RAD_E_INVAL;
    }
    /* Attention: one window for every layer, since the block issues one attention op. A sliding
     * window non-causal within the block is speculators' sliding_window_non_causal; z-lab's
     * layout says is_causal. */
    {
        std::snprintf(key, sizeof key, "%slayer_types", lp);
        std::vector<int> sliding;
        if (const char* lt = rad_meta_gets(meta, key, nullptr))
            RAD_ARCH_TRY(parse_layer_kinds("gemma4 DFlash2", lt, {"sliding_attention"},
                                           "full_attention", &sliding));
        const bool any_full = std::find(sliding.begin(), sliding.end(), 0) != sliding.end();
        const bool any_slide = std::find(sliding.begin(), sliding.end(), 1) != sliding.end();
        if (any_full && any_slide) {
            fprintf(stderr, "radiance: gemma4: the DFlash2 drafter mixes sliding and full layers; "
                            "the block takes one attention width for all of them\n");
            return RAD_E_UNSUPPORTED;
        }
        std::snprintf(key, sizeof key, "%ssliding_window", lp);
        const long long win = rad_meta_geti(meta, key, 0);
        std::snprintf(key, sizeof key, "%suse_sliding_window", lp);
        if (any_slide || rad_meta_geti(meta, key, 0))
            dc.window = df2_geti(meta, "swa_window_size", win);
        const long long nc = rad_meta_geti(meta, "draft.sliding_window_non_causal", -1);
        dc.causal = nc >= 0 ? (nc ? 0 : 1) : (rad_meta_geti(meta, "draft.is_causal", 0) ? 1 : 0);
    }

    /* The query block must be a power of two (the convolution masks a row's position with
     * block_size - 1), and the depth the probe named is the trained one. */
    const long long trained = gemma4::dflash_block(meta);
    if (dc.block < 2 || (dc.block & (dc.block - 1))) {
        fprintf(stderr, "radiance: gemma4: DFlash2's convolution masks the block position with "
                        "block_size - 1, so the query block must be a power of two; %d draft "
                        "tokens make it %lld (trained at %lld)\n",
                ctx->max_spec, (long long)dc.block, trained);
        return RAD_E_UNSUPPORTED;
    }
    if (trained > 0 && trained != dc.block)
        rad_note(b, "DFlash2 was trained at block_size %lld (%lld draft tokens) and runs at %lld; "
                    "it accepts less away from its trained block", trained, trained - 1,
                 (long long)dc.block);
    /* The three scalars rad_block_dflash2.h does not apply (z-lab's names; speculators has none). */
    if (rad_meta_getf(meta, "draft.dflash_config.input_embedding_scale", 1.0) != 1.0 ||
        rad_meta_getf(meta, "draft.dflash_config.output_multiplier", 1.0) != 1.0 ||
        rad_meta_getf(meta, "draft.dflash_config.final_logit_softcapping", 0.0) != 0.0) {
        fprintf(stderr, "radiance: gemma4: this DFlash2 checkpoint scales its embedding or its "
                        "logits, or soft-caps them; the block applies none of the three\n");
        return RAD_E_UNSUPPORTED;
    }

    RAD_ARCH_TRY(drafter_kv_group(b, nm, ctx, dg, dc.window, (int)m.n_main, &m.kv_draft));

    const int64_t qd  = dg.n_head * dg.head_dim;
    const int64_t kvd = dg.n_head_kv * dg.head_dim;
    int64_t qr = dg.max_seqs * dc.block;
    if (qr > dg.max_tok) qr = dg.max_tok;
    const int64_t sr   = dg.max_seqs * dc.steps();
    const int64_t spad = ((sr + 15) / 16) * 16;

    Dflash2::Wire& dw = m.df2w;
    dw.src     = m.b_dfsrc;
    dw.c       = decl_b(b, nm.f("df_c"),    dg.act_dtype, {dg.max_tok, dg.n_embd});
    dw.ck      = decl_b(b, nm.f("df_ck"),   dg.act_dtype, {dg.max_tok, kvd});
    dw.cv      = decl_b(b, nm.f("df_cv"),   dg.act_dtype, {dg.max_tok, kvd});
    dw.x       = decl_b(b, nm.f("df_x"),    dg.act_dtype, {qr, dg.n_embd});
    dw.h       = decl_b(b, nm.f("df_h"),    dg.act_dtype, {qr, dg.n_embd});
    dw.h2      = decl_b(b, nm.f("df_h2"),   dg.act_dtype, {qr, dg.n_embd});
    dw.coef    = decl_b(b, nm.f("df_coef"), dg.act_dtype, {qr, dc.coef_dim(dg.n_embd)});
    dw.q       = decl_b(b, nm.f("df_q"),    dg.act_dtype, {qr, qd});
    dw.k       = decl_b(b, nm.f("df_k"),    dg.act_dtype, {qr, kvd});
    dw.v       = decl_b(b, nm.f("df_v"),    dg.act_dtype, {qr, kvd});
    dw.attn    = decl_b(b, nm.f("df_attn"), dg.act_dtype, {qr, qd});
    dw.gate_up = decl_b(b, nm.f("df_gate_up"), dg.act_dtype, {qr, 2 * dg.n_ff});
    dw.ffn     = decl_b(b, nm.f("df_ffn"),  dg.act_dtype, {qr, dg.n_ff});
    dw.srcq    = decl_b(b, nm.f("df_srcq"), RAD_F8E4M3, {dg.max_tok, dc.n_taps * dg.n_embd});
    dw.srcs    = decl_b(b, nm.f("df_srcs"), RAD_F32,
                        {dg.max_tok, fp8_blocks(dc.n_taps * dg.n_embd)});
    dw.cq      = decl_b(b, nm.f("df_cq"),   RAD_F8E4M3, {dg.max_tok, dg.n_embd});
    dw.cs      = decl_b(b, nm.f("df_cs"),   RAD_F32,    {dg.max_tok, fp8_blocks(dg.n_embd)});
    dw.h2q     = decl_b(b, nm.f("df_h2q"),  RAD_F8E4M3, {qr, dg.n_embd});
    dw.h2s     = decl_b(b, nm.f("df_h2s"),  RAD_F32,    {qr, fp8_blocks(dg.n_embd)});
    dw.atq     = decl_b(b, nm.f("df_atq"),  RAD_F8E4M3, {qr, qd});
    dw.ats     = decl_b(b, nm.f("df_ats"),  RAD_F32,    {qr, fp8_blocks(qd)});
    dw.ffq     = decl_b(b, nm.f("df_ffq"),  RAD_F8E4M3, {qr, dg.n_ff});
    dw.ffs     = decl_b(b, nm.f("df_ffs"),  RAD_F32,    {qr, fp8_blocks(dg.n_ff)});
    dw.sel_h   = decl_b(b, nm.f("df_selh"), dg.act_dtype, {sr, dg.n_embd});
    dw.hp      = decl_b(b, nm.f("df_hp"),   dg.act_dtype, {sr, dc.sel_rank});
    dw.dq      = decl_b(b, nm.f("df_dq"),   RAD_I8,  {sr, dg.n_embd});
    dw.ds      = decl_b(b, nm.f("df_ds"),   RAD_F32, {sr, 1});
    dw.dsum    = decl_b(b, nm.f("df_dsum"), RAD_F32, {dg.n_embd / 128, spad});
    dw.dlog    = decl_b(b, nm.f("df_dlog"), dg.act_dtype, {sr, Dflash2::head_rows(dg)});
    /* No rescoring against the trunk's head: the -df2 recipe's head is already 8-bit, so the
     * candidates are the ones an 8-bit head ranks (rad_block_dflash2.h, head_i8). */
    const int64_t tw = dc.sel_topk;
    dw.cand    = decl_b(b, nm.f("df_cand"),  RAD_I32, {sr, tw});
    dw.unary   = decl_b(b, nm.f("df_unary"), RAD_F32, {sr, tw});
    if (Dflash2::head_sharded(dg)) {
        dw.dpair = decl_b(b, nm.f("df_dpair"), RAD_F32, {sr, tw, 2});
        dw.dgath = decl_b(b, nm.f("df_dgath"), RAD_F32, {sr, (int64_t)dg.world * tw, 2});
    }
    const int64_t nodes = dc.tree_nodes > 0 ? dc.tree_nodes : dc.steps();
    dw.tokens  = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {dg.max_seqs, nodes});
    /* Trees (RadBuildCtx::draft_tree; the trunk's attention already has its tree form): the
     * selector's lattice grown best-first into `steps` nodes, each edge weighed by the softmax of
     * its scores over a temperature. 1.25 measured best at 7 nodes on the speculator benchmarks
     * (greedy, and sampled at 0.7 and 1.0); its path probabilities are calibrated there. */
    if (m.tree) {
        dc.tree = true;
        dc.tree_temp = 1.25;
        dw.parents = decl_b(b, nm.f("dflash_parents"), RAD_I32, {dg.max_seqs, nodes});
        /* A block of 16 (the DBloom drafter): its 15-node tree's verify rows pay at one sequence
         * and not at many, so the query pass is declared at half the block too, over which the
         * drafter grows a 7-node tree of its own: the drafter's short block
         * (rad_drafter_short_block), which the core runs from two decoding sequences on
         * (--draft-short-from auto). A shorter block's tree verifies as few rows as that already. */
        if (dc.block >= 16 && (dc.block & (dc.block - 1)) == 0) {
            dc.short_block = 1;
            dc.short_rows  = dc.block / 2;
        }
    }

    /* The head copy comes from the tied table, under the block's own name; the recipe makes it
     * 8-bit. */
    RAD_ARCH_TRY(m.df2.declare(b, nm, dg, dc, (int)m.n_main, m.kv_draft, m.w_tok, "draft",
                               nm.ckpt("model.language_model.embed_tokens.weight"), dw));
    m.have_df2 = true;
    m.tree = m.df2.cfg.tree;   /* false where dflash_tree did not resolve */
    if (!m.tree && nodes != dc.steps()) {
        fprintf(stderr, "radiance: gemma4: a DFlash2 tree of %lld nodes needs dflash_tree, which no "
                        "library serves here; the block's chain is %lld\n", (long long)nodes,
                (long long)dc.steps());
        return RAD_E_UNSUPPORTED;
    }

    RadDrafterDecl drf{};
    drf.name           = "dflash2";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    drf.block          = dc.block;
    drf.proposal       = dw.tokens;
    drf.proposal_pitch = nodes;
    drf.window         = dc.window;
    drf.mask_token     = (int32_t)df2_geti(meta, "mask_token_id", -1);
    if (drf.mask_token < 0) {
        fprintf(stderr, "radiance: gemma4: this DFlash2 checkpoint declares no mask_token_id, and "
                        "every drafted position embeds it\n");
        return RAD_E_FORMAT;
    }
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));
    if (m.tree) {
        RAD_ARCH_TRY(rad_drafter_tree(b, dw.parents));
        if (m.df2.block_lo > 0) {
            RAD_ARCH_TRY(rad_drafter_short_block(b, m.df2.block_lo));
            rad_note(b, "DFlash2 drafter: a short block of %lld rows, a tree of %lld nodes over it",
                     (long long)m.df2.block_lo, (long long)m.df2.sel_lo_nodes);
        }
        rad_note(b, "DFlash2 drafter: a tree of %lld nodes a step (softmax temperature %.3g) over "
                    "the lattice of its %lld-row block, verified in %lld rows", (long long)nodes,
                 dc.tree_temp, (long long)dc.block, (long long)nodes + 1);
    }

    rad_note(b, "DFlash2 drafter: %lld layers at absolute %lld.., %lld heads / %lld kv x %lld, ffn "
                "%lld, %s attention with a %lld-token window, %d trunk taps; one forward proposes "
                "%lld tokens through the candidate selector, its head %s",
             (long long)dg.n_layer, (long long)m.n_main, (long long)dg.n_head_all,
             (long long)dg.n_head_kv_all, (long long)dg.head_dim, (long long)dg.n_ff_all,
             dc.causal ? "causal" : "non-causal", (long long)dc.window, m.n_taps,
             (long long)dc.steps(), m.df2.head_i8 ? "8-bit" : "2-bit");
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;

    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));

    if (meta->n_expert > 0) {
        fprintf(stderr, "radiance: gemma4_bf16 refuses a checkpoint with %lld experts; that is not this arch\n",
                (long long)meta->n_expert);
        return RAD_E_UNSUPPORTED;
    }

    Geom&  g  = m.g;
    Names& nm = m.nm;

    /* ---- geometry RadModelMeta does not name. */
    /* num_global_key_value_heads may be null (Gemma 4 E2B/E4B): the global layers then take the
     * model's num_key_value_heads, as transformers' per-layer config does. */
    const int64_t kv_g_all    = meta_text_geti(meta, "num_global_key_value_heads",
                                               meta_text_geti(meta, "num_key_value_heads", 0));
    const int64_t head_g      = meta_text_geti(meta, "global_head_dim", 0);
    const int64_t sliding_win = meta_text_geti(meta, "sliding_window", 0);
    const int64_t kv_shared   = meta_text_geti(meta, "num_kv_shared_layers", 0);
    const int64_t interval    = meta_text_geti(meta, "full_attention_interval", 0);
    /* Whether the global layers read V from K (attention_k_eq_v; the 31B's do, and have no
     * v_proj). A checkpoint that does not say is refused rather than guessed. */
    const long long k_eq_v    = meta_text_geti(meta, "attention_k_eq_v", -1);
    if (k_eq_v != 0 && k_eq_v != 1) {
        fprintf(stderr, "radiance: gemma4: the config does not say whether the global layers read "
                        "V from K (attention_k_eq_v)\n");
        return RAD_E_INVAL;
    }
    const double  theta_s     = meta_text_getf(meta, "rope_parameters.sliding_attention.rope_theta", 0.0);
    const double  theta_g     = meta_text_getf(meta, "rope_parameters.full_attention.rope_theta", 0.0);
    const double  softcap     = meta_text_getf(meta, "final_logit_softcapping", 0.0);
    double        prf         = meta_text_getf(meta, "rope_parameters.full_attention.partial_rotary_factor", -1.0);

    if (head_g <= 0 || kv_g_all <= 0) {
        fprintf(stderr, "radiance: gemma4: global_head_dim %lld / num_global_key_value_heads %lld; both required\n",
                (long long)head_g, (long long)kv_g_all);
        return RAD_E_INVAL;
    }
    if (head_g & 1) { fprintf(stderr, "radiance: gemma4: global head_dim %lld is odd\n", (long long)head_g); return RAD_E_INVAL; }
    if (!(theta_s > 0.0) || !(theta_g > 0.0)) {
        fprintf(stderr, "radiance: gemma4: both rope_parameters...rope_theta are required\n");
        return RAD_E_INVAL;
    }
    if (!(softcap > 0.0)) { fprintf(stderr, "radiance: gemma4: final_logit_softcapping is required\n"); return RAD_E_INVAL; }
    if (kv_shared < 0) { fprintf(stderr, "radiance: gemma4: num_kv_shared_layers %lld is negative\n", (long long)kv_shared); return RAD_E_INVAL; }
    if (kv_shared >= g.n_layer) {
        fprintf(stderr, "radiance: gemma4: num_kv_shared_layers %lld leaves no layer with a cache of "
                        "its own (%lld layers)\n", (long long)kv_shared, (long long)g.n_layer);
        return RAD_E_INVAL;
    }
    if (sliding_win <= 0) { fprintf(stderr, "radiance: gemma4: sliding_window %lld is required\n", (long long)sliding_win); return RAD_E_INVAL; }
    m.n_ple = meta_text_geti(meta, "hidden_size_per_layer_input", 0);
    if (m.n_ple < 0) { fprintf(stderr, "radiance: gemma4: hidden_size_per_layer_input %lld is negative\n", (long long)m.n_ple); return RAD_E_INVAL; }
    if (m.n_ple > 0) {
        const int64_t v_ple = meta_text_geti(meta, "vocab_size_per_layer_input", g.n_vocab_all);
        if (v_ple != g.n_vocab_all) {
            fprintf(stderr, "radiance: gemma4: vocab_size_per_layer_input %lld differs from the "
                            "vocabulary (%lld)\n", (long long)v_ple, (long long)g.n_vocab_all);
            return RAD_E_UNSUPPORTED;
        }
    }
    if (!(prf > 0.0)) prf = 1.0;
    const int64_t n_rot_g = (int64_t)(head_g * prf + 0.5);
    if (n_rot_g <= 0 || n_rot_g > head_g || (n_rot_g & 1)) {
        fprintf(stderr, "radiance: gemma4: global rotary width %lld out of range for head_dim %lld\n",
                (long long)n_rot_g, (long long)head_g);
        return RAD_E_INVAL;
    }

    /* The global class's KV heads, replicated like the sliding class's when there are fewer than
     * ranks (geom_from): each rank keeps one, at row kv_off_g. */
    int64_t kv_g = 0, kv_off_g = 0;
    if (g.world > kv_g_all) {
        if (g.world % kv_g_all) {
            fprintf(stderr, "radiance: gemma4: %d ranks cannot replicate %lld global KV heads; the "
                            "world must be a multiple of the KV head count\n",
                    g.world, (long long)kv_g_all);
            return RAD_E_INVAL;
        }
        kv_g = 1;
        kv_off_g = (int64_t)(g.rank / (g.world / kv_g_all)) * head_g;
    } else if (divide_or_fail(kv_g_all, g.world, "num_global_key_value_heads", &kv_g) < 0) {
        return RAD_E_INVAL;
    }

    /* Plain RMSNorm, x_hat * w, on every norm including the post-projection ones. Unlike
     * Gemma 2/3, Gemma 4 does not use the (1 + w) form: vLLM's stock RMSNorm, HF's Gemma4RMSNorm
     * (weight init 1.0) and llama.cpp's LLM_NORM_RMS all multiply by the weight alone. */
    g.wadd    = 0.0f;
    g.wadd_qk = 0.0f;
    const float final_wadd = 0.0f;

    m.n_main = g.n_layer;
    std::vector<int> is_global;
    RAD_ARCH_TRY(g4_schedule(meta, g.n_layer, interval, is_global));
    int n_global = 0;
    for (int v : is_global) n_global += v;

    /* KV sharing: from layer n_layer - num_kv_shared_layers on, a layer keeps no cache and reads the
     * last earlier non-shared layer of its own class (transformers' store_full_length_kv; E4B:
     * sliding layers read 22, full layers 23). kv_src[l] is that layer, or -1 for an own cache. */
    const int64_t first_shared = g.n_layer - kv_shared;
    std::vector<int> kv_src((size_t)g.n_layer, -1);
    if (kv_shared > 0) {
        int last_s = -1, last_f = -1;
        for (int64_t l = 0; l < first_shared; ++l) (is_global[(size_t)l] ? last_f : last_s) = (int)l;
        for (int64_t l = first_shared; l < g.n_layer; ++l) {
            const int src = is_global[(size_t)l] ? last_f : last_s;
            if (src < 0) {
                fprintf(stderr, "radiance: gemma4: KV-shared layer %lld is %s and no earlier layer of "
                                "that class keeps a cache\n", (long long)l,
                        is_global[(size_t)l] ? "full" : "sliding");
                return RAD_E_INVAL;
            }
            kv_src[(size_t)l] = src;
        }
    }

    /* A double-wide MLP (use_double_wide_mlp, Gemma 4 E2B): the layers from first_shared on have
     * 2 * intermediate_size, as transformers' Gemma4TextMLP builds them (`use_double_wide_mlp and
     * is_kv_shared_layer`). Each layer declares its own weights and op widths, and the two widths
     * keep a pair of activation buffers each (b_gu_w). */
    const bool wide_mlp = g4_getb(meta, "use_double_wide_mlp", false) && kv_shared > 0;
    const int64_t n_ff_wide = 2 * g.n_ff;

    /* The sliding class comes from geom_from (head_dim 256, 16 kv heads, full rotation, theta_s);
     * the global class is built from the container header. */
    m.cs.n_head = g.n_head;      m.cs.n_head_kv = g.n_head_kv; m.cs.head_dim = g.head_dim;
    m.cs.n_rot  = g.head_dim;    m.cs.window = sliding_win;    m.cs.theta = theta_s;
    m.cs.scale  = g.rope_scale;  m.cs.head_ladder = false;     m.cs.has_v = true;
    m.cg.n_head = g.n_head;      m.cg.n_head_kv = kv_g;        m.cg.head_dim = head_g;
    m.cg.n_rot  = n_rot_g;       m.cg.window = 0;              m.cg.theta = theta_g;
    m.cg.scale  = g.rope_scale;  m.cg.head_ladder = true;      m.cg.has_v = k_eq_v == 0;
    m.cs.kv_rep = g.world > g.n_head_kv_all;  m.cs.kv_off = g.kv_off;
    m.cs.n_head_kv_all = g.n_head_kv_all;
    m.cg.kv_rep = g.world > kv_g_all;         m.cg.kv_off = kv_off_g;
    m.cg.n_head_kv_all = kv_g_all;

    /* ---- the two KV groups. */
    const uint32_t kvdt = rad_kv_cache_dtype(ctx, RAD_BF16);
    RadKVGroupDecl sd{};
    sd.kind = RAD_KV_WINDOW;  sd.dtype = kvdt;  sd.window = sliding_win;
    sd.n_head_kv = g.n_head_kv;  sd.head_dim = g.head_dim;
    m.kv_s = rad_decl_kv_group(b, nm.f("kv_sliding"), &sd);
    RadKVGroupDecl gd{};
    gd.kind = RAD_KV_FULL;    gd.dtype = kvdt;  gd.window = 0;
    gd.n_head_kv = kv_g;      gd.head_dim = head_g;
    m.kv_g = rad_decl_kv_group(b, nm.f("kv_global"), &gd);
    g.kv_dtype = rad_dtype_name(kvdt);
    if (m.kv_s == 0 || m.kv_g == 0) { fprintf(stderr, "radiance: gemma4: a KV group failed to declare\n"); return RAD_E_INVAL; }

    /* ---- buffers. The two classes have different row widths, so they keep their own rows. */
    m.b_x  = decl_b(b, nm.f("x"), g.act_dtype, {g.max_tok, g.n_embd});
    m.b_h  = decl_b(b, nm.f("h"), g.act_dtype, {g.max_tok, g.n_embd});
    m.b_qs    = decl_b(b, nm.f("attn_qs"),   g.act_dtype, {g.max_tok, m.cs.q_dim()});
    m.b_ks    = decl_b(b, nm.f("attn_ks"),   g.act_dtype, {g.max_tok, m.cs.kv_dim()});
    m.b_vs    = decl_b(b, nm.f("attn_vs"),   g.act_dtype, {g.max_tok, m.cs.kv_dim()});
    m.b_attns = decl_b(b, nm.f("attn_outs"), g.act_dtype, {g.max_tok, m.cs.q_dim()});
    m.b_qg    = decl_b(b, nm.f("attn_qg"),    g.act_dtype, {g.max_tok, m.cg.q_dim()});
    m.b_kg    = decl_b(b, nm.f("attn_kg"),    g.act_dtype, {g.max_tok, m.cg.kv_dim()});
    m.b_vg    = decl_b(b, nm.f("attn_vg"),    g.act_dtype, {g.max_tok, m.cg.kv_dim()});
    m.b_atting= decl_b(b, nm.f("attn_outg"),  g.act_dtype, {g.max_tok, m.cg.q_dim()});
    m.b_gu   = decl_b(b, nm.f("gate_up"), g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.b_ffn  = decl_b(b, nm.f("ffn"),     g.act_dtype, {g.max_tok, g.n_ff});
    if (wide_mlp) {
        m.b_gu_w  = decl_b(b, nm.f("gate_up_wide"), g.act_dtype, {g.max_tok, 2 * n_ff_wide});
        m.b_ffn_w = decl_b(b, nm.f("ffn_wide"),     g.act_dtype, {g.max_tok, n_ff_wide});
        if (!m.b_gu_w || !m.b_ffn_w) return RAD_E_INVAL;
    }
    m.b_hout = decl_b(b, nm.f("h_out"),   g.act_dtype, {g.max_logit_rows, g.n_embd});
    m.b_logits = decl_b(b, nm.f("logits"), RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the DFlash drafter's taps, declared when the container carries z-lab's drafter and this
     * serve speculates. The plane is declared here and written by an op declared inside each layer
     * it taps, because the buffer planner's liveness is declaration order (the tap below). */
    {
        const int n = gemma4::dflash_kind(meta) == 2
            ? gemma4::dflash2_taps(meta, ctx, g.n_layer, m.tap_layer, Model::MAX_TAPS)
            : gemma4::Dflash1::taps(meta, ctx, g.n_layer, m.tap_layer, Model::MAX_TAPS);
        if (n < 0) return n;
        m.n_taps = n;
        if (n > 0)
            m.b_dfsrc = decl_b(b, nm.f("dflash_src"), g.act_dtype,
                               {g.max_tok, (int64_t)n * g.n_embd});
    }

    /* ---- the vocabulary edges. Gemma 4 ties the head to the embedding. */
    const char* pre = "model.language_model.layers";
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("model.language_model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("model.language_model.norm.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"), nm.ckpt("model.language_model.embed_tokens.weight")));
    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    /* bf16, not f32: the model group is pinned to the mapped tier (token_embd's embed_lookup is
     * host-only), and a widened output_norm could not be served from the mapping. */
    m.w_out = decl_w(b, nm.f("output_norm.weight"), RAD_BF16, {g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    /* Declared at its codes' dtype, like the assistant head's lm_head: bf16 as shipped, or int8 /
     * block fp8 where a recipe (`--quant output.weight=...`) makes it so, selecting the matching
     * logits_gemm kernel. It is the trunk's largest single read at decode (n_vocab x n_embd per
     * rank). */
    m.w_lm  = decl_w(b, nm.f("output.weight"), weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                     {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    /* The embedding is scaled by sqrt(n_embd) at the lookup, not in the table (as in llama.cpp's
     * build_inp_embd and HF's ScaledWordEmbedding), so the plain table stays tied to the head.
     * Without it the token signal in the residual stream would be ~73x too small. The scale is
     * sqrt(n_embd) rounded to bf16, as the reference multiplies by it (gemma4::embed_scale: 73.5
     * at 5376, not 73.32). The first norm hides the difference but the residual stream does not,
     * and the DFlash drafter was trained on the reference's hidden states. */
    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                               RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                               RAD_INT("vocab_offset", g.vocab_off),
                               RAD_F64("mult", gemma4::embed_scale(g.n_embd))),
                    RAD_WEIGHTS(m.w_tok)),
                   {}, {m.b_x});
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                          RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                          {m.b_x}, {m.b_x});
    /* Media rows (an image's or an audio clip's) overwrite their placeholders' embeddings after
     * the lookup and before its all-reduce (RadBatch::mm_embd: rank 0 holds the rows, every other
     * rank zeros). The reference scales token embeddings by sqrt(n_embd) and scatters the media
     * features in unscaled. */
    if (((Gemma4Vision::present(meta) || Gemma4UnifiedVision::present(meta)) &&
         ctx->max_enc_patches > 0) ||
        ((Gemma4Audio::present(meta) || Gemma4UnifiedAudio::present(meta)) &&
         ctx->max_audio_frames > 0))
        m.op_mm = rw(b, RAD_OP(b, "scatter_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     {m.b_x}, {m.b_x});

    /* ---- per-layer embeddings (transformers' get_per_layer_inputs / project_per_layer_inputs):
     *   tok = embed_tokens_per_layer[id] * sqrt(n_ple)                      (bf16, sqrt(256) = 16)
     *   ctx = rmsnorm( (e @ W_proj^T) * n_embd^-0.5 ) per (token, layer)   over n_ple, gain w_norm
     *   pli = (ctx + tok) * 2^-0.5
     * `e` is the scaled embedding (after any image rows are scattered in). The projection's scale
     * is folded into the norm's eps, as embgemma2 does: rmsnorm(s x, eps) = rmsnorm(x, eps / s^2),
     * so no bf16 rounding of s x. The table is read a row a token, so the placement puts it in the
     * pinned host pool by default (--embedding-placement ram), off the card. */
    if (m.n_ple > 0) {
        const int64_t P = g.n_layer * m.n_ple;
        RAD_ARCH_TRY(map_copy(b, nm.f("ple_token_embd.weight"),
                              nm.ckpt("model.language_model.embed_tokens_per_layer.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("ple_proj_norm.weight"),
                              nm.ckpt("model.language_model.per_layer_projection_norm.weight")));
        m.w_ple_tok = decl_w(b, nm.f("ple_token_embd.weight"), RAD_BF16, {g.n_vocab, P},
                             RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
        m.w_ple_norm = decl_w(b, nm.f("ple_proj_norm.weight"), RAD_BF16, {m.n_ple},
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
        m.b_ple     = decl_b(b, nm.f("ple_in"),  g.act_dtype, {g.max_tok, P});
        m.b_ple_tok = decl_b(b, nm.f("ple_tok"), g.act_dtype, {g.max_tok, P});
        m.b_pg      = decl_b(b, nm.f("ple_gate"), g.act_dtype, {g.max_tok, m.n_ple});
        if (!m.w_ple_tok || !m.w_ple_norm || !m.b_ple || !m.b_ple_tok || !m.b_pg) return RAD_E_INVAL;
        m.op_ple_tok = rw(b, RAD_OP(b, "embed_lookup",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", P),
                                     RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                     RAD_INT("vocab_offset", g.vocab_off),
                                     RAD_F64("mult", std::sqrt((double)m.n_ple))),
                          RAD_WEIGHTS(m.w_ple_tok)),
                         {}, {m.b_ple_tok});
        /* A rank's lookup writes zeros for ids outside its vocabulary slice; the all-reduce makes
         * every row whole. */
        if (g.world > 1) {
            m.op_ple_ar = rw(b, RAD_OP(b, "all_reduce",
                                 RAD_PARAMS(RAD_INT("world_size", g.world),
                                            RAD_RANGE("numel", P, g.max_tok * P),
                                            RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                                 RAD_NOWEIGHTS),
                             {m.b_ple_tok}, {m.b_ple_tok});
            if (!m.op_ple_ar) return RAD_E_INVAL;
        }
        RAD_ARCH_TRY(m.ln_ple_proj.declare(b, nm, g, nm.f("ple_model_proj"),
                                           { nm.ckpt("model.language_model.per_layer_model_projection.weight") },
                                           P, g.n_embd, RAD_SHARD_NONE, grp_model(), m.b_x, m.b_ple));
        m.op_ple_norm = rw(b, RAD_OP(b, "rmsnorm",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_layer), RAD_INT("n", m.n_ple),
                                      RAD_F64("eps", (double)g.eps * (double)g.n_embd),
                                      RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                           RAD_WEIGHTS(m.w_ple_norm)),
                          {m.b_ple}, {m.b_ple});
        m.op_ple_add = rw(b, RAD_OP(b, "add",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", P),
                                     RAD_STR("dtype", g.dtype), RAD_F64("div", std::sqrt(2.0))),
                          RAD_NOWEIGHTS),
                         {m.b_ple, m.b_ple_tok}, {m.b_ple});
        if (!m.op_ple_tok || !m.op_ple_norm || !m.op_ple_add) return RAD_E_INVAL;
        if (m.op_mm) {
            const int pad = (int)meta_text_geti(meta, "pad_token_id", 0);
            RAD_ARCH_TRY(map_slice(b, nm.f("ple_pad_row.weight"),
                                   nm.ckpt("model.language_model.embed_tokens_per_layer.weight"), pad));
            m.w_ple_pad = decl_w(b, nm.f("ple_pad_row.weight"), RAD_BF16, {P}, RAD_ACCESS_PER_TOKEN,
                                 RAD_SHARD_NONE, grp_model());
            m.b_ple_pad = decl_b(b, nm.f("ple_pad"), g.act_dtype, {g.max_tok, P});
            if (!m.w_ple_pad || !m.b_ple_pad) return RAD_E_INVAL;
            m.op_ple_pfill = rw(b, RAD_OP(b, "fill",
                                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", P),
                                               RAD_STR("dtype", g.dtype),
                                               RAD_F64("value", std::sqrt((double)m.n_ple))),
                                    RAD_NOWEIGHTS),
                                {}, {m.b_ple_pad});
            m.op_ple_pmul = rw(b, RAD_OP(b, "mul_weight",
                                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", P),
                                              RAD_STR("dtype", g.dtype), RAD_INT("scalar", 0)),
                                   RAD_WEIGHTS(m.w_ple_pad)),
                               {m.b_ple_pad}, {m.b_ple_pad});
            m.op_ple_pscat = rw(b, RAD_OP(b, "scatter_rows",
                                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", P),
                                               RAD_STR("dtype", "bf16")),
                                    RAD_NOWEIGHTS),
                                {m.b_ple_pad, m.b_ple_tok}, {m.b_ple_tok});
            if (!m.op_ple_pfill || !m.op_ple_pmul || !m.op_ple_pscat) return RAD_E_INVAL;
        }
    }

    /* ---- the layers. */
    Gemma4Attn::Wire sw{}; sw.x = m.b_x; sw.h = m.b_h; sw.q = m.b_qs; sw.k = m.b_ks; sw.v = m.b_vs;        sw.attn = m.b_attns;
    Gemma4Attn::Wire gw{}; gw.x = m.b_x; gw.h = m.b_h; gw.q = m.b_qg; gw.k = m.b_kg; gw.v = m.b_vg;        gw.attn = m.b_atting;
    Gemma4Mlp::Wire  mw{}; mw.x = m.b_x; mw.h = m.b_h; mw.gu = m.b_gu; mw.ffn = m.b_ffn;

    /* Tree verification for a DFlash2 drafter (RadBuildCtx::draft_tree, which the engine sets
     * from this plugin's probe; rad_drafter_tree): asked of every layer's attention here, before
     * the drafter, and used only if every layer has it. Not where a layer reads another layer's
     * cache (KV sharing, Gemma 4 E2B/E4B): untested there. */
    const bool tree_want = ctx->max_spec > 0 && m.n_taps > 0 && gemma4::dflash_kind(meta) == 2 &&
                           kv_shared == 0 && ctx->draft_tree != 0;
    bool tree_all = tree_want;

    m.layers.resize((size_t)m.n_main);
    for (int64_t l = 0; l < m.n_main; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.global = (is_global[(size_t)l] == 1);
        const ClassGeom&        cgeo = lay.global ? m.cg : m.cs;
        const rad_kvgroup       kg   = lay.global ? m.kv_g : m.kv_s;
        const Gemma4Attn::Wire& aw   = lay.global ? gw : sw;
        /* A KV-shared layer is not bound: the group's pages hold only the layers that write. */
        if (kv_src[(size_t)l] < 0) RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, kg));
        lay.attn.tree = tree_want;
        RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, cgeo, kg, aw, pre,
                                      lay.global ? &m.b_qkvg : &m.b_qkvs, kv_src[(size_t)l]));
        if (tree_want && !lay.attn.op_attn_tree) tree_all = false;
        /* The double-wide layers write their own pair of buffers (see b_gu_w). */
        Gemma4Mlp::Wire lw = mw;
        const bool wide = wide_mlp && l >= first_shared;
        if (wide) { lw.gu = m.b_gu_w; lw.ffn = m.b_ffn_w; }
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, lw, pre, wide ? 2 * g.n_ff : g.n_ff));

        /* The per-layer scalar: `cur = cur * scalar`, a [1] bf16 over the residual stream. */
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.layer_scalar", (int)l),
                              nm.ckpt("%s.%lld.layer_scalar", pre, (long long)l)));
        lay.w_scalar = decl_w(b, nm.f("blk.%d.layer_scalar", (int)l), RAD_BF16, {1},
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer((int)l));
        lay.op_scale = rw(b, RAD_OP(b, "scale",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(lay.w_scalar)),
                           {mw.x}, {mw.x});

        /* The per-layer embedding branch (transformers' Gemma4TextDecoderLayer, after the
         * feed-forward's residual add): gate n_embd -> n_ple, gelu_tanh times this layer's slice of
         * `ple_in`, projection back to n_embd into `h`, its norm, the residual add. The scalar
         * follows it. */
        if (m.n_ple > 0) {
            RAD_ARCH_TRY(lay.ln_pg.declare(b, nm, g, nm.f("blk.%d.ple_gate", (int)l),
                                           { nm.ckpt("%s.%lld.per_layer_input_gate.weight", pre, (long long)l) },
                                           m.n_ple, g.n_embd, RAD_SHARD_NONE, grp_layer((int)l),
                                           m.b_x, m.b_pg));
            RAD_ARCH_TRY(lay.ln_pp.declare(b, nm, g, nm.f("blk.%d.ple_proj", (int)l),
                                           { nm.ckpt("%s.%lld.per_layer_projection.weight", pre, (long long)l) },
                                           g.n_embd, m.n_ple, RAD_SHARD_NONE, grp_layer((int)l),
                                           m.b_pg, m.b_h));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_post_norm.weight", (int)l),
                                  nm.ckpt("%s.%lld.post_per_layer_input_norm.weight", pre, (long long)l)));
            lay.w_ple_post = decl_w(b, nm.f("blk.%d.ple_post_norm.weight", (int)l), RAD_BF16, {g.n_embd},
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer((int)l));
            lay.op_ple_act = rw(b, RAD_OP(b, "act_mul",
                                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", m.n_ple),
                                               RAD_STR("act", "gelu_tanh"), RAD_STR("dtype", g.dtype)),
                                    RAD_NOWEIGHTS),
                                {m.b_pg, m.b_ple}, {m.b_pg});
            lay.op_ple_post = rw(b, RAD_OP(b, "rmsnorm",
                                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                                RAD_STR("dtype", g.dtype)),
                                     RAD_WEIGHTS(lay.w_ple_post)),
                                 {m.b_h}, {m.b_h});
            lay.op_ple_add = rw(b, RAD_OP(b, "add",
                                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                               RAD_STR("dtype", g.dtype)),
                                    RAD_NOWEIGHTS),
                                {m.b_x, m.b_h}, {m.b_x});
            if (!lay.w_ple_post || !lay.op_ple_act || !lay.op_ple_post || !lay.op_ple_add)
                return RAD_E_INVAL;
        }

        /* The tap: a row copy of the layer's output (after the scalar, HF's hidden_states[l + 1])
         * into the column slice the tap owns. It is declared here, in the layer it is issued
         * after, because the buffer planner's liveness is declaration order. Declared after the
         * trunk, `dflash_src` would look born at its end and free to share with what every layer
         * writes: the drafts would fail while the reply stays fluent. */
        for (int t = 0; t < m.n_taps; ++t)
            if (m.tap_layer[t] == (int)l && !lay.op_tap)
                lay.op_tap = rw(b, RAD_OP(b, "cast",
                                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                               RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                                    RAD_NOWEIGHTS),
                                {m.b_x}, {m.b_dfsrc});
    }

    /* ---- the junctions, declared once every layer exists (j_ma reads the next layer's gain).
     * Used only when all resolved and every linear a junction feeds reads its int8 pair (q, k and
     * v through q's, and gate_up), at either grid: a junction writes that pair, not the bf16 norm
     * output (an a4 linear's decode band aside), so a bf16 k or v beside an int8 q would read a
     * stale buffer. Otherwise the blocks keep their own norms, quantisers, post-norms and adds. */
    if (m.n_main > 0) {
        bool ok = true;
        for (const Layer& lay : m.layers)
            ok = ok && lay.attn.reads_pair() && lay.mlp.ln_gu.pair();
        if (ok) {
            RAD_ARCH_TRY(m.j0.declare(b, g, m.b_x, 0, m.layers[0].attn.w_norm, 0, 0,
                                         m.layers[0].attn.ln_in()));
            ok = m.j0.op != 0;
            for (int64_t l = 0; ok && l < m.n_main; ++l) {
                Layer& lay = m.layers[(size_t)l];
                /* the split form's residual on every row: before the last layer's feed-forward
                 * (its own reduction, post-norm and add over every row), and where the drafter
                 * taps the stream */
                bool tapped = false;
                for (int t = 0; t < m.n_taps; ++t) tapped = tapped || m.tap_layer[t] == (int)l;
                /* every row also where the per-layer branch follows: the feed-forward then closes
                 * with its own all-reduce, post-norm and add */
                RAD_ARCH_TRY(lay.j_am.declare(b, g, m.b_h, m.b_x, lay.mlp.w_norm, lay.attn.w_post, 0,
                                              lay.mlp.ln_gu, l + 1 == m.n_main || m.n_ple > 0));
                ok = lay.j_am.op != 0;
                if (ok && l + 1 < m.n_main) {
                    const Layer& nx = m.layers[(size_t)l + 1];
                    /* With per-layer embeddings the junction closes the PLE branch instead of the
                     * feed-forward (whose post-norm and add then run apart): the branch's norm,
                     * its add, the scalar, the next norm and quantiser, in the reference's order. */
                    RAD_ARCH_TRY(lay.j_ma.declare(b, g, m.b_h, m.b_x, nx.attn.w_norm,
                                                  m.n_ple > 0 ? lay.w_ple_post : lay.mlp.w_post,
                                                  lay.w_scalar, nx.attn.ln_in(), tapped));
                    ok = lay.j_ma.op != 0;
                }
            }
        }
        m.fuse = ok;
        bool ar = ok && (g.world == 2 || g.world == 4);
        int64_t rows = g.max_tok;
        for (int64_t l = 0; ar && l < m.n_main; ++l) {
            const Layer& lay = m.layers[(size_t)l];
            ar = lay.j_am.op_ar != 0 && (l + 1 >= m.n_main || lay.j_ma.op_ar != 0);
            if (ar) rows = std::min(rows, lay.j_am.ar_top());
            if (ar && l + 1 < m.n_main) rows = std::min(rows, lay.j_ma.ar_top());
        }
        m.fuse_ar = ar;
        m.ar_rows = ar ? rows : 0;
    }

    /* ---- prefetch beside the reductions (see prefetch_beside): gate_up's first 20 MB beside the
     * attention block's all-reduce and the next layer's q|k|v's (q alone, unfused; after the last
     * layer, the lm_head's) beside the feed-forward's (gemma4::kPrefetchBytes), then down_proj's
     * first 9 MB beside the attention (gemma4::kPrefetchDownBytes). Only at more than one rank,
     * where there are reductions to sit beside, and not under --weight-prefetch off. */
    if (g.world > 1 && !ctx->weight_prefetch_off && rad_op_declared(b, "prefetch")) {
        for (size_t l = 0; l < m.layers.size(); ++l) {
            Layer& lay = m.layers[l];
            lay.attn.pf_w  = lay.mlp.ln_gu.w;
            lay.attn.op_pf = decl_prefetch(b, lay.attn.pf_w, gemma4::kPrefetchBytes);
            /* after the last layer the lm_head is the next weight streamed */
            if (l + 1 < m.layers.size()) {
                const Gemma4Attn& nx = m.layers[l + 1].attn;
                lay.mlp.pf_w = nx.fused ? nx.ln_qkv.w : nx.ln_q.w;
            } else {
                lay.mlp.pf_w = m.w_lm;
            }
            lay.mlp.op_pf = decl_prefetch(b, lay.mlp.pf_w, gemma4::kPrefetchBytes);
        }
        for (Layer& lay : m.layers) {
            lay.attn.pfd_w  = lay.mlp.ln_down.w;
            lay.attn.op_pfd = decl_prefetch(b, lay.attn.pfd_w, gemma4::kPrefetchDownBytes);
        }
    }

    /* ---- the prefill overlap (trunk_overlap). It needs a reduction to hide, every junction fused
     * (the slices issue only the fused forms), and the exact wire, the one shown to give a slice
     * the bits it gives the whole. The buffer the second lane reduces in place is taken out of the
     * arena's sharing, which the runtime asks of every buffer a second lane touches; it lives
     * across the whole trunk already. */
    /* Not with per-layer embeddings: the slices close each layer through j_ma alone, without the
     * per-layer branch. */
    m.overlap = g.world > 1 && g.wire_exact && m.fuse && m.n_ple == 0 &&
                g.max_tok >= 2 * kOverlapRows;
    for (const Layer& lay : m.layers)
        m.overlap = m.overlap && lay.attn.op_ar && lay.mlp.op_ar;
    if (m.overlap) RAD_ARCH_TRY(rad_buf_concurrent(b, m.b_h));
    /* The reductions the overlap runs on lane 1 are their own ops at `overlap` 1. Declared like
     * op_ar, they resolve to the wide launch: the exact two-shot over every word of a 512- to
     * 1024-row slice takes nb_for_shard's ceiling, 24 blocks of 1024 threads, each a whole CU
     * spinning on a peer flag for the ~1 ms the link takes -- a third of the card held away from
     * the other slice's GEMMs. At `overlap` 1 the row is ar_twoshot_4rank_exact_lane: the same
     * entry and the same fold per word (the block count moves no byte), four blocks by default
     * (tArTs4Lane). Its range starts at a slice, which is above every one-shot band. */
    if (m.overlap) {
        auto lane_ar = [&]() {
            return rw(b, RAD_OP(b, "all_reduce",
                         RAD_PARAMS(RAD_INT("world_size", g.world),
                                    RAD_RANGE("numel", kOverlapRows * g.n_embd,
                                              g.max_tok * g.n_embd),
                                    RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                    RAD_INT("min_bytes", g.wire_min_bytes),
                                    RAD_INT("overlap", 1)),
                         RAD_NOWEIGHTS),
                      {m.b_h}, {m.b_h});
        };
        for (Layer& lay : m.layers) {
            lay.op_ar_att_ov = lane_ar();
            lay.op_ar_ff_ov  = lane_ar();
        }
    }
    /* ...and with attention per slice. Run whole between the slices, a layer's attention leaves
     * slice A's attention-side reduction only slice B's o_proj to hide behind, and slice B's
     * feed-forward reduction only slice A's junction and q|k|v. Per slice (A's projection,
     * prologue, attention and o_proj, then B's), each of the four reductions sits beside a whole
     * block: on 4x R9700 the exact two-shot of a 1024-row slice takes ~1.05 ms alone and ~1.4x
     * that beside card-filling compute, so the reductions want whole blocks to hide behind. Slice
     * A's attention reads its context end from kv_len_trim, one a KV group. A one-sequence step
     * only (the slices' rows are then one sequence's), with the one-launch prologue and no
     * multi-component rotary. A slice's attention rows see exactly the keys they see in the whole
     * step, so the output is the whole attention's byte for byte. */
    if (m.overlap) {
        bool ok = !g.rope_mc && rad_op_declared(b, "kv_len_trim");
        for (const Layer& lay : m.layers) ok = ok && lay.attn.slicable();
        for (const Layer& lay : m.layers) {
            if (!ok) break;
            int i = 0;
            while (i < m.n_trim && m.trim[i].kv != lay.attn.kv) ++i;
            if (i < m.n_trim) continue;
            if (m.n_trim == 2) { ok = false; break; }
            SliceTrim& t = m.trim[m.n_trim];
            t.kv = lay.attn.kv;
            t.used = decl_b(b, nm.f("slice_used_%d", m.n_trim), RAD_I32, {g.max_seqs});
            ok = t.used && rad_buf_concurrent(b, t.used) == RAD_OK;
            t.op = ok ? rw(b, RAD_OP(b, "kv_len_trim",
                                RAD_PARAMS(RAD_RANGE("trim", 1, g.max_tok),
                                           RAD_INT("max_seqs", g.max_seqs)),
                                RAD_NOWEIGHTS),
                           {}, {t.used})
                      : 0;
            ok = ok && t.op;
            ++m.n_trim;
        }
        m.slice_attn = ok && m.n_trim > 0;
    }

    /* ---- the tail: final norm, the sampled rows, the lm_head, and the soft-cap. */
    m.op_final = rw(b, RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                 RAD_F64("eps", g.eps), RAD_F64("wadd", final_wadd), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(m.w_out)),
                     {m.b_x}, {m.b_h});
    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                     {m.b_h}, {m.b_hout});
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                                 RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(m.w_lm)),
                     {m.b_hout}, {m.b_logits});
    m.op_cap = rw(b, RAD_OP(b, "logit_softcap",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_vocab),
                                 RAD_F64("cap", softcap), RAD_STR("dtype", "f32")),
                      RAD_NOWEIGHTS),
                     {m.b_logits}, {m.b_logits});

    /* ---- the assistant head, when the container carries one (`rad-convert --draft-model`) and
     * the deployment speculates; otherwise its ~0.5 GB of weights stay off the card. */
    if (ctx->max_spec > 0 && has_assistant(meta)) {
        int last_s = -1, last_f = -1;
        /* the last layer of each class that keeps a cache (a KV-shared layer reads another's) */
        for (int64_t l = 0; l < m.n_main; ++l)
            if (kv_src[(size_t)l] < 0) (is_global[(size_t)l] ? last_f : last_s) = (int)l;
        RAD_ARCH_TRY(m.head.declare(b, nm, meta, g, m.cs, m.cg, m.kv_s, m.kv_g, last_s, last_f,
                                    m.n_main, m.w_tok, m.b_h, m.b_logits));
        m.have_head = true;

        /* Serial drafter: `max_spec` dependent passes, each embedding the last one's proposal,
         * through the ordinary sampler chain (proposal 0). The head has no cache of its own, so
         * the core stages every pass at the committed position. */
        RadDrafterDecl dd{};
        dd.name           = "mtp";
        dd.kind           = RAD_DRAFT_SERIAL;
        dd.depth          = ctx->max_spec;
        dd.proposal       = 0;
        dd.proposal_pitch = 1;
        dd.mask_token     = -1;
        RAD_ARCH_TRY(rad_declare_drafter(b, &dd));
        RAD_ARCH_TRY(rad_drafter_shared_kv(b, kHeadPosOffset));
    }

    /* ---- or z-lab's DFlash drafter, past the trunk's layers. It is declared last because
     * declaration order is the buffer planner's liveness. It reads the taps the trunk copied into
     * `dflash_src` and the trunk's tied table, as its embedding and, where the container holds no
     * 8-bit copy, as its head. */
    if (tree_want && !tree_all)
        rad_note(b, "DFlash2 tree: no library serves tree attention for every layer here "
                    "(attn_paged_qrow_tree / attn_paged_tree); the drafter proposes a chain");
    m.tree = tree_all;
    if (m.n_taps > 0 && gemma4::dflash_kind(meta) == 2) {
        RAD_ARCH_TRY(declare_dflash2(b, meta, ctx, m));
    } else if (m.n_taps > 0) {
        gemma4::Dflash1::Trunk t{};
        t.n_taps     = m.n_taps;
        t.escale     = gemma4::embed_scale(g.n_embd);
        t.src        = m.b_dfsrc;
        t.w_tok      = m.w_tok;
        t.tok_src    = nm.ckpt("model.language_model.embed_tokens.weight");
        t.base_layer = (int)m.n_main;
        RAD_ARCH_TRY(gemma4::Dflash1::declare_drafter(b, meta, ctx, nm, g, t, &m.df, &m.kv_draft));
        m.have_df = true;
    }

    /* ---- the vision tower, its weight groups past the trunk's and the draft model's: the
     * assistant's layers and its exit, or the DFlash drafter's layers. They are counted even when
     * the draft model is not declared, so the groups do not move with max_spec. The audio
     * encoder's groups follow the tower's. */
    {
        int64_t head_groups = 0;
        if (has_assistant(meta))
            head_groups = meta_text_geti(meta, "num_hidden_layers", 0, "draft.") + 1;
        else if (gemma4::dflash_kind(meta) == 1)
            head_groups = rad_meta_geti(meta, "draft.num_hidden_layers", 0);
        else if (gemma4::dflash_kind(meta) == 2) {
            char k[160];
            std::snprintf(k, sizeof k, "%snum_hidden_layers", drafter_layer_prefix(meta));
            head_groups = rad_meta_geti(meta, k, 0);
        }
        RAD_ARCH_TRY(m.vis.declare(b, meta, ctx, nm, g.n_embd, (int)(m.n_main + head_groups),
                                   "model.", "gemma4_bf16"));
        RAD_ARCH_TRY(m.aud.declare(b, meta, ctx, nm, g.n_embd,
                                   (int)(m.n_main + head_groups) + (int)m.vis.L.size(), "model.",
                                   "gemma4_bf16"));
        RAD_ARCH_TRY(m.uvis.declare(b, meta, ctx, nm, g.n_embd, "gemma4_bf16"));
        RAD_ARCH_TRY(m.uaud.declare(b, meta, ctx, nm, g.n_embd, "gemma4_bf16"));
    }

    rad_note(b, "rank %d/%d: %lld layers (%lld global, %lld sliding), sliding window %lld, "
                "slider %lld heads x %lld, global %lld heads x %lld, soft-capped logits",
             g.rank, g.world, (long long)g.n_layer, (long long)n_global,
             (long long)(g.n_layer - n_global), (long long)sliding_win,
              (long long)g.n_head, (long long)g.head_dim, (long long)g.n_head, (long long)head_g);
    if (kv_shared > 0) {
        int src_s = -1, src_f = -1;
        for (int64_t l = first_shared; l < g.n_layer; ++l)
            (is_global[(size_t)l] ? src_f : src_s) = kv_src[(size_t)l];
        rad_note(b, "KV sharing: layers %lld..%lld keep no cache and read the last earlier layer of "
                    "their class (sliding %d, full %d); %lld of %lld layers hold KV",
                 (long long)first_shared, (long long)(g.n_layer - 1), src_s, src_f,
                 (long long)first_shared, (long long)g.n_layer);
    }
    {
        int64_t n_fused = 0, n_row = 0, n_dec = 0, dec_hi = 0;
        for (const Layer& lay : m.layers) {
            n_fused += lay.attn.fused ? 1 : 0;
            n_row   += lay.attn.op_attn_row ? 1 : 0;
            if (lay.attn.op_attn_dec) { ++n_dec; dec_hi = lay.attn.dec_hi; }
        }
        if (n_fused)
            rad_note(b, "q, k and v as one linear (blk.*.attn_qkv) on %lld of %lld layers; attention "
                        "reads q in the row at every height on %lld, up to %lld query rows a "
                        "sequence on %lld, and copies it out above that or one row",
                     (long long)n_fused, (long long)m.layers.size(), (long long)n_row,
                     (long long)dec_hi, (long long)n_dec);
        else
            rad_note(b, "q, k and v as three linears (a container converted before attn_qkv, or "
                        "bf16 weights): convert with a recipe naming blk.*.attn_qkv.weight for one");
    }
    if (g.world > 1) {
        if (m.overlap)
            rad_note(b, "prefill overlap: a step of %lld rows or more runs as two slices, each "
                        "slice's all-reduces on a second stream beside the other's compute",
                     (long long)(2 * kOverlapRows));
        else
            rad_note(b, "prefill overlap: off (%s)",
                     !g.wire_exact ? "the lossy all-reduce wire"
                     : !m.fuse     ? "the residual junctions are not fused"
                     : m.n_ple > 0 ? "the per-layer branch is not split into slices"
                                   : "a step holds fewer than two slices");
    }
    if (m.fuse && (g.world == 2 || g.world == 4)) {
        if (!m.fuse_ar)
            rad_note(b, "residual junctions: the all-reduce in front of each issued apart");
        else if (g.world == 4 && m.layers[0].j_am.op_sp)
            rad_note(b, "residual junctions: the all-reduce in front of each folded in at steps of "
                        "%lld rows or fewer (libr4d's four-rank row: the fold to %lld rows, the "
                        "all-reduce split around the junction from %lld), issued apart above",
                     (long long)m.ar_rows, (long long)kArSpMinRows - 1, (long long)kArSpMinRows);
        else if (g.world == 4)
            rad_note(b, "residual junctions: the all-reduce in front of each folded in at steps of "
                        "%lld rows or fewer (the four-rank one-shot), issued apart above",
                     (long long)m.ar_rows);
        else
            rad_note(b, "residual junctions: the all-reduce in front of each folded in");
    }
    return RAD_OK;
}

/* ==================================================================== the prefill overlap
 *
 * A prefill step in two slices, each slice's all-reduces on the runtime's second lane (rad_lane)
 * beside the other slice's compute: arch/gemma4's trunk_overlap, on this trunk. A layer reduces
 * T x n_embd twice, after o_proj and after down_proj, both column-sharded; run whole, the
 * reductions and the compute take turns. Cut in two, a slice's reduction runs while the other
 * slice computes:
 *
 *   lane 0  o(A) o(B) . jam(A) arm(A) . jam(B) arm(B) . jma(A) qkv(A) . jma(B) qkv(B) attn o(A) ..
 *   lane 1       ar(A)       ar(B)              ar(A)              ar(B)
 *
 * Attention runs whole (the prologue, the cache write and attn_paged): it reads every row's keys,
 * and its batch bounds describe the whole step. The rest is a row at a time: the projections, the
 * junctions, the feed-forward arm and the reductions.
 *
 * The same kernels give the same bits (core/runtime/ctx.h, the second lane's rule four). Both slices
 * are at least kOverlapRows rows: above the int8 GEMMs' skinny band, whose prefill and tiled
 * kernels take each row's K in order whatever M is; above r4d_act_threads' last width change, so
 * every junction, quantiser and norm folds a row's sum of squares as the whole step's does; and the
 * exact all-reduce folds a word in fp32 in ascending rank order whatever message carries it. On
 * the exact wire only.
 *
 * One buffer carries both reductions: o_proj and down_proj both write `h`, and the junctions read
 * it. So the order below never lets lane 0 write rows of `h` lane 1 is reducing: slice A's arm
 * writes h[A] only after A's attention reduction was waited for, while lane 1 reduces h[B]. Every
 * reduction rides lane 1 behind a join from lane 0, which holds its producer; lane 0 waits for lane
 * 1 before the reduction's consumer; each wait goes in after the lane-0 work that does not need it.
 * The embedding's reduction stays on lane 0, ahead of everything lane 1 is first joined behind, so
 * the collectives keep one order on every rank. At two ranks the junctions here carry no
 * all-reduce (the reduction rode lane 1); a step that runs whole keeps the fused one. */
static int64_t split_at(const Model& m, int64_t T) {
    if (!m.overlap || T < 2 * kOverlapRows) return 0;
    return (T / 2) / 128 * 128;   /* slice A whole 128-row tiles; both slices >= kOverlapRows */
}

static void trunk_overlap(RadCtx* c, const Model& m, const RadBatch* batch, int64_t cut) {
    const int64_t T = batch->n_tok, E = m.g.n_embd;
    const int64_t a0 = 0, an = cut, b0 = cut, bn = T - cut;
    bool lanes_ok = true;
    /* An all-reduce in place over rows [r0, r0 + n) of `h`, on lane 1 behind everything lane 0
     * has issued -- its producer among it. */
    auto reduce = [&](rad_op op, int64_t r0, int64_t n) {
        lanes_ok = lanes_ok && rad_lane_join(c, 0, 1) == RAD_OK && rad_lane(c, 1) == RAD_OK;
        RAD_ISSUE_N(c, op, n * E, brow_slice(m.b_h, r0, n, E), RAD_NONE);
        lanes_ok = rad_lane(c, 0) == RAD_OK && lanes_ok;
    };
    /* Lane 0 behind every reduction issued so far. */
    auto join_back = [&] { lanes_ok = rad_lane_join(c, 1, 0) == RAD_OK && lanes_ok; };
    /* Rows [r0, r0 + n) of layer li's end, once their feed-forward reduction is in: the junction
     * into the next layer -- or the last layer's post-norm, add and scalar -- and the drafter's
     * taps of those rows. */
    auto join_ff = [&](size_t li, int64_t r0, int64_t n) {
        const Layer& y = m.layers[li];
        if (li + 1 < m.layers.size()) {
            y.j_ma.step_rows(c, E, r0, n);
        } else {
            y.mlp.tail(c, r0, n);
            RAD_ISSUE_N(c, y.op_scale, n, brow_slice(m.b_x, r0, n, E), RAD_W(y.w_scalar),
                        brow_slice(m.b_x, r0, n, E));
        }
        for (int t = 0; t < m.n_taps && y.op_tap; ++t)
            if (m.tap_layer[t] == (int)li)
                RAD_ISSUE_N(c, y.op_tap, n, brow_slice(m.b_x, r0, n, E),
                            bcol_at(m.b_dfsrc, r0, (int64_t)m.n_taps * E, (int64_t)t * E, E, n));
    };

    m.j0.step(c, T);
    const size_t L = m.layers.size();
    if (m.slice_attn && batch->n_seq == 1) {
        /* Attention per slice (see Model::slice_attn): slice A's context end first, then each
         * layer as A's attention block, B's beside A's attention-side reduction, A's feed-forward
         * beside B's, B's feed-forward beside A's, and the next layer's A block beside B's. Each
         * join is issued before the reduction it must not wait for, so it orders lane 0 behind
         * exactly the ones issued so far; lane 1 always reduces rows of `h` lane 0 is not
         * touching. */
        auto used_of = [&](rad_kvgroup kv, bool slice_a) -> RadOperand {
            if (slice_a)
                for (int i = 0; i < m.n_trim; ++i)
                    if (m.trim[i].kv == kv) return brows(m.trim[i].used, 1);
            const RadKVGroupBatch* kvb = kv_batch(batch, kv);
            return praw(kvb ? kvb->seqused : nullptr, RAD_I32, 1);
        };
        for (int i = 0; i < m.n_trim; ++i) {
            const RadKVGroupBatch* kvb = kv_batch(batch, m.trim[i].kv);
            /* the trim is `tail`'s extent, slice B's rows: a launch reads its ranged parameter
             * as the band's upper bound, not as bn */
            RAD_ISSUE_N(c, m.trim[i].op, bn, praw(kvb ? kvb->seqused : nullptr, RAD_I32, 1),
                        praw(batch->positions + b0, RAD_I32, bn), brows(m.trim[i].used, 1));
        }
        for (size_t li = 0; li < L; ++li) {
            const Layer& y = m.layers[li];
            const rad_op ar_att = y.op_ar_att_ov ? y.op_ar_att_ov : y.attn.op_ar;
            const rad_op ar_ff  = y.op_ar_ff_ov ? y.op_ar_ff_ov : y.mlp.op_ar;
            if (li > 0) join_ff(li - 1, a0, an);     /* its reduction was waited for below */
            y.attn.project(c, a0, an);
            y.attn.attend_rows(c, batch, a0, an, used_of(y.attn.kv, true));
            y.attn.out_proj(c, a0, an);
            if (li > 0) join_back();                 /* the last layer's feed-forward, slice B */
            reduce(ar_att, a0, an);
            if (li > 0) join_ff(li - 1, b0, bn);
            y.attn.project(c, b0, bn);
            y.attn.attend_rows(c, batch, b0, bn, used_of(y.attn.kv, false));
            y.attn.out_proj(c, b0, bn);
            join_back();                             /* attention, slice A */
            reduce(ar_att, b0, bn);
            y.j_am.step_rows(c, E, a0, an);
            y.mlp.arm(c, a0, an);
            join_back();                             /* attention, slice B */
            reduce(ar_ff, a0, an);
            y.j_am.step_rows(c, E, b0, bn);
            y.mlp.arm(c, b0, bn);
            join_back();                             /* the feed-forward, slice A */
            reduce(ar_ff, b0, bn);
        }
        join_ff(L - 1, a0, an);
        join_back();                                 /* the last feed-forward, slice B */
        join_ff(L - 1, b0, bn);
        if (!lanes_ok)
            rad_step_fail(c, "gemma4_bf16: the prefill overlap could not order its second lane "
                             "against the first");
        return;
    }
    for (size_t li = 0; li < L; ++li) {
        const Layer& y = m.layers[li];
        if (li == 0) {
            y.attn.project(c, a0, an);
            y.attn.project(c, b0, bn);
        } else {
            join_ff(li - 1, a0, an);         /* its reduction was waited for last layer */
            y.attn.project(c, a0, an);
            join_back();                     /* the last layer's feed-forward, slice B */
            join_ff(li - 1, b0, bn);
            y.attn.project(c, b0, bn);
        }
        y.attn.attend(c, batch);
        const rad_op ar_att = y.op_ar_att_ov ? y.op_ar_att_ov : y.attn.op_ar;
        const rad_op ar_ff  = y.op_ar_ff_ov ? y.op_ar_ff_ov : y.mlp.op_ar;
        y.attn.out_proj(c, a0, an);
        reduce(ar_att, a0, an);
        y.attn.out_proj(c, b0, bn);
        join_back();                         /* attention, slice A */
        reduce(ar_att, b0, bn);
        y.j_am.step_rows(c, E, a0, an);
        y.mlp.arm(c, a0, an);
        join_back();                         /* attention, slice B */
        reduce(ar_ff, a0, an);
        y.j_am.step_rows(c, E, b0, bn);
        y.mlp.arm(c, b0, bn);
        join_back();                         /* the feed-forward, slice A */
        reduce(ar_ff, b0, bn);
    }
    join_ff(L - 1, a0, an);
    join_back();                             /* the last feed-forward, slice B */
    join_ff(L - 1, b0, bn);
    if (!lanes_ok)
        rad_step_fail(c, "gemma4_bf16: the prefill overlap could not order its second lane against "
                         "the first");
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];

    /* An encoder pass runs only its encoder. */
    if (batch->enc) {
        if (batch->enc == RAD_MM_AUDIO) {
            if (m.uaud.on) m.uaud.step(c, batch);
            else           m.aud.step(c, batch);
        } else if (m.uvis.on) {
            m.uvis.step(c, batch);
        } else {
            m.vis.step(c, batch);
        }
        return;
    }
    /* A draft pass runs only the draft model. The trunk already left its final-normed hidden state
     * in `h`, which the assistant's history pass gathers from, and its taps in `dflash_src`, which
     * the DFlash drafter's context pass reads. */
    if (batch->draft_pass != 0) {
        if (m.have_df) {
            if (batch->draft_pass > 0) m.df.step(c, batch);
            else                       m.df.context(c, batch);
            return;
        }
        if (m.have_df2) {
            if (batch->draft_pass > 0) m.df2.step(c, batch);
            else                       m.df2.context(c, batch);
            return;
        }
        if (!m.have_head) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container carries no `draft.` assistant or DFlash drafter, or declare "
                             "ran at max_spec 0");
            return;
        }
        m.head.step(c, batch);
        return;
    }
    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok),
              brows(m.b_x, batch->n_tok));
    /* Before the all-reduce: after it, every rank past 0 would write zeros over its image rows. */
    if (m.op_mm && batch->n_mm_rows > 0)
        RAD_ISSUE_N(c, m.op_mm, batch->n_mm_rows,
                    praw2(batch->mm_embd, RAD_BF16, batch->n_mm_rows, m.g.n_embd),
                    praw(batch->mm_rows, RAD_I32, batch->n_mm_rows), brows(m.b_x, batch->n_tok));
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.b_x, batch->n_tok), RAD_NONE);

    const int64_t T = batch->n_tok;
    if (m.n_ple > 0) {
        RAD_ISSUE(c, m.op_ple_tok, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_ple_tok),
                  brows(m.b_ple_tok, T));
        /* before the pad rows of image positions go in: every rank writes those whole */
        if (m.op_ple_ar)
            RAD_ISSUE_N(c, m.op_ple_ar, T * m.g.n_layer * m.n_ple, brows(m.b_ple_tok, T), RAD_NONE);
        if (m.op_ple_pscat && batch->n_mm_rows > 0) {
            const int64_t nm_rows = batch->n_mm_rows;
            RAD_ISSUE_N(c, m.op_ple_pfill, nm_rows, brows(m.b_ple_pad, nm_rows));
            RAD_ISSUE_N(c, m.op_ple_pmul, nm_rows, brows(m.b_ple_pad, nm_rows), RAD_W(m.w_ple_pad),
                        brows(m.b_ple_pad, nm_rows));
            RAD_ISSUE_N(c, m.op_ple_pscat, nm_rows, brows(m.b_ple_pad, nm_rows),
                        praw(batch->mm_rows, RAD_I32, nm_rows), brows(m.b_ple_tok, T));
        }
        m.ln_ple_proj.step(c, m.b_x, m.b_ple, T);
        /* [T, n_layer * n_ple] read as [T * n_layer, n_ple]: one norm row per (token, layer). */
        RAD_ISSUE_N(c, m.op_ple_norm, T * m.g.n_layer, brows(m.b_ple, T), RAD_W(m.w_ple_norm),
                    brows(m.b_ple, T));
        RAD_ISSUE(c, m.op_ple_add, brows(m.b_ple, T), brows(m.b_ple_tok, T), brows(m.b_ple, T));
    }
    const int64_t cut = split_at(m, T);
    if (cut > 0) trunk_overlap(c, m, batch, cut);
    else if (m.fuse) m.j0.step(c, T);
    /* the junctions carry the reductions in front of them in a step their fold serves */
    const bool far = m.fuse_ar && T <= m.ar_rows;
    for (size_t li = 0; cut == 0 && li < m.layers.size(); ++li) {
        const Layer& l = m.layers[li];
        const bool last = li + 1 == m.layers.size();
        l.attn.step(c, batch, m.fuse, m.fuse, far);
        if (m.fuse) l.j_am.step(c, T, far);
        const bool ple = m.n_ple > 0;
        l.mlp.step(c, batch, m.fuse, m.fuse && !last && !ple, far);
        if (ple) {
            l.ln_pg.step(c, m.b_x, m.b_pg, T);
            RAD_ISSUE(c, l.op_ple_act, brows(m.b_pg, T),
                      bcol(m.b_ple, (int64_t)li * m.n_ple, m.n_ple, T), brows(m.b_pg, T));
            l.ln_pp.step(c, m.b_pg, m.b_h, T);
            if (!(m.fuse && !last)) {
                RAD_ISSUE(c, l.op_ple_post, brows(m.b_h, T), RAD_W(l.w_ple_post), brows(m.b_h, T));
                RAD_ISSUE(c, l.op_ple_add, brows(m.b_x, T), brows(m.b_h, T), brows(m.b_x, T));
            }
        }
        if (m.fuse && !last) {
            /* [all-reduce,] post-norm, add, layer scalar, next block's norm and quantiser: one op.
             * With per-layer embeddings it closes the branch, whose output every rank holds whole
             * (the feed-forward's reduction ran above): no all-reduce in front. */
            l.j_ma.step(c, T, far && !ple);
        } else {
            /* The per-layer scalar, `cur = cur * scalar` over the residual stream, applied after
             * the mlp add, as vLLM's `hidden_states = hidden_states * self.layer_scalar`. The
             * checkpoint's values (0.03-0.99) are trained weights that rescale the residual toward
             * O(1). */
            RAD_ISSUE(c, l.op_scale, brows(m.b_x, batch->n_tok), RAD_W(l.w_scalar),
                      brows(m.b_x, batch->n_tok));
        }
        /* The drafter's taps, once the layer's output is whole. Either way that is after the
         * scalar, since the junction applies it to the stream it writes. */
        for (int t = 0; t < m.n_taps && l.op_tap; ++t)
            if (m.tap_layer[t] == (int)li)
                RAD_ISSUE(c, l.op_tap, brows(m.b_x, T),
                          bcol(m.b_dfsrc, (int64_t)t * m.g.n_embd, m.g.n_embd, T));
    }

    RAD_ISSUE(c, m.op_final,
              brows(m.b_x, batch->n_tok), RAD_W(m.w_out), brows(m.b_h, batch->n_tok));

    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.b_h, batch->n_tok),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm),
                    brows(m.b_logits, batch->n_out));
        RAD_ISSUE_N(c, m.op_cap, batch->n_out,
                    brows(m.b_logits, batch->n_out),
                    brows(m.b_logits, batch->n_out));
    }
}

/* The reply format: the model's, shared with arch/gemma4 (arch/rad_gemma4_chat.h). */
const RadChatFormat* chat_format(const RadModelMeta*) { return gemma4_chat_format(); }

/* The default draft depth for `--num-speculative-tokens auto`. A serial head's depth is an
 * operating point, not a trained depth: every depth runs. A DFlash drafter's is the block it was
 * trained at, fixed. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    if (has_assistant(meta)) {
        out->draft_depth = kHeadDepth;
        out->draft_depth_fixed = 0;
    }
    /* A DFlash drafter names the block it was trained at, and only that (rad_gemma4_dflash.h); a
     * DFlash2 one too (rad_block_dflash2.h). */
    (void)gemma4::dflash_probe(meta, out);
    (void)gemma4::dflash2_probe(meta, out);
    /* The tower's pooling window, so a pass size the core picks itself is whole windows. */
    if (Gemma4Vision::present(meta)) {
        const int64_t k = rad_meta_geti(meta, "vision_config.pooling_kernel_size", 0);
        if (k > 0) out->enc_merge = k * k;
    }
    if (Gemma4Audio::present(meta)) out->audio_frames = Gemma4Audio::frames_for(meta);
    /* The 12B's embedders: a patch and an audio frame are a row each. */
    if (Gemma4UnifiedVision::present(meta)) out->enc_merge = 1;
    if (Gemma4UnifiedAudio::present(meta)) out->audio_frames = Gemma4UnifiedAudio::frames_for(meta);
    return RAD_OK;
}

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(gemma4_bf16, "gemma4", "compressed-tensors", "0.1.0",
                "Gemma 4 dense (text, images, E2B/E4B/12B audio), w4a8 or w8a8 from its bf16 or its CT-INT4 (QAT) "
                "release: sliding-window and proportional-RoPE global attention over a plain "
                "RMSNorm stream with GELU feed-forward, per-layer scalar scaling and soft-capped "
                "logits; Google's assistant head, z-lab's DFlash drafter or a DFlash2 drafter")
RAD_ARCH_CHAT_FORMAT(gemma4_bf16)
RAD_ARCH_PROBE(gemma4_bf16)
#endif

}  /* namespace gemma4_bf16 */
