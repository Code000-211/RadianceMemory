/* rad_block_attn_gated_fp8.h -- gated GQA attention with block-scaled FP8 projections.
 *
 * The architecture (interleaved q|gate projection, partial M-RoPE, separate k and v, sigmoid
 * gate) is documented in rad_block_attn_gated.h. What differs here:
 *
 *   - q|gate, k, v and o are LinearFP8: an E4M3 weight with a bf16 [N/128][K/128] scale plane,
 *     through `gemm_nt_q` at dtype fp8a8.
 *   - One input quantise feeds the three input projections (they all read the post-norm hidden
 *     state); the gate pass quantises the out-projection's input.
 *   - Per-head norms, rope, kv_store and attn_paged stay bf16.
 *   - The KV cache dtype is `g.kv_dtype`, not `g.dtype`: it is a deployment choice that selects
 *     libr4d's fp8kv attention family. The attention op and kv_store both take it, so they agree.
 *   - Optional fused prologue, QSA sparse attention and an attention-zero mode (see declare).
 *   - Optionally rotated (`rotate`, --kv-recipe `rotated`): q, k and v pass through the normalised
 *     Hadamard (`had_rotate`) before the store and the attention, and the attention output back
 *     through it before the gate, so the cache holds rotated K and V and every kernel that reads
 *     it is the unrotated one.
 *   - Optionally centred (`vmean_sub`/`vmean_add`, --kv-recipe `vmean`): a static mean of V is
 *     subtracted before the store and added to the attention output before its rotation back.
 *   - Optionally predicted (`vpred_map`, --kv-recipe `vpred`): V less an affine map of the K-hat
 *     the cache holds is stored, and a second attention with the keys as values gives P K-hat,
 *     whose map is added to the output before its rotation back.
 *   - An experiment, RADIANCE_DEBUG_KV_E4M3R (kv_e4m3r_debug below): on a bf16 cache the store
 *     rounds every value through E4M3 as the fp8 cache's store does (kv_store at kv_dtype
 *     bf16_e4m3r), and the bf16 attention reads it.
 *
 * The q|gate projection is [2*n_head*head_dim, n_embd] = [12288, 5120] at 27B, scale plane
 * [96, 40]. Row-sharding by head keeps whole 128-row scale blocks only because 2*head_dim = 512 is
 * a multiple of 128; LinearFP8 checks this.
 */
#ifndef RAD_BLOCK_ATTN_GATED_FP8_H
#define RAD_BLOCK_ATTN_GATED_FP8_H

#include "rad_fp8.h"
#include "rad_calib_vmean.h"
#include "rad_calib_vpred.h"

#include <cstdlib>
#include <cstring>

namespace rad {
namespace arch {

/* Whether the streamed store has a row for this cache layout: libr4d's are E4M3 and bf16 only. A
 * packed cache (fp8_v4, k6_v4) must not declare it, since a weightless op with no row is a hole
 * that refuses startup even with --kv-resident off; kvstream_configure then names the layout. */
inline bool kv_store_streamed_dtype(const char* kv_dtype) {
    return kv_dtype && (std::strcmp(kv_dtype, "fp8_e4m3") == 0 ||
                        std::strcmp(kv_dtype, "bf16") == 0);
}

/* Whether the layout has the fused forms' rows: the prologue's store fold
 * (qk_norm_rope_gate_kv_store) and the gate in the decode merge (attn_paged_gate_quant). k5_v4
 * (RAD_KVQ_K5V4) has neither, and a declared op no kernel serves is a hole that refuses startup,
 * so its layers declare neither and take kv_store and the separate gate, which give the same
 * values. Every other layout is as before. */
inline bool kv_fused_dtype(const char* kv_dtype) {
    return !kv_dtype || std::strcmp(kv_dtype, "k5_v4") != 0;
}

/* RADIANCE_DEBUG_KV_E4M3R set (and not "0"), with --kv-cache-dtype bf16: an experiment that
 * separates what the fp8 cache costs in storage rounding from what its attention path costs. Each
 * K and V value is stored as the fp8 store rounds it (k / k_scale clamped to +-448 and rounded to
 * E4M3, the descales the fp8 cache has here, 1.0), decoded and kept in the bf16 cache, so the bf16
 * attention reads exactly the values the fp8 attention would (E4M3 values are exact in bf16 and
 * in the f16 both kernels widen to). Its KL against a bf16 reference is the storage rounding
 * alone; fp8's KL above it is the fp8 path's. The fused prologue's store fold and the streamed
 * store have no such row, so the layer stores with kv_store (a streamed group fails the step).
 * Not a layout: nothing else reads the name, and an fp8 or packed cache ignores the switch. */
inline bool kv_e4m3r_debug() {
    /* Read at each declare (startup only), not cached, so a test can set it. */
    const char* e = std::getenv("RADIANCE_DEBUG_KV_E4M3R");
    return e && *e && std::strcmp(e, "0") != 0;
}

/* Layer `l`'s K-to-V maps for this rank (RadKvVPred, abi/rad_builder.h): this rank's KV heads'
 * [n_head_kv][head_dim + 1][head_dim] f32, a slice of the device table at the rank's first KV head
 * (Geom::kv_off on a spanned split, rank * kv_dim otherwise, as kv_vmean_slices). Null with a
 * message where the table is not this model's or names no entry for the layer. */
inline const void* kv_vpred_slice(const RadKvVPred* t, int64_t l, const Geom& g) {
    if (!t || !t->table || !t->entry || l < 0 || l >= t->n_layers || t->entry[l] < 0 ||
        t->head_dim != g.head_dim || t->n_head_kv != g.n_head_kv_all) {
        fprintf(stderr, "radiance: KV recipe: layer %lld predicts V from K and the engine's maps do "
                        "not fit this model (%lld KV heads of %lld) or name no map for it\n",
                (long long)l, (long long)g.n_head_kv_all, (long long)g.head_dim);
        return nullptr;
    }
    const bool spanned = g.world > g.n_head_kv_all || g.attn_world != g.world;
    const int64_t kv0 = (spanned ? g.kv_off : g.attn_rank * g.kv_dim()) / g.head_dim;
    const int64_t per = (g.head_dim + 1) * g.head_dim;
    return static_cast<const float*>(t->table) + (t->entry[l] * t->n_head_kv + kv0) * per;
}

/* Layer `l`'s V mean for this rank (RadKvVMean, abi/rad_builder.h), in the rotated basis when
 * `rotated`: `sub` is -mu over this rank's KV heads ([kv_dim]) and `add` +mu over its query heads
 * ([q_dim]), both slices of the one device table. A rank's KV heads start at its row span's
 * offset when it has one (replicated heads, a smaller attention world; Geom::kv_off) and at
 * rank * kv_dim on the plain split; its query heads at attn_rank * q_dim either way. Null with a
 * message where the table is not this model's. */
inline bool kv_vmean_slices(const RadKvVMean* t, int64_t l, bool rotated, const Geom& g,
                            const void** sub, const void** add) {
    *sub = *add = nullptr;
    if (!t || !t->table || l < 0 || l >= t->n_layers || t->head_dim != g.head_dim ||
        t->n_head_kv != g.n_head_kv_all || t->n_head != g.n_head_all ||
        std::strcmp(g.dtype, "bf16")) {
        fprintf(stderr, "radiance: KV recipe: layer %lld subtracts a V mean and the engine's "
                        "table does not fit this model (bf16 activations, %lld KV heads of %lld, "
                        "%lld query heads)\n", (long long)l, (long long)g.n_head_kv_all,
                (long long)g.head_dim, (long long)g.n_head_all);
        return false;
    }
    const bool spanned = g.world > g.n_head_kv_all || g.attn_world != g.world;
    const int64_t kv0 = spanned ? g.kv_off : g.attn_rank * g.kv_dim();
    const int64_t q0  = g.attn_rank * g.q_dim();
    const int64_t E   = (t->n_head_kv + t->n_head) * t->head_dim;
    const uint16_t* e = static_cast<const uint16_t*>(t->table) + (l * 2 + (rotated ? 1 : 0)) * E;
    *sub = e + kv0;
    *add = e + t->n_head_kv * t->head_dim + q0;
    return true;
}

struct AttnGatedFP8 {
    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]              the residual stream */
        ActFP8  h{};        /* [max_tok, n_embd]              normed input, then the o output */
        rad_buf qg   = 0;   /* [max_tok, n_head, 2*head_dim]  query and gate, interleaved */
        rad_buf k    = 0;   /* [max_tok, kv_dim] */
        rad_buf v    = 0;   /* [max_tok, kv_dim] */
        rad_buf q    = 0;   /* [max_tok, q_dim]   de-interleaved, normed, rotated */
        /* [rope_table_rows, rot] f32 cos/sin table, shared by every layer, filled once a step. Zero
         * if the plugin builds none; the fused prologue is then not requested. */
        rad_buf cos_sin = 0;
        ActFP8  attn{};     /* [max_tok, q_dim]   attention output, then gated in place */
        /* [max_tok, q_dim] de-interleaved sigmoid gate, used only with bf16 linears
         * (Geom::w_bf16), where the gate is `sigmoid` then `mul`; the fused gate quantiser always
         * writes codes. */
        rad_buf gate = 0;
    };

    struct Src {
        const char* norm   = nullptr;
        const char* qg     = nullptr;   /* self_attn.q_proj.weight -- query AND gate */
        const char* k      = nullptr;
        const char* v      = nullptr;
        const char* q_norm = nullptr;
        const char* k_norm = nullptr;
        const char* o      = nullptr;
    };

    Geom        g{};
    Wire        w{};
    int         layer = 0;
    rad_kvgroup kv    = 0;
    int64_t     n_rot = 0;
    /* The KV page in tokens; 0 means RAD_KV_BLOCK (16). Qwen4-Exp's QSA selects 4-token blocks,
     * and a page of 4 turns the selected set into a paged block table, so no new attention kernel
     * is needed (docs/QSA.md). The attention kernel's rows must carry a matching
     * C_EQ("block_size", ...) or nothing resolves. */
    int64_t     kv_block = 0;
    /* The QSA selection, when this layer has an indexer (produced by arch/common/rad_qsa.h; see
     * docs/QSA.md). On a sparse step the attention is issued with these in place of each row's
     * block table and context length: same op, same kernel, different list. `sel` is `topk + 1`
     * wide and the kernel takes its block-table pitch from the operand. */
    rad_buf     qsa_sel  = 0;
    rad_buf     qsa_nsel = 0;   /* [max_tok] i32, the valid entries of each `qsa_sel` row */
    rad_buf     qsa_sequ = 0;
    /* Context length up to which dense attention gives the same result, `budget + ratio - 1`.
     * Below it every complete block is selected, the paths agree to the bit, and dense is cheaper
     * at prefill (it reads each key once per query tile, sparse reads a table per query). Zero
     * means always sparse. */
    int64_t     qsa_exact_to = 0;
    /* `sel`'s row pitch is `topk + 1`; the extra slot is the partial page
     * (r4d_qsa_select_bf16.hip). */
    int64_t     qsa_topk = 0;
    /* Query rows per sparse attention launch, a scratch budget. The decode kernel's split-KV
     * partials are sized at declare from `max_seqs` and `q_len`, so issuing more rows than
     * `max_seqs` means declaring a larger value, which grows the scratch arena (the maximum over
     * ops) by `rows * q_len_band * n_head * (head_dim + 2) * 4` bytes: about 32 MiB at 256 rows on
     * this model, 63 MiB at 512, 253 MiB at 2048. With max_seqs alone, `--max-num-seqs 4` would
     * split a 2048-token prefill chunk into 512 launches per layer; 256 makes it 8, past which
     * launch cost no longer matters. */
    static constexpr int64_t kQsaRows = 256;
    int64_t qsa_rows() const { return g.max_seqs > kQsaRows ? g.max_seqs : kQsaRows; }
    /* The gated form takes the whole step in one launch: its band is q_len 1, so partials are only
     * `rows * n_head * (2 * head_dim + 8)` bytes (12.8 MiB for 2048 rows on this model). One
     * 2048-row launch is about 2.16 ms per layer versus 3.0 ms for eight 256-row launches, each
     * with its own tail. */
    int64_t qsa_gq_rows() const { return g.max_tok > qsa_rows() ? g.max_tok : qsa_rows(); }
    /* The rotation's pairing. Halves (NeoX, HF's rotate_half over the partial rotary width) and
     * adjacent pairs (GPT-J) are both self-consistent and pass every per-op check, so it must be
     * stated. "mrope" over one position component per token is NeoX (rad_block_attn_gated.h).
     *
     * The q|gate order is fixed as well: q is the first half of each head's pair, the gate the
     * second (as vLLM chunks q_gate). The unfused q norm reads column 0 of each head, the gate
     * quantiser column head_dim, and the fused prologue splits the same way. A swap would pass
     * silently, since q_norm and the sigmoid accept either half. */
    const char* mode  = "mrope";
    /* Store this layer's K and V rotated (RadBuildCtx::kv_recipe's RAD_KVQ_ROTATED), set before
     * declare. H / sqrt(head_dim) is its own inverse, so with q rotated too every score is the
     * unrotated one, and the attention output, a weighted sum of rotated v rows, is the rotated
     * output, which one more pass turns back before the gate. The scores and output change only
     * by the rotation's rounding (libref's `had_rotate`, ref_test); the cache layout, the store
     * and the attention kernels do not change at all. Rotation spreads an outlier channel across
     * the head, which is what lets a narrow integer layout hold K and V.
     *
     * The folds that put the store or the gate inside another kernel are not declared on such a
     * layer, since each would skip a rotation: the prologue with the store folded in (plain and
     * streamed), and the sparse attention with the gate in its merge. The two-piece prefill path
     * (takes_pieces) is not taken either. */
    bool        rotate = false;
    /* Subtract a static V mean (RadBuildCtx::kv_vmean, kv_vmean_slices), set before declare: -mu
     * over this rank's KV heads and +mu over its query heads, bf16 on the device, in the basis
     * the cache holds. One `add` with its row broadcast takes `sub` off V after the rotation and
     * before the store; one more puts `add` back on the attention output before the rotation back
     * (and the gate). A row of attention weights sums to one, so the output is the uncentred one
     * up to the two adds' rounding, while the integer V layouts quantise a centred V. The folds
     * that would store V or gate the output inside another kernel are not declared here either,
     * as for `rotate`. Both null: nothing is declared and the layer is what it was. */
    const void* vmean_sub = nullptr;
    const void* vmean_add = nullptr;
    /* The calibration tap that records V's mean (rad_calib_vmean.h), or null. */
    CalibVMean* vcal = nullptr;
    /* Predict V from K (RadBuildCtx::kv_vpred, kv_vpred_slice), set before declare: this rank's KV
     * heads' maps, f32 on the device, rotated (a vpred layer is a k4h layer, always rotated).
     * `kv_vpred` (docs/OPS.md) takes K-hat A + b off V after the rotation and before the store,
     * K-hat being K through the k4h store's rounding, the K the cache holds; a second attention
     * over the cache at kv_dtype `<layout>_pk` (op_attn_pk) reads the keys as values into `pk`,
     * P K-hat; and kv_vpred adds (P K-hat) A + b to the output before the rotation back. A row of
     * attention weights sums to one, so the output is the unpredicted one up to roundings, while
     * the integer V layout quantises the residual. Not with vmean_sub (the bias is the mean's
     * place). Null: nothing is declared and the layer is what it was. */
    const void* vpred_map = nullptr;
    /* The calibration tap that fits the maps (rad_calib_vpred.h), or null. */
    CalibVPred* pcal = nullptr;

    rad_weight w_norm = 0, w_q_norm = 0, w_k_norm = 0;
    /* The input was normed and quantised by the caller; this block declares no norm. */
    bool       ext_in = false;
    LinearFP8  qg{}, kp{}, vp{}, o{};
    NormQuantFP8 nq_h{};

    rad_op     op_fused = 0;                                 /* qk_norm_rope_gate, if it resolves */
    rad_op     op_fused_kv = 0;                              /* the same with kv_store folded in */
    /* The same with the streamed store folded in, issued instead when the batch hands this group a
     * stream (its mirror and map are the batch's). */
    rad_op     op_fused_stream = 0;
    int64_t    fused_rows = 0;                               /* widest T the fused ops take */
    rad_op     op_q_norm = 0, op_k_norm = 0, op_rope_q = 0, op_rope_k = 0;
    rad_op     op_attn = 0, op_kv_store = 0, op_gq = 0;
    rad_op     op_attn_gq = 0;                               /* attn_paged + gate_quant_fp8 */
    /* had_rotate over head_dim, in place: q, k, v before the store, the output after the attention.
     * Declared only with `rotate`. */
    rad_op     op_rot = 0;
    /* `add` over V's rows and over the output's, with the mean's row broadcast; declared only with
     * vmean_sub / vmean_add. */
    rad_op     op_vsub = 0, op_vadd = 0;
    /* kv_vpred over V's rows (k4h K-hat, minus) and over the output's (P K-hat, plus), and the
     * P K-hat attention into `pk` ([max_tok, q_dim]); declared only with vpred_map. kv_pk_dtype
     * holds the attention's kv_dtype string, `<layout>_pk`, for the op's parameter. */
    rad_op      op_vpsub = 0, op_vpadd = 0, op_attn_pk = 0;
    rad_buf     pk = 0;
    std::string kv_pk_dtype;
    /* kv_store_streamed: the host mirror always, the ring slot when `res` says the block is resident
     * (docs/OPS.md). Zero when no row serves it; a streamed group then fails the step. */
    rad_op     op_kv_stream = 0;
    /* kv_resolve: this layer's selection made resident and rewritten as ring slots, issued before
     * each sparse launch. */
    rad_op     op_resolve = 0;
    /* The output gate as `sigmoid` then `mul`: the bf16-linear form (Geom::w_bf16), and with fp8
     * linears the chain gate_quant_fp8 fuses, where it is not served. `gate` is the sigmoid's
     * output (the wire's with bf16 linears, the block's own otherwise) and `q_gate` the chain's
     * quantiser (no op with bf16 linears). */
    rad_op     op_sig = 0, op_mul = 0;
    rad_buf    gate = 0;
    QuantFP8   q_gate{};
    rad_op     op_ar = 0, op_add = 0;
    rad_op     op_fill = 0;                                  /* the attention-zero rank: see declare */
    /* The next block absorbs this block's closing all-reduce, so this one must not issue it.
     * Counterpart of `ar_in` (this block's norm absorbs the predecessor's); separate because a
     * block is declared without knowing what follows. Both sides ask ar_fused_ok(g, T), so they
     * agree on whether the collective was issued. */
    bool ar_out = false;
    /* Which wires that consumer takes (ArOut, rad_fp8.h): the norm's by default, or a gated
     * residual's own (HyperConn::ar_take). Set before declare; both sides then ask
     * ar_taken(g, T, ar_out, ar_out_take). */
    int  ar_out_take = kArOutNorm;

    /* `fold`: the previous block left its delta in `w.h.x` for this norm to add. `add_out`: the
     * next block will not add, so this one issues the trailing add. See rad_fp8.h. */
    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                 int64_t rotary_dim, const Wire& wire, const Src& src,
                 bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false);
    /* Stores the pass's K and V exactly as step() does, with no query, attention or output. Used
     * by passes that only leave layer state behind, such as the MTP head's history pass over a
     * prompt chunk with no drafting row. */
    void store_kv(RadCtx* c, const RadBatch* batch) const;
    /* The layer's cache store, before its resolve and attention: `kv_store`, or for a streamed
     * group the stage wait and `kv_store_streamed`. */
    void store_cache(RadCtx* c, const RadKVGroupBatch* kvb, const int32_t* slot, int64_t T) const;
    /* A staged pass reads the pool at block-id identity, while the streamed store writes only
     * where the residency map names a ring slot (none before the group's first decode). This
     * stores the same bytes at identity, after the mirror writer. */
    void store_stage_identity(RadCtx* c, const int32_t* slot, int64_t T) const;

    /* V's rows [r0, r0 + n) less the layer's mean (op_vsub), after any rotation and before the
     * store; nothing on an uncentred layer. */
    void centre_v(RadCtx* c, int64_t r0, int64_t n) const {
        if (op_vsub)
            RAD_ISSUE_N(c, op_vsub, n, brow_slice(w.v, r0, n, g.kv_dim()),
                        praw(vmean_sub, RAD_BF16, g.kv_dim()), brow_slice(w.v, r0, n, g.kv_dim()));
    }
    /* V's rows [0, T) less their prediction from the K-hat of the same rows (op_vpsub), after the
     * rotation and before the store; nothing on a layer without vpred. */
    void predict_v(RadCtx* c, int64_t T) const {
        if (op_vpsub)
            RAD_ISSUE_N(c, op_vpsub, T, brows(w.k, T),
                        praw(vpred_map, RAD_F32, g.n_head_kv * (g.head_dim + 1) * g.head_dim),
                        brows(w.v, T), brows(w.v, T));
    }

    /* The block's tail, declared last by every rank: the all-reduce and the trailing add. */
    int  declare_out(RadBuilder* b, bool add_out);
    /* `own_ar` false: the caller issues the all-reduce itself (GdnFP8::step says when). */
    void step(RadCtx* c, const RadBatch* batch, bool own_ar = true,
              ArPieces* pieces = nullptr) const;
    bool issues_ar(int64_t T) const { return op_ar && !ar_taken(g, T, ar_out, ar_out_take); }
    void issue_ar(RadCtx* c, int64_t r0, int64_t rows) const {
        RAD_ISSUE_N(c, op_ar_ov ? op_ar_ov : op_ar, rows * g.n_embd,
                    brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE);
    }
    /* Set before declare: the `overlap` issue_ar's collective is declared at, 0 for none. */
    int    ar_overlap = 0;
    rad_op op_ar_ov = 0;

    /* Whether step() with `own_ar` false runs the dense path a piece at a time over a caller's
     * normed pieces (ArPieces::in_ready): one prefill sequence in two pieces, each the chunk a
     * span of 1 makes, on the unfused prologue with one rotary component per token (every token's
     * components agree, so plane 0 of an M-RoPE batch's positions is each token's position) and
     * the gate quantised by gate_quant_fp8. A streamed group never takes it: the pieces path's
     * plain store never writes the host mirror. */
    bool takes_pieces(const RadBatch* batch, const ArPieces* pieces) const {
        const int64_t T = batch->n_tok;
        const RadKVGroupBatch* kvb = kv_batch(batch, kv);
        return pieces && pieces->in_ready && pieces->cu && pieces->cuts(T) &&
               T <= 2 * pieces->rows && !op_add && !op_fill && op_gq && !op_sig && !qsa_sel &&
               !op_rot && !op_vsub && !op_vpsub &&
               batch->n_seq == 1 && batch->phase == RAD_PHASE_PREFILL &&
               !(T <= fused_rows) && batch->positions && !rope_mixed(batch) &&
               kvb && !kvb->stream && kvb->slot_mapping && kvb->block_table && kvb->seqused;
    }
};

inline int AttnGatedFP8::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                                 rad_kvgroup group, int64_t rotary_dim, const Wire& wire,
                                 const Src& src, bool fold, bool add_out,
                           int ar_in, bool ar_out_) {
    g = geom; w = wire; layer = l;
    ar_out = ar_out_; kv = group; n_rot = rotary_dim;

    const int64_t q_dim  = g.q_dim();
    const int64_t kv_dim = g.kv_dim();
    const int64_t qg_dim = 2 * q_dim;

    /* Row spans, in the container's unsharded rows, used where the even shard is the wrong cut.
     *
     * Above the KV head count geom_from replicates KV heads (rad_arch.h): with 2 KV heads at
     * head_dim 256 and 4 ranks, the k/v entry is 512 rows and ranks 0-1 take [0, 256), ranks 2-3
     * [256, 512), where the even shard would hand rank 2 half a head. The scale plane takes the
     * same span in 128-blocks; kv_off is a multiple of head_dim, so it never cuts a block. At
     * world <= n_head_kv_all the span stays 0/0 and the even shard applies.
     *
     * Under a smaller attention world (Geom::attn_world) q|gate and the output also take this
     * rank's heads [attn_rank * n_head, +n_head) as a span. An attention-zero rank declares none
     * of them (see the `fill` below). */
    const bool    a_spanned  = g.attn_world != g.world;
    const bool    kv_spanned = g.world > g.n_head_kv_all || a_spanned;
    const int64_t kv_span_lo = kv_spanned ? g.kv_off : 0;
    const int64_t kv_span_hi = kv_spanned ? g.kv_off + kv_dim : 0;
    const int64_t qg_span_lo = a_spanned ? g.attn_rank * qg_dim : 0;
    const int64_t qg_span_hi = a_spanned ? (g.attn_rank + 1) * qg_dim : 0;
    const int64_t o_span_lo  = a_spanned ? g.attn_rank * q_dim : 0;
    const int64_t o_span_hi  = a_spanned ? (g.attn_rank + 1) * q_dim : 0;

    if (n_rot <= 0 || n_rot > g.head_dim || (n_rot & 1)) {
        fprintf(stderr, "radiance: rotary_dim %lld is not an even width inside head_dim %lld\n",
                (long long)n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    /* `src.norm == nullptr`: the architecture prepares the input. A pre-norm block owns its norm
     * (rad_fp8.h's NormQuantFP8: add the previous delta, normalise, quantise). In a gated-residual
     * model, rad_block_hc.h's `hc_read` produces the block input from a 4x wider stream, so this
     * block takes `wire.h` as already normed and quantised and declares no norm weight.
     *
     * `ar_in` is refused in that mode: the fused all-reduce lives inside the norm, so it would be
     * silently dropped. The caller emits a standalone collective instead (`add_out`/`ar_out`). */
    ext_in = (src.norm == nullptr);
    if (ext_in) {
        if (!w.h.x || (!g.w_bf16 && (!w.h.q || !w.h.s)) || ar_in != kArNone) return RAD_E_INVAL;
        if (g.w_bf16 && !w.gate) return RAD_E_INVAL;
    } else {
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l), src.norm));
        w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_F32, {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    }
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l), src.q_norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", l), src.k_norm));
    w_q_norm = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", l), RAD_F32, {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_k_norm = decl_w(b, nm.f("blk.%d.attn_k_norm.weight", l), RAD_F32, {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* One norm+quantise op feeds all three input projections (rad_fp8.h: NormQuantFP8). */
    if (!ext_in)
        RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                                  fold ? w.x : 0, 0, ar_in));

    /* An attention-zero rank holds no heads, so its output projection contributes zero to the
     * reduce. Its output wire is zeroed by one `fill` instead of running projections on zero
     * weights (and it keeps no KV pool, rad_kv_group_zero). The all-reduce and residual add run as
     * usual, and the norm above stays because a pre-norm block's `fold` and `ar_in` live in it. */
    if (g.attn_zero) {
        op_fill = rw(b, RAD_OP(b, "fill",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {}, {w.h.x});
        return declare_out(b, add_out);
    }
    RAD_ARCH_TRY(qg.declare(b, nm, g, nm.f("blk.%d.attn_qg", l), qg_dim, g.n_embd,
                            RAD_SHARD_ROW, grp_layer(l), { src.qg }, w.h, w.qg,
                            0, {}, qg_span_lo, qg_span_hi));
    RAD_ARCH_TRY(kp.declare(b, nm, g, nm.f("blk.%d.attn_k", l), kv_dim, g.n_embd,
                            RAD_SHARD_ROW, grp_layer(l), { src.k }, w.h, w.k,
                            0, {}, kv_span_lo, kv_span_hi));
    RAD_ARCH_TRY(vp.declare(b, nm, g, nm.f("blk.%d.attn_v", l), kv_dim, g.n_embd,
                            RAD_SHARD_ROW, grp_layer(l), { src.v }, w.h, w.v,
                            0, {}, kv_span_lo, kv_span_hi));

    /* The fused prologue (qk norms + rope + q|gate split) is a probe; the four unfused ops below
     * are declared either way and run when it does not resolve. Its schema takes a precomputed
     * cos/sin table instead of `theta`: `rope_table` fills `w.cos_sin` once a step for every
     * layer, and without a table the fusion is not requested.
     *
     * The kernel rotates NeoX pairs and has no `mode`. M-RoPE qualifies because the table is
     * filled at one component per token, which is exact when the components agree; step() takes
     * the fused path only then.
     *
     * No `gate_out` is passed: `gate_quant_fp8` reads the gate in place in the interleaved
     * projection.
     *
     * Bit-identical to the unfused ops, asserted in r4d_selftest: the kernel rounds the norm output
     * to bf16 before rotating and reproduces `r4d_rmsnorm_bf16`'s sum-of-squares order element by
     * element. A ULP-level difference would shift the engine's transcript within a few dozen
     * steps. */
    const bool rope_neox = !std::strcmp(mode, "neox") || !std::strcmp(mode, "mrope") || g.rope_mc;
    fused_rows = qk_fuse_rows(g);
    if (w.cos_sin && rope_neox && fused_rows > 0 && rad_op_declared(b, "qk_norm_rope_gate"))
        op_fused = rw(b, RAD_OP(b, "qk_norm_rope_gate",
                         RAD_PARAMS(RAD_RANGE("M", 1, fused_rows), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head", g.n_head),
                                    RAD_INT("n_head_kv", g.n_head_kv), RAD_INT("rot", n_rot),
                                    RAD_STR("q_dtype", g.dtype), RAD_F64("eps", g.eps),
                                    RAD_F64("wadd", g.wadd_qk), RAD_INT("cs_f32", 1)),
                         RAD_WEIGHTS(w_q_norm, w_k_norm)),
                     {w.qg, w.k, w.cos_sin}, {w.q, w.k});

    op_q_norm = rw(b, RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                  RAD_INT("n", g.head_dim),
                                  RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                       RAD_WEIGHTS(w_q_norm)),
                   {w.qg}, {w.q});
    op_k_norm = rw(b, RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head_kv),
                                  RAD_INT("n", g.head_dim),
                                  RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                       RAD_WEIGHTS(w_k_norm)),
                   {w.k}, {w.k});
    op_rope_q = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head, 0, n_rot, mode),
                   {w.q}, {w.q});
    op_rope_k = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head_kv, 0, n_rot, mode),
                   {w.k}, {w.k});

    /* The rotation of a rotated layer: one op over head_dim-wide rows, issued over q, k and v (as
     * T * n_head or T * n_head_kv rows) and over the attention output. It names all four buffers,
     * declared here before the attention so each one's lifetime covers the issues (q's and k's
     * from the prologue, v's from its projection, the output's to the gate). No fold or fallback:
     * a rotated layer without a kernel for it refuses, naming the layer, rather than run
     * unrotated over a cache a recipe said is rotated. */
    if (rotate) {
        op_rot = rw(b, RAD_OP(b, "had_rotate",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                   RAD_INT("n", g.head_dim), RAD_INT("group", g.head_dim),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.q, w.k, w.v, w.attn.x}, {w.q, w.k, w.v, w.attn.x});
        if (!op_rot) {
            fprintf(stderr, "radiance: KV recipe: layer %d is rotated and no kernel serves "
                            "had_rotate at head_dim %lld, %s\n", l, (long long)g.head_dim,
                    g.dtype);
            return RAD_E_UNSUPPORTED;
        }
    }
    /* A centred layer's two adds, declared here for the same reason: V's from its projection to
     * the store, the output's from the attention to the gate. Refused rather than run uncentred:
     * the recipe said what the cache holds. */
    if (vmean_sub || vmean_add) {
        auto add = [&](int64_t n, rad_buf x) {
            return rw(b, RAD_OP(b, "add",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", n),
                                     RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {x}, {x});
        };
        op_vsub = vmean_sub ? add(kv_dim, w.v) : 0;
        op_vadd = vmean_add ? add(q_dim, w.attn.x) : 0;
        if (!op_vsub || !op_vadd) {
            fprintf(stderr, "radiance: KV recipe: layer %d subtracts a V mean and no kernel serves "
                            "add over rows of %lld and %lld at %s\n", l, (long long)kv_dim,
                    (long long)q_dim, g.dtype);
            return RAD_E_UNSUPPORTED;
        }
    }
    /* A predicted layer's two maps, declared here for the same reason, and refused rather than
     * run unpredicted: the recipe said the cache holds residuals. Its P K-hat attention follows
     * op_attn below. */
    if (vpred_map) {
        if (!rotate || vmean_sub || vmean_add) {
            fprintf(stderr, "radiance: KV recipe: layer %d predicts V from K, which takes a "
                            "rotated k4h layer without a V mean\n", l);
            return RAD_E_UNSUPPORTED;
        }
        auto vp_op = [&](int64_t heads, int64_t gqa, const char* kq, int64_t sign, rad_buf x,
                         rad_buf y) {
            return rw(b, RAD_OP(b, "kv_vpred",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                     RAD_INT("n_head", heads), RAD_INT("gqa", gqa),
                                     RAD_STR("k_quant", kq), RAD_INT("sign", sign),
                                     RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {x, y}, {y});
        };
        pk = decl_b(b, nm.f("blk.%d.attn_pk", l), g.act_dtype, {g.max_tok, q_dim});
        op_vpsub = pk ? vp_op(g.n_head_kv, 1, "k4h", -1, w.k, w.v) : 0;
        op_vpadd = pk ? vp_op(g.n_head, g.gqa(), "none", 1, pk, w.attn.x) : 0;
        if (!op_vpsub || !op_vpadd) {
            fprintf(stderr, "radiance: KV recipe: layer %d predicts V from K and no kernel serves "
                            "kv_vpred at head_dim %lld, %s\n", l, (long long)g.head_dim, g.dtype);
            return RAD_E_UNSUPPORTED;
        }
    }

    /* Declared before kv_store, which reads back the resolved `block_size`
     * (rad_block_attn_gated.h). */
    op_attn = decl_attn(b, g, "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                RAD_INT("window", 0),
                                RAD_INT("n_head", g.n_head),
                                RAD_INT("block_size", kv_block > 0 ? kv_block : (int64_t)RAD_KV_BLOCK),
                                /* The sparse path issues one row per query, so the split-KV
                                 * scratch is sized for the widest launch (kQsaRows). */
                                RAD_INT("max_seqs", (qsa_sel && qsa_sequ) ? qsa_rows()
                                                                          : g.max_seqs),
                                RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS);
    /* The QSA selection buffers must be declared as reads of this op. Otherwise their last use is
     * `qsa_select`, the arena may overlay another buffer, and the attention reads garbage page
     * ids: an aperture violation outside the cache. rad_op_reads rather than rw() because the list
     * depends on whether the layer has an indexer. */
    if (op_attn) {
        const rad_buf rd[4] = { w.q, qsa_sel, qsa_sequ, qsa_nsel };
        rad_op_reads(b, op_attn, rd,
                     (qsa_sel && qsa_sequ) ? (qsa_nsel ? 4 : 3) : 1);
        rad_op_writes(b, op_attn, &w.attn.x, 1);
    }
    /* A predicted layer's P K-hat: the same attention over the same cache and selection, at the
     * layout's `_pk` kv_dtype (the keys read as values), into `pk`. Its scores and weights are
     * op_attn's, so P K-hat pairs with P (V - K-hat A - b) row for row. */
    if (vpred_map) {
        kv_pk_dtype = std::string(g.kv_dtype ? g.kv_dtype : "") + "_pk";
        op_attn_pk = decl_attn(b, g, "attn_paged",
                         RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                    RAD_INT("window", 0),
                                    RAD_INT("n_head", g.n_head),
                                    RAD_INT("block_size", kv_block > 0 ? kv_block : (int64_t)RAD_KV_BLOCK),
                                    RAD_INT("max_seqs", (qsa_sel && qsa_sequ) ? qsa_rows()
                                                                              : g.max_seqs),
                                    RAD_INT("max_ctx", g.max_ctx),
                                    RAD_STR("q_dtype", g.dtype),
                                    RAD_STR("kv_dtype", kv_pk_dtype.c_str())),
                         RAD_NOWEIGHTS);
        if (!op_attn_pk) {
            fprintf(stderr, "radiance: KV recipe: layer %d predicts V from K and no kernel serves "
                            "attn_paged at kv_dtype %s\n", l, kv_pk_dtype.c_str());
            return RAD_E_UNSUPPORTED;
        }
        const rad_buf rd[4] = { w.q, qsa_sel, qsa_sequ, qsa_nsel };
        rad_op_reads(b, op_attn_pk, rd, (qsa_sel && qsa_sequ) ? (qsa_nsel ? 4 : 3) : 1);
        rad_op_writes(b, op_attn_pk, &pk, 1);
    }

    /* The resolve belongs here rather than in the indexer: it must follow this layer's store,
     * since a block copied in from the mirror has to carry this step's tokens. Its band is the
     * step's widest launch and its row `qsa_topk + 1` wide, as the attention declares. */
    if (qsa_sel && qsa_nsel)
        op_resolve = kv_resolve_declare(b, g, kv, qsa_sel, qsa_nsel, qsa_topk + 1);

    /* The experiment (kv_e4m3r_debug): the store's name for a bf16 cache rounded through E4M3. */
    const bool e4m3r = kv_e4m3r_debug() && g.kv_dtype && !std::strcmp(g.kv_dtype, "bf16");
    if (e4m3r) {
        static const bool said = (fprintf(stderr, "radiance: RADIANCE_DEBUG_KV_E4M3R: the bf16 "
                                                  "cache stores E4M3-rounded values (an "
                                                  "experiment, not a deployment)\n"), true);
        (void)said;
    }
    op_kv_store = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", e4m3r ? "bf16_e4m3r" : g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     {w.k, w.v}, {});

    /* The streamed store: kv_store's operands and parameters plus the residency map and the layer's
     * mirror row, which the batch carries (store_cache). `block_size` does not bind the group;
     * `attn_paged` already has. Only for a layout with a row (kv_store_streamed_dtype). */
    if (!e4m3r && kv_store_streamed_dtype(g.kv_dtype) && rad_op_declared(b, "kv_store_streamed"))
        op_kv_stream = rw(b, RAD_OP(b, "kv_store_streamed",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                        RAD_INT("head_dim", g.head_dim),
                                        RAD_INT("n_head_kv", g.n_head_kv),
                                        RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                        RAD_STR("kv_dtype", g.kv_dtype)),
                             RAD_NOWEIGHTS),
                         {w.k, w.v}, {});

    /* The fused prologue with the cache store folded in: a k head's block already holds its
     * token's rotated k, so it also writes k and v into the cache, byte-identical to kv_store.
     * Declared here because the cache block size is read back from the attention op above.
     * Resolves for E4M3 and bf16 caches. Not on a rotated layer, whose store follows a rotation. */
    if (op_fused && !rotate && !op_vsub && !op_vpsub && !e4m3r && kv_fused_dtype(g.kv_dtype) &&
        rad_op_declared(b, "qk_norm_rope_gate_kv_store"))
        op_fused_kv = rw(b, RAD_OP(b, "qk_norm_rope_gate_kv_store",
                            RAD_PARAMS(RAD_RANGE("M", 1, fused_rows),
                                       RAD_INT("head_dim", g.head_dim), RAD_INT("n_head", g.n_head),
                                       RAD_INT("n_head_kv", g.n_head_kv), RAD_INT("rot", n_rot),
                                       RAD_STR("q_dtype", g.dtype), RAD_F64("eps", g.eps),
                                       RAD_F64("wadd", g.wadd_qk), RAD_INT("cs_f32", 1),
                                       RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                       RAD_STR("kv_dtype", g.kv_dtype)),
                            RAD_WEIGHTS(w_q_norm, w_k_norm)),
                         {w.qg, w.k, w.cos_sin, w.v}, {w.q, w.k});

    /* The fold for a streamed group: also writes the mirror and consults the residency map, whose
     * operands are added at issue. Gated on the same layouts. */
    if (op_fused && !rotate && !op_vsub && !op_vpsub && !e4m3r && kv_store_streamed_dtype(g.kv_dtype) &&
        rad_op_declared(b, "qk_norm_rope_gate_kv_store_streamed"))
        op_fused_stream = rw(b, RAD_OP(b, "qk_norm_rope_gate_kv_store_streamed",
                            RAD_PARAMS(RAD_RANGE("M", 1, fused_rows),
                                       RAD_INT("head_dim", g.head_dim), RAD_INT("n_head", g.n_head),
                                       RAD_INT("n_head_kv", g.n_head_kv), RAD_INT("rot", n_rot),
                                       RAD_STR("q_dtype", g.dtype), RAD_F64("eps", g.eps),
                                       RAD_F64("wadd", g.wadd_qk), RAD_INT("cs_f32", 1),
                                       RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                       RAD_STR("kv_dtype", g.kv_dtype)),
                            RAD_WEIGHTS(w_q_norm, w_k_norm)),
                         {w.qg, w.k, w.cos_sin, w.v}, {w.q, w.k});

    /* Output gate, multiply and quantiser in one pass (`gate_quant_fp8`); at decode the dispatch
     * gaps between three small ops are comparable to the work. Each row is a (token, head) pair:
     * the gate is a fixed stride inside [q|gate], the attention output is contiguous, and the
     * [T * n_head, head_dim / 128] scale plane is byte-identical to the [T, q_dim / 128] the out
     * projection reads.
     *
     * The bf16 product is still written to `w.attn.x` and the codes are taken over it
     * (r4d_fused_quant_fp8.hip says why). The codes operand's dtype picks the quantiser, int8 when
     * the out projection is fed int8 (ActFP8::q8_fed). Serves prefill and decode: there is no
     * cross-row reduction.
     *
     * With bf16 linears (Geom::w_bf16) the gate is the plain `sigmoid` + `mul` pair instead.
     *
     * Where no loaded library declares gate_quant_fp8, or it does not resolve, the fp8 block runs
     * the chain it fuses: the gate into a buffer of its own, the multiply, then the activation
     * quantiser, which write what the fused op writes. A head of whole 128-column blocks makes
     * the quantiser's [T, q_dim / 128] scale plane the fused op's [T * n_head, head_dim / 128]. */
    auto gate_pair = [&](rad_buf out) {
        gate = out;
        op_sig = rw(b, RAD_OP(b, "sigmoid",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                   RAD_INT("n", g.head_dim), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.qg}, {gate});
        op_mul = rw(b, RAD_OP(b, "mul",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.attn.x, gate}, {w.attn.x});
        return op_sig && op_mul ? RAD_OK : RAD_E_INVAL;
    };
    if (g.w_bf16) {
        RAD_ARCH_TRY(gate_pair(w.gate));
    } else {
        /* Rotated int8 codes (ActFP8::q8_rot): gate_had_quant_i8g, had_quant_act_i8g's codes. */
        const char* gq = w.attn.q8_rot ? "gate_had_quant_i8g" : "gate_quant_fp8";
        if (rad_op_declared(b, gq))
            op_gq = rw(b, RAD_OP(b, gq,
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                      RAD_INT("n", g.head_dim), RAD_INT("group", RAD_FP8_BLOCK),
                                      RAD_STR("dtype", g.dtype), RAD_STR("act", "sigmoid")),
                           RAD_NOWEIGHTS),
                       {w.qg, w.attn.x}, {w.attn.cq(), w.attn.cs(), w.attn.x});
        if (!op_gq && g.head_dim % RAD_FP8_BLOCK == 0) {
            RAD_ARCH_TRY(gate_pair(decl_b(b, nm.f("blk.%d.attn_gate", l), g.act_dtype,
                                          {g.max_tok, q_dim})));
            RAD_ARCH_TRY(q_gate.declare(b, g, w.attn, q_dim));
            if (!q_gate.op) return RAD_E_INVAL;
        }
    }

    /* On the sparse path the gate is applied inside the attention's split-KV merge: each launch
     * row is one (token, head) at q_len 1, and the merge already holds the head_dim values the gate
     * and quantiser read, so one op merges, gates, rounds and quantises (byte-identical to the
     * pair). Declared only with a selection: the dense path also uses the prefill kernel, which
     * has no merge. Not on a rotated layer, whose output is rotated back before the gate. */
    if (op_attn && qsa_sel && qsa_sequ && !g.w_bf16 && !w.attn.q8_rot && !rotate && !op_vsub &&
        !op_vpsub && kv_fused_dtype(g.kv_dtype) && rad_op_declared(b, "attn_paged_gate_quant")) {
        op_attn_gq = RAD_OP(b, "attn_paged_gate_quant",
                            RAD_PARAMS(RAD_INT("q_len", 1), RAD_INT("head_dim", g.head_dim),
                                       RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                       RAD_INT("window", 0), RAD_INT("n_head", g.n_head),
                                       RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                       RAD_INT("max_seqs", qsa_gq_rows()),
                                       RAD_INT("max_ctx", g.max_ctx),
                                       RAD_STR("q_dtype", g.dtype),
                                       RAD_STR("kv_dtype", g.kv_dtype),
                                       RAD_INT("group", RAD_FP8_BLOCK),
                                       RAD_STR("act", "sigmoid")),
                            RAD_NOWEIGHTS);
        if (op_attn_gq) {
            const rad_buf rd[4] = { w.q, qsa_sel, qsa_sequ, w.qg };
            const rad_buf wr[3] = { w.attn.x, w.attn.cq(), w.attn.cs() };
            RAD_ARCH_TRY(rad_op_reads (b, op_attn_gq, rd, 4));
            RAD_ARCH_TRY(rad_op_writes(b, op_attn_gq, wr, 3));
        }
    }

    RAD_ARCH_TRY(o.declare(b, nm, g, nm.f("blk.%d.attn_output", l), g.n_embd, q_dim,
                           RAD_SHARD_COL, grp_layer(l), { src.o }, w.attn, w.h.x,
                           0, {}, o_span_lo, o_span_hi));
    return declare_out(b, add_out);
}

inline int AttnGatedFP8::declare_out(RadBuilder* b, bool add_out) {
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h.x}, {w.h.x});
    /* The same collective for a caller that runs it beside other work (issue_ar, a pipelined
     * prefill): at `overlap` 1 it gets the rows that hold fewer CUs while waiting on the link. */
    if (g.world > 1 && ar_overlap > 0)
        op_ar_ov = rw(b, RAD_OP(b, "all_reduce",
                          RAD_PARAMS(RAD_INT("world_size", g.world),
                                     RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                     RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                     RAD_INT("min_bytes", g.wire_min_bytes),
                                     RAD_INT("overlap", ar_overlap)),
                          RAD_NOWEIGHTS),
                      {w.h.x}, {w.h.x});

    /* Only when the next block's norm will not add (rad_fp8.h: NormQuantFP8). */
    if (add_out)
        op_add = rw(b, RAD_OP(b, "add",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.x, w.h.x}, {w.x});

    return RAD_OK;
}

inline void AttnGatedFP8::store_stage_identity(RadCtx* c, const int32_t* slot, int64_t T) const {
    RAD_ISSUE_N(c, op_kv_store, T, brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T),
                kv_cache(kv, layer));
}

inline void AttnGatedFP8::store_cache(RadCtx* c, const RadKVGroupBatch* kvb,
                                      const int32_t* slot, int64_t T) const {
    if (!kvb || !kvb->stream) {
        RAD_ISSUE(c, op_kv_store,
                  brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));
        return;
    }
    /* A streamed group writes its host mirror, and the ring slot only where the block is resident
     * (docs/OPS.md "kv_store_streamed"). kv_store alone would treat a block id as a ring slot, so
     * a build without the streamed store fails the step. */
    const bool staged = kv_stage_wait(c, kv, kvb, layer);
    const KVStreamLayer L = kv_stream_layer(c, kv, layer, kvb);
    if (!op_kv_stream || !L.host || L.blocks <= 0) {
        rad_step_fail(c, "kv streaming: this layer's KV group streams but the build has no "
                         "kv_store_streamed row to write its host mirror with");
        return;
    }
    RAD_ISSUE(c, op_kv_stream,
              brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer),
              praw(L.res, RAD_I32, L.blocks), praw2(L.host, RAD_U8, L.blocks, L.bbytes));
    if (staged) store_stage_identity(c, slot, T);
}

inline void AttnGatedFP8::store_kv(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (!ext_in) nq_h.step(c, T);
    if (op_fill) return;                /* the attention-zero rank holds no heads to store */
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot = kvb ? kvb->slot_mapping : nullptr;
    kp.step(c, w.h, w.k, T);
    vp.step(c, w.h, w.v, T);
    RAD_ISSUE_N(c, op_k_norm, T * g.n_head_kv, brows(w.k, T), RAD_W(w_k_norm), brows(w.k, T));
    RAD_ISSUE(c, op_rope_k, brows(w.k, T), rope_posmc(g, batch, T));
    if (op_rot) {
        RAD_ISSUE_N(c, op_rot, T * g.n_head_kv, brows(w.k, T));
        RAD_ISSUE_N(c, op_rot, T * g.n_head_kv, brows(w.v, T));
    }
    centre_v(c, 0, T);
    predict_v(c, T);
    store_cache(c, kvb, slot, T);
}

inline void AttnGatedFP8::step(RadCtx* c, const RadBatch* batch, bool own_ar,
                               ArPieces* pieces) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    const int64_t T = batch->n_tok;

    /* One prefill sequence in two pieces over the caller's normed input (takes_pieces): the
     * projections, prologue, cache store, attention, gate and out projection of the first piece,
     * then the second's, so the first piece's reduction runs beside all of the second piece's
     * work. Each piece is the chunk a span of 1 makes: its rows, its positions, its slots, and an
     * attention over its own query bounds (ArPieces::cu) that sees the cache up to its own end --
     * the first piece's position after its last row, the step's seqused for the second, which
     * reads the first piece's keys from the cache as a later chunk would. */
    if (!own_ar && takes_pieces(batch, pieces)) {
        const int64_t R = pieces->rows, hd = g.head_dim, qw = g.q_dim();
        const int64_t qgw = g.n_head * 2 * hd, kvw = g.n_head_kv * hd;
        const int64_t tpitch = kvb->block_table_pitch;
        for (int64_t p = 0; p < 2; ++p) {
            const int64_t r0 = p == 0 ? 0 : R;
            const int64_t n  = p == 0 ? R : T - R;
            const int32_t* pos = (batch->rope_pos ? batch->rope_pos : batch->positions) + r0;
            qg.step(c, w.h, w.qg, T, r0, n);
            kp.step(c, w.h, w.k,  T, r0, n);
            vp.step(c, w.h, w.v,  T, r0, n);
            if (vcal) vcal->tap(c, layer, w.v, r0, n);
            RAD_ISSUE_N(c, op_q_norm, n * g.n_head,
                        bcol_at(w.qg, r0, qgw, 0, hd, n), RAD_W(w_q_norm),
                        brow_slice(w.q, r0, n, qw));
            RAD_ISSUE_N(c, op_k_norm, n * g.n_head_kv,
                        brow_slice(w.k, r0, n, kvw), RAD_W(w_k_norm),
                        brow_slice(w.k, r0, n, kvw));
            RAD_ISSUE_N(c, op_rope_q, n, brow_slice(w.q, r0, n, qw), praw(pos, RAD_I32, n));
            RAD_ISSUE_N(c, op_rope_k, n, brow_slice(w.k, r0, n, kvw), praw(pos, RAD_I32, n));
            if (pcal) pcal->tap(c, layer, w.k, w.v, r0, n);
            RAD_ISSUE_N(c, op_kv_store, n,
                        brow_slice(w.k, r0, n, kvw), brow_slice(w.v, r0, n, kvw),
                        praw(slot + r0, RAD_I32, n), kv_cache(kv, layer));
            RAD_ISSUE_N(c, op_attn, n,
                        brows(w.q, T), kv_cache(kv, layer), praw2(table, RAD_I32, 1, tpitch),
                        praw(p == 0 ? batch->positions + R : used, RAD_I32, 1),
                        RAD_NONE, RAD_NONE, praw(pieces->cu + p, RAD_I32, 2),
                        brows(w.attn.x, T));
            RAD_ISSUE_N(c, op_gq, n * g.n_head,
                        bcol_at(w.qg, r0, qgw, hd, hd, n), brow_slice(w.attn.x, r0, n, qw),
                        brow_slice(w.attn.cq(), r0, n, qw),
                        brow_slice(w.attn.cs(), r0, n, fp8_blocks(qw)),
                        brow_slice(w.attn.x, r0, n, qw));
            o.step(c, w.attn, w.h.x, T, r0, n);
            if (p == 0) pieces->hand(r0, n);
        }
        return;
    }

    if (!ext_in) nq_h.step(c, T);
    if (op_fill) {
        /* Attention-zero rank: zero output, then the reduce and the residual. */
        RAD_ISSUE_N(c, op_fill, T, brows(w.h.x, T));
        if (own_ar && op_ar && !ar_taken(g, T, ar_out, ar_out_take))
            RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
        if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h.x, T), brows(w.x, T));
        return;
    }
    qg.step(c, w.h, w.qg, T);
    kp.step(c, w.h, w.k,  T);
    vp.step(c, w.h, w.v,  T);
    if (vcal) vcal->tap(c, layer, w.v, 0, T);

    /* Fused or unfused prologue. The fused kernel takes the whole [q|gate] projection and splits it
     * itself, and rotates k in place like the unfused ops. It reads one rotary component per
     * token, so a pass with media tokens (differing components) takes the unfused path with the
     * full [3, T] positions. */
    const bool fuse_ok = T <= fused_rows && !rope_mixed(batch);
    /* A streamed group folds the streamed store into the prologue instead (op_fused_stream); one
     * whose layer has no mirror row leaves the store to store_cache, which fails the step. */
    const bool streamed = kvb && kvb->stream;
    const bool fused_kv = op_fused_kv && fuse_ok && !streamed;
    const KVStreamLayer L = streamed && op_fused_stream && fuse_ok
                                ? kv_stream_layer(c, kv, layer, kvb) : KVStreamLayer{};
    const bool folded_stream = L.host && L.blocks > 0;
    if (folded_stream) {
        /* The prologue is the store, so the stage copy must land before it (kv_stage_wait). */
        const bool staged = kv_stage_wait(c, kv, kvb, layer);
        RAD_ISSUE_N(c, op_fused_stream, T,
                    brows(w.qg, T), brows(w.k, T), RAD_B(w.cos_sin), rope_pos1(batch, T),
                    RAD_W(w_q_norm), RAD_W(w_k_norm),
                    brows(w.q, T), brows(w.k, T), RAD_NONE,
                    brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer),
                    praw(L.res, RAD_I32, L.blocks), praw2(L.host, RAD_U8, L.blocks, L.bbytes));
        if (staged) store_stage_identity(c, slot, T);
    } else if (fused_kv) {
        RAD_ISSUE_N(c, op_fused_kv, T,
                    brows(w.qg, T), brows(w.k, T), RAD_B(w.cos_sin), rope_pos1(batch, T),
                    RAD_W(w_q_norm), RAD_W(w_k_norm),
                    brows(w.q, T), brows(w.k, T), RAD_NONE,
                    brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));
    } else if (op_fused && fuse_ok) {
        RAD_ISSUE_N(c, op_fused, T,
                    brows(w.qg, T), brows(w.k, T), RAD_B(w.cos_sin), rope_pos1(batch, T),
                    RAD_W(w_q_norm), RAD_W(w_k_norm),
                    brows(w.q, T), brows(w.k, T), RAD_NONE);
    } else {
        RAD_ISSUE_N(c, op_q_norm, T * g.n_head,
                    bcol(w.qg, 0, g.head_dim, T), RAD_W(w_q_norm), brows(w.q, T));
        RAD_ISSUE_N(c, op_k_norm, T * g.n_head_kv,
                    brows(w.k, T), RAD_W(w_k_norm), brows(w.k, T));
        RAD_ISSUE(c, op_rope_q, brows(w.q, T), rope_posmc(g, batch, T));
        RAD_ISSUE(c, op_rope_k, brows(w.k, T), rope_posmc(g, batch, T));
    }

    /* The K-to-V fit's tap: K after its norm and rope, V after its projection, both unrotated. */
    if (pcal) pcal->tap(c, layer, w.k, w.v, 0, T);
    /* A rotated layer's q, k and v, after the rope (which pairs channels the rotation mixes) and
     * before the store. Neither fold above is declared on such a layer. */
    if (op_rot) {
        RAD_ISSUE_N(c, op_rot, T * g.n_head,    brows(w.q, T));
        RAD_ISSUE_N(c, op_rot, T * g.n_head_kv, brows(w.k, T));
        RAD_ISSUE_N(c, op_rot, T * g.n_head_kv, brows(w.v, T));
    }
    /* A centred layer's V, in the basis it is stored in (neither fold is declared on it); a
     * predicted layer's V less its prediction, likewise. */
    centre_v(c, 0, T);
    predict_v(c, T);
    if (!fused_kv && !folded_stream) store_cache(c, kvb, slot, T);

    /* Sparse attention issues one row per query. `attn_paged` indexes its block table and length
     * by row, and a row need not be a sequence, so T queries issued as T rows of q_len 1, each with
     * its own selection and `seqused`, is per-query sparse attention; at decode T is the sequence
     * count.
     *
     * Below `qsa_exact_to` the dense path gives the same result more cheaply (see the field);
     * `max_ctx_len` is a host-side bound the batch already carries.
     *
     * Launches are chunked because the split-KV scratch is sized at declare from `max_seqs` and
     * `q_len` (rows = max_seqs * q_len_max * heads); more rows are refused as RAD_E_SCRATCH at
     * issue. Rows of q_len 1 up to the declared max_seqs always fit. */
    const int64_t reach  = (int64_t)batch->max_ctx_len + batch->max_q_len;
    /* A pass on a streamed group's ring (a ring smaller than the id space, and not staged) must
     * take the sparse path: on the ring a block id is not a pool address, so the dense path would
     * read ring slots as pages. `T == n_seq` alone misses a verify pass, whose T is
     * n_seq x (1 + n_spec), on a context under qsa_exact_to. A ring that covers the context keeps
     * the dense path, where ids and slots coincide. */
    const RadKVStream* st = kvb ? kvb->stream : nullptr;
    const int64_t st_blocks =
        (st && st->block_bytes > 0) ? st->host_layer_stride / st->block_bytes : 0;
    const bool ring_addressed = st && st->slots > 0 && st_blocks > 0 && st->slots < st_blocks;
    const bool ring_bound = ring_addressed && !kv_stage_layer(c, kv, kvb, layer);
    const bool    sparse = qsa_sel && qsa_sequ &&
                           (ring_bound || T == batch->n_seq || qsa_exact_to <= 0 ||
                            reach > qsa_exact_to);
    const bool gated = sparse && op_attn_gq;
    /* A streamed group's selection, per launch: make the rows' blocks resident and rewrite `sel`
     * as ring slots, after this layer's store (a block copied in from the mirror must carry this
     * step's tokens) and before the attention. A pass on the stage view leaves `sel` as block ids
     * and never resolves, so a prefill cannot overflow the ring. */
    const KVStreamLayer res_l = op_resolve ? kv_stream_layer(c, kv, layer, kvb) : KVStreamLayer{};
    auto resolve_sel = [&](int64_t off, int64_t rows) {
        if (kv_stage_layer(c, kv, kvb, layer)) return;
        if (!res_l.host || res_l.blocks <= 0 || res_l.slots <= 0 || rows <= 0) return;
        kv_resolve_issue(c, op_resolve, res_l, brow_slice(qsa_sel, off, rows, qsa_topk + 1),
                         brow_slice(qsa_nsel, off, rows, 1), kv, layer, rows);
    };
    /* A piece at a time where the caller takes the all-reduce that way (ArPieces). On the gated
     * sparse path every query row is its own (q_len 1) and the gate rides in the merge, so the
     * attention and out projection can finish a piece's rows and hand off its reduction while the
     * next piece's attention runs. Each piece is launched as a step of the piece's width would
     * launch it. */
    if (gated && !own_ar && pieces && pieces->cuts(T) && !op_add) {
        const int64_t chunk = qsa_gq_rows();
        const int64_t qw = g.q_dim(), hd = g.head_dim;
        for (int64_t p0 = 0; p0 < T; p0 += pieces->rows) {
            const int64_t pn = T - p0 < pieces->rows ? T - p0 : pieces->rows;
            for (int64_t off = p0; off < p0 + pn; off += chunk) {
                const int64_t rows = (p0 + pn - off) < chunk ? (p0 + pn - off) : chunk;
                resolve_sel(off, rows);
                RAD_ISSUE_N(c, op_attn_gq, 1,
                            brow_slice(w.q, off, rows, qw),
                            kv_cache(kv, layer),
                            brow_slice(qsa_sel, off, rows, qsa_topk + 1),
                            brow_slice(qsa_sequ, off, rows, 1),
                            RAD_NONE, RAD_NONE, RAD_NONE,
                            brow_slice(w.attn.x, off, rows, qw),
                            bcol_at(w.qg, off, g.n_head * 2 * hd, hd, hd, rows),
                            brow_slice(w.attn.cq(), off, rows, qw),
                            brow_slice(w.attn.cs(), off, rows, fp8_blocks(qw)));
            }
            o.step(c, w.attn, w.h.x, T, p0, pn);
            if (p0 + pn < T) pieces->hand(p0, pn);
        }
        /* Every attention launch of this layer is issued: start the next layer's stage copy. */
        if (kvb && kvb->stream) kv_stage_next(c, kv, layer);
        return;
    }
    if (gated) {
        const int64_t chunk = qsa_gq_rows();
        const int64_t qw = g.q_dim(), hd = g.head_dim;
        for (int64_t off = 0; off < T; off += chunk) {
            const int64_t rows = (T - off) < chunk ? (T - off) : chunk;
            resolve_sel(off, rows);
            RAD_ISSUE_N(c, op_attn_gq, 1,
                        brow_slice(w.q, off, rows, qw),
                        kv_cache(kv, layer),
                        brow_slice(qsa_sel, off, rows, qsa_topk + 1),
                        brow_slice(qsa_sequ, off, rows, 1),
                        RAD_NONE, RAD_NONE, RAD_NONE,
                        brow_slice(w.attn.x, off, rows, qw),
                        bcol_at(w.qg, off, g.n_head * 2 * hd, hd, hd, rows),
                        brow_slice(w.attn.cq(), off, rows, qw),
                        brow_slice(w.attn.cs(), off, rows, fp8_blocks(qw)));
        }
    } else if (sparse) {
        const int64_t chunk = qsa_rows();
        const int64_t qw = g.q_dim();
        for (int64_t off = 0; off < T; off += chunk) {
            const int64_t rows = (T - off) < chunk ? (T - off) : chunk;
            resolve_sel(off, rows);
            RAD_ISSUE_N(c, op_attn, 1,
                        brow_slice(w.q, off, rows, qw),
                        kv_cache(kv, layer),
                        brow_slice(qsa_sel, off, rows, qsa_topk + 1),
                        brow_slice(qsa_sequ, off, rows, 1),
                        RAD_NONE, RAD_NONE, RAD_NONE,
                        brow_slice(w.attn.x, off, rows, qw));
            /* P K-hat over the same, resolved selection (resolved once: a second resolve would
             * read ring slots as block ids). */
            if (op_attn_pk)
                RAD_ISSUE_N(c, op_attn_pk, 1,
                            brow_slice(w.q, off, rows, qw),
                            kv_cache(kv, layer),
                            brow_slice(qsa_sel, off, rows, qsa_topk + 1),
                            brow_slice(qsa_sequ, off, rows, 1),
                            RAD_NONE, RAD_NONE, RAD_NONE,
                            brow_slice(pk, off, rows, qw));
        }
    } else {
        /* Below qsa_exact_to the dense path reads block ids straight from the table and never
         * resolves, so a staged pass stages this layer here (a no-op without a stage view). A
         * ring-bound pass with no selection to resolve cannot be served at all. */
        if (ring_bound) {
            rad_step_fail(c, "kv streaming: a pass on the streamed group's ring has no selection "
                             "to resolve, so its block ids are not pool addresses");
        } else if (kvb && kvb->stream) {
            (void)kv_stage_layer(c, kv, kvb, layer);
        }
        /* A mixed step issues its attention once per band, as the GDN blocks do: one launch at
         * the chunk's band would walk every decode sequence's whole context with one query-tiled
         * workgroup a KV head, the step's tail beside a long chunk. The decode half (sequences
         * [0, D), rows [0, DT)) is a prefix at its own band. The prefill half takes its
         * per-sequence arrays past the decode sequences and keeps q, out and cu at their absolute
         * rows, as the kernels index q and out by cu.
         *
         * The prefill half is issued at the op's top band, not at its own query length: a launch
         * over part of the step must be served by the query-tiled row, while a short chunk's band
         * (10 rows or fewer at gqa 6) is the split-KV decode row's, whose partials are sized at
         * declare for that band and would overflow the scratch. Kernels read exact lengths from
         * cu, so the loose band costs nothing. A chunk no longer than the verify blocks beside it
         * keeps one launch at their band: the whole step is decode-shaped then. */
        int64_t D = 0, DT = 0;
        batch_split(batch, &D, &DT);
        const int64_t pitch = kvb ? kvb->block_table_pitch : 0;
        const int32_t qp = batch->max_q_len_prefill > 0 ? batch->max_q_len_prefill
                                                        : batch->max_q_len;
        if (batch->phase == RAD_PHASE_MIXED && D > 0 && D < batch->n_seq &&
            batch->max_q_len_decode > 0 && qp > batch->max_q_len_decode) {
            const int64_t P = batch->n_seq - D;
            /* The output pass, then a predicted layer's P K-hat pass at the same bands. */
            for (int pass = 0; pass < (op_attn_pk ? 2 : 1); ++pass) {
                const rad_op op = pass ? op_attn_pk : op_attn;
                const rad_buf ob = pass ? pk : w.attn.x;
                RAD_ISSUE_N(c, op, batch->max_q_len_decode,
                            brows(w.q, DT), kv_cache(kv, layer),
                            praw2(table, RAD_I32, D, pitch), praw(used, RAD_I32, D),
                            RAD_NONE, RAD_NONE, praw(batch->cu_seqlens, RAD_I32, D + 1),
                            brows(ob, DT));
                RAD_ISSUE_N(c, op, g.max_tok,
                            brows(w.q, T), kv_cache(kv, layer),
                            praw2(table ? table + D * pitch : nullptr, RAD_I32, P, pitch),
                            praw(used ? used + D : nullptr, RAD_I32, P),
                            RAD_NONE, RAD_NONE, praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                            brows(ob, T));
            }
        } else {
            for (int pass = 0; pass < (op_attn_pk ? 2 : 1); ++pass)
                RAD_ISSUE_N(c, pass ? op_attn_pk : op_attn, batch->max_q_len,
                            brows(w.q, T), kv_cache(kv, layer),
                            praw2(table, RAD_I32, batch->n_seq, pitch),
                            praw(used, RAD_I32, batch->n_seq),
                            RAD_NONE, RAD_NONE,
                            /* cu_seqlens: verify rows may be of different lengths. */
                            praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                            brows(pass ? pk : w.attn.x, T));
        }
    }

    /* Every attention launch of this layer is issued: start the next layer's stage copy, which
     * crosses the link under the out projection and the next layer's projections. */
    if (kvb && kvb->stream) kv_stage_next(c, kv, layer);
    /* A centred layer's mean back on the output, in the cache's basis, so before the rotation;
     * then a rotated layer's output back, before the gate reads it (never `gated`: see declare). */
    if (op_vadd)
        RAD_ISSUE_N(c, op_vadd, T, brows(w.attn.x, T), praw(vmean_add, RAD_BF16, g.q_dim()),
                    brows(w.attn.x, T));
    /* A predicted layer's (P K-hat) A + b back, likewise before the rotation. */
    if (op_vpadd)
        RAD_ISSUE_N(c, op_vpadd, T, brows(pk, T),
                    praw(vpred_map, RAD_F32, g.n_head_kv * (g.head_dim + 1) * g.head_dim),
                    brows(w.attn.x, T), brows(w.attn.x, T));
    if (op_rot) RAD_ISSUE_N(c, op_rot, T * g.n_head, brows(w.attn.x, T));
    if (gated) {
        /* the sparse attention gated and quantised in its merge */
    } else if (op_sig) {
        /* The gate is the strided second half of each head's [q|gate] pair. */
        RAD_ISSUE_N(c, op_sig, T * g.n_head,
                    bcol(w.qg, g.head_dim, g.head_dim, T), brows(gate, T));
        RAD_ISSUE(c, op_mul, brows(w.attn.x, T), brows(gate, T), brows(w.attn.x, T));
        q_gate.step(c, T);
    } else {
        RAD_ISSUE_N(c, op_gq, T * g.n_head,
                    bcol(w.qg, g.head_dim, g.head_dim, T), brows(w.attn.x, T),
                    brows(w.attn.cq(), T), brows(w.attn.cs(), T), brows(w.attn.x, T));
    }

    o.step(c, w.attn, w.h.x, T);
    if (own_ar && op_ar && !ar_taken(g, T, ar_out, ar_out_take))
        RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
    if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h.x, T), brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_ATTN_GATED_FP8_H */
