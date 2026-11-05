/* Spark X2.6 (Spark2_5ForCausalLM), rotated int8 weights and activations.
 *
 * The checkpoint has one [q | k | v] projection per layer. Conversion splits its rows into
 * contiguous matrices because paged attention needs a contiguous query. The MLP's two matrices
 * are concatenated as [gate | up]; act_mul with exact erf GELU matches the reference, unlike
 * gelu_mul (which is the tanh approximation). The attention gate is one sigmoid per head, not
 * per channel. Local and full layers have different NeoX rotary dimensions and theta values.
 *
 * Tensor parallelism splits attention by whole KV groups, replicating KV heads when ranks
 * outnumber them. As in the HF reference, residuals accumulate in f32; rmsnorm reads that stream
 * and writes bf16 for the linear projections.
 *
 * A container built with `--draft-model` also carries a DSpark drafter, which reads the residual
 * stream after the trunk layers its config names and denoises a block of masked positions in one
 * pass. The drafter itself is bf16 whatever the trunk's quantisation (arch/rad_block_dspark_bf16.h
 * says why); the trunk's part is the taps and the drafter pass dispatch below.
 */
#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>
#include <arch/rad_block_dspark_bf16.h>
#include "../spark25_bf16/chat.h"
#include "../spark25_bf16/spark25_prefill_pass_key.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace spark25_w8a8 {
using namespace rad::arch;

struct Layer {
    rad_kvgroup kv = 0;
    rad_weight an = 0, gate = 0, fn = 0;
    LinearFP8 q{}, k{}, v{}, o{}, gu{}, down{};
    rad_op anorm = 0, gp = 0, rq = 0, rk = 0, store = 0, attn = 0;
    rad_op sigmoid = 0, gate_heads = 0, widen_attn = 0, reduce_attn = 0, add_attn = 0;
    rad_op fnorm = 0, act = 0, widen_ffn = 0, reduce_ffn = 0, add_ffn = 0;
    CalibTap tap_in{}, tap_o{}, tap_gu{}, tap_down{};
};

struct Model {
    Names nm{""};
    Geom g{};
    rad_kvgroup kv = 0;
    rad_buf x = 0, delta = 0, h = 0, q = 0, k = 0, v = 0, gate = 0, attn = 0, gu = 0, ff = 0;
    rad_buf hout = 0, logits = 0;
    ActFP8 a_h{}, a_attn{}, a_ff{};
    rad_weight tok = 0, norm = 0, head = 0;
    rad_op embed = 0, widen = 0, reduce_embed = 0, had = 0, final_norm = 0, gather = 0, logits_op = 0;
    std::vector<Layer> layers;
    bool tap_live = false;
    long long calib_rows = 0;
    int64_t calib_lo = 0, calib_hi = 0;
    /* The drafter, declared only when the container carries one and this serve speculates. Its own
     * KV group holds the context it derived from the trunk's taps. */
    DsparkBF16     ds{};
    DsparkBF16::Wire dsw{};
    DsparkTaps     taps{};
    rad_kvgroup    kv_draft = 0;
    bool           have_ds = false;
};
static Model models[MAX_RANKS];

/* Whether the container carries a drafter at all (`--draft-model` merged one under `draft.`), and
 * whether this file implements it. A drafter this plugin cannot run is refused at declare rather
 * than left unused: the container would serve at the speed of a model with no drafter, and the
 * user would have no way to find out that the flag they passed did nothing. */
static int drafter_kind(const RadModelMeta* meta, bool* present) {
    *present = false;
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    if (!da || !*da) {
        /* A drafter's config.json flattened, or a GGUF import, either of which names it. */
        if (rad_meta_geti(meta, "draft.num_hidden_layers", 0) <= 0) return RAD_OK;
        da = rad_meta_gets(meta, "draft.model_type", "");
    }
    if (std::strstr(da, "DSpark") || std::strcmp(da, "dspark") == 0) {
        *present = true;
        return RAD_OK;
    }
    *present = true;
    fprintf(stderr, "radiance: spark25: this container carries a '%s' drafter. Spark's drafts are "
                    "DSpark, with a rank-N Markov head over the target's own table; this "
                    "architecture implements that and nothing else\n", da);
    return RAD_E_UNSUPPORTED;
}

/* The HF config names every layer. Do not infer a periodic schedule: it changes the mask and
 * rotary of a layer and would silently produce a different model. */
static int schedule(const RadModelMeta* meta, int64_t n, std::vector<bool>& local) {
    const char* s = rad_meta_gets(meta, "layer_types", nullptr);
    if (!s) {
        fprintf(stderr, "radiance: spark25: layer_types is required\n");
        return RAD_E_INVAL;
    }
    std::istringstream in(s);
    std::string word;
    while (in >> word) {
        if (word != "sliding_attention" && word != "full_attention") {
            fprintf(stderr, "radiance: spark25: unknown layer type '%s'\n", word.c_str());
            return RAD_E_UNSUPPORTED;
        }
        local.push_back(word == "sliding_attention");
    }
    if ((int64_t)local.size() != n) {
        fprintf(stderr, "radiance: spark25: layer_types has %zu entries for %lld layers\n",
                local.size(), (long long)n);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

struct Span {
    int64_t first = 0, count = 0;
};

/* Remainder-first contiguous slices, in whole units. */
static Span split(int64_t n, int parts, int rank) {
    const int64_t base = n / parts, rem = n % parts;
    return {base * rank + std::min<int64_t>(rank, rem), base + (rank < rem ? 1 : 0)};
}

struct Shards {
    Span q, kv, ff;  /* q and kv in heads; ff in channels */
};

/* A rank's attention heads must describe whole GQA groups, or a slice of a single group with
 * that group's KV head replicated. This also covers non-divisible worlds such as 3 ranks over
 * the checkpoint's 4 KV heads. Projection and FFN spans below give the loader those exact cuts. */
static int shard_geometry(Geom& g, const RadBuildCtx* ctx, Shards& s) {
    g.world = ctx->world_size > 0 ? ctx->world_size : 1;
    g.rank = ctx->rank;
    if (g.rank >= g.world || g.world > g.n_head_all || g.n_head_kv_all <= 0 ||
        g.n_head_all % g.n_head_kv_all || g.world > g.n_ff_all) {
        fprintf(stderr, "radiance: spark25: cannot shard %lld query heads, %lld KV heads and "
                        "%lld FFN channels over %d ranks\n", (long long)g.n_head_all,
                (long long)g.n_head_kv_all, (long long)g.n_ff_all, g.world);
        return RAD_E_INVAL;
    }
    const int64_t group = g.n_head_all / g.n_head_kv_all;
    if (g.world <= g.n_head_kv_all) {
        s.kv = split(g.n_head_kv_all, g.world, g.rank);
        s.q = {s.kv.first * group, s.kv.count * group};
    } else {
        int first_rank = 0;
        for (int64_t k = 0; k < g.n_head_kv_all; ++k) {
            const int n = (int)split(g.world, (int)g.n_head_kv_all, (int)k).count;
            if (g.rank < first_rank + n) {
                const Span q = split(group, n, g.rank - first_rank);
                s.kv = {k, 1};
                s.q = {k * group + q.first, q.count};
                break;
            }
            first_rank += n;
        }
    }
    /* Prefer whole 128-channel blocks for a ragged split, keeping device GEMMs aligned. */
    const int64_t unit = g.n_ff_all % 128 == 0 && g.n_ff_all / 128 >= g.world ? 128 : 1;
    s.ff = split(g.n_ff_all / unit, g.world, g.rank);
    s.ff.first *= unit;
    s.ff.count *= unit;
    g.n_head = s.q.count;
    g.n_head_kv = s.kv.count;
    g.n_ff = s.ff.count;
    g.n_vocab = shard_extent(g.n_vocab_all, g.world, g.rank);
    g.vocab_off = shard_begin(g.n_vocab_all, g.world, g.rank);
    return g.n_head > 0 && g.n_head_kv > 0 && g.n_ff > 0 && g.n_vocab > 0 ? RAD_OK : RAD_E_INVAL;
}

static bool needs_span(const Span& s, int64_t total, const Geom& g) {
    return total % g.world != 0 || s.first != g.rank * (total / g.world) ||
           s.count != total / g.world;
}

static int check_format(const RadModelMeta* meta) {
    const char* method = rad_meta_gets(meta, "quantization_config.quant_method", nullptr);
    const char* fmt = rad_meta_gets(meta, "quantization_config.fmt", nullptr);
    if (!method || std::strcmp(method, "w8a8") != 0 || !fmt || std::strcmp(fmt, "fwht128") != 0) {
        fprintf(stderr, "radiance: spark25_w8a8 serves a rotated int8 trunk (quant_method "
                        "'w8a8', fmt 'fwht128'); this container says '%s' / '%s'\n",
                method ? method : "(none)", fmt ? fmt : "(none)");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    RAD_ARCH_TRY(check_format(meta));
    Model& m = models[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    /* geom_from's even-only split cannot express a ragged GQA/FFN world. Read the whole geometry,
     * then give each rank a group-preserving shard and state its spans on the weights. */
    RadBuildCtx whole = *ctx;
    whole.rank = 0;
    whole.world_size = 1;
    RAD_ARCH_TRY(geom_from(m.g, meta, &whole, "bf16", RAD_BF16));
    Geom& g = m.g;
    Shards shards;
    RAD_ARCH_TRY(shard_geometry(g, ctx, shards));
    /* A column-sharded int8 linear slices its input's scale groups, so every rank's
     * feed-forward span must be whole 128-column groups. shard_geometry prefers whole blocks
     * even for ragged worlds such as TP=3; refuse a geometry that cannot be split that way. */
    if (g.world > 1 && shards.ff.count % RAD_FP8_BLOCK) {
        fprintf(stderr, "radiance: spark25_w8a8 cannot shard %lld FFN channels over %d ranks: a "
                        "rank's %lld are not whole %d-column scale groups. An int8 down projection "
                        "sliced inside a group would read its scale plane at the wrong width\n",
                (long long)g.n_ff_all, g.world, (long long)shards.ff.count, RAD_FP8_BLOCK);
        return RAD_E_UNSUPPORTED;
    }
    Names& nm = m.nm;
    if (g.n_head_kv <= 0 || g.n_head % g.n_head_kv || g.head_dim % 2 || meta->n_expert ||
        rad_meta_geti(meta, "attention_bias", 0) || rad_meta_geti(meta, "mlp_bias", 0) ||
        !rad_meta_geti(meta, "tie_word_embeddings", 0) ||
        !rad_meta_geti(meta, "headwise_attn_output_gate", 0) ||
        std::strcmp(rad_meta_gets(meta, "gate_attn_act_mode", "sigmoid"), "sigmoid") ||
        std::strcmp(rad_meta_gets(meta, "hidden_act", "gelu"), "gelu")) {
        fprintf(stderr, "radiance: spark25: requires a dense, bias-free, tied model with headwise "
                        "sigmoid gates, exact GELU and divisible attention heads\n");
        return RAD_E_UNSUPPORTED;
    }
    /* A single global override cannot describe this model's two rotary ladders. */
    RAD_ARCH_TRY(yarn_from(g, meta, "", g.head_dim));
    if (g.yarn.factor != 0.0f || g.rope_scale != 1.0f || g.rope_mc) {
        fprintf(stderr, "radiance: spark25: rotary scaling overrides are not supported\n");
        return RAD_E_UNSUPPORTED;
    }
    std::vector<bool> local;
    RAD_ARCH_TRY(schedule(meta, g.n_layer, local));
    if (calib_dir()) {
        if (g.world != 1) {
            fprintf(stderr, "radiance: spark25 calibration requires one rank so column-split "
                            "Grams retain their cross terms\n");
            return RAD_E_UNSUPPORTED;
        }
        RAD_ARCH_TRY(calib_layers("spark25", "RADIANCE_SPARK_CALIB_LAYERS", g.n_layer,
                                  &m.calib_lo, &m.calib_hi));
        m.tap_live = true;
    }
    const int64_t window = rad_meta_geti(meta, "sliding_window", 0);
    if (window <= 0) {
        fprintf(stderr, "radiance: spark25: a positive sliding_window is required\n");
        return RAD_E_INVAL;
    }
    double theta[2]{};
    int64_t rot[2]{};
    const char* kinds[] = {"full_attention", "sliding_attention"};
    for (int j = 0; j < 2; ++j) {
        theta[j] = rad_meta_getf(meta, nm.f("rope_parameters.%s.rope_theta", kinds[j]), 10000.0);
        const double factor = rad_meta_getf(meta,
            nm.f("rope_parameters.%s.partial_rotary_factor", kinds[j]), 1.0);
        const char* type = rad_meta_gets(meta, nm.f("rope_parameters.%s.rope_type", kinds[j]), "default");
        if (std::strcmp(type, "default")) {
            fprintf(stderr, "radiance: spark25: %s rotary type '%s' is not supported\n", kinds[j], type);
            return RAD_E_UNSUPPORTED;
        }
        const double r = g.head_dim * factor;
        if (!std::isfinite(theta[j]) || !(theta[j] > 0) || !(factor > 0 && factor <= 1) || !std::isfinite(r) ||
            std::floor(r) != r || (int64_t)r % 2) {
            fprintf(stderr, "radiance: spark25: invalid %s rotary theta %g or factor %g\n",
                    kinds[j], theta[j], factor);
            return RAD_E_INVAL;
        }
        rot[j] = (int64_t)r;
    }

    /* A full cache for both kinds keeps prefix caching valid. The local attention op enforces
     * its own window (q - k < window); the full layers read all past positions. */
    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_FULL;
    kd.dtype = rad_kv_cache_dtype(ctx, RAD_BF16);
    kd.n_head_kv = g.n_head_kv;
    kd.head_dim = g.head_dim;
    m.kv = rad_decl_kv_group(b, nm.f("kv_attn"), &kd);
    if (!m.kv) return RAD_E_INVAL;
    g.kv_dtype = rad_dtype_name(kd.dtype);

    const int64_t T = g.max_tok, E = g.n_embd, Q = g.q_dim(), K = g.kv_dim(), F = g.n_ff;
    m.x = decl_b(b, nm.f("x"), RAD_F32, {T, E});
    m.delta = decl_b(b, nm.f("delta"), RAD_F32, {T, E});
    m.h = decl_b(b, nm.f("h"), RAD_BF16, {T, E});
    m.q = decl_b(b, nm.f("q"), RAD_BF16, {T, Q});
    m.k = decl_b(b, nm.f("k"), RAD_BF16, {T, K});
    m.v = decl_b(b, nm.f("v"), RAD_BF16, {T, K});
    m.gate = decl_b(b, nm.f("gate"), RAD_BF16, {T, g.n_head});
    m.attn = decl_b(b, nm.f("attn"), RAD_BF16, {T, Q});
    m.gu = decl_b(b, nm.f("gate_up"), RAD_BF16, {T, 2 * F});
    m.ff = decl_b(b, nm.f("ff"), RAD_BF16, {T, F});
    m.hout = decl_b(b, nm.f("h_out"), RAD_BF16, {g.max_logit_rows, E});
    m.logits = decl_b(b, nm.f("logits"), RAD_F32, {g.max_logit_rows, g.n_vocab});
    if (!m.x || !m.delta || !m.h || !m.q || !m.k || !m.v || !m.gate || !m.attn || !m.gu || !m.ff ||
        !m.hout || !m.logits) return RAD_E_INVAL;
    m.a_h.x = m.h;
    RAD_ARCH_TRY(m.a_h.declare_qs(b, nm, g, "h", E, 0, true, false));
    m.a_h.q8_fed = true;
    m.a_h.q8_rot = true;
    m.a_attn.x = m.attn;
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "attn_out", Q, 0, true, false));
    m.a_ff.x = m.ff;
    RAD_ARCH_TRY(m.a_ff.declare_qs(b, nm, g, "ff", F, 0, true, false));
    m.had = rw(b, RAD_OP(b, "had_quant_act_i8g",
        RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E), RAD_INT("group", RAD_FP8_BLOCK),
                   RAD_STR("dtype", "bf16")), RAD_NOWEIGHTS),
        {m.h}, {m.a_h.q8, m.a_h.s8});
    if (!m.had) return RAD_E_INVAL;
    RAD_ARCH_TRY(rad_declare_logits(b, m.logits));

    /* ---- the drafter's taps, when the container carries a drafter and this serve speculates.
     * Declared before the layer loop because the stream outlives it: the drafter's context pass
     * runs after the trunk's, on the residual streams the trunk saved here. */
    bool drafter = false;
    RAD_ARCH_TRY(drafter_kind(meta, &drafter));
    if (drafter && ctx->max_spec > 0) {
        m.taps.read(meta, g.n_layer);
        if (m.taps.n <= 0) {
            fprintf(stderr, "radiance: spark25: the drafter names no trunk layer to read, or names "
                            "only layers past this target's last (`draft.target_layer_ids`)\n");
            return RAD_E_FORMAT;
        }
        RAD_ARCH_TRY(m.taps.declare(b, nm, g, g.max_tok, m.x));
    }

    auto weight = [&](const char* name, const char* src, uint32_t dt,
                      std::initializer_list<int64_t> shape, RadWeightGroup group,
                      int access = RAD_ACCESS_PER_TOKEN, int shard = RAD_SHARD_NONE) {
        if (map_copy(b, name, src) != RAD_OK) return (rad_weight)0;
        return decl_w(b, name, dt, shape, access, shard, group);
    };
    m.tok = weight(nm.f("token_embd.weight"), nm.ckpt("model.embedding.weight"), RAD_BF16,
                   {g.n_vocab, E}, grp_model(), RAD_ACCESS_VOCAB, RAD_SHARD_ROW);
    /* The head is the tied table again, at whatever the container holds it: bf16 as shipped, or
     * the int8 grid where the recipe quantised it, which is the widest GEMM a decode step runs. */
    if (map_copy(b, nm.f("output.weight"), nm.ckpt("model.embedding.weight")) != RAD_OK)
        return RAD_E_INVAL;
    m.head = decl_w(b, nm.f("output.weight"),
                    weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                    {g.n_vocab, E}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.norm = weight(nm.f("output_norm.weight"), nm.ckpt("model.norm.weight"), RAD_F32,
                    {E}, grp_model());
    if (!m.tok || !m.head || !m.norm) return RAD_E_INVAL;
    if (g.world > 1 && g.n_vocab_all % g.world) {
        RAD_ARCH_TRY(rad_weight_shard_span(b, m.tok, g.vocab_off, g.vocab_off + g.n_vocab));
        RAD_ARCH_TRY(rad_weight_shard_span(b, m.head, g.vocab_off, g.vocab_off + g.n_vocab));
    }

    auto norm = [&](rad_weight w, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "rmsnorm", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
            RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0), RAD_STR("dtype", "bf16")),
            RAD_WEIGHTS(w)), {x}, {y});
    };
    auto gemm = [&](int64_t n, int64_t k, rad_weight w, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "gemm_nt", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", n),
            RAD_INT("K", k), RAD_STR("dtype", "bf16")), RAD_WEIGHTS(w)), {x}, {y});
    };
    auto add = [&]() {
        return rw(b, RAD_OP(b, "add", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
            RAD_STR("dtype", "f32")), RAD_NOWEIGHTS), {m.x, m.delta}, {m.x});
    };
    m.embed = rw(b, RAD_OP(b, "embed_lookup", RAD_PARAMS(RAD_RANGE("M", 1, T),
        RAD_INT("n_embd", E), RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16"),
        RAD_INT("vocab_offset", g.vocab_off)), RAD_WEIGHTS(m.tok)), {}, {m.h});
    auto widen = [&](rad_buf output) {
        return rw(b, RAD_OP(b, "cast", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
            RAD_STR("from", "bf16"), RAD_STR("to", "f32")), RAD_NOWEIGHTS), {m.h}, {output});
    };
    m.widen = widen(m.x);
    /* Embedding shards contribute disjoint rows, so their widened sum is exact. Projection
     * shards must instead reduce in bf16, rounding the merged output before it is widened and
     * added to the f32 residual, as in the reference's BF16 linear layers. */
    auto reduce = [&](rad_buf x, const char* dtype) {
        if (g.world <= 1) return (rad_op)0;
        return rw(b, RAD_OP(b, "all_reduce", RAD_PARAMS(RAD_INT("world_size", g.world),
            RAD_RANGE("numel", E, T * E), RAD_STR("dtype", dtype), RAD_INT("exact", 1)),
            RAD_NOWEIGHTS), {x}, {x});
    };
    m.reduce_embed = reduce(m.x, "fp32");
    if (g.world > 1 && !m.reduce_embed) {
        fprintf(stderr, "radiance: spark25_w8a8 requires an exact fp32 all-reduce for TP=%d\n", g.world);
        return RAD_E_UNSUPPORTED;
    }

    m.layers.resize((size_t)g.n_layer);
    for (int l = 0; l < g.n_layer; ++l) {
        Layer& a = m.layers[(size_t)l];
        auto src = [&](const char* suffix) { return nm.ckpt("model.layers.%d.%s", l, suffix); };
        auto w = [&](const char* suffix, const char* s, uint32_t dt,
                     std::initializer_list<int64_t> shape, int shard = RAD_SHARD_NONE) {
            return weight(nm.f("blk.%d.%s", l, suffix), src(s), dt, shape, grp_layer(l),
                          RAD_ACCESS_PER_TOKEN, shard);
        };
        a.an = w("attn_norm.weight", "input_layernorm.weight", RAD_F32, {E});
        a.gate = w("attn_gate.weight", "self_attn.g_proj.weight", RAD_BF16, {g.n_head, E}, RAD_SHARD_ROW);
        a.fn = w("ffn_norm.weight", "post_attention_layernorm.weight", RAD_F32, {E});
        const char* qkv = src("self_attn.q_k_v_proj.weight");
        const int64_t qall = g.n_head_all * g.head_dim, kall = g.n_head_kv_all * g.head_dim;
        const int64_t qlo = needs_span(shards.q, g.n_head_all, g) ? shards.q.first * g.head_dim : 0;
        const int64_t qhi = needs_span(shards.q, g.n_head_all, g)
                                ? (shards.q.first + shards.q.count) * g.head_dim : 0;
        const int64_t kvlo = needs_span(shards.kv, g.n_head_kv_all, g)
                                 ? shards.kv.first * g.head_dim : 0;
        const int64_t kvhi = needs_span(shards.kv, g.n_head_kv_all, g)
                                 ? (shards.kv.first + shards.kv.count) * g.head_dim : 0;
        auto map_part = [&](const char* name, int64_t first, int64_t rows) {
            const char* target = nm.f("blk.%d.attn_%s.weight", l, name);
            return map_rows(b, target, {{qkv, (int)first, rows}});
        };
        const char* qb = nm.f("blk.%d.attn_q", l);
        const char* kb = nm.f("blk.%d.attn_k", l);
        const char* vb = nm.f("blk.%d.attn_v", l);
        RAD_ARCH_TRY(map_part("q", 0, qall));
        RAD_ARCH_TRY(map_part("k", qall, kall));
        RAD_ARCH_TRY(map_part("v", qall + kall, kall));
        a.q.mapped = a.k.mapped = a.v.mapped = true;
        RAD_ARCH_TRY(a.q.declare(b, nm, g, qb, Q, E, RAD_SHARD_ROW, grp_layer(l), {qkv},
                                 m.a_h, m.q, 0, {}, qlo, qhi));
        RAD_ARCH_TRY(a.k.declare(b, nm, g, kb, K, E, RAD_SHARD_ROW, grp_layer(l), {qkv},
                                 m.a_h, m.k, 0, {}, kvlo, kvhi));
        RAD_ARCH_TRY(a.v.declare(b, nm, g, vb, K, E, RAD_SHARD_ROW, grp_layer(l), {qkv},
                                 m.a_h, m.v, 0, {}, kvlo, kvhi));
        const int64_t olo = qlo, ohi = qhi;
        RAD_ARCH_TRY(a.o.declare(b, nm, g, nm.f("blk.%d.attn_output", l), E, Q,
                                 RAD_SHARD_COL, grp_layer(l), {src("self_attn.out_proj.weight")},
                                 m.a_attn, m.h, 0, {}, olo, ohi));
        RAD_ARCH_TRY(a.gu.declare(b, nm, g, nm.f("blk.%d.ffn_gate_up", l), 2 * F, E,
                                 RAD_SHARD_ROW, grp_layer(l),
                                 {src("mlp.gate_proj.weight"), src("mlp.up_proj.weight")},
                                 m.a_h, m.gu, 0, {F, F}));
        const int64_t flo = needs_span(shards.ff, g.n_ff_all, g) ? shards.ff.first : 0;
        const int64_t fhi = needs_span(shards.ff, g.n_ff_all, g)
                                ? shards.ff.first + shards.ff.count : 0;
        if (needs_span(shards.ff, g.n_ff_all, g)) {
            const int64_t lo[2] = {shards.ff.first, shards.ff.first};
            const int64_t hi[2] = {shards.ff.first + shards.ff.count,
                                   shards.ff.first + shards.ff.count};
            const int64_t total[2] = {g.n_ff_all, g.n_ff_all};
            RAD_ARCH_TRY(a.gu.span_parts(b, lo, hi, total, 2));
        }
        RAD_ARCH_TRY(a.down.declare(b, nm, g, nm.f("blk.%d.ffn_down", l), E, F,
                                   RAD_SHARD_COL, grp_layer(l), {src("mlp.down_proj.weight")},
                                   m.a_ff, m.h, 0, {}, flo, fhi));
        if (!a.an || !a.gate || !a.fn || !a.q.w || !a.k.w || !a.v.w || !a.o.w ||
            !a.gu.w || !a.down.w) return RAD_E_INVAL;
        if (g.world > 1 && needs_span(shards.q, g.n_head_all, g))
            RAD_ARCH_TRY(rad_weight_shard_span(b, a.gate, shards.q.first,
                                               shards.q.first + shards.q.count));
        a.kv = m.kv;
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, a.kv));
        a.anorm = norm(a.an, m.x, m.h);
        a.gp = gemm(g.n_head, E, a.gate, m.h, m.gate);
        Geom rope = g;
        const int kind = local[(size_t)l] ? 1 : 0;
        rope.theta = (float)theta[kind];
        a.rq = rw(b, decl_rope(b, rope, T, g.head_dim, g.n_head, 0, rot[kind], "neox"), {m.q}, {m.q});
        a.rk = rw(b, decl_rope(b, rope, T, g.head_dim, g.n_head_kv, 0, rot[kind], "neox"), {m.k}, {m.k});
        a.attn = rw(b, decl_attn(b, g, "attn_paged",
            RAD_PARAMS(RAD_RANGE("q_len", 1, T), RAD_INT("head_dim", g.head_dim),
                RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1), RAD_INT("window", kind ? window : 0),
                RAD_INT("n_head", g.n_head), RAD_INT("block_size", RAD_KV_BLOCK),
                RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", g.kv_dtype)), RAD_NOWEIGHTS),
            {m.q}, {m.attn});
        a.store = rw(b, RAD_OP(b, "kv_store", RAD_PARAMS(RAD_RANGE("M", 1, T),
            RAD_INT("head_dim", g.head_dim), RAD_INT("n_head_kv", g.n_head_kv),
            RAD_INT("block_size", rad_kv_block_size(b, a.kv)), RAD_STR("kv_dtype", g.kv_dtype)),
            RAD_NOWEIGHTS), {m.k, m.v}, {});
        a.sigmoid = rw(b, RAD_OP(b, "sigmoid", RAD_PARAMS(RAD_RANGE("M", 1, T),
            RAD_INT("n", g.n_head), RAD_STR("dtype", "bf16")), RAD_NOWEIGHTS), {m.gate}, {m.gate});
        /* View dense [tokens, heads * dim] and [tokens, heads] as one row per head. */
        a.gate_heads = rw(b, RAD_OP(b, "scale_rows", RAD_PARAMS(RAD_RANGE("M", 1, T * g.n_head),
            RAD_INT("n", g.head_dim), RAD_STR("act", "none"), RAD_STR("dtype", "bf16")),
            RAD_NOWEIGHTS), {m.attn, m.gate}, {m.attn});
        a.reduce_attn = reduce(m.h, "bf16");
        a.widen_attn = widen(m.delta);
        a.add_attn = add();
        a.fnorm = norm(a.fn, m.x, m.h);
        a.act = rw(b, RAD_OP(b, "act_mul", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", F),
            RAD_STR("act", "gelu"), RAD_STR("dtype", "bf16")), RAD_NOWEIGHTS), {m.gu}, {m.ff});
        a.reduce_ffn = reduce(m.h, "bf16");
        if (g.world > 1 && (!a.reduce_attn || !a.reduce_ffn)) {
            fprintf(stderr, "radiance: spark25_w8a8 requires exact bf16 all-reduces for TP=%d\n", g.world);
            return RAD_E_UNSUPPORTED;
        }
        a.widen_ffn = widen(m.delta);
        a.add_ffn = add();
        if (m.tap_live && l >= m.calib_lo && l < m.calib_hi) {
            RAD_ARCH_TRY(a.tap_in.declare(b, nm, T, nm.f("blk.%d.attn_in", l), m.h, E));
            RAD_ARCH_TRY(a.tap_o.declare(b, nm, T, nm.f("blk.%d.attn_o", l), m.attn, Q));
            RAD_ARCH_TRY(a.tap_gu.declare(b, nm, T, nm.f("blk.%d.ffn_in", l), m.h, E));
            RAD_ARCH_TRY(a.tap_down.declare(b, nm, T, nm.f("blk.%d.ffn_down_in", l), m.ff, F));
        }
    }
    m.final_norm = norm(m.norm, m.x, m.h);
    m.gather = rw(b, RAD_OP(b, "gather_rows", RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
        RAD_INT("n", E), RAD_STR("dtype", "bf16")), RAD_NOWEIGHTS), {m.h}, {m.hout});
    m.logits_op = rw(b, RAD_OP(b, "logits_gemm", RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
        RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", E), RAD_STR("dtype", "bf16")),
        RAD_WEIGHTS(m.head)), {m.hout}, {m.logits});
    char ids[128] = "";
    /* The drafter is declared last, after the head. The buffer planner takes lifetimes in
     * declaration order, and a drafter's passes run after the whole trunk's, so declaring it in the
     * middle would let a trunk temporary be planned onto bytes the drafter is still reading. */
    if (m.taps.n > 0) {
        RAD_ARCH_TRY(declare_dspark(&m.ds, &m.dsw, &m.kv_draft, b, nm, meta, ctx, g, m.taps,
                                    (int)g.n_layer, m.tok, nm.ckpt("model.embedding.weight")));
        m.have_ds = true;
        for (int t = 0, off = 0; t < m.taps.n && off < (int)sizeof(ids) - 8; ++t)
            off += std::snprintf(ids + off, sizeof ids - off, "%s%d", t ? ", " : "",
                                 m.taps.layer[t]);
        rad_note(b, "the drafter reads the residual stream after layer(s) %s", ids);
    }

    rad_note(b, "spark25 rank %d/%d: %lld layers, %lld query / %lld KV heads x %lld, "
                "FFN %lld, vocab [%lld, %lld), local window %lld", g.rank, g.world,
             (long long)g.n_layer, (long long)g.n_head, (long long)g.n_head_kv,
             (long long)g.head_dim, (long long)g.n_ff, (long long)g.vocab_off,
             (long long)(g.vocab_off + g.n_vocab), (long long)window);
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = models[rad_rank(c)];
    const Geom& g = m.g;
    /* A drafter pass; the sign gives the kind. Positive is the query pass, which denoises the whole
     * block and proposes from it; negative is the context pass, which turns the tokens the trunk
     * just committed into the drafter's own K and V. Zero is a trunk step. */
    if (batch->draft_pass != 0) {
        if (!m.have_ds) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container carries no `draft.` block, or declare ran at max_spec 0");
            return;
        }
        if (batch->draft_pass > 0) m.ds.step(c, batch);
        else                       m.ds.context(c, batch);
        return;
    }
    const int64_t T = batch->n_tok;
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    RAD_ISSUE(c, m.embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.tok), brows(m.h, T));
    RAD_ISSUE(c, m.widen, brows(m.h, T), brows(m.x, T));
    if (m.reduce_embed) RAD_ISSUE_N(c, m.reduce_embed, T * g.n_embd, brows(m.x, T), RAD_NONE);
    for (size_t i = 0; i < m.layers.size(); ++i) {
        const Layer& a = m.layers[i];
        const RadKVGroupBatch* kvb = kv_batch(batch, a.kv);
        const int32_t* slot = kvb ? kvb->slot_mapping : nullptr;
        const int32_t* table = kvb ? kvb->block_table : nullptr;
        const int32_t* used = kvb ? kvb->seqused : nullptr;
        RAD_ISSUE(c, a.anorm, brows(m.x, T), RAD_W(a.an), brows(m.h, T));
        a.tap_in.step(c, m.h, T, m.tap_live);
        RAD_ISSUE(c, m.had, brows(m.h, T), brows(m.a_h.q8, T), brows(m.a_h.s8, T));
        a.q.step(c, m.a_h, m.q, T);
        a.k.step(c, m.a_h, m.k, T);
        a.v.step(c, m.a_h, m.v, T);
        RAD_ISSUE(c, a.gp, brows(m.h, T), RAD_W(a.gate), brows(m.gate, T));
        RAD_ISSUE(c, a.rq, brows(m.q, T), pos);
        RAD_ISSUE(c, a.rk, brows(m.k, T), pos);
        RAD_ISSUE(c, a.store, brows(m.k, T), brows(m.v, T), praw(slot, RAD_I32, T),
                  kv_cache(a.kv, (int)i));
        RAD_ISSUE_N(c, a.attn, batch->max_q_len, brows(m.q, T), kv_cache(a.kv, (int)i),
            praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
            praw(used, RAD_I32, batch->n_seq), RAD_NONE, RAD_NONE,
            praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1), brows(m.attn, T));
        RAD_ISSUE(c, a.sigmoid, brows(m.gate, T), brows(m.gate, T));
        const RadOperand heads = RAD_P_T2(rad_buf_ptr(c, m.attn), RAD_BF16, T * g.n_head, g.head_dim);
        const RadOperand gates = RAD_P_T2(rad_buf_ptr(c, m.gate), RAD_BF16, T * g.n_head, 1);
        RAD_ISSUE_N(c, a.gate_heads, T * g.n_head, heads, gates, RAD_NONE, heads);
        a.tap_o.step(c, m.attn, T, m.tap_live);
        a.o.step(c, m.a_attn, m.h, T);
        if (a.reduce_attn) RAD_ISSUE_N(c, a.reduce_attn, T * g.n_embd, brows(m.h, T), RAD_NONE);
        RAD_ISSUE(c, a.widen_attn, brows(m.h, T), brows(m.delta, T));
        RAD_ISSUE(c, a.add_attn, brows(m.x, T), brows(m.delta, T), brows(m.x, T));
        RAD_ISSUE(c, a.fnorm, brows(m.x, T), RAD_W(a.fn), brows(m.h, T));
        a.tap_gu.step(c, m.h, T, m.tap_live);
        RAD_ISSUE(c, m.had, brows(m.h, T), brows(m.a_h.q8, T), brows(m.a_h.s8, T));
        a.gu.step(c, m.a_h, m.gu, T);
        RAD_ISSUE(c, a.act, bcol(m.gu, 0, g.n_ff, T), bcol(m.gu, g.n_ff, g.n_ff, T), brows(m.ff, T));
        a.tap_down.step(c, m.ff, T, m.tap_live);
        a.down.step(c, m.a_ff, m.h, T);
        if (a.reduce_ffn) RAD_ISSUE_N(c, a.reduce_ffn, T * g.n_embd, brows(m.h, T), RAD_NONE);
        RAD_ISSUE(c, a.widen_ffn, brows(m.h, T), brows(m.delta, T));
        RAD_ISSUE(c, a.add_ffn, brows(m.x, T), brows(m.delta, T), brows(m.x, T));
        /* The tap after this layer, on prefill and decode alike: the drafter conditions on every
         * committed token's residual stream, not only on prefill's. */
        m.taps.step(c, m.x, T, (int64_t)i, g.n_embd);
    }
    RAD_ISSUE(c, m.final_norm, brows(m.x, T), RAD_W(m.norm), brows(m.h, T));
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.gather, batch->n_out, brows(m.h, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.hout, batch->n_out));
        RAD_ISSUE_N(c, m.logits_op, batch->n_out, brows(m.hout, batch->n_out), RAD_W(m.head),
                    brows(m.logits, batch->n_out));
    }
    if (m.tap_live) {
        m.calib_rows += T;
        if (m.calib_rows >= calib_target()) {
            bool ok = true;
            for (int64_t l = m.calib_lo; l < m.calib_hi; ++l) {
                const Layer& a = m.layers[(size_t)l];
                char stem[96];
                for (const char* proj : {"attn_q", "attn_k", "attn_v"}) {
                    std::snprintf(stem, sizeof stem, "blk.%lld.%s", (long long)l, proj);
                    ok = calib_write(c, a.tap_in.h, a.tap_in.K, m.calib_rows, stem, 0) && ok;
                }
                std::snprintf(stem, sizeof stem, "blk.%lld.attn_output", (long long)l);
                ok = calib_write(c, a.tap_o.h, a.tap_o.K, m.calib_rows, stem, 0) && ok;
                std::snprintf(stem, sizeof stem, "blk.%lld.ffn_gate_up", (long long)l);
                ok = calib_write(c, a.tap_gu.h, a.tap_gu.K, m.calib_rows, stem, 0) && ok;
                std::snprintf(stem, sizeof stem, "blk.%lld.ffn_down", (long long)l);
                ok = calib_write(c, a.tap_down.h, a.tap_down.K, m.calib_rows, stem, 0) && ok;
            }
            m.tap_live = false;
            std::fprintf(stderr, "radiance: spark25 rank %d wrote calibration Grams for layers "
                                 "%lld:%lld after %lld tokens to %s%s\n", rad_rank(c),
                         (long long)m.calib_lo, (long long)m.calib_hi, m.calib_rows, calib_dir(),
                         ok ? "" : ", with errors");
        }
    }
}

#ifdef SPARK25_HAVE_PREFILL_SPLIT
static uint64_t pass_key(RadCtx* c, const RadBatch* batch, const int32_t* used_max_h) {
    if (!c || !batch) return 0;
    Model& m = models[rad_rank(c)];
    int group = -1;
    if (batch->kv) for (int i = 0; i < batch->n_kv_groups; ++i)
        if (batch->kv[i].group == m.kv) { group = i; break; }
    spark25_prefill_key::Request r;
    r.phase = batch->phase;
    r.n_tok = batch->n_tok;
    r.n_seq = batch->n_seq;
    r.max_q_len = batch->max_q_len;
    r.max_ctx = m.g.max_ctx;
    r.n_spec = batch->n_spec;
    r.draft_pass = batch->draft_pass;
    r.enc = batch->enc != 0;
    r.head_dim = m.g.head_dim;
    r.gqa = m.g.gqa();
    r.kv_heads = m.g.n_head_kv;
    r.kv_group_found = group >= 0;
    r.used = used_max_h && group >= 0 ? used_max_h[group] : 0;
    r.wgps = rad_ctx_wgp_count(c);
    return spark25_prefill_key::pass_key(r);
}
#endif

/* The speculation depth this container supports: zero without a drafter, or the block length a
 * DSpark drafter was trained at. `draft_depth_fixed` because a block-diffusion drafter denoises
 * exactly that many mask positions, and `declare` refuses to run it at any other depth. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->draft_depth = 0;
    out->draft_depth_fixed = 0;
    bool present = false;
    if (drafter_kind(meta, &present) != RAD_OK) return RAD_E_UNSUPPORTED;
    if (!present) return RAD_OK;
    const long long bs = rad_meta_geti(meta, "draft.block_size", 0);
    if (bs < 2) {
        fprintf(stderr, "radiance: spark25: the container carries a drafter with no "
                        "draft.block_size\n");
        return RAD_E_FORMAT;
    }
    out->draft_depth = (int)bs;
    out->draft_depth_fixed = 1;
    return RAD_OK;
}

static const RadChatFormat* chat_format(const RadModelMeta*) { return spark25_bf16::spark_chat_format(); }
}
#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(spark25_w8a8)
RAD_ARCH_PLUGIN(spark25_w8a8, "spark25", "w8a8_fwht128", "0.1.0",
                "Spark X2.6 with an int8 trunk: i8 weights with a bf16 scale a 128 of K, rotated "
                "by a 128-wide Hadamard, against int8 activations the same rotation quantises")
RAD_ARCH_CHAT_FORMAT(spark25_w8a8)
#ifdef SPARK25_HAVE_PREFILL_SPLIT
extern "C" uint64_t rad_arch_pass_key(RadCtx* c, const RadBatch* batch,
                                      const int32_t* used_max_h) {
    return spark25_w8a8::pass_key(c, batch, used_max_h);
}
#endif
#endif
