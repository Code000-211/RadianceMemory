/* bailing_hybrid -- Ling-3.0-flash and its fine-tunes (`model_type: bailing_hybrid`,
 * BailingMoeV3ForCausalLM), from the bf16 checkpoint, the routed experts as a recipe stores them
 * (data/recipes/ling3-w4a8-rtn.recipe, ling3-w4a8-gptq.recipe).
 *
 * ============================== WHAT IT SERVES ==============================
 *
 *   layers        42 in groups of six: five Kimi delta attention (KDA) layers, then one dense
 *                 latent attention (MLA) layer, so layers 5, 11, ..., 41 are MLA
 *   KDA           32 heads of 128, separate q/k/v projections each with its own causal
 *                 convolution, a full-rank per-channel forget gate (`f_proj`) and output gate
 *                 (`g_proj`), lower bound -5
 *   MLA           no query LoRA, a 512-wide latent and a 64-wide rotary part (interleaved pairs),
 *                 heads of 128 + 64, values of 128, a sigmoid gate per head
 *   feed-forward  layers 0-1 dense (6144); from layer 2, 512 experts of 768 at top-8 under
 *                 group-limited sigmoid routing (8 groups, 4 kept, a correction bias, x2.5), an
 *                 ungated shared expert, and a SwiGLU clipped after the SiLU in the last layers
 *                 (`expert_swiglu_limit_list`, `share_expert_swiglu_limit_list`, as vLLM and
 *                 SGLang serve it; HF's reference ignores it)
 *
 * The KDA core is GLM-5.3's (KdaState), the router GLM-5.3's with groups (GlmRouter), the
 * feed-forward MoeFP8. The KDA and MLA blocks around them are in this file.
 *
 * ============================== THE LATENT ATTENTION ==============================
 *
 * The MLA attends in the latent. The query projection is served as two, its heads' rotary rows and
 * their nope rows (row maps over q_proj), so each is a plain [T, heads * width] plane: `rope` turns
 * the first, and the second is absorbed into the latent through kv_b's key rows (mla_project).
 * `mla_attn_paged` scores against a cache row of [normed latent 512 | rotated key 64] and returns
 * the latent, gated per head; kv_b's value rows expand it. The cache row is stored as one KV head
 * of 288, the latent's first 288 values in the K slot and the rest in the V slot.
 *
 * ============================== THE DRAFTER ==============================
 *
 * The MTP layer (checkpoint layer 42: an MLA layer with the full routed feed-forward) is a serial
 * drafter around GlmMtp, declared when the deployment speculates.
 *
 * docs/ARCHITECTURES.md describes the model and its traps. */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <arch/rad_arch.h>
#include <arch/rad_block_glm_kda.h>
#include <arch/rad_block_glm_mtp.h>
#include <arch/rad_block_mlp_fp8.h>
#include <arch/rad_block_moe_fp8.h>

namespace bailing_hybrid {

using namespace rad::arch;

/* The released geometry this graph is written for; a checkpoint that differs is refused by name. */
struct Shape {
    int64_t n_embd = 0, n_layer = 0, heads = 0, head_dim = 0, nope = 0, rope = 0, v_dim = 0,
            latent = 0, experts = 0, top_k = 0, groups = 0, topk_groups = 0, ff_exp = 0,
            ff_shared = 0, ff_dense = 0, dense_layers = 0, group_size = 0, conv = 0;
    double lower = -5, route_scale = 1;
    std::vector<double> lim_exp, lim_shared;
};

/* A KDA layer: the q/k/v, beta, forget and output-gate projections into KdaState, then o_proj and
 * its all-reduce. */
struct LingKda {
    struct Wire {
        ActFP8 input{};
        rad_buf out = 0;
        ActFP8 core_out{};
        KdaState::Wire core{};
    };
    Geom g{};
    Wire w{};
    KdaState core{};
    LinearFP8 qkv{}, beta{}, forget{}, gate{}, output{};
    QuantFP8 out_quant{};
    rad_op op_ar = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& geom, const Shape& sh, int l,
                const Wire& wire, const char* pre, rad_kvgroup state, rad_kvgroup history) {
        g = geom;
        w = wire;
        const int64_t heads = sh.heads / g.world, width = heads * sh.head_dim;
        const auto grp = grp_layer(l);
        RAD_ARCH_TRY(qkv.declare(b, nm, g, nm.f("blk.%d.attn_qkv", l), 3 * width, g.n_embd,
                                 RAD_SHARD_ROW, grp,
                                 { nm.ckpt("%s.q_proj.weight", pre), nm.ckpt("%s.k_proj.weight", pre),
                                   nm.ckpt("%s.v_proj.weight", pre) },
                                 w.input, w.core.qkv, 0, { width, width, width }));
        RAD_ARCH_TRY(beta.declare(b, nm, g, nm.f("blk.%d.attn_beta", l), heads, g.n_embd,
                                  RAD_SHARD_ROW, grp, { nm.ckpt("%s.b_proj.weight", pre) },
                                  w.input, w.core.beta));
        RAD_ARCH_TRY(forget.declare(b, nm, g, nm.f("blk.%d.attn_forget", l), width, g.n_embd,
                                    RAD_SHARD_ROW, grp, { nm.ckpt("%s.f_proj.weight", pre) },
                                    w.input, w.core.a));
        RAD_ARCH_TRY(gate.declare(b, nm, g, nm.f("blk.%d.attn_gate", l), width, g.n_embd,
                                  RAD_SHARD_ROW, grp, { nm.ckpt("%s.g_proj.weight", pre) },
                                  w.input, w.core.gate));
        KdaState::Config kc;
        kc.heads = heads;
        kc.dim = sh.head_dim;
        kc.width = sh.conv;
        kc.lower = sh.lower;
        KdaState::Src ks{ nm.ckpt("%s.q_conv1d.weight", pre), nm.ckpt("%s.k_conv1d.weight", pre),
                          nm.ckpt("%s.v_conv1d.weight", pre), nm.ckpt("%s.A_log", pre),
                          nm.ckpt("%s.dt_bias", pre),         nm.ckpt("%s.o_norm.weight", pre) };
        RAD_ARCH_TRY(core.declare(b, nm, g, l, state, history, kc, w.core, ks));
        const char* on = nm.f("blk.%d.attn_output", l);
        RAD_ARCH_TRY(output.probe(b, nm, g, on, { nm.ckpt("%s.o_proj.weight", pre) }));
        if (!output.reads_x) RAD_ARCH_TRY(out_quant.declare(b, g, w.core_out, width));
        RAD_ARCH_TRY(output.declare(b, nm, g, on, g.n_embd, width, RAD_SHARD_COL, grp,
                                    { nm.ckpt("%s.o_proj.weight", pre) }, w.core_out, w.out));
        op_ar = GlmTp::reduce(b, g, w.out);
        return g.world > 1 && !op_ar ? RAD_E_INVAL : RAD_OK;
    }
    void step(RadCtx* c, const RadBatch* batch) const {
        const int64_t T = batch->n_tok;
        qkv.step(c, w.input, w.core.qkv, T);
        beta.step(c, w.input, w.core.beta, T);
        forget.step(c, w.input, w.core.a, T);
        gate.step(c, w.input, w.core.gate, T);
        core.step(c, batch);
        out_quant.step(c, T);
        output.step(c, w.core_out, w.out, T);
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.out, T), RAD_NONE);
    }
};

/* A dense MLA layer, attending in the latent (the file's head comment). The latent cache is
 * replicated: every rank projects and stores the whole row, and splits the heads. */
struct LingMla {
    struct Wire {
        ActFP8 input{};
        rad_buf q_rope = 0;   /* [T, heads * rope] */
        rad_buf q_nope = 0;   /* [T, heads * nope] */
        rad_buf row = 0;      /* [T, latent + rope], the cache row */
        rad_buf q_lat = 0;    /* [T, heads, latent] */
        rad_buf gate = 0;     /* [T, heads] */
        rad_buf attn = 0;     /* [T, heads, latent] */
        ActFP8 value{};       /* [T, heads * v] */
        rad_buf out = 0;
    };
    Geom g{};
    Shape s{};
    Wire w{};
    int layer = 0;
    int64_t heads = 0, qw = 0, cw = 0;
    rad_kvgroup kv = 0;
    LinearFP8 q_rope_proj{}, q_nope_proj{}, kv_a{}, g_proj{}, output{};
    QuantFP8 value_quant{};
    rad_weight kv_gain = 0, kv_b = 0;
    rad_op op_kv_norm = 0, op_q_rope = 0, op_k_rope = 0, op_store = 0, op_absorb = 0, op_attn = 0,
           op_expand = 0, op_ar = 0;
    int64_t block = 0;

    static constexpr int64_t kBlock = 16;

    int declare(RadBuilder* b, Names& nm, const Geom& geom, const Shape& shape, int l,
                const Wire& wire, const char* pre, rad_kvgroup group) {
        g = geom;
        s = shape;
        w = wire;
        layer = l;
        kv = group;
        heads = s.heads / g.world;
        qw = s.rope + s.nope;
        cw = s.latent + s.rope;
        const auto grp = grp_layer(l);
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, kv));
        /* One row map a part, a piece a head: [h * (nope + rope) + off, + width). Successive ROWS
         * maps for one name extend it in call order (rad_builder.h), eight pieces a map. */
        const char* qsrc = nm.ckpt("%s.q_proj.weight", pre);
        auto part = [&](const char* name, int64_t off, int64_t width) -> int {
            for (int64_t h0 = 0; h0 < s.heads; h0 += 8) {
                RadNameMap m = map_base(name, RAD_MAP_ROWS);
                for (int64_t h = h0; h < std::min(h0 + 8, s.heads); ++h) {
                    m.src[m.n_src] = qsrc;
                    m.src_index[m.n_src] = (int)(h * qw + off);
                    m.src_rows[m.n_src] = width;
                    ++m.n_src;
                }
                RAD_ARCH_TRY(rad_decl_name_map(b, &m));
            }
            return RAD_OK;
        };
        RAD_ARCH_TRY(part(nm.f("blk.%d.attn_q_rope.weight", l), s.nope, s.rope));
        RAD_ARCH_TRY(part(nm.f("blk.%d.attn_q_nope.weight", l), 0, s.nope));
        q_rope_proj.mapped = q_nope_proj.mapped = true;
        RAD_ARCH_TRY(q_rope_proj.declare(b, nm, g, nm.f("blk.%d.attn_q_rope", l), heads * s.rope,
                                         g.n_embd, RAD_SHARD_ROW, grp, { qsrc }, w.input, w.q_rope));
        RAD_ARCH_TRY(q_nope_proj.declare(b, nm, g, nm.f("blk.%d.attn_q_nope", l), heads * s.nope,
                                         g.n_embd, RAD_SHARD_ROW, grp, { qsrc }, w.input, w.q_nope));
        RAD_ARCH_TRY(kv_a.declare(b, nm, g, nm.f("blk.%d.attn_kv_a", l), cw, g.n_embd,
                                  RAD_SHARD_NONE, grp, { nm.ckpt("%s.kv_a_proj_with_mqa.weight", pre) },
                                  w.input, w.row));
        const char* kn = nm.f("blk.%d.attn_kv_a_norm.weight", l);
        RAD_ARCH_TRY(map_copy(b, kn, nm.ckpt("%s.kv_a_layernorm.weight", pre)));
        kv_gain = decl_w(b, kn, RAD_F32, { s.latent }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        if (!kv_gain) return RAD_E_INVAL;
        op_kv_norm = rw(b,
                        RAD_OP(b, "rmsnorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", s.latent),
                                          RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(kv_gain)),
                        { w.row }, { w.row });
        op_q_rope = rw(b, decl_rope(b, g, g.max_tok, s.rope, heads, 0, s.rope, "gptj"), { w.q_rope },
                       { w.q_rope });
        op_k_rope = rw(b, decl_rope(b, g, g.max_tok, s.rope, 1, 0, s.rope, "gptj"), { w.row },
                       { w.row });
        /* The first op after the group to carry a block size claims it (rad_builder.cpp,
         * find_kv_consumer). */
        op_store = rw(b,
                      RAD_OP(b, "kv_store",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cw / 2),
                                        RAD_INT("n_head_kv", 1), RAD_INT("block_size", kBlock),
                                        RAD_STR("kv_dtype", "bf16")),
                             RAD_NOWEIGHTS),
                      { w.row }, {});
        const char* bn = nm.f("blk.%d.attn_kv_b.weight", l);
        RAD_ARCH_TRY(map_copy(b, bn, nm.ckpt("%s.kv_b_proj.weight", pre)));
        kv_b = decl_w(b, bn, RAD_BF16, { heads * (s.nope + s.v_dim), s.latent }, RAD_ACCESS_PER_TOKEN,
                      RAD_SHARD_ROW, grp);
        if (!op_kv_norm || !op_q_rope || !op_k_rope || !op_store || !kv_b) return RAD_E_INVAL;
        op_absorb = rw(b,
                       RAD_OP(b, "mla_project",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("heads", heads),
                                         RAD_INT("head_dim", s.nope), RAD_INT("latent_dim", s.latent),
                                         RAD_STR("mode", "absorb"), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(kv_b)),
                       { w.q_nope }, { w.q_lat });
        RAD_ARCH_TRY(g_proj.declare(b, nm, g, nm.f("blk.%d.attn_head_gate", l), heads, g.n_embd,
                                    RAD_SHARD_ROW, grp, { nm.ckpt("%s.g_proj.weight", pre) },
                                    w.input, w.gate));
        op_attn = rw(b,
                     RAD_OP(b, "mla_attn_paged",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("heads", heads),
                                       RAD_INT("latent_dim", s.latent), RAD_INT("rope_dim", s.rope),
                                       RAD_INT("block_size", kBlock),
                                       RAD_F64("scale", 1.0 / std::sqrt((double)qw)),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                     { w.q_lat, w.q_rope, w.gate }, { w.attn });
        op_expand = rw(b,
                       RAD_OP(b, "mla_project",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("heads", heads),
                                         RAD_INT("head_dim", s.v_dim), RAD_INT("latent_dim", s.latent),
                                         RAD_STR("mode", "expand"), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(kv_b)),
                       { w.attn }, { w.value.x });
        if (!op_absorb || !op_attn || !op_expand) return RAD_E_INVAL;
        block = rad_kv_block_size(b, kv);
        if (block != kBlock) {
            fprintf(stderr, "radiance: bailing_hybrid: the latent cache's block is %lld tokens and "
                            "its ops are declared at %lld\n", (long long)block, (long long)kBlock);
            return RAD_E_INVAL;
        }
        const char* on = nm.f("blk.%d.attn_output", l);
        RAD_ARCH_TRY(output.probe(b, nm, g, on, { nm.ckpt("%s.dense.weight", pre) }));
        if (!output.reads_x) RAD_ARCH_TRY(value_quant.declare(b, g, w.value, heads * s.v_dim));
        RAD_ARCH_TRY(output.declare(b, nm, g, on, g.n_embd, heads * s.v_dim, RAD_SHARD_COL, grp,
                                    { nm.ckpt("%s.dense.weight", pre) }, w.value, w.out));
        op_ar = GlmTp::reduce(b, g, w.out);
        return g.world > 1 && !op_ar ? RAD_E_INVAL : RAD_OK;
    }
    void step(RadCtx* c, const RadBatch* batch) const {
        const auto* kb = kv_batch(batch, kv);
        if (!kb || !kb->slot_mapping || !kb->block_table || !kb->seqused) {
            rad_step_fail(c, "bailing_hybrid MLA requires the latent cache's slot mapping, block "
                             "table and sequence lengths");
            return;
        }
        const int64_t T = batch->n_tok, S = batch->n_seq;
        const RadOperand pos = praw(batch->positions, RAD_I32, T);
        q_rope_proj.step(c, w.input, w.q_rope, T);
        q_nope_proj.step(c, w.input, w.q_nope, T);
        kv_a.step(c, w.input, w.row, T);
        g_proj.step(c, w.input, w.gate, T);
        RAD_ISSUE_N(c, op_kv_norm, T, bcol(w.row, 0, s.latent, T), RAD_W(kv_gain),
                    bcol(w.row, 0, s.latent, T));
        RAD_ISSUE_N(c, op_q_rope, T, brows(w.q_rope, T), pos);
        RAD_ISSUE_N(c, op_k_rope, T, bcol(w.row, s.latent, s.rope, T), pos);
        RAD_ISSUE_N(c, op_store, T, bcol(w.row, 0, cw / 2, T), bcol(w.row, cw / 2, cw / 2, T),
                    praw(kb->slot_mapping, RAD_I32, T), kv_cache(kv, layer));
        RAD_ISSUE_N(c, op_absorb, T, brows(w.q_nope, T), RAD_W(kv_b), brows(w.q_lat, T));
        RAD_ISSUE_N(c, op_attn, T, brows(w.q_lat, T), brows(w.q_rope, T), kv_cache(kv, layer),
                    praw2(kb->block_table, RAD_I32, S, kb->block_table_pitch),
                    praw(kb->seqused, RAD_I32, S), praw(batch->cu_seqlens, RAD_I32, S + 1),
                    brows(w.gate, T), brows(w.attn, T));
        RAD_ISSUE_N(c, op_expand, T, brows(w.attn, T), RAD_W(kv_b), brows(w.value.x, T));
        value_quant.step(c, T);
        output.step(c, w.value, w.out, T);
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.out, T), RAD_NONE);
    }
};

struct Layer {
    bool mla = false, dense = false;
    rad_weight gain = 0;
    NormQuantFP8 norm{};
    LingKda kda{};
    LingMla attn{};
    rad_op op_add = 0;
    MlpFP8 mlp{};
    MoeFP8 moe{};
};

struct Model {
    Names nm{};
    Geom g{};
    Shape s{};
    std::vector<Layer> layers;
    rad_kvgroup latent = 0, state = 0, conv = 0;
    rad_buf x = 0, y = 0, final_hidden = 0, logit_input = 0, logits = 0;
    ActFP8 act{};
    LingKda::Wire kw{};
    LingMla::Wire aw{};
    MlpFP8::Wire dw{};
    MoeFP8::Wire mw{};
    rad_weight token = 0, head = 0, final_gain = 0;
    rad_op embed = 0, embed_ar = 0, norm = 0, gather = 0, head_op = 0;
    /* The draft head and its layer, declared only when the deployment speculates. */
    bool have_mtp = false;
    GlmMtp mtp{};
    Layer draft{};
};

static Model g_model[MAX_RANKS];

static int64_t cfg(const RadModelMeta* m, const char* key, int64_t fallback) {
    return rad_meta_geti(m, key, fallback);
}

static std::vector<double> cfg_list(const RadModelMeta* m, const char* key) {
    std::vector<double> v;
    for_each_token(rad_meta_gets(m, key, ""), [&](const char* t, size_t) {
        v.push_back(std::strtod(t, nullptr));
        return true;
    });
    return v;
}

/* Layer l's SwiGLU clamp from a per-layer list; 0 (none) past its end or where it says 0. */
static double limit_at(const std::vector<double>& v, int64_t l) {
    return l < (int64_t)v.size() && v[(size_t)l] > 0 ? v[(size_t)l] : 0;
}

/* The last layer of each group of `layer_group_size` is MLA, and so is every layer of an
 * incomplete last group. */
static bool is_mla(const Shape& s, int64_t l) {
    return (l + 1) % s.group_size == 0 || l >= s.n_layer / s.group_size * s.group_size;
}

static int read_shape(const RadModelMeta* meta, Shape& s) {
    s.n_embd = meta->n_embd;
    s.n_layer = meta->n_layers;
    s.heads = cfg(meta, "num_attention_heads", 0);
    s.head_dim = cfg(meta, "head_dim", 0);
    s.nope = cfg(meta, "qk_nope_head_dim", 0);
    s.rope = cfg(meta, "qk_rope_head_dim", 0);
    s.v_dim = cfg(meta, "v_head_dim", 0);
    s.latent = cfg(meta, "kv_lora_rank", 0);
    s.experts = cfg(meta, "num_experts", 0);
    s.top_k = cfg(meta, "num_experts_per_tok", 0);
    s.groups = cfg(meta, "n_group", 1);
    s.topk_groups = cfg(meta, "topk_group", s.groups);
    s.ff_exp = cfg(meta, "moe_intermediate_size", 0);
    s.ff_shared = cfg(meta, "moe_shared_expert_intermediate_size", 0) *
                  cfg(meta, "num_shared_experts", 0);
    s.ff_dense = cfg(meta, "intermediate_size", 0);
    s.dense_layers = cfg(meta, "first_k_dense_replace", 0);
    s.group_size = cfg(meta, "layer_group_size", 0);
    s.conv = cfg(meta, "short_conv_kernel_size", 0);
    s.lower = rad_meta_getf(meta, "kda_lower_bound", 0.0);
    s.route_scale = rad_meta_getf(meta, "routed_scaling_factor", 0.0);
    s.lim_exp = cfg_list(meta, "expert_swiglu_limit_list");
    s.lim_shared = cfg_list(meta, "share_expert_swiglu_limit_list");
    const struct {
        const char* what;
        bool ok;
    } need[] = {
        { "head_dim 128 (KDA)", s.head_dim == 128 },
        { "qk_nope_head_dim == v_head_dim (mla_project's head)", s.nope > 0 && s.nope == s.v_dim },
        { "an even qk_rope_head_dim", s.rope > 0 && !(s.rope & 1) },
        { "an even kv_lora_rank + qk_rope_head_dim", s.latent > 0 && !((s.latent + s.rope) & 1) },
        { "q_lora_rank null", cfg(meta, "q_lora_rank", 0) == 0 },
        { "no_kda_lora", cfg(meta, "no_kda_lora", 0) != 0 },
        { "kda_safe_gate", cfg(meta, "kda_safe_gate", 1) != 0 },
        { "a negative kda_lower_bound", s.lower < 0 && std::isfinite(s.lower) },
        { "rope_interleave", cfg(meta, "rope_interleave", 0) != 0 },
        { "head-wise gated attention",
          !std::strcmp(rad_meta_gets(meta, "gated_attention_proj_granularity_type", ""),
                       "head_wise") },
        { "sigmoid scoring", !std::strcmp(rad_meta_gets(meta, "score_function",
                                                        rad_meta_gets(meta, "scoring_func", "")),
                                          "sigmoid") },
        { "one shared expert", cfg(meta, "num_shared_experts", 0) == 1 },
        { "a layer group", s.group_size > 1 },
        { "a 4-tap short convolution", s.conv == 4 },
        { "experts, top-k and groups", s.experts > 0 && s.top_k > 0 && s.groups >= 1 &&
                                           s.experts % s.groups == 0 && s.topk_groups >= 1 &&
                                           s.topk_groups <= s.groups },
        { "a positive routed_scaling_factor", s.route_scale > 0 },
        { "no rope scaling", !rad_meta_gets(meta, "rope_scaling.type", nullptr) &&
                                 !rad_meta_gets(meta, "rope_scaling.rope_type", nullptr) },
    };
    for (const auto& n : need)
        if (!n.ok) {
            fprintf(stderr, "radiance: bailing_hybrid: this plugin serves %s, and the checkpoint "
                            "differs\n", n.what);
            return RAD_E_UNSUPPORTED;
        }
    return RAD_OK;
}

/* A rank's slice of a width of `blocks` 128-column blocks: blocks [lo, hi) in the order of the
 * ranks starting at `first`, the first `blocks % world` ranks of that order one block wider. */
static void slice_of(int64_t blocks, int world, int rank, int first, int64_t* lo, int64_t* hi) {
    const int p = ((rank - first) % world + world) % world;
    const int64_t base = blocks / world, extra = blocks % world;
    *lo = p * base + std::min<int64_t>(p, extra);
    *hi = *lo + base + (p < extra ? 1 : 0);
}

/* ============================== The trunk's buffers ============================== */

static int buffers(RadBuilder* b, Model& m) {
    auto& g = m.g;
    auto& s = m.s;
    auto& nm = m.nm;
    const int64_t T = g.max_tok, E = g.n_embd, H = s.heads / g.world, W = H * s.head_dim;
    auto bf = [&](const char* n, std::initializer_list<int64_t> shape) {
        return decl_b(b, nm.f("%s", n), RAD_BF16, shape);
    };
    m.x = bf("residual", { T, E });
    m.y = bf("attn_delta", { T, E });
    m.act.x = bf("normed", { T, E });
    RAD_ARCH_TRY(m.act.declare_qs(b, nm, g, "normed", E));
    m.final_hidden = decl_b(b, nm.f("final_hidden"), RAD_BF16, { T, E }, RAD_BUF_PERSIST);
    m.logit_input = bf("logit_input", { g.max_logit_rows, E });
    m.logits = decl_b(b, nm.f("logits"), RAD_F32, { g.max_logit_rows, g.n_vocab });
    RAD_ARCH_TRY(rad_declare_logits(b, m.logits));

    auto& kw = m.kw;
    kw.input = m.act;
    kw.out = m.y;
    kw.core.qkv = bf("kda_qkv", { T, 3 * W });
    kw.core.conv = bf("kda_convolved", { T, 3 * W });
    kw.core.a = bf("kda_forget", { T, W });
    kw.core.beta = bf("kda_beta", { T, H });
    kw.core.gate = bf("kda_gate", { T, W });
    kw.core.out = bf("kda_out", { T, W });
    kw.core_out.x = kw.core.out;
    RAD_ARCH_TRY(kw.core_out.declare_qs(b, nm, g, "kda_out", W));

    auto& aw = m.aw;
    aw.input = m.act;
    aw.out = m.y;
    aw.q_rope = bf("mla_q_rope", { T, H * s.rope });
    aw.q_nope = bf("mla_q_nope", { T, H * s.nope });
    aw.row = bf("mla_row", { T, s.latent + s.rope });
    aw.q_lat = bf("mla_q_latent", { T, H * s.latent });
    aw.gate = bf("mla_gate", { T, H });
    aw.attn = bf("mla_attn_latent", { T, H * s.latent });
    aw.value.x = bf("mla_value", { T, H * s.v_dim });
    RAD_ARCH_TRY(aw.value.declare_qs(b, nm, g, "mla_value", H * s.v_dim));

    auto& dw = m.dw;
    dw.x = m.x;
    dw.h = m.act;
    dw.gate_up = bf("dense_gate_up", { T, 2 * g.n_ff });
    dw.ffn.x = bf("dense_ff", { T, g.n_ff });
    RAD_ARCH_TRY(dw.ffn.declare_qs(b, nm, g, "dense_ff", g.n_ff));
    return RAD_OK;
}

static int moe_buffers(RadBuilder* b, Model& m, const MoeFP8::Config& mc) {
    auto& g = m.g;
    auto& nm = m.nm;
    const int64_t T = g.max_tok, E = g.n_embd, F = mc.n_ff_exp, k = mc.top_k;
    auto bf = [&](const char* n, std::initializer_list<int64_t> shape) {
        return decl_b(b, nm.f("%s", n), RAD_BF16, shape);
    };
    auto& mw = m.mw;
    mw.x = m.x;
    mw.h = m.act;
    mw.logits = decl_b(b, nm.f("router_logits"), RAD_F32, { T, mc.n_expert_all });
    mw.ids = decl_b(b, nm.f("expert_ids"), RAD_I32, { T, k });
    mw.ew = decl_b(b, nm.f("expert_weights"), RAD_F32, { T, k });
    mw.eoff = decl_b(b, nm.f("expert_offsets"), RAD_I32, { mc.n_expert + 1 });
    mw.ecnt = decl_b(b, nm.f("expert_counts"), RAD_I32, { mc.n_expert });
    mw.sorted = decl_b(b, nm.f("expert_sorted"), RAD_I32, { T * k });
    mw.egu = bf("expert_gate_up", { T * k, 2 * F });
    mw.eff.x = bf("expert_ff", { T * k, F });
    RAD_ARCH_TRY(mw.eff.declare_qs(b, nm, g, "expert_ff", F, T * k));
    mw.edn = bf("expert_down", { T * k, E });
    if (mc.expert_rot) {
        mw.hr.q = decl_b(b, nm.f("normed_rot.q"), RAD_F8E4M3, { T, E });
        mw.hr.s = decl_b(b, nm.f("normed_rot.s"), RAD_F32, { T, E / RAD_FP8_BLOCK });
        if (!mw.hr.q || !mw.hr.s) return RAD_E_INVAL;
    }
    if (mc.n_ff_shared > 0) {
        mw.sgu = bf("shared_gate_up", { T, 2 * mc.n_ff_shared });
        mw.sff.x = bf("shared_ff", { T, mc.n_ff_shared });
        RAD_ARCH_TRY(mw.sff.declare_qs(b, nm, g, "shared_ff", mc.n_ff_shared));
        mw.sout = bf("shared_out", { T, E });
        mw.sgate = bf("shared_one", { T, 1 });
    }
    return RAD_OK;
}

/* ============================== declare ============================== */

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->is_draft || ctx->rank < 0 || ctx->rank >= MAX_RANKS)
        return RAD_E_INVAL;
    const int world = ctx->world_size ? ctx->world_size : 1;
    /* A sizing declare writes scratch: the step reads g_model, filled by the real declare at the
     * real max_tok (RadBuildCtx::shape_probe). */
    static thread_local Model probe_model;
    Model& m = ctx->shape_probe ? probe_model : g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    auto& g = m.g;
    auto& s = m.s;
    auto& nm = m.nm;
    RAD_ARCH_TRY(read_shape(meta, s));
    if (world != 1 && world != 2 && world != 4) {
        fprintf(stderr, "radiance: bailing_hybrid: --tp %d; the head and expert splits are built "
                        "for 1, 2 or 4 ranks\n", world);
        return RAD_E_UNSUPPORTED;
    }
    if (s.heads % world || s.ff_exp % RAD_FP8_BLOCK || s.ff_shared % RAD_FP8_BLOCK ||
        s.ff_exp / RAD_FP8_BLOCK < world) {
        fprintf(stderr, "radiance: bailing_hybrid: %lld heads and experts of %lld do not split "
                        "over %d ranks in whole heads and 128-column blocks\n",
                (long long)s.heads, (long long)s.ff_exp, world);
        return RAD_E_UNSUPPORTED;
    }
    if (ctx->max_spec > 31) {
        fprintf(stderr, "radiance: bailing_hybrid: the KDA verify pass takes at most 32 rows a "
                        "sequence, so --num-speculative-tokens is at most 31\n");
        return RAD_E_UNSUPPORTED;
    }
    if (ctx->kv_dtype && ctx->kv_dtype != RAD_BF16) {
        fprintf(stderr, "radiance: bailing_hybrid: the latent and KDA caches are bf16; run with "
                        "--kv-cache-dtype bf16\n");
        return RAD_E_UNSUPPORTED;
    }
    RAD_ARCH_TRY(geom_from(g, meta, ctx, "bf16", RAD_BF16));
    g.head_dim = s.head_dim;

    /* ---- the routed experts' slice of this rank: whole at one rank, halves at two, and at four
     * the six 128-column blocks of an expert split 2, 2, 1, 1 over the ranks, rotated by the
     * expert's number mod 4, so every rank carries six blocks per four experts (moe_gemm_q_mod4).
     * The shared expert splits the same way, unrotated. */
    MoeFP8::Config mc;
    mc.n_expert = mc.n_expert_all = s.experts;
    mc.top_k = s.top_k;
    mc.sigmoid_router = 1;
    mc.route_scale = s.route_scale;
    mc.norm_topk = cfg(meta, "norm_topk_prob", 1) != 0;
    mc.n_group = s.groups;
    mc.topk_group = s.topk_groups;
    mc.shared_ungated = 1;
    mc.swiglu_step = 1;
    mc.rows = g.max_tok;
    /* GPTQ calibration (RADIANCE_CALIB_DIR): the tap writes each expert's down Hessian over its
     * whole width, so a calibration run gives every rank whole experts (below). */
    mc.calib = calib_dir() != nullptr ? 1 : 0;
    mc.probed_expert = (int64_t)g.rank * (s.experts / world);
    const int64_t eb = s.ff_exp / RAD_FP8_BLOCK, sb = s.ff_shared / RAD_FP8_BLOCK;
    if (world == 4) {
        int64_t widest = 0;
        for (int q = 0; q < 4; ++q) {
            int64_t lo, hi;
            slice_of(eb, world, g.rank, q, &lo, &hi);
            mc.ff_lo4[q] = lo * RAD_FP8_BLOCK;
            mc.ff_hi4[q] = hi * RAD_FP8_BLOCK;
            widest = std::max(widest, mc.ff_hi4[q] - mc.ff_lo4[q]);
        }
        mc.n_ff_exp = widest;
    } else if (eb % world == 0) {
        mc.n_ff_exp = s.ff_exp / world;
    } else {
        fprintf(stderr, "radiance: bailing_hybrid: experts of %lld do not halve into 128-column "
                        "blocks\n", (long long)s.ff_exp);
        return RAD_E_UNSUPPORTED;
    }
    if (sb % world == 0) {
        mc.n_ff_shared = s.ff_shared / world;
    } else {
        int64_t lo, hi;
        slice_of(sb, world, g.rank, 0, &lo, &hi);
        mc.shared_lo = lo * RAD_FP8_BLOCK;
        mc.shared_hi = hi * RAD_FP8_BLOCK;
        mc.n_ff_shared = mc.shared_hi - mc.shared_lo;
    }
    if (s.ff_dense % (world * RAD_FP8_BLOCK)) {
        fprintf(stderr, "radiance: bailing_hybrid: the dense feed-forward's %lld columns do not "
                        "split over %d ranks in 128-column blocks\n", (long long)s.ff_dense, world);
        return RAD_E_UNSUPPORTED;
    }
    g.n_ff_all = s.ff_dense;
    g.n_ff = s.ff_dense / world;

    /* ---- each routed layer's expert format, read before any buffer is declared: rotated experts
     * need the rotated activation pair, sized here. The MTP layer follows the trunk's. */
    const bool mtp = ctx->max_spec > 0 && cfg(meta, "num_nextn_predict_layers", 0) == 1;
    std::vector<MoeFP8::Src> msrc((size_t)s.n_layer + 1);
    for (int l = (int)s.dense_layers; l < (int)s.n_layer + (mtp ? 1 : 0); ++l) {
        const char* pre = nm.ckpt("model.layers.%d", l);
        MoeFP8::Src& ms = msrc[(size_t)l];
        ms.cfg = mc;
        ms.cfg.swiglu_limit = limit_at(s.lim_exp, l);
        ms.cfg.shared_swiglu_limit = limit_at(s.lim_shared, l);
        ms.norm = nm.ckpt("%s.post_attention_layernorm.weight", pre);
        ms.router = nm.ckpt("%s.mlp.gate.weight", pre);
        ms.router_bias = nm.ckpt("%s.mlp.gate.expert_bias", pre);
        ms.experts = nm.ckpt("%s.mlp.experts", pre);
        ms.shared = nm.ckpt("%s.mlp.shared_experts", pre);
        RAD_ARCH_TRY(MoeFP8::probe(b, nm, l, ms, &ms.cfg));
        if (ms.cfg.expert_rot) mc.expert_rot = 1;
        if (ms.cfg.expert_bf16) mc.expert_bf16 = 1;
    }
    /* ---- whole experts per rank for a calibration run and for bf16 experts (a slice of a bf16
     * expert is not one container run), as qwen4exp does: rank r owns experts [r E/W, (r+1) E/W).
     * Served to calibrate and compare, not for decode speed. */
    if (world > 1 && (mc.calib || mc.expert_bf16)) {
        int64_t per = 0;
        RAD_ARCH_TRY(divide_or_fail(s.experts, world, "num_experts", &per));
        for (int l = (int)s.dense_layers; l < (int)s.n_layer + (mtp ? 1 : 0); ++l) {
            auto& c = msrc[(size_t)l].cfg;
            c.n_expert = per;
            c.expert_base = (int64_t)g.rank * per;
            c.n_ff_exp = s.ff_exp;
            for (int q = 0; q < 4; ++q) c.ff_lo4[q] = c.ff_hi4[q] = 0;
        }
        mc.n_expert = per;
        mc.expert_base = (int64_t)g.rank * per;
        mc.n_ff_exp = s.ff_exp;
        rad_note(b, "%s: rank %d owns experts [%lld, %lld) of %lld whole",
                 mc.calib ? "calibration (each expert's Hessian is over its whole width)"
                          : "bf16 experts",
                 g.rank, (long long)mc.expert_base, (long long)(mc.expert_base + per),
                 (long long)s.experts);
    }
    RAD_ARCH_TRY(buffers(b, m));
    RAD_ARCH_TRY(moe_buffers(b, m, mc));

    /* ---- the caches: the MLA layers' latent rows, the KDA layers' state and convolution
     * history. */
    {
        RadKVGroupDecl d{};
        d.kind = RAD_KV_FULL;
        d.dtype = RAD_BF16;
        d.n_head_kv = 1;
        d.head_dim = (s.latent + s.rope) / 2;
        m.latent = rad_decl_kv_group(b, nm.f("kv_latent"), &d);
        d = {};
        d.kind = RAD_KV_LINEAR;
        d.dtype = RAD_F32;
        d.n_head_kv = s.heads / world;
        d.state_dim[0] = d.state_dim[1] = s.head_dim;
        m.state = rad_decl_kv_group(b, nm.f("kv_kda"), &d);
        d = {};
        d.kind = RAD_KV_CONV;
        d.dtype = RAD_BF16;
        d.n_head_kv = 1;
        d.head_dim = 3 * (s.heads / world) * s.head_dim;
        d.conv_width = s.conv;
        m.conv = rad_decl_kv_group(b, nm.f("kv_kda_conv"), &d);
        if (!m.latent || !m.state || !m.conv) return RAD_E_INVAL;
    }

    /* ---- the embedding, the layers, the final norm and the head. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("model.word_embeddings.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output.weight"), nm.ckpt("lm_head.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("model.norm.weight")));
    m.token = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, { g.n_vocab, g.n_embd },
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.head = decl_w(b, nm.f("output.weight"), RAD_BF16, { g.n_vocab, g.n_embd }, RAD_ACCESS_VOCAB,
                    RAD_SHARD_ROW, grp_model());
    m.final_gain = decl_w(b, nm.f("output_norm.weight"), RAD_F32, { g.n_embd },
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.embed = rw(b,
                 RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16"),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.token)),
                 {}, { m.x });
    m.embed_ar = GlmTp::reduce(b, g, m.x, true);

    m.layers.resize((size_t)s.n_layer);
    for (int l = 0; l < (int)s.n_layer; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.mla = is_mla(s, l);
        lay.dense = l < s.dense_layers;
        const char* pre = nm.ckpt("model.layers.%d", l);
        const char* nn = nm.f("blk.%d.attn_norm.weight", l);
        RAD_ARCH_TRY(map_copy(b, nn, nm.ckpt("%s.input_layernorm.weight", pre)));
        lay.gain = decl_w(b, nn, RAD_F32, { g.n_embd }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                          grp_layer(l));
        RAD_ARCH_TRY(lay.norm.declare(b, g, lay.gain, m.x, m.act, g.n_embd));
        const char* ap = nm.ckpt("%s.attention", pre);
        if (lay.mla) {
            RAD_ARCH_TRY(lay.attn.declare(b, nm, g, s, l, m.aw, ap, m.latent));
        } else {
            RAD_ARCH_TRY(rad_bind_layer_kv(b, l, m.state));
            RAD_ARCH_TRY(rad_bind_layer_kv(b, l, m.conv));
            RAD_ARCH_TRY(lay.kda.declare(b, nm, g, s, l, m.kw, ap, m.state, m.conv));
        }
        lay.op_add = rw(b,
                        RAD_OP(b, "add",
                               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                        { m.x, m.y }, { m.x });
        if (!lay.op_add) return RAD_E_INVAL;
        if (lay.dense) {
            MlpFP8::Src ds{ nm.ckpt("%s.post_attention_layernorm.weight", pre),
                            nm.ckpt("%s.mlp.gate_proj.weight", pre),
                            nm.ckpt("%s.mlp.up_proj.weight", pre),
                            nm.ckpt("%s.mlp.down_proj.weight", pre) };
            RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, l, m.dw, ds));
        } else {
            RAD_ARCH_TRY(lay.moe.declare(b, nm, g, l, m.mw, msrc[(size_t)l]));
        }
    }
    RAD_ARCH_TRY(rad_kv_group_replicated(b, m.latent));
    m.norm = rw(b,
                RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                  RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                       RAD_WEIGHTS(m.final_gain)),
                { m.x }, { m.final_hidden });
    m.gather = rw(b,
                  RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_embd),
                                    RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                  { m.final_hidden }, { m.logit_input });
    m.head_op =
        rw(b,
           RAD_OP(b, "logits_gemm",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                             RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", "bf16")),
                  RAD_WEIGHTS(m.head)),
           { m.logit_input }, { m.logits });
    if (!m.embed || !m.norm || !m.gather || !m.head_op) return RAD_E_INVAL;

    /* ---- the drafter: GlmMtp at Ling's width and checkpoint names around one more MLA layer
     * with the routed feed-forward. It shares the trunk's buffers, latent cache, embedding and
     * head. */
    if (mtp) {
        const int l = (int)s.n_layer;
        auto& mt = m.mtp;
        mt.width = g.n_embd;
        mt.prefix = nm.ckpt("model.layers.%d", l);
        mt.layer = l;
        mt.norm_src = "final_layernorm.weight";
        const GlmMtp::Wire tw{ m.final_hidden, m.logits, m.token, m.head };
        RAD_ARCH_TRY(mt.declare_inputs(b, g, tw, false));
        Layer& d = m.draft;
        d.mla = true;
        const char* pre = mt.prefix;
        const char* nn = nm.f("blk.%d.attn_norm.weight", l);
        RAD_ARCH_TRY(map_copy(b, nn, nm.ckpt("%s.input_layernorm.weight", pre)));
        d.gain = decl_w(b, nn, RAD_F32, { g.n_embd }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                        grp_layer(l));
        RAD_ARCH_TRY(d.norm.declare(b, g, d.gain, mt.x, m.act, g.n_embd));
        RAD_ARCH_TRY(d.attn.declare(b, nm, g, s, l, m.aw, nm.ckpt("%s.attention", pre), m.latent));
        RAD_ARCH_TRY(mt.declare_join(b, m.y));
        auto mw = m.mw;
        mw.x = mt.x;
        RAD_ARCH_TRY(d.moe.declare(b, nm, g, l, mw, msrc[(size_t)l]));
        RAD_ARCH_TRY(mt.declare_head(b));
        m.have_mtp = true;
    }
    rad_note(b,
             "Ling-3.0 (bailing_hybrid): TP%d, %lld layers, %lld KDA + %lld dense MLA, %lld experts "
             "top-%lld in %lld groups (%lld kept), MTP %s",
             world, (long long)s.n_layer, (long long)(s.n_layer - s.n_layer / s.group_size),
             (long long)(s.n_layer / s.group_size), (long long)s.experts, (long long)s.top_k,
             (long long)s.groups, (long long)s.topk_groups, m.have_mtp ? "enabled" : "not declared");
    return RAD_OK;
}

/* ============================== step ============================== */

/* RADIANCE_LING_DUMP=DIR: the residual stream after each layer's attention and feed-forward, every
 * row of every step, as `<DIR>/s<step>.t<rows>.l<layer>.<attn|ffn>.r<rank>.bf16` ([rows, n_embd]
 * bf16). A diagnostic for comparing against a reference model layer by layer; it synchronises the
 * stream. */
static void dump(RadCtx* c, const RadBatch* batch, rad_buf x, int64_t E, int layer,
                 const char* site) {
    static const char* dir = std::getenv("RADIANCE_LING_DUMP");
    if (!dir || !*dir) return;
    const int64_t T = batch->n_tok;
    void* p = rad_buf_ptr(c, x);
    if (!p) return;
    std::vector<uint16_t> h((size_t)(T * E));
    if (rad_memcpy_async(h.data(), p, T * E * 2, rad_stream(c)) < 0) return;
    rad_stream_sync(rad_stream(c));
    char path[512];
    std::snprintf(path, sizeof path, "%s/s%d.t%lld.l%d.%s.r%d.bf16", dir, batch->step,
                  (long long)T, layer, site, rad_rank(c));
    if (FILE* f = std::fopen(path, "wb")) {
        std::fwrite(h.data(), 2, h.size(), f);
        std::fclose(f);
    }
}

/* A draft pass as GLM-5.3's (glm5next.cpp): the head's inputs, its MLA layer over every row (a
 * history pass fills the draft layer's latent cache for all of them), then on the proposal rows the
 * residual join, the routed feed-forward and the head. */
static void draft_pass(RadCtx* c, Model& m, const RadBatch* batch) {
    if (!m.have_mtp || batch->draft_pass == 1) {
        rad_step_fail(c, "bailing_hybrid MTP pass is not declared or has invalid round 1");
        return;
    }
    const int64_t T = batch->n_tok;
    m.mtp.step_inputs(c, batch);
    auto& l = m.draft;
    l.norm.step(c, T);
    l.attn.step(c, batch);
    if (batch->draft_pass < 0 && batch->n_draft_out <= 0) return;
    m.mtp.step_join(c, batch, m.y);
    l.moe.step_rows(c, GlmMtp::head_rows(batch));
    m.mtp.step_head(c, batch);
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];
    const int64_t T = batch->n_tok, E = m.g.n_embd;
    if (batch->enc) {
        rad_step_fail(c, "bailing_hybrid serves text");
        return;
    }
    if (batch->draft_pass) {
        draft_pass(c, m, batch);
        return;
    }
    RAD_ISSUE_N(c, m.embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.token), brows(m.x, T));
    if (m.embed_ar) RAD_ISSUE_N(c, m.embed_ar, T * E, brows(m.x, T), RAD_NONE);
    for (size_t i = 0; i < m.layers.size(); ++i) {
        const Layer& l = m.layers[i];
        l.norm.step(c, T);
        if (l.mla)
            l.attn.step(c, batch);
        else
            l.kda.step(c, batch);
        RAD_ISSUE_N(c, l.op_add, T, brows(m.x, T), brows(m.y, T), brows(m.x, T));
        dump(c, batch, m.x, E, (int)i, "attn");
        if (l.dense)
            l.mlp.step(c, batch);
        else
            l.moe.step(c, batch);
        dump(c, batch, m.x, E, (int)i, "ffn");
    }
    RAD_ISSUE_N(c, m.norm, T, brows(m.x, T), RAD_W(m.final_gain), brows(m.final_hidden, T));
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.gather, batch->n_out, brows(m.final_hidden, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.logit_input, batch->n_out));
        RAD_ISSUE_N(c, m.head_op, batch->n_out, brows(m.logit_input, batch->n_out), RAD_W(m.head),
                    brows(m.logits, batch->n_out));
    }
}

/* The default draft depth for `--num-speculative-tokens auto`: GlmMtp's operating point, when the
 * checkpoint carries its one MTP layer. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    if (cfg(meta, "num_nextn_predict_layers", 0) == 1) out->draft_depth = GlmMtp::kDepth;
    out->shape_probe_ok = 1;      /* declare() writes scratch under shape_probe */
    return RAD_OK;
}

}  /* namespace bailing_hybrid */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(bailing_hybrid)
RAD_ARCH_PLUGIN(bailing_hybrid, "bailinghybrid", "", "0.1.0",
                "Ling-3.0 (bailing_hybrid): KDA and dense latent attention, group-routed experts")
#endif
