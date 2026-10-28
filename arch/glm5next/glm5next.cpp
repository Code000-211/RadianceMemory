/* glm5next -- GLM-5.3-Flash (`model_type: glm5_next`) from AMD's Quark MXFP4 export: four mHC
 * residual streams over 34 KDA and 11 sparse latent-attention (MLA) layers behind a DSA indexer,
 * 288 MXFP4 experts at top-8 under sigmoid routing, block-FP8 attention and dense layers at f32
 * scales, each weight served as the container stores it; its MTP layer is a serial drafter, and
 * its vision tower is declared when the container carries it and the deployment asks for an
 * encoder. docs/ARCHITECTURES.md describes the model and its traps.
 *
 * The blocks live in arch/common (installed as <arch/...>), one a file:
 *
 *   rad_block_glm_mhc.h       ManifoldConn   the four-stream residual read and write
 *   rad_block_glm_kda.h       GlmKda         the 34 linear layers, KdaState their recurrence
 *   rad_block_glm_mla.h       GlmMla         the 11 sparse MLA layers, MlaLatent their core
 *   rad_block_glm_indexer.h   GlmIndexer     the DSA indexer, DsaCache its key cache and selection
 *   rad_block_glm_router.h    GlmRouter      the correction-bias router (through MoeFP8)
 *   rad_block_glm_mtp.h       GlmMtp         the draft head around the plugin's layer 45
 *   rad_block_glm_vision.h    GlmVision      the tower, MediaRows its rows over the embeddings
 *   rad_glm_tp.h              GlmTp          the shards, the row runs and the split thresholds
 *
 * This file declares the buffers the blocks share, places the blocks and issues the step's
 * order. Declaration order is liveness (the buffer plan takes lifetimes from it) and the issue
 * order is the recorded pass, so both are kept as they read here. This plugin uses only the
 * public ABI and the installed common headers. */
#include <arch/rad_block_glm_mhc.h>
#include <arch/rad_block_glm_kda.h>
#include <arch/rad_block_glm_mla.h>
#include <arch/rad_block_glm_indexer.h>
#include <arch/rad_block_glm_mtp.h>
#include <arch/rad_block_glm_vision.h>
#include <arch/rad_block_mlp_fp8.h>
#include <arch/rad_block_moe_fp8.h>
#include <memory>

namespace glm5next {
using namespace rad::arch;

struct Layer {
    bool full = false, draft = false;
    ManifoldConn attn_hc{}, ffn_hc{};
    NormQuantFP8 norm{};
    GlmKda kda{};
    GlmMla mla{};
    GlmIndexer index{};
    MlpFP8 dense{};
    MoeFP8 moe{};
};
struct Model {
    Names nm{};
    Geom g{};
    /* The MLA's split of a prefill-sized step under TP (RadBuildCtx::tp_attn_rows). The head
     * form declares none of the row form's full-width weights (1.29 GiB a card at eight ranks);
     * the row form is opt-in because it is not bit-identical to it (docs/ARCHITECTURES.md). */
    GlmMla::Form form = GlmMla::Form::heads;
    std::vector<Layer> layers;
    rad_kvgroup latent = 0, index = 0, tail = 0, state = 0, conv = 0;
    /* The streams and the step's stream-wide buffers. */
    rad_buf h0 = 0, h1 = 0, x = 0, y = 0, post = 0, comb = 0, final_hidden = 0, logit_input = 0,
            logits = 0;
    ActFP8 act{};
    /* The blocks' wires: one set of activations, shared by every layer of a kind. */
    GlmKda::Wire kw{};
    GlmMla::Wire aw{};
    GlmIndexer::Wire iw{};
    MlpFP8::Wire dw{};
    MoeFP8::Wire mw{};
    rad_weight token = 0, head = 0, final_gain = 0;
    rad_op embed = 0, embed_ar = 0, expand = 0, contract = 0, norm = 0, gather = 0, head_op = 0;
    /* The draft head and its layer, declared only when the deployment speculates. */
    bool have_mtp = false;
    GlmMtp mtp{};
    Layer draft{};
    /* Media: the tower, and its rows over the trunk's embedding (the draft head scatters its own). */
    bool media = false;
    GlmVision vit{};
    MediaRows mrows{};
};
static std::unique_ptr<Model> models[MAX_RANKS];
static thread_local std::unique_ptr<Model> probe_model;

static int64_t cfg(const RadModelMeta* m, const char* key, int64_t fallback) {
    char nested[160];
    std::snprintf(nested, sizeof nested, "text_config.%s", key);
    return rad_meta_geti(m, key, rad_meta_geti(m, nested, fallback));
}
static rad_op copy(RadBuilder* b, const Geom& g, rad_buf x, rad_buf y) {
    return rw(b,
              RAD_OP(b, "cast",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 4096),
                                RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                     RAD_NOWEIGHTS),
              { x }, { y });
}
static rad_op norm(RadBuilder* b, const Geom& g, rad_weight w, rad_buf x, rad_buf y) {
    return rw(b,
              RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 4096),
                                RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                     RAD_WEIGHTS(w)),
              { x }, { y });
}

/* ------------------------------------------------------------------ the shared activations */
static int stream_buffers(RadBuilder* b, Model& m) {
    auto& g = m.g;
    auto& nm = m.nm;
    const int64_t T = g.max_tok;
    m.h0 = decl_b(b, nm.f("manifold"), RAD_BF16, { T, 16384 });
    m.h1 = decl_b(b, nm.f("manifold_mid"), RAD_BF16, { T, 16384 });
    m.x = decl_b(b, nm.f("collapsed"), RAD_BF16, { T, 4096 });
    /* The MLA row form gathers every rank's run at full length, so its output is rounded up to
     * whole runs; the head form keeps the step's rows. */
    const bool rows = g.world > 1 && m.form == GlmMla::Form::rows;
    m.y = decl_b(b, nm.f("attn_delta"), RAD_BF16,
                 { rows ? g.world * GlmMla::rows_for(T, g.world) : T, 4096 });
    m.post = decl_b(b, nm.f("post"), RAD_F32, { T, 4 });
    m.comb = decl_b(b, nm.f("mix"), RAD_F32, { T, 16 });
    m.act.x = decl_b(b, nm.f("normalized"), RAD_BF16, { T, 4096 });
    RAD_ARCH_TRY(m.act.declare_qs(b, nm, g, "normalized", 4096));
    m.final_hidden = decl_b(b, nm.f("final_hidden"), RAD_BF16, { T, 4096 }, RAD_BUF_PERSIST);
    m.logit_input = decl_b(b, nm.f("logit_input"), RAD_BF16, { g.max_logit_rows, 4096 });
    m.logits = decl_b(b, nm.f("logits"), RAD_F32, { g.max_logit_rows, g.n_vocab });
    return rad_declare_logits(b, m.logits);
}
static int kda_buffers(RadBuilder* b, Model& m) {
    auto& g = m.g;
    auto& nm = m.nm;
    const int64_t T = g.max_tok, W = GlmTp::kda_width(g.world);
    auto bf = [&](const char* n, int64_t width) {
        return decl_b(b, nm.f("%s", n), RAD_BF16, { T, width });
    };
    auto& kw = m.kw;
    kw.input = m.act.x;
    kw.out = m.y;
    kw.forget_low = bf("forget_low", 128);
    kw.gate_low = bf("gate_low", 128);
    kw.core.qkv = bf("qkv", 3 * W);
    kw.core.conv = bf("convolved", 3 * W);
    kw.core.a = bf("forget", W);
    kw.core.beta = bf("beta", GlmTp::heads(g.world));
    kw.core.gate = bf("gate", W);
    kw.core.out = bf("recurrent_out", W);
    return RAD_OK;
}
/* The head form's buffers, then the selection planes the indexer writes and the MLA reads, then
 * the row form's (Model::form). Under TP the selection planes hold every rank's run of rows,
 * which can reach past the step (DsaCache::part_rows). */
static int mla_buffers(RadBuilder* b, Model& m) {
    auto& g = m.g;
    auto& nm = m.nm;
    const int64_t T = g.max_tok, H = GlmTp::heads(g.world), Q = GlmTp::mla_width(g.world);
    auto bf = [&](const char* n, int64_t width) {
        return decl_b(b, nm.f("%s", n), RAD_BF16, { T, width });
    };
    auto& aw = m.aw;
    aw.input = m.act;
    aw.q_raw = bf("q_raw", 1536);
    aw.kv_raw = bf("kv_raw", 512);
    aw.q_low.x = bf("q_low", 1536);
    RAD_ARCH_TRY(aw.q_low.declare_qs(b, nm, g, "q_low", 1536));
    aw.latent = bf("latent", 512);
    aw.out = m.y;
    aw.core.q = bf("query", Q);
    aw.core.q_latent = decl_b(b, nm.f("q_latent"), RAD_BF16, { T, H, 512 });
    aw.core.attn_latent = decl_b(b, nm.f("attn_latent"), RAD_BF16, { T, H, 512 });
    aw.attn.x = bf("expanded", Q);
    RAD_ARCH_TRY(aw.attn.declare_qs(b, nm, g, "expanded", Q));
    aw.core.out = aw.attn.x;
    const int64_t part = g.world > 1 ? DsaCache::part_rows(T, g.world) : 0;
    const int64_t TQ = g.world > 1 ? std::max(T, g.world * part) : T;
    aw.core.selected = decl_b(b, nm.f("selected"), RAD_I32, { TQ, 513 });
    aw.core.length = decl_b(b, nm.f("length"), RAD_I32, { TQ });
    if (g.world > 1 && m.form == GlmMla::Form::rows) {
        auto& rows = aw.rows;
        const int64_t R = GlmMla::rows_for(T, g.world);
        rows.q = decl_b(b, nm.f("rows_query"), RAD_BF16, { R, 16384 });
        rows.q_latent = decl_b(b, nm.f("rows_q_latent"), RAD_BF16, { R, 64, 512 });
        rows.attn_latent = decl_b(b, nm.f("rows_attn_latent"), RAD_BF16, { R, 64, 512 });
        rows.attn.x = decl_b(b, nm.f("rows_expanded"), RAD_BF16, { R, 16384 });
        RAD_ARCH_TRY(rows.attn.declare_qs(b, nm, g, "rows_expanded", 16384, R));
        rows.out = decl_b(b, nm.f("rows_out"), RAD_BF16, { R, 4096 });
    }
    return RAD_OK;
}
static int indexer_buffers(RadBuilder* b, Model& m) {
    auto& g = m.g;
    auto& nm = m.nm;
    const int64_t T = g.max_tok;
    const int64_t part = g.world > 1 ? DsaCache::part_rows(T, g.world) : 0;
    const int64_t TQ = g.world > 1 ? std::max(T, g.world * part) : T;
    const int64_t work = (T + 3) / 4 + g.max_seqs, blocks = (g.max_ctx + 3) / 4 + 1;
    auto bf = [&](const char* n, int64_t width) {
        return decl_b(b, nm.f("%s", n), RAD_BF16, { T, width });
    };
    auto& iw = m.iw;
    iw.input = m.act.x;
    iw.q_low = m.aw.q_low.x;
    iw.q_wide = bf("index_query", 4096);
    iw.kg_raw = bf("index_key_gate", 256);
    auto& c = iw.cache;
    c.query = decl_b(b, nm.f("index_heads"), RAD_BF16, { T * 32, 128 });
    c.codes = decl_b(b, nm.f("index_codes"), RAD_F8E4M3, { TQ * 32, 128 });
    c.scales = decl_b(b, nm.f("index_scales"), RAD_F32, { TQ * 32, 1 });
    c.gate = decl_b(b, nm.f("index_head_weights"), RAD_F32, { TQ, 32 });
    c.kg = bf("index_normalized_key_gate", 256);
    c.stage = decl_b(b, nm.f("pool_stage"), RAD_BF16, { work * 4, 256 });
    c.page = decl_b(b, nm.f("pool_page"), RAD_I32, { work });
    c.bpos = decl_b(b, nm.f("pool_pos"), RAD_I32, { work });
    c.nc = decl_b(b, nm.f("pool_count"), RAD_I32, { g.max_seqs });
    c.pooled = decl_b(b, nm.f("pool_codes"), RAD_F8E4M3, { work, 128 });
    c.pool_scale = decl_b(b, nm.f("pool_scales"), RAD_F32, { work, 1 });
    /* A split step scores one run of rows a rank and an unsplit one is under GlmTp::kSplitRows:
     * at 64k of context the whole step's plane would be 256 MiB. */
    const int64_t scored = g.world > 1 ? std::min(T, std::max(part, GlmTp::kSplitRows)) : T;
    c.score = decl_b(b, nm.f("index_scores"), RAD_F32, { scored, blocks });
    c.selected = m.aw.core.selected;
    c.length = m.aw.core.length;
    c.nsel = decl_b(b, nm.f("selected_count"), RAD_I32, { T });
    /* The MLA block's resolve count: qsa_select's second output, never `length`. */
    m.aw.core.nsel = c.nsel;
    if (g.world > 1) {
        c.cu_part = decl_b(b, nm.f("index_part_bounds"), RAD_I32, { g.max_seqs + 1 });
        c.sel_part = decl_b(b, nm.f("selected_part"), RAD_I32, { part, 513 });
        c.len_part = decl_b(b, nm.f("length_part"), RAD_I32, { part });
    }
    return RAD_OK;
}
static int ffn_buffers(RadBuilder* b, Model& m) {
    auto& g = m.g;
    auto& nm = m.nm;
    const int64_t T = g.max_tok, F = GlmTp::expert_width(g.world);
    auto bf = [&](const char* n, int64_t width) {
        return decl_b(b, nm.f("%s", n), RAD_BF16, { T, width });
    };
    auto& dw = m.dw;
    dw.x = m.x;
    dw.h = m.act;
    dw.gate_up = bf("dense_gate_up", 2 * g.n_ff);
    dw.ffn.x = bf("dense_ff", g.n_ff);
    RAD_ARCH_TRY(dw.ffn.declare_qs(b, nm, g, "dense_ff", g.n_ff));
    auto& mw = m.mw;
    mw.x = m.x;
    mw.h = m.act;
    mw.logits = decl_b(b, nm.f("router_logits"), RAD_F32, { T, 288 });
    mw.ids = decl_b(b, nm.f("expert_ids"), RAD_I32, { T, 8 });
    mw.ew = decl_b(b, nm.f("expert_weights"), RAD_F32, { T, 8 });
    mw.eoff = decl_b(b, nm.f("expert_offsets"), RAD_I32, { 289 });
    /* PERSIST, as every other MoE here declares it: the router writes the counts and no declared
     * op reads them (rad_route_report is a handoff), so as a transient a later buffer could share
     * their bytes and an arena level would move them, while the core copies them at the block's
     * end and the file stager reads them, through the address the last pass reported, at the
     * layer's first expert op. */
    mw.ecnt = decl_b(b, nm.f("expert_counts"), RAD_I32, { 288 }, RAD_BUF_PERSIST);
    mw.sorted = decl_b(b, nm.f("expert_sorted"), RAD_I32, { T * 8 });
    mw.inv = decl_b(b, nm.f("expert_inverse"), RAD_I32, { T * 8 });
    // FE: the routed experts' WHOLE intermediate (2048). Under expert parallelism a rank owns
    // whole experts, so the routed arm's buffers are full-width at every world size; the SHARED
    // expert stays width-sharded (F). The two coincide at world 1.
    const int64_t FE = 2048;
    mw.egu = decl_b(b, nm.f("expert_gate_up"), RAD_BF16, { T * 8, 2 * FE });
    mw.eff.x = decl_b(b, nm.f("expert_ff"), RAD_BF16, { T * 8, FE });
    RAD_ARCH_TRY(mw.eff.declare_qs(b, nm, g, "expert_ff", FE, T * 8));
    mw.edn = decl_b(b, nm.f("expert_down"), RAD_BF16, { T * 8, 4096 });
    mw.sgu = bf("shared_gate_up", 2 * F);
    mw.sff.x = bf("shared_ff", F);
    RAD_ARCH_TRY(mw.sff.declare_qs(b, nm, g, "shared_ff", F));
    mw.sout = bf("shared_out", 4096);
    mw.sgate = bf("shared_one", 1);
    return RAD_OK;
}
static int kv_groups(RadBuilder* b, Model& m) {
    auto& g = m.g;
    auto& nm = m.nm;
    RadKVGroupDecl d{};
    d.kind = RAD_KV_FULL;
    // bf16, or E4M3 under --kv-cache-dtype fp8 (geometry); the other groups keep their own.
    d.dtype = rad_dtype_parse(g.kv_dtype);
    d.n_head_kv = 1;
    d.head_dim = 256;
    m.latent = rad_decl_kv_group(b, nm.f("kv_latent"), &d);
    d = {};
    d.kind = RAD_KV_LINEAR;
    d.dtype = RAD_BF16;
    d.n_head_kv = 1;
    d.state_dim[0] = 4 + g.max_spec;
    d.state_dim[1] = 256;
    m.tail = rad_decl_kv_group(b, nm.f("kv_index_tail"), &d);
    d = {};
    d.kind = RAD_KV_LINEAR;
    d.dtype = RAD_F32;
    d.n_head_kv = GlmTp::heads(g.world);
    d.state_dim[0] = d.state_dim[1] = 128;
    m.state = rad_decl_kv_group(b, nm.f("kv_kda"), &d);
    d = {};
    d.kind = RAD_KV_CONV;
    d.dtype = RAD_BF16;
    d.n_head_kv = 1;
    d.head_dim = 3 * GlmTp::kda_width(g.world);
    d.conv_width = 4;
    m.conv = rad_decl_kv_group(b, nm.f("kv_kda_conv"), &d);
    // The index group is declared by the first indexer, right before its cache consumer.
    return m.latent && m.tail && m.state && m.conv ? RAD_OK : RAD_E_INVAL;
}

/* ------------------------------------------------------------------ one layer */
static int layer(RadBuilder* b, Model& m, Layer& lay, Names& nm, int l, bool draft) {
    const auto& g = m.g;
    lay.draft = draft;
    lay.full = draft || l % 4 == 3;
    const char* pre = nm.ckpt("model.language_model.layers.%d", l);
    const char* ap = nm.ckpt("%s.self_attn", pre);
    rad_buf input = draft ? m.mtp.x : m.x;
    ManifoldConn::Config hc;
    hc.eps = g.eps;
    if (!draft) {
        ManifoldConn::Wire a{ m.h0, m.x, m.y, m.h1, m.post, m.comb };
        ManifoldConn::Src as{ nm.ckpt("%s.hc_attn_fn", pre), nm.ckpt("%s.hc_attn_scale", pre),
                              nm.ckpt("%s.hc_attn_base", pre) };
        RAD_ARCH_TRY(lay.attn_hc.declare(b, nm, g, l, nm.f("blk.%d.hc_attn", l), hc, a, as, true));
    }
    const char* nn = nm.f("blk.%d.attn_norm.weight", l);
    RAD_ARCH_TRY(map_copy(b, nn, nm.ckpt("%s.input_layernorm.weight", pre)));
    const auto gain =
        decl_w(b, nn, RAD_F32, { 4096 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    RAD_ARCH_TRY(lay.norm.declare(b, g, gain, input, m.act, 4096));
    if (lay.full) {
        // The MTP layer keeps the head form: its row form would cost another 126 MB a card.
        auto aw = m.aw;
        if (draft) aw.rows = {};
        RAD_ARCH_TRY(lay.mla.declare(b, nm, g, l, aw, ap, m.latent, true, true));
        DsaCache::Config c;
        c.blocks = (g.max_ctx + 3) / 4 + 1;
        c.work = (g.max_tok + 3) / 4 + g.max_seqs;
        c.ring = 4 + g.max_spec;
        RAD_ARCH_TRY(lay.index.declare(b, nm, g, l, m.iw, nm.ckpt("%s.indexer", ap), c, m.latent,
                                       m.tail, &m.index));
        RAD_ARCH_TRY(lay.mla.declare_finish(b, nm, ap));
    } else {
        KdaState::Src s{ nm.ckpt("%s.q_conv1d.weight", ap), nm.ckpt("%s.k_conv1d.weight", ap),
                         nm.ckpt("%s.v_conv1d.weight", ap), nm.ckpt("%s.A_log", ap),
                         nm.ckpt("%s.dt_bias", ap),         nm.ckpt("%s.o_norm.weight", ap) };
        RAD_ARCH_TRY(lay.kda.declare(b, nm, g, l, m.kw, ap, s, m.state, m.conv));
    }
    // The ffn connection is declared after the attention's post: declaration order is liveness.
    if (!draft) {
        RAD_ARCH_TRY(lay.attn_hc.declare_post(b));
        ManifoldConn::Wire f{ m.h1, m.x, m.act.x, m.h0, m.post, m.comb };
        ManifoldConn::Src fs{ nm.ckpt("%s.hc_ffn_fn", pre), nm.ckpt("%s.hc_ffn_scale", pre),
                              nm.ckpt("%s.hc_ffn_base", pre) };
        RAD_ARCH_TRY(lay.ffn_hc.declare(b, nm, g, l, nm.f("blk.%d.hc_ffn", l), hc, f, fs, true));
    } else {
        RAD_ARCH_TRY(m.mtp.declare_join(b, m.y));
    }
    if (l < 3) {
        auto w = m.dw;
        w.x = input;
        MlpFP8::Src s{ nm.ckpt("%s.post_attention_layernorm.weight", pre),
                       nm.ckpt("%s.mlp.gate_proj.weight", pre),
                       nm.ckpt("%s.mlp.up_proj.weight", pre),
                       nm.ckpt("%s.mlp.down_proj.weight", pre) };
        lay.dense.swiglu_quant = true;
        RAD_ARCH_TRY(lay.dense.declare(b, nm, g, l, w, s, false, draft, kArNone, false, 10));
    } else {
        auto w = m.mw;
        w.x = input;
        MoeFP8::Src s;
        s.norm = nm.ckpt("%s.post_attention_layernorm.weight", pre);
        s.router = nm.ckpt("%s.mlp.gate.weight", pre);
        s.router_bias = nm.ckpt("%s.mlp.gate.e_score_correction_bias", pre);
        s.experts = nm.ckpt("%s.mlp.experts", pre);
        s.shared = nm.ckpt("%s.mlp.shared_experts", pre);
        // EXPERT PARALLELISM AT TP: each rank owns 288/world WHOLE experts (the contiguous
        // window [expert_base, expert_base+n_expert) of the checkpoint's numbering), which the
        // helper declares RAD_SHARD_NONE -- unsharded, so an expert unit is file-direct and can
        // live in the container's mapping or the file tier
        // instead of needing VRAM+pinned-pool residency for all 151 GiB of them. The router
        // stays replicated over n_expert_all; moe_scatter/router_topk_scatter drop slots
        // outside the window (moe_scatter schema: "which is how EXPERT PARALLELISM is
        // expressed"); the shared expert stays width-sharded on its own declaration path; the
        // block's all-reduce sums the partials exactly as width TP did. At world 1 this
        // degenerates to the old whole-model declaration (n_expert == n_expert_all, no spans).
        s.cfg.n_expert_all = 288;
        s.cfg.top_k = 8;
        s.cfg.n_expert = 288 / g.world;
        // A shape probe may pass rank >= world (arch_test's parallel declares reuse i % 4);
        // old code tolerated that because expert_base stayed 0, so clamp to rank 0's window.
        const int ep_rank = g.rank < g.world ? g.rank : 0;
        s.cfg.expert_base = ep_rank * (288 / g.world);
        s.cfg.probed_expert = s.cfg.expert_base;      // probe this rank's first owned expert
        s.cfg.n_ff_exp = 2048;                        // whole: whole experts' intermediate
        s.cfg.n_ff_shared = 2048 / g.world;
        s.cfg.rows = g.max_tok;
        s.cfg.sigmoid_router = 1;
        s.cfg.route_scale = 2.5;
        s.cfg.swiglu_limit = 10;
        s.cfg.shared_ungated = 1;
        s.cfg.router_fuse = 1;
        s.cfg.swiglu_quant = 1;
        // At world 1 the width-slice declaration IS the whole model: keep the spans so the
        // graph stays the pinned one. At world > 1 expert parallelism is in force and whole
        // experts need no width spans (split && ep is refused by the helper).
        if (g.world == 1) {
            for (int q = 0; q < 2; ++q) {
                s.cfg.ff_lo[q] = g.rank * (2048 / g.world);
                s.cfg.ff_hi[q] = (g.rank + 1) * (2048 / g.world);
            }
        }
        RAD_ARCH_TRY(MoeFP8::probe(b, nm, l, s, &s.cfg));
        RAD_ARCH_TRY(lay.moe.declare(b, nm, g, l, w, s, false, draft));
    }
    if (!draft) RAD_ARCH_TRY(lay.ffn_hc.declare_post(b));
    return RAD_OK;
}

/* ------------------------------------------------------------------ declare */
static int geometry(Geom& g, const RadModelMeta* meta, const RadBuildCtx* ctx, int world) {
    g.n_layer = 45;
    g.n_embd = 4096;
    g.head_dim = GlmTp::kMlaDim;
    g.n_head_all = g.n_head_kv_all = GlmTp::kHeads;
    g.n_head = g.n_head_kv = GlmTp::heads(world);
    g.n_ff_all = GlmTp::kDenseFF;
    g.n_ff = GlmTp::dense_width(world);
    g.n_vocab_all = 154880;
    g.n_vocab = shard_extent(154880, world, ctx->rank);
    g.vocab_off = shard_begin(154880, world, ctx->rank);
    g.rank = ctx->rank;
    g.world = world;
    g.attn_world = world;
    g.attn_rank = ctx->rank;
    g.max_tok = ctx->max_tok;
    g.max_ctx = ctx->max_ctx;
    g.max_seqs = ctx->max_seqs;
    g.max_spec = ctx->max_spec;
    /* Filled by hand here, not by geom_from: without it kv_resolve would declare the whole context as
     * the ring. */
    g.kv_resident = ctx->kv_resident_blocks;
    g.max_logit_rows = std::min<int64_t>(g.max_tok, g.max_seqs * (g.max_spec + 1));
    if (ctx->max_out_rows > g.max_logit_rows) g.max_logit_rows = ctx->max_out_rows;
    g.eps = meta->rms_eps;
    /* --kv-cache-dtype names the latent cache's dtype only (kv_groups); declare refuses all but
     * bf16 and fp8. */
    g.kv_dtype = rad_dtype_name(rad_kv_cache_dtype(ctx, RAD_BF16));
    g.wire_exact = !ctx->tp_wire_lossy;
    g.wire_min_bytes = ctx->tp_wire_min_bytes;
    return g.max_tok < 1 || g.max_ctx < 1 || g.max_seqs < 1 || g.eps <= 0 ? RAD_E_INVAL : RAD_OK;
}
static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->is_draft || ctx->rank < 0 || ctx->rank >= MAX_RANKS)
        return RAD_E_INVAL;
    const int world = ctx->world_size ? ctx->world_size : 1;
    if (!GlmTp::world_ok(world)) {
        fprintf(stderr, "radiance: glm5next: --tp %d; the head, KDA channel and expert splits of "
                        "this plugin are built for 1, 2, 4 or 8 ranks\n", world);
        return RAD_E_UNSUPPORTED;
    }
    /* The graph is written for the released GLM-5.3-Flash and reads none of these from the
     * checkpoint, so a checkpoint that differs in any of them is refused by name. */
    const struct {
        const char* what;
        int64_t     got, want;
    } shape[] = { { "hidden_size", meta->n_embd, 4096 },
                  { "num_hidden_layers", meta->n_layers, 45 },
                  { "vocab_size", meta->n_vocab, 154880 },
                  { "num_nextn_predict_layers", cfg(meta, "num_nextn_predict_layers", 0), 1 },
                  { "hc_mult", cfg(meta, "hc_mult", 0), 4 },
                  { "n_routed_experts", cfg(meta, "n_routed_experts", 0), 288 },
                  { "index_topk", cfg(meta, "index_topk", 0), 2048 } };
    for (const auto& s : shape)
        if (s.got != s.want) {
            fprintf(stderr, "radiance: glm5next: the checkpoint's %s is %lld; this plugin serves "
                            "the GLM-5.3-Flash geometry, where it is %lld\n",
                    s.what, (long long)s.got, (long long)s.want);
            return RAD_E_UNSUPPORTED;
        }
    if (ctx->max_spec > 31) {
        fprintf(stderr, "radiance: glm5next: --num-speculative-tokens %d; the KDA verify pass "
                        "takes at most 32 rows a sequence, so the MTP depth is at most 31\n",
                ctx->max_spec);
        return RAD_E_UNSUPPORTED;
    }
    /* fp8 narrows the latent attention cache to E4M3 at a descale of 1 (kv_groups); the index
     * tail, the DSA index and the KDA caches keep their dtypes. */
    if (ctx->kv_dtype && ctx->kv_dtype != RAD_BF16 && ctx->kv_dtype != RAD_F8E4M3) {
        fprintf(stderr, "radiance: glm5next: --kv-cache-dtype %s; this plugin serves its latent "
                        "attention cache in bf16 or fp8 only\n",
                rad_dtype_name(ctx->kv_dtype));
        return RAD_E_UNSUPPORTED;
    }
    auto& slot = ctx->shape_probe ? probe_model : models[ctx->rank];
    slot = std::make_unique<Model>();
    auto& m = *slot;
    auto& g = m.g;
    m.nm = Names(ctx->scope);
    auto& nm = m.nm;
    m.form = ctx->tp_attn_rows ? GlmMla::Form::rows : GlmMla::Form::heads;
    RAD_ARCH_TRY(geometry(g, meta, ctx, world));
    RAD_ARCH_TRY(stream_buffers(b, m));
    RAD_ARCH_TRY(kda_buffers(b, m));
    RAD_ARCH_TRY(mla_buffers(b, m));
    RAD_ARCH_TRY(indexer_buffers(b, m));
    RAD_ARCH_TRY(ffn_buffers(b, m));
    RAD_ARCH_TRY(kv_groups(b, m));
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output.weight"), nm.ckpt("lm_head.weight")));
    RAD_ARCH_TRY(
        map_copy(b, nm.f("output_norm.weight"), nm.ckpt("model.language_model.norm.weight")));
    m.token = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, { g.n_vocab, 4096 }, RAD_ACCESS_VOCAB,
                     RAD_SHARD_ROW, grp_model());
    m.head = decl_w(b, nm.f("output.weight"), RAD_BF16, { g.n_vocab, 4096 }, RAD_ACCESS_VOCAB,
                    RAD_SHARD_ROW, grp_model());
    m.final_gain = decl_w(b, nm.f("output_norm.weight"), RAD_F32, { 4096 }, RAD_ACCESS_PER_TOKEN,
                          RAD_SHARD_NONE, grp_model());
    m.embed = rw(b,
                 RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", 4096),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16"),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.token)),
                 {}, { m.x });
    // The encoder's rows overwrite the lookup before its reduction (MediaRows).
    m.media = GlmVision::present(meta) && ctx->max_enc_patches > 0;
    if (m.media) RAD_ARCH_TRY(m.mrows.declare(b, g.max_tok, 4096, m.x));
    m.embed_ar = GlmTp::reduce(b, g, m.x, true);
    m.expand = copy(b, g, m.x, m.h0);
    m.layers.resize(45);
    for (int l = 0; l < 45; ++l) RAD_ARCH_TRY(layer(b, m, m.layers[l], nm, l, false));
    // Every rank computes the whole normed latent and pools into the whole index cache
    // (rad_block_glm_mla.h, rad_block_glm_indexer.h), so the host tier keeps each once a slot.
    RAD_ARCH_TRY(rad_kv_group_replicated(b, m.latent));
    if (m.index) RAD_ARCH_TRY(rad_kv_group_replicated(b, m.index));
    m.contract = rw(b,
                    RAD_OP(b, "mhc_contract",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 4096),
                                      RAD_INT("hc", 4), RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                    { m.h0 }, { m.x });
    m.norm = norm(b, g, m.final_gain, m.x, m.final_hidden);
    m.gather = rw(b,
                  RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", 4096),
                                    RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                  { m.final_hidden }, { m.logit_input });
    m.head_op =
        rw(b,
           RAD_OP(b, "logits_gemm",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                             RAD_INT("n_embd", 4096), RAD_STR("dtype", "bf16")),
                  RAD_WEIGHTS(m.head)),
           { m.logit_input }, { m.logits });
    if (g.max_spec > 0) {
        GlmMtp::Wire tw{ m.final_hidden, m.logits, m.token, m.head };
        RAD_ARCH_TRY(m.mtp.declare_inputs(b, g, tw, m.media));
        RAD_ARCH_TRY(layer(b, m, m.draft, m.mtp.nm, 45, true));
        RAD_ARCH_TRY(m.mtp.declare_head(b));
        m.have_mtp = true;
    }
    // The tower, last: an encoder pass issues nothing else, so its activations share the arena
    // with every buffer above. Its weight groups follow the 45 layers and the MTP layer's.
    if (m.media) RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, 4096, 46));
    rad_note(b,
             "GLM-5.3: TP%d, 45 mHC layers, 34 KDA + 11 sparse MLA, mixed MXFP4/FP8, native MTP "
             "%s, vision %s",
             world, m.have_mtp ? "enabled" : "disabled", m.media ? "enabled" : "disabled");
    return RAD_OK;
}

/* ------------------------------------------------------------------ step */
static void attention(RadCtx* c, Layer& l, const RadBatch* batch) {
    l.norm.step(c, batch->n_tok);
    if (l.full) {
        l.mla.prepare(c, batch);
        l.index.step(c, batch);
        l.mla.finish(c, batch);
    } else
        l.kda.step(c, batch);
}
static void draft_pass(RadCtx* c, Model& m, const RadBatch* batch) {
    if (!m.have_mtp || batch->draft_pass == 1) {
        rad_step_fail(c, "GLM MTP pass is not declared or has invalid round 1");
        return;
    }
    const int64_t T = batch->n_tok;
    m.mtp.step_inputs(c, batch);
    auto& l = m.draft;
    l.norm.step(c, T);
    l.mla.prepare(c, batch);
    l.index.step(c, batch);
    // A history pass with no proposal row fills the draft caches and ends there.
    if (batch->draft_pass < 0 && batch->n_draft_out <= 0) return;
    l.mla.finish(c, batch);
    m.mtp.step_join(c, batch, m.y);
    l.moe.step_rows(c, GlmMtp::head_rows(batch));
    m.mtp.step_head(c, batch);
}
static void step(RadCtx* c, const RadBatch* batch) {
    auto& m = *models[rad_rank(c)];
    const int64_t T = batch->n_tok;
    // An encoder pass runs only the vision tower (RadBatch::enc).
    if (batch->enc) {
        m.vit.step(c, batch);
        return;
    }
    if (batch->draft_pass) {
        draft_pass(c, m, batch);
        return;
    }
    RAD_ISSUE_N(c, m.embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.token), brows(m.x, T));
    m.mrows.step(c, batch, m.x, T);
    if (m.embed_ar) RAD_ISSUE_N(c, m.embed_ar, T * 4096, brows(m.x, T), RAD_NONE);
    for (int i = 0; i < 4; ++i)
        RAD_ISSUE_N(c, m.expand, T, brows(m.x, T), bcol(m.h0, i * 4096, 4096, T));
    for (int i = 0; i < 45; ++i) {
        auto& l = m.layers[i];
        l.attn_hc.read(c, T);
        attention(c, l, batch);
        l.attn_hc.write(c, T);
        l.ffn_hc.read(c, T);
        if (i < 3)
            l.dense.step(c, batch);
        else
            l.moe.step(c, batch);
        l.ffn_hc.write(c, T);
    }
    RAD_ISSUE_N(c, m.contract, T, brows(m.h0, T), brows(m.x, T));
    RAD_ISSUE_N(c, m.norm, T, brows(m.x, T), RAD_W(m.final_gain), brows(m.final_hidden, T));
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.gather, batch->n_out, brows(m.final_hidden, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.logit_input, batch->n_out));
        RAD_ISSUE_N(c, m.head_op, batch->n_out, brows(m.logit_input, batch->n_out), RAD_W(m.head),
                    brows(m.logits, batch->n_out));
    }
}
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    /* The MLA block declares the streamed store and the resolve (rad_block_glm_mla.h). */
    out->kv_stream_ok = 1;
    if (cfg(meta, "num_nextn_predict_layers", 0) == 1) out->draft_depth = GlmMtp::kDepth;
    // The tower's merged block, so a pass size the core picks itself is whole output rows.
    if (GlmVision::present(meta)) {
        const int64_t k = rad_meta_geti(meta, "vision_config.spatial_merge_size", 0);
        if (k > 0) out->enc_merge = k * k;
    }
    out->shape_probe_ok = 1;
    out->tp_attn_rows_ok = 1;
    return RAD_OK;
}
} // namespace glm5next
#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(glm5next)
RAD_ARCH_PLUGIN(glm5next, "glm5next", "", "0.1.0",
                "GLM-5.3-Flash mixed MXFP4/FP8, mHC/KDA/MLA and native serial MTP")
#endif
