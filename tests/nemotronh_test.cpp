/* nemotronh_test.cpp -- Nemotron-3.5-Bolt's plugin (nemotronh) with its LiLiCorr drafter, declared
 * and stepped against the recording builder of arch_fakes.h.
 *
 * The metadata is a nine-block cut of the released config, as rad-info --meta prints a container
 * converted from it (the vision tower's keys left out): every block kind, one attention block, and
 * the drafter with 32 query heads over 2 KV heads of 128. */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include <iterator>

/* ==================================================================== the plugin under test */
#include "../arch/nemotronh/nemotronh.cpp"

namespace {

const char* const k_nemo[][2] = {
    { "tie_word_embeddings", "0" },
    { "attention_bias", "0" },
    { "block_configs.1.moe_intermediate_size", "2048" },
    { "block_configs.1.n_routed_experts", "240" },
    { "block_configs.1.num_experts_per_tok", "9" },
    { "block_configs.3.moe_intermediate_size", "1536" },
    { "block_configs.3.n_routed_experts", "240" },
    { "block_configs.3.num_experts_per_tok", "9" },
    { "block_configs.5.moe_intermediate_size", "2048" },
    { "block_configs.5.n_routed_experts", "240" },
    { "block_configs.5.num_experts_per_tok", "9" },
    { "block_configs.8.moe_intermediate_size", "2304" },
    { "block_configs.8.n_routed_experts", "304" },
    { "block_configs.8.num_experts_per_tok", "9" },
    { "conv_kernel", "4" },
    { "hidden_act", "relu2" },
    { "layers_block_type", "mamba moe mamba moe mamba moe mamba attention moe" },
    { "mamba_head_dim", "64" },
    { "mamba_hidden_act", "silu" },
    { "mamba_num_heads", "128" },
    { "mamba_proj_bias", "0" },
    { "mlp_bias", "0" },
    { "mlp_hidden_act", "relu2" },
    { "moe_latent_size", "1024" },
    { "moe_shared_expert_intermediate_size", "5376" },
    { "n_group", "1" },
    { "n_groups", "8" },
    { "n_shared_experts", "1" },
    { "norm_topk_prob", "1" },
    { "num_nextn_predict_layers", "0" },
    { "residual_in_fp32", "0" },
    { "routed_scaling_factor", "5" },
    { "scoring_func", "sigmoid" },
    { "ssm_state_size", "128" },
    { "topk_group", "1" },
    { "use_bias", "0" },
    { "use_conv_bias", "1" },
    { "draft.architectures", "LiLiCorrDraftModel" },
    { "draft.attention_sink_bias", "1" },
    { "draft.block_size", "8" },
    { "draft.dflash_config.attention_sink_bias", "1" },
    { "draft.dflash_config.causal", "0" },
    { "draft.dflash_config.conv_group_size", "16" },
    { "draft.dflash_config.conv_kernel_size", "2" },
    { "draft.dflash_config.lilicorr_candidate_topk", "8" },
    { "draft.dflash_config.lilicorr_enabled", "1" },
    { "draft.dflash_config.lilicorr_factor_dim", "1024" },
    { "draft.dflash_config.lilicorr_hidden_size", "1024" },
    { "draft.dflash_config.lilicorr_logit_scale", "8" },
    { "draft.dflash_config.lilicorr_mlp_ratio", "2" },
    { "draft.dflash_config.lilicorr_num_heads", "8" },
    { "draft.dflash_config.lilicorr_num_layers", "2" },
    { "draft.dflash_config.lilicorr_vector_eps", "0.0001" },
    { "draft.dflash_config.mask_token_id", "990" },
    { "draft.dflash_config.projector_type", "lilicorr" },
    { "draft.dflash_config.swa_window_size", "1024" },
    { "draft.dflash_config.target_layer_ids", "[1,2,3,5,6,8]" },
    { "draft.dflash_config.use_swa", "1" },
    { "draft.head_dim", "128" },
    { "draft.hidden_size", "4096" },
    { "draft.intermediate_size", "12288" },
    { "draft.layer_types", "sliding_attention sliding_attention sliding_attention "
                           "sliding_attention sliding_attention" },
    { "draft.num_attention_heads", "32" },
    { "draft.num_hidden_layers", "5" },
    { "draft.num_key_value_heads", "2" },
    { "draft.rms_norm_eps", "9.9999999999999995e-07" },
    { "draft.rope_theta", "10000" },
    { "draft.sliding_window", "1024" },
    { "draft.use_sliding_window", "1" },
    { "draft.vocab_size", "131072" },
};

struct Nemo {
    std::vector<const char*> keys, vals;
    Nemo() {
        for (const auto& kv : k_nemo) {
            keys.push_back(kv[0]);
            vals.push_back(kv[1]);
        }
    }
    RadModelMeta meta() const {
        RadModelMeta m{};
        m.arch_id     = "nemotronh";
        m.name        = "test-nemotron-3.5-bolt-c9";
        m.quant       = "";
        m.n_layers    = 9;
        m.n_embd      = 4096;
        m.n_head      = 32;
        m.n_head_kv   = 2;
        m.head_dim    = 128;
        m.n_ff        = 2688;
        m.n_vocab     = 131072;
        m.n_ctx_train = 262144;
        m.rms_eps     = 1e-5f;
        m.rope_theta  = 10000.0f;
        m.rope_scale  = 1.0f;
        m.n_kv        = (int)keys.size();
        m.kv_key      = keys.data();
        m.kv_val      = vals.data();
        return m;
    }
};

/* The weights as the release recipes store them: the drafter's linears and every trunk linear block
 * fp8, dt's rows of in_proj bf16 (32 a rank at four ranks, no whole scale block). The drafter's
 * keys come first: the fake answers the first key a name contains. */
void nemo_encs(RadBuilder& b) {
    const RadEncoding fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    b.encs = { { "dflash.fc.", fp8 }, { "dflash.blk.", fp8 }, { "ssm_dt", rad_enc_plain(RAD_BF16) },
               { "blk.", fp8 } };
}

int buf_named(const RadBuilder& b, const char* name) {
    for (size_t i = 0; i < b.bufs.size(); ++i)
        if (b.bufs[i].name == name) return (int)(i + 1);
    return 0;
}
bool noted(const RadBuilder& b, const char* text) {
    for (const auto& n : b.notes)
        if (n.find(text) != std::string::npos) return true;
    return false;
}
rad_kvgroup kv_named(const RadBuilder& b, const char* name) {
    for (size_t i = 0; i < b.kv_groups.size(); ++i)
        if (b.kv_groups[i].name == name) return (rad_kvgroup)(i + 1);
    return 0;
}

RadBuildCtx nemo_ctx(int rank, int world) {
    RadBuildCtx c = build_ctx(rank, world);
    c.max_spec = 7;   /* LiLiCorr's trained block of 8 */
    return c;
}

}  // namespace

/* drafter_geom keeps refusing fewer KV heads than ranks unless the drafter block reads k and v at
 * kv_off; with `kv_repl_ok` it replicates them as geom_from does the target's: one head a rank,
 * ranks {0, 1} on head 0 and {2, 3} on head 1. The target's own kv_off is not inherited. */
TEST(drafter_geom_replicates_kv_heads_only_when_the_block_asks) {
    using namespace rad::arch;
    Nemo f;
    const RadModelMeta meta = f.meta();
    for (int world : { 2, 4 })
        for (int rank = 0; rank < world; ++rank) {
            const RadBuildCtx ctx = nemo_ctx(rank, world);
            Geom t{};
            REQUIRE_EQ(geom_from(t, &meta, &ctx, "bf16", RAD_BF16), RAD_OK);
            Geom dg{};
            if (world == 4) {
                CHECK_EQ(t.kv_off, (int64_t)(rank / 2) * 128);   /* the target replicates too */
                CHECK_EQ(drafter_geom(dg, t, &meta, "LiLiCorr"), (int)RAD_E_INVAL);
            }
            REQUIRE_EQ(drafter_geom(dg, t, &meta, "LiLiCorr", true), RAD_OK);
            CHECK_EQ(dg.n_head, 32 / world);
            CHECK_EQ(dg.n_head_kv, 1);
            CHECK_EQ(dg.n_head_kv_all, 2);
            CHECK_EQ(dg.gqa(), 32 / world);
            CHECK_EQ(dg.kv_off, world == 4 ? (int64_t)(rank / 2) * 128 : (int64_t)0);
        }
}

/* The drafter at two and four ranks. Each rank keeps one KV head in `kv_draft`; at four its k and v
 * are the rows of its replicated head, [kv_off, kv_off + 128) of the 256-row projection (codes, and
 * the scale plane in 128-row blocks), and q stays an even share. The sinked attention then runs 8
 * query heads over that head (gqa 8, a libr4d row of its own) where two ranks run 16 (gqa 16). At
 * two ranks nothing is spanned: the even shard is the head. */
TEST(lilicorr_replicates_its_two_kv_heads_at_four_ranks) {
    Nemo f;
    const RadModelMeta meta = f.meta();
    for (int world : { 2, 4 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            nemo_encs(b);
            const RadBuildCtx ctx = nemo_ctx(rank, world);
            REQUIRE_EQ(nemotronh::declare(&b, &meta, &ctx), RAD_OK);
            CHECK(nemotronh::models[rank]->have_df);
            const rad_kvgroup kvd = kv_named(b, "kv_draft");
            REQUIRE(kvd != 0);
            CHECK_EQ(b.kv_groups[kvd - 1].d.n_head_kv, 1);
            CHECK_EQ(b.kv_groups[kvd - 1].d.head_dim, 128);
            CHECK_EQ(b.bufs[buf_named(b, "df_q") - 1].d.shape[1], (int64_t)(32 / world) * 128);
            CHECK_EQ(b.bufs[buf_named(b, "df_k") - 1].d.shape[1], (int64_t)128);
            CHECK_EQ(b.bufs[buf_named(b, "df_cv") - 1].d.shape[1], (int64_t)128);
            int sinks = 0;
            for (const auto& o : b.ops)
                if (o.op == "attn_paged_sink") {
                    CHECK_EQ(pint(&o, "gqa"), 32 / world);
                    CHECK_EQ(pint(&o, "n_head"), 32 / world);
                    CHECK_EQ(pint(&o, "head_dim"), 128);
                    ++sinks;
                }
            CHECK_EQ(sinks, 5);
            const int64_t kv0 = (rank / 2) * 128;
            for (int l = 0; l < 5; ++l) {
                for (const char* kv : { "attn_k", "attn_v" }) {
                    const std::string base = "dflash.blk." + std::to_string(l) + "." + kv;
                    const RecWeight* w = find_w(b, base + ".weight");
                    const RecWeight* s = find_w(b, base + ".scale");
                    REQUIRE(w != nullptr && s != nullptr);
                    CHECK_EQ(w->d.shape[0], 128);
                    CHECK_EQ(w->span_set, world == 4);
                    CHECK_EQ(s->span_set, world == 4);
                    if (world == 4) {
                        CHECK_EQ(w->span_lo, kv0);
                        CHECK_EQ(w->span_hi, kv0 + 128);
                        CHECK_EQ(s->span_lo, kv0 / 128);
                        CHECK_EQ(s->span_hi, kv0 / 128 + 1);
                    }
                }
                const RecWeight* q =
                    find_w(b, "dflash.blk." + std::to_string(l) + ".attn_q.weight");
                REQUIRE(q != nullptr);
                CHECK(!q->span_set);
                CHECK_EQ(q->d.shape[0], (int64_t)(32 / world) * 128);
            }
            /* the target's attention replicates its two heads the same way */
            const rad_kvgroup kva = kv_named(b, "kv_attn");
            REQUIRE(kva != 0);
            CHECK_EQ(b.kv_groups[kva - 1].d.n_head_kv, 1);
            CHECK(noted(b, world == 4 ? "kv (replicated) x 128" : "/ 1 kv x 128"));
        }
}

/* A speculative step at four ranks: the trunk's verify of two sequences, the drafter's context pass
 * over its rows and its query pass, each without a step failure and lint-clean. */
TEST(nemotronh_tp4_steps_lint_clean_with_the_lilicorr_drafter) {
    Nemo f;
    const RadModelMeta meta = f.meta();
    for (int rank : { 0, 3 }) {
        RadBuilder b;
        nemo_encs(b);
        const RadBuildCtx ctx = nemo_ctx(rank, 4);
        REQUIRE_EQ(nemotronh::declare(&b, &meta, &ctx), RAD_OK);
        auto trunk = kit(b, { 8, 8 }, {});
        auto cpass = context_pass(b, *trunk);
        auto qpass = query_pass(b, 2, 8, 7);
        std::vector<int32_t> anchors = { 7, 15 };   /* each sequence's last committed row */
        qpass->b.anchor_rows = anchors.data();
        auto v = run_passes(b, nemotronh::step, { trunk.get(), cpass.get(), qpass.get() }, rank, 4);
        CHECK_PASSES(b, v);
        /* the drafter ran: its five layers' sinked attention and its keys into the cache */
        const std::vector<std::string> q = issued_ops(b, *v[2]);
        CHECK_EQ(count_issued(q, "attn_paged_sink"), 5);
        CHECK_EQ(count_issued(q, "kv_store"), 5);
        CHECK_EQ(count_issued(issued_ops(b, *v[1]), "kv_store"), 5);
    }
}

/* The engine's arena levels declare again at a smaller max_tok (RadBuildCtx::shape_probe) and
 * must get the real declare's graph: the same ops with the same fixed parameters, only ranged row
 * counts and buffer rows smaller. The drafter's split fc gathers its message as a range; a fixed
 * numel there left every level unplanned, and sixteen sequences short of the lent arena. */
TEST(nemotronh_sizing_declares_match_the_real_graph_at_every_arena_level) {
    Nemo f;
    const RadModelMeta meta = f.meta();
    for (int world : { 2, 4 })
        for (int64_t R : { 256, 1024 }) {
            RadBuildCtx c = nemo_ctx(1, world);
            c.max_tok = 4096;
            c.max_seqs = 16;
            RadBuilder full;
            nemo_encs(full);
            REQUIRE_EQ(nemotronh::declare(&full, &meta, &c), RAD_OK);
            RadBuildCtx p = c;
            p.max_tok = R;
            p.shape_probe = 1;
            RadBuilder small;
            nemo_encs(small);
            REQUIRE_EQ(nemotronh::declare(&small, &meta, &p), RAD_OK);
            REQUIRE_EQ(small.ops.size(), full.ops.size());
            int differ = 0;
            for (size_t i = 0; i < full.ops.size(); ++i) {
                if (small.ops[i].op != full.ops[i].op) {
                    ++differ;
                    continue;
                }
                REQUIRE_EQ(small.ops[i].p.size(), full.ops[i].p.size());
                for (size_t k = 0; k < full.ops[i].p.size(); ++k) {
                    const RecParam& a = full.ops[i].p[k];
                    const RecParam& q = small.ops[i].p[k];
                    if (a.kind == RAD_P_RANGE) continue;
                    if (a.ival != q.ival || a.sval != q.sval || a.dval != q.dval) {
                        fprintf(stderr, "    world %d R %lld: op %zu '%s' param '%s': %lld real, "
                                        "%lld small\n", world, (long long)R, i,
                                full.ops[i].op.c_str(), a.key.c_str(), a.ival, q.ival);
                        ++differ;
                    }
                }
            }
            CHECK_EQ(differ, 0);
            REQUIRE_EQ(small.bufs.size(), full.bufs.size());
            for (size_t i = 0; i < full.bufs.size(); ++i)
                for (uint32_t d = 1; d < full.bufs[i].d.rank; ++d)
                    CHECK_EQ(small.bufs[i].d.shape[d], full.bufs[i].d.shape[d]);
        }
}

/* The probe states LiLiCorr's fixed depth, the trained block less its anchor: what --tp 4 serves
 * with `--num-speculative-tokens 7`. */
TEST(nemotronh_probe_names_lilicorrs_depth) {
    Nemo f;
    const RadModelMeta meta = f.meta();
    RadArchProbe pr{};
    REQUIRE_EQ(nemotronh::probe(&meta, &pr), RAD_OK);
    CHECK_EQ(pr.draft_depth, 7);
    CHECK_EQ(pr.draft_depth_fixed, 1);
}

RAD_TEST_MAIN()
