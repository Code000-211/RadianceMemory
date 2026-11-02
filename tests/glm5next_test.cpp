/* glm5next_test.cpp -- the declared graph and the steps of GLM-5.3-Flash's plugin and its blocks,
 * asserted statically against the recording builder of arch_fakes.h. */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include <array>
#include <dlfcn.h>
#include <map>
#include <thread>

/* ==================================================================== the plugin under test */

/* GLM-5.3-Flash: its blocks (through the plugin's includes) and the plugin. */
#include "../arch/glm5next/glm5next.cpp"

/* ==================================================================== the blocks */

TEST(manifold_connections_preserve_saved_coefficients_and_use_a_separate_residual_destination) {
    using namespace rad::arch;
    RadBuilder b;
    Names nm("");
    Geom g{};
    g.n_embd = 4096;
    g.max_tok = 32;
    g.world = 4;
    ManifoldConn::Config cfg;
    ManifoldConn::Src src{ "model.layers.3.hc_attn_fn", "model.layers.3.hc_attn_scale",
                           "model.layers.3.hc_attn_base" };
    ManifoldConn::Wire w;
    w.h = decl_b(&b, "h", RAD_BF16, { 32, 16384 });
    w.x = decl_b(&b, "x", RAD_BF16, { 32, 4096 });
    w.y = decl_b(&b, "y", RAD_BF16, { 32, 4096 });
    w.out = decl_b(&b, "out", RAD_BF16, { 32, 16384 });
    w.post = decl_b(&b, "post", RAD_F32, { 32, 4 });
    w.comb = decl_b(&b, "comb", RAD_F32, { 32, 16 });
    ManifoldConn conn;
    REQUIRE_EQ(conn.declare(&b, nm, g, 3, "blk.3.mhc_attn", cfg, w, src), RAD_OK);
    REQUIRE_EQ(b.weights.size(), (size_t)3);
    CHECK_EQ(b.weights[0].d.shape[0], 24);
    CHECK_EQ(b.weights[0].d.shape[1], 16384);
    CHECK_EQ(b.weights[1].d.shape[0], 3);
    CHECK_EQ(b.weights[2].d.shape[0], 24);
    for (const auto& weight : b.weights) {
        CHECK_EQ(weight.d.dtype, RAD_F32);
        CHECK_EQ(weight.d.shard, RAD_SHARD_NONE);
    }
    CHECK(b.ops[conn.op_pre - 1].reads == std::vector<rad_buf>({ w.h }));
    CHECK(b.ops[conn.op_pre - 1].writes == std::vector<rad_buf>({ w.x, w.post, w.comb }));
    CHECK(b.ops[conn.op_post - 1].reads == std::vector<rad_buf>({ w.y, w.h, w.post, w.comb }));
    CHECK(b.ops[conn.op_post - 1].writes == std::vector<rad_buf>({ w.out }));
    RadCtx ctx;
    conn.read(&ctx, 4);
    conn.write(&ctx, 4);
    REQUIRE_EQ(ctx.issues.size(), (size_t)2);
    CHECK_EQ(ctx.issues[0].op, conn.op_pre);
    CHECK_EQ(ctx.issues[1].op, conn.op_post);
    CHECK_EQ(ctx.issues[0].n, 4);
    CHECK_EQ(ctx.issues[1].n, 4);
    REQUIRE_EQ(ctx.issues[0].opd.size(), (size_t)7);
    CHECK_EQ(ctx.issues[0].opd[0].handle, w.h);
    const rad_weight coefficients[] = { conn.w_fn, conn.w_scale, conn.w_base };
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(ctx.issues[0].opd[i + 1].kind, RAD_OPK_WEIGHT);
        CHECK_EQ(ctx.issues[0].opd[i + 1].handle, coefficients[i]);
    }
    CHECK_EQ(ctx.issues[0].opd[4].handle, w.x);
    CHECK_EQ(ctx.issues[0].opd[5].handle, w.post);
    CHECK_EQ(ctx.issues[0].opd[6].handle, w.comb);
    REQUIRE_EQ(ctx.issues[1].opd.size(), (size_t)5);
    const rad_buf expected[] = { w.y, w.h, w.post, w.comb, w.out };
    for (int i = 0; i < 5; ++i) {
        CHECK_EQ(ctx.issues[1].opd[i].handle, expected[i]);
        CHECK_EQ(ctx.issues[1].opd[i].rows, 4);
    }
    auto invalid = w;
    invalid.out = w.h;
    ManifoldConn refused;
    CHECK_EQ(refused.declare(&b, nm, g, 3, "bad", cfg, invalid, src), RAD_E_INVAL);
}

TEST(manifold_read_splits_its_projection_by_chunks_over_ranks_from_the_row_threshold) {
    using namespace rad::arch;
    for (int world : { 1, 2, 4, 8 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            Names nm("");
            Geom g{};
            g.n_embd = 4096;
            g.max_tok = 512;
            g.world = world;
            g.rank = rank;
            ManifoldConn::Config cfg;
            ManifoldConn::Src src{ "model.layers.3.hc_attn_fn", "model.layers.3.hc_attn_scale",
                                   "model.layers.3.hc_attn_base" };
            ManifoldConn::Wire w;
            w.h = decl_b(&b, "h", RAD_BF16, { 512, 16384 });
            w.x = decl_b(&b, "x", RAD_BF16, { 512, 4096 });
            w.y = decl_b(&b, "y", RAD_BF16, { 512, 4096 });
            w.out = decl_b(&b, "out", RAD_BF16, { 512, 16384 });
            w.post = decl_b(&b, "post", RAD_F32, { 512, 4 });
            w.comb = decl_b(&b, "comb", RAD_F32, { 512, 16 });
            ManifoldConn conn;
            REQUIRE_EQ(conn.declare(&b, nm, g, 3, "blk.3.mhc_attn", cfg, w, src), RAD_OK);
            const bool split = world > 1;
            const int64_t mine = 8 / world;
            CHECK_EQ(count_op(b, "mhc_pre"), 1);
            CHECK_EQ(count_op(b, "mhc_pre_partial"), split ? 1 : 0);
            CHECK_EQ(count_op(b, "all_gather"), split ? 1 : 0);
            CHECK_EQ(count_op(b, "mhc_pre_finish"), split ? 1 : 0);
            CHECK_EQ(b.weights[0].d.shard, RAD_SHARD_NONE);
            if (split) {
                const RecOp* part = find_op(b, "mhc_pre_partial");
                const RecOp* gather = find_op(b, "all_gather");
                const RecOp* finish = find_op(b, "mhc_pre_finish");
                REQUIRE(part && gather && finish);
                CHECK_EQ(pint(part, "chunk"), 2048);
                CHECK_EQ(pint(part, "first"), rank * mine);
                CHECK_EQ(pint(part, "chunks"), mine);
                CHECK(part->w == std::vector<rad_weight>({ conn.w_fn }));
                CHECK(part->reads == std::vector<rad_buf>({ w.h }));
                CHECK(part->writes == std::vector<rad_buf>({ conn.part }));
                CHECK_EQ(pint(gather, "world_size"), world);
                CHECK_EQ(pint(gather, "row"), mine * 28);
                CHECK_EQ(param(gather, "numel")->kind, RAD_P_RANGE);
                CHECK_EQ(param(gather, "numel")->ival, mine * 28);
                CHECK_EQ(param(gather, "numel")->ihi, 512 * mine * 28);
                CHECK_EQ(pstr(gather, "dtype"), "f32");
                CHECK(gather->reads == std::vector<rad_buf>({ conn.part }));
                CHECK(gather->writes == std::vector<rad_buf>({ conn.parts }));
                CHECK_EQ(pint(finish, "chunk"), 2048);
                CHECK_EQ(pint(finish, "iterations"), 20);
                CHECK(finish->w == std::vector<rad_weight>({ conn.w_scale, conn.w_base }));
                CHECK(finish->reads == std::vector<rad_buf>({ conn.parts, w.h }));
                CHECK(finish->writes == std::vector<rad_buf>({ w.x, w.post, w.comb }));
                CHECK_EQ(find_b(b, "blk.3.mhc_attn.parts")->d.shape[1], 8 * 28);
            }
            RadCtx ctx;
            conn.read(&ctx, 300);
            conn.read(&ctx, 255);
            if (split) {
                REQUIRE_EQ(ctx.issues.size(), (size_t)4);
                CHECK_EQ(ctx.issues[0].op, conn.op_partial);
                CHECK_EQ(ctx.issues[0].opd[0].handle, w.h);
                CHECK_EQ(ctx.issues[0].opd[0].rows, 300);
                CHECK_EQ(ctx.issues[0].opd[1].handle, conn.w_fn);
                CHECK_EQ(ctx.issues[0].opd[2].handle, conn.part);
                CHECK_EQ(ctx.issues[1].op, conn.op_gather);
                CHECK_EQ(ctx.issues[1].n, 300 * mine * 28);
                CHECK_EQ(ctx.issues[1].opd[1].handle, conn.parts);
                CHECK_EQ(ctx.issues[1].opd[1].rows, 300);
                CHECK_EQ(ctx.issues[2].op, conn.op_finish);
                REQUIRE_EQ(ctx.issues[2].opd.size(), (size_t)7);
                const uint32_t expected[] = { conn.parts, w.h,    conn.w_scale, conn.w_base,
                                              w.x,        w.post, w.comb };
                for (int i = 0; i < 7; ++i) CHECK_EQ(ctx.issues[2].opd[i].handle, expected[i]);
                CHECK_EQ(ctx.issues[3].op, conn.op_pre);
            } else {
                REQUIRE_EQ(ctx.issues.size(), (size_t)2);
                CHECK_EQ(ctx.issues[0].op, conn.op_pre);
                CHECK_EQ(ctx.issues[1].op, conn.op_pre);
            }
        }
}

TEST(glm_routing_keeps_f32_logits_and_correction_replicated_under_tp4) {
    using namespace rad::arch;
    for (int rank = 0; rank < 4; ++rank) {
        RadBuilder b;
        Names nm("");
        Geom g{};
        g.n_embd = 4096;
        g.max_tok = 128;
        g.world = 4;
        g.rank = rank;
        GlmRouter::Config cfg;
        GlmRouter::Wire w;
        w.input = decl_b(&b, "hidden", RAD_BF16, { 128, 4096 });
        w.logits = decl_b(&b, "logits", RAD_F32, { 32, 288 });
        w.ids = decl_b(&b, "ids", RAD_I32, { 32, 8 });
        w.weights = decl_b(&b, "weights", RAD_F32, { 32, 8 });
        GlmRouter::Src src{ "model.layers.3.mlp.gate.weight",
                            "model.layers.3.mlp.gate.e_score_correction_bias" };
        GlmRouter router;
        REQUIRE_EQ(router.declare(&b, nm, g, 3, "blk.3.router", cfg, w, src), RAD_OK);
        REQUIRE_EQ(b.weights.size(), (size_t)2);
        for (const auto& weight : b.weights) {
            CHECK_EQ(weight.d.dtype, RAD_F32);
            CHECK_EQ(weight.d.shard, RAD_SHARD_NONE);
        }
        CHECK_EQ(b.weights[0].d.shape[0], 288);
        CHECK_EQ(b.weights[0].d.shape[1], 4096);
        CHECK_EQ(b.ops[router.op_project - 1].op, std::string("gemm_nt_f32"));
        CHECK_EQ(b.ops[router.op_select - 1].op, std::string("router_sigmoid_topk"));
        RadCtx ctx;
        router.step(&ctx, 7, 4);
        REQUIRE_EQ(ctx.issues.size(), (size_t)2);
        CHECK_EQ(ctx.issues[0].opd[0].handle, w.input);
        CHECK_EQ(ctx.issues[0].opd[0].offset, 7 * 4096);
        CHECK_EQ(ctx.issues[0].opd[0].rows, 4);
        CHECK_EQ(ctx.issues[1].opd[1].handle, router.correction);
        CHECK_EQ(ctx.issues[1].opd[2].handle, w.ids);
        CHECK_EQ(ctx.issues[1].opd[3].handle, w.weights);
    }
}

TEST(glm_router_and_index_head_split_columns_over_ranks_from_the_row_threshold) {
    using namespace rad::arch;
    for (int world : { 1, 2, 4, 8 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            Names nm("");
            Geom g{};
            g.n_embd = 4096;
            g.max_tok = 512;
            g.world = world;
            g.rank = rank;
            GlmRouter::Config cfg;
            cfg.rows = 512;
            GlmRouter::Wire w;
            w.input = decl_b(&b, "hidden", RAD_BF16, { 512, 4096 });
            w.logits = decl_b(&b, "logits", RAD_F32, { 512, 288 });
            w.ids = decl_b(&b, "ids", RAD_I32, { 512, 8 });
            w.weights = decl_b(&b, "weights", RAD_F32, { 512, 8 });
            GlmRouter::Src src{ "model.layers.3.mlp.gate.weight",
                                "model.layers.3.mlp.gate.e_score_correction_bias" };
            GlmRouter router;
            REQUIRE_EQ(router.declare(&b, nm, g, 3, "blk.3.router", cfg, w, src), RAD_OK);
            const bool split = world > 1;
            const int64_t cols = 288 / world;
            CHECK_EQ(count_op(b, "all_gather"), split ? 1 : 0);
            CHECK_EQ(count_op(b, "gemm_nt_f32"), split ? 2 : 1);
            // The weight stays whole and replicated: decode keeps the unsplit projection.
            CHECK_EQ(b.weights[0].d.shape[0], 288);
            CHECK_EQ(b.weights[0].d.shard, RAD_SHARD_NONE);
            if (split) {
                const RecOp* part = find_op(b, "gemm_nt_f32", 1);
                const RecOp* gather = find_op(b, "all_gather");
                REQUIRE(part && gather);
                CHECK_EQ(pint(part, "N"), cols);
                CHECK_EQ(pint(part, "K"), 4096);
                CHECK(part->w == std::vector<rad_weight>({ router.projection }));
                CHECK(part->reads == std::vector<rad_buf>({ w.input }));
                CHECK(part->writes == std::vector<rad_buf>({ router.split.part }));
                CHECK_EQ(pint(gather, "world_size"), world);
                CHECK_EQ(pint(gather, "row"), cols);
                CHECK_EQ(param(gather, "numel")->kind, RAD_P_RANGE);
                CHECK_EQ(param(gather, "numel")->ival, cols);
                CHECK_EQ(param(gather, "numel")->ihi, 512 * cols);
                CHECK_EQ(pstr(gather, "dtype"), "f32");
                CHECK(gather->reads == std::vector<rad_buf>({ router.split.part }));
                CHECK(gather->writes == std::vector<rad_buf>({ w.logits }));
            }
            RadCtx ctx;
            router.step(&ctx, 7, 300, 7);
            if (split) {
                REQUIRE_EQ(ctx.issues.size(), (size_t)3);
                CHECK_EQ(ctx.issues[0].op, router.split.op_part);
                CHECK_EQ(ctx.issues[0].n, 300);
                CHECK_EQ(ctx.issues[0].opd[0].handle, w.input);
                CHECK_EQ(ctx.issues[0].opd[0].offset, 7 * 4096);
                CHECK_EQ(ctx.issues[0].opd[0].rows, 300);
                CHECK_EQ(ctx.issues[0].opd[1].kind, RAD_OPK_WEIGHT);
                CHECK_EQ(ctx.issues[0].opd[1].handle, router.projection);
                CHECK_EQ(ctx.issues[0].opd[1].offset, rank * cols * 4096);
                CHECK_EQ(ctx.issues[0].opd[1].rows, cols);
                CHECK_EQ(ctx.issues[0].opd[2].handle, router.split.part);
                CHECK_EQ(ctx.issues[0].opd[2].rows, 300);
                CHECK_EQ(ctx.issues[1].op, router.split.op_gather);
                CHECK_EQ(ctx.issues[1].n, 300 * cols);
                CHECK_EQ(ctx.issues[1].opd[0].handle, router.split.part);
                CHECK_EQ(ctx.issues[1].opd[1].handle, w.logits);
                CHECK_EQ(ctx.issues[1].opd[1].offset, 7 * 288);
                CHECK_EQ(ctx.issues[1].opd[1].rows, 300);
                CHECK_EQ(ctx.issues[2].op, router.op_select);
            } else {
                REQUIRE_EQ(ctx.issues.size(), (size_t)2);
                CHECK_EQ(ctx.issues[0].op, router.op_project);
            }
            ctx.issues.clear();
            router.step(&ctx, 0, 255);
            REQUIRE_EQ(ctx.issues.size(), (size_t)2);
            CHECK_EQ(ctx.issues[0].op, router.op_project);
        }
}

TEST(absorbed_mla_shards_complete_heads_and_reads_one_latent_cache) {
    using namespace rad::arch;
    for (int rank = 0; rank < 4; ++rank) {
        RadBuilder b;
        b.kv_block = 4;
        Names nm("");
        Geom g{};
        g.n_embd = 4096;
        g.max_tok = 32;
        g.world = 4;
        g.rank = rank;
        RadKVGroupDecl kd{};
        kd.kind = RAD_KV_FULL;
        kd.dtype = RAD_BF16;
        kd.n_head_kv = 1;
        kd.head_dim = 256;
        const rad_kvgroup kv = rad_decl_kv_group(&b, "latent", &kd);
        MlaLatent::Config cfg;
        MlaLatent::Wire w;
        w.q = decl_b(&b, "q", RAD_BF16, { 32, 16, 256 });
        w.q_latent = decl_b(&b, "ql", RAD_BF16, { 32, 16, 512 });
        w.attn_latent = decl_b(&b, "al", RAD_BF16, { 32, 16, 512 });
        w.out = decl_b(&b, "out", RAD_BF16, { 32, 16, 256 });
        w.selected = decl_b(&b, "selected", RAD_I32, { 32, 513 });
        w.length = decl_b(&b, "length", RAD_I32, { 32 });
        MlaLatent block;
        REQUIRE_EQ(
            block.declare(&b, nm, g, 3, kv, cfg, w, "model.layers.3.self_attn.kv_b_proj.weight"),
            RAD_OK);
        REQUIRE_EQ(b.weights.size(), (size_t)1);
        CHECK_EQ(b.weights[0].d.dtype, RAD_BF16);
        CHECK_EQ(b.weights[0].d.shard, RAD_SHARD_ROW);
        CHECK_EQ(b.weights[0].d.shape[0], 8192);
        CHECK_EQ(b.weights[0].d.shape[1], 512);
        RadCtx ctx;
        block.step(&ctx, 4);
        REQUIRE_EQ(ctx.issues.size(), (size_t)3);
        CHECK_EQ(b.ops[ctx.issues[0].op - 1].op, std::string("mla_project"));
        CHECK_EQ(b.ops[ctx.issues[1].op - 1].op, std::string("mla_attn"));
        CHECK_EQ(b.ops[ctx.issues[2].op - 1].op, std::string("mla_project"));
        CHECK_EQ(ctx.issues[1].opd[1].kind, RAD_OPK_KV);
        CHECK_EQ(ctx.issues[1].opd[2].handle, w.selected);
        CHECK_EQ(ctx.issues[1].opd[3].handle, w.length);
        CHECK_EQ(ctx.issues[2].opd[1].handle, block.projection);
    }
}

/* GLM's MLA from the TP split threshold: each rank runs its run of the step's rows with all 64
 * heads, through q_b, kv_b and o at full width, and an all_gather lays the runs out, instead of
 * its heads over every row and an all-reduce. Runs are ceil(T / world) rows, the last one short;
 * the gather sends whole runs into an output of world runs. Below the threshold, and at TP1, the
 * head form is unchanged, as it is unless the row form is opted into (the default). */
TEST(glm_mla_splits_a_prefill_by_rows_over_ranks_from_the_row_threshold) {
    using namespace rad::arch;
    for (int world : { 1, 2, 4, 8 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            b.kv_block = 4;
            // The container's MLA entries; the row form's names are not among them.
            const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
            b.encs = { { "attn_q_a.weight", fp8 },
                       { "attn_kv_a.weight", fp8 },
                       { "attn_q_b.weight", fp8 },
                       { "attn_output.weight", fp8 },
                       { "attn_kv_b.weight", rad_enc_plain(RAD_BF16) } };
            Names nm("");
            Geom g{};
            g.n_embd = 4096;
            g.max_tok = 512;
            g.max_ctx = 4096;
            g.eps = 1e-5;
            g.world = world;
            g.rank = rank;
            RadKVGroupDecl kd{};
            kd.kind = RAD_KV_FULL;
            kd.dtype = RAD_BF16;
            kd.n_head_kv = 1;
            kd.head_dim = 256;
            const rad_kvgroup kv = rad_decl_kv_group(&b, "latent", &kd);
            const int64_t T = 512, R = GlmMla::rows_for(T, world), H = 64 / world;
            auto act = [&](ActFP8& a, const char* name, int64_t rows, int64_t width) {
                a.x = decl_b(&b, name, RAD_BF16, { rows, width });
                return a.declare_qs(&b, nm, g, name, width, rows);
            };
            GlmMla::Wire w;
            REQUIRE_EQ(act(w.input, "normalized", T, 4096), RAD_OK);
            w.q_raw = decl_b(&b, "q_raw", RAD_BF16, { T, 1536 });
            w.kv_raw = decl_b(&b, "kv_raw", RAD_BF16, { T, 512 });
            REQUIRE_EQ(act(w.q_low, "q_low", T, 1536), RAD_OK);
            w.latent = decl_b(&b, "latent", RAD_BF16, { T, 512 });
            w.out = decl_b(&b, "attn_delta", RAD_BF16, { world * R, 4096 });
            w.core.q = decl_b(&b, "query", RAD_BF16, { T, 16384 / world });
            w.core.q_latent = decl_b(&b, "q_latent", RAD_BF16, { T, H, 512 });
            w.core.attn_latent = decl_b(&b, "attn_latent", RAD_BF16, { T, H, 512 });
            REQUIRE_EQ(act(w.attn, "expanded", T, 16384 / world), RAD_OK);
            w.core.out = w.attn.x;
            w.core.selected = decl_b(&b, "selected", RAD_I32, { T, 513 });
            w.core.length = decl_b(&b, "length", RAD_I32, { T });
            w.rows.q = decl_b(&b, "rows_query", RAD_BF16, { R, 16384 });
            w.rows.q_latent = decl_b(&b, "rows_q_latent", RAD_BF16, { R, 64, 512 });
            w.rows.attn_latent = decl_b(&b, "rows_attn_latent", RAD_BF16, { R, 64, 512 });
            REQUIRE_EQ(act(w.rows.attn, "rows_expanded", R, 16384), RAD_OK);
            w.rows.out = decl_b(&b, "rows_out", RAD_BF16, { R, 4096 });
            GlmMla mla;
            REQUIRE_EQ(mla.declare(&b, nm, g, 3, w, "model.layers.3.self_attn", kv), RAD_OK);
            const bool rows = world > 1;
            CHECK_EQ(mla.op_rows_gather != 0, rows);
            CHECK_EQ(count_op(b, "all_gather"), rows ? 1 : 0);
            CHECK_EQ(count_op(b, "mla_attn"), rows ? 2 : 1);
            const RecWeight* head_q = find_w(b, "blk.3.attn_q_b.weight");
            REQUIRE(head_q);
            CHECK_EQ(head_q->d.shape[0], 16384 / world);
            if (!rows) {
                CHECK(!find_w(b, "blk.3.attn_q_b_rows.weight"));
                continue;
            }
            const RecWeight* q_b = find_w(b, "blk.3.attn_q_b_rows.weight");
            const RecWeight* kv_b = find_w(b, "blk.3.attn_kv_b_rows.weight");
            const RecWeight* o = find_w(b, "blk.3.attn_output_rows.weight");
            REQUIRE(q_b && kv_b && o);
            CHECK_EQ(q_b->d.shard, RAD_SHARD_NONE);
            CHECK_EQ(q_b->d.shape[0], 16384);
            CHECK_EQ(q_b->d.shape[1], 1536);
            CHECK_EQ(kv_b->d.shard, RAD_SHARD_NONE);
            CHECK_EQ(kv_b->d.dtype, RAD_BF16);
            CHECK_EQ(kv_b->d.shape[0], 32768);
            CHECK_EQ(kv_b->d.shape[1], 512);
            CHECK_EQ(o->d.shard, RAD_SHARD_NONE);
            CHECK_EQ(o->d.shape[0], 4096);
            CHECK_EQ(o->d.shape[1], 16384);
            // The full-width weights are whole views of the head form's container entries, so
            // they take its encoding (block fp8 at f32 scales) and need no name map of their own.
            const auto source = [](const RecWeight* w) {
                return std::string(w && w->d.source ? w->d.source : "");
            };
            CHECK_EQ(source(q_b), "blk.3.attn_q_b.weight");
            CHECK_EQ(source(find_w(b, "blk.3.attn_q_b_rows.scale")), "blk.3.attn_q_b.weight");
            CHECK_EQ(source(kv_b), "blk.3.attn_kv_b.weight");
            CHECK_EQ(source(o), "blk.3.attn_output.weight");
            CHECK_EQ(source(find_w(b, "blk.3.attn_output_rows.scale")), "blk.3.attn_output.weight");
            CHECK_EQ(find_w(b, "blk.3.attn_q_b_rows.scale")->d.dtype, RAD_F32);
            for (const auto& map : b.maps) CHECK(map.declared.find("_rows") == std::string::npos);
            CHECK_EQ(pstr(&b.ops[mla.rows_q_b.op - 1], "dtype"), "fp8a8f32");
            CHECK_EQ(pstr(&b.ops[mla.rows_output.op - 1], "dtype"), "fp8a8f32");
            CHECK_EQ(pstr(&b.ops[mla.q_b.op - 1], "dtype"), "fp8a8f32");
            const RecOp* attn = &b.ops[mla.op_rows_attn - 1];
            CHECK_EQ(pint(attn, "heads"), 64);
            CHECK_EQ(pint(attn, "blocks"), 513);
            CHECK_EQ(param(attn, "M")->ihi, R);
            CHECK(attn->reads ==
                  std::vector<rad_buf>({ w.rows.q_latent, w.core.selected, w.core.length }));
            const RecOp* gather = &b.ops[mla.op_rows_gather - 1];
            const RecOp* reduce = &b.ops[mla.op_ar - 1];
            CHECK_EQ(pint(gather, "world_size"), world);
            CHECK_EQ(pstr(gather, "dtype"), "bf16");
            CHECK(!param(gather, "row"));
            // The all-reduce's message bound, so the two share one exchange instance.
            CHECK_EQ(param(gather, "numel")->ihi, param(reduce, "numel")->ihi);
            CHECK(gather->reads == std::vector<rad_buf>({ w.rows.out }));
            CHECK(gather->writes == std::vector<rad_buf>({ w.out }));
            std::vector<int32_t> slots(T);
            RadKVGroupBatch kb{};
            kb.group = kv;
            kb.slot_mapping = slots.data();
            RadBatch batch{};
            batch.kv = &kb;
            batch.n_kv_groups = 1;
            for (int64_t steps : { 300, 255 }) {
                batch.n_tok = steps;
                RadCtx ctx;
                ctx.b = &b;
                ctx.rank = rank;
                ctx.world = world;
                mla.prepare(&ctx, &batch);
                mla.finish(&ctx, &batch);
                CHECK(ctx.step_fail.empty());
                auto issued = [&](rad_op op) {
                    return std::count_if(ctx.issues.begin(), ctx.issues.end(),
                                         [&](const RecIssue& i) { return i.op == op; });
                };
                const bool split = steps >= GlmTp::kSplitRows;
                CHECK_EQ(issued(mla.q_b.op), split ? 0 : 1);
                CHECK_EQ(issued(mla.core.op_attn), split ? 0 : 1);
                CHECK_EQ(issued(mla.op_ar), split ? 0 : 1);
                CHECK_EQ(issued(mla.op_rows_attn), split ? 1 : 0);
                CHECK_EQ(issued(mla.op_rows_gather), split ? 1 : 0);
                if (!split) continue;
                const int64_t run = (steps + world - 1) / world, first = rank * run,
                              n = std::min(run, steps - first);
                for (const auto& i : ctx.issues) {
                    if (i.op == mla.rows_q_b.op) {
                        CHECK_EQ(i.n, n);
                        CHECK_EQ(i.opd[0].handle, w.q_low.q);
                        CHECK_EQ(i.opd[0].offset, first * 1536);
                        CHECK_EQ(i.opd[0].rows, n);
                        CHECK_EQ(i.opd[2].handle, mla.rows_q_b.w);
                        CHECK_EQ(i.opd[4].handle, w.rows.q);
                        CHECK_EQ(i.opd[4].offset, 0);
                        CHECK_EQ(i.opd[4].rows, n);
                    } else if (i.op == mla.op_rows_attn) {
                        CHECK_EQ(i.n, n);
                        CHECK_EQ(i.opd[1].kind, RAD_OPK_KV);
                        CHECK_EQ(i.opd[2].handle, w.core.selected);
                        CHECK_EQ(i.opd[2].offset, first * 513);
                        CHECK_EQ(i.opd[2].rows, n);
                        CHECK_EQ(i.opd[3].handle, w.core.length);
                        CHECK_EQ(i.opd[3].offset, first);
                        CHECK_EQ(i.opd[3].rows, n);
                        CHECK_EQ(i.opd[4].handle, w.rows.attn_latent);
                    } else if (i.op == mla.rows_output.op) {
                        CHECK_EQ(i.n, n);
                        CHECK_EQ(i.opd[2].handle, mla.rows_output.w);
                        CHECK_EQ(i.opd[4].handle, w.rows.out);
                        CHECK_EQ(i.opd[4].offset, 0);
                    } else if (i.op == mla.op_rows_gather) {
                        CHECK_EQ(i.n, run * 4096);
                        CHECK_EQ(i.opd[0].handle, w.rows.out);
                        CHECK_EQ(i.opd[0].rows, run);
                        CHECK_EQ(i.opd[1].handle, w.out);
                        CHECK_EQ(i.opd[1].rows, world * run);
                    }
                }
                // Eight ranks at 300 rows: runs of 38, the last 34.
                if (world == 8) CHECK_EQ(n, rank == 7 ? 34 : 38);
            }
        }
}

TEST(
    glm53_native_graph_declares_all_layers_preserves_mixed_formats_and_runs_mtp_only_in_draft_passes) {
    using namespace rad::arch;
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    for (int world : { 1, 4, 8 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            b.kv_block = 4;
            b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
                       { "mtp.blk.45.ffn_down_exps", fp8 },
                       { "ffn_gate_up_exps", mx },
                       { "ffn_down_exps", mx },
                       { "ffn_gate_up_shexp", mx },
                       { "ffn_down_shexp", mx },
                       { "ffn_gate_up.weight", fp8 },
                       { "ffn_down.weight", fp8 } };
            RadBuildCtx build{};
            build.rank = rank;
            build.world_size = world;
            build.max_tok = rank == 0 ? 512 : 16;
            build.max_seqs = 4;
            build.max_ctx = 4096;
            build.max_spec = 3;
            // With the row form opted into (--tp-attn-split rows).
            build.tp_attn_rows = 1;
            REQUIRE_EQ(glm5next::declare(&b, &meta, &build), RAD_OK);
            const auto& m = *glm5next::models[rank];
            CHECK(m.have_mtp);
            CHECK_EQ(b.drafter.kind, RAD_DRAFT_SERIAL);
            CHECK_EQ(b.drafter.depth, 3);
            CHECK_EQ(m.layers.size(), (size_t)45);
            int full = 0;
            for (int l = 0; l < 45; ++l) {
                const auto& layer = m.layers[l];
                full += layer.full;
                CHECK(layer.attn_hc.op_pre < layer.norm.op);
                CHECK(layer.attn_hc.op_post < layer.ffn_hc.op_pre);
                if (layer.full) {
                    CHECK(layer.index.cache.op_select < layer.mla.core.op_attn);
                    CHECK(layer.mla.output.op < layer.attn_hc.op_post);
                } else
                    CHECK(layer.kda.output.op < layer.attn_hc.op_post);
                if (l >= 3) {
                    CHECK_EQ(layer.moe.c.expert_mx, 1);
                    CHECK(layer.moe.op_gather < layer.ffn_hc.op_post);
                }
            }
            CHECK_EQ(full, 11);
            CHECK_EQ(m.draft.moe.c.expert_f32_scale, 1);
            for (const auto& map : b.maps)
                if (map.declared.starts_with("mtp."))
                    for (const auto& source : map.src) CHECK(!source.starts_with("mtp.model."));
            int32_t cu[] = { 0, 2 }, positions[] = { 0, 1 }, ctxlens[] = { 0 }, ids[] = { 7, 9 },
                    slots[] = { 0, 1 }, out[] = { 1 };
            int32_t sidx[] = { 0, 0, 0, 1, 1, 1 }, ci[] = { 0 }, tables[32]{};
            std::vector<RadKVGroupBatch> kv(b.kv_groups.size());
            for (size_t i = 0; i < kv.size(); ++i) {
                kv[i].group = (rad_kvgroup)i + 1;
                kv[i].slot_mapping = slots;
                kv[i].block_table = tables;
                kv[i].block_table_pitch = 32;
                kv[i].state_index = b.kv_groups[i].d.kind == RAD_KV_CONV ? ci : sidx;
                kv[i].state_index_pitch = b.kv_groups[i].d.kind == RAD_KV_CONV ? 1 : 6;
            }
            RadBatch batch{};
            batch.phase = RAD_PHASE_PREFILL;
            batch.n_tok = 2;
            batch.n_seq = 1;
            batch.token_ids = ids;
            batch.positions = positions;
            batch.cu_seqlens = cu;
            batch.ctx_lens = ctxlens;
            batch.kv = kv.data();
            batch.n_kv_groups = (int)kv.size();
            batch.n_out = 1;
            batch.out_ids = out;
            RadCtx ctx;
            ctx.b = &b;
            ctx.rank = rank;
            ctx.world = world;
            ctx.batch = &batch;
            glm5next::step(&ctx, &batch);
            CHECK(ctx.step_fail.empty());
            const auto ops = issued_ops(b, ctx);
            CHECK_EQ(std::count(ops.begin(), ops.end(), "mhc_pre"), 90);
            CHECK_EQ(std::count(ops.begin(), ops.end(), "kda_update"), 34);
            CHECK_EQ(std::count(ops.begin(), ops.end(), "mla_attn"), 11);
            // Two rows stay under the split threshold, and their history fits the selection
            // (DsaCache::takes_all): the 42 routers run, the 11 index heads and scorers do not.
            CHECK_EQ(std::count(ops.begin(), ops.end(), "gemm_nt_f32"), 42);
            CHECK_EQ(std::count(ops.begin(), ops.end(), "dsa_query"), 0);
            CHECK_EQ(std::count(ops.begin(), ops.end(), "dsa_score"), 0);
            CHECK_EQ(std::count(ops.begin(), ops.end(), "qsa_select"), 11);
            CHECK_EQ(std::count(ops.begin(), ops.end(), "all_gather"), 0);
            // The split is declared on every rank whatever its step size; only rank 0 issues it here.
            const bool split = world > 1;
            // 43 routers (MTP included), the 90 mHC reads, each of the 12 indexers' page
            // tables and lengths (the index head gates follow the indexer's rows,
            // ungathered), and the 11 MLA row forms (the MTP layer keeps its heads).
            CHECK_EQ(count_op(b, "all_gather"), split ? 43 + 90 + 24 + 11 : 0);
            CHECK_EQ(count_op(b, "mhc_pre_partial"), split ? 90 : 0);
            ctx.issues.clear();
            batch.draft_pass = -1;
            batch.n_draft_out = 1;
            batch.draft_out_ids = out;
            glm5next::step(&ctx, &batch);
            CHECK(ctx.step_fail.empty());
            const auto draft_ops = issued_ops(b, ctx);
            CHECK_EQ(std::count(draft_ops.begin(), draft_ops.end(), "mhc_pre"), 0);
            CHECK_EQ(std::count(draft_ops.begin(), draft_ops.end(), "mla_attn"), 1);
            CHECK_EQ(ctx.issues.back().op, m.mtp.op_head);
            CHECK(m.mtp.op_select < m.draft.moe.op_router);
            for (const auto& issue : ctx.issues) {
                if (issue.op == m.draft.moe.op_router || issue.op == m.mtp.op_norm)
                    CHECK_EQ(issue.opd[0].rows, 1);   // Only the row producing a proposal.
                if (issue.op == m.mtp.op_residual)
                    CHECK_EQ(issue.opd[0].rows, 2);   // Full history still fills attention caches.
            }
            ctx.issues.clear();
            batch.n_draft_out = 0;
            glm5next::step(&ctx, &batch);
            const auto history = issued_ops(b, ctx);
            CHECK(std::find(history.begin(), history.end(), "kv_store") != history.end());
            CHECK(std::find(history.begin(), history.end(), "moe_gemm_q") == history.end());
            if (rank == 0) {
                std::vector<int32_t> long_ids(512, 7), long_positions(512), long_slots(512),
                    long_tables(1024);
                for (int i = 0; i < 512; ++i) long_positions[i] = long_slots[i] = i;
                int32_t long_cu[] = { 0, 512 }, last = 511;
                for (auto& group : kv) {
                    group.slot_mapping = long_slots.data();
                    group.block_table = long_tables.data();
                    group.block_table_pitch = 1024;
                }
                batch.n_tok = 512;
                batch.token_ids = long_ids.data();
                batch.positions = long_positions.data();
                batch.cu_seqlens = long_cu;
                batch.out_ids = &last;
                batch.draft_pass = 0;
                batch.max_q_len = 512;
                {
                    // 512 rows whose history fits the selection: the indexer takes every
                    // block and does not split, so no index head runs and nothing of its is
                    // gathered, while the routers, mHC reads and MLA still split.
                    ctx.issues.clear();
                    glm5next::step(&ctx, &batch);
                    CHECK(ctx.step_fail.empty());
                    const auto fit_ops = issued_ops(b, ctx);
                    CHECK_EQ(std::count(fit_ops.begin(), fit_ops.end(), "gemm_nt_f32"), 42);
                    CHECK_EQ(std::count(fit_ops.begin(), fit_ops.end(), "dsa_score"), 0);
                    CHECK_EQ(std::count(fit_ops.begin(), fit_ops.end(), "qsa_select"), 11);
                    CHECK_EQ(std::count(fit_ops.begin(), fit_ops.end(), "all_gather"),
                             world > 1 ? 42 + 90 + 11 : 0);
                }
                batch.max_ctx_len = 3000;   // live blocks past the top-k: the indexer scores
                ctx.issues.clear();
                glm5next::step(&ctx, &batch);
                CHECK(ctx.step_fail.empty());
                {
                    // 512 rows reach the split: each router runs on its column slice, each mHC
                    // read on its chunks, each indexer, its head gates included, on its run
                    // of query rows, and each MLA on its run of rows, all gathered but the
                    // gates. At TP1 the step does not split, and the indexer scores whole.
                    const auto long_ops = issued_ops(b, ctx);
                    CHECK_EQ(std::count(long_ops.begin(), long_ops.end(), "gemm_nt_f32"), 53);
                    CHECK_EQ(std::count(long_ops.begin(), long_ops.end(), "all_gather"),
                             world > 1 ? 42 + 90 + 22 + 11 : 0);
                    // 1 embedding, 45 attention and 45 expert reductions, less the 11 MLA ones.
                    CHECK_EQ(std::count(long_ops.begin(), long_ops.end(), "all_reduce"),
                             world > 1 ? 91 - 11 : 0);
                    CHECK_EQ(std::count(long_ops.begin(), long_ops.end(), "mla_attn"), 11);
                    CHECK_EQ(std::count(long_ops.begin(), long_ops.end(), "mhc_pre"),
                             world > 1 ? 0 : 90);
                    CHECK_EQ(std::count(long_ops.begin(), long_ops.end(), "mhc_pre_finish"),
                             world > 1 ? 90 : 0);
                    for (const auto& issue : ctx.issues) {
                        if (b.ops[issue.op - 1].op != "gemm_nt_f32") continue;
                        const int64_t n = pint(&b.ops[issue.op - 1], "N");
                        if (world == 1 || n == 32)
                            CHECK(n == 288 || n == 32);
                        else {
                            CHECK_EQ(n, 288 / world);
                            CHECK_EQ(issue.opd[1].rows, n);
                        }
                        if (world > 1 && n == 32)
                            CHECK_EQ(issue.opd[0].rows, DsaCache::part_rows(512, world));
                    }
                }
                for (int l = 3; l < 45; ++l) {
                    // A prefill routes its full batch once, reusing each expert's
                    // weights across its rows instead of re-reading them in chunks.
                    CHECK_EQ(std::count_if(ctx.issues.begin(), ctx.issues.end(),
                                           [&](const auto& issue) {
                                               const auto& moe = m.layers[l].moe;
                                               return issue.op == moe.op_router ||
                                                      issue.op == moe.glm_router.split.op_part;
                                           }),
                             1);
                }
                // Arena-lending probes must not replace the model the serving
                // graph's handles refer to, even when they declare fewer rows.
                const auto* live = glm5next::models[rank].get();
                RadBuilder pb;
                pb.kv_block = 4;
                pb.encs = b.encs;
                auto small = build;
                small.shape_probe = 1;
                small.max_tok = 1;
                REQUIRE_EQ(glm5next::declare(&pb, &meta, &small), RAD_OK);
                CHECK_EQ(glm5next::models[rank].get(), live);
                CHECK_EQ(glm5next::models[rank]->g.max_tok, 512);
                RadArchProbe capabilities{};
                REQUIRE_EQ(glm5next::probe(&meta, &capabilities), RAD_OK);
                CHECK_EQ(capabilities.shape_probe_ok, 1);
                CHECK_EQ(capabilities.tp_attn_rows_ok, 1);
                std::array<int, 8> results{};
                std::vector<std::thread> probes;
                for (int i = 0; i < 8; ++i)
                    probes.emplace_back([&, i] {
                        RadBuilder parallel;
                        parallel.kv_block = 4;
                        parallel.encs = b.encs;
                        auto level = build;
                        level.rank = i % 4;
                        level.max_tok = i < 4 ? 256 : 1024;
                        level.shape_probe = 1;
                        results[i] = glm5next::declare(&parallel, &meta, &level);
                    });
                for (auto& thread : probes) thread.join();
                for (int status : results) CHECK_EQ(status, RAD_OK);
                CHECK_EQ(glm5next::models[rank].get(), live);
            }
        }
}

/* From the split threshold each rank scores its run of the step's query rows (rows past the step
 * included, which score -inf) and selects for the rows of it that exist, against the bounds the
 * scorer rebased to the run's first row; two all_gathers join the runs' page tables and lengths
 * into planes that hold every run. The query side (projection, cast, quantiser, head gates) runs
 * on the run's rows that exist. Below the threshold the indexer stays replicated. */
TEST(glm53_indexer_scores_and_selects_one_run_of_query_rows_per_rank) {
    using namespace rad::arch;
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    for (int world : { 2, 8 })
        for (int rank : { 0, world - 1 }) {
            RadBuilder b;
            b.kv_block = 4;
            b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
                       { "mtp.blk.45.ffn_down_exps", fp8 },
                       { "ffn_gate_up_exps", mx },
                       { "ffn_down_exps", mx },
                       { "ffn_gate_up_shexp", mx },
                       { "ffn_down_shexp", mx },
                       { "ffn_gate_up.weight", fp8 },
                       { "ffn_down.weight", fp8 } };
            RadBuildCtx build{};
            build.rank = rank;
            build.world_size = world;
            build.max_tok = 512;
            build.max_seqs = 8;
            build.max_ctx = 65536;
            build.max_spec = 5;
            REQUIRE_EQ(glm5next::declare(&b, &meta, &build), RAD_OK);
            const auto& m = *glm5next::models[rank];
            const auto& cache = m.layers[3].index.cache;
            const auto& w = m.iw.cache;
            const int64_t most = DsaCache::part_rows(512, world);
            const RecOp* score = b.op_of(cache.op_score_part);
            const RecOp* sel = b.op_of(cache.op_gather_sel);
            const RecOp* len = b.op_of(cache.op_gather_len);
            REQUIRE(score && sel && len && cache.op_select_part);
            CHECK_EQ(pint(score, "row_part"), rank);
            CHECK_EQ(param(score, "M")->ihi, most);
            CHECK(score->writes == std::vector<rad_buf>({ w.score, w.cu_part }));
            CHECK_EQ(param(sel, "numel")->ival, 4 * 513);
            CHECK_EQ(param(sel, "numel")->ihi, most * 513);
            CHECK_EQ(param(len, "numel")->ihi, most);
            CHECK(sel->reads == std::vector<rad_buf>({ w.sel_part }));
            CHECK(sel->writes == std::vector<rad_buf>({ w.selected }));
            CHECK(len->writes == std::vector<rad_buf>({ w.length }));
            CHECK_EQ(b.bufs[w.selected - 1].d.shape[0], world * most);
            CHECK_EQ(b.bufs[w.codes - 1].d.shape[0], world * most * 32);
            CHECK_EQ(b.bufs[w.sel_part - 1].d.shape[0], most);
            // Wide page tables, so that live blocks follow the history past the selection.
            auto widen = [](Kit& kt) {
                const int64_t pitch = 1024;
                kt.table.resize((size_t)(kt.b.n_seq * pitch));
                for (int64_t i = 0; i < kt.b.n_seq * pitch; ++i) kt.table[(size_t)i] = (int32_t)i;
                for (auto& g : kt.kv) {
                    g.block_table = kt.table.data();
                    g.block_table_pitch = g.max_blocks = pitch;
                }
            };
            {
                // 500 rows whose history fits the selection take every block without splitting:
                // each rank selects the whole step from the unscored plane, and nothing gathers.
                auto fit = kit(b, { 1, 1 }, { 498 }, 100);
                widen(*fit);
                auto fits = run_passes(b, glm5next::step, { fit.get() }, rank, world);
                CHECK_PASSES(b, fits);
                int all = 0;
                for (const auto& issue : fits[0]->issues) {
                    CHECK(issue.op != cache.op_score_part && issue.op != cache.op_gather_sel &&
                          issue.op != cache.op_score);
                    if (issue.op == cache.op_select_all) {
                        ++all;
                        CHECK_EQ(issue.n, 500);
                        CHECK_EQ(issue.opd[0].handle, w.unscored);
                    }
                }
                CHECK_EQ(all, 1);
            }
            // Two decode rows beside a 498-row chunk of 3000 tokens of history: 500 rows that
            // score, in runs of R.
            auto k = kit(b, { 1, 1 }, { 498 }, 3000);
            widen(*k);
            const int64_t T = 500, R = DsaCache::part_rows(T, world), r0 = rank * R,
                          have = std::min(R, T - r0);
            auto passes = run_passes(b, glm5next::step, { k.get() }, rank, world);
            CHECK_PASSES(b, passes);
            const auto& index = m.layers[3].index;
            // The bf16 query projection follows the runs only where every run is over 64 rows.
            const bool projected = T - (world - 1) * R > 64;
            int scored = 0, selected = 0, gathered = 0, replicated = 0, fronts = 0;
            for (const auto& issue : passes[0]->issues) {
                if (issue.op == index.query.op) {
                    ++fronts;
                    CHECK_EQ(issue.opd[0].rows, projected ? have : T);
                    CHECK_EQ(issue.opd[0].offset, projected ? r0 * 1536 : 0);
                }
                if (issue.op == index.op_query_copy || issue.op == cache.op_query) {
                    ++fronts;
                    CHECK_EQ(issue.n, have * 32);
                    CHECK_EQ(issue.opd[1].offset, r0 * 32 * 128);
                }
                if (issue.op == index.op_head) {
                    ++fronts;
                    CHECK_EQ(issue.n, have);
                    CHECK_EQ(issue.opd[0].offset, r0 * 4096);
                    CHECK_EQ(issue.opd[2].offset, r0 * 32);
                }
                if (issue.op == cache.op_score) ++replicated;
                if (issue.op == cache.op_score_part) {
                    ++scored;
                    CHECK_EQ(issue.n, R);
                    CHECK_EQ(issue.opd[0].rows, world * R * 32);
                    CHECK_EQ(issue.opd[2].rows, world * R);
                    CHECK_EQ(issue.opd[8].rows, R);
                    CHECK_EQ(issue.opd[9].handle, w.cu_part);
                }
                if (issue.op == cache.op_select_part) {
                    ++selected;
                    CHECK_EQ(issue.n, have);
                    CHECK(issue.opd[3].raw == (void*)(k->pos.data() + r0));
                    CHECK_EQ(issue.opd[3].rows, have);
                    CHECK_EQ(issue.opd[4].handle, w.cu_part);
                    CHECK_EQ(issue.opd[5].handle, w.sel_part);
                    CHECK_EQ(issue.opd[7].handle, w.len_part);
                }
                if (issue.op == cache.op_gather_sel) {
                    ++gathered;
                    CHECK_EQ(issue.n, R * 513);
                    CHECK_EQ(issue.opd[1].rows, world * R);
                }
                if (issue.op == cache.op_gather_len) CHECK_EQ(issue.n, R);
            }
            // The handles are layer 3's: one issue each.
            CHECK_EQ(scored, 1);
            CHECK_EQ(selected, have > 0 ? 1 : 0);
            CHECK_EQ(gathered, 1);
            CHECK_EQ(replicated, 0);
            CHECK_EQ(fronts, 4);
            // Under the threshold every rank scores and selects the whole step. 3000 tokens of
            // history reach 815 blocks, past the selection, and 255 rows of them stay under
            // kSplitWork.
            auto small = kit(b, {}, { 255 }, 3000);
            {
                const int64_t pitch = 1024;
                small->table.resize((size_t)pitch);
                for (int64_t i = 0; i < pitch; ++i) small->table[(size_t)i] = (int32_t)i;
                for (auto& g : small->kv) {
                    g.block_table = small->table.data();
                    g.block_table_pitch = g.max_blocks = pitch;
                }
            }
            auto under = run_passes(b, glm5next::step, { small.get() }, rank, world);
            CHECK_PASSES(b, under);
            int whole = 0;
            for (const auto& issue : under[0]->issues) {
                CHECK(issue.op != cache.op_score_part && issue.op != cache.op_gather_sel);
                whole += issue.op == cache.op_score;
                // The engine refuses an issue whose operand count is not the schema's: dsa_score
                // declares ten (cu_part optional, passed absent).
                if (issue.op == cache.op_score || issue.op == cache.op_score_part)
                    CHECK_EQ(issue.opd.size(), (size_t)10);
                if (issue.op == index.op_head) CHECK_EQ(issue.n, 255);
            }
            CHECK_EQ(whole, 1);
            // Without that history every row selects all its blocks: no rank runs the query side
            // or scores, and the selection reads zeros over the selection's width.
            auto fits = kit(b, {}, { 255 });
            auto all = run_passes(b, glm5next::step, { fits.get() }, rank, world);
            CHECK_PASSES(b, all);
            int unscored = 0, picks = 0;
            for (const auto& issue : all[0]->issues) {
                CHECK(issue.op != cache.op_score && issue.op != cache.op_score_part &&
                      issue.op != cache.op_query && issue.op != index.op_query_copy &&
                      issue.op != index.op_head && issue.op != index.query.op);
                if (issue.op == cache.op_unscored) {
                    ++unscored;
                    CHECK_EQ(issue.n, 255);
                    CHECK_EQ(issue.opd[0].handle, w.unscored);
                    CHECK_EQ(b.bufs[w.unscored - 1].d.shape[1], cache.unscored_cols());
                }
                CHECK(issue.op != cache.op_select);
                picks += issue.op == cache.op_select_all;
            }
            CHECK_EQ(unscored, 1);
            CHECK_EQ(picks, 1);
            // A verify of eight sequences at 32k of context (48 rows x 8014 blocks) splits too, a
            // single sequence's six rows do not; a rank whose run lies past the step still writes
            // the runs it gathers.
            for (int seqs : { 8, 1 }) {
                auto dec = kit(b, std::vector<int>((size_t)seqs, 6), {});
                const int64_t pitch = 8192;
                dec->table.resize((size_t)(seqs * pitch));
                for (int64_t i = 0; i < seqs * pitch; ++i) dec->table[(size_t)i] = (int32_t)i;
                for (auto& g : dec->kv) {
                    g.block_table = dec->table.data();
                    g.block_table_pitch = g.max_blocks = pitch;
                }
                dec->b.max_ctx_len = 32000;
                auto v = run_passes(b, glm5next::step, { dec.get() }, rank, world);
                CHECK_PASSES(b, v);
                const int64_t Td = 6 * seqs, Rd = DsaCache::part_rows(Td, world);
                const bool split_d = seqs == 8;
                int parts = 0, gathers = 0, wholes = 0;
                for (const auto& issue : v[0]->issues) {
                    parts += issue.op == cache.op_score_part;
                    gathers += issue.op == cache.op_gather_sel;
                    wholes += issue.op == cache.op_score;
                    if (issue.op == cache.op_score_part) CHECK_EQ(issue.n, Rd);
                    if (issue.op == cache.op_score || issue.op == cache.op_score_part)
                        CHECK_EQ(issue.opd.size(), (size_t)10);
                    if (split_d &&
                        (issue.op == cache.op_select_part || issue.op == cache.op_query)) {
                        const int64_t rows_d = rank * Rd < Td ? std::min(Rd, Td - rank * Rd) : Rd;
                        CHECK_EQ(issue.n, issue.op == cache.op_query ? rows_d * 32 : rows_d);
                    }
                }
                CHECK_EQ(gathers, split_d ? 1 : 0);
                CHECK_EQ(parts, split_d ? 1 : 0);
                CHECK_EQ(wholes, split_d ? 0 : 1);
            }
        }
}

/* The engine's arena levels declare again at a smaller max_tok (RadBuildCtx::shape_probe) and
 * must get the real declare's graph: the same ops with the same fixed parameters, only ranged row
 * counts and buffer rows smaller. The TP splits declare its ops whatever the step size and their
 * gathers' message as a range, in the head and the row form of the MLA. */
TEST(glm53_sizing_declares_match_the_real_graph_at_every_arena_level) {
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    const int world = 8, rank = 3;
    for (int rows_split : { 0, 1 })
    for (int64_t R : { 256, 3072 }) {
        auto encs = [&](RadBuilder& b) {
            b.kv_block = 4;
            b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
                       { "mtp.blk.45.ffn_down_exps", fp8 },
                       { "ffn_gate_up_exps", mx },
                       { "ffn_down_exps", mx },
                       { "ffn_gate_up_shexp", mx },
                       { "ffn_down_shexp", mx },
                       { "ffn_gate_up.weight", fp8 },
                       { "ffn_down.weight", fp8 } };
        };
        RadBuildCtx c{};
        c.rank = rank;
        c.world_size = world;
        c.max_tok = 4096;
        c.max_seqs = 8;
        c.max_ctx = 32768;
        c.max_spec = 3;
        c.tp_attn_rows = rows_split;
        RadBuilder full;
        encs(full);
        REQUIRE_EQ(glm5next::declare(&full, &meta, &c), RAD_OK);
        RadBuildCtx p = c;
        p.max_tok = R;
        p.shape_probe = 1;
        RadBuilder small;
        encs(small);
        REQUIRE_EQ(glm5next::declare(&small, &meta, &p), RAD_OK);
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
                // A capacity may shrink: its kernel writes by its operands' extents.
                if (full.ops[i].op == "qsa_work" && a.key == "work" && q.ival <= a.ival)
                    continue;
                if (a.ival != q.ival || a.sval != q.sval || a.dval != q.dval) {
                    fprintf(stderr,
                            "    R %lld: op %zu '%s' param '%s': "
                            "%lld real, %lld small\n",
                            (long long)R, i, full.ops[i].op.c_str(), a.key.c_str(),
                            a.ival, q.ival);
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

/* Each deployment or checkpoint the plugin cannot serve is refused with a line that names the
 * setting and the reason, not a bare status. */
TEST(glm53_refusals_name_what_was_refused) {
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    RadBuildCtx ok{};
    ok.rank = 0;
    ok.world_size = 8;
    ok.max_tok = 64;
    ok.max_seqs = 2;
    ok.max_ctx = 4096;
    ok.max_spec = 3;
    auto refusal = [&](const RadModelMeta& mt, const RadBuildCtx& c) {
        RadBuilder b;
        b.kv_block = 4;
        int rc = RAD_OK;
        const std::string err = stderr_of([&] { rc = glm5next::declare(&b, &mt, &c); });
        CHECK_EQ(rc, RAD_E_UNSUPPORTED);
        return err;
    };
    RadBuildCtx c = ok;
    c.kv_dtype = RAD_F16;
    CHECK(has(refusal(meta, c), "--kv-cache-dtype f16"));
    c = ok;
    c.max_spec = 32;
    CHECK(has(refusal(meta, c), "--num-speculative-tokens 32"));
    c = ok;
    c.world_size = 3;
    CHECK(has(refusal(meta, c), "--tp 3"));
    RadModelMeta other = meta;
    other.n_vocab = 151552;
    CHECK(has(refusal(other, ok), "vocab_size is 151552"));
    /* A KDA projection stored quantised: the block's linears run as bf16 GEMMs, and the refusal
     * names the weight. */
    {
        const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
        const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
        RadBuilder b;
        b.kv_block = 4;
        b.encs = { { "blk.1.attn_beta", fp8 },        { "mtp.blk.45.ffn_gate_up_exps", fp8 },
                   { "mtp.blk.45.ffn_down_exps", fp8 }, { "ffn_gate_up_exps", mx },
                   { "ffn_down_exps", mx },            { "ffn_gate_up_shexp", mx },
                   { "ffn_down_shexp", mx },           { "ffn_gate_up.weight", fp8 },
                   { "ffn_down.weight", fp8 } };
        int rc = RAD_OK;
        const std::string err = stderr_of([&] { rc = glm5next::declare(&b, &meta, &ok); });
        CHECK(rc < 0);
        CHECK(has(err, "blk.1.attn_beta.weight"));
        CHECK(has(err, "plain bf16"));
    }
}

/* --kv-cache-dtype fp8 narrows the latent attention cache alone to E4M3: its group, every latent
 * store (plain and streamed) and every latent attention (head form, row form, the MTP layer's)
 * name fp8_e4m3, and the index tail, DSA index and KDA groups keep their dtypes. Under bf16 the
 * attention states no kv_dtype, the op it always declared. */
TEST(glm53_fp8_kv_cache_narrows_the_latent_group_only) {
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    for (int kv_dtype : { (int)RAD_DT_INVALID, (int)RAD_BF16, (int)RAD_F8E4M3 }) {
        const bool narrow = kv_dtype == RAD_F8E4M3;
        RadBuilder b;
        b.kv_block = 4;
        b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
                   { "mtp.blk.45.ffn_down_exps", fp8 },
                   { "ffn_gate_up_exps", mx },
                   { "ffn_down_exps", mx },
                   { "ffn_gate_up_shexp", mx },
                   { "ffn_down_shexp", mx },
                   { "ffn_gate_up.weight", fp8 },
                   { "ffn_down.weight", fp8 } };
        RadBuildCtx build{};
        build.rank = 0;
        build.world_size = 4;
        build.max_tok = 512;
        build.max_seqs = 4;
        build.max_ctx = 4096;
        build.max_spec = 3;
        build.tp_attn_rows = 1;
        build.kv_dtype = kv_dtype;
        REQUIRE_EQ(glm5next::declare(&b, &meta, &build), RAD_OK);
        std::map<std::string, uint32_t> dtypes;
        for (const auto& g : b.kv_groups) dtypes[g.name] = g.d.dtype;
        CHECK_EQ(dtypes["kv_latent"], narrow ? (uint32_t)RAD_F8E4M3 : (uint32_t)RAD_BF16);
        CHECK_EQ(dtypes["kv_index_tail"], (uint32_t)RAD_BF16);
        CHECK_EQ(dtypes["kv_dsa_index"], (uint32_t)RAD_U8);
        CHECK_EQ(dtypes["kv_kda"], (uint32_t)RAD_F32);
        CHECK_EQ(dtypes["kv_kda_conv"], (uint32_t)RAD_BF16);
        int stores = 0, attns = 0;
        for (const auto& o : b.ops) {
            if (o.op == "kv_store" || o.op == "kv_store_streamed") {
                ++stores;
                CHECK_EQ(pstr(&o, "kv_dtype"), narrow ? "fp8_e4m3" : "bf16");
            }
            if (o.op == "mla_attn") {
                ++attns;
                CHECK_EQ(pstr(&o, "kv_dtype"), narrow ? "fp8_e4m3" : "<absent>");
            }
        }
        // 11 MLA layers and the MTP layer, each a store and a head-form attention; 11 row forms.
        CHECK(stores >= 12);
        CHECK_EQ(attns, 12 + 11);
    }
}

/* The MoE block's decode fusions: over a two-row trunk step, its draft pass and a 512-row prefill
 * (past the router fusion's one-workgroup sort), each fused launch stands where it fits and the
 * launches it replaces are gone, with the step lint clean throughout. The sort writes each slot's
 * row (`moe_scatter_inv`) and the gather reads it (`moe_gather_inv`) in every MoE pass. */
TEST(glm53_moe_decode_fusions_replace_their_launches_where_they_fit) {
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    using Counts = std::map<std::string, int>;
    /* Launches per step: trunk, draft pass, prefill. The trunk has 42 MoE layers, each a routed
     * and a shared clamped SwiGLU, and three dense ones. */
    struct Want {
        const char* op;
        int trunk, draft, prefill;
    };
    const Want wants[] = {
        { "router_sigmoid_topk_scatter", 42, 1, 0 }, { "router_sigmoid_topk", 0, 0, 42 },
        { "swiglu_clamped_quant_fp8", 87, 2, 87 },   { "swiglu_clamped", 0, 0, 0 },
        { "moe_scatter", 0, 0, 0 },                  { "moe_gather", 0, 0, 0 },
    };
    for (int world : { 1, 8 }) {
        RadBuilder b;
        b.kv_block = 4;
        b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
                   { "mtp.blk.45.ffn_down_exps", fp8 },
                   { "ffn_gate_up_exps", mx },
                   { "ffn_down_exps", mx },
                   { "ffn_gate_up_shexp", mx },
                   { "ffn_down_shexp", mx },
                   { "ffn_gate_up.weight", fp8 },
                   { "ffn_down.weight", fp8 } };
        RadBuildCtx build{};
        build.rank = 0;
        build.world_size = world;
        build.max_tok = 512;
        build.max_seqs = 4;
        build.max_ctx = 4096;
        build.max_spec = 3;
        REQUIRE_EQ(glm5next::declare(&b, &meta, &build), RAD_OK);
        int32_t cu[] = { 0, 2 }, positions[] = { 0, 1 }, ctxlens[] = { 0 }, ids[] = { 7, 9 },
                slots[] = { 0, 1 }, out[] = { 1 };
        int32_t sidx[] = { 0, 0, 0, 1, 1, 1 }, ci[] = { 0 }, tables[32]{};
        std::vector<RadKVGroupBatch> kv(b.kv_groups.size());
        for (size_t i = 0; i < kv.size(); ++i) {
            kv[i].group = (rad_kvgroup)i + 1;
            kv[i].slot_mapping = slots;
            kv[i].block_table = tables;
            kv[i].block_table_pitch = 32;
            kv[i].state_index = b.kv_groups[i].d.kind == RAD_KV_CONV ? ci : sidx;
            kv[i].state_index_pitch = b.kv_groups[i].d.kind == RAD_KV_CONV ? 1 : 6;
        }
        RadBatch batch{};
        batch.phase = RAD_PHASE_PREFILL;
        batch.n_tok = 2;
        batch.n_seq = 1;
        batch.token_ids = ids;
        batch.positions = positions;
        batch.cu_seqlens = cu;
        batch.ctx_lens = ctxlens;
        batch.kv = kv.data();
        batch.n_kv_groups = (int)kv.size();
        batch.n_out = 1;
        batch.out_ids = out;
        RadCtx ctx;
        ctx.b = &b;
        ctx.rank = 0;
        ctx.world = world;
        ctx.batch = &batch;
        Counts got[3];
        /* With the inverse, the fused selection writes it as its ninth operand and the gather
         * reads it as its third. */
        const rad_buf inv = find_b(b, "expert_inverse")
                                ? (rad_buf)(find_b(b, "expert_inverse") - &b.bufs[0] + 1)
                                : 0;
        REQUIRE(inv != 0);
        auto run = [&](int k) {
            ctx.issues.clear();
            glm5next::step(&ctx, &batch);
            CHECK(ctx.step_fail.empty());
            CHECK_LINT(b, &ctx);
            for (const auto& op : issued_ops(b, ctx)) ++got[k][op];
            for (const auto& is : ctx.issues) {
                const RecOp* o = b.op_of(is.op);
                if (!o) continue;
                if (o->op == "router_sigmoid_topk_scatter")
                    CHECK(is.opd.size() > 8 && is.opd[8].kind == RAD_OPK_BUF &&
                          is.opd[8].handle == inv);
                if (o->op == "moe_gather_inv")
                    CHECK(is.opd.size() > 2 && is.opd[2].kind == RAD_OPK_BUF &&
                          is.opd[2].handle == inv);
            }
        };
        run(0);
        batch.draft_pass = -1;
        batch.n_draft_out = 1;
        batch.draft_out_ids = out;
        run(1);
        std::vector<int32_t> long_ids(512, 7), long_positions(512), long_slots(512),
            long_tables(1024);
        for (int i = 0; i < 512; ++i) long_positions[i] = long_slots[i] = i;
        int32_t long_cu[] = { 0, 512 }, last = 511;
        for (auto& group : kv) {
            group.slot_mapping = long_slots.data();
            group.block_table = long_tables.data();
            group.block_table_pitch = 1024;
        }
        batch.n_tok = 512;
        batch.token_ids = long_ids.data();
        batch.positions = long_positions.data();
        batch.cu_seqlens = long_cu;
        batch.out_ids = &last;
        batch.draft_pass = 0;
        batch.n_draft_out = 0;
        batch.max_q_len = 512;
        run(2);
        for (int k = 0; k < 3; ++k)
            for (const Want& w : wants) {
                const int want = k == 0 ? w.trunk : k == 1 ? w.draft : w.prefill;
                if (got[k][w.op] != want)
                    fprintf(stderr, "    world %d step %d: %s %d (want %d)\n", world, k, w.op,
                            got[k][w.op], want);
                CHECK_EQ(got[k][w.op], want);
            }
    }
}

TEST(dsa_cache_uses_distinct_index_and_attention_pages_and_a_speculative_tail_ring) {
    using namespace rad::arch;
    for (bool front : { false, true }) {
        RadBuilder b;
        b.kv_block = 4;
        Names nm("");
        Geom g{};
        g.n_embd = 4096;
        g.max_tok = 32;
        g.max_seqs = 4;
        g.max_spec = 3;
        RadKVGroupDecl ad{};
        ad.kind = RAD_KV_FULL;
        ad.dtype = RAD_BF16;
        ad.n_head_kv = 1;
        ad.head_dim = 256;
        const auto attn = rad_decl_kv_group(&b, "attn", &ad);
        // Claim the attention group's consumer before declaring the index group.
        RAD_OP(&b, "mla_attn", RAD_PARAMS(RAD_INT("block_size", 4)), RAD_NOWEIGHTS);
        RadKVGroupDecl td{};
        td.kind = RAD_KV_LINEAR;
        td.dtype = RAD_BF16;
        td.n_head_kv = 1;
        td.state_dim[0] = 7;
        td.state_dim[1] = 256;
        const auto tail = rad_decl_kv_group(&b, "tail", &td);
        DsaCache::Wire w;
        w.query = decl_b(&b, "query", RAD_BF16, { 1024, 128 });
        w.codes = decl_b(&b, "codes", RAD_F8E4M3, { 1024, 128 });
        w.scales = decl_b(&b, "scales", RAD_F32, { 1024, 1 });
        w.gate = decl_b(&b, "gates", RAD_F32, { 32, 32 });
        w.kg = decl_b(&b, "kg", RAD_BF16, { 32, 256 });
        w.stage = decl_b(&b, "stage", RAD_BF16, { 48, 256 });
        w.page = decl_b(&b, "page", RAD_I32, { 12 });
        w.bpos = decl_b(&b, "bpos", RAD_I32, { 12 });
        w.nc = decl_b(&b, "nc", RAD_I32, { 4 });
        w.pooled = decl_b(&b, "pooled", RAD_F8E4M3, { 12, 128 });
        w.pool_scale = decl_b(&b, "pool_scale", RAD_F32, { 12, 1 });
        w.score = decl_b(&b, "score", RAD_F32, { 32, 1024 });
        w.unscored = decl_b(&b, "unscored", RAD_F32, { 32, 513 });
        w.selected = decl_b(&b, "sel", RAD_I32, { 32, 513 });
        w.nsel = decl_b(&b, "nsel", RAD_I32, { 32 });
        w.length = decl_b(&b, "len", RAD_I32, { 32 });
        DsaCache block;
        DsaCache::Config cfg;
        cfg.blocks = 1024;
        cfg.work = 12;
        cfg.ring = 7;
        rad_kvgroup index = 0;
        GlmIndexer indexer;
        if (front) {
            GlmIndexer::Wire iw;
            iw.cache = w;
            iw.input = decl_b(&b, "input", RAD_BF16, { 32, 4096 });
            iw.q_low = decl_b(&b, "q_low", RAD_BF16, { 32, 1536 });
            iw.q_wide = decl_b(&b, "q_wide", RAD_BF16, { 32, 4096 });
            iw.kg_raw = decl_b(&b, "kg_raw", RAD_BF16, { 32, 256 });
            REQUIRE_EQ(indexer.declare(&b, nm, g, 3, iw, "indexer", cfg, attn, tail, &index),
                       RAD_OK);
            block = indexer.cache;
            CHECK_EQ(b.weights[indexer.head_weight - 1].d.dtype, RAD_F32);
            CHECK_EQ(b.weights[indexer.query.w - 1].d.shard, RAD_SHARD_NONE);
            CHECK_EQ(b.weights[indexer.key_gate.w - 1].d.shard, RAD_SHARD_NONE);
        } else
            REQUIRE_EQ(block.declare(&b, nm, g, 3, cfg, w, attn, tail, &index, "indexer.pool.ape"),
                       RAD_OK);
        CHECK_EQ(b.kv_groups[index - 1].d.dtype, RAD_U8);
        CHECK_EQ(b.kv_groups[index - 1].d.head_dim, 17);
        /* Eight table columns cap the live reach below the selection, so every row takes all its
         * blocks (DsaCache::takes_all); 4000 tokens of history over 1024 columns reach 1004 blocks,
         * past it, and score. */
        for (const int64_t pitch : { 8, 1024 }) {
            const bool scores = pitch > 8;
            std::vector<int32_t> ibt((size_t)(2 * pitch)), abt((size_t)(2 * pitch));
            int32_t sidx[12]{}, cu[] = { 0, 3, 8 }, pos[8]{};
            RadKVGroupBatch kv[3]{};
            kv[0].group = index;
            kv[0].block_table = ibt.data();
            kv[0].block_table_pitch = pitch;
            kv[1].group = tail;
            kv[1].state_index = sidx;
            kv[1].state_index_pitch = 6;
            kv[2].group = attn;
            kv[2].block_table = abt.data();
            kv[2].block_table_pitch = pitch;
            RadBatch batch{};
            batch.n_tok = 8;
            batch.n_seq = 2;
            batch.n_spec = 2;
            batch.kv = kv;
            batch.n_kv_groups = 3;
            batch.positions = pos;
            batch.cu_seqlens = cu;
            // At eight columns the mixed-row bound exceeds the physical table.
            batch.max_ctx_len = scores ? 4000 : 64;
            const int64_t live = scores ? 1004 : 8;
            CHECK_EQ(block.live_blocks(&batch), live);
            CHECK_EQ(block.takes_all(&batch), !scores);
            RadCtx ctx;
            ctx.b = &b;
            ctx.batch = &batch;
            if (front) {
                indexer.step(&ctx, &batch);
                if (scores) {
                    REQUIRE_EQ(ctx.issues.size(), (size_t)12);
                    CHECK_EQ(ctx.issues[1].op, indexer.op_query_copy);
                    CHECK_EQ(ctx.issues[1].opd[1].rows, 256);
                    CHECK_EQ(ctx.issues[3].op, indexer.op_norm);
                    CHECK_EQ(ctx.issues[4].opd[0].offset, 128);
                    CHECK_EQ(ctx.issues[5].op, indexer.op_head);
                    ctx.issues.erase(ctx.issues.begin(), ctx.issues.begin() + 6);
                } else {
                    // The key side alone: key and gate projection, norm, gate copy.
                    REQUIRE_EQ(ctx.issues.size(), (size_t)8);
                    CHECK_EQ(ctx.issues[1].op, indexer.op_norm);
                    CHECK_EQ(ctx.issues[2].op, indexer.op_gate_copy);
                    ctx.issues.erase(ctx.issues.begin(), ctx.issues.begin() + 3);
                }
            } else
                block.step(&ctx, &batch);
            const size_t q = scores ? 1 : 0;   // the query quantiser leads a scored step
            REQUIRE_EQ(ctx.issues.size(), 5 + q);
            if (scores) CHECK_EQ(ctx.issues[0].n, 256);
            CHECK(ctx.issues[q].opd[5].raw == ibt.data());
            CHECK_EQ(ctx.issues[q].opd[4].cols, 6);
            CHECK_EQ(ctx.issues[q + 1].op, block.op_pool);
            CHECK_EQ(ctx.issues[q + 1].opd[5].handle, index);
            CHECK_EQ(ctx.issues[q + 2].op, block.op_tail);
            const auto& fill = ctx.issues[q + 3];
            if (scores) {
                CHECK_EQ(fill.op, block.op_score);
                CHECK_EQ(fill.opd[0].rows, 256);
                CHECK_EQ(fill.opd[3].handle, index);
                CHECK_EQ(fill.opd[4].kind, RAD_OPK_NONE);
                CHECK(fill.opd[5].raw == ibt.data());
                CHECK_EQ(fill.opd[8].cols, live);
            } else {
                CHECK_EQ(fill.op, block.op_unscored);
                CHECK_EQ(fill.n, 8);
                CHECK_EQ(fill.opd[0].handle, w.unscored);
                CHECK_EQ(fill.opd[0].rows, 8);
            }
            CHECK_EQ(ctx.issues[q + 4].op, scores ? block.op_select : block.op_select_all);
            CHECK_EQ(ctx.issues[q + 4].opd[0].handle, scores ? w.score : w.unscored);
            CHECK_EQ(ctx.issues[q + 4].opd[0].cols, live);
            CHECK(ctx.issues[q + 4].opd[1].raw == abt.data());
        }
    }
}

TEST(glm_dense_ffn_keeps_clipping_and_disables_the_unclipped_fp8_epilogue) {
    using namespace rad::arch;
    RadBuilder b;
    Names nm("");
    Geom g{};
    g.n_embd = 4096;
    g.n_ff = 3072;
    g.max_tok = 32;
    g.world = 4;
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    b.encs = { { "ffn_gate_up", fp8 }, { "ffn_down", fp8 } };
    MlpFP8::Wire w;
    w.x = decl_b(&b, "x", RAD_BF16, { 32, 4096 });
    w.h.x = decl_b(&b, "h", RAD_BF16, { 32, 4096 });
    REQUIRE_EQ(w.h.declare_qs(&b, nm, g, "h", 4096), RAD_OK);
    w.gate_up = decl_b(&b, "gate_up", RAD_BF16, { 32, 6144 });
    w.ffn.x = decl_b(&b, "ffn", RAD_BF16, { 32, 3072 });
    REQUIRE_EQ(w.ffn.declare_qs(&b, nm, g, "ffn", 3072), RAD_OK);
    MlpFP8 block;
    MlpFP8::Src src{ "post_attention_layernorm.weight", "mlp.gate_proj.weight",
                     "mlp.up_proj.weight", "mlp.down_proj.weight" };
    REQUIRE_EQ(block.declare(&b, nm, g, 0, w, src, false, false, kArNone, false, 10), RAD_OK);
    CHECK_EQ(block.op_gated, 0u);
    CHECK_EQ(b.ops[block.gq_ffn.op - 1].op, std::string("swiglu_clamped"));
    RadCtx ctx;
    ctx.b = &b;
    block.slice(&ctx, 8, 4, 4);
    const auto ops = issued_ops(b, ctx);
    CHECK_EQ(std::count(ops.begin(), ops.end(), "swiglu_clamped"), 1);
    CHECK(std::find(ops.begin(), ops.end(), "gemm_nt_q_gated") == ops.end());
    CHECK(std::find(ops.begin(), ops.end(), "quant_act_fp8") != ops.end());
}

TEST(glm_mla_stores_one_normalized_latent_and_exposes_q_lora_before_sparse_attention) {
    using namespace rad::arch;
    for (int rank = 0; rank < 4; ++rank) {
        RadBuilder b;
        b.kv_block = 4;
        Names nm("");
        Geom g{};
        g.n_embd = 4096;
        g.max_tok = 32;
        g.world = 4;
        g.rank = rank;
        RadKVGroupDecl kd{};
        kd.kind = RAD_KV_FULL;
        kd.dtype = RAD_BF16;
        kd.n_head_kv = 1;
        kd.head_dim = 256;
        const auto kv = rad_decl_kv_group(&b, "latent", &kd);
        auto buf = [&](const char* name, int64_t n) {
            return decl_b(&b, name, RAD_BF16, { 32, n });
        };
        GlmMla::Wire w;
        w.input.x = buf("input", 4096);
        w.q_raw = buf("q_raw", 1536);
        w.kv_raw = buf("kv_raw", 512);
        w.q_low.x = buf("q_low", 1536);
        w.latent = buf("latent_out", 512);
        w.out = buf("out", 4096);
        REQUIRE_EQ(w.input.declare_qs(&b, nm, g, "input", 4096), RAD_OK);
        REQUIRE_EQ(w.q_low.declare_qs(&b, nm, g, "q_low", 1536), RAD_OK);
        w.core.q = buf("q", 4096);
        w.core.q_latent = decl_b(&b, "q_latent", RAD_BF16, { 32, 16, 512 });
        w.core.attn_latent = decl_b(&b, "attn_latent", RAD_BF16, { 32, 16, 512 });
        w.core.out = buf("expanded", 4096);
        w.attn.x = w.core.out;
        REQUIRE_EQ(w.attn.declare_qs(&b, nm, g, "expanded", 4096), RAD_OK);
        w.core.selected = decl_b(&b, "selected", RAD_I32, { 32, 513 });
        w.core.length = decl_b(&b, "length", RAD_I32, { 32 });
        GlmMla block;
        REQUIRE_EQ(block.declare(&b, nm, g, 3, w, "attn", kv), RAD_OK);
        CHECK_EQ(b.weights[block.q_a.w - 1].d.shard, RAD_SHARD_NONE);
        CHECK_EQ(b.weights[block.q_b.w - 1].d.shard, RAD_SHARD_ROW);
        CHECK_EQ(b.weights[block.output.w - 1].d.shard, RAD_SHARD_COL);
        int32_t slots[4]{};
        RadKVGroupBatch kb{};
        kb.group = kv;
        kb.slot_mapping = slots;
        RadBatch batch{};
        batch.n_tok = 4;
        batch.kv = &kb;
        batch.n_kv_groups = 1;
        RadCtx ctx;
        ctx.b = &b;
        ctx.batch = &batch;
        block.prepare(&ctx, &batch);
        auto find_issue = [&](rad_op op) -> const RecIssue* {
            for (const auto& issue : ctx.issues)
                if (issue.op == op) return &issue;
            return nullptr;
        };
        const auto* store = find_issue(block.op_store);
        REQUIRE(store);
        CHECK_EQ(store->opd[0].handle, w.latent);
        CHECK_EQ(store->opd[0].cols, 256);
        CHECK_EQ(store->opd[1].offset, 256);
        CHECK_EQ(store->opd[1].cols, 256);
        CHECK(store->opd[2].raw == slots);
        const auto before = issued_ops(b, ctx);
        CHECK(std::find(before.begin(), before.end(), "mla_attn") == before.end());
        block.finish(&ctx, &batch);
        const auto* attn = find_issue(block.core.op_attn);
        REQUIRE(attn);
        CHECK_EQ(attn->opd[2].handle, w.core.selected);
        CHECK_EQ(ctx.issues.back().op, block.op_ar);
    }
}

TEST(glm_kda_projects_local_heads_and_reduces_only_after_the_output_projection) {
    using namespace rad::arch;
    for (int rank = 0; rank < 4; ++rank) {
        RadBuilder b;
        Names nm("");
        Geom g{};
        g.n_embd = 4096;
        g.max_tok = 32;
        g.world = 4;
        g.rank = rank;
        g.max_spec = 3;
        RadKVGroupDecl sd{};
        sd.kind = RAD_KV_LINEAR;
        sd.dtype = RAD_F32;
        sd.n_head_kv = 16;
        sd.state_dim[0] = sd.state_dim[1] = 128;
        RadKVGroupDecl cd{};
        cd.kind = RAD_KV_CONV;
        cd.dtype = RAD_BF16;
        cd.n_head_kv = 1;
        cd.head_dim = 6144;
        cd.conv_width = 4;
        const auto state = rad_decl_kv_group(&b, "state", &sd),
                   history = rad_decl_kv_group(&b, "conv", &cd);
        auto buf = [&](const char* name, int64_t width) {
            return decl_b(&b, name, RAD_BF16, { 32, width });
        };
        GlmKda::Wire w;
        w.input = buf("input", 4096);
        w.forget_low = buf("forget_low", 128);
        w.gate_low = buf("gate_low", 128);
        w.out = buf("out", 4096);
        w.core.qkv = buf("qkv", 6144);
        w.core.conv = buf("conv_out", 6144);
        w.core.a = buf("a", 2048);
        w.core.beta = buf("beta", 16);
        w.core.gate = buf("gate", 2048);
        w.core.out = buf("core_out", 2048);
        KdaState::Src src{ "attn.q_conv1d.weight", "attn.k_conv1d.weight", "attn.v_conv1d.weight",
                           "attn.A_log",           "attn.dt_bias",         "attn.o_norm.weight" };
        GlmKda block;
        REQUIRE_EQ(block.declare(&b, nm, g, 0, w, "attn", src, state, history), RAD_OK);
        CHECK_EQ(b.weights[block.qkv.w - 1].d.n_row_parts, 3);
        CHECK_EQ(b.weights[block.forget_a.w - 1].d.shard, RAD_SHARD_NONE);
        CHECK_EQ(b.weights[block.forget_b.w - 1].d.shard, RAD_SHARD_ROW);
        CHECK_EQ(b.weights[block.output.w - 1].d.shard, RAD_SHARD_COL);
        int32_t cu[] = { 0, 4 }, ctxlens[] = { 0 }, indices[] = { 0, 0, 0, 1, 1, 1 }, ci[] = { 0 };
        RadKVGroupBatch kv[2]{};
        kv[0].group = state;
        kv[0].state_index = indices;
        kv[0].state_index_pitch = 6;
        kv[1].group = history;
        kv[1].state_index = ci;
        RadBatch batch{};
        batch.phase = RAD_PHASE_PREFILL;
        batch.n_tok = 4;
        batch.n_seq = 1;
        batch.cu_seqlens = cu;
        batch.ctx_lens = ctxlens;
        batch.kv = kv;
        batch.n_kv_groups = 2;
        RadCtx ctx;
        ctx.b = &b;
        ctx.batch = &batch;
        block.step(&ctx, &batch);
        const auto ops = issued_ops(b, ctx);
        CHECK_EQ(std::count(ops.begin(), ops.end(), "gemm_nt"), 7);
        CHECK_EQ(std::count(ops.begin(), ops.end(), "silu"), 0);
        REQUIRE_EQ(ctx.issues.size(), (size_t)11);
        // The input's four readers first, then both up-projections.
        CHECK_EQ(ctx.issues[2].op, block.forget_a.op);
        CHECK_EQ(ctx.issues[3].op, block.gate_a.op);
        CHECK_EQ(ctx.issues[4].op, block.forget_b.op);
        CHECK_EQ(ctx.issues[5].op, block.gate_b.op);
        CHECK_EQ(ctx.issues[9].op, block.output.op);
        CHECK_EQ(ctx.issues[10].op, block.op_ar);
        CHECK_EQ(ctx.issues[10].opd[0].handle, w.out);
    }
}

TEST(kda_state_core_splits_mixed_batches_and_preserves_absolute_prefill_offsets) {
    using namespace rad::arch;
    RadBuilder b;
    Names nm("");
    Geom g{};
    g.n_embd = 4096;
    g.max_tok = 32;
    g.world = 4;
    g.max_spec = 3;
    RadKVGroupDecl sd{};
    sd.kind = RAD_KV_LINEAR;
    sd.dtype = RAD_F32;
    sd.n_head_kv = 16;
    sd.state_dim[0] = sd.state_dim[1] = 128;
    RadKVGroupDecl cd{};
    cd.kind = RAD_KV_CONV;
    cd.dtype = RAD_BF16;
    cd.n_head_kv = 1;
    cd.head_dim = 6144;
    cd.conv_width = 4;
    const rad_kvgroup st = rad_decl_kv_group(&b, "state", &sd),
                      cv = rad_decl_kv_group(&b, "conv", &cd);
    KdaState::Wire w;
    w.qkv = decl_b(&b, "qkv", RAD_BF16, { 32, 6144 });
    w.conv = decl_b(&b, "filtered", RAD_BF16, { 32, 6144 });
    w.a = decl_b(&b, "a", RAD_BF16, { 32, 2048 });
    w.beta = decl_b(&b, "beta", RAD_BF16, { 32, 16 });
    w.gate = decl_b(&b, "gate", RAD_BF16, { 32, 2048 });
    w.out = decl_b(&b, "out", RAD_BF16, { 32, 16, 128 });
    KdaState::Src src{ "attn.q_conv1d.weight", "attn.k_conv1d.weight", "attn.v_conv1d.weight",
                       "attn.A_log",           "attn.dt_bias",         "attn.o_norm.weight" };
    KdaState block;
    KdaState::Config cfg;
    REQUIRE_EQ(block.declare(&b, nm, g, 0, st, cv, cfg, w, src), RAD_OK);
    CHECK_EQ(b.weights[block.taps - 1].d.n_row_parts, 3);
    for (int p = 0; p < 3; ++p) CHECK_EQ(b.weights[block.taps - 1].d.row_parts[p], 2048);
    CHECK_EQ(b.weights[block.bias - 1].d.shape[0], 2048);
    for (rad_op op : { block.op_prefill, block.op_verify }) {
        bool found = false;
        for (const auto& p : b.ops[op - 1].p)
            if (p.key == "state_form") {
                CHECK_EQ(p.sval, std::string("anchor"));
                found = true;
            }
        CHECK(found);
    }
    int32_t cu[] = { 0, 3, 8 }, ctxlens[] = { 11, 0 }, accepted[] = { 2, 1 };
    int32_t indices[] = { 0, 0, 0, 2, 2, 2, 1, 1, 1, 3, 3, 3 }, conv_indices[] = { 0, 1 };
    RadKVGroupBatch kv[2]{};
    kv[0].group = st;
    kv[0].state_index = indices;
    kv[0].state_index_pitch = 6;
    kv[1].group = cv;
    kv[1].state_index = conv_indices;
    kv[1].state_index_pitch = 1;
    RadBatch batch{};
    batch.phase = RAD_PHASE_MIXED;
    batch.n_tok = 8;
    batch.n_seq = 2;
    batch.n_seq_decode = 1;
    batch.n_tok_decode = 3;
    batch.cu_seqlens = cu;
    batch.ctx_lens = ctxlens;
    batch.num_accepted = accepted;
    batch.kv = kv;
    batch.n_kv_groups = 2;
    RadCtx ctx;
    ctx.b = &b;
    ctx.batch = &batch;
    block.step(&ctx, &batch);
    REQUIRE_EQ(ctx.issues.size(), (size_t)5);
    CHECK_EQ(ctx.issues[0].op, block.op_conv);
    CHECK_EQ(ctx.issues[0].n, 3);
    CHECK_EQ(ctx.issues[1].op, block.op_verify);
    CHECK_EQ(ctx.issues[1].opd[0].cols, 2048);
    CHECK_EQ(ctx.issues[1].opd[1].offset, 2048);
    CHECK_EQ(ctx.issues[1].opd[2].offset, 4096);
    CHECK_EQ(ctx.issues[2].op, block.op_conv);
    CHECK_EQ(ctx.issues[2].n, 8);
    CHECK(ctx.issues[2].opd[3].raw == cu + 1);
    CHECK_EQ(ctx.issues[2].opd[7].kind, RAD_OPK_NONE);
    CHECK_EQ(ctx.issues[3].op, block.op_prefill);
    CHECK(ctx.issues[3].opd[8].raw == indices + 6);
    CHECK(ctx.issues[3].opd[9].raw == cu + 1);
    CHECK_EQ(ctx.issues[3].opd[0].offset, 0);
    CHECK_EQ(ctx.issues[3].opd[0].rows, 8);
    CHECK_EQ(ctx.issues[4].op, block.op_norm);
}

/* ==================================================================== the pinned graph and steps */

/* GLM declares and steps the pinned graph and launches at one and eight ranks: a 4096-row prefill
 * after history, a mixed step, a decode step, a verify and its draft passes, at eight in both
 * MLA forms. */
TEST(glm53_is_the_pinned_graph_and_steps) {
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    struct Case {
        int world;
        uint64_t graph, steps;
        bool heads_only = false;   /* the default: the graph before the row form */
        bool file_tier = false;    /* --weights-disk-tier: whole experts at four ranks and up */
    };
    /* Pins from this test's own `pin:` printout: the head form (heads_only, which declares the
     * streamed ops) and the row form, at one and eight ranks. Re-recorded after every graph
     * change -- the merge re-recorded them for main's builder plus this branch's file-direct
     * expert ops, and the graphs again when the routing counts became RAD_BUF_PERSIST (the
     * steps did not change). */
    const Case cases[] = { { 1, 0xce4368c09bfdbbeeull, 0x19679d4fcc0321ebull },
                           /* Two ranks keep whole experts; four and eight split them by width. */
                           { 2, 0xab04b95724840abeull, 0x76865603b1390635ull },
                           { 4, 0xfdee91cd1c985b0aull, 0xde7ebe5b604c3cd2ull },
                           { 8, 0x59112e7ad7d77fdeull, 0xb9c29d30003a6df7ull },
                           { 8, 0xf1bb904ab21d8813ull, 0xfb28d3dcd8ec5ae1ull, true },
                           /* The file tier's whole experts, file-direct, at four and eight. */
                           { 4, 0x1918906f91fad4e7ull, 0x07d518b9a0f32986ull, false, true },
                           { 8, 0x553f81a93cec76bfull, 0x17f0a90d3bed9713ull, false, true },
                           { 8, 0x9685be82c8d55813ull, 0x75e775f0568636b5ull, true, true } };
    for (const Case& cs : cases) {
        RadBuilder b;
        b.kv_block = 4;
        b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
                   { "mtp.blk.45.ffn_down_exps", fp8 },
                   { "ffn_gate_up_exps", mx },
                   { "ffn_down_exps", mx },
                   { "ffn_gate_up_shexp", mx },
                   { "ffn_down_shexp", mx },
                   { "ffn_gate_up.weight", fp8 },
                   { "ffn_down.weight", fp8 } };
        RadBuildCtx ctx = spec_ctx(0, cs.world, 3);
        ctx.max_tok = 4096;
        ctx.max_seqs = 8;
        ctx.max_ctx = 200000;
        ctx.tp_attn_rows = cs.heads_only ? 0 : 1;
        ctx.expert_file_tier = cs.file_tier;
        REQUIRE_EQ(glm5next::declare(&b, &meta, &ctx), RAD_OK);
        const uint64_t g = graph_hash(b);
        const uint64_t s = pinned_steps(b, glm5next::step, 0, cs.world);
        fprintf(stderr, "    pin: glm5next world %d%s%s: graph 0x%016llxull, steps 0x%016llxull\n",
                cs.world, cs.heads_only ? " heads only" : "", cs.file_tier ? " file tier" : "",
                (unsigned long long)g, (unsigned long long)s);
        CHECK_EQ(g, cs.graph);
        CHECK_EQ(s, cs.steps);
    }
}

/* ==================================================================== GLM-5.3's vision tower */
namespace {

/* GLM-5.3-Flash's container keys with its vision_config, the tower cut to two blocks at the real
 * widths, enough to see a block repeat. */
MetaKV glm53_vl_meta() {
    RadModelMeta m{};
    m.n_layers = 45;
    m.n_embd = 4096;
    m.n_vocab = 154880;
    m.rms_eps = 1e-5f;
    MetaKV k(m);
    k.set("num_nextn_predict_layers", "1").set("hc_mult", "4").set("n_routed_experts", "288")
     .set("index_topk", "2048")
     .set("vision_config.model_type", "glm5_next_vision")
     .set("vision_config.depth", "2")
     .set("vision_config.hidden_size", "1024")
     .set("vision_config.num_heads", "16")
     .set("vision_config.intermediate_size", "4096")
     .set("vision_config.patch_size", "14")
     .set("vision_config.temporal_patch_size", "2")
     .set("vision_config.in_channels", "3")
     .set("vision_config.spatial_merge_size", "2")
     .set("vision_config.out_hidden_size", "4096")
     .set("vision_config.projection_intermediate_size", "10240")
     .set("vision_config.swiglu_limit", "10")
     .set("vision_config.hidden_act", "silu")
     .set("vision_config.attention_bias", "1")
     .set("vision_config.rms_norm_eps", "1.0000000000000001e-05");
    return k;
}

void glm53_encs(RadBuilder& b) {
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    b.kv_block = 4;
    b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
               { "mtp.blk.45.ffn_down_exps", fp8 },
               { "ffn_gate_up_exps", mx },
               { "ffn_down_exps", mx },
               { "ffn_gate_up_shexp", mx },
               { "ffn_down_shexp", mx },
               { "ffn_gate_up.weight", fp8 },
               { "ffn_down.weight", fp8 } };
}

RadBuildCtx glm53_vl_ctx(int rank, int world, int64_t patches, int max_spec = 3) {
    RadBuildCtx c = spec_ctx(rank, world, max_spec);
    c.max_tok = 512;
    c.max_seqs = 4;
    c.max_ctx = 4096;
    c.max_enc_patches = patches;
    return c;
}

}  /* namespace */

/* The latent and the index cache are computed whole on every rank (rad_block_glm_mla.h,
 * rad_block_glm_indexer.h) and the plugin says so, which lets the host tier keep each once a slot
 * under tensor parallelism instead of once a rank. The linear groups travel in the checkpoint
 * store and carry no such flag. The flag is outside graph_hash(), so the pinned graphs stand. */
TEST(glm53_declares_its_latent_and_index_caches_replicated) {
    MetaKV mk = glm53_vl_meta();
    auto ends_with = [](const std::string& s, const char* t) {
        const size_t n = std::strlen(t);
        return s.size() >= n && s.compare(s.size() - n, n, t) == 0;
    };
    for (int world : { 1, 2, 8 }) {
        for (int rank : { 0, world - 1 }) {
            RadBuilder b;
            glm53_encs(b);
            RadBuildCtx ctx = glm53_vl_ctx(rank, world, 0);
            REQUIRE_EQ(glm5next::declare(&b, mk.meta(), &ctx), RAD_OK);
            REQUIRE_EQ(b.kv_groups.size(), 5u);
            int replicated = 0;
            for (const RecKV& g : b.kv_groups) {
                const bool want = ends_with(g.name, "kv_latent") || ends_with(g.name, "kv_dsa_index");
                CHECK(g.replicated == want);
                CHECK(!g.zero);
                replicated += g.replicated;
            }
            CHECK_EQ(replicated, 2);
        }
    }
}

/* The tower is declared only with an encoder budget, last, its weights replicated in their own
 * groups past the 45 layers and the MTP layer's, its 2-D rotary the axial one at sixteen
 * frequencies an axis, gate and up joined gate first; one scatter over the trunk's embedding and
 * one over the draft head's. */
TEST(glm53_declares_its_vision_tower_when_the_engine_asks_for_it) {
    MetaKV mk = glm53_vl_meta();
    for (int world : { 1, 8 }) {
        const int rank = world - 1;
        {
            RadBuilder b;
            glm53_encs(b);
            RadBuildCtx ctx = glm53_vl_ctx(rank, world, 0);
            REQUIRE_EQ(glm5next::declare(&b, mk.meta(), &ctx), RAD_OK);
            CHECK(b.encoder_name.empty());
            CHECK(find_w(b, "v.patch.weight") == nullptr);
            CHECK_EQ(count_op(b, "scatter_rows"), 0);
        }
        RadBuilder b;
        glm53_encs(b);
        RadBuildCtx ctx = glm53_vl_ctx(rank, world, 1024);
        REQUIRE_EQ(glm5next::declare(&b, mk.meta(), &ctx), RAD_OK);
        const auto& m = *glm5next::models[rank];
        CHECK_EQ(b.encoder_name, std::string("vision"));
        CHECK_EQ(b.encoder.modalities, (uint32_t)RAD_MM_IMAGE);
        CHECK_EQ(b.encoder.patch_dim, 3 * 2 * 14 * 14);
        CHECK_EQ(b.encoder.merge, 4);
        CHECK_EQ(b.encoder.max_patches, 1024);
        CHECK_EQ(b.encoder.n_embd, 4096);
        const RecBuf* out = find_b(b, "v.out");
        REQUIRE(out != nullptr);
        CHECK_EQ(out->d.shape[0], 256);
        CHECK_EQ(out->d.shape[1], 4096);
        CHECK_EQ(count_op(b, "scatter_rows"), 2);
        CHECK(m.mrows.op < m.embed_ar || world == 1);
        CHECK(m.mtp.mrows.op > m.mtp.op_embed);
        /* Declared after every text and draft op. */
        CHECK(m.vit.op_patch > m.mtp.op_head);

        /* 2 stem + 12 a block + 8 downsample and merger, each from model.visual.*. */
        int nv = 0;
        for (const auto& w : b.weights)
            if (w.name.rfind("v.", 0) == 0) {
                ++nv;
                const RecMap* mp = find_map(b, w.name);
                REQUIRE(mp != nullptr);
                for (const auto& s : mp->src) CHECK(s.rfind("model.visual.", 0) == 0);
                CHECK_EQ(w.d.shard, (int)RAD_SHARD_NONE);
                CHECK(w.d.group.layer == -1 || w.d.group.layer >= 46);
            }
        CHECK_EQ(nv, 2 + 12 * 2 + 8);
        const RecWeight* blk1 = find_w(b, "v.blk.1.qkv.weight");
        REQUIRE(blk1 != nullptr);
        CHECK_EQ(blk1->d.group.layer, 47);
        const RecMap* gu = find_map(b, "v.blk.1.gate_up.weight");
        REQUIRE(gu != nullptr);
        CHECK_EQ(gu->mode, (int)RAD_MAP_CONCAT);
        REQUIRE_EQ(gu->src.size(), (size_t)2);
        CHECK_EQ(gu->src[0], std::string("model.visual.blocks.1.mlp.gate_proj.weight"));
        CHECK_EQ(gu->src[1], std::string("model.visual.blocks.1.mlp.up_proj.weight"));
        const RecWeight* ds = find_w(b, "v.downsample.weight");
        REQUIRE(ds != nullptr);
        CHECK_EQ(ds->d.shape[0], 4096);
        CHECK_EQ(ds->d.shape[1], 4096);
        const RecWeight* pe = find_w(b, "v.patch.weight");
        REQUIRE(pe != nullptr);
        CHECK_EQ(pe->d.shape[1], 1176);

        int qk = 0;
        for (const auto& o : b.ops)
            if (o.op == "qk_norm_rope") {
                ++qk;
                CHECK_EQ(pstr(&o, "mode"), std::string("axial"));
                CHECK_EQ(pstr(&o, "sections"), std::string("16 16"));
                CHECK_EQ(pint(&o, "head_dim"), 64);
                CHECK_EQ(pint(&o, "rotary_dim"), 64);
                CHECK_EQ(pint(&o, "n_head"), 16);
                CHECK_EQ(pflt(&o, "theta"), 10000.0);
                CHECK_EQ(o.w.size(), (size_t)2);
            }
        CHECK_EQ(qk, 2);
        const RecOp* unf = find_op(b, "conv2d_unfold");
        REQUIRE(unf != nullptr);
        CHECK_EQ(pint(unf, "width"), 2);
        CHECK_EQ(pint(unf, "cin"), 1024);
        CHECK_EQ(pint(unf, "kh"), 2);
        CHECK_EQ(pint(unf, "stride"), 2);
        CHECK_EQ(pint(unf, "cu_div"), 2);
        int clipped_tower = 0, clipped_merger = 0;
        for (const auto& o : b.ops)
            if (o.op == "swiglu_clamped") {
                clipped_tower += pint(&o, "n") == 4096;
                clipped_merger += pint(&o, "n") == 10240;
                CHECK_EQ(pflt(&o, "limit"), 10.0);
            }
        CHECK_EQ(clipped_tower, 2);
        CHECK_EQ(clipped_merger, 1);
        CHECK_EQ(count_op(b, "gelu"), 1);
        int merger_ln = 0;
        for (const auto& o : b.ops) merger_ln += o.op == "layernorm" && pint(&o, "n") == 4096;
        CHECK_EQ(merger_ln, 1);
    }
    /* Without speculation the draft head and its scatter are not declared. */
    RadBuilder nb;
    glm53_encs(nb);
    RadBuildCtx nc = glm53_vl_ctx(0, 1, 1024, 0);
    REQUIRE_EQ(glm5next::declare(&nb, mk.meta(), &nc), RAD_OK);
    CHECK_EQ(count_op(nb, "scatter_rows"), 1);
    RadArchProbe p{};
    CHECK_OK(glm5next::probe(mk.meta(), &p));
    CHECK_EQ(p.enc_merge, 4);
}

/* What the tower does not serve is refused by name at declare, not computed as another model. */
TEST(glm53_refuses_a_vision_tower_it_does_not_serve) {
    const struct { const char* key; const char* val; int rc; } kCases[] = {
        { "vision_config.hidden_act", "gelu", RAD_E_UNSUPPORTED },
        { "vision_config.rope_parameters.rope_type", "default", RAD_E_UNSUPPORTED },
        { "vision_config.out_hidden_size", "2048", RAD_E_INVAL },
    };
    for (const auto& kc : kCases) {
        MetaKV mk = glm53_vl_meta();
        mk.set(kc.key, kc.val);
        RadBuilder b;
        glm53_encs(b);
        RadBuildCtx ctx = glm53_vl_ctx(0, 1, 1024);
        const int rc = stderr_of_rc([&] { return glm5next::declare(&b, mk.meta(), &ctx); });
        if (rc != kc.rc) fprintf(stderr, "    %s = %s: %d\n", kc.key, kc.val, rc);
        CHECK_EQ(rc, kc.rc);
    }
}

/* An encoder pass issues the tower and nothing else: the stem, two blocks, the post-norm, the
 * downsample's unfold and GEMM, and the merger, over two pictures of 8 x 8 patches. The rotary
 * reads the row and column planes, attention and the unfold the segment bounds. */
TEST(a_glm53_encoder_pass_issues_the_tower_and_nothing_else) {
    MetaKV mk = glm53_vl_meta();
    for (int world : { 1, 8 }) {
        const int rank = world - 1;
        RadBuilder b;
        glm53_encs(b);
        RadBuildCtx ctx = glm53_vl_ctx(rank, world, 1024);
        REQUIRE_EQ(glm5next::declare(&b, mk.meta(), &ctx), RAD_OK);
        static uint16_t pix[128 * 1176];
        static int32_t coord[4 * 128];
        static const int32_t cu[] = { 0, 64, 128 };
        RadBatch batch{};
        batch.enc         = 1;
        batch.enc_n_patch = 128;
        batch.enc_pixels  = pix;
        batch.enc_coord   = coord;
        batch.enc_n_seg   = 2;
        batch.enc_cu      = cu;
        batch.enc_max_seg = 64;
        RadCtx c{ .b = &b, .batch = &batch };
        c.rank = rank;
        c.world = world;
        glm5next::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());

        std::vector<std::string> want = { "gemm_nt_bias" };
        for (int l = 0; l < 2; ++l)
            for (const char* s : { "rmsnorm", "gemm_nt_bias", "qk_norm_rope", "attn_dense",
                                   "gemm_nt_bias", "rmsnorm", "gemm_nt_bias", "swiglu_clamped",
                                   "gemm_nt_bias" })
                want.push_back(s);
        for (const char* s : { "rmsnorm", "conv2d_unfold", "gemm_nt_bias", "gemm_nt", "layernorm",
                               "gelu", "gemm_nt", "swiglu_clamped", "gemm_nt" })
            want.push_back(s);
        CHECK_EQ(join(issued_ops(b, c)), join(want));
        CHECK_EQ(c.issues.back().n, 32);
        const RecIssue* qk = issue_of(b, c, "qk_norm_rope");
        REQUIRE(qk != nullptr);
        CHECK(qk->opd[1].raw == coord);
        CHECK_EQ(qk->opd[1].rows, 2);
        CHECK_EQ(qk->opd[1].cols, 128);
        const RecIssue* at = issue_of(b, c, "attn_dense");
        REQUIRE(at != nullptr);
        CHECK_EQ(at->opd[0].cols, 1024);
        CHECK_EQ(at->opd[1].offset, 1024);
        CHECK_EQ(at->opd[2].offset, 2048);
        CHECK(at->opd[3].raw == cu);
        /* The unfold reads the post-norm as 64 rows of two patches and writes 32 merged rows. */
        const RecIssue* un = issue_of(b, c, "conv2d_unfold");
        REQUIRE(un != nullptr);
        CHECK_EQ(un->n, 64);
        CHECK_EQ(un->opd[0].rows, 64);
        CHECK(un->opd[1].raw == cu);
        CHECK_EQ(un->opd[3].rows, 32);

        RadBatch big = batch;
        big.enc_n_patch = 2048;
        RadCtx c2{ .b = &b, .batch = &big };
        glm5next::step(&c2, &big);
        CHECK(!c2.step_fail.empty());
        CHECK(c2.issues.empty());
    }
}

/* A prefill chunk whose tokens 1..4 are a picture's rows: the trunk writes them over the lookup
 * before the embedding's all-reduce, and so does the draft head's history pass over its own
 * lookup; a text-only step on the same program issues no scatter. */
TEST(a_glm53_step_with_a_picture_scatters_its_rows_in_the_trunk_and_the_draft_head) {
    MetaKV mk = glm53_vl_meta();
    for (int world : { 1, 4 }) {
        RadBuilder b;
        glm53_encs(b);
        RadBuildCtx ctx = glm53_vl_ctx(0, world, 1024);
        REQUIRE_EQ(glm5next::declare(&b, mk.meta(), &ctx), RAD_OK);
        static const int32_t rows[] = { 1, 2, 3, 4 };
        static uint16_t embd[4 * 4096];
        auto pf = kit(b, {}, { 6 });
        pf->b.n_mm_rows = 4;
        pf->b.mm_rows = rows;
        pf->b.mm_embd = embd;
        auto his = history_pass(b, *pf);
        his->b.n_mm_rows = 4;
        his->b.mm_rows = rows;
        his->b.mm_embd = embd;
        auto v = run_passes(b, glm5next::step, { pf.get(), his.get() }, 0, world);
        CHECK_PASSES(b, v);
        for (int p = 0; p < 2; ++p) {
            const std::vector<std::string> ops = issued_ops(b, *v[(size_t)p]);
            REQUIRE(ops.size() > 3);
            const size_t at = (size_t)(std::find(ops.begin(), ops.end(), "embed_lookup") -
                                       ops.begin());
            REQUIRE(at + 1 < ops.size());
            CHECK_EQ(ops[at + 1], std::string("scatter_rows"));
            if (world > 1) CHECK_EQ(ops[at + 2], std::string("all_reduce"));
            const RecIssue* sc = issue_of(b, *v[(size_t)p], "scatter_rows");
            REQUIRE(sc != nullptr);
            CHECK_EQ(sc->n, 4);
            CHECK(sc->opd[0].raw == embd);
            CHECK_EQ(sc->opd[0].cols, 4096);
            CHECK(sc->opd[1].raw == rows);
            CHECK_EQ(sc->opd[2].rows, 6);
            CHECK_EQ(count_issued(ops, "scatter_rows"), 1);
        }
        auto dc = kit(b, { 1, 1 }, {});
        auto t = run_passes(b, glm5next::step, { dc.get() }, 0, world);
        CHECK_PASSES(b, t);
        CHECK(issue_of(b, *t[0], "scatter_rows") == nullptr);
    }
}

/* A sizing declare with the tower is the real graph at fewer rows: the same ops with the same
 * fixed parameters (the attention's longest segment included), and nothing declared to the core. */
TEST(glm53_sizing_declares_with_the_vision_tower_match_the_real_graph) {
    MetaKV mk = glm53_vl_meta();
    for (int64_t R : { 256 }) {
        RadBuildCtx c = glm53_vl_ctx(3, 8, 16384);
        c.max_tok = 4096;
        c.max_seqs = 8;
        c.max_ctx = 32768;
        RadBuilder full;
        glm53_encs(full);
        REQUIRE_EQ(glm5next::declare(&full, mk.meta(), &c), RAD_OK);
        CHECK_EQ(full.encoder_name, std::string("vision"));
        RadBuildCtx p = c;
        p.max_tok = R;
        p.shape_probe = 1;
        RadBuilder small;
        glm53_encs(small);
        REQUIRE_EQ(glm5next::declare(&small, mk.meta(), &p), RAD_OK);
        CHECK(small.encoder_name.empty());
        REQUIRE_EQ(small.ops.size(), full.ops.size());
        int differ = 0;
        for (size_t i = 0; i < full.ops.size(); ++i) {
            if (small.ops[i].op != full.ops[i].op) { ++differ; continue; }
            REQUIRE_EQ(small.ops[i].p.size(), full.ops[i].p.size());
            for (size_t k = 0; k < full.ops[i].p.size(); ++k) {
                const RecParam& a = full.ops[i].p[k];
                const RecParam& q = small.ops[i].p[k];
                if (a.kind == RAD_P_RANGE) continue;
                if (full.ops[i].op == "qsa_work" && a.key == "work" && q.ival <= a.ival) continue;
                if (a.ival != q.ival || a.sval != q.sval || a.dval != q.dval) {
                    fprintf(stderr, "    R %lld: op %zu '%s' param '%s': %lld real, %lld small\n",
                            (long long)R, i, full.ops[i].op.c_str(), a.key.c_str(), a.ival,
                            q.ival);
                    ++differ;
                }
            }
        }
        CHECK_EQ(differ, 0);
        REQUIRE_EQ(small.bufs.size(), full.bufs.size());
        for (size_t i = 0; i < full.bufs.size(); ++i) {
            for (uint32_t d = 1; d < full.bufs[i].d.rank; ++d)
                CHECK_EQ(small.bufs[i].d.shape[d], full.bufs[i].d.shape[d]);
            CHECK(small.bufs[i].d.shape[0] <= full.bufs[i].d.shape[0]);
        }
    }
}

TEST(glm53_declaring_without_streamed_ops_ignores_the_flag) {
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    RadBuilder b;
    b.kv_block = 4;
    b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 }, { "mtp.blk.45.ffn_down_exps", fp8 },
               { "ffn_gate_up_exps", mx }, { "ffn_down_exps", mx },
               { "ffn_gate_up_shexp", mx }, { "ffn_down_shexp", mx },
               { "ffn_gate_up.weight", fp8 }, { "ffn_down.weight", fp8 } };
    b.undeclared = { "kv_resolve", "kv_store_streamed" };
    RadBuildCtx build{};
    build.rank = 0;
    build.world_size = 1;
    build.max_tok = 512;
    build.max_seqs = 2;
    build.max_ctx = 4096;
    build.max_spec = 0;
    build.kv_resident_blocks = 32768;   /* the ring floor: 8 192 blocks. This fixture toggles the ops,
                                         * not the ring; the undeclared list is the gate it tests. */
    REQUIRE_EQ(glm5next::declare(&b, &meta, &build), RAD_OK);
    CHECK_EQ(count_op(b, "kv_resolve"), 0);
    CHECK_EQ(count_op(b, "kv_store_streamed"), 0);
    /* The base store is still there: the zeros are the gate, not a layer that never stores. */
    CHECK(count_op(b, "kv_store") > 0);
}

TEST(glm53_streamed_decode_resolves_after_select_and_before_attention_with_the_entry_count) {
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    RadBuilder b;
    b.kv_block = 4;
    b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 }, { "mtp.blk.45.ffn_down_exps", fp8 },
               { "ffn_gate_up_exps", mx }, { "ffn_down_exps", mx },
               { "ffn_gate_up_shexp", mx }, { "ffn_down_shexp", mx },
               { "ffn_gate_up.weight", fp8 }, { "ffn_down.weight", fp8 } };
    RadBuildCtx build{};
    build.rank = 0;
    build.world_size = 1;
    build.max_tok = 512;
    build.max_seqs = 2;
    build.max_ctx = 4096;
    build.max_spec = 3;                 /* so the drafter's layer 45 is the twelfth full layer */
    build.kv_resident_blocks = 32768;   /* the ring floor, 8 192 blocks: above this fixture's spec-3
                                         * raise of 2 x 4 x 513 = 4 104 blocks, so the engine raises
                                         * nothing. The declared slots are the context's 2048 blocks
                                         * either way, which is what the fake stream mirrors. */
    REQUIRE_EQ(glm5next::declare(&b, &meta, &build), RAD_OK);
    const auto& m = *glm5next::models[0];
    REQUIRE_EQ(m.latent, (rad_kvgroup)1);
    REQUIRE_EQ(count_op(b, "kv_resolve"), 12);

    /* A zeroed stream with the figures bring-up fixes: 4096 B a block (12 layers of 2 x 4 x
     * 256 x bf16), a 2048-block mirror, and the ring the declaration states -- the context's
     * 2048 blocks, because a 32 768-token ring is clamped to `blocks` for this 4096-token
     * context. Only the latent group streams. */
    const int64_t blocks = 2048, slots = 2048;
    std::vector<uint8_t> host((size_t)blocks * 4096);
    std::vector<int32_t> res(blocks), slot_block(slots), slot_state(slots * 2), ctl(16);
    RadKVStream st{};
    st.host = host.data();
    st.res = res.data();
    st.slot_block = slot_block.data();
    st.slot_state = slot_state.data();
    st.ctl = ctl.data();
    st.block_bytes = 4096;
    st.host_layer_stride = 4096 * blocks;
    st.slots = slots;
    st.host_bytes = (int64_t)host.size() * 12;

    int32_t cu[] = { 0, 1, 2 }, positions[] = { 0, 1 }, ctxlens[] = { 1, 1 },
            ids[] = { 7, 9 }, slots_map[] = { 0, 1 }, out[] = { 1 };
    int32_t sidx[] = { 0, 0, 0, 1, 1, 1 }, ci[] = { 0 }, tables[64]{};
    std::vector<RadKVGroupBatch> kv(b.kv_groups.size());
    for (size_t i = 0; i < kv.size(); ++i) {
        kv[i].group = (rad_kvgroup)i + 1;
        kv[i].slot_mapping = slots_map;
        kv[i].block_table = tables;
        kv[i].block_table_pitch = 32;
        kv[i].state_index = b.kv_groups[i].d.kind == RAD_KV_CONV ? ci : sidx;
        kv[i].state_index_pitch = b.kv_groups[i].d.kind == RAD_KV_CONV ? 1 : 6;
    }
    kv[m.latent - 1].stream = &st;

    RadBatch batch{};
    batch.phase = RAD_PHASE_DECODE;
    batch.n_tok = 2;
    batch.n_seq = 2;
    batch.token_ids = ids;
    batch.positions = positions;
    batch.cu_seqlens = cu;
    batch.ctx_lens = ctxlens;
    batch.kv = kv.data();
    batch.n_kv_groups = (int)kv.size();
    batch.n_out = 1;
    batch.out_ids = out;
    batch.max_ctx_len = 1;
    batch.max_q_len = 1;
    RadCtx ctx;
    ctx.b = &b;
    ctx.rank = 0;
    ctx.world = 1;
    ctx.batch = &batch;

    auto resolve_count = [&] {
        int n = 0;
        for (const auto& issue : ctx.issues)
            if (issue.op >= 1 && (size_t)issue.op <= b.ops.size() &&
                b.ops[issue.op - 1].op == "kv_resolve")
                ++n;
        return n;
    };
    auto find_issue = [&](rad_op op) -> size_t {
        for (size_t i = 0; i < ctx.issues.size(); ++i)
            if (ctx.issues[i].op == op) return i;
        return SIZE_MAX;
    };
    auto check_layer_order = [&](const glm5next::Layer& l) {
        const size_t store = find_issue(l.mla.op_store_stream);
        // A short history takes every block (DsaCache::takes_all), a long one scores.
        const size_t sel   = std::min(find_issue(l.index.cache.op_select),
                                      find_issue(l.index.cache.op_select_all));
        const size_t res_i = find_issue(l.mla.op_resolve);
        const size_t attn  = find_issue(l.mla.core.op_attn);
        REQUIRE(store != SIZE_MAX);
        REQUIRE(sel != SIZE_MAX);
        REQUIRE(res_i != SIZE_MAX);
        REQUIRE(attn != SIZE_MAX);
        CHECK(store < sel);
        CHECK(sel < res_i);
        CHECK(res_i < attn);
        /* §2: the count is `selected_count`, never the token count. */
        CHECK_EQ(ctx.issues[res_i].opd[1].handle, m.aw.core.nsel);
        CHECK(ctx.issues[res_i].opd[1].handle != m.aw.core.length);
        CHECK_EQ(ctx.issues[res_i].n, (int64_t)2);
    };

    /* A decode pass: 11 trunk full layers resolve, and the draft layer does too. */
    glm5next::step(&ctx, &batch);
    CHECK(ctx.step_fail.empty());
    CHECK_EQ(resolve_count(), 11);
    for (const auto& l : m.layers)
        if (l.full) check_layer_order(l);

    ctx.issues.clear();
    batch.draft_pass = 2;
    glm5next::step(&ctx, &batch);
    CHECK(ctx.step_fail.empty());
    CHECK_EQ(resolve_count(), 1);
    check_layer_order(m.draft);

    /* A staged pass (prefill, or the stage view chosen for it) must not resolve: the selection
     * keeps the block ids the identity stage addresses directly. */
    ctx.issues.clear();
    ctx.stages.clear();
    ctx.stage_on = true;
    batch.draft_pass = 0;
    batch.phase = RAD_PHASE_PREFILL;
    batch.n_tok = 4;
    int32_t cu4[] = { 0, 2, 4 };
    batch.cu_seqlens = cu4;
    batch.max_q_len = 2;
    glm5next::step(&ctx, &batch);
    CHECK(ctx.step_fail.empty());
    CHECK_EQ(resolve_count(), 0);
    CHECK(!ctx.stages.empty());
}

RAD_TEST_MAIN()

/* ==================================================================== the row rule */

/* GlmTp::run_rows is the one rule the MLA's and the indexer's runs follow: ceil(T / world) for
 * the MLA's 8192-byte rows, 4 * ceil(T / (4 * world)) for the indexer's 4-byte lengths. At the
 * split threshold, around a chunk and at the largest step, for every world: run_of covers the
 * step once with no overlap, the last run short where the step ends inside it, and a rank whose
 * run lies wholly past a small step takes the first run instead, so the planes it writes and
 * gathers are never unwritten. */
TEST(glm_tp_row_rule_rounds_to_whole_words_and_covers_a_step_once) {
    using namespace rad::arch;
    for (int world : { 1, 2, 4, 8 })
        for (int64_t T : { 1, 5, 6, 48, 255, 256, 257, 300, 2047, 2048, 2049, 4096 }) {
            const int64_t wide = GlmTp::run_rows(T, world, 8192);
            const int64_t narrow = GlmTp::run_rows(T, world, 4);
            CHECK_EQ(wide, (T + world - 1) / world);
            CHECK_EQ(narrow, 4 * ((T + 4 * world - 1) / (4 * world)));
            CHECK_EQ(GlmMla::rows_for(T, world), wide);
            CHECK_EQ(DsaCache::part_rows(T, world), narrow);
            CHECK(world * wide >= T);
            CHECK(world * narrow >= T);
            CHECK_EQ(narrow % 4, 0);
            for (int64_t row_bytes : { 8192, 4 }) {
                const int64_t run = GlmTp::run_rows(T, world, row_bytes);
                int64_t covered = 0, next = 0;
                for (int rank = 0; rank < world; ++rank) {
                    const GlmTp::Run r = GlmTp::run_of(T, world, rank, row_bytes);
                    CHECK_EQ(r.run, run);
                    CHECK(r.rows >= 1 && r.rows <= run);
                    if ((int64_t)rank * run < T) {
                        CHECK_EQ(r.first, next);
                        CHECK_EQ(r.rows, std::min(run, T - r.first));
                        next = r.first + r.rows;
                        covered += r.rows;
                    } else {
                        CHECK_EQ(r.first, 0);
                        CHECK_EQ(r.rows, run);
                    }
                }
                CHECK_EQ(covered, T);
            }
        }
}

/* ==================================================================== the op schemas */
namespace {

/* The op schemas the kernel libraries under $RADIANCE_HOME/kernels declare: libr4d's first, the
 * served vocabulary, then libref's for what it lacks. Empty without a home or a library, as
 * arch_test's libref vocabulary is. */
struct OpSpec {
    int n_operands = 0, w_req = 0, w_max = 0, n_tab = 0;
};
struct LoadedSchemas {
    std::map<std::string, OpSpec> ops;
    bool r4d = false;
};
const LoadedSchemas& loaded_schemas() {
    static LoadedSchemas s;
    static bool tried = false;
    if (tried) return s;
    tried = true;
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home) return s;
    for (const char* lib : { "libr4d.so", "libref.so" }) {
        const std::string path = std::string(home) + "/kernels/" + lib;
        void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            fprintf(stderr, "    cannot load %s: %s\n", path.c_str(), dlerror());
            continue;
        }
        auto count = (int (*)(void))dlsym(h, "rad_kernel_schema_count");
        auto at = (const RadOpSchema* (*)(int))dlsym(h, "rad_kernel_schema_at");
        if (!count || !at) continue;
        for (int i = 0; i < count(); ++i) {
            const RadOpSchema* sc = at(i);
            if (!sc || !sc->op || s.ops.count(sc->op)) continue;
            OpSpec spec;
            spec.n_operands = sc->n_operands;
            for (int k = 0; sc->operands && k < sc->n_operands; ++k) {
                if (sc->operands[k].role == RAD_OPD_WTAB) {
                    ++spec.n_tab;
                } else if (sc->operands[k].role == RAD_OPD_WEIGHT) {
                    ++spec.w_max;
                    if (!sc->operands[k].optional) ++spec.w_req;
                }
            }
            s.ops[sc->op] = spec;
        }
        if (!std::strcmp(lib, "libr4d.so")) s.r4d = true;
    }
    return s;
}

}  /* namespace */

/* Every op the plugin issues carries as many operands as its schema declares, counted as
 * core/runtime/issue.cpp counts them (an absent optional operand is one of them), and every op it
 * declares names as many weights as rad_builder.cpp admits. The steps: at eight ranks and at
 * one, a mixed step over the split threshold (two decode rows beside a 498-row chunk), a step
 * under it, a six-row verify and six single-row decodes, each with its history pass and a draft
 * round, and a verify of eight sequences at 32k of context, where the indexer splits a decode.
 * The engine refuses a step at the first mismatch, which the recording fakes do not check, so a
 * missing optional operand reaches no other host test. */
TEST(glm53_issues_every_op_with_the_operand_count_its_schema_declares) {
    const LoadedSchemas& lib = loaded_schemas();
    if (lib.ops.empty()) {
        fprintf(stderr, "    no kernel library under $RADIANCE_HOME/kernels; nothing to count\n");
        return;
    }
    MetaKV mk = glm53_vl_meta();
    std::set<std::string> uncovered, wrong;
    for (int world : { 8, 1 }) {
        const int rank = world == 8 ? 3 : 0;
        RadBuilder b;
        glm53_encs(b);
        RadBuildCtx ctx = glm53_vl_ctx(rank, world, 1024, 5);
        ctx.max_tok = 4096;
        ctx.max_seqs = 8;
        ctx.max_ctx = 65536;
        REQUIRE_EQ(glm5next::declare(&b, mk.meta(), &ctx), RAD_OK);
        for (const RecOp& r : b.ops) {
            auto it = lib.ops.find(r.op);
            if (it == lib.ops.end()) {
                uncovered.insert(r.op);
                continue;
            }
            const OpSpec& s = it->second;
            const int n_w = (int)r.w.size();
            const bool ok = s.n_tab > 0 ? n_w - s.w_max >= s.n_tab
                                        : n_w >= s.w_req && n_w <= s.w_max;
            if (!ok && wrong.insert(r.op + " (declare)").second)
                fprintf(stderr,
                        "    world %d: '%s' declares %d weights; the schema has %d (%d required) "
                        "and %d tables\n",
                        world, r.op.c_str(), n_w, s.w_max, s.w_req, s.n_tab);
        }
        std::vector<std::unique_ptr<Kit>> kits;
        auto with_drafts = [&](std::unique_ptr<Kit> k) {
            auto his = history_pass(b, *k);
            auto round = round_pass(b, k->b.n_seq, 2);
            kits.push_back(std::move(k));
            kits.push_back(std::move(his));
            kits.push_back(std::move(round));
        };
        with_drafts(kit(b, { 1, 1 }, { 498 }, 100));
        with_drafts(kit(b, {}, { 255 }));
        with_drafts(kit(b, { 6 }, {}));
        with_drafts(kit(b, std::vector<int>(6, 1), {}));
        {
            auto dec = kit(b, std::vector<int>(8, 6), {});
            const int64_t pitch = 8192;
            dec->table.resize((size_t)(8 * pitch));
            for (int64_t i = 0; i < 8 * pitch; ++i) dec->table[(size_t)i] = (int32_t)i;
            for (auto& g : dec->kv) {
                g.block_table = dec->table.data();
                g.block_table_pitch = g.max_blocks = pitch;
            }
            dec->b.max_ctx_len = 32000;
            kits.push_back(std::move(dec));
        }
        std::vector<const Kit*> passes;
        for (const auto& k : kits) passes.push_back(k.get());
        auto v = run_passes(b, glm5next::step, passes, rank, world);
        CHECK_PASSES(b, v);
        for (size_t p = 0; p < v.size(); ++p)
            for (const RecIssue& issue : v[p]->issues) {
                const RecOp* op = b.op_of(issue.op);
                REQUIRE(op);
                auto it = lib.ops.find(op->op);
                if (it == lib.ops.end()) {
                    uncovered.insert(op->op);
                    continue;
                }
                if ((int)issue.opd.size() != it->second.n_operands && wrong.insert(op->op).second)
                    fprintf(stderr,
                            "    world %d pass %zu: '%s' issued with %zu operands; the schema "
                            "declares %d\n",
                            world, p, op->op.c_str(), issue.opd.size(), it->second.n_operands);
            }
    }
    CHECK(wrong.empty());
    for (const auto& o : uncovered) fprintf(stderr, "    no schema loaded for '%s'\n", o.c_str());
    if (lib.r4d) CHECK(uncovered.empty());
}

/* ==================================================================== the container's names */
namespace {

/* The served container, GLM-5.3-Flash-Quark-MXFP4-vision.rad, as `rad-info -v` lists it: each
 * weight's name and stored encoding, for the model, one KDA layer (0, with the dense feed-forward
 * of layers 0 to 2), one MLA layer (3, with its experts 0 and 287), the MTP layer under its
 * `mtp.` scope (the fp8 linears of its layer under `mtp.mtp.`, as the converter wrote them) and
 * the tower's first block. The other layers of a kind carry the same names at their own index. */
struct ContainerWeight {
    const char* name;
    const char* enc;
};
const ContainerWeight kGlm53Container[] = {
    { "output.weight", "bf16" },
    { "output_norm.weight", "bf16" },
    { "token_embd.weight", "bf16" },
    { "v.downsample.bias", "bf16" },
    { "v.downsample.weight", "bf16" },
    { "v.merger.down.weight", "bf16" },
    { "v.merger.gate_up.weight", "bf16" },
    { "v.merger.norm.bias", "bf16" },
    { "v.merger.norm.weight", "bf16" },
    { "v.merger.proj.weight", "bf16" },
    { "v.patch.bias", "bf16" },
    { "v.patch.weight", "bf16" },
    { "v.post_norm.weight", "bf16" },
    { "blk.0.attn_beta.weight", "bf16" },
    { "blk.0.attn_forget_a.weight", "bf16" },
    { "blk.0.attn_forget_b.weight", "bf16" },
    { "blk.0.attn_gate_a.weight", "bf16" },
    { "blk.0.attn_gate_b.weight", "bf16" },
    { "blk.0.attn_norm.weight", "bf16" },
    { "blk.0.attn_output.weight", "bf16" },
    { "blk.0.attn_qkv.weight", "bf16" },
    { "blk.0.ffn_down.weight", "fp8_e4m3*f32[128x128]" },
    { "blk.0.ffn_gate_up.weight", "fp8_e4m3*f32[128x128]" },
    { "blk.0.ffn_norm.weight", "bf16" },
    { "blk.0.hc_attn.base", "f32" },
    { "blk.0.hc_attn.fn", "bf16" },
    { "blk.0.hc_attn.scale", "f32" },
    { "blk.0.hc_ffn.base", "f32" },
    { "blk.0.hc_ffn.fn", "bf16" },
    { "blk.0.hc_ffn.scale", "f32" },
    { "blk.0.ssm_a", "f32" },
    { "blk.0.ssm_conv.weight", "bf16" },
    { "blk.0.ssm_dt_bias", "f32" },
    { "blk.0.ssm_norm.weight", "bf16" },
    { "blk.3.attn_kv_a.weight", "fp8_e4m3*f32[128x128]" },
    { "blk.3.attn_kv_a_norm.weight", "bf16" },
    { "blk.3.attn_kv_b.weight", "bf16" },
    { "blk.3.attn_norm.weight", "bf16" },
    { "blk.3.attn_output.weight", "fp8_e4m3*f32[128x128]" },
    { "blk.3.attn_q_a.weight", "fp8_e4m3*f32[128x128]" },
    { "blk.3.attn_q_a_norm.weight", "bf16" },
    { "blk.3.attn_q_b.weight", "fp8_e4m3*f32[128x128]" },
    { "blk.3.ffn_down_shexp.weight", "fp4_e2m1*e8m0[1x32]" },
    { "blk.3.ffn_gate_up_shexp.weight", "fp4_e2m1*e8m0[1x32]" },
    { "blk.3.ffn_norm.weight", "bf16" },
    { "blk.3.glm_router.correction", "f32" },
    { "blk.3.glm_router.weight", "bf16" },
    { "blk.3.hc_attn.base", "f32" },
    { "blk.3.hc_attn.fn", "bf16" },
    { "blk.3.hc_attn.scale", "f32" },
    { "blk.3.hc_ffn.base", "f32" },
    { "blk.3.hc_ffn.fn", "bf16" },
    { "blk.3.hc_ffn.scale", "f32" },
    { "blk.3.index_ape", "bf16" },
    { "blk.3.index_head.weight", "bf16" },
    { "blk.3.index_key_gate.weight", "bf16" },
    { "blk.3.index_norm.bias", "bf16" },
    { "blk.3.index_norm.weight", "bf16" },
    { "blk.3.index_query.weight", "bf16" },
    { "blk.3.ffn_gate_up_exps.0.weight", "fp4_e2m1*e8m0[1x32]" },
    { "blk.3.ffn_down_exps.0.weight", "fp4_e2m1*e8m0[1x32]" },
    { "blk.3.ffn_gate_up_exps.287.weight", "fp4_e2m1*e8m0[1x32]" },
    { "blk.3.ffn_down_exps.287.weight", "fp4_e2m1*e8m0[1x32]" },
    { "mtp.blk.45.attn_kv_a_norm.weight", "bf16" },
    { "mtp.blk.45.attn_kv_b.weight", "bf16" },
    { "mtp.blk.45.attn_norm.weight", "bf16" },
    { "mtp.blk.45.attn_q_a_norm.weight", "bf16" },
    { "mtp.blk.45.ffn_norm.weight", "bf16" },
    { "mtp.blk.45.index_ape", "bf16" },
    { "mtp.blk.45.index_head.weight", "bf16" },
    { "mtp.blk.45.index_norm.bias", "bf16" },
    { "mtp.blk.45.index_norm.weight", "bf16" },
    { "mtp.eh_proj.weight", "bf16" },
    { "mtp.enorm.weight", "bf16" },
    { "mtp.hnorm.weight", "bf16" },
    { "mtp.mtp.blk.45.attn_kv_a.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.mtp.blk.45.attn_output.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.mtp.blk.45.attn_q_a.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.mtp.blk.45.attn_q_b.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.mtp.blk.45.ffn_down_shexp.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.mtp.blk.45.ffn_gate_up_shexp.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.mtp.blk.45.glm_router.correction", "f32" },
    { "mtp.mtp.blk.45.glm_router.weight", "bf16" },
    { "mtp.mtp.blk.45.index_key_gate.weight", "bf16" },
    { "mtp.mtp.blk.45.index_query.weight", "bf16" },
    { "mtp.output_norm.weight", "bf16" },
    { "mtp.blk.45.ffn_gate_up_exps.0.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.blk.45.ffn_down_exps.0.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.blk.45.ffn_gate_up_exps.287.weight", "fp8_e4m3*f32[128x128]" },
    { "mtp.blk.45.ffn_down_exps.287.weight", "fp8_e4m3*f32[128x128]" },
    { "v.blk.0.down.bias", "bf16" },
    { "v.blk.0.down.weight", "bf16" },
    { "v.blk.0.gate_up.bias", "bf16" },
    { "v.blk.0.gate_up.weight", "bf16" },
    { "v.blk.0.k_norm.weight", "bf16" },
    { "v.blk.0.norm1.weight", "bf16" },
    { "v.blk.0.norm2.weight", "bf16" },
    { "v.blk.0.proj.bias", "bf16" },
    { "v.blk.0.proj.weight", "bf16" },
    { "v.blk.0.q_norm.weight", "bf16" },
    { "v.blk.0.qkv.bias", "bf16" },
    { "v.blk.0.qkv.weight", "bf16" },
};

RadEncoding container_enc(const std::string& e) {
    if (e == "fp8_e4m3*f32[128x128]") return rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    if (e == "fp4_e2m1*e8m0[1x32]") return rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    if (e == "f32") return rad_enc_plain(RAD_F32);
    if (e != "bf16") fprintf(stderr, "    unknown container encoding '%s'\n", e.c_str());
    return rad_enc_plain(RAD_BF16);
}

/* The table above expanded to every layer: a KDA layer's names from layer 0 (its dense
 * feed-forward in layers 0 to 2 only), an MLA layer's from layer 3, the MoE's from layer 3 into
 * every layer from 3, the tower's 24 blocks from block 0. `encs` answers rad_weight_encoding as
 * the container would; an expert's key stands without its index, so every expert matches it,
 * and the keys run longest first, because the fake matches a key anywhere in the name asked
 * (`output.weight` is inside `blk.3.attn_output.weight`). */
struct Glm53Container {
    std::set<std::string> names;
    std::map<std::string, std::string> enc;
    std::vector<std::pair<std::string, RadEncoding>> encs;
};
const Glm53Container& glm53_container() {
    static Glm53Container c;
    static bool built = false;
    if (built) return c;
    built = true;
    std::set<std::string> keys;
    auto put = [&](const std::string& name, const char* e) {
        if (!c.names.insert(name).second) return;
        c.enc[name] = e;
        std::string key = name;
        const size_t ex = key.find("_exps.");
        if (ex != std::string::npos) key = key.substr(0, ex + 5);
        if (keys.insert(key).second) c.encs.push_back({ key, container_enc(e) });
    };
    auto at_layer = [](const std::string& n, size_t prefix, int l) {
        return "blk." + std::to_string(l) + "." + n.substr(prefix);
    };
    for (const ContainerWeight& w : kGlm53Container) {
        const std::string n = w.name;
        if (n.rfind("blk.0.", 0) == 0) {
            const bool dense = n.find(".ffn_") != std::string::npos;
            for (int l = 0; l < 45; ++l)
                if (l % 4 != 3 && (!dense || l < 3)) put(at_layer(n, 6, l), w.enc);
        } else if (n.rfind("blk.3.", 0) == 0) {
            const bool moe = n.find(".ffn_") != std::string::npos ||
                             n.find(".glm_router.") != std::string::npos;
            for (int l = 3; l < 45; ++l)
                if (moe || l % 4 == 3) put(at_layer(n, 6, l), w.enc);
        } else if (n.rfind("v.blk.0.", 0) == 0) {
            for (int l = 0; l < 24; ++l) put("v.blk." + std::to_string(l) + "." + n.substr(8), w.enc);
        } else {
            put(n, w.enc);
        }
    }
    std::stable_sort(c.encs.begin(), c.encs.end(), [](const auto& a, const auto& b) {
        return a.first.size() > b.first.size();
    });
    return c;
}

/* An expert's entry stands for its siblings: any index reads as expert 0's. */
std::string as_expert_zero(std::string n) {
    const size_t ex = n.find("_exps.");
    if (ex == std::string::npos) return n;
    const size_t dot = n.find('.', ex + 6);
    return dot == std::string::npos ? n : n.substr(0, ex + 6) + "0" + n.substr(dot);
}

}  /* namespace */

/* Every weight the plugin declares reads an entry the served container holds: under its own name,
 * or through `source` for a view (a scale plane, the row form's full-width projections), and
 * every GEMM names the dtype its weight's stored encoding calls for: `fp8a8f32` on the f32-scaled
 * block FP8, `mxfp4a8` on the MXFP4 experts, `bf16` on the plain linears. The encodings come
 * from the table, as the container would answer. A full-width view declared as its own weight
 * at `fp8a8` passed every host test and left 88 bands with no kernel at the eight-card start. */
TEST(glm53_declares_the_served_containers_names_and_their_encodings) {
    const Glm53Container& c = glm53_container();
    MetaKV mk = glm53_vl_meta();
    for (int world : { 8, 1 }) {
        RadBuilder b;
        b.kv_block = 4;
        b.encs = c.encs;
        RadBuildCtx ctx = glm53_vl_ctx(world == 8 ? 5 : 0, world, 1024);
        REQUIRE_EQ(glm5next::declare(&b, mk.meta(), &ctx), RAD_OK);
        std::map<std::string, const RecWeight*> declared;
        for (const RecWeight& w : b.weights) declared[w.name] = &w;
        // The stored entry a declared weight reads: itself, or its source, followed through views.
        auto stored = [&](const RecWeight& w) {
            std::string n = w.name;
            for (int hops = 0; hops < 4; ++hops) {
                auto it = declared.find(n);
                if (it == declared.end() || !it->second->d.source || n == it->second->d.source)
                    break;
                n = it->second->d.source;
            }
            return n;
        };
        int missing = 0, optional_absent = 0;
        for (const RecWeight& w : b.weights) {
            const std::string n = as_expert_zero(stored(w));
            if (c.names.count(n)) continue;
            if (w.d.optional) {
                ++optional_absent;
                continue;
            }
            if (++missing <= 24)
                fprintf(stderr, "    world %d: '%s' reads '%s', which the container does not hold\n",
                        world, w.name.c_str(), n.c_str());
        }
        CHECK_EQ(missing, 0);
        if (optional_absent)
            fprintf(stderr, "    world %d: %d optional weights the container does not hold\n", world,
                    optional_absent);
        int wrong = 0;
        for (const RecOp& r : b.ops) {
            if (r.w.empty() || !r.w[0]) continue;
            const bool quant = r.op == "gemm_nt_q" || r.op == "gemm_nt_q_gated" ||
                               r.op == "gemm_nt_q_bias" || r.op == "moe_gemm_q" ||
                               r.op == "moe_gemm_q_mod4";
            const bool plain = r.op == "gemm_nt" || r.op == "gemm_nt_bias" ||
                               r.op == "logits_gemm" || r.op == "embed_lookup";
            if (!quant && !plain) continue;
            const std::string n = as_expert_zero(stored(b.weights[r.w[0] - 1]));
            auto e = c.enc.find(n);
            if (e == c.enc.end()) continue;
            const char* want = e->second == "fp8_e4m3*f32[128x128]" ? "fp8a8f32"
                             : e->second == "fp4_e2m1*e8m0[1x32]"   ? "mxfp4a8"
                             : e->second == "bf16"                  ? "bf16"
                                                                    : nullptr;
            if (!want) continue;
            if (quant == (e->second == "bf16") && ++wrong <= 24) {
                fprintf(stderr, "    world %d: '%s' over '%s' (%s)\n", world, r.op.c_str(), n.c_str(),
                        e->second.c_str());
                continue;
            }
            if (pstr(&r, "dtype") != want && ++wrong <= 24)
                fprintf(stderr, "    world %d: '%s' over '%s' (%s) at dtype %s, not %s\n", world,
                        r.op.c_str(), n.c_str(), e->second.c_str(), pstr(&r, "dtype").c_str(), want);
        }
        CHECK_EQ(wrong, 0);
    }
}
