/* Spark's declaration and step, without a GPU or checkpoint. */
#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "rad_plugin.h"
#include "../arch/spark25_bf16/spark25_bf16.cpp"
#include "../arch/spark25_w8a8/spark25_w8a8.cpp"

using namespace rad_fake;

static MetaKV spark_meta() {
    RadModelMeta m{};
    m.arch_id = "spark25";
    m.n_layers = 4;
    m.n_embd = 2560;
    m.n_head = 16;
    m.n_head_kv = 4;
    m.head_dim = 256;
    m.n_ff = 10240;
    m.n_vocab = 131072;
    m.n_ctx_train = 1048576;
    m.rms_eps = 1e-6f;
    m.rope_theta = 10000;
    MetaKV k(m);
    k.set("tie_word_embeddings", "1").set("headwise_attn_output_gate", "1")
     .set("gate_attn_act_mode", "sigmoid").set("hidden_act", "gelu")
     .set("layer_types", "sliding_attention sliding_attention sliding_attention full_attention")
     .set("sliding_window", "512")
     .set("rope_parameters.sliding_attention.rope_theta", "10000")
     .set("rope_parameters.sliding_attention.partial_rotary_factor", "1")
     .set("rope_parameters.full_attention.rope_theta", "5000000")
     .set("rope_parameters.full_attention.partial_rotary_factor", "0.25");
    return k;
}

TEST(layer_schedule_rope_and_window_follow_the_checkpoint) {
    MetaKV k = spark_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(spark25_bf16::declare(&b, k.meta(), &ctx), RAD_OK);
    CHECK(spark25_bf16::chat_format(k.meta()) != nullptr);
    REQUIRE_EQ(count_op(b, "attn_paged"), 4);
    REQUIRE_EQ(count_op(b, "rope"), 8);
    REQUIRE_EQ(b.binds.size(), (size_t)4);
    for (int l = 0; l < 4; ++l) {
        const RecOp* a = find_op(b, "attn_paged", l);
        const RecOp* q = find_op(b, "rope", 2*l);
        const RecOp* key = find_op(b, "rope", 2*l+1);
        REQUIRE(a && q && key);
        CHECK_EQ(pint(a, "window"), l == 3 ? 0 : 512);
        CHECK_EQ(pint(a, "gqa"), 4);
        CHECK_EQ(pint(q, "rotary_dim"), l == 3 ? 64 : 256);
        CHECK_EQ(pint(key, "rotary_dim"), l == 3 ? 64 : 256);
        CHECK_EQ(pflt(q, "theta"), l == 3 ? 5000000.0 : 10000.0);
        CHECK_EQ(pflt(key, "theta"), pflt(q, "theta"));
    }
}

TEST(uses_fp32_residual_exact_gelu_and_one_sigmoid_per_head) {
    MetaKV k = spark_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(spark25_bf16::declare(&b, k.meta(), &ctx), RAD_OK);
    const RecBuf* x = find_b(b, "x");
    REQUIRE(x);
    CHECK_EQ(x->d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(count_op(b, "sigmoid"), 4);
    CHECK_EQ(count_op(b, "scale_rows"), 4);
    CHECK_EQ(count_op(b, "act_mul"), 4);
    for (int l = 0; l < 4; ++l) {
        CHECK_EQ(pstr(find_op(b, "act_mul", l), "act"), std::string("gelu"));
        CHECK_EQ(pint(find_op(b, "scale_rows", l), "n"), 256);
        CHECK_EQ(pstr(find_op(b, "scale_rows", l), "act"), std::string("none"));
    }
    for (const auto& o : b.ops) if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights) if (!w.d.source) CHECK(find_map(b, w.name) != nullptr);
}

TEST(prefill_and_decode_issue_a_liveness_safe_graph) {
    MetaKV k = spark_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(spark25_bf16::declare(&b, k.meta(), &ctx), RAD_OK);
    for (int n : {1, 8}) {
        RadBatch batch = decode_batch();
        batch.n_tok = n;
        batch.max_q_len = n;
        RadCtx c{.b = &b, .batch = &batch};
        spark25_bf16::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
    }
}

TEST(tensor_parallel_covers_heads_and_ffn_without_changing_gqa) {
    for (int world = 1; world <= 8; ++world) {
        std::vector<int> query_owners(16, 0), ffn_owners(10240, 0), vocab_owners(131072, 0);
        for (int rank = 0; rank < world; ++rank) {
            MetaKV k = spark_meta();
            RadBuilder b;
            RadBuildCtx ctx = build_ctx();
            ctx.world_size = world;
            ctx.rank = rank;
            REQUIRE_EQ(spark25_bf16::declare(&b, k.meta(), &ctx), RAD_OK);
            const auto& g = spark25_bf16::models[rank].g;
            const RecWeight* q = find_w(b, "blk.0.attn_q.weight");
            const RecWeight* key = find_w(b, "blk.0.attn_k.weight");
            const RecWeight* gate = find_w(b, "blk.0.attn_gate.weight");
            const RecWeight* out = find_w(b, "blk.0.attn_output.weight");
            const RecWeight* gu = find_w(b, "blk.0.ffn_gate_up.weight");
            const RecWeight* down = find_w(b, "blk.0.ffn_down.weight");
            const RecWeight* tok = find_w(b, "token_embd.weight");
            REQUIRE(q && key && gate && out && gu && down && tok);
            CHECK_EQ(q->d.shard, RAD_SHARD_ROW);
            CHECK_EQ(gate->d.shard, RAD_SHARD_ROW);
            CHECK_EQ(out->d.shard, RAD_SHARD_COL);
            CHECK_EQ(gu->d.shard, RAD_SHARD_ROW);
            CHECK_EQ(down->d.shard, RAD_SHARD_COL);
            CHECK_EQ(gu->d.n_row_parts, 2);
            CHECK_EQ(gu->d.row_parts[0], g.n_ff);
            CHECK_EQ(gu->d.row_parts[1], g.n_ff);
            const int64_t q0 = q->span_set ? q->span_lo / g.head_dim : rank * g.n_head;
            const int64_t k0 = key->span_set ? key->span_lo / g.head_dim : rank * g.n_head_kv;
            const int64_t f0 = gu->span_set ? gu->span_lo : rank * g.n_ff;
            const int64_t v0 = tok->span_set ? tok->span_lo : rank * g.n_vocab;
            CHECK_EQ(gate->span_set ? gate->span_lo : rank * g.n_head, q0);
            CHECK_EQ(out->span_set ? out->span_lo : rank * g.q_dim(), q0 * g.head_dim);
            CHECK_EQ(down->span_set ? down->span_lo : rank * g.n_ff, f0);
            CHECK_EQ(v0, g.vocab_off);
            for (int64_t h = 0; h < g.n_head; ++h) {
                REQUIRE(q0 + h < (int64_t)query_owners.size());
                ++query_owners[(size_t)(q0 + h)];
                CHECK_EQ(k0 + h / g.gqa(), (q0 + h) / 4);
            }
            for (int64_t i = 0; i < g.n_ff; ++i) ++ffn_owners[(size_t)(f0 + i)];
            for (int64_t i = 0; i < g.n_vocab; ++i) ++vocab_owners[(size_t)(v0 + i)];
            CHECK_EQ(count_op(b, "all_reduce"), world == 1 ? 0 : 9);
            int reductions = 0;
            for (const auto& op : b.ops) if (op.op == "all_reduce") {
                CHECK_EQ(pstr(&op, "dtype"), std::string(reductions++ == 0 ? "fp32" : "bf16"));
                CHECK_EQ(pint(&op, "exact"), 1);
                CHECK_EQ(pint(&op, "world_size"), world);
            }
            RadBatch batch = decode_batch();
            batch.n_tok = batch.max_q_len = 8;
            RadCtx c{.b = &b, .batch = &batch, .rank = rank, .world = world};
            spark25_bf16::step(&c, &batch);
            CHECK_LINT(b, &c);
            CHECK(c.step_fail.empty());
        }
        CHECK(std::all_of(query_owners.begin(), query_owners.end(), [](int n) { return n == 1; }));
        CHECK(std::all_of(ffn_owners.begin(), ffn_owners.end(), [](int n) { return n == 1; }));
        CHECK(std::all_of(vocab_owners.begin(), vocab_owners.end(), [](int n) { return n == 1; }));
    }
}

static void spark_iq8r_encodings(RadBuilder& b, bool rotated = true) {
    RadEncoding e = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    if (rotated) rad_enc_copy_str(e.transform, "fwht128");
    for (const char* name : {"attn_q.weight", "attn_k.weight", "attn_v.weight",
                             "attn_output.weight", "ffn_gate_up.weight", "ffn_down.weight"})
        b.encs.emplace_back(name, e);
    b.encs.emplace_back("output.weight", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128));
}

TEST(iq8r_graph_uses_rotated_codes_and_is_liveness_safe) {
    for (int world = 1; world <= 8; ++world) {
        for (int rank = 0; rank < world; ++rank) {
            MetaKV k = spark_meta();
            k.m.quant = "w8a8_fwht128";
            k.set("quantization_config.quant_method", "w8a8")
             .set("quantization_config.fmt", "fwht128");
            RadBuilder b;
            spark_iq8r_encodings(b);
            RadBuildCtx ctx = build_ctx();
            ctx.world_size = world;
            ctx.rank = rank;
            REQUIRE_EQ(spark25_w8a8::declare(&b, k.meta(), &ctx), RAD_OK);
            int quantized = 0;
            for (const auto& op : b.ops) if (op.op == "gemm_nt_q") {
                CHECK_EQ(pstr(&op, "dtype"), std::string("i8a8h"));
                ++quantized;
            }
            CHECK_EQ(quantized, 24);  /* six linears x four layers */
            CHECK_EQ(count_op(b, "all_reduce"), world == 1 ? 0 : 9);
            for (int n : {1, 8}) {
                RadBatch batch = decode_batch();
                batch.n_tok = batch.max_q_len = n;
                RadCtx c{.b = &b, .batch = &batch, .rank = rank, .world = world};
                spark25_w8a8::step(&c, &batch);
                CHECK_LINT(b, &c);
                CHECK(c.step_fail.empty());
            }
        }
    }
}

TEST(iq8r_rejects_unrotated_weights_and_missing_collectives) {
    MetaKV k = spark_meta();
    k.m.quant = "w8a8_fwht128";
    k.set("quantization_config.quant_method", "w8a8")
     .set("quantization_config.fmt", "fwht128");
    RadBuildCtx ctx = build_ctx();
    RadBuilder unrotated;
    spark_iq8r_encodings(unrotated, false);
    CHECK_EQ(stderr_of_rc([&] { return spark25_w8a8::declare(&unrotated, k.meta(), &ctx); }),
             RAD_E_DTYPE);
    RadBuilder missing;
    spark_iq8r_encodings(missing);
    missing.refuse.insert("all_reduce");
    ctx.world_size = 2;
    CHECK_EQ(stderr_of_rc([&] { return spark25_w8a8::declare(&missing, k.meta(), &ctx); }),
             RAD_E_UNSUPPORTED);
}

TEST(iq8r_refuses_containers_with_a_different_format) {
    for (const auto& [method, fmt] : {std::pair{"w8a8", "fwht64"},
                                      std::pair{"fp8", "fwht128"}}) {
        MetaKV k = spark_meta();
        k.m.quant = "w8a8_fwht128";
        k.set("quantization_config.quant_method", method)
         .set("quantization_config.fmt", fmt);
        RadBuilder b;
        spark_iq8r_encodings(b);
        RadBuildCtx ctx = build_ctx();
        CHECK_EQ(stderr_of_rc([&] { return spark25_w8a8::declare(&b, k.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
}

TEST(tp_rejects_missing_all_reduce_instead_of_skipping_it) {
    MetaKV k = spark_meta();
    RadBuilder b;
    b.refuse.insert("all_reduce");
    RadBuildCtx ctx = build_ctx();
    ctx.world_size = 2;
    CHECK_EQ(stderr_of_rc([&] { return spark25_bf16::declare(&b, k.meta(), &ctx); }),
             RAD_E_UNSUPPORTED);

    /* No collective is required for single-device inference. */
    RadBuilder single;
    single.refuse.insert("all_reduce");
    ctx.world_size = 1;
    CHECK_EQ(spark25_bf16::declare(&single, k.meta(), &ctx), RAD_OK);
}

TEST(tp_projection_rounds_to_bf16_before_the_fp32_residual_add) {
    MetaKV k = spark_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.world_size = 2;
    REQUIRE_EQ(spark25_bf16::declare(&b, k.meta(), &ctx), RAD_OK);
    const auto& m = spark25_bf16::models[0];
    RadBatch batch = decode_batch();
    RadCtx c{.b = &b, .batch = &batch, .world = 2};
    spark25_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    REQUIRE(c.step_fail.empty());

    /* Both partials are exactly bf16. Their sum is halfway between 1 and the next bf16 value,
     * so the merged projection must round to 1. The residual's low bits must survive the add. */
    constexpr float partial = 0.5f, peer = 0.50390625f, residual = 0.0009765625f;
    const float merged = rad_bf16_to_f32(rad_f32_to_bf16(partial + peer));
    REQUIRE_EQ(merged, 1.0f);
    auto rounded = [&](rad_buf buf, float value) {
        const uint32_t dtype = b.bufs.at(buf - 1).d.dtype;
        CHECK(dtype == RAD_BF16 || dtype == RAD_F32);
        return dtype == RAD_BF16 ? rad_bf16_to_f32(rad_f32_to_bf16(value)) : value;
    };
    for (const auto& layer : m.layers) {
        for (const auto& tail : {std::pair{layer.op, layer.add_attn},
                                 std::pair{layer.dp, layer.add_ffn}}) {
            std::map<rad_buf, float> values{{m.h, partial}, {m.x, residual}};
            bool active = false, added = false;
            int reductions = 0;
            /* Evaluate one channel of the tail the real step issued, not a separately spelled
             * cast/reduce/add sequence. The collective's peer contributes its projection shard. */
            for (const auto& issue : c.issues) {
                if (!active) { active = issue.op == tail.first; continue; }
                const RecOp* op = b.op_of(issue.op);
                REQUIRE(op != nullptr);
                REQUIRE_EQ(op->writes.size(), (size_t)1);
                REQUIRE(!op->reads.empty());
                REQUIRE(values.count(op->reads[0]) != 0);
                float value = values.at(op->reads[0]);
                if (op->op == "all_reduce") {
                    CHECK_EQ(rad_dtype_parse(pstr(op, "dtype").c_str()),
                             b.bufs.at(op->writes[0] - 1).d.dtype);
                    value += peer;
                    ++reductions;
                } else if (op->op == "add") {
                    REQUIRE_EQ(op->reads.size(), (size_t)2);
                    REQUIRE(values.count(op->reads[1]) != 0);
                    value += values.at(op->reads[1]);
                } else {
                    REQUIRE_EQ(op->op, std::string("cast"));
                }
                values[op->writes[0]] = rounded(op->writes[0], value);
                if (issue.op == tail.second) { added = true; break; }
            }
            REQUIRE(added);
            CHECK_EQ(reductions, 1);
            CHECK_EQ(values.at(m.x), residual + merged);
            CHECK(values.at(m.x) != residual + partial + peer);
        }
    }
}

TEST(tp2_refuses_without_the_exact_fp32_embedding_collective) {
    MetaKV k = spark_meta();
    RadBuilder b;
    b.refuse_dtype.insert({"all_reduce", "fp32"});
    RadBuildCtx ctx = build_ctx();
    ctx.world_size = 2;
    CHECK_EQ(stderr_of_rc([&] { return spark25_bf16::declare(&b, k.meta(), &ctx); }),
             RAD_E_UNSUPPORTED);
    REQUIRE_EQ(count_op(b, "all_reduce"), 1);
    const RecOp* embedding = find_op(b, "all_reduce");
    REQUIRE(embedding);
    CHECK_EQ(pstr(embedding, "dtype"), std::string("fp32"));
    CHECK_EQ(pint(embedding, "exact"), 1);
    CHECK_EQ(pint(embedding, "world_size"), 2);
    CHECK(!embedding->resolved);
}

TEST(tp2_refuses_without_the_exact_bf16_projection_collective) {
    MetaKV k = spark_meta();
    RadBuilder b;
    b.refuse_dtype.insert({"all_reduce", "bf16"});
    RadBuildCtx ctx = build_ctx();
    ctx.world_size = 2;
    CHECK_EQ(stderr_of_rc([&] { return spark25_bf16::declare(&b, k.meta(), &ctx); }),
             RAD_E_UNSUPPORTED);
    REQUIRE_EQ(count_op(b, "all_reduce"), 3);
    const RecOp* embedding = find_op(b, "all_reduce");
    const RecOp* attn = find_op(b, "all_reduce", 1);
    const RecOp* ffn = find_op(b, "all_reduce", 2);
    REQUIRE(embedding && attn && ffn);
    CHECK_EQ(pstr(embedding, "dtype"), std::string("fp32"));
    CHECK(embedding->resolved);
    CHECK_EQ(pstr(attn, "dtype"), std::string("bf16"));
    CHECK_EQ(pstr(ffn, "dtype"), std::string("bf16"));
    CHECK_EQ(pint(attn, "exact"), 1);
    CHECK_EQ(pint(ffn, "exact"), 1);
    CHECK_EQ(pint(attn, "world_size"), 2);
    CHECK_EQ(pint(ffn, "world_size"), 2);
    CHECK(!attn->resolved);
    CHECK(!ffn->resolved);
}

TEST(tp1_works_without_collectives) {
    MetaKV k = spark_meta();
    RadBuilder b;
    b.refuse.insert("all_reduce");
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(spark25_bf16::declare(&b, k.meta(), &ctx), RAD_OK);
    CHECK_EQ(count_op(b, "all_reduce"), 0);
    const auto& m = spark25_bf16::models[ctx.rank];
    CHECK_EQ(m.reduce_embed, (rad_op)0);
    for (const auto& layer : m.layers) {
        CHECK_EQ(layer.reduce_attn, (rad_op)0);
        CHECK_EQ(layer.reduce_ffn, (rad_op)0);
    }
    RadBatch batch = decode_batch();
    RadCtx c{.b = &b, .batch = &batch};
    spark25_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
    const auto issued = issued_ops(b, c);
    CHECK(std::find(issued.begin(), issued.end(), "all_reduce") == issued.end());
}

TEST(spark_prefill_keys_match_live_r4d_split_grid) {
#ifdef SPARK25_HAVE_PREFILL_SPLIT
    spark25_prefill_key::Request r;
    r.phase = RAD_PHASE_PREFILL; r.n_tok = r.max_q_len = 256; r.n_seq = 1;
    r.max_ctx = 65536; r.head_dim = 256; r.gqa = 4; r.kv_heads = 4;
    r.kv_group_found = 1; r.used = 1536; r.wgps = 32;
    const uint64_t k4 = spark25_prefill_key::pass_key(r);
    CHECK_EQ(rad_prefill_splits_for_live(96, 48, 256, 256, 1, 4, 65536, 1536, 32), 4);
    r.used = 2048;
    CHECK_EQ(rad_prefill_splits_for_live(96, 48, 256, 256, 1, 4, 65536, 2048, 32), 5);
    CHECK(spark25_prefill_key::pass_key(r) != k4);
    r = {}; r.phase = RAD_PHASE_PREFILL; r.n_tok = r.max_q_len = 256; r.n_seq = 2;
    r.max_ctx = 65536; r.head_dim = 256; r.gqa = 4; r.kv_heads = 4;
    r.kv_group_found = 1; r.used = 1536; r.wgps = 32;
    CHECK_EQ(spark25_prefill_key::pass_key(r), (uint64_t)0);

    MetaKV meta = spark_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(spark25_bf16::declare(&b, meta.meta(), &ctx), RAD_OK);
    RadKVGroupBatch kv{}; kv.group = spark25_bf16::models[0].kv;
    RadBatch batch{}; batch.phase = RAD_PHASE_PREFILL; batch.n_tok = batch.max_q_len = 256;
    batch.n_seq = batch.n_kv_groups = 1; batch.kv = &kv;
    int32_t used[1] = {1536}; RadCtx c{};
    const uint64_t plugin_k4 = spark25_bf16::pass_key(&c, &batch, used);
    REQUIRE(plugin_k4 != 0);
    used[0] = 2048;
    CHECK(spark25_bf16::pass_key(&c, &batch, used) != plugin_k4);
#else
    CHECK_EQ(spark25_prefill_key::pass_key({}), (uint64_t)0);
#endif
}

TEST(refuses_unimplemented_variants) {
    RadBuildCtx ctx = build_ctx();
    for (const auto& change : {std::pair{"tie_word_embeddings", "0"},
                              std::pair{"headwise_attn_output_gate", "0"},
                              std::pair{"gate_attn_act_mode", "silu"},
                              std::pair{"hidden_act", "silu"},
                              std::pair{"attention_bias", "1"},
                              std::pair{"mlp_bias", "1"},
                              std::pair{"layer_types", "sliding_attention"},
                              std::pair{"rope_parameters.full_attention.rope_type", "dynamic"},
                              std::pair{"rope_parameters.full_attention.partial_rotary_factor", "0.3"}}) {
        MetaKV k = spark_meta();
        k.set(change.first, change.second);
        RadBuilder b;
        CHECK(stderr_of_rc([&] { return spark25_bf16::declare(&b, k.meta(), &ctx); }) != RAD_OK);
    }
    MetaKV k = spark_meta();
    k.set("rope_scaling.rope_type", "linear").set("rope_scaling.factor", "2");
    RadBuilder scaled;
    CHECK(stderr_of_rc([&] { return spark25_bf16::declare(&scaled, k.meta(), &ctx); }) == RAD_E_UNSUPPORTED);
    k = spark_meta();
    RadBuilder b;
    ctx.world_size = 17;  /* more ranks than query heads */
    CHECK(stderr_of_rc([&] { return spark25_bf16::declare(&b, k.meta(), &ctx); }) == RAD_E_INVAL);
}

RAD_TEST_MAIN()
