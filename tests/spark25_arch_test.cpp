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
    RadBuildCtx ctx = build_ctx();
    RadBuilder bf16_builder;
    REQUIRE_EQ(spark25_bf16::declare(&bf16_builder, meta.meta(), &ctx), RAD_OK);
    RadKVGroupBatch kv{}; kv.group = spark25_bf16::models[0].kv;
    RadBatch batch{}; batch.phase = RAD_PHASE_PREFILL; batch.n_tok = batch.max_q_len = 256;
    batch.n_seq = batch.n_kv_groups = 1; batch.kv = &kv;
    int32_t used[1] = {1536}; RadCtx c{};
    const uint64_t bf16_k4 = spark25_bf16::pass_key(&c, &batch, used);
    REQUIRE(bf16_k4 != 0);
    used[0] = 2048;
    CHECK(spark25_bf16::pass_key(&c, &batch, used) != bf16_k4);

    MetaKV iqmeta = spark_meta();
    iqmeta.m.quant = "w8a8_fwht128";
    iqmeta.set("quantization_config.quant_method", "w8a8")
          .set("quantization_config.fmt", "fwht128");
    RadBuilder iq_builder; spark_iq8r_encodings(iq_builder);
    REQUIRE_EQ(spark25_w8a8::declare(&iq_builder, iqmeta.meta(), &ctx), RAD_OK);
    kv.group = spark25_w8a8::models[0].kv;
    used[0] = 1536;
    const uint64_t iq_k4 = spark25_w8a8::pass_key(&c, &batch, used);
    REQUIRE(iq_k4 != 0);
    used[0] = 2048;
    CHECK(spark25_w8a8::pass_key(&c, &batch, used) != iq_k4);
#else
    CHECK_EQ(spark25_prefill_key::pass_key({}), (uint64_t)0);
#endif
}

/* A DSpark drafter merged under `draft.`, as `--draft-model` writes it: five sliding layers over a
 * trunk, a seven-token block, a rank-256 Markov head, and the tap list the published Spark drafts
 * carry (translated into the layers whose output is tapped). */
static MetaKV with_dspark(MetaKV k) {
    k.set("draft.architectures", "DSparkDraftModel")
     .set("draft.model_type", "dspark")
     .set("draft.num_hidden_layers", "5")
     .set("draft.hidden_size", "2560")
     .set("draft.intermediate_size", "10240")
     .set("draft.num_attention_heads", "16")
     .set("draft.num_key_value_heads", "4")
     .set("draft.head_dim", "256")
     .set("draft.rms_norm_eps", "1e-06")
     .set("draft.rope_theta", "10000")
     .set("draft.vocab_size", "131072")
     .set("draft.partial_rotary_factor", "1")
     .set("draft.hidden_act", "gelu")
     .set("draft.is_causal", "0")
     .set("draft.use_sliding_window", "1")
     .set("draft.sliding_window", "512")
     .set("draft.block_size", "7")
     .set("draft.sample_from_anchor", "1")
     .set("draft.mask_token_id", "2")
     .set("draft.target_layer_ids", "1 3")
     .set("draft.gguf_target_layers", "2 4")
     .set("draft.tie_word_embeddings", "1")
     .set("draft.headwise_attn_output_gate", "1")
     .set("draft.gate_attn_act_mode", "sigmoid")
     .set("draft.fc_norm", "1")
     .set("draft.markov_rank", "256")
     .set("draft.markov_head_type", "vanilla")
     .set("draft.has_confidence_head", "1");
    return k;
}

static RadBuildCtx spec_ctx(int max_spec) {
    RadBuildCtx ctx = build_ctx();
    ctx.max_spec = max_spec;
    return ctx;
}

TEST(a_dspark_drafter_is_declared_over_the_trunk) {
    MetaKV k = with_dspark(spark_meta());
    RadBuilder b;
    RadBuildCtx ctx = spec_ctx(7);
    REQUIRE_EQ(spark25_bf16::declare(&b, k.meta(), &ctx), RAD_OK);

    /* What the core drives: a block drafter, seven rows and seven drafts, staged against the
     * window the plugin gated its own attention with. */
    CHECK_EQ(b.drafter.kind, (int)RAD_DRAFT_BLOCK);
    CHECK_EQ(b.drafter.depth, 7);
    CHECK_EQ(b.drafter.block, 7);                       /* every row predicts, the anchor included */
    CHECK_EQ(b.drafter.mask_token, 2);
    CHECK_EQ(b.drafter.window, 513);                    /* 512-key history plus anchor */
    CHECK(b.drafter.proposal != 0);
    CHECK_EQ(b.drafter.proposal_pitch, (int64_t)7);
    CHECK_EQ(b.drafter_name, std::string("dspark"));

    /* Its own exact anchor-aligned KV window, and its layers bound past the trunk's four. */
    REQUIRE_EQ(b.kv_groups.size(), (size_t)2);
    const RecKV& draft_kv = b.kv_groups[1];
    CHECK_EQ(draft_kv.d.kind, (int)RAD_KV_WINDOW);
    CHECK_EQ(draft_kv.d.window, (int64_t)513);
    CHECK_EQ(draft_kv.d.n_head_kv, (int64_t)4);
    CHECK_EQ(draft_kv.d.head_dim, (int64_t)256);
    for (int l = 0; l < 5; ++l) {
        auto it = std::find_if(b.binds.begin(), b.binds.end(),
                               [&](const std::pair<int, rad_kvgroup>& p) { return p.first == 4 + l; });
        REQUIRE(it != b.binds.end());
        CHECK_EQ(it->second, (rad_kvgroup)2);   /* the drafter's group, not the trunk's */
    }

    /* The taps: two column slices of one f32 stream, copied out of the trunk as it passes. */
    const RecBuf* src = find_b(b, "draft_src");
    REQUIRE(src);
    CHECK_EQ(src->d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(src->d.rank, 2u);
    CHECK_EQ(src->d.shape[1], (int64_t)2 * 2560);
    CHECK_EQ(count_op(b, "cast") >= 1, true);

    /* The drafter's weights, at the names the container carries -- including the gate under the
     * `g_proj` spelling the target uses, and the Markov pair as vocabulary-wide codebooks. */
    const RecWeight* fc = find_w(b, "dspark.fc.weight");
    REQUIRE(fc);
    CHECK_EQ(fc->d.dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(fc->d.rank, 2u);
    CHECK_EQ(fc->d.shape[0], (int64_t)2560);
    CHECK_EQ(fc->d.shape[1], (int64_t)2 * 2560);
    const RecMap* fc_map = find_map(b, "dspark.fc.weight");
    REQUIRE(fc_map);
    CHECK_EQ(fc_map->src[0], std::string("draft.fc.weight"));
    REQUIRE(find_w(b, "dspark.fc_norm.0.weight") != nullptr);
    REQUIRE(find_w(b, "dspark.fc_norm.1.weight") != nullptr);
    const RecWeight* gate = find_w(b, "dspark.blk.0.attn_gate");
    REQUIRE(gate);
    CHECK_EQ(gate->d.shape[0], (int64_t)16);            /* one gate a head */
    CHECK_EQ(gate->d.shape[1], (int64_t)2560);
    const RecMap* gate_map = find_map(b, "dspark.blk.0.attn_gate");
    REQUIRE(gate_map);
    CHECK_EQ(gate_map->src[0], std::string("draft.layers.0.self_attn.g_proj.weight"));
    /* Every drafter name, norms included, is the container's name under `draft.` -- a missing
     * separator here reads a tensor that is not there. */
    for (const auto& prm : {std::pair{"dspark.blk.0.attn_norm.weight",
                                      "draft.layers.0.input_layernorm.weight"},
                            {"dspark.blk.0.ffn_norm.weight",
                             "draft.layers.0.post_attention_layernorm.weight"},
                            {"dspark.blk.4.ffn_norm.weight",
                             "draft.layers.4.post_attention_layernorm.weight"}}) {
        const RecMap* m = find_map(b, prm.first);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->src[0], std::string(prm.second));
    }
    const RecWeight* gu = find_w(b, "dspark.blk.0.ffn_gate_up");
    REQUIRE(gu);
    CHECK_EQ(gu->d.n_row_parts, 2);                     /* [gate | up], one weight */
    const RecWeight* mk1 = find_w(b, "dspark.markov_w1");
    REQUIRE(mk1);
    CHECK_EQ(mk1->d.access, RAD_ACCESS_VOCAB);
    CHECK_EQ(mk1->d.shard, RAD_SHARD_NONE);
    CHECK_EQ(mk1->d.shape[0], (int64_t)131072);
    CHECK_EQ(mk1->d.shape[1], (int64_t)256);
    const RecWeight* head = find_w(b, "dspark.head.weight");
    REQUIRE(head);
    CHECK_EQ(head->d.dtype, (uint32_t)RAD_BF16);
    const RecMap* head_map = find_map(b, "dspark.head.weight");
    REQUIRE(head_map);
    CHECK_EQ(head_map->src[0], std::string("model.embedding.weight"));   /* the target's tied table */

    /* Its layers: non-causal attention over the exact shared anchor window, exact-erf gelu, one
     * sigmoid gate a head, and a rotary over the whole head. */
    CHECK_EQ(count_op(b, "attn_paged"), 4);
    CHECK_EQ(count_op(b, "attn_paged_anchor_window"), 1);
    const RecOp* d_attn = find_op(b, "attn_paged_anchor_window", 0);
    REQUIRE(d_attn != nullptr);
    CHECK_EQ(pint(d_attn, "causal"), 0);
    CHECK_EQ(pint(d_attn, "window"), 513);
    CHECK_EQ(pint(d_attn, "gqa"), 4);
    /* The M range tops out at the block, not at a token count. */
    const RecParam* ql = nullptr;
    for (const auto& prm : d_attn->p) if (prm.key == "q_len") ql = &prm;
    REQUIRE(ql != nullptr);
    CHECK_EQ(ql->ival, 1);
    CHECK_EQ(ql->ihi, 7);
    CHECK_EQ(pstr(d_attn, "q_dtype"), std::string("bf16"));
    for (const auto& o : b.ops) {
        if (o.op != "act_mul") continue;
        CHECK_EQ(pstr(&o, "act"), std::string("gelu"));  /* exact erf, as the trunk's */
    }
    CHECK_EQ(count_op(b, "rope") >= 4 * 2 + 1, true);
    const RecOp* markov = find_op(b, "dflash_markov_greedy_f32", 0);
    REQUIRE(markov);
    CHECK_EQ(pint(markov, "steps"), 7);
    CHECK_EQ(pint(markov, "rank"), 256);
    CHECK_EQ(pint(markov, "anchor_stride"), 7);         /* token_ids is `block` a sequence */
    CHECK_EQ(pint(markov, "n_vocab"), (int64_t)131072);
    CHECK_EQ(pstr(markov, "dtype"), std::string("bf16"));
    CHECK_EQ(find_op(b, "dflash_select", 0), nullptr);
    CHECK_EQ(find_op(b, "row_topk", 0), nullptr);
    const RecBuf* best_scores = nullptr;
    const RecBuf* best_tokens = nullptr;
    for (const auto& bf : b.bufs) {
        if (bf.name == "ds_markov_score") best_scores = &bf;
        if (bf.name == "ds_markov_token") best_tokens = &bf;
    }
    REQUIRE(best_scores && best_tokens);
    CHECK_EQ(best_scores->d.shape[0], (int64_t)8);
    CHECK_EQ(best_scores->d.shape[1], (int64_t)(131072 + 63) / 64);
    const RecBuf* dlog = nullptr;
    for (const auto& bf : b.bufs) if (bf.name == "ds_dlog") dlog = &bf;
    REQUIRE(dlog != nullptr);
    CHECK_EQ(dlog->d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(count_op(b, "logits_gemm"), 2);            /* trunk and draft F32 heads */
    CHECK(count_op(b, "gemm_nt") > 0);

    /* The drafter is declared after the trunk, and the graph resolves: no op the plugin declared is
     * left unused, and every weight it reads came from a name map. */
    for (const auto& o : b.ops) if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights) if (!w.d.source) CHECK(find_map(b, w.name) != nullptr);
    const std::string notes = [&] {
        std::string n;
        for (const auto& s : b.notes) n += s + "\n";
        return n;
    }();
    CHECK(notes.find("DSpark drafter") != std::string::npos);
    CHECK(notes.find("after layer(s) 1, 3") != std::string::npos);
}

TEST(the_drafter_passes_run_and_the_draft_is_proposed_from_the_block) {
    MetaKV k = with_dspark(spark_meta());
    RadBuilder b;
    RadBuildCtx ctx = spec_ctx(7);
    REQUIRE_EQ(spark25_bf16::declare(&b, k.meta(), &ctx), RAD_OK);

    /* One engine step, as the scheduler drives it: a verify pass over the accepted block, the
     * drafter's context pass over what the trunk just committed, then its query pass over the
     * staged block (the anchor and six masks). */
    const auto ver = kit(b, std::vector<int>(4, 8), {});
    const auto ctx_pass = context_pass(b, *ver);
    const auto qry = query_pass(b, 4, 7, 7);            /* seven rows, seven drafts */
    const auto v = run_passes(b, spark25_bf16::step, { ver.get(), ctx_pass.get(), qry.get() });
    CHECK_PASSES(b, v);

    /* The query pass is the drafter alone: five attention layers, the head, and the walk that
     * writes the proposal buffer -- and no trunk layer at all. */
    RadCtx& q = *v[2];
    int attn = 0, gemm = 0;
    bool selected = false, proposed = false, draft_head = false;
    rad_buf head_logits = 0;
    for (const auto& i : q.issues) {
        const RecOp* o = b.op_of(i.op);
        REQUIRE(o != nullptr);
        if (o->op == "attn_paged_anchor_window") ++attn;
        if (o->op == "gemm_nt") ++gemm;
        if (o->op == "logits_gemm") {
            REQUIRE_EQ(i.opd.size(), (size_t)3);
            REQUIRE_EQ(i.opd[2].kind, (uint8_t)RAD_OPK_BUF);
            head_logits = i.opd[2].handle;
            REQUIRE(head_logits > 0 && head_logits <= b.bufs.size());
            CHECK_EQ(b.bufs[head_logits - 1].d.dtype, (uint32_t)RAD_F32);
            draft_head = true;
        }
        if (o->op == "dflash_markov_greedy_f32") {
            selected = true;
            REQUIRE_EQ(i.opd.size(), (size_t)7);   /* logits, anchor, pred, succ, tiles, tokens */
            CHECK_EQ(i.opd[0].handle, head_logits);
            CHECK_EQ(i.opd[6].kind, (uint8_t)RAD_OPK_BUF);
            CHECK_EQ(i.opd[6].handle, b.drafter.proposal);
            proposed = i.opd[6].handle == b.drafter.proposal;
        }
    }
    CHECK(draft_head);
    CHECK_EQ(attn, 5);
    CHECK(gemm > 0);
    CHECK(selected);
    CHECK(proposed);

    /* The context pass stores K and V for every committed token and proposes nothing. */
    RadCtx& h = *v[1];
    int stores = 0;
    for (const auto& i : h.issues) {
        const RecOp* o = b.op_of(i.op);
        REQUIRE(o != nullptr);
        if (o->op == "kv_store") ++stores;
    }
    CHECK_EQ(stores, 5);
    auto drafter_silent = [&](const RadCtx& pass) {
        for (const auto& i : pass.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o || o->op == "dflash_markov_greedy_f32" || o->op == "dflash_select")
                return false;
            for (const auto& operand : i.opd)
                if (operand.kind == RAD_OPK_BUF &&
                    (operand.handle == b.drafter.proposal || operand.handle == head_logits))
                    return false;
        }
        return true;
    };
    CHECK(drafter_silent(h));
    CHECK(!drafter_silent(q));  /* negative control: a real draft pass must fail the silence test */

    /* And the trunk's own step, with the drafter silent. */
    const auto dec = kit(b, { 1, 1, 1, 1 }, {});
    const auto d = run_passes(b, spark25_bf16::step, { dec.get() });
    CHECK_PASSES(b, d);
    CHECK(drafter_silent(*d[0]));
}

TEST(a_drafter_is_refused_when_the_pair_or_the_depth_is_wrong) {
    /* A serve that asks for another depth: a block-diffusion drafter has no partial block. */
    {
        MetaKV k = with_dspark(spark_meta());
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(3);
        CHECK_EQ(stderr_of_rc([&] { return spark25_bf16::declare(&b, k.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
    /* The head is read replicated, so a vocab-sharded world cannot draft from it. */
    {
        MetaKV k = with_dspark(spark_meta());
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(7);
        ctx.world_size = 2;
        CHECK_EQ(stderr_of_rc([&] { return spark25_bf16::declare(&b, k.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
    for (const auto& change : {std::pair{"draft.hidden_size", "4096"},         /* another target */
                               std::pair{"draft.markov_rank", "0"},            /* no head to walk */
                               std::pair{"draft.markov_rank", "24"},           /* not a strided rank */
                               std::pair{"draft.sample_from_anchor", "0"},     /* a slot with no rows */
                               std::pair{"draft.hidden_act", "gelu_tanh"},     /* not what it trained */
                               std::pair{"draft.gate_attn_act_mode", "tanh"},
                               std::pair{"draft.architectures", "DFlashDraftModel"},
                               std::pair{"draft.sliding_window", "4"},         /* shorter than a block */
                               std::pair{"draft.block_size", "0"}}) {
        MetaKV k = with_dspark(spark_meta());
        k.set(change.first, change.second);
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(7);
        CHECK(stderr_of_rc([&] { return spark25_bf16::declare(&b, k.meta(), &ctx); }) != RAD_OK);
    }
    /* A library that only has the older per-query window or candidate-pruned selector must refuse
     * DSpark declaration, not silently substitute an operation with different semantics. */
    for (const char* op : {"attn_paged_anchor_window", "dflash_markov_greedy_f32"}) {
        MetaKV k = with_dspark(spark_meta());
        RadBuilder b;
        b.refuse.insert(op);
        RadBuildCtx ctx = spec_ctx(7);
        CHECK(stderr_of_rc([&] { return spark25_bf16::declare(&b, k.meta(), &ctx); }) != RAD_OK);
    }
    /* A drafter whose tap list names only layers past the target's last reads nothing, which is not
     * a drafter. */
    {
        MetaKV k = with_dspark(spark_meta());
        k.set("draft.target_layer_ids", "9 11");
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(7);
        CHECK_EQ(stderr_of_rc([&] { return spark25_bf16::declare(&b, k.meta(), &ctx); }),
                 RAD_E_FORMAT);
    }
    /* With no speculation asked for, the drafter is not declared and its weights stay out of the
     * program -- but the container still serves. */
    {
        MetaKV k = with_dspark(spark_meta());
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(0);
        REQUIRE_EQ(spark25_bf16::declare(&b, k.meta(), &ctx), RAD_OK);
        CHECK_EQ(b.drafter.kind, (int)RAD_DRAFT_NONE);
        CHECK(find_w(b, "dspark.fc.weight") == nullptr);
    }
    /* And the probe tells the engine what depth the container supports, before declare. */
    RadArchProbe p{};
    MetaKV k = with_dspark(spark_meta());
    REQUIRE_EQ(spark25_bf16::probe(k.meta(), &p), RAD_OK);
    CHECK_EQ(p.draft_depth, 7);
    CHECK_EQ(p.draft_depth_fixed, 1);
    RadArchProbe plain{};
    MetaKV none = spark_meta();
    REQUIRE_EQ(spark25_bf16::probe(none.meta(), &plain), RAD_OK);
    CHECK_EQ(plain.draft_depth, 0);
}

/* The int8 trunk carries the same drafter: the drafter reads the trunk's residual stream, which is
 * f32 whatever the linears are, so the wiring is the trunk's and the graph is the drafter's. */
TEST(the_int8_trunk_carries_the_same_drafter) {
    MetaKV k = with_dspark(spark_meta());
    k.m.quant = "w8a8_fwht128";
    k.set("quantization_config.quant_method", "w8a8")
     .set("quantization_config.fmt", "fwht128");
    RadBuilder b;
    spark_iq8r_encodings(b);
    RadBuildCtx ctx = spec_ctx(7);
    REQUIRE_EQ(spark25_w8a8::declare(&b, k.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.drafter.kind, (int)RAD_DRAFT_BLOCK);
    CHECK_EQ(b.drafter.depth, 7);
    CHECK_EQ(b.drafter.window, 513);
    CHECK(find_w(b, "dspark.fc.weight") != nullptr);
    CHECK_EQ(count_op(b, "attn_paged_anchor_window"), 1);
    CHECK_EQ(count_op(b, "dflash_markov_greedy_f32"), 1);
    CHECK_EQ(count_op(b, "dflash_select"), 0);
    const auto ver = kit(b, std::vector<int>(4, 8), {});
    const auto ctx_pass = context_pass(b, *ver);
    const auto qry = query_pass(b, 4, 7, 7);
    const auto v = run_passes(b, spark25_w8a8::step, { ver.get(), ctx_pass.get(), qry.get() });
    CHECK_PASSES(b, v);
    /* The trunk's quantised linears are untouched by the drafter's bf16 graph. */
    CHECK(count_op(b, "gemm_nt_q") > 0);

    RadArchProbe p{};
    REQUIRE_EQ(spark25_w8a8::probe(k.meta(), &p), RAD_OK);
    CHECK_EQ(p.draft_depth, 7);
    CHECK_EQ(p.draft_depth_fixed, 1);
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
