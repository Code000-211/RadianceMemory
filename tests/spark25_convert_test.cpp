/* A small Spark checkpoint converted with the production name map. */
#include "rad_test.h"
#include "tiny_model.h"
#include "format/radfile.h"

#include <map>

using namespace rad;

TEST(bf16_conversion_preserves_qkv_rows_gate_up_and_tied_head) {
    tiny::TempDir d("spark25_convert");
    REQUIRE(d.ok());
    const std::string ckpt = d / "model";
    REQUIRE(std::filesystem::create_directory(ckpt));
    constexpr int H = 32, Q = 32, K = 16, F = 64, V = 320;
    nlohmann::ordered_json config = {
        {"model_type", "spark2_5"}, {"architectures", {"Spark2_5ForCausalLM"}},
        {"hidden_size", H}, {"num_hidden_layers", 2}, {"num_attention_heads", 4},
        {"num_key_value_heads", 2}, {"head_dim", 8}, {"intermediate_size", F},
        {"vocab_size", V}, {"max_position_embeddings", 64}, {"rms_norm_eps", 1e-6},
        {"tie_word_embeddings", true}, {"headwise_attn_output_gate", true},
        {"gate_attn_act_mode", "sigmoid"}, {"hidden_act", "gelu"}, {"sliding_window", 4},
        {"layer_types", {"sliding_attention", "full_attention"}},
        {"rope_parameters", {{"sliding_attention", {{"rope_theta", 10000}, {"partial_rotary_factor", 1.0}}},
                             {"full_attention", {{"rope_theta", 5000000}, {"partial_rotary_factor", 0.5}}}}}
    };
    tiny::Rng r(123);
    std::vector<tiny::Tensor> ts;
    ts.push_back(tiny::noise("model.embedding.weight", {V, H}, r, 0.2f));
    ts.push_back(tiny::noise("model.norm.weight", {H}, r, 0.1f, 1.0f));
    for (int l = 0; l < 2; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        ts.push_back(tiny::noise(p + "input_layernorm.weight", {H}, r, 0.1f, 1.0f));
        ts.push_back(tiny::noise(p + "post_attention_layernorm.weight", {H}, r, 0.1f, 1.0f));
        ts.push_back(tiny::linear(p + "self_attn.q_k_v_proj.weight", Q + 2*K, H, r));
        ts.push_back(tiny::linear(p + "self_attn.g_proj.weight", 4, H, r));
        ts.push_back(tiny::linear(p + "self_attn.out_proj.weight", H, Q, r));
        ts.push_back(tiny::linear(p + "mlp.gate_proj.weight", F, H, r));
        ts.push_back(tiny::linear(p + "mlp.up_proj.weight", F, H, r));
        ts.push_back(tiny::linear(p + "mlp.down_proj.weight", H, F, r));
    }
    REQUIRE(tiny::write_checkpoint(ckpt, config, ts));
    std::map<std::string, std::string> source;
    for (const auto& t : ts) source[t.name] = std::string(t.bytes.begin(), t.bytes.end());
    std::string log;
    const int st = tiny::run({RAD_E2E_CONVERT, ckpt, "-o", d / "model.rad", "--max-ctx", "64",
                              "--max-tok", "16"},
                             {std::string("RADIANCE_HOME=") + RAD_E2E_HOME,
                              "ROCR_VISIBLE_DEVICES=99", "OMP_NUM_THREADS=2", "OMP_WAIT_POLICY=passive"},
                             &log);
    if (st) fprintf(stderr, "%s\n", log.c_str());
    REQUIRE_EQ(st, 0);
    RadFile f;
    REQUIRE_EQ(f.open((d / "model.rad").c_str()), RAD_OK);
    CHECK_EQ(std::string(f.str(f.header().arch_id)), std::string("spark25"));
    CHECK_EQ(f.entry_count(), (int64_t)21);
    auto check = [&](const std::string& name, const std::string& bytes) {
        const RadFileEntry* e = f.find(name);
        if (!e) { CHECK(e != nullptr); return; }
        CHECK_EQ(e->n_planes, (uint32_t)1);
        const RadFilePlane& p = f.plane(*e, 0);
        CHECK_EQ(p.bytes, (uint64_t)bytes.size());
        if (p.bytes == bytes.size()) CHECK(std::memcmp(f.data(p), bytes.data(), bytes.size()) == 0);
    };
    check("token_embd.weight", source.at("model.embedding.weight"));
    check("output.weight", source.at("model.embedding.weight"));
    for (int l = 0; l < 2; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        const std::string name = "blk." + std::to_string(l) + ".";
        const std::string& qkv = source.at(p + "self_attn.q_k_v_proj.weight");
        const size_t row = H * 2;
        check(name + "attn_q.weight", qkv.substr(0, Q*row));
        check(name + "attn_k.weight", qkv.substr(Q*row, K*row));
        check(name + "attn_v.weight", qkv.substr((Q+K)*row, K*row));
        check(name + "ffn_gate_up.weight", source.at(p + "mlp.gate_proj.weight") +
                                           source.at(p + "mlp.up_proj.weight"));
        check(name + "attn_gate.weight", source.at(p + "self_attn.g_proj.weight"));
    }
}

RAD_TEST_MAIN()
