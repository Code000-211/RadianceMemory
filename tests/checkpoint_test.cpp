/* checkpoint_test.cpp -- a checkpoint as a source of logical weights (core/format/checkpoint.h).
 *
 * rad-convert and the loader (which serves a trivial checkpoint directly) both go through
 * ckpt_resolve: which tensors make a declared weight, their encoding, and its rows. A wrong answer
 * yields a model that loads but computes wrongly (an expert that is the whole stack, a scale
 * applied to the next piece's rows), so each case builds one real-world layout small and reads it
 * back.
 */
#include "rad_test.h"
#include "gguf_writer.h"
#include "format/checkpoint.h"
#include "format/encoding.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>

using namespace rad;

namespace {

struct T {
    std::string name, dtype;          /* safetensors spelling: BF16, F32, F8_E4M3 */
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

/* A new empty directory under the test's temporary directory, so no checkpoint holds a file
 * another case wrote. */
std::string fresh_dir() {
    std::string t = radtest::tmp_path("ckpt.XXXXXX");
    if (!mkdtemp(t.data())) throw std::runtime_error("cannot make " + t);
    return t;
}

/* One safetensors file holding `ts`, in order. */
void write_shard(const std::string& path, const std::vector<T>& ts) {
    std::string hdr = "{";
    uint64_t off = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        std::string shape;
        for (size_t k = 0; k < ts[i].shape.size(); ++k)
            shape += (k ? "," : "") + std::to_string(ts[i].shape[k]);
        hdr += (i ? "," : "") + std::string("\"") + ts[i].name + "\":{\"dtype\":\"" + ts[i].dtype +
               "\",\"shape\":[" + shape + "],\"data_offsets\":[" + std::to_string(off) + "," +
               std::to_string(off + ts[i].bytes.size()) + "]}";
        off += ts[i].bytes.size();
    }
    hdr += "}";
    while (hdr.size() % 8) hdr += ' ';
    FILE* f = std::fopen(path.c_str(), "wb");
    const uint64_t n = hdr.size();
    std::fwrite(&n, 8, 1, f);
    std::fwrite(hdr.data(), 1, hdr.size(), f);
    for (const T& t : ts)
        if (!t.bytes.empty()) std::fwrite(t.bytes.data(), 1, t.bytes.size(), f);
    std::fclose(f);
}

/* One model.safetensors and a config.json in a fresh directory. */
void write_file(const std::string& path, const std::string& body) {
    FILE* f = std::fopen(path.c_str(), "wb");
    std::fwrite(body.data(), 1, body.size(), f);
    std::fclose(f);
}

std::string write_checkpoint(const std::vector<T>& ts, const std::string& config) {
    const std::string dir = fresh_dir();
    write_shard(dir + "/model.safetensors", ts);
    write_file(dir + "/config.json", config);
    return dir;
}

/* Element i is tag + (i % 64), exact in bf16, so a value identifies its tensor and position. */
T bf16(const std::string& name, std::vector<int64_t> shape, float tag) {
    T t{ name, "BF16", shape, {} };
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    t.bytes.resize((size_t)n * 2);
    for (int64_t i = 0; i < n; ++i) rad_store_f32(t.bytes.data(), RAD_BF16, i, tag + (float)(i % 64));
    return t;
}

T cache_scale_f32(const std::string& name, float value) {
    T t{ name, "F32", {1}, std::vector<uint8_t>(4) };
    std::memcpy(t.bytes.data(), &value, sizeof value);
    return t;
}

T fp8(const std::string& name, std::vector<int64_t> shape) {
    T t{ name, "F8_E4M3", shape, {} };
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    t.bytes.resize((size_t)n);
    for (int64_t i = 0; i < n; ++i) rad_store_f32(t.bytes.data(), RAD_F8E4M3, i, (float)(i % 7) - 3.0f);
    return t;
}

RadNameMap map(const char* declared, int mode, std::vector<const char*> src,
               std::vector<int> index = {}, int concat_dim = 0) {
    RadNameMap m{};
    m.declared = declared;
    m.mode = mode;
    m.n_src = (int)src.size();
    m.concat_dim = concat_dim;
    for (int i = 0; i < 8; ++i) m.src_index[i] = -1;
    for (size_t i = 0; i < src.size(); ++i) m.src[i] = src[i];
    for (size_t i = 0; i < index.size(); ++i) m.src_index[i] = index[i];
    return m;
}

const char* kConfig =
    "{\"model_type\":\"qwen3_5\",\"hidden_size\":256,\"num_hidden_layers\":2,"
    "\"quantization_config\":{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\","
    "\"weight_block_size\":[128,128]}}";

}  /* namespace */

TEST(the_metadata_names_the_architecture_and_the_quantisation) {
    const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK_EQ(ck.meta().arch, std::string("qwen35"));
    CHECK_EQ(ck.meta().quant, std::string("fp8_e4m3"));
    CHECK_EQ(ck.meta().m.n_embd, (int64_t)256);
    CHECK_EQ(ck.fp8_block_rows(), (int64_t)128);
    CHECK_EQ(ck.dir(), dir);
}

TEST(routed_expert_metadata_accepts_the_deepseek_and_glm_config_spellings) {
    const std::string dir = write_checkpoint({ bf16("x", {2, 2}, 0) },
        "{\"model_type\":\"glm5_next\",\"text_config\":{\"hidden_size\":4096,"
        "\"num_attention_heads\":64,\"head_dim\":0,\"n_routed_experts\":288,"
        "\"n_shared_experts\":1,\"num_experts_per_tok\":8}}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK_EQ(ck.meta().arch, std::string("glm5next"));
    CHECK_EQ(ck.meta().m.n_expert, (int64_t)288);
    CHECK_EQ(ck.meta().m.n_expert_shared, (int64_t)1);
    CHECK_EQ(ck.meta().m.n_expert_used, (int64_t)8);
    CHECK_EQ(ck.meta().m.head_dim, (int64_t)0);
}

/* The descriptor derives from quant_method and fmt whoever set them, including rad-convert's
 * --set: a bf16 checkpoint converted to block fp8 states that format and selects the fp8 plugin,
 * so the container never names one format and selects another's plugin. transformers' fp8 states
 * no fmt and is E4M3 (Qwen3.8-Flash-Next-FP8), so it is the same descriptor; another method
 * without a fmt is its name alone. */
TEST(the_descriptor_is_quant_method_and_fmt_whoever_set_them) {
    const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) },
                                             "{\"model_type\":\"qwen3_5\",\"hidden_size\":256}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK_EQ(ck.meta().quant, std::string());
    MetaOwn m = ck.meta();
    m.set("quantization_config.quant_method", "fp8");
    CHECK_EQ(quant_descriptor(m), std::string("fp8_e4m3"));
    m.set("quantization_config.fmt", "e4m3");
    CHECK_EQ(quant_descriptor(m), std::string("fp8_e4m3"));
    MetaOwn ct = ck.meta();
    ct.set("quantization_config.quant_method", "compressed-tensors");
    CHECK_EQ(quant_descriptor(ct), std::string("compressed-tensors"));
}

/* Google's bf16 E2B needs no `--set quantization_config.quant_method`: its per-layer embeddings
 * name the E form, which arch/gemma4 refuses and arch/gemma4_bf16 serves. */
TEST(a_gemma4_e2b_checkpoint_selects_the_bf16_plugin_on_its_own) {
    const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) },
        "{\"model_type\":\"gemma4\",\"text_config\":{\"hidden_size\":1536,"
        "\"intermediate_size\":6144,\"hidden_size_per_layer_input\":256,"
        "\"num_kv_shared_layers\":20,\"use_double_wide_mlp\":true}}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK_EQ(ck.meta().arch, std::string("gemma4"));
    CHECK_EQ(ck.meta().quant, std::string("compressed-tensors"));
}

/* E4B states shared KV layers without per-layer embeddings; either key selects the same plugin. */
TEST(a_gemma4_e4b_checkpoint_selects_the_bf16_plugin_on_its_own) {
    const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) },
        "{\"model_type\":\"gemma4\",\"text_config\":{\"hidden_size\":2560,"
        "\"intermediate_size\":10240,\"num_kv_shared_layers\":6,"
        "\"use_double_wide_mlp\":false}}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK_EQ(ck.meta().quant, std::string("compressed-tensors"));
    /* The rule is gemma4's: another architecture with the same key keeps its own descriptor. */
    const std::string other = write_checkpoint({ bf16("x", { 2, 2 }, 0) },
        "{\"model_type\":\"ltx2\",\"text_config\":{\"num_kv_shared_layers\":6}}");
    Checkpoint ok;
    REQUIRE_EQ(ok.open(other), RAD_OK);
    CHECK_EQ(ok.meta().quant, std::string());
}

/* The 26B-A4B/31B shape declares neither key and keeps the bare descriptor arch/gemma4 owns. */
TEST(a_gemma4_31b_checkpoint_keeps_the_bare_descriptor) {
    const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) },
        "{\"model_type\":\"gemma4\",\"text_config\":{\"hidden_size\":5376,"
        "\"intermediate_size\":21504,\"num_hidden_layers\":62}}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK_EQ(ck.meta().arch, std::string("gemma4"));
    CHECK_EQ(ck.meta().quant, std::string());
}

TEST(a_plain_tensor_is_its_own_encoding_and_reads_back_as_itself) {
    const std::string dir = write_checkpoint({ bf16("a.weight", { 4, 8 }, 1.0f) }, "{}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("a", RAD_MAP_COPY, { "a.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "a", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    CHECK(rad_enc_is(&w.enc, "plain"));
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(w.rows, (int64_t)4);
    CHECK_EQ(w.cols, (int64_t)8);
    std::vector<float> f(8);
    REQUIRE_EQ(ckpt_rows_f32(w, 2, 1, f.data(), &why), RAD_OK);
    for (int j = 0; j < 8; ++j) CHECK_EQ(f[(size_t)j], 1.0f + (float)((16 + j) % 64));
    std::vector<uint8_t> raw(8 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 2, 1, raw.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(raw.data(), (const uint8_t*)ck.find("a.weight")->data + 32, 16), 0);
}

/* A single-file SDXL checkpoint (Stability's sgm layout) is read under the names its adapters use:
 * the UNet and CLIP ViT-L as they are, OpenCLIP bigG in transformers' names with its fused
 * in_proj split into q/k/v rows and its (in, out) text_projection transposed, the VAE decoder in
 * diffusers' names with up blocks counted the other way, a v_pred marker as the prediction type,
 * and bigG's dimensions as the model's. */
TEST(a_single_file_sdxl_checkpoint_is_read_in_its_adapters_names) {
    const std::string g = "conditioner.embedders.1.model.";
    T proj{ g + "text_projection", "F32", { 2, 3 }, {} };
    proj.bytes.resize(6 * 4);
    for (int i = 0; i < 6; ++i) rad_store_f32(proj.bytes.data(), RAD_F32, i, (float)i);
    const std::string dir = fresh_dir();
    const std::string file = dir + "/sdxl.safetensors";
    write_shard(file, {
        bf16("model.diffusion_model.label_emb.0.0.weight", { 2, 4 }, 1.0f),
        bf16("conditioner.embedders.0.transformer.text_model.final_layer_norm.weight", { 4 }, 2.0f),
        bf16(g + "transformer.resblocks.3.attn.in_proj_weight", { 6, 4 }, 0.0f),
        bf16(g + "transformer.resblocks.3.attn.in_proj_bias", { 6 }, 0.0f),
        bf16(g + "transformer.resblocks.3.mlp.c_fc.weight", { 8, 4 }, 3.0f),
        bf16(g + "transformer.resblocks.3.ln_2.bias", { 4 }, 4.0f),
        bf16(g + "ln_final.weight", { 4 }, 5.0f),
        bf16(g + "token_embedding.weight", { 16, 4 }, 0.0f),
        bf16(g + "transformer.resblocks.0.ln_1.weight", { 4 }, 0.0f),
        bf16(g + "transformer.resblocks.0.mlp.c_fc.weight", { 8, 4 }, 0.0f),
        bf16(g + "positional_embedding", { 77, 4 }, 6.0f),
        proj,
        bf16("first_stage_model.decoder.up.0.block.2.nin_shortcut.weight", { 4, 8, 1, 1 }, 7.0f),
        bf16("first_stage_model.decoder.up.3.upsample.conv.bias", { 4 }, 8.0f),
        bf16("first_stage_model.decoder.mid.attn_1.proj_out.weight", { 4, 4, 1, 1 }, 9.0f),
        bf16("first_stage_model.encoder.conv_in.weight", { 4, 3, 3, 3 }, 10.0f),
        T{ "v_pred", "F32", { 0 }, {} },
    });
    Checkpoint ck;
    REQUIRE_EQ(ck.open(file), RAD_OK);
    CHECK_EQ(ck.meta().arch, std::string("sdxl"));
    REQUIRE(ck.meta().get("sdxl.prediction") != nullptr);
    CHECK_EQ(std::string(ck.meta().get("sdxl.prediction")), std::string("v"));
    /* The typed dimensions are bigG's: its width, vocabulary and feed-forward off its tensors,
     * its layers counted from 0 while they run on. */
    CHECK_EQ(ck.meta().m.n_embd, (int64_t)4);
    CHECK_EQ(ck.meta().m.n_vocab, (int64_t)16);
    CHECK_EQ(ck.meta().m.n_ff, (int64_t)8);
    CHECK_EQ(ck.meta().m.n_layers, (int64_t)1);
    CHECK(ck.find("model.diffusion_model.label_emb.0.0.weight") != nullptr);
    CHECK(ck.find("conditioner.embedders.0.transformer.text_model.final_layer_norm.weight") != nullptr);
    const std::string t = "conditioner.embedders.1.transformer.";
    const std::string l = t + "text_model.encoder.layers.3.";
    const SrcTensor* q = ck.find(l + "self_attn.q_proj.weight");
    const SrcTensor* v = ck.find(l + "self_attn.v_proj.weight");
    const SrcTensor* vb = ck.find(l + "self_attn.v_proj.bias");
    REQUIRE(q && v && vb);
    CHECK(q->shape == std::vector<int64_t>({ 2, 4 }));
    /* v is rows 4..5 of in_proj: elements 16..23, each its index (mod 64) in bf16. */
    CHECK_EQ(rad_load_f32(v->data, RAD_BF16, 0), 16.0f);
    CHECK_EQ(rad_load_f32(vb->data, RAD_BF16, 1), 5.0f);
    CHECK(ck.find(g + "transformer.resblocks.3.attn.in_proj_weight") == nullptr);
    CHECK(ck.find(l + "mlp.fc1.weight") != nullptr);
    CHECK(ck.find(l + "layer_norm2.bias") != nullptr);
    CHECK(ck.find(t + "text_model.final_layer_norm.weight") != nullptr);
    CHECK(ck.find(t + "text_model.embeddings.position_embedding.weight") != nullptr);
    const SrcTensor* p = ck.find(t + "text_projection.weight");
    REQUIRE(p != nullptr);
    CHECK(p->shape == std::vector<int64_t>({ 3, 2 }));
    /* P^T: element (j, i) is P(i, j) = 3 i + j. */
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 2; ++i)
            CHECK_EQ(rad_load_f32(p->data, RAD_F32, j * 2 + i), (float)(3 * i + j));
    CHECK(ck.find("vae.decoder.up_blocks.3.resnets.2.conv_shortcut.weight") != nullptr);
    CHECK(ck.find("vae.decoder.up_blocks.0.upsamplers.0.conv.bias") != nullptr);
    CHECK(ck.find("vae.decoder.mid_block.attentions.0.to_out.0.weight") != nullptr);
    CHECK(ck.find("first_stage_model.encoder.conv_in.weight") != nullptr);
}

/* Block-scaled FP8 is one weight of two planes; each value is its code times the scale of the
 * block containing it along both axes. */
TEST(block_fp8_is_the_codes_and_their_scale_plane) {
    const int64_t N = 256, K = 384;
    T s = bf16("p.weight_scale_inv", { 2, 3 }, 0.0f);
    for (int i = 0; i < 6; ++i) rad_store_f32(s.bytes.data(), RAD_BF16, i, 0.5f * (float)(i + 1));
    const std::string dir = write_checkpoint({ fp8("p.weight", { N, K }), s }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("p", RAD_MAP_COPY, { "p.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "p", false, &w, &why), RAD_OK);
    REQUIRE(w.trivial);
    CHECK(rad_enc_is(&w.enc, "affine"));
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)128);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)128);
    CHECK(rad_enc_valid(&w.enc));
    std::vector<float> f((size_t)(2 * K));
    REQUIRE_EQ(ckpt_rows_f32(w, 129, 2, f.data(), &why), RAD_OK);
    for (int64_t r = 0; r < 2; ++r)
        for (int64_t c : { (int64_t)0, (int64_t)130, (int64_t)383 }) {
            const int64_t i = (129 + r) * K + c;
            const float code = (float)(i % 7) - 3.0f;
            const float scale = 0.5f * (float)(3 + c / 128 + 1);   /* block row 1 */
            CHECK_EQ(f[(size_t)(r * K + c)], code * scale);
        }
    std::vector<uint8_t> sp(3 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 1, 1, sp.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(sp.data(), s.bytes.data() + 6, 6), 0);
}

/* A fusion joins the pieces and their scale planes along dim 0. A piece ending inside a scale
 * block is not trivial, since its scales cannot follow it. */
TEST(a_fusion_joins_pieces_and_their_scales) {
    const std::string dir = write_checkpoint(
        { fp8("g.weight", { 256, 128 }), bf16("g.weight_scale_inv", { 2, 1 }, 10.0f),
          fp8("u.weight", { 128, 128 }), bf16("u.weight_scale_inv", { 1, 1 }, 20.0f),
          fp8("odd.weight", { 100, 128 }), bf16("odd.weight_scale_inv", { 1, 1 }, 30.0f) },
        kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    prog.name_map.push_back(map("bad", RAD_MAP_CONCAT, { "odd.weight", "u.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    CHECK_EQ(w.shape[0], (int64_t)384);
    std::vector<uint8_t> sp(3 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 3, sp.data(), &why), RAD_OK);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 0), 10.0f);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 1), 11.0f);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 2), 20.0f);
    std::vector<float> f(128);
    REQUIRE_EQ(ckpt_rows_f32(w, 300, 1, f.data(), &why), RAD_OK);   /* u's row 44, u's scale */
    CHECK_EQ(f[5], ((float)((44 * 128 + 5) % 7) - 3.0f) * 20.0f);

    REQUIRE_EQ(ckpt_resolve(ck, prog, "bad", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("inside a scale block") != std::string::npos);
}

/* A join along dim 1 interleaves the pieces row by row. */
TEST(a_join_along_columns_interleaves_rows) {
    const std::string dir = write_checkpoint({ bf16("l", { 3, 2 }, 0.0f),
                                               bf16("r", { 3, 4 }, 100.0f) }, "{}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("lr", RAD_MAP_CONCAT, { "l", "r" }, {}, 1));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "lr", false, &w, &why), RAD_OK);
    CHECK_EQ(w.shape[1], (int64_t)6);
    std::vector<float> f(6);
    REQUIRE_EQ(ckpt_rows_f32(w, 1, 1, f.data(), &why), RAD_OK);
    const float want[6] = { 2, 3, 104, 105, 106, 107 };
    for (int j = 0; j < 6; ++j) CHECK_EQ(f[(size_t)j], want[j]);
    std::vector<uint8_t> raw(6 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 2, 1, raw.data(), &why), RAD_OK);
    CHECK_EQ(rad_load_f32(raw.data(), RAD_BF16, 0), 4.0f);
    CHECK_EQ(rad_load_f32(raw.data(), RAD_BF16, 2), 108.0f);
}

/* An expert is its own slice of a stacked tensor, never the whole stack, and a stacked FP8
 * tensor's scales are sliced at the same index. */
TEST(an_expert_is_its_slice_of_a_stack) {
    const std::string dir = write_checkpoint(
        { bf16("exps", { 4, 2, 3 }, 0.0f), fp8("q.weight", { 2, 128, 128 }),
          bf16("q.weight_scale_inv", { 2, 1, 1 }, 7.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("e2", RAD_MAP_COPY, { "exps" }, { 2 }));
    prog.name_map.push_back(map("q1", RAD_MAP_COPY, { "q.weight" }, { 1 }));
    prog.name_map.push_back(map("e9", RAD_MAP_COPY, { "exps" }, { 9 }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "e2", false, &w, &why), RAD_OK);
    CHECK_EQ(w.rank, 2u);
    std::vector<float> f(3);
    REQUIRE_EQ(ckpt_rows_f32(w, 1, 1, f.data(), &why), RAD_OK);
    CHECK_EQ(f[0], (float)(2 * 6 + 3));                       /* stack index 2, row 1, col 0 */
    REQUIRE_EQ(ckpt_resolve(ck, prog, "q1", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    std::vector<uint8_t> sp(2);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 1, sp.data(), &why), RAD_OK);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 0), 8.0f);    /* the second expert's scale */
    CHECK(ckpt_resolve(ck, prog, "e9", false, &w, &why) < 0);
}

/* Read sets use checkpoint tensor names: an expert slice "exps[2]" reads "exps", and a block-FP8
 * weight reads its scale plane as well as its codes, so a changed scale changes the weight. */
TEST(a_weight_reads_the_checkpoint_tensors_its_pieces_and_scales_come_from) {
    const std::string dir = write_checkpoint(
        { bf16("exps", { 4, 2, 3 }, 0.0f), fp8("q.weight", { 2, 128, 128 }),
          bf16("q.weight_scale_inv", { 2, 1, 1 }, 7.0f), bf16("a.weight", { 2, 3 }, 0.0f),
          bf16("b.weight", { 2, 3 }, 0.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("e2", RAD_MAP_COPY, { "exps" }, { 2 }));
    prog.name_map.push_back(map("q1", RAD_MAP_COPY, { "q.weight" }, { 1 }));
    prog.name_map.push_back(map("ab", RAD_MAP_CONCAT, { "a.weight", "b.weight" }));
    CkptWeight e2, q1, ab;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "e2", false, &e2, &why), RAD_OK);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "q1", false, &q1, &why), RAD_OK);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "ab", false, &ab, &why), RAD_OK);
    CHECK_EQ(ckpt_tensor_of(*e2.pieces[0]), std::string("exps"));

    std::unordered_set<std::string> hit;
    CHECK(ckpt_reads_any(e2, { "exps" }, &hit));
    CHECK(!ckpt_reads_any(e2, { "exps[2]" }));        /* a slice is not a checkpoint name */
    CHECK(ckpt_reads_any(q1, { "q.weight_scale_inv" }, &hit));
    CHECK(ckpt_reads_any(ab, { "b.weight" }, &hit));   /* one changed piece of a join */
    CHECK(!ckpt_reads_any(ab, { "q.weight", "exps" }));
    CHECK_EQ(hit.size(), (size_t)3);
}

/* A copy's sources are alternatives and the first present wins: a tied head reads the embedding
 * when the checkpoint ships no head. */
TEST(alternatives_take_the_first_tensor_present) {
    const std::string dir = write_checkpoint({ bf16("embed", { 2, 2 }, 5.0f) }, "{}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("out", RAD_MAP_COPY, { "lm_head", "embed" }));
    prog.name_map.push_back(map("need", RAD_MAP_COPY, { "missing" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "out", false, &w, &why), RAD_OK);
    REQUIRE_EQ(w.pieces.size(), (size_t)1);
    CHECK_EQ(w.pieces[0]->name, std::string("embed"));
    CHECK_EQ(ckpt_resolve(ck, prog, "need", false, &w, &why), RAD_E_NOTFOUND);
    CHECK(why.find("missing") != std::string::npos);
    CHECK_EQ(ckpt_resolve(ck, prog, "need", true, &w, &why), RAD_E_NOTFOUND);
    CHECK_EQ(ckpt_resolve(ck, prog, "unmapped", false, &w, &why), RAD_E_NOTFOUND);
}

/* Not trivial, with a reason: FP8 with no scale beside it, and pieces of different dtypes. */
TEST(what_is_not_served_as_it_is_says_why) {
    const std::string dir = write_checkpoint({ fp8("bare", { 128, 128 }),
                                               bf16("b16", { 128, 128 }, 0.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("bare", RAD_MAP_COPY, { "bare" }));
    prog.name_map.push_back(map("mixed", RAD_MAP_CONCAT, { "bare", "b16" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "bare", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("no scale plane") != std::string::npos);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "mixed", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
}

/* ================================================================== imported quantisations */
namespace {

/* A GPTQ-packed weight as auto_gptq's packer writes it: code (row n, column k) is the signed value
 * `q[n][k]`, stored unsigned with zero point 2^(bits-1) as field k % (32/bits) of int32 word
 * [k / (32/bits), n]; `qzeros` holds the v1 zero point (less one) in every field; `scales` is
 * f16 [K / group, N]. */
struct Gptq {
    std::vector<std::vector<int>> q;     /* [N][K] signed codes */
    std::vector<float> scale;            /* [G * N], row g of the scale tensor first */
    int64_t N = 0, K = 0, group = 0;
    int bits = 4;
};

Gptq make_gptq(int64_t N, int64_t K, int64_t group, int bits, int seed) {
    Gptq g;
    g.N = N; g.K = K; g.group = group; g.bits = bits;
    const int lo = -(1 << (bits - 1)), span = 1 << bits;
    g.q.assign((size_t)N, std::vector<int>((size_t)K));
    uint32_t x = 2463534242u ^ (uint32_t)seed;
    for (auto& row : g.q)
        for (int& v : row) {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            v = lo + (int)(x % (uint32_t)span);
        }
    /* f16 scales with low mantissa bits set and both signs, so bf16 rounding and sign are both
     * exercised. */
    for (int64_t i = 0; i < (K / group) * N; ++i)
        g.scale.push_back((i % 3 == 1 ? -1.0f : 1.0f) * (0.0123f + 0.00071f * (float)i));
    return g;
}

void add_gptq(std::vector<T>* ts, const std::string& base, const Gptq& g, bool with_gidx = false) {
    const int64_t per = 32 / g.bits, G = g.K / g.group;
    T qw{ base + ".qweight", "I32", { g.K / per, g.N }, {} };
    qw.bytes.assign((size_t)(g.K / per * g.N * 4), 0);
    for (int64_t n = 0; n < g.N; ++n)
        for (int64_t k = 0; k < g.K; ++k) {
            const uint32_t u = (uint32_t)(g.q[(size_t)n][(size_t)k] + (1 << (g.bits - 1)));
            uint32_t* w = (uint32_t*)qw.bytes.data() + (k / per) * g.N + n;
            *w |= u << ((k % per) * g.bits);
        }
    T sc{ base + ".scales", "F16", { G, g.N }, {} };
    sc.bytes.resize((size_t)(G * g.N * 2));
    for (int64_t i = 0; i < G * g.N; ++i) rad_store_f32(sc.bytes.data(), RAD_F16, i, g.scale[(size_t)i]);
    T qz{ base + ".qzeros", "I32", { G, g.N * g.bits / 32 }, {} };
    qz.bytes.resize((size_t)(G * g.N * g.bits / 32 * 4));
    for (size_t i = 0; i < qz.bytes.size() / 4; ++i)
        ((uint32_t*)qz.bytes.data())[i] = g.bits == 4 ? 0x77777777u : 0x7f7f7f7fu;
    ts->push_back(qw);
    ts->push_back(sc);
    ts->push_back(qz);
    if (with_gidx) {
        T gi{ base + ".g_idx", "I32", { g.K }, {} };
        gi.bytes.resize((size_t)g.K * 4);
        for (int64_t k = 0; k < g.K; ++k) ((int32_t*)gi.bytes.data())[k] = (int32_t)(k / g.group);
        ts->push_back(gi);
    }
}

/* `v` rounded through f16, as the scale tensor holds it. */
float f16_of(float v) {
    uint16_t h;
    rad_store_f32(&h, RAD_F16, 0, v);
    return rad_load_f32(&h, RAD_F16, 0);
}

const char* kAutoRound =
    "{\"model_type\":\"qwen4_exp\",\"quantization_config\":{\"quant_method\":\"auto-round\","
    "\"packing_format\":\"auto_round:auto_gptq\",\"bits\":4,\"group_size\":128,\"sym\":true}}";

}  /* namespace */

/* AutoRound's GPTQ export is imported: gate and up, each a GPTQ-packed int4 weight, become one
 * `i4*bf16[1x128]` weight with the shipped codes (signed, row by row) and the shipped f16 scales
 * rounded to bf16. Read as f32 it is the shipped value: the code times the f16 scale. */
TEST(an_autoround_int4_weight_is_imported_as_its_codes_and_scales) {
    const Gptq gate = make_gptq(8, 256, 128, 4, 1), up = make_gptq(16, 256, 128, 4, 2);
    std::vector<T> ts;
    add_gptq(&ts, "e.0.gate_proj", gate);
    add_gptq(&ts, "e.0.up_proj", up, /*with_gidx=*/true);
    const std::string dir = write_checkpoint(ts, kAutoRound);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::Gptq);
    CHECK_EQ(ck.meta().quant, std::string("auto-round"));
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "e.0.gate_proj.weight", "e.0.up_proj.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    REQUIRE(w.imported);
    CHECK_EQ(w.rows, (int64_t)24);
    CHECK_EQ(w.cols, (int64_t)256);
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_I4);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)1);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)128);
    CHECK(rad_enc_valid(&w.enc));

    /* every row, both pieces: codes, scales and the f32 value */
    std::vector<uint8_t> codes((size_t)(24 * 128)), scales((size_t)(24 * 2 * 2));
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 0, 24, codes.data(), &why), RAD_OK);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 24, scales.data(), &why), RAD_OK);
    std::vector<float> f((size_t)(24 * 256));
    REQUIRE_EQ(ckpt_rows_f32(w, 0, 24, f.data(), &why), RAD_OK);
    int bad_code = 0, bad_scale = 0, bad_f = 0;
    for (int64_t r = 0; r < 24; ++r) {
        const Gptq& g = r < 8 ? gate : up;
        const int64_t n = r < 8 ? r : r - 8;
        for (int64_t k = 0; k < 256; ++k) {
            const int q = g.q[(size_t)n][(size_t)k];
            if ((int)rad_load_f32(codes.data(), RAD_I4, r * 256 + k) != q) ++bad_code;
            const float s = f16_of(g.scale[(size_t)((k / 128) * g.N + n)]);
            if (f[(size_t)(r * 256 + k)] != (float)q * s) ++bad_f;
        }
        for (int64_t c = 0; c < 2; ++c) {
            const float s = f16_of(g.scale[(size_t)(c * g.N + n)]);
            if (rad_load_f32(scales.data(), RAD_BF16, r * 2 + c) !=
                rad_load_f32(std::vector<uint16_t>{ rad_f32_to_bf16(s) }.data(), RAD_BF16, 0))
                ++bad_scale;
        }
    }
    CHECK_EQ(bad_code, 0);
    CHECK_EQ(bad_scale, 0);
    CHECK_EQ(bad_f, 0);
    /* a part of the plane, from the middle of the second piece */
    std::vector<uint8_t> part(128);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 13, 1, part.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(part.data(), codes.data() + 13 * 128, 128), 0);
}

/* An int8 per-channel weight (one scale per row, g_idx all zero) is `i8*bf16[1x128]` with its
 * row's scale in every block. */
TEST(a_gptq_int8_per_channel_weight_repeats_its_scale) {
    const Gptq g = make_gptq(4, 384, 384, 8, 3);
    std::vector<T> ts;
    add_gptq(&ts, "o_proj", g, /*with_gidx=*/true);
    const std::string dir = write_checkpoint(ts, kAutoRound);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("o", RAD_MAP_COPY, { "o_proj.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "o", false, &w, &why), RAD_OK);
    REQUIRE(w.imported);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_I8);
    std::vector<uint8_t> codes(4 * 384), scales(4 * 3 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 0, 4, codes.data(), &why), RAD_OK);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 4, scales.data(), &why), RAD_OK);
    int bad = 0;
    for (int64_t n = 0; n < 4; ++n) {
        for (int64_t k = 0; k < 384; ++k)
            if ((int8_t)codes[(size_t)(n * 384 + k)] != g.q[(size_t)n][(size_t)k]) ++bad;
        for (int c = 0; c < 3; ++c)
            if (rad_load_f32(scales.data(), RAD_BF16, n * 3 + c) !=
                rad_load_f32(scales.data(), RAD_BF16, n * 3))
                ++bad;
    }
    CHECK_EQ(bad, 0);
}

/* An import refuses by name what it cannot read: an asymmetric zero point, an act-order group
 * index, a group narrower than the served 128, and an unknown packing. */
TEST(an_import_refuses_what_it_cannot_read) {
    auto gptq = [](int64_t group, int seed, bool with_gidx) {
        std::vector<T> ts;
        add_gptq(&ts, "x", make_gptq(8, 256, group, 4, seed), with_gidx);
        return ts;
    };
    auto refused = [](const std::vector<T>& ts, const char* expect) {
        const std::string dir = write_checkpoint(ts, kAutoRound);
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        Program prog;
        prog.name_map.push_back(map("x", RAD_MAP_COPY, { "x.weight" }));
        CkptWeight w;
        std::string why;
        CHECK(ckpt_resolve(ck, prog, "x", false, &w, &why) < 0);
        CHECK(why.find(expect) != std::string::npos);
    };
    std::vector<T> ts = gptq(128, 4, true);
    ts[2].bytes[0] = 0x76;
    refused(ts, "symmetric");
    ts = gptq(128, 4, true);
    ((int32_t*)ts[3].bytes.data())[0] = 1;
    refused(ts, "act-order");
    refused(gptq(64, 5, false), "multiple of 128");    /* four scale rows for 256 columns */
    /* AutoRound's own packing is a different layout, and the checkpoint is not imported. */
    {
        const std::string dir = write_checkpoint(
            { bf16("y", { 2, 2 }, 0) },
            "{\"model_type\":\"qwen4_exp\",\"quantization_config\":{\"quant_method\":"
            "\"auto-round\",\"packing_format\":\"auto_round\"}}");
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        CHECK(ck.imports() == Checkpoint::Import::None);
        CHECK(ck.import_refusal().find("auto_round:auto_gptq") != std::string::npos);
    }
}

/* Quark's MXFP4 is `fp4_e2m1*e8m0[1x32]` byte for byte, so trivial: the U8 codes and U8 exponents
 * are the planes, gate and up join along rows, and the f32 value is the e2m1 code times
 * 2^(e - 127). */
TEST(a_quark_mxfp4_weight_is_its_codes_and_exponents) {
    const int64_t K = 64;
    auto codes = [&](const std::string& n, int64_t N, int seed) {
        T t{ n, "U8", { N, K / 2 }, {} };
        for (int64_t i = 0; i < N * K / 2; ++i) t.bytes.push_back((uint8_t)((i * 37 + seed) & 0xff));
        return t;
    };
    auto exps = [&](const std::string& n, int64_t N, int seed) {
        T t{ n, "U8", { N, K / 32 }, {} };
        for (int64_t i = 0; i < N * K / 32; ++i) t.bytes.push_back((uint8_t)(120 + (i + seed) % 9));
        return t;
    };
    const T g = codes("g.weight", 2, 1), gs = exps("g.weight_scale", 2, 1);
    const T u = codes("u.weight", 3, 5), us = exps("u.weight_scale", 3, 4);
    const std::string dir = write_checkpoint(
        { g, gs, u, us, bf16("n.weight", { 4 }, 1.0f) },
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":\"quark\","
        "\"global_quant_config\":{\"weight\":{\"dtype\":\"fp4\",\"qscheme\":\"per_group\","
        "\"group_size\":32,\"scale_format\":\"e8m0\"}},\"layer_quant_config\":{}}}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::QuarkMx);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    prog.name_map.push_back(map("n", RAD_MAP_COPY, { "n.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    REQUIRE(w.trivial);
    CHECK(!w.imported);
    CHECK_EQ(w.rows, (int64_t)5);
    CHECK_EQ(w.cols, K);
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_FP4E2M1);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_E8M0);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)32);
    CHECK(rad_enc_valid(&w.enc));
    std::vector<uint8_t> cp(5 * K / 2), sp(5 * K / 32);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 0, 5, cp.data(), &why), RAD_OK);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 5, sp.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(cp.data(), g.bytes.data(), g.bytes.size()), 0);
    CHECK_EQ(std::memcmp(cp.data() + g.bytes.size(), u.bytes.data(), u.bytes.size()), 0);
    CHECK_EQ(std::memcmp(sp.data() + gs.bytes.size(), us.bytes.data(), us.bytes.size()), 0);
    std::vector<float> f((size_t)K);
    REQUIRE_EQ(ckpt_rows_f32(w, 3, 1, f.data(), &why), RAD_OK);       /* u's row 1 */
    int bad = 0;
    for (int64_t k = 0; k < K; ++k) {
        const uint8_t b = u.bytes[(size_t)(K / 2 + k / 2)];
        const float code = rad_fp4e2m1_to_f32((uint8_t)(k % 2 ? b >> 4 : b & 15));
        const float sc = std::ldexp(1.0f, (int)us.bytes[(size_t)(K / 32 + k / 32)] - 127);
        if (f[(size_t)k] != code * sc) ++bad;
    }
    CHECK_EQ(bad, 0);
    /* an excluded layer is a plain tensor */
    REQUIRE_EQ(ckpt_resolve(ck, prog, "n", false, &w, &why), RAD_OK);
    CHECK(w.trivial && rad_enc_is(&w.enc, "plain"));
}

/* GLM-5.3-Flash's Quark export keeps excluded attention and MTP weights block-FP8, with F32
 * `_scale` companions. They multiply the codes; reading the global MXFP4 scheme into these
 * weights, or overlooking the scale because it is not called `_scale_inv`, corrupts the model. */
TEST(quark_refuses_unsupported_or_incomplete_layer_weight_overrides) {
    for (const char* override_config : {
        "{\"p\":{\"weight\":{\"dtype\":\"fp8_e4m3\",\"qscheme\":\"per_block\",\"block_size\":[64,128]}}}",
        "{\"p\":{\"weight\":{\"dtype\":\"int8\",\"qscheme\":\"per_block\",\"block_size\":[128,128]}}}",
        "{\"p\":{\"weight\":null}}",
        "{\"p\":{\"weight\":{\"dtype\":\"fp8_e4m3\",\"qscheme\":\"per_block\",\"block_size\":[128,128]}},"
        "\"other\":{\"weight\":{\"qscheme\":\"per_block\"}}}" }) {
        const std::string config = std::string("{\"quantization_config\":{\"quant_method\":\"quark\","
            "\"global_quant_config\":{\"weight\":{\"dtype\":\"fp4\",\"qscheme\":\"per_group\","
            "\"group_size\":32,\"scale_format\":\"e8m0\"}},\"layer_quant_config\":") +
            override_config + "}}";
        const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) }, config.c_str());
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        CHECK(ck.imports() != Checkpoint::Import::QuarkMx);
        CHECK(!ck.import_refusal().empty());
    }
}

TEST(a_quark_checkpoint_preserves_excluded_block_fp8_weights) {
    T s{ "p.weight_scale", "F32", { 2, 2 }, std::vector<uint8_t>(4 * sizeof(float)) };
    for (int i = 0; i < 4; ++i) rad_store_f32(s.bytes.data(), RAD_F32, i, 0.25f * (i + 1));
    T bad = s;
    bad.name = "bad.weight_scale";
    bad.shape = { 1, 4 };  /* same number of scales, wrong block geometry */
    const std::string dir = write_checkpoint(
        { fp8("p.weight", { 256, 256 }), s, fp8("bad.weight", { 256, 256 }), bad,
          bf16("norm.weight", { 4 }, 1.0f) },
        "{\"model_type\":\"glm5_next\",\"quantization_config\":{\"quant_method\":\"quark\","
        "\"global_quant_config\":{\"weight\":{\"dtype\":\"fp4\",\"qscheme\":\"per_group\","
        "\"group_size\":32,\"scale_format\":\"e8m0\"}},"
        "\"layer_quant_config\":{\"p\":{\"weight\":{\"dtype\":\"fp8_e4m3\","
        "\"qscheme\":\"per_block\",\"block_size\":[128,128]}}}}}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    REQUIRE(ck.imports() == Checkpoint::Import::QuarkMx);
    Program prog;
    prog.name_map.push_back(map("p", RAD_MAP_COPY, { "p.weight" }));
    prog.name_map.push_back(map("bad", RAD_MAP_COPY, { "bad.weight" }));
    prog.name_map.push_back(map("norm", RAD_MAP_COPY, { "norm.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "p", false, &w, &why), RAD_OK);
    REQUIRE(w.trivial);
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_F8E4M3);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_F32);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)128);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)128);
    std::vector<float> f(256);
    for (int64_t r : { (int64_t)0, (int64_t)129 }) {
        REQUIRE_EQ(ckpt_rows_f32(w, r, 1, f.data(), &why), RAD_OK);
        for (int64_t c : { (int64_t)0, (int64_t)127, (int64_t)128, (int64_t)255 })
            CHECK_EQ(f[(size_t)c], ((float)((r * 256 + c) % 7) - 3.0f) *
                                   (0.25f * (float)(r / 128 * 2 + c / 128 + 1)));
    }
    std::vector<uint8_t> scales(s.bytes.size());
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 2, scales.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(scales.data(), s.bytes.data(), scales.size()), 0);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "bad", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("scale plane") != std::string::npos);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "norm", false, &w, &why), RAD_OK);
    CHECK(w.trivial && rad_enc_is(&w.enc, "plain"));
}

/* A directory merging two checkpoints (a finetune's changed tensors beside links to every base
 * shard) holds some tensors in two shards. The index decides which copy is read; a name in two
 * shards that the index does not settle is refused, not read from whichever shard sorts first. */
TEST(the_index_decides_which_shard_a_tensor_is_read_from) {
    const std::string dir = fresh_dir();
    const T base_x = bf16("x.weight", { 2, 4 }, 1.0f), base_y = bf16("y.weight", { 4 }, 2.0f);
    const T ft_x = bf16("x.weight", { 2, 4 }, 5.0f);
    write_shard(dir + "/a.safetensors", { base_x, base_y });   /* sorts first */
    write_shard(dir + "/b.safetensors", { ft_x });
    write_file(dir + "/config.json", kConfig);
    write_file(dir + "/model.safetensors.index.json",
               R"({"weight_map":{"x.weight":"b.safetensors","y.weight":"a.safetensors"}})");
    {
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        const SrcTensor* x = ck.find("x.weight");
        const SrcTensor* y = ck.find("y.weight");
        REQUIRE(x && y);
        CHECK_EQ(std::memcmp(x->data, ft_x.bytes.data(), ft_x.bytes.size()), 0);
        CHECK_EQ(std::memcmp(y->data, base_y.bytes.data(), base_y.bytes.size()), 0);
    }
    /* the index names both shards but not x: two copies and no way to choose */
    write_file(dir + "/model.safetensors.index.json",
               R"({"weight_map":{"y.weight":"a.safetensors","z.weight":"b.safetensors"}})");
    {
        Checkpoint ck;
        CHECK_EQ(ck.open(dir), RAD_E_FORMAT);
    }
}

/* ================================================================== CT-INT4 */
namespace {

/* A compressed-tensors pack-quantized int4 weight as llm-compressor writes it: code (row n,
 * column k) is the signed value q[n][k], stored as q + 8 in nibble k % 8 of int32 word [n, k / 8];
 * `X.weight_scale` is bf16 [N, K / group]. */
struct Ct {
    std::vector<std::vector<int>> q;   /* [N][K] signed codes, -8..7 */
    std::vector<float> scale;          /* [N * K / group] */
    int64_t N = 0, K = 0, group = 0;
};

Ct make_ct(int64_t N, int64_t K, int64_t group, int seed) {
    Ct c;
    c.N = N; c.K = K; c.group = group;
    c.q.assign((size_t)N, std::vector<int>((size_t)K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            c.q[(size_t)n][(size_t)k] = (int)((n * 7 + k * 3 + seed * 5 + (k * k) % 11) % 16) - 8;
    for (int64_t i = 0; i < N * (K / group); ++i)
        c.scale.push_back(rad_load_f32(std::vector<uint16_t>{ rad_f32_to_bf16(
                              0.01f * (float)(1 + (i * 13 + seed) % 29)) }.data(), RAD_BF16, 0));
    return c;
}

void add_ct(std::vector<T>* ts, const std::string& base, const Ct& c) {
    T p{ base + ".weight_packed", "I32", { c.N, c.K / 8 }, {} };
    p.bytes.assign((size_t)(c.N * c.K / 2), 0);
    for (int64_t n = 0; n < c.N; ++n)
        for (int64_t k = 0; k < c.K; ++k) {
            const uint32_t u = (uint32_t)(c.q[(size_t)n][(size_t)k] + 8);
            uint32_t w;
            std::memcpy(&w, p.bytes.data() + (n * (c.K / 8) + k / 8) * 4, 4);
            w |= u << (4 * (k % 8));
            std::memcpy(p.bytes.data() + (n * (c.K / 8) + k / 8) * 4, &w, 4);
        }
    T s{ base + ".weight_scale", "BF16", { c.N, c.K / c.group }, {} };
    s.bytes.resize((size_t)(c.N * (c.K / c.group) * 2));
    for (size_t i = 0; i < c.scale.size(); ++i) rad_store_f32(s.bytes.data(), RAD_BF16, (int64_t)i, c.scale[i]);
    ts->push_back(p);
    ts->push_back(s);
}

std::string ct_config(const char* symmetric, int group) {
    return std::string("{\"model_type\":\"qwen4_exp\",\"quantization_config\":{\"quant_method\":"
                       "\"compressed-tensors\",\"format\":\"pack-quantized\",\"config_groups\":"
                       "{\"group_0\":{\"format\":\"pack-quantized\",\"targets\":[\"Linear\"],"
                       "\"weights\":{\"num_bits\":4,\"type\":\"int\",\"symmetric\":") + symmetric +
           ",\"strategy\":\"group\",\"group_size\":" + std::to_string(group) +
           ",\"dynamic\":false,\"actorder\":null}}}}}";
}

}  /* namespace */

/* CT-INT4 is imported: gate and up, each a packed weight, become one `i4*bf16[1x32]` weight with
 * the shipped codes made signed and the shipped scales. Read as f32 it is (code - 8) times its
 * group's scale; codes below 8 are negative weights, so the subtraction must be signed. */
TEST(a_ct_int4_weight_is_imported_as_its_signed_codes_and_scales) {
    const Ct gate = make_ct(4, 64, 32, 1), up = make_ct(6, 64, 32, 2);
    std::vector<T> ts;
    add_ct(&ts, "e.0.gate_proj", gate);
    add_ct(&ts, "e.0.up_proj", up);
    T plain = bf16("e.0.norm.weight", { 64 }, 1.0f);   /* an ignored layer is an ordinary tensor */
    ts.push_back(plain);
    const std::string dir = write_checkpoint(ts, ct_config("true", 32).c_str());
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::CtInt4);
    CHECK(ck.import_refusal().empty());
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "e.0.gate_proj.weight", "e.0.up_proj.weight" }));
    prog.name_map.push_back(map("norm", RAD_MAP_COPY, { "e.0.norm.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    REQUIRE(w.imported);
    CHECK_EQ(w.rows, (int64_t)10);
    CHECK_EQ(w.cols, (int64_t)64);
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_I4);
    CHECK_EQ(w.enc.plane[0].block[1], (int64_t)1);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)32);
    CHECK(rad_enc_valid(&w.enc));

    std::vector<uint8_t> codes((size_t)(10 * 32)), scales((size_t)(10 * 2 * 2));
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 0, 10, codes.data(), &why), RAD_OK);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 10, scales.data(), &why), RAD_OK);
    std::vector<float> f((size_t)(10 * 64));
    REQUIRE_EQ(ckpt_rows_f32(w, 0, 10, f.data(), &why), RAD_OK);
    int bad_code = 0, bad_scale = 0, bad_f = 0, negative = 0;
    for (int64_t r = 0; r < 10; ++r) {
        const Ct& c = r < 4 ? gate : up;
        const int64_t n = r < 4 ? r : r - 4;
        for (int64_t k = 0; k < 64; ++k) {
            const int q = c.q[(size_t)n][(size_t)k];
            negative += q < 0;
            if ((int)rad_load_f32(codes.data(), RAD_I4, r * 64 + k) != q) ++bad_code;
            if (f[(size_t)(r * 64 + k)] != (float)q * c.scale[(size_t)(n * 2 + k / 32)]) ++bad_f;
        }
        for (int64_t g = 0; g < 2; ++g)
            if (rad_load_f32(scales.data(), RAD_BF16, r * 2 + g) != c.scale[(size_t)(n * 2 + g)])
                ++bad_scale;
    }
    CHECK(negative > 0);
    CHECK_EQ(bad_code, 0);
    CHECK_EQ(bad_scale, 0);
    CHECK_EQ(bad_f, 0);

    REQUIRE_EQ(ckpt_resolve(ck, prog, "norm", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    CHECK(!w.imported);
}

/* What is not read is refused, not misread: a weight with no zero point under an asymmetric
 * config, a zero point beside a weight the config calls symmetric, and a scale plane that is not
 * one per group. */
TEST(a_ct_int4_checkpoint_this_does_not_read_is_refused) {
    const Ct c = make_ct(4, 64, 32, 3);
    {
        std::vector<T> ts;
        add_ct(&ts, "p", c);
        const std::string dir = write_checkpoint(ts, ct_config("false", 32).c_str());
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        CHECK(ck.imports() == Checkpoint::Import::CtInt4);
        Program prog;
        prog.name_map.push_back(map("p", RAD_MAP_COPY, { "p.weight" }));
        CkptWeight w;
        std::string why;
        CHECK(ckpt_resolve(ck, prog, "p", false, &w, &why) != RAD_OK);
        CHECK(why.find("p.weight_zero_point") != std::string::npos);
    }
    {
        std::vector<T> ts;
        add_ct(&ts, "p", c);
        ts.push_back(T{ "p.weight_zero_point", "I32", { 4, 1 }, std::vector<uint8_t>(16, 0) });
        const std::string dir = write_checkpoint(ts, ct_config("true", 32).c_str());
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        Program prog;
        prog.name_map.push_back(map("p", RAD_MAP_COPY, { "p.weight" }));
        CkptWeight w;
        std::string why;
        CHECK(ckpt_resolve(ck, prog, "p", false, &w, &why) != RAD_OK);
        CHECK(why.find("zero point") != std::string::npos);
    }
    {
        std::vector<T> ts;
        add_ct(&ts, "p", c);   /* scales per 32, the config says 16 */
        const std::string dir = write_checkpoint(ts, ct_config("true", 16).c_str());
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        Program prog;
        prog.name_map.push_back(map("p", RAD_MAP_COPY, { "p.weight" }));
        CkptWeight w;
        std::string why;
        CHECK(ckpt_resolve(ck, prog, "p", false, &w, &why) != RAD_OK);
        CHECK(why.find("weight_scale") != std::string::npos);
    }
}

namespace {

/* Every plane of `w`, read whole, with 8 zeroed bytes of slack past each for reads of a bit
 * stream's last word. Each read is checked; empty when one fails. */
std::vector<std::vector<uint8_t>> read_planes(const CkptWeight& w) {
    std::vector<std::vector<uint8_t>> pl((size_t)w.enc.n_planes);
    std::string why;
    for (int k = 0; k < w.enc.n_planes; ++k) {
        int64_t pr = 0, pc = 0;
        rad_enc_plane_dims(&w.enc.plane[k], w.rows, w.cols, &pr, &pc);
        pl[(size_t)k].assign((size_t)(pr * rad_enc_row_bytes(w.enc.plane[k].dtype, pc)) + 8, 0);
        const int st = ckpt_plane_rows(w, k, 0, pr, pl[(size_t)k].data(), &why);
        CHECK_EQ(st, RAD_OK);
        if (st != RAD_OK) return {};
    }
    return pl;
}

/* AWQ's int32 packing of 4-bit values along N: v[n * rows + r] is element (r, n) of [rows, N],
 * stored in word [r, n / 8] at nibble j where kOrder[j] == n % 8. */
T pack_awq(const std::string& name, const std::vector<int>& v, int64_t rows, int64_t N) {
    static const int kOrder[8] = { 0, 2, 4, 6, 1, 3, 5, 7 };
    T t{ name, "I32", { rows, N / 8 }, {} };
    t.bytes.assign((size_t)(rows * N / 8 * 4), 0);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t n = 0; n < N; ++n) {
            int j = 0;
            while (kOrder[j] != n % 8) ++j;
            uint32_t w;
            uint8_t* p = t.bytes.data() + (r * (N / 8) + n / 8) * 4;
            std::memcpy(&w, p, 4);
            w |= (uint32_t)(v[(size_t)(n * rows + r)] & 15) << (4 * j);
            std::memcpy(p, &w, 4);
        }
    return t;
}

}  /* namespace */

namespace {

/* An asymmetric CT-INT4 weight as compressed-tensors' pack-quantized writes it (W4A16_ASYM, e.g.
 * llm-compressor's AWQ): the signed code q[n][k] stored as q + 8 in X.weight_packed as above, the
 * signed zero point zp[n][g] stored as zp + 8 in nibble n % 8 of int32 word [n / 8, g] of
 * X.weight_zero_point (packed along N, the last word's spare nibbles zero), and a bf16 scale per
 * group. compressed-tensors dequantises (q - zp) * scale. */
struct CtAsym {
    Ct c;                                  /* codes and scales */
    std::vector<std::vector<int>> zp;      /* [N][G] signed, -8..7 */
};

CtAsym make_ct_asym(int64_t N, int64_t K, int64_t group, int seed) {
    CtAsym a;
    a.c = make_ct(N, K, group, seed);
    a.zp.assign((size_t)N, std::vector<int>((size_t)(K / group)));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < K / group; ++g)
            a.zp[(size_t)n][(size_t)g] = (int)((n * 5 + g * 3 + seed) % 16) - 8;
    return a;
}

void add_ct_asym(std::vector<T>* ts, const std::string& base, const CtAsym& a) {
    add_ct(ts, base, a.c);
    const int64_t N = a.c.N, G = a.c.K / a.c.group;
    T z{ base + ".weight_zero_point", "I32", { (N + 7) / 8, G }, {} };
    z.bytes.assign((size_t)((N + 7) / 8 * G * 4), 0);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < G; ++g) {
            uint32_t w;
            std::memcpy(&w, z.bytes.data() + ((n / 8) * G + g) * 4, 4);
            w |= (uint32_t)(a.zp[(size_t)n][(size_t)g] + 8) << (4 * (n % 8));
            std::memcpy(z.bytes.data() + ((n / 8) * G + g) * 4, &w, 4);
        }
    ts->push_back(z);
}

std::string ct_config_ao(const char* symmetric, const char* actorder) {
    std::string s = ct_config(symmetric, 32);
    const std::string from = "\"actorder\":null";
    s.replace(s.find(from), from.size(), std::string("\"actorder\":") + actorder);
    return s;
}

/* `w` decoded from the planes ckpt_plane_rows writes, through the core's decoder. */
std::vector<float> decode_planes(const CkptWeight& w) {
    std::vector<std::vector<uint8_t>> pl((size_t)w.enc.n_planes);
    std::vector<const void*> data;
    std::string why;
    for (int k = 0; k < w.enc.n_planes; ++k) {
        int64_t pr = 0, pc = 0;
        rad_enc_plane_dims(&w.enc.plane[k], w.rows, w.cols, &pr, &pc);
        pl[(size_t)k].assign((size_t)(pr * rad_enc_row_bytes(w.enc.plane[k].dtype, pc)) + 8, 0);
        if (ckpt_plane_rows(w, k, 0, pr, pl[(size_t)k].data(), &why) != RAD_OK) return {};
        data.push_back(pl[(size_t)k].data());
    }
    EncodedView v;
    enc_view(w.enc, w.rows, w.cols, data.data(), &v);
    std::vector<float> out((size_t)(w.rows * w.cols));
    if (enc_decode_rows(v, 0, w.rows, out.data(), &why) != RAD_OK) return {};
    return out;
}

}  /* namespace */

/* An asymmetric CT-INT4 weight imports as `u4*bf16[1x32]-u4[1x32]`: the stored nibbles are the
 * codes and the stored zero points the zero plane, so (u - z) * s is compressed-tensors'
 * (q - zp) * scale bit for bit. Gate (twelve rows: the zero points' last word half full) and up
 * join along rows. */
TEST(an_asymmetric_ct_int4_weight_is_its_codes_zero_points_and_scales) {
    const CtAsym gate = make_ct_asym(12, 64, 32, 1), up = make_ct_asym(8, 64, 32, 2);
    std::vector<T> ts;
    add_ct_asym(&ts, "g", gate);
    add_ct_asym(&ts, "u", up);
    const std::string dir = write_checkpoint(ts, ct_config("false", 32).c_str());
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::CtInt4);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    REQUIRE(w.imported);
    REQUIRE_EQ(w.enc.n_planes, 3);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_U4);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(std::string(w.enc.plane[2].role), std::string("zero"));
    CHECK_EQ(w.enc.plane[2].dtype, (uint32_t)RAD_U4);
    CHECK_EQ(w.enc.plane[2].block[1], (int64_t)32);
    CHECK(rad_enc_valid(&w.enc));
    const auto pl = read_planes(w);
    REQUIRE_EQ(pl.size(), (size_t)3);
    std::vector<float> f((size_t)(20 * 64));
    REQUIRE_EQ(ckpt_rows_f32(w, 0, 20, f.data(), &why), RAD_OK);
    const std::vector<float> dec = decode_planes(w);
    REQUIRE_EQ(dec.size(), f.size());
    int bad_code = 0, bad_zero = 0, bad_f = 0, bad_dec = 0, negative = 0;
    for (int64_t r = 0; r < 20; ++r) {
        const CtAsym& a = r < 12 ? gate : up;
        const int64_t n = r < 12 ? r : r - 12;
        for (int64_t k = 0; k < 64; ++k) {
            const int q = a.c.q[(size_t)n][(size_t)k], z = a.zp[(size_t)n][(size_t)(k / 32)];
            const float want = (float)(q - z) * a.c.scale[(size_t)(n * 2 + k / 32)];
            negative += want < 0;
            if ((int)rad_load_f32(pl[0].data() + r * 32, RAD_U4, k) != q + 8) ++bad_code;
            if (f[(size_t)(r * 64 + k)] != want) ++bad_f;
            if (dec[(size_t)(r * 64 + k)] != want) ++bad_dec;
        }
        for (int64_t g = 0; g < 2; ++g)
            if ((int)rad_load_f32(pl[2].data() + r, RAD_U4, g) != a.zp[(size_t)n][(size_t)g] + 8)
                ++bad_zero;
    }
    CHECK(negative > 0);
    CHECK_EQ(bad_code, 0);
    CHECK_EQ(bad_zero, 0);
    CHECK_EQ(bad_f, 0);
    CHECK_EQ(bad_dec, 0);
}

/* Act-order `static` (alias `weight`) quantises in activation order but keeps each group's
 * columns together, so it reads as no act-order, and an in-order weight_g_idx is accepted;
 * `group` (alias `dynamic`) reorders the groups and is refused by name, as is a g_idx that moves
 * a column out of its group. */
TEST(ct_int4_act_order_static_reads_in_order_and_group_is_refused) {
    const Ct c = make_ct(4, 64, 32, 4);
    for (const char* ao : { "\"static\"", "\"weight\"" }) {
        std::vector<T> ts;
        add_ct(&ts, "p", c);
        const std::string dir = write_checkpoint(ts, ct_config_ao("true", ao));
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        CHECK(ck.imports() == Checkpoint::Import::CtInt4);
    }
    for (const char* ao : { "\"group\"", "\"dynamic\"" }) {
        std::vector<T> ts;
        add_ct(&ts, "p", c);
        const std::string dir = write_checkpoint(ts, ct_config_ao("true", ao));
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        CHECK(ck.imports() == Checkpoint::Import::None);
        CHECK(ck.import_refusal().find("act-order") != std::string::npos);
        CHECK(ck.import_refusal().find(std::string(ao).substr(1, std::strlen(ao) - 2)) !=
              std::string::npos);
    }
    for (int moved : { 0, 1 }) {
        std::vector<T> ts;
        add_ct(&ts, "p", c);
        T gi{ "p.weight_g_idx", "I32", { 64 }, std::vector<uint8_t>(256, 0) };
        for (int k = 0; k < 64; ++k) ((int32_t*)gi.bytes.data())[k] = k / 32;
        if (moved) ((int32_t*)gi.bytes.data())[3] = 1;
        ts.push_back(gi);
        const std::string dir = write_checkpoint(ts, ct_config_ao("true", "\"static\""));
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        Program prog;
        prog.name_map.push_back(map("p", RAD_MAP_COPY, { "p.weight" }));
        CkptWeight w;
        std::string why;
        const int rc = ckpt_resolve(ck, prog, "p", false, &w, &why);
        if (moved) {
            CHECK(rc != RAD_OK);
            CHECK(why.find("act-order") != std::string::npos);
        } else {
            CHECK_EQ(rc, RAD_OK);
        }
    }
}

/* An import rule writes a symmetric grid unsigned under a zero of 8 and its bf16 scales as f16:
 * every plane decodes, through the core's decoder, to the shipped value bit for bit, and with
 * every scale in f16's range nothing is counted as rounded. An asymmetric grid keeps its zero
 * points and changes only the scale's dtype. */
TEST(an_import_rule_re_expresses_an_int4_grid_exactly) {
    const Ct sym = make_ct(6, 64, 32, 5);
    const CtAsym asym = make_ct_asym(9, 64, 32, 6);
    ImportForm f;
    f.codes = RAD_U4;
    f.zero = RAD_U4;
    f.scale = RAD_F16;
    for (int is_asym : { 0, 1 }) {
        std::vector<T> ts;
        if (is_asym) add_ct_asym(&ts, "p", asym);
        else add_ct(&ts, "p", sym);
        const std::string dir = write_checkpoint(ts, ct_config(is_asym ? "false" : "true", 32));
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        Program prog;
        prog.name_map.push_back(map("p", RAD_MAP_COPY, { "p.weight" }));
        CkptWeight w;
        std::string why;
        REQUIRE_EQ(ckpt_resolve(ck, prog, "p", false, &w, &why), RAD_OK);
        std::vector<float> want((size_t)(w.rows * w.cols));
        REQUIRE_EQ(ckpt_rows_f32(w, 0, w.rows, want.data(), &why), RAD_OK);
        REQUIRE_EQ(ckpt_import_form(&w, f, &why), RAD_OK);
        char en[96];
        rad_enc_format(&w.enc, en, sizeof en);
        CHECK_EQ(std::string(en), std::string("u4*f16[1x32]-u4[1x32]"));
        CHECK(w.imported && !w.trivial);
        CHECK_EQ(w.rounded, (int64_t)0);
        const std::vector<float> dec = decode_planes(w);
        REQUIRE_EQ(dec.size(), want.size());
        int bad = 0;
        for (size_t i = 0; i < dec.size(); ++i) bad += dec[i] != want[i];
        CHECK_EQ(bad, 0);
        /* a symmetric grid's zero plane is 8 throughout */
        if (!is_asym) {
            const auto pl = read_planes(w);
            REQUIRE_EQ(pl.size(), (size_t)3);
            for (int64_t r = 0; r < w.rows; ++r)
                for (int64_t g = 0; g < 2; ++g)
                    CHECK_EQ((int)rad_load_f32(pl[2].data() + r, RAD_U4, g), 8);
        }
    }
}

/* AutoRound's GPTQ export with an import rule: the codes unsigned under a zero of 8 and the
 * scales kept in their shipped f16, where the import alone rounds them to bf16 for the expert
 * GEMM. The decode is then the shipped value, code times f16 scale, bit for bit. */
TEST(an_import_rule_keeps_gptq_scales_in_their_shipped_f16) {
    const Gptq g = make_gptq(8, 256, 128, 4, 7);
    std::vector<T> ts;
    add_gptq(&ts, "x", g);
    const std::string dir = write_checkpoint(ts, kAutoRound);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("x", RAD_MAP_COPY, { "x.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "x", false, &w, &why), RAD_OK);
    std::vector<float> want((size_t)(8 * 256));
    REQUIRE_EQ(ckpt_rows_f32(w, 0, 8, want.data(), &why), RAD_OK);
    const std::vector<float> as_bf16 = decode_planes(w);
    int differ = 0;
    for (size_t i = 0; i < want.size(); ++i) differ += as_bf16[i] != want[i];
    CHECK(differ > 0);                       /* the default import rounds the scales */
    ImportForm f;
    f.codes = RAD_U4;
    f.zero = RAD_U4;
    f.scale = RAD_F16;
    REQUIRE_EQ(ckpt_import_form(&w, f, &why), RAD_OK);
    char en[96];
    rad_enc_format(&w.enc, en, sizeof en);
    CHECK_EQ(std::string(en), std::string("u4*f16[1x128]-u4[1x128]"));
    CHECK_EQ(w.rounded, (int64_t)0);
    const std::vector<float> dec = decode_planes(w);
    REQUIRE_EQ(dec.size(), want.size());
    int bad = 0;
    for (size_t i = 0; i < dec.size(); ++i) bad += dec[i] != want[i];
    CHECK_EQ(bad, 0);
}

/* A shipped scale the new dtype cannot hold (a bf16 scale below f16's normal range) is rounded,
 * counted, and its largest relative change reported; what the rule cannot express is refused by
 * name, and a plain weight is declined. */
TEST(an_import_rule_counts_a_rounded_scale_and_refuses_what_it_cannot_express) {
    Ct c = make_ct(2, 64, 32, 8);
    c.scale[1] = rad_load_f32(std::vector<uint16_t>{ rad_f32_to_bf16(1.7e-6f) }.data(), RAD_BF16, 0);
    std::vector<T> ts;
    add_ct(&ts, "p", c);
    ts.push_back(bf16("n.weight", { 64 }, 1.0f));
    const std::string dir = write_checkpoint(ts, ct_config("true", 32));
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("p", RAD_MAP_COPY, { "p.weight" }));
    prog.name_map.push_back(map("n", RAD_MAP_COPY, { "n.weight" }));
    CkptWeight w;
    std::string why;
    ImportForm f;
    f.codes = RAD_U4;
    f.zero = RAD_U4;
    f.scale = RAD_F16;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "p", false, &w, &why), RAD_OK);
    REQUIRE_EQ(ckpt_import_form(&w, f, &why), RAD_OK);
    CHECK_EQ(w.rounded, (int64_t)1);
    CHECK(w.max_rel > 0.0 && w.max_rel < 0.05);

    ImportForm bad = f;
    bad.codes = RAD_U8;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "p", false, &w, &why), RAD_OK);
    CHECK_EQ(ckpt_import_form(&w, bad, &why), (int)RAD_E_INVAL);
    CHECK(why.find("codes=u8") != std::string::npos);
    ImportForm no_zero;
    no_zero.codes = RAD_U4;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "p", false, &w, &why), RAD_OK);
    CHECK_EQ(ckpt_import_form(&w, no_zero, &why), (int)RAD_E_INVAL);
    CHECK(why.find("zero=") != std::string::npos);
    ImportForm blk;
    blk.block[0] = blk.block[1] = 128;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "p", false, &w, &why), RAD_OK);
    CHECK_EQ(ckpt_import_form(&w, blk, &why), (int)RAD_E_INVAL);
    CHECK(why.find("block=") != std::string::npos);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "n", false, &w, &why), RAD_OK);
    CHECK_EQ(ckpt_import_form(&w, f, &why), (int)RAD_E_UNSUPPORTED);
}


/* ParoQuant MXFP4 is Quark's codes and exponents in a rotated basis plus the rotations: each
 * projection's pairs (group-local int16), theta and channel scales. Gate and up are two parts of
 * the fused weight, so the tables carry whole-row pairs, widened angles, inverted channel scales
 * and the part ends. The core's decode, rotated forward again, must give the shipped codes. */
TEST(a_paroquant_mxfp4_weight_carries_its_rotations_as_parts) {
    const int64_t K = 128, S = 2;
    uint32_t rng = 12345;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng >> 8; };
    auto codes = [&](const std::string& n, int64_t N) {
        T t{ n, "U8", { N, K / 2 }, {} };
        for (int64_t i = 0; i < N * K / 2; ++i) t.bytes.push_back((uint8_t)(next() & 0xff));
        return t;
    };
    auto exps = [&](const std::string& n, int64_t N) {
        T t{ n, "U8", { N, K / 32 }, {} };
        for (int64_t i = 0; i < N * K / 32; ++i) t.bytes.push_back((uint8_t)(122 + next() % 5));
        return t;
    };
    auto pairs = [&](const std::string& n) {
        T t{ n, "I16", { S, K }, {} };
        for (int64_t s = 0; s < S; ++s) {
            std::vector<int16_t> perm((size_t)K);
            for (int64_t i = 0; i < K; ++i) perm[(size_t)i] = (int16_t)i;
            for (int64_t i = K - 1; i > 0; --i) std::swap(perm[(size_t)i], perm[next() % (i + 1)]);
            for (int64_t i = 0; i < K; ++i) {
                t.bytes.push_back((uint8_t)(perm[(size_t)i] & 0xff));
                t.bytes.push_back((uint8_t)(perm[(size_t)i] >> 8));
            }
        }
        return t;
    };
    auto f16s = [&](const std::string& n, std::vector<int64_t> shape, float lo, float hi) {
        T t{ n, "F16", shape, {} };
        int64_t cnt = 1;
        for (int64_t d : shape) cnt *= d;
        for (int64_t i = 0; i < cnt; ++i) {
            const float f = lo + (hi - lo) * (float)(next() % 1000) / 1000.0f;
            const uint16_t h = rad_f32_to_f16(f);
            t.bytes.push_back((uint8_t)(h & 0xff));
            t.bytes.push_back((uint8_t)(h >> 8));
        }
        return t;
    };
    const T gw = codes("g.weight", 16), gs = exps("g.weight_scale", 16), gp = pairs("g.pairs"),
            gt = f16s("g.theta", { S, K / 2 }, -3.0f, 3.0f), gc = f16s("g.channel_scales", { 1, K }, 0.5f, 2.0f);
    const T uw = codes("u.weight", 16), us = exps("u.weight_scale", 16), up = pairs("u.pairs"),
            ut = f16s("u.theta", { S, K / 2 }, -3.0f, 3.0f), uc = f16s("u.channel_scales", { 1, K }, 0.5f, 2.0f);
    // a per-row fp8 head beside them: weight_scale is one f32 per row
    T hs{ "h.weight_scale", "F32", { 4 }, {} };
    for (int i = 0; i < 4; ++i) { float f = 0.25f * (float)(i + 1); hs.bytes.resize(hs.bytes.size() + 4); std::memcpy(hs.bytes.data() + 4 * i, &f, 4); }
    const T hw = fp8("h.weight", { 4, 8 });
    const std::string dir = write_checkpoint(
        { gw, gs, gp, gt, gc, uw, us, up, ut, uc, hw, hs },
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":\"paroquant_mxfp4\","
        "\"format\":\"mxfp4\",\"bits\":4,\"group_size\":128,\"mxfp4_block\":32,\"krot\":2}}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::ParoMx);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    prog.name_map.push_back(map("h", RAD_MAP_COPY, { "h.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    REQUIRE(w.paro);
    CHECK(w.imported && !w.trivial);
    REQUIRE(rad_enc_valid(&w.enc));
    CHECK_EQ(std::string(w.enc.transform), std::string("givens"));
    CHECK_EQ(rad_enc_givens_parts(&w.enc), (int64_t)2);
    CHECK_EQ(w.rows, (int64_t)32);
    REQUIRE_EQ(w.enc.n_planes, 7);

    const std::vector<std::vector<uint8_t>> pl = read_planes(w);
    REQUIRE(!pl.empty());
    // codes and exponents are the checkpoint's bytes, joined
    CHECK_EQ(std::memcmp(pl[0].data(), gw.bytes.data(), gw.bytes.size()), 0);
    CHECK_EQ(std::memcmp(pl[0].data() + gw.bytes.size(), uw.bytes.data(), uw.bytes.size()), 0);
    CHECK_EQ(std::memcmp(pl[1].data() + gs.bytes.size(), us.bytes.data(), us.bytes.size()), 0);
    const int32_t* tp = (const int32_t*)pl[2].data();
    const float* ta = (const float*)pl[3].data();
    const float* tsc = (const float*)pl[4].data();
    const int32_t* tpart = (const int32_t*)pl[5].data();
    CHECK_EQ(tpart[0], 16);
    CHECK_EQ(tpart[1], 32);
    for (int64_t c = 0; c < K; ++c) {
        CHECK_EQ(tp[c], (int32_t)(*(const int16_t*)(gp.bytes.data() + 2 * c)));             // part 0, stage 0
        CHECK_EQ(tp[(S + 1) * K + c], (int32_t)(*(const int16_t*)(up.bytes.data() + 2 * (K + c))));  // part 1, stage 1
        const uint16_t h = *(const uint16_t*)(gc.bytes.data() + 2 * c);
        CHECK(std::fabs(tsc[c] - 1.0f / rad_f16_to_f32(h)) < 1e-6f);
    }
    CHECK(std::fabs(ta[0] - rad_f16_to_f32(*(const uint16_t*)gt.bytes.data())) == 0.0f);
    /* t.dense: each (part, group) matrix composes that part's stages, so T x equals x rotated stage
     * by stage. Checked on one vector against the tables above, in f64, to f16 accuracy. */
    REQUIRE_EQ(std::string(w.enc.plane[6].role), std::string("t.dense"));
    {
        const uint16_t* td = (const uint16_t*)pl[6].data();
        const int64_t nkb = K / 128;
        double worst_d = 0;
        for (int64_t part = 0; part < 2; ++part)
            for (int64_t g = 0; g < nkb; ++g) {
                std::vector<double> x(128), y(128, 0.0);
                for (int i = 0; i < 128; ++i) x[(size_t)i] = std::sin(0.7 * i + part + 3.0 * g);
                for (int j = 0; j < 128; ++j)
                    for (int i = 0; i < 128; ++i)
                        y[(size_t)j] += rad_f16_to_f32(td[((part * nkb + g) * 128 + j) * 128 + i]) * x[(size_t)i];
                std::vector<double> v = x;
                for (int64_t st = 0; st < S; ++st)
                    for (int j = 0; j < 64; ++j) {
                        const int64_t row = part * S + st;
                        const int a = tp[row * K + g * 128 + 2 * j] - (int)(g * 128);
                        const int b = tp[row * K + g * 128 + 2 * j + 1] - (int)(g * 128);
                        const double t = ta[row * (K / 2) + g * 64 + j], va = v[(size_t)a], vb = v[(size_t)b];
                        v[(size_t)a] = va * std::cos(t) + vb * std::sin(t);
                        v[(size_t)b] = vb * std::cos(t) - va * std::sin(t);
                    }
                for (int j = 0; j < 128; ++j) worst_d = std::max(worst_d, std::fabs(y[(size_t)j] - v[(size_t)j]));
            }
        CHECK(worst_d < 5e-3);
    }

    // decode, rotate forward again with the checkpoint's own convention, compare to the codes
    const void* data[7];
    for (int k = 0; k < 7; ++k) data[k] = pl[(size_t)k].data();
    EncodedView v;
    enc_view(w.enc, w.rows, w.cols, data, &v);
    std::vector<float> dec((size_t)(w.rows * K));
    REQUIRE_EQ(enc_decode_rows(v, 0, w.rows, dec.data(), &why), RAD_OK);
    const T* cw[2] = { &gw, &uw };
    const T* es[2] = { &gs, &us };
    const T* pp[2] = { &gp, &up };
    const T* th[2] = { &gt, &ut };
    const T* cc[2] = { &gc, &uc };
    double worst = 0;
    for (int64_t r = 0; r < w.rows; ++r) {
        const int part = r >= 16, rr = r - 16 * part;
        std::vector<double> x((size_t)K);
        for (int64_t k = 0; k < K; ++k)
            x[(size_t)k] = (double)dec[(size_t)(r * K + k)] *
                           (1.0 / rad_f16_to_f32(*(const uint16_t*)(cc[part]->bytes.data() + 2 * k)));
        for (int64_t s = 0; s < S; ++s)
            for (int64_t p = 0; p < K / 2; ++p) {
                const int16_t* pv = (const int16_t*)pp[part]->bytes.data();
                const int64_t a = pv[s * K + 2 * p], b = pv[s * K + 2 * p + 1];
                const double t = rad_f16_to_f32(*(const uint16_t*)(th[part]->bytes.data() + 2 * (s * (K / 2) + p)));
                const double c = std::cos(t), sn = std::sin(t);
                const double xa = x[(size_t)a], xb = x[(size_t)b];
                x[(size_t)a] = xa * c + xb * sn;
                x[(size_t)b] = xb * c - xa * sn;
            }
        for (int64_t k = 0; k < K; ++k) {
            const uint8_t bt = cw[part]->bytes[(size_t)(rr * K / 2 + k / 2)];
            const float code = rad_fp4e2m1_to_f32((uint8_t)(k % 2 ? bt >> 4 : bt & 15));
            const float sc = std::ldexp(1.0f, (int)es[part]->bytes[(size_t)(rr * K / 32 + k / 32)] - 127);
            worst = std::max(worst, std::fabs(x[(size_t)k] - (double)code * sc));
        }
    }
    CHECK(worst < 1e-4);

    // the per-row fp8 head: one f32 scale per row, read as [N, 1]
    REQUIRE_EQ(ckpt_resolve(ck, prog, "h", false, &w, &why), RAD_OK);
    REQUIRE(w.trivial);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)1);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)0);
    std::vector<float> f(4 * 8);
    REQUIRE_EQ(ckpt_rows_f32(w, 0, 4, f.data(), &why), RAD_OK);
    for (int r = 0; r < 4; ++r) {
        const float q = rad_load_f32(hw.bytes.data(), RAD_F8E4M3, r * 8);
        CHECK_EQ(f[(size_t)(r * 8)], q * 0.25f * (float)(r + 1));
    }
}

/* ParoQuant M2XFP4 is the MXFP4 form plus one metadata byte a group (`weight_meta`, [N, K/32] u8):
 * the three planes of the `m2xfp4` scheme byte for byte, joined row by row, with the rotations
 * beside them as for ParoMx. A piece without its metadata plane is refused, and so is a grid the
 * import does not read. */
TEST(a_paroquant_m2xfp4_weight_carries_its_meta_plane_and_rotations) {
    const int64_t K = 128, S = 2, N = 16;
    uint32_t rng = 777;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng >> 8; };
    auto u8s = [&](const std::string& n, int64_t cols, uint32_t lo, uint32_t span) {
        T t{ n, "U8", { N, cols }, {} };
        for (int64_t i = 0; i < N * cols; ++i) t.bytes.push_back((uint8_t)(lo + next() % span));
        return t;
    };
    auto pairs = [&](const std::string& n) {
        T t{ n, "I16", { S, K }, {} };
        for (int64_t st = 0; st < S; ++st) {
            std::vector<int16_t> perm((size_t)K);
            for (int64_t i = 0; i < K; ++i) perm[(size_t)i] = (int16_t)i;
            for (int64_t i = K - 1; i > 0; --i) std::swap(perm[(size_t)i], perm[next() % (i + 1)]);
            for (int64_t i = 0; i < K; ++i) {
                t.bytes.push_back((uint8_t)(perm[(size_t)i] & 0xff));
                t.bytes.push_back((uint8_t)(perm[(size_t)i] >> 8));
            }
        }
        return t;
    };
    auto f16s = [&](const std::string& n, std::vector<int64_t> shape, float lo, float hi) {
        T t{ n, "F16", shape, {} };
        int64_t cnt = 1;
        for (int64_t d : shape) cnt *= d;
        for (int64_t i = 0; i < cnt; ++i) {
            const uint16_t h = rad_f32_to_f16(lo + (hi - lo) * (float)(next() % 1000) / 1000.0f);
            t.bytes.push_back((uint8_t)(h & 0xff));
            t.bytes.push_back((uint8_t)(h >> 8));
        }
        return t;
    };
    std::vector<T> ts;
    for (const char* b : { "g", "u" }) {
        const std::string n = b;
        ts.push_back(u8s(n + ".weight", K / 2, 0, 256));
        ts.push_back(u8s(n + ".weight_scale", K / 32, 122, 5));
        ts.push_back(u8s(n + ".weight_meta", K / 32, 0, 256));
        ts.push_back(pairs(n + ".pairs"));
        ts.push_back(f16s(n + ".theta", { S, K / 2 }, -3.0f, 3.0f));
        ts.push_back(f16s(n + ".channel_scales", { 1, K }, 0.5f, 2.0f));
    }
    const std::string cfg = "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":"
                            "\"paroquant_m2xfp4\",\"format\":\"m2xfp4\",\"bits\":4,\"group_size\":128,"
                            "\"mx_block\":32,\"subgroup\":8,\"krot\":2}}";
    Checkpoint ck;
    REQUIRE_EQ(ck.open(write_checkpoint(ts, cfg)), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::ParoM2);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    REQUIRE(w.paro);
    CHECK(w.imported && !w.trivial);
    REQUIRE(rad_enc_valid(&w.enc));
    CHECK(rad_enc_is(&w.enc, "m2xfp4"));
    CHECK_EQ(std::string(w.enc.transform), std::string("givens"));
    CHECK_EQ(rad_enc_givens_parts(&w.enc), (int64_t)2);
    CHECK_EQ(w.rows, (int64_t)2 * N);
    REQUIRE_EQ(w.enc.n_planes, 8);
    CHECK_EQ(std::string(w.enc.plane[2].role), std::string("meta"));
    CHECK_EQ(w.enc.plane[2].dtype, (uint32_t)RAD_U8);
    CHECK_EQ(w.enc.plane[2].block[1], (int64_t)32);
    CHECK(ckpt_reads_any(w, { "u.weight_meta" }));

    // codes, exponents and metadata are the checkpoint's bytes, gate's rows then up's
    const std::vector<std::vector<uint8_t>> pl = read_planes(w);
    REQUIRE(!pl.empty());
    for (int k = 0; k < 3; ++k) {
        const T& g = ts[(size_t)k];
        const T& u = ts[(size_t)(6 + k)];
        CHECK_EQ(std::memcmp(pl[(size_t)k].data(), g.bytes.data(), g.bytes.size()), 0);
        CHECK_EQ(std::memcmp(pl[(size_t)k].data() + g.bytes.size(), u.bytes.data(), u.bytes.size()), 0);
    }
    // the rotation tables follow, as ParoMx's: part 1's stage 0 pairs in whole-row columns
    REQUIRE_EQ(std::string(w.enc.plane[3].role), std::string("t.pairs"));
    const int32_t* tp = (const int32_t*)pl[3].data();
    for (int64_t c = 0; c < K; ++c)
        CHECK_EQ(tp[S * K + c], (int32_t)(*(const int16_t*)(ts[9].bytes.data() + 2 * c)));

    // a piece without its metadata plane is refused by name
    {
        std::vector<T> no_meta;
        for (const T& t : ts)
            if (t.name != "u.weight_meta") no_meta.push_back(t);
        Checkpoint c2;
        REQUIRE_EQ(c2.open(write_checkpoint(no_meta, cfg)), RAD_OK);
        CkptWeight w2;
        why.clear();
        CHECK(ckpt_resolve(c2, prog, "gu", false, &w2, &why) != RAD_OK);
        CHECK(why.find("u.weight_meta") != std::string::npos);
    }
    // another subgroup width is not the grid this reads
    {
        std::string other = cfg;
        other.replace(other.find("\"subgroup\":8"), 12, "\"subgroup\":4");
        Checkpoint c3;
        REQUIRE_EQ(c3.open(write_checkpoint(ts, other)), RAD_OK);
        CHECK(c3.imports() == Checkpoint::Import::None);
        CHECK(c3.import_refusal().find("M2XFP4") != std::string::npos);
    }
}

/* ParoQuant int5-bitplane: low nibbles AWQ-packed along N (qweight [K, N/8], eight columns per
 * word in the order 0,2,4,6,1,3,5,7), a fifth-bit plane (qweight_hi [K, N/32], bit n % 32), and the
 * same two packings for the zero points. The logical [N, K] weight becomes u5 codes (a bit stream
 * along K), an f16 scale and a u8 zero per 128 of K, plus the rotation as `givens` tables. */
TEST(a_paroquant_int5_weight_is_relaid_row_by_row) {
    const int64_t N = 64, K = 256, G = K / 128;
    uint32_t rng = 777;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng >> 8; };
    std::vector<int> code((size_t)(N * K)), zero((size_t)(N * G));
    for (auto& c : code) c = (int)(next() % 32);
    for (auto& z : zero) z = (int)(next() % 32);
    auto pack_hi = [&](const std::vector<int>& v, int64_t rows) {
        T t{ "", "I32", { rows, N / 32 }, {} };
        t.bytes.assign((size_t)(rows * N / 32 * 4), 0);
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t n = 0; n < N; ++n) {
                uint32_t w;
                uint8_t* p = t.bytes.data() + (r * (N / 32) + n / 32) * 4;
                std::memcpy(&w, p, 4);
                w |= (uint32_t)((v[(size_t)(n * rows + r)] >> 4) & 1) << (n % 32);
                std::memcpy(p, &w, 4);
            }
        return t;
    };
    const T qw = pack_awq("q.qweight", code, K, N), qz = pack_awq("q.qzeros", zero, G, N);
    T qwh = pack_hi(code, K), qzh = pack_hi(zero, G);
    qwh.name = "q.qweight_hi"; qzh.name = "q.qzeros_hi";
    T sc{ "q.scales", "F16", { G, N }, {} };
    for (int64_t i = 0; i < G * N; ++i) {
        const uint16_t h = rad_f32_to_f16(0.01f * (float)(1 + next() % 50));
        sc.bytes.push_back((uint8_t)(h & 0xff));
        sc.bytes.push_back((uint8_t)(h >> 8));
    }
    T pr{ "q.pairs", "I16", { 2, K }, {} };
    for (int64_t s = 0; s < 2; ++s)
        for (int64_t i = 0; i < K; ++i) {
            const int16_t v = (int16_t)(i % 128 ^ 1);                       // adjacent swap: a pairing
            pr.bytes.push_back((uint8_t)(v & 0xff));
            pr.bytes.push_back((uint8_t)(v >> 8));
        }
    T th{ "q.theta", "F16", { 2, K / 2 }, {} }, cs{ "q.channel_scales", "F16", { 1, K }, {} };
    for (int64_t i = 0; i < 2 * K / 2; ++i) { const uint16_t h = rad_f32_to_f16(0.3f); th.bytes.push_back(h & 0xff); th.bytes.push_back(h >> 8); }
    for (int64_t i = 0; i < K; ++i) { const uint16_t h = rad_f32_to_f16(1.5f); cs.bytes.push_back(h & 0xff); cs.bytes.push_back(h >> 8); }
    const std::string dir = write_checkpoint(
        { qw, qwh, qz, qzh, sc, pr, th, cs },
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":\"paroquant\","
        "\"bits\":5,\"group_size\":128,\"krot\":2,\"format\":\"int5-bitplane\"}}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::ParoI5);
    Program prog;
    prog.name_map.push_back(map("w", RAD_MAP_COPY, { "q.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "w", false, &w, &why), RAD_OK);
    REQUIRE(w.paro && w.imported);
    CHECK_EQ(w.rows, N);
    CHECK_EQ(w.cols, K);
    REQUIRE(rad_enc_valid(&w.enc));
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_U5);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_F16);
    CHECK_EQ(w.enc.plane[2].dtype, (uint32_t)RAD_U8);
    const std::vector<std::vector<uint8_t>> pl = read_planes(w);
    REQUIRE(!pl.empty());
    const int64_t rb = rad_enc_row_bytes(RAD_U5, K);
    int bad = 0;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k)
            if ((int)rad_load_f32(pl[0].data() + n * rb, RAD_U5, k) != code[(size_t)(n * K + k)]) ++bad;
        for (int64_t g = 0; g < G; ++g) {
            if (pl[2][(size_t)(n * G + g)] != zero[(size_t)(n * G + g)]) ++bad;
            if (std::memcmp(pl[1].data() + (n * G + g) * 2, sc.bytes.data() + (g * N + n) * 2, 2) != 0) ++bad;
        }
    }
    CHECK_EQ(bad, 0);
}

/* Z-Lab's int4 (`paroquant`, bits 4, no `format`): the int5 test's AWQ nibbles without the
 * fifth-bit planes, as a fused gate|up whose two pieces each have their own rotation, so the weight
 * carries two parts. Codes and zeros come out row by row per piece; t.part ends each piece. */
TEST(a_paroquant_int4_fused_weight_is_two_rotated_parts) {
    const int64_t N = 32, K = 128, G = K / 128;
    uint32_t rng = 4242;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng >> 8; };
    auto f16s = [&](const std::string& name, std::vector<int64_t> shape, float v) {
        T t{ name, "F16", shape, {} };
        int64_t cnt = 1;
        for (int64_t d : shape) cnt *= d;
        const uint16_t h = rad_f32_to_f16(v);
        for (int64_t i = 0; i < cnt; ++i) { t.bytes.push_back((uint8_t)(h & 0xff)); t.bytes.push_back((uint8_t)(h >> 8)); }
        return t;
    };
    auto pairs = [&](const std::string& name, int shift) {
        T t{ name, "I16", { 1, K }, {} };
        for (int64_t i = 0; i < K; ++i) {
            const int16_t v = (int16_t)((i ^ shift) % 128);
            t.bytes.push_back((uint8_t)(v & 0xff));
            t.bytes.push_back((uint8_t)(v >> 8));
        }
        return t;
    };
    std::vector<T> ts;
    std::vector<int> code[2], zero[2];
    const char* names[2] = { "g", "u" };
    for (int p = 0; p < 2; ++p) {
        code[p].resize((size_t)(N * K));
        zero[p].resize((size_t)(N * G));
        for (auto& c : code[p]) c = (int)(next() % 16);
        for (auto& z : zero[p]) z = (int)(next() % 16);
        const std::string b = names[p];
        ts.push_back(pack_awq(b + ".qweight", code[p], K, N));
        ts.push_back(pack_awq(b + ".qzeros", zero[p], G, N));
        ts.push_back(f16s(b + ".scales", { G, N }, 0.01f * (float)(p + 1)));
        ts.push_back(pairs(b + ".pairs", p + 1));                         // adjacent / next-but-one swaps
        ts.push_back(f16s(b + ".theta", { 1, K / 2 }, 0.2f));
        ts.push_back(f16s(b + ".channel_scales", { 1, K }, 1.25f));
    }
    const std::string dir = write_checkpoint(ts,
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":\"paroquant\","
        "\"bits\":4,\"group_size\":128,\"krot\":1}}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::ParoI4);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    REQUIRE(w.paro && w.imported);
    CHECK_EQ(w.rows, 2 * N);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_U4);
    CHECK_EQ(rad_enc_givens_parts(&w.enc), (int64_t)2);
    const std::vector<std::vector<uint8_t>> pl = read_planes(w);
    REQUIRE(!pl.empty());
    const int64_t rb = rad_enc_row_bytes(RAD_U4, K);
    int bad = 0;
    for (int p = 0; p < 2; ++p)
        for (int64_t n = 0; n < N; ++n) {
            const int64_t r = p * N + n;
            for (int64_t k = 0; k < K; ++k)
                if ((int)rad_load_f32(pl[0].data() + r * rb, RAD_U4, k) != code[p][(size_t)(n * K + k)]) ++bad;
            for (int64_t g = 0; g < G; ++g)
                if (pl[2][(size_t)(r * G + g)] != zero[p][(size_t)(n * G + g)]) ++bad;
        }
    CHECK_EQ(bad, 0);
    const int tpart = rad_enc_find(&w.enc, "t.part");
    REQUIRE(tpart > 0);
    const int32_t* ends = (const int32_t*)pl[(size_t)tpart].data();
    CHECK_EQ(ends[0], (int32_t)N);
    CHECK_EQ(ends[1], (int32_t)(2 * N));
}

/* The composition is pinned by a hash of its bytes, so a container that stores "t.dense" serves the
 * same matrices as the kernel library's load-time composition from pairs and angles. Each matrix is
 * orthogonal to f16 accuracy. */
TEST(paro_dense_matrices_are_the_exact_composition) {
    const int64_t K = 256, P = 2, S = 3, PS = P * S, G = 128;
    std::vector<int32_t> pairs((size_t)(PS * K));
    std::vector<float> ang((size_t)(PS * K / 2));
    for (int64_t r = 0; r < PS; ++r) {
        const int64_t mul = 2 * r + 3, add = 17 * r + 5;                  /* odd multiplier: a permutation */
        for (int64_t c = 0; c < K; ++c)
            pairs[(size_t)(r * K + c)] = (int32_t)((c / G) * G + (c % G * mul + add) % G);
        for (int64_t j = 0; j < K / 2; ++j) ang[(size_t)(r * K / 2 + j)] = 0.37f * (float)(r + 1) - 0.011f * (float)j;
    }
    std::vector<uint16_t> t((size_t)(P * (K / G) * G * G));
    for (int64_t pt = 0; pt < P; ++pt)
        for (int64_t g = 0; g < K / G; ++g) {
            std::vector<int32_t> ia((size_t)(S * 64)), ib((size_t)(S * 64));
            std::vector<float> th((size_t)(S * 64));
            for (int64_t st = 0; st < S; ++st)
                for (int64_t j = 0; j < 64; ++j) {
                    const int64_t row = pt * S + st, o = st * 64 + j;
                    ia[(size_t)o] = pairs[(size_t)(row * K + g * G + 2 * j)] - (int32_t)(g * G);
                    ib[(size_t)o] = pairs[(size_t)(row * K + g * G + 2 * j + 1)] - (int32_t)(g * G);
                    th[(size_t)o] = ang[(size_t)(row * (K / 2) + g * 64 + j)];
                }
            paro_compose_dense(S, ia.data(), ib.data(), th.data(), t.data() + (pt * (K / G) + g) * G * G);
        }
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint8_t b : std::vector<uint8_t>((const uint8_t*)t.data(), (const uint8_t*)t.data() + t.size() * 2))
        h = (h ^ b) * 1099511628211ull;
    CHECK_EQ(h, (uint64_t)0x8e72bd2bb541c6e6ull);
    double worst = 0;
    for (int64_t pg = 0; pg < P * K / G; ++pg)
        for (int64_t i = 0; i < G; ++i) {
            double nrm = 0;
            for (int64_t j = 0; j < G; ++j) {
                const double v = rad_f16_to_f32(t[(size_t)((pg * G + j) * G + i)]);
                nrm += v * v;
            }
            worst = std::max(worst, std::fabs(nrm - 1.0));
        }
    CHECK(worst < 2e-3);
}

/* ================================================================== config.json, as each family spells it
 *
 * The typed fields of RadModelMeta size every tensor at declare, and a wrong one is not refused:
 * a model is sized from another number. Each family spells the same dimension its own way, so each
 * spelling is pinned. */
namespace {

struct ConfigCase {
    const char* what;
    const char* json;
    const char* arch;
    int64_t n_layers, n_embd, n_head, n_head_kv, head_dim, n_ff, n_vocab, n_ctx, n_expert,
            n_expert_used, n_expert_shared;
    double rms_eps, rope_theta;
    std::vector<std::pair<const char*, const char*>> keys;   /* free-form keys; null: absent */
};

}  /* namespace */

TEST(hyv3_static_kv_cache_scales_are_carried_into_model_metadata) {
    const std::string dir = fresh_dir();
    write_file(dir + "/config.json", R"({"model_type":"hy_v3","num_hidden_layers":1})");
    write_shard(dir + "/kv_cache_scales.safetensors", {
        cache_scale_f32("model.layers.0.self_attn.k_cache.scale", 0.0376674123f),
        cache_scale_f32("model.layers.0.self_attn.v_cache.scale", 0.0004359654f),
    });
    write_file(dir + "/model.safetensors.index.json",
               R"({"weight_map":{"model.layers.0.self_attn.k_cache.scale":"kv_cache_scales.safetensors","model.layers.0.self_attn.v_cache.scale":"kv_cache_scales.safetensors"}})");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    const char* k = ck.meta().get("hyv3.kv_cache.0.k_scale");
    const char* v = ck.meta().get("hyv3.kv_cache.0.v_scale");
    REQUIRE(k && v);
    CHECK(std::fabs(std::strtod(k, nullptr) - 0.0376674123) < 1e-9);
    CHECK(std::fabs(std::strtod(v, nullptr) - 0.0004359654) < 1e-10);
    std::filesystem::remove_all(dir);
}

TEST(hyv3_static_kv_cache_scales_reject_invalid_values_and_shapes) {
    const char* name = "model.layers.0.self_attn.k_cache.scale";
    std::vector<T> invalid;
    for (float value : {0.0f, -1.0f, NAN, INFINITY})
        invalid.push_back(cache_scale_f32(name, value));
    invalid.push_back(bf16(name, {1}, 1.0f));
    T nonscalar = cache_scale_f32(name, 1.0f);
    nonscalar.shape = {2};
    nonscalar.bytes.resize(8);
    invalid.push_back(std::move(nonscalar));
    for (const T& t : invalid) {
        const std::string dir = write_checkpoint({t},
            R"({"model_type":"hy_v3","num_hidden_layers":1})");
        Checkpoint ck;
        CHECK_EQ(ck.open(dir), RAD_E_FORMAT);
    }
    /* A true rank-zero scalar and a one-element vector both satisfy the checkpoint contract. */
    T scalar = cache_scale_f32(name, 0.5f);
    scalar.shape.clear();
    const std::string dir = write_checkpoint({scalar},
        R"({"model_type":"hy_v3","num_hidden_layers":1})");
    Checkpoint ck;
    CHECK_OK(ck.open(dir));
    const char* scale = ck.meta().get("hyv3.kv_cache.0.k_scale");
    REQUIRE(scale != nullptr);
    CHECK_EQ(std::strtod(scale, nullptr), 0.5);
}

TEST(config_json_fills_the_model_meta_as_each_family_spells_it) {
    const ConfigCase cases[] = {
        { "a multimodal MoE: text_config first, head_dim read rather than derived, the expert "
          "width as n_ff, rope_parameters",
          R"({"model_type":"qwen3_5_moe","tie_word_embeddings":false,"hidden_size":999,
              "text_config":{"hidden_size":2048,"num_attention_heads":16,"num_key_value_heads":2,
                "head_dim":256,"num_hidden_layers":40,"moe_intermediate_size":512,
                "num_experts":256,"num_experts_per_tok":8,"shared_expert_intermediate_size":512,
                "rms_norm_eps":1e-06,"rope_parameters":{"rope_theta":10000000},
                "max_position_embeddings":262144,"vocab_size":248320,
                "partial_rotary_factor":0.25},
              "vision_config":{"depth":27}})",
          "qwen35moe", 40, 2048, 16, 2, 256, 512, 248320, 262144, 256, 8, 1, 1e-6, 1e7,
          { { "partial_rotary_factor", "0.25" }, { "text_config.partial_rotary_factor", "0.25" },
            { "vision_config.depth", "27" }, { "tie_word_embeddings", "0" },
            { "hidden_size", "999" } } },
        { "a text MoE at the top level: kv heads default to the heads, head_dim derived, "
          "num_local_experts",
          R"({"model_type":"mixtral","hidden_size":4096,"num_attention_heads":32,
              "num_hidden_layers":32,"intermediate_size":14336,"num_local_experts":8,
              "num_experts_per_tok":2,"rope_theta":1000000.0,"rms_norm_eps":1e-05,
              "max_position_embeddings":32768,"vocab_size":32000})",
          "mixtral", 32, 4096, 32, 32, 128, 14336, 32000, 32768, 8, 2, 0, 1e-5, 1e6, {} },
        { "LFM2-VL: the wrapper maps to lfm2, norm_eps, arrays of strings, a bare key wins",
          R"({"model_type":"lfm2_vl","hidden_size":7,
              "text_config":{"hidden_size":1024,"norm_eps":1e-05,"num_attention_heads":16,
                "num_key_value_heads":8,"layer_types":["conv","full_attention","conv"]},
              "vision_config":{"hidden_size":768}})",
          "lfm2", 0, 1024, 16, 8, 64, 0, 0, 0, 0, 0, 0, 1e-5, 1e4,
          { { "layer_types", "conv full_attention conv" }, { "hidden_size", "7" },
            { "vision_config.hidden_size", "768" } } },
        { "Gemma 4 12B: the unified wrapper maps to gemma4, its model_type stays, null experts",
          R"({"model_type":"gemma4_unified","eos_token_id":[1,106],
              "text_config":{"model_type":"gemma4_unified_text","hidden_size":3840,
                "num_attention_heads":16,"num_key_value_heads":8,"num_global_key_value_heads":1,
                "head_dim":256,"global_head_dim":512,"num_hidden_layers":48,
                "intermediate_size":15360,"moe_intermediate_size":null,"num_experts":null,
                "rms_norm_eps":1e-06,"vocab_size":262144,"max_position_embeddings":262144,
                "rope_parameters":{"full_attention":{"rope_theta":1000000.0},
                                   "sliding_attention":{"rope_theta":10000.0}}},
              "vision_config":{"model_type":"gemma4_unified_vision"}})",
          "gemma4", 48, 3840, 16, 8, 256, 15360, 262144, 262144, 0, 0, 0, 1e-6, 1e4,
          { { "model_type", "gemma4_unified" }, { "global_head_dim", "512" },
            { "num_global_key_value_heads", "1" },
            { "rope_parameters.full_attention.rope_theta", "1000000" },
            { "vision_config.model_type", "gemma4_unified_vision" } } },
        { "rope_parameters wins over rope_theta",
          R"({"model_type":"llama","rope_theta":10000,"rope_parameters":{"rope_theta":5000000}})",
          "llama", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-6, 5e6, {} },
        { "how values flatten: booleans, scalar arrays, objects of arrays dropped, null dropped, "
          "doubles to 17 digits, nested objects by dotted path",
          R"({"model_type":"x_y_z","_name_or_path":"org/model","flag":true,"off":false,
              "nums":[1,2,3],"mix":[1.5,"a",true],"objs":[{"a":1}],"nothing":null,
              "pi":3.141592653589793,"nested":{"deep":{"k":"v"}}})",
          "xyz", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-6, 1e4,
          { { "flag", "1" }, { "off", "0" }, { "nums", "1 2 3" }, { "mix", "1.5 a 1" },
            { "objs", nullptr }, { "nothing", nullptr }, { "pi", "3.1415926535897931" },
            { "nested.deep.k", "v" } } },
    };
    for (const ConfigCase& c : cases) {
        const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) }, c.json);
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        const MetaOwn& m = ck.meta();
        const RadModelMeta& t = m.m;
        const bool ok = m.arch == c.arch && t.n_layers == c.n_layers && t.n_embd == c.n_embd &&
                        t.n_head == c.n_head && t.n_head_kv == c.n_head_kv &&
                        t.head_dim == c.head_dim && t.n_ff == c.n_ff && t.n_vocab == c.n_vocab &&
                        t.n_ctx_train == c.n_ctx && t.n_expert == c.n_expert &&
                        t.n_expert_used == c.n_expert_used &&
                        t.n_expert_shared == c.n_expert_shared &&
                        t.rms_eps == (float)c.rms_eps && t.rope_theta == (float)c.rope_theta;
        if (!ok)
            std::fprintf(stderr, "    %s: arch %s layers %lld embd %lld heads %lld/%lld hd %lld "
                         "ff %lld vocab %lld ctx %lld experts %lld/%lld/%lld eps %g theta %g\n",
                         c.what, m.arch.c_str(), (long long)t.n_layers, (long long)t.n_embd,
                         (long long)t.n_head, (long long)t.n_head_kv, (long long)t.head_dim,
                         (long long)t.n_ff, (long long)t.n_vocab, (long long)t.n_ctx_train,
                         (long long)t.n_expert, (long long)t.n_expert_used,
                         (long long)t.n_expert_shared, (double)t.rms_eps, (double)t.rope_theta);
        CHECK(ok);
        for (const auto& [k, v] : c.keys) {
            const char* got = m.get(k);
            const bool kok = v ? (got && std::string(got) == v) : got == nullptr;
            if (!kok) std::fprintf(stderr, "    %s: key %s is %s\n", c.what, k, got ? got : "absent");
            CHECK(kok);
        }
        /* RadModelMeta's own table is published: every free-form key is in it. */
        CHECK_EQ(t.n_kv, (int)m.keys.size());
    }
    {
        const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) },
            R"({"model_type":"x","_name_or_path":"org/model"})");
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        CHECK_EQ(ck.meta().name, std::string("org/model"));
    }
}

/* Nomic BERT names its dimensions n_layer, n_embd, n_head, n_inner and n_positions, its expert
 * count num_experts and its top-k moe_top_k, and its norm epsilon layer_norm_epsilon. Without the
 * fallbacks the engine refuses the checkpoint at startup and rad-convert sizes nothing. */
TEST(a_nomic_config_fills_the_meta_from_its_own_spellings) {
    const char* cfg =
        R"({"model_type":"nomic_bert","n_layer":12,"n_embd":768,"n_head":12,
            "n_inner":3072,"n_positions":2048,"vocab_size":250048,
            "layer_norm_epsilon":1e-05,"num_experts":8,"moe_top_k":2,
            "moe_every_n_layers":2,"rotary_emb_base":10000,"prenorm":false})";
    const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) }, cfg);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    const MetaOwn& m = ck.meta();
    CHECK_EQ(m.arch, std::string("nomicbert"));
    CHECK_EQ(m.m.n_layers, (int64_t)12);
    CHECK_EQ(m.m.n_embd, (int64_t)768);
    CHECK_EQ(m.m.n_head, (int64_t)12);
    CHECK_EQ(m.m.n_head_kv, (int64_t)12);
    CHECK_EQ(m.m.head_dim, (int64_t)64);
    CHECK_EQ(m.m.n_ff, (int64_t)3072);
    CHECK_EQ(m.m.n_vocab, (int64_t)250048);
    CHECK_EQ(m.m.n_ctx_train, (int64_t)2048);
    CHECK_EQ(m.m.n_expert, (int64_t)8);
    CHECK_EQ(m.m.n_expert_used, (int64_t)2);
    CHECK_EQ(m.m.rms_eps, 1e-5f);
    CHECK_EQ(m.m.rope_theta, 1e4f);
    CHECK_EQ(m.get("moe_every_n_layers"), std::string("2"));
    CHECK_EQ(m.get("num_experts"), std::string("8"));
    CHECK_EQ(m.get("rotary_emb_base"), std::string("10000"));
    CHECK_EQ(m.get("prenorm"), std::string("0"));
}

/* ================================================================== a table declared in pieces
 *
 * A table shipped as many shards is declared as several CONCAT maps under one name (each map holds
 * at most eight sources; arch/common/rad_arch.h, map_concat_shards). They join end to end in
 * declaration order: keeping only the first map, or reordering, gives a table that loads and
 * serves the wrong rows. */
TEST(a_concat_declared_in_pieces_joins_every_piece_in_order) {
    std::vector<T> ts;
    std::vector<std::string> names;
    for (int k = 0; k < 10; ++k) {
        names.push_back("t.shard_" + std::to_string(k) + ".weight");
        ts.push_back(bf16(names.back(), { 2, 4 }, 16.0f * (float)k));
    }
    const std::string dir = write_checkpoint(ts, "{}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    std::vector<const char*> first, second;
    for (int k = 0; k < 8; ++k) first.push_back(names[(size_t)k].c_str());
    for (int k = 8; k < 10; ++k) second.push_back(names[(size_t)k].c_str());
    prog.name_map.push_back(map("tab", RAD_MAP_CONCAT, first));
    prog.name_map.push_back(map("other", RAD_MAP_COPY, { names[0].c_str() }));
    prog.name_map.push_back(map("tab", RAD_MAP_CONCAT, second));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "tab", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    REQUIRE_EQ(w.rows, (int64_t)20);
    CHECK_EQ(w.cols, (int64_t)4);
    std::vector<float> f(20 * 4);
    REQUIRE_EQ(ckpt_rows_f32(w, 0, 20, f.data(), &why), RAD_OK);
    for (int k = 0; k < 10; ++k) {
        CHECK_EQ(f[(size_t)(2 * k * 4)], 16.0f * (float)k);            /* shard k's row 0 */
        CHECK_EQ(f[(size_t)((2 * k + 1) * 4 + 3)], 16.0f * (float)k + 7.0f);
    }
    std::vector<uint8_t> raw(20 * 4 * 2), want;
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 0, 20, raw.data(), &why), RAD_OK);
    for (const T& t : ts) want.insert(want.end(), t.bytes.begin(), t.bytes.end());
    CHECK(raw == want);
    std::unordered_set<std::string> hit;
    CHECK(ckpt_reads_any(w, { names[9] }, &hit));
    CHECK(hit.count(names[9]) == 1);
}

/* ================================================================== GGUF as a source
 *
 * A GGUF tensor's dimensions are stored fastest first and are reversed once into the row-major
 * shape everything else uses; the legacy and Q4_K/Q6_K block types decode by their definitions. */
namespace {

using ggufw::GKv;
using ggufw::GTensor;
using ggufw::write_gguf;

uint16_t h16(float v) { return rad_f32_to_f16(v); }

/* Eight blocks of 32 (a [4, 64] tensor) of one legacy type, built from the format's definition:
 * block b has scale d_b and minimum m_b, and its element j the code q(b, j). `want` gets the
 * values the definition gives. */
GTensor legacy_blocks(const char* name, uint32_t type, std::vector<float>* want) {
    GTensor t{ name, type, { 64, 4 }, {} };
    want->assign(4 * 64, 0.0f);
    for (int b = 0; b < 8; ++b) {
        const float d = 0.5f + 0.25f * (float)b, m = -1.0f + 0.125f * (float)b;
        const uint16_t dh = h16(d), mh = h16(m);
        auto q = [&](int j, int bits) { return (j * 5 + b * 3) % (1 << bits); };
        std::vector<uint8_t> blk;
        auto push16 = [&](uint16_t v) { blk.push_back((uint8_t)v); blk.push_back((uint8_t)(v >> 8)); };
        float* out = want->data() + b * 32;
        switch (type) {
            case 8: {                                          /* Q8_0: d * q, q int8 */
                push16(dh);
                for (int j = 0; j < 32; ++j) {
                    const int v = (j * 37 + b * 11) % 255 - 127;
                    blk.push_back((uint8_t)(int8_t)v);
                    out[j] = d * (float)v;
                }
                break;
            }
            case 2: case 3: {                                  /* Q4_0: d (q - 8); Q4_1: d q + m */
                push16(dh);
                if (type == 3) push16(mh);
                for (int j = 0; j < 16; ++j) blk.push_back((uint8_t)(q(j, 4) | (q(j + 16, 4) << 4)));
                for (int j = 0; j < 32; ++j)
                    out[j] = type == 2 ? d * (float)(q(j, 4) - 8) : d * (float)q(j, 4) + m;
                break;
            }
            default: {                                         /* Q5_0: d (q - 16); Q5_1: d q + m */
                push16(dh);
                if (type == 7) push16(mh);
                uint32_t qh = 0;
                for (int j = 0; j < 32; ++j) qh |= (uint32_t)((q(j, 5) >> 4) & 1) << j;
                for (int k = 0; k < 4; ++k) blk.push_back((uint8_t)(qh >> (8 * k)));
                for (int j = 0; j < 16; ++j)
                    blk.push_back((uint8_t)((q(j, 5) & 15) | ((q(j + 16, 5) & 15) << 4)));
                for (int j = 0; j < 32; ++j)
                    out[j] = type == 6 ? d * (float)(q(j, 5) - 16) : d * (float)q(j, 5) + m;
                break;
            }
        }
        t.bytes.insert(t.bytes.end(), blk.begin(), blk.end());
    }
    return t;
}

std::vector<GKv> llama_kvs(int64_t file_type) {
    std::vector<GKv> kvs = {
        { "general.architecture", 8, 0, { "llama" } },
        { "llama.block_count", 4, 3, {} },
        { "llama.embedding_length", 4, 64, {} },
        { "llama.attention.head_count", 4, 4, {} },
        { "llama.attention.head_count_kv", 4, 2, {} },
        { "llama.attention.key_length", 4, 32, {} },
        { "llama.feed_forward_length", 4, 128, {} },
        { "llama.context_length", 4, 4096, {} },
        { "llama.expert_count", 4, 8, {} },
        { "llama.expert_used_count", 4, 2, {} },
        { "llama.attention.layer_norm_rms_epsilon", 6, 1.0 / 1024, {} },
        { "llama.rope.freq_base", 6, 500000, {} },
        { "tokenizer.ggml.tokens", 9, 0, { "a", "b", "c", "<s>", "</s>" } },
    };
    if (file_type >= 0) kvs.push_back({ "general.file_type", 4, (double)file_type, {} });
    return kvs;
}

}  /* namespace */

TEST(a_gguf_tensor_is_row_major_and_its_legacy_blocks_decode_by_definition) {
    const std::string dir = fresh_dir();
    const std::string path = dir + "/m.gguf";
    std::vector<float> w8, w40, w41, w50, w51;
    std::vector<uint8_t> q4k(144, 0);
    const uint16_t q4d = h16(0.5f), q4min = h16(0.25f);
    std::memcpy(q4k.data(), &q4d, 2);
    std::memcpy(q4k.data() + 2, &q4min, 2);
    const uint8_t q4scales[12] = {0x41, 0x42, 0x43, 0x44, 0x45, 0x46,
                                   0x47, 0x48, 0x21, 0x32, 0x43, 0x54};
    std::memcpy(q4k.data() + 4, q4scales, sizeof q4scales);
    q4k[16] = 0x21;       /* low/high nibbles of values 0 and 32 */
    q4k[16 + 64] = 0x32;  /* values 128 and 160, including packed high scale bits */
    GTensor k4{ "t.q4_k", 12, { 256, 1 }, q4k };
    std::vector<uint8_t> q6k(210, 0);
    const uint16_t q6d = h16(0.5f);
    std::memcpy(q6k.data() + 208, &q6d, 2);
    q6k[0] = 0x21; q6k[32] = 0x43; q6k[128] = 0xD3;
    q6k[192] = 2; q6k[194] = 3; q6k[196] = 4; q6k[198] = 5;
    GTensor k6{ "t.q6_k", 14, { 256, 1 }, q6k };
    GTensor f16{ "t.f16", 1, { 64, 4, 2 }, {} };
    for (int i = 0; i < 64 * 4 * 2; ++i) {
        const uint16_t v = h16((float)i * 0.5f);
        f16.bytes.push_back((uint8_t)v);
        f16.bytes.push_back((uint8_t)(v >> 8));
    }
    write_gguf(path, llama_kvs(7),
               { legacy_blocks("t.q8_0", 8, &w8), legacy_blocks("t.q4_0", 2, &w40),
                 legacy_blocks("t.q4_1", 3, &w41), legacy_blocks("t.q5_0", 6, &w50),
                 legacy_blocks("t.q5_1", 7, &w51), k4, k6, f16 });
    Checkpoint ck;
    REQUIRE_EQ(ck.open(path), RAD_OK);
    CHECK(ck.is_gguf());
    CHECK_EQ(ck.dir(), dir);

    Program prog;
    const struct { const char* name; const std::vector<float>* want; } legacy[] = {
        { "t.q8_0", &w8 }, { "t.q4_0", &w40 }, { "t.q4_1", &w41 }, { "t.q5_0", &w50 },
        { "t.q5_1", &w51 } };
    for (const auto& l : legacy) prog.name_map.push_back(map(l.name, RAD_MAP_COPY, { l.name }));
    prog.name_map.push_back(map("t.q4_k", RAD_MAP_COPY, { "t.q4_k" }));
    prog.name_map.push_back(map("t.q6_k", RAD_MAP_COPY, { "t.q6_k" }));
    prog.name_map.push_back(map("t.f16", RAD_MAP_COPY, { "t.f16" }));
    std::string why;
    for (const auto& l : legacy) {
        CkptWeight w;
        REQUIRE_EQ(ckpt_resolve(ck, prog, l.name, false, &w, &why), RAD_OK);
        CHECK(!w.trivial);                                  /* a block type is converted */
        CHECK(has(w.why, "GGUF"));
        CHECK_EQ(w.rank, 2u);
        CHECK_EQ(w.shape[0], (int64_t)4);                   /* ne [64, 4] is [4, 64] */
        CHECK_EQ(w.shape[1], (int64_t)64);
        std::vector<float> f(4 * 64);
        REQUIRE_EQ(ckpt_rows_f32(w, 0, 4, f.data(), &why), RAD_OK);
        int bad = 0;
        for (size_t i = 0; i < f.size(); ++i) bad += f[i] != (*l.want)[i];
        if (bad) std::fprintf(stderr, "    %s: %d value(s) differ\n", l.name, bad);
        CHECK_EQ(bad, 0);
        std::vector<float> row(64);                         /* one row at an offset */
        REQUIRE_EQ(ckpt_rows_f32(w, 3, 1, row.data(), &why), RAD_OK);
        CHECK_EQ(row[63], (*l.want)[3 * 64 + 63]);
    }
    {
        CkptWeight w;
        REQUIRE_EQ(ckpt_resolve(ck, prog, "t.q4_k", false, &w, &why), RAD_OK);
        CHECK_EQ(w.shape[0], (int64_t)1);
        CHECK_EQ(w.shape[1], (int64_t)256);
        std::vector<float> f(256);
        REQUIRE_EQ(ckpt_rows_f32(w, 0, 1, f.data(), &why), RAD_OK);
        CHECK_EQ(f[0], -0.75f);
        CHECK_EQ(f[32], 0.5f);
        CHECK_EQ(f[128], 12.5f);
        CHECK_EQ(f[160], 22.25f);
    }
    {
        CkptWeight w;
        REQUIRE_EQ(ckpt_resolve(ck, prog, "t.q6_k", false, &w, &why), RAD_OK);
        std::vector<float> f(256);
        REQUIRE_EQ(ckpt_rows_f32(w, 0, 1, f.data(), &why), RAD_OK);
        CHECK_EQ(f[0], 17.0f);
        CHECK_EQ(f[32], -43.5f);
        CHECK_EQ(f[64], -28.0f);
        CHECK_EQ(f[96], 50.0f);
    }
    {
        /* A plain f16 tensor of rank 3 is trivial and reversed in every dimension. */
        CkptWeight w;
        REQUIRE_EQ(ckpt_resolve(ck, prog, "t.f16", false, &w, &why), RAD_OK);
        CHECK(w.trivial);
        REQUIRE_EQ(w.rank, 3u);
        CHECK_EQ(w.shape[0], (int64_t)2);
        CHECK_EQ(w.shape[1], (int64_t)4);
        CHECK_EQ(w.shape[2], (int64_t)64);
        std::vector<float> f(64);
        REQUIRE_EQ(ckpt_rows_f32(w, 5, 1, f.data(), &why), RAD_OK);   /* stack 1, row 1 */
        CHECK_EQ(f[3], (float)(5 * 64 + 3) * 0.5f);
    }

    /* The header fills RadModelMeta from the architecture's keys. */
    const RadModelMeta& m = ck.meta().m;
    CHECK_EQ(ck.meta().arch, std::string("llama"));
    CHECK_EQ(m.n_layers, (int64_t)3);
    CHECK_EQ(m.n_embd, (int64_t)64);
    CHECK_EQ(m.n_head, (int64_t)4);
    CHECK_EQ(m.n_head_kv, (int64_t)2);
    CHECK_EQ(m.head_dim, (int64_t)32);
    CHECK_EQ(m.n_ff, (int64_t)128);
    CHECK_EQ(m.n_ctx_train, (int64_t)4096);
    CHECK_EQ(m.n_expert, (int64_t)8);
    CHECK_EQ(m.n_expert_used, (int64_t)2);
    CHECK_EQ(m.rms_eps, 1.0f / 1024);
    CHECK_EQ(m.rope_theta, 500000.0f);
    CHECK_EQ(m.n_vocab, (int64_t)5);
    /* A drafter merges into a safetensors target's namespace only. */
    CHECK_EQ(ck.add_draft(dir, "draft."), (int)RAD_E_UNSUPPORTED);
    /* A quantised file type names its quantisation; the float ones and none select bf16. */
    CHECK_EQ(ck.meta().quant, std::string("gguf_ftype_7"));
    for (const int64_t ft : { (int64_t)0, (int64_t)1, (int64_t)32, (int64_t)-1 }) {
        const std::string p2 = dir + "/ft" + std::to_string(ft + 1) + ".gguf";
        write_gguf(p2, llama_kvs(ft), { k4 });
        Checkpoint c2;
        REQUIRE_EQ(c2.open(p2), RAD_OK);
        CHECK_EQ(c2.meta().quant, std::string());
    }
}

namespace {

/* `rows` x `cols` blocks of Q4_K (type 12) or Q6_K (type 14) with arbitrary quants and scales and
 * finite f16 super-scales: random bits there would make NaNs, which equal nothing. */
GTensor k_blocks(const char* name, uint32_t type, int64_t rows, int64_t cols, uint32_t seed) {
    GTensor t{ name, type, { (uint64_t)cols, (uint64_t)rows }, {} };
    uint32_t x = seed;
    auto fill = [&](int n) {
        for (int i = 0; i < n; ++i) {
            x = x * 1664525u + 1013904223u;
            t.bytes.push_back((uint8_t)(x >> 24));
        }
    };
    auto push16 = [&](float v) {
        const uint16_t h = h16(v);
        t.bytes.push_back((uint8_t)h);
        t.bytes.push_back((uint8_t)(h >> 8));
    };
    for (int64_t b = 0; b < rows * cols / 256; ++b) {
        const float d = 0.0625f * (float)(1 + (b + seed) % 7);
        if (type == 12) {
            push16(d);
            push16(d / 2);
            fill(12 + 128);
        } else {
            fill(128 + 64 + 16);
            push16(d);
        }
    }
    return t;
}

/* Every plane of `w` through ckpt_plane_rows, and the weight those planes decode to. */
void read_planes(const CkptWeight& w, std::vector<std::vector<uint8_t>>* planes,
                 std::vector<float>* values) {
    std::string why;
    planes->assign((size_t)w.enc.n_planes, {});
    const void* data[RAD_ENC_MAX_PLANES] = { nullptr };
    for (int k = 0; k < w.enc.n_planes; ++k) {
        int64_t pr = 0, pc = 0;
        rad_enc_plane_dims(&w.enc.plane[k], w.rows, w.cols, &pr, &pc);
        (*planes)[(size_t)k].resize((size_t)(pr * rad_enc_row_bytes(w.enc.plane[k].dtype, pc)));
        CHECK_EQ(ckpt_plane_rows(w, k, 0, pr, (*planes)[(size_t)k].data(), &why), (int)RAD_OK);
        data[k] = (*planes)[(size_t)k].data();
    }
    EncodedView v;
    enc_view(w.enc, w.rows, w.cols, data, &v);
    values->assign((size_t)(w.rows * w.cols), 0.0f);
    CHECK_EQ(enc_decode_rows(v, 0, w.rows, values->data(), &why), (int)RAD_OK);
}

/* The planes of `w` decode to exactly what the block definition gives (ckpt_rows_f32). */
void check_planes_decode_by_definition(const CkptWeight& w) {
    std::vector<std::vector<uint8_t>> planes;
    std::vector<float> got, want((size_t)(w.rows * w.cols));
    std::string why;
    read_planes(w, &planes, &got);
    REQUIRE_EQ(ckpt_rows_f32(w, 0, w.rows, want.data(), &why), (int)RAD_OK);
    int bad = 0, nonzero = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        bad += !(got[i] == want[i]);
        nonzero += want[i] != 0.0f;
    }
    if (bad) std::fprintf(stderr, "    %d of %zu value(s) differ\n", bad, want.size());
    CHECK_EQ(bad, 0);
    CHECK(nonzero > (int)want.size() / 2);
}

}  /* namespace */

TEST(a_gguf_q8_0_q4_k_or_q6_k_tensor_is_imported_as_its_planes) {
    const std::string path = fresh_dir() + "/m.gguf";
    std::vector<float> w8;
    const GTensor q8 = legacy_blocks("t.q8_0", 8, &w8);
    const GTensor g = k_blocks("g", 12, 2, 512, 1), u = k_blocks("u", 12, 3, 512, 2);
    const GTensor s = k_blocks("s", 12, 2, 256, 3), k6 = k_blocks("k6", 14, 2, 256, 4);
    const GTensor q5{ "t.q5_k", 13, { 256, 1 }, std::vector<uint8_t>(176, 0) };
    write_gguf(path, llama_kvs(15), { q8, g, u, s, k6, q5 });
    Checkpoint ck;
    REQUIRE_EQ(ck.open(path), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("t.q8_0", RAD_MAP_COPY, { "t.q8_0" }));
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g", "u" }));
    prog.name_map.push_back(map("s", RAD_MAP_COPY, { "s" }));
    prog.name_map.push_back(map("k6", RAD_MAP_COPY, { "k6" }));
    prog.name_map.push_back(map("mix", RAD_MAP_CONCAT, { "s", "k6" }));
    prog.name_map.push_back(map("across", RAD_MAP_CONCAT, { "g", "s" }, {}, 1));
    prog.name_map.push_back(map("t.q5_k", RAD_MAP_COPY, { "t.q5_k" }));
    std::string why;

    /* Q8_0: the codes plane is each block's quants and the scale plane its d, byte for byte. */
    {
        CkptWeight w;
        REQUIRE_EQ(ckpt_resolve(ck, prog, "t.q8_0", false, &w, &why), RAD_OK);
        CHECK(w.imported && w.ggml_blocks);
        CHECK(!w.trivial);
        CHECK(has(w.why, "GGUF"));
        REQUIRE(rad_enc_valid(&w.enc));
        REQUIRE_EQ(w.enc.n_planes, 2);
        std::vector<std::vector<uint8_t>> planes;
        std::vector<float> f;
        read_planes(w, &planes, &f);
        REQUIRE_EQ(planes[0].size(), (size_t)(8 * 32));
        REQUIRE_EQ(planes[1].size(), (size_t)(8 * 2));
        int bad = 0;
        for (int b = 0; b < 8; ++b) {
            bad += std::memcmp(planes[0].data() + 32 * b, q8.bytes.data() + 34 * b + 2, 32) != 0;
            bad += std::memcmp(planes[1].data() + 2 * b, q8.bytes.data() + 34 * b, 2) != 0;
        }
        CHECK_EQ(bad, 0);
        int differ = 0;
        for (size_t i = 0; i < w8.size(); ++i) differ += f[i] != w8[i];
        CHECK_EQ(differ, 0);
    }

    /* Q4_K, gate and up joined along rows: the planes decode to the block definition, and a
     * range that starts in the first piece and ends in the second is the same bytes. */
    {
        CkptWeight w;
        REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
        CHECK(w.imported && w.ggml_blocks);
        CHECK(!w.trivial);
        CHECK_EQ(w.rows, (int64_t)5);
        CHECK_EQ(w.cols, (int64_t)512);
        REQUIRE(rad_enc_valid(&w.enc));
        REQUIRE_EQ(w.enc.n_planes, 5);
        check_planes_decode_by_definition(w);
        std::vector<std::vector<uint8_t>> planes;
        std::vector<float> f;
        read_planes(w, &planes, &f);
        for (int k = 0; k < w.enc.n_planes; ++k) {
            int64_t pr = 0, pc = 0;
            rad_enc_plane_dims(&w.enc.plane[k], w.rows, w.cols, &pr, &pc);
            const int64_t rb = rad_enc_row_bytes(w.enc.plane[k].dtype, pc);
            std::vector<uint8_t> part((size_t)(3 * rb));
            REQUIRE_EQ(ckpt_plane_rows(w, k, 1, 3, part.data(), &why), RAD_OK);
            CHECK_EQ(std::memcmp(part.data(), planes[(size_t)k].data() + rb, part.size()), 0);
        }
    }

    /* A row of one block ends its 6-bit planes mid-word. */
    {
        CkptWeight w;
        REQUIRE_EQ(ckpt_resolve(ck, prog, "s", false, &w, &why), RAD_OK);
        CHECK(w.imported && w.ggml_blocks);
        check_planes_decode_by_definition(w);
    }

    /* Q6_K. */
    {
        CkptWeight w;
        REQUIRE_EQ(ckpt_resolve(ck, prog, "k6", false, &w, &why), RAD_OK);
        CHECK(w.imported && w.ggml_blocks);
        CHECK(!w.trivial);
        REQUIRE(rad_enc_valid(&w.enc));
        REQUIRE_EQ(w.enc.n_planes, 3);
        check_planes_decode_by_definition(w);
    }

    /* What it does not import keeps the GGUF reason. */
    {
        CkptWeight w;
        REQUIRE_EQ(ckpt_resolve(ck, prog, "mix", false, &w, &why), RAD_OK);
        CHECK(!w.imported && !w.ggml_blocks && !w.trivial);
        CHECK(has(w.why, "different block types"));
        REQUIRE_EQ(ckpt_resolve(ck, prog, "across", false, &w, &why), RAD_OK);
        CHECK(!w.imported && !w.ggml_blocks && !w.trivial);
        CHECK(has(w.why, "joined along a column"));
        REQUIRE_EQ(ckpt_resolve(ck, prog, "t.q5_k", false, &w, &why), RAD_OK);
        CHECK(!w.imported && !w.ggml_blocks && !w.trivial);
        CHECK(has(w.why, "GGUF"));
    }
}

/* llama.cpp stacks an MoE's experts in one rank-3 tensor of blocks, and an expert's gate|up is two
 * dim-0 slices joined along rows: their planes are those of the same blocks stored on their own. */
TEST(a_slice_of_a_stacked_gguf_block_tensor_is_imported_as_its_planes) {
    const std::string path = fresh_dir() + "/m.gguf";
    const int64_t E = 4, N = 3, K = 512;
    GTensor gate = k_blocks("gate_exps", 12, E * N, K, 5);
    GTensor up = k_blocks("up_exps", 12, E * N, K, 6);
    gate.ne = up.ne = { (uint64_t)K, (uint64_t)N, (uint64_t)E };
    const size_t eb = gate.bytes.size() / E;
    GTensor own{ "gu2", 12, { (uint64_t)K, (uint64_t)(2 * N) }, {} };
    own.bytes.assign(gate.bytes.begin() + 2 * eb, gate.bytes.begin() + 3 * eb);
    own.bytes.insert(own.bytes.end(), up.bytes.begin() + 2 * eb, up.bytes.begin() + 3 * eb);
    write_gguf(path, llama_kvs(15), { gate, up, own });
    Checkpoint ck;
    REQUIRE_EQ(ck.open(path), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "gate_exps", "up_exps" }, { 2, 2 }));
    prog.name_map.push_back(map("gu2", RAD_MAP_COPY, { "gu2" }));
    std::string why;
    CkptWeight w, o;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu2", false, &o, &why), RAD_OK);
    CHECK(w.imported && w.ggml_blocks);
    CHECK_EQ(w.rows, 2 * N);
    CHECK_EQ(w.cols, K);
    REQUIRE(rad_enc_equal(&w.enc, &o.enc));
    std::vector<std::vector<uint8_t>> got, want;
    std::vector<float> fg, fo;
    read_planes(w, &got, &fg);
    read_planes(o, &want, &fo);
    for (int k = 0; k < w.enc.n_planes; ++k) CHECK(got[(size_t)k] == want[(size_t)k]);
}

TEST(hyv3_gguf_metadata_normalizes_architecture_and_config_fields) {
    const std::string path = fresh_dir() + "/hyv3.gguf";
    const std::vector<GKv> kvs = {
        { "general.architecture", 8, 0, { "hy_v3" } },
        { "hy_v3.block_count", 4, 2, {} },
        { "hy_v3.embedding_length", 4, 4, {} },
        { "hy_v3.attention.head_count", 4, 2, {} },
        { "hy_v3.attention.head_count_kv", 4, 1, {} },
        { "hy_v3.attention.key_length", 4, 2, {} },
        { "hy_v3.feed_forward_length", 4, 8, {} },
        { "hy_v3.context_length", 4, 1024, {} },
        { "hy_v3.expert_count", 4, 4, {} },
        { "hy_v3.expert_used_count", 4, 2, {} },
        { "hy_v3.expert_feed_forward_length", 4, 2, {} },
        { "hy_v3.expert_shared_feed_forward_length", 4, 2, {} },
        { "hy_v3.expert_weights_norm", 4, 1, {} },
        { "hy_v3.expert_weights_scale", 6, 2.826, {} },
        { "hy_v3.expert_gating_func", 4, 2, {} },
        { "hy_v3.attention.layer_norm_rms_epsilon", 6, 1e-5, {} },
        { "hy_v3.rope.freq_base", 6, 11158840.0, {} },
        { "tokenizer.ggml.tokens", 9, 0, { "a", "b" } },
        { "general.file_type", 4, 15, {} },
    };
    auto f32 = [](const char* name, std::vector<uint64_t> ne) {
        size_t count = 1;
        for (uint64_t d : ne) count *= (size_t)d;
        return GTensor{ name, 0, std::move(ne), std::vector<uint8_t>(count * sizeof(float), 0) };
    };
    write_gguf(path, kvs, { f32("token_embd.weight", {4, 4}),
                             f32("blk.0.attn_q_norm.weight", {2}),
                             f32("blk.0.attn_k_norm.weight", {2}),
                             f32("blk.0.ffn_gate.weight", {4, 4}),
                             f32("blk.1.ffn_gate_inp.weight", {4, 4}),
                             f32("blk.1.exp_probs_b", {4}) });
    Checkpoint ck;
    REQUIRE_EQ(ck.open(path), RAD_OK);
    CHECK_EQ(ck.meta().arch, std::string("hyv3"));
    CHECK_EQ(ck.meta().quant, std::string("gguf_ftype_15"));
    CHECK_EQ(ck.meta().m.n_vocab, (int64_t)4);  /* matrix rows exceed tokenizer strings */
    CHECK_EQ(std::string(ck.meta().get("qk_norm")), std::string("1"));
    CHECK_EQ(std::string(ck.meta().get("moe_router_use_sigmoid")), std::string("1"));
    CHECK_EQ(std::string(ck.meta().get("moe_router_enable_expert_bias")), std::string("1"));
    CHECK_EQ(std::string(ck.meta().get("enable_moe_fp32_combine")), std::string("1"));
    CHECK_EQ(std::string(ck.meta().get("route_norm")), std::string("1"));
    CHECK_EQ(std::string(ck.meta().get("first_k_dense_replace")), std::string("1"));
    CHECK_EQ(std::string(ck.meta().get("num_shared_experts")), std::string("1"));
    CHECK_EQ(std::string(ck.meta().get("expert_hidden_dim")), std::string("2"));
    CHECK_EQ(std::string(ck.meta().get("moe_intermediate_size")), std::string("4"));
    CHECK_EQ(std::string(ck.meta().get("num_experts")), std::string("4"));
    CHECK_EQ(std::string(ck.meta().get("num_experts_per_tok")), std::string("2"));
    CHECK_EQ(ck.meta().get("router_scaling_factor") != nullptr, true);
}

/* ================================================================== a diffusers pipeline
 *
 * model_index.json and one folder per component read as one checkpoint: the text encoder's
 * tensors keep their names and its config sizes the model, the generator's and the VAE's tensors
 * and keys go under their component's prefix, the scheduler is optional, the tokenizer is in
 * processor/ (else tokenizer/), and the architecture is the pipeline's class. A wrong prefix makes
 * every generator weight "missing", or lets one match a text-encoder weight of the same name. */
TEST(a_diffusers_pipeline_reads_as_one_checkpoint) {
    namespace fs = std::filesystem;
    const std::string dir = fresh_dir();
    std::error_code ec;
    for (const char* sub : { "/text_encoder", "/transformer", "/vae", "/scheduler", "/processor" })
        ::mkdir((dir + sub).c_str(), 0755);
    write_file(dir + "/model_index.json",
               R"({"_class_name":"QwenImage21Pipeline","_diffusers_version":"0.36.0"})");
    write_file(dir + "/text_encoder/config.json",
               R"({"model_type":"qwen3_vl","hidden_size":64,"num_hidden_layers":2})");
    write_shard(dir + "/text_encoder/model-00001-of-00002.safetensors",
                { bf16("model.embed.weight", { 4, 4 }, 1.0f) });
    write_shard(dir + "/text_encoder/model-00002-of-00002.safetensors",
                { bf16("w", { 2, 2 }, 2.0f) });
    write_file(dir + "/text_encoder/model.safetensors.index.json",
               R"({"weight_map":{"model.embed.weight":"model-00001-of-00002.safetensors",
                   "w":"model-00002-of-00002.safetensors"}})");
    write_file(dir + "/transformer/config.json", R"({"num_layers":3,"patch_size":2})");
    write_shard(dir + "/transformer/diffusion_pytorch_model.safetensors",
                { bf16("w", { 2, 2 }, 3.0f) });
    write_file(dir + "/vae/config.json", R"({"latent_channels":16})");
    write_shard(dir + "/vae/diffusion_pytorch_model.safetensors", { bf16("dec.w", { 2, 2 }, 4.0f) });
    write_file(dir + "/scheduler/scheduler_config.json", R"({"shift":3.0})");
    write_file(dir + "/processor/tokenizer.json", "{}");
    write_file(dir + "/processor/preprocessor_config.json", R"({"patch_size":14})");
    {
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        CHECK(ck.find("model.embed.weight") != nullptr);
        const SrcTensor* te = ck.find("w");
        const SrcTensor* tr = ck.find("transformer.w");
        REQUIRE(te && tr);
        CHECK_EQ(rad_load_f32(te->data, RAD_BF16, 0), 2.0f);
        CHECK_EQ(rad_load_f32(tr->data, RAD_BF16, 0), 3.0f);
        CHECK(ck.find("vae.dec.w") != nullptr);
        CHECK(ck.find("dec.w") == nullptr);
        const MetaOwn& m = ck.meta();
        CHECK_EQ(m.arch, std::string("qwenimage21"));
        CHECK_EQ(m.m.n_embd, (int64_t)64);
        CHECK_EQ(m.m.n_layers, (int64_t)2);
        CHECK(m.get("transformer.num_layers") && std::string(m.get("transformer.num_layers")) == "3");
        CHECK(m.get("vae.latent_channels") && std::string(m.get("vae.latent_channels")) == "16");
        CHECK(m.get("scheduler.shift") && std::string(m.get("scheduler.shift")) == "3");
        CHECK(m.get("preprocessor.patch_size") &&
              std::string(m.get("preprocessor.patch_size")) == "14");
        CHECK(m.get("pipeline._class_name") != nullptr);
        CHECK(m.get("num_layers") == nullptr);              /* a component's key is prefixed */
        CHECK_EQ(ck.tok_dir(), dir + "/processor");
    }
    /* No scheduler and a tokenizer/ folder: still one checkpoint, the tokenizer found there. */
    fs::remove_all(dir + "/scheduler", ec);
    fs::rename(dir + "/processor", dir + "/tokenizer", ec);
    {
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        CHECK(ck.meta().get("scheduler.shift") == nullptr);
        CHECK_EQ(ck.tok_dir(), dir + "/tokenizer");
    }
    /* A pipeline with no component weights is not found, rather than an empty model. */
    for (const char* sub : { "/text_encoder", "/transformer", "/vae" }) fs::remove_all(dir + sub, ec);
    {
        Checkpoint ck;
        CHECK_EQ(ck.open(dir), (int)RAD_E_NOTFOUND);
    }
}

/* ================================================================== a drafter of its own
 *
 * A drafter that ships as its own repository (DFlash2) merges under `draft.`: its tensors, and its
 * config.json flattened with no typed extraction, so the trunk is sized from the target's config
 * alone. Without its config.json it is refused: its geometry is not derivable from the target. */
TEST(a_drafter_merges_under_its_prefix_without_resizing_the_trunk) {
    const std::string dir = write_checkpoint({ bf16("embed.weight", { 4, 4 }, 1.0f) }, kConfig);
    const std::string draft = dir + "_draft";
    ::mkdir(draft.c_str(), 0755);
    write_file(draft + "/config.json",
               R"({"architectures":["DFlashDraftModel"],"hidden_size":999,"num_hidden_layers":1,
                   "dflash_config":{"block_size":16}})");
    write_shard(draft + "/model.safetensors", { bf16("fc.weight", { 2, 2 }, 5.0f) });
    {
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        REQUIRE_EQ(ck.add_draft(draft, "draft."), RAD_OK);
        const SrcTensor* fc = ck.find("draft.fc.weight");
        REQUIRE(fc != nullptr);
        CHECK_EQ(rad_load_f32(fc->data, RAD_BF16, 0), 5.0f);
        CHECK(ck.find("fc.weight") == nullptr);
        const MetaOwn& m = ck.meta();
        CHECK_EQ(m.m.n_embd, (int64_t)256);                  /* the target's hidden_size */
        CHECK_EQ(m.m.n_layers, (int64_t)2);
        CHECK(m.get("draft.hidden_size") && std::string(m.get("draft.hidden_size")) == "999");
        CHECK(m.get("draft.architectures") &&
              std::string(m.get("draft.architectures")) == "DFlashDraftModel");
        CHECK(m.get("draft.dflash_config.block_size") &&
              std::string(m.get("draft.dflash_config.block_size")) == "16");
        CHECK_EQ(m.m.n_kv, (int)m.keys.size());              /* republished after the merge */
    }
    std::remove((draft + "/config.json").c_str());
    {
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        CHECK_EQ(ck.add_draft(draft, "draft."), (int)RAD_E_NOTFOUND);
    }
}

/* ================================================================== NVFP4 and per-row FP8 */
namespace {

/* One NVFP4 linear as compressed-tensors writes it: packed e2m1 codes, an e4m3 scale per 16 of K
 * and an f32 global scale (a divisor). */
void add_nvfp4(std::vector<T>* ts, const std::string& base, int64_t N, int64_t K, float d, int seed,
               bool with_global = true) {
    T p{ base + ".weight_packed", "U8", { N, K / 2 }, {} };
    for (int64_t i = 0; i < N * K / 2; ++i) p.bytes.push_back((uint8_t)((i * 37 + seed) & 0xff));
    ts->push_back(p);
    T s{ base + ".weight_scale", "F8_E4M3", { N, K / 16 }, {} };
    for (int64_t i = 0; i < N * K / 16; ++i) {
        float v = 1.0f + (float)((i * 7 + seed) % 23) / 8.0f;
        uint8_t b;
        rad_store_f32(&b, RAD_F8E4M3, 0, v);
        s.bytes.push_back(b);
    }
    ts->push_back(s);
    if (with_global) {
        T g{ base + ".weight_global_scale", "F32", { 1 }, {} };
        g.bytes.resize(4);
        std::memcpy(g.bytes.data(), &d, 4);
        ts->push_back(g);
    }
}

/* The layout of a mixed-precision NVFP4 release (e.g. unsloth's Qwen3.8-27B-NVFP4): an FP8
 * per-channel group beside an NVFP4 group, under a top-level format of mixed-precision. */
const char* kNvfp4Config =
    "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":\"compressed-tensors\","
    "\"format\":\"mixed-precision\",\"config_groups\":{"
    "\"group_0\":{\"format\":\"float-quantized\",\"weights\":{\"num_bits\":8,\"type\":\"float\","
    "\"strategy\":\"channel\",\"symmetric\":true,\"dynamic\":false}},"
    "\"group_1\":{\"format\":\"nvfp4-pack-quantized\",\"weights\":{\"num_bits\":4,\"type\":"
    "\"float\",\"group_size\":16,\"strategy\":\"tensor_group\",\"symmetric\":true,\"dynamic\":false,"
    "\"actorder\":\"static\",\"scale_dtype\":\"torch.float8_e4m3fn\"}}}}}";

/* An FP8 weight [N, K] whose code (n, k) is (n * K + k) % 7 - 3, beside a bf16 per-row scale of
 * the given shape whose row n is `scale(n)`. */
std::vector<T> fp8_rows(const std::string& base, int64_t N, int64_t K, std::vector<int64_t> sshape,
                        float (*scale)(int64_t)) {
    T s{ base + ".weight_scale", "BF16", sshape, {} };
    s.bytes.resize((size_t)N * 2);
    for (int64_t n = 0; n < N; ++n) rad_store_f32(s.bytes.data(), RAD_BF16, n, scale(n));
    return { fp8(base + ".weight", { N, K }), s };
}

}  /* namespace */

/* NVFP4 is imported as its packed codes beside an f32 scale plane, e4m3 / global per 16 of K,
 * each piece with its own global: gate and up join along rows with their own scales. A weight
 * without the NVFP4 tensors reads stock. */
TEST(an_nvfp4_weight_is_its_codes_and_divided_table) {
    std::vector<T> ts;
    add_nvfp4(&ts, "g", 2, 64, 8.0f, 1);
    add_nvfp4(&ts, "u", 3, 64, 4.0f, 5);
    ts.push_back(bf16("n.weight", { 4 }, 1.0f));
    const std::string dir = write_checkpoint(ts, kNvfp4Config);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::Nvfp4);
    CHECK(ck.import_refusal().empty());
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    prog.name_map.push_back(map("n", RAD_MAP_COPY, { "n.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.imported);
    CHECK_EQ(w.rows, (int64_t)5);
    CHECK_EQ(w.cols, (int64_t)64);
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_FP4E2M1);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_F32);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)1);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)16);
    CHECK(rad_enc_valid(&w.enc));
    /* plane 0 is the file's bytes, the pieces in order; plane 1 is e4m3 over the piece's global */
    const T& gp = ts[0], gs = ts[1], up = ts[3], us = ts[4];
    std::vector<uint8_t> cp(5 * 32), tp(5 * 4 * 4);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 0, 5, cp.data(), &why), RAD_OK);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 5, tp.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(cp.data(), gp.bytes.data(), gp.bytes.size()), 0);
    CHECK_EQ(std::memcmp(cp.data() + gp.bytes.size(), up.bytes.data(), up.bytes.size()), 0);
    int bad_tab = 0;
    for (int64_t r = 0; r < 5; ++r) {
        const T& s = r < 2 ? gs : us;
        const int64_t n = r < 2 ? r : r - 2;
        const float d = r < 2 ? 8.0f : 4.0f;
        for (int64_t c = 0; c < 4; ++c)
            if (rad_load_f32(tp.data(), RAD_F32, r * 4 + c) !=
                rad_fp8e4m3_to_f32(s.bytes[(size_t)(n * 4 + c)]) / d)
                ++bad_tab;
    }
    CHECK_EQ(bad_tab, 0);
    /* logical row 3 is u's row 1: each code, low nibble first, times u's plane */
    std::vector<float> f((size_t)64);
    REQUIRE_EQ(ckpt_rows_f32(w, 3, 1, f.data(), &why), RAD_OK);
    int bad = 0;
    for (int64_t k = 0; k < 64; ++k) {
        const uint8_t b = up.bytes[(size_t)(32 + k / 2)];
        const float code = rad_fp4e2m1_to_f32((uint8_t)(k % 2 ? b >> 4 : b & 15));
        const float sc = rad_fp8e4m3_to_f32(us.bytes[(size_t)(4 + k / 16)]) / 4.0f;
        if (f[(size_t)k] != code * sc) ++bad;
    }
    CHECK_EQ(bad, 0);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "n", false, &w, &why), RAD_OK);
    CHECK(w.trivial && rad_enc_is(&w.enc, "plain"));
}

/* The decode against values fixed independently of the importer: a [2, 32] weight packed as
 * compressed-tensors packs it (element 2j in the low nibble, bit 3 the sign, magnitudes
 * 0, 0.5, 1, 1.5, 2, 3, 4, 6), with block scales 2 and 0.5 and a global scale of 4. The value is
 * code * scale / global; a global read as a multiplier would come out 16 times too large. */
TEST(an_nvfp4_weight_decodes_to_the_values_it_was_quantised_from) {
    static const float kMag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const int64_t N = 2, K = 32;
    const float blk[2][2] = { { 2.0f, 0.5f }, { 0.5f, 2.0f } };
    const float global = 4.0f;
    std::vector<uint8_t> nib((size_t)(N * K));
    std::vector<float> want((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const uint8_t c = (uint8_t)((n * 5 + k * 3) % 16);
            nib[(size_t)(n * K + k)] = c;
            const float v = (c & 8 ? -1.0f : 1.0f) * kMag[c & 7];
            want[(size_t)(n * K + k)] = v * blk[n][k / 16] / global;
        }
    std::vector<T> ts;
    T p{ "w.weight_packed", "U8", { N, K / 2 }, {} };
    for (int64_t i = 0; i < N * K; i += 2)
        p.bytes.push_back((uint8_t)(nib[(size_t)i] | (nib[(size_t)i + 1] << 4)));
    ts.push_back(p);
    T s{ "w.weight_scale", "F8_E4M3", { N, K / 16 }, {} };
    /* 2.0 and 0.5 in E4M3 (bias 7): exponent 8 and 6, mantissa 0 */
    for (int64_t n = 0; n < N; ++n)
        for (int64_t b = 0; b < 2; ++b) s.bytes.push_back(blk[n][b] == 2.0f ? 0x40 : 0x30);
    ts.push_back(s);
    T g{ "w.weight_global_scale", "F32", { 1 }, std::vector<uint8_t>(4) };
    std::memcpy(g.bytes.data(), &global, 4);
    ts.push_back(g);
    const std::string dir = write_checkpoint(ts, kNvfp4Config);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("w", RAD_MAP_COPY, { "w.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "w", false, &w, &why), RAD_OK);
    std::vector<float> f((size_t)(N * K));
    REQUIRE_EQ(ckpt_rows_f32(w, 0, N, f.data(), &why), RAD_OK);
    int bad = 0;
    for (size_t i = 0; i < f.size(); ++i) bad += f[i] != want[i];
    CHECK_EQ(bad, 0);
    CHECK_EQ(f[1], 1.5f * 2.0f / 4.0f);                 /* code 3, in byte 0's high nibble */
    CHECK_EQ(f[(size_t)(K + 2)], -1.5f * 0.5f / 4.0f);  /* row 1, k 2: code 11, the sign set */
}

/* An NVFP4 weight without its global scale, or with a zero one, is refused by name: the value
 * cannot be read without the divisor. */
TEST(an_nvfp4_weight_missing_its_global_is_refused) {
    std::vector<T> ts;
    add_nvfp4(&ts, "g", 2, 64, 8.0f, 1, /*with_global=*/false);
    add_nvfp4(&ts, "z", 2, 64, 0.0f, 2);
    const std::string dir = write_checkpoint(ts, kNvfp4Config);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::Nvfp4);
    Program prog;
    prog.name_map.push_back(map("g", RAD_MAP_COPY, { "g.weight" }));
    prog.name_map.push_back(map("z", RAD_MAP_COPY, { "z.weight" }));
    CkptWeight w;
    std::string why;
    CHECK(ckpt_resolve(ck, prog, "g", false, &w, &why) < 0);
    CHECK(why.find("weight_global_scale") != std::string::npos);
    why.clear();
    CHECK(ckpt_resolve(ck, prog, "z", false, &w, &why) < 0);
    CHECK(why.find("divisor") != std::string::npos);
}

namespace {

/* NVIDIA ModelOpt's MIXED_PRECISION layout (as in the NVFP4 releases of Qwen3.8-27B): NVFP4 for
 * `g`, FP8 per tensor for `a` and `b`, every scale and every static input scale a rank-0 f32. */
const char* kModelOptConfig =
    "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":\"modelopt\","
    "\"quant_algo\":\"MIXED_PRECISION\",\"quantized_layers\":{"
    "\"g\":{\"quant_algo\":\"NVFP4\",\"group_size\":16},"
    "\"a\":{\"quant_algo\":\"FP8\"},\"b\":{\"quant_algo\":\"FP8\"}}}}";

T f32_scalar(const std::string& name, float v) {
    T t{ name, "F32", {}, std::vector<uint8_t>(4) };
    std::memcpy(t.bytes.data(), &v, 4);
    return t;
}

}  /* namespace */

/* ModelOpt's NVFP4 against values fixed independently of the importer: codes packed with element
 * 2j in the low nibble, e4m3 block scales 2 and 0.5, and a weight_scale_2 of 0.25 that multiplies
 * (ModelOpt's e2m1 * (float(weight_scale) * weight_scale_2)); read as compressed-tensors' divisor
 * the value would be 16 times too large. The static input scale is not read. */
TEST(a_modelopt_nvfp4_weight_decodes_with_its_second_scale_multiplying) {
    static const float kMag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const int64_t N = 2, K = 32;
    const float blk[2][2] = { { 2.0f, 0.5f }, { 0.5f, 2.0f } };
    const float s2 = 0.25f;
    std::vector<uint8_t> nib((size_t)(N * K));
    std::vector<float> want((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const uint8_t c = (uint8_t)((n * 7 + k * 5) % 16);
            nib[(size_t)(n * K + k)] = c;
            want[(size_t)(n * K + k)] = (c & 8 ? -1.0f : 1.0f) * kMag[c & 7] * (blk[n][k / 16] * s2);
        }
    std::vector<T> ts;
    T p{ "g.weight", "U8", { N, K / 2 }, {} };
    for (int64_t i = 0; i < N * K; i += 2)
        p.bytes.push_back((uint8_t)(nib[(size_t)i] | (nib[(size_t)i + 1] << 4)));
    ts.push_back(p);
    T s{ "g.weight_scale", "F8_E4M3", { N, K / 16 }, {} };
    for (int64_t n = 0; n < N; ++n)
        for (int64_t b = 0; b < 2; ++b) s.bytes.push_back(blk[n][b] == 2.0f ? 0x40 : 0x30);
    ts.push_back(s);
    ts.push_back(f32_scalar("g.weight_scale_2", s2));
    ts.push_back(f32_scalar("g.input_scale", 3.0f));
    const std::string dir = write_checkpoint(ts, kModelOptConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::ModelOpt);
    CHECK(ck.import_refusal().empty());
    CHECK(ck.meta().get("imported_quantization_config.quant_method") != nullptr);
    Program prog;
    prog.name_map.push_back(map("g", RAD_MAP_COPY, { "g.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "g", false, &w, &why), RAD_OK);
    CHECK(w.imported && !w.trivial);
    char en[96];
    rad_enc_format(&w.enc, en, sizeof en);
    CHECK_EQ(std::string(en), std::string("fp4_e2m1*f32[1x16]"));
    std::vector<float> f((size_t)(N * K));
    REQUIRE_EQ(ckpt_rows_f32(w, 0, N, f.data(), &why), RAD_OK);
    int bad = 0;
    for (size_t i = 0; i < f.size(); ++i) bad += f[i] != want[i];
    CHECK_EQ(bad, 0);
    CHECK_EQ(f[1], 3.0f * (2.0f * 0.25f));       /* k 1: code 5 (3.0), byte 0's high nibble */
}

/* ModelOpt's FP8 is E4M3 codes with one f32 scale per tensor. Each piece's scale is written once
 * a row, so two projections with their own scales join (`fp8_e4m3*f32[1x*]`, imported: the plane
 * is not the file's bytes), and the value is code * scale. An import rule makes it the served
 * block fp8, each block's scale its piece's rounded to bf16 (counted when that changes it); a
 * piece that ends inside a block is refused. */
TEST(a_modelopt_fp8_weight_reads_its_per_tensor_scale_once_a_row) {
    std::vector<T> ts;
    ts.push_back(fp8("a.weight", { 128, 128 }));
    ts.push_back(f32_scalar("a.weight_scale", 0.5f));
    ts.push_back(f32_scalar("a.input_scale", 2.0f));
    ts.push_back(fp8("b.weight", { 128, 128 }));
    ts.push_back(f32_scalar("b.weight_scale", 0.3f));
    ts.push_back(fp8("c.weight", { 64, 128 }));
    ts.push_back(f32_scalar("c.weight_scale", 0.25f));
    const std::string dir = write_checkpoint(ts, kModelOptConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("ab", RAD_MAP_CONCAT, { "a.weight", "b.weight" }));
    prog.name_map.push_back(map("ca", RAD_MAP_CONCAT, { "c.weight", "a.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "ab", false, &w, &why), RAD_OK);
    CHECK(w.imported && !w.trivial);
    char en[96];
    rad_enc_format(&w.enc, en, sizeof en);
    CHECK_EQ(std::string(en), std::string("fp8_e4m3*f32[1x*]"));
    const auto pl = read_planes(w);
    REQUIRE_EQ(pl.size(), (size_t)2);
    CHECK_EQ(rad_load_f32(pl[1].data(), RAD_F32, 0), 0.5f);
    CHECK_EQ(rad_load_f32(pl[1].data(), RAD_F32, 127), 0.5f);
    CHECK_EQ(rad_load_f32(pl[1].data(), RAD_F32, 128), 0.3f);
    CHECK_EQ(rad_load_f32(pl[1].data(), RAD_F32, 255), 0.3f);
    std::vector<float> f((size_t)(256 * 128));
    REQUIRE_EQ(ckpt_rows_f32(w, 0, 256, f.data(), &why), RAD_OK);
    int bad = 0;
    for (int64_t r = 0; r < 256; ++r)
        for (int64_t k = 0; k < 128; ++k)
            bad += f[(size_t)(r * 128 + k)] !=
                   (float)(((r % 128) * 128 + k) % 7 - 3) * (r < 128 ? 0.5f : 0.3f);
    CHECK_EQ(bad, 0);

    ImportForm fb;
    fb.block[0] = fb.block[1] = 128;
    fb.scale = RAD_BF16;
    REQUIRE_EQ(ckpt_import_form(&w, fb, &why), RAD_OK);
    rad_enc_format(&w.enc, en, sizeof en);
    CHECK_EQ(std::string(en), std::string("fp8_e4m3*bf16[128x128]"));
    CHECK_EQ(w.rounded, (int64_t)1);              /* 0.3 is not a bf16; 0.5 is */
    const std::vector<float> dec = decode_planes(w);
    REQUIRE_EQ(dec.size(), f.size());
    const float b3 = rad_load_f32(std::vector<uint16_t>{ rad_f32_to_bf16(0.3f) }.data(), RAD_BF16, 0);
    int bad_dec = 0;
    for (int64_t r = 0; r < 256; ++r)
        for (int64_t k = 0; k < 128; ++k)
            bad_dec += dec[(size_t)(r * 128 + k)] !=
                       (float)(((r % 128) * 128 + k) % 7 - 3) * (r < 128 ? 0.5f : b3);
    CHECK_EQ(bad_dec, 0);

    REQUIRE_EQ(ckpt_resolve(ck, prog, "ca", false, &w, &why), RAD_OK);
    CHECK_EQ(ckpt_import_form(&w, fb, &why), (int)RAD_E_SHAPE);
    CHECK(why.find("'c.weight'") != std::string::npos);
}

/* A ModelOpt algorithm this does not read (AWQ-scaled, or NVFP4 in other groups) is refused by
 * name, as is an NVFP4 weight without its second scale. */
TEST(a_modelopt_release_in_an_unread_form_is_refused_by_name) {
    {
        const std::string dir = write_checkpoint(
            { bf16("x", { 2, 2 }, 0) },
            "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":\"modelopt\","
            "\"quant_algo\":\"W4A8_AWQ\"}}");
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        CHECK(ck.imports() == Checkpoint::Import::None);
        CHECK(ck.import_refusal().find("W4A8_AWQ") != std::string::npos);
    }
    {
        const std::string dir = write_checkpoint(
            { bf16("x", { 2, 2 }, 0) },
            "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":\"modelopt\","
            "\"quant_algo\":\"MIXED_PRECISION\",\"quantized_layers\":{\"m.down\":{\"quant_algo\":"
            "\"NVFP4\",\"group_size\":32}}}}");
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        CHECK(ck.imports() == Checkpoint::Import::None);
        CHECK(ck.import_refusal().find("m.down") != std::string::npos);
        CHECK(ck.import_refusal().find("32") != std::string::npos);
    }
    {
        std::vector<T> ts;
        ts.push_back(T{ "g.weight", "U8", { 2, 16 }, std::vector<uint8_t>(32, 0x21) });
        ts.push_back(T{ "g.weight_scale", "F8_E4M3", { 2, 2 }, std::vector<uint8_t>(4, 0x38) });
        const std::string dir = write_checkpoint(ts, kModelOptConfig);
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        Program prog;
        prog.name_map.push_back(map("g", RAD_MAP_COPY, { "g.weight" }));
        CkptWeight w;
        std::string why;
        CHECK(ckpt_resolve(ck, prog, "g", false, &w, &why) < 0);
        CHECK(why.find("weight_scale_2") != std::string::npos);
    }
}

/* An NVFP4 group whose weights are not 4-bit floats in groups of 16, and a compressed-tensors
 * format this does not read, are refused with the group or format named. */
TEST(a_compressed_tensors_release_in_an_unread_form_is_refused_by_name) {
    {
        const char* cfg =
            "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":"
            "\"compressed-tensors\",\"format\":\"nvfp4-pack-quantized\",\"config_groups\":{"
            "\"group_0\":{\"format\":\"nvfp4-pack-quantized\",\"weights\":{\"num_bits\":4,"
            "\"type\":\"float\",\"group_size\":32}}}}}";
        const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) }, cfg);
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        CHECK(ck.imports() == Checkpoint::Import::None);
        CHECK(ck.import_refusal().find("group_0") != std::string::npos);
    }
    {
        const char* cfg =
            "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":"
            "\"compressed-tensors\",\"format\":\"int-quantized\",\"config_groups\":{"
            "\"group_0\":{\"format\":\"int-quantized\",\"weights\":{\"num_bits\":8,"
            "\"type\":\"int\"}}}}}";
        const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) }, cfg);
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        CHECK(ck.imports() == Checkpoint::Import::None);
        CHECK(ck.import_refusal().find("int-quantized") != std::string::npos);
        CHECK(ck.import_refusal().find("nvfp4-pack-quantized") != std::string::npos);
    }
}

/* A compressed-tensors release whose groups are all FP8 has nothing to import and nothing to
 * refuse: it is a CtFp8 import (so the family's quantised plugin serves it), and each weight
 * reads stock with its per-row scale. */
TEST(a_compressed_tensors_fp8_release_reads_stock_with_no_import_refusal) {
    const char* cfg =
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":"
        "\"compressed-tensors\",\"format\":\"float-quantized\",\"config_groups\":{"
        "\"group_0\":{\"format\":\"float-quantized\",\"weights\":{\"num_bits\":8,"
        "\"type\":\"float\",\"strategy\":\"channel\"}}}}}";
    const std::string dir =
        write_checkpoint(fp8_rows("a", 2, 128, { 2, 1 }, [](int64_t n) { return n ? -2.0f : 4.0f; }),
                         cfg);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::CtFp8);
    CHECK(ck.import_refusal().empty());
    CHECK(ck.meta().get("imported_quantization_config.quant_method") != nullptr);
    Program prog;
    prog.name_map.push_back(map("a", RAD_MAP_COPY, { "a.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "a", false, &w, &why), RAD_OK);
    CHECK(w.trivial && !w.imported);
    std::vector<float> f(128);
    REQUIRE_EQ(ckpt_rows_f32(w, 1, 1, f.data(), &why), RAD_OK);
    CHECK_EQ(f[5], (float)((128 + 5) % 7 - 3) * -2.0f);
}

/* Qwen3.8-Flash-Next-FP8 ships its n-gram table as FP8 shards, `X.shard_<i>.weight`, under one
 * scale, `X.weight_scale`. Joined along rows they are one table with one scale, byte for byte:
 * `fp8_e4m3*bf16[*x*]`, the scale plane its single element, and the rows decode by it. */
TEST(fp8_shards_under_one_shared_scale_are_one_table_with_one_scale) {
    T sc{ "t.weight_scale", "BF16", { 1 }, {} };
    sc.bytes.resize(2);
    rad_store_f32(sc.bytes.data(), RAD_BF16, 0, 0.5f);
    const std::string dir = write_checkpoint(
        { fp8("t.shard_0.weight", { 3, 128 }), fp8("t.shard_1.weight", { 2, 128 }), sc }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("t", RAD_MAP_CONCAT, { "t.shard_0.weight", "t.shard_1.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "t", false, &w, &why), RAD_OK);
    CHECK(w.trivial && !w.imported);
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_F8E4M3);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)0);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)0);
    CHECK_EQ(w.rows, (int64_t)5);
    std::vector<float> f(128);
    REQUIRE_EQ(ckpt_rows_f32(w, 3, 1, f.data(), &why), RAD_OK);   /* shard 1's row 0 */
    CHECK_EQ(f[5], (float)(5 % 7 - 3) * 0.5f);
    std::vector<uint16_t> sp(1);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 1, sp.data(), &why), RAD_OK);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 0), 0.5f);
    std::vector<uint8_t> raw(128);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 3, 1, raw.data(), &why), RAD_OK);
    CHECK_EQ(rad_load_f32(raw.data(), RAD_F8E4M3, 5), (float)(5 % 7 - 3));

    /* One shard alone is the same table, and a scale of another name is not borrowed. */
    prog.name_map.push_back(map("u", RAD_MAP_COPY, { "t.shard_1.weight" }));
    REQUIRE_EQ(ckpt_resolve(ck, prog, "u", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
}

/* Per-row FP8 is `fp8_e4m3*<s>[1x*]` whether the scale ships as [N, 1] (compressed-tensors'
 * channel strategy) or [N], and byte for byte: the plane is the shipped scales in row order. */
TEST(a_per_row_fp8_weight_is_one_scale_per_row_in_either_shape) {
    for (const std::vector<int64_t>& shape : { std::vector<int64_t>{ 2, 1 }, std::vector<int64_t>{ 2 } }) {
        const std::string dir =
            write_checkpoint(fp8_rows("a", 2, 128, shape, [](int64_t n) { return n ? -2.0f : 4.0f; }),
                             kConfig);
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        Program prog;
        prog.name_map.push_back(map("a", RAD_MAP_COPY, { "a.weight" }));
        CkptWeight w;
        std::string why;
        REQUIRE_EQ(ckpt_resolve(ck, prog, "a", false, &w, &why), RAD_OK);
        CHECK(w.trivial);
        REQUIRE_EQ(w.enc.n_planes, 2);
        CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_BF16);
        CHECK_EQ(w.enc.plane[1].block[0], (int64_t)1);
        CHECK_EQ(w.enc.plane[1].block[1], (int64_t)0);
        std::vector<uint8_t> sp(4);
        REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 2, sp.data(), &why), RAD_OK);
        CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 0), 4.0f);
        CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 1), -2.0f);
        std::vector<float> f(128);
        REQUIRE_EQ(ckpt_rows_f32(w, 1, 1, f.data(), &why), RAD_OK);
        int bad = 0;
        for (int64_t k = 0; k < 128; ++k)
            if (f[(size_t)k] != (float)(((128 + k) % 7) - 3) * -2.0f) ++bad;
        CHECK_EQ(bad, 0);
    }
}

/* A row join of per-row pieces reads each piece's own scales at the piece's own rows. */
TEST(a_fusion_of_per_row_fp8_pieces_decodes_each_scale) {
    std::vector<T> ts = fp8_rows("g", 4, 8, { 4, 1 }, [](int64_t n) { return 1.0f + (float)n; });
    for (T& t : fp8_rows("u", 3, 8, { 3, 1 }, [](int64_t n) { return 8.0f + (float)n; }))
        ts.push_back(t);
    const std::string dir = write_checkpoint(ts, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    /* logical row 6 is u's row 2: u's scale 10, u's codes from u's own element 16 */
    std::vector<float> f(8);
    REQUIRE_EQ(ckpt_rows_f32(w, 6, 1, f.data(), &why), RAD_OK);
    int bad = 0;
    for (int64_t k = 0; k < 8; ++k)
        if (f[(size_t)k] != (float)(((16 + k) % 7) - 3) * 10.0f) ++bad;
    CHECK_EQ(bad, 0);
    std::vector<uint8_t> sp(7 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 7, sp.data(), &why), RAD_OK);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 3), 4.0f);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 4), 8.0f);
}

/* A column join of FP8 pieces states no encoding, since the read takes scale planes along rows
 * only, and the read refuses it with the reason. */
TEST(a_column_join_of_fp8_pieces_has_no_encoding_and_refuses_the_read) {
    T l = fp8("l.weight", { 4, 8 }), r = fp8("r.weight", { 4, 16 });
    const std::string dir = write_checkpoint(
        { l, bf16("l.weight_scale", { 4, 1 }, 1.0f), r, bf16("r.weight_scale", { 4, 1 }, 1.0f) },
        kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("lr", RAD_MAP_CONCAT, { "l.weight", "r.weight" }, {}, 1));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "lr", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("column") != std::string::npos);
    CHECK_EQ(w.enc.n_planes, (int)0);
    std::vector<float> f(24);
    CHECK(ckpt_rows_f32(w, 0, 1, f.data(), &why) < 0);
    CHECK(why.find("lr") != std::string::npos);
}

/* Only an FP8 weight takes `X.weight_scale` as its scale plane: a bf16 weight beside an unrelated
 * tensor of that name reads plain. */
TEST(a_bf16_weight_beside_a_weight_scale_tensor_reads_plain) {
    const std::string dir = write_checkpoint(
        { bf16("a.weight", { 2, 8 }, 1.0f), bf16("a.weight_scale", { 2, 1 }, 3.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("a", RAD_MAP_COPY, { "a.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "a", false, &w, &why), RAD_OK);
    CHECK(w.trivial && rad_enc_is(&w.enc, "plain"));
}

/* A row join that ends inside the checkpoint's declared block is not served as it stands, but it
 * is read with that block, not a guessed 128, so a recipe quantises the piece's own values. */
TEST(a_fp8_join_that_ends_inside_a_custom_block_reads_with_that_block) {
    const char* cfg =
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":\"fp8\","
        "\"fmt\":\"e4m3\",\"weight_block_size\":[64,64]}}";
    T p = fp8("p.weight", { 100, 128 }), q = fp8("q.weight", { 64, 128 });
    const std::string dir = write_checkpoint(
        { p, bf16("p.weight_scale_inv", { 2, 2 }, 10.0f), q, bf16("q.weight_scale_inv", { 1, 2 }, 10.0f) },
        cfg);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK_EQ(ck.fp8_block_rows(), (int64_t)64);
    Program prog;
    prog.name_map.push_back(map("pq", RAD_MAP_CONCAT, { "p.weight", "q.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "pq", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("inside a scale block") != std::string::npos);
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)64);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)64);
    /* p's row 90 is in its second 64-row block: element 2 of p's [2, 2] plane, 12 (a 128 block
     * would give 10). Column 100 is block column 1, 13. */
    std::vector<float> f(128);
    REQUIRE_EQ(ckpt_rows_f32(w, 90, 1, f.data(), &why), RAD_OK);
    CHECK_EQ(f[0], ((float)((90 * 128) % 7) - 3.0f) * 12.0f);
    CHECK_EQ(f[100], ((float)((90 * 128 + 100) % 7) - 3.0f) * 13.0f);
    /* q's row 50: element 1 of its [1, 2] plane, 11. */
    REQUIRE_EQ(ckpt_rows_f32(w, 150, 1, f.data(), &why), RAD_OK);
    CHECK_EQ(f[100], ((float)((50 * 128 + 100) % 7) - 3.0f) * 11.0f);
}

/* A weight that joins a block-scaled piece with a per-row one has no encoding that reads it, and
 * the read refuses it with the reason. */
TEST(a_fp8_weight_mixed_block_and_per_row_refuses_the_read) {
    T b = fp8("b.weight", { 128, 128 }), c = fp8("c.weight", { 32, 128 });
    const std::string dir = write_checkpoint(
        { b, bf16("b.weight_scale_inv", { 1, 1 }, 1.0f), c, bf16("c.weight_scale", { 32, 1 }, 1.0f) },
        kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("bc", RAD_MAP_CONCAT, { "b.weight", "c.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "bc", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("disagree") != std::string::npos);
    CHECK_EQ(w.enc.n_planes, (int)0);
    std::vector<float> f(128);
    CHECK(ckpt_rows_f32(w, 0, 1, f.data(), &why) < 0);
}

/* ================================================================== derived weights */
/* f32 values `v`, as an EMA codebook ships. */
T f32(const std::string& name, std::vector<int64_t> shape, const std::vector<float>& v) {
    T t{ name, "F32", shape, {} };
    t.bytes.resize(v.size() * 4);
    std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
    return t;
}

/* A linear kept (in, out) is read (out, in): each leading slice transposed, the weight not
 * trivial (a recipe rule stores it). */
TEST(a_transposed_weight_swaps_its_last_two_dimensions_in_each_slice) {
    const std::string dir = write_checkpoint({ bf16("head.weight", { 2, 3, 4 }, 0.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("ht", RAD_MAP_TRANSPOSE, { "head.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "ht", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    REQUIRE_EQ(w.rank, (uint32_t)3);
    CHECK_EQ(w.shape[0], (int64_t)2);
    CHECK_EQ(w.shape[1], (int64_t)4);
    CHECK_EQ(w.shape[2], (int64_t)3);
    std::vector<float> f(2 * 4 * 3);
    REQUIRE_EQ(ckpt_rows_f32(w, 0, 8, f.data(), &why), RAD_OK);
    /* Element (s, j, i) is the source's (s, i, j), which bf16() made s * 12 + i * 4 + j. */
    for (int s = 0; s < 2; ++s)
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 3; ++i)
                CHECK_EQ(f[(size_t)(s * 12 + j * 3 + i)], (float)(s * 12 + i * 4 + j));
}

/* An expert's w2 is a dim-0 sub-tensor of a stacked [n_expert, in, out] tensor, transposed into
 * the [out, in] table a grouped GEMM reads. Ignoring src_index would give every expert the whole
 * stack, and transposing before slicing would mix experts. */
TEST(a_transposed_expert_slice_takes_its_dim0_sub_tensor_then_transposes) {
    std::vector<float> v(8 * 4 * 3);
    for (size_t i = 0; i < v.size(); ++i) v[i] = (float)i;
    const std::string dir = write_checkpoint({ f32("w2", { 8, 4, 3 }, v) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    std::vector<std::string> names;
    names.reserve(8);
    for (int e = 0; e < 8; ++e) names.push_back("w2." + std::to_string(e));
    for (int e = 0; e < 8; ++e)
        prog.name_map.push_back(map(names[(size_t)e].c_str(), RAD_MAP_TRANSPOSE, { "w2" }, { e }));
    prog.name_map.push_back(map("w2.bad", RAD_MAP_TRANSPOSE, { "w2" }, { 8 }));
    CkptWeight w;
    std::string why;
    for (int e = 0; e < 8; ++e) {
        const std::string& name = names[(size_t)e];
        REQUIRE_EQ(ckpt_resolve(ck, prog, name, false, &w, &why), RAD_OK);
        CHECK(!w.trivial);
        REQUIRE_EQ(w.rank, (uint32_t)2);
        CHECK_EQ(w.shape[0], (int64_t)3);
        CHECK_EQ(w.shape[1], (int64_t)4);
        std::vector<float> f(3 * 4);
        REQUIRE_EQ(ckpt_rows_f32(w, 0, 3, f.data(), &why), RAD_OK);
        /* Element (j, i) is the source's (e, i, j). */
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 4; ++i)
                CHECK_EQ(f[(size_t)(j * 4 + i)], (float)(e * 12 + i * 3 + j));
    }
    CHECK(ckpt_resolve(ck, prog, "w2.bad", false, &w, &why) < 0);
}

/* A stack flattened over the rows of a rank-2 tensor (Nomic's experts.mlp.w1 and w2 are
 * [n_expert * n_ff, n_embd]) takes src_rows as one expert's row count. w1 is a copy of the block,
 * w2 the same block transposed. */
TEST(a_flat_expert_stack_is_one_blocks_rows_before_transposing) {
    std::vector<float> v(8 * 4 * 3);
    for (size_t i = 0; i < v.size(); ++i) v[i] = (float)i;
    const std::string dir = write_checkpoint({ f32("w", { 8 * 4, 3 }, v) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    std::vector<std::string> names;
    names.reserve(16);
    for (int e = 0; e < 8; ++e) names.push_back("w1." + std::to_string(e));
    for (int e = 0; e < 8; ++e) names.push_back("w2." + std::to_string(e));
    for (int e = 0; e < 8; ++e) {
        RadNameMap a = map(names[(size_t)e].c_str(), RAD_MAP_COPY, { "w" }, { e });
        a.src_rows[0] = 4;
        prog.name_map.push_back(a);
        RadNameMap t = map(names[(size_t)(8 + e)].c_str(), RAD_MAP_TRANSPOSE, { "w" }, { e });
        t.src_rows[0] = 4;
        prog.name_map.push_back(t);
    }
    CkptWeight w1, w2;
    std::string why;
    for (int e = 0; e < 8; ++e) {
        REQUIRE_EQ(ckpt_resolve(ck, prog, names[(size_t)e], false, &w1, &why), RAD_OK);
        REQUIRE_EQ(w1.rank, (uint32_t)2);
        CHECK_EQ(w1.shape[0], (int64_t)4);
        CHECK_EQ(w1.shape[1], (int64_t)3);
        std::vector<float> f1(4 * 3);
        REQUIRE_EQ(ckpt_rows_f32(w1, 0, 4, f1.data(), &why), RAD_OK);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 3; ++j)
                CHECK_EQ(f1[(size_t)(i * 3 + j)], (float)(e * 12 + i * 3 + j));

        REQUIRE_EQ(ckpt_resolve(ck, prog, names[(size_t)(8 + e)], false, &w2, &why), RAD_OK);
        REQUIRE_EQ(w2.rank, (uint32_t)2);
        CHECK_EQ(w2.shape[0], (int64_t)3);
        CHECK_EQ(w2.shape[1], (int64_t)4);
        std::vector<float> f2(3 * 4);
        REQUIRE_EQ(ckpt_rows_f32(w2, 0, 3, f2.data(), &why), RAD_OK);
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 4; ++i)
                CHECK_EQ(f2[(size_t)(j * 4 + i)], (float)(e * 12 + i * 3 + j));
    }
    RadNameMap bad = map("w1.bad", RAD_MAP_COPY, { "w" }, { 8 });
    bad.src_rows[0] = 4;
    prog.name_map.push_back(bad);
    CHECK(ckpt_resolve(ck, prog, "w1.bad", false, &w1, &why) < 0);
}

/* An EMA codebook is its sums over max(usage, 1e-5), row by row; codebooks declared in pieces
 * stack along dim 0. */
TEST(an_ema_codebook_divides_its_sums_by_their_usage_and_stacks) {
    const std::string dir = write_checkpoint(
        { f32("a.embedding_sum", { 2, 2 }, { 2.0f, 4.0f, 3.0f, 6.0f }),
          f32("a.cluster_usage", { 2 }, { 2.0f, 0.0f }),
          f32("b.embedding_sum", { 2, 2 }, { 8.0f, 1.0f, 5.0f, 5.0f }),
          f32("b.cluster_usage", { 2 }, { 4.0f, 5.0f }) },
        kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("cb", RAD_MAP_EMA, { "a.embedding_sum", "a.cluster_usage" }));
    prog.name_map.push_back(map("cb", RAD_MAP_EMA, { "b.embedding_sum", "b.cluster_usage" }));
    prog.name_map.push_back(map("short", RAD_MAP_EMA, { "a.embedding_sum", "b.embedding_sum" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "cb", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK_EQ(w.shape[0], (int64_t)4);
    CHECK_EQ(w.shape[1], (int64_t)2);
    std::vector<float> f(8);
    REQUIRE_EQ(ckpt_rows_f32(w, 0, 4, f.data(), &why), RAD_OK);
    const float want[8] = { 1.0f, 2.0f, 3.0f / 1e-5f, 6.0f / 1e-5f, 2.0f, 0.25f, 1.0f, 1.0f };
    for (int i = 0; i < 8; ++i) CHECK_EQ(f[(size_t)i], want[i]);
    /* Usage that is not one count a row is refused. */
    CHECK_EQ(ckpt_resolve(ck, prog, "short", false, &w, &why), (int)RAD_E_SHAPE);
}

/* A folder holding a model of its own beside the checkpoint's weights (a codec) is read as part
 * of it: its tensors and its config under the folder's name. */
TEST(a_companion_folder_is_read_under_its_name) {
    const std::string dir = write_checkpoint({ bf16("w", { 2, 2 }, 1.0f) }, kConfig);
    const std::string sub = dir + "/codec";
    REQUIRE_EQ(mkdir(sub.c_str(), 0755), 0);
    write_shard(sub + "/model.safetensors", { bf16("decoder.x", { 2, 2 }, 5.0f) });
    write_file(sub + "/config.json", "{\"model_type\": \"tok\", \"decoder_config\": {\"dim\": 7}}");
    /* A folder without weights of its own is not a model. */
    REQUIRE_EQ(mkdir((dir + "/notes").c_str(), 0755), 0);
    write_file(dir + "/notes/config.json", "{}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.find("w") != nullptr);
    CHECK(ck.find("codec.decoder.x") != nullptr);
    CHECK(ck.find("decoder.x") == nullptr);
    auto mget = [&](const char* k) {
        const char* v = ck.meta().get(k);
        return std::string(v ? v : "");
    };
    CHECK_EQ(mget("codec.model_type"), std::string("tok"));
    CHECK_EQ(mget("codec.decoder_config.dim"), std::string("7"));
}

namespace {

/* A RAD_MAP_ROWS or RAD_MAP_COLS map: source k starts at index[k] and spans rows[k]. */
RadNameMap ranged(const char* declared, int mode, std::vector<const char*> src,
                  std::vector<int> index, std::vector<int64_t> rows) {
    RadNameMap m = map(declared, mode, std::move(src), std::move(index));
    for (size_t i = 0; i < rows.size(); ++i) m.src_rows[i] = rows[i];
    return m;
}

}  /* namespace */

/* A column range of the last dimension, in every leading slice; a range past the end is
 * refused. */
TEST(a_column_range_is_those_columns_of_every_row) {
    const std::string dir = write_checkpoint({ bf16("x.weight", { 3, 5 }, 0.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(ranged("xc", RAD_MAP_COLS, { "x.weight" }, { 1 }, { 3 }));
    prog.name_map.push_back(ranged("far", RAD_MAP_COLS, { "x.weight" }, { 3 }, { 3 }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "xc", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    REQUIRE_EQ(w.rank, (uint32_t)2);
    CHECK_EQ(w.shape[0], (int64_t)3);
    CHECK_EQ(w.shape[1], (int64_t)3);
    std::vector<float> f(9);
    REQUIRE_EQ(ckpt_rows_f32(w, 0, 3, f.data(), &why), RAD_OK);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) CHECK_EQ(f[(size_t)(r * 3 + c)], (float)(r * 5 + 1 + c));
    CHECK_EQ(ckpt_resolve(ck, prog, "far", false, &w, &why), (int)RAD_E_SHAPE);
}

/* Row pieces declared over several maps of one name join in declaration order, so a permutation
 * of a weight's rows can be stated one row at a time. */
TEST(row_pieces_over_several_maps_join_in_order) {
    const std::string dir = write_checkpoint({ bf16("w", { 4, 3 }, 0.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(ranged("p", RAD_MAP_ROWS, { "w" }, { 2 }, { 1 }));
    prog.name_map.push_back(ranged("p", RAD_MAP_ROWS, { "w", "w" }, { 0, 3 }, { 1, 1 }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "p", false, &w, &why), RAD_OK);
    CHECK_EQ(w.shape[0], (int64_t)3);
    CHECK_EQ(w.shape[1], (int64_t)3);
    std::vector<float> f(9);
    REQUIRE_EQ(ckpt_rows_f32(w, 0, 3, f.data(), &why), RAD_OK);
    const int order[3] = { 2, 0, 3 };
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) CHECK_EQ(f[(size_t)(r * 3 + c)], (float)(order[r] * 3 + c));
}

/* A release split into single files merges them under their own names; a name two files share
 * is refused. The metadata and tokenizer come from the directory set before open(). */
TEST(single_files_merge_by_name_and_read_their_metadata_elsewhere) {
    const std::string dir = fresh_dir();
    write_shard(dir + "/dit.safetensors", { bf16("net.a", { 2, 2 }, 1.0f) });
    write_shard(dir + "/te.safetensors", { bf16("model.b", { 2, 2 }, 2.0f) });
    write_shard(dir + "/dup.safetensors", { bf16("net.a", { 2, 2 }, 3.0f) });
    const std::string meta = fresh_dir();
    write_file(meta + "/config.json", "{\"model_type\":\"qwen3\",\"hidden_size\":64}");
    write_file(meta + "/tokenizer.json", "{}");
    Checkpoint ck;
    ck.set_meta_dir(meta);
    REQUIRE_EQ(ck.open(dir + "/dit.safetensors"), RAD_OK);
    REQUIRE_EQ(ck.add_file(dir + "/te.safetensors"), RAD_OK);
    CHECK(ck.find("net.a") != nullptr);
    CHECK(ck.find("model.b") != nullptr);
    CHECK(ck.add_file(dir + "/dup.safetensors") < 0);
    const char* mt = ck.meta().get("model_type");
    CHECK_EQ(std::string(mt ? mt : ""), std::string("qwen3"));
    CHECK_EQ(ck.tok_dir(), meta);
}

RAD_TEST_MAIN()
