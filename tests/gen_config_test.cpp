/* gen_config_test.cpp -- the sampling a checkpoint states for itself.
 *
 * SamplingParams' defaults are an untruncated sampler. A model tuned with a truncated tail and
 * served that way occasionally draws a token from far down its distribution; end-of-turn is one
 * such token, so the symptom is a reply that ends early and looks like a dropped request. A wrong
 * default is silent, hence a tested component.
 *
 * No model and no container: the three sources (a string, a file and a key lookup) are reached
 * directly.
 */
#include "rad_test.h"

#include "gen_config.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <string>

using namespace rad;

namespace {

/* The container's metadata table as gen_sampling_from_kv sees it: bare field names and string
 * values, as rad-convert writes free-form metadata. */
std::function<const char*(const char*)> kv_of(const std::map<std::string, std::string>& m) {
    return [&m](const char* k) -> const char* {
        auto it = m.find(k);
        return it == m.end() ? nullptr : it->second.c_str();
    };
}

std::string write_temp(const char* name, const std::string& body) {
    const std::string path = radtest::tmp_path(std::string(name) + ".json");
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return std::string();
    std::fwrite(body.data(), 1, body.size(), f);
    std::fclose(f);
    return path;
}

/* Qwen3.8-Flash-Next's generation_config.json, verbatim. */
const char* const kQwenGenConfig = R"({
    "bos_token_id": 248044,
    "do_sample": true,
    "eos_token_id": [248046, 248044],
    "pad_token_id": 248044,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95
})";

}  /* namespace */

TEST(gen_config_reads_a_real_generation_config) {
    GenSampling g;
    CHECK(gen_sampling_parse(kQwenGenConfig, "test", &g));
    CHECK_NEAR(g.temp, 1.0f, 1e-6);
    CHECK_EQ(g.top_k, 20);
    CHECK_NEAR(g.top_p, 0.95f, 1e-6);
    CHECK_NEAR(g.min_p, -1.0f, 1e-6);      /* not in the file, so not claimed */

    /* Overlaid on the built-in defaults, it truncates the tail. */
    const SamplingParams sp = g.onto(SamplingParams{});
    CHECK_NEAR(sp.temp, 1.0f, 1e-6);
    CHECK_EQ(sp.top_k, 20);
    CHECK_NEAR(sp.top_p, 0.95f, 1e-6);
    CHECK_NEAR(sp.min_p, 0.0f, 1e-6);      /* untouched: the struct's default */
    CHECK(!sp.greedy());
}

/* A field a source did not state is not claimed, so sources stack: a sidecar that sets only
 * top_k keeps the temperature underneath it. */
TEST(gen_config_claims_only_what_a_source_states) {
    GenSampling g;
    CHECK(gen_sampling_parse(R"({"top_k": 40})", "test", &g));
    CHECK_EQ(g.top_k, 40);
    CHECK(g.temp < 0.0f);
    CHECK(g.top_p < 0.0f);

    SamplingParams base;
    base.temp = 0.6f;
    base.top_p = 0.8f;
    const SamplingParams sp = g.onto(base);
    CHECK_NEAR(sp.temp, 0.6f, 1e-6);
    CHECK_NEAR(sp.top_p, 0.8f, 1e-6);
    CHECK_EQ(sp.top_k, 40);

    GenSampling none;
    CHECK(!gen_sampling_parse(R"({"bos_token_id": 1})", "test", &none));
    CHECK(!none.any());
}

/* `do_sample: false` is greedy and transformers ignores every other sampler field under it. */
TEST(gen_config_reads_do_sample_false_as_greedy) {
    GenSampling g;
    CHECK(gen_sampling_parse(R"({"do_sample": false, "temperature": 0.7, "top_k": 20})",
                             "test", &g));
    CHECK_NEAR(g.temp, 0.0f, 1e-6);
    CHECK_EQ(g.top_k, -1);                 /* not claimed under greedy */
    CHECK(g.onto(SamplingParams{}).greedy());

    /* do_sample true claims nothing by itself. */
    GenSampling t;
    CHECK(gen_sampling_parse(R"({"do_sample": true, "temperature": 0.7})", "test", &t));
    CHECK_NEAR(t.temp, 0.7f, 1e-6);
}

/* transformers spells "no top_k" as -1 and this sampler spells it 0. */
TEST(gen_config_translates_top_k_minus_one_to_off) {
    GenSampling g;
    CHECK(gen_sampling_parse(R"({"top_k": -1})", "test", &g));
    CHECK_EQ(g.top_k, 0);
}

/* The startup line names what top_k 0 does: the sampler draws from its 1024 most likely
 * candidates, which is neither "off" nor the whole vocabulary. */
TEST(the_sampling_line_says_what_top_k_zero_samples_from) {
    SamplingParams sp;
    sp.top_k = 0;
    const std::string zero = GenSampling::describe(sp);
    CHECK(has(zero, "top_k 0 (the 1024 most likely tokens"));
    CHECK(!has(zero, "(off)"));
    sp.top_k = 20;
    const std::string twenty = GenSampling::describe(sp);
    CHECK(has(twenty, "top_k 20, top_p"));
}

/* An out-of-range value is dropped on its own, not refused: the config ships with the model,
 * and one implausible entry should not discard the rest. */
TEST(gen_config_drops_an_out_of_range_field_and_keeps_the_rest) {
    GenSampling g;
    CHECK(gen_sampling_parse(R"({"temperature": 9.0, "top_p": 0.95, "top_k": 20})", "test", &g));
    CHECK(g.temp < 0.0f);                  /* 9.0 is out of range, so not claimed */
    CHECK_NEAR(g.top_p, 0.95f, 1e-6);
    CHECK_EQ(g.top_k, 20);

    /* top_p is a half-open range: 0 is not a nucleus, 1 is. */
    GenSampling z;
    CHECK(!gen_sampling_parse(R"({"top_p": 0.0})", "test", &z));
    GenSampling one;
    CHECK(gen_sampling_parse(R"({"top_p": 1.0})", "test", &one));
    CHECK_NEAR(one.top_p, 1.0f, 1e-6);
}

TEST(gen_config_survives_text_that_is_not_a_config) {
    GenSampling g;
    CHECK(!gen_sampling_parse("{not json", "test", &g));
    CHECK(!gen_sampling_parse("[1, 2, 3]", "test", &g));
    CHECK(!gen_sampling_parse("", "test", &g));
    CHECK(!g.any());
}

/* The container's copy, where rad-convert flattened every value to a string, and the original
 * file resolve to the same sampler. */
TEST(gen_config_reads_the_container_table_as_the_file) {
    GenSampling from_file;
    CHECK(gen_sampling_parse(kQwenGenConfig, "test", &from_file));

    const std::map<std::string, std::string> table = {
        { "do_sample",   "1" },
        { "temperature", "1" },
        { "top_k",       "20" },
        { "top_p",       "0.95" },
    };
    GenSampling from_kv;
    CHECK(gen_sampling_from_kv(kv_of(table), &from_kv));
    CHECK_NEAR(from_kv.temp, from_file.temp, 1e-6);
    CHECK_EQ(from_kv.top_k, from_file.top_k);
    CHECK_NEAR(from_kv.top_p, from_file.top_p, 1e-6);

    /* An empty table states nothing rather than stating zeros. */
    GenSampling empty;
    CHECK(!gen_sampling_from_kv(kv_of({}), &empty));
    CHECK(!empty.any());

    /* A value that is not a number is dropped, and the rest of the table still reads. */
    GenSampling junk;
    CHECK(gen_sampling_from_kv(kv_of({ { "temperature", "warm" }, { "top_k", "20" } }), &junk));
    CHECK(junk.temp < 0.0f);
    CHECK_EQ(junk.top_k, 20);
}

TEST(gen_config_reads_a_file_and_treats_absence_as_silence) {
    const std::string path = write_temp("sidecar", kQwenGenConfig);
    CHECK(!path.empty());
    GenSampling g;
    CHECK(gen_sampling_from_file(path, /*required=*/false, &g));
    CHECK_EQ(g.top_k, 20);

    /* A checkpoint without a generation config is served from the layers underneath, not
     * refused: a missing file claims nothing and returns false. */
    GenSampling missing;
    CHECK(!gen_sampling_from_file("/nonexistent/generation_config.json", false, &missing));
    CHECK(!missing.any());
}

/* The sidecar is the container's path with its extension replaced. The extension starts at the
 * last dot in the basename, so a dot in a directory name is not one. */
TEST(gen_config_derives_the_sidecar_path) {
    CHECK_EQ(gen_sidecar_path("/m/q38-flashnext-unc-w4-mtp.rad"),
             std::string("/m/q38-flashnext-unc-w4-mtp.generation.json"));
    CHECK_EQ(gen_sidecar_path("model.rad"), std::string("model.generation.json"));
    CHECK_EQ(gen_sidecar_path("/a.b/model"), std::string("/a.b/model.generation.json"));
    CHECK_EQ(gen_sidecar_path("/a.b/model.rad"), std::string("/a.b/model.generation.json"));
}

/* Unlimited-OCR's generation_config.json, verbatim: it states no temperature or top-k, only the
 * windowed no-repeat-n-gram ban, so it must be claimed as stated (or the ban would never reach a
 * request that sends no sampler fields). `do_sample: false` still means greedy and ignores it. */
TEST(gen_config_reads_the_ngram_ban) {
    const char* const kUnlimitedOcr = R"({
    "bos_token_id": 0,
    "eos_token_id": 1,
    "no_repeat_ngram_size": 35,
    "no_repeat_ngram_window_size": 128,
    "max_new_tokens": 32768,
    "use_cache": true
})";
    GenSampling g;
    CHECK(gen_sampling_parse(kUnlimitedOcr, "test", &g));
    CHECK_EQ(g.ngram_ban_n, 35);
    CHECK_EQ(g.ngram_window, 128);
    CHECK(g.temp < 0.0f);
    const SamplingParams sp = g.onto(SamplingParams{});
    CHECK_EQ(sp.ngram_ban_n, 35);
    CHECK_EQ(sp.ngram_window, 128);
    CHECK(GenSampling::describe(sp).find("ngram ban n 35 over the last 128 tokens") !=
          std::string::npos);

    /* The container's flattened copy reads the same. */
    GenSampling kv;
    CHECK(gen_sampling_from_kv(kv_of({ { "no_repeat_ngram_size", "35" },
                                       { "no_repeat_ngram_window_size", "128" } }),
                               &kv));
    CHECK_EQ(kv.ngram_ban_n, 35);
    CHECK_EQ(kv.ngram_window, 128);

    /* This sampler's spellings, and the reference's negative window as "the whole sequence". */
    GenSampling own;
    CHECK(gen_sampling_parse(R"({"ngram_ban_n": 35, "ngram_window": -1})", "test", &own));
    CHECK_EQ(own.ngram_ban_n, 35);
    CHECK_EQ(own.ngram_window, 0);

    /* A size outside [0, 256] is dropped on its own; the window still reads. */
    GenSampling bad;
    CHECK(gen_sampling_parse(R"({"no_repeat_ngram_size": 5000,
                                 "no_repeat_ngram_window_size": 128})", "test", &bad));
    CHECK_EQ(bad.ngram_ban_n, -1);
    CHECK_EQ(bad.ngram_window, 128);

    /* do_sample false is greedy: transformers ignores the ban under it, and so does this. */
    GenSampling greedy;
    CHECK(gen_sampling_parse(R"({"do_sample": false, "no_repeat_ngram_size": 35})", "test",
                             &greedy));
    CHECK_EQ(greedy.ngram_ban_n, -1);
    CHECK_EQ(greedy.onto(SamplingParams{}).ngram_ban_n, 0);
}

TEST(gen_config_reads_suppress_tokens) {
    /* Gemma 4's generation_config.json: the end-of-image and end-of-audio markers. */
    GenSampling g;
    CHECK(gen_sampling_parse(R"({"temperature": 1.0, "suppress_tokens": [258883, 258882]})",
                             "test", &g));
    CHECK_EQ((int)g.suppress.size(), 2);
    CHECK_EQ(g.suppress[0], 258883);
    CHECK_EQ(g.suppress[1], 258882);
    const SamplingParams sp = g.onto(SamplingParams{});
    CHECK_EQ((int)sp.suppress_tokens.size(), 2);
    CHECK(SamplingParams{}.suppress_tokens.empty());

    /* The container's flattened copy: the array as a space-separated string. */
    GenSampling kv;
    CHECK(gen_sampling_from_kv(kv_of({ { "suppress_tokens", "258883 258882" } }), &kv));
    CHECK_EQ((int)kv.suppress.size(), 2);
    CHECK_EQ(kv.suppress[1], 258882);

    /* An entry that is no id is dropped on its own, and a non-array is dropped whole. */
    GenSampling bad;
    CHECK(gen_sampling_parse(R"({"top_k": 64, "suppress_tokens": [7, -1, "x", 9]})", "test", &bad));
    CHECK_EQ((int)bad.suppress.size(), 2);
    GenSampling flat;
    CHECK(gen_sampling_parse(R"({"top_k": 64, "suppress_tokens": 7})", "test", &flat));
    CHECK(flat.suppress.empty());
}

RAD_TEST_MAIN()
