/* rad-convert -- safetensors or GGUF in, .rad out.
 *
 * A .rad holds each logical weight as the canonical planes of its encoding (spec §4.1, §4.2);
 * kernel layouts are applied at load, not here. A recipe maps logical weight names to quantiser
 * plugins and options (spec §4.4); the first matching rule decides. An unmatched weight keeps the
 * checkpoint's encoding, which must be one the engine serves directly (bf16, f16, f32,
 * block-scaled FP8), or the weight is reported.
 *
 * Declare still runs: the architecture plugin's name map says which checkpoint tensors make each
 * weight (fusions, alternatives, expert slices), and each resolved kernel's layout hook is asked
 * about the planned encoding (via rad_weight_encoding), so an unservable recipe is refused before
 * anything is written, by the same check the engine makes at load.
 *
 * Every quantised weight is measured against its source through the core's decoder, and the worst
 * are reported.
 */
#include "iface.h"
#include "format/checkpoint.h"
#include "format/encoding.h"
#include "format/imatrix.h"
#include "format/quantise.h"
#include "format/recipe.h"
#include "plugin/registry.h"
#include "text/tokenizer.h"
#include "rad_plugin.h"
#include "rad_quant.h"

#include <dlfcn.h>
#include <nlohmann/json.hpp>

#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <sstream>
#include <thread>
#include <unordered_set>

using namespace rad;

namespace {

bool file_exists(const std::string& p) { std::ifstream f(p); return (bool)f; }

/* clef (decision.type=clef): the head's geometry from joint_head_config.json beside the
 * checkpoint, and its three learned scalars from the head's own tensors (joint_head.safetensors,
 * merged with --add), into the header. The plugin declares the head from the geometry
 * (arch/common/rad_block_clef.h) and the server applies the scalars (core/server/clef.h). */
int clef_metadata(const std::string& in, const std::string& config_dir, const Checkpoint& ck,
                  MetaOwn& meta) {
    /* The head lives in the clef plugins, selected by the architecture id. A container that
     * says decision.type=clef but keeps its qwen35 id would load the trunk alone and answer
     * /v1/systemone from a head that never ran. */
    if (meta.arch != "clef") {
        RAD_ERR("%s: decision.type=clef needs the clef plugins: convert with --arch clef (the "
                "container's architecture is '%s')", in.c_str(), meta.arch.c_str());
        return -1;
    }
    std::string dir = config_dir;
    if (dir.empty()) {
        struct stat st{};
        if (stat(in.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) dir = in;
        else {
            const size_t slash = in.find_last_of('/');
            dir = slash == std::string::npos ? "." : in.substr(0, slash);
        }
    }
    const std::string path = dir + "/joint_head_config.json";
    std::ifstream f(path);
    if (!f) {
        RAD_ERR("%s: decision.type=clef needs the head's joint_head_config.json beside the "
                "checkpoint (or under --config)", path.c_str());
        return -1;
    }
    nlohmann::json c;
    try { f >> c; } catch (const std::exception& e) {
        RAD_ERR("%s: %s", path.c_str(), e.what());
        return -1;
    }
    for (const char* k : { "hidden_size", "width", "routing_layers", "layers", "heads", "feedforward" }) {
        if (!c.contains(k) || !c[k].is_number_integer() || c[k].get<long long>() <= 0) {
            RAD_ERR("%s: `%s` is missing or not a positive integer", path.c_str(), k);
            return -1;
        }
        meta.set(std::string("decision.clef.") + k, std::to_string(c[k].get<long long>()));
    }
    for (const char* k : { "prior_logit_scale", "joint_logit_scale", "residual_gate" }) {
        const SrcTensor* t = ck.find(k);
        if (!t || !t->data || t->bytes < 2) {
            RAD_ERR("the head's `%s` is not in the checkpoint: add joint_head.safetensors with "
                    "--add", k);
            return -1;
        }
        float v = 0.0f;
        if (t->rad_dtype == RAD_BF16) {
            uint16_t h = 0;
            std::memcpy(&h, t->data, 2);
            const uint32_t u = (uint32_t)h << 16;
            std::memcpy(&v, &u, 4);
        } else if (t->rad_dtype == RAD_F32 && t->bytes >= 4) {
            std::memcpy(&v, t->data, 4);
        } else {
            RAD_ERR("the head's `%s` is %s; bf16 or f32 is read", k, rad_dtype_name(t->rad_dtype));
            return -1;
        }
        char s[32];
        std::snprintf(s, sizeof s, "%.9g", (double)v);
        meta.set(std::string("decision.clef.") + k, s);
    }
    RAD_INFO("clef     head %s x %s, %s routing + %s decoder layers; scalars from the checkpoint",
             meta.get("decision.clef.width"), meta.get("decision.clef.heads"),
             meta.get("decision.clef.routing_layers"), meta.get("decision.clef.layers"));
    return 0;
}

/* The rounding every Laya checkpoint needs: the release ships f16 (laya.train saves .half()), and
 * the laya plugin reads its linears and tables in bf16, as the release computes them on a card.
 * Norms and biases are declared f32, an exact widening of f16, so no rule names them. */
const char* const kLayaBf16 =
    "token_embd.weight          cast dtype=bf16\n"
    "type_emb.weight            cast dtype=bf16\n"
    "scorer.fc1.weight          cast dtype=bf16\n"
    "scorer.fc2.weight          cast dtype=bf16\n"
    "blk.*.attn_qkv.weight      cast dtype=bf16\n"
    "blk.*.attn_out.weight      cast dtype=bf16\n"
    "blk.*.ffn_in_gate.weight   cast dtype=bf16\n"
    "blk.*.ffn_down.weight      cast dtype=bf16\n"
    "head.*.attn_qkv.weight     cast dtype=bf16\n"
    "head.*.attn_out.weight     cast dtype=bf16\n"
    "head.*.ffn_up.weight       cast dtype=bf16\n"
    "head.*.ffn_down.weight     cast dtype=bf16\n";

/* Laya (a checkpoint whose config.json says model_type laya, or --set decision.type=laya): the
 * release keeps its parts in folders, so the encoder's geometry comes from encoder/config.json,
 * the head's settings from rl_agent_config.json and the vocabulary from tokenizer/. The plugin
 * declares from the geometry (arch/laya); the server builds each question's sequence from the
 * special tokens and lengths and applies the fitted temperatures (core/server/laya.h). */
int laya_metadata(const std::string& in, const std::string& config_dir, MetaOwn& meta,
                  std::string* tokjson) {
    if (meta.arch != "laya") {
        RAD_ERR("%s: decision.type=laya needs a Laya checkpoint directory, whose config.json says "
                "model_type laya (this one's architecture is '%s')", in.c_str(), meta.arch.c_str());
        return -1;
    }
    std::string dir = config_dir;
    if (dir.empty()) {
        struct stat st{};
        if (stat(in.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) dir = in;
        else {
            const size_t slash = in.find_last_of('/');
            dir = slash == std::string::npos ? "." : in.substr(0, slash);
        }
    }
    auto read_json = [](const std::string& path, nlohmann::json* j) {
        std::ifstream f(path);
        if (!f) { RAD_ERR("%s: a Laya checkpoint carries it; cannot read it", path.c_str()); return -1; }
        try { f >> *j; } catch (const std::exception& e) {
            RAD_ERR("%s: %s", path.c_str(), e.what());
            return -1;
        }
        return j->is_object() ? 0 : -1;
    };

    /* The encoder's own config: its geometry and free-form keys (layer_types, local_attention,
     * the rotary thetas), keeping the checkpoint's model_type, which named the plugin. */
    MetaOwn enc;
    if (meta_from_config_json(dir + "/encoder", &enc) != RAD_OK) return -1;
    for (size_t i = 0; i < enc.keys.size(); ++i)
        if (enc.keys[i] != "model_type" && enc.keys[i] != "architectures") meta.set(enc.keys[i], enc.vals[i]);
    meta.m = enc.m;

    nlohmann::json rl;
    if (read_json(dir + "/rl_agent_config.json", &rl) < 0) return -1;
    const std::pair<const char*, long long> counts[] = { { "max_len", 1 }, { "head_max_len", 1 },
                                                         { "head_layers", 0 } };
    for (const auto& [k, least] : counts) {
        if (!rl.contains(k) || !rl[k].is_number_integer() || rl[k].get<long long>() < least) {
            RAD_ERR("%s/rl_agent_config.json: `%s` is missing or not a count", dir.c_str(), k);
            return -1;
        }
        meta.set(std::string("decision.laya.") + k, std::to_string(rl[k].get<long long>()));
    }
    if (rl.contains("option_layout") && rl["option_layout"] != "sequential") {
        RAD_ERR("%s/rl_agent_config.json: option_layout '%s'; the sequential layout every "
                "published checkpoint uses is served", dir.c_str(), rl["option_layout"].dump().c_str());
        return -1;
    }
    auto num = [](double v) { char s[32]; std::snprintf(s, sizeof s, "%.9g", v); return std::string(s); };
    std::string temps = "1 1 1";
    if (rl.contains("temperature")) {
        const nlohmann::json& t = rl["temperature"];
        if (!t.is_array() || t.size() != 3 || !t[0].is_number() || !t[1].is_number() || !t[2].is_number()) {
            RAD_ERR("%s/rl_agent_config.json: `temperature` is not three numbers (choice, score, "
                    "noul)", dir.c_str());
            return -1;
        }
        temps = num(t[0].get<double>()) + " " + num(t[1].get<double>()) + " " + num(t[2].get<double>());
    }
    meta.set("decision.laya.temperature", temps);
    if (rl.contains("temperature_by_options") && rl["temperature_by_options"].is_object())
        for (const auto& kv : rl["temperature_by_options"].items())
            if (kv.value().is_number())
                meta.set("decision.laya.temperature_by_options." + kv.key(), num(kv.value().get<double>()));

    /* The special tokens by name (tokenizer_config.json), their ids from tokenizer.json. */
    const std::string tdir = dir + "/tokenizer";
    if (tokjson->empty()) *tokjson = tdir + "/tokenizer.json";
    nlohmann::json tc, tj;
    if (read_json(tdir + "/tokenizer_config.json", &tc) < 0 || read_json(*tokjson, &tj) < 0) return -1;
    for (const char* which : { "cls_token", "sep_token", "mask_token" }) {
        const nlohmann::json* v = tc.contains(which) ? &tc[which] : nullptr;
        if (v && v->is_object() && v->contains("content")) v = &(*v)["content"];
        if (!v || !v->is_string()) {
            RAD_ERR("%s/tokenizer_config.json: no `%s`", tdir.c_str(), which);
            return -1;
        }
        const std::string tok = v->get<std::string>();
        long long id = -1;
        if (tj.contains("added_tokens"))
            for (const auto& a : tj["added_tokens"])
                if (a.contains("content") && a["content"] == tok && a.contains("id")) id = a["id"].get<long long>();
        if (id < 0 && tj.contains("model") && tj["model"].contains("vocab") && tj["model"]["vocab"].is_object() &&
            tj["model"]["vocab"].contains(tok))
            id = tj["model"]["vocab"][tok].get<long long>();
        if (id < 0) {
            RAD_ERR("%s: the %s '%s' is not in the vocabulary", tokjson->c_str(), which, tok.c_str());
            return -1;
        }
        const std::string base = std::string(which).substr(0, std::strlen(which) - 6);
        meta.set("decision.laya." + base, std::to_string(id));
        if (base == "mask") meta.set("decision.laya.mask_token", tok);
    }
    meta.set("decision.type", "laya");
    meta.finish();
    RAD_INFO("laya     %s: encoder %lld x %lld, head %s layers, max_len %s, head_max_len %s",
             dir.c_str(), (long long)meta.m.n_layers, (long long)meta.m.n_embd,
             meta.get("decision.laya.head_layers"), meta.get("decision.laya.max_len"),
             meta.get("decision.laya.head_max_len"));
    return 0;
}

/* `blk.12.attn_q.weight` -> 12; -1 for a model-level name (GGUF naming, as in an imatrix). */
int32_t layer_from_name(const std::string& n) {
    if (n.rfind("blk.", 0) != 0) return -1;
    size_t i = 4, v = 0;
    if (i >= n.size() || n[i] < '0' || n[i] > '9') return -1;
    while (i < n.size() && n[i] >= '0' && n[i] <= '9') v = v * 10 + (size_t)(n[i++] - '0');
    return (int32_t)v;
}

/* ================================================================== the plan */
/* One logical weight this run writes: where it comes from, what decides it, what it becomes. */
struct Logical {
    CkptWeight      src;
    const CompiledRule* rule = nullptr;
    /* The `import` rule that re-expressed it (ckpt_import_form); `rule` is then null. */
    const RecipeRule*   import_rule = nullptr;
    RadEncoding     enc{};
    int             status = RAD_OK;
    std::string     why;
    /* from the declarations that take it */
    bool    declared = false;
    int32_t layer = -1, expert = -1, slot = 0;
    std::string first_decl;
    /* the report */
    double  err = -1, err_imp = -1;
};

RadQuantWeight quant_weight(const Logical& L, const float* importance) {
    RadQuantWeight qw{};
    qw.name = L.src.name.c_str();
    qw.rank = L.src.rank;
    for (uint32_t i = 0; i < RAD_MAX_RANK; ++i) qw.shape[i] = L.src.shape[i];
    qw.rows = L.src.rows;
    qw.cols = L.src.cols;
    qw.layer = L.layer;
    qw.expert = L.expert;
    qw.importance = importance;
    return qw;
}

/* Movement-unit order: model-level weights, then each layer's own, then its experts in id order
 * with their slots adjacent, so experts are addressable as base + id * stride. */
bool plan_less(const Logical* a, const Logical* b) {
    auto key = [](const Logical* p) {
        return std::make_tuple(p->layer < 0 ? -1 : p->layer, p->expert, p->slot, p->src.name);
    };
    return key(a) < key(b);
}

void usage() {
    std::printf(
        "rad-convert -- safetensors or GGUF in, .rad out\n"
        "\n"
        "usage: rad-convert [options] <input> -o <model.rad>\n"
        "\n"
        "  <input>              a .gguf, a .safetensors, an index.json, or a checkpoint directory\n"
        "  -o, --out FILE       the container to write\n"
        "      --recipe FILE    the quantisation recipe: one rule a line,\n"
        "                         PATTERN  QUANTISER  key=value ...\n"
        "                       the first rule whose glob matches a logical weight name decides\n"
        "                       it; a weight no rule matches keeps the checkpoint's encoding.\n"
        "                       $NAME in a value is the environment's\n"
        "      --quant RULE     'PATTERN=QUANTISER:key=value,...', ahead of the recipe file's\n"
        "                       rules. Repeatable\n"
        "      --list-quantizers  every quantiser loaded and the options each takes, then exit\n"
        "      --arch ID        override the architecture id from the checkpoint\n"
        "      --home DIR       $RADIANCE_HOME (kernels/, architectures/, quantizers/)\n"
        "      --kernels A:B    the kernel libraries to load, first wins. Default $RADIANCE_KERNELS,\n"
        "                       else every library found\n"
        "                       Declare checks the recipe against the kernels it resolves\n"
        "      --tokenizer F    tokenizer.json; its declared chain is baked instead of being\n"
        "                       reconstructed from a pre-tokenizer name\n"
        "      --tokenizer2 F   a second tokenizer.json, baked as the container's second vocab:\n"
        "                       a generator that reads its prompt through two tokenizers\n"
        "      --add FILE       one more .safetensors file merged into the checkpoint, its tensor\n"
        "                       names kept: a release shipped as separate single files (a\n"
        "                       diffusion model, a text encoder and a VAE). Repeatable\n"
        "      --config DIR     read config.json, generation_config.json and tokenizer.json from\n"
        "                       DIR, for a single-file checkpoint that ships none of them\n"
        "      --imatrix F      llama.cpp imatrix: the expert-popularity profile, the column\n"
        "                       importances handed to each quantiser, and the weighting of the\n"
        "                       error report\n"
        "      --chat-template F  a jinja template baked as the container's chat template,\n"
        "                       overriding the checkpoint's own (or its absence). A release that\n"
        "                       ships none (DeepSeek's) needs one, or /v1/chat/completions answers\n"
        "                       501. Trailing newlines are stripped\n"
        "      --set K=V        a metadata key, seen by declare AND written into the container,\n"
        "                       so a convert-time choice that must also hold at serve time is\n"
        "                       stated once. Repeatable\n"
        "      --max-tok N      the max_tok declare is run at (default 8192, or a generator's\n"
        "                       largest pass)\n"
        "      --max-ctx N      the context bound declare is run at, as --max-model-len is on\n"
        "                       the engine. Default: the checkpoint's training context\n"
        "      --draft-model DIR  a drafter that ships as its own repository (DFlash2), merged\n"
        "                       under `draft.`\n"
        "      --kv-cache-dtype T  bf16 | fp8, the cache width declare is run at. Default: the\n"
        "                       first of bf16, fp8 at which every op resolves -- the container\n"
        "                       is the same at either, so this only picks the kernels it checks\n"
        "      --max-spec N     the max_spec declare is run at (default 4, or the block a\n"
        "                       drafter states; a number given here is used as given); 0\n"
        "                       leaves a draft head out\n"
        "      --reuse FILE     a container converted earlier from the SAME checkpoint (repeatable;\n"
        "                       the first is searched first). A weight\n"
        "                       it holds with the same encoding, quantiser, options and shape is\n"
        "                       copied from it instead of quantised again\n"
        "      --changed FILE   with --reuse: checkpoint tensors (one name a line) that differ from\n"
        "                       the checkpoint the --reuse container was converted from. Every\n"
        "                       weight that reads one is converted; the rest are copied\n"
        "      --lora FILE|DIR  a PEFT adapter (.safetensors file or directory of them) merged\n"
        "                       into the checkpoint tensors it names before they are converted:\n"
        "                       W + strength * (alpha/r where the adapter states one) * B @ A.\n"
        "                       Repeatable, merged in the order given; each merged tensor joins\n"
        "                       the --changed set, so --reuse never copies an unmerged weight\n"
        "      --lora-strength F\n"
        "                       the strength of every --lora (default 1)\n"
        "      --lora-map FILE  with --lora: a JSON map for adapters trained on module names this\n"
        "                       checkpoint does not use. {\"strip\": [\"prefix\", ...], \"modules\":\n"
        "                       {\"<pat>\": [\"<target>\", ...]}}: a pattern holds at most one '*',\n"
        "                       which matches one run (the block index) and is substituted into\n"
        "                       every target; several targets mean the module is their concate-\n"
        "                       nation (a fused qkv), and the adapter's rows split along them\n"
        "                       in the order listed\n"
        "      --in-place       with --reuse: EXTEND that container instead of writing a new one.\n"
        "                       Its weights stay where they are; new ones are written past its\n"
        "                       end and the header last. Refused when a weight it holds would\n"
        "                       change or a new weight is an expert. <file>.pre-append keeps the\n"
        "                       old header and length\n"
        "      --gpus LIST      the cards (HIP device numbers, comma separated) the quantisers\n"
        "                       may run on, and large weights then run two at a time. A\n"
        "                       quantiser writes the same bytes on a card as on the host; with\n"
        "                       no quantiser plugin that has a device path, refused\n"
        "      --plan-only      declare, plan every weight's encoding, print the plan, and stop\n"
        "                       before writing anything\n"
        "      --report N       the worst N quantisation errors to print (default 20)\n"
        "  -v, --verbose\n"
        "  -h, --help\n");
}

/* Runs fn(a, m, &why) over rows [a, a + m) covering [0, n): in about 256 pieces across threads, or
 * as one call when !parallel. Returns RAD_E_FORMAT with the first failure's message in *why. */
template <class F>
int for_row_pieces(int64_t n, bool parallel, std::string* why, F fn) {
    const int64_t piece = parallel ? std::max<int64_t>(1, (n + 255) / 256) : n;
    const int64_t np = (n + piece - 1) / piece;
    int rc = RAD_OK;
#pragma omp parallel for schedule(dynamic, 1) if (parallel)
    for (int64_t p = 0; p < np; ++p) {
        const int64_t a = p * piece, m = std::min(piece, n - a);
        std::string w1;
        if (fn(a, m, &w1) != RAD_OK) {
#pragma omp critical(rad_convert_err)
            if (rc == RAD_OK) { rc = RAD_E_FORMAT; *why = w1; }
        }
    }
    return rc;
}

}  /* namespace */

int main(int argc, char** argv) {
    std::string in, out, arch_override, home = rad_home(), tokjson, imat_path, draft_dir, chat_tmpl;
    std::string tokjson2, config_dir;
    std::vector<std::string> add_files;
    std::vector<std::string> hierarchy = rad_hierarchy_from_env();
    std::vector<std::string> recipe_files, quant_flags;
    int64_t  max_tok = 8192;
    bool     max_tok_given = false;
    int64_t  max_ctx = 0;            /* 0: the checkpoint's training context */
    static const int kMaxSpecDefault = 4;
    int      max_spec = kMaxSpecDefault;
    bool     max_spec_given = false;
    int      report_n = 20;
    bool     verbose = false, plan_only = false, list_quant = false;
    std::string reuse_path, changed_path;
    std::vector<std::string> reuse_more;   /* --reuse given again: more containers to copy from */
    std::vector<std::string> lora_paths;
    double      lora_strength = 1.0;
    std::string lora_map;
    bool        in_place = false;
    int         kv_dtype = RAD_DT_INVALID;
    std::vector<std::pair<std::string, std::string>> set_kv;
    std::vector<int> gpus;           /* --gpus: the cards the quantisers may run on */

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "rad-convert: %s needs a value\n", what);
                                 std::exit(2); }
            return argv[++i];
        };
        auto num = [&](const char* what, int64_t lo, int64_t hi) {
            return rad_flag_int("rad-convert", what, next(what), lo, hi);
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "-o" || a == "--out") out = next("--out");
        else if (a == "--recipe")    recipe_files.push_back(next("--recipe"));
        else if (a == "--quant")     quant_flags.push_back(next("--quant"));
        else if (a == "--list-quantizers") list_quant = true;
        else if (a == "--arch")      arch_override = next("--arch");
        else if (a == "--home")      home = next("--home");
        else if (a == "--kernels")   hierarchy = rad_split_list(next("--kernels"));
        else if (a == "--tokenizer") tokjson = next("--tokenizer");
        else if (a == "--tokenizer2") tokjson2 = next("--tokenizer2");
        else if (a == "--add")       add_files.push_back(next("--add"));
        else if (a == "--config")    config_dir = next("--config");
        else if (a == "--imatrix")   imat_path = next("--imatrix");
        else if (a == "--chat-template") chat_tmpl = next("--chat-template");
        else if (a == "--set") {
            const std::string kv = next("--set");
            const size_t eq = kv.find('=');
            if (eq == std::string::npos || eq == 0) {
                std::fprintf(stderr, "rad-convert: --set wants KEY=VALUE, got '%s'\n", kv.c_str());
                return 2;
            }
            set_kv.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        }
        else if (a == "--max-tok")   { max_tok = num("--max-tok", 1, INT64_MAX); max_tok_given = true; }
        else if (a == "--max-ctx")   max_ctx = num("--max-ctx", 0, INT64_MAX);
        else if (a == "--max-spec")  { max_spec = (int)num("--max-spec", 0, INT32_MAX);
                                       max_spec_given = true; }
        else if (a == "--draft-model") draft_dir = next("--draft-model");
        else if (a == "--plan-only") plan_only = true;
        else if (a == "--report")    report_n = (int)num("--report", 0, INT32_MAX);
        else if (a == "--reuse") {
            const std::string r = next("--reuse");
            if (reuse_path.empty()) reuse_path = r;
            else reuse_more.push_back(r);
        }
        else if (a == "--changed")   changed_path = next("--changed");
        else if (a == "--lora")      lora_paths.push_back(next("--lora"));
        else if (a == "--lora-strength") {
            const char* v = next("--lora-strength");
            char* end = nullptr;
            lora_strength = std::strtod(v, &end);
            if (!end || *end || end == v) {
                std::fprintf(stderr, "rad-convert: --lora-strength wants a number, got '%s'\n",
                             v);
                return 2;
            }
        }
        else if (a == "--lora-map")  lora_map = next("--lora-map");
        else if (a == "--in-place")  in_place = true;
        else if (a == "--gpus") {
            for (const std::string& d : rad_split_list(next("--gpus")))
                gpus.push_back((int)rad_flag_int("rad-convert", "--gpus", d.c_str(), 0, 1023));
        }
        else if (a == "--kv-cache-dtype") {
            const std::string v = next("--kv-cache-dtype");
            if (v == "fp8")       kv_dtype = RAD_F8E4M3;
            else if (v == "bf16") kv_dtype = RAD_BF16;
            else { std::fprintf(stderr, "rad-convert: --kv-cache-dtype takes bf16 "
                                                         "or fp8, not '%s'\n", v.c_str());
                                    return 2; }
        }
        else if (a == "-v" || a == "--verbose") { verbose = true; log_set_level(Log::Debug); }
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "rad-convert: unrecognised argument '%s'. Try --help.\n",
                         a.c_str());
            return 2;
        } else if (in.empty()) in = a;
        else { std::fprintf(stderr, "rad-convert: more than one input given\n"); return 2; }
    }

    /* ---------------------------------------------------------------- --list-quantizers */
    if (list_quant) {
        Registry& reg = rad_tools_registry();
        std::vector<std::string> qdirs;
        for (const std::string& d : rad_home_dirs(home)) {
            struct stat st {};
            if (stat((d + "/quantizers").c_str(), &st) == 0 && S_ISDIR(st.st_mode))
                qdirs.push_back(d + "/quantizers");
        }
        if (qdirs.empty()) qdirs.push_back(rad_home_dirs(home).front() + "/quantizers");
        if (reg.load_dirs(qdirs, {}) < 0) return 1;
        for (const auto& qr : reg.quantizers()) {
            if (!qr.info) continue;
            std::printf("%s  (%s)\n    %s\n", qr.info->name, qr.plugin.c_str(), qr.info->doc);
            for (int k = 0; k < qr.info->n_options; ++k) {
                const RadQuantOption& o = qr.info->options[k];
                std::printf("    %-12s %-4s %s%s%s\n", o.key,
                            o.type == RAD_P_INT ? "int" : o.type == RAD_P_F64 ? "num" : "str",
                            o.doc ? o.doc : "", o.deflt ? "  [default " : "",
                            o.deflt ? (std::string(o.deflt) + "]").c_str() : "");
            }
        }
        std::printf("%s  (rad-convert)\n    the numbers the checkpoint ships, in another spelling: "
                    "an imported integer grid or an FP8 weight with a scale per row or per tensor\n"
                    "    codes        str  uN: a symmetric grid's codes unsigned, under a zero "
                    "point of 2^(N-1)\n"
                    "    zero         str  the zero plane's dtype (u4, u8)\n"
                    "    scale        str  the scale plane's dtype (f16, bf16, f32), rounded to "
                    "nearest\n"
                    "    block        str  RxC: an FP8 scale per row or per tensor repeated over "
                    "this block grid\n", kImportRule);
        return 0;
    }

    if (!changed_path.empty() && (reuse_path.empty() || in_place)) {
        std::fprintf(stderr, "rad-convert: --changed says which weights not to copy from the "
                             "container --reuse names, and needs --reuse without --in-place (a "
                             "weight cannot be replaced in place)\n");
        return 2;
    }
    if (!lora_paths.empty() && in_place) {
        std::fprintf(stderr, "rad-convert: --lora merges into the source, whose weights "
                             "--in-place copies from --reuse unchanged; convert to a new file\n");
        return 2;
    }
    if (!lora_map.empty() && lora_paths.empty()) {
        std::fprintf(stderr, "rad-convert: --lora-map renames the modules --lora merges; give "
                             "a --lora to go with it\n");
        return 2;
    }
    if (in_place) {
        if (reuse_path.empty()) {
            std::fprintf(stderr, "rad-convert: --in-place extends the container --reuse names\n");
            return 2;
        }
        if (out.empty()) out = reuse_path;
    }
    if (in.empty() || out.empty()) { usage(); return 2; }
    /* The output is opened with O_TRUNC, so it must not be the --reuse container unless
     * --in-place is given. */
    if (!reuse_path.empty()) {
        struct stat so{}, sr{};
        const bool same = ::stat(out.c_str(), &so) == 0 && ::stat(reuse_path.c_str(), &sr) == 0 &&
                          so.st_dev == sr.st_dev && so.st_ino == sr.st_ino;
        if (same != in_place) {
            std::fprintf(stderr, in_place
                ? "rad-convert: --in-place writes the --reuse container itself; -o names another\n"
                : "rad-convert: -o is the --reuse container, and writing it would truncate the "
                  "file its weights are read from. Pass --in-place to extend it instead\n");
            return 2;
        }
    }

    /* ---------------------------------------------------------------- the recipe */
    Recipe recipe;
    {
        std::string why;
        for (const std::string& q : quant_flags)
            if (recipe.add_flag(q, &why) < 0) { RAD_ERR("%s", why.c_str()); return 2; }
        for (const std::string& rf : recipe_files) {
            std::ifstream f(rf);
            if (!f) { RAD_ERR("--recipe %s: cannot read it", rf.c_str()); return 2; }
            std::stringstream ss;
            ss << f.rdbuf();
            if (recipe.add_text(ss.str(), rf, &why) < 0) { RAD_ERR("%s", why.c_str()); return 2; }
        }
    }

    /* ---------------------------------------------------------------- source */
    Checkpoint ck;
    if (!config_dir.empty()) ck.set_meta_dir(config_dir);
    /* A Laya checkpoint as laya's own loader downloads it carries no config.json of its own, only
     * encoder/config.json beside rl_agent_config.json: the geometry is the encoder's, the
     * architecture laya (laya_metadata reads the rest). */
    struct stat in_st{};
    const bool laya_dir = config_dir.empty() && stat(in.c_str(), &in_st) == 0 && S_ISDIR(in_st.st_mode) &&
                          !file_exists(in + "/config.json") && file_exists(in + "/rl_agent_config.json") &&
                          file_exists(in + "/encoder/config.json");
    if (laya_dir) ck.set_meta_dir(in + "/encoder");
    if (ck.open(in) < 0) return 1;
    if (laya_dir) ck.meta().arch = "laya";
    for (const std::string& f : add_files) {
        if (ck.add_file(f) < 0) return 1;
        RAD_INFO("added    %s", f.c_str());
    }
    MetaOwn& meta = ck.meta();
    /* Merge the drafter before declare: the architecture plugin declares a draft block only if
     * `draft.architectures` is present. */
    if (!draft_dir.empty()) {
        if (ck.add_draft(draft_dir, "draft.") < 0) return 1;
        RAD_INFO("drafter  %s (%s)", draft_dir.c_str(),
                 rad_meta_gets(&meta.m, "draft.architectures", "?"));
    }
    if (!arch_override.empty()) meta.arch = arch_override;
    /* --set is applied last, so it overrides the checkpoint's and drafter's config. Declare here
     * and in the engine both read RadModelMeta from the container header, so the value holds for
     * both. */
    for (const auto& kv : set_kv) meta.set(kv.first, kv.second);
    if (const char* dt = meta.get("decision.type"); dt && !std::strcmp(dt, "clef") &&
                                                     clef_metadata(in, config_dir, ck, meta) < 0)
        return 1;
    if (const char* dt = meta.get("decision.type");
        (meta.arch == "laya" || (dt && !std::strcmp(dt, "laya"))) &&
        laya_metadata(in, config_dir, meta, &tokjson) < 0)
        return 1;
    /* f16 to bf16 rounds, so it is a recipe's rule (spec §4.4). Every Laya checkpoint needs the same
     * rules, so they follow any the caller gave, and the container records them with the rest. */
    if (meta.arch == "laya") {
        std::string why;
        if (recipe.add_text(kLayaBf16, "laya's f16 to bf16 (built in)", &why) < 0) {
            RAD_ERR("%s", why.c_str());
            return 1;
        }
    }
    /* The quantisation descriptor (which selects the plugin pair) is derived from
     * quantization_config, so a --set of it re-derives the descriptor. E.g. a bf16 checkpoint
     * whose recipe writes Qwen's block fp8 targets qwen35_fp8 by setting what that release
     * states. */
    for (const auto& kv : set_kv)
        if (kv.first.rfind("quantization_config.", 0) == 0) {
            meta.quant = quant_descriptor(meta);
            meta.finish();
            break;
        }

    /* The --reuse container, read with pread, never through its mapping. */
    RadFile reuse;
    if (!reuse_path.empty() && reuse.open(reuse_path.c_str()) != RAD_OK) return 1;
    /* Further --reuse containers supply only weights the first lacks or holds converted otherwise:
     * a container assembled from several converts (a mixed-precision recipe). */
    std::deque<RadFile> reuse_extra;
    if (!reuse_more.empty() && in_place) {
        std::fprintf(stderr, "rad-convert: --in-place extends one container; give --reuse once\n");
        return 2;
    }
    for (const std::string& r : reuse_more) {
        reuse_extra.emplace_back();
        if (reuse_extra.back().open(r.c_str()) != RAD_OK) return 1;
    }
    /* --changed lists the checkpoint tensors that differ from the --reuse container's source
     * (e.g. a partial fine-tune). --reuse matches on how a weight was converted, not on its
     * source bytes, so without the list it would copy the stale weights. */
    std::unordered_set<std::string> changed, changed_read;
    if (!changed_path.empty()) {
        std::ifstream cf(changed_path);
        if (!cf) { RAD_ERR("--changed %s: cannot read it", changed_path.c_str()); return 1; }
        std::string line;
        while (std::getline(cf, line)) {
            const size_t a = line.find_first_not_of(" \t\r");
            const size_t z = line.find_last_not_of(" \t\r");
            if (a == std::string::npos || line[a] == '#') continue;
            changed.insert(line.substr(a, z - a + 1));
        }
        if (changed.empty()) {
            RAD_ERR("--changed %s names no tensor", changed_path.c_str());
            return 1;
        }
    }
    /* Adapters merge before anything reads the source, and a merged tensor joins the changed set:
     * --reuse matches a weight by how it was converted, not by its source bytes. */
    for (const std::string& lp : lora_paths) {
        std::vector<std::string> merged;
        if (ck.add_lora(lp, lora_strength, &merged,
                        lora_map.empty() ? nullptr : &lora_map) < 0)
            return 1;
        for (std::string& m : merged) changed.insert(std::move(m));
    }
    /* --in-place inherits the container's radiance.* keys (the convert-time choices its weights
     * were made under), except `radiance.encoder`, which this run derives. */
    int inherited = 0;
    if (in_place) {
        for (int64_t i = 0; i < reuse.meta_count(); ++i) {
            const RadFileKV& kv = reuse.meta_at(i);
            const std::string k = reuse.str(kv.key);
            if (k.rfind("radiance.", 0) != 0 || k == "radiance.encoder" || meta.has(k)) continue;
            std::string v;
            if (kv.type == RAD_P_STR)       v = reuse.str(kv.v.s);
            else if (kv.type == RAD_P_INT)  v = fmt("%lld", (long long)kv.v.i);
            else                            v = fmt("%.17g", kv.v.f);
            meta.kv(k, v);
            ++inherited;
        }
    }
    /* Always re-publish: the merges above may have moved the strings finish() pointed at. */
    meta.finish();
    if (inherited)
        RAD_INFO("--in-place: %d radiance.* key(s) carried over from %s", inherited,
                 reuse_path.c_str());
    if (reuse.is_open() && std::string(reuse.str(reuse.header().arch_id)) != meta.arch) {
        RAD_ERR("--reuse %s is a '%s' container and this run writes '%s'.", reuse_path.c_str(),
                reuse.str(reuse.header().arch_id), meta.arch.c_str());
        return 1;
    }

    RAD_INFO("source   %s (%zu tensors)", ck.path().c_str(), ck.tensors().size());
    RAD_INFO("arch     %s%s%s", meta.m.arch_id, meta.quant.empty() ? "" : ", quant ",
             meta.quant.c_str());

    /* ---------------------------------------------------------------- plugins */
    LoadedPlugins plugins;
    const bool imported = ck.imports() != Checkpoint::Import::None;
    if (!ck.import_refusal().empty())
        RAD_ERR("quantisation '%s' cannot be imported: %s", meta.quant.c_str(),
                ck.import_refusal().c_str());
    if (rad_tools_load_plugins(home, hierarchy, meta.arch, meta.quant, &plugins, imported) < 0) {
        RAD_ERR("rad-convert runs the architecture plugin's declare: its name map says which "
                "checkpoint tensors make each weight, and the kernels it resolves are what the "
                "recipe is checked against");
        return 1;
    }
    /* An imported quantisation, or block fp8 served by a family that claims no descriptor for it,
     * is recorded as the descriptor of the plugin serving it; the engine selects by exact match,
     * and each weight states its own encoding. */
    if (!plugins.arch_plugin.empty() && plugins.arch_quant != meta.quant) {
        meta.quant = plugins.arch_quant;
        meta.finish();
    }
    Registry& reg = rad_tools_registry();
    if (!gpus.empty()) {
        /* Every quantiser plugin that runs on a card is handed the list (rad_quant.h,
         * RadQuantDevicesFn); a plugin without the hook stays on the host. */
        int taken = 0;
        for (const Plugin& p : reg.plugins()) {
            if (p.kind != RAD_PLUGIN_QUANT || !p.handle) continue;
            const auto fn = (RadQuantDevicesFn)dlsym(p.handle, "rad_quant_devices");
            if (!fn) continue;
            const int st = fn(gpus.data(), (int)gpus.size());
            if (st == RAD_E_UNSUPPORTED) continue;
            if (st < 0) {
                RAD_ERR("--gpus: quantiser plugin '%s' refuses the list: %s", p.name.c_str(),
                        rad_strerror(st));
                return 2;
            }
            ++taken;
        }
        if (!taken) {
            RAD_ERR("--gpus: no quantiser plugin loaded here has a device path (a build without "
                    "ROCm has none)");
            return 2;
        }
    }
    std::deque<CompiledRule> compiled;
    std::map<const RecipeRule*, const CompiledRule*> by_rule;
    std::map<const RecipeRule*, ImportForm> import_forms;
    for (const RecipeRule& r : recipe.rules()) {
        std::string why;
        /* rad-convert's own rule: the numbers a checkpoint ships in another spelling
         * (ckpt_import_form) instead of a quantiser's. */
        if (r.quantizer == kImportRule) {
            ImportForm f;
            if (ckpt_import_form_parse(r.options, &f, &why) < 0) {
                RAD_ERR("%s: %s", r.origin.c_str(), why.c_str());
                return 2;
            }
            import_forms[&r] = f;
            continue;
        }
        compiled.emplace_back();
        if (compiled.back().build(r, reg, &why) < 0) { RAD_ERR("%s", why.c_str()); return 2; }
        by_rule[&r] = &compiled.back();
    }
    if (!recipe.empty()) RAD_INFO("recipe   %zu rule(s)", recipe.rules().size());

    ImatrixFile im;
    if (!imat_path.empty() && im.open(imat_path) < 0) return 1;

    /* ---------------------------------------------------------------- declare */
    RadBuildCtx ctx{};
    ctx.rank = 0;
    ctx.world_size = 1;
    ctx.max_tok = max_tok;
    /* The context bound is a declare input: an architecture whose graph depends on it (e.g.
     * Qwen3.8-Flash-Next's QSA, dense up to its indexer budget) may refuse some values. */
    ctx.max_seqs = 256;
    ctx.max_ctx = max_ctx > 0 ? max_ctx : meta.m.n_ctx_train;
    ctx.scope = "";
    RadArchProbe probe{};
    (void)rad_tools_probe(plugins, meta.m, &probe);
    /* Non-zero on purpose: the container carries every weight any deployment may need, including
     * the draft head. A drafter trained for one fixed depth reports it via rad_arch_probe, which
     * replaces the default but not a depth the command line states. */
    if (!max_spec_given && probe.draft_depth > 0 && probe.draft_depth_fixed) {
        max_spec = probe.draft_depth;
        RAD_INFO("declaring at max_spec %d, the only depth this drafter serves", max_spec);
    }
    ctx.max_spec = max_spec;
    /* A generator sizes its passes itself (RadArchProbe::gen_rows) and may refuse a smaller one;
     * declare at that size unless the command line names one. */
    if (!max_tok_given && probe.gen_rows > ctx.max_tok) ctx.max_tok = probe.gen_rows;
    /* Likewise declare the encoders, the vision tower at the engine's default patch bound. A
     * checkpoint states one either way round: a flat `vision_config.patch_size`, or only the
     * per-width tables with `vision_config.image_size` above them (Unlimited-OCR's DeepEncoder,
     * whose widths each carry their own patch_size). Either means its weights are here to map,
     * and the plugin declares the encoder only when `max_enc_patches` leaves it a whole view. */
    if (rad_meta_geti(&meta.m, "vision_config.patch_size", 0) > 0 ||
        rad_meta_geti(&meta.m, "vision_config.image_size", 0) > 0) {
        ctx.max_enc_patches = 16384;
        if (probe.enc_merge > 1) ctx.max_enc_patches -= ctx.max_enc_patches % probe.enc_merge;
    }
    ctx.max_audio_frames = probe.audio_frames > 0 ? probe.audio_frames : 0;

    /* Each weight's planned encoding, answered to declare via rad_weight_encoding: the recipe's
     * when a rule matches, else the checkpoint's own when trivial. Answers are memoised, except
     * not-found names, which declare may map later. */
    std::map<std::string, Logical> logical;
    auto plan_weight = [&](const std::string& name, const Program& prog, bool optional,
                           Logical** out) -> int {
        auto it = logical.find(name);
        if (it != logical.end()) { *out = &it->second; return it->second.status; }
        CkptWeight cw;
        std::string why;
        const int rc = ckpt_resolve(ck, prog, name, optional, &cw, &why);
        if (rc == RAD_E_NOTFOUND) return rc;
        Logical L;
        L.src = std::move(cw);
        const RecipeRule* r = rc < 0 ? nullptr : recipe.match(name);
        if (rc < 0) {
            L.status = rc;
            L.why = why;
        } else if (r && r->quantizer == kImportRule) {
            std::string w2;
            const int st = ckpt_import_form(&L.src, import_forms[r], &w2);
            if (st == RAD_OK) {
                L.import_rule = r;
                L.enc = L.src.enc;
            } else if (st == RAD_E_UNSUPPORTED && (L.src.trivial || L.src.imported)) {
                L.enc = L.src.enc;        /* declined: a plain weight, kept as it is */
            } else {
                L.status = st;
                L.why = r->origin + " (import): " + w2;
            }
        } else if (r) {
            L.rule = by_rule[r];
            const RadQuantWeight qw = quant_weight(L, nullptr);
            const int st = L.rule->q->encoding(L.rule->params.data(), (int)L.rule->params.size(),
                                               &qw, &L.enc);
            if (st == RAD_E_UNSUPPORTED &&
                (L.src.trivial || (L.src.imported && !L.src.ggml_blocks))) {
                /* declined (e.g. a norm under a matrix pattern): keep as is. Not a GGUF block
                 * tensor: keeping its planes would hide that the rule did not apply. */
                L.rule = nullptr;
                L.enc = L.src.enc;
            } else if (st != RAD_OK) {
                L.status = st;
                L.why = r->origin + " (" + r->quantizer + ") refuses it: " + rad_strerror(st);
            } else if (!rad_enc_valid(&L.enc)) {
                L.status = RAD_E_FORMAT;
                L.why = r->quantizer + " answered with a malformed encoding";
            }
        } else if (!L.src.trivial && !L.src.imported) {
            L.status = RAD_E_UNSUPPORTED;
            L.why = "no recipe rule matches it, and its checkpoint form is not one the engine "
                    "serves as it is: " + L.src.why;
        } else {
            L.enc = L.src.enc;
        }
        auto ins = logical.emplace(name, std::move(L)).first;
        *out = &ins->second;
        return ins->second.status;
    };
    auto provide = [&](const std::string& name, const Program& prog, WeightSource* ws) -> int {
        Logical* L = nullptr;
        const int st = plan_weight(name, prog, false, &L);
        if (st != RAD_OK) return st;
        ws->enc = L->enc;
        ws->rank = L->src.rank;
        for (int i = 0; i < RAD_MAX_RANK; ++i) ws->shape[i] = L->src.shape[i];
        return RAD_OK;
    };
    auto planning_errors = [&]() {
        int n = 0;
        for (const auto& [name, L] : logical)
            if (L.status != RAD_OK) {
                RAD_ERR("  '%s': %s", name.c_str(), L.why.c_str());
                ++n;
            }
        return n;
    };

    /* Ops resolved in no domain: nothing has checked their weights' planned encoding. Device-only
     * holes are placement options, not refusals. */
    auto dead_ops = [](const Program& p) {
        size_t n = 0;
        for (size_t i = 1; i < p.ops.size(); ++i) {
            bool any = false;
            for (const Band& bd : p.ops[i].bands)
                for (int d = 0; d < RAD_N_DOMAINS && !any; ++d) if (bd.dom[d]) any = true;
            if (!any) ++n;
        }
        return n;
    };
    /* The KV cache width is a serve-time choice the container does not record; only the kernels
     * differ (an fp8-cache attention may have no bf16 form). Without --kv-cache-dtype, check
     * against the first width at which every op resolves, else report the one with fewest holes. */
    const std::vector<int> widths = kv_dtype != RAD_DT_INVALID
                                        ? std::vector<int>{ kv_dtype }
                                        : std::vector<int>{ RAD_BF16, RAD_F8E4M3 };
    auto width_name = [](int dt) { return dt == RAD_F8E4M3 ? "fp8" : "bf16"; };
    Program prog;
    size_t dead = 0;
    for (size_t k = 0; k < widths.size(); ++k) {
        Program trial;
        ctx.kv_dtype = widths[k];
        if (rad_tools_declare(plugins, meta.m, ctx, &trial, {}, nullptr, provide,
                              /*instances=*/false) < 0) {
            if (planning_errors())
                RAD_ERR("the weights above could not be planned, and declare saw each of them as "
                        "the plain dtype it was declared at -- fix those first");
            return 1;
        }
        const size_t d = dead_ops(trial);
        if (k > 0 && d == 0)
            RAD_INFO("checked at --kv-cache-dtype %s: at %s, %zu op(s) have no kernel",
                     width_name(widths[k]), width_name(widths[0]), dead);
        if (k == 0 || d < dead) { prog = std::move(trial); dead = d; }
        if (dead == 0) break;
    }
    if (dead) {
        RAD_ERR("%zu op(s) resolved to no kernel in any domain at --kv-cache-dtype %s, so nothing "
                "can read the weights they take:", dead, width_name(prog.ctx.kv_dtype));
        for (const auto& m : prog.misses) RAD_ERR("    %s", m.line().c_str());
        return 1;
    }

    /* ---------------------------------------------------------------- the plan */
    /* Every logical weight a declaration takes, once, with the group the declarations give it. */
    int bad = 0;
    for (size_t h = 1; h < prog.weights.size(); ++h) {
        const WeightInfo& w = prog.weights[h];
        Logical* L = nullptr;
        const int st = plan_weight(w.source, prog, w.decl.optional != 0, &L);
        if (st == RAD_E_NOTFOUND && w.decl.optional) continue;
        if (st == RAD_E_NOTFOUND) {
            CkptWeight cw;
            std::string why;
            (void)ckpt_resolve(ck, prog, w.source, false, &cw, &why);
            RAD_ERR("'%s' is declared over '%s': %s", w.name.c_str(), w.source.c_str(),
                    why.c_str());
            ++bad;
            continue;
        }
        if (st != RAD_OK) continue;             /* named below, with the others */
        const RadWeightGroup& g = w.decl.group;
        if (!L->declared) {
            L->declared = true;
            L->layer = g.layer;
            L->expert = g.expert;
            L->slot = g.slot;
            L->first_decl = w.name;
        } else if (L->layer != g.layer || L->expert != g.expert || L->slot != g.slot) {
            RAD_ERR("'%s' and '%s' both take '%s' and put it in different movement groups "
                    "((%d, %d, %d) and (%d, %d, %d)); a weight is stored once",
                    L->first_decl.c_str(), w.name.c_str(), w.source.c_str(), L->layer, L->expert,
                    L->slot, g.layer, g.expert, g.slot);
            ++bad;
        }
        /* The encoding declare chose kernels for must be the one written. */
        if (!w.enc_known || !rad_enc_equal(&w.enc, &L->enc)) {
            RAD_ERR("'%s' was declared over an encoding of '%s' other than the one planned for "
                    "it", w.name.c_str(), w.source.c_str());
            ++bad;
        }
    }
    bad += planning_errors();
    if (bad) return 1;

    std::vector<Logical*> plan;
    for (auto& [name, L] : logical)
        if (L.declared && L.status == RAD_OK) plan.push_back(&L);
    std::sort(plan.begin(), plan.end(), plan_less);

    /* The checkpoint tensors no planned weight reads. Only declared weights are written, so a
     * tower or head the plugin does not recognise is left out without an error; this says so. A
     * tensor beside a read one (`X.weight_scale` beside `X.weight`) counts as read. */
    {
        auto stem = [](const std::string& n) { return n.substr(0, n.rfind('.')); };
        std::unordered_set<std::string> read, stems;
        for (const RadNameMap& nm : prog.name_map) {
            const auto it = nm.declared ? logical.find(nm.declared) : logical.end();
            if (it == logical.end() || !it->second.declared || it->second.status != RAD_OK) continue;
            for (int k = 0; k < nm.n_src; ++k)
                if (nm.src[k]) { read.insert(nm.src[k]); stems.insert(stem(nm.src[k])); }
        }
        std::vector<std::string> unread;
        for (const SrcTensor& t : ck.tensors())
            if (!read.count(t.name) && !stems.count(stem(t.name))) unread.push_back(t.name);
        if (!unread.empty()) {
            std::string list;
            for (size_t i = 0; i < unread.size() && i < 8; ++i) list += (i ? ", " : "") + unread[i];
            RAD_WARN("%zu of %zu checkpoint tensors are read by no declared weight and are not "
                     "written: %s%s", unread.size(), ck.tensors().size(), list.c_str(),
                     unread.size() > 8 ? ", ..." : "");
        }
    }

    {
        int64_t n_q = 0, n_kept = 0, n_imp = 0;
        for (const Logical* L : plan) (L->rule ? n_q : L->src.imported ? n_imp : n_kept)++;
        RAD_INFO("plan     %zu weight(s): %lld quantised by the recipe, %lld kept as the "
                 "checkpoint holds them%s", plan.size(), (long long)n_q, (long long)n_kept,
                 n_imp ? fmt(", %lld imported from its packed codes", (long long)n_imp).c_str()
                       : "");
        for (const auto& c : compiled) {
            int64_t n = 0;
            for (const Logical* L : plan) if (L->rule == &c) ++n;
            if (n == 0)
                RAD_WARN("%s: rule '%s %s' matches no weight", c.rule->origin.c_str(),
                         c.rule->pattern.c_str(), c.rule->quantizer.c_str());
        }
        /* The import rules: what each re-expressed, and every scale its dtype rounded. */
        for (const auto& [r, f] : import_forms) {
            int64_t n = 0, n_round = 0, rounded = 0;
            double max_rel = 0;
            for (const Logical* L : plan) {
                if (L->import_rule != r) continue;
                ++n;
                if (L->src.rounded) ++n_round;
                rounded += L->src.rounded;
                max_rel = std::max(max_rel, L->src.max_rel);
            }
            if (n == 0)
                RAD_WARN("%s: rule '%s %s' re-expresses no weight", r->origin.c_str(),
                         r->pattern.c_str(), r->quantizer.c_str());
            else if (rounded == 0)
                RAD_INFO("import   %s: %lld weight(s) re-expressed, every shipped value exact",
                         r->origin.c_str(), (long long)n);
            else
                RAD_INFO("import   %s: %lld weight(s) re-expressed; %lld scale(s) of %lld of them "
                         "rounded to %s, largest relative change %.3e", r->origin.c_str(),
                         (long long)n, (long long)rounded, (long long)n_round,
                         rad_dtype_name(f.scale), max_rel);
        }
    }

    if (plan_only) {
        int64_t bytes = 0;
        for (const Logical* L : plan) {
            int64_t b = 0;
            for (int k = 0; k < L->enc.n_planes; ++k)
                b += rad_enc_plane_bytes(&L->enc.plane[k], L->src.rows, L->src.cols);
            bytes += b;
            if (verbose)
                RAD_DEBUG("  %-52s %-34s %10s  %s%s%s", L->src.name.c_str(),
                          enc_name(L->enc).c_str(), humanb(b).c_str(),
                          L->rule ? L->rule->rule->quantizer.c_str()
                                  : L->import_rule ? kImportRule : "as is",
                          L->rule || L->import_rule ? " " : "",
                          L->rule ? L->rule->rule->options_text().c_str()
                                  : L->import_rule ? L->import_rule->options_text().c_str() : "");
        }
        RAD_INFO("--plan-only: about %s would be written; nothing was written to %s.",
                 humanb(bytes).c_str(), out.c_str());
        return 0;
    }

    /* ---------------------------------------------------------------- write */
    RadWriter w;
    RAD_TRY(in_place ? w.begin_append(out.c_str(), reuse) : w.begin(out.c_str()));

    w.set_arch(meta.arch);
    /* An extended container keeps its name; this run's checkpoint need not be its original. */
    w.set_model_name(in_place ? std::string(reuse.str(reuse.header().model_name))
                              : (meta.name.empty() ? meta.arch : meta.name));
    w.set_quant(meta.quant);
    w.set_recipe(in_place && recipe.empty() ? std::string(reuse.str(reuse.header().recipe))
                                            : recipe.text());
    /* Record the checkpoint as its last two path components (a Hugging Face repository id), not
     * a machine-local path. */
    std::string ck_id = ck.path();
    while (ck_id.size() > 1 && ck_id.back() == '/') ck_id.pop_back();
    if (const size_t a = ck_id.find_last_of('/'); a != std::string::npos && a > 0)
        if (const size_t b = ck_id.find_last_of('/', a - 1); b != std::string::npos)
            ck_id.erase(0, b + 1);
    w.set_created_by(in_place
        ? fmt("%s; extended in place by rad-convert " __DATE__ " from %s",
              reuse.str(reuse.header().created_by), ck_id.c_str())
        : fmt("rad-convert " __DATE__ " from %s", ck_id.c_str()));

    w.meta_i("n_layers",       meta.m.n_layers);
    w.meta_i("n_embd",         meta.m.n_embd);
    w.meta_i("n_head",         meta.m.n_head);
    w.meta_i("n_head_kv",      meta.m.n_head_kv);
    w.meta_i("head_dim",       meta.m.head_dim);
    w.meta_i("n_ff",           meta.m.n_ff);
    w.meta_i("n_vocab",        meta.m.n_vocab);
    w.meta_i("n_ctx_train",    meta.m.n_ctx_train);
    w.meta_i("n_expert",       meta.m.n_expert);
    w.meta_i("n_expert_used",  meta.m.n_expert_used);
    w.meta_i("n_expert_shared",meta.m.n_expert_shared);
    w.meta_f("rms_eps",        meta.m.rms_eps);
    w.meta_f("rope_theta",     meta.m.rope_theta);
    for (int i = 0; i < meta.m.n_kv; ++i) w.meta_s(meta.m.kv_key[i], meta.m.kv_val[i]);
    /* The encoders the container carries, by name (vision, audio or both); `--mm-max-patches auto`
     * keys on this. */
    {
        std::string enc;
        if (prog.encoder.modalities != 0) enc = prog.encoder_name;
        if (prog.audio_encoder.modalities != 0)
            enc += (enc.empty() ? "" : ",") + prog.audio_encoder_name;
        if (!enc.empty()) w.meta_s("radiance.encoder", enc.c_str());
    }

    /* ---------------------------------------------------------------- the weights
     *
     * All weights are added first in plan order, which fixes their plane offsets; plane writes are
     * then pwrites at their own offsets in any order, so writing is parallel (a Flash-Next
     * checkpoint is 49152 expert slices of a few MB each).
     *
     * Small weights run one per thread, in plan order (roughly file order); nested parallelism is
     * off, so a quantiser's OpenMP region stays on its calling thread. Large weights (n-gram
     * table, lm_head, attention linears) run afterwards one at a time (two with --gpus), with
     * decode, quantise and error each parallel over rows. */
#ifdef _OPENMP
    omp_set_max_active_levels(1);
#endif
    constexpr int64_t kChunk = kQuantBandBytes;     /* copies and joins move this much at once */
    constexpr int64_t kSmall = (int64_t)32 << 20;   /* f32 bytes one thread quantises alone */

    struct Job {
        Logical*            L = nullptr;
        int64_t             idx = -1;
        const RadFileEntry* re = nullptr;   /* copied from a --reuse container */
        RadFile*            rf = nullptr;   /* that container */
        int64_t             bytes = 0;      /* its planes */
    };
    std::vector<Job> jobs;
    jobs.reserve(plan.size());
    int64_t reused_inplace = 0, n_inplace = 0, n_changed = 0;

    for (Logical* Lp : plan) {
        Logical& L = *Lp;
        RadWriter::Weight ww;
        ww.name = L.src.name;
        const RecipeRule* rr = L.rule ? L.rule->rule : L.import_rule;
        ww.quantizer = rr ? rr->quantizer : "";
        ww.options = rr ? rr->options_text() : "";
        ww.enc = L.enc;
        ww.rank = L.src.rank;
        for (uint32_t i = 0; i < RAD_MAX_RANK; ++i) ww.shape[i] = L.src.shape[i];
        ww.layer = L.layer;
        ww.expert = L.expert;
        ww.slot = L.slot;
        int64_t plane_total = 0;
        for (int k = 0; k < L.enc.n_planes; ++k)
            plane_total += rad_enc_plane_bytes(&L.enc.plane[k], L.src.rows, L.src.cols);

        /* --- Reused: the same encoding, quantiser, options and shape in the --reuse container give
         * the same planes, assuming the same checkpoint (the operator's responsibility; neither
         * container records e.g. a Hessian's contents). */
        const RadFileEntry* re = reuse.is_open() ? reuse.find(ww.name) : nullptr;
        const bool reads_changed =
            !changed.empty() && ckpt_reads_any(L.src, changed, &changed_read);
        if (re && reads_changed) {
            re = nullptr;
            ++n_changed;
        }
        if (re) {
            /* The movement group is the declaration's either way (add_weight and add_existing
             * record ww's), so a plugin that regroups a weight still reuses its planes; only an
             * expert's layer, index and slot say where its planes sit in a stride table. */
            const bool expert = ww.expert >= 0 || re->expert >= 0;
            bool same = rad_enc_equal(&reuse.encoding(*re), &ww.enc) &&
                        ww.quantizer == reuse.str(re->quantizer) &&
                        ww.options == reuse.str(re->options) && re->rank == ww.rank &&
                        (!expert || (re->layer == ww.layer && re->expert == ww.expert &&
                                     re->slot == ww.slot));
            for (uint32_t i = 0; same && i < RAD_MAX_RANK; ++i)
                if (re->shape[i] != ww.shape[i]) same = false;
            if (in_place) {
                if (!same) {
                    RAD_ERR("--in-place: '%s' is in %s as %s by '%s %s', and this run would write "
                            "it as %s by '%s %s'. A weight cannot be replaced in place; convert "
                            "to a new file.", ww.name.c_str(), reuse_path.c_str(),
                            enc_name(reuse.encoding(*re)).c_str(), reuse.str(re->quantizer),
                            reuse.str(re->options), enc_name(ww.enc).c_str(),
                            ww.quantizer.c_str(), ww.options.c_str());
                    w.abort();
                    return 1;
                }
                if (w.add_existing(ww, reuse, *re) < 0) { w.abort(); return 1; }
                reused_inplace += (int64_t)re->bytes;
                ++n_inplace;
                continue;
            }
            if (!same) re = nullptr;
        }
        RadFile* rf = re ? &reuse : nullptr;
        for (size_t x = 0; !re && !reads_changed && x < reuse_extra.size(); ++x) {
            RadFile& R = reuse_extra[x];
            const RadFileEntry* e = R.find(ww.name);
            if (!e) continue;
            const bool expert = ww.expert >= 0 || e->expert >= 0;
            bool same = rad_enc_equal(&R.encoding(*e), &ww.enc) && ww.quantizer == R.str(e->quantizer) &&
                        ww.options == R.str(e->options) && e->rank == ww.rank &&
                        (!expert || (e->layer == ww.layer && e->expert == ww.expert &&
                                     e->slot == ww.slot));
            for (uint32_t i = 0; same && i < RAD_MAX_RANK; ++i)
                if (e->shape[i] != ww.shape[i]) same = false;
            if (same) { re = e; rf = &R; }
        }
        const int64_t idx = w.add_weight(ww);
        if (idx < 0) { w.abort(); return 1; }
        jobs.push_back(Job{ Lp, idx, re, rf, plane_total });
    }
    /* A --changed name no weight reads is a typo or an unconverted tensor; refuse it. */
    if (!changed.empty()) {
        std::vector<std::string> unread;
        for (const std::string& n : changed)
            if (!changed_read.count(n)) unread.push_back(n);
        if (!unread.empty()) {
            std::sort(unread.begin(), unread.end());
            RAD_ERR("the changed set (--changed and merged adapters) names %zu tensor(s) no "
                    "weight of this conversion reads, e.g. '%s'",
                    unread.size(), unread[0].c_str());
            w.abort();
            return 1;
        }
        RAD_INFO("--changed: %zu tensor(s) read by %lld weight(s), which are converted rather "
                 "than copied", changed.size(), (long long)n_changed);
    }

    /* ---------------------------------------------------------------- vocab */
    /* core/text interprets and serialises the declared chain; the writer places it and owns the
     * string blob holding token text. */
    struct WriterSink : StringSink {
        RadWriter* w;
        rad_stroff intern(std::string_view s) override { return w->intern_string(s); }
    } sink;
    sink.w = &w;

    VocabBuild vb;
    if (!tokjson.empty()) {
        if (parse_tokenizer_json(tokjson.c_str(), vb) < 0) { w.abort(); return 1; }
    } else if (ck.is_gguf()) {
        /* GGUF has no declared chain, only the pre-tokeniser name llama.cpp derived by hashing the
         * original tokenizer.json. core/text reconstructs from the name where it can and refuses
         * otherwise (then pass --tokenizer). */
        if (vocab_from_gguf(ck.gguf(), vb) < 0) {
            RAD_ERR("pass --tokenizer <tokenizer.json> to bake the declared chain instead of the "
                    "one a pre-tokeniser name implies.");
            w.abort();
            return 1;
        }
    } else if (meta_textless(ck.meta().get("radiance.text"))) {
        RAD_INFO("vocab    none: the model reads no text");
        vocab_textless(vb);
    } else if (file_exists(ck.tok_dir() + "/tokenizer.json")) {
        const std::string tj = ck.tok_dir() + "/tokenizer.json";
        RAD_INFO("vocab    %s", tj.c_str());
        if (parse_tokenizer_json(tj.c_str(), vb) < 0) { w.abort(); return 1; }
    } else {
        RAD_ERR("a safetensors checkpoint carries no vocab and there is no tokenizer.json beside "
                "it. Pass --tokenizer tokenizer.json.");
        w.abort();
        return 1;
    }
    /* The operator's template replaces whatever the checkpoint carried (or lacked), so a release
     * that ships none still bakes a container that serves /v1/chat/completions. */
    if (!chat_tmpl.empty() && read_chat_template_path(chat_tmpl.c_str(), vb) < 0) {
        w.abort();
        return 1;
    }
    /* Serialised at section offset 0; finish() rebases once the string blob is sized. */
    std::vector<uint8_t> vocab_blob;
    if (vocab_serialize(vb, sink, 0, vocab_blob) < 0) { w.abort(); return 1; }
    if (w.set_vocab_section(vocab_blob.data(), (int64_t)vocab_blob.size()) < 0) {
        w.abort();
        return 1;
    }
    if (!tokjson2.empty()) {
        VocabBuild vb2;
        RAD_INFO("vocab 2  %s", tokjson2.c_str());
        if (parse_tokenizer_json(tokjson2.c_str(), vb2) < 0) { w.abort(); return 1; }
        std::vector<uint8_t> blob2;
        if (vocab_serialize(vb2, sink, 0, blob2) < 0 ||
            w.set_aux_vocab_section(blob2.data(), (int64_t)blob2.size()) < 0) {
            w.abort();
            return 1;
        }
    }

    /* ---------------------------------------------------------------- expert profile */
    if (im.entries().size()) {
        /* Shares are normalised per layer, so experts are compared within a layer. */
        std::map<int, std::vector<std::pair<int, double>>> per_layer;
        for (const auto& e : im.entries()) {
            if (e.n_expert <= 1 || !e.counts) continue;
            const int32_t layer = layer_from_name(e.name);
            if (layer < 0) continue;
            for (int64_t x = 0; x < e.n_expert; ++x)
                per_layer[layer].emplace_back((int)x, im.count(e.name, x));
        }
        int64_t rows = 0;
        for (auto& [layer, v] : per_layer) {
            double tot = 0;
            for (auto& [e, cnt] : v) tot += cnt;
            if (tot <= 0) continue;
            for (auto& [e, cnt] : v) { w.add_profile(layer, e, (float)(cnt / tot)); ++rows; }
        }
        RAD_INFO("expert profile: %lld row(s) from %s", (long long)rows, imat_path.c_str());
        if (rows == 0)
            RAD_WARN("the imatrix carried no per-expert counts, so the planner will start cold. "
                     "A legacy-format imatrix cannot carry them at all.");
    }
    /* An extended container keeps the profile it had, unless this run brought its own. */
    if (in_place && imat_path.empty())
        for (int64_t i = 0; i < reuse.profile_count(); ++i) {
            const RadFileProfile& pr = reuse.profile()[i];
            w.add_profile(pr.layer, pr.expert, pr.share);
        }

    /* All tables are complete, so seal: the blob is then written in its final place rather than
     * spilled and copied at the end (RadWriter::seal). An extended container already writes in
     * place. */
    if (!in_place && w.seal() < 0) { w.abort(); return 1; }

    std::atomic<int64_t> written{0}, reused{reused_inplace}, n_reused{n_inplace};

    /* Read `n` rows from `r0` of a checkpoint weight as f32, the rows split across threads. */
    auto rows_f32 = [](const CkptWeight& src, int64_t r0, int64_t n, float* out,
                       std::string* why) -> int {
        return for_row_pieces(n, true, why, [&](int64_t a, int64_t m, std::string* w1) {
            return ckpt_rows_f32(src, r0 + a, m, out + a * src.cols, w1);
        });
    };

    /* One weight, start to finish. Its own buffers, so any thread may run it. */
    auto run_job = [&](const Job& j) -> int {
        Logical& L = *j.L;
        const std::string& name = L.src.name;
        const int64_t idx = j.idx;
        std::vector<uint8_t> stage;

        if (j.re) {
            for (int k = 0; k < L.enc.n_planes; ++k) {
                const RadFilePlane& fp = j.rf->plane(*j.re, k);
                for (int64_t off = 0; off < (int64_t)fp.bytes; off += kChunk) {
                    const int64_t n = std::min<int64_t>(kChunk, (int64_t)fp.bytes - off);
                    stage.resize((size_t)n);
                    if (j.rf->read_at(stage.data(), n, (int64_t)fp.offset + off) != RAD_OK ||
                        w.write_plane(idx, k, off, stage.data(), n) != RAD_OK)
                        return RAD_E_IO;
                }
                j.rf->drop_range((int64_t)fp.offset, (int64_t)fp.bytes);
            }
            written += j.bytes;
            reused += j.bytes;
            ++n_reused;
            if (verbose)
                RAD_DEBUG("  %-52s %-30s %10s reused", name.c_str(), enc_name(L.enc).c_str(),
                          humanb(j.bytes).c_str());
            return RAD_OK;
        }

        /* --- Kept as the checkpoint holds it, joining pieces where the name map does. A row join
         * is the pieces in sequence, written straight from the mapping; other joins are assembled
         * a chunk at a time. */
        if (!L.rule) {
            for (int k = 0; k < L.enc.n_planes; ++k) {
                const std::vector<const SrcTensor*>& src = k == 0 ? L.src.pieces : L.src.scales;
                const int64_t pb = w.plane_bytes(idx, k);
                /* An imported weight is re-laid row by row (ckpt_plane_rows); its bytes are not
                 * the plane's. */
                if (!L.src.imported && (L.src.concat_dim == 0 || src.size() == 1)) {
                    /* Pieces that share one scale tensor (shards under `X.weight_scale`) have
                     * that one plane, written once. */
                    const bool shared = k > 0 && std::all_of(src.begin(), src.end(),
                                                             [&](const SrcTensor* s) {
                                                                 return s == src[0];
                                                             });
                    int64_t off = 0;
                    for (const SrcTensor* s : src) {
                        if (w.write_plane(idx, k, off, s->data, s->bytes) != RAD_OK)
                            return RAD_E_IO;
                        off += s->bytes;
                        if (shared) break;
                    }
                    if (off != pb) {
                        RAD_ERR("'%s': its sources hold %lld bytes of plane '%s' and the "
                                "encoding sizes it at %lld", name.c_str(), (long long)off,
                                L.enc.plane[k].role, (long long)pb);
                        return RAD_E_SHAPE;
                    }
                } else {
                    int64_t pr = 0, pc = 0;
                    rad_enc_plane_dims(&L.enc.plane[k], L.src.rows, L.src.cols, &pr, &pc);
                    const int64_t rb = rad_enc_row_bytes(L.enc.plane[k].dtype, pc);
                    const int64_t step = std::max<int64_t>(1, kChunk / std::max<int64_t>(rb, 1));
                    for (int64_t r0 = 0; r0 < pr; r0 += step) {
                        const int64_t nr = std::min(step, pr - r0);
                        stage.resize((size_t)(nr * rb));
                        std::string why;
                        if (ckpt_plane_rows(L.src, k, r0, nr, stage.data(), &why) != RAD_OK) {
                            RAD_ERR("'%s': %s", name.c_str(), why.c_str());
                            return RAD_E_FORMAT;
                        }
                        if (w.write_plane(idx, k, r0 * rb, stage.data(), nr * rb) != RAD_OK)
                            return RAD_E_IO;
                    }
                }
            }
            written += j.bytes;
            if (verbose)
                RAD_DEBUG("  %-52s %-30s %10s as is", name.c_str(), enc_name(L.enc).c_str(),
                          humanb(j.bytes).c_str());
            return RAD_OK;
        }

        /* --- Quantised a band of rows at a time, by the loop core/format/quantise.h describes. */
        const CompiledRule& c = *L.rule;
        std::vector<float> imp;
        if (!imat_path.empty()) (void)im.importance_for(name, L.src.cols, &imp);
        const RadQuantWeight qw = quant_weight(L, imp.empty() ? nullptr : imp.data());
        const int64_t C = L.src.cols;
        /* Error is measured through the decoder readers use: the core's for plain and affine, the
         * quantiser's hook for its own scheme (core/format/encoding.h). Quantiser decoders need
         * not be reentrant, so those rows are decoded in one call. */
        bool core_decodes = true;
        const bool own_scheme = !rad_enc_is(&L.enc, "plain") && !rad_enc_is(&L.enc, "affine");
        std::vector<float> dec;
        double num = 0, den = 0, num_i = 0, den_i = 0;
        QuantIo io;
        io.read = [&](int64_t r0, int64_t n, float* out, std::string* why) {
            return rows_f32(L.src, r0, n, out, why);
        };
        io.write = [&](int k, int64_t off, const void* data, int64_t bytes) {
            return w.write_plane(idx, k, off, data, bytes);
        };
        /* The error over exactly these rows. */
        io.band = [&](int64_t, int64_t n, const float* ref, const void* const* pd) {
            if (!core_decodes) return;
            EncodedView v;
            std::string why;
            enc_view(L.enc, n, C, pd, &v);
            dec.resize((size_t)(n * C));
            const int drc = for_row_pieces(n, !own_scheme, &why,
                                           [&](int64_t a, int64_t m, std::string* w1) {
                return enc_decode_rows(v, a, m, dec.data() + a * C, w1);
            });
            if (drc != RAD_OK) {
                RAD_WARN("'%s': nothing decodes what %s wrote (%s); its error is not reported",
                         name.c_str(), c.q->name, why.c_str());
                core_decodes = false;
                return;
            }
            const float* pi = imp.empty() ? nullptr : imp.data();
            const int64_t cnt = n * C;
#pragma omp parallel for reduction(+ : num, den, num_i, den_i) schedule(static)
            for (int64_t i = 0; i < cnt; ++i) {
                const double d = (double)ref[(size_t)i] - dec[(size_t)i];
                const double x = ref[(size_t)i];
                num += d * d;
                den += x * x;
                if (pi) {
                    const double wj = pi[i % C];
                    num_i += wj * d * d;
                    den_i += wj * x * x;
                }
            }
        };
        {
            std::string why;
            const int st = quantise_weight(*c.q, c.params.data(), (int)c.params.size(), qw, L.enc,
                                           kQuantBandBytes, io, &why);
            if (st != RAD_OK) {
                RAD_ERR("'%s': %s", name.c_str(), why.c_str());
                return st;
            }
        }
        if (core_decodes && den > 0) L.err = std::sqrt(num / den);
        if (core_decodes && den_i > 0) L.err_imp = std::sqrt(num_i / den_i);
        written += j.bytes;
        if (verbose)
            RAD_DEBUG("  %-52s %-30s %10s %s rel %.3e", name.c_str(), enc_name(L.enc).c_str(),
                      humanb(j.bytes).c_str(), c.q->name, L.err);
        return RAD_OK;
    };

    std::vector<size_t> small, large;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const Logical& L = *jobs[i].L;
        const bool big = !jobs[i].re && L.rule && L.src.rows * L.src.cols * 4 > kSmall;
        (big ? large : small).push_back(i);
    }
    std::atomic<int> failed{RAD_OK};
    const auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < small.size(); ++i) {
        if (failed.load(std::memory_order_relaxed) != RAD_OK) continue;
        const int st = run_job(jobs[small[i]]);
        if (st < 0) {
            int ok = RAD_OK;
            failed.compare_exchange_strong(ok, st);
        }
    }
    {
        /* With the quantisers on cards (--gpus), large weights run two at a time, so one's host
         * work (its rows read, its Hessian factored, its error measured) overlaps the other's on
         * the card. Each runs on a thread of its own, whose OpenMP regions get a full team. */
        std::atomic<size_t> next{0};
        auto worker = [&] {
            for (size_t i; failed.load() == RAD_OK && (i = next.fetch_add(1)) < large.size();) {
                const int st = run_job(jobs[large[i]]);
                if (st < 0) {
                    int ok = RAD_OK;
                    failed.compare_exchange_strong(ok, st);
                }
            }
        };
        if (gpus.empty()) {
            worker();
        } else {
            std::thread side(worker);
            worker();
            side.join();
        }
    }
    if (failed.load() != RAD_OK) { w.abort(); return 1; }
    RAD_INFO("weights  %zu written in %.0f s: %zu a thread each, %zu across every thread",
             jobs.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                              .count(), small.size(), large.size());

    const auto tf = std::chrono::steady_clock::now();
    if (w.finish() < 0) return 1;
    RAD_INFO("finish   %.0f s", std::chrono::duration<double>(std::chrono::steady_clock::now() - tf)
                                    .count());

    /* ---------------------------------------------------------------- the report */
    RAD_INFO("wrote %s: %lld weight(s), %s of weights, %u vocab token(s)", out.c_str(),
             (long long)w.weight_count(), humanb(written.load()).c_str(),
             (unsigned)vb.tokens.size());
    if (in_place)
        RAD_INFO("--in-place: %lld weight(s), %s, kept where they were in %s; %lld added past "
                 "its end", (long long)n_reused.load(), humanb(reused.load()).c_str(),
                 reuse_path.c_str(), (long long)(w.weight_count() - n_reused.load()));
    else if (reuse.is_open())
        RAD_INFO("--reuse: %lld weight(s), %s, copied from %s; the other %lld were written",
                 (long long)n_reused.load(), humanb(reused.load()).c_str(), reuse_path.c_str(),
                 (long long)(w.weight_count() - n_reused.load()));

    std::vector<const Logical*> measured;
    for (const Logical* L : plan) if (L->err >= 0) measured.push_back(L);
    if (!measured.empty()) {
        const bool by_imp = !imat_path.empty();
        auto key = [&](const Logical* L) { return by_imp && L->err_imp >= 0 ? L->err_imp : L->err; };
        std::sort(measured.begin(), measured.end(),
                  [&](const Logical* a, const Logical* b) { return key(a) > key(b); });
        std::map<std::string, std::pair<int64_t, double>> per_rule;
        for (const Logical* L : measured) {
            auto& pr = per_rule[L->rule->rule->origin];
            ++pr.first;
            pr.second += L->err;
        }
        RAD_INFO("quantisation error, relative Frobenius norm%s:",
                 by_imp ? " (imatrix-weighted where the imatrix covers the weight)" : "");
        for (const auto& [origin, pr] : per_rule)
            RAD_INFO("  %-24s %6lld weight(s), mean %.3e", origin.c_str(), (long long)pr.first,
                     pr.second / (double)pr.first);
        const size_t n = verbose ? measured.size()
                                 : std::min<size_t>(measured.size(), (size_t)std::max(report_n, 0));
        if (n) RAD_INFO("  worst %zu:", n);
        for (size_t i = 0; i < n; ++i) {
            const Logical* L = measured[i];
            if (L->err_imp >= 0)
                RAD_INFO("    %.3e  (%.3e weighted)  %-52s %s", L->err, L->err_imp,
                         L->src.name.c_str(), enc_name(L->enc).c_str());
            else
                RAD_INFO("    %.3e  %-52s %s", L->err, L->src.name.c_str(),
                         enc_name(L->enc).c_str());
        }
    }
    return 0;
}
