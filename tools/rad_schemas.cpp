/* rad-schemas -- print, and diff, the op schemas two or more kernel plugins declare.
 *
 * The first plugin in hierarchy order to declare an op fixes its schema; a later plugin declaring
 * it differently is refused at load (spec §2.3), because arguments are positional. The loader names
 * one op at a time; this prints every disagreement at once, so a new plugin can be checked against
 * the vocabulary before it is loaded beside another.
 *
 *     rad-schemas libref.so                 print one plugin's vocabulary
 *     rad-schemas libr4d.so libref.so          diff them, in hierarchy order
 */
#include "rad_abi.h"
#include "rad_abi_layout.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

struct Loaded {
    std::string name, path;
    void* h = nullptr;
    std::map<std::string, const RadOpSchema*> ops;
};

const char* ptype(int t) { return t == RAD_P_INT ? "int" : t == RAD_P_STR ? "str"
                                : t == RAD_P_F64 ? "f64" : "?"; }
const char* preq(int r)  { return r == RAD_REQUIRED ? "required"
                                : r == RAD_DERIVED  ? "derived" : "optional"; }
const char* orole(int r) { return r == RAD_OPD_IN ? "in" : r == RAD_OPD_OUT ? "out"
                                : r == RAD_OPD_INOUT ? "inout"
                                : r == RAD_OPD_WTAB ? "wtab" : "weight"; }

std::string params_of(const RadOpSchema* s) {
    std::string o;
    for (int i = 0; i < s->n_params; ++i) {
        if (i) o += ' ';
        o += s->params[i].key;
        o += ':';
        o += ptype(s->params[i].type);
        if (s->params[i].required != RAD_REQUIRED) { o += '/'; o += preq(s->params[i].required); }
    }
    return o;
}

std::string operands_of(const RadOpSchema* s) {
    std::string o;
    for (int i = 0; i < s->n_operands; ++i) {
        if (i) o += ' ';
        o += s->operands[i].name;
        o += ':';
        o += orole(s->operands[i].role);
        if (s->operands[i].optional) o += "?";
    }
    return o;
}

int load(const char* path, Loaded* out) {
    out->path = path;
    out->h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!out->h) { fprintf(stderr, "dlopen(%s): %s\n", path, dlerror()); return -1; }

    auto abi = (uint32_t (*)(void))dlsym(out->h, "rad_plugin_abi_version");
    auto inf = (const RadPluginInfo* (*)(void))dlsym(out->h, "rad_plugin_info");
    auto cnt = (int (*)(void))dlsym(out->h, "rad_kernel_schema_count");
    auto at  = (const RadOpSchema* (*)(int))dlsym(out->h, "rad_kernel_schema_at");
    if (!abi || !cnt || !at) { fprintf(stderr, "%s: not a kernel plugin\n", path); return -1; }
    if (abi() != RAD_ABI_VERSION) {
        fprintf(stderr, "%s: ABI %u, this build is %u\n", path, abi(), RAD_ABI_VERSION);
        return -1;
    }
    /* One version can name two layouts; the engine refuses a mismatched fingerprint, and so does
     * this tool, which reads the same structs. */
    auto layout = (uint64_t (*)(void))dlsym(out->h, "rad_plugin_abi_layout");
    if (!layout || layout() != rad_abi_layout()) {
        fprintf(stderr, "%s: built against another layout of ABI %u, or exports no "
                        "rad_plugin_abi_layout\n", path, RAD_ABI_VERSION);
        return -1;
    }
    out->name = inf && inf()->name ? inf()->name : path;
    for (int i = 0, n = cnt(); i < n; ++i) {
        const RadOpSchema* s = at(i);
        if (s && s->op) out->ops[s->op] = s;
    }
    return 0;
}

}  /* namespace */

static const char kUsage[] =
    "usage: rad-schemas <plugin.so> [more.so ...]\n\n"
    "One plugin prints its vocabulary. Two or more are diffed in hierarchy order:\n"
    "the first to declare an op fixes its schema, and the loader refuses any plugin\n"
    "that disagrees (spec §2.3).\n";

int main(int argc, char** argv) {
    if (argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        fputs(kUsage, stdout);
        return 0;
    }
    if (argc < 2) {
        fputs(kUsage, stderr);
        return 2;
    }

    std::vector<Loaded> pl((size_t)(argc - 1));
    for (int i = 1; i < argc; ++i)
        if (load(argv[i], &pl[(size_t)i - 1]) < 0) return 1;

    if (pl.size() == 1) {
        printf("%s -- %zu ops\n\n", pl[0].name.c_str(), pl[0].ops.size());
        for (auto& [op, s] : pl[0].ops) {
            printf("  %s\n", op.c_str());
            printf("      params   %s\n", params_of(s).c_str());
            printf("      operands %s\n", operands_of(s).c_str());
        }
        return 0;
    }

    /* Every op any plugin declares. */
    std::set<std::string> all;
    for (auto& p : pl) for (auto& [op, s] : p.ops) all.insert(op);

    int conflicts = 0, shared = 0;
    for (const std::string& op : all) {
        std::vector<const Loaded*> have;
        for (auto& p : pl) if (p.ops.count(op)) have.push_back(&p);
        if (have.size() < 2) continue;
        ++shared;

        std::vector<std::pair<std::string, std::string>> desc;   /* (params, operands) */
        for (auto* p : have)
            desc.emplace_back(params_of(p->ops.at(op)), operands_of(p->ops.at(op)));
        bool differs = false;
        for (const auto& d : desc) differs = differs || d != desc[0];
        if (!differs) continue;

        ++conflicts;
        printf("%s\n", op.c_str());
        for (size_t i = 0; i < have.size(); ++i) {
            const char* mark = desc[i] == desc[0] ? " " : "!";
            printf("  %-16s %s params   %s\n", have[i]->name.c_str(), mark, desc[i].first.c_str());
            printf("  %-16s %s operands %s\n", "", mark, desc[i].second.c_str());
        }
        printf("\n");
    }

    printf("%d op(s) declared by more than one plugin, %d disagree.\n", shared, conflicts);
    if (conflicts)
        printf("The loader refuses the whole plugin on any one of these, so none of its rows\n"
               "is reachable until they agree. docs/OPS.md is the contract.\n");
    return conflicts ? 1 : 0;
}
