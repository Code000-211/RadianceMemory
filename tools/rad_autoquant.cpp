/* rad-autoquant -- a --kv-recipe from measurement: each attention layer's KV layout chosen so the
 * model's output moves least for an average bit budget (docs/TOOLS.md, "rad-autoquant").
 *
 *   rad-autoquant --kld-ref REF --layers 3,7,... --bits 6 -o model.kvrecipe -- <radiance flags>
 *
 * REF is a KL-mode reference (`radiance --kld-record`) recorded by the same model with an
 * unquantised (bf16) cache, so a run's KL is what its cache costs and nothing else. The driver runs
 * the engine's KL mode (`--kld-ref`) once with every listed layer in fp8, the base, and once for
 * each layer and each narrower layout with that layer alone narrowed. A layer's cost in a layout
 * is the run's mean KL above the base's; a run under it is measurement noise and counts as no
 * cost. With the costs taken as adding up across layers, kv_autoquant_solve (core/kv_autoquant.h)
 * picks the assignment of least summed cost within the budget, and the recipe is written with
 * the measurements in its comments. The candidate set is every layout a recipe can name
 * (core/config.cpp, kKvLayouts), so a layout added there is calibrated without a change here.
 *
 * Every run is a child `radiance` with the flags after `--`, plus the cache and KL flags this tool
 * sets; its recipe, report and log stay in the work directory, so a surprising number can be
 * looked at afterwards.
 *
 * Long context is the point: a narrowed layer's error is paid where attention reaches far back,
 * which a short document never asks for, so a reference whose documents are shorter than
 * --min-context tokens (default 32768) is refused unless the bound is lowered explicitly.
 */
#include "kv_autoquant.h"
#include "rad_core.h"

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace rad;

namespace {

constexpr int64_t kMinContext = 32768;
/* A group of 32 K and 32 V elements: the unit kv_layout_group_bits() counts in. */
constexpr int kGroupElems = 64;

void usage() {
    std::printf(
        "rad-autoquant -- each attention layer's KV layout from measured KL under a bit budget\n"
        "\n"
        "usage: rad-autoquant --kld-ref DIR --layers N,N,... --bits B -o FILE [options]\n"
        "                     -- <radiance flags: --model and the serve flags>\n"
        "\n"
        "  --kld-ref DIR       a KL reference recorded by this model with a bf16 cache\n"
        "  --layers N,N,...    the model's attention layers, as the checkpoint numbers them\n"
        "  --bits B            the average bits a cached K or V element may cost, scales\n"
        "                      included (fp8 8, fp8_v4 6.25, k6_v4 5.5, k4h_v4 4.5,\n"
        "                      k4h_v2h 3.5)\n"
        "  -o FILE             the recipe to write (--kv-recipe FILE)\n"
        "  --layouts L,L,...   the layouts to choose from (default every one); fp8, the base,\n"
        "                      is always among them\n"
        "  --work DIR          each run's recipe, report and log (default FILE.work)\n"
        "  --min-context N     refuse a reference with a document under N tokens (default\n"
        "                      %lld); 0 measures any\n"
        "  --radiance PATH     the engine (default: radiance beside this tool)\n"
        "  -h, --help\n",
        (long long)kMinContext);
}

[[gnu::format(printf, 1, 2)]] int fail(const char* f, ...) {
    std::fputs("rad-autoquant: ", stderr);
    va_list ap;
    va_start(ap, f);
    std::vfprintf(stderr, f, ap);
    va_end(ap);
    std::fputc('\n', stderr);
    return 1;
}

bool parse_int(const std::string& s, int64_t lo, int64_t hi, int64_t* v) {
    if (s.empty() || s[0] == '-' || s[0] == '+') return false;
    char* end = nullptr;
    errno = 0;
    const long long x = std::strtoll(s.c_str(), &end, 10);
    if (errno || !end || *end || x < lo || x > hi) return false;
    *v = x;
    return true;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string w;
    std::istringstream in(s);
    while (std::getline(in, w, sep)) out.push_back(w);
    return out;
}

std::string tail(const std::string& path, int lines) {
    std::ifstream f(path, std::ios::binary);
    const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t at = s.size();
    for (int n = 0; at > 0 && n <= lines; --at)
        if (s[at - 1] == '\n') ++n;
    return s.substr(at);
}

std::string self_dir() {
    char buf[PATH_MAX];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n <= 0) return ".";
    std::string p(buf, (size_t)n);
    const size_t slash = p.rfind('/');
    return slash == std::string::npos ? "." : p.substr(0, slash);
}

/* The flags this tool sets on every run; given after `--` they would contradict it. */
const char* const kReserved[] = { "--kv-cache-dtype", "--kv-recipe", "--kld-ref", "--kld-out",
                                  "--kld-record", "--kld-corpus", "--kld-top" };

struct Run {
    std::string radiance, ref, work;
    std::vector<std::string> engine;
    std::vector<int> layers;

    /* The mean KL of the engine with layer layers[i] in layout[i], or a negative number, the
     * refusal printed, when the run did not report one. */
    double mean_kl(const std::string& name, const std::vector<int>& layout) const {
        const std::string base = work + "/" + name;
        const std::string recipe = base + ".kvrecipe", out = base + ".json", log = base + ".log";
        {
            std::ofstream f(recipe);
            for (size_t i = 0; i < layers.size(); ++i)
                f << "layer " << layers[i] << " " << kv_layout_name(layout[i]) << "\n";
            if (!f) { fail("cannot write %s", recipe.c_str()); return -1; }
        }
        std::remove(out.c_str());
        std::vector<std::string> a = { radiance };
        a.insert(a.end(), engine.begin(), engine.end());
        for (const char* s : { "--kv-cache-dtype", "fp8", "--kv-recipe" }) a.push_back(s);
        a.push_back(recipe);
        a.push_back("--kld-ref");
        a.push_back(ref);
        a.push_back("--kld-out");
        a.push_back(out);
        std::vector<char*> av;
        for (std::string& s : a) av.push_back(s.data());
        av.push_back(nullptr);

        const int fd = ::open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0) { fail("cannot write %s: %s", log.c_str(), std::strerror(errno)); return -1; }
        std::fflush(nullptr);
        const pid_t pid = ::fork();
        if (pid == 0) {
            ::dup2(fd, 1);
            ::dup2(fd, 2);
            ::execv(av[0], av.data());
            std::fprintf(stderr, "rad-autoquant: cannot run %s: %s\n", av[0], std::strerror(errno));
            ::_exit(127);
        }
        ::close(fd);
        if (pid < 0) { fail("fork: %s", std::strerror(errno)); return -1; }
        int st = 0;
        while (::waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
        const bool ok = WIFEXITED(st) && WEXITSTATUS(st) == 0;

        json rep;
        if (ok) {
            std::ifstream f(out);
            rep = json::parse(f, nullptr, false);
        }
        const json* m = nullptr;
        if (rep.is_object() && rep.contains("all") && rep["all"].is_object() &&
            rep["all"].contains("kld") && rep["all"]["kld"].contains("mean") &&
            rep["all"]["kld"]["mean"].is_number())
            m = &rep["all"]["kld"]["mean"];
        if (!m || !std::isfinite(m->get<double>()) || m->get<double>() < 0) {
            fail("run %s %s; its log, %s, ends:\n%s", name.c_str(),
                 ok ? "reported no mean KL" : "failed", log.c_str(), tail(log, 15).c_str());
            return -1;
        }
        return m->get<double>();
    }
};

}  /* namespace */

int main(int argc, char** argv) {
    std::string ref, out, work, radiance, layers_arg, layouts_arg, bits_arg;
    int64_t min_context = kMinContext;
    std::vector<std::string> engine;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](std::string* v) {
            if (i + 1 >= argc) return false;
            *v = argv[++i];
            return true;
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        if (a == "--") { engine.assign(argv + i + 1, argv + argc); break; }
        std::string v;
        if (a == "--kld-ref" && val(&ref)) continue;
        if ((a == "-o" || a == "--out") && val(&out)) continue;
        if (a == "--work" && val(&work)) continue;
        if (a == "--radiance" && val(&radiance)) continue;
        if (a == "--layers" && val(&layers_arg)) continue;
        if (a == "--layouts" && val(&layouts_arg)) continue;
        if (a == "--bits" && val(&bits_arg)) continue;
        if (a == "--min-context" && val(&v)) {
            if (!parse_int(v, 0, INT64_MAX, &min_context))
                return fail("--min-context '%s' is not a token count", v.c_str());
            continue;
        }
        usage();
        return fail("'%s' is not an option here, or lacks its value; the engine's flags go "
                    "after --", a.c_str());
    }
    if (ref.empty() || out.empty() || layers_arg.empty() || bits_arg.empty()) {
        usage();
        return fail("--kld-ref, --layers, --bits and -o are required");
    }
    if (engine.empty()) return fail("no engine flags after --; the runs need at least --model");
    for (const std::string& e : engine)
        for (const char* r : kReserved)
            if (e == r || e.rfind(std::string(r) + "=", 0) == 0)
                return fail("%s is set by rad-autoquant on every run; leave it out of the "
                            "engine's flags", r);
    if (work.empty()) work = out + ".work";
    if (radiance.empty()) radiance = self_dir() + "/radiance";

    /* The layers, as the recipe names them. The plugin refuses a list that is not exactly the
     * model's attention layers, on the first run. */
    std::vector<int> layers;
    std::set<int64_t> seen;
    for (const std::string& s : split(layers_arg, ',')) {
        int64_t l = 0;
        if (!parse_int(s, 0, 4095, &l)) return fail("--layers: '%s' is not a layer number", s.c_str());
        if (!seen.insert(l).second) return fail("--layers names layer %lld twice", (long long)l);
        layers.push_back((int)l);
    }
    /* The candidates, widest first so that a tie keeps the wider layout; fp8 is the base. */
    std::vector<int> layouts;
    std::set<int> want = { RAD_KVQ_NONE };
    if (layouts_arg.empty()) {
        for (int q : kv_layouts()) want.insert(q);
    } else {
        for (const std::string& s : split(layouts_arg, ',')) {
            const int q = kv_layout_of(s.c_str());
            if (q < 0) return fail("--layouts: '%s' is not a KV layout", s.c_str());
            want.insert(q);
        }
    }
    for (int q : kv_layouts())
        if (want.count(q)) layouts.push_back(q);

    char* end = nullptr;
    const double bits = std::strtod(bits_arg.c_str(), &end);
    if (!end || *end || !(bits > 0) || !std::isfinite(bits))
        return fail("--bits '%s' is not a number of bits", bits_arg.c_str());
    /* The budget in bits a layer's group, summed; the epsilon keeps 6.25 * 64 at 400. */
    const int64_t budget = (int64_t)std::floor(bits * kGroupElems * (double)layers.size() + 1e-6);
    int narrowest = INT_MAX;
    for (int q : layouts) narrowest = std::min(narrowest, kv_layout_group_bits(q));
    if (budget < (int64_t)narrowest * (int64_t)layers.size())
        return fail("--bits %s is under the narrowest layout's %.2f", bits_arg.c_str(),
                    (double)narrowest / kGroupElems);

    /* The reference's documents, against the long-context bound. */
    json manifest;
    {
        std::ifstream f(ref + "/kld.json");
        if (!f) return fail("--kld-ref %s has no kld.json; record one with radiance --kld-record",
                            ref.c_str());
        manifest = json::parse(f, nullptr, false);
    }
    if (!manifest.is_object() || !manifest.contains("docs") || !manifest["docs"].is_array() ||
        manifest["docs"].empty())
        return fail("--kld-ref %s/kld.json lists no documents", ref.c_str());
    int64_t shortest = INT64_MAX;
    size_t shortest_at = 0;
    for (size_t d = 0; d < manifest["docs"].size(); ++d) {
        const json& j = manifest["docs"][d];
        if (!j.is_object() || !j.contains("tokens") || !j["tokens"].is_number_integer())
            return fail("--kld-ref %s/kld.json: document %zu has no token count", ref.c_str(), d);
        if (j["tokens"].get<int64_t>() < shortest) { shortest = j["tokens"]; shortest_at = d; }
    }
    if (shortest < min_context)
        return fail("--kld-ref %s: document %zu holds %lld tokens, under --min-context %lld. A "
                    "narrowed layer's error is paid where attention reaches far back, which a "
                    "short document never asks for, so its KL would understate the cost; record "
                    "the reference over documents of at least %lld tokens, or lower "
                    "--min-context to measure anyway",
                    ref.c_str(), shortest_at, (long long)shortest, (long long)min_context,
                    (long long)min_context);

    if (::mkdir(work.c_str(), 0755) != 0 && errno != EEXIST)
        return fail("cannot create %s: %s", work.c_str(), std::strerror(errno));
    const Run run{ radiance, ref, work, engine, layers };
    const size_t n_runs = 1 + layers.size() * (layouts.size() - 1);
    std::printf("rad-autoquant: %zu layers, %zu layouts, %zu runs over %zu documents (shortest "
                "%lld tokens); work in %s\n", layers.size(), layouts.size(), n_runs,
                manifest["docs"].size(), (long long)shortest, work.c_str());
    std::fflush(stdout);

    const double base = run.mean_kl("base", std::vector<int>(layers.size(), RAD_KVQ_NONE));
    if (base < 0) return 1;
    std::printf("rad-autoquant: every layer fp8: mean KL %.6f\n", base);
    std::fflush(stdout);
    /* cost[i][c]: layer layers[i] alone in layouts[c], above the base. */
    std::vector<std::vector<KvCandidate>> cand(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
        for (int q : layouts) {
            double d = 0;
            if (q != RAD_KVQ_NONE) {
                std::vector<int> lay(layers.size(), RAD_KVQ_NONE);
                lay[i] = q;
                const double kl = run.mean_kl("layer" + std::to_string(layers[i]) + "_" +
                                              kv_layout_name(q), lay);
                if (kl < 0) return 1;
                d = std::max(0.0, kl - base);
                std::printf("rad-autoquant: layer %d %s: mean KL %.6f (+%.3e)\n", layers[i],
                            kv_layout_name(q), kl, d);
                std::fflush(stdout);
            }
            cand[i].push_back({ q, kv_layout_group_bits(q), d });
        }

    KvAssignment pick;
    if (kv_autoquant_solve(cand, budget, &pick) != RAD_OK)
        return fail("no assignment fits %s bits", bits_arg.c_str());
    const double avg = (double)pick.bits / kGroupElems / (double)layers.size();

    std::ostringstream r;
    r << "# rad-autoquant: a KV layout per attention layer within " << bits_arg
      << " bits an element\n";
    r << "# reference " << ref << ", " << manifest["docs"].size()
      << " documents, the shortest " << shortest << " tokens\n";
    char line[256];
    std::snprintf(line, sizeof line, "# every layer fp8: mean KL %.6f\n", base);
    r << line;
    r << "# each layer alone narrowed, the rest fp8: mean KL above that\n#   layer";
    for (int q : layouts)
        if (q != RAD_KVQ_NONE) { std::snprintf(line, sizeof line, " %12s", kv_layout_name(q)); r << line; }
    r << "\n";
    for (size_t i = 0; i < layers.size(); ++i) {
        std::snprintf(line, sizeof line, "#   %5d", layers[i]);
        r << line;
        for (size_t c = 0; c < layouts.size(); ++c)
            if (layouts[c] != RAD_KVQ_NONE) {
                std::snprintf(line, sizeof line, " %12.3e", cand[i][c].kl);
                r << line;
            }
        r << "\n";
    }
    std::snprintf(line, sizeof line,
                  "# chosen: %.4f bits an element, predicted mean KL %.6f (the layers' costs "
                  "summed onto the base)\n", avg, base + pick.kl);
    r << line;
    for (size_t i = 0; i < layers.size(); ++i)
        r << "layer " << layers[i] << " " << kv_layout_name(cand[i][(size_t)pick.pick[i]].kvq)
          << "\n";

    /* What the server reads: refused here rather than at its startup. */
    std::vector<int32_t> check;
    if (kv_recipe_parse(r.str(), out.c_str(), &check) != RAD_OK)
        return fail("the recipe for %s does not parse; this is a bug", out.c_str());
    const std::string tmp = out + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        f << r.str();
        if (!f) return fail("cannot write %s", tmp.c_str());
    }
    if (std::rename(tmp.c_str(), out.c_str()) != 0)
        return fail("cannot write %s: %s", out.c_str(), std::strerror(errno));
    std::printf("rad-autoquant: %s: %.4f bits an element, predicted mean KL %.6f\n", out.c_str(),
                avg, base + pick.kl);
    return 0;
}
