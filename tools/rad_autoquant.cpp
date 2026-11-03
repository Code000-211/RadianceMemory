/* rad-autoquant -- a --kv-recipe from measurement: each attention layer's KV layout chosen so the
 * model's output moves least for an average bit budget, or so the cache is smallest for a KL limit
 * (docs/TOOLS.md, "rad-autoquant").
 *
 *   rad-autoquant --kld-ref REF --layers 3,7,... --bits 6 -o model.kvrecipe -- <radiance flags>
 *   rad-autoquant --kld-ref REF --layers 3,7,... --max-kl 0.035 -o model.kvrecipe -- <flags>
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
 * --max-kl asks the converse: the fewest bits whose mean KL stays within a limit. The per-layer
 * costs do not add up exactly when layouts are mixed (an assignment predicted within the limit can
 * measure above it), so in this mode every solved assignment is measured before it is written.
 * One that measures above the limit tightens the predicted limit by how far the prediction fell
 * short, and the solver is asked again, at most --verify times. Every layout uniform across the
 * layers is measured too, so a uniform layout that is smaller at the limit than any mixed one is
 * what gets written. The recipe written is the smallest whose own run measured within the limit;
 * every layer fp8, the base, always is (the tool refuses a limit under it).
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
/* --max-kl's default number of solved recipes measured. */
constexpr int kVerify = 6;

void usage() {
    std::printf(
        "rad-autoquant -- each attention layer's KV layout from measured KL under a bit budget\n"
        "\n"
        "usage: rad-autoquant --kld-ref DIR --layers N,N,... (--bits B | --max-kl K) -o FILE\n"
        "                     [options] -- <radiance flags: --model and the serve flags>\n"
        "\n"
        "  --kld-ref DIR       a KL reference recorded by this model with a bf16 cache\n"
        "  --layers N,N,...    the model's attention layers, as the checkpoint numbers them\n"
        "  --bits B            the average bits a cached K or V element may cost, scales\n"
        "                      included (fp8 8, fp8_v4 6.25, k6_v4 5.5, k4h_v4 4.5,\n"
        "                      k4h_v2h 3.5): the least KL within them\n"
        "  --max-kl K          the mean KL against the reference a recipe may measure: the\n"
        "                      fewest bits within it, each solved recipe measured\n"
        "  -o FILE             the recipe to write (--kv-recipe FILE)\n"
        "  --verify N          with --max-kl, the solved recipes to measure at most (default\n"
        "                      %d)\n"
        "  --reuse             take a run's KL from the work dir where its recipe and flags\n"
        "                      are the same as before, instead of running it again\n"
        "  --layouts L,L,...   the layouts to choose from (default every one); fp8, the base,\n"
        "                      is always among them\n"
        "  --work DIR          each run's recipe, report and log (default FILE.work)\n"
        "  --min-context N     refuse a reference with a document under N tokens (default\n"
        "                      %lld); 0 measures any\n"
        "  --radiance PATH     the engine (default: radiance beside this tool)\n"
        "  -h, --help\n",
        kVerify, (long long)kMinContext);
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

/* The mean KL a KL-mode report states, or a negative number where it states none. */
double report_mean(const std::string& path) {
    std::ifstream f(path);
    if (!f) return -1;
    const json rep = json::parse(f, nullptr, false);
    if (!rep.is_object() || !rep.contains("all") || !rep["all"].is_object() ||
        !rep["all"].contains("kld") || !rep["all"]["kld"].is_object() ||
        !rep["all"]["kld"].contains("mean") || !rep["all"]["kld"]["mean"].is_number())
        return -1;
    const double m = rep["all"]["kld"]["mean"].get<double>();
    return std::isfinite(m) && m >= 0 ? m : -1;
}

std::string read_file(const std::string& path, bool* ok) {
    std::ifstream f(path, std::ios::binary);
    *ok = (bool)f;
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct Run {
    std::string radiance, ref, work;
    std::vector<std::string> engine;
    std::vector<int> layers;
    bool reuse = false;

    /* The mean KL of the engine with layer layers[i] in layout[i], or a negative number, the
     * refusal printed, when the run did not report one. With `reuse`, a run whose recipe and
     * command line in the work dir are this one's, and whose report states a mean, is not run
     * again: its command line is written beside it (NAME.args) only once it has succeeded. */
    double mean_kl(const std::string& name, const std::vector<int>& layout) const {
        const std::string base = work + "/" + name;
        const std::string recipe = base + ".kvrecipe", out = base + ".json", log = base + ".log",
                          args = base + ".args";
        std::string text;
        for (size_t i = 0; i < layers.size(); ++i)
            text += "layer " + std::to_string(layers[i]) + " " + kv_layout_name(layout[i]) + "\n";
        std::vector<std::string> a = { radiance };
        a.insert(a.end(), engine.begin(), engine.end());
        for (const char* s : { "--kv-cache-dtype", "fp8", "--kv-recipe" }) a.push_back(s);
        a.push_back(recipe);
        a.push_back("--kld-ref");
        a.push_back(ref);
        a.push_back("--kld-out");
        a.push_back(out);
        std::string cmd;
        for (const std::string& s : a) cmd += s + "\n";
        if (reuse) {
            bool ok_r = false, ok_a = false;
            const double m = report_mean(out);
            if (m >= 0 && read_file(recipe, &ok_r) == text && ok_r &&
                read_file(args, &ok_a) == cmd && ok_a) {
                std::printf("rad-autoquant: %s: reused from %s\n", name.c_str(), out.c_str());
                return m;
            }
        }
        std::remove(args.c_str());
        {
            std::ofstream f(recipe, std::ios::binary);
            f << text;
            if (!f) { fail("cannot write %s", recipe.c_str()); return -1; }
        }
        std::remove(out.c_str());
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

        const double m = ok ? report_mean(out) : -1;
        if (m < 0) {
            fail("run %s %s; its log, %s, ends:\n%s", name.c_str(),
                 ok ? "reported no mean KL" : "failed", log.c_str(), tail(log, 15).c_str());
            return -1;
        }
        std::ofstream f(args, std::ios::binary);
        f << cmd;
        return m;
    }
};

}  /* namespace */

int main(int argc, char** argv) {
    std::string ref, out, work, radiance, layers_arg, layouts_arg, bits_arg, max_kl_arg;
    int64_t min_context = kMinContext, verify = kVerify;
    bool reuse = false;
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
        if (a == "--max-kl" && val(&max_kl_arg)) continue;
        if (a == "--reuse") { reuse = true; continue; }
        if (a == "--verify" && val(&v)) {
            if (!parse_int(v, 1, 1000, &verify))
                return fail("--verify '%s' is not a number of runs from 1 to 1000", v.c_str());
            continue;
        }
        if (a == "--min-context" && val(&v)) {
            if (!parse_int(v, 0, INT64_MAX, &min_context))
                return fail("--min-context '%s' is not a token count", v.c_str());
            continue;
        }
        usage();
        return fail("'%s' is not an option here, or lacks its value; the engine's flags go "
                    "after --", a.c_str());
    }
    if (ref.empty() || out.empty() || layers_arg.empty() || (bits_arg.empty() && max_kl_arg.empty())) {
        usage();
        return fail("--kld-ref, --layers, -o and one of --bits and --max-kl are required");
    }
    if (!bits_arg.empty() && !max_kl_arg.empty())
        return fail("--bits and --max-kl ask opposite questions; give one");
    const bool by_kl = !max_kl_arg.empty();
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

    int64_t budget = 0;
    double max_kl = 0;
    if (!by_kl) {
        char* end = nullptr;
        const double bits = std::strtod(bits_arg.c_str(), &end);
        if (!end || *end || !(bits > 0) || !std::isfinite(bits))
            return fail("--bits '%s' is not a number of bits", bits_arg.c_str());
        /* The budget in bits a layer's group, summed; the epsilon keeps 6.25 * 64 at 400. */
        budget = (int64_t)std::floor(bits * kGroupElems * (double)layers.size() + 1e-6);
        int narrowest = INT_MAX;
        for (int q : layouts) narrowest = std::min(narrowest, kv_layout_group_bits(q));
        if (budget < (int64_t)narrowest * (int64_t)layers.size())
            return fail("--bits %s is under the narrowest layout's %.2f", bits_arg.c_str(),
                        (double)narrowest / kGroupElems);
    } else {
        char* end = nullptr;
        max_kl = std::strtod(max_kl_arg.c_str(), &end);
        if (!end || *end || !(max_kl > 0) || !std::isfinite(max_kl))
            return fail("--max-kl '%s' is not a positive mean KL", max_kl_arg.c_str());
    }

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
    const Run run{ radiance, ref, work, engine, layers, reuse };
    const size_t n_runs = 1 + layers.size() * (layouts.size() - 1);
    if (by_kl)
        std::printf("rad-autoquant: %zu layers, %zu layouts, %zu runs, %zu uniform and up to %lld "
                    "to verify, over %zu documents (shortest %lld tokens); work in %s\n",
                    layers.size(), layouts.size(), n_runs, layouts.size() - 1, (long long)verify,
                    manifest["docs"].size(), (long long)shortest, work.c_str());
    else
        std::printf("rad-autoquant: %zu layers, %zu layouts, %zu runs over %zu documents (shortest "
                    "%lld tokens); work in %s\n", layers.size(), layouts.size(), n_runs,
                    manifest["docs"].size(), (long long)shortest, work.c_str());
    std::fflush(stdout);

    const double base = run.mean_kl("base", std::vector<int>(layers.size(), RAD_KVQ_NONE));
    if (base < 0) return 1;
    std::printf("rad-autoquant: every layer fp8: mean KL %.6f\n", base);
    std::fflush(stdout);
    if (by_kl && base > max_kl)
        return fail("every layer fp8 already measures a mean KL of %.6f, above --max-kl %s, and "
                    "no recipe layout is wider than fp8: raise the limit, or serve a bf16 cache",
                    base, max_kl_arg.c_str());
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

    /* The layouts, bits and predicted mean KL of an assignment, from the per-layer costs. */
    auto layouts_of = [&](const KvAssignment& a) {
        std::vector<int> lay;
        for (size_t i = 0; i < layers.size(); ++i) lay.push_back(cand[i][(size_t)a.pick[i]].kvq);
        return lay;
    };
    auto bits_of = [&](const std::vector<int>& lay) {
        int64_t b = 0;
        for (int q : lay) b += kv_layout_group_bits(q);
        return b;
    };
    auto predicted_of = [&](const std::vector<int>& lay) {
        double p = base;
        for (size_t i = 0; i < layers.size(); ++i)
            for (const KvCandidate& c : cand[i])
                if (c.kvq == lay[i]) p += c.kl;
        return p;
    };
    auto per_element = [&](int64_t b) { return (double)b / kGroupElems / (double)layers.size(); };

    /* What is written: a layout per layer, and the comment lines that say why. */
    std::vector<int> chosen;
    std::ostringstream why;
    char line[256];
    double avg = 0, reported = 0;
    if (!by_kl) {
        KvAssignment pick;
        if (kv_autoquant_solve(cand, budget, &pick) != RAD_OK)
            return fail("no assignment fits %s bits", bits_arg.c_str());
        chosen = layouts_of(pick);
        avg = per_element(pick.bits);
        reported = base + pick.kl;
        std::snprintf(line, sizeof line,
                      "# chosen: %.4f bits an element, predicted mean KL %.6f (the layers' costs "
                      "summed onto the base)\n", avg, reported);
        why << line;
    } else {
        /* Every recipe measured: the base, each layout uniform, and the solver's answers. */
        struct Measured {
            std::string name;
            std::vector<int> lay;
            double kl;
        };
        std::vector<Measured> runs = { { "base", std::vector<int>(layers.size(), RAD_KVQ_NONE),
                                         base } };
        auto measured = [&](const std::vector<int>& lay) -> const Measured* {
            for (const Measured& m : runs)
                if (m.lay == lay) return &m;
            return nullptr;
        };
        for (int q : layouts) {
            if (q == RAD_KVQ_NONE) continue;
            const std::vector<int> lay(layers.size(), q);
            const double kl = run.mean_kl(std::string("uniform_") + kv_layout_name(q), lay);
            if (kl < 0) return 1;
            runs.push_back({ std::string("uniform ") + kv_layout_name(q), lay, kl });
            std::printf("rad-autoquant: every layer %s: mean KL %.6f\n", kv_layout_name(q), kl);
            std::fflush(stdout);
        }
        /* The cost above the base the solver may predict. A solved recipe that measures above
         * the limit scales it by how far that prediction fell short of its measurement, and
         * keeps it under that recipe's prediction, so the next answer is a different one. */
        const double room = max_kl - base;
        double allowed = room;
        for (int64_t v = 1; v <= verify; ++v) {
            KvAssignment a;
            if (kv_autoquant_solve_kl(cand, allowed, &a) != RAD_OK) break;
            const std::vector<int> lay = layouts_of(a);
            const Measured* m = measured(lay);
            double kl = m ? m->kl : -1;
            if (!m) {
                kl = run.mean_kl("verify" + std::to_string(v), lay);
                if (kl < 0) return 1;
                runs.push_back({ "verify " + std::to_string(v), lay, kl });
            }
            std::printf("rad-autoquant: solved %lld: %.4f bits an element, predicted mean KL %.6f, "
                        "measured %.6f%s, %s\n", (long long)v, per_element(a.bits), base + a.kl,
                        kl, m ? (" (as " + m->name + ")").c_str() : "",
                        kl <= max_kl ? "within the limit" : "above the limit");
            std::fflush(stdout);
            if (kl <= max_kl) break;
            const double actual = kl - base;
            double next = a.kl > 0 && actual > 0 ? room * (a.kl / actual) : 0;
            next = std::min(next, a.kl - std::max(1e-12, 1e-9 * a.kl));
            if (next < 0) break;
            allowed = next;
        }
        /* The fewest bits among the recipes measured within the limit, the least KL of those. */
        const Measured* best = nullptr;
        for (const Measured& m : runs) {
            if (m.kl > max_kl) continue;
            if (!best || bits_of(m.lay) < bits_of(best->lay) ||
                (bits_of(m.lay) == bits_of(best->lay) && m.kl < best->kl))
                best = &m;
        }
        /* The base is among them: a limit under it was refused before anything else ran. */
        chosen = best->lay;
        avg = per_element(bits_of(chosen));
        reported = best->kl;
        why << "# recipes measured, mean KL predicted (the layers' costs summed onto the base) "
               "and measured\n";
        for (const Measured& m : runs) {
            std::snprintf(line, sizeof line, "#   %-17s %7.4f bits  predicted %.6f  measured %.6f%s\n",
                          m.name.c_str(), per_element(bits_of(m.lay)), predicted_of(m.lay), m.kl,
                          m.kl <= max_kl ? "" : "  above the limit");
            why << line;
        }
        std::snprintf(line, sizeof line,
                      "# chosen: %.4f bits an element, measured mean KL %.6f (%s), the fewest bits "
                      "measured within %s\n", avg, reported, best->name.c_str(), max_kl_arg.c_str());
        why << line;
    }

    std::ostringstream r;
    if (by_kl)
        r << "# rad-autoquant: a KV layout per attention layer, the fewest bits measured within a "
             "mean KL of " << max_kl_arg << "\n";
    else
        r << "# rad-autoquant: a KV layout per attention layer within " << bits_arg
          << " bits an element\n";
    r << "# reference " << ref << ", " << manifest["docs"].size()
      << " documents, the shortest " << shortest << " tokens\n";
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
    r << why.str();
    for (size_t i = 0; i < layers.size(); ++i)
        r << "layer " << layers[i] << " " << kv_layout_name(chosen[i]) << "\n";

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
    std::printf("rad-autoquant: %s: %.4f bits an element, %s mean KL %.6f\n", out.c_str(), avg,
                by_kl ? "measured" : "predicted", reported);
    return 0;
}
