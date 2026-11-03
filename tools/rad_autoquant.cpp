/* rad-autoquant -- a --kv-recipe from measurement: each attention layer's KV layout chosen so the
 * model's output moves least for an average bit budget, or so the cache is smallest for a KL limit
 * (docs/TOOLS.md, "rad-autoquant").
 *
 *   rad-autoquant --kld-ref REF --layers 3,7,... --bits 6 -o model.kvrecipe -- <radiance flags>
 *   rad-autoquant --kld-ref REF --layers 3,7,... --max-kl 0.035 -o model.kvrecipe -- <flags>
 *
 * REF is a KL-mode reference (`radiance --kld-record`) recorded by the same model with an
 * unquantised (bf16) cache, so a run's KL is what its cache costs and nothing else.
 *
 * The candidates are the variants (kv_autoquant_variants): every layout of the recipe table
 * (core/config.cpp, kKvLayouts), each plain, `rotated` and, with --vmean FILE, `vmean`, as far as
 * the recipe parser takes the words. So a layout or recipe word added there is calibrated with no
 * change here. --layouts and --variants restrict the set.
 *
 *   1. Every variant uniform, one run each. The base is the one of least mean KL (bf16, the
 *      unquantised cache, whose KL against the bf16 reference is about zero; i8h with --layouts
 *      leaving bf16 out): a layer's cost in another variant is measured above it, so nothing
 *      reads as better than the base and the clamp at zero below only absorbs noise.
 *   2. Screening: a variant another of no more bits beats uniformly gets no per-layer runs
 *      (--keep-dominated measures it anyway). Its uniform recipe still competes in --max-kl.
 *   3. Each layer alone in each kept variant, the rest at the base. A layer's cost is the run's
 *      mean KL above the base's; a run under it is noise and costs nothing.
 *   4. --bits: with the costs taken as adding up, kv_autoquant_solve picks the assignment of least
 *      summed cost within the budget, written with its prediction, unmeasured.
 *      --max-kl: they do not add up exactly when layouts are mixed, so kv_autoquant_fit measures
 *      each solved assignment and tightens the predicted room by how far the prediction fell
 *      short, at most --verify times. The recipe written is the fewest bits of every recipe
 *      measured within the limits (uniform, single-layer or solved), so it is never worse than the
 *      best uniform variant. --min-top1 adds a floor on top-1 agreement; --strict checks both at
 *      the 95 % bound from each run's .rows, the documents as strata (kv_autoquant_interval).
 *
 * Every run is a child `radiance` with the flags after `--`, plus the cache and KL flags this tool
 * sets; its recipe, report, .rows and log stay in the work directory, so a surprising number can
 * be looked at afterwards, and --reuse reads a run back where its recipe and command line are
 * unchanged.
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

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <deque>
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
        "rad-autoquant -- each attention layer's KV layout from measured KL\n"
        "\n"
        "usage: rad-autoquant --kld-ref DIR --layers N,N,... (--bits B | --max-kl K) -o FILE\n"
        "                     [options] -- <radiance flags: --model and the serve flags>\n"
        "\n"
        "  --kld-ref DIR       a KL reference recorded by this model with a bf16 cache\n"
        "  --layers N,N,...    the model's attention layers, as the checkpoint numbers them\n"
        "  --bits B            the average bits a cached K or V element may cost, scales\n"
        "                      included (bf16 16, i8h 8.06, fp8 8, fp8_v4 6.25, k6_v4 5.5, k4h_v4 4.5,\n"
        "                      k4h_v2h 3.5): the least KL within them, predicted\n"
        "  --max-kl K          the mean KL against the reference a recipe may measure: the\n"
        "                      fewest bits measured within it\n"
        "  -o FILE             the recipe to write (--kv-recipe FILE)\n"
        "  --min-top1 P        with --max-kl, the top-1 agreement (0..1) it must keep too\n"
        "  --strict            with --max-kl, check the 95%% bounds from the runs' .rows (KL's\n"
        "                      upper, top-1's lower), not the means\n"
        "  --verify N          with --max-kl, the solved recipes to measure at most (default\n"
        "                      %d)\n"
        "  --vmean FILE        a V mean file (radiance --calib-dir): the variants that\n"
        "                      subtract it join the candidates\n"
        "  --layouts L,L,...   only these layouts' variants (default every recipe layout)\n"
        "  --variants V,V,...  only these variants, a variant's words joined by '+'\n"
        "                      (k6_v4+rotated+vmean)\n"
        "  --keep-dominated    per-layer runs for the variants screening would drop too\n"
        "  --reuse             take a run's KL from the work dir where its recipe and flags\n"
        "                      are the same as before, instead of running it again\n"
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

std::string read_file(const std::string& path, bool* ok) {
    std::ifstream f(path, std::ios::binary);
    *ok = (bool)f;
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

/* FNV-1a, 64 bits, in hex: a solved recipe's run name and the V mean file's identity in --reuse. */
std::string fnv_hex(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)h);
    return b;
}

/* A variant's words as one token, for run names and --variants: "k6_v4+rotated+vmean". */
std::string tag(const KvVariant& v) {
    std::string t = v.words;
    std::replace(t.begin(), t.end(), ' ', '+');
    return t;
}

/* One measured recipe: what the KL mode reported, and the 95 % intervals from its .rows (half
 * NaN without usable rows; top1 NaN where the report has none). */
struct Meas {
    std::string label;          /* "every layer fp8", "layer 3 k6_v4", "verify 1" */
    std::vector<int32_t> kvq;   /* per --layers entry */
    int64_t bits = 0;           /* summed over the layers, a group of 32 K and 32 V each */
    double kl = 0, top1 = NAN;
    KvInterval kl_ci, top1_ci;
};

/* The limits a --max-kl recipe is checked against: the means, or under --strict the 95 % bound
 * on the bad side. A missing number fails the check it takes part in. */
struct Limits {
    double max_kl = 0;
    double min_top1 = NAN;   /* none */
    bool strict = false;
    double kl(const Meas& m) const { return strict ? m.kl + m.kl_ci.half : m.kl; }
    double top1(const Meas& m) const { return strict ? m.top1 - m.top1_ci.half : m.top1; }
    bool within(const Meas& m) const {
        return kl(m) <= max_kl && (std::isnan(min_top1) || top1(m) >= min_top1);
    }
};

struct Run {
    std::string radiance, ref, work, vmean, vmean_id;   /* vmean absolute, or empty */
    std::vector<std::string> engine;
    std::vector<int> layers;
    std::vector<KvVariant> all;        /* every variant, for the words of a RAD_KVQ_* */
    std::vector<int64_t> doc_rows;     /* the reference's scored positions a document */
    bool reuse = false;

    const KvVariant* variant(int32_t kvq) const {
        for (const KvVariant& v : all)
            if (v.kvq == kvq) return &v;
        return nullptr;
    }
    /* The recipe a run is given: the mean file first where a layer subtracts it. */
    std::string text(const std::vector<int32_t>& kvq) const {
        std::string s;
        if (std::any_of(kvq.begin(), kvq.end(), [](int32_t q) { return q & RAD_KVQ_VMEAN; }))
            s += "vmean " + vmean + "\n";
        for (size_t i = 0; i < kvq.size(); ++i)
            s += "layer " + std::to_string(layers[i]) + " " + variant(kvq[i])->words + "\n";
        return s;
    }

    /* The engine with layer layers[i] in variant kvq[i], into `m`; false, the refusal printed,
     * when the run did not report a mean KL. With `reuse`, a run whose recipe and command line in
     * the work dir are this one's, and whose report states a mean, is not run again: its command
     * line (and the V mean file's hash where the recipe reads it) is written beside it
     * (NAME.args) only once it has succeeded. */
    bool measure(const std::string& name, const std::string& label,
                 const std::vector<int32_t>& kvq, Meas* m) const {
        const std::string base = work + "/" + name;
        const std::string recipe = base + ".kvrecipe", out = base + ".json", log = base + ".log",
                          args = base + ".args", rows = out + ".rows";
        const std::string text = this->text(kvq);
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
        if (text.rfind("vmean ", 0) == 0) cmd += "# vmean " + vmean_id + "\n";
        m->label = label;
        m->kvq = kvq;
        m->bits = 0;
        for (int32_t q : kvq) m->bits += variant(q)->bits;
        bool reused = false;
        if (reuse) {
            bool ok_r = false, ok_a = false;
            reused = read_file(recipe, &ok_r) == text && ok_r && read_file(args, &ok_a) == cmd &&
                     ok_a && report(out, rows, m);
            if (reused) std::printf("rad-autoquant: %s: reused from %s\n", name.c_str(), out.c_str());
        }
        if (!reused) {
            std::remove(args.c_str());
            {
                std::ofstream f(recipe, std::ios::binary);
                f << text;
                if (!f) { fail("cannot write %s", recipe.c_str()); return false; }
            }
            std::remove(out.c_str());
            std::remove(rows.c_str());
            std::vector<char*> av;
            for (std::string& s : a) av.push_back(s.data());
            av.push_back(nullptr);

            const int fd = ::open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
            if (fd < 0) { fail("cannot write %s: %s", log.c_str(), std::strerror(errno)); return false; }
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
            if (pid < 0) { fail("fork: %s", std::strerror(errno)); return false; }
            int st = 0;
            while (::waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
            const bool ok = WIFEXITED(st) && WEXITSTATUS(st) == 0;
            if (!ok || !report(out, rows, m)) {
                fail("run %s %s; its log, %s, ends:\n%s", name.c_str(),
                     ok ? "reported no mean KL" : "failed", log.c_str(), tail(log, 15).c_str());
                return false;
            }
            std::ofstream f(args, std::ios::binary);
            f << cmd;
        }
        std::printf("rad-autoquant: %s: %s\n", label.c_str(), describe(*m).c_str());
        std::fflush(stdout);
        return true;
    }

    /* A KL-mode report's mean KL and top-1 agreement into `m`, and the intervals from its .rows
     * ([positions, 4] f32: KL, the two NLLs, top-1 agreement) where they are the reference's
     * positions; false where it states no mean. */
    bool report(const std::string& out, const std::string& rows, Meas* m) const {
        std::ifstream f(out);
        if (!f) return false;
        const json rep = json::parse(f, nullptr, false);
        if (!rep.is_object() || !rep.contains("all") || !rep["all"].is_object()) return false;
        const json& a = rep["all"];
        if (!a.contains("kld") || !a["kld"].is_object() || !a["kld"].contains("mean") ||
            !a["kld"]["mean"].is_number())
            return false;
        m->kl = a["kld"]["mean"].get<double>();
        if (!std::isfinite(m->kl) || m->kl < 0) return false;
        m->top1 = a.contains("top1_agreement") && a["top1_agreement"].is_number()
                      ? a["top1_agreement"].get<double>() : NAN;
        m->kl_ci = m->top1_ci = KvInterval{};
        m->kl_ci.half = m->top1_ci.half = NAN;
        bool ok = false;
        const std::string raw = read_file(rows, &ok);
        int64_t n = 0;
        for (int64_t d : doc_rows) n += d;
        if (!ok || raw.empty() || raw.size() % 16 || (n && (int64_t)raw.size() != n * 16)) return true;
        std::vector<float> v(raw.size() / 4);
        std::memcpy(v.data(), raw.data(), raw.size());
        std::vector<double> kl, t1;
        for (size_t i = 0; i < v.size(); i += 4) { kl.push_back(v[i]); t1.push_back(v[i + 3]); }
        KvInterval a1, a2;
        if (kv_autoquant_interval(kl, doc_rows, &a1) == RAD_OK &&
            kv_autoquant_interval(t1, doc_rows, &a2) == RAD_OK) {
            m->kl_ci = a1;
            m->top1_ci = a2;
        }
        return true;
    }

    /* "5.5000 bits, mean KL 0.034429 +- 0.003400, top-1 0.9412 +- 0.0021"; without the bits,
     * from "mean KL". */
    std::string describe(const Meas& m, bool bits = true) const {
        char b[200];
        int at = bits ? std::snprintf(b, sizeof b, "%.4f bits, ", per_element(m.bits)) : 0;
        at += std::snprintf(b + at, sizeof b - (size_t)at, "mean KL %.6f", m.kl);
        if (!std::isnan(m.kl_ci.half))
            at += std::snprintf(b + at, sizeof b - (size_t)at, " +- %.6f", m.kl_ci.half);
        if (!std::isnan(m.top1)) {
            at += std::snprintf(b + at, sizeof b - (size_t)at, ", top-1 %.4f", m.top1);
            if (!std::isnan(m.top1_ci.half))
                std::snprintf(b + at, sizeof b - (size_t)at, " +- %.4f", m.top1_ci.half);
        }
        return b;
    }
    double per_element(int64_t bits) const {
        return (double)bits / kGroupElems / (double)layers.size();
    }
};

}  /* namespace */

int main(int argc, char** argv) {
    std::string ref, out, work, radiance, layers_arg, layouts_arg, variants_arg, bits_arg,
        max_kl_arg, min_top1_arg, vmean_arg;
    int64_t min_context = kMinContext, verify = kVerify;
    bool reuse = false, strict = false, keep_dominated = false, have_verify = false;
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
        if (a == "--variants" && val(&variants_arg)) continue;
        if (a == "--vmean" && val(&vmean_arg)) continue;
        if (a == "--bits" && val(&bits_arg)) continue;
        if (a == "--max-kl" && val(&max_kl_arg)) continue;
        if (a == "--min-top1" && val(&min_top1_arg)) continue;
        if (a == "--reuse") { reuse = true; continue; }
        if (a == "--strict") { strict = true; continue; }
        if (a == "--keep-dominated") { keep_dominated = true; continue; }
        if (a == "--verify" && val(&v)) {
            if (!parse_int(v, 1, 1000, &verify))
                return fail("--verify '%s' is not a number of runs from 1 to 1000", v.c_str());
            have_verify = true;
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
    if (!by_kl && (!min_top1_arg.empty() || strict || have_verify))
        return fail("--min-top1, --strict and --verify go with --max-kl; --bits measures nothing "
                    "it writes");
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
    const size_t L = layers.size();

    /* The V mean file: absolute, as the recipes in the work dir and the one written name it, and
     * its content's hash, so --reuse runs again after it is recalibrated. */
    std::string vmean, vmean_id;
    if (!vmean_arg.empty()) {
        char buf[PATH_MAX];
        if (!::realpath(vmean_arg.c_str(), buf))
            return fail("--vmean %s: %s", vmean_arg.c_str(), std::strerror(errno));
        vmean = buf;
        bool ok = false;
        const std::string vt = read_file(vmean, &ok);
        KvVMean vm;
        if (!ok || kv_vmean_parse(vt, vmean.c_str(), &vm) != RAD_OK)
            return fail("--vmean %s is not a V mean file (radiance --calib-dir writes one)",
                        vmean.c_str());
        vmean_id = fnv_hex(vt);
    }

    /* The candidates: the variants, widest first so that a tie keeps the wider one, restricted by
     * --layouts and --variants. */
    const std::vector<KvVariant> every = kv_autoquant_variants(true);
    std::vector<KvVariant> variants = kv_autoquant_variants(!vmean.empty());
    if (!layouts_arg.empty()) {
        std::set<int> want;
        for (const std::string& s : split(layouts_arg, ',')) {
            const int q = kv_layout_of(s.c_str());
            if (q < 0) return fail("--layouts: '%s' is not a KV layout", s.c_str());
            want.insert(q);
        }
        std::erase_if(variants, [&](const KvVariant& v) { return !want.count(v.kvq & RAD_KVQ_LAYOUT); });
    }
    if (!variants_arg.empty()) {
        std::set<int32_t> want;
        for (const std::string& s : split(variants_arg, ',')) {
            const KvVariant* hit = nullptr;
            for (const KvVariant& v : every)
                if (tag(v) == s) hit = &v;
            if (!hit) {
                std::string names;
                for (const KvVariant& v : every) names += (names.empty() ? "" : ", ") + tag(v);
                return fail("--variants: '%s' is not a variant; they are %s", s.c_str(), names.c_str());
            }
            if ((hit->kvq & RAD_KVQ_VMEAN) && vmean.empty())
                return fail("--variants: %s subtracts a V mean, and no --vmean FILE names one",
                            s.c_str());
            want.insert(hit->kvq);
        }
        std::erase_if(variants, [&](const KvVariant& v) { return !want.count(v.kvq); });
    }
    if (variants.empty()) return fail("--layouts and --variants leave no variant to choose from");
    const size_t C = variants.size();

    int64_t budget = 0;
    Limits lim;
    lim.strict = strict;
    if (!by_kl) {
        char* end = nullptr;
        const double bits = std::strtod(bits_arg.c_str(), &end);
        if (!end || *end || !(bits > 0) || !std::isfinite(bits))
            return fail("--bits '%s' is not a number of bits", bits_arg.c_str());
        /* The budget in bits a layer's group, summed; the epsilon keeps 6.25 * 64 at 400. */
        budget = (int64_t)std::floor(bits * kGroupElems * (double)L + 1e-6);
        int narrowest = INT_MAX;
        for (const KvVariant& v : variants) narrowest = std::min(narrowest, v.bits);
        if (budget < (int64_t)narrowest * (int64_t)L)
            return fail("--bits %s is under the narrowest layout's %.2f", bits_arg.c_str(),
                        (double)narrowest / kGroupElems);
    } else {
        char* end = nullptr;
        lim.max_kl = std::strtod(max_kl_arg.c_str(), &end);
        if (!end || *end || !(lim.max_kl > 0) || !std::isfinite(lim.max_kl))
            return fail("--max-kl '%s' is not a positive mean KL", max_kl_arg.c_str());
        if (!min_top1_arg.empty()) {
            lim.min_top1 = std::strtod(min_top1_arg.c_str(), &end);
            if (!end || *end || !(lim.min_top1 >= 0 && lim.min_top1 <= 1))
                return fail("--min-top1 '%s' is not an agreement from 0 to 1", min_top1_arg.c_str());
        }
    }

    /* The reference's documents, against the long-context bound, and each one's scored
     * positions (`[score_from, n - 1)`, core/kld.h): the strata of a run's interval. */
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
    std::vector<int64_t> doc_rows;
    for (size_t d = 0; d < manifest["docs"].size(); ++d) {
        const json& j = manifest["docs"][d];
        if (!j.is_object() || !j.contains("tokens") || !j["tokens"].is_number_integer())
            return fail("--kld-ref %s/kld.json: document %zu has no token count", ref.c_str(), d);
        const int64_t n = j["tokens"].get<int64_t>();
        if (n < shortest) { shortest = n; shortest_at = d; }
        const int64_t from = j.contains("score_from") && j["score_from"].is_number_integer()
                                 ? j["score_from"].get<int64_t>() : 0;
        doc_rows.push_back(std::max<int64_t>(0, n - 1 - from));
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
    const Run run{ radiance, ref, work, vmean, vmean_id, engine, layers, every, doc_rows, reuse };
    std::printf("rad-autoquant: %zu layers, %zu variants, over %zu documents (shortest %lld "
                "tokens); work in %s\n", L, C, manifest["docs"].size(), (long long)shortest,
                work.c_str());
    std::printf("rad-autoquant: runs: %zu uniform, then at most %zu per-layer, L (C - 1), fewer "
                "for the variants screening drops", C, L * (C - 1));
    if (by_kl) std::printf(", and up to %lld to verify", (long long)verify);
    std::printf("\n");
    std::fflush(stdout);

    /* Every measured recipe; a deque, so what points into it stays put. */
    std::deque<Meas> runs;
    auto find = [&](const std::vector<int32_t>& kvq) -> const Meas* {
        for (const Meas& m : runs)
            if (m.kvq == kvq) return &m;
        return nullptr;
    };
    /* What a limit needs of a run, where the run lacks it. */
    auto unusable = [&](const Meas& m) -> const char* {
        if (!std::isnan(lim.min_top1) && std::isnan(m.top1))
            return "reported no top-1 agreement, which --min-top1 needs";
        if (lim.strict && std::isnan(m.kl_ci.half))
            return "left no .rows of the reference's positions, which --strict needs";
        return nullptr;
    };

    /* 1. Every variant uniform; the base is the least mean KL, of equal ones the fewer bits. */
    std::vector<const Meas*> uni;
    size_t b0 = 0;
    for (size_t c = 0; c < C; ++c) {
        Meas m;
        if (!run.measure("uniform_" + tag(variants[c]), "every layer " + variants[c].words,
                         std::vector<int32_t>(L, variants[c].kvq), &m))
            return 1;
        if (by_kl)
            if (const char* why = unusable(m))
                return fail("run uniform_%s %s", tag(variants[c]).c_str(), why);
        runs.push_back(m);
        uni.push_back(&runs.back());
        if (m.kl < uni[b0]->kl || (m.kl == uni[b0]->kl && variants[c].bits < variants[b0].bits))
            b0 = c;
    }
    const Meas& base = *uni[b0];
    std::printf("rad-autoquant: the base, the most accurate: every layer %s\n",
                variants[b0].words.c_str());

    /* 2. Screening: a variant that one of no more bits beats on the uniform recipe gets no
     * per-layer runs; at equal bits the lower KL stays, an exact tie keeps the earlier. A
     * heuristic, as a layer can differ from the model's average; the base is never dropped. */
    std::vector<size_t> kept;
    std::vector<std::string> dropped;
    for (size_t i = 0; i < C; ++i) {
        size_t by = SIZE_MAX;
        for (size_t j = 0; j < C && by == SIZE_MAX && !keep_dominated && i != b0; ++j) {
            if (j == i) continue;
            const int bi = variants[i].bits, bj = variants[j].bits;
            const double ki = uni[i]->kl, kj = uni[j]->kl;
            if ((bj < bi && kj <= ki) || (bj == bi && (kj < ki || (kj == ki && j < i)))) by = j;
        }
        if (by == SIZE_MAX) { kept.push_back(i); continue; }
        char b[256];
        std::snprintf(b, sizeof b, "%s (%.6f at %.2f bits; %s: %.6f at %.2f)",
                      variants[i].words.c_str(), uni[i]->kl, (double)variants[i].bits / kGroupElems,
                      variants[by].words.c_str(), uni[by]->kl, (double)variants[by].bits / kGroupElems);
        dropped.push_back(b);
        std::printf("rad-autoquant: screening drops %s\n", b);
    }
    /* Under --max-kl a base over the limits leaves the costs nothing to give: only the uniform
     * recipes compete. */
    const bool search = !by_kl || lim.within(base);
    std::printf("rad-autoquant: %zu per-layer runs, %zu layers x %zu variants\n",
                search ? L * (kept.size() - 1) : 0, L, kept.size() - 1);
    std::fflush(stdout);

    /* 3. cand[i][k]: layer layers[i] alone in variants[kept[k]], the rest at the base. */
    std::vector<std::vector<KvCandidate>> cand(L);
    for (size_t i = 0; i < L && search; ++i)
        for (size_t k : kept) {
            double d = 0;
            if (k != b0) {
                std::vector<int32_t> kvq(L, variants[b0].kvq);
                kvq[i] = variants[k].kvq;
                Meas m;
                if (!run.measure("layer" + std::to_string(layers[i]) + "_" + tag(variants[k]),
                                 "layer " + std::to_string(layers[i]) + " " + variants[k].words,
                                 kvq, &m))
                    return 1;
                d = std::max(0.0, m.kl - base.kl);
                if (by_kl && !unusable(m) && !find(kvq)) runs.push_back(m);
            }
            cand[i].push_back({ variants[k].kvq, variants[k].bits, d });
        }

    /* The variants of an assignment. */
    auto kvq_of = [&](const KvAssignment& a) {
        std::vector<int32_t> q;
        for (size_t i = 0; i < L; ++i) q.push_back(cand[i][(size_t)a.pick[i]].kvq);
        return q;
    };

    /* What is written: a variant per layer, and the comment lines that say why. */
    std::vector<int32_t> chosen;
    std::ostringstream why;
    char line[512];
    double avg = 0, reported = 0;
    std::string chosen_desc;
    if (!by_kl) {
        KvAssignment pick;
        if (kv_autoquant_solve(cand, budget, &pick) != RAD_OK)
            return fail("no assignment fits %s bits", bits_arg.c_str());
        chosen = kvq_of(pick);
        avg = run.per_element(pick.bits);
        reported = base.kl + pick.kl;
        std::snprintf(line, sizeof line,
                      "# chosen: %.4f bits an element, predicted mean KL %.6f (the layers' costs "
                      "summed onto the base)\n", avg, reported);
        why << line;
    } else {
        /* The fewest bits measured within the limits, the least KL of those. */
        auto best = [&]() -> const Meas* {
            const Meas* w = nullptr;
            for (const Meas& m : runs)
                if (lim.within(m) && (!w || m.bits < w->bits ||
                                      (m.bits == w->bits && lim.kl(m) < lim.kl(*w))))
                    w = &m;
            return w;
        };
        /* 4. Solve, measure, tighten. */
        std::vector<KvRound> trail;
        std::string capped;
        if (search) {
            int round = 0;
            const KvMeasure measure = [&](const KvAssignment& a, std::vector<KvCheck>* ch) {
                const std::vector<int32_t> kvq = kvq_of(a);
                const Meas* m = find(kvq);
                ++round;
                if (!m) {
                    Meas n;
                    if (!run.measure("verify_" + fnv_hex(run.text(kvq)).substr(0, 12),
                                     "verify " + std::to_string(round), kvq, &n))
                        return (int)RAD_E_IO;
                    if (const char* w = unusable(n)) { fail("verify %d %s", round, w); return (int)RAD_E_IO; }
                    runs.push_back(n);
                    m = &runs.back();
                }
                std::printf("rad-autoquant: solved %d: %.4f bits an element, predicted mean KL "
                            "%.6f, measured %.6f (%s), %s\n", round, run.per_element(a.bits),
                            base.kl + a.kl, m->kl, m->label.c_str(),
                            lim.within(*m) ? "within the limit" : "above the limit");
                std::fflush(stdout);
                ch->push_back({ lim.kl(*m) - lim.kl(base), lim.max_kl - lim.kl(base) });
                if (!std::isnan(lim.min_top1))
                    ch->push_back({ lim.top1(base) - lim.top1(*m), lim.top1(base) - lim.min_top1 });
                return (int)RAD_OK;
            };
            /* A solution of no fewer bits than a recipe already measured within cannot be the
             * answer, so it is not measured. */
            const Meas* had = best();
            const int rc = kv_autoquant_fit(cand, lim.max_kl - lim.kl(base), (int)verify, measure,
                                            &trail, had ? had->bits : INT64_MAX);
            if (rc == RAD_E_IO) return 1;
            if (rc != RAD_OK) return fail("the solver refused the per-layer costs; this is a bug");
            if (had && (trail.empty() || (!trail.back().within && (int64_t)trail.size() < verify &&
                                          trail.back().pick.kl > 0))) {
                capped = "the next solution has no fewer bits than " + had->label +
                         ", measured within the limits, and is not measured";
                std::printf("rad-autoquant: %s\n", capped.c_str());
            }
        }
        const Meas* w = best();
        if (!w) {
            std::string t;
            if (!std::isnan(lim.min_top1)) {
                std::snprintf(line, sizeof line, " and top-1 %.4f", lim.top1(base));
                t = line;
            }
            return fail("no recipe measured is within the limits: the most accurate, every layer "
                        "%s, measures a mean KL of %.6f%s%s, above --max-kl %s%s%s",
                        variants[b0].words.c_str(), base.kl, t.c_str(),
                        lim.strict ? " at the 95% bounds" : "", max_kl_arg.c_str(),
                        std::isnan(lim.min_top1) ? "" : " or under --min-top1 ",
                        min_top1_arg.c_str());
        }
        chosen = w->kvq;
        avg = run.per_element(w->bits);
        reported = w->kl;
        /* Each solved recipe: the room it was solved within, its prediction and its run. */
        for (size_t k = 0; k < trail.size(); ++k) {
            const Meas& m = *find(kvq_of(trail[k].pick));
            std::snprintf(line, sizeof line,
                          "# solved %zu: room %.6f above the base, predicted mean KL %.6f, measured "
                          "%s (%s)%s\n", k + 1, trail[k].allowed, base.kl + trail[k].pick.kl,
                          run.describe(m).c_str(), m.label.c_str(),
                          trail[k].within ? "" : ", above the limit");
            why << line;
        }
        if (!capped.empty()) why << "# " << capped << "\n";
        std::snprintf(line, sizeof line, "# chosen: %s (%s), the fewest bits measured within %s%s",
                      run.describe(*w).c_str(), w->label.c_str(), max_kl_arg.c_str(),
                      std::isnan(lim.min_top1) ? "" : (" and top-1 " + min_top1_arg).c_str());
        why << line << (lim.strict ? " at the 95% bounds\n" : "\n");
        why << "# (+- the 95% half-width from the run's .rows, each document a stratum)\n";
        chosen_desc = run.describe(*w);
    }

    std::ostringstream r;
    if (by_kl) {
        r << "# rad-autoquant: a KV layout per attention layer, the fewest bits measured within a "
             "mean KL of " << max_kl_arg;
        if (!std::isnan(lim.min_top1)) r << " and a top-1 agreement of " << min_top1_arg;
        r << (strict ? " (95% bounds)\n" : "\n");
    } else {
        r << "# rad-autoquant: a KV layout per attention layer within " << bits_arg
          << " bits an element\n";
    }
    r << "# reference " << ref << ", " << manifest["docs"].size()
      << " documents, the shortest " << shortest << " tokens\n";
    for (size_t c = 0; c < C; ++c)
        r << "# every layer " << variants[c].words << ": " << run.describe(*uni[c], false)
          << (c == b0 ? " (the base)" : "") << "\n";
    for (const std::string& d : dropped) r << "# no per-layer runs: " << d << "\n";
    if (search && kept.size() > 1) {
        r << "# each layer alone in a variant, the rest the base: mean KL above that\n#   layer";
        for (size_t k : kept)
            if (k != b0) {
                std::snprintf(line, sizeof line, " %*s",
                              (int)std::max<size_t>(12, tag(variants[k]).size()), tag(variants[k]).c_str());
                r << line;
            }
        r << "\n";
        for (size_t i = 0; i < L; ++i) {
            std::snprintf(line, sizeof line, "#   %5d", layers[i]);
            r << line;
            for (size_t c = 0; c < kept.size(); ++c)
                if (kept[c] != b0) {
                    std::snprintf(line, sizeof line, " %*.3e",
                                  (int)std::max<size_t>(12, tag(variants[kept[c]]).size()),
                                  cand[i][c].kl);
                    r << line;
                }
            r << "\n";
        }
    }
    r << why.str();
    r << run.text(chosen);

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
    std::printf("rad-autoquant: %s: %.4f bits an element, %s mean KL %.6f%s\n", out.c_str(), avg,
                by_kl ? "measured" : "predicted", reported,
                by_kl ? (" (" + chosen_desc + ")").c_str() : "");
    return 0;
}
