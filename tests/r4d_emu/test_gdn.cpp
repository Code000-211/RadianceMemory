/* tests/r4d_emu: the delta net's state at f16 (--state-cache-dtype f16, GUIDE §4.4) in libr4d's gdn
 * kernels on the host SIMT emulator, at Qwen3.8-Flash-Next's geometry: a --tp 2 rank's 8 key and
 * 24 value heads of 128, the sigmoid output gate, int8 codes out of the recurrence. The kernels
 * widen the cache to f32 where they load it and round to f16 where they store it, and nothing else
 * differs (SDT), so each case holds:
 *
 *   - the kernel over an f16 cache writes the same `o` (codes and scales, the conv window) as over
 *     that cache widened to f32, byte for byte, and leaves in every cache element the f32 run's
 *     value rounded to nearest-even f16, the anchor form's factor blocks and stamps included;
 *   - against libref (from $RADIANCE_HOME, which ctest sets) over the same operands, `o` and the
 *     cache no further from libref's f16 path, in relative L2, than the f32 run is from libref's
 *     f32 path, plus 1e-3: the kernel reduces in another order (and the scan feeds its WMMA bf16
 *     operands) at either width, and an element at an f16 rounding tie may land an ulp apart.
 *
 * The cases are the ops Flash-Next issues on its delta-net layers: the recurrence per candidate
 * (plain decode) and anchored (the MTP head's verify: one to three tokens replayed out of the drawn
 * factor blocks), the convolution folded into it (anchored, its decode form), and the chunked scan
 * of a prefill (two sequences of 100 and 37 rows, so partial chunks, from cache slots 2 and 0; two
 * key heads and six value heads keep the emulated WMMA quick). r4d_selftest runs the same checks
 * on the card (gdn_f16_state, and gdn_conv_fold at an f16 state against the unfused pair). */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <string>
#include <vector>

extern "C" int r4d_rad_gdn_recurrent_update(const RadArgs* a, RadStream stream);
extern "C" int r4d_rad_gdn_conv_recurrent_update(const RadArgs* a, RadStream stream);
extern "C" int r4d_rad_gdn_chunk_scan(const RadArgs* a, RadStream stream);

namespace {

int g_fails = 0;
void fail(const std::string& what) {
    ++g_fails;
    std::printf("FAIL %s\n", what.c_str());
}

/* libref's row for `op`, through the plugin's own exports, or null. */
RadLaunchFn load_libref(const char* op) {
    static void* h = nullptr;
    if (!h) {
        const char* home = std::getenv("RADIANCE_HOME");
        if (!home || !*home) return nullptr;
        const std::string p = std::string(home) + "/kernels/libref.so";
        h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) return nullptr;
        auto open = (int (*)(void))dlsym(h, "rad_plugin_open"); /* as the loader does */
        if (open && open() < 0) { h = nullptr; return nullptr; }
    }
    auto cnt = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return nullptr;
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* k = at(i);
        if (k && k->op && !std::strcmp(k->op, op)) return k->launch;
    }
    return nullptr;
}

RadTensor tens(const void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = const_cast<void*>(p);
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[(size_t)i];
        t.stride[i] = st;
        st *= shape[(size_t)i];
    }
    return t;
}
RadParam pi(const char* k, int64_t v) { return { k, RAD_P_INT, v, 0, nullptr, 0 }; }
RadParam pf(const char* k, double v) { return { k, RAD_P_F64, 0, 0, nullptr, v }; }
RadParam ps(const char* k, const char* v) { return { k, RAD_P_STR, 0, 0, v, 0 }; }

enum Op { RECUR, FOLD, SCAN };

struct Case {
    const char* name;
    Op op;
    const char* form;      /* the recurrence's state_form; null for the scan */
    int64_t Hg, H;
};

/* Every operand of one case on the host; `st` and the outputs are per run. */
struct Data {
    static constexpr int64_t K = 128, V = 128, MQ = 4, CH = 64, W = 4, SL = W - 2 + MQ;
    const Case& c;
    int64_t N, Td, cols, slots, conv_dim, cslots = 0;
    std::vector<int32_t> cu, sidx, nacc, cidx;
    std::vector<uint16_t> q, k, v, z, a, b, x, cw, A, cs, st16;
    std::vector<float> alog, dtb, nw, g, beta;

    Data(const Case& cs_, std::mt19937& rng) : c(cs_) {
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        const bool scan = c.op == SCAN, anc = !scan && !std::strcmp(c.form, "anchor");
        const int32_t lr[3] = { 1, 4, 2 }, na[3] = { 1, 2, 3 }, ls[2] = { 100, 37 };
        N = scan ? 2 : 3;
        cu.push_back(0);
        for (int64_t n = 0; n < N; ++n) cu.push_back(cu.back() + (scan ? ls[n] : lr[n]));
        Td = cu.back();
        cols = scan ? 1 : (anc ? 6 : MQ);
        slots = scan ? 3 : N * cols;
        conv_dim = 2 * c.Hg * K + c.H * V;
        auto bf = [&](int64_t n, float s) {
            std::vector<uint16_t> r((size_t)n);
            for (auto& e : r) e = rad_f32_to_bf16(s * u(rng));
            return r;
        };
        st16.resize((size_t)(slots * c.H * V * K));
        for (auto& e : st16) e = rad_f32_to_f16(0.1f * u(rng));
        q = bf(Td * c.Hg * K, 1.0f); k = bf(Td * c.Hg * K, 1.0f);
        v = bf(Td * c.H * V, 1.0f); z = bf(Td * c.H * V, 1.0f);
        a = bf(Td * c.H, 1.0f); b = bf(Td * c.H, 1.0f);
        for (int64_t h = 0; h < c.H; ++h) { alog.push_back(-2.0f + 0.0625f * (float)h);
                                            dtb.push_back(-1.0f + 0.0625f * (float)h); }
        for (int64_t e = 0; e < V; ++e) nw.push_back(1.0f + 0.25f * u(rng));
        if (scan) {
            /* As gdn_conv_prep and gdn_kkt_solve leave them: unit-norm k, the gate's cumulative
             * sum restarting each chunk, beta in (0, 1), A unit lower-triangular in a chunk. */
            for (int64_t i = 0; i < Td * c.Hg; ++i) {
                double s = 0;
                for (int64_t j = 0; j < K; ++j)
                    s += (double)rad_bf16_to_f32(k[(size_t)(i * K + j)]) *
                         rad_bf16_to_f32(k[(size_t)(i * K + j)]);
                for (int64_t j = 0; j < K; ++j)
                    k[(size_t)(i * K + j)] = rad_f32_to_bf16(
                        (float)(rad_bf16_to_f32(k[(size_t)(i * K + j)]) / std::sqrt(s)));
            }
            A.assign((size_t)(Td * c.H * CH), 0);
            for (int64_t n = 0; n < N; ++n)
                for (int64_t t = cu[(size_t)n]; t < cu[(size_t)n + 1]; ++t) {
                    const int64_t i = (t - cu[(size_t)n]) % CH;
                    for (int64_t h = 0; h < c.H; ++h) {
                        g.push_back(-0.02f * (float)(i + 1));
                        beta.push_back(0.5f + 0.4f * u(rng));
                        for (int64_t j = 0; j < i; ++j)
                            A[(size_t)((t * c.H + h) * CH + j)] = rad_f32_to_bf16(0.05f * u(rng));
                        A[(size_t)((t * c.H + h) * CH + i)] = rad_f32_to_bf16(1.0f);
                    }
                }
            sidx = { 2, 0 };
        } else {
            for (int64_t i = 0; i < N * cols; ++i) sidx.push_back((int32_t)i);
            nacc.assign(na, na + N);
        }
        if (c.op == FOLD) {
            /* The fold's convolution: a conv slot a sequence, the second at slot 0. */
            cslots = 4;
            x = bf(Td * conv_dim, 1.0f);
            cw = bf(conv_dim * W, 0.5f);
            cs = bf(cslots * conv_dim * SL, 1.0f);
            cidx = { 3, 0, 1 };
        }
    }

    /* The operand list over a state cache `st` of `sdt` and outputs `o`, `oq`, `os`, the conv
     * window `cwin` (the fold's); `oq` null for libref, whose quantiser is the standalone one's. */
    std::vector<RadTensor> operands(void* st, uint32_t sdt, void* o, void* oq, void* os,
                                    void* cwin) const {
        const int64_t H = c.H, Hg = c.Hg;
        std::vector<RadTensor> t;
        const RadTensor tst = tens(st, sdt, { slots, H, V, K });
        if (c.op == SCAN) {
            t = { tens(q.data(), RAD_BF16, { Td, Hg, K }), tens(k.data(), RAD_BF16, { Td, Hg, K }),
                  tens(v.data(), RAD_BF16, { Td, H, V }), tens(A.data(), RAD_BF16, { Td, H, CH }),
                  tens(g.data(), RAD_F32, { Td, H }), tens(beta.data(), RAD_F32, { Td, H }), tst,
                  tens(cu.data(), RAD_I32, { N + 1 }), tens(o, RAD_BF16, { Td, H, V }), tst,
                  tens(sidx.data(), RAD_I32, { N }) };
            return t;
        }
        if (c.op == FOLD)
            t = { tens(x.data(), RAD_BF16, { Td, conv_dim }),
                  tens(cw.data(), RAD_BF16, { conv_dim, W }), RadTensor{},
                  tens(cwin, RAD_BF16, { cslots, conv_dim, SL }),
                  tens(cidx.data(), RAD_I32, { N }) };
        else
            t = { tens(q.data(), RAD_BF16, { Td, Hg, K }), tens(k.data(), RAD_BF16, { Td, Hg, K }),
                  tens(v.data(), RAD_BF16, { Td, H, V }) };
        const RadTensor rest[] = {
            tens(a.data(), RAD_BF16, { Td, H }), tens(b.data(), RAD_BF16, { Td, H }),
            tens(alog.data(), RAD_F32, { H }), tens(dtb.data(), RAD_F32, { H }), tst,
            tens(sidx.data(), RAD_I32, { N, cols }), tens(nacc.data(), RAD_I32, { N }),
            tens(cu.data(), RAD_I32, { N + 1 }), tens(z.data(), RAD_BF16, { Td, H, V }),
            tens(nw.data(), RAD_F32, { V }), tens(o, RAD_BF16, { Td, H, V }) };
        t.insert(t.end(), std::begin(rest), std::end(rest));
        if (oq) {
            t.push_back(tens(oq, RAD_I8, { Td, H * V }));
            t.push_back(tens(os, RAD_F32, { Td, H }));
        }
        return t;
    }

    std::vector<RadParam> params() const {
        if (c.op == SCAN)
            return { pi("M", Td), pi("head_k", K), pi("head_v", V), pi("chunk", CH),
                     pf("scale", 0.088388) };
        std::vector<RadParam> p = { pi("q_len", MQ), pi("head_k", K), pi("head_v", V),
                                    pi("n_head_v", c.H), pf("scale", 0.088388), pf("eps", 1e-6),
                                    ps("act", "sigmoid"), ps("state_form", c.form) };
        if (c.op == FOLD) p.push_back(pi("conv_width", W));
        return p;
    }
};

int launch(RadLaunchFn fn, std::vector<RadTensor> t, std::vector<RadParam> p) {
    RadArgs a{};
    a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size();
    a.world_size = 1;
    return fn(&a, nullptr);
}

double rel_l2(const std::vector<float>& got, const std::vector<float>& want) {
    double e = 0, r = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        e += ((double)got[i] - want[i]) * ((double)got[i] - want[i]);
        r += (double)want[i] * want[i];
    }
    return std::sqrt(e / (r > 0 ? r : 1));
}

void run_case(const Case& c, std::mt19937& rng) {
    const Data d(c, rng);
    const RadLaunchFn emu = c.op == RECUR ? r4d_rad_gdn_recurrent_update
                          : c.op == FOLD  ? r4d_rad_gdn_conv_recurrent_update
                                          : r4d_rad_gdn_chunk_scan;
    const char* op = c.op == RECUR ? "gdn_recurrent_update"
                   : c.op == FOLD  ? "gdn_conv_recurrent_update" : "gdn_chunk_scan";
    const RadLaunchFn ref = load_libref(op);
    if (!ref) { fail(std::string(c.name) + ": libref's " + op + " not loadable (set RADIANCE_HOME)"); return; }

    const size_t cache = d.st16.size(), on = (size_t)(d.Td * c.H * Data::V);
    std::vector<float> st32(cache);
    for (size_t i = 0; i < cache; ++i) st32[i] = rad_f16_to_f32(d.st16[i]);
    std::vector<uint16_t> st16 = d.st16, cw32 = d.cs, cw16 = d.cs;
    std::vector<uint16_t> o32(on), o16(on);
    std::vector<int8_t> q32(on), q16(on);
    std::vector<float> s32((size_t)(d.Td * c.H)), s16 = s32;
    const bool codes = c.op != SCAN;
    const int r32 = launch(emu, d.operands(st32.data(), RAD_F32, o32.data(),
                                           codes ? q32.data() : nullptr, s32.data(), cw32.data()),
                           d.params());
    const int r16 = launch(emu, d.operands(st16.data(), RAD_F16, o16.data(),
                                           codes ? q16.data() : nullptr, s16.data(), cw16.data()),
                           d.params());
    if (r32 != RAD_OK || r16 != RAD_OK) {
        fail(std::string(c.name) + ": refused, f32 " + std::to_string(r32) + ", f16 " +
             std::to_string(r16));
        return;
    }
    std::string why;
    if (std::memcmp(o16.data(), o32.data(), on * 2)) why = "o differs from the f32 run's";
    else if (codes && (std::memcmp(q16.data(), q32.data(), on) ||
                       std::memcmp(s16.data(), s32.data(), s16.size() * 4)))
        why = "the int8 codes or scales differ from the f32 run's";
    else if (cw16 != cw32) why = "the conv window differs from the f32 run's";
    int64_t miss = 0, moved = 0;
    for (size_t i = 0; i < cache; ++i) {
        miss += st16[i] != rad_f32_to_f16(st32[i]);
        moved += st16[i] != d.st16[i];
    }
    if (why.empty() && miss)
        why = std::to_string(miss) + " of " + std::to_string(cache) +
              " cache elements are not the f32 run's rounded to f16";
    /* Each sequence's state at least: a run that stored nothing agrees with one that did not. */
    if (why.empty() && moved < d.N * c.H * Data::V * Data::K / 2) why = "the f16 cache barely moved";
    if (!why.empty()) { fail(std::string(c.name) + ": " + why); return; }

    /* libref over the same operands, at f16 and at f32: the kernel's distance from libref's f16
     * path is held to its distance at f32 (its own reduction order and, in the scan, its bf16
     * WMMA operands), plus 1e-3 for the f16 roundings that land an ulp apart. */
    double lo[2], ls[2];
    for (int f16 = 0; f16 < 2; ++f16) {
        std::vector<uint16_t> r16 = d.st16, rcw = d.cs, ro(on);
        std::vector<float> r32(st32.size());
        for (size_t i = 0; i < cache; ++i) r32[i] = rad_f16_to_f32(d.st16[i]);
        const int rr = launch(ref, d.operands(f16 ? (void*)r16.data() : (void*)r32.data(),
                                              f16 ? RAD_F16 : RAD_F32, ro.data(), nullptr, nullptr,
                                              rcw.data()), d.params());
        if (rr != RAD_OK) { fail(std::string(c.name) + ": libref refused, " + std::to_string(rr)); return; }
        if (rcw != (f16 ? cw16 : cw32))
            fail(std::string(c.name) + ": the conv window differs from libref's");
        const std::vector<uint16_t>& eo16 = f16 ? o16 : o32;
        std::vector<float> eo(on), wo(on), es(cache), ws(cache);
        for (size_t i = 0; i < on; ++i) { eo[i] = rad_bf16_to_f32(eo16[i]); wo[i] = rad_bf16_to_f32(ro[i]); }
        for (size_t i = 0; i < cache; ++i) {
            es[i] = f16 ? rad_f16_to_f32(st16[i]) : st32[i];
            ws[i] = f16 ? rad_f16_to_f32(r16[i]) : r32[i];
        }
        lo[f16] = rel_l2(eo, wo);
        ls[f16] = rel_l2(es, ws);
    }
    if (!(lo[1] <= lo[0] + 1e-3) || !(ls[1] <= ls[0] + 1e-3))
        fail(std::string(c.name) + ": further from libref at f16 (o " + std::to_string(lo[1]) +
             ", cache " + std::to_string(ls[1]) + ") than at f32 (o " + std::to_string(lo[0]) +
             ", cache " + std::to_string(ls[0]) + ") by more than 1e-3");
    std::printf("  %-48s f32's bytes through f16; from libref, o %.1e (f32 %.1e), cache %.1e "
                "(f32 %.1e)\n", c.name, lo[1], lo[0], ls[1], ls[0]);
}

}  // namespace

int main() {
    std::mt19937 rng(20261214);
    const Case cases[] = {
        { "recurrence per candidate, 8 / 24 heads",            RECUR, "per_candidate", 8, 24 },
        { "recurrence anchored, 8 / 24 heads",                 RECUR, "anchor",        8, 24 },
        { "convolution folded in, anchored, 8 / 24 heads",     FOLD,  "anchor",        8, 24 },
        { "chunked scan, 100 and 37 rows, 2 / 6 heads",        SCAN,  nullptr,         2, 6 },
    };
    std::printf("gdn at an f16 state: the emulated kernels against their f32 runs and libref\n");
    for (const Case& c : cases) run_case(c, rng);
    std::printf(g_fails ? "gdn f16 state: %d failure(s)\n" : "gdn f16 state: all cases pass\n",
                g_fails);
    return g_fails ? 1 : 0;
}
