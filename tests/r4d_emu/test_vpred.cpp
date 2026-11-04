/* tests/r4d_emu: a `vpred` layer's map (--kv-recipe `vpred`, RAD_KVQ_VPRED; docs/OPS.md
 * "kv_vpred") on the host SIMT emulator, held to libref byte for byte. libref is the oracle, so
 * the test needs it: ctest sets $RADIANCE_HOME, and without one the test fails rather than
 * checking nothing.
 *
 * A `vpred` layer issues libr4d's kv_vpred_bf16 twice:
 *
 *   - the store's side, in place on V: V less the prediction from K-hat, the K the k4h cache will
 *     hold (k_quant k4h, sign -1, gqa 1), over one and two KV heads;
 *   - the output's side, in place on the attention output: plus the prediction from P K-hat (the
 *     `_pk` attention; k_quant none, sign +1), 24 query heads over 2 KV heads (one rank), 12 over
 *     1 (two) and 6 over 1 (four).
 *
 * Each at a decode's rows, a verify's and a prefill chunk's, with x at a row pitch wider than its
 * heads (a column slice, as K is of a fused row), and the store's K holding k4h's edges: a zero
 * group (scale 0), rint's ties (x.5 steps of a power-of-two scale), a group past the scale cap,
 * an infinity and a NaN. The kernel keeps libref's order -- the k4h encode and decode, b first,
 * fma over A's rows ascending, one rounding -- so every element must be libref's bytes. A negative
 * control corrupts one element of the reference and requires the comparison to see it. The P
 * K-hat attention itself is r4d_emu_k4h's (--layout k4h_v4_pk / k4h_v2h_pk). */
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

extern "C" {
int r4d_rad_kv_vpred_bf16(const RadArgs*, RadStream);
}

namespace {

constexpr int HD = 256;

int g_fails = 0;
void fail(const std::string& what) {
    ++g_fails;
    if (g_fails <= 40) std::printf("FAIL %s\n", what.c_str());
}

using Fn = int (*)(const RadArgs*, RadStream);

/* libref's row of op `op`, through the plugin's own exports, or null. */
Fn load_libref(const char* op) {
    static void* h = nullptr;
    if (!h) {
        const char* home = std::getenv("RADIANCE_HOME");
        if (!home || !*home) return nullptr;
        const std::string p = std::string(home) + "/kernels/libref.so";
        h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) return nullptr;
        auto open = (int (*)(void))dlsym(h, "rad_plugin_open"); /* as the loader does */
        if (open && open() < 0) return nullptr;
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

RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = p;
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

struct Params {
    std::vector<RadParam> p;
    void i(const char* k, long long v) { RadParam x{}; x.key = k; x.kind = RAD_P_INT; x.ival = v; p.push_back(x); }
    void s(const char* k, const char* v) { RadParam x{}; x.key = k; x.kind = RAD_P_STR; x.sval = v; p.push_back(x); }
};

/* One kv_vpred launch: x [M, nh * HD] at row pitch `xld`, the map, y and out the same buffer. */
int launch(Fn fn, const uint16_t* x, int64_t xld, const float* map, uint16_t* yo, int64_t M,
           int64_t nh, int64_t gqa, bool k4) {
    RadTensor t[4];
    t[0] = tens((void*)x, RAD_BF16, { M, nh * HD });
    t[0].stride[0] = xld;
    t[1] = tens((void*)map, RAD_F32, { nh / gqa, HD + 1, HD });
    t[2] = tens(yo, RAD_BF16, { M, nh * HD });
    t[3] = tens(yo, RAD_BF16, { M, nh * HD });
    Params p;
    p.i("M", M); p.i("head_dim", HD); p.i("n_head", nh); p.i("gqa", gqa);
    p.s("k_quant", k4 ? "k4h" : "none"); p.i("sign", k4 ? -1 : 1); p.s("dtype", "bf16");
    RadArgs a{};
    a.t = t;
    a.n_t = 4;
    a.p = p.p.data();
    a.n_p = (int)p.p.size();
    return fn(&a, nullptr);
}

void vpred_case(Fn ref, const char* name, int64_t M, int64_t nh, int64_t gqa, bool k4,
                std::mt19937& rng) {
    const int64_t w = nh * HD, xld = w + 64, nm = nh / gqa;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> x((size_t)(M * xld), 0x7fc0), y((size_t)(M * w));
    std::vector<float> map((size_t)(nm * (HD + 1) * HD));
    for (int64_t r = 0; r < M; ++r)
        for (int64_t i = 0; i < w; ++i) x[(size_t)(r * xld + i)] = rad_f32_to_bf16(2.0f * nd(rng));
    for (auto& e : y) e = rad_f32_to_bf16(nd(rng));
    for (auto& e : map) e = 0.1f * nd(rng);
    if (k4 && M >= 2) {
        uint16_t* r0 = x.data();
        for (int i = 0; i < 32; ++i) r0[i] = 0;                                   /* scale 0 */
        /* scale 1.75 / 7 = 0.25 exactly, then x.5 steps, which rint takes to even */
        const float tie[8] = { 0.125f, 0.375f, 0.625f, -0.125f, -0.375f, 1.75f, -1.75f, 0.875f };
        for (int i = 0; i < 32; ++i) r0[32 + i] = rad_f32_to_bf16(i < 8 ? tie[i] : 0.0f);
        for (int i = 0; i < 32; ++i)                                              /* past the cap */
            r0[64 + i] = rad_f32_to_bf16((i & 1 ? -1.f : 1.f) * 1e5f * (1 + i));
        r0[96 + 3] = 0x7f80;                                                      /* +inf */
        x[(size_t)(xld + 128 + 5)] = 0x7fc0;                                      /* a NaN */
    }
    std::vector<uint16_t> want = y, got = y;
    const int rr = launch(ref, x.data(), xld, map.data(), want.data(), M, nh, gqa, k4);
    const int re = launch(r4d_rad_kv_vpred_bf16, x.data(), xld, map.data(), got.data(), M, nh, gqa, k4);
    if (rr != RAD_OK || re != RAD_OK) {
        fail(std::string(name) + ": libref " + std::to_string(rr) + ", emulated " + std::to_string(re));
        return;
    }
    int64_t diff = 0, moved = 0, nan = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        diff += want[i] != got[i];
        moved += want[i] != y[i];
        nan += std::isnan(rad_bf16_to_f32(want[i]));
    }
    /* The edges stay finite: k4h's clamp takes inf and NaN to +-7 codes. */
    if (diff || nan || moved < (int64_t)want.size() / 2)
        fail(std::string(name) + ": " + std::to_string(diff) + " elements differ from libref, " +
             std::to_string(moved) + " moved, " + std::to_string(nan) + " NaN");
    /* Negative control: one corrupted element of the reference must show. */
    want[want.size() / 3] ^= 0x0001;
    int64_t seen = 0;
    for (size_t i = 0; i < want.size(); ++i) seen += want[i] != got[i];
    if (seen != diff + 1) fail(std::string(name) + ": the comparison missed a corrupted element");
    std::printf("  %-52s %3lld rows x %2lld heads: %s\n", name, (long long)M, (long long)nh,
                diff ? "DIFFERS" : "libref's bytes");
}

}  // namespace

int main() {
    const Fn ref = load_libref("kv_vpred");
    if (!ref) {
        std::printf("FAIL libref's kv_vpred not loadable (set RADIANCE_HOME)\n");
        return 1;
    }
    std::mt19937 rng(20261120);
    std::printf("kv_vpred_bf16: the emulated kernel against libref\n");
    for (int64_t M : { 1, 4, 23 }) {
        vpred_case(ref, "V less its prediction from K-hat, two KV heads", M, 2, 1, true, rng);
        vpred_case(ref, "V less its prediction from K-hat, one KV head", M, 1, 1, true, rng);
        vpred_case(ref, "output plus the prediction, 24 heads over 2", M, 24, 12, false, rng);
        vpred_case(ref, "output plus the prediction, 12 heads over 1", M, 12, 12, false, rng);
        vpred_case(ref, "output plus the prediction, 6 heads over 1", M, 6, 6, false, rng);
    }
    std::printf(g_fails ? "kv_vpred: %d failure(s)\n" : "kv_vpred: all cases pass\n", g_fails);
    return g_fails ? 1 : 0;
}
