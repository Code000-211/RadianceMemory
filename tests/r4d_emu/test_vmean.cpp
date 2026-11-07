/* tests/r4d_emu: a `vmean` layer's arithmetic (--kv-recipe `vmean`, RAD_KVQ_VMEAN) on the host
 * SIMT emulator, held to libref byte for byte. libref is the oracle (docs/OPS.md), so the test
 * needs it: ctest sets $RADIANCE_HOME, and without one the test fails rather than checking nothing.
 *
 * A centred layer adds two things to what it ran before, both libr4d's `add_bf16` with its one
 * broadcast form (a row of exactly n elements down every row):
 *
 *   - -mu by KV head onto V's rows before the store, in place: n = kv_dim, 512 at one rank of
 *     Flash-Next and 256 at two and four;
 *   - +mu by query head onto the attention output's rows before the rotation back, in place:
 *     n = q_dim, 6144, 3072 and 1536.
 *
 * Each is held to libref's `add` at those widths, at a decode's rows, a verify's, a prefill
 * chunk's, and an odd count, over V-like values with a large per-channel offset and the mean's
 * bf16 row, in place and into another buffer. Then the store behind it: libr4d's k4h_v4 and
 * k4h_v2h stores over the centred V are libref's bytes over the whole cache (the fp8_v4 and k6_v4
 * stores have no libref oracle; their rows are r4d_selftest's). The attention kernels read the
 * cache as before, so r4d_emu_k4h's cases cover them unchanged. */
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
int r4d_rad_add(const RadArgs*, RadStream);
int r4d_rad_kv_store_k4hv4(const RadArgs*, RadStream);
int r4d_rad_kv_store_k4hv2h(const RadArgs*, RadStream);
}

namespace {

constexpr int HD = 256, BS = 4;

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

int launch(Fn fn, RadTensor* t, int n_t, Params& p) {
    RadArgs a{};
    a.t = t;
    a.n_t = n_t;
    a.p = p.p.data();
    a.n_p = (int)p.p.size();
    return fn(&a, nullptr);
}

/* y = a + b over [M, n] bf16 with b one row of n, `y` in place over `a` or apart. */
int add(Fn fn, uint16_t* a, const uint16_t* b, uint16_t* y, int64_t M, int64_t n) {
    RadTensor t[3] = { tens(a, RAD_BF16, { M, n }), tens((void*)b, RAD_BF16, { n }),
                       tens(y, RAD_BF16, { M, n }) };
    Params p;
    p.i("M", M); p.i("n", n); p.s("dtype", "bf16");
    return launch(fn, t, 3, p);
}

/* What a layer's V or output looks like: a per-channel offset (Flash-Next's mean is 0.53 to 0.89
 * of V's RMS) plus spread, and the bf16 mean the engine's table holds, negated for V. */
void draw(std::vector<uint16_t>& x, std::vector<uint16_t>& row, int64_t M, int64_t n, bool neg,
          std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> off((size_t)n);
    for (auto& o : off) o = 0.8f * nd(rng);
    x.resize((size_t)(M * n));
    row.resize((size_t)n);
    for (int64_t r = 0; r < M; ++r)
        for (int64_t j = 0; j < n; ++j)
            x[(size_t)(r * n + j)] = rad_f32_to_bf16(off[(size_t)j] + 0.6f * nd(rng));
    for (int64_t j = 0; j < n; ++j) {
        const uint16_t m = rad_f32_to_bf16(off[(size_t)j] * (1.0f + 0.05f * nd(rng)));
        row[(size_t)j] = neg ? (uint16_t)(m ^ 0x8000u) : m;
    }
}

void add_case(Fn ref, int64_t M, int64_t n, bool neg, bool in_place, std::mt19937& rng) {
    std::vector<uint16_t> x, row;
    draw(x, row, M, n, neg, rng);
    std::vector<uint16_t> a = x, b = x, ya((size_t)(M * n), 0x7fa5), yb = ya;
    const std::string name = std::string(neg ? "V less the mean" : "output plus the mean") +
                             " M=" + std::to_string(M) + " n=" + std::to_string(n) +
                             (in_place ? " in place" : "");
    const int rk = add(r4d_rad_add, a.data(), row.data(), in_place ? a.data() : ya.data(), M, n);
    const int rr = add(ref, b.data(), row.data(), in_place ? b.data() : yb.data(), M, n);
    if (rk != RAD_OK || rr != RAD_OK) {
        fail(name + ": refused (" + std::to_string(rk) + ", libref " + std::to_string(rr) + ")");
        return;
    }
    const std::vector<uint16_t>& got = in_place ? a : ya;
    const std::vector<uint16_t>& want = in_place ? b : yb;
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[i]) {
            fail(name + ": element " + std::to_string(i) + " differs from libref");
            return;
        }
    /* And not trivially: the mean moved the values. */
    size_t moved = 0;
    for (size_t i = 0; i < got.size(); ++i) moved += got[i] != x[i];
    if (moved < got.size() / 2) fail(name + ": the add changed too little");
}

/* kv_store over k and the centred v ([T, kvh * HD]) into a fresh cache at `layout`. */
int store(Fn fn, const char* layout, int row, const uint16_t* k, const uint16_t* v, int64_t T,
          int64_t kvh, const int32_t* slots, uint8_t* cache, int64_t nb) {
    RadTensor t[4] = { tens((void*)k, RAD_BF16, { T, kvh * HD }),
                       tens((void*)v, RAD_BF16, { T, kvh * HD }),
                       tens((void*)slots, RAD_I32, { T }),
                       tens(cache, RAD_F8E4M3, { nb, kvh, BS, row }) };
    Params p;
    p.i("M", T); p.i("head_dim", HD); p.i("n_head_kv", kvh); p.i("block_size", BS);
    p.s("kv_dtype", layout);
    return launch(fn, t, 4, p);
}

void store_case(Fn ref_add, Fn ref_store, Fn kernel, const char* layout, int row, int64_t T,
                int64_t kvh, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> k((size_t)(T * kvh * HD)), v, sub;
    for (auto& x : k) x = rad_f32_to_bf16(nd(rng));
    draw(v, sub, T, kvh * HD, true, rng);
    std::vector<uint16_t> va = v, vb = v;
    const int64_t nb = (T + BS - 1) / BS + 2;
    std::vector<int32_t> slots((size_t)T);
    for (int64_t t = 0; t < T; ++t) slots[(size_t)t] = (int32_t)((t * 5 + 3) % (nb * BS));
    std::vector<uint8_t> ca((size_t)(nb * kvh * BS * row), 0x5a), cb = ca;
    const std::string name = std::string(layout) + " store of a centred V, T=" + std::to_string(T) +
                             " kv heads " + std::to_string(kvh);
    if (add(r4d_rad_add, va.data(), sub.data(), va.data(), T, kvh * HD) != RAD_OK ||
        add(ref_add, vb.data(), sub.data(), vb.data(), T, kvh * HD) != RAD_OK ||
        store(kernel, layout, row, k.data(), va.data(), T, kvh, slots.data(), ca.data(), nb) !=
            RAD_OK ||
        store(ref_store, layout, row, k.data(), vb.data(), T, kvh, slots.data(), cb.data(), nb) !=
            RAD_OK) {
        fail(name + ": refused");
        return;
    }
    for (size_t i = 0; i < ca.size(); ++i)
        if (ca[i] != cb[i]) {
            fail(name + ": cache byte " + std::to_string(i) + " differs from libref");
            return;
        }
}

}  /* namespace */

int main() {
    const Fn ref_add = load_libref("add");
    const Fn ref_store = load_libref("kv_store");
    if (!ref_add || !ref_store) {
        std::printf("FAIL libref's add and kv_store could not be loaded ($RADIANCE_HOME/kernels/"
                    "libref.so)\n");
        return 1;
    }
    std::mt19937 rng(9);
    /* V's widths at one, two and four ranks, and the output's; a decode's rows, a verify's, a
     * prefill chunk's and an odd count. */
    for (int64_t M : { 1, 4, 16, 37, 512 })
        for (bool in_place : { true, false }) {
            for (int64_t n : { 256, 512 }) add_case(ref_add, M, n, true, in_place, rng);
            for (int64_t n : { 1536, 3072, 6144 }) add_case(ref_add, M, n, false, in_place, rng);
        }
    for (int64_t T : { 1, 9, 64 })
        for (int64_t kvh : { 1, 2 }) {
            store_case(ref_add, ref_store, r4d_rad_kv_store_k4hv4, "k4h_v4", 18 * HD / 16, T, kvh,
                       rng);
            store_case(ref_add, ref_store, r4d_rad_kv_store_k4hv2h, "k4h_v2h", 14 * HD / 16, T,
                       kvh, rng);
        }
    if (g_fails) {
        std::printf("r4d_emu_vmean: %d failure(s)\n", g_fails);
        return 1;
    }
    std::printf("r4d_emu_vmean: ok\n");
    return 0;
}
