/* tests/r4d_emu: the hierarchical QSA indexer's restriction (`qsa_pool_mask`, --qsa-pool) on the
 * host SIMT emulator: libr4d's qsa_pool_mask_f32 (r4d_qsa_pool_mask.hip) against libref's
 * ref_qsa_pool_mask, the oracle (docs/OPS.md), byte for byte over the whole plane. ctest sets
 * $RADIANCE_HOME; without libref the test fails rather than checking nothing.
 *
 * The fixtures: one decode row and several prefill rows; widths below, at and past the kernel's
 * 256 threads; a pool wider than the threads (its entries walk in strides); the operand narrowed to
 * a column slice of a wider plane (the live width, rad_qsa.h), whose columns past it must keep
 * their bytes; repeated entries, -1 padding and entries past the row; scores that are NaN, -0, inf
 * and -inf, each kept bit for bit when named. A pool naming every column leaves the plane as it
 * was. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

namespace {

int g_fails = 0;
void fail(const std::string& what) {
    ++g_fails;
    if (g_fails <= 40) std::printf("FAIL %s\n", what.c_str());
}

/* libref's row for `name`, through the plugin's own exports, or null (as test_bkey.cpp). */
RadLaunchFn libref(const char* name) {
    static void* h = nullptr;
    if (!h) {
        const char* home = std::getenv("RADIANCE_HOME");
        if (!home || !*home) return nullptr;
        const std::string p = std::string(home) + "/kernels/libref.so";
        h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) return nullptr;
        auto open = (int (*)(void))dlsym(h, "rad_plugin_open");
        if (open && open() < 0) { h = nullptr; return nullptr; }
    }
    auto cnt = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return nullptr;
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* k = at(i);
        if (k && k->op && !std::strcmp(k->op, name)) return k->launch;
    }
    return nullptr;
}

/* A [rows, cols] operand over a plane `pitch` wide (a column slice when cols < pitch). */
RadTensor view(void* p, uint32_t dt, int64_t rows, int64_t cols, int64_t pitch) {
    RadTensor t{};
    t.data = p; t.dtype = dt; t.rank = 2;
    t.shape[0] = rows; t.shape[1] = cols;
    t.stride[0] = pitch; t.stride[1] = 1;
    return t;
}

struct Fixture {
    const char* what;
    int64_t M, cols, pitch, np;
    bool all;   /* the pool names every column */
};

/* Runs `fn` over a copy of the plane; returns the status and the plane's bytes after. */
int launch(RadLaunchFn fn, const Fixture& f, const std::vector<int32_t>& pool,
           std::vector<float>& plane) {
    RadParam ps[3] = { { "M", RAD_P_INT, f.M }, { "blocks", RAD_P_INT, f.pitch },
                       { "pool", RAD_P_INT, f.np } };
    RadTensor t[2] = { view((void*)pool.data(), RAD_I32, f.M, f.np + 1, f.np + 1),
                       view(plane.data(), RAD_F32, f.M, f.cols, f.pitch) };
    RadArgs a{};
    a.t = t; a.n_t = 2; a.p = ps; a.n_p = 3;
    return fn(&a, nullptr);
}

void run(RadLaunchFn ref, const Fixture& f, uint32_t seed) {
    uint32_t s = seed;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return s >> 8; };
    std::vector<float> plane((size_t)(f.M * f.pitch));
    const float special[] = { std::nanf(""), -0.0f, INFINITY, -INFINITY };
    for (size_t i = 0; i < plane.size(); ++i)
        plane[i] = (i % 17 == 5) ? special[(i / 17) % 4] : (float)(rnd() % 2000) / 64.0f - 4.0f;
    std::vector<int32_t> pool((size_t)(f.M * (f.np + 1)));
    for (int64_t m = 0; m < f.M; ++m)
        for (int64_t j = 0; j <= f.np; ++j) {
            int32_t& e = pool[(size_t)(m * (f.np + 1) + j)];
            if (f.all) e = (int32_t)(j % f.cols);
            else if (j % 11 == 10) e = -1;
            else if (j % 13 == 12) e = (int32_t)(f.cols + rnd() % 5);     /* past the row */
            else e = (int32_t)(rnd() % f.cols);                           /* repeats likely */
        }
    std::vector<float> got = plane, want = plane;
    const int rg = launch(r4d_rad_qsa_pool_mask, f, pool, got);
    const int rr = launch(ref, f, pool, want);
    if (rg != RAD_OK || rr != RAD_OK) {
        fail(std::string(f.what) + ": launch " + std::to_string(rg) + ", libref " +
             std::to_string(rr));
        return;
    }
    if (std::memcmp(got.data(), want.data(), got.size() * sizeof(float))) {
        size_t i = 0;
        while (std::memcmp(&got[i], &want[i], sizeof(float)) == 0) ++i;
        fail(std::string(f.what) + ": first differing element " + std::to_string(i));
    }
    if (f.all && std::memcmp(got.data(), plane.data(), got.size() * sizeof(float)))
        fail(std::string(f.what) + ": a pool naming every column changed the plane");
    /* Columns past the operand are never touched. */
    for (int64_t m = 0; m < f.M; ++m)
        for (int64_t b = f.cols; b < f.pitch; ++b) {
            const size_t i = (size_t)(m * f.pitch + b);
            if (std::memcmp(&got[i], &plane[i], sizeof(float)))
                fail(std::string(f.what) + ": a column past the operand was written");
        }
}

}  /* namespace */

int main() {
    const RadLaunchFn ref = libref("qsa_pool_mask");
    if (!ref) {
        std::printf("FAIL libref's qsa_pool_mask not loadable (set RADIANCE_HOME)\n");
        return 1;
    }
    std::printf("qsa_pool_mask: the emulated kernel against libref\n");
    const Fixture fx[] = {
        { "one decode row, a narrow plane", 1, 40, 40, 8, false },
        { "four rows, a column slice of a wider plane", 4, 300, 320, 32, false },
        { "a pool wider than the threads", 3, 700, 700, 600, false },
        { "the row exactly the threads wide", 2, 256, 260, 255, false },
        { "a pool naming every column", 2, 90, 100, 120, true },
        { "a single-column row", 2, 1, 4, 3, false },
    };
    uint32_t seed = 7;
    for (const Fixture& f : fx) run(ref, f, seed += 101);

    /* Refusals the device path must share with libref: a pool narrower than pool + 1 entries, and
     * a pool wider than the kernel's LDS. */
    {
        std::vector<int32_t> pool(4, 0);
        std::vector<float> plane(8, 1.0f);
        RadParam ps[3] = { { "M", RAD_P_INT, 1 }, { "blocks", RAD_P_INT, 8 }, { "pool", RAD_P_INT, 4 } };
        RadTensor t[2] = { view(pool.data(), RAD_I32, 1, 4, 4), view(plane.data(), RAD_F32, 1, 8, 8) };
        RadArgs a{};
        a.t = t; a.n_t = 2; a.p = ps; a.n_p = 3;
        if (r4d_rad_qsa_pool_mask(&a, nullptr) != RAD_E_SHAPE || ref(&a, nullptr) != RAD_E_SHAPE)
            fail("a pool of fewer than pool + 1 entries is refused by both");
        ps[2].ival = 4096;
        if (r4d_rad_qsa_pool_mask(&a, nullptr) != RAD_E_SHAPE)
            fail("a pool past the kernel's 4095 entries is refused");
    }
    std::printf(g_fails ? "qsa_pool_mask: %d failure(s)\n" : "qsa_pool_mask: all cases pass\n",
                g_fails);
    return g_fails ? 1 : 0;
}
