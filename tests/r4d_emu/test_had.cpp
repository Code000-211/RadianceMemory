/* tests/r4d_emu: had_rotate (libr4d/r4d_had_rotate.hip) on the host SIMT emulator, held to libref's
 * had_rotate byte for byte. libref is the oracle (docs/OPS.md), so the test needs it: ctest sets
 * $RADIANCE_HOME, and without one the test fails rather than checking nothing.
 *
 * The cases are the shapes a rotated KV layer issues and the edges of the kernel's walk:
 *
 *   - q as the attention block issues it: [T, n_head * 256] read as rows of one head, n = 256;
 *   - k beside it: a step's rows of a wider buffer, pitched, whose gaps must stay untouched;
 *   - the 128-wide row, and a row of two groups (n = 2 * group), which rotates each group alone;
 *   - one block, which leaves seven of the workgroup's eight waves with nothing to do;
 *   - a few hundred blocks over values from 1e-3 to 1e3 with one outlier channel, the input the
 *     rotation exists for.
 *
 * Inputs are finite bf16: the transform of an infinity is a NaN, whose payload the device's and
 * the host's arithmetic propagate differently, and no K, V or query row the engine rotates holds
 * one. The two shapes the rows do not serve (group 64, an f32 row) must be refused, not rotated. */
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

extern "C" int r4d_rad_had_rotate(const RadArgs* a, RadStream stream);

namespace {

int g_fails = 0;
void fail(const std::string& what) {
    ++g_fails;
    if (g_fails <= 40) std::printf("FAIL %s\n", what.c_str());
}

/* libref's had_rotate row, through the plugin's own exports, or null. */
RadLaunchFn load_libref() {
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home || !*home) return nullptr;
    const std::string p = std::string(home) + "/kernels/libref.so";
    void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) return nullptr;
    auto open = (int (*)(void))dlsym(h, "rad_plugin_open"); /* as the loader does */
    if (open && open() < 0) return nullptr;
    auto cnt = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return nullptr;
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* k = at(i);
        if (k && k->op && !std::strcmp(k->op, "had_rotate")) return k->launch;
    }
    return nullptr;
}

/* A [rows, width] view at `pitch` elements a row over `buf`. */
RadTensor view(void* buf, uint32_t dt, int64_t rows, int64_t width, int64_t pitch) {
    RadTensor t{};
    t.data = buf;
    t.dtype = dt;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = width;
    t.stride[0] = pitch;
    t.stride[1] = 1;
    return t;
}

int launch(RadLaunchFn fn, RadTensor x, int64_t n, int64_t g) {
    RadParam p[4]{};
    p[0] = { "M", RAD_P_INT, x.shape[0] * x.shape[1] / n };
    p[1] = { "n", RAD_P_INT, n };
    p[2] = { "group", RAD_P_INT, g };
    p[3] = { "dtype", RAD_P_STR, 0, 0, "bf16", 0.0 };
    RadArgs a{};
    a.t = &x;
    a.n_t = 1;
    a.p = p;
    a.n_p = 4;
    return fn(&a, nullptr);
}

int emu(const RadArgs* a, RadStream s) { return r4d_rad_had_rotate(a, s); }

struct Case {
    const char* name;
    int64_t rows, width, pitch, n, g;
    float lo, hi;          /* magnitudes drawn log-uniform in [lo, hi] */
    bool outlier;          /* one channel a group 100x the rest */
};

/* One case: the same bytes through both, every element libref's and every gap byte untouched. */
void run_case(RadLaunchFn ref, const Case& c, std::mt19937& rng) {
    const int64_t total = c.rows * c.pitch;
    const uint16_t kGap = 0x7fa5;   /* a NaN neither side may write over */
    std::vector<uint16_t> in((size_t)total, kGap);
    std::uniform_real_distribution<float> e(std::log(c.lo), std::log(c.hi));
    std::bernoulli_distribution sign(0.5);
    for (int64_t r = 0; r < c.rows; ++r)
        for (int64_t i = 0; i < c.width; ++i) {
            float v = std::exp(e(rng)) * (sign(rng) ? -1.0f : 1.0f);
            if (c.outlier && i % c.g == 17) v *= 100.0f;
            in[(size_t)(r * c.pitch + i)] = rad_f32_to_bf16(v);
        }
    std::vector<uint16_t> want = in, got = in;
    const int rr = launch(ref, view(want.data(), RAD_BF16, c.rows, c.width, c.pitch), c.n, c.g);
    const int re = launch(emu, view(got.data(), RAD_BF16, c.rows, c.width, c.pitch), c.n, c.g);
    if (rr != RAD_OK || re != RAD_OK) {
        fail(std::string(c.name) + ": libref " + std::to_string(rr) + ", emulated " +
             std::to_string(re));
        return;
    }
    int64_t diff = 0, moved = 0, first = -1;
    for (int64_t i = 0; i < total; ++i) {
        if (got[(size_t)i] != want[(size_t)i]) { ++diff; if (first < 0) first = i; }
        if (got[(size_t)i] != in[(size_t)i]) ++moved;
        const bool gap = i % c.pitch >= c.width;
        if (gap && (got[(size_t)i] != kGap || want[(size_t)i] != kGap))
            fail(std::string(c.name) + ": a gap byte between rows was written");
    }
    if (diff)
        fail(std::string(c.name) + ": " + std::to_string(diff) + " elements differ from libref, "
             "the first at " + std::to_string(first));
    /* A rotation that wrote nothing would agree with a libref that wrote nothing. */
    if (moved < c.rows * c.width / 2)
        fail(std::string(c.name) + ": the rotation left most of the row as it was");
    std::printf("  %-44s %lld x %lld (pitch %lld), n %lld, group %lld: %s\n", c.name,
                (long long)c.rows, (long long)c.width, (long long)c.pitch, (long long)c.n,
                (long long)c.g, diff ? "DIFFERS" : "libref's bytes");
}

}  // namespace

int main() {
    const RadLaunchFn ref = load_libref();
    if (!ref) {
        std::printf("FAIL libref's had_rotate not loadable (set RADIANCE_HOME)\n");
        return 1;
    }
    std::mt19937 rng(20261103);
    const Case cases[] = {
        { "q, six heads of 256 a token",               3, 6 * 256, 6 * 256, 256, 256, 1e-2f, 4.0f, false },
        { "k, a step's rows of a wider buffer",        5, 4 * 256, 7 * 256, 256, 256, 1e-2f, 4.0f, false },
        { "rows of 128",                               7, 4 * 128, 4 * 128, 128, 128, 1e-2f, 4.0f, false },
        { "a row of two groups, each rotated alone",   2, 2 * 256, 2 * 256, 512, 256, 1e-2f, 4.0f, false },
        { "one block, seven idle waves",               1, 256, 256, 256, 256, 1e-2f, 4.0f, false },
        { "300 blocks from 1e-3 to 1e3, an outlier",  50, 6 * 256, 6 * 256, 256, 256, 1e-3f, 1e3f, true },
        { "128-wide, an outlier, pitched",            33, 2 * 128, 3 * 128, 128, 128, 1e-3f, 1e3f, true },
    };
    std::printf("had_rotate: the emulated kernel against libref\n");
    for (const Case& c : cases) run_case(ref, c, rng);

    /* What the rows do not serve is refused by the kernel, not rotated at another width. */
    std::vector<uint16_t> b16(256, 0x3f80);
    if (launch(emu, view(b16.data(), RAD_BF16, 4, 64, 64), 64, 64) == RAD_OK)
        fail("a 64-wide group was rotated, which no row serves");
    std::vector<float> f32(256, 1.0f);
    if (launch(emu, view(f32.data(), RAD_F32, 1, 256, 256), 256, 256) == RAD_OK)
        fail("an f32 row was rotated by the bf16 kernel");

    std::printf(g_fails ? "had_rotate: %d failure(s)\n" : "had_rotate: all cases pass\n", g_fails);
    return g_fails ? 1 : 0;
}
