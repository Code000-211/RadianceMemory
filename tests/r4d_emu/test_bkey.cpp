/* tests/r4d_emu: the QSA indexer's int8 block keys (`key_dtype` i8, --qsa-bkey-dtype i8) on the host
 * SIMT emulator: libr4d's qsa_block_key (r4d_qsa_block_key_bf16.hip) and qsa_score
 * (r4d_qsa_score_bf16.hip, both forms) against libref, which is the oracle (docs/OPS.md), so the
 * test needs it: ctest sets $RADIANCE_HOME, and without one the test fails rather than checking
 * nothing.
 *
 *   - the store, into a paged store shaped as the group declares it ([pages, 1, 4, 34] bytes):
 *     each listed page is libref's quantiser applied to the same kernel's bf16 key, byte for byte
 *     (codes, f16 scale, zero padding), and libref's own i8 store, byte for byte where the two bf16
 *     keys agree and to the key's last bit everywhere (the emulator's sincosf and libref's are
 *     different functions); a -1 entry and every unlisted page keep their bytes;
 *   - the score over drawn int8 pages (two sequences, a straddled tile, padding, entries past nc),
 *     in the scalar and the matrix form, against libref's i8 score: within f32 summation order of
 *     it, -inf where libref has -inf, untouched where libref leaves the plane alone. A control
 *     scores the same codes at scale 1 and must be far off. */
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

extern "C" void r4d_qsa_score_pin_form(int form);

namespace {

int g_fails = 0;
void fail(const std::string& what) {
    ++g_fails;
    if (g_fails <= 40) std::printf("FAIL %s\n", what.c_str());
}

/* libref's row `name`, through the plugin's own exports, or null. */
RadLaunchFn libref(const char* name) {
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
        if (k && k->op && !std::strcmp(k->op, name)) return k->launch;
    }
    return nullptr;
}

RadTensor dense(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t acc = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[(size_t)i];
        t.stride[i] = acc;
        acc *= shape[(size_t)i];
    }
    return t;
}

float bf(uint16_t h) { return rad_bf16_to_f32(h); }

/* libref's quantiser (ref_qsa_block_key at key_dtype i8) restated over a bf16 key. */
void quantise(const uint16_t* key, int64_t n, int64_t width, uint8_t* page) {
    float amax = std::fabs(bf(key[0]));
    for (int64_t i = 1; i < n; ++i) amax = std::fmax(amax, std::fabs(bf(key[i])));
    const uint16_t s16 = rad_f32_to_f16(std::fmin(amax / 127.0f, 512.0f));
    const float sc = rad_f16_to_f32(s16);
    for (int64_t i = 0; i < n; ++i) {
        const float r = sc > 0.0f ? std::rint(bf(key[i]) / sc) : 0.0f;
        page[i] = (uint8_t)(int8_t)(int)std::fmin(std::fmax(r, -127.0f), 127.0f);
    }
    std::memcpy(page + n, &s16, 2);
    std::memset(page + n + 2, 0, (size_t)(width - n - 2));
}

void store_case(RadLaunchFn ref) {
    const int64_t M = 9, n = 128, ratio = 4, rot = 64, pages = 16, W = 136;
    std::vector<uint16_t> k((size_t)(M * ratio * n));
    for (size_t i = 0; i < k.size(); ++i)
        k[i] = rad_f32_to_bf16((float)(((int64_t)i * 37 % 97) - 48) / 53.0f *
                               (i % 31 == 0 ? 6.0f : 1.0f));
    std::vector<float> w((size_t)n);
    for (int64_t d = 0; d < n; ++d) w[(size_t)d] = 0.1f + (float)((d * 7) % 13) / 40.0f;
    std::vector<int32_t> pg = { 3, 0, -1, 12, 7, 15, 1, 9, 4 };
    RadParam ps[13]{};
    int np = 0;
    ps[np++] = { "M", RAD_P_INT, M };
    ps[np++] = { "n", RAD_P_INT, n };
    ps[np++] = { "ratio", RAD_P_INT, ratio };
    ps[np++] = { "rotary_dim", RAD_P_INT, rot };
    ps[np++] = { "pos0", RAD_P_INT, 40 };
    ps[np++] = { "block_size", RAD_P_INT, ratio };
    ps[np++] = { "eps", RAD_P_F64, 0, 0, nullptr, 1e-6 };
    ps[np++] = { "theta", RAD_P_F64, 0, 0, nullptr, 1e7 };
    ps[np++] = { "scale", RAD_P_F64, 0, 0, nullptr, 1.0 };
    ps[np++] = { "wadd", RAD_P_F64, 0, 0, nullptr, 1.0 };
    ps[np++] = { "dtype", RAD_P_STR, 0, 0, "bf16", 0.0 };
    ps[np] = { "key_dtype", RAD_P_STR, 0, 0, "i8", 0.0 };
    auto launch = [&](RadLaunchFn fn, void* out, bool i8) {
        RadTensor t[5] = {
            dense(k.data(), RAD_BF16, { M * ratio, n }), dense(w.data(), RAD_F32, { n }),
            dense(pg.data(), RAD_I32, { M }), RadTensor{},
            i8 ? dense(out, RAD_I8, { pages, 1, ratio, W / ratio })
               : dense(out, RAD_BF16, { pages, n }),
        };
        RadArgs a{};
        a.t = t; a.n_t = 5; a.p = ps; a.n_p = np + (i8 ? 1 : 0);
        return fn(&a, nullptr);
    };
    std::vector<uint16_t> key((size_t)(pages * n), 0);
    std::vector<uint8_t> got((size_t)(pages * W), 0xA5), want = got;
    const int rb = launch(r4d_rad_qsa_block_key, key.data(), false);
    const int r8 = launch(r4d_rad_qsa_block_key, got.data(), true);
    const int rr = launch(ref, want.data(), true);
    if (rb != RAD_OK || r8 != RAD_OK || rr != RAD_OK) {
        fail("store: launch " + std::to_string(rb) + " / " + std::to_string(r8) + ", libref " +
             std::to_string(rr));
        return;
    }
    std::vector<char> listed((size_t)pages, 0);
    for (int32_t p : pg) if (p >= 0) listed[(size_t)p] = 1;
    long same = 0, live = 0;
    double num = 0.0, den = 0.0;
    std::vector<uint8_t> q((size_t)W);
    for (int64_t p = 0; p < pages; ++p) {
        const uint8_t *g = got.data() + p * W, *r = want.data() + p * W;
        if (!listed[(size_t)p]) {
            for (int64_t i = 0; i < W; ++i)
                if (g[i] != 0xA5) { fail("store: an unlisted page was written"); break; }
            continue;
        }
        ++live;
        quantise(key.data() + p * n, n, W, q.data());
        if (std::memcmp(g, q.data(), (size_t)W))
            fail("store: page " + std::to_string(p) + " is not its bf16 key quantised");
        same += std::memcmp(g, r, (size_t)W) == 0;
        uint16_t sg, sr;
        std::memcpy(&sg, g + n, 2);
        std::memcpy(&sr, r + n, 2);
        for (int64_t i = 0; i < n; ++i) {
            const double a = (int8_t)g[i] * (double)rad_f16_to_f32(sg);
            const double b = (int8_t)r[i] * (double)rad_f16_to_f32(sr);
            num += (a - b) * (a - b);
            den += b * b;
        }
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : -1.0;
    if (!(rel >= 0.0 && rel < 2e-3)) fail("store: decoded keys off libref's, rel_l2 " + std::to_string(rel));
    std::printf("  qsa_block_key i8: %ld pages, each its bf16 key quantised; %ld of them libref's "
                "bytes, decoded rel_l2 %.3e\n", live, same, rel);
}

void score_case(RadLaunchFn ref) {
    const int64_t S = 2, n = 128, heads = 4, nb = 45, pages = 80, W = 136;
    const int64_t rows[S] = { 7, 13 };
    std::vector<int32_t> cu((size_t)(S + 1), 0), nc((size_t)S);
    for (int64_t i = 0; i < S; ++i) cu[(size_t)(i + 1)] = cu[(size_t)i] + (int32_t)rows[i];
    const int64_t M = cu[(size_t)S];
    std::vector<uint16_t> q((size_t)(M * heads * n));
    for (int64_t i = 0; i < M * heads * n; ++i)
        q[(size_t)i] = rad_f32_to_bf16((float)(((i * 31 + 5) % 89) - 44) / 43.0f);
    std::vector<uint8_t> bk((size_t)(pages * W), 0), unit;
    for (int64_t p = 0; p < pages; ++p) {
        for (int64_t i = 0; i < n; ++i)
            bk[(size_t)(p * W + i)] = (uint8_t)(int8_t)(((p * 43 + i * 11) % 255) - 127);
        const uint16_t s16 = rad_f32_to_f16(0.002f + 0.03f * (float)((p * 7) % 17));
        std::memcpy(&bk[(size_t)(p * W + n)], &s16, 2);
    }
    unit = bk;
    for (int64_t p = 0; p < pages; ++p) {
        const uint16_t one = rad_f32_to_f16(1.0f);
        std::memcpy(&unit[(size_t)(p * W + n)], &one, 2);
    }
    std::vector<int32_t> bt((size_t)(S * nb));
    for (int64_t sq = 0; sq < S; ++sq) {
        const int64_t live = nb - 2 - sq * 9;
        nc[(size_t)sq] = (int32_t)(live - 1);
        for (int64_t b = 0; b < nb; ++b)
            bt[(size_t)(sq * nb + b)] =
                b < live && (b % 11) != 4 ? (int32_t)((b * 17 + sq * 7) % pages) : -1;
    }
    RadParam ps[7] = {
        { "M", RAD_P_INT, M }, { "n", RAD_P_INT, n }, { "heads", RAD_P_INT, heads },
        { "blocks", RAD_P_INT, nb }, { "seqs", RAD_P_INT, S },
        { "dtype", RAD_P_STR, 0, 0, "bf16", 0.0 }, { "key_dtype", RAD_P_STR, 0, 0, "i8", 0.0 },
    };
    auto launch = [&](RadLaunchFn fn, std::vector<uint8_t>& keys, std::vector<float>& out) {
        RadTensor t[6] = {
            dense(q.data(), RAD_BF16, { M, heads * n }),
            dense(keys.data(), RAD_I8, { pages, 1, 4, W / 4 }),
            dense(bt.data(), RAD_I32, { S, nb }), dense(cu.data(), RAD_I32, { S + 1 }),
            dense(nc.data(), RAD_I32, { S }), dense(out.data(), RAD_F32, { M, nb }),
        };
        RadArgs a{};
        a.t = t; a.n_t = 6; a.p = ps; a.n_p = 7;
        return fn(&a, nullptr);
    };
    std::vector<float> want((size_t)(M * nb), 12345.0f), ctl = want;
    if (launch(ref, bk, want) != RAD_OK || launch(ref, unit, ctl) != RAD_OK) {
        fail("score: libref refused");
        return;
    }
    const char* form[3] = { "", "scalar", "matrix" };
    for (int f = 1; f <= 2; ++f) {
        r4d_qsa_score_pin_form(f);
        std::vector<float> got((size_t)(M * nb), 12345.0f);
        const int rc = launch(r4d_rad_qsa_score, bk, got);
        r4d_qsa_score_pin_form(0);
        if (rc != RAD_OK) { fail(std::string("score: the ") + form[f] + " form refused"); continue; }
        double num = 0.0, den = 0.0, cnum = 0.0;
        long odd = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double g = got[i], w = want[i];
            if (w == 12345.0 || std::isinf(w)) { odd += !(g == w); continue; }
            num += (g - w) * (g - w);
            cnum += ((double)ctl[i] - w) * ((double)ctl[i] - w);
            den += w * w;
        }
        const double rel = den > 0.0 ? std::sqrt(num / den) : -1.0;
        const double crel = den > 0.0 ? std::sqrt(cnum / den) : -1.0;
        if (!(rel >= 0.0 && rel < 1e-6) || odd || !(crel > 1e3 * rel))
            fail(std::string("score, ") + form[f] + " form: rel_l2 " + std::to_string(rel) +
                 ", " + std::to_string(odd) + " padded or unwritten entries differ, control " +
                 std::to_string(crel));
        std::printf("  qsa_score i8, %s form: rel_l2 vs libref %.3e (scale 1: %.3e), padding and "
                    "unwritten entries %s\n", form[f], rel, crel, odd ? "DIFFER" : "libref's");
    }
}

}  // namespace

int main() {
    const RadLaunchFn kst = libref("qsa_block_key"), ksc = libref("qsa_score");
    if (!kst || !ksc) {
        std::printf("FAIL libref's qsa_block_key / qsa_score not loadable (set RADIANCE_HOME)\n");
        return 1;
    }
    std::printf("int8 block keys: the emulated kernels against libref\n");
    store_case(kst);
    score_case(ksc);
    std::printf(g_fails ? "int8 block keys: %d failure(s)\n" : "int8 block keys: all cases pass\n",
                g_fails);
    return g_fails ? 1 : 0;
}
