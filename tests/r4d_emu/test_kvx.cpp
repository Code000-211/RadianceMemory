/* tests/r4d_emu: the KV row codec of window residency's demote layout (RAD_KVQ_DEMOTE) on the host
 * SIMT emulator -- libr4d's kv_transcode and the expanding qsa_page (r4d_qsa_page_u8.hip, the codec
 * in r4d_kvx.h) -- held to libref's kv_transcode and qsa_page byte for byte. libref is the oracle
 * (docs/OPS.md), so the test needs it: ctest sets $RADIANCE_HOME, and without one the test fails
 * rather than checking nothing.
 *
 * kv_transcode runs every direction the first version serves (i8h and fp8 resident, k4h_v4 and
 * k4h_v2h demoted, both ways) at head_dim 128, 256 (Flash-Next) and 512, over two kinds of rows:
 *
 *   - rows a cache holds: i8h codes under ordinary f16 scales, E4M3 codes that are not NaN, and the
 *     demote layouts as libref's requantise writes them from those;
 *   - every byte drawn: E4M3 NaN codes, f16 scales that are infinite or NaN, every int4 and int2
 *     code. The codec's answers there are defined (the stores' clamps, fmax dropping a NaN), and the
 *     kernel must give libref's bytes on them too.
 *
 * Rows sit at a pitch wider than their width, and the gap bytes must stay as they were.
 *
 * qsa_page with store_layout and pool_layout over Flash-Next's fragments (4-token pages, one KV
 * head a rank at --tp 2, two at one rank): cold entries from a store of demote-layout slots into an
 * i8h or fp8 pool, beside resident and empty entries, the selection, every pool byte and the
 * counter compared with libref's. The four waves of the pager's workgroup split a fragment's rows,
 * so a fragment of four rows leaves no wave idle and one of eight gives each two. The fp8 pool's
 * encode has no cross-lane step, so with it the emulator runs each wave to its end before the next
 * starts: the case that found the pager rewriting its entry before the later waves had read it. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "rad_runtime.h"
#include "r4d_plugin.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <string>
#include <vector>

extern "C" int r4d_rad_kv_transcode(const RadArgs* a, RadStream stream);
extern "C" int r4d_rad_qsa_page(const RadArgs* a, RadStream stream);

namespace {

int g_fails = 0;
void fail(const std::string& what) {
    ++g_fails;
    if (g_fails <= 40) std::printf("FAIL %s\n", what.c_str());
}

/* libref's row for `op`, through the plugin's own exports, or null. */
RadLaunchFn load_libref(const char* op) {
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home || !*home) return nullptr;
    const std::string p = std::string(home) + "/kernels/libref.so";
    void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) return nullptr;
    auto open = (int (*)(void))dlsym(h, "rad_plugin_open");
    if (open && open() < 0) return nullptr;
    auto cnt = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return nullptr;
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* k = at(i);
        if (k && k->op && !std::strcmp(k->op, op)) return k->launch;
    }
    return nullptr;
}

int64_t row_of(const char* lay, int64_t hd) {
    return !std::strcmp(lay, "fp8") ? 2 * hd : !std::strcmp(lay, "i8h") ? 2 * hd + 4
         : !std::strcmp(lay, "k4h_v4") ? 18 * hd / 16 : 14 * hd / 16;
}

RadTensor plane(void* buf, uint32_t dt, int64_t rows, int64_t width, int64_t pitch) {
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

int transcode(RadLaunchFn fn, uint8_t* x, uint8_t* o, int64_t rows, int64_t pitch, int64_t hd,
              const char* from, const char* to) {
    RadTensor t[2] = { plane(x, RAD_U8, rows, row_of(from, hd), pitch),
                       plane(o, RAD_U8, rows, row_of(to, hd), pitch) };
    RadParam p[4]{};
    p[0] = { "M", RAD_P_INT, rows, 0, nullptr, 0.0 };
    p[1] = { "head_dim", RAD_P_INT, hd, 0, nullptr, 0.0 };
    p[2] = { "from", RAD_P_STR, 0, 0, from, 0.0 };
    p[3] = { "to", RAD_P_STR, 0, 0, to, 0.0 };
    RadArgs a{};
    a.t = t; a.n_t = 2; a.p = p; a.n_p = 4;
    return fn(&a, nullptr);
}

uint16_t f16_of(float f) { return rad_f32_to_f16(f); }

/* `rows` rows of a resident layout as a cache holds them, at `pitch`. */
void cache_rows(std::mt19937& rng, const char* lay, int64_t hd, int64_t rows, int64_t pitch,
                uint8_t* out) {
    std::uniform_int_distribution<int> code(-127, 127), byte(0, 255);
    std::uniform_real_distribution<float> lg(-9.0f, 3.0f);
    for (int64_t r = 0; r < rows; ++r) {
        uint8_t* p = out + r * pitch;
        if (!std::strcmp(lay, "i8h")) {
            for (int64_t d = 0; d < 2 * hd; ++d) p[d] = (uint8_t)(int8_t)code(rng);
            for (int w = 0; w < 2; ++w) {
                const uint16_t s = f16_of(std::exp2(lg(rng)));
                std::memcpy(p + 2 * hd + 2 * w, &s, 2);
            }
        } else {
            for (int64_t d = 0; d < 2 * hd; ++d) {
                int b;
                do b = byte(rng); while ((b & 0x7f) == 0x7f);   /* not E4M3's NaN */
                p[d] = (uint8_t)b;
            }
        }
    }
}

const char* const kResident[] = { "i8h", "fp8" };
const char* const kDemote[]   = { "k4h_v4", "k4h_v2h" };

/* One direction over one plane of rows: emulated against libref, the gaps untouched. */
void check_transcode(RadLaunchFn ref, const char* name, const std::vector<uint8_t>& in,
                     int64_t rows, int64_t pitch, int64_t hd, const char* from, const char* to) {
    std::vector<uint8_t> a = in, want((size_t)(rows * pitch), 0xC3), got = want;
    const int rr = transcode(ref, a.data(), want.data(), rows, pitch, hd, from, to);
    const int re = transcode(r4d_rad_kv_transcode, a.data(), got.data(), rows, pitch, hd, from, to);
    if (rr != RAD_OK || re != RAD_OK) {
        fail(std::string(name) + ": libref " + std::to_string(rr) + ", emulated " + std::to_string(re));
        return;
    }
    int64_t diff = 0, first = -1;
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[i]) { ++diff; if (first < 0) first = (int64_t)i; }
    const int64_t w = row_of(to, hd);
    for (size_t i = 0; i < got.size(); ++i)
        if ((int64_t)i % pitch >= w && got[i] != 0xC3) { fail(std::string(name) + ": a gap was written"); break; }
    if (diff)
        fail(std::string(name) + ": " + std::to_string(diff) + " bytes differ from libref, the "
             "first at " + std::to_string(first));
    std::printf("  %-40s %s -> %s at head_dim %lld, %lld rows: %s\n", name, from, to,
                (long long)hd, (long long)rows, diff ? "DIFFERS" : "libref's bytes");
}

/* The expanding pager at one geometry: `rows` (token, head) rows a fragment, two layers in the
 * store, the selection of two decode rows. */
void check_page(RadLaunchFn ref, RadLaunchFn ref_page, std::mt19937& rng, const char* res, const char* dem, int64_t rows) {
    const int64_t hd = 256, layers = 2, slots = 5, blocks = 12;
    const int64_t rf = row_of(dem, hd), rt = row_of(res, hd);
    const int64_t sfrag = rows * rf, pfrag = rows * rt;
    /* The store: resident rows requantised by libref, layer after layer in each slot. */
    std::vector<uint8_t> resident((size_t)(slots * layers * rows * rt));
    cache_rows(rng, res, hd, slots * layers * rows, rt, resident.data());
    std::vector<uint8_t> store((size_t)(slots * layers * sfrag));
    {
        RadTensor t[2] = { plane(resident.data(), RAD_U8, slots * layers * rows, rt, rt),
                           plane(store.data(), RAD_U8, slots * layers * rows, rf, rf) };
        RadParam p[4]{};
        p[0] = { "M", RAD_P_INT, slots * layers * rows, 0, nullptr, 0.0 };
        p[1] = { "head_dim", RAD_P_INT, hd, 0, nullptr, 0.0 };
        p[2] = { "from", RAD_P_STR, 0, 0, res, 0.0 };
        p[3] = { "to", RAD_P_STR, 0, 0, dem, 0.0 };
        RadArgs a{};
        a.t = t; a.n_t = 2; a.p = p; a.n_p = 4;
        if (ref(&a, nullptr) != RAD_OK) { fail("qsa_page fixture: libref would not requantise"); return; }
    }
    for (int64_t layer : { 0, 1 }) {
        const int32_t sel0[] = { rad_kv_cold_entry(3), 4, -1, rad_kv_cold_entry(0),
                                 rad_kv_cold_entry(4), 6, rad_kv_cold_entry(1), -1 };
        const int32_t scr[] = { 7, 0, 0, 8, 9, 0, 10, 0 };
        std::vector<int32_t> sw(sel0, sel0 + 8), se(sel0, sel0 + 8), sc(scr, scr + 8);
        std::vector<uint8_t> pw((size_t)(blocks * pfrag), 0xEE), pe = pw;
        int64_t cw = 5, ce = 5;
        auto go = [&](RadLaunchFn fn, std::vector<int32_t>& sel, std::vector<uint8_t>& pool, int64_t* ctr) {
            RadTensor t[5] = {
                plane(sel.data(), RAD_I32, 2, 4, 4),
                plane(store.data() + layer * sfrag, RAD_U8, slots, sfrag, layers * sfrag),
                plane(pool.data(), RAD_U8, blocks, pfrag, pfrag),
                plane(sc.data(), RAD_I32, 2, 4, 4),
                {} };
            t[4].data = ctr; t[4].dtype = RAD_I64; t[4].rank = 1; t[4].shape[0] = 1; t[4].stride[0] = 1;
            RadParam p[8]{};
            p[0] = { "M", RAD_P_INT, 2, 0, nullptr, 0.0 };
            p[1] = { "topk", RAD_P_INT, 3, 0, nullptr, 0.0 };
            p[2] = { "blocks", RAD_P_INT, blocks, 0, nullptr, 0.0 };
            p[3] = { "seqs", RAD_P_INT, 2, 0, nullptr, 0.0 };
            p[4] = { "block_bytes", RAD_P_INT, pfrag, 0, nullptr, 0.0 };
            p[5] = { "store_layout", RAD_P_STR, 0, 0, dem, 0.0 };
            p[6] = { "pool_layout", RAD_P_STR, 0, 0, res, 0.0 };
            p[7] = { "head_dim", RAD_P_INT, hd, 0, nullptr, 0.0 };
            RadArgs a{};
            a.t = t; a.n_t = 5; a.p = p; a.n_p = 8;
            return fn(&a, nullptr);
        };
        const int rr = go(ref_page, sw, pw, &cw), re = go(r4d_rad_qsa_page, se, pe, &ce);
        const std::string name = std::string("qsa_page ") + dem + " -> " + res + ", " +
                                 std::to_string(rows) + " rows, layer " + std::to_string(layer);
        if (rr != RAD_OK || re != RAD_OK) {
            fail(name + ": libref " + std::to_string(rr) + ", emulated " + std::to_string(re));
            continue;
        }
        int64_t diff = 0;
        for (size_t i = 0; i < pw.size(); ++i) diff += pw[i] != pe[i];
        const bool same_sel = sw == se, same_ctr = cw == ce && ce == 9;
        /* An expansion that wrote nothing would agree with a libref that wrote nothing. */
        int64_t written = 0;
        for (size_t i = 0; i < pe.size(); ++i) written += pe[i] != 0xEE;
        if (diff) fail(name + ": " + std::to_string(diff) + " pool bytes differ from libref");
        if (!same_sel) fail(name + ": the selection differs from libref's");
        if (!same_ctr) fail(name + ": the counter is " + std::to_string(ce) + ", libref's " + std::to_string(cw));
        if (written < 3 * pfrag / 2) fail(name + ": the pages were barely written");
        std::printf("  %-58s %s\n", name.c_str(), diff || !same_sel || !same_ctr ? "DIFFERS" : "libref's bytes");
    }
}

}  // namespace

int main() {
    const RadLaunchFn ref = load_libref("kv_transcode");
    const RadLaunchFn ref_page = load_libref("qsa_page");
    if (!ref || !ref_page) {
        std::printf("FAIL libref's kv_transcode or qsa_page is not loadable from $RADIANCE_HOME/kernels "
                    "(%s); the oracle is required\n", std::getenv("RADIANCE_HOME") ? "set" : "unset");
        return 1;
    }
    std::mt19937 rng(20261);
    for (int64_t hd : { 128, 256, 512 }) {
        const int64_t rows = hd == 256 ? 48 : 12, pitch = 2 * hd + 4 + 12;
        std::uniform_int_distribution<int> byte(0, 255);
        for (const char* res : kResident) {
            std::vector<uint8_t> a((size_t)(rows * pitch), 0x11);
            cache_rows(rng, res, hd, rows, pitch, a.data());
            for (const char* dem : kDemote) {
                check_transcode(ref, "a cache's rows, requantised", a, rows, pitch, hd, res, dem);
                /* The demoted rows libref writes, expanded back. */
                std::vector<uint8_t> d((size_t)(rows * pitch), 0x22);
                if (transcode(ref, a.data(), d.data(), rows, pitch, hd, res, dem) != RAD_OK) {
                    fail("libref would not requantise");
                    continue;
                }
                check_transcode(ref, "the demoted rows, expanded", d, rows, pitch, hd, dem, res);
                std::vector<uint8_t> r((size_t)(rows * pitch));
                for (uint8_t& b : r) b = (uint8_t)byte(rng);
                check_transcode(ref, "every byte drawn, requantised", r, rows, pitch, hd, res, dem);
                for (uint8_t& b : r) b = (uint8_t)byte(rng);
                check_transcode(ref, "every byte drawn, expanded", r, rows, pitch, hd, dem, res);
            }
        }
    }
    for (const char* res : kResident)
        for (const char* dem : kDemote)
            for (int64_t rows : { 4, 8 }) check_page(ref, ref_page, rng, res, dem, rows);
    /* Refused, as libref refuses: the same layout twice, a head_dim off the lane map. */
    {
        std::vector<uint8_t> x(4096), o(4096);
        if (transcode(r4d_rad_kv_transcode, x.data(), o.data(), 2, 600, 256, "i8h", "i8h") == RAD_OK)
            fail("the same layout twice was not refused");
        if (transcode(r4d_rad_kv_transcode, x.data(), o.data(), 2, 600, 96, "i8h", "k4h_v4") == RAD_OK)
            fail("head_dim 96 was not refused");
    }
    if (g_fails) {
        std::printf("%d failure(s)\n", g_fails);
        return 1;
    }
    std::printf("all kv_transcode and expanding qsa_page cases hold libref's bytes\n");
    return 0;
}
