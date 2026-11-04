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
 * starts: the case that found the pager rewriting its entry before the later waves had read it.
 *
 * Importance-tiered demotion (RAD_KVQ_DEMOTE2): qsa_page with store_layout2 over a store whose
 * slots hold k4h_v4 or k4h_v2h by the slot table, and qsa_count over selections past the budget
 * (full rows with and without the partial page, short rows, cold and empty entries, many rows
 * naming the same blocks), each held to libref's bytes and counts. */
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
extern "C" int r4d_rad_qsa_count(const RadArgs* a, RadStream stream);

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

/* The pager over two demote layouts: slots 0, 2 and 4 hold k4h_v4 rows, 1 and 3 k4h_v2h ones at the
 * front of a k4h_v4-wide fragment (the rest of it filler the pager must not read), as libref's
 * requantise writes them; the slot table picks each cold entry's. */
void check_page2(RadLaunchFn ref, RadLaunchFn ref_page, std::mt19937& rng, const char* res, int64_t rows) {
    const int64_t hd = 256, slots = 5, blocks = 12;
    const int64_t rw = row_of("k4h_v4", hd), rt = row_of(res, hd);
    const int64_t sfrag = rows * rw, pfrag = rows * rt;
    const uint8_t lay[slots] = { 0, 1, 0, 1, 0 };
    std::vector<uint8_t> store((size_t)(slots * sfrag), 0x3C);
    for (int64_t s = 0; s < slots; ++s) {
        const char* dem = lay[s] ? "k4h_v2h" : "k4h_v4";
        /* Requantised at the resident row's pitch, then packed at the demote layout's. */
        std::vector<uint8_t> resident((size_t)(rows * rt)), d((size_t)(rows * rt));
        cache_rows(rng, res, hd, rows, rt, resident.data());
        if (transcode(ref, resident.data(), d.data(), rows, rt, hd, res, dem) != RAD_OK) {
            fail("qsa_page fixture: libref would not requantise");
            return;
        }
        for (int64_t r = 0; r < rows; ++r)
            std::memcpy(store.data() + s * sfrag + r * row_of(dem, hd), d.data() + r * rt,
                        (size_t)row_of(dem, hd));
    }
    const int32_t sel0[] = { rad_kv_cold_entry(3), 4, -1, rad_kv_cold_entry(0),
                             rad_kv_cold_entry(1), 6, rad_kv_cold_entry(4), -1 };
    const int32_t scr[] = { 7, 0, 0, 8, 9, 0, 10, 0 };
    std::vector<int32_t> sw(sel0, sel0 + 8), se(sel0, sel0 + 8), sc(scr, scr + 8);
    std::vector<uint8_t> pw((size_t)(blocks * pfrag), 0xEE), pe = pw;
    std::vector<uint8_t> table(lay, lay + slots);
    int64_t cw = 0, ce = 0;
    auto go = [&](RadLaunchFn fn, std::vector<int32_t>& sel, std::vector<uint8_t>& pool, int64_t* ctr) {
        RadTensor t[6] = {
            plane(sel.data(), RAD_I32, 2, 4, 4),
            plane(store.data(), RAD_U8, slots, sfrag, sfrag),
            plane(pool.data(), RAD_U8, blocks, pfrag, pfrag),
            plane(sc.data(), RAD_I32, 2, 4, 4),
            {}, {} };
        t[4].data = ctr; t[4].dtype = RAD_I64; t[4].rank = 1; t[4].shape[0] = 1; t[4].stride[0] = 1;
        t[5].data = table.data(); t[5].dtype = RAD_U8; t[5].rank = 1; t[5].shape[0] = slots; t[5].stride[0] = 1;
        RadParam p[9]{};
        p[0] = { "M", RAD_P_INT, 2, 0, nullptr, 0.0 };
        p[1] = { "topk", RAD_P_INT, 3, 0, nullptr, 0.0 };
        p[2] = { "blocks", RAD_P_INT, blocks, 0, nullptr, 0.0 };
        p[3] = { "seqs", RAD_P_INT, 2, 0, nullptr, 0.0 };
        p[4] = { "block_bytes", RAD_P_INT, pfrag, 0, nullptr, 0.0 };
        p[5] = { "store_layout", RAD_P_STR, 0, 0, "k4h_v4", 0.0 };
        p[6] = { "pool_layout", RAD_P_STR, 0, 0, res, 0.0 };
        p[7] = { "head_dim", RAD_P_INT, hd, 0, nullptr, 0.0 };
        p[8] = { "store_layout2", RAD_P_STR, 0, 0, "k4h_v2h", 0.0 };
        RadArgs a{};
        a.t = t; a.n_t = 6; a.p = p; a.n_p = 9;
        return fn(&a, nullptr);
    };
    const int rr = go(ref_page, sw, pw, &cw), re = go(r4d_rad_qsa_page, se, pe, &ce);
    const std::string name = std::string("qsa_page k4h_v4 and k4h_v2h slots -> ") + res + ", " +
                             std::to_string(rows) + " rows";
    if (rr != RAD_OK || re != RAD_OK) {
        fail(name + ": libref " + std::to_string(rr) + ", emulated " + std::to_string(re));
        return;
    }
    int64_t diff = 0, written = 0;
    for (size_t i = 0; i < pw.size(); ++i) { diff += pw[i] != pe[i]; written += pe[i] != 0xEE; }
    if (diff) fail(name + ": " + std::to_string(diff) + " pool bytes differ from libref");
    if (sw != se) fail(name + ": the selection differs from libref's");
    if (cw != ce || ce != 4) fail(name + ": the counter is " + std::to_string(ce));
    if (written < 3 * pfrag) fail(name + ": the pages were barely written");
    std::printf("  %-58s %s\n", name.c_str(), diff || sw != se ? "DIFFERS" : "libref's bytes");
}

/* qsa_count against libref's counts at Flash-Next's topk (512) over 64 rows, every row counted and
 * with `cu` (decode sequences of one to four rows and an empty one: only each one's first row). */
void check_count(std::mt19937& rng, bool with_cu) {
    const RadLaunchFn ref = load_libref("qsa_count");
    if (!ref) { fail("libref's qsa_count is not loadable"); return; }
    const int64_t M = 64, topk = 512, w = topk + 1, ratio = 4, blocks = 1500;
    std::vector<int32_t> sel((size_t)(M * w)), ns((size_t)M), po((size_t)M);
    std::vector<uint32_t> base((size_t)blocks);
    for (uint32_t& c : base) c = (uint32_t)(rng() % 500);
    for (int64_t m = 0; m < M; ++m) {
        po[(size_t)m] = (int32_t)(4000 + 5 * m);
        const bool part = (po[(size_t)m] + 1) % ratio != 0;
        const int64_t k = m % 4 == 1 ? topk - 3 : topk;
        for (int64_t j = 0; j < w; ++j) {
            int32_t e = (int32_t)(rng() % 700);
            if (rng() % 50 == 0) e = rad_kv_cold_entry((int32_t)(rng() % 9));
            if (rng() % 70 == 0) e = -1;
            if (j >= k + (part ? 1 : 0)) e = -1;
            sel[(size_t)(m * w + j)] = e;
        }
        ns[(size_t)m] = (int32_t)(k + (part ? 1 : 0));
    }
    std::vector<int32_t> cu = { 0 };
    while (cu.back() < M) {
        cu.push_back(std::min<int32_t>((int32_t)M, cu.back() + 1 + (int32_t)(rng() % 4)));
        if (cu.size() == 5) cu.push_back(cu.back());                  /* an empty sequence */
    }
    std::vector<uint32_t> want = base, got = base;
    auto go = [&](RadLaunchFn fn, std::vector<uint32_t>& cnt) {
        RadTensor t[5] = { plane(sel.data(), RAD_I32, M, w, w), {}, {}, {}, {} };
        const std::pair<void*, int64_t> v[4] = { { ns.data(), M }, { po.data(), M }, { cnt.data(), blocks },
                                                 { cu.data(), (int64_t)cu.size() } };
        for (int i = 0; i < (with_cu ? 4 : 3); ++i) {
            t[1 + i].data = v[i].first; t[1 + i].dtype = i == 2 ? RAD_U32 : RAD_I32;
            t[1 + i].rank = 1; t[1 + i].shape[0] = v[i].second; t[1 + i].stride[0] = 1;
        }
        RadParam p[3]{};
        p[0] = { "M", RAD_P_INT, M, 0, nullptr, 0.0 };
        p[1] = { "topk", RAD_P_INT, topk, 0, nullptr, 0.0 };
        p[2] = { "ratio", RAD_P_INT, ratio, 0, nullptr, 0.0 };
        RadArgs a{};
        a.t = t; a.n_t = with_cu ? 5 : 4; a.p = p; a.n_p = 3;
        return fn(&a, nullptr);
    };
    const int rr = go(ref, want), re = go(r4d_rad_qsa_count, got);
    if (rr != RAD_OK || re != RAD_OK) {
        fail("qsa_count: libref " + std::to_string(rr) + ", emulated " + std::to_string(re));
        return;
    }
    int64_t diff = 0, added = 0;
    for (int64_t b = 0; b < blocks; ++b) { diff += got[(size_t)b] != want[(size_t)b]; added += want[(size_t)b] - base[(size_t)b]; }
    if (diff) fail("qsa_count: " + std::to_string(diff) + " counters differ from libref");
    if (added < (with_cu ? 8 : 30) * topk) fail("qsa_count: libref counted too little for the case to mean anything");
    std::printf("  %-58s %s (%lld counted)\n", with_cu ? "qsa_count, 64 rows at topk 512, first rows of cu"
                                                    : "qsa_count, 64 rows at topk 512",
                diff ? "DIFFERS" : "libref's counts", (long long)added);
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
    for (const char* res : kResident)
        for (int64_t rows : { 4, 8 }) check_page2(ref, ref_page, rng, res, rows);
    check_count(rng, false);
    check_count(rng, true);
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
