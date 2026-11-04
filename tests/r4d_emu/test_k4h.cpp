/* tests/r4d_emu: the k4h_v4, k4h_v2h and i8h caches (RAD_KVQ_K4H, RAD_KVQ_K4HV2H, RAD_KVQ_I8H,
 * abi/rad_builder.h) on the host SIMT emulator, held to libref's kv_store and attn_paged at their
 * kv_dtype. libref is the
 * oracle (docs/OPS.md), so the test needs it: ctest sets $RADIANCE_HOME, and without one the test
 * fails rather than checking nothing. Every case below runs once for each layout.
 *
 * The store (kv_store_k4hv4 / kv_store_k4hv2h, r4d_model_bf16.hip) is held to libref's bytes over the whole cache:
 * every written row, every scale, and every byte it must leave alone (a slot of -1, the other
 * slots of a block). The inputs are what a rotated layer stores -- spread values with an outlier
 * channel, k and v as column slices of a wider row, a non-unit descale -- and the edges of the
 * quantiser: an all-zero group (scale 0), ties of rint (x.5 steps of a power-of-two scale, which
 * round to even), values past the scale cap, an infinity and a NaN. k4h_v2h's V takes the same
 * edges, plus a group whose values sit on its quantiser's boundaries (whole multiples of its root
 * mean square), since its variance normalisation sums, divides and floors where k4h_v4 takes a max.
 * i8h's scale is a whole head's, so its edges are a head each: all zero, ties of rint, past the
 * cap, an infinity, a NaN, and a head of NaNs. The bf16_e4m3r experiment's store is held to
 * libref's bytes the same way.
 *
 * The attention (Flash-Next's QSA layers: head_dim 256, 4-token pages, gqa 12 at one and two ranks
 * and gqa 6 at four; prefill, decode and the speculative verify, which is the decode kernel at a
 * few rows a sequence) reads caches libref stored, beside pages and slots poisoned with NaN scales
 * that no visible key lives in. Two kinds of case:
 *
 *   - decisive: each query row's one target key outscores every other visible key by more than
 *     f32's exponent can hold (exp(-224) is 0), its K and V codes are scaled by powers of two, so
 *     the output is that key's V row exactly in any summation order, split or not (k4h_v2h's V:
 *     odd levels, half of them +-3, whose normalised scale is the power of two). Those are held
 *     to libref byte for byte; a K misread anywhere shows as a different winner or a mixed row, a V
 *     misread as different bytes.
 *   - spread: random rotated-looking K, V and queries, held to libref within the attention
 *     rounding (the kernels widen codes to f16, run P in f16 and merge split-KV partials; libref is
 *     f32 two-pass), RMS relative error under 5e-3 and no element out of bounds, as r4d_emu_attn
 *     holds the windowed kernels to their port of the same reference.
 */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
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
int     r4d_rad_kv_store_k4hv4(const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa12_k4hv4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_prefill_h256_gqa12_k4hv4_scratch(const RadArgs*);
int     r4d_rad_attn_decode_h256_gqa12_k4hv4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa12_k4hv4_scratch(const RadArgs*);
int     r4d_rad_attn_prefill_h256_gqa6_k4hv4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_prefill_h256_gqa6_k4hv4_scratch(const RadArgs*);
int     r4d_rad_attn_decode_h256_gqa6_k4hv4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa6_k4hv4_scratch(const RadArgs*);
int     r4d_rad_kv_store_k4hv2h(const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa12_k4hv2h(const RadArgs*, RadStream);
int64_t r4d_rad_attn_prefill_h256_gqa12_k4hv2h_scratch(const RadArgs*);
int     r4d_rad_attn_decode_h256_gqa12_k4hv2h(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa12_k4hv2h_scratch(const RadArgs*);
int     r4d_rad_attn_prefill_h256_gqa6_k4hv2h(const RadArgs*, RadStream);
int64_t r4d_rad_attn_prefill_h256_gqa6_k4hv2h_scratch(const RadArgs*);
int     r4d_rad_attn_decode_h256_gqa6_k4hv2h(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa6_k4hv2h_scratch(const RadArgs*);
int     r4d_rad_kv_store_i8h(const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa12_i8h(const RadArgs*, RadStream);
int64_t r4d_rad_attn_prefill_h256_gqa12_i8h_scratch(const RadArgs*);
int     r4d_rad_attn_decode_h256_gqa12_i8h(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa12_i8h_scratch(const RadArgs*);
int     r4d_rad_attn_prefill_h256_gqa6_i8h(const RadArgs*, RadStream);
int64_t r4d_rad_attn_prefill_h256_gqa6_i8h_scratch(const RadArgs*);
int     r4d_rad_attn_decode_h256_gqa6_i8h(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa6_i8h_scratch(const RadArgs*);
int     r4d_rad_kv_store_bf16_e4m3r(const RadArgs*, RadStream);
}
extern int emu_grids;

namespace {

constexpr int HD = 256, BS = 4;

/* The layout under test: its kv_dtype, a token's row (288, 224 or 516 bytes), where the scales
 * start (after K's and V's codes), whether V is k4h_v2h's int2, and whether the codes are i8h's
 * bytes under one scale a head. */
struct Layout {
    const char* kv_dtype;
    int row, scales;
    bool v2, i8;
};
const Layout kK4hv4  = { "k4h_v4", 18 * HD / 16, HD, false, false };
const Layout kK4hv2h = { "k4h_v2h", 14 * HD / 16, 3 * HD / 4, true, false };
const Layout kI8h    = { "i8h", 2 * HD + 4, 2 * HD, false, true };
const Layout* L = &kK4hv4;

int g_fails = 0;
void fail(const std::string& what) {
    ++g_fails;
    if (g_fails <= 40) std::printf("FAIL %s\n", what.c_str());
}

using Fn = int (*)(const RadArgs*, RadStream);
using ScratchFn = int64_t (*)(const RadArgs*);

/* libref's row `name`, through the plugin's own exports, or null. */
Fn load_libref(const char* name) {
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
        if (k && k->name && !std::strcmp(k->name, name)) return k->launch;
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
    void f(const char* k, double v) { RadParam x{}; x.key = k; x.kind = RAD_P_F64; x.dval = v; p.push_back(x); }
};

/* A 16-byte aligned byte buffer, as the device allocator gives and the kernels' vector loads need. */
struct Buf {
    std::vector<uint8_t> raw;
    uint8_t* p = nullptr;
    explicit Buf(size_t n, uint8_t fill = 0) : raw(n + 64, fill) {
        p = raw.data() + ((16 - ((uintptr_t)raw.data() & 15)) & 15);
    }
};

/* One kv_store call on `cache` (n_blocks x kvh pages of BS rows), k and v [T, ld] bf16. */
int store(Fn fn, const uint16_t* k, const uint16_t* v, int64_t T, int64_t kvh, int64_t ld,
          const int32_t* slots, uint8_t* cache, int64_t nb, double k_scale, double v_scale) {
    RadTensor t[4];
    t[0] = tens((void*)k, RAD_BF16, { T, kvh * HD });
    t[0].stride[0] = ld;
    t[1] = tens((void*)v, RAD_BF16, { T, kvh * HD });
    t[1].stride[0] = ld;
    t[2] = tens((void*)slots, RAD_I32, { T });
    t[3] = tens(cache, RAD_F8E4M3, { nb, kvh, BS, L->row });
    Params p;
    p.i("M", T); p.i("head_dim", HD); p.i("n_head_kv", kvh); p.i("block_size", BS);
    p.s("kv_dtype", L->kv_dtype); p.f("k_scale", k_scale); p.f("v_scale", v_scale);
    RadArgs a{};
    a.t = t;
    a.n_t = 4;
    a.p = p.p.data();
    a.n_p = (int)p.p.size();
    return fn(&a, nullptr);
}

/* ------------------------------------------------------------------ the store */

void store_case(Fn ref, Fn kernel, const char* name, int64_t T, int64_t kvh, int64_t extra,
                double ks, double vs, bool edges, std::mt19937& rng) {
    const int64_t ld = kvh * HD + extra;
    const int64_t nb = (T + BS - 1) / BS + 3;
    std::vector<uint16_t> k((size_t)(T * ld), 0x7fa5), v((size_t)(T * ld), 0x7fa5);
    std::uniform_real_distribution<float> mag(std::log(1e-3f), std::log(1e3f));
    std::normal_distribution<float> nrm(0.f, 1.f);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t g = 0; g < kvh * HD / 32; ++g) {
            const float m = std::exp(mag(rng));
            for (int i = 0; i < 32; ++i) {
                float kx = nrm(rng) * m, vx = nrm(rng) * m;
                if (i == 17) kx *= 30.0f;   /* an outlier channel in every K group */
                k[(size_t)(t * ld + g * 32 + i)] = rad_f32_to_bf16(kx);
                v[(size_t)(t * ld + g * 32 + i)] = rad_f32_to_bf16(vx);
            }
        }
    if (edges && L->i8 && T >= 6 && kvh == 2) {
        /* A head each (token 1 is the dropped slot): token 0's K head 0 all zero (scale 0); its
         * head 1 at amax 31.75, scale 31.75 / 127 = 0.25 exactly, with x.5 steps that round to
         * even; token 2's K head 0 past the cap (512 * 127), so codes clamp at +-127, its head 1
         * holding +inf; token 3's K head 0 a NaN, its V head 1 all NaN; token 4's V head 0 the ties
         * again. */
        auto head = [&](std::vector<uint16_t>& x, int t, int h) { return x.data() + t * ld + h * HD; };
        const float tie[8] = { 0.125f, 0.375f, 0.625f, -0.125f, -0.375f, 1.75f, -1.75f, 0.875f };
        for (int i = 0; i < HD; ++i) {
            head(k, 0, 0)[i] = 0;
            head(k, 0, 1)[i] = rad_f32_to_bf16(i == 0 ? 31.75f : i < 9 ? tie[i - 1] : 0.0f);
            head(k, 2, 0)[i] = rad_f32_to_bf16((i & 1 ? -1.f : 1.f) * 1e5f * (1 + i % 32));
            head(v, 4, 0)[i] = head(k, 0, 1)[i];
            head(v, 3, 1)[i] = 0xffc0;
        }
        head(k, 2, 1)[7] = 0x7f80;
        head(k, 3, 0)[11] = 0x7fc0;
    } else if (edges && T >= 2 && kvh * HD >= 6 * 32) {
        uint16_t* r0 = k.data();
        for (int i = 0; i < 32; ++i) r0[i] = 0;                          /* group 0: all zero */
        /* group 1: scale 7 * 0.25 / 7 = 0.25 exactly, then x.5 steps: rint ties to even */
        const float tie[8] = { 0.125f, 0.375f, 0.625f, -0.125f, -0.375f, 1.75f, -1.75f, 0.875f };
        for (int i = 0; i < 32; ++i) r0[32 + i] = rad_f32_to_bf16(i < 8 ? tie[i] : 0.0f);
        /* group 2: past the cap (8188 * 7), so codes clamp at +-7 */
        for (int i = 0; i < 32; ++i) r0[64 + i] = rad_f32_to_bf16((i & 1 ? -1.f : 1.f) * 1e5f * (1 + i));
        r0[96 + 3] = 0x7f80;                                             /* group 3: +inf */
        r0[128 + 5] = 0x7fc0;                                            /* group 4: a NaN */
        uint16_t* v1 = v.data() + ld;
        for (int i = 0; i < 32; ++i) v1[160 + i] = 0xffc0;               /* V group 5: all NaN */
        if (L->v2) {
            /* V row 0 takes K's edges, and group 5 sits on the boundaries: sixteen 2s and sixteen
             * -2s have a root mean square of 2, so each value is exactly one level step. */
            for (int i = 0; i < 160; ++i) v[(size_t)i] = k[(size_t)i];
            for (int i = 0; i < 32; ++i) v[(size_t)(160 + i)] = rad_f32_to_bf16(i & 1 ? -2.0f : 2.0f);
        }
    }
    /* Distinct slots over the cache, one token dropped (-1), the rest of each page untouched. */
    std::vector<int32_t> slot((size_t)T);
    std::vector<int32_t> all((size_t)(nb * BS));
    for (size_t i = 0; i < all.size(); ++i) all[i] = (int32_t)i;
    std::shuffle(all.begin(), all.end(), rng);
    for (int64_t t = 0; t < T; ++t) slot[(size_t)t] = all[(size_t)t];
    if (T > 2) slot[1] = -1;

    const size_t bytes = (size_t)(nb * kvh * BS * L->row);
    Buf want(bytes, 0xa5), got(bytes, 0xa5);
    const int rr = store(ref, k.data(), v.data(), T, kvh, ld, slot.data(), want.p, nb, ks, vs);
    const int re = store(kernel, k.data(), v.data(), T, kvh, ld, slot.data(), got.p, nb, ks, vs);
    if (rr != RAD_OK || re != RAD_OK) {
        fail(std::string(name) + ": libref " + std::to_string(rr) + ", emulated " + std::to_string(re));
        return;
    }
    int64_t diff = 0, first = -1, moved = 0;
    for (size_t i = 0; i < bytes; ++i) {
        if (got.p[i] != want.p[i]) { ++diff; if (first < 0) first = (int64_t)i; }
        moved += want.p[i] != 0xa5;
    }
    if (diff)
        fail(std::string(name) + ": " + std::to_string(diff) + " cache bytes differ from libref, "
             "the first at " + std::to_string(first));
    /* A store that wrote nothing would agree with a libref that wrote nothing. */
    const int64_t stored = (T - (T > 2 ? 1 : 0)) * kvh * L->row;
    if (moved < stored / 2)
        fail(std::string(name) + ": the store left most of its rows as they were");
    std::printf("  %-48s %3lld tokens x %lld heads: %s\n", name, (long long)T, (long long)kvh,
                diff ? "DIFFERS" : "libref's bytes");
}

/* The bf16_e4m3r experiment's store (kv_store_bf16_e4m3r): a bf16 cache [nb, kvh, BS, 2 * HD]
 * whose every value went through E4M3 at the descale, held to libref's bytes over the whole cache.
 * The inputs reach past E4M3's range (clamped to 448), into its subnormals, and hold an infinity
 * and a NaN (the clamp makes it -448 on both sides). */
void e4m3r_case(Fn ref, const char* name, int64_t T, int64_t kvh, double ks, double vs,
                std::mt19937& rng) {
    const int64_t ld = kvh * HD, nb = (T + BS - 1) / BS + 2;
    std::vector<uint16_t> k((size_t)(T * ld)), v((size_t)(T * ld));
    std::uniform_real_distribution<float> mag(std::log(1e-4f), std::log(2e3f));
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = rad_f32_to_bf16(std::exp(mag(rng)) * (rng() & 1 ? -1.f : 1.f));
        v[i] = rad_f32_to_bf16(std::exp(mag(rng)) * (rng() & 1 ? -1.f : 1.f));
    }
    k[3] = 0x7f80;
    v[5] = 0x7fc0;
    std::vector<int32_t> all((size_t)(nb * BS)), slot((size_t)T);
    for (size_t i = 0; i < all.size(); ++i) all[i] = (int32_t)i;
    std::shuffle(all.begin(), all.end(), rng);
    for (int64_t t = 0; t < T; ++t) slot[(size_t)t] = all[(size_t)t];
    if (T > 2) slot[1] = -1;
    const size_t el = (size_t)(nb * kvh * BS * 2 * HD);
    Buf want(el * 2, 0xa5), got(el * 2, 0xa5);
    auto run = [&](Fn fn, uint8_t* cache) {
        RadTensor t[4];
        t[0] = tens(k.data(), RAD_BF16, { T, kvh * HD });
        t[1] = tens(v.data(), RAD_BF16, { T, kvh * HD });
        t[2] = tens(slot.data(), RAD_I32, { T });
        t[3] = tens(cache, RAD_BF16, { nb, kvh, BS, 2 * HD });
        Params p;
        p.i("M", T); p.i("head_dim", HD); p.i("n_head_kv", kvh); p.i("block_size", BS);
        p.s("kv_dtype", "bf16_e4m3r"); p.f("k_scale", ks); p.f("v_scale", vs);
        RadArgs a{};
        a.t = t; a.n_t = 4; a.p = p.p.data(); a.n_p = (int)p.p.size();
        return fn(&a, nullptr);
    };
    const int rr = run(ref, want.p), re = run(r4d_rad_kv_store_bf16_e4m3r, got.p);
    int64_t diff = 0, moved = 0, nonbf = 0;
    const uint16_t* w = (const uint16_t*)want.p;
    for (size_t i = 0; i < el; ++i) {
        diff += w[i] != ((const uint16_t*)got.p)[i];
        if (w[i] == 0xa5a5) continue;
        ++moved;
        /* Each stored value is an E4M3 value times the descale: at descale 1, three mantissa bits. */
        if (ks == 1.0 && vs == 1.0) nonbf += (w[i] & 0x0f) != 0;
    }
    const bool good = rr == RAD_OK && re == RAD_OK && diff == 0 &&
                      moved >= (T - 1) * kvh * 2 * HD && nonbf == 0;
    if (!good)
        fail(std::string(name) + ": libref " + std::to_string(rr) + ", emulated " +
             std::to_string(re) + ", " + std::to_string(diff) + " elements differ, " +
             std::to_string(moved) + " written, " + std::to_string(nonbf) + " past E4M3's mantissa");
    std::printf("  %-48s %3lld tokens x %lld heads: %s\n", name, (long long)T, (long long)kvh,
                good ? "libref's bytes" : "DIFFERS");
}

/* ------------------------------------------------------------------ the attention */

struct Geo {
    const char* tag;
    int gqa;
    Fn prefill, decode;
    ScratchFn prefill_scratch, decode_scratch;
    int verify_max;   /* the decode rows' q_len bound: 64 / gqa */
};

struct Case {
    std::string name;
    bool decode;                /* the decode (and verify) kernel; else the prefill one */
    int kvh;
    std::vector<int> qlen, ctx; /* per sequence */
    bool ragged;
    bool decisive;
    bool scratch = true;        /* decode always takes it; prefill may split with it */
    bool must_split = false;    /* prefill: the KV split and its merge, two launches */
};

void attn_case(Fn ref, const Geo& g, const Case& c, std::mt19937& rng) {
    const int n_seq = (int)c.qlen.size(), kvh = c.kvh, nh = kvh * g.gqa;
    int rows = 0, maxq = 0, maxctx = 0;
    for (int s = 0; s < n_seq; ++s) {
        rows += c.qlen[(size_t)s];
        maxq = std::max(maxq, c.qlen[(size_t)s]);
        maxctx = std::max(maxctx, c.ctx[(size_t)s]);
    }
    /* Blocks at shuffled ids, a spare column in the table and spare blocks no table names. */
    const int maxb = (maxctx + BS - 1) / BS + 1;
    const int nblocks = n_seq * maxb + 3;
    std::vector<int> ids((size_t)nblocks);
    for (int i = 0; i < nblocks; ++i) ids[(size_t)i] = i;
    std::shuffle(ids.begin(), ids.end(), rng);
    std::vector<int32_t> bt((size_t)(n_seq * maxb), -1);
    int nx = 0;
    for (int s = 0; s < n_seq; ++s)
        for (int b = 0; b < (c.ctx[(size_t)s] + BS - 1) / BS; ++b) bt[(size_t)(s * maxb + b)] = ids[(size_t)nx++];

    /* The cache poisoned first: codes 0x88 (-8, never stored) and NaN scales everywhere, so a
     * kernel that reads a slot no visible key lives in turns its row NaN. */
    const int ROW = L->row;
    const size_t bytes = (size_t)nblocks * kvh * BS * ROW;
    /* i8h's never-stored code is -128. */
    Buf cache(bytes, L->i8 ? 0x80 : 0x88);
    for (size_t r = 0; r < bytes / ROW; ++r)
        for (int i = L->scales; i < ROW; i += 2) { cache.p[r * ROW + i] = 0x00; cache.p[r * ROW + i + 1] = 0x7e; }

    /* Every sequence's keys, stored by libref's kv_store, a token at a time in key order. */
    std::normal_distribution<float> nrm(0.f, 1.f);
    std::uniform_int_distribution<int> code(L->i8 ? -127 : -7, L->i8 ? 127 : 7);
    std::vector<uint16_t> qb((size_t)rows * nh * HD, 0);
    /* Decisive: per (sequence, query row, head), a target key among the visible ones; a target's
     * K is 28 (code 7, scale 4) at a dim no other target of the sequence uses, every other key's K
     * is zero, and the query is 128 at the target's dim. Score 3584 * scale (1/16) = 224 over 0. */
    std::vector<std::vector<int>> dim_of((size_t)n_seq);
    {
        int tok0 = 0;
        for (int s = 0; s < n_seq; ++s) {
            const int ctx = c.ctx[(size_t)s], ql = c.qlen[(size_t)s];
            std::vector<uint16_t> kt((size_t)ctx * kvh * HD), vt((size_t)ctx * kvh * HD);
            std::vector<int>& dim = dim_of[(size_t)s];
            dim.assign((size_t)ctx * kvh, -1);
            if (c.decisive) {
                /* Targets: for each row and head a visible key, drawn so the sequence's targets
                 * of one kv head keep distinct dims (at most HD of them). */
                std::vector<int> used((size_t)kvh, 0);
                for (int i = 0; i < ql; ++i)
                    for (int h = 0; h < nh; ++h) {
                        const int kh = h / g.gqa, pos = ctx - ql + i;
                        int j = (int)(rng() % (unsigned)(pos + 1));
                        if (dim[(size_t)(j * kvh + kh)] < 0) {
                            if (used[(size_t)kh] >= HD) j = -1;
                            else dim[(size_t)(j * kvh + kh)] = used[(size_t)kh]++;
                        }
                        if (j < 0) continue;   /* out of dims: the row is all zero, uniform */
                        const int d = dim[(size_t)(j * kvh + kh)];
                        qb[((size_t)(tok0 + i) * nh + h) * HD + (size_t)d] = rad_f32_to_bf16(128.0f);
                    }
                /* Shuffle which dim each target holds, so a target is not simply its rank. */
                for (int kh = 0; kh < kvh; ++kh) {
                    std::vector<int> perm(HD);
                    for (int d = 0; d < HD; ++d) perm[(size_t)d] = d;
                    std::shuffle(perm.begin(), perm.end(), rng);
                    for (int j = 0; j < ctx; ++j) {
                        int& d = dim[(size_t)(j * kvh + kh)];
                        if (d >= 0) d = perm[(size_t)d];
                    }
                    for (int i = 0; i < ql; ++i)
                        for (int h = kh * g.gqa; h < (kh + 1) * g.gqa; ++h) {
                            uint16_t* qr = &qb[((size_t)(tok0 + i) * nh + h) * HD];
                            std::vector<uint16_t> tmp(qr, qr + HD);
                            for (int d = 0; d < HD; ++d) qr[d] = 0;
                            for (int d = 0; d < HD; ++d)
                                if (tmp[(size_t)d]) qr[perm[(size_t)d]] = tmp[(size_t)d];
                        }
                }
                for (int j = 0; j < ctx; ++j)
                    for (int kh = 0; kh < kvh; ++kh) {
                        const int d = dim[(size_t)(j * kvh + kh)];
                        /* V: whole codes times 2^e, one of them +-7 so the stored scale is 2^e.
                         * k4h_v2h: odd levels, +-3 at even dims and +-1 at odd, a mean square of
                         * 5 a group, so each value is its own level and the scale that keeps the
                         * group's sum of squares is 2^e. i8h: codes to +-127, dim 0 at +-127 (one
                         * scale a head), each exact in f16 and in bf16. K's 28 is then stored as
                         * 127 * f16(28 / 127), about 28: still decisive, only not a power of two. */
                        const int e = (int)(rng() % 5) - 2;
                        for (int x = 0; x < HD; ++x) {
                            const size_t at = ((size_t)j * kvh + kh) * HD + (size_t)x;
                            kt[at] = rad_f32_to_bf16(x == d ? 28.0f : 0.0f);
                            const int cv = L->v2 ? (rng() & 1 ? 1 : -1) * (x % 2 ? 1 : 3)
                                         : x % (L->i8 ? HD : 32) == 0
                                             ? (rng() & 1 ? 1 : -1) * (L->i8 ? 127 : 7) : code(rng);
                            vt[at] = rad_f32_to_bf16(std::ldexp((float)cv, e));
                        }
                    }
            } else {
                for (size_t x = 0; x < kt.size(); ++x) {
                    kt[x] = rad_f32_to_bf16(nrm(rng));
                    vt[x] = rad_f32_to_bf16(nrm(rng) * 0.7f);
                }
                for (int i = 0; i < ql; ++i)
                    for (int x = 0; x < nh * HD; ++x)
                        qb[(size_t)(tok0 + i) * nh * HD + (size_t)x] = rad_f32_to_bf16(nrm(rng));
            }
            std::vector<int32_t> slots((size_t)ctx);
            for (int j = 0; j < ctx; ++j) slots[(size_t)j] = bt[(size_t)(s * maxb + j / BS)] * BS + j % BS;
            const Fn kst = load_libref("ref_kv_store");
            if (!kst || store(kst, kt.data(), vt.data(), ctx, kvh, kvh * HD, slots.data(), cache.p,
                              nblocks, 1.0, 1.0) != RAD_OK) {
                fail(c.name + ": libref's kv_store would not build the cache");
                return;
            }
            tok0 += ql;
        }
    }

    std::vector<int32_t> su(c.ctx.begin(), c.ctx.end()), cu((size_t)n_seq + 1, 0);
    for (int s = 0; s < n_seq; ++s) cu[(size_t)s + 1] = cu[(size_t)s] + c.qlen[(size_t)s];
    const size_t oel = (size_t)rows * nh * HD;
    Buf qbuf(qb.size() * 2), want(oel * 2, 0xff), got(oel * 2, 0xff);
    std::memcpy(qbuf.p, qb.data(), qb.size() * 2);

    RadTensor t[8];
    t[0] = tens(qbuf.p, RAD_BF16, { rows, nh * HD });
    t[1] = tens(cache.p, RAD_F8E4M3, { nblocks, kvh, BS, ROW });
    t[2] = tens(bt.data(), RAD_I32, { n_seq, maxb });
    t[3] = tens(su.data(), RAD_I32, { n_seq });
    t[4] = RadTensor{};
    t[5] = RadTensor{};
    t[6] = c.ragged ? tens(cu.data(), RAD_I32, { n_seq + 1 }) : RadTensor{};
    t[7] = tens(want.p, RAD_BF16, { rows, nh, HD });
    Params p;
    /* The band bound: the row's own for decode, past the step's for prefill. */
    p.i("q_len", c.decode ? g.verify_max : maxq + 3);
    p.i("head_dim", HD); p.i("gqa", g.gqa); p.i("block_size", BS); p.i("causal", 1);
    p.i("window", 0); p.s("q_dtype", "bf16"); p.s("kv_dtype", L->kv_dtype);
    p.f("scale", 1.0 / 16.0);
    p.i("n_head", nh); p.i("max_seqs", n_seq); p.i("max_ctx", maxctx);
    RadArgs a{};
    a.t = t;
    a.n_t = 8;
    a.p = p.p.data();
    a.n_p = (int)p.p.size();
    if (ref(&a, nullptr) != RAD_OK) {
        fail(c.name + ": libref refused the case");
        return;
    }

    t[7] = tens(got.p, RAD_BF16, { rows, nh, HD });
    const ScratchFn sf = c.decode ? g.decode_scratch : g.prefill_scratch;
    const int64_t sb = c.scratch ? sf(&a) : 0;
    Buf scratch((size_t)std::max<int64_t>(sb, 16));
    a.scratch = sb > 0 ? scratch.p : nullptr;
    a.scratch_bytes = sb;
    const int g0 = emu_grids;
    const int rc = (c.decode ? g.decode : g.prefill)(&a, nullptr);
    const int launches = emu_grids - g0;
    if (rc != RAD_OK) {
        fail(c.name + ": the emulated kernel returned " + std::to_string(rc));
        return;
    }

    const uint16_t* w = (const uint16_t*)want.p;
    const uint16_t* o = (const uint16_t*)got.p;
    int64_t diff = 0, bad = 0, nan = 0, nonzero = 0;
    double se = 0, sr = 0;
    for (size_t i = 0; i < oel; ++i) {
        diff += w[i] != o[i];
        const float r = rad_bf16_to_f32(w[i]), x = rad_bf16_to_f32(o[i]);
        nonzero += r != 0.0f;
        if (std::isnan(x) || std::isnan(r)) { ++nan; continue; }
        const double e = std::fabs((double)x - r);
        se += e * e;
        sr += (double)r * r;
        if (e > 1e-2 * std::max(1.0, (double)std::fabs(r)) + 4e-3 * std::fabs(r)) ++bad;
    }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    bool ok = nan == 0 && nonzero > (int64_t)oel / 4;
    if (c.decisive) ok = ok && diff == 0;
    if (c.must_split && launches < 2) {
        fail(c.name + " (" + g.tag + "): ran unsplit, so the split's merge went unchecked");
        ok = false;
    }
    else ok = ok && bad == 0 && rel < 5e-3;
    std::printf("  %-48s %s %3d rows, %d launch(es): %s\n", c.name.c_str(), g.tag, rows, launches,
                c.decisive ? (diff ? "DIFFERS" : "libref's bytes")
                           : (ok ? "within the attention rounding" : "OUT OF BOUNDS"));
    if (!ok)
        fail(c.name + " (" + g.tag + "): " + std::to_string(diff) + " elements differ, rel_l2 " +
             std::to_string(rel) + ", " + std::to_string(bad) + " out of bounds, " +
             std::to_string(nan) + " NaN, " + std::to_string(nonzero) + " nonzero of " +
             std::to_string(oel));
}

}  // namespace

/* [--layout k4h_v4|k4h_v2h|i8h]: one layout's cases only, so ctest runs the three as separate
 * tests (each takes about a third of the whole, which grew past the per-test limit's reach with every
 * layout); the bf16_e4m3r experiment's store runs with i8h. Every layout seeds its own generator, so a
 * layout's cases are the same whether it runs alone or with the others. */
int main(int argc, char** argv) {
    const char* only = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--layout" && i + 1 < argc) only = argv[++i];
        else { std::fprintf(stderr, "usage: %s [--layout k4h_v4|k4h_v2h|i8h]\n", argv[0]); return 2; }
    }
    if (only && std::strcmp(only, "k4h_v4") && std::strcmp(only, "k4h_v2h") && std::strcmp(only, "i8h")) {
        std::fprintf(stderr, "--layout takes k4h_v4, k4h_v2h or i8h\n");
        return 2;
    }
    const Fn kst = load_libref("ref_kv_store");
    const Fn att = load_libref("ref_attn_paged");
    if (!kst || !att) {
        std::printf("FAIL libref's kv_store and attn_paged not loadable (set RADIANCE_HOME)\n");
        return 1;
    }
    std::mt19937 rng(20261104);

    struct Run {
        const Layout* layout;
        Fn store;
        Geo geos[2];
    };
    const Run runs[] = {
        { &kK4hv4, r4d_rad_kv_store_k4hv4,
          { { "gqa12", 12, r4d_rad_attn_prefill_h256_gqa12_k4hv4, r4d_rad_attn_decode_h256_gqa12_k4hv4,
              r4d_rad_attn_prefill_h256_gqa12_k4hv4_scratch,
              r4d_rad_attn_decode_h256_gqa12_k4hv4_scratch, 5 },
            { "gqa6 ", 6, r4d_rad_attn_prefill_h256_gqa6_k4hv4, r4d_rad_attn_decode_h256_gqa6_k4hv4,
              r4d_rad_attn_prefill_h256_gqa6_k4hv4_scratch,
              r4d_rad_attn_decode_h256_gqa6_k4hv4_scratch, 10 } } },
        { &kK4hv2h, r4d_rad_kv_store_k4hv2h,
          { { "gqa12", 12, r4d_rad_attn_prefill_h256_gqa12_k4hv2h, r4d_rad_attn_decode_h256_gqa12_k4hv2h,
              r4d_rad_attn_prefill_h256_gqa12_k4hv2h_scratch,
              r4d_rad_attn_decode_h256_gqa12_k4hv2h_scratch, 5 },
            { "gqa6 ", 6, r4d_rad_attn_prefill_h256_gqa6_k4hv2h, r4d_rad_attn_decode_h256_gqa6_k4hv2h,
              r4d_rad_attn_prefill_h256_gqa6_k4hv2h_scratch,
              r4d_rad_attn_decode_h256_gqa6_k4hv2h_scratch, 10 } } },
        { &kI8h, r4d_rad_kv_store_i8h,
          { { "gqa12", 12, r4d_rad_attn_prefill_h256_gqa12_i8h, r4d_rad_attn_decode_h256_gqa12_i8h,
              r4d_rad_attn_prefill_h256_gqa12_i8h_scratch,
              r4d_rad_attn_decode_h256_gqa12_i8h_scratch, 5 },
            { "gqa6 ", 6, r4d_rad_attn_prefill_h256_gqa6_i8h, r4d_rad_attn_decode_h256_gqa6_i8h,
              r4d_rad_attn_prefill_h256_gqa6_i8h_scratch,
              r4d_rad_attn_decode_h256_gqa6_i8h_scratch, 10 } } },
    };
    for (size_t ri = 0; ri < sizeof runs / sizeof runs[0]; ++ri) {
        const Run& run = runs[ri];
        if (only && std::strcmp(only, run.layout->kv_dtype)) continue;
        L = run.layout;
        rng.seed(20261104u + (unsigned)ri);
        std::printf("kv_store %s: the emulated kernel against libref\n", L->kv_dtype);
        store_case(kst, run.store, "Flash-Next's two kv heads, a prefill chunk", 37, 2, 0, 1.0, 1.0,
                   false, rng);
        store_case(kst, run.store, "slices of the fused qkv row, edges", 9, 2, 3 * 256 + 64, 1.0,
                   1.0, true, rng);
        store_case(kst, run.store, "one kv head (four ranks), a decode token", 1, 1, 0, 1.0, 1.0,
                   false, rng);
        store_case(kst, run.store, "a descale on each side", 6, 2, 0, 0.75, 1.5, true, rng);

        std::printf("attn_paged %s: the emulated kernels against libref\n", L->kv_dtype);
        for (const Geo& g : run.geos) {
            const int vq = g.verify_max;
            const std::vector<Case> cases = {
                { "decode, three sequences, decisive", true, 1, { 1, 1, 1 }, { 1, 77, 301 }, false, true },
                { "decode at depth 1500, decisive", true, 2, { 1 }, { 1500 }, false, true },
                { "decode, spread", true, 2, { 1, 1 }, { 45, 640 }, false, false },
                { "verify, MTP's four rows, decisive", true, 1, { 4, 4 }, { 60, 203 }, false, true },
                { "verify at the band's bound, ragged, spread", true, 2, { vq, 1 }, { 90, 33 }, true, false },
                { "prefill, one chunk from empty, decisive", false, 1, { 70 }, { 70 }, false, true },
                { "prefill chunk over a history, spread", false, 2, { 40 }, { 230 }, false, false },
                { "prefill, ragged with a decode row, decisive", false, 2, { 33, 1 }, { 120, 57 }, true, true },
                { "prefill without scratch, spread", false, 1, { 50 }, { 50 }, false, false, false },
                /* A short chunk over a deep history: the KV split, partials merged. */
                { "prefill chunk over a deep history, split, decisive", false, 1, { 24 }, { 1800 }, false, true, true, true },
                { "prefill chunk over a deep history, split, spread", false, 2, { 40 }, { 1300 }, false, false, true, true },
            };
            for (const Case& c : cases) attn_case(att, g, c, rng);
        }
    }

    if (!only || !std::strcmp(only, "i8h")) {
        rng.seed(20261104u + 100u);
        std::printf("kv_store bf16_e4m3r (the experiment): the emulated kernel against libref\n");
        e4m3r_case(kst, "two kv heads, a prefill chunk", 21, 2, 1.0, 1.0, rng);
        e4m3r_case(kst, "a descale on each side", 7, 1, 0.75, 1.5, rng);
    }

    std::printf(g_fails ? "k4h layouts: %d failure(s)\n" : "k4h layouts: all cases pass\n", g_fails);
    return g_fails ? 1 : 0;
}
