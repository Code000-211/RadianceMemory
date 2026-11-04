/* tests/r4d_emu: QSA block keys split across tensor-parallel ranks (--qsa-bkey-split, docs/QSA.md,
 * "Block keys split across ranks") on the host SIMT emulator, against libref, which is the oracle
 * (docs/OPS.md), so the test needs it: ctest sets $RADIANCE_HOME, and without one the test fails
 * rather than checking nothing.
 *
 *   - the selection: libr4d's qsa_select_part (r4d_qsa_select_bf16.hip) on every rank's share,
 *     each list row byte for byte libref's (zeros past the count and after the count word
 *     included); the rows laid side by side as the row-interleaved all_gather lays them out;
 *     libr4d's qsa_select_join over them byte for byte libref's join AND the emulated unsplit
 *     qsa_select over the whole plane (table, count, seqused). At two and three ranks, both
 *     rankings, a share cached in LDS (64 rows or fewer) and one read from the plane (more rows),
 *     score sets small enough that the threshold is a tie on most rows, dead pages and -inf, and
 *     rows over topk, rows whose shares are each under topk but the union over, and rows under it;
 *   - the work list: libr4d's qsa_work (r4d_qsa_work_bf16.hip) at `ranks` 2, each rank's page,
 *     position, nc and staging plane byte for byte libref's (the other rank's items untouched),
 *     and both ranks together storing every completing block exactly once. */
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

/* libref's row for op `name`, through the plugin's own exports, or null (as test_bkey.cpp). */
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

int launch(RadLaunchFn fn, std::vector<RadTensor> t, std::vector<RadParam> p) {
    RadArgs a{};
    a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size();
    return fn(&a, nullptr);
}

RadParam pi(const char* k, long long v) { return RadParam{ k, RAD_P_INT, v, 0, nullptr, 0.0 }; }

constexpr int32_t kSent = 0x5A5A5A5A;

/* A history of three sequences: one deep (every row past topk), one shallow (under it) and one
 * straddling it, each query row bounded by its own position; scores from `levels` values. */
struct Fix {
    int64_t S = 3, M = 0, nb = 0, topk = 0, ratio = 4;
    bool sgn = false;
    std::vector<float> sc;
    std::vector<int32_t> bt, nc, pos, cu;
};

Fix fixture(uint32_t seed, int64_t nb, int64_t topk, int levels, bool sgn, int64_t rows_each) {
    Fix f;
    f.nb = nb; f.topk = topk; f.sgn = sgn;
    uint32_t x = seed;
    auto rnd = [&]() { x = x * 1664525u + 1013904223u; return x >> 8; };
    const int64_t first[3] = { nb - 3 * rows_each - 4, topk / 2, topk - 3 };
    f.cu = { 0 };
    for (int64_t s = 0; s < f.S; ++s) {
        for (int64_t i = 0; i < rows_each; ++i)
            f.pos.push_back((int32_t)((first[s] + 2 * i) * f.ratio + (int64_t)(rnd() % 4)));
        f.cu.push_back(f.cu.back() + (int32_t)rows_each);
        f.nc.push_back((int32_t)std::min<int64_t>((f.pos.back() + 1) / f.ratio, nb - 1));
    }
    f.M = f.cu.back();
    f.bt.resize((size_t)(f.S * nb));
    for (auto& v : f.bt) v = rnd() % 29 == 0 ? -1 : (int32_t)(rnd() % 100000);
    f.sc.resize((size_t)(f.M * nb));
    for (auto& v : f.sc) {
        const uint32_t u = rnd();
        v = u % 31 == 0 ? -INFINITY
          : u % 37 == 0 ? (sgn ? NAN : -0.0f)
          : sgn ? 0.5f * (float)(int)(u % (uint32_t)levels) - 0.25f * (float)levels
                : 0.5f * (float)(u % (uint32_t)levels);
    }
    /* The deep sequence's first row: every odd block dead, so at two ranks one list is over topk
     * and the other empty -- exactly topk candidates with a flag, the case where only the flag
     * tells the threshold layout from the every-block one. */
    for (int64_t b = 1; b < nb; b += 2) f.sc[(size_t)b] = -INFINITY;
    return f;
}

struct Out { std::vector<int32_t> sel, nsel, su; };

void select_case(const char* what, const Fix& f, int64_t R, RadLaunchFn rpart, RadLaunchFn rjoin) {
    const int64_t W = (2 * f.topk + 1 + 3) / 4 * 4, nbl = (f.nb + R - 1) / R;
    std::vector<RadParam> base = { pi("M", f.M), pi("blocks", f.nb), pi("topk", f.topk),
                                   pi("ratio", f.ratio), pi("seqs", f.S) };
    std::vector<int32_t> bt = f.bt, nc = f.nc, pos = f.pos, cu = f.cu;
    auto common = [&](void* first, uint32_t dt, std::vector<int64_t> shape) {
        return std::vector<RadTensor>{ dense(first, dt, shape), dense(bt.data(), RAD_I32, { f.S, f.nb }),
                                       dense(nc.data(), RAD_I32, { f.S }),
                                       dense(pos.data(), RAD_I32, { f.M }),
                                       dense(cu.data(), RAD_I32, { f.S + 1 }) };
    };
    auto outs = [&](Out& o, std::vector<RadTensor>& t) {
        o.sel.assign((size_t)(f.M * (f.topk + 1)), kSent);
        o.nsel.assign((size_t)f.M, kSent);
        o.su.assign((size_t)f.M, kSent);
        t.push_back(dense(o.sel.data(), RAD_I32, { f.M, f.topk + 1 }));
        t.push_back(dense(o.nsel.data(), RAD_I32, { f.M }));
        t.push_back(dense(o.su.data(), RAD_I32, { f.M }));
    };
    /* The unsplit selection, emulated. */
    std::vector<float> sc = f.sc;
    Out want;
    {
        std::vector<RadTensor> t = common(sc.data(), RAD_F32, { f.M, f.nb });
        outs(want, t);
        std::vector<RadParam> p = base;
        if (f.sgn) p.push_back(pi("signed", 1));
        if (launch(r4d_rad_qsa_select, t, p) != RAD_OK) { fail(std::string(what) + ": qsa_select refused"); return; }
    }
    /* Each rank's list, emulated and libref, and the gathered rows. */
    std::vector<int32_t> gath((size_t)(f.M * R * W), kSent), gref = gath;
    long over = 0, edge = 0;
    for (int64_t r = 0; r < R; ++r) {
        std::vector<float> share((size_t)(f.M * nbl), -INFINITY);
        for (int64_t m = 0; m < f.M; ++m)
            for (int64_t j = 0; j * R + r < f.nb; ++j)
                share[(size_t)(m * nbl + j)] = f.sc[(size_t)(m * f.nb + j * R + r)];
        std::vector<RadParam> p = base;
        p.push_back(pi("ranks", R));
        p.push_back(pi("rank", r));
        if (f.sgn) p.push_back(pi("signed", 1));
        std::vector<int32_t> pe((size_t)(f.M * W), kSent), pr = pe;
        std::vector<RadTensor> te = common(share.data(), RAD_F32, { f.M, nbl });
        std::vector<RadTensor> tr = te;
        te.push_back(dense(pe.data(), RAD_I32, { f.M, W }));
        tr.push_back(dense(pr.data(), RAD_I32, { f.M, W }));
        const int re = launch(r4d_rad_qsa_select_part, te, p), rr = launch(rpart, tr, p);
        if (re != RAD_OK || rr != RAD_OK) {
            fail(std::string(what) + ": qsa_select_part refused, " + std::to_string(re) + " / libref " +
                 std::to_string(rr));
            return;
        }
        if (pe != pr) fail(std::string(what) + ": rank " + std::to_string(r) + "'s list is not libref's");
        for (int64_t m = 0; m < f.M; ++m) {
            std::memcpy(&gath[(size_t)((m * R + r) * W)], &pe[(size_t)(m * W)], (size_t)W * 4);
            std::memcpy(&gref[(size_t)((m * R + r) * W)], &pr[(size_t)(m * W)], (size_t)W * 4);
            over += (pe[(size_t)(m * W + 2 * f.topk)] >> 30) & 1;
        }
    }
    /* Rows whose lists hold exactly topk candidates and a flag: only the flag says threshold. */
    for (int64_t m = 0; m < f.M; ++m) {
        int64_t tot = 0;
        bool flag = false;
        for (int64_t r = 0; r < R; ++r) {
            const int32_t w = gath[(size_t)((m * R + r) * W + 2 * f.topk)];
            tot += w & ~(1 << 30);
            flag = flag || (w >> 30);
        }
        edge += flag && tot == f.topk;
    }
    /* The join, emulated and libref, over the same lists. */
    Out got, ref;
    {
        std::vector<RadParam> p = base;
        p.push_back(pi("ranks", R));
        std::vector<RadTensor> te = common(gath.data(), RAD_I32, { f.M, R * W });
        std::vector<RadTensor> tr = common(gref.data(), RAD_I32, { f.M, R * W });
        outs(got, te);
        outs(ref, tr);
        const int re = launch(r4d_rad_qsa_select_join, te, p), rr = launch(rjoin, tr, p);
        if (re != RAD_OK || rr != RAD_OK) {
            fail(std::string(what) + ": qsa_select_join refused, " + std::to_string(re) + " / libref " +
                 std::to_string(rr));
            return;
        }
    }
    long diff_unsplit = 0, diff_ref = 0, full = 0;
    for (size_t i = 0; i < want.sel.size(); ++i) {
        diff_unsplit += got.sel[i] != want.sel[i];
        diff_ref += got.sel[i] != ref.sel[i];
    }
    for (int64_t m = 0; m < f.M; ++m) {
        diff_unsplit += (got.nsel[(size_t)m] != want.nsel[(size_t)m]) + (got.su[(size_t)m] != want.su[(size_t)m]);
        diff_ref += (got.nsel[(size_t)m] != ref.nsel[(size_t)m]) + (got.su[(size_t)m] != ref.su[(size_t)m]);
        full += want.nsel[(size_t)m] >= (int32_t)f.topk;
    }
    if (diff_unsplit || diff_ref)
        fail(std::string(what) + ": the join differs from qsa_select unsplit in " +
             std::to_string(diff_unsplit) + " entries and from libref's join in " +
             std::to_string(diff_ref));
    if (!(full > 0 && full < f.M && over > 0 && (R != 2 || edge > 0)))
        fail(std::string(what) + ": the fixture no longer reaches every form");
    std::printf("  %s: %lld rows, %ld full, %ld lists over topk, %ld with exactly topk "
                "candidates and a flag; the join is qsa_select's bytes and libref's\n", what,
                (long long)f.M, full, over, edge);
}

void work_case(RadLaunchFn rwork) {
    /* Two sequences: one a chunk starting mid-block (its first block takes three tail keys), one
     * a four-row decode completing a block. Slots permuted, ring 7 against ratio 4. */
    const int64_t n = 32, ratio = 4, ring = 7, seqs = 2, slots = 5, btw = 24;
    std::vector<int32_t> cu = { 0, 22, 26 }, sx = { 3, 1 };
    const int64_t T = cu[2], W = (T + ratio - 1) / ratio + seqs;
    const int64_t start[2] = { 11, 129 };
    std::vector<uint16_t> k((size_t)(T * n)), tail((size_t)(slots * ring * n));
    std::vector<int32_t> pos((size_t)T);
    for (int64_t s = 0, t = 0; s < seqs; ++s)
        for (int64_t i = 0; t < cu[(size_t)(s + 1)]; ++i, ++t) {
            pos[(size_t)t] = (int32_t)(start[s] + i);
            for (int64_t d = 0; d < n; ++d)
                k[(size_t)(t * n + d)] = rad_f32_to_bf16((float)((t * 31 + d * 17) % 89) / 23.0f);
        }
    for (size_t i = 0; i < tail.size(); ++i) tail[i] = rad_f32_to_bf16(-1.0f - (float)(i % 211) / 64.0f);
    std::vector<int32_t> bt((size_t)(seqs * btw));
    for (size_t i = 0; i < bt.size(); ++i) bt[i] = (int32_t)((i * 13 + 5) % 4096);
    /* The host walk's count: blocks p0 / ratio .. (p1 + 1) / ratio - 1 of each sequence. */
    long want = 0;
    for (int64_t s = 0; s < seqs; ++s)
        want += (pos[(size_t)(cu[(size_t)(s + 1)] - 1)] + 1) / ratio - pos[(size_t)cu[(size_t)s]] / ratio;
    long stored = 0;
    for (int64_t r = 0; r < 2; ++r) {
        std::vector<uint16_t> se((size_t)(W * ratio * n), 0x7FC1), sr = se;
        std::vector<int32_t> pe((size_t)W, kSent), pr = pe, be = pe, br = pe, ne((size_t)seqs, kSent), nr = ne;
        auto ts = [&](std::vector<uint16_t>& st, std::vector<int32_t>& pg, std::vector<int32_t>& bp,
                      std::vector<int32_t>& nc) {
            return std::vector<RadTensor>{
                dense(k.data(), RAD_BF16, { T, n }), dense(pos.data(), RAD_I32, { T }),
                dense(cu.data(), RAD_I32, { seqs + 1 }), dense(tail.data(), RAD_BF16, { slots, 1, ring, n }),
                dense(sx.data(), RAD_I32, { seqs, 1 }), dense(bt.data(), RAD_I32, { seqs, btw }),
                RadTensor{}, dense(st.data(), RAD_BF16, { W * ratio, n }), dense(pg.data(), RAD_I32, { W }),
                dense(bp.data(), RAD_I32, { W }), dense(nc.data(), RAD_I32, { seqs }) };
        };
        RadParam dt{ "dtype", RAD_P_STR, 0, 0, "bf16", 0.0 };
        std::vector<RadParam> p = { pi("M", T), pi("n", n), pi("ratio", ratio), pi("ring", ring),
                                    pi("seqs", seqs), pi("work", W), dt, pi("ranks", 2), pi("rank", r) };
        const int re = launch(r4d_rad_qsa_work, ts(se, pe, be, ne), p);
        const int rr = launch(rwork, ts(sr, pr, br, nr), p);
        if (re != RAD_OK || rr != RAD_OK) {
            fail("qsa_work split: refused " + std::to_string(re) + " / libref " + std::to_string(rr));
            return;
        }
        if (se != sr || pe != pr || be != br || ne != nr)
            fail("qsa_work split, rank " + std::to_string(r) + ": not libref's bytes");
        for (int64_t w = 0; w < W; ++w) stored += pe[(size_t)w] >= 0;
    }
    /* Blocks 2..7 of the chunk and block 32 of the decode, each stored on exactly one rank. */
    if (stored != want || want != 7)
        fail("qsa_work split: " + std::to_string(stored) + " blocks stored of " + std::to_string(want));
    std::printf("  qsa_work split across two ranks: libref's bytes on both, %ld of %ld blocks stored "
                "once\n", stored, want);
}

}  // namespace

int main() {
    const RadLaunchFn rpart = libref("qsa_select_part"), rjoin = libref("qsa_select_join");
    const RadLaunchFn rwork = libref("qsa_work");
    if (!rpart || !rjoin || !rwork) {
        std::printf("FAIL libref's qsa_select_part / qsa_select_join / qsa_work not loadable (set "
                    "RADIANCE_HOME)\n");
        return 1;
    }
    std::printf("QSA block keys split across ranks: the emulated kernels against libref\n");
    for (bool sgn : { false, true }) {
        const char* s = sgn ? ", signed" : "";
        /* 18 rows: the share's keys in LDS; 75 rows: read from the score plane. */
        select_case((std::string("2 ranks, ties, LDS") + s).c_str(), fixture(11 + sgn, 700, 64, 5, sgn, 6), 2,
                    rpart, rjoin);
        select_case((std::string("2 ranks, fine scores, from the plane") + s).c_str(),
                    fixture(23 + sgn, 700, 64, 300, sgn, 25), 2, rpart, rjoin);
        select_case((std::string("3 ranks, ties, LDS") + s).c_str(), fixture(37 + sgn, 900, 48, 4, sgn, 5), 3,
                    rpart, rjoin);
        select_case((std::string("2 ranks at topk 512, ties") + s).c_str(),
                    fixture(41 + sgn, 2600, 512, 9, sgn, 4), 2, rpart, rjoin);
    }
    work_case(rwork);
    std::printf(g_fails ? "split block keys: %d failure(s)\n" : "split block keys: all cases pass\n",
                g_fails);
    return g_fails ? 1 : 0;
}
