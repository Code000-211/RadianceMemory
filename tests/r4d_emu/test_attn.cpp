/* Drives libr4d's windowed attn_paged entry points (r4d_attn_paged_window.hip) through the SIMT
 * emulator and checks every output element against a port of libref's ref_attn_paged (same mask
 * predicates, f64 accumulation). */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

static std::mt19937 rng(1234);
extern int emu_grids;
extern int emu_last_block;   /* threads a block of the last grid: 512 is the WMMA kernel */
static float nrm() { static std::normal_distribution<float> d(0.f, 1.f); return d(rng); }

static RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.shape[i] = shape[i]; t.stride[i] = st; st *= shape[i]; }
    return t;
}

struct Case {
    std::string name;
    int hd, gqa, kvh;
    bool fp8, causal, ragged, descale;
    int window;
    std::vector<int> qlen, ctx;          /* per sequence */
    int missing_block = -1;              /* sequence 0's block index to mark -1, or -1 */
    long long max_ctx = 0;               /* 0: not passed */
    int64_t scratch = 0;                 /* 0: no scratch */
    int qpitch_extra = 0;                /* q as a column slice of a wider buffer */
    int expect_wmma = -1;                /* 1: must take the WMMA kernel, 0: must not, -1: either */
};

using Fn = int (*)(const RadArgs*, RadStream);
static Fn pick(int hd, int gqa, bool fp8) {
#define P(H, G) if (hd == H && gqa == G) return fp8 ? r4d_rad_attn_window_h##H##_gqa##G##_fp8 : r4d_rad_attn_window_h##H##_gqa##G##_bf16;
    P(128, 1) P(256, 1) P(256, 2) P(256, 4) P(256, 8) P(512, 1) P(512, 2) P(512, 4) P(512, 8)
    P(512, 16)
#undef P
    return nullptr;
}

static int run(const Case& c) {
    const int n_seq = (int)c.qlen.size(), HD = c.hd, kvh = c.kvh, qh = kvh * c.gqa;
    int rows = 0; for (int q : c.qlen) rows += q;
    /* blocks: each sequence gets ceil(ctx/16) blocks at shuffled ids */
    int maxb = 0; for (int x : c.ctx) maxb = std::max(maxb, (x + 15) / 16);
    maxb += 1;  /* a spare column: the table is wider than any sequence needs */
    const int nblocks = n_seq * maxb + 3;
    std::vector<int> ids(nblocks); for (int i = 0; i < nblocks; ++i) ids[i] = i;
    std::shuffle(ids.begin(), ids.end(), rng);
    std::vector<int> bt(n_seq * maxb, -1);
    int nx = 0;
    for (int s = 0; s < n_seq; ++s) for (int b = 0; b < (c.ctx[s] + 15) / 16; ++b) bt[s * maxb + b] = ids[nx++];
    if (c.missing_block >= 0) bt[c.missing_block] = -1;
    /* the cache, filled everywhere -- including slots past each context, with NaN in some, so a
     * kernel that reads a masked key's V without guarding is caught */
    const int64_t slot = 2 * HD, nel = (int64_t)nblocks * kvh * 16 * slot;
    std::vector<float> kvf(nel);
    std::vector<uint16_t> kvb(nel); std::vector<uint8_t> kv8(nel);
    for (int64_t i = 0; i < nel; ++i) {
        float v = nrm();
        if (c.fp8) { kv8[i] = rad_f32_to_fp8e4m3(v * 4.f); kvf[i] = rad_fp8e4m3_to_f32(kv8[i]); }
        else { kvb[i] = rad_f32_to_bf16(v); kvf[i] = rad_bf16_to_f32(kvb[i]); }
    }
    /* poison: every slot at or past a sequence's context in its last block, and every block no
     * table names */
    std::vector<char> named(nblocks, 0);
    for (int s = 0; s < n_seq; ++s) for (int b = 0; b < maxb; ++b) if (bt[s * maxb + b] >= 0) named[bt[s * maxb + b]] = 1;
    auto poison = [&](int64_t i) { if (c.fp8) { kv8[i] = 0x7f; } else { kvb[i] = 0x7fc0; } kvf[i] = NAN; };
    for (int blk = 0; blk < nblocks; ++blk) if (!named[blk]) for (int64_t i = 0; i < kvh * 16 * slot; ++i) poison((int64_t)blk * kvh * 16 * slot + i);
    for (int s = 0; s < n_seq; ++s) {
        const int last = (c.ctx[s] - 1) / 16, blk = bt[s * maxb + last];
        if (blk < 0) continue;
        for (int o = c.ctx[s] % 16 ? c.ctx[s] % 16 : 16; o < 16; ++o)
            for (int h = 0; h < kvh; ++h) for (int d = 0; d < slot; ++d) poison((((int64_t)blk * kvh + h) * 16 + o) * slot + d);
    }
    const int qcols = qh * HD + c.qpitch_extra;
    std::vector<uint16_t> qb((size_t)rows * qcols); std::vector<float> qf(qb.size());
    for (size_t i = 0; i < qb.size(); ++i) { qb[i] = rad_f32_to_bf16(nrm()); qf[i] = rad_bf16_to_f32(qb[i]); }
    std::vector<int> su(c.ctx), cu(n_seq + 1, 0);
    for (int s = 0; s < n_seq; ++s) cu[s + 1] = cu[s] + c.qlen[s];
    std::vector<float> kd(n_seq * kvh, 1.f), vd(n_seq * kvh, 1.f);
    if (c.descale) for (auto* v : {&kd, &vd}) for (auto& x : *v) x = 0.5f + (float)(rng() % 100) / 100.f;
    std::vector<uint16_t> out((size_t)rows * qh * HD, 0xffff);
    std::vector<char> scratch(c.scratch + 64);
    char* sp = scratch.data() + ((16 - ((uintptr_t)scratch.data() & 15)) & 15);

    RadTensor t[8];
    t[0] = tens(qb.data(), RAD_BF16, {rows, qh * HD}); t[0].stride[0] = qcols;
    t[1] = tens(c.fp8 ? (void*)kv8.data() : (void*)kvb.data(), c.fp8 ? RAD_F8E4M3 : RAD_BF16, {nblocks, kvh, 16, slot});
    t[2] = tens(bt.data(), RAD_I32, {n_seq, maxb});
    t[3] = tens(su.data(), RAD_I32, {n_seq});
    t[4] = c.descale ? tens(kd.data(), RAD_F32, {n_seq, kvh}) : RadTensor{};
    t[5] = c.descale ? tens(vd.data(), RAD_F32, {n_seq, kvh}) : RadTensor{};
    t[6] = c.ragged ? tens(cu.data(), RAD_I32, {n_seq + 1}) : RadTensor{};
    t[7] = tens(out.data(), RAD_BF16, {rows, qh, HD});
    int qmax = 0; for (int q : c.qlen) qmax = std::max(qmax, q);
    std::vector<RadParam> p;
    auto pi = [&](const char* k, long long v) { RadParam x{}; x.key = k; x.kind = RAD_P_INT; x.ival = v; p.push_back(x); };
    auto ps = [&](const char* k, const char* v) { RadParam x{}; x.key = k; x.kind = RAD_P_STR; x.sval = v; p.push_back(x); };
    auto pf = [&](const char* k, double v) { RadParam x{}; x.key = k; x.kind = RAD_P_F64; x.dval = v; p.push_back(x); };
    pi("q_len", qmax + 3);  /* the band bound, above the step's max */
    pi("head_dim", HD); pi("gqa", c.gqa); pi("block_size", 16); pi("causal", c.causal); pi("window", c.window);
    ps("q_dtype", "bf16"); ps("kv_dtype", c.fp8 ? "fp8_e4m3" : "bf16");
    const double scale = 1.0 / std::sqrt((double)HD) * 1.3;   /* not the default: the param must be read */
    pf("scale", scale); pi("n_head", qh); pi("max_seqs", n_seq);
    if (c.max_ctx) pi("max_ctx", c.max_ctx);
    RadArgs a{}; a.t = t; a.n_t = 8; a.p = p.data(); a.n_p = (int)p.size();
    a.scratch = c.scratch ? sp : nullptr; a.scratch_bytes = c.scratch;
    Fn fn = pick(HD, c.gqa, c.fp8);
    const int g0 = emu_grids;
    const int rc = fn(&a, nullptr);
    const int launches = emu_grids - g0;
    if (rc != RAD_OK) { std::printf("%-44s launch returned %d\n", c.name.c_str(), rc); return 1; }

    /* the reference */
    double max_err = 0, max_ref = 0, se = 0, sr = 0; long bad = 0, nan_out = 0;
    for (int s = 0; s < n_seq; ++s) for (int i = 0; i < c.qlen[s]; ++i) for (int h = 0; h < qh; ++h) {
        const int tok = cu[s] + i, kh = h / c.gqa, pos = c.ctx[s] - c.qlen[s] + i;
        auto visible = [&](int j) {
            if (c.causal ? j > pos : j >= c.ctx[s]) return false;
            if (c.window > 0) { const int d = pos - j; if (c.causal ? d >= c.window : (d >= c.window || d <= -c.window)) return false; }
            return bt[s * maxb + j / 16] >= 0;
        };
        std::vector<double> sc(c.ctx[s], -INFINITY); double mx = -INFINITY;
        for (int j = 0; j < c.ctx[s]; ++j) {
            if (!visible(j)) continue;
            const int64_t kb = (((int64_t)bt[s * maxb + j / 16] * kvh + kh) * 16 + j % 16) * slot;
            double dot = 0; for (int d = 0; d < HD; ++d) dot += (double)qf[(size_t)tok * qcols + h * HD + d] * kvf[kb + d] * kd[s * kvh + kh];
            sc[j] = dot * scale; mx = std::max(mx, sc[j]);
        }
        std::vector<double> acc(HD, 0.0); double sum = 0;
        if (mx > -INFINITY)
            for (int j = 0; j < c.ctx[s]; ++j) {
                if (sc[j] == -INFINITY) continue;
                const double w = std::exp(sc[j] - mx); sum += w;
                const int64_t kb = (((int64_t)bt[s * maxb + j / 16] * kvh + kh) * 16 + j % 16) * slot;
                for (int d = 0; d < HD; ++d) acc[d] += w * kvf[kb + HD + d] * vd[s * kvh + kh];
            }
        for (int d = 0; d < HD; ++d) {
            const double ref = sum > 0 ? acc[d] / sum : 0.0;
            const float got = rad_bf16_to_f32(out[((size_t)tok * qh + h) * HD + d]);
            if (std::isnan(got)) { ++nan_out; continue; }
            const double e = std::fabs(got - ref);
            max_err = std::max(max_err, e); max_ref = std::max(max_ref, std::fabs(ref));
            se += e * e; sr += ref * ref;
            if (e > 1e-2 * std::max(1.0, std::fabs(ref)) + 4e-3 * std::fabs(ref)) ++bad;
        }
    }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    const bool ok = bad == 0 && nan_out == 0 && rel < 5e-3;
    std::printf("%-44s rows %4d %s rel_l2 %.2e  max_abs %.2e  bad %ld  nan %ld  %s\n", c.name.c_str(), rows,
                launches > 1 ? "split" : emu_last_block == 512 ? "wmma " : "     ", rel, max_err, bad, nan_out,
                ok ? "PASS" : "FAIL");
    if (c.expect_wmma >= 0 && (emu_last_block == 512) != (c.expect_wmma == 1)) {
        std::printf("%-44s took the %s kernel, expected the %s one\n", c.name.c_str(),
                    emu_last_block == 512 ? "WMMA" : "scalar", c.expect_wmma ? "WMMA" : "scalar");
        return 1;
    }
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    const int64_t s128 = r4d_rad_attn_window_h128_scratch(nullptr);
    const int64_t s256 = r4d_rad_attn_window_h256_scratch(nullptr), s512 = r4d_rad_attn_window_h512_scratch(nullptr);
    std::vector<Case> cases = {
        /* Unlimited-OCR's decoder: h128 MHA, the fused q|k|v plane read at its pitch */
        {"h128 gqa1 decode split", 128, 1, 1, false, true, false, false, 0, {1}, {700}, -1, 0, s128},
        {"h128 gqa1 swa w64 verify16", 128, 1, 1, false, true, false, false, 64, {16}, {1100}},
        {"h128 gqa1 noncausal full", 128, 1, 1, false, false, false, false, 0, {3}, {77}},
        {"wmma h128 gqa1 prefill 200 of 900", 128, 1, 1, false, true, false, false, 0, {200}, {900},
         -1, 0, 0, 0, 1},
        {"wmma h128 gqa1 q-pitch slice", 128, 1, 1, false, true, false, false, 0, {96}, {150}, -1,
         0, 0, 512, 1},
        {"wmma h128 gqa1 swa w1024 ragged", 128, 1, 1, false, true, true, false, 1024, {150, 70},
         {300, 70}, -1, 0, 0, 0, 1},
        /* Gemma 4 sliding: h256 gqa2, causal, windowed; a ragged mixed step */
        {"h256 gqa2 swa w64 ragged bf16", 256, 2, 2, false, true, true, false, 64, {1, 5, 40}, {200, 77, 140}},
        {"h256 gqa2 swa w64 ragged fp8+descale", 256, 2, 2, true, true, true, true, 64, {1, 5, 40}, {200, 77, 140}},
        /* the real window, a context past it, a verify-sized query */
        {"h256 gqa2 swa w1024 verify16", 256, 2, 1, false, true, false, false, 1024, {16}, {1100}},
        /* split-KV decode: one row a sequence, long context, scratch present */
        {"h256 gqa2 swa w1024 decode split", 256, 2, 2, false, true, true, false, 1024, {1, 1}, {1300, 40}, -1, 0, s256},
        /* Gemma 4 full: h512, causal, no window, at the three TP ratios */
        {"h512 gqa8 full ragged bf16", 512, 8, 1, false, true, true, false, 0, {2, 7}, {90, 33}},
        {"h512 gqa8 full decode split fp8", 512, 8, 1, true, true, false, true, 0, {1}, {700}, -1, 4096, s512},
        {"h512 gqa4 full verify16 split", 512, 4, 1, false, true, false, false, 0, {16}, {300}, -1, 1024, s512},
        /* the serving case: max_ctx 262144 gives a decode step 64 splits, most of them empty */
        {"h512 gqa4 full decode, 64 splits, short ctx", 512, 4, 1, false, true, false, false, 0, {4}, {37}, -1, 262144, s512},
        {"h256 gqa2 swa w1024 verify, ragged, splits", 256, 2, 2, true, true, true, true, 1024, {3, 5}, {20, 900}, -1, 262144, s256},
        {"h512 gqa2 full ragged", 512, 2, 2, false, true, true, false, 0, {3, 1, 9}, {50, 64, 9}},
        /* Gemma 4 12B full at TP1: 16 q heads over its one kv head, one position a tile; the
         * assistant's full layer is the same geometry at one row a sequence */
        {"h512 gqa16 full ragged bf16", 512, 16, 1, false, true, true, false, 0, {3, 1, 9}, {50, 64, 9}},
        {"h512 gqa16 full decode split fp8", 512, 16, 1, true, true, false, true, 0, {1}, {700}, -1, 4096, s512},
        {"h512 gqa16 full verify16 split", 512, 16, 1, false, true, false, false, 0, {16}, {300}, -1, 1024, s512},
        {"h512 gqa16 full decode, 3 seqs, 64 splits", 512, 16, 1, false, true, true, false, 0, {1, 1, 1}, {40, 90, 300}, -1, 262144, s512},
        {"h512 gqa16 kvh2 noncausal missing-block", 512, 16, 2, false, false, true, false, 0, {4, 2}, {37, 20}, 1},
        /* the coverage the rows claim beyond Gemma: non-causal, two-sided window, a missing block */
        {"h256 gqa4 noncausal w20 missing-block", 256, 4, 1, false, false, true, false, 20, {6, 2}, {70, 30}, 2},
        {"h512 gqa1 noncausal full", 512, 1, 1, false, false, false, false, 0, {4}, {37}},
        {"h256 gqa8 causal q-pitch slice", 256, 8, 1, false, true, false, false, 0, {5}, {21}, -1, 0, 0, 512},
        /* no scratch where a split was wanted: falls back to the unsplit kernel */
        {"h256 gqa2 decode, split wanted, no scratch", 256, 2, 1, false, true, false, false, 0, {1}, {500}},
        /* Prefill-shaped: the WMMA kernel (a 64-row query, or longer) */
        {"wmma h512 gqa4 prefill 200 of 900", 512, 4, 1, false, true, false, false, 0, {200}, {900}, -1, 262144, s512, 0, 1},
        {"wmma h512 gqa8 prefill fp8+descale", 512, 8, 1, true, true, false, true, 0, {130}, {130}, -1, 0, 0, 0, 1},
        {"wmma h512 gqa2 kvh2 ragged + decode rows", 512, 2, 2, false, true, true, false, 0, {150, 1, 70}, {300, 50, 70}, -1, 0, 0, 0, 1},
        {"wmma h512 gqa1 noncausal", 512, 1, 1, false, false, false, false, 0, {64}, {100}, -1, 0, 0, 0, 1},
        {"wmma h512 gqa16 prefill 200 of 900", 512, 16, 1, false, true, false, false, 0, {200}, {900}, -1, 262144, s512, 0, 1},
        {"wmma h512 gqa16 ragged + decode rows fp8+descale", 512, 16, 1, true, true, true, true, 0, {150, 1, 70}, {300, 50, 70}, -1, 0, 0, 0, 1},
        {"wmma h256 gqa2 swa w1024 prefill 300 of 1500", 256, 2, 2, false, true, false, false, 1024, {300}, {1500}, -1, 0, 0, 0, 1},
        {"wmma h256 gqa2 swa w64 ragged fp8+descale", 256, 2, 2, true, true, true, true, 64, {100, 90}, {250, 90}, -1, 0, 0, 0, 1},
        {"wmma h256 gqa4 noncausal w20 missing-block", 256, 4, 1, false, false, true, false, 20, {80, 70}, {200, 70}, 3, 0, 0, 0, 1},
        {"wmma h256 gqa8 causal q-pitch slice", 256, 8, 1, false, true, false, false, 0, {64}, {64}, -1, 0, 0, 512, 1},
        {"wmma h256 gqa1 causal w5 fp8", 256, 1, 1, true, true, false, false, 5, {96}, {150}, -1, 0, 0, 0, 1},
        /* and what must stay scalar: a verify batch, many sequences, a few rows each */
        {"scalar: 8 x verify16, ragged", 512, 4, 1, false, true, true, false, 0, {16, 16, 16, 16, 16, 16, 16, 16},
         {40, 50, 60, 70, 80, 90, 100, 110}, -1, 262144, s512, 0, 0},
    };
    if (argc > 1 && std::string(argv[1]) == "sweep") {
        const int n = argc > 2 ? std::atoi(argv[2]) : 200;
        int fails = 0, splits = 0, wmmas = 0;
        const int hds[2] = {256, 512}, gqas[4] = {1, 2, 4, 8};
        for (int it = 0; it < n; ++it) {
            Case c;
            c.hd = hds[rng() % 2]; c.gqa = gqas[rng() % 4]; c.kvh = 1 + rng() % 3;
            c.fp8 = rng() % 2; c.causal = rng() % 4 != 0; c.ragged = rng() % 2; c.descale = rng() % 2;
            c.window = rng() % 3 == 0 ? 0 : 1 + rng() % 200;
            const int ns = 1 + rng() % 4;
            const bool pre = rng() % 3 == 0;         /* prefill-shaped: the WMMA kernel */
            const int qspan = pre ? 160 : 40, qmin = pre ? 64 : 1;
            const int ql = qmin + rng() % qspan;
            for (int s = 0; s < ns; ++s) {
                int q = c.ragged ? (int)(rng() % 3 == 0 ? 0 : qmin + rng() % qspan) : ql;
                if (c.ragged && s == 0 && q == 0) q = 1;
                c.qlen.push_back(q);
                c.ctx.push_back(q + (int)(rng() % 400));
            }
            if (rng() % 2) c.scratch = c.hd == 256 ? s256 : s512;
            if (rng() % 2) c.max_ctx = 512;
            if (rng() % 5 == 0) c.missing_block = (int)(rng() % ((c.ctx[0] + 15) / 16));
            c.name = "sweep " + std::to_string(it) + " h" + std::to_string(c.hd) + " g" + std::to_string(c.gqa) +
                     " kvh" + std::to_string(c.kvh) + (c.fp8 ? " fp8" : " bf16") + (c.causal ? " c" : " nc") +
                     " w" + std::to_string(c.window) + (c.ragged ? " rag" : "");
            const int g0 = emu_grids;
            fails += run(c);
            splits += emu_grids - g0 > 1;
            wmmas += emu_last_block == 512;
        }
        std::printf("sweep: %d cases, %d took the split path, %d the WMMA kernel, %d failure(s)\n", n, splits,
                    wmmas, fails);
        return fails ? 1 : 0;
    }
    /* [--shard k/n] [name filter]: --shard runs every n-th case from the k-th, so ctest can split
     * the run into parallel entries. */
    int shard = 0, shards = 1;
    const char* filter = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--shard" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%d/%d", &shard, &shards) != 2 || shards < 1 || shard < 0 ||
                shard >= shards) {
                std::fprintf(stderr, "--shard takes k/n with 0 <= k < n\n");
                return 2;
            }
        } else {
            filter = argv[i];
        }
    }
    int fails = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const Case& c = cases[i];
        if ((int)(i % (size_t)shards) != shard) continue;
        if (filter && c.name.find(filter) == std::string::npos) continue;
        fails += run(c);
    }
    std::printf("%d failure(s)\n", fails);
    return fails ? 1 : 0;
}
