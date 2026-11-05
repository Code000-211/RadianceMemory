/* LiLiCorr's correlator (r4d_lili.hip) through the SIMT emulator, at the drafter's geometry: seven
 * positions of eight candidates, eight heads of 128, a 1024-wide lattice. Each kernel against a host
 * port of libref's ref_lili.cpp in double: lili_cand over two ranks' disjoint vocabulary halves (ids
 * exact, features and the additive part to bf16), lili_attn and lili_factor to bf16, lili_walk's
 * scores to f32 rounding and its tokens exactly wherever the winning margin is not a near-tie. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::mt19937 rng(77);
static float nrm() { static std::normal_distribution<float> d(0.f, 1.f); return d(rng); }

static RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.shape[i] = shape[i]; t.stride[i] = st; st *= shape[i]; }
    return t;
}
static std::vector<uint16_t> bfv(size_t n, float sc, std::vector<float>* f) {
    std::vector<uint16_t> b(n);
    f->resize(n);
    for (size_t i = 0; i < n; ++i) { b[i] = rad_f32_to_bf16(nrm() * sc); (*f)[i] = rad_bf16_to_f32(b[i]); }
    return b;
}
struct Params {
    std::vector<RadParam> p;
    std::vector<std::string> s;
    void i(const char* k, long long v) { RadParam x{}; x.key = k; x.kind = RAD_P_INT; x.ival = v; p.push_back(x); }
    void f(const char* k, double v) { RadParam x{}; x.key = k; x.kind = RAD_P_F64; x.dval = v; p.push_back(x); }
    void str(const char* k, const char* v) { RadParam x{}; x.key = k; x.kind = RAD_P_STR; x.sval = v; p.push_back(x); }
};
/* rel_l2 and the worst element relative to the reference's scale */
static bool close(const char* what, const std::vector<double>& ref, const std::vector<float>& got, double tol) {
    double se = 0, sr = 0, worst = 0, scale = 1e-30;
    long nan = 0;
    for (double r : ref) scale = std::max(scale, std::fabs(r));
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::isnan(got[i])) { ++nan; continue; }
        const double e = std::fabs(got[i] - ref[i]);
        se += e * e; sr += ref[i] * ref[i];
        worst = std::max(worst, e / scale);
    }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    const bool ok = nan == 0 && rel < tol && worst < 8 * tol;
    std::printf("  %-34s rel_l2 %.2e  worst %.2e  nan %ld  %s\n", what, rel, worst, nan, ok ? "PASS" : "FAIL");
    return ok;
}

static int test_cand() {
    const int K = 8, RK = 9, W = 2, S = 7, N = 1024, M = 2 * S, V = 131072, w = 2 + 2 * RK;
    /* stats: each rank's row of its own vocabulary half, sorted, a tie across the ranks on row 3 */
    std::vector<float> st((size_t)M * W * w);
    std::vector<double> lse(M);
    std::vector<std::vector<std::pair<float, int>>> all(M);
    for (int m = 0; m < M; ++m) {
        double tot = 0, gmx = -INFINITY;
        std::vector<double> mxr(W), sumr(W);
        for (int r = 0; r < W; ++r) {
            std::vector<std::pair<float, int>> v;
            for (int j = 0; j < K; ++j) v.push_back({ 2.f * nrm() + 3.f, r * (V / W) + (int)(rng() % (V / W)) });
            if (m == 3 && r == 1) v[0] = { all[m][0].first, all[m][0].second + V / W };   /* a tie */
            std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
            const float mx = v[0].first, sum = 1.0f + 300.f * (float)(rng() % 100) / 100.f;
            float* row = &st[((size_t)m * W + r) * w];
            row[0] = mx; row[1] = sum;
            for (int j = 0; j < K; ++j) { row[2 + 2 * j] = (float)v[j].second; row[3 + 2 * j] = v[j].first; all[m].push_back(v[j]); }
            /* the ninth pair, below the eighth: carried for the all-gather's 16-byte rows, never kept */
            row[2 + 2 * K] = (float)(r * (V / W)); row[3 + 2 * K] = v[K - 1].first - 1.0f;
            mxr[r] = mx; sumr[r] = sum; gmx = std::max(gmx, (double)mx);
        }
        for (int r = 0; r < W; ++r) tot += sumr[r] * std::exp(mxr[r] - gmx);
        lse[m] = gmx + std::log(tot);
        std::sort(all[m].begin(), all[m].end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    }
    std::vector<float> pasf, slotf, rankf;
    auto pas = bfv((size_t)M * N, 1.f, &pasf), slot = bfv((size_t)S * N, 0.5f, &slotf), rank = bfv((size_t)K * N, 0.5f, &rankf);
    std::vector<float> lnw(5), lnb(5), upw((size_t)N * 5), upb(N);
    for (auto* v : { &lnw, &lnb, &upw, &upb }) for (auto& x : *v) x = nrm() * 0.7f;
    std::vector<int32_t> cand((size_t)M * K, -7);
    std::vector<uint16_t> feat((size_t)M * K * N), base((size_t)M * K * N);
    RadTensor t[11] = {
        tens(st.data(), RAD_F32, { M, W * w }), tens(pas.data(), RAD_BF16, { M, N }),
        tens(lnw.data(), RAD_F32, { 5 }), tens(lnb.data(), RAD_F32, { 5 }),
        tens(upw.data(), RAD_F32, { N, 5 }), tens(upb.data(), RAD_F32, { N }),
        tens(slot.data(), RAD_BF16, { S, N }), tens(rank.data(), RAD_BF16, { K, N }),
        tens(cand.data(), RAD_I32, { M, K }), tens(feat.data(), RAD_BF16, { M * K, N }),
        tens(base.data(), RAD_BF16, { M * K, N }),
    };
    Params P;
    P.i("M", M); P.i("k", K); P.i("world", W); P.i("steps", S); P.i("n", N); P.str("dtype", "bf16"); P.f("eps", 1e-5);
    P.i("row_k", RK);
    RadArgs a{}; a.t = t; a.n_t = 11; a.p = P.p.data(); a.n_p = (int)P.p.size();
    const int rc = r4d_rad_lili_cand(&a, nullptr);
    if (rc != RAD_OK) { std::printf("  lili_cand launch %d\n", rc); return 1; }
    bool ok = true;
    std::vector<double> rf, rb;
    std::vector<float> gf, gb;
    for (int m = 0; m < M; ++m) {
        double lp[8];
        for (int j = 0; j < K; ++j) {
            ok = ok && cand[(size_t)m * K + j] == all[m][j].second;
            lp[j] = all[m][j].first - lse[m];
        }
        for (int j = 0; j < K; ++j) {
            const double f[5] = { lp[j], std::exp(lp[j]), lp[j] - lp[0], (double)j / (K - 1), j == 0 ? 1.0 : 0.0 };
            double mean = 0, var = 0;
            for (double x : f) mean += x;
            mean /= 5;
            for (double x : f) var += (x - mean) * (x - mean);
            var /= 5;
            double y[5];
            for (int i = 0; i < 5; ++i) y[i] = (f[i] - mean) / std::sqrt(var + 1e-5) * lnw[i] + lnb[i];
            for (int c = 0; c < N; ++c) {
                double u = upb[c];
                for (int i = 0; i < 5; ++i) u += upw[(size_t)c * 5 + i] * y[i];
                rf.push_back(u / (1 + std::exp(-u)));
                rb.push_back((double)pasf[(size_t)m * N + c] + slotf[(size_t)(m % S) * N + c] + rankf[(size_t)j * N + c]);
                gf.push_back(rad_bf16_to_f32(feat[((size_t)m * K + j) * N + c]));
                gb.push_back(rad_bf16_to_f32(base[((size_t)m * K + j) * N + c]));
            }
        }
    }
    std::printf("  lili_cand ids (two ranks, a tie)   %s\n", ok ? "PASS" : "FAIL");
    ok = close("lili_cand features", rf, gf, 5e-3) && ok;
    ok = close("lili_cand additive part", rb, gb, 5e-3) && ok;
    return ok ? 0 : 1;
}

static int test_attn() {
    const int S = 7, K = 8, L = S * K, NH = 8, HD = 128, N = NH * HD, NS = 2, M = NS * L;
    std::vector<float> qf;
    auto qkv = bfv((size_t)M * 3 * N, 0.25f, &qf);
    std::vector<float> rel((size_t)NH * (2 * S + 1)), same(NH);
    for (auto& x : rel) x = nrm();
    for (auto& x : same) x = nrm();
    std::vector<uint16_t> out((size_t)M * N, 0xffff);
    RadTensor t[4] = { tens(qkv.data(), RAD_BF16, { M, 3 * N }), tens(rel.data(), RAD_F32, { NH, 2 * S + 1 }),
                       tens(same.data(), RAD_F32, { NH }), tens(out.data(), RAD_BF16, { M, N }) };
    Params P;
    P.i("M", M); P.i("steps", S); P.i("k", K); P.i("n_head", NH); P.i("head_dim", HD); P.str("dtype", "bf16");
    RadArgs a{}; a.t = t; a.n_t = 4; a.p = P.p.data(); a.n_p = (int)P.p.size();
    const int rc = r4d_rad_lili_attn(&a, nullptr);
    if (rc != RAD_OK) { std::printf("  lili_attn launch %d\n", rc); return 1; }
    std::vector<double> ref((size_t)M * N);
    const double scale = 1.0 / std::sqrt((double)HD);
    for (int i = 0; i < M; ++i) for (int h = 0; h < NH; ++h) {
        const int q0 = i / L * L, si = (i - q0) / K;
        std::vector<double> sc(L);
        double mx = -INFINITY;
        for (int j = 0; j < L; ++j) {
            double d = 0;
            for (int e = 0; e < HD; ++e) d += (double)qf[(size_t)i * 3 * N + h * HD + e] * qf[(size_t)(q0 + j) * 3 * N + N + h * HD + e];
            const int sj = j / K;
            sc[j] = d * scale + rel[(size_t)h * (2 * S + 1) + si - sj + S] + (si == sj ? same[h] : 0.f);
            mx = std::max(mx, sc[j]);
        }
        double sum = 0;
        for (auto& x : sc) { x = std::exp(x - mx); sum += x; }
        for (int e = 0; e < HD; ++e) {
            double acc = 0;
            for (int j = 0; j < L; ++j) acc += sc[j] * qf[(size_t)(q0 + j) * 3 * N + 2 * N + h * HD + e];
            ref[(size_t)i * N + h * HD + e] = acc / sum;
        }
    }
    std::vector<float> got(out.size());
    for (size_t i = 0; i < out.size(); ++i) got[i] = rad_bf16_to_f32(out[i]);
    return close("lili_attn two lattices", ref, got, 5e-3) ? 0 : 1;
}

static int test_factor() {
    const int L = 56, N = 1024, NS = 3, M = NS * L;
    std::vector<float> hf, af;
    auto h = bfv((size_t)M * N, 1.f, &hf), an = bfv((size_t)NS * N, 1.f, &af);
    std::vector<uint16_t> y((size_t)M * 3 * N);
    RadTensor t[3] = { tens(h.data(), RAD_BF16, { M, N }), tens(an.data(), RAD_BF16, { NS, N }),
                       tens(y.data(), RAD_BF16, { M, 3 * N }) };
    Params P;
    P.i("M", M); P.i("L", L); P.i("n", N); P.str("dtype", "bf16");
    RadArgs a{}; a.t = t; a.n_t = 3; a.p = P.p.data(); a.n_p = (int)P.p.size();
    if (r4d_rad_lili_factor(&a, nullptr) != RAD_OK) { std::printf("  lili_factor launch failed\n"); return 1; }
    long bad = 0;
    for (int i = 0; i < M; ++i) for (int c = 0; c < N; ++c) {
        const uint16_t* r = &y[(size_t)i * 3 * N];
        bad += r[c] != h[(size_t)i * N + c];
        bad += r[N + c] != an[(size_t)(i / L) * N + c];
        bad += r[2 * N + c] != rad_f32_to_bf16(hf[(size_t)i * N + c] * af[(size_t)(i / L) * N + c]);
    }
    std::printf("  %-34s bad %ld  %s\n", "lili_factor (bit for bit)", bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

static int test_walk() {
    const int S = 7, K = 8, N = 1024, M = 5, R = S * K;
    std::vector<float> ef, af;
    auto ed = bfv((size_t)M * R * 2 * N, 1.f, &ef), ao = bfv((size_t)M * N, 1.f, &af);
    std::vector<int32_t> cand((size_t)M * S * K), tok((size_t)M * S, -9);
    for (auto& c : cand) c = (int32_t)(rng() % 131072);
    std::vector<float> scores((size_t)M * S * K * K);
    RadTensor t[5] = { tens(ed.data(), RAD_BF16, { M * R, 2 * N }), tens(ao.data(), RAD_BF16, { M, N }),
                       tens(cand.data(), RAD_I32, { M * S, K }), tens(tok.data(), RAD_I32, { M, S }),
                       tens(scores.data(), RAD_F32, { M, S, K, K }) };
    Params P;
    P.i("M", M); P.i("steps", S); P.i("k", K); P.i("n", N); P.str("dtype", "bf16"); P.f("eps", 1e-4); P.f("scale", 8.0);
    RadArgs a{}; a.t = t; a.n_t = 5; a.p = P.p.data(); a.n_p = (int)P.p.size();
    if (r4d_rad_lili_walk(&a, nullptr) != RAD_OK) { std::printf("  lili_walk launch failed\n"); return 1; }
    std::vector<double> ref((size_t)M * S * K * K);
    long tbad = 0, near = 0;
    for (int m = 0; m < M; ++m) {
        auto vec = [&](const float* p, std::vector<double>& v) {
            double ss = 0;
            v.assign(p, p + N);
            for (double x : v) ss += x * x;
            const double inv = 1.0 / std::max(std::sqrt(ss), 1e-4);
            for (double& x : v) x *= inv;
        };
        std::vector<std::vector<double>> vo(R), vi(R);
        std::vector<double> va;
        vec(&af[(size_t)m * N], va);
        for (int r = 0; r < R; ++r) {
            vec(&ef[((size_t)m * R + r) * 2 * N], vo[r]);
            vec(&ef[((size_t)m * R + r) * 2 * N + N], vi[r]);
        }
        auto dot = [&](const std::vector<double>& x, const std::vector<double>& y) {
            double s = 0;
            for (int c = 0; c < N; ++c) s += x[c] * y[c];
            return 8.0 * s;
        };
        double* sc = &ref[(size_t)m * S * K * K];
        for (int j = 0; j < K; ++j) { const double v = dot(va, vi[j]); for (int i = 0; i < K; ++i) sc[i * K + j] = v; }
        for (int s = 1; s < S; ++s) for (int i = 0; i < K; ++i) for (int j = 0; j < K; ++j)
            sc[(s * K + i) * K + j] = dot(vo[(s - 1) * K + i], vi[s * K + j]);
        /* The walk on the reference scores; a step whose two best differ by less than f32 noise
         * may legitimately go the other way, and then the rest of the path is not compared. */
        int prev = 0;
        for (int s = 0; s < S; ++s) {
            const double* row = sc + (s * K + prev) * K;
            int best = 0;
            for (int j = 1; j < K; ++j) if (row[j] > row[best]) best = j;
            double second = -INFINITY;
            for (int j = 0; j < K; ++j) if (j != best) second = std::max(second, row[j]);
            if (row[best] - second < 1e-4) { ++near; break; }
            tbad += tok[(size_t)m * S + s] != cand[((size_t)m * S + s) * K + best];
            prev = best;
        }
    }
    std::printf("  %-34s bad %ld  near-ties %ld  %s\n", "lili_walk tokens", tbad, near, tbad ? "FAIL" : "PASS");
    const bool ok = close("lili_walk scores", ref, scores, 1e-4) && tbad == 0;
    return ok ? 0 : 1;
}

int main() {
    int fails = 0;
    fails += test_cand();
    fails += test_attn();
    fails += test_factor();
    fails += test_walk();
    std::printf("lili: %d failure(s)\n", fails);
    return fails ? 1 : 0;
}
