/* rad_calib_vpred.h -- each attention layer's least-squares map from K to V over a calibration run,
 * the maps a KV recipe's `vpred` layers predict V with before their store (RAD_KVQ_VPRED,
 * docs/GUIDE.md §4.4).
 *
 * On Flash-Next a per-layer affine map from K to V explains about half of centred V's variance
 * out of sample (layers 0.15 to 0.71), so storing V less K A + b leaves the integer V layouts a
 * smaller residual at no extra bits; this tap fits A and b.
 *
 * Recorded under --calib-dir with RADIANCE_CALIB_VPRED=1 (off by default: the sums cost a
 * head_dim^2 update a token and head on the host, which a --calib-dir run for the V means alone
 * should not pay), over a --calib-corpus run or beside a --kld-record, where K has had its norm and
 * rotary embedding and V left its projection, both before a rotated layer's rotation: the file is
 * the model's and not one recipe's, and the engine rotates it at load (kv_vpred_table). Each
 * step's rows come back to pinned host memory and are summed in f64 -- n, sum K, sum V, sum K K^T,
 * sum K V^T -- one synchronisation a layer and step, as the V mean tap.
 *
 * The fit is ridge least squares on the centred moments: C_kk = E[K K^T] - mu_k mu_k^T, C_kv =
 * E[K V^T] - mu_k mu_v^T, A = (C_kk + lambda I)^-1 C_kv with lambda = 1e-4 * trace(C_kk) / D (keeps
 * a direction the corpus never moved from blowing up), b = mu_v - mu_k A, solved by Cholesky in f64.
 * The fit is on K as computed, not on the K-hat the cache rounds it to: the store predicts from
 * K-hat (kv_vpred), and the map is the same either way up to the K rounding's correlation with V.
 *
 * Every rank sums its own KV heads; the rank whose heads complete the set writes the file, a
 * replicated head taken from the first rank that finishes it (the V mean tap's rule), once a layer
 * has seen RadBuildCtx::calib_tokens rows. The file is `<dir>/kv_vpred.txt` in the format
 * kv_vpred_parse reads (core/config.cpp):
 *
 *     kv_vpred 1 head_dim D kv_heads H rows R
 *     layer L head h bias  b_0 ... b_{D-1}
 *     layer L head h row i  a_i0 ... a_i(D-1)        %.9g, D rows a (layer, KV head)
 */
#pragma once

#include <arch/rad_arch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace rad::arch {

/* RADIANCE_CALIB_VPRED=1: record the maps beside the V means under --calib-dir. */
inline bool calib_vpred_env() {
    const char* e = std::getenv("RADIANCE_CALIB_VPRED");
    return e && *e && std::strcmp(e, "0") != 0;
}

/* The fit of one (layer, KV head) from its sums: [D + 1][D], A's rows then b. False where the
 * moments are not positive definite even with the ridge (no rows, or every K the same). */
inline bool vpred_fit(int64_t D, long long n, const std::vector<double>& sk,
                      const std::vector<double>& sv, const std::vector<double>& skk,
                      const std::vector<double>& skv, std::vector<double>* out) {
    if (n <= 1) return false;
    const double inv = 1.0 / (double)n;
    std::vector<double> mk((size_t)D), mv((size_t)D), c((size_t)(D * D)), r((size_t)(D * D));
    for (int64_t i = 0; i < D; ++i) { mk[(size_t)i] = sk[(size_t)i] * inv; mv[(size_t)i] = sv[(size_t)i] * inv; }
    double tr = 0.0;
    for (int64_t i = 0; i < D; ++i)
        for (int64_t j = 0; j < D; ++j) {
            /* skk holds the upper triangle (j >= i). */
            const double kk = skk[(size_t)(std::min(i, j) * D + std::max(i, j))];
            c[(size_t)(i * D + j)] = kk * inv - mk[(size_t)i] * mk[(size_t)j];
            r[(size_t)(i * D + j)] = skv[(size_t)(i * D + j)] * inv - mk[(size_t)i] * mv[(size_t)j];
        }
    for (int64_t i = 0; i < D; ++i) tr += c[(size_t)(i * D + i)];
    const double lambda = 1e-4 * tr / (double)D;
    if (!(lambda > 0.0)) return false;
    for (int64_t i = 0; i < D; ++i) c[(size_t)(i * D + i)] += lambda;
    /* Cholesky C = L L^T in place (lower triangle). */
    for (int64_t j = 0; j < D; ++j) {
        double d = c[(size_t)(j * D + j)];
        for (int64_t k = 0; k < j; ++k) d -= c[(size_t)(j * D + k)] * c[(size_t)(j * D + k)];
        if (!(d > 0.0)) return false;
        d = std::sqrt(d);
        c[(size_t)(j * D + j)] = d;
        for (int64_t i = j + 1; i < D; ++i) {
            double s = c[(size_t)(i * D + j)];
            for (int64_t k = 0; k < j; ++k) s -= c[(size_t)(i * D + k)] * c[(size_t)(j * D + k)];
            c[(size_t)(i * D + j)] = s / d;
        }
    }
    /* A = C^-1 R, a column of R at a time: forward then back substitution. */
    out->assign((size_t)((D + 1) * D), 0.0);
    std::vector<double> y((size_t)D);
    for (int64_t col = 0; col < D; ++col) {
        for (int64_t i = 0; i < D; ++i) {
            double s = r[(size_t)(i * D + col)];
            for (int64_t k = 0; k < i; ++k) s -= c[(size_t)(i * D + k)] * y[(size_t)k];
            y[(size_t)i] = s / c[(size_t)(i * D + i)];
        }
        for (int64_t i = D - 1; i >= 0; --i) {
            double s = y[(size_t)i];
            for (int64_t k = i + 1; k < D; ++k) s -= c[(size_t)(k * D + i)] * (*out)[(size_t)(k * D + col)];
            (*out)[(size_t)(i * D + col)] = s / c[(size_t)(i * D + i)];
        }
    }
    for (int64_t j = 0; j < D; ++j) {
        double s = mv[(size_t)j];
        for (int64_t i = 0; i < D; ++i) s -= mk[(size_t)i] * (*out)[(size_t)(i * D + j)];
        (*out)[(size_t)(D * D + j)] = s;
    }
    return true;
}

/* Every rank's finished heads, merged in the process. */
struct VPredMerge {
    std::mutex mu;
    std::map<std::pair<int, int64_t>, std::vector<double>> map;   /* (layer, KV head): [D + 1][D] */
    std::map<std::pair<int, int64_t>, bool> failed;
    long long rows = 0;
    bool written = false;
};
inline VPredMerge& vpred_merge() {
    static VPredMerge m;
    return m;
}

struct CalibVPred {
    struct Acc {
        /* per KV head: [D] sums of K and V, [D * D] K K^T (upper triangle) and K V^T */
        std::vector<double> sk, sv, skk, skv;
        long long rows = 0;
        bool done = false;
    };
    bool        on = false;
    std::string dir;
    long long   target = 0;
    int64_t     hd = 0, n_kv = 0, kv0 = 0, n_kv_all = 0;
    std::map<int, Acc> acc;
    bool        merged = false;

    /* As CalibVMean::begin, and only under RADIANCE_CALIB_VPRED. */
    int begin(const RadBuildCtx* ctx, const Geom& g, const std::vector<int>& layers,
              int64_t kv_first) {
        if (!ctx->calib_dir || !*ctx->calib_dir || g.attn_zero || !calib_vpred_env()) return RAD_OK;
        if (ctx->calib_tokens <= 0) {
            std::fprintf(stderr, "radiance: RADIANCE_CALIB_VPRED fits the K-to-V maps over a "
                                 "--calib-corpus run, which states the corpus's size\n");
            return RAD_E_INVAL;
        }
        on       = true;
        dir      = ctx->calib_dir;
        target   = ctx->calib_tokens;
        hd       = g.head_dim;
        n_kv     = g.n_head_kv;
        n_kv_all = g.n_head_kv_all;
        kv0      = kv_first;
        acc.clear();
        for (int l : layers) {
            Acc& a = acc[l];
            a.sk.assign((size_t)(n_kv * hd), 0.0);
            a.sv.assign((size_t)(n_kv * hd), 0.0);
            a.skk.assign((size_t)(n_kv * hd * hd), 0.0);
            a.skv.assign((size_t)(n_kv * hd * hd), 0.0);
        }
        return RAD_OK;
    }

    /* Rows [r0, r0 + n) of `k` and `v` ([max_tok, n_kv * hd] bf16): K after its norm and rotary
     * embedding, V as its projection wrote it, both unrotated. */
    void tap(RadCtx* c, int layer, rad_buf k, rad_buf v, int64_t r0, int64_t n) {
        if (!on || n <= 0) return;
        const auto it = acc.find(layer);
        if (it == acc.end() || it->second.done) return;
        Acc& a = it->second;
        const int64_t w = n_kv * hd;
        const char* dk = static_cast<const char*>(rad_buf_ptr(c, k));
        const char* dv = static_cast<const char*>(rad_buf_ptr(c, v));
        uint16_t* host = reinterpret_cast<uint16_t*>(calib_staging(c, 2 * n * w * 2));
        if (!dk || !dv || !host ||
            rad_memcpy_async(host, dk + r0 * w * 2, n * w * 2, rad_stream(c)) < 0 ||
            rad_memcpy_async(host + n * w, dv + r0 * w * 2, n * w * 2, rad_stream(c)) < 0) {
            rad_step_fail(c, "calibration: copying K and V back for their map failed");
            return;
        }
        rad_stream_sync(rad_stream(c));
        auto f = [](uint16_t b) { const uint32_t u = (uint32_t)b << 16; float x; std::memcpy(&x, &u, 4); return (double)x; };
        std::vector<double> kr((size_t)hd), vr((size_t)hd);
        for (int64_t r = 0; r < n; ++r)
            for (int64_t h = 0; h < n_kv; ++h) {
                for (int64_t d = 0; d < hd; ++d) {
                    kr[(size_t)d] = f(host[r * w + h * hd + d]);
                    vr[(size_t)d] = f(host[n * w + r * w + h * hd + d]);
                    a.sk[(size_t)(h * hd + d)] += kr[(size_t)d];
                    a.sv[(size_t)(h * hd + d)] += vr[(size_t)d];
                }
                double* kk = a.skk.data() + h * hd * hd;
                double* kv = a.skv.data() + h * hd * hd;
                for (int64_t i = 0; i < hd; ++i) {
                    const double ki = kr[(size_t)i];
                    for (int64_t j = i; j < hd; ++j) kk[i * hd + j] += ki * kr[(size_t)j];
                    for (int64_t j = 0; j < hd; ++j) kv[i * hd + j] += ki * vr[(size_t)j];
                }
            }
        a.rows += n;
        if (a.rows < target) return;
        a.done = true;
        for (const auto& [l, b] : acc)
            if (!b.done) return;
        merge(rad_rank(c));
    }

private:
    void merge(int rank) {
        if (merged) return;
        merged = true;
        VPredMerge& m = vpred_merge();
        std::lock_guard<std::mutex> lk(m.mu);
        for (const auto& [l, a] : acc)
            for (int64_t k = 0; k < n_kv; ++k) {
                const std::pair<int, int64_t> key{ l, kv0 + k };
                if (m.map.count(key) || m.failed.count(key)) continue;   /* a replica's, in */
                auto sl = [&](const std::vector<double>& x, int64_t per) {
                    return std::vector<double>(x.begin() + k * per, x.begin() + (k + 1) * per);
                };
                std::vector<double> fit;
                if (vpred_fit(hd, a.rows, sl(a.sk, hd), sl(a.sv, hd), sl(a.skk, hd * hd),
                              sl(a.skv, hd * hd), &fit))
                    m.map[key] = std::move(fit);
                else
                    m.failed[key] = true;
                m.rows = std::max(m.rows, a.rows);
            }
        if (m.written || m.map.size() + m.failed.size() != acc.size() * (size_t)n_kv_all) return;
        m.written = true;
        const std::string path = dir + "/kv_vpred.txt", tmp = path + ".tmp";
        if (!m.failed.empty()) {
            std::fprintf(stderr, "radiance: calibration: rank %d: %zu (layer, KV head) maps from K "
                                 "to V have singular moments; no %s written\n", rank,
                         m.failed.size(), path.c_str());
            return;
        }
        FILE* f = std::fopen(tmp.c_str(), "w");
        bool ok = f != nullptr;
        if (f) {
            std::fprintf(f, "# radiance: each attention layer's least-squares map from K to V by KV "
                            "head, V ~ K A + b, as the projections write them (unrotated); a KV "
                            "recipe's 'vpred FILE'\n");
            std::fprintf(f, "kv_vpred 1 head_dim %lld kv_heads %lld rows %lld\n", (long long)hd,
                         (long long)n_kv_all, m.rows);
            for (const auto& [key, mp] : m.map) {
                std::fprintf(f, "layer %d head %lld bias", key.first, (long long)key.second);
                for (int64_t j = 0; j < hd; ++j) std::fprintf(f, " %.9g", (double)(float)mp[(size_t)(hd * hd + j)]);
                std::fputc('\n', f);
                for (int64_t i = 0; i < hd; ++i) {
                    std::fprintf(f, "layer %d head %lld row %lld", key.first, (long long)key.second,
                                 (long long)i);
                    for (int64_t j = 0; j < hd; ++j) std::fprintf(f, " %.9g", (double)(float)mp[(size_t)(i * hd + j)]);
                    std::fputc('\n', f);
                }
            }
            ok = std::fclose(f) == 0;
        }
        ok = ok && std::rename(tmp.c_str(), path.c_str()) == 0;
        std::fprintf(stderr, "radiance: calibration: rank %d %s the K-to-V maps of %zu layers over "
                             "%lld tokens to %s\n", rank, ok ? "wrote" : "could not write",
                     acc.size(), m.rows, path.c_str());
    }
};

}  /* namespace rad::arch */
