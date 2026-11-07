/* rad_calib_vmean.h -- each attention layer's mean of V over a calibration run, the means a KV
 * recipe's `vmean` layers subtract before their store (RAD_KVQ_VMEAN, docs/GUIDE.md §4.4).
 *
 * On Flash-Next V carries a large per-channel offset (the norm of its mean is 0.53 to 0.89 of V's
 * RMS by layer), which a symmetric group scale spends its range on. A mean taken over other
 * documents is nearly as good as a document's own, so one static vector a layer and KV head
 * serves every request; this tap measures it.
 *
 * Recorded under --calib-dir (RadBuildCtx::calib_dir), over a --calib-corpus run or beside a
 * --kld-record, as V leaves its projection: before a rotated layer's rotation, so the file is the
 * model's and not one recipe's, and the engine rotates it for a rotated layer at load
 * (kv_vmean_table). Each step's rows come back to pinned host memory and are summed in f64; one
 * synchronisation a layer and step, which a calibration run can afford (a corpus of 130K tokens is
 * a few dozen steps).
 *
 * Every rank sums its own KV heads. A file is one per model, and the ranks are threads of one
 * process, so the rank whose heads complete the set (every attention layer, every KV head) writes
 * it; a replicated head (more ranks than KV heads) is taken from the first rank that finishes it.
 * The file, written once a layer has seen RadBuildCtx::calib_tokens rows (the whole corpus), is
 * `<dir>/kv_vmean.txt` in the format kv_vmean_parse reads (core/config.cpp):
 *
 *     kv_vmean 1 head_dim D kv_heads H rows R
 *     layer L head h  v_0 ... v_{D-1}            the mean, %.9g, one line a (layer, KV head)
 */
#pragma once

#include <arch/rad_arch.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace rad::arch {

/* Every rank's finished heads, merged in the process. */
struct VMeanMerge {
    std::mutex mu;
    std::map<std::pair<int, int64_t>, std::vector<double>> mean;   /* (layer, KV head) */
    long long rows = 0;
    bool written = false;
};
inline VMeanMerge& vmean_merge() {
    static VMeanMerge m;
    return m;
}

struct CalibVMean {
    struct Acc {
        std::vector<double> sum;   /* [n_kv * hd] */
        long long rows = 0;
        bool done = false;
    };
    bool        on = false;
    std::string dir;
    long long   target = 0;
    int64_t     hd = 0, n_kv = 0, kv0 = 0, n_kv_all = 0;
    std::map<int, Acc> acc;        /* this rank's attention layers */
    bool        merged = false;

    /* From the build context: nothing unless --calib-dir is set. `layers` are the model's
     * attention layers (the trunk's; the MTP head drafts and is not recorded). `kv_first` is this
     * rank's first KV head over every rank's. An attention-zero rank records nothing. */
    int begin(const RadBuildCtx* ctx, const Geom& g, const std::vector<int>& layers,
              int64_t kv_first) {
        if (!ctx->calib_dir || !*ctx->calib_dir || g.attn_zero) return RAD_OK;
        if (ctx->calib_tokens <= 0) {
            std::fprintf(stderr, "radiance: --calib-dir records the V means over a --calib-corpus "
                                 "run, which states the corpus's size\n");
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
        for (int l : layers) acc[l].sum.assign((size_t)(n_kv * hd), 0.0);
        return RAD_OK;
    }

    /* Rows [r0, r0 + n) of `v` ([max_tok, n_kv * hd] bf16) as layer `layer`'s projection wrote
     * them. */
    void tap(RadCtx* c, int layer, rad_buf v, int64_t r0, int64_t n) {
        if (!on || n <= 0) return;
        const auto it = acc.find(layer);
        if (it == acc.end() || it->second.done) return;
        Acc& a = it->second;
        const int64_t w = n_kv * hd;
        const char* dev = static_cast<const char*>(rad_buf_ptr(c, v));
        uint16_t* host = reinterpret_cast<uint16_t*>(calib_staging(c, n * w * 2));
        if (!dev || !host ||
            rad_memcpy_async(host, dev + r0 * w * 2, n * w * 2, rad_stream(c)) < 0) {
            rad_step_fail(c, "calibration: copying V back for its mean failed");
            return;
        }
        rad_stream_sync(rad_stream(c));
        for (int64_t r = 0; r < n; ++r)
            for (int64_t j = 0; j < w; ++j) {
                const uint32_t u = (uint32_t)host[r * w + j] << 16;
                float x;
                std::memcpy(&x, &u, 4);
                a.sum[(size_t)j] += x;
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
        VMeanMerge& m = vmean_merge();
        std::lock_guard<std::mutex> lk(m.mu);
        for (const auto& [l, a] : acc)
            for (int64_t k = 0; k < n_kv; ++k) {
                auto& slot = m.mean[{ l, kv0 + k }];
                if (!slot.empty()) continue;   /* a replica's, already in */
                slot.resize((size_t)hd);
                for (int64_t d = 0; d < hd; ++d)
                    slot[(size_t)d] = a.sum[(size_t)(k * hd + d)] / (double)a.rows;
                m.rows = std::max(m.rows, a.rows);
            }
        if (m.written || m.mean.size() != acc.size() * (size_t)n_kv_all) return;
        m.written = true;
        const std::string path = dir + "/kv_vmean.txt", tmp = path + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "w");
        bool ok = f != nullptr;
        if (f) {
            std::fprintf(f, "# radiance: each attention layer's mean of V by KV head, as the "
                            "projection writes it (unrotated); a KV recipe's 'vmean FILE'\n");
            std::fprintf(f, "kv_vmean 1 head_dim %lld kv_heads %lld rows %lld\n", (long long)hd,
                         (long long)n_kv_all, m.rows);
            for (const auto& [key, mu] : m.mean) {
                std::fprintf(f, "layer %d head %lld", key.first, (long long)key.second);
                for (double x : mu) std::fprintf(f, " %.9g", (double)(float)x);
                std::fputc('\n', f);
            }
            ok = std::fclose(f) == 0;
        }
        ok = ok && std::rename(tmp.c_str(), path.c_str()) == 0;
        std::fprintf(stderr, "radiance: calibration: rank %d %s the V means of %zu layers over %lld "
                             "tokens to %s\n", rank, ok ? "wrote" : "could not write",
                     acc.size(), m.rows, path.c_str());
    }
};

}  /* namespace rad::arch */
