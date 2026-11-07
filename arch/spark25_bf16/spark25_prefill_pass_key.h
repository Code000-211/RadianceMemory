/* Spark's prefill pass-key geometry. Keep this host-pure and mirror the h256/GQA4
 * R4D launcher exactly; unsupported batches fail closed. */
#pragma once

#include <cstdint>

#if defined(__has_include)
#  if __has_include(<rad_prefill_split.h>)
#    include <rad_prefill_split.h>
#    define SPARK25_HAVE_PREFILL_SPLIT 1
#  endif
#endif

namespace spark25_prefill_key {

constexpr int kNwarps = 24;
constexpr int kTile = 48;
constexpr int kGqa = 4;
constexpr int kHeadDim = 256;
constexpr int kBq = kNwarps * 16 / kGqa;
constexpr int kThreads = kNwarps * 32;
constexpr int kDecodeRowCap = 64;
constexpr int kPhasePrefill = 0;
constexpr int kPhaseDecode = 1;
constexpr int kPhaseMixed = 2;

static_assert(kBq == 96, "Spark GQA4 prefill BQ must match the R4D launcher");
static_assert(kThreads == 768, "Spark prefill launcher uses 768 threads");

struct Request {
    int phase = kPhasePrefill;
    int64_t n_tok = 0, n_seq = 0, max_q_len = 0, max_ctx = 0;
    int64_t n_spec = 0, draft_pass = 0;
    int enc = 0;
    int64_t head_dim = 0, gqa = 0, kv_heads = 0;
    int kv_group_found = 0, used = 0, wgps = 0;
};

#ifdef SPARK25_HAVE_PREFILL_SPLIT
inline uint64_t pass_key(const Request& r) {
    if (r.enc) return 1;
    if (r.draft_pass != 0) return 0;
    /* Mixed and speculative passes can launch attention on sub-batches whose exact geometry
     * is not represented by the whole-batch summary. */
    if (r.phase == kPhaseMixed || r.n_spec > 0) return 0;
    if (r.phase != kPhasePrefill) return 1;
    if (r.head_dim != kHeadDim || r.gqa != kGqa || r.kv_heads <= 0 || !r.kv_group_found)
        return 0;
    /* For a singleton, these equalities prove the R4D q_len band and total_q used by its launch.
     * Do not infer geometry for multi-sequence/ragged batches. */
    if (r.n_seq != 1 || r.n_tok <= 0 || r.max_q_len != r.n_tok || r.max_ctx <= 0 ||
        r.used <= 0 || r.wgps <= 0)
        return 0;
    /* The short-query path resolves to the decode launcher, with a different split law. */
    if (r.max_q_len * r.gqa <= kDecodeRowCap) return 0;

    const int q_len = (int)r.max_q_len;
    const int splits = rad_prefill_splits_for_live(kBq, kTile, q_len, (int)r.n_tok, 1,
                                                   (int)r.kv_heads, (int)r.max_ctx, r.used,
                                                   r.wgps);
    return rad_prefill_grid_key(q_len, kBq, splits, (int)r.kv_heads, 1, kThreads);
}
#else
inline uint64_t pass_key(const Request&) { return 0; }
#endif

}  // namespace spark25_prefill_key
