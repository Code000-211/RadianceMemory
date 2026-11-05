/* r3d_sample_pre_f32.hip -- the sampler's pre-filters: the five stages that edit the logits row
 * before the candidate set is built. `sample_logprobs`, `sample_penalties`, `sample_dry`,
 * `sample_ngram_ban` and `sample_mask`; the draw spine -- argmax, temp, top-k, the narrowings and
 * the pick -- is `r3d_sample_f32.hip`, and the only things shared are statics.
 *
 * The arithmetic is libref's, transcribed, and `abi/rad_sample.h` is the wire format for every
 * scalar (one 96-byte row per sampled position). Order is transcribed wherever it is visible: the
 * penalties law (sign-dependent repetition, count times frequency, presence once per distinct
 * token, all at the token's first occurrence in the window), DRY's longest-suffix repeat per
 * continuation token with its repeated-multiplication penalty, the n-gram ban's windowed prefix
 * search and whitelist exemption, and the mask's bit-per-token rule.
 *
 * ============================== stage shapes ==============================
 *
 * At the band a row is 130560 f32 and there are 2048 rows: the logits plane is 1.02 GiB, about
 * 3.6 ms a pass on gfx1102. The stages fall into two shapes by how many columns they touch, as
 * the reference does:
 *
 *   History stages (penalties, dry, ngram_ban). A window is 64 tokens, so a row touches at most 64
 *   columns: one block of 64 threads a row, a position a thread, the window read straight from L2
 *   (4096 integer reads a row). The quadratic scans cost little, so there is no LDS copy.
 *
 *   Row stages (mask, logprobs). These read or write the whole plane and are bandwidth-bound: 256
 *   threads a row, each lane a contiguous run of columns.
 *
 * `sample_logprobs` reads the row once where libref reads it three times (max, sum of
 * `exp(v - max)`, k-list). Each lane keeps a running max and a running double sum of
 * `exp(v - its own max)`, rescaling the sum when its max rises, and pairs merge by
 *
 *     m = max(a, b);   sum = a.sum * exp(a.mx - m) + b.sum * exp(b.mx - m)
 *
 * which is the reference's sum shifted by a constant per lane; it agrees to about 1e-15 relative,
 * far below the f32 store's 6e-8. The max and the k-list cross lanes by shuffle; the sum merges in
 * LDS (see sample_logprobs_kernel), checked by `r3d_lanes_check` at the end of the file, which
 * `r3d_selftest.cpp` runs. The exp is `exp(double)`, the reference's width, which makes this the
 * file's one fp64-heavy kernel (consumer RDNA3 runs doubles at 1:64); at the band most of its time
 * is the double exp, not the read.
 *
 * The k-list is per lane, k deep, in the reference's order (descending by value, ties to the lower
 * column, `topk_next`'s total order) and merged pairwise in the same butterfly. The pool is 64
 * deep; a larger k is refused, not truncated.
 *
 * ============================== agreement and declines ==============================
 *
 * Maxima use a strict `>` with the max as the first argument (`std::max`'s law), so a NaN is never
 * adopted as the row's max. A NaN in the sum propagates, as the reference's `exp(NaN - mx)` does.
 *
 * A row whose parameters turn the stage off is left bit-exact; the flag test is the reference's,
 * and for penalties so is the identity test (rep 1, freq 0, pres 0).
 *
 * Declines (RAD_E_UNSUPPORTED or a dtype refusal from the shim, none reachable from a row that
 * carries its constraints): a history plane wider than 2^31/8 columns; `k` above 64 for logprobs;
 * an `ngram` plane of the wrong dtype; mask or whitelist planes that are not 32-bit words.
 */

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <rad_sample.h>

#include <cfloat>

/* ================================================================== shared helpers (statics)
 *
 * Helpers are per-unit and internal-linkage: a non-static copy would collide at link with the same
 * helper in another unit.
 */
namespace {

constexpr uint32_t kThreads = 256;              /* 8 waves: the row stages */
constexpr uint32_t kPre     = 64;               /* 2 waves: the history stages, a position a thread */
constexpr uint32_t kWave    = 32;
constexpr uint32_t kWaves   = kThreads / kWave;
constexpr int      kKMax    = 64;               /* the k-list pool's depth, per lane */

/* The row's summary accumulates `exp` in double, as the reference's sum does. At the band the
 * double exp is most of the kernel's time; it stays, so the summary is the reference's arithmetic. */
__device__ __forceinline__ double dexp(double x) { return exp(x); }

/* A whole-register load of an index plane: the history, the n-gram pair, the bitmask words. */
__device__ __forceinline__ int32_t ldg_i32(const int32_t* p) {
    return (int32_t)ldg_u32((const uint32_t*)p);
}
__device__ __forceinline__ uint32_t ldg_u8(const uint8_t* p) { return *(const uint8_t*)p; }

/* Lane partners of the accumulator's pieces. */
template <int N>
__device__ __forceinline__ int32_t xor_i32(int32_t v) {
    return (int32_t)r3d_xor_lane_u32<N>((uint32_t)v);
}
/* No 64-bit numeric lane shuffle: see sample_logprobs_kernel. */

/* The reference's window, verbatim, clamped to the row's own row of the plane: `last_n` 0
 * disables the stage, a negative keeps the whole span, a positive keeps its tail. */
__device__ __forceinline__ bool hist_window(const RadSampleParams& sp, int32_t last_n, int64_t hmax,
                                            int64_t* ho_out, int64_t* hl_out) {
    if (last_n == 0) return false;
    int64_t hl = sp.hist_len;
    int64_t ho = sp.hist_off < 0 ? 0 : sp.hist_off;
    if (ho > hmax) ho = hmax;
    if (hl > hmax - ho) hl = hmax - ho;
    if (last_n > 0 && hl > last_n) {
        ho += hl - last_n;
        hl = last_n;
    }
    *ho_out = ho;
    *hl_out = hl;
    return hl > 0;
}

/* The reference's penalty, verbatim, including the stop short of overflow. Repeated multiplication
 * rather than powf, so every implementation lands on the same float. */
__device__ __forceinline__ float dry_penalty(float multiplier, float base, int64_t e) {
    float pen = multiplier;
    if (base == 1.0f) return pen;
    const float lim = FLT_MAX / base;
    for (int64_t i = 0; i < e; ++i) {
        if (pen > lim) break;
        pen *= base;
    }
    return pen;
}

/* The k-list's order: descending by value, ties to the lower column. `above(x, y)` means x comes
 * first (the reference's comparator, `topk_next`'s chain). */
struct Cand { float v; int32_t i; };
__device__ __forceinline__ bool above(Cand x, Cand y) {
    return x.v > y.v || (x.v == y.v && x.i < y.i);
}

/* ================================================================== sample_logprobs
 *
 * The row's summary: its max, the sum of exp(v - max) in double, and the k largest as (id, logit)
 * pairs descending, ids as floats with `vocab_off` added; an unfilled slot is id -1, logit
 * -infinity. Rows without RAD_SP_LOGPROBS are not written.
 *
 * One pass (see the file header): max, sum and k-list accumulate from the same read, and the sum is
 * rescaled into the final max at each merge.
 */
struct Acc {
    float mx;
    double sum;
    Cand c[kKMax];
    int n;
};

/* A lane's running state over its own elements. A new maximum rescales the sum so far, the cost
 * of not knowing the max up front as the reference does. */
__device__ __forceinline__ void acc_push(Acc& a, int k, uint32_t col, float v) {
    if (v > a.mx) {
        if (a.mx != -INFINITY) a.sum *= dexp((double)a.mx - (double)v);
        a.mx = v;
    }
    a.sum += dexp((double)v - (double)a.mx);
    /* The reference's insertion: an element not above the k-th is dropped; otherwise it goes in
     * the first slot it beats. A NaN beats no slot, so it lands at the tail and the list is
     * unchanged. */
    if (a.n == k && !above(Cand{ v, (int32_t)col }, a.c[a.n - 1])) return;
    int pos = 0;
    while (pos < a.n && !above(Cand{ v, (int32_t)col }, a.c[pos])) ++pos;
    if (pos >= k) return;
    for (int j = (a.n < k ? a.n : k - 1); j > pos; --j) a.c[j] = a.c[j - 1];
    a.c[pos] = Cand{ v, (int32_t)col };
    if (a.n < k) ++a.n;
}

/* The merge of two lanes' states: a strict max, the sum of both shifted sums, and the lists merged
 * in the same order. Equal values at equal columns cannot occur (each column belongs to one lane),
 * and a NaN entry is above nothing. */
__device__ __forceinline__ Acc acc_merge(const Acc& a, const Acc& b, int k) {
    Acc r;
    r.mx = a.mx > b.mx ? a.mx : b.mx;
    r.sum = (a.mx == -INFINITY ? 0.0 : a.sum * dexp((double)a.mx - (double)r.mx)) +
            (b.mx == -INFINITY ? 0.0 : b.sum * dexp((double)b.mx - (double)r.mx));
    int i = 0, j = 0;
    r.n = 0;
    while (r.n < k && (i < a.n || j < b.n)) {
        const bool take_b = (i >= a.n) || (j < b.n && above(b.c[j], a.c[i]));
        r.c[r.n++] = take_b ? b.c[j++] : a.c[i++];
    }
    return r;
}

template <int THR>
__global__ __launch_bounds__(THR) void sample_logprobs_kernel(
    const float* __restrict__ lg, const RadSampleParams* __restrict__ pr,
    float* __restrict__ st, uint32_t n, uint32_t k, uint32_t lg_ld, uint32_t st_ld,
    int32_t voff) {
    const uint32_t row = blockIdx.x;
    if (pr && !(pr[row].flags & RAD_SP_LOGPROBS)) return;   /* no params: every row */

    Acc a;
    a.mx = -INFINITY;
    a.sum = 0.0;
    a.n = 0;
    const float* __restrict__ lr = lg + (size_t)row * lg_ld;
    for (uint32_t c = threadIdx.x; c < n; c += THR) acc_push(a, (int)k, c, ldg_f32(lr + c));

    /* The k-list and the max meet their partners by lane shuffle, five rounds. The shuffle mask
     * on wave32 is `0xFFFFFFFFull`; `~0ull` would name lanes 32..63, which do not exist.
     *
     * The sum does not cross lanes; it merges in LDS below. A double can cross lanes only as its
     * bits: `(unsigned long long)d` converts by value and truncates the partner's term to its
     * integer part (see lanes_check_kernel). The k-list entries are bit patterns (a float's
     * pattern and an int packed into 8 bytes), so they cross as one 64-bit shuffle. */
#define R3D_ACC_XOR(M)                                                                           \
    {                                                                                            \
        Acc b;                                                                                   \
        b.mx = r3d_xor_lane<M>(a.mx);                                                            \
        b.n = xor_i32<M>((int32_t)a.n);                                                          \
        for (int j = 0; j < k; ++j) {                                                            \
            /* One entry, a float's pattern and a column, crosses lanes as one 8-byte pattern. */ \
            const unsigned long long packed =                                                   \
                (unsigned long long)__float_as_uint(a.c[j].v) |                                  \
                ((unsigned long long)(uint32_t)a.c[j].i << 32);                                  \
            const unsigned long long got = __shfl_xor_sync(0xFFFFFFFFull, packed, M);            \
            b.c[j] = Cand{ __uint_as_float((uint32_t)got), (int32_t)(uint32_t)(got >> 32) };      \
        }                                                                                        \
        b.sum = 0.0;                                                                             \
        const float mx = a.mx > b.mx ? a.mx : b.mx;   /* no lane's max is a NaN (strict test) */ \
        a = acc_merge(a, b, (int)k);                                                             \
        a.mx = mx;                                                                               \
    }
    R3D_ACC_XOR(16)
    R3D_ACC_XOR(8)
    R3D_ACC_XOR(4)
    R3D_ACC_XOR(2)
    R3D_ACC_XOR(1)
#undef R3D_ACC_XOR

    /* The wave's sum in LDS: the wave's max, then every lane's sum shifted into it, in lane order.
     * An empty lane (a row narrower than the block) has max -infinity and sum 0 and contributes
     * nothing. */
    __shared__ float s_mx[kThreads];
    __shared__ double s_sum[kThreads];
    __shared__ float w_mx[kWaves];
    __shared__ double w_sum[kWaves];
    __shared__ int w_n[kWaves];
    __shared__ Cand w_c[kWaves][kKMax];
    s_mx[threadIdx.x] = a.mx;
    s_sum[threadIdx.x] = a.sum;
    __syncthreads();

    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) {
        const uint32_t b = wave * kWave;
        float m = s_mx[b];
        for (uint32_t l = 1; l < kWave; ++l)
            if (s_mx[b + l] > m) m = s_mx[b + l];
        double s = 0.0;
        for (uint32_t l = 0; l < kWave; ++l)
            s += s_mx[b + l] == -INFINITY ? 0.0
                                          : s_sum[b + l] * dexp((double)s_mx[b + l] - (double)m);
        w_mx[wave] = m;
        w_sum[wave] = s;
        w_n[wave] = a.n;
        for (int j = 0; j < k; ++j) w_c[wave][j] = a.c[j];
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    /* The eight wave partials, in wave order: the row's max, the row's sum, the row's k-list. */
    float m = w_mx[0];
    double s = w_sum[0];
    int n_list = w_n[0];
    Cand list[kKMax];
    for (int j = 0; j < k; ++j) list[j] = w_c[0][j];
    for (uint32_t w = 1; w < kWaves; ++w) {
        const float mm = w_mx[w] > m ? w_mx[w] : m;
        s = (m == -INFINITY ? 0.0 : s * dexp((double)m - (double)mm)) +
            (w_mx[w] == -INFINITY ? 0.0 : w_sum[w] * dexp((double)w_mx[w] - (double)mm));
        m = mm;
        Acc x, y;
        x.mx = m; x.sum = 0.0; x.n = n_list;
        for (int j = 0; j < k; ++j) x.c[j] = list[j];
        y.mx = w_mx[w]; y.sum = 0.0; y.n = w_n[w];
        for (int j = 0; j < k; ++j) y.c[j] = w_c[w][j];
        const Acc t = acc_merge(x, y, (int)k);
        n_list = t.n;
        for (int j = 0; j < k; ++j) list[j] = t.c[j];
    }

    float* __restrict__ sr = st + (size_t)row * st_ld;
    sr[0] = m;
    sr[1] = (float)s;
    for (uint32_t j = 0; j < k; ++j) {
        const bool have_j = j < (uint32_t)n_list;
        sr[2 + 2 * j] = have_j ? (float)(list[j].i + voff) : -1.0f;
        sr[3 + 2 * j] = have_j ? list[j].v : -INFINITY;
    }
}

/* ================================================================== sample_penalties
 *
 * llama.cpp's ordering, applied at the token's first occurrence in the window:
 *
 *     repetition  logit = logit > 0 ? logit / rep : logit * rep
 *     frequency   logit -= count * freq
 *     presence    logit -= pres
 *
 * The repetition penalty is sign-dependent because dividing a negative logit would raise it. The
 * count carries the multiplicity, so the update runs once per distinct token; the scan is the
 * reference's: walk the window from the front, stop at an earlier occurrence, count the rest.
 *
 * Columns are global ids less `vocab_off`, so a token on another rank's shard is skipped here and
 * penalised there. Each distinct token is one column and one thread, so the writes cannot race.
 */
__global__ __launch_bounds__(kPre) void sample_penalties_kernel(
    float* __restrict__ lg, const RadSampleParams* __restrict__ pr, const int32_t* __restrict__ hist,
    uint32_t n, uint32_t lg_ld, uint32_t hmax, int32_t voff) {
    const uint32_t row = blockIdx.x;
    const RadSampleParams& sp = pr[row];
    if (!(sp.flags & RAD_SP_PENALTY)) return;
    const float rep = sp.rep_penalty, freq = sp.freq_penalty, pres = sp.pres_penalty;
    if (rep == 1.0f && freq == 0.0f && pres == 0.0f) return;

    int64_t ho = 0, hl = 0;
    if (!hist_window(sp, sp.penalty_last_n, (int64_t)hmax, &ho, &hl)) return;

    const int32_t* __restrict__ hr = hist + (size_t)row * hmax + ho;
    float* __restrict__ lr = lg + (size_t)row * lg_ld;
    for (uint32_t i = threadIdx.x; i < (uint32_t)hl; i += kPre) {
        const int32_t tok = ldg_i32(hr + i);
        const int64_t col = (int64_t)tok - voff;
        if (tok < 0 || col < 0 || col >= n) continue;
        bool first = true;
        int64_t count = 0;
        for (int64_t j = 0; j < hl; ++j) {
            if (ldg_i32(hr + j) != tok) continue;
            if (j < (int64_t)i) { first = false; break; }
            ++count;
        }
        if (!first) continue;
        float v = lr[col];
        v = v > 0.0f ? v / rep : v * rep;
        v -= (float)count * freq;
        v -= pres;
        lr[col] = v;
    }
}

/* ================================================================== sample_dry
 *
 * DRY (llama.cpp's llama_sampler_dry_apply): for every earlier position j, n_j is how long a suffix
 * of the window also ends at j, capped at the repeat limit. The token at j+1 would extend that
 * repeat and is penalised once by multiplier * base^(n - allowed), using the longest repeat it
 * follows. A breaker token is never penalised, and only lengths >= `allowed` count.
 *
 * A thread a position, the window read from L2. The reference's per-token map becomes a maximum
 * over the threads sharing a token, applied by the thread at the token's first occurrence, so each
 * column is written once.
 */
__global__ __launch_bounds__(kPre) void sample_dry_kernel(
    float* __restrict__ lg, const RadSampleParams* __restrict__ pr, const int32_t* __restrict__ hist,
    const uint8_t* __restrict__ brk, uint32_t n, uint32_t lg_ld, uint32_t hmax, int32_t voff) {
    const uint32_t row = blockIdx.x;
    const RadSampleParams& sp = pr[row];
    if (!(sp.flags & RAD_SP_DRY) || !(sp.dry_multiplier > 0.0f)) return;
    const int64_t allowed = sp.dry_allowed_length > 0 ? sp.dry_allowed_length : 2;

    int64_t ho = 0, hl = 0;
    if (!hist_window(sp, sp.dry_penalty_last_n, (int64_t)hmax, &ho, &hl)) return;
    if (hl <= allowed) return;
    const int64_t cap = sp.dry_rep_limit > 0 && sp.dry_rep_limit < hl ? sp.dry_rep_limit : hl;
    if (cap < allowed) return;

    const int32_t* __restrict__ hr = hist + (size_t)row * hmax + ho;
    const uint8_t* __restrict__ br = brk ? brk + (size_t)row * hmax + ho : nullptr;
    float* __restrict__ lr = lg + (size_t)row * lg_ld;
    const auto tok_at = [&](int64_t i) { return ldg_i32(hr + i); };
    const auto exempt = [&](int64_t i) { return br && ldg_u8(br + i) != 0u; };
    const float base = sp.dry_base > 0.0f ? sp.dry_base : 1.75f;

    for (uint32_t j = threadIdx.x; j + 1u < (uint32_t)hl; j += kPre) {
        const int64_t lim = ((int64_t)j + 1 < cap) ? (int64_t)j + 1 : cap;
        int64_t m = 0;
        while (m < lim && tok_at(j - m) == tok_at(hl - 1 - m)) ++m;
        if (m < allowed || exempt((int64_t)j + 1)) continue;
        const int32_t tok = tok_at((int64_t)j + 1);
        const int64_t col = (int64_t)tok - voff;
        if (tok < 0 || col < 0 || col >= n) continue;
        /* The longest repeat any position attributes to this token; the lowest qualifying j is
         * the one writer. */
        int64_t longest = m;
        bool first = true;
        for (int64_t q = 0; q + 1 < hl; ++q) {
            if (tok_at(q + 1) != tok) continue;
            const int64_t limq = (q + 1 < cap) ? q + 1 : cap;
            int64_t mq = 0;
            while (mq < limq && tok_at(q - mq) == tok_at(hl - 1 - mq)) ++mq;
            if (mq < allowed || exempt(q + 1)) continue;
            if (mq > longest) longest = mq;
            if (q < (int64_t)j) { first = false; break; }
        }
        if (!first) continue;
        const float v = lr[col];
        lr[col] = v - dry_penalty(sp.dry_multiplier, base, longest - allowed);
    }
}

/* ================================================================== sample_ngram_ban
 *
 * The windowed no-repeat-ngram ban. With the row's window s[0, hl):
 *
 *     search_start = max(0, hl - w);  search_end = hl - n + 1;  prefix = s[hl - n + 1, hl)
 *     for idx in [search_start, search_end): s[idx, idx + n - 1) == prefix  =>  ban s[idx + n - 1]
 *
 * The window bounds the start index, so an n-gram straddling the window's edge is searched. A
 * banned token's logit becomes -infinity; whitelisted columns are exempt and never touched. n == 1
 * bans every token in the window (empty prefix). A row with n 0 is left unchanged.
 *
 * A thread a search index. Several indices may ban the same token, but all write the same
 * -infinity.
 */
__global__ __launch_bounds__(kPre) void sample_ngram_ban_kernel(
    float* __restrict__ lg, const RadSampleParams* __restrict__ pr, const int32_t* __restrict__ hist,
    const int32_t* __restrict__ ngram, const int32_t* __restrict__ wl, uint32_t n, uint32_t lg_ld,
    uint32_t hmax, uint32_t words, int32_t voff) {
    const uint32_t row = blockIdx.x;
    const RadSampleParams& sp = pr[row];
    const int32_t* __restrict__ nr = ngram + (size_t)row * 2;
    const int64_t nn = ldg_i32(nr);
    const int64_t win_req = ldg_i32(nr + 1);
    if (nn <= 0) return;

    int64_t ho = 0, hl = 0;
    if (!hist_window(sp, -1, (int64_t)hmax, &ho, &hl)) return;   /* the whole staged span */
    if (hl < nn) return;
    const int64_t win = (win_req > 0 && win_req < hl) ? win_req : hl;
    const int64_t search_start = hl - win;
    const int64_t search_end = hl - nn + 1;
    if (search_end <= search_start) return;

    const int32_t* __restrict__ hr = hist + (size_t)row * hmax + ho;
    const int32_t* __restrict__ wr = wl ? wl + (size_t)row * words : nullptr;
    float* __restrict__ lr = lg + (size_t)row * lg_ld;
    const auto tok_at = [&](int64_t i) { return ldg_i32(hr + i); };
    const auto exempt = [&](int64_t col) {
        if (!wr) return false;
        const uint32_t word = (uint32_t)ldg_i32(wr + col / 32);
        return ((word >> (uint32_t)(col % 32)) & 1u) != 0u;
    };
    const int64_t plen = nn - 1, p0 = hl - plen;

    for (uint32_t idx = search_start + threadIdx.x; idx < (uint32_t)search_end; idx += kPre) {
        bool match = true;
        for (int64_t j = 0; j < plen; ++j) {
            if (tok_at((int64_t)idx + j) != tok_at(p0 + j)) { match = false; break; }
        }
        if (!match) continue;
        const int32_t tok = tok_at((int64_t)idx + plen);
        const int64_t col = (int64_t)tok - voff;
        if (tok < 0 || col < 0 || col >= n) continue;
        if (exempt(col)) continue;
        lr[col] = -INFINITY;
    }
}

/* ================================================================== sample_mask
 *
 * A set bit means allowed; an unset column's logit becomes -infinity. A thread takes one bitmask
 * word and its 32 columns, a contiguous 128-byte segment. An all-ones word is skipped; an all-zeros
 * word is a straight run of stores.
 *
 * `mask_row` selects the row of the bitmask plane; as in the reference, a row index outside the
 * plane (-1 included) leaves the row unmasked.
 */
__global__ __launch_bounds__(kThreads) void sample_mask_kernel(
    float* __restrict__ lg, const RadSampleParams* __restrict__ pr, const int32_t* __restrict__ bm,
    uint32_t n, uint32_t lg_ld, uint32_t words, uint32_t mask_rows) {
    const uint32_t row = blockIdx.x;
    const RadSampleParams& sp = pr[row];
    if (!(sp.flags & RAD_SP_MASK)) return;
    const int64_t mrow = sp.mask_row;
    if (mrow < 0 || (uint64_t)mrow >= mask_rows) return;

    const int32_t* __restrict__ br = bm + (size_t)mrow * words;
    float* __restrict__ lr = lg + (size_t)row * lg_ld;
    for (uint32_t w = threadIdx.x; w < words; w += kThreads) {
        const uint32_t bits = (uint32_t)ldg_i32(br + w);
        if (bits == 0xffffffffu) continue;                 /* nothing banned in these 32 */
        const uint32_t c0 = 32u * w;
        const uint32_t cnt = n - c0 < 32u ? n - c0 : 32u;  /* the row's last word is partial */
        if (bits == 0u) {
            for (uint32_t c = 0; c < cnt; ++c) lr[c0 + c] = -INFINITY;
            continue;
        }
        for (uint32_t c = 0; c < cnt; ++c)
            if (((bits >> c) & 1u) == 0u) lr[c0 + c] = -INFINITY;
    }
}

/* ------------------------------------------------------------------ shim helpers */

static inline int sp_is_word(const RadTensor* t) {
    return t && (t->dtype == RAD_U32 || t->dtype == RAD_I32 || t->dtype == RAD_F32);
}

/* The params plane: [M, 24] words. The dtype test matters: a 16-bit view passes a byte-count check
 * and then misreads every row. */
static inline int params_ok(const RadTensor* t, long long M) {
    if (!t || !t->data) return RAD_E_INVAL;
    if (!sp_is_word(t)) return RAD_E_DTYPE;
    if (r3d_numel(t) * 4 < M * (long long)sizeof(RadSampleParams)) return RAD_E_SHAPE;
    if (!r3d_contig(t)) return RAD_E_STRIDE;
    if (!r3d_aligned(r3d_p(t), 8)) return RAD_E_ALIGN;      /* seed and pos are 8-byte fields */
    return RAD_OK;
}

static inline int logits_ok(const RadTensor* t) {
    if (!t || !t->data) return RAD_E_INVAL;
    if (!r3d_is_f32(t)) return RAD_E_DTYPE;
    if (!r3d_aligned(r3d_p(t), 4)) return RAD_E_ALIGN;
    return RAD_OK;
}

/* The history plane: i32 [M, max_hist], contiguous, its width the reference's `hmax`. */
static inline int hist_ok(const RadTensor* t, long long M, long long* hmax) {
    if (!t || !t->data) return RAD_E_INVAL;
    if (!r3d_is_i32(t)) return RAD_E_DTYPE;
    if (!r3d_contig(t)) return RAD_E_STRIDE;
    if (!r3d_aligned(r3d_p(t), 4)) return RAD_E_ALIGN;
    const long long w = r3d_numel(t) / M;
    if (w <= 0 || w > 0x7fffffffLL) return RAD_E_SHAPE;
    *hmax = w;
    return RAD_OK;
}

static inline int r3d_launch_status(int rc) {
    if (rc == 0) return RAD_OK;
    switch (rc) {
        case -1: return RAD_E_INVAL;
        case -2: return RAD_E_SHAPE;
        case -3: return RAD_E_STRIDE;
        default: return RAD_E_DEVICE;
    }
}

/* The logits row count, from the plane and never from the band's M. */
static inline int logits_rows(const RadArgs* a, const RadTensor* lg, long long* n_vocab,
                              long long* rows, long long* lg_ld) {
    /* As in the reference: the width from the band, else the plane's last dimension; a rank-1
     * plane has none, which is E_SHAPE. */
    *n_vocab = r3d_geti_or(a, "n_vocab",
                           lg->rank >= 2 ? r3d_dim(lg, (int)lg->rank - 1) : 0);
    if (*n_vocab <= 0 || *n_vocab > 0x7fffffffLL) return RAD_E_SHAPE;
    *lg_ld = r3d_rows_pitch(lg, *n_vocab, rows);
    if (*lg_ld < 0) return RAD_E_STRIDE;
    if (*rows <= 0 || *rows > 0x7fffffffLL) return RAD_E_SHAPE;
    return RAD_OK;
}

}  /* namespace */

#define R3D_NEED(x) \
    do { const int _s = (x); if (_s != RAD_OK) return R3D_NO(_s); } while (0)

/* ================================================================== launchers
 *
 *   lg       [M][n_vocab] f32, INOUT for the four editing stages, read-only for logprobs
 *   params   [M] 24 words a row, the per-row scalars
 *   hist     [M][max_hist] i32, global token ids, `hist_off`/`hist_len` windowing it
 *   stats    [M][2 + 2k] f32
 *   bitmask  [rows][(n_vocab+31)/32] i32, a set bit meaning allowed
 *
 * 0, -1 null, -2 bad extent, -3 pitch, -4 the HIP launch failed. No allocation, no synchronisation.
 */

extern "C" int r3d_sample_logprobs(long lg, long params, long stats, int M, int n_vocab, int k,
                                  long long lg_ld, long long stats_ld, int vocab_off,
                                  long stream) {
    if (!lg || !stats) return -1;
    if (M <= 0 || n_vocab <= 0 || k <= 0) return -2;
    if (k > kKMax) return -2;
    if (!lg_ld) lg_ld = n_vocab;
    if (!stats_ld) stats_ld = 2 + 2 * (long long)k;
    if (lg_ld < n_vocab || stats_ld < 2 + 2 * (long long)k) return -3;
    R3D_KLAUNCH(R3D_KNAME(sample_logprobs_kernel<kThreads>), dim3((uint32_t)M), dim3(kThreads),
                (hipStream_t)(uintptr_t)stream, (const float*)(uintptr_t)lg,
                (const RadSampleParams*)(uintptr_t)params, (float*)(uintptr_t)stats,
                (uint32_t)n_vocab, (uint32_t)k, (uint32_t)lg_ld, (uint32_t)stats_ld,
                (int32_t)vocab_off);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_sample_penalties(long lg, long params, long hist, int M, int n_vocab,
                                   long long lg_ld, long long hist_ld, int vocab_off,
                                   long stream) {
    if (!lg || !params || !hist) return -1;
    if (M <= 0 || n_vocab <= 0) return -2;
    if (!lg_ld) lg_ld = n_vocab;
    if (!hist_ld) return -2;                       /* the window is clamped to the plane's width */
    if (lg_ld < n_vocab || hist_ld <= 0) return -3;
    R3D_KLAUNCH(R3D_KNAME(sample_penalties_kernel), dim3((uint32_t)M), dim3(kPre),
                (hipStream_t)(uintptr_t)stream, (float*)(uintptr_t)lg,
                (const RadSampleParams*)(uintptr_t)params, (const int32_t*)(uintptr_t)hist,
                (uint32_t)n_vocab, (uint32_t)lg_ld, (uint32_t)hist_ld, (int32_t)vocab_off);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_sample_dry(long lg, long params, long hist, long breakers, int M, int n_vocab,
                             long long lg_ld, long long hist_ld, int vocab_off, long stream) {
    if (!lg || !params || !hist) return -1;
    if (M <= 0 || n_vocab <= 0) return -2;
    if (!lg_ld) lg_ld = n_vocab;
    if (!hist_ld) return -2;
    if (lg_ld < n_vocab || hist_ld <= 0) return -3;
    R3D_KLAUNCH(R3D_KNAME(sample_dry_kernel), dim3((uint32_t)M), dim3(kPre),
                (hipStream_t)(uintptr_t)stream, (float*)(uintptr_t)lg,
                (const RadSampleParams*)(uintptr_t)params, (const int32_t*)(uintptr_t)hist,
                (const uint8_t*)(uintptr_t)breakers, (uint32_t)n_vocab, (uint32_t)lg_ld,
                (uint32_t)hist_ld, (int32_t)vocab_off);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_sample_ngram_ban(long lg, long params, long hist, long ngram, long whitelist,
                                   int M, int n_vocab, long long lg_ld, long long hist_ld,
                                   int vocab_off, long stream) {
    if (!lg || !params || !hist || !ngram) return -1;
    if (M <= 0 || n_vocab <= 0) return -2;
    if (!lg_ld) lg_ld = n_vocab;
    if (!hist_ld) return -2;
    if (lg_ld < n_vocab || hist_ld <= 0) return -3;
    const uint32_t words = (uint32_t)((n_vocab + 31) / 32);
    R3D_KLAUNCH(R3D_KNAME(sample_ngram_ban_kernel), dim3((uint32_t)M), dim3(kPre),
                (hipStream_t)(uintptr_t)stream, (float*)(uintptr_t)lg,
                (const RadSampleParams*)(uintptr_t)params, (const int32_t*)(uintptr_t)hist,
                (const int32_t*)(uintptr_t)ngram, (const int32_t*)(uintptr_t)whitelist,
                (uint32_t)n_vocab, (uint32_t)lg_ld, (uint32_t)hist_ld, words, (int32_t)vocab_off);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_sample_mask(long lg, long params, long bitmask, int M, int n_vocab,
                              long long lg_ld, int mask_rows, long stream) {
    if (!lg || !params || !bitmask) return -1;
    if (M <= 0 || n_vocab <= 0) return -2;
    if (!lg_ld) lg_ld = n_vocab;
    if (lg_ld < n_vocab) return -3;
    if (mask_rows < M) return -3;   /* the plane must hold at least M rows */
    const uint32_t words = (uint32_t)((n_vocab + 31) / 32);
    /* mask_rows is the bitmask plane's own row count, the bound the reference tests mask_row
     * against; it may exceed M when a caller passes a larger plane. The grid stays M, the logits
     * row count. */
    R3D_KLAUNCH(R3D_KNAME(sample_mask_kernel), dim3((uint32_t)M), dim3(kThreads),
                (hipStream_t)(uintptr_t)stream, (float*)(uintptr_t)lg,
                (const RadSampleParams*)(uintptr_t)params, (const int32_t*)(uintptr_t)bitmask,
                (uint32_t)n_vocab, (uint32_t)lg_ld, words, (uint32_t)mask_rows);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ================================================================== shims
 *
 * libref's operand lists: {logits, params, history[, breakers]} for the history stages,
 * {logits, params, ngram, whitelist?} for the ban, {logits, params, bitmask} for the mask and
 * {logits, params, stats} for logprobs. M is the logits plane's row count.
 */

extern "C" int r3d_rad_sample_logprobs(const RadArgs* a, RadStream stream) {
    const RadTensor* lg = r3d_opt(a, 0);
    const RadTensor* pr = r3d_opt(a, 1);
    const RadTensor* st = r3d_opt(a, 2);
    R3D_NEED(logits_ok(lg));
    const long long k = r3d_geti_or(a, "k", 0);
    if (k <= 0) return R3D_NO(RAD_E_SHAPE);
    /* The pool is 64 deep per lane; the row's C_LE("k", 64) keeps larger requests out of the band. */
    if (k > kKMax) return R3D_NO(RAD_E_UNSUPPORTED);
    long long n_vocab = 0, rows = 0, lg_ld = 0;
    R3D_NEED(logits_rows(a, lg, &n_vocab, &rows, &lg_ld));
    R3D_NEED(params_ok(pr, rows));
    if (!st || !st->data) return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_f32(st)) return R3D_NO(RAD_E_DTYPE);
    long long srows = 0;
    const long long st_ld = r3d_rows_pitch(st, 2 + 2 * k, &srows);
    if (st_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (srows < rows) return R3D_NO(RAD_E_SHAPE);
    return r3d_launch_status(r3d_sample_logprobs(
        r3d_addr(lg), r3d_addr(pr), r3d_addr(st), (int)rows, (int)n_vocab, (int)k, lg_ld, st_ld,
        (int)r3d_geti_or(a, "vocab_off", 0), (long)(uintptr_t)stream));
}

extern "C" int r3d_rad_sample_penalties(const RadArgs* a, RadStream stream) {
    const RadTensor* lg = r3d_opt(a, 0);
    const RadTensor* pr = r3d_opt(a, 1);
    const RadTensor* hi = r3d_opt(a, 2);
    R3D_NEED(logits_ok(lg));
    long long n_vocab = 0, rows = 0, lg_ld = 0;
    R3D_NEED(logits_rows(a, lg, &n_vocab, &rows, &lg_ld));
    R3D_NEED(params_ok(pr, rows));
    long long hmax = 0;
    R3D_NEED(hist_ok(hi, rows, &hmax));
    return r3d_launch_status(r3d_sample_penalties(
        r3d_addr(lg), r3d_addr(pr), r3d_addr(hi), (int)rows, (int)n_vocab, lg_ld, hmax,
        (int)r3d_geti_or(a, "vocab_off", 0), (long)(uintptr_t)stream));
}

extern "C" int r3d_rad_sample_dry(const RadArgs* a, RadStream stream) {
    const RadTensor* lg = r3d_opt(a, 0);
    const RadTensor* pr = r3d_opt(a, 1);
    const RadTensor* hi = r3d_opt(a, 2);
    const RadTensor* brk = r3d_opt(a, 3);
    R3D_NEED(logits_ok(lg));
    long long n_vocab = 0, rows = 0, lg_ld = 0;
    R3D_NEED(logits_rows(a, lg, &n_vocab, &rows, &lg_ld));
    R3D_NEED(params_ok(pr, rows));
    long long hmax = 0;
    R3D_NEED(hist_ok(hi, rows, &hmax));
    long long baddr = 0;
    if (brk && brk->data) {
        /* One exemption byte per history position. */
        if (!r3d_is_u8(brk)) return R3D_NO(RAD_E_DTYPE);
        if (!r3d_contig(brk)) return R3D_NO(RAD_E_STRIDE);
        if (r3d_numel(brk) < rows * hmax) return R3D_NO(RAD_E_SHAPE);
        baddr = r3d_addr(brk);
    }
    return r3d_launch_status(r3d_sample_dry(
        r3d_addr(lg), r3d_addr(pr), r3d_addr(hi), baddr, (int)rows, (int)n_vocab, lg_ld, hmax,
        (int)r3d_geti_or(a, "vocab_off", 0), (long)(uintptr_t)stream));
}

extern "C" int r3d_rad_sample_ngram_ban(const RadArgs* a, RadStream stream) {
    const RadTensor* lg = r3d_opt(a, 0);
    const RadTensor* pr = r3d_opt(a, 1);
    const RadTensor* hi = r3d_opt(a, 2);
    const RadTensor* ng = r3d_opt(a, 3);
    const RadTensor* wl = r3d_opt(a, 4);
    R3D_NEED(logits_ok(lg));
    long long n_vocab = 0, rows = 0, lg_ld = 0;
    R3D_NEED(logits_rows(a, lg, &n_vocab, &rows, &lg_ld));
    R3D_NEED(params_ok(pr, rows));
    long long hmax = 0;
    R3D_NEED(hist_ok(hi, rows, &hmax));
    if (!ng || !ng->data) return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_i32(ng)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_contig(ng) || r3d_numel(ng) < 2 * rows) return R3D_NO(RAD_E_SHAPE);
    const long long words = (n_vocab + 31) / 32;
    long long waddr = 0;
    if (wl && wl->data) {
        if (wl->dtype != RAD_I32 && wl->dtype != RAD_U32) return R3D_NO(RAD_E_DTYPE);
        if (!r3d_contig(wl)) return R3D_NO(RAD_E_STRIDE);
        if (r3d_numel(wl) < rows * words) return R3D_NO(RAD_E_SHAPE);
        waddr = r3d_addr(wl);
    }
    return r3d_launch_status(r3d_sample_ngram_ban(
        r3d_addr(lg), r3d_addr(pr), r3d_addr(hi), r3d_addr(ng), waddr, (int)rows, (int)n_vocab,
        lg_ld, hmax, (int)r3d_geti_or(a, "vocab_off", 0), (long)(uintptr_t)stream));
}

extern "C" int r3d_rad_sample_mask(const RadArgs* a, RadStream stream) {
    const RadTensor* lg = r3d_opt(a, 0);
    const RadTensor* pr = r3d_opt(a, 1);
    const RadTensor* bm = r3d_opt(a, 2);
    R3D_NEED(logits_ok(lg));
    long long n_vocab = 0, rows = 0, lg_ld = 0;
    R3D_NEED(logits_rows(a, lg, &n_vocab, &rows, &lg_ld));
    R3D_NEED(params_ok(pr, rows));
    if (!bm || !bm->data) return R3D_NO(RAD_E_INVAL);
    if (bm->dtype != RAD_I32 && bm->dtype != RAD_U32) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_contig(bm)) return R3D_NO(RAD_E_STRIDE);
    const long long words = (n_vocab + 31) / 32;
    if (words <= 0 || r3d_numel(bm) < rows * words) return R3D_NO(RAD_E_SHAPE);
    /* The plane's own row count is the bound the reference tests; it may exceed the logits'. */
    const long long mask_rows = r3d_numel(bm) / words;
    return r3d_launch_status(r3d_sample_mask(r3d_addr(lg), r3d_addr(pr), r3d_addr(bm), (int)rows,
                                            (int)n_vocab, lg_ld, (int)mask_rows,
                                            (long)(uintptr_t)stream));
}

/* ================================================================== the lane helpers, proved
 *
 * Every wave-sum in libr3d is a butterfly over lane shuffles, and an error in a lane helper (say, a
 * stale half of a 64-bit value) is invisible to a codec sweep. The logprobs k-list crosses lanes as
 * 8-byte values, so this file carries the on-device control for the helpers: one entry per lane of
 * all 8 waves, seeds at wave-local lanes 6, 9, 19 and 25, five butterfly rounds, over
 * `r3d_xor_lane_u64` (two 32-bit shuffles), and one 64-bit shuffle.
 *
 * The form that must not be written converts by value:
 *
 *     d += (double)r3d_xor_lane_u64<N>((unsigned long long)d);      // wrong
 *
 * The partner's term arrives truncated to its integer part, so a wave of fractions sums to the
 * largest integer part; the error vanishes for integer terms. The correct form moves bits
 * (`bits_of` / `d_from_bits` below), as the k-list's shuffle does. The control catches both traps:
 * summing an IEEE pattern as an integer, and converting a pattern to a value.
 */
__device__ __forceinline__ double d_from_bits(unsigned long long b) {
    double d;
    __builtin_memcpy(&d, &b, sizeof d);
    return d;
}
__device__ __forceinline__ unsigned long long bits_of(double d) {
    unsigned long long b;
    __builtin_memcpy(&b, &d, sizeof b);
    return b;
}

__global__ __launch_bounds__(256) void lanes_check_kernel(double* __restrict__ out) {
    /* kThreads and kWave are ambiguous outside the anonymous namespace, so the control spells its
     * own geometry. */
    constexpr uint32_t kLWave = 32, kLBlock = 256;
    /* Four non-zero lanes per wave, expected sum 0.08945 + 1 + 0.5971 + 0.9768 in the butterfly's
     * order; every wave carries them. */
    const uint32_t lane = threadIdx.x & (kLWave - 1);
    const double seed = (lane == 6) ? 0.08945 : (lane == 9) ? 1.0 : (lane == 19) ? 0.5971
                        : (lane == 25) ? 0.9768 : 0.0;
    __shared__ double s_d[kLBlock];
    __shared__ float  s_f[kLBlock];
    s_d[threadIdx.x] = seed;
    s_f[threadIdx.x] = (float)seed;
    __syncthreads();

    /* Form 0: the 32-bit butterfly, over a value read from LDS so it is not a constant. */
    float f = s_f[threadIdx.x];
    f += r3d_xor_lane<16>(f);
    f += r3d_xor_lane<8>(f);
    f += r3d_xor_lane<4>(f);
    f += r3d_xor_lane<2>(f);
    f += r3d_xor_lane<1>(f);
    out[0 * kLBlock + threadIdx.x] = (double)f;

    /* Form 1: the halves form over a value derived from the lane index, passed as bits. */
    double d = seed;
#define R3D_L64(N) d += d_from_bits(r3d_xor_lane_u64<N>(bits_of(d)));
    R3D_L64(16) R3D_L64(8) R3D_L64(4) R3D_L64(2) R3D_L64(1)
#undef R3D_L64
    out[1 * kLBlock + threadIdx.x] = d;

    /* Form 2: the halves form over a value loaded from LDS, as an accumulator is. */
    d = s_d[threadIdx.x];
#define R3D_L64B(N) d += d_from_bits(r3d_xor_lane_u64<N>(bits_of(d)));
    R3D_L64B(16) R3D_L64B(8) R3D_L64B(4) R3D_L64B(2) R3D_L64B(1)
#undef R3D_L64B
    out[2 * kLBlock + threadIdx.x] = d;

    /* Form 3: one 64-bit shuffle, the k-list's form; the mask names wave32's 32 lanes. */
    d = s_d[threadIdx.x];
#define R3D_S64(N) d += d_from_bits(__shfl_xor_sync(0xFFFFFFFFull, bits_of(d), N));
    R3D_S64(16) R3D_S64(8) R3D_S64(4) R3D_S64(2) R3D_S64(1)
#undef R3D_S64
    out[3 * kLBlock + threadIdx.x] = d;
}

/* Writes four forms x 256 lanes of doubles (8 KiB, caller-owned). 0, -1 null, -4 the launch
 * failed. r3d_selftest.cpp resolves this by name and compares every lane against a host emulation
 * of the same five rounds. */
extern "C" int r3d_lanes_check(long out, long stream) {
    if (!out) return -1;
    R3D_KLAUNCH(R3D_KNAME(lanes_check_kernel), dim3(1), dim3(256),
                (hipStream_t)(uintptr_t)stream, (double*)(uintptr_t)out);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}
