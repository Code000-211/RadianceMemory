/* r3d_moe_scatter.hip -- moe_scatter: the routed slots' stable counting sort.
 *
 * One workgroup does the whole sort (the "small" form, T <= 1024, which covers every decode and
 * verify step): histogram in LDS, offsets an exclusive prefix over n_expert <= 512 counts, and
 * each slot placed by its rank inside its expert's run. Stable and byte-identical to the
 * reference: a slot's place is its position among the same expert's slots in ascending i.
 *
 *   expert_ids [T]  ->  sorted_tok [T], expert_offset [n_expert+1], expert_count [n_expert]
 *
 * `expert_base` is this rank's first expert, subtracted from every routed id so the offsets,
 * counts and (downstream) weight table are indexed locally; an id outside the slice is dropped
 * (sorted to the tail as -1). `protect` is a comma list of local expert indices placed last.
 *
 * gfx1102 note: the prefix scan's per-wave inclusive sum goes through LDS, not the DPP row-shift
 * butterfly (a DPP exchange inside the scan's tile loop miscompiles here). Each wave writes its
 * lane's count to LDS, barriers, and takes the serial prefix of its 32 lanes.
 *
 * Above 1024 slots the sort is three passes (libhalo's general form): a histogram per 256-slot
 * tile into the caller's scratch, a per-expert scan across the tiles, and the placement (an
 * expert's offset, plus its count in earlier tiles, plus the slot's rank in its own tile). Stable
 * too, so both forms give the same bytes. The scan's cross-expert prefix is a serial LDS loop, not
 * a wave scan, for the same gfx1102 reason as above.
 */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_launch.h"

#define R3D_BARRIER() do { __syncthreads(); asm volatile("" ::: "memory"); } while (0)

namespace {

constexpr uint32_t kSortWave = 32;
constexpr uint32_t kMaxExpert = 512;
constexpr uint32_t kMaxProtect = 8;
constexpr uint32_t kScatterSmall = 1024;
constexpr uint32_t kScatWaves = 8;   // 256 threads

struct Protect {
    uint32_t n;
    uint16_t e[kMaxProtect];
};

/* Local expert `e`'s place in the sort: the protected ones last, the rest closed up in order. */
__device__ __forceinline__ int32_t sort_place(int32_t e, uint32_t n_local, const Protect& p) {
    int32_t below = 0;
    for (uint32_t j = 0; j < p.n; ++j) {
        if ((int32_t)p.e[j] == e) return (int32_t)(n_local - p.n + j);
        below += (int32_t)p.e[j] < e;
    }
    return e - below;
}

template <uint32_t NW>
struct ScatterLds {
    static constexpr uint32_t kTiles = (kMaxExpert + NW * kSortWave - 1) / (NW * kSortWave);
    int32_t e[kScatterSmall];
    int32_t tk[kScatterSmall];
    int32_t run[kScatterSmall];
    int32_t cnt[kMaxExpert + 1];
    int32_t off[kMaxExpert + 1];
    int32_t wt[kTiles][NW];
    int32_t scan[NW * kSortWave];   // the LDS inclusive-scan scratch (gfx1102: no DPP here)
};

/* The sort, from a filled slot list on: histogram, offsets and counts, placement. The caller has
 * written sl.e[0,T) and zeroed sl.cnt[0,n_expert), followed by a barrier. */
template <uint32_t NW, bool INV = false>
__device__ __forceinline__ void scatter_small_place(
    int32_t* __restrict__ sorted, int32_t* __restrict__ off, int32_t* __restrict__ cnt,
    uint32_t T, uint32_t n_expert, ScatterLds<NW>& sl, int32_t* __restrict__ inv) {
    const uint32_t tid = threadIdx.x, lane = tid % kSortWave, wave = tid / kSortWave;

    for (uint32_t i = tid; i < T; i += blockDim.x) {
        const int32_t e = sl.e[i];
        if (e >= 0) sl.tk[i] = atomicAdd(&sl.cnt[e], 1);
    }
    R3D_BARRIER();

    int32_t carry = 0;
    for (uint32_t b = 0, t = 0; b < n_expert; b += NW * kSortWave, ++t) {
        const uint32_t e = b + tid;
        const int32_t v = (e < n_expert) ? sl.cnt[e] : 0;
        // Inclusive wave scan through LDS (a DPP row-shift miscompiles on gfx1102).
        sl.scan[wave * kSortWave + lane] = v;
        R3D_BARRIER();
        int32_t x = 0;
        for (int i = 0; i <= (int)lane; ++i) x += sl.scan[wave * kSortWave + i];
        if (lane == kSortWave - 1) sl.wt[t][wave] = x;
        R3D_BARRIER();
        int32_t below = 0, tot = 0;
#pragma unroll
        for (uint32_t k = 0; k < NW; ++k) {
            const int32_t c = sl.wt[t][k];
            below += (k < wave) ? c : 0;
            tot += c;
        }
        if (e < n_expert) {
            const int32_t o = carry + below + x - v;   // inclusive -> exclusive
            sl.off[e] = o;
            off[e] = o;
            cnt[e] = v;
        }
        carry += tot;
    }
    if (tid == 0) off[n_expert] = carry;
    R3D_BARRIER();

    for (uint32_t i = (uint32_t)carry + tid; i < T; i += blockDim.x) sorted[i] = -1;
    for (uint32_t i = tid; i < T; i += blockDim.x) {
        const int32_t e = sl.e[i];
        if (e >= 0) sl.run[sl.off[e] + sl.tk[i]] = (int32_t)i;
    }
    R3D_BARRIER();
    for (uint32_t i = tid; i < T; i += blockDim.x) {
        const int32_t e = sl.e[i];
        if (e < 0) {
            if (INV) inv[i] = -1;
            continue;
        }
        const int32_t o = sl.off[e], n = sl.cnt[e];
        int32_t rank = 0;
        for (int32_t k = 0; k < n; ++k) rank += sl.run[o + k] < (int32_t)i;
        sorted[o + rank] = (int32_t)i;
        if (INV) inv[i] = o + rank;
    }
}

template <bool INV>
__global__ __launch_bounds__(kScatWaves * kSortWave) void scatter_small_kernel(
    const int32_t* __restrict__ ids, int32_t* __restrict__ sorted, int32_t* __restrict__ off,
    int32_t* __restrict__ cnt, uint32_t T, uint32_t n_expert, int32_t base, Protect prot,
    int32_t* __restrict__ inv) {
    __shared__ ScatterLds<kScatWaves> sl;
    const uint32_t tid = threadIdx.x;
    for (uint32_t e = tid; e < n_expert; e += blockDim.x) sl.cnt[e] = 0;
    for (uint32_t i = tid; i < T; i += blockDim.x) {
        const int32_t e = ids[i] - base;
        sl.e[i] = (e >= 0 && e < (int32_t)n_expert) ? sort_place(e, n_expert, prot) : -1;
    }
    R3D_BARRIER();
    scatter_small_place<kScatWaves, INV>(sorted, off, cnt, T, n_expert, sl, inv);
}

/* ================================================================== the general form */

constexpr uint32_t kScatterTile = 256;

/* Pass 1: each tile's count of every expert, bcnt[tile][expert]. */
__global__ __launch_bounds__(kScatterTile) void scatter_hist_kernel(
    const int32_t* __restrict__ ids, int32_t* __restrict__ bcnt, uint32_t T, uint32_t n_expert,
    int32_t base, Protect prot) {
    __shared__ int32_t s_h[kMaxExpert];
    const uint32_t tid = threadIdx.x, b = blockIdx.x;
    for (uint32_t e = tid; e < n_expert; e += kScatterTile) s_h[e] = 0;
    R3D_BARRIER();
    const uint32_t i = b * kScatterTile + tid;
    if (i < T) {
        const int32_t e = ids[i] - base;
        if (e >= 0 && e < (int32_t)n_expert) atomicAdd(&s_h[sort_place(e, n_expert, prot)], 1);
    }
    R3D_BARRIER();
    for (uint32_t e = tid; e < n_expert; e += kScatterTile)
        bcnt[(size_t)b * n_expert + e] = s_h[e];
}

/* Pass 2: per expert, the count in earlier tiles (bcnt rewritten in place to it) and the total;
 * then the offsets, an exclusive prefix over the totals. */
__global__ __launch_bounds__(kMaxExpert) void scatter_scan_kernel(
    int32_t* __restrict__ bcnt, int32_t* __restrict__ off, int32_t* __restrict__ cnt,
    uint32_t n_tile, uint32_t n_expert) {
    __shared__ int32_t s_run[kMaxExpert];
    const uint32_t e = threadIdx.x;
    int32_t run = 0;
    if (e < n_expert) {
        for (uint32_t b = 0; b < n_tile; ++b) {
            const int32_t v = bcnt[(size_t)b * n_expert + e];
            bcnt[(size_t)b * n_expert + e] = run;
            run += v;
        }
        cnt[e] = run;
    }
    s_run[e] = run;
    R3D_BARRIER();
    if (e == 0) {
        int32_t acc = 0;
        for (uint32_t x = 0; x < n_expert; ++x) { off[x] = acc; acc += s_run[x]; }
        off[n_expert] = acc;
    }
}

/* Pass 3: each routed slot at its expert's offset plus the expert's count in earlier tiles plus
 * its rank in this tile. Sorted rows past the live count are -1; no placement lands there. INV
 * also writes each slot's row, -1 when dropped. */
template <bool INV>
__global__ __launch_bounds__(kScatterTile) void scatter_place_kernel(
    const int32_t* __restrict__ ids, const int32_t* __restrict__ bcnt,
    const int32_t* __restrict__ off, int32_t* __restrict__ sorted,
    uint32_t T, uint32_t n_expert, int32_t base, Protect prot, int32_t* __restrict__ inv) {
    __shared__ int32_t s_id[kScatterTile];
    const uint32_t tid = threadIdx.x, b = blockIdx.x;
    const uint32_t i = b * kScatterTile + tid;
    const int32_t l = (i < T) ? ids[i] - base : -1;
    const int32_t e = (l >= 0 && l < (int32_t)n_expert) ? sort_place(l, n_expert, prot) : -1;
    s_id[tid] = e;
    const uint32_t live = (uint32_t)off[n_expert];
    if (i >= live && i < T) sorted[i] = -1;
    R3D_BARRIER();
    if (INV && i < T && e < 0) inv[i] = -1;
    if (i >= T || e < 0) return;
    int32_t rank = 0;
    for (uint32_t j = 0; j < tid; ++j) rank += (s_id[j] == e);
    const int32_t at = off[e] + bcnt[(size_t)b * n_expert + e] + rank;
    sorted[at] = (int32_t)i;
    if (INV) inv[i] = at;
}

/* Both entry points: the small form up to kScatterSmall slots, the three passes above it. */
int scatter_body(const RadArgs* a, RadStream s, bool with_inv)
{
    if (!a || a->n_t < (with_inv ? 5 : 4)) return R3D_NO(RAD_E_INVAL);
    const RadTensor* ids = r3d_opt(a, 0), *sorted = r3d_opt(a, 1), *off = r3d_opt(a, 2),
                    *cnt = r3d_opt(a, 3), *inv = with_inv ? r3d_opt(a, 4) : nullptr;
    if (!r3d_p(ids) || !r3d_wp(sorted) || !r3d_wp(off) || !r3d_wp(cnt) ||
        (with_inv && !r3d_wp(inv)))
        return R3D_NO(RAD_E_INVAL);
    if (ids->dtype != RAD_I32 || sorted->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32 || (inv && inv->dtype != RAD_I32))
        return R3D_NO(RAD_E_DTYPE);
    if (!r3d_contig(ids) || !r3d_contig(sorted) || !r3d_contig(off) || !r3d_contig(cnt) ||
        (inv && !r3d_contig(inv)))
        return R3D_NO(RAD_E_STRIDE);

    const long long n_expert = r3d_geti_or(a, "n_expert", 0);
    if (n_expert <= 0 || n_expert > (long long)kMaxExpert) return R3D_NO(RAD_E_SHAPE);
    const long long T = r3d_numel(ids);
    if (T <= 0) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(sorted) < T || (inv && r3d_numel(inv) < T)) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(off) < n_expert + 1 || r3d_numel(cnt) < n_expert) return R3D_NO(RAD_E_SHAPE);

    const long long base = r3d_geti_or(a, "expert_base", 0);
    if (base < 0) return R3D_NO(RAD_E_SHAPE);

    // protect: a comma list of local expert indices, strictly ascending, at most kMaxProtect.
    Protect prot{};
    if (const char* str = r3d_gets(a, "protect")) {
        const char* p = str;
        long long prev = -1;
        while (*p) {
            char* end = nullptr;
            const long long v = std::strtoll(p, &end, 10);
            if (end == p || v <= prev || v >= n_expert || prot.n >= kMaxProtect)
                return R3D_NO(RAD_E_SHAPE);
            prot.e[prot.n++] = (uint16_t)v;
            prev = v;
            p = end;
            if (*p == ',') ++p;
            else if (*p) return R3D_NO(RAD_E_SHAPE);
        }
    }

    const int32_t* idp = (const int32_t*)r3d_p(ids);
    int32_t* sp = (int32_t*)r3d_wp(sorted);
    int32_t* op = (int32_t*)r3d_wp(off);
    int32_t* cp = (int32_t*)r3d_wp(cnt);
    int32_t* ip = inv ? (int32_t*)r3d_wp(inv) : nullptr;

    if (T <= (long long)kScatterSmall) {
        if (ip)
            R3D_KLAUNCH(scatter_small_kernel<true>, dim3(1), dim3(kScatWaves * kSortWave),
                        (hipStream_t)s, idp, sp, op, cp, (uint32_t)T, (uint32_t)n_expert,
                        (int32_t)base, prot, ip);
        else
            R3D_KLAUNCH(scatter_small_kernel<false>, dim3(1), dim3(kScatWaves * kSortWave),
                        (hipStream_t)s, idp, sp, op, cp, (uint32_t)T, (uint32_t)n_expert,
                        (int32_t)base, prot, ip);
        return (int)hipGetLastError();
    }

    const long long n_tile = (T + kScatterTile - 1) / kScatterTile;
    const long long need = n_tile * n_expert * (long long)sizeof(int32_t);
    if (!a->scratch || a->scratch_bytes < need) return R3D_NO(RAD_E_SCRATCH);
    int32_t* bcnt = (int32_t*)a->scratch;
    R3D_KLAUNCH(scatter_hist_kernel, dim3((unsigned)n_tile), dim3(kScatterTile), (hipStream_t)s,
                idp, bcnt, (uint32_t)T, (uint32_t)n_expert, (int32_t)base, prot);
    R3D_KLAUNCH(scatter_scan_kernel, dim3(1), dim3(kMaxExpert), (hipStream_t)s,
                bcnt, op, cp, (uint32_t)n_tile, (uint32_t)n_expert);
    if (ip)
        R3D_KLAUNCH(scatter_place_kernel<true>, dim3((unsigned)n_tile), dim3(kScatterTile),
                    (hipStream_t)s, idp, bcnt, op, sp, (uint32_t)T, (uint32_t)n_expert,
                    (int32_t)base, prot, ip);
    else
        R3D_KLAUNCH(scatter_place_kernel<false>, dim3((unsigned)n_tile), dim3(kScatterTile),
                    (hipStream_t)s, idp, bcnt, op, sp, (uint32_t)T, (uint32_t)n_expert,
                    (int32_t)base, prot, ip);
    return (int)hipGetLastError();
}

}  // namespace

extern "C" {

int r3d_rad_moe_scatter(const RadArgs* a, RadStream s)
{
    return scatter_body(a, s, false);
}

/* moe_scatter_inv: the same sort, plus the inverse permutation (each slot's row in sorted_tok). */
int r3d_rad_moe_scatter_inv(const RadArgs* a, RadStream s)
{
    return scatter_body(a, s, true);
}

/* The three-pass form's per-tile counts, [n_tile, n_expert] i32; the small form needs none. */
int64_t r3d_rad_moe_scatter_scratch(const RadArgs* a)
{
    if (!a) return 0;
    const long long n_expert = r3d_geti_or(a, "n_expert", 0);
    /* Called at declare with the band's bound, when the ids may carry no data yet: then
     * M x top_k. */
    const RadTensor* ids = r3d_opt(a, 0);
    const long long T = ids ? r3d_numel(ids) : r3d_geti_or(a, "M", 0) * r3d_geti_or(a, "top_k", 1);
    if (n_expert <= 0 || T <= (long long)kScatterSmall) return 0;
    const long long n_tile = (T + kScatterTile - 1) / kScatterTile;
    return n_tile * n_expert * (long long)sizeof(int32_t);
}

}  // extern "C"
