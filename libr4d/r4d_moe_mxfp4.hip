/* Canonical MXFP4 experts: packed E2M1 codes and E8M0 bytes remain in their
 * source layout, including mapped host expert tables. A wave decodes packed
 * words into exact E4M3 staging bytes for FP8 WMMA. Activation then weight
 * scales fold in F32 per 32 values, matching the reference grouping. */
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_fp8_wmma.h"
#include "r4d_mxfp4.h"
#include <climits>
#include <algorithm>
#include <cstring>

namespace {
struct MxMeta {
    long wstep[2], sstep[2];
    int wrow[2], srow[2], n[2], k[2];
    long ar, asr, yr;
    int T, Arows, N, K, experts, topk, sorted_a, parts;
    int stage_out, out16;
};
__device__ const uint8_t* entry(const void* base, long step, int e) {
    return step ? (const uint8_t*)base + (long)e * step : ((const uint8_t* const*)base)[e];
}
/* E2M1 values are exactly representable in E4M3. Stage those codes directly,
 * use FP8 WMMA, and fold each 32-wide E8M0 scale into the F32 accumulator.
 * No persistent expansion or checkpoint relayout is needed. */
/* Eight codes of a packed word to their E4M3 bytes in k order, by table (r4d_mxfp4.h, v_perm on
 * the magnitude): the bytes of the bytewise arithmetic decode in about a third of the
 * instructions. */
__device__ __forceinline__ uint2 mx_codes8(uint32_t packed) {
    uint32_t lo, hi;
    r4d_mxfp4_unpack8_exact(packed, &lo, &hi);
    return uint2{ lo, hi };
}
/* The row-group form: a one-wave block per (16 columns, 16 sorted rows), the rows split at each
 * expert boundary. */
__global__ void moe_mxfp4_fp8_kernel(const uint8_t* a, const float* as, const void* w0,
                                     const void* s0, const void* w1, const void* s1,
                                     const int* sorted, const int* offset, uint16_t* y, MxMeta m) {
    __shared__ __align__(16) uint8_t aa[16 * 136], bb[16 * 136];
    const int tid = threadIdx.x, lane = tid & 31, col0 = blockIdx.x * 16;
    const int first = blockIdx.y * 16, group_end = min((int)(blockIdx.y + 1) * 16, m.T);
    for (int start = first; start < group_end;) {
        if (sorted[start] < 0 || offset[m.experts] <= start) {
            if (tid < 16 && col0 + tid < m.N) y[(long)start * m.yr + col0 + tid] = 0;
            ++start;
            continue;
        }
        int lo = 0, hi = m.experts;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) / 2;
            if (offset[mid] <= start)
                lo = mid;
            else
                hi = mid;
        }
        const int cls = lo & 1, end = min(group_end, offset[lo + 1]);
        if (end <= start) return;
        const auto* w = entry(cls ? w1 : w0, m.wstep[cls], lo / 2);
        const auto* ws = entry(cls ? s1 : s0, m.sstep[cls], lo / 2);
        auto total = r4d_fp8::acc_zero();
        for (int kb = 0; kb < m.k[cls]; kb += 128) {
            for (int j = tid; j < 16 * 16; j += 32) {
                const int r = j / 16, d = (j % 16) * 8;
                const int ar = start + r < end && sorted[start + r] >= 0
                                   ? (m.sorted_a ? start + r : sorted[start + r] / m.topk)
                                   : -1;
                uint2 av{};
                if (ar >= 0 && ar < m.Arows) __builtin_memcpy(&av, a + (long)ar * m.ar + kb + d, 8);
                *reinterpret_cast<uint2*>(aa + r * 136 + d) = av;
            }
            for (int j = tid; j < 16 * 16; j += 32) {
                const int r = j / 16, d = (j % 16) * 8;
                const int n = col0 + r, part = n / (m.N / m.parts), within = n % (m.N / m.parts);
                const int wn = part * (m.n[cls] / m.parts) + within;
                uint32_t packed = 0;
                if (n < m.N && within < m.n[cls] / m.parts)
                    __builtin_memcpy(&packed, w + (long)wn * m.wrow[cls] + (kb + d) / 2, 4);
                *reinterpret_cast<uint2*>(bb + r * 136 + d) = mx_codes8(packed);
            }
            __syncthreads();
            const int n = col0 + (lane & 15), part = n / (m.N / m.parts),
                      within = n % (m.N / m.parts);
            const int wn = part * (m.n[cls] / m.parts) + within;
            const bool valid = n < m.N && within < m.n[cls] / m.parts;
#pragma unroll
            for (int group = 0; group < 4; ++group) {
                const int exp = valid ? ws[(long)wn * m.srow[cls] + kb / 32 + group] : 127;
                auto raw = r4d_fp8::acc_zero();
#pragma unroll
                for (int k = group * 32; k < (group + 1) * 32; k += 16)
                    raw = r4d_fp8::mma(r4d_fp8::frag<128>(aa, 0, k, lane),
                                       r4d_fp8::frag<128>(bb, 0, k, lane), raw);
                // E8M0 maps directly to the F32 exponent field; code zero is 2^-127,
                // a representable subnormal, and code 255 is NaN.
                const float scale = __uint_as_float(exp == 255 ? 0x7fc00000u
                                                    : exp      ? ((uint32_t)exp << 23)
                                                               : 0x00400000u);
#pragma unroll
                for (int row = 0; row < 8; ++row) {
                    const int r = start + (lane >> 4) * 8 + row;
                    if (r < end) {
                        const int ar = sorted[r] < 0 ? -1 : m.sorted_a ? r : sorted[r] / m.topk;
                        if (ar >= 0 && ar < m.Arows)
                            total[row] += raw[row] * as[(long)ar * m.asr + kb / 128] * scale;
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int row = 0; row < 8; ++row) {
            const int r = start + (lane >> 4) * 8 + row, n = col0 + (lane & 15);
            if (r < end && n < m.N) y[(long)r * m.yr + n] = f32_to_bf16(total[row]);
        }
        start = end;
    }
}

/* An expert's weight and exponent rows reached through entry() are generic pointers, so their loads
 * compile to flat loads, which every LDS wait (s_wait_dscnt) also waits for: a barrier then stalls
 * on the prefetch. The tables hold device or device-mapped host addresses, so a global load reads
 * the same bytes. Byte-aligned, as a stacked plane's row pitch may be odd. */
typedef r4d_u32x4 mx_g16 __attribute__((aligned(1)));
typedef uint32_t mx_g4 __attribute__((aligned(1)));
__device__ __forceinline__ r4d_u32x4 mx_ldg16(const uint8_t* p) {
    return *(const __attribute__((address_space(1))) mx_g16*)p;
}
__device__ __forceinline__ uint32_t mx_ldg4(const uint8_t* p) {
    return *(const __attribute__((address_space(1))) mx_g4*)p;
}
/* The GLM prefill form: jobs of 32 rows of one expert by 64 columns, eight waves, with its loads a
 * stage ahead. Expert e begins at job floor(offset[e]/32)+e: the extra job per expert bounds the
 * padding without a prefix-sum pass, and every live job owns one row tile of one expert, so hot
 * experts still spread across independent blocks. A job's row table resolves into LDS once; the
 * next 128 of K (activation bytes, weight codes, an exponent dword a column, the activation
 * scales) is loaded into registers while the current stage computes from LDS, where a form that
 * loads after the barrier waits on three dependent latencies a stage: the routing entry, the
 * activation row it names, and the exponent bytes. A job begins at its first row even when that
 * slot is dropped; rows are independent, so the bytes are the same.
 * The exponents are decoded once, at staging; the fold carries no row predicate, the epilogue
 * writes a dropped row's zero instead.
 * One stage ahead only: a deeper prefetch measured slower on the expert-major forms. The barriers
 * order LDS only (lds_barrier), so the prefetch stays in flight across them and L0 keeps the
 * activation rows the next column block reads. */
__global__ __launch_bounds__(256)
    R4D_WAVES(16) void moe_mxfp4_prefill_kernel(const uint8_t* a, const float* as, const void* w0,
                                                const void* s0, const void* w1, const void* s1,
                                                const int* sorted, const int* offset, uint16_t* y,
                                                MxMeta m) {
    constexpr int BN = 64, BM = 32, Threads = 256, P = 136;
    static_assert(BM * 128 / 16 == Threads && BN * 64 / 16 == Threads,
                  "one 16-byte chunk a thread and operand");
    __shared__ __align__(16) uint8_t aa[BM * P], bb[BN * P];
    __shared__ __align__(16) float ss[4 * BN];
    __shared__ float sa[BM];
    __shared__ int src[BM];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, col0 = blockIdx.x * BN;
    const int wave_col = wave & 3, wave_row = wave >> 2;
    const int job = blockIdx.y, tail = offset[m.experts] / BM + m.experts;
    if (job >= tail) {
        const int r0 = offset[m.experts] + (job - tail) * BM;
        for (int r = r0; r < min(r0 + BM, m.T); ++r)
            if (tid < BN && col0 + tid < m.N) y[(long)r * m.yr + col0 + tid] = 0;
        return;
    }
    int lo = 0, hi = m.experts;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) / 2;
        if (offset[mid] / BM + mid <= job)
            lo = mid;
        else
            hi = mid;
    }
    const int first = offset[lo] + (job - (offset[lo] / BM + lo)) * BM,
              end = min(first + BM, offset[lo + 1]);
    if (first >= end) return;
    const int cls = lo & 1, kc = m.k[cls], pw = m.N / m.parts, cw = m.n[cls] / m.parts;
    const auto* w = entry(cls ? w1 : w0, m.wstep[cls], lo / 2);
    const auto* ws = entry(cls ? s1 : s0, m.sstep[cls], lo / 2);
    if (tid < BM) {
        const int r = first + tid, flat = r < end ? sorted[r] : -1;
        const int ar = flat < 0 ? -1 : m.sorted_a ? r : flat / m.topk;
        src[tid] = ar >= 0 && ar < m.Arows ? ar : -1;
    }
    // The block's columns are fixed, so each thread's weight row and exponent row are too.
    const int wcol = tid >> 2, wd = (tid & 3) * 16, wn_c = col0 + wcol, wn_w = wn_c % pw;
    const bool wvalid = wn_c < m.N && wn_w < cw;
    const uint8_t* wp = w + (long)((wn_c / pw) * cw + wn_w) * m.wrow[cls] + wd;
    const int en_c = col0 + tid, en_w = en_c % pw;
    const bool evalid = tid < BN && en_c < m.N && en_w < cw;
    const uint8_t* ep = ws + (long)((en_c / pw) * cw + en_w) * m.srow[cls];
    lds_barrier();
    const int arow = tid >> 3, ad = (tid & 7) * 16, a_src = src[arow];
    const uint8_t* ap = a + (long)max(a_src, 0) * m.ar + ad;
    const int sa_row = tid - BN, sa_src = sa_row >= 0 && sa_row < BM ? src[sa_row] : -1;
    const float* asp = as + (long)max(sa_src, 0) * m.asr;
    const int rowbase = wave_row * 16 + (lane >> 4) * 8, mycol = wave_col * 16 + (lane & 15);
    const bool live = first + wave_row * 16 < end;
    bool rv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) rv[j] = src[rowbase + j] >= 0;
    r4d_u32x4 ra, rw;
    uint32_t rs = 0x7f7f7f7fu;
    float rsa = 0;
    auto load = [&](int k0) {
        ra = r4d_u32x4{ 0, 0, 0, 0 };
        if (a_src >= 0) __builtin_memcpy(&ra, ap + k0, 16);
        rw = r4d_u32x4{ 0, 0, 0, 0 };
        if (wvalid) rw = mx_ldg16(wp + k0 / 2);
        if (evalid) rs = mx_ldg4(ep + k0 / 32);
        if (sa_src >= 0) rsa = asp[k0 / 128];
    };
    auto store = [&]() {
        *reinterpret_cast<r4d_u32x4*>(aa + arow * P + ad) = ra;
#pragma unroll
        for (int t = 0; t < 4; ++t)
            *reinterpret_cast<uint2*>(bb + wcol * P + 2 * wd + 8 * t) = mx_codes8(rw[t]);
        if (tid < BN) {
            // E8M0 maps directly to the F32 exponent field; code zero is 2^-127, a representable
            // subnormal, and code 255 is NaN.
            float sc[4];
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const uint32_t exp = (rs >> (8 * g)) & 0xffu;
                sc[g] = __uint_as_float(exp == 255 ? 0x7fc00000u : exp ? (exp << 23) : 0x00400000u);
            }
            *reinterpret_cast<float4*>(ss + 4 * tid) = make_float4(sc[0], sc[1], sc[2], sc[3]);
        }
        if (sa_row >= 0 && sa_row < BM) sa[sa_row] = rsa;
    };
    auto total = r4d_fp8::acc_zero();
    load(0);
    for (int k0 = 0; k0 < kc; k0 += 128) {
        store();
        lds_barrier();
        if (k0 + 128 < kc) load(k0 + 128);
        if (live) {
            const float4 sc4 = *reinterpret_cast<const float4*>(ss + 4 * mycol);
            const float scales[4] = { sc4.x, sc4.y, sc4.z, sc4.w };
            float activation[8];
#pragma unroll
            for (int row = 0; row < 8; ++row) activation[row] = sa[rowbase + row];
#pragma unroll
            for (int group = 0; group < 4; ++group) {
                auto raw = r4d_fp8::acc_zero();
#pragma unroll
                for (int k = group * 32; k < (group + 1) * 32; k += 16)
                    raw = r4d_fp8::mma(r4d_fp8::frag<128>(aa, wave_row * 16, k, lane),
                                       r4d_fp8::frag<128>(bb, wave_col * 16, k, lane), raw);
// A dropped row folds zeros (or NaN under a code-255 scale) and is written as 0.
#pragma unroll
                for (int row = 0; row < 8; ++row)
                    total[row] += raw[row] * activation[row] * scales[group];
                // Two groups' WMMA chains at a time: with all four in flight the kernel takes 104
                // VGPRs, 12 waves a SIMD at gfx1201's 24-register granule instead of 16.
                if (group == 1) sched_barrier();
            }
        }
        lds_barrier();
    }
#pragma unroll
    for (int row = 0; row < 8; ++row) {
        const int r = first + rowbase + row, n = col0 + mycol;
        if (r < end && n < m.N) y[(long)r * m.yr + n] = rv[row] ? f32_to_bf16(total[row]) : 0;
    }
}

/* The wide prefill form (P64): jobs of BM rows of one expert, blocks of BN columns, eight waves
 * as BM/16 row waves times 8/(BM/16) column waves, so a wave holds BN/(16*column waves) tiles of
 * 16x16 and reuses each A fragment across them. M32N64 reads every expert slice once per 32 rows
 * and a fragment pair per WMMA pair; at 57 rows an expert (2048 tokens) BM 64 halves the slice
 * reads, and a row wave past the run's end skips its compute, so the taller job costs no WMMA.
 *
 * Resident (K <= 256, down): the job's whole A tile sits in LDS for the block's CT column tiles,
 * walked as one stage sequence whose next stage (weight codes, exponents) is loaded while the
 * current one computes, across tile boundaries; A is read once per CT tiles instead of once per
 * column block. Otherwise (gate_up) A streams with the weights, a stage ahead.
 *
 * Each element keeps the row-group form's sequence: per 128 of K in order, per 32-wide group a
 * fresh WMMA pair from zero, folded as (raw * activation scale) * E8M0 scale and added, two
 * roundings and an add (the unit is built -ffp-contract=off). The tiles of a wave are separate
 * accumulators, so no element sees another order. `stage_out` writes the output through LDS in
 * 16-byte rows instead of a lane's 2-byte column stores. */
template <int BM, int BN, int StageK, int CT, bool Resident>
struct MxP64 {
    static constexpr int Threads = 256, WR = BM / 16, WC = 8 / WR, TC = BN / (16 * WC),
                         KB = StageK / 128;
    static constexpr int KR = 256, AK = Resident ? KR : StageK, PA = AK + 8, PB = StageK + 8,
                         PO = 2 * BN + 16;
    static constexpr int TPA = Threads / BM, ACh = StageK / 16 / TPA, RCh = KR / 16 / TPA;
    static constexpr int TPW = Threads / BN, WCh = StageK / 32 / TPW,
                         SAK = Resident ? KR / 128 : KB;
    static constexpr int BB = BN * PB > BM * PO ? BN * PB : BM * PO;
    static constexpr int Lds = BM * PA + 16 * KB * BN + 4 * SAK * BM + BB + 4 * BM;
    // The waves a SIMD the LDS admits (128 KiB a WGP, a block's eight waves over four SIMDs):
    // stating it keeps the compiler from trading the stage registers for waves it cannot get.
    static constexpr int Waves = 2 * (131072 / Lds > 8 ? 8 : 131072 / Lds);
};
template <int BM, int BN, int StageK, int CT, bool Resident>
__global__ __launch_bounds__(256)
    R4D_WAVES((MxP64<BM, BN, StageK, CT, Resident>::Waves)) void moe_mxfp4_p64_kernel(
        const uint8_t* a, const float* as, const void* w0, const void* s0, const void* w1,
        const void* s1, const int* sorted, const int* offset, uint16_t* y, MxMeta m) {
    using G = MxP64<BM, BN, StageK, CT, Resident>;
    constexpr int Threads = G::Threads, WC = G::WC, TC = G::TC, KB = G::KB, AK = G::AK, PA = G::PA,
                  PB = G::PB;
    constexpr int PO = G::PO, TPA = G::TPA, TPW = G::TPW, SAK = G::SAK;
    static_assert(BM % 16 == 0 && 8 % G::WR == 0 && TC >= 1 && TC * 16 * WC == BN &&
                      (Resident || CT == 1) && G::ACh >= 1 && G::WCh >= 1 && BN * KB <= Threads &&
                      BM * SAK <= Threads && G::Lds <= 65536,
                  "stage chunks must divide evenly over the block, in 64 KiB of LDS");
    __shared__ __align__(16) uint8_t aa[BM * PA], bb[G::BB];
    __shared__ __align__(16) float ss[4 * KB * BN], sa[SAK * BM];
    __shared__ int src[BM];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, wave_row = wave / WC,
              wave_col = wave % WC;
    const int colb = blockIdx.x * (BN * CT), job = blockIdx.y,
              tail = offset[m.experts] / BM + m.experts;
    if (job >= tail) {
        const int r0 = offset[m.experts] + (job - tail) * BM, rows = min(r0 + BM, m.T) - r0;
        for (int i = tid; i < rows * BN * CT; i += Threads) {
            const int n = colb + i % (BN * CT);
            if (n < m.N) y[(long)(r0 + i / (BN * CT)) * m.yr + n] = 0;
        }
        return;
    }
    int lo = 0, hi = m.experts;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) / 2;
        if (offset[mid] / BM + mid <= job)
            lo = mid;
        else
            hi = mid;
    }
    const int first = offset[lo] + (job - (offset[lo] / BM + lo)) * BM,
              end = min(first + BM, offset[lo + 1]);
    if (first >= end) return;
    const int cls = lo & 1, kc = m.k[cls], pw = m.N / m.parts, cw = m.n[cls] / m.parts;
    const auto* w = entry(cls ? w1 : w0, m.wstep[cls], lo / 2);
    const auto* ws = entry(cls ? s1 : s0, m.sstep[cls], lo / 2);
    if (tid < BM) {
        const int r = first + tid, flat = r < end ? sorted[r] : -1;
        const int ar = flat < 0 ? -1 : m.sorted_a ? r : flat / m.topk;
        src[tid] = ar >= 0 && ar < m.Arows ? ar : -1;
    }
    lds_barrier();
    // A thread's A chunks share one row and its weight chunks one column, so each keeps a single
    // row pointer; neighbouring threads cover neighbouring 16 bytes, as one coalesced row.
    const int arow = tid / TPA, a0 = tid % TPA, a_src = src[arow];
    const uint8_t* ap = a + (long)max(a_src, 0) * m.ar + a0 * 16;
    const int wcol = tid / TPW, w0c = tid % TPW, ecol = tid / KB, ekb = tid % KB, skb = tid / BM;
    const int sa_src = tid < BM * SAK ? src[tid % BM] : -1;
    const float* asp = as + (long)max(sa_src, 0) * m.asr;
    const int rowbase = wave_row * 16 + (lane >> 4) * 8, colw = wave_col * 16 * TC;
    const bool live = first + wave_row * 16 < end;
    const uint8_t *wp = w, *ep = ws;
    bool wvalid = false, evalid = false;
    auto columns = [&](int col0) {
        const int wn = col0 + wcol, wi = wn % pw, en = col0 + ecol, ei = en % pw;
        wvalid = wn < m.N && wi < cw;
        wp = w + (long)((wn / pw) * cw + wi) * m.wrow[cls] + w0c * 16;
        evalid = tid < BN * KB && en < m.N && ei < cw;
        ep = ws + (long)((en / pw) * cw + ei) * m.srow[cls];
    };
    if constexpr (Resident) {
#pragma unroll
        for (int i = 0; i < G::RCh; ++i) {
            const int d = (a0 + i * TPA) * 16;
            r4d_u32x4 v{ 0, 0, 0, 0 };
            if (a_src >= 0 && d < kc) __builtin_memcpy(&v, ap + i * TPA * 16, 16);
            *reinterpret_cast<r4d_u32x4*>(aa + arow * PA + d) = v;
        }
        if (tid < BM * SAK) sa[tid] = sa_src >= 0 && skb * 128 < kc ? asp[skb] : 0;
    }
    const int ksn = (kc + StageK - 1) / StageK,
              tiles = Resident ? min(CT, (m.N - colb + BN - 1) / BN) : 1, S = tiles * ksn;
    r4d_u32x4 ra[Resident ? 1 : G::ACh], rw[G::WCh];
    uint32_t re = 0x7f7f7f7fu;
    float rsa = 0;
    int cur = -1;
    auto load = [&](int s) {
        const int tile = s / ksn, k0 = (s - tile * ksn) * StageK;
        if (tile != cur) {
            cur = tile;
            columns(colb + tile * BN);
        }
        if constexpr (!Resident) {
#pragma unroll
            for (int i = 0; i < G::ACh; ++i) {
                ra[i] = r4d_u32x4{ 0, 0, 0, 0 };
                if (a_src >= 0 && k0 + (a0 + i * TPA) * 16 < kc)
                    __builtin_memcpy(&ra[i], ap + k0 + i * TPA * 16, 16);
            }
            if (tid < BM * KB) rsa = sa_src >= 0 && k0 + skb * 128 < kc ? asp[k0 / 128 + skb] : 0;
        }
#pragma unroll
        for (int i = 0; i < G::WCh; ++i) {
            rw[i] = r4d_u32x4{ 0, 0, 0, 0 };
            if (wvalid && k0 + (w0c + i * TPW) * 32 < kc)
                rw[i] = mx_ldg16(wp + k0 / 2 + i * TPW * 16);
        }
        re = 0x7f7f7f7fu;
        if (evalid && k0 + ekb * 128 < kc) re = mx_ldg4(ep + (k0 + ekb * 128) / 32);
    };
    auto store = [&]() {
        if constexpr (!Resident) {
#pragma unroll
            for (int i = 0; i < G::ACh; ++i)
                *reinterpret_cast<r4d_u32x4*>(aa + arow * PA + (a0 + i * TPA) * 16) = ra[i];
            if (tid < BM * KB) sa[tid] = rsa;
        }
#pragma unroll
        for (int i = 0; i < G::WCh; ++i) {
#pragma unroll
            for (int t = 0; t < 4; ++t)
                *reinterpret_cast<uint2*>(bb + wcol * PB + (w0c + i * TPW) * 32 + 8 * t) =
                    mx_codes8(rw[i][t]);
        }
        if (tid < BN * KB) {
// E8M0 maps directly to the F32 exponent field; code zero is 2^-127, a representable
// subnormal, and code 255 is NaN.
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const uint32_t exp = (re >> (8 * g)) & 0xffu;
                ss[(ekb * 4 + g) * BN + ecol] =
                    __uint_as_float(exp == 255 ? 0x7fc00000u
                                    : exp      ? (exp << 23)
                                               : 0x00400000u);
            }
        }
    };
    r4d_fp8::f8v total[TC];
#pragma unroll
    for (int t = 0; t < TC; ++t) total[t] = r4d_fp8::acc_zero();
    auto compute = [&](int s) {
        const int k0 = (s % ksn) * StageK;
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) {
            if (k0 + kb * 128 >= kc) break;
            const float* sar = sa + (Resident ? k0 / 128 + kb : kb) * BM + rowbase;
            float activation[8];
#pragma unroll
            for (int row = 0; row < 8; ++row) activation[row] = sar[row];
            const int ka = (Resident ? k0 : 0) + kb * 128;
#pragma unroll
            for (int group = 0; group < 4; ++group) {
                const auto fa0 = r4d_fp8::frag<AK>(aa, wave_row * 16, ka + group * 32, lane);
                const auto fa1 = r4d_fp8::frag<AK>(aa, wave_row * 16, ka + group * 32 + 16, lane);
#pragma unroll
                for (int t = 0; t < TC; ++t) {
                    auto raw = r4d_fp8::mma(
                        fa0, r4d_fp8::frag<StageK>(bb, colw + t * 16, kb * 128 + group * 32, lane),
                        r4d_fp8::acc_zero());
                    raw = r4d_fp8::mma(
                        fa1,
                        r4d_fp8::frag<StageK>(bb, colw + t * 16, kb * 128 + group * 32 + 16, lane),
                        raw);
                    const float scale = ss[(kb * 4 + group) * BN + colw + t * 16 + (lane & 15)];
// A dropped row folds zeros (or NaN under a code-255 scale) and is written as 0.
#pragma unroll
                    for (int row = 0; row < 8; ++row)
                        total[t][row] += raw[row] * activation[row] * scale;
                }
                // One group's TC chains at a time: they already give the WMMA pipe independent
                // work, and a second group in flight would double the raw registers.
                sched_barrier();
            }
        }
    };
    auto epilogue = [&](int col0) {
        if (m.stage_out) {
            if (live) {
#pragma unroll
                for (int t = 0; t < TC; ++t) {
#pragma unroll
                    for (int row = 0; row < 8; ++row)
                        *reinterpret_cast<uint16_t*>(bb + (rowbase + row) * PO +
                                                     2 * (colw + t * 16 + (lane & 15))) =
                            src[rowbase + row] >= 0 ? f32_to_bf16(total[t][row]) : 0;
                }
            }
            lds_barrier();
            for (int j = tid; j < BM * BN / 8; j += Threads) {
                const int r = j / (BN / 8), c = (j % (BN / 8)) * 8, n = col0 + c;
                if (first + r >= end || n >= m.N) continue;
                const r4d_u32x4 v = *reinterpret_cast<const r4d_u32x4*>(bb + r * PO + 2 * c);
                uint16_t* dst = y + (long)(first + r) * m.yr + n;
                if (m.out16 && n + 8 <= m.N)
                    *reinterpret_cast<r4d_u32x4*>(dst) = v;
                else
                    for (int q = 0; q < 8 && n + q < m.N; ++q)
                        dst[q] = (uint16_t)(v[q / 2] >> (16 * (q & 1)));
            }
            lds_barrier();
        } else if (live) {
#pragma unroll
            for (int t = 0; t < TC; ++t) {
#pragma unroll
                for (int row = 0; row < 8; ++row) {
                    const int r = first + rowbase + row, n = col0 + colw + t * 16 + (lane & 15);
                    if (r < end && n < m.N)
                        y[(long)r * m.yr + n] =
                            src[rowbase + row] >= 0 ? f32_to_bf16(total[t][row]) : 0;
                }
            }
        }
    };
    load(0);
    for (int s = 0; s < S; ++s) {
        store();
        lds_barrier();
        if (s + 1 < S) load(s + 1);
        if (live) compute(s);
        lds_barrier();
        if (s % ksn == ksn - 1) {
            epilogue(colb + (s / ksn) * BN);
#pragma unroll
            for (int t = 0; t < TC; ++t) total[t] = r4d_fp8::acc_zero();
        }
    }
}

/* The decode form, expert-major: a block per (column block, expert) takes every row of its expert
 * in 16-row tiles, so an expert slice is read from memory once however its rows fall in the
 * sorted order (row groups re-read it once per group its run spans). Parallelism is column
 * blocks times touched experts. Split-K would sum partials in another order than the row-group
 * form and lose its bits, so each element keeps that form's sequence: per 128 of K in order, per
 * 32-wide group a fresh WMMA pair, folded as raw * activation scale * E8M0 scale.
 *
 * A stage of StageK of K is loaded into registers while the previous one computes from LDS; at
 * gate_up's K of 4096 the 256 stage measured fastest. Block row `experts` zeroes the dropped rows
 * past offset[experts]. */
template <int Waves, int StageK>
__global__ __launch_bounds__(32 * Waves) void moe_mxfp4_expert_kernel(
    const uint8_t* a, const float* as, const void* w0, const void* s0, const void* w1,
    const void* s1, const int* sorted, const int* offset, uint16_t* y, MxMeta m) {
    constexpr int BN = 16 * Waves, Threads = 32 * Waves, KB = StageK / 128, P = StageK + 8;
    constexpr int AChunks = StageK / Threads, WChunks = BN * StageK / 32 / Threads;
    constexpr int SChunks = (BN * KB + Threads - 1) / Threads;
    static_assert(StageK % Threads == 0 && (BN * StageK / 32) % Threads == 0 && 16 * KB <= Threads,
                  "stage chunks must divide evenly over the block");
    __shared__ __align__(16) uint8_t aa[16 * P], bb[BN * P];
    __shared__ uint32_t ss[BN * KB];
    __shared__ float sa[16 * KB];
    __shared__ int src[16];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, col0 = blockIdx.x * BN,
              e = blockIdx.y;
    if (e == m.experts) {
        const int r0 = offset[m.experts];
        for (long i = tid; i < (long)(m.T - r0) * BN; i += Threads) {
            const int n = col0 + (int)(i % BN);
            if (n < m.N) y[(long)(r0 + i / BN) * m.yr + n] = 0;
        }
        return;
    }
    const int lo = offset[e], hi = offset[e + 1];
    if (lo >= hi) return;
    const int cls = e & 1, kc = m.k[cls], pw = m.N / m.parts, cw = m.n[cls] / m.parts;
    const auto* w = entry(cls ? w1 : w0, m.wstep[cls], e / 2);
    const auto* ws = entry(cls ? s1 : s0, m.sstep[cls], e / 2);
    const int rowbase = (lane >> 4) * 8, mycol = wave * 16 + (lane & 15);
    auto wrow = [&](int c, bool& valid) {
        const int n = col0 + c, within = n % pw;
        valid = n < m.N && within < cw;
        return (n / pw) * cw + within;
    };
    for (int r0 = lo; r0 < hi; r0 += 16) {
        __syncthreads();
        if (tid < 16) {
            const int r = r0 + tid, flat = r < hi ? sorted[r] : -1;
            const int ar = flat < 0 ? -1 : m.sorted_a ? r : flat / m.topk;
            src[tid] = ar >= 0 && ar < m.Arows ? ar : -1;
        }
        __syncthreads();
        bool rv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) rv[j] = src[rowbase + j] >= 0;
        uint4 ra[AChunks], rw[WChunks];
        uint32_t rs[SChunks];
        float rsa = 0;
        auto load = [&](int k0) {
#pragma unroll
            for (int i = 0; i < AChunks; ++i) {
                const int j = tid + i * Threads, r = j / (StageK / 16),
                          d = (j % (StageK / 16)) * 16, ar = src[r];
                ra[i] = uint4{ 0, 0, 0, 0 };
                if (ar >= 0 && k0 + d < kc)
                    __builtin_memcpy(&ra[i], a + (long)ar * m.ar + k0 + d, 16);
            }
#pragma unroll
            for (int i = 0; i < WChunks; ++i) {
                const int j = tid + i * Threads, c = j / (StageK / 32),
                          d = (j % (StageK / 32)) * 16;
                bool valid;
                const int wn = wrow(c, valid);
                rw[i] = uint4{ 0, 0, 0, 0 };
                if (valid && k0 + 2 * d < kc)
                    __builtin_memcpy(&rw[i], w + (long)wn * m.wrow[cls] + k0 / 2 + d, 16);
            }
#pragma unroll
            for (int i = 0; i < SChunks; ++i) {
                const int j = tid + i * Threads, c = j / KB, kb = k0 + (j % KB) * 128;
                bool valid;
                const int wn = wrow(c, valid);
                rs[i] = 0x7f7f7f7fu;
                if (j < BN * KB && valid && kb < kc)
                    __builtin_memcpy(&rs[i], ws + (long)wn * m.srow[cls] + kb / 32, 4);
            }
            if (tid < 16 * KB) {
                const int ar = src[tid / KB], kb = k0 + (tid % KB) * 128;
                rsa = ar >= 0 && kb < kc ? as[(long)ar * m.asr + kb / 128] : 0;
            }
        };
        auto store = [&]() {
#pragma unroll
            for (int i = 0; i < AChunks; ++i) {
                const int j = tid + i * Threads, r = j / (StageK / 16),
                          d = (j % (StageK / 16)) * 16;
                *reinterpret_cast<uint4*>(aa + r * P + d) = ra[i];
            }
#pragma unroll
            for (int i = 0; i < WChunks; ++i) {
                const int j = tid + i * Threads, c = j / (StageK / 32),
                          d = (j % (StageK / 32)) * 16;
                const uint32_t q[4] = { rw[i].x, rw[i].y, rw[i].z, rw[i].w };
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    *reinterpret_cast<uint2*>(bb + c * P + 2 * d + 8 * t) = mx_codes8(q[t]);
            }
#pragma unroll
            for (int i = 0; i < SChunks; ++i) {
                const int j = tid + i * Threads;
                if (j < BN * KB) ss[j] = rs[i];
            }
            if (tid < 16 * KB) sa[tid] = rsa;
        };
        auto total = r4d_fp8::acc_zero();
        load(0);
        for (int k0 = 0; k0 < kc; k0 += StageK) {
            store();
            __syncthreads();
            if (k0 + StageK < kc) load(k0 + StageK);
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) {
                if (k0 + kb * 128 >= kc) break;
                const uint32_t exps = ss[mycol * KB + kb];
#pragma unroll
                for (int group = 0; group < 4; ++group) {
                    auto raw = r4d_fp8::acc_zero();
#pragma unroll
                    for (int k = kb * 128 + group * 32; k < kb * 128 + (group + 1) * 32; k += 16)
                        raw = r4d_fp8::mma(r4d_fp8::frag<StageK>(aa, 0, k, lane),
                                           r4d_fp8::frag<StageK>(bb, wave * 16, k, lane), raw);
                    const uint32_t exp = (exps >> (8 * group)) & 0xffu;
                    const float scale = __uint_as_float(exp == 255 ? 0x7fc00000u
                                                        : exp      ? (exp << 23)
                                                                   : 0x00400000u);
#pragma unroll
                    for (int row = 0; row < 8; ++row)
                        if (rv[row]) total[row] += raw[row] * sa[(rowbase + row) * KB + kb] * scale;
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int row = 0; row < 8; ++row) {
            const int r = r0 + rowbase + row, n = col0 + mycol;
            if (r < hi && n < m.N) y[(long)r * m.yr + n] = f32_to_bf16(total[row]);
        }
    }
}

/* The expert-major form for a short K, same per-element sequence as the one above: a block walks
 * ColTiles column tiles of 64, each over K in stages of StageK, as one stage sequence with the
 * next stage loaded into registers while the current one computes. At down's K of 256 a single
 * stage leaves a block nothing to overlap; 128 stages over four tiles give it eight, and halve
 * LDS and registers a block, so twice the blocks are resident. Measured fastest for down at
 * every routed-row count up to 16384, prefill forms included. A deeper prefetch, two-wave
 * blocks and non-temporal weight loads measured slower. */
typedef unsigned mx_u4 __attribute__((ext_vector_type(4)));
template <int StageK>
struct MxStage {
    static constexpr int Threads = 128, BN = 64, KB = StageK / 128;
    static constexpr int AChunks = StageK / Threads, WChunks = BN * StageK / 32 / Threads;
    static constexpr int SChunks = (BN * KB + Threads - 1) / Threads;
    mx_u4 a[AChunks], w[WChunks];
    uint32_t s[SChunks];
    float sa;
};
template <int StageK, int ColTiles>
__global__ __launch_bounds__(128) void moe_mxfp4_expert_staged_kernel(
    const uint8_t* a, const float* as, const void* w0, const void* s0, const void* w1,
    const void* s1, const int* sorted, const int* offset, uint16_t* y, MxMeta m) {
    using St = MxStage<StageK>;
    constexpr int BN = St::BN, Threads = St::Threads, KB = St::KB, P = StageK + 8;
    static_assert(StageK % Threads == 0 && (BN * StageK / 32) % Threads == 0 && 16 * KB <= Threads,
                  "stage chunks must divide evenly over the block");
    __shared__ __align__(16) uint8_t aa[16 * P], bb[BN * P];
    __shared__ uint32_t ss[BN * KB];
    __shared__ float sa[16 * KB];
    __shared__ int src[16];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, e = blockIdx.y;
    const int colb = blockIdx.x * (BN * ColTiles);
    if (e == m.experts) {
        const int r0 = offset[m.experts];
        for (long i = tid; i < (long)(m.T - r0) * BN * ColTiles; i += Threads) {
            const int n = colb + (int)(i % (BN * ColTiles));
            if (n < m.N) y[(long)(r0 + i / (BN * ColTiles)) * m.yr + n] = 0;
        }
        return;
    }
    const int lo = offset[e], hi = offset[e + 1];
    if (lo >= hi) return;
    const int cls = e & 1, kc = m.k[cls], pw = m.N / m.parts, cw = m.n[cls] / m.parts;
    const auto* w = entry(cls ? w1 : w0, m.wstep[cls], e / 2);
    const auto* ws = entry(cls ? s1 : s0, m.sstep[cls], e / 2);
    const int ksn = (kc + StageK - 1) / StageK, tiles = min(ColTiles, (m.N - colb + BN - 1) / BN),
              S = tiles * ksn;
    const int rowbase = (lane >> 4) * 8, mycol = wave * 16 + (lane & 15);
    auto wrow = [&](int n, bool& valid) {
        const int within = n % pw;
        valid = n < m.N && within < cw;
        return (n / pw) * cw + within;
    };
    for (int r0 = lo; r0 < hi; r0 += 16) {
        __syncthreads();
        if (tid < 16) {
            const int r = r0 + tid, flat = r < hi ? sorted[r] : -1;
            const int ar = flat < 0 ? -1 : m.sorted_a ? r : flat / m.topk;
            src[tid] = ar >= 0 && ar < m.Arows ? ar : -1;
        }
        __syncthreads();
        bool rv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) rv[j] = src[rowbase + j] >= 0;
        auto load = [&](int s, St& R) {
            const int col0 = colb + (s / ksn) * BN, k0 = (s % ksn) * StageK;
#pragma unroll
            for (int i = 0; i < St::AChunks; ++i) {
                const int j = tid + i * Threads, r = j / (StageK / 16),
                          d = (j % (StageK / 16)) * 16, ar = src[r];
                R.a[i] = mx_u4{ 0, 0, 0, 0 };
                if (ar >= 0 && k0 + d < kc)
                    __builtin_memcpy(&R.a[i], a + (long)ar * m.ar + k0 + d, 16);
            }
#pragma unroll
            for (int i = 0; i < St::WChunks; ++i) {
                const int j = tid + i * Threads, c = j / (StageK / 32),
                          d = (j % (StageK / 32)) * 16;
                bool valid;
                const int wn = wrow(col0 + c, valid);
                R.w[i] = mx_u4{ 0, 0, 0, 0 };
                if (valid && k0 + 2 * d < kc)
                    __builtin_memcpy(&R.w[i], w + (long)wn * m.wrow[cls] + k0 / 2 + d, 16);
            }
#pragma unroll
            for (int i = 0; i < St::SChunks; ++i) {
                const int j = tid + i * Threads, c = j / KB, kb = k0 + (j % KB) * 128;
                bool valid;
                const int wn = wrow(col0 + c, valid);
                R.s[i] = 0x7f7f7f7fu;
                if (j < BN * KB && valid && kb < kc)
                    __builtin_memcpy(&R.s[i], ws + (long)wn * m.srow[cls] + kb / 32, 4);
            }
            if (tid < 16 * KB) {
                const int ar = src[tid / KB], kb = k0 + (tid % KB) * 128;
                R.sa = ar >= 0 && kb < kc ? as[(long)ar * m.asr + kb / 128] : 0;
            }
        };
        auto store = [&](const St& R) {
#pragma unroll
            for (int i = 0; i < St::AChunks; ++i) {
                const int j = tid + i * Threads, r = j / (StageK / 16),
                          d = (j % (StageK / 16)) * 16;
                *reinterpret_cast<mx_u4*>(aa + r * P + d) = R.a[i];
            }
#pragma unroll
            for (int i = 0; i < St::WChunks; ++i) {
                const int j = tid + i * Threads, c = j / (StageK / 32),
                          d = (j % (StageK / 32)) * 16;
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    *reinterpret_cast<uint2*>(bb + c * P + 2 * d + 8 * t) = mx_codes8(R.w[i][t]);
            }
#pragma unroll
            for (int i = 0; i < St::SChunks; ++i) {
                const int j = tid + i * Threads;
                if (j < BN * KB) ss[j] = R.s[i];
            }
            if (tid < 16 * KB) sa[tid] = R.sa;
        };
        auto total = r4d_fp8::acc_zero();
        auto compute = [&](int s) {
            const int k0 = (s % ksn) * StageK;
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) {
                if (k0 + kb * 128 >= kc) break;
                const uint32_t exps = ss[mycol * KB + kb];
#pragma unroll
                for (int group = 0; group < 4; ++group) {
                    auto raw = r4d_fp8::acc_zero();
#pragma unroll
                    for (int k = kb * 128 + group * 32; k < kb * 128 + (group + 1) * 32; k += 16)
                        raw = r4d_fp8::mma(r4d_fp8::frag<StageK>(aa, 0, k, lane),
                                           r4d_fp8::frag<StageK>(bb, wave * 16, k, lane), raw);
                    const uint32_t exp = (exps >> (8 * group)) & 0xffu;
                    const float scale = __uint_as_float(exp == 255 ? 0x7fc00000u
                                                        : exp      ? (exp << 23)
                                                                   : 0x00400000u);
#pragma unroll
                    for (int row = 0; row < 8; ++row)
                        if (rv[row]) total[row] += raw[row] * sa[(rowbase + row) * KB + kb] * scale;
                }
            }
            if (s % ksn == ksn - 1) {
                const int n = colb + (s / ksn) * BN + mycol;
#pragma unroll
                for (int row = 0; row < 8; ++row) {
                    const int r = r0 + rowbase + row;
                    if (r < hi && n < m.N) y[(long)r * m.yr + n] = f32_to_bf16(total[row]);
                }
                total = r4d_fp8::acc_zero();
            }
        };
        St R;
        load(0, R);
        for (int s = 0; s < S; ++s) {
            store(R);
            __syncthreads();
            if (s + 1 < S) load(s + 1, R);
            compute(s);
            __syncthreads();
        }
    }
}

/* The small-M form at GLM's TP8 slices: expert-major with the same per-element sequence as the two
 * forms above, so their bytes. A verify pass of one stream routes 8 to 48 rows over as many
 * experts, and there those forms spend their time around the loads: an expert's block was a long
 * chain of short stages, and the blocks the LDS admitted came in rounds whose last one ran nearly
 * empty. This form shortens and thins the chain:
 *  - Block row y takes the y-th touched expert (a wave counts the non-empty runs by ballot), so the
 *    grid has min(experts, rows) + 1 rows rather than one an expert, most of them idle.
 *  - Weight codes stay packed in LDS and are decoded at the fragment read, which halves the
 *    weight tile; gate_up stages 512 of K, so a stage reads 256 bytes of each weight row where the
 *    256 stage read 128, and down reads its 128-byte rows whole over four column tiles.
 *  - Loads are unconditional from clamped addresses and masked at the store, so a store waits for
 *    its own stage and not the prefetch; weight rows load through global pointers, the barriers
 *    order LDS only, and a 128 of K reads its activation scales and eight fragment pairs before its
 *    WMMA chains. */
template <int StageK>
struct MxSmall {
    static constexpr int Threads = 128, BN = 64, KB = StageK / 128, PA = StageK + 8,
                         PB = StageK / 2 + 8;
    static constexpr int AChunks = StageK / Threads, WChunks = BN * StageK / 32 / Threads,
                         SChunks = (BN * KB + Threads - 1) / Threads;
    mx_u4 a[AChunks], w[WChunks];
    uint32_t s[SChunks];
    float sa;
    uint32_t ok;
};
template <int StageK, int ColTiles>
__global__ __launch_bounds__(128) void moe_mxfp4_small_kernel(
    const uint8_t* a, const float* as, const void* w0, const void* s0, const void* w1,
    const void* s1, const int* sorted, const int* offset, uint16_t* y, MxMeta m) {
    using St = MxSmall<StageK>;
    constexpr int BN = St::BN, Threads = St::Threads, KB = St::KB, PA = St::PA, PB = St::PB;
    static_assert(St::AChunks <= 8 && St::WChunks <= 16 && St::SChunks <= 4 &&
                      16 * KB <= Threads && BN * PB >= 4 * 513,
                  "stage chunks must fit the load mask and the offsets the weight tile");
    __shared__ __align__(16) uint8_t aa[16 * PA], bb[BN * PB];
    __shared__ uint32_t ss[BN * KB];
    __shared__ float sa[16 * KB];
    __shared__ int src[16];
    __shared__ int pick;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int colb = blockIdx.x * (BN * ColTiles);
    if ((int)blockIdx.y == (int)gridDim.y - 1) {
        const int r0 = offset[m.experts];
        for (long i = tid; i < (long)(m.T - r0) * BN * ColTiles; i += Threads) {
            const int n = colb + (int)(i % (BN * ColTiles));
            if (n < m.N) y[(long)(r0 + i / (BN * ColTiles)) * m.yr + n] = 0;
        }
        return;
    }
    {
        // The offsets in the weight tile, whose bytes are free until the first stage.
        int* offs = reinterpret_cast<int*>(bb);
#pragma unroll
        for (int q = 0; q < 5; ++q) {
            const int x = tid + q * Threads;
            if (x <= m.experts) offs[x] = offset[x];
        }
        lds_barrier();
        if (wave == 0) {
            int want = blockIdx.y, found = -1;
            for (int base = 0; base < m.experts; base += 32) {
                const int x = base + lane;
                const bool live = x < m.experts && offs[x + 1] > offs[x];
                const uint32_t mask = __builtin_amdgcn_ballot_w32(live);
                const int n = __builtin_popcount(mask);
                if (want < n) {
                    const uint32_t below = mask & ((1u << lane) - 1u);
                    found = base + __builtin_ctz(__builtin_amdgcn_ballot_w32(
                                       live && __builtin_popcount(below) == want));
                    break;
                }
                want -= n;
            }
            if (lane == 0) pick = found;
        }
        lds_barrier();
    }
    const int e = pick;
    if (e < 0) return;
    const int lo = offset[e], hi = offset[e + 1];
    const int cls = e & 1, kc = m.k[cls], pw = m.N / m.parts, cw = m.n[cls] / m.parts;
    const auto* w = entry(cls ? w1 : w0, m.wstep[cls], e / 2);
    const auto* ws = entry(cls ? s1 : s0, m.sstep[cls], e / 2);
    const int ksn = (kc + StageK - 1) / StageK, tiles = min(ColTiles, (m.N - colb + BN - 1) / BN),
              S = tiles * ksn;
    const int rowbase = (lane >> 4) * 8, mycol = wave * 16 + (lane & 15);
    auto wrow = [&](int n, bool& valid) {
        const int within = n % pw;
        valid = n < m.N && within < cw;
        return (n / pw) * cw + within;
    };
    for (int r0 = lo; r0 < hi; r0 += 16) {
        lds_barrier();
        if (tid < 16) {
            const int r = r0 + tid, flat = r < hi ? sorted[r] : -1;
            const int ar = flat < 0 ? -1 : m.sorted_a ? r : flat / m.topk;
            src[tid] = ar >= 0 && ar < m.Arows ? ar : -1;
        }
        lds_barrier();
        bool rv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) rv[j] = src[rowbase + j] >= 0;
        auto load = [&](int s, St& L) {
            const int col0 = colb + (s / ksn) * BN, k0 = (s % ksn) * StageK;
            uint32_t ok = 0;
#pragma unroll
            for (int i = 0; i < St::AChunks; ++i) {
                const int j = tid + i * Threads, r = j / (StageK / 16),
                          d = (j % (StageK / 16)) * 16, ar = src[r];
                const bool v = ar >= 0 && k0 + d < kc;
                __builtin_memcpy(&L.a[i], a + (v ? (long)ar * m.ar + k0 + d : 0), 16);
                ok |= (uint32_t)v << i;
            }
#pragma unroll
            for (int i = 0; i < St::WChunks; ++i) {
                const int j = tid + i * Threads, c = j / (StageK / 32),
                          d = (j % (StageK / 32)) * 16;
                bool valid;
                const int wn = wrow(col0 + c, valid);
                const bool v = valid && k0 + 2 * d < kc;
                const r4d_u32x4 q = mx_ldg16(w + (v ? (long)wn * m.wrow[cls] + k0 / 2 + d : 0));
                L.w[i] = mx_u4{ q[0], q[1], q[2], q[3] };
                ok |= (uint32_t)v << (8 + i);
            }
#pragma unroll
            for (int i = 0; i < St::SChunks; ++i) {
                const int j = tid + i * Threads, c = j / KB, kb = k0 + (j % KB) * 128;
                bool valid;
                const int wn = wrow(col0 + c, valid);
                const bool v = j < BN * KB && valid && kb < kc;
                L.s[i] = mx_ldg4(ws + (v ? (long)wn * m.srow[cls] + kb / 32 : 0));
                ok |= (uint32_t)v << (24 + i);
            }
            {
                const int ar = src[(tid / KB) & 15], kb = k0 + (tid % KB) * 128;
                const bool v = tid < 16 * KB && ar >= 0 && kb < kc;
                L.sa = as[v ? (long)ar * m.asr + kb / 128 : 0];
                ok |= (uint32_t)v << 28;
            }
            L.ok = ok;
        };
        auto store = [&](const St& L) {
#pragma unroll
            for (int i = 0; i < St::AChunks; ++i) {
                const int j = tid + i * Threads, r = j / (StageK / 16),
                          d = (j % (StageK / 16)) * 16;
                *reinterpret_cast<mx_u4*>(aa + r * PA + d) =
                    (L.ok >> i) & 1u ? L.a[i] : mx_u4{ 0, 0, 0, 0 };
            }
#pragma unroll
            for (int i = 0; i < St::WChunks; ++i) {
                const int j = tid + i * Threads, c = j / (StageK / 32),
                          d = (j % (StageK / 32)) * 16;
                const mx_u4 q = (L.ok >> (8 + i)) & 1u ? L.w[i] : mx_u4{ 0, 0, 0, 0 };
                *reinterpret_cast<uint2*>(bb + c * PB + d) = uint2{ q[0], q[1] };
                *reinterpret_cast<uint2*>(bb + c * PB + d + 8) = uint2{ q[2], q[3] };
            }
#pragma unroll
            for (int i = 0; i < St::SChunks; ++i) {
                const int j = tid + i * Threads;
                if (j < BN * KB) ss[j] = (L.ok >> (24 + i)) & 1u ? L.s[i] : 0x7f7f7f7fu;
            }
            if (tid < 16 * KB) sa[(tid % KB) * 16 + tid / KB] = (L.ok >> 28) & 1u ? L.sa : 0.f;
        };
        auto total = r4d_fp8::acc_zero();
        auto compute = [&](int s) {
            const int k0 = (s % ksn) * StageK;
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) {
                if (k0 + kb * 128 >= kc) break;
                const uint32_t exps = ss[mycol * KB + kb];
                float act[8];
#pragma unroll
                for (int row = 0; row < 8; ++row) act[row] = sa[kb * 16 + rowbase + row];
                r4d_fp8::i2v fa[8], fb[8];
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const int k = kb * 128 + 16 * t;
                    fa[t] = r4d_fp8::frag<StageK>(aa, 0, k, lane);
                    const uint2 v = mx_codes8(
                        *reinterpret_cast<const uint32_t*>(bb + mycol * PB + (k + rowbase) / 2));
                    fb[t] = r4d_fp8::i2v{ (int)v.x, (int)v.y };
                }
                r4d_fp8::f8v raw[4];
#pragma unroll
                for (int g = 0; g < 4; ++g)
                    raw[g] = r4d_fp8::mma(fa[2 * g], fb[2 * g], r4d_fp8::acc_zero());
#pragma unroll
                for (int g = 0; g < 4; ++g) raw[g] = r4d_fp8::mma(fa[2 * g + 1], fb[2 * g + 1], raw[g]);
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const uint32_t exp = (exps >> (8 * g)) & 0xffu;
                    const float scale = __uint_as_float(exp == 255 ? 0x7fc00000u
                                                        : exp      ? (exp << 23)
                                                                   : 0x00400000u);
#pragma unroll
                    for (int row = 0; row < 8; ++row)
                        if (rv[row]) total[row] += raw[g][row] * act[row] * scale;
                }
            }
            if (s % ksn == ksn - 1) {
                const int n = colb + (s / ksn) * BN + mycol;
#pragma unroll
                for (int row = 0; row < 8; ++row) {
                    const int r = r0 + rowbase + row;
                    if (r < hi && n < m.N) y[(long)r * m.yr + n] = f32_to_bf16(total[row]);
                }
                total = r4d_fp8::acc_zero();
            }
        };
        St L;
        load(0, L);
        for (int s = 0; s < S; ++s) {
            store(L);
            lds_barrier();
            if (s + 1 < S) load(s + 1, L);
            compute(s);
            lds_barrier();
        }
    }
}

int table(const RadTensor* t, uint32_t dtype, int bits, long experts, long n, long k, long* step,
          int* row) {
    if (!t || !t->data || t->rank != 3 || t->dtype != dtype || t->shape[0] < experts ||
        t->shape[1] != n || t->shape[2] != k || t->stride[2] != 1 || t->stride[1] < k ||
        (t->stride[1] * bits) % 8 || t->stride[1] * bits / 8 > INT_MAX)
        return RAD_E_SHAPE;
    if (t->stride[0] && (t->stride[0] < n * t->stride[1] || (t->stride[0] * bits) % 8))
        return RAD_E_STRIDE;
    *step = t->stride[0] * bits / 8;
    *row = (int)(t->stride[1] * bits / 8);
    return RAD_OK;
}
/* r4d_moe_mxfp4_pin_small: 0 the rule, 1 the small-M form at any GLM slice up to 384 routed rows
 * (TP4 as well as TP8, past the rule's bounds), 2 never, 3 as 1 with the K256 x4 tiles whatever
 * K. For r4d_selftest --perfmx to measure the form where the rule does not take it. */
int g_mx_small_pin = 0;
int body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 9) return RAD_E_INVAL;
    const auto *aq = rad_arg_in(a, 0), *as = rad_arg_in(a, 1), *sorted = rad_arg_in(a, 4),
               *off = rad_arg_in(a, 5);
    const auto* y = rad_arg_in(a, 8);
    if (!aq || !as || !sorted || !off || !y) return RAD_E_INVAL;
    MxMeta m{};
    const long n0 = r4d_geti_or(a, "N", 0), k0 = r4d_geti_or(a, "K", 0);
    const long nn[2] = { n0, r4d_geti_or(a, "N_odd", n0) },
               kk[2] = { k0, r4d_geti_or(a, "K_odd", k0) };
    const long ne = r4d_geti_or(a, "n_expert", 0), top = r4d_geti_or(a, "top_k", 0),
               parts = r4d_geti_or(a, "parts", 1);
    if (ne < 2 || ne > 512 || top < 1 || top > ne || parts < 1 || parts > INT_MAX ||
        r4d_geti_or(a, "group", 0) != 128)
        return RAD_E_SHAPE;
    const RadTensor* w[2] = { rad_arg_in(a, 2), rad_arg_in(a, 6) };
    const RadTensor* s[2] = { rad_arg_in(a, 3), rad_arg_in(a, 7) };
    for (int q = 0; q < 2; ++q) {
        if (nn[q] < 1 || nn[q] > INT_MAX - 16 || kk[q] < 128 || kk[q] > INT_MAX || kk[q] % 128 ||
            nn[q] % parts)
            return RAD_E_SHAPE;
        m.n[q] = (int)nn[q];
        m.k[q] = (int)kk[q];
        int rc =
            table(w[q], RAD_FP4E2M1, 4, (ne + 1 - q) / 2, nn[q], kk[q], &m.wstep[q], &m.wrow[q]);
        if (rc < 0) return rc;
        const uint32_t dt = s[q] ? s[q]->dtype : 0;
        if (dt != RAD_U8 && dt != RAD_E8M0) return RAD_E_DTYPE;
        rc = table(s[q], dt, 8, (ne + 1 - q) / 2, nn[q], kk[q] / 32, &m.sstep[q], &m.srow[q]);
        if (rc < 0) return rc;
    }
    m.N = (int)std::max(nn[0], nn[1]);
    m.K = (int)std::max(kk[0], kk[1]);
    if (aq->dtype != RAD_F8E4M3 || as->dtype != RAD_F32 || y->dtype != RAD_BF16 ||
        sorted->dtype != RAD_I32 || off->dtype != RAD_I32)
        return RAD_E_DTYPE;
    if (aq->rank != 2 || aq->shape[0] > INT_MAX || aq->shape[1] != m.K || aq->stride[1] != 1 ||
        aq->stride[0] < m.K || as->rank != 2 || as->shape[0] != aq->shape[0] ||
        as->shape[1] != m.K / 128 || as->stride[1] != 1 || as->stride[0] < m.K / 128 ||
        y->rank != 2 || y->shape[0] < 1 || y->shape[0] > INT_MAX || y->shape[1] != m.N ||
        y->stride[1] != 1 || y->stride[0] < m.N || sorted->rank != 1 || sorted->stride[0] != 1 ||
        sorted->shape[0] < y->shape[0] || off->rank != 1 || off->stride[0] != 1 ||
        off->shape[0] < ne + 1)
        return RAD_E_SHAPE;
    const char* ord = r4d_gets(a, "a_order");
    if (!ord || (std::strcmp(ord, "token") && std::strcmp(ord, "sorted"))) return RAD_E_INVAL;
    m.sorted_a = !std::strcmp(ord, "sorted");
    m.T = (int)y->shape[0];
    m.Arows = (int)aq->shape[0];
    if (m.T % top || m.Arows < (m.sorted_a ? m.T : m.T / top)) return RAD_E_SHAPE;
    m.experts = (int)ne;
    m.topk = (int)top;
    m.parts = (int)parts;
    m.ar = aq->stride[0];
    m.asr = as->stride[0];
    m.yr = y->stride[0];
    // Validated GLM projections at TP4 and TP8: retain the small/decode path and other models.
    const bool glm_prefill = ne == 288 && top == 8 && nn[0] == nn[1] && kk[0] == kk[1] &&
                             ((parts == 2 && (nn[0] == 1024 || nn[0] == 512) && kk[0] == 4096) ||
                              (parts == 1 && nn[0] == 4096 && (kk[0] == 512 || kk[0] == 256)));
    const bool tp8 = glm_prefill && (nn[0] == 512 || kk[0] == 256);
    /* The P64 form from a routed-row bound per projection, ahead of the expert-major band (its
     * down bound overlaps it): gate_up from 769 rows, where the prefill forms begin, down from
     * 20 rows an expert at GLM's 288. At GLM's TP8 slices only, the measured ones; a grid P64
     * cannot address takes the forms below. The tile follows the rows an expert, measured at the
     * TP8 slices (perfmx, cold): down BM32 BN64 K128 over four column tiles up to 3072 tokens
     * (2048: 647 against 831 us), above BM64 BN128 K128 x4 with the LDS output (4096: 1101
     * against 1607); gate_up BM32 BN64 K256 up to 1536 tokens (768: 563 against 660), BM64 BN64
     * K256 up to 3072 (2048: 941 against 995), above BM64 BN128 K128 with the LDS output (4096:
     * 1715 against 1915). */
    const bool resident = m.K <= 256, wide = m.T >= 24576;
    const long bm = wide || (!resident && m.T >= 12288) ? 64 : 32;
    if (tp8 && m.T >= (m.sorted_a ? 5760 : 769) && (m.T + bm - 1) / bm + m.experts <= 65535) {
        m.stage_out = wide;
        m.out16 = (reinterpret_cast<uintptr_t>(y->data) & 15) == 0 && m.yr % 8 == 0;
        const long jobs = (m.T + bm - 1) / bm + m.experts;
#define MX_P64(BM, BN, SK, CT, R)                                                                  \
    R4D_EMIT(stream, (moe_mxfp4_p64_kernel<BM, BN, SK, CT, R>),                                    \
             dim3(((long)m.N + BN * CT - 1) / (BN * CT), jobs), dim3(256), 0,                      \
             (const uint8_t*)aq->data, (const float*)as->data, w[0]->data, s[0]->data, w[1]->data, \
             s[1]->data, (const int*)sorted->data, (const int*)off->data, (uint16_t*)y->data, m)
        if (resident) return wide ? MX_P64(64, 128, 128, 4, true) : MX_P64(32, 64, 128, 4, true);
        if (wide) return MX_P64(64, 128, 128, 1, false);
        return bm == 64 ? MX_P64(64, 64, 256, 1, false) : MX_P64(32, 64, 256, 1, false);
#undef MX_P64
    }
    /* The small-M form at GLM's TP8 slices, ahead of the expert-major band: gate_up up to 384
     * routed rows, down up to 32, where it stops gaining (at 5 to 8 tokens it reads 3 to 5 %
     * slower than the staged form). At those slices on gfx1201 (perfmx, cold) it is 6 to 16 %
     * faster than the expert-major form for gate_up from 1 to 48 tokens, and 4 to 23 % for down
     * from 1 to 4. */
    if (g_mx_small_pin == 1 || g_mx_small_pin == 3
            ? glm_prefill && m.T <= 384
            : g_mx_small_pin != 2 && tp8 && m.T <= (m.K <= 256 ? 32 : 384)) {
        const long rows = std::min<long>(m.experts, m.T) + 1;
#define MX_SMALL(SK, CT)                                                                           \
    R4D_EMIT(stream, (moe_mxfp4_small_kernel<SK, CT>),                                             \
             dim3(((long)m.N + 64 * CT - 1) / (64 * CT), rows), dim3(128), 0,                      \
             (const uint8_t*)aq->data, (const float*)as->data, w[0]->data, s[0]->data,             \
             w[1]->data, s[1]->data, (const int*)sorted->data, (const int*)off->data,              \
             (uint16_t*)y->data, m)
        return m.K <= 256 || g_mx_small_pin == 3 ? MX_SMALL(256, 4) : MX_SMALL(512, 1);
#undef MX_SMALL
    }
    /* The expert-major form up to a routed-row bound per projection (token rows: gate_up, sorted
     * rows: down), measured at GLM's TP8 slices: gate_up up to 768 rows, where the prefill form
     * overtakes it, down up to 6143 rows (at 8192 the pipelined prefill form reads the slice
     * 2.2 times less often and wins by a quarter). Other shapes keep the decode band. A K of 256
     * takes 128 stages over four column tiles, a longer K the 256 stage over one. */
    if (m.T <= (tp8 ? (m.sorted_a ? 6143 : 768) : 511)) {
#define MX_EXPERTS(KERNEL, C)                                                                      \
    R4D_EMIT(stream, KERNEL, dim3(((long)m.N + 64 * C - 1) / (64 * C), m.experts + 1), dim3(128),  \
             0, (const uint8_t*)aq->data, (const float*)as->data, w[0]->data, s[0]->data,          \
             w[1]->data, s[1]->data, (const int*)sorted->data, (const int*)off->data,              \
             (uint16_t*)y->data, m)
        if (m.K <= 256) return MX_EXPERTS((moe_mxfp4_expert_staged_kernel<128, 4>), 4);
        return MX_EXPERTS((moe_mxfp4_expert_kernel<4, 256>), 1);
#undef MX_EXPERTS
    }
    /* From 512 routed rows (past the expert-major band) the GLM prefill form at GLM's slices, in
     * jobs of 32 rows of one expert; elsewhere the row groups, 16 sorted rows a block. */
    const bool pipe = glm_prefill && m.T >= 512;
    const int jobs = pipe ? (m.T + 31) / 32 + m.experts : (m.T + 15) / 16;
    if (jobs > 65535) return RAD_E_SHAPE;
    if (pipe)
        return R4D_EMIT(stream, moe_mxfp4_prefill_kernel, dim3(((long)m.N + 63) / 64, jobs),
                        dim3(256), 0, (const uint8_t*)aq->data, (const float*)as->data,
                        w[0]->data, s[0]->data, w[1]->data, s[1]->data, (const int*)sorted->data,
                        (const int*)off->data, (uint16_t*)y->data, m);
    return R4D_EMIT(stream, moe_mxfp4_fp8_kernel, dim3(((long)m.N + 15) / 16, jobs), dim3(32), 0,
                    (const uint8_t*)aq->data, (const float*)as->data, w[0]->data, s[0]->data,
                    w[1]->data, s[1]->data, (const int*)sorted->data, (const int*)off->data,
                    (uint16_t*)y->data, m);
}
}
extern "C" int r4d_rad_moe_mxfp4(const RadArgs* a, RadStream stream) { return body(a, stream); }
extern "C" void r4d_moe_mxfp4_pin_small(int form) { g_mx_small_pin = form; }
