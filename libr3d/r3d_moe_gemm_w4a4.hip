/* r3d_moe_gemm_w4a4.hip -- moe_gemm_w4a4: the grouped int4xint4 GEMM over the expert-sorted rows.
 *
 *   y[T,N] = a[sorted_tok[i] / top_k, :] @ w[e(i)]^T, with a per-(row, 128-of-K) scale fold.
 *
 * Both the activation and the experts are two's-complement int4 (nibbles, -8..7) packed two per
 * byte. The activation has one f32 scale per (row, 128 of K); each expert has one bf16 scale per
 * (row, 128 of K); the output is bf16 in sorted order. The routed rows are in SORTED order (one row
 * per (token, expert slot)); row i reads its activation from `a[sorted_tok[i] / top_k]` (or a[i]
 * under a_order "sorted") and its weight from expert e(i)'s table (the parity tables, r3d_moe_tab.h).
 *
 * The kernel widens both the int4 activation and the int4 weight to int8 and runs the native int8
 * wave-matrix (v_wmma_i32_16x16x16_iu8, exact i32 accumulate): no f16 rounding anywhere in the
 * chain. Each 128-wide block of K is accumulated raw, then folded by (sub * scale_a[src][b]) *
 * scale_w[e][n][b] in f32 -- the exact order the libref reference runs, so it agrees bit for bit.
 *
 * Decode band: at most 16 sorted rows per block (M * top_k <= 16); prefill finds no row and falls
 * back. The activation stride a_ld is in BYTES (K/2 per row: K int4 nibbles).
 */
#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include "r3d_codecs.h"
#include "r3d_moe_tab.h"

namespace {

constexpr int kThreads = 256;           // eight waves
constexpr int kWaves   = kThreads / kWave;
constexpr int kGroup   = 128;           // a scale block: one k tile, eight wmma steps
constexpr int kPitch   = 33;            // dwords a staged row: 33 mod 32 = 1, bank-free

__device__ __forceinline__ int32_t nib_i8(uint32_t nib)
{
    return (nib & 8u) ? (int32_t)nib - 16 : (int32_t)(nib & 7u);
}

__device__ __forceinline__ float bf16_f32(uint16_t h) { return __builtin_bit_cast(float, uint32_t(h) << 16); }

// The block's rows' source tokens + the expert runs. `a_sorted` selects the gather (token: sorted
// row i reads a[sorted[i]/top_k]; sorted: a[i]). -1 marks a dropped slot.
__device__ __forceinline__ void moe_w4a4_setup(
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, uint32_t r0, uint32_t r1,
    uint32_t tid, int T, int n_expert, int top_k, int a_sorted, int32_t* s_tok, int32_t* s_nruns,
    int32_t* s_rs, int32_t* s_re, int32_t* s_e)
{
    if (tid < 16) {
        const uint32_t i = r0 + tid;
        const int32_t f = (i < (uint32_t)T) ? sorted[i] : -1;
        s_tok[tid] = (f < 0) ? -1 : (a_sorted ? (int32_t)i : f / top_k);
    }
    if (tid == 0) {
        int nruns = 0, i = (int)r0;
        while (i < (int)r1) {
            int32_t ee = -1;
            for (int32_t x = 0; x < n_expert; ++x) if (off[x] <= i && i < off[x + 1]) ee = x;
            const int re = (ee < 0) ? (int)r1 : (int)(off[ee + 1] < r1 ? off[ee + 1] : r1);
            s_rs[nruns] = i; s_re[nruns] = re; s_e[nruns] = ee;
            ++nruns; i = re;
        }
        *s_nruns = nruns;
    }
    group_barrier();
}

/* ------------------------------------------------------- int4-activation arm (i32 wave-matrix) */

template <int MR, int WV>
__global__ __launch_bounds__(WV * kWave) R3D_WAVES(WV) void moe_w4a4_i8_kernel(
    const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, R3dMoeTabs tw,
    uint16_t* __restrict__ y, int T, int N, int K, int a_ld, int as_ld, int y_ld,
    int n_expert, int top_k, int a_sorted)
{
    __shared__ __align__(16) uint32_t s_a[MR * kPitch];
    __shared__ int32_t s_tok[MR];
    __shared__ int32_t s_nruns;
    __shared__ int32_t s_rs[MR], s_re[MR], s_e[MR];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wave = tid / kWave;
    const uint32_t r = lane & 15;

    const uint32_t r0 = (uint32_t)blockIdx.y * MR;
    const uint32_t ncol = ((uint32_t)blockIdx.x * WV + wave) * 16;   // the wave's 16 columns
    const uint32_t r1 = (r0 + MR < (uint32_t)T) ? r0 + MR : (uint32_t)T;

    moe_w4a4_setup(sorted, off, r0, r1, tid, T, n_expert, top_k, a_sorted, s_tok, &s_nruns, s_rs,
                   s_re, s_e);
    const int nruns = s_nruns;

    const uint32_t nc = ncol + r;
    const int n_real = (int)(nc < (uint32_t)N ? nc : (uint32_t)N - 1);

    for (int run = 0; run < nruns; ++run) {
        const int rs = s_rs[run], re = s_re[run], e = s_e[run];
        v8float acc = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        if (e >= 0) {
        const int cls = e & 1, e2 = e >> 1;
        const uint8_t* wrow = r3d_tab_at(tw.w[cls], e2) + (size_t)n_real * tw.w[cls].ld / 2;
        const uint16_t* wsb = (const uint16_t*)r3d_tab_at(tw.s[cls], e2);
        const int ws_ld = tw.s[cls].ld;

        for (int b = 0; b < K / kGroup; ++b) {
            const int kt = b * kGroup;

            // Stage 16 rows x 128 i4 (as 128 i8) of the gathered activation. Each iteration reads
            // 2 bytes = 4 nibbles and widens them to 4 int8 (one dword), so dword d holds i8 [4d..4d+3]
            // and the wave-matrix A-fragment's k/4 indexing lands on the right dword; padding rows read
            // the run's first.
            for (int i = (int)tid; i < MR * (kGroup / 4); i += (WV * kWave)) {
                const int row = i / (kGroup / 4), d = i % (kGroup / 4);
                const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                   : s_tok[rs - (int)r0];
                const int rr = (src < 0) ? 0 : src;
                const uint32_t w = *(const uint16_t*)(a + (size_t)rr * a_ld + (kt + 4 * d) / 2);
                int32_t v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t byte = (w >> (8 * (j / 2))) & 0xFFu;
                    v[j] = nib_i8((j & 1) ? (byte >> 4) : (byte & 0xFu));
                }
                s_a[(size_t)row * kPitch + d] = (uint32_t)(v[0] & 0xFF) | (uint32_t)(v[1] & 0xFF) << 8 |
                                                (uint32_t)(v[2] & 0xFF) << 16 | (uint32_t)(v[3] & 0xFF) << 24;
            }
            group_barrier();
            const uint32_t* ap = s_a + (size_t)r * kPitch;
            v8int sub = v8int{ 0, 0, 0, 0, 0, 0, 0, 0 };
#pragma unroll
            for (int k = 0; k < kGroup; k += 16) {
                const r3d_u32x4 aq = *(const r3d_u32x4*)(ap + k / 4);
                const uint2 wp = *(const uint2*)(wrow + (kt + k) / 2);   // 8 bytes = 16 nibbles
                int32_t v[16];
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const int byt = j / 2;
                    const bool high = (j & 1);
                    const uint32_t byte = (byt < 4) ? ((wp.x >> (8 * byt)) & 0xFFu)
                                                      : ((wp.y >> (8 * (byt - 4))) & 0xFFu);
                    v[j] = nib_i8(high ? (byte >> 4) : (byte & 0xFu));
                }
                v4int bf{ (int)((uint32_t)(v[0] & 0xFF) | (uint32_t)(v[1] & 0xFF) << 8 |
                            (uint32_t)(v[2] & 0xFF) << 16 | (uint32_t)(v[3] & 0xFF) << 24),
                          (int)((uint32_t)(v[4] & 0xFF) | (uint32_t)(v[5] & 0xFF) << 8 |
                            (uint32_t)(v[6] & 0xFF) << 16 | (uint32_t)(v[7] & 0xFF) << 24),
                          (int)((uint32_t)(v[8] & 0xFF) | (uint32_t)(v[9] & 0xFF) << 8 |
                            (uint32_t)(v[10] & 0xFF) << 16 | (uint32_t)(v[11] & 0xFF) << 24),
                          (int)((uint32_t)(v[12] & 0xFF) | (uint32_t)(v[13] & 0xFF) << 8 |
                            (uint32_t)(v[14] & 0xFF) << 16 | (uint32_t)(v[15] & 0xFF) << 24) };
                sub = wmma_i8(v4int{ (int)aq.x, (int)aq.y, (int)aq.z, (int)aq.w }, bf, sub);
            }

            // Fold the 128-block's scales: (sub * scale_a[src][b]) * scale_w[e][n][b], libref's order.
            const float sw = bf16_f32(wsb[(size_t)n_real * ws_ld + b]);
#pragma unroll
            for (int g2 = 0; g2 < 8; ++g2) {
                const int row = (int)(g2 * 2 + (lane >> 4));
                const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                   : s_tok[rs - (int)r0];
                const float sa = a_scale[(size_t)(src < 0 ? 0 : src) * as_ld + b];
                acc[g2] += (float)sub[g2] * sa * sw;
            }
            group_barrier();
        }

        }
        // Store only the run's rows.
#pragma unroll
        for (int g2 = 0; g2 < 8; ++g2) {
            const int row = (int)(g2 * 2 + (lane >> 4));
            const int m = (int)(r0 + row);
            if (rs <= m && m < re && nc < (uint32_t)N) y[(size_t)m * y_ld + nc] = f32_to_bf16(acc[g2]);
        }
    }
}

}  // namespace

extern "C" {

/* moe_gemm_w4a4: the grouped int4xint4 GEMM. The activation and the experts are both two's-
 * complement int4; the exact i32 wave-matrix widens both to int8. The weight scale is bf16 and the
 * output bf16. */
int r3d_rad_moe_gemm_w4a4(const RadArgs* a, RadStream s)
{
    R3dMoeCall c;
    int rc = r3d_moe_call(a, true, &c);
    if (rc != RAD_OK) return rc;
    if (!c.a_scale) return R3D_NO(RAD_E_INVAL);
    if (c.a->dtype != RAD_I4) return R3D_NO(RAD_E_DTYPE);
    if (c.a_scale->dtype != RAD_F32 || c.y->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (c.K % kGroup) return R3D_NO(RAD_E_SHAPE);   // the N tail clamps

    R3dMoeTabs tw;
    rc = r3d_moe_q_tabs(c, RAD_I4, 4, RAD_BF16, 16, c.N, c.K / kGroup, &tw);
    if (rc != RAD_OK) return rc;
    const int a_ld = (int)c.a->stride[0], as_ld = (int)c.a_scale->stride[0];
    const int y_ld = (int)c.y->stride[0];

    const int mrows = r3d_geti_or(a, "mrows", 16);
    const int wv = r3d_geti_or(a, "wv", 8);
    const int ctile = wv * 16;
    dim3 grid((unsigned)((c.N + ctile - 1) / ctile), (unsigned)((c.T + mrows - 1) / mrows)), blk(wv * kWave);
    const int32_t* ps = (const int32_t*)r3d_p(c.sorted);
    const int32_t* po = (const int32_t*)r3d_p(c.off);
    const float* psa = (const float*)r3d_p(c.a_scale);
    uint16_t* py = (uint16_t*)r3d_wp(c.y);
    const uint8_t* pa = (const uint8_t*)r3d_p(c.a);
    if (mrows == 8 && wv == 4)
        moe_w4a4_i8_kernel<8,4><<<grid, blk, 0, (hipStream_t)s>>>(pa, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted);
    else if (mrows == 8)
        moe_w4a4_i8_kernel<8,8><<<grid, blk, 0, (hipStream_t)s>>>(pa, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted);
    else if (wv == 4)
        moe_w4a4_i8_kernel<16,4><<<grid, blk, 0, (hipStream_t)s>>>(pa, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted);
    else
        moe_w4a4_i8_kernel<16,8><<<grid, blk, 0, (hipStream_t)s>>>(pa, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted);
    return (int)hipGetLastError();
}


}  // extern "C"
