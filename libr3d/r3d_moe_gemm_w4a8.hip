/* r3d_moe_gemm_w4a8.hip -- moe_gemm_q dtype w4a8: the grouped int4 GEMM over the expert-sorted rows.
 *
 *   y[T,N] = a[sorted_tok[i] / top_k, :] @ w[e(i)]^T, with a per-(row, 128-of-K) scale fold.
 *
 * The routed rows are in SORTED order (one row per (token, expert slot)); row i reads its
 * activation from `a[sorted_tok[i] / top_k]` (or a[i] under a_order "sorted") and its weight from
 * expert e(i)'s table, where e(i) is the expert owning sorted row i (the one whose [off, off') span
 * contains i). The experts are weight tables (RAD_OPD_WTAB, r3d_moe_tab.h): the even experts in
 * `w` / `w_scale`, the odd in `w_odd` / `w_odd_scale`, expert e at entry e/2 of class e%2, resolved
 * on the device (a pointer array or a stacked plane).
 *
 * The weight is two's-complement int4 (nibbles, -8..7) packed two per byte, with one bf16 scale per
 * (row, 128 of K); the output is bf16 in sorted order. Two arms, picked by the activation dtype:
 *   * int8 a  -> the native int8 wave-matrix (v_wmma_i32_16x16x16_iu8, exact i32 accumulate): the
 *     int4 weight is widened to int8, so no f16 rounding anywhere in the chain.
 *   * E4M3 a  -> the f16 wave-matrix (v_wmma_f32_16x16x16_f16, f32 accumulate): the E4M3 activation
 *     and the int4 weight are decoded exactly to f16 (libr4d's w4a8 form).
 * Each 128-wide block of K is accumulated raw, then folded by (sub * scale_a[src][b]) *
 * scale_w[e][n][b] in f32 -- the exact order the libref reference runs, so the int8 arm agrees bit
 * for bit.
 *
 * Decode band: at most 16 sorted rows per block (M * top_k <= 16); prefill finds no row and falls
 * back. STATUS: the w4a8 form is served (both activation arms). Remaining: the codebook NL forms
 * (w4nl64a8h / w4nl32a8h / w5nl64a8h, magnitude-table decode), a prefill path above 16 sorted rows,
 * and the SK/UW/RS tuning.
 */
#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include "r3d_codecs.h"
#include "r3d_fp8_dequant.h"
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

__device__ __forceinline__ _Float16 h16(uint32_t w) { return __builtin_bit_cast(_Float16, (uint16_t)w); }

// The block's rows' source tokens + the expert runs, shared by both arms. `a_sorted` selects the
// gather (token: sorted row i reads a[sorted[i]/top_k]; sorted: a[i]). -1 marks a dropped slot.
__device__ __forceinline__ void moe_w4a8_setup(
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

/* ------------------------------------------------------------------ int8 arm (i32 wave-matrix) */

template <int MR, int WV>
__global__ __launch_bounds__(WV * kWave) R3D_WAVES(WV) void moe_w4a8_i8_kernel(
    const int8_t* __restrict__ a, const float* __restrict__ a_scale,
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

    moe_w4a8_setup(sorted, off, r0, r1, tid, T, n_expert, top_k, a_sorted, s_tok, &s_nruns, s_rs,
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

            // Stage 16 rows x 128 i8 of the gathered activation; padding rows read the run's first.
            for (int i = (int)tid; i < MR * (kGroup / 4); i += (WV * kWave)) {
                const int row = i / (kGroup / 4), d = i % (kGroup / 4);
                const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                  : s_tok[rs - (int)r0];
                const int rr = (src < 0) ? 0 : src;
                s_a[(size_t)row * kPitch + d] = *(const uint32_t*)(a + (size_t)rr * a_ld + kt + 4 * d);
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

/* ------------------------------------------------------------- E4M3 arm (f16 wave-matrix) */

template <int MR, int WV>
__global__ __launch_bounds__(WV * kWave) R3D_WAVES(WV) void moe_w4a8_e4m3_kernel(
    const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, R3dMoeTabs tw,
    uint16_t* __restrict__ y, int T, int N, int K, int a_ld, int as_ld, int y_ld,
    int n_expert, int top_k, int a_sorted)
{
    __shared__ __align__(16) uint32_t s_a[16 * kPitch];
    __shared__ __align__(16) uint32_t s_w[kWaves * 16 * kPitch];   // the block's 128 cols x 128 K
    __shared__ __align__(16) uint16_t s_tbl[R3D_E4M3_TBL];
    __shared__ int32_t s_tok[16];
    __shared__ int32_t s_nruns;
    __shared__ int32_t s_rs[16], s_re[16], s_e[16];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wave = tid / kWave;
    const uint32_t r = lane & 15;

    r3d_e4m3_f16_init(s_tbl, (int)tid, kThreads);

    const uint32_t r0 = (uint32_t)blockIdx.y * MR;
    const uint32_t n0 = (uint32_t)blockIdx.x * kWaves * 16;
    const uint32_t r1 = (r0 + MR < (uint32_t)T) ? r0 + MR : (uint32_t)T;

    moe_w4a8_setup(sorted, off, r0, r1, tid, T, n_expert, top_k, a_sorted, s_tok, &s_nruns, s_rs,
                   s_re, s_e);
    const int nruns = s_nruns;

    const uint32_t nc = n0 + wave * 16 + r;   // this wave's output column
    const int n_real = (int)(nc < (uint32_t)N ? nc : (uint32_t)N - 1);

    for (int run = 0; run < nruns; ++run) {
        const int rs = s_rs[run], re = s_re[run], e = s_e[run];
        v8float acc = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        if (e >= 0) {
        const int cls = e & 1, e2 = e >> 1;
        const uint8_t* wbase = r3d_tab_at(tw.w[cls], e2);
        const int w_ld = tw.w[cls].ld;
        const uint16_t* wsb = (const uint16_t*)r3d_tab_at(tw.s[cls], e2);
        const int ws_ld = tw.s[cls].ld;

        for (int b = 0; b < K / kGroup; ++b) {
            const int kt = b * kGroup;

            // Stage 16 rows x 128 E4M3 of the gathered activation; padding rows read the run's first.
            for (int i = (int)tid; i < MR * (kGroup / 4); i += (WV * kWave)) {
                const int row = i / (kGroup / 4), d = i % (kGroup / 4);
                const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                  : s_tok[rs - (int)r0];
                const int rr = (src < 0) ? 0 : src;
                s_a[(size_t)row * kPitch + d] = *(const uint32_t*)(a + (size_t)rr * a_ld + kt + 4 * d);
            }
            // Stage the block's 128 cols x 128 K of the int4 weight (packed nibbles).
            for (int i = (int)tid; i < kWaves * 16 * (kGroup / 8); i += (WV * kWave)) {
                const int col = i / (kGroup / 8), d = i % (kGroup / 8);
                const int nn = (int)(n0 + col < (uint32_t)N ? n0 + col : (uint32_t)N - 1);
                s_w[(size_t)col * kPitch + d] =
                    *(const uint32_t*)(wbase + (size_t)nn * w_ld / 2 + kt / 2 + 4 * d);
            }
            group_barrier();

            const uint32_t* ap = s_a + (size_t)r * kPitch;
            const uint32_t* wp = s_w + (size_t)(wave * 16 + r) * kPitch;   // this wave's 16 cols
            v8float blk = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
#pragma unroll
            for (int k = 0; k < kGroup; k += 16) {
                uint32_t ain[4], af[8], bf[8];
#pragma unroll
                for (int g = 0; g < 4; ++g) ain[g] = ap[k / 4 + g];
                r3d_e4m3x16_decode<0>(s_tbl, ain, af);
                // Decode the 16 packed int4 weights of this k tile to f16, two per word (as af[8]).
                const uint2 wq = *(const uint2*)(wp + k / 8);   // 8 bytes = 16 nibbles
#pragma unroll
                for (int g = 0; g < 8; ++g) {
                    const uint32_t byte = (g < 4) ? ((wq.x >> (8 * g)) & 0xFFu)
                                                     : ((wq.y >> (8 * (g - 4))) & 0xFFu);
                    const _Float16 h0 = (_Float16)(float)nib_i8(byte & 0xFu);        // weight 2g
                    const _Float16 h1 = (_Float16)(float)nib_i8((byte >> 4) & 0xFu); // weight 2g+1
                    bf[g] = (uint32_t)__builtin_bit_cast(uint16_t, h0)
                          | ((uint32_t)__builtin_bit_cast(uint16_t, h1) << 16);
                }
                v16half fa, fb;
#pragma unroll
                for (int g = 0; g < 8; ++g) { fa[2 * g] = h16(af[g] & 0xffffu); fa[2 * g + 1] = h16(af[g] >> 16); }
#pragma unroll
                for (int g = 0; g < 8; ++g) { fb[2 * g] = h16(bf[g] & 0xffffu); fb[2 * g + 1] = h16(bf[g] >> 16); }
                blk = wmma_f16(fa, fb, blk);
            }
            group_barrier();

            // Fold the 128-block's scales: (sub * scale_a[src][b]) * scale_w[e][n][b], libref's order.
            const float sw = bf16_f32(wsb[(size_t)n_real * ws_ld + b]);
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                const int row = (int)(g * 2 + (lane >> 4));
                const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                  : s_tok[rs - (int)r0];
                const float sa = a_scale[(size_t)(src < 0 ? 0 : src) * as_ld + b];
                acc[g] += blk[g] * sa * sw;
            }
        }

        }
        // Store only the run's rows.
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int row = (int)(g * 2 + (lane >> 4));
            const int m = (int)(r0 + row);
            if (rs <= m && m < re && nc < (uint32_t)N) y[(size_t)m * y_ld + nc] = f32_to_bf16(acc[g]);
        }
    }
}

}  // namespace

extern "C" {

/* moe_gemm_q dtype w4a8: the grouped int4 GEMM. The arm is picked by the activation dtype (int8 ->
 * the exact i32 wave-matrix; E4M3 -> the f16 wave-matrix, libr4d's form). The weight scale is bf16
 * and the output bf16 in both arms. */
int r3d_rad_moe_gemm_w4a8(const RadArgs* a, RadStream s)
{
    R3dMoeCall c;
    int rc = r3d_moe_call(a, true, &c);
    if (rc != RAD_OK) return rc;
    if (!c.a_scale) return R3D_NO(RAD_E_INVAL);
    if (c.a->dtype != RAD_I8 && c.a->dtype != RAD_F8E4M3) return R3D_NO(RAD_E_DTYPE);
    if (c.a_scale->dtype != RAD_F32 || c.y->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (c.K % kGroup) return R3D_NO(RAD_E_SHAPE);   // the N tail clamps

    R3dMoeTabs tw;
    rc = r3d_moe_q_tabs(c, RAD_I4, 4, RAD_BF16, 16, c.N, c.K / kGroup, &tw);
    if (rc != RAD_OK) return rc;
    const int a_ld = (int)c.a->stride[0], as_ld = (int)c.a_scale->stride[0];
    const int y_ld = (int)c.y->stride[0];


    const int32_t* ps = (const int32_t*)r3d_p(c.sorted);
    const int32_t* po = (const int32_t*)r3d_p(c.off);
    const float* psa = (const float*)r3d_p(c.a_scale);
    uint16_t* py = (uint16_t*)r3d_wp(c.y);
    const int mrows = r3d_geti_or(a, "mrows", 16);
    const int wv = r3d_geti_or(a, "wv", 8);
    const int ctile = wv * 16;
    dim3 grid((unsigned)((c.N + ctile - 1) / ctile), (unsigned)((c.T + mrows - 1) / mrows)), blk(wv * kWave);
    if (c.a->dtype == RAD_I8) {
        const int8_t* pa8 = (const int8_t*)r3d_p(c.a);
        if (mrows == 8 && wv == 4)
            moe_w4a8_i8_kernel<8,4><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted);
        else if (mrows == 8)
            moe_w4a8_i8_kernel<8,8><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted);
        else if (wv == 4)
            moe_w4a8_i8_kernel<16,4><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted);
        else
            moe_w4a8_i8_kernel<16,8><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted);
    } else {
        const uint8_t* pa8 = (const uint8_t*)r3d_p(c.a);
        if (mrows == 8 && wv == 4)
            moe_w4a8_e4m3_kernel<8,4><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted);
        else if (mrows == 8)
            moe_w4a8_e4m3_kernel<8,8><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted);
        else if (wv == 4)
            moe_w4a8_e4m3_kernel<16,4><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted);
        else
            moe_w4a8_e4m3_kernel<16,8><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted);
    }
    return (int)hipGetLastError();
}

}  // extern "C"
