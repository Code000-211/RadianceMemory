/* r3d_moe_gemm_mxfp4a8.hip -- moe_gemm_mxfp4a8: the grouped GEMM with canonical MXFP4 experts.
 *
 *   y[i, n] = sum_k a[tok(i), k] * w_e[n, k],   e = the expert of sorted row i
 *
 * The activation is E4M3 with one f32 scale per (row, 128 of K). The weight is canonical MXFP4:
 * packed E2M1 codes (two a byte) with one E8M0 scale per 32 of K. E2M1's eight magnitudes
 * {0, .5, 1, 1.5, 2, 3, 4, 6} are exactly representable in E4M3, so each weight code is decoded to
 * an E4M3 byte (the v_perm_b32 unpack, the exact d=0 row) and both operands run on the f16
 * wave-matrix (f32 accumulate). Each 32-wide E8M0 scale folds into the f32 accumulator per group.
 *
 * One block per 16 expert-sorted rows, walking the runs (contiguous rows per expert) in its tile,
 * the same run-walk / gather structure as moe_gemm_q. Expert e is entry e/2 of the even (`w` /
 * `w_scale`) or odd (`w_odd` / `w_odd_scale`) weight tables (r3d_moe_tab.h). K is a multiple of
 * 128; the N tail clamps.
 */
#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_launch.h"
#include "r3d_fp8_dequant.h"
#include "r3d_moe_tab.h"

namespace {

constexpr int kThreads = 256;           // eight waves
constexpr int kWaves   = kThreads / kWave;
constexpr int kNBlock  = 128;           // a block is 16 rows x 128 cols: eight waves x 16 cols
constexpr int kGroup   = 128;           // a k tile: four 32-wide E8M0 groups
constexpr int kAPitch  = 33;            // dwords a staged activation row: 32 (128 E4M3) + 1
constexpr int kWPitch  = 17;            // dwords a staged weight row: 16 (128 E2M1 = 64 B) + 1

/* Eight packed E2M1 codes (four bytes, element 2j in the low nibble of byte j) -> eight E4M3 bytes
 * in k order (word 0 = elements 0..3, word 1 = 4..7). The exact d=0 row of the folded-magnitude
 * table (R4D_MXFP4_MAG0); E2M1's magnitudes are a subset of E4M3, so this is lossless. */
__device__ __forceinline__ void e2m1x8_to_e4m3x8(uint32_t wv, uint32_t* lo, uint32_t* hi)
{
    const uint32_t t0 = 0x3c383000u, t1 = 0x4c484440u;
    const uint32_t ev = wv & 0x0F0F0F0Fu;          // low nibble of each byte  -> even elements
    const uint32_t od = (wv >> 4) & 0x0F0F0F0Fu;   // high nibble              -> odd elements
    const uint32_t be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
    const uint32_t bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
    *lo = __builtin_amdgcn_perm(bo, be, 0x05010400u);
    *hi = __builtin_amdgcn_perm(bo, be, 0x07030602u);
}

/* An E8M0 byte to f32: the byte is the f32 exponent field (bias 127). Code 255 is NaN, code 0 is
 * 2^-127 (a representable subnormal). */
__device__ __forceinline__ float e8m0_to_f32(uint32_t e)
{
    return __uint_as_float(e == 255 ? 0x7fc00000u : e ? (e << 23) : 0x00200000u);
}

__device__ __forceinline__ _Float16 h16(uint32_t w) { return __builtin_bit_cast(_Float16, (uint16_t)w); }

__global__ __launch_bounds__(kThreads) R3D_WAVES(kWaves) void moe_mx_kernel(
    const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, R3dMoeTabs tw,
    uint16_t* __restrict__ y, int T, int N, int K, int a_ld, int as_ld, int y_ld,
    int n_expert, int top_k, int a_sorted)
{
    __shared__ __align__(16) uint32_t s_a[16 * kAPitch];
    __shared__ __align__(16) uint32_t s_w[kNBlock * kWPitch];
    __shared__ __align__(16) uint16_t s_tbl[R3D_E4M3_TBL];
    __shared__ int32_t s_tok[16];
    __shared__ int32_t s_nruns;
    __shared__ int32_t s_rs[16], s_re[16], s_e[16];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wv = tid / kWave;          // the wave index (0..7): owns cols wv*16..wv*16+16
    const uint32_t r = lane & 15;

    r3d_e4m3_f16_init(s_tbl, (int)tid, kThreads);   // the E4M3 -> f16 table

    const uint32_t r0 = (uint32_t)blockIdx.y * 16;   // the block's sorted rows
    const uint32_t n0 = (uint32_t)blockIdx.x * kNBlock;
    const uint32_t r1 = (r0 + 16 < (uint32_t)T) ? r0 + 16 : (uint32_t)T;

    // The block's rows' source rows of `a` (the gather, or the row itself under a_order
    // "sorted"). -1 marks a dropped slot.
    if (tid < 16) {
        const uint32_t i = r0 + tid;
        const int32_t f = (i < (uint32_t)T) ? sorted[i] : -1;
        s_tok[tid] = (f < 0) ? -1 : (a_sorted ? (int32_t)i : f / top_k);
    }
    // The runs in [r0, r1): each a contiguous set of rows for one expert. The block walks them.
    if (tid == 0) {
        int nruns = 0, i = (int)r0;
        while (i < (int)r1) {
            int32_t ee = -1;
            for (int32_t x = 0; x < n_expert; ++x) if (off[x] <= i && i < off[x + 1]) ee = x;
            const int re = (ee < 0) ? (int)r1 : (int)(off[ee + 1] < r1 ? off[ee + 1] : r1);
            s_rs[nruns] = i; s_re[nruns] = re; s_e[nruns] = ee;
            ++nruns; i = re;
        }
        s_nruns = nruns;
    }
    group_barrier();
    const int nruns = s_nruns;

    const uint32_t nc = n0 + wv * 16 + r;   // this wave's output column
    const int n_real = (int)(nc < (uint32_t)N ? nc : (uint32_t)N - 1);

    // One GEMM per run: the run's rows read the gathered activation and expert e's weight. Rows the
    // run does not own are padding (reading the run's first row) and their output is dropped.
    for (int run = 0; run < nruns; ++run) {
        const int rs = s_rs[run], re = s_re[run], e = s_e[run];
        v8float acc = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        if (e >= 0) {
        // Expert e is entry e/2 of class e%2's tables: E2M1 codes two a byte, one E8M0 exponent
        // per (row, 32 of K).
        const int cls = e & 1, e2 = e >> 1;
        const uint8_t* wbase = r3d_tab_at(tw.w[cls], e2);
        const int w_ld_b = tw.w[cls].ld / 2;
        const uint8_t* wsb = r3d_tab_at(tw.s[cls], e2);
        const int ws_ld = tw.s[cls].ld;

        for (int b = 0; b < K / kGroup; ++b) {
            const int kt = b * kGroup;

            // Stage 16 rows x 128 E4M3 of the gathered activation; padding rows read the run's first.
            for (int i = (int)tid; i < 16 * (kGroup / 4); i += kThreads) {
                const int row = i / (kGroup / 4), d = i % (kGroup / 4);
                const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                 : s_tok[rs - (int)r0];
                const int rr = (src < 0) ? 0 : src;
                s_a[(size_t)row * kAPitch + d] = *(const uint32_t*)(a + (size_t)rr * a_ld + kt + 4 * d);
            }
            // Stage 128 cols x 128 E2M1 of the expert's weight (64 bytes = 16 dwords a row).
            for (int i = (int)tid; i < kNBlock * (kGroup / 8); i += kThreads) {
                const int col = i / (kGroup / 8), d = i % (kGroup / 8);
                const int nn = (int)(n0 + col < (uint32_t)N ? n0 + col : (uint32_t)N - 1);
                s_w[(size_t)col * kWPitch + d] =
                    *(const uint32_t*)(wbase + (size_t)nn * w_ld_b + kt / 2 + 4 * d);
            }
            group_barrier();

            const uint32_t* ap = s_a + (size_t)r * kAPitch;
            const uint32_t* wp = s_w + (size_t)(wv * 16 + r) * kWPitch;   // this wave's 16 cols
            // The weight scale for this lane's column, per 32-of-K group.
            const int wn = n_real;
#pragma unroll
            for (int g32 = 0; g32 < 4; ++g32) {
                v8float blk = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
#pragma unroll
                for (int k = g32 * 32; k < g32 * 32 + 32; k += 16) {
                    uint32_t af[8], bf[8];
                    // 16 E4M3 (four dwords) -> 16 f16 (eight dwords).
                    r3d_e4m3x8_decode<0>(s_tbl, uint2{ ap[k / 4], ap[k / 4 + 1] }, af);
                    r3d_e4m3x8_decode<0>(s_tbl, uint2{ ap[k / 4 + 2], ap[k / 4 + 3] }, af + 4);
                    // 16 E2M1 (two dwords) -> 16 E4M3 (four dwords) -> 16 f16 (eight dwords).
                    uint32_t e4lo, e4hi, e4lo2, e4hi2;
                    e2m1x8_to_e4m3x8(wp[k / 8], &e4lo, &e4hi);
                    e2m1x8_to_e4m3x8(wp[k / 8 + 1], &e4lo2, &e4hi2);
                    r3d_e4m3x8_decode<0>(s_tbl, uint2{ e4lo, e4hi }, bf);
                    r3d_e4m3x8_decode<0>(s_tbl, uint2{ e4lo2, e4hi2 }, bf + 4);
                    v16half fa, fb;
#pragma unroll
                    for (int q = 0; q < 8; ++q) { fa[2 * q] = h16(af[q] & 0xffffu); fa[2 * q + 1] = h16(af[q] >> 16); }
#pragma unroll
                    for (int q = 0; q < 8; ++q) { fb[2 * q] = h16(bf[q] & 0xffffu); fb[2 * q + 1] = h16(bf[q] >> 16); }
                    blk = wmma_f16(fa, fb, blk);
                }
                group_barrier();
                // Fold the g32-group's scales: the weight E8M0 (per 32) and the activation f32 (per 128).
                const float sw = e8m0_to_f32((uint32_t)wsb[(size_t)wn * ws_ld + b * 4 + g32]);
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

        }
        // Store only the run's rows, as bf16.
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int row = (int)(g * 2 + (lane >> 4));
            const int m = (int)(r0 + row);
            if (rs <= m && m < re && nc < (uint32_t)N)
                y[(size_t)m * y_ld + nc] = f32_to_bf16(acc[g]);
        }
    }
}

}  // namespace

extern "C" {

int r3d_rad_moe_gemm_mxfp4a8(const RadArgs* a, RadStream s)
{
    R3dMoeCall c;
    int rc = r3d_moe_call(a, true, &c);
    if (rc != RAD_OK) return rc;
    if (!c.a_scale) return R3D_NO(RAD_E_INVAL);
    if (c.a->dtype != RAD_F8E4M3 || c.a_scale->dtype != RAD_F32 || c.y->dtype != RAD_BF16)
        return R3D_NO(RAD_E_DTYPE);
    if (c.K % kGroup) return R3D_NO(RAD_E_SHAPE);   // the N tail clamps
    R3dMoeTabs tw;
    rc = r3d_moe_q_tabs(c, RAD_FP4E2M1, 4, RAD_E8M0, 8, c.N, c.K / 32, &tw);
    if (rc != RAD_OK) return rc;
    if ((tw.w[0].ld | tw.w[1].ld) & 1) return R3D_NO(RAD_E_STRIDE);   // whole bytes a row
    const int a_ld = (int)c.a->stride[0], as_ld = (int)c.a_scale->stride[0];
    const int y_ld = (int)c.y->stride[0];

    dim3 grid((unsigned)((c.N + kNBlock - 1) / kNBlock), (unsigned)((c.T + 15) / 16)), blk(kThreads);
    moe_mx_kernel<<<grid, blk, 0, (hipStream_t)s>>>(
        (const uint8_t*)r3d_p(c.a), (const float*)r3d_p(c.a_scale), (const int32_t*)r3d_p(c.sorted),
        (const int32_t*)r3d_p(c.off), tw, (uint16_t*)r3d_wp(c.y), (int)c.T, (int)c.N, (int)c.K,
        a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted);
    return (int)hipGetLastError();
}

}  // extern "C"
