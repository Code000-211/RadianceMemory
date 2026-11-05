/* r3d_moe_gemm_fp8_f32scale.hip -- moe_gemm_q (the wave-matrix form): the grouped GEMM with E4M3
 * experts and exact F32 block scales.
 *
 *   y[i, n] = sum_g as[tok(i), g] * ws_e[n/128][g] * sum_{k in g} a[tok(i), k] * w_e[n, k]
 *
 * The moe_gemm_q contract at fp8a8f32: a and the weight entries are E4M3, one F32 scale per (row,
 * 128 of K) for the activation and one per 128x128 tile for the weight; expert e is entry e/2 of
 * class e%2's weight tables (r3d_moe_tab.h: a pointer per expert, or a stacked plane), each entry
 * the checkpoint's own [N, K] row-major codes; y is bf16 in sorted order. f32 accumulate over the
 * whole of K.
 *
 * gfx11 has no native FP8 wave-matrix, so the codes widen to f16 (exact: every E4M3 grid point is
 * an f16 grid point) and the multiply runs on the 16x16x16 f16 tile. K is walked in 128-deep
 * tiles (the scale group); each tile's eight 16-wide steps accumulate into a fresh f32 partial,
 * which is folded as raw * (activation scale * weight scale) into the running sum. The 16-row
 * tile is the same run-walk / gather structure as moe_gemm_bf16.
 */
#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_fp8_dequant.h"
#include "r3d_launch.h"
#include "r3d_moe_tab.h"

namespace {

constexpr int kThreads = 256;           // eight waves
constexpr int kWaves   = kThreads / kWave;
constexpr int kNBlock  = 128;           // eight waves x 16 cols
constexpr int kKT      = 128;           // a k tile: one scale group (eight 16-wide wmma steps)
constexpr int kDW      = kKT / 4;       // dwords a row: 128 fp8 are 32 dwords (four fp8 a dword)
constexpr int kPitch   = kDW + 1;       // dwords a staged row: +1, bank-free
constexpr int kStageW  = kNBlock * kDW / kThreads;
constexpr int kP16     = kKT + 1;       // f16 halfs a staged row: 128 + 1, bank-free

/* The kernel, emitted once per register-budget point. amdgpu_num_vgpr takes a literal, so the
 * budget cannot live on a template; and the definitions stay monolithic, because the allocator
 * schedules the inlined-body form (a __device__ helper) noticeably worse.
 *
 * Register budgets (gfx11: 1024 VGPRs/SIMD, 8 waves needs <= 128; measured on gfx1102 at
 * T=N=K=4096 NE=8, median of 15 launches): the natural 155 VGPRs give 6 waves; the 128 cap lands
 * at 103 with zero spills = 8 waves, +2-4%. The tune exposes both points (tMoeGemmFp8) and
 * defaults to the cap. Re-measure before changing a budget here. */

/* The accumulator's mode (the tune's `acc` axis): 0 = f32, 1 = the f16-precision WMMA
 * accumulator (the 16x16x16_f16 builtin, OPSEL false: the 8 live values in the even indices of a
 * v16, 4 VGPRs). `blk` is the per-k-tile accumulator; it is folded (with the per-row scale) into
 * the f32 `acc` after each 128-wide k tile, so the half-precision span is one tile, not the K.
 *
 * PRECISION (measured on gfx1102): the acc=1 arm quantises the per-tile f16 partial sum at 16-bit
 * precision per wave-matrix step (the hardware's documented low-precision mode, same builtin
 * chain AMD's ck uses). On exactly-representable E4M3 code data (0.5/1.0) it is bit-exact to the
 * f32 arm (store rounding only); on arbitrary f16 data the deviation grows with the k-tile depth
 * (~0.17 abs over 4 steps of 16-wide random f16 in [-1,1], measured). No throughput gain from
 * acc=1 (measured +0.2%..-1%). It exists as a register-pressure option, not a speed one; acc=0
 * is the correct default for accuracy-sensitive shapes. */
template <bool ACC>
struct R3dMoeAccF16 { using type = std::conditional_t<ACC, v16half, v8float>; };
template <bool ACC>
__device__ __forceinline__ void r3d_moe_acc_f16_init(typename R3dMoeAccF16<ACC>::type& x)
{
    if constexpr (ACC) x = v16half{ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 };
    else x = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
}
template <bool ACC>
__device__ __forceinline__ typename R3dMoeAccF16<ACC>::type r3d_moe_wmma_f16(
    const v16half& fa, const v16half& fb, const typename R3dMoeAccF16<ACC>::type& c)
{
    if constexpr (ACC) return wmma_f16acc<false>(fa, fb, c);
    else return wmma_f16(fa, fb, c);
}
template <bool ACC>
__device__ __forceinline__ float r3d_moe_acc_f16_read(const typename R3dMoeAccF16<ACC>::type& x, int g)
{
    /* Direct f16 -> f32 value conversion. NOT a bit-cast: type-punning the OPSEL
     * (16-bit-accumulator) vector through a union breaks the srcC == result physical-
     * register constraint, and the compiler then reads the wmma result from the stale
     * srcC copy (measured: the accumulated tile comes back as zero). */
    if constexpr (ACC) return (float)x[2 * g];
    return x[g];
}

#define R3D_MFS_BODY                                                                          \
__global__ __launch_bounds__(kThreads) R3D_WAVES(kWaves)                                       \
__attribute__((amdgpu_num_vgpr(MFS_VGPR))) void MFS_KNAME(                                   \
    const uint8_t* __restrict__ a, const float* __restrict__ as,                               \
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, R3dMoeTabs tw,        \
    uint16_t* __restrict__ y, int T, int N, int K, int a_ld, int as_ld, int y_ld,             \
    int n_expert, int top_k, int a_sorted)                                                     \
{                                                                                              \
    constexpr int kMBlock = 16;                                                                \
    constexpr int kRB     = 1;                                                                 \
    __shared__ __align__(16) uint16_t s_a[kMBlock * kP16];   /* activation f16 (128 per row) */ \
    __shared__ __align__(16) uint16_t s_w[kNBlock * kP16];   /* weight f16 (128 per row) */     \
    __shared__ uint16_t s_tbl[256];                            /* the e4m3 -> f16 table */      \
    __shared__ int32_t s_tok[kMBlock];                                                 \
    __shared__ int32_t s_nruns;                                                      \
    __shared__ int32_t s_rs[kMBlock], s_re[kMBlock], s_e[kMBlock];                    \
                                                                                    \
    const uint32_t tid = threadIdx.x;                                               \
    const uint32_t lane = tid & (kWave - 1);                                        \
    const uint32_t wv = tid / kWave;                                                \
    const uint32_t r = lane & 15;                                                   \
                                                                                    \
    /* The e4m3 -> f16 table: filled once before any lane reads it. */               \
    r3d_e4m3_f16_init(s_tbl, (int)tid, (int)kThreads);                               \
                                                                                    \
    const uint32_t r0 = (uint32_t)blockIdx.y * kMBlock;                              \
    const uint32_t n0 = (uint32_t)blockIdx.x * kNBlock;                              \
    const uint32_t r1 = (r0 + kMBlock < (uint32_t)T) ? r0 + kMBlock : (uint32_t)T;   \
                                                                                    \
    if (tid < kMBlock) {                                                            \
        const uint32_t i = r0 + tid;                                                \
        const int32_t f = (i < (uint32_t)T) ? sorted[i] : -1;                       \
        s_tok[tid] = (f < 0) ? -1 : (a_sorted ? (int32_t)i : f / top_k);           \
    }                                                                               \
    if (tid == 0) {                                                                 \
        int nruns = 0, i = (int)r0;                                                 \
        while (i < (int)r1) {                                                       \
            int32_t ee = -1;                                                         \
            for (int32_t x = 0; x < n_expert; ++x) if (off[x] <= i && i < off[x + 1]) ee = x; \
            const int re = (ee < 0) ? (int)r1 : (int)(off[ee + 1] < r1 ? off[ee + 1] : r1);               \
            s_rs[nruns] = i; s_re[nruns] = re; s_e[nruns] = ee;                      \
            ++nruns; i = re;                                                        \
        }                                                                           \
        s_nruns = nruns;                                                            \
    }                                                                               \
    group_barrier();                                                                \
    const int nruns = s_nruns;                                                      \
                                                                                    \
    const uint32_t nc = n0 + wv * 16 + r;                                           \
    const int n_real = (int)(nc < (uint32_t)N ? nc : (uint32_t)N - 1);              \
                                                                                    \
    for (int run = 0; run < nruns; ++run) {                                         \
        const int rs = s_rs[run], re = s_re[run], e = s_e[run]; \
        float acc[kRB][8];                                                          \
        _Pragma("unroll")                                                              \
        for (int rb = 0; rb < kRB; ++rb)                                            \
            _Pragma("unroll")                                                          \
            for (int g = 0; g < 8; ++g) acc[rb][g] = 0.f;                           \
        if (e >= 0) {                                                       \
        /* Expert e's entry: e/2 of class e%2's tables. */                           \
        const int cls = e & 1, e2 = e >> 1;                                         \
        const uint8_t* wbase = r3d_tab_at(tw.w[cls], e2);                           \
        const int w_ld = tw.w[cls].ld;                                              \
        const float* wsb = (const float*)r3d_tab_at(tw.s[cls], e2);                 \
        const int ws_ld = tw.s[cls].ld;                                             \
                                                                                    \
                                                                                    \
        const int ntile = (K + kKT - 1) / kKT;                                      \
        for (int t = 0; t < ntile; ++t) {                                           \
            const int kt = t * kKT;                                                 \
            /* Stage the activation and weight for this 128-k tile, dequantizing E4M3 -> f16 */ \
            /* through the table (the mxfp4a8 pattern: f16 in LDS, WMMA reads f16 directly). */  \
            for (int i = (int)tid; i < kMBlock * kKT; i += kThreads) {              \
                const int row = i / kKT, k = i % kKT;                               \
                const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row] \
                                                                                 : s_tok[rs - (int)r0]; \
                const int rr = (src < 0) ? 0 : src;                                 \
                s_a[(size_t)row * kP16 + k] = s_tbl[a[(size_t)rr * a_ld + kt + k]]; \
            }                                                                       \
            for (int i = (int)tid; i < kNBlock * kKT; i += kThreads) {              \
                const int col = i / kKT, k = i % kKT;                               \
                const int nn = (int)(n0 + col < (uint32_t)N ? n0 + col : (uint32_t)N - 1); \
                s_w[(size_t)col * kP16 + k] = s_tbl[wbase[(size_t)nn * w_ld + kt + k]]; \
            }                                                                       \
            group_barrier();                                                        \
            /* The activation scale for this tile (per row) and the weight scale (per 128x128 tile). */ \
            const float wscale = wsb[(size_t)(n_real / 128) * ws_ld + t];             \
            /* Eight 16-wide f16 WMMA steps; f16 values read directly from LDS. The fragment */  \
            /* elements are f16 BIT PATTERNS, so the LDS halfs must be bit-cast into the */       \
            /* _Float16 vector: any assignment from a uint16_t source into a _Float16 (a plain */ \
            /* `fa[j] = ap[k+j]`, or a union whose half members are uint16_t) is a VALUE */       \
            /* conversion (code 0x4300, i.e. f16 3.5, would become the number 17152.0) and */     \
            /* corrupts every product. The union's half members must themselves be _Float16. */    \
            _Pragma("unroll")                                                          \
            for (int rb = 0; rb < kRB; ++rb) {                                      \
                typename R3dMoeAccF16<MFS_ACC>::type blk;                           \
                r3d_moe_acc_f16_init<MFS_ACC>(blk);                                   \
                const uint16_t* ap = s_a + (size_t)(rb * 16 + r) * kP16;             \
                const uint16_t* wp = s_w + (size_t)(wv * 16 + r) * kP16;             \
                _Pragma("unroll")                                                      \
                for (int k = 0; k < kKT; k += 16) {                                 \
                    v16half fa, fb;                                                 \
                    const uint32_t* ap32 = (const uint32_t*)(ap + k);                \
                    const uint32_t* wp32 = (const uint32_t*)(wp + k);                \
                    _Pragma("unroll")                                                  \
                    for (int j = 0; j < 8; ++j) {                                   \
                        union { uint32_t u; _Float16 h[2]; } ca, cb;                \
                        ca.u = ap32[j]; cb.u = wp32[j];                             \
                        fa[2 * j] = ca.h[0]; fa[2 * j + 1] = ca.h[1];               \
                        fb[2 * j] = cb.h[0]; fb[2 * j + 1] = cb.h[1];               \
                    }                                                               \
                    blk = r3d_moe_wmma_f16<MFS_ACC>(fa, fb, blk);                  \
                }                                                                   \
                /* Fold: raw * (activation scale * weight scale). Each output g is for a different */ \
                /* row (rb*16 + g*2 + (lane>>4)), so each needs its own source token and scale. */    \
                _Pragma("unroll")                                                      \
                for (int g = 0; g < 8; ++g) {                                       \
                    const int row = rb * 16 + g * 2 + (int)(lane >> 4);             \
                    const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) \
                                    ? s_tok[row] : s_tok[rs - (int)r0];              \
                    const float ascale = (src < 0) ? 0.f : as[(size_t)src * as_ld + t] * wscale; \
                    acc[rb][g] += r3d_moe_acc_f16_read<MFS_ACC>(blk, g) * ascale;   \
                }                                                                   \
            }                                                                       \
            group_barrier();                                                        \
        }                                                                           \
        } \
        /* Store only the run's rows and the real columns, as bf16. */                \
        _Pragma("unroll")                                                              \
        for (int rb = 0; rb < kRB; ++rb)                                            \
            _Pragma("unroll")                                                          \
            for (int g = 0; g < 8; ++g) {                                           \
                const int m = (int)r0 + rb * 16 + g * 2 + (int)(lane >> 4);         \
                if (m < re && m >= rs && nc < (uint32_t)N)                          \
                    y[(size_t)m * y_ld + nc] = f32_to_bf16(acc[rb][g]);              \
            }                                                                       \
    }                                                                               \
}
#define MFS_KNAME moe_fp8_f32scale_kernel16
#define MFS_VGPR 512
#define MFS_ACC 0
R3D_MFS_BODY
#undef MFS_KNAME MFS_VGPR MFS_ACC
#define MFS_KNAME moe_fp8_f32scale_kernel16_a1
#define MFS_VGPR 512
#define MFS_ACC 1
R3D_MFS_BODY
#undef MFS_KNAME MFS_VGPR MFS_ACC
#define MFS_KNAME moe_fp8_f32scale_kernel16_c128
#define MFS_VGPR 128
#define MFS_ACC 0
R3D_MFS_BODY
#undef MFS_KNAME MFS_VGPR MFS_ACC
#define MFS_KNAME moe_fp8_f32scale_kernel16_c128_a1
#define MFS_VGPR 128
#define MFS_ACC 1
R3D_MFS_BODY
#undef MFS_KNAME MFS_VGPR MFS_ACC
#undef R3D_MFS_BODY

}  // namespace

/* The RadArgs dispatch, moe_gemm_q's arm 2 (r3d_moe_gemm_q.hip): the contract's operands
 * (r3d_moe_tab.h) at fp8a8f32, the tune's register budget and accumulator. */
extern "C" int r3d_moe_gemm_fp8_f32scale_run(const RadArgs* a, RadStream s, int vgpr, int acc)
{
    R3dMoeCall c;
    int rc = r3d_moe_call(a, true, &c);
    if (rc != RAD_OK) return rc;
    if (!c.a_scale) return R3D_NO(RAD_E_INVAL);
    if (c.a->dtype != RAD_F8E4M3 || c.a_scale->dtype != RAD_F32 || c.y->dtype != RAD_BF16)
        return R3D_NO(RAD_E_DTYPE);
    if (c.K % kKT) return R3D_NO(RAD_E_SHAPE);
    R3dMoeTabs tw;
    rc = r3d_moe_q_tabs(c, RAD_F8E4M3, 8, RAD_F32, 32, (c.N + 127) / 128, c.K / 128, &tw);
    if (rc != RAD_OK) return rc;
    const int T = (int)c.T, N = (int)c.N, K = (int)c.K;
    const int a_ld = (int)c.a->stride[0], as_ld = (int)c.a_scale->stride[0];
    const int y_ld = (int)c.y->stride[0];
    const uint8_t* pa = (const uint8_t*)r3d_p(c.a);
    const float* pas = (const float*)r3d_p(c.a_scale);
    const int32_t* ps = (const int32_t*)r3d_p(c.sorted);
    const int32_t* po = (const int32_t*)r3d_p(c.off);
    uint16_t* py = (uint16_t*)r3d_wp(c.y);
    const dim3 grid((N + kNBlock - 1) / kNBlock, (T + 16 - 1) / 16), block(kThreads);
#define R3D_MFS_LAUNCH(KERN)                                                                  \
    R3D_KLAUNCH(KERN, grid, block, (hipStream_t)s, pa, pas, ps, po, tw, py, T, N, K, a_ld,    \
                as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted)
    if (vgpr == 128) {
        if (acc) R3D_MFS_LAUNCH(moe_fp8_f32scale_kernel16_c128_a1);
        else     R3D_MFS_LAUNCH(moe_fp8_f32scale_kernel16_c128);
    }
    else {
        if (acc) R3D_MFS_LAUNCH(moe_fp8_f32scale_kernel16_a1);
        else     R3D_MFS_LAUNCH(moe_fp8_f32scale_kernel16);
    }
#undef R3D_MFS_LAUNCH
    return (int)hipGetLastError();
}

extern "C" int r3d_rad_moe_gemm_fp8_f32scale(const RadArgs* a, RadStream s)
{
    R3dMoeCfg c; r3d_cfg_moe(a, &c);   // the checker entry point's budget (128 default)
    return r3d_moe_gemm_fp8_f32scale_run(a, s, c.vgpr, c.acc);
}
