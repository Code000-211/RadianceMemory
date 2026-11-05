/* r3d_moe_gemm_bf16.hip -- moe_gemm_bf16: the grouped GEMM with unquantised bf16 experts.
 *
 *   y[i, n] = sum_k a[tok(i), k] * w_e[n, k],   e = the expert of sorted row i
 *
 * One block per 16 expert-sorted rows, walking the runs (contiguous rows per expert) in its tile.
 * bf16 activation, bf16 weight as the checkpoint stores it (each expert an [N, K] row-major entry
 * of one weight table, r3d_moe_tab.h; no layout hook, every expert in class 0), f32 accumulate over
 * the whole of K, bf16 out. The 16-row tile is
 * the same run-walk / gather structure as moe_gemm_q, minus the E4M3 decode and the scale fold:
 * the operands are bf16, so they stage to LDS and load straight into the bf16 wave-matrix
 * fragments (v_wmma_f32_16x16x16_bf16, f32 accumulate).
 *
 * K is walked in 64-deep tiles (four 16-wide wave-matrix steps); N and K are multiples of 16.
 */
#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_launch.h"
#include "r3d_moe_tab.h"

namespace {

constexpr int kThreads = 256;           // eight waves
constexpr int kWaves   = kThreads / kWave;
constexpr int kNBlock  = 128;           // eight waves x 16 cols
constexpr int kKT      = 64;            // a k tile: four 16-wide wmma steps
constexpr int kDW      = kKT / 2;       // dwords a row: 64 bf16 are 32 dwords (two bf16 a dword)
constexpr int kPitch   = kDW + 1;       // dwords a staged row: +1, bank-free
constexpr int kStageW  = kNBlock * kDW / kThreads;     // ...and of the weight (128 cols)

/* MROWS is the block's row count (16, 32 or 64), the tune's point: a taller block reads the
 * weight T/MROWS times, so the tune picks per shape -- 16 for decode, 64 for a tall prefill. */
/* The kernel, emitted once per (row count, register budget) point. amdgpu_num_vgpr takes a
 * literal, so the budget cannot live on a template; and the definitions stay monolithic, because
 * the allocator schedules the inlined-body form (a __device__ helper) noticeably worse.
 *
 * Register budgets (gfx11: 1024 VGPRs/SIMD, 8 waves needs <= 128; measured on gfx1102 at
 * T=N=K=4096 NE=8, median of 15 launches):
 *   (32, 128): 144 VGPR, 6 spills = 7 waves -- +38% vs (32, 512) at 171 VGPR = 5 waves: the middle
 *        height is latency-bound, so the extra waves beat the spill traffic.
 *   (16, 128) and (64, 128) measure 7-8% SLOWER (145/8 and 193/54 spills), so those rows keep the
 *        natural allocation only.
 * Re-measure before touching a budget here; occupancy is a means to hide latency, not a goal
 * (GPUOpen, "Occupancy explained": only latency-bound work benefits, and spills can cost more). */

/* The accumulator's mode (the tune's `acc` axis): 0 = f32, 1 = the bf16-precision WMMA
 * accumulator (the 16x16x16_bf16 builtin, OPSEL false: the 8 live values in the even indices of
 * a v16, 4 VGPRs). `blk` is the per-k-tile accumulator; it is folded into the f32 `acc` after
 * each 64-wide k tile, so the half-precision span is one tile, not the whole K.
 *
 * PRECISION (measured on gfx1102, random bf16 in [-1,1], K = 128): the acc=1 arm deviates from
 * the f32 reference by up to ~0.36 absolute (~3% of a typical |y| ~ 11; larger relative to
 * near-zero outputs). The 16-bit WMMA accumulator is a documented low-precision mode (AMD ROCm
 * Optimization Guide, RDNA3 dense WMMA builtins; the same builtin chain AMD's ck uses): the
 * hardware quantises the running sum at 16-bit precision per step, and the loss grows with the
 * number of steps. Exact on integer-valued data (e.g. +/-1 inputs). There is NO throughput gain
 * from the acc=1 arm (measured +0.05%..-0.2%): it exists as a register-pressure option for
 * future occupancy work, not a speed one. Do not select acc=1 for accuracy-sensitive shapes;
 * the f32 arm (acc=0) is the correct default. */
template <bool ACC>
struct R3dMoeAcc { using type = std::conditional_t<ACC, v16short, v8float>; };
template <bool ACC>
__device__ __forceinline__ void r3d_moe_acc_init(typename R3dMoeAcc<ACC>::type& x)
{
    if constexpr (ACC) x = v16short{ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 };
    else x = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
}
template <bool ACC>
__device__ __forceinline__ typename R3dMoeAcc<ACC>::type r3d_moe_wmma_bf16(
    const v16short& fa, const v16short& fb, const typename R3dMoeAcc<ACC>::type& c)
{
    if constexpr (ACC) return wmma_bf16acc<false>(fa, fb, c);
    else return wmma_bf16(fa, fb, c);
}
template <bool ACC>
__device__ __forceinline__ float r3d_moe_acc_read(const typename R3dMoeAcc<ACC>::type& x, int g)
{
    if constexpr (ACC) {
        const uint32_t u = (uint32_t)(uint16_t)x[2 * g] << 16;   /* the bf16 bits -> f32 */
        return *(const float*)&u;
    }
    return x[g];
}

#define R3D_MBE_BODY                                                                          \
__global__ __launch_bounds__(kThreads) R3D_WAVES(kWaves)                                      \
__attribute__((amdgpu_num_vgpr(MBE_VGPR))) void MBE_KNAME(                                   \
    const uint16_t* __restrict__ a,                                                           \
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, R3dTab tw,           \
    uint16_t* __restrict__ y, int T, int N, int K, int a_ld, int y_ld,                        \
    int n_expert, int top_k, int a_sorted)                                                    \
{                                                                                             \
    const int w_ld = tw.ld;                  /* elements between an expert's rows */         \
    constexpr int kMBlock = MBE_MR;          /* a block is MBE_MR rows x 128 cols */              \
    constexpr int kRB     = MBE_MR / 16;     /* row-blocks a wave accumulates */              \
    constexpr int kStageA = MBE_MR * kDW / kThreads;   /* dwords a thread stages of the activation */ \
    __shared__ __align__(16) uint32_t s_a[2][kMBlock * kPitch];   /* double-buffered activation (32 rows) */ \
    __shared__ __align__(16) uint32_t s_w[2][kNBlock * kPitch];   /* ...and weight (128 cols) */ \
    __shared__ int32_t s_tok[kMBlock];                                            \
    __shared__ int32_t s_nruns;                                                   \
    __shared__ int32_t s_rs[kMBlock], s_re[kMBlock], s_e[kMBlock];                 \
                                                                                 \
    const uint32_t tid = threadIdx.x;                                             \
    const uint32_t lane = tid & (kWave - 1);                                      \
    const uint32_t wv = tid / kWave;          /* the wave index (0..7): owns cols wv*16..wv*16+16 */ \
    const uint32_t r = lane & 15;                                                 \
                                                                                 \
    const uint32_t r0 = (uint32_t)blockIdx.y * kMBlock;   /* the block's sorted rows */ \
    const uint32_t n0 = (uint32_t)blockIdx.x * kNBlock;                            \
    const uint32_t r1 = (r0 + kMBlock < (uint32_t)T) ? r0 + kMBlock : (uint32_t)T; \
                                                                                 \
    /* The block's rows' source tokens (the gather). -1 marks a dropped slot. */   \
    if (tid < kMBlock) {                                                          \
        const uint32_t i = r0 + tid;                                              \
        const int32_t f = (i < (uint32_t)T) ? sorted[i] : -1;                     \
        s_tok[tid] = (f < 0) ? -1 : (a_sorted ? (int32_t)i : f / top_k);         \
    }                                                                             \
    /* The runs in [r0, r1): each a contiguous set of rows for one expert. The block walks them. */ \
    if (tid == 0) {                                                               \
        int nruns = 0, i = (int)r0;                                               \
        while (i < (int)r1) {                                                     \
            int32_t ee = -1;                                                       \
            for (int32_t x = 0; x < n_expert; ++x) if (off[x] <= i && i < off[x + 1]) ee = x; \
            const int re = (ee < 0) ? (int)r1 : (int)(off[ee + 1] < r1 ? off[ee + 1] : r1);             \
            s_rs[nruns] = i; s_re[nruns] = re; s_e[nruns] = ee;                    \
            ++nruns; i = re;                                                      \
        }                                                                         \
        s_nruns = nruns;                                                          \
    }                                                                             \
    group_barrier();                                                              \
    const int nruns = s_nruns;                                                    \
                                                                                 \
    const uint32_t nc = n0 + wv * 16 + r;   /* this wave's output column */        \
    const int n_real = (int)(nc < (uint32_t)N ? nc : (uint32_t)N - 1);            \
                                                                                 \
    /* One GEMM per run: the run's rows read the gathered activation and expert e's weight. Rows the */ \
    /* run does not own are padding (reading the run's first row) and their output is dropped. */      \
    for (int run = 0; run < nruns; ++run) {                                       \
        const int rs = s_rs[run], re = s_re[run], e = s_e[run]; \
        v8float acc[kRB];                                                         \
        _Pragma("unroll")                                                            \
        for (int rb = 0; rb < kRB; ++rb)                                          \
            acc[rb] = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };           \
        if (e >= 0) {                                                        \
        const uint16_t* wbase = (const uint16_t*)r3d_tab_at(tw, e);              \
                                                                                 \
        const int ntile = (K + kKT - 1) / kKT;                                    \
        /* Prologue: stage tile 0 into buffer 0. */                               \
        for (int i = (int)tid; i < kMBlock * kDW; i += kThreads) {                \
            const int row = i / kDW, d = i % kDW;                                 \
            const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row] \
                                                                             : s_tok[rs - (int)r0]; \
            const int rr = (src < 0) ? 0 : src;                                   \
            s_a[0][(size_t)row * kPitch + d] = *(const uint32_t*)(a + (size_t)rr * a_ld + 2 * d); \
        }                                                                         \
        for (int i = (int)tid; i < kNBlock * kDW; i += kThreads) {                \
            const int col = i / kDW, d = i % kDW;                                 \
            const int nn = (int)(n0 + col < (uint32_t)N ? n0 + col : (uint32_t)N - 1); \
            s_w[0][(size_t)col * kPitch + d] = *(const uint32_t*)(wbase + (size_t)nn * w_ld + 2 * d); \
        }                                                                         \
        group_barrier();                                                          \
        /* Pipelined: the next tile's global loads are issued into registers before this tile's WMMA, */ \
        /* so the ~300-cycle VRAM latency is hidden behind the matrix-core work; they land in the */   \
        /* other buffer after the WMMA (the s_waitcnt of the LDS store waits on the in-flight loads). */ \
        for (int t = 0; t < ntile; ++t) {                                         \
            const int b = t & 1, nb = b ^ 1;                                      \
            uint32_t ra[kStageA], rw[kStageW];                                    \
            if (t + 1 < ntile) {                                                  \
                const int kt1 = (t + 1) * kKT;                                    \
_Pragma("unroll")                                                                    \
                for (int i = (int)tid; i < kMBlock * kDW; i += kThreads) {        \
                    const int li = (i - (int)tid) / kThreads;                     \
                    const int row = i / kDW, d = i % kDW;                         \
                    const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row] \
                                                                                     : s_tok[rs - (int)r0]; \
                    const int rr = (src < 0) ? 0 : src;                           \
                    ra[li] = *(const uint32_t*)(a + (size_t)rr * a_ld + kt1 + 2 * d); \
                }                                                                 \
                for (int i = (int)tid; i < kNBlock * kDW; i += kThreads) {        \
                    const int li = (i - (int)tid) / kThreads;                     \
                    const int col = i / kDW, d = i % kDW;                         \
                    const int nn = (int)(n0 + col < (uint32_t)N ? n0 + col : (uint32_t)N - 1); \
                    rw[li] = *(const uint32_t*)(wbase + (size_t)nn * w_ld + kt1 + 2 * d); \
                }                                                                 \
            }                                                                     \
            const uint32_t* wp = s_w[b] + (size_t)(wv * 16 + r) * kPitch;   /* this wave's 16 cols */ \
_Pragma("unroll")                                                                    \
            for (int rb = 0; rb < kRB; ++rb) {   /* the 16-row blocks; the B-fragment is shared */ \
                const uint32_t* ap = s_a[b] + (size_t)(rb * 16 + r) * kPitch;     \
                typename R3dMoeAcc<MBE_ACC>::type blk;                            \
                r3d_moe_acc_init<MBE_ACC>(blk);                                   \
_Pragma("unroll")                                                                    \
                for (int k = 0; k < kKT; k += 16) {                               \
                    v16short fa, fb;                                              \
_Pragma("unroll")                                                                    \
                    for (int g = 0; g < 8; ++g) {                                 \
                        const uint32_t dw = ap[k / 2 + g];                        \
                        fa[2 * g] = (short)(dw & 0xffffu); fa[2 * g + 1] = (short)(dw >> 16); \
                        const uint32_t dwb = wp[k / 2 + g];                       \
                        fb[2 * g] = (short)(dwb & 0xffffu); fb[2 * g + 1] = (short)(dwb >> 16); \
                    }                                                             \
                    blk = r3d_moe_wmma_bf16<MBE_ACC>(fa, fb, blk);             \
                }                                                                 \
                _Pragma("unroll")                                                    \
                for (int g = 0; g < 8; ++g) acc[rb][g] += r3d_moe_acc_read<MBE_ACC>(blk, g); \
            }                                                                     \
            if (t + 1 < ntile) {                                                  \
                for (int i = (int)tid; i < kMBlock * kDW; i += kThreads) {        \
                    const int li = (i - (int)tid) / kThreads;                     \
                    const int row = i / kDW, d = i % kDW;                         \
                    s_a[nb][(size_t)row * kPitch + d] = ra[li];                   \
                }                                                                 \
                for (int i = (int)tid; i < kNBlock * kDW; i += kThreads) {        \
                    const int li = (i - (int)tid) / kThreads;                     \
                    const int col = i / kDW, d = i % kDW;                         \
                    s_w[nb][(size_t)col * kPitch + d] = rw[li];                   \
                }                                                                 \
            }                                                                     \
            group_barrier();                                                      \
        }                                                                         \
                                                                                 \
        } \
        /* Store only the run's rows, as bf16, across the 16-row blocks. */       \
_Pragma("unroll")                                                                    \
        for (int rb = 0; rb < kRB; ++rb)                                          \
            _Pragma("unroll")                                                        \
            for (int g = 0; g < 8; ++g) {                                         \
                const int row = (int)(rb * 16 + g * 2 + (lane >> 4));             \
                const int m = (int)(r0 + row);                                    \
                if (rs <= m && m < re && nc < (uint32_t)N)                        \
                    y[(size_t)m * y_ld + nc] = f32_to_bf16(acc[rb][g]);           \
            }                                                                     \
    }                                                                             \
}
#define MBE_KNAME moe_bf16_kernel16
#define MBE_MR 16
#define MBE_VGPR 512
#define MBE_ACC 0
R3D_MBE_BODY
#undef MBE_KNAME MBE_MR MBE_VGPR MBE_ACC
#define MBE_KNAME moe_bf16_kernel16_a1
#define MBE_MR 16
#define MBE_VGPR 512
#define MBE_ACC 1
R3D_MBE_BODY
#undef MBE_KNAME MBE_MR MBE_VGPR MBE_ACC
#define MBE_KNAME moe_bf16_kernel32
#define MBE_MR 32
#define MBE_VGPR 512
#define MBE_ACC 0
R3D_MBE_BODY
#undef MBE_KNAME MBE_MR MBE_VGPR MBE_ACC
#define MBE_KNAME moe_bf16_kernel32_a1
#define MBE_MR 32
#define MBE_VGPR 512
#define MBE_ACC 1
R3D_MBE_BODY
#undef MBE_KNAME MBE_MR MBE_VGPR MBE_ACC
#define MBE_KNAME moe_bf16_kernel32_c128
#define MBE_MR 32
#define MBE_VGPR 128
#define MBE_ACC 0
R3D_MBE_BODY
#undef MBE_KNAME MBE_MR MBE_VGPR MBE_ACC
#define MBE_KNAME moe_bf16_kernel32_c128_a1
#define MBE_MR 32
#define MBE_VGPR 128
#define MBE_ACC 1
R3D_MBE_BODY
#undef MBE_KNAME MBE_MR MBE_VGPR MBE_ACC
#define MBE_KNAME moe_bf16_kernel64
#define MBE_MR 64
#define MBE_VGPR 512
#define MBE_ACC 0
R3D_MBE_BODY
#undef MBE_KNAME MBE_MR MBE_VGPR MBE_ACC
#define MBE_KNAME moe_bf16_kernel64_a1
#define MBE_MR 64
#define MBE_VGPR 512
#define MBE_ACC 1
R3D_MBE_BODY
#undef MBE_KNAME MBE_MR MBE_VGPR MBE_ACC
#undef R3D_MBE_BODY

}  // namespace

extern "C" {

int r3d_rad_moe_gemm_bf16(const RadArgs* a, RadStream s)
{
    R3dMoeCall mc;
    int rc = r3d_moe_call(a, false, &mc);
    if (rc != RAD_OK) return rc;
    if (mc.a->dtype != RAD_BF16 || mc.y->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    const long long T = mc.T, N = mc.N, K = mc.K;
    if (K & 63) return R3D_NO(RAD_E_SHAPE);   // N tail clamps
    R3dTab tw;
    rc = r3d_tab_of(mc.w, RAD_BF16, 16, mc.n_expert, N, K, &tw);
    if (rc != RAD_OK) return rc;
    const int n_expert = (int)mc.n_expert, top_k = (int)mc.top_k, a_sorted = mc.a_sorted;
    const int a_ld = (int)mc.a->stride[0];
    const int y_ld = (int)mc.y->stride[0];
    const RadTensor *at = mc.a, *st = mc.sorted, *ot = mc.off, *yt = mc.y;

    R3dMoeCfg c; r3d_cfg_moe(a, &c);   // the tune's row count (16, 32 or 64), 64 untuned
#define R3D_LM(MR, KERN)                                                                            \
    do {                                                                                            \
        dim3 grid((unsigned)((N + kNBlock - 1) / kNBlock), (unsigned)((T + MR - 1) / MR)),          \
             blk(kThreads);                                                                         \
        KERN<<<grid, blk, 0, (hipStream_t)s>>>(                                                     \
            (const uint16_t*)r3d_p(at), (const int32_t*)r3d_p(st), (const int32_t*)r3d_p(ot), tw,   \
            (uint16_t*)r3d_wp(yt), (int)T, (int)N, (int)K, a_ld, y_ld, n_expert, top_k, a_sorted);  \
    } while (0)
    if (c.mrows <= 16) {
        if (c.acc) R3D_LM(16, moe_bf16_kernel16_a1);
        else R3D_LM(16, moe_bf16_kernel16);
    }
    else if (c.mrows <= 32) {
        if (c.vgpr == 128) {
            if (c.acc) R3D_LM(32, moe_bf16_kernel32_c128_a1);
            else R3D_LM(32, moe_bf16_kernel32_c128);
        }
        else {
            if (c.acc) R3D_LM(32, moe_bf16_kernel32_a1);
            else R3D_LM(32, moe_bf16_kernel32);
        }
    }
    else {
        if (c.acc) R3D_LM(64, moe_bf16_kernel64_a1);
        else R3D_LM(64, moe_bf16_kernel64);
    }
#undef R3D_LM
    return (int)hipGetLastError();
}

}  // extern "C"
