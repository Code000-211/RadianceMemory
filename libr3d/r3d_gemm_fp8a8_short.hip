// r3d_gemm_fp8a8_short.hip -- gemm_nt_q_short: the narrow fp8 GEMM with the short-prompt split rule.
// At N <= 512, K <= 5120 and 41..64 rows the split's per-slice floors dominate, so the row runs
// unsplit (sk=1).
#include <hip/hip_runtime.h>
#include <cstdint>
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_launch.h"

extern "C" int r3d_rad_gemm_nt_q_short_fp8(const RadArgs* a, RadStream s)
{
    enum { Q_A = 0, Q_ASCALE, Q_B, Q_BSCALE, Q_Y };
    const RadTensor* A  = r3d_opt(a, Q_A);
    const RadTensor* AS = r3d_opt(a, Q_ASCALE);
    const RadTensor* B  = r3d_opt(a, Q_B);
    const RadTensor* BS = r3d_opt(a, Q_BSCALE);
    const RadTensor* Y  = r3d_opt(a, Q_Y);
    if (!r3d_p(A) || !r3d_p(B) || !r3d_p(BS) || !r3d_wp(Y)) return R3D_NO(RAD_E_INVAL);

    const int M = (int)(Y->rank >= 1 ? Y->shape[0] : r3d_geti_or(a, "M", 0));
    const int N = (int)(Y->rank >= 2 ? Y->shape[1] : r3d_geti_or(a, "N", 0));
    const int K = (int) r3d_geti_or(a, "K", B->rank >= 2 ? B->shape[B->rank - 1] : 0);
    const int group = (int) r3d_geti_or(a, "group", 128);
    const int pb1 = (int) r3d_geti_or(a, "pb1", group);
    const int pb2 = (int) r3d_geti_or(a, "pb2", group);
    const int a_ld = (A->rank >= 2) ? (int)A->stride[0] : K;
    const int y_ld = (Y->rank >= 2) ? (int)Y->stride[0] : N;
    const int bs_dt = BS->dtype;

    // Short-prompt rule: force sk=1 (no split)
    int rc = r3d_gemm_nt_q_fp8_ex((long)(uintptr_t)r3d_p(A),
                                   AS ? (long)(uintptr_t)r3d_p(AS) : 0,
                                   (long)(uintptr_t)r3d_p(B),
                                   (long)(uintptr_t)r3d_p(BS),
                                   (long)(uintptr_t)r3d_wp(Y),
                                   M, N, K, group, pb1, pb2, a_ld, y_ld,
                                   1, 1, 0, bs_dt, (long)(uintptr_t)s, 0);
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}
