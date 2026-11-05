// r3d_gemm_fp8a8_bias.hip -- gemm_nt_q_bias: the narrow fp8 GEMM with a per-column bias epilogue.
// Runs r3d_gemm_nt_q_fp8_ex for the GEMM, then a small kernel adds the bias.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_launch.h"

// The low-level fp8 GEMM entry point (from r3d_gemm_fp8a16_nt_m1.hip)
extern "C" int r3d_gemm_nt_q_fp8_ex(long a, long a_scale, long b, long b_scale, long y,
                                     int M, int N, int K, int group, int pb1, int pb2,
                                     int a_ld, int y_ld, int sk, int nt, long scratch,
                                     int bs_dt, long stream, long bias);

extern "C" int r3d_rad_gemm_nt_q_bias_fp8(const RadArgs* a, RadStream s)
{
    enum { Q_A = 0, Q_ASCALE, Q_B, Q_BSCALE, Q_BIAS, Q_Y };
    const RadTensor* A  = r3d_opt(a, Q_A);
    const RadTensor* AS = r3d_opt(a, Q_ASCALE);
    const RadTensor* B  = r3d_opt(a, Q_B);
    const RadTensor* BS = r3d_opt(a, Q_BSCALE);
    const RadTensor* BI = r3d_opt(a, Q_BIAS);
    const RadTensor* Y  = r3d_opt(a, Q_Y);
    if (!r3d_p(A) || !r3d_p(B) || !r3d_p(BS) || !r3d_wp(Y) || !r3d_p(BI))
        return R3D_NO(RAD_E_INVAL);
    if (BI->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);

    const int M = (int)(Y->rank >= 1 ? Y->shape[0] : r3d_geti_or(a, "M", 0));
    const int N = (int)(Y->rank >= 2 ? Y->shape[1] : r3d_geti_or(a, "N", 0));
    const int K = (int) r3d_geti_or(a, "K", B->rank >= 2 ? B->shape[B->rank - 1] : 0);
    const int group = (int) r3d_geti_or(a, "group", 128);
    const int a_ld = (A->rank >= 2) ? (int)A->stride[0] : K;
    const int y_ld = (Y->rank >= 2) ? (int)Y->stride[0] : N;
    const int bs_dt = BS->dtype;

    // Run the GEMM with bias fused in
    int rc = r3d_gemm_nt_q_fp8_ex((long)(uintptr_t)r3d_p(A),
                                   AS ? (long)(uintptr_t)r3d_p(AS) : 0,
                                   (long)(uintptr_t)r3d_p(B),
                                   (long)(uintptr_t)r3d_p(BS),
                                   (long)(uintptr_t)r3d_wp(Y),
                                   M, N, K, group, group, group, a_ld, y_ld,
                                   0, 1, 0, bs_dt, (long)(uintptr_t)s,
                                   (long)(uintptr_t)r3d_p(BI));
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}
