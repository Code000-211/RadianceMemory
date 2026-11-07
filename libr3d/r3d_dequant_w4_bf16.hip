// r3d_dequant_w4_bf16.hip -- dequant_w4: the packed 4-bit weight back to a plain row-major
// bf16 [N, K].
//
// Used at prefill. W4A8 gains nothing there (iu8 and fp8 matrix rates are within a few percent
// and the shape is compute-bound), and the skinny W4A8 GEMM re-reads the whole weight slice once
// per 64 output rows, so its rate is flat in M. Dequantising one layer into scratch and running
// the tuned bf16 GEMM costs one extra write and read of the bf16 weight, a few percent of the GEMM.
//
// Two encodings share these bytes and differ by bit 3, so the caller must say which. The
// symmetric grid stores a two's complement code, value (n^8)-8, so the int8 GEMM can read the
// high-nibble byte signed and get 16*q. The asymmetric grid stores an unsigned nibble minus a
// per-group zero, corrected in the GEMM by an activation row-sum. Mixing them up agrees for n<8
// and is off by 16 above.
//
// Nearly lossless: a 4-bit code times an f16 scale fits bf16's 8-bit mantissa to within a ULP.
//
// Fragment order: radiance_w4.pack lays a dword out as [t][kb][lh][r][s] with eight nibbles along
// i, where t = n/16, r = n%16, kb = k/64, s = (k%64)/16, and the k inside the step is
// 4*lh + _KOFF[i] with _KOFF = {0,8,1,9,2,10,3,11}. One thread per dword scatters its eight
// nibbles. The scale dword at [t][g][r] holds the f16 scale low and -(1024+zero) high, so one
// routine serves the symmetric grid (zero == 8) and the asymmetric one.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_launch.h"

#define R3D_DQ_KPB     64
#define R3D_DQ_THREADS 256

static const int r3d_dq_koff[8] = {0, 8, 1, 9, 2, 10, 3, 11};

__global__ __launch_bounds__(R3D_DQ_THREADS) void r3d_dequant_w4_bf16_kernel(
    const unsigned* __restrict__ Wq, const unsigned* __restrict__ Ws,
    __hip_bfloat16* __restrict__ Out, int N, int K, int group, int twos)
{
    const long total = (long)N * K / 8;                 // one thread per packed dword
    const int nkb = K / R3D_DQ_KPB, ng = K / group;
    for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x) {
        // [t][kb][lh][r][s]
        const int s  = (int)(idx & 3);
        const int r  = (int)((idx >> 2) & 15);
        const int lh = (int)((idx >> 6) & 1);
        const int kb = (int)((idx >> 7) % nkb);
        const int t  = (int)((idx >> 7) / nkb);
        const int row_n = t * 16 + r;
        const unsigned w = Wq[idx];

        const int kbase = kb * R3D_DQ_KPB + s * 16 + 4 * lh;
        const unsigned sz = Ws[(size_t)(t * ng + kbase / group) * 16 + r];
        const float sc = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
        const float nz = __half2float(__ushort_as_half((unsigned short)(sz >> 16)));
        const float zero = -nz - 1024.f;                  // exact for zero in 0..15

#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int n = (int)((w >> (4 * i)) & 0xF);
            const float q = twos ? (float)((n ^ 8) - 8) : (float)n;
            const int k = kbase + r3d_dq_koff[i];
            Out[(size_t)row_n * K + k] = __float2bfloat16((q - (twos ? 0.f : zero)) * sc);
        }
    }
}

extern "C" int r3d_rad_dequant_w4(const RadArgs* a, RadStream s)
{
    const RadTensor* wq = r3d_opt(a, 0), *ws = r3d_opt(a, 1), *y = r3d_opt(a, 2);
    if (!r3d_p(wq) || !r3d_p(ws) || !r3d_wp(y)) return R3D_NO(RAD_E_INVAL);
    if (y->rank != 2 || y->stride[1] != 1) return R3D_NO(RAD_E_STRIDE);

    const long long N = y->shape[0];
    const long long K = y->shape[1];
    const long long group = r3d_geti_or(a, "group", 128);
    const long long twos  = r3d_geti_or(a, "twos", 1);
    if (N <= 0 || K <= 0 || N % 16 || K % 64) return R3D_NO(RAD_E_SHAPE);
    if (group < 1 || K % group) return R3D_NO(RAD_E_SHAPE);
    if (twos != 0 && twos != 1) return R3D_NO(RAD_E_INVAL);

    const long long total = N * K / 8;
    const int blocks = (int)((total + R3D_DQ_THREADS - 1) / R3D_DQ_THREADS);
    dim3 grid(blocks < 65535 ? blocks : 65535), blk(R3D_DQ_THREADS);
    R3D_KLAUNCH(r3d_dequant_w4_bf16_kernel, grid, blk, (hipStream_t)s,
                (const unsigned*)r3d_p(wq), (const unsigned*)r3d_p(ws),
                (__hip_bfloat16*)r3d_wp(y), (int)N, (int)K, (int)group, (int)twos);
    return (int)hipGetLastError();
}

