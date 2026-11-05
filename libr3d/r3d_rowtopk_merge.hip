// r3d_rowtopk_merge.hip -- row_topk_merge: the R largest of the all-gathered per-rank candidate
// sets. One wave per row over world_size * R pairs; the cost is its launch, it only has to be
// exact.
#include <hip/hip_runtime.h>
#include <cstdint>
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#define R3D_TOPK_WAVE  32
#define R3D_TOPK_PT2   32      // >= R, so a lane can hold the whole answer

__device__ __forceinline__ unsigned long long r3d_tk_key_f32(float v, unsigned idx) {
    unsigned fu = __float_as_uint(v);
    fu = (fu & 0x80000000u) ? ~fu : (fu | 0x80000000u);
    return ((unsigned long long)fu << 32) | (unsigned)(0xFFFFFFFFu - idx);
}

__device__ __forceinline__ float r3d_tk_val(unsigned long long k) {
    unsigned u = (unsigned)(k >> 32);
    u = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
    return __uint_as_float(u);
}

__device__ __forceinline__ int r3d_tk_idx(unsigned long long k) {
    return (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull));
}

template <int N>
__device__ __forceinline__ unsigned long long r3d_tk_xor_max(unsigned long long m) {
    const unsigned long long p = r3d_xor_lane_u64<N>(m);
    return p > m ? p : m;
}

template <int PT>
__device__ __forceinline__ void r3d_tk_insert(unsigned long long* t, unsigned long long k) {
    if (k > t[PT - 1]) {
        unsigned long long c = k;
#pragma unroll
        for (int q = 0; q < PT; q++) {
            const unsigned long long hi = t[q] > c ? t[q] : c;
            c = t[q] > c ? c : t[q];
            t[q] = hi;
        }
    }
}

template <int PT>
__device__ __forceinline__ unsigned long long r3d_tk_round(unsigned long long* t) {
    unsigned long long m = t[0];
    m = r3d_tk_xor_max<16>(m);
    m = r3d_tk_xor_max<8>(m);
    m = r3d_tk_xor_max<4>(m);
    m = r3d_tk_xor_max<2>(m);
    m = r3d_tk_xor_max<1>(m);
    if (m == t[0]) {
        // Drop the head and re-sort: shift the rest down, 0 at the tail.
#pragma unroll
        for (int q = 0; q < PT - 1; q++) t[q] = t[q + 1];
        t[PT - 1] = 0ull;
    }
    return m;
}

__global__ __launch_bounds__(R3D_TOPK_WAVE) void r3d_rowtopk_merge_kernel(
    const float* __restrict__ G, float* __restrict__ OV, int* __restrict__ OI, int R, int n)
{
    const int m = blockIdx.x;
    const int lane = threadIdx.x;

    unsigned long long t[R3D_TOPK_PT2];
#pragma unroll
    for (int q = 0; q < R3D_TOPK_PT2; q++) t[q] = 0ull;

    const float* const row = G + (size_t)m * n * 2;
    for (int i = lane; i < n; i += R3D_TOPK_WAVE) {
        int id;
        __builtin_memcpy(&id, row + (size_t)i * 2, sizeof(int));
        if (id < 0) continue;
        r3d_tk_insert<R3D_TOPK_PT2>(t, r3d_tk_key_f32(row[(size_t)i * 2 + 1], (unsigned)id));
    }

    for (int r = 0; r < R; r++) {
        const unsigned long long w = r3d_tk_round<R3D_TOPK_PT2>(t);
        if (lane == 0) {
            OV[(size_t)m * R + r] = w ? r3d_tk_val(w) : -INFINITY;
            OI[(size_t)m * R + r] = w ? r3d_tk_idx(w) : -1;
        }
    }
}

extern "C" int r3d_rad_row_topk_merge(const RadArgs* a, RadStream s)
{
    const RadTensor* g   = r3d_opt(a, 0);
    const RadTensor* idx = r3d_opt(a, 1);
    const RadTensor* val = r3d_opt(a, 2);
    if (!r3d_p(g) || !r3d_p(idx) || !r3d_wp(val)) return R3D_NO(RAD_E_INVAL);
    if (g->dtype != RAD_F32) return R3D_NO(RAD_E_DTYPE);

    const long long M = (g->rank >= 1) ? g->shape[0] : r3d_geti_or(a, "M", 0);
    const long long R = r3d_geti_or(a, "R", 0);
    const long long W = r3d_geti_or(a, "world_size", 0);
    if (M < 1 || R < 1 || W < 1) return R3D_NO(RAD_E_SHAPE);
    if (R > 32) return R3D_NO(RAD_E_SHAPE);

    dim3 grid((unsigned)M), blk(R3D_TOPK_WAVE);
    R3D_KLAUNCH(r3d_rowtopk_merge_kernel, grid, blk, (hipStream_t)s,
                (const float*)r3d_p(g), (float*)r3d_wp(val), (int*)r3d_wp(idx),
                (int)R, (int)(W * R));
    return (int)hipGetLastError();
}
