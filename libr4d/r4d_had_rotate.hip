// had_rotate: the normalised Walsh-Hadamard rotation alone, in place on bf16 rows (docs/OPS.md).
// Each `group`-wide block of a row becomes H x / sqrt(group). A layer whose KV is stored rotated
// (--kv-recipe `rotated`) runs it over q, k and v before its cache store and over the attention
// output after the attention: H / sqrt(group) is its own inverse, so the scores and the output are
// the unrotated ones and the attention and store kernels stay as they are.
//
// One wave a block, r4d_fwht_wave in registers (lane l holds elements l*VPL .. l*VPL+VPL-1, no LDS,
// no barrier), so a block is read whole before any of it is written and the op is safe in place.
// The bytes are libref's (ref_had_rotate), held to them on the host emulator (tests/r4d_emu,
// test_had.cpp):
//   - the butterflies pair and order as ref_common.h's fwht (r4d_fwht.h says why the cross-lane
//     form is bit-identical: the high lane's `-v + o` is `a - b`);
//   - then one f32 multiply by 1/sqrt(group), formed on the host as ref_common.h's had_norm forms
//     it (a power of two, times 1/sqrt(2) rounded to f32 for an odd log2), so both sides multiply
//     by the same float;
//   - then bf16 round-to-nearest-even with a NaN kept NaN, as rad_f32_to_bf16 stores.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_args.h"
#include "r4d_fwht.h"

#include <cmath>

namespace {

constexpr uint32_t kWave = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves = kThreads / kWave;

// Block `u` of `units`: row u / nblk, block u % nblk of it, rows `ld` elements apart. A wave past
// the end leaves as a whole, so no exchange runs on a partial wave.
template <int G>
__global__ __launch_bounds__(kThreads) void had_rotate_kernel(uint16_t* __restrict__ x,
                                                              uint32_t units, uint32_t nblk,
                                                              uint32_t ld, float s) {
    constexpr int VPL = G / (int)kWave;
    const uint32_t lane = threadIdx.x & (kWave - 1);
    const uint32_t u = blockIdx.x * kWaves + threadIdx.x / kWave;
    if (u >= units) return;
    uint16_t* p = x + (size_t)(u / nblk) * ld + (size_t)(u % nblk) * G + (size_t)lane * VPL;
    float v[VPL];
#pragma unroll
    for (int j = 0; j < VPL; ++j) v[j] = bf16_to_f32(p[j]);
    r4d_fwht_wave<G>(v, (int)lane);
#pragma unroll
    for (int j = 0; j < VPL; ++j) p[j] = (uint16_t)(bf16_hi_rne_nan(v[j] * s) >> 16);
}

}  // namespace

// 1/sqrt(g) as libref's had_norm forms it (see the file header).
static float had_norm(long long g) {
    int k = 0;
    while ((1LL << k) < g) ++k;
    const float p = std::ldexp(1.0f, -(k / 2));
    return (k & 1) ? p * 0.70710678118654752f : p;
}

// x: [rows, width] bf16 over its trailing axes, width a multiple of `n` and `n` of `group` (128 or
// 256, the rows' constraint), rows at one pitch (a step's prefix of a wider buffer, or a slice).
extern "C" int r4d_rad_had_rotate(const RadArgs* a, RadStream stream) {
    const RadTensor* x = r4d_opt(a, 0);
    if (!x) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const long long g = r4d_geti_or(a, "group", 0), n = r4d_geti_or(a, "n", 0);
    if ((g != 128 && g != 256) || n <= 0 || n % g) return R4D_NO(RAD_E_SHAPE);
    long long rows = 0, nblk = 0, ld = 0;
    if (!r4d_head_rows(x, g, &rows, &nblk, &ld)) return R4D_NO(RAD_E_STRIDE);
    if (rows <= 0 || (nblk * g) % n) return R4D_NO(RAD_E_SHAPE);
    const long long units = rows * nblk;
    if (units > 0xFFFFFFFFLL || ld > 0xFFFFFFFFLL) return R4D_NO(RAD_E_SHAPE);
    const dim3 grid((uint32_t)((units + kWaves - 1) / kWaves));
    const float s = had_norm(g);
    return g == 256
        ? R4D_EMIT(stream, had_rotate_kernel<256>, grid, dim3(kThreads), 0, (uint16_t*)x->data,
                   (uint32_t)units, (uint32_t)nblk, (uint32_t)ld, s)
        : R4D_EMIT(stream, had_rotate_kernel<128>, grid, dim3(kThreads), 0, (uint16_t*)x->data,
                   (uint32_t)units, (uint32_t)nblk, (uint32_t)ld, s);
}
