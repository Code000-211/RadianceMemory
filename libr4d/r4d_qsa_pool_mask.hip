// The hierarchical QSA indexer's restriction (--qsa-pool, arch/common/rad_qsa.h): a later QSA
// layer ranks only the candidate pool the first QSA layer chose, so every score of its plane that
// the pool does not name is set to -inf, which qsa_select reads as dead in both rankings. libref's
// ref_qsa_pool_mask is the contract (docs/OPS.md, "`qsa_pool_mask`"); this kernel gives its bytes.
//
// `pool` is [M, pool + 1] block INDICES (the first layer's qsa_select over an identity table, -1
// padded), not page ids: the mask is applied before this layer's own select, which maps indices
// to pages through the real table.
//
// One workgroup a row, in three phases separated by barriers: every pool entry naming a column
// of the row saves that column's score in LDS; the whole row is set to -inf; the saved scores are
// written back. Repeated entries save and restore the same value, so the result does not depend
// on which thread wins. O(cols + pool) a row, no search. `pool + 1` <= kMaxPool entries fit the
// LDS (32 KiB); the host refuses a wider pool.
//
// The phases' global writes are ordered by __syncthreads, which orders global as well as LDS
// accesses within the workgroup; no other workgroup touches the row.
//
// Returns a RAD status rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"

#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxPool = 4096;   // pool + 1 entries: 16 KiB of indices and 16 KiB of scores

__global__ __launch_bounds__(kThreads) void qsa_pool_mask_kernel(
    const int32_t* __restrict__ pool, float* __restrict__ score,
    int M, int cols, int np1, int p_ld, int s_ld) {

    __shared__ int32_t s_idx[kMaxPool];
    __shared__ float   s_val[kMaxPool];

    const int m = (int)blockIdx.x;
    if (m >= M) return;
    float* row = score + (size_t)m * s_ld;
    const int32_t* pr = pool + (size_t)m * p_ld;

    for (int j = (int)threadIdx.x; j < np1; j += kThreads) {
        const int32_t b = pr[j];
        const bool in = b >= 0 && b < cols;
        s_idx[j] = in ? b : -1;
        s_val[j] = in ? row[b] : 0.0f;
    }
    __syncthreads();
    for (int b = (int)threadIdx.x; b < cols; b += kThreads) row[b] = -__builtin_inff();
    __syncthreads();
    for (int j = (int)threadIdx.x; j < np1; j += kThreads)
        if (s_idx[j] >= 0) row[s_idx[j]] = s_val[j];
}

}  // namespace

//   pool   [M][np1]   i32, `p_ld` elements a row; block indices, -1 (or out of range) names nothing
//   score  [M][cols]  f32, `s_ld` elements a row; rewritten in place
static int r4d_qsa_pool_mask_emit(long pool, long score, int M, int cols, int np1, int p_ld,
                                  int s_ld, RadStream stream) {
    if (!pool || !score) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || cols <= 0 || np1 <= 1 || np1 > kMaxPool) return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(stream, qsa_pool_mask_kernel, dim3((unsigned)M), dim3(kThreads), 0,
                    (const int32_t*)pool, (float*)score, M, cols, np1, p_ld, s_ld);
}

// ============================================================== the plugin ABI entry point

enum { QP_POOL = 0, QP_SCORE = 1 };

extern "C" int r4d_rad_qsa_pool_mask(const RadArgs* a, RadStream stream) {
    const RadTensor* po = r4d_opt(a, QP_POOL);
    const RadTensor* sc = r4d_opt(a, QP_SCORE);
    if (!po || !sc) return R4D_NO(RAD_E_INVAL);
    if (po->rank != 2 || sc->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (po->dtype != RAD_I32 || sc->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (po->stride[1] != 1 || sc->stride[1] != 1) return R4D_NO(RAD_E_STRIDE);
    const long long np = r4d_geti_or(a, "pool", 0);
    if (np <= 0 || np + 1 > kMaxPool) return R4D_NO(RAD_E_SHAPE);
    const long long M = sc->shape[0], cols = sc->shape[1];
    if (M < 1 || cols < 1 || po->shape[0] < M || po->shape[1] < np + 1) return R4D_NO(RAD_E_SHAPE);
    const long long pp = r4d_row_pitch(po), sp = r4d_row_pitch(sc);
    if (pp < 0 || sp < 0) return R4D_NO(RAD_E_STRIDE);
    return r4d_qsa_pool_mask_emit(r4d_addr(po), r4d_addr(sc), (int)M, (int)cols, (int)(np + 1),
                                  (int)pp, (int)sp, stream);
}
