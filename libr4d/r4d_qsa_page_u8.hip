// The QSA pager (docs/OPS.md, "`qsa_page`"; the reference is libref's ref_qsa_page, which this
// kernel is held to byte for byte).
//
// For each position (m, j) of the selection holding a cold entry (<= -2), the first `block_bytes`
// of store row -2 - entry (this layer's view, its offset folded in by the plugin: the row pitch is
// the slot's bytes) are copied into pool block scratch[m][j], and the entry becomes that block id.
// Resident (>= 0) and empty (-1) entries are untouched, so a row without a cold entry costs one
// compare a position.
//
// One launch per layer per step: a workgroup owns one position, every position has
// one writer of its `sel` word, and the scratch ids of cold positions are pairwise distinct (the
// reference refuses otherwise), so no two workgroups write the same byte. The grid is as wide as the
// selection, not the count of cold entries, because the count is device data the launch cannot see.
// The only atomic is the optional counter of rewritten entries (a sum, so order-free); no byte of
// `sel` or `pool` has two writers.
//
// A device kernel cannot refuse after the fact, as the reference does: an entry whose slot is
// outside the store or whose scratch id is outside the pool is skipped and left as it was, so a bad
// table never writes outside the planes. The scheduler's tables are valid by construction.
//
// Returns a RAD status rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"

#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 128;

__global__ __launch_bounds__(kThreads) void qsa_page_kernel(
    int32_t* __restrict__ sel, const int32_t* __restrict__ scratch,
    const uint8_t* __restrict__ store, uint8_t* __restrict__ pool,
    int w, int64_t block_bytes, int64_t slots, int64_t blocks,
    int64_t sel_ld, int64_t sc_ld, int64_t slot_pitch, int64_t pool_pitch, int wide,
    int64_t* __restrict__ paged) {

    const int j = (int)blockIdx.x, m = (int)blockIdx.y;
    int32_t* const so = sel + (size_t)m * sel_ld + j;
    const int32_t e = *so;
    if (e > -2) return;                       /* resident or empty: pass through */
    const int64_t slot = -2 - (int64_t)e;
    const int64_t pg = scratch[(size_t)m * sc_ld + j];
    if (slot >= slots || pg < 0 || pg >= blocks) return;

    const uint8_t* src = store + (size_t)slot * slot_pitch;
    uint8_t*       dst = pool  + (size_t)pg * pool_pitch;
    if (wide) {
        /* Both planes, both pitches and the length are 16-byte multiples: one 16-byte move a lane. */
        for (int64_t i = (int64_t)threadIdx.x; i < block_bytes / 16; i += kThreads)
            ((uint4*)dst)[i] = ((const uint4*)src)[i];
    } else {
        for (int64_t i = (int64_t)threadIdx.x; i < block_bytes; i += kThreads) dst[i] = src[i];
    }
    /* The entry is rewritten by the one thread that no copy depends on; the copy reads `store`,
     * not `sel`, so no barrier is needed. */
    if (threadIdx.x == 0) {
        *so = (int32_t)pg;
        /* One add a rewritten entry, from the thread that owns it: the count is a sum, so the order
         * the workgroups finish in does not change it. Absent (null), nothing is counted. */
        if (paged) atomicAdd((unsigned long long*)paged, 1ull);
    }
}

}  // namespace

//   sel      [M][w]               i32 in place, `sel_ld` elements a row; w = topk + 1
//   scratch  [M][w]               i32, `sc_ld` elements a row
//   store    [slots][>= bb]       u8, row pitch `slot_pitch` bytes
//   pool     [blocks][>= bb]      u8, row pitch `pool_pitch` bytes; only scratch blocks written
static int r4d_qsa_page_u8_emit(long sel, long scratch, long store, long pool, int M, int w,
                               long long bb, long long slots, long long blocks, long long sel_ld,
                               long long sc_ld, long long slot_pitch, long long pool_pitch,
                               long paged, RadStream stream) {
    if (!sel || !scratch || !store || !pool) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || w <= 0 || bb <= 0 || slots <= 0 || blocks <= 0) return R4D_NO(RAD_E_SHAPE);
    const int wide = ((((uintptr_t)store | (uintptr_t)pool) |
                       (uint64_t)(slot_pitch | pool_pitch | bb)) & 15u) == 0u;
    return R4D_EMIT(stream, qsa_page_kernel, dim3((unsigned)w, (unsigned)M), dim3(kThreads), 0,
                    (int32_t*)sel, (const int32_t*)scratch, (const uint8_t*)store,
                    (uint8_t*)pool, w, (int64_t)bb, (int64_t)slots, (int64_t)blocks,
                    (int64_t)sel_ld, (int64_t)sc_ld, (int64_t)slot_pitch, (int64_t)pool_pitch,
                    wide, (int64_t*)paged);
}

// ============================================================== the plugin ABI entry point

enum { QP_SEL = 0, QP_STORE = 1, QP_POOL = 2, QP_SCRATCH = 3, QP_PAGED = 4 };

extern "C" int r4d_rad_qsa_page(const RadArgs* a, RadStream stream) {
    const RadTensor* se = r4d_opt(a, QP_SEL);
    const RadTensor* st = r4d_opt(a, QP_STORE);
    const RadTensor* pl = r4d_opt(a, QP_POOL);
    const RadTensor* sc = r4d_opt(a, QP_SCRATCH);
    if (!se || !st || !pl || !sc) return R4D_NO(RAD_E_INVAL);
    /* `pool` is the layer's cache in its own shape ([blocks, ...] of the group's dtype, as the core
     * binds a KV operand); the pager moves bytes, so a block is its leading stride in bytes,
     * contiguous. `store` is [slots, slot_bytes] of bytes. */
    if (se->rank != 2 || st->rank != 2 || pl->rank < 2 || sc->rank != 2)
        return R4D_NO(RAD_E_SHAPE);
    if (se->dtype != RAD_I32 || sc->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    const long long ebits = rad_dtype_bits(pl->dtype);
    if (st->dtype != RAD_U8 || ebits <= 0 || ebits % 8) return R4D_NO(RAD_E_DTYPE);
    if (st->stride[1] != 1 || pl->stride[pl->rank - 1] != 1) return R4D_NO(RAD_E_STRIDE);
    /* A block's elements past the leading dimension are packed, so its bytes from its start are
     * contiguous for `inner` elements; the pitch between blocks may be anything. */
    long long inner = 1;
    for (uint32_t d = pl->rank - 1; d >= 1; --d) {
        if (pl->stride[d] != inner) return R4D_NO(RAD_E_STRIDE);
        inner *= pl->shape[d];
    }
    const long long pitch = pl->stride[0] * (ebits / 8);
    const long long topk = r4d_geti_or(a, "topk", 0);
    const long long bb   = r4d_geti_or(a, "block_bytes", 0);
    if (topk <= 0 || bb <= 0) return R4D_NO(RAD_E_INVAL);
    const long long M = se->shape[0], w = topk + 1;
    if (M < 1 || se->shape[1] < w || sc->shape[0] < M || sc->shape[1] < w)
        return R4D_NO(RAD_E_SHAPE);
    if (st->shape[1] < bb || inner * (ebits / 8) < bb || pitch < bb) return R4D_NO(RAD_E_SHAPE);
    const long long sp = r4d_row_pitch(se), cp = r4d_row_pitch(sc);
    if (sp < 0 || cp < 0 || st->stride[0] < bb) return R4D_NO(RAD_E_STRIDE);
    /* The optional counter: one i64 the kernel adds to. */
    const RadTensor* pg = r4d_opt(a, QP_PAGED);
    if (pg && pg->dtype != RAD_I64) return R4D_NO(RAD_E_DTYPE);
    return r4d_qsa_page_u8_emit(r4d_addr(se), r4d_addr(sc), r4d_addr(st), r4d_addr(pl), (int)M,
                                (int)w, bb, st->shape[0], pl->shape[0], sp, cp, st->stride[0],
                                pitch, r4d_addr(pg), stream);
}
