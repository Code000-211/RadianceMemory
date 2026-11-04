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
// With `store_layout` and `pool_layout` (a tiered layer, RAD_KVQ_DEMOTE) the copy expands: the
// slot holds the fragment's rows in the demote layout, and each of the workgroup's four waves turns
// rows w, w + 4, ... back into the pool's layout (r4d_kvx.h, libref's kvx_rows byte for byte).
//
// `kv_transcode` (docs/OPS.md) is the same row codec over a plane of rows, a wave a row: the move
// path's requantise into the demote layout before a block's copy to the store, and its expand after
// a promotion's copy back (core/runtime/tierexec.cpp).
//
// With `store_layout2` as well (RAD_KVQ_DEMOTE2) each slot holds one of two demote layouts, the
// second where the `slot_layout` table's byte for it is not 0; the workgroup reads that byte and
// expands from its slot's own layout.
//
// `qsa_count` (docs/OPS.md) is importance-tiered demotion's selection counter, a thread a selection
// entry: a full row (topk complete blocks) adds one to each of its pool ids' counters.
//
// Returns a RAD status rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_kvx.h"

#include <algorithm>
#include <cstring>

#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 128;

__global__ __launch_bounds__(kThreads) void qsa_page_kernel(
    int32_t* __restrict__ sel, const int32_t* __restrict__ scratch,
    const uint8_t* __restrict__ store, uint8_t* __restrict__ pool,
    int w, int64_t block_bytes, int64_t slots, int64_t blocks,
    int64_t sel_ld, int64_t sc_ld, int64_t slot_pitch, int64_t pool_pitch, int wide,
    int64_t* __restrict__ paged, int xf, int xt, uint32_t hd, int xf2,
    const uint8_t* __restrict__ slot_layout) {

    const int j = (int)blockIdx.x, m = (int)blockIdx.y;
    int32_t* const so = sel + (size_t)m * sel_ld + j;
    const int32_t e = *so;
    if (e > -2) return;                       /* resident or empty: pass through */
    const int64_t slot = -2 - (int64_t)e;
    const int64_t pg = scratch[(size_t)m * sc_ld + j];
    if (slot >= slots || pg < 0 || pg >= blocks) return;

    const uint8_t* src = store + (size_t)slot * slot_pitch;
    uint8_t*       dst = pool  + (size_t)pg * pool_pitch;
    if (xf >= 0) {
        /* Expanding: a wave a row, every lane of it through the row's reductions, from the slot's
         * own layout (one read of its byte, the same for the whole workgroup). */
        const int from = xf2 >= 0 && slot_layout[slot] ? xf2 : xf;
        const int64_t rf = r4d_kvx::row_bytes(from, hd), rt = r4d_kvx::row_bytes(xt, hd);
        const uint32_t wv = threadIdx.x / 32u, lane = threadIdx.x % 32u;
        for (int64_t r = wv; r < block_bytes / rt; r += kThreads / 32)
            r4d_kvx::row(from, xt, src + r * rf, dst + r * rt, hd, lane);
    } else if (wide) {
        /* Both planes, both pitches and the length are 16-byte multiples: one 16-byte move a lane. */
        for (int64_t i = (int64_t)threadIdx.x; i < block_bytes / 16; i += kThreads)
            ((uint4*)dst)[i] = ((const uint4*)src)[i];
    } else {
        for (int64_t i = (int64_t)threadIdx.x; i < block_bytes; i += kThreads) dst[i] = src[i];
    }
    /* The entry is rewritten by one thread, and only after every thread of the workgroup has read
     * it: a wave that started after the rewrite would read a resident id and skip its share of
     * the copy (the waves of a workgroup are not scheduled together; the emulator runs them one
     * after another and showed it). The early returns above are uniform, since no thread writes
     * the entry before this barrier. */
    __syncthreads();
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
                               long paged, int xf, int xt, int hd, int xf2, long slot_layout,
                               RadStream stream) {
    if (!sel || !scratch || !store || !pool) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || w <= 0 || bb <= 0 || slots <= 0 || blocks <= 0) return R4D_NO(RAD_E_SHAPE);
    const int wide = xf < 0 && ((((uintptr_t)store | (uintptr_t)pool) |
                       (uint64_t)(slot_pitch | pool_pitch | bb)) & 15u) == 0u;
    return R4D_EMIT(stream, qsa_page_kernel, dim3((unsigned)w, (unsigned)M), dim3(kThreads), 0,
                    (int32_t*)sel, (const int32_t*)scratch, (const uint8_t*)store,
                    (uint8_t*)pool, w, (int64_t)bb, (int64_t)slots, (int64_t)blocks,
                    (int64_t)sel_ld, (int64_t)sc_ld, (int64_t)slot_pitch, (int64_t)pool_pitch,
                    wide, (int64_t*)paged, xf, xt, (uint32_t)hd, xf2,
                    (const uint8_t*)slot_layout);
}

// A thread an entry (m, j < topk): a full row adds one to its pool id's counter.
__global__ __launch_bounds__(256) void qsa_count_kernel(
    const int32_t* __restrict__ sel, const int32_t* __restrict__ nsel, const int32_t* __restrict__ pos,
    uint32_t* __restrict__ count, int64_t M, int topk, int ratio, int64_t sel_ld, int64_t ns_ld,
    int64_t po_ld, int64_t n, const int32_t* __restrict__ cu, int64_t n_cu) {
    const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= M * topk) return;
    const int64_t m = i / topk, j = i % topk;
    /* With `cu`, only a row opening a (non-empty) sequence: binary search of the row offsets. */
    if (cu) {
        int64_t lo = 0, hi = n_cu - 2;       /* the last s with cu[s] <= m */
        if (hi < 0 || cu[0] > m) return;
        while (lo < hi) {
            const int64_t mid = (lo + hi + 1) / 2;
            if (cu[mid] <= m) lo = mid; else hi = mid - 1;
        }
        if (cu[lo] != m || cu[lo + 1] <= m) return;
    }
    const int32_t p = pos[m * po_ld];
    if (nsel[m * ns_ld] - ((p + 1) % ratio != 0 ? 1 : 0) != topk) return;
    const int32_t e = sel[m * sel_ld + j];
    /* An add per counted entry: a sum, so the order the threads run in does not change it. */
    if (e >= 0 && e < n) atomicAdd(count + e, 1u);
}

// A wave a row: x [M][row(from)] -> out [M][row(to)], row pitches in bytes.
__global__ __launch_bounds__(32) void kv_transcode_kernel(
    const uint8_t* __restrict__ x, uint8_t* __restrict__ out, int xf, int xt, uint32_t hd,
    int64_t x_ld, int64_t o_ld) {
    const int64_t r = blockIdx.x;
    r4d_kvx::row(xf, xt, x + r * x_ld, out + r * o_ld, hd, threadIdx.x);
}

// The layout words of kv_transcode and qsa_page, as libref's kvx_layout: -1 for anything else.
static int kvx_layout(const char* s) {
    if (!s) return -1;
    return !std::strcmp(s, "fp8") ? r4d_kvx::kFp8 : !std::strcmp(s, "k4h_v4") ? r4d_kvx::kK4h
         : !std::strcmp(s, "k4h_v2h") ? r4d_kvx::kK4hV2h : !std::strcmp(s, "i8h") ? r4d_kvx::kI8h
         : -1;
}

// Every row is 4-byte aligned (the layouts' widths are multiples of four), as the codec's word
// loads and stores need; head_dim a multiple of 128 up to 512, the codec's lane map.
extern "C" int r4d_rad_kv_transcode(const RadArgs* a, RadStream stream) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* o = r4d_opt(a, 1);
    if (!x || !o) return R4D_NO(RAD_E_INVAL);
    const int xf = kvx_layout(r4d_gets(a, "from")), xt = kvx_layout(r4d_gets(a, "to"));
    const long long hd = r4d_geti_or(a, "head_dim", 0);
    if (xf < 0 || xt < 0 || xf == xt) return R4D_NO(RAD_E_INVAL);
    if (hd <= 0 || hd % 128 || hd > 128 * r4d_kvx::kMaxNi) return R4D_NO(RAD_E_SHAPE);
    if (x->dtype != RAD_U8 || o->dtype != RAD_U8) return R4D_NO(RAD_E_DTYPE);
    if (x->rank != 2 || o->rank != 2 || x->shape[1] != r4d_kvx::row_bytes(xf, hd) ||
        o->shape[1] != r4d_kvx::row_bytes(xt, hd) || x->shape[0] != o->shape[0] || x->shape[0] <= 0)
        return R4D_NO(RAD_E_SHAPE);
    if (x->stride[1] != 1 || o->stride[1] != 1 || ((x->stride[0] | o->stride[0]) & 3) ||
        (((uintptr_t)r4d_addr(x) | (uintptr_t)r4d_addr(o)) & 3u))
        return R4D_NO(RAD_E_STRIDE);
    return R4D_EMIT(stream, kv_transcode_kernel, dim3((unsigned)x->shape[0]), dim3(32), 0,
                    (const uint8_t*)r4d_addr(x), (uint8_t*)r4d_addr(o), xf, xt, (uint32_t)hd,
                    (int64_t)x->stride[0], (int64_t)o->stride[0]);
}

// ============================================================== the plugin ABI entry point

enum { QP_SEL = 0, QP_STORE = 1, QP_POOL = 2, QP_SCRATCH = 3, QP_PAGED = 4, QP_SLOTS = 5 };

//   sel [M][>= topk + 1] i32, nsel [M] i32, pos [M] i32, count [n] u32 added to in place.
extern "C" int r4d_rad_qsa_count(const RadArgs* a, RadStream stream) {
    const RadTensor* se = r4d_opt(a, 0);
    const RadTensor* ns = r4d_opt(a, 1);
    const RadTensor* po = r4d_opt(a, 2);
    const RadTensor* ct = r4d_opt(a, 3);
    if (!se || !ns || !po || !ct) return R4D_NO(RAD_E_INVAL);
    if (se->dtype != RAD_I32 || ns->dtype != RAD_I32 || po->dtype != RAD_I32 || ct->dtype != RAD_U32)
        return R4D_NO(RAD_E_DTYPE);
    const long long topk = r4d_geti_or(a, "topk", 0), ratio = r4d_geti_or(a, "ratio", 0);
    if (topk <= 0 || ratio <= 0 || se->rank != 2 || ct->rank != 1) return R4D_NO(RAD_E_SHAPE);
    const long long M = se->shape[0], sp = r4d_row_pitch(se);
    if (M <= 0 || se->shape[1] < topk + 1 || ns->shape[0] < M || po->shape[0] < M ||
        ct->shape[0] <= 0)
        return R4D_NO(RAD_E_SHAPE);
    if (sp < 0 || ct->stride[0] != 1) return R4D_NO(RAD_E_STRIDE);
    const RadTensor* cu = r4d_opt(a, 4);
    if (cu && (cu->dtype != RAD_I32 || cu->stride[0] != 1)) return R4D_NO(RAD_E_DTYPE);
    const long long cells = M * topk;
    return R4D_EMIT(stream, qsa_count_kernel, dim3((unsigned)((cells + 255) / 256)), dim3(256), 0,
                    (const int32_t*)r4d_addr(se), (const int32_t*)r4d_addr(ns),
                    (const int32_t*)r4d_addr(po), (uint32_t*)r4d_addr(ct), (int64_t)M, (int)topk,
                    (int)ratio, (int64_t)sp, (int64_t)ns->stride[0], (int64_t)po->stride[0],
                    (int64_t)ct->shape[0], cu ? (const int32_t*)r4d_addr(cu) : nullptr,
                    cu ? (int64_t)cu->shape[0] : 0ll);
}

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
    /* Expanding: both layout words or neither; the store's fragment is as many rows of its own
     * width as the pool's `block_bytes` holds of the pool's. */
    const char* sl = r4d_gets(a, "store_layout");
    const char* pw = r4d_gets(a, "pool_layout");
    const char* sl2 = r4d_gets(a, "store_layout2");
    int xf = -1, xt = -1, xf2 = -1;
    long long hd = 0, sbb = bb;
    if (sl || pw) {
        xf = kvx_layout(sl); xt = kvx_layout(pw); hd = r4d_geti_or(a, "head_dim", 0);
        if (xf < 0 || xt < 0 || xf == xt) return R4D_NO(RAD_E_INVAL);
        if (hd <= 0 || hd % 128 || hd > 128 * r4d_kvx::kMaxNi ||
            bb % r4d_kvx::row_bytes(xt, hd))
            return R4D_NO(RAD_E_SHAPE);
        sbb = bb / r4d_kvx::row_bytes(xt, hd) * r4d_kvx::row_bytes(xf, hd);
        if (((uintptr_t)r4d_addr(st) | (uintptr_t)r4d_addr(pl) |
             (uint64_t)(st->stride[0] | pitch)) & 3u)
            return R4D_NO(RAD_E_STRIDE);
    }
    /* The second demote layout, with the slot table it is picked by (libref's refusals). */
    const RadTensor* sly = r4d_opt(a, QP_SLOTS);
    if (sl2) {
        xf2 = kvx_layout(sl2);
        if (xf < 0 || xf2 < 0 || xf2 == xf || xf2 == xt || !sly) return R4D_NO(RAD_E_INVAL);
        if (sly->dtype != RAD_U8) return R4D_NO(RAD_E_DTYPE);
        if (sly->shape[0] < st->shape[0] || sly->stride[0] != 1) return R4D_NO(RAD_E_SHAPE);
        sbb = std::max(sbb, bb / r4d_kvx::row_bytes(xt, hd) * r4d_kvx::row_bytes(xf2, hd));
    }
    if (st->shape[1] < sbb || inner * (ebits / 8) < bb || pitch < bb) return R4D_NO(RAD_E_SHAPE);
    const long long sp = r4d_row_pitch(se), cp = r4d_row_pitch(sc);
    if (sp < 0 || cp < 0 || st->stride[0] < sbb) return R4D_NO(RAD_E_STRIDE);
    /* The optional counter: one i64 the kernel adds to. */
    const RadTensor* pg = r4d_opt(a, QP_PAGED);
    if (pg && pg->dtype != RAD_I64) return R4D_NO(RAD_E_DTYPE);
    return r4d_qsa_page_u8_emit(r4d_addr(se), r4d_addr(sc), r4d_addr(st), r4d_addr(pl), (int)M,
                                (int)w, bb, st->shape[0], pl->shape[0], sp, cp, st->stride[0],
                                pitch, r4d_addr(pg), xf, xt, (int)hd, xf2,
                                sl2 ? r4d_addr(sly) : 0, stream);
}
