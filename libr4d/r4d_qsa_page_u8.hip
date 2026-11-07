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
// Returns a RAD status rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_kvx.h"

#include <cstring>

#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 128;

// One fragment from a store slot into a pool block, by the whole workgroup (see the kernel below).
__device__ inline void page_copy(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                 int64_t block_bytes, int wide, int xf, int xt, uint32_t hd) {
    if (xf >= 0) {
        /* Expanding: a wave a row, every lane of it through the row's reductions. */
        const int64_t rf = r4d_kvx::row_bytes(xf, hd), rt = r4d_kvx::row_bytes(xt, hd);
        const uint32_t wv = threadIdx.x / 32u, lane = threadIdx.x % 32u;
        for (int64_t r = wv; r < block_bytes / rt; r += kThreads / 32)
            r4d_kvx::row(xf, xt, src + r * rf, dst + r * rt, hd, lane);
    } else if (wide) {
        /* Both planes, both pitches and the length are 16-byte multiples: one 16-byte move a lane. */
        for (int64_t i = (int64_t)threadIdx.x; i < block_bytes / 16; i += kThreads)
            ((uint4*)dst)[i] = ((const uint4*)src)[i];
    } else {
        for (int64_t i = (int64_t)threadIdx.x; i < block_bytes; i += kThreads) dst[i] = src[i];
    }
}

__global__ __launch_bounds__(kThreads) void qsa_page_kernel(
    int32_t* __restrict__ sel, const int32_t* __restrict__ scratch,
    const uint8_t* __restrict__ store, uint8_t* __restrict__ pool,
    int w, int64_t block_bytes, int64_t slots, int64_t blocks,
    int64_t sel_ld, int64_t sc_ld, int64_t slot_pitch, int64_t pool_pitch, int wide,
    int64_t* __restrict__ paged, int xf, int xt, uint32_t hd) {

    const int j = (int)blockIdx.x, m = (int)blockIdx.y;
    int32_t* const so = sel + (size_t)m * sel_ld + j;
    const int32_t e = *so;
    if (e > -2) return;                       /* resident or empty: pass through */
    const int64_t slot = -2 - (int64_t)e;
    const int64_t pg = scratch[(size_t)m * sc_ld + j];
    if (slot >= slots || pg < 0 || pg >= blocks) return;

    page_copy(store + (size_t)slot * slot_pitch, pool + (size_t)pg * pool_pitch, block_bytes, wide,
              xf, xt, hd);
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

// ------------------------------------------------------------------ the page cache (`tags`)
//
// Two launches. The plan, a workgroup a row, decides every cold position's page as libref's plan
// does and rewrites every cold entry of the row at once: a hit (some page of the row is tagged with
// the entry's slot, the lowest such page) to that page, a miss (the k-th in position order) to the
// k-th page no hit took (page order), whose tag becomes -(1 + slot): planned, not yet filled. The
// fill, a workgroup a page, copies the fragment into each planned page and only then tags it
// 1 + slot, so a page whose fill never ran is never a hit. A page's tag has one writer per launch
// and every byte of a page one writer; the plan's only atomics are on its own LDS and the counters.
//
// The lookup is an LDS hash of the row's tags (open addressing, linear probing, 2 * kMaxW slots):
// a key's slot is claimed by compare-and-swap and the page kept is the lowest (atomicMin), so the
// answer does not depend on which thread inserted first. A row is at most kMaxW pages wide.
//
// Unlike libref the plan cannot refuse; a page whose scratch id is outside the pool is never
// taken and a position whose slot is outside the store is left cold, so a bad table writes nothing
// out of bounds, and a miss for which no page remains (only possible with such pages) stays cold.

constexpr int kPlanThreads = 256;
constexpr int kMaxW        = 1025;            // topk up to 1024
constexpr int kHash        = 2048;            // a power of two >= 2 * kMaxW - 2
constexpr int kChunk       = (kMaxW + kPlanThreads - 1) / kPlanThreads;

__device__ inline uint32_t tag_hash(int32_t t) { return ((uint32_t)t * 2654435761u) >> 21; }

// Exclusive prefix sum over the workgroup of one value a thread; `tot` gets the total.
__device__ inline int scan_excl(int v, int* buf, int* tot) {
    const int t = (int)threadIdx.x;
    buf[t] = v;
    __syncthreads();
    for (int o = 1; o < kPlanThreads; o <<= 1) {
        const int x = t >= o ? buf[t - o] : 0;
        __syncthreads();
        buf[t] += x;
        __syncthreads();
    }
    const int incl = buf[t];
    *tot = buf[kPlanThreads - 1];
    __syncthreads();
    return incl - v;
}

__global__ __launch_bounds__(kPlanThreads) void qsa_page_plan_kernel(
    int32_t* __restrict__ sel, const int32_t* __restrict__ scratch, int32_t* __restrict__ tags,
    int w, int64_t slots, int64_t blocks, int64_t sel_ld, int64_t sc_ld, int64_t tg_ld,
    int64_t* __restrict__ paged, int64_t* __restrict__ hits) {
    __shared__ int32_t hkey[kHash];
    __shared__ int32_t hval[kHash];
    __shared__ int32_t busy[kMaxW];   // page: taken by a hit, or unusable
    __shared__ int32_t pick[kMaxW];   // position: its hit page, -2 a miss, -1 nothing to do
    __shared__ int32_t freep[kMaxW];  // the free pages in order
    __shared__ int32_t buf[kPlanThreads];
    const int m = (int)blockIdx.x, t = (int)threadIdx.x;
    int32_t* const so = sel + (size_t)m * sel_ld;
    const int32_t* const sc = scratch + (size_t)m * sc_ld;
    int32_t* const tg = tags + (size_t)m * tg_ld;

    for (int h = t; h < kHash; h += kPlanThreads) { hkey[h] = 0; hval[h] = 0x7fffffff; }
    __syncthreads();
    for (int p = t; p < w; p += kPlanThreads) {
        const int32_t pg = sc[p], k = tg[p];
        const bool ok = pg >= 0 && pg < blocks;
        busy[p] = ok ? 0 : 1;
        if (!ok || k <= 0) continue;
        for (uint32_t h = tag_hash(k);; h = (h + 1) & (kHash - 1)) {
            const int32_t o = atomicCAS(&hkey[h], 0, k);
            if (o == 0 || o == k) { atomicMin(&hval[h], p); break; }
        }
    }
    __syncthreads();
    for (int j = t; j < w; j += kPlanThreads) {
        const int32_t e = so[j];
        int32_t r = -1;
        if (e <= -2 && -2 - (int64_t)e < slots) {
            r = -2;
            const int32_t k = -1 - e;
            for (uint32_t h = tag_hash(k);; h = (h + 1) & (kHash - 1)) {
                const int32_t x = hkey[h];
                if (x == 0) break;
                if (x == k) { r = hval[h]; busy[r] = 1; break; }
            }
        }
        pick[j] = r;
    }
    __syncthreads();
    // The k-th free page in page order, then the k-th miss in position order takes it: a thread
    // owns a contiguous chunk of each, so one scan of its counts orders both.
    const int c0 = t * kChunk, c1 = c0 + kChunk < w ? c0 + kChunk : w;
    int nf = 0;
    for (int p = c0; p < c1; ++p) nf += busy[p] == 0;
    int n_free = 0;
    int base = scan_excl(nf, buf, &n_free);
    for (int p = c0; p < c1; ++p) if (busy[p] == 0) freep[base++] = p;
    int nm = 0;
    for (int j = c0; j < c1; ++j) nm += pick[j] == -2;
    int n_miss = 0;
    base = scan_excl(nm, buf, &n_miss);   // its barriers also publish freep
    int rw = 0, hit = 0;
    for (int j = c0; j < c1; ++j) {
        const int32_t r = pick[j];
        if (r >= 0) { so[j] = sc[r]; ++rw; ++hit; continue; }
        if (r != -2) continue;
        const int k = base++;
        if (k >= n_free) continue;        // no page left: only with unusable pages; stays cold
        const int p = freep[k];
        tg[p] = so[j] + 1;                // -(1 + slot): planned, filled by the next launch
        so[j] = sc[p];
        ++rw;
    }
    int n_rw = 0, n_hit = 0;
    (void)scan_excl(rw, buf, &n_rw);
    (void)scan_excl(hit, buf, &n_hit);
    if (t == 0) {
        if (paged && n_rw) atomicAdd((unsigned long long*)paged, (unsigned long long)n_rw);
        if (hits && n_hit) atomicAdd((unsigned long long*)hits, (unsigned long long)n_hit);
    }
}

// A workgroup a page (p, m): a planned page (tag < 0) receives its slot's fragment, then its tag.
__global__ __launch_bounds__(kThreads) void qsa_page_fill_kernel(
    const int32_t* __restrict__ scratch, int32_t* __restrict__ tags,
    const uint8_t* __restrict__ store, uint8_t* __restrict__ pool, int64_t block_bytes,
    int64_t slots, int64_t blocks, int64_t sc_ld, int64_t tg_ld, int64_t slot_pitch,
    int64_t pool_pitch, int wide, int xf, int xt, uint32_t hd) {
    const int p = (int)blockIdx.x, m = (int)blockIdx.y;
    int32_t* const to = tags + (size_t)m * tg_ld + p;
    const int32_t k = *to;
    if (k >= 0) return;
    const int64_t slot = -1 - (int64_t)k;
    const int64_t pg = scratch[(size_t)m * sc_ld + p];
    if (slot >= slots || pg < 0 || pg >= blocks) return;
    page_copy(store + (size_t)slot * slot_pitch, pool + (size_t)pg * pool_pitch, block_bytes, wide,
              xf, xt, hd);
    __syncthreads();                      // every thread read the tag before it changes
    if (threadIdx.x == 0) *to = (int32_t)(slot + 1);
}

}  // namespace

//   sel      [M][w]               i32 in place, `sel_ld` elements a row; w = topk + 1
//   scratch  [M][w]               i32, `sc_ld` elements a row
//   store    [slots][>= bb]       u8, row pitch `slot_pitch` bytes
//   pool     [blocks][>= bb]      u8, row pitch `pool_pitch` bytes; only scratch blocks written
static int r4d_qsa_page_u8_emit(long sel, long scratch, long store, long pool, int M, int w,
                               long long bb, long long slots, long long blocks, long long sel_ld,
                               long long sc_ld, long long slot_pitch, long long pool_pitch,
                               long paged, int xf, int xt, int hd, long tags, long long tg_ld,
                               long hits, RadStream stream) {
    if (!sel || !scratch || !store || !pool) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || w <= 0 || bb <= 0 || slots <= 0 || blocks <= 0) return R4D_NO(RAD_E_SHAPE);
    const int wide = xf < 0 && ((((uintptr_t)store | (uintptr_t)pool) |
                       (uint64_t)(slot_pitch | pool_pitch | bb)) & 15u) == 0u;
    if (tags) {
        if (w > kMaxW) return R4D_NO(RAD_E_UNSUPPORTED);
        const int st = R4D_EMIT(stream, qsa_page_plan_kernel, dim3((unsigned)M),
                                dim3(kPlanThreads), 0, (int32_t*)sel, (const int32_t*)scratch,
                                (int32_t*)tags, w, (int64_t)slots, (int64_t)blocks,
                                (int64_t)sel_ld, (int64_t)sc_ld, (int64_t)tg_ld, (int64_t*)paged,
                                (int64_t*)hits);
        if (st != RAD_OK) return st;
        return R4D_EMIT(stream, qsa_page_fill_kernel, dim3((unsigned)w, (unsigned)M),
                        dim3(kThreads), 0, (const int32_t*)scratch, (int32_t*)tags,
                        (const uint8_t*)store, (uint8_t*)pool, (int64_t)bb, (int64_t)slots,
                        (int64_t)blocks, (int64_t)sc_ld, (int64_t)tg_ld, (int64_t)slot_pitch,
                        (int64_t)pool_pitch, wide, xf, xt, (uint32_t)hd);
    }
    return R4D_EMIT(stream, qsa_page_kernel, dim3((unsigned)w, (unsigned)M), dim3(kThreads), 0,
                    (int32_t*)sel, (const int32_t*)scratch, (const uint8_t*)store,
                    (uint8_t*)pool, w, (int64_t)bb, (int64_t)slots, (int64_t)blocks,
                    (int64_t)sel_ld, (int64_t)sc_ld, (int64_t)slot_pitch, (int64_t)pool_pitch,
                    wide, (int64_t*)paged, xf, xt, (uint32_t)hd);
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

enum { QP_SEL = 0, QP_STORE = 1, QP_POOL = 2, QP_SCRATCH = 3, QP_PAGED = 4, QP_TAGS = 5,
       QP_HITS = 6 };

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
    int xf = -1, xt = -1;
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
    if (st->shape[1] < sbb || inner * (ebits / 8) < bb || pitch < bb) return R4D_NO(RAD_E_SHAPE);
    const long long sp = r4d_row_pitch(se), cp = r4d_row_pitch(sc);
    if (sp < 0 || cp < 0 || st->stride[0] < sbb) return R4D_NO(RAD_E_STRIDE);
    /* The optional counter: one i64 the kernel adds to. */
    const RadTensor* pg = r4d_opt(a, QP_PAGED);
    if (pg && pg->dtype != RAD_I64) return R4D_NO(RAD_E_DTYPE);
    /* The page cache: this layer's plane of slot tags, [rows >= M, >= w], and its hit counter. */
    const RadTensor* tg = r4d_opt(a, QP_TAGS);
    const RadTensor* ht = r4d_opt(a, QP_HITS);
    long long tp = 0;
    if (tg) {
        if (tg->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
        if (tg->rank != 2 || tg->shape[0] < M || tg->shape[1] < w) return R4D_NO(RAD_E_SHAPE);
        tp = r4d_row_pitch(tg);
        if (tp < 0) return R4D_NO(RAD_E_STRIDE);
    }
    if (ht && ht->dtype != RAD_I64) return R4D_NO(RAD_E_DTYPE);
    return r4d_qsa_page_u8_emit(r4d_addr(se), r4d_addr(sc), r4d_addr(st), r4d_addr(pl), (int)M,
                                (int)w, bb, st->shape[0], pl->shape[0], sp, cp, st->stride[0],
                                pitch, r4d_addr(pg), xf, xt, (int)hd, tg ? r4d_addr(tg) : 0, tp,
                                tg ? r4d_addr(ht) : 0, stream);
}
