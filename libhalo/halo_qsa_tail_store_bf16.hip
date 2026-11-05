/* Derived from libr4d/r4d_qsa_tail_store_bf16.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// The QSA indexer's tail: raw index keys of tokens that have not yet completed a block (a block key
// needs all `ratio` raw keys, and at decode most of them come from earlier steps).
//
// The tail is a per-layer linear KV state owned by the sequence (allocated zeroed, kept for the
// sequence's life, checkpointed with the linear state), not a plane indexed by the step-local batch
// row (arch/common/rad_qsa.h's Wire). `state` is one layer's cache, [slots, 1, ring, n]; `sidx`
// names the sequence's slot; position p lands at row p % ring.
//
// The ring is ratio + n_spec, not ratio: the next step restarts at the first rejected position, so
// a step may still owe positions back to p0 - ratio + 1 while writing up to p0 + n_spec. With that
// ring length a row is only overwritten by a position past any step that could still need it.
//
// A token's sequence comes from cu_seqlens (slot_mapping is a flat KV slot and does not identify
// the sequence); a linear scan suffices for a handful of sequences.
//
// Only the last `ring` tokens of a sequence's chunk write: positions p and p + ring share a row, so
// writing all of a prefill chunk would race. The last `ring` positions hit distinct rows, giving
// what a token-by-token loop would leave; older tokens' blocks are built from the chunk itself.
//
// Returns a RAD status rather than throwing, for the reason given in halo_gemm_fp8a8.hip.

#include "halo.h"
#include "halo_args.h"
#include "halo_emit.h"

#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

__global__ __launch_bounds__(kThreads) void qsa_tail_store_kernel(
    const uint16_t* __restrict__ k, const int32_t* __restrict__ pos,
    const int32_t* __restrict__ cu, const int32_t* __restrict__ sidx,
    uint16_t* __restrict__ state,
    int T, int n, int ring, int seqs, int slots, int k_ld, int s_slot, int s_row, int sidx_ld) {

    const int t = (int)blockIdx.x;
    if (t >= T) return;

    /* This token's sequence, by a linear scan in every thread (cheaper than LDS plus a barrier). */
    int s = -1;
    for (int i = 0; i < seqs; ++i)
        if (t >= cu[i] && t < cu[i + 1]) { s = i; break; }
    if (s < 0) return;

    /* Only the last `ring` tokens of this sequence's chunk -- see the header. */
    if (t < cu[s + 1] - ring) return;

    /* Column 0 is the committed slot; -1 marks a sequence with no state. A slot past the
     * state plane's would write past its allocation. */
    const int si = sidx[(size_t)s * sidx_ld];
    if (si < 0 || si >= slots) return;

    const int row = pos[t] % ring;
    const size_t src = (size_t)t * k_ld;
    const size_t dst = (size_t)si * s_slot + (size_t)row * s_row;
    for (int d = (int)threadIdx.x; d < n; d += kThreads) state[dst + d] = k[src + d];
}

}  // namespace

//   k      [T][n]                bf16, `k_ld` elements a row (0 = n)
//   pos    [T]                   i32, each token's position in its sequence
//   cu     [seqs + 1]            i32, cu_seqlens
//   sidx   [seqs][sidx_ld]       i32, the tail group's state row; column 0 is the committed slot
//   state  [slots][ring][n]      bf16, this layer's tail cache; written only, untargeted rows
//                                keep their contents
static int halo_qsa_tail_store_bf16_emit(long k, long pos, long cu, long sidx, long state, int T,
                                       int n, int ring, int seqs, int slots, int k_ld, int s_slot,
                                       int s_row, int sidx_ld, RadStream stream) {
    if (!k || !pos || !cu || !sidx || !state) return HALO_NO(RAD_E_INVAL);
    if (T <= 0 || n <= 0 || ring <= 0 || seqs <= 0 || slots <= 0) return HALO_NO(RAD_E_SHAPE);
    if (!k_ld) k_ld = n;
    if (!s_row) s_row = n;
    if (!sidx_ld) sidx_ld = 1;
    return HALO_EMIT(stream, qsa_tail_store_kernel, dim3((unsigned)T), dim3(kThreads), 0,
                    (const uint16_t*)k, (const int32_t*)pos, (const int32_t*)cu,
                    (const int32_t*)sidx, (uint16_t*)state, T, n, ring, seqs, slots, k_ld, s_slot,
                    s_row, sidx_ld);
}

// ============================================================== the plugin ABI entry point

enum { QT_K = 0, QT_POS = 1, QT_CU = 2, QT_STATE = 3, QT_SIDX = 4 };

extern "C" int halo_rad_qsa_tail_store(const RadArgs* a, RadStream stream) {
    const RadTensor* k  = halo_opt(a, QT_K);
    const RadTensor* po = halo_opt(a, QT_POS);
    const RadTensor* cu = halo_opt(a, QT_CU);
    const RadTensor* st = halo_opt(a, QT_STATE);
    const RadTensor* sx = halo_opt(a, QT_SIDX);
    if (!k || !po || !cu || !st || !sx) return HALO_NO(RAD_E_INVAL);
    if (k->rank != 2) return HALO_NO(RAD_E_SHAPE);
    /* The state arrives as the group declared it: [slots, heads, ring, n], heads 1
     * (core/engine_bringup.cpp). */
    if (st->rank != 4 || st->shape[1] != 1) return HALO_NO(RAD_E_SHAPE);
    if (st->stride[3] != 1) return HALO_NO(RAD_E_STRIDE);
    const long long n     = halo_geti_or(a, "n", 0);
    const long long ratio = halo_geti_or(a, "ratio", 0);
    const long long ring  = halo_geti_or(a, "ring", 0);
    if (n <= 0 || ratio <= 0 || ring < ratio) return HALO_NO(RAD_E_INVAL);
    const long long T = k->shape[0];
    if (T < 1 || k->shape[1] != n) return HALO_NO(RAD_E_SHAPE);
    if (st->shape[2] < ring || st->shape[3] != n) return HALO_NO(RAD_E_SHAPE);
    /* The sequence count comes from cu_seqlens, not the state's slot count (the declared
     * max_seqs). */
    const long long seqs = halo_numel(cu) - 1;
    if (seqs < 1 || halo_numel(po) < T) return HALO_NO(RAD_E_SHAPE);
    const long long sxp = sx->rank >= 2 ? halo_row_pitch(sx) : 1;
    if (sxp < 1 || halo_dim(sx, 0) < seqs) return HALO_NO(RAD_E_SHAPE);
    const long long kp = halo_row_pitch(k);
    if (kp < 0) return HALO_NO(RAD_E_STRIDE);
    if (k->dtype != RAD_BF16 || st->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    return halo_qsa_tail_store_bf16_emit(halo_addr(k), halo_addr(po), halo_addr(cu), halo_addr(sx),
                                        halo_addr(st), (int)T, (int)n, (int)ring, (int)seqs,
                                        (int)st->shape[0], (int)kp, (int)st->stride[0],
                                        (int)st->stride[2], (int)sxp, stream);
}

