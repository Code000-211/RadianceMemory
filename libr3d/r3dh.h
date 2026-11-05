/* r3dh.h -- the entry points of the gated-delta-net kernels ported from libhalo (libhalo/halo.h's
 * gated-delta-net section, the halo_ -> r3dh_ rename): the chunk preamble, the chunked scan and the
 * recurrent update. libhalo's kernels are libr4d's ported to the gfx11 wave matrix and checked
 * against libref on gfx1151 (rad-kbench); they compile for gfx11-generic unchanged, so libr3d
 * carries them for the whole family. Plain C ABI on raw device pointers and a stream; an entry
 * point returns a negative code for a geometry it was not compiled for. */
#pragma once
#include <hip/hip_runtime.h>

extern "C" {

// ---- gated delta net -----------------------------------------------------------------------
// One chunked scan over N variable-length sequences: the WY recompute, the recurrent state scan
// and the output in one kernel (FLA's recompute_w_u_fwd + chunk_gated_delta_rule_fwd_h +
// chunk_fwd_o). Layouts, all contiguous and bf16 unless stated:
//   q,k [T, Hg, K]   v,o [T, H, V]   A [T, H, bt]   g,beta [T, H] fp32
//   h0,ht [N, H, V, K] fp32          cu [N+1] int32
// K, V and bt must be the compile-time 128/128/64 in the name; a mismatch returns -1. Prefill and
// chunked prefill run this; decode runs the recurrent update.
int r3dh_gdn_chunk_scan_k128_v128_c64_bf16(
        const void* q, const void* k, const void* v, const void* A,
        const void* g, const void* beta, const void* h0, void* o, void* ht,
        const void* cu, int N, int H, int Hg, int K, int V, int bt,
        float scale, int state_fp16, void* stream);
// The same scan against a slot-indexed state cache: `ssm_state_indices` [N] names each batch
// row's slot, and the two strides are the cache's, in elements. The call above is this one with a
// null index and packed [N, H, V, K] strides, which is correct only while slot == batch row.
int r3dh_gdn_chunk_scan_k128_v128_c64_bf16_sidx(
        const void* q, const void* k, const void* v, const void* A,
        const void* g, const void* beta, const void* h0, void* o, void* ht,
        const void* cu, const void* ssm_state_indices, long indices_stride,
        long state_slot_stride, long state_head_stride,
        int N, int H, int Hg, int K, int V, int bt,
        float scale, int state_fp16, void* stream);
void r3dh_gdn_dims(int* head_k, int* head_v, int* chunk);

// The chunk preamble: A = (I + strict_lower(diag(beta) K K^T e^{g_i-g_j}))^-1 per chunk, i.e.
// FLA's chunk_scaled_dot_kkt_fwd then solve_tril, with the fp32 gram kept out of HBM.
// k [T,Hg,K] bf16; beta, g [T,H] fp32 (g already summed along the chunk); A [T,H,64] bf16 out;
// cu [N+1] int32. Returns -1 for a geometry it was not compiled for.
int r3dh_gdn_kkt_solve_k128_c64_bf16(const void* k, const void* beta, const void* g, void* A,
                                    const void* cu, int N, int T, int H, int Hg, int K, int bt,
                                    void* stream);

// The recurrent delta-rule update decode runs: gating, the qk l2 norm, the state update and the
// output, against the paged state cache. The state is fp32 (per the model config) and one is
// written per candidate token, which dominates the cost. Replaces
// fused_sigmoid_gating_delta_rule_update.
int r3dh_gdn_recurrent_update_k128_v128_bf16_fp32state(
        const void* q, const void* k, const void* v, const void* a, const void* b,
        long ab_stride, int ab_is_bf16, const void* A_log, const void* dt_bias, void* state,
        long state_slot_stride, long state_head_stride, void* o, void* oq, void* os,
        long oq_pitch, long os_ld, const void* cu,
        const void* ssm_state_indices, long indices_stride, const void* num_accepted,
        const void* z_gate, long z_pitch, const void* norm_weight, float norm_eps, int norm_act,
        int N, int H, int Hg, int K, int V, float scale, float softplus_thr, int state_fp16,
        void* stream);

}  // extern "C"
