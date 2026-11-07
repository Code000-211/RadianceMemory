/* Ported from libhalo/halo_gdn_conv_tap.h, verbatim but for the halo_ -> r3dh_ rename, so libr3d builds and loads
 * on its own. libhalo's gated-delta-net kernels are libr4d's ported to the gfx11 wave matrix and
 * checked against libref on gfx1151 (rad-kbench); they compile for gfx11-generic unchanged. */
/* Derived from libr4d/r4d_gdn_conv_tap.h by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. Op schemas, row names and stored weight layouts are
 * shared with libr4d by design. Everything else in this file is libhalo's own
 * and diverges as it is measured on gfx1151. */
// r3dh_gdn_conv_tap.h -- the arithmetic of one gated-delta-net convolution output, shared by the
// standalone conv kernels (r3dh_gdn_conv_w4_h128_bf16.hip) and the recurrent update that folds the
// decode convolution into its prologue (r3dh_gdn_recurrent_update_k128_v128_bf16_fp32state.hip).
// A single definition keeps the fold byte-identical to the standalone path: the reciprocal form
// below changes the last bit of some bf16 outputs.
#pragma once

#define R3DH_GDN_CONV_W 4        // conv width; the state holds R3DH_GDN_CONV_W - 1 taps of history

// The result is rounded to bf16 immediately, so an IEEE-exact divide buys nothing: v_rcp_f32 is
// 1 ULP (~1e-7 relative) against bf16's 2^-8 quantum. The exact form is ~10 dependent instructions
// (v_div_scale, v_rcp, Newton, v_div_fmas, v_div_fixup); this is two. The conv update runs four
// waves on five workgroups, so its time is its instruction count.
__device__ __forceinline__ float cp_silu(float z) {
  return z * __builtin_amdgcn_rcpf(1.0f + __expf(-z));
}
