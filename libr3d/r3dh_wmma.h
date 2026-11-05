/* Ported from libhalo/halo_wmma.h, verbatim but for the halo_ -> r3dh_ rename, so libr3d builds and loads
 * on its own. libhalo's gated-delta-net kernels are libr4d's ported to the gfx11 wave matrix and
 * checked against libref on gfx1151 (rad-kbench); they compile for gfx11-generic unchanged. */
/* Carried over from the strix-halo-vllm project's libhalo, where this layer was
 * established and measured on the part; see libhalo/README.md. */
// Wave32 WMMA fragment algebra for gfx1151 / RDNA 3.5.
//
// This is the gfx11 counterpart of libr4d's gfx12 fragment layer, and it exists
// because the two architectures do NOT agree on how a wave32 WMMA fragment is
// spread over the wave:
//
//                       A / B per lane        the k a lane's element e holds
//   gfx1201 (RDNA4)     8 bf16  (v[0:3])      k = 8*(e>>2) + 4*(lane>>4) + (e&3)
//   gfx1151 (RDNA3.5)   16 bf16 (v[0:7])      k = e
//
// RDNA 3.5 replicates each operand row across both halves of the wave, so a lane
// holds the WHOLE 16-wide k run; RDNA4 splits it. Every fragment gather ported
// from libr4d therefore loses its half-wave term -- the port is a simplification
// at the gather sites, and a change of row mapping at the result.
//
// The layout below was established on the part itself by bench/wmma_layout_probe.hip,
// which tests candidate mappings against a host matmul and finds exactly one with
// zero error (the others are off by ~350 on the same data). Do not "fix" any of it
// from a vendor doc -- rerun the probe.
//
//   A: lane l holds row (l & 15) of A;  element e is k = e   (lanes 16-31 replicate)
//   B: lane l holds col (l & 15) of B;  element e is k = e
//   D: lane l, element e  ->  D[2*e + (l>>4)][l & 15]
//
// Note what the D mapping preserves: the COLUMN is still (lane & 15). libr4d's
// attention kernels are built around a transposed score matrix S^T = K.Q^T so that
// a fragment gives each lane exactly one query row and the softmax is lane-private.
// That property survives here unchanged, which is why those kernels are portable
// at all.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "r3dh_common.h"

/* Byte order of stored fp8 weights along k, inside each group of four, and which
 * e4m3 -> bf16 unpack deq16_e4m3 below compiles. In the vLLM libhalo these lived
 * in its r3dh_common.h; here the stored layout is radiance's own (r3dh_layout.cpp
 * chooses it at load), so the swizzled order is available when a relayout asks
 * for it. Defaults: natural order, perm-input unpack. */
#ifndef R3DH_FP8_SWIZ
#define R3DH_FP8_SWIZ 0
#endif
#ifndef R3DH_FP8_DEQ
#define R3DH_FP8_DEQ 1
#endif

typedef __bf16   v16bf __attribute__((ext_vector_type(16)));
typedef __fp16   v16hf __attribute__((ext_vector_type(16)));  // what the f16 WMMA builtin takes
typedef float    v8f   __attribute__((ext_vector_type(8)));
typedef float    v2f   __attribute__((ext_vector_type(2)));

namespace halo {

// ---- where a lane's data lives ------------------------------------------------
__device__ __forceinline__ int a_row(int lane) { return lane & 15; }
__device__ __forceinline__ int b_col(int lane) { return lane & 15; }
__device__ __forceinline__ int d_col(int lane) { return lane & 15; }
__device__ __forceinline__ int d_row(int lane, int e) { return 2 * e + (lane >> 4); }
// The half a lane sits in. On gfx12 this selects which k it loads; here it selects
// only which D rows it receives, and the gather does not use it.
__device__ __forceinline__ int half(int lane) { return lane >> 4; }

// ---- the instruction ----------------------------------------------------------
__device__ __forceinline__ v8f mma(v16bf a, v16bf b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}
__device__ __forceinline__ v8f mma(v16hf a, v16hf b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// ---- building a fragment ------------------------------------------------------
// From eight dwords, i.e. a lane's full 16-wide k run already packed as 16-bit
// pairs. This is the gfx11 replacement for libr4d's DT16::mk(uint2 lo, uint2 hi):
// twice the payload, and no half-wave selection.
template <typename FRAG>
__device__ __forceinline__ FRAG mk(const uint32_t w[8]) {
    union { uint32_t u[8]; FRAG f; } q;
#pragma unroll
    for (int i = 0; i < 8; ++i) q.u[i] = w[i];
    return q.f;
}

// From a 32-byte-aligned pointer to 16 contiguous 16-bit values -- the common case
// once the half-wave split is gone, and two dwordx4 loads.
template <typename FRAG>
__device__ __forceinline__ FRAG mk_ldg(const void* p) {
    union { uint4 v[2]; FRAG f; } q;
    q.v[0] = *(const uint4*)p;
    q.v[1] = *((const uint4*)p + 1);
    return q.f;
}

// f32 -> bf16, round-to-nearest-even. gfx1151 has no v_cvt_pk_bf16_f32 either, so
// this stays the software form libr4d uses.
__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ uint32_t pk_bf16(float x, float y) {
    return (uint32_t)f32_to_bf16(x) | ((uint32_t)f32_to_bf16(y) << 16);
}
// f16 packs in ONE instruction, which is why libr4d prefers it on the softmax path.
__device__ __forceinline__ uint32_t pk_f16(float x, float y) {
    typedef __fp16 v2hf __attribute__((ext_vector_type(2)));   // what cvt_pkrtz returns
    union { v2hf h; uint32_t u; } q;
    q.h = __builtin_amdgcn_cvt_pkrtz(x, y);
    return q.u;
}

// Exchange a value with the lane 16 away -- the two halves a D fragment splits its
// rows across. The one cross-lane step the transposed-score formulation needs.
__device__ __forceinline__ float swap16(float x) { return __shfl_xor(x, 16, 32); }
__device__ __forceinline__ uint32_t swap16_u(uint32_t x) {
    return __builtin_bit_cast(uint32_t, swap16(__builtin_bit_cast(float, x)));
}

// ---- e4m3 -> bf16, one B fragment (16 bytes) ----------------------------------
// The single copy. It used to be inlined in r3dh_moe_fp8_group and again in
// r3dh_gemm_fp8_nt, and the two had already drifted; they are the same sixteen
// bytes becoming the same fragment, so they are the same function.
//
// A dword holds four e4m3 and must become two dwords of two bf16. Bytes 0 and 2
// sit at bit 0 and bit 16, so ONE left shift moves both their payloads and one
// more moves both their signs; bytes 1 and 3 sit at bit 8 and 24 and one right
// shift serves both of theirs. Four instructions place all four values -- but
// that pairs k with k+2, and WMMA wants k with k+1.
//
// v_perm_b32 fixes the pairing on the INPUT for one instruction: reorder the
// dword to (b0, b2, b1, b3) and the natural output of the two shifts IS k
// order. Eight instructions per dword against the twelve the mask-and-restitch
// version needs (ten bit ops plus two v_mov_b16 the register allocator adds),
// and -- unlike doing it in storage -- it needs no change to the stored
// weights, which is what makes it shippable: the layout belongs to vLLM's
// Triton fused_experts, which reads the same tensors on every prefill.
//
// The exponent is NOT biased. Every value comes out low by exactly 2^-120,
// which the caller folds into the block scale as R3DH_FP8_FIX; see docs/31.
__device__ __forceinline__ v16bf deq16_e4m3(const uint32_t* __restrict__ w4) {
    union { uint32_t u[8]; v16bf f; } q;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        const uint32_t w = w4[d];
#if !R3DH_FP8_DEQ
        // The control: mask each byte in place, then restitch the halves into k
        // order. Ten bit ops, and the allocator adds two v_mov_b16 on top.
        const uint32_t lo = ((w & 0x007F007Fu) << 4) | ((w & 0x00800080u) << 8);
        const uint32_t hi = ((w & 0x7F007F00u) >> 4) | (w & 0x80008000u);
        q.u[d * 2 + 0] = (lo & 0x0000FFFFu) | (hi << 16);
        q.u[d * 2 + 1] = (lo >> 16) | (hi & 0xFFFF0000u);
#else
#if R3DH_FP8_SWIZ
        const uint32_t x = w;      // stored (k0, k2, k1, k3): no perm needed
#else
        const uint32_t x = __builtin_amdgcn_perm(w, w, 0x03010200u);
#endif
        q.u[d * 2 + 0] = ((x << 8) & 0x80008000u) | ((x << 4) & 0x07F007F0u);
        q.u[d * 2 + 1] = (x & 0x80008000u) | ((x >> 4) & 0x07F007F0u);
#endif
    }
    return q.f;
}

}  // namespace halo
