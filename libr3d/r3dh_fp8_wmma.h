/* Ported from libhalo/halo_fp8_wmma.h, verbatim but for the halo_ -> r3dh_ rename, so libr3d builds and loads
 * on its own. libhalo's gated-delta-net kernels are libr4d's ported to the gfx11 wave matrix and
 * checked against libref on gfx1151 (rad-kbench); they compile for gfx11-generic unchanged. */
/* Derived from libr4d/r4d_fp8_wmma.h by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. Op schemas, row names and stored weight layouts are
 * shared with libr4d by design. Everything else in this file is libhalo's own
 * and diverges as it is measured on gfx1151. */
// Shared device code for the block-scaled FP8 family: the E4M3 decoders and the bf16 block-scale
// load. libr4d's version of this header also carries the gfx12 fp8 WMMA fragment map and its LDS
// staging; gfx1151 has no fp8 matrix instruction, so halo's fp8 GEMMs dequantise to bf16 in
// registers and run on r3dh_wmma.h's bf16 WMMA instead. That machinery lands with the fp8 GEMM
// units, in r3dh_wmma.h's terms; nothing here pretends otherwise.
#pragma once
#include "r3dh_common.h"
#include "r3dh_fp8_frag.h"

#include <hip/hip_runtime.h>
#include <cstdint>

namespace r3dh_fp8 {

/* The fp8 matrix machinery (libr4d's mma/mma8, fragment map and staging) is not
 * carried over: it names gfx12 instructions and a fragment layout this part does
 * not have. The fp8 GEMM units bring their own, built on r3dh_wmma.h. */

// ---- E4M3 -> f32 -------------------------------------------------------------------------------
//
// One instruction on gfx1201 (v_cvt_f32_fp8); the software fallback is ~11 VALU ops. They agree on
// all bytes except 0x7F and 0xFF, E4M3-fn's NaN encodings: hardware returns NaN, the fallback
// +-480. Both quantisers clamp to +-448 = 0x7E, so real data never holds them.
//
// No subnormal-free fast path: a checkpoint weight may hold subnormals, which such a path would
// decode as 2^-7 * 1.MMM instead of MMM * 2^-9.
__device__ __forceinline__ float e4m3(uint32_t b) {
#if __has_builtin(__builtin_amdgcn_cvt_f32_fp8)
    return __builtin_amdgcn_cvt_f32_fp8(b, 0);
#else
    const uint32_t sign = (b & 0x80u) << 24;
    const uint32_t ex   = (b >> 3) & 0x0Fu;
    const uint32_t mn   = b & 0x07u;
    const float nrm = __uint_as_float(((ex + 120u) << 23) | (mn << 20));
    const float sub = (float)mn * (1.0f / 512.0f);
    return __uint_as_float(__float_as_uint(ex ? nrm : sub) | sign);
#endif
}

// Two E4M3 bytes of a packed dword to two f32 in one instruction (v_cvt_pk_f32_fp8). `HI` selects
// bytes 2,3 over 0,1 and is a template parameter because the builtin's word select is a constant.
// Returns r3dh_common.h's `v2f`, the builtin's own return type.
template <bool HI>
__device__ __forceinline__ v2f e4m3_pk(uint32_t q) {
#if __has_builtin(__builtin_amdgcn_cvt_pk_f32_fp8)
    return __builtin_amdgcn_cvt_pk_f32_fp8(q, HI);
#else
    constexpr uint32_t s = HI ? 16u : 0u;
    v2f r;
    r.x = e4m3((q >> s) & 0xFFu);
    r.y = e4m3((q >> (s + 8u)) & 0xFFu);
    return r;
#endif
}

// ---- f32 -> E4M3 -------------------------------------------------------------------------------
//
// gfx1151 has no v_cvt_pkfp8_f32 (fp8-conversion-insts), so the encode is software. It is a
// transcription of abi/rad_plugin.h's rad_f32_to_fp8e4m3 -- round-to-nearest-even in both the
// normal and subnormal branches, saturating at 448, NaN to 0x7f -- because that function is the
// reference every fp8 store is checked against; a faster form must first be shown to agree with
// it on every input class (the host probe in the unit's bring-up compared 2^31 values).
__device__ __forceinline__ uint32_t f32_to_e4m3(float f) {
    const uint32_t u = __builtin_bit_cast(uint32_t, f);
    const uint32_t s = (u >> 24) & 0x80u;
    const uint32_t a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return s | 0x7fu;                    /* nan */
    if (a >= 0x43e00000u) return s | 0x7eu;                   /* >= 448: clamp to the largest */
    const int e = (int)(a >> 23) - 127;
    uint32_t m = a & 0x7fffffu;
    if (e < -10) return s;
    if (e < -6) {                                             /* subnormal fp8 */
        const int sh = -6 - e + 20;
        m |= 0x800000u;
        const uint32_t lo = m & ((1u << sh) - 1u);
        uint32_t q = m >> sh;
        const uint32_t half = 1u << (sh - 1);
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return s | q;
    }
    uint32_t q = m >> 20;
    const uint32_t lo = m & 0xfffffu;
    if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
    int ee = e + 7;
    if (q == 0x8u) { q = 0; ++ee; }
    if (ee > 15 || (ee == 15 && q >= 7u)) return s | 0x7eu;
    return s | ((uint32_t)ee << 3) | q;
}

/* __builtin_amdgcn_cvt_pk_fp8_f32's shape (two encodes into one half of `old`), so a call site
 * moves onto it by name alone. Callers still clamp to +-448 first, as they did for the hardware
 * instruction, whose out-of-range result is NaN rather than saturation; with the clamp the bytes
 * are the same ones. */
__device__ __forceinline__ int cvt_pk_fp8_f32(float lo_f, float hi_f, int old, bool word_sel) {
    const uint32_t pair = f32_to_e4m3(lo_f) | (f32_to_e4m3(hi_f) << 8);
    const uint32_t o = (uint32_t)old;
    return (int)(word_sel ? (o & 0x0000ffffu) | (pair << 16) : (o & 0xffff0000u) | pair);
}

// ---- block scale --------------------------------------------------------------------------------
//
// The checkpoint's BF16 `weight_scale_inv` (one per 128x128 tile, row-major) is read natively:
// decode is one shift and one bitcast. Converting to UE8M0 would round every tile's gain to a
// power of two and force requantising the E4M3 bytes, for no saving (the scale plane is 21.8 KiB
// against an 89 MiB weight at gate_proj's shape). Unlike UE8M0, bf16 needs no zero test:
// `0u << 16` is +0.0f.
__device__ __forceinline__ float bscale(uint16_t s) {
    return __uint_as_float((uint32_t)s << 16);
}
/* The same scale zero-extended in a full register (see u16_load_w in r3dh_moe.hip). The upper
 * sixteen bits must be zero. */
__device__ __forceinline__ float bscale(uint32_t s) {
    return __uint_as_float(s << 16);
}

}  // namespace r3dh_fp8
