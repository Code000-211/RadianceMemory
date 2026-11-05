/* Ported from libhalo/halo_q8.h, verbatim but for the halo_ -> r3dh_ rename, so libr3d builds and loads
 * on its own. libhalo's gated-delta-net kernels are libr4d's ported to the gfx11 wave matrix and
 * checked against libref on gfx1151 (rad-kbench); they compile for gfx11-generic unchanged. */
/* Derived from libr4d/r4d_q8.h by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. Op schemas, row names and stored weight layouts are
 * shared with libr4d by design. Everything else in this file is libhalo's own
 * and diverges as it is measured on gfx1151. */
// r3dh_q8.h -- the 8-bit activation codes the trunk GEMMs read, written by every quantising producer:
// E4M3 for the fp8a8 GEMMs, int8 for their int8 twins (gemm_i8a8_*). One scale per (row, 128
// columns); only the largest code and the conversion differ.
//
// E4M3: scale `amax * (1 / 448)`, clamp to +-448, two packed conversions. The clamp is required:
// v_cvt_pk_fp8_f32 returns NaN for out-of-range input instead of saturating, and
// `amax * (1 / scale)` can land an ulp above 448.
//
// Int8: as quant_act_i8g (r3dh_quant_act_fp8.hip), scale `amax * (1 / 127)` and round-to-nearest-
// even into -127..127, so a fused producer writes the same codes as the standalone quantiser.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "r3dh_fp8_wmma.h"   // r3dh_fp8::cvt_pk_fp8_f32 (no fp8-conversion-insts on gfx1151)

template <bool I8>
__device__ __forceinline__ constexpr float q8_top() {
    return I8 ? 127.0f : 448.0f;
}

// Four values times `inv` (the reciprocal of the scale), as four codes in one dword, value 0 in
// the low byte.
template <bool I8>
__device__ __forceinline__ uint32_t q8_pack4(float a, float b, float c, float d, float inv) {
    if constexpr (I8) {
        const int ia = (int)rintf(fminf(fmaxf(a * inv, -127.0f), 127.0f));
        const int ib = (int)rintf(fminf(fmaxf(b * inv, -127.0f), 127.0f));
        const int ic = (int)rintf(fminf(fmaxf(c * inv, -127.0f), 127.0f));
        const int id = (int)rintf(fminf(fmaxf(d * inv, -127.0f), 127.0f));
        return ((uint32_t)ia & 0xFFu) | (((uint32_t)ib & 0xFFu) << 8) |
               (((uint32_t)ic & 0xFFu) << 16) | (((uint32_t)id & 0xFFu) << 24);
    } else {
        int pk = 0;
        pk = r3dh_fp8::cvt_pk_fp8_f32(fminf(fmaxf(a * inv, -448.0f), 448.0f),
                                       fminf(fmaxf(b * inv, -448.0f), 448.0f), pk, false);
        pk = r3dh_fp8::cvt_pk_fp8_f32(fminf(fmaxf(c * inv, -448.0f), 448.0f),
                                       fminf(fmaxf(d * inv, -448.0f), 448.0f), pk, true);
        return (uint32_t)pk;
    }
}
