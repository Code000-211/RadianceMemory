// 16-bit operand traits. The QK and PV matmuls take a 16-bit A/B pair, and both sources are exact
// in either format: an fp8 e4m3 K/V has 3 mantissa bits (bf16 has 7, f16 10), and a bf16 query's
// 7 bits are exact in f16 for |q| in [6.1e-5, 65504].
// The costs differ. gfx1201 has no bf16 convert, so each f32->bf16 is ~6 VALU of software RTNE,
// while v_cvt_pkrtz_f16_f32 converts two f32 in one; on the softmax path that is 24 VALU per
// 8-element P fragment against 4. f16 also has 3 more mantissa bits; it gives up exponent range,
// which GROW below accounts for.
#pragma once
#include "r4d_common.h"

typedef __bf16   v8bf __attribute__((ext_vector_type(8)));
typedef __fp16   v8hf __attribute__((ext_vector_type(8)));  // what the f16 WMMA builtin takes
typedef __fp16   v2hf __attribute__((ext_vector_type(2)));  // what cvt_pkrtz returns

// GROW = how many octaves the running row max may exceed the reference max before the accumulator
// must be rescaled. p = 2^(s - m_ref) must stay inside the format, so GROW is log2(format max) with
// margin. Eager online softmax pays 128 multiplies every m-tile; with a reference max they are paid
// only when the max moves by more than GROW, which at long context is rare.
template<int F16> struct DT16;

template<> struct DT16<0> {                       // bf16: exponent range of f32
    typedef v8bf frag;
    static constexpr float SHIFT = 0.0f;
    static constexpr float GROW  = 60.0f;         // 2^60 * |v| * n stays far under f32 max
    __device__ __forceinline__ static frag mk(uint2 lo, uint2 hi) {
        union { uint32_t w[4]; frag f; } u;       // union, never a pointer cast
        u.w[0] = lo.x; u.w[1] = lo.y; u.w[2] = hi.x; u.w[3] = hi.y; return u.f;
    }
    __device__ __forceinline__ static v8f wmma(frag a, frag b, v8f c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    }
    __device__ __forceinline__ static uint32_t pk(float x, float y) {
        return (uint32_t)f32_to_bf16(x) | ((uint32_t)f32_to_bf16(y) << 16);
    }
    __device__ __forceinline__ static uint32_t pkv(v2f x) {   // what cvt_pk_f32_fp8 returns
        union { v2f f; uint32_t u[2]; } q; q.f = x;
        return __builtin_amdgcn_perm(q.u[1], q.u[0], 0x07060302u);
    }
    __device__ __forceinline__ static uint32_t from_bf16w(uint32_t w) { return w; }  // identity
};

template<> struct DT16<1> {                       // f16: 3 more mantissa bits, 1 convert instead of 6
    typedef v8hf frag;
    static constexpr float SHIFT = 0.0f;
    static constexpr float GROW  = 14.0f;         // p <= 2^14 = 16384 < 65504
    __device__ __forceinline__ static frag mk(uint2 lo, uint2 hi) {
        union { uint32_t w[4]; frag f; } u;
        u.w[0] = lo.x; u.w[1] = lo.y; u.w[2] = hi.x; u.w[3] = hi.y; return u.f;
    }
    __device__ __forceinline__ static v8f wmma(frag a, frag b, v8f c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    }
    __device__ __forceinline__ static uint32_t pk(float x, float y) {
        union { v2hf h; uint32_t u; } q; q.h = __builtin_amdgcn_cvt_pkrtz(x, y); return q.u;
    }
    __device__ __forceinline__ static uint32_t pkv(v2f x) {
        union { v2f f; float e[2]; } s; s.f = x;
        union { v2hf h; uint32_t u; } q; q.h = __builtin_amdgcn_cvt_pkrtz(s.e[0], s.e[1]); return q.u;
    }
    // two bf16 in a dword -> two f16 in a dword. bf16->f32 is a shift, so this is 3 VALU.
    __device__ __forceinline__ static uint32_t from_bf16w(uint32_t w) {
        union { v2hf h; uint32_t u; } q;
        q.h = __builtin_amdgcn_cvt_pkrtz(__builtin_bit_cast(float, w << 16),
                                         __builtin_bit_cast(float, w & 0xffff0000u));
        return q.u;
    }
};

// Widen 8 fp8 (2 dwords) to 8 16-bit values (4 dwords) in element order.
// Eight K dims of the k6_v4 cache as four f16 dwords, dims (2i, 2i+1) in dword i. `lo` holds
// their low nibbles (dim 2i in bits 8i..8i+3, 2i+1 in 8i+4..8i+7) and `hb` their 16-dim chunk's
// high-bit word (byte j: bits 0-1 dim 2j, 2-3 dim 2j+1, 4-5 dim 8+2j, 6-7 dim 9+2j); `half` says
// which eight of the sixteen these are. One mask and shift put the high bits beside the low
// nibbles, a byte a dim. Codes are 6-bit two's complement: ^ 32 makes them offset binary u,
// 0x6400 | u is the f16 1024 + u, and (x - 1056) * s is the value times the group scale `s2` (the
// f16 in both halves), rounded once.
__device__ __forceinline__ void k6x8_to_f16x4w(uint32_t lo, uint32_t hb, int half, uint32_t s2,
                                               uint32_t* w4) {
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    const uint32_t ev = ((lo & 0x0F0F0F0Fu) | ((half ? hb : hb << 4) & 0x30303030u)) ^ 0x20202020u;
    const uint32_t od = (((lo >> 4) & 0x0F0F0F0Fu) | ((half ? hb >> 2 : hb << 2) & 0x30303030u))
                      ^ 0x20202020u;
    const h2 s = __builtin_bit_cast(h2, s2);
    const h2 bias = { (_Float16)1056.0f, (_Float16)1056.0f };
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        // Byte b of the even dims low, of the odd dims high; perm takes bytes 0-3 from its second
        // operand, 4-7 from its first, and 0x0c is a zero byte.
        const uint32_t sel = (uint32_t)b | 0x0c00u | ((uint32_t)(4 + b) << 16) | 0x0c000000u;
        const uint32_t x = __builtin_amdgcn_perm(od, ev, sel) | 0x64006400u;
        w4[b] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2, x) - bias) * s);
    }
}

// Eight K dims of the k5_v4 cache as four f16 dwords, dims (2i, 2i+1) in dword i. `lo` holds their
// low nibbles as k6x8_to_f16x4w takes them and `hb` their group's high-bit word shifted right by
// 4 * (chunk % 2), so byte j's bit 2 * half holds dim 2j's fifth bit and the next bit dim 2j+1's
// (abi/rad_builder.h's RAD_KVQ_K5V4). One shift and mask a parity put each bit at bit 4 of its
// byte. Codes are 5-bit two's complement: ^ 16 makes them offset binary u, and (1024 + u - 1040)
// * s is the value times the group scale `s2` (the f16 in both halves), rounded once.
__device__ __forceinline__ void k5x8_to_f16x4w(uint32_t lo, uint32_t hb, int half, uint32_t s2,
                                               uint32_t* w4) {
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    const uint32_t sh = half ? 2u : 0u;
    const uint32_t ev = ((lo & 0x0F0F0F0Fu) | (((hb >> sh) & 0x01010101u) << 4)) ^ 0x10101010u;
    const uint32_t od = (((lo >> 4) & 0x0F0F0F0Fu) | (((hb >> (sh + 1)) & 0x01010101u) << 4))
                      ^ 0x10101010u;
    const h2 s = __builtin_bit_cast(h2, s2);
    const h2 bias = { (_Float16)1040.0f, (_Float16)1040.0f };
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const uint32_t sel = (uint32_t)b | 0x0c00u | ((uint32_t)(4 + b) << 16) | 0x0c000000u;
        const uint32_t x = __builtin_amdgcn_perm(od, ev, sel) | 0x64006400u;
        w4[b] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2, x) - bias) * s);
    }
}

// Eight K dims of the k4h_v4 cache as four f16 dwords, dims (2i, 2i+1) in dword i, from `lo`, their
// int4 codes as k6x8_to_f16x4w takes the low nibbles, and the group scale `s2`. The codes are V's
// (two's complement, ^ 8 is offset binary u), so this is V's widening on K: (1024 + u - 1032) * s,
// rounded once.
__device__ __forceinline__ void k4x8_to_f16x4w(uint32_t lo, uint32_t s2, uint32_t* w4) {
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    const uint32_t ev = (lo & 0x0F0F0F0Fu) ^ 0x08080808u;
    const uint32_t od = ((lo >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
    const h2 s = __builtin_bit_cast(h2, s2);
    const h2 bias = { (_Float16)1032.0f, (_Float16)1032.0f };
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const uint32_t sel = (uint32_t)b | 0x0c00u | ((uint32_t)(4 + b) << 16) | 0x0c000000u;
        const uint32_t x = __builtin_amdgcn_perm(od, ev, sel) | 0x64006400u;
        w4[b] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2, x) - bias) * s);
    }
}

// V's eight dims of one key from its code word `w`, as the two byte words the V widening of the
// h256 attention kernels interleaves: `n` dims 0, 2, 4, 6 and `m` dims 1, 3, 5, 7, a byte each, so
// that the f16 0x6400 | byte (1024 + byte) less VBias is the code's level, which times the group
// scale is the value. int4 (fp8_v4, k6_v4, k4h_v4): eight two's complement nibbles, ^ 8 makes
// them offset binary, VBias 1032. int2 (V2, k4h_v2h): eight codes in w's low 16 bits, dim d at bits
// 2d; a code u goes in as 2u against VBias 1027, so the level is the odd 2u - 3 the store wrote
// (libref/ref_attn.cpp). Both exact in f16.
template<int V2>
__device__ __forceinline__ void v_codes_nm(uint32_t w, uint32_t& n, uint32_t& m) {
    if constexpr (V2) {
        // Nibble j (dims 2j and 2j + 1) into byte j, then each byte's two codes apart, doubled.
        const uint32_t s = (w & 0xFu) | ((w & 0xF0u) << 4) | ((w & 0xF00u) << 8) | ((w & 0xF000u) << 12);
        n = (s & 0x03030303u) << 1;
        m = (s >> 1) & 0x06060606u;
    } else {
        n = (w & 0x0F0F0F0Fu) ^ 0x08080808u;
        m = ((w >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
    }
}
template<int V2> struct VBias { static constexpr float v = V2 ? 1027.0f : 1032.0f; };

// The i8h cache's eight dims of one key, int8 codes in `lo` (dims 0-3, a byte each) and `hi` (dims
// 4-7), as v_codes_nm's two words: `n` dims 0, 2, 4, 6 and `m` dims 1, 3, 5, 7, each code ^ 0x80,
// which is offset binary u = c + 128 (1..255). 0x6400 | u is the f16 1024 + u, and less 1152 it
// is c, exact in f16. perm takes bytes 0-3 from its second operand and 4-7 from its first.
__device__ __forceinline__ void v8_codes_nm(uint32_t lo, uint32_t hi, uint32_t& n, uint32_t& m) {
    n = __builtin_amdgcn_perm(hi, lo, 0x06040200u) ^ 0x80808080u;
    m = __builtin_amdgcn_perm(hi, lo, 0x07050301u) ^ 0x80808080u;
}
constexpr float kI8Bias = 1152.0f;

// Eight K dims of the i8h cache as four f16 dwords, dims (2i, 2i+1) in dword i, from their codes
// `lo` / `hi` (v8_codes_nm) and the token's scale `s2` (the f16 in both halves): (1024 + u - 1152)
// * s, rounded once, as k4x8_to_f16x4w widens int4.
__device__ __forceinline__ void k8x8_to_f16x4w(uint32_t lo, uint32_t hi, uint32_t s2, uint32_t* w4) {
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    uint32_t ev, od;
    v8_codes_nm(lo, hi, ev, od);
    const h2 s = __builtin_bit_cast(h2, s2);
    const h2 bias = { (_Float16)kI8Bias, (_Float16)kI8Bias };
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const uint32_t sel = (uint32_t)b | 0x0c00u | ((uint32_t)(4 + b) << 16) | 0x0c000000u;
        const uint32_t x = __builtin_amdgcn_perm(od, ev, sel) | 0x64006400u;
        w4[b] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2, x) - bias) * s);
    }
}

template<class D>
__device__ __forceinline__ void fp8x8_to_16x4w(uint2 raw, uint32_t* w4) {
    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int wv = (q == 0) ? (int)raw.x : (int)raw.y;
        w4[q * 2 + 0] = D::pkv(__builtin_amdgcn_cvt_pk_f32_fp8(wv, false));
        w4[q * 2 + 1] = D::pkv(__builtin_amdgcn_cvt_pk_f32_fp8(wv, true));
    }
}

__device__ __forceinline__ int wave_any(bool c) {
    return (int)__builtin_amdgcn_ballot_w32(c);
}
