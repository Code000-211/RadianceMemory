// r4d_kvx.h -- one (token, head) row of a KV cache layout decoded and re-encoded in another, a wave
// of 32 a row: the window-residency store's demote layout (RAD_KVQ_DEMOTE, abi/rad_builder.h).
// `kv_transcode` (the move path's requantise and expand) and the expanding `qsa_page` (the pager)
// both run this, so a block paged in and the same block promoted carry one set of bytes. libref's
// kvx_rows (libref/ref_attn.cpp) is the contract, held byte for byte.
//
// Layouts by RAD_KVQ_* id: 0 fp8 (E4M3 K then V, 2 * hd bytes), 3 k4h_v4, 4 k4h_v2h, 5 i8h; the rows
// are abi/rad_builder.h's. Lane l holds dims 4l + 128i (i < hd / 128) of K and of V, as the i8h
// store does, so a 32-dim group of k4h is the eight lanes 8k..8k+7 of one i and its reductions
// are the stores' xor butterflies over 4, 2, 1; the i8h head's is the one over 16..1.
//
// Decode is exact (an E4M3 code, or an int8, int4 or odd int2 level times an f16 scale, is an f32),
// so each encode below is the layout's store encode (r4d_model_bf16.hip's kv_store_fp8v4 /
// kv_store_i8h and kv_store_fp8's clamp) applied to the values in the stored units, operation for
// operation: no descale, rotation or V mean, which both layouts of a layer share.
#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace r4d_kvx {

enum { kFp8 = 0, kK4h = 3, kK4hV2h = 4, kI8h = 5, kMaxNi = 4 };

__host__ __device__ inline int64_t row_bytes(int lay, int64_t hd) {
    return lay == kFp8 ? 2 * hd : lay == kK4h ? 18 * hd / 16 : lay == kK4hV2h ? 14 * hd / 16
         : lay == kI8h ? 2 * hd + 4 : -1;
}

__device__ inline float f16at(const uint8_t* p) {
    return (float)__builtin_bit_cast(__fp16, *(const uint16_t*)p);
}

// A row's K and V, this lane's dims: x[w][i][e] is dim 4 * lane + 128 * i + e of K (w 0) or V (1).
__device__ inline void decode(int lay, const uint8_t* s, uint32_t hd, uint32_t lane,
                              float x[2][kMaxNi][4]) {
    const uint32_t ni = hd / 128u;
    for (uint32_t i = 0; i < ni; ++i) {
        const uint32_t d = lane * 4u + 128u * i;
        if (lay == kFp8) {
            for (int w = 0; w < 2; ++w) {
                const int wd = (int)*(const uint32_t*)(s + (size_t)w * hd + d);
                const auto lo = __builtin_amdgcn_cvt_pk_f32_fp8(wd, false);
                const auto hi = __builtin_amdgcn_cvt_pk_f32_fp8(wd, true);
                x[w][i][0] = lo[0]; x[w][i][1] = lo[1]; x[w][i][2] = hi[0]; x[w][i][3] = hi[1];
            }
        } else if (lay == kI8h) {
            for (int w = 0; w < 2; ++w) {
                const uint32_t c = *(const uint32_t*)(s + (size_t)w * hd + d);
                const float sc = f16at(s + 2 * (size_t)hd + 2 * w);
#pragma unroll
                for (int e = 0; e < 4; ++e) x[w][i][e] = (float)(int8_t)(uint8_t)(c >> (8 * e)) * sc;
            }
        } else {
            // K int4 for both k4h layouts; its scales at hd (k4h_v4) or 3hd/4 (k4h_v2h).
            const uint32_t ksoff = lay == kK4hV2h ? hd * 3u / 4u : hd;
            const uint32_t kc = *(const uint16_t*)(s + d / 2);
            const float ks = f16at(s + ksoff + (d / 32) * 2);
#pragma unroll
            for (int e = 0; e < 4; ++e) x[0][i][e] = (float)((int)(((kc >> (4 * e)) & 0xFu) ^ 8u) - 8) * ks;
            if (lay == kK4hV2h) {
                const uint32_t vc = s[hd / 2 + d / 4];
                const float vs = f16at(s + hd * 13u / 16u + (d / 32) * 2);
#pragma unroll
                for (int e = 0; e < 4; ++e) x[1][i][e] = (float)(2 * (int)((vc >> (2 * e)) & 3u) - 3) * vs;
            } else {
                const uint32_t vc = *(const uint16_t*)(s + hd / 2 + d / 2);
                const float vs = f16at(s + hd * 17u / 16u + (d / 32) * 2);
#pragma unroll
                for (int e = 0; e < 4; ++e) x[1][i][e] = (float)((int)(((vc >> (4 * e)) & 0xFu) ^ 8u) - 8) * vs;
            }
        }
    }
}

// k4h's int4 group (kv_store_fp8v4's K4 branch): absmax / 7 over the group's eight lanes.
__device__ inline void enc_int4(const float f[4], uint8_t* codes, uint8_t* scales, uint32_t d) {
    float amax = fmaxf(fmaxf(fabsf(f[0]), fabsf(f[1])), fmaxf(fabsf(f[2]), fabsf(f[3])));
    for (int m = 4; m >= 1; m >>= 1) amax = fmaxf(amax, __shfl_xor(amax, m, 32));
    const __fp16 s16 = (__fp16)fminf(amax / 7.0f, 8188.0f);
    const float sc = (float)s16;
    uint32_t q[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float r = sc > 0.0f ? rintf(f[e] / sc) : 0.0f;
        q[e] = (uint32_t)((int)fminf(fmaxf(r, -7.0f), 7.0f)) & 0xFu;
    }
    *(uint16_t*)(codes + d / 2) = (uint16_t)(q[0] | (q[1] << 4) | (q[2] << 8) | (q[3] << 12));
    if ((d & 31u) == 0) *(uint16_t*)(scales + (d / 32) * 2) = __builtin_bit_cast(uint16_t, s16);
}

// k4h_v2h's V group (kv_store_fp8v4's V2 branch): the variance-normalised odd int2 levels.
__device__ inline void enc_v2h(const float vf[4], uint8_t* codes, uint8_t* scales, uint32_t d) {
    float sum = fmaf(vf[3], vf[3], fmaf(vf[2], vf[2], fmaf(vf[1], vf[1], vf[0] * vf[0])));
    for (int m = 4; m >= 1; m >>= 1) sum = sum + __shfl_xor(sum, m, 32);
    sum = fminf(sum, 3.40282347e38f);
    const float rms = sqrtf(sum / 32.0f);
    float lv[4], c2 = 0.0f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float r = rms > 0.0f ? 2.0f * floorf(vf[e] / rms) + 1.0f : 1.0f;
        lv[e] = fminf(fmaxf(r, -3.0f), 3.0f);
        c2 += lv[e] * lv[e];
    }
    for (int m = 4; m >= 1; m >>= 1) c2 = c2 + __shfl_xor(c2, m, 32);
    const __fp16 s16 = (__fp16)fminf(sqrtf(sum / c2), 21824.0f);
    uint32_t b = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) b |= (uint32_t)(((int)lv[e] + 3) / 2) << (2 * e);
    codes[d / 4] = (uint8_t)b;
    if ((d & 31u) == 0) *(uint16_t*)(scales + (d / 32) * 2) = __builtin_bit_cast(uint16_t, s16);
}

__device__ inline void encode(int lay, uint8_t* o, uint32_t hd, uint32_t lane,
                              const float x[2][kMaxNi][4]) {
    const uint32_t ni = hd / 128u;
    if (lay == kI8h) {
        // kv_store_i8h: one f16 scale a head, absmax / 127 over every lane and both halves.
        for (int w = 0; w < 2; ++w) {
            float amax = __builtin_nanf("");
            for (uint32_t i = 0; i < ni; ++i)
#pragma unroll
                for (int e = 0; e < 4; ++e) amax = fmaxf(amax, fabsf(x[w][i][e]));
            for (int m = 16; m >= 1; m >>= 1) amax = fmaxf(amax, __shfl_xor(amax, m, 32));
            const __fp16 s16 = (__fp16)fminf(amax / 127.0f, 512.0f);
            const float sc = (float)s16;
            for (uint32_t i = 0; i < ni; ++i) {
                const uint32_t d = lane * 4u + 128u * i;
                uint32_t pk = 0;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float r = sc > 0.0f ? rintf(x[w][i][e] / sc) : 0.0f;
                    pk |= ((uint32_t)(int)fminf(fmaxf(r, -127.0f), 127.0f) & 0xFFu) << (8 * e);
                }
                *(uint32_t*)(o + (size_t)w * hd + d) = pk;
            }
            if (lane == 0) *(uint16_t*)(o + 2 * (size_t)hd + 2 * w) = __builtin_bit_cast(uint16_t, s16);
        }
        return;
    }
    for (uint32_t i = 0; i < ni; ++i) {
        const uint32_t d = lane * 4u + 128u * i;
        if (lay == kFp8) {
            // kv_store_fp8: clamped to +-448, then E4M3 to nearest even.
            for (int w = 0; w < 2; ++w) {
                const float* f = x[w][i];
                int pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(f[0], -448.0f), 448.0f),
                                                         fminf(fmaxf(f[1], -448.0f), 448.0f), 0, false);
                pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(f[2], -448.0f), 448.0f),
                                                     fminf(fmaxf(f[3], -448.0f), 448.0f), pk, true);
                *(uint32_t*)(o + (size_t)w * hd + d) = (uint32_t)pk;
            }
            continue;
        }
        enc_int4(x[0][i], o, o + (lay == kK4hV2h ? hd * 3u / 4u : hd), d);
        if (lay == kK4hV2h) enc_v2h(x[1][i], o + hd / 2, o + hd * 13u / 16u, d);
        else                enc_int4(x[1][i], o + hd / 2, o + hd * 17u / 16u, d);
    }
}

// The whole wave: row `s` in layout `from` becomes row `o` in layout `to`.
__device__ inline void row(int from, int to, const uint8_t* s, uint8_t* o, uint32_t hd,
                           uint32_t lane) {
    float x[2][kMaxNi][4];
    decode(from, s, hd, lane, x);
    encode(to, o, hd, lane, x);
}

}  // namespace r4d_kvx
