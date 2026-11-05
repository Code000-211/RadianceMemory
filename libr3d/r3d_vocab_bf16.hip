/* r3d_vocab_bf16.hip -- the lm_head projection: `logits_gemm`, and the same tiling under `gemm_nt`
 * at prefill sizes.
 *
 *   y[M,N] = x[M,K] @ w[N,K]^T, f32 accumulate, k ascending.
 *
 * `logits_gemm` spells N and K as n_vocab and n_embd, and its y is f32 because the schema says so
 * (`ref_logits_gemm_shape`) and every sampler stage reads f32 logits. `gemm_nt` is the same
 * arithmetic with a bf16 y, narrowed once by the store. Both entries run the same two kernels;
 * `YF32` is the only difference.
 *
 * libref's `ref_logits_gemm` is a CPU GEMM (OpenMP), about 1.1 TFLOP at the M=2048,
 * n_vocab=130560, n_embd=2048 band. The head is read in either of its stored forms, with nothing
 * repacked: plain bf16 (no layout hook), or E4M3 codes with a bf16 scale per 128x128 block as two
 * planes of one `rad.planes` container, answered by r3d_layout_logits_head below.
 *
 * ---- the prefill arm (M > 16) -------------------------------------------------------------------
 *
 * A block is TM x TN = 128 x 128 outputs, 256 threads (eight waves), grid (N/TN) x (M/TM). Lane
 * `tid` owns rows m0 + (tid/16)*8 .. +7 by columns n0 + (tid%16)*8 .. +7: 64 f32 accumulators, one
 * 8x8 square, stored as 32-byte runs of f32 (16-byte runs of bf16).
 *
 * k is walked in KC = 32 chunks. Both tiles are row-major f16 (a row is 32 k-values, 64 bytes),
 * filled by one `uint4` load of 8 contiguous k a thread and widened by `bf16x2_to_f16x2`, which is
 * exact for every value f16 can hold and gives inf for the rest. The compute reads f16x2 pairs:
 * 16 LDS reads against 64 `fdot2` per two k.
 *
 * Tile rows are XOR-swizzled in eight-element units (`srow`, r3d_tiling.h). The staging store is a
 * `uint4`, so chunks must stay eight-element aligned; unswizzled, the 64-byte row stride puts the
 * eight rows a lane reads on one bank (32-way). Swizzling by `row & 7` brings that to two-way (a
 * lane's rows sit at stride eight, so a period-8 swizzle repeats within them): 512 LDS cycles a
 * block-chunk against 2048 of dots.
 *
 * ---- the decode arm (M <= 16) -------------------------------------------------------------------
 *
 * The prefill tile reads the head M/TM times, so at M=1 it would read 127/128 of it for nothing.
 * At M <= 16 the block uses r3d_gemm_bf16_nt_m16.hip's decode shape: a wave is MT activation rows
 * by 32/MT weight rows, lane l owns (m = l % MT, n = n0 + l / MT), one accumulator from k=0 to K
 * and no wave reduction, so each sum is the reference's ascending chain in the pair-dot's grouping.
 * The activation is staged in 128-k chunks in LDS (MT x 128 f16, 4 KiB at MT=16, same swizzle);
 * the weight goes from global to registers in 16-byte loads, eight values at a time.
 *
 * `nt=1` requests non-temporal head loads. On gfx1102 it measures within 1% of the default at M=1,
 * so it is off by default.
 *
 * The quantised head is half the bytes of the bf16 one (255 MiB of codes plus 32 KiB of scales
 * against 510 MiB at the band), which is its case for decode.
 *
 * ---- the accumulation ---------------------------------------------------------------------------
 *
 * libref's law per output is `acc += a*b` over ascending k from zero, no bias. Both arms keep one
 * accumulator per output, k ascending, no reassociation across lanes. The pair-dot folds two
 * products per rounding (`acc + a0*b0 + a1*b1`) where the reference rounds each, so results are
 * close but not bit-equal: rel_l2 about 1.8e-6 at K = 2048 with f32 output (the widening to f16
 * contributes nothing), far below the 4.9e-4 quantisation of the head's bf16 values.
 *
 * This is not integer arithmetic, so the v_fdot2_f32_f16 mantissa limit that bounds
 * r3d_gemm_nt_q_w2a8.hip does not apply.
 *
 * ---- the quantised head -------------------------------------------------------------------------
 *
 * The band's `dtype=bf16` is the activation's dtype; the head may be stored quantised.
 *
 *   r3d_layout_logits_head  (operand 1, as `ref_layout_head` is `layout_scaled(operand, 1, -1, ...)`)
 *       Tag `rad.planes`; codes E4M3 (libref's `scaled_rows` also admits int8) with a bf16, f16 or
 *       f32 scale on a {128, 128} block (`R3D_FP8_BLOCK`); ends in `rad_planes_layout` and moves no
 *       byte. `ref_layout_head` is static inside libref, so it and `layout_scaled`,
 *       `enc_scaled_rows` and `float_scale_dt` are transcribed here with their checks in order.
 *       `rad_enc_is` compares `scheme`, and `transform` is the optional pre-transform: the encoding
 *       is affine with an empty transform, so a fixture that writes "affine" into `transform` makes
 *       the hook decline every head.
 *
 *   The law is `ScaledRows::at(r, c) = codes.at(r, c) * scale.cover(r, c)` in f32, then the plain
 *   ascending-k GEMM. Decoding a code to f16 is exact (E4M3 has 4 exponent and 3 mantissa bits, f16
 *   5 and 10; int8 fits whole). Here the scale is folded into the staged f16 weight instead (see
 *   scale_pair): once per staged 8-column group in the tiled arm and once per 128-k chunk in the
 *   decode arm, both inside one 128-column block. That costs one f16 rounding per weight, rel_l2
 *   about 2.8e-4 at M=1, K=2048 against the reference, 250x finer than the E4M3 quantum (2^-4).
 *   An f32 scale plane adds one rounding (the scale to f16); a bf16 scale is f16-exact. y stays f32.
 *
 *   Placement dominates the kernel: a head on the mapped tier (hipHostMalloc, read over PCIe) runs
 *   about 50x slower than in VRAM, and an unprefetched managed head faults page by page because
 *   this ROCm does not migrate managed pages for read-mostly device access.
 *
 *   The container header picks the form: the shim calls `rad_planes_header(w)` and a plain bf16
 *   operand takes the bf16 path. The shim reads the header on the host (as `scaled_rows()` does)
 *   and the kernel reads the same bytes on the device, so the operand must be managed memory
 *   (prefetched) or hipHostMalloc; plain malloc faults on the kernel's first read.
 *
 *   gfx1102 has no v_cvt_pk_f16_fp8 (clang lists the builtin but it expands to software), so the
 *   E4M3 decode uses r3d_fp8_dequant.h's 256-entry f16 table in LDS (512 bytes, eight reads per
 *   eight codes). The table is built from the codec, so results are bit-identical; it pays most at
 *   M=8 (about 2.7x per token), where decoding is the binding resource. At M=1 the quantised arm
 *   runs far below memory bandwidth; the likely limit is its 8-byte code load (the bf16 arm
 *   loads 16 bytes a lane).
 *
 * ---- refusals -----------------------------------------------------------------------------------
 *
 * The grid is vocab-wide at every M (1020 blocks at the band for the prefill arm, 8160 for
 * the decode arm at MT=16), so a split would buy nothing: sk = 1 fixed. kb: KC = 32 only, since
 * another pitch is a different swizzle. Also refused: K not a multiple of KC, an x pitch not a
 * multiple of 8, a y
 * pitch not a multiple of 4 (f32) or 8 (bf16), a non-contiguous weight. The head branch, in order:
 *   a quantised head on the bf16-y row (logits are f32)
 *   a `zero` plane: codes+scale+zero is gemm_nt_q's asymmetric grid (r3d_gemm_nt_q_w2a8.hip)
 *   a joined operand that is not two planes
 *   planes that are not codes + scale (a `table`, a `bias`)
 *   codes tiled by anything but one code a cell
 *   sub-byte codes (4-bit)
 *   e5m2 and every other code dtype: no decoder here
 *   a scale plane that is a table, not a tiling
 *   an integer scale (not in float_scale_dt's set)
 *   a scale block that is not 128x128: per-(row, group) scales are gemm_nt_q's grid, and a per-row
 *        scale indexes the plane differently from ScaledRows::cover
 *   planes that do not cover the band (codes N x K, scales ceil(N/128) x ceil(K/128)), or a
 *        row_bytes shorter than the row
 * The bf16 branch refuses a plain head that is not bf16, wrong x and y dtypes, bad pitches, a short
 * or non-contiguous head, absurd shapes and misalignment. The launcher refuses a head with
 * a null plane, a dtype outside the served set, a row pitch a vector load cannot walk (a 16-bit
 * scale row is 2 bytes a column, so K = 320, 6 bytes a row, is legal), and misaligned addresses.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_fp8_dequant.h"   /* the E4M3 -> f16 LDS table */
#include "r3d_tiling.h"          /* LDS tile addressing: srow, srow_pair */
#include "r3d_args.h"
#include "r3d_launch.h"

#include "rad_planes.h"

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWaves        = kBlockThreads / kWave;   /* 8 */
constexpr int kTM           = 128;                     /* activation rows a block */
constexpr int kTN           = 128;                     /* weight rows a block */
constexpr int kKC           = 32;                      /* codes a tile step */
constexpr int kSkinnyMax    = 16;                      /* the MT ceiling of the decode arm */
constexpr int kChunkK       = 128;                     /* the decode arm's activation chunk */

#if defined(__has_builtin)
#  if __has_builtin(__builtin_nontemporal_load)
#    define R3D_HAS_NT 1
#  endif
#endif
#ifndef R3D_HAS_NT
#  define R3D_HAS_NT 0
#endif

/* Tile addressing (srow, srow_pair) is r3d_tiling.h's: `pitch`-wide f16 rows whose eight-element
 * chunks are XORed by the row's low bits, used here at pitches 32 and 128. */

/* Eight bf16 -> eight f16, exact for every value f16 can hold. */
__device__ __forceinline__ uint4 widen8(uint4 w)
{
    return make_uint4(bf16x2_to_f16x2(w.x), bf16x2_to_f16x2(w.y), bf16x2_to_f16x2(w.z),
                      bf16x2_to_f16x2(w.w));
}

constexpr int kFp8Block = 128;        /* the head's scale block: R3D_FP8_BLOCK, llama_dense_fp8 */

/* The head as the container stores it: E4M3 (or int8) codes with one float scale per 128x128
 * block. Passed by value; the pointers are device addresses the shim resolved out of the joined
 * bytes (rad_planes_find). */
struct HeadQ {
    const uint8_t* codes;        /* [N][K] */
    const uint8_t* scale;        /* [ceil(N/128)][ceil(K/128)] */
    long long code_row;          /* bytes a codes row: the plane's row_bytes, not K's element count */
    long long scale_row;         /* bytes a scale row */
    int sdt;                     /* the scale's dtype: 0 bf16, 1 f16, 2 f32 (float_scale_dt's set) */
    int cdt;                     /* the codes' dtype: 0 E4M3, 1 int8 */
};

/* The scale covering logical (n, k), the reference's `ScaledRows::cover`. The plane is 32 KiB at
 * the band and stays in cache, so the codes are the traffic, not these loads. */
__device__ __forceinline__ float scale_at(const HeadQ h, int n, int k)
{
    const uint8_t* p = h.scale + (long long)(n >> 7) * h.scale_row
                     + (long long)(k >> 7) * (h.sdt == 2 ? 4 : 2);
    if (h.sdt == 2) { float f; __builtin_memcpy(&f, p, 4); return f; }
    const uint16_t u = *(const uint16_t*)p;
    return h.sdt == 1 ? f16_to_f32(u) : bf16_to_f32(u);
}

/* Eight codes -> eight f16, exact: E4M3 (4 exponent, 3 mantissa bits) and int8 both fit f16, so
 * decoding is a widening, not a rounding. The pair decoder reads the two codes in a dword's low
 * half, so each dword is decoded as two pairs (d and d >> 16): two dwords in, four pairs out. */
#if defined(__has_builtin)
#  if __has_builtin(__builtin_amdgcn_cvt_pk_f16_fp8)
#    define R3D_HAS_FP8CVT 1        /* two codes to two f16; software on gfx1102, see the header */
#  endif
#endif
#ifndef R3D_HAS_FP8CVT
#  define R3D_HAS_FP8CVT 0
#endif

__device__ __forceinline__ uint32_t cvt_pk_f16_fp8(uint32_t b)
{
#if R3D_HAS_FP8CVT
    return __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pk_f16_fp8(b));
#else
    return f16x2_pack(e4m3_to_f32((uint8_t)(b & 0xffu)), e4m3_to_f32((uint8_t)((b >> 8) & 0xffu)));
#endif
}

template <int ALU>   /* ALU=1: the per-byte ALU decode (no table); ALU=0: the LDS table */
__device__ __forceinline__ void codes8_to_f16x4(const uint2 c, int cdt, const uint16_t* tbl,
                                                uint32_t* w)
{
    const uint32_t d[2] = { c.x, c.y };
    if (!cdt && (ALU || tbl)) {
        /* The shared E4M3 decode: ALU=1 the per-byte formula, ALU=0 the LDS table
         * (r3d_fp8_dequant.h). A null tbl with ALU=0 selects the codec below. */
        r3d_e4m3x8_decode<ALU>(tbl, c, w);
        return;
    }
    if (!cdt) {
        /* The codec: on gfx1102 the builtin expands to software, about 100 instructions per eight
         * codes, which is why the table path exists. A target with v_cvt_pk_f16_fp8 gets the
         * single instruction. */
        w[0] = cvt_pk_f16_fp8(d[0]);
        w[1] = cvt_pk_f16_fp8(d[0] >> 16);
        w[2] = cvt_pk_f16_fp8(d[1]);
        w[3] = cvt_pk_f16_fp8(d[1] >> 16);
        return;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t pair = (d[j >> 1] >> (16 * (j & 1))) & 0xffffu;
        w[j] = i8x2_to_f16x2(pair);
    }
}

/* The fold: the decoded code times the scale covering it, in f16 pairs.
 *
 * The reference dequantises each element into f32 (`ScaledRows::at`). Folding into the staged f16
 * costs one f16 rounding per weight (2.4e-4 relative, against the code's own 2^-4 quantum) and
 * keeps the tile f16, the dot loop unchanged and the prefill arm's 237 VGPRs as they are. A
 * per-block partial summed and scaled separately would need 64 more accumulators and spill. */
__device__ __forceinline__ uint32_t scale_pair(uint32_t w, float sc)
{
#if defined(__HIP_DEVICE_COMPILE__)
    /* One hardware f16 multiply (v_mul_f16), round to nearest, subnormals kept.
     *
     * Not f16x2_pack of an f32 product: that is v_cvt_pkrtz_f16_f32, which rounds toward zero and
     * is only for values f16 represents exactly (r3d_codecs.h). A folded weight generally is not
     * (code 431 times a scale of 4.5e-5 has eleven significant bits), and truncation biases every
     * product toward zero and flushes the subnormal end.
     */
    const uint16_t lo16 = (uint16_t)(w & 0xffffu), hi16 = (uint16_t)(w >> 16);
    const _Float16 a = __builtin_bit_cast(_Float16, lo16);
    const _Float16 b = __builtin_bit_cast(_Float16, hi16);
    const _Float16 s = (_Float16)sc;                 /* f32 -> f16, round to nearest even */
    const _Float16 rl = a * s, rh = b * s;           /* v_mul_f16: one rounding, subnormals kept */
    return (uint32_t)__builtin_bit_cast(uint16_t, rl) |
           ((uint32_t)__builtin_bit_cast(uint16_t, rh) << 16);
#else
    return (uint32_t)f32_to_f16(f16_to_f32((uint16_t)(w & 0xffffu)) * sc) |
           ((uint32_t)f32_to_f16(f16_to_f32((uint16_t)(w >> 16)) * sc) << 16);
#endif
}

/* The head's 16-byte load, ordinary or streamed-once. */
template <int NT>
__device__ __forceinline__ uint4 loadw8(const uint16_t* p)
{
#if R3D_HAS_NT
    if (NT) {
        /* The builtin wants a native vector, not HIP's class: four unsigned ints, bit-cast. */
        typedef unsigned int v4u __attribute__((ext_vector_type(4)));
        const v4u t = __builtin_nontemporal_load((const v4u*)p);
        uint4 r;
        __builtin_memcpy(&r, &t, 16);
        return r;
    }
#else
    (void)NT;
#endif
    return *(const uint4*)p;
}

/* The quantised head's 8-byte load of eight codes, ordinary or streamed-once. */
template <int NT>
__device__ __forceinline__ uint2 loadc8(const uint8_t* p)
{
#if R3D_HAS_NT
    if (NT) {
        typedef unsigned int v2u __attribute__((ext_vector_type(2)));
        const v2u t = __builtin_nontemporal_load((const v2u*)p);
        uint2 r;
        __builtin_memcpy(&r, &t, 8);
        return r;
    }
#else
    (void)NT;
#endif
    return *(const uint2*)p;
}

/* ============================================================ the prefill arm, M > 16
 *
 * y is f32 for logits_gemm and bf16 for gemm_nt (`YF32`). WQ reads the quantised head through h.
 */
template <bool YF32, bool WQ, int ALU>
__global__ void __launch_bounds__(kBlockThreads) R3D_WAVES(2)
gemm_tiled_kernel(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, HeadQ h,
                  void* __restrict__ y, int M, int N, int K, int a_ld, int y_ld)
{
    (void)b;   /* the plain head; WQ reads the container through h */
    __shared__ __align__(16) uint16_t sA[kTM * kKC];   /* f16, TM rows x KC, swizzled */
    __shared__ __align__(16) uint16_t sB[kTN * kKC];   /* f16, TN rows x KC, swizzled */

    const int tid = (int)threadIdx.x;
    __shared__ __align__(16) uint16_t sTbl[R3D_E4M3_TBL];   /* the E4M3 decoder, 512 bytes */
    if (WQ) r3d_e4m3_f16_init(sTbl, tid, kBlockThreads);    /* fills, then barriers */
    const uint16_t* const r3d_tbl = (WQ && !ALU) ? sTbl : (const uint16_t*)nullptr;
    const int m0 = (int)blockIdx.y * kTM;
    const int n0 = (int)blockIdx.x * kTN;
    const int rg = tid >> 4, cg = tid & 15;           /* this lane's 8x8 square */

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;

    const int m_r = m0 + rg * 8, n_c = n0 + cg * 8;

    for (int k0 = 0; k0 < K; k0 += kKC) {
        /* Stage 128 rows x 4 chunks of eight per side, two uint4 a thread a side. Rows past M and
         * k past K stage as zeros, which contribute nothing to a sum. */
#pragma unroll
        for (int j = tid; j < kTM * (kKC / 8); j += kBlockThreads) {
            const int mm = j / (kKC / 8), c8 = j % (kKC / 8);
            const int m = m0 + mm, k = k0 + c8 * 8;
            uint4 w = make_uint4(0, 0, 0, 0);
            if (m < M && k < K) w = *(const uint4*)(a + (long long)m * a_ld + k);
            *(uint4*)&sA[srow(mm, c8, kKC)] = widen8(w);
        }
#pragma unroll
        for (int j = tid; j < kTN * (kKC / 8); j += kBlockThreads) {
            const int nn = j / (kKC / 8), c8 = j % (kKC / 8);
            const int n = n0 + nn, k = k0 + c8 * 8;
            uint4 staged = make_uint4(0, 0, 0, 0);
            if (n < N && k < K) {
                if (WQ) {
                    /* Eight codes, one scale: the staged run lies inside one 128x128 block, and
                     * the scale is indexed by this staged row n, not the lane's output row. */
                    const uint2 c = *(const uint2*)(h.codes + (long long)n * h.code_row + k);
                    uint32_t w[4];
                    codes8_to_f16x4<ALU>(c, h.cdt, r3d_tbl, w);
                    const float sc = scale_at(h, n, k);
                    staged = make_uint4(scale_pair(w[0], sc), scale_pair(w[1], sc),
                                        scale_pair(w[2], sc), scale_pair(w[3], sc));
                } else {
                    staged = widen8(*(const uint4*)(b + (long long)n * K + k));
                }
            }
            *(uint4*)&sB[srow(nn, c8, kKC)] = staged;
        }
        __syncthreads();

#pragma unroll
        for (int kc = 0; kc < kKC; kc += 2) {
            uint32_t af[8], bf[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                af[i] = *(const uint32_t*)&sA[srow_pair(rg * 8 + i, kc, kKC)];
#pragma unroll
            for (int j = 0; j < 8; ++j)
                bf[j] = *(const uint32_t*)&sB[srow_pair(cg * 8 + j, kc, kKC)];
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = dot2_f16x2(af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

    /* The store, eight columns at a time: 32 bytes of f32, 16 of bf16. Rows past M are skipped
     * and a ragged N tail is written column by column. */
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int m = m_r + i;
        if (m >= M) break;
        if (n_c + 8 <= N) {
            if (YF32) {
                float* row = (float*)y + (long long)m * y_ld;
#pragma unroll
                for (int j = 0; j < 8; j += 4)
                    *(uint4*)(row + n_c + j) =
                        make_uint4(__float_as_uint(acc[i][j]), __float_as_uint(acc[i][j + 1]),
                                   __float_as_uint(acc[i][j + 2]), __float_as_uint(acc[i][j + 3]));
            } else {
                uint16_t* row = (uint16_t*)y + (long long)m * y_ld;
                uint32_t p[4];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    p[j] = (uint32_t)f32_to_bf16(acc[i][2 * j]) |
                           ((uint32_t)f32_to_bf16(acc[i][2 * j + 1]) << 16);
                *(uint4*)(row + n_c) = make_uint4(p[0], p[1], p[2], p[3]);
            }
        } else {
            /* Unrolled: a dynamic index into acc would put the 64 accumulators in local memory
             * (80 KiB of spill against a 64 KiB limit). */
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (n_c + j < N) {
                    if (YF32) ((float*)y)[(long long)m * y_ld + n_c + j] = acc[i][j];
                    else ((uint16_t*)y)[(long long)m * y_ld + n_c + j] = f32_to_bf16(acc[i][j]);
                }
            }
        }
    }
}

/* ============================================================ the decode arm, M <= 16
 *
 * A wave is MT activation rows by KLW = 32/MT weight rows, one accumulator a lane over all of k in
 * ascending order. The activation tile is block-wide, so every thread stages it, including lanes
 * whose weight row is past the vocabulary; those lanes skip the dot and the store.
 */
template <int MT, bool YF32, int NT, bool WQ, int ALU>
__global__ void __launch_bounds__(kBlockThreads) R3D_WAVES(4)
logits_skinny_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, HeadQ h,
                     void* __restrict__ y, int M, int N, int K, int x_ld, int y_ld)
{
    constexpr int KLW = kWave / MT;
    constexpr int STEPS = MT * (kChunkK / 8);          /* uint4 staging steps a chunk */

    __shared__ __align__(16) uint16_t sX[MT * kChunkK];   /* f16, MT rows x chunk, swizzled */
    __shared__ __align__(16) uint16_t sTbl[R3D_E4M3_TBL];
    if (WQ && !ALU) r3d_e4m3_f16_init(sTbl, (int)threadIdx.x, kBlockThreads);
    const uint16_t* const r3d_tbl = (WQ && !ALU) ? sTbl : (const uint16_t*)nullptr;

    const int lane = (int)threadIdx.x & (kWave - 1);
    const int warp = (int)threadIdx.x / kWave;
    const int m = lane & (MT - 1);
    const int n = (int)blockIdx.x * (kWaves * KLW) + warp * KLW + lane / MT;
    const bool live = n < N;

    /* A dead lane's row pointer is clamped to N-1 so it never points past the head. */
    const uint16_t* wrow = w + (long long)(live ? n : N - 1) * K;
    const int nsteps = M < MT ? M : MT;
    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += kChunkK) {
        for (int j = (int)threadIdx.x; j < STEPS; j += kBlockThreads) {
            const int r = j / (kChunkK / 8), c8 = j % (kChunkK / 8);
            const int k = k0 + c8 * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (r < nsteps && k < K) v = *(const uint4*)(x + (long long)r * x_ld + k);
            *(uint4*)&sX[srow(r, c8, kChunkK)] = widen8(v);
        }
        __syncthreads();

        if (live) {
            const uint16_t* wp = wrow + k0;
            const uint8_t* cp = h.codes + (long long)(live ? n : N - 1) * h.code_row + k0;
            /* One scale a chunk: kChunkK equals the scale block, so a lane's 128 codes share one
             * (n-block, k-block). */
            const float sc = WQ ? scale_at(h, live ? n : N - 1, k0) : 0.0f;
            const int kend = K - k0 < kChunkK ? K - k0 : kChunkK;   /* a multiple of 8: the
                                                                     * launcher takes K % 8 == 0 */
            for (int kc = 0; kc < kend; kc += 8) {
                uint32_t w0, w1, w2, w3;
                if (WQ) {
                    uint32_t q[4];
                    codes8_to_f16x4<ALU>(loadc8<NT>(cp + kc), h.cdt, r3d_tbl, q);
                    w0 = scale_pair(q[0], sc); w1 = scale_pair(q[1], sc);
                    w2 = scale_pair(q[2], sc); w3 = scale_pair(q[3], sc);
                } else {
                    const uint4 wv = loadw8<NT>(wp + kc);
                    w0 = bf16x2_to_f16x2(wv.x); w1 = bf16x2_to_f16x2(wv.y);
                    w2 = bf16x2_to_f16x2(wv.z); w3 = bf16x2_to_f16x2(wv.w);
                }
                const int s = srow_pair(m, kc, kChunkK);
                acc = dot2_f16x2(*(const uint32_t*)&sX[s], w0, acc);
                acc = dot2_f16x2(*(const uint32_t*)&sX[s + 2], w1, acc);
                acc = dot2_f16x2(*(const uint32_t*)&sX[s + 4], w2, acc);
                acc = dot2_f16x2(*(const uint32_t*)&sX[s + 6], w3, acc);
            }
        }
        __syncthreads();
    }

    if (live && m < M) {
        if (YF32) ((float*)y)[(long long)m * y_ld + n] = acc;
        else ((uint16_t*)y)[(long long)m * y_ld + n] = f32_to_bf16(acc);
    }
}

/* ------------------------------------------------------------------ the shared launcher
 *
 * Returns 0, -1 a null pointer, -2 a shape it does not serve, -3 a pitch it cannot read, -4 the HIP
 * launch failed, -5 an address it cannot read at 16 bytes.
 */
template <bool YF32>
int gemm_launch(long xv, long wv, long yv, int M, int N, int K, int x_ld, int y_ld, int sk, int kb,
                int nt, long stream, const HeadQ* hq = nullptr, int alu = 1)
{
    if (!xv || !wv || !yv) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (sk > 1) return -2;                       /* the grid is vocab-wide at every M, see header */
    if (kb == 0) kb = kKC;
    if (kb != kKC) return -2;                    /* 32 is the tile's k-chunk and the only one */
    if (nt < 0 || nt > 1) return -2;
    if (K % kKC) return -2;                      /* the tile's chunk; 2048 at the band */
    if (K % 8) return -3;                        /* 16-byte weight loads */
    if (!x_ld) x_ld = K;
    if (!y_ld) y_ld = N;
    if (x_ld % 8) return -3;                     /* 16-byte activation loads (bf16) */
    if (y_ld % (YF32 ? 4 : 8)) return -3;        /* 16-byte store runs (f32: 4, bf16: 8) */
    if (x_ld < K || y_ld < N) return -3;
    if ((xv & 15) || (wv & 15) || (yv & 15)) return -5;

    const uint16_t* x = (const uint16_t*)(uintptr_t)xv;
    const uint16_t* w = (const uint16_t*)(uintptr_t)wv;
    void* y = (void*)(uintptr_t)yv;
    hipStream_t s = (hipStream_t)(uintptr_t)stream;

    /* `hq` is the quantised head the shim resolved from the container; its pointers are checked
     * here. The declared entries pass null and get the plain bf16 head their r3d.h signature
     * describes. */
    HeadQ h;
    std::memset(&h, 0, sizeof h);
    const int wq = hq != nullptr;
    if (hq) {
        h = *hq;
        if (!h.codes || !h.scale) return -1;
        if (h.sdt < 0 || h.sdt > 2 || h.cdt < 0 || h.cdt > 1) return -2;
        if (h.code_row & 7) return -3;        /* 8-byte code loads, one row a stride */
        if (h.scale_row & (h.sdt == 2 ? 3 : 1))
            return -3;                        /* scale loads are 4 bytes (f32) or 2 (bf16/f16), so a
                                               * 3-column 16-bit row (6 bytes, K = 320) is legal */
        if (((long long)(uintptr_t)h.codes & 7) || ((long long)(uintptr_t)h.scale & 3)) return -5;
    }

    if (M <= kSkinnyMax) {
#define R3D_SKINNY(MTV, NTV, WQV, ALUV)                                                                \
    do {                                                                                         \
        const int klw = kWave / (MTV);                                                           \
        const int per = kWaves * klw;                                                            \
        const dim3 grid((unsigned)((N + per - 1) / per)), block(kBlockThreads);                  \
        R3D_KLAUNCH(R3D_KNAME((logits_skinny_kernel<MTV, YF32, NTV, WQV, ALUV>)), grid, block, s, x, w, \
                    h, y, M, N, K, x_ld, y_ld);                                                  \
    } while (0)
#define R3D_SKINNY_MT(MTV)                                                                       \
    do {                                                                                         \
        if (wq) {                                                                                 \
            if (alu) { if (nt) R3D_SKINNY(MTV, 1, 1, 1); else R3D_SKINNY(MTV, 0, 1, 1); }         \
            else     { if (nt) R3D_SKINNY(MTV, 1, 1, 0); else R3D_SKINNY(MTV, 0, 1, 0); }         \
        } else {                                                                                  \
            if (nt) R3D_SKINNY(MTV, 1, 0, 0); else R3D_SKINNY(MTV, 0, 0, 0);                      \
        }                                                                                        \
    } while (0)
        if (M == 1) R3D_SKINNY_MT(1);
        else if (M <= 2) R3D_SKINNY_MT(2);
        else if (M <= 4) R3D_SKINNY_MT(4);
        else if (M <= 8) R3D_SKINNY_MT(8);
        else R3D_SKINNY_MT(16);
#undef R3D_SKINNY_MT
#undef R3D_SKINNY
    } else {
        const dim3 grid((unsigned)((N + kTN - 1) / kTN), (unsigned)((M + kTM - 1) / kTM));
        const dim3 block(kBlockThreads);
        if (wq) {
            if (alu)
                R3D_KLAUNCH(R3D_KNAME((gemm_tiled_kernel<YF32, true, 1>)), grid, block, s, x, w, h, y, M,
                            N, K, x_ld, y_ld);
            else
                R3D_KLAUNCH(R3D_KNAME((gemm_tiled_kernel<YF32, true, 0>)), grid, block, s, x, w, h, y, M,
                            N, K, x_ld, y_ld);
        } else
            R3D_KLAUNCH(R3D_KNAME((gemm_tiled_kernel<YF32, false, 0>)), grid, block, s, x, w, h, y, M,
                        N, K, x_ld, y_ld);
    }
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* The shim body both ops share (ref_logits_gemm is gemm_nt_impl with the axes renamed).
 * `y_is_f32` is the op's output dtype; nkey/kkey are libref's axis names ("n_vocab"/"n_embd" or
 * "N"/"K"). */
int rad_gemm_run(const RadArgs* a, RadStream s, int y_is_f32, const char* nkey, const char* kkey, int alu)
{
    if (a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data)
        return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = &a->t[0];
    const RadTensor* w = &a->t[1];
    const RadTensor* y = &a->t[2];

    const RadPlanesEntry* ce = nullptr;
    const RadPlanesEntry* se = nullptr;
    HeadQ h;
    std::memset(&h, 0, sizeof h);
    int joined = 0;

    /* ---- the head's stored form ---------------------------------------------------------------
     * The launch-side twin of r3d_layout_logits_head: a `rad.planes` header means the quantised
     * head, read as its two planes; no header means a plain bf16 head. The checks follow the
     * reference's `scaled_rows` in order: roles, then dtypes and blocks, then coverage (below,
     * once N and K are known).
     */
    const RadPlanesHeader* ph = rad_planes_header(r3d_p(w));
    if (ph) {
        if (!y_is_f32) return R3D_NO(RAD_E_DTYPE);   /* the bf16-y row takes a bf16 head only */
        if (rad_planes_find(r3d_p(w), "zero", nullptr))
            return R3D_NO(RAD_E_DTYPE);   /* codes+scale+zero: gemm_nt_q's asymmetric grid */
        if (ph->n_planes != 2) return R3D_NO(RAD_E_DTYPE);
        const uint8_t* cb = (const uint8_t*)rad_planes_find(r3d_p(w), "codes", &ce);
        const uint8_t* sb = (const uint8_t*)rad_planes_find(r3d_p(w), "scale", &se);
        if (!cb || !sb || !ce || !se)
            return R3D_NO(RAD_E_DTYPE);   /* a joined head that is not codes + one scale */
        if (ce->kind != RAD_PLANE_TILED || ce->block[0] != 1 || ce->block[1] != 1)
            return R3D_NO(RAD_E_DTYPE);   /* codes must be one code an element, tiled plain */
        if (ce->dtype != RAD_F8E4M3 && ce->dtype != RAD_I8) {
            if (rad_dtype_bytes(ce->dtype, 8) < 8)
                return R3D_NO(RAD_E_DTYPE);   /* sub-byte codes */
            return R3D_NO(RAD_E_DTYPE);       /* e5m2 and the rest: no decoder here */
        }
        if (se->kind != RAD_PLANE_TILED) return R3D_NO(RAD_E_DTYPE);
        if (se->dtype == RAD_BF16) h.sdt = 0;
        else if (se->dtype == RAD_F16) h.sdt = 1;
        else if (se->dtype == RAD_F32) h.sdt = 2;
        else return R3D_NO(RAD_E_DTYPE);        /* not in float_scale_dt's set */
        if (se->block[0] != kFp8Block || se->block[1] != kFp8Block)
            return R3D_NO(RAD_E_SHAPE);   /* only 128x128 blocks; other grids are gemm_nt_q's */
        h.codes = cb; h.scale = sb; h.code_row = ce->row_bytes; h.scale_row = se->row_bytes;
        h.cdt = ce->dtype == RAD_I8 ? 1 : 0;
        joined = 1;
    } else if (w->dtype != RAD_BF16) {
        return R3D_NO(RAD_E_DTYPE);
    }
    if (x->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (y->dtype != (y_is_f32 ? (uint32_t)RAD_F32 : (uint32_t)RAD_BF16)) return R3D_NO(RAD_E_DTYPE);

    long long N = r3d_geti_or(a, nkey, 0);
    long long K = r3d_geti_or(a, kkey, 0);
    if (N <= 0) N = r3d_dim(w, 0);
    if (K <= 0) K = r3d_dim(w, 1);
    if (N <= 0 || K <= 0) return R3D_NO(RAD_E_SHAPE);

    long long M = 0, x_rows = 0;
    const long long y_pitch = r3d_rows_pitch(y, N, &M);      /* M = numel(y)/N, the reference's fold */
    if (y_pitch < N || M <= 0) return R3D_NO(RAD_E_STRIDE);
    const long long x_pitch = r3d_rows_pitch(x, K, &x_rows);
    if (x_pitch < K || x_rows < M) return R3D_NO(RAD_E_STRIDE);
    if (r3d_numel(w) < N * K) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(w)) return R3D_NO(RAD_E_STRIDE);         /* N rows of K contiguous codes */
    if (joined) {
        /* `scaled_rows`' coverage checks: the planes must cover the band, the scale plane by
         * ceil-divided blocks, so a ragged vocabulary is fine (131 rows need 2 scale rows). */
        const long long sb_r = (N + kFp8Block - 1) / kFp8Block;
        const long long sb_c = (K + kFp8Block - 1) / kFp8Block;
        if (ce->rows != N || ce->cols != K) return R3D_NO(RAD_E_SHAPE);
        if (se->rows != sb_r || se->cols != sb_c) return R3D_NO(RAD_E_SHAPE);
        if (ce->row_bytes < ce->cols || se->row_bytes < se->cols * (h.sdt == 2 ? 4 : 2))
            return R3D_NO(RAD_E_STRIDE);
    }
    if (M > (long long)1 << 24 || N > (long long)1 << 28 || K > (long long)1 << 28)
        return R3D_NO(RAD_E_SHAPE);
    if (!r3d_aligned(r3d_p(x), 16) || !r3d_aligned(r3d_p(w), 16) || !r3d_aligned(r3d_p(y), 16))
        return R3D_NO(RAD_E_ALIGN);

    /* No `sk`/`kb`/`nt` in the rows' tune tables (impl only): the grid is vocab-wide, KC = 32 and
     * the non-temporal stream is off, see the header's refusals. */
    const int sk = 1, kb = 0, nt = 0;

    const HeadQ* hp = joined ? &h : nullptr;
    const int e = y_is_f32
        ? gemm_launch<true>(r3d_addr(x), r3d_addr(w), r3d_addr(y), (int)M, (int)N, (int)K,
                            (int)x_pitch, (int)y_pitch, sk, kb, nt, (long)(uintptr_t)s, hp, alu)
        : gemm_launch<false>(r3d_addr(x), r3d_addr(w), r3d_addr(y), (int)M, (int)N, (int)K,
                             (int)x_pitch, (int)y_pitch, sk, kb, nt, (long)(uintptr_t)s, hp, alu);
    if (e == 0) return RAD_OK;
    return R3D_NO(e == -1 ? RAD_E_INVAL : e == -2 ? RAD_E_SHAPE
                        : e == -3 ? RAD_E_STRIDE : e == -5 ? RAD_E_ALIGN : RAD_E_DEVICE);
}

/* The reference's `enc_scaled_rows` and `float_scale_dt`, transcribed (libref keeps them static
 * inline), checks in its order: affine, untransformed, two planes, the roles in order, codes tiled
 * with block {1,1} in an admitted dtype, the scale tiled with a float dtype. */
bool float_scale_dt(uint32_t dt)
{
    return dt == RAD_BF16 || dt == RAD_F16 || dt == RAD_F32;
}

bool enc_scaled_rows(const RadEncoding* enc, const int* sel, int n)
{
    if (!rad_enc_is(enc, "affine") || enc->transform[0] || enc->n_planes != 2) return false;
    if (n != 2 || !sel) return false;                 /* sel_roles(enc, sel, n, "codes", "scale") */
    const char* want[2] = { "codes", "scale" };
    for (int i = 0; i < 2; ++i) {
        if (sel[i] < 0 || sel[i] >= enc->n_planes) return false;
        if (std::strncmp(enc->plane[sel[i]].role, want[i], RAD_ENC_STR) != 0) return false;
    }
    const RadEncPlane& c = enc->plane[sel[0]];
    const RadEncPlane& sc = enc->plane[sel[1]];
    return c.kind == RAD_PLANE_TILED && c.block[0] == 1 && c.block[1] == 1 &&
           (c.dtype == RAD_F8E4M3 || c.dtype == RAD_I8) && sc.kind == RAD_PLANE_TILED &&
           float_scale_dt(sc.dtype);
}

}  // namespace

/* logits[M, n_vocab] = x[M, n_embd] @ lm_head[n_vocab, n_embd]^T, the declared entry (r3d.h's
 * r3d_vocab_bf16.hip section). M <= 16 takes the decode arm, above it the prefill tile. */
extern "C" int r3d_logits_gemm_bf16(long x, long lm, long y, int M, int n_vocab, int n_embd,
                                    int x_ld, int y_ld, int sk, int nt, long scratch, long stream)
{
    (void)scratch;                       /* no split, so no partials: see the header */
    return gemm_launch<true>(x, lm, y, M, n_vocab, n_embd, x_ld, y_ld, sk, kKC, nt, stream);
}

/* The same tiling with a bf16 y, for gemm_nt at prefill sizes (M >= 17; M <= 16 is the m16 row). */
extern "C" int r3d_gemm_nt_bf16_tiled(long a, long b, long y, int M, int N, int K, int a_ld,
                                      int y_ld, int sk, int kb, int nt, long scratch, long stream)
{
    (void)scratch;
    return gemm_launch<false>(a, b, y, M, N, K, a_ld, y_ld, sk, kb, nt, stream);
}

static int r3d_logits_gemm_bf16_dispatch(const RadArgs* a, RadStream s, int alu)
{ return rad_gemm_run(a, s, 1, "n_vocab", "n_embd", alu); }
static int r3d_gemm_nt_bf16_tiled_dispatch(const RadArgs* a, RadStream s, int alu)
{ return rad_gemm_run(a, s, 0, "N", "K", alu); }

extern "C" int r3d_rad_gemm_nt_bf16_tiled_wmma(const RadArgs*, RadStream);
extern "C" int r3d_rad_gemm_nt_bf16_m64_wmma(const RadArgs*, RadStream);

/* The rows' impl axis picks the kernel: the table decode (0) and the ALU decode (1) share the
 * dispatch, and the bf16 tiled row's two wave-matrix arms (2 = the 128x128 tile, 3 = the
 * 16-column-tile kernel) run through their own shims, which stay as entry points. */
extern "C" int r3d_rad_logits_gemm_bf16(const RadArgs* a, RadStream s)
{
    R3dGemmCfg c;
    r3d_cfg_gemm(a, &c);
    return r3d_logits_gemm_bf16_dispatch(a, s, c.impl);
}
extern "C" int r3d_rad_logits_gemm_bf16_alu(const RadArgs* a, RadStream s)
{ return r3d_logits_gemm_bf16_dispatch(a, s, 1); }
extern "C" int r3d_rad_gemm_nt_bf16_tiled(const RadArgs* a, RadStream s)
{
    R3dGemmCfg c;
    r3d_cfg_gemm(a, &c);
    if (c.impl <= 1) return r3d_gemm_nt_bf16_tiled_dispatch(a, s, c.impl);
    if (c.impl == 2) return r3d_rad_gemm_nt_bf16_tiled_wmma(a, s);
    return r3d_rad_gemm_nt_bf16_m64_wmma(a, s);
}
extern "C" int r3d_rad_gemm_nt_bf16_tiled_alu(const RadArgs* a, RadStream s)
{ return r3d_gemm_nt_bf16_tiled_dispatch(a, s, 1); }

/* The head's relayout writer. A layout query that answers RAD_OK promises the loader that
 * relayout will produce the described form (rad_abi.h). The container stores the quantiser's
 * pair and the plan wants one operand, so rad_planes_relayout assembles it losslessly: the header,
 * the entries and each plane's rows, copied. No code is transformed, so the served numbers do not
 * depend on which library loaded the container. The query and the writer share one plan
 * (rad_planes_plan). The operand guard repeats the hook's, so a misrouted call writes nothing. */
extern "C" int r3d_relayout_logits_head(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* planes, int n_planes, void* dst,
                                        int64_t dst_bytes)
{
    (void)p; (void)n_p;
    if (operand != 1) return R3D_NO(RAD_E_UNSUPPORTED);
    return rad_planes_relayout(enc, sel, planes, n_planes, dst, dst_bytes);
}

/* The head's layout hook, `ref::ref_layout_head` transcribed: operand 1 of logits_gemm, two planes
 * (codes and one scale), untransformed, answered with the container's own form and no byte moved.
 * Operand 0 (the activations), a one-plane selection and every other encoding are declined, as
 * `layout_scaled` declines them. */
extern "C" int r3d_layout_logits_head(const RadParam* p, int n_p, int operand,
                                      const RadEncoding* enc, const int* sel,
                                      const RadTensor* planes, int n, RadLayout* out)
{
    (void)p; (void)n_p;                                        /* layout_scaled ignores both */
    if (n < 2 || operand != 1) return RAD_E_UNSUPPORTED;       /* layout_scaled(operand, 1, -1, ...) */
    if (!enc_scaled_rows(enc, sel, n)) return RAD_E_DTYPE;
    return rad_planes_layout(enc, sel, planes, n, out);
}
