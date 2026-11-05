/* r3d_gemm_mxfp4a8.hip -- gemm_nt_q at dtype=mxfp4a8: an OCP-MXFP4 weight against an fp8
 * activation, at both heights.
 *
 *   y[M,N] = a[M,K] @ b[N,K]^T, a in E4M3 with one f32 scale per (row, 128 of K), b in e2m1
 *   with one e8m0 scale per (row, 32 of K). f32 accumulate, one bf16 narrowing at the store.
 *   There is no zero point: the scale is symmetric.
 *
 * ---- the format, and the fold it gives ----------------------------------------------------------------
 *
 * OCP MXFP4 stores a weight row as e2m1 codes, two per byte, element 2j in the low nibble of
 * byte j, with one e8m0 exponent per 32 K elements: a u8 plane [N, K/32] beside the u8 codes
 * plane [N, K/2]. e8m0 value e means the factor 2^(e-127); e = 0 and e = 255 are reserved (zero,
 * NaN), and a plane carrying either is refused, as the reference does.
 *
 * Per 32-block:
 *
 *   1. the code for element k becomes e4m3(mag): all eight e2m1 magnitudes
 *      {0, .5, 1, 1.5, 2, 3, 4, 6} are e4m3 grid points, so this is exact. The unpack is a
 *      4-bit lookup on the nibble through the e4m3 bytes {0x00, 0x30, 0x38, 0x3C, 0x40, 0x44,
 *      0x48, 0x4C}, with the code's sign bit moved to the byte's sign bit.
 *   2. the products run on dot2_f16x2. A code product is at most 6 x 448 = 2688, inside f16's
 *      range (65504), and an eight-term partial is under 21504.
 *   3. the block factor 2^(e-127) is applied in the epilogue: the f32 32-block partial is
 *      multiplied by it (a power of two, exact short of under/overflow) and added to the running
 *      f32 sum. One multiply and one add per block per output, no per-element scale traffic.
 *   4. the store narrows once, to bf16.
 *
 * The roundings are the activation-scale fold at staging (fold_pair, f16), the dot's fused
 * rounding, and the store.
 *
 * The RDNA4 sibling folds the e8 into a d-indexed table row because its WMMA path needs one scale
 * per row; the dot2 path here applies it per block in the epilogue, so the table is the eight
 * fixed magnitude bytes.
 *
 * ---- the tiled row (M >= 17) -------------------------------------------------------------------------
 *
 * A 128x128 tile, 8x8 f32 accumulators per lane, k ascending in 32-code chunks. A chunk is one OCP
 * 32-block: the e8 byte of each tile column is staged through LDS, and the chunk's partials flush
 * at its end:
 *
 *     total[i][j] += acc[i][j] * 2^(e8[n_c + j][blk] - 127);  acc[i][j] = 0;
 *
 * `acc` is the 32-block partial, `total` the running f32 sum over blocks in ascending k. The
 * activation stages as in the fp8a8 tiled row: E4M3 through the LDS table, with the f32 scale per
 * (row, 128 of K) folded in at staging. 32 divides 128, so a chunk never straddles a group.
 *
 * ---- the m16 row (M <= 16) ----------------------------------------------------------------------------
 *
 * One wave per weight row, as in the w2a8 row: lane l owns row m = l % MT and k-lane kl = l / MT,
 * a chunk is CHUNK = 2048/MT codes, and each lane takes 64 of them, the chunk's 32-blocks 2*kl and
 * 2*kl+1. Each lane sums its two blocks in k and scales each by its e8; one shfl_xor pair puts the
 * group's four block terms in ascending order in LDS, and the kl == 0 lane adds them to the
 * accumulator in that order. This is the reference's ascending-k accumulation at the 32-block
 * granularity the format fixes.
 *
 * ---- refusals -----------------------------------------------------------------------------------------
 *
 * The shim refuses every structural violation: the dtypes of all seven operands (codes as
 * fp4_e2m1 or raw bytes, the e8 plane as e8m0 or u8, `b_ref` the e8 plane again, ignored), K % 32
 * and K % 128, the codes plane's extent and 16-byte row alignment, the e8 plane's [N, K/32]
 * extent, the pitches and the band's M. A reserved e8 (0, 255) is refused with E_INVAL where the
 * host can read the plane: the shim scans it once per (pointer, size) pair. A plane the host
 * cannot see is not scanned; the kernel gives reserved values the scale zero, so the output stays
 * finite.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_fp8_dequant.h"
#include "r3d_tiling.h"
#include "r3d_launch.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWaves = 8;
constexpr int kTM = 128;                     /* rows per tile */
constexpr int kTN = 128;                     /* columns per tile */
constexpr int kKC = 32;                      /* k per chunk: one OCP 32-block */
constexpr int kMX = 32;                      /* K per e8m0 exponent: the OCP MX block */
constexpr int kGroup = 128;                  /* the activation's scale group */
constexpr int kMMin = 17;                    /* the tiled row's floor: the m16 row's ceiling plus one */
constexpr int kMTMax = 16;
constexpr int kTileQuads = 512;              /* MT * CHUNK/4, constant over MT: 2048 codes, 4 KiB of f16 */

/* The eight e2m1 magnitudes as e4m3 bytes, each exact: 0, .5, 1, 1.5 in the low word, 2, 3, 4, 6
 * in the high. */
#define R3D_MXFP4_MAG_LO 0x3c383000u
#define R3D_MXFP4_MAG_HI 0x4c484440u

/* Eight e2m1 codes (four packed bytes, element 2j in the low nibble of byte j) as eight e4m3
 * bytes in k order (word 0 = elements 0..3, word 1 = 4..7). Even and odd nibbles each go through
 * the magnitude bytes with a v_perm_b32 on the 3-bit magnitude, bit 3 of the code shifts onto the
 * e4m3 sign, and two more v_perm_b32 interleave the halves. */
__device__ __forceinline__ void mxfp4_unpack8(uint32_t wv, uint32_t* lo, uint32_t* hi)
{
    const uint32_t ev = wv & 0x0F0F0F0Fu;              /* low nibble of each byte -> even elements */
    const uint32_t od = (wv >> 4) & 0x0F0F0F0Fu;       /* high nibble              -> odd elements  */
    const uint32_t be = __builtin_amdgcn_perm(R3D_MXFP4_MAG_HI, R3D_MXFP4_MAG_LO, ev & 0x07070707u)
                        | ((ev & 0x08080808u) << 4);
    const uint32_t bo = __builtin_amdgcn_perm(R3D_MXFP4_MAG_HI, R3D_MXFP4_MAG_LO, od & 0x07070707u)
                        | ((od & 0x08080808u) << 4);
    *lo = __builtin_amdgcn_perm(bo, be, 0x05010400u);
    *hi = __builtin_amdgcn_perm(bo, be, 0x07030602u);
}

/* The e8m0 factor: 2^(e-127). The reserved values (e = 0 the format's zero, e = 255 its NaN) take
 * the scale zero, so such a block computes zero and stays finite; the shim refuses such a plane
 * where it can read it. */
__device__ __forceinline__ float e8_scale(uint8_t e)
{
    return (e == 0 || e == 255) ? 0.0f : ldexpf(1.0f, (int)e - 127);
}

/* ================================================================== the tiled row */

struct ActQ {
    const float* scale;          /* a_scale, f32: [M][ceil(K/group)] */
    long long row;               /* elements a row */
    int gmode;                   /* 1: one scale per (row, group) */
};

struct MxWQ {
    const uint8_t* codes;        /* e2m1, two per byte, [N][K/2] */
    const uint8_t* e8;           /* e8m0, one per 32, [N][K/32] */
    long long code_row;          /* bytes a codes row */
    long long e8_row;            /* bytes an e8 row */
};

/*
 * The tile: 128 rows x 128 columns of y, 256 threads, eight waves, 8x8 outputs per lane (64 f32
 * accumulators for the 32-block partials and 64 for the running sum), grid (N/kTN) x ceil(M/kTM).
 * k is walked in chunks of 32. The activation's E4M3 codes decode through the block's table and
 * are staged as f16 with their scale folded in. The weight's e2m1 nibbles unpack to e4m3 and are
 * staged as f16 without the block scale, which is applied to the partials at the chunk's end.
 */
template <int NT, int ALU>   /* ALU: 1 decode in registers, 0 through the LDS table */
__global__ void __launch_bounds__(kBlockThreads) R3D_WAVES(2)
mxfp4a8_tiled_kernel(const uint8_t* __restrict__ a, ActQ aq, MxWQ h, void* __restrict__ y,
                     int M, int N, int K, int a_ld, int y_ld)
{
    __shared__ __align__(16) uint16_t sA[kTM * kKC];
    __shared__ __align__(16) uint16_t sB[kTN * kKC];
    __shared__ __align__(16) uint16_t sTbl[R3D_E4M3_TBL];
    __shared__ uint8_t sE8[kTN];         /* the chunk's e8, one per tile row */

    const int tid = (int)threadIdx.x;
    if (!ALU) r3d_e4m3_f16_init(sTbl, tid, kBlockThreads);

    const int m0 = (int)blockIdx.y * kTM;
    const int n0 = (int)blockIdx.x * kTN;
    const int rg = tid >> 4, cg = tid & 15;

    float acc[8][8];
    float total[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) { acc[i][j] = 0.0f; total[i][j] = 0.0f; }

    const int m_r = m0 + rg * 8, n_c = n0 + cg * 8;

    for (int k0 = 0; k0 < K; k0 += kKC) {
        const int g = k0 / kGroup;
        const int blk = k0 / kMX;        /* the chunk is one 32-block */

#pragma unroll
        for (int j = tid; j < kTM * (kKC / 8); j += kBlockThreads) {
            const int mm = j / (kKC / 8), c8 = j % (kKC / 8);
            const int m = m0 + mm, k = k0 + c8 * 8;
            uint4 staged = make_uint4(0, 0, 0, 0);
            if (m < M) {
                const float sa = aq.scale[(long long)m * aq.row + (aq.gmode ? g : 0)];
                uint32_t w[4];
                r3d_e4m3x8_decode<ALU>(sTbl, r3d_loadc8<NT>(a + (long long)m * a_ld + k), w);
                staged = make_uint4(r3d_fold_pair(w[0], sa), r3d_fold_pair(w[1], sa),
                                    r3d_fold_pair(w[2], sa), r3d_fold_pair(w[3], sa));
            }
            *(uint4*)&sA[srow(mm, c8, kKC)] = staged;
        }
        for (int nn = tid; nn < kTN; nn += kBlockThreads) {
            const int n = n0 + nn;
            uint4 st[4] = { make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0),
                            make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0) };
            /* The column's e8, written even past N (as zero), so the flush multiplies a staged
             * zero by a defined value. */
            sE8[nn] = (n < N) ? h.e8[(long long)n * h.e8_row + blk] : 0;
            if (n < N) {
                /* 32 codes, 16 bytes, one load, aligned because k0 is a multiple of 32 and the
                 * row pitch a multiple of 16 bytes. */
                uint4 cv;
                __builtin_memcpy(&cv, h.codes + (long long)n * h.code_row + (k0 >> 1), 16);
                const uint32_t d[4] = { cv.x, cv.y, cv.z, cv.w };
                uint32_t w[16];
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    uint32_t lo, hi;
                    mxfp4_unpack8(d[q], &lo, &hi);
                    r3d_e4m3x4_decode<ALU>(sTbl, lo, w + 4 * q);
                    r3d_e4m3x4_decode<ALU>(sTbl, hi, w + 4 * q + 2);
                }
                st[0] = make_uint4(w[0], w[1], w[2], w[3]);
                st[1] = make_uint4(w[4], w[5], w[6], w[7]);
                st[2] = make_uint4(w[8], w[9], w[10], w[11]);
                st[3] = make_uint4(w[12], w[13], w[14], w[15]);
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) *(uint4*)&sB[srow(nn, q, kKC)] = st[q];
        }
        __syncthreads();

        /* The chunk's e8 for the lane's eight columns, in registers for the flush. sE8 holds
         * this tile's 128 columns, so the index is tile-local. */
        float e8s[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) e8s[j] = e8_scale(sE8[cg * 8 + j]);

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
        /* The 32-block boundary: scale the partial by the block's factor, add it to the running
         * sum in ascending k, reset the partial. */
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                total[i][j] += acc[i][j] * e8s[j];
                acc[i][j] = 0.0f;
            }
        __syncthreads();
    }

    /* y is bf16: one narrowing, 16-byte runs of eight columns. Rows past M write nothing. */
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int m = m_r + i;
        if (m >= M) break;
        uint16_t* row = (uint16_t*)y + (long long)m * y_ld;
        if (n_c + 8 <= N) {
            uint32_t p[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                p[j] = (uint32_t)f32_to_bf16(total[i][2 * j]) |
                       ((uint32_t)f32_to_bf16(total[i][2 * j + 1]) << 16);
            *(uint4*)(row + n_c) = make_uint4(p[0], p[1], p[2], p[3]);
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (n_c + j < N) row[n_c + j] = f32_to_bf16(total[i][j]);
        }
    }
}

/* ================================================================== the m16 row */

/* The LDS swizzle, at the tile's access granularity: a quad of four codes (two dwords) moved as one
 * 8-byte word. Quads are numbered q = r*(CHUNK/4) + c; p_ix XORs q's low four bits with bits 4..7,
 * a bijection on 16-quad groups. */
__device__ __forceinline__ int p_ix(int q) { return q ^ ((q >> 4) & 15); }

template <int MT, int ALU>   /* ALU: 1 decode in registers, 0 through the LDS table */
__global__ void __launch_bounds__(kBlockThreads) R3D_WAVES(2)
gemm_mxfp4a8_m16_kernel(const uint8_t* __restrict__ a, const float* __restrict__ sa, int sa_stride,
                        const uint8_t* __restrict__ b, const uint8_t* __restrict__ e8,
                        int e8_row, uint16_t* __restrict__ y, int M, int N, int K, int chunk,
                        int a_pitch, int y_pitch)
{
    constexpr int KL = kWave / MT;                    /* k-lanes per activation row */
    constexpr int CL = 64;                            /* codes per lane per chunk: two 32-blocks */
    constexpr int BPR = (2048 / MT) / kMX;            /* 32-blocks per chunk row: 64/MT */

    __shared__ __align__(16) uint16_t sTbl[R3D_E4M3_TBL];
    __shared__ __align__(16) uint2 sA[kTileQuads];    /* MT rows x CHUNK/2 f16, swizzled */
    __shared__ float sB[kWaves * 64];                 /* the chunk's block terms: warp*64 + m*BPR + bl */

    const int lane = threadIdx.x & (kWave - 1);
    const int warp = (int)(threadIdx.x / kWave);
    const int n = blockIdx.x * kWaves + warp;
    const bool live = n < N;
    if ((int)blockIdx.x * kWaves >= N) return;

    const int m = lane & (MT - 1);
    const int kl = lane / MT;
    const int nchunk = K / chunk;

    if (!ALU) r3d_e4m3_f16_init(sTbl, threadIdx.x, kBlockThreads);

    float acc = 0.0f;
    for (int ch = 0; ch < nchunk; ++ch) {
        __syncthreads();
        /* The chunk's activation rows, decoded and scaled on the way in. Cells are four codes:
         * MT * CHUNK/4 = 512 at every MT, two per thread. A cell's four codes sit in one 128-wide
         * scale group, so one scale per cell. */
        for (int i = threadIdx.x; i < MT * (chunk >> 2); i += kBlockThreads) {
            const int r = i / (chunk >> 2), c = i % (chunk >> 2);
            uint32_t lo = 0, hi = 0;
            if (r < M) {
                const int64_t k0 = (int64_t)ch * (chunk >> 2) * 4 + c * 4;
                const uint32_t w = *(const uint32_t*)(a + (long long)r * a_pitch + k0);
                const float s = sa[(long long)r * sa_stride + k0 / kGroup];
                uint32_t w2[2];
                r3d_e4m3x4_decode<ALU>(sTbl, w, w2);
                lo = r3d_fold_pair(w2[0], s);
                hi = r3d_fold_pair(w2[1], s);
            }
            const int q = r * (chunk >> 2) + c;
            sA[p_ix(q)] = make_uint2(lo, hi);
        }
        __syncthreads();

        if (live) {
            /* 64 codes, 32 bytes: two 16-byte loads, aligned by the row pitch and k0 % 32 == 0. */
            const uint8_t* bp = b + (long long)n * (K >> 1) + (int64_t)ch * (chunk >> 1)
                                + kl * (CL >> 1);
            uint4 qw0 = make_uint4(0, 0, 0, 0), qw1 = make_uint4(0, 0, 0, 0);
            __builtin_memcpy(&qw0, bp, 16);
            __builtin_memcpy(&qw1, bp + 16, 16);
            const uint32_t w8[8] = { qw0.x, qw0.y, qw0.z, qw0.w, qw1.x, qw1.y, qw1.z, qw1.w };

            const int qb = m * (chunk >> 2) + kl * (CL >> 2);
            float p0 = 0.0f, p1 = 0.0f;               /* the lane's two 32-block partials */
#pragma unroll
            for (int v = 0; v < 8; ++v) {             /* dwords 0..3: block 2*kl, 4..7: 2*kl+1 */
                uint32_t elo, ehi;
                mxfp4_unpack8(w8[v], &elo, &ehi);
                float& p = (v < 4) ? p0 : p1;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    uint32_t w2[2];
                    r3d_e4m3x4_decode<ALU>(sTbl, e ? ehi : elo, w2);
                    const uint2 av = sA[p_ix(qb + 2 * v + e)];
                    p = dot4_f16x4(av.x, av.y, w2[0], w2[1], p);
                }
            }

            /* The lane's two blocks, scaled by their e8. The pair (kl, kl^1) owns the group's
             * four blocks (kl even the lower two, kl odd the upper two), so one exchange puts
             * them in ascending block order. */
            const int blk0 = ch * (chunk / kMX) + 2 * kl;
            const float q0f = p0 * e8_scale(e8[(long long)n * e8_row + blk0]);
            const float q1f = p1 * e8_scale(e8[(long long)n * e8_row + blk0 + 1]);
            const float o0f = __shfl_xor(q0f, MT);
            const float o1f = __shfl_xor(q1f, MT);
            const bool lo = !(kl & 1);
            const float t0 = lo ? q0f : o0f;
            const float t1 = lo ? q1f : o1f;
            const float t2 = lo ? o0f : q0f;
            const float t3 = lo ? o1f : q1f;

            /* One slot per (row, 32-block-in-chunk); MT*BPR is 64 at every MT. Both half-lanes
             * of a group write the same four values. */
            const int j = kl >> 1;
            const int bb = warp * 64 + m * BPR + 4 * j;
            sB[bb + 0] = t0;
            sB[bb + 1] = t1;
            sB[bb + 2] = t2;
            sB[bb + 3] = t3;
        }
        __syncthreads();
        /* The chunk's block terms in ascending block order, into the row's owner lane: f32 adds
         * are not associative and the reference walks k ascending. */
        if (live && kl == 0) {
#pragma unroll
            for (int bl = 0; bl < BPR; ++bl)
                if (m < M) acc += sB[warp * 64 + m * BPR + bl];
        }
        __syncthreads();
    }
    if (live && kl == 0 && m < M) y[(long long)m * y_pitch + n] = f32_to_bf16(acc);
}

/* ================================================================== the launchers */

/* The tile's rows: the power of two that holds M, widened while its chunk (2048 / mt codes) does
 * not divide K. The extra rows are idle. At 16 rows the chunk is 128, which the row's constraints
 * (K a multiple of 128) always divide. */
static int mt_for(int M, int K)
{
    int mt = 1;
    while ((mt < M || K % (2048 / mt)) && mt < kMTMax) mt *= 2;
    return mt;
}

extern "C" int r3d_gemm_nt_q_mxfp4a8_m16(long a, long a_scale, long b, long b_scale, long y,
                                         int M, int N, int K, int group, int a_ld, int y_ld,
                                         long long e8_row, long stream, int alu = 1)
{
    if (!a || !a_scale || !b || !b_scale || !y) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (M > kMTMax) return -2;
    if (group != kGroup) return -2;
    if (K % kGroup) return -2;                /* the activation's scale group */
    if (K % kMX) return -2;                   /* the weight's 32-block */
    const int mt = mt_for(M, K);
    const int chunk = 2048 / mt;
    if (K % chunk) return -2;
    if (!a_ld) a_ld = K;
    if (!y_ld) y_ld = N;
    if (a_ld < K) return -3;
    if (a_ld % 4) return -3;                  /* four codes per dword, staged whole */
    if (y_ld < N) return -3;
    if (!e8_row) e8_row = K / kMX;
    if (e8_row < K / kMX) return -3;
    if (((uintptr_t)a & 3u) || ((uintptr_t)b & 15u)) return -5;

    const dim3 grid((N + kWaves - 1) / kWaves), block(kBlockThreads);
 #define R3D_MXFP4A8_M16(MT)                                                                          \
    do {                                                                                             \
        if (alu)                                                                                     \
            R3D_KLAUNCH(R3D_KNAME((gemm_mxfp4a8_m16_kernel<MT, 1>)), grid, block, (hipStream_t)stream, \
                        (const uint8_t*)a, (const float*)a_scale, K / kGroup, (const uint8_t*)b,      \
                        (const uint8_t*)b_scale, (int)e8_row, (uint16_t*)y, M, N, K, chunk, a_ld,      \
                        y_ld);                                                                        \
        else                                                                                         \
            R3D_KLAUNCH(R3D_KNAME((gemm_mxfp4a8_m16_kernel<MT, 0>)), grid, block, (hipStream_t)stream, \
                        (const uint8_t*)a, (const float*)a_scale, K / kGroup, (const uint8_t*)b,      \
                        (const uint8_t*)b_scale, (int)e8_row, (uint16_t*)y, M, N, K, chunk, a_ld,      \
                        y_ld);                                                                        \
    } while (0)
    switch (mt) {
        case 1:  R3D_MXFP4A8_M16(1);  break;
        case 2:  R3D_MXFP4A8_M16(2);  break;
        case 4:  R3D_MXFP4A8_M16(4);  break;
        case 8:  R3D_MXFP4A8_M16(8);  break;
        default: R3D_MXFP4A8_M16(16); break;
    }
#undef R3D_MXFP4A8_M16
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

#define L_MX(why, a1, v1, a2, v2)                                                    \
    do {                                                                              \
        if (r3d_trace_on())                                                           \
            std::fprintf(stderr, "[r3d] %s:%d %s: %s = %lld, %s = %lld\n",           \
                         __FILE__, __LINE__, (why), (a1), (long long)(v1), (a2),     \
                         (long long)(v2));                                            \
        return R3D_NO(-3);                                                            \
    } while (0)

int launch_mxfp4a8_tiled(long av, long asv, long bvc, long bsv, long yv, int M, int N, int K,
                         int group, int a_ld, long long b_row, int as_ld, long long e8_row,
                         int y_ld, int nt, long stream, int alu = 1)
{
    if (!av || !asv || !bvc || !bsv || !yv) return -1;
    if (M < kMMin || N < 1 || K < 1) return -2;
    if (group != kGroup) return -2;              /* kKC 32 divides the group 128 */
    if (K % kGroup) return -2;                   /* a ragged k-block reads past the scale planes */
    if (K > (1 << 24)) return -2;
    if (nt < 0 || nt > 1) return -2;
    if (a_ld % 8) L_MX("a pitch: 8-byte code loads, 8 codes a lane", "a_ld", a_ld, "K", K);
    if (b_row & 15) L_MX("b row pitch: 16-byte code loads", "b_row", b_row, "K", K);
    if (as_ld < (K + kGroup - 1) / kGroup) L_MX(
            "a_scale: one per (row, group)", "as_ld", as_ld, "ceil(K/group)",
            (K + kGroup - 1) / kGroup);
    if (y_ld % 8) L_MX("y pitch: 16-byte bf16 store runs", "y_ld", y_ld, "N", N);
    if (a_ld < K) L_MX("a pitch shorter than K", "a_ld", a_ld, "K", K);
    if (!b_row) b_row = K >> 1;
    if (b_row < (K >> 1)) L_MX("b row pitch shorter than K/2", "b_row", b_row, "K", K);
    if (!e8_row) e8_row = K / kMX;
    if (e8_row < K / kMX) L_MX("e8 row pitch shorter than K/32", "e8_row", e8_row, "K", K);
    if (y_ld < N) L_MX("y pitch shorter than N", "y_ld", y_ld, "N", N);
    if ((av & 7) || (asv & 3) || (bvc & 15) || (bsv & 1) || (yv & 7)) return -5;

    const dim3 blk(kBlockThreads);
    const dim3 grd((unsigned)((N + kTN - 1) / kTN), (unsigned)((M + kTM - 1) / kTM));
    /* Value-init so the ActQ's 4-byte padding hole is deterministic: the tape audit byte-compares
     * the serialized kernel arguments, and aggregate-init leaves that padding indeterminate, which
     * reads as "the key does not carry" the argument and stops the engine. See fp8a8's launcher. */
    ActQ aq{};
    aq.scale = (const float*)(uintptr_t)asv;
    aq.row = as_ld;
    aq.gmode = 1;
    MxWQ h{};
    h.codes = (const uint8_t*)(uintptr_t)bvc;
    h.e8 = (const uint8_t*)(uintptr_t)bsv;
    h.code_row = b_row;
    h.e8_row = e8_row;
    hipStream_t s = (hipStream_t)(uintptr_t)stream;
    if (alu) {
        if (nt) mxfp4a8_tiled_kernel<1,1><<<grd, blk, 0, s>>>(
            (const uint8_t*)(uintptr_t)av, aq, h, (void*)(uintptr_t)yv, M, N, K, a_ld, y_ld);
        else mxfp4a8_tiled_kernel<0,1><<<grd, blk, 0, s>>>(
            (const uint8_t*)(uintptr_t)av, aq, h, (void*)(uintptr_t)yv, M, N, K, a_ld, y_ld);
    } else {
        if (nt) mxfp4a8_tiled_kernel<1,0><<<grd, blk, 0, s>>>(
            (const uint8_t*)(uintptr_t)av, aq, h, (void*)(uintptr_t)yv, M, N, K, a_ld, y_ld);
        else mxfp4a8_tiled_kernel<0,0><<<grd, blk, 0, s>>>(
            (const uint8_t*)(uintptr_t)av, aq, h, (void*)(uintptr_t)yv, M, N, K, a_ld, y_ld);
    }
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_gemm_nt_q_mxfp4a8_tiled(long a, long a_scale, long b_codes, long b_scale, long y,
                                           int M, int N, int K, int group, int a_ld,
                                           long long b_row, int as_ld, long long e8_row,
                                           int y_ld, int nt, long stream, int alu)
{
    return launch_mxfp4a8_tiled(a, a_scale, b_codes, b_scale, y, M, N, K, group, a_ld, b_row,
                                as_ld, e8_row, y_ld, nt, stream, alu);
}

/* ================================================================== the shims
 *
 * gemm_nt_q's slots: 0 a (E4M3), 1 a_scale (f32, one per (row, 128 of K)), 2 b (the e2m1 codes,
 * as fp4_e2m1 or raw u8 bytes, two per byte), 3 b_scale (the e8 plane, e8m0 or u8, one per 32 of
 * K), 4 y (bf16), 5 a_sum (absent: symmetric), 6 b_ref (the e8 plane again, accepted and
 * ignored).
 */

/* The reserved-e8 scan, once per (pointer, size) pair for the process's life: the plane is a
 * weight at a stable address. The host must be able to read the plane (the mapped tier's
 * placement); a plane it cannot see is not scanned, and the kernel's e8 guard handles reserved
 * values there. */
struct E8ScanKey { const void* p; long long n; };
static bool e8_scan_once(const void* p, long long n)
{
    static E8ScanKey seen[16];
    static int ns = 0;
    for (int i = 0; i < ns; ++i)
        if (seen[i].p == p && seen[i].n == n) return true;
    if (ns < 16) seen[ns++] = E8ScanKey{ p, n };
    void* dev = nullptr;
    if (hipHostGetDevicePointer(&dev, (void*)p, 0) != hipSuccess) return true;
    const uint8_t* e8 = (const uint8_t*)p;
    for (long long i = 0; i < n; ++i)
        if (e8[i] == 0 || e8[i] == 255) return false;
    return true;
}

static int r3d_mxfp4a8_m16_dispatch(const RadArgs* a, RadStream s, int alu)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const char* dt = r3d_gets(a, "dtype");
    if (dt && std::strcmp(dt, "mxfp4a8") != 0) return R3D_NO(RAD_E_DTYPE);

    const RadTensor* A = r3d_opt(a, 0);
    const RadTensor* AS = r3d_opt(a, 1);
    const RadTensor* B = r3d_opt(a, 2);
    const RadTensor* BS = r3d_opt(a, 3);
    const RadTensor* Y = r3d_opt(a, 4);
    const RadTensor* ASUM = r3d_opt(a, 5);
    const RadTensor* BREF = r3d_opt(a, 6);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(BS) || !r3d_p(Y))
        return R3D_NO(RAD_E_INVAL);
    if (r3d_has(ASUM)) return R3D_NO(RAD_E_DTYPE);      /* symmetric: no zero point */
    /* `b_ref` is the reference's per-row exponent, derived from `b_scale`. It changes no product,
     * so with an e2m1 weight only its dtype is checked. */
    if (r3d_has(BREF) && BREF->dtype != RAD_E8M0 && BREF->dtype != RAD_U8)
        return R3D_NO(RAD_E_DTYPE);

    if (!r3d_is_e4m3(A)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(AS)) return R3D_NO(RAD_E_DTYPE);
    if (B->dtype != RAD_FP4E2M1 && B->dtype != RAD_U8) return R3D_NO(RAD_E_DTYPE);  /* e2m1, 2 per byte */
    if (BS->dtype != RAD_E8M0 && BS->dtype != RAD_U8) return R3D_NO(RAD_E_DTYPE);   /* e8m0, one per 32 */
    if (!r3d_is_bf16(Y)) return R3D_NO(RAD_E_DTYPE);

    const long long N = r3d_geti_or(a, "N", 0), K = r3d_geti_or(a, "K", 0);
    /* The `group` key names the band; the kernel's blocks (128 for the activation scale, 32 for
     * the weight's e8) are fixed by the format, so the key is not checked. */
    if (N < 1 || K < 1) return R3D_NO(RAD_E_SHAPE);
    if (K % kGroup || K % kMX) return R3D_NO(RAD_E_SHAPE);

    const long long ynel = r3d_numel(Y);
    if (ynel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = ynel / N;
    if (M < 1 || M > kMTMax) return R3D_NO(RAD_E_SHAPE);

    if (r3d_numel(A) < M * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(AS) < M * (K / kGroup)) return R3D_NO(RAD_E_SHAPE);
    const long long bbytes = B->dtype == RAD_FP4E2M1 ? r3d_numel(B) / 2 : r3d_numel(B);
    if (bbytes < N * (K / 2)) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(BS) < N * (K / kMX)) return R3D_NO(RAD_E_SHAPE);

    const long long ap = r3d_row_pitch(A);
    if (ap < K || ap % 4) return R3D_NO(RAD_E_STRIDE);
    const long long as_p = r3d_row_pitch(AS);
    if (as_p < K / kGroup) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_contig(BS)) return R3D_NO(RAD_E_STRIDE);
    long long yrows = 0;
    const long long yp = r3d_rows_pitch(Y, N, &yrows);
    if (yp < N) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_aligned(r3d_p(A), 4) || !r3d_aligned(r3d_p(B), 16)) return R3D_NO(RAD_E_ALIGN);

    if (!e8_scan_once(r3d_p(BS), N * (K / kMX))) return R3D_NO(RAD_E_INVAL);

    const int rc = r3d_gemm_nt_q_mxfp4a8_m16(r3d_addr(A), r3d_addr(AS), r3d_addr(B), r3d_addr(BS),
                                             r3d_addr(Y), (int)M, (int)N, (int)K, kGroup,
                                             (int)ap, (int)yp, (long long)r3d_row_pitch(BS),
                                             (long)s, alu);
    if (rc == -1) return R3D_NO(RAD_E_INVAL);
    if (rc == -2) return R3D_NO(RAD_E_SHAPE);
    if (rc == -3) return R3D_NO(RAD_E_STRIDE);
    if (rc == -5) return R3D_NO(RAD_E_ALIGN);
    if (rc == -4) return R3D_NO(RAD_E_DEVICE);
    return RAD_OK;
}

static int r3d_mxfp4a8_tiled_dispatch(const RadArgs* a, RadStream s, int alu)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const char* dt = r3d_gets(a, "dtype");
    if (dt && std::strcmp(dt, "mxfp4a8") != 0) return R3D_NO(RAD_E_DTYPE);

    const RadTensor* A = r3d_opt(a, 0);
    const RadTensor* AS = r3d_opt(a, 1);
    const RadTensor* B = r3d_opt(a, 2);
    const RadTensor* BS = r3d_opt(a, 3);
    const RadTensor* Y = r3d_opt(a, 4);
    const RadTensor* ASUM = r3d_opt(a, 5);
    const RadTensor* BREF = r3d_opt(a, 6);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(BS) || !r3d_p(Y))
        return R3D_NO(RAD_E_INVAL);
    if (r3d_has(ASUM)) return R3D_NO(RAD_E_DTYPE);      /* symmetric: no zero point */
    if (r3d_has(BREF) && BREF->dtype != RAD_E8M0 && BREF->dtype != RAD_U8)
        return R3D_NO(RAD_E_DTYPE);

    if (!r3d_is_e4m3(A)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(AS)) return R3D_NO(RAD_E_DTYPE);
    if (B->dtype != RAD_FP4E2M1 && B->dtype != RAD_U8) return R3D_NO(RAD_E_DTYPE);
    if (BS->dtype != RAD_E8M0 && BS->dtype != RAD_U8) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_bf16(Y)) return R3D_NO(RAD_E_DTYPE);

    /* The band's keys from the schema: N, K, group. A band without N or K takes them from the
     * codes plane. */
    const long long M_band = r3d_geti_or(a, "M", 0);
    const long long N0 = r3d_geti_or(a, "N", 0), K0 = r3d_geti_or(a, "K", 0);
    const long long N = N0 > 0 ? N0 : r3d_dim(B, 0);
    /* Without a K key, K comes from the codes plane: [N, K] of fp4_e2m1, [N, K/2] of raw bytes.
     * The `group` key is not checked (see the m16 shim). */
    const long long K = K0 > 0 ? K0
                    : (B->dtype == RAD_FP4E2M1 ? r3d_dim(B, 1) : 2 * r3d_dim(B, 1));
    if (N <= 0 || K <= 0) return R3D_NO(RAD_E_SHAPE);
    if (K > (1 << 24)) return R3D_NO(RAD_E_INVAL);
    if (K % kGroup || K % kMX) return R3D_NO(RAD_E_SHAPE);

    long long a_rows = 0;
    const long long ap = r3d_rows_pitch(A, K, &a_rows);
    if (ap < K || a_rows < 1) return R3D_NO(RAD_E_STRIDE);
    if (ap % 8) return R3D_NO(RAD_E_STRIDE);          /* 8-byte code loads, 8 codes per lane */
    /* The step's M is the activation's row count; the band key M is only the upper bound. */
    const long long M = a_rows;
    if (M_band > 0 && M > M_band) return R3D_NO(RAD_E_SHAPE);
    if (M < kMMin) return R3D_NO(RAD_E_UNSUPPORTED);   /* the m16 row serves M <= 16 */

    const long long ynel = r3d_numel(Y);
    if (ynel % N) return R3D_NO(RAD_E_SHAPE);
    if (ynel / N < M) return R3D_NO(RAD_E_SHAPE);

    if (r3d_numel(A) < M * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(AS) < M * (K / kGroup)) return R3D_NO(RAD_E_SHAPE);
    const long long bbytes = B->dtype == RAD_FP4E2M1 ? r3d_numel(B) / 2 : r3d_numel(B);
    if (bbytes < N * (K / 2)) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(BS) < N * (K / kMX)) return R3D_NO(RAD_E_SHAPE);

    const long long as_p = r3d_row_pitch(AS);
    if (as_p < K / kGroup) return R3D_NO(RAD_E_STRIDE);
    /* The codes plane's pitch in bytes: fp4_e2m1 counts elements, two per byte. */
    const long long cp0 = r3d_row_pitch(B);
    const long long cp = B->dtype == RAD_FP4E2M1 ? cp0 / 2 : cp0;
    if (cp < K / 2) return R3D_NO(RAD_E_STRIDE);
    if (cp & 15) return R3D_NO(RAD_E_ALIGN);
    const long long e8p = r3d_row_pitch(BS);
    if (e8p < K / kMX) return R3D_NO(RAD_E_STRIDE);
    long long yrows = 0;
    const long long yp = r3d_rows_pitch(Y, N, &yrows);
    if (yp < N) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_aligned(r3d_p(A), 8) || !r3d_aligned(r3d_p(AS), 4) ||
        !r3d_aligned(r3d_p(B), 16) || !r3d_aligned(r3d_p(Y), 8))
        return R3D_NO(RAD_E_ALIGN);

    if (!e8_scan_once(r3d_p(BS), N * (K / kMX))) return R3D_NO(RAD_E_INVAL);

    const int nt = 0;   /* no `nt` axis in the row's tune table: the non-temporal stream is fixed off */
    const int rc = r3d_gemm_nt_q_mxfp4a8_tiled(r3d_addr(A), r3d_addr(AS), r3d_addr(B),
                                               r3d_addr(BS), r3d_addr(Y), (int)M, (int)N,
                                               (int)K, kGroup, (int)ap, cp, (int)as_p, e8p,
                                               (int)yp, nt, (long)s, alu);
    if (rc == -1) return R3D_NO(RAD_E_INVAL);
    if (rc == -2) return R3D_NO(RAD_E_SHAPE);
    if (rc == -3) return R3D_NO(RAD_E_STRIDE);
    if (rc == -5) return R3D_NO(RAD_E_ALIGN);
    if (rc == -4) return R3D_NO(RAD_E_DEVICE);
    return RAD_OK;
}

/* The rows' impl axis picks the E4M3 decode: the LDS table (0) and the per-byte ALU formula (1).
 * The _alu entry points stay for a checker that names them directly. */
extern "C" int r3d_rad_gemm_nt_q_mxfp4a8_m16(const RadArgs* a, RadStream s)
{
    R3dGemmCfg c;
    r3d_cfg_gemm(a, &c);
    return r3d_mxfp4a8_m16_dispatch(a, s, c.impl);
}
extern "C" int r3d_rad_gemm_nt_q_mxfp4a8_m16_alu(const RadArgs* a, RadStream s)
{ return r3d_mxfp4a8_m16_dispatch(a, s, 1); }
extern "C" int r3d_rad_gemm_nt_q_mxfp4a8_tiled(const RadArgs* a, RadStream s)
{
    R3dGemmCfg c;
    r3d_cfg_gemm(a, &c);
    return r3d_mxfp4a8_tiled_dispatch(a, s, c.impl);
}
extern "C" int r3d_rad_gemm_nt_q_mxfp4a8_tiled_alu(const RadArgs* a, RadStream s)
{ return r3d_mxfp4a8_tiled_dispatch(a, s, 1); }

}  // namespace
