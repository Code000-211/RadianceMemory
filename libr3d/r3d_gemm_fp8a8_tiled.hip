/* r3d_gemm_fp8a8_tiled.hip -- gemm_nt_q at prefill M, fp8a8, on a tiled f16 dot.
 *
 * gemm_nt_q needs a device row at prefill M. Without one its bands resolve to a host kernel, which
 * keeps the layer's weights (norms included) in the container's mapping, where the mapped tier
 * cannot widen the norms' stored form.
 *
 * Bands: dtype in {fp8a8}, group == 128, M >= 17, K % 128 == 0. The target model's prefill bands
 * are M=2048 with (N, K) in {(12288, 2048), (2560, 2048), (2048, 6144), (2048, 2048), (2048,
 * 10240)}. M <= 16 is gemm_nt_q_fp8a8_m16's; the floor `kMMin` is that row's ceiling plus one.
 *
 * ---- the law ------------------------------------------------------------------------------------
 *
 * gemm_nt_q's registry law: the group sub-sum is accumulated from the raw codes and scaled once
 * per group. The reference's quant_dot (also in r3d_gemm_fp8a8_gated.hip and
 * r3d_gemm_fp8a16_nt_m1.hip):
 *
 *     for g in groups:  sub = 0
 *                       for k in g, ascending:  sub = sub + code_a[m][k] * code_b[n][k]
 *                       acc = acc + (sub * a_scale[m][g]) * b_scale[n/128][g]
 *
 * Codes are raw E4M3 values widened exactly to f32. Operands (gemm_q_shape, schema order): a E4M3
 * {M,K}; a_scale F32 {M, ceil(K/group)}; b E4M3 {N,K}, the checkpoint's bytes; b_scale BF16
 * {ceil(N/group), ceil(K/group)} (f32 at fp8a8f32); y BF16 {M,N}; a_sum absent (fp8a8 is
 * symmetric); b_ref absent. f32 accumulate, one bf16 narrowing at the store.
 *
 * ---- the fold -----------------------------------------------------------------------------------
 *
 * The tile is f16, so both scales enter as f16 multiplies at staging: `a_scale[m][g]` per staged
 * row, `b_scale[nb][g]` once a chunk (TN = 128 is one scale row, and KC = 32 divides 128, so a
 * chunk sits inside one k-block). (a*sa)*(b*sb) = (a*b)*sa*sb, so the accumulator stays one f32
 * per output, 64 a lane. A per-group partial scaled into the accumulator would need 64 more
 * registers on an arm at 237 VGPRs, and spills.
 *
 * The fold uses v_mul_f16, never f16x2_pack (v_cvt_pkrtz_f16_f32, round toward zero, only for
 * values f16 represents exactly; r3d_codecs.h). Truncating folded products biases them toward
 * zero.
 *
 * The fold's cost against the reference is the two f16 roundings of the staged values, about
 * rel_l2 2.8e-4, 250x finer than the E4M3 quantum (2^-4).
 *
 * gfx11 has no fp8 -> f16 instruction (the clang builtin expands to software on gfx1102), so the
 * E4M3 decode uses r3d_fp8_dequant.h's 256-entry f16 LDS table.
 *
 * ---- tolerance ----------------------------------------------------------------------------------
 *
 * The fold's floor is 2^-11 per staged rounding, twice, on S = sqrt(sum_k (a_k b_k)^2), the
 * quadrature scale of the dequantised terms: 9.8e-4 S. At a bf16 output the store's own ulp
 * (3.9e-3 of the output's binade) is the larger bound. |y| is not a usable denominator, since a
 * sum of random products is often near zero, so the per-output gate is fold floor + store ulp
 * against S.
 *
 * ---- known performance cliff --------------------------------------------------------------------
 *
 * The grid is (ceil(N/128), ceil(M/128)) at one block a CU. Odd column-tile counts 19 and 21
 * (N=2432, N=2688) run 11x and 21x slower per block, deterministically and with correct results;
 * every band the target model issues has an even count. Untried remedies: a block-index swizzle,
 * then `nt` streaming loads.
 *
 * ---- placement ----------------------------------------------------------------------------------
 *
 * The shim reads the container header on the host and the kernel reads the same bytes on the
 * device, so operands must be managed (prefetched) or hipHostMalloc memory; plain malloc faults.
 *
 * ---- refusals -----------------------------------------------------------------------------------
 *
 * The shim: n_t < 5 (E_INVAL); a_sum present (E_DTYPE: fp8a8 is symmetric, the asymmetric grid is
 * r3d_gemm_nt_q_w2a8.hip's); a container with a zero plane, planes that are not codes + scale,
 * codes not tiled one a cell, sub-byte codes, e5m2 and other code dtypes, a table scale, an
 * integer scale (E_DTYPE); a scale block other than {group, group} (E_SHAPE); planes that do not
 * cover the band (E_SHAPE); a row_bytes shorter than the row (E_STRIDE); a not E4M3, a_scale not
 * f32, b not E4M3, b_scale not bf16/f16/f32, y not bf16 (E_DTYPE); M < 17 (E_UNSUPPORTED);
 * group != 128 (E_SHAPE: the fold indexes 128); K % 128 (E_SHAPE); pitches a vector load cannot
 * walk (E_STRIDE); addresses not 16-byte aligned (E_ALIGN); absurd shapes (E_INVAL).
 *
 * N and M need no divisibility: the last tile stages rows and columns past the edge as zeros and
 * guards its stores, so ragged shapes (M=65, 129, 258, 512) are served. K % 128 is the kernel's
 * rule: a 32-code chunk must sit inside one scale group, and a ragged K would read the scale
 * plane past its columns. The scale plane must be exactly ceil(N/128) x ceil(K/128) and the codes
 * N x K, as the reference's `scaled_rows` requires.
 *
 * The band's shape keys are M, N, K (gemm_q_shape, pGemmQ), not logits_gemm's n_vocab/n_embd.
 * When the band names no N or K, they come from the weight's dims, since b is [N, K].
 *
 * The launcher's private codes are -1 a null plane, -2 a shape, -3 a pitch, -4 HIP, -5 an address;
 * they collide numerically with the RAD codes, so the shim maps them.
 *
 * ---- code order ---------------------------------------------------------------------------------
 *
 * The codes plane is read row-major [N, K], as the container stores it and libref reads it. libr4d
 * keeps fp8a8 weights in WMMA fragment order (r4d.fp8.w.frag64), a different reading of the same
 * bytes, so no code is ever reordered here; the relayout writer below only assembles planes.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_fp8_dequant.h"
#include "r3d_tiling.h"          /* LDS tile addressing: srow, srow_pair */
#include "r3d_launch.h"
#include "rad_planes.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kBlockThreads = 256;
constexpr int kTM           = 128;                     /* rows a tile */
constexpr int kTN           = 128;                     /* columns a tile: one scale-row */
constexpr int kKC           = 32;                      /* k a chunk: divides the group */
constexpr int kGroup        = 128;                     /* R3D_FP8_BLOCK: the band's group */
/* One above gemm_nt_q_fp8a8_m16's declared ceiling (M <= 16). M=17..64 is a batched decode step,
 * and one host band pulls the whole model into one host run, so the floor stays low although the
 * 128-row tile wastes 2x at M=64 and 7.5x at M=17; that is still about two orders of magnitude
 * faster than the host loop. */
constexpr int kMMin         = 17;

/* Tile addressing (srow, srow_pair) is r3d_tiling.h's; 8-element granularity keeps the swizzle
 * uint4-aligned. */

/* A 16-byte load (32 U4 codes, or 16 E4M3 codes), memcpy'd in rather than cast, since an
 * unaligned cast is UB. Its 8-byte sibling u8v is declared in loadc8s. */
typedef uint32_t __attribute__((ext_vector_type(4))) u16v;   /* a 16-byte streamed load */

/* The two operands' scale planes, read the way the container hands them. */
struct ActQ {
    const float* scale;          /* a_scale, f32: [M][ceil(K/group)], or [M] with gmode 0 */
    long long row;               /* elements a row: ceil(K/group) in group mode, 1 in row mode */
    int gmode;                   /* 1: one scale per (row, group), `a_scale[m][g]`;
                                  * 0: one per row, `a_scale[m]` (the ZP_I8_ROW form) */
};

struct WQ {
    const uint8_t* codes;        /* b's codes plane, [N][K] */
    const uint8_t* scale;        /* b_scale's plane: the {128,128} block, or the {1,group} row */
    const uint8_t* zero;         /* the asym grid's zero plane, one per (row, group); null if none */
    long long code_row;          /* bytes a codes row */
    long long scale_row;         /* bytes a scale row */
    long long zero_row;          /* bytes a zero row */
    int sdt;                     /* 0 bf16, 1 f16, 2 f32 */
    int cdt;                     /* 0 E4M3, 1 int8, 2 U2, 3 U4 */
    int srows;                    /* 0: scale rows are 128-blocks (fp8a8); 1: scale rows are rows (w2a8, w4a8_asym) */
};

/* quant_act_i8's step order swaps bits 2 and 3 of a column, which are bits 0 and 1 of its dword
 * index: dword L of a row is stored at dword i8_dw(L), so a 16-code run's dwords are read in the
 * order 0, 2, 1, 3 and no dword is split. The map is its own inverse. */
__device__ __forceinline__ int i8_dw(int L)
{
    return (L & ~3) | ((L & 1) << 1) | ((L & 2) >> 1);
}

/* (code - zero) * scale in f16 for a sub-byte code read through the block's table, which holds
 * f16(i) for each code value: one LDS read, one v_sub_f16, one v_mul_f16. Summed over a group this
 * equals the reference's `sub(g) = is(g) - z[n][g] * a_sum[g][m]` term for term. */
__device__ __forceinline__ uint32_t fold_pair_q(uint32_t codepair, const uint16_t* tbl,
                                                _Float16 s, _Float16 zf, int bits)
{
    const uint32_t m = (1u << bits) - 1u;
    /* The table entry is an f16 bit pattern, so it is bit_cast, not converted. The zero comes off
     * before the scale: code minus zero is exact in f16 for a 4-bit code and a U8 zero, so the
     * staged value costs one rounding, where (code*s) - (z*s) costs two and cancels. */
    const _Float16 a = (__builtin_bit_cast(_Float16, tbl[codepair & m]) - zf) * s;
    const _Float16 b = (__builtin_bit_cast(_Float16, tbl[(codepair >> bits) & m]) - zf) * s;
    return (uint32_t)__builtin_bit_cast(uint16_t, a) |
           ((uint32_t)__builtin_bit_cast(uint16_t, b) << 16);
}

__device__ __forceinline__ float scale_w(const WQ h, int nb, int g)
{
    const uint8_t* p = h.scale + (long long)nb * h.scale_row + (long long)g * (h.sdt == 2 ? 4 : 2);
    if (h.sdt == 2) { float f; __builtin_memcpy(&f, p, 4); return f; }
    const uint16_t u = *(const uint16_t*)p;
    return h.sdt == 1 ? f16_to_f32(u) : bf16_to_f32(u);
}

/* The scale of (row n, group g) in the plane's grid:
 *   fp8a8  b_scale rows are 128-row blocks ({group, group}): row n is in block n>>7, so one scalar
 *          serves a tile's 128 rows.
 *   w2a8   b_scale rows are the weight's rows ({1, group}): 128 scalars a staged chunk.
 * `srows` must match the grid. A scale read from the wrong row is a systematic error of a few
 * percent, not noise, and passes a loose gate. */
__device__ __forceinline__ float scale_at_w(const WQ h, int n, int g)
{
    return scale_w(h, h.srows ? n : (n >> 7), g);
}

__device__ __forceinline__ float zero_at_w(const WQ h, int n, int g)
{
    if (!h.zero) return 0.0f;
    return (float)h.zero[(long long)n * h.zero_row + g];   /* U8 zero plane */
}



/* code * scale in f16: one v_mul_f16, round to nearest, subnormals kept. Not f16x2_pack, which
 * truncates (see the header). */
/* Eight codes -> four f16 pairs. E4M3 through the ALU decode (ALU=1) or the LDS table (ALU=0);
 * int8 through the codec's hardware convert (two v_cvt_f32_i8 and a pack). ALU is a template
 * parameter so the two rows (the table and the ALU decode) share the kernel and the tune picks. */
template <int ALU>
__device__ __forceinline__ void codes8(const uint2 c, int cdt, const uint16_t* tbl, uint32_t* w)
{
    if (!cdt) {
        r3d_e4m3x8_decode<ALU>(tbl, c, w);
        return;
    }
    const uint32_t d[2] = { c.x, c.y };
#pragma unroll
    for (int j = 0; j < 4; ++j)
        w[j] = i8x2_to_f16x2((d[j >> 1] >> (16 * (j & 1))) & 0xffffu);
}

/*
 * The tile: 128 x 128 outputs, 256 threads (eight waves), an 8x8 square a lane (64 f32
 * accumulators), grid ceil(N/kTN) x ceil(M/kTM). k is walked in chunks of 32; both sides are
 * decoded and leave staging as f16 with their scale folded in, so the dot loop is
 * r3d_vocab_bf16.hip's and does no scaling.
 *
 * ARM 0 is fp8a8 (E4M3 or int8 weight codes); ARM 1 is the zero-point grid (U2 or U4 codes against
 * an int8 activation in step order).
 */
template <int NT, int ARM, int CDT, int ALU>   /* CDT: 0 E4M3, 1 int8, 2 U2, 3 U4; ALU: 1 decode
                                                * in registers, 0 through the LDS table */
__global__ void __launch_bounds__(kBlockThreads) R3D_WAVES(2)
fp8a8_tiled_kernel(const uint8_t* __restrict__ a, ActQ aq, const uint8_t* __restrict__ b, WQ h,
                   void* __restrict__ y, int M, int N, int K, int a_ld, int y_ld)
{
    __shared__ __align__(16) uint16_t sA[2][kTM * kKC];
    __shared__ __align__(16) uint16_t sB[2][kTN * kKC];
    __shared__ __align__(16) uint16_t sTbl[R3D_E4M3_TBL];

    const int tid = (int)threadIdx.x;
    if (ARM == 0 && !ALU) {
        r3d_e4m3_f16_init(sTbl, tid, kBlockThreads);   /* E4M3 -> f16 decoder table (ALU=0) */
    } else if (ARM != 0) {
        /* U2 and U4 codes are unsigned integers (at most 16 values): the table holds f16(i). */
        for (int i = tid; i < 16; i += kBlockThreads) sTbl[i] = f32_to_f16((float)i);
        __syncthreads();
    }
    const int m0 = (int)blockIdx.y * kTM;
    const int n0 = (int)blockIdx.x * kTN;
    const int rg = tid >> 4, cg = tid & 15;

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;

    const int m_r = m0 + rg * 8, n_c = n0 + cg * 8;
    const int nb = n0 / kGroup;                              /* TN is one scale-plane row */

    /* ARM 1: thread tid stages weight row n0+tid in every chunk, so the (scale, zero) pair it needs
     * changes only with the k-group (every fourth chunk) and is loaded one chunk ahead, overlapping
     * the dot. Loading it inside the fold puts a round trip on the staging critical path, which at
     * M=64 N=12288 K=2048 is about 90x slower. */
    const int  b_row  = n0 + tid;
    const bool b_has  = (ARM == 1) && (tid < kTN) && (b_row < N);
    _Float16 s_cur = (_Float16)0.0f, zf_cur = (_Float16)0.0f;
    if (b_has) {
        s_cur  = (_Float16)scale_at_w(h, b_row, 0);
        zf_cur = (_Float16)zero_at_w(h, b_row, 0);
    }
    /* The code stream is limited by sector amplification, not loads in flight (a one-chunk lead
     * on the codes load gains under 1%). kKC = 32 U2 codes are 8 bytes from a 512-byte-strided
     * plane, so each row costs a whole 64-byte sector, 8x the bytes used. A wider k-chunk (64: 4x,
     * 128: 2x) would cut that, at a cost in LDS and occupancy. */

    /* Stage slab KS into the BUF-th LDS plane: the A rows and the B rows (or the sub-byte weight
     * codes) for k in [KS, KS + kKC), decoded and scale-folded to f16. The double-buffered walk
     * calls it for slab k+1 into the other plane while slab k's dot runs, so the global codes load
     * overlaps the multiply. */
    auto stage_slab = [&](int BUF, int KS) {
        const int g = KS / kGroup;                           /* one group a chunk: 32 divides 128 */
        const float sb = scale_w(h, nb, g);                  /* one scale for the whole tile */
#pragma unroll
        for (int j = tid; j < kTM * (kKC / 8); j += kBlockThreads) {
            const int mm = j / (kKC / 8), c8 = j % (kKC / 8);
            const int m = m0 + mm, k = KS + c8 * 8;
            uint4 staged = make_uint4(0, 0, 0, 0);
            if (m < M) {
                if (ARM == 0) {
                    /* One scale a staged run, which lies inside one group. In row form
                     * (`aq.gmode` 0) the group term is selected away without a branch. */
                    const float sa = aq.scale[(long long)m * aq.row + (aq.gmode ? g : 0)];
                    uint32_t w[4];
                    codes8<ALU>(r3d_loadc8<NT>(a + (long long)m * a_ld + k), 0, sTbl, w);
                    staged = make_uint4(r3d_fold_pair(w[0], sa), r3d_fold_pair(w[1], sa),
                                        r3d_fold_pair(w[2], sa), r3d_fold_pair(w[3], sa));
                } else {
                    /* int8 codes in quant_act_i8's step order, read a dword at the permuted
                     * address (i8_dw), with one scale per row (ZP_I8_ROW). */
                    const float sa = aq.scale[(long long)m * aq.row];
                    const int L0 = k >> 2;
                    const uint8_t* ap = a + (long long)m * a_ld;
                    const uint32_t d0 = *(const uint32_t*)(ap + (i8_dw(L0) << 2));
                    const uint32_t d1 = *(const uint32_t*)(ap + (i8_dw(L0 + 1) << 2));
                    const uint32_t d[2] = { d0, d1 };
                    uint32_t w[4];
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                        w[q] = i8x2_to_f16x2((d[q >> 1] >> (16 * (q & 1))) & 0xffffu);
                    staged = make_uint4(r3d_fold_pair(w[0], sa), r3d_fold_pair(w[1], sa),
                                        r3d_fold_pair(w[2], sa), r3d_fold_pair(w[3], sa));
                }
            }
            *(uint4*)&sA[BUF][srow(mm, c8, kKC)] = staged;
        }
        if (ARM == 0) {
#pragma unroll
            for (int j = tid; j < kTN * (kKC / 8); j += kBlockThreads) {
                const int nn = j / (kKC / 8), c8 = j % (kKC / 8);
                const int n = n0 + nn, k = KS + c8 * 8;
                uint4 staged = make_uint4(0, 0, 0, 0);
                if (n < N) {
                    uint32_t w[4];
                    codes8<ALU>(r3d_loadc8<NT>(h.codes + (long long)n * h.code_row + k), h.cdt, sTbl, w);
                    staged = make_uint4(r3d_fold_pair(w[0], sb), r3d_fold_pair(w[1], sb),
                                        r3d_fold_pair(w[2], sb), r3d_fold_pair(w[3], sb));
                }
                *(uint4*)&sB[BUF][srow(nn, c8, kKC)] = staged;
            }
        } else {
            /* The sub-byte weight: w2a8 (U2, four codes a byte, K>>2 bytes a row) or w4a8_asym
             * (U4, two a byte, K>>1). A thread loads a row's whole 32-code chunk at once, 8 bytes
             * of U2 or 16 of U4, aligned because KS is a multiple of 32. Scale and zero are per
             * (row, group), prefetched one slab ahead. */
            constexpr int BITS = (CDT == 2) ? 2 : 4;
            constexpr int FIELD = 2 * BITS;                 /* bits a pair of codes occupies */
#pragma unroll
            for (int nn = tid; nn < kTN; nn += kBlockThreads) {
                const int n = n0 + nn;
                uint4 st[4] = { make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0),
                                make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0) };
                if (n < N) {
                    const _Float16 s = s_cur, zf = zf_cur;   /* the group's pair, prefetched */
                    if (b_has && KS + kKC < K && (KS + kKC) / kGroup != g) {
                        const int gn = (KS + kKC) / kGroup;
                        s_cur  = (_Float16)scale_at_w(h, b_row, gn);
                        zf_cur = (_Float16)zero_at_w(h, b_row, gn);
                    }
                    uint32_t d[4] = { 0, 0, 0, 0 };
                    const uint8_t* cp = h.codes + (long long)n * h.code_row
                                        + (KS >> (CDT == 2 ? 2 : 1));
                    if (CDT == 2) { uint2 v; __builtin_memcpy(&v, cp, 8); d[0] = v.x; d[1] = v.y; }
                    else { u16v v; __builtin_memcpy(&v, cp, 16);
                           d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w; }
                    uint32_t w[16];
#pragma unroll
                    for (int q = 0; q < 16; ++q) {
                        /* Pair q sits at bit FIELD*q: a 4-bit field for U2 (eight a dword), an 8-bit
                         * field for U4 (four a dword). No field straddles a dword. */
                        const int bit = q * FIELD;
                        w[q] = fold_pair_q((d[bit >> 5] >> (bit & 31)) & ((1u << FIELD) - 1u),
                                          sTbl, s, zf, BITS);
                    }
                    st[0] = make_uint4(w[0], w[1], w[2], w[3]);
                    st[1] = make_uint4(w[4], w[5], w[6], w[7]);
                    st[2] = make_uint4(w[8], w[9], w[10], w[11]);
                    st[3] = make_uint4(w[12], w[13], w[14], w[15]);
                }
#pragma unroll
                for (int q = 0; q < 4; ++q) *(uint4*)&sB[BUF][srow(nn, q, kKC)] = st[q];
            }
        }
    };

    /* The double-buffered k-walk: slab 0 is staged before the loop; each iteration stages slab
     * k+1 into the other LDS plane while slab k's dot runs from the current plane, then dots. One
     * barrier a slab, after the dot, keeps the next slab's stage clear of the dot's LDS reads. */
    int tbuf = 0;
    stage_slab(0, 0);
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += kKC) {
        const int k1 = k0 + kKC;
        if (k1 < K) stage_slab(1 - tbuf, k1);
#pragma unroll
        for (int kc = 0; kc < kKC; kc += 2) {
            uint32_t af[8], bf[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                af[i] = *(const uint32_t*)&sA[tbuf][srow_pair(rg * 8 + i, kc, kKC)];
#pragma unroll
            for (int j = 0; j < 8; ++j)
                bf[j] = *(const uint32_t*)&sB[tbuf][srow_pair(cg * 8 + j, kc, kKC)];
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = dot2_f16x2(af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
        tbuf = 1 - tbuf;
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
                p[j] = (uint32_t)f32_to_bf16(acc[i][2 * j]) |
                       ((uint32_t)f32_to_bf16(acc[i][2 * j + 1]) << 16);
            *(uint4*)(row + n_c) = make_uint4(p[0], p[1], p[2], p[3]);
        } else {
            /* Unrolled: a dynamic index into acc[i] would put the 64 accumulators in local
             * memory. */
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (n_c + j < N) row[n_c + j] = f32_to_bf16(acc[i][j]);
        }
    }
}

/* ------------------------------------------------------------------ the launcher ------------------ */
/* A pitch refusal: traces what it saw and returns the internal -3, which the shim maps to
 * E_STRIDE. */
#define L_STR(why, a1, v1, a2, v2)                                                    \
    do {                                                                              \
        if (r3d_trace_on())                                                           \
            std::fprintf(stderr, "[r3d] %s:%d %s: %s = %lld, %s = %lld\n",           \
                         __FILE__, __LINE__, (why), (a1), (long long)(v1), (a2),      \
                         (long long)(v2));                                            \
        return R3D_NO(-3);                                                            \
    } while (0)

int launch(long av, long asv, long bvc, long bsv, long yv, int M, int N, int K, int group,
           int a_ld, long long b_row, int as_ld, long long bs_row, int y_ld, int nt, long stream,
           int sdt, int cdt, long zv = 0, long long z_row = 0, int srows = 0, int arm = 0,
           int alu = 1)
{
    if (!av || !asv || !bvc || !bsv || !yv) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (group != kGroup) return -2;              /* the fold indexes 128; kKC 32 divides it */
    if (K % kGroup) return -2;                   /* a ragged k-block reads past the scale plane */
    if (nt < 0 || nt > 1) return -2;
    if (a_ld % 8) L_STR("a pitch: 8-byte code loads, 8 codes a lane", "a_ld", a_ld, "K", K);
    if (b_row & 7) L_STR("b row pitch is not 8-aligned", "b_row", b_row, "N", N);
    /* The activation scale's pitch: ARM 0 takes one scale per (row, group) at width ceil(K/group)
     * or one per row at width 1 (ZP_I8_ROW, a per-row quantiser's output); `as_ld == 1` selects row
     * mode. ARM 1 is always one scale per row. */
    if (arm == 0 && as_ld != 1 && as_ld < (K + kGroup - 1) / kGroup) L_STR(
            "a_scale: neither one per row nor one per (row, group)", "as_ld", as_ld,
            "ceil(K/group)", (K + kGroup - 1) / kGroup);
    if (arm == 1 && as_ld < 1) return -3;
    if (y_ld % 8) L_STR("y pitch: 16-byte bf16 store runs", "y_ld", y_ld, "N", N);
    if (a_ld < K) L_STR("a pitch shorter than K", "a_ld", a_ld, "K", K);
    if (y_ld < N) L_STR("y pitch shorter than N", "y_ld", y_ld, "N", N);
    if ((av & 7) || (bvc & 7) || (yv & 7) || (asv & 3) || (bsv & 3)) return -5;
    if (bs_row & (sdt == 2 ? 3 : 1)) return -3;  /* scale loads are 4 bytes (f32) or 2 (bf16/f16);
                                                    a 3-column 16-bit row (6 bytes) is legal */
    if (arm == 1) {
        if (!(arm == 1 && (cdt == 2 || cdt == 3))) return -2;
        if (!zv) return -1;                      /* the zero-point grid needs its zero plane */
        if (z_row < 1) return -3;
        if (zv & 1) return -5;
        if (cdt == 3 && (bvc & 15)) return -5;    /* a 16-byte U4 load; U2 loads 8 */
    }

    const dim3 blk(kBlockThreads);
    const dim3 grd((unsigned)((N + kTN - 1) / kTN), (unsigned)((M + kTM - 1) / kTM));
    /* Value-init (not aggregate-init): the tape audit byte-compares the serialized kernel
     * arguments, and aggregate-init leaves a struct's trailing padding indeterminate. A padding
     * byte that differs between a pass's recording and its audit is read as "the key does not
     * carry" this argument and stops the engine (the ActQ/WQ holes are 4 bytes each). Zero the
     * whole struct so the padding is deterministic; the kernel reads none of it. */
    ActQ aq{};
    aq.scale = (const float*)(uintptr_t)asv;
    aq.row = as_ld;
    aq.gmode = as_ld != 1;
    WQ h{};
    h.codes = (const uint8_t*)(uintptr_t)bvc;
    h.scale = (const uint8_t*)(uintptr_t)bsv;
    h.zero = (const uint8_t*)(uintptr_t)zv;
    h.code_row = b_row;
    h.scale_row = bs_row;
    h.zero_row = z_row;
    h.sdt = sdt;
    h.cdt = cdt;
    h.srows = srows;
    hipStream_t s = (hipStream_t)(uintptr_t)stream;
#define R3D_RUN(NT, ARM, CD, AL)                                                \
    fp8a8_tiled_kernel<NT, ARM, CD, AL><<<grd, blk, 0, s>>>(                   \
        (const uint8_t*)(uintptr_t)av, aq, (const uint8_t*)(uintptr_t)bvc,      \
        h, (void*)(uintptr_t)yv, M, N, K, a_ld, y_ld)
#define R3D_LAUNCH(ARM, CD)                                                     \
    do {                                                                        \
        if (alu) { if (nt) R3D_RUN(1, ARM, CD, 1); else R3D_RUN(0, ARM, CD, 1); } \
        else     { if (nt) R3D_RUN(1, ARM, CD, 0); else R3D_RUN(0, ARM, CD, 0); } \
    } while (0)
    if (arm == 0) { if (cdt == 1) R3D_LAUNCH(0, 1); else R3D_LAUNCH(0, 0); }
    else          { if (cdt == 3) R3D_LAUNCH(1, 3); else R3D_LAUNCH(1, 2); }
#undef R3D_RUN
#undef R3D_LAUNCH
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
        if (r3d_trace_on())
            std::fprintf(stderr, "[launch] arm=%d cdt=%d nt=%d grid(%u,%u) block(%u) M=%d N=%d K=%d: %s\n",
                         arm, cdt, nt, grd.x, grd.y, kBlockThreads, M, N, K, hipGetErrorString(le));
        return -4;
    }
    return 0;
}

/* libref's float_scale_dt and enc_scaled_rows, transcribed (libref keeps them static inline),
 * checks and order unchanged. */
bool float_scale_dt(uint32_t dt) { return dt == RAD_BF16 || dt == RAD_F16 || dt == RAD_F32; }

bool enc_scaled_rows(const RadEncoding* enc, const int* sel, int n)
{
    if (!rad_enc_is(enc, "affine") || enc->transform[0] || enc->n_planes != 2) return false;
    if (n != 2 || !sel) return false;                     /* sel_roles(enc, sel, n, "codes","scale") */
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

/* ==================================================================================================
 * The layout hook: b (operand 2) as codes + one scale.
 *
 * The reference's gemm_nt_q hook, `ref_layout_gemm_q`, serves only the zero-point grids
 * (`zp_grid` lists w2a8, w4a8_asym, w4a16_asym, w5a16, w4a8i8_asym, w5a8i8) and returns E_DTYPE
 * for fp8a8, leaving no reader. For fp8a8 this hook follows `ref_layout_head` instead,
 * `layout_scaled(operand, 2, -1, ...)`: two planes, "codes" and "scale", no zero plane (fp8a8 is
 * symmetric), codes tiled one a cell, the scale on the {group, group} block the shape function
 * names.
 * ================================================================================================== */
/* The writer for the layout the hook names: rad_planes_relayout assembles the container's codes
 * and per-block scales as one operand, with the hook's operand guard. */
extern "C" int r3d_relayout_gemm_q_fp8_weight(const RadParam* p, int n_p, int operand,
                                              const RadEncoding* enc, const int* sel,
                                              const RadTensor* planes, int n_planes, void* dst,
                                              int64_t dst_bytes)
{
    /* The query and the writer are one plan; see r3d_layout_gemm_q_fp8_weight. */
    const char* dt = r3d_pgets(p, n_p, "dtype");
    if (dt && (std::strcmp(dt, "w2a8") == 0 || std::strcmp(dt, "w4a8_asym") == 0))
        return r3d_relayout_gemm_nt_q_w2a8(p, n_p, operand, enc, sel, planes, n_planes, dst,
                                           dst_bytes);   /* the decode row's plan, one writer */
    if (operand != 2) return R3D_NO(RAD_E_UNSUPPORTED);
    return rad_planes_relayout(enc, sel, planes, n_planes, dst, dst_bytes);
}

/* Traces a layout query (operand, band dtype, planes offered, code returned) when tracing is on
 * (RAD_R3D_TRACE=1), since a refused query otherwise leaves no record. */
static int r3d_layout_trace(const char* who, int operand, const char* dt, int n_planes, int rc) {
    if (r3d_trace_on())
        std::fprintf(stderr, "[r3d] %s: operand %d dtype %s planes %d -> %d\n", who, operand,
                     dt ? dt : "(none)", n_planes, rc);
    return rc;
}

/* The grids this row reads behind operand 2, in two branches that share no checks. fp8a8 is the
 * symmetric pair (codes, scale) on {1,1} + {group,group}. w2a8 and w4a8_asym are the zero-point
 * triple (codes, scale, zero); the codes dtype follows the grid (w2a8 -> U2, w4a8_asym -> U4).
 *
 * The zero-point grids delegate to the decode row's hook, r3d_layout_gemm_nt_q_w2a8 (codes plain
 * on operand 2, the (scale, zero) pair joined on operand 3, {1, group} block, f16 scale, unsigned
 * zero). The engine stores one form per weight across every row that reads it, so both rows must
 * describe the weight identically or the model does not load. */
extern "C" int r3d_layout_gemm_q_fp8_weight(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const RadTensor* planes, int n, RadLayout* out)
{
    const char* dt = r3d_pgets(p, n_p, "dtype");
    if (dt && (std::strcmp(dt, "w2a8") == 0 || std::strcmp(dt, "w4a8_asym") == 0))
        return r3d_layout_trace("r3d_layout_gemm_q_fp8_weight", operand, r3d_pgets(p, n_p, "dtype"), n, r3d_layout_gemm_nt_q_w2a8(p, n_p, operand, enc, sel, planes, n, out));

    if (n < 2 || operand != 2) return r3d_layout_trace("r3d_layout_gemm_q_fp8_weight", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_UNSUPPORTED);   /* b is operand 2 of gemm_nt_q */

    if (!enc_scaled_rows(enc, sel, n)) return r3d_layout_trace("r3d_layout_gemm_q_fp8_weight", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
    const long long group = r3d_pgeti_or(p, n_p, "group", 0);
    if (group <= 0) return r3d_layout_trace("r3d_layout_gemm_q_fp8_weight", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);                    /* the band names the block */
    if (enc->plane[sel[1]].block[0] != group || enc->plane[sel[1]].block[1] != group)
        return r3d_layout_trace("r3d_layout_gemm_q_fp8_weight", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);                                /* {group, group} */
    return r3d_layout_trace("r3d_layout_gemm_q_fp8_weight", operand, r3d_pgets(p, n_p, "dtype"), n, rad_planes_layout(enc, sel, planes, n, out));
}

/* ==================================================================================================
 * The planes' entry: the caller has resolved the two planes and passes their addresses. fp8a8,
 * group 128, y bf16; codes row-major, scales [ceil(N/128)][ceil(K/128)] read as bf16 (fp8a8f32's
 * f32 scales go through the shim, which reads the dtype off the plane). b_row and bs_row of 0 mean
 * the dense pitch.
 * ================================================================================================== */
extern "C" int r3d_gemm_nt_q_fp8_tiled(long a, long a_scale, long b_codes, long b_scale, long y,
                                       int M, int N, int K, int group, int a_ld, long long b_row,
                                       int as_ld, long long bs_row, int y_ld, int nt, long stream)
{
    const int st = launch(a, a_scale, b_codes, b_scale, y, M, N, K, group, a_ld, b_row ? b_row : K,
                          as_ld, bs_row ? bs_row : (long long)((K + kGroup - 1) / kGroup) * 2, y_ld,
                          nt, stream, 0, 0);
    if (st == -1) return R3D_NO(RAD_E_INVAL);
    if (st == -2) return R3D_NO(RAD_E_SHAPE);
    if (st == -3) return R3D_NO(RAD_E_STRIDE);
    if (st == -5) return R3D_NO(RAD_E_ALIGN);
    if (st == -4) return R3D_NO(RAD_E_DEVICE);
    return RAD_OK;
}

/* ==================================================================================================
 * The RadArgs shim. Schema positions: a 0, a_scale 1, b 2, b_scale 3, y 4, a_sum 5, b_ref 6. When
 * the placement joins b and b_scale into one container, 3 is absent and the header is read here
 * on the host; a caller that resolved the planes passes 3 and takes the plain path.
 * ================================================================================================== */

/* Shims take RadStream so they match RadLaunchFn (rad_abi.h); the long-taking entries cast at the
 * boundary. */

/* Latch for the trace plan line: true once per distinct (M, N, K, ranks, pitches), since the shim
 * runs many times a forward pass. Unlocked: a race can only print a shape twice, and only with
 * tracing on. */
struct GemmPlanKey { long long M, N, K; unsigned a_rank, as_rank; long long a_ld, y_ld; };

static bool gemm_plan_once(long long M, long long N, long long K, unsigned a_rank, unsigned as_rank,
                           long long a_ld, long long y_ld)
{
    static GemmPlanKey seen[96];
    static int n = 0;
    for (int i = 0; i < n; ++i)
        if (seen[i].M == M && seen[i].N == N && seen[i].K == K && seen[i].a_rank == a_rank &&
            seen[i].as_rank == as_rank && seen[i].a_ld == a_ld && seen[i].y_ld == y_ld)
            return false;
    if (n < 96) seen[n++] = GemmPlanKey{M, N, K, a_rank, as_rank, a_ld, y_ld};
    return true;
}

static int r3d_fp8a8_tiled_dispatch(const RadArgs* args, RadStream s, int alu)
{
    if (!args || args->n_t < 5) return R3D_NO(RAD_E_INVAL);
    const RadTensor* A = r3d_opt(args, 0);
    const RadTensor* AS = r3d_opt(args, 1);
    const RadTensor* B = r3d_opt(args, 2);
    const RadTensor* BS = r3d_opt(args, 3);
    const RadTensor* Y = r3d_opt(args, 4);
    const RadTensor* ASUM = r3d_opt(args, 5);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(Y)) return R3D_NO(RAD_E_INVAL);

    /* gemm_nt_q's keys are M, N, K, group, dtype (r3d_rows.cpp pGemmQ), not logits_gemm's
     * n_vocab/n_embd. Reading the wrong keys gives N = K = 0 and silently refuses every band. A
     * band that names no N or K falls back to the weight's dims. */
    const long long M_band = r3d_geti_or(args, "M", 0), N0 = r3d_geti_or(args, "N", 0);
    const long long K0 = r3d_geti_or(args, "K", 0);
    const long long N = N0 > 0 ? N0 : r3d_dim(B, 0);
    const long long K = K0 > 0 ? K0 : r3d_dim(B, 1);
    const long long group = r3d_geti_or(args, "group", 0);
    if (N <= 0 || K <= 0) return R3D_NO(RAD_E_SHAPE);
    if (K > (1 << 24)) return R3D_NO(RAD_E_INVAL);   /* absurd K */
    if (group != kGroup) return R3D_NO(RAD_E_SHAPE);   /* the fold indexes 128 */
    if (K % kGroup) return R3D_NO(RAD_E_SHAPE);
    if (ASUM && r3d_p(ASUM))
        return R3D_NO(RAD_E_DTYPE);       /* fp8a8 is symmetric; a_sum belongs to the zero-point
                                             grid (r3d_gemm_nt_q_w2a8.hip) */

    /* ---- b: a container of codes + scale, or the two planes as tensors -------------------------- */
    const void* bvc = r3d_p(B);
    const void* bsv = nullptr;
    long long b_row = 0, bs_row = 0;
    int sdt = 0, cdt = 0;
    const RadPlanesHeader* ph = (const RadPlanesHeader*)rad_planes_header(bvc);
    if (ph) {
        /* rad_planes_find points into the container; nothing is copied. */
        const RadPlanesEntry* ce = nullptr;
        const RadPlanesEntry* se = nullptr;
        if (rad_planes_find(bvc, "zero", nullptr))
            return R3D_NO(RAD_E_DTYPE);   /* codes+scale+zero: the zero-point grid */
        if (ph->n_planes != 2) return R3D_NO(RAD_E_DTYPE);
        const void* cb = rad_planes_find(bvc, "codes", &ce);
        const void* sb = rad_planes_find(bvc, "scale", &se);
        if (!cb || !sb) return R3D_NO(RAD_E_DTYPE);      /* a joined weight that is not codes + scale */
        if (ce->kind != RAD_PLANE_TILED || ce->block[0] != 1 || ce->block[1] != 1)
            return R3D_NO(RAD_E_DTYPE);                  /* codes: one a cell, tiled plain */
        if (ce->dtype != RAD_F8E4M3 && ce->dtype != RAD_I8) {
            if (rad_dtype_bytes(ce->dtype, 8) < 8)
                return R3D_NO(RAD_E_DTYPE);              /* sub-byte codes */
            return R3D_NO(RAD_E_DTYPE);                  /* e5m2 and the rest: no decoder here */
        }
        if (se->kind != RAD_PLANE_TILED) return R3D_NO(RAD_E_DTYPE);
        if (se->dtype == RAD_BF16) sdt = 0;
        else if (se->dtype == RAD_F16) sdt = 1;
        else if (se->dtype == RAD_F32) sdt = 2;
        else return R3D_NO(RAD_E_DTYPE);                 /* not in float_scale_dt's set */
        if (se->block[0] != group || se->block[1] != group)
            return R3D_NO(RAD_E_SHAPE);   /* only {group, group} blocks */
        bvc = cb; bsv = sb; b_row = ce->row_bytes; bs_row = se->row_bytes;
        cdt = ce->dtype == RAD_I8;
        const long long sb_r = (N + group - 1) / group, sb_c = (K + group - 1) / group;
        if (ce->rows != N || ce->cols != K) return R3D_NO(RAD_E_SHAPE);
        if (se->rows != sb_r || se->cols != sb_c) return R3D_NO(RAD_E_SHAPE);
        if (ce->row_bytes < K) return R3D_NO(RAD_E_STRIDE);   /* one byte a code */
        if (se->row_bytes < sb_c * (sdt == 2 ? 4 : 2)) return R3D_NO(RAD_E_STRIDE);
    } else {
        if (!BS || !r3d_p(BS))
            return R3D_NO(RAD_E_DTYPE);   /* neither a container nor a scale tensor */
        b_row = K;
        const long long sb_r = (N + group - 1) / group, sb_c = (K + group - 1) / group;
        /* b_scale may itself be a container: codes plain on b, the scale joined behind b_scale (the
         * decode row's spelling for a shared weight, which r3d_relayout_gemm_q_fp8_weight can
         * write). The header is read on the host; the kernel gets the scale plane's bytes and the
         * entry's row pitch, the same as the two-tensor spelling. A `zero` plane is refused: fp8a8
         * is symmetric and the reference defines no law for it here. */
        const RadPlanesHeader* sph = (const RadPlanesHeader*)rad_planes_header(r3d_p(BS));
        if (sph) {
            if (rad_planes_find(r3d_p(BS), "zero", nullptr))
                return R3D_NO(RAD_E_DTYPE);   /* codes + scale + zero: the zero-point grid */
            const RadPlanesEntry* se = nullptr;
            const void* sb = rad_planes_find(r3d_p(BS), "scale", &se);
            if (!sb || !se)
                return R3D_NO(RAD_E_DTYPE);   /* a joined stream with no `scale` plane */
            if (se->kind != RAD_PLANE_TILED) return R3D_NO(RAD_E_DTYPE);
            if (se->dtype == RAD_BF16) sdt = 0;
            else if (se->dtype == RAD_F16) sdt = 1;
            else if (se->dtype == RAD_F32) sdt = 2;
            else return R3D_NO(RAD_E_DTYPE);  /* not in float_scale_dt's set */
            if (se->block[0] != group || se->block[1] != group)
                return R3D_NO(RAD_E_SHAPE);   /* one scale per 128 rows x 128 codes only */
            if (se->rows != sb_r || se->cols != sb_c) return R3D_NO(RAD_E_SHAPE);
            if (se->row_bytes < sb_c * (sdt == 2 ? 4 : 2)) return R3D_NO(RAD_E_STRIDE);
            bsv = sb;
            bs_row = se->row_bytes;
        } else {
            bsv = r3d_p(BS);
            long long bs_rows = 0;
            const long long bs_pitch = r3d_rows_pitch(BS, sb_c, &bs_rows);
            if (bs_pitch < 0 || bs_rows < sb_r) return R3D_NO(RAD_E_STRIDE);
            if (BS->dtype == RAD_BF16) sdt = 0;
            else if (BS->dtype == RAD_F16) sdt = 1;
            else if (BS->dtype == RAD_F32) sdt = 2;
            else return R3D_NO(RAD_E_DTYPE);
            if (r3d_dim(BS, 0) < sb_r || r3d_dim(BS, 1) < sb_c) return R3D_NO(RAD_E_SHAPE);
            bs_row = bs_pitch * (sdt == 2 ? 4 : 2);      /* the kernel walks the plane's bytes */
        }
        if (B->dtype == RAD_F8E4M3) cdt = 0;
        else if (B->dtype == RAD_I8) cdt = 1;
        else if (rad_dtype_bytes(B->dtype, 8) < 8) return R3D_NO(RAD_E_DTYPE);
        else return R3D_NO(RAD_E_DTYPE);
        if (r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);
        if (r3d_dim(B, 0) != N || r3d_dim(B, 1) != K) return R3D_NO(RAD_E_SHAPE);
    }

    if (A->dtype != RAD_F8E4M3) return R3D_NO(RAD_E_DTYPE);
    if (AS->dtype != RAD_F32) return R3D_NO(RAD_E_DTYPE);
    if (Y->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);   /* gemm_q_shape: y is bf16 */
    if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);

    long long a_rows = 0;
    long long a_ld = r3d_rows_pitch(A, K, &a_rows);
    /* The step's M is the activation's row count, as libref and the m16 row derive it. The band
     * key `M` is only an upper bound, checked below. */
    if (a_ld < K || a_rows < 1) {
        /* A group-quantised activation may arrive as rank 3, [M][ceil(K/group)][group], which
         * r3d_rows_pitch cannot read as rows of K. A contiguous plane whose numel divides by K is
         * rows K wide. */
        if (rad_tensor_is_contiguous(A) && K > 0 && r3d_numel(A) % K == 0) {
            const long long r = r3d_numel(A) / K;
            if (r >= 1) { a_ld = K; a_rows = r; }
        }
    }
    if (a_ld < K || a_rows < 1) { r3d_trace_desc("a", A);
        std::fprintf(stderr, "[r3d] a: pitch %lld rows %lld, want pitch >= %lld and at least one row\n",
                     a_ld, a_rows, (long long)K);
        return R3D_NO(RAD_E_STRIDE); }
    const long long M = a_rows;
    if (M_band > 0 && M > M_band) { r3d_trace_desc("a", A);
        std::fprintf(stderr, "[r3d] a: %lld rows in a band that declared M <= %lld\n", M, M_band);
        return R3D_NO(RAD_E_SHAPE); }
    if (M * N > (1ll << 34)) return R3D_NO(RAD_E_INVAL);   /* 32-bit tile indices, at the
                                                           * step's M */
    if (M < kMMin) return R3D_NO(RAD_E_UNSUPPORTED);   /* gemm_nt_q_fp8a8_m16 serves M <= 16 */
    const long long kb = (K + group - 1) / group;
    /* The activation's scales: [M][ceil(K/group)] from a group quantiser, or rank-1 [M] from a
     * per-row one (ZP_I8_ROW). The rank picks the width asked of r3d_rows_pitch, which would
     * refuse the rank-1 plane at width kb. */
    const int as_row = (AS->rank <= 1);
    const long long as_w = as_row ? 1 : kb;
    long long as_rows = 0;
    const long long as_ld = r3d_rows_pitch(AS, as_w, &as_rows);
    if (as_ld < as_w || as_rows < M) { r3d_trace_desc("a_scale", AS);
        std::fprintf(stderr, "[r3d] a_scale: want width %lld (rank %d), pitch %lld, rows %lld, M %lld\n",
                     as_w, AS->rank, as_ld, as_rows, M);
        return R3D_NO(RAD_E_STRIDE); }
    long long y_rows = 0;
    const long long y_ld = r3d_rows_pitch(Y, N, &y_rows);
    if (y_ld < N || y_rows < M) { r3d_trace_desc("y", Y);
        std::fprintf(stderr, "[r3d] y: pitch %lld rows %lld, want pitch >= %lld rows >= %lld\n",
                     y_ld, y_rows, (long long)N, M);
        return R3D_NO(RAD_E_STRIDE); }
    if (!r3d_aligned(r3d_p(A), 8) || !r3d_aligned(bvc, 8) || !r3d_aligned(r3d_p(Y), 8))
        return R3D_NO(RAD_E_ALIGN);
    if (!r3d_aligned(r3d_p(AS), 4) || !r3d_aligned(bsv, 4)) return R3D_NO(RAD_E_ALIGN);

    /* ---- the plan line ---------------------------------------------------------------------------
     *
     * With tracing on, prints once per shape the step's M, the operands' ranks, pitches and
     * strides, and the arena padding ceil(M/16)*16, after every check has passed. It covers the
     * plan's rank-1 `[M]` scale and rank-3 codes, which the shape-hook fixtures do not draw. */
    if (r3d_trace_on() && gemm_plan_once(M, N, K, A->rank, AS->rank, a_ld, y_ld)) {
        std::fprintf(stderr,
                     "[r3d] plan: gemm M=%lld (band M<=%lld) N=%lld K=%lld group=%lld "
                     "a rank=%u numel=%lld stride=[%lld,%lld] pitch=%lld rows=%lld "
                     "a_scale rank=%u numel=%lld pitch=%lld form=%s "
                     "y rank=%u numel=%lld stride=[%lld,%lld] pitch=%lld Mpad=%lld\n",
                     M, M_band, N, K, group, (unsigned)A->rank, r3d_numel(A),
                     r3d_stride(A, 0), r3d_stride(A, 1), a_ld, a_rows, (unsigned)AS->rank,
                     r3d_numel(AS), as_ld, as_row ? "row[M]" : "group[M][kb]", (unsigned)Y->rank,
                     r3d_numel(Y), r3d_stride(Y, 0), r3d_stride(Y, 1), y_ld, (M + 15) / 16 * 16);
        r3d_trace_desc("gemm a", A);
        r3d_trace_desc("gemm a_scale", AS);
        r3d_trace_desc("gemm y", Y);
    }

    const int nt = 0;   /* no `nt` axis in the row's tune table: the non-temporal stream is fixed off */
    const int st = launch(r3d_addr(A), r3d_addr(AS), (long)(uintptr_t)bvc, (long)(uintptr_t)bsv,
                          r3d_addr(Y), (int)M, (int)N, (int)K, (int)group, (int)a_ld, b_row,
                          (int)as_ld, bs_row, (int)y_ld, nt, (long)s, sdt, cdt, 0, 0, 0, 0, alu);
    if (st == -1) return R3D_NO(RAD_E_INVAL);
    if (st == -2) return R3D_NO(RAD_E_SHAPE);
    if (st == -3) return R3D_NO(RAD_E_STRIDE);
    if (st == -5) return R3D_NO(RAD_E_ALIGN);
    if (st == -4) return R3D_NO(RAD_E_DEVICE);
    return RAD_OK;
}

/* ==================================================================================================
 * The w2a8 arm's shim: gemm_nt_q at dtype=w2a8 (U2 codes) or w4a8_asym (U4 codes), the same tile
 * at M >= kMMin; below it gemm_nt_q_w2a8_m16 serves (C_LE("M", 16)). A typical band is a draft
 * head, N=130560 K=2048 group=128 at M=1792.
 *
 * The law is the reference's gemm_zp, ZP_I8_ROW arm, run in a folded form:
 *
 *   q[k]   = the grid's code of weight row n, column k                  U2: bits [2k,2k+2) of k/4
 *   sub(g) = SUM_{k in g} (int)a[m][i8_step_pos(k)] * ((int)q[k] - (int)z[n][g])   integers
 *   acc    = acc + (sub(g) * sa[m]) * sb[n][g]                          g ASCENDING
 *   y[m,n] = bf16(acc)
 *
 * This equals the reference's `sub = is - z * a_sum`, since sum a*(q - z) = sum a*q - z*sum a over
 * the same group. The staged tile carries (q - z) * sb and the a side a * sa[m], so the zero is
 * removed per element and a_sum is never read. a_sum is still required (gemm_zp reads it and the
 * schema carries it at slot 5), while the fp8a8 shim refuses it; the two arms keep separate checks.
 * The bracketing differs from the reference's per-group bracket by the fold's floor, which at a
 * bf16 output is below the store's ulp.
 *
 * The activation is in quant_act_i8's step order with one f32 scale per row (ZP_I8_ROW). The
 * weight's scale and zero are per (row, group).
 * ================================================================================================== */
extern "C" int r3d_rad_gemm_nt_q_w2a8_tiled(const RadArgs* args, RadStream s)
{
    if (!args || args->n_t < 5) return R3D_NO(RAD_E_INVAL);

    const char* dt = r3d_gets(args, "dtype");
    int cdt = -1;
    if (dt && std::strcmp(dt, "w2a8") == 0) cdt = 2;
    else if (dt && std::strcmp(dt, "w4a8_asym") == 0) cdt = 3;
    if (cdt < 0)
        return R3D_NO(RAD_E_DTYPE);   /* w4a8i8_asym and w5a8i8 use ZP_I8_GROUP (no per-row
                                       * scale), which this arm does not read */

    const RadTensor* A = r3d_opt(args, 0);
    const RadTensor* AS = r3d_opt(args, 1);
    const RadTensor* B = r3d_opt(args, 2);
    const RadTensor* BS = r3d_opt(args, 3);
    const RadTensor* Y = r3d_opt(args, 4);
    const RadTensor* ASUM = r3d_opt(args, 5);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(Y)) return R3D_NO(RAD_E_INVAL);
    if (!r3d_p(ASUM))
        return R3D_NO(RAD_E_DTYPE);   /* required by the schema, though never read (see above) */

    const long long M_band = r3d_geti_or(args, "M", 0), N0 = r3d_geti_or(args, "N", 0);
    const long long K0 = r3d_geti_or(args, "K", 0);
    const long long N = N0 > 0 ? N0 : r3d_dim(B, 0);       /* b is [N, K] */
    const long long K = K0 > 0 ? K0 : r3d_dim(B, 1);
    const long long group = r3d_geti_or(args, "group", 0);
    if (N <= 0 || K <= 0) return R3D_NO(RAD_E_SHAPE);
    /* The step's M comes from the activation, as in the fp8a8 shim; the band key `M` is the upper
     * bound. */
    long long M = 0;
    if (K > 0 && r3d_numel(A) % K == 0) M = r3d_numel(A) / K;
    if (M < 1) M = M_band;
    if (M_band > 0 && M > M_band) { r3d_trace_desc("a", A);
        std::fprintf(stderr, "[r3d] w2a8 a: %lld rows in a band that declared M <= %lld\n", M, M_band);
        return R3D_NO(RAD_E_SHAPE); }
    if (group != kGroup)
        return R3D_NO(RAD_E_SHAPE);   /* the fold indexes the group as kGroup */
    if (K % kGroup) return R3D_NO(RAD_E_SHAPE);            /* a ragged k-chunk reads past the plane */
    if (M < kMMin)
        return R3D_NO(RAD_E_UNSUPPORTED);   /* gemm_nt_q_w2a8_m16 serves M <= 16 */
    const long long G = (K + kGroup - 1) / kGroup;
    const long long mpad = ((M + 15) / 16) * 16;           /* a_sum is group-major [G][Mpad] */

    if (A->dtype != RAD_I8) return R3D_NO(RAD_E_DTYPE);    /* quant_act_i8's codes */
    if (Y->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(AS) || !r3d_is_f32(ASUM)) return R3D_NO(RAD_E_DTYPE);
    if (r3d_numel(AS) < M) return R3D_NO(RAD_E_INVAL);     /* one scale per ROW (ZP_I8_ROW) */
    if (r3d_numel(ASUM) < G * mpad) return R3D_NO(RAD_E_INVAL);
    if (r3d_numel(A) < M * K) return R3D_NO(RAD_E_SHAPE);
    if (M * N > (1ll << 34)) return R3D_NO(RAD_E_INVAL);   /* 32-bit tile indices */

    /* The weight's planes, found by role in either arrangement: codes, scale and zero all behind b,
     * or the decode row's (codes plain on b, scale+zero joined behind b_scale). One stored form is
     * settled per weight, so a weight shared with the decode row may arrive in either. */
    const void* bbase = r3d_p(B);
    const RadPlanesHeader* bh = rad_planes_header(bbase);
    bool codes_joined = false;                 /* the codes plane inside b's container? */
    const RadPlanesEntry *ce = nullptr, *se = nullptr, *ze = nullptr;
    const void* sp = nullptr; const void* zp = nullptr;
    long long c_row = 0, s_row = 0, z_row = 0;
    const void* sv = nullptr; const void* zv = nullptr;
    if (bh) {
        const long long cb = (cdt == 2) ? (K >> 2) : (K >> 1);
        sp = rad_planes_find(bbase, "scale", &se);
        zp = rad_planes_find(bbase, "zero", &ze);
        const void* cp = rad_planes_find(bbase, "codes", &ce);
        if (!cp || !sp || !zp) return R3D_NO(RAD_E_DTYPE);  /* three planes or nothing */
        if (ce->dtype != (uint32_t)(cdt == 2 ? RAD_U2 : RAD_U4)) return R3D_NO(RAD_E_DTYPE);
        if (ce->row_bytes < cb) return R3D_NO(RAD_E_STRIDE);
        c_row = ce->row_bytes;
        codes_joined = true;
    } else {
        if (!r3d_p(BS)) return R3D_NO(RAD_E_DTYPE);         /* the zero plane lives only in a
                                                             * container */
        const RadPlanesHeader* h3 = rad_planes_header(r3d_p(BS));
        if (!h3) return R3D_NO(RAD_E_DTYPE);               /* scale+zero joined behind b_scale */
        sp = rad_planes_find(r3d_p(BS), "scale", &se);
        zp = rad_planes_find(r3d_p(BS), "zero", &ze);
        if (!sp || !zp) return R3D_NO(RAD_E_DTYPE);
        if (B->dtype != (uint32_t)(cdt == 2 ? RAD_U2 : RAD_U4)) return R3D_NO(RAD_E_DTYPE);
        if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);
        c_row = rad_dtype_bytes(B->dtype, r3d_dim(B, 1));   /* the row in bytes (K/4 for U2, K/2
                                                             * for U4), as the joined entry carries
                                                             * it; integer byte-per-element math
                                                             * truncates U2 to zero */
        (void)ce;
    }
    if (!se || !ze) return R3D_NO(RAD_E_DTYPE);
    if (se->dtype != RAD_F16)
        return R3D_NO(RAD_E_DTYPE);   /* the zero-point grid's scale is f16 */
    if (ze->dtype != RAD_U8)
        return R3D_NO(RAD_E_DTYPE);   /* a sub-byte zero plane would need a bit extract
                                       * (rad_load_bits), which the m16 row also declines */
    if (se->rows != N || se->cols != G || ze->rows != N || ze->cols != G)
        return R3D_NO(RAD_E_SHAPE);   /* {1, group}: scale rows are weight rows */
    if ((se->row_bytes & 1) || se->row_bytes < 2 * G || ze->row_bytes < G)
        return R3D_NO(RAD_E_STRIDE);
    if (bh) {
        if (ce->rows != N || ce->cols != K) return R3D_NO(RAD_E_SHAPE);
        if (ce->row_bytes < c_row) return R3D_NO(RAD_E_STRIDE);
    } else if (r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);

    long long a_rows = 0;
    const long long a_ld = r3d_rows_pitch(A, K, &a_rows);
    if (a_ld < K || (a_ld & 3)) return R3D_NO(RAD_E_STRIDE);  /* dword reads in step order */
    long long yrows = 0;
    const long long y_ld = r3d_rows_pitch(Y, N, &yrows);
    if (y_ld < N) return R3D_NO(RAD_E_STRIDE);
    if (yrows > 0 && yrows < M) return R3D_NO(RAD_E_STRIDE);
    const long long as_ld = 1;                               /* a_scale is rank 1: one per row */
    sv = se ? (const char*)sp : nullptr;
    zv = ze ? (const char*)zp : nullptr;
    s_row = se->row_bytes; z_row = ze->row_bytes;
    const long long bvc = codes_joined
                        ? (long long)(uintptr_t)((const char*)bbase + ce->offset)  /* codes plane */
                        : (long long)(uintptr_t)r3d_p(B);

    if (!r3d_aligned(r3d_p(A), 8) || !r3d_aligned(r3d_p(Y), 8) ||
        !r3d_aligned(sv, 2) || !r3d_aligned(zv, 1)) return R3D_NO(RAD_E_ALIGN);

    const int nt = 0;   /* no `nt` axis in the row's tune table: the non-temporal stream is fixed off */
    const int st = launch(r3d_addr(A), r3d_addr(AS), bvc, (long)(uintptr_t)sv, r3d_addr(Y),
                          (int)M, (int)N, (int)K, (int)group, (int)a_ld, c_row, (int)as_ld,
                          s_row, (int)y_ld, nt, (long)(uintptr_t)s, /*sdt*/ 1, cdt,
                          (long)(uintptr_t)zv, z_row, /*srows*/ 1, /*arm*/ 1);
    if (st == -1) return R3D_NO(RAD_E_INVAL);
    if (st == -2) return R3D_NO(RAD_E_SHAPE);
    if (st == -3) return R3D_NO(RAD_E_STRIDE);
    if (st == -5) return R3D_NO(RAD_E_ALIGN);
    if (st == -4) return R3D_NO(RAD_E_DEVICE);
    return RAD_OK;
}

/* The row's impl axis picks the decode: the table (0) and the ALU formula (1). The _alu entry
 * point stays for a checker that names it directly. */
extern "C" int r3d_rad_gemm_nt_q_fp8_tiled(const RadArgs* a, RadStream s)
{
    R3dGemmCfg c;
    r3d_cfg_gemm(a, &c);
    return r3d_fp8a8_tiled_dispatch(a, s, c.impl);
}
extern "C" int r3d_rad_gemm_nt_q_fp8_tiled_alu(const RadArgs* a, RadStream s)
{ return r3d_fp8a8_tiled_dispatch(a, s, 1); }
