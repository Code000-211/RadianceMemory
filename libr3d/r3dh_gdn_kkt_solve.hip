/* Ported from libhalo/halo_gdn_kkt_solve.hip, verbatim but for the halo_ -> r3dh_ rename, so libr3d builds and loads
 * on its own. libhalo's gated-delta-net kernels are libr4d's ported to the gfx11 wave matrix and
 * checked against libref on gfx1151 (rad-kbench); they compile for gfx11-generic unchanged. */
/* Derived from libr4d/r4d_gdn_kkt_solve_k128_c64_bf16.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// r3dh_gdn_kkt_solve_k128_c64_bf16.hip -- the gated-delta-net chunk preamble in one kernel:
//
//     A = (I + strict_lower(diag(beta) . K K^T . e^{g_i - g_j}))^{-1}     per chunk of 64
//
// Replaces FLA's chunk_scaled_dot_kkt_fwd + solve_tril, whose 64x64 fp32 intermediate per
// (chunk, head) round-trips through HBM. Here the gram stays in LDS and only the bf16 inverse is
// stored. Also:
//
//   * The gram is computed once per k head, not per v head: K K^T does not depend on beta or g,
//     so one workgroup owns a (chunk, k head) and applies each v head's scales to it.
//   * The gram is computed as its upper-triangle tiles. K K^T is symmetric, and read transposed
//     into A's strictly lower half each accumulator holds eight consecutive columns of one row:
//     one 16-byte LDS store instead of eight 2-byte ones.
//
// The decay is applied per element as e^{g_i-g_j}, not split about a per-chunk reference c: each
// half of e^{g_i-c} * e^{c-g_j} can overflow fp32 on its own. Here the exponent is always <= 0
// (g decreases along the chunk and only i > j is kept), and the eight exps per tile are cheap in
// a memory-bound kernel.
//
// The inverse is blocked 16 -> 32 -> 64. The four diagonal blocks use fp32 forward substitution
// (one wave each, one lane per column), cheaper than a WMMA Neumann series at this size and exact
// in fp32, as in FLA. The 32-merges and the 64-merge are WMMA, using
//
//     [[P, 0], [R, Q]]^-1 = [[P^-1, 0], [-Q^-1 R P^-1, Q^-1]]
//
// mma(X, Y) computes X . Y^T from two k-contiguous operands, so a left operand is row-major and a
// right operand is stored transposed. An accumulator writes its transpose contiguously, so the
// intermediates are staged transposed; the final inverse is assembled row-major for the HBM store.
//
// ---- gfx1151 port notes (everything below the gram loop is unchanged in arithmetic) -----------
// A gfx11 accumulator element e is D[2e + (lane>>4)][lane & 15], not gfx12's D[8*(lane>>4) + e]
// [lane & 15], so NO write-out here can keep libr4d's store arithmetic (r3dh_gdn_wmma.h). Two
// cases, and the kernel is split between them by where the value has to land:
//
//   * a TRANSPOSED stage (TMP, and the gram's slice of A) -- eight consecutive rows of one column.
//     accPack() reassembles exactly that from the half-pair, so the store is libr4d's (dst[lo]
//     [8*hi]) with the eight values coming from accPack instead of f2bf4. The per-element scale
//     and mask must be applied BEFORE the exchange: the two halves scale different rows.
//   * a ROW-MAJOR stage into A (the two merge results), eight consecutive columns of one row.
//     accPack() of the gfx11 accumulator gives rows, so the merge is taken with its two operands
//     swapped -- mma is X.Y^T, so swapping them transposes the result and accPack then hands out
//     C[lo][8*hi + 0..7]. Same two fragments, read on the other side.
#include <hip/hip_runtime.h>
#include "r3dh.h"
#include "r3dh_common.h"   // lds_barrier
#include "r3dh_gdn_args.h"
#include "r3dh_gdn_wmma.h"

// libr4d's rounding helpers (r4d_gdn_wmma.h) -- f2bf / bf2f -- are r3dh_gdn_wmma.h's (included
// above): f2bf is round-half-AWAY, one add, which is what makes this kernel bit-comparable with the
// libr4d path. gfx1151 has no v_cvt_pk_bf16_f32 either.

// One accumulator stage: the eight elements accPack() assembles, as two 8-byte stores. The buffers
// here are only 8-byte aligned (AP = 68 shorts), so a 16-byte store would be misaligned on odd rows.
// accPack() packs with halo::pk_bf16, i.e. round-to-nearest-even, where libr4d's f2bf4 rounds half
// away; the two differ only on an exact bf16 tie, and these are inverse intermediates. Both operand
// shapes are offered so a call site reads the same whether it holds a v8f or a float[8].
__device__ __forceinline__ void accSt(unsigned short* dst, v8f v, int lane) {
  union { v8f f; float s[8]; } u;
  u.f = v;
  const uint4 w = accPack(u.s, lane);
  *(uint2*)dst       = make_uint2(w.x, w.y);
  *(uint2*)(dst + 4) = make_uint2(w.z, w.w);
}
__device__ __forceinline__ void accSt(unsigned short* dst, const float v[8], int lane) {
  const uint4 w = accPack(v, lane);
  *(uint2*)dst       = make_uint2(w.x, w.y);
  *(uint2*)(dst + 4) = make_uint2(w.z, w.w);
}

#define BT   64                 // chunk
#define KD  128                 // head_k
#define NBK   4                 // 16x16 blocks per chunk edge
// Row pitches are multiples of 8 shorts (16 B): r3dh_gdn_wmma.h's fragP reads a lane's whole 16-wide
// k run as two 16-byte LDS loads, so a buffer whose pitch is 136 B or 264 B (libr4d's AP = 68 / TMP
// = 36) puts every odd row 8 bytes off a 16-byte boundary, and a misaligned ds_load_b128 returns
// garbage. See r3dh_gdn_chunk_scan.hip's note: measured there as infs in a partial chunk.
#define AP   72                 // row pitch of the [BT][BT] buffers, in shorts (144 B), 16-byte
                                // aligned for the paired stores; 16 rows still cover every bank
#define TMPP 40                 // row pitch of the 32x32 transposed scratch, in shorts (80 B)
#define NW    4
#define NTHR (NW * 32)
#define NTILE 10                // upper-triangular tiles of a 4x4 tiling
#define MAXT  ((NTILE + NW - 1) / NW)

// LDS limits occupancy, and this kernel is memory-bound, so buffers are minimised. No transposed
// inverse is kept (fragT rebuilds the three right operands that need it). The inverse is formed
// in place in A, since each stage reads the block it overwrites before writing it, with barriers
// between stages:
//   diagonal p    reads A[p][p], writes it after the substitution has finished reading
//   32-merge      reads A10 and the two diagonal inverses, writes A10
//   64-merge      reads R = A[32:,:32] into TMP everywhere, barrier, then writes it
// One 64x64 buffer plus a 32x32 scratch: 11 KB instead of 31 KB.
struct KkSmem {
  unsigned short A[BT][AP];     // A on the way in, its inverse on the way out
  unsigned short TMP[32][TMPP];   // one 32x32 intermediate, transposed
  float gb[BT], bt[BT];         // the chunk's gate cumsum and beta, per token
};

// (mt, nt) of upper-triangular tile ti, computed with wave-uniform arithmetic instead of a
// __constant__ table load.
__device__ __forceinline__ void kkt_tile(int ti, int* mt, int* nt) {
  int m = 0, n = ti;
  while (n >= NBK - m) { n -= (NBK - m); ++m; }
  *mt = m; *nt = n + m;
}

// No occupancy hint: at HG_RATIO=3 the kernel uses 190 VGPRs (8 waves per SIMD; LDS would allow
// 10), and forcing 10 waves spills 148 bytes per lane and is ~57% slower.
template <int HG_RATIO>
__global__
__launch_bounds__(NTHR)
void r3dh_gdn_kkt_solve_kernel(const unsigned short* __restrict__ k,
                              const float* __restrict__ beta,
                              const float* __restrict__ g,
                              unsigned short* __restrict__ Ao,
                              const int* __restrict__ cu,
                              int N, int H, int Hg)
{
  __shared__ KkSmem s;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int lo = lane & 15, hi = lane >> 4;
  const int hg = blockIdx.y;

  // Resolve the flat chunk id to (sequence, chunk). The grid is an upper bound on the chunk
  // count, so surplus workgroups exit here; this saves the host a device-to-host sync to size
  // the grid exactly.
  int n = -1, t0 = 0, bos = 0, rows = 0;
  {
    int acc = 0, want = blockIdx.x;
    for (int i = 0; i < N; ++i) {
      const int b = cu[i], e = cu[i + 1];
      const int nc = (e - b + BT - 1) / BT;
      if (want < acc + nc) { n = i; bos = b; t0 = (want - acc) * BT;
                             rows = min(BT, e - b - t0); break; }
      acc += nc;
    }
    if (n < 0) return;
  }

  // ---- zero A once: the parts no store reaches (the upper tiles) are the same for every head.
  // 16-byte stores (the buffer is 8704 bytes, 16-byte aligned).
  for (int i = tid; i < BT * AP / 8; i += NTHR) ((uint4*)&s.A[0][0])[i] = make_uint4(0, 0, 0, 0);

  // ---- the gram, once for all HG_RATIO v heads of this k head -----------------------------
  const unsigned short* kp = k + ((size_t)(bos + t0) * Hg + hg) * KD;
  const unsigned krow = (unsigned)(Hg * KD);
  v8f P[MAXT];
#pragma unroll
  for (int t = 0; t < MAXT; ++t) P[t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int t = 0; t < MAXT; ++t) {
    const int ti = warp + t * NW;
    if (ti < NTILE) {
      int mt, nt; kkt_tile(ti, &mt, &nt);
      // One 32-bit offset per operand row, hoisted out of the k loop (see rowGo/fragGo).
      const unsigned om = rowGo(krow, mt * 16, rows - 1, lane);
      const unsigned on = rowGo(krow, nt * 16, rows - 1, lane);
#pragma unroll
      for (int sk = 0; sk < KD / 16; ++sk)
        P[t] = mma(fragGo(kp, om + sk * 16), fragGo(kp, on + sk * 16), P[t]);
    }
  }

  for (int hh = 0; hh < HG_RATIO; ++hh) {
    const int hv = hg * HG_RATIO + hh;

    // ---- per-head gate and beta for this chunk
    if (tid < BT) {
      const int t = min(tid, rows - 1);
      s.gb[tid] = g[(size_t)(bos + t0 + t) * H + hv];
      s.bt[tid] = beta[(size_t)(bos + t0 + t) * H + hv];
    }
    lds_barrier();

    // ---- scale, mask and place the gram as the strictly lower half of A ---------------------
    // Tile (mt, nt) holds P[m][n]; the element written is A[n][m], in the lower triangle when
    // n > m. n is fixed per lane and m runs over the eight accumulator elements: one row of A,
    // one 8-short store.
    //
    // gfx1151: element e is m = mt*16 + 2e + hi, not mt*16 + 8hi + e, so the eight columns this
    // lane owns are two-apart and interleaved with its half-partner's. Each element is scaled by
    // its OWN exponent first, and accPack() then reassembles the 16 consecutive columns of row i
    // out of the half-pair: this lane keeps columns mt*16 + 8*hi .. +7, as on gfx12.
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
      const int ti = warp + t * NW;
      if (ti < NTILE) {
        int mt, nt; kkt_tile(ti, &mt, &nt);
        const int i = nt * 16 + lo;                      // row of A
        const int j0 = mt * 16 + 8 * hi;                 // first column of A, after the exchange
        const bool live = (i < rows);
        const float bi = live ? s.bt[i] : 0.0f;
        const float gi = live ? s.gb[i] : 0.0f;
        float vv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int j = mt * 16 + 2 * e + hi;
          // For a live row, i > j gives g_i <= g_j, so the exponent is never positive. Padding
          // rows and columns must be masked to -inf: with gi = 0 the exponent -gb[j] is large and
          // positive, 0 * inf = NaN, and the WMMA merges in the blocked inverse would spread it
          // into live rows.
          const float d = (live && i > j && j < rows) ? (gi - s.gb[j]) : -INFINITY;
          vv[e] = bi * __expf(d) * P[t][e];
        }
        accSt(&s.A[i][j0], vv, lane);
      }
    }
    lds_barrier();

    // ---- 1. the four diagonal blocks, by fp32 forward substitution --------------------------
    // D = (I + N)^-1 solves D = I - N D row by row. Lane l owns column l of D, so the dot product
    // over j < i is in 16 registers and N[i][j] is a broadcast LDS read.
    for (int p = warp; p < NBK; p += NW) {
      const int b0 = p * 16;
      float d[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (j < i) acc += bf2f(s.A[b0 + i][b0 + j]) * d[j];
        d[i] = ((lo == i) ? 1.0f : 0.0f) - acc;
      }
      if (lane < 16) {
#pragma unroll
        for (int i = 0; i < 16; ++i) s.A[b0 + i][b0 + lo] = f2bf(d[i]);
      }
    }
    lds_barrier();

    // ---- 2. the two 32-merges: M10 = -D1 A10 D0 --------------------------------------------
    for (int blk = warp; blk < 2; blk += NW) {
      const int r0 = blk * 32;                            // 32-block origin
      // X = A10 . D0     (left A10 row-major, right D0 transposed = MT)
      v8f X = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
      X = mma(frag(&s.A[0][0], AP, r0 + 16, r0, lane),
              fragT(&s.A[0][0], AP, r0, r0, lane), X);
      // Stage X transposed for the next product; each of the two merges owns 16 rows of TMP.
      accSt(&s.TMP[blk * 16 + lo][8 * hi], X, lane);
      // Same wave writes and reads TMP here, and LDS accesses from one wave complete in order.
      __builtin_amdgcn_wave_barrier();
      // M10 = -D1 . X    (left D1 row-major, in place in A; right X transposed, in TMP)
      // Operands swapped on gfx1151: this accumulates Y^T, so accPack() below hands out eight
      // consecutive rows of Y -- which is a row-major store into A. TMP already holds X^T (what
      // libr4d stages), which is this product's LEFT operand here, and D1 the right one.
      v8f Y = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
      Y = mma(frag(&s.TMP[0][0], TMPP, blk * 16, 0, lane),
              frag(&s.A[0][0], AP, r0 + 16, r0 + 16, lane), Y);
      float yv[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) yv[e] = -Y[e];
      accSt(&s.A[r0 + 16 + lo][r0 + 8 * hi], yv, lane);   // rows of M10, 8 columns each
    }
    lds_barrier();

    // ---- 3. the 64-merge: M[32:,:32] = -Q^-1 (R P^-1) --------------------------------------
    // Y = R . P^-1, R = A[32:,:32] row-major, P^-1 read transposed out of A[:32,:32] by fragT
    for (int t = warp; t < 4; t += NW) {
      const int mt = t >> 1, nt = t & 1;
      v8f Y = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int sk = 0; sk < 2; ++sk)
        Y = mma(frag(&s.A[0][0], AP, 32 + mt * 16, sk * 16, lane),
                fragT(&s.A[0][0], AP, nt * 16, sk * 16, lane), Y);
      accSt(&s.TMP[nt * 16 + lo][mt * 16 + 8 * hi], Y, lane);
    }
    lds_barrier();                  // every wave has read R into TMP; now it may be overwritten
    // Z = -Q^-1 . Y, Q^-1 = A[32:,32:] row-major, Y transposed = TMP
    // Swapped, as M10 above: this accumulates Z^T so accPack() gives a row-major store. Y^T sits
    // in TMP ([col][row]) and is the left operand here; Q^-1's rows are the right one.
    for (int t = warp; t < 4; t += NW) {
      const int mt = t >> 1, nt = t & 1;
      v8f Z = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int sk = 0; sk < 2; ++sk)
        Z = mma(frag(&s.TMP[0][0], TMPP, nt * 16, sk * 16, lane),
                frag(&s.A[0][0], AP, 32 + mt * 16, 32 + sk * 16, lane), Z);
      float zv[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) zv[e] = -Z[e];
      accSt(&s.A[32 + mt * 16 + lo][nt * 16 + 8 * hi], zv, lane);
    }
    lds_barrier();

    // ---- store: one row of the inverse is 64 bf16, contiguous in HBM -----------------------
    {
      unsigned short* op = Ao + ((size_t)(bos + t0) * H + hv) * BT;
      const size_t orow = (size_t)H * BT;
      for (int r = tid / 8; r < rows; r += NTHR / 8) {
        const int c8 = (tid & 7) * 8;
        *(uint4*)(op + (size_t)r * orow + c8) = *(const uint4*)(&s.A[r][c8]);
      }
    }
    if (hh + 1 < HG_RATIO) lds_barrier();
  }
}

extern "C" int r3dh_gdn_kkt_solve_k128_c64_bf16(const void* k, const void* beta, const void* g,
                                               void* A, const void* cu, int N, int T, int H,
                                               int Hg, int K, int bt, void* stream)
{
  if (K != KD || bt != BT) return -1;
  if (H % Hg) return -2;
  const int nchunk_ub = (T + BT - 1) / BT + N;   // sum of per-sequence chunk counts, bounded
  dim3 grid(nchunk_ub, Hg), blk(NTHR);
#define KK_LAUNCH(R)                                                                          \
  hipLaunchKernelGGL((r3dh_gdn_kkt_solve_kernel<R>), grid, blk, 0, (hipStream_t)stream,        \
                     (const unsigned short*)k, (const float*)beta, (const float*)g,           \
                     (unsigned short*)A, (const int*)cu, N, H, Hg)
  switch (H / Hg) {
    case 1: KK_LAUNCH(1); break;
    case 2: KK_LAUNCH(2); break;
    case 3: KK_LAUNCH(3); break;
    case 4: KK_LAUNCH(4); break;
    case 6: KK_LAUNCH(6); break;
    case 8: KK_LAUNCH(8); break;
    default: return -3;
  }
#undef KK_LAUNCH
  return (int)hipGetLastError();
}

// ============================================================== the plugin ABI entry points
// Shared operand helpers: r3dh_gdn_args.h.
/* ------------------------------------------------------------------ kkt solve */

enum { KS_K = 0, KS_BETA, KS_G, KS_CU, KS_A };

extern "C" int r3dh_rad_gdn_kkt_solve(const RadArgs* a, RadStream s) {
    if (gdn_dims_ok() != RAD_OK) return R3DH_NO(RAD_E_SHAPE);
    const RadTensor* k  = r3dh_opt(a, KS_K);
    const RadTensor* bt = r3dh_opt(a, KS_BETA);
    const RadTensor* g  = r3dh_opt(a, KS_G);
    const RadTensor* cu = r3dh_opt(a, KS_CU);
    const RadTensor* A  = r3dh_opt(a, KS_A);
    if (!k || !bt || !g || !cu || !A) return R3DH_NO(RAD_E_INVAL);
    if (k->rank != 3 || bt->rank != 2 || g->rank != 2 || A->rank != 3) return R3DH_NO(RAD_E_SHAPE);
    if (!r3dh_contig(k) || !r3dh_contig(bt) || !r3dh_contig(g) || !r3dh_contig(A))
        return R3DH_NO(RAD_E_STRIDE);
    /* bf16 k and A (the output, written as bf16); fp32 beta and gate. */
    R3DH_DT(k, RAD_BF16); R3DH_DT(A, RAD_BF16); R3DH_DT(bt, RAD_F32); R3DH_DT(g, RAD_F32);

    int N = 0;
    int rc = seqs_of(cu, &N);
    if (rc != RAD_OK) return rc;

    const long long T  = k->shape[0];
    const long long Hg = k->shape[1];
    const long long K  = k->shape[2];
    const long long H  = bt->shape[1];
    const long long chunk = A->shape[2];
    if (g->shape[0] != T || g->shape[1] != H || bt->shape[0] != T) return R3DH_NO(RAD_E_SHAPE);
    if (A->shape[0] != T || A->shape[1] != H) return R3DH_NO(RAD_E_SHAPE);
    if (H == 0 || Hg == 0 || H % Hg) return R3DH_NO(RAD_E_SHAPE);

    return R3DH_RAW(r3dh_gdn_kkt_solve_k128_c64_bf16(k->data, bt->data, g->data, A->data,
                                                   cu->data, N, (int)T, (int)H, (int)Hg, (int)K,
                                                   (int)chunk, s));
}

