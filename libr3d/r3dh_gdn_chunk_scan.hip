/* Ported from libhalo/halo_gdn_chunk_scan.hip, verbatim but for the halo_ -> r3dh_ rename, so libr3d builds and loads
 * on its own. libhalo's gated-delta-net kernels are libr4d's ported to the gfx11 wave matrix and
 * checked against libref on gfx1151 (rad-kbench); they compile for gfx11-generic unchanged. */
/* Derived from libr4d/r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// r3dh_gdn: the gated-delta-net chunked scan, gfx1201 WMMA. Entry point
// r3dh_gdn_chunk_scan_k128_v128_c64_bf16 -- head_k 128, head_v 128, chunk 64, bf16 in.
// Used by prefill and chunked prefill; decode uses the recurrent update kernel. Replaces FLA's
// recompute_w_u_fwd + chunk_gated_delta_rule_fwd_h + chunk_fwd_o: the recurrent state stays in
// WMMA accumulators across the chunk loop, so h / w / u / v_new never touch HBM.
//
// Invariant: every decay factor formed is <= 1. The intra-chunk gate span reaches 558 nats on
// real Qwen3.8 prefill activations, so splitting the exponent about a per-chunk reference c
// (e^{g_i-c} * e^{c-g_j}) overflows fp32 on one side and underflows on the other for any c.
// e^{g_i-g_j} (j <= i) and e^{g_last-g_j} stay in (0, 1].
//
//   ge[j]=e^{g_j}   gg[j]=g_j   gs[j]=e^{g_last-g_j}
//   U  = K@S            W  = beta*(V - ge*U)      V' = A@W          (undecayed)
//   P  = Q@K^T          sA = scale*e^{g_i-g_j}*tril(P)
//   O  = scale*ge*(Q@S) + sA@V'
//   S  = e^{g_last}*S + V'^T@(gs*K)
// A is FLA's solve_tril output (decay folded in), so no decay enters W or V'. The (i,j) weight
// is applied per element in the score step (eight __expf per tile), and the state's
// e^{g_last-g_j} is applied to K, which is otherwise dead by then. V' stays undecayed because its
// consumers need different references (e^{g_i-g_j} for the output, e^{g_last-g_j} for the state),
// and serving both from one scaling would need the unsafe split.
#include <hip/hip_runtime.h>
#include "r3dh.h"
#include "r3dh_common.h"
#include "r3dh_gdn_args.h"
#include "r3dh_gdn_wmma.h"
// Barriers are lds_barrier(): __syncthreads() fences all address spaces, so the barrier after the
// global O store would wait on s_wait_storecnt_dscnt 0x0.
//
// ---- gfx1151 port notes ---------------------------------------------------------------------
// r3dh_gdn_wmma.h carries the fragment layer, so the *gathers* port by themselves: rowP/rowG lose
// libr4d's `4*(lane>>4)` k term and fragP reads the lane's whole 16-wide k run. What does NOT port
// is every accumulator write-out: on gfx11 element e is D[2e + (lane>>4)][lane & 15], not
// D[8*(lane>>4) + e][lane & 15], so
//
//   * a transposed stage (s.S, s.A's score tile, s.W, s.D's V') keeps libr4d's destination and
//     takes its eight values from accPackX(), after scaling them per element -- the two halves of
//     the wave own different rows, so nothing can be scaled after the exchange;
//   * O needs nothing: its hardware row is the v dim, so accPackX()'s eight consecutive rows are
//     eight consecutive v of one token -- the contiguous HBM direction -- and both O loops, the
//     per-lane scale and the store stay exactly as libr4d wrote them.
// The per-lane state (St[]) is the one place the wider fragment scatters: a lane's eight k values
// are two apart, so h0/ht go through an f32 LDS tile (see SF) rather than straight to HBM.

// libr4d's rounding helpers (r4d_gdn_wmma.h) -- f2bf / f2bf2 / bf2f -- are r3dh_gdn_wmma.h's
// (included above): f2bf is round-half-AWAY, one add, so the bf16 stages keep libr4d's bits.
// gfx1151 has no v_cvt_pk_bf16_f32 either.

// One accumulator stage: the eight elements accPackX() assembles, as two 8-byte stores (every
// buffer here is only 8-byte aligned, so a 16-byte store would be misaligned on odd rows).
// accPackX() packs with halo::pk_bf16's round-to-nearest-even, as accPack() does, where libr4d's
// f2bf4 rounds half away; the two differ only on an exact bf16 tie. It is accPack()'s values
// without its ds_bpermute exchange and per-pack branches (see r3dh_gdn_wmma.h). Both operand
// shapes are offered so a call site reads the same whether it holds a v8f or a float[8].
__device__ __forceinline__ uint4 accPk(v8f v, int lane) {
  union { v8f f; float s[8]; } u;
  u.f = v;
  return accPackX(u.s, lane);
}
__device__ __forceinline__ void accSt(unsigned short* dst, v8f v, int lane) {
  const uint4 w = accPk(v, lane);
  *(uint2*)dst       = make_uint2(w.x, w.y);
  *(uint2*)(dst + 4) = make_uint2(w.z, w.w);
}
__device__ __forceinline__ void accSt(unsigned short* dst, const float v[8], int lane) {
  const uint4 w = accPackX(v, lane);
  *(uint2*)dst       = make_uint2(w.x, w.y);
  *(uint2*)(dst + 4) = make_uint2(w.z, w.w);
}

#define BT   64
#define KD  128
#define VD  128
#define BV   64                 // V per workgroup; also halves the Q@K^T redundancy
// Row pitches are multiples of 8 shorts (16 B) because r3dh_gdn_wmma.h's fragP reads a lane's whole
// 16-wide k run as two 16-byte LDS loads. They used KP = 132 / AP = 68 on gfx12, where a fragment is
// two 8-byte reads 16 bytes apart and 8-byte alignment suffices; on gfx1151 a buffer whose pitch is
// 136 B or 264 B puts every odd row 8 bytes off a 16-byte boundary, and a misaligned ds_load_b128
// returns garbage -- silently, and only for some register allocations (measured: infs in a
// trailing partial chunk with the gfx12 pitches, none with these).
#define KP  136                 // row pitch of the [.][KD] buffers, in shorts (272 B)
#define AP   72                 // row pitch of the [.][BT] buffers, in shorts (144 B)
#define WP  AP
// Waves per workgroup. The [4 token][4 v] tile grid is fixed by BT and BV, so 8 waves carry two
// tiles each.
#define NW    8
#define NTHR (NW * 32)
#define NTV   2                 // n tiles per wave in the main / V' / O loops
// A wave's slice of the 8 k-tile x 4 v-tile state grid is 2 x 2.
#define SKT   2
#define SVT   2
#define NST (SKT * SVT)         // state tiles per wave

// gfx1151 register plan. libr4d's plan (state, k/v prefetch, six main-loop accumulators) fits
// gfx12's 4-VGPR fragment; ported as is, with gfx11's 8-VGPR fragment, every instantiation sat at
// 256 VGPRs with 357-379 spilled (1.3 KB of scratch a lane: loop-invariant LDS addresses -- fragT's
// sixteen per-row addresses for the transposed K above all -- hoisted out of the chunk loop and
// reloaded inside it), and ~100 us a chunk. Measured on gfx1151 (kbench, M=512, the two fixture
// head counts: 820 / 493 us before, 155 / 118 after), what replaced it:
//   * the state update reads gs*K from a transposed copy, s.KT, written by the pass that used to
//     scale K in place, so every fragment in the kernel is a fragP. s.KT shares storage with the
//     bf16 state s.S, whose last reader (the main loop) retires at barrier (2); the state is
//     restaged at the top of the next chunk, after barrier (4) has retired the s.KT reads, so the
//     alias costs no barrier.
//   * every per-lane LDS address is one base plus immediates (a wave-uniform term goes in the
//     base, never in a per-element index), so nothing is left for LICM to hoist.
//   * no divergent branch on the chunk path: the accumulator write-outs use accPackX() (accPack()
//     lowers to ds_bpermute + an LDS wait + a branch per pack), the score mask is a select after
//     the exp, and the prefetch's zero fill is a select at consumption (-12% together).
//   * the triangular work stops at the diagonal (P^T tiles, V' and O += sA@V' k steps): 544
//     WMMAs a chunk instead of 640, and the kernel is WMMA-throughput bound now (dropping P^T
//     entirely saved its full share).
//   * the first q fragments and the V' A fragments are issued ahead of the barrier that precedes
//     their loop, so the first WMMAs do not wait out an HBM round trip (-13% / -4%).
//   * h0 / ht pass through LDS (see SF): with element-wise HBM access the state traffic was ~45%
//     of a one-chunk call.
// The result is 239-248 VGPRs, no spills, with ROCm 7.2.3 and 7.2.4. The margin is not wide: the
// main loop with its P^T count fixed (one all-tiles body instead of the three below) spills ~150.
// 16 waves of one tile each (<= 192 VGPRs, so two workgroups per WGP still fit) is ~12% slower.
struct Smem {
  unsigned short k[BT][KP];     // A: K   /  B: K^T for P=Q@K^T
  unsigned short A[BT][AP];     // B-source for sA (written as [i][j])
  union {
    unsigned short S[BV][KP];   // B: state, V-major (matches FLA's [V,K] layout); main loop only
    unsigned short KT[KD][AP];  // A: gs*K, transposed ([k][token]); state update only
  };
  unsigned short W[BV][WP];     // B: W (transposed)
  unsigned short D[BV][AP];     // staged V as [token][v]; then V' as [v][token].
                                // V dies in the W update; V' is written after barrier (2).
  float ge[BT], gv[BT], gg[BT], bt[BT];   // gv = gs, gg = raw g
};

// Choices measured against their alternatives on gfx1201, kept so they are not undone:
//   - LDS rows in the main loop are opaque 32-bit byte offsets (fragO), global q / A fragments a
//     32-bit offset off a uniform base (fragGo).
//   - k and v are staged eight shorts per lane (global_load_b128), issued as a block before the
//     main loop: interleaving them into it is ~5% slower.
//   - Every bf16 pack rounds (f2bf2 / accPackX). Truncating W, sA and V' is ~2% faster but
//     roughly triples the fp64-oracle error, since they feed the recurrence.
// And on gfx1151: prefetching the gates (g, beta, g_last) with k / v, and sched_barrier fences
// in the main loop, measured neutral; more q fragments ahead of barrier (1) than QP spill.
// Byte offset, from the base of the shared block, of the fragment row this lane reads out of
// buf[base + (lane&15)][...]. r3dh_gdn_wmma.h's rowO is libr4d's ROWO without the `4*hi` k term:
// on gfx1151 the half-wave no longer selects which quarter of the k run a lane holds.
#define ROWO(buf, pitch, base) \
  rowO((unsigned)((const char*)&(buf)[0][0] - (const char*)&s), (pitch), (base), lane)

// SDT is the state cache's element type (0 = fp32, 1 = fp16) and nothing else; all arithmetic is
// fp32. The cache is shared with the recurrent update (whose time is the state traffic, so it
// favours fp16) and allocated from vLLM's --mamba-ssm-cache-dtype, so both kernels must agree.
template <int SDT> struct cs_state { using T = float; };
template <> struct cs_state<1>     { using T = _Float16; };

// H0 / HT: whether the optional initial and final state operands are present (template parameters,
// so the common both-present form has no runtime test). As in libref, no initial state starts from
// zero and no final state stores nothing.
template <int SDT, bool H0 = true, bool HT = true>
__global__ __launch_bounds__(NTHR) void r3dh_gdn_chunk_scan_kernel(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, const unsigned short* __restrict__ Amat,
    const float* __restrict__ gcs, const float* __restrict__ beta,
    const typename cs_state<SDT>::T* __restrict__ h0, unsigned short* __restrict__ o,
    typename cs_state<SDT>::T* __restrict__ ht, const int* __restrict__ cu,
    const int* __restrict__ sidx, long sidx_stride, long st_slot, long st_head,
    int H, int Hg, float scale)
{
  __shared__ Smem s;
  const int bv = blockIdx.x, hv = blockIdx.y, nq = blockIdx.z;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int hq = hv / (H / Hg), v0 = bv * BV;
  const int tb = cu[nq], te = cu[nq + 1];
  // The state slot comes from the cache manager (sidx), not the batch position; without sidx the
  // kernel uses state[nq], valid only while slots match batch order. A row may name several slots
  // (speculative decode adds scratch slots); column 0 is the state and the only one used here.
  const long si = sidx ? (long)sidx[(size_t)nq * sidx_stride] : (long)nq;
  if (si < 0) return;                                        // a padded row owns no state
  const size_t st_base = (size_t)si * st_slot + (size_t)hv * st_head;
  const int nchunk = (te - tb + BT - 1) / BT;
  const int lo = lane & 15, hi = lane >> 4;

  // [BT][BV]: 2 tiles/wave. The token tile mt sets the wave's share of the triangular work (P^T,
  // V', O += sA@V' all stop at the diagonal), so waves w and w+4 -- likely one SIMD -- pair
  // mt with 3 - mt, and the n-tile pair is chosen to even out the P^T tiles (3, 2, 3, 2).
  const int mt = (w >> 1) ^ (w >> 2), ntb = ((w ^ (w >> 2)) & 1) * 2;
  // The state accumulator is held transposed, C[m=k-dim][n=v]. A and B fragments share a lane
  // layout, so swapping the state update's operands is free, and k-dim becomes the per-lane
  // contiguous axis: the per-chunk writeback to s.S is four 8-short stores instead of 32 scalar.
  const int kt0 = (w >> 1) * 2, vt0 = (w & 1) * 2;           // 2 x 2

#define PFW 8                   // shorts staged per lane per step
#define NKB (BT * KD / (NTHR * PFW))
#define NVB (BT * BV / (NTHR * PFW))
  uint4 kb[NKB], vb[NVB];
  // Staging index, split into a wave-uniform row and a loop-invariant lane column. As
  // `i = tid*4 + u*NTHR*4; r = i>>7` the row looks divergent to LLVM and each load costs ~13
  // instructions of 64-bit address math; readfirstlane makes the row provably uniform (SALU).
  // Identities (exact, since u*NTHR*8 is a multiple of both 128 and 64):
  //   k: r = 2*(tid>>5) + kh + KRU*u   c = (tid*8) & 127   kh = (tid>>4) & 1
  //   v: r = 4*(tid>>5) + vh + VRU*u   c = (tid*8) & 63    vh = (tid>>3) & 3
  const int wu = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int kc = (tid * PFW) & (KD - 1), vc = (tid * PFW) & (BV - 1);
  const int kr0 = 2 * wu + ((tid >> 4) & 1), vr0 = 4 * wu + ((tid >> 3) & 3);
#define KRU (2 * NW)
#define VRU (4 * NW)
  // Do not change: staged rows past the chunk end are zeroed, not clamped to the last live row
  // (~4% faster). A clamp looks harmless (beta and gv are 0 there, the tril mask drops padding j,
  // the O store is guarded), yet with it any prompt whose trailing chunk is not a multiple of 16
  // tokens returns one repeated token. Zeroing k alone fixes it, so the leak is through some
  // consumer of s.k whose path is not yet understood.
  // The load itself is clamped (pnv >= 1 wherever this runs) and the zero selected when the
  // staging store consumes it (PFSEL, against that chunk's nval): straight-line code, and a select
  // at issue would make the prefetch wait for its own loads.
#define PFSTORE(dst, val)                                                            \
  do {                                                                               \
    unsigned short* dp_ = (dst);                                                     \
    *(uint2*)dp_ = make_uint2((val).x, (val).y);                                      \
    *(uint2*)(dp_ + 4) = make_uint2((val).z, (val).w);                                \
  } while (0)
#define PFSEL(r, val)                                                                \
  make_uint4((r) < nval ? (val).x : 0u, (r) < nval ? (val).y : 0u,                   \
             (r) < nval ? (val).z : 0u, (r) < nval ? (val).w : 0u)
#define PREFETCH(CH)                                                                 \
  {                                                                                  \
    const int pc0 = tb + (CH) * BT, pnv = min(BT, te - pc0);                         \
    const unsigned short* kb_ = k + ((size_t)pc0 * Hg + hq) * KD;                   \
    const unsigned short* vb_ = v + ((size_t)pc0 * H + hv) * VD + v0;               \
    _Pragma("unroll") for (int u = 0; u < NKB; ++u) {                                \
      const int r = kr0 + KRU * u, rc = min(r, pnv - 1);                             \
      kb[u] = *(const uint4*)(kb_ + (unsigned)(rc * (Hg * KD)) + kc);                \
    }                                                                                \
    _Pragma("unroll") for (int u = 0; u < NVB; ++u) {                                \
      const int r = vr0 + VRU * u, rc = min(r, pnv - 1);                             \
      vb[u] = *(const uint4*)(vb_ + (unsigned)(rc * (H * VD)) + vc);                 \
    }                                                                                \
  }

  v8f St[NST];                  // state: SKT k-tiles x SVT v-tiles
#define KT(t) (kt0 + (t) / SVT)
#define VT(t) (vt0 + (t) % SVT)
// State element e is v = VT*16 + lo, k = KT*16 + 2e + hi on gfx1151 (NOT 8*hi + e): a lane's eight
// values are two apart in the [v][k] state. Read straight from HBM that is 32 scalar loads a lane,
// each wave load touching sixteen rows for 8 bytes apiece; at one or two chunks the state traffic
// is half the kernel. So h0 and ht pass through LDS (free before the chunk loop and after barrier
// (4)) as an f32 [v][k] tile: coalesced 16-byte global accesses on one side, the accumulator
// layout on the other (pitch FP: lanes lo / hi land on banks 4*lo + hi, no conflict).
#define FP (KD + 4)
  static_assert(sizeof(Smem) >= sizeof(float) * BV * FP, "the f32 state tile overlays Smem");
  float (*const F)[FP] = reinterpret_cast<float (*)[FP]>(&s);
#define SF(t, e) F[VT(t) * 16 + lo][KT(t) * 16 + 2 * (e) + hi]
  if constexpr (!H0) {
#pragma unroll
    for (int t = 0; t < NST; ++t)
#pragma unroll
      for (int e = 0; e < 8; ++e) St[t][e] = 0.0f;
  } else {
    const typename cs_state<SDT>::T* p = h0 + st_base + (size_t)v0 * KD;
    if constexpr (SDT == 0) {             // a row is 32 float4: wave wu takes rows wu + NW*u
      float4 r[BV / NW];
#pragma unroll
      for (int u = 0; u < BV / NW; ++u) r[u] = *(const float4*)(p + (wu + NW * u) * KD + lane * 4);
#pragma unroll
      for (int u = 0; u < BV / NW; ++u) *(float4*)&F[wu + NW * u][lane * 4] = r[u];
    } else {                              // a row is 16 x 8 halves: rows tid/16 + (NTHR/16)*u
      const int r0 = tid >> 4, c = (tid & 15) * 8;
      uint4 r[BV * 16 / NTHR];
#pragma unroll
      for (int u = 0; u < BV * 16 / NTHR; ++u)
        r[u] = *(const uint4*)(p + (r0 + (NTHR / 16) * u) * KD + c);
#pragma unroll
      for (int u = 0; u < BV * 16 / NTHR; ++u) {
        union { uint4 w; _Float16 h[8]; } x;
        x.w = r[u];
        float* d = &F[r0 + (NTHR / 16) * u][c];
        *(float4*)d       = make_float4(x.h[0], x.h[1], x.h[2], x.h[3]);
        *(float4*)(d + 4) = make_float4(x.h[4], x.h[5], x.h[6], x.h[7]);
      }
    }
    lds_barrier();
#pragma unroll
    for (int t = 0; t < NST; ++t)
#pragma unroll
      for (int e = 0; e < 8; ++e) St[t][e] = SF(t, e);
    lds_barrier();                        // the chunk loop's staging overwrites the tile
  }
  if (nchunk > 0) PREFETCH(0);

  const char* const sbase = (const char*)&s;
  // scale goes inside the score exponent, exp(g_i + log(scale) - g_j), saving eight multiplies
  // per tile.
  const float lgs = __logf(scale);
  for (int ci = 0; ci < nchunk; ++ci) {
    const int c0 = tb + ci * BT, nval = min(BT, te - c0);
    const int lim = nval - 1;
    const unsigned short* qg = q + ((size_t)c0 * Hg + hq) * KD;
    const unsigned short* ag = Amat + ((size_t)c0 * H + hv) * BT;
    const size_t qrs = (size_t)Hg * KD, ars = (size_t)H * BT;
    const float gl = gcs[(size_t)(c0 + lim) * H + hv];

    // Issued before the staging stores so the LDS writes cover their latency. Only waves 0-1
    // (tid < BT) run this; the others wait at (1).
    float ggr = 0.f, btr = 0.f;
    if (tid < BT) {
      bool ok = tid < nval;
      ggr = ok ? gcs[(size_t)(c0 + tid) * H + hv] : 0.f;
      btr = ok ? beta[(size_t)(c0 + tid) * H + hv] : 0.f;
    }
    // The main loop's first QP q fragments, issued ahead of the staging and barrier (1) so the
    // first WMMAs do not wait out a full HBM round trip.
#define QP 4
    const unsigned qo = rowGo((unsigned)qrs, mt * 16, lim, lane);
    v16bf aq0[QP];
#pragma unroll
    for (int sk = 0; sk < QP; ++sk) aq0[sk] = fragGo(qg, qo + sk * 16);
    // The bf16 state for this chunk's U and O. s.S aliases s.KT, whose last reads (the previous
    // chunk's state update) retired at barrier (4).
#pragma unroll
    for (int t = 0; t < NST; ++t)
      accSt(&s.S[VT(t) * 16 + lo][KT(t) * 16 + 8 * hi], St[t], lane);
    { // consume the previous iteration's prefetch, issued a whole chunk of compute ago
#pragma unroll
      for (int u = 0; u < NKB; ++u) PFSTORE(&s.k[kr0 + KRU * u][kc], PFSEL(kr0 + KRU * u, kb[u]));
#pragma unroll
      for (int u = 0; u < NVB; ++u) PFSTORE(&s.D[vr0 + VRU * u][vc], PFSEL(vr0 + VRU * u, vb[u]));
    }
    if (tid < BT) {
      bool ok = tid < nval;
      s.bt[tid] = btr;
      s.ge[tid] = __expf(ggr);
      // g is a within-chunk cumsum of log-sigmoid gates, so it is non-increasing and gl is its
      // minimum: gl - ggr <= 0, and gs underflowing to 0 is correct.
      // A padding row gets gl, not 0: it is scored against live j with very negative g_j, and
      // g_i = 0 would give exp(+400) = inf in s.A. With gl (the chunk minimum) the exponent
      // stays in (0, scale]. The row is discarded anyway (O store is guarded).
      s.gg[tid] = ok ? ggr : gl;
      s.gv[tid] = ok ? __expf(gl - ggr) : 0.f;                           // gs
    }
    lds_barrier();                                           // (1)
    if (ci + 1 < nchunk) PREFETCH(ci + 1);   // issue only; consumed next iteration

    // ---- U = K@S, O^T = S@Q^T and P^T = K@Q^T share the q and k fragments.
    // P^T tiles above the diagonal are skipped (sA masks them). On gfx12 that was 2.4% slower --
    // some wave still carries two tiles, so the barrier arrival does not move -- but here the
    // kernel is throughput bound with two workgroups per WGP, so the WMMAs saved are time saved.
    v8f O[NTV], u[NTV], p[NTV];
#pragma unroll
    for (int i = 0; i < NTV; ++i) { O[i] = (v8f){}; u[i] = (v8f){}; p[i] = (v8f){}; }
    {
      const char* sb = sbase;
      // The two n tiles are 16 rows apart, so one base per buffer serves both (an immediate).
      unsigned okm = ROWO(s.k, KP, mt * 16), oSn = ROWO(s.S, KP, ntb * 16),
               okn = ROWO(s.k, KP, ntb * 16);
      LDSOPAQUE(okm); LDSOPAQUE(oSn); LDSOPAQUE(okn);
#define pSnf(i, ko) fragO(sb, oSn + ((i) * 16 * KP + (ko)) * 2u)
#define pkmf(ko)    fragO(sb, okm + (ko) * 2u)
#define pknf(i, ko) fragO(sb, okn + ((i) * 16 * KP + (ko)) * 2u)
#define pqf(ko) ((ko) / 16 < QP ? aq0[min((ko) / 16, QP - 1)] : fragGo(qg, qo + (ko)))
      // P^T tile ntb+i is wholly above the diagonal (masked in sA) when ntb + i > mt: only the
      // first np tiles are computed. np is wave-uniform, so each count gets its own loop.
      auto mainloop = [&](auto npc) {
        constexpr int NP = decltype(npc)::value;
#pragma unroll
        for (int sk = 0; sk < KD / 16; ++sk) {
          const int ko = sk * 16;
          v16bf ak = pkmf(ko), aq = pqf(ko);
#pragma unroll
          for (int i = 0; i < NTV; ++i) {
            v16bf b = pSnf(i, ko);
            u[i] = mma(ak, b, u[i]);                     // U: m = token, n = v-tile
            O[i] = mma(b, aq, O[i]);                     // O^T: m = v-tile, n = token-tile
            if (i < NP) p[i] = mma(pknf(i, ko), aq, p[i]);   // P^T: m = j
          }
        }
      };
      const int np = __builtin_amdgcn_readfirstlane(min(max(mt - ntb + 1, 0), NTV));
      if (np == NTV)  mainloop(std::integral_constant<int, NTV>{});
      else if (np)    mainloop(std::integral_constant<int, 1>{});
      else            mainloop(std::integral_constant<int, 0>{});
    }
    {
      // O's hardware row is the v dim and its column the token, which is the one place gfx1151 and
      // gfx12 agree: accPackX()'s eight consecutive rows are eight consecutive v of one token, so
      // this is libr4d's scale and store unchanged (a lane's token is mt*16 + lo).
      float g = scale * s.ge[mt * 16 + lo];             // token index is constant per lane
#pragma unroll
      for (int i = 0; i < NTV; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) O[i][e] *= g;
    }
    // The V' loop's A fragments come from HBM/L2: issued here, their latency is covered by the
    // W / sA epilogue and barrier (2) instead of stalling the first V' WMMA.
    v16bf af[BT / 16];
    {
      const unsigned ao = rowGo((unsigned)ars, mt * 16, lim, lane);
#pragma unroll
      for (int sk = 0; sk < BT / 16; ++sk) af[sk] = fragGo(ag, ao + sk * 16);
    }
    // W = beta * (V - ge*U)   (beta folded here: (A*beta_col)@W == A@(beta_row*W))
    {
      // Element e is token mt*16 + 2e + hi: one row pointer per lane, e and i are immediates.
      const unsigned short* dv = &s.D[mt * 16 + hi][ntb * 16 + lo];
      const float* bp = &s.bt[mt * 16 + hi];
      const float* gp = &s.ge[mt * 16 + hi];
      float wa[NTV][8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {           // beta / ge read once for both tiles
        const float f = bp[2 * e], g = gp[2 * e];
#pragma unroll
        for (int i = 0; i < NTV; ++i)
          wa[i][e] = f * (bf2f(dv[2 * e * AP + i * 16]) - g * u[i][e]);
      }
#pragma unroll
      for (int i = 0; i < NTV; ++i)
        accSt(&s.W[(ntb + i) * 16 + lo][mt * 16 + 8 * hi], wa[i], lane);
    }
    // sA = scale * e^{g_i-g_j} * tril(P). Row ii is constant per lane and the eight g_j are
    // uniform per 16 lanes: eight broadcast LDS reads, v_sub and v_exp per tile (log(scale) is
    // in the exponent). Where kept, g_i - g_j <= 0, so the exp cannot overflow.
    // gfx1151: element e is the key (ntb+i)*16 + 2e + hi, so j and its mask are per element; the
    // 16 consecutive keys of the row are reassembled by accPackX().
    {
      const int ii = mt * 16 + lo;
      const float gis = s.gg[ii] + lgs;
      const float* gj = &s.gg[ntb * 16 + hi];
#pragma unroll
      for (int i = 0; i < NTV; ++i) {
        // The exp is taken for every element and the mask applied as a select after it: written
        // as one conditional expression it lowers to a divergent branch around each LDS read
        // and exp. Above the diagonal the exponent may overflow; the select discards it.
        float a[8], x[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          x[e] = p[i][e] * __expf(gis - gj[i * 16 + 2 * e]);
          asm volatile("" : "+v"(x[e]));      // keeps MachineSink from re-forming the branch
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int j = (ntb + i) * 16 + 2 * e + hi;
          a[e] = (j <= ii) ? x[e] : 0.f;
        }
        accSt(&s.A[ii][(ntb + i) * 16 + 8 * hi], a, lane);
      }
    }
    lds_barrier();                                           // (2)

    // ---- V' = gv * (A @ W)  -> D (transposed store is one contiguous pair of dwords)
    {
      v8f d[NTV];
#pragma unroll
      for (int i = 0; i < NTV; ++i) d[i] = (v8f){};
      const unsigned short* pwn = rowP(&s.W[0][0], WP, ntb * 16, lane);
#define pwnf(i, ko) fragP(pwn + (i) * 16 * WP + (ko))
#pragma unroll
      for (int sk = 0; sk < BT / 16; ++sk) {
        if (sk <= mt) {                       // A is lower triangular: A[mt][sk > mt] = 0
          const int ko = sk * 16;
#pragma unroll
          for (int i = 0; i < NTV; ++i) d[i] = mma(af[sk], pwnf(i, ko), d[i]);
        }
      }
#pragma unroll
      for (int i = 0; i < NTV; ++i)
        accSt(&s.D[(ntb + i) * 16 + lo][mt * 16 + 8 * hi], d[i], lane);
    }
    // KT = (gs * K)^T, for the state's V'^T @ K: (diag(gs) V')^T K == V'^T (diag(gs) K). All raw-K
    // reads (U, P) and s.S reads retired before barrier (2); barrier (3) publishes KT. A lane
    // takes a token pair and eight k: two 16-byte row reads, eight dword stores of a [k] row
    // (32 lanes, 32 consecutive dwords: no bank conflict). Same rounding as the old in-place
    // scale: one f2bf of bf16(k) * gs per element.
    {
      const int t0 = 2 * lane;
      const float f0 = s.gv[t0], f1 = s.gv[t0 + 1];
#pragma unroll
      for (int it = 0; it < KD / (8 * NW); ++it) {
        const int c = (it * NW + wu) * 8;
        const uint4 a = *(const uint4*)&s.k[t0][c], b = *(const uint4*)&s.k[t0 + 1][c];
        const unsigned av[4] = { a.x, a.y, a.z, a.w }, bw[4] = { b.x, b.y, b.z, b.w };
        unsigned* kt = (unsigned*)&s.KT[c][t0];
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          const float a0 = __builtin_bit_cast(float, av[h] << 16);
          const float a1 = __builtin_bit_cast(float, av[h] & 0xffff0000u);
          const float b0 = __builtin_bit_cast(float, bw[h] << 16);
          const float b1 = __builtin_bit_cast(float, bw[h] & 0xffff0000u);
          kt[(2 * h) * (AP / 2)]     = f2bf2(a0 * f0, b0 * f1);
          kt[(2 * h + 1) * (AP / 2)] = f2bf2(a1 * f0, b1 * f1);
        }
      }
    }
    lds_barrier();                                           // (3)

    // ---- O += sA @ V' ; store.  V' also feeds the state update, unchanged.
    {
      const unsigned short* pA = rowP(&s.A[0][0], AP, mt * 16, lane);
      const unsigned short* pdn = rowP(&s.D[0][0], AP, ntb * 16, lane);
#define pAf(ko)     fragP(pA + (ko))
#define pdnf(i, ko) fragP(pdn + (i) * 16 * AP + (ko))
#pragma unroll
      for (int sk = 0; sk < BT / 16; ++sk) {
        if (sk <= mt) {                       // sA is masked to j <= i
          const int ko = sk * 16;
          v16bf b = pAf(ko);
#pragma unroll
          for (int i = 0; i < NTV; ++i) O[i] = mma(pdnf(i, ko), b, O[i]);
        }
      }
    }
    {
      // The half-wave exchange accPackX() does is a wave-wide lane operation (v_permlanex16), so it is
      // taken for the whole wave and only the stores are guarded. A trailing partial chunk leaves
      // lanes without a token, and a lane op under a divergent exec mask reads an inactive partner:
      // measured as infinities in exactly those lanes' rows (see the pitch note above).
      const int tok = mt * 16 + lo;
      uint4 w[NTV];
#pragma unroll
      for (int i = 0; i < NTV; ++i) w[i] = accPk(O[i], lane);
      if (tok < nval) {
        // O's eight packed values are eight consecutive v of row tok (see the scale above).
        unsigned short* op = o + ((size_t)(c0 + tok) * H + hv) * VD + v0;
#pragma unroll
        for (int i = 0; i < NTV; ++i)
          *(uint4*)(op + (ntb + i) * 16 + 8 * hi) = w[i];
      }
    }

    // ---- S = e^{gl} * S + V'^T @ (gs*K)   (A = KT rows, B = V' rows: both k-contiguous)
    {
      const float ec = __expf(gl);            // gl <= 0: no lam, the decay is already on K
#pragma unroll
      for (int t = 0; t < NST; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) St[t][e] *= ec;
      const unsigned short* pvn = rowP(&s.D[0][0], AP, vt0 * 16, lane);
      const unsigned short* pkt = rowP(&s.KT[0][0], AP, kt0 * 16, lane);
#pragma unroll
      for (int sk = 0; sk < BT / 16; ++sk) {
        v16bf bv[SVT];
#pragma unroll
        for (int j = 0; j < SVT; ++j) bv[j] = fragP(pvn + j * 16 * AP + sk * 16);
#pragma unroll
        for (int ki = 0; ki < SKT; ++ki) {
          v16bf ak = fragP(pkt + ki * 16 * AP + sk * 16);          // K^T: m = k-dim
#pragma unroll
          for (int j = 0; j < SVT; ++j)
            St[ki * SVT + j] = mma(ak, bv[j], St[ki * SVT + j]);
        }
      }
    }
    lds_barrier();                                           // (4)
  }

  if constexpr (!HT) return;
  // Barrier (4) (or, with no chunk, the h0 load's last barrier) has retired every LDS read.
#pragma unroll
  for (int t = 0; t < NST; ++t)
#pragma unroll
    for (int e = 0; e < 8; ++e) SF(t, e) = St[t][e];
  lds_barrier();
  typename cs_state<SDT>::T* p = ht + st_base + (size_t)v0 * KD;
  if constexpr (SDT == 0) {
#pragma unroll
    for (int u = 0; u < BV / NW; ++u)
      *(float4*)(p + (wu + NW * u) * KD + lane * 4) = *(const float4*)&F[wu + NW * u][lane * 4];
  } else {
    const int r0 = tid >> 4, c = (tid & 15) * 8;
#pragma unroll
    for (int u = 0; u < BV * 16 / NTHR; ++u) {
      const float* d = &F[r0 + (NTHR / 16) * u][c];
      const float4 a = *(const float4*)d, b = *(const float4*)(d + 4);
      union { uint4 w; _Float16 h[8]; } x;
      x.h[0] = (_Float16)a.x; x.h[1] = (_Float16)a.y; x.h[2] = (_Float16)a.z; x.h[3] = (_Float16)a.w;
      x.h[4] = (_Float16)b.x; x.h[5] = (_Float16)b.y; x.h[6] = (_Float16)b.z; x.h[7] = (_Float16)b.w;
      *(uint4*)(p + (r0 + (NTHR / 16) * u) * KD + c) = x.w;
    }
  }
}

// Slot-indexed form (the one radiance calls). `sidx` names each batch row's state slot; without
// it the cache is indexed by row. `st_slot` and `st_head` are the cache strides in elements, which
// may exceed H*V*K when a caller pads its slots (vLLM pads to the attention page size). Pass
// sidx = null and strides 0 for the packed [N, H, V, K] layout.
extern "C" int r3dh_gdn_chunk_scan_k128_v128_c64_bf16_sidx(
    const void* q, const void* k, const void* v, const void* A, const void* g,
    const void* beta, const void* h0, void* o, void* ht, const void* cu,
    const void* ssm_state_indices, long indices_stride, long state_slot_stride,
    long state_head_stride,
    int N, int H, int Hg, int K, int V, int bt, float scale, int state_fp16, void* stream)
{
  if (K != KD || V != VD || bt != BT) return -1;
  // hq = hv / (H / Hg) truncates: H not a multiple of Hg would silently map v heads to the wrong
  // k heads. Refused with -2, as in the kkt_solve and recurrent_update siblings.
  if (Hg <= 0 || H % Hg) return -2;
  const long st_head = state_head_stride > 0 ? state_head_stride : (long)VD * KD;
  const long st_slot = state_slot_stride > 0 ? state_slot_stride : (long)H * VD * KD;
  dim3 grid(V / BV, H, N), blk(NTHR);
#define CS_LAUNCH(SDTC, H0C, HTC)                                                          \
  hipLaunchKernelGGL((r3dh_gdn_chunk_scan_kernel<SDTC, H0C, HTC>), grid, blk, 0,            \
      (hipStream_t)stream,                                                                 \
      (const unsigned short*)q, (const unsigned short*)k, (const unsigned short*)v,        \
      (const unsigned short*)A, (const float*)g, (const float*)beta,                       \
      (const cs_state<SDTC>::T*)h0, (unsigned short*)o, (cs_state<SDTC>::T*)ht,            \
      (const int*)cu, (const int*)ssm_state_indices, (indices_stride > 0 ? indices_stride : 1),  \
      st_slot, st_head, H, Hg, scale)
#define CS_PICK(SDTC)                                                                      \
  do { if (h0 && ht) CS_LAUNCH(SDTC, true, true);                                          \
       else if (h0)  CS_LAUNCH(SDTC, true, false);                                         \
       else if (ht)  CS_LAUNCH(SDTC, false, true);                                         \
       else          CS_LAUNCH(SDTC, false, false); } while (0)
  if (state_fp16) CS_PICK(1); else CS_PICK(0);
#undef CS_PICK
#undef CS_LAUNCH
  return (int)hipGetLastError();
}

// Packed-layout entry point for the out-of-tree vLLM caller: one state per batch row,
// [N, H, V, K].
extern "C" int r3dh_gdn_chunk_scan_k128_v128_c64_bf16(
    const void* q, const void* k, const void* v, const void* A, const void* g,
    const void* beta, const void* h0, void* o, void* ht, const void* cu,
    int N, int H, int Hg, int K, int V, int bt, float scale, int state_fp16, void* stream)
{
  return r3dh_gdn_chunk_scan_k128_v128_c64_bf16_sidx(q, k, v, A, g, beta, h0, o, ht, cu,
                                                    nullptr, 0, 0, 0, N, H, Hg, K, V, bt, scale,
                                                    state_fp16, stream);
}

// The compiled geometry, so callers can check a model's GDN shape against it.
extern "C" void r3dh_gdn_dims(int* head_k, int* head_v, int* chunk) {
  if (head_k) *head_k = KD;
  if (head_v) *head_v = VD;
  if (chunk)  *chunk  = BT;
}

// ============================================================== the plugin ABI entry points
// Shared operand helpers: r3dh_gdn_args.h.
/* ------------------------------------------------------------------ chunk scan */

enum { CS_Q = 0, CS_K, CS_V, CS_A, CS_G, CS_BETA, CS_H0, CS_CU, CS_O, CS_HT, CS_SIDX };

extern "C" int r3dh_rad_gdn_chunk_scan(const RadArgs* a, RadStream s) {
    if (gdn_dims_ok() != RAD_OK) return R3DH_NO(RAD_E_SHAPE);
    const RadTensor* q  = r3dh_opt(a, CS_Q);
    const RadTensor* k  = r3dh_opt(a, CS_K);
    const RadTensor* v  = r3dh_opt(a, CS_V);
    const RadTensor* A  = r3dh_opt(a, CS_A);
    const RadTensor* g  = r3dh_opt(a, CS_G);
    const RadTensor* bt = r3dh_opt(a, CS_BETA);
    const RadTensor* h0 = r3dh_opt(a, CS_H0);
    const RadTensor* cu = r3dh_opt(a, CS_CU);
    const RadTensor* o  = r3dh_opt(a, CS_O);
    const RadTensor* ht = r3dh_opt(a, CS_HT);
    const RadTensor* sidx = r3dh_opt(a, CS_SIDX);
    if (!q || !k || !v || !A || !g || !bt || !cu || !o) return R3DH_NO(RAD_E_INVAL);
    if (q->rank != 3 || k->rank != 3 || v->rank != 3 || o->rank != 3) return R3DH_NO(RAD_E_SHAPE);
    if (A->rank != 3 || g->rank != 2 || bt->rank != 2) return R3DH_NO(RAD_E_SHAPE);

    int N = 0;
    int rc = seqs_of(cu, &N);
    if (rc != RAD_OK) return rc;

    /* The entry point takes no pitches: everything must be dense. */
    if (!r3dh_contig(q) || !r3dh_contig(k) || !r3dh_contig(v) || !r3dh_contig(A) ||
        !r3dh_contig(g) || !r3dh_contig(bt) || !r3dh_contig(o)) return R3DH_NO(RAD_E_STRIDE);

    /* bf16 q/k/v/A/out, fp32 gate/beta. */
    R3DH_DT(q, RAD_BF16); R3DH_DT(k, RAD_BF16); R3DH_DT(v, RAD_BF16); R3DH_DT(A, RAD_BF16);
    R3DH_DT(o, RAD_BF16); R3DH_DT(g, RAD_F32);  R3DH_DT(bt, RAD_F32);
    /* The state cache is f32 or f16, taken from the operand's dtype (a property of the
     * allocation), so this kernel and the recurrent update, which share the cache, agree on it. */
    const uint32_t st_dt = h0 ? h0->dtype : (ht ? ht->dtype : (uint32_t)RAD_F32);
    if (st_dt != RAD_F32 && st_dt != RAD_F16) return R3DH_NO(RAD_E_DTYPE);
    if (h0) R3DH_DT(h0, st_dt);
    if (ht) R3DH_DT(ht, st_dt);
    /* Slot strides come from the tensor, not from H, V and K, so a cache with padded slots is
     * served too; the engine's own state cache is dense ([n_states, H, V, K] per layer). h0 and
     * ht must share strides, since the kernel uses one base offset for both. */
    long long st_slot = 0, st_head = 0;
    if (h0) {
        if (h0->rank < 2) return R3DH_NO(RAD_E_SHAPE);
        st_slot = h0->stride[0];
        st_head = h0->stride[1];
    }
    if (ht) {
        if (ht->rank < 2) return R3DH_NO(RAD_E_SHAPE);
        if (h0 && (ht->stride[0] != st_slot || ht->stride[1] != st_head))
            return R3DH_NO(RAD_E_STRIDE);
        st_slot = ht->stride[0];
        st_head = ht->stride[1];
    }
    if (sidx) {
        R3DH_DT(sidx, RAD_I32);
        if (sidx->rank < 1 || sidx->shape[0] != N) return R3DH_NO(RAD_E_SHAPE);
        if (sidx->stride[sidx->rank - 1] != 1) return R3DH_NO(RAD_E_STRIDE);
    }

    const long long T  = q->shape[0];
    const long long Hg = q->shape[1];
    const long long K  = q->shape[2];
    const long long H  = v->shape[1];
    const long long V  = v->shape[2];
    const long long bt_chunk = A->shape[2];
    if (k->shape[0] != T || k->shape[1] != Hg || k->shape[2] != K) return R3DH_NO(RAD_E_SHAPE);
    if (v->shape[0] != T || o->shape[0] != T || o->shape[1] != H || o->shape[2] != V)
        return R3DH_NO(RAD_E_SHAPE);
    if (A->shape[0] != T || A->shape[1] != H) return R3DH_NO(RAD_E_SHAPE);
    if (g->shape[0] != T || g->shape[1] != H) return R3DH_NO(RAD_E_SHAPE);
    if (bt->shape[0] != T || bt->shape[1] != H) return R3DH_NO(RAD_E_SHAPE);
    if (H == 0 || Hg == 0 || H % Hg) return R3DH_NO(RAD_E_SHAPE);

    const float scale = (float)r3dh_getf(a, "scale", 1.0);
    const int state_fp16 = st_dt == RAD_F16;

    return R3DH_RAW(r3dh_gdn_chunk_scan_k128_v128_c64_bf16_sidx(
        q->data, k->data, v->data, A->data, g->data, bt->data,
        h0 ? h0->data : nullptr, o->data, ht ? ht->data : nullptr, cu->data,
        sidx ? sidx->data : nullptr, sidx && sidx->rank >= 2 ? (long)sidx->stride[0] : 1,
        (long)st_slot, (long)st_head,
        N, (int)H, (int)Hg, (int)K, (int)V, (int)bt_chunk, scale, state_fp16, s));
}

