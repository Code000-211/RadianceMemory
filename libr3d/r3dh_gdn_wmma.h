/* Ported from libhalo/halo_gdn_wmma.h, verbatim but for the halo_ -> r3dh_ rename, so libr3d builds and loads
 * on its own. libhalo's gated-delta-net kernels are libr4d's ported to the gfx11 wave matrix and
 * checked against libref on gfx1151 (rad-kbench); they compile for gfx11-generic unchanged. */
/* Carried over from the strix-halo-vllm project's libhalo, where this layer was
 * established and measured on the part; see libhalo/README.md. */
// Fragment layer for the GDN kernels on gfx1151 / RDNA 3.5.
//
// This is the gfx11 counterpart of libr4d's r4d_gdn_wmma.h, and the difference is
// the whole reason the GDN kernels cannot simply be copied across:
//
//                    A/B per lane   the k that lane's element e holds
//   gfx1201 (RDNA4)  8 bf16         k = 8*(e>>2) + 4*(lane>>4) + (e&3)
//   gfx1151 (RDNA3.5) 16 bf16       k = e
//
// RDNA 3.5 replicates each operand row across both halves of the wave, so a lane
// holds the WHOLE 16-wide k run. Established on the part itself by
// bench/wmma_layout_probe.hip; see r3dh_wmma.h.
//
// For this layer that is a simplification, not extra work:
//
//   * a fragment is ONE 32-byte contiguous read of the row, so two dwordx4 rather
//     than libr4d's two 8-byte groups sixteen bytes apart. All of its careful
//     ds_read2_b32 / ds_load_2addr_b64 pairing notes are about making THAT split
//     addressable in one register; there is no split here to pair.
//   * the row pointer loses its k term entirely -- `4 * (lane >> 4)` disappears
//     from rowP, rowG and rowGo, because the half-wave no longer selects k.
//
// What does NOT simplify is fragT, the transposed source: a lane needs 16 strided
// scalar reads instead of 8.
#pragma once
#include "r3dh_common.h"
#include "r3dh_wmma.h"

// ---- the gated residual's own bf16 arithmetic ----------------------------------
// libr4d's r4d_gdn_wmma.h carries these three and the kernels written against it use them
// (r3dh_ar_hc.h, r3dh_hc_bf16.hip); the port's header lost them, so they are restored here, in
// the same place and with the same rounding libr4d uses (round-half-away: one add + a shift,
// against RTNE's three instructions, and they differ only when the discarded bits are exactly
// 0x8000). halo::f32_to_bf16 is the RTNE form the WMMA helpers pack with; keep them apart.
__device__ __forceinline__ unsigned short f2bf(float f) {
    return (unsigned short)((__builtin_bit_cast(unsigned, f) + 0x8000u) >> 16);
}
__device__ __forceinline__ unsigned f2bf2(float a, float b) {
    const unsigned ua = __builtin_bit_cast(unsigned, a) + 0x8000u;
    const unsigned ub = __builtin_bit_cast(unsigned, b) + 0x8000u;
    return __builtin_amdgcn_perm(ub, ua, 0x07060302u);   // {b_hi16, a_hi16}
}
__device__ __forceinline__ float bf2f(unsigned short h) {
    return __builtin_bit_cast(float, (unsigned)h << 16);
}

// K-contiguous fragment from a row pointer already advanced to this lane's row.
__device__ __forceinline__ v16bf fragP(const unsigned short* p) {
    union { uint4 w[2]; v16bf f; } u;
    u.w[0] = ((const uint4*)p)[0];
    u.w[1] = ((const uint4*)p)[1];
    return u.f;
}

// Row pointer for the fragment above: LDS source, `pitch` in elements. No k term --
// on gfx11 the lane's half does not select k.
__device__ __forceinline__ const unsigned short* rowP(const unsigned short* src,
                                                      int pitch, int base, int lane) {
    return src + (size_t)(base + (lane & 15)) * pitch;
}

// Row pointer for a global source; `lim` clamps so a partial last chunk cannot read
// past the tensor (those rows are discarded downstream).
__device__ __forceinline__ const unsigned short* rowG(const unsigned short* base,
                                                      size_t rowstride, int row0,
                                                      int lim, int lane) {
    int r = row0 + (lane & 15);
    r = r < lim ? r : lim;
    return base + (size_t)r * rowstride;
}

// 32-bit element offset for a global fragment source, split from a uniform base:
// keeps the whole k loop as global_load_dwordx4 vdst, voff, s[base] with an
// immediate, instead of rematerialising 64-bit address arithmetic per load.
__device__ __forceinline__ unsigned rowGo(unsigned rowstride, int row0, int lim,
                                          int lane) {
    int r = row0 + (lane & 15);
    r = r < lim ? r : lim;
    return (unsigned)r * rowstride;
}
__device__ __forceinline__ v16bf fragGo(const unsigned short* base, unsigned off) {
    return fragP(base + off);
}

// Fragment from a 32-bit LDS byte offset off the shared-block base. Holding the
// offset in one opaque register keeps the two dwordx4 as immediates rather than
// letting LLVM rematerialise a base add in front of every fragment.
__device__ __forceinline__ v16bf fragO(const char* sb, unsigned off) {
    return fragP((const unsigned short*)(sb + off));
}
#define LDSOPAQUE(x) asm("" : "+v"(x))

// K-contiguous fragment: src[idx][k], row pitch `pitch` (in elements)
__device__ __forceinline__ v16bf frag(const unsigned short* src, int pitch,
                                      int base, int kbase, int lane) {
    return fragP(rowP(src, pitch, base, lane) + kbase);
}

// Same fragment, source stored transposed: src[k][idx]. 16 scalar reads on gfx11
// against 8 on gfx12 -- the one place the wider fragment costs rather than saves.
__device__ __forceinline__ v16bf fragT(const unsigned short* src, int pitch,
                                       int base, int kbase, int lane) {
    const int idx = base + (lane & 15);
    union { unsigned short h[16]; v16bf f; } u;
#pragma unroll
    for (int e = 0; e < 16; ++e) u.h[e] = src[(size_t)(kbase + e) * pitch + idx];
    return u.f;
}

// Fragment straight from global memory (no LDS staging). `lim` clamps the row so a
// partial last chunk cannot read past the tensor.
__device__ __forceinline__ v16bf fragG(const unsigned short* base, size_t rowstride,
                                       int row0, int lim, int kbase, int lane) {
    int r = row0 + (lane & 15);
    r = r < lim ? r : lim;
    return fragP(base + (size_t)r * rowstride + kbase);
}

__device__ __forceinline__ v8f mma(v16bf a, v16bf b, v8f c) {
    return halo::mma(a, b, c);
}

// C[mt*16..][nt*16..] += A[M][KA] @ B[N][KA]^T over KSTEP 16-wide steps
template <int KSTEP>
__device__ __forceinline__ v8f gemm(const unsigned short* A, int Ap, int mt,
                                    const unsigned short* B, int Bp, int nt,
                                    int lane, v8f acc) {
#pragma unroll
    for (int s = 0; s < KSTEP; ++s)
        acc = mma(frag(A, Ap, mt * 16, s * 16, lane),
                  frag(B, Bp, nt * 16, s * 16, lane), acc);
    return acc;
}

// Where a result element lives. gfx11 wave32 puts D[2*e + (lane>>4)][lane & 15] in
// element e, which is NOT gfx12's mapping -- any kernel that writes an accumulator
// out by hand has to use these rather than the row/col arithmetic it came with.
__device__ __forceinline__ int accRow(int lane, int e) { return 2 * e + (lane >> 4); }
__device__ __forceinline__ int accCol(int lane) { return lane & 15; }

// ---- writing an accumulator out ------------------------------------------------
// gfx11 hands lane l element e the value D[2e + (l>>4)][l & 15]: eight rows of ONE
// column, stride two. libr4d's gfx12 mapping is D[8*(l>>4) + e][l & 15] -- eight
// CONSECUTIVE rows -- which is why every one of its accumulator write-outs is a
// single 16-byte store of a transposed tile. Ported naively that inverts into eight
// strided 2-byte stores.
//
// It need not. The partner lane l^16 holds exactly the eight rows interleaved
// between this lane's, so ONE cross-half exchange per dword reassembles sixteen
// consecutive rows, and each half-wave then owns eight of them -- the same single
// 16-byte store, for four v_permlanex16.
//
// Exchange the half the PARTNER needs rather than the half this lane owns: that is
// what keeps it at four shuffles instead of eight (a shuffle reads one VGPR for the
// whole wave, so the two halves cannot ask for different elements).
//
// v[] must ALREADY be scaled per element, because the two halves are scaling
// different rows -- see the gram store in r3dh_gdn_kkt_solve.hip.
__device__ __forceinline__ uint4 accPack(const float v[8], int lane) {
    const int hi = lane >> 4;
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float mine = hi ? v[i + 4] : v[i];   // the row this lane already holds
        const float send = hi ? v[i] : v[i + 4];   // the row the partner is missing
        const float got  = halo::swap16(send);
        w[i] = hi ? halo::pk_bf16(got, mine) : halo::pk_bf16(mine, got);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}
// First row of the eight accPack() returns for this lane.
__device__ __forceinline__ int accPackRow(int lane) { return 8 * (lane >> 4); }

// accPack() with the same values and rounding, for a kernel where the write-out sits on the
// critical path. accPack() exchanges four floats per lane through halo::swap16 -- a __shfl_xor,
// so ds_bpermute_b32 and an LDS wait per dword -- and LLVM lowers its
// `hi ? pk(got, mine) : pk(mine, got)` to a divergent branch around each pack. This one rounds
// first (pk_bf16's round-to-nearest-even, the same bits), so the partner's four rows travel as two
// packed dwords over v_permlanex16, and the half-dependent order of the final pairs is a per-lane
// v_perm selector rather than a select per value:
//
//   p[i] = {v[2i], v[2i+1]}   rows {4i + hi, 4i + 2 + hi}
//   lanes 0-15 keep p[0..1] and need the partner's p[0..1] (rows 1, 3, 5, 7)
//   lanes 16-31 keep p[2..3] and need the partner's p[2..3] (rows 8, 10, 12, 14)
//
// Same precondition as accPack(): the whole wave is active.
__device__ __forceinline__ uint4 accPackX(const float v[8], int lane) {
    const bool hi = (lane >> 4) != 0;
    uint32_t r[8], p[4];
#pragma unroll
    for (int e = 0; e < 8; ++e) {           // f32_to_bf16's rounding, the result in the high half
        const uint32_t u = __builtin_bit_cast(uint32_t, v[e]);
        r[e] = u + 0x7fffu + ((u >> 16) & 1u);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __builtin_amdgcn_perm(r[2 * i + 1], r[2 * i], 0x07060302u);
    const uint32_t k0 = hi ? p[2] : p[0], k1 = hi ? p[3] : p[1];      // kept
    const uint32_t g0 = r3dh_xor_lane_u32<16>(hi ? p[0] : p[2]);      // got: the partner's kept
    const uint32_t g1 = r3dh_xor_lane_u32<16>(hi ? p[1] : p[3]);
    // {own, partner} below, {partner, own} above: low halves, then high halves.
    const uint32_t sl = hi ? 0x01000504u : 0x05040100u, sh = hi ? 0x03020706u : 0x07060302u;
    return make_uint4(__builtin_amdgcn_perm(g0, k0, sl), __builtin_amdgcn_perm(g0, k0, sh),
                      __builtin_amdgcn_perm(g1, k1, sl), __builtin_amdgcn_perm(g1, k1, sh));
}

// Byte offset, from the base of a shared block, of the fragment row this lane reads out
// of `buf[base + (lane&15)][...]` with the given row pitch in elements.
//
// This is the gfx11 form of libr4d's ROWO macro, and the difference is the whole port in
// one line: gfx12 adds `4 * (lane >> 4)` because the half-wave selects which quarter of
// the k run the lane holds, and gfx11 has no such term because the lane holds all of it.
// Every LDS fragment site in a ported kernel goes through here, so getting it wrong is a
// single-point failure rather than a scattered one.
__device__ __forceinline__ unsigned rowO(unsigned buf_off, int pitch, int base, int lane) {
    return buf_off + (unsigned)(((base + (lane & 15)) * pitch) * 2);
}
