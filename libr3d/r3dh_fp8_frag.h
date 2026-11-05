/* Ported from libhalo/halo_fp8_frag.h, verbatim but for the halo_ -> r3dh_ rename, so libr3d builds and loads
 * on its own. libhalo's gated-delta-net kernels are libr4d's ported to the gfx11 wave matrix and
 * checked against libref on gfx1151 (rad-kbench); they compile for gfx11-generic unchanged. */
/* Derived from libr4d/r4d_fp8_frag.h by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. Op schemas, row names and stored weight layouts are
 * shared with libr4d by design. Everything else in this file is libhalo's own
 * and diverges as it is measured on gfx1151. */
// r3dh_fp8_frag.h -- where a byte of a block-scaled FP8 weight lives in WMMA fragment order. Pure
// C++ (no HIP, no halo.h) so the host layout hook and the device kernels share one definition.
//
// Why permute: the fp8a8 GEMMs read operands through r3dh_fp8::frag(), which wants lane `l` to hold
// row `l & 15` of a 16-row tile at k `(l >> 4) * 8`, eight bytes. From a row-major weight those
// sixteen rows are `K` bytes apart, so a wave's load is sixteen transactions; the kernels stage
// through LDS to transpose, at the cost of a barrier, 36 KiB per block and a second pass over the
// weight. In fragment order the bytes are already in lane order.
//
// Why 64 k and not 16: a single fragment (16 rows x 16 k = 256 bytes) gives a wave one 256-byte
// cacheline per request, which loses even to LDS staging. Four fragments give each lane 32
// contiguous bytes and the wave a 1 KiB request, which beats both at every N and K in the family.
//
// The unit is a (16 row, 64 k) tile of 1024 bytes, laid out as 32 lanes x 32 bytes. Lane `l`'s
// 32 bytes are its four fragments back to back: bytes `8t` are fragment `t`, covering
// k `t*16 + (l>>4)*8` of row `l & 15`.
//
// K must be a multiple of 64 and N of 16; the 128 scale block already guarantees both.
#pragma once
#include <cstdint>

namespace r3dh_fp8 {

enum { kFragRows = 16, kFragK = 16, kSuperK = 64, kTileBytes = 1024, kLaneBytes = 32 };

// Bytes one [N][K] weight occupies in fragment order: exactly N*K, since this is a permutation.
inline int64_t frag_bytes(int64_t N, int64_t K) { return N * K; }

// The source (row, k) of a given (tile, lane, fragment, byte). The single definition of the
// permutation: the host transform walks it forwards, the device kernels index it backwards.
inline void frag_src(int64_t rt, int64_t ks, int64_t lane, int64_t t, int64_t j,
                     int64_t* row, int64_t* k) {
    *row = rt * (int64_t)kFragRows + (lane & 15);
    *k   = ks * (int64_t)kSuperK + t * (int64_t)kFragK + (lane >> 4) * 8 + j;
}

}  // namespace r3dh_fp8
