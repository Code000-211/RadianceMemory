/* Ported from libhalo/halo_common.h, verbatim but for the halo_ -> r3dh_ rename, so libr3d builds and loads
 * on its own. libhalo's gated-delta-net kernels are libr4d's ported to the gfx11 wave matrix and
 * checked against libref on gfx1151 (rad-kbench); they compile for gfx11-generic unchanged. */
/* Derived from libr4d/r4d_common.h by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. Op schemas, row names and stored weight layouts are
 * shared with libr4d by design. Everything else in this file is libhalo's own
 * and diverges as it is measured on gfx1151. */
// Shared device-side machinery for Halo. The layout algebra here was verified on hardware with
// targeted probes; do not "fix" it from a vendor doc.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// The W4A8 scale group lives here, not in either GEMM, because both read the same packed weight and
// a disagreement would be silently wrong, not a link error. r3dh_gemm_args.h restates it and checks
// it against the layout constant.
#define R3DH_GEMM_W4A8_GROUP 128      // K per scale

// ---- ragged query batches ----------------------------------------------------------------------
// A paged-attention batch may have a different query length per sequence (HaloArgs::cu_q in halo.h).
// These are the two lookups every kernel indexing the dense q or out block needs, plus the inverse
// for the split-KV combine, which walks output rows rather than sequences. With cu_q null they
// compile to one multiply and no load; the branch is wave-uniform (cu_q is a kernel argument).
__device__ __forceinline__ int r3dh_q_off(const int* cu_q, int seq, int q_len) {
    return cu_q ? cu_q[seq] : seq * q_len;
}
__device__ __forceinline__ int r3dh_q_len(const int* cu_q, int seq, int q_len) {
    return cu_q ? (cu_q[seq + 1] - cu_q[seq]) : q_len;
}
// num_seqs is at most a few tens, so the search is a few scalar loads, once per block.
__device__ __forceinline__ int r3dh_q_seq(const int* cu_q, int num_seqs, int q_len, int tok) {
    if (!cu_q) return tok / q_len;
    int lo = 0, hi = num_seqs;                       // invariant: cu_q[lo] <= tok < cu_q[hi]
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (cu_q[mid] <= tok) lo = mid; else hi = mid;
    }
    return lo;
}

// ---- the occupancy a kernel is built for ---------------------------------------------------------
// R3DH_WAVES(n) fixes the waves per SIMD the compiler schedules a kernel for. Otherwise the
// scheduler picks that target itself and the choice moves between compiler releases (LLVM 24
// serialises a GEMM's loads to win one more wave, paying a load round trip per WMMA). A kernel
// whose speed rests on a register budget states it here. `n` may depend on template arguments.
#define R3DH_WAVES(n) __attribute__((amdgpu_waves_per_eu((n), (n))))

typedef short    v8s __attribute__((ext_vector_type(8)));
typedef float    v8f __attribute__((ext_vector_type(8)));
typedef float    v2f __attribute__((ext_vector_type(2)));   // what cvt_pk_f32_fp8 actually returns

/* A 16-byte staging register as a native vector. HIP's uint4 is a class template wrapping a
 * union, and an array of them carried across a K loop (the next tile's chunks) is kept in scratch
 * by LLVM 22 and 24, adding a scratch store and load per chunk; an ext_vector_type array is
 * promoted to registers. Loaded and stored whole only. r3dh_dflash_conv_body.h declares the same
 * type for a source also compiled without this header. */
typedef unsigned int r3dh_u32x4 __attribute__((ext_vector_type(4)));

// exp2 with the hardware transcendental.

__device__ __forceinline__ float ex2(float x) {
#if __has_builtin(__builtin_amdgcn_exp2f)
    return __builtin_amdgcn_exp2f(x);
#else
    return exp2f(x);
#endif
}

// f32 -> bf16, round-to-nearest-even, in software: gfx1201 has no v_cvt_pk_bf16_f32.
__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// And back: a shift, since every bf16 is exactly a float.
__device__ __forceinline__ float bf16_to_f32(uint16_t h) {
    return __builtin_bit_cast(float, (uint32_t)h << 16);
}

// ---- 16-bit values held in whole registers -------------------------------------------------------
// LLVM 24 gives gfx12 real 16-bit registers: a 16-bit value lives in one half of a VGPR. Packing
// two bf16 then costs a shift and a half move per pair instead of one v_perm_b32, widening costs a
// half move and a shift, and a 16-bit load into a half (global_load_d16_b16) depends on the
// register's earlier contents, so the waitcnt pass waits for them. The helpers below keep bf16 data
// in whole 32-bit registers from load to store, which compiles the same under LLVM 22 and 24 and
// gives the same bits as the 16-bit forms.

// f32 -> bf16 rounded to nearest even, as the high half of a word whose low half is not zero
// (f32_to_bf16 before its final shift). A NaN can round to infinity, as there.
__device__ __forceinline__ uint32_t bf16_hi_rne(float f) {
    const uint32_t u = __float_as_uint(f);
    return u + 0x7fffu + ((u >> 16) & 1u);
}
// Rounds as (__hip_bfloat16)f does (LLVM's expansion): a NaN stays NaN, quieted; everything else
// rounds to nearest even. Both results are formed before the select so it compiles to one
// v_cndmask; `nan ? u | 0x400000 : u + ...` makes each element an exec-masked branch.
__device__ __forceinline__ uint32_t bf16_hi_rne_nan(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint32_t r = u + 0x7fffu + ((u >> 16) & 1u), q = u | 0x400000u;
    return f != f ? q : r;
}
// Two such words as one bf16 pair, `lo`'s high half in the low half: one byte permute.
__device__ __forceinline__ uint32_t bf16_pack_hi(uint32_t lo, uint32_t hi) {
    return __builtin_amdgcn_perm(hi, lo, 0x07060302u);
}
// Two floats as a bf16 pair (`lo` in the low half), rounded as f32_to_bf16 rounds.
__device__ __forceinline__ uint32_t bf16x2_rne(float lo, float hi) {
    return bf16_pack_hi(bf16_hi_rne(lo), bf16_hi_rne(hi));
}
// Two floats as a bf16 pair, rounded as (__hip_bfloat16) rounds.
__device__ __forceinline__ uint32_t bf16x2_rne_nan(float lo, float hi) {
    return bf16_pack_hi(bf16_hi_rne_nan(lo), bf16_hi_rne_nan(hi));
}
// A bf16 pair word's two values as floats: a shift and a mask, no conversion.
__device__ __forceinline__ float bf16w_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16w_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// One 16-bit element of global memory, zero-extended into a whole register (global_load_u16).
// Loaded as uint16_t it would go into a register half (global_load_d16_b16 or d16_hi_b16) and wait
// for the other half; through a generic pointer it would also be a flat load. The pointer must
// address global memory.
typedef __attribute__((address_space(1))) const uint16_t* r3dh_gu16;
__device__ __forceinline__ uint32_t ldg_u16(const uint16_t* p) { return *(r3dh_gu16)p; }

// `lane ^ N` without the LDS pipe. `__shfl_xor` lowers to `ds_bpermute_b32` plus `s_wait_dscnt`,
// an LDS round trip that dominates a dependency chain (e.g. a dot product closed in a token loop).
// gfx10+ DPP16 has ROW_XMASK (0x160 | N), so for N < 16 the partner read is a modifier on the
// consuming VALU op (`v_add_f32_dpp ... row_xmask:2`). N = 16 crosses the two 16-lane rows of a
// wave32 and uses v_permlanex16, still one VALU op. Same lane mapping and value as the shuffle,
// provided all lanes are active, which every caller's whole-wave butterfly guarantees.
template <int N>
__device__ __forceinline__ uint32_t r3dh_xor_lane_u32(uint32_t v) {
    static_assert(N >= 1 && N <= 16 && (N & (N - 1)) == 0, "a lane xor within the wave");
    if constexpr (N < 16)
        return (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, 0x160 | N, 0xf, 0xf, true);
    else
        return (uint32_t)__builtin_amdgcn_permlanex16((int)v, (int)v, 0x76543210u, 0xFEDCBA98u,
                                                      false, false);
}

template <int N>
__device__ __forceinline__ float r3dh_xor_lane(float v) {
    return __uint_as_float(r3dh_xor_lane_u32<N>(__float_as_uint(v)));
}

// A 64-bit value's partner: the halves exchanged separately, as a 64-bit `__shfl_xor` does.
template <int N>
__device__ __forceinline__ unsigned long long r3dh_xor_lane_u64(unsigned long long v) {
    const uint32_t lo = r3dh_xor_lane_u32<N>((uint32_t)v);
    const uint32_t hi = r3dh_xor_lane_u32<N>((uint32_t)(v >> 32));
    return ((unsigned long long)hi << 32) | lo;
}

// Exchange between lane L and lane L^16 (the two halves a wave32 WMMA fragment splits a row
// across). v_permlanex16_b32 swaps only with identity selectors (each 4-bit field names the source
// lane in the other half; sel0 covers lanes 0-7, sel1 lanes 8-15); zero selectors would silently
// broadcast lane 0.
__device__ __forceinline__ float swap16(float x) { return r3dh_xor_lane<16>(x); }

// A wave32 sum on DPP instead of the LDS crossbar. `__shfl_xor` is a `ds_bpermute_b32` on gfx1201
// at any offset, queued behind other LDS traffic; where little arithmetic separates reductions, a
// five-stage shuffle butterfly dominates the run. Offsets 8..1 use row_xmask (dpp_ctrl 0x160 | N)
// as a modifier on the add; 16 uses v_permlanex16_b32 (see swap16).
//
// The lanes exchanged and the order of the adds match the shuffle loop, so this is bit-identical
// and a kernel can move onto it without its output changing. Row and bank masks are all-ones since
// every lane participates; `bound_ctrl` is moot because every lane has a partner.
template <int CTRL>
__device__ __forceinline__ float r3dh_dpp_xor_add(float v) {
    return v + __builtin_bit_cast(float,
        __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xF, 0xF, true));
}

__device__ __forceinline__ float r3dh_wave_sum(float v) {
    v += swap16(v);                        /* xor 16 -- crosses the two rows of sixteen */
    v = r3dh_dpp_xor_add<0x168>(v);         /* xor  8 -- row_xmask from here down */
    v = r3dh_dpp_xor_add<0x164>(v);         /* xor  4 */
    v = r3dh_dpp_xor_add<0x162>(v);         /* xor  2 */
    v = r3dh_dpp_xor_add<0x161>(v);         /* xor  1 */
    return v;
}

// The same butterfly over a group of W lanes (W a power of two up to the wave): the exchanges of
// `for (o = W / 2; o; o >>= 1) v = op(v, __shfl_xor(v, o))`, in its order, so bit-identical to
// that loop.
template <int W>
__device__ __forceinline__ float r3dh_xor_sum(float v) {
    static_assert(W >= 1 && W <= 32 && (W & (W - 1)) == 0, "a power-of-two group within the wave");
    if constexpr (W >= 32) v += r3dh_xor_lane<16>(v);
    if constexpr (W >= 16) v += r3dh_xor_lane<8>(v);
    if constexpr (W >= 8)  v += r3dh_xor_lane<4>(v);
    if constexpr (W >= 4)  v += r3dh_xor_lane<2>(v);
    if constexpr (W >= 2)  v += r3dh_xor_lane<1>(v);
    return v;
}

template <int W>
__device__ __forceinline__ float r3dh_xor_max(float v) {
    static_assert(W >= 1 && W <= 32 && (W & (W - 1)) == 0, "a power-of-two group within the wave");
    if constexpr (W >= 32) v = fmaxf(v, r3dh_xor_lane<16>(v));
    if constexpr (W >= 16) v = fmaxf(v, r3dh_xor_lane<8>(v));
    if constexpr (W >= 8)  v = fmaxf(v, r3dh_xor_lane<4>(v));
    if constexpr (W >= 4)  v = fmaxf(v, r3dh_xor_lane<2>(v));
    if constexpr (W >= 2)  v = fmaxf(v, r3dh_xor_lane<1>(v));
    return v;
}

__device__ __forceinline__ float r3dh_wave_max(float v) { return r3dh_xor_max<32>(v); }

/* A wave's inclusive prefix and suffix sums in registers: row_shr (or row_shl) 1, 2, 4 and 8
 * within each row of sixteen, then the other row's total from one readlane. Lane L ends with the
 * sum over lanes 0..L (prefix) or L..31 (suffix). Replaces LDS step-doubling scans (two barriers a
 * round); integer sums, so combine order does not matter. Every lane must be active. */
__device__ __forceinline__ uint32_t r3dh_wave_prefix_u32(uint32_t x, int lane) {
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x111, 0xF, 0xF, true);   /* row_shr:1 */
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x112, 0xF, 0xF, true);   /* row_shr:2 */
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x114, 0xF, 0xF, true);   /* row_shr:4 */
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x118, 0xF, 0xF, true);   /* row_shr:8 */
    const uint32_t lo = (uint32_t)__builtin_amdgcn_readlane((int)x, 15);
    return lane >= 16 ? x + lo : x;
}

__device__ __forceinline__ uint32_t r3dh_wave_suffix_u32(uint32_t x, int lane) {
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x101, 0xF, 0xF, true);   /* row_shl:1 */
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x102, 0xF, 0xF, true);   /* row_shl:2 */
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x104, 0xF, 0xF, true);   /* row_shl:4 */
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x108, 0xF, 0xF, true);   /* row_shl:8 */
    const uint32_t hi = (uint32_t)__builtin_amdgcn_readlane((int)x, 16);
    return lane < 16 ? x + hi : x;
}

// 8x8 transpose of 16-bit elements across each group of 8 lanes. Lane j receives element j from
// each of the 8 lanes in its group; the 8 addresses may be arbitrarily strided.
//
// gfx1151 has no global_load_tr_b128, so this loads each lane's 16 B and transposes in registers:
// lane (base+i)'s output element j is lane (base+j)'s loaded element i, eight shuffles of the one
// dword that holds it. Same result, more instructions; a caller on a hot path should say so in
// its header so the transpose can be folded into its addressing instead.
__device__ __forceinline__ v8s load_tr_b128(const void* p) {
    const uint4 v = *(const uint4*)p;
    const uint32_t w[4] = { v.x, v.y, v.z, v.w };
    const int lane = (int)(threadIdx.x & 31u);
    const int base = lane & ~7, i = lane & 7;
    uint32_t o[4] = { 0, 0, 0, 0 };
    // A shuffle returns the value the SOURCE lane passed, so a lane cannot ask for "my element i"
    // in one shuffle: at lane base+j the argument `w[i >> 1]` is evaluated with THAT lane's i (j),
    // not this one's. Fetch all four of the source lane's dwords and select this lane's element
    // afterwards -- that is the transpose, and it is 32 shuffles rather than 8.
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t d0 = (uint32_t)__shfl((int)w[0], base + j, 32);
        const uint32_t d1 = (uint32_t)__shfl((int)w[1], base + j, 32);
        const uint32_t d2 = (uint32_t)__shfl((int)w[2], base + j, 32);
        const uint32_t d3 = (uint32_t)__shfl((int)w[3], base + j, 32);
        const int d = i >> 1;
        const uint32_t wd = d == 0 ? d0 : d == 1 ? d1 : d == 2 ? d2 : d3;
        const uint32_t e = (i & 1) ? (wd >> 16) : (wd & 0xffffu);
        o[j >> 1] |= e << (16 * (j & 1));
    }
    v8s r;
    __builtin_memcpy(&r, o, sizeof(o));
    return r;
}

// Gather the low (sel=0) or high (sel=1) byte of each of four 16-bit lanes of {w0,w1} into one
// dword. Written so the compiler can contract each to a single v_perm_b32.
__device__ __forceinline__ uint32_t byte_gather(uint32_t w0, uint32_t w1, int sel) {
    uint32_t s = (uint32_t)sel * 8u;
    return ((w0 >> s) & 0xffu)
         | (((w0 >> (16 + s)) & 0xffu) << 8)
         | (((w1 >> s) & 0xffu) << 16)
         | (((w1 >> (16 + s)) & 0xffu) << 24);
}

__device__ __forceinline__ void sched_barrier() { __builtin_amdgcn_sched_barrier(0); }

// __syncthreads() is a workgroup-scope acquire-release fence over all address spaces, so on gfx12
// it emits `global_inv scope:SCOPE_SE` on both sides, discarding the L0/L1 lines the next tile's
// KV loads would hit. These barriers only order LDS traffic, so the fence is scoped to the local
// address space and the invalidate disappears.
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Two-dword form of swap16.
__device__ __forceinline__ uint2 swap16_u2p(uint2 v) {
    return make_uint2(__builtin_bit_cast(uint32_t, swap16(__builtin_bit_cast(float, v.x))),
                      __builtin_bit_cast(uint32_t, swap16(__builtin_bit_cast(float, v.y))));
}

// ---------------------------------------------------------------------------------------------
// Workgroup width for the activation-path kernels
//
// had_quant_act_i8, rmsnorm_had_quant_i8, gated_had_quant_i8 and gdn_gated_norm_had_quant_i8 launch
// one workgroup per row, so at decode batch sizes the grid is a few tens of workgroups on a 64-CU
// part and they are latency bound. Widening the workgroup is the only knob that adds parallelism,
// and it is the large win at small M.
//
// It reverses once the grid alone fills the machine. At 1024 threads there is a residency cliff
// just past M=32 where the wider workgroup stops fitting and the cost roughly doubles, so 1024 is
// used only with clearance below it.
//
// Where 512 stops paying depends on K, because the K floats of LDS a row needs cap how many
// workgroups a CU holds. At K=8704 only one fits at either width, so wider wins out to large M; at
// K=3072 five fit, the grid fills the part, and past M~72 the narrow width is even. The thresholds
// below are those crossovers.
__host__ __device__ __forceinline__ int r3dh_act_threads(int M, int K) {
    if (M <= 32) return 1024;
    const int lim = K >= 8192 ? (1 << 24) : (K >= 4096 ? 120 : 72);
    return M <= lim ? 512 : 256;
}

// libr4d's r4d_cliff_rows pads a grid past the R9700's full-card cliff (a launch in the last 32
// wavefronts below a multiple of its 2048 wave slots runs ~5x slower). gfx1151 has no such cliff:
// measured 2026-10-26 with a fixed 200-iteration kernel at 256-thread workgroups, per-launch time
// rises smoothly through both candidate multiples (253..258 blocks: 10.18 -> 10.46 us, against the
// R9700's 27.6 vs 5.2), with only the ordinary second-round step at 161 blocks -- 160 blocks x 8
// waves = 1280 = 40 CUs x 2 SIMDs x 16 wave slots exactly. So the row count to launch is the row
// count asked for, and this stays a function because every launch site already calls it: if a
// future part or driver grows a cliff, the pad goes back here and nowhere else.
__host__ __device__ __forceinline__ uint32_t r3dh_cliff_rows(uint32_t rows, uint32_t per_row,
                                                            uint32_t waves) {
    (void)per_row;
    (void)waves;
    return rows;
}
