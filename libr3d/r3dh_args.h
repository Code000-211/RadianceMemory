/* Ported from libhalo/halo_args.h, verbatim but for the halo_ -> r3dh_ rename, so libr3d builds and loads
 * on its own. libhalo's gated-delta-net kernels are libr4d's ported to the gfx11 wave matrix and
 * checked against libref on gfx1151 (rad-kbench); they compile for gfx11-generic unchanged. */
/* Derived from libr4d/r4d_args.h by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. Op schemas, row names and stored weight layouts are
 * shared with libr4d by design. Everything else in this file is libhalo's own
 * and diverges as it is measured on gfx1151. */
/* r3dh_args.h -- how a libhalo entry point reads its arguments.
 *
 * There is no adapter layer. An entry point takes `(const RadArgs*, RadStream)` and returns a
 * status, the same RadLaunchFn contract every other plugin implements; an out-of-tree author can
 * write the same kernel without an adapter they could not reach.
 *
 * This header holds the shared helpers: reading a parameter, finding an operand, testing
 * contiguity. Most are one line over rad_abi.h.
 *
 * Nothing here includes halo.h or HIP: host translation units (the row table, the layouts, the
 * shape hooks) include it too, so those tables compile and can be checked without ROCm.
 */
#ifndef R3DH_ARGS_H
#define R3DH_ARGS_H

#include "rad_abi.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ================================================================== parameter access
 *
 * Forwards to rad_abi.h's static inline helpers, the ones a third-party plugin gets, keeping
 * libhalo's spelling at its call sites. */
static inline int r3dh_geti(const RadArgs* a, const char* k, long long* out) {
    return rad_args_geti(a, k, out);
}

static inline long long r3dh_geti_or(const RadArgs* a, const char* k, long long dflt) {
    return rad_args_geti_or(a, k, dflt);
}

static inline const char* r3dh_gets(const RadArgs* a, const char* k) {
    return rad_args_gets(a, k);
}

/* A float parameter is RAD_P_F64 (rad_types.h, docs/OPS.md "The four rules"), so `dval` arrives
 * and the step path runs no strtod. Deliberately laxer than rad_args_getf_or: string and integer
 * spellings are still read, because a tool or selftest may call a launch directly with no declare
 * to enforce the type. */
static inline double r3dh_getf(const RadArgs* a, const char* k, double dflt) {
    if (!a || !k) return dflt;
    for (int i = 0; i < a->n_p; ++i) {
        if (!a->p[i].key || std::strcmp(a->p[i].key, k) != 0) continue;
        if (a->p[i].kind == RAD_P_F64) return a->p[i].dval;
        if (a->p[i].kind == RAD_P_INT) return (double)a->p[i].ival;
        if (a->p[i].kind == RAD_P_STR && a->p[i].sval && a->p[i].sval[0]) {
            char* end = nullptr;
            const double v = std::strtod(a->p[i].sval, &end);
            return (end && end != a->p[i].sval) ? v : dflt;
        }
        return dflt;
    }
    return dflt;
}

/* An enumerated parameter: `act` is "silu" or "sigmoid", not 0 or 1, so a --debug-graph dump says
 * what runs; the mapping to libhalo's integer lives in the shim. Returns -1 for an unknown name,
 * which every caller turns into RAD_E_INVAL. `alt` is libhalo's own integer spelling of the same
 * key (`mode` on gated_had_quant_i8), consulted only when the name is absent. */
static inline int r3dh_act_code(const RadArgs* a, const char* key, const char* alt, int dflt) {
    if (const char* s = r3dh_gets(a, key)) {
        if (!std::strcmp(s, "silu") || !std::strcmp(s, "swish")) return 0;
        if (!std::strcmp(s, "sigmoid")) return 1;
        return -1;
    }
    long long v = 0;
    if (alt && r3dh_geti(a, alt, &v)) return (v == 0 || v == 1) ? (int)v : -1;
    return dflt;
}

/* Same, off a bare RadParam list (init() and the layout hooks); forwards to rad_plugin.h's
 * rad_param_*. */
static inline long long r3dh_pgeti_or(const RadParam* p, int n_p, const char* k, long long dflt) {
    return rad_param_geti(p, n_p, k, dflt);
}

static inline const char* r3dh_pgets(const RadParam* p, int n_p, const char* k) {
    return rad_param_gets(p, n_p, k, nullptr);
}

/* ================================================================== operand access
 *
 * An absent optional operand is a null RadTensor.data (rad_abi.h). r3dh_opt() returns null both
 * past the end and for a present-but-null operand, so an optional test is one pointer test. For a
 * required operand the caller turns null into RAD_E_INVAL: it is a bug in the issue, not a shape
 * the kernel declines.
 */
static inline const RadTensor* r3dh_opt(const RadArgs* a, int i) { return rad_arg_in(a, i); }

static inline const void* r3dh_p(const RadTensor* t) { return t ? t->data : nullptr; }

/* The same for an operand a kernel writes, so an output cast reads as one rather than as a
 * pointer laundered through `long`. */
static inline void* r3dh_wp(const RadTensor* t) { return t ? t->data : nullptr; }
static inline long r3dh_addr(const RadTensor* t) { return t ? (long)(uintptr_t)t->data : 0L; }

static inline long long r3dh_dim(const RadTensor* t, int i) {
    if (!t || i < 0 || (uint32_t)i >= t->rank) return 0;
    return t->shape[i];
}

static inline long long r3dh_stride(const RadTensor* t, int i) {
    if (!t || i < 0 || (uint32_t)i >= t->rank) return 0;
    return t->stride[i];
}

static inline long long r3dh_numel(const RadTensor* t) { return rad_tensor_numel(t); }

/* [rows..., K] flattened to (M, K). Every per-row kernel in libhalo owns whole rows (one
 * workgroup per row), so the leading axes are only a row count. The GEMM makes the same fold
 * explicitly (see gemm_ops). */

static inline int r3dh_rows_of(const RadTensor* t, long long* M, long long* K) {
    if (!t || t->rank < 1) return RAD_E_SHAPE;
    *K = t->shape[t->rank - 1];
    *M = 1;
    for (uint32_t i = 0; i + 1 < t->rank; ++i) *M *= t->shape[i];
    return (*M >= 1 && *K >= 1) ? RAD_OK : RAD_E_SHAPE;
}

/* Dense row-major over the whole tensor. libhalo's entry points take at most one pitch, so
 * everything else must be tight, and the shim checks it. This is the "necessary condition only"
 * boundary of spec §2.1: the constraint table settles geometry, contiguity is a property of the
 * call. */
static inline int r3dh_contig(const RadTensor* t) { return rad_tensor_is_contiguous(t); }

/* Contiguous in every axis but the leading one, whose stride is the row pitch libhalo takes as an
 * explicit argument (`xpitch`, `a_stride`, `xrow`, ...). Returns the pitch in elements, or -1. */
static inline long long r3dh_row_pitch(const RadTensor* t) {
    if (!t || t->rank == 0) return -1;
    if (t->rank == 1) return 1;
    long long acc = 1;
    for (int i = (int)t->rank - 1; i >= 1; --i) {
        if (t->shape[i] != 1 && t->stride[i] != acc) return -1;
        acc *= t->shape[i];
    }
    return t->stride[0];
}

/* [rows..., n] read as (rows, n): the leading axes are a row count and only the last is
 * structural. Returns the pitch in elements, or -1.
 *
 * A column slice must refuse. k out of a fused [tok, 2*kv_dim] projection, read as
 * [tok*n_head_kv, head_dim] for a per-head norm, has rows `n` apart within a token and a buffer
 * row apart between tokens; no single pitch describes that, and the leading stride would make the
 * consumer read most heads from garbage. So the trailing axes must cover the whole row. */
static inline long long r3dh_rows_pitch(const RadTensor* t, long long n, long long* rows) {
    if (!t || t->rank < 1 || n <= 0) return -1;
    const long long nel = rad_tensor_numel(t);
    if (nel % n) return -1;
    *rows = nel / n;
    if (t->rank == 1) return n;
    if (t->shape[t->rank - 1] == n) {
        if (t->stride[t->rank - 1] != 1) return -1;
        return t->rank >= 2 ? t->stride[t->rank - 2] : n;
    }
    if (rad_tensor_is_contiguous(t)) return n;
    long long trail = 1;
    for (int i = 1; i < (int)t->rank; ++i) trail *= t->shape[i];
    return trail == n ? r3dh_row_pitch(t) : -1;
}

/* An attention operand, [tokens, heads * head_dim] over its trailing axes: the token count, head
 * count and row pitch in elements. False when the trailing axes are not whole heads or no single
 * pitch describes the rows. */
static inline bool r3dh_head_rows(const RadTensor* t, long long head_dim, long long* n_tok,
                                 long long* n_head, long long* ld) {
    long long width = 1;
    for (uint32_t i = 1; i < t->rank; ++i) width *= t->shape[i];
    if (t->rank < 2 || width % head_dim) return false;
    *n_head = width / head_dim;
    *ld = r3dh_rows_pitch(t, width, n_tok);
    return *ld >= 0;
}

/* 16-byte alignment, needed by every kernel that pushes over P2P or issues a global_load_b128.
 * The kernels do not check it, so the shim does. */
static inline int r3dh_aligned(const void* p, unsigned bytes) {
    return p ? (((uintptr_t)p & (uintptr_t)(bytes - 1)) == 0) : 1;
}

/* ================================================================== which check refused
 *
 * A shim refuses in many places with the same few codes, so the engine's report names the kernel
 * and geometry but not the failing condition. R3DH_NO() wraps a refusal and prints its own line
 * when RAD_HALO_TRACE is set; otherwise it is the bare return plus one cached-flag load on a path
 * that is failing anyway. */
static inline int r3dh_trace_on(void) {
    static int on = -1;
    if (on < 0) { const char* e = std::getenv("RAD_HALO_TRACE"); on = (e && *e && *e != '0'); }
    return on;
}
static inline int r3dh_refused(int code, const char* file, int line) {
    if (r3dh_trace_on()) std::fprintf(stderr, "[r4d] %s:%d refused -> %d\n", file, line, code);
    return code;
}
#define R3DH_NO(code) r3dh_refused((code), __FILE__, __LINE__)

/* ================================================================== dtype helpers */
/* Widest world any libhalo all-reduce serves, fixed by the 8-rank entry points' argument lists
 * (seven explicit peers). */
enum { R3DH_MAX_RANKS = 8, R3DH_MAX_PEERS = 7 };

/* libhalo's all-reduce dtype code: 0 bf16, 1 fp16, 2 fp32. Returns -1 for anything else. */
static inline int r3dh_ar_dtype_code(const char* s) {
    if (!s) return -1;
    if (!std::strcmp(s, "bf16")) return 0;
    if (!std::strcmp(s, "fp16") || !std::strcmp(s, "f16")) return 1;
    if (!std::strcmp(s, "fp32") || !std::strcmp(s, "f32")) return 2;
    return -1;
}

static inline int r3dh_ar_esize(int code) { return code == 2 ? 4 : 2; }


/* ================================================================== weight layout constants
 *
 * libhalo's compile-time constants, restated because r3dh_layout.cpp is host code built without
 * ROCm and cannot include halo.h, and because the .rad container records them in the layout tag.
 * r3dh_gemm_args.h static_asserts the group constants against libhalo's macros.
 */
enum {
    R3DH_W4_GROUP = 128,   /* K per (scale, zero) dword; libhalo R3DH_GEMM_W4_GROUP  */
    R3DH_W4_KPB   = 64,    /* K per packed block: 4 k steps, 16 bytes per lane     */
    R3DH_W8_GROUP = 128,   /* libhalo R3DH_GEMM_W8_GROUP and R3DH_GEMM_W8A8_GROUP     */
    R3DH_W8_KPB   = 32,    /* K per packed block: 2 k steps, 16 bytes per lane     */
    R3DH_W2_GROUP = 128,   /* libhalo R3DH_GEMM_W2A8_GROUP                           */
    R3DH_MXFP4_GROUP = 32, /* K per E8M0 exponent, fixed by the OCP MX format      */
    /* The block-scaled fp8 family's block, on both axes. Not a quantiser choice: it is the
     * checkpoint's `quantization_config.weight_block_size`, and the bytes on disk follow it. It
     * cannot be static_asserted against `r3dh_gemm_fp8a8_block()` (a function, not a macro), so
     * r3dh_gemm_fp8a8.hip checks it at launch. */
    R3DH_FP8_BLOCK = 128,
    R3DH_WAVE     = 32,
    R3DH_NTILE    = 16     /* output rows per WMMA n-tile                          */
};

/* Plugin-private dtypes. The core never interprets these: rad_types.h reserves everything at or
 * above RAD_DT_PLUGIN_BASE for this, and the layout hook sizes them. */
enum {
    R3DH_DT_W4_FRAG    = RAD_DT_PLUGIN_BASE + 0,  /* 4-bit codes in WMMA fragment order      */
    R3DH_DT_W4_SZ      = RAD_DT_PLUGIN_BASE + 1,  /* (f16 scale, f16 -(1024+zero)) dwords    */
    R3DH_DT_W8_FRAG    = RAD_DT_PLUGIN_BASE + 2,  /* 8-bit codes in the same fragment order  */
    R3DH_DT_W8_S       = RAD_DT_PLUGIN_BASE + 3,  /* f16-only scale plane (w8a8)             */
    R3DH_DT_W2_FRAG    = RAD_DT_PLUGIN_BASE + 4,
    R3DH_DT_MXFP4_FRAG = RAD_DT_PLUGIN_BASE + 5,
    R3DH_DT_E8M0       = RAD_DT_PLUGIN_BASE + 6,  /* [K/32][N] shared exponents, and Wref[N]  */
    /* The grouped MoE GEMM's 4-bit expert weight: row-major packed nibbles, each a code into a
     * sixteen-entry table, expanded to E4M3 on the way into LDS. Not R3DH_DT_W4_FRAG, which is a
     * WMMA fragment permutation for a kernel loading its weight from global; this one stages
     * through LDS because its rows are gathered. See r3dh_layout.cpp. */
    R3DH_DT_MOEW4      = RAD_DT_PLUGIN_BASE + 7,
    /* The same GEMM's 5-bit expert weight: per row, R3DH_DT_MOEW4's nibble plane of the codes' low
     * four bits (the magnitude), then the sign bits, one dword per 32 elements in the order the
     * kernel merges them (r3dh_layout.cpp, moew5_row). A row is 5K/8 bytes. */
    R3DH_DT_MOEW5      = RAD_DT_PLUGIN_BASE + 8
};

/* The grouped MoE GEMM's code tables, as the four dwords w4_e4m3's byte permutes read
 * (r3dh_moe.hip): entries 0..3, 4..7, 8..11 and 12..15, one E4M3 byte an entry, lowest entry in the
 * low byte. `w4a8` and `w4a8h` decode a two's complement nibble, -8..7. `w4nla8h` decodes
 * libquant's `table=w4nl` (magnitudes ascending, then their negatives); kW4nlTable is that table
 * as a container stores it, which the layout hook compares with the container's table plane
 * before serving it. */
static constexpr uint32_t kW4IntPerm[4] = { 0x44403800u, 0x4E4C4A48u, 0xCACCCED0u, 0xB8C0C4C8u };
static constexpr uint32_t kW4nlPerm[4]  = { 0x504B4539u, 0x5B595653u, 0xD0CBC5B9u, 0xDBD9D6D3u };
static constexpr float kW4nlTable[16] = { 1.125f, 3.25f, 5.5f, 8.0f, 11.0f, 14.0f, 18.0f, 22.0f,
                                          -1.125f, -3.25f, -5.5f, -8.0f, -11.0f, -14.0f, -18.0f,
                                          -22.0f };

/* `w5nl64a8h` decodes libquant's `table=w5nl`: thirty-two levels, sixteen magnitudes ascending
 * then their negatives, so a code's high bit is its sign and its low four bits index the
 * magnitudes. The kernel expands the magnitude from kW5nlMagPerm (their E4M3 bytes in w4_e4m3's
 * dword order) and sets the sign bit from the sign plane. Every level is exact in E4M3. */
static constexpr uint32_t kW5nlMagPerm[4] = { 0x4B484235u, 0x5453514Fu, 0x5A595856u, 0x5F5D5C5Bu };
static constexpr float kW5nlTable[32] = {
    0.8125f, 2.5f, 4.0f, 5.5f, 7.5f, 9.0f, 11.0f, 12.0f,
    14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 30.0f,
    -0.8125f, -2.5f, -4.0f, -5.5f, -7.5f, -9.0f, -11.0f, -12.0f,
    -14.0f, -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -30.0f };

/* The fixed second-level scale of `w4nl64a8h`, `w4nl32a8h` and `w5nl64a8h`: it multiplies a
 * weight's per-(row, 64 or 32 of K) E4M3 scale. A container states it (libquant's scale2_value=),
 * and the layout hook compares it with this before serving. Fixed rather than per row because the
 * experts' scales (with the rotation's 1/128 folded in) lie between 2^-15 and 2^-10.6 on
 * Qwen3.8-Flash-Next at group 64 (slightly lower at 32, half a binade lower for w5nl). Under
 * 2^-13 each is a normal E4M3 value, and E4M3 has three mantissa bits in every binade. */
static constexpr float kW4G64Scale = 0x1p-13f;


/* ================================================================== skinny-GEMM variants
 *
 * libhalo's five skinny-GEMM knobs, exposed as RadVariant.config so `rad-tune` can walk them and
 * `--debug-graph` can name the pick (spec §15). The best setting depends on (N, K) and the M band.
 *
 * sk == 0 means "choose the largest legal split count at issue". Split-K requires
 * K % (SK * group) == 0, so no fixed default is legal at every shape, and an illegal default would
 * make an untuned install fail rather than run slowly (spec §15). The sentinel is resolved in one
 * place, r3dh_gemm_args.h:pick_sk.
 */
struct HaloGemmCfg { int wv, sk, mb, npw, nt; };

/* The tiled and prefill GEMMs take one opaque variant index into libhalo's own switch of compiled
 * tile arms; the row's tunable lists the arms a tuner may pick (r3dh_rows.cpp, kTileArms). */
struct HaloTileCfg { int variant; int bm; int bn; };

/* The MXFP4 tiled GEMM's column width in 32-column units per wave: TN=2 gives a 256 x 64 block,
 * TN=4 a 256 x 128 one. The crossover depends on M (see r3dh_gemm_mxfp4a8_tiled.hip), so 0, the
 * default, decides at issue; a fixed value is for a tuner. */
struct HaloMxTileCfg { int tn; };

/* The MXFP4 decode GEMM's split-K width, K slab and weight-load hint. 0 on each is the rule in
 * r3dh_gemm_mxfp4a8_decode.hip (it needs M and the shape, which exist only at issue); `nt` 1 streams
 * the weight past L2/MALL, 2 loads it plainly. */
struct HaloMxDecCfg { int ks; int bk; int nt; };

/* Split-KV segment count for the paged decode kernels. 0 is libhalo's split law (table in
 * r3dh_attn_paged_h256_gqa6.hip), which needs num_seqs and max_ctx from the issue, so it stays the
 * default; fixed counts are for a tuner. RadScratchFn gets the same variant index, so the arena
 * is sized for what this picks. */
struct HaloSplitCfg { int splits; };

/* The all-reduce launch geometry, passed to libhalo as nblocks/nthreads/drain/acq (and `pub` at
 * width). Defaults: nthreads=1024, drain=3 (s_wait_storecnt, which fine-grained uncached scratch
 * requires), acq=0, pub=1, and nblocks scaled by message size and clamped; nblocks == 0 selects
 * that rule. */
struct HaloArCfg { int nblocks; int nthreads; int drain; int acq; int pub; };

/* rowtopk: RC is how many candidates a chunk keeps (>= R), NCH the requested chunk width (libhalo
 * rounds it up to a multiple of eight columns, so the scratch hook asks r3dh_rowtopk_bf16_chunks()
 * for the real one), PT1 the stage-1 thread count. */
struct HaloTopkCfg { int rc; int nch; int pt1; };

/* The block-scaled fp8 matvec's rows a wave. Blocking trades waves for memory-level parallelism,
 * so the best value depends on whether the kernel waits on loads or the ALU. The default rule is
 * fitted to the current inner loop: any change to it (scale load width, activation dtype)
 * requires re-fitting. 0 selects that rule; other values pin it for rad-tune. */
struct HaloMvCfg { int rb; };



extern "C" {
/* -- the tuned configuration (r3dh_rows.cpp) --
 *
 * The core picked a point in the space the row declared and appended it to RadArgs::p. These read
 * it back, falling to the axis's default (defined once, in the RadTunable table) when a caller
 * supplied none, so a checker with no tuned key runs what an untuned install runs. They cannot
 * fail: the loader refuses a plugin whose default is not one of the axis's values. */
void r3dh_cfg_gemm   (const RadArgs*, struct HaloGemmCfg*);
void r3dh_cfg_tiled  (const RadArgs*, struct HaloTileCfg*);
void r3dh_cfg_prefill(const RadArgs*, struct HaloTileCfg*);
void r3dh_cfg_mxfp4_tiled(const RadArgs*, struct HaloMxTileCfg*);
void r3dh_cfg_mxfp4_decode(const RadArgs*, struct HaloMxDecCfg*);
void r3dh_cfg_split  (const RadArgs*, struct HaloSplitCfg*);
void r3dh_cfg_ar     (const RadArgs*, struct HaloArCfg*);
/* Whether the four-rank fused junction takes its tail form from 5 rows (the row's `tail` axis). */
int  r3dh_cfg_ar_tail(const RadArgs*);
/* The four-rank exact one-shot's size, in elements, from which it takes the exact two-shot instead
 * (0: never); the all-reduce rows' `ts_min` axis. */
long r3dh_cfg_ar_ts_min(const RadArgs*);
/* The prefetch's workgroup count (the row's `nb` axis). */
int  r3dh_cfg_prefetch_nb(const RadArgs*);
void r3dh_cfg_topk   (const RadArgs*, struct HaloTopkCfg*);
void r3dh_cfg_mv     (const RadArgs*, struct HaloMvCfg*);
/* rmsnorm_quant_i8's decode form: 1 the row's operands in registers before the reductions, 0 the
 * pass-per-read kernel; the row's `reg` axis. */
int  r3dh_cfg_rmsq8_reg(const RadArgs*);

}  /* extern "C" */

#endif  /* R3DH_ARGS_H */
