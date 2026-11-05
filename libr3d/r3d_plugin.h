/* r3d_plugin.h -- what libr3d's translation units share.
 *
 * The row table (r3d_rows.cpp) and the operand descriptions (r3d_shapes.cpp) are pure host C++ and
 * include only this file and the ABI; the launch shims include r3d.h and so <hip/hip_runtime.h>.
 * The split lets the constraint table compile and be checked without ROCm, which is why every shim
 * is declared here rather than beside its definition.
 *
 * Nothing in this header may include r3d.h.
 */
#ifndef RAD_R3D_PLUGIN_H
#define RAD_R3D_PLUGIN_H

#include "r3d_args.h"

/* ================================================================== the priority ladder
 *
 * The selector ranks candidate rows by an integer priority (spec §2.1); where two rows of one op
 * match a geometry, it decides which serves. Each group states the reason for its order. Values are
 * spaced by ten so a row can be inserted between two without renumbering.
 *
 * Most ops have a single row: the matrix work runs on the dot2 path, whose lane map takes the dtype
 * as a template argument, so the kernel that serves bf16 also serves f16. Where one op has two
 * rows, the reason is geometry and is stated next to the pair.
 */
enum {
    /* One row for the op; nothing competes with it. */
    R3D_PRIO_ONLY = 50,

    /* `attn_paged` is banded on q_len, and the decode row's bound (q_len <= 16) overlaps the
     * prefill row: a speculative verify at 8 query tokens matches both, and the decode kernel
     * (split-KV, one head group per workgroup, no query tile) is faster there. Decode outranks
     * prefill, as in libr4d. */
    R3D_PRIO_ATTN_DECODE  = 70,
    R3D_PRIO_ATTN_PREFILL = 60,

    /* The skinny bf16 GEMM's band is M <= 16 and the tiled one's starts at M >= 16. At 16 rows the
     * skinny kernel wins: it keeps the whole weight stream in flight with one wave per row and has
     * no LDS staging. The overlap is deliberate; a caller that wants the tiled kernel at 16 rows
     * asks with M=17, as at libr4d's m64/tiled boundary. */
    R3D_PRIO_GEMM_M16   = 60,
    R3D_PRIO_GEMM_TILED = 40,

    /* The vocab GEMM (N = the vocabulary): one row, M <= 16, split-K across the whole card. The
     * tiled GEMM has no row for these N, so this priority only matters against a future tiled
     * vocab row. */
    R3D_PRIO_VOCAB_M1 = 60,

    /* An fp8 weight row and an int8 weight row never compete: no dtype string reaches both, and
     * the container's encoding picks which is legal. */
    R3D_PRIO_GEMM_FP8 = 60,

    /* One row per sampler stage, on f32 logits, so there is nothing to rank yet. A second row for
     * a stage (an i8 logits form, say) would rank against this value. */
    R3D_PRIO_SAMPLE = 50
};

/* ================================================================== the dtype codes a C
 * launcher takes
 *
 * The launchers in r3d.h take a dtype as an int: it is a runtime value for the ops that are one
 * kernel over several dtypes (cast, fill, dequant) and a template argument everywhere else. The
 * codes are libr3d's own, not rad_types.h's RAD_DT_*, because the launcher signature is private to
 * a shim and its kernel and a dense 0..n switches more cleanly than the ABI's sparse enum. Every
 * shim translates. */
enum {
    R3D_DT_BF16 = 0,
    R3D_DT_F16  = 1,
    R3D_DT_F32  = 2,
    R3D_DT_I32  = 3,
    R3D_DT_I64  = 4,
    R3D_DT_U8   = 5,   /* byte codes: fp8 e4m3, read with the codec */
    R3D_DT_I8   = 6,
    R3D_DT_E5M2 = 7,
    R3D_DT_N    = 8
};

/* A RadTensor's dtype to the code above, or -1 for one libr3d has no kernel for. */
static inline int r3d_dt_code(uint32_t dt) {
    switch (dt) {
        case RAD_BF16:   return R3D_DT_BF16;
        case RAD_F16:    return R3D_DT_F16;
        case RAD_F32:    return R3D_DT_F32;
        case RAD_I32:
        case RAD_U32:    return R3D_DT_I32;
        case RAD_I64:    return R3D_DT_I64;
        case RAD_F8E4M3: return R3D_DT_U8;
        case RAD_F8E5M2: return R3D_DT_E5M2;
        case RAD_I8:     return R3D_DT_I8;
        case RAD_U8:     return R3D_DT_U8;
        default:         return -1;
    }
}

/* The bytes an element of that code occupies. */
static inline int r3d_dt_size(int code) {
    static const int kSize[R3D_DT_N] = { 2, 2, 4, 4, 8, 1, 1, 1 };
    return (code >= 0 && code < R3D_DT_N) ? kSize[code] : 0;
}

/* The activation an `act` parameter names, as the kernels' template argument. */
enum { R3D_ACT_SILU = 0, R3D_ACT_SIGMOID = 1, R3D_ACT_GELU = 2 };

/* rope's `mode`, numbered as libref's kModes. Only neox and gptj are implemented, and the rows'
 * constraints say so, so a model that needs mrope is not routed here. */
enum { R3D_ROPE_NEOX = 0, R3D_ROPE_GPTJ = 1 };

/* ================================================================== the row table's entry points
 *
 * Declared here so r3d_rows.cpp, host code built without ROCm, can name them. Each is defined and
 * documented beside its kernel, in the unit named below. A shim is a `RadLaunchFn`: operands read
 * with r3d_args.h, dtype from the tensor, row count from the operand. A shape hook is a
 * `RadShapeFn`: the extent and content domain of one operand (r3d_shapes.cpp).
 */
extern "C" {

/* --- r3d_model_bf16.hip --- */
int r3d_rad_rmsnorm       (const RadArgs* a, RadStream s);
int r3d_rad_layernorm     (const RadArgs* a, RadStream s);
int r3d_rad_add           (const RadArgs* a, RadStream s);
int r3d_rad_mul           (const RadArgs* a, RadStream s);
int r3d_rad_cast          (const RadArgs* a, RadStream s);
int r3d_rad_silu          (const RadArgs* a, RadStream s);
int r3d_rad_gelu          (const RadArgs* a, RadStream s);
int r3d_rad_sigmoid       (const RadArgs* a, RadStream s);
int r3d_rad_act_mul       (const RadArgs* a, RadStream s);
int r3d_rad_softmax       (const RadArgs* a, RadStream s);
int r3d_rad_scale         (const RadArgs* a, RadStream s);
int r3d_rad_fill          (const RadArgs* a, RadStream s);
int r3d_rad_rope          (const RadArgs* a, RadStream s);
int r3d_rad_kv_store      (const RadArgs* a, RadStream s);
int r3d_rad_embed_lookup  (const RadArgs* a, RadStream s);
int r3d_rad_gather_rows   (const RadArgs* a, RadStream s);
int r3d_rad_scatter_rows  (const RadArgs* a, RadStream s);

/* --- r3d_quant_fp8.hip --- */
int r3d_rad_quant_act_fp8   (const RadArgs* a, RadStream s);
int r3d_rad_quant_act_i8    (const RadArgs* a, RadStream s);
int r3d_rad_quant_act_i8g   (const RadArgs* a, RadStream s);
int r3d_rad_gated_quant_fp8 (const RadArgs* a, RadStream s);
int r3d_rad_dequant         (const RadArgs* a, RadStream s);

/* --- r3d_gemm_bf16_nt_m16.hip, r3d_gemm_f16_nt_m16.hip, r3d_gemm_fp8a16_nt_m1.hip --- */
int r3d_rad_gemm_nt_bf16_m16 (const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_f16_m16  (const RadArgs* a, RadStream s);

/* --- r3d_gemm_w4a16_nt_m64.hip: the 4-bit weight against an f16 activation at M <= 64. The
 * codes are read in the container's order; the layout and relayout hooks present the
 * (scale, zero) pair as the rad.planes joined operand, the rule's own form. --- */
int r3d_rad_gemm_nt_q_w4a16_m64 (const RadArgs* a, RadStream s);
int r3d_layout_gemm_nt_q_w4a16 (const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                const int* sel, const RadTensor* pl, int n, RadLayout* out);
int r3d_relayout_gemm_nt_q_w4a16(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                 const int* sel, const RadTensor* pl, int n, void* dst,
                                 int64_t dst_bytes);

/* --- r3d_gemm_w5a16_nt_m64.hip: the 5-bit weight (ParoQuant int5) against an f16
 * activation at M <= 64: the 4-bit row with the fifth-bit plane. --- */
int r3d_rad_gemm_nt_q_w5a16_m64 (const RadArgs* a, RadStream s);
int r3d_layout_gemm_nt_q_w5a16 (const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                const int* sel, const RadTensor* pl, int n, RadLayout* out);
int r3d_relayout_gemm_nt_q_w5a16(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                 const int* sel, const RadTensor* pl, int n, void* dst,
                                 int64_t dst_bytes);
int r3d_rad_gemm_nt_q_fp8    (const RadArgs* a, RadStream s);

/* --- r3dh_gdn_kkt_solve.hip (libhalo's): the gated-delta-net chunk preamble, the gram and the
 * blocked inverse of I + the strictly lower decayed gram, one 64-chunk a workgroup. --- */
extern "C" int r3dh_rad_gdn_kkt_solve(const RadArgs* a, RadStream s);

/* --- r3dh_gdn_chunk_scan.hip (libhalo's): the gated-delta-net chunked scan, the chunk's output
 * and its closing state, the recurrent state in wave-matrix accumulators. --- */
extern "C" int r3dh_rad_gdn_chunk_scan(const RadArgs* a, RadStream s);

/* --- r3d_gdn_gated_rmsnorm_h128.hip: the gated RMS norm a gated-delta-net layer applies to its
 * own output, one wave per (token, head) row. */
int r3d_rad_gdn_gated_rmsnorm(const RadArgs* a, RadStream s);

/* --- r3d_gdn_gated_norm_had_quant_i8.hip: the GDN block's gated RMS norm, block Hadamard and
 * per-row int8 quantiser in one pass. */
int r3d_rad_gdn_gated_norm_hq_i8(const RadArgs* a, RadStream s);

/* --- r3d_gdn_conv_w4_h128.hip: the GDN depthwise causal convolution (width 4, silu) and the
 * work fused around it: the prefill prep and the decode update. */
int r3d_rad_gdn_conv_prep(const RadArgs* a, RadStream s);
int r3d_rad_gdn_conv_update(const RadArgs* a, RadStream s);

/* --- r3dh_gdn_recurrent_update.hip (libhalo's): the gated-delta-net recurrent state update at
 * decode against the paged state cache, in either state form (per_candidate or anchor). */
extern "C" int r3dh_rad_gdn_recurrent_update(const RadArgs* a, RadStream s);

/* --- r3d_expert_scale_bf16.hip: each routing weight times its expert's own scale, in place. */
int r3d_rad_expert_scale_bf16(const RadArgs* a, RadStream s);

/* --- r3d_moe_router_topk_bf16.hip: softmax over the expert logits, the top_k, renormalised. */
int r3d_rad_moe_router_topk_bf16(const RadArgs* a, RadStream s);

/* --- r3d_moe_scatter.hip: the routed slots' stable counting sort (small form). */
int r3d_rad_moe_scatter(const RadArgs* a, RadStream s);
int r3d_rad_moe_scatter_inv(const RadArgs* a, RadStream s);
int64_t r3d_rad_moe_scatter_scratch(const RadArgs* a);

/* --- r3d_moe_gather.hip: fold the expert outputs back to the token. */
int r3d_rad_moe_gather(const RadArgs* a, RadStream s);
int r3d_rad_moe_gather_inv(const RadArgs* a, RadStream s);
int r3d_rad_router_topk_scatter(const RadArgs* a, RadStream s);
int r3d_rad_router_sigmoid_topk(const RadArgs* a, RadStream s);

/* --- r3d_moe_gemm_q.hip: the grouped quantised GEMM (E4M3 x E4M3, the per-(row, grid) scale fold). */
int r3d_rad_moe_gemm_q(const RadArgs* a, RadStream s);
int r3d_rad_moe_gemm_q_alu(const RadArgs* a, RadStream s);
int r3d_rad_moe_gemm_bf16(const RadArgs* a, RadStream s);
int r3d_rad_moe_gemm_fp8_f32scale(const RadArgs* a, RadStream s);
int r3d_rad_moe_gemm_mxfp4a8(const RadArgs* a, RadStream s);

/* --- r3d_moe_gemm_w4a8.hip: the grouped GEMM with int4-weights / int8-activation experts. */
int r3d_rad_moe_gemm_w4a8(const RadArgs* a, RadStream s);

/* --- r3d_moe_gemm_w4a4.hip: the grouped GEMM with int4-weights / int4-activation experts. */
int r3d_rad_moe_gemm_w4a4(const RadArgs* a, RadStream s);
int r3d_rad_dequant_w4(const RadArgs* a, RadStream s);
int r3d_rad_row_topk_merge(const RadArgs* a, RadStream s);
int r3d_rad_ngram_ids(const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_bias_fp8(const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_short_fp8(const RadArgs* a, RadStream s);
int r3d_rad_moe_gemm_q_mod4(const RadArgs* a, RadStream s);

/* --- r3d_had_quant_act_i8.hip: a block Hadamard along K, then per-row symmetric int8. */
int r3d_rad_had_quant_act_i8(const RadArgs* a, RadStream s);

/* --- r3d_gelu_mul_quant_i8.hip: gelu(gate) * up, then per-row symmetric int8. */
int r3d_rad_gelu_mul_quant_i8(const RadArgs* a, RadStream s);

/* --- r3d_kv_len_trim.hip: per-sequence context lengths less `trim`. */
int r3d_rad_kv_len_trim(const RadArgs* a, RadStream s);

/* --- r3d_qsa_tail_store.hip: the QSA indexer's tail, into a per-layer linear KV state. */
int r3d_rad_qsa_tail_store(const RadArgs* a, RadStream s);

/* --- r3d_gated_had_quant_i8.hip: a gated product, block Hadamard, then per-row int8. */
int r3d_rad_gated_had_quant_i8(const RadArgs* a, RadStream s);

/* --- r3d_had_quant_act_fp8.hip: a 128-wide block Hadamard, then per-group E4M3. */
int r3d_rad_had_quant_act_fp8(const RadArgs* a, RadStream s);

/* --- r3d_gated_had_quant_fp8.hip: a gated product, 128-wide Hadamard, then per-group E4M3. */
int r3d_rad_gated_had_quant_fp8(const RadArgs* a, RadStream s);

/* --- r3d_quant_cols_fp8.hip: E4M3 codes with one scale per column over every row. */
int64_t r3d_quant_cols_fp8_scratch(const RadArgs* a);
int r3d_rad_quant_cols_fp8(const RadArgs* a, RadStream s);

/* --- r3d_had_quant_act_i8g.hip: a 128-wide block Hadamard, then per-group int8 (row-major). */
int r3d_rad_had_quant_act_i8g(const RadArgs* a, RadStream s);

/* --- r3d_gated_had_quant_i8g.hip: a gated product, 128-wide Hadamard, then per-group int8. */
int r3d_rad_gated_had_quant_i8g(const RadArgs* a, RadStream s);

/* --- r3d_rmsnorm_quant_i8.hip: the RMS norm and the per-row int8 quantiser, one pass. */
int r3d_rad_rmsnorm_quant_i8(const RadArgs* a, RadStream s);

/* --- r3d_shapes.cpp --- */
int r3d_shape_rmsnorm         (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_layernorm       (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_add             (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_mul             (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_act_mul         (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_cast            (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_silu            (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gelu            (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_sigmoid         (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_softmax         (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_scale           (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_fill            (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_rope            (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_kv_store        (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_embed_lookup    (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gather_rows     (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_scatter_rows    (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_quant_act_fp8   (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_quant_act_i8    (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_had_quant_act_i8(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gelu_mul_quant_i8(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_kv_len_trim(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_qsa_tail_store(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gated_had_quant_i8(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_had_quant_act_fp8(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gated_had_quant_fp8(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_quant_cols_fp8(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_had_quant_act_i8g(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gated_had_quant_i8g(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_rmsnorm_quant_i8(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_quant_act_i8g   (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gated_quant_fp8 (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_dequant         (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gemm_nt         (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gemm_nt_q       (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gemm_nt_q_mxfp4a8(const RadParam* p, int n_p, int operand, RadOpdDesc* out);

/* --- r3d_shapes.cpp: the fused-op, attention and sampler shape hooks --- */
extern "C" int r3d_shape_silu_mul          (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_attn_paged        (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_gemm_nt_q_gated (const RadParam*, int, int, RadOpdDesc*);
/* r3d_gemm_nt_q_w2a8.hip: the w2a8 vocab projection's layout hook, one of the library's three
 * (with the fp8 weight's and the logits head's, below). It repacks nothing: it presents the
 * container's `scale` and `zero` planes as the single joined `b_scale` operand the way rad_planes
 * stores them, transcribed from ref_layout_gemm_q. Without it the plan would have a placement with
 * no reader. */
extern "C" int r3d_layout_gemm_nt_q_w2a8(const RadParam*, int, int, const RadEncoding*,
                                         const int* sel, const RadTensor* planes, int n_planes,
                                         RadLayout*);
extern "C" int r3d_shape_dflash_select     (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_row_topk          (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_argmax     (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_temp       (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_topk       (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_topp       (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_minp       (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_typical    (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_xtc        (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_pick       (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_penalties  (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_dry        (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_ngram_ban  (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_mask       (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_logprobs   (const RadParam*, int, int, RadOpdDesc*);

/* --- r3d_gemm_nt_q_w2a8.hip, r3d_gemm_fp8a8_gated.hip, r3d_dflash_f32.hip; r3d_sample_f32.hip's
     draw spine (row_topk, argmax, temp, topk, pick, the four narrowings) and
     r3d_sample_pre_f32.hip's pre-filters (penalties, dry, ngram_ban, mask, logprobs) --- */
int r3d_rad_gemm_nt_q_w2a8     (const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_gated  (const RadArgs* a, RadStream s);
int r3d_rad_dflash_select      (const RadArgs* a, RadStream s);
int r3d_rad_row_topk           (const RadArgs* a, RadStream s);
int r3d_rad_sample_argmax      (const RadArgs* a, RadStream s);
int r3d_rad_sample_temp        (const RadArgs* a, RadStream s);
int r3d_rad_sample_topk        (const RadArgs* a, RadStream s);
int r3d_rad_sample_pick        (const RadArgs* a, RadStream s);
int r3d_rad_sample_topp        (const RadArgs* a, RadStream s);
int r3d_rad_sample_minp        (const RadArgs* a, RadStream s);
int r3d_rad_sample_typical     (const RadArgs* a, RadStream s);
int r3d_rad_sample_xtc         (const RadArgs* a, RadStream s);
int r3d_rad_sample_penalties   (const RadArgs* a, RadStream s);
int r3d_rad_sample_dry         (const RadArgs* a, RadStream s);
int r3d_rad_sample_ngram_ban   (const RadArgs* a, RadStream s);
int r3d_rad_sample_mask        (const RadArgs* a, RadStream s);
int r3d_rad_sample_logprobs    (const RadArgs* a, RadStream s);

/* --- r3d_attn_paged_h128.hip: the paged attention, banded on q_len. The decode kernel is
 * split-KV over the cache with one head group per workgroup; the prefill kernel stages a 16-token
 * query tile. */
int r3d_rad_attn_decode_h128       (const RadArgs* a, RadStream s);
int r3d_rad_attn_prefill_h128      (const RadArgs* a, RadStream s);

/* --- r3d_gemm_fp8a8_tiled.hip: the quantised GEMM above the decode heights, and the reader for
 * the container's fp8 weight: codes and per-128x128 scale as two planes behind `b_scale`. The row
 * carries its own layout hook because the reference's gemm_nt_q hook answers nothing for this
 * dtype (ref_layout_gemm_q reads zp_grid, which has no fp8a8 entry). */
int r3d_rad_gemm_nt_q_fp8_tiled  (const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_fp8_tiled_alu(const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_w2a8_tiled  (const RadArgs* a, RadStream s);
extern "C" int r3d_layout_gemm_q_fp8_weight(const RadParam*, int, int, const RadEncoding*,
                                            const int*, const RadTensor*, int, RadLayout*);

/* --- r3d_gemm_mxfp4a8.hip: the OCP-MXFP4 weight against the fp8 activation, both heights. The
 * weight is two plain u8 planes (e2m1 codes [N, K/2], e8m0 scale [N, K/32]) read in place, so the
 * row has no layout hook. Its shape hook is separate because these planes are not the fp8 grid's
 * E4M3 codes and block scale. */
int r3d_rad_gemm_nt_q_mxfp4a8_m16  (const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_mxfp4a8_tiled(const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_mxfp4a8_m16_alu  (const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_mxfp4a8_tiled_alu(const RadArgs* a, RadStream s);

/* --- r3d_gemm_inta8.hip: the symmetric int4 and int8 weight grids at decode heights. The weight
 * is signed codes (I4 two per byte, or I8) with a plain f16 scale per (row, 128 of K). No zero
 * point and no a_sum, so both planes are read in place and the rows have no layout hook. */
int r3d_rad_gemm_nt_q_w4a8_m16(const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_w8a8_m16(const RadArgs* a, RadStream s);
int r3d_shape_gemm_nt_q_w4a8(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gemm_nt_q_w8a8(const RadParam* p, int n_p, int operand, RadOpdDesc* out);

/* The relayout writers for the three layout hooks, each delegating to rad_planes_relayout. The
 * loader converts a quantised weight into the joined operand the plan names, and only the row that
 * reads it can say how. See abi/rad_abi.h. */
extern "C" int r3d_relayout_logits_head(const RadParam*, int, int, const RadEncoding*, const int*,
                                        const RadTensor*, int, void*, int64_t);
extern "C" int r3d_relayout_gemm_q_fp8_weight(const RadParam*, int, int, const RadEncoding*,
                                              const int*, const RadTensor*, int, void*, int64_t);
extern "C" int r3d_relayout_gemm_nt_q_w2a8(const RadParam*, int, int, const RadEncoding*,
                                           const int*, const RadTensor*, int, void*, int64_t);

/* --- r3d_vocab_bf16.hip: the vocabulary projection, both heights. Its shape hook
 * (r3d_shapes.cpp) transcribes ref_logits_gemm_shape, three plain tensors with nothing joined, so
 * the band the engine declares is the band kbench fills. */
int r3d_rad_logits_gemm_bf16     (const RadArgs* a, RadStream s);
int r3d_rad_logits_gemm_bf16_alu (const RadArgs* a, RadStream s);
/* The head's layout hook: `output.weight` at a llama_dense_fp8 checkpoint is E4M3 codes and a bf16
 * scale per 128x128 block, stored as two container planes behind the one `lm_head` operand, and the
 * mapped tier does not place a weight a loaded library cannot read in place. Transcribed from
 * ref_layout_head, with the statics it uses inlined. It moves no bytes: a round-trip through its
 * relayout writer (above) reproduces the planes exactly, and the writer exists for that
 * round-trip. */
extern "C" int r3d_layout_logits_head(const RadParam*, int, int, const RadEncoding*,
                                      const int*, const RadTensor*, int, RadLayout*);
int r3d_rad_gemm_nt_bf16_tiled   (const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_bf16_tiled_alu(const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_bf16_tiled_wmma(const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_bf16_m64_wmma (const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_w8a8_m64   (const RadArgs* a, RadStream s);
int r3d_shape_logits_gemm_bf16  (const RadParam*, int, int, RadOpdDesc*);

/* --- r3d_model_bf16.hip: the fused MLP activation and the fused norm-quantiser --- */
int r3d_rad_silu_mul             (const RadArgs* a, RadStream s);
int r3d_rad_rmsnorm_quant_fp8    (const RadArgs* a, RadStream s);
int r3d_shape_rmsnorm_quant_fp8  (const RadParam*, int, int, RadOpdDesc*);

}  /* extern "C" */

#endif  /* RAD_R3D_PLUGIN_H */
