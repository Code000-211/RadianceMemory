/* r3d_rows.cpp -- the libr3d row table: what each kernel computes and for which geometry.
 *
 * One row per (kernel, band). A row is a `RadKernelInfo`: names, prose, the geometry it serves as
 * constraints, the tuning axes and the entry points that launch it. The engine reads this table at
 * load, matches a requested op and geometry against it and ranks the candidates by `priority`. It
 * never calls into this file to select, and nothing outside libr3d names libr3d's kernels.
 *
 * ============================================================ the schema table
 *
 * The first plugin in hierarchy order to declare an op fixes its schema, and a plugin whose schema
 * disagrees is refused at load (rad_abi.h, spec §2.3): arguments are positional, so two schemas
 * would give one call two meanings. libref declares the vocabulary (`docs/OPS.md`), so every schema
 * below is libref's: the same keys, types, requiredness, operand roles and optionality, in the same
 * order. The prose describes this library's rows; the loader compares only the positional slots.
 *
 * libr3d declares no new op: an op declared only by a kernel plugin has no reference to be checked
 * against.
 *
 * ============================================================ the constraint table
 *
 * A constraint is a necessary condition on the geometry (spec §2.1). An operand's dtype is a
 * property of the call, and the shim checks it on the tensor. Constraints state what the kernel
 * cannot do (the divisibility its lane map assumes, the band its grid is sized for, the scheme its
 * decoders implement) and what the selector needs to choose between rows. A constraint the kernel
 * does not actually honour is a promise the tests will break.
 *
 * ============================================================ rows are positional
 *
 * `RadKernelInfo` is initialised positionally, as `rad_abi.h` requires: a field added anywhere but
 * the end shifts every later initialiser, and because `nullptr` converts to any function pointer,
 * part of the shift still compiles. Unused trailing slots are left off and zero-initialised. A row's
 * entry points are, in order: init, fini, launch, scratch, layout, relayout, opd_shape, unrelayout,
 * replaces.
 *
 * The `layout` and `relayout` hooks here only present the container's own planes as the joined
 * operands they store (the w2a8 row's `b_scale`, the fp8 weight's codes and scale, the logits head)
 * and repack nothing, so a container converted on this card keeps row-major codes and scales. A
 * dot2 GEMM reads a weight in stored order; libr4d's fragment permutations exist for WMMA, which
 * these kernels do not use (see r3d_common.h).
 *
 * Pure host C++ (no r3d.h, no HIP), so the table compiles and can be tested without ROCm.
 */

#include "r3d_plugin.h"
#include "rad_abi_layout.h"

#ifndef R3D_BUILD_TARGET
#define R3D_BUILD_TARGET "gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151"
#endif

#include <dirent.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define ROW(a)  (a), (int)(sizeof(a) / sizeof((a)[0]))
#define NELEM(a) ((int)(sizeof(a) / sizeof((a)[0])))

/* Parameter and constraint spellings, shared with libref's table. */
#define P_INT(k)   { (k), RAD_P_INT, RAD_REQUIRED }
#define P_INTO(k)  { (k), RAD_P_INT, RAD_OPTIONAL }
#define P_INTD(k)  { (k), RAD_P_INT, RAD_DERIVED }
/* The sequence-chunk parameter: derived, and the one the core reads by meaning (RAD_PROLE_SEQ_CHUNK).
 * Roles are part of a schema, so this must match libref's spelling, not only P_INTD's. */
#define P_CHUNK(k) { (k), RAD_P_INT, RAD_DERIVED, RAD_PROLE_SEQ_CHUNK }
#define P_STR(k)   { (k), RAD_P_STR, RAD_REQUIRED }
#define P_STRO(k)  { (k), RAD_P_STR, RAD_OPTIONAL }
#define P_F64(k)   { (k), RAD_P_F64, RAD_REQUIRED }
#define P_F64O(k)  { (k), RAD_P_F64, RAD_OPTIONAL }

#define C_EQ(k, v)  { (k), RAD_C_EQ,  (long long)(v), nullptr }
#define C_LE(k, v)  { (k), RAD_C_LE,  (long long)(v), nullptr }
#define C_GE(k, v)  { (k), RAD_C_GE,  (long long)(v), nullptr }
#define C_DIV(k, v) { (k), RAD_C_DIV, (long long)(v), nullptr }
#define C_IN(k, s)  { (k), RAD_C_IN,  0, (s) }

/* A row with one implementation and nothing to measure. */
#define NOTUNE nullptr, 0, nullptr

#define SCHEMA(op, p, o, doc) { (op), ROW(p), ROW(o), (doc) }

/* ================================================================== schemas
 *
 * Operand names are diagnostics only; roles and optionality are positional and must match.
 * Operand names may differ from libref's (`b` for a gate, say) without changing the schema.
 */

/* --- norms and elementwise --- */
static const RadParamSpec pRmsNorm[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype"),
                                         P_F64O("wadd"), P_F64O("mult") };
static const RadOperandSpec oRmsNorm[] = { { "x", RAD_OPD_IN, 0 },
                                           { "w", RAD_OPD_WEIGHT, 1 },
                                           { "y", RAD_OPD_OUT, 0 } };

static const RadParamSpec pLayernorm[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype") };
static const RadOperandSpec oLayernorm[] = { { "x", RAD_OPD_IN, 0 },
                                             { "w", RAD_OPD_WEIGHT, 0 },
                                             { "b", RAD_OPD_WEIGHT, 1 },
                                             { "y", RAD_OPD_OUT, 0 } };

static const RadParamSpec pAdd[] = { P_INT("M"), P_INT("n"), P_STR("dtype"), P_F64O("div") };
static const RadParamSpec pMnDtype[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
static const RadOperandSpec oBinary[] = { { "a", RAD_OPD_IN, 0 }, { "b", RAD_OPD_IN, 0 },
                                          { "y", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oUnary[] = { { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };

static const RadParamSpec pActMul[] = { P_INT("M"), P_INT("n"), P_STR("act"), P_STR("dtype") };

static const RadParamSpec pCast[] = { P_INT("M"), P_INT("n"), P_STR("from"), P_STR("to") };

static const RadParamSpec pScale[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
/* `s` is a weight operand, as the reference declares it: the issue resolves a handle and waits on
 * the mover, because a scale plane comes from the container or a quantiser, not from a buffer the
 * planner allocated for the step. The loader compares roles positionally, so declaring it IN would
 * get the whole library refused. */
static const RadOperandSpec oScale[] = { { "x", RAD_OPD_IN, 0 }, { "s", RAD_OPD_WEIGHT, 0 },
                                         { "y", RAD_OPD_OUT, 0 } };

static const RadParamSpec pFill[] = { P_INT("M"), P_INT("n"), P_STR("dtype"), P_F64O("value") };
static const RadOperandSpec oFill[] = { { "y", RAD_OPD_OUT, 0 } };

/* --- the model's odds and ends --- */
static const RadParamSpec pRope[] = { P_INT("M"), P_INT("head_dim"), P_INT("n_head"),
                                      P_INT("n_head_kv"), P_F64("theta"), P_F64("scale"),
                                      P_STR("mode"), P_INTO("rotary_dim"), P_STRO("sections"),
                                      P_F64O("yarn_factor"), P_F64O("yarn_low"),
                                      P_F64O("yarn_high"), P_F64O("yarn_magnitude"),
                                      P_STRO("ladder"), P_F64O("llama3_factor"),
                                      P_F64O("llama3_low"), P_F64O("llama3_high"),
                                      P_F64O("llama3_orig") };
static const RadOperandSpec oRope[] = { { "qkv", RAD_OPD_INOUT, 0 },
                                        { "positions", RAD_OPD_IN, 0 } };

static const RadParamSpec pKvStore[] = { P_INT("M"), P_INT("head_dim"), P_INT("n_head_kv"),
                                         P_INT("block_size"), P_STR("kv_dtype"),
                                         P_F64O("k_scale"), P_F64O("v_scale") };
static const RadOperandSpec oKvStore[] = { { "k", RAD_OPD_IN, 0 }, { "v", RAD_OPD_IN, 0 },
                                           { "slot_mapping", RAD_OPD_IN, 0 },
                                           { "kv_cache", RAD_OPD_INOUT, 0 } };

static const RadParamSpec pEmbed[] = { P_INT("M"), P_INT("n_embd"), P_INT("n_vocab"),
                                       P_STR("dtype"), P_INTO("vocab_offset"), P_F64O("wscale"),
                                       P_F64O("mult") };
static const RadOperandSpec oEmbed[] = { { "tokens", RAD_OPD_IN, 0 },
                                         { "wte", RAD_OPD_WEIGHT, 0 },
                                         { "x", RAD_OPD_OUT, 0 } };

static const RadOperandSpec oGather[] = { { "x", RAD_OPD_IN, 0 }, { "idx", RAD_OPD_IN, 0 },
                                          { "y", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oScatter[] = { { "v", RAD_OPD_IN, 0 }, { "idx", RAD_OPD_IN, 0 },
                                           { "x", RAD_OPD_INOUT, 0 } };

/* --- quantisation --- */
static const RadParamSpec pQuant[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype") };
static const RadOperandSpec oQuantFp8[] = { { "x", RAD_OPD_IN, 0 }, { "q", RAD_OPD_OUT, 0 },
                                            { "scale", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oQuant[] = { { "x", RAD_OPD_IN, 0 }, { "q", RAD_OPD_OUT, 0 },
                                         { "scale", RAD_OPD_OUT, 0 },
                                         { "asum", RAD_OPD_OUT, 1 } };

static const RadParamSpec pGatedQuant[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype"),
                                            P_STRO("act") };

 /* router_topk_scatter: the pair as one op -- logits, the selection (expert_ids, expert_w), and
  * the sort (sorted_tok, expert_offset, expert_count). */
static const RadOperandSpec oRouterTopkScatter[] = { { "logits", RAD_OPD_IN, 0 },
                                                     { "expert_ids", RAD_OPD_OUT, 0 },
                                                     { "expert_w", RAD_OPD_OUT, 0 },
                                                     { "sorted_tok", RAD_OPD_OUT, 0 },
                                                     { "expert_offset", RAD_OPD_OUT, 0 },
                                                     { "expert_count", RAD_OPD_OUT, 0 } };
static const RadParamSpec pRouterTopkScatter[] = { P_INT("M"), P_INT("n_expert"), P_INTO("n_local"),
                                                   P_INT("top_k"), P_INT("norm"), P_STR("dtype"),
                                                   P_INTO("expert_base"), P_STRO("protect") };

 /* router_sigmoid_topk: the GLM-style router -- logits, the per-expert correction bias, and the
  * selection (expert_ids, expert_w). */
static const RadOperandSpec oRouterSigmoid[] = { { "logits", RAD_OPD_IN, 0 },
                                                 { "bias", RAD_OPD_WEIGHT, 0 },
                                                 { "ids", RAD_OPD_OUT, 0 },
                                                 { "weights", RAD_OPD_OUT, 0 } };
static const RadParamSpec pRouterSigmoid[] = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
                                               P_INT("norm"), P_F64("route_scale"), P_STR("dtype") };
static const RadOperandSpec oGatedQuant[] = { { "gate_up", RAD_OPD_IN, 0 },
                                              { "q", RAD_OPD_OUT, 0 },
                                              { "scale", RAD_OPD_OUT, 0 },
                                              { "out_bf16", RAD_OPD_OUT, 1 } };
static const RadOperandSpec oGeluMulQuantI8[] = { { "gate_up", RAD_OPD_IN, 0 },
                                                  { "q", RAD_OPD_OUT, 0 },
                                                  { "scale", RAD_OPD_OUT, 0 },
                                                  { "out_bf16", RAD_OPD_OUT, 1 } };
static const RadParamSpec pKvLenTrim[] = { P_INT("trim"), P_INT("max_seqs") };
static const RadOperandSpec oKvLenTrim[] = { { "seqused", RAD_OPD_IN, 0 },
                                             { "tail", RAD_OPD_IN, 0 },
                                             { "out", RAD_OPD_OUT, 0 } };
static const RadParamSpec pQsaTailStore[] = { P_INT("M"), P_INT("n"), P_INT("ratio"),
                                              P_INT("ring"), P_INT("seqs"), P_STR("dtype") };
static const RadOperandSpec oQsaTailStore[] = { { "k", RAD_OPD_IN, 0 },
                                                { "pos", RAD_OPD_IN, 0 },
                                                { "cu", RAD_OPD_IN, 0 },
                                                { "state", RAD_OPD_INOUT, 0 },
                                                { "sidx", RAD_OPD_IN, 0 } };
static const RadParamSpec pGatedHadQuantI8[] = { P_INT("M"), P_INT("n"), P_INT("group"),
                                                 P_STR("dtype"), P_STRO("act"), P_INTO("mode") };
static const RadOperandSpec oGatedHadQuantI8[] = { { "a", RAD_OPD_IN, 0 },
                                                   { "b", RAD_OPD_IN, 1 },
                                                   { "q", RAD_OPD_OUT, 0 },
                                                   { "scale", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oQuantCols[] = { { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 },
                                             { "scale", RAD_OPD_OUT, 0 } };
static const RadParamSpec pRmsQuantI8[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype"),
                                            P_F64O("wadd") };
static const RadOperandSpec oRmsQuantI8[] = { { "x", RAD_OPD_IN, 0 },
                                              { "residual", RAD_OPD_INOUT, 1 },
                                              { "w", RAD_OPD_WEIGHT, 0 },
                                              { "q", RAD_OPD_OUT, 0 },
                                              { "scale", RAD_OPD_OUT, 0 },
                                              { "out_bf16", RAD_OPD_OUT, 1 },
                                              { "w_post", RAD_OPD_WEIGHT, 1 },
                                              { "scalar", RAD_OPD_WEIGHT, 1 } };

static const RadOperandSpec oDequant[] = { { "q", RAD_OPD_IN, 0 }, { "scale", RAD_OPD_IN, 0 },
                                           { "y", RAD_OPD_OUT, 0 } };

/* --- gemm --- */
static const RadParamSpec pGemm[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_STR("dtype") };
static const RadOperandSpec oGemm[] = { { "a", RAD_OPD_IN, 0 }, { "b", RAD_OPD_WEIGHT, 0 },
                                        { "y", RAD_OPD_OUT, 0 } };

/* `w_rb` is libr4d's (docs/OPS.md), accepted and ignored: an int8 weight's per-row scale plane
 * holds each block's value in every row, and the trellis forms that need it are IQR dtypes, which
 * no row here serves. */
static const RadParamSpec pGemmQ[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("group"),
                                       P_STR("dtype"), P_INTO("pb1"), P_INTO("pb2"),
                                       P_INTO("split_n"), P_INTO("w_rb") };
static const RadOperandSpec oGemmQ[] = { { "a", RAD_OPD_IN, 0 }, { "a_scale", RAD_OPD_IN, 1 },
                                         { "b", RAD_OPD_WEIGHT, 0 },
                                         { "b_scale", RAD_OPD_WEIGHT, 0 },
                                         { "y", RAD_OPD_OUT, 0 }, { "a_sum", RAD_OPD_IN, 1 },
                                         { "b_ref", RAD_OPD_WEIGHT, 1 } };

/* --- gated-delta-net --- */
static const RadParamSpec pGdnKktSolve[] = { P_INT("M"), P_INT("head_k"), P_CHUNK("chunk") };
static const RadOperandSpec oGdnKktSolve[] = { { "k", RAD_OPD_IN, 0 },
                                               { "beta", RAD_OPD_IN, 0 },
                                               { "g", RAD_OPD_IN, 0 },
                                               { "cu", RAD_OPD_IN, 0 },
                                               { "A", RAD_OPD_OUT, 0 } };

static const RadParamSpec pGdnChunkScan[] = { P_INT("M"), P_INT("head_k"), P_INT("head_v"),
                                              P_CHUNK("chunk"), P_F64O("scale"), P_INTO("state_fp16") };
static const RadOperandSpec oGdnChunkScan[] = { { "q", RAD_OPD_IN, 0 },
                                                { "k", RAD_OPD_IN, 0 },
                                                { "v", RAD_OPD_IN, 0 },
                                                { "A", RAD_OPD_IN, 0 },
                                                { "g", RAD_OPD_IN, 0 },
                                                { "beta", RAD_OPD_IN, 0 },
                                                { "h0", RAD_OPD_IN, 1 },
                                                { "cu", RAD_OPD_IN, 0 },
                                                { "o", RAD_OPD_OUT, 0 },
                                                { "ht", RAD_OPD_OUT, 1 },
                                                { "state_idx", RAD_OPD_IN, 1 } };

static const RadParamSpec pGdnGatedRmsnorm[] = { P_INT("M"), P_INT("channels"),
                                                P_F64("eps"), P_STR("act") };
static const RadOperandSpec oGdnGatedRmsnorm[] = { { "x", RAD_OPD_IN, 0 },
                                                  { "z", RAD_OPD_IN, 0 },
                                                  { "w", RAD_OPD_WEIGHT, 0 },
                                                  { "o", RAD_OPD_OUT, 0 } };

static const RadParamSpec pGdnGatedNormHq[] = { P_INT("M"), P_INT("n"), P_INT("head_dim"),
                                               P_INT("group"), P_F64("eps"), P_STR("dtype"),
                                               P_INTO("zgrp"), P_INTO("zblk") };
static const RadOperandSpec oGdnGatedNormHq[] = { { "x", RAD_OPD_IN, 0 },
                                                 { "z", RAD_OPD_IN, 0 },
                                                 { "w", RAD_OPD_WEIGHT, 0 },
                                                 { "q", RAD_OPD_OUT, 0 },
                                                 { "scale", RAD_OPD_OUT, 0 } };

static const RadParamSpec pGdnConvPrep[] = { P_INT("M"), P_INT("head_k"), P_INT("head_v"),
                                            P_CHUNK("chunk"), P_INT("conv_width"),
                                            P_F64O("l2_eps"), P_F64O("softplus_thr") };
static const RadOperandSpec oGdnConvPrep[] = { { "x", RAD_OPD_IN, 0 },
                                              { "w", RAD_OPD_WEIGHT, 0 },
                                              { "b", RAD_OPD_WEIGHT, 1 },
                                              { "A_log", RAD_OPD_WEIGHT, 0 },
                                              { "dt_bias", RAD_OPD_WEIGHT, 0 },
                                              { "conv_state", RAD_OPD_INOUT, 0 },
                                              { "cu", RAD_OPD_IN, 0 },
                                              { "a", RAD_OPD_IN, 1 },
                                              { "b_gate", RAD_OPD_IN, 1 },
                                              { "cache_idx", RAD_OPD_IN, 1 },
                                              { "has_init", RAD_OPD_IN, 1 },
                                              { "q", RAD_OPD_OUT, 0 },
                                              { "k", RAD_OPD_OUT, 0 },
                                              { "v", RAD_OPD_OUT, 0 },
                                              { "g", RAD_OPD_OUT, 0 },
                                              { "beta", RAD_OPD_OUT, 0 } };

static const RadParamSpec pGdnConvUpdate[] = { P_INT("q_len"), P_INT("head_k"), P_INT("head_v"),
                                              P_INT("conv_width"), P_INTO("max_query_len") };
static const RadOperandSpec oGdnConvUpdate[] = { { "x", RAD_OPD_IN, 0 },
                                                { "w", RAD_OPD_WEIGHT, 0 },
                                                { "b", RAD_OPD_WEIGHT, 1 },
                                                { "conv_state", RAD_OPD_INOUT, 0 },
                                                { "state_idx", RAD_OPD_IN, 0 },
                                                { "num_accepted", RAD_OPD_IN, 0 },
                                                { "cu", RAD_OPD_IN, 0 },
                                                { "q", RAD_OPD_OUT, 0 },
                                                { "k", RAD_OPD_OUT, 0 },
                                                { "v", RAD_OPD_OUT, 0 } };

static const RadParamSpec pExpertScale[] = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
                                             P_STR("dtype") };
static const RadOperandSpec oExpertScale[] = { { "ids", RAD_OPD_IN, 0 },
                                               { "w_e", RAD_OPD_WEIGHT, 0 },
                                               { "ew", RAD_OPD_INOUT, 0 } };

static const RadParamSpec pMoeRouterTopk[] = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
                                               P_INT("norm"), P_STR("dtype") };
static const RadOperandSpec oMoeRouterTopk[] = { { "logits", RAD_OPD_IN, 0 },
                                                 { "expert_ids", RAD_OPD_OUT, 0 },
                                                 { "expert_w", RAD_OPD_OUT, 0 } };

static const RadParamSpec pMoeScatter[] = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
                                            P_INTO("expert_base"), P_STRO("protect") };
static const RadOperandSpec oMoeScatter[] = { { "expert_ids", RAD_OPD_IN, 0 },
                                              { "sorted_tok", RAD_OPD_OUT, 0 },
                                              { "expert_offset", RAD_OPD_OUT, 0 },
                                              { "expert_count", RAD_OPD_OUT, 0 } };
 /* moe_gather: fold the expert outputs back to the token. y_expert [Te, n], expert_w [Mw, top_k],
  * the slot table (sorted_tok), the optional shared arm, and y [M, n]. */
static const RadOperandSpec oMoeGather[] = { { "y_expert", RAD_OPD_IN, 0 },
                                              { "expert_w", RAD_OPD_IN, 0 },
                                              { "sorted_tok", RAD_OPD_IN, 0 },
                                              { "shared", RAD_OPD_IN, 1 },
                                              { "shared_gate", RAD_OPD_IN, 1 },
                                              { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pMoeGather[] = { P_INT("M"), P_INT("n"), P_INT("top_k"), P_STR("dtype"),
                                           P_STRO("act") };

/* moe_gemm_q: the grouped quantised GEMM, libref's schema. The experts are weight tables
 * (RAD_OPD_WTAB): the even experts in `w` / `w_scale`, the odd in `w_odd` / `w_odd_scale`, expert e
 * at entry e/2 (r3d_moe_tab.h). */
static const RadOperandSpec oMoeGemmQ[] = { { "a", RAD_OPD_IN, 0 },
                                            { "a_scale", RAD_OPD_IN, 1 },
                                            { "w", RAD_OPD_WTAB, 0 },
                                            { "w_scale", RAD_OPD_WTAB, 0 },
                                            { "sorted_tok", RAD_OPD_IN, 0 },
                                            { "expert_offset", RAD_OPD_IN, 0 },
                                            { "w_odd", RAD_OPD_WTAB, 0 },
                                            { "w_odd_scale", RAD_OPD_WTAB, 0 },
                                            { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pMoeGemmQ[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"),
                                          P_INT("top_k"), P_INT("group"), P_STR("a_order"),
                                          P_STR("dtype"), P_INTO("N_odd"), P_INTO("K_odd"),
                                          P_INTO("parts") };

/* moe_gemm_q dtype w4a8: the grouped int4 GEMM (r3d_moe_gemm_w4a8.hip) is a row of the moe_gemm_q
 * op above (two's-complement int4 experts, a bf16 scale per (row, 128 of K), bf16 out; the int8 or
 * E4M3 activation arm is picked by `a`'s dtype). It reuses moe_gemm_q's nine-operand parity schema. */

 /* moe_gemm: the grouped GEMM with unquantised experts, libref's schema; one weight table. */
static const RadOperandSpec oMoeGemm[] = { { "a", RAD_OPD_IN, 0 },
                                          { "w", RAD_OPD_WTAB, 0 },
                                          { "sorted_tok", RAD_OPD_IN, 0 },
                                          { "expert_offset", RAD_OPD_IN, 0 },
                                          { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pMoeGemm[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"),
                                         P_INT("top_k"), P_STR("a_order"), P_STR("dtype") };

static const RadParamSpec pGdnRecurrent[] = { P_INT("q_len"), P_INT("head_k"), P_INT("head_v"),
                                             P_INTO("n_head_v"), P_F64O("scale"), P_F64O("eps"),
                                             P_F64O("l2_eps"), P_F64O("softplus_thr"),
                                             P_STRO("act"), P_INTO("state_fp16"), P_STRO("state_form") };
static const RadOperandSpec oGdnRecurrent[] = { { "q", RAD_OPD_IN, 0 },
                                               { "k", RAD_OPD_IN, 0 },
                                               { "v", RAD_OPD_IN, 0 },
                                               { "a", RAD_OPD_IN, 0 },
                                               { "b", RAD_OPD_IN, 0 },
                                               { "A_log", RAD_OPD_WEIGHT, 0 },
                                               { "dt_bias", RAD_OPD_WEIGHT, 0 },
                                               { "state", RAD_OPD_INOUT, 0 },
                                               { "state_idx", RAD_OPD_IN, 0 },
                                               { "num_accepted", RAD_OPD_IN, 0 },
                                               { "cu", RAD_OPD_IN, 0 },
                                               { "z", RAD_OPD_IN, 1 },
                                               { "norm_w", RAD_OPD_WEIGHT, 1 },
                                               { "o", RAD_OPD_OUT, 0 },
                                               { "o_q", RAD_OPD_OUT, 1 },
                                               { "o_scale", RAD_OPD_OUT, 1 } };

/* gdn_conv_recurrent_update: the decode conv and the recurrent update as one kernel, the
 * convolution's operands then the recurrence's minus q, k and v (never materialised). */
static const RadParamSpec pGdnConvRecurrent[] = { P_INT("q_len"), P_INT("head_k"), P_INT("head_v"),
                                                 P_INT("conv_width"), P_INTO("n_head_v"),
                                                 P_F64O("scale"), P_F64O("eps"), P_F64O("l2_eps"),
                                                 P_F64O("softplus_thr"), P_STRO("act"),
                                                 P_INTO("state_fp16"), P_STRO("state_form") };
static const RadOperandSpec oGdnConvRecurrent[] = { { "x", RAD_OPD_IN, 0 },
                                                   { "conv_w", RAD_OPD_WEIGHT, 0 },
                                                   { "conv_b", RAD_OPD_WEIGHT, 1 },
                                                   { "conv_state", RAD_OPD_INOUT, 0 },
                                                   { "conv_idx", RAD_OPD_IN, 0 },
                                                   { "a", RAD_OPD_IN, 0 },
                                                   { "b", RAD_OPD_IN, 0 },
                                                   { "A_log", RAD_OPD_WEIGHT, 0 },
                                                   { "dt_bias", RAD_OPD_WEIGHT, 0 },
                                                   { "state", RAD_OPD_INOUT, 0 },
                                                   { "state_idx", RAD_OPD_IN, 0 },
                                                   { "num_accepted", RAD_OPD_IN, 0 },
                                                   { "cu", RAD_OPD_IN, 0 },
                                                   { "z", RAD_OPD_IN, 0 },
                                                   { "norm_w", RAD_OPD_WEIGHT, 0 },
                                                   { "o", RAD_OPD_OUT, 0 },
                                                   { "o_q", RAD_OPD_OUT, 1 },
                                                   { "o_scale", RAD_OPD_OUT, 1 } };


/* ================================================================== the vocab and the sampler
 *
 * The vocab and sampler schemas, transcribed from libref. `r3d_selftest`'s schema phase compares
 * every declared schema against libref's, slot for slot, when the library loads, so a misspelled
 * key is caught at load rather than when a row first uses it.
 *
 * The sampler's behaviour is fully specified: the candidate set, the top-k tie-break, the order of
 * the renormalisations and the seeded draw are all observable, and libref is the oracle for each.
 */

/* The gated MLP's activation as llama-family graphs emit it: one fused [M, 2n] projection, the
 * gate half activated and multiplied by the other half. `act_mul` is the same arithmetic over two
 * operands; an architecture emits one op or the other. */
static const RadOperandSpec oGateUp[] = { { "gate_up", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };

/* The gated MLP band: one weight of N = 2n rows (gate first, then up), with the GEMM, the gate and
 * the output quantiser in one launch so the [M, N] intermediate is never stored. `act` is a required
 * string because an implementation compiles one gate; an optional key would let a row resolve for a
 * gate it does not have. */
static const RadParamSpec pGemmQGated[] = { P_INT("M"), P_INT("N"), P_INT("K"),
                                            P_INT("group"), P_STR("dtype"), P_STR("act") };
/* The vocabulary projection's schema, slot for slot with the reference's: the geometry is the
 * head's (`n_vocab` x `n_embd`), the dtype is the activation's, and `logits` is f32, which is why
 * the op is separate from gemm_nt. */
static const RadParamSpec pLogits[] = { P_INT("M"), P_INT("n_vocab"), P_INT("n_embd"),
                                        P_STR("dtype") };
static const RadOperandSpec oLogits[] = {
    { "x", RAD_OPD_IN, 0 }, { "lm_head", RAD_OPD_WEIGHT, 0 }, { "logits", RAD_OPD_OUT, 0 } };

static const RadOperandSpec oGemmQGated[] = {
    { "a", RAD_OPD_IN, 0 },        { "a_scale", RAD_OPD_IN, 0 },
    { "b", RAD_OPD_WEIGHT, 0 },    { "b_scale", RAD_OPD_WEIGHT, 0 },
    { "q", RAD_OPD_OUT, 0 },       { "scale", RAD_OPD_OUT, 0 },
    { "out_bf16", RAD_OPD_OUT, 1 },
};

/* DFlash2's candidate walk: a draft block of `steps` proposals with `top_k` candidates each,
 * rescored against the target's rank-d codebooks. `anchor_stride` is optional and defaults to 1.
 * `hp` is an optional per-edge scaling of the predecessor's codebook row; absent, it is a plane of
 * ones, as the reference computes. */
static const RadParamSpec pDflashSelect[] = { P_INT("M"), P_INT("steps"), P_INT("top_k"),
                                              P_INT("rank"), P_INT("n_vocab"),
                                              P_INTO("anchor_stride"), P_STR("dtype") };
static const RadOperandSpec oDflashSelect[] = {
    { "cand", RAD_OPD_IN, 0 },    { "unary", RAD_OPD_IN, 0 },
    { "hp", RAD_OPD_IN, 1 },      { "anchor", RAD_OPD_IN, 0 },
    { "pred", RAD_OPD_WEIGHT, 0 },{ "succ", RAD_OPD_WEIGHT, 0 },
    { "tokens", RAD_OPD_OUT, 0 },
};

/* Paged attention. `block_size` is RAD_DERIVED (the container's cache geometry, admitted by a
 * constraint that only refuses its absence). `n_head`, `max_seqs`, `max_ctx` and `prefill_fp8` are
 * listed because a plugin's scratch hook sizes the split-KV buffer from them and both libraries must
 * agree on the parameter list; libref ignores them. */
static const RadParamSpec pAttnPaged[] = {
    P_INT("q_len"), P_INT("head_dim"), P_INT("gqa"), P_INTD("block_size"), P_INT("causal"),
    P_INT("window"), P_STR("q_dtype"), P_STR("kv_dtype"), P_F64O("scale"), P_INTO("max_ctx"),
    P_INTO("n_head"), P_INTO("max_seqs"), P_INTO("prefill_fp8")
};
static const RadOperandSpec oAttnPaged[] = {
    { "q", RAD_OPD_IN, 0 }, { "kv_cache", RAD_OPD_IN, 0 }, { "block_table", RAD_OPD_IN, 0 },
    { "seqused", RAD_OPD_IN, 0 }, { "k_descale", RAD_OPD_IN, 1 }, { "v_descale", RAD_OPD_IN, 1 },
    { "cu_seqlens", RAD_OPD_IN, 1 }, { "out", RAD_OPD_OUT, 0 }
};

/* The row's top-R over a vocabulary slice, and the sampler's stages. The history stages write this
 * rank's columns and read global ids, so they carry `vocab_off`; the narrowing stages work on the
 * candidate set the top-k produced and need only its width. */
static const RadParamSpec pRowTopk[]  = { P_INT("M"), P_INT("N"), P_INT("R"), P_INTO("vocab_off"),
                                          P_STR("dtype") };
static const RadOperandSpec oRowTopk[] = { { "x", RAD_OPD_IN, 0 }, { "idx", RAD_OPD_OUT, 0 },
                                           { "val", RAD_OPD_OUT, 0 },
                                           { "pairs", RAD_OPD_OUT, 1 } };
static const RadParamSpec pMVocab[]  = { P_INT("M"), P_INT("n_vocab") };
static const RadParamSpec pMCand[]   = { P_INT("M"), P_INT("n_cand") };
static const RadParamSpec pPenal[]   = { P_INT("M"), P_INT("n_vocab"), P_INTO("vocab_off") };
static const RadParamSpec pTopk[]    = { P_INT("M"), P_INT("n_vocab"), P_INTO("vocab_off") };
static const RadParamSpec pLogprobs[] = { P_INT("M"), P_INT("n_vocab"), P_INT("k"),
                                          P_INTO("vocab_off") };
static const RadOperandSpec oTemp[]      = { { "logits", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 } };
static const RadOperandSpec oMask[]      = { { "logits", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "bitmask", RAD_OPD_IN, 0 } };
static const RadOperandSpec oPenalties[] = { { "logits", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "history", RAD_OPD_IN, 0 } };
static const RadOperandSpec oDry[]       = { { "logits", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "history", RAD_OPD_IN, 0 },
                                             { "breakers", RAD_OPD_IN, 1 } };
static const RadOperandSpec oNgramBan[]  = { { "logits", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "history", RAD_OPD_IN, 0 },
                                             { "ngram", RAD_OPD_IN, 0 },
                                             { "whitelist", RAD_OPD_IN, 1 } };
static const RadOperandSpec oTopk[]      = { { "logits", RAD_OPD_IN, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "cand_idx", RAD_OPD_OUT, 0 },
                                             { "cand_val", RAD_OPD_OUT, 0 },
                                             { "pairs", RAD_OPD_OUT, 1 } };
static const RadOperandSpec oArgmax[]    = { { "logits", RAD_OPD_IN, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "token", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oLogprobs[]  = { { "logits", RAD_OPD_IN, 0 },
                                             { "params", RAD_OPD_IN, 1 },
                                             { "stats", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oNarrow[]    = { { "cand_idx", RAD_OPD_INOUT, 0 },
                                             { "cand_val", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 } };
static const RadOperandSpec oPick[]      = { { "cand_idx", RAD_OPD_IN, 0 },
                                             { "cand_val", RAD_OPD_IN, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "token", RAD_OPD_OUT, 0 } };


/* The fused norm-quantiser from the model unit, libref's slot for slot. `residual` is INOUT because
 * the sum it stores is the one the variance reads. */
static const RadParamSpec pRmsQuantFp8[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_INT("group"),
                                             P_STR("dtype"), P_F64O("wadd") };
/* `residual` is optional because the op has a no-residual form (the launcher's `res` pointer is 0);
 * libref declares it INOUT optional and the loader compares the flag. */
static const RadOperandSpec oRmsQuantFp8[] = {
    { "x", RAD_OPD_IN, 0 },        { "residual", RAD_OPD_INOUT, 1 }, { "w", RAD_OPD_WEIGHT, 0 },
    { "q", RAD_OPD_OUT, 0 },       { "scale", RAD_OPD_OUT, 0 },      { "out_bf16", RAD_OPD_OUT, 1 }
};

/* --- new Tier-1 ops --- */
static const RadParamSpec pDequantW4[] = { P_INT("N"), P_INT("K"), P_INT("group"), P_STR("dtype"),
                                           P_INT("twos") };
static const RadOperandSpec oDequantW4[] = { { "wq", RAD_OPD_WEIGHT, 0 },
                                             { "ws", RAD_OPD_WEIGHT, 0 },
                                             { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pRowTopkMerge[] = { P_INT("M"), P_INT("R"), P_INT("world_size"),
                                              P_STR("dtype") };
static const RadOperandSpec oRowTopkMerge[] = { { "gathered", RAD_OPD_IN, 0 },
                                                { "idx", RAD_OPD_OUT, 0 },
                                                { "val", RAD_OPD_OUT, 0 } };
static const RadParamSpec pNgramIds[] = { P_INT("M"), P_INT("heads"), P_INT("ngram"), P_INT("eos") };
static const RadOperandSpec oNgramIds[] = { { "tok", RAD_OPD_IN, 0 },
                                            { "state", RAD_OPD_INOUT, 0 },
                                            { "cu", RAD_OPD_IN, 0 },
                                            { "mult", RAD_OPD_WEIGHT, 0 },
                                            { "vocab_sizes", RAD_OPD_WEIGHT, 0 },
                                            { "offsets", RAD_OPD_WEIGHT, 0 },
                                            { "cache_idx", RAD_OPD_IN, 1 },
                                            { "has_init", RAD_OPD_IN, 1 },
                                            { "num_accepted", RAD_OPD_IN, 1 },
                                            { "ids", RAD_OPD_OUT, 0 },
                                            { "ahead_ids", RAD_OPD_OUT, 1 } };
static const RadParamSpec pGemmQBias[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("group"), P_STR("dtype") };
static const RadOperandSpec oGemmQBias[] = { { "a", RAD_OPD_IN, 0 }, { "a_scale", RAD_OPD_IN, 0 },
                                             { "b", RAD_OPD_WEIGHT, 0 },
                                             { "b_scale", RAD_OPD_WEIGHT, 0 },
                                             { "bias", RAD_OPD_WEIGHT, 0 },
                                             { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pGemmQShort[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("group"),
                                            P_STR("dtype"), P_INTO("pb1"), P_INTO("pb2"),
                                            P_INTO("split_n"), P_INTO("w_rb") };
static const RadOperandSpec oGemmQShort[] = { { "a", RAD_OPD_IN, 0 }, { "a_scale", RAD_OPD_IN, 1 },
                                              { "b", RAD_OPD_WEIGHT, 0 },
                                              { "b_scale", RAD_OPD_WEIGHT, 0 },
                                              { "y", RAD_OPD_OUT, 0 },
                                              { "a_sum", RAD_OPD_IN, 1 },
                                              { "b_ref", RAD_OPD_WEIGHT, 1 } };
static const RadParamSpec pMoeGemmQMod4[] = { P_INT("M"), P_INT("N"), P_INT("K"),
                                              P_INT("n_expert"), P_INT("top_k"), P_INT("group"),
                                              P_STR("a_order"), P_STR("dtype"), P_INTO("N_1"),
                                              P_INTO("K_1"), P_INTO("N_2"), P_INTO("K_2"),
                                              P_INTO("N_3"), P_INTO("K_3"), P_INTO("parts") };
static const RadOperandSpec oMoeGemmQMod4[] = { { "a", RAD_OPD_IN, 0 }, { "a_scale", RAD_OPD_IN, 1 },
                                                { "w", RAD_OPD_WTAB, 0 }, { "w_scale", RAD_OPD_WTAB, 0 },
                                                { "sorted_tok", RAD_OPD_IN, 0 }, { "expert_offset", RAD_OPD_IN, 0 },
                                                { "w_1", RAD_OPD_WTAB, 0 }, { "w_1_scale", RAD_OPD_WTAB, 0 },
                                                { "w_2", RAD_OPD_WTAB, 0 }, { "w_2_scale", RAD_OPD_WTAB, 0 },
                                                { "w_3", RAD_OPD_WTAB, 0 }, { "w_3_scale", RAD_OPD_WTAB, 0 },
                                                { "y", RAD_OPD_OUT, 0 } };

static const RadOpSchema kSchemas[] = {
    SCHEMA("rmsnorm", pRmsNorm, oRmsNorm,
           "y = x * rsqrt(mean(x^2) + eps) * (w + wadd) * mult. `wadd` is GemmaRMSNorm's `1 +`, "
           "added to the gain in f32; `mult` scales the normed row before its one rounding. No "
           "gain operand at all means a gain of 1 and a `wadd` of 0."),
    SCHEMA("layernorm", pLayernorm, oLayernorm,
           "y = (x - mean) / sqrt(var + eps) * w + b, the mean and the variance taken in two "
           "passes over the row."),
    SCHEMA("add", pAdd, oBinary,
           "y = a + b. `b` carrying exactly n elements against an output of M rows broadcasts "
           "along rows, and nothing else broadcasts at all. With `div`, y = rd(rd(a + b) / div)."),
    SCHEMA("mul", pMnDtype, oBinary, "y = a * b, with the same one broadcast rule as `add`."),
    SCHEMA("act_mul", pActMul, oBinary,
           "y = act(a) * b over two [M, n] operands, `act` naming silu, gelu (the exact erf form), "
           "gelu_tanh, sigmoid or quick_gelu. The feed-forward's gated middle."),
    SCHEMA("silu", pMnDtype, oUnary, "y = x * sigmoid(x)."),
    SCHEMA("gelu", pMnDtype, oUnary,
           "y = 0.5 * x * (1 + erf(x / sqrt(2))), the exact form; the tanh approximation is a "
           "different op name."),
    SCHEMA("sigmoid", pMnDtype, oUnary, "y = 1 / (1 + exp(-x))."),
    SCHEMA("softmax", pMnDtype, oUnary, "Row-wise, f32, with the maximum subtracted."),
    SCHEMA("cast", pCast, oUnary,
           "y = x, from the dtype `from` to the dtype `to`, one rounding on the way. An integer "
           "pair goes through without touching f32, so ids above 2^24 survive."),
    SCHEMA("scale", pScale, oScale,
           "y = x * s, `s` one element broadcast over the whole output. A row of scales is "
           "`mul_weight`."),
    SCHEMA("fill", pFill, oFill, "y = value over [M, n]. The output is the only operand."),
    SCHEMA("rope", pRope, oRope,
           "Rotary positions applied in place to a fused qkv row: q heads, then the key heads, the "
           "value heads untouched. `mode` is neox, gptj, mrope, imrope or axial; the yarn, llama3 "
           "and ladder parameters adjust the frequency ladder."),
    SCHEMA("kv_store", pKvStore, oKvStore,
           "K and V into a paged cache at slot_mapping's slots, converting to the cache's dtype. A "
           "negative slot is skipped; the cache is [blocks, kv_heads, block_size, 2*head_dim]."),
    SCHEMA("embed_lookup", pEmbed, oEmbed,
           "x[r] = wte[tokens[r] - vocab_offset] * wscale * mult, the id global and the table "
           "sharded by `vocab_offset`. An id outside the shard's range yields a zero row."),
    SCHEMA("gather_rows", pMnDtype, oGather, "y[r] = x[idx[r]], rows of n elements."),
    SCHEMA("scatter_rows", pMnDtype, oScatter, "x[idx[r]] = v[r], rows of n elements."),
    SCHEMA("quant_act_fp8", pQuant, oQuantFp8,
           "Symmetric per-(row, group) E4M3: scale = amax * (1/448) with the reciprocal "
           "multiplied, not divided; an all-zero group gets scale 1 and the codes 0."),
    SCHEMA("quant_act_i8", pQuant, oQuant,
           "Symmetric int8 with one scale per row, max(amax, 1e-8) / 127, codes round-to-nearest-"
           "ties-to-even and saturating, stored in the step order the int8 WMMA reads. `asum` takes "
           "the group-major row sums the zero-pointed GEMMs reduce."),
    SCHEMA("had_quant_act_i8", pQuant, oQuantFp8,
           "A block Hadamard along K (width `group`), then per-row symmetric int8: the rotation "
           "shrinks the row maximum, so int8 beats e4m3 on these activations."),
    SCHEMA("gelu_mul_quant_i8", pMnDtype, oGeluMulQuantI8,
           "gelu(gate) * up over a fused [M, 2n] projection, then per-row symmetric int8: the "
           "tanh-form product rounded to bf16, the codes and scale taken over those values."),
    SCHEMA("kv_len_trim", pKvLenTrim, oKvLenTrim,
           "A batch's per-sequence context lengths less `trim`, floored at zero: the prefill "
           "overlap's slice sees its own last row, not the step's. `trim` is the tail's extent."),
    SCHEMA("qsa_tail_store", pQsaTailStore, oQsaTailStore,
           "The QSA indexer's tail: the raw index keys of tokens that have not completed a block, "
           "into a per-layer linear KV state. Position p lands at row p % ring; only the last "
           "ring tokens of a sequence's chunk write."),
    SCHEMA("gated_had_quant_i8", pGatedHadQuantI8, oGatedHadQuantI8,
           "A gated product (mode 0 silu(a)*b, mode 1 sigmoid(a)*b), a block Hadamard along n, "
           "then per-row symmetric int8: the rotation shrinks the row maximum, so int8 beats e4m3."),
    SCHEMA("had_quant_act_fp8", pQuant, oQuantFp8,
           "A 128-wide block Hadamard, then per-group E4M3: the rotation shrinks the group maximum "
           "so e4m3's exponent bits carry the outliers better. A separate op, not a parameter."),
    SCHEMA("gated_had_quant_fp8", pGatedQuant, oGatedQuant,
           "A gated product (silu or sigmoid), a 128-wide block Hadamard, then per-group E4M3: the "
           "rotation shrinks the group maximum so e4m3's exponent bits carry the outliers better."),
    SCHEMA("quant_cols_fp8", pMnDtype, oQuantCols,
           "E4M3 codes with one scale per column over every row: an attention's V for its fp8 P.V, "
           "the group running down a column. Two passes over the rows, split into chunks."),
    SCHEMA("had_quant_act_i8g", pQuant, oQuantFp8,
           "A 128-wide block Hadamard, then per-group int8: the rotated form for the int8 trunk "
           "GEMMs, the codes row-major (the i8a8 dot2 GEMM reads stored order)."),
    SCHEMA("gated_had_quant_i8g", pGatedQuant, oGatedQuant,
           "A gated product (silu or sigmoid), a 128-wide block Hadamard, then per-group int8: the "
           "rotated int8 down projection, the codes row-major for the i8a8 dot2 GEMM."),
    SCHEMA("rmsnorm_quant_i8", pRmsQuantI8, oRmsQuantI8,
           "The RMS norm and the per-row int8 quantiser in one pass, optionally with the residual "
           "junction in front (a post-norm, the residual add, a scalar): the sandwich-norm chain "
           "as one dispatch, byte-identical to the unfused chain."),
    SCHEMA("quant_act_i8g", pQuant, oQuantFp8,
           "The same int8 code at `group` granularity: one f32 scale per (row, group), amax / 127, "
           "1 for an all-zero group, codes row-major."),
    SCHEMA("gated_quant_fp8", pGatedQuant, oGatedQuant,
           "act(gate) * up over a fused [M, 2n] projection, quantised in the same pass: the codes "
           "are taken over the rounded product, which `out_bf16` receives when present."),
    SCHEMA("dequant", pQuant, oDequant,
           "y = code * scale, the read of a container's own quantised weight: `group`-wide f32 "
           "scales against E4M3, E5M2 or int8 codes."),
    SCHEMA("gemm_nt", pGemm, oGemm, "y[M,N] = a[M,K] @ b[N,K]^T, f32 accumulate."),
    SCHEMA("gemm_nt_q", pGemmQ, oGemmQ,
           "The same product against a quantised weight. `dtype` names the scheme: fp8a16 a bf16 "
           "activation against E4M3 codes, fp8a8 quant_act_fp8's codes and scales, i8a8 int8 on "
           "both axes. `group` is the activation's scale width; `pb1` and `pb2` the weight's block, "
           "which the container states. `a_sum` and `b_ref` are the row sums a zero-pointed form "
           "reduces."),

    SCHEMA("gdn_kkt_solve", pGdnKktSolve, oGdnKktSolve,
           "A = (I + strict_lower(diag(beta) K K^T e^{g_i - g_j}))^-1 per 64-chunk: the gram once "
           "per k head, the blocked inverse 16 -> 32 -> 64."),

    SCHEMA("gdn_chunk_scan", pGdnChunkScan, oGdnChunkScan,
           "The gated delta net's chunked scan: the chunk's output and its closing state, the "
           "recurrent state held in the wave matrix across the chunk loop."),

    SCHEMA("gdn_gated_rmsnorm", pGdnGatedRmsnorm, oGdnGatedRmsnorm,
            "The gated RMS norm a gated-delta-net layer applies to its own output: "
            "x * rsqrt(mean(x^2) + eps) * w * act(z), one (token, head) row of `channels`."),

    SCHEMA("gdn_gated_norm_had_quant_i8", pGdnGatedNormHq, oGdnGatedNormHq,
            "The GDN block's gated RMS norm, block Hadamard and per-row int8 quantiser in one pass: "
            "x * rsqrt(mean_head(x^2) + eps) * w * silu(z), rotated over `group` and quantised to "
            "int8 with a per-row f32 `scale`."),

    SCHEMA("gdn_conv_prep", pGdnConvPrep, oGdnConvPrep,
            "The GDN depthwise causal convolution (width 4, silu) plus the split, l2norm, gate "
            "(g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b)) and the per-chunk cumsum of "
            "g, in one prefill kernel; the conv state is read at a sequence's first chunk and "
            "rewritten from its last tokens."),

    SCHEMA("gdn_conv_update", pGdnConvUpdate, oGdnConvUpdate,
            "The GDN depthwise causal convolution at decode: a speculative window over a rolling "
            "conv-state buffer, read at the slot the last accepted token left and rewritten "
            "shifted by one; writes q, k and v in their own layouts."),

    SCHEMA("gdn_recurrent_update", pGdnRecurrent, oGdnRecurrent,
            "The gated-delta-net recurrent state update at decode: g = -exp(A_log) softplus(a + "
            "dt_bias), beta = sigmoid(b), q/k l2-normalised, S *= e^g, u = beta (v - S k), S += u "
            "k^T, o = S q, per token against a paged fp32 state cache; the gated RMS norm is "
            "folded in when `z` and `norm_w` are present."),

    SCHEMA("gdn_conv_recurrent_update", pGdnConvRecurrent, oGdnConvRecurrent,
            "gdn_conv_update followed by gdn_recurrent_update, with q, k and v never written: the "
            "operands are the convolution's (`conv_idx` is its state_idx) and then the recurrence's "
            "without those three."),
    SCHEMA("expert_scale", pExpertScale, oExpertScale,
           "Each routing weight times its expert's own per-expert scale, in place: the router's "
           "renormalised top-k weight scaled by the [n_expert] table, an out-of-table id reading "
           "zero."),
    SCHEMA("router_topk", pMoeRouterTopk, oMoeRouterTopk,
           "Softmax over the expert logits, the top_k by value (ties to the lower id), and the "
           "kept weights renormalised to sum to one when `norm`."),
    SCHEMA("moe_scatter", pMoeScatter, oMoeScatter,
           "The routed slots' stable counting sort: expert_ids to sorted_tok, the exclusive "
           "expert_offset prefix, and the expert_count. An id outside the slice is dropped to the "
           "tail; `protect` places its experts last."),

    SCHEMA("moe_gather", pMoeGather, oMoeGather,
            "Fold the expert outputs back to the token: y[t,c] = sum_j expert_w[t,j] * "
            "y_expert[slot(t,j), c], the slots from sorted_tok. An optional shared arm adds "
            "shared[t,c] * act(gate[t]). bf16 in and out."),

    SCHEMA("router_topk_scatter", pRouterTopkScatter, oRouterTopkScatter,
            "The router top-k and the counting sort as one op: the selection over the whole "
            "n_expert logits, then the sort over this rank's n_local slice. expert_w is the "
            "selection output the sort reads."),

    SCHEMA("router_sigmoid_topk", pRouterSigmoid, oRouterSigmoid,
            "The GLM-style router: scores = sigmoid(logits), the selection on scores + bias, the "
            "kept weights the scores scaled by route_scale (renormalised if `norm`). Ties to the "
            "lower id; n_expert <= 512."),

    SCHEMA("moe_gemm_q", pMoeGemmQ, oMoeGemmQ,
            "The grouped quantised GEMM: y[i] = a[sorted_tok[i]/top_k] @ w_e^T with a per-(row, "
            "grid-of-K) scale fold, e the expert of sorted row i. E4M3 operands upcast to f16 in "
            "registers; the even experts are `w` / `w_scale`, the odd `w_odd` / `w_odd_scale`."),

    SCHEMA("moe_gemm", pMoeGemm, oMoeGemm,
            "The grouped GEMM with unquantised experts: y[i] = a[sorted_tok[i]/top_k] @ w_e^T, e "
            "the expert of sorted row i. bf16 a and w, f32 accumulate over the whole of K, bf16 out."),

    SCHEMA("rmsnorm_quant_fp8", pRmsQuantFp8, oRmsQuantFp8,
           "RMS norm with an optional residual add, then the E4M3 quantiser, in one pass. The "
           "residual sum is stored as bf16 before the variance reads it, so the norm is taken over "
           "the rounded sum, as the `add` + `rmsnorm` pair computes it (`rmsnorm_add` normalises "
           "the unrounded sum). `wadd` is added to every "
           "gain in f32. `out_bf16` holds exactly the value the codes were taken over."),

    /* ---- the vocabulary and the sampler: schemas first, rows with their kernels ---- */
    SCHEMA("silu_mul", pMnDtype, oGateUp,
           "y[i] = silu(gate_up[i]) * gate_up[n + i] over one fused [M, 2n] plane, the gate the "
           "first half -- rad-convert's fusion order -- and the product of an activation that was "
           "never narrowed. act_mul is the same arithmetic over two operands, narrowing the "
           "activation first, as a stored activation is."),
    SCHEMA("attn_paged", pAttnPaged, oAttnPaged,
           "Scaled dot-product attention over a paged KV cache: q [tokens, n_head, head_dim], the "
           "cache [blocks, kv_heads, block_size, 2*head_dim], `block_table` a row's block list, "
           "`seqused` its length. `gqa` is the head ratio; `causal` and `window` are the mask; "
           "`scale` is 1/sqrt(head_dim) unless the graph says otherwise. k_descale and v_descale "
           "come with an fp8 cache, cu_seqlens with a packed prefill."),
    SCHEMA("row_topk", pRowTopk, oRowTopk,
           "Exact per-row top-R, descending, ties to the lower column. `vocab_off` shifts every "
           "column reported, so a rank's slice reports global ids; `pairs` is the [M, R, 2] plane an "
           "all_gather moves, with id -1 for a slot the row could not fill."),
    SCHEMA("sample_argmax", pMVocab, oArgmax,
           "The most likely token of each row, the lowest id on a tie."),
    SCHEMA("sample_temp", pMVocab, oTemp,
           "logits / temperature, in place. The temperature is a row parameter, so a batch mixes "
           "greedy and sampled rows."),
    SCHEMA("sample_topk", pTopk, oTopk,
           "The k most likely candidates a row: ids and values, descending, ties to the lower id. "
           "`pairs` is the vocab-parallel form the merge reads after the all-gather."),
    SCHEMA("sample_topp", pMCand, oNarrow,
           "Narrow the candidate set to the smallest prefix whose probabilities reach p, in the "
           "order the candidates arrive -- not in sorted order, which is a different set at a tie."),
    SCHEMA("sample_minp", pMCand, oNarrow,
           "Drop a candidate whose probability is under min_p times the best candidate's."),
    SCHEMA("sample_typical", pMCand, oNarrow,
           "Drop the candidates furthest from the set's entropy: keep those within `sigma` of the "
           "mean surprise, the token set a reader would not have predicted least and most alike."),
    SCHEMA("sample_xtc", pMCand, oNarrow,
           "The exclusion-the-model-would-not-choose filter: drop every candidate above p, keeping "
           "the last one standing when all of them are."),
    SCHEMA("sample_pick", pMCand, oPick,
           "The seeded inverse-CDF draw over the candidate set: the law the request's seed selects, "
           "in the order the candidates arrive. A kernel that draws from the same distribution "
           "differs from the reference on every token, and kbench reads that as a failure, because "
           "for a sampler it is one."),
    SCHEMA("sample_penalties", pPenal, oPenalties,
           "Presence, frequency and repetition penalties in place, over the row's history, with the "
           "DRY and penalty terms libref spells out."),
    SCHEMA("sample_dry", pPenal, oDry,
           "Distributed Repetition Penalty: a sequence of `last_n` that repeats a sequence of "
           "`baseline` is penalised, growing with the length of the repeat. `breakers` marks the "
           "positions that end a sequence and are never penalised."),
    SCHEMA("sample_ngram_ban", pPenal, oNgramBan,
           "Ban the n-grams of length `n` that the history shows inside `window`, to -inf. "
           "`whitelist` exempts a set of ids from the ban."),
    SCHEMA("sample_mask", pMVocab, oMask,
           "The grammar's and the tool-caller's doing: set to -inf every id whose bit is clear in "
           "the row's bitmask, addressed by the row's mask_row, which is -1 for a row with no "
           "constraint and so shares a short plane."),
    SCHEMA("sample_logprobs", pLogprobs, oLogprobs,
           "The row's log-softmax statistics for the response: max, sum of exp, and the k best "
           "(id, logit) pairs, in [M, 2 + 2k]."),
    SCHEMA("logits_gemm", pLogits, oLogits,
           "logits[M, n_vocab] = x[M, n_embd] @ lm_head[n_vocab, n_embd]^T. Separate from gemm_nt "
           "because lm_head is its own access class and the sampler wants to fuse against it: the "
           "head is the largest read in a decode step and the output is f32, which every sampler "
           "stage downstream reads."),
    SCHEMA("gemm_nt_q_gated", pGemmQGated, oGemmQGated,
           "The gated MLP band in one launch: `b` is N = 2n rows, gate over up, the dot products are "
           "narrowed to bf16 as the unfused pair would leave them, the gate is `gt * sigmoid(gt)`, the "
           "product is narrowed, and the row is coded per group of `group` columns with the amax that "
           "group's scale comes from. `out_bf16`, when asked for, is the narrowed gated tensor before "
           "the coding."),
    SCHEMA("dflash_select", pDflashSelect, oDflashSelect,
           "DFlash2's greedy walk over a draft block: at each of `steps` steps the candidate with "
           "the best `unary + pred[pid] . (hp * succ[cand])` is picked, and becomes the predecessor "
           "of the next step. `cand`/`unary` have one row per (sequence, step); `anchor` is the "
           "block's first predecessor, at `m * anchor_stride`."),

    SCHEMA("dequant_w4", pDequantW4, oDequantW4,
           "Packed int4 weight to row-major bf16. Each 32-bit word holds 8 codes; the scale is "
           "f16 in the low 16 bits, the zero point -(1024+zero) as f16 in the high 16 bits."),
    SCHEMA("row_topk_merge", pRowTopkMerge, oRowTopkMerge,
           "Top-R of gathered per-rank candidate sets. Each rank contributes a sorted list; the "
           "merge selects the top-R global candidates by score."),
    SCHEMA("ngram_ids", pNgramIds, oNgramIds,
           "Hashed n-gram row ids for PLE. The FNV-1a mix over the ngram-1 token window is "
            "folded into each head's embedding table; per, W and the ids pitch derive from the "
            "operands, and `eos` cuts the hash window."),
    SCHEMA("gemm_nt_q_bias", pGemmQBias, oGemmQBias,
           "Narrow fp8 GEMM (M <= 16) with per-column bf16 bias fused into the epilogue."),
    SCHEMA("gemm_nt_q_short", pGemmQShort, oGemmQShort,
           "Narrow fp8 GEMM (M <= 16) for short-prompt decode, unsplit K."),
    SCHEMA("moe_gemm_q_mod4", pMoeGemmQMod4, oMoeGemmQMod4,
           "Grouped GEMM over the expert-sorted rows, quantised operands, four classes by "
           "expert number mod 4. The E4M3 wave-matrix core over 8 R3dTab tables (class q = "
           "entry e/4 of tables q); unequal class shapes are refused."),
};

extern "C" int rad_kernel_schema_count(void) {
    return (int)(sizeof(kSchemas) / sizeof(kSchemas[0]));
}
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    if (i < 0 || i >= rad_kernel_schema_count()) return nullptr;
    return &kSchemas[i];
}

/* ================================================================== tuning
 *
 * The axes a kernel is instantiated along, measured per card by `rad-tune` (spec §15). Without a
 * matrix instruction to size a tile against, the rows have few axes: how the work is split across
 * the card and how the weight stream is read.
 */

/* Split-K. 0 is not a point in the space but the kernel's rule, "the largest split legal at this
 * K" (r3d_args.h: libr3d's tunables). */
static const int64_t kSk[] = { 0, 1, 2, 4, 8, 16 };
/* Waves per block. 8 is 256 threads, which is the largest block that keeps the register budget of
 * a wave-per-N-tile kernel inside gfx11's 1024 VGPRs a SIMD at four waves. */
static const int64_t kWv[] = { 1, 2, 4, 8 };
/* Non-temporal weight streaming: a decode GEMM reads a weight once per token, so the L2 serves
 * latency, not reuse. RDNA3 has no transposed-load instruction, so this axis only switches the cache
 * hints on the plain 128-bit stream. */
static const int64_t kNt[] = { 0, 1 };
/* The implementation arm. 0 is the dot2 pair path the row has always run; 1 is the 16x16x16
 * wave matrix (V_WMMA) where the ISA has a tile for the row's dtype. The arm is a point like
 * the rest: the tuner walks it per shape per machine, an untuned install runs the dot2 arm,
 * and a dtype with no tile simply has no legal point 1 (gemm_tune_valid). */
static const int64_t kImpl[] = { 0, 1 };
/* The m64 band's axes: row tiles per block and column tiles a wave. NPW 1 and 4 only: the
 * kernel's activation re-read is 4*MT/NPW, and the instantiations in between cost registers
 * without buying a ratio. */
static const int64_t kMb[]  = { 1, 2, 3, 4 };
static const int64_t kNpw[] = { 1, 4 };
/* The accumulator: 0 = the f32 arm, 1 = the bf16 arm (the 16x16x16 half-precision-accumulate
 * tile; its 8 live elements sit in the even indices of the v16short accumulator, the epilogue
 * reads only those). The axis is the selector; the tuner walks it per shape per machine. */
static const int64_t kAcc[] = { 0, 1 };
/* The kernel-implementation arms of a row that ships several kernels for one contract. Two arms:
 * the table/codec decode and its per-byte ALU twin. Three arms: the grouped E4M3 GEMM's 16x16 tile
 * (LDS table), the same tile with the ALU decode, and the 16x128 block. Four arms: the bf16 tiled
 * row's fdot2 tile (table), the same tile (ALU), the 128x128 wave-matrix tile, and the 64-row
 * wave-matrix kernel. The axis is the selector, as the rest are: an untuned install runs arm 0. */
static const int64_t kImpl2[] = { 0, 1 };
static const int64_t kImpl3[] = { 0, 1, 2 };
static const int64_t kImpl4[] = { 0, 1, 2, 3 };
/* The MoE GEMM's row block: 16 (decode), 32, 64 (a tall prefill). A taller block reads the weight
 * T/mrows times, so the tune picks per shape; 64 is the untuned default (the tall band's best). */
static const int64_t kMrowsMoE[] = { 8, 16 };
static const int64_t kWvMoE[] = { 4, 8 };
static const RadTunable tMoeGemmQW4[] = {
    { "mrows", kMrowsMoE, NELEM(kMrowsMoE), 16, "rows per block: 8 for decode, 16 for prefill" },
    { "wv", kWvMoE, NELEM(kWvMoE), 8, "waves per block: 8 = 128-col tile, 4 = 64-col tile" },
};
static const int64_t kMrows[] = { 16, 32, 64 };
static const RadTunable tMoeGemm[] = {
    { "mrows", kMrows, NELEM(kMrows), 64, "the block's row count; the weight is read T/mrows times" },
};
/* The per-wave register budget: 128 is the 8-wave cap on gfx11 (1024 VGPRs/SIMD), 512 the natural
 * allocation. It is a compile-time attribute, so each legal point is its own compiled kernel; see
 * the kernels' budget comments for the measurements. */
static const int64_t kVgpr[] = { 128, 512 };
static const RadTunable tMoeGemmBf16[] = {
    { "mrows", kMrows, NELEM(kMrows), 64, "the block's row count; the weight is read T/mrows times" },
    { "vgpr",  kVgpr,  NELEM(kVgpr),  512, "the per-wave register budget: 128 is the 8-wave cap, 512 natural" },
    { "acc",   kAcc,   NELEM(kAcc),   0,
      "the WMMA accumulator: 0 = f32, 1 = the half-precision tile (a documented low-precision mode: "
      "the loss grows with k depth, no throughput gain; see the README)" },
};
static const RadTunable tMoeGemmFp8[] = {
    { "vgpr", kVgpr, NELEM(kVgpr), 128, "the per-wave register budget: 128 is the 8-wave cap, 512 natural" },
    { "acc",  kAcc,  NELEM(kAcc),  0,
      "the WMMA accumulator: 0 = f32, 1 = the half-precision tile (a documented low-precision mode: "
      "the loss grows with k depth, no throughput gain; see the README)" },
};
/* Every mrows is legal: the kernel clamps the tail, so the tune is free to pick any point. */
static int moe_tune_valid(const RadParam* p, int n_p, const RadParam* c, int n_c) {
    (void)p; (void)n_p; (void)c; (void)n_c; return 1;
}
/* Only the compiled (mrows, vgpr) points are legal: <32> has both budgets (the capped one measures
 * +38% at T=N=K=4096 NE=8), while <16> and <64> have the natural allocation only -- they measure
 * 7-8% slower under the cap, because their spill traffic outweighs the extra wave. */
static int moe_bf16_tune_valid(const RadParam* p, int n_p, const RadParam* c, int n_c) {
    (void)p; (void)n_p;
    const long long mrows = rad_param_geti(c, n_c, "mrows", 64);
    const long long vgpr = rad_param_geti(c, n_c, "vgpr", 512);
    if (vgpr == 128 && mrows != 32) return 0;
    return 1;
}
/* A single implementation axis: the kernel the row runs. No geometry interaction, so any arm is
 * legal at any point. */
static const RadTunable tImpl2[] = {
    { "impl", kImpl2, NELEM(kImpl2), 0,
      "0 = the table/codec decode, 1 = the per-byte ALU decode" } };
static const RadTunable tImpl4[] = {
    { "impl", kImpl4, NELEM(kImpl4), 0,
      "0 = the 128x128 fdot2 tile (table decode), 1 = the same tile, per-byte ALU decode, 2 = the "
      "128x128 wave-matrix tile, 3 = the 64-row wave-matrix kernel" } };
static int impl_tune_valid(const RadParam* p, int n_p, const RadParam* c, int n_c) {
    (void)p; (void)n_p;
    const long long impl = rad_param_geti(c, n_c, "impl", 0);
    return impl >= 0 && impl <= 3;
}
/* The grouped E4M3 GEMM's three kernels: arm 0 the 16x16 tile with the LDS decode table, arm 1
 * the same tile with the per-byte ALU decode, arm 2 the 16x128 block with the F32 scales folded.
 * The vgpr / acc axes are arm 2's (the block's register budget and accumulator); on arms 0 and 1
 * they are pinned to their defaults so the tuner does not spend time on an axis the arm ignores. */
static const RadTunable tMoeGemmQ[] = {
    { "impl", kImpl3, NELEM(kImpl3), 0,
      "0 = the 16x16 tile, LDS decode table, 1 = the same tile, per-byte ALU decode, 2 = the 16x128 "
      "block, F32 scales folded in f32" },
    { "vgpr", kVgpr, NELEM(kVgpr), 128, "the block's per-wave register budget (arm 2 only)" },
    { "acc",  kAcc,  NELEM(kAcc),  0,
      "the WMMA accumulator: 0 = f32, 1 = the half-precision tile (arm 2 only; a documented "
      "low-precision mode, the loss grows with k depth)" } };
static int moe_q_tune_valid(const RadParam* p, int n_p, const RadParam* c, int n_c) {
    (void)p; (void)n_p;
    const long long impl = rad_param_geti(c, n_c, "impl", 0);
    const long long vgpr = rad_param_geti(c, n_c, "vgpr", 128);
    const long long acc  = rad_param_geti(c, n_c, "acc", 0);
    if (impl < 0 || impl > 2 || acc < 0 || acc > 1) return 0;
    if (impl == 2) return (vgpr == 128 || vgpr == 512);
    return vgpr == 128;
}
/* No `kb` axis: only a GEMM that stages a K slab through LDS has a slab width to tune, and the
 * skinny rows stream the weight through registers. */

static const RadTunable tGemmM16[] = {
    { "sk", kSk, NELEM(kSk), 0, "split-K slices; 0 is the widest legal for this K" },
    { "wv", kWv, NELEM(kWv), 8, "waves per block" },
    { "nt", kNt, NELEM(kNt), 1, "non-temporal weight streaming" },
    { "impl", kImpl, NELEM(kImpl), 0, "0 = dot2 pairs, 1 = the 16x16x16 wave matrix" },
};
/* The m64 band: the wave-matrix kernel's own axes. There is no `impl` -- the band is the
 * wave matrix, and the dot2 row's band ends where this one begins. The block is `wv * 32`
 * threads; the k split is a block split over grid.y, so `sk` costs grid size, not workgroup
 * size, and the validity check bounds the two separately. */
static const RadTunable tGemmM64[] = {
    { "sk", kSk, NELEM(kSk), 1, "split-K slabs of whole 128-wide groups" },
    { "wv", kWv, NELEM(kWv), 8, "waves per block" },
    { "mb", kMb, NELEM(kMb), 1, "row tiles per block" },
    { "npw", kNpw, NELEM(kNpw), 1, "column tiles a wave: 1 or 4" },
    { "nt", kNt, NELEM(kNt), 1, "non-temporal weight streaming" },
    { "acc", kAcc, NELEM(kAcc), 0,
      "accumulator: 0 = f32, 1 = the half-precision tile (a documented low-precision mode: the "
      "loss grows with k depth, no throughput gain; see the README)" },
};
/* No `wv` axis: the quantised kernel's launcher uses a fixed block of eight waves, because its
 * weight decode (sixteen codes a lane into eight dot2) fills a wave's issue budget at any block
 * size. An axis the kernel ignores would only spend tuner time. */
static const RadTunable tGemmQ[] = {
    { "sk", kSk, NELEM(kSk), 0, "split-K slices; 0 is the widest legal for this K" },
    { "nt", kNt, NELEM(kNt), 1, "non-temporal weight streaming" },
    { "impl", kImpl, NELEM(kImpl), 0, "0 = dot2 pairs, 1 = the 16x16x16 wave matrix" },
};

/* Cross-axis legality, in one place so the tuner, the scratch hook and the launch agree. A split's
 * slice must end on a k multiple the kernel's inner loop consumes: 8 elements for a bf16 dot2 pair
 * group, and one whole scale group for a quantised weight, since a slice that ends mid-block cannot
 * apply its block scale. */
static int gemm_tune_valid(const RadParam* p, int n_p, const RadParam* c, int n_c) {
    const long long K = rad_param_getdim(p, n_p, "K", 0);
    const long long sk = rad_param_geti(c, n_c, "sk", 0);
    const long long wv = rad_param_geti(c, n_c, "wv", 8);
    const long long impl = rad_param_geti(c, n_c, "impl", 0);
    if (K <= 0 || wv < 1 || wv > 8 || impl < 0 || impl > 1) return 0;
    /* The wave-matrix arm exists where the ISA has a tile for the row's dtype: the bf16 and f16
     * skinny rows and the w8a8 grid. The other dtypes have no 16x16x16 form on this ISA, and a
     * point the kernel would refuse is a point the tuner should not spend. */
    if (impl == 1) {
        const char* dt = rad_param_gets(p, n_p, "dtype", 0);
        if (!dt || (std::strcmp(dt, "bf16") && std::strcmp(dt, "f16") && std::strcmp(dt, "w8a8")))
            return 0;
    }
    long long unit = rad_param_getdim(p, n_p, "group", 0) > 0
                         ? rad_param_getdim(p, n_p, "group", 0)
                         : 8;
    /* The wave-matrix arm's slab is a 16-k step, wider than the dot2 pair group's 8. */
    if (impl == 1 && unit < 16) unit = 16;
    return sk >= 0 && (sk == 0 || K % (sk * unit) == 0);
}

/* The m64 band's cross-axis legality, in one place so the tuner and the launch agree. The slab
 * is whole 128-wide groups; the split is a block split, so the workgroup is WV*32 threads, and
 * the one shape the kernel's lowering cannot carry -- four column tiles a wave with more than
 * one row tile -- is declined here as well as at the launch. */
static int gemm_tune_valid_m64(const RadParam* p, int n_p, const RadParam* c, int n_c)
{
    const long long K = rad_param_getdim(p, n_p, "K", 0);
    const long long sk = rad_param_geti(c, n_c, "sk", 1);
    const long long wv = rad_param_geti(c, n_c, "wv", 8);
    const long long mb = rad_param_geti(c, n_c, "mb", 1);
    const long long npw = rad_param_geti(c, n_c, "npw", 1);
    const long long nt = rad_param_geti(c, n_c, "nt", 1);
    const long long acc = rad_param_geti(c, n_c, "acc", 0);
    if (K <= 0) return 0;
    if (sk < 1 || sk > 32) return 0;
    if (wv < 1 || wv > 32) return 0;
    if (mb < 1 || mb > 4) return 0;
    if (npw != 1 && npw != 4) return 0;
    if (nt < 0 || nt > 1) return 0;
    if (acc != 0) return 0;
    if (npw == 4 && mb > 1) return 0;
    return K % (sk * 128) == 0;
}

/* The chosen point, read back. `TUNE` falls back to the axis's default when the core appended
 * nothing, so a caller with no tuned key runs what an untuned install runs. */
static long long tune_i(const RadArgs* a, const RadTunable* t, int n, const char* key) {
    if (a && a->p)
        for (int i = 0; i < a->n_p; ++i)
            if (a->p[i].key && !std::strcmp(a->p[i].key, key) && a->p[i].kind == RAD_P_INT)
                return a->p[i].ival;
    for (int i = 0; i < n; ++i)
        if (!std::strcmp(t[i].key, key)) return t[i].deflt;
    return 0;
}
#define TUNE(a, tab, key) ((int)tune_i((a), (tab), NELEM(tab), (key)))

extern "C" void r3d_cfg_gemm(const RadArgs* a, R3dGemmCfg* out) {
    out->wv = TUNE(a, tGemmM16, "wv");
    out->sk = TUNE(a, tGemmM16, "sk");
    out->kb = 0;
    out->nt = TUNE(a, tGemmM16, "nt");
    out->impl = TUNE(a, tGemmM16, "impl");
}
extern "C" void r3d_cfg_moe(const RadArgs* a, R3dMoeCfg* out) {
    out->mrows = TUNE(a, tMoeGemm, "mrows");
    out->vgpr = TUNE(a, tMoeGemmBf16, "vgpr");   /* 512 natural where the row has no vgpr axis */
    out->acc = TUNE(a, tMoeGemmBf16, "acc");     /* 0 where the row has no acc axis */
}
extern "C" void r3d_cfg_moe_q(const RadArgs* a, R3dMoeQCfg* out) {
    out->impl = TUNE(a, tMoeGemmQ, "impl");
    out->vgpr = TUNE(a, tMoeGemmQ, "vgpr");
    out->acc = TUNE(a, tMoeGemmQ, "acc");
}
extern "C" void r3d_cfg_split(const RadArgs* a, R3dSplitCfg* out) {
    out->splits = TUNE(a, tGemmM16, "sk");
}
extern "C" void r3d_cfg_topk(const RadArgs* a, R3dTopkCfg* out) {
    out->rc = 0;
    out->nch = 0;
}

/* Split-KV partials: one record of 128 accumulator floats plus the running max and sum (132
 * floats) per (query row, head, slice). The shim takes the slice count from the bytes the engine
 * offers, halving it until they fit, so this is the request at the slice count the walk benefits
 * from, capped at 16 MiB. Attaching scratch makes the row non-concurrent (rad_kernel_concurrent is
 * `scratch == nullptr`), which is the cost of the split. */
static int64_t attn_scratch(const RadArgs* a) {
    /* Sized from the shape: a scratch hook runs at declare, when the operands' shapes are bound
     * but every tensor's `data` is still null. A hook that required data would answer 0 and the
     * split pass would find no buffer. gemm_scratch sizes from RadParam dims for the same reason. */
    if (!a || a->n_t < 1 || a->t[0].rank == 0) return 0;
    long long n_head = 0, head_dim = 0;
    const int64_t nh = rad_args_geti(a, "n_head", &n_head) ? n_head : 0;
    if (!rad_args_geti(a, "head_dim", &head_dim) || !head_dim || !nh) return 0;
    const long long rows = rad_tensor_numel(&a->t[0]) / (nh * head_dim);
    if (rows <= 0) return 0;
    const long long want = rows * nh * 8 * 132 * (long long)sizeof(float);
    const long long cap = 16LL << 20;
    return want < cap ? want : cap;
}

/* A split's partials are f32 over the output, plus one arrival counter per split. Sized at the
 * band's bounds and the axis's widest split, because a scratch hook runs at declare, where the
 * range may still be open: the buffer must hold any point the tuner can pick. */
static int64_t gemm_scratch(const RadParam* p, int n_p) {
    const long long M = rad_param_getdim(p, n_p, "M", 0), N = rad_param_getdim(p, n_p, "N", 0);
    const long long sk = kSk[NELEM(kSk) - 1];
    if (M <= 0 || N <= 0) return 0;
    return sk * M * N * 4 + sk * 64;
}

/* ================================================================== constraints
 *
 * What the kernel cannot do, and what the selector needs. Divisibility follows the unit's lane map:
 * a wave covering a bf16 row reads four elements a lane (two pairs), so a row of 4m elements needs
 * no tail; where the kernel has a tail path the constraint says 2, not 4.
 */

/* bf16 elementwise, norms and the GEMMs: rows of an even width, so a lane owns a pair. */
static const RadConstraint cBf16Row[]  = { C_IN("dtype", "bf16"), C_DIV("n", 2) };
static const RadConstraint cF32Row[]   = { C_IN("dtype", "f32"), C_DIV("n", 2) };
static const RadConstraint cNormBf16[] = { C_IN("dtype", "bf16"), C_DIV("n", 2),
                                           C_GE("n", 2) };
static const RadConstraint cNormF32[]  = { C_IN("dtype", "f32"), C_DIV("n", 2), C_GE("n", 2) };
static const RadConstraint cSoftmax[]  = { C_IN("dtype", "f32"), C_GE("n", 2) };

/* A cast row is its pair of dtype names. */
static const RadConstraint cCastBf16F32[] = { C_IN("from", "bf16"), C_IN("to", "f32"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastF32Bf16[] = { C_IN("from", "f32"), C_IN("to", "bf16"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastBf16[]    = { C_IN("from", "bf16"), C_IN("to", "bf16"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastF32[]     = { C_IN("from", "f32"), C_IN("to", "f32"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastI64F32[]  = { C_IN("from", "i64"), C_IN("to", "f32"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastI32I64[]  = { C_IN("from", "i32"), C_IN("to", "i64"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastI64I32[]  = { C_IN("from", "i64"), C_IN("to", "i32"),
                                              C_DIV("n", 2) };

/* act_mul names its activation, because a row for gelu and a row for silu are different template
 * instantiations and a model asks for one of them. */
static const RadConstraint cActSilu[]    = { C_IN("act", "silu"), C_IN("dtype", "bf16"),
                                             C_DIV("n", 2) };
static const RadConstraint cActGelu[]    = { C_IN("act", "gelu"), C_IN("dtype", "bf16"),
                                             C_DIV("n", 2) };
static const RadConstraint cActSigmoid[] = { C_IN("act", "sigmoid"), C_IN("dtype", "bf16"),
                                             C_DIV("n", 2) };

/* A fill row is its dtype; the value is a parameter, and every dtype takes every value. */
static const RadConstraint cFillBf16[] = { C_IN("dtype", "bf16") };
static const RadConstraint cFillF32[]  = { C_IN("dtype", "f32") };
static const RadConstraint cDequantW4[] = { C_DIV("N", 16), C_DIV("K", 64) };
static const RadConstraint cRowTopkMerge[] = { C_GE("R", 1), C_LE("R", 32) };
static const RadConstraint cNgramIds[] = { C_GE("ngram", 2), C_LE("ngram", 8) };
static const RadConstraint cGemmQBias[] = { C_IN("dtype", "fp8a8"), C_GE("M", 1), C_LE("M", 16) };
static const RadConstraint cGemmQShort[] = { C_IN("dtype", "fp8a8"), C_GE("M", 1), C_LE("M", 64) };
static const RadConstraint cMoeGemmQMod4[] = { C_GE("n_expert", 4) };
static const RadConstraint cFillI32[]  = { C_IN("dtype", "i32") };
static const RadConstraint cFillI64[]  = { C_IN("dtype", "i64") };

/* rope: the two modes this kernel implements and an even head_dim. The yarn, llama3 and ladder
 * parameters are absent from these rows' bands, so a geometry carrying them is not routed here.
 * `rotary_dim` cannot be constrained: it is optional, a constraint on a key the band does not carry
 * does not hold (see r3d_selftest.cpp's band-reachability phase), and llama-family graphs, which
 * rotate the whole head, omit it. The shim refuses an odd rotary_dim (RAD_E_SHAPE); absent, it is
 * head_dim, which the constraint keeps even. */
static const RadConstraint cRope[] = { C_IN("mode", "neox gptj"), C_DIV("head_dim", 2) };

/* The cache's dtype is the row: a bf16 cache stores the value, an E4M3 cache encodes it with the
 * store's scale. head_dim even because a lane writes a pair; block_size a multiple of 16 so a
 * block's slice of a row is a whole number of lanes. */
static const RadConstraint cKvBf16[] = { C_IN("kv_dtype", "bf16"), C_DIV("head_dim", 2),
                                         C_DIV("block_size", 16) };
static const RadConstraint cKvFp8[]  = { C_IN("kv_dtype", "fp8_e4m3"), C_DIV("head_dim", 2),
                                         C_DIV("block_size", 16) };
static const RadConstraint cKvF16[]  = { C_IN("kv_dtype", "f16"), C_DIV("head_dim", 2),
                                         C_DIV("block_size", 16) };

static const RadConstraint cEmbed[] = { C_IN("dtype", "bf16"), C_DIV("n_embd", 2) };

/* The gather's and scatter's dtype, and a row wide enough for a lane to own a pair. */
static const RadConstraint cGatherBf16[]   = { C_IN("dtype", "bf16"), C_DIV("n", 2) };
static const RadConstraint cGatherF32[]    = { C_IN("dtype", "f32"), C_DIV("n", 2) };
static const RadConstraint cScatterBf16[]  = { C_IN("dtype", "bf16"), C_DIV("n", 2) };

/* The quantisers: the scale group is the checkpoint's block, 128, and a row wide enough that a
 * wave's four-elements-a-lane store is whole. */
static const RadConstraint cQuantFp8[]   = { C_IN("dtype", "bf16"), C_EQ("group", R3D_FP8_BLOCK),
                                             C_DIV("n", 4) };
/* `quant_act_i8` is the per-row int8 whose codes are in the int8 WMMA's step order, so its width
 * is a whole 16-byte step. `quant_act_i8g` is per group with row-major codes, as a dot2 GEMM reads
 * them; it is the activation of this library's i8a8 row. */
static const RadConstraint cQuantI8[]    = { C_IN("dtype", "bf16"), C_EQ("group", R3D_W8_GROUP),
                                             C_DIV("n", 16) };
static const RadConstraint cQuantI8g[]   = { C_IN("dtype", "bf16"), C_EQ("group", R3D_W8_GROUP),
                                             C_DIV("n", 4) };
static const RadConstraint cHadQuantI8[] = { C_IN("dtype", "bf16"), C_GE("group", 128),
                                             C_LE("group", 512), C_DIV("n", 16) };
static const RadConstraint cGeluMulQuantI8[] = { C_IN("dtype", "bf16"), C_DIV("n", 16) };
static const RadConstraint cKvLenTrim[] = { C_GE("max_seqs", 1), C_LE("max_seqs", 64) };
static const RadConstraint cQsaTailStore[] = { C_IN("dtype", "bf16"), C_GE("seqs", 1) };
/* group is the block width: the kernel instantiates 128, 256 and 512 (the shim declines the
 * DIV-legal 384, which no wave form serves). */
static const RadConstraint cGatedHadQuantI8[] = { C_IN("dtype", "bf16"), C_DIV("group", 128),
                                                  C_GE("group", 128), C_LE("group", 512),
                                                  C_DIV("n", 16) };
static const RadConstraint cHadQuantFp8[] = { C_IN("dtype", "bf16"), C_EQ("group", 128),
                                              C_EQ("n", 128) };
static const RadConstraint cGatedHadQuantFp8[] = { C_IN("dtype", "bf16"), C_EQ("group", 128),
                                                   C_EQ("n", 128) };
static const RadConstraint cQuantCols[] = { C_IN("dtype", "bf16"), C_DIV("n", 8) };
static const RadConstraint cHadQuantI8g[] = { C_IN("dtype", "bf16"), C_EQ("group", 128), C_EQ("n", 128) };
static const RadConstraint cGatedHadQuantI8g[] = { C_IN("dtype", "bf16"), C_EQ("group", 128),
                                                   C_EQ("n", 128) };
static const RadConstraint cRmsQuantI8[] = { C_IN("dtype", "bf16"), C_DIV("n", 16) };
/* Not `C_IN("act", "silu sigmoid")`: `act` is optional on the op (a graph may carry libref's
 * integer `mode` instead, and libref defaults to silu), so a band with neither would match no row.
 * The shim reads `act`, then `mode`, then defaults to silu, and refuses gelu, which this unit has no
 * kernel for, with RAD_E_UNSUPPORTED. */
static const RadConstraint cGateQuant[]  = { C_IN("dtype", "bf16"),
                                             C_EQ("group", R3D_FP8_BLOCK), C_DIV("n", 4) };
static const RadConstraint cDequantFp8[] = { C_IN("dtype", "fp8_e4m3"),
                                             C_EQ("group", R3D_FP8_BLOCK), C_DIV("n", 4) };
static const RadConstraint cDequantE5[]  = { C_IN("dtype", "fp8_e5m2"),
                                             C_EQ("group", R3D_FP8_BLOCK), C_DIV("n", 4) };
static const RadConstraint cDequantI8[]  = { C_IN("dtype", "i8"), C_EQ("group", R3D_W8_GROUP),
                                             C_DIV("n", 4) };

/* The GEMMs. The skinny row's band is the decode and verify height, M <= 16: it covers a
 * speculative verify with room to spare and stays below the height where staging the activations
 * through LDS pays. K even because the inner loop consumes pairs. N has no bound: a wave covering N
 * handles the tail, and feed-forward and vocabulary widths (11008, 14336, 128256, a checkpoint's
 * `intermediate_size`) vary. */
static const RadConstraint cGemmM16[] = { C_IN("dtype", "bf16"), C_GE("M", 1), C_LE("M", 16),
                                          C_DIV("K", 2), C_GE("N", 1) };
static const RadConstraint cGemmM16F16[] = { C_IN("dtype", "f16"), C_GE("M", 1), C_LE("M", 16),
                                             C_DIV("K", 2), C_GE("N", 1) };
/* The a16 band: the 16x16x16 f16 tile's M, the group's K, the tile's N. K divisible by 128
 * because the slab is a whole (scale, zero) group; N by 16 because a wave's column tile is a
 * tile. */
static const RadConstraint cGemmM64W4A16A[] = { C_GE("M", 1), C_LE("M", 64), C_DIV("K", 128),
                                                C_DIV("N", 16), C_IN("dtype", "w4a16_asym") };
/* The 5-bit row: the 4-bit row's band, ParoQuant's int5. */
static const RadConstraint cGemmM64W5A16[] = { C_GE("M", 1), C_LE("M", 64), C_DIV("K", 128),
                                               C_DIV("N", 16), C_IN("dtype", "w5a16") };

/* The gated-delta-net preamble: the chunk width and the key head size are compiled in. */
static const RadConstraint cGdnKktC64[] = { C_GE("M", 1), C_EQ("head_k", 128), C_EQ("chunk", 64) };

/* The chunked scan: the tile is compiled in for both head sizes and the chunk width. */
static const RadConstraint cGdnChunkScanC64[] = { C_GE("M", 1), C_EQ("head_k", 128),
                                                  C_EQ("head_v", 128), C_EQ("chunk", 64) };
static const RadConstraint cGdnGatedRmsnorm[] = { C_GE("M", 1), C_EQ("channels", 128) };
static const RadConstraint cGdnGatedNormHq[] = { C_GE("M", 1), C_EQ("head_dim", 128) };
static const RadConstraint cGdnConvW4[] = { C_GE("M", 1), C_EQ("head_k", 128), C_EQ("head_v", 128),
                                           C_EQ("conv_width", 4) };
static const RadConstraint cGdnConvUpdW4[] = { C_GE("q_len", 1), C_EQ("head_k", 128),
                                              C_EQ("head_v", 128), C_EQ("conv_width", 4) };
static const RadConstraint cGdnRecurrent[] = { C_GE("q_len", 1), C_EQ("head_k", 128),
                                              C_EQ("head_v", 128) };
static const RadConstraint cGdnConvRecurrent[] = { C_GE("q_len", 1), C_EQ("head_k", 128),
                                                   C_EQ("head_v", 128), C_EQ("conv_width", 4) };
static const RadConstraint cExpertScaleBf16[] = { C_IN("dtype", "bf16"), C_GE("top_k", 1),
                                                  C_LE("top_k", 64), C_GE("n_expert", 1) };
static const RadConstraint cMoeRouterTopkBf16[] = { C_IN("dtype", "bf16"), C_GE("top_k", 1),
                                                    C_LE("top_k", 32), C_GE("n_expert", 1),
                                                    C_LE("n_expert", 512) };
static const RadConstraint cMoeScatter[] = { C_GE("n_expert", 1), C_LE("n_expert", 512),
                                             C_GE("top_k", 1) };
/* The grouped E4M3 GEMM: the model's `fp8a8f32` experts (E4M3 codes, f32 scales per 128 of K,
 * the kServed table in arch/common/rad_block_moe_fp8.h). The `mxfp4a8` experts (E2M1 weights,
 * E8M0 scales) are a different dtype and a different row. */
static const RadConstraint cMoeGemmQ[] = { C_IN("dtype", "fp8a8f32"), C_GE("n_expert", 1),
                                           C_GE("top_k", 1) };
static const RadConstraint cMoeGemmMxfp4[] = { C_IN("dtype", "mxfp4a8"), C_GE("n_expert", 1),
                                               C_GE("top_k", 1) };
static const RadConstraint cMoeGemmQW4A8[] = { C_IN("dtype", "w4a8"), C_GE("n_expert", 1),
                                               C_GE("top_k", 1) };
static const RadConstraint cMoeGemmW4A4[] = { C_IN("dtype", "w4a4"), C_GE("n_expert", 1),
                                                C_GE("top_k", 1) };
static const RadConstraint cMoeGather[] = { C_GE("M", 1), C_GE("n", 1), C_GE("top_k", 1),
                                             C_IN("dtype", "bf16") };
/* No constraint on `n_local`: it is optional, and a band that leaves it out (every expert local)
 * would not resolve to a row that constrains it; the kernel reads it absent as `n_expert`. */
static const RadConstraint cRouterTopkScatter[] = { C_GE("M", 1), C_GE("n_expert", 1),
                                                    C_GE("top_k", 1), C_LE("n_expert", 512),
                                                    C_IN("dtype", "bf16") };
static const RadConstraint cRouterSigmoid[] = { C_GE("M", 1), C_GE("n_expert", 1),
                                                C_GE("top_k", 1), C_LE("n_expert", 512),
                                                C_IN("dtype", "bf16") };
static const RadConstraint cMoeGemmBf16[] = { C_IN("dtype", "bf16"), C_GE("n_expert", 1),
                                              C_GE("top_k", 1) };
/* The vocabulary projection has no `dtype` constraint because its band carries no `dtype` key: the
 * head's dtype is a property of the weight (libref checks it with a `check_dtypes` hook). A
 * constraint on a key the band lacks does not hold, which would leave only the host kernel and the
 * engine would refuse to start, so the shim refuses an unsupported dtype instead.
 * `n_embd % 32` is the entry's chunk rule, stated so a band outside it is refused at declare rather
 * at launch. `n_vocab >= 4` is the store's granularity (16-byte runs); a narrower band (a
 * shard's padding, a 1-id test vocabulary) is refused here. Above that floor the grid spans the
 * vocabulary at every M, so every vocabulary width runs the same kernel. */
static const RadConstraint cLogitsGemm[] = { C_GE("M", 1), C_GE("n_vocab", 4),
                                             C_DIV("n_embd", 32) };
/* The quantised GEMM above decode heights. Here `dtype` is the operand scheme (gemm_q_shape
 * switches on it), so C_IN filters rows, unlike logits_gemm, where the band's dtype names the
 * activation. M >= 17 complements gemm_nt_q_fp8a8_m16's M <= 16; a higher floor would leave batched
 * decode at 17..64 rows unserved. The tile's k-chunk (32) divides the container's scale block
 * (128), so `group` is an equality and `K` a divisibility. */
static const RadConstraint cGemmQFP8Tiled[] = { C_IN("dtype", "fp8a8 fp8a8f32"),
                                                C_EQ("group", R3D_FP8_BLOCK), C_GE("M", 17),
                                                C_DIV("K", 128) };
/* The OCP-MXFP4 rows: e2m1 weight codes with one e8m0 scale per 32 of K, the activation scaled per
 * 128. Both are fixed by the format, so the `group` key has no kernel-side meaning and is not
 * constrained (the reference derives the planes from it). K % 128 covers the 32-block, the bands
 * split at M = 16 / 17 as for fp8a8, and the dtype names the whole scheme. */
static const RadConstraint cGemmQMxfp4a8M16[]  = { C_IN("dtype", "mxfp4a8"),
                                                   C_GE("M", 1),
                                                   C_LE("M", 16), C_DIV("K", 128), C_GE("N", 1) };
static const RadConstraint cGemmQMxfp4a8Tiled[] = { C_IN("dtype", "mxfp4a8"),
                                                    C_GE("M", 17),
                                                    C_DIV("K", 128) };

/* The symmetric int4 / int8 weight grids at decode heights: the block is the weight's scale group
 * (128) and the band is M <= 16. No row serves the 8-bit grid at prefill heights. */
static const RadConstraint cGemmQW4A8M16[]  = { C_IN("dtype", "w4a8"),
                                                C_GE("M", 1), C_LE("M", 16),
                                                C_DIV("K", 128), C_GE("N", 1) };
static const RadConstraint cGemmQW8A8M16[]  = { C_IN("dtype", "w8a8"),
                                                C_GE("M", 1), C_LE("M", 16),
                                                C_DIV("K", 128), C_GE("N", 1) };

/* The w8a8 wave-matrix arm at prefill heights. 17 <= M <= 127 complements cGemmQW8A8M16's
 * M <= 16, so the two rows never compete for a band. K % 128 is the scale group. */
static const RadConstraint cGemmQW8A8M64[]  = { C_IN("dtype", "w8a8"),
                                                C_GE("M", 17), C_LE("M", 127),
                                                C_DIV("K", 128), C_GE("N", 1) };

/* The tiled bf16 GEMM at prefill heights. M >= 17 complements cGemmM16's M <= 16, so the two rows
 * never compete for a band: the skinny shape (a wave a run of columns, the weight streamed through
 * registers) suits up to 16 rows and the 128x128 tile suits more. K % 32 is the tile's k-chunk and
 * N >= 8 the bf16 store run. */
static const RadConstraint cGemmNTTiled[] = { C_IN("dtype", "bf16"), C_GE("M", 17), C_DIV("K", 32),
                                              C_GE("N", 8) };
/* The quantised GEMM: the scheme names the row, the weight's block is the container's, and a split
 * must end on a block boundary (gemm_tune_valid). `group` is required and every band names it, so
 * the row constrains it. `pb1` and `pb2` are optional, and a constraint on a key the band omits does
 * not hold, so constraining them would send every GEMM band of a container using the default block
 * to the reference's host kernel. The launcher defaults them to `group` (r3d_gemm_fp8a16_nt_m1.hip)
 * and refuses a pb2 that differs from the code group, the container's own invariant. */
static const RadConstraint cGemmQFp8a16[] = { C_IN("dtype", "fp8a16"),
                                              C_EQ("group", R3D_FP8_BLOCK), C_GE("M", 1),
                                              C_LE("M", 16), C_DIV("K", 128), C_GE("N", 1) };
static const RadConstraint cGemmQFp8a8[]  = { C_IN("dtype", "fp8a8"),
                                              C_EQ("group", R3D_FP8_BLOCK), C_GE("M", 1),
                                              C_LE("M", 16), C_DIV("K", 128), C_GE("N", 1) };
/* The int8 GEMM's constraints. No row uses them: a row must name an entry point the library
 * defines, or the plugin fails to load. */
static const RadConstraint cGemmQI8[]     = { C_IN("dtype", "i8a8"),
                                              C_EQ("group", R3D_W8_GROUP), C_GE("M", 1),
                                              C_LE("M", 16), C_DIV("K", 128), C_GE("N", 1) };

/* ================================================================== the rows
 *
 * Declaration order is the tie-break within one priority, so rows that share an op are declared
 * narrowest first.
 */
/* Constraint arrays are declared ahead of the rows that use them, each with a comment naming the
 * row it serves. */

/* ================================================================== the fused-MLP rows
 *
 * Two rows llama graphs ask for that the model unit serves. `silu_mul` constrains only the dtype
 * and an even `n`: `act` is not in this op's parameter list (the op fuses one fixed activation), and
 * a constraint on a key the band lacks does not hold, so the act_mul constraints would match no band.
 * `rmsnorm_quant_fp8` requires the group the kernel covers with a wave to divide `n`; the standalone
 * quantiser's `n % 4` would admit bands this kernel refuses at launch.
 */
static const RadConstraint cSiluMul[]     = { C_IN("dtype", "bf16"), C_DIV("n", 2) };
static const RadConstraint cRmsQuantFp8[] = { C_IN("dtype", "bf16"), C_EQ("group", R3D_FP8_BLOCK),
                                              C_DIV("n", R3D_FP8_BLOCK) };

/* The sampler's pre-filters constrain only `n_vocab`. Their bands carry `n_vocab` and `M` (plus
 * `vocab_off`, and `k` for logprobs) and no `dtype` key, so a dtype constraint would match no band;
 * the shims check that the tensors are f32. `k <= 64` is the depth of the logprobs kernel's pool, so
 * the row serves exactly the bands its launcher accepts.
 */
static const RadConstraint cSampleF32[]      = { C_GE("n_vocab", 1) };
static const RadConstraint cSampleLogprobs[] = { C_GE("n_vocab", 1), C_GE("k", 1), C_LE("k", 64) };

/* The attention rows bind the two dtype keys, the head width the kernel's tile is built for, and
 * the q_len bucket boundary the two kernels are split at. `gqa`, `window` and `causal` are left
 * free: the kernel walks the mask for any gqa the shim accepts and any window, and a constraint on
 * them would only exclude bands it serves. */
static const RadConstraint cAttnBf16Decode[]  = { C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
                                                  C_EQ("head_dim", 128), C_LE("q_len", 16) };
static const RadConstraint cAttnBf16Prefill[] = { C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
                                                  C_EQ("head_dim", 128), C_GE("q_len", 17) };
static const RadConstraint cAttnFp8Decode[]   = { C_IN("q_dtype", "bf16"),
                                                  C_IN("kv_dtype", "fp8_e4m3"),
                                                  C_EQ("head_dim", 128), C_LE("q_len", 16) };
static const RadConstraint cAttnFp8Prefill[]  = { C_IN("q_dtype", "bf16"),
                                                  C_IN("kv_dtype", "fp8_e4m3"),
                                                  C_EQ("head_dim", 128), C_GE("q_len", 17) };

/* `row_topk` is banded on dtype, so it has two rows. The candidate stages are banded on `n_cand`,
 * capped at the kernel's ceiling: five f32 arrays of n_cand in LDS is 20 KiB of the 64 KiB, and the
 * shim refuses a wider band (RAD_E_SHAPE). The vocab stages carry `n_vocab` and no dtype key; the
 * shim checks that the logits are f32.
 */
static const RadConstraint cRowTopkBf16[] = { C_IN("dtype", "bf16"), C_GE("N", 1), C_GE("R", 1) };
static const RadConstraint cRowTopkF32[]  = { C_IN("dtype", "f32"),  C_GE("N", 1), C_GE("R", 1) };
static const RadConstraint cSampleCand[]  = { C_GE("n_cand", 1), C_LE("n_cand", 1024) };

/* The container's 2-bit vocab weights. `group` is both the weight's K block and the scale's block.
 * K must also be a multiple of the launcher's chunk (2048 / M-tile), which the launcher refuses and
 * the row does not express. */
static const RadConstraint cGemmQW2A8[]  = { C_IN("dtype", "w2a8"), C_EQ("group", R3D_FP8_BLOCK),
                                             C_GE("M", 1), C_LE("M", 16), C_DIV("K", 128),
                                             C_GE("N", 1) };

/* The packed-code grids at the heights gemm_nt_q_w2a8_m16 declines. The grid is the container's --
 * one scale and one zero per `group` of codes on a {1, group} plane -- so `group` is the block, and
 * K divides by it because a staged 32-code chunk must sit inside one scale group. */
static const RadConstraint cGemmQW2A8Tiled[] = { C_IN("dtype", "w2a8 w4a8_asym"),
                                                 C_EQ("group", R3D_FP8_BLOCK), C_GE("M", 17),
                                                 C_DIV("K", 128) };

/* `group` is one key for three things: the weight's K block, its row block and the quantiser's
 * group. `act` is C_IN because it is a required string here, not the optional `mode` form. M <= 64
 * is the launcher's ceiling. */
static const RadConstraint cGemmGated[]  = { C_IN("dtype", "fp8a8"),
                                             C_IN("act", "silu sigmoid"),
                                             C_EQ("group", R3D_FP8_BLOCK), C_GE("M", 1),
                                             C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 2),
                                             C_GE("N", 2) };

/* Every key the walk reads is required and in the band, so all are bound; `top_k <= 256` gives each
 * candidate one lane. `anchor_stride` is deliberately unbound: it is optional, defaults to 1, and
 * the kernel serves any value >= 1, so a band that omits it must still resolve. */
static const RadConstraint cDflashSelect[] = { C_IN("dtype", "bf16"), C_GE("M", 1),
                                               C_GE("steps", 1), C_GE("top_k", 1),
                                               C_LE("top_k", 256), C_GE("rank", 1),
                                               C_GE("n_vocab", 1) };

static const RadKernelInfo kKernels[] = {
/* ------------------------------------------------------------------ r3d_model_bf16.hip
 *
 * One workgroup a row. The `shape` column states the geometry the grid is built for; the
 * `computes` column states the arithmetic, in the order the kernel performs it.
 */
{ "rmsnorm_bf16", "rmsnorm", "norm",
  "root-mean-square norm with a gain, two passes over the row",
  "any n; a workgroup a row; the sum of squares accumulates in f32 and only the store narrows",
  "bf16 x and y, bf16 or f32 gain, fp32 reduction",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cNormBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_rmsnorm, nullptr, nullptr, nullptr,
  r3d_shape_rmsnorm },

{ "rmsnorm_f32", "rmsnorm", "norm",
  "root-mean-square norm with a gain, two passes over the row",
  "any n; a workgroup a row",
  "f32 x, y and gain, fp32 reduction",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cNormF32), NOTUNE,
  nullptr, nullptr, r3d_rad_rmsnorm, nullptr, nullptr, nullptr,
  r3d_shape_rmsnorm },

{ "layernorm_bf16", "layernorm", "norm",
  "layer norm with a gain and an optional bias, mean and variance in two passes",
  "any n; a workgroup a row; both passes accumulate in f32",
  "bf16 x, y, gain and bias, fp32 reduction",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cNormBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_layernorm, nullptr, nullptr, nullptr,
  r3d_shape_layernorm },

{ "add_bf16", "add", "elem",
  "a + b, with b's row broadcast and an optional divisor",
  "any n; a workgroup a row; f32 arithmetic, one narrowing at the store",
  "bf16 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_add, nullptr, nullptr, nullptr,
  r3d_shape_add },

{ "add_f32", "add", "elem",
  "a + b, with b's row broadcast and an optional divisor",
  "any n; a workgroup a row",
  "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r3d_rad_add, nullptr, nullptr, nullptr,
  r3d_shape_add },

{ "mul_bf16", "mul", "elem",
  "a * b, with b's row broadcast",
  "any n; a workgroup a row",
  "bf16 in and out, f32 product, one narrowing at the store",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_mul, nullptr, nullptr, nullptr,
  r3d_shape_mul },

{ "mul_f32", "mul", "elem",
  "a * b, with b's row broadcast",
  "any n; a workgroup a row",
  "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r3d_rad_mul, nullptr, nullptr, nullptr,
  r3d_shape_mul },

/* The casts, one row per pair the engine issues. bf16 to f32 and back is the residual stream's
 * width change; i64 to f32 is a position table read by a rotary; the integer pairs move token ids
 * between planes and must not pass through f32, which loses the low bits above 2^24. */
{ "cast_bf16_f32", "cast", "elem", "bf16 to f32, exact", "any n; a wave 128 elements",
  "bf16 in, f32 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastBf16F32), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_f32_bf16", "cast", "elem", "f32 to bf16, one rounding", "any n; a wave 128 elements",
  "f32 in, bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastF32Bf16), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_bf16_bf16", "cast", "elem", "bf16 to bf16, a copy", "any n; a wave 128 elements",
  "bf16 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_f32_f32", "cast", "elem", "f32 to f32, a copy", "any n; a wave 128 elements",
  "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastF32), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_i64_f32", "cast", "elem", "i64 to f32", "any n; a wave 128 elements",
  "i64 in, f32 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastI64F32), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_i32_i64", "cast", "elem", "i32 to i64, exact", "any n; a wave 128 elements",
  "i32 in, i64 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastI32I64), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_i64_i32", "cast", "elem", "i64 to i32, the ids' narrowing", "any n; a wave 128 elements",
  "i64 in, i32 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastI64I32), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "silu_bf16", "silu", "elem", "x * sigmoid(x), the sigmoid in f32", "any n; a wave 128 elements",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_silu, nullptr, nullptr, nullptr, r3d_shape_silu },

{ "silu_f32", "silu", "elem", "x * sigmoid(x)", "any n; a wave 128 elements", "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r3d_rad_silu, nullptr, nullptr, nullptr, r3d_shape_silu },

{ "gelu_bf16", "gelu", "elem",
  "the exact erf GELU: 0.5 * x * (1 + erf(x / sqrt(2)))",
  "any n; a wave 128 elements; erff, the accurate form, not the tanh approximation",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_gelu, nullptr, nullptr, nullptr, r3d_shape_gelu },

{ "sigmoid_bf16", "sigmoid", "elem", "1 / (1 + exp(-x))", "any n; a wave 128 elements",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_sigmoid, nullptr, nullptr, nullptr, r3d_shape_sigmoid },

{ "act_mul_silu_bf16", "act_mul", "elem",
  "silu(gate) * up, the gated feed-forward's middle",
  "any n; a workgroup a row; the gate is operand 0",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cActSilu), NOTUNE,
  nullptr, nullptr, r3d_rad_act_mul, nullptr, nullptr, nullptr, r3d_shape_act_mul },

{ "act_mul_gelu_bf16", "act_mul", "elem",
  "gelu(gate) * up, the exact erf form",
  "any n; a workgroup a row",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cActGelu), NOTUNE,
  nullptr, nullptr, r3d_rad_act_mul, nullptr, nullptr, nullptr, r3d_shape_act_mul },

{ "act_mul_sigmoid_bf16", "act_mul", "elem", "sigmoid(gate) * up", "any n; a workgroup a row",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cActSigmoid), NOTUNE,
  nullptr, nullptr, r3d_rad_act_mul, nullptr, nullptr, nullptr, r3d_shape_act_mul },

{ "softmax_f32", "softmax", "elem",
  "row softmax with the maximum subtracted, the denominator summed in ascending order",
  "any n; a workgroup a row",
  "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSoftmax), NOTUNE,
  nullptr, nullptr, r3d_rad_softmax, nullptr, nullptr, nullptr, r3d_shape_softmax },

{ "scale_bf16", "scale", "elem", "x * s, one scalar over the whole output",
  "any n; a wave 128 elements", "bf16 in and out, fp32 product",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_scale, nullptr, nullptr, nullptr, r3d_shape_scale },

{ "scale_f32", "scale", "elem", "x * s, one scalar over the whole output",
  "any n; a wave 128 elements", "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r3d_rad_scale, nullptr, nullptr, nullptr, r3d_shape_scale },

{ "fill_bf16", "fill", "elem", "the value over [M, n]", "any n; a wave 128 elements",
  "bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cFillBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_fill, nullptr, nullptr, nullptr, r3d_shape_fill },

{ "fill_f32", "fill", "elem", "the value over [M, n]", "any n; a wave 128 elements", "f32 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cFillF32), NOTUNE,
  nullptr, nullptr, r3d_rad_fill, nullptr, nullptr, nullptr, r3d_shape_fill },

{ "fill_i32", "fill", "elem", "the value over [M, n], the value as an integer",
  "any n; a wave 128 elements", "i32 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cFillI32), NOTUNE,
  nullptr, nullptr, r3d_rad_fill, nullptr, nullptr, nullptr, r3d_shape_fill },

{ "fill_i64", "fill", "elem", "the value over [M, n], the value as an integer",
  "any n; a wave 128 elements", "i64 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cFillI64), NOTUNE,
  nullptr, nullptr, r3d_rad_fill, nullptr, nullptr, nullptr, r3d_shape_fill },

{ "rope_bf16", "rope", "attn",
  "rotary positions applied in place to a fused qkv row, neox and gptj pair layouts",
  "head_dim even; rotary_dim optional and even, defaulting to head_dim; a wave a head; the "
  "angles from accurate powf and sincosf, the position divided by `scale`",
  "bf16 qkv in place, i32 positions, fp32 angles",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRope), NOTUNE,
  nullptr, nullptr, r3d_rad_rope, nullptr, nullptr, nullptr, r3d_shape_rope },

{ "kv_store_bf16", "kv_store", "attn",
  "K and V into a paged cache at the slot mapping, bf16 to bf16",
  "any head_dim and block size; a wave a (token, head) pair; a negative slot skipped",
  "bf16 k and v, i32 slot mapping, bf16 cache",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cKvBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_kv_store, nullptr, nullptr, nullptr, r3d_shape_kv_store },

{ "kv_store_f16", "kv_store", "attn",
  "K and V into a paged cache at the slot mapping, bf16 to f16",
  "any head_dim and block size; a wave a (token, head) pair",
  "bf16 k and v, i32 slot mapping, f16 cache",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cKvF16), NOTUNE,
  nullptr, nullptr, r3d_rad_kv_store, nullptr, nullptr, nullptr, r3d_shape_kv_store },

{ "kv_store_fp8", "kv_store", "attn",
  "K and V into a paged cache at the slot mapping, encoding E4M3 with the store's scale",
  "any head_dim and block size; a wave a (token, head) pair; the scale divides before the encode",
  "bf16 k and v, i32 slot mapping, fp8_e4m3 cache",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cKvFp8), NOTUNE,
  nullptr, nullptr, r3d_rad_kv_store, nullptr, nullptr, nullptr, r3d_shape_kv_store },

{ "embed_lookup_bf16", "embed_lookup", "gemm",
  "the codebook row an id names, scaled, with out-of-shard ids yielding a zero row",
  "any n_embd; a workgroup a row of the table",
  "bf16 table and output, i32 ids",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cEmbed), NOTUNE,
  nullptr, nullptr, r3d_rad_embed_lookup, nullptr, nullptr, nullptr, r3d_shape_embed_lookup },

{ "gather_rows_bf16", "gather_rows", "elem", "y[r] = x[idx[r]]",
  "any n; a workgroup a row", "bf16 rows, i32 indices",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGatherBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_gather_rows, nullptr, nullptr, nullptr, r3d_shape_gather_rows },

{ "gather_rows_f32", "gather_rows", "elem", "y[r] = x[idx[r]]",
  "any n; a workgroup a row", "f32 rows, i32 indices",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGatherF32), NOTUNE,
  nullptr, nullptr, r3d_rad_gather_rows, nullptr, nullptr, nullptr, r3d_shape_gather_rows },

{ "scatter_rows_bf16", "scatter_rows", "elem", "x[idx[r]] = v[r]",
  "any n; a workgroup a row; distinct indices, so no two rows write one destination",
  "bf16 rows, i32 indices",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cScatterBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_scatter_rows, nullptr, nullptr, nullptr, r3d_shape_scatter_rows },

/* ------------------------------------------------------------------ r3d_quant_fp8.hip
 *
 * Plain row-major code planes with one f32 scale per (row, group). A dot2 GEMM reads codes in
 * stored order, so they carry no fragment permutation, and a container converted on this card is
 * readable by libref and by `rad-convert` on any card. Only `quant_act_i8` keeps libref's
 * permutation, because its order is part of the op (see the row).
 */
{ "quant_act_fp8_bf16", "quant_act_fp8", "quant",
  "symmetric per-(row, group) E4M3: scale = amax * (1/448), the codes the value times 1/scale",
  "group 128; n a multiple of 4; a wave covers four columns a lane, so a group is a quarter-wipe",
  "bf16 in, fp8_e4m3 codes, f32 scales",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQuantFp8), NOTUNE,
  nullptr, nullptr, r3d_rad_quant_act_fp8, nullptr, nullptr, nullptr,
  r3d_shape_quant_act_fp8 },

/* The per-row int8, in libref's step order. The order belongs to the op: the codes are the WMMA
 * A-fragment's and the oracle compares the plane, so the kernel's store address applies the
 * permutation. libr3d's own int8 GEMM reads the row-major `quant_act_i8g` plane (two rows below);
 * this row exists so an architecture that emits the op resolves to a kernel here rather than to the
 * reference. */
{ "quant_act_i8_bf16", "quant_act_i8", "quant",
  "symmetric int8 with one scale per row: scale = max(amax, 1e-8) / 127, ties to even, saturating",
  "n a multiple of 16, a whole WMMA step; a wave a row of the row sums; the codes stored in the "
  "step order, the middle two dwords of every 16 bytes exchanged",
  "bf16 in, i8 codes, f32 scale per row and the group-major row sums",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQuantI8), NOTUNE,
  nullptr, nullptr, r3d_rad_quant_act_i8, nullptr, nullptr, nullptr,
  r3d_shape_quant_act_i8 },

{ "quant_act_i8g_bf16", "quant_act_i8g", "quant",
  "symmetric per-(row, group) int8: scale = amax / 127, 1 for an all-zero group, ties to even",
  "group 128; n a multiple of 4; a wave covers four columns a lane, so a lane's codes are one "
  "dword and a wave's are 128 contiguous bytes",
  "bf16 in, i8 codes row-major, f32 scales",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQuantI8g), NOTUNE,
  nullptr, nullptr, r3d_rad_quant_act_i8g, nullptr, nullptr, nullptr,
  r3d_shape_quant_act_i8g },

{ "had_quant_act_i8_bf16", "had_quant_act_i8", "quant",
  "a block Hadamard along K, then per-row symmetric int8: the rotation shrinks the row maximum",
  "group 128/256/512, n a multiple of 16; the FWHT runs once per wave (K/group <= 8) so its DPP "
  "exchanges stay out of a loop; the codes in the int8 WMMA step order",
  "bf16 in, i8 codes, f32 scale per row",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cHadQuantI8), NOTUNE,
  nullptr, nullptr, r3d_rad_had_quant_act_i8, nullptr, nullptr, nullptr,
  r3d_shape_had_quant_act_i8 },

{ "gelu_mul_quant_i8_bf16", "gelu_mul_quant_i8", "quant",
  "gelu(gate) * up over a fused [M, 2n] projection, then per-row symmetric int8",
  "the tanh-form product (0.5*g*(1+tanh(k0*(g+k1*g^3)))) rounded to bf16, the absmax, scale and "
  "codes taken over those values; n a multiple of 16, codes in the int8 WMMA step order",
  "bf16 gate_up, i8 codes, f32 scale per row, optional bf16 product",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGeluMulQuantI8), NOTUNE,
  nullptr, nullptr, r3d_rad_gelu_mul_quant_i8, nullptr, nullptr, nullptr,
  r3d_shape_gelu_mul_quant_i8 },

{ "kv_len_trim_i32", "kv_len_trim", "kv",
  "per-sequence context lengths less `trim`, floored at zero: the slice sees its own last row",
  "one workgroup of 64, a thread a sequence; `trim` is the tail operand's extent, its values "
  "unread; max_seqs <= 64",
  "i32 seqused, tail (extent), out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cKvLenTrim), NOTUNE,
  nullptr, nullptr, r3d_rad_kv_len_trim, nullptr, nullptr, nullptr,
  r3d_shape_kv_len_trim },

{ "qsa_tail_store_bf16", "qsa_tail_store", "kv",
  "the QSA indexer's tail into a per-layer linear KV state: position p lands at row p % ring",
  "one block a token; the sequence by a linear cu_seqlens scan; only the last ring tokens of a "
  "chunk write (p and p + ring share a row); ring = ratio + n_spec",
  "bf16 k, i32 pos/cu/sidx, bf16 state [slots,1,ring,n] inout",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQsaTailStore), NOTUNE,
  nullptr, nullptr, r3d_rad_qsa_tail_store, nullptr, nullptr, nullptr,
  r3d_shape_qsa_tail_store },

{ "gated_had_quant_i8_bf16", "gated_had_quant_i8", "quant",
  "a gated product (silu(a)*b or sigmoid(a)*b), a block Hadamard along n, then per-row int8",
  "mode 0 down_proj's input, mode 1 o_proj's input; the rotation shrinks the row maximum; the "
  "FWHT runs once per wave (n == group) so its DPP stages stay out of a loop; codes in step order",
  "bf16 a (gate) and b (up, optional fused), i8 codes, f32 scale per row",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGatedHadQuantI8), NOTUNE,
  nullptr, nullptr, r3d_rad_gated_had_quant_i8, nullptr, nullptr, nullptr,
  r3d_shape_gated_had_quant_i8 },

{ "had_quant_act_fp8_bf16", "had_quant_act_fp8", "quant",
  "a 128-wide block Hadamard, then per-group E4M3: the rotation shrinks the group maximum",
  "n == group == 128 (single-group form, the FWHT runs once per wave so its DPP stages stay out of "
  "a loop); the codes row-major, the scale carrying the 1/sqrt(128)",
  "bf16 in, E4M3 codes, f32 scale per group",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cHadQuantFp8), NOTUNE,
  nullptr, nullptr, r3d_rad_had_quant_act_fp8, nullptr, nullptr, nullptr,
  r3d_shape_had_quant_act_fp8 },

{ "gated_had_quant_fp8_bf16", "gated_had_quant_fp8", "quant",
  "a gated product (silu or sigmoid), a 128-wide block Hadamard, then per-group E4M3",
  "n == group == 128 (single-group form, the FWHT runs once per wave so its DPP stages stay out of "
  "a loop); the gated product in fp32, the codes row-major, the scale carrying the 1/sqrt(128)",
  "bf16 gate_up [M,2n], E4M3 codes, f32 scale per group, optional bf16 product",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGatedHadQuantFp8), NOTUNE,
  nullptr, nullptr, r3d_rad_gated_had_quant_fp8, nullptr, nullptr, nullptr,
  r3d_shape_gated_had_quant_fp8 },

{ "quant_cols_fp8_bf16", "quant_cols_fp8", "quant",
  "E4M3 codes with one scale per column over every row: an attention's V for its fp8 P.V",
  "two passes over the rows split into 32 chunks: the column amax partials, then the scales and "
  "codes; a thread owns eight columns; the software e4m3 codec (RDNA3 has no fp8 builtin)",
  "bf16 in, E4M3 codes, f32 scale per column; kColChunks * n floats of scratch",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQuantCols), NOTUNE,
  nullptr, nullptr, r3d_rad_quant_cols_fp8, r3d_quant_cols_fp8_scratch, nullptr, nullptr,
  r3d_shape_quant_cols_fp8 },

{ "had_quant_act_i8g_bf16", "had_quant_act_i8g", "quant",
  "a 128-wide block Hadamard, then per-group int8: the rotated form for the int8 trunk GEMMs",
  "n == group == 128 (single-group form, the FWHT runs once per wave so its DPP stages stay out of "
  "a loop); the codes row-major for the i8a8 dot2 GEMM, the scale carrying the 1/sqrt(128)",
  "bf16 in, int8 codes (row-major), f32 scale per group",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cHadQuantI8g), NOTUNE,
  nullptr, nullptr, r3d_rad_had_quant_act_i8g, nullptr, nullptr, nullptr,
  r3d_shape_had_quant_act_i8g },

{ "gated_had_quant_i8g_bf16", "gated_had_quant_i8g", "quant",
  "a gated product (silu or sigmoid), a 128-wide block Hadamard, then per-group int8",
  "n == group == 128 (single-group form, the FWHT runs once per wave so its DPP stages stay out of "
  "a loop); the gated product in fp32, the codes row-major for the i8a8 dot2 GEMM, the scale "
  "carrying the 1/sqrt(128)",
  "bf16 gate_up [M,2n], int8 codes (row-major), f32 scale per group, optional bf16 product",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGatedHadQuantI8g), NOTUNE,
  nullptr, nullptr, r3d_rad_gated_had_quant_i8g, nullptr, nullptr, nullptr,
  r3d_shape_gated_had_quant_i8g },

{ "rmsnorm_quant_i8_bf16", "rmsnorm_quant_i8", "quant",
  "the RMS norm and the per-row int8 quantiser in one pass, optionally with the residual junction",
  "post-norm, residual add, scalar, then the norm and the int8 codes in the w4a8 fragment order; "
  "every intermediate rounded to bf16 so the chain is byte-identical; one workgroup a row",
  "bf16 x and optional residual (inout), weight, i8 codes, f32 scale per row, optional bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRmsQuantI8), NOTUNE,
  nullptr, nullptr, r3d_rad_rmsnorm_quant_i8, nullptr, nullptr, nullptr,
  r3d_shape_rmsnorm_quant_i8 },

{ "gated_quant_fp8_bf16", "gated_quant_fp8", "quant",
  "silu(gate) * up over a fused [M, 2n] projection, quantised in the same pass",
  "group 128; n a multiple of 4; the product is rounded to bf16 before its amax, as every fused "
  "implementation rounds it",
  "bf16 gate_up in, fp8_e4m3 codes, f32 scales, optional bf16 product",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGateQuant), NOTUNE,
  nullptr, nullptr, r3d_rad_gated_quant_fp8, nullptr, nullptr, nullptr,
  r3d_shape_gated_quant_fp8 },

{ "dequant_fp8", "dequant", "quant",
  "E4M3 code times its group's f32 scale",
  "group 128; n a multiple of 4",
  "fp8_e4m3 codes and f32 scales in, bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cDequantFp8), NOTUNE,
  nullptr, nullptr, r3d_rad_dequant, nullptr, nullptr, nullptr, r3d_shape_dequant },

{ "dequant_e5m2", "dequant", "quant",
  "E5M2 code times its group's f32 scale",
  "group 128; n a multiple of 4",
  "fp8_e5m2 codes and f32 scales in, bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cDequantE5), NOTUNE,
  nullptr, nullptr, r3d_rad_dequant, nullptr, nullptr, nullptr, r3d_shape_dequant },

{ "dequant_i8", "dequant", "quant",
  "int8 code times its group's f32 scale",
  "group 128; n a multiple of 4",
  "i8 codes and f32 scales in, bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cDequantI8), NOTUNE,
  nullptr, nullptr, r3d_rad_dequant, nullptr, nullptr, nullptr, r3d_shape_dequant },

/* ------------------------------------------------------------------ r3d_gemm_bf16_nt_m16.hip
 *
 * The decode GEMM: M is 1 to 16, so a wave owns a run of N columns and streams the weight's rows
 * past an activation kept in registers. There is no fragment layout, so the kernel reads the
 * checkpoint's rows in order.
 */
{ "gemm_nt_bf16_m16", "gemm_nt", "gemm",
  "a[M,K] @ b[N,K]^T for the decode and verify heights, f32 accumulate, one narrowing at the store",
  "M <= 16; a wave a run of N columns, one output element a lane, k ascending in pairs; split-K "
  "over scratch with the partials reduced in split order; a ragged K runs the scalar tail at sk=1",
  "bf16 a, b and y, fp32 accumulation; the pair is folded into the accumulator by one rounding, so "
  "about 0.3% of a long-K output lands one bf16 store ulp from the reference's, and no output is "
  "further. bf16 reaches the dot as f16, exact over f16's normal range, which every checkpoint's "
  "weights and activations live inside",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_M16,
  ROW(cGemmM16), tGemmM16, NELEM(tGemmM16), gemm_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_bf16_m16,
  [](const RadArgs* a) -> int64_t { return gemm_scratch(a->p, a->n_p); },
  nullptr, nullptr, r3d_shape_gemm_nt },

/* ------------------------------------------------------------------ r3d_gemm_f16_nt_m16.hip
 *
 * The same shape over an f16 checkpoint: the bytes are the dot's native pair, so the dot2 arm
 * stages a straight copy and the weight streams into the fold with no widening between. The
 * wave-matrix arm rides the ISA's 16x16x16 f16 tile, the `impl` point, the way the bf16 row's
 * does on its tile.
 */
{ "gemm_nt_f16_m16", "gemm_nt", "gemm",
  "a[M,K] @ b[N,K]^T for the decode and verify heights, f32 accumulate, one narrowing at the store",
  "M <= 16; a wave a run of N columns, one output element a lane, k ascending in pairs; split-K "
  "over scratch with the partials reduced in split order; a ragged K runs the scalar tail at sk=1",
  "f16 a, b and y, fp32 accumulation; the pair is folded into the accumulator by one rounding, so "
  "about 0.3% of a long-K output lands one f16 store ulp from the reference's, and no output is "
  "further. f16 is the pair the dot takes, so there is no widening to price and the row's range "
  "is f16's: a checkpoint whose values sit outside it is not a checkpoint this row serves",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_M16,
  ROW(cGemmM16F16), tGemmM16, NELEM(tGemmM16), gemm_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_f16_m16,
  [](const RadArgs* a) -> int64_t { return gemm_scratch(a->p, a->n_p); },
  nullptr, nullptr, r3d_shape_gemm_nt },

/* ------------------------------------------------------------------ r3d_gemm_w4a16_nt_m64.hip
 *
 * The 4-bit weight against an f16 activation, at the verify and small-prefill heights: the
 * weight's codes run the 16x16x16 f16 tile through the 1024-offset dequant, and the group
 * scale lands on the f32 accumulator after the zero point's cancellation. The weight arrives
 * in the hook's fragment order, permuted at load.
 */
{ "gemm_nt_q_w4a16asym_m64", "gemm_nt_q", "gemm",
  "a[M,K] @ dequant(wq)[N,K]^T: unsigned 4-bit codes with an f16 scale and an integer zero per "
  "128 of K, against an f16 activation, f32 accumulate, one narrowing at the store",
  "1 <= M <= 64, N a multiple of 16, K a multiple of the group and of the split's slab; the "
  "weight read in the container's stored order, no permutation; split-K slabs reduced by a "
  "second kernel over a grow-only pool",
  "f16 a, U4 wq with the f16 scale and the integer zero per (row, 128 of K), bf16 y, fp32 "
  "accumulation; the dequant is the integer code minus the zero, exact, and the store is the "
  "only narrowing",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmM64W4A16A), tGemmM64, NELEM(tGemmM64), gemm_tune_valid_m64,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w4a16_m64,
  nullptr, r3d_layout_gemm_nt_q_w4a16, r3d_relayout_gemm_nt_q_w4a16,
  nullptr, nullptr },

/* ------------------------------------------------------------------ r3d_gemm_w5a16_nt_m64.hip
 *
 * The 5-bit weight against an f16 activation: the 4-bit row with a fifth bit per code,
 * ParoQuant's int5. The fifth bits ride a plane of their own, one word a (tile, 64 of K,
 * lane), joined behind the (scale, zero) pair.
 */
{ "gemm_nt_q_w5a16_m64", "gemm_nt_q", "gemm",
  "a[M,K] @ dequant(wq, hi)[N,K]^T: 5-bit codes with an f16 scale and an integer zero per "
  "128 of K, against an f16 activation, f32 accumulate, one narrowing at the store",
  "1 <= M <= 64, N a multiple of 16, K a multiple of the group and of the split's slab; the "
  "weight read in the container's stored order, the fifth bits a plane of their own, no "
  "permutation; split-K slabs reduced by a second kernel over a grow-only pool",
  "f16 a, U4 wq with the f16 scale, the integer zero per (row, 128 of K) and the fifth-bit "
  "plane per (tile, 64 of K, lane), bf16 y, fp32 accumulation; the dequant is the integer "
  "code minus the zero, exact, and the store is the only narrowing",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmM64W5A16), tGemmM64, NELEM(tGemmM64), gemm_tune_valid_m64,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w5a16_m64,
  nullptr, r3d_layout_gemm_nt_q_w5a16, r3d_relayout_gemm_nt_q_w5a16,
  nullptr, nullptr },

/* ------------------------------------------------------------------ r3d_gdn_kkt_solve_c64.hip
 *
 * The gated-delta-net chunk preamble: the gram and the blocked inverse in one kernel, the
 * FLA pair's 64x64 fp32 intermediate kept in LDS.
 */
{ "gdn_kkt_solve_c64", "gdn_kkt_solve", "gdn",
  "A = (I + strict_lower(diag(beta) K K^T e^{g_i - g_j}))^-1 per 64-chunk of a gated delta net",
  "head_k 128, chunk 64, varlen; one gram per k head, applied to every v head of the group, the "
  "inverse blocked 16 -> 32 -> 64 on the wave matrix (libhalo's kernel, r3dh_gdn_kkt_solve.hip)",
  "bf16 k and A, f32 beta, g, gram and inverse, i32 cu",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnKktC64), NOTUNE,
  nullptr, nullptr, r3dh_rad_gdn_kkt_solve,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

/* ------------------------------------------------------------------ r3d_gdn_chunk_scan_k128_v128_c64.hip
 *
 * The gated-delta-net chunked scan: the chunk's output and its closing state, the recurrent
 * state in wave-matrix accumulators across the chunk loop.
 */
{ "gdn_chunk_scan_k128_v128_c64", "gdn_chunk_scan", "gdn",
  "O = scale*ge*(Q@S) + sA@V', S = e^{g_last}S + V'^T@(gs*K) per 64-chunk of a gated delta net, "
  "V' = A@W, W = beta*(V - ge*U), U = K@S, sA = the kkt_solve row's output with the "
  "scale*e^{g_i-g_j} score decay",
  "head_k 128, head_v 128, chunk 64, H a multiple of head groups; the state stays in "
  "accumulators across the chunk loop, so only q, k, v, A, g, beta, the state cache and o "
  "touch HBM; two workgroups per v head each carry 64 of the v dimension",
  "bf16 q, k, v, A and o, f32 g and beta, f32 or f16 state cache (the operand's dtype), i32 cu "
  "and the optional state slot index (libhalo's kernel, r3dh_gdn_chunk_scan.hip)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnChunkScanC64), NOTUNE,
  nullptr, nullptr, r3dh_rad_gdn_chunk_scan,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

 /* ------------------------------------------------------------------ r3d_gdn_gated_rmsnorm_h128.hip
  *
  * The gated RMS norm a gated-delta-net layer applies to its own output before the output
  * projection: one wave per (token, head) row, a barrier-free wave sum and one 8-byte access per
  * lane for each load and the store.
  */
{ "gdn_gated_rmsnorm_h128", "gdn_gated_rmsnorm", "gdn",
  "x * rsqrt(mean(x^2) + eps) * w * act(z), one (token, head) row of 128 channels",
  "channels 128; act is silu (the default) or sigmoid; the gate may be a flat projection row or a "
  "column slice of a fused [q|k|v|z] row",
  "bf16 x, z and o, f32 w; one wave per row, four channels per lane",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnGatedRmsnorm), NOTUNE,
  nullptr, nullptr, r3d_rad_gdn_gated_rmsnorm,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

 /* ------------------------------------------------------------------ r3d_gdn_gated_norm_had_quant_i8.hip
  *
  * The GDN block's gated RMS norm, block Hadamard and per-row int8 quantiser in one pass: the
  * out_proj input on the linear-attention layers. One workgroup per row; the rotation is a
  * wave-local FWHT in registers, the int8 scale a whole-row max.
  */
{ "gdn_gated_norm_had_quant_i8", "gdn_gated_norm_had_quant_i8", "gdn",
  "x * rsqrt(mean_head(x^2) + eps) * w * silu(z), Hadamard-rotated over `group`, int8-quantised",
  "head_dim 128, `group` the rotation width (128, 256 or 512); the RMS is per 128-head, the "
  "rotation and the int8 scale per token over the flattened row",
  "bf16 x and z, f32 w, i8 q, f32 scale; the gate may be a column slice of a fused [q|k|v|z] row "
  "(`zgrp`, `zblk`)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnGatedNormHq), NOTUNE,
  nullptr, nullptr, r3d_rad_gdn_gated_norm_hq_i8,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

 /* ------------------------------------------------------------------ r3d_gdn_conv_w4_h128.hip
  *
  * The GDN depthwise causal convolution (width 4, silu) and the work fused around it: the prefill
  * prep (conv + split + l2norm + gate + cumsum) and the decode update (a speculative window over a
  * rolling conv-state buffer).
  */
{ "gdn_conv_prep_w4_h128", "gdn_conv_prep", "gdn",
  "The GDN depthwise causal convolution (width 4, silu) plus split, l2norm, gate and per-chunk "
  "cumsum in one prefill kernel",
  "head_k 128, head_v 128, conv_width 4, chunk 64; the conv state is read at a sequence's first "
  "chunk and rewritten from its last tokens",
  "bf16 x, w, conv_state and q/k/v, f32 A_log, dt_bias, g and beta, a and b_gate bf16 or f32; the "
  "conv output is rounded to bf16 before the norm and the store",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnConvW4), NOTUNE,
  nullptr, nullptr, r3d_rad_gdn_conv_prep,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

{ "gdn_conv_update_w4_h128", "gdn_conv_update", "gdn",
  "The GDN depthwise causal convolution at decode: a speculative window over a rolling conv-state "
  "buffer",
  "head_k 128, head_v 128, conv_width 4; the state buffer is read at the slot the last accepted "
  "token left and rewritten shifted by one",
  "bf16 x, w, conv_state and q/k/v; `state_idx` the rolling buffer's slot, `num_accepted` the "
  "window's read offset",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnConvUpdW4), NOTUNE,
  nullptr, nullptr, r3d_rad_gdn_conv_update,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

{ "expert_scale_bf16", "expert_scale", "moe",
  "Each routing weight times its expert's own per-expert scale, in place",
  "the router's renormalised top-k weight scaled by the [n_expert] table, an out-of-table id "
  "reading zero",
  "bf16 ids (i32), w_e and ew; one thread per element",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cExpertScaleBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_expert_scale_bf16,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

{ "moe_router_topk_bf16", "router_topk", "moe",
  "Softmax over the expert logits, the top_k by value, renormalised if `norm`",
  "one wave per row; top_k passes of a wave minimum (ties to the lower id)",
  "bf16 logits and expert_w; i32 expert_ids; n_expert <= 512, top_k <= 32",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cMoeRouterTopkBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_moe_router_topk_bf16,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

 { "moe_router_topk_scatter", "router_topk_scatter", "moe",
  "The router top-k and the counting sort as one op: the selection over the whole n_expert logits, "
  "then the sort over this rank's n_local slice",
  "phase 1 the top-k (one wave per row), phase 2 the small counting sort (one workgroup); the "
  "selection's expert_w is the sort's input",
  "bf16 logits [M,ne] and expert_w [M,k]; i32 expert_ids [M,k], sorted_tok [M*k], expert_offset "
  "[nl+1], expert_count [nl]; n_expert <= 512, n_local <= n_expert, T <= 1024",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRouterTopkScatter), NOTUNE,
  nullptr, nullptr, r3d_rad_router_topk_scatter,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

 { "moe_router_sigmoid_topk", "router_sigmoid_topk", "moe",
  "The GLM-style router: scores = sigmoid(logits), the selection on scores + bias, the kept "
  "weights the scores scaled by route_scale (renormalised if norm)",
  "one wave per row; a 64-bit selection key (the select monotone bits, the complemented id), the "
  "scores staged in LDS; ties to the lower id",
  "bf16 logits [M,ne], f32 bias [ne]; i32 expert_ids [M,k], bf16 expert_w [M,k]; n_expert <= 512, "
  "top_k <= 32",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRouterSigmoid), NOTUNE,
  nullptr, nullptr, r3d_rad_router_sigmoid_topk,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

{ "moe_scatter_small", "moe_scatter", "moe",
  "The routed slots' stable counting sort",
  "up to 1024 slots one workgroup (histogram in LDS, an exclusive prefix over the expert counts, "
  "each slot placed by its rank inside its expert's run); above, three passes over 256-slot tiles "
  "(histogram, scan across tiles, placement) with the per-tile counts in scratch",
  "i32 expert_ids, sorted_tok, expert_offset, expert_count; n_expert <= 512",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cMoeScatter), NOTUNE,
  nullptr, nullptr, r3d_rad_moe_scatter,
  r3d_rad_moe_scatter_scratch, nullptr, nullptr,
  nullptr, nullptr },

  /* ------------------------------------------------------------------ r3d_moe_gather.hip */
 { "moe_gather", "moe_gather", "moe",
  "Fold the expert outputs back to the token: y[t,c] = sum_j expert_w[t,j] * "
  "y_expert[slot(t,j),c]",
  "one block per token, the slot table (the inverse of sorted_tok, scanned in LDS) and the expert "
  "weights hoisted so the per-column loads pipeline; the optional shared arm adds shared*act(gate)",
  "bf16 y_expert [Te,n], expert_w [Mw,top_k] (bf16 or f32), sorted_tok [T], optional shared/gate, "
  "bf16 y [M,n]; top_k <= 64",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cMoeGather), NOTUNE,
  nullptr, nullptr, r3d_rad_moe_gather,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

 /* ------------------------------------------------------------------ r3d_moe_gemm_q.hip
  *
  * The grouped quantised GEMM: y[i] = a[sorted_tok[i]/top_k] @ w_e^T with a per-(row, grid-of-K)
  * scale fold, e the expert of sorted row i. E4M3 operands staged compact and upcast to f16 in
  * registers (the 256-entry decode table), the f16 wave-matrix, the even experts in `w` and the
  * odd in `w_odd`.
  */
 { "moe_gemm_q_e4m3", "moe_gemm_q", "moe",
  "The grouped quantised GEMM over the model's fp8a8f32 experts: y[i] = a[sorted_tok[i]/top_k] @ "
  "w_e^T with a per-(row, grid-of-K) scale fold, e the expert of sorted row i",
  "three kernels, one contract, the tune's impl axis picks: 0 = 16x16 tiles through LDS, the E4M3 "
  "operands staged compact (8-bit) and upcast to f16 in registers through the block's 512-byte "
  "decode table; 1 = the same tile, the E4M3 decode in registers (the per-byte ALU formula, no LDS "
  "table); 2 = the 16x128 block, the E4M3 operands widened exactly to f16 and the F32 scales "
  "folded in f32 per 128 of K. All arms run the f16 wave-matrix (f32 accumulate) and walk the runs "
  "in the block's 16 rows, so a block spanning several experts (short runs) is correct; the even "
  "experts are entry e/2 of `w` / `w_scale`, the odd of `w_odd` / `w_odd_scale`",
  "E4M3 a and the weight tables, f32 a_scale per (row, 128 of K), f32 w_scale per 128x128 tile; "
  "K % 128; equal classes (N_odd = N, K_odd = K); token or sorted a; y bf16 in sorted order",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cMoeGemmQ), tMoeGemmQ, NELEM(tMoeGemmQ), moe_q_tune_valid,
  nullptr, nullptr, r3d_rad_moe_gemm_q,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },


 /* ------------------------------------------------------------------ r3d_moe_gemm_bf16.hip */
 { "moe_gemm_bf16", "moe_gemm", "moe",
   "The grouped GEMM with unquantised experts: y[i] = a[sorted_tok[i]/top_k] @ w_e^T, e the expert "
   "of sorted row i",
   "16x16 tiles through LDS, the bf16 operands staged and loaded straight into the bf16 wave-matrix "
   "fragments (f32 accumulate over the whole of K); the block walks the runs in its 16 rows, so a "
   "block spanning several experts (short runs) is correct; the weight is a table of the "
   "checkpoint's own [N, K] row-major planes (no layout hook)",
   "bf16 a and w, bf16 y in sorted order; token or sorted a; K a multiple of 64, the N tail "
   "clamped",
   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
   ROW(cMoeGemmBf16), tMoeGemmBf16, NELEM(tMoeGemmBf16), moe_bf16_tune_valid,
   nullptr, nullptr, r3d_rad_moe_gemm_bf16,
   nullptr, nullptr, nullptr,
   nullptr, nullptr },

 /* ------------------------------------------------------------------ r3d_moe_gemm_mxfp4a8.hip */
 { "moe_gemm_mxfp4a8", "moe_gemm_q", "moe",
   "The grouped GEMM with canonical MXFP4 experts: y[i] = a[sorted_tok[i]/top_k] @ w_e^T, e the "
   "expert of sorted row i",
   "16x16 tiles through LDS; the E4M3 activation and the E2M1 weight (decoded exactly to E4M3) "
   "upcast to f16 in registers, the f16 wave-matrix (f32 accumulate); the E8M0 weight scales fold "
   "in f32 per 32 of K; the block walks the runs in its 16 rows; the even experts are entry e/2 "
   "of `w` / `w_scale`, the odd of `w_odd` / `w_odd_scale`",
   "E4M3 a with f32 a_scale per (row, 128 of K); E2M1 (packed, two a byte) weight tables with E8M0 "
   "w_scale per (row, 32 of K); K % 128; equal classes; token or sorted a; y bf16 in sorted order",
   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
   ROW(cMoeGemmMxfp4), NOTUNE,
   nullptr, nullptr, r3d_rad_moe_gemm_mxfp4a8,
   nullptr, nullptr, nullptr,
   nullptr, nullptr },

  /* ------------------------------------------------------------------ r3d_moe_gemm_w4a8.hip
   *
   * The grouped int4 GEMM: the two's-complement int4 weight (nibbles, -8..7) against an int8 or
   * E4M3 activation, one 128-block of K per scale fold. The int8 arm runs the exact i32 wave-matrix
   * (no f16 rounding); the E4M3 arm the f16 wave-matrix (libr4d's form). A row of moe_gemm_q,
   * selected by dtype=w4a8; the even experts in `w` / `w_scale`, the odd in `w_odd` / `w_odd_scale`.
   */
  { "moe_gemm_w4a8", "moe_gemm_q", "moe",
   "The grouped int4 GEMM over the expert-sorted rows: y[i] = a[sorted_tok[i]/top_k] @ w_e^T, e the "
   "expert of sorted row i; the two's-complement int4 weight (nibbles, -8..7) against an int8 or "
   "E4M3 activation, one 128-block of K per scale fold",
   "16 sorted rows per block, eight waves of 16 columns; the block walks its expert runs, the "
   "padding rows reading the run's first; the decode band (M * top_k <= 16 the tuned point, more "
   "runs as extra 16-row blocks, prefill falls back to libavx); K % 128, N % 64; the even experts "
   "are entry e/2 of `w` / `w_scale`, the odd of `w_odd` / `w_odd_scale`; token or sorted a",
   "I4 (packed, two a byte) weight tables with a bf16 w_scale per (row, 128 of K); int8 a (an f32 "
   "a_scale per (row, 128 of K)) or E4M3 a (an f32 a_scale); bf16 y in sorted order; the int8 arm is "
   "bit-exact against the reference, the E4M3 arm to the f16 store ulp",
   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
   ROW(cMoeGemmQW4A8), tMoeGemmQW4, NELEM(tMoeGemmQW4), nullptr,
   nullptr, nullptr, r3d_rad_moe_gemm_w4a8,
   nullptr, nullptr, nullptr,
   nullptr, nullptr },
   /* ------------------------------------------------------------------ r3d_moe_gemm_w4a4.hip
   *
   * The grouped int4xint4 GEMM: the two's-complement int4 weight AND int4 activation, one 128-block
   * of K per scale fold. Both are widened to int8 and run on the exact i32 wave-matrix (no f16
   * rounding). A row of moe_gemm_q, selected by dtype=w4a4; the even experts in `w` / `w_scale`, the
   * odd in `w_odd` / `w_odd_scale`.
   */
   { "moe_gemm_w4a4", "moe_gemm_q", "moe",
    "The grouped int4xint4 GEMM over the expert-sorted rows: y[i] = a[sorted_tok[i]/top_k] @ w_e^T, e "
    "the expert of sorted row i; the two's-complement int4 weight and int4 activation, one 128-block "
    "of K per scale fold",
    "16 sorted rows per block, eight waves of 16 columns; the block walks its expert runs, the "
    "padding rows reading the run's first; the decode band (M * top_k <= 16 the tuned point, more "
    "runs as extra 16-row blocks, prefill falls back to libavx); K % 128, N % 64; the even experts "
    "are entry e/2 of `w` / `w_scale`, the odd of `w_odd` / `w_odd_scale`; token or sorted a",
    "I4 (packed, two a byte) weight tables with a bf16 w_scale per (row, 128 of K); int4 a (packed, "
    "two a byte, an f32 a_scale per (row, 128 of K)); bf16 y in sorted order; bit-exact against the "
    "reference",
    RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
    ROW(cMoeGemmW4A4), tMoeGemmQW4, NELEM(tMoeGemmQW4), nullptr,
    nullptr, nullptr, r3d_rad_moe_gemm_w4a4,
    nullptr, nullptr, nullptr,
    nullptr, nullptr },

    /* ------------------------------------------------------------------ r3d_dequant_w4_bf16.hip
   *
   * The packed 4-bit weight back to a plain row-major bf16 [N, K], for the prefill path where
   * W4A8 gains nothing.
   */
  { "dequant_w4_bf16", "dequant_w4", "quant",
   "The packed 4-bit weight back to a plain row-major bf16 [N, K]",
   "One thread per packed dword scatters its eight nibbles to row-major bf16; the scale dword holds "
   "the f16 scale low and -(1024+zero) high; N % 16, K % 64, group divides K",
   "I4 (packed, two a byte) wq with an f16 ws scale per (row, group); bf16 y [N, K]; twos 1 = "
   "symmetric two's-complement nibble, 0 = asymmetric unsigned nibble minus a per-group zero",
   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
   ROW(cDequantW4), NOTUNE,
   nullptr, nullptr, r3d_rad_dequant_w4,
   nullptr, nullptr, nullptr,
   nullptr, nullptr, nullptr },

  { "rowtopk_merge", "row_topk_merge", "select",
   "The R largest of the all-gathered per-rank candidate sets, one wave per row",
   "One wave per row over world_size * R pairs; tk_insert into a 32-slot top-R heap, tk_round "
   "extracts via a 5-step xor-butterfly; ties break on the lower id; R <= 32",
   "f32 g (M * world_size * R * 2), i32 idx (M * R), f32 val (M * R); R <= 32",
   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
   ROW(cRowTopkMerge), NOTUNE,
   nullptr, nullptr, r3d_rad_row_topk_merge,
   nullptr, nullptr, nullptr,
   nullptr, nullptr, nullptr },

  { "ngram_ids_i32", "ngram_ids", "vocab",
   "The hashed row ids of the Per-Layer Embedding: mixed(k) = XOR of t_s * mult[s], "
   "ids[h] = mixed % vocab_sizes[h] + offsets[h]",
   "One thread a token, looping its heads; n-gram shifts with EOS walk; 64-bit wrapping "
   "multiply and floor-mod; ngram <= 8, heads divisible by (ngram-1)",
   "i32 tok, i32 state, i32 cu, i64 mult, i64 vsz, i64 off, i32 ids; "
   "optional i32 cidx, hini, nacc, ahead",
   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
   ROW(cNgramIds), NOTUNE,
   nullptr, nullptr, r3d_rad_ngram_ids,
   nullptr, nullptr, nullptr,
   nullptr, nullptr, nullptr },

  { "gemm_nt_q_bias_fp8a8_m16", "gemm_nt_q_bias", "gemm",
   "The narrow fp8 GEMM with a per-column bias epilogue: y = a @ b^T + bias",
   "M <= 16, K a multiple of the block; fp8a8 codes and scales, bf16 bias and output; "
   "GEMM then bias-add in two passes",
   "fp8_e4m3 a and b, f32 scales, bf16 bias [N], bf16 y [M,N]",
   RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
   ROW(cGemmQBias), tGemmM16, NELEM(tGemmM16), nullptr,
   nullptr, nullptr, r3d_rad_gemm_nt_q_bias_fp8,
   nullptr, nullptr, nullptr,
   nullptr, nullptr, nullptr },

  { "gemm_nt_q_short_fp8a8_m16", "gemm_nt_q_short", "gemm",
   "The narrow fp8 GEMM with the short-prompt split rule: unsplit at N <= 512, K <= 5120, "
   "41..64 rows",
   "M <= 64, K a multiple of 128, N a multiple of 16; fp8a8 codes and scales, bf16 output; "
   "sk=1 forced (no split)",
   "fp8_e4m3 a and b, f32/bf16 scales, bf16 y [M,N]",
   RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
   ROW(cGemmQShort), tGemmM16, NELEM(tGemmM16), nullptr,
   nullptr, nullptr, r3d_rad_gemm_nt_q_short_fp8,
   nullptr, nullptr, nullptr,
   nullptr, nullptr, nullptr },

  { "moe_gemm_q_mod4_e4m3", "moe_gemm_q_mod4", "moe",
   "The grouped quantised GEMM with 4 expert classes: y[i] = a[sorted_tok[i]/top_k] @ w_e^T, "
   "e%4 picks the class",
   "same as moe_gemm_q_e4m3 but 4-class routing; group = 128; the class shapes default to N/K "
   "and an unequal N_1..K_3 is refused; token or sorted a",
   "E4M3 a and weight tables (4 classes), f32 a_scale and w_scale; y bf16 in sorted order",
   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
   ROW(cMoeGemmQMod4), tMoeGemmQW4, NELEM(tMoeGemmQW4), nullptr,
   nullptr, nullptr, r3d_rad_moe_gemm_q_mod4,
   nullptr, nullptr, nullptr,
   nullptr, nullptr, nullptr },

   /* ------------------------------------------------------------------ r3d_gdn_recurrent_update_k128_v128.hip
  *
  * The gated-delta-net recurrent state update at decode: the per-token state step against a paged
  * fp32 state cache, the state in registers (one thread per v row and half its k range), the
  * gated RMS norm folded into the epilogue.
  */
{ "gdn_recurrent_update_k128_v128", "gdn_recurrent_update", "gdn",
  "The gated-delta-net recurrent state update at decode: S *= e^g, u = beta (v - S k), S += u k^T, "
  "o = S q per token",
  "head_k 128, head_v 128, paged state in either form: per_candidate (one state per candidate "
  "token) or anchor (one committed state and a replay scratch slot a sequence); the gated RMS "
  "norm folded in when z and norm_w are present (libhalo's kernel, r3dh_gdn_recurrent_update.hip)",
  "bf16 q, k, v and o, f32 a/b, A_log and dt_bias; the state cache is f32 or f16 and the operand "
  "says which",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnRecurrent), NOTUNE,
  nullptr, nullptr, r3dh_rad_gdn_recurrent_update,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

{ "gdn_conv_recurrent_update_k128_v128", "gdn_conv_recurrent_update", "gdn",
  "The gated-delta-net decode step as one kernel: the rolling-window conv and the recurrent state "
  "update, with q, k and v never materialised",
  "head_k 128, head_v 128, conv width 4; the speculative window (state_len_max - 2) must fit the "
  "register buffer; the gated RMS norm is folded in (z and norm_w are required); the state cache "
  "is f32 or f16 (libhalo's kernel, r3dh_gdn_recurrent_update.hip)",
  "bf16 x and o, f32 a, b, A_log, dt_bias and norm_w, the conv state bf16, the state f32 or f16",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnConvRecurrent), NOTUNE,
  nullptr, nullptr, r3dh_rad_gdn_conv_recurrent_update,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

/* ------------------------------------------------------------------ r3d_gemm_fp8a16_nt_m1.hip
 *
 * The same shape against a block-scaled weight. The E4M3 codes are widened to f16 exactly and
 * accumulated by dot2; the block scale is applied once per 128-element slab, after the slab's
 * products, where the reference applies it.
 */
{ "gemm_nt_q_fp8a16_m16", "gemm_nt_q", "gemm",
  "a[M,K] @ b[N,K]^T against E4M3 codes with a scale per 128x128 block",
  "M <= 16, K a multiple of the block, group = pb2 = 128 and pb1 the rows sharing a scale row; a "
  "wave a run of N columns; the codes widened to f16 exactly and the block scale applied after each "
  "slab; a split ends on a block boundary (gemm_tune_valid)",
  "bf16 a, fp8_e4m3 b, f32 b_scale, bf16 y, fp32 accumulation; an f32 a, a bf16 scale plane, a "
  "ragged last block and the zero-point planes a_sum and b_ref are refused, not approximated",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQFp8a16), tGemmQ, NELEM(tGemmQ), gemm_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_q_fp8,
  [](const RadArgs* a) -> int64_t { return gemm_scratch(a->p, a->n_p); },
  nullptr, nullptr, r3d_shape_gemm_nt_q },

{ "gemm_nt_q_fp8a8_m16", "gemm_nt_q", "gemm",
  "quant_act_fp8's codes and scales against E4M3 codes with a scale per 128x128 block",
  "M <= 16, K a multiple of the block; both axes coded, so both scales apply",
  "fp8_e4m3 a and b, f32 scales on both, bf16 y, fp32 accumulation",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQFp8a8), tGemmQ, NELEM(tGemmQ), gemm_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_q_fp8,
  [](const RadArgs* a) -> int64_t { return gemm_scratch(a->p, a->n_p); },
  nullptr, nullptr, r3d_shape_gemm_nt_q },

{ "gemm_nt_q_mxfp4a8_m16", "gemm_nt_q", "gemm",
  "fp8 a against OCP-MXFP4 weights: e2m1 codes, two a byte, with an e8m0 scale per 32 of K, the "
  "block's partials scaled and summed in ascending k",
  "M <= 16, K a multiple of 128; one wave a weight row, 64 codes a lane, the tune's impl axis "
  "picks the E4M3 decode (0 = the LDS table, 1 = the per-byte ALU formula), the lane's two 32-blocks "
  "scaled at the block boundary",
  "fp8_e4m3 a with an f32 scale per (row, 128 of K), fp4_e2m1 codes with an e8m0 scale per (row, "
  "32 of K) -- `b_ref`, the same exponent plane, is accepted and ignored, bf16 y, fp32 "
  "accumulation; a reserved e8 (0, 255) is refused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQMxfp4a8M16), tImpl2, NELEM(tImpl2), impl_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_q_mxfp4a8_m16,
  nullptr, nullptr, nullptr, r3d_shape_gemm_nt_q_mxfp4a8 },

{ "gemm_nt_q_w4a8_m16", "gemm_nt_q", "gemm",
  "the symmetric 4-bit weight grid at decode: signed int4 codes against int8 activations, the "
  "group's integer sub-sum scaled once by the activation's row scale and the weight's group scale",
  "M <= 16, K a multiple of 128; one wave a weight row, 64 codes a lane from two 16-byte loads, "
  "the pair fold at the group boundary, the group terms scaled in the epilogue",
  "I8 a in the step order with an f32 scale per row, I4 codes b (two a byte) with an f16 scale "
  "per (row, 128 of K), bf16 y, fp32 accumulation; the sub-sums are exact integers under 2^24, "
  "and the store is the only narrowing; no zero point, so no a_sum",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQW4A8M16), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w4a8_m16,
  nullptr, nullptr, nullptr, r3d_shape_gemm_nt_q_w4a8 },

{ "gemm_nt_q_w8a8_m16", "gemm_nt_q", "gemm",
  "the symmetric 8-bit weight grid at decode: signed int8 codes against int8 activations, the "
  "group's integer sub-sum scaled once by the activation's row scale and the weight's group scale",
  "M <= 16, K a multiple of 128; one wave a weight row, 64 codes a lane from four 16-byte loads, "
  "the pair fold at the group boundary, the group terms scaled in the epilogue",
  "I8 a in the step order with an f32 scale per row, I8 codes b with an f16 scale per (row, 128 "
  "of K), bf16 y, fp32 accumulation; the sub-sums are exact integers under 2^24, and the store is "
  "the only narrowing; no zero point, so no a_sum; M > 16 is declared out. The tune's impl axis "
  "picks the arm: 0 = the dot2 pair path, 1 = the 16x16x16 wave matrix",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQW8A8M16), tImpl2, NELEM(tImpl2), impl_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w8a8_m16,
  nullptr, nullptr, nullptr, r3d_shape_gemm_nt_q_w8a8 },

 { "gemm_nt_q_w8a8_m64", "gemm_nt_q", "gemm",
   "the symmetric 8-bit weight grid at prefill heights: signed int8 codes against int8 "
   "activations, the group's integer sub-sum scaled once by the activation's row scale and the "
   "weight's group scale",
   "17 <= M <= 127, K a multiple of 128; two 16-row tiles a wave, the weight column loaded once "
   "per 16-k step and reused over the tiles, the group terms scaled in the epilogue",
   "I8 a in the step order with an f32 scale per row, I8 codes b with an f16 scale per (row, 128 "
   "of K), bf16 y, fp32 accumulation; the sub-sums are exact integers under 2^24, and the store is "
   "the only narrowing; no zero point, so no a_sum",
   RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
   ROW(cGemmQW8A8M64), NOTUNE,
   nullptr, nullptr, r3d_rad_gemm_nt_q_w8a8_m64,
   nullptr, nullptr, nullptr, r3d_shape_gemm_nt_q_w8a8 },

    { "silu_mul_bf16", "silu_mul", "elem",
      "silu(gate) * up over one fused gate_up plane, the gate the first half",
      "any n; a workgroup a row; the activation is not narrowed before the product, as ref's is not "
      "(act_mul, over two operands, narrows it, as a stored activation is)",
      "bf16 in and out, fp32 arithmetic",
      RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
      ROW(cSiluMul), NOTUNE,
      nullptr, nullptr, r3d_rad_silu_mul, nullptr, nullptr, nullptr, r3d_shape_silu_mul },

    { "rmsnorm_quant_fp8_bf16", "rmsnorm_quant_fp8", "quant",
      "residual add, RMS norm and the E4M3 quantiser in one pass: a 5120-element bf16 plane a token "
      "is written once and read once inside one kernel, where the pair writes it and reads it back",
      "group 128 and a multiple of 128; a workgroup a row; the residual sum is stored before the "
      "variance reads it, so the norm is over the bf16 sum, and the codes are taken over the bf16 "
      "normalised value, which out_bf16 receives",
      "bf16 x, residual, gain and optional out_bf16, fp8_e4m3 codes, f32 scales, fp32 arithmetic",
      RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
      ROW(cRmsQuantFp8), NOTUNE,
      nullptr, nullptr, r3d_rad_rmsnorm_quant_fp8, nullptr, nullptr, nullptr,
      r3d_shape_rmsnorm_quant_fp8 },


/* ------------------------------------------------------------------ r3d_sample_pre_f32.hip
 *
 * The sampler's pre-filters: one workgroup a row over the logits plane. The history stages are
 * sparse (a row's history is a few hundred tokens against a vocabulary of about 130K), so walking
 * the window rather than the row, the reference's shape, is also the fast one.
 */
{ "sample_penalties_f32", "sample_penalties", "sample",
  "presence, frequency and repetition penalties from the row's history, in place",
  "a workgroup a row; the count is taken from the token's FIRST occurrence forward, and the "
  "repetition penalty is sign-dependent -- v > 0 divides, v < 0 multiplies -- as the reference has "
  "it; a penalty of zero writes nothing",
  "f32 logits, f32 params, i32 history; writes only inside the row's own n_vocab columns",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_penalties, nullptr, nullptr, nullptr,
  r3d_shape_sample_penalties },

{ "sample_dry_f32", "sample_dry", "sample",
  "the DRY penalty: a suffix of the history that repeats after a token penalises it, by its length",
  "a workgroup a row; the longest repeat per token, capped at dry_allowed_length; the multiplier "
  "grows by repeated multiply and stops at FLT_MAX/base, which is where an exponent would overflow; "
  "one penalty per token, from its longest repeat, and the FIRST occurrence is the one written",
  "f32 logits, f32 params, i32 history, u8 breakers; a breaker position ends a sequence and is "
  "never penalised; a history no longer than dry_allowed_length penalises nothing",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_dry, nullptr, nullptr, nullptr,
  r3d_shape_sample_dry },

{ "sample_ngram_ban_f32", "sample_ngram_ban", "sample",
  "the n-grams the history shows inside the window are banned to -inf",
  "a workgroup a row; the prefix compare over n-1 tokens, the ban landing on s[idx+n-1]; the "
  "whitelist is removed from the banned set BEFORE the ban, so an exempt id survives; the window is "
  "the row's own ngram pair's second word (0 clamps to the whole staged span), and n = 0 bans "
  "nothing",
  "f32 logits, f32 params, i32 history, i32 n-gram pairs, an optional i32 whitelist plane",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_ngram_ban, nullptr, nullptr, nullptr,
  r3d_shape_sample_ngram_ban },

{ "sample_mask_f32", "sample_mask", "sample",
  "the grammar's and the tool-caller's doing: every id whose bit is clear becomes -inf",
  "a workgroup a row over 4080 words at this vocabulary; a word of all ones is skipped and the last "
  "partial word honours n_vocab; a NEGATIVE mask_row skips the row, which is how an unconstrained "
  "request ends up with no mask applied -- the descriptor calls the same thing "
  "a shared row, and the body is the spec; the bound is the bitmask plane's own row count, not M, "
  "so a row addressing the plane's second half is masked as the reference masks it",
  "f32 logits, f32 params, i32 bitmask; a plane with fewer rows than the logits is refused "
  "(RAD_E_SHAPE), the case the geometry never produces",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_mask, nullptr, nullptr, nullptr,
  r3d_shape_sample_mask },

{ "sample_logprobs_f32", "sample_logprobs", "sample",
  "the row's summary for the response: max, the sum of exp, and the k best (id, logit) pairs",
  "one read of the logits plane where the reference reads it three (max, sum, k-list); the sum is "
  "SUM exp(v - max) in DOUBLE, the reference's law, and a row's k-list is one insertion list in "
  "registers, not an upper_bound per element; stats is [M][2 + 2k], and rows the flag leaves alone "
  "are not written (no params at all: every row runs)",
  "f32 logits, f32 params, f32 stats; k <= 64 is the pool's depth and above it the shim refuses "
  "(RAD_E_UNSUPPORTED); on a row that contains a NaN the reference's list freezes at the first k it "
  "saw: once a NaN sits in the k-list's tail, every later real value fails the tail's compare "
  "(anything against NaN is false) and is never inserted -- this kernel reports the row's true "
  "k-best, a divergence a plane has to be fed NaNs to reach",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleLogprobs), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_logprobs, nullptr, nullptr, nullptr,
  r3d_shape_sample_logprobs },


/* ------------------------------------------------------------------ r3d_attn_paged_h128.hip
 *
 * Paged attention, head_dim 128, banded on q_len. The two kernels share the block walk and differ
 * in what a workgroup is: decode is one head group over 16 query rows with split-KV, prefill a
 * 16-token query tile. The walk's block bounds are per workgroup: a tile is staged once into LDS
 * and every wave consumes it, so a wave must not narrow the walk to its own query positions, or the
 * waves would stage different blocks into the same LDS.
 */
{ "attn_paged_bf16_decode", "attn_paged", "attn",
  "paged attention over a bf16 cache: one head group a workgroup, 16-token tiles, split-KV",
  "head_dim 128; q_len <= 16; two passes over the blocks -- scores are recomputed in the second so "
  "the softmax sum accumulates ascending like the reference, and a masked key is never exponentiated; "
  "the block bounds come from the workgroup's 16 rows, because the staged tile is shared",
  "bf16 q and out, bf16 cache, i32 block_table and seqused; a descale on a bf16 cache is refused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_DECODE,
  ROW(cAttnBf16Decode), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_decode_h128, attn_scratch, nullptr, nullptr,
  r3d_shape_attn_paged },

{ "attn_paged_bf16_prefill", "attn_paged", "attn",
  "the same walk over a 16-token query tile, for a chunk that is wider than the decode bucket",
  "head_dim 128; q_len >= 17; a workgroup is 4 waves over 16 query rows and one KV head; the tile's "
  "idle lanes clamp to the last query row instead of reading past the q plane",
  "bf16 q and out, bf16 cache, i32 block_table and seqused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_PREFILL,
  ROW(cAttnBf16Prefill), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_prefill_h128, nullptr, nullptr, nullptr,   /* no split: no scratch */
  r3d_shape_attn_paged },

{ "attn_paged_fp8_decode", "attn_paged", "attn",
  "paged attention over an E4M3 cache, the descale riding the widening into the tile",
  "head_dim 128; q_len <= 16; a 16-code chunk is the aligned 16-byte unit -- eight codes are 8 bytes "
  "and a 16-byte load at a slot's odd half returns garbage; the descale "
  "multiplies into the f16 tile, exact for a power-of-two scale inside f16's range",
  "bf16 q and out, fp8_e4m3 cache, i32 block_table and seqused, f32 k_descale and v_descale "
  "optional (absent is scale 1.0)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_DECODE,
  ROW(cAttnFp8Decode), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_decode_h128, attn_scratch, nullptr, nullptr,
  r3d_shape_attn_paged },

{ "attn_paged_fp8_prefill", "attn_paged", "attn",
  "the E4M3 cache over a 16-token query tile",
  "head_dim 128; q_len >= 17; the same staged walk as the bf16 prefill, with the codes widened as "
  "they land in the tile",
  "bf16 q and out, fp8_e4m3 cache, i32 block_table and seqused, f32 descales optional (absent is "
  "scale 1.0)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_PREFILL,
  ROW(cAttnFp8Prefill), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_prefill_h128, nullptr, nullptr, nullptr,   /* no split: no scratch */
  r3d_shape_attn_paged },


/* ------------------------------------------------------------------ r3d_sample_f32.hip
 *
 * The sampler's spine: the row's top-R and the stages that turn its candidate plane into one token.
 * The candidate set is the contract between them: `cand_val` carries logits, so each narrowing
 * stage takes its own softmax and the stages compose in any order; a dropped candidate is
 * `cand_idx` -1 and `cand_val` -inf; the set is descending by value, then ascending by id; and every
 * stage leaves a greedy row byte-identical, because a greedy row's token is `sample_argmax`'s.
 * `topk_next` is a sort by (-value, +column) written as a loop (`v == *pv && i <= *pi` steps through
 * a run of equal values), so one pass keeping the best R returns the reference's R pairs, in its
 * order and with its tie-breaks to the lower column.
 */
{ "row_topk_bf16", "row_topk", "moe",
  "the row's top R, with the reference's tie-breaks, in one scan of the row",
  "one pass keeps the best R in the reference's order, where the reference scans the row R times; "
  "R beyond a pass's pool rescans with the chain's state carried, so the answer is the reference's "
  "at any R and only the time grows; a slot a short row cannot fill is idx -1 and val -inf; a row "
  "that opens with a NaN reports (0, NaN) R times, because the reference's seed takes the first "
  "element it sees and nothing displaces a NaN",
  "bf16 x, i32 idx [M,R], f32 val [M,R], an optional f32 pairs [M,R,2] (the id's bits and the value)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRowTopkBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_row_topk, nullptr, nullptr, nullptr, r3d_shape_row_topk },

{ "row_topk_f32", "row_topk", "moe",
  "the row's top R over an f32 plane, in one scan",
  "the same one-pass insertion as the bf16 row; the band's router row costs the reference 7.0 GB of "
  "scans and this kernel 468 MB, and at the band it is the sifts that cost, not the bytes: the f32 "
  "plane moves half again the traffic for the same 23 ms",
  "f32 x, i32 idx, f32 val, optional f32 pairs",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRowTopkF32), NOTUNE,
  nullptr, nullptr, r3d_rad_row_topk, nullptr, nullptr, nullptr, r3d_shape_row_topk },

{ "sample_argmax_f32", "sample_argmax", "sample",
  "the greedy token: the row's argmax, ties to the lower id",
  "a workgroup a row; the max is a strict `>`, so a NaN is never adopted -- the reference's scan "
  "keeps the first element it sees, which is why a row opening with a NaN yields token 0, and a row "
  "of all -inf yields 0; the stage that reads no scalar at all, so a missing params operand is not "
  "a refusal: the reference describes the operand and does not read it",
  "f32 logits, i32 token [M]; the params plane is described, not read",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_argmax, nullptr, nullptr, nullptr,
  r3d_shape_sample_argmax },

{ "sample_temp_f32", "sample_temp", "sample",
  "temperature, in place: every logit multiplied by the reciprocal of temp",
  "a workgroup a row; a row is skipped when temp is not positive or is exactly 1; the reciprocal is "
  "formed ONCE, as the reference forms it, not a divide per element -- a divide per element is a "
  "different f32, and the argmax after it is only bit-exact under the transcription",
  "f32 logits INOUT, f32 params; the most expensive row in the slice at 11 ms a step, and the cost "
  "is the plane: 1 GiB in and 1 GiB out for one multiply an element",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_temp, nullptr, nullptr, nullptr,
  r3d_shape_sample_temp },

{ "sample_topk_f32", "sample_topk", "sample",
  "the candidate set the rest of the sampler draws from, written whole every step",
  "the set's width is the BUFFER's, not the row's: top_k 40 in a 1024-wide plane is 40 candidates "
  "and 984 holes, and a stale candidate left in a hole is a token the pick can draw; the row's "
  "top_k only narrows inside the width, and `<= 0` or a width above it takes the whole width, as "
  "the reference clamps; cand_val carries the LOGIT, which is what lets the narrowing stages "
  "compose in any order; a greedy row takes none",
  "f32 logits, f32 params, i32 cand_idx [M,n_cand] (local columns), f32 cand_val, an optional i32 "
  "pairs [M,2*n_cand] of (global id, logit bits) with 0xffffffff for a hole; n_cand <= 1024",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_topk, nullptr, nullptr, nullptr,
  r3d_shape_sample_topk },

{ "sample_pick_f32", "sample_pick", "sample",
  "the draw: one uniform over the candidate set's own softmax, in index order",
  "splitmix64 over the seed and the position, the top 24 bits over 2^24, keyed by the pick's own "
  "key; the exps are computed by the wave in parallel and the CDF accumulates on one lane in "
  "candidate index order, because the order is the law -- parallel values, serial sums; an empty "
  "set writes -1; a set whose mass falls below u falls back to the last candidate above -infinity, "
  "which is why a masked tail is never the fallback",
  "i32 cand_idx, f32 cand_val, f32 params (seed, pos), i32 token [M]; only token is written",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_pick, nullptr, nullptr, nullptr,
  r3d_shape_sample_pick },

{ "sample_topp_f32", "sample_topp", "sample",
  "nucleus: the shortest prefix whose cumulative probability reaches top_p",
  "the stage takes its own softmax over the set (max by strict `>` over live candidates, then exp, "
  "then the predicate), because the set carries logits; at least one candidate survives; it is a "
  "PREFIX rule in the set's order, which is why the set's order is contract and why no stage sorts "
  "it",
  "i32 cand_idx INOUT, f32 cand_val INOUT, f32 params; off when top_p is out of range; a greedy row "
  "and an all-holes row are left alone",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_topp, nullptr, nullptr, nullptr,
  r3d_shape_sample_topp },

{ "sample_minp_f32", "sample_minp", "sample",
  "min-p: drop what is below p times the best candidate's probability",
  "the stage's own softmax again; the best always survives; a relative cut under the best, not a "
  "prefix, so it composes with top-p in either order",
  "i32 cand_idx INOUT, f32 cand_val INOUT, f32 params; off when min_p is out of range",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_minp, nullptr, nullptr, nullptr,
  r3d_shape_sample_minp },

{ "sample_typical_f32", "sample_typical", "sample",
  "locally typical: drop the candidates whose surprisal is furthest from the set's entropy",
  "the set's entropy from the same exps, the kept mass overshooting p by design and not depending "
  "on the order, ties kept -- a stage that sorted the set would genuinely disagree with the schema",
  "i32 cand_idx INOUT, f32 cand_val INOUT, f32 params; off when typical_p is out of range",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_typical, nullptr, nullptr, nullptr,
  r3d_shape_sample_typical },

{ "sample_xtc_f32", "sample_xtc", "sample",
  "the exclusion of top candidates: a coin, then drop all but the least likely of the clearers",
  "its own coin (the seed keyed by XTC) decides whether the stage fires at all; when it fires it "
  "drops every candidate at or above the threshold EXCEPT the least likely of them, and does nothing "
  "when fewer than two clear it -- a stage that dropped unconditionally would fire on 100% of steps",
  "i32 cand_idx INOUT, f32 cand_val INOUT, f32 params; off when the probability or the threshold is "
  "out of range",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_xtc, nullptr, nullptr, nullptr,
  r3d_shape_sample_xtc },

/* ------------------------------------------------------------------ r3d_dflash_f32.hip
 *
 * The speculative decoder's accept step. It is a gather, not arithmetic: two 63.8 MiB codebooks and
 * `M * steps * rank` gathered elements, so each score's rank sum is one lane's serial chain. There
 * is no work to hide the gather behind, and a tree reduction would change the sum for no gain.
 */
/* ------------------------------------------------------------------ r3d_gemm_nt_q_w2a8.hip
 *
 * `output.weight`: U2 codes, int8 activations, one f32 scale per activation row (ZP_I8_ROW), and a
 * joined `b_scale` stored as two planes (scale f16, zero u8) behind one operand, which only a layout
 * hook can present. libref's shape hook declines the dtype for that reason, so kbench has no case
 * for the band: the row's opd_shape is null (the tool's named skip), and the joined plane is built
 * with rad_planes_relayout, the same builder the driver uses.
 *
 * Exactness: `v_fdot2_f32_f16` carries about 22 bits of mantissa on integer operands, not 24, so an
 * integer dot through it is inexact above 2^20. The error is about 2^-22 of the magnitude, so a
 * smaller bias shrinks the tail; the bias is 128 (sampled worst |err| 0.015625, against 0.125 at
 * bias 1024). 0.1-0.2% of outputs land one bf16 ulp off the reference (rel_l2 5.0e-3, max_abs
 * 9.8e-4, no non-finite values). This row is not bit-exact; the unit's dot probe states the bound.
 *
 * M <= 16 is decode.
 */
{ "gemm_nt_q_w2a8_m16", "gemm_nt_q", "gemm",
  "the vocab projection at decode: 2-bit codes, int8 activations, the zero removed with a_sum",
  "a wave takes one weight row; lane l owns group l&15 and half of it, 64 codes to a 16-byte load, "
  "the pair (l, l^16) folding the group in one shuffle; lanes 0..15 hold the per-group terms and "
  "lane 0 sums the groups in ascending order in one accumulator, which is the reference's order. "
  "The codes widen to f16 through a bias of 128, so the dot's integer answer comes back with "
  "B*sum(a) on it and a_sum -- group-major, the reference's own 16-row pad -- removes it",
  "U2 codes b, I8 activations a, f32 a_scale (one per row), the joined b_scale (f16 scale + u8 zero "
  "as rad_planes stores them), f32 a_sum, bf16 y. fp8a8, fp8a16 and w4a8 take their own rows; a zero "
  "plane stored sub-byte (U2/U4/U5) is refused, not approximated -- the bit-extract path is 10 lines "
  "of rad_load_bits and is not written; M > 16 is refused, prefill's vocab is logits_gemm",
  RAD_DOMAIN_DEVICE, R3D_PRIO_VOCAB_M1,
  ROW(cGemmQW2A8), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w2a8, nullptr,
  r3d_layout_gemm_nt_q_w2a8, r3d_relayout_gemm_nt_q_w2a8, nullptr },

{ "gemm_nt_q_w2a8_tiled", "gemm_nt_q", "gemm",
  "the packed-code grids at the heights the skinny row declines: 2-bit and 4-bit codes, int8 "
  "activations, the batched decode and the draft head of a 2-bit vocab projection",
  "the same 128x128 tile as the fp8a8 arm, with the codes unpacked on the way into LDS -- 8 codes "
  "from a 4-byte load at U2, 16 from a 16-byte load at U4 -- and the epilogue folding "
  "(code - zero) * scale per staged element, so the dot runs over values and no a_sum term is "
  "reduced: the zero comes off term by term, which is why Σ a(q-z) needs no Σ a here while the m16 "
  "row needs it to lift the integer bias off its f16-as-int dot. The row's scale and zero are one "
  "per ROW on a {1, group} plane, so the staging reads 128 scalars a chunk where the fp8a8 arm reads "
  "one, at the row the grid puts it; issued one chunk ahead of the fold that consumes them, because "
  "a scalar load with nothing to overlap it dominates the band otherwise (the output bits are "
  "the same either way). r3d_tiling.h's swizzle is shared, proven byte-identical",
  "M <= 16 is gemm_nt_q_w2a8_m16's by its declared C_LE(\"M\", 16); fp8a8 and fp8a16 take their own "
  "rows, and the fp8a8 arm refuses a band carrying a_sum at its own line -- a row that silently "
  "ignores a band operand passes a checkpoint whose a_sum is garbage. A zero plane stored sub-byte "
  "(U2/U4) is refused and named, not approximated (rad_load_bits reads codes, not zeros). At "
  "kKC=32 a staged chunk draws 8 bytes from a row of pitch 512, so the band moves 8x the bytes it "
  "uses and is far from its read floor; a wider k-chunk, not more loads in flight, is what would "
  "close it",
  RAD_DOMAIN_DEVICE, R3D_PRIO_VOCAB_M1,
  ROW(cGemmQW2A8Tiled), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w2a8_tiled, nullptr,
  r3d_layout_gemm_q_fp8_weight, r3d_relayout_gemm_q_fp8_weight, r3d_shape_gemm_nt_q },

/* ------------------------------------------------------------------ r3d_gemm_fp8a8_gated.hip
 *
 * The gate couples column i to column n+i, so a lane owns both dots and the coupling costs no extra
 * weight traffic. The block is one group by 256/group rows so the group's amax closes in-block. That
 * block shape is the quantiser's rule, so there is nothing to tune, and the row states its M ceiling
 * in the prose. Throughput is low (about 0.7% of the card's fp8 peak at M=64), but the alternative
 * is the reference's host GEMM.
 */
{ "gemm_nt_q_gated_fp8a8", "gemm_nt_q_gated", "gemm",
  "the gated MLP band's GEMM, its gate and its output quantiser in one launch",
  "a block is one 128-wide group by 256/group rows, so the group's amax closes in-block: wave max "
  "then an LDS fold over the lanes that share a group. The gate couples column i to column n+i, so a "
  "lane owns both dots and the coupling costs no extra weight traffic. No [M,N] bf16 intermediate "
  "exists: the row is narrowed, gated and coded from registers, which is why there is no scratch and "
  "no second pass. The narrowing matches the unfused chain -- both halves bf16 before the gate, the "
  "activation NOT narrowed before the product, the product formed in f32 then narrowed -- so the "
  "codes are the codes the chain would have written, byte for byte at the band",
  "fp8_e4m3 a and b, f32 a_scale, bf16 b_scale (f32 accepted and equal bit for bit; fp8a8f32 is a "
  "different dtype and takes no row here), f32 scale, E4M3 q, bf16 out_bf16 optional; f32 "
  "accumulation with -ffp-contract=off; f16x4 dot pairing, the code widening exact. M > 64, N odd, K "
  "% group != 0, group outside {32,64,128,256} or wider than n, sk > 1 and an absent act are refused, "
  "not approximated",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmGated), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_gated, nullptr, nullptr, nullptr,
  r3d_shape_gemm_nt_q_gated },

{ "dflash_select_bf16", "dflash_select", "draft",
  "DFlash2's greedy walk over the candidate lattice, one lane a (sequence, candidate) pair",
  "score[l][c] = unary[l][c] + SUM_r pred[pid(l)][r] * hp[l][r] * succ[cand[l][c]][r], the rank sum "
  "ascending in r in f32 with ax[r] = pred * hp rounded to f32 BEFORE it meets succ, because the "
  "reference writes it to a float array first; argmax strict `>` over c ascending, so the earliest "
  "candidate wins a tie and a NaN never takes the lead; pid(0) is the anchor at m * anchor_stride "
  "and pid(l) the walk's own previous pick, so the steps loop is serial with the two LDS round "
  "trips its dependence requires; hp absent is a plane of ones; a predecessor outside the vocabulary "
  "contributes zero and a candidate outside it is scored by unary alone, which is what makes a "
  "shard's padding rows legal input",
  "bf16 codebooks (pred, succ, optional hp) with contiguous rows, f32 unary, i32 cand, anchor and "
  "tokens; pitched rows served on cand, unary, tokens and hp, a row stride INSIDE a codebook row "
  "refused (RAD_E_STRIDE); top_k > 256 refused (RAD_E_UNSUPPORTED); a cand plane describing more "
  "sequences than the band's M is refused, not truncated",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cDflashSelect), NOTUNE,
  nullptr, nullptr, r3d_rad_dflash_select, nullptr, nullptr, nullptr,
  r3d_shape_dflash_select },

/* ------------------------------------------------------------------ r3d_vocab_bf16.hip
 *
 * The head is the largest read in a decode step (510 MiB at 130560 x 2048), and the sampler reads
 * f32, so the store is f32 rather than narrowed and a prefill band writes about 1 GB. Against the
 * reference the f32 plane is at rel_l2 1.77e-6 (about 30 f32 ulps at K=2048), from dot2 folding two
 * products into one rounding; widening the operands to f16 first changes nothing. The head's own
 * bf16 quantisation (4.9e-4 an element) is 300x coarser.
 */
{ "logits_gemm_bf16", "logits_gemm", "vocab",
  "logits[M, n_vocab] = x[M, n_embd] @ lm_head[n_vocab, n_embd]^T, the head read once a token",
  "M <= 16: a wave a run of vocabulary columns, one output element a lane, the head's rows streamed "
  "past an activation kept in registers, no wave reduction. M > 16: 128x128 tiles through LDS, 8x8 "
  "f32 accumulators a lane, the head staged with an 8-element swizzle, k ascending in 32-code "
  "chunks. The grid is vocab-wide at every M, so there is no split-K and no scratch",
  "bf16 x, f32 logits: every sampler stage reads f32, so the narrowing the schema does not ask for "
  "is not performed, and a prefill band's output plane is M x n_vocab f32. The head arrives in the "
  "form the container stores and the row reads both, chosen by the operand's header, not by a "
  "parameter: bf16 rows in the order they are stored, or -- what this checkpoint's output.weight "
  "actually is -- E4M3 codes with a bf16 scale per 128x128 block as two container planes, which the "
  "row's layout hook presents as the one operand the plan says they are, moving no byte. E4M3 and "
  "int8 widen to f16 exactly, so the fold into f16 is the only arithmetic the codes pay: 2.77e-04 "
  "from the reference's exact law, one f16 rounding, 250x finer than the code's own quantum. A head "
  "with a zero plane is w2a8's grid and is refused at the shim's own line; so is a scale block that "
  "is not 128x128, which is gemm_nt_q's grid",
  RAD_DOMAIN_DEVICE, R3D_PRIO_VOCAB_M1,
  ROW(cLogitsGemm), tImpl2, NELEM(tImpl2), impl_tune_valid,
  nullptr, nullptr, r3d_rad_logits_gemm_bf16, nullptr,
  r3d_layout_logits_head, r3d_relayout_logits_head, r3d_shape_logits_gemm_bf16 },

/* ------------------------------------------------------------------ r3d_gemm_fp8a8_tiled.hip
 *
 * The prefill projection. At these densities the dot product, not memory bandwidth, is the limit
 * (about 27-36% of the card's dense f16 peak at the graph's bands), which is why the E4M3 decoder is
 * a table: the decode ALU is on the critical path.
 */
{ "gemm_nt_q_fp8a8_tiled", "gemm_nt_q", "gemm",
  "y[M,N] = a[M,K] @ dequant(b)^T at prefill heights, the group sub-sum from the raw codes and one "
  "scale per group, which is the reference's law and the reason the codes are never pre-scaled",
  "the tune's impl axis picks the decode: 0 = the block's 512-byte E4M3->f16 table, 1 = the "
  "per-byte ALU formula. 128x128 tiles through LDS, 64 f32 accumulators a lane, k in 32-code chunks, "
  "the family's 8-element "
  "XOR swizzle on the staged rows; both sides decode through the block's 512-byte E4M3->f16 table "
  "and leave staging as f16 with their scale folded in -- a_scale per staged row, b_scale one scalar "
  "per tile, because the tile's 128 columns are exactly the scale plane's row extent and 32 divides "
  "128. One accumulator, no per-group partials, 237 VGPRs and no spill",
  "E4M3 codes both sides, f32 a_scale per (row, group of 128), bf16 b_scale per 128x128 block -- the "
  "container's own block; a {1,group} scale block is w2a8's grid and is refused, as is a zero plane "
  "and an a_sum, fp8a8 being symmetric. y bf16 with one narrowing at the store, and at a bf16 output "
  "the store's ulp (3.9e-3) is the binding floor, not the fold's (9.8e-4 of the quadrature scale): "
  "83% of outputs are bit-exact, 99.93% land within the fold plus one store ulp, none beyond. The "
  "fold is v_mul_f16, never f16x2_pack, which rounds toward zero and flushes subnormals",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQFP8Tiled), tImpl2, NELEM(tImpl2), impl_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_q_fp8_tiled, nullptr,
  r3d_layout_gemm_q_fp8_weight, r3d_relayout_gemm_q_fp8_weight, r3d_shape_gemm_nt_q },

{ "gemm_nt_q_mxfp4a8_tiled", "gemm_nt_q", "gemm",
  "y[M,N] = a[M,K] @ dequant(b)^T at prefill heights, the weight in OCP MXFP4: e2m1 codes, two a "
  "byte, one e8m0 scale per 32 of K; the 32-block partial scales at the block boundary and the "
  "blocks sum in ascending k",
  "the tune's impl axis picks the decode: 0 = the block's 512-byte table, 1 = the per-byte ALU "
  "formula. 128x128 tiles through LDS, 64 f32 accumulators a lane for the block partials and 64 more "
  "for the running sum, k in 32-code chunks -- one OCP 32-block a chunk -- the family's 8-element XOR "
  "swizzle on the staged rows; the codes unpack to e4m3 through eight fixed magnitude bytes and "
  "decode through the block's 512-byte table, a_scale folded per staged row, the e8 read at "
  "staging and held in registers for the flush",
  "E4M3 a with an f32 scale per (row, 128 of K), fp4_e2m1 codes with an e8m0 scale per (row, 32 "
  "of K) -- `b_ref`, the same exponent plane, is accepted and ignored, bf16 y, fp32 accumulation. "
  "The e2m1 magnitude is exact in e4m3, the code products are "
  "exact in f16, the 32-block partial scales by a power of two in f32, and the store is the only "
  "narrowing; a reserved e8 (0, 255) is refused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQMxfp4a8Tiled), tImpl2, NELEM(tImpl2), impl_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_q_mxfp4a8_tiled, nullptr,
  nullptr, nullptr, r3d_shape_gemm_nt_q_mxfp4a8 },

/* The same tiling on gemm_nt's bf16 operands, the prefill projection the reference runs on the
 * host. The exactness is the m16 row's, because it is the same dot. */
{ "gemm_nt_bf16_tiled", "gemm_nt", "gemm",
  "a[M,K] @ b[N,K]^T at prefill heights, f32 accumulate, one narrowing at the store",
  "the tune's impl axis picks the kernel: 0 = 128x128 tiles through LDS, 8x8 f32 accumulators a "
  "lane, k ascending in 32-code chunks, the partials closed in the tile; 1 = the same tile, the bf16 "
  "decode in registers (the per-byte ALU formula); 2 = the same tile on the wave-matrix (eight 16x16 "
  "tiles a wave); 3 = the 16-column-tile wave-matrix kernel, MT 16-row tiles a block (the M = 17..127 "
  "band); M <= 16 stays with gemm_nt_bf16_m16, whose wave-per-run-of-columns "
  "shape is the right one there -- the two rows' M ranges are complements, so neither competes for "
  "the other's band",
  "bf16 a, b and y, fp32 accumulation; the pair is folded into the accumulator by one rounding, so "
  "about 0.3% of a long-K output lands one bf16 store ulp from the reference's, and no output is "
  "further",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_TILED,
  ROW(cGemmNTTiled), tImpl4, NELEM(tImpl4), impl_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_bf16_tiled, nullptr,
  nullptr, nullptr, r3d_shape_gemm_nt },


};

/* ================================================================== the exports
 *
 * The two exports every plugin needs, the four that make a kernel library, and
 * `rad_kernel_concurrent`.
 *
 * No `rad_kernel_chain`: chaining replaces the queue's ordering between two dependent launches with
 * a spin in the kernel, and no row here has a chained form, so every answer would be 0. Omitting the
 * symbol means the same (ABI, `rad_kernel_chain`: "without the symbol, or with 0, the engine never
 * asks").
 *
 * No `rad_kernel_collective`: these rows run on single-GPU cards, and an all-reduce over the PCIe x1
 * slot most of them sit in is slower than going through the host.
 */
extern "C" int rad_kernel_count(void) {
    return (int)(sizeof(kKernels) / sizeof(kKernels[0]));
}
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    if (i < 0 || i >= rad_kernel_count()) return nullptr;
    return &kKernels[i];
}

/* Which rows touch nothing but their operands and their scratch.
 *
 * The elementwise, norm, quant, cast, rope, cache and gather rows: one launch, one pass, no state
 * between launches, so two may run at once and the engine can start one before an independent
 * earlier one finishes.
 *
 * Not the GEMM rows, because of the scratch: the arena belongs to the op instance, so two layers'
 * decode GEMMs launched together share it and their split-K partials and arrival counters would
 * collide. With sk = 1 a GEMM has no partials, but the promise is per row and sk is chosen per
 * geometry, so the row states the worst case.
 */
extern "C" int rad_kernel_concurrent(int i) {
    const RadKernelInfo* k = rad_kernel_at(i);
    if (!k) return 0;
    return k->scratch == nullptr;
}

static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL,
    "libr3d",
    "0.1.0",
    "the RDNA3 (gfx11) kernel library: the elementwise, norm, rope, cache and embedding rows, fp8 "
    "and int8 activation quantisation with its dequant and gated forms, and the skinny bf16 and "
    "fp8a16 GEMMs, the matrix work through v_dot2acc_f32_f16 rather than RDNA3's WMMA "
    "fragment layout",
    /* The architectures this build's code objects run on, spelled as `RadDeviceProps::arch`. The
     * loader warns once when the code object it picked names a card this list does not
     * (`target_covers`), so the string must be the build's real targets. R3D_BUILD_TARGET comes
     * from CMakeLists.txt as a compile definition; the fallback is the whole family. */
    R3D_BUILD_TARGET,
};

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }

/* libr3d declines a machine with no gfx11 card. The loader would leave its device rows out there
 * (no code object fits), but rad-convert and the other tools resolve kernels without that check,
 * and libr3d, ahead of libr4d in file-name order, would choose the stored form of every weight it
 * has a row for: an RDNA4 container laid out for RDNA3 kernels, or one weight wanted in two forms.
 *
 * The test reads the KFD topology (gfx_target_version 110000..119999 is the gfx11 family) rather
 * than asking a runtime, which open must not initialise. KFD lists only the cards the process may
 * open, so a container without /dev/kfd sees the CPU node alone; with no topology at all there is
 * no card, and the library declines too. */
extern "C" int rad_plugin_open(void) {
    const char* root = "/sys/class/kfd/kfd/topology/nodes";
    DIR* d = opendir(root);
    if (!d) return RAD_E_UNSUPPORTED;
    bool gfx11 = false;
    while (const dirent* e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        const std::string path = std::string(root) + "/" + e->d_name + "/properties";
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) continue;
        char key[64];
        unsigned long long v = 0;
        while (std::fscanf(f, "%63s %llu", key, &v) == 2)
            if (!std::strcmp(key, "gfx_target_version") && v >= 110000ull && v < 120000ull)
                gfx11 = true;
        std::fclose(f);
    }
    closedir(d);
    return gfx11 ? RAD_OK : RAD_E_UNSUPPORTED;
}
