/* r3d.h -- the entry points libr3d exports, and the launchers behind them.
 *
 * Every unit has two layers, as in libr4d:
 *
 *   The launcher (`r3d_<op>(...)`, this header) takes pointers as longs, a geometry as ints and a
 *   stream as a long. It checks what a `__global__` cannot (nulls, non-positive extents, a pitch
 *   that does not describe the operand), computes the grid and issues one kernel with
 *   hipLaunchKernelGGL on the given stream. It returns 0 or a negative code: -1 a null pointer,
 *   -2 a bad extent, -3 a bad pitch, -4 the HIP launch failed. It allocates nothing and
 *   synchronises on nothing.
 *
 *   The shim (`r3d_rad_<op>(const RadArgs*, RadStream)`, declared in the unit) is the RadLaunchFn
 *   the row table names. It reads the operands with r3d_args.h, takes the dtype from the tensor
 *   (not the `dtype` parameter) and the row count from the operand (not `M`, which is the band's
 *   upper bound), checks strides, translates ABI dtypes to the codes below and returns a RAD
 *   status.
 *
 * A launcher is defined beside its kernel, in the same translation unit as the shim that calls
 * it, so one unit shows the whole path from RadArgs to the grid.
 *
 * Streams: a RadStream is the engine's queue, usable with hipLaunchKernel (and so <<<>>>),
 * hipMemcpyAsync, hipMemcpy2DAsync and hipMemsetAsync and with nothing else in HIP
 * (docs/PLUGIN.md §5). libr3d passes it to the launch and nowhere else.
 */
#ifndef RAD_R3D_H
#define RAD_R3D_H

#include "r3d_plugin.h"

extern "C" {

/* ================================================================== r3d_model_bf16.hip
 *
 * One workgroup per row. Every launcher takes a row pitch in elements for each operand that can be
 * a slice (a column slice of a fused projection is the usual case), 0 meaning "dense, so n".
 */

/* y = x * rsqrt(mean(x^2) + eps) * (w + wadd) * mult. `wdt` is R3D_DT_BF16 or R3D_DT_F32 for a
 * gain, R3D_DT_N for none (gain 1; the shim then refuses wadd, as libref does). */
int r3d_rmsnorm_bf16(long x, long w, long y, int M, int n, float eps, float wadd, float mult,
                     int x_ld, int y_ld, int wdt, long stream);

/* The same at f32 activations, for an f32 checkpoint's residual stream; the store narrows
 * nothing. `wdt` is the gain's dtype code, R3D_DT_N for none. */
int r3d_rmsnorm_f32(long x, long w, long y, int M, int n, float eps, float wadd, float mult,
                    int x_ld, int y_ld, int wdt, long stream);

/* y = (x - mean) / sqrt(var + eps) * w + b, mean and variance in the reference's two passes.
 * `b` may be 0 (no bias). */
int r3d_layernorm_bf16(long x, long w, long b, long y, int M, int n, float eps,
                       int x_ld, int y_ld, long stream);

/* y = a + b and y = a * b. `bcast`: b holds n elements, applied to every row. `div` is the `add`
 * op's optional divisor, 0 for none; with it, y = rd(rd(a + b) / div). */
int r3d_add_bf16(long a, long b, long y, int M, int n, int a_ld, int b_ld, int y_ld, float div,
                 int bcast, long stream);
int r3d_add_f32(long a, long b, long y, int M, int n, int a_ld, int b_ld, int y_ld, float div,
                int bcast, long stream);
int r3d_mul_bf16(long a, long b, long y, int M, int n, int a_ld, int b_ld, int y_ld, int bcast,
                 long stream);
int r3d_mul_f32(long a, long b, long y, int M, int n, int a_ld, int b_ld, int y_ld, int bcast,
                long stream);

/* y = x, one dtype to another, over M rows of n at the given pitches. The dtypes are R3D_DT_*.
 * An integer pair is copied directly (ids above 2^24 survive an i64 -> i32 copy); everything else
 * widens to f32 and narrows once on the store. */
int r3d_cast(long x, long y, int M, int n, int x_ld, int y_ld, int from_dt, int to_dt, long stream);

/* Unary activations computed in f32 and narrowed once by the store: silu x*sigmoid(x), gelu
 * 0.5x(1+erf(x/sqrt2)), sigmoid 1/(1+exp(-x)). */
int r3d_silu_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream);
int r3d_gelu_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream);
int r3d_sigmoid_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream);
int r3d_silu_f32(long x, long y, int M, int n, int x_ld, int y_ld, long stream);

/* y = act(gate) * up, `act` R3D_ACT_SILU, R3D_ACT_GELU (the exact erf form) or R3D_ACT_SIGMOID.
 * The two inputs are the halves of a fused [M, 2n] projection, so their pitches are separate. */
int r3d_act_mul_bf16(long gate, long up, long y, int M, int n, int g_ld, int u_ld, int y_ld,
                     int act, long stream);

/* y[i] = silu(gate_up[i]) * gate_up[n + i]: the llama MLP's gated product over one fused plane,
 * gate in the first half (rad-convert's fusion order). The activation is not narrowed before the
 * product; `act_mul`, over two operands, stands for the unfused pair that stores it. `gu_ld` is the
 * fused row's pitch (>= 2n), `y_ld` the output's; 0 takes each one's dense width. */
int r3d_silu_mul_bf16(long gate_up, long y, int M, int n, int gu_ld, int y_ld, long stream);

/* Row-wise softmax over f32, maximum subtracted, the denominator summed in ascending order. */
int r3d_softmax_f32(long x, long y, int M, int n, int x_ld, int y_ld, long stream);

/* y = x * s, where `s` is the op's [1] weight plane in device memory, dtype `s_dt` (R3D_DT_BF16 or
 * R3D_DT_F32). The kernel reads the scalar, not the launcher, because a launch may not copy device
 * memory to the host. Every lane loads the same address, one uniform load per block. `s` is 0 only
 * for a call the shim has already refused.
 *
 * fill (y = v) takes its value as a parameter, so it is passed by value. */
int r3d_scale_bf16(long x, long s, long y, int M, int n, int s_dt, int x_ld, int y_ld, long stream);
int r3d_scale_f32(long x, long s, long y, int M, int n, int s_dt, int x_ld, int y_ld, long stream);
int r3d_fill(long y, int M, int n, int y_ld, int dt, float v, long stream);

/* In-place rotary on a fused qkv row: q, then k; v is untouched. `mode` is R3D_ROPE_NEOX or
 * R3D_ROPE_GPTJ; `rot` the rotary width (<= head_dim, even). `scale` divides the position: the
 * angle of pair i is (pos / scale) * theta^-i, as in ref_rope. YaRN's magnitude (`yarn_factor`) is not
 * implemented. `pos` is M i32 positions, one per row. */
int r3d_rope_bf16(long qkv, long pos, int M, int head_dim, int n_head, int n_head_kv, float theta,
                  float scale, int mode, int rot, int qkv_ld, long stream);

/* K and V into a paged cache at slot_mapping's slots. The cache is [blocks, kv_heads, block_size,
 * 2*head_dim] with the strides the shim reads off the operand (sb/sh/so/sd). A negative slot is
 * skipped, and so is one at or past `slot_max` = n_blocks * block_size, computed as
 * `r3d_shape_kv_store` sizes the cache. The reference refuses such a slot on the host; a launch
 * cannot refuse a slot it has not read, so the kernel drops it rather than write past the cache.
 * `kv_dt`: R3D_DT_BF16 stores the values, R3D_DT_U8 encodes E4M3 with the reciprocal of
 * k_scale / v_scale, R3D_DT_F16 stores f16. */
int r3d_kv_store(long k, long v, long sm, long kv, int M, int head_dim, int n_head_kv,
                 int block_size, int kv_dt, float k_scale, float v_scale, int k_ld, int v_ld,
                 long long slot_max, long sb, long sh, long so, long sd, long stream);

/* x[r] = wte[ids[r] - vocab_offset] * wscale, narrowed once. A row whose id is outside the table
 * is zeroed, as in libref. */
int r3d_embed_lookup(long ids, long wte, long x, int M, int n_embd, int n_vocab, int vocab_offset,
                     float wscale, float mult, int x_ld, int w_ld, int dt, long stream);

/* y[r] = x[idx[r]] and x[idx[r]] = v[r], rows of n elements.
 *
 * `src_rows` / `dst_rows` are the plane's own row count, which only the shim knows (from the
 * operand's numel). The reference refuses an out-of-range index on the host; a launch cannot refuse
 * an index it has not read, so the kernel enforces the bound: a gather at idx >= src_rows writes
 * the zero row of libref's padding form, a scatter at idx >= dst_rows drops the row. This keeps
 * every access inside the plane. 0 means unchecked and is a shim bug, not a caller option. */
int r3d_gather_rows(long x, long idx, long y, int M, int n, int x_ld, int y_ld, int dt,
                    long long src_rows, long stream);
int r3d_scatter_rows(long v, long idx, long x, int M, int n, int v_ld, int x_ld, int dt,
                     long long dst_rows, long stream);

/* ================================================================== r3d_quant_fp8.hip
 *
 * The quantisers write a plain row-major code plane and one f32 scale per (row, group), with no
 * fragment permutation, because a dot2 GEMM reads codes in order.
 */

/* Per-(row, group) symmetric quantisation. scale = amax * (1/top) with top 448 for E4M3 and 127
 * for int8, the codes v * (1/scale) clamped to +-top and rounded by the codec. An all-zero group
 * gets scale 1. */
int r3d_quant_act_fp8(long x, long q, long s, int M, int n, int group, int x_ld, int s_ld,
                      long stream);
int r3d_quant_act_i8(long x, long q, long s, long asum, int M, int n, int group, int x_ld, int s_ld,
                     long stream);

/* The same int8 code at `group` granularity (amax / 127 per (row, group), 1 for an all-zero
 * group), stored row-major. This is the plane libr3d's i8a8 GEMM reads; `quant_act_i8` above is the
 * per-row plane the WMMA-ordered ops read. The two are not interchangeable. */
int r3d_quant_act_i8g(const void* x, void* q, void* s, int M, int n, int group,
                      long long x_ld, long long s_ld, long stream);

/* The gated form: gate_up is [M, 2n] (gate in the first n, up in the second), and what is
 * quantised is act(gate) * up. `q`/`s` carry that product's codes and scales; `out_bf16`, when
 * present, also receives the product in bf16. */
int r3d_gated_quant_fp8(long gate_up, long q, long s, long out_bf16, int M, int n, int group,
                        int act, long stream);

/* Residual add, RMS norm and E4M3 quantiser in one pass, so the normalised row never goes back to
 * VRAM. `res` is the residual plane (in/out), 0 when the op has none. With it, the sum is stored
 * narrowed to bf16 before the variance reads it, so the norm is over the bf16 sum: the `add` +
 * `rmsnorm` pair, not `rmsnorm_add`, which normalises the unrounded sum. `ob`, when present,
 * receives the bf16 values the codes were taken over. Pitches are in elements of their own operand,
 * `s_ld` in scales (dense: n/group). `group` must divide n; this pass has no zero-padded partial
 * group. */
int r3d_rmsnorm_quant_fp8(long x, long res, long w, long q, long s, long ob, int M, int n,
                          int group, float eps, float wadd, int x_ld, int res_ld, int ob_ld,
                          int q_ld, int s_ld, int wdt, long stream);

/* y = code * scale, the read of a container's own quantised weight: group-wide f32 scales, E4M3 or
 * int8 codes. */
int r3d_dequant(long q, long s, long y, int M, int n, int group, int q_dt, int s_ld, int y_ld,
                long stream);

/* ================================================================== r3d_gemm_*.hip
 *
 * y[M,N] = a[M,K] @ b[N,K]^T. One output element per lane, k ascending, f32 accumulate, the sum
 * narrowed once by the store. Split-K (sk > 0) writes partials to `scratch` and reduces them in a
 * second kernel on the same stream; the row's scratch hook sizes the arena for the largest sk the
 * band allows.
 */

/* bf16 (and f16) operands. `nt` 1 streams the weight with non-temporal loads. */
int r3d_gemm_nt_bf16_m16(long a, long b, long y, int M, int N, int K, int a_ld, int y_ld, int sk,
                         int wv, int nt, long scratch, long stream);
int r3d_gemm_nt_bf16_tiled(long a, long b, long y, int M, int N, int K, int a_ld, int y_ld, int sk,
                           int kb, int nt, long scratch, long stream);

/* A block-scaled fp8 weight: b is E4M3 codes, b_scale one scale per (pb1 rows of N, pb2 of K)
 * block, pb1/pb2 the container's block sizes (both default to `group`). The scale plane is bf16 in
 * the container's quantised family (`gemm_q_shape` writes `o.t(f32s ? RAD_F32 : RAD_BF16, {nb,
 * kb})`) and f32 only under the separate dtype `fp8a8f32`. a is bf16 (fp8a16, no a_scale) or E4M3
 * codes with per-(row, group) f32 a_scale (fp8a8). bs_dt is the scale plane's RadDataType:
 * RAD_BF16 reads bf16, anything else (0 included) reads f32. */
int r3d_gemm_nt_q_fp8_ex(long a, long a_scale, long b, long b_scale, long y, int M, int N, int K,
                         int group, int pb1, int pb2, int a_ld, int y_ld, int sk, int nt,
                         long scratch, int bs_dt, long stream, long bias);

/* ================================================================== r3d_attn_paged_h128.hip
 *
 * Paged causal attention. `kv` is the cache the shim read its strides from; `bt` a
 * [num_seqs, blocks] block table; `seqused` per-sequence lengths; cu_seqlens, when present, the
 * ragged query layout; k_descale / v_descale the per-(seq, head) fp8 cache descales.
 */
int r3d_attn_decode_h128(long q, long kv, long bt, long seqused, long k_desc, long v_desc,
                         long cu, long out, int num_seqs, int q_len, int head_dim, int gqa,
                         int n_head, int block_size, int bt_cols, int bt_rs, int bt_cs,
                         int max_ctx, int window, int causal,
                         float scale, int kv_dt, long q_ld, long out_ld, int splits, long scratch,
                         long stream);
int r3d_attn_prefill_h128(long q, long kv, long bt, long seqused, long k_desc, long v_desc,
                          long cu, long out, int num_seqs, int q_len, int head_dim, int gqa,
                          int n_head, int block_size, int bt_cols, int bt_rs, int bt_cs,
                          int max_ctx, int window, int causal,
                          float scale, int kv_dt, long q_ld, long out_ld, long scratch,
                          long stream);

/* ================================================================== r3d_gemm_fp8a8_tiled.hip
 *
 * The quantised GEMM at prefill heights. b_row/bs_row are the container's row strides, 0 meaning
 * dense; `group` is the scale block, 128 on every declared band. The scales arrive as the
 * container's two planes behind one operand, as the row's layout hook presents them; the plain
 * two-tensor form is the same entry with bs_row dense.
 */
int r3d_gemm_nt_q_fp8_tiled(long a, long a_scale, long b_codes, long b_scale, long y, int M, int N,
                            int K, int group, int a_ld, long long b_row, int as_ld, long long bs_row,
                            int y_ld, int nt, long stream);

/* ================================================================== r3d_gemm_mxfp4a8.hip
 *
 * The quantised GEMM at dtype=mxfp4a8: an OCP-MXFP4 weight against the fp8 activation. `b` is the
 * e2m1 codes, two per byte, [N, K/2] u8; `b_scale` the e8m0 scale, one per 32 of K, [N, K/32] u8;
 * `a` E4M3 with `a_scale` f32 per (row, 128 of K); `y` bf16. b_row/e8_row are row strides in bytes,
 * 0 meaning dense. No zero point (symmetric scale); a reserved e8 (0, 255) is refused. M <= 16
 * takes the m16 launcher, M >= 17 the tiled one.
 */
int r3d_gemm_nt_q_mxfp4a8_m16(long a, long a_scale, long b, long b_scale, long y, int M, int N,
                              int K, int group, int a_ld, int y_ld, long long e8_row, long stream,
                               int alu);
int r3d_gemm_nt_q_mxfp4a8_tiled(long a, long a_scale, long b_codes, long b_scale, long y, int M,
                                int N, int K, int group, int a_ld, long long b_row, int as_ld,
                                long long e8_row, int y_ld, int nt, long stream, int alu);

/* ================================================================== r3d_gemm_inta8.hip
 *
 * The quantised GEMM at dtype=w4a8 and w8a8: symmetric int4 and int8 weight grids at decode
 * heights. `b` is the signed codes (I4 two per byte, or I8), [N, K]; `b_scale` the f16 scale, one
 * per (row, 128 of K), [N, K/128]; `a` int8 in step order with `a_scale` f32 per row; `y` bf16.
 * b_rb and sb_rb are the row strides (bytes / elements), 0 meaning dense. No zero point: the
 * dequant is code times scale. M <= 16 only; prefill heights are served by the tiled rows, and the
 * 8-bit grid at prefill heights is declared outside this unit.
 */
int r3d_gemm_nt_q_w4a8_m16(long a, long a_scale, long b, long b_scale, long y, int M, int N,
                           int K, int a_ld, int y_ld, long long b_rb, long long sb_rb, long stream);
int r3d_gemm_nt_q_w8a8_m16(long a, long a_scale, long b, long b_scale, long y, int M, int N,
                           int K, int a_ld, int y_ld, long long b_rb, long long sb_rb, long stream);

/* ================================================================== r3d_vocab_bf16.hip
 *
 * The vocab-sharded logits and the top-k that follows them.
 */

/* logits[M, n_vocab] = x[M, n_embd] @ lm_head[n_vocab, n_embd]^T: a decode arm for M <= 16 and a
 * prefill arm above, over a vocabulary-wide N. */
int r3d_logits_gemm_bf16(long x, long lm, long y, int M, int n_vocab, int n_embd, int x_ld,
                         int y_ld, int sk, int nt, long scratch, long stream);

/* ================================================================== r3d_sample_f32.hip
 * The sampler, and the vocabulary row it starts from.
 *
 * The shapes and the `params` plane are libref's: one `RadSampleParams` per row, 96 bytes of f32
 * words, read by field. `logits` is f32 and the candidate planes are word-typed (`cand_idx` i32,
 * `cand_val` f32, the pair planes i32 holding an id's bits beside a logit), as the reference
 * writes them.
 *
 * The full-vocabulary stages (argmax, temp, topk, logprobs, penalties, dry, ngram_ban) run one
 * block per row and are bandwidth-bound: at MiniCPM5-2B's band, 2048 rows of 130560 f32, the logits
 * are 1.02 GiB. The candidate-set stages (the four narrowings and pick) run one block per row of
 * `n_cand` and are arithmetic-bound. `pick` keeps the reference's `(float)std::exp((double)x)`,
 * because a hardware `v_exp_f32` two ulp away moves a draw whenever a probability lands within an
 * ulp of the `u < cum` boundary.
 *
 * M comes from the operand, never from the band, whose `M` is only an upper bound.
 */
int r3d_row_topk(long x, long idx, long val, long pairs, int M, int N, int R, int x_dt,
                 long long x_ld, long long idx_ld, long long val_ld, long long pairs_ld,
                 int vocab_off, long stream);
int r3d_sample_argmax(long lg, long tok, int M, int n_vocab, long long lg_ld, long stream);
int r3d_sample_temp(long lg, long params, int M, int n_vocab, long long lg_ld, long stream);
int r3d_sample_topk(long lg, long params, long ci, long cv, long pairs, int M, int n_vocab,
                    int n_cand, long long lg_ld, long long ci_ld, long long cv_ld,
                    long long pairs_ld, int vocab_off, long stream);
int r3d_sample_pick(long ci, long cv, long params, long tok, int M, int n_cand,
                    long long ci_ld, long long cv_ld, long stream);

/* The four narrowing stages share one shape and edit `cand_idx` / `cand_val` in place: top-p is a
 * prefix rule in candidate order, min-p a floor under the best candidate's probability, typical a
 * band around the set's entropy, XTC the exclude-top-choices cut. */
int r3d_sample_topp(long ci, long cv, long params, int M, int n_cand, long long ci_ld,
                    long long cv_ld, long stream);
int r3d_sample_minp(long ci, long cv, long params, int M, int n_cand, long long ci_ld,
                    long long cv_ld, long stream);
int r3d_sample_typical(long ci, long cv, long params, int M, int n_cand, long long ci_ld,
                       long long cv_ld, long stream);
int r3d_sample_xtc(long ci, long cv, long params, int M, int n_cand, long long ci_ld,
                   long long cv_ld, long stream);

/* The pre-filters over the logits row. `hist` is the row's history of global token ids, i32
 * [M, max_hist]. Its window, and the rule that a token is penalised at its first occurrence in the
 * window, are libref's (`hist_window`); the descriptor's `fill_chunk` writes the window into the
 * params plane. `vocab_off` shifts the ids a stage reports, for a split vocabulary. */
int r3d_sample_penalties(long lg, long params, long hist, int M, int n_vocab, long long lg_ld,
                         long long hist_ld, int vocab_off, long stream);
int r3d_sample_dry(long lg, long params, long hist, long breakers, int M, int n_vocab,
                   long long lg_ld, long long hist_ld, int vocab_off, long stream);
int r3d_sample_ngram_ban(long lg, long params, long hist, long ngram, long whitelist, int M,
                         int n_vocab, long long lg_ld, long long hist_ld, int vocab_off,
                         long stream);
int r3d_sample_mask(long lg, long params, long bitmask, int M, int n_vocab, long long lg_ld,
                    int mask_rows, long stream);
/* The gated MLP GEMM: b is E4M3 codes of N = 2n rows (gate rows [0,n), up rows [n,N)), b_scale one
 * bf16 scale per (group, group) block, as the container writes it. The row is narrowed to bf16 as
 * the unfused pair leaves it, gated (act: R3D_ACT_SILU | R3D_ACT_SIGMOID), narrowed again and coded
 * per `group` columns: q is the codes [M,n], scale the per-(row, group) amax scales, out_bf16 the
 * narrowed gated tensor if wanted (y_ld its row pitch, n when absent). M <= 64, sk == 1; no
 * scratch, because the row never leaves the registers. */
int r3d_gemm_nt_q_gated(long a, long a_scale, long b, long b_scale, long q, long scale,
                        long out_bf16, int M, int N, int K, int group, int act,
                        int a_ld, int q_ld, int y_ld, int sk, int nt, long scratch, long stream);

/* The same kernel naming the scale plane's dtype. The entry above passes RAD_BF16, the container's
 * form for this op; f32 is accepted and gives bit-identical results. */
int r3d_gemm_nt_q_gated_ex(long a, long a_scale, long b, long b_scale, long q, long scale,
                           long out_bf16, int M, int N, int K, int group, int act,
                           int a_ld, int q_ld, int y_ld, int sk, int nt, long scratch,
                           int bs_dt, long stream);

/* DFlash2's greedy walk over the candidate lattice. `hp` may be 0, meaning the plane of ones the
 * reference uses when the operand is absent. cand, unary and tokens have M * steps rows ordered
 * (sequence, step); `anchor` is a row of M * anchor_stride ids. */
int r3d_dflash_select(long cand, long unary, long hp, long anchor, long pred, long succ,
                      long tokens, int M, int steps, int top_k, int rank, int n_vocab,
                      int anchor_stride, long long cand_ld, long long un_ld, long long hp_ld,
                      long long pred_ld, long long succ_ld, long long tok_ld, long stream);
int r3d_sample_logprobs(long lg, long params, long stats, int M, int n_vocab, int k,
                        long long lg_ld, long long stats_ld, int vocab_off, long stream);

}  /* extern "C" */

#endif  /* RAD_R3D_H */
