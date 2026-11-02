/* avx_quant.cpp -- the quantisers, the rotation, and the Gram accumulator.
 *
 * Walsh-Hadamard, bit-identical to libref. The rotation is butterflies at strides 1, 2, 4, ... g/2,
 * pairing i with i + h where (i & h) == 0. The stride matrices commute in exact arithmetic but not
 * in f32, so libref's ascending order is kept: each butterfly is the same add and subtract on the
 * same two floats, the rotated block is bit-identical, and the rotated-int8 family holds at the
 * quantiser tolerance (1e-6). Only the lane mapping is vectorised (avx_had.h):
 *
 *   h >= VF_N   the pair (i, i+h) is two whole vectors: one add, one sub, two stores.
 *   h <  VF_N   the partner is a swizzle within one vector (adjacent lanes, lane pairs, 128-bit
 *               lanes, 256-bit halves), results selected by a lane mask. Four such strides at
 *               AVX-512, three at AVX2; left scalar they would dominate at g = 128.
 *
 * The rotation runs once per row. libref rotates each block twice (once for amax, once for the
 * codes) because it has no row-sized scratch; here the row is rotated into scratch once.
 *
 * `group` has three meanings (docs/OPS.md); reading the wrong one gives plausible numbers:
 *
 *   quant_act_i8                           one scale per row; `group` is the width of the code
 *                                          sums `asum` holds.
 *   quant_act_i8g, quant_act_fp8, dequant  contiguous elements sharing one scale; `scale` is
 *   and the fp8 fusions                    [M, ceil(n/group)].
 *   had_* int8 forms                       the rotation width; one scale per row, carrying
 *                                          1/sqrt(group).
 *   had_* fp8 forms                        both, with no normalisation: <Hw, Ha> = g <w, a>, and
 *                                          the whole g is folded into the stored weight scale at
 *                                          convert, where dividing a bf16 by a power of two is
 *                                          exact. Splitting it as int8 does would round
 *                                          amax/7/sqrt(g) into bf16 (~0.2% error per group).
 */
#include "avx_vec.h"
#include "avx_common.h"

using namespace avx;

#include "avx_had.h"
#include <algorithm>
#include <vector>

/* Codes are stored with out_row; for RAD_I8 that is the vectorised clamp + cvtps/packs path
 * (avx_vec.h), bit-exact with libref's nearbyintf-and-clamp. */

/* Scale a row by a scalar, in place. */
static inline void scale_row(float* v, int64_t n, float s) {
    const vf vs = vf_set1(s);
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) vf_storeu(v + i, vf_mul(vf_loadu(v + i), vs));
    for (; i < n; ++i) v[i] *= s;
}

/* ------------------------------------------------------------------ quant_act_i8 */
/* libref's arithmetic (ref_quant.cpp): one scale per row, max(amax, 1e-8) / 127, codes rounded to
 * nearest even and stored in the step order, the middle two dwords of every 16-byte step
 * exchanged. `asum` gets the per-`group` code sums, group-major [n/group][ceil(M/16)*16], taken
 * from the row-major codes before the exchange. */
AVX_KERNEL(quant_act_i8) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const RadTensor* as = rad_arg_in(a, 3);
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0 || n % 16) return RAD_E_SHAPE;
    if (q->dtype != RAD_I8) return RAD_E_DTYPE;
    if (!dense_row(q)) return RAD_E_STRIDE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows) return RAD_E_SHAPE;
    const int64_t g = as ? p_int(a, "group", 0) : 0;
    const int64_t mpad = (rows + 15) / 16 * 16;
    if (as && (g <= 0 || n % g || numel(as) < n / g * mpad)) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        const float* v = in_row(x, rowoff(x, r, n), n, buf);
        int8_t* qb = (int8_t*)byte_at(q->data, q->dtype, rowoff(q, r, n));
        const float scale = std::max(row_amax(v, n), 1e-8f) / 127.0f;
        cvt_scaled_f32_to_i8(v, 1.0f / scale, qb, n);
        stt(s, rowoff(s, r, 1), scale);
        for (int64_t gi = 0; as && gi < n / g; ++gi) {
            int64_t sum = 0;
            for (int64_t i = gi * g; i < (gi + 1) * g; ++i) sum += qb[i];
            stt(as, gi * mpad + r, (float)sum);
        }
        for (int64_t k = 0; k < n; k += 16) {
            uint32_t d1, d2;
            std::memcpy(&d1, qb + k + 4, 4);
            std::memcpy(&d2, qb + k + 8, 4);
            std::memcpy(qb + k + 4, &d2, 4);
            std::memcpy(qb + k + 8, &d1, 4);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ quant_act_i8g */
/* libr4d's quant_act_i8g: per `group` contiguous elements a scale amax * (1/127) (1 for an
 * all-zero group), codes x * (1/scale) rounded to nearest even, row-major. */
AVX_KERNEL(quant_act_i8g) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;

    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    const bool dense_i8 = q->dtype == RAD_I8 && dense_row(q);
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        const float* v = in_row(x, rowoff(x, r, n), n, buf);
        int8_t* qb = dense_i8 ? (int8_t*)byte_at(q->data, q->dtype, rowoff(q, r, n)) : nullptr;
        float* w = dense_i8 ? nullptr : scratch_f32_b((size_t)n);
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
            const float amax = row_amax(v + k0, k1 - k0);
            const float sc = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;
            const float inv = 1.0f / sc;
            if (dense_i8) {
                cvt_scaled_f32_to_i8(v + k0, inv, qb + k0, k1 - k0);
            } else {
                const vf vi = vf_set1(inv);
                int64_t i = k0;
                for (; i + VF_N <= k1; i += VF_N) vf_storeu(w + i, vf_mul(vf_loadu(v + i), vi));
                for (; i < k1; ++i) w[i] = v[i] * inv;
            }
            stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
        }
        if (!dense_i8) out_row(q, rowoff(q, r, n), n, w);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ had_quant_act_i8 */
/* Rotation at width `group`, then symmetric int8 with one scale per row carrying the
 * 1/sqrt(group); amax is over the whole rotated row. */
AVX_KERNEL(had_quant_act_i8) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = p_int(a, "group", 0);
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (g > AVX_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows) return RAD_E_SHAPE;
    const float norm = 1.0f / std::sqrt((float)g);

    /* Dense i8: one fused scale+encode call per row. */
    if (q->dtype == RAD_I8 && dense_row(q)) {
        AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
        for (int64_t r = 0; r < rows; ++r) {
            float* buf = scratch_f32((size_t)n);
            const float* v = in_row(x, rowoff(x, r, n), n, buf);
            if (v != buf) std::memcpy(buf, v, (size_t)n * sizeof(float));
            fwht_row(buf, n, g);
            const float amax = row_amax(buf, n);
            cvt_scaled_f32_to_i8(buf, amax > 0.0f ? 127.0f / amax : 0.0f,
                                 (int8_t*)byte_at(q->data, q->dtype, rowoff(q, r, n)), n);
            stt(s, r * laststride(s), amax / 127.0f * norm);
        }
        return RAD_OK;
    }

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        /* in_row may return the operand itself for dense f32; copy before the in-place rotation
         * so the input is not modified. */
        const float* v = in_row(x, rowoff(x, r, n), n, buf);
        if (v != buf) std::memcpy(buf, v, (size_t)n * sizeof(float));
        fwht_row(buf, n, g);
        const float amax = row_amax(buf, n);
        scale_row(buf, n, amax > 0.0f ? 127.0f / amax : 0.0f);
        out_row(q, rowoff(q, r, n), n, buf);
        stt(s, r * laststride(s), amax / 127.0f * norm);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rmsnorm_had_quant_i8 */
/* Fused rmsnorm + Hadamard + quant_act_i8, the w4a8 counterpart of a plain rmsnorm.
 *
 * With `residual` (the fused_add_rms_norm form) the sum is written back into `residual` (INOUT;
 * the next layer's shortcut reads it). As in libref, the variance comes from the f32 sum and the
 * rotation reads the sum narrowed to the residual's dtype.
 *
 * `out_bf16` receives the post-norm, pre-rotation row for a layer that also has a bf16 consumer. */
AVX_KERNEL(rmsnorm_had_quant_i8) {
    if (a->n_t < 5) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[2], *q = &a->t[3], *s = &a->t[4];
    const RadTensor* res = rad_arg_in(a, 1);
    const RadTensor* ob  = rad_arg_in(a, 5);
    if (!x->data || !w->data || !q->data || !s->data) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const int64_t g = p_int(a, "group", 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (g > AVX_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows || numel(w) < n) return RAD_E_SHAPE;
    if (res && numel(res) < rows * n) return RAD_E_SHAPE;
    if (ob && numel(ob) < rows * n) return RAD_E_SHAPE;
    const float norm = 1.0f / std::sqrt((float)g);

    /* Gain with `wadd` folded in, widened once per launch (as in avx_norm.cpp). */
    float* gain = (float*)scratch_raw((size_t)n * sizeof(float));
    {
        const int64_t ws = laststride(w);
        if (ws == 1) row_to_f32(w->dtype, w->data, gain, n);
        else for (int64_t i = 0; i < n; ++i) gain[i] = ldt(w, i * ws);
        if (wadd != 0.0f) for (int64_t i = 0; i < n; ++i) gain[i] += wadd;
    }

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* src = scratch_f32((size_t)n);
        float* buf = scratch_f32_b((size_t)n);
        float* tmp = scratch_f32_c((size_t)n);

        float ss;
        if (res) {
            /* x + residual, stored, its variance taken, then narrowed through the residual's
             * dtype in scratch. */
            const float* xv = in_row(x, rowoff(x, r, n), n, src);
            const float* rv = in_row(res, rowoff(res, r, n), n, tmp);
            for (int64_t i = 0; i < n; ++i) buf[i] = xv[i] + rv[i];
            out_row(res, rowoff(res, r, n), n, buf);
            ss = row_sumsq(buf, n);
            if (res->dtype != RAD_F32) {
                row_from_f32(res->dtype, buf, tmp, n);
                row_to_f32(res->dtype, tmp, buf, n);
            }
            std::memcpy(src, buf, (size_t)n * sizeof(float));
        } else {
            const float* xv = in_row(x, rowoff(x, r, n), n, src);
            if (xv != src) std::memcpy(src, xv, (size_t)n * sizeof(float));
            ss = row_sumsq(src, n);
        }

        const float sc = 1.0f / std::sqrt(ss / (float)n + eps);
        const vf vsc = vf_set1(sc);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N)
            vf_storeu(buf + i, vf_mul(vf_mul(vf_loadu(src + i), vsc), vf_loadu(gain + i)));
        for (; i < n; ++i) buf[i] = src[i] * sc * gain[i];

        if (ob) out_row(ob, rowoff(ob, r, n), n, buf);   /* post-norm, pre-rotation */

        fwht_row(buf, n, g);
        const float amax = row_amax(buf, n);
        scale_row(buf, n, amax > 0.0f ? 127.0f / amax : 0.0f);
        out_row(q, rowoff(q, r, n), n, buf);
        stt(s, r * laststride(s), amax / 127.0f * norm);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gated_had_quant_i8 */
/* Gated product, then the rotation and the quantiser. `act` selects the w4a8 site: silu(a)*b is
 * down_proj's input, sigmoid(a)*b is o_proj's on a gated-attention layer. `mode` is libr4d's
 * integer spelling of the same choice.
 *
 * With `b` absent, `a` is a packed [M, 2n] gate_up plane, gate first. */
AVX_KERNEL(gated_had_quant_i8) {
    if (a->n_t < 4) return RAD_E_INVAL;
    const RadTensor *ga = &a->t[0], *q = &a->t[2], *s = &a->t[3];
    const RadTensor* up = rad_arg_in(a, 1);
    if (!ga->data || !q->data || !s->data) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = p_int(a, "group", 0);
    static const char* kActs[] = { "silu", "sigmoid" };
    const int act = p_enum(a, "act", kActs, 2, (int)p_int(a, "mode", 0));
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (act != 0 && act != 1) return RAD_E_INVAL;
    if (g > AVX_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rows = numel(q) / n;
    if (rows <= 0 || numel(s) < rows) return RAD_E_SHAPE;
    if (numel(ga) < rows * (up ? n : 2 * n)) return RAD_E_SHAPE;
    if (up && numel(up) < rows * n) return RAD_E_SHAPE;
    const float norm = 1.0f / std::sqrt((float)g);
    const int64_t gw = up ? n : 2 * n;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* gbuf = scratch_f32((size_t)gw);
        float* ubuf = scratch_f32_b((size_t)n);
        float* out  = scratch_f32_c((size_t)n);
        const float* gv = in_row(ga, rowoff(ga, r, gw), gw, gbuf);
        const float* uv = up ? in_row(up, rowoff(up, r, n), n, ubuf) : gv + n;

        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) {
            const vf gt = vf_loadu(gv + i);
            const vf act_v = (act == 1) ? vf_sigmoid(gt) : vf_silu(gt);
            vf_storeu(out + i, vf_mul(act_v, vf_loadu(uv + i)));
        }
        for (; i < n; ++i) {
            const float gt = gv[i];
            const float sg = 1.0f / (1.0f + sc_exp(-gt));
            out[i] = (act == 1 ? sg : gt * sg) * uv[i];
        }

        fwht_row(out, n, g);
        const float amax = row_amax(out, n);
        scale_row(out, n, amax > 0.0f ? 127.0f / amax : 0.0f);
        out_row(q, rowoff(q, r, n), n, out);
        stt(s, r * laststride(s), amax / 127.0f * norm);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ dequant */
/* q * scale, with the scale tensor's extent giving the group count. The inverse of quant_act_i8
 * only; it does not undo a rotation. */
AVX_KERNEL(dequant) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *s = &a->t[1], *y = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(y) / n;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;
    const int64_t ngrp = numel(s) / rows > 0 ? numel(s) / rows : 1;
    int64_t g = p_int(a, "group", 0);
    if (g <= 0) g = (n + ngrp - 1) / ngrp;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        float* out = scratch_f32_b((size_t)n);
        const float* v = in_row(q, rowoff(q, r, n), n, buf);
        const int64_t sb = rowoff(s, r, ngrp), ss = laststride(s);
        /* Per group, with the scale hoisted (libref divides per element to find it). Same
         * product. */
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            const int64_t k0 = gi * g;
            if (k0 >= n) break;
            const int64_t k1 = (k0 + g < n) ? k0 + g : n;
            const float sc = ldt(s, sb + gi * ss);
            const vf vs = vf_set1(sc);
            int64_t i = k0;
            for (; i + VF_N <= k1; i += VF_N) vf_storeu(out + i, vf_mul(vf_loadu(v + i), vs));
            for (; i < k1; ++i) out[i] = v[i] * sc;
        }
        /* Elements past the described groups use the last group's scale, as libref clamps
         * (`gi < ngrp ? gi : ngrp - 1`). */
        if (ngrp * g < n) {
            const float sc = ldt(s, sb + (ngrp - 1) * ss);
            for (int64_t i = ngrp * g; i < n; ++i) out[i] = v[i] * sc;
        }
        out_row(y, rowoff(y, r, n), n, out);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ quant_act_fp8 */
/* Conventions shared with libr4d's kernel:
 *
 *   448        E4M3's largest finite value. scale = amax / 448 is a dequant multiplier, the same
 *              sense as a checkpoint's `weight_scale_inv`, so the GEMM multiplies the two scales.
 *   scale 1    for an all-zero group (amax/448 = 0 has no reciprocal), because the GEMM
 *              multiplies the block scales and 0 would erase the weight's.
 *   the clamp  amax * (1/scale) can land an ulp above 448 after rounding the reciprocal, which
 *              would encode as NaN. */
AVX_KERNEL(quant_act_fp8) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;

    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        float* out = scratch_f32_b((size_t)n);
        const float* v = in_row(x, rowoff(x, r, n), n, buf);
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
            float sc;
            fp8_group(v, out, k0, k1, &sc);
            stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
        }
        out_row(q, rowoff(q, r, n), n, out);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- quant_cols_fp8 */
/* libref's ref_quant_cols_fp8: one scale per column over all rows (amax * (1/448), 1 for an
 * all-zero column), codes clamp(x * (1/scale), +-448) to nearest even. Two passes over the rows,
 * each row's columns at once: the column maxima, then the codes. */
AVX_KERNEL(quant_cols_fp8) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(y) < rows * n || numel(s) < n) return RAD_E_SHAPE;
    std::vector<float> amax((size_t)n, 0.0f), inv((size_t)n);
    {
        float* buf = scratch_f32((size_t)n);
        for (int64_t r = 0; r < rows; ++r) {
            const float* v = in_row(x, rowoff(x, r, n), n, buf);
            for (int64_t c = 0; c < n; ++c)
                amax[(size_t)c] = std::max(amax[(size_t)c], std::fabs(v[c]));
        }
    }
    for (int64_t c = 0; c < n; ++c) {
        const float sc = amax[(size_t)c] > 0.0f ? amax[(size_t)c] * (1.0f / 448.0f) : 1.0f;
        inv[(size_t)c] = 1.0f / sc;
        stt(s, c * laststride(s), sc);
    }
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        float* out = scratch_f32_b((size_t)n);
        const float* v = in_row(x, rowoff(x, r, n), n, buf);
        for (int64_t c = 0; c < n; ++c)
            out[c] = std::min(std::max(v[c] * inv[(size_t)c], -448.0f), 448.0f);
        out_row(y, rowoff(y, r, n), n, out);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- rmsnorm_quant_fp8 */
/* RMS norm, optional residual add, and quant_act_fp8 in one pass.
 *
 * With `residual`, the sum is written back narrowed to that tensor's dtype and the whole norm,
 * variance included, is taken over the narrowed sum: the `add` + `rmsnorm` pair, unlike
 * rmsnorm_add, which takes its variance from the unrounded sum. The normalised row is narrowed to
 * bf16 before the codes are taken (avx_had.h). */
AVX_KERNEL(rmsnorm_quant_fp8) {
    if (a->n_t < 5) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[2], *q = &a->t[3], *s = &a->t[4];
    const RadTensor* res = rad_arg_in(a, 1);
    const RadTensor* ob  = rad_arg_in(a, 5);
    if (!x->data || !w->data || !q->data || !s->data) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0) return RAD_E_SHAPE;
    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows * ngrp || numel(w) < n)
        return RAD_E_SHAPE;
    if (res && numel(res) < rows * n) return RAD_E_SHAPE;
    if (ob && numel(ob) < rows * n) return RAD_E_SHAPE;

    /* Gain with `wadd` folded in, widened once on the calling thread (as in avx_norm.cpp). */
    float* gain = (float*)scratch_raw((size_t)n * sizeof(float));
    {
        const int64_t ws = laststride(w);
        if (ws == 1) row_to_f32(w->dtype, w->data, gain, n);
        else for (int64_t i = 0; i < n; ++i) gain[i] = ldt(w, i * ws);
        if (wadd != 0.0f) for (int64_t i = 0; i < n; ++i) gain[i] += wadd;
    }

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* src = scratch_f32((size_t)n);
        float* buf = scratch_f32_b((size_t)n);
        float* tmp = scratch_f32_c((size_t)n);

        if (res) {
            const float* xv = in_row(x, rowoff(x, r, n), n, src);
            const float* rv = in_row(res, rowoff(res, r, n), n, tmp);
            for (int64_t i = 0; i < n; ++i) buf[i] = xv[i] + rv[i];
            out_row(res, rowoff(res, r, n), n, buf);
            if (res->dtype != RAD_F32) {
                row_from_f32(res->dtype, buf, tmp, n);
                row_to_f32(res->dtype, tmp, buf, n);
            }
            std::memcpy(src, buf, (size_t)n * sizeof(float));
        } else {
            const float* xv = in_row(x, rowoff(x, r, n), n, src);
            if (xv != src) std::memcpy(src, xv, (size_t)n * sizeof(float));
        }

        const float sc = 1.0f / std::sqrt(row_sumsq(src, n) / (float)n + eps);
        const vf vsc = vf_set1(sc);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N)
            vf_storeu(buf + i, vf_mul(vf_mul(vf_loadu(src + i), vsc), vf_loadu(gain + i)));
        for (; i < n; ++i) buf[i] = src[i] * sc * gain[i];
        narrow_bf16(buf, n, tmp);
        fp8_emit(buf, src, n, g, q, s, ob, r);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- gated_quant_fp8 */
/* act(gate) * up over a packed [M, 2n] gate_up plane (gate first), then quant_act_fp8 over the
 * bf16-narrowed product. `act` is silu or sigmoid (`mode` is libr4d's integer form); default
 * silu. */
AVX_KERNEL(gated_quant_fp8) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *gu = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const RadTensor* ob = rad_arg_in(a, 3);
    const int64_t n = p_int(a, "n", 0);
    static const char* kActs[] = { "silu", "sigmoid" };
    const int act = p_enum(a, "act", kActs, 2, (int)p_int(a, "mode", 0));
    if (n <= 0) return RAD_E_SHAPE;
    if (act != 0 && act != 1) return RAD_E_INVAL;
    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    const int64_t rows = numel(gu) / (2 * n);
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows * ngrp) return RAD_E_SHAPE;
    if (ob && numel(ob) < rows * n) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* gbuf = scratch_f32((size_t)(2 * n));
        float* out  = scratch_f32_b((size_t)n);
        float* tmp  = scratch_f32_c((size_t)n);
        const float* gv = in_row(gu, rowoff(gu, r, 2 * n), 2 * n, gbuf);
        gate_bf16(gv, gv + n, out, n, act, tmp);
        fp8_emit(out, tmp, n, g, q, s, ob, r);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- had_quant_act_fp8 */
/* quant_act_fp8 with a `group`-wide Hadamard before the absmax, which shrinks the maximum the grid
 * is scaled by.
 *
 * Unlike had_quant_act_i8, 1/sqrt(g) is not applied here. <Hw, Ha> = g <w, a>, and the whole g is
 * divided out of the weight scale instead, where dividing a bf16 by a power of two is exact.
 *
 * had_quant_act_i8g is the same at int8 codes (`top` 127), for the i8a8h weight. */
static int had_quant_q8(const RadArgs* a, float top) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = p_int(a, "group", 0);
    /* n % g != 0 is refused: a block straddling the row end would mix in padding, unlike any
     * rotated weight block. */
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (g > AVX_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rows = numel(x) / n;
    const int64_t ngrp = n / g;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        float* out = scratch_f32_b((size_t)n);
        const float* v = in_row(x, rowoff(x, r, n), n, buf);
        if (v != buf) std::memcpy(buf, v, (size_t)n * sizeof(float));
        fwht_row(buf, n, g);
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            float sc;
            fp8_group(buf, out, gi * g, gi * g + g, &sc, top);
            stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
        }
        out_row(q, rowoff(q, r, n), n, out);
    }
    return RAD_OK;
}

AVX_KERNEL(had_quant_act_fp8) {
    return had_quant_q8(a, 448.0f);
}

AVX_KERNEL(had_quant_act_i8g) {
    return had_quant_q8(a, 127.0f);
}

/* ------------------------------------------------------------------ had_rotate */
/* The normalised rotation alone, in place (libref/ref_quant.cpp: ref_had_rotate): fwht_row's
 * butterflies, then one f32 multiply by 1/sqrt(group) and x's rounding. The factor is formed as
 * libref's had_norm forms it (a power of two, times 1/sqrt(2) rounded to f32 when log2(g) is odd),
 * not as 1/sqrt((float)g), so the bytes are libref's and libr4d's. */
AVX_KERNEL(had_rotate) {
    if (!rad_args_have(a, 1)) return RAD_E_INVAL;
    const RadTensor* x = &a->t[0];
    const int64_t n = p_int(a, "n", 0), g = p_int(a, "group", 0);
    if (n <= 0 || g < 2 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (g > AVX_HAD_MAX) return RAD_E_UNSUPPORTED;
    if (x->dtype != RAD_BF16 && x->dtype != RAD_F16 && x->dtype != RAD_F32) return RAD_E_DTYPE;
    const int64_t nel = numel(x), rows = nel / n;
    if (nel <= 0 || nel % n != 0) return RAD_E_SHAPE;
    int k = 0;
    while (((int64_t)1 << k) < g) ++k;
    const float half = std::ldexp(1.0f, -(k / 2));
    const float s = (k & 1) ? half * 0.70710678118654752f : half;

    AVX_PARALLEL_FOR_IF(nel >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        /* in_row may hand back x itself (dense f32); rotate a copy, then store it over x. */
        const float* v = in_row(x, rowoff(x, r, n), n, buf);
        if (v != buf) std::memcpy(buf, v, (size_t)n * sizeof(float));
        fwht_row(buf, n, g);
        scale_row(buf, n, s);
        out_row(x, rowoff(x, r, n), n, buf);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- givens_rotate */
/* ParoQuant's activation side for a 16-bit GEMM: each row is divided by its channel scales and
 * rotated by `stages` rounds of disjoint Givens pairs, once per part, parts side by side in y.
 * libref's arithmetic and order (ref_quant.cpp): the divide, then a' = a cos t + b sin t,
 * b' = b cos t - a sin t in f32, stage by stage. The pair walk is a gather/scatter and stays
 * scalar; the cos and sin of each angle are computed once per call. */
AVX_KERNEL(givens_rotate) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *pr = &a->t[1], *an = &a->t[2], *sc = &a->t[3], *y = &a->t[4];
    const int64_t n = p_int(a, "n", 0), S = p_int(a, "stages", 0), P = p_int(a, "parts", 1);
    const int64_t g = p_int(a, "group", 0);
    if (n <= 0 || S <= 0 || P <= 0 || (n & 1)) return RAD_E_SHAPE;
    if (g <= 0 || g > n || n % g != 0) return RAD_E_SHAPE;
    const int64_t NP = P * S * (n / 2);
    if (numel(pr) < P * S * n || numel(an) < NP || numel(sc) < P * n) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(y) < rows * P * n) return RAD_E_SHAPE;

    int32_t* idx = (int32_t*)scratch_raw((size_t)(P * S * n) * sizeof(int32_t) +
                                         (size_t)(2 * NP + P * n) * sizeof(float) + 64);
    float* cs = (float*)(idx + P * S * n);
    float* sn = cs + NP;
    float* rs = sn + NP;
    for (int64_t i = 0; i < P * S * n; ++i) {
        const int64_t c = ldi(pr, i);
        if (c < 0 || c >= n) return RAD_E_INVAL;
        idx[i] = (int32_t)c;
    }
    for (int64_t j = 0; j < NP; ++j) {
        const float th = ldt(an, j);
        cs[j] = sc_cos(th);
        sn[j] = sc_sin(th);
    }
    for (int64_t k = 0; k < P * n; ++k) rs[k] = ldt(sc, k);

    AVX_PARALLEL_FOR_IF(rows * P * n * S >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        float* out = scratch_f32_b((size_t)(P * n));
        const float* v = in_row(x, rowoff(x, r, n), n, buf);
        for (int64_t p = 0; p < P; ++p) {
            float* w = out + p * n;
            for (int64_t k = 0; k < n; ++k) w[k] = v[k] / rs[p * n + k];
            for (int64_t st = 0; st < S; ++st) {
                const int32_t* ip = idx + (p * S + st) * n;
                const int64_t a0 = (p * S + st) * (n / 2);
                for (int64_t j = 0; j < n / 2; ++j) {
                    const float c = cs[a0 + j], s1 = sn[a0 + j];
                    const float va = w[ip[2 * j]], vb = w[ip[2 * j + 1]];
                    w[ip[2 * j]] = va * c + vb * s1;
                    w[ip[2 * j + 1]] = vb * c - va * s1;
                }
            }
        }
        out_row(y, rowoff(y, r, P * n), P * n, out);
    }
    return RAD_OK;
}

/* --------------------------------------------------------- givens_dense_quant */
/* Quantising form: the rotation as one dense 128 x 128 f16 matrix per (part, group) (`tdense`,
 * which rad-convert composes from the pairs and angles as the weight's "t.dense" table), then the
 * block-scaled quantiser named by `out`: "fp8" E4M3 with scale amax / 448, or "i8" with scale
 * amax / 127, rounded to nearest; scale 1 for an all-zero group, as libref. The matrices are widened
 * to f32 once per call. Each group is a matrix-vector product summed in f32 (libref uses f64), so a
 * code on a rounding edge can differ from libref's. */
AVX_KERNEL(givens_dense_quant) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *tm = &a->t[1], *sc = &a->t[2], *q = &a->t[3], *s = &a->t[4];
    const int64_t n = p_int(a, "n", 0), P = p_int(a, "parts", 1), g = p_int(a, "group", 0);
    if (n <= 0 || P <= 0 || g != 128 || n % g != 0) return RAD_E_SHAPE;
    const bool i8 = q->dtype == RAD_I8;
    if (!i8 && q->dtype != RAD_F8E4M3) return RAD_E_DTYPE;
    const char* outs = p_str(a, "out", "");
    if (std::strcmp(outs, i8 ? "i8" : "fp8") != 0) return RAD_E_DTYPE;   // `out` names q's dtype
    const int64_t rows = numel(x) / n, ngrp = n / g;
    if (rows <= 0 || numel(tm) < P * n * g || numel(sc) < P * n || numel(q) < rows * P * n ||
        numel(s) < rows * P * ngrp) return RAD_E_SHAPE;
    const float qmax = i8 ? 127.0f : 448.0f;

    float* T = (float*)scratch_raw((size_t)(P * n * g + P * n) * sizeof(float) + 64);
    float* rs = T + P * n * g;
    for (int64_t i = 0; i < P * n; ++i) {
        row_to_f32(tm->dtype, byte_at(tm->data, tm->dtype, rowoff(tm, i, g)), T + i * g, g);
        rs[i] = ldt(sc, i);
    }

    AVX_PARALLEL_FOR_IF(rows * P * n * g >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        float* out = scratch_f32_b((size_t)(P * n));
        float* in = scratch_f32_c((size_t)g);
        const float* v = in_row(x, rowoff(x, r, n), n, buf);
        for (int64_t p = 0; p < P; ++p)
            for (int64_t gi = 0; gi < ngrp; ++gi) {
                for (int64_t i = 0; i < g; ++i) in[i] = v[gi * g + i] / rs[p * n + gi * g + i];
                float* yv = out + p * n + gi * g;
                const float* Tg = T + (p * n + gi * g) * g;
                for (int64_t j = 0; j < g; ++j) {
                    vf acc = vf_zero();
                    for (int64_t i = 0; i < g; i += VF_N)
                        acc = vf_fma(vf_loadu(Tg + j * g + i), vf_loadu(in + i), acc);
                    yv[j] = vf_hsum(acc);
                }
                const float amax = row_amax(yv, g);
                const float scl = amax > 0.0f ? amax * (1.0f / qmax) : 1.0f;
                const float inv = 1.0f / scl;
                for (int64_t j = 0; j < g; ++j) {
                    float t = yv[j] * inv;
                    t = t > qmax ? qmax : (t < -qmax ? -qmax : t);
                    yv[j] = i8 ? std::nearbyint(t) : t;
                }
                stt(s, rowoff(s, r, P * ngrp) + (p * ngrp + gi) * laststride(s), scl);
            }
        out_row(q, rowoff(q, r, P * n), P * n, out);
    }
    return RAD_OK;
}

/* ---------------------------------------------------------------- the Gram accumulator
 *
 * h[k1][k2] += sum over rows of x[m][k1] * x[m][k2], where x is the dequantised E4M3 input, or a
 * bf16 input as is when no scale is passed: the second moment GPTQ's error feedback needs, over the
 * operand the expert GEMM contracts.
 *
 * `h` accumulates across launches: the caller zeroes it once and runs a calibration corpus through.
 *
 * Accumulation is f64 here (f32 on the device), as in libref: the result is consumed offline by
 * the packer, and the kbench tolerance covers the gap. The f64 inner reduction stays scalar; only
 * the dequantisation is vectorised, once per row into a scratch plane rather than in the k2 loop. */
AVX_KERNEL(gram_accum) {
    /* `a_scale` is optional; without it `a` is read as is (the bf16 form). */
    if (a->n_t < 3) return RAD_E_INVAL;
    const RadTensor *x = rad_arg_in(a, 0), *s = rad_arg_in(a, 1), *h = rad_arg_in(a, 2);
    if (!x || !h) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = s ? p_int(a, "group", 0) : n;
    if (n <= 0 || g <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    const int64_t ngrp = (n + g - 1) / g;
    if (rows <= 0 || (s && numel(s) < rows * ngrp)) return RAD_E_SHAPE;
    if (h->rank != 2 || h->shape[0] != n || h->shape[1] != n) return RAD_E_SHAPE;

    float* xs = (float*)scratch_raw((size_t)rows * (size_t)n * sizeof(float) + 64);
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        const float* v = in_row(x, rowoff(x, r, n), n, buf);
        float* dst = xs + r * n;
        const int64_t sb = s ? rowoff(s, r, ngrp) : 0, sst = s ? laststride(s) : 0;
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
            const float sc = s ? ldt(s, sb + gi * sst) : 1.0f;
            const vf vs = vf_set1(sc);
            int64_t k = k0;
            for (; k + VF_N <= k1; k += VF_N) vf_storeu(dst + k, vf_mul(vf_loadu(v + k), vs));
            for (; k < k1; ++k) dst[k] = v[k] * sc;
        }
    }

    /* The upper triangle, each sum added to both (k1, k2) and (k2, k1): the products commute, so
     * the two sums are the same bits. Rows of the triangle shrink, hence the dynamic schedule.
     * `upper` 1 leaves (k2, k1) as it was. */
    const bool upper = p_int(a, "upper", 0) != 0;
    const int64_t hst = laststride(h);
#ifdef _OPENMP
    AVX_PRAGMA(omp parallel for schedule(dynamic, 8) if(n * n >= 1024 || avx_par_forced())
               num_threads(avx_team_size()))
#endif
    for (int64_t k1 = 0; k1 < n; ++k1) {
        for (int64_t k2 = k1; k2 < n; ++k2) {
            double acc = 0.0;
            for (int64_t r = 0; r < rows; ++r)
                acc += (double)xs[r * n + k1] * (double)xs[r * n + k2];
            const int64_t o12 = rowoff(h, k1, n) + k2 * hst, o21 = rowoff(h, k2, n) + k1 * hst;
            stt(h, o12, (float)((double)ldt(h, o12) + acc));
            if (k2 != k1 && !upper) stt(h, o21, (float)((double)ldt(h, o21) + acc));
        }
    }
    return RAD_OK;
}
