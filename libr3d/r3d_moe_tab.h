/* r3d_moe_tab.h -- the grouped GEMMs' contract operands: weight tables, `a_order` and the classes.
 *
 * `moe_gemm` and `moe_gemm_q` take their experts as weight tables (RAD_OPD_WTAB, docs/OPS.md):
 * a `[entries, rows, cols]` tensor whose data is a device array of one pointer per entry when
 * stride[0] is 0 (each expert is its own movement unit, so the core hands a pointer each), or the
 * base of a stacked plane with entries stride[0] elements apart. Both are read through
 * r3d_tab_at, resolved on the device: the table lives in device memory and the mover may rewrite
 * it between launches.
 *
 * `moe_gemm_q` holds the even experts in `w`/`w_scale` and the odd in `w_odd`/`w_odd_scale`,
 * expert e at entry e/2 of class e%2. The contract lets the odd class be narrower
 * (`N_odd`/`K_odd`, a rank holding an uneven slice of every expert); these kernels serve equal
 * classes and refuse the rest by name. With equal classes `parts` maps output column n to weight
 * row n, so it needs no handling beyond the divisibility check.
 */
#pragma once

#include "r3d_args.h"

#include <cstring>

/* One class's table as the kernels read it: the table, the bytes between stacked entries (0 for
 * a pointer array) and the elements between rows of one entry. */
struct R3dTab {
    const void* const* tab;
    unsigned long long s0;
    int ld;
};

/* Both classes' codes and scales, passed to a kernel by value. Class 1 repeats class 0 when the
 * call has a single expert and so no odd entries. */
struct R3dMoeTabs {
    R3dTab w[2];
    R3dTab s[2];
};

#if defined(__HIPCC__)
__device__ __forceinline__ const uint8_t* r3d_tab_at(const R3dTab& t, int i) {
    if (t.s0 == 0) return (const uint8_t*)t.tab[i];
    return (const uint8_t*)t.tab + (size_t)i * t.s0;
}
#endif

/* A weight-table operand of `entries` entries, each at least `rows` x `cols`, with the dtype the
 * row reads; `bits` is the element width, so a 4-bit plane's stacked stride keeps its half-byte. */
static inline int r3d_tab_of(const RadTensor* t, uint32_t dtype, int bits, long long entries,
                             long long rows, long long cols, R3dTab* out) {
    if (!t || !t->data || t->rank != 3) return R3D_NO(RAD_E_SHAPE);
    if (t->dtype != dtype) return R3D_NO(RAD_E_DTYPE);
    if (t->shape[0] < entries || t->shape[1] < rows || t->shape[2] < cols)
        return R3D_NO(RAD_E_SHAPE);
    if (t->stride[2] != 1 || t->stride[1] < cols) return R3D_NO(RAD_E_STRIDE);
    out->tab = (const void* const*)t->data;
    out->s0  = (unsigned long long)((t->stride[0] * (long long)bits) / 8);
    out->ld  = (int)t->stride[1];
    return RAD_OK;
}

/* The operands and parameters both grouped GEMMs share, checked against the contract. */
struct R3dMoeCall {
    const RadTensor *a, *a_scale, *sorted, *off, *y;
    const RadTensor *w, *w_scale, *w_odd, *w_odd_scale;   /* w_scale and the odd class: _q only */
    long long T, N, K, n_expert, top_k;
    int a_sorted;
};

/* `quant`: moe_gemm_q's nine operands (a, a_scale?, w, w_scale, sorted_tok, expert_offset, w_odd,
 * w_odd_scale -> y), else moe_gemm's five (a, w, sorted_tok, expert_offset -> y). */
static inline int r3d_moe_call(const RadArgs* a, bool quant, R3dMoeCall* c) {
    if (!a || a->n_t < (quant ? 9 : 5)) return R3D_NO(RAD_E_INVAL);
    *c = R3dMoeCall{};
    c->a       = r3d_opt(a, 0);
    c->a_scale = quant ? r3d_opt(a, 1) : nullptr;
    c->w       = r3d_opt(a, quant ? 2 : 1);
    c->w_scale = quant ? r3d_opt(a, 3) : nullptr;
    c->sorted  = r3d_opt(a, quant ? 4 : 2);
    c->off     = r3d_opt(a, quant ? 5 : 3);
    c->w_odd       = quant ? r3d_opt(a, 6) : nullptr;
    c->w_odd_scale = quant ? r3d_opt(a, 7) : nullptr;
    c->y       = r3d_opt(a, quant ? 8 : 4);
    if (!c->a || !c->w || !c->sorted || !c->off || !r3d_wp(c->y)) return R3D_NO(RAD_E_INVAL);
    if (quant && !c->w_scale) return R3D_NO(RAD_E_INVAL);

    /* "token" (the default, gate_up): sorted row i reads a[sorted_tok[i] / top_k]. "sorted" (the
     * down projection): a already has one row per (token, slot) in the scatter's order. */
    const char* ord = r3d_gets(a, "a_order");
    c->a_sorted = ord && std::strcmp(ord, "sorted") == 0;
    if (ord && !c->a_sorted && std::strcmp(ord, "token") != 0) return R3D_NO(RAD_E_UNSUPPORTED);

    c->N = r3d_geti_or(a, "N", c->w->rank >= 3 ? c->w->shape[1] : 0);
    c->K = r3d_geti_or(a, "K", c->w->rank >= 3 ? c->w->shape[2] : 0);
    c->n_expert = r3d_geti_or(a, "n_expert", 0);
    c->top_k = r3d_geti_or(a, "top_k", 0);
    if (c->N <= 0 || c->K <= 0 || c->n_expert <= 0 || c->top_k <= 0) return R3D_NO(RAD_E_SHAPE);
    if (quant) {
        if (r3d_geti_or(a, "N_odd", c->N) != c->N || r3d_geti_or(a, "K_odd", c->K) != c->K)
            return R3D_NO(RAD_E_UNSUPPORTED);
        const long long parts = r3d_geti_or(a, "parts", 1);
        if (parts < 1 || c->N % parts) return R3D_NO(RAD_E_SHAPE);
        if (c->n_expert >= 2 && (!c->w_odd || !c->w_odd_scale)) return R3D_NO(RAD_E_INVAL);
    }

    /* One output row per (token, slot), the rows the sort placed. */
    long long yr = 0, yc = 0;
    if (r3d_rows_of(c->y, &yr, &yc) != RAD_OK || yc < c->N) return R3D_NO(RAD_E_SHAPE);
    c->T = yr;
    if (r3d_numel(c->sorted) < c->T || r3d_numel(c->off) < c->n_expert + 1)
        return R3D_NO(RAD_E_SHAPE);
    long long ar = 0, ak = 0;
    if (r3d_rows_of(c->a, &ar, &ak) != RAD_OK || ak < c->K) return R3D_NO(RAD_E_SHAPE);
    if (ar < (c->a_sorted ? c->T : c->T / c->top_k)) return R3D_NO(RAD_E_SHAPE);
    return RAD_OK;
}

/* Both classes' tables of a moe_gemm_q call: codes at `wdt`/`wbits`, scales at `sdt` (or u8 for
 * an E8M0 plane typed raw) with `srows` x `scols` an entry. */
static inline int r3d_moe_q_tabs(const R3dMoeCall& c, uint32_t wdt, int wbits, uint32_t sdt,
                                 int sbits, long long srows, long long scols, R3dMoeTabs* t) {
    const long long n_even = (c.n_expert + 1) / 2, n_odd = c.n_expert / 2;
    auto sdt_of = [&](const RadTensor* x) {
        return (sdt == RAD_E8M0 && x && x->dtype == RAD_U8) ? (uint32_t)RAD_U8 : sdt;
    };
    int rc = r3d_tab_of(c.w, wdt, wbits, n_even, c.N, c.K, &t->w[0]);
    if (rc == RAD_OK) rc = r3d_tab_of(c.w_scale, sdt_of(c.w_scale), sbits, n_even, srows, scols,
                                      &t->s[0]);
    if (rc != RAD_OK) return rc;
    if (n_odd == 0) { t->w[1] = t->w[0]; t->s[1] = t->s[0]; return RAD_OK; }
    rc = r3d_tab_of(c.w_odd, wdt, wbits, n_odd, c.N, c.K, &t->w[1]);
    if (rc == RAD_OK) rc = r3d_tab_of(c.w_odd_scale, sdt_of(c.w_odd_scale), sbits, n_odd, srows,
                                      scols, &t->s[1]);
    return rc;
}
