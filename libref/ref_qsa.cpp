/* ref_qsa.cpp -- the QSA indexer's score, selection, tail store and work list, and DFlash2's
 * grouped convolution (docs/OPS.md, "Sparse attention (QSA)" and "Draft heads").
 *
 * None of these is a fusion: an architecture that issues them has no chain to fall back to, so
 * without a row here the host backend could not run the model and a device kernel would have no
 * oracle. Each follows its libr4d kernel's contract (r4d_qsa_*_bf16.hip,
 * r4d_dflash_conv_body.h), including which elements it leaves unwritten and the order of every
 * output, with f32 sums in ascending order and no FMA (ref_common.h).
 *
 * `cu` holds absolute row indices: sequence s owns rows [cu[s], cu[s+1]) of every per-token
 * operand, and cu[0] need not be 0.
 */
#include "ref_common.h"
#include "ref_gemm.h"   /* row_stride */
#include "ref_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace ref;

namespace {

/* The sequence whose cu span holds row `m`, or `none`. */
int64_t seq_holding(const RadTensor* cu, int64_t n_seq, int64_t m, int64_t none) {
    for (int64_t s = 0; s < n_seq; ++s)
        if (m >= cu_at(cu, s) && m < cu_at(cu, s + 1)) return s;
    return none;
}

/* Offset of entry `i` of a rank-1 operand, or of column 0 of row `i` of a rank-2 one (a state
 * index row's committed slot). */
int64_t col0(const RadTensor* t, int64_t i) {
    return t->rank >= 2 ? i * t->stride[0] : offlin(t, i);
}

int64_t ldi(const RadTensor* t, int64_t off) { return ld_int(t->dtype, t->data, off); }

}  /* namespace */

/* ------------------------------------------------------------------ qsa_score */
/* score[m][b] = (sum over heads h of relu(q[m][h] . bk[bt[seq(m)][b]])) / sqrt(n), for b below
 * min(nc[seq], the table's width, the score plane's width); a negative page id scores -inf. Each
 * dot is an f32 sum over the n lanes in order, the relus are summed over the heads in order.
 * Columns past the bound, and rows no sequence of `cu` holds, are not written.
 *
 * `key_dtype` i8 reads qsa_block_key's int8 pages (a byte-wide store, rows of >= n + 2 bytes): the
 * scale is folded out of the dot, relu(scale * (q . codes)), so each product q * code is exact in
 * f32 (a bf16 query has 8 significant bits and a code 7) and the scale costs one multiply a
 * block and head. */
int ref_qsa_score(const RadArgs* a, RadStream) {
    if (!have(a, 6)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *bk = &a->t[1], *bt = &a->t[2], *cu = &a->t[3];
    const RadTensor *nc = &a->t[4], *o = &a->t[5];
    const int64_t n = p_int(a, "n", 0), heads = p_int(a, "heads", 0);
    if (n <= 0 || heads <= 0) return RAD_E_SHAPE;
    if (bt->rank != 2 || o->rank != 2 || bk->rank < 2) return RAD_E_SHAPE;
    const int64_t M = o->shape[0], nb = std::min(bt->shape[1], o->shape[1]);
    const int64_t n_seq = numel(cu) - 1, pages = bk->shape[0];
    if (M <= 0 || nb <= 0 || n_seq <= 0 || pages <= 0) return RAD_E_SHAPE;
    if (bt->shape[0] < n_seq || numel(nc) < n_seq || rows_of(q, heads * n) < M) return RAD_E_SHAPE;
    const char* kd = p_str(a, "key_dtype", nullptr);
    if (kd && std::strcmp(kd, "i8")) return RAD_E_UNSUPPORTED;
    const bool i8 = kd != nullptr;
    if (i8 ? numel(bk) / pages < n + 2 || rad_dtype_bits(bk->dtype) != 8 : numel(bk) / pages != n)
        return RAD_E_SHAPE;
    if (!rows_dense(bk)) return RAD_E_STRIDE;
    const int64_t bt_rs = bt->stride[0], bt_cs = bt->stride[1];
    /* Every page id the loop will read must name a page of `bk`. */
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t nbs = std::min(ldi(nc, offlin(nc, s)), nb);
        for (int64_t b = 0; b < nbs; ++b)
            if (ldi(bt, s * bt_rs + b * bt_cs) >= pages) return RAD_E_INVAL;
    }
    const int64_t q_rs = row_stride(q, heads * n), q_cs = laststride(q);
    const int64_t pg_rs = bk->stride[0], o_rs = o->stride[0], o_cs = o->stride[1];
    const float rn = std::sqrt((float)n);
    REF_PAR_FOR(M * nb * heads * n)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s = seq_holding(cu, n_seq, m, -1);
        if (s < 0) continue;
        const int64_t nbs = std::min(ldi(nc, offlin(nc, s)), nb);
        for (int64_t b = 0; b < nbs; ++b) {
            const int64_t pg = ldi(bt, s * bt_rs + b * bt_cs);
            if (pg < 0) { stt(o, m * o_rs + b * o_cs, -INFINITY); continue; }
            float sum = 0.0f;
            if (i8) {
                const uint8_t* row = (const uint8_t*)bk->data + pg * pg_rs;
                uint16_t s16;
                std::memcpy(&s16, row + n, 2);
                const float sc = f16_to_f32(s16);
                for (int64_t h = 0; h < heads; ++h) {
                    float d = 0.0f;
                    for (int64_t i = 0; i < n; ++i)
                        d += ldt(q, m * q_rs + (h * n + i) * q_cs) * (float)(int8_t)row[i];
                    sum += std::fmax(d * sc, 0.0f);
                }
                stt(o, m * o_rs + b * o_cs, sum / rn);
                continue;
            }
            for (int64_t h = 0; h < heads; ++h) {
                float d = 0.0f;
                for (int64_t i = 0; i < n; ++i)
                    d += ldt(q, m * q_rs + (h * n + i) * q_cs) * ldt(bk, pg * pg_rs + i);
                sum += std::fmax(d, 0.0f);
            }
            stt(o, m * o_rs + b * o_cs, sum / rn);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ qsa_select */
/* The page-id table the sparse attention reads (docs/OPS.md): over the query's complete blocks
 * n = min(nc[seq], (pos + 1) / ratio, width), the live ones (page id >= 0, score > -1e30) ranked
 * by max(score, 0); with `signed` 1 the live ones are those with a finite score, ranked by it (so
 * -0 ties +0). At most `topk` are taken: all of them when that many or fewer are live, else
 * every block above the topk-th largest value in block order, then the blocks equal to it in block
 * order until topk are taken. The partial page bt[n] follows when (pos + 1) % ratio != 0 and
 * n < width, and -1 pads the row to topk + 1. nsel counts the entries, seqused is
 * (nsel - 1) * ratio + pos % ratio + 1 (0 for an empty table). A row no sequence holds reads
 * sequence 0, as libr4d's does. */
int ref_qsa_select(const RadArgs* a, RadStream) {
    if (!have(a, 8)) return RAD_E_INVAL;
    const RadTensor *sc = &a->t[0], *bt = &a->t[1], *nc = &a->t[2], *po = &a->t[3];
    const RadTensor *cu = &a->t[4], *se = &a->t[5], *ns = &a->t[6], *su = &a->t[7];
    const int64_t topk = p_int(a, "topk", 0), ratio = p_int(a, "ratio", 0);
    const int64_t sgn = p_int(a, "signed", 0);
    if (topk <= 0 || ratio <= 0) return RAD_E_SHAPE;
    if (sgn != 0 && sgn != 1) return RAD_E_INVAL;
    if (sc->rank != 2 || bt->rank != 2 || se->rank != 2) return RAD_E_SHAPE;
    const int64_t M = sc->shape[0], nb = std::min(bt->shape[1], sc->shape[1]);
    const int64_t n_seq = numel(cu) - 1;
    if (M <= 0 || nb <= 0 || n_seq <= 0) return RAD_E_SHAPE;
    if (bt->shape[0] < n_seq || numel(nc) < n_seq) return RAD_E_SHAPE;
    if (se->shape[0] < M || se->shape[1] < topk + 1) return RAD_E_SHAPE;
    if (numel(ns) < M || numel(po) < M || numel(su) < M) return RAD_E_SHAPE;
    for (int64_t s = 0; s < n_seq; ++s)
        if (ldi(nc, offlin(nc, s)) < 0) return RAD_E_INVAL;
    for (int64_t m = 0; m < M; ++m)
        if (ldi(po, offlin(po, m)) < 0) return RAD_E_INVAL;
    const int64_t s_rs = sc->stride[0], s_cs = sc->stride[1];
    const int64_t bt_rs = bt->stride[0], bt_cs = bt->stride[1];
    const int64_t o_rs = se->stride[0], o_cs = se->stride[1];
    REF_PAR_FOR(M * nb * 4)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s = seq_holding(cu, n_seq, m, 0);
        const int64_t p = ldi(po, offlin(po, m));
        const int64_t n = std::min(std::min(ldi(nc, offlin(nc, s)), nb), (p + 1) / ratio);
        std::vector<float> val((size_t)n);
        std::vector<int64_t> live;
        for (int64_t b = 0; b < n; ++b) {
            const float v = ldt(sc, m * s_rs + b * s_cs);
            if (ldi(bt, s * bt_rs + b * bt_cs) < 0) continue;
            if (sgn ? !std::isfinite(v) : !(v > -1e30f)) continue;
            val[(size_t)b] = sgn ? v : v > 0.0f ? v : 0.0f;
            live.push_back(b);
        }
        std::vector<int64_t> take;
        if ((int64_t)live.size() <= topk) {
            take = live;
        } else {
            std::vector<float> ord;
            for (int64_t b : live) ord.push_back(val[(size_t)b]);
            std::nth_element(ord.begin(), ord.begin() + (topk - 1), ord.end(), std::greater<float>());
            const float thr = ord[(size_t)(topk - 1)];
            for (int64_t b : live)
                if (val[(size_t)b] > thr) take.push_back(b);
            for (int64_t b : live)
                if ((int64_t)take.size() < topk && val[(size_t)b] == thr) take.push_back(b);
        }
        int64_t out = 0;
        for (int64_t b : take) st_int(se->dtype, se->data, m * o_rs + (out++) * o_cs,
                                      ldi(bt, s * bt_rs + b * bt_cs));
        if (n < nb && (p + 1) % ratio != 0) {
            const int64_t part = ldi(bt, s * bt_rs + n * bt_cs);
            if (part >= 0) st_int(se->dtype, se->data, m * o_rs + (out++) * o_cs, part);
        }
        for (int64_t i = out; i <= topk; ++i) st_int(se->dtype, se->data, m * o_rs + i * o_cs, -1);
        st_int(ns->dtype, ns->data, offlin(ns, m), out);
        st_int(su->dtype, su->data, offlin(su, m), out == 0 ? 0 : (out - 1) * ratio + p % ratio + 1);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ qsa_pool_mask */
/* The hierarchical indexer's restriction (docs/OPS.md, "`qsa_pool_mask`"): in each row m of
 * `score`, every column b in [0, cols) that no entry pool[m][0 .. pool] names is set to -inf, which
 * qsa_select reads as dead in both rankings; a named column keeps its bytes. `pool` holds block
 * indices (the first QSA layer's qsa_select run over an identity table), -1 padded; an entry
 * outside [0, cols) names nothing. Columns past `cols` (the plane's declared width) are not
 * touched. */
int ref_qsa_pool_mask(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *po = &a->t[0], *sc = &a->t[1];
    const int64_t np = p_int(a, "pool", 0);
    if (np <= 0) return RAD_E_SHAPE;
    if (po->rank != 2 || sc->rank != 2) return RAD_E_SHAPE;
    if (sc->dtype != RAD_F32) return RAD_E_DTYPE;
    const int64_t M = sc->shape[0], cols = sc->shape[1];
    if (M <= 0 || cols <= 0 || po->shape[0] < M || po->shape[1] < np + 1) return RAD_E_SHAPE;
    for (int64_t m = 0; m < M; ++m) {
        std::vector<char> keep((size_t)cols, 0);
        for (int64_t j = 0; j <= np; ++j) {
            const int64_t b = ldi(po, m * po->stride[0] + j * po->stride[1]);
            if (b >= 0 && b < cols) keep[(size_t)b] = 1;
        }
        for (int64_t b = 0; b < cols; ++b)
            if (!keep[(size_t)b]) stt(sc, m * sc->stride[0] + b * sc->stride[1], -INFINITY);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ qsa_tail_store */
/* state[sidx[s][0]][0][pos[t] % ring] = k[t] for the last `ring` rows of each sequence, rows in
 * ascending order (a row whose slot is -1, or that no sequence holds, writes nothing). Every other
 * row of `state` keeps its contents. */
int ref_qsa_tail_store(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *k = &a->t[0], *po = &a->t[1], *cu = &a->t[2], *st = &a->t[3];
    const RadTensor *sx = &a->t[4];
    const int64_t n = p_int(a, "n", 0), ratio = p_int(a, "ratio", 0), ring = p_int(a, "ring", 0);
    if (n <= 0 || ratio <= 0 || ring < ratio) return RAD_E_INVAL;
    if (k->rank != 2 || st->rank != 4 || st->shape[1] != 1) return RAD_E_SHAPE;
    const int64_t T = k->shape[0], slots = st->shape[0], n_seq = numel(cu) - 1;
    if (T <= 0 || k->shape[1] != n || st->shape[2] < ring || st->shape[3] != n) return RAD_E_SHAPE;
    if (n_seq <= 0 || numel(po) < T) return RAD_E_SHAPE;
    if ((sx->rank >= 2 ? sx->shape[0] : numel(sx)) < n_seq) return RAD_E_SHAPE;
    /* The rows that write, validated before any write. */
    std::vector<std::pair<int64_t, int64_t>> w;     /* (row, slot) */
    for (int64_t t = 0; t < T; ++t) {
        const int64_t s = seq_holding(cu, n_seq, t, -1);
        if (s < 0 || t < cu_at(cu, s + 1) - ring) continue;
        const int64_t si = ldi(sx, col0(sx, s));
        if (si < 0) continue;
        if (si >= slots || ldi(po, offlin(po, t)) < 0) return RAD_E_INVAL;
        w.push_back({ t, si });
    }
    for (const auto& [t, si] : w) {
        const int64_t row = ldi(po, offlin(po, t)) % ring;
        const int64_t db = si * st->stride[0] + row * st->stride[2];
        for (int64_t d = 0; d < n; ++d)
            stt(st, db + d * st->stride[3], ldt(k, t * k->stride[0] + d * k->stride[1]));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ qsa_page */
/* The pager: for each position (m, j) of `sel` holding a cold entry (<= -2), the `block_bytes` at
 * the start of store slot -2 - entry (this layer's view, docs/OPS.md) are copied into pool block scratch[m][j], and the entry
 * becomes that block id. Resident (>= 0) and empty (-1) entries are left alone, so a row without a
 * cold entry is a no-op. Every position has one writer and the scratch ids of cold positions are
 * pairwise distinct, so no two copies overlap. The device kernel is held to these bytes.
 *
 * Refused before any write (RAD_E_INVAL): a slot outside the store,
 * a scratch id outside the pool, or two cold positions sharing a scratch page.
 *
 * An optional fifth operand, one i64, is the pager's counter: the entries this call rewrote are
 * added to it (not stored), so a caller reads how much the window paged over many launches.
 *
 * With `store_layout` and `pool_layout` (RAD_KVQ_DEMOTE, a tiered layer) the copy expands: the
 * slot holds the fragment's rows in the demote layout and each is decoded and re-encoded into the
 * pool's layout (kv_transcode's codec), so the attention reads the one layout it was declared at. */
int ref_qsa_page(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *sel = &a->t[0], *st = &a->t[1], *pl = &a->t[2], *sc = &a->t[3];
    const int64_t topk = p_int(a, "topk", 0), bb = p_int(a, "block_bytes", 0);
    if (topk <= 0 || bb <= 0) return RAD_E_SHAPE;
    /* `pool` is the layer's cache in its own shape -- [blocks, ...] of the group's dtype, as the
     * core binds a KV operand -- and the pager moves bytes: a block is its leading stride's
     * worth of elements, contiguous. `store` is [slots, slot_bytes] of bytes. */
    if (sel->rank != 2 || sc->rank != 2 || st->rank != 2 || pl->rank < 2) return RAD_E_SHAPE;
    const int64_t ebits = rad_dtype_bits(pl->dtype);
    if (st->dtype != RAD_U8 || ebits <= 0 || ebits % 8) return RAD_E_DTYPE;
    if (st->stride[1] != 1 || pl->stride[pl->rank - 1] != 1) return RAD_E_STRIDE;
    /* A block's elements past the leading dimension are packed, so its bytes from its start are
     * contiguous for `inner` elements; the pitch between blocks may be anything. */
    int64_t inner = 1;
    for (uint32_t d = pl->rank - 1; d >= 1; --d) {
        if (pl->stride[d] != inner) return RAD_E_STRIDE;
        inner *= pl->shape[d];
    }
    const int64_t pitch = pl->stride[0] * (ebits / 8);           /* bytes a block */
    const int64_t M = sel->shape[0], w = topk + 1;
    if (M <= 0 || sel->shape[1] < w || sc->shape[0] < M || sc->shape[1] < w) return RAD_E_SHAPE;
    /* `store` is this layer's view of the store (the plugin folds the layer's offset in): row s
     * is slot s, its first block_bytes are this layer's fragment. */
    if (inner * (ebits / 8) < bb || pitch < bb) return RAD_E_SHAPE;
    /* Expanding (RAD_KVQ_DEMOTE): the store holds the fragment in `store_layout`, the pool takes
     * it in `pool_layout`, row by row (kvx_rows, ref_attn.cpp); `block_bytes` stays the pool's
     * fragment, and the store's is as many rows of its own width. Both words or neither. */
    const char* sl = p_str(a, "store_layout", nullptr);
    const char* plw = p_str(a, "pool_layout", nullptr);
    int xf = -1, xt = -1;
    int64_t hd = 0, rows = 0, sbb = bb;
    if (sl || plw) {
        xf = kvx_layout(sl); xt = kvx_layout(plw); hd = p_int(a, "head_dim", 0);
        if (xf < 0 || xt < 0 || xf == xt) return RAD_E_INVAL;
        if (hd <= 0 || hd % 32 || bb % kvx_row_bytes(xt, hd)) return RAD_E_SHAPE;
        if (hd > 2048) return RAD_E_UNSUPPORTED;
        rows = bb / kvx_row_bytes(xt, hd);
        sbb  = rows * kvx_row_bytes(xf, hd);
    }
    if (st->shape[1] < sbb) return RAD_E_SHAPE;
    const int64_t slots = st->shape[0], slot_b = st->stride[0], blocks = pl->shape[0];
    int64_t* paged = a->n_t > 4 && a->t[4].data ? (int64_t*)a->t[4].data : nullptr;
    if (paged && a->t[4].dtype != RAD_I64) return RAD_E_DTYPE;
    int64_t n_cold = 0;
    std::vector<bool> used((size_t)blocks);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t j = 0; j < w; ++j) {
            const int64_t e = ldi(sel, m * sel->stride[0] + j * sel->stride[1]);
            if (e > -2) continue;
            const int64_t slot = -2 - e, pg = ldi(sc, m * sc->stride[0] + j * sc->stride[1]);
            if (slot >= slots || pg < 0 || pg >= blocks || used[(size_t)pg]) return RAD_E_INVAL;
            used[(size_t)pg] = true;
            ++n_cold;
        }
    const uint8_t* sb = (const uint8_t*)st->data;
    uint8_t* pb = (uint8_t*)pl->data;
    for (int64_t m = 0; m < M; ++m)
        for (int64_t j = 0; j < w; ++j) {
            const int64_t so = m * sel->stride[0] + j * sel->stride[1];
            const int64_t e = ldi(sel, so);
            if (e > -2) continue;
            const int64_t pg = ldi(sc, m * sc->stride[0] + j * sc->stride[1]);
            if (xf >= 0)
                kvx_rows(xf, xt, hd, sb + (-2 - e) * slot_b, kvx_row_bytes(xf, hd), pb + pg * pitch,
                         kvx_row_bytes(xt, hd), rows);
            else
                std::memcpy(pb + pg * pitch, sb + (-2 - e) * slot_b, (size_t)bb);
            st_int(sel->dtype, sel->data, so, pg);
        }
    if (paged) *paged += n_cold;
    return RAD_OK;
}

/* ------------------------------------------------------------------ qsa_work */
/* The blocks that complete in this batch, in sequence order and ascending within one: sequence s
 * with positions p0 .. p1 completes blocks p0 / ratio .. (p1 + 1) / ratio - 1. Work item w gets its
 * block's page bt[s][b] (-1 when the sequence has no tail slot), its first-token position b * ratio
 * in bpos (or, with `rope`, three components: a first token in the batch reads its own, an earlier
 * one its position plus the shift rope[0][cu[s]] - p0), and, when the slot exists, the block's
 * `ratio` raw keys in stage rows w * ratio + i: key q from the batch row cu[s] + q - p0 when q >= p0,
 * else from tail row q % ring. Items past the count get page -1 and position 0 and leave `stage`
 * alone. nc[s] = (last position + 1) / ratio, 0 for an empty sequence. */
int ref_qsa_work(const RadArgs* a, RadStream) {
    if (a->n_t < 11) return RAD_E_INVAL;
    const RadTensor *k = t_in(a, 0), *po = t_in(a, 1), *cu = t_in(a, 2), *tl = t_in(a, 3);
    const RadTensor *sx = t_in(a, 4), *bt = t_in(a, 5), *rp = t_in(a, 6), *stg = t_in(a, 7);
    const RadTensor *pg = t_in(a, 8), *bp = t_in(a, 9), *nc = t_in(a, 10);
    if (!k || !po || !cu || !tl || !sx || !bt || !stg || !pg || !bp || !nc) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), ratio = p_int(a, "ratio", 0), ring = p_int(a, "ring", 0);
    if (n <= 0 || ratio <= 0 || ring < ratio) return RAD_E_INVAL;
    if (k->rank != 2 || bt->rank != 2 || stg->rank != 2) return RAD_E_SHAPE;
    if (tl->rank != 4 || tl->shape[1] != 1 || tl->shape[2] < ring || tl->shape[3] != n)
        return RAD_E_SHAPE;
    const int64_t T = k->shape[0], n_seq = numel(cu) - 1, slots = tl->shape[0];
    if (T <= 0 || n_seq <= 0 || numel(po) < T || k->shape[1] != n || stg->shape[1] != n)
        return RAD_E_SHAPE;
    if (bt->shape[0] < n_seq || (sx->rank >= 2 ? sx->shape[0] : numel(sx)) < n_seq)
        return RAD_E_SHAPE;
    const int64_t mw = numel(pg);
    if (mw < n_seq || stg->shape[0] < mw * ratio || numel(nc) < n_seq) return RAD_E_SHAPE;
    int64_t bstep = 1;
    if (bp->rank == 2) {
        if (bp->shape[0] < mw || bp->stride[1] != 1 || bp->stride[0] < bp->shape[1] ||
            (rp && bp->shape[1] < 3))
            return RAD_E_SHAPE;
        bstep = bp->stride[0];
    } else if (rp || numel(bp) < mw) {
        return RAD_E_SHAPE;
    }
    if (rp && (rp->rank != 2 || rp->shape[0] < 3 || rp->shape[1] < T)) return RAD_E_SHAPE;
    for (int64_t t = 0; t < T; ++t)
        if (ldi(po, offlin(po, t)) < 0) return RAD_E_INVAL;

    /* The work list, validated before any write. */
    struct Item { int64_t s, b; };
    std::vector<Item> items;
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t t0 = cu_at(cu, s), t1 = cu_at(cu, s + 1);
        if (t1 <= t0) continue;
        if (t0 < 0 || t1 > T) return RAD_E_SHAPE;
        const int64_t p0 = ldi(po, offlin(po, t0)), p1 = ldi(po, offlin(po, t1 - 1));
        for (int64_t b = p0 / ratio; b <= (p1 + 1) / ratio - 1; ++b) {
            if ((int64_t)items.size() >= mw) break;
            if (b >= bt->shape[1]) return RAD_E_SHAPE;
            const int64_t si = ldi(sx, col0(sx, s));
            if (si >= slots) return RAD_E_INVAL;
            for (int64_t i = 0; si >= 0 && i < ratio; ++i) {
                const int64_t q = b * ratio + i;
                if (q >= p0 && t0 + (q - p0) >= T) return RAD_E_SHAPE;
            }
            items.push_back({ s, b });
        }
    }

    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t t0 = cu_at(cu, s), t1 = cu_at(cu, s + 1);
        st_int(nc->dtype, nc->data, offlin(nc, s),
               t1 > t0 ? (ldi(po, offlin(po, t1 - 1)) + 1) / ratio : 0);
    }
    const int ncomp = rp ? 3 : 1;
    auto bpos_at = [&](int64_t w, int c) { return bp->rank == 2 ? w * bstep + c : offlin(bp, w); };
    for (int64_t w = 0; w < mw; ++w) {
        if (w >= (int64_t)items.size()) {
            st_int(pg->dtype, pg->data, offlin(pg, w), -1);
            for (int c = 0; c < ncomp; ++c) st_int(bp->dtype, bp->data, bpos_at(w, c), 0);
            continue;
        }
        const int64_t s = items[(size_t)w].s, b = items[(size_t)w].b;
        const int64_t t0 = cu_at(cu, s), p0 = ldi(po, offlin(po, t0));
        const int64_t si = ldi(sx, col0(sx, s));
        st_int(pg->dtype, pg->data, offlin(pg, w),
               si < 0 ? -1 : ldi(bt, s * bt->stride[0] + b * bt->stride[1]));
        const int64_t f = b * ratio;
        if (!rp) {
            st_int(bp->dtype, bp->data, bpos_at(w, 0), f);
        } else {
            const int64_t rs = rp->stride[0], rc = rp->stride[1];
            for (int c = 0; c < 3; ++c)
                st_int(bp->dtype, bp->data, bpos_at(w, c),
                       f >= p0 ? ldi(rp, c * rs + (t0 + f - p0) * rc)
                               : f + (ldi(rp, t0 * rc) - p0));
        }
        if (si < 0) continue;
        for (int64_t i = 0; i < ratio; ++i) {
            const int64_t q = f + i;
            const RadTensor* src = q >= p0 ? k : tl;
            const int64_t sb = q >= p0 ? (t0 + q - p0) * k->stride[0]
                                       : si * tl->stride[0] + (q % ring) * tl->stride[2];
            const int64_t sc = q >= p0 ? k->stride[1] : tl->stride[3];
            const int64_t db = (w * ratio + i) * stg->stride[0];
            for (int64_t d = 0; d < n; ++d)
                stt(stg, db + d * stg->stride[1], ldt(src, sb + d * sc));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ dflash_conv */
/* DFlash2's grouped dynamic depthwise convolution over the [T, H] rows of x, blocks of
 * `block_size` rows starting at row 0:
 *
 *   out[t][h] = sum over taps j with j == 0 or t % block_size >= j of
 *               (base[j][h] + delta[t][j][h / group_size]) * x[t - j][h]
 *
 * in tap order, each term `(base + delta) * x` rounded as written, rounded once to out's dtype.
 * `delta` is [T, taps, NG], or [T, >= taps * NG] read at its own row pitch (a slice of the
 * [T, 2, taps, NG] coefficient projection). `out` must not alias `x`: tap j reads row t - j. */
int ref_dflash_conv(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *dl = &a->t[1], *bs = &a->t[2], *o = &a->t[3];
    const int64_t taps = p_int(a, "taps", 0), group = p_int(a, "group_size", 0);
    const int64_t NG = p_int(a, "NG", 0), bsz = p_int(a, "block_size", 0);
    if (taps <= 0 || group <= 0 || NG <= 0 || bsz <= 0) return RAD_E_INVAL;
    if (x->data == o->data) return RAD_E_INVAL;
    if (x->rank != 2 || o->rank != 2) return RAD_E_SHAPE;
    const int64_t T = x->shape[0], H = x->shape[1];
    if (T <= 0 || o->shape[0] != T || o->shape[1] != H || H != NG * group) return RAD_E_SHAPE;
    if (numel(bs) < taps * H) return RAD_E_SHAPE;
    int64_t d_rs = 0, d_ts = 0, d_gs = 0;
    if (dl->rank == 3) {
        if (dl->shape[0] < T || dl->shape[1] < taps || dl->shape[2] < NG) return RAD_E_SHAPE;
        d_rs = dl->stride[0]; d_ts = dl->stride[1]; d_gs = dl->stride[2];
    } else if (dl->rank == 2) {
        if (dl->shape[0] < T || dl->shape[1] < taps * NG) return RAD_E_SHAPE;
        d_rs = dl->stride[0]; d_gs = dl->stride[1]; d_ts = NG * d_gs;
    } else {
        return RAD_E_SHAPE;
    }
    const int64_t x_rs = x->stride[0], x_cs = x->stride[1];
    const int64_t o_rs = o->stride[0], o_cs = o->stride[1];
    REF_PAR_FOR(T * H * taps)
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < H; ++h) {
            const int64_t g = h / group;
            float acc = 0.0f;
            for (int64_t j = 0; j < taps; ++j) {
                if (j > 0 && t % bsz < j) continue;
                const float c = ldt(bs, rowoff(bs, j, H) + h * laststride(bs)) +
                                ldt(dl, t * d_rs + j * d_ts + g * d_gs);
                const float term = c * ldt(x, (t - j) * x_rs + h * x_cs);
                acc = j == 0 ? term : acc + term;
            }
            stt(o, t * o_rs + h * o_cs, acc);
        }
    return RAD_OK;
}
