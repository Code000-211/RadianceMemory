/* rad_qsa.h -- Qwen4-Exp's QSA indexer: decides which blocks of key history a full-attention
 * layer's attention reads.
 *
 * docs/QSA.md has the design; this is the declaration side. The kernels are libr4d's
 * `qsa_block_key`, `qsa_score` and `qsa_select`; the sparse attention is the ordinary `attn_paged`
 * issued with the selection in place of the sequence's block table (both plain operands, so core/
 * knows nothing of QSA).
 *
 * Contents: the projection and query path, the block-key path with its two states, and the score
 * and selection. Below `budget + ratio - 1` tokens of context every visible block is selected, so
 * the sparse path must reproduce the dense path's text byte for byte; that is the test for any
 * change here.
 */
#ifndef RAD_QSA_H
#define RAD_QSA_H

#include "rad_arch.h"
#include "rad_fp8.h"

#include <atomic>

namespace rad {
namespace arch {

struct QsaIndexer {
    struct Config {
        int64_t     heads    = 0;   /* indexer_n_heads      -- 4 */
        int64_t     head_dim = 0;   /* indexer_head_dim     -- 128 */
        int64_t     ratio    = 0;   /* indexer_compress_ratio -- 4 tokens a block */
        int64_t     budget   = 0;   /* indexer_budget       -- 2048 tokens, so 512 blocks */
        int64_t     rot      = 0;   /* the layer's rotary width, shared by the indexer */
        const char* mode     = "neox";
        /* Checkpoint names. Absent means this layer has no indexer and declare() is a no-op. */
        const char* qk_proj  = nullptr;
        const char* q_norm   = nullptr;
        const char* k_norm   = nullptr;
        /* Derived; set by the caller because they size buffers it declares. */
        int64_t     topk       = 0;   /* budget / ratio -- blocks the selection keeps */
        int64_t     max_blocks = 0;   /* max_ctx / ratio -- the widest block table */
        int64_t     max_work   = 0;   /* ceil(max_tok / ratio) + max_seqs */
        /* The tail's ring length, `ratio + max_spec`. A ring of `ratio` indexed by `pos % ratio`
         * fails under speculation: the next step restarts at the first rejected position, so
         * overwritten slots are needed again (at ratio 4, depth 3, positions 9..12 write slots
         * 1,2,3,0 and 12 overwrites the slot of 8; accepting one then rebuilds block 2 from token
         * 12, corrupting that block key permanently). A step can still owe positions
         * [p0 - ratio + 1, p0 + n_spec], so `ratio + max_spec` slots keep every owed position. */
        int64_t     ring       = 0;   /* ratio + max_spec */
        /* The attention group, whose page ids the selection outputs. The block-key group is
         * declared on first use (see declare()). */
        rad_kvgroup kv_attn    = 0;
        /* The tail's group: one RAD_KV_LINEAR state per sequence per layer, `ring` rows of
         * `head_dim` (a KV group because it is per-sequence state; see Wire). */
        rad_kvgroup kv_tail    = 0;
        /* Whether anything reads the selection. False on the dense path; the stateful block-key
         * half still runs so the sparse path can later read the history, but score and select (the
         * indexer's largest ops) are skipped. Declared either way, so the graph and buffer plan do
         * not depend on it. */
        bool        emit_sel   = true;
        /* The rows the score plane was declared with, 0 for max_tok. Fewer only where a step
         * wider than them is one sequence (RadBuildCtx::max_mixed_tok); such a step scores and
         * selects in windows of this many rows (step()). */
        int64_t     plane_rows = 0;
        /* Issue `qsa_page` between the selection and the attention (spec §7.3): the group
         * declares a window (declare() always sets one) and the deployment asked for the store
         * (RadBuildCtx::kv_residency). False declares no pager op, and the graph is the one a
         * server that keeps every block resident runs. Also false on a rank that keeps no
         * attention cache (Geom::attn_zero): there is no pool to page into, and nothing reads
         * that rank's selection. */
        bool        page       = false;
        /* The bytes of one layer's fragment of an attention block (K and V of `ratio` tokens), the
         * pager's `block_bytes`: what one cold block copies per layer. */
        int64_t     page_block_bytes = 0;
        /* Block keys in int8 (RadBuildCtx::qsa_bkey_dtype RAD_I8): a page holds the `head_dim`
         * codes and their f16 scale (docs/QSA.md, "Block-key storage"), and `qsa_block_key` and
         * `qsa_score` take `key_dtype` i8. False is today's bf16 page and the same program. */
        bool        bkey_i8 = false;
        /* A tiered group (RAD_KVQ_DEMOTE): the store holds the fragment in `page_store_layout`
         * and the pager expands it into `page_pool_layout` at `page_head_dim` (the op's
         * store_layout, pool_layout and head_dim). Null: the bytes are copied as they are. */
        const char* page_store_layout = nullptr;
        const char* page_pool_layout  = nullptr;
        int64_t     page_head_dim     = 0;
        /* Block keys split across the tensor-parallel ranks (RadBuildCtx::qsa_bkey_split; docs/
         * QSA.md, "Block keys split across ranks"): `split` ranks, this one `split_rank`. Block b
         * is stored and scored only on rank b % split, the block-key group is paged at
         * `split * ratio` tokens (one key a page, so a rank holds 1/split of them), and the
         * selection is qsa_select_part, an all_gather of the lists and qsa_select_join: the
         * unsplit selection byte for byte. 0 is today's program, parameter for parameter. */
        int         split      = 0;
        int         split_rank = 0;
    };

    /* `qk` is the whole projection (`heads` query heads, then one raw key head); `q` is the query
     * heads normed and rotated, contiguous.
     *
     * The raw key is not normed or rotated here: it is pooled over `ratio` tokens first, and the
     * pooled result is normed and rotated at the block's first token's position (`qsa_block_key`),
     * unlike the q path. */
    struct Wire {
        rad_buf qk = 0;   /* [max_tok, (heads + 1) * head_dim] */
        rad_buf q  = 0;   /* [max_tok, heads * head_dim] */
        /* These buffers are within-step scratch shared by every layer. The tail is not here: it
         * carries an unfinished block's raw index keys across steps, per layer (each layer has its
         * own `index_qk_proj`, the MTP head included) and per sequence (batch rows are step-local
         * and shift between steps). A KV group has a stable per-sequence slot, zeroed on hand-out
         * and checkpointed with the linear state by the prefix cache, so the tail is
         * `Config::kv_tail` (RAD_KV_LINEAR), bound to this layer in declare(). */
        rad_buf stage = 0;   /* [max_work * ratio, head_dim] */
        rad_buf page  = 0;   /* [max_work]      i32 */
        rad_buf bpos  = 0;   /* [max_work] i32, or [max_work, 3] under M-RoPE */
        rad_buf nc    = 0;   /* [max_seqs]      i32 */
        /* The selection, one row per query, not per sequence (the same at decode; at prefill each
         * query of a chunk selects from the shared history, docs/QSA.md, "Per-query selection"). `sel` and
         * `seqused` replace the sequence's block table and length for the sparse attention.
         *
         * `score` is the largest: max_tok * max_ctx/ratio floats, 134 MB at a 2048-token chunk
         * and 64K context, one buffer for the whole model. A step of one sequence is tiled on the
         * query axis when it is wider than the plane (Config::plane_rows): a window of it is a
         * sequence of its own rows. A step of several is not: that would need a per-token
         * sequence id operand (from qsa_work), since `cu` is absolute over the batch. */
        rad_buf score = 0;   /* [max_tok, max_blocks] f32 */
        rad_buf sel   = 0;   /* [max_tok, topk + 1]   i32 */
        rad_buf nsel  = 0;   /* [max_tok]             i32 */
        rad_buf sequ  = 0;   /* [max_tok]             i32 */
        /* The model's cos/sin plane, [rope_table_rows, rot], or 0 if it builds none. Filled once a
         * step by `rope_table` for the attention block; the indexer rotates at the same `rot`,
         * `theta` and `scale`. Without it the unfused per-head norms run. */
        rad_buf cos_sin = 0;
        /* Under Config::split only: this rank's top-k list, [plane rows, part_width(topk)] i32,
         * and every rank's side by side as the row-interleaved all_gather lays them out,
         * [plane rows, split * part_width(topk)]. Scratch shared by every layer, like `sel`. */
        rad_buf part = 0;
        rad_buf gath = 0;
    };

    /* One rank's list row (qsa_select_part, docs/OPS.md): 2 * topk + 1 i32 words rounded up to
     * whole 16-byte words, which a row-interleaved all_gather moves. */
    static int64_t part_width(int64_t topk) { return (2 * topk + 1 + 3) / 4 * 4; }

    Geom       g{};
    Config     c{};
    Wire       w{};
    int        layer = 0;
    rad_weight  w_qn = 0, w_kn = 0;
    LinearFP8   proj{};
    rad_kvgroup kv_bk = 0;
    rad_op      op_norm = 0, op_rope = 0, op_qprep = 0;
    /* Most rows `op_qprep` may be issued over (0 when not declared): `qk_fuse_rows`, a correctness
     * bound, since the cos/sin table is filled only for steps at or under it. */
    int64_t     qprep_rows = 0;
    rad_op      op_work = 0, op_bkey = 0, op_tail = 0, op_score = 0, op_select = 0, op_page = 0;
    /* Under Config::split, in place of op_select. */
    rad_op      op_part = 0, op_gather = 0, op_join = 0;
    /* Ranks the block keys are split over: Config::split, or 1. */
    int64_t     ranks() const { return c.split > 1 ? c.split : 1; }

    bool on() const { return op_rope != 0; }

    /* `bk` is the block-key KV group, created on first use and shared by all QSA layers (hence
     * in/out). It must be declared immediately before `qsa_block_key`: `Builder::find_kv_consumer`
     * binds a group to the first later-declared op with a `block_size`. */
    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Config& cfg,
                const Wire& wire, const ActFP8& in, rad_kvgroup* bk);
    /* `win_cu`, device: {0, plane_rows}, the one sequence of a window of a step wider than the
     * score plane (Config::plane_rows); null where no step is. */
    void step(RadCtx* c, const ActFP8& in, const RadBatch* batch,
              const int32_t* win_cu = nullptr) const;
    /* Stores the pass's block keys and tail exactly as step() does, with no query, score or
     * selection (see AttnGatedFP8::store_kv). */
    void store(RadCtx* c, const ActFP8& in, const RadBatch* batch) const;
    bool store_keys(RadCtx* c, const RadBatch* batch) const;
};

inline int QsaIndexer::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                               const Config& cfg, const Wire& wire, const ActFP8& in,
                               rad_kvgroup* bk) {
    g = geom; c = cfg; w = wire; layer = l;
    if (!c.qk_proj || !c.q_norm) return RAD_OK;          /* no indexer on this layer */
    if (c.heads <= 0 || c.head_dim <= 0 || c.ratio <= 0 || c.budget <= 0) return RAD_E_INVAL;
    if (!w.qk || !w.q) return RAD_E_INVAL;
    /* The layer's rotary width must fit inside the index head (as the checkpoint's validator also
     * requires). */
    if (c.rot < 0 || c.rot > c.head_dim || (c.rot & 1)) return RAD_E_SHAPE;

    /* The residency window (spec §7.3). QSA reads `budget` tokens of a history, selected from
     * anywhere in it, plus the tail, so the blocks the attention does not read are dead weight in
     * VRAM: a 200K-token context's attention cache is 4.8 GiB of which a decode step reads about
     * 48 MiB. Declaring the reach here, and not in the attention helper, is what lets the core
     * move the rest to host memory: the helper declares a group whose consumer might read all of
     * it, and only the indexer knows the model does not.
     *
     * The number is a policy hint and never a correctness bound: a selection that reaches past it
     * is promoted in the step that reads it (KVManager::residency_promote), which costs link
     * traffic and nothing else. The block-key group is deliberately not annotated -- the scorer
     * reads every block key, so its pages must stay resident (3.25 of the 29.25 KiB a 4-token
     * block costs at bf16, 1.73 in int8, and the part no window can shrink). */
    RAD_ARCH_TRY(rad_kv_set_residency_window(b, c.kv_attn, c.budget));

    const int64_t nw = (c.heads + 1) * c.head_dim;       /* the projection width */

    RAD_ARCH_TRY(proj.declare(b, nm, g, nm.f("blk.%d.qsa_qk", l), nw, g.n_embd,
                              RAD_SHARD_NONE, grp_layer(l), { c.qk_proj }, in, w.qk));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.qsa_q_norm.weight", l), c.q_norm));
    w_qn = decl_w(b, nm.f("blk.%d.qsa_q_norm.weight", l), RAD_F32, { c.head_dim },
                  RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    if (!w_qn) return RAD_E_INVAL;

    /* Unfused, one norm op is issued once per head: in `qk` (`heads + 1` heads wide) the query
     * heads are 128 apart within a token and 640 apart across tokens, which no rank-2 view can
     * express, but a one-head column slice can. It writes a contiguous `q` that the rope takes
     * whole. For 128-wide rows these launches are nearly all overhead.
     *
     * `op_qprep` fuses them (five launches to one) using `qk_norm_rope_gate`'s k side: it reads
     * [tokens, heads, head_dim] at the operand's row pitch, writes heads contiguously, and rounds
     * the norm to bf16 before rotating. With no gate, `n_head` 0 selects that half and passes no
     * q operands. It needs the model's cos/sin table (same `rot`, `theta`, `scale` as the
     * attention block) and is used only up to `qk_fuse_rows`, a correctness bound: the table is
     * filled only for steps that small, so a larger step would rotate by stale angles. The bound
     * comes from the same function as the attention block's. */
    const bool qprep_neox = !std::strcmp(c.mode, "neox") || !std::strcmp(c.mode, "mrope") ||
                            g.rope_mc;
    qprep_rows = qk_fuse_rows(g);
    if (w.cos_sin && qprep_neox && qprep_rows > 0 && (c.head_dim % 32) == 0 &&
        rad_op_declared(b, "qk_norm_rope_gate"))
        op_qprep = rw(b, RAD_OP(b, "qk_norm_rope_gate",
                         RAD_PARAMS(RAD_RANGE("M", 1, qprep_rows),
                                    RAD_INT("head_dim", c.head_dim), RAD_INT("n_head", 0),
                                    RAD_INT("n_head_kv", c.heads), RAD_INT("rot", c.rot),
                                    RAD_STR("q_dtype", g.dtype), RAD_F64("eps", g.eps),
                                    RAD_F64("wadd", g.wadd_qk), RAD_INT("cs_f32", 1)),
                         RAD_WEIGHTS(w_qn)),
                     {w.qk, w.cos_sin}, {w.q});

    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.head_dim),
                               RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk),
                               RAD_STR("dtype", g.dtype)),
                    RAD_WEIGHTS(w_qn)),
                {w.qk}, {w.q});
    op_rope = rw(b, decl_rope(b, g, g.max_tok, c.head_dim, c.heads, 0, c.rot, c.mode),
                {w.q}, {w.q});

    /* ---- the block-key path. Nothing below reads anything above except `w.qk`'s raw-key half. */
    if (!c.k_norm || !w.stage || !c.topk || !c.max_blocks || !c.max_work)
        return RAD_E_INVAL;
    if (!c.kv_tail || c.ring < c.ratio) return RAD_E_INVAL;
    if (c.split == 1 || c.split < 0 || (c.split > 1 && (c.split_rank < 0 || c.split_rank >= c.split ||
                                                        !w.part || !w.gath)))
        return RAD_E_INVAL;
    const int64_t R = ranks();
    /* This layer's tail state: the caller's group holds one state per sequence per bound layer. */
    RAD_ARCH_TRY(rad_bind_layer_kv(b, l, c.kv_tail));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.qsa_k_norm.weight", l), c.k_norm));
    w_kn = decl_w(b, nm.f("blk.%d.qsa_k_norm.weight", l), RAD_F32, { c.head_dim },
                  RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    if (!w_kn) return RAD_E_INVAL;

    /* Split, `ranks` and `rank` send every block another rank stores to page -1 and the rest to
     * page bt[b / ranks]; unsplit the parameters are absent, today's op parameter for parameter. */
    {
        const RadParam ps[] = { RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.head_dim),
                                RAD_INT("ratio", c.ratio), RAD_INT("ring", c.ring),
                                RAD_INT("seqs", g.max_seqs),
                                RAD_INT("work", c.max_work), RAD_STR("dtype", g.dtype),
                                RAD_INT("ranks", R), RAD_INT("rank", c.split_rank) };
        const int n_ps = (int)(sizeof ps / sizeof ps[0]) - (R > 1 ? 0 : 2);
        op_work = rw(b, rad_decl_op(b, "qsa_work", ps, n_ps, RAD_NOWEIGHTS),
                     {w.qk}, {w.stage, w.page, w.bpos, w.nc});
    }

    /* The block-key group, declared immediately before its consumer (see the declaration). One KV
     * head, `block_size` = ratio and `head_dim` = n / (2 * ratio) make a page hold exactly one
     * block key of `n` values, one page per `ratio` tokens.
     *
     * In int8 a page is `n` codes and an f16 scale, n + 2 bytes, which a page of `2 * ratio *
     * head_dim` bytes holds only rounded up: the group is declared RAD_I8 with head_dim
     * ceil((n + 2) / (2 * ratio)), 136 bytes at n 128 and ratio 4 (6 of them zero padding), so
     * the pool, the prefix cache and the residency store size the page from the declaration as
     * they do any other. Every QSA layer and the MTP head share the group, so all take the same
     * dtype (one Config). */
    /* Split across R ranks the group is paged at R * ratio tokens: a page still holds one key (the
     * key of this rank's block in that span), so a rank allocates one page per R blocks and the
     * pool, prefix cache and residency store follow from the declaration as before. Unsplit
     * `bs` is `ratio` and everything below is today's arithmetic. */
    const int64_t bs = R * c.ratio;
    const bool first = *bk == 0;
    if (first) {
        if ((c.head_dim % (2 * bs)) != 0) return RAD_E_SHAPE;
        RadKVGroupDecl d{};
        d.kind      = RAD_KV_FULL;
        d.dtype     = c.bkey_i8 ? RAD_I8 : RAD_BF16;
        d.n_head_kv = 1;
        d.head_dim  = c.bkey_i8 ? (c.head_dim + 2 + 2 * bs - 1) / (2 * bs)
                                : c.head_dim / (2 * bs);
        *bk = rad_decl_kv_group(b, nm.f("kv_qsa_bkey"), &d);
        if (!*bk) return RAD_E_INVAL;
        /* qsa_score reads every complete block of the history each step, so this group stays resident
         * and holds every sequence's context while a sibling streams (kvstream/carve.h). */
        RAD_ARCH_TRY(rad_kv_group_dense_read(b, *bk));
    }
    kv_bk = *bk;
    /* Under M-RoPE a block key rotates at its first token's three components, which qsa_work
     * writes into a [work, 3] `bpos` when given the step's rotary planes. */
    {
        const RadParam ps[] = { RAD_RANGE("M", 1, c.max_work), RAD_INT("n", c.head_dim),
                                RAD_INT("ratio", c.ratio), RAD_INT("rotary_dim", c.rot),
                                RAD_INT("block_size", bs),
                                RAD_F64("eps", g.eps), RAD_F64("theta", g.theta),
                                RAD_F64("scale", g.rope_scale), RAD_F64("wadd", g.wadd_qk),
                                RAD_STR("dtype", g.dtype), RAD_STR("mode", g.rope_mode),
                                RAD_STR("sections", g.rope_sections),
                                RAD_STR("key_dtype", "i8") };
        /* `key_dtype` only in int8 (here and on `qsa_score`), so the bf16 program is today's
         * parameter for parameter; `mode` and `sections` only under M-RoPE. */
        RadParam pk[sizeof ps / sizeof ps[0]];
        int n_ps = 0;
        /* Split, only the op the group binds to states `block_size` (the kernels default it to
         * `ratio` and only check it). The attention groups are all declared before any op and
         * bind to the first unclaimed op with a block size (Builder::find_kv_consumer), so a
         * later layer's qsa_block_key stating 2 * ratio would hand a fifth layout's group a block
         * of 8 tokens and its attention a shape no kernel serves. Unsplit every op states `ratio`,
         * today's program parameter for parameter. */
        for (const RadParam& p : ps)
            if ((g.rope_mc || (std::strcmp(p.key, "mode") && std::strcmp(p.key, "sections"))) &&
                (c.bkey_i8 || std::strcmp(p.key, "key_dtype")) &&
                (first || R == 1 || std::strcmp(p.key, "block_size")))
                pk[n_ps++] = p;
        op_bkey = rw(b, decl_rotary(b, g, "qsa_block_key", pk, n_ps, &w_kn, 1),
                     {w.stage, w.page, w.bpos}, {});
    }
    /* Bind the group to this op now, before a layer's attention asks for its own group's block
     * size and claims it first (the binding is first-come, declaration order). */
    if (first && rad_kv_block_size(b, kv_bk) <= 0) return RAD_E_STATE;
    RAD_ARCH_TRY(rad_bind_layer_kv(b, l, kv_bk));

    op_tail = rw(b, RAD_OP(b, "qsa_tail_store",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.head_dim),
                               RAD_INT("ratio", c.ratio), RAD_INT("ring", c.ring),
                               RAD_INT("seqs", g.max_seqs),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.qk}, {});

    /* M is queries for both, up to a whole prefill chunk: the selection is per query. */
    {
        const RadParam ps[] = { RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.head_dim),
                                RAD_INT("heads", c.heads), RAD_INT("blocks", c.max_blocks),
                                RAD_INT("seqs", g.max_seqs), RAD_STR("dtype", g.dtype),
                                RAD_STR("key_dtype", "i8") };
        const int n_ps = (int)(sizeof ps / sizeof ps[0]) - (c.bkey_i8 ? 0 : 1);
        op_score = rw(b, rad_decl_op(b, "qsa_score", ps, n_ps, RAD_NOWEIGHTS), {w.q, w.nc},
                      {w.score});
    }
    if (R == 1) {
        op_select = rw(b, RAD_OP(b, "qsa_select",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                     RAD_INT("blocks", c.max_blocks), RAD_INT("topk", c.topk),
                                     RAD_INT("ratio", c.ratio), RAD_INT("seqs", g.max_seqs)),
                          RAD_NOWEIGHTS),
                      {w.score, w.nc}, {w.sel, w.nsel, w.sequ});
    } else {
        /* Split: this rank's top-k list, every rank's lists side by side (a row-interleaved
         * gather, one list row a query), and their merge into the table qsa_select would build
         * over the unsplit scores. `M` is a window of the score plane at most (step()). */
        const int64_t PW = part_width(c.topk);
        const int64_t rows = c.plane_rows > 0 ? c.plane_rows : g.max_tok;
        op_part = rw(b, RAD_OP(b, "qsa_select_part",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                   RAD_INT("blocks", c.max_blocks), RAD_INT("topk", c.topk),
                                   RAD_INT("ratio", c.ratio), RAD_INT("seqs", g.max_seqs),
                                   RAD_INT("ranks", R), RAD_INT("rank", c.split_rank)),
                        RAD_NOWEIGHTS),
                     {w.score, w.nc}, {w.part});
        op_gather = rw(b, RAD_OP(b, "all_gather",
                          RAD_PARAMS(RAD_INT("world_size", R), RAD_RANGE("numel", PW, rows * PW),
                                     RAD_STR("dtype", "f32"), RAD_INT("row", PW)),
                          RAD_NOWEIGHTS),
                       {w.part}, {w.gath});
        op_join = rw(b, RAD_OP(b, "qsa_select_join",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                   RAD_INT("blocks", c.max_blocks), RAD_INT("topk", c.topk),
                                   RAD_INT("ratio", c.ratio), RAD_INT("seqs", g.max_seqs),
                                   RAD_INT("ranks", R)),
                        RAD_NOWEIGHTS),
                     {w.gath, w.nc}, {w.sel, w.nsel, w.sequ});
        if (!op_part || !op_gather || !op_join) return RAD_E_INVAL;
    }

    /* When the attention group streams, the attention block resolves `sel` to ring slots after its
     * layer's store (AttnGatedFP8::step); prefill runs on the stage and never resolves. */
    /* The pager rewrites `sel` in place, so it both reads and writes it; the attention, declared
     * before this op, reads `sel` through its own liveness declaration. `M` is the decode rows,
     * the ones that own scratch pages (RadKVGroupBatch::residency_scratch). A prefill row's
     * history is promoted before its step instead, so it never names a cold block. */
    if (c.page && c.page_store_layout) {
        /* Tiered: the same op expanding the demote layout back into the pool's, so the attention
         * reads one layout. */
        if (c.page_block_bytes <= 0 || !c.page_pool_layout || c.page_head_dim <= 0)
            return RAD_E_INVAL;
        op_page = rw(b, RAD_OP(b, "qsa_page",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("topk", c.topk),
                                     RAD_INT("blocks", c.max_blocks),
                                     RAD_INT("seqs", g.max_seqs),
                                     RAD_INT("block_bytes", c.page_block_bytes),
                                     RAD_STR("store_layout", c.page_store_layout),
                                     RAD_STR("pool_layout", c.page_pool_layout),
                                     RAD_INT("head_dim", c.page_head_dim)),
                          RAD_NOWEIGHTS),
                     {w.sel}, {w.sel});
    } else if (c.page) {
        if (c.page_block_bytes <= 0) return RAD_E_INVAL;
        op_page = rw(b, RAD_OP(b, "qsa_page",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("topk", c.topk),
                                     RAD_INT("blocks", c.max_blocks),
                                     RAD_INT("seqs", g.max_seqs),
                                     RAD_INT("block_bytes", c.page_block_bytes)),
                          RAD_NOWEIGHTS),
                     {w.sel}, {w.sel});
    }
    return RAD_OK;
}

/* Stores the block keys of the blocks the pass completes and the tail of raw keys after them,
 * from the projection's raw-key head. False when the step has no batch for the indexer's
 * groups. */
inline bool QsaIndexer::store_keys(RadCtx* c_, const RadBatch* batch) const {
    const int64_t T = batch->n_tok, S = batch->n_seq;
    const int64_t nw = (c.heads + 1) * c.head_dim;
    const int64_t koff = c.heads * c.head_dim;   /* the raw-key head, after the query heads */
    const RadKVGroupBatch* bkb = kv_batch(batch, kv_bk);
    const RadKVGroupBatch* tlb = kv_batch(batch, c.kv_tail);
    if (!bkb || !tlb) return false;
    /* Column 0 of the state row is the committed slot; the row is wider when speculating (linear
     * groups get a second copy), so use the batch's pitch. */
    const int64_t tw = tlb->state_index_pitch > 0 ? tlb->state_index_pitch : 1;

    /* Invariant: the work list runs before the tail store. The gather reads a completing block's
     * raw keys, up to `ratio - 1` of which are in the tail from earlier steps and are about to be
     * overwritten. The ring also spaces overwrites by `ring` positions, but the order still decides
     * what a straddling block reads. */
    const int64_t work = (T + c.ratio - 1) / c.ratio + S;
    /* Block positions: one component per block, or three under M-RoPE with planes. Under M-RoPE
     * `bpos` is [max_work, 3]; this step passes [work, 3], or only column 0 when the pass has no
     * planes (qsa_work then writes component 0 and the block key reads the one-component form). */
    const bool mc = g.rope_mc && batch->rope_pos;
    RadOperand bpos = g.rope_mc ? RAD_B_COL(w.bpos, 0, mc ? 3 : 1) : brows(w.bpos, work);
    if (g.rope_mc) bpos.rows = work;
    RAD_ISSUE_N(c_, op_work, T,
                bcol_at(w.qk, 0, nw, koff, c.head_dim, T),
                praw(batch->positions, RAD_I32, T),
                praw(batch->cu_seqlens, RAD_I32, S + 1),
                kv_cache(c.kv_tail, layer),
                praw2(tlb->state_index, RAD_I32, S, tw),
                praw2(bkb->block_table, RAD_I32, S, bkb->block_table_pitch),
                mc ? praw2(batch->rope_pos, RAD_I32, 3, T) : RAD_NONE,
                brows(w.stage, work * c.ratio), brows(w.page, work), bpos,
                brows(w.nc, S));
    RAD_ISSUE_N(c_, op_bkey, work,
                brows(w.stage, work * c.ratio), RAD_W(w_kn),
                brows(w.page, work), bpos,
                kv_cache(kv_bk, layer));
    RAD_ISSUE_N(c_, op_tail, T,
                bcol_at(w.qk, 0, nw, koff, c.head_dim, T),
                praw(batch->positions, RAD_I32, T),
                praw(batch->cu_seqlens, RAD_I32, S + 1),
                kv_cache(c.kv_tail, layer),
                praw2(tlb->state_index, RAD_I32, S, tw));
    return true;
}

inline void QsaIndexer::store(RadCtx* c_, const ActFP8& in, const RadBatch* batch) const {
    if (!on()) return;
    proj.step(c_, in, w.qk, batch->n_tok);
    (void)store_keys(c_, batch);
}

inline void QsaIndexer::step(RadCtx* c_, const ActFP8& in, const RadBatch* batch,
                             const int32_t* win_cu) const {
    if (!on()) return;
    const int64_t T = batch->n_tok, S = batch->n_seq;
    const int64_t qw = c.heads * c.head_dim;
    const int64_t nw = (c.heads + 1) * c.head_dim;

    proj.step(c_, in, w.qk, T);
    /* One fused launch or `heads` + 1 unfused. The fused form takes the whole projection at its
     * row pitch and writes the query heads contiguous and rotated. `n_head` 0 is the k-only form:
     * no q_gate, q_w or q_out, and `gate_out` absent. `T <= qprep_rows` is a correctness bound
     * (see declare()). */
    if (op_qprep && T <= qprep_rows && !rope_mixed(batch)) {
        RAD_ISSUE_N(c_, op_qprep, T,
                    RAD_NONE, brows(w.qk, T), RAD_B(w.cos_sin), rope_pos1(batch, T),
                    RAD_NONE, RAD_W(w_qn),
                    RAD_NONE, brows(w.q, T), RAD_NONE);
    } else {
        for (int64_t h = 0; h < c.heads; ++h)
            RAD_ISSUE_N(c_, op_norm, T,
                        bcol_at(w.qk, 0, nw, h * c.head_dim, c.head_dim, T), RAD_W(w_qn),
                        bcol_at(w.q,  0, qw, h * c.head_dim, c.head_dim, T));
        RAD_ISSUE_N(c_, op_rope, T, brows(w.q, T), rope_posmc(g, batch, T));
    }

    /* The keys first: the selection below reads the blocks they complete. */
    const RadKVGroupBatch* bkb = kv_batch(batch, kv_bk);
    const RadKVGroupBatch* atb = kv_batch(batch, c.kv_attn);
    if (!atb || !store_keys(c_, batch)) return;

    /* Score and select per query: both ops take `cu`, the block table stays per sequence, and `m`
     * is a query row, so each query of a prefill chunk selects from the shared history, as the
     * checkpoint does. */
    if (!c.emit_sel) return;

    /* Narrow the score columns to the live history. Both kernels size their grid from the
     * operand's width (the live count is device data), so the full `max_ctx` width would make a
     * 2000-token prompt walk 50000 block slots; the pitch stays declared, only the column count
     * moves.
     *
     * This must be an upper bound: an underestimate silently drops blocks. A query has at most
     * (position + 1) / ratio complete blocks; positions are bounded by the host context bound plus
     * the step's tokens plus the draft window; the +1 keeps `n < nb` so the partial page `bt[n]`
     * is readable. */
    const int64_t reach = (int64_t)batch->max_ctx_len + batch->n_tok + batch->n_spec + c.ratio;
    int64_t live = reach / c.ratio + 1;
    if (live > c.max_blocks) live = c.max_blocks;
    if (live < 1) live = 1;
    /* Split, this rank scores column j = block j * R + rank: ceil(live / R) columns hold every
     * block of its share below `live` (qsa_select_part takes min(table, R * columns) as the
     * width, which only ever bounds the query's n from above, as `live` does unsplit). */
    const int64_t R = ranks();
    const int64_t live_s = (live + R - 1) / R;
    /* The selection of the score plane's first `rows` rows into rows [r0, r0 + rows) of
     * sel/nsel/sequ: qsa_select, or split this rank's list, the gather and the merge, which
     * writes the bytes qsa_select would. */
    auto selection = [&](int64_t rows, int64_t r0, RadOperand cu, const int32_t* pos) {
        const RadOperand sc = bcol_at(w.score, 0, c.max_blocks, 0, live_s, rows);
        const RadOperand bt = praw2(atb->block_table, RAD_I32, S, atb->block_table_pitch);
        if (R == 1) {
            RAD_ISSUE_N(c_, op_select, rows, sc, bt, brows(w.nc, S), praw(pos, RAD_I32, rows), cu,
                        brow_slice(w.sel, r0, rows, c.topk + 1), brow_slice(w.nsel, r0, rows, 1),
                        brow_slice(w.sequ, r0, rows, 1));
            return;
        }
        const int64_t PW = part_width(c.topk);
        RAD_ISSUE_N(c_, op_part, rows, sc, bt, brows(w.nc, S), praw(pos, RAD_I32, rows), cu,
                    brows(w.part, rows));
        RAD_ISSUE_N(c_, op_gather, rows * PW, brows(w.part, rows), brows(w.gath, rows));
        RAD_ISSUE_N(c_, op_join, rows, brows(w.gath, rows), bt, brows(w.nc, S),
                    praw(pos, RAD_I32, rows), cu,
                    brow_slice(w.sel, r0, rows, c.topk + 1), brow_slice(w.nsel, r0, rows, 1),
                    brow_slice(w.sequ, r0, rows, 1));
    };

    /* A step wider than the score plane is one sequence (Config::plane_rows), scored and selected
     * in windows of the plane's rows from the front, the last one ending at the step's end and
     * overlapping the one before. A window is handed as a sequence of its own rows (`win_cu` =
     * {0, rows}). Every operand a query row reads is its own (query, position, selection) and the
     * block table and complete-block count are the step's, so each row is scored and selected as
     * in one pass over the step, and a row two windows share is written twice with the same
     * bytes. */
    const int64_t PR = c.plane_rows > 0 ? c.plane_rows : T;
    if (T > PR) {
        if (S != 1 || !win_cu) {
            rad_step_fail(c_, "qsa: a step wider than the score plane must be one sequence with "
                              "its window bounds (RadBuildCtx::max_mixed_tok)");
            return;
        }
        for (int64_t w0 = 0;;) {
            RAD_ISSUE_N(c_, op_score, PR,
                        brow_slice(w.q, w0, PR, qw), kv_cache(kv_bk, layer),
                        praw2(bkb->block_table, RAD_I32, S, bkb->block_table_pitch),
                        praw(win_cu, RAD_I32, 2), brows(w.nc, S),
                        bcol_at(w.score, 0, c.max_blocks, 0, live_s, PR));
            selection(PR, w0, praw(win_cu, RAD_I32, 2), batch->positions + w0);
            if (w0 + PR >= T) break;
            w0 = w0 + PR + PR <= T ? w0 + PR : T - PR;
        }
        return;
    }
    RAD_ISSUE_N(c_, op_score, T,
                brows(w.q, T), kv_cache(kv_bk, layer),
                praw2(bkb->block_table, RAD_I32, S, bkb->block_table_pitch),
                praw(batch->cu_seqlens, RAD_I32, S + 1), brows(w.nc, S),
                bcol_at(w.score, 0, c.max_blocks, 0, live_s, T));
    selection(T, 0, praw(batch->cu_seqlens, RAD_I32, S + 1), batch->positions);

    /* The pager, on the decode rows (they come first, RadBatch::n_seq_decode, and own the scratch
     * pages). Only with the store open this step: the declare asked for the op, the batch says
     * whether a store exists, and a step with none has no cold entry to rewrite. A step with no
     * decode row has no scratch and nothing to page. A windowed prefill (above) never gets here:
     * its history was promoted before the step. */
    if (op_page && atb->residency_store && atb->residency_slots > 0 && atb->residency_layer_off &&
        atb->residency_scratch &&
        batch->n_tok_decode > 0)
        RAD_ISSUE_N(c_, op_page, batch->n_tok_decode,
                    brows(w.sel, batch->n_tok_decode),
                    /* The store as this layer's fragments see it: the offset is folded in here, on the
                     * host, because the offset table is host memory a device kernel cannot read.
                     * Slot s of this layer starts at store + s * slot_bytes (the operand's extent). */
                    praw2((char*)atb->residency_store + atb->residency_layer_off[layer], RAD_U8,
                          atb->residency_slots, atb->residency_slot_bytes),
                    kv_cache(c.kv_attn, layer),
                    praw2(atb->residency_scratch, RAD_I32, batch->n_tok_decode,
                          atb->residency_scratch_pitch),
                    /* The optional counter (null: not counted; a null operand is an absent one). */
                    praw(atb->residency_paged, RAD_I64, 1));

    /* RADIANCE_DEBUG_QSA: prints the first few selections (nc, nsel, sequ, leading sel and score
     * entries) to tell a divergence from intended dropping (`nc` above `topk`). Synchronises the
     * stream; off unless set. */
    static const bool dbg = std::getenv("RADIANCE_DEBUG_QSA") != nullptr;
    if (dbg) {
        static std::atomic<int> seen{0};
        if (seen.fetch_add(1) < 6) {
            rad_stream_sync(rad_stream(c_));
            std::vector<int32_t> hnc(1), hns(1), hsu(1), hsel(8);
            std::vector<float>   hsc(8);
            rad_memcpy_async(hnc.data(), rad_buf_ptr(c_, w.nc),   4, rad_stream(c_));
            rad_memcpy_async(hns.data(), rad_buf_ptr(c_, w.nsel), 4, rad_stream(c_));
            rad_memcpy_async(hsu.data(), rad_buf_ptr(c_, w.sequ), 4, rad_stream(c_));
            rad_memcpy_async(hsel.data(), rad_buf_ptr(c_, w.sel), 32, rad_stream(c_));
            rad_memcpy_async(hsc.data(), rad_buf_ptr(c_, w.score), 32, rad_stream(c_));
            rad_stream_sync(rad_stream(c_));
            std::fprintf(stderr, "D qsa L%d nc=%d nsel=%d sequ=%d sel[%d %d %d %d] "
                                 "score[%.4g %.4g %.4g %.4g]\n", layer,
                         hnc[0], hns[0], hsu[0], hsel[0], hsel[1], hsel[2], hsel[3],
                         hsc[0], hsc[1], hsc[2], hsc[3]);
        }
    }
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_QSA_H */
