/* kv.h -- the KV block manager (spec §7.2).
 *
 * Paged per KV group; each group has its own block geometry, pool and slot mapping. Block size
 * comes from KVGroupInfo::block_size, taken from the resolved attention kernel's constraints,
 * never from a core constant, so a kernel change needs no core edit.
 *
 * Four state kinds (hybrid models are the normal case):
 *
 *   RAD_KV_FULL    paged, per-token, growing
 *   RAD_KV_WINDOW  paged, per-token, bounded by the window; blocks below the window are recycled
 *   RAD_KV_LINEAR  per-sequence, fixed size: the GDN recurrent state
 *   RAD_KV_CONV    per-sequence, fixed size: a rolling window of conv_width-1 + n_spec entries
 *
 * A RAD_KV_RING group is a RAD_KV_WINDOW whose floor is per sequence and set at runtime (R-SWA,
 * see SeqWindow); it is planned as a WINDOW group with `runtime_win` set, so no kind of its own
 * reaches the plan.
 *
 * The two stateful kinds are also allocated, mapped and freed here, so one allocator decides what
 * a sequence owns. With two, a freed sequence's state slot could still be handed to the next
 * request by the other.
 */
#pragma once
#include "kvstream/mirror.h"
#include "mem/hostarena.h"
#include "mem/pools.h"
#include "rad_core.h"
#include "rad_runtime.h"   /* rad_kv_cold_entry: the cold block-table entry */

#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rad {

/* ------------------------------------------------------------------ group sizing */
/* Layer bundling compared with vLLM. vLLM splits a hybrid model's layers into equal-sized groups
 * and pads each bundle to a multiple of that size with real memory: on a 48 linear / 16 full /
 * 5 drafter model its rule picks size 5, giving 15 groups x 5 = 75 slots for 69 layers and a
 * per-request round-up per group. A least-waste search recovers part of that.
 *
 * Here each state kind is one group with its own pool (spec §7.2), so there is no uniform page
 * size to pad to: padding is zero and the group count is minimal. This struct lets
 * `--debug-placement` print the comparison and lets a test assert it. */
struct BundleComparison {
    std::vector<int64_t> bundles;        /* layers per state kind, as declared */
    int64_t stock_group_size = 0;        /* vLLM's rule: the smallest bundle (with the 1.5x escape) */
    int64_t stock_groups = 0;
    int64_t stock_slots = 0;             /* allocated layer-slots, padding included */
    int64_t patched_group_size = 0;      /* the least-waste group size for the same rule */
    int64_t patched_groups = 0;
    int64_t patched_slots = 0;
    int64_t our_groups = 0;              /* == bundles.size() */
    int64_t our_slots = 0;               /* == sum(bundles); no padding */
    std::string report() const;
};
BundleComparison compare_bundling(const std::vector<int64_t>& bundles);

/* A RAD_KV_* kind's name, for reports and refusals. */
const char* kv_kind_name(int kind);

/* ------------------------------------------------------------------ the runtime window */
/* A sequence's per-sequence window on a RAD_KV_WINDOW group, set at runtime (R-SWA, the
 * "reference + ring" attention of Unlimited-OCR: a full causal prefill over the prompt and image
 * tokens, then a ring of W decode slots right after it, rewritten in place forever).
 *
 * A fixed sliding window (`KVGroupPlan::window_blocks`, gemma4's) keeps the last W tokens: the
 * table starts where the window begins and blocks leave from its front. A runtime window instead
 * retains the whole prefill as the reference and appends into a ring after it:
 *
 *     visible = [0, visible_lo)  U  [visible_lo, visible_lo + W)
 *
 * `visible_lo` is the sequence's own prefill length (image tokens included), so the visible set is
 * per sequence. The reference is never recycled and first_block_pos stays 0 (the table still
 * starts at position 0); only the ring's slot mapping recycles, so RoPE positions keep increasing
 * with the token's own position and a decode query sees the whole reference plus the ring.
 *
 * Blocks are whole, so the ring can start only on a block boundary: it starts at
 * `ring_pos = ceil(visible_lo / block) * block` with `ceil(W / block)` blocks, and the tokens in
 * [visible_lo, ring_pos) stay in the reference's last block (they are the oldest of the tail and
 * are never rewritten). The visible tail is therefore W .. W + block - 1 tokens when W is a block
 * multiple -- with block_size 128 = W the 128-256 token granularity of the plan, exact even when
 * the floor is not aligned -- and up to one block more otherwise, always a superset of the
 * reference implementation's last W. The ring's blocks hold the newest positions in place (no
 * block is allocated past them, so the sequence's bytes are flat once the floor is set):
 *
 *     block index = ring_pos / block + ((pos - ring_pos) / block) % ring_blocks
 *
 * A sequence with a runtime window is excluded from prefix-cache reuse, window snapshots, forking
 * and speculative rollback: all three assume the declare-time window, and the runtime length is
 * not in any cache key (see KVManager::set_window). */
struct SeqWindow {
    bool    set = false;
    int64_t window = 0;       /* W as set */
    int64_t visible_lo = 0;   /* the retained reference ends here */
    int64_t ring_pos = 0;     /* block-aligned first ring token position */
    int64_t ring_blocks = 0;  /* whole blocks the ring holds */
    int64_t ring_span(int64_t block) const { return ring_blocks * block; }
};

/* ------------------------------------------------------------------ per-group plan */
struct KVGroupPlan {
    int32_t     index = -1;            /* position in Program::kv_groups */
    std::string name;
    int         kind = RAD_KV_FULL;
    int64_t     n_layers = 0;

    int64_t     block_size = 0;        /* tokens per block, from the resolved kernel */
    int64_t     bytes_per_block = 0;   /* covers every layer bound to the group */
    int64_t     bytes_per_state = 0;   /* LINEAR/CONV: per sequence, every layer */
    int64_t     conv_slots = 0;        /* CONV: conv_width - 1 + n_spec */
    int64_t     window_blocks = 0;     /* WINDOW: the per-sequence block cap */
    /* WINDOW: sequences may be given a runtime window (declare_runtime_window). Such a group is
     * sized for the whole context plus one ring per sequence instead of being capped at
     * window_blocks, its window is not snapshotted, and a sequence armed with one is bounded by
     * its own reference + ring rather than by window_blocks. */
    bool        runtime_win = false;
    int64_t     runtime_tokens = 0;    /* the W set_window arms a sequence with */
    /* The most ring blocks one sequence needs: ceil(W / block), the reference's fraction d of
     * the ring's first block being frozen in it (see SeqWindow). */
    int64_t ring_blocks_max() const {
        return runtime_tokens > 0 && block_size > 0
                   ? (runtime_tokens + block_size - 1) / block_size : 0;
    }
    /* WINDOW: a checkpoint slot carries this group's window (has_window_snapshot), after any
     * linear state. Set for a trunk group, and for a drafter's group on a model without linear
     * state where the engine saves the snapshot behind the drafter's context pass
     * (set_drafter_window_snapshot). Elsewhere a drafter's window is recomputed by draft refill,
     * so its snapshot would not be read. */
    bool        win_snap = false;
    bool        drafter = false;           /* a drafter's group (KVGroupInfo::drafter) */

    int64_t     n_blocks = 0;          /* paged groups: blocks the group addresses (its context) */
    /* Paged groups: blocks the device pool backs, fewer than `n_blocks` when the group streams
     * (docs/KV-STREAMING.md). Block ids stay dense over `n_blocks`: the allocator, refcounts, block
     * table, prefix cache and every `id < n_blocks` guard count the context. Only addressing uses
     * this: the pool and its layer stride are `n_slots` blocks deep, and the residency map turns a
     * block into a slot. Equal to `n_blocks` otherwise. */
    int64_t     n_slots = 0;
    /* Layer stripes the staged view presents: 0 means `n_layers` (the ring, and every group that
     * does not stream), 1 means one stripe that every layer aliases. A depth rather than a
     * rewritten `layer_slot`, which is also the host mirror's row (docs/KV-STREAMING.md §3.5.1). Set
     * by carve() when the ring is smaller than the context. */
    int64_t     depth = 0;

    /* Window residency (RadKVGroupDecl::residency_window, spec §7.3): the tokens of a sequence's
     * history this group's consumer reads, and the same window in table entries. 0 is no window:
     * every block stays in VRAM. The manager moves no bytes; it names the moves (BlockMove) and
     * the engine runs them, so a window is inert until a residency store is set. */
    int64_t     residency_window = 0;
    int64_t     residency_blocks = 0;
    bool        residency() const { return residency_blocks > 0; }
    /* A tiered group (--kv-recipe `demote LAYOUT`, RAD_KVQ_DEMOTE): its layers' blocks sit in the
     * pool in layout `res_from` and in a store slot in layout `res_to` (RAD_KVQ_* ids), at head_dim
     * `res_head_dim`; a store slot holds `res_frag` bytes of each layer instead of the pool's
     * fragment. The transfers requantise and expand on the card (TierExec::issue_moves, the op
     * `kv_transcode`) and the pager expands (`qsa_page`). res_to -1: untiered, the bytes move as
     * they are. Set by plan_groups() from the recipe; every layer of the group shares the pair,
     * since the plugin keys its groups by it. */
    int         res_from = -1, res_to = -1;
    int64_t     res_head_dim = 0;
    int64_t     res_frag = 0;
    bool        tiered() const { return res_to >= 0; }
    /* One layer's bytes of a block in a store slot, and the whole block's. */
    int64_t     residency_frag() const { return tiered() ? res_frag : layer_bytes_per_block(); }
    int64_t     residency_bytes() const { return n_layers * residency_frag(); }
    /* The pager's scratch pages (spec §7.3): ordinary pool blocks the engine owns, ids
     * [0, scratch_blocks) of the group, never on its free list. A row of the step's decode batch
     * gets `scratch_pitch` = residency_blocks + 1 of them (the selection's topk plus the tail's
     * block), and a step can carry max_seqs x (1 + n_spec) rows. Both stay 0 unless a store is to
     * be opened (--kv-residency-mib), so a window that is declared and not paged costs nothing.
     * They are part of n_blocks (they are carved) and of fixed_ (they are charged before the
     * split), but not of total_blocks(): no sequence can hold them. */
    int64_t     scratch_pitch = 0;
    int64_t     scratch_blocks = 0;
    std::vector<int> layers;           /* model layers bound to the group, as rad_bind_layer_kv */

    /* rad_kv_group_zero: this rank keeps no cache for the group. The plan keeps its position and
     * block count (one block table serves every rank); its costs are zero and nothing is carved.
     * No op on this rank names it (issue refuses a group with no pool bound). */
    bool        zero = false;
    /* rad_kv_group_dense_read: read over the whole history every step, so it cannot stream; while a
     * sibling streams its carve is a floor of max_seqs full contexts (kvstream/carve.h). */
    bool        dense_read = false;
    /* rad_kv_group_replicated: every rank holds the same bytes, so a tier slot carries this
     * group once, in a window at its front that rank 0 writes and every rank reads
     * (TierExec::slot_layout). The pool is carved as for any group. */
    bool        replicated = false;
    int64_t     n_states = 0;          /* LINEAR/CONV: n_seq_states * state_copies */
    /* Physical slots per logical state: 1, or 2 under speculation. A verify runs 1 + n_spec
     * tokens through the recurrence before acceptance is known; libr4d keeps the committed state
     * in copy 0 and the per-token replay factors in copy 1, and folds the accepted prefix in at
     * the next step. The factors (a few hundred floats a head, against a 128x128 state) fit in
     * one slot, so the pool grows 2x rather than (1 + n_spec)x. Copy k of logical slot s is
     * physical slot s + k * n_seq_states. */
    int64_t     state_copies = 1;
    int64_t n_seq_states() const { return state_copies > 0 ? n_states / state_copies : n_states; }
    int64_t     pool_offset = 0;
    int64_t     pool_bytes = 0;
    void*       base = nullptr;

    bool paged() const { return kind == RAD_KV_FULL || kind == RAD_KV_WINDOW; }
    bool stateful() const { return kind == RAD_KV_LINEAR || kind == RAD_KV_CONV; }
    /* Blocks of device pool this group owns: `n_slots`, which is `n_blocks` unless the group
     * streams. Every carve and stride goes through here; every allocation and guard uses
     * `n_blocks`. */
    int64_t     pool_blocks() const { return n_slots > 0 ? n_slots : n_blocks; }
    /* The pool holds fewer blocks than the group addresses; a residency map says which. */
    bool        streamed() const { return paged() && n_slots > 0 && n_slots < n_blocks; }

    /* The pool is layer-major, as the ABI requires: a RAD_OPK_KV operand carries the layer as its
     * offset and resolves to `base + layer * layer_stride` (RadOperand in rad_runtime.h,
     * KVPoolBinding in core/runtime/ctx.h), so a kernel gets one base pointer per layer and
     * indexes it with the block table and slot mapping.
     *
     * Layer i owns [base + i*layer_stride, + layer_stride): pool_blocks() blocks of
     * `bytes_per_block / n_layers` for a paged group, n_states states of
     * `bytes_per_state / n_layers` for a stateful one. */
    int64_t layer_bytes_per_block() const {
        return n_layers > 0 ? bytes_per_block / n_layers : bytes_per_block;
    }
    int64_t layer_bytes_per_state() const {
        return n_layers > 0 ? bytes_per_state / n_layers : bytes_per_state;
    }
    int64_t layer_stride() const {
        return paged() ? pool_blocks() * layer_bytes_per_block()
                       : n_states * layer_bytes_per_state();
    }
};

/* A copy the caller must execute on its copy stream before the destination block is read. The
 * manager has no stream and issues no device copies, since a memcpy on the step path would
 * synchronise (spec §5.4). */
struct BlockCopy {
    int32_t group = -1;
    int32_t src_block = -1;
    int32_t dst_block = -1;
};

/* One block of a residency group moving between VRAM and host memory (see the residency block in
 * KVManager). Named by the manager and run by the engine, as with BlockCopy: the manager owns the
 * state machine, the engine owns the stream. */
struct BlockMove {
    int32_t group = -1;
    int32_t block = -1;
    int32_t host_slot = -1;   /* the arena slot holding (or about to hold) this block */
};

/* A state slot that still holds the previous tenant's state and must be zeroed before first use.
 * The manager names the work and the engine issues it, as with BlockCopy.
 *
 * A paged sequence reads only the blocks its table names, so a recycled block is unreachable
 * until written. A recurrent slot has no table: `gdn_chunk_scan` reads `h0` from the slot and has
 * no operand for "no initial state", so an uncleared slot continues the previous sequence's
 * history. Conv slots carry `has_init` but are cleared too, so one rule covers the pool. */
struct StateClear {
    int32_t group = -1;
    int32_t slot  = -1;
};

/* ------------------------------------------------------------------ manager */
/* The window-residency store's budget split (the engine's bring-up): one store per residency
 * group, of slots `slot_bytes` wide (a block of that group, every rank's shard), a group's blocks
 * `block_size` tokens. The groups demote the same tokens, so each store holds the same tokens and
 * the budget goes in proportion to a group's slot bytes a token. Returns those tokens, the most
 * whose blocks -- floor(tokens / block_size) slots a group -- fit `cap_bytes` together; group g's
 * store is then (tokens / block_size) slots. 0 for no group or a budget too small. */
struct ResidencySplit { int64_t slot_bytes = 0, block_size = 0; };
int64_t residency_split_tokens(int64_t cap_bytes, const std::vector<ResidencySplit>& groups);

class KVManager {
public:
    /* ---------------- window residency (spec §7.3) ----------------
     *
     * A paged group whose consumer reads only part of the history declares a window
     * (RadKVGroupDecl::residency_window). Blocks outside the window may leave VRAM for host memory
     * and give their VRAM back: the pool block returns to the free list and the table entry
     * becomes the cold encoding, rad_kv_cold_entry(slot) (abi/rad_runtime.h), naming the store slot
     * that holds the bytes. A cold block is therefore no pool block at all: it has no refcount, is
     * never indexed by the prefix cache and is never shared with a fork (a shared or cached block
     * is not demoted), so the slot belongs to exactly one table entry and is freed where that
     * entry goes -- free, preempt, rollback -- or when a promotion's copy has landed.
     *
     * Every reader of a block table must therefore treat an entry < 0 as not-a-block-id: it is
     * empty (-1) or cold (<= -2). block_table() returns the raw entries, which is the ABI contract
     * the pager kernels read; a caller that wants pool ids (a fork, a prefix-cache publish)
     * promotes first or stops at the first non-id entry.
     *
     * The manager cannot decide the resident set by itself: what a step reads is device data (QSA
     * picks its blocks in qsa_select). The caller supplies it -- `keep_blocks` for a positional
     * consumer, a list of table entries for a selected one -- and this class owns the state, the
     * arena slots and the accounting.
     *
     * The store is a HostArena per residency group, of that group's `residency_slot_bytes(g)`
     * slots, opened by the engine like the snapshot store, one slot per cold block: a slot holds one
     * block of one group, so a group's own arena sizes it to that group's (layout's) block and a
     * mixed recipe's narrow layouts waste no host bytes. A cold entry's slot id is an index into its
     * group's arena. Under tensor parallelism a slot holds every rank's shard at its own offset
     * (set_residency_store's rank_off), and rank 0 hands out slot ids.
     * The two-argument form puts every residency group in one arena (slots as wide as the widest
     * group's block); the engine opens one per group (the three-argument form). */
    /* RAD_E_STATE if this manager still holds cold entries (or in-flight transfers) of a group in
     * the store it has: their slots would name bytes in a store that is gone. RAD_E_INVAL for a
     * store whose slot cannot hold this rank's part past `rank_off`, or a group with no window. */
    int    set_residency_store(HostArena* store, int64_t rank_off);
    int    set_residency_store(int32_t group, HostArena* store, int64_t rank_off);
    bool   residency_enabled() const {
        for (HostArena* a : res_arena_) if (a) return true;
        return false;
    }
    /* The pager's counter on this rank's card (RadKVGroupBatch::residency_paged), owned by the
     * engine (TierExec::paged_counter); the manager only carries the address to the batch. */
    void     set_residency_paged(int64_t* dev) { res_paged_ = dev; }
    int64_t* residency_paged() const { return res_paged_; }
    /* RADIANCE_DEBUG_KV_TIER=emulate (Engine): the emulation the residency-on output is held to.
     * residency_demote() then names every block the sweep would demote -- outside sink and tail,
     * not shared, not already done -- with host_slot -1 and leaves its table entry and pool block
     * alone; the mover runs the demote -> expand round trip on the block in place
     * (TierExec::issue_moves). A block is round-tripped once in its life, as under residency a
     * tiered block is requantised once (its promotion keeps its origin slot, see res_origin_): the
     * mark goes only when the block is freed. No store slot is taken, so the two agree only while
     * residency's store has room (a full store leaves blocks resident there). */
    void residency_emulate(bool on) { res_emulate_ = on; }
    bool residency_emulating() const { return res_emulate_; }
    bool residency_emulated(int32_t group, int32_t block) const;
    /* Bytes one block of `group` occupies in its slot; the widest of them (what one arena shared
     * by every group needs a rank); and a group's slot's whole size (every rank's part: the open
     * store's slot once one is set, else this rank's offset plus its block, in 256-byte units).
     * The argumentless residency_slot_bytes() is the widest group's, a shared arena's slot. Under a
     * per-layer recipe (--kv-recipe) the groups are layouts, so a block's bytes, and its layers'
     * fragments, are its layout's -- the demote layout's for a tiered group
     * (KVGroupPlan::residency_bytes). */
    int64_t residency_block_bytes(int32_t group) const;
    int64_t residency_shard_bytes() const;
    int64_t residency_slot_bytes(int32_t group) const;
    int64_t residency_slot_bytes() const;
    /* A group's store as the card addresses it, null when none is open (HostArena::open_mapped),
     * and its capacity in slots. */
    void*   residency_device_base(int32_t group) const {
        const HostArena* a = res_store(group);
        return a ? a->device_base() : nullptr;
    }
    int64_t residency_slots(int32_t group) const {
        const HostArena* a = res_store(group);
        return a ? a->capacity() : 0;
    }
    /* What the pager (`qsa_page`) reads for a group, host side. scratch_table() is
     * [scratch_rows(), scratch_pitch] pool ids, row-major: the same every step, since a row's
     * pages are valid for the step that wrote them only and the next step overwrites them.
     * residency_layer_off() is indexed by model layer: the byte offset of that layer's fragment
     * inside a slot of the group's store (this rank's shard included), -1 for a layer the group
     * does not own.
     * All empty for a group with no window or no scratch reserved. */
    int64_t scratch_rows() const { return max_seqs_ * (1 + (n_spec_ > 0 ? n_spec_ : 0)); }
    const std::vector<int32_t>& scratch_table(int32_t group) const;
    const std::vector<int64_t>& residency_layer_off(int32_t group) const;

    /* Demote every block of `group` in `seq`'s table that the window does not cover: the first
     * `keep_head` table entries and the last `keep_tail` stay resident, everything between them
     * moves. Two counts and not one, because a sparse consumer reads two ends of a history and
     * neither is positional: QSA's indexer selects from everywhere, but the first blocks of a
     * conversation are the attention sinks every step scores, and the last are the tail it always
     * attends. A single positional window would demote the sinks and promote them again every
     * step, which is the one thing this must not do.
     *
     * Appends the moves for a caller to run on its copy stream, VRAM -> host, and encodes the table
     * entries cold at once (the sequence's context is untouched: the same positions, the same
     * token count). The pool block is NOT yet free: it is held until the caller reports the copy
     * landed with residency_demote_landed(), because the pool must not hand it to another sequence
     * while the gather still reads it. Marks the table changed from the first demoted entry.
     *
     * A block that is shared (the prefix cache or a forked child holds a reference) is skipped:
     * another sequence's step may read it, and this sequence cannot promote it for that reader. So
     * is the sequence's last block, which is partial and the one an append writes.
     * `demote_skipped` counts both.
     *
     * RAD_E_INVAL on a group with no window or a sequence that does not hold its blocks. A full
     * store leaves the rest resident and reports `skipped`: the caller retries after a sweep.
     *
     * `max_blocks` bounds how many entries this call turns cold, so a scheduler's sweep costs a
     * bounded copy per step; the rest wait for the next sweep. */
    int residency_demote(uint64_t seq, int32_t group, int64_t keep_head, int64_t keep_tail,
                         std::vector<BlockMove>* out, int64_t max_blocks = INT64_MAX);

    /* The copies residency_demote() named have landed: the held pool blocks return to the free
     * list (the VRAM freed) and a later ensure() may hand them to another sequence. */
    void residency_demote_landed();

    /* Promote the named table entries of `group` back before the step that reads them. Each takes
     * a fresh pool block, which becomes the table entry; appends the moves (block = the fresh
     * block, host_slot = the slot to read), host -> VRAM, for the caller to run and make the
     * compute stream wait on. An entry already resident is skipped, so a caller may pass a
     * superset of what it needs. RAD_E_INVAL names an index outside this sequence's table: another
     * sequence's bytes would be served. RAD_E_FULL when the pool has no block for one; the entries
     * promoted before it stay promoted and their moves are in `out`. */
    int residency_promote(uint64_t seq, int32_t group, const std::vector<int64_t>& table_idx,
                          std::vector<BlockMove>* out);

    /* Promote every cold entry of `seq`, in every residency group, as a dense step needs. All or
     * nothing on room: RAD_E_FULL with nothing changed when the pool cannot take them all. */
    int residency_promote_all(uint64_t seq, std::vector<BlockMove>* out);

    /* A table ENTRY (not a block id) is cold: its bytes are in host memory and must be promoted
     * before a kernel reads them. residency_slot() is its store slot, -1 for a pool id or empty. */
    bool    residency_cold(int32_t group, int32_t entry) const;
    int32_t residency_slot(int32_t group, int32_t entry) const;
    /* ---- cold entries held by more than one owner ----
     * A store slot is reference-counted: a cold entry may sit in several sequence tables and in
     * prefix-cache entries at once, and the slot goes back to the store when the last reference
     * does. retain_entry()/release_entry() are retain_block()/release_block() for a table ENTRY --
     * a pool id's block reference, or a cold entry's slot reference -- and are what the prefix
     * cache uses for everything it holds (RAD_E_INVAL for an empty entry). entry_refs() is the
     * count behind an entry (block refcount or slot references; 0 for an empty one), so "does a
     * sequence also hold this" is `entry_refs() > 1` for either kind. */
    int     retain_entry(int32_t group, int32_t entry);
    void    release_entry(int32_t group, int32_t entry);
    int32_t entry_refs(int32_t group, int32_t entry) const;
    /* The entry's bytes in host memory, or null when it is not cold. */
    void*   residency_ptr(int32_t group, int32_t entry) const;
    /* A move residency_demote()/residency_promote() handed out and its landing has not been
     * reported; the transfer path asserts a caller runs exactly what the manager named. */
    bool    residency_move_pending(const BlockMove& m, bool to_host) const;
    /* A store slot's bytes for `group`, for a transfer that already holds the slot id (a promotion
     * clears the block's state before the bytes are back, so residency_ptr() no longer names it). */
    void*   residency_slot_ptr(int32_t group, int32_t slot) const;
    /* The caller ran the moves residency_promote() named. A promotion marks the block resident at
     * once -- a step may issue the restore and then read the block -- and holds the slot until this
     * call gives it back, so a slot is never freed while its bytes are still being read. Called
     * once per batch, after the transfer landed. (Demotions have their own landing call.) */
    void    residency_complete();
    /* Entries of `group` in `seq`'s table that are cold, as table indices. A caller that hands a
     * table to a kernel asserts this is empty for the part the step reads. */
    std::vector<int64_t> residency_cold_entries(uint64_t seq, int32_t group) const;

    struct ResidencyStats {
        int64_t demoted = 0, promoted = 0;         /* blocks, over the manager's life */
        int64_t demoted_bytes = 0, promoted_bytes = 0;
        int64_t demote_skipped = 0;                /* shared (once per decision) or the store was full (each try) */
        int64_t freed_blocks = 0, freed_bytes = 0; /* pool blocks given back by landed demotions: VRAM freed */
        int64_t promoted_misses = 0;               /* a promotion a step needed: the link traffic */
        int64_t repointed = 0;                     /* demotions that moved no bytes: the entry went back to its origin slot */
    };
    const ResidencyStats& residency_stats() const { return res_stats_; }
    /* The same counters per sequence (= per session: the scheduler keys both by request id). A
     * sequence's entry outlives free_sequence(), since a session is listed after its request
     * ends; it exists only once something moved or was skipped for it. Blocks shared with a fork
     * or the prefix cache are never demoted, but a fork that diverged afterwards may have been
     * counted under its parent, so per-sequence numbers can overlap a server total's parts. */
    struct SeqResidency { int64_t demoted = 0, promoted = 0, skipped = 0, link_bytes = 0, freed_bytes = 0; };
    const std::map<uint64_t, SeqResidency>& residency_by_seq() const { return res_seq_; }
    /* The most sessions kept: a server pages for every long request it serves, and an entry per
     * request ever would grow without bound and be copied into every metrics snapshot. Request ids
     * are monotonic, so the oldest go first -- the ones the session view has long dropped. */
    static constexpr size_t kResidencySessions = 4096;
    int64_t residency_scratch_bytes() const;   /* VRAM the pager's scratch pages hold, all groups */
    int64_t residency_host_bytes() const;      /* bytes of the stores in use, every group's */
    int64_t residency_host_cap_bytes() const;  /* the stores' capacity, 0 when unset */

    /* Sizes every group out of one pool, refusing by name and shortfall if the fixed costs alone
     * do not fit. The paged groups are sized to run out at the same total token count; any other
     * split strands memory in whichever group outlasts the others.
     *
     * The budget is whatever `kv_pool` has left (vram_kv), as the budget resolver set it.
     * Linear-state snapshots live in pinned host memory (set_checkpoint_store), so the pool holds
     * only the live cache (spec §6, §7.3). */
    int configure(const std::vector<KVGroupInfo>& groups, const Config& cfg, Pool& kv_pool);

    /* configure() in two steps for the budget resolver. plan_groups() sizes every group from the
     * declaration and run configuration (no pool), and ceiling() gives the cost of a full cache
     * for `ctx` tokens on max_seqs sequences. Pools are allocated before carve(), so the KV pool
     * is sized and capped from these two alone (core/mem/vram_budget.cpp); VRAM above the ceiling
     * would never be addressed. carve() spends the pool it is given. */
    int     plan_groups(const std::vector<KVGroupInfo>& groups, const Config& cfg);
    /* Whether a drafter's WINDOW group rides in the window snapshot too (KVGroupPlan::win_snap).
     * The engine sets it, before plan_groups, where the drafter stores its context right behind
     * the trunk step and the snapshot is saved after that (a block drafter on the card's driver);
     * a snapshot taken before would hold the window without the step's rows. Kept across
     * forget(). */
    void    set_drafter_window_snapshot(bool on) { snap_drafter_ = on; }
    /* Some drafter group rides in the window snapshot (after plan_groups). */
    bool    drafter_window_snapshot() const {
        for (const auto& p : plans_) if (p.win_snap && p.drafter) return true;
        return false;
    }
    int64_t ceiling(int64_t ctx) const;
    /* The context-independent part of every ceiling: recurrent and conv state for max_seqs
     * sequences, plus alignment padding. Set by plan_groups(). */
    int64_t fixed_bytes() const { return fixed_; }
    /* The token count the paged groups settle on for `budget_bytes` of pool (carve()'s split,
     * without carving). One block table serves every rank, so all ranks must hold the same block
     * count, but under an uneven tensor-parallel split their pools differ. The engine asks every
     * rank and passes the least to cap_paged_tokens() before carve(); 0 is no cap.
     * kUnboundedTokens when every paged group on this rank is zero (rad_kv_group_zero): this rank
     * then limits nothing. */
    static constexpr int64_t kUnboundedTokens = INT64_MAX;
    int64_t paged_tokens(int64_t budget_bytes) const { return paged_split(budget_bytes, nullptr); }
    /* Inverse of paged_tokens: the least whole MiB of pool whose paged split reaches `tokens`,
     * searched up to max_mib; -1 if max_mib does not reach it. Pure arithmetic on the plan, used
     * by Engine::plan to size a cheaper rank to the token count of its dearest peer. Exact,
     * because paged_split is monotone in the budget. */
    int64_t pool_mib_for_tokens(int64_t tokens, int64_t max_mib) const;
    void    cap_paged_tokens(int64_t tokens) { tokens_cap_ = tokens > 0 ? tokens : 0; }

    /* ---- KV streaming: KVStreamStore's state and lifecycle under the manager's names, so callers
     * need not include kvstream/mirror.h, which documents each. */
    using StreamMap   = RadKVStream;      /* the ABI's view: the mirror and the ring's maps */
    using StreamCost  = KVStreamCost;
    using StreamStats = KVStreamStats;
    void cap_resident_tokens(int32_t group, int64_t tokens) { stream_.cap(group, tokens); }
    void set_stream_map(int32_t group, const StreamMap& m) { stream_.set_map(group, m); }
    const StreamMap*  stream_map(int32_t group) const { return stream_.map(group); }
    const RadKVStream* stream_view(int32_t group) const {
        return stream_.view(plan(group), group);
    }
    StreamCost  stream_cost() const { return stream_.cost(plans_, max_ctx_, max_seqs_); }
    int  alloc_stream(MemAllocator& host_alloc, Pool& kv_pool) {
        return stream_.alloc(plans_, max_ctx_, max_seqs_, host_alloc, kv_pool);
    }
    int  init_stream(RadStream s) { return stream_.init(plans_, s); }
    void free_stream(MemAllocator& host_alloc) { stream_.release(host_alloc); }
    StreamStats stream_stats() const { return stream_.stats(plans_); }
    void note_stream_traffic(int64_t misses, int64_t bytes) {
        stream_.note_traffic(misses, bytes);
    }
    int     carve(Pool& kv_pool);

    const std::vector<KVGroupPlan>& plans() const { return plans_; }
    const KVGroupPlan* plan(int32_t group) const;
    int32_t n_groups() const { return (int32_t)plans_.size(); }
    std::string report() const;

    /* ---------------- sequences ---------------- */
    int  add_sequence(uint64_t seq);
    bool has_sequence(uint64_t seq) const;
    void free_sequence(uint64_t seq);
    /* Count of sequence frees and preemptions over the manager's life. These are the only
     * refcount drops the prefix cache cannot see, so a change tells it a block it saw shared may
     * no longer be. */
    uint64_t seq_frees() const { return seq_frees_; }

    /* Grow the sequence's mapping to cover `n_tokens` in every group. Unshares a shared partial
     * block first (copy-on-write) and appends the copy request to the pending list.
     * RAD_E_FULL when a group's free list is empty -- the caller preempts and retries.
     *
     * `keep_from` is the position of the first query the coming step runs (the scheduler's
     * n_computed); -1 means the sequence's current end. A WINDOW group recycles only blocks below
     * keep_from - window + 1, not below the window of the step's end, which would drop keys the
     * first rows of a multi-block prefill chunk still read. */
    int  ensure(uint64_t seq, int64_t n_tokens, int64_t keep_from = -1);

    /* A WINDOW group's caps. A sequence holds the window behind the step's first query plus the
     * step's own tokens (at most window_blocks + ceil(max_tok / block) + 1), and one step carries
     * at most max_tok tokens in total, so the pool needs max_seqs windows plus one step. */
    int64_t window_seq_cap(const KVGroupPlan& p) const;
    int64_t window_pool_cap(const KVGroupPlan& p) const;
    /* ceil(max_tok / block): the blocks one step's tokens can fill. */
    int64_t step_blocks(const KVGroupPlan& p) const;

    /* Recycles a committed chunk's window blocks. ensure() keeps a chunk's blocks beside the
     * window and only the next ensure() recycles them, so a prefill waiting for its next chunk
     * would hold a whole chunk of blocks; several such prefills can exhaust the pool. The
     * scheduler calls this when it advances n_computed, passing the new n_computed.
     *
     * A sequence with a runtime window retains its reference and rewrites its ring in place, so
     * nothing below the floor may be recycled and this is a no-op for it. */
    void trim_window(uint64_t seq, int64_t keep_from);

    /* ---------------- the runtime window (R-SWA; see SeqWindow) ---------------- */
    /* Mark a WINDOW group as carrying per-sequence runtime windows of `window_tokens`. Call before
     * plan_groups()/configure(): it changes the plan (the group is sized for the whole context
     * plus one ring a sequence, its window is not snapshotted, and the ring joins the fixed
     * costs), so a group declared after the plan is refused. A fixed-window group is untouched.
     *
     * A group that is not marked still accepts set_window() -- the caller then sizes the pool
     * itself, e.g. by declaring the group's window as the whole context -- and each sequence's
     * capacity is checked at set_window()/ensure() against the blocks actually carved. Once any
     * group is marked, set_window() arms only marked groups: a plain fixed-window group beside a
     * declared runtime window is refused, not silently converted (its pool is carved for the
     * fixed window). */
    int declare_runtime_window(int32_t group, int64_t window_tokens);

    /* Give a sequence a runtime window with `visible_lo` (its prefill length, image tokens
     * included) as the retained reference's end and a ring of `window_tokens` after it. Every
     * paged WINDOW group of the sequence gets the same floor. Call it before the sequence's KV
     * grows past `visible_lo` -- at admission, where the scheduler knows the prompt length -- so
     * the reference is written linearly and no checkpoint can be reserved for it (a snapshot
     * would describe a window the sequence no longer has).
     *
     * RAD_E_UNSUPPORTED  no windowed group, or the fixed/runtime-window mix above;
     * RAD_E_INVAL        a negative floor or a window of 0;
     * RAD_E_STATE        the sequence already holds tokens past the floor;
     * RAD_E_FULL         the reference plus the ring exceeds what this group carved. */
    int set_window(uint64_t seq, int64_t visible_lo, int64_t window_tokens);

    /* The sequence's window in `group`, or false when it has none (a fixed-window group keeps
     * KVGroupPlan::window_blocks and first_block_pos). The step batch maps a ring position with
     * SeqWindow's rule; every other position maps as (pos - first_block_pos) / block_size. */
    bool runtime_window(uint64_t seq, int32_t group, SeqWindow* out) const;
    /* Whether any group has a runtime window set for the sequence. Prefix publish, checkpoint
     * reservation and snapshots are skipped for it (see set_window). */
    bool has_runtime_window(uint64_t seq) const;
    /* The W declare_runtime_window gave the group, 0 if it carries no runtime windows. */
    int64_t runtime_window_tokens(int32_t group) const;

    /* Adopt blocks a prefix-cache hit found. `n_tokens` must be a whole multiple of the group's
     * block size: a partial block is never shared, because the next token written into it would
     * corrupt the cache entry. Bumps each block's refcount. A residency group also accepts cold
     * entries (the cache indexes them): the slot's reference is bumped, and the next dense step
     * promotes them (residency_promote_all). Checked whole before anything is bumped. */
    int  adopt(uint64_t seq, int32_t group, const std::vector<int32_t>& blocks, int64_t n_tokens);

    /* The child shares every whole block of the parent and copies the partial one on first write
     * (spec §7.3). A recurrent state cannot be shared, so the child gets its own slot and the
     * caller must restore a snapshot into it. */
    int  fork(uint64_t parent, uint64_t child);

    std::vector<BlockCopy> take_pending_copies();

    /* Prefix-cache and tier access to blocks. The cache holds a reference on every block it
     * indexes, so a cached block is never in the free list and cannot be handed to another
     * sequence under a cache entry. Nothing else may call these. */
    /* Takes a free block with no sequence attached, for the idle-session tiers: a restore needs
     * somewhere to put bytes before any sequence asks for them, and the prefix cache takes the
     * reference afterwards. The holder must release_block() if the restore fails. Does not
     * recall the loan; it stops at what the cache keeps in hand. */
    int  reserve_block(int32_t group, int32_t* out);
    /* Up to `n` blocks appended to `out`; returns how many. Recalls the loan for all of them up
     * front, since a block-at-a-time recall would drain the slab one buffer at a time. */
    int64_t reserve_blocks(int32_t group, int64_t n, std::vector<int32_t>* out);
    /* Takes one particular block another rank already chose. Block ids are shared across ranks
     * (RadBatch carries one table), so under tensor parallelism rank 0 picks and the others take
     * the same id. Refuses (RAD_E_STATE) a held block or one above the live mark rather than
     * substituting, since a rank restoring into a different block would corrupt only that rank's
     * output. */
    int  reserve_specific_block(int32_t group, int32_t block);
    /* The cache's reference on a block a sequence already holds. The block is marked as the
     * cache's for cache_only_blocks(). */
    int  retain_block(int32_t group, int32_t block);
    /* A reserved block's reference handed to the cache as it stands: marked, not counted again. */
    void cache_reserved_block(int32_t group, int32_t block);
    /* Releases the cache's reference on a marked block, or a reservation's on any other. */
    void release_block(int32_t group, int32_t block);
    int32_t block_refcount(int32_t group, int32_t block) const;
    /* Blocks only the cache holds: marked, at refcount one. This is what an eviction would free.
     * Maintained at every refcount change because it is read under the scheduler's lock, where a
     * walk of the cache would stall the step. */
    int64_t cache_only_blocks(int32_t group) const;

    const std::vector<int32_t>& block_table(uint64_t seq, int32_t group) const;
    int32_t state_slot(uint64_t seq, int32_t group) const;
    int     state_index_row(uint64_t seq, int32_t group, int32_t* out, int max) const;
    int64_t state_index_width(int32_t group) const;

    /* Slots handed to a sequence since the last drain, oldest first. The engine zeroes each on the
     * compute stream before the step that first reads it and then calls clear_state_clears(). */
    const std::vector<StateClear>& state_clears() const { return clears_; }
    void clear_state_clears() { clears_.clear(); }
    int64_t seq_tokens(uint64_t seq) const;

    /* ---------------- accounting ---------------- */
    int64_t free_blocks(int32_t g) const;

    /* Blocks of a group that something holds. Use this for occupancy, not `carve - free`:
     * free_blocks() is bounded by the live mark while the cache lends, so that difference would
     * count lent blocks as occupied. */
    int64_t held_blocks(int32_t g) const;

    /* The carve: the range every block id is addressed against. live_blocks() is what the cache
     * keeps unlent. */
    int64_t total_blocks(int32_t group) const;
    /* One past the highest held block, rounded up to a `held_hist_` bucket (0 when nothing is held):
     * an upper bound on every id a block table can name, which a streamed group's stage copies up
     * to for a step whose own tables it cannot read (kvstream/stage.h, "Volume"). It counts blocks
     * the prefix cache and other sequences hold too. */
    int64_t held_top_blocks(int32_t group) const;
    int64_t free_states(int32_t group) const;

    /* Pool size in bytes, the only unit in which groups can be summed: paged and stateful groups
     * count different objects, and paged groups share one token axis.
     *
     * pool_bytes() excludes lent blocks, which the expert slab reports as its own. Used bytes
     * count held blocks, owned by a sequence or referenced only by the prefix cache; a block
     * stops counting when it reaches the free list. */
    int64_t pool_bytes(int32_t group) const;
    int64_t pool_bytes_used(int32_t group) const;
    /* Bytes the budget carved for the group; the difference from pool_bytes() is lent to the
     * expert slab. */
    int64_t pool_bytes_carved(int32_t group) const;
    int64_t blocks_for(int32_t group, int64_t n_tokens) const;

    /* Whether a sequence of this length fits an empty pool. If not, preemption cannot help and
     * the request is failed rather than deadlocking the engine (spec §17). */
    bool can_ever_fit(int64_t n_tokens) const;

    /* Whether `seq` can grow to `n_tokens` without taking blocks from another sequence. */
    bool can_grow(uint64_t seq, int64_t n_tokens) const;
    /* Drop every sequence's state slot and put back any slot neither free nor held. The
     * scheduler calls this only when nothing is running (IKVManager::release_idle_states). */
    int release_idle_states();

    /* Blocks a paged group can still hand out, including the lendable part. */
    int64_t available_blocks(int32_t group) const;

    /* ---------------- the loan ----------------
     *
     * The paged groups lend unheld blocks to the expert slab in place. The carve fixes and backs
     * every address once and nothing here unmaps it; the loan only lowers how many of a group's
     * blocks the cache may hand out (the live mark). The slab keeps expert units in the blocks
     * above the mark in every layer's stripe, so the MoE GEMM need not read them across the link.
     * No memory is allocated or freed either way, so the loan makes no driver call.
     *
     * The loan is in tokens, counted down from the top of the carve. FULL groups are sized to run
     * out at the same token count, so a loan of T tokens is the top floor(T / block_size) blocks
     * of every FULL group. WINDOW groups are sized per sequence and never lent.
     *
     * A held block (owned by a sequence or indexed by the prefix cache) is never lent, so a loan
     * evicts nothing. The min-heap free list keeps free blocks at the top, where a loan reaches
     * them.
     *
     * The buffer is the free blocks the cache keeps in hand so a request finds room without a
     * recall: the larger of one prefill chunk (the most one step consumes) and the largest recent
     * shortfall. The shortfall term decays, so an idle server settles at the floor. */
    int64_t buffer_bytes() const;

    /* True for the FULL paged groups only. */
    bool    lendable(int32_t group) const;

    /* Tokens the cache can lend now: over the lendable groups, the least distance from the top of
     * the carve down to the highest held block plus the buffer. Rank 0 only (a follower tracks no
     * held blocks). Also decays the buffer's shortfall term, so call it once per arbiter tick. */
    int64_t spare_tokens();

    /* Lends `tokens` from the top, or takes back what a smaller number no longer covers. Lending
     * past a held block is refused (RAD_E_STATE). Taking back always succeeds here; the caller
     * must have moved the slab out of those blocks first. */
    int     set_loan(int64_t tokens);
    int64_t loan_tokens() const { return loan_; }

    /* Recall hook. A request needing more blocks than the cache has in hand calls it with the
     * loan it can still afford, and the slab owner drains to that before returning, so admission
     * is decided against the whole carve. Without a hook, lent blocks are not recalled and the
     * request gets RAD_E_FULL. */
    void    set_reclaim(std::function<int(int64_t tokens)> f) { reclaim_ = std::move(f); }

    /* Copies the leader's loan onto a follower rank. Block ids are shared across ranks and only
     * rank 0 tracks sequences, so a follower takes the leader's mark instead of computing one;
     * its slab occupies the same block ids of its own pool. */
    int     follow(const KVManager& leader);

    /* Blocks of a group the cache may hand out: the carve less the loan. */
    int64_t live_blocks(int32_t group) const;

    /* ---------------- preemption by recompute (spec §7.1, §17) ---------------- */
    /* Frees the sequence's blocks and state slots and resets it to zero tokens. A recurrent
     * state that was not snapshotted cannot be resumed, so the sequence replays from its last
     * checkpoint or from zero. No "needs re-prefill" flag is kept: the scheduler's
     * `r.n_computed = 0` (preempt_back) drives the replay, and seq_tokens() == 0 reflects it. */
    int  preempt(uint64_t seq);

    /* ---------------- speculative rejection (spec §10) ---------------- */
    /* Shrinks the sequence to `n_tokens` and frees the paged blocks past it. A verify writes KV
     * for 1 + n_spec positions and the sampler accepts a prefix; `ensure` is monotone and cannot
     * undo the rest.
     *
     * Paged groups only. The linear state and conv window have already absorbed the rejected
     * tokens: the conv cache is a rolling window of conv_width-1 + n_spec entries, and the kernels
     * handle a rejection by moving the read offset (core/sample/accept.h). Freeing a state slot
     * here would destroy the state that offset points into. Unlike preempt, the sequence keeps
     * its state slots and window base. */
    int  rollback(uint64_t seq, int64_t n_tokens);

    /* The absolute token position block_table(seq, group)[0] covers: zero for a FULL group, the
     * recycled prefix for a WINDOW group. The slot mapping is (pos - first_block_pos) /
     * block_size; without the offset a sliding-window sequence writes to the wrong slot once it
     * passes its window.
     *
     * A sequence with a runtime window (set_window) keeps 0 for the group's whole life: its table
     * is [reference][ring] from position 0 and only the ring's slot index recycles (SeqWindow),
     * so the mapping above holds for the reference and SeqWindow's rule for the ring. */
    int64_t first_block_pos(uint64_t seq, int32_t group) const;

    /* The first table entry changed since the last call; entries below it are unchanged, and
     * kNoDirty means none changed. The call clears the mark, so it has one reader: the batch
     * builder, which stages only the changed tail. An unseen sequence returns 0. */
    int64_t take_table_changes(uint64_t seq, int32_t group) const;
    static constexpr int64_t kNoDirty = INT64_MAX;

    /* Free victims, in the order given, until `seq` can be extended to `n_tokens`.
     *   RAD_OK        room was made; *preempted lists who paid for it
     *   RAD_E_FULL    even an empty pool cannot serve it; fail the request, do not retry
     * The caller supplies the victim order, since priority is the scheduler's policy. */
    int  make_room(uint64_t seq, int64_t n_tokens,
                   const std::vector<uint64_t>& victims,
                   std::vector<uint64_t>* preempted);

    /* ---------------- linear-state checkpoints (spec §7.3) ----------------
     *
     * Snapshots live in pinned host memory: no kernel reads them, they are only copied back into
     * a live state slot. A save is one copy across the link after the step that produced the
     * state, a restore one copy back before the step that resumes from it (a few ms for a
     * 50-100 MiB snapshot).
     *
     * The store is owned by the engine and shared by every rank: a slot holds each rank's shard
     * at its own offset (`rank_off`). Rank 0 hands out slot ids and the other ranks use them.
     * Without a store no slot is handed out and a hybrid model's prefix hits replay their linear
     * layers from zero. `checkpoint_bytes()` is this rank's shard. */
    void    set_checkpoint_store(HostArena* store, int64_t rank_off);
    int64_t checkpoint_bytes() const { return checkpoint_bytes_; }
    int64_t checkpoint_slots() const { return ck_store_ ? ck_store_->capacity() : 0; }
    int64_t free_checkpoint_slots() const { return ck_store_ ? ck_store_->free_slots() : 0; }
    int  alloc_checkpoint(int32_t* slot_out);
    void free_checkpoint(int32_t slot);
    void* checkpoint_ptr(int32_t slot) const;
    /* The store holds `slot`: it was allocated and not given back since. */
    bool  checkpoint_held(int32_t slot) const;

    /* Snapshot and resume a sequence's whole linear state. On a hybrid model a prefix hit carries
     * the attention half in adopted blocks and the recurrent half only through these calls;
     * without them a hit resumes from whatever the slot last held, which still reads as fluent
     * text.
     *
     * The pool is layer-major, so the blob is a gather: for each stateful group in `plans()`
     * order, for each layer, `layer_bytes_per_state()` bytes. Both directions walk it the same
     * way; a checkpoint is only read back by the same build. Only copy 0 (the committed state) is
     * saved; the other copies hold replay factors for unaccepted tokens.
     *
     * Enqueued on `s`, so stream order puts a save after the step that produced the state and a
     * restore before the step that reads it. One pitched copy per stateful group. RAD_E_INVAL on
     * a bad slot or an unknown sequence. */
    int checkpoint_save(const int32_t* state_slots, int n_slots, int32_t ck_slot, RadStream s,
                        const std::vector<int32_t>* win = nullptr);
    int checkpoint_restore(const int32_t* state_slots, int n_slots, int32_t ck_slot, RadStream s,
                           const std::vector<int32_t>* win = nullptr);

    /* ---------------- window snapshots (the sliding-window half of a prefix hit) ----------------
     * A WINDOW group keeps only the blocks its window can still read, so the prefix cache's block
     * lists do not describe it. A prompt extending a cached prefix at position P needs blocks
     * [lo, P/B), lo = max(0, P/B - (window_blocks - 1)), the floor recycle_window keeps for a
     * query at P. A checkpoint slot therefore carries, after the linear state, window_blocks pages
     * per snapshot group (every layer, layer-major rows), restored into fresh blocks seeded at
     * that floor.
     *
     * has_window_snapshot(): this rank's slot carries a window section.
     * window_rows(): from the leader's table, the block ids [lo, P/B) of every non-zero win_snap
     * group in plans() order, each padded with -1 to window_blocks; every rank addresses its own
     * pool with it, since followers keep no tables. False if the sequence no longer holds that
     * range.
     * seed_window(): gives a sequence with an empty table in each snapshot group the blocks
     * [lo, P/B) a restore will fill, before the ensure that extends it. Other WINDOW groups start
     * at the floor ensure sets. */
    bool has_window_snapshot() const { return window_ck_bytes_ > 0; }
    bool window_rows(uint64_t seq, int64_t pos, std::vector<int32_t>* out) const;
    int  seed_window(uint64_t seq, int64_t pos);
    /* The floor and end (block indices) of the window a query at `pos` reads in a group. */
    static int64_t window_lo(const KVGroupPlan& p, int64_t pos) {
        const int64_t b = pos / p.block_size - (p.window_blocks - 1);
        return b > 0 ? b : 0;
    }

    /* Byte address of a state slot, for the slot-mapping builder. Null when the group has no
     * base. */
    void*   state_ptr(int32_t group, int32_t layer, int32_t slot) const;

private:
    /* carve() wraps carve_into() and calls forget() on failure, so a carve that stops part way
     * cannot leave plans_ passing guards for block state that was never sized. plan_groups() also
     * starts from forget(). */
    int  carve_into(Pool& kv_pool);
    /* The token count at which the paged groups run out together for `budget_bytes` of pool,
     * under tokens_cap_, and each group's block count in `blocks` (by group index; a windowed
     * group capped at its window). carve_into() spends what this returns. */
    int64_t paged_split(int64_t budget_bytes, std::vector<int64_t>* blocks) const;
    /* A zero group's share of `tokens`: the block count of the shared table, capped at the window
     * for a windowed group. 0 for kUnboundedTokens, which is not a count. */
    int64_t zero_blocks(const KVGroupPlan& p, int64_t tokens) const;
    void forget();
    /* Turns a WINDOW group into a runtime-window one (declare_runtime_window). Returns false for a
     * group that is not windowed. */
    bool apply_runtime_window(KVGroupPlan& p) const;

    /* One traversal serving both checkpoint directions; see checkpoint_save. */
    int checkpoint_copy(const int32_t* state_slots, int n_slots, int32_t ck_slot, bool save,
                        RadStream s, const std::vector<int32_t>* win);

    struct Seq {
        uint64_t id = 0;
        int64_t  n_tokens = 0;
        std::vector<std::vector<int32_t>> blocks;   /* per group */
        std::vector<int64_t> first_block_pos;       /* WINDOW: token pos of blocks[0] */
        std::vector<SeqWindow> win;                 /* per group: the runtime window, if any */
        std::vector<int32_t> state;                 /* per group, -1 when not stateful */
        /* Per group, the lowest table entry written since the batch builder last staged this
         * sequence. An append or truncation lowers it to where it touched; a shift lowers it to
         * zero; kNoDirty is unchanged. Mutable because take_table_changes() clears it. */
        mutable std::vector<int64_t> dirty_from;
        /* Per group, where the residency sweep resumes: every entry in [head, from) was decided
         * by an earlier residency_demote() with that keep_head -- cold, emulated, or kept because
         * shared -- under the skip epoch `epoch` (res_skip_epoch_). `from` is lowered with
         * dirty_from by every table change (table_changed), so a sweep is O(new entries) a step,
         * not O(context). */
        struct ResCursor { int64_t from = 0, head = -1; uint64_t epoch = 0; };
        std::vector<ResCursor> res;
    };
    /* A table entry of `s` in group `g` from `from` on changed: for the batch builder's upload
     * (dirty_from) and the residency sweep's resume point (res_from). */
    void table_changed(Seq* s, size_t g, int64_t from) {
        s->dirty_from[g] = std::min<int64_t>(s->dirty_from[g], from);
        s->res[g].from   = std::min<int64_t>(s->res[g].from, from);
    }

    int  grab_block(int32_t g, int32_t* out);
    void drop_block(int32_t g, int32_t b);
    /* Where each residency group's bytes sit in its store's slot, and how wide that slot is. Depends
     * on the plans (the carve) and on the store's rank offset, so it is redone when either is set:
     * set_residency_store() may arrive after carve(), which is the engine's order. */
    void size_residency();
    /* Increments a held block's refcount and keeps the cache-only count in step. */
    void add_ref(size_t g, int32_t b);
    int  grab_state(int32_t g, int32_t* out);
    void drop_state(int32_t g, int32_t s);
    Seq* find(uint64_t seq);
    const Seq* find(uint64_t seq) const;
    /* Recycle a WINDOW group's blocks no query at or past `keep_from` can read. A runtime window
     * (SeqWindow) is not trimmed: its reference must stay and its ring rewrites slots in place. */
    void recycle_window(Seq* s, int64_t keep_from);
    /* The table length a runtime-windowed sequence needs to hold `n_tokens`: the reference's
     * blocks, then the ring's up to its whole span. */
    static int64_t window_table_len(const KVGroupPlan& p, const SeqWindow& w, int64_t n_tokens);

    std::vector<KVGroupPlan> plans_;
    std::vector<std::vector<int32_t>> free_blocks_;   /* per group */
    std::vector<std::vector<int32_t>> refcount_;      /* per group, per block */
    /* Per group, per block: the prefix cache holds one of the references. A free block is never
     * marked. `cache_only_` is, per group, how many marked blocks are at a count of one. */
    std::vector<std::vector<uint8_t>> cached_;
    std::vector<int64_t>              cache_only_;
    std::vector<std::vector<int32_t>> free_states_;   /* per group */
    std::unordered_map<uint64_t, Seq> seqs_;
    std::vector<BlockCopy> pending_;

    HostArena* ck_store_ = nullptr;                   /* set_checkpoint_store */
    int64_t   ck_rank_off_ = 0;
    int64_t   checkpoint_bytes_ = 0;
    int64_t   window_ck_bytes_ = 0;   /* the window section of checkpoint_bytes_ */
    std::vector<StateClear> clears_;                  /* grabbed slots the engine must zero */

    /* ---- window residency (see the public block) ----
     *
     * The cold state lives in the table entry itself (rad_kv_cold_entry(slot)), so there is no
     * per-block array to keep in step with it. A cold entry owns its store slot; drop_entry() is
     * the one place a table entry goes away, and it returns the slot (cold) or the block
     * reference (resident), so a preempt, a rollback and a free_sequence leak neither. */
    /* Per group (by group index; a set store survives a re-plan, as the engine owns it): the
     * group's store, this rank's offset in its slot, and the slot's whole width. */
    std::vector<HostArena*> res_arena_;
    std::vector<int64_t>    res_rank_off_;
    std::vector<int64_t>    res_slot_bytes_;
    HostArena* res_store(int32_t g) const {
        return g >= 0 && (size_t)g < res_arena_.size() && res_arena_[(size_t)g] &&
                       res_arena_[(size_t)g]->enabled() ? res_arena_[(size_t)g] : nullptr;
    }
    /* RAD_OK when `store` may become group g's at `rank_off` (see set_residency_store). */
    int     res_store_check(int32_t g, const HostArena* store, int64_t rank_off) const;
    int64_t*   res_paged_ = nullptr;
    int64_t    res_cold_ = 0;                         /* cold entries held across all tables */
    /* Blocks a demotion encoded cold whose copy has not been reported landed: still referenced
     * (refcount 1, in no table) so the pool cannot reuse them under the gather. */
    struct Demoting { int32_t group, block, slot; uint64_t seq; };
    std::vector<Demoting> res_demoting_;
    /* Drops one table entry: a pool id's reference, or a cold entry's slot. */
    void drop_entry(int32_t g, int32_t entry);
    /* Slot references (see retain_entry): per group, indexed by slot id in the group's store,
     * grown on demand. res_slots_ counts, per group, the slots with a reference, which is what
     * set_residency_store() looks at, since the cache holds slots no table does. */
    std::vector<std::vector<int32_t>> res_refs_;
    std::vector<int64_t>              res_slots_;
    int32_t slot_new(int32_t g);              /* a store slot at one reference, -1 when full */
    void    slot_ref(int32_t g, int32_t slot);
    void    slot_drop(int32_t g, int32_t slot);
    int32_t slot_refs(int32_t g, int32_t slot) const {
        return g >= 0 && (size_t)g < res_refs_.size() && slot >= 0 &&
                       (size_t)slot < res_refs_[(size_t)g].size()
                   ? res_refs_[(size_t)g][(size_t)slot] : 0;
    }
    /* Where a promoted block's bytes came from, per group: block -> slot, while the slot may still
     * hold identical bytes. Each entry owns a slot reference, so the slot cannot be reused under
     * it. Set when the slot has another holder (the cache), or the group is tiered: re-demoting a
     * tiered block from its expanded bytes would requantise it a second time, so its slot stays
     * until the block is written or freed and a re-demotion points back at it. An untiered slot
     * this table alone held is freed at once, as before. Ended by a write into the block (ensure, rollback), by the block
     * going free, or by a re-demotion, which turns the reference into the table entry's. */
    std::vector<std::unordered_map<int32_t, int32_t>> res_origin_;
    /* The emulation's blocks already round-tripped, per group (residency_emulate). */
    bool res_emulate_ = false;
    std::vector<std::unordered_set<int32_t>> res_emul_;
    void forget_origin(int32_t g, int32_t b);
    /* Gives back every slot this manager holds in the store (a re-plan, which keeps the store). */
    void release_residency();
    /* Slots a promotion released from a block and no transfer has finished reading yet, as
     * (group, slot). residency_complete() frees them. */
    std::vector<std::pair<int32_t, int32_t>> res_pending_;
    ResidencyStats res_stats_;
    /* Bumped when a block stops being shared (drop_block leaves it one holder and no cache
     * mark): an entry the sweep kept because it was shared may now move, so every resume point
     * decided under an older epoch is void. */
    uint64_t res_skip_epoch_ = 0;
    /* Per-session counters, oldest request id first, at most kResidencySessions (seq_res()). */
    std::map<uint64_t, SeqResidency> res_seq_;
    SeqResidency& seq_res(uint64_t id);
    void*     pool_base_ = nullptr;
    BundleComparison bundling_;
    std::string report_;
    int64_t   max_seqs_ = 0;
    int64_t   n_spec_ = 0;
    std::vector<std::vector<int32_t>> scratch_;       /* per group, see scratch_table() */
    std::vector<std::vector<int64_t>> res_layer_off_; /* per group, see residency_layer_off() */
    int64_t   max_ctx_ = 0;
    int64_t   max_tok_ = 0;      /* --max-num-batched-tokens; the buffer floor is one of these */
    bool    snap_drafter_ = false;   /* set_drafter_window_snapshot */
    /* plan_groups() computes these; carve() spends against them and ceiling() reports on them. */
    int64_t   fixed_ = 0;      /* the non-negotiable floor: states, conv slots, padding */
    int64_t   ckpts_ = 0;      /* --checkpoint-slots, kept for the warning on a model with none */
    int64_t   tokens_cap_ = 0; /* cap_paged_tokens(): the least any rank can carve; 0 is none */
    /* KV streaming's state by group handle (kvstream/mirror.h); empty when nothing streams. */
    KVStreamStore stream_;

    /* declare_runtime_window(): group -> W. Survives forget(), which plan_groups() calls at its
     * start: it is a declaration about the program, not about a plan. */
    std::unordered_map<int32_t, int64_t> runtime_decl_;

    /* ---- the loan (see the public block) ----
     *
     * `live_` is the carve less the loan. Every carved id stays in `free_blocks_`; ids at or
     * above live_ are just not handed out. The free list is a min-heap so the lowest free id goes
     * out first: held blocks pack at the bottom and the top, the only part a loan reaches, drains.
     * A stack would hand out the most recently freed id, often a high one. The heap also makes the
     * gate a single check: if any free id is below live_, the smallest one is.
     *
     * `held_hist_` counts held blocks per bucket, so finding the highest held block is a walk of
     * a few thousand counters, not a refcount scan. It bounds how far a loan may reach, so it is
     * maintained on the allocation path; a stale value would lend a held block. */
    /* The buffer per group, in blocks. `chunk_blocks_` is the floor (one prefill chunk, fixed at
     * plan time). `short_peak_` is the largest recent admission shortfall, raised by grow_for and
     * decayed by spare_tokens().
     *
     * Only the shortfall is recorded, never the recall: grow_for recalls shortfall plus buffer,
     * and feeding that back would grow the buffer from its own value until it reached the carve. */
    std::vector<int64_t>  chunk_blocks_;
    std::vector<int64_t>  short_peak_;
    Pool*    pool_ = nullptr;
    int64_t  loan_ = 0;                     /* tokens lent from the top of every lendable group */
    uint64_t seq_frees_ = 0;
    std::function<int(int64_t)> reclaim_;
    std::vector<int64_t>  live_;            /* per group: blocks the cache may hand out */
    std::vector<int64_t>  free_usable_;     /* per group: free ids below live_ */
    std::vector<int64_t>  bucket_blocks_;   /* per group: blocks one held_hist_ bucket covers */
    std::vector<std::vector<int32_t>> held_hist_;   /* per group, per bucket: held blocks */

    int64_t  buf_blocks(size_t i) const;
    /* One past the highest held block, rounded up to a bucket. */
    int64_t  held_top(size_t i) const;
    /* The blocks a loan of `tokens` takes off the top of group i. */
    int64_t  lent_blocks(size_t i, int64_t tokens) const;
    /* Moves every lendable group's mark to a loan of `tokens`, unchecked (set_loan checks). */
    void     apply_loan(int64_t tokens);
    int  grow_for(int32_t g, int64_t blocks);
    void note_held(int32_t g, int32_t b, int delta);
    bool      configured_ = false;
    static const std::vector<int32_t> kNoBlocks;
};

}  /* namespace rad */
