/* mem_iface.h -- the seam between core/sched and core/mem.
 *
 * The scheduler binds to pure-virtual mirrors of core/mem's `KVManager` (kv.h) and `PrefixCache`
 * (prefix.h) instead of the classes themselves:
 *
 *   - tests/sched_test.cpp can drive the scheduler against an arbitrarily small block pool, the
 *     only way to test admission refusal and preemption;
 *   - the cost is one indirect call per sequence per step, not per token.
 *
 * `KVBinding` / `PCBinding` at the bottom adapt the concrete classes. `CacheHit` comes from
 * core/mem/prefix.h; do not redefine it here (ODR).
 */
#pragma once
#include <functional>
#include "mem/prefix.h"
#include "rad_core.h"
#include "mm/processor.h"
#include "sched/metrics.h"

#include <vector>

namespace rad {

/* ------------------------------------------------------------------ the KV manager */
/* Keyed by sequence id (the request id). The manager owns the block tables; the scheduler does
 * not read Request::blocks, so there is one owner and a freed state slot cannot be handed out
 * while its previous sequence still reads it. */
struct IKVManager {
    virtual ~IKVManager() = default;

    virtual int  add_sequence(uint64_t seq) = 0;
    virtual bool has_sequence(uint64_t seq) const = 0;
    virtual void free_sequence(uint64_t seq) = 0;

    /* Grow the sequence's mapping to cover `n_tokens` in every group. RAD_E_FULL when a group's
     * free list is empty; the caller preempts a victim and retries. */
    virtual int  ensure(uint64_t seq, int64_t n_tokens) = 0;
    /* As ensure(), for a step whose first query is at `keep_from` (the request's n_computed): a
     * window group keeps every block that query can still read (KVManager::ensure). The default
     * suits a manager with no windowed group. */
    virtual int  ensure_from(uint64_t seq, int64_t n_tokens, int64_t keep_from) {
        (void)keep_from;
        return ensure(seq, n_tokens);
    }
    /* After a committed step, a window group frees what no later query can read
     * (KVManager::trim_window). */
    virtual void trim_window(uint64_t seq, int64_t keep_from) { (void)seq; (void)keep_from; }
    /* R-SWA (KVManager::declare_runtime_window): the W this group arms a sequence with, 0 when it
     * carries no runtime windows. The scheduler arms every sequence it admits with
     * set_window(seq, prompt length, W); a manager with no runtime window refuses it and the
     * scheduler never asks. */
    virtual int64_t runtime_window_tokens(int32_t group) const { (void)group; return 0; }
    virtual int  set_window(uint64_t seq, int64_t visible_lo, int64_t window_tokens) {
        (void)seq; (void)visible_lo; (void)window_tokens;
        return RAD_E_UNSUPPORTED;
    }
    /* The sequence's runtime window in `group`, if it has one (KVManager::runtime_window). A
     * sequence with one is excluded from prefix publish, checkpoint snapshots and speculation:
     * the window is runtime data and the ring rewrites its slots in place. */
    virtual bool runtime_window(uint64_t seq, int32_t group, SeqWindow* out) const {
        (void)seq; (void)group; (void)out;
        return false;
    }

    virtual bool has_runtime_window(uint64_t seq) const { (void)seq; return false; }
    /* Window snapshot of a prefix hit (KVManager::has_window_snapshot / seed_window). The
     * defaults mean no hit on a windowed trunk is ever resumed. */
    virtual bool has_window_snapshot() const { return false; }
    virtual int  seed_window(uint64_t seq, int64_t pos) { (void)seq; (void)pos; return -1; }

    /* Window residency (spec §7.3): the blocks a sparse consumer's window does not cover, moved to
     * host memory and back, and the store they live in. All zero on a manager with no store, which
     * is every server that does not pass --kv-residency-mib. */
    struct Residency {
        int64_t demoted = 0, promoted = 0, skipped = 0, misses = 0;
        int64_t host_bytes = 0, cap_bytes = 0;
        int64_t freed_bytes = 0;     /* VRAM given back by landed demotions */
        int64_t link_bytes = 0;      /* bytes over the link, both directions */
        int64_t scratch_bytes = 0;   /* VRAM the pager's scratch pages hold (reserved, not traffic) */
        int64_t paged_in = 0;        /* cold entries the pager rewrote, from the card's counter */
        int64_t paged_hits = 0;      /* of them, served by the page cache without a copy */
    };
    virtual Residency residency() const { return Residency{}; }
    /* The same per request id, which is the session id; an overlapping view (see SessionStat). */
    using SeqResidency = rad::SeqResidency;
    virtual std::vector<SeqResidency> residency_seqs() const { return {}; }

    /* What the scheduler's sweep and dense-step promotion ask of the manager (KVManager's
     * residency block). Window tokens of a group the sweep may demote in: 0 for every group
     * without a window and for every manager whose store is closed, which is what makes
     * --kv-residency-mib 0 inert here. */
    virtual int32_t n_groups() const { return 0; }
    virtual int64_t residency_window(int32_t group) const { (void)group; return 0; }
    /* Demote at most `max_blocks` of `seq`'s blocks in `group` outside the first `keep_head` and
     * last `keep_tail` table entries; the moves are appended for the caller to run. */
    virtual int  residency_demote(uint64_t seq, int32_t group, int64_t keep_head, int64_t keep_tail,
                                  int64_t max_blocks, std::vector<BlockMove>* out) {
        (void)seq; (void)group; (void)keep_head; (void)keep_tail; (void)max_blocks; (void)out;
        return RAD_E_INVAL;
    }
    /* Demote at most `max_blocks` blocks only the prefix cache holds (idle conversations); the
     * moves are appended like residency_demote()'s. Returns how many were appended. */
    virtual int64_t residency_demote_cached(int64_t max_blocks, std::vector<BlockMove>* out) {
        (void)max_blocks; (void)out;
        return 0;
    }
    /* The issued copies have landed: demoted pool blocks go free, promoted slots are given back. */
    virtual void residency_demote_landed() {}
    virtual void residency_complete() {}
    /* Every cold entry of `seq` back in a pool block (RAD_E_FULL, nothing changed, when the pool
     * cannot take them all). */
    virtual int  residency_promote_all(uint64_t seq, std::vector<BlockMove>* out) {
        (void)seq; (void)out;
        return RAD_OK;
    }

    /* What the batch hands the pager (RadKVGroupBatch::residency_*) for a group on a rank: the
     * store as the card addresses it, its slot size, the per-layer fragment offsets (host array,
     * stable for the engine's life) and the scratch-page table, [scratch_rows, pitch] pool ids.
     * All zero for a group with no window or a closed store -- and then the batch fields are too.
     * scratch_pitch() is asked at init to size the batch's staging, before the store is known to
     * be open, so it is the reservation's, not the store's. */
    struct Pager {
        void*          store = nullptr;
        int64_t        slot_bytes = 0;
        int64_t        slots = 0;
        const int64_t* layer_off = nullptr;
        const int32_t* scratch = nullptr;
        int64_t*       paged = nullptr;   /* the pager's counter on this rank's card, or null */
        /* The page cache (RadKVGroupBatch::residency_tags): this rank's planes of the group, their
         * rows and bytes, its hit counter, and the leader's residency_epoch(). */
        int32_t*       tags = nullptr;
        int64_t        tags_rows = 0;
        int64_t        tags_bytes = 0;
        int64_t*       hits = nullptr;
        int64_t        epoch = 0;
    };
    virtual Pager   pager(int32_t group, int rank) const { (void)group; (void)rank; return {}; }
    virtual int64_t scratch_pitch(int32_t group) const { (void)group; return 0; }

    /* Adopt blocks a prefix-cache hit found. `n_tokens` must be a multiple of the group's block
     * size; a shared partial block would be corrupted by the next token written into it. */
    virtual int  adopt(uint64_t seq, int32_t group, const std::vector<int32_t>& blocks,
                       int64_t n_tokens) = 0;

    /* Free the blocks and mark the sequence for re-prefill. */
    virtual int  preempt(uint64_t seq) = 0;

    /* Whether a sequence of this length fits an empty pool. If not, the request fails at once
     * instead of deadlocking the engine (spec §17). */
    virtual bool can_ever_fit(int64_t n_tokens) const = 0;

    /* Whether `seq` can be grown to `n_tokens` without taking blocks from another sequence.
     * A sequence the manager does not know is not a refusal: the caller adds it first. */
    virtual bool can_grow(uint64_t seq, int64_t n_tokens) const = 0;
    /* Return state slots no running sequence holds, and any slot neither free nor held.
     * The caller guarantees nothing is running. A slot held by a zeroed or forgotten sequence
     * would otherwise keep an overdue preempted prefill waiting forever (spec §17). Returns
     * how many slots were put back. A manager with no state leaves this as a no-op. */
    virtual int release_idle_states() { return 0; }

    /* How many blocks a paged group can still hand out, including those lent to the expert slab
     * and recallable on demand. */
    virtual int64_t available_blocks(int32_t group) const = 0;

    virtual const std::vector<int32_t>& block_table(uint64_t seq, int32_t group) const = 0;
    virtual int32_t state_slot(uint64_t seq, int32_t group) const = 0;

    /* The sequence's state_index row: normally one entry (its state slot), plus any scratch
     * slots the group's kernel asked for. The manager owns the column-to-slot mapping; the
     * scheduler stages the row without reading it. Returns the width written (0 for a paged
     * group), at most `max`. */
    virtual int state_index_row(uint64_t seq, int32_t group, int32_t* out, int max) const = 0;
    virtual int64_t state_index_width(int32_t group) const = 0;

    virtual int64_t free_blocks(int32_t group) const = 0;
    virtual int64_t total_blocks(int32_t group) const = 0;
    /* Blocks held by anything, as opposed to free_blocks (what can be handed out now). On an
     * elastic pool they differ by what the cache released (KVManager::held_blocks). */
    virtual int64_t held_blocks(int32_t group) const = 0;
    /* KVManager::held_top_blocks; a manager that cannot answer says the whole row. */
    virtual int64_t held_top_blocks(int32_t group) const { (void)group; return INT64_MAX; }

    /* The pool in bytes, the only unit that sums across groups. Asked of the manager because its
     * plan knows whether a group counts blocks or states. */
    virtual int64_t pool_bytes(int32_t group) const = 0;
    virtual int64_t pool_bytes_used(int32_t group) const = 0;
    /* What the carve gave the group; an elastic pool may have released part of it. */
    virtual int64_t pool_bytes_carved(int32_t group) const = 0;

    /* A streamed group's RadKVStream for the step batch, or null for a resident group; its addresses
     * are fixed from bring-up. */
    virtual const RadKVStream* stream_view(int32_t group) const { (void)group; return nullptr; }
    /* The resolve's cumulative traffic and the rings' depth, for the metrics; zero without
     * streaming. */
    virtual int64_t  stream_misses() const { return 0; }
    virtual int64_t  stream_bytes_in() const { return 0; }
    virtual int64_t  stream_resident_blocks() const { return 0; }

    /* Tokens one block covers, or 0 for a stateful group. */
    virtual int64_t block_tokens(int32_t group) const = 0;
    /* A checkpoint slot carries this group's window (KVGroupPlan::win_snap): a prefix hit, which
     * resumes from such a slot, restores it whole. */
    virtual bool window_snapshot(int32_t group) const { (void)group; return false; }

    /* Speculative rejection rollback (spec §10). A verify writes KV for 1 + n_spec positions and
     * the sampler accepts a prefix; this shrinks the sequence to `n_tokens` and frees blocks past
     * it, which the monotone ensure() cannot. Only the slot table changes: linear and conv groups
     * roll back by read offset inside their kernels. */
    virtual int  rollback(uint64_t seq, int64_t n_tokens) = 0;

    /* Absolute token position that block_table(seq, group)[0] covers: zero for a full group,
     * non-zero for a window group once blocks below the window are recycled. The step batch maps
     * a slot as `(pos - first_block_pos) / block_size`.
     *
     * A sequence with a runtime window (runtime_window) keeps first_block_pos 0 even once the ring
     * recycles: its table is [reference][ring] from position 0 and the ring's positions map
     * modulo ring_blocks (SeqWindow), so only the slot index recycles, never the table. */
    virtual int64_t first_block_pos(uint64_t seq, int32_t group) const = 0;

    /* Index of the first entry of block_table(seq, group) changed since the previous call;
     * INT64_MAX if none.
     * Reading clears it, so the batch builder, which keeps the device copy and stages only the
     * changed tail, must be the only caller. 0 is always correct (restage the whole table). */
    virtual int64_t take_table_changes(uint64_t seq, int32_t group) const = 0;

    /* Whether the snapshot store still holds `slot`. The scheduler checks before building a
     * restore step, since a released slot may hold another session's state. */
    virtual bool checkpoint_held(int32_t slot) const = 0;
};

/* ------------------------------------------------------------------ the prefix cache */
/* What IPrefixCache::restore did: brought back every entry the tiers hold for the prompt (or there
 * were none), brought back part of them because the pool could not hold the chain, or, asked for
 * the whole chain only, left it on the host. */
enum class RestoreResult : uint8_t { Whole, Cut, Declined };

struct IPrefixCache {
    virtual ~IPrefixCache() = default;

    /* Bring back whatever the idle tiers hold for this prompt, before the lookup. The engine
     * performs it (it allocates blocks and runs a device scatter). A failure only leaves a
     * shorter prefix, which is always safe. With `whole_only`, a chain the pool cannot be made to
     * hold whole is left on the host (RestoreResult::Declined). */
    virtual RestoreResult restore(const Request&, bool whole_only) {
        (void)whole_only;
        return RestoreResult::Whole;
    }
    /* A step's admissions begin; the last step's lookup results have been consumed. */
    virtual void begin_step() {}
    /* Keep this waiting request's chain in VRAM until release(). Called when admission is refused
     * after restore(), so the restored chain is not moved out again before the next attempt. */
    virtual void hold(const Request&) {}
    /* The request finished, failed or was cancelled and holds nothing any more. */
    virtual void release(const Request&) {}

    /* Longest cached prefix, plus the checkpoint covering the linear state for a prefix of it.
     * `blocks_out` receives block ids per cached group, in cached_groups() order, for
     * IKVManager::adopt: n_attn_tokens over the group's own block size of them. `cap`, when not
     * null, ends the hit early (HitCap, core/mem/prefix.h). */
    virtual CacheHit lookup(const Request& r, std::vector<std::vector<int32_t>>* blocks_out,
                            const HitCap* cap) const = 0;
    virtual const std::vector<int32_t>& cached_groups() const = 0;
    /* The drafter's refill window (Scheduler::init): retention keeps a snapshot that deep below a
     * session's tip, so a refill-capped hit has one to end on (PrefixCache::set_drafter_floor). */
    virtual void set_drafter_floor(int64_t) {}

    /* Publish what this sequence computed so a later request sharing the prefix can adopt it.
     * `blocks[i]` is the block table of cached_groups()[i]; `n_tokens` is how many tokens are
     * backed. Takes a KV reference on every newly indexed block, so call it before the sequence's
     * own blocks are released. */
    virtual int insert(const Request& r, const std::vector<std::vector<int32_t>>& blocks,
                       int64_t n_tokens) = 0;

    /* Release the least recently used entries and their KV references. The scheduler tries this
     * before preempting a running request. */
    virtual int64_t evict(int64_t n_entries) = 0;
    /* Free up to `n_entries` for the reserve without losing anything a later turn needs. Unlike
     * evict(), it skips entries a live sequence still holds (no block would be freed) and entries
     * whose only copy is in VRAM. An entry is one block in each cached group, or more in a group
     * whose blocks are shorter than the entry (PrefixCache::blocks_per_entry). Returns the number
     * of entries freed. */
    virtual int64_t trim(int64_t n_entries) = 0;

    /* Drop every entry in every tier. Implementations with idle tiers must override this: there,
     * evict() keeps an entry whose host copy has landed. Returns the number of entries dropped. */
    virtual int64_t reset() {
        int64_t dropped = 0;
        for (int guard = 0; guard < 65536; ++guard) {
            const int64_t n = evict(1024);
            if (n <= 0) break;
            dropped += n;
        }
        return dropped;
    }

    /* Entries evicted, cumulative. Once the pool fills past the scheduler's reserve (a quarter of
     * it), the cache evicts prefixes about to be reused; this count rises before the hit rate
     * falls. */
    virtual int64_t evictions() const = 0;
    virtual int64_t ckpt_evictions() const = 0;
    /* ckpt_evictions by reason, and reservations that found nothing to give up
     * (PrefixCache::Stats). */
    struct CkptCounts { int64_t retired = 0, lru = 0, refused = 0; };
    virtual CkptCounts ckpt_counts() const { return {}; }
    /* Entries expired by --prefix-cache-ttl; not counted as evictions. */
    virtual int64_t expired() const { return 0; }
    virtual int64_t ckpt_expired() const { return 0; }
    /* Entries and snapshots a client released on purpose (release_session). */
    virtual int64_t released() const { return 0; }
    virtual int64_t ckpt_released() const { return 0; }

    /* Forget one stored conversation now, by the client's label or the engine's session id
     * (PrefixCache::release_session). RAD_E_UNSUPPORTED from a cache without a session table. */
    virtual int release_session(const std::string& /*label*/, uint64_t /*id*/,
                                int64_t /*keep_tokens*/, SessionRelease* /*out*/) {
        return RAD_E_UNSUPPORTED;
    }

    /* Tokens of context held only by the cache right now. Blocks a running sequence also reads
     * are excluded. Separates a working cache from a leak in the pool figures. */
    virtual int64_t held_tokens() const = 0;

    /* The next position strictly after `pos` at which the scheduler must split a chunk so that a
     * checkpoint lands on a boundary. */
    virtual int64_t next_checkpoint_after(int64_t pos) const = 0;

    /* Reserve the snapshot slot the scheduler is about to write. Retention is budgeted (§7.3: on
     * Qwen3.5-27B a snapshot is 144 MiB, and a 128K session at the default interval is 64 of them,
     * about 9 GiB), so RAD_E_FULL is ordinary: no checkpoint is written at that boundary and a
     * later hit replays from the previous one. */
    virtual int reserve_checkpoint(const Request& r, int64_t pos, int32_t* slot_out) = 0;

    /* The reserved state is now in its slot. Reservation happens at planning, before the step
     * computes the state, so lookup does not offer the slot until this is called. */
    virtual void commit_checkpoint(const Request& r, int64_t pos) = 0;
};

/* ------------------------------------------------------------------ block geometry */
/* What the step batch needs to turn a token position into a KV slot, read from the declared
 * program: KVGroupInfo::block_size comes from the resolved attention kernel's constraints
 * (§7.2). */
struct KVGeom {
    struct G {
        int     kind = RAD_KV_FULL;
        int64_t block_size = 1;
        int64_t window = 0;
        bool    paged = true;
        /* Columns in this group's state_index row. The manager decides it, so KVGeom::of leaves
         * it at 1 and the scheduler fills it in from the configured pool. */
        int64_t sidx_width = 1;
        /* Columns of the pager's scratch-page table (IKVManager::scratch_pitch); 0 when the group
         * reserves none. Filled in by the scheduler beside sidx_width. */
        int64_t scratch_pitch = 0;
    };
    std::vector<G> g;

    static KVGeom of(const std::vector<KVGroupInfo>& groups) {
        KVGeom k;
        k.g.reserve(groups.size());
        for (const KVGroupInfo& gi : groups) {
            G e;
            e.kind = gi.decl.kind;
            e.paged = (e.kind == RAD_KV_FULL || e.kind == RAD_KV_WINDOW || e.kind == RAD_KV_RING);
            e.block_size = gi.block_size > 0 ? gi.block_size : 1;
            e.window = gi.decl.window;
            k.g.push_back(e);
        }
        return k;
    }

    /* Flat KV slot for absolute token position `pos`, given the sequence's block table for this
     * group and the position its first block covers. */
    int32_t slot(int gi, const std::vector<int32_t>& bt, int64_t first_pos, int64_t pos) const {
        const G& e = g[(size_t)gi];
        int64_t bi = (pos - first_pos) / e.block_size;
        if (bi < 0 || bi >= (int64_t)bt.size()) return -1;
        return (int32_t)(bt[(size_t)bi] * e.block_size + (pos % e.block_size));
    }

    /* The same slot for a runtime window (SeqWindow): the reference maps linearly (first_pos is
     * 0 there), and a position at or past the ring's start lives in block
     * ring_pos/block + ((pos - ring_pos)/block) % ring_blocks, which is what rewrites the ring's
     * slots in place. ring_pos is block-aligned, so `pos % block` stays the in-block offset. */
    int32_t slot_ring(int gi, const std::vector<int32_t>& bt, int64_t first_pos, int64_t pos,
                      const SeqWindow& w) const {
        const G& e = g[(size_t)gi];
        int64_t bi;
        if (w.set && pos >= w.ring_pos) {
            bi = w.ring_pos / e.block_size + ((pos - w.ring_pos) / e.block_size) % w.ring_blocks;
        } else {
            bi = (pos - first_pos) / e.block_size;
        }
        if (bi < 0 || bi >= (int64_t)bt.size()) return -1;
        return (int32_t)(bt[(size_t)bi] * e.block_size + (pos % e.block_size));
    }
};

/* ------------------------------------------------------------------ binding the concrete classes */
/* Templates, so a mismatch with core/mem is reported where the engine instantiates them. */
template <class KV>
struct KVBinding final : IKVManager {
    KV* m;
    explicit KVBinding(KV* k) : m(k) {}

    int  add_sequence(uint64_t s) override          { return m->add_sequence(s); }
    bool has_sequence(uint64_t s) const override    { return m->has_sequence(s); }
    void free_sequence(uint64_t s) override         { m->free_sequence(s); }
    int  ensure(uint64_t s, int64_t n) override     { return m->ensure(s, n); }
    int  ensure_from(uint64_t s, int64_t n, int64_t k) override { return m->ensure(s, n, k); }
    void trim_window(uint64_t s, int64_t k) override { m->trim_window(s, k); }
    int64_t runtime_window_tokens(int32_t g) const override { return m->runtime_window_tokens(g); }
    int  set_window(uint64_t s, int64_t lo, int64_t w) override { return m->set_window(s, lo, w); }
    bool runtime_window(uint64_t s, int32_t g, SeqWindow* o) const override {
        return m->runtime_window(s, g, o);
    }
    bool has_runtime_window(uint64_t s) const override { return m->has_runtime_window(s); }
    bool has_window_snapshot() const override { return m->has_window_snapshot(); }
    int  seed_window(uint64_t s, int64_t p) override { return m->seed_window(s, p); }
    int32_t n_groups() const override { return (int32_t)m->plans().size(); }
    int64_t residency_window(int32_t g) const override {
        const KVGroupPlan* p = m->plan(g);
        return p && p->residency() && m->residency_enabled() ? p->residency_window : 0;
    }
    int residency_demote(uint64_t s, int32_t g, int64_t h, int64_t t, int64_t n,
                         std::vector<BlockMove>* out) override {
        return m->residency_demote(s, g, h, t, out, n);
    }
    int64_t residency_demote_cached(int64_t n, std::vector<BlockMove>* out) override {
        return m->residency_demote_cached(out, n);
    }
    void residency_demote_landed() override { m->residency_demote_landed(); }
    void residency_complete() override { m->residency_complete(); }
    int  residency_promote_all(uint64_t s, std::vector<BlockMove>* out) override {
        return m->residency_promote_all(s, out);
    }
    Residency residency() const override {
        const auto& st = m->residency_stats();
        Residency r;
        r.demoted    = st.demoted;
        r.promoted   = st.promoted;
        r.skipped    = st.demote_skipped;
        r.misses     = st.promoted_misses;
        r.freed_bytes = st.freed_bytes;
        r.host_bytes = m->residency_host_bytes();
        r.cap_bytes  = m->residency_host_cap_bytes();
        r.link_bytes = st.demoted_bytes + st.promoted_bytes;
        r.scratch_bytes = m->residency_scratch_bytes();
        if (paged_read) r.paged_in = paged_read();
        if (hits_read) r.paged_hits = hits_read();
        return r;
    }
    std::vector<SeqResidency> residency_seqs() const override {
        std::vector<SeqResidency> v;
        for (const auto& [id, s] : m->residency_by_seq())
            v.push_back({ id, s.demoted, s.promoted, s.skipped, s.link_bytes, s.freed_bytes });
        return v;
    }
    /* Ranks past 0, whose managers hold their own shard offsets; the engine fills it in. */
    std::vector<KV*> peers;
    /* Reads the pager's counter back from the card (TierExec::paged_in); set by the engine when
     * the pager is live. Called per statistics poll, never per step. */
    std::function<int64_t()> paged_read;
    std::function<int64_t()> hits_read;   /* the page cache's hits (TierExec::hits_in), the same way */
    Pager pager(int32_t g, int rank) const override {
        const KV* k = rank > 0 && (size_t)rank <= peers.size() ? peers[(size_t)rank - 1] : m;
        Pager p;
        const KVGroupPlan* pl = k->plan(g);
        if (!pl || pl->scratch_blocks <= 0 || !k->residency_device_base(g)) return p;
        p.store      = k->residency_device_base(g);   /* the group's own store */
        p.slot_bytes = k->residency_slot_bytes(g);
        p.slots      = k->residency_slots(g);
        p.layer_off  = k->residency_layer_off(g).data();
        p.scratch    = k->scratch_table(g).data();
        p.paged      = k->residency_paged();
        p.tags       = k->residency_tags(g);
        p.tags_rows  = p.tags ? k->scratch_rows() : 0;
        p.tags_bytes = p.tags ? k->residency_tags_bytes(g) : 0;
        p.hits       = k->residency_hits();
        p.epoch      = m->residency_epoch();
        return p;
    }
    int64_t scratch_pitch(int32_t g) const override {
        const KVGroupPlan* pl = m->plan(g);
        return pl ? pl->scratch_pitch : 0;
    }
    int  adopt(uint64_t s, int32_t g, const std::vector<int32_t>& b, int64_t n) override {
        return m->adopt(s, g, b, n);
    }
    int  preempt(uint64_t s) override               { return m->preempt(s); }
    bool can_ever_fit(int64_t n) const override     { return m->can_ever_fit(n); }
    bool can_grow(uint64_t s, int64_t n) const override { return m->can_grow(s, n); }
    int  release_idle_states() override                 { return m->release_idle_states(); }
    const std::vector<int32_t>& block_table(uint64_t s, int32_t g) const override {
        return m->block_table(s, g);
    }
    int32_t state_slot(uint64_t s, int32_t g) const override { return m->state_slot(s, g); }
    int state_index_row(uint64_t s, int32_t g, int32_t* o, int n) const override {
        return m->state_index_row(s, g, o, n);
    }
    int64_t state_index_width(int32_t g) const override { return m->state_index_width(g); }
    int64_t free_blocks(int32_t g) const override   { return m->free_blocks(g); }
    int64_t available_blocks(int32_t g) const override { return m->available_blocks(g); }
    int64_t total_blocks(int32_t g) const override  { return m->total_blocks(g); }
    int64_t held_blocks(int32_t g) const override   { return m->held_blocks(g); }
    int64_t held_top_blocks(int32_t g) const override { return m->held_top_blocks(g); }
    int64_t pool_bytes(int32_t g) const override      { return m->pool_bytes(g); }
    int64_t pool_bytes_used(int32_t g) const override { return m->pool_bytes_used(g); }
    int64_t pool_bytes_carved(int32_t g) const override { return m->pool_bytes_carved(g); }
    const RadKVStream* stream_view(int32_t g) const override { return m->stream_view(g); }
    int64_t stream_misses() const override { return (int64_t)m->stream_stats().misses; }
    int64_t stream_bytes_in() const override { return (int64_t)m->stream_stats().bytes_in; }
    int64_t stream_resident_blocks() const override {
        return m->stream_stats().resident_blocks;
    }
    int64_t block_tokens(int32_t g) const override {
        const KVGroupPlan* p = m->plan(g);
        return p && p->paged() ? p->block_size : 0;
    }
    bool window_snapshot(int32_t g) const override {
        const KVGroupPlan* p = m->plan(g);
        return p && p->win_snap;
    }
    int     rollback(uint64_t s, int64_t n) override { return m->rollback(s, n); }
    int64_t first_block_pos(uint64_t s, int32_t g) const override {
        return m->first_block_pos(s, g);
    }
    int64_t take_table_changes(uint64_t s, int32_t g) const override {
        return m->take_table_changes(s, g);
    }
    bool checkpoint_held(int32_t slot) const override { return m->checkpoint_held(slot); }
};

/* Content keys for a request's media runs. Placeholder ids are identical for every picture, so
 * the prefix cache also hashes each run's item content key (core/mm, over the processed patches)
 * and segment: the same picture at the same place hits, two pictures never alias. */
inline void media_keys(const Request& r, std::vector<MMContentKey>* out,
                       uint64_t control_vector_salt = 0) {
    out->clear();
    /* The runtime LoRA adapter is part of every block's key: KV written under an adapter is not
     * the base's. It enters the chain as one span over the whole sequence, keyed by the adapter's
     * load uid, so every block overlaps it; the base (uid 0) adds no span and its hashes are
     * unchanged. The uid and not the name, because a reload under the same name is new weights. */
    if (r.lora_uid != 0) {
        MMContentKey k;
        k.tok_begin = 0;
        k.tok_end = INT32_MAX;
        k.hash = (r.lora_uid * 0x9e3779b97f4a7c15ull) ^ 0x4c6f5241ull;   /* "LoRA" */
        out->push_back(k);
    }
    /* A control vector is part of every block's key for the same reason: a projected step's hidden
     * states are not the stock model's, so its KV is not the stock model's either. The vector's own
     * identity rides in as one span over the whole sequence, so every block overlaps it, and a
     * request that runs without the vector (or a deployment with none) adds no span and hashes
     * exactly as it did. An adapter and a vector are independent spans and compose. */
    if (control_vector_salt != 0 && r.sp.use_control_vector) {
        MMContentKey k;
        k.tok_begin = 0;
        k.tok_end = INT32_MAX;
        k.hash = control_vector_salt ^ 0x43455643ull;   /* "CVEC" */
        out->push_back(k);
    }
    if (!r.media) return;
    const mm::PromptMedia& pm = *r.media;
    for (const mm::RopeLayout::Run& run : pm.rope.runs) {
        const mm::Item& it = *pm.items[(size_t)run.item].item;
        MMContentKey k;
        k.tok_begin = run.tok;
        k.tok_end = run.tok + run.n;
        k.hash = it.key_hi ^ (it.key_lo * 0x9e3779b97f4a7c15ull) ^
                 ((uint64_t)(uint32_t)run.row0 * 0xc2b2ae3d27d4eb4full);
        out->push_back(k);
    }
}

template <class PC>
struct PCBinding final : IPrefixCache {
    PC* m;
    /* The control vector's identity (Engine::cvec_salt_for), or 0 with none loaded. A request that
     * runs with it on keys its blocks and its linear-state snapshots under a chain that includes it,
     * so its KV is never handed to a request that runs without it: the hidden states differ, and
     * the cache stores hidden states' consequences. Per request, not per cache: two sequences of one
     * step may differ. */
    uint64_t salt_id = 0;
    /* Scratch for a request's media keys, reused so a text request's lookup allocates nothing. */
    mutable std::vector<MMContentKey> mm_;
    explicit PCBinding(PC* p) : m(p) {}

    CacheHit lookup(const Request& r, std::vector<std::vector<int32_t>>* out,
                    const HitCap* cap) const override {
        media_keys(r, &mm_, salt_id);
        /* r.id: the admission takes its conversation's snapshots over. */
        return m->lookup(r.prompt, mm_, out, cap, r.id);
    }

    /* Installed by the engine when the idle tiers are on; empty otherwise. */
    std::function<RestoreResult(const Request&, bool whole_only)> restore_fn;
    RestoreResult restore(const Request& r, bool whole_only) override {
        return restore_fn ? restore_fn(r, whole_only) : RestoreResult::Whole;
    }
    /* The entries restore_fn brings back, in chain order: the whole chain if the pool can be made
     * to hold it, otherwise cut (PrefixCache::restorable_from). Room is made as evict() makes it,
     * dropping entries with a landed host copy, then landing pending copies and dropping those,
     * but never erasing one, so no other conversation is lost to a restore. The pool's free
     * quarter can go to it; the reserve's trim after the next publish frees it again. `*res`
     * says whether the chain was cut; with `whole_only`, a chain that would be cut comes back
     * empty and Declined, and the room made stays for the next attempt. */
    std::vector<BlockHash> restore_plan(const Request& r, uint64_t salt, bool whole_only = false,
                                        RestoreResult* res = nullptr) {
        if (res) *res = RestoreResult::Whole;
        media_keys(r, &mm_, salt);
        std::vector<BlockHash> want = m->restorable_from(r.prompt, mm_);
        const int64_t need = (int64_t)want.size() - m->restore_room();
        if (need <= 0) return want;
        const int64_t k = m->trim_for(r.prompt, mm_, need);
        if (k < need && flush_fn && flush_fn()) (void)m->trim_for(r.prompt, mm_, need - k);
        /* Landing copies may have moved records; take the chain again against the room made. */
        if (whole_only) {
            want = m->restorable_from(r.prompt, mm_);
            if ((int64_t)want.size() <= m->restore_room()) return want;
            if (res) *res = RestoreResult::Declined;
            return {};
        }
        bool cut = false;
        want = m->restorable_from(r.prompt, mm_, m->restore_room(), &cut);
        if (cut && res) *res = RestoreResult::Cut;
        return want;
    }
    void begin_step() override { m->begin_step(); }
    void hold(const Request& r) override {
        media_keys(r, &mm_, salt_id);
        m->hold(r.id, r.prompt, mm_);
    }
    void release(const Request& r) override { m->release(r.id); }
    const std::vector<int32_t>& cached_groups() const override { return m->cached_groups(); }
    void set_drafter_floor(int64_t tokens) override { m->set_drafter_floor(tokens); }

    /* Indexes prompt ++ output: the next turn sends the reply back in its prompt, so indexing the
     * prompt alone would stop hitting at the first assistant token. */
    int insert(const Request& r, const std::vector<std::vector<int32_t>>& blocks,
               int64_t n_tokens) override {
        std::vector<int32_t> toks;
        toks.reserve(r.prompt.size() + r.output.size());
        toks.insert(toks.end(), r.prompt.begin(), r.prompt.end());
        toks.insert(toks.end(), r.output.begin(), r.output.end());
        if (n_tokens > (int64_t)toks.size()) n_tokens = (int64_t)toks.size();
        /* r.id is the session, the same id reserve_checkpoint keys its snapshots by. */
        media_keys(r, &mm_, salt_id);
        return m->insert(r.id, toks, mm_, blocks, n_tokens, r.session_label);
    }
    /* Installed by the engine when the idle tiers are on: lands every pending host copy now and
     * returns whether any landed. Runs on the step thread under the scheduler's lock. */
    std::function<bool()> flush_fn;
    /* Under pressure, first drop entries whose host copy has landed (trim), then land pending
     * copies and trim again. Only the rest is erased; with tiers on, an erase is warned about,
     * since a later turn will prefill that conversation again. */
    int64_t evict(int64_t n) override {
        int64_t k = m->trim(n);
        if (k < n && flush_fn && flush_fn()) k += m->trim(n - k);
        if (k >= n) return k;
        const int64_t before = m->stats().evictions;
        k += m->evict_lru(n - k);
        const int64_t lost = m->stats().evictions - before;
        if (lost > 0 && flush_fn) {
            const uint64_t now = rad_mono_ns();
            if (now - lost_warned_ns_ >= 60ull * 1000000000ull) {
                lost_warned_ns_ = now;
                RAD_WARN("prefix cache: %lld entries erased with no copy off the card -- the host "
                         "tier had no room for them in time (--prefix-cache-mib); "
                         "radiance:prefix_cache_evictions_total counts every one",
                         (long long)lost);
            }
        }
        return k;
    }
    int64_t trim(int64_t n) override { return m->trim(n); }
    int64_t reset() override {
        const int64_t n = m->size();
        m->clear();
        return n;
    }
    int64_t evictions() const override { return m->stats().evictions; }
    int64_t ckpt_evictions() const override { return m->stats().ckpt_evictions; }
    CkptCounts ckpt_counts() const override {
        const auto& st = m->stats();
        return { st.ckpt_retired, st.ckpt_lru, st.ckpt_refused };
    }
    int64_t expired() const override { return m->stats().expired; }
    int64_t ckpt_expired() const override { return m->stats().ckpt_expired; }
    int64_t released() const override { return m->stats().released; }
    int64_t ckpt_released() const override { return m->stats().ckpt_released; }
    int release_session(const std::string& label, uint64_t id, int64_t keep_tokens,
                        SessionRelease* out) override {
        return m->release_session(label, id, keep_tokens, out);
    }
    int64_t held_tokens() const override { return m->unshared_tokens(); }

    int64_t next_checkpoint_after(int64_t p) const override { return m->next_checkpoint_after(p); }
    int reserve_checkpoint(const Request& r, int64_t pos, int32_t* slot_out) override {
        /* A snapshot is keyed by the chained hash of the prefix it covers, built here because the
         * cache does not hold the tokens. It must match the chain lookup() walks, or the linear
         * cache never hits. The prefix is prompt ++ output, so continuations of one prompt do not
         * share a snapshot; whole blocks only, since a partial block's key would change. */
        return m->reserve_checkpoint(r.id, pos, prefix_hash(r, pos), slot_out);
    }

    void commit_checkpoint(const Request& r, int64_t pos) override {
        m->commit_checkpoint(prefix_hash(r, pos));
    }

private:
    uint64_t lost_warned_ns_ = 0;
    mutable std::vector<int32_t> blk_;   /* prefix_hash's block of tokens, kept for its capacity */

    /* The chain reserve_checkpoint and commit_checkpoint both key on; shared so they cannot
     * drift apart. */
    BlockHash prefix_hash(const Request& r, int64_t pos) const {
        media_keys(r, &mm_, salt_id);
        const int64_t bs = m->block_size();
        BlockHash h{};
        if (bs <= 0) return h;
        const int64_t np = (int64_t)r.prompt.size(), no = (int64_t)r.output.size();
        const int64_t have = np + no;
        const int64_t n_blocks = (pos < have ? pos : have) / bs;
        blk_.resize((size_t)bs);
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t base = b * bs;
            for (int64_t j = 0; j < bs; ++j) {
                const int64_t k = base + j;
                blk_[(size_t)j] = k < np ? r.prompt[(size_t)k] : r.output[(size_t)(k - np)];
            }
            h = chain_block_hash(h, blk_.data(), (int32_t)bs, mm_.data(), (int32_t)mm_.size(), base);
        }
        return h;
    }
};

}  /* namespace rad */
