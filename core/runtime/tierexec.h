/* tierexec.h -- executes KV tier transfers.
 *
 * core/mem/kvtier.h decides what moves and hands out TierJobs; this runs them. core/mem has no
 * stream and must not get one, because the design avoids host synchronisation on the step path
 * (spec §5.4). Same split as KVManager and BlockCopy.
 *
 * Nothing here waits on the host. Copies are enqueued and ordered by events:
 *   - a copy out waits on the transfer stream for everything the compute stream had been given
 *     when it was issued (including the step that wrote the blocks) and signals an event the
 *     engine polls between steps. The engine keeps building steps meanwhile and the scheduler
 *     lock is not held;
 *   - a restore in makes the compute stream wait for it, so the step reading the restored blocks
 *     runs after they land.
 *
 * Transfer shape. A paged pool is layer-major: layer l of group g owns
 * `[base + l*layer_stride, + layer_stride)` and a block in it is at
 * `block * layer_bytes_per_block()`. A logical block is thus `n_layers` fragments of a few hundred
 * bytes, too small to move one at a time. Within a layer, blocks are dense rows of a matrix, so
 * packing a set is `gather_rows` and restoring it is `scatter_rows`: one launch per (group, layer)
 * per staging batch.
 *
 * Staging layout. Device staging is `max_batch * entry_bytes`; one entry is [group][layer][fragment]
 * end to end, the replicated groups first. For a given (group, layer) the destination rows sit at
 * stride `entry_bytes` from a base in the first entry, which is `gather_rows`'s `y` pitch, so one
 * launch writes every entry's fragment in place and the host side is one strided copy per window
 * per run of consecutive arena slots.
 *
 * The host slot. Under tensor parallelism a slot holds the whole engine's entry. A group every
 * rank holds whole (rad_kv_group_replicated) is in the slot once, in one window at its front
 * that rank 0 writes and every rank reads back; the groups each rank holds a shard of follow,
 * one window per rank. slot_layout() is the one place the arithmetic lives.
 */
#pragma once
#include "mem/kv.h"
#include "mem/kvtier.h"
#include "plugin/registry.h"
#include "rad_core.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

class TierExec {
public:
    ~TierExec();
    /* `groups` is PrefixCache::entry_groups(): the group of each block an entry holds, in the
     * order of the entry's `blocks` vector (a group with blocks shorter than the entry is listed
     * once per block). The staging layout is derived from them here only.
     * `max_batch` is the staging buffer in entries, `max_copy` the most entries per copy out and
     * `max_chain` the longest restore chain; the index buffers are sized from the last two. */
    int  configure(Registry* reg, KVManager* kv, IdleTiers* tiers,
                   const std::vector<int32_t>& groups, int max_batch, int64_t max_copy,
                   int64_t max_chain, bool leader, int device, int64_t slot_off,
                   RadStream shared);

    /* One shared host slot across the ranks. `[0, rep_bytes)` is the replicated window, the
     * replicated cached groups in cached-group order; rank r's own groups follow at
     * `own_off[r]`, `own_bytes[r]` wide (uneven heads and rad_kv_group_zero make these differ).
     * `entry_bytes` is the largest rank's staging entry, rep_bytes + own_bytes, which sizes()
     * takes. Checked here: a group replicated on one rank is replicated, not zero and the same
     * width on every rank, since every rank reads the one window back; a violation stops bringup
     * with the group named (RAD_E_STATE). */
    struct SlotLayout {
        int64_t slot_bytes = 0;
        int64_t rep_bytes = 0;
        int64_t entry_bytes = 0;
        std::vector<int64_t> own_off, own_bytes;
        std::vector<std::string> replicated;   /* the replicated groups' names, for the report */
    };
    static int slot_layout(const std::vector<const KVManager*>& ranks,
                           const std::vector<int32_t>& groups, SlotLayout* out);
    void close();
    bool enabled() const { return enabled_; }
    std::string report() const { return report_; }

    /* configure()'s sizes from one entry's bytes on this rank, the longest context and the block
     * size, and the VRAM device_init() takes for them. The VRAM budget reserves device_bytes() and
     * bringup configures with sizes(), so the two cannot drift apart. */
    struct Sizes { int max_batch = 0; int64_t max_copy = 0, max_chain = 0; };
    static Sizes   sizes(int64_t entry_bytes, int64_t max_ctx, int64_t block_size);
    static int64_t device_bytes(int64_t entry_bytes, int n_groups, const Sizes& s);

    /* Planning and transfer are separate for tensor parallelism. The policy describes conversations,
     * which are not sharded; the transfer is per rank, since each rank holds its own shard of every
     * block. The leader plans, every rank runs the same job list, and the leader reports once. A
     * follower consulting the policy itself would find records already pending and silently move
     * nothing, corrupting KV on one card only. */

    /* ---------------- copy on idle ----------------
     *
     * Enqueues a batch of WriteBack jobs on this rank, ordered after everything `compute` had been
     * given; `copy_state` says when it has landed. One batch at a time: the engine issues the next
     * only after this one reports, so the staging and index buffers are reused without a wait. */
    int  issue_copy(std::vector<TierJob>& jobs, RadStream compute);
    /* 1 when nothing is out or the last batch has landed, 0 while it is in flight, negative when
     * the transfer stream reported an error. Taking the answer clears it. */
    int  copy_state();
    /* copy_state after waiting for the batch to land, for a caller that needs the host copy
     * first (an eviction that would otherwise lose the entry). */
    int  wait_copy();
    /* Waits until everything this rank enqueued has landed. Used before a block is given back on a
     * failure path, where a late in-flight copy would overwrite its next owner's writes. */
    void quiesce();

    /* ---------------- restore ----------------
     *
     * Enqueues a chain of ToDevice jobs in chain order and makes `compute` wait for them. The
     * leader chooses every block first, recalling the loan for the whole chain at once; a chain
     * the pool cannot hold is cut where the reservation ran out and the rest left unstarted.
     * `ok[i]` is whether job i was fully enqueued on this rank. */
    int  issue_restore(std::vector<TierJob>& jobs, RadStream compute, std::vector<char>* ok);
    /* Leader only. Gives back a failed restore's blocks and reports every job to the policy.
     * `now_ns` is the pass time; a promotion counts as a use. */
    void report_pass(std::vector<TierJob>& jobs, const std::vector<char>& ok, uint64_t now_ns);

    /* A lookup hit off-device: the job that would bring it back, or false when the entry is not
     * off-device or already has work outstanding. */
    bool plan_promote(const BlockHash& key, TierJob* out);           /* leader only */

    /* ---------------- window residency (spec §7.3) ----------------
     *
     * Moves one group's blocks between the pool and the manager's pinned store. `to_host` packs them
     * out of the pool with gather_rows and writes their slots; the reverse copies the slots in and
     * scatters them back, then makes `compute` wait -- the step that follows reads them.
     *
     * One group a call, up to kMoveBatch blocks. Both directions run on the transfer stream, so a
     * promotion always sees what the demotion before it wrote. The caller owns the order: a demote
     * belongs after the step that wrote the blocks, a promote before the step that reads them.
     *
     * A block the manager has no slot for is refused rather than half-moved: a promotion into a
     * block the sequence no longer holds would serve another sequence's bytes.
     *
     * A tiered group (KVGroupPlan::tiered, RAD_KVQ_DEMOTE) is requantised on the card between the
     * gather and the copy out (`kv_transcode` into the demote layout) and expanded between the
     * copy in and the scatter, so the slot holds the narrower bytes and the pool only ever the
     * resident layout. Under the emulation (KVManager::residency_emulate) a demotion names no
     * slot (host_slot -1) and is that round trip in place: gather, both transcodes, scatter into
     * the same blocks, no copy; an untiered group's round trip is no change and moves nothing.
     *
     * Every demotion, the emulation's too, makes `compute` wait for it before the steps that
     * follow: they read the block as cold through the pager (or, emulated, its new bytes), and a
     * pager reading a slot the copy has not finished filling would serve stale bytes. Steps
     * issued before it already ran past the block, which it waited for. */
    static constexpr int kMoveBatch = 16;
    int  issue_moves(const std::vector<BlockMove>& moves, RadStream compute, bool to_host);
    struct MoveStats { int64_t out_blocks = 0, in_blocks = 0, out_bytes = 0, in_bytes = 0; };
    const MoveStats& move_stats() const { return mv_; }
    /* Whether every move issued so far has landed: a poll of one event, no host sync. True when
     * nothing was issued. A failed query counts as landed, so a broken card cannot hold pool
     * blocks and store slots for ever (the failure itself was reported by the move). */
    bool moves_landed() const;
    /* Only the leader's manager holds the move records (rank 0 hands out slot ids and blocks); a
     * follower carries out the same list into its own shard and has nothing to check it against. */

    /* ---------------- the pager's counter ----------------
     *
     * One int64 on this rank's card that `qsa_page` adds the cold entries it rewrote to
     * (RadKVGroupBatch::residency_paged). Allocated by the first call, zeroed, and never moved, so
     * the batch can carry its address from the start. null when the card has no memory for it.
     * paged_in() reads it back with one small copy on the transfer stream and waits for that copy
     * only -- never for the step -- so it belongs on a statistics poll, not on the step path. */
    int64_t* paged_counter();
    int64_t  paged_in();

    /* ---------------- importance-tiered demotion (KVGroupPlan::split) ----------------
     *
     * slot_table(): this rank's [slots] u8 table of which demote layout each store slot holds
     * (RadKVGroupBatch::residency_slot_layout), zeroed, allocated by the first call like the
     * pager's counter. issue_moves() writes a split group's slot byte on the transfer stream right
     * behind the slot's bytes, so the pager, which waits for the demotion, reads the two together.
     *
     * count_buffer(): the selection counter of split group `g`, `n` u32 on the card (zeroed) and
     * its pinned host copy (`*host`), allocated by the first call; the leader's only.
     * count_snapshot() enqueues the copy of every counter to its host copy on `s` (rank 0's compute
     * stream, behind the step, so the step's collect waits for it); count_clear() zeroes the named
     * blocks' words on `s`, runs of adjacent ids one memset, before the step that may select them. */
    uint8_t*  slot_table(int64_t slots);
    uint32_t* count_buffer(int32_t g, int64_t n, const uint32_t** host);
    int       count_snapshot(RadStream s);
    int       count_clear(const std::vector<std::pair<int32_t, int32_t>>& blocks, RadStream s);


    /* ---------------- RADIANCE_DEBUG_TIER_VERIFY, the replicated window ----------------
     *
     * A replicated group's bytes are the plugin's claim. Under the variable a follower gathers the
     * replicated groups it would otherwise leave to the leader into a pinned scratch beside the
     * copy and hashes them once landed; the engine, once every rank has landed, takes each sum
     * and has the leader compare it with the window it wrote. Off, a follower moves nothing for
     * the replicated groups. */
    bool verifying_replicated() const { return verify_ && rep_bytes_ > 0; }
    /* Follower: the sum of the replicated bytes it gathered for `key` in the last copy out, taken
     * once. False when there is none (not verifying, no replicated group, or not in that copy). */
    bool take_replicated_sum(const BlockHash& key, uint64_t* sum);
    /* Leader: compares `follower_sum` from rank `rank` with the window the slot holds for job
     * `j`; a mismatch is logged and counted (Stats::rep_mismatch). */
    bool verify_replicated(const TierJob& j, int rank, uint64_t follower_sum);

    struct Stats {
        int64_t passes = 0, jobs = 0, bytes_out = 0, bytes_in = 0;
        int64_t rep_checked = 0, rep_mismatch = 0;   /* verify_replicated's tallies */
    };
    const Stats& stats() const { return st_; }

private:
    struct GroupInfo {
        int32_t  g = -1;
        int64_t  n_layers = 0;
        int64_t  frag = 0;          /* layer_bytes_per_block() */
        int64_t  layer_stride = 0;
        int64_t  n_blocks = 0;
        void*    base = nullptr;
        int64_t  off = 0;           /* this group's start inside one entry's staging bytes */
        /* rad_kv_group_zero: no shard on this rank. Kept in place because a job's block ids are
         * one per cached group in the same order on every rank; it moves no bytes. */
        bool     zero = false;
        /* rad_kv_group_replicated: staged at the front of the entry and in the slot's replicated
         * window; a follower gathers it only to verify, and scatters it on every restore. */
        bool     replicated = false;
        Resolved gather, scatter;
        /* A tiered residency group (KVGroupPlan::tiered): `kv_transcode` into the store's layout
         * and back, one layer's fragment of a slot `sfrag` bytes, `rows` (token, head) rows of
         * `rf` bytes in the pool and `rt` in the store. Unresolved for any other group. */
        Resolved to_store, to_pool;
        int64_t  sfrag = 0, rows = 0, rf = 0, rt = 0;
        /* Two demote layouts (KVGroupPlan::split): the narrow one's transcodes, its row and its
         * per-layer bytes, which a slot holds at the front of each `sfrag`. */
        Resolved to_store2, to_pool2;
        int64_t  sfrag2 = 0, rt2 = 0;
    };

    /* Builds the staging and index buffers, events and transfer stream lazily on the first pass
     * with a job, never at bringup (see the definition for the cost). */
    int  device_init();
    /* Frees what device_init() allocated; the stream is the caller's to drain first. */
    void release_device();
    /* Whether every id this rank is about to address belongs to the cache. The kernels only bound
     * ids by the carve, [0, n_blocks), but blocks above live_blocks() are lent to the expert slab,
     * and a restore into one overwrites weights in use. That corrupts output silently, so it is
     * checked here, where the job can still be refused. */
    bool blocks_backed(const std::vector<int32_t>& blocks, const char* what);
    void warn_once(const char* what, int st);
    /* One launch per (group, layer). `pitch` is the staging stride between consecutive jobs:
     * one cache entry for the idle tiers, one block for a residency move. `stage_off` is where
     * this group's bytes start inside a job's staging slot. */
    int  run_rows(const Resolved& r, bool scatter, const GroupInfo& gi, int64_t layer,
                  void* stage_base, const int32_t* d_idx, int64_t n_jobs,
                  int64_t pitch, int64_t stage_off);
    /* One strided copy per window per run of consecutive arena slots, `n` entries starting at job
     * `first` of `order`. To host when `out` (the replicated window from the leader only), from
     * host otherwise. */
    int  copy_runs(const std::vector<TierJob*>& order, size_t first, int64_t n, bool out);
    /* Whether this rank gathers group `gi` for a copy out: its own groups always, a replicated
     * one as the leader or to verify. */
    bool gathers(const GroupInfo& gi) const {
        return !gi.zero && (!gi.replicated || leader_ || verify_rep_);
    }
    /* The hash of one entry as the slot holds it: the replicated window, then this rank's own. */
    uint64_t entry_sum(const char* slot) const;
    int  enqueue_copy(std::vector<TierJob>& jobs, RadStream compute);

    bool      enabled_ = false;
    bool      leader_ = false;   /* this rank chooses block ids; the others follow */
    bool      warned_ = false;
    /* Separate from `warned_`: an unaddressable id is a broken invariant, not a transfer failure.
     * Logged once, since the pass retries the refusal every tick. */
    bool      warned_unbacked_ = false;
    /* This rank's card. Allocations and launches use the current device, and the engine loop leaves
     * the last rank's device current, so it is set explicitly. */
    int       device_ = -1;
    /* This rank's windows of a shared host slot (SlotLayout): its own groups at `slot_off_`,
     * `own_bytes_` wide, and the replicated groups at the slot's front, `rep_bytes_` wide, which
     * only the leader writes. The staging entry is [replicated][own], entry_bytes_ in all. */
    int64_t   slot_off_ = 0;
    int64_t   slot_bytes_ = 0;   /* the whole shared slot */
    int64_t   rep_bytes_ = 0;
    int64_t   own_bytes_ = 0;
    Registry* reg_ = nullptr;
    KVManager* kv_ = nullptr;
    IdleTiers* tiers_ = nullptr;
    std::vector<GroupInfo> gi_;
    int64_t   entry_bytes_ = 0;
    int       max_batch_ = 0;
    int64_t   max_copy_ = 0;
    int64_t   max_chain_ = 0;
    void*     stage_ = nullptr;     /* device, max_batch_ * entry_bytes_ */
    /* ---- the block indices ----
     *
     * One region for the copy out and a ring of kRing for restores, pinned on the host and mirrored
     * on the device. A region may be refilled only after the upload that last read it has run. One
     * copy is out at a time, so its region is always free; several restores can be issued per step,
     * so each takes the next ring region and waits on its event (the upload kRing restores ago). */
    static constexpr int kRing = 4;
    int32_t*  idx_host_ = nullptr;  /* pinned: copy region, then kRing restore regions */
    int32_t*  idx_dev_ = nullptr;
    int64_t   copy_idx_ = 0;        /* entries a copy region holds, per group */
    int64_t   chain_idx_ = 0;       /* entries a restore region holds, per group */
    RadEvent  ring_ev_[kRing] = {};
    bool      ring_used_[kRing] = {};
    int       ring_next_ = 0;
    RadEvent  ev_compute_ = nullptr;  /* the compute stream, as a copy out found it */
    RadEvent  ev_copy_ = nullptr;     /* the last copy out, landed */
    RadEvent  ev_restore_ = nullptr;  /* the last restore, landed */
    RadEvent  ev_move_ = nullptr;     /* the last residency move, either direction, landed */
    bool      moved_ = false;         /* a move was recorded on ev_move_ */
    int64_t*  paged_dev_ = nullptr;   /* paged_counter(), device */
    int64_t*  paged_host_ = nullptr;  /* its readback target, pinned */
    uint8_t*  slot_dev_ = nullptr;    /* slot_table(), device */
    uint8_t*  lay_host_ = nullptr;    /* pinned {0, 1}: the bytes the slot table is written from */
    struct Count { int32_t g = -1; int64_t n = 0; uint32_t* dev = nullptr; uint32_t* host = nullptr; };
    std::vector<Count> counts_;       /* count_buffer()s */
    bool      copy_out_ = false;
    /* Transfer stream. Each stream costs a hardware queue, and a fourth queue costs this card 65% of
     * its decode step, so the tier borrows the expert mover's maintenance stream
     * (Mover::transfer_stream) when there is one. `own_stream_` tells close() whether to destroy it. */
    RadStream stream_ = nullptr;
    RadStream shared_ = nullptr;
    bool      own_stream_ = false;
    std::vector<TierJob*> order_;
    /* ---- RADIANCE_DEBUG_TIER_VERIFY ----
     *
     * Round-trip check: hashes each entry going out and coming back and names the entry when they
     * differ, separating "the tier moved wrong bytes" from a downstream fault. A diagnostic, not a
     * guard: host hashing costs on every entry. Off unless set. The round trip is the leader's
     * (the map is keyed by an object followers do not own); a follower's part is the replicated
     * window, above. A copy out is hashed once landed (first seen by copy_state). */
    bool      verify_ = false;
    bool      verify_rep_ = false;    /* a follower gathering the replicated groups to verify */
    /* Residency staging (issue_moves): one batch of one group's blocks, both directions, plus the
     * block indices it gathers with. Allocated with the rest of the device state and only when some
     * group declares a residency window, so a server that never moves a block pays nothing. */
    void*     move_stage_ = nullptr;
    /* A tiered group's blocks in the store's layout, between the transcode and the copy. */
    void*     move_stage2_ = nullptr;
    /* One `kv_transcode` launch over `rows` contiguous rows of `src` into `dst`. */
    int       run_xc(const Resolved& r, const void* src, int64_t src_row, void* dst, int64_t dst_row,
                     int64_t rows);
    /* Pinned, kMoveIdxCap entries used as a ring: the upload is a kernel reading host memory when
     * the transfer stream reaches it, which is after the compute it waits for, so a batch issued
     * later in the same sweep must not overwrite the indices of one not yet uploaded. */
    static constexpr int64_t kMoveIdxCap = 64 * kMoveBatch;
    int32_t*  move_idx_host_ = nullptr;
    int64_t   move_idx_off_ = 0;      /* the next free entry of move_idx_host_ */
    int32_t*  move_idx_dev_ = nullptr;
    int64_t   move_bytes_ = 0;      /* bytes one block of the widest residency group takes */
    MoveStats mv_;
    std::unordered_map<BlockHash, uint64_t, BlockHashHash> out_sum_;
    std::unordered_map<BlockHash, uint64_t, BlockHashHash> rep_sum_;   /* follower */
    struct VerifyNote { BlockHash key; int32_t host_slot = -1; int64_t ord = -1; };
    std::vector<VerifyNote> verify_pending_;
    char*     vrep_ = nullptr;        /* pinned, max_copy_ * rep_bytes_: a follower's gather */
    void      verify_note(const BlockHash& key, const char* slot);
    void      verify_check(const BlockHash& key, const char* slot);

    std::string report_;
    Stats     st_;
};

}  /* namespace rad */
