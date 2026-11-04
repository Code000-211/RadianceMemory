#include "runtime/tierexec.h"

#include "rad_internal.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace rad {

TierExec::~TierExec() { close(); }

/* gather_rows and scatter_rows are resolved once here, at the width they are called at (`n` is the
 * per-group fragment in bytes). Nothing re-resolves on the maintenance path, so the kernel serving
 * a transfer cannot change in the middle of a batch. */
int TierExec::configure(Registry* reg, KVManager* kv, IdleTiers* tiers,
                        const std::vector<int32_t>& groups, int max_batch, int64_t max_copy,
                        int64_t max_chain, bool leader, int device, int64_t slot_off,
                        RadStream shared) {
    close();
    if (!reg || !kv || !tiers || groups.empty() || max_batch <= 0 || max_copy <= 0 ||
        max_chain <= 0)
        return RAD_E_INVAL;
    reg_ = reg; kv_ = kv; tiers_ = tiers; leader_ = leader; device_ = device;
    max_batch_ = max_batch; max_copy_ = max_copy; max_chain_ = max_chain;
    slot_off_ = slot_off;
    slot_bytes_    = tiers->entry_bytes();
    shared_        = shared;
    RAD_TRY(rad_dev_set(device_));

    for (int32_t g : groups) {
        const KVGroupPlan* p = kv->plan(g);
        if (!p || !p->paged()) continue;
        /* A streamed group's host mirror is its tier (docs/KV-STREAMING.md §4.1). */
        if (p->streamed()) continue;
        GroupInfo gi;
        gi.g = g;
        gi.n_layers = p->n_layers;
        gi.frag = p->layer_bytes_per_block();
        gi.layer_stride = p->layer_stride();
        gi.n_blocks = p->pool_blocks();
        gi.base = p->base;
        gi.replicated = p->replicated;
        if (p->zero) {
            gi.zero = true;
            gi_.push_back(gi);
            continue;
        }
        /* The kernel is picked on the fragment width (libr4d's byte-row pair needs n % 4 == 0).
         * An unsupported width fails here rather than at launch inside a batch. */
        Geometry q;
        q.set_s("dtype", "u8");
        q.set_i("n", gi.frag);
        q.set_i("M", max_batch);
        gi.gather  = reg->resolve("gather_rows",  q, RAD_DOMAIN_DEVICE);
        gi.scatter = reg->resolve("scatter_rows", q, RAD_DOMAIN_DEVICE);
        if (!gi.gather || !gi.scatter) {
            RAD_WARN("kv tiers: no device gather/scatter for group %d at a %lld-byte fragment, "
                     "so a finished conversation cannot leave VRAM. The tiers are off.",
                     (int)g, (long long)gi.frag);
            close();
            return RAD_E_UNSUPPORTED;
        }
        /* A tiered residency group: the two transcodes, at a batch's rows. */
        if (p->residency() && p->tiered()) {
            gi.sfrag = p->residency_frag();
            gi.rf = kv_layout_row_bytes(p->res_from, p->res_head_dim);
            gi.rt = kv_layout_row_bytes(p->res_to, p->res_head_dim);
            gi.rows = gi.rf > 0 ? gi.frag / gi.rf : 0;
            Geometry x;
            x.set_i("M", (int64_t)kMoveBatch * gi.n_layers * gi.rows);
            x.set_i("head_dim", p->res_head_dim);
            x.set_s("from", kv_layout_name(p->res_from));
            x.set_s("to", kv_layout_name(p->res_to));
            gi.to_store = reg->resolve("kv_transcode", x, RAD_DOMAIN_DEVICE);
            x.set_s("from", kv_layout_name(p->res_to));
            x.set_s("to", kv_layout_name(p->res_from));
            gi.to_pool = reg->resolve("kv_transcode", x, RAD_DOMAIN_DEVICE);
            if (!gi.to_store || !gi.to_pool || gi.rows <= 0) {
                RAD_WARN("kv residency: no device kv_transcode between %s and %s at head_dim %lld "
                         "for group %d, so its blocks cannot be stored in the demote layout",
                         kv_layout_name(p->res_from), kv_layout_name(p->res_to),
                         (long long)p->res_head_dim, (int)g);
                close();
                return RAD_E_UNSUPPORTED;
            }
            /* The second demote layout (KVGroupPlan::split): its own pair of transcodes. */
            if (p->split()) {
                gi.sfrag2 = p->res_frag2;
                gi.rt2 = kv_layout_row_bytes(p->res_to2, p->res_head_dim);
                x.set_s("from", kv_layout_name(p->res_from));
                x.set_s("to", kv_layout_name(p->res_to2));
                gi.to_store2 = reg->resolve("kv_transcode", x, RAD_DOMAIN_DEVICE);
                x.set_s("from", kv_layout_name(p->res_to2));
                x.set_s("to", kv_layout_name(p->res_from));
                gi.to_pool2 = reg->resolve("kv_transcode", x, RAD_DOMAIN_DEVICE);
                if (!gi.to_store2 || !gi.to_pool2) {
                    RAD_WARN("kv residency: no device kv_transcode between %s and %s at head_dim "
                             "%lld for group %d, so its blocks cannot take the second demote "
                             "layout", kv_layout_name(p->res_from), kv_layout_name(p->res_to2),
                             (long long)p->res_head_dim, (int)g);
                    close();
                    return RAD_E_UNSUPPORTED;
                }
            }
        }
        gi_.push_back(gi);
    }
    if (gi_.empty()) return RAD_E_UNSUPPORTED;
    /* The staging entry is the replicated groups first, then this rank's own, each in cached-group
     * order, so the two windows of the slot are each one contiguous run of the entry. */
    int64_t off = 0;
    for (GroupInfo& gi : gi_)
        if (gi.replicated && !gi.zero) { gi.off = off; off += gi.n_layers * gi.frag; }
    rep_bytes_ = off;
    for (GroupInfo& gi : gi_)
        if (!gi.replicated && !gi.zero) { gi.off = off; off += gi.n_layers * gi.frag; }
    own_bytes_ = off - rep_bytes_;
    /* Every cached group is zero on this rank, so its entry would be empty. One rank that cannot
     * take part turns the tiers off for all. */
    if (off == 0) {
        RAD_WARN("kv tiers: this rank keeps no cache for any group the prefix cache holds "
                 "(rad_kv_group_zero), so it has no part of an entry to move. The tiers are off.");
        close();
        return RAD_E_UNSUPPORTED;
    }
    entry_bytes_ = off;

    /* Nothing is built on the device here; see device_init(). */
    verify_ = std::getenv("RADIANCE_DEBUG_TIER_VERIFY") != nullptr;
    verify_rep_ = verify_ && !leader_ && rep_bytes_ > 0;

    /* A group whose blocks are shorter than an entry is listed once per block it holds there. */
    size_t n_groups = 0;
    for (size_t i = 0; i < gi_.size(); ++i) {
        bool seen = false;
        for (size_t j = 0; j < i && !seen; ++j) seen = gi_[j].g == gi_[i].g;
        if (!seen) ++n_groups;
    }
    const std::string blocks =
        n_groups != gi_.size() ? fmt(" in %zu blocks", gi_.size()) : std::string();
    char buf[400];
    std::snprintf(buf, sizeof buf,
                  "kv tier transfer: %zu paged group(s), %s an entry%s%s%s%s, batches of %d "
                  "through gather_rows/scatter_rows at u8",
                  n_groups, humanb(entry_bytes_).c_str(), blocks.c_str(),
                  rep_bytes_ > 0 ? " (" : "",
                  rep_bytes_ > 0 ? humanb(rep_bytes_).c_str() : "",
                  rep_bytes_ > 0 ? " of it replicated on every rank and held once a slot)" : "",
                  max_batch_);
    report_ = buf;
    enabled_ = true;
    return RAD_OK;
}

int TierExec::slot_layout(const std::vector<const KVManager*>& ranks,
                          const std::vector<int32_t>& groups, SlotLayout* out) {
    if (!out || ranks.empty()) return RAD_E_INVAL;
    for (const KVManager* kv : ranks) if (!kv) return RAD_E_INVAL;
    SlotLayout L;
    L.own_off.assign(ranks.size(), 0);
    L.own_bytes.assign(ranks.size(), 0);
    for (int32_t g : groups) {
        const KVGroupPlan* p0 = ranks[0]->plan(g);
        if (!p0 || !p0->paged()) continue;
        /* Skipped as in configure(), so the slot layout and the gather offsets agree. */
        if (p0->streamed()) continue;
        bool replicated = false;
        for (size_t i = 0; i < ranks.size(); ++i) {
            const KVGroupPlan* p = ranks[i]->plan(g);
            replicated = replicated || (p && p->replicated);
        }
        if (!replicated) {
            for (size_t i = 0; i < ranks.size(); ++i) {
                const KVGroupPlan* p = ranks[i]->plan(g);
                if (p && p->paged()) L.own_bytes[i] += p->bytes_per_block;
            }
            continue;
        }
        /* Every rank reads the one window back into its own pool, so the group must be the same
         * width everywhere, declared replicated everywhere and carved everywhere. */
        for (size_t i = 0; i < ranks.size(); ++i) {
            const KVGroupPlan* p = ranks[i]->plan(g);
            const char* why = !p || !p->paged()        ? "does not hold it as a paged group"
                              : p->zero                ? "declares it zero (rad_kv_group_zero)"
                              : !p->replicated         ? "does not declare it replicated"
                              : p->bytes_per_block != p0->bytes_per_block
                                                       ? "holds it at another width"
                                                       : nullptr;
            if (!why) continue;
            RAD_ERR("kv tiers: KV group '%s' is declared replicated (rad_kv_group_replicated) on "
                    "one rank, but rank %zu %s. A replicated group is held whole and alike on "
                    "every rank, since every rank restores it from the one copy a slot holds.",
                    p0->name.c_str(), i, why);
            return RAD_E_STATE;
        }
        L.rep_bytes += p0->bytes_per_block;
        if (std::find(L.replicated.begin(), L.replicated.end(), p0->name) == L.replicated.end())
            L.replicated.push_back(p0->name);
    }
    int64_t slot = L.rep_bytes;
    for (size_t i = 0; i < ranks.size(); ++i) {
        L.own_off[i] = slot;
        slot += L.own_bytes[i];
        L.entry_bytes = std::max(L.entry_bytes, L.rep_bytes + L.own_bytes[i]);
    }
    L.slot_bytes = slot;
    *out = std::move(L);
    return RAD_OK;
}

TierExec::Sizes TierExec::sizes(int64_t entry_bytes, int64_t max_ctx, int64_t block_size) {
    Sizes s;
    if (entry_bytes <= 0) return s;
    /* 16 MiB of staging per rank, clamped to 64..4096 entries (rationale at the bringup call site).
     * A copy out is four batches; a restore is a whole chain as long as the longest context. */
    s.max_batch = (int)std::min<int64_t>(4096, std::max<int64_t>(64, (16ll << 20) / entry_bytes));
    s.max_copy  = 4ll * s.max_batch;
    const int64_t bs = std::max<int64_t>(1, block_size);
    s.max_chain = (std::max<int64_t>(1, max_ctx) + bs - 1) / bs + 1;
    return s;
}

/* What device_init() allocates, each allocation rounded to the VRAM pool granule. */
int64_t TierExec::device_bytes(int64_t entry_bytes, int n_groups, const Sizes& s) {
    if (entry_bytes <= 0 || n_groups <= 0 || s.max_batch <= 0) return 0;
    const int64_t idx_n = (int64_t)n_groups * (s.max_copy + (int64_t)kRing * s.max_chain);
    return align_up((int64_t)s.max_batch * entry_bytes, (int64_t)RAD_ALIGN_POOL) +
           align_up(idx_n * (int64_t)sizeof(int32_t), (int64_t)RAD_ALIGN_POOL);
}

void TierExec::close() {
    /* Drain before freeing buffers the stream may still read. */
    if (stream_) (void)rad_stream_sync(stream_);
    if (stream_ && own_stream_) rad_stream_destroy(stream_);
    stream_ = nullptr; shared_ = nullptr; own_stream_ = false;
    release_device();
    if (paged_dev_)  { (void)rad_dev_set(device_); rad_dev_free(paged_dev_, RAD_MEM_DEVICE); paged_dev_ = nullptr; }
    if (paged_host_) { rad_dev_free(paged_host_, RAD_MEM_HOST_PINNED); paged_host_ = nullptr; }
    if (slot_dev_)   { (void)rad_dev_set(device_); rad_dev_free(slot_dev_, RAD_MEM_DEVICE); slot_dev_ = nullptr; }
    if (lay_host_)   { rad_dev_free(lay_host_, RAD_MEM_HOST_PINNED); lay_host_ = nullptr; }
    for (Count& c : counts_) {
        (void)rad_dev_set(device_);
        if (c.dev)  rad_dev_free(c.dev, RAD_MEM_DEVICE);
        if (c.host) rad_dev_free(c.host, RAD_MEM_HOST_PINNED);
    }
    counts_.clear();
    copy_out_ = false;
    gi_.clear();
    order_.clear();
    verify_pending_.clear();
    entry_bytes_ = 0;
    rep_bytes_ = own_bytes_ = 0;
    verify_ = verify_rep_ = false;
    out_sum_.clear();
    rep_sum_.clear();
    enabled_ = false;
    leader_ = false;
    reg_ = nullptr; kv_ = nullptr; tiers_ = nullptr;
    report_.clear();
}

void TierExec::release_device() {
    if (stage_)    { rad_dev_free(stage_, RAD_MEM_DEVICE); stage_ = nullptr; }
    if (idx_dev_)  { rad_dev_free(idx_dev_, RAD_MEM_DEVICE); idx_dev_ = nullptr; }
    if (idx_host_) { rad_dev_free(idx_host_, RAD_MEM_HOST_PINNED); idx_host_ = nullptr; }
    if (vrep_)     { rad_dev_free(vrep_, RAD_MEM_HOST_PINNED); vrep_ = nullptr; }
    if (move_stage_)    { rad_dev_free(move_stage_, RAD_MEM_DEVICE); move_stage_ = nullptr; }
    if (move_stage2_)   { rad_dev_free(move_stage2_, RAD_MEM_DEVICE); move_stage2_ = nullptr; }
    if (move_idx_dev_)  { rad_dev_free(move_idx_dev_, RAD_MEM_DEVICE); move_idx_dev_ = nullptr; }
    if (move_idx_host_) { rad_dev_free(move_idx_host_, RAD_MEM_HOST_PINNED); move_idx_host_ = nullptr; }
    move_bytes_ = 0;
    for (int i = 0; i < kRing; ++i) {
        if (ring_ev_[i]) rad_event_destroy(ring_ev_[i]);
        ring_ev_[i] = nullptr;
        ring_used_[i] = false;
    }
    if (ev_compute_) { rad_event_destroy(ev_compute_); ev_compute_ = nullptr; }
    if (ev_copy_)    { rad_event_destroy(ev_copy_);    ev_copy_ = nullptr; }
    if (ev_restore_) { rad_event_destroy(ev_restore_); ev_restore_ = nullptr; }
    if (ev_move_)    { rad_event_destroy(ev_move_);    ev_move_ = nullptr; }
    moved_ = false;
}

/* Builds the device side on first use, never at bringup (same rule as Ctx::lane_init()). Built
 * eagerly, it makes every decode step 65% longer even when nothing ever moves: on two R9700s with
 * Flash-Next at 200K context, 20.5 -> 33.9 ms/step, all of it device time. The cost is not the
 * transfers, the pinned arena size or the staging VRAM; it is the extra hardware queue (see
 * stream_ in tierexec.h). */
int TierExec::device_init() {
    if (stream_) return RAD_OK;
    RAD_TRY(rad_dev_set(device_));
    const int64_t ng = (int64_t)gi_.size();
    copy_idx_  = max_copy_;
    chain_idx_ = max_chain_;
    const int64_t idx_n = ng * (copy_idx_ + (int64_t)kRing * chain_idx_);
    stage_    = rad_dev_alloc((int64_t)max_batch_ * entry_bytes_, RAD_MEM_DEVICE);
    idx_dev_  = (int32_t*)rad_dev_alloc(idx_n * (int64_t)sizeof(int32_t), RAD_MEM_DEVICE);
    idx_host_ = (int32_t*)rad_dev_alloc(idx_n * (int64_t)sizeof(int32_t), RAD_MEM_HOST_PINNED);
    bool ev = rad_event_create(&ev_compute_) == RAD_OK && rad_event_create(&ev_copy_) == RAD_OK &&
              rad_event_create(&ev_restore_) == RAD_OK && rad_event_create(&ev_move_) == RAD_OK;
    for (int i = 0; i < kRing && ev; ++i) ev = rad_event_create(&ring_ev_[i]) == RAD_OK;
    /* The verify scratch is a diagnostic's: without it the replicated groups go unchecked, and
     * the tiers run as they would without RADIANCE_DEBUG_TIER_VERIFY. */
    if (verify_rep_) {
        vrep_ = (char*)rad_dev_alloc(copy_idx_ * rep_bytes_, RAD_MEM_HOST_PINNED);
        if (!vrep_) {
            RAD_WARN("kv tiers: RADIANCE_DEBUG_TIER_VERIFY: no pinned host memory (%s) for this "
                     "rank's copy of the replicated groups, so they are not compared",
                     humanb(copy_idx_ * rep_bytes_).c_str());
            verify_rep_ = false;
        }
    }
    if (!stage_ || !idx_dev_ || !idx_host_ || !ev) {
        release_device();
        return RAD_E_NOMEM;
    }
    /* Residency staging, only when a group declares a window (spec §7.3). One block of the widest
     * such group, a batch of them, and the indices to gather with: nothing here is touched by the
     * idle tiers, so the two paths cannot share a buffer by accident. */
    for (const GroupInfo& x : gi_)
        if (!x.zero && kv_->plan(x.g) && kv_->plan(x.g)->residency())
            move_bytes_ = std::max(move_bytes_, x.n_layers * x.frag);
    if (move_bytes_ > 0) {
        move_stage_    = rad_dev_alloc((int64_t)kMoveBatch * move_bytes_, RAD_MEM_DEVICE);
        move_idx_dev_  = (int32_t*)rad_dev_alloc((int64_t)kMoveBatch * sizeof(int32_t), RAD_MEM_DEVICE);
        move_idx_host_ = (int32_t*)rad_dev_alloc(kMoveIdxCap * (int64_t)sizeof(int32_t),
                                                 RAD_MEM_HOST_PINNED);
        move_idx_off_  = 0;
        if (!move_stage_ || !move_idx_dev_ || !move_idx_host_) { release_device(); return RAD_E_NOMEM; }
        int64_t sbytes = 0;
        for (const GroupInfo& x : gi_)
            if (!x.zero && x.to_store) sbytes = std::max(sbytes, x.n_layers * x.sfrag);
        if (sbytes > 0 &&
            !(move_stage2_ = rad_dev_alloc((int64_t)kMoveBatch * sbytes, RAD_MEM_DEVICE))) {
            release_device();
            return RAD_E_NOMEM;
        }
    }
    /* Borrow the mover's maintenance stream if there is one; both move bytes nobody waits for,
     * and a fourth hardware queue costs about 13 ms per decode step. Never the step's own
     * stream, or a copy out would become a dependency of the next token. */
    if (shared_) {
        stream_ = shared_;
        own_stream_ = false;
    } else if (rad_stream_create(&stream_, /*high_priority=*/0) == RAD_OK) {
        own_stream_ = true;
    } else {
        stream_ = nullptr;
        release_device();
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}

/* One (group, layer) launch. Staging rows are at stride entry_bytes_ from `stage_base` (the first
 * entry's slot), so each fragment lands inside its own contiguous staging entry. */
int TierExec::run_rows(const Resolved& r, bool scatter, const GroupInfo& gi, int64_t layer,
                       void* stage_base, const int32_t* d_idx, int64_t n_jobs,
                       int64_t pitch, int64_t stage_off) {
    RadTensor pool{};
    pool.dtype = RAD_U8;
    pool.rank = 2;
    pool.shape[0] = gi.n_blocks;  pool.shape[1] = gi.frag;
    pool.stride[0] = gi.frag;     pool.stride[1] = 1;
    pool.data = (char*)gi.base + layer * gi.layer_stride;

    RadTensor idx{};
    idx.dtype = RAD_I32; idx.rank = 1;
    idx.shape[0] = n_jobs; idx.stride[0] = 1;
    idx.data = (void*)d_idx;

    RadTensor pack{};
    pack.dtype = RAD_U8;
    pack.rank = 2;
    pack.shape[0] = n_jobs;      pack.shape[1] = gi.frag;
    pack.stride[0] = pitch;          /* one entry's stride, not one row's */
    pack.stride[1] = 1;
    pack.data = (char*)stage_base + stage_off + layer * gi.frag;

    RadTensor t3[3];
    if (scatter) { t3[0] = pack; t3[1] = idx; t3[2] = pool; }
    else         { t3[0] = pool; t3[1] = idx; t3[2] = pack; }

    /* Pass the resolved geometry, which carries the winning kernel's derived keys. Its M is the
     * band's upper bound; kernels take the row count from the operand extent. */
    RadArgs a{};
    a.t = t3; a.n_t = 3;
    a.p = r.geom.params(); a.n_p = r.geom.n_params();
    a.instance = r.instance;
    return r.row->info->launch(&a, stream_);
}

int TierExec::run_xc(const Resolved& r, const void* src, int64_t src_row, void* dst,
                     int64_t dst_row, int64_t rows) {
    RadTensor t2[2] = {};
    for (int i = 0; i < 2; ++i) {
        t2[i].dtype = RAD_U8;
        t2[i].rank = 2;
        t2[i].shape[0] = rows;
        t2[i].shape[1] = i ? dst_row : src_row;
        t2[i].stride[0] = t2[i].shape[1];
        t2[i].stride[1] = 1;
    }
    t2[0].data = (void*)src;
    t2[1].data = dst;
    RadArgs a{};
    a.t = t2; a.n_t = 2;
    a.p = r.geom.params(); a.n_p = r.geom.n_params();
    a.instance = r.instance;
    return r.row->info->launch(&a, stream_);
}

/* ------------------------------------------------------------------ RADIANCE_DEBUG_TIER_VERIFY */

/* FNV-1a, continued from `h` so one sum can run over the two windows of a slot. */
static uint64_t tier_fnv_from(uint64_t h, const void* p, int64_t n) {
    const unsigned char* b = (const unsigned char*)p;
    for (int64_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
static uint64_t tier_fnv(const void* p, int64_t n) {
    return tier_fnv_from(1469598103934665603ull, p, n);
}

/* One sum over the entry as the slot holds it: the replicated window, then this rank's own. */
uint64_t TierExec::entry_sum(const char* slot) const {
    const uint64_t h = tier_fnv(slot, rep_bytes_);
    return tier_fnv_from(h, slot + slot_off_, own_bytes_);
}

void TierExec::verify_note(const BlockHash& key, const char* slot) {
    if (!verify_ || !leader_ || !slot) return;
    out_sum_[key] = entry_sum(slot);
}

void TierExec::verify_check(const BlockHash& key, const char* slot) {
    if (!verify_ || !leader_ || !slot) return;
    auto it = out_sum_.find(key);
    if (it == out_sum_.end()) return;               /* never went out through this process */
    const uint64_t back = entry_sum(slot);
    if (back == it->second) return;
    RAD_ERR("kv tiers: the attention blocks for %s came back different (%016llx out, %016llx in); "
            "the tier moved the wrong bytes",
            key.str().c_str(), (unsigned long long)it->second, (unsigned long long)back);
}

bool TierExec::take_replicated_sum(const BlockHash& key, uint64_t* sum) {
    auto it = rep_sum_.find(key);
    if (it == rep_sum_.end()) return false;
    if (sum) *sum = it->second;
    rep_sum_.erase(it);
    return true;
}

bool TierExec::verify_replicated(const TierJob& j, int rank, uint64_t follower_sum) {
    if (!verify_ || !leader_ || rep_bytes_ <= 0 || !tiers_ || j.host_slot < 0) return false;
    const char* slot = (const char*)tiers_->arena().at(j.host_slot);
    if (!slot) return false;
    ++st_.rep_checked;
    const uint64_t mine = tier_fnv(slot, rep_bytes_);
    if (mine == follower_sum) return true;
    ++st_.rep_mismatch;
    RAD_ERR("kv tiers: the replicated groups of %s differ between rank 0 (%016llx) and rank %d "
            "(%016llx). The plugin declared them replicated (rad_kv_group_replicated) and rank %d "
            "holds other bytes; its restore would read rank 0's.",
            j.key.str().c_str(), (unsigned long long)mine, rank, (unsigned long long)follower_sum,
            rank);
    return false;
}

/* ------------------------------------------------------------------ helpers */

/* Like RAD_TRY, but the first failure is logged with the call it came from. A failing tier is
 * otherwise invisible: entries stay put and the server only gets slower. */
#define TIER_TRY(call, what)                                                                  \
    do { const int _s = (call);                                                               \
         if (_s != RAD_OK) { warn_once(what, _s); return _s; } } while (0)

void TierExec::warn_once(const char* what, int st) {
    if (warned_) return;
    warned_ = true;
    RAD_WARN("kv tiers: %s failed with %s. The entry stays where it was; this is said once, and "
             "the running failure count is in /sessions.", what, rad_strerror(st));
}

bool TierExec::blocks_backed(const std::vector<int32_t>& blocks, const char* what) {
    if (!kv_) return false;
    for (size_t k = 0; k < gi_.size(); ++k) {
        if (k >= blocks.size()) continue;          /* an absent group is the gather's -1 row */
        const int32_t b = blocks[k];
        if (b < 0 || gi_[k].zero) continue;
        const int64_t live = kv_->live_blocks(gi_[k].g);
        if (b < live) continue;
        if (warned_unbacked_) return false;
        warned_unbacked_ = true;
        RAD_ERR("kv tier %s: block %d of group %d is at or past this rank's live mark of %lld, so "
                "it is lent to the expert slab and holds weights. The transfer is refused; the "
                "entry stays where it is.", what, (int)b, (int)gi_[k].g, (long long)live);
        return false;
    }
    return true;
}

int TierExec::copy_runs(const std::vector<TierJob*>& order, size_t first, int64_t n, bool out) {
    int64_t k = 0;
    while (k < n) {
        const int32_t s0 = order[first + (size_t)k]->host_slot;
        int64_t run = 1;
        while (k + run < n && order[first + (size_t)(k + run)]->host_slot == s0 + run) ++run;
        char* host = (char*)tiers_->arena().at(s0);
        if (!host) return RAD_E_STATE;
        char* dev = (char*)stage_ + k * entry_bytes_;
        int st = RAD_OK;
        if (out) {
            /* The replicated window is the leader's to write; a verifying follower keeps its own
             * gather beside the copy, indexed by the job's place in the pass. */
            if (rep_bytes_ > 0 && leader_)
                st = rad_memcpy_2d_async(host, slot_bytes_, dev, entry_bytes_, rep_bytes_, run,
                                         stream_);
            if (st == RAD_OK && verify_rep_ && vrep_)
                st = rad_memcpy_2d_async(vrep_ + (int64_t)(first + (size_t)k) * rep_bytes_,
                                         rep_bytes_, dev, entry_bytes_, rep_bytes_, run, stream_);
            if (st == RAD_OK && own_bytes_ > 0)
                st = rad_memcpy_2d_async(host + slot_off_, slot_bytes_, dev + rep_bytes_,
                                         entry_bytes_, own_bytes_, run, stream_);
        } else {
            if (rep_bytes_ > 0)
                st = rad_memcpy_2d_async(dev, entry_bytes_, host, slot_bytes_, rep_bytes_, run,
                                         stream_);
            if (st == RAD_OK && own_bytes_ > 0)
                st = rad_memcpy_2d_async(dev + rep_bytes_, entry_bytes_, host + slot_off_,
                                         slot_bytes_, own_bytes_, run, stream_);
        }
        if (st != RAD_OK) return st;
        k += run;
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ copy on idle */

int TierExec::issue_copy(std::vector<TierJob>& jobs, RadStream compute) {
    if (!enabled_ || jobs.empty()) return RAD_OK;
    TIER_TRY(rad_dev_set(device_), "selecting the rank device");
    TIER_TRY(device_init(), "building the transfer staging");
    const int st = enqueue_copy(jobs, compute);
    /* Drain a partly enqueued copy before its host slots are freed by the report, or a queued
     * transfer would land on the slot's next owner. */
    if (st != RAD_OK) (void)rad_stream_sync(stream_);
    return st;
}

int TierExec::enqueue_copy(std::vector<TierJob>& jobs, RadStream compute) {
    order_.clear();
    for (TierJob& j : jobs) if (!j.blocks.empty()) order_.push_back(&j);
    const int64_t n = (int64_t)order_.size();
    if (n > copy_idx_) return RAD_E_INVAL;
    /* Check every id before any launch: a gather over an unbacked id faults the whole card. */
    for (TierJob* j : order_)
        if (!blocks_backed(j->blocks, "copy to host")) return RAD_E_STATE;

    /* Stream-ordered after everything given to compute, including the step that wrote these. */
    TIER_TRY(rad_event_record(ev_compute_, compute), "ordering the copy after the step");
    TIER_TRY(rad_event_wait(stream_, ev_compute_), "ordering the copy after the step");

    if (n > 0) {
        const int64_t ng = (int64_t)gi_.size();
        int32_t* h = idx_host_;                           /* the copy region */
        for (int64_t g = 0; g < ng; ++g)
            for (int64_t k = 0; k < n; ++k) {
                const std::vector<int32_t>& b = order_[(size_t)k]->blocks;
                h[g * n + k] = (size_t)g < b.size() ? b[(size_t)g] : -1;
            }
        TIER_TRY(rad_memcpy_async(idx_dev_, h, ng * n * (int64_t)sizeof(int32_t), stream_),
                 "the block-index upload");
        for (int64_t s = 0; s < n; s += max_batch_) {
            const int64_t c = std::min<int64_t>(max_batch_, n - s);
            for (size_t gidx = 0; gidx < gi_.size(); ++gidx)
                for (int64_t l = 0; l < gi_[gidx].n_layers && gathers(gi_[gidx]); ++l)
                    TIER_TRY(run_rows(gi_[gidx].gather, /*scatter=*/false, gi_[gidx], l, stage_,
                                      idx_dev_ + (int64_t)gidx * n + s, c, entry_bytes_,
                                      gi_[gidx].off),
                             "the gather out of the pool");
            TIER_TRY(copy_runs(order_, (size_t)s, c, /*out=*/true), "the copy to host memory");
        }
        st_.bytes_out += n * (leader_ ? entry_bytes_ : own_bytes_);
    }

    TIER_TRY(rad_event_record(ev_copy_, stream_), "marking the copy");
    copy_out_ = true;
    ++st_.passes;
    st_.jobs += (int64_t)jobs.size();
    if (verify_) {
        int64_t ord = 0;
        for (const TierJob& j : jobs) {
            const bool moved = !j.blocks.empty();
            verify_pending_.push_back({j.key, moved ? j.host_slot : -1, moved ? ord : -1});
            if (moved) ++ord;
        }
    }
    return RAD_OK;
}
/* Window residency transfers (spec §7.3). Same split as everywhere else in this file: the manager
 * names the blocks and holds the cold/resident bit, this runs the bytes.
 *
 * A demotion is ordered after the step that wrote the blocks (`compute`), so the gather reads what
 * the kernels wrote. A promotion is ordered on the transfer stream and then made a dependency of
 * `compute`, so the step that reads those blocks runs after they land -- the only wait this costs,
 * and it is the step's own, not the host's. */
int TierExec::issue_moves(const std::vector<BlockMove>& moves, RadStream compute, bool to_host) {
    if (moves.empty()) return RAD_OK;
    if (!enabled_ || !kv_) return RAD_E_STATE;
    if ((int64_t)moves.size() > kMoveBatch) return RAD_E_INVAL;
    const int32_t g = moves[0].group;
    for (const BlockMove& m : moves)
        if (m.group != g || m.block < 0) return RAD_E_INVAL;
    const GroupInfo* gi = nullptr;
    for (const GroupInfo& x : gi_) if (x.g == g) gi = &x;
    if (!gi) return RAD_E_INVAL;
    /* A rank that keeps no cache for this group (rad_kv_group_zero) has no bytes of the block to
     * move: the other ranks' shards still go, and this one is done. */
    if (gi->zero) return RAD_OK;
    if (RAD_OK != device_init()) return RAD_E_DEVICE;
    if (!move_stage_ || move_bytes_ <= 0) return RAD_E_INVAL;
    /* Every id must be a block of this group's carve: a gather over an unbacked id faults the
     * whole card, and a scatter into one overwrites expert weights in use. */
    std::vector<int32_t> ids;
    ids.reserve(moves.size());
    for (const BlockMove& m : moves) ids.push_back(m.block);
    if (!blocks_backed(ids, to_host ? "residency demote" : "residency promote")) return RAD_E_STATE;
    /* A move is one the manager handed out in this direction and has not yet been told landed
     * (a demote still holds its pool block; a promote still holds its slot); anything else is a
     * caller that reordered or invented its list. */
    /* The emulation's demotions name no slot: the leader holds each block's mark instead. */
    const bool emul = to_host && kv_->residency_emulating();
    for (const BlockMove& m : moves)
        if (emul ? m.host_slot != -1 || (leader_ && !kv_->residency_emulated(g, m.block))
                 : m.host_slot < 0 || (leader_ && !kv_->residency_move_pending(m, to_host)))
            return RAD_E_STATE;
    if (to_host ? !gi->gather.row : !gi->scatter.row) return RAD_E_NOKERNEL;
    const bool tiered = gi->to_store && gi->to_pool && move_stage2_;
    if (emul && !tiered) return RAD_OK;            /* a round trip through the same bytes */
    /* Two demote layouts (KVGroupPlan::split): the batch with its wide blocks first, a stable
     * partition, so each layout's blocks are one run of the stage and take one transcode. Any
     * other group names layout 0 only, and its order is the caller's. */
    const bool split = tiered && gi->to_store2 && gi->to_pool2;
    std::vector<BlockMove> mv(moves);
    if (split)
        std::stable_partition(mv.begin(), mv.end(), [](const BlockMove& m) { return m.layout == 0; });
    int64_t n0 = 0;
    for (const BlockMove& m : mv) {
        if (m.layout != 0 && (!split || m.layout != 1)) return RAD_E_INVAL;
        n0 += m.layout == 0;
    }
    if (split && !emul && !slot_dev_) return RAD_E_STATE;   /* the pager could not tell the layouts apart */

    const int64_t n = (int64_t)mv.size();
    const int64_t blk = gi->n_layers * gi->frag;
    /* What a slot holds of the group's block: the demote layout's rows when tiered; a narrow
     * block (split) the narrow rows at the front of each layer's fragment. */
    const int64_t sblk = tiered ? gi->n_layers * gi->sfrag : blk;
    const int64_t nblk = split ? gi->n_layers * gi->sfrag2 : sblk;
    const int64_t rb = gi->n_layers * gi->rows;          /* a block's rows, tiered */
    /* Block i's bytes in the second stage: the wide run, then the narrow one packed. */
    auto off2 = [&](int64_t i) { return i < n0 ? i * sblk : n0 * sblk + (i - n0) * nblk; };
    if (blk > move_bytes_) return RAD_E_INVAL;

    /* Both directions after everything compute had been given when this was issued. A demotion
     * reads blocks the steps wrote. A promotion writes into a block the pool just handed out, and
     * a step issued ahead may still read that block under its previous owner -- a block freed
     * when its demotion landed, or a re-pointed one freed at once -- so the write must not overtake
     * those reads. A stream wait, not a host sync. */
    if (compute) {
        TIER_TRY(rad_event_record(ev_compute_, compute), "ordering the move after the steps");
        TIER_TRY(rad_event_wait(stream_, ev_compute_), "ordering the move after the steps");
    }

    /* Each batch its own entries (see move_idx_host_): the engine issues a sweep's batches back to
     * back, and with one shared buffer every batch queued behind a busy compute stream gathered the
     * last batch's blocks. A wrap waits for the uploads still queued, rarely: the ring holds sixteen
     * full decode sweeps, and a sweep goes out only once the last one landed. */
    if (move_idx_off_ + n > kMoveIdxCap) {
        TIER_TRY(rad_stream_sync(stream_), "reusing the residency block indices");
        move_idx_off_ = 0;
    }
    int32_t* idx_host = move_idx_host_ + move_idx_off_;
    move_idx_off_ += n;
    for (int64_t i = 0; i < n; ++i) idx_host[i] = mv[(size_t)i].block;
    TIER_TRY(rad_memcpy_async(move_idx_dev_, idx_host, n * (int64_t)sizeof(int32_t), stream_),
             "the residency block-index upload");

    if (to_host) {
        for (int64_t l = 0; l < gi->n_layers; ++l)
            TIER_TRY(run_rows(gi->gather, /*scatter=*/false, *gi, l, move_stage_, move_idx_dev_, n,
                              blk, /*stage_off=*/0),
                     "the residency gather out of the pool");
        /* The staged blocks are back to back, each its layers' fragments, so the stage is one
         * plane of rows in either layout: a run per layout. */
        if (tiered && n0 > 0)
            TIER_TRY(run_xc(gi->to_store, move_stage_, gi->rf, move_stage2_, gi->rt, n0 * rb),
                     "the residency requantise into the demote layout");
        if (split && n > n0)
            TIER_TRY(run_xc(gi->to_store2, (char*)move_stage_ + n0 * blk, gi->rf,
                            (char*)move_stage2_ + off2(n0), gi->rt2, (n - n0) * rb),
                     "the residency requantise into the second demote layout");
        if (emul) {
            if (n0 > 0)
                TIER_TRY(run_xc(gi->to_pool, move_stage2_, gi->rt, move_stage_, gi->rf, n0 * rb),
                         "the emulated expand back into the resident layout");
            if (split && n > n0)
                TIER_TRY(run_xc(gi->to_pool2, (char*)move_stage2_ + off2(n0), gi->rt2,
                                (char*)move_stage_ + n0 * blk, gi->rf, (n - n0) * rb),
                         "the emulated expand back from the second demote layout");
            for (int64_t l = 0; l < gi->n_layers; ++l)
                TIER_TRY(run_rows(gi->scatter, /*scatter=*/true, *gi, l, move_stage_,
                                  move_idx_dev_, n, blk, /*stage_off=*/0),
                         "the emulated round trip back into the pool");
        } else {
            const char* out = tiered ? (const char*)move_stage2_ : (const char*)move_stage_;
            for (int64_t i = 0; i < n; ++i) {
                const int32_t slot = mv[(size_t)i].host_slot;
                void* host = kv_->residency_slot_ptr(g, slot);
                if (!host) return RAD_E_STATE;
                if (i < n0)
                    TIER_TRY(rad_memcpy_async(host, out + off2(i), sblk, stream_),
                             "the residency copy to host memory");
                else      /* each layer's narrow rows at the front of its fragment in the slot */
                    TIER_TRY(rad_memcpy_2d_async(host, gi->sfrag, out + off2(i), gi->sfrag2,
                                                 gi->sfrag2, gi->n_layers, stream_),
                             "the residency copy to host memory");
                /* The slot's layout byte, behind its bytes on the same stream (slot_table). */
                if (split)
                    TIER_TRY(rad_memcpy_async(slot_dev_ + slot, lay_host_ + (i < n0 ? 0 : 1), 1,
                                              stream_),
                             "the residency slot's layout");
                mv_.out_bytes += i < n0 ? sblk : nblk;
            }
            mv_.out_blocks += n;
        }
        /* The steps after this read the demoted blocks through the pager (or the emulation's new
         * bytes in place), so they wait for it (see the header). */
        TIER_TRY(rad_event_record(ev_restore_, stream_), "marking the residency demotion");
        if (compute) {
            TIER_TRY(rad_event_wait(compute, ev_restore_),
                     "ordering the next step after the residency demotion");
        }
    } else {
        char* in = tiered ? (char*)move_stage2_ : (char*)move_stage_;
        for (int64_t i = 0; i < n; ++i) {
            const void* host = kv_->residency_slot_ptr(g, mv[(size_t)i].host_slot);
            if (!host) return RAD_E_STATE;
            if (i < n0)
                TIER_TRY(rad_memcpy_async(in + off2(i), host, sblk, stream_),
                         "the residency copy out of host memory");
            else
                TIER_TRY(rad_memcpy_2d_async(in + off2(i), gi->sfrag2, host, gi->sfrag,
                                             gi->sfrag2, gi->n_layers, stream_),
                         "the residency copy out of host memory");
            mv_.in_bytes += i < n0 ? sblk : nblk;
        }
        if (tiered && n0 > 0)
            TIER_TRY(run_xc(gi->to_pool, move_stage2_, gi->rt, move_stage_, gi->rf, n0 * rb),
                     "the residency expand into the resident layout");
        if (split && n > n0)
            TIER_TRY(run_xc(gi->to_pool2, (char*)move_stage2_ + off2(n0), gi->rt2,
                            (char*)move_stage_ + n0 * blk, gi->rf, (n - n0) * rb),
                     "the residency expand from the second demote layout");
        for (int64_t l = 0; l < gi->n_layers; ++l)
            TIER_TRY(run_rows(gi->scatter, /*scatter=*/true, *gi, l, move_stage_, move_idx_dev_, n,
                              blk, /*stage_off=*/0),
                     "the residency scatter into the pool");
        /* The step that follows reads these blocks: it waits for them, not for the host. */
        TIER_TRY(rad_event_record(ev_restore_, stream_), "marking the residency restore");
        if (compute) {
            TIER_TRY(rad_event_wait(compute, ev_restore_),
                     "ordering the step after the residency restore");
        }
        mv_.in_blocks += n;
    }
    /* Last, so the event covers the whole batch on the transfer stream, in either direction. */
    TIER_TRY(rad_event_record(ev_move_, stream_), "marking the residency move");
    moved_ = true;
    ++st_.passes;
    return RAD_OK;
}

bool TierExec::moves_landed() const {
    if (!moved_ || !ev_move_) return true;
    return rad_event_query(ev_move_) != 0;
}

int64_t* TierExec::paged_counter() {
    if (paged_dev_) return paged_dev_;
    if (rad_dev_set(device_) != RAD_OK) return nullptr;
    paged_dev_  = (int64_t*)rad_dev_alloc(sizeof(int64_t), RAD_MEM_DEVICE);
    paged_host_ = (int64_t*)rad_dev_alloc(sizeof(int64_t), RAD_MEM_HOST_PINNED);
    /* Zeroed on a stream of its own, once: no stream exists yet (device_init is lazy), and the
     * default stream is not valid on every backend. */
    RadStream zs = nullptr;
    const bool zeroed = paged_dev_ && paged_host_ && rad_stream_create(&zs, 0) == RAD_OK &&
                        rad_memset_async(paged_dev_, 0, sizeof(int64_t), zs) == RAD_OK &&
                        rad_stream_sync(zs) == RAD_OK;
    if (zs) rad_stream_destroy(zs);
    if (!zeroed) {
        if (paged_dev_)  rad_dev_free(paged_dev_, RAD_MEM_DEVICE);
        if (paged_host_) rad_dev_free(paged_host_, RAD_MEM_HOST_PINNED);
        paged_dev_ = nullptr; paged_host_ = nullptr;
    }
    return paged_dev_;
}

/* A zeroed device allocation, on a stream of its own as paged_counter()'s. */
static void* zeroed_device(int64_t bytes) {
    void* p = rad_dev_alloc(bytes, RAD_MEM_DEVICE);
    RadStream zs = nullptr;
    const bool ok = p && rad_stream_create(&zs, 0) == RAD_OK &&
                    rad_memset_async(p, 0, bytes, zs) == RAD_OK && rad_stream_sync(zs) == RAD_OK;
    if (zs) rad_stream_destroy(zs);
    if (!ok && p) { rad_dev_free(p, RAD_MEM_DEVICE); p = nullptr; }
    return p;
}

uint8_t* TierExec::slot_table(int64_t slots) {
    if (slot_dev_) return slot_dev_;
    if (slots <= 0 || rad_dev_set(device_) != RAD_OK) return nullptr;
    slot_dev_ = (uint8_t*)zeroed_device(slots);
    lay_host_ = (uint8_t*)rad_dev_alloc(2, RAD_MEM_HOST_PINNED);
    if (!slot_dev_ || !lay_host_) {
        if (slot_dev_) rad_dev_free(slot_dev_, RAD_MEM_DEVICE);
        if (lay_host_) rad_dev_free(lay_host_, RAD_MEM_HOST_PINNED);
        slot_dev_ = nullptr; lay_host_ = nullptr;
        return nullptr;
    }
    lay_host_[0] = 0; lay_host_[1] = 1;
    return slot_dev_;
}

uint32_t* TierExec::count_buffer(int32_t g, int64_t n, const uint32_t** host) {
    for (const Count& c : counts_)
        if (c.g == g) { if (host) *host = c.host; return c.dev; }
    if (n <= 0 || rad_dev_set(device_) != RAD_OK) return nullptr;
    Count c;
    c.g = g; c.n = n;
    c.dev = (uint32_t*)zeroed_device(n * (int64_t)sizeof(uint32_t));
    c.host = (uint32_t*)rad_dev_alloc(n * (int64_t)sizeof(uint32_t), RAD_MEM_HOST_PINNED);
    if (!c.dev || !c.host) {
        if (c.dev)  rad_dev_free(c.dev, RAD_MEM_DEVICE);
        if (c.host) rad_dev_free(c.host, RAD_MEM_HOST_PINNED);
        return nullptr;
    }
    std::memset(c.host, 0, (size_t)n * sizeof(uint32_t));
    counts_.push_back(c);
    if (host) *host = c.host;
    return c.dev;
}

int TierExec::count_snapshot(RadStream s) {
    for (const Count& c : counts_)
        if (rad_memcpy_async(c.host, c.dev, c.n * (int64_t)sizeof(uint32_t), s) != RAD_OK)
            return RAD_E_DEVICE;
    return RAD_OK;
}

int TierExec::count_clear(const std::vector<std::pair<int32_t, int32_t>>& blocks, RadStream s) {
    if (blocks.empty() || counts_.empty()) return RAD_OK;
    std::vector<std::pair<int32_t, int32_t>> b(blocks);
    std::sort(b.begin(), b.end());
    for (size_t i = 0; i < b.size();) {
        size_t j = i + 1;
        while (j < b.size() && b[j].first == b[i].first && b[j].second <= b[j - 1].second + 1) ++j;
        const Count* c = nullptr;
        for (const Count& x : counts_) if (x.g == b[i].first) c = &x;
        const int64_t lo = b[i].second, hi = b[j - 1].second + 1;
        if (c && lo >= 0 && hi <= c->n &&
            rad_memset_async(c->dev + lo, 0, (hi - lo) * (int64_t)sizeof(uint32_t), s) != RAD_OK)
            return RAD_E_DEVICE;
        i = j;
    }
    return RAD_OK;
}

int64_t TierExec::paged_in() {
    if (!paged_dev_ || !paged_host_ || rad_dev_set(device_) != RAD_OK) return 0;
    /* Before the first move there is no transfer stream, and nothing can be cold, so the count is
     * zero; the default stream would serialise with the step. */
    if (!stream_) return 0;
    *paged_host_ = 0;
    if (rad_memcpy_async(paged_host_, paged_dev_, sizeof(int64_t), stream_) != RAD_OK ||
        rad_stream_sync(stream_) != RAD_OK)
        return 0;
    return *paged_host_;
}


int TierExec::copy_state() {
    if (!copy_out_) return 1;
    (void)rad_dev_set(device_);
    const int q = rad_event_query(ev_copy_);
    if (q == 0) return 0;
    copy_out_ = false;
    if (q > 0 && verify_) {
        for (const VerifyNote& v : verify_pending_) {
            if (v.host_slot < 0) continue;
            if (leader_)
                verify_note(v.key, (const char*)tiers_->arena().at(v.host_slot));
            else if (verify_rep_ && vrep_ && v.ord >= 0)
                rep_sum_[v.key] = tier_fnv(vrep_ + v.ord * rep_bytes_, rep_bytes_);
        }
    }
    verify_pending_.clear();
    if (q < 0) warn_once("the copy to host memory", q);
    return q > 0 ? 1 : q;
}

int TierExec::wait_copy() {
    if (!copy_out_) return 1;
    (void)rad_dev_set(device_);
    const int st = rad_event_sync(ev_copy_);
    if (st != RAD_OK) warn_once("waiting for the copy to host memory", st);
    return copy_state();
}

/* ------------------------------------------------------------------ restore */

void TierExec::quiesce() {
    if (!enabled_ || !stream_) return;
    if (rad_dev_set(device_) < 0) return;
    (void)rad_stream_sync(stream_);
}

int TierExec::issue_restore(std::vector<TierJob>& jobs, RadStream compute,
                            std::vector<char>* ok) {
    ok->assign(jobs.size(), 0);
    if (!enabled_ || jobs.empty()) return RAD_OK;
    TIER_TRY(rad_dev_set(device_), "selecting the rank device");
    TIER_TRY(device_init(), "building the transfer staging");
    const size_t ng = gi_.size();

    /* Block ids are shared across ranks (one block table indexes every shard), so only the leader
     * chooses and reserves them; its allocator is the one the prefix cache (rank 0's) references.
     * Followers write into the ids they are handed.
     *
     * The whole chain is reserved before anything moves, with one loan recall. If the pool runs
     * short the chain is cut there: a hit needs a contiguous prefix, so entries past a gap are
     * useless. */
    if (leader_) {
        const int64_t n = (int64_t)jobs.size();
        std::vector<std::vector<int32_t>> ids(ng);
        int64_t m = n;
        for (size_t k = 0; k < ng; ++k) {
            ids[k].reserve((size_t)n);
            m = std::min(m, kv_->reserve_blocks(gi_[k].g, m, &ids[k]));
        }
        for (size_t k = 0; k < ng; ++k)
            for (size_t i = (size_t)m; i < ids[k].size(); ++i) kv_->release_block(gi_[k].g, ids[k][i]);
        for (size_t i = 0; i < jobs.size(); ++i) {
            jobs[i].blocks.clear();
            if ((int64_t)i >= m) continue;
            jobs[i].blocks.resize(ng);
            for (size_t k = 0; k < ng; ++k) jobs[i].blocks[k] = ids[k][i];
        }
    }
    size_t m = 0;
    while (m < jobs.size() && jobs[m].blocks.size() == ng) ++m;
    /* The leader chose ids against its own live mark and the recall should have moved every
     * follower's with it; verify that here. */
    for (size_t i = 0; i < m; ++i)
        if (!blocks_backed(jobs[i].blocks, "restore")) { m = i; break; }

    /* Followers find the chain by which jobs carry block ids, so every job past the cut releases
     * its ids and carries none; otherwise a follower would copy into blocks the cache does not own. */
    if (leader_) {
        for (size_t i = m; i < jobs.size(); ++i) {
            for (size_t k = 0; k < ng && k < jobs[i].blocks.size(); ++k)
                if (jobs[i].blocks[k] >= 0) kv_->release_block(gi_[k].g, jobs[i].blocks[k]);
            jobs[i].blocks.clear();
        }
    }
    if (m == 0) return RAD_OK;

    order_.clear();
    for (size_t i = 0; i < m; ++i) order_.push_back(&jobs[i]);
    for (size_t i = 0; i < m; ++i)
        verify_check(jobs[i].key, (const char*)tiers_->arena().at(jobs[i].host_slot));

    /* From here a failure leaves blocks partly overwritten. Mark jobs started first, so report()
     * drops a failed started job instead of trusting its host copy. The compute stream waits on
     * whatever was enqueued, success or not, because a failed job's blocks return to the pool and
     * must not be reused before the queued transfers into them. */
    for (size_t i = 0; i < m; ++i) jobs[i].started = true;
    const int ri = ring_next_;
    ring_next_ = (ring_next_ + 1) % kRing;
    int32_t* h = idx_host_ + ng * copy_idx_ + (int64_t)ri * (int64_t)ng * chain_idx_;
    int32_t* d = idx_dev_  + ng * copy_idx_ + (int64_t)ri * (int64_t)ng * chain_idx_;
    int st = RAD_OK;
    size_t done = 0;
    for (size_t s = 0; s < m && st == RAD_OK; s += (size_t)chain_idx_) {
        const int64_t c = std::min<int64_t>(chain_idx_, (int64_t)(m - s));
        if (ring_used_[ri] && (st = rad_event_sync(ring_ev_[ri])) != RAD_OK) break;
        for (size_t g = 0; g < ng; ++g)
            for (int64_t k = 0; k < c; ++k) h[(int64_t)g * c + k] = jobs[s + (size_t)k].blocks[g];
        if ((st = rad_memcpy_async(d, h, (int64_t)ng * c * (int64_t)sizeof(int32_t), stream_)) != RAD_OK)
            break;
        if ((st = rad_event_record(ring_ev_[ri], stream_)) != RAD_OK) break;
        ring_used_[ri] = true;
        for (int64_t b = 0; b < c && st == RAD_OK; b += max_batch_) {
            const int64_t cc = std::min<int64_t>(max_batch_, c - b);
            if ((st = copy_runs(order_, s + (size_t)b, cc, /*out=*/false)) != RAD_OK) break;
            for (size_t gidx = 0; gidx < ng && st == RAD_OK; ++gidx)
                for (int64_t l = 0; l < gi_[gidx].n_layers && st == RAD_OK && !gi_[gidx].zero; ++l)
                    st = run_rows(gi_[gidx].scatter, /*scatter=*/true, gi_[gidx], l, stage_,
                                  d + (int64_t)gidx * c + b, cc, entry_bytes_, gi_[gidx].off);
        }
        if (st == RAD_OK) done = s + (size_t)c;
    }
    if (st != RAD_OK) warn_once("the restore into the pool", st);
    for (size_t i = 0; i < done; ++i) (*ok)[i] = 1;
    st_.bytes_in += (int64_t)done * entry_bytes_;

    /* The reading step waits on the device; if the event cannot be enqueued, fall back to a host
     * wait. */
    if (rad_event_record(ev_restore_, stream_) != RAD_OK ||
        rad_event_wait(compute, ev_restore_) != RAD_OK) {
        (void)rad_stream_sync(stream_);
    }
    ++st_.passes;
    st_.jobs += (int64_t)m;
    return st;
}

/* ------------------------------------------------------------------ the report */

void TierExec::report_pass(std::vector<TierJob>& jobs, const std::vector<char>& ok,
                           uint64_t now_ns) {
    if (!enabled_ || !leader_) return;
    bool quiet = false;
    for (size_t i = 0; i < jobs.size(); ++i) {
        TierJob& j = jobs[i];
        const bool good = i < ok.size() && ok[i];
        /* A failed promotion releases the leader's blocks once nothing queued against them can
         * still land; after report() nothing names them again. */
        if (j.kind == TierJob::Kind::ToDevice && !good) {
            if (!quiet) { quiesce(); quiet = true; }
            for (size_t k = 0; k < gi_.size() && k < j.blocks.size(); ++k)
                if (j.blocks[k] >= 0) kv_->release_block(gi_[k].g, j.blocks[k]);
            j.blocks.clear();
        }
        tiers_->report(j, good, now_ns);
    }
}

bool TierExec::plan_promote(const BlockHash& key, TierJob* out) {
    if (!enabled_ || !leader_) return false;
    return tiers_->plan_promote(key, out);
}

}  /* namespace rad */
