/* prefix.cpp -- see prefix.h. */
#include "mem/prefix.h"

#include "mem/kvtier.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <unordered_set>

namespace rad {

/* ------------------------------------------------------------------ hashing */
/* A 128-bit chained mix, not a cryptographic hash: every entry stores its block's token ids and
 * compares them on a hit, so a collision is a miss, and the hash only needs to keep buckets short.
 *
 * splitmix64's finaliser is a 64-bit bijection, so folding one token cannot lose the bits of the
 * previous one (the weakness of a plain multiply-xor chain). */
static inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

std::string BlockHash::str() const { return fmt("%016llx%016llx", (unsigned long long)hi,
                                                (unsigned long long)lo); }

BlockHash chain_block_hash(const BlockHash& parent, const int32_t* toks, int32_t n,
                           const MMContentKey* mm, int32_t n_mm, int64_t block_begin) {
    /* The parent seeds both lanes, so identical tokens under a different prefix hash differently. */
    BlockHash h;
    h.hi = mix64(parent.hi ^ 0x243f6a8885a308d3ull);
    h.lo = mix64(parent.lo ^ 0x13198a2e03707344ull);
    h.hi = mix64(h.hi ^ (uint64_t)block_begin);

    for (int32_t i = 0; i < n; ++i) {
        uint64_t t = (uint64_t)(uint32_t)toks[i];
        h.lo = mix64(h.lo ^ t);
        h.hi = mix64(h.hi + h.lo + t);
    }

    /* Fold in overlapping multimodal spans by content hash plus offsets, so different images
     * behind identical placeholder ids never alias (spec §11). */
    const int64_t lo_pos = block_begin, hi_pos = block_begin + n;
    for (int32_t i = 0; i < n_mm; ++i) {
        const MMContentKey& m = mm[i];
        if (m.tok_end <= lo_pos || m.tok_begin >= hi_pos) continue;
        h.lo = mix64(h.lo ^ m.hash);
        h.hi = mix64(h.hi ^ mix64(m.hash + (uint64_t)(m.tok_begin - lo_pos) * 1000003ull
                                  + (uint64_t)(m.tok_end - lo_pos)));
    }
    /* Never let a block hash to zero: a zeroed BlockHash is the chain root and must be unique. */
    if (!h.valid()) h.lo = 1;
    return h;
}

/* ------------------------------------------------------------------ retention policies */

void GeometricBackoff::select(const std::vector<int64_t>& pos, std::vector<int64_t>* keep) const {
    keep->clear();
    if (pos.empty()) return;
    const int64_t tip = pos.back();
    std::vector<char> mark(pos.size(), 0);
    mark.back() = 1;                          /* the tip is always kept */
    for (int64_t s = interval_;; s *= 2) {
        const int64_t b = tip / s;
        for (int64_t j = b > 0 ? b - 1 : 0; j <= b; ++j) {
            const auto it = std::lower_bound(pos.begin(), pos.end(), j * s);
            if (it != pos.end() && *it < (j + 1) * s) mark[(size_t)(it - pos.begin())] = 1;
        }
        /* At this scale every position is in bucket 0, and so at every larger one. */
        if (b == 0 || s > INT64_MAX / 4) break;
    }
    for (size_t i = 0; i < pos.size(); ++i) if (mark[i]) keep->push_back(pos[i]);
}

void KeepTipOnly::select(const std::vector<int64_t>& pos, std::vector<int64_t>* keep) const {
    keep->clear();
    if (pos.empty()) return;
    const int64_t tip = pos.back();
    if (floor_ > 0) {
        const auto it = std::upper_bound(pos.begin(), pos.end(), tip - floor_);
        if (it != pos.begin()) keep->push_back(*(it - 1));
    }
    keep->push_back(tip);
}

void KeepAll::select(const std::vector<int64_t>& pos, std::vector<int64_t>* keep) const {
    *keep = pos;
}

/* A session's tip is usually a prompt's tail between two interval checkpoints, so each step also
 * counts the ladder with a tail half an interval on as its tip. */
int64_t CheckpointPolicy::held(int64_t ctx, int64_t interval) const {
    if (interval <= 0 || ctx < interval) return 1;
    std::vector<int64_t> kept, keep;
    size_t most = 0;
    for (int64_t p = interval; p <= ctx; p += interval) {
        kept.push_back(p);
        select(kept, &keep);
        std::sort(keep.begin(), keep.end());
        kept.swap(keep);
        most = std::max(most, kept.size());
        if (p + interval / 2 <= ctx) {
            std::vector<int64_t> tipped = kept;
            tipped.push_back(p + interval / 2);
            select(tipped, &keep);
            most = std::max(most, keep.size());
        }
    }
    return (int64_t)most;
}

int64_t KeepAll::held(int64_t ctx, int64_t interval) const {
    return interval > 0 && ctx >= interval ? ctx / interval : 1;
}

std::unique_ptr<CheckpointPolicy> make_checkpoint_policy(const std::string& name,
                                                         int64_t interval) {
    if (name == "tip-only") return std::make_unique<KeepTipOnly>();
    if (name == "keep-all") return std::make_unique<KeepAll>();
    return std::make_unique<GeometricBackoff>(interval);
}

CheckpointSlots checkpoint_slots_auto(const CheckpointPolicy& policy, int64_t max_seqs,
                                      int64_t ctx, int64_t interval, int64_t slot_bytes,
                                      int64_t share_bytes, int64_t room_bytes) {
    CheckpointSlots c;
    c.per_seq = policy.held(ctx, interval) + 1;
    c.want = checkpoint_slots_want(policy, max_seqs, ctx, interval);
    if (slot_bytes <= 0) return c;
    const int64_t share = std::max<int64_t>(share_bytes, 0) / slot_bytes;
    const int64_t room = std::max<int64_t>(room_bytes, 0) / slot_bytes;
    c.slots = std::min(c.want, std::max(share, std::min(c.want, kCheckpointSlotsFloor)));
    if (c.slots < c.want) c.bound = CheckpointSlots::Bound::Share;
    if (room < c.slots) {
        c.slots = room;
        c.bound = CheckpointSlots::Bound::Room;
    }
    return c;
}

int64_t checkpoint_slots_want(const CheckpointPolicy& policy, int64_t max_seqs, int64_t ctx,
                              int64_t interval) {
    const int64_t per = policy.held(ctx, interval) + 1;
    const int64_t seqs = std::max<int64_t>(max_seqs, 1);
    return seqs > INT64_MAX / per ? INT64_MAX : seqs * per;
}

int64_t host_reserve_for_floor(int64_t need_bytes, int64_t room_bytes, int64_t margin_bytes) {
    if (need_bytes <= 0 || margin_bytes <= 0 || room_bytes >= need_bytes) return margin_bytes;
    /* Whole MiB, the unit --host-headroom-mib takes, and the shortfall rounding up: the reserve
     * gives back at least what the floor asked for, so the store is never a byte short of it. */
    const int64_t mib = (int64_t)1 << 20;
    const int64_t shortfall = (need_bytes - room_bytes + mib - 1) / mib;
    const int64_t margin = margin_bytes / mib;
    const int64_t keep = margin - shortfall;
    return std::min(margin, std::max(kHostReserveFloor / mib, keep)) * mib;
}

/* ------------------------------------------------------------------ configure */

int PrefixCache::configure(const Config& cfg, KVManager* kv, int64_t checkpoint_interval) {
    kv_ = kv;
    enabled_ = false;
    has_linear_ = false;
    block_size_ = 0;
    interval_ = 0;
    groups_.clear();
    per_.clear();
    entry_groups_.clear();
    map_.clear(); lru_.clear();
    ck_by_prefix_.clear(); ck_by_session_.clear();
    /* set_policy() may override this afterwards. */
    policy_ = make_checkpoint_policy(cfg.checkpoint_policy, checkpoint_interval);
    policy_->set_floor(drafter_floor_);
    report_.clear();

    if (!kv) return RAD_E_INVAL;
    if (!cfg.prefix_cache) {
        report_ = "prefix cache: disabled by --no-prefix-cache\n";
        return RAD_OK;
    }

    /* Every paged group but a windowed one is cached. An entry covers the lcm of their block
     * sizes and holds lcm / B blocks of a group with B-token blocks, so a hit ends on a block of
     * every group, which adopt() needs. The chunk quantum is a multiple of every paged group's
     * block (core/sched/geometry.cpp), so the lcm divides every checkpoint position. */
    for (const auto& p : kv->plans()) {
        if (p.stateful()) has_linear_ = true;
        if (p.paged() && p.kind != RAD_KV_WINDOW && p.block_size > 0)
            block_size_ = block_size_ > 0 ? std::lcm(block_size_, p.block_size) : p.block_size;
    }
    /* With only windowed groups, nothing is indexed and a hit is a window snapshot alone, keyed
     * on the chain at the first group's block. */
    if (block_size_ == 0) {
        for (const auto& p : kv->plans()) if (p.paged()) { block_size_ = p.block_size; break; }
    }
    if (block_size_ <= 0) {
        report_ = "prefix cache: no paged KV group to cache; disabled\n";
        RAD_WARN("%s", report_.c_str());
        return RAD_OK;
    }
    std::string skipped, split;
    for (const auto& p : kv->plans()) {
        if (!p.paged()) continue;
        /* A window group is not prefix-cached. Its blocks below the window are recycled, adopt()
         * would overfill a table capped at window_blocks, and a hit does not set first_block_pos.
         * A checkpoint slot can carry a trunk's window instead (KVManager::has_window_snapshot);
         * a drafter or a windowed layer without one starts its cache where the hit ended. */
        if (p.kind == RAD_KV_WINDOW) {
            skipped += fmt("  group '%s' is windowed (%lld blocks): not cached%s\n",
                           p.name.c_str(), (long long)p.window_blocks,
                           p.win_snap ? "; a checkpoint slot carries its window" : "");
            continue;
        }
        groups_.push_back(p.index);
        per_.push_back((int32_t)(block_size_ / p.block_size));
        for (int64_t k = 0; k < block_size_ / p.block_size; ++k) entry_groups_.push_back(p.index);
        if (p.block_size != block_size_)
            split += fmt("  group '%s' block %lld: %lld blocks an entry\n", p.name.c_str(),
                         (long long)p.block_size, (long long)(block_size_ / p.block_size));
    }

    /* An entry can be demoted only when the host tier can hold every group in it. A streamed
     * group's rows are its pinned mirror, addressed by the block id the sequence used, so an entry
     * holding one stays in VRAM and is evicted whole (docs/KV-STREAMING.md §4.1). */
    demotable_ = true;
    for (int32_t g : groups_) {
        const KVGroupPlan* p = kv->plan(g);
        if (p && p->streamed()) demotable_ = false;
    }

    interval_ = checkpoint_interval;
    /* A window snapshot is state too: a hit is reusable only at a checkpoint, whose slot carries
     * the window behind it (KVManager::has_window_snapshot). */
    if (kv->has_window_snapshot() && interval_ > 0) has_linear_ = true;
    if (has_linear_) {
        if (interval_ <= 0) {
            RAD_ERR("checkpoint_interval is %lld but the model has linear-attention state, which "
                    "needs a checkpoint for a hit to be usable", (long long)interval_);
            return RAD_E_INVAL;
        }
        if (interval_ % block_size_ != 0) {
            /* A checkpoint is found by the chained hash at its position, which exists only on
             * block boundaries, so an interval off the block grid would never hit. */
            RAD_ERR("the checkpoint interval resolves to %lld (from --checkpoint-interval %lld, "
                    "rounded up to the chunk quantum), which is not a multiple of the cache's "
                    "%lld-token entry, so no checkpoint would ever hit (spec §7.3)",
                    (long long)interval_, (long long)cfg.checkpoint_interval,
                    (long long)block_size_);
            return RAD_E_INVAL;
        }
    }

    enabled_ = true;
    report_ = fmt("prefix cache: %zu paged group(s) in %lld-token entries; checkpoints "
                  "every %lld tokens (%lld slots of %s a rank, pinned host memory), retention "
                  "'%s'\n",
                  groups_.size(), (long long)block_size_,
                  (long long)(has_linear_ ? interval_ : 0),
                  (long long)kv->checkpoint_slots(), humanb(kv->checkpoint_bytes()).c_str(),
                  policy_->name());
    report_ += split;
    report_ += skipped;
    if (!demotable_) {
        /* Stated whether or not a host tier is configured. */
        report_ += "  a streamed group is cached, so no entry can be demoted to the host tier: a "
                   "streamed group's rows are its pinned mirror, addressed by block id\n";
    }
    return RAD_OK;
}

void PrefixCache::set_policy(std::unique_ptr<CheckpointPolicy> p) {
    if (p) policy_ = std::move(p);
    if (policy_) policy_->set_floor(drafter_floor_);
}

void PrefixCache::set_drafter_floor(int64_t tokens) {
    drafter_floor_ = tokens > 0 ? tokens : 0;
    if (policy_) policy_->set_floor(drafter_floor_);
}

std::string PrefixCache::report() const { return report_; }

/* ------------------------------------------------------------------ full attention */

std::vector<BlockHash> PrefixCache::chain_all(const std::vector<int32_t>& tokens,
                                              const std::vector<MMContentKey>& mm,
                                              int64_t n_blocks) const {
    std::vector<BlockHash> hs;
    hs.reserve((size_t)n_blocks);
    BlockHash parent;
    for (int64_t b = 0; b < n_blocks; ++b) {
        parent = chain_block_hash(parent, tokens.data() + b * block_size_, (int32_t)block_size_,
                                  mm.data(), (int32_t)mm.size(), b * block_size_);
        hs.push_back(parent);
    }
    return hs;
}

/* A snapshot is on the chain when its key equals the chain hash at its position. */
std::unordered_map<BlockHash, PrefixCache::Ckpt, BlockHashHash>::iterator
PrefixCache::deepest_ckpt(const std::vector<BlockHash>& hs, int64_t limit) const {
    auto best = ck_by_prefix_.end();
    for (auto it = ck_by_prefix_.begin(); it != ck_by_prefix_.end(); ++it) {
        const Ckpt& c = it->second;
        if (!c.valid || c.slot < 0 || c.pos <= 0 || c.pos > limit || c.pos % block_size_ != 0 ||
            c.pos / block_size_ > (int64_t)hs.size())
            continue;
        if (best != ck_by_prefix_.end() && c.pos <= best->second.pos) continue;
        if (hs[(size_t)(c.pos / block_size_ - 1)] != it->first) continue;
        best = it;
    }
    return best;
}

void PrefixCache::touch(const BlockHash& h, uint64_t now_ns) const {
    auto it = map_.find(h);
    if (it == map_.end() || !it->second.in_lru) return;
    lru_.splice(lru_.end(), lru_, it->second.lru);
    it->second.used_ns = now_ns;
}

CacheHit PrefixCache::lookup(const std::vector<int32_t>& tokens,
                             const std::vector<MMContentKey>& mm,
                             std::vector<std::vector<int32_t>>* blocks_out,
                             const HitCap* cap, uint64_t session) const {
    CacheHit hit;
    if (blocks_out) { blocks_out->assign(groups_.size(), {}); }
    if (!enabled_ || tokens.empty()) return hit;
    ++stats_.lookups;

    const int64_t n_full = (int64_t)tokens.size() / block_size_;
    std::vector<BlockHash> hs = chain_all(tokens, mm, n_full);

    int64_t matched = 0;
    bool departed = false;              /* the prompt goes on where nothing is cached */
    for (; matched < n_full; ++matched) {
        auto it = map_.find(hs[(size_t)matched]);
        if (it == map_.end()) { departed = true; break; }
        /* An off-device entry is not a hit: demotion released its blocks and its recorded ids now
         * belong to other sequences. The prefix stops here (also while a promotion is in flight);
         * the caller may promote and look up again (restorable_from). */
        if (!it->second.in_lru) break;
        if (!same_tokens(it->second, tokens.data() + matched * block_size_)) break;
    }

    if (departed && matched > 0 && map_.find(hs[(size_t)matched - 1])->second.n_children >= 2)
        hit.branch_at = (int32_t)(matched * block_size_);

    /* Back a full hit off by one entry so one token remains to compute a logit from. */
    if (matched * block_size_ >= (int64_t)tokens.size() && matched > 0) --matched;

    auto deepest = [&](int64_t limit) { return deepest_ckpt(hs, limit); };
    /* Tokens a hit of `m` blocks lets a sequence skip: all of them, or with recurrent state only
     * up to the deepest checkpoint (Scheduler::admit_one resumes there). */
    auto reusable = [&](int64_t m) -> int64_t {
        if (!has_linear_) return m * block_size_;
        const auto b = deepest(m * block_size_);
        return b == ck_by_prefix_.end() ? 0 : b->second.pos;
    };
    if (cap && matched > cap->tokens / block_size_) {
        const int64_t whole = reusable(matched);
        if (whole > cap->tokens) {
            const int64_t m = cap->tokens > 0 ? cap->tokens / block_size_ : 0;
            const int64_t lost = whole - reusable(m);
            if (lost <= cap->max_give_back) {
                matched = m;
                hit.given_back = (int32_t)lost;
            }
        }
    }

    const uint64_t now = rad_mono_ns();
    /* Deepest first, so the chain's first block is the most recent: under pressure a conversation
     * loses its end before its start, and what is left still hits. */
    for (int64_t b = matched - 1; b >= 0; --b) touch(hs[(size_t)b], now);
    if (blocks_out)
        for (int64_t b = 0; b < matched; ++b) {
            const Entry& e = map_.find(hs[(size_t)b])->second;
            for (size_t k = 0, g = 0; g < groups_.size(); ++g)
                for (int32_t j = 0; j < per_[g]; ++j, ++k)
                    (*blocks_out)[g].push_back(e.blocks[k]);
        }
    /* A hit is a use for the tiers' idle clock too; otherwise a prefix hit every turn, or one just
     * promoted for this lookup, would look idle since its publish. */
    if (tiers_ && matched > 0) tiers_->note_hits(hs.data(), matched, now);
    hit.n_attn_tokens = (int32_t)(matched * block_size_);
    stats_.block_hits += matched;
    stats_.block_misses += n_full - matched;

    /* ---- the split: how far the linear state can be reused ----
     * Recurrent state is reusable only at the deepest snapshot at or below n_attn_tokens; the
     * scheduler resumes there (Scheduler::admit_one) and recomputes the tokens in between. */
    if (!has_linear_) {
        hit.n_linear_tokens = hit.n_attn_tokens;
        return hit;
    }
    const auto best = deepest(hit.n_attn_tokens);
    if (best == ck_by_prefix_.end()) {
        ++stats_.ckpt_misses;
        return hit;
    }
    hit.n_linear_tokens = (int32_t)best->second.pos;
    hit.checkpoint_slot = best->second.slot;
    best->second.used = best->second.resumed = ++clock_;
    best->second.used_ns = now;
    handed_.push_back(best->first);
    ++stats_.ckpt_hits;
    if (session != 0) take_over(session, hs, hit.n_attn_tokens);
    return hit;
}

bool PrefixCache::retain_entry(const std::vector<std::vector<int32_t>>& bt, int64_t b,
                               std::vector<int32_t>* out) {
    out->assign(entry_groups_.size(), -1);
    size_t k = 0;
    for (size_t g = 0; g < groups_.size(); ++g)
        for (int32_t j = 0; j < per_[g]; ++j, ++k) {
            const int32_t blk = bt[g][(size_t)(b * per_[g] + j)];
            /* An entry, not a block: a cold id (KVManager window residency) is retained on its
             * store slot, so the cache indexes it and a hit adopts it cold. */
            if (kv_->retain_entry(groups_[g], blk) >= 0) { (*out)[k] = blk; continue; }
            /* Unwind exactly what was retained: anything else leaks a block or frees one under a
             * live sequence. */
            for (size_t i = 0; i < k; ++i) kv_->release_entry(entry_groups_[i], (*out)[i]);
            out->clear();
            return false;
        }
    return true;
}

int PrefixCache::insert(uint64_t session,
                        const std::vector<int32_t>& tokens,
                        const std::vector<MMContentKey>& mm,
                        const std::vector<std::vector<int32_t>>& blocks,
                        int64_t n_tokens, const std::string& label) {
    if (!enabled_) return RAD_OK;
    if (blocks.size() != groups_.size()) return RAD_E_INVAL;
    if (n_tokens > (int64_t)tokens.size()) return RAD_E_INVAL;

    int64_t n_full = n_tokens / block_size_;
    if (n_full <= 0) return RAD_OK;
    for (size_t g = 0; g < blocks.size(); ++g)
        if ((int64_t)blocks[g].size() < n_full * per_[g]) return RAD_E_INVAL;

    /* A cold entry (< 0: KVManager window residency) is indexed like a pool id: the cache holds a
     * reference on its store slot (retain_entry), and a hit adopts it cold. Only the idle tiers
     * cannot take one (see cold_entry). */
    std::vector<BlockHash> hs = chain_all(tokens, mm, n_full);
    const uint64_t now = rad_mono_ns();
    for (int64_t b = 0; b < n_full; ++b) {
        const BlockHash& h = hs[(size_t)b];
        auto it = map_.find(h);
        /* With snapshots (recurrent state or a sliding window) an existing entry keeps the bytes
         * it was published with, wherever it is held: a snapshot deeper in the chain was computed
         * from them, and a recompute in another step shape can round the KV differently. The
         * recomputed blocks go with the request's sequence; a later hit restores the tier copy. */
        if (it != map_.end() && (it->second.in_lru || has_linear_)) continue;   /* touched below */
        if (it != map_.end()) {
            /* Without snapshots, an off-device entry this request recomputed (a restore that did
             * not fit, or a chain cut short): its blocks become the entry's device copy and the
             * tier copy goes (IdleTiers::rebound). An entry with a promotion in flight is left
             * alone. */
            if (!tiers_ || !tiers_->resident_off_device(h)) continue;
            /* The tiers gather pool blocks; a cold id cannot be one. The entry stays off-device
             * and the next turn recomputes it. */
            bool cold = false;
            for (const auto& bt : blocks) for (int32_t j = 0; j < per_[(size_t)(&bt - &blocks[0])]; ++j)
                cold = cold || bt[(size_t)(b * per_[(size_t)(&bt - &blocks[0])] + j)] < 0;
            if (cold) continue;
            std::vector<int32_t> mine;
            if (!retain_entry(blocks, b, &mine)) return RAD_E_STATE;
            Entry& e = it->second;
            e.blocks = std::move(mine);
            e.used_ns = now;
            lru_.push_back(h);
            e.lru = std::prev(lru_.end());
            e.in_lru = true;
            tiers_->rebound(h, e.blocks, now, b);
            continue;
        }

        Entry e;
        if (!retain_entry(blocks, b, &e.blocks)) return RAD_E_STATE;
        e.tokens.assign(tokens.begin() + b * block_size_,
                        tokens.begin() + (b + 1) * block_size_);
        e.used_ns = now;
        lru_.push_back(h);
        e.lru = std::prev(lru_.end());
        e.in_lru = true;
        map_.emplace(h, std::move(e));
        if (b > 0) {
            auto p = map_.find(hs[(size_t)b - 1]);
            if (p != map_.end() && p->second.n_children < 255) ++p->second.n_children;
        }
    }
    /* Deepest first, as in lookup: the first block ends most recent. */
    for (int64_t b = n_full - 1; b >= 0; --b) touch(hs[(size_t)b], now);

    /* The tiers learn the chain only here: insert() is the one call that knows a chain was
     * completed, and the only place every entry's block ids are in hand.
     *
     * The ids come from the map, not from `blocks`. For an entry that already existed the loop
     * above kept the cache's blocks, and `blocks` belongs to a sequence about to be freed; the
     * tiers would copy from blocks reassigned to another sequence. */
    if (tiers_ && demotable_) {
        for (int64_t b = 0; b < n_full; ++b) {
            auto it = map_.find(hs[(size_t)b]);
            if (it == map_.end() || cold_entry(it->second)) continue;
            tiers_->note_used(hs[(size_t)b], it->second.blocks, now, b);
        }
    }
    /* The session index names no blocks, so it is kept with or without a host copy. */
    if (tiers_) tiers_->note_session(session, hs, n_tokens, now, label);

    take_over(session, hs, n_full * block_size_);
    return RAD_OK;
}

void PrefixCache::take_over(uint64_t session, const std::vector<BlockHash>& hs,
                            int64_t upto) const {
    for (auto& kvp : ck_by_prefix_) {
        Ckpt& c = kvp.second;
        if (c.session == session || c.pos <= 0 || c.pos > upto || c.pos % block_size_ != 0 ||
            hs[(size_t)(c.pos / block_size_ - 1)] != kvp.first)
            continue;
        auto old = ck_by_session_.find(c.session);
        if (old != ck_by_session_.end()) {
            old->second.erase(std::remove(old->second.begin(), old->second.end(), kvp.first),
                              old->second.end());
            if (old->second.empty()) ck_by_session_.erase(old);
        }
        c.session = session;
        ck_by_session_[session].push_back(kvp.first);
    }
}


/* ------------------------------------------------------------------ idle session tiers */

std::vector<BlockHash> PrefixCache::restorable_from(const std::vector<int32_t>& tokens,
                                                    const std::vector<MMContentKey>& mm,
                                                    int64_t fit, bool* cut) const {
    std::vector<BlockHash> out;
    if (cut) *cut = false;
    if (!enabled_ || !tiers_ || tokens.empty()) return out;

    /* Only blocks a hit can use: one token must remain, so a prompt ending on a block boundary
     * never reuses its last block (lookup backs off by one). */
    const int64_t n_full = ((int64_t)tokens.size() - 1) / block_size_;
    std::vector<BlockHash> hs = chain_all(tokens, mm, n_full);

    /* Walk as lookup does, past where it stopped, and collect every off-device block on the
     * chain, not just the first run. Stopping at the first gap would make the hit length depend on
     * how far the maintenance pass had got; the scheduler chunks the rest of the prefill against
     * that length, and greedy output is tie-level sensitive to the GEMM shapes (§19). Restoring
     * the whole chain makes the hit length a function of the prompt. */
    std::vector<int64_t> at;     /* each entry's block on the chain */
    for (int64_t b = 0; b < n_full; ++b) {
        auto it = map_.find(hs[(size_t)b]);
        if (it == map_.end()) break;
        if (!same_tokens(it->second, tokens.data() + b * block_size_)) break;
        if (!tiers_->resident_off_device(hs[(size_t)b])) continue;
        out.push_back(hs[(size_t)b]);
        at.push_back(b);
    }
    if (out.empty() || (int64_t)out.size() <= fit) return out;

    /* Restored up to `fit`, the hit ends at the first entry left behind. */
    if (cut) *cut = true;
    int64_t keep = std::max<int64_t>(0, fit);
    if (has_linear_) {
        const auto best = deepest_ckpt(hs, at[(size_t)keep] * block_size_);
        const int64_t upto = best == ck_by_prefix_.end() ? 0 : best->second.pos / block_size_;
        keep = std::lower_bound(at.begin(), at.begin() + keep, upto) - at.begin();
    }
    RAD_DEBUG("prefix cache: restore of %zu entries cut to %lld: the pool holds %lld of them, and "
              "the hit ends at token %lld", out.size(), (long long)keep,
              (long long)std::max<int64_t>(0, fit), (long long)(at[(size_t)keep] * block_size_));
    out.resize((size_t)keep);
    return out;
}

int64_t PrefixCache::restore_room() const {
    if (!kv_ || groups_.empty()) return 0;
    int64_t room = INT64_MAX;
    for (size_t i = 0; i < groups_.size(); ++i)
        room = std::min(room, (kv_->total_blocks(groups_[i]) - kv_->held_blocks(groups_[i])) /
                                  std::max<int32_t>(per_[i], 1));
    return std::max<int64_t>(0, room);
}

int64_t PrefixCache::trim_for(const std::vector<int32_t>& tokens,
                              const std::vector<MMContentKey>& mm, int64_t n_entries) {
    /* Without a host tier a trim erases, and making room for a restore must lose nothing. */
    if (!enabled_ || n_entries <= 0 || !tiers_ || !tiers_->host_on()) return 0;
    const std::vector<BlockHash> hs = chain_all(tokens, mm, (int64_t)tokens.size() / block_size_);
    std::unordered_set<BlockHash, BlockHashHash> spare(hs.begin(), hs.end());
    for (const auto& h : held_) spare.insert(h.first);
    return trim_sparing(n_entries, &spare);
}

void PrefixCache::drop_offdevice(const BlockHash& key) {
    auto it = map_.find(key);
    if (it == map_.end()) return;
    /* No release_block: the ids name blocks the pool has reassigned (see the declaration). */
    if (it->second.in_lru) lru_.erase(it->second.lru);
    map_.erase(it);
    ++stats_.evictions;
    /* A snapshot under the same key stays: no lookup reaches it past the missing block, and
     * retention or expiry retires it via retire_ckpt. It cannot be retired here because the tiers
     * call this holding their lock. */
}

int PrefixCache::rebind(const BlockHash& key, const std::vector<int32_t>& blocks) {
    auto it = map_.find(key);
    if (it == map_.end()) return RAD_E_NOTFOUND;
    if (blocks.size() != entry_groups_.size()) return RAD_E_SHAPE;
    for (int32_t b : blocks) if (b < 0) return RAD_E_INVAL;
    /* The reservation's reference becomes the cache's: KVManager::reserve_block left each block at
     * refcount one with no sequence attached, and the cache's release path frees it from now on.
     * No extra retain. */
    Entry& e = it->second;
    e.blocks = blocks;
    for (size_t k = 0; k < entry_groups_.size(); ++k)
        kv_->cache_reserved_block(entry_groups_[k], blocks[k]);
    if (!e.in_lru) {
        /* A restore is a use: a request is about to hit it. */
        e.used_ns = rad_mono_ns();
        lru_.push_back(key);
        e.lru = std::prev(lru_.end());
        e.in_lru = true;
    }
    return RAD_OK;
}

bool PrefixCache::shared_with_live(const Entry& e) const {
    for (size_t k = 0; k < entry_groups_.size() && k < e.blocks.size(); ++k)
        if (kv_->entry_refs(entry_groups_[k], e.blocks[k]) > 1) return true;
    return false;
}

bool PrefixCache::cold_entry(const Entry& e) const {
    for (int32_t b : e.blocks) if (b < 0) return true;
    return false;
}

/* Token ids are compared too, so a hash collision is a miss rather than another conversation's
 * context. */
bool PrefixCache::same_tokens(const Entry& e, const int32_t* src) const {
    return (int64_t)e.tokens.size() == block_size_ &&
           std::memcmp(e.tokens.data(), src, (size_t)block_size_ * sizeof(int32_t)) == 0;
}

void PrefixCache::release_blocks(const Entry& e) {
    for (size_t k = 0; k < entry_groups_.size() && k < e.blocks.size(); ++k)
        kv_->release_entry(entry_groups_[k], e.blocks[k]);
}

int64_t PrefixCache::evict_lru(int64_t n) {
    int64_t done = 0;
    /* One pass over the list as it was on entry. An entry a live sequence also holds is in use:
     * it moves to the back, stamped, so the list stays in clock order and the next call does not
     * walk the same shared run. */
    const uint64_t now = rad_mono_ns();
    for (int64_t seen = (int64_t)lru_.size(); done < n && seen > 0 && !lru_.empty(); --seen) {
        const BlockHash h = lru_.front();
        auto it = map_.find(h);
        if (it == map_.end()) { lru_.pop_front(); continue; }
        Entry& e = it->second;
        if (shared_with_live(e)) {
            lru_.splice(lru_.end(), lru_, lru_.begin());
            e.used_ns = now;
            continue;
        }
        lru_.pop_front();
        e.in_lru = false;
        release_blocks(e);
        if (tiers_) {
            /* Host copy landed: the entry leaves VRAM and stays in the index. */
            if (tiers_->droppable(h)) {
                tiers_->drop_commit(h);
                ++stats_.drops;
                ++done;
                continue;
            }
            tiers_->forget(h);
        }
        map_.erase(it);
        ++done;
        ++stats_.evictions;
    }
    return done;
}

int64_t PrefixCache::trim(int64_t n_entries) { return trim_sparing(n_entries, nullptr); }

int64_t PrefixCache::trim_sparing(int64_t n_entries,
                                  const std::unordered_set<BlockHash, BlockHashHash>* spare) {
    /* Tiers attached but off count as no tiers: no copy will ever land, so do not wait for one. */
    const bool keep = tiers_ && tiers_->host_on();
    const uint64_t now = rad_mono_ns();
    int64_t done = 0;
    auto it = lru_.begin();
    for (int64_t seen = (int64_t)lru_.size(); it != lru_.end() && done < n_entries && seen > 0;
         --seen) {
        const BlockHash h = *it;
        auto m = map_.find(h);
        if (m == map_.end()) { it = lru_.erase(it); continue; }
        Entry& e = m->second;
        if (spare && spare->count(h)) { ++it; continue; }
        if (shared_with_live(e)) {
            /* In use: to the back, as in evict_lru. */
            auto next = std::next(it);
            lru_.splice(lru_.end(), lru_, it);
            e.used_ns = now;
            it = next;
            continue;
        }
        /* Nothing in the pool to give back: an entry of cold ids frees store slots, not blocks. */
        bool pooled = false;
        for (int32_t b : e.blocks) pooled = pooled || b >= 0;
        if (!pooled) { ++it; continue; }
        /* Skip an entry whose copy is still in flight; one the tier cannot hold never gets a copy. */
        if (keep && demotable_ && !tiers_->droppable(h)) { ++it; continue; }
        it = lru_.erase(it);
        e.in_lru = false;
        release_blocks(e);
        if (keep && demotable_) {
            tiers_->drop_commit(h);
            ++stats_.drops;
        } else {
            if (tiers_) tiers_->forget(h);
            map_.erase(m);
            ++stats_.evictions;
        }
        ++done;
    }
    return done;
}

int64_t PrefixCache::drop_clean() {
    if (!tiers_ || !kv_) return 0;
    drop_scratch_.clear();
    tiers_->take_clean(&drop_scratch_);
    const uint64_t frees = kv_->seq_frees();
    if (frees != frees_seen_ || released_) {
        frees_seen_ = frees;
        released_ = false;
        for (const BlockHash& h : drop_wait_) {
            auto it = map_.find(h);
            if (it != map_.end()) it->second.drop_wait = false;
        }
        drop_scratch_.insert(drop_scratch_.end(), drop_wait_.begin(), drop_wait_.end());
        drop_wait_.clear();
    }
    int64_t n = 0;
    for (const BlockHash& h : drop_scratch_) {
        auto it = map_.find(h);
        if (it == map_.end()) continue;
        Entry& e = it->second;
        if (!e.in_lru || !tiers_->droppable(h)) continue;
        /* Skip held entries and blocks a running sequence also holds: dropping would free nothing,
         * and the next lookup would restore a second copy beside the one still in VRAM. */
        if (held_.count(h) > 0 || shared_with_live(e)) {
            if (!e.drop_wait) { e.drop_wait = true; drop_wait_.push_back(h); }
            continue;
        }
        release_blocks(e);
        lru_.erase(e.lru);
        e.in_lru = false;
        ++stats_.drops;
        tiers_->drop_commit(h);
        ++n;
    }
    return n;
}

/* The KV manager tracks how many cache-held blocks are at refcount one. One group only: the paged
 * groups share a token axis and an entry is blocks_per_entry() blocks in each, so summing groups
 * would multiply the answer. Off-device entries hold no blocks and count for nothing. */
int64_t PrefixCache::unshared_tokens() const {
    if (!kv_ || groups_.empty() || block_size_ <= 0 || per_.empty() || per_[0] <= 0) return 0;
    return kv_->cache_only_blocks(groups_[0]) * (block_size_ / per_[0]);
}

void PrefixCache::clear() {
    for (auto& kv : map_) {
        if (kv.second.in_lru) release_blocks(kv.second);
        if (tiers_) tiers_->forget(kv.first);
        ++stats_.evictions;
    }
    map_.clear();
    lru_.clear();
    drop_wait_.clear();
}

std::vector<BlockHash> PrefixCache::lru_order() const {
    return std::vector<BlockHash>(lru_.begin(), lru_.end());
}

/* ------------------------------------------------------------------ checkpoints */

int64_t PrefixCache::next_checkpoint_after(int64_t pos) const {
    if (interval_ <= 0) return -1;
    if (pos < 0) pos = 0;
    return (pos / interval_ + 1) * interval_;
}

std::vector<int64_t> PrefixCache::checkpoints(uint64_t session) const {
    std::vector<int64_t> out;
    auto it = ck_by_session_.find(session);
    if (it == ck_by_session_.end()) return out;
    for (const BlockHash& h : it->second) {
        auto e = ck_by_prefix_.find(h);
        if (e != ck_by_prefix_.end()) out.push_back(e->second.pos);
    }
    std::sort(out.begin(), out.end());
    return out;
}

int64_t PrefixCache::retain(uint64_t session) {
    auto it = ck_by_session_.find(session);
    if (it == ck_by_session_.end()) return 0;

    std::vector<std::pair<int64_t, BlockHash>> all;
    std::vector<BlockHash> kept;
    for (const BlockHash& h : it->second) {
        auto e = ck_by_prefix_.find(h);
        if (e == ck_by_prefix_.end()) continue;
        all.push_back({e->second.pos, h});
    }
    std::sort(all.begin(), all.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<int64_t> pos;
    for (const auto& a : all) pos.push_back(a.first);
    std::vector<int64_t> keep;
    policy_->select(pos, &keep);
    /* A policy may return its keep set in any order; the lookup below needs it sorted. */
    std::sort(keep.begin(), keep.end());
    const uint64_t tip_made = all.empty() ? 0 : ck_by_prefix_.find(all.back().second)->second.made;

    int64_t freed = 0;
    for (const auto& a : all) {
        if (std::binary_search(keep.begin(), keep.end(), a.first)) { kept.push_back(a.second); continue; }
        /* Keep one handed out this step or reserved and not yet committed regardless of policy:
         * its admission restores from the slot ahead of the step, or the step's save lands in it. */
        if (handed(a.second) || reserved(a.second)) { kept.push_back(a.second); continue; }
        auto e = ck_by_prefix_.find(a.second);
        if (e == ck_by_prefix_.end()) continue;
        /* And one a lookup resumed from since the session's tip was written. The policy places
         * its ladder from the tip, but a later turn that resumed below the tip has left this
         * chain there (a rewritten last message), and its next turn most likely resumes there
         * again. */
        if (e->second.resumed > tip_made) { kept.push_back(a.second); continue; }
        RAD_DEBUG("prefix cache: snapshot at %lld of session %llu released by retention '%s' "
                  "(unused for %.1f s)", (long long)e->second.pos,
                  (unsigned long long)e->second.session, policy_->name(),
                  idle_s(e->second.used_ns));
        retire_ckpt(e);
        ++freed;
        ++stats_.ckpt_evictions;
        ++stats_.ckpt_retired;
    }
    /* Erase an emptied session. Sessions are request ids, so otherwise the map grows by one key
     * per request that ever checkpointed, and reserve_checkpoint walks every key on each
     * allocation failure. */
    if (kept.empty()) { ck_by_session_.erase(it); return freed; }
    it->second = std::move(kept);
    return freed;
}

int PrefixCache::reserve_checkpoint(uint64_t session, int64_t pos, const BlockHash& prefix,
                                    int32_t* slot_out) {
    if (!enabled_ || !has_linear_) return RAD_E_UNSUPPORTED;
    if (!slot_out) return RAD_E_INVAL;
    /* The key is the chained hash of whole blocks, so `pos` must be on a block boundary. The
     * scheduler places checkpoints on its chunk quantum, a multiple of the block. */
    if (interval_ <= 0 || pos <= 0 || pos % block_size_ != 0) {
        RAD_ERR("checkpoint at %lld is not on a %lld-token block boundary, which the scheduler's "
                "chunk quantum rules out (spec §7.3)",
                (long long)pos, (long long)block_size_);
        return RAD_E_INVAL;
    }
    auto existing = ck_by_prefix_.find(prefix);
    if (existing != ck_by_prefix_.end()) {
        existing->second.used = ++clock_;
        existing->second.used_ns = rad_mono_ns();
        /* Another session's snapshot, this step's save target: retention run by a later
         * reservation in this step places its ladder from THAT session's tip and may not keep
         * this position. Held until the save is committed. */
        if (!reserved(prefix)) reserved_.push_back(prefix);
        *slot_out = existing->second.slot;
        return RAD_OK;
    }

    int32_t slot = -1;
    if (kv_->alloc_checkpoint(&slot) < 0) {
        /* Run retention over every session. Iterate a copy of the keys: retain() erases sessions
         * it empties. */
        std::vector<uint64_t> sessions;
        sessions.reserve(ck_by_session_.size());
        for (const auto& s : ck_by_session_) sessions.push_back(s.first);
        for (uint64_t s : sessions) retain(s);
        if (kv_->alloc_checkpoint(&slot) < 0) {
            /* Evict the least recently used snapshot, where a use is its reservation or a lookup
             * resuming from it, and never one handed out or reserved this step (its admission
             * restores from the slot ahead of the step; the save lands in it after). A tip gets
             * no protection beyond its recency: sessions are request ids, and a client that
             * rewrites its last message leaves one tip per turn that no later turn reaches, while
             * the anchor every turn resumes from is used each turn. */
            const Ckpt* victim = nullptr;
            for (const auto& kvp : ck_by_prefix_) {
                if (handed(kvp.first) || reserved(kvp.first)) continue;
                if (!victim || kvp.second.used < victim->used) victim = &kvp.second;
            }
            if (!victim) {
                ++stats_.ckpt_refused;
                RAD_DEBUG("prefix cache: no snapshot at %lld for session %llu: every one of the "
                          "%lld slots was handed out this step or awaits its save", (long long)pos,
                          (unsigned long long)session, (long long)ck_by_prefix_.size());
                return RAD_E_FULL;
            }
            RAD_DEBUG("prefix cache: store full, snapshot at %lld of session %llu (%s) evicted "
                      "as the least recently used, unused for %.1f s", (long long)victim->pos,
                      (unsigned long long)victim->session,
                      is_tip(*victim) ? "its tip" : "below its tip", idle_s(victim->used_ns));
            const BlockHash vh = victim->prefix;
            const uint64_t vs = victim->session;
            retire_ckpt(ck_by_prefix_.find(vh));
            ++stats_.ckpt_evictions;
            ++stats_.ckpt_lru;
            unindex_ckpt(vs, vh);
            if (kv_->alloc_checkpoint(&slot) < 0) { ++stats_.ckpt_refused; return RAD_E_FULL; }
        }
    }

    Ckpt c;
    c.session = session; c.pos = pos; c.prefix = prefix; c.slot = slot;
    c.used = c.made = ++clock_;
    c.used_ns = rad_mono_ns();
    ck_by_prefix_.emplace(prefix, c);
    ck_by_session_[session].push_back(prefix);
    /* A save target until committed: a lookup's take_over may index it under another session
     * before the save, where retention run by a later reservation in this step would weigh it by
     * that session's ladder. */
    reserved_.push_back(prefix);
    *slot_out = slot;
    return RAD_OK;
}

void PrefixCache::commit_checkpoint(const BlockHash& prefix) {
    /* The save has landed: the slot is a snapshot like any other from here. */
    reserved_.erase(std::remove(reserved_.begin(), reserved_.end(), prefix), reserved_.end());
    auto it = ck_by_prefix_.find(prefix);
    if (it == ck_by_prefix_.end()) return;
    /* One that lost its slot since the reservation holds nothing to serve. */
    if (it->second.slot < 0) return;
    it->second.valid = true;
    /* Tell the tiers (for the session view) only now that the slot holds this state. */
    if (tiers_) tiers_->note_snapshot(prefix, it->second.slot, it->second.pos);
}


/* ------------------------------------------------------------------ the linear half of a session */

void PrefixCache::retire_ckpt(std::unordered_map<BlockHash, Ckpt, BlockHashHash>::iterator it) {
    if (it == ck_by_prefix_.end()) return;
    /* The tiers record the slot for the session view and must stop naming it before it is reused. */
    if (tiers_) tiers_->release_snapshot(it->first);
    if (it->second.slot >= 0) kv_->free_checkpoint(it->second.slot);
    ck_by_prefix_.erase(it);
}

bool PrefixCache::is_tip(const Ckpt& c) const {
    auto s = ck_by_session_.find(c.session);
    if (s == ck_by_session_.end()) return false;
    for (const BlockHash& h : s->second) {
        auto e = ck_by_prefix_.find(h);
        if (e != ck_by_prefix_.end() && e->second.pos > c.pos) return false;
    }
    return true;
}

double PrefixCache::idle_s(uint64_t used_ns) {
    const uint64_t now = rad_mono_ns();
    return now > used_ns ? (double)(now - used_ns) * 1e-9 : 0.0;
}

void PrefixCache::unindex_ckpt(uint64_t session, const BlockHash& key) {
    auto sit = ck_by_session_.find(session);
    if (sit == ck_by_session_.end()) return;
    sit->second.erase(std::remove(sit->second.begin(), sit->second.end(), key), sit->second.end());
    if (sit->second.empty()) ck_by_session_.erase(sit);
}

void PrefixCache::hold(uint64_t req, const std::vector<int32_t>& tokens,
                       const std::vector<MMContentKey>& mm) {
    if (!enabled_ || !tiers_ || tokens.empty() || holds_.count(req)) return;
    const int64_t n_full = ((int64_t)tokens.size() - 1) / block_size_;
    const std::vector<BlockHash> hs = chain_all(tokens, mm, n_full);
    std::vector<BlockHash>& mine = holds_[req];
    for (int64_t b = 0; b < n_full; ++b) {
        auto it = map_.find(hs[(size_t)b]);
        if (it == map_.end()) break;
        if (!same_tokens(it->second, tokens.data() + b * block_size_)) break;
        mine.push_back(hs[(size_t)b]);
        ++held_[hs[(size_t)b]];
    }
}

void PrefixCache::release(uint64_t req) {
    auto it = holds_.find(req);
    if (it == holds_.end()) return;
    for (const BlockHash& h : it->second) {
        auto k = held_.find(h);
        if (k != held_.end() && --k->second <= 0) held_.erase(k);
    }
    holds_.erase(it);
    released_ = true;
}

/* ------------------------------------------------------------------ expiry */

int64_t PrefixCache::expire(uint64_t now_ns, uint64_t ttl_ns) {
    if (!enabled_ || ttl_ns == 0 || now_ns <= ttl_ns) return 0;
    const uint64_t cutoff = now_ns - ttl_ns;
    int64_t n = 0;

    /* ---- in VRAM: the LRU is in clock order, so stop at the first entry kept. A block a running
     * sequence also holds stays allocated; that sequence republishes it when it ends. */
    while (!lru_.empty()) {
        const BlockHash h = lru_.front();
        auto it = map_.find(h);
        if (it == map_.end()) { lru_.pop_front(); continue; }
        Entry& e = it->second;
        if (e.used_ns >= cutoff) break;
        lru_.pop_front();
        e.in_lru = false;
        release_blocks(e);
        /* The host copy goes with it: an expired entry is not demoted. */
        if (tiers_) tiers_->forget(h);
        map_.erase(it);
        ++n;
    }

    /* ---- only in host memory: the tiers keep these in clock order too. */
    if (tiers_) {
        std::vector<BlockHash> keys;
        tiers_->expired(cutoff, (size_t)-1, &keys);
        for (const BlockHash& h : keys) {
            tiers_->forget(h);
            auto it = map_.find(h);
            if (it == map_.end() || it->second.in_lru) continue;
            map_.erase(it);
            ++n;
        }
    }

    /* ---- snapshots: skip uncommitted reservations, ones handed out this step and ones a save
     * is pending on. At most --checkpoint-slots of them, so walk the whole index. */
    std::vector<BlockHash> gone;
    for (const auto& kvp : ck_by_prefix_) {
        const Ckpt& c = kvp.second;
        if (!c.valid || handed(kvp.first) || reserved(kvp.first)) continue;
        auto e = map_.find(kvp.first);
        const uint64_t used = std::max(c.used_ns, e == map_.end() ? 0 : e->second.used_ns);
        if (used < cutoff) gone.push_back(kvp.first);
    }
    for (const BlockHash& h : gone) {
        auto it = ck_by_prefix_.find(h);
        const uint64_t session = it->second.session;
        retire_ckpt(it);
        unindex_ckpt(session, h);
    }
    stats_.expired += n;
    stats_.ckpt_expired += (int64_t)gone.size();
    return n;
}

/* ------------------------------------------------------------------ release on request */

int PrefixCache::release_session(const std::string& label, uint64_t id, int64_t keep_tokens,
                                 SessionRelease* out) {
    if (!out) return RAD_E_INVAL;
    *out = SessionRelease{};
    if (!enabled_ || !tiers_ || block_size_ <= 0) return RAD_E_UNSUPPORTED;
    if (label.empty() && id == 0) return RAD_E_INVAL;
    if (keep_tokens < 0) return RAD_E_INVAL;

    std::vector<IdleTiers::SessionChain> rows;
    tiers_->find_sessions(label, id, &rows);
    if (rows.empty()) return RAD_E_NOTFOUND;
    std::vector<uint64_t> ids;
    for (const IdleTiers::SessionChain& r : rows) ids.push_back(r.id);

    /* A kept token keeps its whole block: the block holding it is what the next turn hits. */
    const int64_t keep_blocks = (keep_tokens + block_size_ - 1) / block_size_;
    int64_t entry_bytes = 0;
    for (int32_t g : entry_groups_) {
        const KVGroupPlan* p = kv_->plan(g);
        if (p) entry_bytes += p->bytes_per_block;
    }

    for (const IdleTiers::SessionChain& row : rows) {
        const int64_t n = (int64_t)row.chain.size();
        const int64_t shared = tiers_->shared_prefix_blocks(row.chain, ids);
        const int64_t floor = std::max(keep_blocks, shared);
        ++out->sessions;
        out->kept_tokens += std::min(keep_blocks, n) * block_size_;
        out->kept_shared += std::max<int64_t>(0, std::min(shared, n) - keep_blocks);

        /* Tip inwards, as expire() and the host tier's pressure do: what is left is still a
         * prefix a lookup reaches. `live_upto` is one past the deepest block a request kept:
         * the row is cut there, not at the kept prefix, so a later release reaches it. */
        int64_t live_upto = 0;
        for (int64_t b = n - 1; b >= floor; --b) {
            const BlockHash& h = row.chain[(size_t)b];
            auto it = map_.find(h);
            if (it != map_.end()) {
                Entry& e = it->second;
                /* A sequence reads it, or a waiting request restored it and holds it against
                 * drop_clean: releasing would free nothing and lose the entry under them. Its
                 * snapshot stays with it; that request most likely resumes from it. */
                if (e.in_lru && (shared_with_live(e) || held_.count(h) > 0)) {
                    ++out->kept_live;
                    if (live_upto == 0) live_upto = b + 1;
                    continue;
                }
                if (e.in_lru) {
                    lru_.erase(e.lru);
                    e.in_lru = false;
                    release_blocks(e);
                    out->device_bytes += entry_bytes;
                }
                /* The host copy goes with the entry (forget), so count it before. */
                if (tiers_->resident_off_device(h) || tiers_->droppable(h))
                    out->host_bytes += tiers_->entry_bytes();
                tiers_->forget(h);
                map_.erase(it);
                ++out->entries;
            }
            auto ck = ck_by_prefix_.find(h);
            if (ck != ck_by_prefix_.end() && ck->second.valid && !handed(h) && !reserved(h)) {
                const uint64_t session = ck->second.session;
                retire_ckpt(ck);
                unindex_ckpt(session, h);
                ++out->snapshots;
            }
        }
        const int64_t cut = std::min(n, std::max(keep_blocks, live_upto));
        tiers_->cut_session(row.id, cut, cut * block_size_);
    }
    out->tokens = out->entries * block_size_;
    stats_.released += out->entries;
    stats_.ckpt_released += out->snapshots;
    return RAD_OK;
}
}  /* namespace rad */
