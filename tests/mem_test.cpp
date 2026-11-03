/* mem_test.cpp -- core/mem: pools, the KV block manager, prefix caching, the idle tiers.
 *
 * These properties fail silently: a free list leaking one block per fork stops admission after an
 * hour, and a chained-hash collision mid-prefix serves one conversation another's context as
 * fluent wrong output (spec §17).
 */
#include "rad_test.h"

#include "mem/kv.h"
#include "kvstream/bringup.h"
#include "kvstream/bind.h"
#include "kvstream/plan.h"
#include "mem/pools.h"
#include "mem/prefix.h"
#include "mem/vram_budget.h"
#include "mem/kvtier.h"
#include "device/device.h"
#include "plugin/registry.h"
#include "runtime/tierexec.h"
#include "sched/mem_iface.h"
#include "rad_internal.h"

/* Only FakeKV: this binary runs on the real device layer. */
#define SCHED_FAKES_KV_ONLY
#include "sched_fakes.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <unistd.h>

using namespace rad;

/* ------------------------------------------------------------------ fixtures */

static Config small_config() {
    Config c;
    c.vram_weights_mib = 1;
    c.vram_kv_mib = 1;
    c.gpu_headroom_mib = 1;
    c.host_pool_mib = 0;
    c.max_seqs = 8;
    c.checkpoint_interval = 64;
    c.prefix_cache = true;
    return c;
}

/* Handle 0 is the null group. The builder reserves slot 0 of every handle table so a zero handle
 * means "none", and KVManager indexes its plans by handle, so a real group at index 0 would read
 * as "no group". The sentinel keeps fixtures in the shape the engine produces. */
static KVGroupInfo null_group() { return KVGroupInfo{}; }

/* The handles hybrid_groups() hands out: with the sentinel at 0, the paged group is 1 and the
 * linear one 2. */
enum : int32_t { G_NULL = 0, G_FULL = 1, G_GDN = 2 };

/* A hybrid model in miniature: full attention over 2 layers at a 16-token block, and GDN
 * recurrent state over 4 layers. Small sizes, real geometry. */
static std::vector<KVGroupInfo> hybrid_groups(bool with_linear = true) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    {
        KVGroupInfo k;
        k.name = "full";
        k.decl.kind = RAD_KV_FULL;
        k.decl.dtype = RAD_BF16;
        k.decl.n_head_kv = 2;
        k.decl.head_dim = 8;
        k.block_size = 16;             /* off the resolved attention kernel, never a constant */
        k.layers = { 0, 1 };
        g.push_back(k);                /* page = 16*2*8*2*2*2 = 2048 B */
    }
    if (with_linear) {
        KVGroupInfo k;
        k.name = "gdn";
        k.decl.kind = RAD_KV_LINEAR;
        k.decl.dtype = RAD_F32;
        k.decl.n_head_kv = 4;
        k.decl.state_dim[0] = 16;
        k.decl.state_dim[1] = 16;
        k.layers = { 2, 3, 4, 5 };
        g.push_back(k);                /* state = 4*16*16*4*4 = 16 KiB */
    }
    return g;
}

/* Pools over plain malloc for both slabs, as every test here builds them. */
static int configure_pools(Pools& p, const Config& c, int64_t capacity = 0) {
    return p.configure(c, capacity, &rad_malloc_allocator(), &rad_malloc_allocator());
}

struct Fixture {
    Pools pools;
    HostArena ck;              /* the engine's snapshot store, in miniature */
    KVManager kv;
    PrefixCache pc;
    Config cfg = small_config();

    int build(bool with_linear = true, int64_t ckpt_slots = 4) {
        cfg.checkpoint_slots = with_linear ? ckpt_slots : 0;   /* --checkpoint-slots */
        return build_groups(hybrid_groups(with_linear));
    }
    int build_groups(const std::vector<KVGroupInfo>& groups) {
        int s = configure_pools(pools, cfg);
        if (s < 0) return s;
        s = kv.configure(groups, cfg, pools.kv());
        if (s < 0) return s;
        if (kv.checkpoint_bytes() > 0 && cfg.checkpoint_slots > 0) {
            s = ck.open(kv.checkpoint_bytes(), kv.checkpoint_bytes() * cfg.checkpoint_slots);
            if (s < 0) return s;
            kv.set_checkpoint_store(&ck, 0);
        }
        return pc.configure(cfg, &kv, cfg.checkpoint_interval);
    }
};

/* The streamed path a plugin declares for `g`, as the builder records it: a kv_resolve stating the
 * group's per-layer block pitch and selection pitch, and a kv_store_streamed stating its page.
 * `with_resolve`/`with_store` build a model that declares one or neither (as the DSA-index models
 * do). `store_resolved` false is the packed-cache shape: the store declared with no kernel row, so
 * the builder keeps it with one band and both domains null. */
static KernelRow& a_resolved_row() {
    /* Static, as a plugin table's rows are: a Resolved points at one. Nothing in this test issues
     * the op, so the row carries no kernel. */
    static KernelRow row;
    return row;
}

static Program stream_program(const KVGroupInfo& g, int64_t sel_pitch, bool with_resolve = true,
                              bool with_store = true, bool store_resolved = true) {
    Program p;
    p.meta.arch_id = "test-arch";
    const int64_t n_bound = g.layers.empty() ? 1 : (int64_t)g.layers.size();
    const int64_t layer_bytes = g.bytes_per_block > 0 ? g.bytes_per_block / n_bound : 0;
    if (with_resolve) {
        OpInfo oi;
        oi.op = "kv_resolve";
        oi.bands.resize(1);          /* resolved: the builder records one band per domain */
        oi.bands[0].dom[RAD_DOMAIN_DEVICE].row = &a_resolved_row();
        oi.base.set_i("block_bytes", layer_bytes);
        oi.base.set_i("sel_pitch", sel_pitch);
        p.ops.push_back(std::move(oi));
    }
    if (with_store) {
        OpInfo oi;
        oi.op = "kv_store_streamed";
        oi.bands.resize(1);
        if (store_resolved) oi.bands[0].dom[RAD_DOMAIN_DEVICE].row = &a_resolved_row();
        else oi.bands[0].miss[RAD_DOMAIN_DEVICE] = "kv_dtype in {fp8_e4m3}, have fp8_v4";
        oi.base.set_i("block_size", g.block_size);
        p.ops.push_back(std::move(oi));
    }
    return p;
}

static std::vector<int32_t> tok_run(int32_t base, int n) {
    std::vector<int32_t> v;
    for (int i = 0; i < n; ++i) v.push_back(base + i);
    return v;
}

/* Give the sequence `id` enough blocks for `n_tok` and publish its whole blocks under `tokens`. */
static void publish(Fixture& f, uint64_t id, const std::vector<int32_t>& tokens, int64_t n_tok) {
    CHECK_OK(f.kv.add_sequence(id));
    CHECK_OK(f.kv.ensure(id, (int64_t)tokens.size()));
    std::vector<std::vector<int32_t>> bt;
    for (int32_t g : f.pc.cached_groups()) bt.push_back(f.kv.block_table(id, g));
    CHECK_OK(f.pc.insert(id, tokens, {}, bt, n_tok));
}

/* The chained block hashes of `tokens`, one per block, as the cache computes them. */
static std::vector<BlockHash> chain_of(const std::vector<int32_t>& tokens, int bs = 16) {
    std::vector<BlockHash> hs;
    BlockHash parent;
    for (int b = 0; b + bs <= (int)tokens.size(); b += bs) {
        parent = chain_block_hash(parent, tokens.data() + b, bs, nullptr, 0, b);
        hs.push_back(parent);
    }
    return hs;
}

/* Reserve and commit the checkpoint at `pos` (a multiple of 16) under `hs`; the slot it took. */
static int32_t ckpt_at(Fixture& f, uint64_t session, const std::vector<BlockHash>& hs,
                       int64_t pos) {
    int32_t slot = -1;
    CHECK_OK(f.pc.reserve_checkpoint(session, pos, hs[(size_t)(pos / 16 - 1)], &slot));
    CHECK(slot >= 0);
    f.pc.commit_checkpoint(hs[(size_t)(pos / 16 - 1)]);
    return slot;
}

/* ================================================================== pools */

TEST(pools_refuse_a_split_the_resolver_never_made) {
    Config c;
    c.vram_weights_mib = 0;     /* the resolver never ran */
    c.vram_kv_mib = 0;
    Pools p;
    CHECK_EQ(configure_pools(p, c), RAD_E_INVAL);
    CHECK(p.fit_report().find("vram_weights resolved to 0 MiB") != std::string::npos);
    CHECK(p.fit_report().find("vram_kv resolved to 0 MiB") != std::string::npos);
    CHECK(!p.configured());
}

TEST(pool_oversubscription_is_refused_by_name) {
    Config c;
    c.vram_weights_mib  = 24 * 1024;
    c.vram_kv_mib       = 12 * 1024;
    c.gpu_headroom_mib  =  1 * 1024;
    Pools p;
    const int64_t capacity = 32ll * 1024 * 1024 * 1024;   /* a 32 GiB card */
    CHECK_EQ(configure_pools(p, c, capacity),
             RAD_E_NOMEM);
    /* Named, with the shortfall: refusing to start beats silently shrinking a pool (spec §6). */
    const std::string& r = p.fit_report();
    CHECK(r.find("vram_weights") != std::string::npos);
    CHECK(r.find("vram_kv") != std::string::npos);
    CHECK(r.find("over by") != std::string::npos);
    CHECK(r.find("4.00 GiB") != std::string::npos);  /* 36 GiB wanted against a 32 GiB card */
    CHECK(!p.configured());
}

TEST(pool_reservation_never_falls_through) {
    Config c = small_config();
    Pools p;
    CHECK_OK(configure_pools(p, c));

    /* The KV pool is 1 MiB; asking for 2 is refused by name and leaves the pool untouched. No
     * path leads to the weight slab (spec §6): the mover works in its own slab and cannot evict a
     * KV block to promote an expert. */
    void* q = p.kv().reserve("oversized block array", 2 << 20, RAD_ALIGN_UNIT);
    CHECK(q == nullptr);
    CHECK(!p.kv().ok());
    REQUIRE_EQ((long long)p.kv().shortfalls().size(), 1ll);
    CHECK_EQ(p.kv().shortfalls()[0].pool, std::string("vram_kv"));
    CHECK_EQ(p.kv().shortfalls()[0].missing(), (2ll << 20) - (1ll << 20));
    CHECK_EQ(p.kv().used(), 0ll);
    CHECK_EQ(p.weights().used(), 0ll);          /* nothing leaked into the weight slab */

    void* ok = p.kv().reserve("block array", 512 << 10, RAD_ALIGN_UNIT);
    CHECK(ok != nullptr);
    CHECK_EQ(p.kv().used(), 512ll << 10);
}

/* A refused backing allocation is reported as a refusal, with the allocator's reason, not as
 * "0 B available", which would read as a measurement of the card. */
TEST(a_refused_backing_allocation_says_it_was_refused_and_why) {
    struct Refuses final : MemAllocator {
        void* alloc(int64_t, int) override { return nullptr; }
        void  free(void*, int) override {}
        const char* name() const override { return "refuses"; }
    } none;
    Config c = small_config();
    Pools p;
    CHECK_EQ(p.configure(c, 0, &none, &none), RAD_E_NOMEM);
    const std::string& r = p.fit_report();
    CHECK(r.find("backing allocation of") != std::string::npos);
    CHECK(r.find("refused: the allocator returned nothing") != std::string::npos);
    CHECK(r.find("0 B available") == std::string::npos);
    CHECK(!p.configured());
}

/* ================================================================== the derived budget */

TEST(a_device_with_no_capacity_sizes_its_pools_from_the_program) {
    /* A device reporting no capacity: pools are derived from what the program needs, not from a
     * share. */
    Config c = small_config();
    c.vram_weights_mib = 0;
    c.vram_kv_mib      = 0;
    VramFacts f{};
    f.statics    = 0;                 /* every op on the host: its weights stay in the mapping */
    f.kv_ceiling = 300LL << 20;
    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK_EQ(c.vram_weights_mib, (int64_t)1);           /* a pool nothing is placed in */
    CHECK_EQ(c.vram_kv_mib, (int64_t)300);              /* every sequence at the full context */

    f.statics = 5LL << 20;                              /* a device-domain op's weights */
    Config c2 = small_config();
    c2.vram_weights_mib = 0;
    c2.vram_kv_mib      = 0;
    CHECK_OK(vram_budget_resolve(c2, { f }, &b));
    CHECK_EQ(c2.vram_weights_mib, (int64_t)5);
}

TEST(a_unified_device_charges_its_host_pool_to_the_same_ledger) {
    /* Unified memory: the pinned host pool is the same RAM as the card, so it comes off
     * claimable before the split. The same facts without the flag charge nothing. */
    auto facts = [] {
        VramFacts f{};
        f.capacity   = 16LL << 30;
        f.free       = f.capacity;
        f.arena      = 64LL << 20;
        f.kv_ceiling = 300LL << 20;
        return f;
    };
    Config c1 = small_config();
    VramBudget b1;
    CHECK_OK(vram_budget_resolve(c1, { facts() }, &b1));
    CHECK(!b1.unified);
    CHECK_EQ(b1.host_pinned, (int64_t)0);

    VramFacts f = facts();
    f.unified     = true;
    f.host_pinned = 2LL << 30;
    Config c2 = small_config();
    VramBudget b2;
    CHECK_OK(vram_budget_resolve(c2, { f }, &b2));
    CHECK(b2.unified);
    CHECK_EQ(b2.host_pinned, 2LL << 30);
    CHECK_EQ(b1.claimable - b2.claimable, 2LL << 30);
    CHECK(b2.report.find("unified memory") != std::string::npos);
}

TEST(a_program_that_copies_no_weight_still_gets_a_weights_pool) {
    /* With every op on the host (CPU-only serving), weights are read from the container mapping
     * and nothing is copied to the weights pool. The pool must still exist, since the pools refuse
     * a zero budget as an unresolved split. */
    Config c = small_config();
    VramFacts f{};
    f.capacity   = 16LL << 30;        /* the host backend presents half the machine as a card */
    f.free       = f.capacity;
    f.arena      = 64LL << 20;
    f.kv_ceiling = 300LL << 20;
    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK_EQ(c.vram_weights_mib, (int64_t)1);
    CHECK(c.vram_kv_mib >= 299);
}

TEST(derived_budget_takes_the_statics_off_the_top_then_divides_the_rest) {
    /* With no lender the experts come first; the cases below cover each side's share. */
    Config c = small_config();
    c.vram_weights_mib = 0;              /* both unset: derive */
    c.vram_kv_mib      = 0;
    c.gpu_headroom_mib = 96;
    c.max_seqs = 8;
    c.max_ctx  = 2048;

    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);   /* driver + code objects already resident */
    f.arena      = 144LL << 20;
    f.statics    = 2LL << 30;                    /* this rank's shard of the non-expert weights */
    f.experts    = 200LL << 30;                  /* far more than fits: the plane is elastic */
    f.kv_ceiling = 100LL << 30;                  /* far above any share, so no cap fires */
    f.kv_one     = 1LL << 30;

    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));

    /* Claimable is capacity minus what is held, the driver's drift on it, the headroom and the
     * arena. */
    CHECK_EQ(b.drift, 16LL << 20);
    CHECK_EQ(b.claimable, f.capacity - (512LL << 20) - (16LL << 20) - (96LL << 20) - (144LL << 20));

    /* The statics come out whole before the split, so the cache's share can never demote an
     * attention projection to host memory. */
    CHECK_EQ(b.statics, f.statics);
    CHECK_EQ(b.elastic, b.claimable - f.statics);
    CHECK_EQ(b.experts + b.kv, b.elastic);
    CHECK_EQ(b.weights, b.statics + b.experts);

    const int64_t MiB = 1024 * 1024;
    const int64_t unclaimed = f.capacity - (512LL << 20) - f.arena
                            - c.vram_weights_mib * MiB - c.vram_kv_mib * MiB;
    CHECK(unclaimed >= 112 * MiB);                /* at least the headroom plus drift */
    CHECK(unclaimed < 114 * MiB);                 /* at most that plus MiB rounding */
}

static Config derive_config() {
    Config c = small_config();
    c.vram_weights_mib = 0; c.vram_kv_mib = 0; c.gpu_headroom_mib = 96;
    return c;
}
static VramFacts big_moe_facts() {
    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);
    f.arena      = 144LL << 20;
    f.statics    = 4LL << 30;
    f.experts    = 40LL << 30;                   /* more than the card: some must stream */
    f.kv_ceiling = 100LL << 30;
    f.kv_one     = 2LL << 30;
    return f;
}

/* A lending cache takes everything the experts can do without. The experts keep on the card only
 * what the host pool cannot hold (expert_vram_floor, checked against the planner in place_test);
 * every block the cache does not use is lent back to them. */
TEST(a_lending_cache_takes_everything_the_experts_can_do_without) {
    VramFacts f = big_moe_facts();
    f.lends     = true;
    f.expert_floor = 16LL << 30;                 /* the host pool takes 24 of the 40 */
    Config c = derive_config();
    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK(b.lends);
    CHECK_EQ(b.expert_floor, 16LL << 30);
    CHECK_EQ(b.experts, b.expert_floor);
    CHECK_EQ(b.kv, b.elastic - b.expert_floor);
    CHECK_EQ(b.weights, b.statics + b.experts);

    /* A larger host pool takes experts off the card, and the cache gets exactly that much more. */
    VramFacts g = f;
    g.expert_floor = 12LL << 30;
    Config c2 = derive_config();
    VramBudget b2;
    CHECK_OK(vram_budget_resolve(c2, { g }, &b2));
    CHECK_EQ(b2.experts, b.experts - (4LL << 30));
    CHECK_EQ(b2.kv, b.kv + (4LL << 30));
    CHECK_EQ(b2.statics, b.statics);
}

/* Without a lender an unused cache block sits empty, so the experts come first and the cache
 * keeps one --max-model-len sequence. */
TEST(without_a_lender_the_experts_come_first_and_the_cache_keeps_one_sequence) {
    VramFacts f = big_moe_facts();
    f.lends     = false;
    f.expert_floor = 16LL << 30;
    Config c = derive_config();
    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK(!b.lends);
    CHECK_EQ(b.kv, f.kv_one);
    CHECK_EQ(b.experts, b.elastic - f.kv_one);
}

/* The cache always gets at least one full-length sequence: an accepted context no pool can hold
 * would be a configuration error, and any expert shortfall is the planner's to report. */
TEST(the_cache_keeps_one_full_length_sequence_even_against_the_expert_floor) {
    VramFacts f = big_moe_facts();
    f.lends     = true;
    f.expert_floor = f.experts;                  /* every expert wants the card */
    Config c = derive_config();
    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK_EQ(b.kv, f.kv_one);
    CHECK_EQ(b.experts, b.elastic - f.kv_one);

    /* A cache capped by its ceiling gives the rest to the experts. */
    VramFacts g = big_moe_facts();
    g.lends      = true;
    g.expert_floor = 1LL << 30;
    g.kv_ceiling = 3LL << 30;
    Config c2 = derive_config();
    VramBudget b2;
    CHECK_OK(vram_budget_resolve(c2, { g }, &b2));
    CHECK_EQ(b2.kv, g.kv_ceiling);
    CHECK_EQ(b2.experts, b2.elastic - g.kv_ceiling);
}

/* A dense model whose weights nearly fill the card: the statics are a floor, so all_vram leaves
 * the cache short of one full-length sequence, and an explicit layer_offload instead takes static
 * weight off the card (the planner spills those layers) until that sequence fits. */
TEST(an_explicit_layer_offload_takes_statics_off_the_card_for_one_sequence) {
    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);
    f.arena      = 4LL << 30;
    f.statics    = 24LL << 30;
    f.kv_ceiling = 64LL << 30;
    f.kv_one     = 8LL << 30;
    for (const char* pl : { "all_vram", "layer_offload" }) {
        Config c = small_config();
        c.vram_weights_mib = 0;
        c.vram_kv_mib      = 0;
        c.gpu_headroom_mib = 96;
        c.placement = pl;
        VramBudget b;
        CHECK_OK(vram_budget_resolve(c, { f }, &b));
        if (!std::strcmp(pl, "all_vram")) {
            CHECK_EQ(b.statics, f.statics);
            CHECK(b.kv < f.kv_one);
        } else {
            CHECK_EQ(b.statics, b.claimable - f.kv_one);
            CHECK_EQ(b.kv, f.kv_one);
            CHECK_EQ(b.weights, b.statics);
        }
    }
}

/* With a lending cache at its floor, the prefill staging region is paid by expert residency
 * above the host pool's floor, and never by an expert that has no other home. */
TEST(prefill_staging_can_borrow_expert_slack_when_the_cache_is_small) {
    VramFacts f = big_moe_facts();
    f.lends = true;
    f.kv_one = f.kv_ceiling = 64LL << 20;
    f.expert_floor = 16LL << 30;
    f.expert_layer_max = 2LL << 30;
    Config c = derive_config();
    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK(b.stage > 0);
    CHECK(b.report.find("out of the experts") != std::string::npos);
    CHECK_EQ(b.kv, f.kv_one);
    CHECK(b.experts >= f.expert_floor);
    CHECK_EQ(b.weights, b.statics + b.experts);
    CHECK_EQ(b.experts + b.kv + b.stage, b.elastic);

    // With no host-pool slack, staging must not evict an expert that has no other home.
    f.expert_floor = b.elastic - f.kv_one;
    Config tight = derive_config();
    VramBudget no_ring;
    CHECK_OK(vram_budget_resolve(tight, { f }, &no_ring));
    CHECK_EQ(no_ring.stage, 0ll);
    CHECK_EQ(no_ring.kv, f.kv_one);
    CHECK_EQ(no_ring.experts, f.expert_floor);
}

/* When experts can pay, the staging region leaves the cache at its full coverage. */
TEST(prefill_staging_preserves_cache_coverage_when_experts_can_pay) {
    VramFacts f = big_moe_facts();
    f.lends = true;
    f.kv_ceiling = 2LL << 30;
    f.kv_one = 768LL << 20;
    f.expert_floor = 16LL << 30;
    f.expert_layer_max = 2LL << 30;
    Config c = derive_config();
    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK(b.stage > 0);
    CHECK_EQ(b.kv, f.kv_ceiling);
    CHECK(b.experts >= f.expert_floor);
    CHECK(b.report.find("out of the experts") != std::string::npos);
    CHECK_EQ(b.experts + b.kv + b.stage, b.elastic);
}

/* A hybrid model's recurrent and conv slots stay allocated with one request active, so the
 * one-sequence floor is the inverse of the live plan, not a separate one-sequence plan: that plan
 * drops the fixed state and falls short of max_ctx. */
TEST(hybrid_context_floor_accounts_for_all_configured_state_slots) {
    Config c = small_config();
    c.max_seqs = 4;
    c.max_ctx = 32768;
    c.checkpoint_slots = 0;
    auto groups = hybrid_groups();
    KVManager actual, single;
    CHECK_OK(actual.plan_groups(groups, c));
    Config one = c;
    one.max_seqs = 1;
    CHECK_OK(single.plan_groups(groups, one));
    // A one-sequence declaration drops fixed state that the live four-slot plan owns.
    CHECK(actual.paged_tokens(single.ceiling(c.max_ctx)) < c.max_ctx);
    constexpr int64_t MiB = 1ll << 20;
    const int64_t ceiling_mib = (actual.ceiling(c.max_ctx) + MiB - 1) / MiB;
    const int64_t floor_mib = actual.pool_mib_for_tokens(c.max_ctx, ceiling_mib);
    CHECK(floor_mib > 0);
    CHECK(actual.paged_tokens(floor_mib * MiB) >= c.max_ctx);
    CHECK((floor_mib - 1) * MiB < actual.fixed_bytes() ||
          actual.paged_tokens((floor_mib - 1) * MiB) < c.max_ctx);
}

TEST(neither_side_is_given_more_than_it_could_use) {
    /* A model that fits: 4 GiB of experts and a cache that can address 1.2 GiB cannot use a
     * 32 GiB card, so the remainder is reported unusable rather than allocated to an idle pool. */
    Config c = small_config();
    c.vram_weights_mib = 0; c.vram_kv_mib = 0; c.gpu_headroom_mib = 96;

    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);
    f.arena      = 144LL << 20;
    f.statics    = 2LL << 30;
    f.experts    = 4LL << 30;
    f.kv_ceiling = 1200LL << 20;

    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK_EQ(b.experts, f.experts);
    CHECK_EQ(b.kv, f.kv_ceiling);
    CHECK_EQ(b.weights, b.statics + b.experts);
    CHECK_EQ(b.experts + b.kv + b.unusable, b.elastic);
    CHECK(b.unusable > 20LL << 30);               /* a big card and a small model */
}

TEST(a_dense_model_has_no_expert_share_to_lose) {
    /* A dense model: no experts, so no expert share. Giving the weights an unclaimable share
     * would reserve tens of GiB for nothing and leave the cache short of its worst case. */
    Config c = small_config();
    c.vram_weights_mib = 0; c.vram_kv_mib = 0; c.gpu_headroom_mib = 96;
    c.max_seqs = 8;
    c.max_ctx  = 131072;

    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);
    f.arena      = 144LL << 20;
    /* Deliberately not a whole MiB: with no expert share padding it, the weights budget is the
     * static footprint to the byte, and MiB truncation could leave the pool short. */
    f.statics    = (1331LL << 20) + 12345;
    f.experts    = 0;
    f.kv_ceiling = 10LL << 30;

    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { f }, &b));
    CHECK_EQ(b.experts, 0ll);
    CHECK_EQ(b.kv, f.kv_ceiling);                 /* everything a paged cache can address */
    CHECK_EQ(b.experts + b.kv + b.unusable, b.elastic);

    /* The MiB figure the pool is built from must still cover the statics. */
    const int64_t MiB = 1024 * 1024;
    CHECK_EQ(b.weights, f.statics);               /* no expert share left to pad it */
    CHECK(c.vram_weights_mib * MiB >= f.statics); /* rounded up, not truncated */
    CHECK(c.vram_weights_mib * MiB - f.statics < MiB);
}

/* The engine's hardware queues are a fixed term taken before the split. Each costs VRAM the driver
 * backs it with, and streams are created after the card is read, so omitting them eats the
 * headroom. The rank that needs most sets the term, as with every fold here. */
TEST(the_engine_s_queues_come_off_the_card_before_the_split) {
    Config c = small_config();
    c.vram_weights_mib = 0;
    c.vram_kv_mib      = 0;
    c.gpu_headroom_mib = 96;
    VramFacts f{};
    f.capacity   = 32LL << 30;
    f.free       = f.capacity - (512LL << 20);
    f.arena      = 144LL << 20;
    f.staging    = 98LL << 20;
    f.statics    = 2LL << 30;
    f.experts    = 200LL << 30;
    f.kv_ceiling = 100LL << 30;
    VramBudget b0;
    CHECK_OK(vram_budget_resolve(c, { f }, &b0));

    /* A fresh config, since the resolver writes its answer back into the one it is given. */
    Config c2 = small_config();
    c2.vram_weights_mib = 0;
    c2.vram_kv_mib      = 0;
    c2.gpu_headroom_mib = 96;
    VramFacts g = f, h = f;
    g.n_queues = 2; g.queue_bytes = 33LL << 20;
    h.n_queues = 3; h.queue_bytes = 33LL << 20;     /* a rank with a second compute lane */
    VramBudget b;
    CHECK_OK(vram_budget_resolve(c2, { g, h }, &b));
    CHECK_EQ(b.queues, 99LL << 20);
    CHECK_EQ(b.n_queues, 3);
    CHECK_EQ(b.claimable, b0.claimable - (99LL << 20));
    CHECK(b.report.find("hardware queues") != std::string::npos);

    /* After the pools, queues and staging, what remains is still the headroom plus drift. */
    const int64_t MiB = 1024 * 1024;
    const int64_t unclaimed = f.capacity - (512LL << 20) - f.arena - f.staging - b.queues
                            - c2.vram_weights_mib * MiB - c2.vram_kv_mib * MiB;
    CHECK(unclaimed >= 112 * MiB);
    CHECK(unclaimed < 114 * MiB);
}

TEST(the_smallest_card_binds_and_a_short_one_refuses_by_name) {
    Config c = small_config();
    c.vram_weights_mib = 0;
    c.vram_kv_mib      = 0;
    c.gpu_headroom_mib = 96;

    VramFacts big{}, small{};
    big.capacity   = 32LL << 30;  big.free   = big.capacity - (512LL << 20);
    big.arena      = 144LL << 20; big.kv_ceiling = 100LL << 30;
    small = big;
    small.capacity = 16LL << 30;  small.free = small.capacity - (512LL << 20);

    VramBudget b;
    CHECK_OK(vram_budget_resolve(c, { big, small }, &b));
    CHECK_EQ(b.capacity, 16LL << 30);             /* one split must fit every card */

    /* A card the fixed terms alone overrun is refused with the shortfall named. */
    Config c2 = small_config();
    c2.vram_weights_mib = 0; c2.vram_kv_mib = 0;
    c2.gpu_headroom_mib = 96;
    VramFacts tiny{};
    tiny.capacity = 256LL << 20;
    tiny.free     = 64LL << 20;
    tiny.arena    = 144LL << 20;
    VramBudget b2;
    CHECK(vram_budget_resolve(c2, { tiny }, &b2) < 0);
    CHECK(b2.report.find("short by") != std::string::npos);
}

/* ================================================================== group sizing */

TEST(group_sizing_wastes_nothing) {
    /* 48 gated-delta-net + 16 full + 5 drafter layers. vLLM's rule picks the smallest bundle, 5,
     * which divides neither 48 nor 16: 75 slots for 69 layers in 15 groups; patch_kv_group_size's
     * search gives 72/9. Radiance uses 69 in 3, since each KV group is one state kind with its own
     * pool and no uniform page size to pad to. */
    BundleComparison c = compare_bundling({ 48, 16, 5 });
    CHECK_EQ(c.stock_group_size, 5ll);
    CHECK_EQ(c.stock_groups, 15ll);
    CHECK_EQ(c.stock_slots, 75ll);
    CHECK_EQ(c.patched_group_size, 8ll);
    CHECK_EQ(c.patched_groups, 9ll);
    CHECK_EQ(c.patched_slots, 72ll);
    CHECK_EQ(c.our_groups, 3ll);
    CHECK_EQ(c.our_slots, 69ll);           /* == sum(bundles): no padding */

    /* An n:1 model pads nothing under vLLM's rule, nor here. */
    BundleComparison n1 = compare_bundling({ 50, 10 });
    CHECK_EQ(n1.stock_slots, 60ll);
    CHECK_EQ(n1.our_slots, 60ll);
    CHECK_EQ(n1.our_groups, 2ll);
}

/* ================================================================== KV manager */

TEST(kv_configure_reports_and_sizes) {
    Fixture f;
    CHECK_OK(f.build());
    const KVGroupPlan* full = f.kv.plan(G_FULL);
    const KVGroupPlan* gdn  = f.kv.plan(G_GDN);
    CHECK(full && gdn);
    CHECK_EQ(full->block_size, 16ll);
    CHECK_EQ(full->bytes_per_block, 2048ll);      /* 16 tok * 2 kv-heads * 8 dim * 2 B * K/V * 2 layers */
    CHECK_EQ(gdn->bytes_per_state, 16384ll);      /* 4 heads * 16x16 * 4 B * 4 layers */
    CHECK_EQ(gdn->n_states, 8ll);                 /* exactly max_seqs */
    CHECK_EQ(f.kv.checkpoint_bytes(), 16384ll);
    CHECK_EQ(f.kv.checkpoint_slots(), 4ll);
    CHECK(full->n_blocks > 0);
    /* 1 MiB minus 8 states and one alignment unit, in 2 KiB pages. The unit covers reserve()'s
     * alignment padding, which a carve of every remaining byte would overrun. The four snapshots
     * live in host memory and cost the pool nothing. */
    CHECK_EQ(full->n_blocks, ((1ll << 20) - 8 * 16384 - RAD_ALIGN_UNIT) / 2048);
}

/* A --kv-recipe of mixed layouts reaches the core as one attention group per layout, each declared
 * at its row's bytes over two as E4M3 columns (qwen35_fp8: 32 at fp8, 22 at k6_v4 for head_dim
 * 32). The pool is split so every group holds the same tokens, each paying its own page, so the
 * narrow layers buy context: four fp8 layers cost 4 x 2048 bytes a block, one fp8 and three k6_v4
 * layers 2048 + 3 x 1408. */
TEST(a_mixed_kv_recipe_sizes_the_pool_from_each_layouts_own_page) {
    auto group = [](const char* name, int64_t head_dim, std::vector<int> layers) {
        KVGroupInfo k;
        k.name = name;
        k.decl.kind = RAD_KV_FULL;
        k.decl.dtype = RAD_F8E4M3;
        k.decl.n_head_kv = 2;
        k.decl.head_dim = head_dim;
        k.block_size = 16;
        k.layers = std::move(layers);
        return k;
    };
    Fixture uni, mix;
    uni.cfg.checkpoint_slots = mix.cfg.checkpoint_slots = 0;
    CHECK_OK(uni.build_groups({ null_group(), group("kv_attn", 32, { 3, 7, 11, 15 }) }));
    CHECK_OK(mix.build_groups({ null_group(), group("kv_attn_fp8", 32, { 3 }),
                                group("kv_attn_k6_v4", 22, { 7, 11, 15 }) }));
    const KVGroupPlan* u  = uni.kv.plan(1);
    const KVGroupPlan* f8 = mix.kv.plan(1);
    const KVGroupPlan* k6 = mix.kv.plan(2);
    REQUIRE(u && f8 && k6);
    CHECK_EQ(u->bytes_per_block, 4 * 2048ll);
    CHECK_EQ(f8->bytes_per_block, 2048ll);
    CHECK_EQ(k6->bytes_per_block, 3 * 1408ll);
    CHECK_EQ(f8->n_blocks, k6->n_blocks);   /* one block table: the same tokens in every layer */
    const int64_t page = f8->bytes_per_block + k6->bytes_per_block;
    CHECK(f8->n_blocks * page <= (1ll << 20));
    CHECK((f8->n_blocks + 1) * page > (1ll << 20) - 2 * RAD_ALIGN_UNIT);
    CHECK(f8->n_blocks > u->n_blocks);
    CHECK_EQ(f8->n_blocks, ((1ll << 20) - 2 * RAD_ALIGN_UNIT) / page);
}

TEST(kv_ranks_with_unequal_pools_carve_the_least_ranks_blocks_under_a_cap) {
    /* One block table serves every rank (Engine::serve), so every rank's paged groups hold the
     * same block count. With unequal pools (here 1 MiB and 2 MiB), Engine::configure_kv plans each
     * rank, asks what it could page, and caps all at the least before carving. */
    Fixture a, b;
    a.cfg.checkpoint_slots = b.cfg.checkpoint_slots = 0;
    b.cfg.vram_kv_mib = 2;
    CHECK_OK(configure_pools(a.pools, a.cfg));
    CHECK_OK(configure_pools(b.pools, b.cfg));
    CHECK_OK(a.kv.plan_groups(hybrid_groups(), a.cfg));
    CHECK_OK(b.kv.plan_groups(hybrid_groups(), b.cfg));
    const int64_t ta = a.kv.paged_tokens(a.pools.kv().avail());
    const int64_t tb = b.kv.paged_tokens(b.pools.kv().avail());
    CHECK(ta > 0);
    CHECK(tb > ta);                              /* the larger pool pages more uncapped */
    const int64_t least = ta < tb ? ta : tb;
    a.kv.cap_paged_tokens(least);
    b.kv.cap_paged_tokens(least);
    CHECK_OK(a.kv.carve(a.pools.kv()));
    CHECK_OK(b.kv.carve(b.pools.kv()));
    CHECK_EQ(a.kv.plan(G_FULL)->n_blocks, b.kv.plan(G_FULL)->n_blocks);
    CHECK_EQ(a.kv.plan(G_FULL)->n_blocks, least / 16);
    /* The cap does not affect unpaged groups: each rank keeps its own state count. */
    CHECK_EQ(a.kv.plan(G_GDN)->n_states, b.kv.plan(G_GDN)->n_states);
    /* paged_tokens() carves nothing; uncapped, carve() spends exactly its figure. */
    Fixture c;
    c.cfg.checkpoint_slots = 0;
    CHECK_OK(configure_pools(c.pools, c.cfg));
    CHECK_OK(c.kv.plan_groups(hybrid_groups(), c.cfg));
    const int64_t tc = c.kv.paged_tokens(c.pools.kv().avail());
    CHECK_OK(c.kv.carve(c.pools.kv()));
    CHECK_EQ(c.kv.plan(G_FULL)->n_blocks, tc / 16);
}

/* A rank whose only paged group is zero (rad_kv_group_zero) pages nothing itself: it reports no
 * token limit, takes the cap set by the ranks holding the cache, carves nothing for the group, and
 * still holds the shared table's block count. Without a cap the carve is refused rather than
 * leaving the group at zero blocks. */
TEST(kv_a_rank_whose_only_paged_group_is_zero_takes_the_others_block_count) {
    std::vector<KVGroupInfo> zg = hybrid_groups();
    zg[1].zero = true;                           /* the only paged group */
    Fixture a, z;
    a.cfg.checkpoint_slots = z.cfg.checkpoint_slots = 0;
    CHECK_OK(configure_pools(a.pools, a.cfg));
    CHECK_OK(configure_pools(z.pools, z.cfg));
    CHECK_OK(a.kv.plan_groups(hybrid_groups(), a.cfg));
    CHECK_OK(z.kv.plan_groups(zg, z.cfg));
    const int64_t ta = a.kv.paged_tokens(a.pools.kv().avail());
    CHECK(ta > 0);
    CHECK_EQ(z.kv.paged_tokens(z.pools.kv().avail()), KVManager::kUnboundedTokens);
    a.kv.cap_paged_tokens(ta);
    z.kv.cap_paged_tokens(ta);
    CHECK_OK(a.kv.carve(a.pools.kv()));
    CHECK_OK(z.kv.carve(z.pools.kv()));
    CHECK_EQ(z.kv.plan(G_FULL)->n_blocks, a.kv.plan(G_FULL)->n_blocks);
    CHECK(z.kv.plan(G_FULL)->base == nullptr);
    CHECK_EQ(z.kv.plan(G_FULL)->pool_bytes, (int64_t)0);
    CHECK(!z.kv.lendable(G_FULL));
    /* Its recurrent state is carved as on any rank. */
    CHECK(z.kv.plan(G_GDN)->base != nullptr);

    Fixture u;
    u.cfg.checkpoint_slots = 0;
    CHECK_OK(configure_pools(u.pools, u.cfg));
    CHECK_OK(u.kv.plan_groups(zg, u.cfg));
    CHECK_EQ(u.kv.carve(u.pools.kv()), RAD_E_STATE);
}

/* In an uneven world the cheaper rank pages the common token count in a smaller pool.
 * pool_mib_for_tokens inverts paged_tokens exactly at MiB granularity: its size reaches the count
 * and one MiB less does not; a rank whose only paged group is zero still covers its fixed costs. */
TEST(kv_pool_mib_for_tokens_is_the_inverse_of_paged_tokens) {
    constexpr int64_t MiB = 1ll << 20;
    Fixture dear, cheap, zero;
    for (Fixture* f : { &dear, &cheap, &zero }) { f->cfg.checkpoint_slots = 0; f->cfg.vram_kv_mib = 8; }
    std::vector<KVGroupInfo> cg = hybrid_groups();
    cg[G_FULL].decl.n_head_kv = 1;                  /* half the dear rank's page */
    std::vector<KVGroupInfo> zg = hybrid_groups();
    zg[G_FULL].zero = true;
    CHECK_OK(dear.kv.plan_groups(hybrid_groups(), dear.cfg));
    CHECK_OK(cheap.kv.plan_groups(cg, cheap.cfg));
    CHECK_OK(zero.kv.plan_groups(zg, zero.cfg));
    const int64_t T = dear.kv.paged_tokens(8 * MiB);
    CHECK(T > 0);
    const int64_t need = cheap.kv.pool_mib_for_tokens(T, 8);
    CHECK(need > 0 && need < 8);
    CHECK(cheap.kv.paged_tokens(need * MiB) >= T);
    CHECK(cheap.kv.paged_tokens((need - 1) * MiB) < T);
    /* The dear rank needs its whole pool for its own count. */
    CHECK_EQ(dear.kv.pool_mib_for_tokens(T, 8), (int64_t)8);
    /* Beyond what the largest size pages: -1, not a short pool. */
    CHECK_EQ(cheap.kv.pool_mib_for_tokens(T * 1000, 8), (int64_t)-1);
    /* All paged groups zero: only the fixed state limits the size. */
    const int64_t zn = zero.kv.pool_mib_for_tokens(T, 8);
    CHECK(zn >= 0 && zn * MiB >= zero.kv.fixed_bytes());
    CHECK(zn == 0 || (zn - 1) * MiB < zero.kv.fixed_bytes());
}

/* Checks the inverse on a group shape, and the carve it promises. The dear rank's `common_mib`
 * pool sets the token count; the cheap rank's inverse names the least whole MiB that pages it (one
 * less fails, or cannot hold the fixed costs); a pool of exactly that size, capped as configure_kv
 * caps every rank, carves each paged group to the dear rank's block count. Returns the MiB the
 * cheap rank needed. */
static int64_t check_inverse_and_carve(const std::vector<KVGroupInfo>& dear_g,
                                       const std::vector<KVGroupInfo>& cheap_g, int64_t common_mib) {
    constexpr int64_t MiB = 1ll << 20;
    Fixture dear, cheap;
    dear.cfg.checkpoint_slots = cheap.cfg.checkpoint_slots = 0;
    dear.cfg.vram_kv_mib = common_mib;
    CHECK_OK(dear.kv.plan_groups(dear_g, dear.cfg));
    CHECK_OK(cheap.kv.plan_groups(cheap_g, cheap.cfg));
    const int64_t T = dear.kv.paged_tokens(common_mib * MiB);
    CHECK(T > 0 && T != KVManager::kUnboundedTokens);
    const int64_t need = cheap.kv.pool_mib_for_tokens(T, common_mib);
    CHECK(need > 0 && need < common_mib);
    if (need <= 0) return need;
    CHECK(cheap.kv.paged_tokens(need * MiB) >= T);
    CHECK((need - 1) * MiB < cheap.kv.fixed_bytes() || cheap.kv.paged_tokens((need - 1) * MiB) < T);

    cheap.cfg.vram_kv_mib = need;
    CHECK_OK(configure_pools(dear.pools, dear.cfg));
    CHECK_OK(configure_pools(cheap.pools, cheap.cfg));
    CHECK_EQ(cheap.pools.kv().avail(), need * MiB);
    CHECK_EQ(dear.kv.paged_tokens(dear.pools.kv().avail()), T);
    dear.kv.cap_paged_tokens(T);
    cheap.kv.cap_paged_tokens(T);
    CHECK_OK(dear.kv.carve(dear.pools.kv()));
    CHECK_OK(cheap.kv.carve(cheap.pools.kv()));
    for (int32_t g = 1; g < (int32_t)dear_g.size(); ++g) {
        if (!dear.kv.plan(g)->paged()) continue;
        CHECK_EQ(cheap.kv.plan(g)->n_blocks, dear.kv.plan(g)->n_blocks);
        CHECK(dear.kv.plan(g)->n_blocks > 0);
    }
    return need;
}

static KVGroupInfo paged_group(const char* name, int kind, int64_t heads, int64_t block,
                               std::vector<int32_t> layers, int64_t window = 0) {
    KVGroupInfo k;
    k.name = name;
    k.decl.kind = kind;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = heads;
    k.decl.head_dim = 8;
    k.decl.window = window;
    k.block_size = block;
    k.layers = std::move(layers);
    return k;
}

/* The streamed group is the widest full group with a declared streamed path, never a dense-read
 * one. GLM-5.3's two full groups show the difference: kv_latent (49 152 B a block) streams;
 * kv_dsa_index (1 632 B) is read over the whole history every step and stays resident. */
TEST(kv_stream_group_picks_a_wired_full_group_not_merely_the_widest) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    const std::vector<int32_t> full_layers = { 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45 };
    KVGroupInfo latent = paged_group("kv_latent", RAD_KV_FULL, 1, 4, full_layers);
    latent.decl.dtype = RAD_BF16;
    latent.decl.head_dim = 256;
    latent.bytes_per_block = 49152;              /* 12 layers x 4 tokens x 256 x bf16 */
    KVGroupInfo index = paged_group("kv_dsa_index", RAD_KV_FULL, 1, 4, full_layers);
    index.decl.dtype = RAD_U8;
    index.decl.head_dim = 17;
    index.bytes_per_block = 1632;                /* 12 layers x 4 tokens x 17 x u8 */
    g.push_back(latent);
    g.push_back(index);

    auto check_pick = [&](const Program& p, int32_t want) {
        Program prog = p;
        prog.kv_groups = g;                      /* index-aligned, as Engine's program is */
        CHECK_EQ(kv_stream_group(g, &prog), want);
        /* The cap probe and the carve must agree, so the plans overload takes the same rule. */
        Fixture f;
        f.cfg.checkpoint_slots = 0;
        f.cfg.max_ctx = 4096;
        f.cfg.max_seqs = 2;
        CHECK_OK(f.kv.plan_groups(g, f.cfg));
        CHECK_EQ(kv_stream_group(f.kv, &prog), want);
    };

    /* Wired latent, narrower: it is picked. */
    check_pick(stream_program(g[1], 513), 1);
    /* The index group made wider is still not picked: it is not wired. This is the width
     * coincidence the rule removes. */
    g[2].bytes_per_block = 65536;
    check_pick(stream_program(g[1], 513), 1);
    g[2].bytes_per_block = 1632;
    /* A resolve that states the index group's pitch (136 B a layer) is the only wired path, so
     * the narrower group is the one that streams. */
    check_pick(stream_program(g[2], 513), 2);
    /* A dense-read group is never a candidate, even when its own path is declared. Nothing else
     * is wired here, so nothing streams. */
    g[2].dense_read = true;
    check_pick(stream_program(g[2], 513), -1);
    g[2].dense_read = false;
}

/* A windowed group is capped at its window on both ranks, and the cap shifts the freed bytes to
 * the other groups, a kink in paged_split the inverse must cross exactly. The dear rank's count is
 * past the window, so the window group holds max_seqs x window_blocks on both ranks. */
TEST(kv_pool_mib_for_tokens_crosses_a_windowed_groups_cap) {
    std::vector<KVGroupInfo> dear = hybrid_groups();
    dear.push_back(paged_group("swa", RAD_KV_WINDOW, 4, 16, { 6, 7, 8, 9 }, 2048));
    std::vector<KVGroupInfo> cheap = dear;
    cheap[G_FULL].decl.n_head_kv = 1;
    check_inverse_and_carve(dear, cheap, 32);
    /* With the window group cheaper instead, all the savings are in the window. */
    cheap = dear;
    cheap[3].decl.n_head_kv = 1;
    check_inverse_and_carve(dear, cheap, 32);
}

/* Two paged groups at different block sizes: one token count, two block counts, each matching
 * the dear rank's. */
TEST(kv_pool_mib_for_tokens_holds_across_block_sizes) {
    std::vector<KVGroupInfo> dear = hybrid_groups();
    dear.push_back(paged_group("full32", RAD_KV_FULL, 2, 32, { 6, 7, 8 }));
    std::vector<KVGroupInfo> cheap = dear;
    cheap[G_FULL].decl.n_head_kv = 1;
    check_inverse_and_carve(dear, cheap, 8);
    cheap = dear;
    cheap[3].decl.n_head_kv = 1;
    check_inverse_and_carve(dear, cheap, 8);
}

/* A zero attention group beside another paged group costs this rank nothing and takes no part in
 * the split, so the other group alone sets the pool; the zero group still holds the shared
 * table's block count after the carve. */
TEST(kv_pool_mib_for_tokens_with_a_zero_group_beside_a_paged_one) {
    std::vector<KVGroupInfo> dear = hybrid_groups();
    dear.push_back(paged_group("full32", RAD_KV_FULL, 2, 32, { 6, 7, 8 }));
    std::vector<KVGroupInfo> cheap = dear;
    cheap[G_FULL].zero = true;
    const int64_t need = check_inverse_and_carve(dear, cheap, 8);
    /* Cheaper than a rank that keeps the group at half its heads. */
    std::vector<KVGroupInfo> kept = dear;
    kept[G_FULL].decl.n_head_kv = 1;
    CHECK(need < check_inverse_and_carve(dear, kept, 8));
}

/* The re-split moves the cheaper rank's KV savings into its weights pool, never past all weights
 * resident and never below the common pool; the remainder is reported unusable, not allocated. A
 * need not below the common pool (the even case) changes nothing. */
TEST(an_uneven_worlds_cheaper_rank_gives_its_saved_kv_to_its_weights_up_to_all_resident) {
    constexpr int64_t MiB = 1ll << 20;
    RankSplit s = vram_budget_rank_split(1000, 800, 200, 1300 * MiB);
    CHECK(s.resplit);
    CHECK_EQ(s.kv_mib, (int64_t)200);
    CHECK_EQ(s.weights_mib, (int64_t)1300);
    CHECK_EQ(s.unusable_mib, (int64_t)300);
    CHECK(s.weights_mib + s.kv_mib <= 1000 + 800);
    s = vram_budget_rank_split(1000, 800, 200, 5000 * MiB);     /* room for all of it */
    CHECK_EQ(s.weights_mib, (int64_t)1600);
    CHECK_EQ(s.unusable_mib, (int64_t)0);
    s = vram_budget_rank_split(1000, 800, 200, 500 * MiB);      /* already all resident */
    CHECK_EQ(s.weights_mib, (int64_t)1000);
    CHECK_EQ(s.kv_mib, (int64_t)200);
    CHECK_EQ(s.unusable_mib, (int64_t)600);
    s = vram_budget_rank_split(1000, 800, 800, 5000 * MiB);     /* same cost: untouched */
    CHECK(!s.resplit);
    CHECK_EQ(s.weights_mib, (int64_t)1000);
    CHECK_EQ(s.kv_mib, (int64_t)800);
    s = vram_budget_rank_split(1000, 800, -1, 5000 * MiB);      /* no answer: untouched */
    CHECK(!s.resplit);
    CHECK_EQ(s.kv_mib, (int64_t)800);
}

TEST(a_streamed_group_is_refused_at_carve_without_its_host_copy) {
    /* A streamed plan is refused unless something holds the blocks the pool does not back: a store
     * for block 40 of a four-block pool would write past the layer stride. Each variant removes
     * one of the three pieces the carve requires. */
    for (int variant = 0; variant < 3; ++variant) {
        Fixture f;
        f.cfg.checkpoint_slots = 0;
        f.cfg.max_ctx = 1024;                  /* the context the group will address, in host RAM */
        CHECK_OK(configure_pools(f.pools, f.cfg));
        f.kv.cap_resident_tokens(G_FULL, 64);
        CHECK_OK(f.kv.plan_groups(hybrid_groups(), f.cfg));
        /* How many blocks the group has is decided at carve; the host copy has to be there first,
         * so size it from the context the carve will compute: every sequence at full length. */
        const int64_t ctx_blocks = f.cfg.max_seqs * (f.cfg.max_ctx / 16);
        const int64_t page = f.kv.plan(G_FULL)->bytes_per_block;
        const int64_t need = ctx_blocks * page;
        std::vector<char> host((size_t)need, 0);
        std::vector<int32_t> res((size_t)ctx_blocks, -1);

        KVManager::StreamMap m;
        m.host       = variant == 0 ? nullptr : (uint8_t*)host.data();  /* 0: no copy at all */
        m.host_bytes = variant == 1 ? need - 1 : need;         /* 1: a copy too small */
        m.res        = variant == 2 ? nullptr : res.data();    /* 2: no map */
        f.kv.set_stream_map(G_FULL, m);
        CHECK_EQ(f.kv.carve(f.pools.kv()), RAD_E_INVAL);
    }
}

TEST(a_streamed_group_addresses_its_context_but_backs_only_the_resident_window) {
    /* A streamed group allocates block ids over its whole context while the pool backs only
     * --kv-resident of them. This pins the places that multiply by one of the two numbers: the
     * carve, the layer stride, the kernel's operand shape and the loan. A mistake there writes one
     * layer's block into the next. The split is decided at carve, which knows the budget. */
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    f.cfg.max_ctx = 1024;                                  /* the context the group addresses */
    CHECK_OK(configure_pools(f.pools, f.cfg));
    f.kv.cap_resident_tokens(G_FULL, 64);                  /* four blocks of 16 tokens */
    CHECK_OK(f.kv.plan_groups(hybrid_groups(), f.cfg));

    /* Both numbers come from the carve: the context from --max-model-len and --max-num-seqs, the
     * window from --kv-resident. The host copy must already cover the context. */
    const int64_t ctx_blocks = f.cfg.max_seqs * (f.cfg.max_ctx / 16);
    const int64_t page = f.kv.plan(G_FULL)->bytes_per_block;
    CHECK(ctx_blocks > 4);                                 /* the pool holds more than the window */
    std::vector<char> host((size_t)(ctx_blocks * page), 0);
    std::vector<int32_t> res((size_t)ctx_blocks, -1);
    KVManager::StreamMap sm_in;
    sm_in.host       = (uint8_t*)host.data();
    sm_in.res        = res.data();
    sm_in.host_bytes = (int64_t)host.size();
    f.kv.set_stream_map(G_FULL, sm_in);
    const KVManager::StreamMap* sm = f.kv.stream_map(G_FULL);
    CHECK(sm != nullptr && sm->host == (uint8_t*)host.data() && sm->res == res.data() &&
          sm->host_bytes >= ctx_blocks * page);
    CHECK_OK(f.kv.carve(f.pools.kv()));

    const KVGroupPlan* p = f.kv.plan(G_FULL);
    CHECK_EQ(p->n_blocks, ctx_blocks);
    CHECK_EQ(p->n_slots, 4);
    CHECK(p->streamed());
    CHECK_EQ(p->pool_blocks(), 4);
    CHECK_EQ(p->layer_stride(), 4 * p->layer_bytes_per_block());
    /* The carve is one reservation of max(ring, stage): the ring's four blocks of every layer, or
     * the whole context one layer deep. Here the context's one layer is the larger, so the pool is
     * a stage and the ring fits inside it (docs/KV-STREAMING.md §4.1). */
    CHECK_EQ(f.kv.pool_bytes_carved(G_FULL),
             std::max(4 * page, ctx_blocks * p->layer_bytes_per_block()));
    CHECK_EQ(rad::kv_ceiling_bytes(*p, ctx_blocks, 4), ctx_blocks * p->layer_bytes_per_block());
    /* The startup report must not read as a pool that holds the whole context. */
    CHECK(f.kv.report().find("streaming:") != std::string::npos);

    /* Not lendable: the window is what a residency map reads, and a block lent to the expert slab
     * is one the attention has nowhere to put. */
    CHECK(!f.kv.lendable(G_FULL));
    /* The linear group next to it is untouched. */
    CHECK_EQ(f.kv.plan(G_GDN)->n_slots, 0);
    CHECK(!f.kv.plan(G_GDN)->streamed());

    /* Block ids still run over the context. A sequence longer than the window asks for more ids
     * than the pool backs and gets them, because its table names logical blocks -- which is exactly
     * why a streamed plan may not be carved without somewhere for them to live. */
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 8));                      /* eight blocks, twice the window */
    const std::vector<int32_t>& bt = f.kv.block_table(1, G_FULL);
    CHECK(bt.size() >= 8);
    int32_t hi = 0;
    for (int32_t b : bt) hi = std::max(hi, b);
    CHECK(hi >= 4);                                        /* an id past the window */

    /* And a server that does not stream carves exactly what it did before the split: nothing
     * moves for a group with no cap_resident_tokens(). */
    Fixture plain;
    plain.cfg.checkpoint_slots = 0;
    CHECK_OK(plain.build());
    const KVGroupPlan* q = plain.kv.plan(G_FULL);
    CHECK(q->n_blocks > 0);
    CHECK_EQ(q->n_slots, q->n_blocks);
    CHECK(!q->streamed());
    CHECK_EQ(plain.kv.pool_bytes_carved(G_FULL), q->n_blocks * q->bytes_per_block);
}

TEST(a_resident_window_that_does_not_fit_vram_kv_is_refused_by_its_own_flag) {
    /* A window wider than the pool is refused by name; left to the split, it would leave the other
     * groups zero blocks and point the operator at --host-pool-mib. */
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    f.cfg.max_ctx = 1024;
    CHECK_OK(configure_pools(f.pools, f.cfg));
    f.kv.cap_resident_tokens(G_FULL, 100000);              /* 6250 blocks against a 1 MiB pool */
    CHECK_OK(f.kv.plan_groups(hybrid_groups(), f.cfg));
    /* The host copy is the right size for the context; the window is what does not fit. */
    const int64_t ctx_blocks = f.cfg.max_seqs * (f.cfg.max_ctx / 16);
    std::vector<char> host((size_t)(ctx_blocks * f.kv.plan(G_FULL)->bytes_per_block), 0);
    std::vector<int32_t> res((size_t)ctx_blocks, -1);
    KVManager::StreamMap sm_in;
    sm_in.host       = (uint8_t*)host.data();
    sm_in.res        = res.data();
    sm_in.host_bytes = (int64_t)host.size();
    f.kv.set_stream_map(G_FULL, sm_in);
    CHECK_EQ(f.kv.carve(f.pools.kv()), RAD_E_NOMEM);
}

TEST(a_streamed_window_group_is_refused_by_name) {
    /* A window group's ids are its window's, not its context's, so it cannot stream. */
    auto groups = hybrid_groups();
    groups[G_FULL].decl.kind = RAD_KV_WINDOW;
    groups[G_FULL].decl.window = 128;                      /* window_blocks = 128/16 + 1 */
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    CHECK_OK(configure_pools(f.pools, f.cfg));
    f.kv.cap_resident_tokens(G_FULL, 64);
    CHECK_OK(f.kv.plan_groups(groups, f.cfg));
    CHECK_EQ(f.kv.carve(f.pools.kv()), RAD_E_INVAL);
}

/* The residency maps are device memory carved from the KV pool, which on a card is VRAM a host
 * load faults on; so a check copies them back on a stream first, as the engine reads `ctl`. Counts:
 * `res` is `n_layers * blocks`, `slot_block` `n_layers * slots`, `slot_state` twice that. */
struct StreamMapsHost {
    std::vector<int32_t> res, slot_block, slot_state, ctl;
};

static int read_stream_maps(const KVManager::StreamMap& m, const KVGroupPlan& p, RadStream st,
                            StreamMapsHost* out) {
    const int64_t layers = p.n_layers > 0 ? p.n_layers : 1;
    out->res.assign((size_t)(layers * p.n_blocks), 0);
    out->slot_block.assign((size_t)(layers * m.slots), 0);
    out->slot_state.assign((size_t)(layers * m.slots * 2), 0);
    out->ctl.assign(16, 0);
    auto copy = [&](std::vector<int32_t>& dst, const int32_t* src) {
        return rad_memcpy_async(dst.data(), src,
                                (int64_t)dst.size() * (int64_t)sizeof(int32_t), st);
    };
    int rc = copy(out->res, m.res);
    if (rc == RAD_OK) rc = copy(out->slot_block, m.slot_block);
    if (rc == RAD_OK) rc = copy(out->slot_state, m.slot_state);
    if (rc == RAD_OK) rc = copy(out->ctl, m.ctl);
    if (rc == RAD_OK) rc = rad_stream_sync(st);
    return rc;
}

TEST(a_streamed_plan_allocates_reports_and_frees_its_mirror) {
    /* The host mirror and the ring's maps, end to end: what stream_cost() promises is what
     * alloc_stream() takes, the mirror is zeroed (a block nothing wrote yet must read as zeros, not
     * as a page's leftovers), the maps come out of the KV pool and hold the reset state a resolve
     * launch expects, the report shows both tiers, and free_stream() releases the mirror and is
     * safe to call twice. */
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    f.cfg.max_ctx = 1024;
    CHECK_OK(configure_pools(f.pools, f.cfg));
    f.kv.cap_resident_tokens(G_FULL, 64);                  /* four blocks of 16 tokens */
    CHECK_OK(f.kv.plan_groups(hybrid_groups(), f.cfg));

    const KVManager::StreamCost cost = f.kv.stream_cost();
    CHECK_EQ(cost.groups, (int64_t)1);
    CHECK_EQ(cost.host_bytes, f.cfg.max_seqs * (f.cfg.max_ctx / 16) *
                              f.kv.plan(G_FULL)->bytes_per_block);
    CHECK(cost.map_bytes > 0);

    const int64_t pool_used = f.pools.kv().used();
    CHECK_OK(f.kv.alloc_stream(rad_malloc_allocator(), f.pools.kv()));
    /* Every map byte is a reservation in the KV pool, so the paged split sees it. */
    CHECK_EQ(f.pools.kv().used() - pool_used, cost.map_bytes);

    const KVManager::StreamMap* m = f.kv.stream_map(G_FULL);
    REQUIRE(m != nullptr);
    CHECK(m->host != nullptr && m->res != nullptr && m->ctl != nullptr);
    CHECK_EQ(m->host_bytes, cost.host_bytes);
    CHECK_EQ(m->block_bytes, f.kv.plan(G_FULL)->layer_bytes_per_block());
    CHECK_EQ(m->host_layer_stride, (f.cfg.max_seqs * (f.cfg.max_ctx / 16)) * m->block_bytes);
    /* Zeroed at allocation: a miss copies these bytes to the ring before attention reads them. */
    int64_t nonzero = 0;
    for (int64_t i = 0; i < m->host_bytes; ++i) nonzero += m->host[i] != 0;
    CHECK_EQ(nonzero, (int64_t)0);

    CHECK_OK(f.kv.carve(f.pools.kv()));
    RadStream st = nullptr;
    REQUIRE_EQ(rad_stream_create(&st, 0), RAD_OK);
    CHECK_OK(f.kv.init_stream(st));
    CHECK_OK(rad_stream_sync(st));

    /* The reset the reference contract states: both maps -1, state and control zero, in every
     * layer's slice (the maps are device memory; read_stream_maps copies them back). */
    const KVGroupPlan* p = f.kv.plan(G_FULL);
    StreamMapsHost got;
    CHECK_OK(read_stream_maps(*m, *p, st, &got));
    rad_stream_destroy(st);
    int64_t wrong = 0;
    for (int64_t b = 0; b < (int64_t)got.res.size(); ++b) wrong += got.res[(size_t)b] != -1;
    for (int64_t s = 0; s < (int64_t)got.slot_block.size(); ++s)
        wrong += got.slot_block[(size_t)s] != -1;
    for (int64_t i = 0; i < (int64_t)got.slot_state.size(); ++i)
        wrong += got.slot_state[(size_t)i] != 0;
    CHECK_EQ(wrong, (int64_t)0);
    for (int i = 0; i < 16; ++i) CHECK_EQ(got.ctl[(size_t)i], 0);
    CHECK_EQ(m->slots, p->pool_blocks());
    CHECK(f.kv.stream_view(G_FULL) == m);

    const KVManager::StreamStats stats = f.kv.stream_stats();
    CHECK_EQ(stats.resident_blocks, p->pool_blocks());

    /* Both tiers in one line, and the ring placed in the pool it was carved from. */
    CHECK(f.kv.report().find("streaming:") != std::string::npos);
    CHECK(f.kv.report().find("vram_kv") != std::string::npos);
    CHECK(f.kv.report().find("pinned host memory") != std::string::npos);

    f.kv.free_stream(rad_malloc_allocator());
    CHECK(f.kv.stream_view(G_FULL) == nullptr);
    CHECK(f.kv.stream_map(G_FULL) == nullptr || f.kv.stream_map(G_FULL)->host == nullptr);
    f.kv.free_stream(rad_malloc_allocator());              /* teardown after a partial bring-up */
}

TEST(a_streamed_plans_maps_are_device_memory_read_back_on_its_stream) {
    /* The maps come out of the KV pool, which on a card is VRAM: only the mirror may be read with a
     * host load. Their reset contents are checked by
     * `a_streamed_groups_maps_are_charged_before_the_pool_is_sized`. */
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    f.cfg.max_ctx = 1024;
    CHECK_OK(configure_pools(f.pools, f.cfg));
    f.kv.cap_resident_tokens(G_FULL, 64);
    CHECK_OK(f.kv.plan_groups(hybrid_groups(), f.cfg));
    CHECK_OK(f.kv.alloc_stream(rad_malloc_allocator(), f.pools.kv()));
    CHECK_OK(f.kv.carve(f.pools.kv()));

    const KVManager::StreamMap* m = f.kv.stream_map(G_FULL);
    REQUIRE(m != nullptr);
    /* Every map array is a reservation of this pool, and the pool is what the card backs: the
     * mirror is the only thing here the host may read with a load. */
    const char* lo = (const char*)f.pools.kv().base();
    const char* hi = lo + f.pools.kv().bytes();
    CHECK((const char*)m->res >= lo && (const char*)m->res < hi);
    CHECK(m->host < (const uint8_t*)lo || m->host >= (const uint8_t*)hi);
    if (!device_is_host()) CHECK(rad_dev_host_ptr(m->res) == nullptr);

    f.kv.free_stream(rad_malloc_allocator());
}

TEST(a_plan_that_does_not_stream_allocates_and_reports_nothing) {
    /* The inert case every existing server runs: no cap_resident_tokens(), so stream_cost() is
     * zero, alloc_stream() takes nothing, the ABI view is null (every group fully resident) and
     * the report has no streaming line. */
    Fixture f;
    CHECK_OK(f.build());
    const KVManager::StreamCost cost = f.kv.stream_cost();
    CHECK_EQ(cost.groups, (int64_t)0);
    CHECK_EQ(cost.host_bytes, (int64_t)0);
    CHECK_EQ(cost.map_bytes, (int64_t)0);

    const int64_t pool_used = f.pools.kv().used();
    CHECK_OK(f.kv.alloc_stream(rad_malloc_allocator(), f.pools.kv()));
    CHECK_EQ(f.pools.kv().used(), pool_used);
    CHECK(f.kv.stream_view(G_FULL) == nullptr);
    CHECK(f.kv.stream_map(G_FULL) == nullptr);

    const KVManager::StreamStats stats = f.kv.stream_stats();
    CHECK_EQ(stats.resident_blocks, (int64_t)0);

    CHECK(f.kv.report().find("streaming:") == std::string::npos);
    f.kv.free_stream(rad_malloc_allocator());
    CHECK_EQ(f.pools.kv().used(), pool_used);
}

TEST(a_ring_that_covers_the_context_streams_nothing_and_the_engine_pays_nothing) {
    /* --kv-resident absent or 0 streams nothing, and neither does a ring that covers the group's
     * context: such a group keeps the resident path, with no mirror, maps or stage and a null
     * RadKVStream, so no kv_resolve can be issued (docs/KV-STREAMING.md §3.1). */
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    f.cfg.max_ctx = 1024;
    f.cfg.max_seqs = 2;                                    /* context: 2 sequences x 1024 tokens */
    std::vector<KVGroupInfo> groups = hybrid_groups();
    groups[G_FULL].bytes_per_block = 2048;                 /* 16 tokens x 128 B, as declare sets it */
    const int64_t ctx_blocks = 2 * (1024 / 16);            /* 128 blocks of 16 tokens */
    /* A ring that covers the context. The group is bf16 here on purpose: a configuration that fits
     * is never refused for a cache the streamed store has no row for, because nothing streams. */
    f.cfg.kv_resident = 1024 * 2;
    KVStreamSetup s;
    CHECK_OK(kvstream_configure(f.cfg, 1024, groups, stream_program(groups[G_FULL], 513), &s));
    CHECK_EQ(s.group, -1);
    CHECK_EQ(s.asked_tokens, (int64_t)2048);
    CHECK_EQ(s.ctx_blocks, ctx_blocks);
    CHECK_EQ(s.ring_blocks, ctx_blocks);
    CHECK_EQ(s.stage_blocks, (int64_t)0);                  /* no stage view is declared */
    CHECK_EQ(s.group_name, std::string("full"));

    /* And the engine applies that decision: configure_kv caps only a group that streams, so the
     * manager is never told about a window, alloc_stream() takes nothing and the carve is the
     * ordinary one-view pool. */
    CHECK_OK(configure_pools(f.pools, f.cfg));
    CHECK_OK(f.kv.plan_groups(groups, f.cfg));
    const KVManager::StreamCost cost = f.kv.stream_cost();
    CHECK_EQ(cost.groups, (int64_t)0);
    CHECK_EQ(cost.host_bytes, (int64_t)0);
    CHECK_EQ(cost.map_bytes, (int64_t)0);
    const int64_t pool_used = f.pools.kv().used();
    CHECK_OK(f.kv.alloc_stream(rad_malloc_allocator(), f.pools.kv()));
    CHECK_EQ(f.pools.kv().used(), pool_used);              /* no mirror and no maps were taken */
    CHECK_OK(f.kv.carve(f.pools.kv()));
    const KVGroupPlan* p = f.kv.plan(G_FULL);
    CHECK_EQ(p->n_slots, p->n_blocks);                     /* every block has its own slot */
    CHECK(!p->streamed());
    CHECK_EQ(p->depth, (int64_t)0);                        /* the arch sees no stage view */
    CHECK_EQ(f.kv.pool_bytes_carved(G_FULL), p->n_blocks * p->bytes_per_block);
    CHECK(f.kv.stream_view(G_FULL) == nullptr);            /* so no kv_resolve can be issued */
    CHECK(f.kv.stream_map(G_FULL) == nullptr);
    CHECK(f.kv.report().find("streaming:") == std::string::npos);
    f.kv.free_stream(rad_malloc_allocator());

    /* Absent (the parser's default) and 0 are the same: nothing is asked for, nothing is decided,
     * nothing is capped. */
    {
        Fixture g;
        g.cfg.checkpoint_slots = 0;
        g.cfg.max_ctx = 1024;
        g.cfg.max_seqs = 2;
        g.cfg.kv_resident = 0;
        KVStreamSetup o;
        CHECK_OK(kvstream_configure(g.cfg, 1024, groups, stream_program(groups[G_FULL], 513), &o));
        CHECK_EQ(o.group, -1);
        CHECK_EQ(o.asked_tokens, (int64_t)0);
        CHECK_EQ(o.ring_blocks, (int64_t)0);
        CHECK_EQ(o.stage_blocks, (int64_t)0);
        CHECK_OK(g.build_groups(groups));
        CHECK_EQ(g.kv.stream_cost().groups, (int64_t)0);
        CHECK(g.kv.stream_view(G_FULL) == nullptr);
        CHECK(!g.kv.plan(G_FULL)->streamed());
        CHECK_EQ(g.kv.plan(G_FULL)->depth, (int64_t)0);
    }
}

/* The prefix cache's host tier is turned off when --kv-resident really streams, and only then.
 * The three cases that must leave it on are the ones an operator hits without meaning to: the flag
 * absent or 0, a ring that covers the group's context (nothing streams), and a flag on a model the
 * tier does not stream. `cfg.kv_streams` is the one answer to all three (kvstream_cap_probe). */
TEST(the_prefix_cache_host_tier_is_off_only_when_streaming_is_really_on) {
    std::vector<KVGroupInfo> groups = hybrid_groups();
    groups[G_FULL].bytes_per_block = 2048;                 /* 16 tokens x 128 B, as declare sets it */
    /* A pitch of 513 blocks is this fixture's resolve; the ceiling is max_seqs x (1 + n_spec) x
     * pitch, so a ring under a 128-block context needs a small pitch: 8 x 2 = 16 blocks. */
    const Program wide = stream_program(groups[G_FULL], 513);
    const Program narrow = stream_program(groups[G_FULL], 8);

    struct Case {
        const char* what;
        int64_t     resident;      /* --kv-resident */
        bool        streams;
        int64_t     tier_after;    /* --prefix-cache-mib once bring-up has had its say */
    };
    const Case cases[] = {
        { "the flag absent",           0,          false, 2048 },
        { "a ring covering the ctx",   2 * 65536,  false, 2048 },
        { "a ring under the ctx",      256,        true,  0    },
    };
    for (const Case& c : cases) {
        Fixture f;
        f.cfg.checkpoint_slots = 0;
        f.cfg.max_ctx = 65536;                             /* clears the 32 768-token floor */
        f.cfg.max_seqs = 2;
        f.cfg.kv_resident = c.resident;
        f.cfg.prefix_cache_mib = 2048;
        CHECK_OK(f.kv.plan_groups(groups, f.cfg));
        CHECK_OK(kvstream_cap_probe(f.kv, f.cfg, c.streams ? narrow : wide, f.cfg.max_ctx));
        CHECK_EQ(f.cfg.kv_streams, c.streams);
        const int64_t off = kvstream_prefix_host_off(f.cfg);
        CHECK_EQ(f.cfg.prefix_cache_mib, c.tier_after);
        CHECK_EQ(off, c.streams ? 2048ll : 0ll);
        CHECK_EQ(kvstream_prefix_host_off(f.cfg), 0ll);    /* idempotent: a second rank says nothing */
        f.kv.free_stream(rad_malloc_allocator());
    }
}

TEST(a_decode_pass_wider_than_a_verify_stages_instead_of_resolving_the_ring) {
    /* The serial drafter's history pass is RAD_PHASE_DECODE, but after a prefill chunk it carries
     * the chunk's tokens of every sequence: more rows a sequence than a decode step can select for
     * (1 + n_spec). Its selection is the union of the histories it spans, not one row a selection,
     * so it reads the one-layer stage -- identity-addressed over the whole context -- and cannot
     * overflow the ring. A verify or a history pass over the verified window stays on the ring:
     * routing those to the stage would stage every decode step. */
    using rad::KVView;
    CHECK(kv_pass_view(RAD_PHASE_PREFILL, 2048, 4, 3) == KVView::Stage);
    CHECK(kv_pass_view(RAD_PHASE_MIXED, 900, 3, 3) == KVView::Stage);
    /* The ring's passes: a verify step (4 rows a sequence with n_spec 3), one row a sequence, and
     * the history pass that redoes the verified positions. */
    CHECK(kv_pass_view(RAD_PHASE_DECODE, 16, 4, 3) == KVView::Ring);
    CHECK(kv_pass_view(RAD_PHASE_DECODE, 4, 4, 3) == KVView::Ring);
    CHECK(kv_pass_view(RAD_PHASE_DECODE, 12, 3, 3) == KVView::Ring);
    CHECK(kv_pass_view(RAD_PHASE_DECODE, 4, 4, 0) == KVView::Ring);
    /* The history pass over a prefill chunk, and every prefill-shaped decode pass. */
    CHECK(kv_pass_view(RAD_PHASE_DECODE, 2048, 4, 3) == KVView::Stage);
    CHECK(kv_pass_view(RAD_PHASE_DECODE, 2039, 3, 3) == KVView::Stage);
    CHECK(kv_pass_view(RAD_PHASE_DECODE, 2048, 4, 0) == KVView::Stage);
}

TEST(a_ring_below_one_decode_step_s_selection_is_raised_to_the_ceiling) {
    /* A ring under one decode step's selection, max_seqs x (1 + n_spec) x sel_pitch blocks, would
     * serve a table that no longer addresses the pool, so the flag is raised to it with one
     * warning and the carve takes the raised ring. */
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    /* The context must clear the shipped floor (kStreamFloorTokens, 32 768 tokens) for the step
     * ceiling to be the binding bound: 2 x 65 536 tokens = 8192 blocks of 16. */
    f.cfg.max_ctx = 65536;
    f.cfg.max_seqs = 2;
    f.cfg.n_spec = 0;
    f.cfg.vram_kv_mib = 64;
    std::vector<KVGroupInfo> groups = hybrid_groups();
    groups[G_FULL].decl.dtype = RAD_F8E4M3;               /* the streamed store's one row */
    groups[G_FULL].bytes_per_block = 2048;                 /* as declare sets it */
    const int64_t sel_pitch = 2048;                        /* a synthetic indexer's topk + 1 */
    const int64_t ctx_blocks = 2 * (65536 / 16);           /* 8192 blocks of 16 tokens */
    const Program prog = stream_program(groups[G_FULL], sel_pitch);
    CHECK_OK(configure_pools(f.pools, f.cfg));
    CHECK_OK(f.kv.plan_groups(groups, f.cfg));
    /* The ceiling is 2 sequences x 1 row x sel_pitch = 4096 blocks = 65 536 tokens, above the
     * shipped floor of 2048 blocks (32 768 tokens) and under the 8192-block context. */
    f.cfg.kv_resident = 128;                               /* 8 blocks: far under it */
    KVStreamSetup s;
    const std::string log = stderr_of([&] {
        /* The budget probe prices the ring the carve will take, both of them raised. */
        CHECK_OK(kvstream_cap_probe(f.kv, f.cfg, prog, 65536));
        CHECK_OK(kvstream_configure(f.cfg, 65536, groups, prog, &s));
    });
    CHECK(has(log, "--kv-resident 128 raised to 65536 tokens"));
    CHECK(has(log, "one decode step can select 4096 blocks of 16 tokens"));
    CHECK(has(log, "(--max-num-seqs 2 x 1 rows x 2048)"));
    /* Raised, not refused: the flag is as the operator left it and the plan is the ceiling. */
    CHECK_EQ(f.cfg.kv_resident, (int64_t)128);
    CHECK_EQ(s.group, (int32_t)G_FULL);
    CHECK_EQ(s.asked_tokens, (int64_t)128);
    CHECK_EQ(s.raised_tokens, (int64_t)65536);
    CHECK_EQ(s.ring_blocks, (int64_t)4096);
    CHECK_EQ(s.ring_tokens, (int64_t)65536);
    CHECK_EQ(s.ctx_blocks, ctx_blocks);
    CHECK_EQ(s.stage_blocks, ctx_blocks);                  /* one layer, the whole context */

    /* The carve takes the ceiling, not the flag's 8 blocks: cap_resident_tokens is handed the
     * raised ring, and the residency map, the pool and the resolve's slot ceiling are that depth. */
    CHECK_OK(f.kv.alloc_stream(rad_malloc_allocator(), f.pools.kv()));
    CHECK_OK(f.kv.carve(f.pools.kv()));
    const KVGroupPlan* p = f.kv.plan(G_FULL);
    CHECK(p->streamed());
    CHECK_EQ(p->n_blocks, ctx_blocks);
    CHECK_EQ(p->n_slots, (int64_t)4096);                   /* the ceiling */
    CHECK_EQ(p->n_slots, s.ring_blocks);
    CHECK_EQ(f.kv.stream_map(G_FULL)->slots, (int64_t)4096);
    CHECK_EQ(f.kv.stream_stats().resident_blocks, (int64_t)4096);

    /* And the startup report says so in the ring's own line, not only in the warning: the ring is
     * no longer what the operator typed, and the report is what people read afterwards. */
    KVStreamHostUse use;
    kvstream_host_add(&use, f.kv);
    const std::string rep = stderr_of([&] { kvstream_report(s, use, 1); });
    CHECK(has(rep, "--kv-resident 128 became 65536 tokens"));
    CHECK(has(rep, "(8 -> 4096 blocks of 16 tokens)"));
    CHECK(has(rep, "ring 4096 blocks"));                     /* the staging line quotes the ring */
    f.kv.free_stream(rad_malloc_allocator());

    /* Under speculation a verify step runs the drafts beside the accepted position, so the rows --
     * and the ceiling -- scale by 1 + n_spec: 2 x 4 x 512 = 4096 blocks = 65 536 tokens. */
    {
        const Program spec = stream_program(groups[G_FULL], 512);
        Fixture g;
        g.cfg.checkpoint_slots = 0;
        g.cfg.max_ctx = 65536;
        g.cfg.max_seqs = 2;
        g.cfg.n_spec = 3;
        g.cfg.vram_kv_mib = 64;
        g.cfg.kv_resident = 512;                           /* 32 blocks: far under the ceiling */
        KVStreamSetup o;
        const std::string l = stderr_of([&] {
            CHECK_OK(kvstream_configure(g.cfg, 65536, groups, spec, &o));
        });
        CHECK(has(l, "(--max-num-seqs 2 x 4 rows x 512)"));
        CHECK_EQ(o.group, (int32_t)G_FULL);
        CHECK_EQ(o.ring_blocks, (int64_t)4096);
        CHECK_EQ(o.raised_tokens, (int64_t)65536);
    }
    /* At or above the ceiling nothing is raised and the ring is the flag's own number. */
    {
        Fixture g;
        g.cfg.checkpoint_slots = 0;
        g.cfg.max_ctx = 65536;
        g.cfg.max_seqs = 2;
        g.cfg.n_spec = 0;
        g.cfg.kv_resident = 65536;                         /* exactly 2 x 1 x 2048 blocks of 16 */
        KVStreamSetup o;
        const std::string l = stderr_of([&] {
            CHECK_OK(kvstream_configure(g.cfg, 65536, groups, prog, &o));
        });
        CHECK(!has(l, "raised"));
        CHECK_EQ(o.raised_tokens, (int64_t)0);
        CHECK_EQ(o.ring_blocks, (int64_t)4096);
        CHECK_EQ(o.ring_tokens, (int64_t)65536);
    }
}

TEST(a_model_whose_architecture_declares_no_streamed_ops_ignores_the_flag) {
    /* An architecture with no streamed path ignores the flag with one line (docs/KV-STREAMING.md
     * §4.5): without the resolve the attention would read block ids past the ring, and without the
     * streamed store a miss would read zeros. */
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    f.cfg.max_ctx = 65536;   /* above the 32 768-token ring floor */
    f.cfg.max_seqs = 2;
    f.cfg.kv_resident = 256;                             /* under the ring floor, so it is raised */
    std::vector<KVGroupInfo> groups = hybrid_groups();
    groups[G_FULL].bytes_per_block = 2048;               /* 16 tokens x 2 kv heads x 8 x 2 B x 2 layers */
    const Program none =
        stream_program(groups[G_FULL], 8, /*with_resolve=*/false, /*with_store=*/false);
    KVStreamSetup s;
    int st = RAD_OK;
    const std::string log = stderr_of([&] { st = kvstream_configure(f.cfg, 65536, groups, none, &s); });
    CHECK_OK(st);                                        /* ignored, not refused */
    CHECK_EQ(s.group, -1);
    CHECK_EQ(s.asked_tokens, (int64_t)0);                /* nothing is claimed, so nothing is reported */
    CHECK_EQ(s.ring_blocks, (int64_t)0);
    CHECK_EQ(s.stage_blocks, (int64_t)0);
    CHECK(has(log, "ignored"));
    CHECK(has(log, "test-arch"));                        /* the architecture, not just the group */
    CHECK(has(log, "kv_resolve"));                       /* the op that is missing, by name */
    CHECK(!has(log, "raised to"));   /* that is the other message: this model cannot stream at all */

    /* Half a path is no path: the resolve with no streamed store would leave the mirror unwritten
     * and the miss list meaningless, so the warning names that op instead. */
    KVStreamSetup half;
    const std::string log2 = stderr_of([&] {
        CHECK_OK(kvstream_configure(f.cfg, 65536, groups,
                                    stream_program(groups[G_FULL], 8, true, false), &half));
    });
    CHECK_EQ(half.group, -1);
    CHECK(has(log2, "kv_store_streamed"));

    /* And the fail-safe never turns a working architecture off: the same configuration with both
     * ops declared streams the ring it was sized for. */
    KVStreamSetup ok;
    CHECK_OK(kvstream_configure(f.cfg, 65536, groups, stream_program(groups[G_FULL], 8), &ok));
    CHECK_EQ(ok.group, (int32_t)G_FULL);
    CHECK_EQ(ok.ring_blocks, (int64_t)2048);   /* raised to the 32 768-token floor */
    CHECK_EQ(ok.stage_blocks, ok.ctx_blocks);
}

TEST(a_declared_streamed_store_with_no_kernel_is_not_a_streamed_path) {
    /* A declaration with no kernel is not a streamed path: the builder keeps the op with one band
     * and no domain bound, and counting it would stream a group whose store never runs. */
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    f.cfg.max_ctx = 65536;   /* above the 32 768-token ring floor */
    f.cfg.max_seqs = 2;
    f.cfg.kv_resident = 256;                             /* under the ring floor, so it is raised */
    std::vector<KVGroupInfo> groups = hybrid_groups();
    groups[G_FULL].decl.dtype = RAD_F8E4M3;
    groups[G_FULL].bytes_per_block = 2048;
    /* The resolve resolves; the store is declared with no row (both domains null). */
    const Program unresolved = stream_program(groups[G_FULL], 8, true, true, /*store_resolved=*/false);
    REQUIRE_EQ(unresolved.ops.size(), (size_t)2);
    REQUIRE_EQ(unresolved.ops[1].bands.size(), (size_t)1);
    CHECK(unresolved.ops[1].bands[0].dom[RAD_DOMAIN_DEVICE].row == nullptr);

    const KVStreamOps ops = kvstream_ops(unresolved, &groups[G_FULL]);
    CHECK(ops.resolve);
    CHECK(!ops.store);
    CHECK(!ops.folded);
    CHECK(!ops.streams());                               /* no writer, so no streamed path */

    KVStreamSetup s;
    int st = RAD_OK;
    const std::string log =
        stderr_of([&] { st = kvstream_configure(f.cfg, 65536, groups, unresolved, &s); });
    CHECK_OK(st);                                        /* ignored, not carved */
    CHECK_EQ(s.group, -1);
    CHECK_EQ(s.asked_tokens, (int64_t)0);
    CHECK_EQ(s.ring_blocks, (int64_t)0);
    CHECK_EQ(s.stage_blocks, (int64_t)0);
    CHECK(has(log, "ignored"));
    CHECK(has(log, "kv_store_streamed"));                /* the op that is missing, by name */

}

TEST(a_packed_cache_warns_that_the_layout_has_no_writer_and_serves_non_streamed) {
    /* --kv-cache-dtype fp8_v4 / k6_v4: the architecture declares the resolve and no streamed store,
     * because libr4d's streamed rows are E4M3 and bf16 and a packed cache has base-store rows only
     * (arch/common/rad_block_attn_gated_fp8.h). The group is declared to the core as plain E4M3, so
     * no dtype gate downstream can see the layout: the operator has to be told the *layout* has no
     * writer, not that the architecture cannot stream -- and the group must run the base path.
     * Before this the op was declared against a table with no row for it, which is a weightless
     * device hole: startup refused with or without --kv-resident, naming a kernel band and never the
     * flag. */
    for (const char* dtype : { "fp8_v4", "k6_v4" }) {
        Fixture f;
        f.cfg.checkpoint_slots = 0;
        f.cfg.max_ctx = 65536;   /* above the 32 768-token ring floor */
        f.cfg.max_seqs = 2;
        f.cfg.kv_resident = 256;                         /* under the ring floor, so it is raised */
        f.cfg.kv_cache_dtype = dtype;
        std::vector<KVGroupInfo> groups = hybrid_groups();
        groups[G_FULL].decl.dtype = RAD_F8E4M3;          /* what declare gives the core, packed or not */
        groups[G_FULL].bytes_per_block = 2048;
        /* What the gated plugin emits for a packed cache: the layout-blind resolve, no store. */
        const Program packed = stream_program(groups[G_FULL], 8, true, /*with_store=*/false);
        KVStreamSetup s;
        int st = RAD_OK;
        const std::string log = stderr_of([&] {
            st = kvstream_configure(f.cfg, 65536, groups, packed, &s);
        });
        CHECK_OK(st);                                    /* started, not refused */
        CHECK_EQ(s.group, -1);
        CHECK_EQ(s.asked_tokens, (int64_t)0);
        CHECK_EQ(s.stage_blocks, (int64_t)0);
        CHECK(has(log, "ignored"));
        CHECK(has(log, std::string("group 'full' has a ") + dtype + " cache"));
        CHECK(has(log, "has no writer for that layout (E4M3 and bf16 only)"));
        CHECK(!has(log, "declares no"));                 /* not that sentence: the arch declared it */
        CHECK(!has(log, "raised to"));                   /* and not the ring's */

    }
}

TEST(a_resolve_declared_for_another_group_does_not_enable_streaming) {
    /* The check is per group, not per program: the resolve states the layer-bytes pitch of the
     * group it was emitted for, so one declared for a narrower full group cannot make streaming
     * of the widest group look wired. That is the difference between "some group resolves" and
     * "the group the plan streams resolves", and only the second is safe. */
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    f.cfg.max_ctx = 65536;   /* above the 32 768-token ring floor */
    f.cfg.max_seqs = 2;
    f.cfg.kv_resident = 256;
    std::vector<KVGroupInfo> groups = hybrid_groups();
    groups[G_FULL].bytes_per_block = 2048;               /* 1024 B a layer */
    Program other = stream_program(groups[G_FULL], 8);
    other.ops[0].base.set_i("block_bytes", 4096);         /* a wider group's pitch, not this one's */
    KVStreamSetup s;
    int st = RAD_OK;
    const std::string log = stderr_of([&] { st = kvstream_configure(f.cfg, 65536, groups, other, &s); });
    CHECK_OK(st);
    CHECK_EQ(s.group, -1);
    CHECK(has(log, "kv_resolve"));
}

TEST(kv_refuses_a_group_with_no_block_size) {
    /* Block size comes from the resolved attention kernel; zero means it did not resolve, and the
     * manager refuses rather than inventing one. */
    auto g = hybrid_groups();
    g[G_FULL].block_size = 0;
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    CHECK_OK(configure_pools(f.pools, f.cfg));
    CHECK_EQ(f.kv.configure(g, f.cfg, f.pools.kv()), RAD_E_NOKERNEL);
}

TEST(kv_refuses_when_fixed_costs_do_not_fit) {
    /* --max-seqs 128 at 16 KiB of GDN state each is 2 MiB of fixed cost against a 1 MiB KV pool:
     * refused by name with the arithmetic, not served with fewer sequences. */
    Fixture f;
    f.cfg.max_seqs = 128;
    f.cfg.checkpoint_slots = 0;
    CHECK_OK(configure_pools(f.pools, f.cfg));
    CHECK_EQ(f.kv.configure(hybrid_groups(), f.cfg, f.pools.kv()), RAD_E_NOMEM);

    /* Snapshots are not part of the fixed cost: at 8 sequences the config fits whatever
     * --checkpoint-slots says (64 x 16 KiB is the whole pool, but they live in host memory), and
     * the paged groups get the same as with none. */
    Fixture g2;
    g2.cfg.max_seqs = 8;
    g2.cfg.checkpoint_slots = 0;
    CHECK_OK(configure_pools(g2.pools, g2.cfg));
    CHECK_OK(g2.kv.configure(hybrid_groups(), g2.cfg, g2.pools.kv()));
    Fixture g3;
    g3.cfg.max_seqs = 8;
    g3.cfg.checkpoint_slots = 64;
    CHECK_OK(configure_pools(g3.pools, g3.cfg));
    CHECK_OK(g3.kv.configure(hybrid_groups(), g3.cfg, g3.pools.kv()));
    CHECK_EQ(g3.kv.fixed_bytes(), g2.kv.fixed_bytes());
    CHECK_EQ(g3.kv.plan(G_FULL)->n_blocks, g2.kv.plan(G_FULL)->n_blocks);
}

TEST(a_manager_whose_carve_was_refused_refuses_rather_than_reads_past_a_vector) {
    /* The bounds check and the read use different vectors. plan_groups() fills plans_ and carve()
     * sizes the block state against it, so a refused carve can leave plans_ passing every bounds
     * check while free_usable_ is empty. reserve_block checks plans_ and reads free_usable_, and is
     * the tiers' restore path from host memory. After a refused carve it must return an error, not
     * read past the vector. */
    Fixture f;
    f.cfg.max_seqs = 128;                 /* fixed costs do not fit, as above */
    f.cfg.checkpoint_slots = 0;
    CHECK_OK(configure_pools(f.pools, f.cfg));
    CHECK_EQ(f.kv.configure(hybrid_groups(), f.cfg, f.pools.kv()), RAD_E_NOMEM);

    int32_t b = 7;
    CHECK(f.kv.reserve_block(G_FULL, &b) < 0);
    CHECK_EQ(b, -1);                      /* no usable block id left behind */
    CHECK(f.kv.reserve_specific_block(G_FULL, 0) < 0);
    CHECK(f.kv.retain_block(G_FULL, 0) < 0);
    CHECK_EQ(f.kv.n_groups(), 0);
    CHECK_EQ(f.kv.total_blocks(G_FULL), 0);
    CHECK_EQ(f.kv.live_blocks(G_FULL), 0);
    CHECK_EQ(f.kv.free_blocks(G_FULL), 0);
}

TEST(kv_alloc_extend_free_and_the_free_list) {
    Fixture f;
    CHECK_OK(f.build());
    const int64_t n0 = f.kv.free_blocks(G_FULL);
    const int64_t s0 = f.kv.free_states(G_GDN);

    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 1));
    CHECK_EQ((long long)f.kv.block_table(1, G_FULL).size(), 1ll);
    CHECK(f.kv.state_slot(1, G_GDN) >= 0);                 /* the same manager owns the GDN slot */
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0 - 1);
    CHECK_EQ(f.kv.free_states(G_GDN), s0 - 1);

    CHECK_OK(f.kv.ensure(1, 16));
    CHECK_EQ((long long)f.kv.block_table(1, G_FULL).size(), 1ll);
    CHECK_OK(f.kv.ensure(1, 17));
    CHECK_EQ((long long)f.kv.block_table(1, G_FULL).size(), 2ll);
    CHECK_OK(f.kv.ensure(1, 64));
    CHECK_EQ((long long)f.kv.block_table(1, G_FULL).size(), 4ll);

    f.kv.free_sequence(1);
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0);
    CHECK_EQ(f.kv.free_states(G_GDN), s0);
    CHECK(!f.kv.has_sequence(1));

    /* Churn catches a free list that leaks one block per sequence. */
    for (uint64_t i = 0; i < 200; ++i) {
        CHECK_OK(f.kv.add_sequence(100 + i));
        CHECK_OK(f.kv.ensure(100 + i, 40));
        f.kv.free_sequence(100 + i);
    }
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0);
    CHECK_EQ(f.kv.free_states(G_GDN), s0);
}

TEST(kv_copy_on_write_on_fork) {
    Fixture f;
    CHECK_OK(f.build());
    const int64_t n0 = f.kv.free_blocks(G_FULL);

    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 24));                       /* 2 blocks; the second is half full */
    auto parent = f.kv.block_table(1, G_FULL);
    CHECK_EQ((long long)parent.size(), 2ll);

    CHECK_OK(f.kv.fork(1, 2));
    CHECK(f.kv.block_table(2, G_FULL) == parent);            /* shared, no copy yet */
    CHECK_EQ(f.kv.block_refcount(G_FULL, parent[0]), 2);
    CHECK_EQ(f.kv.block_refcount(G_FULL, parent[1]), 2);
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0 - 2);              /* the fork costs no blocks */
    CHECK(f.kv.state_slot(2, G_GDN) >= 0);
    CHECK(f.kv.state_slot(2, G_GDN) != f.kv.state_slot(1, G_GDN));  /* a GDN state cannot be shared */

    /* The child writes: only the partial block is copied; the full one stays shared. */
    CHECK_OK(f.kv.ensure(2, 32));
    auto child = f.kv.block_table(2, G_FULL);
    CHECK_EQ(child[0], parent[0]);
    CHECK(child[1] != parent[1]);
    CHECK_EQ(f.kv.block_refcount(G_FULL, parent[1]), 1);
    auto copies = f.kv.take_pending_copies();
    CHECK_EQ((long long)copies.size(), 1ll);
    CHECK_EQ(copies[0].src_block, parent[1]);
    CHECK_EQ(copies[0].dst_block, child[1]);
    CHECK_EQ(copies[0].group, (int32_t)G_FULL);

    /* The parent now holds its partial block alone, so its next write copies nothing. */
    CHECK_OK(f.kv.ensure(1, 32));
    CHECK_EQ(f.kv.block_table(1, G_FULL)[1], parent[1]);
    CHECK_EQ((long long)f.kv.take_pending_copies().size(), 0ll);

    f.kv.free_sequence(1);
    f.kv.free_sequence(2);
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0);
}

TEST(kv_preemption_by_recompute_and_the_undeadlockable_request) {
    Fixture f;
    CHECK_OK(f.build());
    const int64_t total = f.kv.total_blocks(G_FULL);

    /* A request bigger than the whole pool is failed, not preempted for: preempting everything
     * would not make it fit (the deadlock in spec §17). */
    CHECK(!f.kv.can_ever_fit((total + 8) * 16));
    CHECK_OK(f.kv.add_sequence(1));
    std::vector<uint64_t> none;
    CHECK_EQ(f.kv.make_room(1, (total + 8) * 16, none, nullptr), RAD_E_FULL);
    f.kv.free_sequence(1);

    /* Fill the pool, then admit one more by preempting. Four sequences leave state slots for the
     * hog and the newcomer (max_seqs is 8, one state slot per sequence). */
    const int64_t per = 16 * 16;             /* 16 blocks each */
    for (uint64_t i = 1; i <= 4; ++i) {
        CHECK_OK(f.kv.add_sequence(i));
        CHECK_OK(f.kv.ensure(i, per));
    }

    /* Drain the rest so the next ensure cannot be served. */
    uint64_t hog = 900;
    CHECK_OK(f.kv.add_sequence(hog));
    CHECK_OK(f.kv.ensure(hog, f.kv.free_blocks(G_FULL) * 16));
    CHECK_EQ(f.kv.free_blocks(G_FULL), 0ll);

    uint64_t newcomer = 999;
    CHECK_OK(f.kv.add_sequence(newcomer));
    CHECK_EQ(f.kv.ensure(newcomer, per), RAD_E_FULL);

    std::vector<uint64_t> victims = { hog, 1 };
    std::vector<uint64_t> preempted;
    CHECK_OK(f.kv.make_room(newcomer, per, victims, &preempted));
    CHECK(!preempted.empty());
    /* seq_tokens() == 0 means "needs re-prefill" (see KVManager::preempt). */
    CHECK_EQ(f.kv.seq_tokens(preempted[0]), 0ll);
    CHECK(f.kv.block_table(preempted[0], G_FULL).empty());
    CHECK_OK(f.kv.ensure(newcomer, per));
}

/* The window behind a prefix hit: a checkpoint slot carries window_blocks pages per window group,
 * window_rows() names the pages a query at P reads from the table that wrote them, and
 * seed_window() gives a resumed sequence fresh pages at the same floor. A table whose floor moved
 * past P answers false, falling back to a full re-prefill. */
TEST(kv_window_snapshot_rows_and_seed) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    g.push_back(paged_group("swa", RAD_KV_WINDOW, 2, 16, { 0 }, 64));
    Config cfg = small_config();
    Pools p;
    CHECK_OK(configure_pools(p, cfg));
    KVManager kv;
    CHECK_OK(kv.configure(g, cfg, p.kv()));
    CHECK(kv.has_window_snapshot());
    CHECK_EQ(kv.checkpoint_bytes(), 5 * kv.plan(G_FULL)->bytes_per_block);

    CHECK_OK(kv.add_sequence(1));
    CHECK_OK(kv.ensure(1, 400));                         /* a prompt of 400 in one chunk */
    std::vector<int32_t> rows;
    CHECK(kv.window_rows(1, 400, &rows));
    CHECK_EQ((long long)rows.size(), 5ll);               /* padded to window_blocks */
    /* A query at 400 reads back to 337: blocks 21..24, the floor recycle_window keeps. */
    for (int j = 0; j < 4; ++j) CHECK_EQ(rows[(size_t)j], kv.block_table(1, G_FULL)[(size_t)(21 + j)]);
    CHECK_EQ(rows[4], -1);
    /* A short prefix: the window starts at zero and the rest is padding. */
    CHECK(kv.window_rows(1, 32, &rows));
    CHECK_EQ(rows[0], kv.block_table(1, G_FULL)[0]);
    CHECK_EQ(rows[2], -1);
    /* Not on a block: no rows. */
    CHECK(!kv.window_rows(1, 40, &rows));
    /* The floor moved past it (a decode step slid the window): fallback. */
    CHECK_OK(kv.ensure(1, 401));
    CHECK(!kv.window_rows(1, 64, &rows));
    CHECK(kv.window_rows(1, 400, &rows));

    /* A resumed sequence is seeded at the floor, extends from there, and has its own rows. */
    CHECK_OK(kv.add_sequence(2));
    CHECK_OK(kv.seed_window(2, 400));
    CHECK_EQ(kv.first_block_pos(2, G_FULL), 336ll);
    CHECK_EQ((long long)kv.block_table(2, G_FULL).size(), 4ll);
    CHECK_EQ(kv.seq_tokens(2), 400ll);
    CHECK_OK(kv.ensure(2, 432, 400));
    CHECK_EQ(kv.first_block_pos(2, G_FULL), 336ll);
    CHECK_EQ((long long)kv.block_table(2, G_FULL).size(), 6ll);
    std::vector<int32_t> rows2;
    CHECK(kv.window_rows(2, 400, &rows2));
    for (int j = 0; j < 4; ++j) CHECK_EQ(rows2[(size_t)j], kv.block_table(2, G_FULL)[(size_t)j]);
    /* A table already holding window pages is not seeded twice. */
    CHECK_EQ(kv.seed_window(2, 400), RAD_E_STATE);
    kv.free_sequence(1);
    kv.free_sequence(2);
}

/* A window is snapshotted only where a hit resumes from it: the trunk's window, with or without
 * linear state beside it. A drafter's window is recomputed by draft refill, so a window section
 * there would be copied and never read. */
TEST(kv_window_snapshot_only_where_a_hit_resumes_from_it) {
    KVGroupInfo w = paged_group("swa", RAD_KV_WINDOW, 2, 16, { 6 }, 64);
    const Config cfg = small_config();
    {   /* beside linear state: the slot holds the state, then the window. Without the window a
         * hit could not resume on such a model at all (the scheduler refuses a trunk hole). */
        std::vector<KVGroupInfo> g = hybrid_groups();
        g.push_back(w);
        Pools p;
        CHECK_OK(configure_pools(p, cfg));
        KVManager kv;
        CHECK_OK(kv.configure(g, cfg, p.kv()));
        CHECK(kv.has_window_snapshot());
        const KVGroupPlan* wp = kv.plan(3);
        REQUIRE(wp && wp->win_snap);
        CHECK_EQ(kv.checkpoint_bytes(), 16384ll + wp->window_blocks * wp->bytes_per_block);
    }
    {   /* a drafter's own window over a full-attention trunk */
        std::vector<KVGroupInfo> g = hybrid_groups(/*with_linear=*/false);
        w.drafter = true;
        g.push_back(w);
        Pools p;
        CHECK_OK(configure_pools(p, cfg));
        KVManager kv;
        CHECK_OK(kv.configure(g, cfg, p.kv()));
        CHECK(!kv.has_window_snapshot());
        CHECK_EQ(kv.checkpoint_bytes(), 0ll);
        CHECK(!kv.drafter_window_snapshot());
    }
    {   /* the same, where the engine saves behind the drafter's context pass: the drafter's window
         * rides in the snapshot, every page of it */
        std::vector<KVGroupInfo> g = hybrid_groups(/*with_linear=*/false);
        w.drafter = true;
        g.push_back(w);
        Pools p;
        CHECK_OK(configure_pools(p, cfg));
        KVManager kv;
        kv.set_drafter_window_snapshot(true);
        CHECK_OK(kv.configure(g, cfg, p.kv()));
        CHECK(kv.has_window_snapshot());
        CHECK(kv.drafter_window_snapshot());
        const KVGroupPlan* d = kv.plan((int32_t)g.size() - 1);
        REQUIRE(d && d->win_snap && d->drafter);
        CHECK_EQ(kv.checkpoint_bytes(), d->window_blocks * d->bytes_per_block);
    }
    {   /* beside linear state the flag changes nothing: a hit there re-prefills the window */
        std::vector<KVGroupInfo> g = hybrid_groups();
        w.drafter = true;
        g.push_back(w);
        Pools p;
        CHECK_OK(configure_pools(p, cfg));
        KVManager kv;
        kv.set_drafter_window_snapshot(true);
        CHECK_OK(kv.configure(g, cfg, p.kv()));
        CHECK(!kv.has_window_snapshot());
        CHECK(!kv.drafter_window_snapshot());
    }
}

TEST(window_prefix_tail_edit_resumes_at_an_earlier_interval_snapshot) {
    Fixture f;
    f.cfg.checkpoint_interval = 256;
    f.cfg.checkpoint_slots = 8;
    auto groups = hybrid_groups(/*with_linear=*/false);
    groups.push_back(paged_group("swa", RAD_KV_WINDOW, 2, 16, { 2 }, 64));
    REQUIRE_EQ(f.build_groups(groups), RAD_OK);
    CHECK(f.kv.has_window_snapshot());
    auto tokens = tok_run(1000, 1100);
    publish(f, 1, tokens, 1100);
    const auto hashes = chain_of(tokens);
    for (int64_t p : { 256, 512, 768, 1024, 1072 }) ckpt_at(f, 1, hashes, p);
    f.pc.retain(1);

    /* Replacing the final 123 tokens invalidates both the tail and 1024 snapshot,
     * but the unchanged prefix still reaches the snapshot at 768. */
    for (size_t i = 977; i < tokens.size(); ++i) tokens[i] += 10000;
    const CacheHit hit = f.pc.lookup(tokens, {});
    CHECK_EQ(hit.n_attn_tokens, 976);
    CHECK_EQ(hit.n_linear_tokens, 768);
    CHECK(hit.checkpoint_slot >= 0);
}

TEST(kv_window_group_recycles_below_the_window) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    g.push_back(paged_group("swa", RAD_KV_WINDOW, 2, 16, { 0 }, 64));

    Config cfg = small_config();
    Pools p;
    CHECK_OK(configure_pools(p, cfg));
    KVManager kv;
    cfg.checkpoint_slots = 0;
    CHECK_OK(kv.configure(g, cfg, p.kv()));
    /* Window 64 at block 16 is 4 blocks plus one for the straddle. */
    CHECK_EQ(kv.plan(G_FULL)->window_blocks, 5ll);
    /* Sized to the cap, not the whole budget: a window per sequence, one step's tokens (max_tok),
     * and a straddle block per sequence, since a chunk's blocks are held beside the window behind
     * its first query. More would never be addressed. */
    CHECK_EQ(kv.total_blocks(G_FULL), cfg.max_seqs * 5 + (cfg.max_tok + 15) / 16 + cfg.max_seqs);

    const int64_t n0 = kv.free_blocks(G_FULL);
    CHECK_OK(kv.add_sequence(1));
    CHECK_OK(kv.ensure(1, 80));                          /* a chunk from 0: all held */
    CHECK_EQ(kv.first_block_pos(1, G_FULL), 0ll);
    CHECK_EQ((long long)kv.block_table(1, G_FULL).size(), 5ll);
    /* A chunk from 80 to 400: its first query (80) reads back to 17, so block 1 stays and only
     * block 0 goes. Trimming to the window behind 400 would leave rows 80..335 without keys. */
    CHECK_OK(kv.ensure(1, 400));
    CHECK_EQ(kv.first_block_pos(1, G_FULL), 16ll);
    CHECK_EQ((long long)kv.block_table(1, G_FULL).size(), 24ll);
    /* A decode step from 400 shrinks back to the window, recycling rather than growing. */
    CHECK_OK(kv.ensure(1, 401));
    CHECK_EQ(kv.first_block_pos(1, G_FULL), 336ll);
    CHECK_EQ((long long)kv.block_table(1, G_FULL).size(), 5ll);
    CHECK_EQ(kv.free_blocks(G_FULL), n0 - 5);
    kv.free_sequence(1);
    CHECK_EQ(kv.free_blocks(G_FULL), n0);

    /* A prompt longer than the window, in one chunk from 0: every position keeps its slot, so a
     * sliding layer has keys for every query in the chunk. */
    CHECK_OK(kv.add_sequence(2));
    CHECK_OK(kv.ensure(2, 400));
    CHECK_EQ(kv.first_block_pos(2, G_FULL), 0ll);
    CHECK_EQ((long long)kv.block_table(2, G_FULL).size(), 25ll);
    /* The scheduler's form passes the step's first query: a chunk grown twice in one step keeps
     * what the first call's first query reads. */
    CHECK_OK(kv.ensure(2, 432, 400));
    CHECK_EQ(kv.first_block_pos(2, G_FULL), 336ll);
    CHECK_OK(kv.ensure(2, 464, 400));
    CHECK_EQ(kv.first_block_pos(2, G_FULL), 336ll);
    CHECK_EQ((long long)kv.block_table(2, G_FULL).size(), 8ll);
    kv.free_sequence(2);
    CHECK_EQ(kv.free_blocks(G_FULL), n0);

    /* A committed chunk returns its blocks at once: trim_window to the new end keeps only the
     * window behind it, so a prefill waiting for its next chunk holds a window, not a chunk. */
    CHECK_OK(kv.add_sequence(3));
    CHECK_OK(kv.ensure(3, 400));
    CHECK_EQ((long long)kv.block_table(3, G_FULL).size(), 25ll);
    kv.trim_window(3, 400);
    CHECK_EQ(kv.first_block_pos(3, G_FULL), 336ll);
    CHECK_EQ((long long)kv.block_table(3, G_FULL).size(), 4ll);
    CHECK_EQ(kv.free_blocks(G_FULL), n0 - 4);
    CHECK_OK(kv.ensure(3, 401));
    CHECK_EQ((long long)kv.block_table(3, G_FULL).size(), 5ll);
    kv.free_sequence(3);
    CHECK_EQ(kv.free_blocks(G_FULL), n0);
}

TEST(kv_conv_slot_is_a_rolling_window) {
    /* libr4d's conv update treats the state as a rolling window of width-1 + num_spec entries,
     * read at the last accepted token's slot, so a speculative rejection changes the read offset
     * rather than recomputing (spec §10). The same manager sizes the slot; two allocators would
     * disagree about what a sequence owns. */
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    KVGroupInfo k;
    k.name = "conv";
    k.decl.kind = RAD_KV_CONV;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = 2;
    k.decl.head_dim = 8;
    k.decl.conv_width = 4;
    k.layers = { 0, 1 };
    g.push_back(k);                           /* handle 1 */
    g.push_back(hybrid_groups()[G_FULL]);     /* a paged group beside it, as in a hybrid */

    Config cfg = small_config();
    cfg.n_spec = 3;
    Pools p;
    CHECK_OK(configure_pools(p, cfg));
    KVManager kv;
    cfg.checkpoint_slots = 0;
    CHECK_OK(kv.configure(g, cfg, p.kv()));

    /* The conv group is handle 1 here; the sentinel is 0. */
    const int32_t G_CONV = 1;
    CHECK_EQ(kv.plan(G_CONV)->conv_slots, 6ll);                    /* (4-1) + 3 */
    CHECK_EQ(kv.plan(G_CONV)->bytes_per_state, 6ll * 2 * 8 * 2 * 2);
    CHECK_EQ(kv.plan(G_CONV)->n_states, cfg.max_seqs);

    const int64_t s0 = kv.free_states(G_CONV);
    CHECK_OK(kv.add_sequence(1));
    CHECK_OK(kv.ensure(1, 8));
    CHECK(kv.state_slot(1, G_CONV) >= 0);
    CHECK_EQ(kv.free_states(G_CONV), s0 - 1);
    kv.free_sequence(1);
    CHECK_EQ(kv.free_states(G_CONV), s0);
}

/* The manager's slot must be the slot the kernels index. Kernels take a conv slot's layout from
 * the state tensor's last extent, conv_width-1 + n_spec; the manager clears, saves and restores a
 * slot in the layout bytes_per_state describes. If the builder's figure were wider, clearing slot s
 * would zero the start of slot s+1, a live sequence's window, so the manager refuses any
 * mismatch. */
TEST(kv_refuses_a_state_slot_the_kernels_would_not_index) {
    auto group = [](int kind, int64_t bytes_per_state) {
        KVGroupInfo k;
        k.name = "state";
        k.decl.kind = kind;
        k.decl.dtype = RAD_BF16;
        k.decl.n_head_kv = 2;
        k.decl.head_dim = 8;
        k.decl.state_dim[0] = 4;
        k.decl.state_dim[1] = 8;
        k.decl.conv_width = 4;
        k.layers = { 0, 1 };
        k.bytes_per_state = bytes_per_state;
        return k;
    };
    auto configure = [](const KVGroupInfo& k) -> int {
        std::vector<KVGroupInfo> g;
        g.push_back(null_group());
        g.push_back(k);
        Config cfg = small_config();
        cfg.n_spec = 3;
        cfg.checkpoint_slots = 0;
        Pools p;
        if (configure_pools(p, cfg) < 0)
            return RAD_E_NOMEM;
        KVManager kv;
        return kv.configure(g, cfg, p.kv());
    };
    /* Conv: (4-1) + 3 = 6 entries of 2 x 8 bf16 channels, over two layers. */
    CHECK_OK(configure(group(RAD_KV_CONV, 6LL * 2 * 8 * 2 * 2)));
    CHECK_EQ(configure(group(RAD_KV_CONV, 7LL * 2 * 8 * 2 * 2)), RAD_E_SHAPE);
    /* Linear: 2 heads of 4 x 8 bf16, over two layers. */
    CHECK_OK(configure(group(RAD_KV_LINEAR, 2LL * 4 * 8 * 2 * 2)));
    CHECK_EQ(configure(group(RAD_KV_LINEAR, 2LL * 4 * 8 * 2 * 2 + 64)), RAD_E_SHAPE);
}

TEST(kv_adopts_the_blocks_a_cache_hit_found) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    auto prompt = tok_run(555, 96);
    publish(f, 1, prompt, 64);                 /* 4 blocks indexed */

    std::vector<std::vector<int32_t>> found;
    CacheHit h = f.pc.lookup(prompt, {}, &found);
    CHECK_EQ(h.n_attn_tokens, 64);
    CHECK_EQ((long long)found.size(), 1ll);
    CHECK_EQ((long long)found[0].size(), 4ll);

    const int64_t before = f.kv.free_blocks(G_FULL);
    CHECK_OK(f.kv.add_sequence(2));
    CHECK_OK(f.kv.adopt(2, f.pc.cached_groups()[0], found[0], h.n_attn_tokens));
    CHECK_EQ(f.kv.free_blocks(G_FULL), before);                 /* a hit costs no blocks */
    CHECK_EQ(f.kv.seq_tokens(2), 64ll);
    CHECK_EQ(f.kv.block_refcount(G_FULL, found[0][0]), 3);      /* seq 1 + cache + seq 2 */

    /* A partial block is never adopted: the next token written into it would corrupt any other
     * holder, and the cache cannot serve it anyway. */
    CHECK_OK(f.kv.add_sequence(3));
    CHECK_EQ(f.kv.adopt(3, 0, found[0], 60), RAD_E_INVAL);

    /* Extending past the hit allocates only new blocks; shared ones stay shared. */
    CHECK_OK(f.kv.ensure(2, 96));
    CHECK_EQ((long long)f.kv.block_table(2, G_FULL).size(), 6ll);
    CHECK_EQ(f.kv.block_table(2, G_FULL)[0], found[0][0]);
    CHECK_EQ((long long)f.kv.take_pending_copies().size(), 0ll);
}

/* ================================================================== chained hash */

TEST(chained_hash_identity) {
    auto a = tok_run(1000, 16);
    auto b = tok_run(2000, 16);
    auto shared = tok_run(3000, 16);
    std::vector<MMContentKey> no_mm;

    BlockHash root;
    BlockHash a0 = chain_block_hash(root, a.data(), 16, nullptr, 0, 0);
    BlockHash b0 = chain_block_hash(root, b.data(), 16, nullptr, 0, 0);
    CHECK(a0 != b0);
    CHECK(a0.valid());

    /* The same prefix hits. */
    CHECK(chain_block_hash(root, a.data(), 16, nullptr, 0, 0) == a0);

    /* The same span under a different prefix does not: chaining keeps a block from the middle of
     * another conversation from being served here. */
    BlockHash a1 = chain_block_hash(a0, shared.data(), 16, nullptr, 0, 16);
    BlockHash b1 = chain_block_hash(b0, shared.data(), 16, nullptr, 0, 16);
    CHECK(a1 != b1);

    /* The same span, same parent, different position is also a different block. */
    CHECK(chain_block_hash(a0, shared.data(), 16, nullptr, 0, 32) != a1);

    /* Multimodal blocks hash by content: identical placeholder ids, different content keys. */
    MMContentKey m1{ 0, 16, 0x1111 };
    MMContentKey m2{ 0, 16, 0x2222 };
    BlockHash h1 = chain_block_hash(root, a.data(), 16, &m1, 1, 0);
    BlockHash h2 = chain_block_hash(root, a.data(), 16, &m2, 1, 0);
    CHECK(h1 != h2);
    CHECK(h1 != a0);                                     /* neither aliases the text-only block */
    CHECK(chain_block_hash(root, a.data(), 16, &m1, 1, 0) == h1);   /* same image, same hash */
}

/* ================================================================== prefix cache */

TEST(prefix_cache_hits_the_shared_prefix_only) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    CHECK(f.pc.enabled());
    CHECK_EQ(f.pc.block_size(), 16ll);

    auto p1 = tok_run(10, 64);                  /* 4 blocks */
    publish(f, 1, p1, 64);

    /* Same prefix, longer prompt: all four indexed blocks reused. */
    auto p2 = p1;
    for (int i = 0; i < 32; ++i) p2.push_back(500 + i);
    CacheHit h = f.pc.lookup(p2, {});
    CHECK_EQ(h.n_attn_tokens, 64);

    /* Diverging at block 2: only blocks 0 and 1 are reusable. */
    auto p3 = p1;
    p3[35] = 77777;
    CacheHit h3 = f.pc.lookup(p3, {});
    CHECK_EQ(h3.n_attn_tokens, 32);

    /* A block under a different prefix does not collide: p4's blocks 1..3 equal p1's but block 0
     * differs, so nothing matches. */
    auto p4 = p1;
    for (int i = 0; i < 16; ++i) p4[i] = 9000 + i;
    CHECK_EQ(f.pc.lookup(p4, {}).n_attn_tokens, 0);

    /* A full hit backs off one block: the step needs a token to compute a logit from. */
    CHECK_EQ(f.pc.lookup(p1, {}).n_attn_tokens, 48);
}

/* Where a hit ends at a head two cached prompts already left differently (CacheHit::branch_at):
 * the same instructions in front of different texts. One cached prompt past it is not enough, the
 * next prompt could be the same conversation going on; a prompt that follows one of them further, or
 * is all cached, is at no branch. */
TEST(a_hit_reports_the_head_two_cached_prompts_left_differently) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    const auto head = tok_run(10, 64);                  /* 4 blocks every prompt starts with */
    auto with_tail = [&](int32_t base) {
        auto p = head;
        for (int i = 0; i < 48; ++i) p.push_back(base + i);
        return p;
    };
    const auto a = with_tail(1000), b = with_tail(2000), c = with_tail(3000);
    publish(f, 1, a, 112);
    CacheHit h1 = f.pc.lookup(b, {});
    CHECK_EQ(h1.n_attn_tokens, 64);
    CHECK_EQ(h1.branch_at, 0);
    publish(f, 2, b, 112);
    CacheHit h2 = f.pc.lookup(c, {});
    CHECK_EQ(h2.n_attn_tokens, 64);
    CHECK_EQ(h2.branch_at, 64);

    auto a2 = a;
    for (int i = 0; i < 32; ++i) a2.push_back(4000 + i);
    CHECK_EQ(f.pc.lookup(a2, {}).n_attn_tokens, 112);
    CHECK_EQ(f.pc.lookup(a2, {}).branch_at, 0);
    CHECK_EQ(f.pc.lookup(a, {}).branch_at, 0);
    /* Publishing the same prompt again adds no way on. */
    publish(f, 3, a, 112);
    auto d = head;
    for (int i = 0; i < 16; ++i) d.push_back(1000 + i);
    for (int i = 0; i < 32; ++i) d.push_back(5000 + i);
    CHECK_EQ(f.pc.lookup(d, {}).n_attn_tokens, 80);
    CHECK_EQ(f.pc.lookup(d, {}).branch_at, 0);
}

TEST(prefix_cache_lru_eviction_order) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));

    auto a = tok_run(100, 32), b = tok_run(200, 32), c = tok_run(300, 32);
    publish(f, 1, a, 16);           /* one indexed block each */
    publish(f, 2, b, 16);
    publish(f, 3, c, 16);
    CHECK_EQ(f.pc.size(), 3ll);
    /* While sequences hold the blocks, eviction takes nothing: releasing the cache's reference
     * would free no block and only lose the entry. */
    CHECK_EQ(f.pc.evict_lru(3), 0ll);
    CHECK_EQ(f.pc.size(), 3ll);
    for (uint64_t id : { 1, 2, 3 }) f.kv.free_sequence(id);

    /* Touch A, so the order becomes B, C, A. */
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 16);
    auto order = f.pc.lru_order();
    CHECK_EQ((long long)order.size(), 3ll);

    CHECK_EQ(f.pc.evict_lru(1), 1ll);
    CHECK_EQ(f.pc.size(), 2ll);
    CHECK_EQ(f.pc.lookup(b, {}).n_attn_tokens, 0);      /* B was the oldest and went first */
    CHECK_EQ(f.pc.lookup(c, {}).n_attn_tokens, 16);
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 16);
}

/* A chain is stamped with one clock, and pressure takes its deepest block first: what is left is
 * still a prefix a lookup reaches. Taking the first block first would leave the rest unreachable
 * and still held. */
TEST(pressure_takes_a_conversation_s_end_before_its_start) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    const auto a = tok_run(100, 16 * 5);
    publish(f, 1, a, 16 * 4);
    f.kv.free_sequence(1);
    CHECK_EQ(f.pc.evict_lru(1), 1ll);
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 48);
    /* A lookup stamps the chain the same way. */
    CHECK_EQ(f.pc.evict_lru(1), 1ll);
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 32);
}

TEST(prefix_cache_holds_a_reference_on_every_indexed_block) {
    /* Invariant: a block named by the cache is never on the free list, so it cannot be handed to
     * another sequence under a cache entry. */
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    auto a = tok_run(100, 32);
    publish(f, 1, a, 32);
    int32_t blk = f.kv.block_table(1, G_FULL)[0];
    CHECK_EQ(f.kv.block_refcount(G_FULL, blk), 2);           /* sequence + cache */
    const int64_t before = f.kv.free_blocks(G_FULL);
    f.kv.free_sequence(1);
    CHECK_EQ(f.kv.block_refcount(G_FULL, blk), 1);           /* still held by the cache */
    f.pc.clear();
    CHECK_EQ(f.kv.free_blocks(G_FULL), before + 2);
}

/* The cache reports the context it holds, which distinguishes a full KV pool from a leaking one:
 * the pool counts every held block, and on a conversational server most of a full pool is
 * finished turns kept for reuse. */
TEST(the_cache_reports_the_context_it_is_holding) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    CHECK_EQ(f.pc.size() * f.pc.block_size(), 0);

    auto a = tok_run(100, 32);
    publish(f, 1, a, 32);
    CHECK_EQ(f.pc.size() * f.pc.block_size(), 32);

    /* A second conversation sharing the prefix adds only its new blocks: entries are keyed by the
     * chain, so shared blocks are one entry held once, and the figure is a share of the pool, not a
     * sum over sessions. */
    auto b = tok_run(100, 32);
    for (size_t i = 16; i < b.size(); ++i) b[i] += 1000;
    publish(f, 2, b, 32);
    CHECK_EQ(f.pc.size() * f.pc.block_size(), 48);

    /* It falls when the cache releases blocks. */
    f.kv.free_sequence(1);
    f.kv.free_sequence(2);
    f.pc.evict_lru(1);
    CHECK_EQ(f.pc.size() * f.pc.block_size(), 32);
    f.pc.clear();
    CHECK_EQ(f.pc.size() * f.pc.block_size(), 0);
}

/* ================================================================== the split lookup */

TEST(split_lookup_reuses_attention_between_checkpoints) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    CHECK_EQ(f.pc.checkpoint_interval(), 64ll);
    CHECK_EQ(f.pc.block_size(), 16ll);

    /* The scheduler splits chunks at these positions, so checkpoints land on the interval, not
     * wherever a step boundary falls (spec §7.3). */
    CHECK_EQ(f.pc.next_checkpoint_after(0), 64ll);
    CHECK_EQ(f.pc.next_checkpoint_after(1), 64ll);
    CHECK_EQ(f.pc.next_checkpoint_after(64), 128ll);
    CHECK_EQ(f.pc.next_checkpoint_after(65), 128ll);
    CHECK(f.pc.is_checkpoint(128));
    CHECK(!f.pc.is_checkpoint(80));

    auto prompt = tok_run(7000, 336);         /* 21 blocks */
    publish(f, 1, prompt, 320);               /* 20 blocks indexed */

    /* Without a checkpoint the linear layers replay from zero. */
    CacheHit h0 = f.pc.lookup(prompt, {});
    CHECK_EQ(h0.n_attn_tokens, 320);
    CHECK_EQ(h0.n_linear_tokens, 0);
    CHECK_EQ(h0.checkpoint_slot, -1);

    /* A checkpoint at 128, keyed by the chained block hash at that position (as the attention
     * lookup computes it). */
    const std::vector<BlockHash> hs = chain_of(prompt);
    int32_t slot = -1;
    CHECK_OK(f.pc.reserve_checkpoint(/*session=*/1, 128, hs[128 / 16 - 1], &slot));
    CHECK(slot >= 0);

    /* Reserved is not written. The reservation happens when the chunk is planned and the state
     * exists only after the step runs; until then the slot holds the previous tenant's history,
     * so lookup skips it until commit_checkpoint reports the copy issued. */
    CacheHit pending = f.pc.lookup(prompt, {});
    CHECK_EQ(pending.n_attn_tokens, 320);
    CHECK_EQ(pending.n_linear_tokens, 0);
    CHECK_EQ(pending.checkpoint_slot, -1);
    f.pc.commit_checkpoint(hs[128 / 16 - 1]);

    /* The hit lands between checkpoints: attention is reused to 320, linear state restored at 128,
     * and only [128, 320) replays through the linear layers. This is why the interval is decoupled
     * from the block size. */
    CacheHit h1 = f.pc.lookup(prompt, {});
    CHECK_EQ(h1.n_attn_tokens, 320);
    CHECK_EQ(h1.n_linear_tokens, 128);
    CHECK_EQ(h1.checkpoint_slot, slot);

    /* A later checkpoint shortens the replay. */
    const int32_t slot2 = ckpt_at(f, 1, hs, 256);
    CacheHit h2 = f.pc.lookup(prompt, {});
    CHECK_EQ(h2.n_attn_tokens, 320);
    CHECK_EQ(h2.n_linear_tokens, 256);
    CHECK_EQ(h2.checkpoint_slot, slot2);

    /* A checkpoint under a different prefix is unreachable. */
    auto other = tok_run(9000, 336);
    CHECK_EQ(f.pc.lookup(other, {}).n_linear_tokens, 0);
}

/* A capped lookup (HitCap) ends the hit at a block and the linear state at a checkpoint at or
 * below the cap, unless that gives back more than max_give_back, in which case the whole hit
 * stands. Statistics count one lookup either way. */
TEST(a_capped_lookup_lands_on_a_checkpoint_at_or_below_the_cap) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    auto prompt = tok_run(7000, 336);
    publish(f, 1, prompt, 320);
    const std::vector<BlockHash> hs = chain_of(prompt);
    const int32_t s128 = ckpt_at(f, 1, hs, 128);
    ckpt_at(f, 1, hs, 256);

    const CacheHit u = f.pc.lookup(prompt, {});
    CHECK_EQ(u.n_attn_tokens, 320);
    CHECK_EQ(u.n_linear_tokens, 256);
    CHECK_EQ(u.given_back, 0);

    HitCap cap;
    cap.tokens = 250;
    cap.max_give_back = 1000;
    const int64_t lookups = f.pc.stats().lookups;
    const CacheHit c = f.pc.lookup(prompt, {}, nullptr, &cap);
    CHECK_EQ(f.pc.stats().lookups, lookups + 1);
    CHECK_EQ(c.n_attn_tokens, 240);
    CHECK_EQ(c.n_linear_tokens, 128);
    CHECK_EQ(c.checkpoint_slot, s128);
    CHECK_EQ(c.given_back, 128);              /* 256 reusable before, 128 after */

    cap.tokens = 100;                         /* below every checkpoint: all 256 go back */
    const CacheHit z = f.pc.lookup(prompt, {}, nullptr, &cap);
    CHECK_EQ(z.n_attn_tokens, 96);
    CHECK_EQ(z.n_linear_tokens, 0);
    CHECK_EQ(z.checkpoint_slot, -1);
    CHECK_EQ(z.given_back, 256);

    cap.max_give_back = 255;                  /* ...more than this cap allows */
    const CacheHit w = f.pc.lookup(prompt, {}, nullptr, &cap);
    CHECK_EQ(w.n_attn_tokens, 320);
    CHECK_EQ(w.n_linear_tokens, 256);
    CHECK_EQ(w.given_back, 0);

    cap.tokens = 100000;                      /* a cap above the hit changes nothing */
    CHECK_EQ(f.pc.lookup(prompt, {}, nullptr, &cap).n_attn_tokens, 320);
}

/* tip-only under a drafter whose KV the cache does not hold (--draft-refill): a hit must end the
 * drafter's window before the prompt's end, on a snapshot no more than max_give_back below the tip.
 * Keeping the tip alone leaves no such snapshot once retention runs, the capped lookup keeps the
 * whole hit, and the request decodes without drafts (measured: 8 x 22k-token sessions, 5 of 8 ran
 * unspeculated, decode 424-521 t/s against 738-754 under geometric backoff). With the floor set,
 * the policy also keeps the deepest snapshot at or below tip - floor. */
TEST(tip_only_keeps_a_snapshot_below_the_drafter_floor) {
    std::vector<int64_t> keep;
    KeepTipOnly bare;
    bare.select({64, 128, 192, 256}, &keep);
    CHECK_EQ(keep.size(), (size_t)1);
    CHECK_EQ(keep[0], 256ll);

    KeepTipOnly floored;
    floored.set_floor(100);
    floored.select({64, 128, 192, 256}, &keep);
    CHECK_EQ(keep.size(), (size_t)2);
    CHECK_EQ(keep[0], 128ll);                 /* the deepest at or below 256 - 100 */
    CHECK_EQ(keep[1], 256ll);
    floored.select({200, 256}, &keep);        /* nothing below the floor: the tip alone */
    CHECK_EQ(keep.size(), (size_t)1);
    CHECK_EQ(floored.held(4096, 64), 2ll);
    CHECK_EQ(bare.held(4096, 64), 1ll);
}

TEST(a_refill_capped_hit_after_tip_only_retention_still_gives_the_drafter_its_window) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    f.pc.set_policy(std::make_unique<KeepTipOnly>());
    f.pc.set_drafter_floor(100);
    auto prompt = tok_run(7000, 336);
    publish(f, 1, prompt, 320);
    const std::vector<BlockHash> hs = chain_of(prompt);
    for (int64_t pos = 64; pos <= 256; pos += 64) {
        f.pc.begin_step();
        ckpt_at(f, 1, hs, pos);
    }
    /* Another session's reservation finds the store full: retention runs over every session. */
    auto other = tok_run(90000, 128);
    const std::vector<BlockHash> ho = chain_of(other);
    f.pc.begin_step();
    int32_t s = -1;
    CHECK_OK(f.pc.reserve_checkpoint(2, 64, ho[3], &s));
    const std::vector<int64_t> left = f.pc.checkpoints(1);
    CHECK_EQ(left.size(), (size_t)2);
    CHECK_EQ(left[0], 128ll);
    CHECK_EQ(left[1], 256ll);

    /* Scheduler::admit_one's cap for a drafter window of 100: end the hit 100 before the prompt's
     * end, give back at most 200. */
    HitCap cap;
    cap.tokens = (int64_t)prompt.size() - 100;
    cap.max_give_back = 200;
    const CacheHit c = f.pc.lookup(prompt, {}, nullptr, &cap);
    CHECK_EQ(c.n_linear_tokens, 128);
    CHECK_EQ(c.given_back, 128);
}

/* A checkpoint at a prompt's tail, off the interval, is one the next turn resumes from: lookup
 * takes the deepest snapshot under the hit at any block boundary. */
TEST(a_checkpoint_at_the_tail_of_a_prompt_is_found_and_the_deepest_wins) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    const auto prompt = tok_run(7000, 336);
    publish(f, 1, prompt, 320);
    const std::vector<BlockHash> hs = chain_of(prompt);
    ckpt_at(f, 1, hs, 128);                   /* on the interval */
    ckpt_at(f, 1, hs, 304);                   /* the tail: a block boundary, off the interval */
    CacheHit h = f.pc.lookup(prompt, {});
    CHECK_EQ(h.n_attn_tokens, 320);
    CHECK_EQ(h.n_linear_tokens, 304);

    /* A prompt that shares only the first 200 tokens reaches 128 and not 304. */
    auto other = prompt;
    for (size_t k = 200; k < other.size(); ++k) other[k] += 50000;
    CHECK_EQ(f.pc.lookup(other, {}).n_linear_tokens, 128);

    /* Off a block boundary there is no key for it. */
    int32_t slot = -1;
    CHECK_EQ(f.pc.reserve_checkpoint(1, 300, hs[300 / 16 - 1], &slot), RAD_E_INVAL);
}

/* A conversation's snapshots belong to its latest turn, so retention sees one session with one
 * tip rather than one tip per turn. A diverging conversation takes over only the snapshots within
 * the prefix it shares. */
TEST(a_turn_that_extends_a_conversation_takes_over_its_snapshots) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/8));
    const auto p1 = tok_run(7000, 336);
    publish(f, 1, p1, 320);
    const std::vector<BlockHash> hs = chain_of(p1);
    ckpt_at(f, 1, hs, 128);
    ckpt_at(f, 1, hs, 256);
    CHECK_EQ(f.pc.checkpoints(1).size(), 2u);

    auto p2 = p1;
    for (int k = 0; k < 64; ++k) p2.push_back(90000 + k);
    publish(f, 2, p2, 400);
    CHECK(f.pc.checkpoints(1).empty());
    CHECK(f.pc.checkpoints(2) == (std::vector<int64_t>{ 128, 256 }));
    CHECK_EQ(f.pc.sessions_tracked(), 1ll);

    auto p3 = p1;
    for (size_t k = 200; k < p3.size(); ++k) p3[k] += 50000;
    publish(f, 3, p3, 320);
    CHECK(f.pc.checkpoints(3) == (std::vector<int64_t>{ 128 }));
    CHECK(f.pc.checkpoints(2) == (std::vector<int64_t>{ 256 }));
}

TEST(checkpoint_interval_must_be_a_block_multiple) {
    Fixture f;
    f.cfg.checkpoint_interval = 100;          /* not a multiple of 16 */
    CHECK_OK(configure_pools(f.pools, f.cfg));
    f.cfg.checkpoint_slots = 2;
    CHECK_OK(f.kv.configure(hybrid_groups(true), f.cfg, f.pools.kv()));
    /* It would never produce a hit, so it is refused. */
    CHECK_EQ(f.pc.configure(f.cfg, &f.kv, f.cfg.checkpoint_interval), RAD_E_INVAL);
}

/* ================================================================== mixed block sizes
 *
 * A model can keep KV at different strides: DeepSeek-V4's compressed attention stores one row per
 * r tokens (a block of r) beside groups at one row a token. The cache indexes every such group at
 * the lcm of their blocks, an entry holding lcm / B blocks of a B-token group, so a hit ends on a
 * block of each and adopt() takes every group. A group left out would leave the trunk a hole, and
 * the scheduler refuses every hit on such a model.
 */
static KVGroupInfo gdn_group(std::vector<int32_t> layers) {
    KVGroupInfo k;
    k.name = "gdn";
    k.decl.kind = RAD_KV_LINEAR;
    k.decl.dtype = RAD_F32;
    k.decl.n_head_kv = 4;
    k.decl.state_dim[0] = 16;
    k.decl.state_dim[1] = 16;
    k.layers = std::move(layers);
    return k;
}

/* Handles: 1 = the coarse group (`a`-token blocks), 2 = the fine one (`b`), 3 = linear state. */
static std::vector<KVGroupInfo> mixed_groups(int64_t a, int64_t b, bool with_linear) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    g.push_back(paged_group("coarse", RAD_KV_FULL, 2, a, { 0, 1 }));
    g.push_back(paged_group("fine", RAD_KV_FULL, 1, b, { 2 }));
    if (with_linear) g.push_back(gdn_group({ 3, 4 }));
    return g;
}

static std::vector<int32_t> head(const std::vector<int32_t>& v, int64_t n) {
    return std::vector<int32_t>(v.begin(), v.begin() + std::min<int64_t>(n, (int64_t)v.size()));
}

TEST(a_cache_over_mixed_block_sizes_indexes_every_group_at_their_lcm) {
    for (const auto& ab : { std::pair<int64_t, int64_t>{ 2, 1 }, { 2, 4 }, { 16, 2 } }) {
        const int64_t a = ab.first, b = ab.second, L = std::lcm(a, b);
        Fixture f;
        REQUIRE_EQ(f.build_groups(mixed_groups(a, b, false)), RAD_OK);
        REQUIRE(f.pc.enabled());
        CHECK_EQ(f.pc.block_size(), L);
        CHECK(f.pc.cached_groups() == (std::vector<int32_t>{ 1, 2 }));
        CHECK_EQ(f.pc.blocks_per_entry(0), L / a);
        CHECK_EQ(f.pc.blocks_per_entry(1), L / b);
        std::vector<int32_t> eg;
        for (int64_t k = 0; k < L / a; ++k) eg.push_back(1);
        for (int64_t k = 0; k < L / b; ++k) eg.push_back(2);
        CHECK(f.pc.entry_groups() == eg);
        /* Named once in the report, and nothing left out. */
        CHECK(f.pc.report().find("not cached") == std::string::npos);
        CHECK(f.pc.report().find("blocks an entry") != std::string::npos);
        const int64_t free1 = f.kv.free_blocks(1), free2 = f.kv.free_blocks(2);

        /* 101 tokens publish their whole entries: 96 or 100 of them. */
        const auto p = tok_run(300, 101);
        const int64_t whole = 101 / L * L;
        publish(f, 1, p, 101);
        CHECK_EQ(f.pc.size(), whole / L);
        const std::vector<int32_t> t1 = f.kv.block_table(1, 1), t2 = f.kv.block_table(1, 2);
        for (int64_t k = 0; k < whole / a; ++k) CHECK_EQ(f.kv.block_refcount(1, t1[(size_t)k]), 2);
        for (int64_t k = 0; k < whole / b; ++k) CHECK_EQ(f.kv.block_refcount(2, t2[(size_t)k]), 2);
        /* The partial block past the last entry is the sequence's alone. */
        if ((int64_t)t2.size() > whole / b) CHECK_EQ(f.kv.block_refcount(2, t2[(size_t)(whole / b)]), 1);

        /* A longer prompt hits every published entry, with each group's own blocks. */
        auto longer = p;
        for (int i = 0; i < 40; ++i) longer.push_back(900 + i);
        std::vector<std::vector<int32_t>> got;
        CHECK_EQ(f.pc.lookup(longer, {}, &got).n_attn_tokens, (int32_t)whole);
        REQUIRE_EQ(got.size(), 2u);
        CHECK(got[0] == head(t1, whole / a));
        CHECK(got[1] == head(t2, whole / b));
        /* A full hit backs off one entry, and a divergence ends the hit on the entry below it. */
        CHECK_EQ(f.pc.lookup(head(p, whole), {}).n_attn_tokens, (int32_t)(whole - L));
        auto edited = p;
        edited[37] += 5000;
        CHECK_EQ(f.pc.lookup(edited, {}).n_attn_tokens, (int32_t)(37 / L * L));

        /* The hit adopts in every group, and the blocks are shared, not copied. */
        CHECK_OK(f.kv.add_sequence(2));
        for (size_t i = 0; i < 2; ++i)
            CHECK_OK(f.kv.adopt(2, f.pc.cached_groups()[i], got[i], whole));
        CHECK_OK(f.kv.ensure(2, (int64_t)longer.size()));
        CHECK_EQ(f.kv.block_refcount(2, t2[0]), 3);
        CHECK(head(f.kv.block_table(2, 2), whole / b) == head(t2, whole / b));
        f.kv.free_sequence(1);
        f.kv.free_sequence(2);
        CHECK_EQ(f.kv.block_refcount(1, t1[0]), 1);
        CHECK_EQ(f.pc.unshared_tokens(), whole);

        /* Eviction takes the deepest entry: its blocks in both groups go back together. */
        const int64_t e1 = f.kv.free_blocks(1), e2 = f.kv.free_blocks(2);
        CHECK_EQ(f.pc.evict_lru(1), 1ll);
        CHECK_EQ(f.kv.free_blocks(1), e1 + L / a);
        CHECK_EQ(f.kv.free_blocks(2), e2 + L / b);
        CHECK_EQ(f.pc.lookup(longer, {}).n_attn_tokens, (int32_t)(whole - L));
        CHECK_EQ(f.pc.unshared_tokens(), whole - L);
        /* trim() counts entries too. */
        CHECK_EQ(f.pc.trim(2), 2ll);
        CHECK_EQ(f.kv.free_blocks(1), e1 + 3 * (L / a));
        CHECK_EQ(f.kv.free_blocks(2), e2 + 3 * (L / b));
        /* A restore's room is in entries: the group shortest of whole entries decides. */
        const int64_t room = std::min((f.kv.total_blocks(1) - f.kv.held_blocks(1)) / (L / a),
                                      (f.kv.total_blocks(2) - f.kv.held_blocks(2)) / (L / b));
        CHECK_EQ(f.pc.restore_room(), room);
        f.pc.clear();
        CHECK_EQ(f.kv.free_blocks(1), free1);
        CHECK_EQ(f.kv.free_blocks(2), free2);
    }
}

/* A publish whose tables are short of an entry's blocks in one group is refused whole, and one
 * that cannot take a reference in a later group gives back what it took in the earlier ones. */
TEST(a_mixed_publish_short_of_a_groups_blocks_is_refused_and_takes_nothing) {
    Fixture f;
    REQUIRE_EQ(f.build_groups(mixed_groups(2, 1, false)), RAD_OK);
    const auto p = tok_run(40, 32);
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 32));
    std::vector<std::vector<int32_t>> bt = { f.kv.block_table(1, 1), f.kv.block_table(1, 2) };
    std::vector<std::vector<int32_t>> shorter = bt;
    shorter[1].resize(31);                           /* 16 entries need 32 one-token blocks */
    CHECK_EQ(f.pc.insert(1, p, {}, shorter, 32), RAD_E_INVAL);
    CHECK_EQ(f.pc.size(), 0ll);
    /* A free block in the fine group cannot be retained: the entry is unwound. */
    std::vector<std::vector<int32_t>> bad = bt;
    int32_t loose = -1;
    CHECK_OK(f.kv.reserve_block(2, &loose));
    f.kv.release_block(2, loose);                    /* back on the free list */
    bad[1][1] = loose;
    CHECK_EQ(f.pc.insert(1, p, {}, bad, 32), RAD_E_STATE);
    CHECK_EQ(f.kv.block_refcount(1, bt[0][0]), 1);
    CHECK_EQ(f.kv.block_refcount(2, bt[1][0]), 1);
    CHECK_OK(f.pc.insert(1, p, {}, bt, 32));
    CHECK_EQ(f.kv.block_refcount(2, bt[1][0]), 2);
}

/* Checkpoints are keyed on the entry grid: the chain at the lcm of the blocks, as the scheduler's
 * binding builds it (PCBinding::prefix_hash). A position off the grid has no key. */
TEST(a_checkpoint_over_mixed_blocks_sits_on_the_entry_grid) {
    for (const auto& ab : { std::pair<int64_t, int64_t>{ 2, 1 }, { 2, 4 } }) {
        const int64_t L = std::lcm(ab.first, ab.second);
        Fixture f;
        f.cfg.checkpoint_slots = 4;
        REQUIRE_EQ(f.build_groups(mixed_groups(ab.first, ab.second, true)), RAD_OK);
        CHECK_EQ(f.pc.checkpoint_interval(), 64ll);
        const auto prompt = tok_run(7000, 202);
        publish(f, 1, prompt, 202);
        const std::vector<BlockHash> hs = chain_of(prompt, (int)L);
        const int64_t tail = 198 / L * L;
        for (int64_t pos : { (int64_t)64, tail }) {
            int32_t slot = -1;
            CHECK_OK(f.pc.reserve_checkpoint(1, pos, hs[(size_t)(pos / L - 1)], &slot));
            f.pc.commit_checkpoint(hs[(size_t)(pos / L - 1)]);
        }
        auto longer = prompt;
        for (int i = 0; i < 30; ++i) longer.push_back(80000 + i);
        CacheHit h = f.pc.lookup(longer, {});
        CHECK_EQ(h.n_attn_tokens, (int32_t)(202 / L * L));
        CHECK_EQ(h.n_linear_tokens, (int32_t)tail);
        CHECK(h.checkpoint_slot >= 0);
        /* Diverging below the tail resumes at the interval snapshot. */
        auto edited = prompt;
        edited[150] += 9;
        CHECK_EQ(f.pc.lookup(edited, {}).n_linear_tokens, 64);
        /* Off the grid: refused. */
        int32_t slot = -1;
        CHECK_EQ(f.pc.reserve_checkpoint(1, L + 1, hs[0], &slot), RAD_E_INVAL);

        /* The scheduler's binding keys a snapshot on the same chain. */
        PCBinding<PrefixCache> bind(&f.pc);
        Request r;
        r.id = 5;
        r.prompt = longer;
        CHECK_OK(bind.reserve_checkpoint(r, 128, &slot));
        bind.commit_checkpoint(r, 128);
        auto at128 = prompt;
        for (size_t k = 130; k < at128.size(); ++k) at128[k] += 7;
        CHECK_EQ(f.pc.lookup(at128, {}).n_linear_tokens, 128);
    }
}

/* The interval must be a multiple of an entry; the chunk quantum, which the engine rounds it to,
 * is a multiple of every paged group's block and so of the lcm. */
TEST(a_mixed_cache_refuses_an_interval_off_its_entry_grid) {
    Fixture f;
    f.cfg.checkpoint_interval = 66;                  /* a multiple of 2, not of lcm(2, 4) */
    f.cfg.checkpoint_slots = 2;
    CHECK_OK(configure_pools(f.pools, f.cfg));
    CHECK_OK(f.kv.configure(mixed_groups(2, 4, true), f.cfg, f.pools.kv()));
    log_set_level(Log::Error);
    CHECK_EQ(f.pc.configure(f.cfg, &f.kv, f.cfg.checkpoint_interval), RAD_E_INVAL);
    log_set_level(Log::Info);
    CHECK_OK(f.pc.configure(f.cfg, &f.kv, 68));
    CHECK_EQ(f.pc.block_size(), 4ll);
}

/* A model with linear state beside a trunk window and groups at two strides (DeepSeek-V4.1's
 * shape): the cache indexes both full groups, and the window rides in the checkpoint slot after
 * the state, so a hit resumes at a snapshot with nothing left out. */
TEST(a_hybrid_with_a_window_and_mixed_blocks_leaves_only_the_window_to_the_slot) {
    Fixture f;
    f.cfg.checkpoint_slots = 4;
    std::vector<KVGroupInfo> g = mixed_groups(2, 1, true);
    g.push_back(paged_group("swa", RAD_KV_WINDOW, 2, 16, { 5 }, 64));     /* handle 4 */
    REQUIRE_EQ(f.build_groups(g), RAD_OK);
    CHECK(f.kv.has_window_snapshot());
    const KVGroupPlan* w = f.kv.plan(4);
    REQUIRE(w && w->win_snap);
    CHECK_EQ(f.kv.checkpoint_bytes(),
             f.kv.plan(3)->bytes_per_state + w->window_blocks * w->bytes_per_block);
    CHECK(f.pc.cached_groups() == (std::vector<int32_t>{ 1, 2 }));
    CHECK_EQ(f.pc.block_size(), 2ll);
    CHECK(f.pc.report().find("a checkpoint slot carries its window") != std::string::npos);
}

/* ================================================================== retention */

TEST(geometric_backoff_keeps_the_tip_and_its_ancestors) {
    GeometricBackoff g(64);
    std::vector<int64_t> pos;
    for (int i = 1; i <= 16; ++i) pos.push_back(i * 64);     /* 64 .. 1024 */
    std::vector<int64_t> keep;
    g.select(pos, &keep);
    /* Per scale, the shallowest checkpoint of the tip's bucket and of the one before it: 960 at
     * 64 tokens, 896 at 128, 768 at 256, 512 at 512 and 64 from 1024 up, then the tip. */
    std::vector<int64_t> want = { 64, 512, 768, 896, 960, 1024 };
    CHECK_EQ((long long)keep.size(), (long long)want.size());
    for (size_t i = 0; i < want.size() && i < keep.size(); ++i) CHECK_EQ(keep[i], want[i]);
    /* What it keeps it keeps again. */
    std::vector<int64_t> again;
    g.select(keep, &again);
    CHECK(again == keep);

    /* log(n): a 128K session at a 2048-token interval has 64 checkpoints and keeps 8. */
    GeometricBackoff g2(2048);
    std::vector<int64_t> big;
    for (int i = 1; i <= 64; ++i) big.push_back(i * 2048);
    g2.select(big, &keep);
    CHECK_EQ((long long)keep.size(), 8ll);
    CHECK_EQ(keep.front(), 2048ll);
    CHECK_EQ(keep.back(), 64ll * 2048);                       /* the tip is always kept */

    /* A tail off the interval is the tip; an earlier turn's tail between interval checkpoints
     * is not the shallowest of any bucket and goes. */
    std::vector<int64_t> tails = { 64, 128, 150, 192, 230 };
    g.select(tails, &keep);
    CHECK(keep == (std::vector<int64_t>{ 64, 128, 192, 230 }));

    /* Alternative policies are classes (spec §19.1). */
    KeepTipOnly t; t.select(big, &keep);
    CHECK_EQ((long long)keep.size(), 1ll);
    CHECK_EQ(keep[0], 64ll * 2048);
    KeepAll a; a.select(big, &keep);
    CHECK_EQ((long long)keep.size(), 64ll);
}

/* Retention runs each time the store fills, so the policy is applied again and again to what it
 * kept before. The ladder survives that: a prompt of 37 intervals and a tail, checkpointed into 8
 * slots one interval at a time, still resumes a turn that diverges anywhere in it less than twice
 * the divergence's depth below it (or two intervals, near the tail). */
TEST(geometric_backoff_reaches_deep_through_repeated_retention) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/8));
    const int64_t tip = 37 * 64 + 48;
    const auto prompt = tok_run(4242, (int)tip + 16);
    const std::vector<BlockHash> hs = chain_of(prompt);
    for (int i = 1; i <= 37; ++i) ckpt_at(f, 1, hs, i * 64);
    ckpt_at(f, 1, hs, tip);
    const std::vector<int64_t> held = f.pc.checkpoints(1);
    CHECK((long long)held.size() <= 8ll);
    REQUIRE(!held.empty());
    CHECK_EQ(held.back(), tip);

    for (int64_t diverge = 64; diverge < tip; diverge += 16) {
        int64_t below = 0;
        for (int64_t p : held) if (p <= diverge) below = std::max(below, p);
        const int64_t depth = tip - diverge;
        REQUIRE(below > 0);
        CHECK(diverge - below < std::max<int64_t>(2 * depth, 2 * 64));
    }
}

TEST(checkpoint_retention_is_budgeted_and_keeps_the_tip) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/8));

    auto prompt = tok_run(4242, 16 * 64);
    const std::vector<BlockHash> hs = chain_of(prompt);
    /* Eight checkpoints at 64, 128, ... 512 fill the pool exactly, one step each. */
    for (int i = 1; i <= 8; ++i) {
        f.pc.begin_step();
        int32_t s = -1;
        CHECK_OK(f.pc.reserve_checkpoint(1, i * 64, hs[i * 4 - 1], &s));
    }
    CHECK_EQ((long long)f.pc.checkpoints(1).size(), 8ll);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 0ll);

    /* The ninth forces retention. Geometric backoff over {64..512} keeps 512, 448, 384, 256 and
     * 64 and releases the rest; the tip survives, as an appending agent transcript needs. */
    f.pc.begin_step();
    int32_t s9 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(1, 9 * 64, hs[9 * 4 - 1], &s9));
    auto held = f.pc.checkpoints(1);
    CHECK(held.size() <= 6);
    CHECK_EQ(held.back(), 9ll * 64);
    bool has_512 = false;
    for (int64_t p : held) if (p == 512) has_512 = true;
    CHECK(has_512);
    CHECK_EQ(f.pc.stats().ckpt_retired, 3ll);
    CHECK_EQ(f.pc.stats().ckpt_lru, 0ll);
}

/* A snapshot handed out by a lookup keeps its slot until the next step begins. Its admission
 * restores from the slot ahead of the step, while a later admission in the same step may reserve a
 * checkpoint, which on a full store runs retention and then evicts; taking the handed snapshot
 * would leave the restore naming a released slot. Geometric backoff over {64..512} releases 192,
 * so the hit is made to land there. */
TEST(a_snapshot_handed_out_this_step_survives_retention) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/8));

    const auto a = tok_run(4242, 16 * 33);
    publish(f, 1, a, 16 * 32);
    const std::vector<BlockHash> hs = chain_of(a);
    std::vector<int32_t> slot_at(9, -1);
    for (int i = 1; i <= 8; ++i) slot_at[(size_t)i] = ckpt_at(f, 1, hs, i * 64);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 0ll);

    /* A prompt leaving the chain after 200 tokens: the deepest snapshot under it is 192. */
    f.pc.begin_step();
    auto b = a;
    b.resize(16 * 20);
    for (int i = 200; i < (int)b.size(); ++i) b[(size_t)i] = 900000 + i;
    std::vector<std::vector<int32_t>> hit;
    const CacheHit h = f.pc.lookup(b, {}, &hit);
    REQUIRE_EQ(h.n_linear_tokens, 192);
    REQUIRE_EQ(h.checkpoint_slot, slot_at[3]);

    /* The same step's next reservation finds the store full. */
    BlockHash other = hs[3];
    other.lo ^= 0x5eedull;
    int32_t s9 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(2, 64, other, &s9));
    CHECK(s9 >= 0);
    CHECK(s9 != h.checkpoint_slot);
    CHECK(f.kv.checkpoint_held(h.checkpoint_slot));
    bool kept_192 = false;
    for (int64_t p : f.pc.checkpoints(1)) if (p == 192) kept_192 = true;
    CHECK(kept_192);
}

/* Eviction after retention also skips it. With one slot, the handed snapshot is the only
 * candidate, so the reservation is refused: that chunk goes without a checkpoint and the restore
 * keeps its state. */
/* Retention places its ladder from the tip, but a turn that resumed below the tip left the chain
 * there, and its next turn resumes there again. That snapshot is kept past the step until the
 * session writes a newer tip. Geometric backoff over {64..512} would release 192. */
TEST(a_snapshot_a_lookup_resumed_from_outlives_retention_until_a_newer_tip) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/16));
    const auto a = tok_run(4242, 16 * 41);
    publish(f, 1, a, 16 * 40);
    const std::vector<BlockHash> hs = chain_of(a);
    for (int i = 1; i <= 8; ++i) ckpt_at(f, 1, hs, i * 64);

    auto b = a;
    b.resize(16 * 20);
    for (int i = 200; i < (int)b.size(); ++i) b[(size_t)i] = 900000 + i;
    f.pc.begin_step();
    REQUIRE_EQ(f.pc.lookup(b, {}).n_linear_tokens, 192);
    f.pc.begin_step();
    f.pc.retain(1);
    auto held = f.pc.checkpoints(1);
    CHECK(std::find(held.begin(), held.end(), 192) != held.end());
    CHECK(std::find(held.begin(), held.end(), 128) == held.end());

    ckpt_at(f, 1, hs, 576);
    f.pc.retain(1);
    held = f.pc.checkpoints(1);
    CHECK(std::find(held.begin(), held.end(), 192) == held.end());
    CHECK_EQ(held.back(), 576ll);
}

/* The store full of tips and one snapshot every turn resumes from: the tips go first, oldest
 * first, since a hand-out is a use and a tip has no other protection. */
TEST(a_full_store_evicts_the_least_recently_used_snapshot_tip_or_not) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    f.pc.set_policy(std::make_unique<KeepAll>());
    const auto a = tok_run(5151, 16 * 9);
    publish(f, 1, a, 16 * 8);
    const std::vector<BlockHash> hs = chain_of(a);
    ckpt_at(f, 1, hs, 64);                    /* the anchor, an interior snapshot of session 1 */
    ckpt_at(f, 1, hs, 128);                   /* session 1's tip */
    for (uint64_t sess = 2; sess <= 3; ++sess) {
        BlockHash h = hs[7];
        h.lo ^= sess;
        int32_t s = -1;
        CHECK_OK(f.pc.reserve_checkpoint(sess, 128, h, &s));   /* each its session's tip */
    }
    f.pc.begin_step();
    REQUIRE_EQ(f.pc.lookup(a, {}).n_linear_tokens, 128);
    auto p = a;
    for (int i = 70; i < (int)p.size(); ++i) p[(size_t)i] += 77777;
    f.pc.begin_step();
    REQUIRE_EQ(f.pc.lookup(p, {}).n_linear_tokens, 64);

    /* Sessions 2 and 3's tips are the least recently used; the anchor and session 1's tip were
     * handed out since. */
    f.pc.begin_step();
    BlockHash n1 = hs[3]; n1.lo ^= 0x77ull;
    BlockHash n2 = hs[3]; n2.lo ^= 0x78ull;
    int32_t s = -1;
    CHECK_OK(f.pc.reserve_checkpoint(9, 64, n1, &s));
    CHECK_OK(f.pc.reserve_checkpoint(10, 64, n2, &s));
    CHECK(f.pc.checkpoints(2).empty());
    CHECK(f.pc.checkpoints(3).empty());
    CHECK(f.pc.checkpoints(1) == (std::vector<int64_t>{ 64, 128 }));
    /* Counted by reason: keep-all released nothing, the full store gave up two. */
    CHECK_EQ(f.pc.stats().ckpt_retired, 0ll);
    CHECK_EQ(f.pc.stats().ckpt_lru, 2ll);
    CHECK_EQ(f.pc.stats().ckpt_evictions, 2ll);
}

TEST(a_snapshot_handed_out_this_step_is_not_evicted) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/1));
    const auto a = tok_run(5151, 16 * 9);
    publish(f, 1, a, 16 * 8);
    const std::vector<BlockHash> hs = chain_of(a);
    const int32_t s1 = ckpt_at(f, 1, hs, 64);

    f.pc.begin_step();
    std::vector<std::vector<int32_t>> hit;
    const CacheHit h = f.pc.lookup(a, {}, &hit);
    REQUIRE_EQ(h.checkpoint_slot, s1);

    int32_t s2 = -1;
    CHECK_EQ(f.pc.reserve_checkpoint(1, 128, hs[7], &s2), RAD_E_FULL);
    CHECK(f.kv.checkpoint_held(s1));
    CHECK_EQ(f.pc.stats().ckpt_evictions, 0ll);
    CHECK_EQ(f.pc.stats().ckpt_refused, 1ll);

    f.pc.begin_step();
    CHECK_OK(f.pc.reserve_checkpoint(1, 128, hs[7], &s2));
}

/* A reservation made this step is a save target: the engine copies the state into its slot
 * after the step, and nothing checks the slot again before that copy. A later reservation in the
 * same step that finds the store full runs retention over every session, and the policy, placing
 * its ladder from each session's tip, may release the very slot just handed out -- a second
 * request's reservation lands on a snapshot a first session already holds (the same prefix, so
 * the same key), and that snapshot is below the first session's tip. Geometric backoff over
 * {64..512} releases 192, so the shared reservation is made to land there. The save then faults
 * ("the store does not hold that slot") and the step loop stops the engine. A reservation keeps
 * its slot until its commit, or until the next step begins, as a handed-out snapshot does. */
TEST(a_reservation_this_step_survives_retention) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/8));

    const auto a = tok_run(4242, 16 * 33);
    publish(f, 1, a, 16 * 32);
    const std::vector<BlockHash> hs = chain_of(a);
    std::vector<int32_t> slot_at(9, -1);
    for (int i = 1; i <= 8; ++i) slot_at[(size_t)i] = ckpt_at(f, 1, hs, i * 64);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 0ll);

    /* Session 2 prefilling the same prompt reserves at 192: the same key, so session 1's slot. */
    f.pc.begin_step();
    int32_t s2 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(2, 192, hs[11], &s2));
    CHECK_EQ(s2, slot_at[3]);

    /* The same step's next reservation finds the store full: retention runs over session 1,
     * whose ladder does not keep 192. */
    BlockHash other = hs[3];
    other.lo ^= 0x5eedull;
    int32_t s3 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(3, 64, other, &s3));
    CHECK(s3 >= 0);
    CHECK(s3 != s2);
    CHECK(f.kv.checkpoint_held(s2));
    bool kept_192 = false;
    for (int64_t p : f.pc.checkpoints(1)) if (p == 192) kept_192 = true;
    CHECK(kept_192);
    CHECK(f.pc.stats().ckpt_retired >= 1ll);

    /* The next step may release it again: an uncommitted reservation whose save was skipped (its
     * request gone) must not hold a slot for good. (A committed one is a snapshot like any other
     * from the commit on, tested below.) */
    f.pc.begin_step();
    for (int i = 0; i < 3; ++i) {
        BlockHash fresh = hs[3];
        fresh.lo ^= 0x1000ull + (uint64_t)i;
        int32_t s = -1;
        CHECK_OK(f.pc.reserve_checkpoint(10 + (uint64_t)i, 64, fresh, &s));
    }
    bool still_192 = false;
    for (int64_t p : f.pc.checkpoints(1)) if (p == 192) still_192 = true;
    CHECK(!still_192);
}

/* Eviction after retention skips it too. With one slot, the reservation is the only candidate,
 * so a second reservation in the same step is refused instead of taking it; once committed, it
 * is a snapshot like any other and the next step's reservation may take the slot. */
TEST(a_reservation_this_step_is_not_evicted) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/1));
    const auto a = tok_run(5151, 16 * 9);
    const std::vector<BlockHash> hs = chain_of(a);

    f.pc.begin_step();
    int32_t s1 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(1, 64, hs[3], &s1));
    CHECK(s1 >= 0);

    BlockHash other = hs[3];
    other.lo ^= 0x5eedull;
    int32_t s2 = -1;
    CHECK_EQ(f.pc.reserve_checkpoint(2, 64, other, &s2), RAD_E_FULL);
    CHECK(f.kv.checkpoint_held(s1));
    CHECK_EQ(f.pc.stats().ckpt_evictions, 0ll);
    CHECK_EQ(f.pc.stats().ckpt_refused, 1ll);

    f.pc.commit_checkpoint(hs[3]);
    f.pc.begin_step();
    CHECK_OK(f.pc.reserve_checkpoint(2, 64, other, &s2));
    CHECK_EQ(s2, s1);
}

TEST(a_dead_sessions_tip_is_evictable_so_the_slot_pool_cannot_wedge) {
    /* One checkpoint each from more distinct sessions than slots, each its session's tip. If tips
     * were unevictable the pool would wedge and every later request get RAD_E_FULL. A session is a
     * request id; requests end but their checkpoints stay (keyed by prefix hash for the next turn),
     * so a tip is not necessarily live. */
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/8));

    auto prompt = tok_run(909, 16 * 8);
    const std::vector<BlockHash> hs = chain_of(prompt);

    /* Each under a distinct prefix, so each takes its own slot; eight sessions fill eight
     * slots. */
    for (uint64_t sess = 1; sess <= 8; ++sess) {
        BlockHash h = hs[3];
        h.lo ^= sess;
        int32_t s = -1;
        CHECK_OK(f.pc.reserve_checkpoint(sess, 64, h, &s));
        CHECK(s >= 0);
    }
    CHECK_EQ(f.kv.free_checkpoint_slots(), 0ll);

    /* The ninth session, a step later, is served by retiring the least recently used tip. */
    f.pc.begin_step();
    BlockHash fresh = hs[3];
    fresh.lo ^= 0x5eedull;
    int32_t s9 = -1;
    CHECK_OK(f.pc.reserve_checkpoint(99, 64, fresh, &s9));
    CHECK(s9 >= 0);
    /* The evicted session is forgotten: eight in, one evicted, one added, eight tracked. An index
     * that kept emptied keys would grow per request forever, and reserve_checkpoint walks every key
     * on an allocation failure. */
    CHECK_EQ(f.pc.sessions_tracked(), 8ll);
}

TEST(a_sessions_index_entry_does_not_outlive_its_last_snapshot) {
    /* A session emptied by retention rather than eviction. Drive one session past the slot budget
     * under keep-tip-only (which releases all but the tip on the first squeeze), then evict that
     * tip via other sessions. No index entry may remain. */
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    f.pc.set_policy(std::make_unique<KeepTipOnly>());

    auto prompt = tok_run(313, 16 * 32);
    const std::vector<BlockHash> hs = chain_of(prompt);
    for (int i = 1; i <= 5; ++i) {
        f.pc.begin_step();
        int32_t s = -1;
        CHECK_OK(f.pc.reserve_checkpoint(7, i * 64, hs[i * 4 - 1], &s));
    }
    /* The squeeze happens on the fifth reservation: retention over {64,128,192,256} keeps 256,
     * then 320 is allocated, leaving two. */
    CHECK_EQ((long long)f.pc.checkpoints(7).size(), 2ll);
    CHECK_EQ(f.pc.sessions_tracked(), 1ll);

    /* Four more sessions, one snapshot each, a step apart: session 7's tip is the LRU and is
     * evicted. */
    for (uint64_t sess = 20; sess < 24; ++sess) {
        f.pc.begin_step();
        BlockHash h = hs[9];
        h.lo ^= sess;
        int32_t s = -1;
        CHECK_OK(f.pc.reserve_checkpoint(sess, 64, h, &s));
    }
    CHECK(f.pc.checkpoints(7).empty());
    CHECK_EQ(f.pc.sessions_tracked(), 4ll);
}

/* ================================================================== a rewritten last message
 *
 * Turn t sends the history plus a volatile block; turn t+1 sends the history, the reply, a new
 * message and a new block, so it leaves turn t's chain where the block began. Each turn looks up,
 * writes checkpoints where Scheduler::plan_chunk puts them (every interval past the resume point,
 * and the tail when it is a quarter interval past the last one), and publishes prompt ++ reply.
 * The store must keep a snapshot below each divergence for the next turn to resume from. */
struct RewriteTurn {
    int64_t prompt = 0;     /* tokens */
    int64_t diverge = 0;    /* where the next turn leaves this one */
    int64_t resumed = 0;    /* where this turn's lookup resumed the linear state */
};

/* One chunk's checkpoint, in a step after the one whose lookup resumed the turn. */
static void ckpt_try(Fixture& f, uint64_t session, const std::vector<BlockHash>& hs, int64_t pos) {
    int32_t slot = -1;
    const BlockHash& key = hs[(size_t)(pos / 16 - 1)];
    f.pc.begin_step();
    /* A refusal leaves that boundary without a snapshot, as in the scheduler. */
    if (f.pc.reserve_checkpoint(session, pos, key, &slot) == RAD_OK) f.pc.commit_checkpoint(key);
}

static std::vector<RewriteTurn> rewrite_conversation(Fixture& f, int turns, int64_t history,
                                                     int64_t block, int64_t grow) {
    const int64_t I = f.pc.checkpoint_interval(), bs = f.pc.block_size();
    std::vector<int32_t> hist = tok_run(1, (int)history);
    int32_t fresh = 1000000;               /* token ids no earlier turn sent */
    std::vector<RewriteTurn> out;
    for (int t = 0; t < turns; ++t) {
        std::vector<int32_t> prompt = hist;
        for (int64_t i = 0; i < block; ++i) prompt.push_back(fresh++);
        RewriteTurn rt;
        rt.prompt = (int64_t)prompt.size();
        rt.diverge = (int64_t)hist.size();
        const uint64_t id = (uint64_t)t + 1;
        f.pc.begin_step();
        rt.resumed = f.pc.lookup(prompt, {}, nullptr, nullptr, id).n_linear_tokens;

        std::vector<int32_t> published = prompt;
        for (int64_t i = 0; i < grow / 2; ++i) published.push_back(fresh++);   /* the reply */
        const std::vector<BlockHash> hs = chain_of(published);
        for (int64_t p = (rt.resumed / I + 1) * I; p < rt.prompt; p += I) ckpt_try(f, id, hs, p);
        const int64_t tail = rt.prompt / bs * bs;
        if (tail > I && tail % I >= std::max(bs, I / 4)) ckpt_try(f, id, hs, tail);
        publish(f, id, published, (int64_t)published.size() / bs * bs);
        f.kv.free_sequence(id);

        hist.assign(published.begin(), published.begin() + rt.diverge);
        hist.insert(hist.end(), published.begin() + rt.prompt, published.end());
        for (int64_t i = 0; i < grow - grow / 2; ++i) hist.push_back(fresh++);  /* next message */
        out.push_back(rt);
    }
    return out;
}

/* Every turn after the first resumes below where it left the previous one, less than twice the
 * rewritten tail plus two intervals below it. */
static void check_rewrites_resume(const std::vector<RewriteTurn>& turns, int64_t interval) {
    for (size_t t = 1; t < turns.size(); ++t) {
        const RewriteTurn& prev = turns[t - 1];
        const int64_t r = turns[t].resumed, tail = prev.prompt - prev.diverge;
        if (r <= 0 || r > prev.diverge || prev.diverge - r >= 2 * tail + 2 * interval) {
            fprintf(stderr, "    turn %zu resumed at %lld; it left turn %zu at %lld, %lld before "
                            "that turn's end\n", t, (long long)r, t - 1, (long long)prev.diverge,
                    (long long)tail);
            CHECK(false);
        }
    }
}

/* The shipped retention (geometric backoff, 8 slots), scaled: a 37-interval prompt whose last
 * 5.2 intervals are rewritten every turn. */
TEST(a_rewritten_last_message_resumes_every_turn_at_the_default_retention) {
    Fixture f;
    f.cfg.vram_kv_mib = 8;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/8));
    REQUIRE_EQ(f.pc.checkpoint_interval(), 64ll);
    const auto turns = rewrite_conversation(f, 12, 32 * 64, 336, 32);
    CHECK_EQ(turns[0].resumed, 0ll);
    check_rewrites_resume(turns, 64);
}

/* Keep-all with 16 slots, the deeper retention: each turn's tail stays its own session's tip and
 * no later turn reaches it, while every turn resumes from the same anchor, which must outlast the
 * tips. */
TEST(a_rewritten_last_message_keeps_its_anchor_over_the_tips_it_leaves) {
    Fixture f;
    f.cfg.vram_kv_mib = 8;
    f.cfg.checkpoint_interval = 256;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/16));
    f.pc.set_policy(std::make_unique<KeepAll>());
    const auto turns = rewrite_conversation(f, 40, 900, 80, 16);
    check_rewrites_resume(turns, 256);
}

/* ================================================================== agent sessions sharing the store
 *
 * Several agent conversations at once, as a coding client with subagents drives the server. Every
 * conversation starts from one system prompt. A parent grows turn by turn: a turn sends the whole
 * history plus a new message and publishes it with the reply. Every few turns a parent hands a
 * task to a subagent and waits for it: a short conversation from the same system prompt that runs
 * a few turns of its own. The conversations that are not waiting take their turns together: every
 * lookup first, then the checkpoints chunk by chunk across them, one step each, then the
 * publishes. Checkpoints go where Scheduler::plan_chunk puts them: every interval past the resume
 * point and the prompt's tail. Lookups name the request, as PCBinding does.
 *
 * Scaled from the shipped geometry: the 64-token interval stands for 2048, so a parent that stops
 * at 6272 tokens stands for a 200K-token conversation. */
struct AgentLoad {
    int     parents = 4;
    int64_t system = 384;               /* the system prompt every conversation starts with */
    int64_t parent_len = 6272;          /* a parent stops growing here */
    int64_t grow_lo = 32, grow_hi = 192;   /* tokens a turn adds: the reply and the next message */
    int     sub_every = 4;              /* parent turns between subagents */
    int     sub_turns = 5;              /* turns a subagent runs */
    int64_t task = 64;                  /* the subagent's first message */
    bool    rewrite = false;            /* a parent's next turn drops the message this turn
                                           answered, as a client that rewrites it does */
};

enum AgentTurn { kParent, kParentAfterSub, kSubFirst, kSubLater };

struct AgentTally {
    int64_t turns = 0;
    int64_t prompt = 0;                 /* prompt tokens sent */
    int64_t computed = 0;               /* prompt tokens computed, past the resume point */
    std::vector<int64_t> resumed;       /* per turn, in issue order */
    std::vector<AgentTurn> kind;
};

static AgentTally run_agents(Fixture& f, const AgentLoad& L, uint32_t seed = 7) {
    const int64_t I = f.pc.checkpoint_interval(), bs = f.pc.block_size();
    uint64_t rng = 0x9e3779b97f4a7c15ull ^ seed;
    auto rnd = [&](int64_t lo, int64_t hi) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return lo + (int64_t)(rng % (uint64_t)(hi - lo + 1));
    };
    int32_t fresh = 1000000;
    auto add = [&](std::vector<int32_t>* v, int64_t n) {
        for (int64_t i = 0; i < n; ++i) v->push_back(fresh++);
    };
    const std::vector<int32_t> sys = tok_run(1, (int)L.system);

    struct Conv {
        std::vector<int32_t> hist;
        int  turns = 0;            /* turns the parent has taken */
        int  left = 0;             /* the running subagent's turns to go; 0 when none runs */
        bool after_sub = false;    /* the parent's next turn follows its subagent */
        bool done = false;
        std::vector<int32_t> sub;  /* the running subagent's history */
    };
    std::vector<Conv> cs((size_t)L.parents);
    for (Conv& c : cs) { c.hist = sys; add(&c.hist, L.task); }

    struct Turn {
        std::vector<int32_t> prompt;
        std::vector<int32_t>* hist = nullptr;
        uint64_t id = 0;
        int64_t grow = 0, resumed = 0, keep = -1;
        AgentTurn kind = kParent;
        std::vector<int64_t> at;            /* checkpoint positions, in chunk order */
        std::vector<BlockHash> hs;          /* the published chain */
    };
    AgentTally t;
    uint64_t next_id = 1;
    for (;;) {
        std::vector<Turn> turns;
        for (Conv& c : cs) {
            if (c.done) continue;
            Turn tn;
            tn.id = next_id++;
            tn.grow = rnd(L.grow_lo, L.grow_hi);
            if (c.left > 0) {
                tn.hist = &c.sub;
                tn.kind = c.left == L.sub_turns ? kSubFirst : kSubLater;
            } else {
                if ((int64_t)c.hist.size() >= L.parent_len) { c.done = true; continue; }
                tn.hist = &c.hist;
                tn.kind = c.after_sub ? kParentAfterSub : kParent;
                tn.keep = L.rewrite ? (int64_t)c.hist.size() : -1;
                c.after_sub = false;
            }
            tn.prompt = *tn.hist;
            add(&tn.prompt, tn.grow / 2);
            turns.push_back(std::move(tn));
        }
        if (turns.empty()) break;

        f.pc.begin_step();
        for (Turn& tn : turns) {
            tn.resumed = f.pc.lookup(tn.prompt, {}, nullptr, nullptr, tn.id).n_linear_tokens;
            const int64_t n = (int64_t)tn.prompt.size();
            for (int64_t p = (tn.resumed / I + 1) * I; p < n; p += I) tn.at.push_back(p);
            const int64_t tail = n / bs * bs;
            if (tail > I && tail % I >= std::max(bs, I / 4)) tn.at.push_back(tail);
            t.turns++;
            t.prompt += n;
            t.computed += n - tn.resumed;
            t.resumed.push_back(tn.resumed);
            t.kind.push_back(tn.kind);
        }
        /* The reply comes after the prompt: no checkpoint lands in it. */
        for (Turn& tn : turns) {
            std::vector<int32_t> pub = tn.prompt;
            add(&pub, tn.grow - tn.grow / 2);
            tn.hs = chain_of(pub);
            *tn.hist = std::move(pub);
        }
        /* The first chunk is planned in the admitting step, so its reservation cannot take a
         * snapshot a lookup just handed out. */
        for (size_t k = 0;; ++k) {
            bool any = false;
            if (k > 0) f.pc.begin_step();
            for (Turn& tn : turns) {
                if (k >= tn.at.size()) continue;
                any = true;
                int32_t slot = -1;
                const BlockHash& key = tn.hs[(size_t)(tn.at[k] / bs - 1)];
                if (f.pc.reserve_checkpoint(tn.id, tn.at[k], key, &slot) == RAD_OK)
                    f.pc.commit_checkpoint(key);
            }
            if (!any) break;
        }
        for (Turn& tn : turns) {
            publish(f, tn.id, *tn.hist, (int64_t)tn.hist->size() / bs * bs);
            f.kv.free_sequence(tn.id);
            /* A rewriting client sends the reply back without the message it answered. */
            if (tn.keep >= 0)
                tn.hist->erase(tn.hist->begin() + tn.keep,
                               tn.hist->begin() + (int64_t)tn.prompt.size());
        }
        /* Subagents start and finish between rounds; a finished one's answer joins its parent. */
        for (Conv& c : cs) {
            if (c.done) continue;
            if (c.left > 0) {
                if (--c.left == 0) {
                    add(&c.hist, rnd(L.grow_lo, L.grow_hi));
                    c.after_sub = true;
                }
                continue;
            }
            if (++c.turns % L.sub_every == 0) {
                c.sub = sys;
                add(&c.sub, L.task);
                c.left = L.sub_turns;
            }
        }
    }
    return t;
}

/* The same load against an unbounded keep-all store: every turn resumes as deep as checkpoints
 * were written. */
static AgentTally agents_unbounded(const AgentLoad& L) {
    Fixture f;
    f.cfg.vram_kv_mib = 128;
    if (f.build(/*with_linear=*/true, /*ckpt_slots=*/8192) != RAD_OK) return {};
    f.pc.set_policy(std::make_unique<KeepAll>());
    return run_agents(f, L);
}

/* The load against a store of `slots` under the default retention. */
static AgentTally agents_in(const AgentLoad& L, int64_t slots) {
    Fixture f;
    f.cfg.vram_kv_mib = 128;
    if (f.build(/*with_linear=*/true, slots) != RAD_OK) return {};
    return run_agents(f, L);
}

/* Turns of `kind` that resumed less than half as deep as the unbounded store let them: their
 * session's snapshots were gone. */
static int64_t agents_lost(const AgentTally& a, const AgentTally& best, AgentTurn kind) {
    int64_t n = 0;
    for (size_t i = 0; i < a.resumed.size() && i < best.resumed.size(); ++i)
        if (a.kind[i] == kind && a.resumed[i] * 2 < best.resumed[i]) ++n;
    return n;
}

/* Prompt tokens computed beyond the unbounded store's, as a fraction of all prompt tokens. */
static double agents_excess(const AgentTally& a, const AgentTally& best) {
    return best.prompt > 0 ? (double)(a.computed - best.computed) / (double)best.prompt : 1.0;
}

/* The store a configuration gives the load: its count, or under auto what the engine resolves for
 * one sequence a parent, with memory to spare. */
static int64_t agents_store(const Config& c, const AgentLoad& L, int64_t interval) {
    if (c.checkpoint_slots >= 0) return c.checkpoint_slots;
    const auto pol = make_checkpoint_policy(c.checkpoint_policy, interval);
    return checkpoint_slots_auto(*pol, L.parents, L.parent_len, interval, 1, INT64_MAX, INT64_MAX)
        .slots;
}

/* An admission takes over every snapshot on the chain it hit, so a conversation is one session
 * from the start of its turn, as it is after the publish. A snapshot past the hit stays where it
 * was; a lookup naming no request moves nothing. */
TEST(an_admission_takes_over_the_snapshots_on_the_chain_it_hit) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/8));
    const auto p1 = tok_run(7000, 336);
    publish(f, 1, p1, 320);
    const std::vector<BlockHash> hs = chain_of(p1);
    ckpt_at(f, 1, hs, 128);
    ckpt_at(f, 1, hs, 256);

    auto p2 = p1;
    for (size_t k = 200; k < p2.size(); ++k) p2[k] += 50000;
    f.pc.begin_step();
    CHECK_EQ(f.pc.lookup(p2, {}).n_linear_tokens, 128);
    CHECK(f.pc.checkpoints(1) == (std::vector<int64_t>{ 128, 256 }));
    f.pc.begin_step();
    CHECK_EQ(f.pc.lookup(p2, {}, nullptr, nullptr, 2).n_linear_tokens, 128);
    CHECK(f.pc.checkpoints(2) == (std::vector<int64_t>{ 128 }));
    CHECK(f.pc.checkpoints(1) == (std::vector<int64_t>{ 256 }));
    CHECK_EQ(f.pc.sessions_tracked(), 2ll);
}

/* A running turn's checkpoints and its conversation's earlier ones are retained as one ladder, so
 * a full store does not take another conversation's snapshot for them. Conversation P holds one
 * snapshot, the oldest in the store; A holds seven and its next turn writes five more. As two
 * sessions (the turn before and the turn running) A keeps two ladders and the store evicts P's
 * snapshot; as one it keeps one ladder and P's survives. */
TEST(a_running_turn_does_not_evict_a_waiting_conversations_snapshot) {
    for (const bool named : { false, true }) {
        Fixture f;
        CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/8));
        const auto p = tok_run(500000, 96);
        const std::vector<BlockHash> ph = chain_of(p);
        ckpt_at(f, 100, ph, 64);

        const auto a = tok_run(1, 16 * 60);
        const std::vector<BlockHash> ah = chain_of(a);
        std::vector<int32_t> turn1(a.begin(), a.begin() + 464);
        publish(f, 1, turn1, 464);
        f.kv.free_sequence(1);
        for (int64_t pos = 64; pos <= 448; pos += 64) ckpt_at(f, 1, ah, pos);
        CHECK_EQ(f.kv.free_checkpoint_slots(), 0ll);

        std::vector<int32_t> turn2(a.begin(), a.begin() + 800);
        f.pc.begin_step();
        REQUIRE_EQ(f.pc.lookup(turn2, {}, nullptr, nullptr, named ? 2 : 0).n_linear_tokens, 448);
        for (int64_t pos = 512; pos <= 768; pos += 64) {
            int32_t slot = -1;
            CHECK_OK(f.pc.reserve_checkpoint(2, pos, ah[(size_t)(pos / 16 - 1)], &slot));
            f.pc.commit_checkpoint(ah[(size_t)(pos / 16 - 1)]);
            f.pc.begin_step();
        }
        if (named) {
            CHECK(f.pc.checkpoints(100) == (std::vector<int64_t>{ 64 }));
            CHECK(f.pc.checkpoints(1).empty());
            CHECK(f.pc.checkpoints(2) ==
                  (std::vector<int64_t>{ 64, 256, 512, 576, 640, 704, 768 }));
            CHECK_EQ(f.pc.stats().ckpt_lru, 0ll);
        } else {
            CHECK(f.pc.checkpoints(100).empty());
            CHECK_EQ(f.pc.stats().ckpt_lru, 1ll);
        }
    }
}

/* What a session needs from the store, and that retention holds it to that: a 200K-token session
 * at a 2048-token interval keeps at most 9 snapshots under geometric backoff. The scaled session
 * (98 intervals of 64 tokens), checkpointed one interval at a time with a tail checkpoint half an
 * interval on after each and retention after every one, never holds more than its policy's figure
 * and reaches it. */
TEST(a_sessions_standing_share_is_what_retention_keeps) {
    CHECK_EQ(GeometricBackoff(2048).held(200000, 2048), 9ll);
    CHECK_EQ(GeometricBackoff(2048).held(32768, 2048), 6ll);
    CHECK_EQ(GeometricBackoff(2048).held(1000, 2048), 1ll);
    CHECK_EQ(KeepTipOnly().held(200000, 2048), 1ll);
    CHECK_EQ(KeepAll().held(200000, 2048), 97ll);
    CHECK(std::string(make_checkpoint_policy("tip-only", 64)->name()) == "tip-only");
    CHECK(std::string(make_checkpoint_policy("keep-all", 64)->name()) == "keep-all");
    CHECK(std::string(make_checkpoint_policy("geometric-backoff", 64)->name()) ==
          "geometric-backoff");

    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/128));
    const auto s = tok_run(1, 6272 + 16);
    const std::vector<BlockHash> hs = chain_of(s);
    size_t most = 0;
    for (int64_t pos = 64; pos <= 6272; pos += 64) {
        for (const int64_t at : { pos, pos + 32 }) {
            if (at > 6272) continue;
            ckpt_at(f, 1, hs, at);
            f.pc.retain(1);
            most = std::max(most, f.pc.checkpoints(1).size());
        }
    }
    CHECK_EQ((int64_t)most, GeometricBackoff(64).held(6272, 64));
    CHECK_EQ((int64_t)most, 9ll);
}

/* The reserve --host-headroom-mib charges a store's floor: it gives back exactly what the floor is
 * short, never below a gigabyte, never above what was asked for, and stays put when the floor
 * already fits. A byte rule, so a wider slot (any hybrid model's linear state) or a wider mirror
 * (any KV encoding) needs no second case. */
TEST(the_host_reserve_gives_way_to_a_stores_floor_and_no_further) {
    const int64_t gib = (int64_t)1 << 30, mib = (int64_t)1 << 20;
    const int64_t slot = 113 * mib, floor8 = 8 * slot, reserve = 4 * gib;

    /* The floor fits: the reserve never moves, so a host with room to spare sees nothing. */
    CHECK_EQ(host_reserve_for_floor(floor8, 20 * floor8, reserve), reserve);
    CHECK_EQ(host_reserve_for_floor(floor8, floor8, reserve), reserve);
    /* Short: the shortfall comes back in whole MiB, rounded up so the store is never a byte
     * short of its floor, and the rest of the reserve stays. */
    CHECK_EQ(host_reserve_for_floor(floor8, slot / 2, reserve), reserve - 848 * mib);
    CHECK_EQ(host_reserve_for_floor(floor8, floor8 - 1, reserve), reserve - 1 * mib);
    CHECK_EQ(host_reserve_for_floor(floor8, floor8 - mib, reserve), reserve - 1 * mib);
    CHECK_EQ(host_reserve_for_floor(floor8, floor8 - mib - 1, reserve), reserve - 2 * mib);
    /* A floor beyond both the spare memory and the reserve: down to the floor, never below. */
    CHECK_EQ(host_reserve_for_floor(64 * gib, 0, reserve), gib);
    /* Never above the margin asked for, so a small --host-headroom-mib is still a real bound. */
    CHECK_EQ(host_reserve_for_floor(floor8, 0, 512 * mib), 512 * mib);
    /* Nothing to protect. */
    CHECK_EQ(host_reserve_for_floor(0, 0, reserve), reserve);
    CHECK_EQ(host_reserve_for_floor(floor8, 0, 0), 0);

    /* The count a floor is taken from: the policy's hold plus a running turn, per sequence. */
    const GeometricBackoff g(2048);
    CHECK_EQ(checkpoint_slots_want(g, 8, 200000, 2048), 80ll);
    CHECK_EQ(checkpoint_slots_want(g, 1, 200000, 2048), 10ll);
    CHECK_EQ(checkpoint_slots_want(g, 0, 200000, 2048), 10ll);
    CHECK_EQ(checkpoint_slots_want(KeepTipOnly(), 1, 200000, 2048), 2ll);
}

/* --checkpoint-slots auto: the policy's share plus one for each sequence, held to the share of
 * memory but not below the floor, and to the room the host has. */
TEST(the_auto_store_gives_every_sequence_its_sessions_share) {
    const GeometricBackoff g(2048);
    const int64_t slot = 113 << 20, lots = INT64_MAX;
    CheckpointSlots c = checkpoint_slots_auto(g, 8, 200000, 2048, slot, lots, lots);
    CHECK_EQ(c.per_seq, 10ll);
    CHECK_EQ(c.want, 80ll);
    CHECK_EQ(c.slots, 80ll);
    CHECK(c.bound == CheckpointSlots::Bound::Want);

    c = checkpoint_slots_auto(g, 8, 200000, 2048, slot, 40 * slot + slot / 2, lots);
    CHECK_EQ(c.slots, 40ll);
    CHECK(c.bound == CheckpointSlots::Bound::Share);
    c = checkpoint_slots_auto(g, 8, 200000, 2048, slot, 2 * slot, lots);
    CHECK_EQ(c.slots, kCheckpointSlotsFloor);
    CHECK(c.bound == CheckpointSlots::Bound::Share);
    c = checkpoint_slots_auto(g, 8, 200000, 2048, slot, lots, 5 * slot);
    CHECK_EQ(c.slots, 5ll);
    CHECK(c.bound == CheckpointSlots::Bound::Room);
    c = checkpoint_slots_auto(g, 8, 200000, 2048, slot, 2 * slot, 3 * slot);
    CHECK_EQ(c.slots, 3ll);
    CHECK(c.bound == CheckpointSlots::Bound::Room);
    c = checkpoint_slots_auto(g, 8, 200000, 2048, slot, lots, slot - 1);
    CHECK_EQ(c.slots, 0ll);

    /* The floor never exceeds what the policy wants. */
    c = checkpoint_slots_auto(KeepTipOnly(), 1, 200000, 2048, slot, 0, lots);
    CHECK_EQ(c.want, 2ll);
    CHECK_EQ(c.slots, 2ll);
    c = checkpoint_slots_auto(KeepTipOnly(), 4, 200000, 2048, slot, lots, lots);
    CHECK_EQ(c.slots, 8ll);
    c = checkpoint_slots_auto(KeepAll(), 2, 200000, 2048, slot, lots, lots);
    CHECK_EQ(c.slots, 196ll);

    /* --max-num-seqs's own default, 256, and no state to snapshot. */
    c = checkpoint_slots_auto(g, 256, 200000, 2048, slot, (int64_t)8 << 30, lots);
    CHECK_EQ(c.want, 2560ll);
    CHECK_EQ(c.slots, ((int64_t)8 << 30) / slot);
    c = checkpoint_slots_auto(g, INT64_MAX, 200000, 2048, slot, lots, lots);
    CHECK_EQ(c.want, INT64_MAX);
    CHECK_EQ(c.slots, INT64_MAX / slot);
    c = checkpoint_slots_auto(g, 8, 200000, 2048, 0, lots, lots);
    CHECK_EQ(c.slots, 0ll);
}

/* Four agent conversations with subagents, scaled to 200K tokens each, in the store the default
 * configuration gives them. While a parent waits on its subagent, the subagent's snapshots fill
 * the store; in the fixed 8-slot store they evict all of the waiting parent's, and its next turn
 * prefills its whole history again. In the default store every continuing turn resumes where an
 * unbounded store would. What is left is the subagents' first turns: the system prompt they share
 * is past every snapshot the policy keeps. */
TEST(agent_sessions_with_subagents_keep_their_snapshots_in_the_default_store) {
    const AgentLoad L;
    const AgentTally best = agents_unbounded(L);
    REQUIRE(best.turns > 300);

    Config d;
    d.checkpoint_interval = 64;
    const int64_t slots = agents_store(d, L, 64);
    CHECK_EQ(slots, 40ll);
    const AgentTally a = agents_in(L, slots);
    const AgentTally fixed = agents_in(L, 8);
    fprintf(stderr, "    %lld turns; prompt tokens computed past an unbounded store's: %.1f%% in %lld "
            "slots, %.1f%% in 8\n", (long long)a.turns, 100 * agents_excess(a, best),
            (long long)slots, 100 * agents_excess(fixed, best));
    CHECK_EQ(agents_lost(a, best, kParent), 0ll);
    CHECK_EQ(agents_lost(a, best, kParentAfterSub), 0ll);
    CHECK_EQ(agents_lost(a, best, kSubLater), 0ll);
    CHECK(agents_excess(a, best) < 0.03);

    CHECK(agents_lost(fixed, best, kParentAfterSub) >= 30);
    CHECK(agents_excess(fixed, best) > 0.30);
}

/* The same with a client that rewrites its last message every turn, so a turn diverges below the
 * previous turn's tip and needs a snapshot behind it: the ladder must survive beside the
 * subagents. */
TEST(a_rewriting_client_with_subagents_resumes_in_the_default_store) {
    AgentLoad L;
    L.rewrite = true;
    const AgentTally best = agents_unbounded(L);
    REQUIRE(best.turns > 300);
    Config d;
    d.checkpoint_interval = 64;
    const AgentTally a = agents_in(L, agents_store(d, L, 64));
    fprintf(stderr, "    %lld turns; %.1f%% computed past an unbounded store's\n",
            (long long)a.turns, 100 * agents_excess(a, best));
    const int64_t lost = agents_lost(a, best, kParent) + agents_lost(a, best, kParentAfterSub) +
                         agents_lost(a, best, kSubLater);
    CHECK(lost <= 2);
    CHECK(agents_excess(a, best) < 0.03);
}

/* ================================================================== the KV loan
 *
 * The cache and the expert plane share one card, split once at startup against a worst case.
 * The cache lends the expert slab the blocks it is not holding, in place. This is safe only if a
 * held block is never lent, and useful only if held blocks stay packed at the bottom of the pool,
 * leaving a top to lend.
 *
 * Both fail silently: a block lent from under a sequence gets expert weights written over its KV,
 * and a pool with scattered held blocks never lends. These tests cover the mark and the ordering.
 * None needs virtual memory management; a loan maps and unmaps nothing.
 */

struct Lender {
    Pools     pools;
    KVManager kv;
    Config    cfg = small_config();

    /* The buffer derives from the chunk: its floor is one step's worth of the group, and a step
     * batches at most --max-num-batched-tokens tokens. A test wanting a bigger buffer asks for a
     * bigger chunk, the same lever the engine has. */
    int build(int64_t chunk_tokens = 2048) {
        cfg.vram_kv_mib     = 256;
        cfg.max_tok         = chunk_tokens;
        cfg.checkpoint_slots = 0;
        int s = configure_pools(pools, cfg);
        if (s < 0) return s;
        return kv.configure(hybrid_groups(/*with_linear=*/false), cfg, pools.kv());
    }
    int64_t bs() const { return kv.plan(G_FULL)->block_size; }
};

TEST(a_new_cache_lends_nothing_and_hands_out_its_whole_carve) {
    Lender e;
    CHECK_OK(e.build());
    /* The first request never waits for a recall: lending is the arbiter's decision once the
     * cache reports what it can spare, not a starting state. */
    CHECK_EQ(e.kv.loan_tokens(), 0);
    CHECK(e.kv.lendable(G_FULL));
    CHECK_EQ(e.kv.live_blocks(G_FULL), e.kv.total_blocks(G_FULL));
    CHECK_EQ(e.kv.free_blocks(G_FULL), e.kv.total_blocks(G_FULL));
}

TEST(an_idle_cache_lends_all_but_its_buffer) {
    Lender e;
    CHECK_OK(e.build());
    const int64_t all = e.kv.total_blocks(G_FULL);
    const int64_t spare = e.kv.spare_tokens();
    CHECK(spare > 0);
    CHECK_OK(e.kv.set_loan(spare));
    CHECK_EQ(e.kv.loan_tokens(), spare);

    /* It keeps the buffer; lending everything would force a recall on the first request. */
    const int64_t live = e.kv.live_blocks(G_FULL);
    CHECK(live > 0);
    CHECK(live < all);
    CHECK_EQ(all - live, spare / e.bs());
    CHECK_EQ(e.kv.free_blocks(G_FULL), live);
    CHECK_EQ(e.kv.held_blocks(G_FULL), 0);
    /* The cache's own figure shrinks by the loan; the carve (the budget) does not. */
    CHECK_EQ(e.kv.pool_bytes(G_FULL), live * e.kv.plan(G_FULL)->bytes_per_block);
    CHECK_EQ(e.kv.pool_bytes_carved(G_FULL), all * e.kv.plan(G_FULL)->bytes_per_block);

    /* The free list obeys the mark: a block above it belongs to the slab, and a restore naming
     * one is refused. */
    CHECK_EQ(e.kv.reserve_specific_block(G_FULL, (int32_t)live), RAD_E_STATE);
    CHECK_OK(e.kv.reserve_specific_block(G_FULL, (int32_t)live - 1));

    /* Returning the loan just moves the mark; every id stayed in the list. */
    CHECK_OK(e.kv.set_loan(0));
    CHECK_EQ(e.kv.live_blocks(G_FULL), all);
    CHECK_EQ(e.kv.free_blocks(G_FULL), all - 1);
}

TEST(a_block_something_holds_is_never_lent) {
    Lender e;
    CHECK_OK(e.build());
    const int64_t all = e.kv.total_blocks(G_FULL);

    /* Take every block, then release all but the highest: one held block at the top, which the
     * loan must refuse. */
    std::vector<int32_t> got;
    int32_t b = -1;
    while (e.kv.reserve_block(G_FULL, &b) == RAD_OK) got.push_back(b);
    CHECK_EQ((int64_t)got.size(), all);
    int32_t top = 0;
    for (int32_t g : got) top = std::max(top, g);
    CHECK_EQ((int64_t)top, all - 1);
    for (int32_t g : got) if (g != top) e.kv.release_block(G_FULL, g);

    /* Nothing can be lent: the pool is layer-major and the boundary is a mark, not a set, so it
     * cannot lend around the block. */
    CHECK_EQ(e.kv.spare_tokens(), 0);
    CHECK_EQ(e.kv.set_loan(e.bs()), RAD_E_STATE);
    CHECK_EQ(e.kv.live_blocks(G_FULL), all);            /* a refusal moves nothing */
    CHECK_EQ(e.kv.loan_tokens(), 0);

    /* Once released, the loan can follow. A sequence ending, an eviction or an idle-tier demotion
     * all free blocks this way. */
    e.kv.release_block(G_FULL, top);
    CHECK(e.kv.spare_tokens() > 0);
    CHECK_OK(e.kv.set_loan(e.kv.spare_tokens()));
    CHECK(e.kv.live_blocks(G_FULL) < all);
}

TEST(the_lowest_free_block_is_the_one_handed_out) {
    Lender e;
    CHECK_OK(e.build());
    int32_t a = -1, b = -1, c = -1;
    CHECK_OK(e.kv.reserve_block(G_FULL, &a));
    CHECK_OK(e.kv.reserve_block(G_FULL, &b));
    CHECK_OK(e.kv.reserve_block(G_FULL, &c));
    CHECK(a < b && b < c);

    /* Freed highest last, the worst order for a stack: a stack would hand back c, the high block
     * a loan wants, so the highest held block would never fall. */
    e.kv.release_block(G_FULL, a);
    e.kv.release_block(G_FULL, b);
    e.kv.release_block(G_FULL, c);
    int32_t again = -1;
    CHECK_OK(e.kv.reserve_block(G_FULL, &again));
    CHECK_EQ(again, a);
}

/* Avoids a livelock. A long prefill asks for its blocks in one call; a cache that reclaimed its
 * loan only on a timer would refuse it, and the scheduler waiting or preempting would never create
 * the demand that triggers the reclaim. So admission is decided against the whole carve, and the
 * recall happens on the path that needs it, through the slab owner. */
TEST(a_request_past_the_mark_takes_the_loan_back) {
    Lender e;
    CHECK_OK(e.build());
    const int64_t all = e.kv.total_blocks(G_FULL);
    const int64_t spare = e.kv.spare_tokens();
    CHECK_OK(e.kv.set_loan(spare));
    const int64_t idle = e.kv.live_blocks(G_FULL);

    int calls = 0;
    int64_t asked = -1;
    e.kv.set_reclaim([&](int64_t t) { ++calls; asked = t; return (int)RAD_OK; });

    /* Longer than the cache keeps, well within its carve. */
    const int64_t tokens = (idle + all) / 2 * e.bs();
    CHECK(e.kv.can_ever_fit(tokens));
    CHECK_OK(e.kv.add_sequence(1));
    CHECK_OK(e.kv.ensure(1, tokens));
    CHECK_EQ(calls, 1);                                 /* one recall for the whole request */
    CHECK(asked >= 0 && asked < spare);
    CHECK_EQ(e.kv.loan_tokens(), asked);
    CHECK(e.kv.live_blocks(G_FULL) > idle);
    CHECK_EQ((int64_t)e.kv.block_table(1, G_FULL).size(), (tokens + e.bs() - 1) / e.bs());
    /* The buffer came back too, so the next block does not recall again. */
    CHECK(e.kv.free_blocks(G_FULL) > 0);

    /* A request past the carve is still refused: lending decides who uses the carve, not its
     * size. */
    CHECK_OK(e.kv.add_sequence(2));
    CHECK_EQ(e.kv.ensure(2, all * e.bs()), RAD_E_FULL);
}

/* If the slabs cannot make the recall, the request is refused and nothing moves; taking the
 * blocks back regardless would hand a sequence memory an expert unit is still reading from. */
TEST(a_recall_the_slab_cannot_make_refuses_the_request) {
    Lender e;
    CHECK_OK(e.build());
    const int64_t all = e.kv.total_blocks(G_FULL);
    CHECK_OK(e.kv.set_loan(e.kv.spare_tokens()));
    const int64_t idle = e.kv.live_blocks(G_FULL);
    const int64_t loan = e.kv.loan_tokens();
    e.kv.set_reclaim([](int64_t) { return (int)RAD_E_DEVICE; });

    CHECK_OK(e.kv.add_sequence(1));
    CHECK(e.kv.ensure(1, (idle + all) / 2 * e.bs()) < 0);
    CHECK_EQ(e.kv.loan_tokens(), loan);
    CHECK_EQ(e.kv.live_blocks(G_FULL), idle);
    CHECK_EQ((int64_t)e.kv.block_table(1, G_FULL).size(), 0);
}

/* The kept buffer is one step's worth, so a chunked prefill recalls once per chunk rather than per
 * block. A bigger chunk means a bigger buffer, the operator's one lever on the derivation. */
TEST(the_derived_buffer_is_one_chunk_of_the_group) {
    Lender small, big;
    CHECK_OK(small.build(/*chunk_tokens=*/2048));
    CHECK_OK(big.build(/*chunk_tokens=*/65536));
    CHECK(big.kv.buffer_bytes() > small.kv.buffer_bytes());
    CHECK(big.kv.spare_tokens() < small.kv.spare_tokens());
}

/* A restore cannot land above the mark. Under tensor parallelism rank 0 picks the block id and
 * every rank takes the same one, so a rank with a deeper loan must refuse rather than overwrite
 * its slab's weights. */
TEST(a_named_block_above_the_mark_is_refused) {
    Lender e;
    CHECK_OK(e.build());
    CHECK_OK(e.kv.set_loan(e.kv.spare_tokens()));
    const int64_t live = e.kv.live_blocks(G_FULL);
    CHECK(live < e.kv.total_blocks(G_FULL));

    CHECK_EQ(e.kv.reserve_specific_block(G_FULL, (int32_t)live), RAD_E_STATE);
    CHECK_OK(e.kv.reserve_specific_block(G_FULL, (int32_t)live - 1));
    /* Taken from the middle: the free count reflects it even though the heap still names it. */
    CHECK_EQ(e.kv.free_blocks(G_FULL), live - 1);
    CHECK_EQ(e.kv.reserve_specific_block(G_FULL, (int32_t)live - 1), RAD_E_STATE);
}

/* A follower lends what the leader lends: block ids are shared across ranks and only rank 0 knows
 * what is held, so a follower takes the leader's mark in both directions. */
TEST(a_follower_takes_the_leaders_mark) {
    Lender lead, fol;
    CHECK_OK(lead.build());
    CHECK_OK(fol.build());
    CHECK_OK(lead.kv.set_loan(lead.kv.spare_tokens()));
    CHECK_OK(fol.kv.follow(lead.kv));
    CHECK_EQ(fol.kv.loan_tokens(), lead.kv.loan_tokens());
    CHECK_EQ(fol.kv.live_blocks(G_FULL), lead.kv.live_blocks(G_FULL));
    CHECK_OK(lead.kv.set_loan(0));
    CHECK_OK(fol.kv.follow(lead.kv));
    CHECK_EQ(fol.kv.live_blocks(G_FULL), fol.kv.total_blocks(G_FULL));
}

/* ================================================================== the snapshot store */

/* A snapshot slot is host memory the KV pool never sees: taking, writing and releasing one changes
 * nothing the pool reports, and checkpoint_ptr writes land in the store's slot. */
TEST(a_checkpoint_slot_is_host_memory_and_costs_the_pool_nothing) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    CHECK_EQ(f.kv.checkpoint_slots(), 4);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 4);
    const int64_t used = f.pools.kv().used();
    int32_t a = -1, b = -1;
    CHECK_OK(f.kv.alloc_checkpoint(&a));
    CHECK_OK(f.kv.alloc_checkpoint(&b));
    CHECK(a != b);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 2);
    CHECK_EQ(f.pools.kv().used(), used);
    char* p = (char*)f.kv.checkpoint_ptr(a);
    REQUIRE(p != nullptr);
    CHECK(p == (char*)f.ck.at(a));
    std::memset(p, 0x5a, (size_t)f.kv.checkpoint_bytes());
    CHECK_EQ(((unsigned char*)f.ck.at(a))[f.kv.checkpoint_bytes() - 1], 0x5a);
    f.kv.free_checkpoint(a);
    f.kv.free_checkpoint(a);                         /* a double free is ignored, not counted */
    f.kv.free_checkpoint(b);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 4);
}

/* Without a store no slot is handed out, rather than a pointer to unowned memory. */
TEST(without_a_store_no_checkpoint_slot_is_handed_out) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/0));
    CHECK_EQ(f.kv.checkpoint_slots(), 0);
    int32_t s = -1;
    CHECK_EQ(f.kv.alloc_checkpoint(&s), RAD_E_FULL);
    CHECK(f.kv.checkpoint_ptr(0) == nullptr);
}

/* One slot holds every rank's shard. Rank 0 hands out ids and each rank addresses its own window
 * of the slot; at a shared offset the last writer's shard would be restored on both. */
TEST(each_rank_addresses_its_own_window_of_a_shared_snapshot_slot) {
    Fixture lead, follow;
    CHECK_OK(lead.build(/*with_linear=*/true, /*ckpt_slots=*/0));
    CHECK_OK(follow.build(/*with_linear=*/true, /*ckpt_slots=*/0));
    const int64_t shard = lead.kv.checkpoint_bytes();
    HostArena store;
    CHECK_OK(store.open(2 * shard, 2 * shard * 3));
    lead.kv.set_checkpoint_store(&store, 0);
    follow.kv.set_checkpoint_store(&store, shard);
    int32_t s = -1;
    CHECK_OK(lead.kv.alloc_checkpoint(&s));
    CHECK((char*)lead.kv.checkpoint_ptr(s) == (char*)store.at(s));
    CHECK((char*)follow.kv.checkpoint_ptr(s) == (char*)store.at(s) + shard);
    CHECK_EQ(follow.kv.free_checkpoint_slots(), 2);  /* the follower sees the same free list */
    lead.kv.free_checkpoint(s);
    CHECK_EQ(store.used_slots(), 0);
}

/* ================================================================== idle session tiers
 *
 * The state machine only, with no bytes moving: tests drive IdleTiers directly and report jobs
 * back as the engine would. A tier's use-after-free hazards live in this bookkeeping, not in the
 * transfers, so none needs a GPU.
 */

static IdleTiers::Config tier_config(int64_t host_bytes, int64_t entry) {
    IdleTiers::Config c;
    c.host_cap_bytes = host_bytes;
    c.entry_bytes = entry;
    return c;
}
static BlockHash bh(uint64_t n) { return BlockHash{n, n * 0x9e3779b97f4a7c15ull + 1}; }
static const uint64_t SEC = 1000000000ull;

/* Plans every waiting copy and reports each as landed, as the engine does. */
static std::vector<TierJob> copy_now(IdleTiers& t, bool ok = true) {
    std::vector<TierJob> jobs;
    t.plan_writeback(1 << 20, &jobs);
    for (TierJob& j : jobs) t.report(j, ok);
    return jobs;
}

/* Gives up an entry's VRAM as the cache does, if nothing else is reading it. */
static bool drop_now(IdleTiers& t, const BlockHash& k) {
    if (!t.droppable(k)) return false;
    t.drop_commit(k);
    return true;
}

TEST(a_published_entry_is_copied_at_once_and_keeps_its_vram) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    CHECK(t.host_on());

    t.note_used(bh(1), {10}, 1 * SEC);
    CHECK(t.writeback_waiting());

    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK(jobs[0].kind == TierJob::Kind::WriteBack);
    CHECK(jobs[0].key == bh(1));
    REQUIRE_EQ(jobs[0].blocks.size(), 1u);
    CHECK_EQ(jobs[0].blocks[0], 10);
    CHECK(jobs[0].host_slot >= 0);

    /* Pending excludes it from a second plan and from a drop: until the copy lands, the VRAM copy
     * is the only complete one. */
    std::vector<TierJob> again;
    t.plan_writeback(16, &again);
    CHECK_EQ(again.size(), 0u);
    CHECK(!t.droppable(bh(1)));

    t.report(jobs[0], true);
    /* A copy is not a move: the entry stays in VRAM, where a return finds it for free, but giving
     * the VRAM up now costs nothing. */
    CHECK(t.tier_of(bh(1)) == KVTier::Device);
    CHECK(t.droppable(bh(1)));
    CHECK_EQ(t.stats().to_host, 1);
    CHECK_EQ(t.stats().dropped, 0);
    CHECK_EQ(t.arena().used_slots(), 1);
    CHECK(!t.writeback_waiting());
    std::vector<BlockHash> clean;
    t.take_clean(&clean);
    REQUIRE_EQ(clean.size(), 1u);
    CHECK(clean[0] == bh(1));
}

TEST(giving_up_a_clean_entry_s_vram_moves_no_bytes) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    t.note_used(bh(1), {10}, 1 * SEC);
    copy_now(t);

    CHECK(drop_now(t, bh(1)));
    CHECK(t.tier_of(bh(1)) == KVTier::Host);
    CHECK(t.resident_off_device(bh(1)));
    CHECK_EQ(t.stats().dropped, 1);
    CHECK_EQ(t.stats().to_host, 1);               /* the one earlier copy */
    CHECK_EQ(t.arena().used_slots(), 1);          /* the copy is now the entry */
    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    CHECK_EQ(jobs.size(), 0u);                    /* nothing left to copy */
}

TEST(a_failed_copy_leaves_the_entry_in_vram_and_tries_again) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(1024, 1024)));   /* exactly one slot */
    t.note_used(bh(1), {10}, 0);

    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK(t.arena().full());

    t.report(jobs[0], false);
    CHECK(t.tier_of(bh(1)) == KVTier::Device);    /* the pool still holds the blocks */
    CHECK(!t.droppable(bh(1)));            /* and may not release them */
    CHECK(!t.arena().full());                     /* the slot went back */
    CHECK_EQ(t.stats().failures, 1);
    CHECK_EQ(t.stats().to_host, 0);

    /* A failed copy is retried. */
    std::vector<TierJob> retry;
    t.plan_writeback(16, &retry);
    CHECK_EQ(retry.size(), 1u);
}

/* One copy per run keeps long conversations affordable at a 4-token block, so entries planned
 * together take consecutive slots. */
TEST(a_batch_of_copies_takes_consecutive_arena_slots) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(64 * 1024, 1024)));
    for (uint64_t i = 1; i <= 8; ++i) t.note_used(bh(i), {(int32_t)(10 + i)}, i * SEC);
    std::vector<TierJob> jobs;
    t.plan_writeback(64, &jobs);
    REQUIRE_EQ(jobs.size(), 8u);
    for (size_t i = 0; i < jobs.size(); ++i)
        CHECK_EQ(jobs[i].host_slot, jobs[0].host_slot + (int32_t)i);
    /* Oldest first. */
    CHECK(jobs[0].key == bh(1));
    CHECK(jobs[7].key == bh(8));
}

TEST(host_arena_runs_stop_at_a_used_slot_and_wrap) {
    HostArena a;
    CHECK_OK(a.open(1024, 8 * 1024));
    REQUIRE_EQ(a.capacity(), 8);
    int32_t got = 0;
    CHECK_EQ(a.alloc_run(3, &got), 0);
    CHECK_EQ(got, 3);
    CHECK_EQ(a.alloc_run(3, &got), 3);
    CHECK_EQ(got, 3);
    a.free(1);
    /* The run ends at the arena's end; the caller asks again for the rest. */
    CHECK_EQ(a.alloc_run(4, &got), 6);
    CHECK_EQ(got, 2);
    /* The cursor wraps to the hole behind it. */
    CHECK_EQ(a.alloc_run(4, &got), 1);
    CHECK_EQ(got, 1);
    CHECK(a.full());
    CHECK_EQ(a.alloc_run(1, &got), -1);
    CHECK_EQ(got, 0);
    /* A double free is ignored, keeping the used count (which the view reports) correct. */
    a.free(1);
    a.free(1);
    CHECK_EQ(a.used_slots(), 7);
}

TEST(a_full_arena_holds_the_copy_back_and_makes_room_from_the_oldest_host_entry) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 1024)));   /* two slots */
    std::vector<BlockHash> lost;
    t.set_drop_sink([&lost](const BlockHash& h) { lost.push_back(h); });

    t.note_used(bh(1), {11}, 1 * SEC);
    t.note_used(bh(2), {12}, 2 * SEC);
    CHECK_EQ(copy_now(t).size(), 2u);
    drop_now(t, bh(1));
    drop_now(t, bh(2));

    t.note_used(bh(3), {13}, 3 * SEC);
    CHECK_EQ(copy_now(t).size(), 0u);             /* nowhere to put it */
    CHECK(t.starved());
    /* Still queued, but not planned again until a slot is freed. */
    CHECK(!t.writeback_waiting());
    CHECK(t.tier_of(bh(3)) == KVTier::Device);

    /* The maintenance plan frees exactly what was refused, LRU first, and tells the cache so it
     * stops offering the entry. */
    std::vector<TierJob> room;
    t.plan(16, &room);
    REQUIRE_EQ(room.size(), 1u);
    CHECK(room[0].kind == TierJob::Kind::DropHost);
    CHECK(room[0].key == bh(1));
    t.report(room[0], true);
    CHECK_EQ(t.stats().host_drops, 1);
    REQUIRE_EQ(lost.size(), 1u);
    CHECK(lost[0] == bh(1));
    CHECK(!t.starved());
    CHECK(t.writeback_waiting());

    CHECK_EQ(copy_now(t).size(), 1u);
    CHECK(t.droppable(bh(3)));
}

TEST(forgetting_an_entry_mid_copy_does_not_erase_it_under_the_engine) {
    /* An entry evicted from the prefix cache mid-copy must keep its record: the engine's job names
     * it and report() dereferences it afterwards. */
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    t.note_used(bh(7), {70}, 0);

    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    CHECK(!t.doomed(bh(7)));

    t.forget(bh(7));                              /* evicted while pending */
    CHECK(t.doomed(bh(7)));

    t.report(jobs[0], true);                      /* the engine finishes anyway */
    CHECK(t.doomed(bh(7)));                       /* gone: unknown reads as doomed */
    CHECK(t.tier_of(bh(7)) == KVTier::Device);    /* and as Device, meaning "not ours" */
    CHECK_EQ(t.arena().used_slots(), 0);          /* its slot came back */
    CHECK(!t.droppable(bh(7)));
}

TEST(a_restore_keeps_the_host_copy_so_the_entry_lands_clean) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    t.note_used(bh(5), {50}, 0);
    copy_now(t);
    drop_now(t, bh(5));

    TierJob up;
    CHECK(t.plan_promote(bh(5), &up));
    CHECK(up.kind == TierJob::Kind::ToDevice);
    CHECK(up.host_slot >= 0);
    /* An outstanding promotion is not handed out twice. */
    TierJob dup;
    CHECK(!t.plan_promote(bh(5), &dup));

    up.blocks = {99};                             /* the engine allocated a different block */
    up.started = true;
    t.report(up, true, 100 * SEC);
    CHECK(t.tier_of(bh(5)) == KVTier::Device);
    CHECK_EQ(t.stats().promoted, 1);
    CHECK_EQ(t.stats().host_hits, 1);
    /* The host copy stays, so the VRAM can be dropped again for free with nothing to copy. */
    CHECK_EQ(t.arena().used_slots(), 1);
    CHECK(t.droppable(bh(5)));
    CHECK(!t.writeback_waiting());
    std::vector<BlockHash> clean;
    t.take_clean(&clean);
    CHECK(std::find(clean.begin(), clean.end(), bh(5)) != clean.end());
}

TEST(only_what_a_turn_added_is_copied_after_a_restore) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(64 * 1024, 1024)));
    for (uint64_t i = 1; i <= 3; ++i) t.note_used(bh(i), {(int32_t)i}, 1 * SEC);
    copy_now(t);
    for (uint64_t i = 1; i <= 3; ++i) {
        drop_now(t, bh(i));
        TierJob up;
        REQUIRE(t.plan_promote(bh(i), &up));
        up.blocks = {(int32_t)(20 + i)};
        up.started = true;
        t.report(up, true, 10 * SEC);
    }
    /* The next turn publishes the whole chain again, two blocks longer. */
    for (uint64_t i = 1; i <= 5; ++i) t.note_used(bh(i), {(int32_t)(20 + i)}, 20 * SEC);
    std::vector<TierJob> jobs;
    t.plan_writeback(64, &jobs);
    REQUIRE_EQ(jobs.size(), 2u);
    CHECK(jobs[0].key == bh(4));
    CHECK(jobs[1].key == bh(5));
}

TEST(a_failed_restore_that_started_drops_the_entry) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    std::vector<BlockHash> lost;
    t.set_drop_sink([&lost](const BlockHash& h) { lost.push_back(h); });
    t.note_used(bh(3), {30}, 0);
    copy_now(t);
    drop_now(t, bh(3));

    TierJob up;
    CHECK(t.plan_promote(bh(3), &up));
    up.blocks = {31};
    up.started = true;
    t.report(up, false);
    /* Dropped, not kept on the host tier: a partial scatter leaves blocks that no longer match the
     * tokens, so only a miss is safe, and the cache is told so it stops serving the old ids. */
    CHECK(t.tier_of(bh(3)) == KVTier::Device);
    CHECK(!t.resident_off_device(bh(3)));
    CHECK_EQ(t.arena().used_slots(), 0);
    REQUIRE_EQ(lost.size(), 1u);
    CHECK(lost[0] == bh(3));
}

TEST(a_restore_that_never_started_leaves_the_entry_where_it_was) {
    /* The pool ran out before this entry, so nothing was touched and its host copy stays valid. */
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    t.note_used(bh(3), {30}, 0);
    copy_now(t);
    drop_now(t, bh(3));

    TierJob up;
    CHECK(t.plan_promote(bh(3), &up));
    t.report(up, false);
    CHECK(t.tier_of(bh(3)) == KVTier::Host);
    CHECK(t.resident_off_device(bh(3)));
    CHECK_EQ(t.arena().used_slots(), 1);
    TierJob again;
    CHECK(t.plan_promote(bh(3), &again));
}

TEST(a_recomputed_entry_gives_up_its_tier_copy_and_is_copied_afresh) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    t.note_used(bh(2), {20}, 1 * SEC);
    copy_now(t);
    drop_now(t, bh(2));
    CHECK_EQ(t.arena().used_slots(), 1);

    t.rebound(bh(2), {77}, 10 * SEC);
    CHECK(t.tier_of(bh(2)) == KVTier::Device);
    CHECK_EQ(t.arena().used_slots(), 0);
    CHECK(!t.droppable(bh(2)));            /* the new blocks have no copy yet */
    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    REQUIRE_EQ(jobs.size(), 1u);
    REQUIRE_EQ(jobs[0].blocks.size(), 1u);
    CHECK_EQ(jobs[0].blocks[0], 77);
}

TEST(the_session_view_reports_where_a_conversation_actually_is) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 1024)));
    t.set_model("q38-flashnext");

    std::vector<BlockHash> chain = {bh(1), bh(2), bh(3)};
    for (uint64_t i = 1; i <= 3; ++i) t.note_used(bh(i), {(int32_t)i}, 0);
    t.note_session(42, chain, 48, 0);

    std::vector<IdleTiers::SessionInfo> s = t.sessions();
    REQUIRE_EQ(s.size(), 1u);
    CHECK_EQ(s[0].id, 42u);
    CHECK(s[0].model == "q38-flashnext");
    CHECK_EQ(s[0].n_tokens, 48);
    CHECK_EQ(s[0].bytes, 3 * 1024);
    CHECK_EQ(s[0].on_device, 3);

    /* Two of three fit in the arena, so the session straddles two tiers and the view reports
     * both. */
    copy_now(t);
    drop_now(t, bh(1));
    drop_now(t, bh(2));
    s = t.sessions();
    REQUIRE_EQ(s.size(), 1u);
    CHECK_EQ(s[0].on_host, 2);
    CHECK_EQ(s[0].on_device, 1);
}

/* A return reuses up to the deepest snapshot, not all blocks: on a hybrid model the scheduler
 * clamps a hit to the last reachable linear checkpoint. Off-card bytes are counted once, however
 * many conversations share them. */
TEST(the_session_view_says_what_a_return_reuses_and_what_is_off_the_card) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 1024)));

    for (uint64_t i = 1; i <= 3; ++i) t.note_used(bh(i), {(int32_t)i}, 0);
    t.note_session(1, {bh(1), bh(2), bh(3)}, 48, 0);
    /* A second conversation past the same first two blocks. */
    t.note_used(bh(9), {9}, 100 * SEC);
    t.note_session(2, {bh(1), bh(2), bh(9)}, 48, 100 * SEC);

    IdleTiers::Occupancy occ;
    std::vector<IdleTiers::SessionInfo> s = t.sessions(&occ);
    CHECK_EQ(s.size(), 2u);
    CHECK_EQ(s[0].resume_tokens, 0);             /* no snapshot: nothing reused */
    CHECK_EQ(occ.host_bytes, 0);
    CHECK_EQ(occ.host_cap_bytes, 2048);

    /* The shared blocks were published first, so they are copied first and fill the arena's two
     * slots; each is one slot however many rows name it. */
    copy_now(t);
    drop_now(t, bh(1));
    drop_now(t, bh(2));
    s = t.sessions(&occ);
    CHECK_EQ(occ.host_bytes, t.arena().used_slots() * 1024);
    CHECK_EQ(t.arena().used_slots(), 2);
    int64_t named = 0;
    for (const auto& r : s) named += r.on_host;
    CHECK(named > t.arena().used_slots());

    /* A return resumes at the deepest snapshot its chain reaches; one on the shared prefix is
     * reachable from both. */
    IdleTiers u;
    CHECK_OK(u.configure(tier_config(0, 1024)));
    for (uint64_t i : {1, 2, 3, 9}) u.note_used(bh(i), {(int32_t)i}, 0);
    u.note_snapshot(bh(1), 0, 16);
    u.note_snapshot(bh(9), 1, 48);
    u.note_session(1, {bh(1), bh(2), bh(3)}, 48, 0);
    u.note_session(2, {bh(1), bh(2), bh(9)}, 48, 1 * SEC);
    s = u.sessions();
    CHECK_EQ(s.size(), 2u);
    for (const auto& r : s) {
        if (r.id == 2) CHECK_EQ(r.resume_tokens, 48);
        if (r.id == 1) CHECK_EQ(r.resume_tokens, 16);
    }
}

TEST(tiers_stay_off_until_a_size_is_given) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(0, 1024)));
    CHECK(!t.host_on());
    CHECK(!t.host_on());
    t.note_used(bh(1), {10}, 0);
    CHECK(!t.writeback_waiting());
    std::vector<TierJob> jobs;
    t.plan_writeback(16, &jobs);
    t.plan(16, &jobs);
    CHECK_EQ(jobs.size(), 0u);
    /* No per-block state either, which would cost every operator a map proportional to the
     * cache. Session records (one per conversation) are kept either way; see below. */
    CHECK_EQ(t.sessions().size(), 0u);
}

/* ================================================================== the linear half
 *
 * On a hybrid model a session also has a recurrent-state snapshot in a slot of the engine's
 * checkpoint store (pinned host memory, --checkpoint-slots for the whole server). Snapshots do not
 * move through the tiers: they are restored from their slot, and the tiers only record them for
 * the session view.
 */

TEST(a_record_that_only_ever_held_a_snapshot_is_forgotten_with_it) {
    /* note_snapshot runs at commit, mid-request; note_used runs when the chain is published at
     * request close. A request that never publishes (a rebased block table, a cancel before the
     * first whole block) would otherwise leave an unreachable record per checkpoint. */
    IdleTiers t;
    IdleTiers::Config c = tier_config(1 << 20, 1024);
    c.ckpt_bytes = 2048;
    c.ckpt_slots = 8;
    CHECK_OK(t.configure(c));
    t.note_snapshot(bh(1), 7, 2048);
    std::vector<TierJob> jobs;
    t.plan(16, &jobs);
    CHECK_EQ(jobs.size(), 0u);                         /* an unpublished request costs nothing */
    t.release_snapshot(bh(1));
    CHECK(t.tier_of(bh(1)) == KVTier::Device);         /* "no record" answers Device */
    CHECK(!t.writeback_waiting());
    CHECK_EQ(t.sessions().size(), 0u);
}

/* ------------------------------------------------------------------ the cache's half of it */

TEST(the_tiers_are_told_the_cache_s_block_ids_and_not_the_requests) {
    /* Two requests compute the same prefix; only one set of blocks is referenced. The second
     * missed (its lookup stopped early, as happens while the tiers hold the entry) and filled its
     * own blocks; insert() keeps the existing entry and lets the second copy go.
     *
     * Given the second request's ids, the tiers would later gather from blocks the pool had
     * reassigned, storing another sequence's KV under this prefix, and release a reference nobody
     * took. */
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false, /*ckpt_slots=*/0));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(1 << 20, 1024)));
    f.pc.set_tiers(&t);

    auto prompt = tok_run(7400, 336);
    publish(f, 1, prompt, 320);
    std::vector<int32_t> first = f.kv.block_table(1, f.pc.cached_groups()[0]);
    CHECK(first.size() >= 20u);

    /* The second sequence has its own blocks for the same tokens; the cache keeps the first's. */
    publish(f, 2, prompt, 320);
    std::vector<int32_t> second = f.kv.block_table(2, f.pc.cached_groups()[0]);
    CHECK(second[0] != first[0]);

    std::vector<TierJob> jobs;
    t.plan_writeback(512, &jobs);
    CHECK(jobs.size() >= 20u);
    /* Every job names a block the cache holds. */
    for (const TierJob& j : jobs) {
        bool from_cache = false, from_second = false;
        for (size_t i = 0; i < first.size(); ++i)  if (j.blocks[0] == first[i])  from_cache = true;
        for (size_t i = 0; i < second.size(); ++i) if (j.blocks[0] == second[i]) from_second = true;
        CHECK(from_cache);
        CHECK(!from_second);
    }
}

/* ================================================================== release on request
 *
 * A client that deleted a conversation, or kept a prefix of it and replaced the rest, forgets it
 * through the session table: by label, from the tip inwards, host copy and snapshots with it.
 * What another conversation also walks stays, as does what a live sequence holds.
 */

/* Publishes `tokens` under `id` with a session label, as the scheduler does at a turn's close,
 * and frees the sequence, as it does right after. */
static void publish_turn(Fixture& f, uint64_t id, const std::vector<int32_t>& tokens,
                         const std::string& label) {
    CHECK_OK(f.kv.add_sequence(id));
    CHECK_OK(f.kv.ensure(id, (int64_t)tokens.size()));
    std::vector<std::vector<int32_t>> bt;
    for (int32_t g : f.pc.cached_groups()) bt.push_back(f.kv.block_table(id, g));
    CHECK_OK(f.pc.insert(id, tokens, {}, bt, (int64_t)tokens.size(), label));
    f.kv.free_sequence(id);
}

/* The label's rows in the session table. */
static std::vector<IdleTiers::SessionInfo> rows_of(IdleTiers& t, const std::string& label) {
    std::vector<IdleTiers::SessionInfo> out;
    for (const IdleTiers::SessionInfo& s : t.sessions())
        if (s.label == label) out.push_back(s);
    return out;
}

TEST(a_released_conversation_goes_from_the_tip_down_to_what_another_one_shares) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false, /*ckpt_slots=*/0));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(1 << 20, 1024)));
    f.pc.set_tiers(&t);

    /* Conversation A over three turns under three request ids: each turn extends the last. The
     * first two blocks are a system prompt conversation B opens with too. */
    const auto a1 = tok_run(100, 64);                      /* 4 blocks */
    auto a2 = a1; for (int i = 0; i < 32; ++i) a2.push_back(700 + i);    /* 6 */
    auto a3 = a2; for (int i = 0; i < 32; ++i) a3.push_back(800 + i);    /* 8 */
    publish_turn(f, 1, a1, "chat-a");
    publish_turn(f, 2, a2, "chat-a");
    publish_turn(f, 3, a3, "chat-a");
    auto b = tok_run(100, 32); for (int i = 0; i < 64; ++i) b.push_back(900 + i);   /* 6 */
    publish_turn(f, 4, b, "chat-b");
    CHECK_EQ(f.pc.size(), 12ll);                            /* 8 + 4 own blocks of B */
    REQUIRE_EQ(rows_of(t, "chat-a").size(), 1u);            /* one row per conversation */
    CHECK_EQ(rows_of(t, "chat-a")[0].n_tokens, 128ll);
    /* Every entry has its host copy. */
    copy_now(t);
    CHECK_EQ(t.arena().used_slots(), 12);
    const int64_t pool_before = f.kv.free_blocks(G_FULL);

    SessionRelease r;
    CHECK_EQ(f.pc.release_session("chat-c", 0, 0, &r), RAD_E_NOTFOUND);
    CHECK_EQ(f.pc.release_session("", 0, 0, &r), RAD_E_INVAL);

    CHECK_OK(f.pc.release_session("chat-a", 0, 0, &r));
    CHECK_EQ(r.sessions, 1ll);
    CHECK_EQ(r.entries, 6ll);                               /* 8 less the 2 B walks too */
    CHECK_EQ(r.tokens, 96ll);
    CHECK_EQ(r.kept_shared, 2ll);
    CHECK_EQ(r.kept_live, 0ll);
    CHECK_EQ(r.kept_tokens, 0ll);
    CHECK(r.device_bytes > 0);
    CHECK_EQ(r.host_bytes, 6 * 1024ll);
    /* Pool and arena got the six back; the index holds B's chain and the shared start. */
    CHECK_EQ(f.kv.free_blocks(G_FULL), pool_before + 6);
    CHECK_EQ(t.arena().used_slots(), 6);
    CHECK_EQ(f.pc.size(), 6ll);
    CHECK_EQ(f.pc.stats().released, 6ll);
    /* A's next turn hits the shared system prompt and nothing more; B is whole. */
    CHECK_EQ(f.pc.lookup(a3, {}).n_attn_tokens, 32);
    CHECK_EQ(f.pc.lookup(b, {}).n_attn_tokens, 80);
    CHECK_EQ(rows_of(t, "chat-a").size(), 0u);
    REQUIRE_EQ(rows_of(t, "chat-b").size(), 1u);
    CHECK_EQ(rows_of(t, "chat-b")[0].n_tokens, 96ll);

    /* Released twice is not found the second time. */
    CHECK_EQ(f.pc.release_session("chat-a", 0, 0, &r), RAD_E_NOTFOUND);
}

TEST(keep_tokens_leaves_the_chains_start_standing_in_whole_blocks) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false, /*ckpt_slots=*/0));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(1 << 20, 1024)));
    f.pc.set_tiers(&t);

    auto b = tok_run(100, 32); for (int i = 0; i < 64; ++i) b.push_back(900 + i);   /* 6 blocks */
    publish_turn(f, 4, b, "chat-b");

    /* 40 tokens keep three blocks: the one holding token 40 goes with them. */
    SessionRelease r;
    CHECK_OK(f.pc.release_session("chat-b", 0, 40, &r));
    CHECK_EQ(r.entries, 3ll);
    CHECK_EQ(r.kept_tokens, 48ll);
    CHECK_EQ(r.kept_shared, 0ll);
    CHECK_EQ(f.pc.lookup(b, {}).n_attn_tokens, 48);
    /* The row stays, cut to the kept prefix: the client extends it again. */
    REQUIRE_EQ(rows_of(t, "chat-b").size(), 1u);
    CHECK_EQ(rows_of(t, "chat-b")[0].n_tokens, 48ll);
    const uint64_t id = rows_of(t, "chat-b")[0].id;

    /* The same by engine id, keeping more than is left: nothing goes, nothing breaks. */
    CHECK_OK(f.pc.release_session("", id, 200, &r));
    CHECK_EQ(r.entries, 0ll);
    CHECK_EQ(r.kept_tokens, 48ll);
    CHECK_EQ(f.pc.lookup(b, {}).n_attn_tokens, 48);

    /* A later turn extends the stub into a full row again. */
    publish_turn(f, 5, b, "chat-b");
    REQUIRE_EQ(rows_of(t, "chat-b").size(), 1u);
    CHECK_EQ(rows_of(t, "chat-b")[0].n_tokens, 96ll);
    CHECK_EQ(f.pc.lookup(b, {}).n_attn_tokens, 80);
}

TEST(a_release_takes_the_sessions_snapshots_past_the_kept_prefix_and_leaves_live_blocks) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    IdleTiers t;
    IdleTiers::Config tc = tier_config(1 << 20, 1024);
    tc.ckpt_bytes = f.kv.checkpoint_bytes();
    tc.ckpt_slots = 4;
    CHECK_OK(t.configure(tc));
    f.pc.set_tiers(&t);

    const auto a = tok_run(100, 128);                      /* 8 blocks, checkpoints at 64, 128 */
    const std::vector<BlockHash> hs = chain_of(a);
    CHECK_OK(f.kv.add_sequence(3));
    CHECK_OK(f.kv.ensure(3, 128));
    ckpt_at(f, 3, hs, 64);
    ckpt_at(f, 3, hs, 128);
    f.pc.begin_step();
    {
        std::vector<std::vector<int32_t>> bt;
        for (int32_t g : f.pc.cached_groups()) bt.push_back(f.kv.block_table(3, g));
        CHECK_OK(f.pc.insert(3, a, {}, bt, 128, "chat-a"));
    }
    f.kv.free_sequence(3);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 2ll);
    CHECK_EQ(rows_of(t, "chat-a")[0].snapshots, 2ll);

    /* A waiting request restored the chain and holds it; a running one adopted the first
     * five blocks. Both keep their blocks through a release. */
    std::vector<std::vector<int32_t>> found;
    const CacheHit h = f.pc.lookup(a, {}, &found);
    CHECK_EQ(h.n_attn_tokens, 112);
    CHECK_OK(f.kv.add_sequence(9));
    std::vector<int32_t> five(found[0].begin(), found[0].begin() + 5);
    CHECK_OK(f.kv.adopt(9, f.pc.cached_groups()[0], five, 80));
    f.pc.begin_step();

    SessionRelease r;
    CHECK_OK(f.pc.release_session("chat-a", 0, 64, &r));
    /* Blocks 4..7 are past the kept 64 tokens; block 4 is live, 5..7 go. */
    CHECK_EQ(r.entries, 3ll);
    CHECK_EQ(r.kept_live, 1ll);
    CHECK_EQ(r.kept_tokens, 64ll);
    /* The snapshot at 128 sat on block 7 and went with it; the one at 64 is on the kept prefix. */
    CHECK_EQ(r.snapshots, 1ll);
    CHECK_EQ(f.kv.free_checkpoint_slots(), 3ll);
    const std::vector<int64_t> left = f.pc.checkpoints(3);
    REQUIRE_EQ(left.size(), 1u);
    CHECK_EQ(left[0], 64ll);
    CHECK_EQ(f.pc.stats().ckpt_released, 1ll);
    /* The live block is still indexed and the sequence still reads it; the row reaches it. */
    CHECK_EQ(f.kv.block_refcount(G_FULL, five[4]), 2);
    REQUIRE_EQ(rows_of(t, "chat-a").size(), 1u);
    CHECK_EQ(rows_of(t, "chat-a")[0].n_tokens, 80ll);
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 80);
    CHECK_EQ(f.pc.lookup(a, {}).n_linear_tokens, 64);

    /* The sequence is gone: the block it held goes on the next release. */
    f.kv.free_sequence(9);
    CHECK_OK(f.pc.release_session("chat-a", 0, 64, &r));
    CHECK_EQ(r.entries, 1ll);
    CHECK_EQ(r.kept_live, 0ll);
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 64);
    CHECK_EQ(f.pc.lookup(a, {}).n_linear_tokens, 64);
    CHECK_EQ(rows_of(t, "chat-a")[0].n_tokens, 64ll);
}

/* ================================================================== the pool in bytes
 *
 * Blocks and tokens cannot be summed across KV groups: a recurrent group has no token axis and
 * hands out states, and paged groups share one token axis, so adding their capacities overstates
 * the pool several times. Bytes are disjoint VRAM extents and sum correctly, so they answer how
 * much of the card the KV cache holds.
 */

TEST(kv_pool_bytes_are_the_plan_and_used_is_what_is_held) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true));

    const KVGroupPlan* pg = f.kv.plan(G_FULL);
    const KVGroupPlan* ps = f.kv.plan(G_GDN);
    CHECK(pg && ps);

    /* Capacity is the plan's own arithmetic, not a second derivation. */
    CHECK_EQ(f.kv.pool_bytes(G_FULL), pg->n_blocks * pg->bytes_per_block);
    CHECK_EQ(f.kv.pool_bytes(G_GDN),  ps->n_states * ps->bytes_per_state);
    /* The sentinel at handle 0 is summed with every real group, so it must contribute nothing. */
    CHECK_EQ(f.kv.pool_bytes(G_NULL), 0ll);
    CHECK_EQ(f.kv.pool_bytes_used(G_NULL), 0ll);
    /* Nor does a handle with no carved group. */
    CHECK_EQ(f.kv.pool_bytes(99), 0ll);
    CHECK_EQ(f.kv.pool_bytes_used(99), 0ll);

    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 0ll);

    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 32));                       /* two 16-token blocks */
    CHECK_EQ(f.kv.block_table(1, G_FULL).size(), 2u);
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 2 * pg->bytes_per_block);
    /* The recurrent state is charged as soon as the sequence exists, when its slot becomes
     * unavailable. */
    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), ps->bytes_per_state);

    f.kv.free_sequence(1);
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 0ll);
    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), 0ll);
}

/* Used bytes track the free list, not the sequence table, so they mean "VRAM the next request
 * cannot get": a finished turn's blocks held by the prefix cache still occupy the card. */
TEST(kv_bytes_in_use_counts_a_block_only_the_cache_still_holds) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    const int64_t per = f.kv.plan(G_FULL)->bytes_per_block;

    publish(f, 1, tok_run(100, 32), 32);
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 2 * per);

    f.kv.free_sequence(1);                    /* the turn ends; the cache keeps its reference */
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 2 * per);

    f.pc.clear();                             /* now nothing holds them */
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 0ll);
}

/* ================================================================== a dropped copy gives VRAM back
 *
 * IdleTiers decides an entry's VRAM may go and KVManager owns the free list; the release that
 * connects them is PrefixCache::drop_clean, which also holds back blocks a running sequence still
 * reads. A drop that moved the record without releasing blocks would pass the tier and KV tests
 * alone, so these drive the whole path and check the pool, not the policy.
 */
struct TierFixture {
    Fixture f;
    IdleTiers t;
    int64_t per = 0;
    int build(bool with_linear = false, int64_t ckpt_slots = 0) {
        int s = f.build(with_linear, ckpt_slots);
        if (s < 0) return s;
        s = t.configure(tier_config(1 << 20, 1024));
        if (s < 0) return s;
        f.pc.set_tiers(&t);
        per = f.kv.plan(G_FULL)->bytes_per_block;
        return RAD_OK;
    }
};

TEST(dropping_a_clean_entry_gives_its_vram_back_and_keeps_it_restorable) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7600, 336);
    publish(x.f, 1, prompt, 320);                   /* twenty 16-token blocks */
    x.f.kv.free_sequence(1);                        /* finished: only the cache holds them */
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 20 * x.per);

    CHECK_EQ(copy_now(x.t).size(), 20u);
    CHECK_EQ(x.f.pc.drop_clean(), 20);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 0ll);
    CHECK_EQ(x.f.pc.resident(), 0ll);
    CHECK_EQ(x.f.pc.size(), 20ll);                  /* every entry still indexed */
    CHECK_EQ(x.f.pc.stats().drops, 20ll);
    CHECK_EQ(x.f.pc.stats().evictions, 0ll);        /* nothing lost */
    /* Lookup stops where VRAM stops; the restore path sees the whole chain. */
    CHECK_EQ(x.f.pc.lookup(prompt, {}).n_attn_tokens, 0);
    CHECK_EQ(x.f.pc.restorable_from(prompt, {}).size(), 20u);
    CHECK_EQ(x.f.pc.drop_clean(), 0);               /* nothing left to drop */
}

TEST(a_recomputed_shared_prefix_keeps_its_original_host_copy_and_snapshot) {
    /* A short request shares less than one checkpoint with an off-device conversation, so it
     * recomputes instead of adopting that prefix. Its different prefill shape can round the KV
     * differently. Publishing it must not replace the original bytes underneath the longer
     * conversation's still-valid recurrent snapshot. Device-resident entries already keep the
     * first copy; host-resident entries must obey the same rule. */
    TierFixture x;
    CHECK_OK(x.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    const auto prompt = tok_run(7600, 336);
    const auto keys = chain_of(prompt);
    publish(x.f, 1, prompt, 320);
    const int32_t snapshot = ckpt_at(x.f, 1, keys, 320);
    x.f.kv.free_sequence(1);

    const auto copied = copy_now(x.t);
    REQUIRE_EQ(copied.size(), 20u);
    const int32_t host_slot = copied[0].host_slot;
    REQUIRE(host_slot >= 0);
    std::memset(x.t.arena().at(host_slot), 0x5a, 1024);
    CHECK_EQ(x.f.pc.drop_clean(), 20);
    CHECK_EQ(x.t.arena().used_slots(), 20ll);

    std::vector<int32_t> short_prompt(prompt.begin(), prompt.begin() + 32);
    const auto tail = tok_run(9600, 32);
    short_prompt.insert(short_prompt.end(), tail.begin(), tail.end());
    CHECK_EQ(x.f.pc.lookup(short_prompt, {}).n_linear_tokens, 0);
    publish(x.f, 2, short_prompt, 64);
    x.f.kv.free_sequence(2);

    CHECK(x.t.resident_off_device(keys[0]));
    CHECK(x.t.resident_off_device(keys[1]));
    CHECK_EQ(x.t.arena().used_slots(), 20ll);
    CHECK_EQ(x.f.pc.restorable_from(prompt, {}).size(), 20u);
    CHECK(x.f.kv.checkpoint_held(snapshot));
    const auto* bytes = static_cast<const unsigned char*>(x.t.arena().at(host_slot));
    for (int i = 0; i < 1024; ++i) CHECK_EQ(bytes[i], 0x5a);
    TierJob restore;
    CHECK(x.t.plan_promote(keys[0], &restore));
    CHECK_EQ(restore.host_slot, host_slot);
    x.t.report(restore, false);                  /* leave the unstarted restore in host memory */
}

TEST(an_entry_a_running_sequence_reads_is_dropped_once_the_sequence_lets_go) {
    /* Dropping a block a sequence also holds would free nothing while the entry read off-device,
     * so the next lookup would restore a second copy beside it. */
    TierFixture x;
    CHECK_OK(x.build());
    publish(x.f, 1, tok_run(7600, 336), 320);       /* sequence 1 is still running */
    copy_now(x.t);
    CHECK_EQ(x.f.pc.drop_clean(), 0);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 21 * x.per);

    x.f.kv.free_sequence(1);
    CHECK_EQ(x.f.pc.drop_clean(), 20);              /* after the sequence lets go */
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 0ll);
}

TEST(a_failed_copy_leaves_the_vram_exactly_where_it_was) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7600, 336);
    publish(x.f, 1, prompt, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t, /*ok=*/false);
    CHECK_EQ(x.f.pc.drop_clean(), 0);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 20 * x.per);
    CHECK_EQ(x.f.pc.lookup(prompt, {}).n_attn_tokens, 320);
}

TEST(pressure_on_the_pool_moves_a_clean_entry_and_drops_one_with_no_copy) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto a = tok_run(7600, 336);
    const auto b = tok_run(9600, 336);
    publish(x.f, 1, a, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t);                                  /* a is clean */
    publish(x.f, 2, b, 320);
    x.f.kv.free_sequence(2);                        /* b has no copy yet */

    CHECK_EQ(x.f.pc.evict_lru(1000), 40);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 0ll);
    CHECK_EQ(x.f.pc.restorable_from(a, {}).size(), 20u);   /* kept, off the card */
    CHECK_EQ(x.f.pc.restorable_from(b, {}).size(), 0u);    /* gone */
    CHECK_EQ(x.f.pc.size(), 20ll);
    CHECK_EQ(x.f.pc.stats().drops, 20ll);
    CHECK_EQ(x.f.pc.stats().evictions, 20ll);
    /* Kept entries are off the list pressure walks: evicting again frees and loses nothing. */
    CHECK_EQ(x.f.pc.evict_lru(1000), 0);
    CHECK_EQ(x.f.pc.size(), 20ll);
}

/* The reserve's trim loses no conversation. It runs right after a publish, while the publishing
 * sequence holds its blocks and before host copies land, so it must skip entries a sequence holds
 * and entries with no copy yet. */
TEST(the_trim_keeps_what_a_sequence_holds_and_what_has_no_copy_yet) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto a = tok_run(7600, 336);
    const auto b = tok_run(9600, 336);
    const auto c = tok_run(11600, 336);
    publish(x.f, 1, a, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t);                                  /* a is clean */
    publish(x.f, 2, b, 320);
    x.f.kv.free_sequence(2);                        /* b has no copy yet */
    publish(x.f, 3, c, 320);                        /* c's sequence is still running */

    CHECK_EQ(x.f.pc.trim(1000), 20);                /* a, and only a */
    CHECK_EQ(x.f.pc.restorable_from(a, {}).size(), 20u);
    CHECK_EQ(x.f.pc.lookup(b, {}).n_attn_tokens, 320);
    CHECK_EQ(x.f.pc.lookup(c, {}).n_attn_tokens, 320);
    CHECK_EQ(x.f.pc.stats().drops, 20ll);
    CHECK_EQ(x.f.pc.stats().evictions, 0ll);

    copy_now(x.t);                                  /* b and c land */
    CHECK_EQ(x.f.pc.trim(1000), 20);                /* b; c is still read by its sequence */
    CHECK_EQ(x.f.pc.lookup(c, {}).n_attn_tokens, 320);
    x.f.kv.free_sequence(3);
    CHECK_EQ(x.f.pc.trim(5), 5);                    /* no more than asked */
    CHECK_EQ(x.f.pc.stats().evictions, 0ll);
    CHECK_EQ(x.f.pc.size(), 60ll);                  /* every entry still indexed */
}

TEST(without_tiers_the_trim_frees_only_what_no_sequence_holds) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    const auto a = tok_run(7600, 336);
    const auto b = tok_run(9600, 336);
    publish(f, 1, a, 320);
    f.kv.free_sequence(1);
    publish(f, 2, b, 320);                          /* still running */
    CHECK_EQ(f.pc.trim(1000), 20);
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 0);
    CHECK_EQ(f.pc.lookup(b, {}).n_attn_tokens, 320);
    CHECK_EQ(f.pc.stats().evictions, 20ll);
    CHECK_EQ(f.pc.trim(1000), 0);
}

/* Pressure lands a waiting copy instead of erasing. The binding drops entries with a host copy,
 * then has the engine land in-flight copies and drops those; only what the tiers still cannot
 * take is erased. */
TEST(pressure_lands_a_waiting_copy_instead_of_erasing_the_entry) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto a = tok_run(7600, 336);
    publish(x.f, 1, a, 320);
    x.f.kv.free_sequence(1);                        /* finished; its copy has not landed */
    PCBinding<PrefixCache> b(&x.f.pc);
    int flushes = 0;
    b.flush_fn = [&]() { ++flushes; return !copy_now(x.t).empty(); };
    CHECK_EQ(b.evict(20), 20);
    CHECK_EQ(flushes, 1);
    CHECK_EQ(x.f.pc.stats().evictions, 0ll);
    CHECK_EQ(x.f.pc.stats().drops, 20ll);
    CHECK_EQ(x.f.pc.restorable_from(a, {}).size(), 20u);

    /* With no way to land the copies, the binding erases. */
    const auto c = tok_run(9600, 336);
    publish(x.f, 2, c, 320);
    x.f.kv.free_sequence(2);
    PCBinding<PrefixCache> bare(&x.f.pc);
    CHECK_EQ(bare.evict(20), 20);
    CHECK_EQ(x.f.pc.stats().evictions, 20ll);
}

/* A waiting request's chain stays on the card. An admission refused after its restore gives the
 * hit back with clean entries; while held, the eager drop skips them until the request is gone, so
 * the next attempt finds them in place. */
TEST(a_held_chain_is_not_dropped_until_its_request_lets_go) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7600, 336);
    publish(x.f, 1, prompt, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t);
    x.f.pc.hold(7, prompt, {});
    x.f.pc.hold(7, prompt, {});                     /* once per request */
    CHECK_EQ(x.f.pc.drop_clean(), 0);
    CHECK_EQ(x.f.pc.lookup(prompt, {}).n_attn_tokens, 320);
    /* Pressure still takes it: a held entry has its copy, so trim loses nothing. */
    CHECK_EQ(x.f.pc.trim(5), 5);
    x.f.pc.release(7);
    CHECK_EQ(x.f.pc.drop_clean(), 15);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 0ll);
    CHECK_EQ(x.f.pc.stats().evictions, 0ll);
}

/* Tiers attached but off behave as no tiers: the engine always attaches them, and a trim waiting
 * for a copy nobody will make would never keep the reserve. */
TEST(with_the_tiers_attached_and_off_the_trim_still_frees) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(0, 1024)));
    f.pc.set_tiers(&t);
    const auto a = tok_run(7600, 336);
    publish(f, 1, a, 320);
    f.kv.free_sequence(1);
    CHECK_EQ(f.pc.trim(1000), 20);
    CHECK_EQ(f.kv.pool_bytes_used(G_FULL), 0ll);
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 0);
    CHECK_EQ(f.pc.size(), 0ll);
}

TEST(a_prefix_computed_again_takes_the_new_blocks_as_its_device_copy) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7600, 336);
    publish(x.f, 1, prompt, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t);
    CHECK_EQ(x.f.pc.drop_clean(), 20);
    CHECK_EQ(x.t.arena().used_slots(), 20ll);

    /* A request that could not reach the tier copy computed the same tokens. */
    publish(x.f, 2, prompt, 320);
    CHECK_EQ(x.f.pc.lookup(prompt, {}).n_attn_tokens, 320);
    CHECK_EQ(x.f.pc.resident(), 20ll);
    CHECK_EQ(x.t.arena().used_slots(), 0ll);        /* the older copy is released */
    CHECK(x.t.writeback_waiting());                 /* the new blocks need their own copy */
    const std::vector<int32_t> bt = x.f.kv.block_table(2, G_FULL);
    CHECK_EQ(x.f.kv.block_refcount(G_FULL, bt[0]), 2);
    x.f.kv.free_sequence(2);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 20 * x.per);
}

TEST(evicting_or_clearing_an_entry_on_its_way_back_releases_nothing) {
    /* An entry with a promotion in flight still names its old ids, which the pool may have given
     * to someone else since. */
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7700, 336);
    publish(x.f, 1, prompt, 320);
    x.f.kv.free_sequence(1);
    copy_now(x.t);
    CHECK_EQ(x.f.pc.drop_clean(), 20);

    const std::vector<BlockHash> keys = x.f.pc.restorable_from(prompt, {});
    REQUIRE(!keys.empty());
    CHECK_OK(x.f.kv.add_sequence(2));
    CHECK_OK(x.f.kv.ensure(2, 16));
    const int32_t taken = x.f.kv.block_table(2, G_FULL)[0];

    TierJob up;
    CHECK(x.t.plan_promote(keys[0], &up));          /* out, on its way back */
    x.f.pc.evict_lru(1000);
    CHECK_EQ(x.f.kv.block_refcount(G_FULL, taken), 1);
    x.f.pc.clear();
    CHECK_EQ(x.f.kv.block_refcount(G_FULL, taken), 1);
    CHECK(x.t.doomed(keys[0]));                     /* forgotten, finished off on report */
    CHECK_EQ(x.t.arena().used_slots(), 1ll);        /* the one the promotion is reading */
    x.t.report(up, false);
    CHECK_EQ(x.t.arena().used_slots(), 0ll);
}

/* ================================================================== expiry
 *
 * --prefix-cache-ttl: an entry unused for that long leaves the cache from whichever tier holds it
 * (VRAM blocks to the pool, host slot to the arena), and a snapshot goes with its prefix. Nothing
 * is moved first. Uses are stamped with the monotonic clock, so tests read it between uses and
 * expire against it.
 */

/* The host list is ordered by last use, not arrival (arrival is when VRAM was given up), and a
 * full arena drops the least recently used. */
TEST(a_full_arena_drops_the_least_recently_used_host_entry) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 1024)));
    std::vector<BlockHash> lost;
    t.set_drop_sink([&lost](const BlockHash& h) { lost.push_back(h); });
    t.note_used(bh(1), {11}, 5 * SEC);                 /* used last */
    t.note_used(bh(2), {12}, 1 * SEC);
    CHECK_EQ(copy_now(t).size(), 2u);
    drop_now(t, bh(1));                                /* but arrives on the host list first */
    drop_now(t, bh(2));
    t.note_used(bh(3), {13}, 6 * SEC);
    CHECK_EQ(copy_now(t).size(), 0u);
    std::vector<TierJob> room;
    t.plan(16, &room);
    REQUIRE_EQ(room.size(), 1u);
    CHECK(room[0].key == bh(2));
    t.report(room[0], true);
    REQUIRE_EQ(lost.size(), 1u);
    CHECK(lost[0] == bh(2));
}

/* A conversation whose first block the arena dropped is gone, and so is its session row: no hit
 * reaches past a missing first block, and a stale row would count its blocks as in VRAM forever. */
/* A conversation larger than the host tier keeps its start there. Its blocks share one clock;
 * a copy of its end must not drop its first block (which loses the whole conversation), so the
 * copies past the tier's size are held back and stay in VRAM only. A newer conversation still
 * makes room, from the old one's deepest block. */
TEST(a_full_host_tier_keeps_a_conversation_s_start_and_holds_back_its_end) {
    IdleTiers t;
    IdleTiers::Config c = tier_config(4096, 1024);    /* four slots */
    c.block_tokens = 16;
    CHECK_OK(t.configure(c));
    std::vector<BlockHash> lost;
    t.set_drop_sink([&lost](const BlockHash& h) { lost.push_back(h); });

    for (uint64_t b = 0; b < 6; ++b) t.note_used(bh(1 + b), { (int32_t)(10 + b) }, 1 * SEC, (int64_t)b);
    CHECK_EQ(copy_now(t).size(), 4u);
    CHECK_EQ(t.stats().held_back, 2ll);
    CHECK(!t.starved());
    for (uint64_t b = 0; b < 4; ++b) CHECK(drop_now(t, bh(1 + b)));

    /* The next turn publishes the chain again: the end still finds no entry worth less. */
    for (uint64_t b = 0; b < 6; ++b) t.note_used(bh(1 + b), { (int32_t)(10 + b) }, 2 * SEC, (int64_t)b);
    CHECK_EQ(copy_now(t).size(), 0u);
    CHECK_EQ(t.stats().held_back, 4ll);
    CHECK(!t.starved());
    std::vector<TierJob> room;
    t.plan(16, &room);
    CHECK_EQ(room.size(), 0u);
    CHECK(lost.empty());
    for (uint64_t b = 0; b < 4; ++b) CHECK(t.resident_off_device(bh(1 + b)));
    CHECK(!t.droppable(bh(5)));

    /* A conversation used later takes the old one's deepest host block. */
    t.note_used(bh(20), { 30 }, 3 * SEC, 0);
    CHECK_EQ(copy_now(t).size(), 0u);
    CHECK(t.starved());
    t.plan(16, &room);
    REQUIRE_EQ(room.size(), 1u);
    CHECK(room[0].key == bh(4));
    t.report(room[0], true);
    CHECK_EQ(copy_now(t).size(), 1u);
    CHECK(t.resident_off_device(bh(1)));
}

TEST(a_full_arena_dropping_a_conversation_s_first_block_takes_its_row) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(2048, 1024)));
    t.note_used(bh(1), {11}, 1 * SEC);
    t.note_used(bh(2), {12}, 2 * SEC);
    t.note_session(7, {bh(1), bh(2)}, 32, 2 * SEC);
    CHECK_EQ(copy_now(t).size(), 2u);
    drop_now(t, bh(1));
    drop_now(t, bh(2));
    CHECK_EQ(t.sessions().size(), 1u);
    t.note_used(bh(3), {13}, 3 * SEC);
    CHECK_EQ(copy_now(t).size(), 0u);
    std::vector<TierJob> room;
    t.plan(16, &room);
    REQUIRE_EQ(room.size(), 1u);
    CHECK(room[0].key == bh(1));
    t.report(room[0], true);
    CHECK_EQ(t.sessions().size(), 0u);
}

TEST(the_host_tier_names_what_is_older_than_the_cutoff_oldest_first) {
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(4096, 1024)));
    for (uint64_t i = 1; i <= 3; ++i) t.note_used(bh(i), {(int32_t)i}, (4 - i) * SEC);
    copy_now(t);
    for (uint64_t i = 1; i <= 3; ++i) drop_now(t, bh(i));
    t.note_used(bh(4), {4}, 0);                        /* in VRAM: on the cache's own list */
    std::vector<BlockHash> out;
    t.expired(2 * SEC + 1, 16, &out);
    REQUIRE_EQ(out.size(), 2u);
    CHECK(out[0] == bh(3));                            /* used at 1 s */
    CHECK(out[1] == bh(2));                            /* used at 2 s */
    out.clear();
    t.expired(10 * SEC, 1, &out);                      /* bounded */
    CHECK_EQ(out.size(), 1u);
    /* A record with a job in flight is not named until the job reports. */
    TierJob up;
    REQUIRE(t.plan_promote(bh(3), &up));
    out.clear();
    t.expired(10 * SEC, 16, &out);
    CHECK_EQ(out.size(), 2u);
    CHECK(std::find(out.begin(), out.end(), bh(3)) == out.end());
}

TEST(an_entry_no_request_used_expires_from_vram_and_a_hit_is_a_use) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false, /*ckpt_slots=*/0));
    const auto a = tok_run(7600, 336), b = tok_run(9600, 336);
    const uint64_t t0 = rad_mono_ns();
    publish(f, 1, a, 320);
    f.kv.free_sequence(1);
    publish(f, 2, b, 320);
    f.kv.free_sequence(2);
    const uint64_t mid = rad_mono_ns();
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 320);  /* a hit is a use */
    const int64_t free0 = f.kv.free_blocks(G_FULL);
    const uint64_t ttl = 3600 * SEC;
    CHECK_EQ(f.pc.expire(t0 + ttl, ttl), 0ll);        /* nothing is that old */
    CHECK_EQ(f.pc.expire(mid + ttl, 0), 0ll);         /* 0 keeps everything */
    CHECK_EQ(f.pc.expire(mid + ttl, ttl), 20ll);      /* b, and only b */
    CHECK_EQ(f.kv.free_blocks(G_FULL), free0 + 20);
    CHECK_EQ(f.pc.lookup(b, {}).n_attn_tokens, 0);
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 320);
    CHECK_EQ(f.pc.stats().expired, 20ll);
    CHECK_EQ(f.pc.stats().evictions, 0ll);            /* expiry, not eviction */
}

/* A shared prefix outlives a conversation that expires past it: the other conversation's turn
 * stamped the shared blocks too. */
TEST(a_conversation_expires_from_its_tip_and_a_shared_prefix_stays) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false, /*ckpt_slots=*/0));
    const auto a = tok_run(7600, 336);
    auto b = a;
    for (int i = 160; i < 336; ++i) b[(size_t)i] = 20000 + i;   /* ten blocks shared */
    publish(f, 1, a, 320);
    f.kv.free_sequence(1);
    const uint64_t mid = rad_mono_ns();
    publish(f, 2, b, 320);
    f.kv.free_sequence(2);
    const uint64_t ttl = 60 * SEC;
    CHECK_EQ(f.pc.expire(mid + ttl, ttl), 10ll);      /* a's own ten */
    CHECK_EQ(f.pc.lookup(a, {}).n_attn_tokens, 160);
    CHECK_EQ(f.pc.lookup(b, {}).n_attn_tokens, 320);
}

TEST(an_entry_only_host_memory_holds_expires_and_its_slot_goes_back) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto a = tok_run(7600, 336);
    publish(x.f, 1, a, 320);
    x.f.kv.free_sequence(1);
    CHECK_EQ(copy_now(x.t).size(), 20u);
    CHECK_EQ(x.f.pc.drop_clean(), 20);
    CHECK_EQ(x.f.pc.restorable_from(a, {}).size(), 20u);
    CHECK_EQ(x.t.arena().used_slots(), 20);
    CHECK_EQ(x.t.sessions().size(), 1u);
    const uint64_t later = rad_mono_ns();
    const uint64_t ttl = 60 * SEC;
    CHECK_EQ(x.f.pc.expire(later + ttl, ttl), 20ll);
    CHECK_EQ(x.t.arena().used_slots(), 0);
    CHECK_EQ(x.f.pc.size(), 0ll);
    CHECK_EQ(x.f.pc.restorable_from(a, {}).size(), 0u);
    CHECK_EQ(x.t.sessions().size(), 0u);              /* the row goes with its first block */
    CHECK_EQ(x.f.pc.stats().evictions, 0ll);
}

/* A snapshot expires with its prefix, but one handed out by a lookup this step is kept until the
 * step's admission has restored from it. */
TEST(a_snapshot_expires_with_its_prefix_and_its_slot_goes_back) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/true, /*ckpt_slots=*/4));
    const auto prompt = tok_run(7000, 336);
    publish(f, 1, prompt, 320);
    f.kv.free_sequence(1);
    const std::vector<BlockHash> hs = chain_of(prompt);
    const int64_t free0 = f.kv.free_checkpoint_slots();
    ckpt_at(f, 1, hs, 128);
    ckpt_at(f, 1, hs, 256);
    CHECK_EQ(f.pc.lookup(prompt, {}).n_linear_tokens, 256);
    const uint64_t later = rad_mono_ns();
    const uint64_t ttl = 60 * SEC;
    CHECK_EQ(f.pc.expire(later + ttl, ttl), 20ll);
    CHECK(f.pc.checkpoints(1) == (std::vector<int64_t>{ 256 }));
    CHECK_EQ(f.pc.stats().ckpt_expired, 1ll);
    f.pc.begin_step();
    f.pc.expire(later + ttl, ttl);
    CHECK(f.pc.checkpoints(1).empty());
    CHECK_EQ(f.kv.free_checkpoint_slots(), free0);
    CHECK_EQ(f.pc.sessions_tracked(), 0ll);
    CHECK_EQ(f.pc.stats().ckpt_evictions, 0ll);
}

/* ================================================================== the cache-only count
 *
 * KVManager updates the count of blocks only the cache holds at every reference change, so reading
 * it is free. Maintained at several call sites, it can drift, so these check it after every
 * operation against a recount: a block at refcount one that no live sequence holds is the
 * cache's.
 */
static int64_t cache_only_recount(Fixture& f, const std::vector<uint64_t>& live) {
    const int64_t nb = f.kv.plan(G_FULL)->n_blocks;
    std::vector<uint8_t> in_seq((size_t)nb, 0);
    for (uint64_t id : live)
        for (int32_t b : f.kv.block_table(id, G_FULL)) in_seq[(size_t)b] = 1;
    int64_t n = 0;
    for (int64_t b = 0; b < nb; ++b)
        if (f.kv.block_refcount(G_FULL, (int32_t)b) == 1 && !in_seq[(size_t)b]) ++n;
    return n;
}

TEST(the_cache_only_count_follows_every_reference_change) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false));
    uint64_t x = 0x2545f4914f6cdd1dull;
    auto rnd = [&x](uint64_t n) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x % n; };
    std::vector<uint64_t> live;
    uint64_t next_id = 1;
    int64_t hits = 0, forks = 0, evicted = 0, bad = 0;
    for (int it = 0; it < 4000; ++it) {
        const uint64_t op = rnd(6);
        if (op <= 1 && live.size() < 8) {
            /* A prompt from one of six families, so it hits what its family published before. */
            const auto prompt = tok_run(1000 * (int32_t)rnd(6), 16 + (int)rnd(160));
            const uint64_t id = next_id++;
            CHECK_OK(f.kv.add_sequence(id));
            std::vector<std::vector<int32_t>> found;
            const CacheHit h = f.pc.lookup(prompt, {}, &found);
            if (h.n_attn_tokens > 0) {
                CHECK_OK(f.kv.adopt(id, G_FULL, found[0], h.n_attn_tokens));
                ++hits;
            }
            int st = f.kv.ensure(id, (int64_t)prompt.size());
            if (st == RAD_E_FULL) { evicted += f.pc.evict_lru(64); st = f.kv.ensure(id, (int64_t)prompt.size()); }
            if (st != RAD_OK) { f.kv.free_sequence(id); continue; }
            std::vector<std::vector<int32_t>> bt{ f.kv.block_table(id, G_FULL) };
            CHECK_OK(f.pc.insert(id, prompt, {}, bt, (int64_t)prompt.size()));
            live.push_back(id);
        } else if (op == 2 && !live.empty() && live.size() < 8) {
            const uint64_t id = next_id++;
            if (f.kv.fork(live[(size_t)rnd(live.size())], id) == RAD_OK) { live.push_back(id); ++forks; }
        } else if (op <= 4 && !live.empty()) {
            const size_t k = (size_t)rnd(live.size());
            f.kv.free_sequence(live[k]);
            live.erase(live.begin() + (ptrdiff_t)k);
        } else if (op == 5) {
            evicted += f.pc.evict_lru((int64_t)rnd(9));
        }
        if (f.pc.unshared_tokens() != 16 * cache_only_recount(f, live)) ++bad;
    }
    CHECK_EQ(bad, 0ll);
    /* Every kind of change must have occurred for the loop to mean anything. */
    CHECK(hits > 100);
    CHECK(forks > 100);
    CHECK(evicted > 100);
    for (uint64_t id : live) f.kv.free_sequence(id);
    live.clear();
    CHECK_EQ(f.pc.unshared_tokens(), 16 * f.pc.resident());
    f.pc.clear();
    CHECK_EQ(f.pc.unshared_tokens(), 0ll);
    CHECK_EQ(cache_only_recount(f, live), 0ll);
}

TEST(the_cache_only_count_follows_a_drop_and_a_restore) {
    TierFixture x;
    CHECK_OK(x.build());
    const auto prompt = tok_run(7600, 336);
    publish(x.f, 1, prompt, 320);                   /* twenty blocks, read by sequence 1 */
    CHECK_EQ(x.f.pc.unshared_tokens(), 0ll);
    x.f.kv.free_sequence(1);
    CHECK_EQ(x.f.pc.unshared_tokens(), 320ll);

    copy_now(x.t);
    CHECK_EQ(x.f.pc.drop_clean(), 20);              /* off the card: none counts */
    CHECK_EQ(x.f.pc.unshared_tokens(), 0ll);

    /* A restore reserves blocks and hands each reservation to the cache as its reference. */
    const std::vector<BlockHash> keys = x.f.pc.restorable_from(prompt, {});
    REQUIRE(keys.size() == 20u);
    int64_t handed = 0;
    for (const BlockHash& k : keys) {
        int32_t b = -1;
        CHECK_OK(x.f.kv.reserve_block(G_FULL, &b));
        CHECK_EQ(x.f.pc.unshared_tokens(), 16 * handed);   /* a reservation is not the cache's */
        CHECK_OK(x.f.pc.rebind(k, { b }));
        ++handed;
    }
    CHECK_EQ(x.f.pc.unshared_tokens(), 320ll);

    std::vector<std::vector<int32_t>> found;
    const CacheHit h = x.f.pc.lookup(prompt, {}, &found);
    REQUIRE(h.n_attn_tokens == 320);
    CHECK_OK(x.f.kv.add_sequence(2));
    CHECK_OK(x.f.kv.adopt(2, G_FULL, found[0], h.n_attn_tokens));
    CHECK_EQ(x.f.pc.unshared_tokens(), 0ll);
    x.f.kv.free_sequence(2);
    CHECK_EQ(x.f.pc.unshared_tokens(), 320ll);
    x.f.pc.clear();
    CHECK_EQ(x.f.pc.unshared_tokens(), 0ll);
    CHECK_EQ(x.f.kv.pool_bytes_used(G_FULL), 0ll);
}

/* ================================================================== the tier names
 *
 * People and scrapers read these on the sessions view, so names are stable and distinct. "?"
 * marks an enum value this switch does not know, rather than a neighbouring tier's name.
 */
TEST(every_tier_has_its_own_stable_name) {
    CHECK_EQ(std::string(kv_tier_name(KVTier::Device)), std::string("device"));
    CHECK_EQ(std::string(kv_tier_name(KVTier::Host)), std::string("host"));
    /* A value outside the enum reads "?". */
    CHECK_EQ(std::string(kv_tier_name((KVTier)99)), std::string("?"));
}

/* ================================================================== the loan */

/* A restore is sized before it starts, so it recalls the loan for the whole chain at once; a
 * recall per block would drain the slab a buffer at a time. */
TEST(a_restore_reserves_its_whole_chain_against_one_recall) {
    Lender e;
    CHECK_OK(e.build());
    const int64_t all = e.kv.total_blocks(G_FULL);
    CHECK_OK(e.kv.set_loan(e.kv.spare_tokens()));
    const int64_t idle = e.kv.live_blocks(G_FULL);
    int calls = 0;
    e.kv.set_reclaim([&](int64_t) { ++calls; return (int)RAD_OK; });

    const int64_t want = (idle + all) / 2;
    std::vector<int32_t> got;
    CHECK_EQ(e.kv.reserve_blocks(G_FULL, want, &got), want);
    CHECK_EQ((int64_t)got.size(), want);
    CHECK_EQ(calls, 1);
    for (int32_t b : got) CHECK(b < e.kv.live_blocks(G_FULL));

    /* Past the carve it takes what is left and returns the count. */
    std::vector<int32_t> more;
    const int64_t n = e.kv.reserve_blocks(G_FULL, all, &more);
    CHECK_EQ(n, all - want);
    CHECK_EQ((int64_t)more.size(), n);
}

/* ================================================================== the transfer, end to end
 *
 * The byte movement is where another conversation's context could leak into a restored entry:
 * layer-major addressing, the staging pitch, batches cut into staging-sized pieces, non-consecutive
 * arena slots, and under tensor parallelism each rank's window of a shared slot. A chain is copied
 * out through the real TierExec, its VRAM dropped and overwritten, restored into different blocks,
 * and every byte of every layer compared. The host backend's streams are real threaded queues, so
 * missing ordering between copies and row kernels shows up as wrong bytes.
 */
static std::string rows_plugin() {
    return home() + "/testrows/radtest_rows.so";
}

/* The byte a pool holds at (rank, layer, block, i) before anything moves. A full avalanche, so
 * neighbouring blocks, layers and bytes all differ; a multiply-only hash would leave the low byte
 * unchanged for small input differences, which a wrong offset could reproduce. */
static unsigned char pat(int rank, int64_t layer, int32_t block, int64_t i) {
    uint64_t z = ((uint64_t)rank << 60) ^ ((uint64_t)layer << 48) ^ ((uint64_t)block << 20) ^
                 (uint64_t)i;
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return (unsigned char)(z ^ (z >> 31));
}

/* `alike`: a replicated group holds rank 0's bytes on every rank, as rad_kv_group_replicated
 * promises. Without it the ranks differ there on purpose, so a follower restored from its own
 * gather rather than the leader's window would show. */
static void fill(Fixture& f, int rank, bool alike = false) {
    for (int32_t g : f.pc.cached_groups()) {
        const KVGroupPlan* p = f.kv.plan(g);
        if (p->zero) continue;
        const int r = alike && p->replicated ? 0 : rank;
        const int64_t frag = p->layer_bytes_per_block();
        for (int64_t l = 0; l < p->n_layers; ++l)
            for (int32_t b = 0; b < (int32_t)p->n_blocks; ++b) {
                unsigned char* row = (unsigned char*)p->base + l * p->layer_stride() + b * frag;
                for (int64_t i = 0; i < frag; ++i) row[i] = pat(r, g * 16 + l, b, i);
            }
    }
}

static void scribble(Fixture& f) {
    for (int32_t g : f.pc.cached_groups()) {
        const KVGroupPlan* p = f.kv.plan(g);
        if (p->zero) continue;
        for (int64_t l = 0; l < p->n_layers; ++l)
            std::memset((char*)p->base + l * p->layer_stride(), 0xEE, (size_t)p->layer_stride());
    }
}

/* The number of entries whose blocks in `now` hold, in every cached group and layer, exactly what
 * their blocks in `was` held before the move: this rank's own bytes, and for a replicated group
 * rank 0's, the one copy the slot holds. An entry is blocks_per_entry(gi) blocks of group gi. */
static int64_t same_bytes(Fixture& f, int rank, const std::vector<std::vector<int32_t>>& was,
                          const std::vector<std::vector<int32_t>>& now) {
    const std::vector<int32_t>& cg = f.pc.cached_groups();
    int64_t good = 0;
    const int64_t n = (int64_t)was[0].size() / f.pc.blocks_per_entry(0);
    for (int64_t e = 0; e < n; ++e) {
        bool ok = true;
        for (size_t gi = 0; gi < cg.size() && ok; ++gi) {
            const KVGroupPlan* p = f.kv.plan(cg[gi]);
            if (p->zero) continue;
            const int r = p->replicated ? 0 : rank;
            const int64_t frag = p->layer_bytes_per_block(), per = f.pc.blocks_per_entry(gi);
            for (int64_t k = e * per; k < (e + 1) * per && ok; ++k)
                for (int64_t l = 0; l < p->n_layers && ok; ++l) {
                    const unsigned char* row = (const unsigned char*)p->base +
                                               l * p->layer_stride() + now[gi][(size_t)k] * frag;
                    for (int64_t i = 0; i < frag && ok; ++i)
                        ok = row[i] == pat(r, cg[gi] * 16 + l, was[gi][(size_t)k], i);
                }
        }
        good += ok;
    }
    return good;
}

/* Two paged groups of different widths, as in production: the attention cache beside a narrower
 * per-block key cache. An entry is a block in each, so confused offsets within a staging entry
 * would swap group bytes. */
static std::vector<KVGroupInfo> two_paged_groups() {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    KVGroupInfo a;
    a.name = "attn";
    a.decl.kind = RAD_KV_FULL;
    a.decl.dtype = RAD_BF16;
    a.decl.n_head_kv = 2;
    a.decl.head_dim = 8;
    a.block_size = 16;
    a.layers = { 0, 1 };
    g.push_back(a);
    KVGroupInfo k;
    k.name = "bkey";
    k.decl.kind = RAD_KV_FULL;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = 1;
    k.decl.head_dim = 4;
    k.block_size = 16;
    k.layers = { 2, 3, 4 };
    g.push_back(k);
    return g;
}

/* `zero_last`: the last rank keeps no attention cache (rad_kv_group_zero), so its window of a slot
 * is the narrower group alone and the windows are unequal; every rank still takes part in every
 * job. `replicated`: the attention group is the same on every rank (rad_kv_group_replicated), so
 * the slot holds it once at its front, written by rank 0 and read back by all. `verify`: under
 * RADIANCE_DEBUG_TIER_VERIFY each follower's gather of the replicated group is compared with the
 * leader's window; `alike` fills the replicated group the same on every rank, as the flag
 * promises, so the comparison passes, and without it every entry is caught. */
struct TransferCase {
    int  world = 1;
    bool fragment = false;
    bool zero_last = false;
    bool replicated = false;
    bool alike = false;
    bool verify = false;
    /* The key cache at 8-token blocks: an entry is one attention block and two key blocks, and
     * the slot, the staging entry and the jobs all go by the entry's blocks. */
    bool mixed = false;
};
static void run_transfer(const TransferCase& tc) {
    const int world = tc.world;
    const bool fragment = tc.fragment, zero_last = tc.zero_last;
    if (tc.verify) setenv("RADIANCE_DEBUG_TIER_VERIFY", "1", 1);
    else unsetenv("RADIANCE_DEBUG_TIER_VERIFY");
    /* Host backend only: the stand-in kernels copy rows on the CPU, where host-backend "device"
     * memory lives; on a card the staging buffer is VRAM and production kernels apply. */
    if (!device_is_host()) {
        std::printf("  SKIP  the tier transfer needs the host backend (%s here)\n",
                    device_backend_name());
        return;
    }
    Registry reg;
    REQUIRE_EQ(reg.load_plugin(rows_plugin(), 0), RAD_OK);
    CHECK_OK(reg.check_complete());

    std::vector<std::unique_ptr<Fixture>> fx;
    for (int r = 0; r < world; ++r) {
        fx.push_back(std::make_unique<Fixture>());
        fx.back()->cfg.checkpoint_slots = 0;
        std::vector<KVGroupInfo> groups = two_paged_groups();
        /* The zero group is the one not replicated: attention, or the key cache beside a
         * replicated attention group, which leaves that rank's own window empty. */
        if (zero_last && r == world - 1) groups[tc.replicated ? 2 : 1].zero = true;
        if (tc.replicated) groups[1].replicated = true;
        if (tc.mixed) groups[2].block_size = 8;
        REQUIRE_EQ(fx.back()->build_groups(groups), RAD_OK);
    }
    Fixture& f = *fx[0];
    REQUIRE_EQ(f.pc.cached_groups().size(), 2u);
    /* One group per block an entry holds (PrefixCache::entry_groups), as bringup passes them. */
    const std::vector<int32_t>& cg = f.pc.entry_groups();
    REQUIRE_EQ(cg.size(), tc.mixed ? 3u : 2u);
    /* The slot as bringup lays it out: a replicated group once at the front, then each rank's own
     * window following the previous ones. Checked against the plain sums here. */
    std::vector<const KVManager*> kvs;
    for (auto& fp : fx) kvs.push_back(&fp->kv);
    TierExec::SlotLayout L;
    REQUIRE_EQ(TierExec::slot_layout(kvs, cg, &L), RAD_OK);
    const int64_t attn = f.kv.plan(cg[0])->bytes_per_block;
    int64_t sum = 0;
    for (int r = 0; r < world; ++r)
        for (int32_t g : cg) sum += fx[(size_t)r]->kv.plan(g)->bytes_per_block;
    CHECK_EQ(L.rep_bytes, tc.replicated ? attn : 0);
    CHECK_EQ(L.slot_bytes, tc.replicated ? sum - (world - 1) * attn : sum);
    CHECK_EQ(L.own_off[0], L.rep_bytes);
    for (int r = 1; r < world; ++r)
        CHECK_EQ(L.own_off[(size_t)r], L.own_off[(size_t)r - 1] + L.own_bytes[(size_t)r - 1]);
    CHECK_EQ(L.own_off[(size_t)world - 1] + L.own_bytes[(size_t)world - 1], L.slot_bytes);
    const std::vector<int64_t>& win_off = L.own_off;
    const int64_t slot = L.slot_bytes;

    /* Forty slots. Fragmented, every other one is taken and the copy goes out as twenty runs of
     * one; whole, as one run of twenty, where a rank's pitch through the shared slots decides where
     * its rows land. */
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(40 * slot, slot)));
    REQUIRE_EQ(t.arena().capacity(), 40);
    if (fragment) {
        for (int i = 0; i < 40; ++i) CHECK_EQ(t.arena().alloc(), i);
        for (int i = 1; i < 40; i += 2) t.arena().free(i);
    }
    const int64_t held_before = t.arena().used_slots();
    f.pc.set_tiers(&t);

    std::vector<RadStream> compute((size_t)world, nullptr);
    std::vector<std::unique_ptr<TierExec>> x;
    for (int r = 0; r < world; ++r) {
        REQUIRE_EQ(rad_stream_create(&compute[(size_t)r], 0), RAD_OK);
        x.push_back(std::make_unique<TierExec>());
        /* A three-entry staging buffer, so twenty entries cross it in seven pieces. */
        REQUIRE_EQ(x.back()->configure(&reg, &fx[(size_t)r]->kv, &t, cg, /*max_batch=*/3,
                                       /*max_copy=*/64, /*max_chain=*/64, /*leader=*/r == 0,
                                       /*device=*/0, /*slot_off=*/win_off[(size_t)r],
                                       /*shared=*/nullptr), RAD_OK);
    }

    /* Skew the groups' block ids: they allocate in step, so an index row read for the wrong group
     * would otherwise be right by accident. */
    int32_t skew = -1;
    CHECK_OK(f.kv.reserve_block(cg[1], &skew));

    const auto prompt = tok_run(4100, 336);
    publish(f, 1, prompt, 320);
    f.kv.free_sequence(1);
    std::vector<std::vector<int32_t>> hit;
    REQUIRE_EQ(f.pc.lookup(prompt, {}, &hit).n_attn_tokens, 320);
    const std::vector<std::vector<int32_t>> was = hit;
    CHECK(was[0][0] != was[1][0]);
    CHECK_EQ((int64_t)was[1].size(), tc.mixed ? 40ll : 20ll);
    for (int r = 0; r < world; ++r) fill(*fx[(size_t)r], r, tc.alike);

    /* ---- out ---- */
    std::vector<TierJob> jobs;
    t.plan_writeback(64, &jobs);
    REQUIRE_EQ(jobs.size(), 20u);
    for (int r = 0; r < world; ++r)
        CHECK_OK(x[(size_t)r]->issue_copy(jobs, compute[(size_t)r]));
    for (int r = 0; r < world; ++r) {
        int st = 0;
        for (int spin = 0; spin < 20000 && (st = x[(size_t)r]->copy_state()) == 0; ++spin)
            usleep(100);
        CHECK_EQ(st, 1);
    }
    /* Every rank landed: the follower's hash of the replicated group against the leader's window
     * (Engine::tier_tick's order). Bytes out: a follower writes only its own window. */
    if (tc.verify && tc.replicated) {
        for (int r = 1; r < world; ++r)
            for (const TierJob& j : jobs) {
                uint64_t sum = 0;
                REQUIRE(x[(size_t)r]->take_replicated_sum(j.key, &sum));
                x[0]->verify_replicated(j, r, sum);
                CHECK(!x[(size_t)r]->take_replicated_sum(j.key, &sum));
            }
        CHECK_EQ(x[0]->stats().rep_checked, 20ll * (world - 1));
        CHECK_EQ(x[0]->stats().rep_mismatch, tc.alike ? 0ll : 20ll * (world - 1));
    }
    for (int r = 0; r < world; ++r)
        CHECK_EQ(x[(size_t)r]->stats().bytes_out,
                 20 * (r == 0 ? L.rep_bytes + L.own_bytes[0] : L.own_bytes[(size_t)r]));
    x[0]->report_pass(jobs, std::vector<char>(jobs.size(), 1), 0);
    CHECK_EQ(t.stats().to_host, 20);

    /* ---- the VRAM is dropped and overwritten ---- */
    CHECK_EQ(f.pc.drop_clean(), 20);
    for (auto& fp : fx) scribble(*fp);
    /* A new sequence takes the lowest blocks, so the chain cannot return to its old blocks. */
    CHECK_OK(f.kv.add_sequence(2));
    CHECK_OK(f.kv.ensure(2, 5 * 16));

    /* ---- back ---- */
    const std::vector<BlockHash> keys = f.pc.restorable_from(prompt, {});
    REQUIRE_EQ(keys.size(), 20u);
    std::vector<TierJob> up;
    for (const BlockHash& k : keys) {
        TierJob j;
        REQUIRE(x[0]->plan_promote(k, &j));
        up.push_back(j);
    }
    std::vector<char> ok(up.size(), 1);
    for (int r = 0; r < world; ++r) {
        std::vector<char> mine;
        CHECK_OK(x[(size_t)r]->issue_restore(up, compute[(size_t)r], &mine));
        for (size_t i = 0; i < ok.size(); ++i) ok[i] = ok[i] && i < mine.size() && mine[i];
    }
    for (char c : ok) CHECK(c);
    x[0]->report_pass(up, ok, 0);
    for (const TierJob& j : up) CHECK_OK(f.pc.rebind(j.key, j.blocks));
    /* The reading step waits on the device, not the host: drain the compute stream as a reader
     * would. */
    for (int r = 0; r < world; ++r) CHECK_OK(rad_stream_sync(compute[(size_t)r]));

    REQUIRE_EQ(f.pc.lookup(prompt, {}, &hit).n_attn_tokens, 320);
    const std::vector<std::vector<int32_t>> now = hit;
    int64_t moved = 0;
    for (size_t k = 0; k < was[0].size(); ++k) moved += was[0][k] != now[0][k];
    CHECK(moved > 0);
    for (int r = 0; r < world; ++r)
        CHECK_EQ(same_bytes(*fx[(size_t)r], r, was, now), 20);
    /* The host copy stays, so the entries are clean on landing. */
    CHECK_EQ(t.arena().used_slots(), held_before + 20);
    for (const BlockHash& k : keys) CHECK(t.droppable(k));

    for (auto& xp : x) xp->close();
    for (RadStream s : compute) rad_stream_destroy(s);
    f.pc.set_tiers(nullptr);
    unsetenv("RADIANCE_DEBUG_TIER_VERIFY");
}

TEST(a_conversation_restored_from_host_is_byte_identical_in_new_blocks) {
    run_transfer({ 1, /*fragment=*/false });
    run_transfer({ 1, /*fragment=*/true });
}
TEST(each_rank_restores_its_own_window_of_a_shared_slot) {
    run_transfer({ 2, /*fragment=*/false });
    run_transfer({ 2, /*fragment=*/true });
}
/* An entry over groups at different block sizes moves as its blocks: one attention block and two
 * key-cache blocks a slot, each through its own staging window, and every byte comes back. With a
 * replicated attention group beside a zero rank the windows still stay apart. */
TEST(an_entry_over_mixed_block_sizes_moves_through_the_tiers_block_by_block) {
    TransferCase tc;
    tc.mixed = true;
    for (int world : { 1, 2 })
        for (bool fragment : { false, true }) {
            tc.world = world;
            tc.fragment = fragment;
            run_transfer(tc);
        }
    tc.world = 3;
    tc.fragment = false;
    tc.zero_last = true;
    tc.replicated = true;
    run_transfer(tc);
}
TEST(a_rank_with_a_zero_group_moves_only_its_own_groups) {
    run_transfer({ 3, /*fragment=*/false, /*zero_last=*/true });
    run_transfer({ 3, /*fragment=*/true, /*zero_last=*/true });
}
/* The replicated group is in the slot once: the leader writes it, every rank reads it back
 * beside its own shard of the other group, and the slot is smaller by (world - 1) copies. At one
 * rank nothing changes but the layout's bookkeeping. The ranks' copies differ on purpose, so the
 * bytes a follower gets back are provably the leader's. */
TEST(a_replicated_group_is_held_once_a_slot_and_restored_on_every_rank) {
    for (int world : { 1, 2, 8 }) {
        run_transfer({ world, /*fragment=*/false, false, /*replicated=*/true });
        run_transfer({ world, /*fragment=*/true, false, /*replicated=*/true });
    }
}
/* A replicated group beside a zero one: the last rank's own window is empty, so it writes nothing
 * on a copy out and reads the replicated window alone on a restore. */
TEST(a_replicated_group_beside_a_zero_rank_keeps_the_windows_apart) {
    run_transfer({ 3, /*fragment=*/false, /*zero_last=*/true, /*replicated=*/true });
    run_transfer({ 3, /*fragment=*/true, /*zero_last=*/true, /*replicated=*/true });
}
/* RADIANCE_DEBUG_TIER_VERIFY: a follower hashes the replicated bytes it would have written and
 * the leader compares them with its window. Alike, nothing is reported (seven followers at eight
 * ranks); differing, every entry, which the two-rank arm logs twenty times. */
TEST(tier_verify_catches_a_follower_whose_replicated_group_differs) {
    run_transfer({ 2, /*fragment=*/false, false, /*replicated=*/true, /*alike=*/true, /*verify=*/true });
    run_transfer({ 8, /*fragment=*/true, false, /*replicated=*/true, /*alike=*/true, /*verify=*/true });
    run_transfer({ 2, /*fragment=*/true, false, /*replicated=*/true, /*alike=*/false, /*verify=*/true });
}
/* The flag must agree across the ranks: one rank's replicated group that another rank declared
 * zero or did not declare replicated has no one window every rank can read, so bringup stops. */
TEST(a_group_replicated_on_one_rank_only_is_refused_by_the_slot_layout) {
    if (!device_is_host()) {
        std::printf("  SKIP  the tier layout fixture needs the host backend\n");
        return;
    }
    for (int bad : { 0, 1 }) {
        std::vector<std::unique_ptr<Fixture>> fx;
        for (int r = 0; r < 2; ++r) {
            fx.push_back(std::make_unique<Fixture>());
            fx.back()->cfg.checkpoint_slots = 0;
            std::vector<KVGroupInfo> groups = two_paged_groups();
            groups[1].replicated = r == 0 || bad == 1;
            if (bad == 1 && r == 1) groups[1].zero = true;
            REQUIRE_EQ(fx.back()->build_groups(groups), RAD_OK);
        }
        std::vector<const KVManager*> kvs;
        for (auto& fp : fx) kvs.push_back(&fp->kv);
        TierExec::SlotLayout L;
        CHECK_EQ(TierExec::slot_layout(kvs, fx[0]->pc.cached_groups(), &L), RAD_E_STATE);
    }
}

/* With speculation each sequence has two physical state slots while the free list counts logical
 * ones; mixing them would read an idle pool as half full. */
TEST(kv_bytes_in_use_count_a_speculating_state_once_per_sequence) {
    Fixture f;
    f.cfg.n_spec = 2;
    CHECK_OK(f.build(/*with_linear=*/true));
    const KVGroupPlan* ps = f.kv.plan(G_GDN);
    REQUIRE(ps != nullptr);
    CHECK_EQ(ps->state_copies, 2ll);

    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), 0ll);
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16));
    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), 2 * ps->bytes_per_state);
    f.kv.free_sequence(1);
    CHECK_EQ(f.kv.pool_bytes_used(G_GDN), 0ll);
}

/* ================================================================== randomised checks
 *
 * Seeded xorshift. A failing check prints its seed and the operation that broke it, so a rerun
 * replays it exactly.
 */
struct Xorshift {
    uint64_t x;
    explicit Xorshift(uint64_t seed) : x(seed ? seed : 0x9e3779b97f4a7c15ull) {}
    uint64_t next() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; }
    uint64_t below(uint64_t n) { return n ? next() % n : 0; }
};

/* ================================================================== the host arena */

/* The arena backs the snapshot store and the KV host tier, so a slot handed out twice puts two
 * sessions' state in one slot. It is checked against its documented rule played on a plain
 * vector: the first free slot at or after the cursor, wrapping once, extended while free and
 * inside the capacity, with the cursor left just past the run. The capacities end the bitmap one
 * short of, on, one past and inside a 64-bit word, where the full-word skip and the partial last
 * word apply. */
TEST(the_host_arena_hands_out_runs_by_its_rule_across_word_boundaries) {
    const uint64_t seed = 0x51ed2701a3c4b5d7ull;
    for (int64_t cap : { 63, 64, 65, 200 }) {
        HostArena a;
        REQUIRE_EQ(a.open(16, 16 * cap), RAD_OK);
        REQUIRE_EQ(a.capacity(), cap);
        std::vector<char> used((size_t)cap, 0);
        int64_t cursor = 0, n_used = 0, fulls = 0, wraps = 0, cut = 0;
        Xorshift r(seed ^ (uint64_t)cap);
        bool bad = false;
        for (int op = 0; op < 5000 && !bad; ++op) {
            /* Alternate allocation-heavy and free-heavy phases: the first fills the arena, the
             * second fragments it. */
            const bool alloc = r.below(10) < ((op / 500) % 2 ? 4u : 7u);
            if (alloc) {
                const int32_t want = 1 + (int32_t)r.below(9);
                int64_t s = -1;
                if (n_used < cap) {
                    for (int64_t i = cursor; i < cap && s < 0; ++i) if (!used[(size_t)i]) s = i;
                    for (int64_t i = 0; i < cursor && s < 0; ++i) if (!used[(size_t)i]) s = i;
                }
                int64_t n = 0;
                if (s >= 0) {
                    while (n < want && s + n < cap && !used[(size_t)(s + n)]) used[(size_t)(s + n++)] = 1;
                    wraps += s < cursor;
                    cut += n < want;
                    n_used += n;
                    cursor = s + n >= cap ? 0 : s + n;
                }
                int32_t got = -7;
                const int32_t at = a.alloc_run(want, &got);
                if (at != s || got != n) {
                    fprintf(stderr, "    seed %#llx cap %lld op %d: alloc_run(%d) gave slot %d x %d, "
                            "the rule gives %lld x %lld\n", (unsigned long long)(seed ^ (uint64_t)cap),
                            (long long)cap, op, want, at, got, (long long)s, (long long)n);
                    bad = true;
                }
            } else {
                const int32_t slot = (int32_t)r.below((uint64_t)cap);
                if (used[(size_t)slot]) { used[(size_t)slot] = 0; --n_used; }
                a.free(slot);                         /* a double free is ignored */
            }
            fulls += n_used == cap;
            bool same = a.used_slots() == n_used && a.full() == (n_used == cap) &&
                        a.free_slots() == cap - n_used;
            for (int64_t i = 0; i < cap && same; ++i) same = a.in_use((int32_t)i) == (used[(size_t)i] != 0);
            if (!same && !bad) {
                fprintf(stderr, "    seed %#llx cap %lld op %d: the arena's use count or bitmap "
                        "left the model (%lld used, model %lld)\n",
                        (unsigned long long)(seed ^ (uint64_t)cap), (long long)cap, op,
                        (long long)a.used_slots(), (long long)n_used);
                bad = true;
            }
        }
        CHECK(!bad);
        /* The run must have filled the arena, wrapped the cursor and cut a run at a used slot. */
        CHECK(fulls > 0);
        CHECK(wraps > 0);
        CHECK(cut > 0);
        CHECK(!a.in_use(-1));
        CHECK(!a.in_use((int32_t)cap));
    }
}

/* ================================================================== KV manager: every state kind */

/* A hybrid model with every state kind: full attention over 8 layers, a 64-token sliding window
 * over 1, linear state over 2 and a conv window over 1. In a 1 MiB pool at max_tok 256 the full
 * group carves 116 blocks and the window group its cap of 64, small enough that requests run out
 * of blocks. */
enum : int32_t { HG_FULL = 1, HG_WIN = 2, HG_LIN = 3, HG_CONV = 4 };
static std::vector<KVGroupInfo> every_kind_groups() {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    g.push_back(paged_group("full", RAD_KV_FULL, 2, 16, { 0, 1, 2, 3, 4, 5, 6, 7 }));
    g.push_back(paged_group("swa", RAD_KV_WINDOW, 2, 16, { 8 }, 64));
    KVGroupInfo lin;
    lin.name = "gdn";
    lin.decl.kind = RAD_KV_LINEAR;
    lin.decl.dtype = RAD_F32;
    lin.decl.n_head_kv = 2;
    lin.decl.state_dim[0] = 8;
    lin.decl.state_dim[1] = 8;
    lin.layers = { 9, 10 };
    g.push_back(lin);
    KVGroupInfo cv;
    cv.name = "conv";
    cv.decl.kind = RAD_KV_CONV;
    cv.decl.dtype = RAD_BF16;
    cv.decl.n_head_kv = 2;
    cv.decl.head_dim = 8;
    cv.decl.conv_width = 4;
    cv.layers = { 11 };
    g.push_back(cv);
    return g;
}

/* The hybrid fixture's groups plus a conv window over one layer (handle 3). */
enum : int32_t { G_CV = 3 };
static std::vector<KVGroupInfo> hybrid_conv_groups() {
    std::vector<KVGroupInfo> g = hybrid_groups();
    KVGroupInfo cv;
    cv.name = "conv";
    cv.decl.kind = RAD_KV_CONV;
    cv.decl.dtype = RAD_BF16;
    cv.decl.n_head_kv = 2;
    cv.decl.head_dim = 8;
    cv.decl.conv_width = 4;
    cv.layers = { 6 };
    g.push_back(cv);
    return g;
}

static int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

/* A sequence's block-table rows as the card holds them, staged as BatchBuilder::build_groups
 * stages them: a row seen before rewrites from min(first changed entry, old length, new length);
 * a new row is written whole. Comparing the result with the manager's table proves the change
 * marks name every entry a mutation touched. */
struct DeviceRows {
    std::map<std::pair<uint64_t, int32_t>, std::vector<int32_t>> rows;

    /* Returns the change mark the manager gave. */
    int64_t stage(const KVManager& kv, uint64_t seq, int32_t g) {
        const std::vector<int32_t>& bt = kv.block_table(seq, g);
        const int64_t chg = kv.take_table_changes(seq, g);
        const int64_t nb = (int64_t)bt.size();
        auto it = rows.find({ seq, g });
        int64_t lo = 0;
        if (it != rows.end()) lo = std::min(std::min(chg, (int64_t)it->second.size()), nb);
        std::vector<int32_t>& row = rows[{ seq, g }];
        row.resize((size_t)nb);
        for (int64_t i = lo; i < nb; ++i) row[(size_t)i] = bt[(size_t)i];
        return chg;
    }
    bool same(const KVManager& kv, uint64_t seq, int32_t g) const {
        auto it = rows.find({ seq, g });
        return it != rows.end() && it->second == kv.block_table(seq, g);
    }
    void forget(uint64_t seq) {
        for (auto it = rows.begin(); it != rows.end();)
            it = it->first.first == seq ? rows.erase(it) : std::next(it);
    }
};

/* Speculative rejection. A verify writes KV for 1 + n_spec positions and keeps a prefix, and
 * ensure() only grows, so rollback is the one call that gives the rest back. It frees only paged
 * blocks past the kept length, keeps the state slots (the recurrent kernels roll back by read
 * offset), drops only this sequence's reference on shared blocks and rebases a window that the
 * kept length falls below. Without it every rejected block-crossing leaks a block. */
TEST(rollback_frees_only_the_rejected_tail_and_keeps_the_state) {
    Fixture f;
    f.cfg.n_spec = 3;
    REQUIRE_EQ(f.build(/*with_linear=*/true), RAD_OK);
    const int64_t n0 = f.kv.free_blocks(G_FULL);
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 40));                                  /* three blocks */
    const std::vector<int32_t> was = f.kv.block_table(1, G_FULL);
    const int32_t slot = f.kv.state_slot(1, G_GDN);
    REQUIRE_EQ(was.size(), 3u);

    CHECK_OK(f.kv.rollback(1, 33));                                /* still three blocks */
    CHECK(f.kv.block_table(1, G_FULL) == was);
    CHECK_EQ(f.kv.seq_tokens(1), 33ll);
    CHECK_OK(f.kv.rollback(1, 31));                                /* the third block goes */
    CHECK_EQ(f.kv.block_table(1, G_FULL).size(), 2u);
    CHECK_EQ(f.kv.block_table(1, G_FULL)[1], was[1]);
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0 - 2);
    CHECK_EQ(f.kv.block_refcount(G_FULL, was[2]), 0);
    CHECK_EQ(f.kv.state_slot(1, G_GDN), slot);
    CHECK_EQ(f.kv.seq_tokens(1), 31ll);
    /* Every draft accepted: nothing to give back, and no special path for the caller. */
    CHECK_OK(f.kv.rollback(1, 50));
    CHECK_EQ(f.kv.seq_tokens(1), 31ll);
    CHECK_EQ(f.kv.block_table(1, G_FULL).size(), 2u);
    CHECK_EQ(f.kv.rollback(1, -1), RAD_E_INVAL);
    CHECK_EQ(f.kv.rollback(77, 0), RAD_E_NOTFOUND);
    CHECK_EQ(f.kv.seq_tokens(1), 31ll);

    /* A forked child shares the parent's blocks; its rollback drops only its own references. */
    CHECK_OK(f.kv.add_sequence(2));
    CHECK_OK(f.kv.ensure(2, 40));
    const std::vector<int32_t> par = f.kv.block_table(2, G_FULL);
    const int64_t before_fork = f.kv.free_blocks(G_FULL);
    CHECK_OK(f.kv.fork(2, 3));
    CHECK_OK(f.kv.rollback(3, 20));
    CHECK_EQ(f.kv.block_table(3, G_FULL).size(), 2u);
    CHECK(f.kv.block_table(2, G_FULL) == par);
    CHECK_EQ(f.kv.block_refcount(G_FULL, par[0]), 2);
    CHECK_EQ(f.kv.block_refcount(G_FULL, par[1]), 2);
    CHECK_EQ(f.kv.block_refcount(G_FULL, par[2]), 1);
    CHECK_EQ(f.kv.free_blocks(G_FULL), before_fork);               /* the parent still holds it */
    CHECK(f.kv.state_slot(3, G_GDN) >= 0);
    f.kv.free_sequence(1);
    f.kv.free_sequence(2);
    f.kv.free_sequence(3);
    CHECK_EQ(f.kv.free_blocks(G_FULL), n0);
}

/* A window that slid past the kept length has already recycled those keys, so the table is
 * dropped and rebased at the kept length's block; the next step grows from there. */
TEST(rollback_below_a_slid_window_rebases_it) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    g.push_back(paged_group("swa", RAD_KV_WINDOW, 2, 16, { 0 }, 64));
    Config cfg = small_config();
    cfg.checkpoint_slots = 0;
    Pools p;
    REQUIRE_EQ(configure_pools(p, cfg), RAD_OK);
    KVManager kv;
    REQUIRE_EQ(kv.configure(g, cfg, p.kv()), RAD_OK);
    const int64_t n0 = kv.free_blocks(G_FULL);
    CHECK_OK(kv.add_sequence(1));
    CHECK_OK(kv.ensure(1, 400));
    CHECK_OK(kv.ensure(1, 401));                                   /* a decode step slides it */
    CHECK_EQ(kv.first_block_pos(1, G_FULL), 336ll);
    CHECK_EQ(kv.block_table(1, G_FULL).size(), 5u);
    (void)kv.take_table_changes(1, G_FULL);

    CHECK_OK(kv.rollback(1, 300));
    CHECK_EQ(kv.first_block_pos(1, G_FULL), 288ll);
    CHECK(kv.block_table(1, G_FULL).empty());
    CHECK_EQ(kv.free_blocks(G_FULL), n0);
    CHECK_EQ(kv.take_table_changes(1, G_FULL), 0ll);
    CHECK_EQ(kv.seq_tokens(1), 300ll);
    CHECK_OK(kv.ensure(1, 301));
    CHECK_EQ(kv.block_table(1, G_FULL).size(), 1u);
    CHECK_EQ(kv.first_block_pos(1, G_FULL), 288ll);
    kv.free_sequence(1);
    CHECK_EQ(kv.free_blocks(G_FULL), n0);
}

/* A block the prefix cache indexes keeps the cache's reference when a rollback drops the
 * sequence's: it becomes cache-only, which is what the cache's held-token figure counts. */
TEST(rollback_below_a_published_block_leaves_the_cache_its_reference) {
    Fixture f;
    REQUIRE_EQ(f.build(/*with_linear=*/false), RAD_OK);
    const auto prompt = tok_run(300, 48);
    publish(f, 1, prompt, 48);                                     /* three blocks indexed */
    const std::vector<int32_t> was = f.kv.block_table(1, G_FULL);
    CHECK_EQ(f.pc.unshared_tokens(), 0ll);
    CHECK_OK(f.kv.rollback(1, 20));
    CHECK_EQ(f.kv.block_table(1, G_FULL).size(), 2u);
    CHECK_EQ(f.kv.block_refcount(G_FULL, was[2]), 1);
    CHECK_EQ(f.pc.unshared_tokens(), 16ll);
    f.kv.free_sequence(1);
    CHECK_EQ(f.pc.unshared_tokens(), 48ll);
    CHECK_EQ(f.pc.lookup(tok_run(300, 64), {}).n_attn_tokens, 48);
}

/* The batch builder stages only the block-table entries changed since a sequence's last step. A
 * mutation that rewrites an entry without lowering the mark leaves the card reading a stale block
 * id: another sequence's KV, with fluent output. Each mutation kind is staged into a copy of the
 * card's row and compared with the table. */
TEST(every_table_mutation_marks_the_entries_it_changed) {
    Fixture f;
    REQUIRE_EQ(f.build(/*with_linear=*/false), RAD_OK);
    DeviceRows dev;
    KVManager& kv = f.kv;

    CHECK_EQ(kv.take_table_changes(42, G_FULL), 0ll);             /* unseen: restage it all */
    CHECK_OK(kv.add_sequence(1));
    CHECK_OK(kv.ensure(1, 16));
    CHECK_EQ(dev.stage(kv, 1, G_FULL), 0ll);
    CHECK_EQ(kv.take_table_changes(1, G_FULL), KVManager::kNoDirty);
    CHECK_OK(kv.ensure(1, 17));                                    /* appends block 1 */
    CHECK_EQ(dev.stage(kv, 1, G_FULL), 1ll);
    CHECK(dev.same(kv, 1, G_FULL));
    CHECK_OK(kv.ensure(1, 30));                                    /* same block: nothing */
    CHECK_EQ(dev.stage(kv, 1, G_FULL), KVManager::kNoDirty);
    CHECK(dev.same(kv, 1, G_FULL));

    /* Copy-on-write in the middle of the child's table, at the partial block. */
    CHECK_OK(kv.ensure(1, 40));
    CHECK_EQ(dev.stage(kv, 1, G_FULL), 2ll);
    CHECK_OK(kv.fork(1, 2));
    CHECK_EQ(dev.stage(kv, 2, G_FULL), 0ll);
    CHECK_OK(kv.ensure(2, 41));
    CHECK_EQ(dev.stage(kv, 2, G_FULL), 2ll);
    CHECK(dev.same(kv, 2, G_FULL));
    CHECK(kv.block_table(2, G_FULL)[2] != kv.block_table(1, G_FULL)[2]);
    CHECK_EQ(kv.take_pending_copies().size(), 1u);
    /* Two marks accumulate to the lower one between steps. */
    CHECK_OK(kv.ensure(2, 64));
    CHECK_OK(kv.rollback(2, 20));
    CHECK_EQ(dev.stage(kv, 2, G_FULL), 2ll);
    CHECK(dev.same(kv, 2, G_FULL));
    CHECK_OK(kv.ensure(2, 70));
    CHECK_OK(kv.rollback(2, 1));
    CHECK_EQ(dev.stage(kv, 2, G_FULL), 1ll);
    CHECK(dev.same(kv, 2, G_FULL));

    /* Preempted and grown again: the whole row. */
    CHECK_OK(kv.preempt(1));
    CHECK_OK(kv.ensure(1, 50));
    CHECK_EQ(dev.stage(kv, 1, G_FULL), 0ll);
    CHECK(dev.same(kv, 1, G_FULL));

    /* An adopted prefix: the whole row. */
    const auto prompt = tok_run(900, 64);
    publish(f, 3, prompt, 64);
    std::vector<std::vector<int32_t>> found;
    REQUIRE(f.pc.lookup(tok_run(900, 80), {}, &found).n_attn_tokens == 64);
    CHECK_OK(kv.add_sequence(4));
    CHECK_EQ(dev.stage(kv, 4, G_FULL), 0ll);
    CHECK_OK(kv.adopt(4, G_FULL, found[0], 64));
    CHECK_EQ(dev.stage(kv, 4, G_FULL), 0ll);
    CHECK(dev.same(kv, 4, G_FULL));
}

/* The window's shift moves every entry down, so it marks the whole row; so does seeding a
 * resumed sequence's window. */
TEST(a_window_shift_and_a_seeded_window_mark_the_whole_row) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    g.push_back(paged_group("swa", RAD_KV_WINDOW, 2, 16, { 0 }, 64));
    Config cfg = small_config();
    Pools p;
    REQUIRE_EQ(configure_pools(p, cfg), RAD_OK);
    KVManager kv;
    REQUIRE_EQ(kv.configure(g, cfg, p.kv()), RAD_OK);
    DeviceRows dev;
    CHECK_OK(kv.add_sequence(1));
    CHECK_OK(kv.ensure(1, 400));
    CHECK_EQ(dev.stage(kv, 1, G_FULL), 0ll);
    CHECK_OK(kv.ensure(1, 401));                                   /* the front slides */
    CHECK_EQ(dev.stage(kv, 1, G_FULL), 0ll);
    CHECK(dev.same(kv, 1, G_FULL));
    CHECK_OK(kv.ensure(1, 416));                                   /* no new block, no slide */
    CHECK_EQ(dev.stage(kv, 1, G_FULL), KVManager::kNoDirty);
    CHECK_OK(kv.ensure(1, 417));
    CHECK_EQ(dev.stage(kv, 1, G_FULL), 0ll);
    CHECK(dev.same(kv, 1, G_FULL));
    kv.trim_window(1, 417);
    (void)dev.stage(kv, 1, G_FULL);
    CHECK(dev.same(kv, 1, G_FULL));

    CHECK_OK(kv.add_sequence(2));
    CHECK_EQ(dev.stage(kv, 2, G_FULL), 0ll);
    CHECK_OK(kv.seed_window(2, 400));
    CHECK_EQ(dev.stage(kv, 2, G_FULL), 0ll);
    CHECK(dev.same(kv, 2, G_FULL));
}

/* A recycled state slot holds its last tenant's state, and the kernels take whatever a slot holds
 * as the initial state, so the manager names every slot it hands out for zeroing. Under
 * speculation a linear group's sequence owns two physical slots (copy k of s at s + k *
 * n_seq_states) and both are named; a conv group keeps one copy. The index row the kernels read
 * names copy 0 three times, then copy 1. */
TEST(a_handed_out_state_slot_is_named_for_clearing_once_per_copy) {
    Fixture f;
    f.cfg.n_spec = 2;
    REQUIRE_EQ(f.build_groups(hybrid_conv_groups()), RAD_OK);
    const KVGroupPlan* lin = f.kv.plan(G_GDN);
    const KVGroupPlan* cv = f.kv.plan(G_CV);
    REQUIRE(lin && cv);
    CHECK_EQ(lin->state_copies, 2ll);
    CHECK_EQ(cv->state_copies, 1ll);
    const int32_t span = (int32_t)lin->n_seq_states();
    auto clears = [&]() {
        std::vector<std::pair<int32_t, int32_t>> v;
        for (const StateClear& c : f.kv.state_clears()) v.push_back({ c.group, c.slot });
        return v;
    };
    using Clears = std::vector<std::pair<int32_t, int32_t>>;

    CHECK_OK(f.kv.add_sequence(1));
    CHECK(f.kv.state_clears().empty());                 /* nothing handed out until it grows */
    CHECK_OK(f.kv.ensure(1, 16));
    const int32_t s1 = f.kv.state_slot(1, G_GDN), c1 = f.kv.state_slot(1, G_CV);
    CHECK(clears() == (Clears{ { G_GDN, s1 }, { G_GDN, s1 + span }, { G_CV, c1 } }));
    CHECK_OK(f.kv.ensure(1, 40));                       /* a held slot is not named again */
    CHECK_EQ(f.kv.state_clears().size(), 3u);
    f.kv.clear_state_clears();
    CHECK(f.kv.state_clears().empty());
    CHECK_OK(f.kv.fork(1, 2));
    const int32_t s2 = f.kv.state_slot(2, G_GDN), c2 = f.kv.state_slot(2, G_CV);
    CHECK(s2 != s1 && c2 != c1);
    CHECK(clears() == (Clears{ { G_GDN, s2 }, { G_GDN, s2 + span }, { G_CV, c2 } }));

    CHECK_EQ(f.kv.state_index_width(G_GDN), 6ll);
    CHECK_EQ(f.kv.state_index_width(G_CV), 1ll);
    CHECK_EQ(f.kv.state_index_width(G_FULL), 0ll);
    int32_t row[8];
    std::fill(row, row + 8, -9);
    CHECK_EQ(f.kv.state_index_row(1, G_GDN, row, 8), 6);
    for (int j = 0; j < 6; ++j) CHECK_EQ(row[j], j < 3 ? s1 : s1 + span);
    CHECK_EQ(row[6], -9);
    CHECK_EQ(f.kv.state_index_row(1, G_CV, row, 8), 1);
    CHECK_EQ(row[0], c1);
    CHECK_EQ(f.kv.state_index_row(1, G_GDN, row, 4), 4);           /* never past `max` */
    CHECK_EQ(f.kv.state_index_row(1, G_FULL, row, 8), 0);
    CHECK_EQ(f.kv.state_index_row(99, G_GDN, row, 8), 6);          /* no slot: -1 across */
    for (int j = 0; j < 6; ++j) CHECK_EQ(row[j], -1);

    /* Zeroing the named slots, as the engine does (layer_bytes_per_state at state_ptr for every
     * layer), touches no other slot of either group. */
    if (!device_is_host()) {
        std::printf("  SKIP  the state bytes are read on the host backend only (%s here)\n",
                    device_backend_name());
        return;
    }
    f.kv.clear_state_clears();
    CHECK_OK(f.kv.preempt(1));
    CHECK_OK(f.kv.ensure(1, 16));
    auto region = [&](int32_t g, int64_t l, int64_t s) {
        const KVGroupPlan* p = f.kv.plan(g);
        const int64_t lb = p->bytes_per_state / p->n_layers;
        return (unsigned char*)p->base + l * p->n_states * lb + s * lb;
    };
    for (int32_t g : { (int32_t)G_GDN, (int32_t)G_CV }) {
        const KVGroupPlan* p = f.kv.plan(g);
        for (int64_t l = 0; l < p->n_layers; ++l)
            for (int64_t s = 0; s < p->n_states; ++s)
                std::memset(region(g, l, s), (int)(0x40 + s), (size_t)(p->bytes_per_state / p->n_layers));
    }
    RadStream st = nullptr;
    REQUIRE_EQ(rad_stream_create(&st, 0), RAD_OK);
    std::set<std::pair<int32_t, int64_t>> zeroed;
    for (const StateClear& c : f.kv.state_clears()) {
        const KVGroupPlan* p = f.kv.plan(c.group);
        for (int64_t l = 0; l < p->n_layers; ++l)
            CHECK_OK(rad_memset_async(f.kv.state_ptr(c.group, (int32_t)l, c.slot), 0,
                                      p->layer_bytes_per_state(), st));
        zeroed.insert({ c.group, c.slot });
    }
    CHECK_OK(rad_stream_sync(st));
    rad_stream_destroy(st);
    CHECK_EQ(zeroed.size(), 3u);
    int64_t wrong = 0;
    for (int32_t g : { (int32_t)G_GDN, (int32_t)G_CV }) {
        const KVGroupPlan* p = f.kv.plan(g);
        const int64_t lb = p->bytes_per_state / p->n_layers;
        for (int64_t l = 0; l < p->n_layers; ++l)
            for (int64_t s = 0; s < p->n_states; ++s) {
                const unsigned char want = zeroed.count({ g, s }) ? 0 : (unsigned char)(0x40 + s);
                const unsigned char* b = region(g, l, s);
                for (int64_t i = 0; i < lb; ++i) wrong += b[i] != want;
            }
    }
    CHECK_EQ(wrong, 0ll);
}

/* ------------------------------------------------------------------ the model-based run
 *
 * The KV invariants hold over any interleaving of the manager's mutations: every block's refcount
 * is the number of tables and reservations naming it, a lent block is never held, every state slot
 * has one owner, each table covers its sequence's tokens, a refused ensure() changes nothing, and
 * the change marks name every rewritten entry. Each is a silent-corruption failure (another
 * sequence's KV, or expert weights written over held blocks), so the run recounts all of them
 * after every operation. */
namespace {
struct KVRun {
    uint64_t seed;
    Xorshift r;
    Pools pools;
    KVManager kv;
    Config cfg = small_config();
    std::vector<uint64_t> live;
    std::vector<std::pair<int32_t, int32_t>> held;     /* the test's own reservations */
    DeviceRows dev;
    uint64_t next_id = 1;
    int op = 0;
    std::string what;
    bool failed = false;
    std::map<std::string, int64_t> seen;               /* how often each outcome happened */

    explicit KVRun(uint64_t s) : seed(s), r(s) {}

    bool fail(const std::string& why) {
        if (!failed)
            fprintf(stderr, "    seed %#llx op %d (%s): %s\n", (unsigned long long)seed, op,
                    what.c_str(), why.c_str());
        failed = true;
        return false;
    }
    int build() {
        cfg.max_tok = 256;
        cfg.n_spec = 2;
        cfg.checkpoint_slots = 0;
        int s = configure_pools(pools, cfg);
        if (s < 0) return s;
        s = kv.configure(every_kind_groups(), cfg, pools.kv());
        if (s < 0) return s;
        kv.set_reclaim([](int64_t) { return (int)RAD_OK; });
        return RAD_OK;
    }
    uint64_t pick() { return live[(size_t)r.below(live.size())]; }
    void drop_live(uint64_t id) {
        live.erase(std::find(live.begin(), live.end(), id));
        dev.forget(id);
    }

    /* Everything the manager says about one sequence that a refused ensure() must leave alone. */
    struct Snap {
        int64_t n = 0;
        std::vector<int32_t> full, win;
        int32_t lin = -1, conv = -1;
        int64_t first = 0;
    };
    Snap snap(uint64_t id) const {
        Snap s;
        s.n = kv.seq_tokens(id);
        s.full = kv.block_table(id, HG_FULL);
        s.win = kv.block_table(id, HG_WIN);
        s.lin = kv.state_slot(id, HG_LIN);
        s.conv = kv.state_slot(id, HG_CONV);
        s.first = kv.first_block_pos(id, HG_WIN);
        return s;
    }

    bool check_copies(uint64_t id) {
        for (const BlockCopy& c : kv.take_pending_copies()) {
            ++seen["copy-on-write"];
            if (c.group != HG_FULL && c.group != HG_WIN) return fail("a copy names a non-paged group");
            if (c.src_block == c.dst_block) return fail("a copy onto itself");
            if (kv.block_refcount(c.group, c.src_block) < 1 || kv.block_refcount(c.group, c.dst_block) < 1)
                return fail(fmt("copy %d -> %d names a free block", c.src_block, c.dst_block));
            const std::vector<int32_t>& bt = kv.block_table(id, c.group);
            if (std::find(bt.begin(), bt.end(), c.dst_block) == bt.end())
                return fail("a copy's destination is not in the writer's table");
        }
        return true;
    }

    bool check() {
        for (int32_t g : { (int32_t)HG_FULL, (int32_t)HG_WIN }) {
            const KVGroupPlan* p = kv.plan(g);
            std::vector<int32_t> count((size_t)p->n_blocks, 0);
            for (uint64_t id : live)
                for (int32_t b : kv.block_table(id, g)) {
                    if (b < 0 || b >= p->n_blocks) return fail(fmt("group %d names block %d", g, b));
                    ++count[(size_t)b];
                }
            for (const auto& h : held) if (h.first == g) ++count[(size_t)h.second];
            int64_t n_held = 0, below = 0;
            for (int64_t b = 0; b < p->n_blocks; ++b) {
                if (kv.block_refcount(g, (int32_t)b) != count[(size_t)b])
                    return fail(fmt("group %d block %lld: refcount %d, named %d times", g,
                                    (long long)b, kv.block_refcount(g, (int32_t)b), count[(size_t)b]));
                if (count[(size_t)b] == 0) continue;
                ++n_held;
                if (b >= kv.live_blocks(g))
                    return fail(fmt("group %d block %lld is held above the live mark %lld", g,
                                    (long long)b, (long long)kv.live_blocks(g)));
                ++below;
            }
            if (kv.held_blocks(g) != n_held)
                return fail(fmt("group %d: held_blocks %lld, recount %lld", g,
                                (long long)kv.held_blocks(g), (long long)n_held));
            if (kv.free_blocks(g) != kv.live_blocks(g) - below)
                return fail(fmt("group %d: free_blocks %lld, live %lld less held %lld", g,
                                (long long)kv.free_blocks(g), (long long)kv.live_blocks(g),
                                (long long)below));
            if (kv.pool_bytes_used(g) != n_held * p->bytes_per_block)
                return fail(fmt("group %d: pool_bytes_used is not the held blocks", g));
        }
        if (kv.cache_only_blocks(HG_FULL) != 0) return fail("cache-only blocks with no cache");
        for (int32_t g : { (int32_t)HG_LIN, (int32_t)HG_CONV }) {
            const KVGroupPlan* p = kv.plan(g);
            std::set<int32_t> slots;
            for (uint64_t id : live) {
                const int32_t s = kv.state_slot(id, g);
                if (s < 0) continue;
                if (s >= p->n_seq_states()) return fail(fmt("group %d: slot %d past the logical slots", g, s));
                if (!slots.insert(s).second) return fail(fmt("group %d: slot %d has two owners", g, s));
            }
            if (kv.free_states(g) + (int64_t)slots.size() != p->n_seq_states())
                return fail(fmt("group %d: %lld free and %zu owned of %lld slots", g,
                                (long long)kv.free_states(g), slots.size(),
                                (long long)p->n_seq_states()));
        }
        const KVGroupPlan* pf = kv.plan(HG_FULL);
        const KVGroupPlan* pw = kv.plan(HG_WIN);
        for (uint64_t id : live) {
            const int64_t n = kv.seq_tokens(id);
            if ((int64_t)kv.block_table(id, HG_FULL).size() != cdiv(n, pf->block_size))
                return fail(fmt("seq %llu: %zu full blocks for %lld tokens", (unsigned long long)id,
                                kv.block_table(id, HG_FULL).size(), (long long)n));
            const int64_t first = kv.first_block_pos(id, HG_WIN);
            const int64_t nw = (int64_t)kv.block_table(id, HG_WIN).size();
            if (first % pw->block_size != 0) return fail("a window base off the block grid");
            if (nw > 0 && first / pw->block_size + nw != cdiv(n, pw->block_size))
                return fail(fmt("seq %llu: window [%lld, +%lld blocks) does not end at %lld tokens",
                                (unsigned long long)id, (long long)first, (long long)nw, (long long)n));
            if (nw > kv.window_seq_cap(*pw))
                return fail(fmt("seq %llu: %lld window blocks past the cap %lld",
                                (unsigned long long)id, (long long)nw,
                                (long long)kv.window_seq_cap(*pw)));
        }
        /* The batch builder stages a random subset of the sequences each step. */
        for (uint64_t id : live)
            for (int32_t g : { (int32_t)HG_FULL, (int32_t)HG_WIN }) {
                if (r.below(2)) continue;
                dev.stage(kv, id, g);
                if (!dev.same(kv, id, g))
                    return fail(fmt("seq %llu group %d: the staged row differs from the table",
                                    (unsigned long long)id, g));
            }
        return true;
    }

    void step() {
        uint64_t k = r.below(100);
        /* Past a dozen sequences an add, fork or adopt frees one instead, so the population stays
         * above the 8 state slots (refusals) without growing without bound. */
        if (live.size() >= 12 && (k < 8 || (k >= 50 && k < 62))) k = 62 + r.below(8);
        if (k < 8 || live.empty()) {
            const uint64_t id = next_id++;
            what = fmt("add %llu", (unsigned long long)id);
            if (kv.add_sequence(id) != RAD_OK) { fail("add refused"); return; }
            if (kv.add_sequence(id) != RAD_E_DUPLICATE) { fail("a second add was not refused"); return; }
            live.push_back(id);
        } else if (k < 38) {
            const uint64_t id = pick();
            const int64_t cur = kv.seq_tokens(id);
            int64_t keep = -1;
            const uint64_t m = r.below(3);
            if (m == 1) keep = cur;
            if (m == 2) keep = cur - (int64_t)r.below((uint64_t)std::min<int64_t>(cur, 32) + 1);
            const int64_t n = cur + 1 + (int64_t)(r.below(8) ? r.below(64) : r.below(200));
            what = fmt("ensure %llu %lld -> %lld from %lld", (unsigned long long)id, (long long)cur,
                       (long long)n, (long long)keep);
            const Snap before = snap(id);
            const int rc = kv.ensure(id, n, keep);
            if (rc == RAD_E_FULL) {
                ++seen["ensure refused"];
                const Snap after = snap(id);
                if (after.n != before.n || after.full != before.full || after.lin != before.lin ||
                    after.conv != before.conv) { fail("a refused ensure changed the sequence"); return; }
                /* Pass 0 may recycle the window's front; nothing else moves. */
                if (after.win.size() > before.win.size() ||
                    !std::equal(after.win.begin(), after.win.end(),
                                before.win.end() - (ptrdiff_t)after.win.size())) {
                    fail("a refused ensure rewrote the window"); return;
                }
                if (!kv.take_pending_copies().empty()) { fail("a refused ensure queued a copy"); return; }
            } else if (rc != RAD_OK) {
                fail(fmt("ensure returned %s", rad_strerror(rc))); return;
            } else {
                ++seen["ensure"];
                if (kv.seq_tokens(id) != n) { fail("ensure did not reach its length"); return; }
                if (kv.first_block_pos(id, HG_WIN) > before.first) ++seen["window slid"];
                if (kv.state_slot(id, HG_LIN) < 0 || kv.state_slot(id, HG_CONV) < 0) {
                    fail("a grown sequence holds no state slot"); return;
                }
                check_copies(id);
            }
        } else if (k < 50) {
            const uint64_t id = pick();
            const int64_t cur = kv.seq_tokens(id);
            const int64_t n = r.below(6) ? (int64_t)r.below((uint64_t)cur + 1) : cur + (int64_t)r.below(9);
            what = fmt("rollback %llu %lld -> %lld", (unsigned long long)id, (long long)cur, (long long)n);
            const Snap before = snap(id);
            if (kv.rollback(id, n) != RAD_OK) { fail("rollback refused"); return; }
            const Snap after = snap(id);
            if (after.n != std::min(n, cur)) { fail("rollback left the wrong length"); return; }
            if (after.lin != before.lin || after.conv != before.conv) {
                fail("rollback moved a state slot"); return;
            }
            if (!std::equal(after.full.begin(), after.full.end(), before.full.begin())) {
                fail("rollback rewrote a kept full block"); return;
            }
            ++seen[n < cur ? "rollback" : "rollback (none)"];
        } else if (k < 56) {
            const uint64_t parent = pick(), child = next_id++;
            what = fmt("fork %llu -> %llu", (unsigned long long)parent, (unsigned long long)child);
            const int rc = kv.fork(parent, child);
            if (rc == RAD_E_FULL) {
                if (kv.has_sequence(child)) { fail("a refused fork left the child"); return; }
                ++seen["fork refused"];
                return;
            }
            if (rc != RAD_OK) { fail(fmt("fork returned %s", rad_strerror(rc))); return; }
            live.push_back(child);
            if (kv.seq_tokens(child) != kv.seq_tokens(parent) ||
                kv.block_table(child, HG_FULL) != kv.block_table(parent, HG_FULL) ||
                kv.block_table(child, HG_WIN) != kv.block_table(parent, HG_WIN) ||
                kv.first_block_pos(child, HG_WIN) != kv.first_block_pos(parent, HG_WIN)) {
                fail("the child does not share the parent's tables"); return;
            }
            ++seen["fork"];
        } else if (k < 62) {
            const uint64_t donor = pick();
            const int64_t whole = kv.seq_tokens(donor) / 16;
            if (whole <= 0) return;
            const int64_t nb = 1 + (int64_t)r.below((uint64_t)whole);
            const std::vector<int32_t>& t = kv.block_table(donor, HG_FULL);
            const std::vector<int32_t> blocks(t.begin(), t.begin() + nb);
            const uint64_t id = next_id++;
            what = fmt("adopt %lld blocks of %llu into %llu", (long long)nb,
                       (unsigned long long)donor, (unsigned long long)id);
            if (kv.add_sequence(id) != RAD_OK) { fail("add refused"); return; }
            live.push_back(id);
            if (kv.adopt(id, HG_FULL, blocks, nb * 16 - 1) != RAD_E_INVAL) {
                fail("a partial block was adopted"); return;
            }
            if (kv.adopt(id, HG_FULL, blocks, nb * 16) != RAD_OK) { fail("adopt refused"); return; }
            if (kv.adopt(id, HG_FULL, blocks, nb * 16) != RAD_E_STATE) {
                fail("a second adopt into a filled table was not refused"); return;
            }
            if (kv.seq_tokens(id) != nb * 16) { fail("adopt left the wrong length"); return; }
            ++seen["adopt"];
        } else if (k < 70) {
            const uint64_t id = pick();
            what = fmt("free %llu", (unsigned long long)id);
            const uint64_t frees = kv.seq_frees();
            kv.free_sequence(id);
            drop_live(id);
            if (kv.has_sequence(id) || kv.seq_frees() != frees + 1) { fail("free left the sequence"); return; }
            ++seen["free"];
        } else if (k < 74) {
            const uint64_t id = pick();
            what = fmt("preempt %llu", (unsigned long long)id);
            if (kv.preempt(id) != RAD_OK) { fail("preempt refused"); return; }
            const Snap s = snap(id);
            if (s.n != 0 || !s.full.empty() || !s.win.empty() || s.lin >= 0 || s.conv >= 0) {
                fail("a preempted sequence kept something"); return;
            }
            ++seen["preempt"];
        } else if (k < 80) {
            const uint64_t id = pick();
            const int64_t cur = kv.seq_tokens(id);
            const int64_t n = cur + 1 + (int64_t)r.below(256);
            std::vector<uint64_t> victims;
            for (uint64_t v : live) if (r.below(3)) victims.push_back(v);
            for (size_t i = victims.size(); i > 1; --i) std::swap(victims[i - 1], victims[(size_t)r.below(i)]);
            what = fmt("make_room %llu -> %lld over %zu victims", (unsigned long long)id,
                       (long long)n, victims.size());
            std::vector<uint64_t> out;
            const int rc = kv.make_room(id, n, victims, &out);
            for (uint64_t v : out) {
                if (v == id) { fail("make_room preempted the sequence it made room for"); return; }
                if (kv.seq_tokens(v) != 0) { fail("a reported victim kept its tokens"); return; }
            }
            seen["preempted for room"] += (int64_t)out.size();
            if (rc == RAD_OK) {
                const int e = kv.ensure(id, n);
                if (e != RAD_OK) { fail(fmt("make_room said yes and ensure said %s", rad_strerror(e))); return; }
                check_copies(id);
                ++seen["make_room"];
            } else if (rc != RAD_E_FULL) {
                fail(fmt("make_room returned %s", rad_strerror(rc))); return;
            }
        } else if (k < 84) {
            const uint64_t id = pick();
            const int64_t at = (int64_t)r.below((uint64_t)kv.seq_tokens(id) + 1);
            what = fmt("trim_window %llu at %lld", (unsigned long long)id, (long long)at);
            kv.trim_window(id, at);
            ++seen["trim_window"];
        } else if (k < 88) {
            const int32_t g = r.below(2) ? HG_FULL : HG_WIN;
            const uint64_t how = r.below(3);
            if (how == 0) {
                int32_t b = -1;
                what = fmt("reserve_block in %d", g);
                const int rc = kv.reserve_block(g, &b);
                if (rc == RAD_OK) { held.push_back({ g, b }); ++seen["reserve"]; }
                else if (rc != RAD_E_FULL || b != -1) { fail("a refused reservation named a block"); return; }
            } else if (how == 1) {
                std::vector<int32_t> got;
                const int64_t want = 1 + (int64_t)r.below(4);
                what = fmt("reserve_blocks %lld in %d", (long long)want, g);
                const int64_t n = kv.reserve_blocks(g, want, &got);
                if (n != (int64_t)got.size() || n > want) { fail("reserve_blocks miscounted"); return; }
                for (int32_t b : got) held.push_back({ g, b });
                seen["reserve"] += n;
            } else {
                const int32_t b = (int32_t)r.below((uint64_t)kv.total_blocks(g));
                what = fmt("reserve_specific_block %d in %d", b, g);
                const bool can = kv.block_refcount(g, b) == 0 && b < kv.live_blocks(g);
                const int rc = kv.reserve_specific_block(g, b);
                if ((rc == RAD_OK) != can || (rc != RAD_OK && rc != RAD_E_STATE)) {
                    fail(fmt("reserve_specific_block returned %s", rad_strerror(rc))); return;
                }
                if (rc == RAD_OK) { held.push_back({ g, b }); ++seen["reserve specific"]; }
            }
        } else if (k < 93) {
            if (held.empty()) return;
            const size_t i = (size_t)r.below(held.size());
            what = fmt("release_block %d in %d", held[i].second, held[i].first);
            kv.release_block(held[i].first, held[i].second);
            held.erase(held.begin() + (ptrdiff_t)i);
            ++seen["release"];
        } else {
            const int64_t spare = kv.spare_tokens();
            const uint64_t how = r.below(4);
            const int64_t t = how == 0 ? 0 : how == 1 ? (int64_t)r.below(2000) : spare;
            what = fmt("set_loan %lld (spare %lld)", (long long)t, (long long)spare);
            const int64_t loan0 = kv.loan_tokens(), live0 = kv.live_blocks(HG_FULL);
            const int rc = kv.set_loan(t);
            if (rc == RAD_E_STATE) {
                if (kv.loan_tokens() != loan0 || kv.live_blocks(HG_FULL) != live0) {
                    fail("a refused loan moved the mark"); return;
                }
                ++seen["loan refused"];
            } else if (rc != RAD_OK) {
                fail(fmt("set_loan returned %s", rad_strerror(rc))); return;
            } else {
                ++seen[t > 0 ? "loan" : "loan back"];
            }
            if (t == spare && rc != RAD_OK) { fail("the spare figure could not be lent"); return; }
        }
    }
};
}  /* namespace */

TEST(kv_invariants_hold_over_random_interleavings_of_every_mutation) {
    KVRun run(0x6a09e667f3bcc909ull);
    REQUIRE_EQ(run.build(), RAD_OK);
    REQUIRE_EQ(run.kv.total_blocks(HG_FULL), 116ll);
    REQUIRE_EQ(run.kv.total_blocks(HG_WIN), 64ll);
    for (run.op = 0; run.op < 5000 && !run.failed; ++run.op) {
        run.step();
        if (!run.failed) run.check();
    }
    CHECK(!run.failed);
    /* Every kind of change must have happened, refusals included, for the run to mean anything. */
    for (const char* k : { "ensure", "ensure refused", "rollback", "fork", "fork refused", "adopt",
                           "free", "preempt", "make_room", "preempted for room", "trim_window",
                           "window slid", "copy-on-write", "reserve", "reserve specific", "release",
                           "loan", "loan back", "loan refused" }) {
        if (run.seen[k] <= 0) fprintf(stderr, "    no '%s' in the run\n", k);
        CHECK(run.seen[k] > 0);
    }
    /* Everything given back: the whole carve is free and every slot is back. */
    for (uint64_t id : run.live) run.kv.free_sequence(id);
    run.live.clear();
    for (const auto& h : run.held) run.kv.release_block(h.first, h.second);
    run.held.clear();
    CHECK_OK(run.kv.set_loan(0));
    for (int32_t g : { (int32_t)HG_FULL, (int32_t)HG_WIN })
        CHECK_EQ(run.kv.free_blocks(g), run.kv.total_blocks(g));
    for (int32_t g : { (int32_t)HG_LIN, (int32_t)HG_CONV })
        CHECK_EQ(run.kv.free_states(g), run.kv.plan(g)->n_seq_states());
    CHECK(run.check());
}

/* ================================================================== checkpoint bytes
 *
 * A snapshot blob has no header: for each stateful group in plans() order, for each layer, that
 * layer's slot bytes of copy 0, then window_blocks pages per snapshot group. A wrong pitch, layer
 * stride, group order or rank offset restores another sequence's or another rank's state, which
 * reads as fluent text. These copy real bytes through the host backend's queued copies and compare
 * every byte against a layout computed here from the plan.
 */
static unsigned char* state_region(const KVManager& kv, int32_t g, int64_t layer, int64_t slot) {
    const KVGroupPlan* p = kv.plan(g);
    const int64_t lb = p->bytes_per_state / p->n_layers;
    return (unsigned char*)p->base + layer * p->n_states * lb + slot * lb;
}

/* Fills every physical state slot of every stateful group, every layer, with its own pattern. */
static void fill_states(const KVManager& kv, int rank) {
    for (const KVGroupPlan& p : kv.plans()) {
        if (!p.stateful()) continue;
        const int64_t lb = p.bytes_per_state / p.n_layers;
        for (int64_t l = 0; l < p.n_layers; ++l)
            for (int64_t s = 0; s < p.n_states; ++s) {
                unsigned char* b = state_region(kv, p.index, l, s);
                for (int64_t i = 0; i < lb; ++i) b[i] = pat(rank, p.index * 16 + l, (int32_t)s, i);
            }
    }
}

/* Bytes of the state pools that differ from fill_states(rank), except slot `dst` of each group
 * (copy 0 only), which must hold `src`'s pattern instead. */
static int64_t states_wrong(const KVManager& kv, int rank, const std::vector<int32_t>& dst,
                            const std::vector<int32_t>& src) {
    int64_t wrong = 0;
    for (const KVGroupPlan& p : kv.plans()) {
        if (!p.stateful()) continue;
        const int64_t lb = p.bytes_per_state / p.n_layers;
        for (int64_t l = 0; l < p.n_layers; ++l)
            for (int64_t s = 0; s < p.n_states; ++s) {
                const int32_t from = s == dst[(size_t)p.index] ? src[(size_t)p.index] : (int32_t)s;
                const unsigned char* b = state_region(kv, p.index, l, s);
                for (int64_t i = 0; i < lb; ++i) wrong += b[i] != pat(rank, p.index * 16 + l, from, i);
            }
    }
    return wrong;
}

/* The slots `seq` holds, indexed by group handle as checkpoint_save takes them. */
static std::vector<int32_t> slots_of(const KVManager& kv, uint64_t seq) {
    std::vector<int32_t> v((size_t)kv.n_groups(), -1);
    for (int32_t g = 0; g < kv.n_groups(); ++g) v[(size_t)g] = kv.state_slot(seq, g);
    return v;
}

static bool host_backend_or_skip(const char* what) {
    if (device_is_host()) return true;
    std::printf("  SKIP  %s reads pool bytes on the host backend only (%s here)\n", what,
                device_backend_name());
    return false;
}

/* The linear state and conv window of one sequence go into a slot in the documented layout and
 * come back into another sequence's slots, copy 0 only: copy 1 holds replay factors for tokens
 * not yet accepted, and every other slot is untouched. */
TEST(a_checkpoint_carries_copy_zero_of_every_state_layer_in_plan_order) {
    if (!host_backend_or_skip("the checkpoint round trip")) return;
    Fixture f;
    f.cfg.n_spec = 2;
    f.cfg.checkpoint_slots = 4;
    REQUIRE_EQ(f.build_groups(hybrid_conv_groups()), RAD_OK);
    const KVGroupPlan* lin = f.kv.plan(G_GDN);
    const KVGroupPlan* cv = f.kv.plan(G_CV);
    REQUIRE_EQ(f.kv.checkpoint_bytes(), lin->bytes_per_state + cv->bytes_per_state);
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.add_sequence(2));
    CHECK_OK(f.kv.ensure(1, 16));
    CHECK_OK(f.kv.ensure(2, 16));
    const std::vector<int32_t> a = slots_of(f.kv, 1), b = slots_of(f.kv, 2);
    fill_states(f.kv, 0);

    int32_t ck = -1;
    REQUIRE_EQ(f.kv.alloc_checkpoint(&ck), RAD_OK);
    unsigned char* blob = (unsigned char*)f.kv.checkpoint_ptr(ck);
    std::memset(blob, 0xA5, (size_t)f.kv.checkpoint_bytes());
    RadStream st = nullptr;
    REQUIRE_EQ(rad_stream_create(&st, 0), RAD_OK);
    CHECK_OK(f.kv.checkpoint_save(a.data(), (int)a.size(), ck, st));
    CHECK_OK(rad_stream_sync(st));

    /* The blob, walked here: linear layers, then conv layers, each one layer's bytes of A's slot. */
    int64_t off = 0, wrong = 0;
    for (int32_t g : { (int32_t)G_GDN, (int32_t)G_CV }) {
        const KVGroupPlan* p = f.kv.plan(g);
        const int64_t lb = p->bytes_per_state / p->n_layers;
        for (int64_t l = 0; l < p->n_layers; ++l, off += lb)
            for (int64_t i = 0; i < lb; ++i)
                wrong += blob[off + i] != pat(0, g * 16 + l, a[(size_t)g], i);
    }
    CHECK_EQ(off, f.kv.checkpoint_bytes());
    CHECK_EQ(wrong, 0ll);
    CHECK_EQ(states_wrong(f.kv, 0, b, b), 0ll);                    /* a save writes no slot */

    CHECK_OK(f.kv.checkpoint_restore(b.data(), (int)b.size(), ck, st));
    CHECK_OK(rad_stream_sync(st));
    CHECK_EQ(states_wrong(f.kv, 0, b, a), 0ll);

    /* Refusals move no bytes: a slot the store does not hold, too few groups named, a sequence
     * with no state slot, and a manager with no store. */
    fill_states(f.kv, 0);
    std::vector<int32_t> none((size_t)f.kv.n_groups(), -1);
    int32_t gone = -1;
    REQUIRE_EQ(f.kv.alloc_checkpoint(&gone), RAD_OK);
    f.kv.free_checkpoint(gone);
    CHECK_EQ(f.kv.checkpoint_restore(b.data(), (int)b.size(), gone, st), RAD_E_INVAL);
    CHECK_EQ(f.kv.checkpoint_restore(b.data(), 2, ck, st), RAD_E_INVAL);
    CHECK_EQ(f.kv.checkpoint_restore(none.data(), (int)none.size(), ck, st), RAD_E_STATE);
    KVManager bare;
    Pools p2;
    REQUIRE_EQ(configure_pools(p2, f.cfg), RAD_OK);
    REQUIRE_EQ(bare.configure(hybrid_conv_groups(), f.cfg, p2.kv()), RAD_OK);
    CHECK_EQ(bare.checkpoint_save(b.data(), (int)b.size(), 0, st), RAD_E_INVAL);
    CHECK_OK(rad_stream_sync(st));
    CHECK_EQ(states_wrong(f.kv, 0, b, b), 0ll);
    rad_stream_destroy(st);
}

/* Under tensor parallelism one slot holds every rank's shard at its own offset, and rank 0 hands
 * out the id. Each rank restores its own shard: at a shared offset the last writer's state would
 * be restored on both. */
TEST(each_rank_restores_its_own_shard_of_a_shared_checkpoint) {
    if (!host_backend_or_skip("the checkpoint round trip")) return;
    Fixture lead, fol;
    lead.cfg.checkpoint_slots = fol.cfg.checkpoint_slots = 0;
    REQUIRE_EQ(lead.build_groups(hybrid_conv_groups()), RAD_OK);
    REQUIRE_EQ(fol.build_groups(hybrid_conv_groups()), RAD_OK);
    const int64_t shard = lead.kv.checkpoint_bytes();
    HostArena store;
    REQUIRE_EQ(store.open(2 * shard, 2 * shard * 2), RAD_OK);
    lead.kv.set_checkpoint_store(&store, 0);
    fol.kv.set_checkpoint_store(&store, shard);
    for (Fixture* x : { &lead, &fol }) {
        CHECK_OK(x->kv.add_sequence(1));
        CHECK_OK(x->kv.add_sequence(2));
        CHECK_OK(x->kv.ensure(1, 16));
        CHECK_OK(x->kv.ensure(2, 16));
    }
    fill_states(lead.kv, 0);
    fill_states(fol.kv, 1);
    int32_t ck = -1;
    REQUIRE_EQ(lead.kv.alloc_checkpoint(&ck), RAD_OK);
    RadStream st = nullptr;
    REQUIRE_EQ(rad_stream_create(&st, 0), RAD_OK);
    const std::vector<int32_t> a = slots_of(lead.kv, 1), b = slots_of(lead.kv, 2);
    REQUIRE(slots_of(fol.kv, 1) == a);
    CHECK_OK(lead.kv.checkpoint_save(a.data(), (int)a.size(), ck, st));
    CHECK_OK(fol.kv.checkpoint_save(a.data(), (int)a.size(), ck, st));
    CHECK_OK(lead.kv.checkpoint_restore(b.data(), (int)b.size(), ck, st));
    CHECK_OK(fol.kv.checkpoint_restore(b.data(), (int)b.size(), ck, st));
    CHECK_OK(rad_stream_sync(st));
    CHECK_EQ(states_wrong(lead.kv, 0, b, a), 0ll);
    CHECK_EQ(states_wrong(fol.kv, 1, b, a), 0ll);
    rad_stream_destroy(st);
}

/* A model with a sliding window and no linear state snapshots the window behind a checkpoint:
 * window_blocks pages per group, page j at position j, each page every layer's row of one block.
 * A -1 row is a page below position zero: nothing is copied for it either way. A resumed sequence
 * is seeded with fresh blocks at the same floor and gets the pages back index for index. */
TEST(a_checkpoint_carries_the_window_page_by_page_and_skips_the_padding) {
    if (!host_backend_or_skip("the window snapshot")) return;
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    g.push_back(paged_group("swa", RAD_KV_WINDOW, 2, 16, { 0, 1 }, 64));
    Config cfg = small_config();
    Pools pools;
    REQUIRE_EQ(configure_pools(pools, cfg), RAD_OK);
    KVManager kv;
    REQUIRE_EQ(kv.configure(g, cfg, pools.kv()), RAD_OK);
    const KVGroupPlan* p = kv.plan(G_FULL);
    REQUIRE(p && p->win_snap);
    const int64_t page = p->bytes_per_block, lb = p->layer_bytes_per_block();
    REQUIRE_EQ(kv.checkpoint_bytes(), p->window_blocks * page);
    HostArena store;
    REQUIRE_EQ(store.open(kv.checkpoint_bytes(), kv.checkpoint_bytes() * 2), RAD_OK);
    kv.set_checkpoint_store(&store, 0);
    auto block = [&](int64_t l, int32_t b) {
        return (unsigned char*)p->base + l * p->layer_stride() + b * lb;
    };
    for (int64_t l = 0; l < p->n_layers; ++l)
        for (int32_t b = 0; b < (int32_t)p->n_blocks; ++b)
            for (int64_t i = 0; i < lb; ++i) block(l, b)[i] = pat(5, l, b, i);

    CHECK_OK(kv.add_sequence(1));
    CHECK_OK(kv.ensure(1, 400));
    std::vector<int32_t> rows;
    REQUIRE(kv.window_rows(1, 400, &rows));
    REQUIRE_EQ(rows.size(), 5u);
    REQUIRE_EQ(rows[4], -1);
    int32_t ck = -1;
    REQUIRE_EQ(kv.alloc_checkpoint(&ck), RAD_OK);
    unsigned char* blob = (unsigned char*)kv.checkpoint_ptr(ck);
    std::memset(blob, 0x3C, (size_t)kv.checkpoint_bytes());
    const std::vector<int32_t> none(2, -1);
    RadStream st = nullptr;
    REQUIRE_EQ(rad_stream_create(&st, 0), RAD_OK);
    CHECK_OK(kv.checkpoint_save(none.data(), 2, ck, st, &rows));
    CHECK_OK(rad_stream_sync(st));
    int64_t wrong = 0;
    for (int64_t j = 0; j < 5; ++j)
        for (int64_t l = 0; l < p->n_layers; ++l)
            for (int64_t i = 0; i < lb; ++i) {
                const unsigned char want = j < 4 ? pat(5, l, rows[(size_t)j], i) : 0x3C;
                wrong += blob[j * page + l * lb + i] != want;
            }
    CHECK_EQ(wrong, 0ll);

    /* Back into a fresh sequence's seeded blocks; every other block keeps its bytes. */
    for (int64_t l = 0; l < p->n_layers; ++l)
        std::memset(block(l, 0), 0xEE, (size_t)p->layer_stride());
    CHECK_OK(kv.add_sequence(2));
    CHECK_OK(kv.seed_window(2, 400));
    std::vector<int32_t> rows2;
    REQUIRE(kv.window_rows(2, 400, &rows2));
    REQUIRE_EQ(rows2.size(), 5u);
    CHECK_OK(kv.checkpoint_restore(none.data(), 2, ck, st, &rows2));
    CHECK_OK(rad_stream_sync(st));
    wrong = 0;
    for (int64_t l = 0; l < p->n_layers; ++l)
        for (int32_t b = 0; b < (int32_t)p->n_blocks; ++b) {
            const auto at = std::find(rows2.begin(), rows2.begin() + 4, b);
            for (int64_t i = 0; i < lb; ++i) {
                const unsigned char want =
                    at != rows2.begin() + 4 ? pat(5, l, rows[(size_t)(at - rows2.begin())], i) : 0xEE;
                wrong += block(l, b)[i] != want;
            }
        }
    CHECK_EQ(wrong, 0ll);

    /* Refusals: no rows, too few rows, and a block past the carve at the first page, which must
     * fail before any page is copied. */
    std::memset(blob, 0x3C, (size_t)kv.checkpoint_bytes());
    CHECK_EQ(kv.checkpoint_save(none.data(), 2, ck, st, nullptr), RAD_E_INVAL);
    std::vector<int32_t> shorter(rows.begin(), rows.begin() + 3);
    CHECK_EQ(kv.checkpoint_save(none.data(), 2, ck, st, &shorter), RAD_E_STATE);
    std::vector<int32_t> past = rows;
    past[0] = (int32_t)p->n_blocks;
    CHECK_EQ(kv.checkpoint_save(none.data(), 2, ck, st, &past), RAD_E_STATE);
    CHECK_OK(rad_stream_sync(st));
    int64_t moved = 0;
    for (int64_t i = 0; i < kv.checkpoint_bytes(); ++i) moved += blob[i] != 0x3C;
    CHECK_EQ(moved, 0ll);
    rad_stream_destroy(st);
}

/* Linear state, a conv window and a trunk window in one slot, as on DeepSeek-V4.1: the state
 * section first in plan order, then the window pages, and a restore puts each back where it
 * belongs -- the state into the resumed sequence's slots, the pages into its seeded blocks. */
TEST(a_checkpoint_carries_linear_state_and_the_window_together) {
    if (!host_backend_or_skip("the hybrid checkpoint round trip")) return;
    Fixture f;
    f.cfg.checkpoint_slots = 4;
    std::vector<KVGroupInfo> g = hybrid_conv_groups();
    g.push_back(paged_group("swa", RAD_KV_WINDOW, 2, 16, { 7, 8 }, 64));   /* handle 4 */
    REQUIRE_EQ(f.build_groups(g), RAD_OK);
    const KVGroupPlan* w = f.kv.plan(4);
    REQUIRE(w && w->win_snap && f.kv.has_window_snapshot());
    const int64_t state = f.kv.plan(G_GDN)->bytes_per_state + f.kv.plan(G_CV)->bytes_per_state;
    const int64_t page = w->bytes_per_block, lb = w->layer_bytes_per_block();
    REQUIRE_EQ(f.kv.checkpoint_bytes(), state + w->window_blocks * page);
    auto block = [&](int64_t l, int32_t b) {
        return (unsigned char*)w->base + l * w->layer_stride() + b * lb;
    };
    for (int64_t l = 0; l < w->n_layers; ++l)
        for (int32_t b = 0; b < (int32_t)w->n_blocks; ++b)
            for (int64_t i = 0; i < lb; ++i) block(l, b)[i] = pat(7, l, b, i);

    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 400));
    const std::vector<int32_t> a = slots_of(f.kv, 1);
    fill_states(f.kv, 0);
    std::vector<int32_t> rows;
    REQUIRE(f.kv.window_rows(1, 400, &rows));
    int32_t ck = -1;
    REQUIRE_EQ(f.kv.alloc_checkpoint(&ck), RAD_OK);
    unsigned char* blob = (unsigned char*)f.kv.checkpoint_ptr(ck);
    std::memset(blob, 0x5A, (size_t)f.kv.checkpoint_bytes());
    RadStream st = nullptr;
    REQUIRE_EQ(rad_stream_create(&st, 0), RAD_OK);
    CHECK_OK(f.kv.checkpoint_save(a.data(), (int)a.size(), ck, st, &rows));
    CHECK_OK(rad_stream_sync(st));
    int64_t off = 0, wrong = 0;
    for (int32_t gi : { (int32_t)G_GDN, (int32_t)G_CV }) {
        const KVGroupPlan* p = f.kv.plan(gi);
        const int64_t sl = p->bytes_per_state / p->n_layers;
        for (int64_t l = 0; l < p->n_layers; ++l, off += sl)
            for (int64_t i = 0; i < sl; ++i) wrong += blob[off + i] != pat(0, gi * 16 + l, a[(size_t)gi], i);
    }
    CHECK_EQ(off, state);
    for (int64_t j = 0; j < w->window_blocks; ++j)
        for (int64_t l = 0; l < w->n_layers; ++l)
            for (int64_t i = 0; i < lb; ++i) {
                const unsigned char want = rows[(size_t)j] >= 0 ? pat(7, l, rows[(size_t)j], i) : 0x5A;
                wrong += blob[state + j * page + l * lb + i] != want;
            }
    CHECK_EQ(wrong, 0ll);

    /* Back into a fresh sequence: its state slots take A's copy 0, its seeded window the pages. */
    CHECK_OK(f.kv.add_sequence(2));
    CHECK_OK(f.kv.seed_window(2, 400));
    CHECK_OK(f.kv.ensure(2, 432, 400));              /* the chunk after the hit, as admit_one */
    const std::vector<int32_t> b = slots_of(f.kv, 2);
    REQUIRE(b[(size_t)G_GDN] >= 0 && b[(size_t)G_CV] >= 0);
    std::vector<int32_t> rows2;
    REQUIRE(f.kv.window_rows(2, 400, &rows2));
    for (int64_t l = 0; l < w->n_layers; ++l)
        for (int32_t r : rows2) if (r >= 0) std::memset(block(l, r), 0xEE, (size_t)lb);
    CHECK_OK(f.kv.checkpoint_restore(b.data(), (int)b.size(), ck, st, &rows2));
    CHECK_OK(rad_stream_sync(st));
    CHECK_EQ(states_wrong(f.kv, 0, b, a), 0ll);
    wrong = 0;
    for (size_t j = 0; j < rows2.size(); ++j) {
        if (rows2[j] < 0) continue;
        for (int64_t l = 0; l < w->n_layers; ++l)
            for (int64_t i = 0; i < lb; ++i) wrong += block(l, rows2[j])[i] != pat(7, l, rows[j], i);
    }
    CHECK_EQ(wrong, 0ll);
    rad_stream_destroy(st);
}

/* ================================================================== multimodal prefix keys
 *
 * Placeholder token ids are the same for every picture, so the content key of each media span
 * goes into the hash of every block it overlaps (spec §11). Two pictures behind identical
 * placeholders must never share a block; the same picture at the same place must.
 */
static void publish_mm(Fixture& f, uint64_t id, const std::vector<int32_t>& tokens,
                       const std::vector<MMContentKey>& mm) {
    CHECK_OK(f.kv.add_sequence(id));
    CHECK_OK(f.kv.ensure(id, (int64_t)tokens.size()));
    std::vector<std::vector<int32_t>> bt;
    for (int32_t g : f.pc.cached_groups()) bt.push_back(f.kv.block_table(id, g));
    CHECK_OK(f.pc.insert(id, tokens, mm, bt, (int64_t)tokens.size()));
}

TEST(a_media_span_keys_every_block_it_overlaps_and_only_those) {
    Fixture f;
    REQUIRE_EQ(f.build(/*with_linear=*/false), RAD_OK);
    const auto prompt = tok_run(5000, 96);                 /* six blocks */
    const MMContentKey k1{ 32, 48, 0x1111111111111111ull };
    publish_mm(f, 1, prompt, { k1 });
    /* The same picture: every block, backed off by one so a token remains. */
    CHECK_EQ(f.pc.lookup(prompt, { k1 }).n_attn_tokens, 80);
    /* Another picture, or none, behind the same placeholder ids: the blocks before it only. */
    CHECK_EQ(f.pc.lookup(prompt, { MMContentKey{ 32, 48, 0x2222222222222222ull } }).n_attn_tokens, 32);
    CHECK_EQ(f.pc.lookup(prompt, {}).n_attn_tokens, 32);
    /* The same picture one block later is another prompt. */
    CHECK_EQ(f.pc.lookup(prompt, { MMContentKey{ 48, 64, k1.hash } }).n_attn_tokens, 32);
    /* A span ending at a block boundary does not reach the next block. */
    CHECK_EQ(f.pc.lookup(prompt, { MMContentKey{ 32, 48, k1.hash }, MMContentKey{ 96, 112, 7 } })
                 .n_attn_tokens, 80);

    /* A span straddling blocks 1 and 2 changes both. */
    const auto other = tok_run(6000, 64);
    const MMContentKey k3{ 24, 40, 0x3333333333333333ull };
    publish_mm(f, 2, other, { k3 });
    CHECK_EQ(f.pc.lookup(other, {}).n_attn_tokens, 16);
    CHECK_EQ(f.pc.lookup(other, { k3 }).n_attn_tokens, 48);
}

/* A held chain stays on the card through drop_clean, and a hold names the chain its own media
 * keys give: a prompt with another picture behind the same placeholders shares only the blocks
 * before the picture, and the rest of its chain is dropped as usual. */
TEST(a_hold_with_media_keys_keeps_only_that_chain) {
    TierFixture x;
    REQUIRE_EQ(x.build(), RAD_OK);
    const auto prompt = tok_run(7600, 97);
    const MMContentKey k1{ 32, 48, 0xa1ull }, k2{ 32, 48, 0xb2ull };
    publish_mm(x.f, 1, prompt, { k1 });
    publish_mm(x.f, 2, prompt, { k2 });
    x.f.kv.free_sequence(1);
    x.f.kv.free_sequence(2);
    copy_now(x.t);
    x.f.pc.hold(9, prompt, { k1 });
    /* Six blocks each, two shared: the four of k2's chain past the picture go. */
    CHECK_EQ(x.f.pc.drop_clean(), 4);
    CHECK_EQ(x.f.pc.lookup(prompt, { k1 }).n_attn_tokens, 96);
    CHECK_EQ(x.f.pc.lookup(prompt, { k2 }).n_attn_tokens, 32);
    x.f.pc.release(9);
    CHECK_EQ(x.f.pc.drop_clean(), 6);
}

/* ================================================================== the scheduler's binding
 *
 * PCBinding keys a snapshot itself, with prefix_hash over prompt ++ output and the request's media
 * keys, and the next turn's lookup walks the cache's own chain over its prompt. If the two drift,
 * a hybrid model's linear state never hits: every prefix hit replays from zero, a silent loss of
 * the whole hit.
 */
static std::shared_ptr<mm::PromptMedia> one_picture(int32_t tok, int32_t n, uint64_t key) {
    auto pm = std::make_shared<mm::PromptMedia>();
    auto item = std::make_shared<mm::Item>();
    item->key_hi = key;
    item->key_lo = ~key;
    pm->items.push_back(mm::Placed{ item, tok });
    mm::RopeLayout::Run run;
    run.tok = tok;
    run.n = n;
    run.item = 0;
    pm->rope.runs.push_back(run);
    return pm;
}

static Request turn(uint64_t id, std::vector<int32_t> prompt, std::vector<int32_t> output,
                    std::shared_ptr<mm::PromptMedia> media = nullptr) {
    Request r;
    r.id = id;
    r.prompt = std::move(prompt);
    r.output = std::move(output);
    r.media = std::move(media);
    return r;
}

/* Runs one turn through the binding as the scheduler does: a snapshot reserved and committed at
 * `ck`, then the prompt ++ output published. Returns the snapshot's slot. */
static int32_t serve_turn(Fixture& f, PCBinding<PrefixCache>& b, const Request& r, int64_t ck) {
    const int64_t n = (int64_t)(r.prompt.size() + r.output.size());
    CHECK_OK(f.kv.add_sequence(r.id));
    CHECK_OK(f.kv.ensure(r.id, n));
    int32_t slot = -1;
    CHECK_OK(b.reserve_checkpoint(r, ck, &slot));
    b.commit_checkpoint(r, ck);
    std::vector<std::vector<int32_t>> bt;
    for (int32_t g : f.pc.cached_groups()) bt.push_back(f.kv.block_table(r.id, g));
    CHECK_OK(b.insert(r, bt, n));
    f.kv.free_sequence(r.id);
    return slot;
}

static std::vector<int32_t> cat(std::vector<int32_t> a, const std::vector<int32_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

TEST(the_bindings_snapshot_key_is_the_chain_the_next_turn_looks_up) {
    Fixture f;
    REQUIRE_EQ(f.build(/*with_linear=*/true, /*ckpt_slots=*/4), RAD_OK);
    PCBinding<PrefixCache> b(&f.pc);
    /* The snapshot at 320 lies past the 300-token prompt, in the reply. */
    const Request r = turn(7, tok_run(100, 300), tok_run(9000, 40));
    const int32_t slot = serve_turn(f, b, r, 320);
    REQUIRE(slot >= 0);
    const Request next = turn(8, cat(cat(r.prompt, r.output), tok_run(20000, 50)), {});
    std::vector<std::vector<int32_t>> found;
    const CacheHit h = b.lookup(next, &found, nullptr);
    CHECK_EQ(h.n_attn_tokens, 336);
    CHECK_EQ(h.n_linear_tokens, 320);
    CHECK_EQ(h.checkpoint_slot, slot);
    /* The snapshot belongs to the prompt and the reply: the same prompt with another reply
     * shares the prompt's blocks and not the snapshot. */
    const Request reply = turn(9, cat(cat(r.prompt, tok_run(30000, 40)), tok_run(20000, 50)), {});
    const CacheHit h2 = b.lookup(reply, nullptr, nullptr);
    CHECK_EQ(h2.n_attn_tokens, 288);
    CHECK_EQ(h2.checkpoint_slot, -1);
}

TEST(the_bindings_snapshot_key_carries_the_requests_pictures) {
    Fixture f;
    REQUIRE_EQ(f.build(/*with_linear=*/true, /*ckpt_slots=*/4), RAD_OK);
    PCBinding<PrefixCache> b(&f.pc);
    const Request r = turn(7, tok_run(100, 300), tok_run(9000, 40), one_picture(32, 48, 0xc0ffee));
    const int32_t slot = serve_turn(f, b, r, 320);
    REQUIRE(slot >= 0);
    const auto again = cat(cat(r.prompt, r.output), tok_run(20000, 50));
    const CacheHit same = b.lookup(turn(8, again, {}, one_picture(32, 48, 0xc0ffee)), nullptr, nullptr);
    CHECK_EQ(same.n_linear_tokens, 320);
    CHECK_EQ(same.checkpoint_slot, slot);
    /* Another picture behind the same placeholders reuses only the blocks before it. */
    const CacheHit other = b.lookup(turn(9, again, {}, one_picture(32, 48, 0xbeef)), nullptr, nullptr);
    CHECK_EQ(other.n_attn_tokens, 32);
    CHECK_EQ(other.n_linear_tokens, 0);
    CHECK_EQ(other.checkpoint_slot, -1);
    /* No picture at all: the same. */
    const CacheHit text = b.lookup(turn(10, again, {}), nullptr, nullptr);
    CHECK_EQ(text.n_attn_tokens, 32);
    CHECK_EQ(text.checkpoint_slot, -1);
}

/* Runtime LoRA: KV written under an adapter is keyed by its load's uid from the first block on, so
 * a base request and a request through another load (the same adapter reloaded included) share
 * nothing with it, and the base's own keys are those of an engine without adapters. */
TEST(the_bindings_keys_carry_the_requests_adapter_load) {
    Fixture f;
    REQUIRE_EQ(f.build(/*with_linear=*/true, /*ckpt_slots=*/4), RAD_OK);
    PCBinding<PrefixCache> b(&f.pc);
    auto via = [](Request r, uint64_t uid) {
        r.lora_slot = uid ? 0 : -1;
        r.lora_uid = uid;
        return r;
    };
    const Request r = via(turn(7, tok_run(100, 300), tok_run(9000, 40)), 5);
    const int32_t slot = serve_turn(f, b, r, 320);
    REQUIRE(slot >= 0);
    const auto again = cat(cat(r.prompt, r.output), tok_run(20000, 50));
    const CacheHit same = b.lookup(via(turn(8, again, {}), 5), nullptr, nullptr);
    CHECK_EQ(same.n_attn_tokens, 336);
    CHECK_EQ(same.checkpoint_slot, slot);
    for (uint64_t other : { 0ull, 6ull }) {
        const CacheHit h = b.lookup(via(turn(9, again, {}), other), nullptr, nullptr);
        CHECK_EQ(h.n_attn_tokens, 0);
        CHECK_EQ(h.checkpoint_slot, -1);
    }
    /* The base's blocks are keyed as without adapters: a bare-prompt lookup finds them. */
    const Request base = turn(10, tok_run(40000, 300), tok_run(41000, 40));
    serve_turn(f, b, base, 320);
    const auto next = cat(cat(base.prompt, base.output), tok_run(42000, 50));
    CHECK_EQ(f.pc.lookup(next, {}).n_attn_tokens, 336);
    CHECK_EQ(b.lookup(via(turn(11, next, {}), 5), nullptr, nullptr).n_attn_tokens, 0);
}

/* ================================================================== retention policies */

/* A retention policy is a documented extension point and nothing requires its keep set in any
 * order. The tip must survive whatever order a policy returns, or an appending conversation loses
 * the snapshot it resumes from. */
namespace {
class KeepAllTipFirst final : public CheckpointPolicy {
public:
    const char* name() const override { return "keep-all, tip first"; }
    void select(const std::vector<int64_t>& pos, std::vector<int64_t>* keep) const override {
        keep->assign(pos.rbegin(), pos.rend());
    }
};
}  /* namespace */

TEST(retention_keeps_what_the_policy_names_in_any_order) {
    Fixture f;
    REQUIRE_EQ(f.build(/*with_linear=*/true, /*ckpt_slots=*/8), RAD_OK);
    f.pc.set_policy(std::make_unique<KeepAllTipFirst>());
    const auto prompt = tok_run(4000, 400);
    const auto hs = chain_of(prompt);
    for (int64_t pos : { 64, 128, 192, 256, 320 }) ckpt_at(f, 1, hs, pos);
    CHECK_EQ(f.pc.retain(1), 0ll);
    CHECK(f.pc.checkpoints(1) == (std::vector<int64_t>{ 64, 128, 192, 256, 320 }));
}

/* ================================================================== the drop recheck list */

/* drop_clean keeps a clean entry a sequence still reads on a recheck list, once. An entry that
 * leaves the card, comes back clean and is read again while still on the list must not be listed a
 * second time: the list would grow by one per round trip. */
TEST(an_entry_waiting_for_its_readers_is_listed_once) {
    TierFixture x;
    REQUIRE_EQ(x.build(), RAD_OK);
    const auto prompt = tok_run(8100, 49);
    CHECK_OK(x.f.kv.add_sequence(1));
    CHECK_OK(x.f.kv.ensure(1, 48));
    CHECK_OK(x.f.pc.insert(1, prompt, {}, { x.f.kv.block_table(1, G_FULL) }, 48));
    copy_now(x.t);
    CHECK_EQ(x.f.pc.drop_clean(), 0);                   /* sequence 1 still reads all three */
    CHECK_EQ(x.f.pc.drop_waiting(), 3u);

    /* A rejected draft gives the blocks back without ending the sequence, so nothing triggers the
     * recheck; pressure moves the entries off the card. */
    CHECK_OK(x.f.kv.rollback(1, 0));
    CHECK_EQ(x.f.pc.trim(3), 3);
    /* A restore brings them back clean, and a new reader adopts them. */
    const std::vector<BlockHash> keys = x.f.pc.restorable_from(prompt, {});
    REQUIRE_EQ(keys.size(), 3u);
    for (const BlockHash& k : keys) {
        TierJob j;
        REQUIRE(x.t.plan_promote(k, &j));
        int32_t b = -1;
        CHECK_OK(x.f.kv.reserve_block(G_FULL, &b));
        j.blocks = { b };
        j.started = true;
        x.t.report(j, true);
        CHECK_OK(x.f.pc.rebind(k, { b }));
    }
    std::vector<std::vector<int32_t>> found;
    REQUIRE_EQ(x.f.pc.lookup(prompt, {}, &found).n_attn_tokens, 48);
    CHECK_OK(x.f.kv.add_sequence(2));
    CHECK_OK(x.f.kv.adopt(2, G_FULL, found[0], 48));
    CHECK_EQ(x.f.pc.drop_clean(), 0);
    CHECK_EQ(x.f.pc.drop_waiting(), 3u);
    /* Once the reader goes, one recheck drops each entry once. */
    x.f.kv.free_sequence(2);
    x.f.kv.free_sequence(1);
    CHECK_EQ(x.f.pc.drop_clean(), 3);
    CHECK_EQ(x.f.pc.drop_waiting(), 0u);
}

/* ================================================================== the VRAM budget */

/* Hand-computed splits. A 16 GiB card, 1 GiB already held, 96 MiB headroom, 16 MiB drift and a
 * 152 MiB arena leave 15096 MiB claimable; 4096 MiB of static weights leave 11000 MiB elastic. The
 * expert plane is 20000 MiB at 500 MiB a layer, and one full-length sequence's cache is 1000 MiB.
 *   No lender: the cache keeps one sequence (1000), the experts take 10000, half the plane is off
 *   the card, so the prefill stage is two buffers of 250 MiB taken from the experts.
 *   Lender: the experts keep their 5000 MiB floor and the cache takes 6000; three quarters of the
 *   plane is off the card, so the stage is two buffers of 375 MiB taken from the cache. */
static VramFacts hand_facts(bool lends) {
    constexpr int64_t MiB = 1 << 20;
    VramFacts f{};
    f.capacity = 16384 * MiB;
    f.free = f.capacity - 1024 * MiB;
    f.arena = 152 * MiB;
    f.statics = 4096 * MiB;
    f.experts = 20000 * MiB;
    f.expert_layer_max = 500 * MiB;
    f.expert_floor = 5000 * MiB;
    f.kv_ceiling = 8000 * MiB;
    f.kv_one = 1000 * MiB;
    f.lends = lends;
    return f;
}

TEST(the_prefill_stage_is_two_layers_of_what_the_card_leaves_off) {
    constexpr int64_t MiB = 1 << 20;
    {
        Config c = derive_config();
        VramBudget b;
        CHECK_OK(vram_budget_trial(c, { hand_facts(false) }, &b));
        CHECK_EQ(b.elastic, 11000 * MiB);
        CHECK_EQ(b.kv, 1000 * MiB);
        CHECK_EQ(b.stage, 500 * MiB);
        CHECK_EQ(b.experts, 9500 * MiB);
        CHECK_EQ(b.weights, 13596 * MiB);
        int rc = 0;
        stderr_of([&] { rc = vram_budget_resolve(c, { hand_facts(false) }, &b); });
        CHECK_OK(rc);
        CHECK_EQ(c.vram_weights_mib, 13596ll);
        CHECK_EQ(c.vram_kv_mib, 1000ll);
        CHECK(has(b.report, "prefill staging"));
    }
    {
        Config c = derive_config();
        VramBudget b;
        int rc = 0;
        stderr_of([&] { rc = vram_budget_resolve(c, { hand_facts(true) }, &b); });
        CHECK_OK(rc);
        CHECK_EQ(b.experts, 5000 * MiB);
        CHECK_EQ(b.stage, 750 * MiB);
        CHECK_EQ(b.kv, 5250 * MiB);
        CHECK_EQ(c.vram_weights_mib, 9096ll);
        CHECK_EQ(c.vram_kv_mib, 5250ll);
    }
    /* A placement that moves no unit has nothing to stage. */
    for (const char* pl : { "all_vram", "layer_offload" }) {
        Config c = derive_config();
        c.placement = pl;
        VramBudget b;
        CHECK_OK(vram_budget_trial(c, { hand_facts(false) }, &b));
        CHECK_EQ(b.stage, 0ll);
        CHECK_EQ(b.experts, 10000 * MiB);
    }
    /* A cache that fits whole gets a pool covering its ceiling: one under a MiB is not truncated
     * to an empty pool, nor one just over a MiB to a pool short of its context. */
    for (int64_t ceiling : { 600 * 1024ll, MiB + 200 * 1024ll }) {
        Config c = derive_config();
        VramFacts f = hand_facts(false);
        f.experts = 0;
        f.expert_layer_max = 0;
        f.expert_floor = 0;
        f.kv_ceiling = ceiling;
        f.kv_one = ceiling;
        VramBudget b;
        int rc = 0;
        stderr_of([&] { rc = vram_budget_resolve(c, { f }, &b); });
        CHECK_OK(rc);
        CHECK_EQ(b.kv, ceiling);
        CHECK_EQ(c.vram_kv_mib, (ceiling + MiB - 1) / MiB);
        CHECK_EQ(c.vram_weights_mib, 4096ll);
    }
    /* Integrated graphics: the host's share replaces a smaller headroom, by name. */
    {
        Config c = derive_config();
        VramFacts f = hand_facts(false);
        f.host_reserve = 2048 * MiB;
        VramBudget b;
        int rc = 0;
        stderr_of([&] { rc = vram_budget_resolve(c, { f, hand_facts(false) }, &b); });
        CHECK_OK(rc);
        CHECK_EQ(b.headroom, 2048 * MiB);
        CHECK_EQ(b.claimable, (15096 - 1952) * MiB);
        CHECK(has(b.report, "left to the host"));
    }
}

/* Over random cards and programs the split either refuses by name or hands out no more than the
 * card: every granted pool plus every fixed term fits the capacity, with at most the MiB each of
 * the two pools may round into the headroom. The shares add up to the elastic part, the static
 * weights are always covered, the cache gets at least one full-length sequence when that fits,
 * and the trial gives the resolver's split without writing the configuration. */
TEST(the_vram_budget_never_grants_more_than_the_card) {
    constexpr int64_t MiB = 1 << 20;
    const uint64_t seed = 0x3c6ef372fe94f82bull;
    Xorshift r(seed);
    std::string failures;
    int64_t refused = 0, granted = 0, staged = 0, lent = 0, shared = 0;
    auto rnd = [&](int64_t hi) { return hi > 0 ? (int64_t)r.below((uint64_t)hi + 1) : 0; };
    stderr_of([&] {
        for (int it = 0; it < 3000; ++it) {
            Config cfg = derive_config();
            cfg.gpu_headroom_mib = rnd(512);
            static const char* pl[] = { "auto", "expert_tiered", "all_vram", "layer_offload" };
            cfg.placement = pl[r.below(4)];
            const int ranks = 1 + (int)r.below(4);
            const int64_t cap = (256 + rnd(64 * 1024 - 256)) * MiB;
            const bool moe = r.below(10) < 7, lends = r.below(2) != 0;
            std::vector<VramFacts> fs;
            int64_t kv_one = 0, kv_cap = 0, exp_cap = 0, floor_max = 0;
            for (int k = 0; k < ranks; ++k) {
                VramFacts f{};
                f.capacity = cap - rnd(8) * MiB;
                f.free = f.capacity - rnd(cap / 8);
                f.arena = rnd(cap / 6);
                f.staging = rnd(64 * MiB);
                f.n_queues = (int)r.below(5);
                f.queue_bytes = r.below(2) ? 33 * MiB : 0;
                f.prefetch = r.below(3) ? 0 : rnd(256 * MiB);
                f.kv_ceiling = MiB + rnd(cap);
                f.kv_fixed = rnd(f.kv_ceiling / 4);
                f.kv_one = f.kv_fixed + rnd(f.kv_ceiling - f.kv_fixed);
                f.statics = rnd(cap / 2);
                f.experts = moe ? rnd(2 * cap) : 0;
                f.expert_layer_max = f.experts / (1 + rnd(60));
                f.expert_floor = rnd(f.experts);
                f.lends = lends;
                f.host_reserve = r.below(5) ? 0 : rnd(4096 * MiB);
                kv_one = std::max(kv_one, f.kv_one);
                kv_cap = std::max(kv_cap, f.kv_ceiling);
                exp_cap = std::max(exp_cap, f.experts);
                floor_max = std::max(floor_max, f.expert_floor);
                fs.push_back(f);
            }
            const Config before = cfg;
            VramBudget t;
            const int trc = vram_budget_trial(cfg, fs, &t);
            const bool untouched = cfg.vram_weights_mib == before.vram_weights_mib &&
                                   cfg.vram_kv_mib == before.vram_kv_mib;
            VramBudget b;
            const int rc = vram_budget_resolve(cfg, fs, &b);
            auto bad = [&](const char* what) {
                if (failures.size() < 2000)
                    failures += fmt("    seed %#llx case %d: %s\n", (unsigned long long)seed, it, what);
            };
            if (trc != rc || t.kv != b.kv || t.experts != b.experts || t.stage != b.stage ||
                t.weights != b.weights)
                bad("the trial disagrees with the resolver");
            if (!untouched) bad("the trial wrote the configuration");
            if (rc != RAD_OK) {
                ++refused;
                /* Short of the fixed costs, or the weights leave the cache nothing. */
                const bool short_by = b.claimable < 0 && has(b.report, "short by");
                const bool no_cache = has(b.report, "The KV cache gets nothing");
                if (rc != RAD_E_NOMEM || !(short_by || no_cache))
                    bad("a refusal without its shortfall");
                continue;
            }
            ++granted;
            staged += b.stage > 0;
            lent += b.lends;
            shared += b.headroom > cfg.gpu_headroom_mib * MiB;
            const int64_t w = cfg.vram_weights_mib * MiB, kvb = cfg.vram_kv_mib * MiB;
            const int64_t total = w + kvb + b.stage + b.already_held + b.drift + b.headroom +
                                  b.arena + b.staging + b.queues + b.prefetch;
            if (total > b.capacity + 2 * MiB) bad("the pools and fixed terms exceed the card");
            if (b.headroom < cfg.gpu_headroom_mib * MiB) bad("the headroom shrank");
            if (w < b.statics) bad("the weights pool does not cover the static weights");
            if (b.experts + b.kv + b.unusable + b.stage != b.elastic) bad("the shares do not add up");
            if (b.experts < 0 || b.kv < 0 || b.unusable < 0 || b.stage < 0) bad("a negative share");
            if (kvb > std::max(b.kv, (kv_cap + MiB - 1) / MiB * MiB))
                bad("the KV pool exceeds the cache's share");
            if (b.kv >= kv_cap && kvb < kv_cap) bad("a cache given its ceiling cannot hold it");
            if (b.kv > kv_cap) bad("the cache past its ceiling");
            if (b.experts > std::max(exp_cap, floor_max)) bad("the experts past their ceiling");
            if (b.kv < std::min(kv_one, b.elastic)) bad("the cache below one full-length sequence");
            if (b.stage > 0 && (cfg.placement == "all_vram" || cfg.placement == "layer_offload"))
                bad("a stage under a placement that moves nothing");
        }
    });
    if (!failures.empty()) fprintf(stderr, "%s", failures.c_str());
    CHECK(failures.empty());
    CHECK(refused > 0);
    CHECK(granted > 1000);
    CHECK(staged > 0);
    CHECK(lent > 0);
    CHECK(shared > 0);
}

/* ================================================================== the scheduler's fake KV manager
 *
 * sched_test drives the scheduler against FakeKV, a small block manager, so every scheduler test
 * assumes the fake answers the scheduler's calls as KVManager does. One script runs through both
 * behind the same IKVManager interface, and every answer the scheduler can read is compared. Block
 * ids may differ (the real free list is a min-heap, the fake's a stack); refusals, counts, change
 * marks and slot ownership may not.
 *
 * sched_test builds its program without the null group at handle 0, so the fake's group g is the
 * manager's g + 1. The fake is tests/sched_fakes.h's, the one sched_test runs on.
 */

/* Every answer the scheduler can read about `ids`, from both managers. */
static bool kv_agree(IKVManager& real, IKVManager& fake, const std::vector<uint64_t>& ids,
                     const char* step) {
    bool ok = true;
    auto differ = [&](const char* what, long long a, long long b) {
        fprintf(stderr, "    after %s: %s is %lld in KVManager and %lld in FakeKV\n", step, what, a, b);
        ok = false;
    };
    for (int32_t g = 1; g < 4; ++g) {
        if (real.block_tokens(g) != fake.block_tokens(g - 1))
            differ("block_tokens", real.block_tokens(g), fake.block_tokens(g - 1));
    }
    if (real.free_blocks(G_FULL) != fake.free_blocks(0))
        differ("free_blocks", real.free_blocks(G_FULL), fake.free_blocks(0));
    if (real.held_blocks(G_FULL) != fake.held_blocks(0))
        differ("held_blocks", real.held_blocks(G_FULL), fake.held_blocks(0));
    if (real.total_blocks(G_FULL) != fake.total_blocks(0))
        differ("total_blocks", real.total_blocks(G_FULL), fake.total_blocks(0));
    for (uint64_t id : ids) {
        if (real.has_sequence(id) != fake.has_sequence(id))
            differ("has_sequence", real.has_sequence(id), fake.has_sequence(id));
        if (real.block_table(id, G_FULL).size() != fake.block_table(id, 0).size())
            differ("the block table's length", (long long)real.block_table(id, G_FULL).size(),
                   (long long)fake.block_table(id, 0).size());
        const int64_t rc = real.take_table_changes(id, G_FULL), fc = fake.take_table_changes(id, 0);
        if (rc != fc) differ("take_table_changes", rc, fc);
        if (real.first_block_pos(id, G_FULL) != fake.first_block_pos(id, 0))
            differ("first_block_pos", real.first_block_pos(id, G_FULL), fake.first_block_pos(id, 0));
        for (int32_t g : { (int32_t)G_GDN, (int32_t)G_CV })
            if ((real.state_slot(id, g) >= 0) != (fake.state_slot(id, g - 1) >= 0))
                differ("holding a state slot", real.state_slot(id, g) >= 0,
                       fake.state_slot(id, g - 1) >= 0);
    }
    return ok;
}

TEST(the_schedulers_fake_kv_manager_answers_as_the_real_one) {
    Fixture f;
    REQUIRE_EQ(f.build_groups(hybrid_conv_groups()), RAD_OK);
    KVBinding<KVManager> real(&f.kv);
    sched_fake::FakeKV fake;
    std::vector<KVGroupInfo> groups = hybrid_conv_groups();
    groups.erase(groups.begin());
    fake.setup(groups, f.cfg.max_ctx, f.kv.total_blocks(G_FULL), f.kv.plan(G_GDN)->n_seq_states());
    const int64_t total = f.kv.total_blocks(G_FULL);
    std::vector<uint64_t> ids;
    bool ok = true;
    auto both = [&](const char* step, const std::function<int(IKVManager&, int32_t)>& call) {
        const int a = call(real, 0), b = call(fake, -1);
        if (a != b) {
            fprintf(stderr, "    %s: KVManager returned %s and FakeKV %s\n", step, rad_strerror(a),
                    rad_strerror(b));
            ok = false;
        }
        ok = kv_agree(real, fake, ids, step) && ok;
    };
    /* `off` maps a group handle: 0 for the manager, -1 for the fake. */
    for (uint64_t id : { 1, 2, 3 }) {
        ids.push_back(id);
        both("add", [&](IKVManager& m, int32_t) { return m.add_sequence(id); });
    }
    both("ensure 1 to 40", [](IKVManager& m, int32_t) { return m.ensure(1, 40); });
    both("ensure 2 to 17 from 0", [](IKVManager& m, int32_t) { return m.ensure_from(2, 17, 0); });
    both("ensure 3 to 16", [](IKVManager& m, int32_t) { return m.ensure(3, 16); });
    both("ensure 1 to 41", [](IKVManager& m, int32_t) { return m.ensure(1, 41); });
    both("ensure 1 to 49", [](IKVManager& m, int32_t) { return m.ensure(1, 49); });
    both("rollback 1 to 20", [](IKVManager& m, int32_t) { return m.rollback(1, 20); });
    both("rollback 1 to 20 again", [](IKVManager& m, int32_t) { return m.rollback(1, 20); });
    ids.push_back(4);
    both("add 4", [](IKVManager& m, int32_t) { return m.add_sequence(4); });
    both("adopt 2's first block into 4", [](IKVManager& m, int32_t off) {
        const std::vector<int32_t> one(1, m.block_table(2, G_FULL + off)[0]);
        return m.adopt(4, G_FULL + off, one, 16);
    });
    both("ensure 4 to 20", [](IKVManager& m, int32_t) { return m.ensure(4, 20); });
    both("preempt 2", [](IKVManager& m, int32_t) { return m.preempt(2); });
    both("ensure 2 to 16", [](IKVManager& m, int32_t) { return m.ensure(2, 16); });
    both("free 3", [](IKVManager& m, int32_t) { m.free_sequence(3); return 0; });
    /* Eight state slots: the ninth sequence asking for one is refused, and the refusal takes no
     * block either. */
    for (uint64_t id = 5; id <= 12; ++id) {
        ids.push_back(id);
        both("add", [&](IKVManager& m, int32_t) { return m.add_sequence(id); });
        both("ensure to 16 with the state slots running out",
             [&](IKVManager& m, int32_t) { return m.ensure(id, 16); });
    }
    /* The pool's edge. */
    both("ensure 1 past the pool", [&](IKVManager& m, int32_t) { return m.ensure(1, total * 16 + 1); });
    both("ensure 1 to the whole pool", [&](IKVManager& m, int32_t) { return m.ensure(1, total * 16); });
    CHECK_EQ(real.can_ever_fit(total * 16), fake.can_ever_fit(total * 16));
    CHECK_EQ(real.can_ever_fit(total * 16 + 1), fake.can_ever_fit(total * 16 + 1));
    for (uint64_t id : ids)
        both("free", [&](IKVManager& m, int32_t) { m.free_sequence(id); return 0; });
    CHECK(ok);
    CHECK_EQ(real.free_blocks(G_FULL), total);
}

/* ================================================================== the tiers, at random
 *
 * The prefix cache, the idle tiers and the KV manager hand blocks and records between them in an
 * order no scenario test fixes: publishes and hits, copies that land or fail, drops, trims,
 * evictions, capacity drops, holds and restores that start or never start. After every operation
 * the three must agree: each held block is named by a live sequence or exactly one resident entry,
 * the cache-only figure is a recount, an off-device entry is exactly a host record, the arena holds
 * exactly the host copies, and a lookup hands out only held blocks, each the one block of its
 * prefix position.
 */
namespace {
struct TierRun {
    uint64_t seed;
    Xorshift r;
    TierFixture x;
    std::vector<uint64_t> live;
    uint64_t next_id = 1;
    int op = 0;
    std::string what;
    bool failed = false;
    std::map<std::string, int64_t> seen;
    static constexpr int kFamilies = 6, kMaxLen = 176;

    explicit TierRun(uint64_t s) : seed(s), r(s) {}

    bool fail(const std::string& why) {
        if (!failed)
            fprintf(stderr, "    seed %#llx op %d (%s): %s\n", (unsigned long long)seed, op,
                    what.c_str(), why.c_str());
        failed = true;
        return false;
    }
    int build() {
        int s = x.f.build(/*with_linear=*/false, /*ckpt_slots=*/0);
        if (s < 0) return s;
        /* A small arena, so capacity drops happen. */
        s = x.t.configure(tier_config(24 * 1024, 1024));
        if (s < 0) return s;
        x.f.pc.set_tiers(&x.t);
        x.t.set_drop_sink([this](const BlockHash& h) { x.f.pc.drop_offdevice(h); });
        return RAD_OK;
    }
    static std::vector<int32_t> prompt(int fam, int len) { return tok_run(1000 * (fam + 1), len); }

    /* The engine's restore, against the real tiers and manager: promote the chain, reserve its
     * blocks with one recall, cut it where the pool ran out, report each job, rebind the ones that
     * landed. A started copy may fail; then its blocks go back and the entry is dropped. */
    void restore(const std::vector<int32_t>& toks) {
        const std::vector<BlockHash> want = x.f.pc.restorable_from(toks, {});
        if (want.empty()) return;
        std::vector<TierJob> jobs;
        for (const BlockHash& h : want) {
            TierJob j;
            if (!x.t.plan_promote(h, &j)) break;
            jobs.push_back(std::move(j));
        }
        std::vector<int32_t> ids;
        const int64_t m = x.f.kv.reserve_blocks(G_FULL, (int64_t)jobs.size(), &ids);
        for (size_t i = 0; i < jobs.size(); ++i) {
            TierJob& j = jobs[i];
            const bool reserved = (int64_t)i < m;
            j.started = reserved;
            j.blocks.clear();
            if (reserved) j.blocks.push_back(ids[i]);
            const bool ok = reserved && r.below(6) != 0;
            if (!ok) {
                for (int32_t b : j.blocks) x.f.kv.release_block(G_FULL, b);
                j.blocks.clear();
                ++seen[reserved ? "restore failed" : "restore never started"];
            }
            x.t.report(j, ok, rad_mono_ns());
            if (!ok) continue;
            if (x.f.pc.rebind(j.key, j.blocks) != RAD_OK) {
                x.f.kv.release_block(G_FULL, j.blocks[0]);
                x.t.forget(j.key);
                ++seen["rebind refused"];
            } else {
                ++seen["restored"];
            }
        }
    }

    bool check() {
        const KVGroupPlan* p = x.f.kv.plan(G_FULL);
        std::vector<int32_t> in_seq((size_t)p->n_blocks, 0);
        int64_t tables = 0;
        for (uint64_t id : live)
            for (int32_t b : x.f.kv.block_table(id, G_FULL)) { ++in_seq[(size_t)b]; ++tables; }
        int64_t refs = 0, held = 0, cache_only = 0;
        for (int64_t b = 0; b < p->n_blocks; ++b) {
            const int32_t rc = x.f.kv.block_refcount(G_FULL, (int32_t)b);
            refs += rc;
            held += rc > 0;
            cache_only += rc == 1 && in_seq[(size_t)b] == 0;
            if (rc < in_seq[(size_t)b]) return fail(fmt("block %lld is in tables %d times at refcount %d",
                                                         (long long)b, in_seq[(size_t)b], rc));
        }
        if (refs != tables + x.f.pc.resident())
            return fail(fmt("%lld references for %lld table entries and %lld resident entries",
                            (long long)refs, (long long)tables, (long long)x.f.pc.resident()));
        if (x.f.kv.held_blocks(G_FULL) != held) return fail("held_blocks disagrees with a recount");
        if (x.f.kv.pool_bytes_used(G_FULL) != held * p->bytes_per_block) return fail("pool bytes");
        if (x.f.pc.unshared_tokens() != 16 * cache_only)
            return fail(fmt("unshared_tokens %lld, recount %lld", (long long)x.f.pc.unshared_tokens(),
                            (long long)(16 * cache_only)));

        /* Every key the families can reach: off-device entries are host records and nothing
         * else, the arena holds the host copies of those and of clean device entries, and a hit
         * hands out held blocks, one per prefix position. */
        int64_t off = 0, clean = 0;
        std::map<int32_t, std::pair<int, int>> owner;
        for (int fam = 0; fam < kFamilies; ++fam) {
            const std::vector<int32_t> toks = prompt(fam, kMaxLen);
            const std::vector<BlockHash> hs = chain_of(toks);
            for (const BlockHash& h : hs) {
                off += x.t.resident_off_device(h);
                clean += x.t.droppable(h);
            }
            std::vector<std::vector<int32_t>> found;
            const CacheHit hit = x.f.pc.lookup(toks, {}, &found);
            for (size_t k = 0; k < found[0].size(); ++k) {
                const int32_t b = found[0][k];
                if (x.f.kv.block_refcount(G_FULL, b) < 1) return fail(fmt("a hit hands out free block %d", b));
                const auto at = owner.emplace(b, std::make_pair(fam, (int)k));
                if (!at.second && at.first->second != std::make_pair(fam, (int)k))
                    return fail(fmt("block %d serves two prefixes", b));
            }
            if (hit.n_attn_tokens != (int32_t)(16 * found[0].size())) return fail("hit length");
        }
        if (x.f.pc.size() - x.f.pc.resident() != off)
            return fail(fmt("%lld entries off the card and %lld host records",
                            (long long)(x.f.pc.size() - x.f.pc.resident()), (long long)off));
        if (x.t.arena().used_slots() != off + clean)
            return fail(fmt("%lld arena slots for %lld host and %lld clean entries",
                            (long long)x.t.arena().used_slots(), (long long)off, (long long)clean));
        return true;
    }

    void step() {
        const uint64_t k = r.below(100);
        if (k < 22) {
            const int fam = (int)r.below(kFamilies), len = 16 + (int)r.below(kMaxLen - 15);
            if (live.size() >= 8) return;
            const std::vector<int32_t> toks = prompt(fam, len);
            const uint64_t id = next_id++;
            what = fmt("publish %llu: family %d, %d tokens", (unsigned long long)id, fam, len);
            if (r.below(2)) restore(toks);
            if (x.f.kv.add_sequence(id) != RAD_OK) { fail("add refused"); return; }
            std::vector<std::vector<int32_t>> found;
            const CacheHit h = x.f.pc.lookup(toks, {}, &found);
            if (h.n_attn_tokens > 0) {
                if (x.f.kv.adopt(id, G_FULL, found[0], h.n_attn_tokens) != RAD_OK) { fail("adopt"); return; }
                ++seen["hit"];
            }
            int st = x.f.kv.ensure(id, len);
            if (st == RAD_E_FULL) {
                x.f.pc.trim(64);
                x.f.pc.evict_lru(64);
                st = x.f.kv.ensure(id, len);
            }
            if (st != RAD_OK) { x.f.kv.free_sequence(id); ++seen["publish refused"]; return; }
            if (x.f.pc.insert(id, toks, {}, { x.f.kv.block_table(id, G_FULL) }, len) != RAD_OK) {
                fail("insert refused");
                return;
            }
            live.push_back(id);
            ++seen["publish"];
        } else if (k < 36) {
            if (live.empty()) return;
            const size_t i = (size_t)r.below(live.size());
            what = fmt("free %llu", (unsigned long long)live[i]);
            x.f.kv.free_sequence(live[i]);
            live.erase(live.begin() + (ptrdiff_t)i);
            ++seen["free"];
        } else if (k < 52) {
            std::vector<TierJob> jobs;
            x.t.plan_writeback(1 + (int64_t)r.below(8), &jobs);
            what = fmt("copy %zu", jobs.size());
            for (TierJob& j : jobs) {
                const bool ok = r.below(5) != 0;
                x.t.report(j, ok);
                ++seen[ok ? "copied" : "copy failed"];
            }
        } else if (k < 57) {
            std::vector<TierJob> jobs;
            x.t.plan(1 + (int)r.below(4), &jobs);
            what = fmt("capacity plan %zu", jobs.size());
            for (TierJob& j : jobs) { x.t.report(j, true); ++seen["host drop"]; }
        } else if (k < 67) {
            what = "drop_clean";
            seen["dropped"] += x.f.pc.drop_clean();
        } else if (k < 75) {
            what = "trim";
            seen["trimmed"] += x.f.pc.trim((int64_t)r.below(16));
        } else if (k < 79) {
            what = "evict_lru";
            seen["evicted"] += x.f.pc.evict_lru((int64_t)r.below(6));
        } else if (k < 89) {
            const int fam = (int)r.below(kFamilies), len = 16 + (int)r.below(kMaxLen - 15);
            what = fmt("restore family %d, %d tokens", fam, len);
            restore(prompt(fam, len));
        } else if (k < 95) {
            const uint64_t req = 1 + r.below(4);
            const int fam = (int)r.below(kFamilies);
            if (r.below(2)) {
                what = fmt("hold %llu", (unsigned long long)req);
                x.f.pc.hold(req, prompt(fam, kMaxLen), {});
                ++seen["hold"];
            } else {
                what = fmt("release %llu", (unsigned long long)req);
                x.f.pc.release(req);
            }
        } else if (k < 99) {
            what = "begin_step";
            x.f.pc.begin_step();
        } else {
            what = "expire";
            seen["expired"] += x.f.pc.expire(rad_mono_ns(), 1);
        }
    }
};
}  /* namespace */

TEST(the_cache_tiers_and_pool_agree_over_random_interleavings) {
    TierRun run(0xbb67ae8584caa73bull);
    REQUIRE_EQ(run.build(), RAD_OK);
    for (run.op = 0; run.op < 3000 && !run.failed; ++run.op) {
        run.step();
        if (!run.failed) run.check();
    }
    CHECK(!run.failed);
    for (const char* k : { "publish", "hit", "free", "copied", "copy failed", "host drop", "dropped",
                           "trimmed", "evicted", "restored", "restore failed", "hold" }) {
        if (run.seen[k] <= 0) fprintf(stderr, "    no '%s' in the run\n", k);
        CHECK(run.seen[k] > 0);
    }
    for (uint64_t id : run.live) run.x.f.kv.free_sequence(id);
    run.live.clear();
    run.x.f.pc.clear();
    CHECK_EQ(run.x.f.kv.held_blocks(G_FULL), 0ll);
    CHECK_EQ(run.x.f.pc.unshared_tokens(), 0ll);
    CHECK_EQ(run.x.t.arena().used_slots(), 0ll);
}

/* ================================================================== a restore makes room
 *
 * A conversation whose blocks went to the host tier comes back whole on its next turn, when the
 * pool can be made to hold it. The reserve's trim leaves a quarter of the pool free after every
 * publish; a restore larger than that gives up VRAM copies other idle conversations keep on the
 * host as well, the same way pressure does, rather than stopping where the free list ends. These
 * drive the engine's restore against the real cache, tiers and manager, with no bytes moving.
 */

/* The engine's restore (the binding's restore_fn) without the transfer: which entries come back,
 * into which blocks, and what the tiers and the cache are told. Returns how many came back;
 * `res`, when given, is whether the plan was cut. */
static int64_t engine_restore(TierFixture& x, PCBinding<PrefixCache>& b, const Request& r,
                              RestoreResult* res = nullptr) {
    const std::vector<BlockHash> want = b.restore_plan(r, 0, /*whole_only=*/false, res);
    std::vector<TierJob> jobs;
    for (const BlockHash& h : want) {
        TierJob j;
        if (!x.t.plan_promote(h, &j)) break;
        jobs.push_back(std::move(j));
    }
    std::vector<int32_t> ids;
    const int64_t m = x.f.kv.reserve_blocks(G_FULL, (int64_t)jobs.size(), &ids);
    int64_t back = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        TierJob& j = jobs[i];
        const bool ok = (int64_t)i < m;
        j.started = ok;
        j.blocks.clear();
        if (ok) j.blocks.push_back(ids[i]);
        x.t.report(j, ok, rad_mono_ns());
        if (ok) { CHECK_OK(x.f.pc.rebind(j.key, j.blocks)); ++back; }
    }
    return back;
}

/* Two conversations of 0.6 of the pool each, as the scheduler leaves them: A published and copied
 * to the host, B admitted by giving up A's VRAM under pressure, published, and the reserve's trim
 * run while B's sequence still holds its blocks (Scheduler::close_request publishes before it
 * frees). A ends up mostly on the host with its first blocks still in VRAM. `b_live` keeps B's
 * sequence running; otherwise it ends with no host copy yet. */
struct TwoSessions {
    int64_t n = 0;                       /* blocks each conversation published */
    std::vector<int32_t> a, b;           /* prompts that hit the whole of each */
    std::vector<BlockHash> ha;
};
static TwoSessions two_sessions(TierFixture& x, bool b_live, bool a_snapshots) {
    TwoSessions s;
    const int64_t total = x.f.kv.total_blocks(G_FULL);
    s.n = (total * 3 / 5) & ~(int64_t)3;            /* on the 64-token checkpoint interval */
    CHECK(2 * s.n <= x.t.arena().capacity());
    s.a = tok_run(100000, (int)(s.n * 16 + 16));
    s.b = tok_run(900000, (int)(s.n * 16 + 16));
    s.ha = chain_of(s.a);

    publish(x.f, 1, s.a, s.n * 16);
    if (a_snapshots) {
        /* A snapshot on every interval, retention after each, as a long prefill leaves them. */
        for (int64_t pos = 64; pos <= s.n * 16; pos += 64) {
            x.f.pc.begin_step();
            ckpt_at(x.f, 1, s.ha, pos);
            x.f.pc.retain(1);
        }
        x.f.pc.begin_step();
    }
    x.f.kv.free_sequence(1);
    copy_now(x.t);

    const int64_t room = total - x.f.kv.held_blocks(G_FULL);
    CHECK_EQ(x.f.pc.trim(s.n + 1 - room), s.n + 1 - room);
    publish(x.f, 2, s.b, s.n * 16);
    const int64_t short_by = total / 4 - (total - x.f.kv.held_blocks(G_FULL));
    if (short_by > 0) (void)x.f.pc.trim(short_by);
    if (!b_live) x.f.kv.free_sequence(2);
    return s;
}

TEST(a_restore_gives_up_other_clean_copies_to_bring_the_whole_chain_back) {
    TierFixture x;
    CHECK_OK(x.build());
    const TwoSessions s = two_sessions(x, /*b_live=*/false, /*a_snapshots=*/false);
    const int64_t off = (int64_t)x.f.pc.restorable_from(s.a, {}).size();
    const int64_t free_now = x.f.kv.total_blocks(G_FULL) - x.f.kv.held_blocks(G_FULL);
    REQUIRE(off > free_now);                        /* the free list alone cannot hold it */
    const int64_t evictions = x.f.pc.stats().evictions;

    PCBinding<PrefixCache> b(&x.f.pc);
    int flushes = 0;
    b.flush_fn = [&]() { ++flushes; return !copy_now(x.t).empty(); };
    RestoreResult res = RestoreResult::Cut;
    CHECK_EQ(engine_restore(x, b, turn(3, s.a, {}), &res), off);
    CHECK(res == RestoreResult::Whole);
    CHECK_EQ(x.f.pc.lookup(s.a, {}).n_attn_tokens, (int32_t)(s.n * 16));
    /* B had no host copy yet: its copies were landed first, then its VRAM went. Nothing lost. */
    CHECK_EQ(flushes, 1);
    CHECK_EQ(x.f.pc.stats().evictions, evictions);
    CHECK_EQ(x.f.pc.size(), 2 * s.n);
    CHECK_EQ((int64_t)x.f.pc.restorable_from(s.b, {}).size(), 2 * s.n - x.f.pc.resident());
}

/* With recurrent state the hit resumes at the deepest snapshot the restored prefix reaches; with
 * the whole chain back, that is the tip, not a snapshot down the retention ladder. */
TEST(a_restored_hybrid_conversation_resumes_at_its_tip_snapshot) {
    TierFixture x;
    CHECK_OK(x.build(/*with_linear=*/true, /*ckpt_slots=*/32));
    x.f.pc.set_policy(std::make_unique<GeometricBackoff>(64));
    const TwoSessions s = two_sessions(x, /*b_live=*/false, /*a_snapshots=*/true);
    REQUIRE_EQ(x.f.pc.checkpoints(1).back(), s.n * 16);

    PCBinding<PrefixCache> b(&x.f.pc);
    b.flush_fn = [&]() { return !copy_now(x.t).empty(); };
    RestoreResult res = RestoreResult::Cut;
    (void)engine_restore(x, b, turn(3, s.a, {}), &res);
    CHECK(res == RestoreResult::Whole);
    const CacheHit hit = x.f.pc.lookup(s.a, {});
    CHECK_EQ(hit.n_attn_tokens, (int32_t)(s.n * 16));
    CHECK_EQ(hit.n_linear_tokens, (int32_t)(s.n * 16));
}

/* When the pool cannot be made to hold the chain, what comes back ends at a snapshot: blocks past
 * the deepest snapshot the shorter prefix reaches would be restored for nothing. Making room
 * never takes the conversation's own blocks, which would only have to come back too, nor a chain
 * a running sequence reads, which frees nothing. */
TEST(a_restore_the_pool_cannot_hold_stops_at_a_snapshot_and_spares_its_own_chain) {
    TierFixture x;
    CHECK_OK(x.build(/*with_linear=*/true, /*ckpt_slots=*/32));
    x.f.pc.set_policy(std::make_unique<GeometricBackoff>(64));
    const TwoSessions s = two_sessions(x, /*b_live=*/true, /*a_snapshots=*/true);
    const int64_t off = (int64_t)x.f.pc.restorable_from(s.a, {}).size();
    const int64_t resident = s.n - off;             /* A's first blocks, still in VRAM */
    const int64_t room = x.f.kv.total_blocks(G_FULL) - x.f.kv.held_blocks(G_FULL);
    REQUIRE(resident > 0);
    REQUIRE(room < off);
    /* The deepest snapshot the prefix reaches with `room` more blocks. */
    int64_t snap = 0;
    for (int64_t p : x.f.pc.checkpoints(1))
        if (p <= (resident + room) * 16) snap = p;
    REQUIRE(snap > 0);
    const int64_t drops = x.f.pc.stats().drops;

    PCBinding<PrefixCache> b(&x.f.pc);
    b.flush_fn = [&]() { return !copy_now(x.t).empty(); };
    RestoreResult res = RestoreResult::Whole;
    const int64_t back = engine_restore(x, b, turn(3, s.a, {}), &res);
    CHECK(res == RestoreResult::Cut);
    CHECK_EQ(back, std::max<int64_t>(0, snap / 16 - resident));
    CHECK_EQ(x.f.pc.stats().drops, drops);          /* no VRAM copy given up: A's own, or B's */
    const CacheHit hit = x.f.pc.lookup(s.a, {});
    CHECK_EQ(hit.n_linear_tokens, (int32_t)snap);
    CHECK_EQ(hit.n_attn_tokens, (int32_t)std::max<int64_t>(snap, resident * 16));
    CHECK_EQ(x.f.pc.lookup(s.b, {}).n_attn_tokens, (int32_t)(s.n * 16));
}

/* A chain another waiting request holds is kept too: that request restored it and was refused,
 * and taking it would only have its next attempt restore it again. */
TEST(a_restore_does_not_take_a_chain_a_waiting_request_holds) {
    TierFixture x;
    CHECK_OK(x.build());
    const TwoSessions s = two_sessions(x, /*b_live=*/false, /*a_snapshots=*/false);
    copy_now(x.t);                                  /* B is clean, so a trim could take it */
    x.f.pc.hold(9, s.b, {});
    const int64_t room = x.f.kv.total_blocks(G_FULL) - x.f.kv.held_blocks(G_FULL);
    const int64_t drops = x.f.pc.stats().drops;

    PCBinding<PrefixCache> b(&x.f.pc);
    RestoreResult res = RestoreResult::Whole;
    CHECK_EQ(engine_restore(x, b, turn(3, s.a, {}), &res), room);
    CHECK(res == RestoreResult::Cut);
    CHECK_EQ(x.f.pc.stats().drops, drops);
    CHECK_EQ(x.f.pc.lookup(s.b, {}).n_attn_tokens, (int32_t)(s.n * 16));

    x.f.pc.release(9);
    const int64_t rest = (int64_t)x.f.pc.restorable_from(s.a, {}).size();
    CHECK_EQ(engine_restore(x, b, turn(3, s.a, {}), &res), rest);
    CHECK(res == RestoreResult::Whole);
    CHECK_EQ(x.f.pc.lookup(s.a, {}).n_attn_tokens, (int32_t)(s.n * 16));
}

/* A restore asked to come back whole either does, or brings back nothing and is declined: the
 * admission waits a step and asks again (Scheduler::admit_one). */
TEST(a_whole_only_restore_comes_back_whole_or_not_at_all) {
    TierFixture x;
    CHECK_OK(x.build(/*with_linear=*/true, /*ckpt_slots=*/32));
    x.f.pc.set_policy(std::make_unique<GeometricBackoff>(64));
    const TwoSessions s = two_sessions(x, /*b_live=*/true, /*a_snapshots=*/true);
    const int64_t off = (int64_t)x.f.pc.restorable_from(s.a, {}).size();
    PCBinding<PrefixCache> b(&x.f.pc);
    b.flush_fn = [&]() { return !copy_now(x.t).empty(); };
    RestoreResult res = RestoreResult::Whole;
    CHECK(b.restore_plan(turn(3, s.a, {}), 0, /*whole_only=*/true, &res).empty());
    CHECK(res == RestoreResult::Declined);
    CHECK_EQ((int64_t)x.f.pc.restorable_from(s.a, {}).size(), off);   /* nothing moved */

    x.f.kv.free_sequence(2);                         /* B's sequence ends: its VRAM can go */
    CHECK_EQ((int64_t)b.restore_plan(turn(3, s.a, {}), 0, true, &res).size(), off);
    CHECK(res == RestoreResult::Whole);
}

/* Making room for a restore never erases an entry. Without a host tier, or with one attached but
 * off, a trim would erase, since no copy will ever land; so it takes nothing. */
TEST(making_room_for_a_restore_without_a_host_tier_takes_nothing) {
    Fixture f;
    CHECK_OK(f.build(/*with_linear=*/false, /*ckpt_slots=*/0));
    publish(f, 1, tok_run(7600, 336), 320);         /* twenty 16-token blocks */
    f.kv.free_sequence(1);                          /* finished: only the cache holds them */
    const auto other = tok_run(900000, 336);
    CHECK_EQ(f.pc.trim_for(other, {}, 20), 0ll);
    CHECK_EQ(f.pc.size(), 20ll);
    CHECK_EQ(f.pc.resident(), 20ll);

    IdleTiers off;
    CHECK_OK(off.configure(tier_config(0, 1024)));
    REQUIRE(!off.host_on());
    f.pc.set_tiers(&off);
    CHECK_EQ(f.pc.trim_for(other, {}, 20), 0ll);
    CHECK_EQ(f.pc.size(), 20ll);
    CHECK_EQ(f.pc.stats().evictions, 0ll);
    f.pc.set_tiers(nullptr);
}

/* Tools and tests build and tear down the tier executors and pools repeatedly, and on a card every
 * event is a hardware resource and every committed byte VRAM. A two-rank transfer pass gives back
 * every event and device byte it took. */
TEST(a_tier_transfer_gives_back_every_event_and_device_byte) {
    if (!host_backend_or_skip("the tier transfer")) return;
    const int64_t events0 = rad_dev_live_events();
    int64_t vram0 = 0, vram1 = 0;
    CHECK_OK(dev_vram_used(0, &vram0));
    run_transfer({ 2, /*fragment=*/true });
    CHECK_EQ(rad_dev_live_events(), events0);
    CHECK_OK(dev_vram_used(0, &vram1));
    CHECK_EQ(vram1, vram0);
}

/* The streamed group's residency maps are reserved from the KV pool before the carve, so the
 * budget probe must charge them before it sizes the pool. Left out, the pool it names is short by
 * their bytes and the group that stays resident is carved below one full-length conversation --
 * the target shape's `kv_qsa_bkey` at 47 185 of the 51 200 blocks it needed (188 740 tokens
 * against --max-model-len 204800). The probe has no pool and cannot reserve them; the carve has
 * already taken them. paged_split() charges them and carve_into() adds back what was reserved, so
 * both paths see the same bytes (kvstream/carve.h). */
TEST(a_streamed_groups_maps_are_charged_before_the_pool_is_sized) {
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    f.cfg.max_seqs = 4;
    f.cfg.max_ctx = 65536;                         /* the maps are more than the MiB rounding */
    std::vector<KVGroupInfo> groups = hybrid_groups();
    groups.push_back(paged_group("bkey", RAD_KV_FULL, 2, 4, { 6, 7 }));
    CHECK_OK(f.kv.plan_groups(groups, f.cfg));
    f.kv.cap_resident_tokens(G_FULL, 64);          /* four blocks of 16 tokens */
    const int64_t need = f.kv.pool_mib_for_tokens(f.cfg.max_ctx, 256);
    REQUIRE(need > 0);
    f.cfg.vram_kv_mib = need;
    CHECK_OK(configure_pools(f.pools, f.cfg));
    CHECK_OK(f.kv.alloc_stream(rad_malloc_allocator(), f.pools.kv()));
    CHECK_OK(f.kv.carve(f.pools.kv()));
    /* The group that stays resident holds one whole conversation, not the pool's remainder. */
    const KVGroupPlan* bkey = f.kv.plan(3);
    REQUIRE(bkey != nullptr);
    CHECK(!bkey->streamed());
    CHECK(bkey->n_blocks * bkey->block_size >= f.cfg.max_ctx);
}

/* A dense-read group is read over its whole live history every decode step, so it cannot stream.
 * While a sibling does, its carve is a floor of max_seqs full contexts: the host mirror next to it
 * is sized for max_seqs sequences, and an indexer funded for one makes the advertised capacity a
 * promise the engine cannot keep. The probe sizes the pool for the floor before the carve spends
 * it (kvstream/carve.h). */
TEST(a_dense_read_groups_floor_holds_every_sequence_while_a_sibling_streams) {
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    f.cfg.max_seqs = 4;
    f.cfg.max_ctx = 65536;
    std::vector<KVGroupInfo> groups = hybrid_groups();
    groups.push_back(paged_group("bkey", RAD_KV_FULL, 2, 4, { 6, 7 }));
    groups.back().dense_read = true;
    CHECK_OK(f.kv.plan_groups(groups, f.cfg));
    f.kv.cap_resident_tokens(G_FULL, 64);          /* four blocks of 16 tokens */
    const int64_t need = f.kv.pool_mib_for_tokens(f.cfg.max_ctx, 512);
    REQUIRE(need > 0);
    f.cfg.vram_kv_mib = need;
    CHECK_OK(configure_pools(f.pools, f.cfg));
    CHECK_OK(f.kv.alloc_stream(rad_malloc_allocator(), f.pools.kv()));
    CHECK_OK(f.kv.carve(f.pools.kv()));
    const KVGroupPlan* bkey = f.kv.plan(3);
    REQUIRE(bkey != nullptr);
    CHECK(!bkey->streamed());
    CHECK_EQ(bkey->n_blocks, f.cfg.max_seqs * ((f.cfg.max_ctx + 3) / 4));
    CHECK_EQ(bkey->n_slots, bkey->n_blocks);
}

/* A prefix entry covers every cached group, the streamed one included: a hit re-adopts its block
 * ids, and the mirror behind them is that group's tier. A streamed group's rows are addressed by
 * the block id the sequence used, so its ids cannot be given up for a host slot: an entry that
 * contains one stays in VRAM, the tiers are never told about it, and an eviction takes every
 * group's ids with it. */
TEST(an_entry_with_a_streamed_group_is_never_demoted) {
    if (!host_backend_or_skip("the tier restore")) return;
    Fixture f;
    f.cfg.checkpoint_slots = 0;
    f.cfg.max_seqs = 1;
    f.cfg.max_ctx = 1024;
    std::vector<KVGroupInfo> groups = hybrid_groups();
    groups.push_back(paged_group("bkey", RAD_KV_FULL, 2, 16, { 6, 7 }));
    CHECK_OK(f.kv.plan_groups(groups, f.cfg));
    f.kv.cap_resident_tokens(G_FULL, 64);              /* four blocks of 16 tokens */
    const int64_t need = f.kv.pool_mib_for_tokens(f.cfg.max_ctx, 256);
    REQUIRE(need > 0);
    f.cfg.vram_kv_mib = need;
    CHECK_OK(configure_pools(f.pools, f.cfg));
    CHECK_OK(f.kv.alloc_stream(rad_malloc_allocator(), f.pools.kv()));
    CHECK_OK(f.kv.carve(f.pools.kv()));

    const std::vector<int32_t>& cg = f.pc.cached_groups();
    IdleTiers t;
    REQUIRE_EQ(t.configure(tier_config(4096 * 1024, 4096 * 1024)), RAD_OK);
    f.pc.set_tiers(&t);
    REQUIRE_EQ(f.pc.configure(f.cfg, &f.kv, f.cfg.checkpoint_interval), RAD_OK);
    REQUIRE_EQ(cg.size(), 2u);
    CHECK(f.kv.plan(cg[0])->streamed());
    /* The one fact the tier path depends on: a slot cannot hold this cache's entries, so no
     * write-back is planned, nothing is queued, and nothing can be dropped to the host. */
    CHECK(!f.pc.demotable());

    const auto tokens = tok_run(5000, 64);
    publish(f, 1, tokens, 64);
    f.kv.free_sequence(1);
    const std::vector<BlockHash> hs = chain_of(tokens);
    REQUIRE_EQ(hs.size(), 4u);

    std::vector<TierJob> wb;
    t.plan_writeback(64, &wb);
    CHECK_EQ(wb.size(), 0u);                           /* the streamed entry is never copied */
    CHECK_EQ(f.pc.drop_clean(), 0ll);                  /* so it is never demoted */
    CHECK(!t.droppable(hs[0]));

    /* The trim still frees it: an entry the tier cannot hold must not be skipped as though its
     * copy were in flight. All four blocks are at refcount one and the trim takes them. */
    CHECK_EQ(f.pc.trim(4), 4ll);
    CHECK_EQ(f.pc.size(), 0ll);
    CHECK_EQ(f.kv.block_refcount(cg[0], 0), 0);
    CHECK_EQ(f.kv.block_refcount(cg[1], 0), 0);
    f.pc.set_tiers(nullptr);
}

/* ------------------------------------------------------------------ window residency (spec §7.3) */

/* QSA's shape in miniature: one group whose consumer reads a window of the history (`kv_attn`)
 * and one whose consumer reads all of it (`kv_qsa_bkey`), both paged at 16 tokens. */
enum : int32_t { G_RES = 1, G_KEYS = 2 };

static std::vector<KVGroupInfo> residency_groups(int64_t window_tokens) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    KVGroupInfo k = paged_group("attn", RAD_KV_FULL, 2, 16, { 0, 1 });
    k.decl.residency_window = window_tokens;
    g.push_back(k);                                    /* page = 2048 B */
    g.push_back(paged_group("keys", RAD_KV_FULL, 1, 16, { 2, 3 }));
    return g;
}

/* The engine opens the store after the KV configure, because a slot is one block and the block
 * count is only known then. The fixture does the same. */
struct ResidencyFixture {
    Pools pools;
    HostArena store;
    KVManager kv;
    Config cfg = small_config();

    int build(int64_t window_tokens, int64_t slots) {
        return build_groups(residency_groups(window_tokens), slots);
    }
    int build_groups(const std::vector<KVGroupInfo>& groups, int64_t slots) {
        int s = configure_pools(pools, cfg);
        if (s < 0) return s;
        s = kv.configure(groups, cfg, pools.kv());
        if (s < 0) return s;
        if (kv.residency_slot_bytes() <= 0) return RAD_E_STATE;
        s = store.open(kv.residency_slot_bytes(), kv.residency_slot_bytes() * slots);
        if (s < 0) return s;
        kv.set_residency_store(&store, 0);
        return RAD_OK;
    }
};

/* The one test that would fail if nothing ever moved: a server that keeps every block resident
 * passes every other check here and saves no VRAM at all. */
TEST(a_residency_window_keeps_both_ends_and_moves_only_what_lies_between_them) {
    ResidencyFixture f;
    CHECK_OK(f.build(/*window_tokens=*/512, /*slots=*/256));
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));                 /* 40 blocks, 640 tokens */
    const std::vector<int32_t> before = f.kv.block_table(1, G_RES);
    CHECK_EQ((int64_t)before.size(), 40);

    const int64_t free_before = f.kv.free_blocks(G_RES);
    std::vector<BlockMove> moves;
    CHECK_OK(f.kv.residency_demote(1, G_RES, /*keep_head=*/2, /*keep_tail=*/8, &moves));
    CHECK_EQ((int64_t)moves.size(), 30);
    CHECK(f.kv.residency_stats().demoted == 30);
    CHECK(f.kv.residency_stats().demoted_bytes == 30 * 2048);

    /* The sequence's context is untouched: same positions, same token count. */
    const std::vector<int32_t> after = f.kv.block_table(1, G_RES);
    CHECK_EQ((int64_t)after.size(), 40);
    CHECK_EQ(f.kv.seq_tokens(1), 640ll);

    /* The sinks and the tail stayed, the middle moved, and nothing else did. */
    for (int64_t i = 0; i < 2; ++i) CHECK_EQ(after[(size_t)i], before[(size_t)i]);
    for (int64_t i = 2; i < 32; ++i) CHECK(f.kv.residency_cold(G_RES, after[(size_t)i]));
    for (int64_t i = 32; i < 40; ++i) CHECK_EQ(after[(size_t)i], before[(size_t)i]);

    /* The VRAM comes back only when the copy has landed: the gather still reads the blocks. */
    CHECK_EQ(f.kv.free_blocks(G_RES), free_before);
    f.kv.residency_demote_landed();
    CHECK_EQ(f.kv.free_blocks(G_RES), free_before + 30);
    CHECK_EQ(f.kv.residency_stats().freed_blocks, 30ll);
    CHECK_EQ(f.kv.residency_stats().freed_bytes, 30ll * 2048);

    /* Every move names a distinct slot inside the store, the entry is the cold encoding of that
     * slot, and the pointer is that block's. */
    std::set<int32_t> slots;
    for (size_t k = 0; k < moves.size(); ++k) {
        const BlockMove& m = moves[k];
        CHECK_EQ(m.group, G_RES);
        CHECK(m.host_slot >= 0);
        CHECK_EQ(m.block, before[2 + k]);
        CHECK_EQ(after[2 + k], rad_kv_cold_entry(m.host_slot));
        slots.insert(m.host_slot);
        CHECK(f.kv.residency_ptr(G_RES, after[2 + k]) != nullptr);
        CHECK_EQ(f.kv.residency_slot(G_RES, after[2 + k]), m.host_slot);
    }
    CHECK_EQ((int64_t)slots.size(), 30);
    CHECK_EQ(f.kv.residency_host_bytes(), 30 * f.kv.residency_slot_bytes());

    /* The group that reads all of its history is not part of this. */
    CHECK_EQ((int64_t)f.kv.residency_cold_entries(1, G_KEYS).size(), 0ll);

    /* A second sweep with the same window moves nothing: the state is not recomputed. */
    std::vector<BlockMove> again;
    CHECK_OK(f.kv.residency_demote(1, G_RES, 2, 8, &again));
    CHECK(again.empty());
    CHECK(f.kv.residency_stats().demoted == 30);
}

/* The per-session numbers /sessions reports: each sequence gets its own demoted, promoted, link
 * bytes and freed VRAM, they move with the sweeps, and they survive the sequence's end because
 * the session is listed after its request is gone. */
TEST(residency_counters_are_kept_per_sequence_and_outlive_the_sequence) {
    ResidencyFixture f;
    CHECK_OK(f.build(512, 256));
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.add_sequence(2));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    CHECK_OK(f.kv.ensure(2, 16 * 20));
    CHECK(f.kv.residency_by_seq().empty());
    std::vector<BlockMove> m1, m2;
    CHECK_OK(f.kv.residency_demote(1, G_RES, 2, 8, &m1));
    CHECK_OK(f.kv.residency_demote(2, G_RES, 2, 8, &m2));
    f.kv.residency_demote_landed();
    std::vector<BlockMove> up;
    CHECK_OK(f.kv.residency_promote(1, G_RES, { 10, 11 }, &up));

    const auto& by = f.kv.residency_by_seq();
    CHECK_EQ(by.at(1).demoted, 30ll);
    CHECK_EQ(by.at(2).demoted, 10ll);
    CHECK_EQ(by.at(1).promoted, 2ll);
    CHECK_EQ(by.at(2).promoted, 0ll);
    CHECK_EQ(by.at(1).link_bytes, 32ll * 2048);
    CHECK_EQ(by.at(1).freed_bytes, 30ll * 2048);
    CHECK_EQ(by.at(2).freed_bytes, 10ll * 2048);
    CHECK_EQ(f.kv.residency_stats().demoted_bytes + f.kv.residency_stats().promoted_bytes,
             by.at(1).link_bytes + by.at(2).link_bytes);

    f.kv.free_sequence(1);
    CHECK_EQ(f.kv.residency_by_seq().at(1).demoted, 30ll);
}

TEST(a_promotion_returns_the_named_entries_and_charges_the_link_traffic) {
    ResidencyFixture f;
    CHECK_OK(f.build(512, 256));
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    std::vector<BlockMove> moves;
    CHECK_OK(f.kv.residency_demote(1, G_RES, 2, 8, &moves));
    CHECK_EQ((int64_t)moves.size(), 30);
    f.kv.residency_demote_landed();
    const std::vector<int32_t> cold_tbl = f.kv.block_table(1, G_RES);

    /* What a step that selects those blocks asks for: table entries, not block ids. */
    std::vector<BlockMove> up;
    CHECK_OK(f.kv.residency_promote(1, G_RES, { 10, 11, 12 }, &up));
    CHECK_EQ((int64_t)up.size(), 3);
    for (size_t k = 0; k < up.size(); ++k) {
        const BlockMove& m = up[k];
        /* A fresh pool block is the entry now, and the slot to read is the one it replaced. */
        CHECK_EQ(f.kv.block_table(1, G_RES)[10 + k], m.block);
        CHECK(m.block >= 0);
        CHECK_EQ(m.host_slot, rad_kv_entry_slot(cold_tbl[10 + k]));
        CHECK(!f.kv.residency_cold(G_RES, m.block));
        CHECK(f.kv.residency_ptr(G_RES, m.block) == nullptr);
    }
    CHECK(f.kv.residency_stats().promoted == 3);
    CHECK(f.kv.residency_stats().promoted_misses == 3);
    CHECK(f.kv.residency_stats().promoted_bytes == 3 * 2048);

    /* Resident entries are skipped, so a caller may pass a superset of what it needs. */
    std::vector<BlockMove> again;
    CHECK_OK(f.kv.residency_promote(1, G_RES, { 10, 11, 12 }, &again));
    CHECK(again.empty());

    /* A demoted block promoted and then swept again moves back: the window is a state, not a
     * one-way door. */
    std::vector<BlockMove> down;
    CHECK_OK(f.kv.residency_demote(1, G_RES, 2, 8, &down));
    CHECK_EQ((int64_t)down.size(), 3);
    CHECK(f.kv.residency_cold(G_RES, f.kv.block_table(1, G_RES)[10]));

    /* An index outside the table names no block of this sequence: served, it would be another
     * conversation's context, so it is refused. */
    std::vector<BlockMove> bad;
    CHECK_EQ(f.kv.residency_promote(1, G_RES, { 40 }, &bad), RAD_E_INVAL);
    CHECK_EQ(f.kv.residency_promote(1, G_RES, { -1 }, &bad), RAD_E_INVAL);
    CHECK(bad.empty());
}

TEST(the_partial_last_block_and_a_shared_block_never_leave_vram) {
    ResidencyFixture f;
    CHECK_OK(f.build(512, 256));
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    const std::vector<int32_t> before = f.kv.block_table(1, G_RES);

    /* keep_tail 0 still keeps the last block: it is partial, and the next token's write would
     * land in host memory. A moved block keeps its refcount, so its refcount is unchanged. */
    std::vector<BlockMove> moves;
    CHECK_OK(f.kv.residency_demote(1, G_RES, /*keep_head=*/0, /*keep_tail=*/0, &moves));
    CHECK_EQ((int64_t)moves.size(), 39);
    CHECK_EQ(f.kv.block_table(1, G_RES)[39], before[39]);
    CHECK_EQ(f.kv.block_refcount(G_RES, before[39]), 1);

    /* A block the prefix cache indexes is read by whoever hits it, and this sequence cannot
     * promote it for that reader: it stays, and the skip is counted. */
    ResidencyFixture g2;
    CHECK_OK(g2.build(512, 256));
    CHECK_OK(g2.kv.add_sequence(7));
    CHECK_OK(g2.kv.ensure(7, 16 * 40));
    const std::vector<int32_t> b2 = g2.kv.block_table(7, G_RES);
    CHECK_OK(g2.kv.retain_block(G_RES, b2[20]));
    std::vector<BlockMove> m2;
    CHECK_OK(g2.kv.residency_demote(7, G_RES, 0, 0, &m2));
    CHECK_EQ(g2.kv.block_table(7, G_RES)[20], b2[20]);
    CHECK(g2.kv.residency_stats().demote_skipped >= 1);
    CHECK_EQ((int64_t)m2.size(), 38);
}

TEST(a_demoted_block_gives_its_slot_back_when_its_last_reference_goes) {
    ResidencyFixture f;
    CHECK_OK(f.build(512, 256));
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    std::vector<BlockMove> moves;
    CHECK_OK(f.kv.residency_demote(1, G_RES, 2, 8, &moves));
    CHECK_EQ(f.store.used_slots(), 30ll);
    const int64_t total = f.kv.free_blocks(G_RES) + 40;    /* 40 held by the sequence, 30 pinned */

    /* A preemption replays the sequence from zero, so its entries go -- the cold ones give their
     * slots back and the resident ones their blocks; the 30 blocks a landing has not yet released
     * stay pinned until it reports. */
    CHECK_OK(f.kv.preempt(1));
    CHECK_EQ(f.store.used_slots(), 0ll);
    f.kv.residency_demote_landed();
    CHECK_EQ(f.kv.free_blocks(G_RES), total);

    /* The same for a sequence freed outright. */
    CHECK_OK(f.kv.add_sequence(2));
    CHECK_OK(f.kv.ensure(2, 16 * 40));
    std::vector<BlockMove> m2;
    CHECK_OK(f.kv.residency_demote(2, G_RES, 2, 8, &m2));
    CHECK_EQ(f.store.used_slots(), 30ll);
    f.kv.residency_demote_landed();
    f.kv.free_sequence(2);
    CHECK_EQ(f.store.used_slots(), 0ll);
    CHECK_EQ(f.kv.free_blocks(G_RES), total);
}

TEST(a_full_store_leaves_the_rest_resident_and_reports_the_skips) {
    ResidencyFixture f;
    CHECK_OK(f.build(512, /*slots=*/4));
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    std::vector<BlockMove> moves;
    CHECK_OK(f.kv.residency_demote(1, G_RES, 2, 8, &moves));
    CHECK_EQ((int64_t)moves.size(), 4);                /* the store's whole capacity */
    CHECK(f.kv.residency_stats().demote_skipped >= 26);
    CHECK_EQ((int64_t)f.kv.residency_cold_entries(1, G_RES).size(), 4ll);
}

TEST(a_residency_window_without_a_store_refuses_rather_than_moving_nothing) {
    ResidencyFixture f;
    CHECK_OK(f.build(512, 256));
    f.kv.set_residency_store(nullptr, 0);              /* --qsa-window-mib off */
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    std::vector<BlockMove> moves;
    CHECK_EQ(f.kv.residency_demote(1, G_RES, 2, 8, &moves), RAD_E_STATE);
    CHECK(moves.empty());
}

TEST(a_residency_window_is_refused_on_a_group_that_recycles_its_blocks) {
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    KVGroupInfo k = paged_group("swa", RAD_KV_WINDOW, 4, 16, { 0, 1 }, /*window=*/2048);
    k.decl.residency_window = 512;                     /* both mechanisms on one table */
    g.push_back(k);
    Pools pools;
    Config c = small_config();
    CHECK_EQ(configure_pools(pools, c), RAD_OK);
    KVManager kv;
    CHECK_EQ(kv.plan_groups(g, c), RAD_E_INVAL);
}

/* The window is a property of the group, so a plan that drops it drops its state too: a re-plan
 * must not leak slots from a manager that keeps the same store. */
TEST(a_re_plan_returns_the_slots_the_previous_one_held) {
    ResidencyFixture f;
    CHECK_OK(f.build(512, 256));
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    std::vector<BlockMove> moves;
    CHECK_OK(f.kv.residency_demote(1, G_RES, 2, 8, &moves));
    CHECK_EQ(f.store.used_slots(), 30ll);
    CHECK_OK(f.kv.plan_groups(residency_groups(512), f.cfg));
    CHECK_EQ(f.store.used_slots(), 0ll);
}

/* ------------------------------------------------------------------ window residency: the transfer */

/* A block's bytes as the pool holds them, layer by layer: layer l of block b sits at
 * base + l*layer_stride + b*frag, so a test that compared the whole region would pass with the
 * layers of a block in any order. */
static unsigned char resid_pat(int64_t layer, int32_t block, int64_t i) {
    return (unsigned char)((layer * 131 + block * 17 + i * 7 + 3) & 0xff);
}

static void resid_fill(KVManager& kv, int32_t g, int64_t salt = 0) {
    const KVGroupPlan* p = kv.plan(g);
    auto* base = (char*)p->base;
    const int64_t frag = p->layer_bytes_per_block();
    for (int64_t b = 0; b < p->n_blocks; ++b)
        for (int64_t l = 0; l < p->n_layers; ++l)
            for (int64_t i = 0; i < frag; ++i)
                base[l * p->layer_stride() + b * frag + i] = (char)resid_pat(l + salt, (int32_t)b, i);
}

/* How many (layer, byte) positions of a block differ from what resid_fill wrote for `pat_block`
 * (the block itself unless a promotion put its bytes in a different one), under the same `salt`
 * (a per-group one tells two groups' blocks apart). */
static int64_t resid_mismatches(KVManager& kv, int32_t g, int32_t block, int32_t pat_block = -1,
                                int64_t salt = 0) {
    if (pat_block < 0) pat_block = block;
    const KVGroupPlan* p = kv.plan(g);
    auto* base = (char*)p->base;
    const int64_t frag = p->layer_bytes_per_block();
    int64_t bad = 0;
    for (int64_t l = 0; l < p->n_layers; ++l)
        for (int64_t i = 0; i < frag; ++i)
            if ((unsigned char)base[l * p->layer_stride() + block * frag + i] !=
                resid_pat(l + salt, pat_block, i))
                ++bad;
    return bad;
}

static void resid_scribble(KVManager& kv, int32_t g) {
    const KVGroupPlan* p = kv.plan(g);
    std::memset(p->base, 0x5a, (size_t)(p->n_blocks * p->bytes_per_block));
}

/* The check the idle tiers get from scripts/tierident.sh, at the block: a demoted block comes back
 * byte for byte after the pool has been overwritten, and nothing the promotion did not name moved.
 * Corrupt KV reads as fluent wrong text, so a count of mismatching bytes is the only honest check. */
TEST(a_demoted_block_comes_back_byte_for_byte_after_the_pool_is_overwritten) {
    if (!device_is_host()) {
        std::printf("  SKIP  the residency transfer needs the host backend (%s here)\n",
                    device_backend_name());
        return;
    }
    Registry reg;
    REQUIRE_EQ(reg.load_plugin(rows_plugin(), 0), RAD_OK);
    CHECK_OK(reg.check_complete());

    ResidencyFixture f;
    CHECK_OK(f.build(/*window_tokens=*/512, /*slots=*/64));
    /* The tier exec wants a tier object for the entry-shaped path it also serves; this one has no
     * host tier and is not used by a residency move. */
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(0, 0)));
    RadStream compute = nullptr;
    REQUIRE_EQ(rad_stream_create(&compute, 0), RAD_OK);
    TierExec x;
    const std::vector<int32_t> groups{ G_RES };
    CHECK_OK(x.configure(&reg, &f.kv, &t, groups, TierExec::kMoveBatch, 64, 64,
                         /*leader=*/true, /*device=*/0, /*slot_off=*/0, /*shared=*/nullptr));

    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    resid_fill(f.kv, G_RES);
    const std::vector<int32_t> tbl = f.kv.block_table(1, G_RES);
    REQUIRE_EQ((int64_t)tbl.size(), 40ll);

    std::vector<BlockMove> down;
    CHECK_OK(f.kv.residency_demote(1, G_RES, /*keep_head=*/2, /*keep_tail=*/8, &down));
    REQUIRE_EQ(down.size(), 30u);
    for (size_t i = 0; i < down.size(); i += TierExec::kMoveBatch) {
        const std::vector<BlockMove> batch(down.begin() + (long)i,
                                           down.begin() + (long)std::min(down.size(), i + TierExec::kMoveBatch));
        CHECK_OK(x.issue_moves(batch, compute, /*to_host=*/true));
        x.quiesce();
    }
    f.kv.residency_demote_landed();      /* the copies landed: the pool may reuse the 30 blocks */
    CHECK_EQ(x.move_stats().out_blocks, 30ll);

    /* What the pool does to VRAM once the blocks have left: another owner's bytes. */
    resid_scribble(f.kv, G_RES);
    /* Every byte of both layers: the pattern is unique per (layer, byte), so a mismatch count near
     * the block size is what a scribbled block gives; a handful of coincidences (0x5a) are fine. */
    CHECK(resid_mismatches(f.kv, G_RES, tbl[10]) > 2000);

    std::vector<BlockMove> up;
    CHECK_OK(f.kv.residency_promote(1, G_RES, { 10, 11, 12 }, &up));
    REQUIRE_EQ(up.size(), 3u);
    CHECK_OK(x.issue_moves(up, compute, /*to_host=*/false));
    x.quiesce();
    f.kv.residency_complete();       /* the slots the promotion held can go back */
    CHECK_EQ(x.move_stats().in_blocks, 3ll);

    /* The bytes sit in a fresh pool block, whatever id it got, and are the original block's. */
    for (int64_t e : { 10, 11, 12 }) {
        const int32_t now = f.kv.block_table(1, G_RES)[(size_t)e];
        CHECK(now >= 0);
        CHECK_EQ(resid_mismatches(f.kv, G_RES, now, tbl[(size_t)e]), 0ll);
    }
    /* Nothing else was brought back: a promotion that moved the whole sequence would pass the
     * check above and hide a window that is not doing anything. */
    CHECK(f.kv.residency_cold(G_RES, f.kv.block_table(1, G_RES)[20]));
    CHECK_EQ(f.store.used_slots(), 27ll);

    CHECK_OK(rad_stream_sync(compute));
    rad_stream_destroy(compute);
}

/* ------------------------------------------------------------------ window residency: cold entries */

/* Demotes entries [2, 32) of a 40-block sequence and lands them, as a sweep does. */
static void resid_go_cold(ResidencyFixture& f, uint64_t seq) {
    std::vector<BlockMove> moves;
    CHECK_OK(f.kv.residency_demote(seq, G_RES, 2, 8, &moves));
    f.kv.residency_demote_landed();
}

/* The acceptance check for the whole feature: the VRAM a demotion frees is really handed to
 * someone else, and the owner still reads its own bytes after promotion. */
TEST(a_freed_block_id_serves_another_sequence_while_its_owner_comes_back_byte_for_byte) {
    if (!device_is_host()) {
        std::printf("  SKIP  the residency transfer needs the host backend (%s here)\n",
                    device_backend_name());
        return;
    }
    Registry reg;
    REQUIRE_EQ(reg.load_plugin(rows_plugin(), 0), RAD_OK);
    CHECK_OK(reg.check_complete());
    ResidencyFixture f;
    CHECK_OK(f.build(512, 64));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(0, 0)));
    RadStream compute = nullptr;
    REQUIRE_EQ(rad_stream_create(&compute, 0), RAD_OK);
    TierExec x;
    const std::vector<int32_t> groups{ G_RES };
    CHECK_OK(x.configure(&reg, &f.kv, &t, groups, TierExec::kMoveBatch, 64, 64,
                         /*leader=*/true, /*device=*/0, /*slot_off=*/0, /*shared=*/nullptr));

    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    resid_fill(f.kv, G_RES);
    const std::vector<int32_t> orig = f.kv.block_table(1, G_RES);

    std::vector<BlockMove> down;
    CHECK_OK(f.kv.residency_demote(1, G_RES, 2, 8, &down));
    REQUIRE_EQ(down.size(), 30u);
    for (size_t i = 0; i < down.size(); i += TierExec::kMoveBatch) {
        const std::vector<BlockMove> batch(down.begin() + (long)i,
                                           down.begin() + (long)std::min(down.size(), i + TierExec::kMoveBatch));
        CHECK_OK(x.issue_moves(batch, compute, /*to_host=*/true));
        x.quiesce();
    }
    f.kv.residency_demote_landed();

    /* Another sequence takes the pool, so it holds freed ids; it overwrites them. */
    CHECK_OK(f.kv.add_sequence(2));
    CHECK_OK(f.kv.ensure(2, 16 * 30));
    const std::vector<int32_t> other = f.kv.block_table(2, G_RES);
    std::set<int32_t> freed;
    for (size_t i = 2; i < 32; ++i) freed.insert(orig[i]);
    int64_t reused = 0;
    for (int32_t b : other) reused += freed.count(b);
    CHECK(reused > 0);                                  /* a freed id really changed hands */
    resid_scribble(f.kv, G_RES);

    /* The owner promotes everything and reads what it wrote. */
    std::vector<BlockMove> up;
    CHECK_OK(f.kv.residency_promote_all(1, &up));
    REQUIRE_EQ(up.size(), 30u);
    for (size_t i = 0; i < up.size(); i += TierExec::kMoveBatch) {
        const std::vector<BlockMove> batch(up.begin() + (long)i,
                                           up.begin() + (long)std::min(up.size(), i + TierExec::kMoveBatch));
        CHECK_OK(x.issue_moves(batch, compute, /*to_host=*/false));
        x.quiesce();
    }
    f.kv.residency_complete();
    const std::vector<int32_t> back = f.kv.block_table(1, G_RES);
    for (size_t i = 2; i < 32; ++i) {
        CHECK(back[i] >= 0);
        CHECK_EQ(resid_mismatches(f.kv, G_RES, back[i], orig[i]), 0ll);
    }
    /* Nobody holds a block twice: the owner's fresh blocks are disjoint from the other's. */
    for (int32_t b : back) for (int32_t o : other) CHECK(b != o);
    CHECK_EQ(f.store.used_slots(), 0ll);
    CHECK(f.kv.residency_cold_entries(1, G_RES).empty());
    CHECK_EQ(f.kv.residency_stats().promoted, 30ll);

    CHECK_OK(rad_stream_sync(compute));
    rad_stream_destroy(compute);
}

/* ------------------------------------------------------------------ window residency: a per-layer recipe */

/* Flash-Next under a mixed --kv-recipe, in miniature: the trunk's attention layers in one group per
 * layout, each declared at its row's bytes over two as E4M3 columns (32 at fp8, 22 at k6_v4 for
 * head_dim 32), both carrying the indexer's window, beside the block keys. One fp8 layer is 2048
 * bytes a block, each k6_v4 layer 1408. */
enum : int32_t { G_MIX_F8 = 1, G_MIX_K6 = 2, G_MIX_KEYS = 3 };

static std::vector<KVGroupInfo> mixed_residency_groups() {
    auto attn = [](const char* name, int64_t head_dim, std::vector<int32_t> layers) {
        KVGroupInfo k = paged_group(name, RAD_KV_FULL, 2, 16, std::move(layers));
        k.decl.dtype = RAD_F8E4M3;
        k.decl.head_dim = head_dim;
        k.decl.residency_window = 512;
        return k;
    };
    std::vector<KVGroupInfo> g;
    g.push_back(null_group());
    g.push_back(attn("kv_attn_fp8", 32, { 3 }));               /* 2048 B a block */
    g.push_back(attn("kv_attn_k6_v4", 22, { 7, 11, 15 }));     /* 3 x 1408 = 4224 B */
    g.push_back(paged_group("kv_qsa_bkey", RAD_KV_FULL, 1, 16, { 3, 7, 11, 15 }));
    return g;
}

/* A slot holds one block of one group -- a demotion takes a fresh slot for each group's entry --
 * so its size is the widest layout's block, not the sum of every layout's: the sum would leave
 * every slot of a two-layout recipe partly empty and the store holding fewer blocks than its
 * bytes buy. Each group's block starts at the rank's offset in the slot, its layers' fragments
 * one after the other at that layer's own width, which is what the pager's layer offsets name.
 * Under tensor parallelism the pitch the pager walks is the shared slot, every rank's window, and
 * not the part one rank would need alone. */
TEST(a_mixed_recipes_store_slot_is_its_widest_layouts_block_and_each_layer_its_own_fragment) {
    ResidencyFixture f;
    CHECK_OK(f.build_groups(mixed_residency_groups(), /*slots=*/16));
    const KVGroupPlan* f8 = f.kv.plan(G_MIX_F8);
    const KVGroupPlan* k6 = f.kv.plan(G_MIX_K6);
    REQUIRE(f8 && k6 && f8->residency() && k6->residency());
    CHECK_EQ(f8->bytes_per_block, 2048ll);
    CHECK_EQ(k6->bytes_per_block, 3 * 1408ll);
    CHECK_EQ(f.kv.residency_block_bytes(G_MIX_F8), 2048ll);
    CHECK_EQ(f.kv.residency_block_bytes(G_MIX_K6), 4224ll);
    CHECK_EQ(f.kv.residency_block_bytes(G_MIX_KEYS), 0ll);     /* no window: never in the store */
    CHECK_EQ(f.kv.residency_shard_bytes(), 4224ll);
    CHECK_EQ(f.kv.residency_slot_bytes(), 4352ll);             /* 4224 up to the 256-byte unit */
    CHECK_EQ(f.store.slot_bytes(), 4352ll);

    const std::vector<int64_t>& o8 = f.kv.residency_layer_off(G_MIX_F8);
    const std::vector<int64_t>& o6 = f.kv.residency_layer_off(G_MIX_K6);
    REQUIRE_EQ(o8.size(), (size_t)4);
    REQUIRE_EQ(o6.size(), (size_t)16);
    CHECK_EQ(o8[3], 0ll);
    CHECK_EQ(o8[0], -1ll); CHECK_EQ(o8[1], -1ll); CHECK_EQ(o8[2], -1ll);
    CHECK_EQ(o6[7], 0ll); CHECK_EQ(o6[11], 1408ll); CHECK_EQ(o6[15], 2816ll);
    CHECK_EQ(o6[3], -1ll);                                     /* layer 3 is the fp8 group's */
    CHECK(o6[15] + k6->layer_bytes_per_block() <= f.kv.residency_slot_bytes());

    /* Two ranks, the engine's way: the slot is the sum of the ranks' widest blocks, rank 1's window
     * starts after rank 0's, and both walk the store at the shared slot's pitch. */
    Pools p1;
    Config c1 = small_config();
    REQUIRE_EQ(configure_pools(p1, c1), RAD_OK);
    KVManager r1;
    REQUIRE_EQ(r1.configure(mixed_residency_groups(), c1, p1.kv()), RAD_OK);
    const int64_t shard = f.kv.residency_shard_bytes();
    HostArena shared;
    CHECK_OK(shared.open(2 * shard, 2 * shard * 8));
    CHECK_OK(f.kv.set_residency_store(&shared, 0));
    /* A store whose slot cannot hold a rank's window past its offset is refused, not overrun. */
    CHECK_EQ(r1.set_residency_store(&shared, shard + 1), (int)RAD_E_INVAL);
    CHECK(!r1.residency_enabled());
    CHECK_OK(r1.set_residency_store(&shared, shard));
    CHECK_EQ(f.kv.residency_slot_bytes(), 2 * shard);
    CHECK_EQ(r1.residency_slot_bytes(), 2 * shard);
    CHECK_EQ(r1.residency_layer_off(G_MIX_F8)[3], shard);
    CHECK_EQ(r1.residency_layer_off(G_MIX_K6)[15], shard + 2816);
    CHECK_OK(f.kv.set_residency_store(nullptr, 0));   /* `shared` goes before the managers do */
    CHECK_OK(r1.set_residency_store(nullptr, 0));
}

/* A recipe with bf16 layers (RAD_KVQ_BF16) beside i8h ones, in miniature: the bf16 group is declared
 * as --kv-cache-dtype bf16 declares its cache, RAD_BF16 at head_dim 32 (4096 bytes a layer and
 * block), the i8h one as 34 E4M3 columns (a 68-byte row, 2176 bytes). The bf16 block is now the
 * widest, so it sets the slot; each layer's fragment keeps its own width, the bf16 layers' at two
 * bytes an element, and a second rank's window starts past the first's whole bf16 block. */
TEST(a_bf16_recipe_layer_sizes_the_residency_slot_at_its_own_two_byte_fragment) {
    auto attn = [](const char* name, uint32_t dtype, int64_t head_dim, std::vector<int32_t> layers) {
        KVGroupInfo k = paged_group(name, RAD_KV_FULL, 2, 16, std::move(layers));
        k.decl.dtype = dtype;
        k.decl.head_dim = head_dim;
        k.decl.residency_window = 512;
        return k;
    };
    auto groups = [&] {
        std::vector<KVGroupInfo> g;
        g.push_back(null_group());
        g.push_back(attn("kv_attn_i8h", RAD_F8E4M3, 34, { 11, 15 }));
        g.push_back(attn("kv_attn_bf16", RAD_BF16, 32, { 3, 7 }));
        g.push_back(paged_group("kv_qsa_bkey", RAD_KV_FULL, 1, 16, { 3, 7, 11, 15 }));
        return g;
    };
    enum : int32_t { G_I8H = 1, G_BF16 = 2 };
    ResidencyFixture f;
    CHECK_OK(f.build_groups(groups(), /*slots=*/16));
    CHECK_EQ(f.kv.residency_block_bytes(G_I8H), 2 * 2176ll);
    CHECK_EQ(f.kv.residency_block_bytes(G_BF16), 2 * 4096ll);
    CHECK_EQ(f.kv.residency_shard_bytes(), 8192ll);
    CHECK_EQ(f.kv.residency_slot_bytes(), 8192ll);
    CHECK_EQ(f.store.slot_bytes(), 8192ll);
    const std::vector<int64_t>& ob = f.kv.residency_layer_off(G_BF16);
    const std::vector<int64_t>& oi = f.kv.residency_layer_off(G_I8H);
    REQUIRE_EQ(ob.size(), (size_t)8);
    REQUIRE_EQ(oi.size(), (size_t)16);
    CHECK_EQ(ob[3], 0ll); CHECK_EQ(ob[7], 4096ll); CHECK_EQ(ob[0], -1ll);
    CHECK_EQ(oi[11], 0ll); CHECK_EQ(oi[15], 2176ll); CHECK_EQ(oi[3], -1ll);

    Pools p1;
    Config c1 = small_config();
    REQUIRE_EQ(configure_pools(p1, c1), RAD_OK);
    KVManager r1;
    REQUIRE_EQ(r1.configure(groups(), c1, p1.kv()), RAD_OK);
    HostArena shared;
    CHECK_OK(shared.open(2 * 8192, 2 * 8192 * 8));
    CHECK_OK(f.kv.set_residency_store(&shared, 0));
    CHECK_EQ(r1.set_residency_store(&shared, 8192 + 1), (int)RAD_E_INVAL);
    CHECK_OK(r1.set_residency_store(&shared, 8192));
    CHECK_EQ(r1.residency_slot_bytes(), 2 * 8192ll);
    CHECK_EQ(r1.residency_layer_off(G_BF16)[7], 8192 + 4096ll);
    CHECK_EQ(r1.residency_layer_off(G_I8H)[15], 8192 + 2176ll);
    CHECK_OK(f.kv.set_residency_store(nullptr, 0));   /* `shared` goes before the managers do */
    CHECK_OK(r1.set_residency_store(nullptr, 0));
}

/* Demote, overwrite, promote with two layouts in one model: each group's blocks leave at their own
 * width, the pool is overwritten, and promotion brings every layer of every block back byte for
 * byte into fresh blocks, from slots that never alias across groups. Nothing about the movement
 * is layout specific -- the bytes are packed K, V and scales, moved as bytes -- so the check is
 * that the per-layer widths and offsets are honoured end to end. */
TEST(blocks_of_mixed_kv_layouts_demote_and_come_back_byte_for_byte_after_the_pool_is_overwritten) {
    if (!device_is_host()) {
        std::printf("  SKIP  the residency transfer needs the host backend (%s here)\n",
                    device_backend_name());
        return;
    }
    Registry reg;
    REQUIRE_EQ(reg.load_plugin(rows_plugin(), 0), RAD_OK);
    CHECK_OK(reg.check_complete());
    ResidencyFixture f;
    CHECK_OK(f.build_groups(mixed_residency_groups(), /*slots=*/64));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(0, 0)));
    RadStream compute = nullptr;
    REQUIRE_EQ(rad_stream_create(&compute, 0), RAD_OK);
    TierExec x;
    const std::vector<int32_t> groups{ G_MIX_F8, G_MIX_K6 };
    CHECK_OK(x.configure(&reg, &f.kv, &t, groups, TierExec::kMoveBatch, 64, 64,
                         /*leader=*/true, /*device=*/0, /*slot_off=*/0, /*shared=*/nullptr));

    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 20));
    std::vector<int32_t> orig[3];
    /* Salted by group: a slot that aliased across groups would bring back the other layout's
     * bytes, which the same block id in the other pool would otherwise match. */
    for (int32_t g : groups) {
        resid_fill(f.kv, g, 10 * g);
        orig[g] = f.kv.block_table(1, g);
        REQUIRE_EQ((int64_t)orig[g].size(), 20ll);
    }

    /* The sweep's order: every residency group of the sequence, the mover one group a call. */
    auto run = [&](std::vector<BlockMove>& mv, bool to_host) {
        for (size_t a = 0; a < mv.size();) {
            size_t z = a;
            while (z < mv.size() && z - a < (size_t)TierExec::kMoveBatch &&
                   mv[z].group == mv[a].group) ++z;
            const std::vector<BlockMove> part(mv.begin() + (long)a, mv.begin() + (long)z);
            CHECK_OK(x.issue_moves(part, compute, to_host));
            x.quiesce();
            a = z;
        }
    };
    std::vector<BlockMove> down;
    for (int32_t g : groups)
        CHECK_OK(f.kv.residency_demote(1, g, /*keep_head=*/2, /*keep_tail=*/4, &down));
    REQUIRE_EQ(down.size(), 28u);                      /* entries [2, 16) of both groups */
    run(down, /*to_host=*/true);
    f.kv.residency_demote_landed();
    CHECK_EQ(x.move_stats().out_blocks, 28ll);
    CHECK_EQ(x.move_stats().out_bytes, 14 * 2048ll + 14 * 4224ll);
    CHECK_EQ(f.store.used_slots(), 28ll);              /* one slot per group and entry */

    for (int32_t g : groups) resid_scribble(f.kv, g);
    CHECK(resid_mismatches(f.kv, G_MIX_K6, orig[G_MIX_K6][5], -1, 10 * G_MIX_K6) > 4000);

    std::vector<BlockMove> up;
    CHECK_OK(f.kv.residency_promote_all(1, &up));
    REQUIRE_EQ(up.size(), 28u);
    run(up, /*to_host=*/false);
    f.kv.residency_complete();
    CHECK_EQ(x.move_stats().in_bytes, 14 * 2048ll + 14 * 4224ll);
    for (int32_t g : groups) {
        const std::vector<int32_t> back = f.kv.block_table(1, g);
        for (size_t i = 2; i < 16; ++i) {
            CHECK(back[i] >= 0);
            CHECK_EQ(resid_mismatches(f.kv, g, back[i], orig[g][i], 10 * g), 0ll);
        }
        CHECK(f.kv.residency_cold_entries(1, g).empty());
    }
    CHECK_EQ(f.store.used_slots(), 0ll);

    CHECK_OK(rad_stream_sync(compute));
    rad_stream_destroy(compute);
}

/* ------------------------------------------------------------------ window residency: tiered precision */

/* A tiered layer in miniature (--kv-recipe `layer N i8h demote k4h_v2h`, RAD_KVQ_DEMOTE): two i8h
 * layers at head_dim 32 -- a 68-byte row, declared as 34 E4M3 columns -- two KV heads, 16-token
 * blocks, the indexer's window. A layer's fragment is 32 rows: 2176 bytes resident, 896 in
 * k4h_v2h's 28-byte rows. `demote` 0 is the same recipe without a demote layout. */
enum : int32_t { G_TIER = 1 };

static std::vector<KVGroupInfo> tiered_groups() {
    KVGroupInfo k = paged_group("kv_attn", RAD_KV_FULL, 2, 16, { 3, 7 });
    k.decl.dtype = RAD_F8E4M3;
    k.decl.head_dim = 34;
    k.decl.residency_window = 512;
    return { null_group(), k };
}

static Config tiered_config(int demote) {
    Config c = small_config();
    c.kv_recipe.assign(8, -1);
    for (int l : { 3, 7 })
        c.kv_recipe[(size_t)l] = (int)RAD_KVQ_I8H | (int)RAD_KVQ_ROTATED | (demote << RAD_KVQ_DEMOTE_SHIFT);
    return c;
}

struct TieredFixture {
    Pools pools;
    HostArena store;
    KVManager kv;
    Config cfg;
    int build(int demote, int64_t slots) {
        cfg = tiered_config(demote);
        int s = configure_pools(pools, cfg);
        if (s < 0) return s;
        s = kv.configure(tiered_groups(), cfg, pools.kv());
        if (s < 0) return s;
        s = store.open(kv.residency_slot_bytes(), kv.residency_slot_bytes() * slots);
        return s < 0 ? s : kv.set_residency_store(&store, 0);
    }
};

/* The store's slot is the demote layout's block, not the pool's: each layer's fragment is its 32
 * rows at k4h_v2h's width, one after the other, which is what the pager's layer offsets and the
 * demotion counters name. The pool keeps i8h. Without the demote layout the same recipe sizes the
 * slot from the pool's block exactly as before. */
TEST(a_tiered_groups_store_slot_holds_its_demote_layouts_block) {
    TieredFixture t;
    CHECK_OK(t.build(RAD_KVQ_K4HV2H, 16));
    const KVGroupPlan* p = t.kv.plan(G_TIER);
    REQUIRE(p && p->residency() && p->tiered());
    CHECK_EQ(p->res_from, (int)RAD_KVQ_I8H);
    CHECK_EQ(p->res_to, (int)RAD_KVQ_K4HV2H);
    CHECK_EQ(p->res_head_dim, 32ll);
    CHECK_EQ(p->layer_bytes_per_block(), 2176ll);
    CHECK_EQ(p->residency_frag(), 896ll);
    CHECK_EQ(t.kv.residency_block_bytes(G_TIER), 1792ll);
    CHECK_EQ(t.kv.residency_shard_bytes(), 1792ll);
    CHECK_EQ(t.kv.residency_slot_bytes(), 1792ll);              /* already whole 256-byte units */
    CHECK_EQ(t.kv.residency_layer_off(G_TIER)[3], 0ll);
    CHECK_EQ(t.kv.residency_layer_off(G_TIER)[7], 896ll);
    CHECK_OK(t.kv.add_sequence(1));
    CHECK_OK(t.kv.ensure(1, 16 * 20));
    std::vector<BlockMove> mv;
    CHECK_OK(t.kv.residency_demote(1, G_TIER, 2, 4, &mv));
    CHECK_EQ(mv.size(), 14u);
    CHECK_EQ(t.kv.residency_stats().demoted_bytes, 14 * 1792ll);
    t.kv.residency_demote_landed();
    CHECK_EQ(t.kv.residency_stats().freed_bytes, 14 * 4352ll);   /* VRAM: the resident block */

    TieredFixture u;
    CHECK_OK(u.build(0, 16));
    const KVGroupPlan* q = u.kv.plan(G_TIER);
    REQUIRE(q && q->residency() && !q->tiered());
    CHECK_EQ(q->residency_frag(), 2176ll);
    CHECK_EQ(u.kv.residency_block_bytes(G_TIER), 4352ll);
    CHECK_EQ(u.kv.residency_slot_bytes(), 4352ll);
    CHECK_EQ(u.kv.residency_layer_off(G_TIER)[7], 2176ll);
}

/* Layers that share a group must share its pair: a slot holds one width. The plugin keys its groups
 * by (resident, demote), so a group that mixes them is refused by name rather than sized wrong. */
TEST(a_group_whose_layers_name_different_demote_layouts_is_refused) {
    Config c = tiered_config(RAD_KVQ_K4HV2H);
    c.kv_recipe[7] = (int)RAD_KVQ_I8H | (int)RAD_KVQ_ROTATED | ((int)RAD_KVQ_K4H << RAD_KVQ_DEMOTE_SHIFT);
    Pools p;
    REQUIRE_EQ(configure_pools(p, c), RAD_OK);
    KVManager kv;
    CHECK_EQ(kv.configure(tiered_groups(), c, p.kv()), (int)RAD_E_INVAL);
}

/* A tiered block is requantised once in its life. Its promotion keeps the slot it came from as its
 * origin even though no cache holds it -- the block's bytes are the slot's expanded, and copying
 * them out again would round them through the demote layout a second time -- so the next sweep
 * points the entry back at the slot and moves nothing. A write into the block ends that, as for a
 * cached origin; the store holds the slot meanwhile. */
TEST(a_promoted_tiered_block_keeps_its_slot_and_a_re_demotion_moves_no_bytes) {
    TieredFixture t;
    CHECK_OK(t.build(RAD_KVQ_K4HV2H, 64));
    CHECK_OK(t.kv.add_sequence(1));
    CHECK_OK(t.kv.ensure(1, 16 * 20));
    std::vector<BlockMove> mv;
    CHECK_OK(t.kv.residency_demote(1, G_TIER, 2, 4, &mv));
    REQUIRE_EQ(mv.size(), 14u);
    t.kv.residency_demote_landed();
    const std::vector<int32_t> cold = t.kv.block_table(1, G_TIER);
    std::vector<BlockMove> up;
    CHECK_OK(t.kv.residency_promote_all(1, &up));
    REQUIRE_EQ(up.size(), 14u);
    t.kv.residency_complete();
    CHECK_EQ(t.store.used_slots(), 14ll);                 /* the origins */
    const auto before = t.kv.residency_stats();
    mv.clear();
    CHECK_OK(t.kv.residency_demote(1, G_TIER, 2, 4, &mv));
    CHECK(mv.empty());
    CHECK_EQ(t.kv.residency_stats().repointed - before.repointed, 14ll);
    CHECK_EQ(t.kv.residency_stats().demoted_bytes, before.demoted_bytes);
    for (size_t i = 2; i < 16; ++i) CHECK_EQ(t.kv.block_table(1, G_TIER)[i], cold[i]);
    CHECK_EQ(t.store.used_slots(), 14ll);
    t.kv.free_sequence(1);
    CHECK_EQ(t.store.used_slots(), 0ll);
}

/* The emulation (RADIANCE_DEBUG_KV_TIER=emulate): the sweep's demotion names exactly the blocks
 * residency would demote -- outside sink and tail, the partial last block kept, in table order,
 * under the same bound -- with no slot, and leaves the table, the pool and the store alone. A
 * block is named once; freeing it (or writing it) clears that, so a later owner's demotion of the
 * same id is a first one again. */
TEST(the_emulation_names_the_blocks_residency_would_demote_and_moves_none_of_them) {
    TieredFixture t;
    CHECK_OK(t.build(RAD_KVQ_K4HV2H, 64));
    t.kv.residency_emulate(true);
    CHECK_OK(t.kv.add_sequence(1));
    CHECK_OK(t.kv.ensure(1, 16 * 20));
    const std::vector<int32_t> tbl = t.kv.block_table(1, G_TIER);
    const int64_t free0 = t.kv.free_blocks(G_TIER);
    std::vector<BlockMove> mv;
    CHECK_OK(t.kv.residency_demote(1, G_TIER, 2, 4, &mv, /*max_blocks=*/10));
    REQUIRE_EQ(mv.size(), 10u);
    for (size_t i = 0; i < mv.size(); ++i) {
        CHECK_EQ(mv[i].block, tbl[2 + i]);
        CHECK_EQ(mv[i].host_slot, -1);
        CHECK(t.kv.residency_emulated(G_TIER, mv[i].block));
    }
    mv.clear();
    CHECK_OK(t.kv.residency_demote(1, G_TIER, 2, 4, &mv));
    REQUIRE_EQ(mv.size(), 4u);                           /* the rest, entries 12..15 */
    CHECK_EQ(mv[0].block, tbl[12]);
    mv.clear();
    CHECK_OK(t.kv.residency_demote(1, G_TIER, 2, 4, &mv));
    CHECK(mv.empty());
    t.kv.residency_demote_landed();
    CHECK(t.kv.block_table(1, G_TIER) == tbl);
    CHECK_EQ(t.kv.free_blocks(G_TIER), free0);
    CHECK_EQ(t.store.used_slots(), 0ll);
    CHECK(t.kv.residency_cold_entries(1, G_TIER).empty());
    std::vector<BlockMove> up;
    CHECK_OK(t.kv.residency_promote_all(1, &up));
    CHECK(up.empty());
    t.kv.free_sequence(1);
    for (size_t i = 2; i < 16; ++i) CHECK(!t.kv.residency_emulated(G_TIER, tbl[i]));
}

/* The tiered transfer end to end on the host backend, with the test plugin's stand-in for
 * kv_transcode (a row keeps its first bytes, a wider one is padded with 0xA5; the codec itself is
 * libref's and libr4d's, held elsewhere): a demotion writes each block's rows at the demote width
 * into its slot, layer after layer, and only those bytes; a promotion expands them back into a
 * fresh block at the resident width. The emulation's round trip leaves the same bytes in the block
 * it started in, with nothing in the store -- the bytes residency serves after a promotion. */
TEST(a_tiered_block_leaves_at_the_demote_width_and_comes_back_expanded_or_emulated_in_place) {
    if (!device_is_host()) {
        std::printf("  SKIP  the residency transfer needs the host backend (%s here)\n",
                    device_backend_name());
        return;
    }
    Registry reg;
    REQUIRE_EQ(reg.load_plugin(rows_plugin(), 0), RAD_OK);
    const int64_t rf = 68, rt = 28, rows = 32, frag = rows * rf, sfrag = rows * rt;
    /* What a block's layer l holds after the round trip: each row's first rt bytes, then 0xA5. */
    auto trip = [&](int64_t l, int32_t pat_block, int64_t i) -> unsigned char {
        return i % rf < rt ? resid_pat(l, pat_block, i) : (unsigned char)0xA5;
    };
    for (bool emul : { false, true }) {
        TieredFixture t;
        CHECK_OK(t.build(RAD_KVQ_K4HV2H, 64));
        t.kv.residency_emulate(emul);
        IdleTiers it;
        CHECK_OK(it.configure(tier_config(0, 0)));
        RadStream compute = nullptr;
        REQUIRE_EQ(rad_stream_create(&compute, 0), RAD_OK);
        TierExec x;
        CHECK_OK(x.configure(&reg, &t.kv, &it, { G_TIER }, TierExec::kMoveBatch, 64, 64,
                             /*leader=*/true, /*device=*/0, /*slot_off=*/0, /*shared=*/nullptr));
        CHECK_OK(t.kv.add_sequence(1));
        CHECK_OK(t.kv.ensure(1, 16 * 20));
        resid_fill(t.kv, G_TIER);
        const std::vector<int32_t> orig = t.kv.block_table(1, G_TIER);
        std::vector<BlockMove> down;
        CHECK_OK(t.kv.residency_demote(1, G_TIER, 2, 4, &down));
        REQUIRE_EQ(down.size(), 14u);
        for (size_t a = 0; a < down.size(); a += TierExec::kMoveBatch) {
            const std::vector<BlockMove> part(
                down.begin() + (long)a,
                down.begin() + (long)std::min(down.size(), a + TierExec::kMoveBatch));
            CHECK_OK(x.issue_moves(part, compute, /*to_host=*/true));
            x.quiesce();
        }
        t.kv.residency_demote_landed();
        const KVGroupPlan* p = t.kv.plan(G_TIER);
        auto* base = (const unsigned char*)p->base;
        auto block_is_trip = [&](int32_t b, int32_t pat_block) {
            for (int64_t l = 0; l < 2; ++l)
                for (int64_t i = 0; i < frag; ++i)
                    if (base[l * p->layer_stride() + b * frag + i] != trip(l, pat_block, i))
                        return false;
            return true;
        };
        if (emul) {
            /* In place: the same blocks, the table unchanged, nothing copied out. */
            CHECK_EQ(t.store.used_slots(), 0ll);
            CHECK_EQ(x.move_stats().out_bytes, 0ll);
            for (size_t i = 2; i < 16; ++i) CHECK(block_is_trip(orig[i], orig[i]));
            CHECK_EQ(resid_mismatches(t.kv, G_TIER, orig[1]), 0ll);   /* the sink untouched */
            CHECK_EQ(resid_mismatches(t.kv, G_TIER, orig[16]), 0ll);  /* the tail untouched */
        } else {
            CHECK_EQ(x.move_stats().out_bytes, 14 * 2 * sfrag);
            /* Each slot: layer 3's 32 rows of 28 bytes, then layer 7's. */
            const std::vector<int32_t> cold = t.kv.block_table(1, G_TIER);
            for (size_t e = 2; e < 16; ++e) {
                const auto* s = (const unsigned char*)t.store.at(rad_kv_entry_slot(cold[e]));
                int64_t bad = 0;
                for (int64_t l = 0; l < 2; ++l)
                    for (int64_t r = 0; r < rows; ++r)
                        for (int64_t j = 0; j < rt; ++j)
                            bad += s[l * sfrag + r * rt + j] != resid_pat(l, orig[e], r * rf + j);
                CHECK_EQ(bad, 0ll);
            }
            resid_scribble(t.kv, G_TIER);
            std::vector<BlockMove> up;
            CHECK_OK(t.kv.residency_promote_all(1, &up));
            REQUIRE_EQ(up.size(), 14u);
            for (size_t a = 0; a < up.size(); a += TierExec::kMoveBatch) {
                const std::vector<BlockMove> part(
                    up.begin() + (long)a,
                    up.begin() + (long)std::min(up.size(), a + TierExec::kMoveBatch));
                CHECK_OK(x.issue_moves(part, compute, /*to_host=*/false));
                x.quiesce();
            }
            t.kv.residency_complete();
            CHECK_EQ(x.move_stats().in_bytes, 14 * 2 * sfrag);
            const std::vector<int32_t> back = t.kv.block_table(1, G_TIER);
            for (size_t e = 2; e < 16; ++e) CHECK(block_is_trip(back[e], orig[e]));
        }
        t.kv.free_sequence(1);
        CHECK_OK(rad_stream_sync(compute));
        rad_stream_destroy(compute);
    }
}

TEST(a_promotion_with_no_free_block_stops_with_full_and_loses_no_slot) {
    ResidencyFixture f;
    CHECK_OK(f.build(512, 256));
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    resid_go_cold(f, 1);
    const int64_t used = f.store.used_slots();
    /* Another sequence drains the pool, so the owner cannot get 30 blocks back. */
    CHECK_OK(f.kv.add_sequence(2));
    int64_t n = 0;
    while (f.kv.ensure(2, 16 * (n + 1)) == RAD_OK) ++n;
    /* ensure stops at the first group that runs out, which need not be this one; take what this
     * group still has so it is the one that is short. */
    std::vector<int32_t> held;
    (void)f.kv.reserve_blocks(G_RES, f.kv.free_blocks(G_RES), &held);
    std::vector<BlockMove> up;
    CHECK_EQ(f.kv.residency_promote_all(1, &up), RAD_E_FULL);
    CHECK(up.empty());                                  /* all or nothing */
    CHECK_EQ(f.store.used_slots(), used);
    CHECK_EQ((int64_t)f.kv.residency_cold_entries(1, G_RES).size(), 30ll);
}

TEST(rollback_frees_the_cold_entries_it_cuts_and_keeps_the_slots_of_the_rest) {
    ResidencyFixture f;
    CHECK_OK(f.build(512, 256));
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    resid_go_cold(f, 1);
    CHECK_EQ(f.store.used_slots(), 30ll);
    /* Back to 20 blocks: the cold entries 20..31 go, 2..19 stay cold. The cut ends on a block
     * boundary, so the new tail is full and the next append starts a fresh block. */
    CHECK_OK(f.kv.rollback(1, 16 * 20));
    CHECK_EQ(f.store.used_slots(), 18ll);
    CHECK_EQ((int64_t)f.kv.residency_cold_entries(1, G_RES).size(), 18ll);
    /* A cut into a cold block would leave the next token's write in host memory: refused, and
     * nothing is freed. */
    CHECK_EQ(f.kv.rollback(1, 16 * 10 + 5), RAD_E_STATE);
    CHECK_EQ(f.store.used_slots(), 18ll);
    CHECK_EQ(f.kv.seq_tokens(1), 16ll * 20);
    /* Appending past a cold entry is fine: only the partial tail is written. */
    CHECK_OK(f.kv.ensure(1, 16 * 20 + 3));
    f.kv.free_sequence(1);
    CHECK_EQ(f.store.used_slots(), 0ll);
}

TEST(a_sequence_with_cold_blocks_cannot_be_forked_until_it_is_promoted) {
    ResidencyFixture f;
    CHECK_OK(f.build(512, 256));
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    resid_go_cold(f, 1);
    CHECK_EQ(f.kv.fork(1, 2), RAD_E_STATE);
    CHECK(!f.kv.has_sequence(2));                       /* nothing was half-built */
    CHECK_EQ(f.store.used_slots(), 30ll);
    std::vector<BlockMove> up;
    CHECK_OK(f.kv.residency_promote_all(1, &up));
    f.kv.residency_complete();
    CHECK_OK(f.kv.fork(1, 2));
    CHECK(f.kv.block_table(2, G_RES) == f.kv.block_table(1, G_RES));
    CHECK_EQ(f.store.used_slots(), 0ll);
}

/* ------------------------------------------------------------------ window residency: the prefix cache */

/* A 40-block conversation of the residency group, paged out between its sink and its window the
 * way a decode sweep leaves it (entries [2, 32) cold), published and freed: the cache alone holds
 * it, 30 store slots at one reference each. */
static std::vector<int32_t> resid_publish_cold(Fixture& f, uint64_t id, int32_t base) {
    const std::vector<int32_t> tokens = tok_run(base, 16 * 40);
    CHECK_OK(f.kv.add_sequence(id));
    CHECK_OK(f.kv.ensure(id, (int64_t)tokens.size()));
    std::vector<BlockMove> moves;
    CHECK_OK(f.kv.residency_demote(id, G_RES, 2, 8, &moves));
    f.kv.residency_demote_landed();
    std::vector<std::vector<int32_t>> bt;
    for (int32_t g : f.pc.cached_groups()) bt.push_back(f.kv.block_table(id, g));
    CHECK_OK(f.pc.insert(id, tokens, {}, bt, (int64_t)tokens.size()));
    f.kv.free_sequence(id);
    return tokens;
}

static void resid_open_store(Fixture& f, HostArena& store, int64_t slots = 256) {
    CHECK_OK(f.build_groups(residency_groups(512)));
    CHECK_OK(store.open(f.kv.residency_slot_bytes(), f.kv.residency_slot_bytes() * slots));
    CHECK_OK(f.kv.set_residency_store(&store, 0));
}

/* The finding in #85: a long conversation's middle goes cold during decode, and a cache that
 * refused cold blocks re-prefilled almost the whole context on the next turn. */
TEST(the_prefix_cache_indexes_cold_blocks_and_a_hit_adopts_them_cold) {
    Fixture f;
    HostArena store;
    resid_open_store(f, store);
    const int64_t pool0 = f.kv.free_blocks(G_RES);
    const std::vector<int32_t> tokens = resid_publish_cold(f, 1, 1);

    CHECK_EQ(f.pc.size(), 40ll);                             /* the chain is not cut at the first cold entry */
    CHECK_EQ(store.used_slots(), 30ll);                      /* the cache's references keep the slots */
    CHECK_EQ(f.kv.free_blocks(G_RES), pool0 - 10);      /* only the 10 resident entries hold blocks */

    std::vector<int32_t> prompt = tokens;
    prompt.push_back(99999);
    std::vector<std::vector<int32_t>> hit;
    const CacheHit h = f.pc.lookup(prompt, {}, &hit);
    CHECK_EQ(h.n_attn_tokens, 16 * 40);
    REQUIRE_EQ(hit.size(), 2u);
    int64_t cold = 0;
    for (int32_t e : hit[0]) cold += f.kv.residency_cold(G_RES, e);
    CHECK_EQ(cold, 30ll);

    /* Adopting bumps the slots to two owners; the sequence's table is cold where the cache's is. */
    CHECK_OK(f.kv.add_sequence(2));
    for (size_t i = 0; i < hit.size(); ++i)
        CHECK_OK(f.kv.adopt(2, f.pc.cached_groups()[i], hit[i], h.n_attn_tokens));
    CHECK_EQ((int64_t)f.kv.residency_cold_entries(2, G_RES).size(), 30ll);
    CHECK_EQ(store.used_slots(), 30ll);                      /* shared, not copied */
    CHECK_EQ(f.kv.entry_refs(G_RES, f.kv.block_table(2, G_RES)[5]), 2);

    /* The sequence going away leaves the cache's reference; the cache going away leaves nothing. */
    f.kv.free_sequence(2);
    CHECK_EQ(store.used_slots(), 30ll);
    f.pc.clear();
    CHECK_EQ(store.used_slots(), 0ll);
    CHECK_EQ(f.kv.free_blocks(G_RES), pool0);
    /* A cold id is a slot reference, never a pool block: the block calls refuse it. */
    CHECK_EQ(f.kv.retain_block(G_RES, -2), RAD_E_INVAL);
    CHECK_EQ(f.kv.retain_entry(G_RES, -1), RAD_E_INVAL);
    CHECK_EQ(f.kv.retain_entry(G_RES, rad_kv_cold_entry(5)), RAD_E_STATE);   /* a free slot */
}

/* The acceptance check for the whole change: what the next turn promotes is what the last turn
 * paged out. */
TEST(a_published_cold_chain_comes_back_byte_for_byte_through_a_cache_hit) {
    if (!device_is_host()) {
        std::printf("  SKIP  the residency transfer needs the host backend (%s here)\n",
                    device_backend_name());
        return;
    }
    Registry reg;
    REQUIRE_EQ(reg.load_plugin(rows_plugin(), 0), RAD_OK);
    CHECK_OK(reg.check_complete());
    Fixture f;
    HostArena store;
    resid_open_store(f, store, 64);
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(0, 0)));
    RadStream compute = nullptr;
    REQUIRE_EQ(rad_stream_create(&compute, 0), RAD_OK);
    TierExec x;
    const std::vector<int32_t> groups{ G_RES };
    CHECK_OK(x.configure(&reg, &f.kv, &t, groups, TierExec::kMoveBatch, 64, 64,
                         /*leader=*/true, /*device=*/0, /*slot_off=*/0, /*shared=*/nullptr));
    const int64_t pool0 = f.kv.free_blocks(G_RES);

    const std::vector<int32_t> tokens = tok_run(1, 16 * 40);
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, (int64_t)tokens.size()));
    resid_fill(f.kv, G_RES);
    const std::vector<int32_t> orig = f.kv.block_table(1, G_RES);
    std::vector<BlockMove> down;
    CHECK_OK(f.kv.residency_demote(1, G_RES, 2, 8, &down));
    REQUIRE_EQ(down.size(), 30u);
    for (size_t i = 0; i < down.size(); i += TierExec::kMoveBatch) {
        const std::vector<BlockMove> batch(down.begin() + (long)i,
                                           down.begin() + (long)std::min(down.size(), i + TierExec::kMoveBatch));
        CHECK_OK(x.issue_moves(batch, compute, /*to_host=*/true));
        x.quiesce();
    }
    f.kv.residency_demote_landed();
    std::vector<std::vector<int32_t>> bt;
    for (int32_t g : f.pc.cached_groups()) bt.push_back(f.kv.block_table(1, g));
    CHECK_OK(f.pc.insert(1, tokens, {}, bt, (int64_t)tokens.size()));
    f.kv.free_sequence(1);
    resid_scribble(f.kv, G_RES);                             /* the pool moves on to other owners */

    std::vector<int32_t> prompt = tokens;
    prompt.push_back(99999);
    std::vector<std::vector<int32_t>> hit;
    const CacheHit h = f.pc.lookup(prompt, {}, &hit);
    CHECK_EQ(h.n_attn_tokens, 16 * 40);
    CHECK_OK(f.kv.add_sequence(2));
    for (size_t i = 0; i < hit.size(); ++i)
        CHECK_OK(f.kv.adopt(2, f.pc.cached_groups()[i], hit[i], h.n_attn_tokens));

    std::vector<BlockMove> up;
    CHECK_OK(f.kv.residency_promote_all(2, &up));
    REQUIRE_EQ(up.size(), 30u);
    for (size_t i = 0; i < up.size(); i += TierExec::kMoveBatch) {
        const std::vector<BlockMove> batch(up.begin() + (long)i,
                                           up.begin() + (long)std::min(up.size(), i + TierExec::kMoveBatch));
        CHECK_OK(x.issue_moves(batch, compute, /*to_host=*/false));
        x.quiesce();
    }
    f.kv.residency_complete();
    for (int64_t e = 2; e < 32; ++e) {
        const int32_t now = f.kv.block_table(2, G_RES)[(size_t)e];
        CHECK(now >= 0);
        CHECK_EQ(resid_mismatches(f.kv, G_RES, now, orig[(size_t)e]), 0ll);
    }
    CHECK_EQ(store.used_slots(), 30ll);                      /* the cache still holds every slot */

    f.kv.free_sequence(2);
    f.pc.clear();
    CHECK_EQ(store.used_slots(), 0ll);
    CHECK_EQ(f.kv.free_blocks(G_RES), pool0);
    CHECK_OK(rad_stream_sync(compute));
    rad_stream_destroy(compute);
}

/* Every way an entry leaves the cache gives its slot back, and a re-plan gives back whatever the
 * cache still held. */
TEST(no_store_slot_or_pool_block_outlives_a_cold_entry_however_the_cache_lets_it_go) {
    enum { Evict, Trim, Clear, Expire, RePlan };
    for (int how : { Evict, Trim, Clear, Expire, RePlan }) {
        Fixture f;
        HostArena store;
        resid_open_store(f, store);
        const int64_t pool0 = f.kv.free_blocks(G_RES), keys0 = f.kv.free_blocks(G_KEYS);
        (void)resid_publish_cold(f, 1, 1);
        CHECK_EQ(store.used_slots(), 30ll);
        switch (how) {
        case Evict:  CHECK_EQ(f.pc.evict_lru(1000), 40ll); break;
        case Trim:   CHECK_EQ(f.pc.trim(1000), 40ll); break;
        case Clear:  f.pc.clear(); break;
        case Expire: CHECK_EQ(f.pc.expire(rad_mono_ns() + 1000000000ull, 1), 40ll); break;
        case RePlan: CHECK_OK(f.kv.plan_groups(residency_groups(512), f.cfg)); break;
        }
        CHECK_EQ(store.used_slots(), 0ll);
        if (how == RePlan) continue;                         /* the pool is re-carved with the plan */
        CHECK_EQ(f.pc.size(), 0ll);
        CHECK_EQ(f.kv.free_blocks(G_RES), pool0);
        CHECK_EQ(f.kv.free_blocks(G_KEYS), keys0);
    }
}

/* A block promoted from a slot another owner still holds, and untouched since, goes back by
 * pointing at that slot: no bytes move and the store gains no second copy of the prefix. */
TEST(re_demoting_an_unchanged_promoted_block_moves_no_bytes_and_a_written_one_is_copied_fresh) {
    Fixture f;
    HostArena store;
    resid_open_store(f, store);
    const int64_t pool0 = f.kv.free_blocks(G_RES);
    const std::vector<int32_t> tokens = resid_publish_cold(f, 1, 1);
    std::vector<int32_t> prompt = tokens;
    prompt.push_back(99999);
    std::vector<std::vector<int32_t>> hit;
    const CacheHit h = f.pc.lookup(prompt, {}, &hit);
    CHECK_OK(f.kv.add_sequence(2));
    for (size_t i = 0; i < hit.size(); ++i)
        CHECK_OK(f.kv.adopt(2, f.pc.cached_groups()[i], hit[i], h.n_attn_tokens));
    const std::vector<int32_t> slots = f.kv.block_table(2, G_RES);     /* cold ids of the cache */
    std::vector<BlockMove> up;
    CHECK_OK(f.kv.residency_promote_all(2, &up));
    f.kv.residency_complete();
    CHECK_EQ(store.used_slots(), 30ll);                      /* promoting copied out of the cache's slots */
    CHECK_EQ(f.kv.free_blocks(G_RES), pool0 - 40);

    const auto before = f.kv.residency_stats();
    std::vector<BlockMove> down;
    CHECK_OK(f.kv.residency_demote(2, G_RES, 2, 8, &down));
    CHECK(down.empty());                                     /* nothing to copy: the bytes are in the slots */
    const auto after = f.kv.residency_stats();
    CHECK_EQ(after.repointed - before.repointed, 30ll);
    CHECK_EQ(after.demoted_bytes, before.demoted_bytes);
    CHECK_EQ(after.freed_blocks - before.freed_blocks, 30ll);
    CHECK_EQ(store.used_slots(), 30ll);                      /* the same slots, no new copy */
    for (int64_t e = 2; e < 32; ++e) CHECK_EQ(f.kv.block_table(2, G_RES)[(size_t)e], slots[(size_t)e]);
    CHECK_EQ(f.kv.free_blocks(G_RES), pool0 - 10);      /* the 30 blocks are free at once */

    /* Promote again, then write into one block: rolling back into entry 10 and growing past it is
     * what a tail does. That block no longer equals its slot, so it takes a fresh one. */
    up.clear();
    CHECK_OK(f.kv.residency_promote_all(2, &up));
    f.kv.residency_complete();
    CHECK_OK(f.kv.rollback(2, 16 * 10 + 5));
    CHECK_OK(f.kv.ensure(2, 16 * 14));
    down.clear();
    CHECK_OK(f.kv.residency_demote(2, G_RES, 2, 2, &down));
    /* Entries 2..9 are untouched promotions; 10 was written and 11 is new: those two are copied. */
    CHECK_EQ(down.size(), 2u);
    const std::set<int32_t> cache_slots(slots.begin() + 2, slots.begin() + 32);
    for (const BlockMove& m : down) CHECK(cache_slots.count(rad_kv_cold_entry(m.host_slot)) == 0);
    CHECK_EQ(f.kv.residency_stats().repointed - before.repointed, 38ll);
    f.kv.residency_demote_landed();

    f.kv.free_sequence(2);
    f.pc.clear();
    CHECK_EQ(store.used_slots(), 0ll);
    CHECK_EQ(f.kv.free_blocks(G_RES), pool0);
}

/* The tiers gather pool blocks; a cold id would index the pool out of range, and the residency
 * group's bytes are already in host memory. */
TEST(idle_tiers_are_never_handed_a_cache_entry_with_a_cold_component) {
    Fixture f;
    HostArena store;
    resid_open_store(f, store);
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(64 * 1024, 1024)));
    f.pc.set_tiers(&t);
    (void)resid_publish_cold(f, 1, 1);
    std::vector<TierJob> jobs;
    t.plan_writeback(1 << 20, &jobs);
    CHECK_EQ(jobs.size(), 10u);                              /* the sink and window entries only */
    for (const TierJob& j : jobs)
        for (int32_t b : j.blocks) CHECK(b >= 0);
    for (TierJob& j : jobs) t.report(j, true);
    /* Dropping what the tiers hold leaves the cold chain alone. */
    CHECK_EQ(f.pc.drop_clean(), 10ll);
    CHECK_EQ(store.used_slots(), 30ll);
    f.pc.clear();
    CHECK_EQ(store.used_slots(), 0ll);
}

TEST(a_re_plan_with_cold_entries_and_a_demote_in_flight_leaks_no_slot) {
    ResidencyFixture f;
    CHECK_OK(f.build(512, 256));
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    std::vector<BlockMove> moves;
    CHECK_OK(f.kv.residency_demote(1, G_RES, 2, 8, &moves));     /* not landed */
    CHECK_OK(f.kv.add_sequence(2));
    CHECK_OK(f.kv.ensure(2, 16 * 40));
    resid_go_cold(f, 2);
    std::vector<BlockMove> up;
    CHECK_OK(f.kv.residency_promote(2, G_RES, { 5, 6 }, &up));   /* a promote in flight */
    CHECK_EQ(f.store.used_slots(), 60ll);
    /* The store cannot be swapped under cold entries... */
    CHECK_EQ(f.kv.set_residency_store(nullptr, 0), RAD_E_STATE);
    /* ...and a re-plan returns every slot, the landed, the in-flight and the held. */
    CHECK_OK(f.kv.plan_groups(residency_groups(512), f.cfg));
    CHECK_EQ(f.store.used_slots(), 0ll);
}

TEST(a_demotion_marks_the_table_changed_so_the_device_copy_is_restaged) {
    ResidencyFixture f;
    CHECK_OK(f.build(512, 256));
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    (void)f.kv.take_table_changes(1, G_RES);            /* the mirror is current */
    std::vector<BlockMove> moves;
    CHECK_OK(f.kv.residency_demote(1, G_RES, 2, 8, &moves));
    CHECK_EQ(f.kv.take_table_changes(1, G_RES), 2ll);   /* from the first demoted entry */
    std::vector<BlockMove> up;
    CHECK_OK(f.kv.residency_promote(1, G_RES, { 10 }, &up));
    CHECK_EQ(f.kv.take_table_changes(1, G_RES), 10ll);
}


/* ------------------------------------------------------------------ the pager's scratch pages */

/* A host-mapped store plus the reservation the pager needs: --kv-residency-mib on, a 128-token
 * window (8 table entries) on the 16-token group, 8 sequences. */
struct ScratchFixture : ResidencyFixture {
    ScratchFixture(int64_t residency_mib) {
        cfg.vram_kv_mib = 8;
        cfg.residency_mib = residency_mib;
    }
    int open_store(int64_t slots) {
        const int s = store.open_mapped(kv.residency_slot_bytes(), kv.residency_slot_bytes() * slots);
        if (s < 0) return s;
        return kv.set_residency_store(&store, 0);
    }
    int configure() {
        int s = configure_pools(pools, cfg);
        return s < 0 ? s : kv.configure(residency_groups(128), cfg, pools.kv());
    }
};

/* Reserved at carve and charged before the split: the budget resolver reads ceiling() and
 * fixed_bytes() of an unallocated manager, so the reservation must be in both, or the resolver
 * would over-commit the card by exactly the scratch. A row of the decode batch (max_seqs x
 * (1 + n_spec)) needs topk + 1 pages of the window group and none of the group without a window. */
TEST(the_pagers_scratch_pages_are_charged_to_the_vram_budget_before_the_split) {
    Config off = small_config(), on = small_config();
    on.residency_mib = 64;
    KVManager a, b;
    CHECK_OK(a.plan_groups(residency_groups(128), off));
    CHECK_OK(b.plan_groups(residency_groups(128), on));

    const int64_t pages = (128 / 16 + 1) * off.max_seqs;      /* topk + 1, per row */
    const int64_t bytes = pages * 2048;
    CHECK_EQ(b.plan(G_RES)->scratch_blocks, pages);
    CHECK_EQ(b.plan(G_RES)->scratch_pitch, 128ll / 16 + 1);
    CHECK_EQ(b.plan(G_KEYS)->scratch_blocks, 0ll);
    CHECK_EQ(a.plan(G_RES)->scratch_blocks, 0ll);             /* a window nobody pages costs nothing */
    CHECK_EQ(b.fixed_bytes() - a.fixed_bytes(), bytes);
    CHECK_EQ(b.ceiling(1024) - a.ceiling(1024), bytes);
    /* The pool a full-length sequence needs grows by at least the scratch. */
    const int64_t ma = a.pool_mib_for_tokens(256, 64), mb = b.pool_mib_for_tokens(256, 64);
    CHECK(ma > 0 && mb * (1 << 20) >= ma * (1 << 20) + bytes - (1 << 20));

    /* And the resolver caps the cache's share at the larger ceiling. */
    VramFacts f = big_moe_facts();
    f.lends = true;
    f.expert_floor = 1LL << 30;
    f.kv_ceiling = b.ceiling(1024);
    Config c = derive_config();
    VramBudget bud;
    CHECK_OK(vram_budget_resolve(c, { f }, &bud));
    CHECK_EQ(bud.kv, f.kv_ceiling);
    CHECK(bud.kv - [&] {
        VramFacts g = f;
        g.kv_ceiling = a.ceiling(1024);
        Config c2 = derive_config();
        VramBudget b2;
        CHECK_OK(vram_budget_resolve(c2, { g }, &b2));
        return b2.kv;
    }() == bytes);
}

/* Engine-owned: ids [0, scratch) of the group, referenced for good, on no free list. A sequence
 * that could be handed one would have the pager overwrite its keys and values mid-step. */
TEST(scratch_pages_are_carved_but_no_sequence_is_ever_handed_one) {
    ScratchFixture f(64);
    CHECK_OK(f.configure());
    const KVGroupPlan* p = f.kv.plan(G_RES);
    const int64_t scratch = p->scratch_blocks;
    CHECK_EQ(scratch, 9ll * 8);
    /* Carved (pool bytes) but not countable capacity. */
    CHECK_EQ(p->n_blocks, f.kv.total_blocks(G_RES) + scratch);
    CHECK_EQ(f.kv.free_blocks(G_RES), f.kv.total_blocks(G_RES));
    CHECK_EQ(f.kv.held_blocks(G_RES), scratch);

    /* The table is rows x pitch, row r holding pages [r * pitch, (r + 1) * pitch). */
    const std::vector<int32_t>& tbl = f.kv.scratch_table(G_RES);
    CHECK_EQ((int64_t)tbl.size(), scratch);
    CHECK_EQ(f.kv.scratch_rows(), 8ll);
    for (int64_t i = 0; i < scratch; ++i) CHECK_EQ(tbl[(size_t)i], (int32_t)i);
    CHECK(f.kv.scratch_table(G_KEYS).empty());

    /* Drain the group: every id handed out is past the scratch and none repeats. */
    std::set<int32_t> seen;
    for (uint64_t id = 1; id < 1000 && f.kv.free_blocks(G_RES) > 0 && f.kv.free_blocks(G_KEYS) > 0; ++id) {
        CHECK_OK(f.kv.add_sequence(id));
        if (f.kv.ensure(id, 16 * 8) != RAD_OK) break;
        for (int32_t b : f.kv.block_table(id, G_RES)) {
            CHECK(b >= scratch);
            CHECK(seen.insert(b).second);
        }
    }
    CHECK(!seen.empty());
    /* The pool cannot lend the top over the scratch either: nothing below a held block is lent. */
    CHECK(f.kv.can_ever_fit(16) && !f.kv.can_ever_fit(16 * (f.kv.total_blocks(G_RES) + 1)));
}

/* The batch gets the pager's operands through the scheduler's interface: the card's address of
 * the store, the slot size, each layer's fragment inside a slot and the scratch table. */
TEST(the_manager_hands_the_batch_its_store_layer_offsets_and_scratch_table) {
    ScratchFixture f(64);
    CHECK_OK(f.configure());
    KVBinding<KVManager> bind(&f.kv);
    /* No store yet: nothing for the pager. */
    CHECK(bind.pager(G_RES, 0).store == nullptr);
    CHECK_OK(f.open_store(16));
    CHECK(f.store.device_base() != nullptr);

    const IKVManager::Pager p = bind.pager(G_RES, 0);
    CHECK(p.store == f.store.device_base());
    CHECK_EQ(p.slot_bytes, f.kv.residency_slot_bytes());
    CHECK(p.scratch == f.kv.scratch_table(G_RES).data());
    CHECK_EQ(bind.scratch_pitch(G_RES), 9ll);
    /* Layers 0 and 1 are the group's; each fragment is one block of that layer's pool, in order. */
    const KVGroupPlan* pl = f.kv.plan(G_RES);
    CHECK(p.layer_off != nullptr);
    CHECK(p.layer_off[0] >= 0);
    CHECK_EQ(p.layer_off[1] - p.layer_off[0], pl->layer_bytes_per_block());
    CHECK(p.layer_off[1] + pl->layer_bytes_per_block() <= f.kv.residency_slot_bytes());
    /* A group with no window has none of it. */
    const IKVManager::Pager none = bind.pager(G_KEYS, 0);
    CHECK(none.store == nullptr && none.layer_off == nullptr && none.scratch == nullptr);
    CHECK_EQ(bind.scratch_pitch(G_KEYS), 0ll);
}

/* Without the flag the store is never opened, so nothing pages and nothing is reserved. */
TEST(no_scratch_is_reserved_when_no_residency_store_is_asked_for) {
    ScratchFixture f(0);
    CHECK_OK(f.configure());
    CHECK_EQ(f.kv.plan(G_RES)->scratch_blocks, 0ll);
    CHECK(f.kv.scratch_table(G_RES).empty());
    KVBinding<KVManager> bind(&f.kv);
    CHECK(bind.pager(G_RES, 0).store == nullptr);
    CHECK_EQ(f.kv.free_blocks(G_RES), f.kv.total_blocks(G_RES));
}

/* The pager reads the store from the card, so a host whose pinned memory has no address on the
 * card must refuse by the flag's name, and leave the arena closed. */
TEST(a_host_without_svm_refuses_the_residency_store_and_names_the_flag) {
    HostArena a;
    /* The refusal is what the operator reads, so capture stderr, where the log goes. */
    const std::string path = radtest::tmp_path("svm.log");
    fflush(stderr);
    const int saved = dup(2);
    CHECK(freopen(path.c_str(), "w", stderr) != nullptr);
    const int rc = a.open_mapped(4096, 4096 * 4, [](void*) -> void* { return nullptr; });
    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    clearerr(stderr);
    CHECK_EQ(rc, RAD_E_UNSUPPORTED);
    CHECK(!a.enabled());
    CHECK(a.device_base() == nullptr);
    std::string log;
    if (FILE* fp = fopen(path.c_str(), "r")) {
        char buf[512];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, fp)) > 0) log.append(buf, n);
        fclose(fp);
    }
    CHECK(log.find("--kv-residency-mib") != std::string::npos);

    HostArena ok;
    CHECK_OK(ok.open_mapped(4096, 4096 * 4));
    CHECK(ok.enabled() && ok.device_base() != nullptr);
}


/* A server pages for every long request it serves, so the per-session counters are bounded: past
 * kResidencySessions the oldest request ids go, and the session being counted never does. */
TEST(the_per_session_residency_counters_keep_only_the_newest_sessions) {
    ResidencyFixture f;
    CHECK_OK(f.build(512, 256));
    const uint64_t n = KVManager::kResidencySessions + 10;
    for (uint64_t id = 1; id <= n; ++id) {
        CHECK_OK(f.kv.add_sequence(id));
        CHECK_OK(f.kv.ensure(id, 16 * 4));
        std::vector<BlockMove> moves;
        CHECK_OK(f.kv.residency_demote(id, G_RES, 1, 1, &moves));
        f.kv.residency_demote_landed();
        f.kv.free_sequence(id);
    }
    const auto& by = f.kv.residency_by_seq();
    CHECK_EQ(by.size(), (size_t)KVManager::kResidencySessions);
    CHECK(by.count(1) == 0);                    /* the oldest went */
    CHECK(by.count(n) == 1);                    /* the newest stayed */
    CHECK_EQ(by.begin()->first, n - KVManager::kResidencySessions + 1);
}

/* The engine's half of the sweep: it polls, never waits, for a move to land, and a follower rank
 * carries out the list rank 0 planned though its own manager holds no record of it. Before the
 * first move there is nothing to wait for. */
TEST(a_move_is_polled_until_it_lands_and_a_follower_rank_runs_the_leaders_list) {
    if (!device_is_host()) {
        std::printf("  SKIP  the residency transfer needs the host backend (%s here)\n",
                    device_backend_name());
        return;
    }
    Registry reg;
    REQUIRE_EQ(reg.load_plugin(rows_plugin(), 0), RAD_OK);
    CHECK_OK(reg.check_complete());
    ResidencyFixture f;
    CHECK_OK(f.build(512, 64));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(0, 0)));
    RadStream compute = nullptr;
    REQUIRE_EQ(rad_stream_create(&compute, 0), RAD_OK);
    const std::vector<int32_t> groups{ G_RES };
    TierExec lead, follower;
    CHECK_OK(lead.configure(&reg, &f.kv, &t, groups, TierExec::kMoveBatch, 1, 1, /*leader=*/true, 0, 0, nullptr));
    CHECK_OK(follower.configure(&reg, &f.kv, &t, groups, TierExec::kMoveBatch, 1, 1, /*leader=*/false, 0, 0, nullptr));
    CHECK(lead.moves_landed());                      /* nothing issued, nothing to wait for */

    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    resid_fill(f.kv, G_RES);
    std::vector<BlockMove> down;
    CHECK_OK(f.kv.residency_demote(1, G_RES, 2, 8, &down, /*max=*/TierExec::kMoveBatch));
    REQUIRE_EQ(down.size(), (size_t)TierExec::kMoveBatch);
    CHECK_OK(lead.issue_moves(down, compute, /*to_host=*/true));
    CHECK_OK(follower.issue_moves(down, compute, /*to_host=*/true));
    lead.quiesce();
    for (int i = 0; i < 1000 && !lead.moves_landed(); ++i) {}
    CHECK(lead.moves_landed());
    CHECK(follower.moves_landed() || (follower.quiesce(), follower.moves_landed()));

    /* A list the manager never handed out is the leader's to refuse; a follower has no record to
     * check it against. */
    std::vector<BlockMove> invented = down;
    for (BlockMove& m : invented) m.host_slot += 40;
    CHECK_EQ(lead.issue_moves(invented, compute, true), (int)RAD_E_STATE);

    CHECK_OK(rad_stream_sync(compute));
    rad_stream_destroy(compute);
}

/* The pager's counter is one int64 per rank, allocated once at a fixed address (the batch carries
 * it from the first step), zero at the start, and read back by a poll. */
TEST(the_pagers_counter_starts_at_zero_keeps_its_address_and_reads_back_what_the_card_added) {
    if (!device_is_host()) {
        std::printf("  SKIP  the residency transfer needs the host backend (%s here)\n",
                    device_backend_name());
        return;
    }
    Registry reg;
    REQUIRE_EQ(reg.load_plugin(rows_plugin(), 0), RAD_OK);
    CHECK_OK(reg.check_complete());
    ResidencyFixture f;
    CHECK_OK(f.build(512, 64));
    IdleTiers t;
    CHECK_OK(t.configure(tier_config(0, 0)));
    RadStream compute = nullptr;
    REQUIRE_EQ(rad_stream_create(&compute, 0), RAD_OK);
    TierExec x;
    CHECK_OK(x.configure(&reg, &f.kv, &t, { G_RES }, TierExec::kMoveBatch, 1, 1, true, 0, 0, nullptr));
    int64_t* c = x.paged_counter();
    REQUIRE(c != nullptr);
    CHECK(x.paged_counter() == c);
    CHECK_EQ(x.paged_in(), 0ll);
    /* The manager only carries the address to the batch. */
    f.kv.set_residency_paged(c);
    CHECK(f.kv.residency_paged() == c);

    /* A first move creates the transfer stream the readback uses. On the host backend the card's
     * memory is host memory, so adding to the counter stands for the kernel's atomic add. */
    CHECK_OK(f.kv.add_sequence(1));
    CHECK_OK(f.kv.ensure(1, 16 * 40));
    std::vector<BlockMove> down;
    CHECK_OK(f.kv.residency_demote(1, G_RES, 2, 8, &down, TierExec::kMoveBatch));
    CHECK_OK(x.issue_moves(down, compute, true));
    x.quiesce();
    *c += 5;
    CHECK_EQ(x.paged_in(), 5ll);
    *c += 7;
    CHECK_EQ(x.paged_in(), 12ll);
    CHECK_OK(rad_stream_sync(compute));
    rad_stream_destroy(compute);
}

/* What /stats reports: the binding hands the poll the card's count, and with no pager (no reader)
 * it is zero, which is every server without --kv-residency-mib. */
TEST(the_manager_binding_reports_the_pagers_count_only_when_a_reader_is_installed) {
    ResidencyFixture f;
    CHECK_OK(f.build(512, 64));
    KVBinding<KVManager> bind(&f.kv);
    CHECK_EQ(bind.residency().paged_in, 0ll);
    int64_t counted = 41;
    bind.paged_read = [&] { return counted; };
    CHECK_EQ(bind.residency().paged_in, 41ll);
    counted = 42;
    CHECK_EQ(bind.residency().paged_in, 42ll);
}

RAD_TEST_MAIN()

/* A streamed-only op exists only when the architecture's capability table says it can stream AND
 * the deployment asked for a ring (core/kvstream/bringup.cpp, consulted from rad_op_declared). One
 * binary serves both runs, so this is the only place the two questions meet; no model file carries
 * the flag, and an architecture that cannot stream keeps the graph it had before streaming existed. */
TEST(a_streamed_op_exists_only_for_a_capable_architecture_and_a_ring) {
    RadBuildCtx ctx{};
    ctx.kv_stream_ok = 1;
    ctx.kv_resident_blocks = 32768;
    /* capable and asked: the ops declare as they always did */
    CHECK(!kvstream_op_gated(ctx, "kv_store_streamed"));
    CHECK(!kvstream_op_gated(ctx, "qk_norm_rope_gate_kv_store_streamed"));
    CHECK(!kvstream_op_gated(ctx, "kv_resolve"));
    /* nothing that works without a ring is ever gated */
    CHECK(!kvstream_op_gated(ctx, "kv_store"));
    CHECK(!kvstream_op_gated(ctx, "qk_norm_rope_gate_kv_store"));
    CHECK(!kvstream_op_gated(ctx, "attn_paged"));
    CHECK(!kvstream_op_gated(ctx, "qsa_select"));
    CHECK(!kvstream_op_gated(ctx, nullptr));

    /* no flag: the ops that exist only for a ring are gone */
    ctx.kv_resident_blocks = 0;
    CHECK(kvstream_op_gated(ctx, "kv_store_streamed"));
    CHECK(kvstream_op_gated(ctx, "kv_resolve"));
    CHECK(!kvstream_op_gated(ctx, "kv_store"));

    /* an architecture with no streamed path declares none even when the flag asks */
    ctx.kv_resident_blocks = 32768;
    ctx.kv_stream_ok = 0;
    CHECK(kvstream_op_gated(ctx, "kv_store_streamed"));
    CHECK(kvstream_op_gated(ctx, "qk_norm_rope_gate_kv_store_streamed"));
    CHECK(kvstream_op_gated(ctx, "kv_resolve"));
    CHECK(!kvstream_op_gated(ctx, "kv_store"));
    CHECK(!kvstream_op_gated(ctx, "attn_paged"));
}
