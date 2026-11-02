/* tests/dashboard_test.cpp -- what the dashboard and its JSON routes show.
 *
 * The page at `/` polls /stats and /sessions and reads dozens of their fields by name; a field
 * renamed on either side blanks a panel with no error anywhere. So the routes are checked for the
 * values a populated engine gives them, and the page's own script is read for every field it
 * takes from them, each of which a route must emit. The KV figures behind both come from
 * SchedulerBridge's aggregation over the scheduler's groups, checked here on a hybrid model's
 * groups, where summing or picking the wrong group goes wrong silently.
 */
#include "server_fakes.h"
#include "rad_internal.h"   /* rad_mono_ns: a synthetic bringup start */

#include <set>

namespace {

/* A scheduler with a live request table, stored sessions and idle-tier counters. */
struct StatsScheduler : FakeScheduler {
    int requests(std::vector<RequestStat>* out) const override {
        RequestStat r;
        r.id = 41;
        r.state = "decode";
        r.prompt_tokens = 900;
        r.computed_tokens = 900;
        r.cached_tokens = 512;
        r.output_tokens = 33;
        r.max_tokens = 4096;
        r.ctx_tokens = 933;
        r.kv_blocks = 120;
        r.kv_tokens = 960;
        r.n_draft = 3;
        r.slot = 2;
        r.age_s = 4.5;
        r.ttft_s = 0.25;
        r.decode_tps = 61.0;
        out->assign(1, r);
        return 1;
    }
    int64_t session_res_demoted = 0;
    int sessions(std::vector<SessionStat>* out, SessionOccupancy* occ) const override {
        SessionStat s;
        s.id = 7;
        s.model = "m";
        s.label = "conv-7";
        s.n_tokens = 5000;
        s.bytes = 1 << 20;
        s.idle_s = 12.0;
        s.age_s = 300.0;
        s.on_device = 40;
        s.on_host = 10;
        s.snapshots = 2;
        s.snap_bytes = 4096;
        s.resume_tokens = 4096;
        s.moving = true;
        s.res_demoted = session_res_demoted;
        out->assign(1, s);
        occ->host_bytes = 8 << 20;
        occ->host_cap_bytes = 64 << 20;
        return 1;
    }
    /* The one stored conversation is "conv-7" (engine id 7); "busy" has a request in flight. */
    std::vector<std::pair<std::string, int64_t>> released;
    int release_session(const std::string& label, uint64_t id, int64_t keep_tokens,
                        SessionReleased* out) override {
        if (label == "busy") return RAD_E_STATE;
        if (label != "conv-7" && id != 7) return RAD_E_NOTFOUND;
        released.push_back({ label.empty() ? "id" : label, keep_tokens });
        out->sessions = 1;
        out->entries = 48 - keep_tokens / 16;
        out->tokens = out->entries * 16;
        out->device_bytes = out->entries * 4096;
        out->host_bytes = out->entries * 2048;
        out->snapshots = 2;
        out->kept_tokens = keep_tokens;
        out->kept_shared = 2;
        return RAD_OK;
    }
    TierCounters tier_counters() const override {
        TierCounters c;
        c.to_host = 3; c.dropped = 2; c.promoted = 1; c.host_drops = 4; c.failures = 0;
        c.host_bytes = 8 << 20; c.ck_bytes = 2048; c.ck_slots = 16;
        c.restores = 5; c.restore_ns = 2000000; c.flushes = 6; c.flush_ns = 3000000;
        c.host_on = true;
        return c;
    }
};

/* Every value a snapshot carries, set, so every optional field is emitted. */
void populated(LiveSnapshot& s) {
    CardStat c;
    c.index = 0;
    c.name = "R9700";
    c.util = 0.5;
    c.vram_used = 10LL << 30;
    c.vram_total = 32LL << 30;
    c.power_w = 200;
    c.temp_c = 60;
    s.cards = { c };
    s.experts.n_layers = 2;
    s.experts.n_experts = 3;
    s.experts.tier = { 0, 1, 2, 0, 0, 1 };
    for (int t = 0; t < kLiveTiers; ++t) { s.experts.units[t] = t + 1; s.experts.bytes[t] = (t + 1) << 20; }
    s.experts.n_units = 6;
    s.experts.total_bytes = 6 << 20;
    s.experts.movable = 4;
    s.experts.layer_units = { 3, 3 };
    s.experts.layer_resident = { 2, 3 };
    s.experts.promotions = 9; s.experts.demotions = 8; s.experts.readmits = 2;
    s.experts.distinct_promoted = 7; s.experts.dispatches = 20; s.experts.refused = 3;
    s.experts.no_candidate = 1; s.experts.routed = 100; s.experts.routed_resident = 80;
    s.experts.flex_bytes = 1 << 20; s.experts.flex_capacity = 2 << 20;
    s.experts.heat_enabled = true;
    s.link.present = true;
    s.link.world = 2;
    s.link.ar_calls = 50; s.link.ar_bytes = 5000; s.link.h2d_bytes = 900; s.link.d2h_bytes = 800;
    s.link.ssd_reads = 4; s.link.ssd_bytes = 400; s.link.stream_bytes = 700;
    s.link.prefetched = 2; s.link.prefetch_bytes = 300; s.link.staged_bytes = 200;
    s.link.kv_streaming = true; s.link.kv_stream_active = true; s.link.kv_stream_bytes = 600;
    s.passes.present = true;
    s.passes.played = 10; s.passes.recorded = 2;
}

struct Dash {
    StatsScheduler sched;
    FakeTokenizer  tok;
    std::unique_ptr<Server> srv;
    explicit Dash(bool with_graph) {
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        d.live = populated;
        ServerOptions o;
        o.model_id = "dash";
        o.max_seqs = 4;
        if (with_graph) o.graph_json = [] { return std::string(R"({"order":[]})"); };
        sched.m.running = 1;
        sched.m.waiting = 2;
        sched.m.kv_tokens_used = 1000;
        sched.m.kv_tokens_total = 4000;
        sched.m.kv_tokens_cached = 300;
        sched.m.kv_tokens_max = 6000;
        sched.m.prefix_ckpt_lru = 3;
        sched.m.hit_cut_no_snapshot = 2;
        sched.m.hit_cut_tokens = 9000;
        sched.m.hit_cut_resumed_below = 4;
        sched.m.prefix_restore_cuts = 1;
        sched.m.draft_depth_seen = 2;
        sched.m.draft_pos_reached[0] = 10; sched.m.draft_pos_accepted[0] = 8;
        sched.m.draft_pos_reached[1] = 8;  sched.m.draft_pos_accepted[1] = 2;
        srv.reset(new Server(d, o));
    }
    json get(const char* path) {
        HttpResponse r = srv->router().call("GET", path);
        CHECK_EQ(r.status, 200);
        return json::parse(r.body, nullptr, false);
    }
};

/* The identifiers `var.<name>` that `text` reads, and the string arguments of `calls("<name>")`. */
std::set<std::string> fields(const std::string& text, const std::string& var) {
    std::set<std::string> out;
    const std::string pat = var + ".";
    for (size_t at = text.find(pat); at != std::string::npos; at = text.find(pat, at + 1)) {
        if (at > 0) {
            const char b = text[at - 1];
            if (std::isalnum((unsigned char)b) || b == '_' || b == '.' || b == '$') continue;
        }
        size_t e = at + pat.size();
        while (e < text.size() && (std::isalnum((unsigned char)text[e]) || text[e] == '_')) ++e;
        if (e > at + pat.size()) out.insert(text.substr(at + pat.size(), e - at - pat.size()));
    }
    return out;
}

std::set<std::string> called(const std::string& text, const std::string& fn) {
    std::set<std::string> out;
    const std::string pat = fn + "(\"";
    for (size_t at = text.find(pat); at != std::string::npos; at = text.find(pat, at + 1)) {
        if (at > 0 && (std::isalnum((unsigned char)text[at - 1]) || text[at - 1] == '_')) continue;
        const size_t b = at + pat.size(), e = text.find('"', b);
        out.insert(text.substr(b, e - b));
    }
    return out;
}

/* The page's script from `from` up to the line that closes the function starting there. */
std::string function_body(const std::string& page, const std::string& from) {
    const size_t at = page.find(from);
    if (at == std::string::npos) return std::string();
    const size_t end = page.find("\n}\n", at);
    return page.substr(at, end == std::string::npos ? std::string::npos : end - at);
}

void each_field_is_emitted(const std::set<std::string>& names, const json& o, const char* what) {
    for (const std::string& n : names) {
        const bool there = o.is_object() && o.contains(n);
        CHECK(there);
        if (!there) fprintf(stderr, "    the page reads %s.%s, which the route does not emit\n", what, n.c_str());
    }
}

}  /* namespace */

/* ================================================================== the routes */

/* /stats carries the scheduler's figures, the live request table, the draft positions with their
 * reach, and the engine's snapshot (cards, experts by named tier, link, passes); /graph is the
 * provider's JSON, or a 501 naming the missing provider. */
TEST(stats_carries_the_engine_the_requests_and_the_snapshot) {
    Dash d(false);
    const json s = d.get("/stats");
    CHECK_EQ(s.value("model", ""), std::string("dash"));
    CHECK_EQ(s.value("queue_cap", (int64_t)0), (int64_t)32);
    CHECK_EQ(s.value("in_flight", (int64_t)-1), (int64_t)0);
    CHECK_EQ(s.value("running", (int64_t)0), (int64_t)1);
    CHECK_EQ(s.value("kv_tokens_cached", (int64_t)0), (int64_t)300);
    CHECK_EQ(s.value("kv_tokens_max", (int64_t)0), (int64_t)6000);
    REQUIRE(s["draft_positions"].size() == 2);
    CHECK_NEAR(s["draft_positions"][1]["rate"].get<double>(), 0.25, 1e-12);
    CHECK_EQ(s["draft_positions"][1]["reached"].get<int64_t>(), (int64_t)8);
    REQUIRE(s["requests"].size() == 1);
    const json& r = s["requests"][0];
    CHECK_EQ(r.value("state", ""), std::string("decode"));
    CHECK_EQ(r.value("kv_tokens", (int64_t)0), (int64_t)960);
    CHECK_NEAR(r.value("ttft_s", -1.0), 0.25, 1e-12);
    REQUIRE(s["cards"].size() == 1);
    CHECK_EQ(s["cards"][0].value("name", ""), std::string("R9700"));
    REQUIRE(s["experts"]["tiers"].size() == (size_t)kTierCount);
    for (int t = 0; t < kTierCount; ++t) {
        CHECK_EQ(s["experts"]["tiers"][t].value("name", ""), std::string(tier_name((Tier)t)));
        CHECK_EQ(s["experts"]["tiers"][t].value("units", (int64_t)0), (int64_t)(t + 1));
    }
    CHECK_EQ(s["link"].value("stream_bytes", (int64_t)0), (int64_t)700);
    CHECK_EQ(s["link"].value("kv_stream_bytes", (int64_t)0), (int64_t)600);
    CHECK(s["link"].value("kv_streaming", false));
    CHECK(s["link"].value("kv_stream_active", false));
    CHECK_EQ(s["passes"].value("played", (int64_t)0), (int64_t)10);
    CHECK_EQ(s["ckpt_evictions"].value("store_full", (int64_t)0), (int64_t)3);
    CHECK_EQ(s["hit_cuts"].value("no_snapshot", (int64_t)0), (int64_t)2);
    CHECK_EQ(s["hit_cuts"].value("resumed_below", (int64_t)0), (int64_t)4);
    CHECK_EQ(s.value("hit_cut_tokens", (int64_t)0), (int64_t)9000);

    CHECK_EQ(d.srv->router().call("GET", "/graph").status, 501);
    Dash g(true);
    CHECK_EQ(g.get("/graph").value("order", json()).size(), (size_t)0);
}

/* The residency counters are on /stats and, per session, on /sessions, under their final names,
 * and they follow the scheduler: the second poll after a sweep reads larger numbers. The
 * per-session ones are the session's own, so two sessions may both include a shared block. */
TEST(stats_and_sessions_report_residency_counters_and_they_move) {
    Dash d(false);
    const json a = d.get("/stats")["kv_residency"];
    for (const char* k : { "demoted", "promoted", "skipped", "misses", "host_bytes", "cap_bytes",
                           "freed_bytes", "link_bytes", "scratch_bytes" })
        CHECK(a.contains(k));
    CHECK_EQ(a.value("demoted", (int64_t)-1), (int64_t)0);

    d.sched.m.res_demoted = 30;   d.sched.m.res_promoted = 2;  d.sched.m.res_skipped = 4;
    d.sched.m.res_link_bytes = 65536; d.sched.m.res_freed_bytes = 61440;
    d.sched.m.res_host_bytes = 61440; d.sched.m.res_scratch_bytes = 8192;
    const json b = d.get("/stats")["kv_residency"];
    CHECK_EQ(b.value("demoted", (int64_t)0), (int64_t)30);
    CHECK_EQ(b.value("promoted", (int64_t)0), (int64_t)2);
    CHECK_EQ(b.value("skipped", (int64_t)0), (int64_t)4);
    CHECK_EQ(b.value("link_bytes", (int64_t)0), (int64_t)65536);
    CHECK_EQ(b.value("freed_bytes", (int64_t)0), (int64_t)61440);
    CHECK_EQ(b.value("host_bytes", (int64_t)0), (int64_t)61440);
    CHECK_EQ(b.value("scratch_bytes", (int64_t)0), (int64_t)8192);

    d.sched.session_res_demoted = 12;
    const json s = d.get("/sessions")["sessions"][0]["kv_residency"];
    CHECK_EQ(s.value("demoted", (int64_t)0), (int64_t)12);
    for (const char* k : { "promoted", "skipped", "link_bytes", "freed_bytes" }) CHECK(s.contains(k));
}

/* /sessions lists each stored conversation with where its blocks are, the pool as the card holds
 * it (each block once), the tiers' counters and the snapshot store. */
TEST(sessions_lists_each_conversation_and_what_the_tiers_did) {
    Dash d(false);
    const json s = d.get("/sessions");
    REQUIRE(s["sessions"].size() == 1);
    const json& e = s["sessions"][0];
    CHECK_EQ(e.value("label", ""), std::string("conv-7"));
    CHECK_EQ(e.value("on_device", (int64_t)0), (int64_t)40);
    CHECK_EQ(e.value("resume_tokens", (int64_t)0), (int64_t)4096);
    CHECK_EQ(s["blocks_sum_overlapping"].value("host", (int64_t)0), (int64_t)10);
    CHECK_EQ(s["pool"].value("tokens_used", (int64_t)0), (int64_t)1000);
    CHECK_EQ(s["pool"].value("unshared_tokens", (int64_t)0), (int64_t)300);
    CHECK_EQ(s["moved"].value("to_host", (int64_t)0), (int64_t)3);
    CHECK_EQ(s["snapshot_store"].value("bytes_each", (int64_t)0), (int64_t)2048);
    CHECK_NEAR(s["stall"].value("restore_ms", 0.0), 2.0, 1e-9);
    CHECK_EQ(s["occupancy"].value("host_cap_bytes", (int64_t)0), (int64_t)(64 << 20));
    CHECK(s.value("enabled", false));
    CHECK_EQ(s["linear"]["evictions"].value("store_full", (int64_t)0), (int64_t)3);
    CHECK_EQ(s["linear"]["hit_cuts"].value("no_snapshot", (int64_t)0), (int64_t)2);
    CHECK_EQ(s["linear"]["hit_cuts"].value("resumed_below", (int64_t)0), (int64_t)4);
    CHECK_EQ(s["moved"].value("restore_cuts", (int64_t)0), (int64_t)1);
    CHECK_EQ(s["linear"].value("hit_cut_tokens", (int64_t)0), (int64_t)9000);
}

/* DELETE /sessions forgets one stored conversation by its label or engine id and answers with
 * what went and what stayed; an unknown session is 404, one with a request in flight 409, a
 * malformed count 400, and an engine without a prefix cache 501. */
TEST(a_session_is_released_by_label_or_id_and_the_answer_says_what_went) {
    Dash d(false);
    HttpResponse r = d.srv->router().call("DELETE", "/sessions/conv-7");
    CHECK_EQ(r.status, 200);
    json o = json::parse(r.body, nullptr, false);
    CHECK_EQ(o.value("label", ""), std::string("conv-7"));
    CHECK_EQ(o.value("entries", (int64_t)0), (int64_t)48);
    CHECK_EQ(o.value("tokens", (int64_t)0), (int64_t)768);
    CHECK_EQ(o.value("snapshots", (int64_t)0), (int64_t)2);
    CHECK_EQ(o["kept"].value("shared", (int64_t)0), (int64_t)2);
    CHECK_EQ(o["kept"].value("tokens", (int64_t)0), (int64_t)0);
    REQUIRE_EQ(d.sched.released.size(), 1u);
    CHECK_EQ(d.sched.released[0].second, (int64_t)0);

    /* keep_tokens reaches the scheduler and comes back in the answer. */
    r = d.srv->router().call("DELETE", "/sessions/conv-7?keep_tokens=160");
    CHECK_EQ(r.status, 200);
    o = json::parse(r.body, nullptr, false);
    CHECK_EQ(o.value("keep_tokens", (int64_t)0), (int64_t)160);
    CHECK_EQ(o.value("entries", (int64_t)0), (int64_t)38);
    CHECK_EQ(o["kept"].value("tokens", (int64_t)0), (int64_t)160);
    REQUIRE_EQ(d.sched.released.size(), 2u);
    CHECK_EQ(d.sched.released[1].second, (int64_t)160);

    /* By engine id, and by a label the path cannot carry. */
    r = d.srv->router().call("DELETE", "/sessions?id=7");
    CHECK_EQ(r.status, 200);
    o = json::parse(r.body, nullptr, false);
    CHECK_EQ(o.value("id", (int64_t)0), (int64_t)7);
    CHECK(!o.contains("label"));
    r = d.srv->router().call("DELETE", "/sessions?label=conv-7");
    CHECK_EQ(r.status, 200);

    CHECK_EQ(d.srv->router().call("DELETE", "/sessions/conv-8").status, 404);
    CHECK_EQ(d.srv->router().call("DELETE", "/sessions?id=8").status, 404);
    CHECK_EQ(d.srv->router().call("DELETE", "/sessions/busy").status, 409);
    CHECK_EQ(d.srv->router().call("DELETE", "/sessions").status, 400);
    CHECK_EQ(d.srv->router().call("DELETE", "/sessions?id=seven").status, 400);
    CHECK_EQ(d.srv->router().call("DELETE", "/sessions/conv-7?keep_tokens=-1").status, 400);
    CHECK_EQ(d.sched.released.size(), 4u);

    /* The stored-sessions view carries the counters beside the expiry ones. */
    const json s = d.get("/sessions");
    CHECK(s.contains("released"));
    CHECK(s["released"].contains("entries"));
    CHECK(s["released"].contains("snapshots"));
}

TEST(a_session_release_without_a_prefix_cache_is_not_implemented) {
    FakeScheduler sched;      /* the base: no release_session */
    FakeTokenizer tok;
    Deps dd;
    dd.sched = &sched;
    dd.tok = &tok;
    ServerOptions o;
    o.model_id = "dash";
    Server srv(dd, o);
    CHECK_EQ(srv.router().call("DELETE", "/sessions/conv-7").status, 501);
}

/* ================================================================== the page's field contract */

/* Every field the page's script reads from /stats (in its update and the request table) and from
 * /sessions (its loader and the session rows) is one the routes emit for a populated engine. A
 * rename on either side otherwise blanks a panel and reports nothing. */
/* /stats reports the measured startup time, the cost side of any decision to restart; the page
 * does not read it. A test starts in microseconds, where a real note_ready() would compute 0 (the
 * same as "never called"), so a synthetic start exercises the arithmetic through the real route. */
TEST(stats_reports_how_long_bringup_took_once_the_server_listens) {
    Dash d(false);
    /* Before the server has listened, there is no measured cost to report. */
    CHECK_EQ(d.get("/stats").value("startup_s", (int64_t)-1), (int64_t)0);

    d.srv->note_ready(rad_mono_ns() - 47ull * 1000000000ull);
    CHECK_EQ(d.get("/stats").value("startup_s", (int64_t)-1), (int64_t)47);

    /* A start in the future is not a negative cost: the figure is a duration, and a page that
     * divided by it would otherwise be handed a sign flip. */
    d.srv->note_ready(rad_mono_ns() + 10ull * 1000000000ull);
    CHECK_EQ(d.get("/stats").value("startup_s", (int64_t)-1), (int64_t)47);

    /* 0 means "not yet", so it must not overwrite a measured figure. */
    d.srv->note_ready(0);
    CHECK_EQ(d.get("/stats").value("startup_s", (int64_t)-1), (int64_t)47);
}

/* The pooled acceptance ring's element shape is a contract between two functions of the page,
 * and no browser runs here: accWindow returns {ratio, dr, da}, and acceptTriple reads the ring's
 * entries as `d` and `last`. A field name it reads that the ring does not carry is undefined in
 * JavaScript, and the pooled figures render as NaN. */
TEST(the_pooled_acceptance_ring_reads_the_names_accWindow_writes) {
    Dash d(false);
    const std::string p = d.srv->router().call("GET", "/").body;
    const std::string aw = function_body(p, "function accWindow(");
    const std::string at = function_body(p, "function acceptTriple(");
    REQUIRE(!aw.empty() && !at.empty());
    /* what goes into the ring */
    CHECK(aw.find("dr: dr") != std::string::npos);
    CHECK(aw.find("da: da") != std::string::npos);
    /* what comes out of it: every `d.<field>` and `last.<field>` must be a ring name */
    const std::set<std::string> ring = {"dr", "da"};
    for (const std::string& n : fields(at, "d")) {
        CHECK(ring.count(n) == 1);
        if (!ring.count(n))
            fprintf(stderr, "    acceptTriple reads d.%s, which the ring does not carry\n", n.c_str());
    }
    for (const std::string& n : fields(at, "last")) {
        CHECK(ring.count(n) == 1);
        if (!ring.count(n))
            fprintf(stderr, "    acceptTriple reads last.%s, which the ring does not carry\n", n.c_str());
    }
    /* only polls that drafted enter the ring */
    CHECK(aw.find("dr <= 0") != std::string::npos);
}

TEST(every_field_the_page_reads_is_one_the_routes_emit) {
    Dash d(false);
    HttpResponse page = d.srv->router().call("GET", "/");
    REQUIRE(page.status == 200 && has(page.body, "function update(s, dt)"));
    const std::string& p = page.body;
    const json stats = d.get("/stats");
    const json sessions = d.get("/sessions");

    const std::string update = function_body(p, "function update(s, dt)");
    const std::string batch = function_body(p, "function renderBatch(list, s)");
    REQUIRE(!update.empty() && !batch.empty());
    std::set<std::string> top = fields(update, "s");
    for (const std::string& n : fields(batch, "s")) top.insert(n);
    CHECK(top.size() > 15);
    each_field_is_emitted(top, stats, "stats");
    each_field_is_emitted(fields(update, "ex"), stats["experts"], "experts");
    each_field_is_emitted(fields(update, "t"), stats["experts"]["tiers"][0], "experts.tiers[]");
    std::set<std::string> link = fields(update, "lk");
    for (const char* fn : { "d", "pre", "dP" })
        for (const std::string& n : called(update, fn)) link.insert(n);
    each_field_is_emitted(link, stats["link"], "link");
    each_field_is_emitted(fields(update, "c"), stats["cards"][0], "cards[]");
    each_field_is_emitted(fields(update, "p"), stats["draft_positions"][0], "draft_positions[]");

    /* The request table: renderBatch through renderRequests. */
    const size_t rb = p.find("function renderBatch("), up = p.find("function update(s, dt)");
    REQUIRE(rb != std::string::npos && up > rb);
    each_field_is_emitted(fields(p.substr(rb, up - rb), "r"), stats["requests"][0], "requests[]");

    /* The sessions view: its loader, and the rows from reuseOf through sessRow. */
    const std::string load = function_body(p, "async function loadSessions()");
    REQUIRE(!load.empty());
    each_field_is_emitted(fields(load, "s"), sessions, "sessions");
    each_field_is_emitted(fields(load, "mv"), sessions["snapshot_store"], "snapshot_store");
    each_field_is_emitted(fields(load, "ln"), sessions["linear"], "linear");
    each_field_is_emitted(fields(load, "pool"), sessions["pool"], "pool");
    each_field_is_emitted(fields(load, "occ"), sessions["occupancy"], "occupancy");
    const size_t ro = p.find("const reuseOf"), lo = p.find("async function loadSessions()");
    REQUIRE(ro != std::string::npos && lo > ro);
    std::set<std::string> rows = fields(p.substr(ro, lo - ro), "r");
    for (const std::string& n : fields(load, "r")) rows.insert(n);
    each_field_is_emitted(rows, sessions["sessions"][0], "sessions[]");
}

/* ================================================================== the bridge's KV figures */

/* SchedulerBridge's aggregation over a hybrid model's groups: two paged groups share one token
 * axis, so the token figures are the fuller one's, not their sum; a linear state has no token
 * axis and adds only bytes; a windowed group's carve bounds lookback, not fit, so it neither
 * binds the token figures (a drafter's window sits full for the server's life) nor bounds
 * kv_tokens_max. Utilisation is the maximum over every group, bytes their sum, the cache's
 * unshared share is clamped to what is used, and the draft depth to the table's size. */
TEST(the_bridge_reads_the_binding_paged_group_and_sums_only_bytes) {
    rad::SchedMetrics m;
    m.n_kv_groups = 4;
    auto group = [&](int i, int64_t bt, int64_t total, int64_t free_, bool window, int64_t bytes) {
        m.kv[i].block_tokens = bt;
        m.kv[i].blocks_total = total;
        m.kv[i].blocks_free = free_;
        m.kv[i].utilisation = total ? (float)(1.0 - (double)free_ / (double)total) : 0.0f;
        m.kv[i].window = window;
        m.kv[i].bytes_total = bytes;
        m.kv[i].bytes_used = bytes / 2;
        m.kv[i].bytes_carved = bytes * 2;
    };
    group(0, 16, 100, 40, false, 1000);   /* attention, 60% */
    group(1, 16, 100, 10, false, 200);    /* a second paged group on the same axis, 90% */
    group(2, 0, 8, 0, false, 500);        /* linear state, full, no token axis */
    group(3, 16, 4, 0, true, 64);         /* a drafter's window, always full */
    m.prefix_held_tokens = 5000;
    m.hit_cut_resumed_below = 6;
    m.prefix_restore_cuts = 7;
    m.draft_depth_seen = 70;
    for (int j = 0; j < 64; ++j) { m.draft_pos_reached[j] = 100 - j; m.draft_pos_accepted[j] = j; }

    const server::SchedMetrics o = bridge_metrics(m);
    CHECK_NEAR(o.kv_util, 1.0, 1e-6);
    CHECK_EQ(o.kv_tokens_total, (int64_t)(100 * 16));
    CHECK_EQ(o.kv_tokens_used, (int64_t)(90 * 16));
    CHECK_EQ(o.kv_tokens_max, (int64_t)(100 * 16));
    CHECK_EQ(o.kv_bytes_total, (int64_t)(1000 + 200 + 500 + 64));
    CHECK_EQ(o.kv_bytes_used, (int64_t)(500 + 100 + 250 + 32));
    CHECK_EQ(o.kv_bytes_carved, (int64_t)(2000 + 400 + 1000 + 128));
    CHECK_EQ(o.kv_tokens_cached, o.kv_tokens_used);
    CHECK_EQ(o.draft_depth_seen, server::SchedMetrics::MAX_DRAFT_POS);
    CHECK_EQ(o.draft_pos_reached[63], (int64_t)37);
    CHECK_EQ(o.hit_cut_resumed_below, (int64_t)6);
    CHECK_EQ(o.prefix_restore_cuts, (int64_t)7);

    /* The fuller group binds wherever it sits in the list. */
    group(0, 16, 200, 10, false, 1000);
    const server::SchedMetrics p = bridge_metrics(m);
    CHECK_EQ(p.kv_tokens_total, (int64_t)(200 * 16));
    CHECK_EQ(p.kv_tokens_used, (int64_t)(190 * 16));
    CHECK_EQ(p.kv_tokens_max, (int64_t)(100 * 16));
}

RAD_TEST_MAIN()
