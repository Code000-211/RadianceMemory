/* core/server/dashboard.cpp -- the web view of a running engine.
 *
 * `/` serves the page and `/stats` the JSON it polls (plus `/sessions` and `/graph`). No
 * framework: the page is a string literal in this file, so there is no asset directory, no build
 * step that can go stale against the binary, and no network fetch; the engine runs on machines
 * with no route to a CDN.
 *
 * /metrics serves Prometheus, but answering "what is this process doing right now" from it needs a
 * scrape config, a time-series database and a Grafana dashboard. This is the terminal view's
 * snapshot (liveview.h, spec §16) in a browser, viewable from another machine.
 *
 * The per-position draft panel shows what no pooled metric does: one late draft position
 * collapsing while the pooled acceptance rate stays plausible.
 */
#include "server.h"
#include "oai.h"

#include "rad_internal.h"

#include <nlohmann/json.hpp>

#include <vector>

namespace rad {
namespace server {

/* ================================================================== /stats */
/* One polled object: the scheduler's metrics, the per-position draft table and whatever the
 * engine's snapshot function provides. Not a stable public API; automation should scrape /metrics.
 *
 * Derived rates are the page's job: they are counter differences over wall time, and only the
 * browser holds both ends of its poll interval. Computing them here would need per-client state. */

/* ------------------------------------------------------------------ /admin/control-vector
 *
 * The control vector's runtime switch (core/cvec.h). Turning it on for the next
 * request costs a boolean: the direction table was uploaded at startup and the per-request mask
 * is a derived buffer the core fills every step, so there is nothing to rebuild and nothing to
 * reload. The engine's own log records the load; this records the flip, because a deployment's
 * answer to "was that text produced with the vector" should not depend on a browser's memory.
 *
 * GET reports the state (and is what the Model view renders); POST takes {"enabled": true|false}
 * and answers with the same object. A deployment with no vector loaded answers 404 on both: it has
 * no such feature, and its requests refuse `control_vector: true` by name. */
HttpResponse Server::handle_control_vector(const HttpRequest& req) {
    /* Keyless, the vector and its switch belong to the machine: a rebinding page, a page on
     * any origin a localhost proxy fronts, or a plain remote client reads neither the state
     * nor the 404-that-means-no-vector. With --api-key both verbs are behind the key anyway. */
    if (opt_.api_key.empty() && !local_operator(req)) {
        ApiError e;
        e.status  = 403;
        e.type    = "permission_error";
        e.message = "without --api-key, the control vector is read and its default changed only "
                    "from a loopback client naming this machine; start the server with --api-key "
                    "to reach it from elsewhere";
        return error_response(e, req.peer);
    }
    const bool have = deps_.control_vector.available && deps_.control_vector.available();
    if (!have) {
        /* 404, like a model the server does not serve: the route exists, this deployment has nothing
         * behind it, and that is what the page tests for before hiding the panel. */
        ApiError e;
        e.status  = 404;
        e.type    = "not_found_error";
        e.message = "this deployment found no control vector for its model; there is nothing to "
                    "switch";
        return error_response(e, req.peer);
    }

    if (req.method == "POST") {
        bool over = false, deep = false;
        json body = parse_json_bounded(req.body.empty() ? "{}" : req.body, 0, &over, &deep);
        if (body.is_discarded()) return error_response(err_bad("body", "not JSON"), req.peer);
        if (over || deep)
            return error_response(err_bad("body", "expected {\"enabled\": true|false} and "
                                                  "nothing deeper than that"), req.peer);
        const auto it = body.find("enabled");
        if (it == body.end() || !it->is_boolean())
            return error_response(err_bad("enabled", "a boolean is required"), req.peer);
        const bool on = it->get<bool>();
        if (deps_.control_vector.set_default) deps_.control_vector.set_default(on);
        RAD_INFO("control vector: default %s by an admin request%s", on ? "ON" : "OFF",
                 req.peer.empty() ? "" : (" from " + req.peer).c_str());
    }

    metrics_.observe_http(200);
    return HttpResponse::json_body(
        200, deps_.control_vector.json ? deps_.control_vector.json() : std::string("{}"));
}

/* ------------------------------------------------------------------ /sessions
 *
 * A separate route, not a /stats field: /stats is polled every second, while a stored session
 * changes on the scale of its idle thresholds (tens of seconds at the fastest), and the view that
 * reads this is usually off screen. */
HttpResponse Server::handle_sessions(const HttpRequest&) {
    json o;
    std::vector<SessionStat> v;
    SessionOccupancy occ;
    if (deps_.sched) deps_.sched->sessions(&v, &occ);

    int64_t dev = 0, host = 0, bytes = 0;
    int64_t snaps = 0;
    json a = json::array();
    for (const SessionStat& s : v) {
        json e;
        e["id"]        = s.id;
        e["model"]     = s.model;
        if (!s.label.empty()) e["label"] = s.label;
        e["tokens"]    = s.n_tokens;
        e["bytes"]     = s.bytes;
        e["idle_s"]    = s.idle_s;
        e["age_s"]     = s.age_s;
        e["on_device"] = s.on_device;
        e["on_host"]   = s.on_host;
        e["moving"]    = s.moving;
        /* On a hybrid model a session also needs its linear-state snapshots back: a different
         * object size, kept in a separate store in host memory. */
        e["snapshots"]   = s.snapshots;
        e["snap_bytes"]  = s.snap_bytes;
        e["resume_tokens"] = s.resume_tokens;
        /* Overlapping across sessions that share blocks; see SessionStat. */
        e["kv_residency"] = { { "demoted", s.res_demoted }, { "promoted", s.res_promoted },
                              { "skipped", s.res_skipped }, { "link_bytes", s.res_link_bytes },
                              { "freed_bytes", s.res_freed_bytes } };
        a.push_back(std::move(e));
        dev += s.on_device; host += s.on_host;
        snaps += s.snapshots;
        bytes += s.bytes;
    }
    o["sessions"] = std::move(a);
    /* Sessions sharing a prefix share its blocks, so these sums over-count (by more than an order
     * of magnitude with long agent transcripts); the field names say so. */
    o["blocks_sum_overlapping"] = { { "device", dev }, { "host", host } };
    o["snapshots_sum_overlapping"] = snaps;
    o["bytes_sum_overlapping"] = bytes;
    const SchedMetrics sm = deps_.sched ? deps_.sched->metrics() : SchedMetrics{};
    /* What the card actually holds: the pool counts a block once however many sessions name it,
     * which the per-session rows cannot yield. Tokens are the binding group's only; see
     * SchedMetrics::kv_tokens_used for why paged groups are not additive. */
    o["pool"] = { { "tokens_used",     sm.kv_tokens_used },
                  { "tokens_total",    sm.kv_tokens_total },
                  { "unshared_tokens", sm.kv_tokens_cached },
                  { "bytes_used",      sm.kv_bytes_used },
                  { "bytes_total",     sm.kv_bytes_total } };
    const IScheduler::TierCounters c =
        deps_.sched ? deps_.sched->tier_counters() : IScheduler::TierCounters{};
    o["moved"] = { { "to_host", c.to_host }, { "dropped", c.dropped },
                   { "promoted", c.promoted }, { "host_drops", c.host_drops },
                   { "held_back", c.held_back },
                   { "restore_cuts", sm.prefix_restore_cuts },
                   { "failures", c.failures } };
    /* Snapshot store on a hybrid model: bytes per snapshot and slot count. `bytes_each` is 0
     * on a model with no recurrent state. */
    o["snapshot_store"] = { { "bytes_each", c.ck_bytes }, { "slots", c.ck_slots } };
    /* Entries and snapshots removed by --prefix-cache-ttl, from whichever tier held them. */
    o["expired"] = { { "entries", sm.prefix_expired }, { "snapshots", sm.prefix_ckpt_expired } };
    /* And the ones a client gave up on purpose (DELETE /sessions). */
    o["released"] = { { "entries", sm.prefix_released }, { "snapshots", sm.prefix_ckpt_released } };
    /* Step-thread time spent on tier moves; every request waits through it. */
    o["stall"] = { { "restores", c.restores }, { "restore_ms", (double)c.restore_ns / 1e6 },
                   { "flushes", c.flushes }, { "flush_ms", (double)c.flush_ns / 1e6 } };
    /* Whether restored snapshots are served. A hybrid server can move snapshots and still replay
     * from token zero if the restored checkpoint is not one a hit can reach. */
    o["linear"] = { { "hit_rate", sm.linear_hit_rate },
                    { "cached_tokens", sm.linear_cached_tokens },
                    { "checkpoints_written", sm.checkpoints_written },
                    { "evictions", { { "retention", sm.prefix_ckpt_retired },
                                     { "store_full", sm.prefix_ckpt_lru } } },
                    { "refused", sm.prefix_ckpt_refused },
                    { "hit_cuts", { { "no_snapshot", sm.hit_cut_no_snapshot },
                                    { "windowed_group", sm.hit_cut_window },
                                    { "adopt_failed", sm.hit_cut_adopt },
                                    { "resumed_below", sm.hit_cut_resumed_below } } },
                    { "hit_cut_tokens", sm.hit_cut_tokens } };
    /* Whether a tier exists to move sessions to. A session is recorded whether or not it can be
     * demoted, so a non-empty table only says something is held. */
    o["tiers"] = { { "host", c.host_on } };
    /* Off-card bytes, counted once; see SessionOccupancy. */
    o["occupancy"] = { { "host_bytes", occ.host_bytes },
                       { "host_cap_bytes", occ.host_cap_bytes } };
    /* The attention hit rate: the whole answer on a model with no recurrent state. On a hybrid
     * model, read `linear.hit_rate`. */
    o["prefix_hit_rate"] = sm.prefix_hit_rate;
    o["enabled"] = c.host_on;
    return HttpResponse::json_body(200, json_dump(o));
}

/* ------------------------------------------------------------------ DELETE /sessions
 *
 * A client that deletes a conversation, or replaces everything past a prefix of it, knows the
 * rest is dead before the TTL does. Named by the label it sent (`session_id`, or Anthropic's
 * `metadata.user_id`), or by the engine id `/sessions` shows. `keep_tokens` keeps the chain's
 * start. Shared blocks and blocks a live request reads stay; a request of the same conversation
 * in flight is a 409, since its close would publish the chain again. */
HttpResponse Server::handle_session_release(const HttpRequest& req) {
    if (!deps_.sched)
        return error_response(err_not_implemented("the server has no engine attached"), req.peer);

    /* Releasing a session destroys another client's cached conversation: keyless, that is an
     * operator act, not something any page or client that reaches the port may do. */
    if (opt_.api_key.empty() && !local_operator(req))
        return error_response(
            err_forbidden("without --api-key, sessions are released only from a loopback client "
                          "naming this machine; start the server with --api-key to do it from "
                          "elsewhere"),
            req.peer);

    const std::string label = req.param("label");
    uint64_t id = 0;
    const std::string id_s = req.param("id");
    if (!id_s.empty()) {
        if (id_s.find_first_not_of("0123456789") != std::string::npos || id_s.size() > 19)
            return error_response(err_bad("id", "expected the session's engine id, a decimal "
                                                "integer as /sessions reports it"), req.peer);
        id = std::stoull(id_s);
        if (id == 0) return error_response(err_bad("id", "0 names no session"), req.peer);
    }
    if (label.empty() && id == 0)
        return error_response(err_bad("label", "name the session: DELETE /sessions/<label>, "
                                               "?label=<label> or ?id=<engine id>"), req.peer);
    int64_t keep_tokens = 0;
    const std::string keep_s = req.param("keep_tokens");
    if (!keep_s.empty()) {
        if (keep_s.find_first_not_of("0123456789") != std::string::npos || keep_s.size() > 12)
            return error_response(err_bad("keep_tokens", "expected a non-negative integer: the "
                                                         "prefix to keep, in tokens"), req.peer);
        keep_tokens = std::stoll(keep_s);
    }

    SessionReleased r;
    const int rc = deps_.sched->release_session(label, id, keep_tokens, &r);
    if (rc == RAD_E_UNSUPPORTED)
        return error_response(err_not_implemented(
            "this engine has no prefix cache (started with --no-prefix-cache, or the model "
            "declares no cacheable KV group)"), req.peer);
    const std::string who = label.empty() ? "id " + std::to_string(id) : "'" + label + "'";
    if (rc == RAD_E_NOTFOUND) {
        ApiError e;
        e.status = 404;
        e.type = "not_found_error";
        e.param = label.empty() ? "id" : "label";
        e.message = "no stored session " + who + "; GET /sessions lists them";
        return error_response(e, req.peer);
    }
    if (rc == RAD_E_STATE) {
        ApiError e;
        e.status = 409;
        e.type = "conflict_error";
        e.code = "session_in_use";
        e.message = "a request of session " + who + " is queued or running; cancel it or wait "
                    "for it to finish, then release again";
        return error_response(e, req.peer);
    }
    if (rc == RAD_E_INVAL)
        return error_response(err_bad("keep_tokens", rad_strerror(rc)), req.peer);
    if (rc < 0) return error_response(err_internal(rad_strerror(rc)), req.peer);

    json o;
    if (!label.empty()) o["label"] = label;
    if (id != 0) o["id"] = id;
    o["keep_tokens"]  = keep_tokens;
    o["sessions"]     = r.sessions;
    o["entries"]      = r.entries;
    o["tokens"]       = r.tokens;
    o["device_bytes"] = r.device_bytes;
    o["host_bytes"]   = r.host_bytes;
    o["snapshots"]    = r.snapshots;
    o["kept"] = { { "tokens", r.kept_tokens }, { "shared", r.kept_shared },
                  { "live", r.kept_live } };
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

HttpResponse Server::handle_stats(const HttpRequest&) {
    json o;
    o["model"]     = opt_.model_id;
    o["version"]   = opt_.version;
    /* The control vector (core/cvec.h), when this deployment has one: `available` is what a UI
     * tests before offering the control, `default` what a request that names no choice runs. The
     * dashboard's own control asks /admin/control-vector, which also reports this. */
    if (deps_.control_vector.json) {
        try {
            o["control_vector"] = json::parse(deps_.control_vector.json());
        } catch (const std::exception&) {
            /* A malformed snapshot is not worth failing /stats over; the field is omitted. */
        }
    }
    o["uptime_s"]  = unix_now() - started_at_;
    /* The process's start, unix seconds: with a request id, a key unique across restarts. */
    o["started_at"] = started_at_;
    /* Seconds this process took to come up, the cost of a restart; 0 before the server listens. */
    o["startup_s"] = startup_s_;
    o["in_flight"] = in_flight_.load(std::memory_order_relaxed);
    o["queue_cap"] = queue_depth_;
    o["now_ms"]    = (int64_t)std::time(nullptr) * 1000;

    if (deps_.sched) {
        const SchedMetrics m = deps_.sched->metrics();
        o["running"]         = m.running;
        o["waiting"]         = m.waiting;
        o["preempted"]       = m.preempted;
        o["kv_util"]         = m.kv_util;
        o["kv_tokens_used"]  = m.kv_tokens_used;
        o["kv_tokens_total"] = m.kv_tokens_total;
        /* The part of `used` no running request reads: cached blocks of finished turns. Without
         * the split, a working cache and a pool that never releases look the same. */
        o["kv_tokens_cached"] = m.kv_tokens_cached;
        /* The pool in bytes beside tokens: tokens say how much context fits, bytes how much VRAM
         * is held. Not a unit conversion: tokens are the binding paged group's, bytes are every
         * group's, recurrent included. */
        o["kv_bytes_used"]   = (double)m.kv_bytes_used;
        o["kv_bytes_total"]  = (double)m.kv_bytes_total;
        o["kv_bytes_carved"] = (double)m.kv_bytes_carved;
        o["kv_tokens_max"]   = m.kv_tokens_max;
        o["prefix_hit_rate"] = m.prefix_hit_rate;
        /* Leading indicator beside the lagging hit rate: past ~75% kv utilisation the cache
         * starts dropping prefixes the next turn wants. */
        o["prefix_evictions"] = (double)m.prefix_evictions;
        /* Window residency (spec §7.3). `demoted`/`promoted` count blocks moved between VRAM and
         * the pinned store, `skipped` blocks a sweep left because they were shared or the store was
         * full, and `misses` promotions a step needed because its selection named a cold block. All
         * zero while --kv-residency-mib is 0 or no group declares a window. */
        o["kv_residency"] = json::object();
        o["kv_residency"]["demoted"]    = m.res_demoted;
        o["kv_residency"]["promoted"]   = m.res_promoted;
        o["kv_residency"]["skipped"]    = m.res_skipped;
        o["kv_residency"]["misses"]     = m.res_misses;
        o["kv_residency"]["host_bytes"] = m.res_host_bytes;
        o["kv_residency"]["cap_bytes"]  = m.res_cap_bytes;
        o["kv_residency"]["freed_bytes"] = m.res_freed_bytes;
        o["kv_residency"]["link_bytes"]  = m.res_link_bytes;
        o["kv_residency"]["scratch_bytes"] = m.res_scratch_bytes;
        /* Cold entries the pager rewrote to scratch pages (summed over layers and decode rows),
         * counted on the card and read back per poll; nonzero is the proof a step read paged bytes. */
        o["kv_residency"]["paged_in"] = m.res_paged_in;
        /* Of those, the ones the page cache served from a scratch page that still held the block
         * (no copy): paged_hits / paged_in is the cache's hit rate. */
        o["kv_residency"]["paged_hits"] = m.res_paged_hits;
        /* Linear-state caching, separate from the attention hit rate: a hybrid prompt can be
         * fully attention-cached and still re-run every recurrent layer over its prefix, since
         * the two hit at different granularities. Zero on a model with no recurrent state. */
        o["linear_hit_rate"] = m.linear_hit_rate;
        o["linear_cached_tokens"] = (double)m.linear_cached_tokens;
        o["checkpoints_written"] = (double)m.checkpoints_written;
        /* Why snapshots went and why hits reused nothing: on a hybrid model these say why a
         * long prompt was prefilled again. `hit_cut_tokens` is the checkpoint shortfall,
         * prompt tokens the attention cache held and admission computed again. */
        o["ckpt_evictions"]  = { { "retention", m.prefix_ckpt_retired },
                                 { "store_full", m.prefix_ckpt_lru } };
        o["ckpt_refused"]    = m.prefix_ckpt_refused;
        o["released"]        = { { "entries", m.prefix_released },
                                 { "snapshots", m.prefix_ckpt_released } };
        o["hit_cuts"]        = { { "no_snapshot", m.hit_cut_no_snapshot },
                                 { "windowed_group", m.hit_cut_window },
                                 { "adopt_failed", m.hit_cut_adopt },
                                 { "resumed_below", m.hit_cut_resumed_below } };
        o["hit_cut_tokens"]  = (double)m.hit_cut_tokens;
        o["prefill_tps"]     = m.prefill_tps;
        o["decode_tps"]      = m.decode_tps;
        o["draft_accept"]    = m.draft_acceptance;
        o["preemptions"]     = m.preemptions;
        o["total_requests"]  = m.total_requests;
        o["steps"]           = m.steps;
        o["steps_prefill"]   = m.steps_prefill;
        o["steps_mixed"]     = m.steps_mixed;
        o["prefill_chunks_capped"] = m.prefill_chunks_capped;
        o["steps_decode"]    = m.steps_decode;
        o["steps_held"]      = m.steps_held;
        /* Time inside steps. The one rate the page cannot derive: wall clock between polls also
         * covers idle time (see SchedMetrics::step_ns). */
        o["step_ns"]         = m.step_ns;
        o["step_ns_decode"]  = m.step_ns_decode;
        o["step_ns_decode_best"] = m.step_ns_decode_best;
        o["step_ns_decode_recent"] = m.step_ns_decode_recent;
        o["rows_unspec"]     = m.decode_rows_unspeculated;

        /* Conditional rate per position with its reach count: a rate of 0.0 over a reach of 3 is
         * noise, over a reach of 40000 a broken draft round. */
        json pos = json::array();
        for (int j = 0; j < m.draft_depth_seen && j < SchedMetrics::MAX_DRAFT_POS; ++j) {
            json p;
            p["reached"]  = m.draft_pos_reached[j];
            p["accepted"] = m.draft_pos_accepted[j];
            p["rate"] = m.draft_pos_reached[j] > 0
                      ? (double)m.draft_pos_accepted[j] / (double)m.draft_pos_reached[j] : 0.0;
            pos.push_back(std::move(p));
        }
        o["draft_positions"] = std::move(pos);

        /* Live request table: counters and timestamps. `decode_tps` is the request's own average
         * since its first token. */
        std::vector<RequestStat> rq;
        deps_.sched->requests(&rq);
        json reqs = json::array();
        for (const RequestStat& q : rq) {
            json j;
            j["id"]        = q.id;
            j["state"]     = q.state;
            j["prompt"]    = q.prompt_tokens;
            j["computed"]  = q.computed_tokens;
            j["cached"]    = q.cached_tokens;
            j["output"]    = q.output_tokens;
            j["max"]       = q.max_tokens;
            j["ctx"]       = q.ctx_tokens;
            j["kv_tokens"] = q.kv_tokens;
            j["kv_blocks"] = q.kv_blocks;
            j["draft"]     = q.n_draft;
            j["slot"]      = q.slot;
            j["age_s"]     = q.age_s;
            if (q.ttft_s >= 0) j["ttft_s"] = q.ttft_s;
            if (q.prefill_s >= 0) j["prefill_s"] = q.prefill_s;
            j["tps"]       = q.decode_tps;
            reqs.push_back(std::move(j));
        }
        o["requests"] = std::move(reqs);
    }

    /* The engine's snapshot. Absent without a snapshot function, and the page then omits those
     * panels rather than drawing zeroes (an empty VRAM bar would read as "no memory used"). */
    if (deps_.live) {
        LiveSnapshot s;
        deps_.live(s);

        json cards = json::array();
        for (const CardStat& c : s.cards) {
            json k;
            k["index"]      = c.index;
            k["name"]       = c.name;
            k["vram_used"]  = c.vram_used;
            k["vram_total"] = c.vram_total;
            /* 0 means "not sampled" for these three; the page shows nothing rather than a zero. */
            if (c.util > 0)    k["util"]    = c.util;
            if (c.power_w > 0) k["power_w"] = c.power_w;
            if (c.temp_c > 0)  k["temp_c"]  = c.temp_c;
            cards.push_back(std::move(k));
        }
        o["cards"] = std::move(cards);

        /* Expert panel data (see ExpertPlane in liveview.h). Tallied once in the engine, not here
         * or in the page, so all views agree. */
        if (s.experts.n_layers > 0 && s.experts.n_experts > 0) {
            json e;
            e["n_layers"]   = s.experts.n_layers;
            e["n_experts"]  = s.experts.n_experts;
            e["tier"]       = s.experts.tier;
            e["n_units"]    = s.experts.n_units;
            e["bytes"]      = s.experts.total_bytes;
            e["movable"]    = s.experts.movable;
            e["heat"]       = s.experts.heat_enabled;
            /* One named row per tier, so the page holds no copy of the enum's order. */
            json tiers = json::array();
            for (int t = 0; t < kTierCount; ++t) {
                json r;
                r["name"]  = tier_name((Tier)t);
                r["units"] = s.experts.units[t];
                r["bytes"] = s.experts.bytes[t];
                tiers.push_back(std::move(r));
            }
            e["tiers"] = std::move(tiers);
            e["layer_units"]    = s.experts.layer_units;
            e["layer_resident"] = s.experts.layer_resident;
            e["promotions"] = s.experts.promotions;
            e["demotions"]  = s.experts.demotions;
            e["distinct_promoted"] = s.experts.distinct_promoted;
            e["readmits"]    = s.experts.readmits;
            e["routed"]      = s.experts.routed;
            e["routed_res"]  = s.experts.routed_resident;
            e["dispatches"]  = s.experts.dispatches;
            e["refused"]     = s.experts.refused;
            e["no_candidate"] = s.experts.no_candidate;
            e["no_record"]   = s.experts.no_record;
            e["starved_slab"] = s.experts.starved_slab;
            e["flex_bytes"]    = (double)s.experts.flex_bytes;
            e["flex_capacity"] = (double)s.experts.flex_capacity;
            e["starved_pool"] = s.experts.starved_pool;
            e["quarantined"] = s.experts.quarantined;
            e["stage_layers"]   = s.experts.stage_layers;
            e["staged_units"]   = s.experts.staged_units;
            e["staged_bytes"]   = (double)s.experts.staged_bytes;
            e["stage_overflow"] = s.experts.stage_overflow;
            e["stage_copies"]   = s.experts.stage_copies;
            o["experts"] = std::move(e);
        }

        /* Link traffic: cumulative byte counters only; the page derives rates. */
        if (s.link.present) {
            json l;
            l["world"]          = s.link.world;
            l["ar_calls"]       = s.link.ar_calls;
            l["ar_bytes"]       = s.link.ar_bytes;
            l["h2d_bytes"]      = s.link.h2d_bytes;
            l["d2h_bytes"]      = s.link.d2h_bytes;
            l["ssd_reads"]      = s.link.ssd_reads;
            l["ssd_bytes"]      = s.link.ssd_bytes;
            l["ssd_staging_bytes"] = s.link.ssd_staging_bytes;
            l["ssd_promoted_bytes"] = s.link.ssd_promoted_bytes;
            l["ssd_deposits"]      = s.link.ssd_deposits;
            l["ssd_deposit_bytes"] = s.link.ssd_deposit_bytes;
            l["pool_evictions"]     = s.link.pool_evictions;
            l["prefetched"]     = s.link.prefetched;
            l["prefetch_bytes"] = s.link.prefetch_bytes;
            l["staged_bytes"]   = s.link.staged_bytes;
            l["stream_bytes"]   = s.link.stream_bytes;
            l["kv_stream_bytes"]  = s.link.kv_stream_bytes;
            l["kv_streaming"]   = s.link.kv_streaming;
            l["kv_stream_active"] = s.link.kv_stream_active;
            o["link"] = std::move(l);
        }
        if (s.passes.present) {
            json p;
            p["played"]         = s.passes.played;
            p["recorded"]       = s.passes.recorded;
            p["audited"]        = s.passes.audited;
            p["issued_unkeyed"] = s.passes.issued_unkeyed;
            p["issued_refused"] = s.passes.issued_refused;
            p["issued_first"]   = s.passes.issued_first;
            p["dispatched"]     = s.passes.dispatched;
            p["unordered"]      = s.passes.unordered;
            o["passes"] = std::move(p);
        }
    }

    /* Not counted in metrics_.observe_http, as for /health: a page polling twice a second would
     * make the request counters describe the dashboard. */
    return HttpResponse::json_body(200, json_dump(o));
}


/* ================================================================== /graph */
/* The declared compute graph: every op, its bucket table, the kernel that resolved each band, its
 * library, the constraints it matched and the chain it stands in for. The same walk as
 * `--debug-graph` (dump_graph_json shares its helpers), served from the running process. Static
 * after declare, so it is rendered once. */
HttpResponse Server::handle_graph(const HttpRequest& req) {
    if (!opt_.graph_json)
        return error_response(err_not_implemented(
            "the server was not given a declared graph; this build has no engine attached"),
                req.peer);
    HttpResponse r;
    r.status = 200;
    r.body = opt_.graph_json();
    return r;
}

/* ================================================================== the page */
/* One string, so it cannot go missing or stale against the binary and needs no build step or
 * network fetch. It parses no untrusted input: every value comes from same-origin /stats and goes
 * in through textContent or a numeric attribute, never innerHTML.
 *
 * Built once and updated in place: elements are created only when a panel's shape changes (draft
 * depth, card count); after that only textContent and one style property are written. Rebuilding
 * the DOM every poll would relayout, flicker and drop the reader's text selection.
 *
 * Display rules: an unsampled number is blank, not zero; a section without data is hidden;
 * nothing moves that is not changing. Black, one weight, one accent. */
static const char* const kPage = R"HTML(<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>radiance</title>
<style>
:root {
  --bg:#000; --card:#0a0a0a; --edge:#1c1c1c; --rule:#151515; --hover:#101010;
  --fg:#f5f5f5; --dim:#7b7b7b; --faint:#4d4d4d; --ghost:#2b2b2b; --track:#1e1e1e;
  --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; --cool:#60a5fa;
  --ui:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --r:12px;
  /* THE SECOND COLUMN'S WIDTH IS SHARED, because two grids have to agree about it. The strip
     above the body is a different grid from the body, and a metric cell whose edge lands three
     quarters of the way across the panel underneath it reads as a mistake rather than as a
     separate row. One declaration, so they cannot drift.

     AND IT MUST EXCEED THE COLUMN'S OWN MIN-CONTENT, or the track is not fixed at all. A grid
     item does not shrink below its min-content unless it is told it may, so a panel whose widest
     unbreakable row is wider than this number pushes the track out to that width -- and since the
     rows carry figures whose text length changes every poll, the track then breathes and both
     columns move sideways under the reader. The rule below that gives every panel `min-width:0`
     is what makes that impossible; this number is what keeps it from being paid for in ellipses. */
  --side:372px;
}
/* THE SCROLLBAR IS PART OF THE LAYOUT. The page is a centred column, so a scrollbar that comes
   and goes moves every panel sideways by half its width -- and this page changes height
   constantly: panels are revealed once the first poll says which of them this deployment has,
   the request table fills, and the views are three different lengths. The gutter is reserved
   always, and the page stops moving left and right while it is being read. */
html { overflow-y:scroll; scrollbar-gutter:stable; }
* { box-sizing:border-box; }
html,body { background:var(--bg); }
body { margin:0; color:var(--fg); font:400 13px/1.5 var(--ui);
       -webkit-font-smoothing:antialiased; }
.wrap { max-width:1400px; margin:0 auto; padding:28px 24px 80px; }
[hidden] { display:none !important; }

header { display:flex; align-items:center; gap:10px; padding-bottom:16px; }
.dot { width:6px; height:6px; border-radius:50%; background:var(--ok); flex:none;
       box-shadow:0 0 0 3px rgba(74,222,128,.12); }
.dot.stale { background:var(--bad); box-shadow:0 0 0 3px rgba(248,113,113,.12); }
h1 { margin:0; font-size:13.5px; font-weight:600; letter-spacing:-.005em; }
.arch { color:var(--dim); font-size:11px; font-family:var(--mono); padding:3px 7px;
        border:1px solid var(--edge); border-radius:5px; }
.ver { margin-left:auto; color:var(--faint); font-size:11.5px; font-family:var(--mono); }
#key { margin-left:auto; width:160px; font:400 12px/1 var(--mono); color:var(--fg);
       background:var(--card); border:1px solid var(--edge); border-radius:4px; padding:5px 8px; }
#key:not([hidden]) + .ver { margin-left:0; }

.panel { background:var(--card); border:1px solid var(--edge); border-radius:var(--r); }
.phead { display:flex; align-items:baseline; justify-content:space-between; gap:14px;
         padding:13px 16px 12px; border-bottom:1px solid var(--rule); }
.cap { font-size:10.5px; font-weight:600; letter-spacing:.1em; text-transform:uppercase;
       color:var(--dim); }
.aside { font-family:var(--mono); font-size:11.5px; color:var(--dim); }
/* A panel header's left side can hold a title plus a control; the header itself is
 * space-between, so without this group the control would land in the middle. */
.pleft { display:flex; align-items:baseline; gap:9px; }
.pbtn { background:none; border:0; padding:0; margin:0; color:var(--faint); cursor:pointer;
        display:inline-flex; align-items:center; }
.pbtn:hover { color:var(--fg); }
.pbtn:focus-visible { outline:1px solid var(--edge); outline-offset:3px; }
/* The glyph is nudged down 2px so the tray bar's bottom lines up with the bottom of the
 * caption's letters. align-items:baseline puts the SVG's BOX on the text baseline, but the
 * baseline sits above the box bottom, so the bar floated ~2px high -- measured in headless
 * Chrome, not eyeballed. The viewBox-flush geometry alone does not fix this. */
#reqjson svg { transform: translateY(2px); }
.pbody { padding:14px 16px 16px; }

/* The control vector's switch: a checkbox row with its explanation under it. The two need a gap
   between them, because the explanation is set at the size every caption on this page uses, and a
   line that sits flush under the one above it reads as that line's tail, not as a sentence. It is
   inset a little too, so it does not start on the same edge as the row it belongs to. */
.cvrow { display:flex; align-items:center; gap:8px; }
#cvnote { margin-top:9px; padding-left:4px; }

.pollrow { display:flex; align-items:center; gap:10px; font-size:13px; color:var(--fg); }
.pollrow + .pollnote { margin-top:6px; }
.pollnote { max-width:74ch; }
.pollrow input[type=number] { width:11ch; font:400 15px/1.4 var(--mono); color:var(--fg);
                              background:var(--track); border:1px solid var(--edge);
                              border-radius:5px; padding:7px 10px; }

/* ------------------------------------------------ THE FIRST ROW IS FIXED, THE REST FIT THEIR CONTENT
   A card whose content changes LENGTH moves every card under it while it is being read, and one
   card here does that every poll: the list of requests gains and loses rows. So the first row --
   the requests and the batch beside them -- is a definite height, the same in both columns, and
   its bodies scroll.
   Every other card holds a shape that is fixed for the life of the deployment: a card count, a
   draft depth, a tier ladder, a plane of layers by experts. Each line in them is clipped to one
   line or reserved at its full height whether or not it has text, so a card sized by its content
   cannot move -- and a definite height on one of them is either empty space under the figures or
   a scrollbar over figures that belong on screen. Below the first row the two columns therefore
   divide at different lines; they are separate questions and are not read across.
   The scrollbar gutter is reserved whether or not a body scrolls, for the same reason the
   document's is -- a bar that comes and goes is eight pixels of sideways movement -- and so the
   figures in every card of one column end at one x.
   min-width:0 is the other half of it -- see --side. Without it a panel whose widest row does not
   fit stretches the column instead of scrolling, and both columns move. */
.body .panel { display:flex; flex-direction:column; min-width:0; }
.body .panel > .pbody { flex:1; min-height:0; display:flex; flex-direction:column;
                        overflow-y:auto; overscroll-behavior:contain; scrollbar-gutter:stable;
                        scrollbar-width:thin; scrollbar-color:var(--edge) transparent; }
.body .panel > .pbody::-webkit-scrollbar { width:8px; }
.body .panel > .pbody::-webkit-scrollbar-thumb { background:var(--edge); border-radius:4px; }
.body .panel > .pbody::-webkit-scrollbar-track { background:transparent; }
#reqpanel, #batchpanel { height:306px; }

/* metric strip: the 1px grid gap over an edge-coloured ground draws every divider, and keeps
   drawing them correctly when the row wraps. */
.strip { display:grid; grid-template-columns:repeat(auto-fit,minmax(148px,1fr)); gap:1px;
         background:var(--edge); border:1px solid var(--edge); border-radius:var(--r);
         overflow:hidden; margin-bottom:14px; }
/* THE STRIP IS DIVIDED WHERE THE PAGE IS, once there is room for the body to have two columns.
   Six equal cells put their dividers wherever six divides the width, and the strongest line on
   the page -- the gutter between the two columns of panels -- ran through the middle of a cell.
   The last two sit over the second column instead: four flexible cells, then two halves of the
   side width less the 1px gap between them, so the fifth cell's left edge and the gutter are the
   same x. Four and two rather than five and one because a single cell as wide as the whole
   second column is twice its neighbours and reads as a mistake of its own.
   The floor is the body's own: below 1001px there is one column and nothing to agree with, so
   the strip goes back to dividing the width evenly. The four cells are 143px there rather than
   the 148 the auto-fit track asks for, and nothing in them overflows or wraps at that size. */
@media (min-width:1001px) {
  /* #strip and not .strip: the sessions view has a strip of its own whose cells hide with the
     model, and it has no second column under it to agree with. */
  #strip { grid-template-columns:repeat(4,minmax(0,1fr))
                                 repeat(2,calc((var(--side) - 2px) / 2)); }
}
/* EVERY CELL IS THE SAME HEIGHT AND THE SPARKLINE SITS ON ONE BASELINE. Only some cells carry a
   breakdown line, so sized by their contents the cells were three different heights, the strip
   took the tallest, and the plots inside it stood at three different y -- a row of charts that
   cannot be compared by eye is a row of charts nobody reads across. The column is a flex column
   of a fixed height with the plot pushed to the bottom, and the breakdown's two lines are
   reserved in every cell whether or not that cell has anything to put there. */
.cell { background:var(--card); padding:11px 14px 10px; min-width:0; height:136px;
        display:flex; flex-direction:column; }
.cell .n { font:500 22px/1 var(--mono); letter-spacing:-.03em; white-space:nowrap;
           font-variant-numeric:tabular-nums; }
.cell .n u { font:400 12px/1 var(--mono); color:var(--dim); text-decoration:none;
             margin-left:5px; letter-spacing:0; }
.cell .cap { display:block; margin-top:6px; }
/* A BREAKDOWN IS NOT A CAPTION. Set in the caption's uppercase and letter-spacing it wrapped to
   three ragged lines and stopped being readable as either one.
   TWO LINES, CLIPPED. The text is assembled from figures whose width changes, so a slot sized to
   the text is a slot that grows a third line on the poll where one of them gains a digit. */
.cell .det { display:block; margin-top:5px; height:32px; overflow:hidden;
             font:400 10.5px/1.5 var(--mono); color:var(--faint); }
.cell svg { display:block; width:100%; height:17px; margin-top:auto; }
.cell svg path { fill:none; stroke:#b4b4b4; stroke-width:1.25; stroke-linejoin:round;
                 stroke-linecap:round; vector-effect:non-scaling-stroke; }

/* THE SECOND COLUMN IS SIZED BY ITS LONGEST LABEL. At 330px the interconnect and residency rows
   ran out of room and ellipsed their own names -- "All-red...", "Promot..." -- which is a legend
   that cannot be read at all rather than one that is merely tight. */
.body { display:grid; gap:14px; grid-template-columns:minmax(0,1fr) var(--side); align-items:start; }
/* WHICH PANELS EXIST IS A PROPERTY OF THE DEPLOYMENT, and it is not known until the first poll
   answers: a dense model has no expert residency, a single rank has no interconnect, a model
   with no draft head has no acceptance chart. Drawing the body before that answer arrives means
   drawing a layout that is about to change, and the change lands a few hundred milliseconds in,
   which is exactly when it is being read. So the body waits for the answer instead -- on a
   dashboard served beside the engine it is one round trip -- and a timeout reveals it anyway so
   that a server that cannot be reached still renders its own frame. */
#view-dash.boot > .body { display:none; }
#view-dash.boot > .strip { opacity:.45; }
.strip { transition:opacity .18s linear; }
.stack { display:grid; gap:14px; }
@media (max-width:1000px) {
  .body { grid-template-columns:minmax(0,1fr); }
  /* In one column a nested scroller with overscroll-behavior:contain (there so a desktop wheel
   * over the request table does not move the page) traps a touch drag once it bottoms out, so
   * narrow screens let the scroll chain to the document. `.body .hold` rather than `.hold`: the
   * base .hold rule comes later in this sheet and would win at equal specificity. */
  .body .panel > .pbody, .body .hold { overscroll-behavior:auto; }
  /* The history table is the card's scroller, and the same trap on a phone. */
  #hist .hlist { overscroll-behavior:auto; }
}

/* the request table */
table { width:100%; border-collapse:collapse; font-family:var(--mono); font-size:11.5px;
        font-variant-numeric:tabular-nums; }
thead th { font:600 9.5px/1 var(--ui); letter-spacing:.09em; text-transform:uppercase;
           color:var(--faint); text-align:right; padding:0 0 10px; white-space:nowrap; }
thead th:first-child, tbody td:first-child { text-align:left; }
tbody td { padding:7px 0; border-top:1px solid var(--rule); text-align:right;
           white-space:nowrap; }
tbody tr:hover td { background:var(--hover); }
th + th, td + td { padding-left:14px; }
.tag { display:inline-block; font:500 9.5px/1 var(--ui); letter-spacing:.06em;
       text-transform:uppercase; padding:3px 6px; border-radius:4px; background:var(--track);
       color:var(--dim); }
.tag.decode    { background:rgba(74,222,128,.13);  color:#7fe0a3; }
.tag.prefill   { background:rgba(96,165,250,.13);  color:#8dbcfb; }
.tag.waiting   { background:rgba(251,191,36,.12);  color:#e9c065; }
.tag.preempted { background:rgba(248,113,113,.13); color:#f29d9d; }
.mini { display:inline-block; width:44px; height:3px; border-radius:2px; background:var(--track);
        vertical-align:middle; margin-right:8px; overflow:hidden; }
.mini > i { display:block; height:100%; background:var(--dim); border-radius:2px; }
.empty { color:var(--faint); font-size:12px; padding:6px 0 2px; }
/* THE PANEL IS A FIXED WINDOW ONTO A LIST THAT CHANGES LENGTH EVERY POLL. The running requests
   are the one figure on this page that moves by itself, and a table that grows a row pushed the
   chart and the residency panel below it down the screen while they were being read. The card's
   own height is fixed (see above), so this simply fills it and scrolls under a pinned header --
   the window moves, never the document. */
.hold { height:100%; overflow-y:auto; overscroll-behavior:contain; }
.hold thead th { position:sticky; top:0; background:var(--card); z-index:1; }
.hold::-webkit-scrollbar { width:8px; }
.hold::-webkit-scrollbar-thumb { background:var(--edge); border-radius:4px; }
.hold::-webkit-scrollbar-track { background:transparent; }
.hold { scrollbar-width:thin; scrollbar-color:var(--edge) transparent; }


/* The requests card: the live table, which stays put, over the history, which scrolls. The card
   is half again the stock height to make room for history rows. */
#reqpanel { height:459px; }
#reqpanel > .pbody { overflow:hidden; }
#reqpanel .hold { flex:0 0 auto; height:auto; overflow:visible; }
#reqpanel .hold thead th { position:static; }   /* the live table is short: it never scrolls */
#hist { flex:1 1 auto; min-height:0; display:flex; flex-direction:column;
        margin-top:14px; padding-top:12px; border-top:1px solid var(--rule); }
#hist .hhead { display:flex; align-items:baseline; gap:10px; margin-bottom:8px; }
#hist .hsum { flex:1; font-family:var(--mono); font-size:11px; color:var(--dim); line-height:1.5; }
/* Each category is one atomic unit: if it cannot fit on the line it moves whole, rather than
 * breaking mid-figure and orphaning a unit -- "810" at the end of one line, "ms" on the next.
 * The separator is its own element with padding because HTML collapses runs of spaces, so
 * spacing in the string could never have widened the category break. */
#hist .hgrp { white-space:nowrap; }
#hist .hsep { color:var(--faint); padding:0 9px; }
#hist .hbtn { font-family:var(--mono); font-size:10px; color:var(--faint); cursor:pointer;
              user-select:none; padding:0 4px; }
#hist .hbtn:hover { color:var(--bad); }
#hist .hlist { flex:1 1 auto; min-height:0; overflow-y:auto; overscroll-behavior:contain;
               scrollbar-width:thin; scrollbar-color:var(--edge) transparent; }
#hist .hlist::-webkit-scrollbar { width:8px; }
#hist .hlist::-webkit-scrollbar-thumb { background:var(--edge); border-radius:4px; }
#hist thead th { position:sticky; top:0; z-index:1; background:var(--card); }
#hist tbody td.b { font-weight:700; }
#hist tbody td.dim { color:var(--faint); }
#hist tr.gap td { text-align:left; color:var(--faint); font-size:11px; white-space:normal; }

/* the draft chart */
.chart { display:grid; grid-template-columns:max-content minmax(0,1fr); gap:0 12px;
         align-items:center; }
/* Tall enough to read a difference off. The two panels of the pair are the same height whatever
   this is, so a short plot does not buy any page back -- it only makes two bars a few points
   apart look identical. */
.yax { display:flex; flex-direction:column; justify-content:space-between; height:96px;
       font:400 9.5px/1 var(--mono); color:var(--faint); text-align:right; }
.plot { position:relative; height:96px; border-bottom:1px solid var(--edge); }
.rules { position:absolute; inset:0; display:flex; flex-direction:column;
         justify-content:space-between; }
.rules i { display:block; height:1px; background:var(--rule); }
.cols { position:absolute; inset:0; display:flex; gap:8px; align-items:flex-end; }
.cols > div { flex:1; min-width:0; height:100%; display:flex; flex-direction:column;
              justify-content:flex-end; align-items:center; }
.cols .bar { width:100%; max-width:40px; background:var(--fg); border-radius:2px 2px 0 0;
             transition:height .25s cubic-bezier(.4,0,.2,1); }
.cols .lo .bar { background:var(--ghost); }
.cols .pv { font:500 9px/1 var(--mono); margin-bottom:3px; }
.cols .lo .pv { color:var(--faint); }
.xax, .nax { display:flex; gap:8px; }
.xax > span, .nax > span { flex:1; text-align:center; min-width:0; }
.xax { font:400 10px/1 var(--mono); color:var(--dim); padding-top:5px; }
.nax { font:400 10px/1 var(--mono); color:var(--faint); padding-top:3px; }
.nkey { font:400 9.5px/1 var(--mono); color:var(--faint); text-align:right; padding-top:3px; }

/* rows and meters */
/* THE VALUE COLUMN IS FIXED for the reason the rows above are: at max-content the boundary
   between the labels and the figures moved every time a counter gained a digit, which is once a
   second on a busy server and takes the whole panel with it. */
.kv { display:grid; grid-template-columns:minmax(0,1fr) 132px; }
.kv > div { padding:5px 0; border-top:1px solid var(--rule); }
.kv > div:nth-child(-n+2) { border-top:0; padding-top:0; }
.kv .k { color:var(--dim); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.kv .v { text-align:right; font-family:var(--mono); font-size:12px; white-space:nowrap;
         font-variant-numeric:tabular-nums; }
.meter + .meter, .dev { margin-top:14px; }
.dev { padding-top:14px; border-top:1px solid var(--rule); }
/* THE LABEL YIELDS, THE FIGURE DOES NOT. A flex item does not shrink below its min-content
   unless told to, so without min-width:0 the label's full untruncated width was part of the
   panel's min-content -- which is how a one-line caption came to decide how wide the column
   holding it was, and to re-decide it whenever the figure beside it changed length. */
.mrow { display:flex; justify-content:space-between; align-items:baseline; gap:12px;
        margin-bottom:7px; min-width:0; }
.mrow .t { color:var(--dim); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
           min-width:0; }
.mrow .t b { color:var(--faint); font-weight:400; font-family:var(--mono); margin-right:7px; }
.mrow .v { font-family:var(--mono); font-size:11.5px; white-space:nowrap;
           font-variant-numeric:tabular-nums; }
.track { height:4px; border-radius:2px; background:var(--track); overflow:hidden; }
.track > i { display:block; height:100%; width:0; background:var(--fg); border-radius:2px;
             transition:width .25s cubic-bezier(.4,0,.2,1); }
.track > i.warn { background:var(--warn); }
.track > i.bad  { background:var(--bad); }
/* A SECOND SEGMENT, for a bar whose fill is two things. The KV pool is held by running requests
   and by the prefix cache keeping finished turns for their next one, and the second is usually
   the larger -- drawn in the ghost tone beside the live fill, because it is occupied VRAM that
   nothing is reading and gives way the moment a live request needs it. */
.track { display:flex; }
.track > b { display:block; height:100%; width:0; background:var(--ghost); border-radius:2px;
             transition:width .25s cubic-bezier(.4,0,.2,1); flex:none; }
/* The held segment takes the warning too, at the same remove from the live one: a pool near the
   wall is mostly cache on any server carrying conversations, so colouring only the live sliver
   would put the signal on the thinnest part of the bar. */
.track > b.warn { background:rgba(251,191,36,.34); }
.track > b.bad  { background:rgba(248,113,113,.34); }
.track > i { flex:none; }

/* the draft chart and its summary, side by side. The chart does not need the whole column: seven
   bars in 900px is a poster, not a chart. */
.pair { display:grid; gap:14px; grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);
        align-items:stretch; }
.pair > .panel { min-width:0; }
@media (max-width:780px) { .pair { grid-template-columns:minmax(0,1fr); } }
.duo { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:10px; }
.duo .n { font:500 19px/1 var(--mono); letter-spacing:-.03em;
          font-variant-numeric:tabular-nums; }
.duo .n u { font:400 13px/1 var(--mono); color:var(--dim); text-decoration:none;
            margin-left:4px; letter-spacing:0; }
.duo .cap { display:block; margin-top:5px; }

/* the batch: one cell per scheduler slot. Occupancy is a picture, not a count -- eight of
   thirty-two full and eight of eight full are the same number and not the same situation. */
.slots { display:grid; grid-template-columns:repeat(auto-fill,minmax(13px,1fr)); gap:3px; }
.slots i { display:block; aspect-ratio:1; border-radius:2px; background:var(--track); }
.slots i.decode    { background:#4ade80; }
.slots i.prefill   { background:#60a5fa; }
.slots i.waiting   { background:#fbbf24; }
.slots i.preempted { background:#f87171; }
.legend { display:flex; flex-wrap:wrap; gap:12px; margin-top:12px; font-size:10.5px;
          color:var(--faint); }
.legend span { display:flex; align-items:center; gap:5px; }
.legend b { width:7px; height:7px; border-radius:2px; display:block; }
.qbar { margin-top:14px; padding-top:14px; border-top:1px solid var(--rule); }

/* The line under a split meter naming what its two segments are. TWO LINES, RESERVED: its
   figures change length, and in a card sized by its content a line that wraps on one poll and not
   the next moves everything under it. */
.ksub { margin-top:6px; font:400 10.5px/1.4 var(--mono); color:var(--faint); height:2.8em;
        overflow:hidden; }

/* ---------------------------------------------------------------- the sessions view
   ONE QUESTION A ROW: what happens if this conversation comes back now. The rows are split by the
   answer -- those that resume from a saved state and those that start over -- rather than carrying
   it as a dot, because on a server with long-lived agent clients the second group is most of the
   table and none of it needs reading to understand the first. */
#view-sess .cell { height:auto; padding-bottom:12px; }
/* Stacked panels in one view are 14px apart, the same gap the strip's own margin-bottom gives. */
#view-sess .panel + .panel, #view-model .panel + .panel { margin-top:14px; }
.stbl td, .stbl th { width:1%; }
/* The two text columns take the slack and read left to right; the figures stay narrow. */
.stbl .l { text-align:left; width:auto; }
.stbl td.l { color:var(--dim); }
.stbl .ret b { color:var(--fg); font-weight:500; }
/* Fixed, so the tier column starts at one x in both tables. */
.stbl .ret { width:340px; }
.sid { font-weight:500; }
.slabel { margin-left:8px; color:var(--faint); }
.mv { font:500 10px/1 var(--ui); color:var(--faint); border:1px solid var(--rule);
      border-radius:4px; padding:2px 5px; margin-left:6px; }
/* WHERE IT IS, by share of the conversation's context, one colour a tier. A conversation is
   routinely in two tiers at once -- a maintenance pass is bounded -- so this is proportions and
   never a label. The tier names beside it are set in the same colours and are the legend. */
.tbar { display:inline-flex; width:120px; height:6px; border-radius:3px; overflow:hidden;
        background:var(--track); vertical-align:middle; margin-right:10px; }
.tbar > i { display:block; height:100%; }
.t-dev  { background:#4ade80; } .tn-dev  { color:#7fe0a3; }
.t-host { background:#60a5fa; } .tn-host { color:#8dbcfb; }
/* The group that starts over is folded by default: it is a count first and rows second. */
details.sgrp > summary { list-style:none; cursor:pointer; }
details.sgrp > summary::-webkit-details-marker { display:none; }
details.sgrp > summary .cap::before { content:"▸"; display:inline-block; width:14px;
                                      color:var(--faint); }
details.sgrp[open] > summary .cap::before { content:"▾"; }
details.sgrp:not([open]) > summary { border-bottom:0; }
details.sgrp > summary:hover .cap { color:var(--fg); }

/* navigation. Two views, one document: the Model view is static after declare, so switching to it
   is a fetch that happens once and never again -- there is nothing to route to a second page for. */
nav { display:flex; gap:2px; margin-left:14px; }
nav button { font:500 12px/1 var(--ui); color:var(--dim); background:none; border:0;
             padding:7px 11px; border-radius:6px; cursor:pointer; }
nav button:hover { color:var(--fg); background:var(--hover); }
nav button.on { color:var(--fg); background:var(--track); }

/* the model view */
/* ---------------------------------------------------------------- the compute graph, as a flow
   A vertical flow of nodes in declared order. The repeats are collapsed into a stack with a
   multiplier, because a 64-layer model is 1569 nodes and sixty-four identical layers drawn
   sixty-four times is the same information spread over enough screen that nobody reads it. */
.gwrap { display:grid; grid-template-columns:minmax(0,1fr) 460px; gap:14px; align-items:start; }
@media (max-width:1180px) { .gwrap { grid-template-columns:minmax(0,1fr); } }

.flow { display:flex; flex-direction:column; align-items:stretch; padding:8px 0 24px;
        max-width:620px; margin:0 auto; }
.link { width:1px; height:14px; background:var(--edge); flex:none; }
.link.into { position:relative; }
.link.into::after { content:""; position:absolute; left:-3px; bottom:0; width:7px; height:7px;
                    border-right:1px solid var(--edge); border-bottom:1px solid var(--edge);
                    transform:rotate(45deg) translate(-1px,-1px); }

.grp { position:relative; width:100%; max-width:600px; }
.grp .inner { display:flex; flex-direction:column; align-items:stretch; position:relative;
              z-index:2; }
.grp.rep { padding:14px 14px 14px 0; }
/* two offset plates behind the block, so a repeat reads as depth rather than as a note */
.grp.rep::before, .grp.rep::after {
  content:""; position:absolute; inset:14px 0 14px 14px; border:1px solid var(--edge);
  border-radius:11px; background:var(--card); z-index:0;
}
.grp.rep::before { transform:translate(11px,11px); opacity:.4; }
.grp.rep::after  { transform:translate(6px,6px);   opacity:.7; }
.grp.rep > .inner { background:var(--card); border:1px solid var(--edge); border-radius:11px;
                    padding:12px; }
/* A repeat is COLLAPSED by default. Expanded, this model is 1569 nodes and no amount of nesting
   makes that a picture; collapsed, the whole graph is one screen and the reader opens the block
   they are asking about. */
.rephead { display:flex; align-items:center; gap:10px; cursor:pointer; padding:11px 13px;
           background:var(--card); border:1px solid var(--edge); border-radius:11px;
           position:relative; z-index:2; }
.rephead:hover { background:var(--hover); border-color:#333; }
.rephead .tw { color:var(--dim); font:400 10px/1 var(--mono); width:8px; }
.rephead .lbl { font:500 12.5px/1.2 var(--mono); }
.rephead .sub { font:400 10.5px/1.3 var(--mono); color:var(--faint); margin-left:auto;
                text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.grp.rep.open > .rephead { border-bottom-left-radius:0; border-bottom-right-radius:0;
                           border-bottom-color:transparent; }
.grp.rep.open > .inner { border-top-left-radius:0; border-top-right-radius:0; }

.mult { position:absolute; right:-3px; top:-3px; z-index:3; font:600 11px/1 var(--ui);
        letter-spacing:.05em; color:var(--fg); background:var(--track);
        border:1px solid var(--edge); border-radius:6px; padding:5px 8px; }
/* A repeat of ONE op is not a block -- it is that op, twice. Plating it costs a row of chrome and
   a click to say something the badge already says. */
.gnode .mult { top:-8px; right:-8px; padding:4px 7px; font-size:10.5px; }

/* ---------------------------------------------------------------- either/or
   THE ALTERNATION, WHICH THE FLAT LIST DREW AS A REDUNDANCY. An architecture that asks for a fused
   op and emits the chain it stands for when the fusion cannot serve declares BOTH, and a flow that
   draws them in sequence says `all_reduce -> rmsnorm_quant_fp8 -> ar_rmsnorm_quant_fp8` -- three
   launches, one after another, none of which is what runs. Only one of them ever does.

   Nothing here is told which is which. The fused kernel declares the chain it replaces
   (RadFuseFn), the page expands each neighbour into its own chain, and where the concatenation is
   equal the two are alternatives by construction. */
.alt { position:relative; width:100%; max-width:600px; border:1px dashed var(--ghost);
       border-radius:11px; padding:10px; display:flex; flex-direction:column; gap:8px; }
.alt > .cap { position:absolute; top:-7px; left:12px; background:var(--bg); padding:0 6px;
              font-size:9px; }
.altrow { display:flex; align-items:stretch; gap:10px; }
.altrow > .lane { flex:1; min-width:0; display:flex; flex-direction:column; gap:6px; }
.altrow > .when { flex:none; width:112px; align-self:center; text-align:right;
                  font:400 11px/1.4 var(--mono); color:var(--faint); }
.altor { display:flex; align-items:center; gap:9px; color:var(--faint);
         font:500 9.5px/1 var(--ui); letter-spacing:.14em; text-transform:uppercase; }
.altor::before, .altor::after { content:""; height:1px; background:var(--rule); flex:1; }
/* the unfused lane is a chain of small pills: four ops that never all run are four ops that should
   not each take a full card */
.chain { display:flex; flex-wrap:wrap; align-items:center; gap:5px; }
.pill { font:400 11.5px/1 var(--mono); color:var(--dim); background:var(--card);
        border:1px solid var(--edge); border-radius:6px; padding:6px 8px; cursor:pointer; }
.pill:hover { color:var(--fg); border-color:#333; }
.pill.sel { color:var(--fg); border-color:#4a4a4a; background:var(--hover); }
.chain .arr { color:var(--faint); font-size:11px; }

.gnode { position:relative; z-index:2; background:var(--card); border:1px solid var(--edge);
         border-radius:9px; padding:10px 13px; cursor:pointer; width:100%;
         transition:border-color .12s, background .12s; }
.gnode:hover { border-color:#333; background:var(--hover); }
.gnode.sel { border-color:#4a4a4a; background:var(--hover); }
.gnode .t { display:flex; align-items:baseline; gap:9px; flex-wrap:wrap; }
.gnode .on { font:500 12.5px/1.2 var(--mono); }
.gnode .kn { font:400 11px/1.2 var(--mono); color:var(--dim); margin-left:auto; }
.gnode .kn i { font-style:normal; color:#8dbcfb; }
.gnode .kn i.ref { color:var(--bad); }
.gnode .chips { display:flex; gap:5px; flex-wrap:wrap; margin-top:7px; }
.chip { font:400 9.5px/1 var(--mono); letter-spacing:.02em; color:var(--dim);
        background:var(--track); border-radius:4px; padding:3px 6px; }
.chip.fuse { color:#7fe0a3; background:rgba(74,222,128,.1); }
.chip.band { color:#e9c065; background:rgba(251,191,36,.1); }
.chip.dead { color:var(--faint); }

/* ---------------------------------------------------------------- the inspector
   Everything the core knows about ONE op. The flow carries the shape; this carries the detail,
   and the split is the whole reason the detail can be this dense -- none of it costs a reader who
   has not pointed at anything. */
.ginfo { position:sticky; top:14px; }
.ginfo .pbody { max-height:calc(100vh - 116px); overflow:auto; padding:0; }
.gsec { padding:13px 16px; border-top:1px solid var(--rule); }
.gsec:first-child { border-top:0; }
.gsec > .h { font-size:10.5px; font-weight:600; letter-spacing:.11em; text-transform:uppercase;
             color:var(--faint); margin-bottom:9px; }

.gtitle { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
.gtitle .nm { font:500 17px/1.15 var(--mono); letter-spacing:-.01em; word-break:break-all; }
.gtitle .ct { font:500 11px/1 var(--ui); color:var(--fg); background:var(--track);
              border-radius:5px; padding:4px 7px; }
.gwhere { margin-top:8px; color:var(--faint); font:400 11.5px/1.5 var(--mono); }
.gdoc { margin-top:11px; color:var(--dim); font:400 12.5px/1.6 var(--ui); }

/* a two-column key/value list; the key column is monospace so keys line up down the panel */
.mkv { display:grid; grid-template-columns:auto minmax(0,1fr); gap:4px 14px; align-items:baseline; }
.mkv > dt { color:var(--faint); font:400 11.5px/1.55 var(--mono); }
.mkv > dd { margin:0; font:400 12px/1.55 var(--mono); word-break:break-word; }
.mkv > dd em { font-style:normal; color:var(--faint); font-size:10.5px; margin-left:6px; }

.mtags { display:flex; flex-wrap:wrap; gap:5px; }
.mtag { font:400 11px/1.1 var(--mono); color:var(--dim); background:var(--track);
       border-radius:4px; padding:4px 7px; }
.mtag b { font-weight:400; color:var(--fg); }
.mtag.out  { color:#7fe0a3; background:rgba(74,222,128,.09); }
.mtag.wgt  { color:#e9c065; background:rgba(251,191,36,.09); }
.mtag.opt  { color:var(--faint); }

/* one band: the winner as a card, the losers as a list under it */
.bandbox { border:1px solid var(--edge); border-radius:9px; background:#070707;
           margin-top:9px; overflow:hidden; }
.bandbox:first-of-type { margin-top:0; }
.bandbox > .bh { display:flex; align-items:baseline; gap:8px; padding:8px 11px;
                 background:var(--card); border-bottom:1px solid var(--rule); }
.bandbox > .bh .sp { font:500 12px/1 var(--mono); color:#e9c065; }
.bandbox > .bh .dm { margin-left:auto; font:400 10px/1 var(--ui); letter-spacing:.1em;
                     text-transform:uppercase; color:var(--faint); }
.bandbox > .bb { padding:10px 11px; }
.win { display:flex; align-items:baseline; gap:8px; flex-wrap:wrap; }
.win .k { font:500 13px/1.25 var(--mono); word-break:break-all; }
.win .lib { font:400 11.5px/1 var(--mono); color:#8dbcfb; }
.win .lib.ref { color:var(--bad); }
.win .pr { font:400 10.5px/1 var(--mono); color:var(--faint); margin-left:auto; }
.prose { margin-top:8px; color:var(--dim); font:400 12px/1.55 var(--ui); }
.msub { margin-top:8px; }
.msub > .sh { font:400 10px/1 var(--ui); letter-spacing:.1em; text-transform:uppercase;
             color:var(--faint); margin-bottom:5px; }

.cands { display:grid; gap:3px; }
.cand { display:grid; grid-template-columns:auto auto minmax(0,1fr); gap:4px 10px;
        align-items:baseline; font:400 11.5px/1.5 var(--mono); }
.cand .m { color:var(--faint); }
.cand.ok  .m { color:var(--ok); }
.cand .kk { color:var(--dim); word-break:break-all; }
.cand.ok .kk { color:var(--fg); }
.cand .wy { color:var(--faint); word-break:break-word; }

/* the fused chain, as the little vertical diagram it is */
.fchain { display:grid; gap:0; }
.fstep { display:grid; grid-template-columns:16px minmax(0,1fr); gap:9px; align-items:baseline;
         padding:5px 0; }
.fstep + .fstep { border-top:1px solid var(--rule); }
.fstep .i { font:400 9.5px/1.5 var(--mono); color:var(--ghost); }
.fstep .o { font:400 12px/1.4 var(--mono); color:#7fe0a3; }
.fstep .g { font:400 11px/1.45 var(--mono); color:var(--faint); word-break:break-word; }

.wlist { display:grid; gap:8px; }
.witem { display:grid; gap:3px; }
.witem .wn { font:400 12px/1.35 var(--mono); word-break:break-all; }
.witem .wm { display:flex; flex-wrap:wrap; gap:4px 12px; font:400 11px/1.45 var(--mono);
             color:var(--faint); }
.witem .wm b { font-weight:400; color:var(--dim); }
.bar { height:3px; border-radius:2px; background:var(--track); overflow:hidden; }
.bar i { display:block; height:100%; background:#3f6fa8; }

.gfilter { display:flex; gap:10px; align-items:center; }
.gfilter input { flex:1; min-width:0; background:var(--bg); border:1px solid var(--edge);
                 border-radius:6px; color:var(--fg); font:12px var(--mono); padding:7px 10px; }
.gfilter input:focus { outline:none; border-color:#3a3a3a; }
.libs { display:flex; flex-wrap:wrap; gap:6px; margin-top:12px; }
/* what the architecture plugin said about the model it built, and what did not resolve. Both are
   in the graph already and neither had anywhere to be read. */
#gmetawrap { margin-top:13px; padding-top:12px; border-top:1px solid var(--rule); }
#gmetawrap > summary { cursor:pointer; color:var(--dim); font:400 11.5px/1.4 var(--ui);
                       list-style:none; display:flex; align-items:center; gap:8px; }
#gmetawrap > summary::-webkit-details-marker { display:none; }
#gmetawrap > summary::before { content:"▸"; color:var(--faint); font:400 9px/1 var(--mono); }
#gmetawrap[open] > summary::before { content:"▾"; }
#gmetawrap > summary:hover { color:var(--fg); }
.gmeta { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px 24px;
         margin-top:14px; }
.gmeta section > .h { font-size:10px; font-weight:600; letter-spacing:.11em;
                      text-transform:uppercase; color:var(--faint); margin-bottom:8px; }
.gmeta li { list-style:none; color:var(--dim); font:400 11.5px/1.6 var(--ui);
            padding-left:13px; text-indent:-13px; }
.gmeta li::before { content:"·"; color:var(--ghost); margin-right:7px; }
.gmeta ul { margin:0; padding:0; }
.gmeta .bad li { color:#f29d9d; }
.lib { display:flex; align-items:baseline; gap:7px; border:1px solid var(--edge);
       border-radius:7px; padding:6px 9px; background:#070707; }
.lib .o { font:400 9.5px/1 var(--mono); color:var(--faint); }
.lib .n { font:500 11.5px/1 var(--mono); }
.lib .c { font:400 10.5px/1 var(--mono); color:var(--dim); }

/* ---------------------------------------------------------------- the expert panels
   A STACKED PROPORTION BAR plus a named row per segment. The bar answers "how much of it is
   where" at a glance and the rows carry the numbers, which is the split the rest of this page
   already uses -- a legend of five colours with no figures beside them is decoration. */
/* THE CAVEAT LINE IS ALWAYS THERE, because it is not always the same caveat. It says what the
   shares below are measured over, and that changes with the link: hiding it when the link is
   busy moved the bar, the rows, the chart and the counters under it by its own height, every
   time the engine went quiet and came back. */
.hint { font:400 11px/1.45 var(--ui); color:var(--faint); margin-bottom:9px; min-height:16px; }
.sbar { display:flex; height:9px; border-radius:5px; overflow:hidden; background:var(--track); }
.sbar > i { display:block; height:100%; min-width:0;
            transition:width .25s cubic-bezier(.4,0,.2,1); }
.sbar > i + i { border-left:1px solid var(--card); }
.rows { margin-top:13px; }
/* THE FIGURE COLUMNS ARE FIXED, NOT max-content. Each row is its own grid, so content-sized
   columns were sized per row: the three figure columns landed at a different x in every row of
   the same panel, and every poll that changed a digit count moved them again. A decimal point
   that will not stay still cannot be read down the list, and the widest row was also setting the
   panel's min-content and stretching the whole column with it. */
.rows > div { display:grid;
              grid-template-columns:9px minmax(0,1fr) 76px 68px 46px;
              gap:0 8px; align-items:baseline; padding:6px 0; border-top:1px solid var(--rule);
              font:400 11.5px/1.5 var(--mono); font-variant-numeric:tabular-nums; }
.rows > div:first-child { border-top:0; padding-top:0; }
.rows > div.zero { opacity:.38; }
.rows b { width:7px; height:7px; border-radius:2px; display:block; align-self:center; }
.rows .nm { font:400 12px/1.45 var(--ui); color:var(--dim); overflow:hidden;
            text-overflow:ellipsis; white-space:nowrap; }
.rows .pc { color:var(--faint); text-align:right; }
.rows .am { text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.sub { margin-top:16px; padding-top:14px; border-top:1px solid var(--rule); }
/* A SUB-SECTION'S FIGURE IS A DETAIL AND MAY BE TRIMMED. The headline meters' values are sized
   to fit; these are sentences of several numbers whose length is data, and a sentence allowed to
   set the column width sets it again on the poll that lengthens it. */
.sub .mrow .v { min-width:0; overflow:hidden; text-overflow:ellipsis; }

/* RESIDENCY PER LAYER, one bar a layer. This is the picture the global percentage cannot draw:
   a layer-major planner fill leaves the last layers starved while the total still looks healthy.
   The weakest layer is coloured, because it is the one that misses on nearly every token however
   good the ranking inside the others is. */
.lbars { display:flex; align-items:flex-end; gap:2px; height:46px; }
.lbars > i { display:block; flex:1; min-width:1px; background:#5e5e5e; border-radius:1px 1px 0 0; }
.lbars > i.min { background:var(--warn); }
.lbars > i.full { background:#f5f5f5; }
.xlab { display:flex; justify-content:space-between; font:400 9.5px/1 var(--mono);
        color:var(--faint); padding-top:7px; }

/* MOVES OVER TIME. Two rates on one axis -- promotions and demotions -- because the interesting
   reading is the RELATION between them: equal and sustained is thrash, promotions alone is the
   engine still filling, both near zero is a settled working set. */
/* min-width:0 because an inline SVG is a replaced element: its min-content is the viewBox's own
   300px whatever `width` says, so without this the chart -- not the figures, the chart -- was
   what decided how wide the column holding it had to be. */
.hchart { display:block; width:100%; height:56px; min-width:0; }
/* The chart's scale is the window's peak, labelled on its axis as the draft chart's is. */
.yax.hy { height:56px; }
.hgrid { stroke:var(--rule); stroke-width:1; vector-effect:non-scaling-stroke; }

/* The full width of the card, cells square: the canvas's own bitmap sets the aspect ratio. */
.plane { display:block; width:100%; height:auto; image-rendering:pixelated;
         image-rendering:crisp-edges; }
</style></head><body>
<div class="wrap">
  <header>
    <span class="dot" id="dot"></span>
    <h1 id="title">radiance</h1>
    <span class="arch" id="arch"></span>
    <nav>
      <button id="tab-dash" class="on">Dashboard</button>
      <button id="tab-sess">Sessions</button>
      <button id="tab-model">Model</button>
      <button id="tab-adv">Advanced</button>
    </nav>
    <input id="key" type="password" placeholder="API key" autocomplete="off" hidden>
    <span class="ver" id="ver"></span>
  </header>
  <div id="view-dash" class="boot">

  <!-- THE CELLS ARE IN THE DOCUMENT, NOT BUILT BY THE FIRST POLL. A strip assembled in script
       is absent until a fetch returns, and everything below it starts one strip higher than it
       ends up: on this page that was the first of several hundred pixels of movement after the
       page had already been drawn. The values are placeholders; only they are filled in. -->
  <div class="strip" id="strip">
    <div class="cell"><div class="n" id="m-step">–<u>ms</u></div>
         <span class="cap">decode step</span>
         <span class="det" id="m-stepdet"></span>
         <svg viewBox="0 0 200 22" preserveAspectRatio="none"><path/></svg></div>
    <div class="cell"><div class="n" id="m-decode">–<u>tok/s</u></div>
         <span class="cap">decode</span>
         <span class="det" id="m-decodedet"></span>
         <svg viewBox="0 0 200 22" preserveAspectRatio="none"><path/></svg></div>
    <div class="cell"><div class="n" id="m-prefill">–<u>tok/s</u></div>
         <span class="cap">prefill</span>
         <span class="det" id="m-prefilldet"></span>
         <svg viewBox="0 0 200 22" preserveAspectRatio="none"><path/></svg></div>
    <!-- REQUESTS, AND THE CAPTION SAYS SO. "In flight" beside a token-rate cell and above a
         memory panel that also uses the phrase was read as tokens. -->
    <div class="cell"><div class="n" id="m-flight">–<u></u></div>
         <span class="cap">requests in flight</span>
         <span class="det" id="m-flightdet"></span>
         <svg viewBox="0 0 200 22" preserveAspectRatio="none"><path/></svg></div>
    <!-- VRAM, NOT A SHARE OF THE TOKEN POOL. What the cache costs is memory the weights could
         otherwise have had, and a percentage of the paged pool's token capacity says nothing
         about it: it is a preemption signal, so it is kept, underneath, where it reads as one. -->
    <div class="cell"><div class="n" id="m-kv">–<u></u></div>
         <span class="cap" id="m-kvcap">kv cache in vram</span>
         <span class="det" id="m-kvdet"></span>
         <svg viewBox="0 0 200 22" preserveAspectRatio="none"><path/></svg></div>
    <div class="cell"><div class="n" id="m-accept">–<u>%</u></div>
         <span class="cap">acceptance</span>
         <span class="det" id="m-acceptdet"></span>
         <svg viewBox="0 0 200 22" preserveAspectRatio="none"><path/></svg></div>
  </div>

  <div class="body">
    <div class="stack">
      <section class="panel" id="reqpanel">
        <div class="phead"><span class="pleft"><span class="cap">Requests</span>
                           <button type="button" class="pbtn" id="reqjson"
                                   title="Download JSON file" aria-label="Download JSON file">
                             <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true">
                               <!-- The tray bar's stroke bottom lands exactly on the viewBox
                                    bottom, so baseline alignment puts the bar on the text
                                    baseline. An inset bar floats a hair above the caps. -->
                               <path d="M8 2v9m0 0L4.6 7.6M8 11l3.4-3.4M2.2 15.2h11.6"
                                     fill="none" stroke="currentColor" stroke-width="1.6"
                                     stroke-linecap="round" stroke-linejoin="round"/>
                             </svg></button></span>
                           <span class="aside" id="reqaside"></span></div>
        <div class="pbody">
          <div class="hold">
            <table>
              <thead><tr id="rhead"></tr></thead>
              <tbody id="rbody"></tbody>
            </table>
            <div class="empty" id="rempty">nothing in flight</div>
          </div>
          <div id="hist">
            <div class="hhead"><span class="cap">history</span>
              <span class="hsum" id="hsum" title="mean of the per-request figures over all captured rows; each request counts once, so this is not the pooled rate the DECODE tile shows. last 20 and last 5 are the same mean over the most recent 20 and 5 requests. prefill is fresh prompt tokens over the time from admission to first token, so queue wait is not charged to it; on an engine that reports no admission time it falls back to time to first token, which does include queue wait. ttft is the median time to first token, including queue wait."></span>
              <span class="hbtn" id="hclr" title="clear the stored history">clear</span></div>
            <div class="hlist">
              <table>
                <thead><tr id="hhead"></tr></thead>
                <tbody id="hbody"></tbody>
              </table>
              <div class="empty" id="hempty">no completed requests captured yet</div>
            </div>
          </div>
        </div>
      </section>

      <div class="pair" id="pair" hidden>
        <section class="panel" id="draftpanel">
          <div class="phead"><span class="cap">Draft acceptance by position</span>
                             <span class="aside" id="draftaside"></span></div>
          <div class="pbody">
            <div class="chart">
              <div class="yax"><span>100</span><span>50</span><span>0</span></div>
              <div class="plot">
                <div class="rules"><i></i><i></i><i></i></div>
                <div class="cols" id="cols"></div>
              </div>
              <div></div><div class="xax" id="xax"></div>
              <div class="nkey">reach</div><div class="nax" id="nax"></div>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="phead"><span class="cap">Speculation</span>
                             <span class="aside" id="specaside"></span></div>
          <div class="pbody">
            <div class="duo">
              <div><div class="n" id="tps_n">–<u>tok</u></div>
                   <span class="cap">per step</span></div>
              <div><div class="n" id="acc_n">–<u>%</u></div>
                   <span class="cap">conditionally accepted</span></div>
            </div>
            <div class="kv" id="spec"></div>
          </div>
        </section>
      </div>

      <section class="panel" id="exppanel" hidden>
        <div class="phead"><span class="cap">Expert residency</span>
                           <span class="aside" id="expaside"></span></div>
        <div class="pbody">
          <div class="sbar" id="tierbar"></div>
          <div class="rows" id="tierrows"></div>

          <div class="sub">
            <div class="mrow"><div class="t">Routed to a resident expert<b> hit rate</b></div>
                              <div class="v" id="hitv">–</div></div>
            <div class="track"><i id="hiti"></i></div>
          </div>

          <div class="sub">
            <div class="mrow"><div class="t">Resident per layer</div>
                              <div class="v" id="layv">–</div></div>
            <div class="lbars" id="lbars"></div>
            <div class="xlab" id="laylab"><span>L0</span><span></span></div>
          </div>

          <div class="sub">
            <div class="mrow"><div class="t">Placement plane<b> layer × expert</b></div>
                              <div class="v" id="planev">–</div></div>
            <canvas class="plane" id="plane"></canvas>
          </div>
        </div>
      </section>

    </div>

    <div class="stack">
      <section class="panel" id="batchpanel">
        <div class="phead"><span class="cap">Batch</span>
                           <span class="aside" id="batchaside"></span></div>
        <div class="pbody">
          <div class="slots" id="slots"></div>
          <div class="legend">
            <span><b style="background:#4ade80"></b>decode</span>
            <span><b style="background:#60a5fa"></b>prefill</span>
            <span><b style="background:#fbbf24"></b>waiting</span>
            <span><b style="background:#f87171"></b>preempted</span>
            <span><b style="background:#1e1e1e"></b>free</span>
          </div>
          <div class="qbar">
            <div class="mrow"><div class="t">Admission queue</div>
                              <div class="v" id="qv">–</div></div>
            <div class="track"><i id="qi"></i></div>
          </div>
          <div class="qbar"><div class="kv" id="totals"></div></div>
        </div>
      </section>

      <section class="panel" id="mempanel">
        <div class="phead"><span class="cap">Memory</span></div>
        <div class="pbody" id="mem"></div>
      </section>

      <section class="panel" id="linkpanel" hidden>
        <div class="phead"><span class="cap">Interconnect</span>
                           <span class="aside" id="linkaside"></span></div>
        <div class="pbody">
          <div class="duo">
            <div><div class="n" id="lnk_n">–<u>MB/s</u></div>
                 <span class="cap">link, all cards</span></div>
            <div id="mvcell"><div class="n" id="mvr_n">–<u>/s</u></div>
                 <span class="cap">expert moves</span></div>
          </div>
          <div class="hint" id="linkhint"></div>
          <div class="sbar" id="linkbar"></div>
          <div class="rows" id="linkrows"></div>

          <div class="sub" id="movesub">
            <div class="mrow"><div class="t">Moves a second<b> 60s</b></div>
                              <div class="v" id="movev">–</div></div>
            <div class="chart">
              <div class="yax hy"><span id="movepk">–</span><span>0</span></div>
              <svg class="hchart" id="movechart" viewBox="0 0 300 56"
                   preserveAspectRatio="none"></svg>
            </div>
          </div>

          <div class="sub" id="movekvsub"><div class="kv" id="movekv"></div></div>
        </div>
      </section>
    </div>
  </div>

  </div>

  <!-- CHOSEN ONCE AND THEN LEFT ALONE. A view of its own, reached from the header like the others:
       these settings are not read alongside the figures, and at the foot of the dashboard column
       they sat under everything a reader scrolls past to get to them. -->
  <div id="view-adv" hidden>
    <section class="panel">
      <div class="phead"><span class="cap">Advanced</span>
                         <span class="aside" id="advsum"></span></div>
      <div class="pbody">
        <label class="pollrow" for="pollidle">Idle interval
          <input id="pollidle" type="number" min="100" max="5000" step="50"> ms</label>
        <label class="pollrow" style="margin-top:12px" for="pollbusy">Busy interval
          <input id="pollbusy" type="number" min="100" max="5000" step="50"> ms</label>
        <div class="aside pollnote">Idle is how often the page fetches when no request is running;
          busy is how often while one is in flight. Set them equal for a single fixed rate.
          100&ndash;5000 ms each.</div>
      </div>
    </section>
  </div>

  <!-- WHAT HAPPENS IF A STORED CONVERSATION COMES BACK. The strip says how well the cache is
       serving and where what it holds is; the tables say, a conversation a row, what its next
       turn reuses and what it prefills. Every figure is tokens or bytes. -->
  <div id="view-sess" hidden>
    <div class="strip">
      <div class="cell"><div class="n" id="s-hit">–<u>%</u></div>
           <span class="cap">of prompt tokens from cache</span>
           <span class="det" id="s-hitdet"></span></div>
      <div class="cell"><div class="n" id="s-live">–<u id="s-liveof"></u></div>
           <span class="cap">conversations can resume</span>
           <span class="det" id="s-livedet"></span></div>
      <div class="cell"><div class="n" id="s-vram">–<u>tok</u></div>
           <span class="cap" id="s-vramcap">context in vram</span>
           <span class="det" id="s-vramdet"></span></div>
      <div class="cell" id="s-hostcell"><div class="n" id="s-host">–<u></u></div>
           <span class="cap">moved to host ram</span>
           <span class="det" id="s-hostdet"></span></div>
    </div>

    <section class="panel">
      <div class="phead"><span class="cap" id="s-livehead">Will resume</span>
                         <span class="aside">most recently active first</span></div>
      <div class="pbody">
        <table class="stbl"><thead><tr id="s-livecols"></tr></thead>
               <tbody id="s-liverows"></tbody></table>
        <div class="empty" id="s-liveempty" hidden></div>
      </div>
    </section>

    <details class="panel sgrp" id="s-coldgrp" hidden>
      <summary class="phead"><span class="cap" id="s-coldhead">Will start over</span>
        <span class="aside">context kept, but no saved state to resume from: a return prefills
                            in full</span></summary>
      <div class="pbody">
        <table class="stbl"><thead><tr id="s-coldcols"></tr></thead>
               <tbody id="s-coldrows"></tbody></table>
      </div>
    </details>
  </div>

  <div id="view-model" hidden>
    <div class="strip" id="gstrip"></div>
    <!-- The control vector (core/cvec.h): one direction per layer, projected out of the residual
         stream at run time. Shown only on a deployment that found one; flipping it needs no
         reload, and every request that names its own choice overrides it. -->
    <section class="panel" id="cvpanel" hidden>
      <div class="phead">
        <span class="cap">Control vector</span>
        <span class="aside" id="cvsum"></span>
      </div>
      <div class="pbody">
        <label class="cvrow">
          <input id="cvbox" type="checkbox">
          <span id="cvlabel">Remove safety behavior</span>
        </label>
        <div class="aside" id="cvnote"></div>
      </div>
    </section>
    <section class="panel">
      <div class="phead">
        <span class="cap">Kernel libraries</span>
        <span class="aside" id="gsum"></span>
      </div>
      <div class="pbody">
        <div class="gfilter">
          <input id="gq" placeholder="highlight by op, kernel, library, constraint or fusion"
                 spellcheck="false">
          <span class="aside" id="gcount"></span>
        </div>
        <div class="libs" id="glibs"></div>
        <details id="gmetawrap"><summary id="gmetasum"></summary>
          <div id="gmeta"></div>
        </details>
      </div>
    </section>
    <div class="gwrap" style="margin-top:14px">
      <section class="panel"><div class="pbody"><div class="flow" id="flow"></div></div></section>
      <section class="panel ginfo">
        <div class="phead"><span class="cap">Selected op</span>
                           <span class="aside" id="ginfoaside"></span></div>
        <div class="pbody" id="ginfo"></div>
      </section>
    </div>
  </div>
</div>
<script>
"use strict";
/* BUILT ONCE, UPDATED IN PLACE. Rebuilding the DOM on every poll re-lays out the document twice a
 * second, drops the reader's text selection and flickers -- on the machine the engine is being
 * watched from. Nodes are created when their SHAPE changes (a draft depth, a card count, a
 * request arriving or leaving); after that only textContent and one style property are written,
 * and polling stops entirely while the tab is hidden. */
const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";

/* THE SERVER'S BEARER KEY, when it was started with one. This page is served without it and every
 * fetch of data sends the key this browser was given; a 401 shows the field in the header. The key
 * is kept in this browser's storage and sent nowhere but this server. */
let apiKey = "";
try { apiKey = localStorage.getItem("radiance.key") || ""; } catch (e) {}
async function api(path) {
  const r = await fetch(path, { cache: "no-store",
                                headers: apiKey ? { Authorization: "Bearer " + apiKey } : {} });
  if (r.status === 401) { $("key").hidden = false; throw new Error("401"); }
  return r;
}
$("key").addEventListener("change", (e) => {
  apiKey = e.target.value.trim();
  try { localStorage.setItem("radiance.key", apiKey); } catch (err) {}
  e.target.value = "";
  e.target.hidden = true;
  identify();
  poll();
  /* The Model view's controls are fetched on demand, so a key entered after the tab was opened
   * has to re-ask: otherwise the control-vector panel stays in its "sign in" state until the
   * tab is left and re-opened. */
  loadControlVector();
});
const HIST_S = 60;   /* the sparkline's window in SECONDS: the span it draws and prunes to */

const el = (t, c, x) => { const e = document.createElement(t);
                          if (c) e.className = c; if (x !== undefined) e.textContent = x;
                          return e; };
const sv = (t, a) => { const e = document.createElementNS(NS, t);
                       for (const k in a) e.setAttribute(k, a[k]); return e; };
/* ---------------------------------------------------------------- numbers
 *
 * EVERY FIGURE ON THE PAGE GOES THROUGH THIS BLOCK, so two figures in one panel cannot disagree
 * about how many digits they are entitled to or which unit they are in. The rules:
 *
 * THREE SIGNIFICANT FIGURES, AT MOST TWO DECIMALS, ROUNDED BEFORE THE UNIT IS CHOSEN. Choosing
 * the unit first prints 1023.7 MiB as "1024 MiB" and 999.96k as "1000.0k": a figure one digit
 * wider than its column, in a unit the next one up exists to replace. Three figures is also what
 * keeps a four-digit throughput beside a two-digit one readable at a glance.
 *
 * SIZES IN BINARY UNITS -- B, KiB, MiB, GiB, TiB -- because the card's memory, the cache carve and
 * every --*-mib flag count in them. RATES OF BYTES IN DECIMAL SI -- B/s, kB/s, MB/s, GB/s --
 * because link and disk bandwidth are specified that way, and a rate is read against a ceiling
 * quoted in those units.
 *
 * COUNTS TAKE SI PREFIXES -- k, M, G, T -- where a count is glanced at. Where one is READ (a
 * cumulative counter, a row of a table) it is exact and grouped in threes by a narrow space, as
 * SI writes it: a locale's separator is a comma in one browser and a full stop in another, and a
 * full stop beside this page's decimal points makes "1.234" mean two different numbers. Four
 * digits are left ungrouped, also as SI writes them.
 *
 * A SHARE NEVER ROUNDS ONTO 0% OR 100%. A pool at 99.96% is not full and a layer at 0.04% is not
 * empty, and those are exactly the two readings rounding would put on the screen. */
const DASH = "–", NNBSP = "\u202f";
const PFX = ["", "k", "M", "G", "T", "P"];
const IEC = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
const SIR = ["B/s", "kB/s", "MB/s", "GB/s", "TB/s", "PB/s"];
const finite = (v) => v != null && isFinite(v);

/* The digits of 0 <= a < 999.5: three significant figures, never more than two decimals. The
 * thresholds are where the rounded value gains a digit, so 99.96 is "100", not "100.0". */
const sig3 = (a) => a.toFixed(a >= 99.95 ? 0 : a >= 9.995 ? 1 : 2);
/* A value on a unit ladder, as { n, u }. It steps up while the value would ROUND to four
 * digits, which is the rule above; `exact` prints the ladder's first rung as a whole number,
 * for bytes and counts, where "5.00 B" is a precision that does not exist. */
function ladder(v, base, units, exact) {
  if (!finite(v)) return { n: DASH, u: "" };
  if (v === 0) return { n: "0", u: units[0] };        /* nothing is not "0.00" of anything */
  let a = Math.abs(v), i = 0;
  while (a >= 999.5 && i < units.length - 1) { a /= base; i++; }
  const n = i === 0 && exact ? String(Math.round(a)) : sig3(a);
  return { n: (v < 0 && n !== "0" ? "-" : "") + n, u: units[i] };
}
const withUnit = (x) => x.u ? x.n + " " + x.u : x.n;
/* A size, whole or split: a strip cell sets the figure and its unit in different type. */
const hbs  = (b) => ladder(b, 1024, IEC, true);
const hb   = (b) => withUnit(hbs(b));
/* Two sizes that are one quantity's used and total: the unit once when it is the same unit. */
const hbOf = (a, b, sep) => { const x = hbs(a), y = hbs(b);
                              return x.u === y.u ? x.n + sep + y.n + " " + y.u
                                                 : withUnit(x) + sep + withUnit(y); };
/* A rate of bytes, whole or split. */
const rates = (b) => ladder(b, 1000, SIR, true);
const rate  = (b) => withUnit(rates(b));
/* A count to glance at: exact below a thousand, then three figures and a prefix. */
const cnt3 = (n) => { const x = ladder(n, 1000, PFX, true); return x.n + x.u; };
/* A continuous quantity -- moves a second, misses a second -- to three figures and a prefix. */
const num3 = (v) => { const x = ladder(v, 1000, PFX, false); return x.n + x.u; };
/* A count to read: exact, grouped in threes past four digits. */
function cnt(n) {
  if (!finite(n)) return DASH;
  const s = String(Math.round(Math.abs(n)));
  let o = "";
  for (let i = 0; i < s.length; i++) {
    if (i && s.length > 4 && (s.length - i) % 3 === 0) o += NNBSP;
    o += s[i];
  }
  return (n < 0 && o !== "0" ? "-" : "") + o;
}
/* Tokens a second, never scaled by a prefix: rates are read against each other digit by digit,
 * and 6.12k beside 5.98k hides what 6116 beside 5979 shows. Whole from a hundred up. */
const tps = (v) => !finite(v) ? DASH : v === 0 ? "0" : v >= 99.95 ? cnt(v) : sig3(v);
/* A share of one, as a percentage to `d` decimals without the sign -- for a cell that sets the
 * sign as its unit -- and with it. Only exactly 0 reads 0 and only exactly 1 reads 100. */
function pctn(f, d = 1) {
  if (!finite(f)) return DASH;
  const p = 100 * f, s = p.toFixed(d);
  if (p > 0 && +s === 0)   return "<" + (d ? "0." + "0".repeat(d - 1) + "1" : "1");
  if (p < 100 && +s === 100) return ">" + (d ? "99." + "9".repeat(d) : "99");
  return s;
}
const pct = (f, d = 1) => { const s = pctn(f, d); return s === DASH ? s : s + "%"; };
/* A duration, one ladder for every one on the page: a request's age, a time to first token, how
 * long a conversation has been idle, how long the server has been up. Rounded to the resolution
 * it is shown at BEFORE it is split, so 59.97 s is "1m 00s" and not "60.0 s" or "0m 60s". */
function dur(s) {
  if (!finite(s)) return DASH;
  if (s < 0.9995) return Math.round(Math.max(0, s) * 1000) + " ms";
  if (s < 9.995)  return s.toFixed(2) + " s";
  if (s < 59.95)  return s.toFixed(1) + " s";
  const two = (x) => String(x).padStart(2, "0");
  const t = Math.round(s);
  if (t < 3600)  return Math.floor(t / 60) + "m " + two(t % 60) + "s";
  const m = Math.round(s / 60);
  if (m < 1440)  return Math.floor(m / 60) + "h " + two(m % 60) + "m";
  const h = Math.round(s / 3600);
  return Math.floor(h / 24) + "d " + two(h % 24) + "h";
}
/* THE STEP TIME KEEPS TWO DECIMALS below 100 ms, as { n, u }. It is the page's one performance
 * figure and it is compared across runs at hundredths of a millisecond; three significant
 * figures would hide exactly the differences it is read for. Past a second it is seconds. */
const stepTime = (ms) => !finite(ms) ? { n: DASH, u: "ms" }
                       : ms < 99.995 ? { n: ms.toFixed(2), u: "ms" }
                       : ms < 999.95 ? { n: ms.toFixed(1), u: "ms" }
                       : { n: sig3(Math.min(ms / 1000, 999)), u: "s" };

/* ---------------------------------------------------------------- metric strip */
/* The caption and the unit live in the document beside the cell they label. Holding a second
 * copy of them here would be holding a copy that can differ. */
const METRICS = ["step", "decode", "prefill", "flight", "kv", "accept"];
/* One monotonic stamp per poll: every ring is pruned and drawn against it, so a window means
 * the same span of seconds whatever interval the browser polls at. performance.now() so a wall
 * clock adjustment cannot stretch or shrink a window. */
let nowS = 0;
const M = {}, hist = {};
for (const k of METRICS) {
  hist[k] = [];
  const n = $("m-" + k);
  M[k] = { n, u: n.querySelector("u"), line: n.parentNode.querySelector("path") };
}
/* The axis follows the WINDOW's own minimum and maximum, not zero. This is a sparkline: what it
 * is for is the variation, and the number above it already states the level -- scaled from zero
 * every steady state is a flat line on the floor. No variation draws down the middle.
 *
 * The ends of the range are the 5th and 95th percentiles, not the extremes: on a min/max axis one
 * rare spike owns the box and the typical variation is squeezed into a pixel. A spike past the
 * 95th draws pinned to the top; the number above carries the level and the detail line the
 * extremes. The stroke is 1.25 units wide, so a one-unit inset keeps it off the edge. */
function spark(key, pts) {
  const s = M[key], w = 200, h = 22;
  if (pts.length < 3) { s.line.setAttribute("d", ""); return; }
  const vals = pts.map(p => p.v);
  const q = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)))];
  const sorted = [...vals].sort((a, b) => a - b);
  let lo = q(sorted, 0.05), hi = q(sorted, 0.95);
  /* A window that is one value apart from rare spikes puts both percentiles on that value; the
   * extremes then serve, so the spikes still show. */
  if (!(hi > lo)) { lo = sorted[0]; hi = sorted[sorted.length - 1]; }
  const span = hi - lo;
  /* X IS TIME, NOT SAMPLE INDEX. Index spacing is only honest while every sample is the same age
   * apart; once the poll rate varies -- adaptive, or a browser throttling a busy tab -- a busy
   * stretch of 100 ms samples would be drawn five times wider than the seconds it took. The newest
   * sample sits on the right edge and the window spans HIST_S back from it, so a stretch gets the
   * room its seconds deserve and a gap in the samples shows as a gap in the line. */
  const t1 = pts[pts.length - 1].t;
  let d = "";
  for (let i = 0; i < pts.length; i++) {
    let t = span > Math.abs(hi) * 1e-3 ? (vals[i] - lo) / span : 0.5;
    if (t < 0) t = 0; else if (t > 1) t = 1;
    const x = Math.min(w, Math.max(0, w * (1 - (t1 - pts[i].t) / HIST_S)));
    d += (i ? "L" : "M") + x.toFixed(1) + "," + (h - 1 - (h - 2) * t).toFixed(1);
  }
  s.line.setAttribute("d", d);
}
function setMetric(key, value, text) {
  M[key].n.firstChild.nodeValue = text;
  if (value != null) {
    const r = hist[key];
    r.push({ t: nowS, v: value });
    while (r.length && r[0].t < nowS - HIST_S) r.shift();
  }
  spark(key, hist[key]);
}

/* Best / recent / now on the decode, prefill, in-flight and acceptance tiles. The engine
 * publishes a floor and a recent window only for the decode step; for these it publishes one
 * figure per poll, so the three figures are the page's own:
 *   best / highest - the extreme this page has seen at the current concurrency; it resets on
 *                    reload AND on a server restart (a new started_at runs newEngineRun()).
 *                    "highest" where more is not better.
 *   recent         - the mean of the last RECENT_S seconds at that concurrency.
 *   now            - the current value as a percentage of that extreme.
 * Figures are kept per concurrency ("@cN" past one), because a batched rate read against the
 * single-stream best reports a batch arriving as a level change. Requests in flight keeps one
 * bucket, since its value is the concurrency, and it samples idle polls too. Idle polls record
 * nothing in a concurrency bucket, and the line keeps the last active concurrency's figures
 * without a now clause. There is no ~99 cap as on the decode step: the extreme comes from the
 * same series as `now`, so 100% is exact.
 *
 * RECENT_S is shorter than the sparkline's HIST_S: a 60 s mean still shows a burst long after the lane
 * went quiet, and 15 s decays within a quarter-minute while still tracking the mean mid-burst.
 * Prefill's window is 5 s, so `recent` follows a prefill of several seconds while it runs instead
 * of diluting it; the engine's prefill rate is itself already windowed. */
const RECENT_S = 15;       /* the `recent` window in seconds, for a figure sampled every poll */
const RECENT_PREFILL_S = 5;   /* prefill's is shorter, so a several-second prefill shows up in it */
const extAt = {}, winAt = {}, extAtS = {};   /* extAtS: the poll at which each extreme was set */
let lastActiveC = 0, lastStart = null;
/* A new engine process is a different discontinuity from a quiet minute: the previous run's
 * requests are gone and its configuration may differ, so nothing from it carries over. Every
 * other figure already refuses to pool across a restart -- accWindow drops its rings, histIngest
 * resets its base -- and the session extremes are no different, since a best from a run that no
 * longer exists is not this lane's best. */
function newEngineRun() {
  resetWindows();
  for (const k in extAt) delete extAt[k];
  for (const k in extAtS) delete extAtS[k];
}
/* Whether the server streams its KV (--kv-resident). Settled at bringup, so it cannot change under
 * a running page: the cells that name a memory tier read it instead of guessing. */
let kvStreamOn = false;
/* `win` false records the extreme only: acceptance pools its window in accPool and reads only
 * `highest` from extAt. */
function noteSample(key, c, value, capS, win) {
  if (value == null || !finite(value)) return;
  const e = extAt[key] || (extAt[key] = {});
  if (!(c in e) || value > e[c]) {
    e[c] = value;
    /* When the extreme was set, so a figure that outlives its window can say how old it is. */
    const t = extAtS[key] || (extAtS[key] = {});
    t[c] = nowS;
  }
  if (win === false) return;
  const w = winAt[key] || (winAt[key] = {});
  const s = w[c] || (w[c] = []);
  const lim = capS || RECENT_S;
  s.push({ t: nowS, v: value });
  while (s.length && s[0].t < nowS - lim) s.shift();
}
function triple(key, c, label, fmt, showNow) {
  /* `recent` is a window mean: a window maximum would equal `best` on any steady lane. */
  const e = extAt[key];
  if (!e || !(c in e)) return "";
  /* A bucket whose extreme is zero still renders (requests in flight before the first request);
   * only the `now` clause needs a nonzero extreme to divide by. */
  /* The extreme is a property of the session and outlives a cleared window (a gap, a restart);
   * `recent` and `now` are properties of the window, so they are dropped until it fills again.
   * Rendering the extreme alone keeps the line's one surviving figure rather than blanking it. */
  const s = (winAt[key] || {})[c];
  const tag = typeof c === "number" && c > 1 ? "@c" + c : "";
  let out = label + " " + fmt(e[c]);
  if (!s || !s.length) {
    /* The extreme outlives the window by design, so the age is what makes it honest: without it a
     * figure from before a gap -- or from before an engine restart, which the page treats as a
     * discontinuity everywhere else -- reads as the lane's current best. */
    const t = (extAtS[key] || {})[c];
    /* Only once it is actually stale: an extreme set on this poll is current, and "0 ms ago"
     * would be noise. A second is the resolution dur() itself reports below. */
    if (typeof t === "number" && nowS - t >= 1) out += " | " + dur(nowS - t) + " ago";
    return out;
  }
  let sum = 0;
  for (const p of s) sum += p.v;
  const recent = sum / s.length;
  const now = s[s.length - 1].v;
  out += " | recent " + fmt(recent);
  /* No `now` clause while nothing runs, as on the decode step: a zero rate is an idle lane, and
   * "now 0% of best" would read as a fault. */
  if (showNow !== false && e[c] > 0) out += " | now " + Math.round(100 * now / e[c]) + "% of " + label + tag;
  return out;
}
/* The acceptance tile's detail line. `highest` is the session's extreme of per-poll ratios;
 * `recent` is a pooled ratio of sums over the ring (accWindow) and `now` the newest poll's. */
function acceptTriple(c, showNow) {
  const e = extAt["accept"];
  const ring = accPool[c];
  if (!e || !(c in e)) return "";
  const tag = typeof c === "number" && c > 1 ? "@c" + c : "";
  let out = "highest " + pct(e[c], 0);
  if (!ring || !ring.length) {
    const t = (extAtS["accept"] || {})[c];
    if (typeof t === "number" && nowS - t >= 1) out += " | " + dur(nowS - t) + " ago";
    return out;
  }
  let R = 0, A = 0;
  for (const d of ring) { R += d.dr; A += d.da; }
  if (R <= 0) return out;
  const last = ring[ring.length - 1];
  out += " | recent " + pct(A / R, 0);
  if (showNow !== false && e[c] > 0 && last.dr > 0)
    out += " | now " + Math.round(100 * (last.da / last.dr) / e[c]) + "% of highest" + tag;
  return out;
}

/* Acceptance over a window. The engine's draft_accept is accepted / drafted over the life of
 * the process, so it barely moves on a long-running server. The per-position counters are
 * cumulative too, and their difference between two polls is the exact conditional acceptance
 * (accepted / reached) over that interval. Each ring entry is one poll that drafted; the ring is
 * kept per concurrency and pooled as sum(accepted) / sum(reached), because polls carry unequal
 * draft counts and a mean of their ratios would weigh a quiet poll like a busy one. */
const accPool = {};          /* per concurrency: ring of {dr, da} for the pooled tile window */
let accPrev = null, accUp = -1;
function accWindow(s) {
  let reach = 0, hit = 0;
  for (const p of (s.draft_positions || [])) { reach += p.reached || 0; hit += p.accepted || 0; }
  /* A server restart: the counters go backwards, or, for a hidden tab that stopped polling while
   * a new process drafted past the old totals, uptime_s does. Either clears every ring, so counts
   * from two processes never pool, and yields no delta for this poll. */
  const up = s.uptime_s;
  const restart = accPrev !== null && (reach < accPrev.r || hit < accPrev.a ||
                                       (typeof up === "number" && up < accUp));
  if (typeof up === "number") accUp = up;
  if (accPrev === null || restart) {
    accPrev = { r: reach, a: hit };
    if (restart) for (const k in accPool) delete accPool[k];
    return null;
  }
  const dr = reach - accPrev.r, da = hit - accPrev.a;
  accPrev = { r: reach, a: hit };
  if (dr <= 0) return null;
  return { ratio: da / dr, dr: dr, da: da };
}


/* ---------------------------------------------------------------- the batch
 *
 * ONE CELL PER SCHEDULER SLOT, coloured by what occupies it. Occupancy is a picture and not a
 * count: four of thirty-two full and four of four full are the same number and not the same
 * situation, and only one of them means the queue is the bottleneck. */
let maxSeqs = 0, slotCells = [], drafterName = DASH;
function renderBatch(list, s) {
  let n = maxSeqs;
  for (const r of list) if (r.slot + 1 > n) n = r.slot + 1;
  if (n <= 0) n = Math.max(1, list.length);
  if (slotCells.length !== n) {
    $("slots").textContent = "";
    slotCells = [];
    for (let i = 0; i < n; i++) slotCells.push($("slots").appendChild(el("i")));
  }
  for (const c of slotCells) { c.className = ""; c.title = ""; }
  let used = 0;
  for (const r of list) {
    if (r.slot < 0 || r.slot >= n) continue;
    slotCells[r.slot].className = r.state;
    slotCells[r.slot].title = "#" + r.id + " " + r.state + "  " + cnt(r.output) + " tokens";
    if (r.state !== "waiting") used++;
  }
  $("batchaside").textContent = used + " / " + n + " slots";

  const q = s.queue_cap ? s.in_flight / s.queue_cap : 0;
  $("qv").textContent = s.in_flight + " / " + s.queue_cap;
  setBar($("qi"), q, true);
}

/* ---------------------------------------------------------------- key/value panels */
function kvPanel(host, keys) {
  const out = {};
  for (const key of keys) {
    const kd = el("div"), vd = el("div");
    kd.appendChild(el("span", "k", key));
    const v = el("span", "v", "–");
    vd.appendChild(v);
    host.appendChild(kd); host.appendChild(vd);
    out[key] = v;
  }
  return out;
}
const totals = kvPanel($("totals"), ["Requests", "Steps"]);
/* "Accepted of drafted" is the engine's lifetime draft_accept, accepted / drafted. The panel's
 * big figure divides by the positions reached instead, so the two are not comparable; each label
 * names its denominator (docs/GUIDE.md, section 6.1). */
const spec = kvPanel($("spec"), ["Drafter", "Depth", "Accepted of drafted"]);

/* ---------------------------------------------------------------- meters */
/* `split` gives the track a second segment for a fill that is two things -- see setSplit. */
function meter(host, cls, title, idx, split) {
  const wrap = el("div", cls), row = el("div", "mrow"), t = el("div", "t");
  if (idx !== undefined) t.appendChild(el("b", null, String(idx)));
  t.appendChild(document.createTextNode(title));
  const v = el("div", "v", "–");
  row.appendChild(t); row.appendChild(v);
  const tr = el("div", "track"), i = el("i");
  tr.appendChild(i);
  let b = null;
  if (split) { b = el("b"); tr.appendChild(b); }
  wrap.appendChild(row); wrap.appendChild(tr);
  host.appendChild(wrap);
  return { wrap, v, i, b };
}
/* ONE MERGED KV FIGURE. The pool is split into groups by state kind -- paged attention, the
 * linear state, the drafter's own window -- and that split is an implementation detail of how
 * the memory is carved, not a question anybody watching a server is asking. Summed.
 *
 * THE SPLIT THAT IS WORTH DRAWING IS THE OTHER ONE. A finished turn's blocks stay held by the
 * prefix cache so the next turn can reuse them, so on a server carrying conversations most of a
 * full pool is cache rather than running requests -- and one bar for the whole of it makes a
 * cache doing its job look identical to a pool that never releases. */
const mKv = meter($("mem"), "meter", "KV cache", undefined, true);
const mKvSub = el("div", "ksub");
mKv.wrap.appendChild(mKvSub);
/* AND HOW FAR IT CAN GROW: every block of the carve, the ones lent to the expert slab taken
   back -- which a request that needs them does, the same step. */
const mKvMax = el("div", "ksub");
mKv.wrap.appendChild(mKvMax);
const mPc = meter($("mem"), "meter", "Prefix hits");
/* `hot` is for a meter whose ceiling is a FAILURE -- KV utilisation near 100% preempts. A device
 * VRAM bar near 100% is the engine doing its job, so it passes false and stays neutral. */
function setBar(i, f, hot) {
  i.style.width = Math.max(0, Math.min(1, f || 0)) * 100 + "%";
  i.className = !hot ? "" : f > 0.92 ? "bad" : f > 0.8 ? "warn" : "";
}
/* THE LIVE SHARE IS DRAWN AT FULL STRENGTH AND THE HELD SHARE BESIDE IT. Both are occupying the
   card; only the second gives way on demand, and the colour is the whole of that distinction.
   The hot classes follow the TOTAL, because what preempts is the pool being full. */
function setSplit(m, live, held, total) {
  const f = total > 0 ? (live + held) / total : 0;
  /* BOTH SEGMENTS TAKE THE STATE OF THE WHOLE FILL, because what preempts is the pool being
     full and neither share alone says whether it is. */
  const hot = f > 0.92 ? "bad" : f > 0.8 ? "warn" : "";
  m.i.style.width = Math.max(0, Math.min(1, total > 0 ? live / total : 0)) * 100 + "%";
  m.b.style.width = Math.max(0, Math.min(1, total > 0 ? held / total : 0)) * 100 + "%";
  m.i.className = hot;
  m.b.className = hot;
}

/* ---------------------------------------------------------------- proportion bars
 *
 * A STACKED BAR PLUS A NAMED ROW PER SEGMENT, used twice: expert residency by tier, and what is
 * on the interconnect. The bar is the proportion at a glance and the rows carry the figures --
 * the two questions "how much of it is where" and "how much is that" are asked together and
 * answered in different units, so neither form alone does the job.
 *
 * Built once per SHAPE and updated in place after that, like everything else here. */
function segs(barId, rowId) {
  const made = [];
  return function (list) {
    if (made.length !== list.length) {
      $(barId).textContent = ""; $(rowId).textContent = "";
      made.length = 0;
      for (let q = 0; q < list.length; q++) {
        const i = $(barId).appendChild(el("i"));
        const row = $(rowId).appendChild(el("div"));
        const sw = row.appendChild(el("b"));
        const nm = row.appendChild(el("span", "nm"));
        const am = row.appendChild(el("span", "am"));
        const sc = row.appendChild(el("span", "am"));
        const pc = row.appendChild(el("span", "pc"));
        made.push({ i, row, sw, nm, am, sc, pc });
      }
    }
    const tot = list.reduce((a, x) => a + (x.w > 0 ? x.w : 0), 0);
    list.forEach((x, j) => {
      const m = made[j], f = tot > 0 ? x.w / tot : 0;
      m.i.style.width = (100 * f).toFixed(3) + "%";
      m.i.style.background = x.ink;
      m.sw.style.background = x.ink;
      m.nm.textContent = x.name;
      m.am.textContent = x.a;
      m.sc.textContent = x.b || "";
      /* A segment at exactly zero is DIMMED AND KEPT, not dropped: "no expert is on the SSD" is
       * a fact about this deployment, and a row that disappears when it reaches zero makes the
       * list jump every time the engine crosses the boundary. */
      m.pc.textContent = tot > 0 ? pct(f) : DASH;
      m.row.className = x.w > 0 ? "" : "zero";
      m.i.title = x.name + "  " + x.a + (x.b ? "  " + x.b : "");
    });
  };
}
const tierSegs = segs("tierbar", "tierrows");
const linkSegs = segs("linkbar", "linkrows");

/* The tier ladder's ink, ONE definition. The plane is drawn from the same array, so a colour in
 * the legend and the same colour in the picture cannot drift apart. Bright is resident and fast;
 * the ladder darkens with distance. `Mapped` is not on that ladder at all -- it is the container's
 * own mapping, read in place -- so it is the one that is not a grey. */
const TIER_INK = ["#f5f5f5", "#9a9a9a", "#5e5e5e", "#2b2b2b", "#3f6fa8"];
const INK_AR = "#60a5fa", INK_UP = "#4ade80", INK_DOWN = "#fbbf24", INK_SSD = "#a78bfa";
const INK_SSD_STAGE = "#f9a8d4", INK_SSD_PROMOTE = "#86efac", INK_SSD_DEPOSIT = "#c4b5fd";
/* The Expert SSD rows take their inks from where their bytes come from or go: staging is a
 * lighter Expert streaming -- it is that traffic's own second leg, the same extents copied
 * host to GPU after the drive read them; deposits wear a light tier violet, the kept copy in
 * host RAM that lands just short of VRAM; promotions are a lighter Promotions (#4ade80), the
 * same keep-forever copy heat's is, one tier further down. */
const INK_STREAM = "#f472b6", INK_PREFETCH = "#2dd4bf", INK_KV = "#fb923c";
/* The plane is painted as raw pixels, so the same inks are wanted as RGB triples. Parsed from
 * TIER_INK rather than written out a second time: two lists of five colours is one list that
 * will drift, and the drift is a legend that names the wrong band of the picture. */
const planeInk = TIER_INK.map((s) => [1, 3, 5].map((i) => parseInt(s.substr(i, 2), 16)));

/* ---------------------------------------------------------------- a two-series line chart
 *
 * The same sparkline rules as the metric strip -- the window's own maximum, not an absolute one --
 * except that this one is SHARED between the two series, because the whole reading is their
 * relation. Promotions and demotions scaled independently would draw a settled engine and a
 * thrashing one identically. The floor IS zero here for the same reason: "no moves" has to look
 * like nothing, and a window-relative floor would draw noise as a full-height wave. */
function lines(svgId, series) {
  const svg = $(svgId), w = 300, h = 56;
  if (!svg.childNodes.length) {
    svg.appendChild(sv("path", { class: "hgrid", d: "M0,55.5 L300,55.5" }));
    for (const s of series) {
      const a = { fill: "none", stroke: s.ink, "stroke-width": "1.25",
                  "stroke-linejoin": "round", "stroke-linecap": "round",
                  "vector-effect": "non-scaling-stroke" };
      /* THE SECOND SERIES IS DASHED, and this is not decoration. Promotions and demotions of a
       * settled engine are nearly EQUAL -- that is what steady state means -- so two solid lines
       * lie exactly on top of each other and the one drawn second erases the other. A reader then
       * sees one amber trace and concludes the engine only evicts. Dashed, the coincidence is
       * visible as the green showing through, which is the reading that was wanted. */
      if (s.dash) a["stroke-dasharray"] = "3 3";
      svg.appendChild(sv("path", a));
    }
  }
  let hi = 0;
  for (const s of series) for (const p of s.pts) if (p.v > hi) hi = p.v;
  /* X IS TIME, as on the sparklines: both series share one window, so a busy stretch is the width
   * of its seconds in both and the two lines stay comparable to each other. */
  let t1 = 0;
  for (const s of series) if (s.pts.length) t1 = Math.max(t1, s.pts[s.pts.length - 1].t);
  series.forEach((s, k) => {
    const p = svg.childNodes[k + 1];
    if (s.pts.length < 2) { p.setAttribute("d", ""); return; }
    let d = "";
    for (const q of s.pts) {
      const x = Math.min(w, Math.max(0, w * (1 - (t1 - q.t) / HIST_S)));
      d += (d ? "L" : "M") + x.toFixed(1) + "," +
           (h - 1 - (h - 4) * (hi > 0 ? q.v / hi : 0)).toFixed(1);
    }
    p.setAttribute("d", d);
  });
  return hi;
}
const moveHist = { up: [], down: [] };
const moveKv = kvPanel($("movekv"), ["Promotions", "Demotions", "Re-admitted", "Refused",
                                     "No candidate", "Starved", "SSD reads", "Borrowed VRAM"]);

/* ---------------------------------------------------------------- request table */
const COLS = [
  { h: "id",     f: (r) => "#" + r.id },
  { h: "state",  f: null },                       /* a tag */
  { h: "slot",   f: (r) => r.slot >= 0 ? String(r.slot) : DASH },
  { h: "prompt", f: (r) => cnt3(r.prompt) + (r.cached ? " (" + cnt3(r.cached) + " hit)" : "") },
  { h: "out",    f: (r) => cnt3(r.output) + (r.max ? " / " + cnt3(r.max) : "") },
  { h: "ctx",    f: (r) => cnt3(r.ctx) },
  { h: "kv",     f: null },                       /* a bar, then tokens and a share of the pool */
  { h: "draft",  f: (r) => r.draft ? String(r.draft) : DASH },
  { h: "ttft",   f: (r) => dur(r.ttft_s) },
  { h: "tok/s",  f: (r) => r.tps > 0 ? tps(r.tps) : DASH },
  { h: "age",    f: (r) => dur(r.age_s) },
];
const C_STATE = 1, C_PROMPT = 3, C_KV = 6;
for (const c of COLS) $("rhead").appendChild(el("th", null, c.h));
let rows = new Map();

/* Two cells hold a node before their text, so every cell writes through an OWNED text node rather
 * than textContent -- which would delete the node on the next poll. The prompt cell carries
 * prefill progress, because "1024" and "1024, half of it still behind a chunk boundary" are
 * different facts and only one of them is a number; the kv cell carries this request's share of
 * the paged pool, which is the thing that decides who gets preempted. */
function requestRow() {
  const tr = el("tr"), cells = [], text = [];
  const bar = (td) => { const m = el("span", "mini"), f = el("i");
                        m.appendChild(f); td.appendChild(m); return f; };
  let promptFill = null, kvFill = null, tag = null;
  for (let i = 0; i < COLS.length; i++) {
    const td = tr.appendChild(el("td"));
    cells.push(td);
    if (i === C_STATE) { tag = td.appendChild(el("span", "mtag")); text.push(null); continue; }
    if (i === C_PROMPT) promptFill = bar(td);
    if (i === C_KV)     kvFill = bar(td);
    text.push(td.appendChild(document.createTextNode("")));
  }
  return { tr, cells, text, tag, promptFill, kvFill };
}

function renderRequests(list, poolTokens) {
  const seen = new Set();
  let prevRow = null;
  for (const r of list) {
    seen.add(r.id);
    let row = rows.get(r.id);
    if (!row) { row = requestRow(); rows.set(r.id, row); }
    /* Keep DOM order equal to list order without rebuilding: insertBefore on a node already in
     * place is a no-op in every engine that matters. */
    const want = prevRow ? prevRow.tr.nextSibling : $("rbody").firstChild;
    if (row.tr !== want) $("rbody").insertBefore(row.tr, want);
    prevRow = row;

    row.tag.textContent = r.state;
    row.tag.className = "tag " + r.state;

    const prefilling = r.state === "prefill";
    row.promptFill.parentNode.style.visibility = prefilling ? "" : "hidden";
    if (prefilling)
      row.promptFill.style.width = (100 * (r.prompt ? r.computed / r.prompt : 1)).toFixed(1) + "%";

    /* The bar is this request's share of the WHOLE paged pool, not of its own context: the reader
     * is asking who is filling the cache, and a bar normalised per request answers a question
     * nobody has. Tokens and the share are both printed, because 8k of 245k and 8k of 8k are the
     * same tokens and not the same situation. */
    const share = poolTokens > 0 ? r.kv_tokens / poolTokens : 0;
    row.kvFill.parentNode.style.visibility = r.kv_tokens > 0 ? "" : "hidden";
    row.kvFill.style.width = (100 * Math.min(1, share)).toFixed(2) + "%";
    row.text[C_KV].nodeValue = r.kv_tokens > 0
        ? cnt3(r.kv_tokens) + "  " + pct(share) : DASH;

    for (let i = 0; i < COLS.length; i++)
      if (COLS[i].f) row.text[i].nodeValue = COLS[i].f(r);
  }
  for (const [id, row] of rows)
    if (!seen.has(id)) { row.tr.remove(); rows.delete(id); }
  $("rempty").hidden = list.length > 0;
}

/* ---------------------------------------------------------------- history
 *
 * A request is in /stats only while it exists: its finished row leaves the payload within about
 * 20 ms (a one-token reply) to half a second. The history keeps every finished request a poll
 * saw, per browser in localStorage, and shows the last HMAX under the live table.
 *
 * A request is recorded when a poll sees it "done", or, usually, from its last sighting once it
 * has left the payload. Requests that began and ended between two polls are counted in a grey
 * row. Rows are keyed by the engine's start and request id, so every tab and reload keys a
 * request alike; a save unions with what is stored, and a clear is a timestamp every tab
 * honours. The history id shown starts at 0 again after a clear. */
const HK  = "radiance.history.v3";         /* rows */
const HKN = HK + ".next";                  /* next history id: 0 after a clear */
const HKC = HK + ".clear";                 /* clear marker: rows older than this stay deleted */
const HMAX = 200;

function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }

let hRows = [];                             /* oldest first, unique by k */
let hNext = 0;                             /* next history id; the clear button writes 0 here */
let hTracked = new Map();                  /* engine id -> { last, fin } */
let hSeen = new Set(), hBase = null, hBaseSeen = 0;
let hDrawn = null;

/* Stored rows, minus the ones a clear deleted. A corrupt read is ignored rather than shown. */
function histStored() {
  try { const a = JSON.parse(lsGet(HK) || "[]"); return Array.isArray(a) ? a : []; }
  catch (e) { return []; }
}
function histMerge(lists) {
  const cleared = +lsGet(HKC) || 0, by = new Map();
  for (const rows of lists) for (const h of rows)
    if (h && h.k != null && (h.at || 0) > cleared) {
      const p = by.get(h.k);
      if (!p || (h.at || 0) > (p.at || 0)) by.set(h.k, h);
    }
  return [...by.values()].sort((a, b) => a.at - b.at).slice(-HMAX);
}
/* The whole table, on load and on every poll: another tab's rows join this one's. */
function histLoad() {
  hRows = histMerge([histStored(), hRows]);
  let n = 0;
  for (const h of hRows) n = Math.max(n, (h.id || 0) + 1);
  hNext = Math.max(n, +lsGet(HKN) || 0);
}
function histSave() {
  hRows = histMerge([hRows]);
  lsSet(HK, JSON.stringify(hRows));
  lsSet(HKN, String(hNext));
}

/* Prefill rate: the prompt tokens computed (not from the prefix cache) over the time to first
 * token; none for a fully cached prompt. */
function histPrefill(r) {
  if (!finite(r.prompt)) return null;
  const fresh = r.prompt - (r.cached || 0);
  if (fresh <= 0) return null;
  /* GUIDE: prefill speed is the prefill's own duration, not the prompt over time to
   * first token -- the latter charges the admission queue to the prefill. Fall back to
   * ttft for a server that does not report prefill_s. */
  const t = finite(r.prefill_s) && r.prefill_s > 0 ? r.prefill_s
          : (finite(r.ttft_s) && r.ttft_s > 0 ? r.ttft_s : null);
  return t ? fresh / t : null;
}
function histFinish(r, run) {
  if (!r || !r.id) return;
  hRows.push({
    k: run + ":" + r.id,
    id: hNext++,                       /* 0 after a clear, then 1, 2, … */
    prompt: finite(r.prompt) ? r.prompt : null,
    out: finite(r.output) ? r.output : null,
    ctx: finite(r.ctx) ? r.ctx : null,
    ttft: finite(r.ttft_s) ? r.ttft_s : null,
    pf: histPrefill(r),
    tps: finite(r.tps) && r.tps > 0 ? r.tps : null,
    at: Date.now(),
  });
  if (hRows.length > HMAX) hRows.shift();
  histSave();
}
/* Returns the requests that finished unseen between two polls, for the grey row. */
function histIngest(s) {
  const total = s.total_requests || 0;
  const run = s.started_at || 0;
  if (hBase === null) { hBase = total; hBaseSeen = hSeen.size; }
  else if (total < hBase) {            /* a new engine run: its ids restart at 1 */
    hBase = total; hBaseSeen = 0; hSeen.clear(); hTracked.clear();
  }
  histLoad();
  const live = new Set();
  for (const r of s.requests || []) {
    if (!r.id) continue;
    live.add(r.id); hSeen.add(r.id);
    let t = hTracked.get(r.id);
    if (!t) { t = {}; hTracked.set(r.id, t); }
    t.last = r;
    if (r.state === "done" && !t.fin) { t.fin = true; histFinish(r, run); }
  }
  /* Gone without a "done" sighting: finalise it from its last sighting, then forget it. */
  for (const [id, t] of [...hTracked])
    if (!live.has(id)) { if (!t.fin) histFinish(t.last, run); hTracked.delete(id); }

  return Math.max(0, total - hBase - (hSeen.size - hBaseSeen));
}

const HCOLS = [
  { h: "id",            f: (h) => "#" + h.id },
  { h: "prompt",        f: (h) => cnt3(h.prompt) },
  { h: "out",           f: (h) => cnt3(h.out) },
  { h: "ctx",           f: (h) => cnt3(h.ctx) },
  { h: "ttft",          f: (h) => dur(h.ttft) },
  { h: "prefill tok/s", f: (h) => cnt3(h.pf) },
  { h: "tok/s",         f: (h) => tps(h.tps) },
  { h: "age",           f: null },     /* rewritten in place every poll */
];
const HTITLE = {
  "id": "this browser's history id; 0 again after a clear",
  "prompt": "prompt tokens",
  "out": "output tokens",
  "ctx": "context tokens, prompt plus output",
  "ttft": "time to first token, measured from arrival, so it includes queue wait",
  "prefill tok/s": "fresh prompt tokens over the time from admission to first token, so queue wait is not charged to it; a mostly-cached prompt scores low",
  "tok/s": "this request's own decode rate since its first token",
  "age": "how long ago this request finished",
};
for (const c of HCOLS) { const th = el("th", null, c.h); if (HTITLE[c.h]) th.title = HTITLE[c.h]; $("hhead").appendChild(th); }

function histRender(missed) {
  const mean = (a) => a.length ? a.reduce((p, c) => p + c, 0) / a.length : null;
  /* Overall, then the last 20 and last 5 captured requests: one long request should not decide
   * the headline, and the recent window is what a change in the workload shows up in first. */
  const group = (vals, fmt, label) => {
    const all = mean(vals), a20 = mean(vals.slice(-20)), a5 = mean(vals.slice(-5));
    /* A window whose rows are a subset of a wider one shown is the same mean twice: show
     * "last 20" only once 20 requests exist and "last 5" only once 5 do, so every figure on
     * the line is a different question. */
    const w = [];
    if (vals.length >= 20) w.push("last 20: " + fmt(a20));
    if (vals.length >= 5) w.push("last 5: " + fmt(a5));
    return label + " mean " + fmt(all) + (w.length ? " | " + w.join(" | ") : "");
  };
  const tpsV = hRows.map((h) => h.tps).filter((v) => v != null);
  const pfV  = hRows.map((h) => h.pf).filter((v) => v != null);
  const tfs  = hRows.map((h) => h.ttft).filter((v) => v != null).sort((a, b) => a - b);
  const parts = [hRows.length + " done"];
  if (tpsV.length) parts.push(group(tpsV, tps, "tok/s"));
  if (pfV.length)  parts.push(group(pfV, cnt3, "prefill"));
  if (tfs.length)  parts.push("ttft median " + dur(tfs[Math.floor(tfs.length / 2)]));
  /* One element per category so it wraps whole, and the dot between them is its own element so
   * its spacing survives. The dot marks a category change; the pipe inside each group marks a
   * different question about the same category. */
  const hsum = $("hsum");
  hsum.textContent = "";
  parts.forEach((p, i) => {
    /* The zero-width space is the line's ONLY break opportunity. The groups are nowrap so they
     * cannot split, and with no break point here a long summary overflows the row and clips
     * whatever follows it -- which is what the padded separators did to this line. */
    if (i) hsum.appendChild(document.createTextNode("\u200b"));
    const g = el("span", "hgrp");
    /* The dot travels WITH the category it introduces: as a sibling it could be left stranded
     * at the end of a line when the category wrapped, which is the orphan this avoids. */
    if (i) g.appendChild(el("span", "hsep", "\u00b7"));
    g.appendChild(document.createTextNode(p));
    hsum.appendChild(g);
  });

  /* Newest first. The rows are rebuilt only when the set changes; a poll only rewrites the ages. */
  const body = $("hbody");
  const shape = hRows.length + ":" + (hRows.length ? hRows[hRows.length - 1].k : "") + ":" + missed;
  if (shape !== hDrawn) {
    hDrawn = shape;
    body.textContent = "";
    const frag = document.createDocumentFragment();
    for (let i = hRows.length - 1; i >= 0; i--) {
      const h = hRows[i], tr = el("tr");
      for (const c of HCOLS) {
        const td = tr.appendChild(el("td", c.f === null ? "dim" : null));
        if (c.f) td.textContent = c.f(h);
        else td.dataset.at = h.at;
      }
      frag.appendChild(tr);
    }
    if (missed > 0) {
      const tr = el("tr", "gap");
      const td = tr.appendChild(el("td", null,
                   missed + " request(s) finished between polls — not captured"));
      td.colSpan = HCOLS.length;
      frag.appendChild(tr);
    }
    body.appendChild(frag);
    $("hempty").hidden = hRows.length > 0 || missed > 0;
  }
  for (const td of body.querySelectorAll("td.dim"))
    td.textContent = dur((Date.now() - td.dataset.at) / 1000);
}

$("hclr").addEventListener("click", () => {
  hRows = []; hTracked.clear(); hSeen.clear(); hBase = null; hNext = 0;
  lsSet(HK, "[]"); lsSet(HKN, "0"); lsSet(HKC, String(Date.now()));
  hDrawn = null; histRender(0);
});
$("reqjson").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(hRows, null, 2)],
                                           { type: "application/json" }));
  const a = el("a");
  a.href = url; a.download = "radiance-history.json";
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
histLoad();
histRender(0);

/* ---------------------------------------------------------------- update */
let prev = null, prevAt = 0, bars = [], devs = [], layerBars = [];
let lastStepMs = null;

/* The single-stream floor and the engine's most recent window, shown as two numbers with no
 * verdict. The engine records the fastest single-sequence decode window it has measured
 * (step_ns_decode_best) and its most recent one (step_ns_decode_recent). Their gap is not engine
 * drift: on a healthy lane the recent window moves 3-11% above a flat floor with whatever else the
 * lane is doing, so a restart priced against it would not pay back. Until the engine's first
 * window completes, the fastest single-stream step this tab has seen stands in. */
let c1hist = [];
const C1CAP_S = 3600;   /* the fallback floor's horizon in seconds */
const bestAt = {};                        /* fastest step seen at each concurrency, this process */

function update(s, dt) {
  nowS = performance.now() / 1000;
  $("ver").textContent = s.version + "  |  up " + dur(s.uptime_s);

  /* THE DECODE STEP TIME, OUT OF THE ENGINE'S OWN CLOCK, and deliberately not out of the
   * browser's. Wall time over step count is wrong in two directions at once, both of them silent:
   * the browser's interval contains the engine's IDLE time, which gets charged to whatever steps
   * surround it, and a prefill chunk costs an order of magnitude more than a decode step, so on a
   * mixed load the answer is the mean of two populations. `step_ns_decode / steps_decode` is
   * neither -- it is what a decode step cost.
   *
   * A window with no decode steps reads 0, the page's idle figure for every live value; a dash
   * is kept for a quantity that has no value at all, and a held value would present a stale
   * reading as current. The engine's floor and recent window below keep their values. */
  const dSteps = prev ? s.steps_decode - prev.steps_decode : 0;
  const stepMs = dSteps > 0 ? (s.step_ns_decode - prev.step_ns_decode) / 1e6 / dSteps : null;
  if (stepMs != null) lastStepMs = stepMs;
  const liveStepMs = stepMs === null ? 0 : stepMs;   /* no steps in the window -> 0, the idle figure */
  const st = stepTime(liveStepMs);
  M.step.u.textContent = st.u;
  setMetric("step",    liveStepMs,    st.n);

  /* Single-stream steps only: the floor is a single-sequence window, and a batched step read
   * against it would report a batch arriving as a slowdown. */
  if (s.running === 1 && stepMs != null) {
    c1hist.push({ t: nowS, v: stepMs });
    while (c1hist.length && c1hist[0].t < nowS - C1CAP_S) c1hist.shift();
  }
  const engBestMs = s.step_ns_decode_best > 0 ? s.step_ns_decode_best / 1e6 : 0;
  const engRecentMs = s.step_ns_decode_recent > 0 ? s.step_ns_decode_recent / 1e6 : 0;
  const floorMs = engBestMs > 0 ? engBestMs
                : (c1hist.length ? Math.min(...c1hist.map(x => x.v)) : 0);
  let det = "";
  if (floorMs > 0) {
    det = "best " + floorMs.toFixed(2) + "ms";
    if (engRecentMs > 0) det += " | recent " + engRecentMs.toFixed(2) + "ms";
  }
  /* The step as a percentage of the fastest at this concurrency (`@cN` above one). `now` is a
   * ~500 ms poll mean and `best` a 32-step window mean, so a short fast stretch can beat the
   * floor; the figure is capped at 99 and shown as ~99, "at the best", rather than claiming a
   * record the window cannot confirm. */
  /* Only for a window with decode steps in it: during a prefill-only stretch the headline reads
   * 0 and lastStepMs is an older poll's, which `now` must not show. */
  if (stepMs != null && s.running > 0) {
    const k = s.running;
    if (!(k in bestAt) || lastStepMs < bestAt[k]) bestAt[k] = lastStepMs;
    const ref = k === 1 && floorMs > 0 ? floorMs : bestAt[k];
    const shown = ref > 0 ? Math.min(99, Math.round(100 * ref / lastStepMs)) : 0;
    if (ref > 0) det += " | now " + (shown === 99 ? "~99" : shown) + "% of best" + (k === 1 ? "" : "@c" + k);
  }
  $("m-stepdet").textContent = det;

  setMetric("decode",  s.decode_tps,  tps(s.decode_tps));
  setMetric("prefill", s.prefill_tps, tps(s.prefill_tps));
  setMetric("flight",  s.in_flight,   String(s.in_flight));
  const busy = s.running > 0;
  if (busy) lastActiveC = s.running;
  /* Sampled every poll, idle included. The engine decays its own rate toward zero on idle, so
   * feeding that decayed value into the last active concurrency's window lets `recent` wind down
   * with the engine. Gating this on busy froze the detail line at the last busy mean, so it read
   * "recent 214" beside a headline of 0. Skipped before any activity (lastActiveC 0) so a fresh
   * idle page shows no figures rather than a column of zeros. */
  if (lastActiveC > 0) {
    noteSample("decode", lastActiveC, s.decode_tps);
    noteSample("prefill", lastActiveC, s.prefill_tps, RECENT_PREFILL_S);
  }
  noteSample("flight", "*", s.in_flight);
  const cNow = busy ? s.running : lastActiveC;
  $("m-decodedet").textContent  = triple("decode", cNow, "best", tps, busy);
  $("m-prefilldet").textContent = triple("prefill", cNow, "best", tps, busy);
  $("m-flightdet").textContent  = triple("flight", "*", "highest", (v) => String(Math.round(v)), s.in_flight > 0);
  /* THE MERGED POOL, not the maximum over groups. The max is the right preemption signal and the
   * wrong headline: the drafter's own window group is sized one block-run per slot and therefore
   * sits at 100% for the whole life of the server, so a headline reading the max reads 100% on a
   * server using 4% of its cache. */
  /* ONE UNIT FOR ONE METER. The figure is bytes -- every group's, which is the only unit groups
   * can be summed in -- and the bar used to be TOKENS, the binding group's. Groups hold different
   * bytes per token, so the two were answering at different rates: the figure reached its total
   * while the bar was half full and then the figure stood still while the bar went on climbing.
   * A streamed group's host mirror, whose pages arrive in their own order, made the gap wide
   * enough to read as a bug, and it was one. Both are bytes now, and the percentage is the bar. */
  const kvFrac = s.kv_bytes_total > 0 ? (s.kv_bytes_used || 0) / s.kv_bytes_total : 0;
  /* HELD, NOT READ. A finished turn's blocks stay held by the prefix cache so the next turn can
   * reuse them, and they occupy the card for exactly as long as they are held -- so they belong
   * in this figure. What leaves it is a block that reached the free list, which is where an
   * eviction, a cancellation and an idle-tier demotion to host memory all put one. */
  const kvb = hbs(s.kv_bytes_used);
  M.kv.u.textContent = kvb.u;
  setMetric("kv", s.kv_bytes_used, kvb.n);
  /* AND WHAT IT LENT, when it lent anything. The cache lends the expert slab the blocks nothing
   * holds, in place, so what it keeps for itself is below what the carve gave it -- and the gap is
   * the expert plane's "Borrowed VRAM". The line says both numbers rather than one, because
   * "of 3.1 GiB" alone would read as a smaller budget. */
  const kvGave = (s.kv_bytes_carved || 0) - (s.kv_bytes_total || 0);
  /* KV STREAMING CHANGES WHAT THESE BYTES ARE -- but only where it is switched on in fact, not
   * where it was asked for: --kv-resident is a preference (zeroed when the architecture refuses
   * it, and a no-op at or above the group's context), while kv_stream_active says a group really
   * keeps a resident window under its context. Without it every block the cache holds is on the
   * card, so "of <total>" is VRAM and the carve is its ceiling. With it the card holds only the
   * ring and every other block's K/V sits in a pinned host mirror: the figure above is the
   * mirror, which is the whole context and therefore larger than the carve -- printing the carve
   * beside it as a "max", or below as "can grow to", put a smaller number next to the larger one,
   * and the pool read full the moment the mirror did. The line names the tier instead, and the
   * ring keeps its own figure. */
  const kvStream = !!(s.link && s.link.kv_stream_active);
  /* THE TIER THE NUMBER IS IN, spelled. A streamed group's held context lives in the host mirror,
   * so the same figure is "in ram" on this card; without streaming every block is on the card and
   * it stays "in vram". The sessions view asks the same question of the same pool and renames its
   * cell from the flag this poll settles. `kvStream` is the ring's actual state, not the switch. */
  kvStreamOn = kvStream;
  $("m-kvcap").textContent = kvStream ? "kv cache in ram" : "kv cache in vram";
  $("m-kvdet").textContent = s.kv_bytes_total > 0
      ? (kvStream
          ? "of " + hb(s.kv_bytes_total) + " in host mirror  |  " + cnt3(s.kv_tokens_used) + " tok"
            + "  ·  " + hb(s.kv_bytes_carved || 0) + " VRAM ring"
          : "of " + hb(s.kv_bytes_total) + " (max " + hb(s.kv_bytes_carved || 0) + ")  |  "
            + cnt3(s.kv_tokens_used) + " tok"
            + (kvGave > 0 ? "  ·  " + hb(kvGave) + " lent to experts" : ""))
      : "";
  const aw = accWindow(s);
  const awR = aw ? aw.ratio : null;
  if (awR != null) noteSample("accept", s.running > 0 ? s.running : lastActiveC, awR, null, false);
  /* The pooled ring (see accWindow), per concurrency. */
  if (aw != null && aw.dr > 0) {
    const c = s.running > 0 ? s.running : lastActiveC;
    const ring = accPool[c] || (accPool[c] = []);
    ring.push({ t: nowS, dr: aw.dr, da: aw.da });
    while (ring.length && ring[0].t < nowS - RECENT_S) ring.shift();
  }
  /* The headline is this poll's conditional acceptance, as every other tile's big number is the
   * current value; the lifetime ratio stays in the Speculation panel, on /stats and on /metrics.
   * A poll that drafted nothing reads 0, the page's idle figure, and the dash means there is no
   * draft head. The sparkline takes a sample every poll, idle ones included, like the other
   * tiles; without a draft head it takes none. */
  if (!(s.draft_positions && s.draft_positions.length)) setMetric("accept", null, DASH);
  else if (awR != null) setMetric("accept", 100 * awR, pctn(awR, 0));
  else setMetric("accept", 0, pctn(0, 0));   /* drafted nothing this poll -> 0, the idle figure */
  /* `now` only when this poll drafted: the ring holds only polls that did, so its newest entry
   * may be older than this poll, whose headline then reads 0. */
  $("m-acceptdet").textContent = acceptTriple(cNow, busy && awR != null);
  M.flight.u.textContent = "/ " + s.queue_cap;

  const list = s.requests || [];
  renderRequests(list, s.kv_tokens_total || 0);
  renderBatch(list, s);
  $("reqaside").textContent = list.length ? s.running + " running | " + s.waiting + " waiting" : "";
  histRender(histIngest(s));

  totals["Requests"].textContent    = cnt(s.total_requests);
  totals["Steps"].textContent       = cnt(s.steps);

  /* ONE MERGED FIGURE. The pool is carved into groups by state kind -- paged attention, the
   * linear state, the drafter's own window -- and which of them a token landed in is an
   * implementation detail of the carve, not a question anybody watching a server is asking. */
  /* THE FILL IS IN TWO PARTS, and the second is the one that answers "why is the pool full when
   * one conversation is running". The share comes from the tokens (which blocks a running request
   * reads versus which only the cache keeps is a fact about blocks) and is then spent in BYTES,
   * the unit the figure and the percentage beside it are in, so the three cannot disagree. */
  const kvCached = Math.min(s.kv_tokens_cached || 0, s.kv_tokens_used || 0);
  const kvLive = Math.max(0, (s.kv_tokens_used || 0) - kvCached);
  const kvCachedShare = (s.kv_tokens_used || 0) > 0 ? kvCached / s.kv_tokens_used : 0;
  mKv.v.textContent = hbOf(s.kv_bytes_used, s.kv_bytes_total, " / ") + "  ·  " + pct(kvFrac);
  /* The two shares are FRACTIONS OF THE FILL, not of the track: spent as widths against a total
   * of 1 they would draw a full bar for a pool holding 29 KiB. */
  setSplit(mKv, (1 - kvCachedShare) * kvFrac, kvCachedShare * kvFrac, 1);
  /* The segments named by what reclaiming them would cost, since a dimmer bar is not
     self-explanatory. The split is by who references the blocks: the first part is context a
     request on the card is reading, the second context nothing reads and the next eviction frees. */
  mKvSub.textContent = s.kv_tokens_used > 0
      ? cnt3(kvLive) + " tok read by running requests  |  " + cnt3(kvCached) + " kept for reuse"
      : "";
  mKvMax.textContent = kvStream
      ? (s.kv_tokens_max > 0
          ? "a miss streams the missing blocks from the mirror into the ring"
          : "")
      : (s.kv_tokens_max > 0
          ? "can grow to " + hb(s.kv_bytes_carved || 0) + "  ·  " + cnt3(s.kv_tokens_max)
            + " tok, with what the experts borrowed taken back"
          : "");
  mPc.v.textContent = pct(s.prefix_hit_rate); setBar(mPc.i, s.prefix_hit_rate, false);

  /* ---- draft. The bar is the CONDITIONAL acceptance at that position; the reach row under the
   * axis is how many verifies got there at all. A 90% bar over eleven samples and a 90% bar over
   * forty thousand are the same picture without the second. */
  const dp = s.draft_positions || [];
  $("pair").hidden = dp.length === 0;
  if (dp.length) {
    if (bars.length !== dp.length) {
      $("cols").textContent = ""; $("xax").textContent = ""; $("nax").textContent = "";
      bars = dp.map((_, j) => {
        const col = el("div"), pv = el("div", "pv"), bar = el("div", "bar");
        col.appendChild(pv); col.appendChild(bar);
        $("cols").appendChild(col);
        $("xax").appendChild(el("span", null, String(j + 1)));
        return { col, pv, bar, n: $("nax").appendChild(el("span")) };
      });
    }
    const maxReach = Math.max(...dp.map((p) => p.reached), 1);
    dp.forEach((p, j) => {
      const b = bars[j], lo = p.reached < maxReach * 0.02;
      b.col.className = lo ? "lo" : "";
      b.bar.style.height = Math.max(2, p.rate * 100) + "%";
      b.bar.title = cnt(p.accepted) + " of " + cnt(p.reached) + " verifies reaching position " + (j + 1);
      b.pv.textContent = lo ? DASH : pct(p.rate, 0);
      b.n.textContent = cnt3(p.reached);
    });
    $("draftaside").textContent = "conditional on reaching j";

    /* TOKENS A STEP is the product speculation is bought for, and it is a ratio of two rates the
     * page already has: a step emits one verified token plus whatever it accepted, so decode
     * throughput divided by DECODE steps a second IS that number.
     *
     * Steps a second off the browser's interval was the wrong denominator twice over -- it
     * counted prefill chunks as steps and idle time as elapsed, so the figure fell whenever the
     * client paused, which is exactly when an agent pauses. Out of the engine's own clock it is
     * decode steps over the seconds the engine spent taking them. */
    const dDec  = prev ? s.steps_decode - prev.steps_decode : 0;
    const dSecs = prev ? (s.step_ns_decode - prev.step_ns_decode) / 1e9 : 0;
    const sps = dDec > 0 && dSecs > 0 ? dDec / dSecs : 0;
    const seqs = Math.max(1, s.running || 1);
    /* PER SEQUENCE. `decode_tps` is pooled over the batch, so dividing only by steps a second
     * gives tokens a step for the whole batch -- which grows with concurrency and says nothing
     * about the drafter. What speculation buys is tokens a step for ONE sequence. */
    const perStep = sps > 0 && s.decode_tps > 0 ? s.decode_tps / sps / seqs : null;
    /* 0 for a window with no decode steps, as on the step tile it derives from. */
    $("tps_n").firstChild.nodeValue = perStep == null ? "0" : sig3(perStep);
    /* The same per-poll conditional acceptance as the accept tile's headline (accWindow), so the
     * two cannot disagree; 0 when this poll drafted nothing. The lifetime ratio is the
     * "Accepted of drafted" bullet below. */
    if (awR != null) $("acc_n").firstChild.nodeValue = pctn(awR, 0);
    else $("acc_n").firstChild.nodeValue = pctn(0, 0);   /* idle: 0, the page's idle figure */
    $("specaside").textContent = "depth " + dp.length;
    spec["Drafter"].textContent       = drafterName;
    spec["Depth"].textContent         = dp.length;
    /* Always shown: 0% is also what a drafter that accepts nothing reads, which a dash would hide. */
    spec["Accepted of drafted"].textContent = pct(s.draft_accept, 0);
  }

  /* ---- devices, in the same panel as the cache meter: they are one question. */
  const cards = s.cards || [];
  if (devs.length !== cards.length) {
    /* Rebuild means REPLACE. The old rows must leave the panel: a card-count change -- a
     * TP flip, a card swap -- otherwise stacks a fresh set on the stale one, and the panel
     * grows by the old count every time until the page is reloaded. Per-row removal rather
     * than the sibling guards' container-clear: #mem is shared with the KV and prefix
     * meters, which are built once and must survive. Same shape as the request rows above. */
    devs.forEach((d) => d.wrap.remove());
    devs = cards.map((c) => meter($("mem"), "dev", c.name, c.index));
  }
  cards.forEach((c, i) => {
    setBar(devs[i].i, c.vram_total ? c.vram_used / c.vram_total : 0, false);
    devs[i].v.textContent = hbOf(c.vram_used, c.vram_total, " / ");
  });

  /* ---- the experts, when the model has any.
   *
   * WHAT THIS PANEL IS FOR. Expert offload is the one placement decision this engine makes at
   * run time rather than at load, and a picture of it is not enough to judge it by. The picture
   * cannot say how many BYTES are resident; it cannot say whether residency is spread evenly
   * across layers, and an uneven fill starves the layers at one end while the total still looks
   * healthy; and it cannot say whether the moves it draws are the engine finding a working set or
   * re-admitting what it evicted a moment ago. Each of those is a number the engine exports. */
  const ex = s.experts;
  $("exppanel").hidden = !ex;
  if (ex) {
    /* RESIDENCY BY TIER, absolute and proportional, in units and in bytes. Units are what the
     * plane draws and bytes are what the VRAM bar spends, and they are not the same ratio: the
     * tiers hold different weight classes. */
    tierSegs((ex.tiers || []).map((t, i) => ({
      name: t.name, ink: TIER_INK[i] || TIER_INK[3], w: t.units,
      a: cnt(t.units), b: hb(t.bytes) })));
    const res = ex.tiers && ex.tiers.length ? ex.tiers[0].units : 0;
    $("expaside").textContent = cnt(ex.n_units) + " units | " + hb(ex.bytes) +
        (ex.heat ? "  |  heat engine on" : "  |  static plan");

    /* THE HIT RATE, WHICH IS NOT THE RESIDENCY. Residency is a property of the PLAN -- how many
     * units sit in VRAM -- and this is a property of the TRAFFIC: how often the router asked for
     * one that was there. A plan holding 80% of the units still misses on most tokens if the hot
     * ones are the 20% outside, and that gap is the only thing that says whether the ranking is
     * working. Differenced over the poll interval so it tracks the CURRENT workload, falling back
     * to the run's totals while nothing is routing -- the same rule the link bar follows, and for
     * the same reason: a rate of zero over zero is not 0%.
     */
    const dr  = prev && prev.experts && dt > 0 ? ex.routed - prev.experts.routed : 0;
    const drr = prev && prev.experts && dt > 0 ? ex.routed_res - prev.experts.routed_res : 0;
    const hf  = dr > 0 ? drr / dr : (ex.routed > 0 ? ex.routed_res / ex.routed : 0);
    /* The misses over the interval are a count over the POLL, which is not a second: divided by
     * it, or the figure labelled a second is the count over however long the poll happened to
     * take. */
    $("hitv").textContent = (dr > 0 || ex.routed > 0)
        ? pct(hf) + "  ·  " + (dr > 0 ? num3((dr - drr) / dt) + " missed/s"
                                      : cnt3(ex.routed - ex.routed_res) + " missed")
        : DASH;
    setBar($("hiti"), hf, false);

    /* PER LAYER. The bar is the layer's resident FRACTION, so layers of different widths -- the
     * draft head's expert count is not the trunk's -- are comparable; the weakest is called out
     * by name, because that is the one that decides the miss rate. */
    const lu = ex.layer_units || [], lr = ex.layer_resident || [];
    if (layerBars.length !== lu.length) {
      $("lbars").textContent = "";
      layerBars = lu.map(() => $("lbars").appendChild(el("i")));
      $("laylab").lastChild.textContent = lu.length ? "L" + (lu.length - 1) : "";
    }
    let worst = -1, sum = 0, wf = 2;
    lu.forEach((n, i) => {
      const f = n > 0 ? lr[i] / n : 0;
      sum += f;
      if (f < wf) { wf = f; worst = i; }
      const b = layerBars[i];
      b.style.height = Math.max(2, 100 * f).toFixed(1) + "%";
      b.className = f >= 0.999 ? "full" : "";
      b.title = "layer " + i + ": " + lr[i] + " of " + n + " resident  |  " + pct(f);
    });
    /* Only when there IS a weakest layer. Every layer fully resident is the answer this panel
     * most wants to be able to give, and colouring an arbitrary one of them amber says the
     * opposite of what is true. */
    if (worst >= 0 && wf < 0.999) layerBars[worst].className = "min";
    $("layv").textContent = !lu.length ? DASH
        : wf >= 0.999 ? "all resident"
        : pct(sum / lu.length) + " mean  |  worst L" + worst + " " + pct(wf);
    /* Residency, not capability: every bucket says where units are, and the buckets add up to
     * the plan. The old "N movable" read as if cold disk experts could not move -- deposits and
     * the stager move them constantly; what they lack is an address heat can relocate without
     * reading the container. Tier names come from the engine; the page only phrases the two
     * common ones. */
    const rest = (ex.tiers || []).slice(1).filter(t => t.units > 0).map(t =>
        cnt(t.units) + " " + (t.name === "pinned" ? "in host pool"
                            : t.name === "ssd" ? "on the file tier" : t.name));
    $("planev").textContent = cnt(res) + " of " + cnt(ex.n_units) + " in VRAM" +
        (rest.length ? "  ·  " + rest.join("  ·  ") : "  |  fully resident");

    /* ---- the plane itself.
     *
     * ONE TEXEL A CELL, STRETCHED TO THE CARD'S WIDTH with nearest-neighbour sampling (.plane).
     * The factor is not an integer and does not need to be, because nothing separates the cells:
     * every device pixel is some cell's ink, so rounding moves the boundary between two cells by at
     * most one pixel and a uniform region stays one colour. What aliases is a fixed-width GAP beside
     * cells whose width is rounded -- a grid of elements with `gap:1px` -- where the ratio of ink to
     * gap walks across the width and draws bands of density over a plane that has none. So the
     * plane is drawn as a bitmap and never as a grid. */
    const cv = $("plane"), W = ex.n_experts, H = ex.n_layers, n = W * H;
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(W, H);
    for (let i = 0; i < n; i++) {
      const c = planeInk[ex.tier[i]] || planeInk[3];
      img.data[i*4] = c[0]; img.data[i*4+1] = c[1]; img.data[i*4+2] = c[2]; img.data[i*4+3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  /* ---- the interconnect, and what shares it.
   *
   * TWO THINGS ARE ON THIS LINK AND THEY ARE NEVER DISCUSSED TOGETHER: the tensor-parallel
   * all-reduce, which is a property of the model's width and the batch and is therefore fixed
   * work, and the expert mover's promotions and demotions, which are a property of a policy and
   * can be tuned away. Sharing an axis is the point -- "the link is busy" is not actionable and
   * "the link is busy and 40% of it is placement churn" is. */
  const lk = s.link;
  $("linkpanel").hidden = !lk;
  if (lk && prev && prev.link && dt > 0) {
    const d = (f) => Math.max(0, (lk[f] - prev.link[f]) / dt);
    /* THE MOVER'S HOST-TO-DEVICE BYTES ARE THREE THINGS, and only one of them is a promotion.
     * The prefill stager's copies of the next layer's pooled experts, and the exact prefetch's,
     * are counted there too; they are traffic a prefill chunk spends ahead of its own GEMMs, not
     * placement churn, so they get their own row and come out of the promotions. A server from
     * before a counter existed reports none, which reads as zero. */
    const pre = (f) => lk[f] || 0;
    const dP  = (f) => prev.link[f] == null ? 0 : Math.max(0, (pre(f) - prev.link[f]) / dt);
    const ar = d("ar_bytes"), dn = d("d2h_bytes"), sd = d("ssd_bytes"), st = d("stream_bytes");
    const sg = d("ssd_staging_bytes");
    const sp = d("ssd_promoted_bytes");
    const wp = d("ssd_deposits");
    /* Null-safe until a server reports it: an older one, or the switch off, reads as zero. */
    const kv = lk.kv_streaming ? dP("kv_stream_bytes") : 0;
    const pf = dP("staged_bytes") + dP("prefetch_bytes");
    /* Promotions is a residual: every host-to-device byte that is not prefill prefetch.
     * Under the file tier the biggest such bytes are the SSD tier's staging leg -- the same
     * extents ssd_bytes read from the container, copied window to staging buffer -- which
     * would otherwise make Promotions read as the streaming it is spent on, not heat's churn.
     * The counter pairs with ssd_bytes (two legs of the same extents). The hot cache's
     * promotions are the third leg: extents copied into a slab slot that
     * stays, which get their own row and come out of the residual too. */
    const up = Math.max(0, d("h2d_bytes") - d("ssd_staging_bytes") - sp - pf);
    const pfAll = pre("staged_bytes") + pre("prefetch_bytes");
    const upAll = Math.max(0, lk.h2d_bytes - pfAll);
    /* AN IDLE LINK STILL HAS A COMPOSITION, and an empty bar is not it. Between two polls of a
     * server nobody is talking to, every one of these rates is exactly zero -- so a bar driven by
     * the rate alone collapses the moment the load stops and takes the answer with it. The
     * proportion then falls back to the CUMULATIVE bytes, which is the same question asked over
     * the whole run, and the panel says out loud which of the two it is showing. The rates stay
     * in the rows either way: they are never wrong, only sometimes zero. */
    const live = ar + up + pf + dn + sd + st + kv > 0;
    const w = (now, all) => live ? now : all;
    linkSegs([
      { name: "All-reduce", ink: INK_AR,   w: w(ar, lk.ar_bytes),
        a: rate(ar), b: hb(lk.ar_bytes) },
      { name: "Promotions", ink: INK_UP,   w: w(up, upAll),
        a: rate(up), b: hb(upAll) },
      { name: "Prefill prefetch", ink: INK_PREFETCH, w: w(pf, pfAll),
        a: rate(pf), b: hb(pfAll) },
      { name: "Demotions",  ink: INK_DOWN, w: w(dn, lk.d2h_bytes),
        a: rate(dn), b: hb(lk.d2h_bytes) },
      /* THE EXPERT SSD TIER, AND ONLY THAT. The counter behind this row is incremented in one
       * place -- the mover's blocking read of an expert unit whose tier is Tier::SSD -- so it is
       * not "what this server reads from disk" and must not be named as though it were. What it
       * leaves out is every mapped read: a Tier::Mapped weight, the PLE n-gram table above all,
       * is read in the container's own mmap by a host op, which is a page-cache reference and a
       * fault when the page is not there. Neither goes past this counter, and neither crosses
       * the link this panel is about. A zero here means no expert was fetched from the SSD tier;
       * it does not mean the process touched no disk. */
      { name: "Expert streaming", ink: INK_STREAM, w: w(st, lk.stream_bytes),
        a: rate(st), b: hb(lk.stream_bytes) },
      /* KV STREAMING, --kv-resident (core/kvstream): each step's resolve copies the blocks a
       * streamed group missed from the pinned host mirror into the VRAM ring. The row rides the
       * switch, not the traffic: a server that took --kv-resident but never streams -- a pipeline
       * that never enters the streaming path -- is exactly what a zero row with the switch on
       * says, and that is a question worth answering at a glance. The bytes are rank 0's resolve
       * counters times the world: the ranks see the same misses through maps brought up
       * identically, so each card's link carries one rank's figure. */
      ...(lk.kv_streaming ? [{ name: "KV streaming", ink: INK_KV,
        w: w(kv, pre("kv_stream_bytes")),
        a: rate(kv), b: hb(pre("kv_stream_bytes")) }] : []),
      /* THE HOT CACHE'S OWN ROW: SSD-tier reads whose destination was a
       * slab slot -- the expert stays resident -- read above the deposits row, whose bytes
       * stop in host RAM. A server from before the hot cache reports none, which reads as
       * zero. Ink: a lighter Promotions, the same keep-forever copy one tier further down. */
      { name: "Expert SSD promotions", ink: INK_SSD_PROMOTE, w: w(sp, pre("ssd_promoted_bytes")),
        a: rate(sp), b: hb(pre("ssd_promoted_bytes")) },
      { name: "Expert SSD deposits", ink: INK_SSD_DEPOSIT, w: w(wp, pre("ssd_deposit_bytes")),
        a: rate(wp), b: hb(pre("ssd_deposit_bytes")) },
      { name: "Expert SSD staging", ink: INK_SSD_STAGE,  w: w(sg, lk.ssd_staging_bytes),
        a: rate(sg), b: hb(lk.ssd_staging_bytes) },
      { name: "Expert SSD tier",  ink: INK_SSD,  w: w(sd, lk.ssd_bytes),
        a: rate(sd), b: hb(lk.ssd_bytes) },
    ]);
    /* THE HEADLINE IS THE PCIe TOTAL, which is the all-reduce plus the moves. The SSD tier is
     * left out of it on purpose: those bytes come off a disk into host memory and only the ones
     * that go on to be promoted cross the link, so adding them would double-count the promotion
     * they turn into. It has its own row.
     *
     * EXPERT STREAMING IS IN IT. A routed expert that was not resident is read by the MoE GEMM
     * where it lies, which for every tier above VRAM means crossing the link on the critical
     * path of the step. It is not a move and not a collective, and it is the traffic a promotion
     * is spent to remove -- so the two belong on one axis.
     *
     * KV STREAMING IS IN IT WHILE THE SWITCH IS ON. Its resolve copies cross the same PCIe the
     * movers use, already summed over the cards; with the switch off the term is zero and the
     * headline reads as it did without the counter. */
    const lr = rates(ar + up + pf + dn + st + kv);
    $("lnk_n").firstChild.nodeValue = lr.n;
    $("lnk_n").querySelector("u").textContent = lr.u;
    /* WHAT IS SUMMED AND WHAT IS NOT. The move and stream rows are every card's added up,
       because each pulls its own shard over its own link; the all-reduce is one rank's, because
       the ranks are symmetric in it and adding them would double it. Hold the headline against
       world x a single link, not against one. */
    /* THE CAVEAT SITS ON THE BAR IT QUALIFIES. Run along the panel's title it was a third line
       of heading, and it describes one element rather than the panel. */
    $("linkaside").textContent = lk.world > 1
        ? lk.world + " cards  |  " + cnt3(lk.ar_calls) + " collectives" : "single rank";
    $("linkhint").textContent = live
        ? "shares below are the rate over the last poll"
        : "idle — shares below are lifetime totals, not rates";
  }
  /* The move half of this panel belongs to the expert mover, so a dense model under tensor
   * parallel gets the all-reduce rows and nothing else rather than three zeroed sections. */
  $("mvcell").hidden = $("movesub").hidden = $("movekvsub").hidden = !ex;
  if (lk && ex) {
    /* MOVES OVER TIME. Counted from the EXPERT panel's own counters rather than from bytes,
     * because a move is the decision and its size is a property of the weight class -- the two
     * answer different questions and only this one is comparable across models. */
    const rup = prev && prev.experts && dt > 0
              ? Math.max(0, (ex.promotions - prev.experts.promotions) / dt) : 0;
    const rdn = prev && prev.experts && dt > 0
              ? Math.max(0, (ex.demotions - prev.experts.demotions) / dt) : 0;
    if (prev && prev.experts && dt > 0) {
      for (const pair of [[moveHist.up, rup], [moveHist.down, rdn]]) {
        pair[0].push({ t: nowS, v: pair[1] });
        while (pair[0].length && pair[0][0].t < nowS - HIST_S) pair[0].shift();
      }
    }
    const peak = lines("movechart", [{ ink: INK_UP, pts: moveHist.up },
                                     { ink: INK_DOWN, pts: moveHist.down, dash: true }]);
    $("mvr_n").firstChild.nodeValue = num3(rup + rdn);
    /* TWO FIGURES, because the column is a fixed width and their digit count is the data; the
       window's peak is the chart's scale and sits on its axis. */
    $("movev").textContent = peak > 0 ? num3(rup) + " up | " + num3(rdn) + " down" : "settled";
    $("movepk").textContent = num3(peak);

    moveKv["Promotions"].textContent  = cnt(ex.promotions);
    moveKv["Demotions"].textContent   = cnt(ex.demotions);
    /* PROGRESS OR THRASH. Re-admitting a unit just demoted spends both directions of the link to
     * arrive back where it started; the share of promotions that are re-admissions is the number
     * that says whether the guards are set right. */
    moveKv["Re-admitted"].textContent = ex.promotions > 0
        ? cnt(ex.readmits) + "  ·  " + pct(ex.readmits / ex.promotions, 0)
        : cnt(ex.readmits);
    moveKv["Refused"].textContent      = ex.dispatches > 0
        ? pct(ex.refused / ex.dispatches, 0) + " of " + cnt3(ex.dispatches) : DASH;
    moveKv["No candidate"].textContent = cnt(ex.no_candidate);
    /* THE MOVER HAVING NOWHERE TO PUT THE BYTES, which is a different failure from the guard
     * declining and reads identically in a move count: both are "nothing moved". */
    moveKv["Starved"].textContent = cnt(ex.starved_slab + ex.starved_pool + ex.no_record);
    moveKv["SSD reads"].textContent = lk ? cnt(lk.ssd_reads) + "  ·  " + hb(lk.ssd_bytes) : DASH;
    /* WHAT THE SLAB TOOK FROM THE CACHE. The cache's own line says what it released; this says
     * how much of it is actually holding experts, and the two are read against each other --
     * released-but-not-taken is VRAM in neither pool and a boundary that is not moving. */
    moveKv["Borrowed VRAM"].textContent = ex.flex_capacity > 0
        ? hbOf(ex.flex_bytes, ex.flex_capacity, " of ") : DASH;
  }
}



/* ---------------------------------------------------------------- the model view
 *
 * THE DECLARED GRAPH AS A FLOW. `--debug-graph` has printed this since the beginning; printing it
 * is not the same as being able to look at it, and a 1569-row table is not either. What a reader
 * wants from a compute graph is its SHAPE -- what runs, in what order, how many times, and which
 * of those launches is standing in for three ops -- and that is a picture. What they want NEXT is
 * everything about one node, which is a panel, and the two want opposite densities. Hence the
 * split: the flow says almost nothing per node and the inspector says everything.
 *
 * Fetched ONCE. The graph is fixed the moment declare finishes, so a view that polled would be
 * polling for an answer that cannot change. */
let graph = null, nodes = [], selected = -1;

/* The band a reader means when they say "which kernel runs this": the one covering the largest
 * slice of the declared range. The others are on the node as a count and in the panel in full. */
function mainBand(o) {
  let best = null, bestSpan = -1;
  let lo = o.range_lo != null ? o.range_lo - 1 : 0;
  for (const b of o.bands) {
    const hi = b.hi != null ? b.hi : lo + 1;
    if (b.device && hi - lo > bestSpan) { bestSpan = hi - lo; best = b; }
    lo = hi;
  }
  return best || o.bands[0] || null;
}

/* THE CHAIN AN OP STANDS FOR, in the conventional vocabulary. An ordinary op stands for itself; a
 * fused one stands for what its kernel declares it replaces. This is the whole basis of the
 * either/or detection below, and it is a declaration the kernel made rather than anything this
 * page inferred. */
function leafChain(gi) {
  const o = graph.ops[gi], b = mainBand(o), p = b && b.device;
  return (p && p.replaces && p.replaces.length >= 2) ? p.replaces : null;
}

/* WHERE THE ALTERNATIONS ARE.
 *
 * `rad_op_resolved`-else-emit-the-chain (spec 2.3) declares EVERY form whenever the fusion is
 * banded or optional, and exactly one of them is issued at any step.
 * The all-reduce fold declares three: the plain `all_reduce` -> `rmsnorm_quant_fp8` pair, the
 * fused kernel over the exact wire, and the fused kernel over the six-bit rotated wire. Drawn as a
 * flat list that reads as four launches in a row, none of which is what runs. That is the
 * "redundant nodes" a reader sees, and it is not redundancy -- it is a branch drawn as a sequence.
 *
 * DETECTED FROM THE DECLARATION AND NOTHING ELSE. A fused kernel publishes the chain it replaces
 * (RadFuseFn); a run of declared ops that spells that chain is the same computation by the
 * kernel's own claim, so the two are alternatives. Both spellings are accepted at each step -- the
 * op's own name, or the chain IT replaces -- because a fusion may name another fusion:
 * `ar_rmsnorm_quant_fp8` replaces `all_reduce -> rmsnorm_quant_fp8`, and the second of those is
 * itself a fold of three. Nothing here knows any op name, any architecture, or that layers exist.
 *
 * A LANE is one run of nodes spelling the chain. Two or more lanes back to back are a branch. */
function laneEnd(list, j, chain) {
  let ci = 0, k = j;
  while (ci < chain.length) {
    if (k >= list.length) return -1;
    const nd = list[k];
    if (nd.kind !== "op") return -1;
    if (graph.ops[nd.gi].op === chain[ci]) { ci++; k++; continue; }  /* the op, as the chain spells it */
    const lc = leafChain(nd.gi);                                     /* or a fusion standing for a run */
    if (lc && ci + lc.length <= chain.length &&
        lc.every((x, i) => x === chain[ci + i])) { ci += lc.length; k++; continue; }
    return -1;
  }
  return k - 1;
}

function lanesFrom(list, i, chain) {
  const lanes = [];
  let j = i;
  for (;;) {
    const k = laneEnd(list, j, chain);
    if (k < 0) break;
    lanes.push(list.slice(j, k + 1));
    j = k + 1;
  }
  return lanes;
}

function foldAlternatives(list) {
  const out = [];
  for (let i = 0; i < list.length; ) {
    /* Every fusion within a short window is a candidate for the chain the branch is over, and the
     * one that explains the most consecutive nodes wins. Taking the first would let
     * `rmsnorm_quant_fp8`'s own three-op chain claim a window whose branch is over the
     * two-op chain of the all-reduce fold two positions later. */
    let best = null;
    for (let j = i; j < Math.min(list.length, i + 8); j++) {
      const nd = list[j];
      if (nd.kind !== "op") break;
      const c = leafChain(nd.gi);
      if (!c) continue;
      const lanes = lanesFrom(list, i, c);
      if (lanes.length < 2) continue;
      const span = lanes.reduce((a, l) => a + l.length, 0);
      if (span <= j - i) continue;                   /* the fusion itself must be inside */
      if (!best || span > best.span) best = { lanes, span };
    }
    if (best) { out.push({ kind: "alt", lanes: best.lanes }); i += best.span; }
    else      { out.push(list[i]); i++; }
  }
  return out;
}

/* WHERE THE REPEATS ARE, found in the sequence itself rather than assumed from layer numbering.
 *
 * A GREEDY SCAN, RUN REPEATEDLY. One pass collapses a run of identical consecutive units into a
 * repeat node; running it again over ITS output collapses repeats of repeats. That matters here
 * and not as a refinement: this model interleaves three linear layers with one full-attention
 * layer, so a single pass finds "linear layer x3" and stops, and the sixteen copies of
 * (linear x3, attention) stay expanded. Two passes find both, and the picture is the nesting.
 *
 * Nothing here knows that layers exist, or that this model is hybrid. The signature of a node is
 * its own shape, so the algorithm sees only "this thing repeats". */
function nodeSig(n) {
  return n.kind === "op"  ? "o" + n.gi
       : n.kind === "alt" ? "a(" + n.lanes.map((l) => l.map(nodeSig).join(",")).join("|") + ")"
                          : "r" + n.times + "(" + n.children.map(nodeSig).join(",") + ")";
}

function collapse(list) {
  const sig = list.map(nodeSig), n = list.length, out = [];
  let i = 0;
  while (i < n) {
    let bp = 0, br = 0;
    const maxP = Math.min(256, (n - i) >> 1);
    for (let p = 1; p <= maxP; p++) {
      if (sig[i] !== sig[i + p]) continue;              /* cheap reject before the O(p) walk */
      let r = 1;
      while (i + (r + 1) * p <= n) {
        let same = true;
        for (let j = 0; j < p; j++)
          if (sig[i + j] !== sig[i + r * p + j]) { same = false; break; }
        if (!same) break;
        r++;
      }
      if (r >= 2 && p * r > bp * br) { bp = p; br = r; }
    }
    if (bp) { out.push({ kind: "rep", times: br, children: list.slice(i, i + bp) });
              i += bp * br; }
    else    { out.push(list[i]); i++; }
  }
  return out;
}

function buildTree(order) {
  let list = foldAlternatives(order.map((gi) => ({ kind: "op", gi })));
  for (let pass = 0; pass < 6; pass++) {
    const next = collapse(list);
    if (next.length === list.length) break;
    list = next;
  }
  return list;
}

function nodeEl(o, idx, times) {
  const n = el("div", "gnode");
  n.dataset.op = idx;
  const t = el("div", "t");
  t.appendChild(el("span", "on", o.op));
  const b = mainBand(o), p = b && b.device;
  const kn = el("span", "kn");
  if (p) {
    kn.appendChild(document.createTextNode(p.kernel + " "));
    const lib = el("i", p.plugin.indexOf("ref") >= 0 ? "ref" : null, p.plugin);
    kn.appendChild(lib);
  } else {
    kn.appendChild(el("i", "ref", "no device kernel"));
  }
  t.appendChild(kn);
  n.appendChild(t);

  const chips = el("div", "chips");
  if (p && p.replaces && p.replaces.length)
    chips.appendChild(el("span", "chip fuse", "fuses " + p.replaces.join(" → ")));
  if (o.bands.length > 1) chips.appendChild(el("span", "chip band", o.bands.length + " bands"));
  if (p && p.variant) chips.appendChild(el("span", "chip", p.variant));
  if (chips.children.length) n.appendChild(chips);
  /* A repeat of ONE op is that op twice, not a block containing it. */
  if (times > 1) n.appendChild(el("div", "mult", "×" + times));

  n.addEventListener("click", () => select(idx));
  return n;
}

function pillEl(gi) {
  const p = el("div", "pill", graph.ops[gi].op);
  p.addEventListener("click", () => select(gi));
  return p;
}

function link(into) {
  return el("div", into ? "link into" : "link");
}

/* Rendering follows the tree: a repeat becomes a plated stack carrying its multiplier, and a
 * repeat of repeats nests inside one. Connectors are drawn BETWEEN siblings only, so the flow
 * reads top to bottom at every level.
 *
 * A repeat renders COLLAPSED, as one card naming what is inside it. Expanded, this model is 1569
 * nodes and no amount of nesting makes that a picture; collapsed it is one screen, and the reader
 * opens the block they came to ask about. */
function repSummary(nd) {
  let count = 0;
  const names = [];
  (function walk(list) {
    for (const c of list) {
      if (c.kind === "op")  { count++; if (names.length < 4) names.push(graph.ops[c.gi].op); }
      else if (c.kind === "alt") { count++; if (names.length < 4)
                                     names.push(graph.ops[c.lanes[0][0].gi].op + "…"); }
      else { for (let i = 0; i < c.times; i++) walk(c.children); }
    }
  })(nd.children);
  return { count, head: names.join(" → ") + (count > names.length ? " → …" : "") };
}

/* WHAT TELLS THE LANES APART, said in the declarations' own words.
 *
 * Two lanes of one branch may be the same op at two geometries -- the all-reduce fold is declared
 * once over the exact wire and once over the six-bit rotated one -- so the label cannot be the op
 * name. It is the geometry KEYS that differ across the lanes and nothing else: every key they
 * agree on is not what the architecture is choosing between. A lane spelling the chain the long
 * way is labelled for what it is. */
function laneLabels(lanes) {
  const geoms = lanes.map((l) => {
    if (l.length !== 1) return null;
    const g = {};
    for (const part of (graph.ops[l[0].gi].geometry || "").split(" ")) {
      const eq = part.indexOf("=");
      if (eq > 0) g[part.slice(0, eq)] = part.slice(eq + 1);
    }
    return g;
  });
  const keys = {};
  for (const g of geoms) if (g) for (const k in g) keys[k] = true;
  const differing = Object.keys(keys).filter((k) => {
    let seen = null;
    for (const g of geoms) {
      const v = g ? (g[k] === undefined ? "—" : g[k]) : null;
      if (v === null) continue;
      if (seen === null) seen = v; else if (seen !== v) return true;
    }
    return false;
  });
  return lanes.map((l, i) => {
    if (l.length !== 1) return "spelled out";
    const g = geoms[i];
    const bits = differing.map((k) => k + " " + (g[k] === undefined ? "—" : g[k]));
    return bits.length ? bits.join(" | ") : "fused";
  });
}

/* One branch, one lane a row. The lanes are alternatives, so they are drawn beside a shared rule
 * rather than in a column of connectors -- a connector between two of these would say the second
 * runs after the first, which is the reading this whole node exists to remove. */
function drawAlt(nd, host) {
  const wrap = el("div", "alt");
  wrap.appendChild(el("span", "cap", "one of these " + nd.lanes.length + " runs"));
  const labels = laneLabels(nd.lanes);

  nd.lanes.forEach((lane, i) => {
    if (i) wrap.appendChild(el("div", "altor", "or"));
    const row = el("div", "altrow");
    const l = el("div", "lane");
    if (lane.length === 1) {
      const n = nodeEl(graph.ops[lane[0].gi], lane[0].gi, 1);
      nodes.push({ el: n, gi: lane[0].gi });
      l.appendChild(n);
    } else {
      const ch = el("div", "chain");
      lane.forEach((c, j) => {
        if (j) ch.appendChild(el("span", "arr", "→"));
        const pe = pillEl(c.gi);
        nodes.push({ el: pe, gi: c.gi });
        ch.appendChild(pe);
      });
      l.appendChild(ch);
    }
    row.appendChild(l);
    row.appendChild(el("div", "when", labels[i]));
    wrap.appendChild(row);
  });
  host.appendChild(wrap);
}

function drawList(list, host) {
  list.forEach((nd, i) => {
    if (i) host.appendChild(link(true));
    if (nd.kind === "op") {
      const n = nodeEl(graph.ops[nd.gi], nd.gi, 1);
      host.appendChild(n);
      nodes.push({ el: n, gi: nd.gi });
      return;
    }
    if (nd.kind === "alt") { drawAlt(nd, host); return; }
    /* a repeat of a single op is a badge on that op, not a plate around it */
    if (nd.children.length === 1 && nd.children[0].kind === "op") {
      const gi = nd.children[0].gi;
      const n = nodeEl(graph.ops[gi], gi, nd.times);
      host.appendChild(n);
      nodes.push({ el: n, gi });
      return;
    }
    const wrap = el("div", "grp rep");
    wrap.appendChild(el("div", "mult", "×" + nd.times));
    const s = repSummary(nd);
    const head = el("div", "rephead");
    const tw = el("span", "tw", "▸");
    head.appendChild(tw);
    head.appendChild(el("span", "lbl", s.count + " ops ×" + nd.times));
    head.appendChild(el("span", "sub", s.head));
    wrap.appendChild(head);

    const inner = el("div", "inner");
    inner.hidden = true;
    drawList(nd.children, inner);
    wrap.appendChild(inner);
    head.addEventListener("click", () => {
      inner.hidden = !inner.hidden;
      tw.textContent = inner.hidden ? "▸" : "▾";
      wrap.classList.toggle("open", !inner.hidden);
    });
    host.appendChild(wrap);
  });
}

function drawFlow() {
  const flow = $("flow");
  flow.textContent = "";
  nodes = [];
  drawList(buildTree(graph.order), flow);
  highlight($("gq").value);
}

/* Filtering DIMS rather than removes: a flow with rows taken out of it is a different flow, and
 * the reader is asking "where is this in the graph", not "show me a list". */
function highlight(q) {
  const needle = q.trim().toLowerCase();
  let hit = 0;
  for (const nd of nodes) {
    const on = !needle || graph.ops[nd.gi]._hay.indexOf(needle) >= 0;
    nd.el.style.opacity = on ? "" : ".25";
    if (!on || !needle) continue;
    hit++;
    /* A match inside a collapsed block is a match pointing at nothing, so the blocks above it
     * open. Walking up from the node is what makes that work at every nesting depth. */
    for (let p = nd.el.parentNode; p && p.id !== "flow"; p = p.parentNode)
      if (p.classList && p.classList.contains("inner") && p.hidden) {
        p.hidden = false;
        p.parentNode.classList.add("open");
        const tw = p.parentNode.querySelector(".tw");
        if (tw) tw.textContent = "▾";
      }
  }
  $("gcount").textContent = needle ? hit + " of " + nodes.length + " nodes match"
                                   : cnt(graph.order.length) + " declared ops  |  " +
                                     graph.ops.length + " distinct";
}

/* ---------------------------------------------------------------- the inspector */
function sec(host, title) {
  const d = el("div", "gsec");
  if (title) d.appendChild(el("div", "h", title));
  host.appendChild(d);
  return d;
}

function kvRow(dl, key, val, note) {
  dl.appendChild(el("dt", null, key));
  const dd = el("dd", null, val);
  if (note) dd.appendChild(el("em", null, note));
  dl.appendChild(dd);
}

/* The declared geometry, with each key's schema role beside it. `M=[1,8192]` is a range and the
 * whole reason there is a bucket table; `block_size` is DERIVED and was supplied by the kernel
 * that won, not by the architecture -- two facts a reader cannot get from the geometry string. */
function drawGeometry(host, o) {
  const need = {};
  if (o.schema) for (const p of o.schema.params) need[p.key] = p.need;
  const d = sec(host, "declared geometry");
  const dl = el("dl", "mkv");
  let any = false;
  for (const part of (o.geometry || "").split(" ")) {
    const eq = part.indexOf("=");
    if (eq < 0) continue;
    any = true;
    const key = part.slice(0, eq);
    const n = need[key] && need[key] !== "required" ? need[key] : null;
    kvRow(dl, key, part.slice(eq + 1), n);
  }
  if (o.ranged)
    kvRow(dl, o.ranged, "[" + cnt(o.range_lo) + ", " + cnt(o.range_hi) + "]", "ranged"), any = true;
  if (!any) kvRow(dl, "—", "no parameters");
  d.appendChild(dl);
}

function drawOperands(host, o) {
  if (!o.schema || !o.schema.operands.length) return;
  const d = sec(host, "operands — positional, in schema order");
  const t = el("div", "mtags");
  o.schema.operands.forEach((op, i) => {
    const cls = op.role === "out" || op.role === "inout" ? "out"
              : op.role === "weight" ? "wgt" : (op.optional ? "opt" : "");
    const tag = el("span", "mtag " + cls);
    tag.appendChild(el("b", null, i + " " + op.name));
    tag.appendChild(document.createTextNode(" " + op.role + (op.optional ? "?" : "")));
    t.appendChild(tag);
  });
  d.appendChild(t);
}

/* ONE KERNEL, as itself: what it is, what it says it computes, what it demanded, what it can
 * describe about itself, and which of its variants this machine is running. */
function drawKernel(host, p) {
  const w = el("div", "win");
  w.appendChild(el("span", "k", p.kernel));
  w.appendChild(el("span", "lib" + (p.plugin.indexOf("ref") >= 0 ? " ref" : ""), p.plugin));
  w.appendChild(el("span", "pr", "prio " + p.priority));
  host.appendChild(w);

  if (p.computes) host.appendChild(el("div", "prose", p.computes));

  const dl = el("dl", "mkv");
  /* THE FROZEN GEOMETRY, which is the query plus whatever the winner supplied for a DERIVED key.
   * `block_size=16` appears here and in no declaration: spec 7.2's circularity is resolved by the
   * kernel answering it, and this is where the answer is visible. */
  if (p.geometry) kvRow(dl, "resolved at", p.geometry);
  if (p.shape)   kvRow(dl, "compiled for", p.shape);
  if (p.dtypes)  kvRow(dl, "dtypes", p.dtypes);
  if (p.variant) kvRow(dl, "variant", p.variant,
                       p.variant_source + " | " + p.n_variants + " offered");
  if (p.scratch) kvRow(dl, "scratch", hb(p.scratch));
  if (dl.children.length) host.appendChild(dl);

  if (p.constraints && p.constraints.length) {
    const s = el("div", "msub");
    s.appendChild(el("div", "sh", "matched on"));
    const t = el("div", "mtags");
    for (const c of p.constraints) t.appendChild(el("span", "mtag", c));
    s.appendChild(t);
    host.appendChild(s);
  }
  if (p.variants && p.variants.length > 1) {
    const s = el("div", "msub");
    s.appendChild(el("div", "sh", "variants"));
    const t = el("div", "mtags");
    p.variants.forEach((v, i) => t.appendChild(
      el("span", "mtag" + (i === p.variant_index ? "" : " opt"), v)));
    s.appendChild(t);
    host.appendChild(s);
  }
  if (p.hooks && p.hooks.length) {
    const s = el("div", "msub");
    s.appendChild(el("div", "sh", "describes itself with"));
    const t = el("div", "mtags");
    for (const h of p.hooks) t.appendChild(el("span", "mtag", h));
    s.appendChild(t);
    host.appendChild(s);
  }
  if (p.fuse_steps && p.fuse_steps.length) {
    const s = el("div", "msub");
    s.appendChild(el("div", "sh", "stands in for, in one launch"));
    const c = el("div", "fchain");
    p.fuse_steps.forEach((st, i) => {
      const r = el("div", "fstep");
      r.appendChild(el("span", "i", String(i + 1)));
      const b = el("div");
      b.appendChild(el("div", "o", st.op));
      if (st.params) b.appendChild(el("div", "g", st.params));
      r.appendChild(b);
      c.appendChild(r);
    });
    s.appendChild(c);
    host.appendChild(s);
  }
}

/* WHO ELSE WANTED THIS BAND. The winner is above; this is the rest of the walk, with the
 * constraint that turned each one away -- which is the answer to "why this kernel" that a list of
 * the winner's own constraints cannot give. */
function drawCandidates(host, cands, winner) {
  const rest = (cands || []).filter((c) => c.kernel !== winner);
  if (!rest.length) return;
  const s = el("div", "msub");
  s.appendChild(el("div", "sh", "also considered"));
  const box = el("div", "cands");
  for (const c of rest) {
    const r = el("div", "cand" + (c.ok ? " ok" : ""));
    r.appendChild(el("span", "m", c.ok ? "✓" : "✗"));
    r.appendChild(el("span", "kk", c.kernel));
    r.appendChild(el("span", "wy", c.ok ? "matched — lost on priority " + c.priority
                                        : c.why));
    box.appendChild(r);
  }
  s.appendChild(box);
  host.appendChild(s);
}

function drawBand(host, b) {
  const box = el("div", "bandbox");
  const bh = el("div", "bh");
  bh.appendChild(el("span", "sp", b.span));
  bh.appendChild(el("span", "dm", b.device ? "device" : "unresolved"));
  box.appendChild(bh);
  const bb = el("div", "bb");
  if (b.device) {
    drawKernel(bb, b.device);
    drawCandidates(bb, b.device_considered, b.device.kernel);
  } else {
    bb.appendChild(el("div", "prose", b.device_miss || "no device kernel matched this band"));
    drawCandidates(bb, b.device_considered, "");
  }
  if (b.host) {
    const s = el("div", "msub");
    s.appendChild(el("div", "sh", "host domain — the placement planner may run it here"));
    const t = el("div", "mtags");
    t.appendChild(el("span", "mtag", b.host.kernel + " | " + b.host.plugin));
    s.appendChild(t);
    bb.appendChild(s);
  }
  box.appendChild(bb);
  host.appendChild(box);
}

function drawWeights(host, o) {
  if (!o.weights.length) return;
  const many = o.count > 1;
  const d = sec(host, "weights — " + o.weights.length +
                      (many ? ", of the first of " + o.count + " declarations" : ""));
  const list = el("div", "wlist");
  let max = 1;
  for (const w of o.weights) max = Math.max(max, w.bytes);
  for (const w of o.weights) {
    const it = el("div", "witem");
    it.appendChild(el("div", "wn", w.name));
    const m = el("div", "wm");
    const add = (kk, vv) => { const s = el("span"); s.appendChild(el("b", null, kk + " "));
                              s.appendChild(document.createTextNode(vv)); m.appendChild(s); };
    add("size", hb(w.bytes) + (w.logical && w.logical !== w.bytes
                               ? " (logical " + hb(w.logical) + ")" : ""));
    add("dtype", w.dtype);
    if (w.shape) add("shape", w.shape);
    add("layout", w.layout);
    add("lives", w.tier + " | " + w.site);
    if (w.layer >= 0) add("layer", String(w.layer));
    if (w.expert != null) add("expert", String(w.expert));
    it.appendChild(m);
    const bar = el("div", "bar");
    bar.appendChild(el("i")).style.width = (100 * w.bytes / max).toFixed(1) + "%";
    it.appendChild(bar);
    list.appendChild(it);
  }
  d.appendChild(list);
}

function drawBuffers(host, o) {
  if (!o.writes.length && !o.frees.length) return;
  const d = sec(host, "arena — buffers that begin and end here");
  const dl = el("dl", "mkv");
  for (const b of o.writes) kvRow(dl, "writes", b.name + "  " + hb(b.bytes), "live " + b.live);
  for (const b of o.frees)  kvRow(dl, "last use", b.name + "  " + hb(b.bytes),
                                  "storage is reused after this");
  d.appendChild(dl);
}

function select(gi) {
  selected = gi;
  for (const nd of nodes) {
    const on = nd.gi === gi;
    nd.el.classList.toggle("sel", on);
  }
  const o = graph.ops[gi], host = $("ginfo");
  const b0 = mainBand(o), p0 = b0 && b0.device;
  host.textContent = "";
  $("ginfoaside").textContent = (p0 && p0.family) || (o.count > 1 ? "" : "one launch");

  /* ---- the header: what it is, and where in the flow it is */
  const head = sec(host, null);
  const t = el("div", "gtitle");
  t.appendChild(el("span", "nm", o.op));
  if (o.count > 1) t.appendChild(el("span", "ct", "×" + cnt(o.count)));
  head.appendChild(t);

  const where = [];
  where.push(o.count > 1 ? "positions " + cnt(o.first_at + 1) + "–" + cnt(o.last_at + 1) +
                           " of " + cnt(graph.order.length)
                         : "position " + cnt(o.first_at + 1) + " of " + cnt(graph.order.length));
  where.push(o.bands.length === 1 ? "one band" : o.bands.length + " bands");
  head.appendChild(el("div", "gwhere", where.join("  ·  ")));
  if (o.schema && o.schema.doc) head.appendChild(el("div", "gdoc", o.schema.doc));

  drawGeometry(host, o);
  drawOperands(host, o);

  const bandsSec = sec(host, o.bands.length === 1 ? "selection"
                                                  : "selection — one row per band");
  for (const b of o.bands) drawBand(bandsSec, b);

  drawWeights(host, o);
  drawBuffers(host, o);
  host.scrollTop = 0;
}

/* ---------------------------------------------------------------- the summary strip */
function drawSummary() {
  const c = graph.counts, m = graph.model;
  let fusedLaunches = 0, fusedOps = 0, distinctFused = 0, resolved = 0, missing = 0;
  const libs = {};
  for (const o of graph.ops) {
    let f = null;
    for (const b of o.bands) {
      if (b.device) { resolved += o.count; libs[b.device.plugin] = (libs[b.device.plugin] || 0) + o.count; }
      else missing += o.count;
      if (b.device && b.device.replaces && b.device.replaces.length && !f) f = b.device.replaces;
    }
    if (f) { distinctFused++; fusedLaunches += o.count; fusedOps += f.length * o.count; }
  }
  const cell = (n, unit, cap) => {
    const d = el("div", "cell");
    const v = el("div", "n", n);
    if (unit) v.appendChild(el("u", null, unit));
    d.appendChild(v);
    d.appendChild(el("span", "cap", cap));
    return d;
  };
  const s = $("gstrip");
  s.textContent = "";
  s.appendChild(cell(cnt(c.ops), "", "declared ops"));
  s.appendChild(cell(String(graph.ops.length), "", "distinct shapes"));
  s.appendChild(cell(cnt(c.weights), "", "weights"));
  const wb = hbs(c.weight_bytes), ab = hbs(c.arena_bytes);
  s.appendChild(cell(wb.n, wb.u, "on this rank"));
  s.appendChild(cell(cnt(fusedLaunches), "", "fused launches"));
  s.appendChild(cell(cnt(fusedOps - fusedLaunches), "", "launches removed"));
  s.appendChild(cell(ab.n, ab.u, "activation arena"));
  s.appendChild(cell(String(m.n_layers), "", "layers | rank " + m.rank + "/" + m.world));

  const gl = $("glibs");
  gl.textContent = "";
  for (const pl of graph.plugins) {
    /* KERNEL libraries only, which is what the panel says. An architecture plugin exports no
       kernels and no schemas -- it declares a model's shape and leaves every op to be resolved
       against a kernel library -- so listing qwen35_fp8 beside libr4d here said the two were the
       same kind of thing, and gave each of the five arch plugins a "0 kernels" row. What they
       had to say about the model they built is in the meta section below. */
    if (!pl.kernels && !pl.schemas) continue;
    const d = el("div", "lib");
    d.appendChild(el("span", "o", "#" + pl.order));
    d.appendChild(el("span", "n", pl.name));
    const used = libs[pl.name] || 0;
    d.appendChild(el("span", "c", pl.kernels + " kernels" +
                                 (used ? "  ·  " + cnt(used) + " launches" : "  |  unused")));
    gl.appendChild(d);
  }
  $("gsum").textContent = missing ? cnt(missing) + " op(s) with no device kernel"
                                  : cnt(resolved) + " launches resolved, none missing";
  drawMeta();
}

/* WHAT THE PLUGIN SAID, AND WHAT DID NOT RESOLVE. Both have been in the graph since it had a
 * JSON form and neither had anywhere to be read: the notes are the architecture plugin explaining
 * the model it just built, the KV groups are what the resolved attention kernels asked for, and a
 * miss is a hole a request can fall into. */
function drawMeta() {
  const host = $("gmeta");
  host.textContent = "";
  host.className = "gmeta";
  const dm = (graph.misses || []).filter((m) => m.domain !== "host").length;
  $("gmetasum").textContent =
    (graph.notes ? graph.notes.length : 0) + " notes from the architecture plugin  |  " +
    (graph.kv_groups ? graph.kv_groups.length : 0) + " kv groups  |  " +
    (graph.drafter ? graph.drafter.name + " drafter  |  " : "") +
    (dm ? dm + " unresolved device band(s)" : "no unresolved device band");
  const block = (title, cls) => {
    const se = el("section", cls);
    se.appendChild(el("div", "h", title));
    const ul = el("ul");
    se.appendChild(ul);
    host.appendChild(se);
    return ul;
  };
  if (graph.notes && graph.notes.length) {
    const ul = block("from the architecture plugin");
    for (const n of graph.notes) ul.appendChild(el("li", null, n));
  }
  if (graph.kv_groups && graph.kv_groups.length) {
    const ul = block("kv groups — sized by the resolved attention kernels");
    for (const g of graph.kv_groups)
      ul.appendChild(el("li", null,
        g.name + " | " + g.kind + " | " + g.layers + " layer" + (g.layers === 1 ? "" : "s") +
        (g.block_size ? " | block " + g.block_size + " tokens, " + hb(g.bytes_per_block)
                      : " | " + hb(g.bytes_per_state) + " a sequence")));
  }
  if (graph.drafter) {
    const ul = block("draft head");
    ul.appendChild(el("li", null, graph.drafter.name + " | " + graph.drafter.kind + " | " +
                                  graph.drafter.depth + " token(s) a step" +
                                  (graph.drafter.window ? " | window " + graph.drafter.window
                                                        : "")));
  }
  /* ABSENCE IS ASYMMETRIC (spec 2.1), and so is this list. No DEVICE kernel for a band is a hole
   * a request can fall into and every one is named; no HOST kernel removes one placement option
   * for that op and nothing else, which is the ordinary case for a GPU kernel library -- there are
   * five hundred of them on this model and printing each would bury the ones that matter. */
  const dev = (graph.misses || []).filter((m) => m.domain !== "host");
  const hst = (graph.misses || []).filter((m) => m.domain === "host");
  if (dev.length) {
    const ul = block("unresolved — a device miss is a hole a request can fall into", "bad");
    for (const m of dev) ul.appendChild(el("li", null, m.op + "  " + m.span + "  " + m.why));
  }
  if (hst.length) {
    const names = [];
    for (const m of hst) if (names.indexOf(m.op) < 0 && names.length < 4) names.push(m.op);
    const ul = block("host domain — where the placement planner may not go");
    ul.appendChild(el("li", null,
      cnt(hst.length) + " band(s) over " +
      cnt(new Set(hst.map((m) => m.op)).size) + " op(s) have no host kernel (" +
      names.slice(0, 3).join(", ") + (names.length > 3 ? ", …" : "") +
      ") — host-site execution is unavailable for them, and every weight TIER is untouched"));
  }
}

async function loadGraph() {
  if (graph) return;
  try {
    graph = await (await api("graph")).json();
  } catch (e) {
    $("gsum").textContent = "the graph is not available from this server";
    return;
  }
  /* One flat haystack per op group, built once, so filtering is a substring test rather than a
   * walk of every band on every keystroke. */
  for (const o of graph.ops) {
    let s = o.op + " " + o.geometry;
    for (const b of o.bands) {
      const p = b.device;
      if (p) s += " " + p.kernel + " " + p.plugin + " " + (p.variant || "") + " " +
                  (p.family || "") + " " + (p.constraints || []).join(" ") + " " +
                  (p.replaces || []).join(" ");
    }
    for (const w of o.weights) s += " " + w.name + " " + w.layout;
    o._hay = s.toLowerCase();
  }
  drawSummary();
  drawFlow();
  select(graph.order[0]);
}

$("gq").addEventListener("input", (e) => highlight(e.target.value));

/* ---------------------------------------------------------------- tabs */
function show(which) {
  const model = which === "model", sess = which === "sess", adv = which === "adv";
  /* The Advanced view holds no live figures, so the dashboard's poll pauses while it is open, as
   * it does for the Model view. Returning to the dashboard after more than a window there clears
   * the sampled windows, which is what poll() already does for a backgrounded tab. */
  $("view-dash").hidden  = model || sess || adv;
  $("view-sess").hidden  = !sess;
  $("view-model").hidden = !model;
  $("view-adv").hidden   = !adv;
  $("tab-dash").className  = (model || sess || adv) ? "" : "on";
  $("tab-sess").className  = sess ? "on" : "";
  $("tab-model").className = model ? "on" : "";
  $("tab-adv").className   = adv ? "on" : "";
  if (model) { loadGraph(); loadControlVector(); }
  if (sess) loadSessions();
  window.scrollTo(0, 0);
  const h = model ? "#model" : sess ? "#sessions" : adv ? "#advanced" : "#";
  if (location.hash !== h) history.replaceState(null, "", h);
}
$("tab-dash").addEventListener("click", () => show("dash"));
$("tab-sess").addEventListener("click", () => show("sess"));
$("tab-model").addEventListener("click", () => show("model"));
$("tab-adv").addEventListener("click", () => show("adv"));

/* WHAT THE ENGINE IS STILL HOLDING BETWEEN TURNS. Polled on its own schedule and only while the
   view is open: a stored session changes on the scale of its idle thresholds -- tens of seconds at
   the fastest -- so the per-second dashboard poll would carry the same bytes a hundred times for
   each change in them. */

/* WHAT A RETURN REUSES. On a hybrid model the scheduler clamps a hit to the last linear checkpoint
 * the chain reaches, so the reuse is the deepest snapshot's position, and a conversation with no
 * snapshot reuses nothing however many of its blocks are kept. On a model with no recurrent state
 * the blocks are the whole answer and every stored token is reused. */
const reuseOf = (r, hybrid) => hybrid ? Math.min(r.resume_tokens || 0, r.tokens) : r.tokens;
const TIERS = [["dev", "on_device", "vram"], ["host", "on_host", "host"]];

function sessCols(tr, manyModels) {
  tr.textContent = "";
  tr.appendChild(el("th", null, "conversation"));
  if (manyModels) tr.appendChild(el("th", "l", "model"));
  tr.appendChild(el("th", null, "context"));
  tr.appendChild(el("th", null, "last active"));
  tr.appendChild(el("th", "l ret", "on return"));
  tr.appendChild(el("th", "l", "where it is"));
}

function sessRow(r, hybrid, manyModels) {
  const tr = el("tr");
  const id = tr.appendChild(el("td"));
  id.appendChild(el("span", "sid", "#" + r.id));
  /* The client's own name for the conversation (`session_id`), when it sent one. */
  if (r.label) id.appendChild(el("span", "slabel", r.label));
  if (r.moving) id.appendChild(el("span", "mv", "moving"));
  if (manyModels) tr.appendChild(el("td", "l", r.model || DASH));
  tr.appendChild(el("td", null, cnt3(r.tokens) + " tok"));
  tr.appendChild(el("td", null, dur(r.idle_s) + " ago"));
  /* The next turn's own new tokens are prefilled either way; this is what it pays for the part
     that was stored. */
  const reuse = reuseOf(r, hybrid), ret = tr.appendChild(el("td", "l ret"));
  if (reuse > 0) {
    ret.appendChild(el("b", null, "reuses " + cnt3(reuse)));
    ret.appendChild(document.createTextNode("  |  prefills " + cnt3(r.tokens - reuse)));
  } else {
    ret.textContent = "prefills all " + cnt3(r.tokens);
  }
  /* Shares of the conversation's own context, each named with its share: a shared system prompt
     keeps a sliver of an otherwise demoted conversation on the card, and three bare names would
     read as three equal parts. The per-tier counts are blocks, and a block is the same number of
     tokens in every tier, so the proportion is the proportion of its tokens. */
  const wh = tr.appendChild(el("td", "l")), bar = wh.appendChild(el("span", "tbar"));
  const held = r.on_device + r.on_host, parts = [];
  for (const [k, f, name] of TIERS) {
    if (!(r[f] > 0)) continue;
    bar.appendChild(el("i", "t-" + k)).style.width = (100 * r[f] / held).toFixed(1) + "%";
    if (parts.length) wh.appendChild(document.createTextNode("  ·  "));
    wh.appendChild(el("span", "tn-" + k, name));
    wh.appendChild(document.createTextNode(" " + pct(r[f] / held, 0)));
    parts.push(name);
  }
  return tr;
}

let sessShape = null;
async function loadSessions() {
  let s;
  try {
    s = await (await api("sessions")).json();
  } catch (e) {
    $("s-liveempty").textContent = "the server did not answer";
    $("s-liveempty").hidden = false;
    return;
  }
  const rows = (s.sessions || []).slice().sort((x, y) => x.idle_s - y.idle_s);
  const mv = s.snapshot_store || {}, ln = s.linear || {};
  const pool = s.pool || {}, occ = s.occupancy || {};
  /* The model has recurrent state if its snapshots have a size. */
  const hybrid = mv.bytes_each > 0;
  const live = rows.filter((r) => reuseOf(r, hybrid) > 0);
  const cold = rows.filter((r) => !(reuseOf(r, hybrid) > 0));

  /* THE RATE THAT DECIDES THE COST. On a hybrid model reuse stops at the last linear checkpoint,
     so the linear rate is what the cache saved; the attention rate is higher and counts blocks the
     scheduler could not use. */
  $("s-hit").firstChild.nodeValue = pctn((hybrid ? ln.hit_rate : s.prefix_hit_rate) || 0);
  $("s-hitdet").textContent = hybrid && ln.cached_tokens > 0
      ? cnt3(ln.cached_tokens) + " tok reused" : "";
  $("s-live").firstChild.nodeValue = cnt(live.length);
  $("s-liveof").textContent = "of " + cnt(rows.length);
  $("s-livedet").textContent = cold.length ? cnt(cold.length) + " start over" : "";
  /* VRAM IN TOKENS, HOST MEMORY IN BYTES. The pool's token count is the context the card can
     serve without a copy; the host tier is sized by --prefix-cache-mib and read against it. A
     tier that is off has no cell. */
  /* A streamed server holds that context in its host mirror, not on the card, so the cell says
   * where the tokens are; the flag comes from the stats poll (kvStreamOn). */
  $("s-vramcap").textContent = kvStreamOn ? "context in ram" : "context in vram";
  $("s-vram").firstChild.nodeValue = cnt3(pool.tokens_used || 0);
  $("s-vramdet").textContent = pool.tokens_total ? "of " + cnt3(pool.tokens_total) + " tok" : "";
  for (const [id, used, cap] of [["s-host", occ.host_bytes, occ.host_cap_bytes]]) {
    $(id + "cell").hidden = !(cap > 0);
    const x = hbs(used || 0);
    $(id).firstChild.nodeValue = x.n;
    $(id).querySelector("u").textContent = x.u;
    $(id + "det").textContent = cap > 0 ? "of " + hb(cap) : "";
  }

  /* A column every row answers the same way is not a column: one model serving means its name
     belongs nowhere in the table. */
  const manyModels = new Set(rows.map((r) => r.model || "")).size > 1;
  if (sessShape !== manyModels) {
    sessShape = manyModels;
    sessCols($("s-livecols"), manyModels);
    sessCols($("s-coldcols"), manyModels);
  }
  const fill = (tb, list) => {
    tb.textContent = "";
    for (const r of list) tb.appendChild(sessRow(r, hybrid, manyModels));
  };
  fill($("s-liverows"), live);
  $("s-livehead").textContent = "Will resume  " + cnt(live.length);
  $("s-liveempty").hidden = live.length > 0;
  $("s-liveempty").textContent = rows.length === 0
      ? "Nothing stored yet. A conversation appears here once one of its turns finishes."
      : "No stored conversation can resume: each of them would prefill in full.";
  /* The folded group is a count until it is opened, and its rows are only built while it is. */
  $("s-coldgrp").hidden = cold.length === 0;
  $("s-coldhead").textContent = "Will start over  " + cnt(cold.length);
  if ($("s-coldgrp").open) fill($("s-coldrows"), cold);
}
$("s-coldgrp").addEventListener("toggle", () => { if ($("s-coldgrp").open) loadSessions(); });
setInterval(() => { if (!document.hidden && !$("view-sess").hidden) loadSessions(); }, 5000);

/* WHAT THIS PROCESS IS, asked once. Configuration does not change while the server runs, so
 * polling it would be polling for an answer that cannot differ. What it settles is the page's
 * identity: the title, the architecture and whether there is a drafter to name. The rest of what
 * the process was started with stays on /server_info for anything reading the server rather than
 * watching it. */
async function identify() {
  try {
    const s = await (await api("server_info")).json();
    const e = s.engine || {};
    maxSeqs = s.server.max_num_seqs || 0;
    /* THE CONTAINER'S FILE NAME, not the model name inside it. Two deployments of the same
     * architecture at different quantisations carry the same `model_name` and are different
     * servers, and the file is what an operator typed on the command line. */
    const file = (e.container || "").split("/").pop() || s.server.model;
    $("title").textContent = "radiance / " + file;
    $("arch").textContent = e.arch || "";
    document.title = file + " · radiance";
    const sp = e.speculation && e.speculation.enabled;
    drafterName = sp ? e.speculation.drafter : "off";
  } catch (err) { /* the header simply keeps its placeholder */ }
}

/* The control vector, read when the Model view opens rather than polled: only this page or an
 * admin call changes it. 404 means the deployment has none and hides the panel; any other failure
 * keeps the panel and says why, so a page without the API key does not look like a build without
 * the feature. */
async function loadControlVector() {
  try {
    const r = await api("admin/control-vector");
    if (r.status === 404) { $("cvpanel").hidden = true; return; }
    if (!r.ok) { const err = new Error(String(r.status)); err.status = r.status; throw err; }
    const p = await r.json();
    $("cvpanel").hidden = false;
    $("cvbox").disabled = false;
    $("cvbox").checked = !!p.default;
    /* The family the loader validated the file against, then its range and scale. */
    $("cvsum").textContent = (p.arch ? p.arch + " | " : "") + "layers " + p.first +
                               "\u2013" + p.last + " | " + p.mode + " | scale " + p.scale;
    /* A running request keeps the choice it was admitted with, and a refusal already in a
     * session's context is in the prompt the next turn reads. */
    $("cvnote").textContent = "New requests only. A session with a refusal already in its " +
                               "context may repeat it.";
  } catch (e) {
    $("cvpanel").hidden = false;
    $("cvbox").disabled = true;
    $("cvbox").checked = false;
    $("cvsum").textContent = "";
    $("cvnote").textContent = e.status === 401
      ? "Sign in with the API key (the field in the header) to read or change this: the vector " +
        "is present on this deployment, but its state needs the key."
      : "The server did not report the control vector's state (" + (e.status || e.message) + ").";
  }
}
$("cvbox").addEventListener("change", async (e) => {
  const want = e.target.checked;
  e.target.disabled = true;
  try {
    const r = await fetch("admin/control-vector", {
      method: "POST", cache: "no-store",
      headers: Object.assign({ "Content-Type": "application/json" },
                             apiKey ? { Authorization: "Bearer " + apiKey } : {}),
      body: JSON.stringify({ enabled: want }),
    });
    if (!r.ok) {
      let msg = String(r.status);
      try { const j = await r.json(); if (j && j.error && j.error.message) msg = j.error.message; }
      catch (_) { /* the status alone */ }
      throw new Error(msg);
    }
    const p = await r.json();
    $("cvbox").checked = !!p.default;
  } catch (err) {
    e.target.checked = !want;   // the server kept the old value
    $("cvnote").textContent = "Not changed: " + err.message;
  } finally {
    e.target.disabled = false;
  }
});

/* THE POLL CADENCE, SET BY THIS BROWSER. Held in localStorage, not on the server: the cadence is
 * a property of who is reading, not of what is serving, and browser storage outlives an image swap
 * or an engine restart that a file beside the engine would not. Two intervals, both editable: the
 * page fetches at the idle interval while nothing runs and the busy interval while a request is in
 * flight. Set them equal for one fixed rate. One key holds both, so a change is one write. */
const POLL_K = "radiance.poll";
const POLL_MIN = 100, POLL_MAX = 5000;
const clampMs = (v, d) => Math.min(POLL_MAX, Math.max(POLL_MIN, Math.round(+v || d)));
let pollIdleMs = 500, pollBusyMs = 300;
try {
  const raw = lsGet(POLL_K);
  if (raw) {
    const p = JSON.parse(raw) || {};
    if (typeof p.idle === "number") pollIdleMs = clampMs(p.idle, 500);
    if (typeof p.busy === "number") pollBusyMs = clampMs(p.busy, pollIdleMs);
    else if (typeof p.ms === "number") { pollIdleMs = clampMs(p.ms, 500); pollBusyMs = pollIdleMs; }
  }
} catch (e) { /* the defaults */ }
const pollCadence = (inFlight) => inFlight > 0 ? pollBusyMs : pollIdleMs;
function pollSummary() {
  $("advsum").textContent = pollIdleMs === pollBusyMs ? pollIdleMs + " ms"
    : pollIdleMs + " ms idle | " + pollBusyMs + " ms busy";
}
function pollSave() {
  lsSet(POLL_K, JSON.stringify({ idle: pollIdleMs, busy: pollBusyMs }));
  pollSummary();
}
$("pollidle").value = pollIdleMs;
$("pollbusy").value = pollBusyMs;
pollSummary();
$("pollidle").addEventListener("change", (e) => {
  pollIdleMs = clampMs(e.target.value, 500);
  e.target.value = pollIdleMs;   /* show what was stored, not what was typed */
  pollSave();
  schedulePoll(pollIdleMs);
});
$("pollbusy").addEventListener("change", (e) => {
  pollBusyMs = clampMs(e.target.value, 300);
  e.target.value = pollBusyMs;
  pollSave();
  schedulePoll(pollIdleMs);
});
let pollTimer = null;
function schedulePoll(ms) { clearTimeout(pollTimer); pollTimer = setTimeout(poll, ms); }
/* Past one window's length of not polling, the sampled windows hold only pre-gap samples and the
 * first poll back would charge the whole span to a single sample. There is nothing to keep, so
 * the windows and the poll's baseline clear and the page refills from the return, rather than
 * presenting the pre-gap lane as current. The session's extremes (extAt) and the single-stream
 * floor (c1hist) survive: a fastest-ever and a floor are properties of the session, not of the
 * last minute. Checked in poll(), not on the visibility event, so a view switch, a backgrounded
 * tab and a laptop sleep all take the same path. */
const GAP_S = HIST_S;   /* the same span the sparklines draw: past it there is nothing to keep */
let lastPollAt = 0;
function resetWindows() {
  for (const k in hist) hist[k].length = 0;
  for (const k in winAt) delete winAt[k];
  for (const k in accPool) delete accPool[k];
  moveHist.up.length = 0; moveHist.down.length = 0;
  prev = null; prevAt = 0; accPrev = null;
}
async function poll() {
  /* A background tab watches nothing, and neither does the Model view -- which is static. */
  if (document.hidden || $("view-dash").hidden) { schedulePoll(pollIdleMs); return; }
  const wall = Date.now();
  if (lastPollAt && (wall - lastPollAt) / 1000 > GAP_S) resetWindows();
  lastPollAt = wall;
  try {
    const r = await api("stats");
    if (!r.ok) throw new Error(r.status);
    const s = await r.json();
    /* A new engine process is a discontinuity, not a new measurement, and started_at names it.
     * It is a whole-second clock, so two processes could share one and the clear would be
     * missed. For that, the replacement would have to start within the same second as the one
     * the page last read -- and the page can only read it once /stats answers, startup_s after
     * the old process began (69 s here). A replacement that starts after its predecessor stopped
     * answering is therefore stamped a later second. The guarantee needs startup_s >= 1 s; a
     * model that came up inside a second could collide, and the cost would be one stale extreme
     * carrying a true age.
     *
     * uptime_s is NOT a backstop for that case, because it is derived: unix_now() - started_at_,
     * both whole seconds, so started_at + uptime_s is exactly the wall clock at every poll. A
     * later poll always reports a larger uptime for the same started_at, so a comparison firing
     * on a smaller one detects a backward NTP step and nothing else. Checked before update() so
     * the new run's samples land in fresh windows. */
    if (lastStart !== null && s.started_at !== lastStart) newEngineRun();
    lastStart = s.started_at;
    const now = performance.now();
    update(s, prevAt ? (now - prevAt) / 1000 : 0);
    prev = s; prevAt = now;
    $("view-dash").classList.remove("boot");
    $("dot").className = "dot";
    schedulePoll(pollCadence(s.in_flight || 0));
  } catch (e) {
    $("dot").className = "dot stale";
    schedulePoll(pollIdleMs);
  }
}
identify();
if (location.hash === "#model") show("model");
else if (location.hash === "#sessions") show("sess");
else if (location.hash === "#advanced") show("adv");
/* A server that never answers still gets its frame drawn, rather than an empty page that gives
 * the reader nothing to tell "starting up" apart from "broken". */
setTimeout(() => $("view-dash").classList.remove("boot"), 1500);
poll();
document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
</script>
</body></html>
)HTML";

HttpResponse Server::handle_dashboard(const HttpRequest&) {
    HttpResponse r;
    r.status = 200;
    r.content_type = "text/html; charset=utf-8";
    r.body = kPage;
    /* No caching: the page must match the binary's /stats shape, and a stale copy shows only
     * blank panels. */
    r.set_header("Cache-Control", "no-store");
    /* The page is fixed markup that talks only to its own origin, and CSP is how that promise
     * becomes enforceable: even a future markup bug could not phone an injected payload's
     * findings home, and could not reach the key beyond this origin. script/style 'unsafe-inline'
     * is the cost of shipping as one compiled-in literal with its own <style> and <script>;
     * frame-ancestors keeps a hostile page from framing the Model view and clickjacking its
     * guardrail toggle. */
    r.set_header("Content-Security-Policy",
                 "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                 "connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
                 "form-action 'none'");
    r.set_header("X-Frame-Options", "DENY");
    return r;
}

}  /* namespace server */
}  /* namespace rad */
