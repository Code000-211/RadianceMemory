/* arch_fakes.h -- the recording builder, the run-phase fakes, the step lint and the fixtures the
 * architecture tests share (arch_test.cpp, glm5next_test.cpp).
 *
 * No GPU, no model file, no core. With two-phase construction (spec §3) the whole model is known
 * between declare and plan, so a unit test can check what a plugin declares. The fake builder
 * records every decl_weight, decl_buffer, decl_op, buffer use and name-map call, and the fake RadCtx
 * records every issue, so the run phase's order and operands are checked too.
 *
 * The fake is deliberately not core/build/'s RadBuilder: a defect there must not fail these tests,
 * and a plugin that only works against one builder has an undeclared dependency.
 *
 * It defines the declare- and run-phase C ABI, so exactly one translation unit of a test binary
 * includes it, and that binary does not link rad_core (tests/CMakeLists.txt, RAD_TESTS_NO_CORE). */
#ifndef RAD_ARCH_FAKES_H
#define RAD_ARCH_FAKES_H

#include "rad_test.h"

#include "rad_builder.h"
#include "rad_device.h"
#include "rad_runtime.h"

#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

/* ==================================================================== the recording builder */
namespace rad_fake {

struct RecParam {
    std::string key;
    int         kind = 0;
    long long   ival = 0, ihi = 0;
    double      dval = 0.0;
    std::string sval;
};

struct RecWeight {
    std::string     name;
    RadWeightDecl   d{};
    /* The uneven slice this weight was handed (rad_weight_shard_span), in the container's
     * unsharded units, so tests can check which checkpoint rows each rank reads. Even shards have
     * `span_set` false. */
    bool            span_set = false;
    int64_t         span_lo = 0, span_hi = 0;
    /* rad_weight_shard_span_parts, recorded likewise. */
    int             n_part_spans = 0;
    int64_t         part_lo[RAD_MAX_ROW_PARTS] = {0}, part_hi[RAD_MAX_ROW_PARTS] = {0};
    int64_t         part_total[RAD_MAX_ROW_PARTS] = {0};
};
struct RecBuf    { std::string name; RadBufDecl    d{}; };
struct RecOp {
    std::string             op;
    std::vector<RecParam>   p;
    std::vector<rad_weight> w;
    std::vector<rad_buf>    reads, writes;
    bool                    resolved = true;
    bool                    uses_declared = false;   /* rad_op_reads/writes were called */
};
struct RecMap {
    std::string              declared;
    int                      mode = 0;
    std::vector<std::string> src;
    int                      concat_dim = 0;
    std::vector<int>         src_index;   /* RadNameMap::src_index, one per source */
};
struct RecKV    { std::string name; RadKVGroupDecl d{}; bool zero = false; bool replicated = false;
                  bool dense = false; };
struct RecIssue {
    rad_op                  op = 0;
    std::vector<RadOperand> opd;
    int64_t                 n = 0;
    int                     lane = 0;   /* the lane it was issued on (rad_lane) */
};
/* A rad_lane_join: `to` behind everything `from` issued before issue number `at`. */
struct RecJoin  { size_t at = 0; int from = 0, to = 0; };

}  /* namespace rad_fake */
using namespace rad_fake;

struct RadBuilder {
    rad_buf    score_hidden = 0;   /* rad_declare_scores */
    rad_weight score_table = 0;
    std::vector<RecWeight> weights;
    std::vector<RecBuf>    bufs;
    std::vector<RecOp>     ops;
    std::vector<RecMap>    maps;
    std::vector<RecKV>     kv_groups;
    std::vector<std::pair<int, rad_kvgroup>> binds;
    std::vector<std::string> notes;

    /* Ops the fake kernel hierarchy refuses, to drive the fused-op fallback; and ops no loaded
     * library declares (rad_op_declared), as without an optional library. */
    std::set<std::string> refuse;
    std::set<std::pair<std::string, std::string>> refuse_dtype;  /* (op, dtype) forms */
    std::set<std::string> undeclared;
    /* When set, the only ops any loaded library declares (a CPU-only engine's: libref's
     * vocabulary); every other op is undeclared. */
    const std::set<std::string>* vocab = nullptr;
    std::vector<std::string> asked_undeclared;   /* what a real engine would refuse */
    bool declared(const std::string& op) const {
        return vocab ? vocab->count(op) != 0 : undeclared.count(op) == 0;
    }
    int64_t kv_block = 16;

    /* The buffer the plugin named as this step's logits (rad_declare_logits), or as its pooled
     * output (rad_declare_pooled). */
    rad_buf logits = 0;
    rad_buf pooled = 0;
    /* The words a request may hand the plugin (rad_declare_plugin_words). */
    int64_t plugin_words = 0;

    /* The draft head's declaration (rad_declare_drafter), and whether it keeps no cache of its
     * own (rad_drafter_shared_kv) and at which rotary offset. */
    RadDrafterDecl drafter{};
    std::string    drafter_name;
    bool           drafter_shared = false;
    int            drafter_pos = -1;
    int64_t        drafter_short = 0;     /* rad_drafter_short_block */
    /* A tree drafter's parents buffer (rad_drafter_tree), 0 for a chain. */
    rad_buf        drafter_parents = 0;

    /* The buffers named to rad_buf_concurrent: the ones a second lane touches. */
    std::vector<rad_buf> concurrent;
    /* The second lane's window (rad_lane1_window_begin), first and last op index; -1 when none. */
    int32_t lane1_w0 = -1, lane1_w1 = -1;

    /* The vision and audio encoder declarations (rad_declare_encoder), if any. */
    RadEncoderDecl encoder{};
    std::string    encoder_name;
    RadEncoderDecl audio_encoder{};
    std::string    audio_encoder_name;

    /* The image generator's declaration (rad_declare_generator), if any. */
    RadGeneratorDecl generator{};
    std::string      generator_name;

    /* Adapter key names (rad_lora_root, rad_lora_alias): (native, spellings), and each alias as
     * (native, from, to, from_lo, from_hi, to_lo). */
    std::vector<std::pair<std::string, std::string>> lora_roots;
    struct LoraAliasRec { std::string native, from, to; int64_t from_lo, from_hi, to_lo; };
    std::vector<LoraAliasRec> lora_aliases;

    /* What rad_weight_encoding answers: the first entry whose key the source name contains. Empty
     * means no model, so every weight is plain in its declared dtype. */
    std::vector<std::pair<std::string, RadEncoding>> encs;

    const RecOp* op_of(rad_op h) const { return h ? &ops[h - 1] : nullptr; }
};

struct StageRec {
    size_t issue = 0;   /* the issue count when the stage copy was issued */
    int    layer = -1;
};

struct RadCtx {
    RadBuilder*     b     = nullptr;
    const RadBatch* batch = nullptr;
    int             rank  = 0;
    int             world = 1;
    std::vector<RecIssue> issues;
    std::vector<int>      routed_layers;
    std::string           step_fail;
    int                   lane = 0;     /* where the next issue goes */
    std::vector<RecJoin>  joins;
    /* Device copies into a buffer (rad_memcpy_async to a rad_buf_ptr address): the buffer and the
     * issue number the copy came before. The step lint counts each as a write there. */
    std::vector<std::pair<size_t, rad_buf>> copies;
    /* The prefill-stage test: a stage call is recorded at the current issue index so the test can
     * assert it sits between the layer's store and its attention. */
    bool                  stage_on = false;
    int                   stage_last = -1;
    std::vector<StageRec> stages;
    /* kv_stage_next: after which issue index the next layer's copy was handed over. */
    std::vector<StageRec> stages_next;
};

/* -------------------------------------------------------------------- the declare-side ABI */
extern "C" rad_weight rad_decl_weight(RadBuilder* b, const char* name, const RadWeightDecl* d) {
    b->weights.push_back({name, *d});
    return (rad_weight)b->weights.size();
}

extern "C" int rad_weight_encoding(RadBuilder* b, const char* source, RadEncoding* out, int64_t*,
                                   uint32_t*) {
    for (const auto& [key, e] : b->encs)
        if (source && std::strstr(source, key.c_str())) { *out = e; return RAD_OK; }
    return RAD_E_NOTFOUND;
}
extern "C" rad_buf rad_decl_buffer(RadBuilder* b, const char* name, const RadBufDecl* d) {
    b->bufs.push_back({name, *d});
    return (rad_buf)b->bufs.size();
}

extern "C" rad_op rad_decl_op(RadBuilder* b, const char* op, const RadParam* params, int n_params,
                              const rad_weight* weights, int n_weights) {
    if (!b->declared(op)) {
        b->asked_undeclared.push_back(op);
        return RAD_NULL_HANDLE;
    }
    RecOp r;
    r.op = op;
    for (int i = 0; i < n_params; ++i) {
        RecParam p;
        p.key  = params[i].key ? params[i].key : "";
        p.kind = params[i].kind;
        p.ival = params[i].ival;
        p.ihi  = params[i].ihi;
        p.dval = params[i].dval;
        p.sval = params[i].sval ? params[i].sval : "";
        r.p.push_back(p);
    }
    for (int i = 0; i < n_weights; ++i) r.w.push_back(weights[i]);
    bool dtype_refused = false;
    for (const auto& p : r.p)
        if (p.key == "dtype" && b->refuse_dtype.count({r.op, p.sval})) dtype_refused = true;
    r.resolved = b->refuse.count(r.op) == 0 && !dtype_refused;
    b->ops.push_back(r);
    /* Declare always completes. An unresolved op is RAD_NULL_HANDLE, an error only if the plugin
     * issues it (rad_builder.h). */
    return r.resolved ? (rad_op)b->ops.size() : RAD_NULL_HANDLE;
}

extern "C" int rad_op_reads(RadBuilder* b, rad_op h, const rad_buf* bufs, int n) {
    if (!h) return RAD_E_INVAL;
    RecOp& r = b->ops[h - 1];
    r.uses_declared = true;
    for (int i = 0; i < n; ++i) r.reads.push_back(bufs[i]);
    return RAD_OK;
}

extern "C" int rad_op_writes(RadBuilder* b, rad_op h, const rad_buf* bufs, int n) {
    if (!h) return RAD_E_INVAL;
    RecOp& r = b->ops[h - 1];
    r.uses_declared = true;
    for (int i = 0; i < n; ++i) r.writes.push_back(bufs[i]);
    return RAD_OK;
}

extern "C" int rad_op_resolved(RadBuilder* b, rad_op h) {
    return h != RAD_NULL_HANDLE && b->ops[h - 1].resolved;
}

extern "C" int rad_op_declared(RadBuilder* b, const char* op) {
    return op && b->declared(op);
}

extern "C" rad_kvgroup rad_decl_kv_group(RadBuilder* b, const char* name,
                                         const RadKVGroupDecl* d) {
    b->kv_groups.push_back({name, *d, false});
    return (rad_kvgroup)b->kv_groups.size();
}

extern "C" int64_t rad_kv_block_size(RadBuilder* b, rad_kvgroup g) { (void)g; return b->kv_block; }

extern "C" int64_t rad_kv_layer_bytes(RadBuilder* b, rad_kvgroup g) {
    if (!b || !g || g > b->kv_groups.size()) return 0;
    const RadKVGroupDecl& d = b->kv_groups[g - 1].d;
    if (d.kind != RAD_KV_FULL && d.kind != RAD_KV_WINDOW) return 0;
    const int64_t nh = d.n_head_kv > 0 ? d.n_head_kv : 1;
    return rad_dtype_bytes(d.dtype, 2 * b->kv_block * nh * d.head_dim);
}

extern "C" int rad_bind_layer_kv(RadBuilder* b, int layer, rad_kvgroup g) {
    b->binds.push_back({layer, g});
    return RAD_OK;
}

extern "C" int rad_decl_name_map(RadBuilder* b, const RadNameMap* m) {
    RecMap r;
    r.declared   = m->declared;
    r.mode       = m->mode;
    r.concat_dim = m->concat_dim;
    for (int i = 0; i < m->n_src; ++i) {
        r.src.push_back(m->src[i]);
        r.src_index.push_back(m->src_index[i]);
    }
    b->maps.push_back(r);
    return RAD_OK;
}

extern "C" void rad_note(RadBuilder* b, const char* fmt, ...) {
    char buf[512];
    va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    b->notes.push_back(buf);
}

/* The buffer the core samples from, recorded so tests can assert the plugin named the narrow
 * one: [max_tok, n_vocab] is legal but 8 GiB at this vocabulary. */
extern "C" int rad_declare_logits(RadBuilder* b, rad_buf logits) {
    if (!b || !logits) return RAD_E_INVAL;
    b->logits = logits;
    return RAD_OK;
}

/* Scored steps' hidden rows and head table, recorded for the tests. */
extern "C" int rad_declare_scores(RadBuilder* b, rad_buf hidden, rad_weight table) {
    if (!b || !hidden || !table) return RAD_E_INVAL;
    b->score_hidden = hidden;
    b->score_table = table;
    return RAD_OK;
}

/* An embedding program names its pooled output instead of logits; a program declares one. */
extern "C" int rad_declare_pooled(RadBuilder* b, rad_buf pooled) {
    if (!b || !pooled || b->logits) return RAD_E_INVAL;
    b->pooled = pooled;
    return RAD_OK;
}

/* The words a request may hand the plugin, recorded as the core records them. */
extern "C" int rad_declare_plugin_words(RadBuilder* b, int64_t max_words) {
    if (!b || max_words < 0) return RAD_E_INVAL;
    b->plugin_words = max_words;
    return RAD_OK;
}

/* An image generator's declaration, recorded whole; it too excludes logits. */
extern "C" int rad_declare_generator(RadBuilder* b, const RadGeneratorDecl* d) {
    if (!b || !d || !d->image_out || b->logits || b->pooled) return RAD_E_INVAL;
    b->generator = *d;
    b->generator_name = d->name ? d->name : "";
    return RAD_OK;
}

/* Recorded: a generator's adapter key names. */
extern "C" int rad_lora_root(RadBuilder* b, const char* native, const char* spellings) {
    if (!b || !native || !*native) return RAD_E_INVAL;
    b->lora_roots.push_back({ native, spellings ? spellings : "" });
    return RAD_OK;
}
extern "C" int rad_lora_alias(RadBuilder* b, const char* native, const char* from, const char* to,
                              int64_t from_lo, int64_t from_hi, int64_t to_lo) {
    if (!b || !native || !from || !to) return RAD_E_INVAL;
    b->lora_aliases.push_back({ native, from, to, from_lo, from_hi, to_lo });
    return RAD_OK;
}

/* Recorded: gemma4's prefill overlap names the buffers its second lane reduces in place. */
extern "C" int rad_buf_concurrent(RadBuilder* b, rad_buf buf) {
    if (!b || !buf) return RAD_E_INVAL;
    b->concurrent.push_back(buf);
    return RAD_OK;
}
/* Recorded: the second lane's window (qwen35_fp8, gemma3), applied by liveness() as the real builder
 * applies it. Ops are handed out from 1 in declaration order, so the next op's index is size + 1. */
extern "C" int rad_lane1_window_begin(RadBuilder* b) {
    if (!b || b->lane1_w0 >= 0) return RAD_E_STATE;
    b->lane1_w0 = (int32_t)b->ops.size() + 1;
    return RAD_OK;
}
extern "C" int rad_lane1_window_end(RadBuilder* b) {
    if (!b || b->lane1_w0 < 0 || b->lane1_w1 >= 0 || (int32_t)b->ops.size() < b->lane1_w0)
        return RAD_E_STATE;
    b->lane1_w1 = (int32_t)b->ops.size();
    return RAD_OK;
}
extern "C" int rad_weight_shard_span(RadBuilder* b, rad_weight w, int64_t lo, int64_t hi) {
    if (!b || w == 0 || w > b->weights.size()) return RAD_E_INVAL;
    if (lo < 0 || hi <= lo) return RAD_E_INVAL;
    /* The real Builder also checks the declared extent against hi-lo (core/build/rad_builder.cpp);
     * the fake only records the span for tests. */
    RecWeight& rw = b->weights[w - 1];
    rw.span_set = true;
    rw.span_lo  = lo;
    rw.span_hi  = hi;
    return RAD_OK;
}
extern "C" int rad_weight_shard_span_parts(RadBuilder* b, rad_weight w, const int64_t* lo,
                                           const int64_t* hi, const int64_t* total, int n) {
    if (!b || w == 0 || w > b->weights.size() || !lo || !hi || !total) return RAD_E_INVAL;
    if (n < 2 || n > RAD_MAX_ROW_PARTS) return RAD_E_INVAL;
    RecWeight& rw = b->weights[w - 1];
    if (rw.d.n_row_parts != n) return RAD_E_INVAL;
    for (int g = 0; g < n; ++g) {
        if (lo[g] < 0 || hi[g] <= lo[g] || hi[g] > total[g]) return RAD_E_INVAL;
        if (rw.d.row_parts[g] != hi[g] - lo[g]) return RAD_E_SHAPE;
        rw.part_lo[g] = lo[g];
        rw.part_hi[g] = hi[g];
        rw.part_total[g] = total[g];
    }
    rw.n_part_spans = n;
    return RAD_OK;
}
extern "C" int rad_kv_group_zero(RadBuilder* b, rad_kvgroup g) {
    if (!b || g == 0 || g > b->kv_groups.size()) return RAD_E_INVAL;
    if (b->kv_groups[g - 1].replicated) return RAD_E_INVAL;
    b->kv_groups[g - 1].zero = true;
    return RAD_OK;
}
/* Recorded like zero, and like the real builder refused on a non-full group. Not part of
 * graph_hash(): the flag changes how much of the group the carve funds, not the graph, so the
 * pinned graphs stand. */
extern "C" int rad_kv_group_dense_read(RadBuilder* b, rad_kvgroup g) {
    if (!b || g == 0 || g > b->kv_groups.size()) return RAD_E_INVAL;
    if (b->kv_groups[g - 1].d.kind != RAD_KV_FULL) return RAD_E_INVAL;
    b->kv_groups[g - 1].dense = true;
    return RAD_OK;
}
/* Recorded like zero, and like the real builder refused on a zero or unpaged group. Not part of
 * graph_hash(): the flag changes what a host tier slot holds, not the graph, so the pinned
 * graphs stand. */
extern "C" int rad_kv_group_replicated(RadBuilder* b, rad_kvgroup g) {
    if (!b || g == 0 || g > b->kv_groups.size()) return RAD_E_INVAL;
    RecKV& k = b->kv_groups[g - 1];
    if (k.zero || (k.d.kind != RAD_KV_FULL && k.d.kind != RAD_KV_WINDOW)) return RAD_E_INVAL;
    k.replicated = true;
    return RAD_OK;
}
extern "C" int rad_declare_drafter(RadBuilder* b, const RadDrafterDecl* d) {
    if (!b || !d) return RAD_E_INVAL;
    b->drafter = *d;
    b->drafter_name = d->name ? d->name : "";
    return RAD_OK;
}
extern "C" int rad_drafter_shared_kv(RadBuilder* b, int pos_offset) {
    if (!b || b->drafter.kind != RAD_DRAFT_SERIAL || (pos_offset != 0 && pos_offset != 1))
        return RAD_E_STATE;
    b->drafter_shared = true;
    b->drafter_pos = pos_offset;
    return RAD_OK;
}
extern "C" int rad_drafter_tree(RadBuilder* b, rad_buf parents) {
    if (!b || b->drafter.kind != RAD_DRAFT_BLOCK || parents == 0) return RAD_E_STATE;
    b->drafter_parents = parents;
    return RAD_OK;
}
/* The core's chaining decision as the recorder answers it (it has no chaining backend): 0 unless a
 * test holds a ChainOn, to see where a plugin places work when launches chain. */
static int g_chain_on = 0;
extern "C" int rad_chain_on(void) { return g_chain_on; }
struct ChainOn {
    explicit ChainOn(int on) { g_chain_on = on; }
    ~ChainOn() { g_chain_on = 0; }
};

extern "C" int rad_drafter_short_block(RadBuilder* b, int64_t block) {
    const int64_t full = b ? (b->drafter.block > 0 ? b->drafter.block : b->drafter.depth + 1) : 0;
    if (!b || b->drafter.kind != RAD_DRAFT_BLOCK) return RAD_E_STATE;
    if (b->drafter_parents ? (block < 2 || block >= full)
                           : (block >= full || block - (full - b->drafter.depth) < 1))
        return RAD_E_INVAL;
    b->drafter_short = block;
    return RAD_OK;
}
extern "C" int rad_declare_encoder(RadBuilder* b, const RadEncoderDecl* d) {
    if (!b || !d || !d->name || !d->modalities) return RAD_E_INVAL;
    if (d->modalities == RAD_MM_AUDIO) {
        b->audio_encoder = *d;
        b->audio_encoder_name = d->name;
        return RAD_OK;
    }
    b->encoder = *d;
    b->encoder_name = d->name;
    return RAD_OK;
}

/* -------------------------------------------------------------------- model metadata */
extern "C" long long rad_meta_geti(const RadModelMeta* m, const char* key, long long dflt) {
    for (int i = 0; i < m->n_kv; ++i)
        if (std::strcmp(m->kv_key[i], key) == 0) return atoll(m->kv_val[i]);
    return dflt;
}
extern "C" double rad_meta_getf(const RadModelMeta* m, const char* key, double dflt) {
    for (int i = 0; i < m->n_kv; ++i)
        if (std::strcmp(m->kv_key[i], key) == 0) return atof(m->kv_val[i]);
    return dflt;
}
extern "C" const char* rad_meta_gets(const RadModelMeta* m, const char* key, const char* dflt) {
    for (int i = 0; i < m->n_kv; ++i)
        if (std::strcmp(m->kv_key[i], key) == 0) return m->kv_val[i];
    return dflt;
}

/* -------------------------------------------------------------------- the run-phase ABI */
extern "C" int rad_issue(RadCtx* c, rad_op op, const RadOperand* opd, int n_opd, int64_t n) {
    RecIssue r;
    r.op = op;
    r.n  = n;
    for (int i = 0; i < n_opd; ++i) r.opd.push_back(opd[i]);
    r.lane = c->lane;
    c->issues.push_back(r);
    return RAD_OK;
}
/* The second lane, recorded: each issue's lane, and each join's position among the issues. */
extern "C" int rad_lane(RadCtx* c, int lane) {
    if (lane != 0 && lane != 1) return RAD_E_INVAL;
    c->lane = lane;
    return RAD_OK;
}
extern "C" int rad_lane_join(RadCtx* c, int from, int to) {
    if (from < 0 || from > 1 || to < 0 || to > 1) return RAD_E_INVAL;
    c->joins.push_back({ c->issues.size(), from, to });
    return RAD_OK;
}
/* No pool binding here, so a layer is at the index it names: the arch test uses layer 0 where it
 * checks the mirror offset. */
extern "C" int rad_kv_layer_index(RadCtx* c, rad_kvgroup g, int layer) {
    (void)c; (void)g;
    return layer;
}
/* The prefill stage: no device here, so record the first call for a layer at its place in the
 * issue stream, which is before the layer's store. Off unless a test selects the stage view. */
extern "C" int rad_kv_stage_layer(RadCtx* c, rad_kvgroup g, const RadKVGroupBatch* kvb, int layer) {
    (void)g;
    if (!c || !kvb || !kvb->stream || !c->stage_on) return 0;
    if (layer == c->stage_last) return 1;
    c->stage_last = layer;
    c->stages.push_back(StageRec{ c->issues.size(), layer });
    return 1;
}
/* The next layer's copy is handed to the DMA engine after this layer's attention. */
extern "C" int rad_kv_stage_next(RadCtx* c, rad_kvgroup g, int layer) {
    (void)g;
    if (!c || !c->stage_on) return 0;
    c->stages_next.push_back(StageRec{ c->issues.size(), layer });
    return 0;
}
/* A block that refuses the batch records the reason for the tests. */
extern "C" int rad_step_fail(RadCtx* c, const char* what) {
    c->step_fail = what ? what : "";
    return RAD_E_SHAPE;
}
extern "C" const RadBatch* rad_batch(RadCtx* c)      { return c->batch; }
extern "C" int             rad_rank(RadCtx* c)       { return c->rank; }
extern "C" int             rad_world_size(RadCtx* c) { return c->world; }
extern "C" int             rad_ctx_wgp_count(RadCtx*) { return 32; }
/* The context itself, so a copy issued on the step's stream is recorded in it (rad_memcpy_async). */
extern "C" RadStream       rad_stream(RadCtx* c)     { return (RadStream)c; }
extern "C" int             rad_stream_sync(RadStream s) { (void)s; return RAD_OK; }
extern "C" int rad_route_report(RadCtx* c, int layer, const RadRouting* r) {
    (void)r;
    c->routed_layers.push_back(layer);
    return RAD_OK;
}
/* No card-side histogram here, so a router writes and reports its own buffer. */
extern "C" int32_t* rad_route_counts(RadCtx* c, int layer, int64_t n_expert) {
    (void)c; (void)layer; (void)n_expert;
    return nullptr;
}
/* No prefetch here: the three calls are no-ops, as in a core without one. */
extern "C" int rad_expert_prefetch(RadCtx* c, const RadPrefetchNext* next, rad_buf h, int64_t r0,
                                   int64_t rows, int64_t n_embd) {
    (void)c; (void)next; (void)h; (void)r0; (void)rows; (void)n_embd;
    return RAD_OK;
}
extern "C" int rad_expert_prefetch_mark(RadCtx* c) { (void)c; return RAD_OK; }
extern "C" int rad_expert_prefetch_step(RadCtx* c, int layer) { (void)c; (void)layer; return RAD_OK; }
/* Never dereferenced: a plugin passes these through to rad_route_report or offsets one into a raw
 * operand. Each handle gets its own 2^36-byte window above any user-space address, tagged by
 * kind in the top bits, so the step lint can map a raw operand inside a buffer back to that buffer
 * (fake_buf_of). */
constexpr int       kFakePtrShift = 36;
constexpr uintptr_t kFakeBufTag   = (uintptr_t)1 << 61;
constexpr uintptr_t kFakeWeightTag = (uintptr_t)1 << 62;
extern "C" void* rad_weight_ptr(RadCtx* c, rad_weight w) {
    (void)c;
    return (void*)(kFakeWeightTag | (uintptr_t)w << kFakePtrShift);
}
extern "C" void* rad_buf_ptr(RadCtx* c, rad_buf b) {
    (void)c;
    return (void*)(kFakeBufTag | (uintptr_t)b << kFakePtrShift);
}
/* The buffer a raw operand points into, if it came from rad_buf_ptr; 0 otherwise. */
static rad_buf fake_buf_of(const void* p, size_t n_bufs) {
    const uintptr_t u = (uintptr_t)p;
    if ((u & ~(kFakeBufTag - 1)) != kFakeBufTag) return 0;
    const uintptr_t h = (u & (kFakeBufTag - 1)) >> kFakePtrShift;
    return h >= 1 && h <= n_bufs ? (rad_buf)h : 0;
}
/* Pinned host memory a plugin fills itself (qwen4exp_fp8's piece bounds, the calibration tap's
 * staging), as host memory; nothing else is handed out. */
extern "C" void* rad_dev_alloc(int64_t bytes, int kind) {
    return kind == RAD_MEM_HOST_PINNED && bytes > 0 ? std::calloc(1, (size_t)bytes) : nullptr;
}
extern "C" void  rad_dev_free(void* p, int kind) { if (kind == RAD_MEM_HOST_PINNED) std::free(p); }
/* The device copies: nothing is copied. One into a buffer on a step's stream is recorded as a
 * write of that buffer before the next issue (RadCtx::copies). */
extern "C" int rad_memcpy_async(void* dst, const void* src, int64_t bytes, RadStream s) {
    (void)src; (void)bytes;
    RadCtx* c = (RadCtx*)s;
    if (c && c->b)
        if (const rad_buf h = fake_buf_of(dst, c->b->bufs.size()))
            c->copies.push_back({ c->issues.size(), h });
    return RAD_OK;
}
/* The pitched copy: recorded exactly as the 1-D one (its destination's first byte names the
 * buffer); the unlimitedocr tower packs patch rows and lays out its map with it. */
extern "C" int rad_memcpy_2d_async(void* dst, int64_t dpitch, const void* src, int64_t spitch,
                                   int64_t width, int64_t height, RadStream s) {
    (void)dpitch; (void)src; (void)spitch; (void)width; (void)height;
    RadCtx* c = (RadCtx*)s;
    if (c && c->b)
        if (const rad_buf h = fake_buf_of(dst, c->b->bufs.size()))
            c->copies.push_back({ c->issues.size(), h });
    return RAD_OK;
}
extern "C" int rad_memset_async(void* dst, int value, int64_t bytes, RadStream s) {
    (void)dst; (void)value; (void)bytes; (void)s;
    return RAD_OK;
}
/* Events complete at once: the fake stream runs nothing. The calibration tap waits on one. */
extern "C" int  rad_event_create(RadEvent* out) { if (out) *out = (RadEvent)out; return RAD_OK; }
extern "C" void rad_event_destroy(RadEvent e)   { (void)e; }
extern "C" int  rad_event_record(RadEvent e, RadStream s) { (void)e; (void)s; return RAD_OK; }
extern "C" int  rad_event_sync(RadEvent e)      { (void)e; return RAD_OK; }

/* ==================================================================== fixtures and helpers */
namespace rad_fake {

/* A deployment: one rank, 256 rows a step, eight sequences, 32k of context. */
RadBuildCtx build_ctx(int rank = 0, int world = 1) {
    RadBuildCtx c{};
    c.rank       = rank;
    c.world_size = world;
    c.max_tok    = 256;
    c.max_seqs   = 8;
    c.max_ctx    = 32768;
    c.scope      = "";
    return c;
}

/* The sampled rows (RadBatch::out_ids). The plugin skips the lm_head when n_out is zero (a chunk
 * finishing no sequence), so a fake batch must set them or it silently tests the no-logits
 * path. */
static const int32_t kDecodeOut[]  = { 0, 1, 2, 3 };
static const int32_t kPrefillOut[] = { 31, 63 };   /* last token of each 32-token sequence */

RadBatch decode_batch() {
    RadBatch b{};
    b.phase       = RAD_PHASE_DECODE;
    b.n_tok       = 4;
    b.n_seq       = 4;
    b.max_q_len   = 1;
    b.max_ctx_len = 100;
    b.n_out       = 4;
    b.out_ids     = kDecodeOut;
    return b;
}

RadBatch prefill_batch() {
    RadBatch b{};
    b.phase       = RAD_PHASE_PREFILL;
    b.n_tok       = 64;
    b.n_seq       = 2;
    b.max_q_len   = 32;
    b.max_ctx_len = 32;
    b.n_out       = 2;
    b.out_ids     = kPrefillOut;
    return b;
}

/* --- small queries over the recorded graph --- */
std::vector<std::string> op_names(const RadBuilder& b) {
    std::vector<std::string> v;
    for (const auto& o : b.ops) v.push_back(o.op);
    return v;
}

const RecOp* find_op(const RadBuilder& b, const std::string& name, int nth = 0) {
    for (const auto& o : b.ops)
        if (o.op == name && nth-- == 0) return &o;
    return nullptr;
}

int count_op(const RadBuilder& b, const std::string& name) {
    int n = 0;
    for (const auto& o : b.ops) if (o.op == name) ++n;
    return n;
}

const RecParam* param(const RecOp* o, const char* key) {
    if (!o) return nullptr;
    for (const auto& p : o->p) if (p.key == key) return &p;
    return nullptr;
}

long long pint(const RecOp* o, const char* key) {
    const RecParam* p = param(o, key);
    return p ? p->ival : -12345;
}

double pflt(const RecOp* o, const char* key) {
    const RecParam* p = param(o, key);
    return p ? p->dval : -12345.0;
}

std::string pstr(const RecOp* o, const char* key) {
    const RecParam* p = param(o, key);
    return p ? p->sval : std::string("<absent>");
}

const RecWeight* find_w(const RadBuilder& b, const std::string& name) {
    for (const auto& w : b.weights) if (w.name == name) return &w;
    return nullptr;
}

const RecBuf* find_b(const RadBuilder& b, const std::string& name) {
    for (const auto& x : b.bufs) if (x.name == name) return &x;
    return nullptr;
}

const RecMap* find_map(const RadBuilder& b, const std::string& declared) {
    for (const auto& m : b.maps) if (m.declared == declared) return &m;
    return nullptr;
}

std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += " "; s += v[i]; }
    return s;
}

std::vector<std::string> issued_ops(const RadBuilder& b, const RadCtx& c) {
    std::vector<std::string> got;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        got.push_back(o ? o->op : "<null>");
    }
    return got;
}

/* `fn`'s status, with what it prints kept out of the test output. */
int stderr_of_rc(const std::function<int()>& fn) {
    int rc = 0;
    stderr_of([&] { rc = fn(); });
    return rc;
}

/* Model metadata built key by key; `meta()` is valid while this object lives. */
struct MetaKV {
    RadModelMeta              m{};
    std::vector<std::string>  keys, vals;
    std::vector<const char*>  kp, vp;
    explicit MetaKV(const RadModelMeta& base) : m(base) {
        for (int i = 0; i < base.n_kv; ++i) set(base.kv_key[i], base.kv_val[i]);
    }
    MetaKV& set(const char* k, const char* v) {
        for (size_t i = 0; i < keys.size(); ++i)
            if (keys[i] == k) { vals[i] = v; return *this; }
        keys.push_back(k);
        vals.push_back(v);
        return *this;
    }
    MetaKV& drop(const char* k) {
        for (size_t i = 0; i < keys.size(); ++i)
            if (keys[i] == k) {
                keys.erase(keys.begin() + (long)i);
                vals.erase(vals.begin() + (long)i);
                break;
            }
        return *this;
    }
    const RadModelMeta* meta() {
        kp.clear(); vp.clear();
        for (size_t i = 0; i < keys.size(); ++i) {
            kp.push_back(keys[i].c_str());
            vp.push_back(vals[i].c_str());
        }
        m.n_kv = (int)kp.size();
        m.kv_key = kp.data();
        m.kv_val = vp.data();
        return &m;
    }
};

RadBuildCtx spec_ctx(int rank, int world, int max_spec) {
    RadBuildCtx c = build_ctx(rank, world);
    c.max_spec = max_spec;
    return c;
}

const RecIssue* issue_of(const RadBuilder& b, const RadCtx& c, const std::string& name,
                         int nth = 0) {
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == name && nth-- == 0) return &i;
    }
    return nullptr;
}

int count_issued(const std::vector<std::string>& v, const std::string& name) {
    return (int)std::count(v.begin(), v.end(), name);
}

}  /* namespace rad_fake */

/* ==================================================================== the step lint
 *
 * The buffer plan takes liveness from declaration order: a transient lives over the [first, last]
 * op indices of its declared reads and writes, and two transients of one arena whose ranges are
 * disjoint may share bytes (core/build/rad_bufplan.cpp). A step that issues ops in another order,
 * or across passes of one engine step (trunk, drafter history, draft rounds), can write one buffer
 * while another that shares its bytes still holds rows a later issue reads. The runtime refuses
 * only an op outside a buffer's range, and only on the branch a real model runs, so the class is
 * checked here for every recorded step.
 *
 * lint_step walks the issues of one engine step's passes in order and reports:
 *   alias       a transient read whose last writer is followed, before the read, by a write of a
 *               transient of the same arena whose range is disjoint from its own;
 *   undeclared  an op issued outside the declared range of a buffer operand
 *               (core/runtime/issue.cpp refuses this);
 *   span        an operand that reaches past its buffer or weight, rows and columns applied as
 *               issue.cpp applies them;
 *   null op     an issue of RAD_NULL_HANDLE (a fused op that did not resolve, with no chain);
 *   unwritten   a transient read before any issue, or device copy into it, of the step wrote it;
 *   kv          a cache operand naming a group or layer no rad_bind_layer_kv bound, or a group
 *               this rank declared zero;
 *   strided     a GEMM whose input is a column slice narrower than its buffer (libr4d's GEMMs
 *               read whole rows and refuse it). */
namespace rad_fake {

/* One buffer's planned lifetime. `planned` is false for a buffer live over the whole program
 * (PERSIST, DERIVED, rad_buf_concurrent without a second-lane window, or no declared use), which
 * shares bytes with nothing. */
struct Live {
    int32_t lo = 0, hi = 0;
    bool    planned = false;
    bool    transient = false;
    int     domain = RAD_DOMAIN_DEVICE;
};

/* Liveness by the rule of Builder::finish_buffer_liveness (core/build/rad_builder.cpp). Op handle
 * h is op index h: the real builder keeps a sentinel at index 0, and the fake hands out ops in
 * declaration order from 1. */
std::vector<Live> liveness(const RadBuilder& b) {
    const size_t  nb = b.bufs.size();
    const int32_t last_op = (int32_t)b.ops.size();
    std::vector<int32_t> first(nb + 1, -1), last(nb + 1, -1);
    for (size_t i = 0; i < b.ops.size(); ++i)
        for (const std::vector<rad_buf>* v : { &b.ops[i].reads, &b.ops[i].writes })
            for (rad_buf h : *v) {
                if (h == 0 || h > nb) continue;
                if (first[h] < 0) first[h] = (int32_t)(i + 1);
                last[h] = (int32_t)(i + 1);
            }
    std::set<rad_buf> conc(b.concurrent.begin(), b.concurrent.end());
    std::vector<Live> L(nb + 1);
    for (size_t h = 1; h <= nb; ++h) {
        const RadBufDecl& d = b.bufs[h - 1].d;
        Live& l = L[h];
        l.transient = d.kind == RAD_BUF_TRANSIENT;
        l.domain    = d.domain;
        if (!l.transient || first[h] < 0 || (conc.count((rad_buf)h) && b.lane1_w1 < 0)) {
            l.lo = 0;
            l.hi = last_op;
            continue;
        }
        if (conc.count((rad_buf)h)) {   /* under the window: across it and over its own uses */
            l.planned = true;
            l.lo = std::min(first[h], b.lane1_w0);
            l.hi = std::max(last[h], b.lane1_w1);
            continue;
        }
        l.planned = true;
        l.lo = first[h];
        l.hi = last[h];
        if ((rad_buf)h == b.logits || (rad_buf)h == b.pooled) l.hi = last_op;
    }
    return L;
}

/* Whether operand `o` stays inside a buffer or weight of `shape` (packed strides, rows and cols
 * applied as issue.cpp applies them), as issue.cpp's operand_fits decides. */
bool operand_fits(const int64_t* shape, uint32_t rank, const RadOperand& o) {
    if (o.offset < 0) return false;
    int64_t sh[RAD_MAX_RANK] = {}, st[RAD_MAX_RANK] = {};
    int64_t numel = 1;
    for (uint32_t d = 0; d < rank; ++d) { sh[d] = shape[d]; numel *= shape[d]; }
    int64_t s = 1;
    for (int d = (int)rank - 1; d >= 0; --d) { st[d] = s; s *= sh[d]; }
    if (o.rows > 0 && rank >= 1) sh[0] = o.rows;
    if (o.cols > 0 && rank >= 1) sh[rank - 1] = o.cols;
    int64_t span = o.offset;
    for (uint32_t d = 0; d < rank; ++d) {
        if (sh[d] <= 0) return true;
        span += (sh[d] - 1) * st[d];
    }
    return span < numel;
}

bool in(const std::vector<rad_buf>& v, rad_buf h) {
    return std::find(v.begin(), v.end(), h) != v.end();
}

std::vector<std::string> lint_step(const RadBuilder& b, const std::vector<const RadCtx*>& passes) {
    std::vector<std::string> bad;
    const std::vector<Live> L = liveness(b);
    const size_t nb = b.bufs.size();
    auto name = [&](rad_buf h) { return h && h <= nb ? b.bufs[h - 1].name : std::string("?"); };
    auto shares = [&](rad_buf x, rad_buf y) {
        const Live& p = L[x];
        const Live& q = L[y];
        return p.planned && q.planned && p.domain == q.domain && (p.hi < q.lo || q.hi < p.lo);
    };
    std::set<std::pair<int, rad_kvgroup>> bound(b.binds.begin(), b.binds.end());
    std::vector<int64_t> last_write(nb + 1, -1);
    int64_t pos = 0;
    char where[96];
    for (size_t pi = 0; pi < passes.size(); ++pi) {
        const RadCtx& c = *passes[pi];
        size_t next_copy = 0;
        for (size_t ii = 0; ii < c.issues.size(); ++ii, ++pos) {
            for (; next_copy < c.copies.size() && c.copies[next_copy].first <= ii; ++next_copy)
                last_write[c.copies[next_copy].second] = pos;
            const RecIssue& is = c.issues[ii];
            const RecOp* o = b.op_of(is.op);
            std::snprintf(where, sizeof where, "pass %zu issue %zu (%s, op %u)", pi, ii,
                          o ? o->op.c_str() : "<null>", (unsigned)is.op);
            if (!o) { bad.push_back(std::string(where) + ": null op issued"); continue; }
            std::vector<rad_buf> rd, wr;
            for (size_t k = 0; k < is.opd.size(); ++k) {
                const RadOperand& od = is.opd[k];
                const std::string at = std::string(where) + " operand " + std::to_string(k);
                if (od.kind == RAD_OPK_BUF) {
                    const rad_buf h = od.handle;
                    if (h == 0 || h > nb) {
                        bad.push_back(at + ": buffer handle out of range");
                        continue;
                    }
                    const RadBufDecl& d = b.bufs[h - 1].d;
                    if (!operand_fits(d.shape, d.rank, od))
                        bad.push_back(at + ": span past '" + name(h) + "' (offset " +
                                      std::to_string(od.offset) + ", rows " +
                                      std::to_string(od.rows) + ", cols " +
                                      std::to_string(od.cols) + ")");
                    const Live& l = L[h];
                    if (l.planned && ((int32_t)is.op < l.lo || (int32_t)is.op > l.hi))
                        bad.push_back(at + ": undeclared use of '" + name(h) + "' (planned ops " +
                                      std::to_string(l.lo) + ".." + std::to_string(l.hi) + ")");
                    const bool r = in(o->reads, h), w = in(o->writes, h);
                    if (r || !w) rd.push_back(h);
                    if (w || !r) wr.push_back(h);
                    /* A GEMM's input is read as contiguous rows: libr4d refuses a column slice
                     * narrower than its buffer. */
                    const int64_t width = d.rank ? d.shape[d.rank - 1] : 0;
                    if (k == 0 && o->op.rfind("gemm_nt", 0) == 0 && od.cols > 0 &&
                        (od.cols != width || od.offset % width))
                        bad.push_back(at + ": " + o->op + " reads " + std::to_string(od.cols) +
                                      " of " + std::to_string(width) + " columns of '" + name(h) +
                                      "' as its input, which is not contiguous");
                } else if (od.kind == RAD_OPK_WEIGHT) {
                    const uint32_t h = od.handle;
                    if (h == 0 || h > b.weights.size()) {
                        bad.push_back(at + ": weight handle out of range");
                        continue;
                    }
                    const RadWeightDecl& d = b.weights[h - 1].d;
                    if ((od.offset || od.rows > 0 || od.cols > 0) &&
                        !operand_fits(d.shape, d.rank, od))
                        bad.push_back(at + ": span past weight '" + b.weights[h - 1].name + "'");
                } else if (od.kind == RAD_OPK_WTAB) {
                    if (od.handle == 0 || od.rows <= 0 || od.offset || od.cols ||
                        (size_t)(od.handle + od.rows - 1) > b.weights.size())
                        bad.push_back(at + ": weight table out of range");
                } else if (od.kind == RAD_OPK_KV) {
                    const rad_kvgroup g = od.handle;
                    if (g == 0 || g > b.kv_groups.size())
                        bad.push_back(at + ": KV group handle out of range");
                    else if (b.kv_groups[g - 1].zero)
                        bad.push_back(at + ": KV group '" + b.kv_groups[g - 1].name +
                                      "' is declared zero on this rank");
                    else if (!bound.count({ (int)od.offset, g }))
                        bad.push_back(at + ": layer " + std::to_string(od.offset) +
                                      " is not bound to KV group '" + b.kv_groups[g - 1].name +
                                      "'");
                } else if (od.kind == RAD_OPK_RAW) {
                    if (od.offset && od.dtype == RAD_DT_INVALID)
                        bad.push_back(at + ": raw operand with an offset and no dtype");
                    /* A pointer into a buffer (rad_buf_ptr) is that buffer to the plan, so it is
                     * classified by the op's declared sets like a buffer operand. */
                    if (const rad_buf h = fake_buf_of(od.raw, nb)) {
                        const Live& l = L[h];
                        if (l.planned && ((int32_t)is.op < l.lo || (int32_t)is.op > l.hi))
                            bad.push_back(at + ": undeclared use of '" + name(h) +
                                          "' through a raw pointer");
                        const bool r = in(o->reads, h), w = in(o->writes, h);
                        if (r || !w) rd.push_back(h);
                        if (w || !r) wr.push_back(h);
                    }
                }
            }
            for (rad_buf h : rd) {
                if (!L[h].transient) continue;
                const int64_t w = last_write[h];
                if (w < 0) {
                    bad.push_back(std::string(where) + ": reads '" + name(h) +
                                  "' before any issue of this step wrote it");
                    continue;
                }
                for (size_t x = 1; x <= nb; ++x)
                    if (x != h && last_write[x] > w && shares(h, (rad_buf)x))
                        bad.push_back(std::string(where) + ": reads '" + name(h) + "' (ops " +
                                      std::to_string(L[h].lo) + ".." + std::to_string(L[h].hi) +
                                      ") after a write of '" + name((rad_buf)x) + "' (ops " +
                                      std::to_string(L[x].lo) + ".." + std::to_string(L[x].hi) +
                                      "), which may share its bytes");
            }
            for (rad_buf h : wr) last_write[h] = pos;
        }
    }
    return bad;
}

/* ---------------------------------------------------------------- whole step batches
 *
 * A batch as the core builds it: every per-token and per-sequence array filled and a group batch
 * for every KV group the builder declared, so a plugin's step can take any path. Decode sequences
 * come first (RadBatch::n_seq_decode); each `dq` entry is a decode sequence's query length (1, or
 * 1 + n_spec for a verify), each `pq` entry a prefill chunk. Every decode row is sampled (a verify
 * needs logits at each row) and a chunk's last row; a chunk continues `prefill_ctx` tokens of
 * history. */
struct Kit {
    std::vector<int32_t> tok, pos, cu, ctxl, qlen, out, nacc, dout;
    std::vector<int32_t> slot, table, used, sidx;
    std::vector<RadKVGroupBatch> kv;
    RadBatch b{};
};

std::unique_ptr<Kit> kit(const RadBuilder& bld, const std::vector<int>& dq,
                         const std::vector<int>& pq, int32_t prefill_ctx = 0) {
    auto k = std::make_unique<Kit>();
    const int64_t nd = (int64_t)dq.size(), ns = nd + (int64_t)pq.size();
    std::vector<int> ql(dq);
    ql.insert(ql.end(), pq.begin(), pq.end());
    int64_t T = 0, max_q = 0, max_ctx = 0, td = 0, mqd = 0, mqp = 0;
    k->cu.push_back(0);
    for (int64_t s = 0; s < ns; ++s) {
        const bool dec = s < nd;
        const int32_t ctx = dec ? (int32_t)(40 + 7 * s) : prefill_ctx;
        for (int i = 0; i < ql[(size_t)s]; ++i) {
            k->tok.push_back((int32_t)(3 + (T + i) % 97));
            k->pos.push_back(ctx + i);
            if (dec || i + 1 == ql[(size_t)s]) k->out.push_back((int32_t)(T + i));
        }
        T += ql[(size_t)s];
        k->cu.push_back((int32_t)T);
        k->ctxl.push_back(ctx);
        k->qlen.push_back(ql[(size_t)s]);
        k->nacc.push_back(1);
        k->dout.push_back((int32_t)T - 1);
        max_q = std::max<int64_t>(max_q, ql[(size_t)s]);
        max_ctx = std::max<int64_t>(max_ctx, ctx + ql[(size_t)s]);
        if (dec) { td += ql[(size_t)s]; mqd = std::max<int64_t>(mqd, ql[(size_t)s]); }
        else     mqp = std::max<int64_t>(mqp, ql[(size_t)s]);
    }
    k->tok.resize((size_t)T + 64, 5);   /* room for a following prompt (RadBatch::n_ahead) */
    const int64_t pitch = 16;
    for (int64_t i = 0; i < T; ++i) k->slot.push_back((int32_t)i);
    for (int64_t i = 0; i < ns * pitch; ++i) k->table.push_back((int32_t)i);
    for (int64_t s = 0; s < ns; ++s) {
        k->used.push_back(k->ctxl[(size_t)s] + k->qlen[(size_t)s]);
        k->sidx.push_back((int32_t)s);
    }
    for (size_t g = 0; g < bld.kv_groups.size(); ++g) {
        RadKVGroupBatch gb{};
        gb.group = (rad_kvgroup)(g + 1);
        gb.slot_mapping = k->slot.data();
        gb.block_table = k->table.data();
        gb.block_table_pitch = pitch;
        gb.seqused = k->used.data();
        gb.state_index = k->sidx.data();
        gb.state_index_pitch = 1;
        gb.block_size = bld.kv_block;
        gb.max_blocks = pitch;
        k->kv.push_back(gb);
    }
    RadBatch& b = k->b;
    b.phase = pq.empty() ? RAD_PHASE_DECODE : dq.empty() ? RAD_PHASE_PREFILL : RAD_PHASE_MIXED;
    b.n_tok = T;
    b.n_seq = ns;
    b.token_ids = k->tok.data();
    b.positions = k->pos.data();
    b.cu_seqlens = k->cu.data();
    b.n_out = (int64_t)k->out.size();
    b.out_ids = k->out.data();
    b.q_lens = k->qlen.data();
    b.ctx_lens = k->ctxl.data();
    b.max_q_len = (int32_t)max_q;
    b.max_ctx_len = (int32_t)max_ctx;
    b.n_kv_groups = (int)k->kv.size();
    b.kv = k->kv.data();
    b.n_spec = mqd > 1 ? (int)(mqd - 1) : 0;
    b.num_accepted = k->nacc.data();
    b.n_seq_decode = nd;
    b.n_tok_decode = td;
    b.max_q_len_decode = (int32_t)mqd;
    b.max_q_len_prefill = (int32_t)mqp;
    return k;
}

/* The draft passes the core stages for a step of `n_seq` sequences.
 *
 * A serial head's history pass (draft_pass -1) covers every row the trunk step had and carries
 * round 1 on each sequence's last row; round r (2..depth) is one row a sequence. A block
 * drafter's context pass (-1) covers the trunk's rows; its query pass (1) is `block` rows a
 * sequence, the anchor first, and samples the `steps` rows after it (RadDrafterDecl::block). */
std::unique_ptr<Kit> history_pass(const RadBuilder& bld, const Kit& trunk) {
    std::vector<int> rows;
    for (int64_t s = 0; s < trunk.b.n_seq; ++s) rows.push_back(trunk.qlen[(size_t)s]);
    auto k = kit(bld, rows, {});
    k->b.draft_pass = -1;
    k->out.clear();
    for (int64_t i = 0; i < k->b.n_tok; ++i) k->out.push_back((int32_t)i);
    k->b.out_ids = k->out.data();
    k->b.n_out = k->b.n_tok;
    k->b.n_draft_out = k->b.n_seq;
    k->b.draft_out_ids = k->dout.data();
    return k;
}
std::unique_ptr<Kit> round_pass(const RadBuilder& bld, int64_t n_seq, int r) {
    auto k = kit(bld, std::vector<int>((size_t)n_seq, 1), {});
    k->b.draft_pass = r;
    return k;
}
std::unique_ptr<Kit> context_pass(const RadBuilder& bld, const Kit& trunk) {
    auto k = history_pass(bld, trunk);
    k->b.n_draft_out = 0;
    k->b.draft_out_ids = nullptr;
    return k;
}
std::unique_ptr<Kit> query_pass(const RadBuilder& bld, int64_t n_seq, int64_t block,
                                int64_t steps) {
    auto k = kit(bld, std::vector<int>((size_t)n_seq, (int)block), {});
    k->b.draft_pass = 1;
    k->out.clear();
    for (int64_t s = 0; s < n_seq; ++s)
        for (int64_t i = 0; i < steps; ++i)
            k->out.push_back((int32_t)(s * block + block - steps + i));
    k->b.out_ids = k->out.data();
    k->b.n_out = (int64_t)k->out.size();
    return k;
}

using StepFn = void (*)(RadCtx*, const RadBatch*);

/* Runs one engine step's passes in order, each on a context of its own. */
std::vector<std::unique_ptr<RadCtx>> run_passes(RadBuilder& b, StepFn step,
                                                const std::vector<const Kit*>& passes,
                                                int rank = 0, int world = 1) {
    std::vector<std::unique_ptr<RadCtx>> out;
    for (const Kit* k : passes) {
        out.push_back(std::make_unique<RadCtx>());
        RadCtx& c = *out.back();
        c.b = &b;
        c.batch = &k->b;
        c.rank = rank;
        c.world = world;
        step(&c, &k->b);
    }
    return out;
}

std::vector<const RadCtx*> view(const std::vector<std::unique_ptr<RadCtx>>& v) {
    std::vector<const RadCtx*> p;
    for (const auto& c : v) p.push_back(c.get());
    return p;
}

/* The lint as a check: every finding is printed, and the step passes only with none. */
size_t report_lint(const std::vector<std::string>& f) {
    for (size_t i = 0; i < f.size() && i < 12; ++i) fprintf(stderr, "    lint: %s\n", f[i].c_str());
    if (f.size() > 12) fprintf(stderr, "    lint: ... %zu more\n", f.size() - 12);
    return f.size();
}

/* Every pass of one engine step ran without a step failure. */
bool steps_ok(const std::vector<std::unique_ptr<RadCtx>>& v) {
    bool ok = true;
    for (size_t i = 0; i < v.size(); ++i)
        if (!v[i]->step_fail.empty()) {
            fprintf(stderr, "    pass %zu failed: %s\n", i, v[i]->step_fail.c_str());
            ok = false;
        }
    return ok;
}

}  /* namespace rad_fake */

#define CHECK_LINT(b, ...) CHECK_EQ(report_lint(lint_step((b), { __VA_ARGS__ })), (size_t)0)
/* The passes of one engine step (run_passes): none failed and the lint is clean over them. */
#define CHECK_PASSES(b, v)                                                                   \
    do {                                                                                     \
        CHECK(steps_ok(v));                                                                  \
        CHECK_EQ(report_lint(lint_step((b), view(v))), (size_t)0);                           \
    } while (0)

/* ==================================================================== fingerprints */
namespace rad_fake {

/* A 64-bit FNV-1a fingerprint of everything a declare recorded: ops with their parameters,
 * weights, read and write sets and resolution; buffers, weights, KV groups, bindings, name maps,
 * the logits / pooled / drafter / generator declarations and the concurrent list. Two declares
 * give one value only if the core would build the same program from them. */
struct Fnv {
    uint64_t h = 1469598103934665603ull;
    Fnv& raw(const void* p, size_t n) {
        const auto* c = (const unsigned char*)p;
        for (size_t i = 0; i < n; ++i) { h ^= c[i]; h *= 1099511628211ull; }
        return *this;
    }
    Fnv& s(const std::string& x) { return raw(x.data(), x.size()).i((int64_t)x.size()); }
    Fnv& s(const char* x) { return s(std::string(x ? x : "")); }
    Fnv& i(int64_t v) { return raw(&v, sizeof v); }
    Fnv& f(double v) { return raw(&v, sizeof v); }
};
uint64_t graph_hash(const RadBuilder& b) {
    Fnv f;
    for (const RecOp& o : b.ops) {
        f.s(o.op).i(o.resolved).i((int64_t)o.p.size());
        for (const RecParam& p : o.p) f.s(p.key).i(p.kind).i(p.ival).i(p.ihi).f(p.dval).s(p.sval);
        for (const auto* v : { &o.w, &o.reads, &o.writes }) {
            f.i((int64_t)v->size());
            for (uint32_t x : *v) f.i(x);
        }
    }
    for (const RecBuf& x : b.bufs) {
        f.s(x.name).i(x.d.dtype).i(x.d.rank).i(x.d.kind).i(x.d.domain).i(x.d.kv).s(x.d.derive);
        for (uint32_t d = 0; d < x.d.rank; ++d) f.i(x.d.shape[d]);
    }
    for (const RecWeight& w : b.weights) {
        f.s(w.name).i(w.d.dtype).i(w.d.rank).i(w.d.shard).i(w.d.access).i(w.d.group.layer)
         .i(w.d.group.expert).s(w.d.source).s(w.d.planes).i(w.d.n_row_parts).i(w.span_lo)
         .i(w.span_hi).i(w.n_part_spans);
        for (uint32_t d = 0; d < w.d.rank; ++d) f.i(w.d.shape[d]);
    }
    for (const RecKV& g : b.kv_groups)
        f.s(g.name).i(g.d.kind).i(g.d.dtype).i(g.d.window).i(g.d.n_head_kv).i(g.d.head_dim)
         .i(g.d.state_dim[0]).i(g.d.state_dim[1]).i(g.d.conv_width).i(g.zero);
    for (const auto& [l, g] : b.binds) f.i(l).i(g);
    for (const RecMap& m : b.maps) {
        f.s(m.declared).i(m.mode).i(m.concat_dim);
        for (const auto& x : m.src) f.s(x);
    }
    f.i(b.logits).i(b.pooled).s(b.drafter_name).i(b.drafter.kind).i(b.drafter.depth)
     .i(b.drafter.block).i(b.drafter.proposal).i(b.drafter.mask_token).i(b.drafter_shared)
     .s(b.encoder_name).s(b.audio_encoder_name).s(b.generator_name);
    for (rad_buf x : b.concurrent) f.i(x);
    return f.h;
}

/* What a step issued, as a fingerprint: each launch's op (by name and declared parameters), row
 * count and lane, every operand by kind, buffer or weight name, offset and extent, a raw operand
 * by the batch array and element it points at, and every lane join. */
uint64_t issue_hash(const RadBuilder& b, const std::vector<std::unique_ptr<RadCtx>>& v,
                    const std::vector<const Kit*>& kits) {
    Fnv f;
    auto raw_at = [&](const void* p) {
        if (!p) { f.i(-1); return; }
        if (const rad_buf h = fake_buf_of(p, b.bufs.size())) {
            f.i(1).s(b.bufs[h - 1].name)
             .i((int64_t)((uintptr_t)p - (uintptr_t)rad_buf_ptr(nullptr, h)));
            return;
        }
        for (size_t k = 0; k < kits.size(); ++k) {
            const Kit& t = *kits[k];
            const std::vector<int32_t>* arrays[] = { &t.tok, &t.pos, &t.cu, &t.ctxl, &t.qlen,
                                                     &t.out, &t.nacc, &t.dout, &t.slot, &t.table,
                                                     &t.used, &t.sidx };
            for (size_t a = 0; a < sizeof arrays / sizeof arrays[0]; ++a) {
                const std::vector<int32_t>& x = *arrays[a];
                const int32_t* q = (const int32_t*)p;
                if (!x.empty() && q >= x.data() && q < x.data() + x.size()) {
                    f.i(2).i((int64_t)a).i((int64_t)(q - x.data()));
                    return;
                }
            }
        }
        f.i(3);
    };
    for (const auto& c : v) {
        f.i((int64_t)c->issues.size());
        for (const RecIssue& r : c->issues) {
            const RecOp* o = b.op_of(r.op);
            f.s(o ? o->op : "<null>");
            if (o)
                for (const RecParam& p : o->p) f.s(p.key).i(p.ival).i(p.ihi).f(p.dval).s(p.sval);
            f.i(r.n).i(r.lane).i((int64_t)r.opd.size());
            for (const RadOperand& d : r.opd) {
                f.i(d.kind).i(d.dtype).i(d.offset).i(d.rows).i(d.cols);
                if (d.kind == RAD_OPK_BUF)
                    f.s(d.handle && d.handle <= b.bufs.size() ? b.bufs[d.handle - 1].name : "?");
                else if (d.kind == RAD_OPK_WEIGHT || d.kind == RAD_OPK_WTAB)
                    f.s(d.handle && d.handle <= b.weights.size() ? b.weights[d.handle - 1].name
                                                                 : "?");
                else if (d.kind == RAD_OPK_RAW)
                    raw_at(d.raw);
                else
                    f.i(d.handle);
            }
        }
        for (const RecJoin& j : c->joins) f.i((int64_t)j.at).i(j.from).i(j.to);
    }
    return f.h;
}

/* The steps of the fingerprint: a 4096-row prefill after 4096 tokens of history (wider than a
 * MoE pass), a mixed step, a decode step, and with a drafter a verify and its first passes. */
uint64_t pinned_steps(RadBuilder& b, StepFn step, int rank, int world) {
    const auto pf = kit(b, {}, { 4096 }, 4096);
    const auto mx = kit(b, { 1, 1 }, { 2000 }, 3000);
    const auto dc = kit(b, { 1, 1, 1 }, {});
    std::vector<std::unique_ptr<RadCtx>> all;
    std::vector<const Kit*> kits;
    for (const Kit* k : { pf.get(), mx.get(), dc.get() }) {
        auto v = run_passes(b, step, { k }, rank, world);
        CHECK_PASSES(b, v);
        for (auto& c : v) all.push_back(std::move(c));
        kits.push_back(k);
    }
    std::unique_ptr<Kit> ver, his, r2;
    if (b.drafter.kind == RAD_DRAFT_SERIAL) {
        ver = kit(b, std::vector<int>(3, 4), {});
        his = history_pass(b, *ver);
        r2 = round_pass(b, 3, 2);
        auto v = run_passes(b, step, { ver.get(), his.get(), r2.get() }, rank, world);
        CHECK_PASSES(b, v);
        for (auto& c : v) all.push_back(std::move(c));
        kits.push_back(ver.get()); kits.push_back(his.get()); kits.push_back(r2.get());
    }
    return issue_hash(b, all, kits);
}

}  /* namespace rad_fake */

#endif /* RAD_ARCH_FAKES_H */
