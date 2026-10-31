/* nemotronh -- NVIDIA's Nemotron-H hybrids from the bf16 checkpoint: Nemotron-3.5-Bolt-74B-A9B
 * (`model_type: nemotron_h_omni` around a `nemotron_h_puzzle` language model,
 * NemotronH_Omni_Reasoning_V3), its language model; the routed experts as the recipe stores them.
 *
 *   88 blocks, each one mixer behind its own RMS norm and a residual add: 40 Mamba-2, 40 routed
 *   feed-forward and 8 attention, in the order `layers_block_type` gives
 *   Mamba-2: 128 heads of 64 in 8 groups, a 128-wide state, a 4-tap causal convolution with bias
 *   and SiLU over x|B|C, dt = softplus(dt + dt_bias), a D skip, and RMSNormGated over groups of
 *   1024 of y * silu(z)
 *   attention: 32 query heads and 2 KV heads of 128, no position encoding (NoPE)
 *   feed-forward: a latent MoE. The router scores the normed hidden state (sigmoid, a correction
 *   bias for the selection only, top-9, normalised, x5); the routed experts run in a 1024-wide
 *   latent (fc1_latent_proj in front, fc2_latent_proj behind) and are ungated,
 *   down(relu(up(x))^2). Count and width differ by layer (Puzzle's `block_configs`: 240 to 496
 *   experts of 1536 to 2560). A shared expert of 5376, relu^2 as well, reads the hidden state.
 *
 * ============================== THE LATENT FEED-FORWARD ==============================
 *
 * MoeFP8 serves the routed arm with its Geom at the latent width: its input is fc1's output
 * (quantised here), the router reads the hidden state over Config::router_k, and its gather writes
 * the routed sum back into the latent plane. fc2 has no bias and is linear, so under tensor
 * parallelism each rank applies it (replicated) to its partial latent sum and adds its slice of the
 * shared expert; one all-reduce over the hidden width then sums both arms (Config::no_reduce). fc1
 * is replicated too: the latent is every rank's input.
 *
 * ============================== THE MAMBA-2 MIXER ==============================
 *
 * in_proj's rows are z | x | B | C | dt (8192, 8192, 1024, 1024, 128). It is served as three linears
 * over row maps of the one tensor: z, x|B|C (the convolution's input) and dt. Under tensor
 * parallelism each rank takes whole heads and their groups' B and C, and dt's 128 rows split 32 a
 * rank at four, which is no whole 128-row scale block; dt is its own linear so a recipe can keep
 * it bf16 while the others are block fp8.
 *
 * The scan is `ssd_update`, Mamba-2's recurrence a token at a time (libref/ref_ssd.cpp), for
 * prefill and decode. dt is not clamped: the checkpoint's `time_step_limit` is (0, inf), as vLLM
 * serves it; HF's torch path clamps to time_step_min, which moves 13 % of the steps of the first
 * layers.
 *
 * RMSNormGated is three ops: act_mul (silu of z times y), a gainless rmsnorm over rows of 1024 (the
 * group), and mul_weight with the 8192 gains. A rank's heads are whole groups.
 *
 * ============================== RANKS, SPECULATION, VISION ==============================
 *
 * One, two or four ranks. At four each of the 2 KV heads is replicated over a rank pair (AttnFP8's
 * spans), the LiLiCorr drafter's 2 the same way (Dflash2's spans). Speculating, the scan keeps its
 * state in ssd_update's anchor form (one committed state and a scratch of per-token records), at
 * most 31 drafts a step; the convolution's window is ple_conv's rolling one. Pictures and video go
 * through the C-RADIO tower (radio.h) when the container holds it. A finite `time_step_limit` is
 * refused by name, and so is every other geometry the release does not have.
 * docs/ARCHITECTURES.md describes the model. */
#include <arch/rad_block_attn_fp8.h>
#include <arch/rad_block_moe_fp8.h>
#include <arch/rad_block_vit.h>
#include <arch/rad_block_dflash2.h>
#include <arch/rad_block_lilicorr.h>
#include "radio.h"
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <utility>

namespace nemotronh {
using namespace rad::arch;

enum Kind { kMamba, kMoe, kAttn };

/* The released geometry this graph is written for; a checkpoint that differs is refused by name. */
struct Shape {
    int64_t n_embd = 0, n_layer = 0, heads = 0, head_dim = 0, state = 0, groups = 0, conv = 0,
            latent = 0, ff_shared = 0;
    double route_scale = 1;
    std::vector<Kind> kind;
    std::vector<int64_t> experts, ff, top_k;   /* per layer; 0 off the routed layers */
};

static int64_t cfg(const RadModelMeta* m, const char* key, int64_t fallback) {
    return rad_meta_geti(m, key, fallback);
}

/* One all-reduce of [rows, n_embd] in place, or 0 at one rank. */
static rad_op reduce(RadBuilder* b, const Geom& g, rad_buf x, bool exact = false) {
    if (g.world <= 1) return 0;
    return rw(b, RAD_OP(b, "all_reduce",
                     RAD_PARAMS(RAD_INT("world_size", g.world),
                                RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                RAD_STR("dtype", g.dtype),
                                RAD_INT("exact", exact ? 1 : g.wire_exact),
                                RAD_INT("min_bytes", g.wire_min_bytes)),
                     RAD_NOWEIGHTS),
              { x }, { x });
}

struct Mamba {
    struct Wire {
        ActFP8 input{};       /* [T, n_embd]  the normed hidden state */
        rad_buf z = 0;        /* [T, W] */
        rad_buf xbc = 0;      /* [T, W + 2 GN]  the convolution's input */
        rad_buf conv = 0;     /* [T, W + 2 GN]  its output */
        rad_buf dt = 0;       /* [T, H] */
        rad_buf y = 0;        /* [T, W]  the scan's output */
        ActFP8 core_out{};    /* [T, W]  the gated norm's, the out-projection's input */
        rad_buf out = 0;      /* [T, n_embd] */
    };
    Geom g{};
    Wire w{};
    int layer = 0;
    /* This rank's heads, groups and widths. */
    int64_t H = 0, P = 0, N = 0, G = 0, W = 0, GN = 0;
    rad_kvgroup state = 0, history = 0;
    LinearFP8 in_z{}, in_xbc{}, in_dt{}, output{};
    QuantFP8 out_quant{};
    rad_weight taps = 0, bias = 0, alog = 0, dtb = 0, dskip = 0, gain = 0;
    rad_op op_conv = 0, op_prefill = 0, op_verify = 0, op_gate = 0, op_norm = 0, op_gain = 0,
           op_ar = 0;

    static constexpr int64_t kNormGroup = 1024;

    int declare(RadBuilder* b, Names& nm, const Geom& geom, const Shape& s, int l,
                const Wire& wire, const char* pre, rad_kvgroup st, rad_kvgroup cv) {
        g = geom;
        w = wire;
        layer = l;
        state = st;
        history = cv;
        H = s.heads / g.world;
        P = s.head_dim;
        N = s.state;
        G = s.groups / g.world;
        W = H * P;
        GN = G * N;
        const int64_t Wa = s.heads * s.head_dim, GNa = s.groups * s.state;
        const auto grp = grp_layer(l);
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, st));
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, cv));
        /* in_proj as three row maps: z, x|B|C, dt. */
        const char* src = nm.ckpt("%s.in_proj.weight", pre);
        RAD_ARCH_TRY(map_rows(b, nm.f("blk.%d.ssm_in_z.weight", l), { { src, 0, Wa } }));
        RAD_ARCH_TRY(map_rows(b, nm.f("blk.%d.ssm_in_xbc.weight", l),
                              { { src, (int)Wa, Wa + 2 * GNa } }));
        RAD_ARCH_TRY(map_rows(b, nm.f("blk.%d.ssm_dt.weight", l),
                              { { src, (int)(2 * Wa + 2 * GNa), s.heads } }));
        in_z.mapped = in_xbc.mapped = in_dt.mapped = true;
        RAD_ARCH_TRY(in_z.declare(b, nm, g, nm.f("blk.%d.ssm_in_z", l), W, g.n_embd, RAD_SHARD_ROW,
                                  grp, { src }, w.input, w.z));
        RAD_ARCH_TRY(in_xbc.declare(b, nm, g, nm.f("blk.%d.ssm_in_xbc", l), W + 2 * GN, g.n_embd,
                                    RAD_SHARD_ROW, grp, { src }, w.input, w.xbc, 0,
                                    { W, GN, GN }));
        RAD_ARCH_TRY(in_dt.declare(b, nm, g, nm.f("blk.%d.ssm_dt", l), H, g.n_embd, RAD_SHARD_ROW,
                                   grp, { src }, w.input, w.dt));

        const char *tn = nm.f("blk.%d.ssm_conv1d.weight", l), *bn = nm.f("blk.%d.ssm_conv1d.bias", l),
                   *an = nm.f("blk.%d.ssm_a_log", l), *dn = nm.f("blk.%d.ssm_dt.bias", l),
                   *sn = nm.f("blk.%d.ssm_d", l), *gn = nm.f("blk.%d.ssm_norm.weight", l);
        RAD_ARCH_TRY(map_copy(b, tn, nm.ckpt("%s.conv1d.weight", pre)));
        RAD_ARCH_TRY(map_copy(b, bn, nm.ckpt("%s.conv1d.bias", pre)));
        RAD_ARCH_TRY(map_copy(b, an, nm.ckpt("%s.A_log", pre)));
        RAD_ARCH_TRY(map_copy(b, dn, nm.ckpt("%s.dt_bias", pre)));
        RAD_ARCH_TRY(map_copy(b, sn, nm.ckpt("%s.D", pre)));
        RAD_ARCH_TRY(map_copy(b, gn, nm.ckpt("%s.norm.weight", pre)));
        taps = decl_w(b, tn, RAD_BF16, { W + 2 * GN, s.conv }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,
                      grp, 0, { W, GN, GN });
        bias = decl_w(b, bn, RAD_BF16, { W + 2 * GN }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp, 0,
                      { W, GN, GN });
        alog = decl_w(b, an, RAD_F32, { H }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp);
        dtb = decl_w(b, dn, RAD_F32, { H }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp);
        dskip = decl_w(b, sn, RAD_F32, { H }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp);
        gain = decl_w(b, gn, RAD_F32, { W }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp);
        if (!taps || !bias || !alog || !dtb || !dskip || !gain) return RAD_E_INVAL;

        op_conv = rw(b,
                     RAD_OP(b, "ple_conv_bias",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", W + 2 * GN),
                                       RAD_INT("width", s.conv), RAD_INT("dilation", 1),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(taps, bias)),
                     { w.xbc }, { w.conv });
        for (int verify = 0; verify < 2; ++verify) {
            const rad_op op =
                rw(b,
                   RAD_OP(b, "ssd_update",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("heads", H),
                                     RAD_INT("head_dim", P), RAD_INT("d_state", N),
                                     RAD_INT("groups", G),
                                     RAD_STR("mode", verify ? "verify" : "prefill"),
                                     RAD_STR("dtype", "bf16"),
                                     RAD_STR("state_form",
                                             g.max_spec > 0 ? "anchor" : "per_candidate")),
                          RAD_WEIGHTS(alog, dtb, dskip)),
                   { w.conv, w.dt }, { w.y });
            if (verify)
                op_verify = op;
            else
                op_prefill = op;
        }
        /* RMSNormGated: norm(y * silu(z)) * gain, normed over groups of 1024 (whole groups on a
         * rank; read as rows of 1024 of the contiguous plane). */
        op_gate = rw(b,
                     RAD_OP(b, "act_mul",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", W),
                                       RAD_STR("act", "silu"), RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                     { w.z, w.y }, { w.core_out.x });
        op_norm = rw(b,
                     RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * (W / kNormGroup)),
                                       RAD_INT("n", kNormGroup), RAD_F64("eps", g.eps),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                     { w.core_out.x }, { w.core_out.x });
        op_gain = rw(b,
                     RAD_OP(b, "mul_weight",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", W),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(gain)),
                     { w.core_out.x }, { w.core_out.x });
        if (!op_conv || !op_prefill || !op_verify || !op_gate || !op_norm || !op_gain)
            return RAD_E_INVAL;
        const char* on = nm.f("blk.%d.ssm_out", l);
        RAD_ARCH_TRY(output.probe(b, nm, g, on, { nm.ckpt("%s.out_proj.weight", pre) }));
        if (!output.reads_x) RAD_ARCH_TRY(out_quant.declare(b, g, w.core_out, W));
        RAD_ARCH_TRY(output.declare(b, nm, g, on, g.n_embd, W, RAD_SHARD_COL, grp,
                                    { nm.ckpt("%s.out_proj.weight", pre) }, w.core_out, w.out));
        op_ar = reduce(b, g, w.out);
        return g.world > 1 && !op_ar ? RAD_E_INVAL : RAD_OK;
    }

    /* The recurrent core over the decode rows (verify) and then the prefill rows, as KdaState. */
    void step(RadCtx* c, const RadBatch* batch) const {
        const auto *st = kv_batch(batch, state), *cv = kv_batch(batch, history);
        if (!st || !cv || !st->state_index || !cv->state_index) {
            rad_step_fail(c, "nemotronh Mamba-2 requires recurrent and convolution state indices");
            return;
        }
        const int64_t T = batch->n_tok, S = batch->n_seq;
        const int64_t pitch = st->state_index_pitch > 0 ? st->state_index_pitch : 1;
        in_z.step(c, w.input, w.z, T);
        in_xbc.step(c, w.input, w.xbc, T);
        in_dt.step(c, w.input, w.dt, T);
        int64_t D = 0, DT = 0;
        batch_split(batch, &D, &DT);
        for (int verify = 1; verify >= 0; --verify) {
            const int64_t seq0 = verify ? 0 : D, seqs = verify ? D : S - D, rows = verify ? DT : T;
            if (!seqs) continue;
            const RadOperand cu = praw(batch->cu_seqlens + seq0, RAD_I32, seqs + 1);
            const RadOperand init = praw(batch->ctx_lens + seq0, RAD_I32, seqs);
            const RadOperand accepted =
                verify ? praw(batch->num_accepted, RAD_I32, seqs) : RAD_NONE;
            RAD_ISSUE_N(c, op_conv, rows, brows(w.xbc, rows), RAD_W(taps),
                        kv_cache(history, layer), cu, RAD_NONE,
                        praw(cv->state_index + seq0, RAD_I32, seqs), init, accepted,
                        RAD_W(bias), brows(w.conv, rows));
            RAD_ISSUE_N(c, verify ? op_verify : op_prefill, rows, bcol(w.conv, 0, W, rows),
                        bcol(w.conv, W, GN, rows), bcol(w.conv, W + GN, GN, rows),
                        brows(w.dt, rows), RAD_W(alog), RAD_W(dtb), RAD_W(dskip),
                        kv_cache(state, layer),
                        praw2(st->state_index + seq0 * pitch, RAD_I32, seqs, pitch), cu, accepted,
                        init, brows(w.y, rows));
        }
        RAD_ISSUE_N(c, op_gate, T, brows(w.z, T), brows(w.y, T), brows(w.core_out.x, T));
        RAD_ISSUE_N(c, op_norm, T * (W / kNormGroup), brows(w.core_out.x, T), RAD_NONE,
                    brows(w.core_out.x, T));
        RAD_ISSUE_N(c, op_gain, T, brows(w.core_out.x, T), RAD_W(gain), brows(w.core_out.x, T));
        out_quant.step(c, T);
        output.step(c, w.core_out, w.out, T);
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.out, T), RAD_NONE);
    }
};

/* The latent feed-forward around MoeFP8 (see the header). */
struct LatentMoe {
    struct Wire {
        ActFP8 h{};           /* [T, n_embd]  the normed hidden state */
        ActFP8 lat{};         /* [T, latent]  fc1's output, then the routed sum */
        rad_buf routed = 0;   /* [T, n_embd]  fc2's output, then the block's delta */
        rad_buf sup = 0;      /* [T, ff_shared / world] */
        ActFP8 sff{};         /* [T, ff_shared / world] */
        rad_buf sout = 0;     /* [T, n_embd] */
        MoeFP8::Wire moe{};   /* the routed arm's own planes; its h and route are set here */
    };
    Geom g{};
    Wire w{};
    LinearFP8 fc1{}, fc2{}, sh_up{}, sh_down{};
    QuantFP8 lat_in{}, lat_out{};
    GateQuantFP8 sh_act{};
    MoeFP8 moe{};
    rad_op op_join = 0, op_ar = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& geom, const Shape& s, int l,
                const Wire& wire, const char* pre, MoeFP8::Src& ms, int64_t sh_lo, int64_t sh_hi,
                int64_t sh_w) {
        g = geom;
        w = wire;
        const auto grp = grp_layer(l);
        RAD_ARCH_TRY(fc1.declare(b, nm, g, nm.f("blk.%d.ffn_latent_in", l), s.latent, g.n_embd,
                                 RAD_SHARD_NONE, grp,
                                 { nm.ckpt("%s.fc1_latent_proj.weight", pre) }, w.h, w.lat.x));
        /* The routed arm reads the latent's codes unless its experts are bf16. */
        if (!ms.cfg.expert_bf16) RAD_ARCH_TRY(lat_in.declare(b, g, w.lat, s.latent));
        Geom ge = g;
        ge.n_embd = s.latent;
        ge.n_ff_all = s.ff[(size_t)l];
        MoeFP8::Wire mw = w.moe;
        mw.h = w.lat;
        mw.route = w.h.x;
        RAD_ARCH_TRY(moe.declare(b, nm, ge, l, mw, ms, false, false));
        const char* f2 = nm.f("blk.%d.ffn_latent_out", l);
        RAD_ARCH_TRY(fc2.probe(b, nm, g, f2, { nm.ckpt("%s.fc2_latent_proj.weight", pre) }));
        if (!fc2.reads_x) RAD_ARCH_TRY(lat_out.declare(b, g, w.lat, s.latent));
        RAD_ARCH_TRY(fc2.declare(b, nm, g, f2, g.n_embd, s.latent, RAD_SHARD_NONE, grp,
                                 { nm.ckpt("%s.fc2_latent_proj.weight", pre) }, w.lat, w.routed));
        /* The shared expert over the hidden state, this rank's slice of its 5376 columns. */
        if (sh_w > 0) {
            const char* dn = nm.f("blk.%d.ffn_down_shexp", l);
            RAD_ARCH_TRY(sh_down.probe(b, nm, g, dn,
                                       { nm.ckpt("%s.shared_experts.down_proj.weight", pre) }));
            RAD_ARCH_TRY(sh_up.declare(b, nm, g, nm.f("blk.%d.ffn_up_shexp", l), sh_w, g.n_embd,
                                       RAD_SHARD_ROW, grp,
                                       { nm.ckpt("%s.shared_experts.up_proj.weight", pre) }, w.h,
                                       w.sup, 0, {}, sh_lo, sh_hi));
            RAD_ARCH_TRY(sh_act.declare(b, g, w.sup, w.sff, sh_w, "relu2", 0, true, false));
            RAD_ARCH_TRY(sh_down.declare(b, nm, g, dn, g.n_embd, sh_w, RAD_SHARD_COL, grp,
                                         { nm.ckpt("%s.shared_experts.down_proj.weight", pre) },
                                         w.sff, w.sout, 0, {}, sh_lo, sh_hi));
            op_join = rw(b,
                         RAD_OP(b, "add",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                           RAD_STR("dtype", g.dtype)),
                                RAD_NOWEIGHTS),
                         { w.routed, w.sout }, { w.routed });
            if (!op_join) return RAD_E_INVAL;
        }
        op_ar = reduce(b, g, w.routed);
        return g.world > 1 && !op_ar ? RAD_E_INVAL : RAD_OK;
    }
    void step(RadCtx* c, const RadBatch* batch) const {
        const int64_t T = batch->n_tok;
        fc1.step(c, w.h, w.lat.x, T);
        lat_in.step(c, T);
        moe.step(c, batch);
        lat_out.step(c, T);
        fc2.step(c, w.lat, w.routed, T);
        if (op_join) {
            sh_up.step(c, w.h, w.sup, T);
            sh_act.step(c, w.sup, T);
            sh_down.step(c, w.sff, w.sout, T);
            RAD_ISSUE_N(c, op_join, T, brows(w.routed, T), brows(w.sout, T), brows(w.routed, T));
        }
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.routed, T), RAD_NONE);
    }
};

struct Layer {
    Kind kind = kMamba;
    rad_weight gain = 0;
    NormQuantFP8 norm{};
    Mamba mamba{};
    LatentMoe moe{};
    AttnFP8 attn{};
    rad_op op_add = 0;
};
struct Model {
    Names nm{};
    Geom g{};
    Shape s{};
    std::vector<Layer> layers;
    rad_kvgroup kv_attn = 0, kv_state = 0, kv_conv = 0;
    rad_buf x = 0, y = 0, final_hidden = 0, logit_input = 0, logits = 0;
    ActFP8 act{};
    Mamba::Wire mw{};
    AttnFP8::Wire aw{};
    LatentMoe::Wire ew{};
    /* The routed arm's planes by (experts, width): layers of one class share them. */
    std::map<std::pair<int64_t, int64_t>, MoeFP8::Wire> classes;
    rad_weight token = 0, head = 0, final_gain = 0;
    rad_op embed = 0, embed_ar = 0, norm = 0, gather = 0, head_op = 0;
    /* The vision encoder (C-RADIO and its projector) and the scatter of its rows over the token
     * embeddings they replace. */
    Radio radio{};
    MediaRows media{};
    /* LiLiCorr, the drafter (`--draft-model`): the trunk's residual stream after each tapped block
     * into one column slice of `dfsrc` a tap; the DFlash2 backbone in LiLiCorr's mode at absolute
     * layers n_layer.. of its own KV group, and the correlator after it. */
    enum { MAX_TAPS = 8 };
    int      n_taps = 0;
    int      tap_layer[MAX_TAPS] = {};
    rad_buf  dfsrc = 0;
    rad_op   op_tap = 0;
    bool     have_df = false;
    rad_kvgroup kv_draft = 0;
    Dflash2  df{};
    Dflash2::Wire dfw{};
    LiliCorr lili{};
    LiliCorr::Wire lw{};
};
static std::unique_ptr<Model> models[MAX_RANKS];
static thread_local std::unique_ptr<Model> probe_model;

static int read_shape(const RadModelMeta* meta, Shape& s) {
    s.n_embd = meta->n_embd;
    s.n_layer = meta->n_layers;
    s.heads = cfg(meta, "mamba_num_heads", 0);
    s.head_dim = cfg(meta, "mamba_head_dim", 0);
    s.state = cfg(meta, "ssm_state_size", 0);
    s.groups = cfg(meta, "n_groups", 0);
    s.conv = cfg(meta, "conv_kernel", 0);
    s.latent = cfg(meta, "moe_latent_size", 0);
    s.ff_shared = cfg(meta, "moe_shared_expert_intermediate_size", 0) *
                  cfg(meta, "n_shared_experts", 0);
    s.route_scale = rad_meta_getf(meta, "routed_scaling_factor", 0.0);
    /* `layers_block_type` names each block; `block_configs.<l>.*` (Puzzle) gives a routed layer's
     * expert count, width and top-k, the top-level keys where it is absent. */
    bool kinds_ok = true;
    for_each_token(rad_meta_gets(meta, "layers_block_type", ""), [&](const char* t, size_t n) {
        const std::string k(t, n);
        if (k == "mamba") s.kind.push_back(kMamba);
        else if (k == "moe") s.kind.push_back(kMoe);
        else if (k == "attention") s.kind.push_back(kAttn);
        else kinds_ok = false;
        return true;
    });
    s.experts.assign((size_t)s.n_layer, 0);
    s.ff.assign((size_t)s.n_layer, 0);
    s.top_k.assign((size_t)s.n_layer, 0);
    bool moe_ok = true;
    for (int64_t l = 0; l < (int64_t)s.kind.size() && l < s.n_layer; ++l) {
        if (s.kind[(size_t)l] != kMoe) continue;
        char key[96];
        auto per = [&](const char* field, const char* global) {
            std::snprintf(key, sizeof key, "block_configs.%lld.%s", (long long)l, field);
            return cfg(meta, key, cfg(meta, global, 0));
        };
        s.experts[(size_t)l] = per("n_routed_experts", "n_routed_experts");
        s.ff[(size_t)l] = per("moe_intermediate_size", "moe_intermediate_size");
        s.top_k[(size_t)l] = per("num_experts_per_tok", "num_experts_per_tok");
        moe_ok = moe_ok && s.experts[(size_t)l] > 0 && s.ff[(size_t)l] > 0 &&
                 s.top_k[(size_t)l] > 0 && s.top_k[(size_t)l] <= s.experts[(size_t)l] &&
                 s.experts[(size_t)l] <= 512 && s.ff[(size_t)l] % RAD_FP8_BLOCK == 0;
    }
    /* A clamp the scan does not apply: present only when the limit is a list of numbers. */
    bool unclamped = true;
    {
        std::vector<double> lim;
        for_each_token(rad_meta_gets(meta, "time_step_limit", ""), [&](const char* t, size_t) {
            lim.push_back(std::strtod(t, nullptr));
            return true;
        });
        unclamped = lim.empty() || (lim.size() == 2 && lim[0] <= 0 && std::isinf(lim[1]));
    }
    const char* act = rad_meta_gets(meta, "mlp_hidden_act", rad_meta_gets(meta, "hidden_act", ""));
    const struct {
        const char* what;
        bool ok;
    } need[] = {
        { "a block type per layer (mamba, moe, attention)",
          kinds_ok && (int64_t)s.kind.size() == s.n_layer },
        { "Mamba-2 heads of 64 in groups that divide them",
          s.head_dim == 64 && s.heads > 0 && s.groups > 0 && s.heads % s.groups == 0 },
        { "a Mamba-2 inner width of whole 1024-wide norm groups",
          (s.heads * s.head_dim) % (s.groups * Mamba::kNormGroup) == 0 ||
              (s.heads * s.head_dim) == s.groups * Mamba::kNormGroup },
        { "a state of 128", s.state == 128 },
        { "a 4-tap convolution with bias", s.conv == 4 && cfg(meta, "use_conv_bias", 0) != 0 },
        { "no in_proj or out_proj bias", cfg(meta, "mamba_proj_bias", 0) == 0 &&
                                             cfg(meta, "use_bias", 0) == 0 },
        { "an unclamped dt (time_step_limit (0, inf))", unclamped },
        { "a SiLU Mamba activation",
          !std::strcmp(rad_meta_gets(meta, "mamba_hidden_act", "silu"), "silu") },
        { "relu2 experts", !std::strcmp(act, "relu2") },
        { "routed layers with experts, width and top-k", moe_ok },
        { "a latent of whole 128-column blocks",
          s.latent > 0 && s.latent % RAD_FP8_BLOCK == 0 },
        { "one shared expert of whole 128-column blocks",
          cfg(meta, "n_shared_experts", 0) == 1 && s.ff_shared % RAD_FP8_BLOCK == 0 },
        { "sigmoid scoring",
          !std::strcmp(rad_meta_gets(meta, "scoring_func", ""), "sigmoid") },
        { "ungrouped routing", cfg(meta, "n_group", 1) == 1 && cfg(meta, "topk_group", 1) == 1 },
        { "a positive routed_scaling_factor", s.route_scale > 0 },
        { "no attention or MLP bias", cfg(meta, "attention_bias", 0) == 0 &&
                                          cfg(meta, "mlp_bias", 0) == 0 },
        { "no MTP layers", cfg(meta, "num_nextn_predict_layers", 0) == 0 },
        { "a residual stream in the activation dtype", cfg(meta, "residual_in_fp32", 0) == 0 },
        { "untied embeddings", cfg(meta, "tie_word_embeddings", 0) == 0 },
    };
    for (const auto& n : need)
        if (!n.ok) {
            fprintf(stderr, "radiance: nemotronh: this plugin serves %s, and the checkpoint "
                            "differs\n", n.what);
            return RAD_E_UNSUPPORTED;
        }
    return RAD_OK;
}

/* A rank's slice of a width of `blocks` 128-column blocks: blocks [lo, hi) in the order of the
 * ranks starting at `first`, the first `blocks % world` ranks of that order one block wider. */
static void slice_of(int64_t blocks, int world, int rank, int first, int64_t* lo, int64_t* hi) {
    const int p = ((rank - first) % world + world) % world;
    const int64_t base = blocks / world, extra = blocks % world;
    *lo = p * base + std::min<int64_t>(p, extra);
    *hi = *lo + base + (p < extra ? 1 : 0);
}

/* ------------------------------------------------------------------ declare */
static int buffers(RadBuilder* b, Model& m) {
    auto& g = m.g;
    auto& s = m.s;
    auto& nm = m.nm;
    const int64_t T = g.max_tok, E = g.n_embd;
    const int64_t H = s.heads / g.world, W = H * s.head_dim, GN = s.groups / g.world * s.state;
    auto bf = [&](const char* n, std::initializer_list<int64_t> shape) {
        return decl_b(b, nm.f("%s", n), RAD_BF16, shape);
    };
    m.x = bf("residual", { T, E });
    m.y = bf("mixer_delta", { T, E });
    m.act.x = bf("normed", { T, E });
    RAD_ARCH_TRY(m.act.declare_qs(b, nm, g, "normed", E));
    m.final_hidden = decl_b(b, nm.f("final_hidden"), RAD_BF16, { T, E }, RAD_BUF_PERSIST);
    m.logit_input = bf("logit_input", { g.max_logit_rows, E });
    m.logits = decl_b(b, nm.f("logits"), RAD_F32, { g.max_logit_rows, g.n_vocab });
    RAD_ARCH_TRY(rad_declare_logits(b, m.logits));

    auto& mw = m.mw;
    mw.input = m.act;
    mw.out = m.y;
    mw.z = bf("ssm_z", { T, W });
    mw.xbc = bf("ssm_xbc", { T, W + 2 * GN });
    mw.conv = bf("ssm_conv", { T, W + 2 * GN });
    mw.dt = bf("ssm_dt", { T, H });
    mw.y = bf("ssm_y", { T, W });
    mw.core_out.x = bf("ssm_out", { T, W });
    RAD_ARCH_TRY(mw.core_out.declare_qs(b, nm, g, "ssm_out", W));

    auto& aw = m.aw;
    aw.x = m.x;
    aw.h = m.act;
    aw.qkv = bf("attn_qkv", { T, g.qkv_dim() });
    aw.q = bf("attn_q", { T, g.q_dim() });
    aw.attn.x = bf("attn_out", { T, g.q_dim() });
    RAD_ARCH_TRY(aw.attn.declare_qs(b, nm, g, "attn_out", g.q_dim()));
    return RAD_OK;
}

/* The latent feed-forward's planes, and the routed arm's per (experts, width) class. */
static int moe_buffers(RadBuilder* b, Model& m, const std::vector<MoeFP8::Src>& msrc,
                       int64_t sh_w) {
    auto& g = m.g;
    auto& s = m.s;
    auto& nm = m.nm;
    const int64_t T = g.max_tok, E = g.n_embd, L = s.latent;
    auto bf = [&](const char* n, std::initializer_list<int64_t> shape) {
        return decl_b(b, nm.f("%s", n), RAD_BF16, shape);
    };
    auto& ew = m.ew;
    ew.h = m.act;
    ew.lat.x = bf("latent", { T, L });
    RAD_ARCH_TRY(ew.lat.declare_qs(b, nm, g, "latent", L));
    ew.routed = m.y;
    if (sh_w > 0) {
        ew.sup = bf("shared_up", { T, sh_w });
        ew.sff.x = bf("shared_ff", { T, sh_w });
        RAD_ARCH_TRY(ew.sff.declare_qs(b, nm, g, "shared_ff", sh_w));
        ew.sout = bf("shared_out", { T, E });
    }
    Geom gl = g;
    gl.n_embd = L;
    for (int64_t l = 0; l < s.n_layer; ++l) {
        if (s.kind[(size_t)l] != kMoe) continue;
        const MoeFP8::Config& c = msrc[(size_t)l].cfg;
        const auto key = std::make_pair(c.n_expert_all, c.n_ff_exp);
        if (m.classes.count(key)) continue;
        const int64_t F = c.n_ff_exp, k = c.top_k, rows = c.rows > 0 ? c.rows : T;
        char tag[64];
        std::snprintf(tag, sizeof tag, "e%lldf%lld", (long long)c.n_expert_all, (long long)F);
        MoeFP8::Wire w{};
        w.logits = decl_b(b, nm.f("%s.router_logits", tag), RAD_F32, { T, c.n_expert_all });
        w.ids = decl_b(b, nm.f("%s.expert_ids", tag), RAD_I32, { T, k });
        w.ew = decl_b(b, nm.f("%s.expert_weights", tag), RAD_F32, { T, k });
        w.eoff = decl_b(b, nm.f("%s.expert_offsets", tag), RAD_I32, { c.n_expert + 1 });
        w.ecnt = decl_b(b, nm.f("%s.expert_counts", tag), RAD_I32, { c.n_expert });
        w.sorted = decl_b(b, nm.f("%s.expert_sorted", tag), RAD_I32, { rows * k });
        w.egu = bf(nm.f("%s.expert_up", tag), { rows * k, F });
        w.eff.x = bf(nm.f("%s.expert_ff", tag), { rows * k, F });
        RAD_ARCH_TRY(w.eff.declare_qs(b, nm, gl, nm.f("%s.expert_ff", tag), F, rows * k));
        w.edn = bf(nm.f("%s.expert_down", tag), { rows * k, L });
        if (c.expert_rot) {
            w.hr.q = decl_b(b, nm.f("%s.latent_rot.q", tag), RAD_F8E4M3, { T, L });
            w.hr.s = decl_b(b, nm.f("%s.latent_rot.s", tag), RAD_F32, { T, L / RAD_FP8_BLOCK });
            if (!w.hr.q || !w.hr.s) return RAD_E_INVAL;
        }
        if (!w.logits || !w.ids || !w.ew || !w.eoff || !w.ecnt || !w.sorted || !w.egu ||
            !w.eff.x || !w.edn)
            return RAD_E_INVAL;
        m.classes[key] = w;
    }
    return RAD_OK;
}

/* LiLiCorr: the DFlash2 backbone with sinks (rad_block_dflash2.h, Config::lili), its KV group and
 * buffers, then the correlator (rad_block_lilicorr.h) over the backbone's draft rows, proposing a
 * block of `max_spec` tokens through the walk. The drafter's own config is the container's
 * `draft.` keys (rad-convert --draft-model); what this implementation does not apply is refused. */
static int declare_lilicorr(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                            Model& m, int base_layer) {
    Names& nm = m.nm;
    Geom   dg{};
    /* Two KV heads: at four ranks each is replicated over two (Dflash2 reads k and v at kv_off). */
    RAD_ARCH_TRY(drafter_geom(dg, m.g, meta, "LiLiCorr", true));
    auto dc_i = [&](const char* k, long long d) {
        char key[160];
        std::snprintf(key, sizeof key, "draft.dflash_config.%s", k);
        return rad_meta_geti(meta, key, d);
    };
    auto dc_f = [&](const char* k, double d) {
        char key[160];
        std::snprintf(key, sizeof key, "draft.dflash_config.%s", k);
        return rad_meta_getf(meta, key, d);
    };
    const long long trained = rad_meta_geti(meta, "draft.block_size", dc_i("block_size", 0));
    Dflash2::Config dc{};
    dc.lili   = 1;
    dc.n_taps = m.n_taps;
    dc.block  = ctx->max_spec + 1;
    dc.taps   = dc_i("conv_kernel_size", 2);
    dc.group  = dc_i("conv_group_size", 16);
    dc.causal = dc_i("causal", 0) ? 1 : 0;
    if (dc_i("use_swa", 0))
        dc.window = dc_i("swa_window_size", rad_meta_geti(meta, "draft.sliding_window", 0));
    /* The correlator has a slot embedding a draft position, trained at one block: run at it. */
    if (trained <= 1 || dc.block != trained) {
        fprintf(stderr, "radiance: nemotronh: LiLiCorr was trained at block_size %lld (%lld draft "
                        "tokens) and its correlator has a learned embedding for each of them; this "
                        "serve asks for %d draft tokens\n",
                trained, trained - 1, ctx->max_spec);
        return RAD_E_UNSUPPORTED;
    }
    if (!dc_i("lilicorr_enabled", 0) || !rad_meta_geti(meta, "draft.attention_sink_bias",
                                                         dc_i("attention_sink_bias", 0))) {
        fprintf(stderr, "radiance: nemotronh: the drafter's dflash_config is not LiLiCorr's "
                        "(lilicorr_enabled and attention_sink_bias)\n");
        return RAD_E_UNSUPPORTED;
    }
    const double escale = dc_f("input_embedding_scale", 1.0);
    if (escale != 1.0) {
        fprintf(stderr, "radiance: nemotronh: LiLiCorr scaling its input embedding by %g is not "
                        "implemented\n", escale);
        return RAD_E_UNSUPPORTED;
    }
    LiliCorr::Config lc{};
    lc.steps   = dc.steps();
    lc.k       = dc_i("lilicorr_candidate_topk", 8);
    lc.hs      = dc_i("lilicorr_hidden_size", 1024);
    lc.n_head  = dc_i("lilicorr_num_heads", 8);
    lc.n_layer = dc_i("lilicorr_num_layers", 2);
    lc.mlp     = (int64_t)std::llround((double)lc.hs * dc_f("lilicorr_mlp_ratio", 2.0));
    lc.fdim    = dc_i("lilicorr_factor_dim", 1024);
    lc.eps     = dg.eps;
    lc.vec_eps = dc_f("lilicorr_vector_eps", 1e-4);
    lc.scale   = dc_f("lilicorr_logit_scale", 8.0);
    if (lc.k < 2 || lc.k > 64 || (lc.k & (lc.k - 1)) || lc.steps * lc.k > 64 ||
        lc.hs % lc.n_head || lc.hs / lc.n_head > 128) {
        fprintf(stderr, "radiance: nemotronh: LiLiCorr's lattice (%lld positions of %lld, %lld "
                        "heads over %lld) is outside what lili_attn serves\n",
                (long long)lc.steps, (long long)lc.k, (long long)lc.n_head, (long long)lc.hs);
        return RAD_E_UNSUPPORTED;
    }

    RAD_ARCH_TRY(drafter_kv_group(b, nm, ctx, dg, dc.window, base_layer, &m.kv_draft));

    const int64_t qd  = dg.n_head * dg.head_dim;
    const int64_t kvd = dg.n_head_kv * dg.head_dim;
    int64_t qr = dg.max_seqs * dc.block;
    if (qr > dg.max_tok) qr = dg.max_tok;
    const int64_t sr = dg.max_seqs * dc.steps();
    const int64_t lr = sr * lc.k;
    const int64_t E  = dg.n_embd;
    const uint32_t A = dg.act_dtype;

    Dflash2::Wire& dw = m.dfw;
    dw.src     = m.dfsrc;
    /* PERSIST: the context pass writes it, the next query pass reads the anchor's row from it,
     * so no transient may share it in between. */
    dw.c       = decl_b(b, nm.f("df_c"),    A, {dg.max_tok, E}, RAD_BUF_PERSIST);
    dw.ck      = decl_b(b, nm.f("df_ck"),   A, {dg.max_tok, kvd});
    dw.cv      = decl_b(b, nm.f("df_cv"),   A, {dg.max_tok, kvd});
    dw.x       = decl_b(b, nm.f("df_x"),    A, {qr, E});
    dw.h       = decl_b(b, nm.f("df_h"),    A, {qr, E});
    dw.h2      = decl_b(b, nm.f("df_h2"),   A, {qr, E});
    dw.coef    = decl_b(b, nm.f("df_coef"), A, {qr, dc.coef_dim(E)});
    dw.q       = decl_b(b, nm.f("df_q"),    A, {qr, qd});
    dw.k       = decl_b(b, nm.f("df_k"),    A, {qr, kvd});
    dw.v       = decl_b(b, nm.f("df_v"),    A, {qr, kvd});
    dw.attn    = decl_b(b, nm.f("df_attn"), A, {qr, qd});
    dw.gate_up = decl_b(b, nm.f("df_gate_up"), A, {qr, 2 * dg.n_ff});
    dw.ffn     = decl_b(b, nm.f("df_ffn"),  A, {qr, dg.n_ff});
    dw.srcq    = decl_b(b, nm.f("df_srcq"), RAD_F8E4M3, {dg.max_tok, dc.n_taps * E});
    dw.srcs    = decl_b(b, nm.f("df_srcs"), RAD_F32, {dg.max_tok, fp8_blocks(dc.n_taps * E)});
    dw.cq      = decl_b(b, nm.f("df_cq"),   RAD_F8E4M3, {dg.max_tok, E});
    dw.cs      = decl_b(b, nm.f("df_cs"),   RAD_F32,    {dg.max_tok, fp8_blocks(E)});
    dw.h2q     = decl_b(b, nm.f("df_h2q"),  RAD_F8E4M3, {qr, E});
    dw.h2s     = decl_b(b, nm.f("df_h2s"),  RAD_F32,    {qr, fp8_blocks(E)});
    dw.atq     = decl_b(b, nm.f("df_atq"),  RAD_F8E4M3, {qr, qd});
    dw.ats     = decl_b(b, nm.f("df_ats"),  RAD_F32,    {qr, fp8_blocks(qd)});
    dw.ffq     = decl_b(b, nm.f("df_ffq"),  RAD_F8E4M3, {qr, dg.n_ff});
    dw.ffs     = decl_b(b, nm.f("df_ffs"),  RAD_F32,    {qr, fp8_blocks(dg.n_ff)});
    dw.sel_h   = decl_b(b, nm.f("df_selh"), A, {sr, E});
    /* The proposal buffer (RadDrafterDecl::proposal), written by the walk. */
    dw.tokens  = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {dg.max_seqs, dc.steps()});
    RAD_ARCH_TRY(m.df.declare(b, nm, dg, dc, base_layer, m.kv_draft, m.token, "draft",
                              nm.ckpt("language_model.lm_head.weight"), dw));

    LiliCorr::Wire& lw = m.lw;
    const int64_t sw = 2 + 2 * LiliCorr::row_k(lc.k);
    lw.sel_h  = dw.sel_h;
    lw.c      = dw.c;
    lw.tokens = dw.tokens;
    lw.logits = decl_b(b, nm.f("lili_logits"), RAD_F32, {sr, dg.n_vocab});
    lw.stats  = decl_b(b, nm.f("lili_stats"), RAD_F32, {sr, sw});
    if (dg.world > 1)
        lw.sgath = decl_b(b, nm.f("lili_sgath"), RAD_F32, {sr, (int64_t)dg.world * sw});
    lw.pas    = decl_b(b, nm.f("lili_pas"),   A, {sr, lc.hs});
    lw.cand   = decl_b(b, nm.f("lili_cand"),  RAD_I32, {lr});
    lw.feat   = decl_b(b, nm.f("lili_feat"),  A, {lr, lc.hs});
    lw.hh     = decl_b(b, nm.f("lili_hh"),    A, {lr, lc.hs});
    lw.emb    = decl_b(b, nm.f("lili_emb"),   A, {lr, E});
    lw.y      = decl_b(b, nm.f("lili_y"),     A, {lr, lc.hs});
    lw.qkv    = decl_b(b, nm.f("lili_qkv"),   A, {lr, 3 * lc.hs});
    lw.att    = decl_b(b, nm.f("lili_att"),   A, {lr, lc.hs});
    lw.up     = decl_b(b, nm.f("lili_up"),    A, {lr, lc.mlp});
    lw.ups    = decl_b(b, nm.f("lili_ups"),   A, {lr, lc.mlp});
    lw.cat    = decl_b(b, nm.f("lili_cat"),   A, {lr, 3 * lc.hs});
    lw.fh     = decl_b(b, nm.f("lili_fh"),    A, {lr, lc.hs});
    lw.fhs    = decl_b(b, nm.f("lili_fhs"),   A, {lr, lc.hs});
    lw.edges  = decl_b(b, nm.f("lili_edges"), A, {lr, 2 * lc.fdim});
    lw.anc    = decl_b(b, nm.f("lili_anc"),   A, {dg.max_seqs, E});
    lw.ast    = decl_b(b, nm.f("lili_ast"),   A, {dg.max_seqs, lc.hs});
    lw.an     = decl_b(b, nm.f("lili_an"),    A, {dg.max_seqs, lc.hs});
    lw.aout   = decl_b(b, nm.f("lili_aout"),  A, {dg.max_seqs, lc.fdim});
    RAD_ARCH_TRY(m.lili.declare(b, nm, dg, lc, m.head, m.token, "draft", lw));
    m.have_df = true;

    RadDrafterDecl drf{};
    drf.name           = "lilicorr";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    drf.block          = dc.block;
    drf.proposal       = dw.tokens;
    drf.proposal_pitch = dc.steps();
    drf.window         = dc.window;
    drf.mask_token     = (int32_t)dc_i("mask_token_id", -1);
    if (drf.mask_token < 0) {
        fprintf(stderr, "radiance: nemotronh: the LiLiCorr checkpoint declares no "
                        "dflash_config.mask_token_id\n");
        return RAD_E_FORMAT;
    }
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));
    rad_note(b, "LiLiCorr drafter: %lld layers at absolute %d..%d, %lld heads / %lld kv%s x %lld "
                "with sinks, ffn %lld, %s attention with a %lld-token window, %d trunk taps; a "
                "%lld x %lld candidate lattice proposes %lld tokens",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1,
             (long long)dg.n_head, (long long)dg.n_head_kv,
             dg.world > dg.n_head_kv_all ? " (replicated)" : "", (long long)dg.head_dim,
             (long long)dg.n_ff, dc.causal ? "causal" : "non-causal", (long long)dc.window,
             m.n_taps, (long long)lc.steps, (long long)lc.k, (long long)dc.steps());
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->is_draft || ctx->rank < 0 || ctx->rank >= MAX_RANKS)
        return RAD_E_INVAL;
    const int world = ctx->world_size ? ctx->world_size : 1;
    auto& slot = ctx->shape_probe ? probe_model : models[ctx->rank];
    slot = std::make_unique<Model>();
    auto& m = *slot;
    auto& g = m.g;
    auto& s = m.s;
    m.nm = Names(ctx->scope);
    auto& nm = m.nm;
    RAD_ARCH_TRY(read_shape(meta, s));
    if (world != 1 && world != 2 && world != 4) {
        fprintf(stderr, "radiance: nemotronh: --tp %d; the head and expert splits are built for 1, "
                        "2 or 4 ranks\n", world);
        return RAD_E_UNSUPPORTED;
    }
    if (s.heads % world || s.groups % world || (s.heads / world * s.head_dim) % Mamba::kNormGroup) {
        fprintf(stderr, "radiance: nemotronh: %lld Mamba heads in %lld groups do not split over %d "
                        "ranks in whole groups of the gated norm\n",
                (long long)s.heads, (long long)s.groups, world);
        return RAD_E_UNSUPPORTED;
    }
    if (ctx->max_spec > 31) {
        fprintf(stderr, "radiance: nemotronh: the Mamba-2 verify pass keeps at most 32 records a "
                        "sequence, so --num-speculative-tokens is at most 31\n");
        return RAD_E_UNSUPPORTED;
    }
    RAD_ARCH_TRY(geom_from(g, meta, ctx, "bf16", RAD_BF16));

    /* Each routed layer's experts: its own count and width; the slice of each expert this rank
     * holds, even, or by slice_of at four ranks (2, 2, 1, 1 of six blocks, by number mod 4). */
    std::vector<MoeFP8::Src> msrc((size_t)s.n_layer);
    bool calib = calib_dir() != nullptr, any_bf16 = false;
    for (int l = 0; l < (int)s.n_layer; ++l) {
        if (s.kind[(size_t)l] != kMoe) continue;
        const char* pre = nm.ckpt("language_model.backbone.layers.%d.mixer", l);
        MoeFP8::Src& ms = msrc[(size_t)l];
        MoeFP8::Config& c = ms.cfg;
        const int64_t ne = s.experts[(size_t)l], ff = s.ff[(size_t)l];
        c.n_expert = c.n_expert_all = ne;
        c.top_k = s.top_k[(size_t)l];
        c.sigmoid_router = 1;
        c.route_scale = s.route_scale;
        c.norm_topk = cfg(meta, "norm_topk_prob", 1) != 0;
        c.ungated = 1;
        c.router_k = g.n_embd;
        c.no_reduce = 1;
        c.rows = g.max_tok;
        /* RADIANCE_CALIB_LAYERS splits the calibration over runs (calib_layer_on). */
        c.calib = calib && calib_layer_on(l) ? 1 : 0;
        c.probed_expert = (int64_t)g.rank * (ne / world);
        const int64_t eb = ff / RAD_FP8_BLOCK;
        if (eb % world == 0) {
            c.n_ff_exp = ff / world;
        } else if (world == 4) {
            int64_t widest = 0;
            for (int q = 0; q < 4; ++q) {
                int64_t lo, hi;
                slice_of(eb, world, g.rank, q, &lo, &hi);
                c.ff_lo4[q] = lo * RAD_FP8_BLOCK;
                c.ff_hi4[q] = hi * RAD_FP8_BLOCK;
                widest = std::max(widest, c.ff_hi4[q] - c.ff_lo4[q]);
            }
            c.n_ff_exp = widest;
        } else {
            fprintf(stderr, "radiance: nemotronh: layer %d's experts of %lld do not split over %d "
                            "ranks in 128-column blocks\n", l, (long long)ff, world);
            return RAD_E_UNSUPPORTED;
        }
        ms.router = nm.ckpt("%s.gate.weight", pre);
        ms.router_bias = nm.ckpt("%s.gate.e_score_correction_bias", pre);
        ms.experts = nm.ckpt("%s.experts", pre);
        RAD_ARCH_TRY(MoeFP8::probe(b, nm, l, ms, &c));
        any_bf16 = any_bf16 || c.expert_bf16;
    }
    /* Whole experts per rank for a calibration run and for bf16 experts: rank r owns experts
     * [r E/W, (r+1) E/W) of each layer. */
    if (world > 1 && (calib || any_bf16))
        for (int l = 0; l < (int)s.n_layer; ++l) {
            if (s.kind[(size_t)l] != kMoe) continue;
            auto& c = msrc[(size_t)l].cfg;
            int64_t per = 0;
            RAD_ARCH_TRY(divide_or_fail(c.n_expert_all, world, "n_routed_experts", &per));
            c.n_expert = per;
            c.expert_base = (int64_t)g.rank * per;
            c.n_ff_exp = s.ff[(size_t)l];
            for (int q = 0; q < 4; ++q) c.ff_lo4[q] = c.ff_hi4[q] = 0;
        }
    /* The shared expert's slice: even, or slice_of's whole blocks. */
    int64_t sh_lo = 0, sh_hi = 0, sh_w = s.ff_shared / world;
    {
        const int64_t sb = s.ff_shared / RAD_FP8_BLOCK;
        if (sb % world) {
            slice_of(sb, world, g.rank, 0, &sh_lo, &sh_hi);
            sh_lo *= RAD_FP8_BLOCK;
            sh_hi *= RAD_FP8_BLOCK;
            sh_w = sh_hi - sh_lo;
        }
    }
    RAD_ARCH_TRY(buffers(b, m));
    RAD_ARCH_TRY(moe_buffers(b, m, msrc, sh_w));
    {
        RadKVGroupDecl d{};
        d.kind = RAD_KV_FULL;
        d.dtype = rad_kv_cache_dtype(ctx, RAD_BF16);
        d.n_head_kv = g.n_head_kv;
        d.head_dim = g.head_dim;
        m.kv_attn = rad_decl_kv_group(b, nm.f("kv_attn"), &d);
        g.kv_dtype = rad_dtype_name(d.dtype);
        d = {};
        d.kind = RAD_KV_LINEAR;
        d.dtype = RAD_F32;
        d.n_head_kv = s.heads / world;
        d.state_dim[0] = s.head_dim;
        d.state_dim[1] = s.state;
        m.kv_state = rad_decl_kv_group(b, nm.f("kv_ssm"), &d);
        d = {};
        d.kind = RAD_KV_CONV;
        d.dtype = RAD_BF16;
        d.n_head_kv = 1;
        d.head_dim = (s.heads * s.head_dim + 2 * s.groups * s.state) / world;
        d.conv_width = s.conv;
        m.kv_conv = rad_decl_kv_group(b, nm.f("kv_ssm_conv"), &d);
        if (!m.kv_attn || !m.kv_state || !m.kv_conv) return RAD_E_INVAL;
    }

    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"),
                          nm.ckpt("language_model.backbone.embeddings.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output.weight"), nm.ckpt("language_model.lm_head.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"),
                          nm.ckpt("language_model.backbone.norm_f.weight")));
    m.token = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, { g.n_vocab, g.n_embd },
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.head = decl_w(b, nm.f("output.weight"), RAD_BF16, { g.n_vocab, g.n_embd }, RAD_ACCESS_VOCAB,
                    RAD_SHARD_ROW, grp_model());
    m.final_gain = decl_w(b, nm.f("output_norm.weight"), RAD_F32, { g.n_embd },
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.embed = rw(b,
                 RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16"),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.token)),
                 {}, { m.x });
    /* Exact: each token's embedding comes from one rank and the others add zero. */
    m.embed_ar = reduce(b, g, m.x, true);
    /* Images and video: the tower's rows replace their placeholders' embeddings before that
     * reduction (rank 0 carries them, the others zeros). Its weight groups follow the layers'. */
    RAD_ARCH_TRY(m.radio.declare(b, meta, ctx, nm, g.n_embd, (int)s.n_layer));
    if (m.radio.on) RAD_ARCH_TRY(m.media.declare(b, g.max_tok, g.n_embd, m.x));

    /* LiLiCorr's taps, only when the container carries the drafter and the deployment speculates:
     * the residual stream after each block `target_layer_ids` names (vLLM's aux layer id + 1, the
     * output of that block), copied into its column slice of `dfsrc` by `cast`. */
    if (ctx->max_spec > 0) {
        const char* da = rad_meta_gets(meta, "draft.architectures", "");
        if (da && std::strstr(da, "LiLiCorr")) {
            m.n_taps = parse_layer_ids(
                rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", nullptr), s.n_layer,
                m.tap_layer, Model::MAX_TAPS);
            if (m.n_taps <= 0) {
                fprintf(stderr, "radiance: nemotronh: the LiLiCorr drafter names no trunk layer "
                                "this model has (dflash_config.target_layer_ids)\n");
                return RAD_E_UNSUPPORTED;
            }
            m.dfsrc = decl_b(b, nm.f("dflash_src"), RAD_BF16,
                             { g.max_tok, (int64_t)m.n_taps * g.n_embd });
            m.op_tap = rw(b, RAD_OP(b, "cast",
                                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                               RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                                    RAD_NOWEIGHTS),
                          { m.x }, { m.dfsrc });
            if (!m.dfsrc || !m.op_tap) return RAD_E_INVAL;
        }
    }

    m.layers.resize((size_t)s.n_layer);
    for (int l = 0; l < (int)s.n_layer; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.kind = s.kind[(size_t)l];
        const char* pre = nm.ckpt("language_model.backbone.layers.%d", l);
        const char* mx = nm.ckpt("%s.mixer", pre);
        if (lay.kind == kAttn) {
            AttnFP8::Src as{};
            as.norm = nm.ckpt("%s.norm.weight", pre);
            as.q = nm.ckpt("%s.q_proj.weight", mx);
            as.k = nm.ckpt("%s.k_proj.weight", mx);
            as.v = nm.ckpt("%s.v_proj.weight", mx);
            as.o = nm.ckpt("%s.o_proj.weight", mx);
            RAD_ARCH_TRY(rad_bind_layer_kv(b, l, m.kv_attn));
            RAD_ARCH_TRY(lay.attn.declare(b, nm, g, l, m.kv_attn, 0, m.aw, as));
            continue;
        }
        const char* nn = nm.f(lay.kind == kMoe ? "blk.%d.ffn_norm.weight" : "blk.%d.attn_norm.weight",
                              l);
        RAD_ARCH_TRY(map_copy(b, nn, nm.ckpt("%s.norm.weight", pre)));
        lay.gain = decl_w(b, nn, RAD_F32, { g.n_embd }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                          grp_layer(l));
        RAD_ARCH_TRY(lay.norm.declare(b, g, lay.gain, m.x, m.act, g.n_embd));
        if (lay.kind == kMamba) {
            RAD_ARCH_TRY(lay.mamba.declare(b, nm, g, s, l, m.mw, mx, m.kv_state, m.kv_conv));
        } else {
            LatentMoe::Wire ew = m.ew;
            const MoeFP8::Config& c = msrc[(size_t)l].cfg;
            ew.moe = m.classes[std::make_pair(c.n_expert_all, c.n_ff_exp)];
            RAD_ARCH_TRY(lay.moe.declare(b, nm, g, s, l, ew, mx, msrc[(size_t)l], sh_lo, sh_hi,
                                         sh_w));
        }
        lay.op_add = rw(b,
                        RAD_OP(b, "add",
                               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                        { m.x, m.y }, { m.x });
        if (!lay.op_add) return RAD_E_INVAL;
    }
    m.norm = rw(b,
                RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                  RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                       RAD_WEIGHTS(m.final_gain)),
                { m.x }, { m.final_hidden });
    m.gather = rw(b,
                  RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_embd),
                                    RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                  { m.final_hidden }, { m.logit_input });
    m.head_op =
        rw(b,
           RAD_OP(b, "logits_gemm",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                             RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", "bf16")),
                  RAD_WEIGHTS(m.head)),
           { m.logit_input }, { m.logits });
    if (!m.embed || !m.norm || !m.gather || !m.head_op) return RAD_E_INVAL;
    if (m.n_taps > 0) RAD_ARCH_TRY(declare_lilicorr(b, meta, ctx, m, (int)s.n_layer));
    int64_t n_m = 0, n_e = 0, n_a = 0;
    for (Kind k : s.kind) (k == kMamba ? n_m : k == kMoe ? n_e : n_a) += 1;
    rad_note(b,
             "Nemotron-H (nemotronh): TP%d, %lld blocks: %lld Mamba-2, %lld latent MoE, %lld NoPE "
             "attention; experts %s",
             world, (long long)s.n_layer, (long long)n_m, (long long)n_e, (long long)n_a,
             world > 1 && (calib || any_bf16) ? "whole per rank" : "sliced per rank");
    return RAD_OK;
}

/* ------------------------------------------------------------------ step */
/* RADIANCE_DEBUG_NEMOTRONH_DUMP=DIR: the residual stream after each block, every row of every
 * step, as `<DIR>/s<step>.t<rows>.l<layer>.r<rank>.bf16` ([rows, n_embd] bf16). A diagnostic for
 * comparing against a reference model block by block; it synchronises the stream. */
static void dump(RadCtx* c, const RadBatch* batch, rad_buf x, int64_t E, int layer) {
    static const char* dir = std::getenv("RADIANCE_DEBUG_NEMOTRONH_DUMP");
    if (!dir || !*dir) return;
    const int64_t T = batch->n_tok;
    void* p = rad_buf_ptr(c, x);
    if (!p) return;
    std::vector<uint16_t> h((size_t)(T * E));
    if (rad_memcpy_async(h.data(), p, T * E * 2, rad_stream(c)) < 0) return;
    rad_stream_sync(rad_stream(c));
    char path[512];
    std::snprintf(path, sizeof path, "%s/s%d.t%lld.l%d.r%d.bf16", dir, batch->step, (long long)T,
                  layer, rad_rank(c));
    if (FILE* f = std::fopen(path, "wb")) {
        std::fwrite(h.data(), 2, h.size(), f);
        std::fclose(f);
    }
}

/* The drafter's side of the same diagnostic, `<DIR>/d<step>.<what>.r<rank>.<dtype>`: on its query
 * pass the batch's tokens (the anchor, then the masks) and anchor rows, the context pass's normed
 * fc rows `c` (still the last context pass's), the draft rows `selh`, the candidates and the
 * proposal. With the trunk step's per-block dumps (the taps) a reference recomputes the pass. */
static void dump_draft(RadCtx* c, const RadBatch* batch, const char* what, const void* p,
                       int64_t bytes, const char* dt) {
    static const char* dir = std::getenv("RADIANCE_DEBUG_NEMOTRONH_DUMP");
    if (!dir || !*dir || !p || bytes <= 0) return;
    std::vector<uint8_t> h((size_t)bytes);
    if (rad_memcpy_async(h.data(), p, bytes, rad_stream(c)) < 0) return;
    rad_stream_sync(rad_stream(c));
    char path[512];
    std::snprintf(path, sizeof path, "%s/d%d.%s.r%d.%s", dir, batch->step, what, rad_rank(c), dt);
    if (FILE* f = std::fopen(path, "wb")) {
        std::fwrite(h.data(), 1, h.size(), f);
        std::fclose(f);
    }
}

static void step(RadCtx* c, const RadBatch* batch) {
    auto& m = *models[rad_rank(c)];
    const int64_t T = batch->n_tok, E = m.g.n_embd;
    if (batch->enc) {
        m.radio.step(c, batch);
        return;
    }
    /* A drafter pass runs only the drafter: positive is LiLiCorr's query pass (the backbone, then
     * the correlator over its draft rows), negative its context pass over the rows the trunk step
     * just ran, whose taps are in `dfsrc`. */
    if (batch->draft_pass) {
        if (!m.have_df) {
            rad_step_fail(c, "nemotronh: a drafter pass, and no drafter is declared (the container "
                             "carries none, or the deployment does not speculate)");
            return;
        }
        if (batch->draft_pass > 0) {
            const int64_t n = batch->n_seq, S = n * m.lili.cfg.steps, E = m.g.n_embd;
            m.df.step(c, batch);
            /* Before the correlator: `selh` is dead after its last read there, and the arena may
             * hand its bytes to a later op (at four ranks it does). */
            dump_draft(c, batch, "selh", rad_buf_ptr(c, m.dfw.sel_h), S * E * 2, "bf16");
            m.lili.step(c, batch);
            dump_draft(c, batch, "tok", batch->token_ids, batch->n_tok * 4, "i32");
            dump_draft(c, batch, "arow", batch->anchor_rows, n * 4, "i32");
            dump_draft(c, batch, "c", rad_buf_ptr(c, m.dfw.c), m.g.max_tok * E * 2, "bf16");
            dump_draft(c, batch, "cand", rad_buf_ptr(c, m.lw.cand), S * m.lili.cfg.k * 4, "i32");
            dump_draft(c, batch, "tokens", rad_buf_ptr(c, m.dfw.tokens), S * 4, "i32");
        } else {
            m.df.context(c, batch);
        }
        return;
    }
    RAD_ISSUE_N(c, m.embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.token), brows(m.x, T));
    m.media.step(c, batch, m.x, T);
    if (m.embed_ar) RAD_ISSUE_N(c, m.embed_ar, T * E, brows(m.x, T), RAD_NONE);
    for (const auto& l : m.layers) {
        if (l.kind == kAttn) {
            l.attn.step(c, batch);
        } else {
            l.norm.step(c, T);
            if (l.kind == kMamba)
                l.mamba.step(c, batch);
            else
                l.moe.step(c, batch);
            RAD_ISSUE_N(c, l.op_add, T, brows(m.x, T), brows(m.y, T), brows(m.x, T));
        }
        const int li = (int)(&l - m.layers.data());
        dump(c, batch, m.x, E, li);
        /* The drafter's taps: this block's output, which the next block overwrites. */
        for (int t = 0; t < m.n_taps; ++t)
            if (m.tap_layer[t] == li)
                RAD_ISSUE(c, m.op_tap, brows(m.x, T), bcol(m.dfsrc, (int64_t)t * E, E, T));
    }
    RAD_ISSUE_N(c, m.norm, T, brows(m.x, T), RAD_W(m.final_gain), brows(m.final_hidden, T));
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.gather, batch->n_out, brows(m.final_hidden, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.logit_input, batch->n_out));
        RAD_ISSUE_N(c, m.head_op, batch->n_out, brows(m.logit_input, batch->n_out), RAD_W(m.head),
                    brows(m.logits, batch->n_out));
    }
}
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->shape_probe_ok = 1;
    /* LiLiCorr proposes exactly its trained block less the anchor. */
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    if (da && std::strstr(da, "LiLiCorr")) {
        const long long bs = rad_meta_geti(meta, "draft.block_size",
                                           rad_meta_geti(meta, "draft.dflash_config.block_size", 8));
        out->draft_depth = bs > 1 ? (int)(bs - 1) : 7;
        out->draft_depth_fixed = 1;
    }
    /* The vision encoder's rows per output row, so a pass bound is whole merged rows. */
    if (Radio::present(meta)) {
        const double r = rad_meta_getf(meta, "downsample_ratio", 0.0);
        const int64_t mg = r > 0.0 ? (int64_t)std::llround(1.0 / r) : 0;
        out->enc_merge = mg * mg;
    }
    return RAD_OK;
}
} // namespace nemotronh
#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(nemotronh)
RAD_ARCH_PLUGIN(nemotronh, "nemotronh", "", "0.1.0",
                "Nemotron-H (nemotron_h_omni/puzzle): Mamba-2, latent relu^2 MoE and NoPE attention")
#endif
