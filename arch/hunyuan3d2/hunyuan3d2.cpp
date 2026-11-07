/* hunyuan3d2 -- Hunyuan3D-2's shape model (tencent/Hunyuan3D-2, hunyuan3d-dit-v2-0): one picture
 * to an untextured 3-D shape.
 *
 * ============================== THE MODEL ==============================
 *
 * Three networks in one checkpoint (hy3dgen's Hunyuan3DDiTFlowMatchingPipeline):
 *
 *   Image encoder: DINOv2-giant (40 layers of 1536, 24 heads, SwiGLU feed-forward and layer
 *   scales) over the prepared picture, 518 x 518 in 14 x 14 patches and the class token, 1370 rows.
 *   Its last hidden state, after the final LayerNorm, is the conditioning; the unconditioned
 *   branch's is zeros.
 *
 *   Denoiser: a FLUX-shaped transformer of width 1024 (16 heads of 64) over a set of 3072 latent
 *   rows of 64 values. The conditioning goes in through `cond_in`, the latent through
 *   `latent_in`; 16 double-stream blocks keep the two streams' weights apart and attend over both
 *   together, then 32 single-stream blocks run [conditioning | latent] as one. Each block's
 *   LayerNorms (no affine) are modulated by shift, scale and gate projected from the timestep
 *   embedding, and q and k are RMS-normed per head. Nothing is positional: the latent is a set. The
 *   output layer maps the latent rows to a velocity.
 *
 *   Decoder: the shape VAE's latent network (64 -> 1024, then 16 self-attention blocks with
 *   per-head LayerNorms on q and k) turns the latent into 3072 rows the geometry decoder reads.
 *   The geometry decoder embeds a query point (x, sin and cos at 8 octaves, 51 values), projects
 *   it to 1024, attends once over those rows (one cross-attention block with its MLP), and maps
 *   the result to the occupancy logit there.
 *
 * Sampling is flow matching from sigma 0 (noise) to 1 (the shape), Euler steps, guided at 5
 * against the unconditioned branch (abi/rad_builder.h, RadShapeGenDecl). Both branches run in one
 * step: rows [txt | latent] for the picture's branch, then [latent | txt] for the unconditioned
 * one, so each branch is one run for the attention and the two latents are one run for every
 * latent-stream op:
 *
 *   rows [0, C)            the picture's conditioning      (C = 1370)
 *   rows [C, C + L)        the picture's latent            (L = 3072)
 *   rows [C + L, C + 2L)   the unconditioned latent
 *   rows [C + 2L, 2C + 2L) the unconditioned conditioning
 *
 * ============================== PRECISION ==============================
 *
 * The reference runs in fp16; here every weight and activation is bf16, rounded where the
 * reference stores a tensor, except the picture's patches and the query points' embedding, which
 * stay f32 into their projections, and the timestep, which the embedding reads in f32.
 *
 * One card: the model is 2.2B parameters (4.4 GB in bf16), and every pass is one request's.
 */
#include <arch/rad_arch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

namespace hunyuan3d2 {

using namespace rad::arch;

/* ================================================================== the model's constants */
/* DINOv2's picture: 518 x 518 in 14 x 14 patches, and its normalisation. */
enum { kEncSize = 518, kPatch = 14, kGrid = kEncSize / kPatch, kNPatch = kGrid * kGrid,
       kCond = kNPatch + 1, kPatchDim = 3 * kPatch * kPatch };
static const double kMean[3] = { 0.485, 0.456, 0.406 };
static const double kStd[3] = { 0.229, 0.224, 0.225 };
/* ImageProcessorV2's canvas and border. */
enum { kCanvas = 512 };
/* Requests in flight; a pass is one request's. */
enum { kLanes = 2 };
/* The query points a pass carries. */
enum { kMaxQuery = 32768 };
/* The query embedding: x, then sin and cos at kFreqs octaves (1, 2, 4, ..., without pi). */
enum { kFreqs = 8 };
/* The grid: the reference's default bounds, and the resolutions served (Hunyuan3D-2's own app
 * offers 16..512, 256 by default). */
static const double kBounds = 1.01;
enum { kResDef = 256, kResMin = 16, kResMax = 512 };

enum : int {
    kGrpDino = 1000,   /* the encoder's layers kGrpDino + 1.. */
    kGrpDit = 2000,    /* the double blocks kGrpDit + 1.., the single blocks kGrpSingle + 1.. */
    kGrpSingle = 2100,
    kGrpVae = 3000,
};

/* ================================================================== the model */
/* A bf16 linear: weight, bias, and its op. */
struct Lin {
    rad_weight w = 0, b = 0;
    rad_op op = 0;
};

struct DinoLayer {
    rad_weight n1w = 0, n1b = 0, n2w = 0, n2b = 0, ls1 = 0, ls2 = 0;
    Lin qkv, o, up, down;
    rad_op op_n1 = 0, op_attn = 0, op_ls1 = 0, op_add1 = 0, op_n2 = 0, op_gu = 0, op_ls2 = 0,
           op_add2 = 0;
};

/* One stream of a double block. */
struct Stream {
    rad_weight qn = 0, kn = 0;
    Lin mod, qkv, proj, mlp1, mlp2;
    rad_op op_n1 = 0, op_qk = 0, op_n2 = 0;
};

struct Double {
    Stream s[2];          /* kImg, kTxt */
    rad_op op_attn = 0;
};
enum { kImg = 0, kTxt = 1 };

struct Single {
    rad_weight qn = 0, kn = 0;
    Lin mod, l1, l2;
    rad_op op_n = 0, op_qk = 0, op_attn = 0, op_act = 0;
};

struct VaeLayer {
    rad_weight n1w = 0, n1b = 0, n2w = 0, n2b = 0, qnw = 0, qnb = 0, knw = 0, knb = 0;
    Lin q, k, v, proj, fc, out;
    rad_op op_n1 = 0, op_qn = 0, op_kn = 0, op_attn = 0, op_n2 = 0;
};

struct Model {
    Names nm{""};
    /* Parameter strings longer than Names keeps (its names are at most 255 characters). */
    std::deque<std::string> strs;
    /* Dimensions, from config.yaml (the importer's metadata). */
    int64_t D = 1024, H = 16, hd = 64, ff = 4096, C = 64, L = 3072;
    int64_t E = 1536, EH = 24, Eff = 4096;
    int64_t n_double = 16, n_single = 32, n_vae = 16, n_dino = 40;
    double  scale_factor = 1.0;
    int64_t rows = 0;     /* a step's: 2 (kCond + L) */
    int64_t lanes = kLanes;

    /* ---- the image encoder */
    rad_weight w_patch = 0, b_patchw = 0, w_cls = 0, w_pos = 0, w_lnw = 0, w_lnb = 0;
    rad_op op_ids = 0, op_pos = 0, op_cls = 0, op_cls_add = 0, op_patch = 0, op_dcu = 0,
           op_dln = 0;
    std::vector<DinoLayer> dino;
    rad_buf b_ids = 0, b_pos = 0, b_cls = 0, b_dx = 0, b_dh = 0, b_dqkv = 0, b_datt = 0,
            b_do = 0, b_dgu = 0, b_dact = 0, b_dcu = 0, b_dout = 0;
    /* ---- cond_in, both branches, into the lane's store */
    Lin cond_in;
    rad_op op_zero = 0;
    rad_buf b_zero = 0, b_cstore = 0;

    /* ---- the denoiser */
    Lin latent_in, t_in1, t_out, fin_mod, fin;
    rad_op op_randn = 0, op_lgather = 0, op_ccopy = 0, op_temb = 0, op_tsilu = 0, op_vsilu = 0,
           op_cu_txt = 0, op_cu_img = 0, op_cu_all = 0, op_cu_ent = 0, op_gate_img = 0,
           op_gate_txt = 0, op_fin_n = 0, op_euler = 0;
    std::vector<Double> dbl;
    std::vector<Single> sgl;
    rad_buf b_lat = 0, b_lx = 0, b_temb = 0, b_t1 = 0, b_t1s = 0, b_vec = 0, b_svec = 0,
            b_mods = 0, b_x = 0, b_h = 0, b_qkv = 0, b_att = 0, b_o = 0, b_ff = 0, b_f = 0,
            b_l1 = 0, b_cat = 0, b_v = 0, b_cu_txt = 0, b_cu_img = 0, b_cu_all = 0, b_cu_ent = 0;
    int64_t mods_w = 0;
    std::vector<int64_t> mod_dbl[2], mod_sgl;   /* each block's columns in b_mods */
    int64_t mod_fin = 0;

    /* ---- the decoder's latent network, then the geometry decoder's keys and values */
    Lin post_kl;
    rad_op op_unscale = 0, op_vcu = 0;
    std::vector<VaeLayer> vae;
    rad_weight g_n2w = 0, g_n2b = 0, g_knw = 0, g_knb = 0;
    Lin g_k, g_v;
    rad_op op_gn2 = 0, op_gkn = 0;
    rad_buf b_vl = 0, b_vx = 0, b_vh = 0, b_vq = 0, b_vk = 0, b_vv = 0, b_vqn = 0, b_vkn = 0,
            b_vatt = 0, b_vff = 0, b_vcu = 0, b_gk = 0, b_gv = 0;

    /* ---- the geometry decoder at the query points */
    rad_weight q_n1w = 0, q_n1b = 0, q_n3w = 0, q_n3b = 0, q_qnw = 0, q_qnb = 0, q_pw = 0,
               q_pb = 0;
    Lin q_proj, q_q, q_o, q_fc, q_out, q_head;
    rad_op op_four = 0, op_qn1 = 0, op_qqn = 0, op_qattn = 0, op_qn3 = 0, op_qpost = 0,
           op_qcast = 0;
    rad_buf b_feat = 0, b_qx = 0, b_qh = 0, b_qq = 0, b_qqn = 0, b_qatt = 0, b_qff = 0,
            b_qlogit = 0, b_field = 0;
};

static Model g_model;

/* ================================================================== declare helpers */
static rad_weight wcopy(RadBuilder* b, const char* dn, const char* src, uint32_t dtype,
                        std::initializer_list<int64_t> shape, int access, int group) {
    if (map_copy(b, dn, src) < 0) return 0;
    return decl_w(b, dn, dtype, shape, access, RAD_SHARD_NONE, grp_layer(group));
}

/* A biased linear [N, K] (bias absent with `bias` false), read whole, over at most `rows` rows,
 * its activation and residual in the epilogue. The weight is mapped by the caller (`mapped`) or
 * copied from `src`.weight. `adt` is the activation's dtype ("bf16", or "f32" with an f32
 * weight). */
static int linear(RadBuilder* b, Names& nm, Lin& l, const char* dn, const char* src, int64_t N,
                  int64_t K, int64_t rows, rad_buf in, rad_buf out, int access, int group,
                  bool bias = true, const char* act = "none", rad_buf res = 0, bool mapped = false,
                  const char* adt = "bf16") {
    const uint32_t wdt = std::strcmp(adt, "f32") == 0 ? RAD_F32 : RAD_BF16;
    const char* wn = nm.f("%s.weight", dn);
    if (!mapped && map_copy(b, wn, nm.ckpt("%s.weight", src)) < 0) return RAD_E_INVAL;
    l.w = decl_w(b, wn, wdt, { N, K }, access, RAD_SHARD_NONE, grp_layer(group));
    if (!l.w) return RAD_E_INVAL;
    if (bias) {
        l.b = wcopy(b, nm.f("%s.bias", dn), nm.ckpt("%s.bias", src), wdt, { N }, access, group);
        if (!l.b) return RAD_E_INVAL;
        l.op = RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", adt), RAD_STR("act", act)),
                      RAD_WEIGHTS(l.w, l.b));
    } else {
        l.op = RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", adt)),
                      RAD_WEIGHTS(l.w));
    }
    if (!l.op) return RAD_E_INVAL;
    /* An input read from the batch (`in` 0) has no buffer to name. */
    const rad_buf reads[2] = { in, res };
    for (rad_buf r : reads)
        if (r) rad_op_reads(b, l.op, &r, 1);
    rad_op_writes(b, l.op, &out, 1);
    return RAD_OK;
}

/* Issue a linear over `n` rows: `in` -> `out`, the residual `res` added (RAD_NONE for none). */
static void lin_issue(RadCtx* c, const Lin& l, int64_t n, RadOperand in, RadOperand out,
                      RadOperand res = RAD_NONE) {
    if (l.b) RAD_ISSUE_N(c, l.op, n, in, RAD_W(l.w), RAD_W(l.b), res, out);
    else     RAD_ISSUE_N(c, l.op, n, in, RAD_W(l.w), out);
}

/* A cu_seqlens of `n` + 1 entries, every run `step` rows: iota's constant runs. */
static rad_op cu_op(RadBuilder* b, int64_t n, int64_t step, rad_buf y) {
    return rw(b, RAD_OP(b, "iota",
                        RAD_PARAMS(RAD_RANGE("M", 1, n + 1), RAD_INT("start", 0),
                                   RAD_INT("step", step)),
                        RAD_NOWEIGHTS),
              {}, { y });
}

static rad_op attn_op(RadBuilder* b, const char* kind, int64_t rows, int64_t hd, rad_buf q,
                      rad_buf cu, rad_buf out) {
    return rw(b, RAD_OP(b, kind,
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", hd),
                                   RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                   RAD_STR("dtype", "bf16"),
                                   RAD_F64("scale", 1.0 / std::sqrt((double)hd))),
                        RAD_NOWEIGHTS),
              { q, cu }, { out });
}

/* norm(x) (1 + scale) + shift, a LayerNorm with no affine, after the gated branch before it. */
static rad_op mod_norm(RadBuilder* b, int64_t rows, int64_t n, bool gated, rad_buf x, rad_buf y,
                       rad_buf mods, rad_buf cu, rad_buf z) {
    rad_op op = RAD_OP(b, "rms_mod",
                       RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", 1e-6),
                                  RAD_INT("gated", gated ? 1 : 0), RAD_STR("dtype", "bf16"),
                                  RAD_STR("norm", "layer")),
                       RAD_NOWEIGHTS);
    if (!op) return op;
    if (gated) rw(b, op, { x, y, mods, cu }, { x, z });
    else       rw(b, op, { x, mods, cu }, { z });
    return op;
}

/* `n` copies of `v` as chan_affine's list, f32 digits (the op keeps the list in f32 at round 0). */
static const char* list_of(Model& m, int64_t n, double v) {
    std::string s;
    char one[40];
    std::snprintf(one, sizeof one, "%.9g", v);
    for (int64_t i = 0; i < n; ++i) { if (i) s += ' '; s += one; }
    m.strs.push_back(std::move(s));
    return m.strs.back().c_str();
}

/* ================================================================== check_format */
static int check_format(const RadModelMeta* meta) {
    const char* dit = rad_meta_gets(meta, "model.target", "");
    const char* vae = rad_meta_gets(meta, "vae.target", "");
    const char* enc = rad_meta_gets(meta, "conditioner.params.main_image_encoder.type", "");
    if (std::strcmp(dit, "hy3dgen.shapegen.models.Hunyuan3DDiT") != 0 ||
        std::strcmp(vae, "hy3dgen.shapegen.models.ShapeVAE") != 0 ||
        std::strcmp(enc, "DinoImageEncoder") != 0) {
        fprintf(stderr, "radiance: hunyuan3d2: the checkpoint is '%s' with '%s' and '%s'; this "
                        "plugin serves Hunyuan3DDiT with ShapeVAE and DINOv2 "
                        "(hunyuan3d-dit-v2-0)\n",
                dit, vae, enc);
        return RAD_E_UNSUPPORTED;
    }
    if (rad_meta_geti(meta, "model.params.guidance_embed", 0) != 0) {
        fprintf(stderr, "radiance: hunyuan3d2: a guidance-distilled DiT (guidance_embed) is not "
                        "served yet\n");
        return RAD_E_UNSUPPORTED;
    }
    if (rad_meta_geti(meta, "model.params.depth", 16) <= 0 ||
        rad_meta_geti(meta, "vae.params.num_freqs", kFreqs) != kFreqs ||
        rad_meta_geti(meta, "vae.params.include_pi", 0) != 0 ||
        rad_meta_geti(meta, "conditioner.params.main_image_encoder.kwargs.image_size", kEncSize) !=
            kEncSize ||
        rad_meta_geti(meta, "conditioner.params.main_image_encoder.kwargs.config.patch_size",
                      kPatch) != kPatch ||
        rad_meta_geti(meta, "conditioner.params.main_image_encoder.kwargs.config.use_swiglu_ffn",
                      1) != 1) {
        fprintf(stderr, "radiance: hunyuan3d2: the checkpoint's geometry (query embedding, "
                        "encoder picture or feed-forward) is not hunyuan3d-dit-v2-0's\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* ================================================================== declare */
static int declare_dino(RadBuilder* b, Model& m) {
    Names& nm = m.nm;
    const int64_t R = kCond, E = m.E;
    const char* P = "conditioner.main_image_encoder.model";
    const int g0 = kGrpDino;
    const int acc = RAD_ACCESS_PER_REQUEST;
    m.b_ids  = decl_b(b, nm.f("dino.ids"), RAD_I32, { R });
    m.b_pos  = decl_b(b, nm.f("dino.pos"), RAD_BF16, { R, E });
    m.b_cls  = decl_b(b, nm.f("dino.cls"), RAD_BF16, { 1, E });
    m.b_dx   = decl_b(b, nm.f("dino.x"), RAD_BF16, { R, E });
    m.b_dh   = decl_b(b, nm.f("dino.h"), RAD_BF16, { R, E });
    m.b_dqkv = decl_b(b, nm.f("dino.qkv"), RAD_BF16, { R, 3 * E });
    m.b_datt = decl_b(b, nm.f("dino.att"), RAD_BF16, { R, E });
    m.b_do   = decl_b(b, nm.f("dino.o"), RAD_BF16, { R, E });
    m.b_dgu  = decl_b(b, nm.f("dino.gate_up"), RAD_BF16, { R, 2 * m.Eff });
    m.b_dact = decl_b(b, nm.f("dino.act"), RAD_BF16, { R, m.Eff });
    m.b_dcu  = decl_b(b, nm.f("dino.cu"), RAD_I32, { 2 });
    m.b_dout = decl_b(b, nm.f("dino.out"), RAD_BF16, { R, E });

    m.op_ids = rw(b, RAD_OP(b, "iota",
                            RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("start", 0),
                                       RAD_INT("step", 1)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_ids });
    m.op_dcu = cu_op(b, 1, R, m.b_dcu);
    /* The position table and the class token as rows of the stream: embedding lookups of
     * the table's own rows. */
    m.w_pos = wcopy(b, nm.f("dino.pos_embed"), nm.ckpt("%s.embeddings.position_embeddings", P),
                    RAD_BF16, { R, E }, RAD_ACCESS_VOCAB, g0);
    m.w_cls = wcopy(b, nm.f("dino.cls_token"), nm.ckpt("%s.embeddings.cls_token", P), RAD_BF16,
                    { 1, E }, RAD_ACCESS_VOCAB, g0);
    if (!m.w_pos || !m.w_cls) return RAD_E_INVAL;
    auto lookup = [&](int64_t rows, int64_t vocab, rad_weight w, rad_buf out) {
        return rw(b, RAD_OP(b, "embed_lookup",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n_embd", E),
                                       RAD_INT("n_vocab", vocab), RAD_STR("dtype", "bf16"),
                                       RAD_INT("vocab_offset", 0)),
                            RAD_WEIGHTS(w)),
                  { m.b_ids }, { out });
    };
    m.op_pos = lookup(R, R, m.w_pos, m.b_pos);
    m.op_cls = lookup(1, 1, m.w_cls, m.b_cls);
    m.op_cls_add = rw(b, RAD_OP(b, "add",
                                RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", E),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_cls, m.b_pos }, { m.b_dx });
    /* The 14 x 14 patch convolution as a linear over the f32 patches (its kernel read as
     * [1536, 3 x 14 x 14]), the position rows added. */
    Lin patch;
    RAD_ARCH_TRY(linear(b, nm, patch, "dino.patch",
                        nm.ckpt("%s.embeddings.patch_embeddings.projection", P), E, kPatchDim,
                        kNPatch, 0, m.b_dx, acc, g0, true, "none", m.b_pos, false, "f32"));
    m.w_patch = patch.w;
    m.b_patchw = patch.b;
    m.op_patch = patch.op;
    m.dino.resize((size_t)m.n_dino);
    for (int64_t l = 0; l < m.n_dino; ++l) {
        DinoLayer& Y = m.dino[(size_t)l];
        const int g = g0 + 1 + (int)l;
        const char* lp = nm.ckpt("%s.encoder.layer.%lld", P, (long long)l);
        auto W = [&](const char* dn, const char* src, std::initializer_list<int64_t> sh) {
            return wcopy(b, nm.f("dino.blk.%lld.%s", (long long)l, dn), nm.ckpt("%s.%s", lp, src),
                         RAD_BF16, sh, acc, g);
        };
        Y.n1w = W("norm1.weight", "norm1.weight", { E });
        Y.n1b = W("norm1.bias", "norm1.bias", { E });
        Y.n2w = W("norm2.weight", "norm2.weight", { E });
        Y.n2b = W("norm2.bias", "norm2.bias", { E });
        Y.ls1 = W("ls1", "layer_scale1.lambda1", { E });
        Y.ls2 = W("ls2", "layer_scale2.lambda1", { E });
        if (!Y.n1w || !Y.n1b || !Y.n2w || !Y.n2b || !Y.ls1 || !Y.ls2) return RAD_E_INVAL;
        Y.op_n1 = rw(b, RAD_OP(b, "layernorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", E),
                                          RAD_F64("eps", 1e-6), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(Y.n1w, Y.n1b)),
                     { m.b_dx }, { m.b_dh });
        const char* qn = nm.f("dino.blk.%lld.qkv", (long long)l);
        if (map_concat(b, nm.f("%s.weight", qn),
                       { nm.ckpt("%s.attention.attention.query.weight", lp),
                         nm.ckpt("%s.attention.attention.key.weight", lp),
                         nm.ckpt("%s.attention.attention.value.weight", lp) }) < 0 ||
            map_concat(b, nm.f("%s.bias", qn),
                       { nm.ckpt("%s.attention.attention.query.bias", lp),
                         nm.ckpt("%s.attention.attention.key.bias", lp),
                         nm.ckpt("%s.attention.attention.value.bias", lp) }) < 0)
            return RAD_E_INVAL;
        /* map_concat named both; the linear declares them without copying again. */
        Y.qkv.w = decl_w(b, nm.f("%s.weight", qn), RAD_BF16, { 3 * E, E }, acc, RAD_SHARD_NONE,
                         grp_layer(g));
        Y.qkv.b = decl_w(b, nm.f("%s.bias", qn), RAD_BF16, { 3 * E }, acc, RAD_SHARD_NONE,
                         grp_layer(g));
        if (!Y.qkv.w || !Y.qkv.b) return RAD_E_INVAL;
        Y.qkv.op = rw(b, RAD_OP(b, "gemm_nt_bias",
                                RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("N", 3 * E),
                                           RAD_INT("K", E), RAD_STR("dtype", "bf16"),
                                           RAD_STR("act", "none")),
                                RAD_WEIGHTS(Y.qkv.w, Y.qkv.b)),
                      { m.b_dh }, { m.b_dqkv });
        Y.op_attn = attn_op(b, "attn_dense", R, 64, m.b_dqkv, m.b_dcu, m.b_datt);
        RAD_ARCH_TRY(linear(b, nm, Y.o, nm.f("dino.blk.%lld.attn_out", (long long)l),
                            nm.ckpt("%s.attention.output.dense", lp), E, E, R, m.b_datt, m.b_do,
                            acc, g));
        auto ls = [&](rad_weight w) {
            return rw(b, RAD_OP(b, "mul_weight",
                                RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", E),
                                           RAD_STR("dtype", "bf16")),
                                RAD_WEIGHTS(w)),
                      { m.b_do }, { m.b_do });
        };
        auto add = [&]() {
            return rw(b, RAD_OP(b, "add",
                                RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", E),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_do, m.b_dx }, { m.b_dx });
        };
        Y.op_ls1 = ls(Y.ls1);
        Y.op_add1 = add();
        Y.op_n2 = rw(b, RAD_OP(b, "layernorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", E),
                                          RAD_F64("eps", 1e-6), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(Y.n2w, Y.n2b)),
                     { m.b_dx }, { m.b_dh });
        RAD_ARCH_TRY(linear(b, nm, Y.up, nm.f("dino.blk.%lld.ffn_in", (long long)l),
                            nm.ckpt("%s.mlp.weights_in", lp), 2 * m.Eff, E, R, m.b_dh, m.b_dgu,
                            acc, g));
        Y.op_gu = rw(b, RAD_OP(b, "silu_mul",
                               RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", m.Eff),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_dgu }, { m.b_dact });
        RAD_ARCH_TRY(linear(b, nm, Y.down, nm.f("dino.blk.%lld.ffn_out", (long long)l),
                            nm.ckpt("%s.mlp.weights_out", lp), E, m.Eff, R, m.b_dact, m.b_do,
                            acc, g));
        Y.op_ls2 = ls(Y.ls2);
        Y.op_add2 = add();
        if (!Y.op_n1 || !Y.qkv.op || !Y.op_attn || !Y.op_ls1 || !Y.op_add1 || !Y.op_n2 ||
            !Y.op_gu || !Y.op_ls2 || !Y.op_add2)
            return RAD_E_INVAL;
    }
    m.w_lnw = wcopy(b, nm.f("dino.norm.weight"), nm.ckpt("%s.layernorm.weight", P), RAD_BF16,
                    { E }, acc, g0);
    m.w_lnb = wcopy(b, nm.f("dino.norm.bias"), nm.ckpt("%s.layernorm.bias", P), RAD_BF16, { E },
                    acc, g0);
    if (!m.w_lnw || !m.w_lnb) return RAD_E_INVAL;
    m.op_dln = rw(b, RAD_OP(b, "layernorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", E),
                                       RAD_F64("eps", 1e-6), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(m.w_lnw, m.w_lnb)),
                  { m.b_dx }, { m.b_dout });

    /* ---- cond_in for both branches: the picture's rows, and the unconditioned branch's zeros
     * (whose rows are cond_in's bias), into the lane's store */
    m.b_zero = decl_b(b, nm.f("cond.zero"), RAD_BF16, { R, E });
    m.op_zero = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", E),
                                        RAD_STR("dtype", "bf16"), RAD_F64("value", 0.0)),
                             RAD_NOWEIGHTS),
                   {}, { m.b_zero });
    RAD_ARCH_TRY(linear(b, nm, m.cond_in, "dit.cond_in", "model.cond_in", m.D, E, R, m.b_dout,
                        m.b_cstore, acc, kGrpDit));
    rw(b, m.cond_in.op, { m.b_dout, m.b_zero }, { m.b_cstore });
    return m.op_ids && m.op_dcu && m.op_pos && m.op_cls && m.op_cls_add && m.op_patch &&
                   m.op_dln && m.op_zero
               ? RAD_OK : RAD_E_INVAL;
}

/* One stream of a double block over `rows` rows. */
static int declare_stream(RadBuilder* b, Model& m, Stream& S, int64_t l, int which, int64_t rows,
                          int g) {
    Names& nm = m.nm;
    const char* nmS = which == kImg ? "img" : "txt";
    const char* src = nm.ckpt("model.double_blocks.%lld", (long long)l);
    const int acc = RAD_ACCESS_PER_TOKEN;
    const char* dn = nm.f("dit.dbl.%lld.%s", (long long)l, nmS);
    S.qn = wcopy(b, nm.f("%s.q_norm", dn), nm.ckpt("%s.%s_attn.norm.query_norm.scale", src, nmS),
                 RAD_F32, { m.hd }, acc, g);
    S.kn = wcopy(b, nm.f("%s.k_norm", dn), nm.ckpt("%s.%s_attn.norm.key_norm.scale", src, nmS),
                 RAD_F32, { m.hd }, acc, g);
    if (!S.qn || !S.kn) return RAD_E_INVAL;
    RAD_ARCH_TRY(linear(b, nm, S.mod, nm.f("%s.mod", dn), nm.ckpt("%s.%s_mod.lin", src, nmS),
                        6 * m.D, m.D, 1, m.b_svec, m.b_mods, acc, g));
    const rad_buf cu = which == kImg ? m.b_cu_img : m.b_cu_txt;
    S.op_n1 = mod_norm(b, rows, m.D, l > 0, m.b_x, m.b_f, m.b_mods, cu, m.b_h);
    RAD_ARCH_TRY(linear(b, nm, S.qkv, nm.f("%s.qkv", dn), nm.ckpt("%s.%s_attn.qkv", src, nmS),
                        3 * m.D, m.D, rows, m.b_h, m.b_qkv, acc, g));
    S.op_qk = rw(b, RAD_OP(b, "qk_norm_rope",
                           RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", m.hd),
                                      RAD_INT("n_head", m.H), RAD_INT("n_head_kv", m.H),
                                      RAD_F64("theta", 10000.0), RAD_F64("eps", 1e-6),
                                      RAD_INT("rotary_dim", 0)),
                           RAD_WEIGHTS(S.qn, S.kn)),
                 { m.b_qkv }, { m.b_qkv });
    return S.op_n1 && S.op_qk ? RAD_OK : RAD_E_INVAL;
}

static int declare_stream_out(RadBuilder* b, Model& m, Stream& S, int64_t l, int which,
                              int64_t rows, int g) {
    Names& nm = m.nm;
    const char* nmS = which == kImg ? "img" : "txt";
    const char* src = nm.ckpt("model.double_blocks.%lld", (long long)l);
    const int acc = RAD_ACCESS_PER_TOKEN;
    const char* dn = nm.f("dit.dbl.%lld.%s", (long long)l, nmS);
    RAD_ARCH_TRY(linear(b, nm, S.proj, nm.f("%s.proj", dn), nm.ckpt("%s.%s_attn.proj", src, nmS),
                        m.D, m.D, rows, m.b_att, m.b_o, acc, g));
    S.op_n2 = mod_norm(b, rows, m.D, true, m.b_x, m.b_o, m.b_mods,
                       which == kImg ? m.b_cu_img : m.b_cu_txt, m.b_h);
    RAD_ARCH_TRY(linear(b, nm, S.mlp1, nm.f("%s.mlp_in", dn), nm.ckpt("%s.%s_mlp.0", src, nmS),
                        m.ff, m.D, rows, m.b_h, m.b_ff, acc, g, true, "gelu_tanh"));
    RAD_ARCH_TRY(linear(b, nm, S.mlp2, nm.f("%s.mlp_out", dn), nm.ckpt("%s.%s_mlp.2", src, nmS),
                        m.D, m.ff, rows, m.b_ff, m.b_f, acc, g));
    return S.op_n2 ? RAD_OK : RAD_E_INVAL;
}

static int declare_dit(RadBuilder* b, Model& m) {
    Names& nm = m.nm;
    const int64_t N = m.rows, D = m.D, L2 = 2 * m.L;
    const int acc = RAD_ACCESS_PER_TOKEN;
    const int gx = kGrpDit;
    m.b_lx   = decl_b(b, nm.f("dit.lat"), RAD_BF16, { L2, m.C });
    m.b_temb = decl_b(b, nm.f("dit.temb"), RAD_BF16, { 1, 256 });
    m.b_t1   = decl_b(b, nm.f("dit.t1"), RAD_BF16, { 1, D });
    m.b_t1s  = decl_b(b, nm.f("dit.t1s"), RAD_BF16, { 1, D });
    m.b_vec  = decl_b(b, nm.f("dit.vec"), RAD_BF16, { 1, D });
    m.b_svec = decl_b(b, nm.f("dit.svec"), RAD_BF16, { 1, D });
    /* Every block's modulation, one row: each double block's two streams (6 D each), each single
     * block's (3 D) and the output layer's (2 D). */
    for (int s = 0; s < 2; ++s) m.mod_dbl[s].clear();
    m.mod_sgl.clear();
    int64_t col = 0;
    for (int64_t l = 0; l < m.n_double; ++l)
        for (int s = 0; s < 2; ++s) { m.mod_dbl[s].push_back(col); col += 6 * D; }
    for (int64_t l = 0; l < m.n_single; ++l) { m.mod_sgl.push_back(col); col += 3 * D; }
    m.mod_fin = col;
    col += 2 * D;
    m.mods_w = col;
    m.b_mods = decl_b(b, nm.f("dit.mods"), RAD_BF16, { 1, m.mods_w });
    m.b_x   = decl_b(b, nm.f("dit.x"), RAD_BF16, { N, D });
    m.b_h   = decl_b(b, nm.f("dit.h"), RAD_BF16, { N, D });
    m.b_qkv = decl_b(b, nm.f("dit.qkv"), RAD_BF16, { N, 3 * D });
    m.b_att = decl_b(b, nm.f("dit.att"), RAD_BF16, { N, D });
    m.b_o   = decl_b(b, nm.f("dit.o"), RAD_BF16, { N, D });
    m.b_ff  = decl_b(b, nm.f("dit.ff"), RAD_BF16, { N, m.ff });
    m.b_f   = decl_b(b, nm.f("dit.f"), RAD_BF16, { N, D });
    m.b_l1  = decl_b(b, nm.f("dit.l1"), RAD_BF16, { N, 3 * D + m.ff });
    m.b_cat = decl_b(b, nm.f("dit.cat"), RAD_BF16, { N, D + m.ff });
    m.b_v   = decl_b(b, nm.f("dit.v"), RAD_BF16, { L2, m.C });
    m.b_cu_txt = decl_b(b, nm.f("dit.cu_txt"), RAD_I32, { 2 });
    m.b_cu_img = decl_b(b, nm.f("dit.cu_img"), RAD_I32, { 2 });
    m.b_cu_all = decl_b(b, nm.f("dit.cu_all"), RAD_I32, { 2 });
    m.b_cu_ent = decl_b(b, nm.f("dit.cu_ent"), RAD_I32, { 3 });

    /* ---- the latents: the first step's noise, both entries' rows of the lane, latent_in */
    m.op_randn = rw(b, RAD_OP(b, "randn_lanes",
                              RAD_PARAMS(RAD_RANGE("M", 1, L2), RAD_INT("n", m.C),
                                         RAD_INT("stride", m.L), RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_lat }, { m.b_lat });
    m.op_lgather = rw(b, RAD_OP(b, "lane_gather",
                                RAD_PARAMS(RAD_RANGE("M", 1, L2), RAD_INT("n", m.C),
                                           RAD_INT("stride", m.L), RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_lat }, { m.b_lx });
    RAD_ARCH_TRY(linear(b, nm, m.latent_in, "dit.latent_in", "model.latent_in", D, m.C, L2,
                        m.b_lx, m.b_x, acc, gx));
    /* The conditioning rows from the lane's store. */
    m.op_ccopy = rw(b, RAD_OP(b, "cast",
                              RAD_PARAMS(RAD_RANGE("M", 1, kCond), RAD_INT("n", D),
                                         RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_cstore }, { m.b_x });
    /* ---- the timestep: the sinusoid of 1000 sigma, time_in, silu. The reference passes its
     * time_factor (1000) where timestep_embedding takes max_period, so the model was trained on
     * a period of 1000, and the time is scaled by the default factor, also 1000. */
    m.op_temb = rw(b, RAD_OP(b, "timestep_embed",
                             RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 256),
                                        RAD_F64("scale", 1000.0), RAD_F64("max_period", 1000.0),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   {}, { m.b_temb });
    RAD_ARCH_TRY(linear(b, nm, m.t_in1, "dit.time_in.in", "model.time_in.in_layer", D, 256, 1,
                        m.b_temb, m.b_t1, acc, gx));
    m.op_tsilu = rw(b, RAD_OP(b, "silu", RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", D),
                                                    RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_t1 }, { m.b_t1s });
    RAD_ARCH_TRY(linear(b, nm, m.t_out, "dit.time_in.out", "model.time_in.out_layer", D, D, 1,
                        m.b_t1s, m.b_vec, acc, gx));
    m.op_vsilu = rw(b, RAD_OP(b, "silu", RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", D),
                                                    RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_vec }, { m.b_svec });
    m.op_cu_txt = cu_op(b, 1, kCond, m.b_cu_txt);
    m.op_cu_img = cu_op(b, 1, L2, m.b_cu_img);
    m.op_cu_all = cu_op(b, 1, N, m.b_cu_all);
    m.op_cu_ent = cu_op(b, 2, N / 2, m.b_cu_ent);

    /* ---- the double blocks: each stream's norm, qkv and q/k norm, the joint attention, then
     * each stream's projection, second norm and MLP, in the order the step issues them */
    m.dbl.resize((size_t)m.n_double);
    for (int64_t l = 0; l < m.n_double; ++l) {
        Double& B = m.dbl[(size_t)l];
        const int g = gx + 1 + (int)l;
        RAD_ARCH_TRY(declare_stream(b, m, B.s[kImg], l, kImg, L2, g));
        RAD_ARCH_TRY(declare_stream(b, m, B.s[kTxt], l, kTxt, kCond, g));
        B.op_attn = attn_op(b, "attn_dense", N, m.hd, m.b_qkv, m.b_cu_ent, m.b_att);
        RAD_ARCH_TRY(declare_stream_out(b, m, B.s[kImg], l, kImg, L2, g));
        RAD_ARCH_TRY(declare_stream_out(b, m, B.s[kTxt], l, kTxt, kCond, g));
        if (!B.op_attn) return RAD_E_INVAL;
    }
    auto gate_op = [&](int64_t rows, rad_buf cu) {
        return rw(b, RAD_OP(b, "gate_add",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", D),
                                       RAD_STR("dtype", "bf16"), RAD_STR("act", "none")),
                            RAD_NOWEIGHTS),
                  { m.b_f, m.b_mods, cu, m.b_x }, { m.b_x });
    };
    m.op_gate_img = gate_op(L2, m.b_cu_img);
    m.op_gate_txt = gate_op(kCond, m.b_cu_txt);

    /* ---- the single blocks over every row */
    m.sgl.resize((size_t)m.n_single);
    for (int64_t l = 0; l < m.n_single; ++l) {
        Single& S = m.sgl[(size_t)l];
        const int g = kGrpSingle + 1 + (int)l;
        const char* src = nm.ckpt("model.single_blocks.%lld", (long long)l);
        const char* dn = nm.f("dit.sgl.%lld", (long long)l);
        S.qn = wcopy(b, nm.f("%s.q_norm", dn), nm.ckpt("%s.norm.query_norm.scale", src), RAD_F32,
                     { m.hd }, acc, g);
        S.kn = wcopy(b, nm.f("%s.k_norm", dn), nm.ckpt("%s.norm.key_norm.scale", src), RAD_F32,
                     { m.hd }, acc, g);
        if (!S.qn || !S.kn) return RAD_E_INVAL;
        RAD_ARCH_TRY(linear(b, nm, S.mod, nm.f("%s.mod", dn), nm.ckpt("%s.modulation.lin", src),
                            3 * D, D, 1, m.b_svec, m.b_mods, acc, g));
        S.op_n = mod_norm(b, N, D, l > 0, m.b_x, m.b_f, m.b_mods, m.b_cu_all, m.b_h);
        RAD_ARCH_TRY(linear(b, nm, S.l1, nm.f("%s.linear1", dn), nm.ckpt("%s.linear1", src),
                            3 * D + m.ff, D, N, m.b_h, m.b_l1, acc, g));
        S.op_qk = rw(b, RAD_OP(b, "qk_norm_rope",
                               RAD_PARAMS(RAD_RANGE("M", 1, N), RAD_INT("head_dim", m.hd),
                                          RAD_INT("n_head", m.H), RAD_INT("n_head_kv", m.H),
                                          RAD_F64("theta", 10000.0), RAD_F64("eps", 1e-6),
                                          RAD_INT("rotary_dim", 0)),
                               RAD_WEIGHTS(S.qn, S.kn)),
                     { m.b_l1 }, { m.b_l1 });
        S.op_attn = attn_op(b, "attn_dense", N, m.hd, m.b_l1, m.b_cu_ent, m.b_cat);
        S.op_act = rw(b, RAD_OP(b, "gelu_tanh",
                                RAD_PARAMS(RAD_RANGE("M", 1, N), RAD_INT("n", m.ff),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_l1 }, { m.b_cat });
        RAD_ARCH_TRY(linear(b, nm, S.l2, nm.f("%s.linear2", dn), nm.ckpt("%s.linear2", src), D,
                            D + m.ff, N, m.b_cat, m.b_f, acc, g));
        if (!S.op_n || !S.op_qk || !S.op_attn || !S.op_act) return RAD_E_INVAL;
    }

    /* ---- the output layer over the latent rows, and the guided step */
    RAD_ARCH_TRY(linear(b, nm, m.fin_mod, "dit.final.mod", "model.final_layer.adaLN_modulation.1",
                        2 * D, D, 1, m.b_svec, m.b_mods, acc, gx));
    m.op_fin_n = mod_norm(b, L2, D, true, m.b_x, m.b_f, m.b_mods, m.b_cu_img, m.b_h);
    RAD_ARCH_TRY(linear(b, nm, m.fin, "dit.final.linear", "model.final_layer.linear", m.C, D, L2,
                        m.b_h, m.b_v, acc, gx));
    m.op_euler = rw(b, RAD_OP(b, "flow_euler",
                              RAD_PARAMS(RAD_RANGE("M", 1, L2), RAD_INT("n", m.C),
                                         RAD_INT("stride", m.L), RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_v, m.b_lat }, { m.b_lat });
    return m.op_randn && m.op_lgather && m.op_ccopy && m.op_temb && m.op_tsilu && m.op_vsilu &&
                   m.op_cu_txt && m.op_cu_img && m.op_cu_all && m.op_cu_ent && m.op_gate_img &&
                   m.op_gate_txt && m.op_fin_n && m.op_euler
               ? RAD_OK : RAD_E_INVAL;
}

/* Rows of a head-interleaved projection: for every head h of `heads`, rows
 * [h * per + first, h * per + first + width) of `src`, joined (RAD_MAP_ROWS, eight a map). */
static int map_heads(RadBuilder* b, const char* declared, const char* src, int64_t heads,
                     int64_t per, int64_t first, int64_t width) {
    for (int64_t h0 = 0; h0 < heads; h0 += 8) {
        RadNameMap mp = map_base(declared, RAD_MAP_ROWS);
        for (int64_t h = h0; h < std::min<int64_t>(heads, h0 + 8); ++h) {
            mp.src[mp.n_src] = src;
            mp.src_index[mp.n_src] = (int)(h * per + first);
            mp.src_rows[mp.n_src] = width;
            ++mp.n_src;
        }
        RAD_ARCH_TRY(rad_decl_name_map(b, &mp));
    }
    return RAD_OK;
}

/* A per-head LayerNorm (gain and bias of head_dim) over [rows x heads] heads in place of rows. */
static rad_op head_norm(RadBuilder* b, const Model& m, int64_t rows, rad_weight w, rad_weight bb,
                        rad_buf in, rad_buf out) {
    return rw(b, RAD_OP(b, "layernorm",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows * m.H), RAD_INT("n", m.hd),
                                   RAD_F64("eps", 1e-6), RAD_STR("dtype", "bf16")),
                        RAD_WEIGHTS(w, bb)),
              { in }, { out });
}

static rad_op affine_norm(RadBuilder* b, int64_t rows, int64_t n, double eps, rad_weight w,
                          rad_weight bb, rad_buf in, rad_buf out) {
    return rw(b, RAD_OP(b, "layernorm",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                   RAD_STR("dtype", "bf16")),
                        RAD_WEIGHTS(w, bb)),
              { in }, { out });
}

static int declare_vae(RadBuilder* b, Model& m) {
    Names& nm = m.nm;
    const int64_t L = m.L, D = m.D;
    const int acc = RAD_ACCESS_PER_REQUEST;
    m.b_vl  = decl_b(b, nm.f("vae.lat"), RAD_BF16, { L, m.C });
    m.b_vx  = decl_b(b, nm.f("vae.x"), RAD_BF16, { L, D });
    m.b_vh  = decl_b(b, nm.f("vae.h"), RAD_BF16, { L, D });
    m.b_vq  = decl_b(b, nm.f("vae.q"), RAD_BF16, { L, D });
    m.b_vk  = decl_b(b, nm.f("vae.k"), RAD_BF16, { L, D });
    m.b_vv  = decl_b(b, nm.f("vae.v"), RAD_BF16, { L, D });
    m.b_vqn = decl_b(b, nm.f("vae.qn"), RAD_BF16, { L, D });
    m.b_vkn = decl_b(b, nm.f("vae.kn"), RAD_BF16, { L, D });
    m.b_vatt = decl_b(b, nm.f("vae.att"), RAD_BF16, { L, D });
    m.b_vff = decl_b(b, nm.f("vae.ff"), RAD_BF16, { L, 4 * D });
    m.b_vcu = decl_b(b, nm.f("vae.cu"), RAD_I32, { 2 });
    m.op_vcu = cu_op(b, 1, L, m.b_vcu);
    /* 1 / scale_factor times the latent, in f32 and rounded once. */
    m.op_unscale = rw(b, RAD_OP(b, "chan_affine",
                                RAD_PARAMS(RAD_RANGE("M", 1, L), RAD_INT("n", m.C),
                                           RAD_STR("mul", list_of(m, m.C, 1.0 / m.scale_factor)),
                                           RAD_STR("add", list_of(m, m.C, 0.0)),
                                           RAD_STR("mode", "fma"), RAD_STR("dtype", "bf16"),
                                           RAD_INT("round", 0)),
                                RAD_NOWEIGHTS),
                      { m.b_lat }, { m.b_vl });
    RAD_ARCH_TRY(linear(b, nm, m.post_kl, "vae.post_kl", "vae.post_kl", D, m.C, L, m.b_vl, m.b_vx,
                        acc, kGrpVae));
    m.vae.resize((size_t)m.n_vae);
    for (int64_t l = 0; l < m.n_vae; ++l) {
        VaeLayer& V = m.vae[(size_t)l];
        const int g = kGrpVae + 1 + (int)l;
        const char* src = nm.ckpt("vae.transformer.resblocks.%lld", (long long)l);
        const char* dn = nm.f("vae.blk.%lld", (long long)l);
        auto W = [&](const char* d, const char* s, int64_t n) {
            return wcopy(b, nm.f("%s.%s", dn, d), nm.ckpt("%s.%s", src, s), RAD_BF16, { n }, acc,
                         g);
        };
        V.n1w = W("ln_1.weight", "ln_1.weight", D);
        V.n1b = W("ln_1.bias", "ln_1.bias", D);
        V.n2w = W("ln_2.weight", "ln_2.weight", D);
        V.n2b = W("ln_2.bias", "ln_2.bias", D);
        V.qnw = W("q_norm.weight", "attn.attention.q_norm.weight", m.hd);
        V.qnb = W("q_norm.bias", "attn.attention.q_norm.bias", m.hd);
        V.knw = W("k_norm.weight", "attn.attention.k_norm.weight", m.hd);
        V.knb = W("k_norm.bias", "attn.attention.k_norm.bias", m.hd);
        if (!V.n1w || !V.n1b || !V.n2w || !V.n2b || !V.qnw || !V.qnb || !V.knw || !V.knb)
            return RAD_E_INVAL;
        V.op_n1 = affine_norm(b, L, D, 1e-6, V.n1w, V.n1b, m.b_vx, m.b_vh);
        /* c_qkv holds each head's q, k and v side by side (192 rows a head). */
        const char* qkv = nm.ckpt("%s.attn.c_qkv.weight", src);
        const char* part[3] = { "q", "k", "v" };
        Lin* lin3[3] = { &V.q, &V.k, &V.v };
        const rad_buf out3[3] = { m.b_vq, m.b_vk, m.b_vv };
        for (int p = 0; p < 3; ++p) {
            const char* pn = nm.f("%s.attn_%s", dn, part[p]);
            RAD_ARCH_TRY(map_heads(b, nm.f("%s.weight", pn), qkv, m.H, 3 * m.hd, p * m.hd, m.hd));
            RAD_ARCH_TRY(linear(b, nm, *lin3[p], pn, nullptr, D, D, L, m.b_vh, out3[p], acc, g,
                                false, "none", 0, true));
        }
        V.op_qn = head_norm(b, m, L, V.qnw, V.qnb, m.b_vq, m.b_vqn);
        V.op_kn = head_norm(b, m, L, V.knw, V.knb, m.b_vk, m.b_vkn);
        V.op_attn = attn_op(b, "attn_dense", L, m.hd, m.b_vqn, m.b_vcu, m.b_vatt);
        rw(b, V.op_attn, { m.b_vqn, m.b_vkn, m.b_vv }, { m.b_vatt });
        RAD_ARCH_TRY(linear(b, nm, V.proj, nm.f("%s.attn_out", dn), nm.ckpt("%s.attn.c_proj", src),
                            D, D, L, m.b_vatt, m.b_vx, acc, g, true, "none", m.b_vx));
        V.op_n2 = affine_norm(b, L, D, 1e-6, V.n2w, V.n2b, m.b_vx, m.b_vh);
        RAD_ARCH_TRY(linear(b, nm, V.fc, nm.f("%s.mlp_in", dn), nm.ckpt("%s.mlp.c_fc", src),
                            4 * D, D, L, m.b_vh, m.b_vff, acc, g, true, "gelu"));
        RAD_ARCH_TRY(linear(b, nm, V.out, nm.f("%s.mlp_out", dn), nm.ckpt("%s.mlp.c_proj", src),
                            D, 4 * D, L, m.b_vff, m.b_vx, acc, g, true, "none", m.b_vx));
        if (!V.op_n1 || !V.op_qn || !V.op_kn || !V.op_attn || !V.op_n2) return RAD_E_INVAL;
    }
    /* ---- the geometry decoder's keys and values over the latent network's rows, normed and
     * kept in the lane (c_kv holds each head's k and v side by side, 128 rows a head) */
    const char* gd = "vae.geo_decoder.cross_attn_decoder";
    const int g = kGrpVae + 100;
    auto GW = [&](const char* dn, const char* pre, const char* src, int64_t n) {
        return wcopy(b, nm.f("%s", dn), nm.ckpt("%s.%s", pre, src), RAD_BF16, { n }, acc, g);
    };
    m.g_n2w = GW("geo.ln_2.weight", gd, "ln_2.weight", D);
    m.g_n2b = GW("geo.ln_2.bias", gd, "ln_2.bias", D);
    m.g_knw = wcopy(b, nm.f("geo.k_norm.weight"), nm.ckpt("%s.attn.attention.k_norm.weight", gd),
                    RAD_BF16, { m.hd }, acc, g);
    m.g_knb = wcopy(b, nm.f("geo.k_norm.bias"), nm.ckpt("%s.attn.attention.k_norm.bias", gd),
                    RAD_BF16, { m.hd }, acc, g);
    if (!m.g_n2w || !m.g_n2b || !m.g_knw || !m.g_knb) return RAD_E_INVAL;
    m.op_gn2 = affine_norm(b, L, D, 1e-6, m.g_n2w, m.g_n2b, m.b_vx, m.b_vh);
    const char* kv = nm.ckpt("%s.attn.c_kv.weight", gd);
    RAD_ARCH_TRY(map_heads(b, nm.f("geo.attn_k.weight"), kv, m.H, 2 * m.hd, 0, m.hd));
    RAD_ARCH_TRY(map_heads(b, nm.f("geo.attn_v.weight"), kv, m.H, 2 * m.hd, m.hd, m.hd));
    RAD_ARCH_TRY(linear(b, nm, m.g_k, "geo.attn_k", nullptr, D, D, L, m.b_vh, m.b_vk, acc, g,
                        false, "none", 0, true));
    RAD_ARCH_TRY(linear(b, nm, m.g_v, "geo.attn_v", nullptr, D, D, L, m.b_vh, m.b_gv, acc, g,
                        false, "none", 0, true));
    m.op_gkn = head_norm(b, m, L, m.g_knw, m.g_knb, m.b_vk, m.b_gk);
    return m.op_vcu && m.op_unscale && m.op_gn2 && m.op_gkn ? RAD_OK : RAD_E_INVAL;
}

static int declare_query(RadBuilder* b, Model& m) {
    Names& nm = m.nm;
    const int64_t Q = kMaxQuery, D = m.D, F = 3 * (2 * kFreqs + 1);
    const int acc = RAD_ACCESS_PER_REQUEST;
    const int g = kGrpVae + 101;
    const char* gd = "vae.geo_decoder";
    m.b_feat = decl_b(b, nm.f("geo.feat"), RAD_F32, { Q, F });
    m.b_qx   = decl_b(b, nm.f("geo.x"), RAD_BF16, { Q, D });
    m.b_qh   = decl_b(b, nm.f("geo.h"), RAD_BF16, { Q, D });
    m.b_qq   = decl_b(b, nm.f("geo.q"), RAD_BF16, { Q, D });
    m.b_qqn  = decl_b(b, nm.f("geo.qn"), RAD_BF16, { Q, D });
    m.b_qatt = decl_b(b, nm.f("geo.att"), RAD_BF16, { Q, D });
    m.b_qff  = decl_b(b, nm.f("geo.ff"), RAD_BF16, { Q, 4 * D });
    m.b_qlogit = decl_b(b, nm.f("geo.logit"), RAD_BF16, { Q, 1 });
    m.b_field = decl_b(b, nm.f("geo.field"), RAD_F32, { Q });
    m.op_four = rw(b, RAD_OP(b, "point_fourier",
                             RAD_PARAMS(RAD_RANGE("M", 1, Q), RAD_INT("n", 3),
                                        RAD_INT("freqs", kFreqs), RAD_INT("pi", 0),
                                        RAD_STR("dtype", "f32")),
                             RAD_NOWEIGHTS),
                   {}, { m.b_feat });
    RAD_ARCH_TRY(linear(b, nm, m.q_proj, "geo.query_proj", "vae.geo_decoder.query_proj", D, F, Q,
                        m.b_feat, m.b_qx, acc, g, true, "none", 0, false, "f32"));
    const char* cb = "vae.geo_decoder.cross_attn_decoder";
    auto GW = [&](const char* dn, const char* pre, const char* src, int64_t n) {
        return wcopy(b, nm.f("%s", dn), nm.ckpt("%s.%s", pre, src), RAD_BF16, { n }, acc, g);
    };
    m.q_n1w = GW("geo.ln_1.weight", cb, "ln_1.weight", D);
    m.q_n1b = GW("geo.ln_1.bias", cb, "ln_1.bias", D);
    m.q_n3w = GW("geo.ln_3.weight", cb, "ln_3.weight", D);
    m.q_n3b = GW("geo.ln_3.bias", cb, "ln_3.bias", D);
    m.q_qnw = wcopy(b, nm.f("geo.q_norm.weight"), nm.ckpt("%s.attn.attention.q_norm.weight", cb),
                    RAD_BF16, { m.hd }, acc, g);
    m.q_qnb = wcopy(b, nm.f("geo.q_norm.bias"), nm.ckpt("%s.attn.attention.q_norm.bias", cb),
                    RAD_BF16, { m.hd }, acc, g);
    m.q_pw = GW("geo.ln_post.weight", gd, "ln_post.weight", D);
    m.q_pb = GW("geo.ln_post.bias", gd, "ln_post.bias", D);
    if (!m.q_n1w || !m.q_n1b || !m.q_n3w || !m.q_n3b || !m.q_qnw || !m.q_qnb || !m.q_pw || !m.q_pb)
        return RAD_E_INVAL;
    m.op_qn1 = affine_norm(b, Q, D, 1e-6, m.q_n1w, m.q_n1b, m.b_qx, m.b_qh);
    RAD_ARCH_TRY(linear(b, nm, m.q_q, "geo.attn_q", nm.ckpt("%s.attn.c_q", cb), D, D, Q, m.b_qh,
                        m.b_qq, acc, g, false));
    m.op_qqn = head_norm(b, m, Q, m.q_qnw, m.q_qnb, m.b_qq, m.b_qqn);
    m.op_qattn = rw(b, RAD_OP(b, "attn_cross",
                              RAD_PARAMS(RAD_RANGE("M", 1, Q), RAD_INT("head_dim", m.hd),
                                         RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                         RAD_STR("dtype", "bf16"),
                                         RAD_F64("scale", 1.0 / std::sqrt((double)m.hd))),
                              RAD_NOWEIGHTS),
                    { m.b_qqn, m.b_gk, m.b_gv }, { m.b_qatt });
    RAD_ARCH_TRY(linear(b, nm, m.q_o, "geo.attn_out", nm.ckpt("%s.attn.c_proj", cb), D, D, Q,
                        m.b_qatt, m.b_qx, acc, g, true, "none", m.b_qx));
    m.op_qn3 = affine_norm(b, Q, D, 1e-6, m.q_n3w, m.q_n3b, m.b_qx, m.b_qh);
    RAD_ARCH_TRY(linear(b, nm, m.q_fc, "geo.mlp_in", nm.ckpt("%s.mlp.c_fc", cb), 4 * D, D, Q,
                        m.b_qh, m.b_qff, acc, g, true, "gelu"));
    RAD_ARCH_TRY(linear(b, nm, m.q_out, "geo.mlp_out", nm.ckpt("%s.mlp.c_proj", cb), D, 4 * D, Q,
                        m.b_qff, m.b_qx, acc, g, true, "none", m.b_qx));
    /* torch's LayerNorm default eps, 1e-5, unlike the blocks' 1e-6. */
    m.op_qpost = affine_norm(b, Q, D, 1e-5, m.q_pw, m.q_pb, m.b_qx, m.b_qh);
    RAD_ARCH_TRY(linear(b, nm, m.q_head, "geo.output_proj", "vae.geo_decoder.output_proj", 1, D, Q,
                        m.b_qh, m.b_qlogit, acc, g));
    m.op_qcast = rw(b, RAD_OP(b, "cast",
                              RAD_PARAMS(RAD_RANGE("M", 1, Q), RAD_INT("n", 1),
                                         RAD_STR("from", "bf16"), RAD_STR("to", "f32")),
                              RAD_NOWEIGHTS),
                    { m.b_qlogit }, { m.b_field });
    return m.op_four && m.op_qn1 && m.op_qqn && m.op_qattn && m.op_qn3 && m.op_qpost &&
                   m.op_qcast
               ? RAD_OK : RAD_E_INVAL;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    Model& m = g_model;
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(check_format(meta));
    if (ctx->world_size > 1) {
        fprintf(stderr, "radiance: hunyuan3d2 runs on one card (--tp 1): the model is 2.2B and a "
                        "pass is one request's\n");
        return RAD_E_UNSUPPORTED;
    }
    const char* dc = "conditioner.params.main_image_encoder.kwargs.config";
    char key[160];
    auto ci = [&](const char* k, long long d) {
        std::snprintf(key, sizeof key, "%s.%s", dc, k);
        return (int64_t)rad_meta_geti(meta, key, d);
    };
    m.D = rad_meta_geti(meta, "model.params.hidden_size", 1024);
    m.H = rad_meta_geti(meta, "model.params.num_heads", 16);
    m.hd = m.D / std::max<int64_t>(m.H, 1);
    m.ff = (int64_t)(m.D * rad_meta_getf(meta, "model.params.mlp_ratio", 4.0));
    m.C = rad_meta_geti(meta, "model.params.in_channels", 64);
    m.n_double = rad_meta_geti(meta, "model.params.depth", 16);
    m.n_single = rad_meta_geti(meta, "model.params.depth_single_blocks", 32);
    m.L = rad_meta_geti(meta, "vae.params.num_latents", 3072);
    m.n_vae = rad_meta_geti(meta, "vae.params.num_decoder_layers", 16);
    m.scale_factor = rad_meta_getf(meta, "vae.params.scale_factor", 1.0);
    m.E = ci("hidden_size", 1536);
    m.EH = ci("num_attention_heads", 24);
    m.n_dino = ci("num_hidden_layers", 40);
    /* DINOv2's SwiGLU width: 2/3 of mlp_ratio times the width, rounded up to 8. */
    m.Eff = ((int64_t)(m.E * ci("mlp_ratio", 4) * 2 / 3) + 7) / 8 * 8;
    if (m.hd != 64 || m.E / std::max<int64_t>(m.EH, 1) != 64 ||
        rad_meta_geti(meta, "vae.params.width", m.D) != m.D ||
        rad_meta_geti(meta, "vae.params.heads", m.H) != m.H ||
        rad_meta_geti(meta, "vae.params.embed_dim", m.C) != m.C ||
        rad_meta_geti(meta, "model.params.context_in_dim", m.E) != m.E ||
        !(m.scale_factor > 0.0)) {
        fprintf(stderr, "radiance: hunyuan3d2: the networks' widths disagree (DiT %lld x %lld, "
                        "encoder %lld x %lld, VAE scale %g); every attention here is 64 wide\n",
                (long long)m.H, (long long)m.hd, (long long)m.EH, (long long)m.E, m.scale_factor);
        return RAD_E_UNSUPPORTED;
    }
    /* Lanes are requests holding state, not a pass's sequences (a pass is one request's, and
     * --max-num-seqs is 1): one denoising or decoding while the next is conditioned. */
    m.rows = 2 * (kCond + m.L);
    m.lanes = kLanes;
    Names& nm = m.nm;
    m.b_lat    = decl_b(b, nm.f("lat_state"), RAD_BF16, { m.lanes * m.L, m.C }, RAD_BUF_PERSIST);
    m.b_cstore = decl_b(b, nm.f("cond_store"), RAD_BF16, { m.lanes * 2 * kCond, m.D },
                        RAD_BUF_PERSIST);
    m.b_gk = decl_b(b, nm.f("geo_k"), RAD_BF16, { m.lanes * m.L, m.D }, RAD_BUF_PERSIST);
    m.b_gv = decl_b(b, nm.f("geo_v"), RAD_BF16, { m.lanes * m.L, m.D }, RAD_BUF_PERSIST);
    RAD_ARCH_TRY(declare_dino(b, m));
    RAD_ARCH_TRY(declare_dit(b, m));
    RAD_ARCH_TRY(declare_vae(b, m));
    RAD_ARCH_TRY(declare_query(b, m));

    RadShapeGenDecl d{};
    d.name = "hunyuan3d-2";
    d.canvas = rad_meta_geti(meta, "image_processor.params.size", kCanvas);
    d.border_ratio = rad_meta_getf(meta, "image_processor.params.border_ratio", 0.15);
    d.enc_size = kEncSize;
    d.enc_patch = kPatch;
    for (int c = 0; c < 3; ++c) { d.pix_mean[c] = kMean[c]; d.pix_std[c] = kStd[c]; }
    d.latent_rows = m.L;
    d.latent_channels = m.C;
    d.step_rows = m.rows;
    d.default_steps = 50;
    d.max_steps = 200;
    d.default_guidance = 5.0;
    d.bounds = kBounds;
    d.level = 0.0;
    d.default_resolution = kResDef;
    d.min_resolution = kResMin;
    d.max_resolution = kResMax;
    d.max_lanes = m.lanes;
    d.max_query = kMaxQuery;
    d.field_out = m.b_field;
    RAD_ARCH_TRY(rad_declare_shape_generator(b, &d));
    rad_note(b, "Hunyuan3D-2 shape: DINOv2 %lld x %lld, DiT %lld double + %lld single blocks of "
                "%lld, VAE %lld blocks over %lld latents; %lld lanes, %d query points a pass",
             (long long)m.n_dino, (long long)m.E, (long long)m.n_double, (long long)m.n_single,
             (long long)m.D, (long long)m.n_vae, (long long)m.L, (long long)m.lanes, kMaxQuery);
    return RAD_OK;
}

/* ================================================================== step */
static void step_cond(RadCtx* c, const RadShapeGenBatch* S, const Model& m) {
    const int64_t R = kCond, E = m.E, D = m.D;
    RAD_ISSUE_N(c, m.op_ids, R, brows(m.b_ids, R));
    RAD_ISSUE_N(c, m.op_dcu, 2, brows(m.b_dcu, 2));
    RAD_ISSUE_N(c, m.op_pos, R, brows(m.b_ids, R), RAD_W(m.w_pos), brows(m.b_pos, R));
    RAD_ISSUE_N(c, m.op_cls, 1, brows(m.b_ids, 1), RAD_W(m.w_cls), brows(m.b_cls, 1));
    RAD_ISSUE_N(c, m.op_cls_add, 1, brows(m.b_cls, 1), brows(m.b_pos, 1), brows(m.b_dx, 1));
    RAD_ISSUE_N(c, m.op_patch, kNPatch, praw2(S->patches, RAD_F32, kNPatch, kPatchDim),
                RAD_W(m.w_patch), RAD_W(m.b_patchw), brow_slice(m.b_pos, 1, kNPatch, E),
                brow_slice(m.b_dx, 1, kNPatch, E));
    const RadOperand cu = brows(m.b_dcu, 2);
    for (const DinoLayer& Y : m.dino) {
        RAD_ISSUE_N(c, Y.op_n1, R, brows(m.b_dx, R), RAD_W(Y.n1w), RAD_W(Y.n1b), brows(m.b_dh, R));
        lin_issue(c, Y.qkv, R, brows(m.b_dh, R), brows(m.b_dqkv, R));
        RAD_ISSUE_N(c, Y.op_attn, R, bcol(m.b_dqkv, 0, E, R), bcol(m.b_dqkv, E, E, R),
                    bcol(m.b_dqkv, 2 * E, E, R), cu, brows(m.b_datt, R));
        lin_issue(c, Y.o, R, brows(m.b_datt, R), brows(m.b_do, R));
        RAD_ISSUE_N(c, Y.op_ls1, R, brows(m.b_do, R), RAD_W(Y.ls1), brows(m.b_do, R));
        RAD_ISSUE_N(c, Y.op_add1, R, brows(m.b_do, R), brows(m.b_dx, R), brows(m.b_dx, R));
        RAD_ISSUE_N(c, Y.op_n2, R, brows(m.b_dx, R), RAD_W(Y.n2w), RAD_W(Y.n2b), brows(m.b_dh, R));
        lin_issue(c, Y.up, R, brows(m.b_dh, R), brows(m.b_dgu, R));
        RAD_ISSUE_N(c, Y.op_gu, R, brows(m.b_dgu, R), brows(m.b_dact, R));
        lin_issue(c, Y.down, R, brows(m.b_dact, R), brows(m.b_do, R));
        RAD_ISSUE_N(c, Y.op_ls2, R, brows(m.b_do, R), RAD_W(Y.ls2), brows(m.b_do, R));
        RAD_ISSUE_N(c, Y.op_add2, R, brows(m.b_do, R), brows(m.b_dx, R), brows(m.b_dx, R));
    }
    RAD_ISSUE_N(c, m.op_dln, R, brows(m.b_dx, R), RAD_W(m.w_lnw), RAD_W(m.w_lnb),
                brows(m.b_dout, R));
    /* The lane's store: the picture's branch, then the unconditioned one. */
    const int64_t base = (int64_t)S->lane * 2 * R;
    RAD_ISSUE_N(c, m.op_zero, R, brows(m.b_zero, R));
    lin_issue(c, m.cond_in, R, brows(m.b_dout, R), brow_slice(m.b_cstore, base, R, D));
    lin_issue(c, m.cond_in, R, brows(m.b_zero, R), brow_slice(m.b_cstore, base + R, R, D));
}

/* One stream's rows of a step: the latents' one run, or the conditioning's two. */
template <class F> static void each_run(int which, const Model& m, F&& f) {
    if (which == kImg) { f((int64_t)kCond, 2 * m.L); return; }
    f(0, (int64_t)kCond);
    f((int64_t)kCond + 2 * m.L, (int64_t)kCond);
}

static void step_dit(RadCtx* c, const RadShapeGenBatch* S, const Model& m) {
    const int64_t N = m.rows, D = m.D, L2 = 2 * m.L, Cn = kCond;
    const RadOperand lat_cu = praw(S->lat_cu, RAD_I32, 3);
    const RadOperand lanes = praw(S->lanes, RAD_I32, 2);
    const int64_t LR = m.lanes * m.L;
    /* ---- the latent rows: noise on the first step, both entries from the lane, latent_in */
    RAD_ISSUE_N(c, m.op_randn, L2, praw(S->flags, RAD_I32, 2), praw(S->seed, RAD_U32, 4), lanes,
                lat_cu, brows(m.b_lat, LR));
    RAD_ISSUE_N(c, m.op_lgather, L2, brows(m.b_lat, LR), lanes, lat_cu, brows(m.b_lx, L2));
    lin_issue(c, m.latent_in, L2, brows(m.b_lx, L2), brow_slice(m.b_x, Cn, L2, D));
    /* ---- the conditioning rows from the lane's store */
    const int64_t base = (int64_t)S->lane * 2 * Cn;
    RAD_ISSUE_N(c, m.op_ccopy, Cn, brow_slice(m.b_cstore, base, Cn, D),
                brow_slice(m.b_x, 0, Cn, D));
    RAD_ISSUE_N(c, m.op_ccopy, Cn, brow_slice(m.b_cstore, base + Cn, Cn, D),
                brow_slice(m.b_x, Cn + L2, Cn, D));
    /* ---- the timestep and every modulation */
    RAD_ISSUE_N(c, m.op_temb, 1, praw(S->sigma, RAD_F32, 1), brows(m.b_temb, 1));
    lin_issue(c, m.t_in1, 1, brows(m.b_temb, 1), brows(m.b_t1, 1));
    RAD_ISSUE_N(c, m.op_tsilu, 1, brows(m.b_t1, 1), brows(m.b_t1s, 1));
    lin_issue(c, m.t_out, 1, brows(m.b_t1s, 1), brows(m.b_vec, 1));
    RAD_ISSUE_N(c, m.op_vsilu, 1, brows(m.b_vec, 1), brows(m.b_svec, 1));
    for (int64_t l = 0; l < m.n_double; ++l)
        for (int s = 0; s < 2; ++s)
            lin_issue(c, m.dbl[(size_t)l].s[s].mod, 1, brows(m.b_svec, 1),
                      bcol(m.b_mods, m.mod_dbl[s][(size_t)l], 6 * D, 1));
    for (int64_t l = 0; l < m.n_single; ++l)
        lin_issue(c, m.sgl[(size_t)l].mod, 1, brows(m.b_svec, 1),
                  bcol(m.b_mods, m.mod_sgl[(size_t)l], 3 * D, 1));
    lin_issue(c, m.fin_mod, 1, brows(m.b_svec, 1), bcol(m.b_mods, m.mod_fin, 2 * D, 1));
    RAD_ISSUE_N(c, m.op_cu_txt, 2, brows(m.b_cu_txt, 2));
    RAD_ISSUE_N(c, m.op_cu_img, 2, brows(m.b_cu_img, 2));
    RAD_ISSUE_N(c, m.op_cu_all, 2, brows(m.b_cu_all, 2));
    RAD_ISSUE_N(c, m.op_cu_ent, 3, brows(m.b_cu_ent, 3));
    auto mod = [&](int64_t col, int64_t off, int64_t w) {
        return bcol(m.b_mods, col + off * D, w, 1);
    };
    auto rcu = [&](int64_t n) { return n == L2 ? brows(m.b_cu_img, 2) : brows(m.b_cu_txt, 2); };

    /* ---- the double blocks */
    for (int64_t l = 0; l < m.n_double; ++l) {
        const Double& B = m.dbl[(size_t)l];
        for (int s = 0; s < 2; ++s) {
            const Stream& T = B.s[s];
            const int64_t col = m.mod_dbl[s][(size_t)l];
            const int64_t pcol = l > 0 ? m.mod_dbl[s][(size_t)(l - 1)] : 0;
            each_run(s, m, [&](int64_t r0, int64_t n) {
                const RadOperand x = brow_slice(m.b_x, r0, n, D);
                if (l > 0)
                    RAD_ISSUE_N(c, T.op_n1, n, brow_slice(m.b_f, r0, n, D), mod(pcol, 5, D),
                                mod(col, 0, 2 * D), RAD_NONE, rcu(n), x,
                                brow_slice(m.b_h, r0, n, D));
                else
                    RAD_ISSUE_N(c, T.op_n1, n, RAD_NONE, RAD_NONE, mod(col, 0, 2 * D), RAD_NONE,
                                rcu(n), x, brow_slice(m.b_h, r0, n, D));
                lin_issue(c, T.qkv, n, brow_slice(m.b_h, r0, n, D),
                          brow_slice(m.b_qkv, r0, n, 3 * D));
                RAD_ISSUE_N(c, T.op_qk, n, brow_slice(m.b_qkv, r0, n, 3 * D), RAD_NONE,
                            RAD_W(T.qn), RAD_W(T.kn));
            });
        }
        RAD_ISSUE_N(c, B.op_attn, N, bcol(m.b_qkv, 0, D, N), bcol(m.b_qkv, D, D, N),
                    bcol(m.b_qkv, 2 * D, D, N), brows(m.b_cu_ent, 3), brows(m.b_att, N));
        for (int s = 0; s < 2; ++s) {
            const Stream& T = B.s[s];
            const int64_t col = m.mod_dbl[s][(size_t)l];
            each_run(s, m, [&](int64_t r0, int64_t n) {
                lin_issue(c, T.proj, n, brow_slice(m.b_att, r0, n, D), brow_slice(m.b_o, r0, n, D));
                RAD_ISSUE_N(c, T.op_n2, n, brow_slice(m.b_o, r0, n, D), mod(col, 2, D),
                            mod(col, 3, 2 * D), RAD_NONE, rcu(n), brow_slice(m.b_x, r0, n, D),
                            brow_slice(m.b_h, r0, n, D));
                lin_issue(c, T.mlp1, n, brow_slice(m.b_h, r0, n, D),
                          brow_slice(m.b_ff, r0, n, m.ff));
                lin_issue(c, T.mlp2, n, brow_slice(m.b_ff, r0, n, m.ff),
                          brow_slice(m.b_f, r0, n, D));
            });
        }
    }
    /* The last double block's MLP, gated in, each stream by its own gate. */
    {
        const int64_t l = m.n_double - 1;
        for (int s = 0; s < 2; ++s) {
            const int64_t col = m.mod_dbl[s][(size_t)l];
            each_run(s, m, [&](int64_t r0, int64_t n) {
                RAD_ISSUE_N(c, s == kImg ? m.op_gate_img : m.op_gate_txt, n,
                            brow_slice(m.b_f, r0, n, D), mod(col, 5, D), rcu(n),
                            brow_slice(m.b_x, r0, n, D));
            });
        }
    }
    /* ---- the single blocks */
    for (int64_t l = 0; l < m.n_single; ++l) {
        const Single& T = m.sgl[(size_t)l];
        const int64_t col = m.mod_sgl[(size_t)l];
        if (l > 0)
            RAD_ISSUE_N(c, T.op_n, N, brows(m.b_f, N), mod(m.mod_sgl[(size_t)(l - 1)], 2, D),
                        mod(col, 0, 2 * D), RAD_NONE, brows(m.b_cu_all, 2), brows(m.b_x, N),
                        brows(m.b_h, N));
        else
            RAD_ISSUE_N(c, T.op_n, N, RAD_NONE, RAD_NONE, mod(col, 0, 2 * D), RAD_NONE,
                        brows(m.b_cu_all, 2), brows(m.b_x, N), brows(m.b_h, N));
        lin_issue(c, T.l1, N, brows(m.b_h, N), brows(m.b_l1, N));
        RAD_ISSUE_N(c, T.op_qk, N, bcol(m.b_l1, 0, 3 * D, N), RAD_NONE, RAD_W(T.qn), RAD_W(T.kn));
        RAD_ISSUE_N(c, T.op_attn, N, bcol(m.b_l1, 0, D, N), bcol(m.b_l1, D, D, N),
                    bcol(m.b_l1, 2 * D, D, N), brows(m.b_cu_ent, 3), bcol(m.b_cat, 0, D, N));
        RAD_ISSUE_N(c, T.op_act, N, bcol(m.b_l1, 3 * D, m.ff, N), bcol(m.b_cat, D, m.ff, N));
        lin_issue(c, T.l2, N, brows(m.b_cat, N), brows(m.b_f, N));
    }
    /* ---- the output layer over the latent rows (the last single block gated in), the step */
    const int64_t lcol = m.mod_sgl.back();
    RAD_ISSUE_N(c, m.op_fin_n, L2, brow_slice(m.b_f, Cn, L2, D), mod(lcol, 2, D),
                mod(m.mod_fin, 0, 2 * D), RAD_NONE, brows(m.b_cu_img, 2),
                brow_slice(m.b_x, Cn, L2, D), brow_slice(m.b_h, Cn, L2, D));
    lin_issue(c, m.fin, L2, brow_slice(m.b_h, Cn, L2, D), brows(m.b_v, L2));
    RAD_ISSUE_N(c, m.op_euler, L2, brows(m.b_v, L2), praw(S->sigma, RAD_F32, 2),
                praw(S->sigma_next, RAD_F32, 2), praw(S->cfg, RAD_F32, 2),
                praw(S->pair, RAD_I32, 2), lanes, praw(S->flags, RAD_I32, 2),
                praw(S->seed, RAD_U32, 4), lat_cu, RAD_NONE, RAD_NONE, brows(m.b_lat, LR),
                RAD_NONE);
}

/* A per-head view of rows [first_row, first_row + rows) of a [*, H hd] bf16 buffer: the same
 * bytes as [rows H, hd], one head a row (the buffer's liveness is declared on the op). */
static RadOperand heads(RadCtx* c, rad_buf h, int64_t first_row, int64_t rows, const Model& m) {
    const char* p = (const char*)rad_buf_ptr(c, h) + first_row * m.D * 2;
    return RAD_P_T2(p, RAD_BF16, rows * m.H, m.hd);
}

static void step_latent(RadCtx* c, const RadShapeGenBatch* S, const Model& m) {
    const int64_t L = m.L, D = m.D;
    const int64_t base = (int64_t)S->lane * L;
    RAD_ISSUE_N(c, m.op_vcu, 2, brows(m.b_vcu, 2));
    RAD_ISSUE_N(c, m.op_unscale, L, brow_slice(m.b_lat, base, L, m.C), brows(m.b_vl, L));
    lin_issue(c, m.post_kl, L, brows(m.b_vl, L), brows(m.b_vx, L));
    for (const VaeLayer& V : m.vae) {
        RAD_ISSUE_N(c, V.op_n1, L, brows(m.b_vx, L), RAD_W(V.n1w), RAD_W(V.n1b), brows(m.b_vh, L));
        lin_issue(c, V.q, L, brows(m.b_vh, L), brows(m.b_vq, L));
        lin_issue(c, V.k, L, brows(m.b_vh, L), brows(m.b_vk, L));
        lin_issue(c, V.v, L, brows(m.b_vh, L), brows(m.b_vv, L));
        RAD_ISSUE_N(c, V.op_qn, L * m.H, heads(c, m.b_vq, 0, L, m), RAD_W(V.qnw), RAD_W(V.qnb),
                    heads(c, m.b_vqn, 0, L, m));
        RAD_ISSUE_N(c, V.op_kn, L * m.H, heads(c, m.b_vk, 0, L, m), RAD_W(V.knw), RAD_W(V.knb),
                    heads(c, m.b_vkn, 0, L, m));
        RAD_ISSUE_N(c, V.op_attn, L, brows(m.b_vqn, L), brows(m.b_vkn, L), brows(m.b_vv, L),
                    brows(m.b_vcu, 2), brows(m.b_vatt, L));
        lin_issue(c, V.proj, L, brows(m.b_vatt, L), brows(m.b_vx, L), brows(m.b_vx, L));
        RAD_ISSUE_N(c, V.op_n2, L, brows(m.b_vx, L), RAD_W(V.n2w), RAD_W(V.n2b), brows(m.b_vh, L));
        lin_issue(c, V.fc, L, brows(m.b_vh, L), brows(m.b_vff, L));
        lin_issue(c, V.out, L, brows(m.b_vff, L), brows(m.b_vx, L), brows(m.b_vx, L));
    }
    RAD_ISSUE_N(c, m.op_gn2, L, brows(m.b_vx, L), RAD_W(m.g_n2w), RAD_W(m.g_n2b), brows(m.b_vh, L));
    lin_issue(c, m.g_k, L, brows(m.b_vh, L), brows(m.b_vk, L));
    lin_issue(c, m.g_v, L, brows(m.b_vh, L), brow_slice(m.b_gv, base, L, D));
    RAD_ISSUE_N(c, m.op_gkn, L * m.H, heads(c, m.b_vk, 0, L, m), RAD_W(m.g_knw), RAD_W(m.g_knb),
                heads(c, m.b_gk, base, L, m));
}

static void step_query(RadCtx* c, const RadShapeGenBatch* S, const Model& m) {
    const int64_t n = S->n_query, D = m.D, L = m.L;
    const int64_t base = (int64_t)S->lane * L;
    RAD_ISSUE_N(c, m.op_four, n, praw2(S->query, RAD_F32, n, 3), brows(m.b_feat, n));
    lin_issue(c, m.q_proj, n, brows(m.b_feat, n), brows(m.b_qx, n));
    RAD_ISSUE_N(c, m.op_qn1, n, brows(m.b_qx, n), RAD_W(m.q_n1w), RAD_W(m.q_n1b), brows(m.b_qh, n));
    lin_issue(c, m.q_q, n, brows(m.b_qh, n), brows(m.b_qq, n));
    RAD_ISSUE_N(c, m.op_qqn, n * m.H, heads(c, m.b_qq, 0, n, m), RAD_W(m.q_qnw), RAD_W(m.q_qnb),
                heads(c, m.b_qqn, 0, n, m));
    RAD_ISSUE_N(c, m.op_qattn, n, brows(m.b_qqn, n), brow_slice(m.b_gk, base, L, D),
                brow_slice(m.b_gv, base, L, D), praw(S->query_cu, RAD_I32, 2),
                praw(S->lat_cu, RAD_I32, 2), brows(m.b_qatt, n));
    lin_issue(c, m.q_o, n, brows(m.b_qatt, n), brows(m.b_qx, n), brows(m.b_qx, n));
    RAD_ISSUE_N(c, m.op_qn3, n, brows(m.b_qx, n), RAD_W(m.q_n3w), RAD_W(m.q_n3b), brows(m.b_qh, n));
    lin_issue(c, m.q_fc, n, brows(m.b_qh, n), brows(m.b_qff, n));
    lin_issue(c, m.q_out, n, brows(m.b_qff, n), brows(m.b_qx, n), brows(m.b_qx, n));
    RAD_ISSUE_N(c, m.op_qpost, n, brows(m.b_qx, n), RAD_W(m.q_pw), RAD_W(m.q_pb), brows(m.b_qh, n));
    lin_issue(c, m.q_head, n, brows(m.b_qh, n), brows(m.b_qlogit, n));
    RAD_ISSUE_N(c, m.op_qcast, n, brows(m.b_qlogit, n), brows(m.b_field, n));
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model;
    const RadShapeGenBatch* S = batch->sgen_b;
    if (!batch->gen || !S) {
        rad_step_fail(c, "hunyuan3d2 is a shape generator and runs generator passes only");
        return;
    }
    switch (batch->gen) {
    case RAD_GEN_PASS_SCOND:   step_cond(c, S, m); return;
    case RAD_GEN_PASS_SSTEP:   step_dit(c, S, m); return;
    case RAD_GEN_PASS_SLATENT: step_latent(c, S, m); return;
    case RAD_GEN_PASS_SQUERY:  step_query(c, S, m); return;
    default: rad_step_fail(c, "hunyuan3d2: a generator pass of a kind this plugin does not know");
    }
}

/* ================================================================== the probe */
/* The rows of the largest pass: a query pass's points, against a step's two branches. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    const int64_t L = rad_meta_geti(meta, "vae.params.num_latents", 3072);
    out->gen_rows = std::max<int64_t>(kMaxQuery, 2 * (kCond + L));
    out->gen_shape = 1;
    return RAD_OK;
}

}  /* namespace hunyuan3d2 */

RAD_ARCH_PROBE(hunyuan3d2)
RAD_ARCH_PLUGIN(hunyuan3d2, "hunyuan3d2", "", "0.1.0",
                "Hunyuan3D-2's shape model: DINOv2-giant over the picture, a flow-matching DiT of "
                "16 double and 32 single blocks over 3072 latents, and the shape VAE's geometry "
                "decoder sampled at the core's grid points; bf16")
