/* radio.h -- Nemotron-H Omni's vision encoder: C-RADIO (v4-H) and the projector after it (mlp1).
 *
 * Per encoder pass (RadBatch::enc). The processor (core/mm, Family::Radio) gives every segment
 * `n_prefix` zero rows ahead of its patches, with coordinate (-1, k):
 *
 *   x  = rows @ W^T                              one GEMM for both projections: a row is a
 *                                                tubelet's frames then a picture's patch, the part
 *                                                not in use zero, and W = [video_embedder |
 *                                                embedder] concatenated at convert; no bias
 *   x += CPE(table, row, col), or prefix[k]      grid_embed_cpe: the 128 x 128 table resampled
 *                                                half-pixel bilinear onto a square of the grid's
 *                                                longer side and cropped to it; a prefix row takes
 *                                                the k-th class or register token
 *   depth times (pre-norm, LayerScale 1):
 *     x += proj(attn(qkv(LN1(x))))               full attention within a segment, no rotary
 *     x += fc2(gelu(fc1(LN2(x))))                exact GELU
 *   y  = each segment's patch rows               gather_rows: the prefix rows dropped
 *   out = l2(relu(l1(rms(y viewed [P/4, 4 * width])))^2)
 *
 * which is transformers' RadioModel (make_preprocessor_external) then NemotronH_Omni's vision
 * projector with vision_final_layernorm off. The pixel shuffle (ps v2) is the view: the processor
 * keeps a 2 x 2 block's patches consecutive, (dy, dx), the order the shuffle concatenates them;
 * the tower is order-independent within a segment (attention is permutation-equivariant, every
 * other layer is per row, the position term is sampled at each row's own coordinate).
 *
 * A pass is one grid (RadEncoderDecl::one_grid), so its segments are equally long and each
 * segment's patch rows sit at a fixed offset, which `step` gathers with slices of one identity
 * index. Replicated, per-request weights and pass-sized activations, as in
 * arch/common/rad_block_vit.h's tower.
 */
#ifndef NEMOTRONH_RADIO_H
#define NEMOTRONH_RADIO_H

#include <arch/rad_arch.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace nemotronh {

using namespace rad::arch;

struct Radio {
    int64_t depth = 0, width = 0, heads = 0, ff = 0, patch = 0, merge = 0, side = 0, npre = 0;
    int64_t tubelet = 0, head_dim = 0, patch_dim = 0, proj_hidden = 0, n_embd = 0, P = 0, R = 0;
    float   eps = 1e-6f;
    bool    on = false;

    struct Layer {
        rad_weight n1w = 0, n1b = 0, n2w = 0, n2b = 0;
        rad_weight qkv_w = 0, qkv_b = 0, out_w = 0, out_b = 0;
        rad_weight up_w = 0, up_b = 0, down_w = 0, down_b = 0;
        rad_op ln1 = 0, qkv = 0, attn = 0, proj = 0, ln2 = 0, up = 0, down = 0;
    };
    std::vector<Layer> L;
    rad_weight w_patch = 0, w_pos = 0, w_pre = 0, mn_w = 0, m1_w = 0, m2_w = 0;
    rad_op op_patch = 0, op_pos = 0, op_seq = 0, op_gather = 0, op_mln = 0, op_m1 = 0;
    rad_op op_act = 0, op_m2 = 0;
    rad_buf b_x = 0, b_h = 0, b_qkv = 0, b_att = 0, b_ff = 0, b_seq = 0, b_g = 0, b_mn = 0;
    rad_buf b_mf = 0, b_ma = 0, b_out = 0;

    /* A tower to declare: the container is a Nemotron-H Omni one, whose vision_config is RADIO's. */
    static bool present(const RadModelMeta* m) {
        return !std::strcmp(rad_meta_gets(m, "vision_config.model_type", ""), "radio") &&
               rad_meta_geti(m, "vision_config.num_hidden_layers", 0) > 0;
    }

    int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm,
                int64_t model_embd, int first_group);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int Radio::declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm,
                          int64_t model_embd, int first_group) {
    on = false;
    if (!present(m) || ctx->max_enc_patches <= 0) return RAD_OK;

    depth   = rad_meta_geti(m, "vision_config.num_hidden_layers", 0);
    width   = rad_meta_geti(m, "vision_config.hidden_size", 0);
    heads   = rad_meta_geti(m, "vision_config.num_attention_heads", 0);
    ff      = (int64_t)std::llround(rad_meta_getf(m, "vision_config.mlp_ratio", 0.0) * (double)width);
    patch   = rad_meta_geti(m, "vision_config.patch_size", 0);
    eps     = (float)rad_meta_getf(m, "vision_config.layer_norm_eps", 1e-6);
    side    = rad_meta_geti(m, "vision_config.max_img_size", 0) / (patch > 0 ? patch : 1);
    npre    = rad_meta_geti(m, "vision_config.num_cls_tokens", 0) +
              rad_meta_geti(m, "vision_config.num_registers", 0);
    tubelet = rad_meta_geti(m, "vision_config.video_temporal_patch_size",
                            rad_meta_geti(m, "video_temporal_patch_size", 0));
    const double ratio = rad_meta_getf(m, "downsample_ratio", 0.0);
    merge   = ratio > 0.0 ? (int64_t)std::llround(1.0 / ratio) : 0;
    proj_hidden = rad_meta_geti(m, "projector_hidden_size", 0);
    n_embd  = model_embd;
    const int64_t chans = rad_meta_geti(m, "vision_config.num_channels", 3);
    if (depth <= 0 || width <= 0 || heads <= 0 || ff <= 0 || patch <= 0 || side <= 0 ||
        npre <= 0 || tubelet <= 0 || merge <= 0 || proj_hidden <= 0 || width % heads ||
        chans != 3) {
        fprintf(stderr, "radiance: nemotronh: the container's RADIO vision_config is incomplete or "
                        "inconsistent (layers %lld, hidden %lld, heads %lld, ff %lld, patch %lld, "
                        "table %lld, prefix %lld, tubelet %lld, merge %lld, projector %lld)\n",
                (long long)depth, (long long)width, (long long)heads, (long long)ff,
                (long long)patch, (long long)side, (long long)npre, (long long)tubelet,
                (long long)merge, (long long)proj_hidden);
        return RAD_E_INVAL;
    }
    /* What this tower serves of RADIO's options: the plain GELU MLP, LayerScale 1 (no stored
     * scales), q/k/v with biases, and no final layer norm before the projector. */
    const char* act = rad_meta_gets(m, "vision_config.hidden_act", "gelu");
    const bool swiglu = rad_meta_geti(m, "vision_config.use_swiglu_ffn", 0) != 0 ||
                        !std::strcmp(rad_meta_gets(m, "vision_config.use_swiglu_ffn", ""), "true");
    const bool final_ln = rad_meta_geti(m, "vision_final_layernorm", 0) != 0 ||
                          !std::strcmp(rad_meta_gets(m, "vision_final_layernorm", ""), "true");
    if (std::strcmp(act, "gelu") || swiglu || final_ln ||
        rad_meta_getf(m, "vision_config.layerscale_value", 1.0) != 1.0) {
        fprintf(stderr, "radiance: nemotronh: this RADIO tower (activation '%s'%s%s, layer scale "
                        "%g) is not the one served: GELU MLP, layer scale 1, no final layer norm\n",
                act, swiglu ? ", SwiGLU" : "", final_ln ? ", final layer norm" : "",
                rad_meta_getf(m, "vision_config.layerscale_value", 1.0));
        return RAD_E_UNSUPPORTED;
    }
    head_dim  = width / heads;
    const int64_t pp = chans * patch * patch;
    patch_dim = pp * (1 + tubelet);
    const int64_t mm = merge * merge, mw = mm * width;

    /* A pass's rows, prefix rows included; the output rows are at most a quarter of them. */
    P = ctx->max_enc_patches;
    const int64_t max_seg = P;
    if (ctx->shape_probe) {
        const int64_t r = std::min<int64_t>(P, ctx->max_tok);
        P = std::max<int64_t>(mm, r / mm * mm);
    }
    R = P / mm;
    if (R <= 0) return RAD_E_SHAPE;
    const int ACC = RAD_ACCESS_PER_REQUEST;

    auto bw = [&](const char* dn, const char* src, std::initializer_list<int64_t> shp,
                  RadWeightGroup g) -> rad_weight {
        if (map_copy(b, dn, src) < 0) return 0;
        return decl_w(b, dn, RAD_BF16, shp, ACC, RAD_SHARD_NONE, g);
    };
    const char* pg = "vision_model.radio_model.model.patch_generator.";
    /* [video_embedder | embedder] along K: one GEMM for a row of either kind. */
    if (map_concat(b, nm.f("v.patch.weight"),
                   { nm.ckpt("%svideo_embedder.weight", pg), nm.ckpt("%sembedder.weight", pg) },
                   1) < 0)
        return RAD_E_INVAL;
    /* Whether the tower is here at all: its stem's encoding, which convert answers from the
     * checkpoint and load from the container alike. A text-only cut (a few blocks for a host
     * reference) or a container converted without it serves text; the map left behind is inert. */
    {
        RadEncoding enc;
        if (rad_weight_encoding(b, nm.f("v.patch.weight"), &enc, nullptr, nullptr) != RAD_OK) {
            rad_note(b, "nemotronh: vision_config names a RADIO tower its weights are not here "
                        "for; serving text");
            return RAD_OK;
        }
    }
    w_patch = decl_w(b, nm.f("v.patch.weight"), RAD_BF16, { width, patch_dim }, ACC,
                     RAD_SHARD_NONE, grp_model());
    /* The checkpoint's [1, side * side, width] read as the [side * side, width] table. */
    w_pos = bw(nm.f("v.pos.weight"), nm.ckpt("%spos_embed", pg), { side * side, width }, grp_model());
    w_pre = bw(nm.f("v.prefix.weight"), nm.ckpt("%scls_token.token", pg), { npre, width },
               grp_model());
    L.assign((size_t)depth, Layer{});
    for (int64_t i = 0; i < depth; ++i) {
        Layer& l = L[(size_t)i];
        const RadWeightGroup g = grp_layer(first_group + (int)i);
        const int li = (int)i;
        const char* e = "vision_model.radio_model.model.blocks.%d.%s";
        l.n1w    = bw(nm.f("v.blk.%d.norm1.weight", li), nm.ckpt(e, li, "norm1.weight"), { width }, g);
        l.n1b    = bw(nm.f("v.blk.%d.norm1.bias", li),   nm.ckpt(e, li, "norm1.bias"),   { width }, g);
        l.qkv_w  = bw(nm.f("v.blk.%d.qkv.weight", li),   nm.ckpt(e, li, "attn.qkv.weight"), { 3 * width, width }, g);
        l.qkv_b  = bw(nm.f("v.blk.%d.qkv.bias", li),     nm.ckpt(e, li, "attn.qkv.bias"),   { 3 * width }, g);
        l.out_w  = bw(nm.f("v.blk.%d.proj.weight", li),  nm.ckpt(e, li, "attn.proj.weight"), { width, width }, g);
        l.out_b  = bw(nm.f("v.blk.%d.proj.bias", li),    nm.ckpt(e, li, "attn.proj.bias"),   { width }, g);
        l.n2w    = bw(nm.f("v.blk.%d.norm2.weight", li), nm.ckpt(e, li, "norm2.weight"), { width }, g);
        l.n2b    = bw(nm.f("v.blk.%d.norm2.bias", li),   nm.ckpt(e, li, "norm2.bias"),   { width }, g);
        l.up_w   = bw(nm.f("v.blk.%d.fc1.weight", li),   nm.ckpt(e, li, "mlp.fc1.weight"), { ff, width }, g);
        l.up_b   = bw(nm.f("v.blk.%d.fc1.bias", li),     nm.ckpt(e, li, "mlp.fc1.bias"),   { ff }, g);
        l.down_w = bw(nm.f("v.blk.%d.fc2.weight", li),   nm.ckpt(e, li, "mlp.fc2.weight"), { width, ff }, g);
        l.down_b = bw(nm.f("v.blk.%d.fc2.bias", li),     nm.ckpt(e, li, "mlp.fc2.bias"),   { width }, g);
        if (!l.n1w || !l.n1b || !l.n2w || !l.n2b || !l.qkv_w || !l.qkv_b || !l.out_w ||
            !l.out_b || !l.up_w || !l.up_b || !l.down_w || !l.down_b)
            return RAD_E_INVAL;
    }
    mn_w = bw(nm.f("v.proj.norm.weight"), nm.ckpt("mlp1.0.weight"), { mw }, grp_model());
    m1_w = bw(nm.f("v.proj.fc1.weight"),  nm.ckpt("mlp1.1.weight"), { proj_hidden, mw }, grp_model());
    m2_w = bw(nm.f("v.proj.fc2.weight"),  nm.ckpt("mlp1.3.weight"), { n_embd, proj_hidden }, grp_model());
    if (!w_patch || !w_pos || !w_pre || !mn_w || !m1_w || !m2_w) return RAD_E_INVAL;

    /* ---- activations. `g` holds the gathered patch rows at `width` and is read by the norm as
     * R rows of mm * width (the pixel shuffle's view, no copy). */
    b_x   = decl_b(b, nm.f("v.x"),   RAD_BF16, { P, width });
    b_h   = decl_b(b, nm.f("v.h"),   RAD_BF16, { P, width });
    b_qkv = decl_b(b, nm.f("v.qkv"), RAD_BF16, { P, 3 * width });
    b_att = decl_b(b, nm.f("v.att"), RAD_BF16, { P, width });
    b_ff  = decl_b(b, nm.f("v.ff"),  RAD_BF16, { P, ff });
    b_seq = decl_b(b, nm.f("v.seq"), RAD_I32,  { P });
    b_g   = decl_b(b, nm.f("v.g"),   RAD_BF16, { R, mw });
    b_mn  = decl_b(b, nm.f("v.mn"),  RAD_BF16, { R, mw });
    b_mf  = decl_b(b, nm.f("v.mf"),  RAD_BF16, { R, proj_hidden });
    b_ma  = decl_b(b, nm.f("v.ma"),  RAD_BF16, { R, proj_hidden });
    b_out = decl_b(b, nm.f("v.out"), RAD_BF16, { R, n_embd });
    if (!b_x || !b_h || !b_qkv || !b_att || !b_ff || !b_seq || !b_g || !b_mn || !b_mf || !b_ma ||
        !b_out)
        return RAD_E_INVAL;

    auto gemm_b = [&](int64_t rows, int64_t N, int64_t K, const char* a, rad_weight w,
                      rad_weight bias) -> rad_op {
        return RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16"), RAD_STR("act", a)),
                      RAD_WEIGHTS(w, bias));
    };
    auto gemm = [&](int64_t rows, int64_t N, int64_t K, rad_weight w) -> rad_op {
        return RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto lnorm = [&](int64_t rows, rad_weight w, rad_weight bias) -> rad_op {
        return RAD_OP(b, "layernorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", width),
                                 RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w, bias));
    };

    op_patch = rw(b, gemm(P, width, patch_dim, w_patch), {}, { b_x });
    op_pos = rw(b, RAD_OP(b, "grid_embed_cpe",
                          RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                     RAD_INT("side", side), RAD_INT("n_prefix", npre),
                                     RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w_pos, w_pre)),
                { b_x }, { b_x });
    op_seq = rw(b, RAD_OP(b, "iota",
                          RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("start", 0), RAD_INT("step", 1)),
                          RAD_NOWEIGHTS),
                {}, { b_seq });
    for (Layer& l : L) {
        l.ln1  = rw(b, lnorm(P, l.n1w, l.n1b), { b_x }, { b_h });
        l.qkv  = rw(b, gemm_b(P, 3 * width, width, "none", l.qkv_w, l.qkv_b), { b_h }, { b_qkv });
        l.attn = rw(b, RAD_OP(b, "attn_dense",
                              RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", head_dim),
                                         RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                         RAD_STR("dtype", "bf16"),
                                         RAD_F64("scale", 1.0 / std::sqrt((double)head_dim)),
                                         RAD_INT("max_seqlen", max_seg)),
                              RAD_NOWEIGHTS),
                    { b_qkv }, { b_att });
        l.proj = rw(b, gemm_b(P, width, width, "none", l.out_w, l.out_b), { b_att, b_x }, { b_x });
        l.ln2  = rw(b, lnorm(P, l.n2w, l.n2b), { b_x }, { b_h });
        l.up   = rw(b, gemm_b(P, ff, width, "gelu", l.up_w, l.up_b), { b_h }, { b_ff });
        l.down = rw(b, gemm_b(P, width, ff, "none", l.down_w, l.down_b), { b_ff, b_x }, { b_x });
    }
    op_gather = rw(b, RAD_OP(b, "gather_rows",
                             RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { b_x, b_seq }, { b_g });
    op_mln = rw(b, RAD_OP(b, "rmsnorm",
                          RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", mw), RAD_F64("eps", 1e-5),
                                     RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(mn_w)),
                { b_g }, { b_mn });
    op_m1  = rw(b, gemm(R, proj_hidden, mw, m1_w), { b_mn }, { b_mf });
    /* relu(x)^2: act_mul's "relu" with the one tensor in both operands. */
    op_act = rw(b, RAD_OP(b, "act_mul",
                          RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", proj_hidden),
                                     RAD_STR("act", "relu"), RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS),
                { b_mf }, { b_ma });
    op_m2  = rw(b, gemm(R, n_embd, proj_hidden, m2_w), { b_ma }, { b_out });

    if (!ctx->shape_probe) {
        RadEncoderDecl e{};
        e.name        = "vision";
        e.modalities  = RAD_MM_IMAGE | RAD_MM_VIDEO;
        e.patch_dim   = patch_dim;
        e.merge       = mm;
        e.max_patches = P;
        e.n_embd      = n_embd;
        e.out         = b_out;
        e.one_grid    = 1;
        RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        rad_note(b, "RADIO tower: %lld blocks of %lld (%lld heads x %lld), ff %lld, patch %lld "
                    "(tubelet %lld), %lld x %lld position table, %lld prefix rows; projector %lld "
                    "-> %lld -> %lld; %lld rows a pass",
                 (long long)depth, (long long)width, (long long)heads, (long long)head_dim,
                 (long long)ff, (long long)patch, (long long)tubelet, (long long)side,
                 (long long)side, (long long)npre, (long long)mw, (long long)proj_hidden,
                 (long long)n_embd, (long long)P);
    }
    on = true;
    return RAD_OK;
}

inline void Radio::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t Pn = batch->enc_n_patch, S = batch->enc_n_seg;
    const int64_t mm = merge * merge;
    const int64_t gh = S > 0 && batch->enc_seg_h ? batch->enc_seg_h[0] : 0;
    const int64_t gw = S > 0 && batch->enc_seg_w ? batch->enc_seg_w[0] : 0;
    const int64_t seg = npre + gh * gw, Rn = S * gh * gw / (mm > 0 ? mm : 1);
    if (!on || Pn <= 0 || Pn > P || S <= 0 || gh <= 0 || gw <= 0 || (gh * gw) % mm ||
        S * seg != Pn || Rn > R) {
        rad_step_fail(c, "nemotronh: an encoder pass this program's RADIO tower cannot serve: no "
                         "tower, more rows than it was declared for, or segments that are not one "
                         "grid of whole merged rows behind their prefix rows");
        return;
    }
    const RadOperand pix = praw2(batch->enc_pixels, RAD_BF16, Pn, patch_dim);
    const RadOperand crd = praw2(batch->enc_coord, RAD_I32, 4, Pn);
    const RadOperand cu  = praw(batch->enc_cu, RAD_I32, S + 1);

    RAD_ISSUE_N(c, op_patch, Pn, pix, RAD_W(w_patch), brows(b_x, Pn));
    RAD_ISSUE_N(c, op_pos, Pn, brows(b_x, Pn), RAD_W(w_pos), crd, RAD_W(w_pre));
    RAD_ISSUE_N(c, op_seq, Pn, brows(b_seq, Pn));
    for (const Layer& l : L) {
        RAD_ISSUE_N(c, l.ln1, Pn, brows(b_x, Pn), RAD_W(l.n1w), RAD_W(l.n1b), brows(b_h, Pn));
        RAD_ISSUE_N(c, l.qkv, Pn, brows(b_h, Pn), RAD_W(l.qkv_w), RAD_W(l.qkv_b), RAD_NONE,
                    brows(b_qkv, Pn));
        RAD_ISSUE_N(c, l.attn, Pn,
                    bcol(b_qkv, 0, width, Pn), bcol(b_qkv, width, width, Pn),
                    bcol(b_qkv, 2 * width, width, Pn), cu, brows(b_att, Pn));
        RAD_ISSUE_N(c, l.proj, Pn, brows(b_att, Pn), RAD_W(l.out_w), RAD_W(l.out_b),
                    brows(b_x, Pn), brows(b_x, Pn));
        RAD_ISSUE_N(c, l.ln2, Pn, brows(b_x, Pn), RAD_W(l.n2w), RAD_W(l.n2b), brows(b_h, Pn));
        RAD_ISSUE_N(c, l.up, Pn, brows(b_h, Pn), RAD_W(l.up_w), RAD_W(l.up_b), RAD_NONE,
                    brows(b_ff, Pn));
        RAD_ISSUE_N(c, l.down, Pn, brows(b_ff, Pn), RAD_W(l.down_w), RAD_W(l.down_b),
                    brows(b_x, Pn), brows(b_x, Pn));
    }
    /* Each segment's patch rows, behind its prefix rows, into `g` at width: segment s's are rows
     * [s * seg + npre, (s + 1) * seg) of x, which the identity index names by a slice. */
    const int64_t hw = gh * gw;
    for (int64_t s = 0; s < S; ++s)
        RAD_ISSUE_N(c, op_gather, hw, brows(b_x, Pn), brow_slice(b_seq, s * seg + npre, hw, 1),
                    brow_slice(b_g, s * hw / mm, hw / mm, mm * width));
    RAD_ISSUE_N(c, op_mln, Rn, brows(b_g, Rn), RAD_W(mn_w), brows(b_mn, Rn));
    RAD_ISSUE_N(c, op_m1, Rn, brows(b_mn, Rn), RAD_W(m1_w), brows(b_mf, Rn));
    RAD_ISSUE_N(c, op_act, Rn, brows(b_mf, Rn), brows(b_mf, Rn), brows(b_ma, Rn));
    RAD_ISSUE_N(c, op_m2, Rn, brows(b_ma, Rn), RAD_W(m2_w), brows(b_out, Rn));
}

}  /* namespace nemotronh */

#endif /* NEMOTRONH_RADIO_H */
