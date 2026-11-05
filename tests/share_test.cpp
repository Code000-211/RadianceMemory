/* share_test.cpp -- which bytes of a container entry one rank takes (core/format/share.h).
 *
 * The loader reads exactly the rectangles weight_share names, so a wrong one makes a rank serve
 * another rank's rows: a model that loads, runs and is silently wrong (spec §17). Each case is a
 * shape tensor parallelism produces, worked by hand.
 */
#include "rad_test.h"
#include "format/share.h"

#include <cstring>
#include <string>

using namespace rad;

namespace {

/* A declaration of `planes` of `e` over a logical [R, C] weight at rank `world`, as the builder
 * records it: one plane at its own extents, several at the logical ones, this rank's share of
 * them under a shard. */
WeightInfo decl(const RadEncoding& e, std::initializer_list<int> planes, int64_t R, int64_t C,
                int shard, int world, std::initializer_list<int64_t> parts = {}) {
    WeightInfo w;
    w.enc = e;
    w.enc_known = true;
    w.decl.shard = shard;
    w.decl.rank = 2;
    int64_t r = R, c = C;
    if (shard == RAD_SHARD_ROW && world > 1) r = R / world;
    if (shard == RAD_SHARD_COL && world > 1) c = C / world;
    w.decl.shape[0] = r;
    w.decl.shape[1] = c;
    int n = 0;
    for (int64_t p : parts) w.decl.row_parts[n++] = p;
    w.decl.n_row_parts = n;
    for (int k : planes) {
        w.sel[w.n_sel] = k;
        rad_enc_plane_dims(&e.plane[k], r, c, &w.sel_rows[w.n_sel], &w.sel_cols[w.n_sel]);
        ++w.n_sel;
    }
    return w;
}

RadEncoding fp8_block() {
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_add_plane(&e, "codes", RAD_F8E4M3, 1, 1);
    rad_enc_add_plane(&e, "scale", RAD_BF16, 128, 128);
    return e;
}

int share(const WeightInfo& w, int64_t R, int64_t C, int world, int index, PlaneShare* out,
          std::string* why = nullptr) {
    const int64_t shape[2] = { R, C };
    return weight_share(w, w.enc, 2, shape, world, index, out, why);
}

}  /* namespace */

/* Unsharded, a rank takes every plane whole: one rectangle, every byte of every row. */
TEST(an_unsharded_weight_is_every_plane_whole) {
    const RadEncoding e = fp8_block();
    const WeightInfo w = decl(e, { 0, 1 }, 256, 384, RAD_SHARD_NONE, 2);
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(share(w, 256, 384, 2, 1, s), RAD_OK);
    REQUIRE_EQ(s[0].rects.size(), (size_t)1);
    CHECK_EQ(s[0].rects[0].row0, (int64_t)0);
    CHECK_EQ(s[0].rects[0].rows, (int64_t)256);
    CHECK_EQ(s[0].rects[0].bytes, (int64_t)384);
    CHECK_EQ(s[0].dense_bytes(), (int64_t)(256 * 384));
    CHECK_EQ(s[1].rows, (int64_t)2);
    CHECK_EQ(s[1].cols, (int64_t)3);
    CHECK_EQ(s[1].dense_bytes(), (int64_t)(2 * 3 * 2));
}

/* A ROW shard of a block-fp8 linear is rows of every plane, the scales a block of 128 rows each. */
TEST(a_row_shard_takes_rows_of_the_codes_and_their_scale_blocks) {
    const RadEncoding e = fp8_block();
    for (int index = 0; index < 2; ++index) {
        const WeightInfo w = decl(e, { 0, 1 }, 512, 256, RAD_SHARD_ROW, 2);
        PlaneShare s[RAD_ENC_MAX_PLANES];
        REQUIRE_EQ(share(w, 512, 256, 2, index, s), RAD_OK);
        CHECK_EQ(s[0].rects[0].row0, (int64_t)(index * 256));
        CHECK_EQ(s[0].rects[0].rows, (int64_t)256);
        CHECK_EQ(s[1].rects[0].row0, (int64_t)(index * 2));
        CHECK_EQ(s[1].rects[0].rows, (int64_t)2);
        CHECK_EQ(s[1].rects[0].bytes, (int64_t)(2 * 2));
    }
    /* the scale plane declared on its own, as gemm_nt_q's b_scale operand: its own extents */
    const WeightInfo ws = decl(e, { 1 }, 512, 256, RAD_SHARD_ROW, 2);
    CHECK_EQ(ws.sel_rows[0], (int64_t)2);
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(share(ws, 512, 256, 2, 1, s), RAD_OK);
    CHECK_EQ(s[0].rects[0].row0, (int64_t)2);
    CHECK_EQ(s[0].rects[0].rows, (int64_t)2);
}

/* A stacked [gate|up] is cut in each part: rank 1 of 2 takes the second half of gate and the
 * second half of up, never "all of up". */
TEST(a_stacked_weight_is_cut_in_every_part) {
    const RadEncoding e = rad_enc_plain(RAD_BF16);
    /* entry [2 * 512, 64]: gate rows 0..511, up rows 512..1023; a rank's parts are 256 rows each */
    const WeightInfo w = decl(e, { 0 }, 1024, 64, RAD_SHARD_ROW, 2, { 256, 256 });
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(share(w, 1024, 64, 2, 1, s), RAD_OK);
    REQUIRE_EQ(s[0].rects.size(), (size_t)2);
    CHECK_EQ(s[0].rects[0].row0, (int64_t)256);
    CHECK_EQ(s[0].rects[1].row0, (int64_t)768);
    CHECK_EQ(s[0].rows, (int64_t)512);
    /* parts that do not sum to the rank's share are refused */
    const WeightInfo bad = decl(e, { 0 }, 1024, 64, RAD_SHARD_ROW, 2, { 200, 256 });
    std::string why;
    CHECK_EQ(share(bad, 1024, 64, 2, 0, s, &why), RAD_E_SHAPE);
    CHECK(why.find("row parts") != std::string::npos);
}

/* A COL shard is columns of each row, and a column cut has to land on a whole byte. */
TEST(a_col_shard_takes_a_byte_range_of_every_row) {
    const RadEncoding e = rad_enc_plain(RAD_BF16);
    const WeightInfo w = decl(e, { 0 }, 8, 128, RAD_SHARD_COL, 4);
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(share(w, 8, 128, 4, 3, s), RAD_OK);
    REQUIRE_EQ(s[0].rects.size(), (size_t)1);
    CHECK_EQ(s[0].rects[0].rows, (int64_t)8);
    CHECK_EQ(s[0].rects[0].byte0, (int64_t)(96 * 2));
    CHECK_EQ(s[0].rects[0].bytes, (int64_t)(32 * 2));
    CHECK_EQ(s[0].row_bytes, (int64_t)(128 * 2));

    RadEncoding i4;
    rad_enc_clear(&i4);
    rad_enc_copy_str(i4.scheme, "plain");
    rad_enc_add_plane(&i4, "codes", RAD_I4, 1, 1);
    const WeightInfo odd = decl(i4, { 0 }, 4, 6, RAD_SHARD_COL, 2);   /* 3 nibbles a rank */
    std::string why;
    CHECK_EQ(share(odd, 4, 6, 2, 1, s, &why), RAD_E_SHAPE);
    CHECK(why.find("whole bytes") != std::string::npos);
}

/* A cut that splits a scale block belongs to neither rank, and is refused. */
TEST(a_shard_that_splits_a_block_is_refused) {
    const RadEncoding e = fp8_block();
    const WeightInfo w = decl(e, { 0, 1 }, 384, 256, RAD_SHARD_ROW, 2);   /* 192 rows a rank */
    PlaneShare s[RAD_ENC_MAX_PLANES];
    std::string why;
    CHECK_EQ(share(w, 384, 256, 2, 0, s, &why), RAD_E_SHAPE);
}

/* An uneven span: the plugin names this rank's rows of each part, and the scale plane takes the
 * same slice in its own blocks. */
TEST(an_uneven_span_is_the_rows_the_plugin_names) {
    const RadEncoding e = fp8_block();
    WeightInfo w = decl(e, { 0, 1 }, 640, 128, RAD_SHARD_ROW, 2);
    w.decl.shape[0] = 384;
    w.shard_lo = 256;
    w.shard_hi = 640;
    rad_enc_plane_dims(&e.plane[0], 384, 128, &w.sel_rows[0], &w.sel_cols[0]);
    rad_enc_plane_dims(&e.plane[1], 384, 128, &w.sel_rows[1], &w.sel_cols[1]);
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(share(w, 640, 128, 2, 1, s), RAD_OK);
    CHECK_EQ(s[0].rects[0].row0, (int64_t)256);
    CHECK_EQ(s[0].rows, (int64_t)384);
    CHECK_EQ(s[1].rects[0].row0, (int64_t)2);
    CHECK_EQ(s[1].rows, (int64_t)3);
}

/* Per-part spans: a stacked weight whose parts differ in width (the delta net's [q|k|v|z]) takes
 * a span per part, each in its own part's rows. Three ranks over 16 K heads of 128: rank 1 takes
 * K heads [6, 11) of q and of k, and V heads [18, 33) of 48 in v. */
TEST(per_part_spans_cut_each_stacked_part_by_its_own_rows) {
    const RadEncoding e = rad_enc_plain(RAD_BF16);
    /* entry: q 2048 | k 2048 | v 6144 rows; rank 1 of 3 declares 640 + 640 + 1920 */
    WeightInfo w = decl(e, { 0 }, 2048 + 2048 + 6144, 64, RAD_SHARD_ROW, 3, { 640, 640, 1920 });
    w.decl.shape[0] = 640 + 640 + 1920;
    rad_enc_plane_dims(&e.plane[0], w.decl.shape[0], 64, &w.sel_rows[0], &w.sel_cols[0]);
    w.n_part_spans = 3;
    const int64_t lo[3] = { 768, 768, 2304 }, hi[3] = { 1408, 1408, 4224 };
    const int64_t tot[3] = { 2048, 2048, 6144 };
    for (int g = 0; g < 3; ++g) { w.part_lo[g] = lo[g]; w.part_hi[g] = hi[g]; w.part_total[g] = tot[g]; }
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(share(w, 2048 + 2048 + 6144, 64, 3, 1, s), RAD_OK);
    REQUIRE_EQ(s[0].rects.size(), (size_t)3);
    CHECK_EQ(s[0].rects[0].row0, (int64_t)768);
    CHECK_EQ(s[0].rects[0].rows, (int64_t)640);
    CHECK_EQ(s[0].rects[1].row0, (int64_t)(2048 + 768));
    CHECK_EQ(s[0].rects[2].row0, (int64_t)(4096 + 2304));
    CHECK_EQ(s[0].rects[2].rows, (int64_t)1920);
    CHECK_EQ(s[0].rows, (int64_t)3200);
    /* totals that do not sum to the entry are refused */
    w.part_total[2] = 6000;
    std::string why;
    CHECK_EQ(share(w, 2048 + 2048 + 6144, 64, 3, 1, s, &why), RAD_E_SHAPE);
    CHECK(why.find("stated rows") != std::string::npos);
}

/* A per-tensor scale is one value every rank carries whole, once, whatever the codes' cut. */
TEST(a_per_tensor_scale_is_carried_whole_by_every_rank) {
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_add_plane(&e, "codes", RAD_F8E4M3, 1, 1);
    rad_enc_add_plane(&e, "scale", RAD_BF16, 0, 0);
    const WeightInfo w = decl(e, { 0, 1 }, 1000, 64, RAD_SHARD_ROW, 2, { 250, 250 });
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(share(w, 1000, 64, 2, 1, s), RAD_OK);
    CHECK_EQ(s[0].rects.size(), (size_t)2);
    REQUIRE_EQ(s[1].rects.size(), (size_t)1);
    CHECK_EQ(s[1].rows, (int64_t)1);
    CHECK_EQ(s[1].cols, (int64_t)1);
}

/* A rank-1 ROW shard is a contiguous slice of the vector: one value per head. */
TEST(a_sharded_vector_is_a_slice_of_it) {
    const RadEncoding e = rad_enc_plain(RAD_F32);
    WeightInfo w;
    w.enc = e;
    w.enc_known = true;
    w.decl.shard = RAD_SHARD_ROW;
    w.decl.rank = 1;
    w.decl.shape[0] = 16;
    w.n_sel = 1;
    w.sel[0] = 0;
    w.sel_rows[0] = 1;
    w.sel_cols[0] = 16;
    const int64_t shape[1] = { 64 };
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(weight_share(w, e, 1, shape, 4, 2, s, nullptr), RAD_OK);
    CHECK_EQ(s[0].rects[0].byte0, (int64_t)(32 * 4));
    CHECK_EQ(s[0].rects[0].bytes, (int64_t)(16 * 4));
}

/* A rank-1 weight of stacked parts (a fused [value | gate] projection's bias) is cut like its
 * matrix: this rank's slice of each part. A contiguous slice would hand rank 0 every value bias
 * and rank 1 every gate bias. */
TEST(a_stacked_vector_is_cut_in_every_part) {
    const RadEncoding e = rad_enc_plain(RAD_BF16);
    WeightInfo w;
    w.enc = e;
    w.enc_known = true;
    w.decl.shard = RAD_SHARD_ROW;
    w.decl.rank = 1;
    w.decl.shape[0] = 32;
    w.decl.n_row_parts = 2;
    w.decl.row_parts[0] = 16;
    w.decl.row_parts[1] = 16;
    w.n_sel = 1;
    w.sel[0] = 0;
    w.sel_rows[0] = 1;
    w.sel_cols[0] = 32;
    /* entry [64]: values 0..31, gates 32..63; rank 1 owns values 16..31 and gates 48..63 */
    const int64_t shape[1] = { 64 };
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(weight_share(w, e, 1, shape, 2, 1, s, nullptr), RAD_OK);
    REQUIRE_EQ(s[0].rects.size(), (size_t)2);
    CHECK_EQ(s[0].rects[0].byte0, (int64_t)(16 * 2));
    CHECK_EQ(s[0].rects[0].bytes, (int64_t)(16 * 2));
    CHECK_EQ(s[0].rects[1].byte0, (int64_t)(48 * 2));
    CHECK_EQ(s[0].rects[1].bytes, (int64_t)(16 * 2));
    /* parts that do not sum to the rank's share are refused */
    w.decl.row_parts[1] = 8;
    std::string why;
    CHECK_EQ(weight_share(w, e, 1, shape, 2, 0, s, &why), RAD_E_SHAPE);
    CHECK(why.find("declared parts") != std::string::npos);
}

/* A vector's parts may differ in width: a Mamba convolution's bias is [x | B | C], and a rank takes
 * its heads' x and its groups' B and C. The parts lie end to end as one row of the share. */
TEST(a_stacked_vector_may_have_parts_of_different_widths) {
    const RadEncoding e = rad_enc_plain(RAD_BF16);
    WeightInfo w;
    w.enc = e;
    w.enc_known = true;
    w.decl.shard = RAD_SHARD_ROW;
    w.decl.rank = 1;
    w.decl.shape[0] = 12;
    w.decl.n_row_parts = 3;
    w.decl.row_parts[0] = 8;
    w.decl.row_parts[1] = 2;
    w.decl.row_parts[2] = 2;
    w.n_sel = 1;
    w.sel[0] = 0;
    w.sel_rows[0] = 1;
    w.sel_cols[0] = 12;
    /* entry [24]: x 0..15, B 16..19, C 20..23; rank 1 owns x 8..15, B 18..19 and C 22..23 */
    const int64_t shape[1] = { 24 };
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(weight_share(w, e, 1, shape, 2, 1, s, nullptr), RAD_OK);
    REQUIRE_EQ(s[0].rects.size(), (size_t)3);
    CHECK_EQ(s[0].rects[0].byte0, (int64_t)(8 * 2));
    CHECK_EQ(s[0].rects[0].bytes, (int64_t)(8 * 2));
    CHECK_EQ(s[0].rects[1].byte0, (int64_t)(18 * 2));
    CHECK_EQ(s[0].rects[1].bytes, (int64_t)(2 * 2));
    CHECK_EQ(s[0].rects[2].byte0, (int64_t)(22 * 2));
    CHECK_EQ(s[0].rects[2].bytes, (int64_t)(2 * 2));
    CHECK_EQ(s[0].dense_bytes(), (int64_t)(12 * 2));
}

/* A reshape: a conv kernel [O, C, T, H, W] declared as the [O, C*T*H*W] matrix it is. The bytes
 * do not move, so the share is the whole plane in the declared view; a shard of a reshape would
 * be a different cut, so it is refused. */
TEST(a_whole_plane_may_be_declared_under_another_shape) {
    const RadEncoding e = rad_enc_plain(RAD_BF16);
    WeightInfo w = decl(e, { 0 }, 1152, 1536, RAD_SHARD_NONE, 2);
    const int64_t conv[5] = { 1152, 3, 2, 16, 16 };
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(weight_share(w, e, 5, conv, 2, 1, s, nullptr), RAD_OK);
    CHECK_EQ(s[0].rows, (int64_t)1152);
    CHECK_EQ(s[0].cols, (int64_t)1536);
    CHECK_EQ(s[0].dense_bytes(), (int64_t)(1152 * 1536 * 2));

    WeightInfo r = decl(e, { 0 }, 1152, 1536, RAD_SHARD_ROW, 2);
    std::string why;
    CHECK_EQ(weight_share(r, e, 5, conv, 2, 0, s, &why), RAD_E_SHAPE);
}

/* A depthwise conv kernel [C, 1, K] is the [C, K] matrix its inner 1 leaves, so a row shard is
 * contiguous rows of K elements. An inner dimension other than 1 is still a rank-3 weight, and
 * which dimension a rank owns is not the loader's to guess. */
TEST(a_unit_inner_dimension_is_sharded_as_the_matrix_it_leaves) {
    const RadEncoding e = rad_enc_plain(RAD_BF16);
    WeightInfo w = decl(e, { 0 }, 10240, 4, RAD_SHARD_ROW, 2);
    const int64_t conv[3] = { 10240, 1, 4 };
    PlaneShare s[RAD_ENC_MAX_PLANES];
    REQUIRE_EQ(weight_share(w, e, 3, conv, 2, 1, s, nullptr), RAD_OK);
    REQUIRE_EQ(s[0].rects.size(), (size_t)1);
    CHECK_EQ(s[0].rects[0].row0, (int64_t)5120);
    CHECK_EQ(s[0].rects[0].rows, (int64_t)5120);
    CHECK_EQ(s[0].rects[0].byte0, (int64_t)0);
    CHECK_EQ(s[0].rects[0].bytes, (int64_t)(4 * 2));

    const int64_t stack[3] = { 10240, 2, 2 };
    std::string why;
    CHECK_EQ(weight_share(w, e, 3, stack, 2, 0, s, &why), RAD_E_SHAPE);
}

TEST(glm_kda_tp4_takes_each_head_shard_from_all_three_convolution_parts) {
    const auto e = rad_enc_plain(RAD_BF16);
    const auto w = decl(e, { 0 }, 3 * 8192, 4, RAD_SHARD_ROW, 4, { 2048, 2048, 2048 });
    const int64_t shape[] = { 3 * 8192, 1, 4 };
    for (int rank = 0; rank < 4; ++rank) {
        PlaneShare s[RAD_ENC_MAX_PLANES];
        REQUIRE_EQ(weight_share(w, e, 3, shape, 4, rank, s, nullptr), RAD_OK);
        REQUIRE_EQ(s[0].rects.size(), (size_t)3);
        for (int part = 0; part < 3; ++part) {
            CHECK_EQ(s[0].rects[part].row0, (int64_t)(part * 8192 + rank * 2048));
            CHECK_EQ(s[0].rects[part].rows, (int64_t)2048);
            CHECK_EQ(s[0].rects[part].bytes, (int64_t)8);
        }
    }
}

RAD_TEST_MAIN()
