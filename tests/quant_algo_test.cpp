/* quant_algo_test.cpp -- libquant's calibrated algorithms (GPTQ act order, AWQ, AutoRound,
 * ParoQuant), each against round-to-nearest on the same grid.
 *
 * The metric is what they all minimise: the layer's output error, the quadratic form
 * sum_r (w_hat_r - w_r) H (w_hat_r - w_r)^T, where H is the Gram of a synthetic input whose
 * channels are as uneven as a real layer's (a few carry most of the energy; neighbours
 * correlate). The weight is decoded through the core's decoder, so a transform or column scale
 * the reader fails to undo shows up as an error of order one. Each algorithm also runs at one and
 * four threads and must produce the same bytes, since its output is a file.
 */
#include "quant_fixture.h"

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace rad;
using namespace qfix;

namespace {

constexpr int64_t kR = 64, kC = 256, kT = 4096;

/* The input: per-channel energy spread over two decades, a few channels ten times the rest, and
 * each channel mixed with its neighbours. */
std::vector<float> gram() {
    const std::vector<float> z = draw(kT * kC, 101);
    std::vector<float> amp((size_t)kC);
    for (int64_t c = 0; c < kC; ++c)
        amp[(size_t)c] = (float)std::pow(10.0, (double)((c * 37) % kC) / kC * 2.0) *
                         ((c % 31 == 3) ? 10.0f : 1.0f);
    std::vector<float> x((size_t)(kT * kC));
    for (int64_t t = 0; t < kT; ++t)
        for (int64_t c = 0; c < kC; ++c) {
            const float* zr = &z[(size_t)(t * kC)];
            const float v = zr[c] + 0.5f * zr[(c + 1) % kC] + 0.25f * zr[(c + 7) % kC];
            x[(size_t)(t * kC + c)] = v * amp[(size_t)c];
        }
    std::vector<double> h((size_t)(kC * kC), 0.0);
    for (int64_t t = 0; t < kT; ++t)
        for (int64_t i = 0; i < kC; ++i) {
            const double xi = x[(size_t)(t * kC + i)];
            for (int64_t j = 0; j < kC; ++j) h[(size_t)(i * kC + j)] += xi * x[(size_t)(t * kC + j)];
        }
    std::vector<float> hf(h.size());
    for (size_t i = 0; i < h.size(); ++i) hf[i] = (float)h[i];
    return hf;
}

const std::vector<float>& H() {
    static const std::vector<float> h = gram();
    return h;
}

/* A calibration directory holding kName's Gram, made once and removed at exit. */
const std::string& calib_dir() {
    static const struct Dir {
        TempDir t{ "rad_quant_algo" };
        std::string d, gram;
        Dir() {
            if (t.path.empty()) return;
            d = t.path;
            gram = d + "/gram.r0.blk.2.attn_q.bin";
            write_gram(gram, kC, kT, H());
        }
    } dir;
    return dir.d;
}

/* Byte-identical results: as many planes, each equal. */
bool same_planes(const Quantised& a, const Quantised& b) {
    if (a.enc.n_planes != b.enc.n_planes) return false;
    for (int i = 0; i < a.enc.n_planes; ++i)
        if (a.planes[(size_t)i] != b.planes[(size_t)i]) return false;
    return true;
}

double out_err(const std::vector<float>& w, const std::vector<float>& d) {
    double e = 0.0;
    const std::vector<float>& h = H();
    for (int64_t r = 0; r < kR; ++r)
        for (int64_t i = 0; i < kC; ++i) {
            const double di = (double)d[(size_t)(r * kC + i)] - w[(size_t)(r * kC + i)];
            if (di == 0.0) continue;
            double hi = 0.0;
            for (int64_t j = 0; j < kC; ++j)
                hi += (double)h[(size_t)(i * kC + j)] *
                      ((double)d[(size_t)(r * kC + j)] - w[(size_t)(r * kC + j)]);
            e += di * hi;
        }
    return e;
}

const std::vector<float>& weight() {
    static const std::vector<float> w = draw(kR * kC, 102, 6.0f);
    return w;
}

const char* kName = "blk.2.attn_q.weight";

/* The thread count of the OpenMP runtime the plugin loaded, so a run can be repeated at another. */
void threads(int n) {
    using Fn = void (*)(int);
    if (Fn f = (Fn)dlsym(RTLD_DEFAULT, "omp_set_num_threads")) f(n);
}

/* rtn, then the algorithm, on the same grid; the algorithm decoded, its output error below rtn's
 * by `margin`, and its bytes the same at one thread and at four. */
void beats_rtn(const char* algo, Opts grid, Opts extra, double margin, const char* want_enc) {
    const std::vector<float>& w = weight();
    Quantised r = quantise(quantizer("rtn"), grid, kName, kR, kC, w);
    REQUIRE_EQ(r.status, RAD_OK);
    Opts o = grid;
    for (size_t i = 0; i < extra.p.size(); ++i) {
        if (extra.p[i].kind == RAD_P_STR) o.s(extra.keys[i].c_str(), extra.vals[i].c_str());
        else if (extra.p[i].kind == RAD_P_INT) o.i(extra.keys[i].c_str(), extra.p[i].ival);
        else o.f(extra.keys[i].c_str(), extra.p[i].dval);
    }
    o.s("calib", calib_dir().c_str());
    threads(1);
    Quantised a = quantise(quantizer(algo), o, kName, kR, kC, w);
    threads(4);
    Quantised b = quantise(quantizer(algo), o, kName, kR, kC, w);
    REQUIRE_EQ(a.status, RAD_OK);
    REQUIRE_EQ(b.status, RAD_OK);
    if (want_enc) CHECK_EQ(enc_name(a.enc), std::string(want_enc));
    CHECK(same_planes(a, b));
    const std::vector<float> dr = decode(r, kR, kC), da = decode(a, kR, kC);
    REQUIRE(!da.empty());
    const double er = out_err(w, dr), ea = out_err(w, da);
    std::fprintf(stderr, "    %s: output error %.4g of rtn's, weight error %.4f (rtn %.4f)\n",
                 algo, ea / er, rel_err(w, da), rel_err(w, dr));
    CHECK(ea < er * margin);
}

}  /* namespace */

TEST(the_calibrated_quantisers_are_in_the_registry) {
    NEED_LIBQUANT();
    for (const char* q : { "awq", "autoround", "paroquant", "cast" }) CHECK(quantizer(q) != nullptr);
}

TEST(gptq_act_order_stores_the_permutation_it_quantised_in) {
    NEED_LIBQUANT();
    const std::vector<float>& w = weight();
    Opts o;
    o.s("codes", "i4").i("group", 32).s("scale", "f16").i("act_order", 1)
     .s("calib", calib_dir().c_str());
    Quantised q = quantise(quantizer("gptq"), o, kName, kR, kC, w);
    REQUIRE_EQ(q.status, RAD_OK);
    CHECK_EQ(enc_name(q.enc), std::string("i4*f16[1x32]/perm"));
    /* the permutation is the columns by descending Hessian diagonal */
    std::vector<int32_t> perm((size_t)kC);
    std::memcpy(perm.data(), q.planes[2].data(), (size_t)kC * 4);
    int bad = 0;
    for (int64_t j = 1; j < kC; ++j)
        bad += H()[(size_t)(perm[(size_t)j - 1] * (kC + 1))] < H()[(size_t)(perm[(size_t)j] * (kC + 1))];
    CHECK_EQ(bad, 0);
    Opts n;
    n.s("codes", "i4").i("group", 32).s("scale", "f16").s("calib", calib_dir().c_str());
    Opts m;
    m.s("codes", "i4").i("group", 32).s("scale", "f16");
    Quantised g = quantise(quantizer("gptq"), n, kName, kR, kC, w);
    Quantised r = quantise(quantizer("rtn"), m, kName, kR, kC, w);
    const double ea = out_err(w, decode(q, kR, kC)), eg = out_err(w, decode(g, kR, kC));
    const double er = out_err(w, decode(r, kR, kC));
    std::fprintf(stderr, "    output error of rtn's: gptq %.4f, gptq act order %.4f\n", eg / er,
                 ea / er);
    CHECK(ea < er * 0.8);
}

/* The width-n Hadamard over each n-block of a row, unnormalised: libquant's own butterflies, in
 * its order, so a file this writes is the bytes libquant would make of the model-domain one. */
void fwht_rows(std::vector<double>& h, int64_t k, int64_t n) {
    for (int64_t r = 0; r < k; ++r) {
        double* v = h.data() + r * k;
        for (int64_t b0 = 0; b0 < k; b0 += n)
            for (int64_t s = 1; s < n; s *= 2)
                for (int64_t i = b0; i < b0 + n; i += 2 * s)
                    for (int64_t j = i; j < i + s; ++j) {
                        const double x = v[j], y = v[j + s];
                        v[j] = x + y;
                        v[j + s] = x - y;
                    }
    }
}

TEST(a_hessian_is_brought_into_the_domain_its_grid_quantises_in) {
    /* The same Gram as the model's input (domain 0, a bf16 model's tap) and rotated by the
     * width-128 Hadamard (domain 128, a rotated container's tap). A rotated grid must quantise
     * byte-identically against either file, since the reader rotates the first exactly as the
     * second was rotated; an unrotated grid against the rotated file is checked below. */
    NEED_LIBQUANT();
    const std::vector<float>& hf = H();
    std::vector<double> h(hf.begin(), hf.end());
    fwht_rows(h, kC, 128);
    std::vector<double> t((size_t)(kC * kC));
    for (int64_t r = 0; r < kC; ++r)
        for (int64_t c = 0; c < kC; ++c) t[(size_t)(c * kC + r)] = h[(size_t)(r * kC + c)];
    fwht_rows(t, kC, 128);
    std::vector<float> rot((size_t)(kC * kC));
    for (int64_t r = 0; r < kC; ++r)
        for (int64_t c = 0; c < kC; ++c) rot[(size_t)(r * kC + c)] = (float)t[(size_t)(c * kC + r)];

    const TempDir td("rad_quant_dom");
    REQUIRE(!td.path.empty());
    const std::string dm = td.path;
    REQUIRE(write_gram(dm + "/gram.r0.blk.2.attn_q.bin", kC, kT, rot, 128));

    auto run = [&](const std::string& dir, bool rotated) {
        Opts o;
        o.s("codes", "i4").i("group", 128).s("scale", "f16").s("calib", dir.c_str());
        if (rotated) o.s("transform", "fwht128");
        return quantise(quantizer("gptq"), o, kName, kR, kC, weight());
    };
    const Quantised a = run(calib_dir(), true), b = run(dm, true);
    REQUIRE_EQ(a.status, RAD_OK);
    REQUIRE_EQ(b.status, RAD_OK);
    CHECK(same_planes(a, b));

    /* The way back: a rotated file read for an unrotated grid is the model's Gram up to the
     * rounding of a rotation there and back, so the two results are nearly equal. */
    const Quantised c = run(calib_dir(), false), d = run(dm, false);
    REQUIRE_EQ(c.status, RAD_OK);
    REQUIRE_EQ(d.status, RAD_OK);
    const double ec = out_err(weight(), decode(c, kR, kC)), ed = out_err(weight(), decode(d, kR, kC));
    std::fprintf(stderr, "    unrotated grid: model-domain %.6g, back from rotated %.6g\n", ec, ed);
    CHECK(std::fabs(ec - ed) <= 1e-3 * ec);
}

TEST(an_expert_reads_its_own_hessian_before_its_layers) {
    /* A layer's pooled Gram and one expert's own (the pooled one with its channels reversed). An
     * expert with its own file quantises exactly as against that Gram alone, whatever rank wrote
     * it. An expert with no own file, or one whose file has fewer rows than libquant trusts,
     * quantises exactly as against the pooled Gram. The two results must differ, or the test
     * would prove nothing. */
    NEED_LIBQUANT();
    const std::vector<float>& pool = H();
    std::vector<float> own((size_t)(kC * kC));
    for (int64_t i = 0; i < kC; ++i)
        for (int64_t j = 0; j < kC; ++j)
            own[(size_t)(i * kC + j)] = pool[(size_t)((kC - 1 - i) * kC + (kC - 1 - j))];

    const TempDir t1("rad_quant_exp"), t2("rad_quant_own"), t3("rad_quant_pool");
    REQUIRE(!t1.path.empty() && !t2.path.empty() && !t3.path.empty());
    const std::string both = t1.path, only_own = t2.path, only_pool = t3.path;
    const std::vector<std::string> files = {
        both + "/gram.r0.blk.2.ffn_down_exps.bin", both + "/gram.r1.blk.2.ffn_down_exps.7.bin",
        both + "/gram.r0.blk.2.ffn_down_exps.9.bin", only_own + "/gram.r0.blk.2.ffn_down_exps.bin",
        only_pool + "/gram.r0.blk.2.ffn_down_exps.bin" };
    REQUIRE(write_gram(files[0], kC, kT, pool));
    /* libquant's lq::kOwnRowsPerK: the rows an own Gram needs, in units of K */
    constexpr int64_t kOwnRowsPerK = 4;
    REQUIRE(write_gram(files[1], kC, kOwnRowsPerK * kC, own));
    REQUIRE(write_gram(files[2], kC, kOwnRowsPerK * kC - 1, own));
    REQUIRE(write_gram(files[3], kC, kT, own));
    REQUIRE(write_gram(files[4], kC, kT, pool));

    auto run = [&](const std::string& dir, const char* name) -> Quantised {
        Opts o;
        o.s("codes", "i4").i("group", 32).s("scale", "f16").s("calib", dir.c_str());
        Quantised q = quantise(quantizer("gptq"), o, name, kR, kC, weight());
        CHECK_EQ(q.status, RAD_OK);
        return q;
    };
    const char* e7 = "blk.2.ffn_down_exps.7.weight";
    const char* e9 = "blk.2.ffn_down_exps.9.weight";
    const char* e11 = "blk.2.ffn_down_exps.11.weight";
    CHECK(same_planes(run(both, e7), run(only_own, e7)));
    CHECK(!same_planes(run(both, e7), run(only_pool, e7)));
    CHECK(same_planes(run(both, e9), run(only_pool, e9)));
    CHECK(same_planes(run(both, e11), run(only_pool, e11)));
}

TEST(a_vocabulary_heads_width_class_reads_the_heads_hessian) {
    /* A calibration run records the lm_head's input as `output`; the vocabulary-weighted head's
     * classes (output.k3, output.k4, ...) are rows of that one weight and quantise against it, as
     * a weight named `output` does. With no Gram at all GPTQ rounds to nearest, which differs. */
    NEED_LIBQUANT();
    const TempDir t("rad_quant_vhead"), e("rad_quant_novhead");
    REQUIRE(!t.path.empty() && !e.path.empty());
    REQUIRE(write_gram(t.path + "/gram.r0.output.bin", kC, kT, H()));
    auto run = [&](const std::string& dir, const char* name) -> Quantised {
        Opts o;
        o.s("codes", "i4").i("group", 32).s("scale", "f16").s("calib", dir.c_str());
        Quantised q = quantise(quantizer("gptq"), o, name, kR, kC, weight());
        CHECK_EQ(q.status, RAD_OK);
        return q;
    };
    CHECK(same_planes(run(t.path, "output.k4.weight"), run(t.path, "output.weight")));
    CHECK(!same_planes(run(t.path, "output.k4.weight"), run(e.path, "output.k4.weight")));
}

TEST(gptq_through_lapack_matches_the_loops) {
    /* Where OpenBLAS was found the factorisations go through LAPACK; RAD_LQ_LAPACK=0 takes the
     * single-threaded loops, the reference. The same Gram in two directories (the factor cache is
     * keyed by directory), one run each way: the output error agrees to rounding. Codes may
     * differ where error feedback lands on a rounding boundary. Without OpenBLAS both runs take
     * the loops and agree trivially. */
    NEED_LIBQUANT();
    const TempDir fast_dir("rad_quant_lapack"), loop_dir("rad_quant_loops");
    REQUIRE(!fast_dir.path.empty() && !loop_dir.path.empty());
    REQUIRE(write_gram(fast_dir.path + "/gram.r0.blk.2.attn_q.bin", kC, kT, H()));
    REQUIRE(write_gram(loop_dir.path + "/gram.r0.blk.2.attn_q.bin", kC, kT, H()));
    auto run = [&](const std::string& dir) {
        Opts o;
        o.s("codes", "i4").i("group", 32).s("scale", "f16").s("calib", dir.c_str());
        return quantise(quantizer("gptq"), o, kName, kR, kC, weight());
    };
    const Quantised fast = run(fast_dir.path);
    setenv("RAD_LQ_LAPACK", "0", 1);
    const Quantised loops = run(loop_dir.path);
    unsetenv("RAD_LQ_LAPACK");
    REQUIRE_EQ(fast.status, RAD_OK);
    REQUIRE_EQ(loops.status, RAD_OK);
    const double ef = out_err(weight(), decode(fast, kR, kC));
    const double el = out_err(weight(), decode(loops, kR, kC));
    std::fprintf(stderr, "    LAPACK %.9g, loops %.9g\n", ef, el);
    CHECK(std::fabs(ef - el) <= 1e-6 * el);
}

TEST(a_column_split_hessian_is_assembled_from_its_rank_parts) {
    /* Under tensor parallelism a down projection's tap sees one rank's K / 2 input columns, and each
     * rank writes the Gram of its own as `<stem>.part`. GPTQ against the two parts must quantise
     * exactly as against one whole file holding them block diagonally (the cross-rank products
     * zero, never formed). Parts that do not cover K are reported and the weight is rounded to
     * nearest, as with no calibration at all. */
    NEED_LIBQUANT();
    const std::vector<float>& h = H();
    const int64_t k2 = kC / 2;
    std::vector<float> p0((size_t)(k2 * k2)), p1((size_t)(k2 * k2)), whole((size_t)(kC * kC), 0.0f);
    for (int64_t i = 0; i < k2; ++i)
        for (int64_t j = 0; j < k2; ++j) {
            p0[(size_t)(i * k2 + j)] = h[(size_t)(i * kC + j)];
            p1[(size_t)(i * k2 + j)] = h[(size_t)((k2 + i) * kC + k2 + j)];
            whole[(size_t)(i * kC + j)] = p0[(size_t)(i * k2 + j)];
            whole[(size_t)((k2 + i) * kC + k2 + j)] = p1[(size_t)(i * k2 + j)];
        }
    const TempDir tp("rad_quant_parts"), tw("rad_quant_whole"), ts("rad_quant_short"),
                  tn("rad_quant_none");
    REQUIRE(!tp.path.empty() && !tw.path.empty() && !ts.path.empty() && !tn.path.empty());
    REQUIRE(write_gram(tp.path + "/gram.r0.blk.2.ffn_down.part.bin", k2, kT, p0));
    REQUIRE(write_gram(tp.path + "/gram.r1.blk.2.ffn_down.part.bin", k2, kT, p1));
    REQUIRE(write_gram(tw.path + "/gram.r0.blk.2.ffn_down.bin", kC, kT, whole));
    REQUIRE(write_gram(ts.path + "/gram.r0.blk.2.ffn_down.part.bin", k2, kT, p0));

    auto run = [&](const std::string& dir) -> Quantised {
        Opts o;
        o.s("codes", "i4").i("group", 32).s("scale", "f16").s("calib", dir.c_str());
        Quantised q = quantise(quantizer("gptq"), o, "blk.2.ffn_down.weight", kR, kC, weight());
        CHECK_EQ(q.status, RAD_OK);
        return q;
    };
    const Quantised parts = run(tp.path), one = run(tw.path), short_ = run(ts.path), none = run(tn.path);
    CHECK(same_planes(parts, one));
    CHECK(!same_planes(parts, none));
    CHECK(same_planes(short_, none));
}

TEST(awq_scales_the_columns_the_input_says_matter) {
    NEED_LIBQUANT();
    Opts g, x;
    g.s("codes", "i4").i("group", 64).s("scale", "f16");
    beats_rtn("awq", g, x, 0.9, "i4*f16[1x64]*f16[*x1]");
    /* asymmetric, and over a codebook */
    Opts g2, g3;
    g2.s("codes", "u3").s("zero", "u4").i("group", 64).s("scale", "f16");
    beats_rtn("awq", g2, x, 0.95, nullptr);
    g3.s("table", "nf4").i("group", 64).s("scale", "bf16");
    beats_rtn("awq", g3, x, 0.95, nullptr);
}

TEST(autoround_descends_below_round_to_nearest) {
    NEED_LIBQUANT();
    Opts g, x, g2;
    g.s("codes", "i4").i("group", 32).s("scale", "f16");
    x.i("iters", 60);
    beats_rtn("autoround", g, x, 0.8, "i4*f16[1x32]");
    g2.s("codes", "u2").s("zero", "u4").i("group", 64).s("scale", "f16");
    beats_rtn("autoround", g2, x, 0.9, nullptr);
}

TEST(paroquant_learns_a_rotation_the_reader_undoes) {
    NEED_LIBQUANT();
    Opts g, x;
    g.s("codes", "u4").s("zero", "u4").i("group", 128).s("scale", "f16");
    x.i("steps", 60).i("iters", 40).i("sample_rows", 32);
    beats_rtn("paroquant", g, x, 0.85, "u4*f16[1x128]-u4[1x128]/givens");
    /* the rotation alone, stage two off, is already below rounding to nearest */
    Opts x1;
    x1.i("steps", 60).i("iters", 0).i("sample_rows", 32);
    beats_rtn("paroquant", g, x1, 0.95, nullptr);
    /* the pairs: every stage a perfect matching inside each 128-channel group */
    Opts ident = g;
    ident.i("steps", 0).i("iters", 0);
    Quantised q = quantise(quantizer("paroquant"), ident, kName, kR, kC, weight());
    REQUIRE_EQ(q.status, RAD_OK);
    const int32_t* p = (const int32_t*)q.planes[4].data();
    int bad = 0;
    for (int s = 0; s < 8; ++s) {
        std::vector<int> seen((size_t)kC, 0);
        for (int64_t k = 0; k < kC; k += 2) {
            const int32_t a = p[s * kC + k], b = p[s * kC + k + 1];
            bad += a / 128 != b / 128 || a == b;
            ++seen[(size_t)a];
            ++seen[(size_t)b];
        }
        for (int v : seen) bad += v != 1;
    }
    CHECK_EQ(bad, 0);
}

/* A calibration file GPTQ cannot use is reported on stderr, by path and reason, and skipped; with
 * nothing usable the weight is rounded to nearest, byte for byte, and that is said too. The report
 * is the only sign of it in an hours-long conversion, so each reason is pinned: a file that is not
 * a Gram, one of another width, one cut short, domains that cannot both tile K, and a Hessian that
 * is not positive definite even damped. */
TEST(an_unusable_calibration_file_is_reported_and_gptq_rounds_to_nearest) {
    NEED_LIBQUANT();
    const int64_t R = 8, C = 128;
    const char* name = "blk.6.attn_k.weight";
    const std::vector<float> w = draw(R * C, 61, 4.0f);
    std::vector<float> spd((size_t)(C * C), 0.0f), neg((size_t)(C * C), 0.0f);
    for (int64_t i = 0; i < C; ++i) {
        spd[(size_t)(i * C + i)] = 2.0f;
        if (i + 1 < C) spd[(size_t)(i * C + i + 1)] = spd[(size_t)((i + 1) * C + i)] = 0.5f;
        neg[(size_t)(i * C + i)] = -1.0f;
    }
    Opts ro;
    ro.s("codes", "i4").i("group", 32).s("scale", "bf16");
    const Quantised rtn = quantise(quantizer("rtn"), ro, name, R, C, w);
    REQUIRE_EQ(rtn.status, RAD_OK);

    struct Bad {
        const char* what;
        const char* says;
        std::function<bool(const std::string&)> write;
    };
    const std::string file = "/gram.r0.blk.6.attn_k.bin";
    const Bad cases[] = {
        { "not a Gram", "is not a RADGRAM2 file", [&](const std::string& d) {
              FILE* f = std::fopen((d + file).c_str(), "wb");
              if (!f) return false;
              const char junk[64] = "RADGRAM1";
              std::fwrite(junk, 1, sizeof junk, f);
              return std::fclose(f) == 0; } },
        { "another width", "holds a 64-wide Hessian and this weight's K is 128",
          [&](const std::string& d) {
              return write_gram(d + file, 64, 100, std::vector<float>(64 * 64, 1.0f)); } },
        { "cut short", "is short", [&](const std::string& d) {
              if (!write_gram(d + file, C, 100, spd)) return false;
              return ::truncate((d + file).c_str(), 32 + 4 * C * C / 2) == 0; } },
        { "a domain that cannot tile K", "do not both tile K=128", [&](const std::string& d) {
              return write_gram(d + file, C, 100, spd, 96); } },
        { "not positive definite", "not positive definite even damped", [&](const std::string& d) {
              return write_gram(d + file, C, 100, neg); } },
    };
    for (const Bad& b : cases) {
        const TempDir dir("rad_quant_badcalib");
        REQUIRE(!dir.path.empty());
        REQUIRE(b.write(dir.path));
        Opts go;
        go.s("codes", "i4").i("group", 32).s("scale", "bf16").s("calib", dir.path.c_str());
        Quantised g;
        const std::string log = stderr_of([&] { g = quantise(quantizer("gptq"), go, name, R, C, w); });
        REQUIRE_EQ(g.status, RAD_OK);
        const bool said = has(log, b.says) && has(log, "rounding it to nearest");
        if (!said) std::fprintf(stderr, "    %s: %s", b.what, log.c_str());
        CHECK(said);
        CHECK(same_planes(g, rtn));
    }

    /* A usable file in the same place does move codes, so the equality above is the fallback's. */
    const TempDir dir("rad_quant_goodcalib");
    REQUIRE(write_gram(dir.path + file, C, 100, spd));
    Opts go;
    go.s("codes", "i4").i("group", 32).s("scale", "bf16").s("calib", dir.path.c_str());
    Quantised g;
    (void)stderr_of([&] { g = quantise(quantizer("gptq"), go, name, R, C, w); });
    REQUIRE_EQ(g.status, RAD_OK);
    CHECK(!same_planes(g, rtn));
}

TEST(a_calibrated_quantiser_refuses_a_grid_it_cannot_serve) {
    NEED_LIBQUANT();
    RadQuantWeight qw{};
    qw.name = kName;
    qw.rank = 2; qw.shape[0] = kR; qw.shape[1] = kC; qw.rows = kR; qw.cols = kC;
    RadEncoding e;
    auto refused = [&](const char* q, Opts o) {
        return quantizer(q)->encoding(o.data(), o.n(), &qw, &e) != RAD_OK;
    };
    { Opts o; o.s("codes", "i4").s("transform", "fwht128");              CHECK(refused("awq", o)); }
    { Opts o; o.s("codes", "fp4_e2m1").i("group", 32);                   CHECK(refused("autoround", o)); }
    { Opts o; o.s("codes", "i4").s("block", "128x128");                  CHECK(refused("autoround", o)); }
    { Opts o; o.s("codes", "i4").s("transform", "fwht128");              CHECK(refused("paroquant", o)); }
    { Opts o; o.s("codes", "i4").i("rot_group", 96);                     CHECK(refused("paroquant", o)); }
    { Opts o; o.s("codes", "i4").i("act_order", 1).s("transform", "fwht128"); CHECK(refused("gptq", o)); }
}

RAD_TEST_MAIN()
