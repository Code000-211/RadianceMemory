/* Drives libr4d's MLA matrix row (r4d_mla.hip, attend_matrix) through the SIMT emulator and checks
 * every output element against an f64 softmax over each row's own table. The cases put the paired
 * tile (eight heads of two neighbouring rows) on the tables it meets in serving: a prefill chunk
 * whose rows share one table, and rows past the DSA budget, where each row selects its own top
 * blocks and neighbours share most blocks but rarely a table position. The conflict counters
 * (RADIANCE_MLA_CONFLICT_STATS) show which of the two the pairing saw. Both pairings run on the
 * same operands: the union at the case's table width, the positional pairing on the table
 * widened past the union's lists (kUnionBlocks, 520 entries) with absent entries no row reaches.
 * The rows row (attend_rows, 64 heads of one row a workgroup) runs on its own cases, and a mixed
 * TP step runs both ways, split by heads over the matrix row and by rows over the rows row. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" void r4d_mla_matrix_pin_form(int form);
extern "C" void r4d_mla_matrix_stats(unsigned long long* out);
extern "C" void r4d_mla_rows_pin_form(int form);

namespace {

constexpr int D = 512, BS = 4, kUnionBlocks = 520;

enum class Fill { prefill, topk, ragged, repeat };

struct Case {
    std::string name;
    int rows, heads, blocks;
    Fill fill;
    int swaps = 0;            /* topk: blocks exchanged between neighbouring rows */
    int context = 0;          /* topk: tokens of the sequence the rows end */
    bool scratch = false;     /* lets the launcher split the history */
    int form = 0;             /* r4d_mla_matrix_pin_form */
    int expect_second = -1;   /* positional pairing: 0 no second pass, 1 some, -1 either */
    /* Unsplit only: with scratch the widened table gives the positional run more splits. */
    bool bitwise_a = true; /* the union's request r equals the positional pairing's bit for bit */
    bool bitwise_all = false; /* both requests do (the tables agree position by position) */
    bool union_falls_back = false; /* the table is wider than the union's lists */
};

/* Tables as qsa_select emits them past its budget: for the queries at positions context-rows ..
 * context-1 of one sequence, each row the top blocks-1 of its complete blocks, the strictly higher
 * ones in block order and a short tie run after them, then the partial page. Row r's choice is row
 * r-1's with `swaps` members exchanged for non-members. Pages are a fixed shuffle of the blocks. */
void topk_tables(int rows, int blocks, int context, int swaps, std::mt19937& rng,
                 std::vector<int>& bt, std::vector<int>& len, int& pages) {
    const int topk = blocks - 1, universe = context / BS + 1, ties = std::min(4, topk);
    pages = universe;
    std::vector<int> page(universe);
    for (int i = 0; i < universe; ++i) page[i] = i;
    std::shuffle(page.begin(), page.end(), rng);
    std::vector<char> in(universe, 0);
    std::vector<int> members;
    auto tie_rank = [](int b) { return (uint32_t)b * 2654435761u ^ 0x9e3779b9u; };
    bt.assign((size_t)rows * blocks, -1);
    len.assign(rows, 0);
    for (int r = 0; r < rows; ++r) {
        const int p = context - rows + r, n = (p + 1) / BS;
        if (n <= topk) {
            for (int b = 0; b < n; ++b)
                if (!in[b]) {
                    in[b] = 1;
                    members.push_back(b);
                }
        } else {
            while ((int)members.size() < topk) {
                const int b = (int)(rng() % (unsigned)n);
                if (!in[b]) {
                    in[b] = 1;
                    members.push_back(b);
                }
            }
            for (int s = 0; s < swaps; ++s) {
                int add;
                do add = (int)(rng() % (unsigned)n);
                while (in[add]);
                const size_t drop = rng() % members.size();
                in[members[drop]] = 0;
                members[drop] = add;
                in[add] = 1;
            }
        }
        std::vector<int> order(members);
        std::sort(order.begin(), order.end());
        if (n > topk) {
            std::vector<int> tied(order);
            std::sort(tied.begin(), tied.end(),
                      [&](int a, int b) { return tie_rank(a) < tie_rank(b); });
            tied.resize(ties);
            std::sort(tied.begin(), tied.end());
            order.erase(std::remove_if(
                            order.begin(), order.end(),
                            [&](int b) { return std::binary_search(tied.begin(), tied.end(), b); }),
                        order.end());
            order.insert(order.end(), tied.begin(), tied.end());
        }
        int out = 0;
        for (int b : order) bt[(size_t)r * blocks + out++] = page[b];
        if ((p + 1) % BS) bt[(size_t)r * blocks + out++] = page[n];
        len[r] = out ? (out - 1) * BS + p % BS + 1 : 0;
    }
}

/* The case's page tables and visible lengths; returns the pages the cache holds. */
int fill_tables(const Case& c, std::mt19937& rng, std::vector<int>& bt, std::vector<int>& len) {
    const int M = c.rows, B = c.blocks;
    int pages = 0;
    if (c.fill == Fill::topk) {
        topk_tables(M, B, c.context, c.swaps, rng, bt, len, pages);
    } else if (c.fill == Fill::prefill) {
        /* one sequence, its rows the last M positions of the capacity, one table */
        pages = B + 1;
        bt.assign((size_t)M * B, -1);
        len.assign(M, 0);
        for (int r = 0; r < M; ++r) {
            for (int b = 0; b < B; ++b) bt[(size_t)r * B + b] = (b * 7 + 3) % pages;
            len[r] = B * BS - M + r + 1;
        }
    } else if (c.fill == Fill::repeat) {
        /* every page twice in a row's table and shifted by one entry from row to row, so a page
         * repeats within a table and between the pair */
        pages = M + B;
        bt.assign((size_t)M * B, -1);
        len.assign(M, 0);
        for (int r = 0; r < M; ++r) {
            for (int b = 0; b < B; ++b) bt[(size_t)r * B + b] = ((b + r) / 2 * 5) % pages;
            len[r] = B * BS - r % 3;
        }
    } else {
        /* every row its own scattered table, with holes and out-of-range pages */
        pages = M * B;
        bt.assign((size_t)M * B, -1);
        len.assign(M, 0);
        for (int r = 0; r < M; ++r) {
            for (int b = 0; b < B; ++b)
                bt[(size_t)r * B + b] = b % 17 == 3   ? -1
                                        : b % 19 == 5 ? pages + 3
                                                      : (b * 7 + r * 11) % pages;
            len[r] = r % 5 == 4 ? 0 : B * BS - 1 - r % 7;
        }
    }
    return pages;
}

RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[i];
        t.stride[i] = st;
        st *= shape[i];
    }
    return t;
}

/* An f64 softmax over each row's own table. */
std::vector<double> reference(int M, int H, int B, int pages, const std::vector<int>& bt,
                              const std::vector<int>& len, const std::vector<uint16_t>& qb,
                              const std::vector<uint16_t>& kvb, double scale) {
    std::vector<double> want((size_t)M * H * D);
    std::vector<double> sc;
    for (int r = 0; r < M; ++r)
        for (int h = 0; h < H; ++h) {
            const int live = std::min(std::max(len[r], 0), B * BS);
            sc.assign(live, -INFINITY);
            double mx = -INFINITY;
            auto row_of = [&](int j) -> const uint16_t* {
                const int pg = bt[(size_t)r * B + j / BS];
                return pg >= 0 && pg < pages ? &kvb[((size_t)pg * BS + j % BS) * D] : nullptr;
            };
            const uint16_t* qr = &qb[((size_t)r * H + h) * D];
            for (int j = 0; j < live; ++j) {
                const uint16_t* k = row_of(j);
                if (!k) continue;
                double dot = 0;
                for (int d = 0; d < D; ++d)
                    dot += (double)rad_bf16_to_f32(qr[d]) * rad_bf16_to_f32(k[d]);
                sc[j] = dot * scale;
                mx = std::max(mx, sc[j]);
            }
            std::vector<double> acc(D, 0.0);
            double sum = 0;
            for (int j = 0; j < live; ++j) {
                if (sc[j] == -INFINITY) continue;
                const double w = std::exp(sc[j] - mx);
                sum += w;
                const uint16_t* k = row_of(j);
                for (int d = 0; d < D; ++d) acc[d] += w * rad_bf16_to_f32(k[d]);
            }
            for (int d = 0; d < D; ++d)
                want[((size_t)r * H + h) * D + d] = sum > 0 ? acc[d] / sum : 0.0;
        }
    return want;
}

int run(const Case& c, std::mt19937& rng) {
    const int M = c.rows, H = c.heads, B = c.blocks;
    std::vector<int> bt, len;
    const int pages = fill_tables(c, rng, bt, len);
    std::normal_distribution<float> nrm(0.f, 1.f);
    std::vector<uint16_t> qb((size_t)M * H * D), kvb((size_t)pages * BS * D);
    for (auto& v : qb) v = rad_f32_to_bf16(0.25f * nrm(rng));
    for (auto& v : kvb) v = rad_f32_to_bf16(nrm(rng));
    std::vector<uint16_t> out((size_t)M * H * D, 0xffff);

    RadTensor t[5] = { tens(qb.data(), RAD_BF16, { M, H, D }),
                       tens(kvb.data(), RAD_BF16, { pages, 1, BS, D }),
                       tens(bt.data(), RAD_I32, { M, B }), tens(len.data(), RAD_I32, { M }),
                       tens(out.data(), RAD_BF16, { M, H, D }) };
    std::vector<RadParam> p;
    auto pi = [&](const char* k, long long v) {
        RadParam x{};
        x.key = k;
        x.kind = RAD_P_INT;
        x.ival = v;
        p.push_back(x);
    };
    auto pf = [&](const char* k, double v) {
        RadParam x{};
        x.key = k;
        x.kind = RAD_P_F64;
        x.dval = v;
        p.push_back(x);
    };
    pi("M", M);
    pi("latent_dim", D);
    pi("heads", H);
    pi("blocks", B);
    pi("block_size", BS);
    const double scale = 0.0625;
    pf("scale", scale);
    RadArgs a{};
    a.t = t;
    a.n_t = 5;
    a.p = p.data();
    a.n_p = (int)p.size();
    std::vector<char> scratch;
    if (c.scratch) {
        const int64_t need = r4d_rad_mla_attn_matrix_scratch(&a);
        scratch.assign((size_t)need + 16, 0);
        a.scratch = scratch.data() + ((16 - ((uintptr_t)scratch.data() & 15)) & 15);
        a.scratch_bytes = need;
    }

    const std::vector<double> want = reference(M, H, B, pages, bt, len, qb, kvb, scale);

    /* Both pairings on the same operands: the positional one, on the table widened past the
     * union's lists, and at the case's own width the launcher's choice, the union, which must
     * never take a second pass. A tile of 16 heads has no pairing and runs once. */
    const bool paired = H % 16 == 8;
    const int wide = std::max(B, kUnionBlocks + 1);
    std::vector<int> bt_wide((size_t)M * wide, -1);
    for (int r = 0; r < M; ++r)
        std::copy(bt.begin() + (size_t)r * B, bt.begin() + (size_t)(r + 1) * B,
                  bt_wide.begin() + (size_t)r * wide);
    std::vector<uint16_t> by_pos;
    int fails = 0;
    for (const char* pairing : { "pos", "union" }) {
        const bool by_page = !std::strcmp(pairing, "union");
        if (!paired && by_page) break;
        const bool widened = paired && !by_page;
        t[2] = widened ? tens(bt_wide.data(), RAD_I32, { M, wide })
                       : tens(bt.data(), RAD_I32, { M, B });
        p[3].ival = widened ? wide : B;
        std::fill(out.begin(), out.end(), 0xffff);
        unsigned long long stats[3];
        r4d_mla_matrix_stats(stats);
        r4d_mla_matrix_pin_form(c.form);
        const int rc = r4d_rad_mla_attn_matrix(&a, nullptr);
        r4d_mla_matrix_pin_form(0);
        r4d_mla_matrix_stats(stats);
        if (rc != RAD_OK) {
            std::printf("%-40s %-5s launch returned %d\n", c.name.c_str(), pairing, rc);
            ++fails;
            continue;
        }
        double max_err = 0;
        long bad = 0;
        for (size_t i = 0; i < out.size(); ++i) {
            const float got = rad_bf16_to_f32(out[i]);
            const double e = std::fabs(got - want[i]);
            if (!std::isfinite(got) || e > 1e-5 + 0.008 * std::fabs(want[i])) ++bad;
            if (std::isfinite(got)) max_err = std::max(max_err, e);
        }
        bool ok = bad == 0;
        std::string note;
        if (!by_page) {
            by_pos = out;
            if (c.expect_second >= 0 && (stats[1] > 0) != (c.expect_second == 1)) {
                note +=
                    c.expect_second ? "  (expected a second pass)" : "  (expected no second pass)";
                ok = false;
            }
        } else {
            if ((stats[1] > 0) != c.union_falls_back) {
                note += c.union_falls_back ? "  (expected the positional fallback)"
                                           : "  (the union took a second pass)";
                ok = false;
            }
            /* Request r of each pair walks its own table in its own order, as the positional
             * pairing does, so at one split it must agree to the bit; r+1 sums in another order. */
            size_t first_diff_a = 0, diff_b = 0;
            double max_ab = 0;
            for (int r = 0; r < M; ++r)
                for (size_t i = (size_t)r * H * D; i < (size_t)(r + 1) * H * D; ++i) {
                    if (out[i] == by_pos[i]) continue;
                    if (r % 2 == 0)
                        ++first_diff_a;
                    else
                        ++diff_b;
                    max_ab = std::max(max_ab, (double)std::fabs(rad_bf16_to_f32(out[i]) -
                                                                rad_bf16_to_f32(by_pos[i])));
                }
            char buf[160];
            std::snprintf(buf, sizeof buf, "  vs pos: %zu/%zu differ (r/r+1), max abs %.2e",
                          first_diff_a, diff_b, max_ab);
            note += buf;
            if (c.bitwise_a && !c.scratch && first_diff_a) {
                note += "  (request r differs)";
                ok = false;
            }
            if (c.bitwise_all && !c.scratch && diff_b) {
                note += "  (request r+1 differs)";
                ok = false;
            }
        }
        std::printf("%-40s %-5s M %3d H %2d blocks %3d  passes %5llu raster %5llu second %5llu  "
                    "max_abs %.2e  bad %ld%s  %s\n",
                    c.name.c_str(), pairing, M, H, B, stats[0], stats[2], stats[1], max_err, bad,
                    note.c_str(), ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    return fails;
}

/* The rows row (attend_rows): one query row and 64 heads a workgroup, as GLM's prefill runs MLA
 * split by rows. Each form against the f64 softmax and the forms bit for bit against each other
 * (the element-wise key load; 32 heads a workgroup, whose heads sum alike); unsplit, a run of the
 * rows against the same rows of the whole batch, since a rank computing its run must produce the
 * bytes the batch would. */
int run_rows(const Case& c, std::mt19937& rng) {
    const int M = c.rows, H = c.heads, B = c.blocks;
    std::vector<int> bt, len;
    const int pages = fill_tables(c, rng, bt, len);
    std::normal_distribution<float> nrm(0.f, 1.f);
    std::vector<uint16_t> qb((size_t)M * H * D), kvb((size_t)pages * BS * D);
    for (auto& v : qb) v = rad_f32_to_bf16(0.25f * nrm(rng));
    for (auto& v : kvb) v = rad_f32_to_bf16(nrm(rng));
    const double scale = 0.0625;
    const std::vector<double> want = reference(M, H, B, pages, bt, len, qb, kvb, scale);
    std::vector<char> scratch;
    auto launch = [&](int first, int rows, int form, std::vector<uint16_t>& out) {
        const size_t q0 = (size_t)first * H * D;
        RadTensor t[5] = { tens(qb.data() + q0, RAD_BF16, { rows, H, D }),
                           tens(kvb.data(), RAD_BF16, { pages, 1, BS, D }),
                           tens(bt.data() + (size_t)first * B, RAD_I32, { rows, B }),
                           tens(len.data() + first, RAD_I32, { rows }),
                           tens(out.data() + q0, RAD_BF16, { rows, H, D }) };
        RadParam p[6]{};
        const char* keys[5] = { "M", "latent_dim", "heads", "blocks", "block_size" };
        const long long vals[5] = { rows, D, H, B, BS };
        for (int i = 0; i < 5; ++i) {
            p[i].key = keys[i];
            p[i].kind = RAD_P_INT;
            p[i].ival = vals[i];
        }
        p[5].key = "scale";
        p[5].kind = RAD_P_F64;
        p[5].dval = scale;
        RadArgs a{};
        a.t = t;
        a.n_t = 5;
        a.p = p;
        a.n_p = 6;
        if (c.scratch) {
            const int64_t need = r4d_rad_mla_attn_matrix_scratch(&a);
            scratch.assign((size_t)need + 16, 0);
            a.scratch = scratch.data() + ((16 - ((uintptr_t)scratch.data() & 15)) & 15);
            a.scratch_bytes = need;
        }
        if (form < 0) return r4d_rad_mla_attn_matrix(&a, nullptr);
        r4d_mla_rows_pin_form(form);
        const int rc = r4d_rad_mla_attn_rows(&a, nullptr);
        r4d_mla_rows_pin_form(0);
        return rc;
    };
    int fails = 0;
    std::vector<uint16_t> rule;
    for (int form : { 3, 1, 2, 0 }) {
        std::vector<uint16_t> out((size_t)M * H * D, 0xffff);
        const int rc = launch(0, M, form, out);
        if (rc != RAD_OK) {
            std::printf("%-40s form %d launch returned %d  FAIL\n", c.name.c_str(), form, rc);
            ++fails;
            continue;
        }
        double max_err = 0;
        long bad = 0;
        for (size_t i = 0; i < out.size(); ++i) {
            const float got = rad_bf16_to_f32(out[i]);
            const double e = std::fabs(got - want[i]);
            if (!std::isfinite(got) || e > 1e-5 + 0.008 * std::fabs(want[i])) ++bad;
            if (std::isfinite(got)) max_err = std::max(max_err, e);
        }
        bool ok = bad == 0;
        std::string note;
        /* A split follows the workgroup count, which 32 heads a workgroup doubles; under 32 rows
         * the rule is the matrix row's kernel. */
        if (form == 3)
            rule = out;
        else if (form == 0 && M < 32) {
            std::vector<uint16_t> matrix((size_t)M * H * D, 0xffff);
            const int mrc = launch(0, M, -1, matrix);
            if (mrc != RAD_OK || matrix != out) {
                note = "  (the rule under 32 rows is not the matrix row)";
                ok = false;
            }
        } else if ((form == 1 || !c.scratch) && out != rule) {
            note = "  (differs from form 3)";
            ok = false;
        }
        std::printf("%-40s form %d M %3d H %2d blocks %3d  max_abs %.2e  bad %ld%s  %s\n",
                    c.name.c_str(), form, M, H, B, max_err, bad, note.c_str(), ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    if (!c.scratch && M >= 3 && !rule.empty()) {
        std::vector<uint16_t> out((size_t)M * H * D, 0xffff);
        const int rc = launch(1, M - 2, 3, out);
        bool same = rc == RAD_OK, untouched = true;
        for (int r = 0; r < M; ++r)
            for (size_t i = (size_t)r * H * D; i < (size_t)(r + 1) * H * D; ++i) {
                if (r == 0 || r == M - 1)
                    untouched = untouched && out[i] == 0xffff;
                else
                    same = same && out[i] == rule[i];
            }
        std::printf("%-40s rows 1..%d alone: %s, rows outside %s  %s\n", c.name.c_str(), M - 2,
                    same ? "bit-identical to the batch" : "DIFFER from the batch",
                    untouched ? "untouched" : "WRITTEN", same && untouched ? "PASS" : "FAIL");
        fails += !(same && untouched);
    }
    return fails;
}

/* A TP step's attention both ways on one operand set: by heads, each of `world` ranks running
 * the matrix row on its heads of every row (views into the whole [T, 64, 512]), and by rows, each
 * rank the rows row on its run of ceil(T / world) rows with every head, the last run short. The
 * step mixes what a step batch carries, decode rows first: single-row decodes on scattered tables,
 * a verify window of four rows on one table, then a prefill continuing past the DSA budget and a
 * fresh prompt's first chunk. Both forms against the f64 softmax; how far apart they are is
 * printed. */
int run_split(const char* name, int world, int fresh_rows, std::mt19937& rng) {
    constexpr int H = 64, B = 65;
    std::vector<int> bt, len;
    int pages = 0;
    auto append = [&](const std::vector<int>& t, const std::vector<int>& l, int segment_pages) {
        for (int v : t) bt.push_back(v >= 0 ? v + pages : v);
        len.insert(len.end(), l.begin(), l.end());
        pages += segment_pages;
    };
    {   /* three decodes, each its own table with holes */
        std::vector<int> t((size_t)3 * B), l(3);
        for (int r = 0; r < 3; ++r) {
            for (int b = 0; b < B; ++b) t[(size_t)r * B + b] = b % 13 == 5 ? -1 : (b * 5 + r * 17) % (3 * B);
            l[r] = B * BS - 1 - 3 * r;
        }
        append(t, l, 3 * B);
    }
    {   /* a verify window: four rows of one sequence on one table */
        std::vector<int> t((size_t)4 * B), l(4);
        for (int r = 0; r < 4; ++r) {
            for (int b = 0; b < B; ++b) t[(size_t)r * B + b] = (b * 3 + 1) % B;
            l[r] = B * BS - 4 + r + 1;
        }
        append(t, l, B);
    }
    {   /* a prefill past the DSA budget: every row its own top blocks */
        std::vector<int> t, l;
        int segment = 0;
        topk_tables(23, B, 1200, 4, rng, t, l, segment);
        append(t, l, segment);
    }
    {   /* a fresh prompt's chunk: causal on one table, rows starting at position 0 */
        std::vector<int> t((size_t)fresh_rows * B, -1), l(fresh_rows);
        for (int r = 0; r < fresh_rows; ++r) {
            for (int b = 0; b <= r / BS && b < B; ++b) t[(size_t)r * B + b] = b;
            l[r] = r + 1;
        }
        append(t, l, B);
    }
    const int T = (int)len.size();
    std::normal_distribution<float> nrm(0.f, 1.f);
    std::vector<uint16_t> qb((size_t)T * H * D), kvb((size_t)pages * BS * D);
    for (auto& v : qb) v = rad_f32_to_bf16(0.25f * nrm(rng));
    for (auto& v : kvb) v = rad_f32_to_bf16(nrm(rng));
    const double scale = 0.0625;
    const std::vector<double> want = reference(T, H, B, pages, bt, len, qb, kvb, scale);
    auto launch = [&](bool by_rows, int first, int rows, int h0, int heads,
                      std::vector<uint16_t>& out) {
        const size_t at = ((size_t)first * H + h0) * D;
        RadTensor t[5] = { tens(qb.data() + at, RAD_BF16, { rows, heads, D }),
                           tens(kvb.data(), RAD_BF16, { pages, 1, BS, D }),
                           tens(bt.data() + (size_t)first * B, RAD_I32, { rows, B }),
                           tens(len.data() + first, RAD_I32, { rows }),
                           tens(out.data() + at, RAD_BF16, { rows, heads, D }) };
        t[0].stride[0] = t[4].stride[0] = (int64_t)H * D;
        RadParam p[6]{};
        const char* keys[5] = { "M", "latent_dim", "heads", "blocks", "block_size" };
        const long long vals[5] = { rows, D, heads, B, BS };
        for (int i = 0; i < 5; ++i) {
            p[i].key = keys[i];
            p[i].kind = RAD_P_INT;
            p[i].ival = vals[i];
        }
        p[5].key = "scale";
        p[5].kind = RAD_P_F64;
        p[5].dval = scale;
        RadArgs a{};
        a.t = t;
        a.n_t = 5;
        a.p = p;
        a.n_p = 6;
        if (!by_rows) return r4d_rad_mla_attn_matrix(&a, nullptr);
        /* Runs this short take the matrix kernel by rule; the rows kernel is what is under test. */
        r4d_mla_rows_pin_form(3);
        const int rc = r4d_rad_mla_attn_rows(&a, nullptr);
        r4d_mla_rows_pin_form(0);
        return rc;
    };
    std::vector<uint16_t> heads_out((size_t)T * H * D, 0xffff), rows_out((size_t)T * H * D, 0xffff);
    const int per_heads = H / world, run = (T + world - 1) / world;
    int fails = 0;
    /* What each rank's run buffer holds after its rows: written through the same launches into
     * the plane at the run's rows, then copied out as the all_gather moves it, rank r's run of
     * `run` rows to rows [r * run, (r + 1) * run) of a [world * run] output. */
    std::vector<uint16_t> gathered((size_t)world * run * H * D, 0xffff);
    for (int rank = 0; rank < world; ++rank) {
        fails += launch(false, 0, T, rank * per_heads, per_heads, heads_out) != RAD_OK;
        const int first = rank * run, n = std::min(run, T - first);
        if (n > 0) fails += launch(true, first, n, 0, H, rows_out) != RAD_OK;
        std::vector<uint16_t> part((size_t)run * H * D, 0xeeee);
        for (int r = 0; r < n; ++r)
            std::copy(rows_out.begin() + (size_t)(first + r) * H * D,
                      rows_out.begin() + (size_t)(first + r + 1) * H * D,
                      part.begin() + (size_t)r * H * D);
        std::copy(part.begin(), part.end(), gathered.begin() + (size_t)rank * run * H * D);
    }
    bool placed = true;
    for (size_t i = 0; i < (size_t)T * H * D; ++i) placed = placed && gathered[i] == rows_out[i];
    if (!placed) {
        std::printf("%-40s the gathered runs do not land on their rows  FAIL\n", name);
        ++fails;
    }
    long bad[2] = { 0, 0 };
    double max_err[2] = { 0, 0 }, apart = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const float got[2] = { rad_bf16_to_f32(heads_out[i]), rad_bf16_to_f32(rows_out[i]) };
        for (int f = 0; f < 2; ++f) {
            const double e = std::fabs(got[f] - want[i]);
            if (!std::isfinite(got[f]) || e > 1e-5 + 0.008 * std::fabs(want[i])) ++bad[f];
            if (std::isfinite(got[f])) max_err[f] = std::max(max_err[f], e);
        }
        apart = std::max(apart, (double)std::fabs(got[0] - got[1]));
    }
    const bool ok = !fails && !bad[0] && !bad[1];
    std::printf("%-40s T %3d world %d run %2d last %2d  heads max_abs %.2e bad %ld  rows max_abs "
                "%.2e bad %ld  apart %.2e  %s\n",
                name, T, world, run, T - (world - 1) * run, max_err[0], bad[0], max_err[1], bad[1],
                apart, ok ? "PASS" : "FAIL");
    return !ok;
}

/* The head form against the row form, row by row and bit by bit, on a TP8 prefill chunk past the
 * DSA budget: the head form is the paired union tile (eight heads of rows 2i and 2i+1), the row
 * form's two kernels are the 16-head matrix tile (r4d_mla_rows_pin_form(4)) and the rows kernel.
 * The union's first row of a pair walks its own table in its own order, as the 16-head tile does,
 * so it must agree to the bit; the second row and the rows kernel sum in other orders, within
 * rounding. Printed: rows bit-identical, elements differing, and the largest step in bf16 units. */
int run_rowwise(std::mt19937& rng) {
    constexpr int H = 64, B = 65, M = 16;
    std::vector<int> bt, len;
    int pages = 0;
    topk_tables(M, B, 1200, 16, rng, bt, len, pages);
    std::normal_distribution<float> nrm(0.f, 1.f);
    std::vector<uint16_t> qb((size_t)M * H * D), kvb((size_t)pages * BS * D);
    for (auto& v : qb) v = rad_f32_to_bf16(0.25f * nrm(rng));
    for (auto& v : kvb) v = rad_f32_to_bf16(nrm(rng));
    auto launch = [&](int kernel, int h0, int heads, std::vector<uint16_t>& out) {
        RadTensor t[5] = { tens(qb.data() + (size_t)h0 * D, RAD_BF16, { M, heads, D }),
                           tens(kvb.data(), RAD_BF16, { pages, 1, BS, D }),
                           tens(bt.data(), RAD_I32, { M, B }), tens(len.data(), RAD_I32, { M }),
                           tens(out.data() + (size_t)h0 * D, RAD_BF16, { M, heads, D }) };
        t[0].stride[0] = t[4].stride[0] = (int64_t)H * D;
        RadParam p[6]{};
        const char* keys[5] = { "M", "latent_dim", "heads", "blocks", "block_size" };
        const long long vals[5] = { M, D, heads, B, BS };
        for (int i = 0; i < 5; ++i) {
            p[i].key = keys[i];
            p[i].kind = RAD_P_INT;
            p[i].ival = vals[i];
        }
        p[5].key = "scale";
        p[5].kind = RAD_P_F64;
        p[5].dval = 0.0625;
        RadArgs a{};
        a.t = t;
        a.n_t = 5;
        a.p = p;
        a.n_p = 6;
        if (kernel < 2) return r4d_rad_mla_attn_matrix(&a, nullptr);
        r4d_mla_rows_pin_form(3);
        const int rc = r4d_rad_mla_attn_rows(&a, nullptr);
        r4d_mla_rows_pin_form(0);
        return rc;
    };
    std::vector<uint16_t> head((size_t)M * H * D), m16((size_t)M * H * D), rows((size_t)M * H * D);
    int rc = 0;
    for (int r = 0; r < 8; ++r) rc |= launch(0, r * 8, 8, head);
    rc |= launch(1, 0, H, m16);
    rc |= launch(2, 0, H, rows);
    auto ord = [](uint16_t h) { return (h & 0x8000) ? -(int)(h & 0x7fff) : (int)h; };
    int fails = rc != RAD_OK;
    for (const auto& [name, got] : { std::pair<const char*, const std::vector<uint16_t>*>{
                                         "16-head tile", &m16 },
                                     { "rows kernel", &rows } }) {
        int same_even = 0, same_odd = 0, max_ulp = 0;
        long diff = 0;
        double max_abs = 0;
        for (size_t i = 0; i < head.size(); ++i)
            max_abs = std::max(max_abs, (double)std::fabs(rad_bf16_to_f32((*got)[i]) -
                                                          rad_bf16_to_f32(head[i])));
        for (int r = 0; r < M; ++r) {
            bool same = true;
            for (size_t i = (size_t)r * H * D; i < (size_t)(r + 1) * H * D; ++i) {
                const int u = std::abs(ord((*got)[i]) - ord(head[i]));
                same = same && !u;
                diff += u != 0;
                max_ulp = std::max(max_ulp, u);
            }
            (r % 2 ? same_odd : same_even) += same;
        }
        /* The 16-head tile must give the union's first rows bit for bit. */
        const bool ok = std::strcmp(name, "16-head tile") || same_even == M / 2;
        std::printf("rowwise: %-12s vs head form (union pairs): rows bit-identical %d/%d first, "
                    "%d/%d second of a pair; %ld of %d elements differ, at most %d bf16 steps, "
                    "max abs %.2e  %s\n",
                    name, same_even, M / 2, same_odd, M / 2, diff, M * H * D, max_ulp, max_abs,
                    ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    return fails;
}

/* The launcher prints the running totals every 1024th counted launch and must leave them
 * standing: 1024 one-chunk launches of a pair, then the hook reads exactly 1024 passes. */
int periodic_report() {
    std::vector<uint16_t> q(2 * 8 * D, 0x3f80), kv(4 * BS * D, 0x3f80), out(2 * 8 * D);
    std::vector<int> bt = { 0, 1 }, len = { 4, 3 };
    RadTensor t[5]{};
    auto set = [](RadTensor& x, void* p, uint32_t dt, std::vector<int64_t> shape) {
        x.data = p;
        x.dtype = dt;
        x.rank = (uint32_t)shape.size();
        int64_t st = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            x.shape[i] = shape[i];
            x.stride[i] = st;
            st *= shape[i];
        }
    };
    set(t[0], q.data(), RAD_BF16, { 2, 8, D });
    set(t[1], kv.data(), RAD_BF16, { 4, 1, BS, D });
    set(t[2], bt.data(), RAD_I32, { 2, 1 });
    set(t[3], len.data(), RAD_I32, { 2 });
    set(t[4], out.data(), RAD_BF16, { 2, 8, D });
    RadParam p[6]{};
    const char* keys[5] = { "M", "latent_dim", "heads", "blocks", "block_size" };
    const long long vals[5] = { 2, D, 8, 1, BS };
    for (int i = 0; i < 5; ++i) {
        p[i].key = keys[i];
        p[i].kind = RAD_P_INT;
        p[i].ival = vals[i];
    }
    p[5].key = "scale";
    p[5].kind = RAD_P_F64;
    p[5].dval = 0.0625;
    RadArgs a{};
    a.t = t;
    a.n_t = 5;
    a.p = p;
    a.n_p = 6;
    unsigned long long stats[3];
    r4d_mla_matrix_stats(stats);
    for (int i = 0; i < 1024; ++i)
        if (r4d_rad_mla_attn_matrix(&a, nullptr) != RAD_OK) {
            std::printf("periodic report: launch failed\n");
            return 1;
        }
    r4d_mla_matrix_stats(stats);
    const bool ok = stats[0] == 1024 && stats[2] == 1024 && stats[1] == 0;
    std::printf("periodic report: %llu passes over %llu raster chunks after 1024 launches  %s\n",
                stats[0], stats[2], ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* mla_attn_paged (Ling-3.0's MLA) under speculation: the k + 1 rows of a verify, launched
 * together, against k + 1 single-row decodes at the same positions, each with the table its own
 * step carries and scratch as the hook sizes it. Greedy output equals the run without
 * speculation only if they agree bit for bit; a split taken from the launch's rows (16, 16 and
 * 11 ways at eight heads for one, two and three rows) or from its longest table does not. */
int run_paged_verify(int heads, int context, std::mt19937& rng) {
    constexpr int PB = 16, RD = 64, W = D + RD, kMaxRows = 32;
    const int pages = (context + kMaxRows + PB - 1) / PB;
    std::normal_distribution<float> nrm(0.f, 1.f);
    std::vector<uint16_t> kvb((size_t)pages * PB * W), qb((size_t)kMaxRows * heads * D),
        rb((size_t)kMaxRows * heads * RD), gb((size_t)kMaxRows * heads);
    for (auto& v : kvb) v = rad_f32_to_bf16(nrm(rng));
    for (auto& v : qb) v = rad_f32_to_bf16(0.25f * nrm(rng));
    for (auto& v : rb) v = rad_f32_to_bf16(0.25f * nrm(rng));
    for (auto& v : gb) v = rad_f32_to_bf16(nrm(rng));
    std::vector<int> page(pages);
    for (int i = 0; i < pages; ++i) page[i] = i;
    std::shuffle(page.begin(), page.end(), rng);
    /* Rows [row0, row0 + rows) of one sequence whose context after the step is `used`. */
    auto launch = [&](int row0, int rows, int used, std::vector<uint16_t>& out) {
        const int blocks = (used + PB - 1) / PB;
        std::vector<int> bt(page.begin(), page.begin() + blocks), su = { used }, cu = { 0, rows };
        RadTensor t[8] = { tens(qb.data() + (size_t)row0 * heads * D, RAD_BF16, { rows, heads, D }),
                           tens(rb.data() + (size_t)row0 * heads * RD, RAD_BF16,
                                { rows, heads, RD }),
                           tens(kvb.data(), RAD_BF16, { pages, 1, PB, W }),
                           tens(bt.data(), RAD_I32, { 1, blocks }),
                           tens(su.data(), RAD_I32, { 1 }),
                           tens(cu.data(), RAD_I32, { 2 }),
                           tens(gb.data() + (size_t)row0 * heads, RAD_BF16, { rows, heads }),
                           tens(out.data() + (size_t)row0 * heads * D, RAD_BF16,
                                { rows, heads, D }) };
        RadParam p[6]{};
        const char* keys[5] = { "M", "heads", "latent_dim", "rope_dim", "block_size" };
        const long long vals[5] = { rows, heads, D, RD, PB };
        for (int i = 0; i < 5; ++i) {
            p[i].key = keys[i];
            p[i].kind = RAD_P_INT;
            p[i].ival = vals[i];
        }
        p[5].key = "scale";
        p[5].kind = RAD_P_F64;
        p[5].dval = 1.0 / std::sqrt(192.0);
        RadArgs a{};
        a.t = t;
        a.n_t = 8;
        a.p = p;
        a.n_p = 6;
        const int64_t need = r4d_rad_mla_attn_paged_scratch(&a);
        if (need < 0) return (int)need;
        std::vector<char> scratch((size_t)need + 16, 0);
        a.scratch = scratch.data() + ((16 - ((uintptr_t)scratch.data() & 15)) & 15);
        a.scratch_bytes = need;
        return r4d_rad_mla_attn_paged(&a, nullptr);
    };
    int fails = 0;
    for (int rows : { 2, 3, 4, 5, 8, kMaxRows }) {
        std::vector<uint16_t> verify((size_t)rows * heads * D), single(verify.size());
        int rc = launch(0, rows, context + rows, verify);
        for (int i = 0; i < rows && rc == RAD_OK; ++i) rc = launch(i, 1, context + i + 1, single);
        int same = 0;
        for (int i = 0; i < rows; ++i)
            same += !std::memcmp(&verify[(size_t)i * heads * D], &single[(size_t)i * heads * D],
                                 (size_t)heads * D * 2);
        const bool ok = rc == RAD_OK && same == rows;
        std::printf("paged verify: %2d heads, context %4d, %2d rows: %2d/%2d rows equal the "
                    "single-row decode bit for bit  %s\n",
                    heads, context, rows, same, rows, ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    return fails;
}

/* The E4M3-cache twins (r4d_rad_mla_attn*_fp8kv): every E4M3 value is a bf16, so each kernel form
 * on a cache of codes must give the bytes its bf16 instance gives on the same cache decoded to
 * bf16. Each launcher also refuses the other cache dtype. */
int run_fp8kv(const Case& c, std::mt19937& rng) {
    const int M = c.rows, H = c.heads, B = c.blocks;
    std::vector<int> bt, len;
    const int pages = fill_tables(c, rng, bt, len);
    std::normal_distribution<float> nrm(0.f, 1.f);
    std::vector<uint16_t> qb((size_t)M * H * D), kvb((size_t)pages * BS * D);
    std::vector<uint8_t> kv8(kvb.size());
    for (auto& v : qb) v = rad_f32_to_bf16(0.25f * nrm(rng));
    for (size_t i = 0; i < kv8.size(); ++i) {
        /* Codes over most exponents, subnormals included. */
        const float x = nrm(rng);
        kv8[i] = rad_f32_to_fp8e4m3(x * x * x * 4.0f);
        kvb[i] = rad_f32_to_bf16(rad_fp8e4m3_to_f32(kv8[i]));
        if (rad_bf16_to_f32(kvb[i]) != rad_fp8e4m3_to_f32(kv8[i])) {
            std::printf("%-40s E4M3 code %u is not a bf16  FAIL\n", c.name.c_str(), kv8[i]);
            return 1;
        }
    }
    std::vector<char> scratch;
    typedef int (*Launch)(const RadArgs*, RadStream);
    auto launch = [&](Launch fn, bool fp8, std::vector<uint16_t>& out) {
        RadTensor t[5] = { tens(qb.data(), RAD_BF16, { M, H, D }),
                           fp8 ? tens(kv8.data(), RAD_F8E4M3, { pages, 1, BS, D })
                               : tens(kvb.data(), RAD_BF16, { pages, 1, BS, D }),
                           tens(bt.data(), RAD_I32, { M, B }), tens(len.data(), RAD_I32, { M }),
                           tens(out.data(), RAD_BF16, { M, H, D }) };
        RadParam p[6]{};
        const char* keys[5] = { "M", "latent_dim", "heads", "blocks", "block_size" };
        const long long vals[5] = { M, D, H, B, BS };
        for (int i = 0; i < 5; ++i) {
            p[i].key = keys[i];
            p[i].kind = RAD_P_INT;
            p[i].ival = vals[i];
        }
        p[5].key = "scale";
        p[5].kind = RAD_P_F64;
        p[5].dval = 0.0625;
        RadArgs a{};
        a.t = t;
        a.n_t = 5;
        a.p = p;
        a.n_p = 6;
        if (c.scratch) {
            const int64_t need = r4d_rad_mla_attn_matrix_scratch(&a);
            scratch.assign((size_t)need + 16, 0);
            a.scratch = scratch.data() + ((16 - ((uintptr_t)scratch.data() & 15)) & 15);
            a.scratch_bytes = need;
        }
        return fn(&a, nullptr);
    };
    struct Form {
        const char* name;
        Launch bf16, fp8;
        void (*pin)(int);
        int form;
    };
    std::vector<Form> forms = { { "wave", r4d_rad_mla_attn, r4d_rad_mla_attn_fp8kv, nullptr, 0 } };
    if (H % 8 == 0) {
        forms.push_back({ "matrix", r4d_rad_mla_attn_matrix, r4d_rad_mla_attn_matrix_fp8kv,
                          r4d_mla_matrix_pin_form, 0 });
        forms.push_back({ "matrix, element load", r4d_rad_mla_attn_matrix,
                          r4d_rad_mla_attn_matrix_fp8kv, r4d_mla_matrix_pin_form, 1 });
    }
    if (H % 32 == 0)
        for (int form : { 0, 3, 1, 2 })
            forms.push_back({ "rows", r4d_rad_mla_attn_rows, r4d_rad_mla_attn_rows_fp8kv,
                              r4d_mla_rows_pin_form, form });
    int fails = 0;
    for (const Form& f : forms) {
        std::vector<uint16_t> want((size_t)M * H * D, 0xffff), got(want.size(), 0xffff);
        if (f.pin) f.pin(f.form);
        const int rb = launch(f.bf16, false, want), rf = launch(f.fp8, true, got);
        std::vector<uint16_t> spare(want.size(), 0xffff);
        const int refuse_b = launch(f.bf16, true, spare), refuse_f = launch(f.fp8, false, spare);
        if (f.pin) f.pin(0);
        size_t differ = 0;
        for (size_t i = 0; i < got.size(); ++i) differ += got[i] != want[i];
        const bool untouched = std::all_of(spare.begin(), spare.end(),
                                           [](uint16_t v) { return v == 0xffff; });
        const bool ok = rb == RAD_OK && rf == RAD_OK && !differ && refuse_b == RAD_E_DTYPE &&
                        refuse_f == RAD_E_DTYPE && untouched;
        std::printf("%-40s fp8kv %-20s form %d  rc %d/%d  %zu differ from bf16, refusals %d/%d  "
                    "%s\n",
                    c.name.c_str(), f.name, f.form, rb, rf, differ, refuse_b, refuse_f,
                    ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    return fails;
}

}   // namespace

int main(int argc, char** argv) {
    setenv("RADIANCE_MLA_CONFLICT_STATS", "1", 1);
    /* name, rows, heads, blocks, fill, swaps, context, scratch, form, expect_second, bitwise_a,
     * bitwise_all, union_falls_back */
    std::vector<Case> cases = {
        { "prefill, one table", 8, 8, 65, Fill::prefill, 0, 0, false, 0, 0, true, true },
        { "prefill, odd rows, split", 7, 8, 33, Fill::prefill, 0, 0, true, 0, 0, true, true },
        { "prefill, 16 heads", 4, 16, 33, Fill::prefill, 0, 0, false, 0, 0 },
        { "prefill, 24 heads (three pairs)", 5, 24, 17, Fill::prefill, 0, 0, false, 0, 0, true,
          true },
        { "topk d=0", 8, 8, 65, Fill::topk, 0, 600, false, 0, 0, true, true },
        { "topk d=4", 8, 8, 65, Fill::topk, 4, 600, false, 0, 1 },
        { "topk d=16", 8, 8, 65, Fill::topk, 16, 600, false, 0, 1 },
        { "topk d=64 (disjoint neighbours)", 6, 8, 65, Fill::topk, 64, 600, false, 0, 1 },
        { "topk d=16, 513 blocks", 4, 8, 513, Fill::topk, 16, 8192, false, 0, 1 },
        /* a split cuts the union's walk elsewhere than the positional raster */
        { "topk d=4, odd rows, split", 5, 8, 33, Fill::topk, 4, 400, true, 0, 1, false },
        { "topk d=4, element load", 4, 8, 33, Fill::topk, 4, 400, false, 1, 1 },
        { "topk d=4, 16 heads", 4, 16, 33, Fill::topk, 4, 400, false, 0, 0 },
        { "topk, table past the union's lists", 4, 8, 600, Fill::topk, 16, 4000, false, 0, 1, true,
          true, true },
        { "repeated pages", 6, 8, 33, Fill::repeat, 0, 0, false, 0, -1 },
        { "repeated pages, split", 3, 8, 33, Fill::repeat, 0, 0, true, 0, -1, false },
        { "ragged, holes and bad pages", 7, 8, 17, Fill::ragged, 0, 0, false, 0, -1 },
        { "ragged, split", 3, 8, 17, Fill::ragged, 0, 0, true, 0, -1, false },
    };
    /* The rows row at 64 heads, and at 32, which it serves with two groups a workgroup. */
    std::vector<Case> rows_cases = {
        { "rows: prefill, one table", 8, 64, 65, Fill::prefill },
        { "rows: topk d=4", 8, 64, 65, Fill::topk, 4, 600 },
        { "rows: topk d=16, 513 blocks", 3, 64, 513, Fill::topk, 16, 8192 },
        { "rows: repeated pages", 6, 64, 33, Fill::repeat },
        { "rows: ragged, holes and bad pages", 7, 64, 17, Fill::ragged },
        { "rows: ragged, split", 3, 64, 17, Fill::ragged, 0, 0, true },
        { "rows: prefill, one row, split", 1, 64, 65, Fill::prefill, 0, 0, true },
        { "rows: 32 heads", 5, 32, 33, Fill::prefill },
        { "rows: 32 heads, topk d=4, split", 2, 32, 33, Fill::topk, 4, 400, true },
    };
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int fails = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        if (filter && cases[i].name.find(filter) == std::string::npos) continue;
        std::mt19937 rng(20261018u + (unsigned)i);
        fails += run(cases[i], rng);
    }
    for (size_t i = 0; i < rows_cases.size(); ++i) {
        if (filter && rows_cases[i].name.find(filter) == std::string::npos) continue;
        std::mt19937 rng(20261020u + (unsigned)i);
        fails += run_rows(rows_cases[i], rng);
    }
    /* The E4M3-cache twins against their bf16 instances. */
    std::vector<Case> fp8_cases = {
        { "fp8kv: ragged, holes and bad pages", 7, 8, 17, Fill::ragged },
        { "fp8kv: prefill, 16 heads, split", 4, 16, 33, Fill::prefill, 0, 0, true },
        { "fp8kv: topk d=4, 8 heads", 6, 8, 33, Fill::topk, 4, 400 },
        { "fp8kv: rows, ragged", 3, 64, 17, Fill::ragged },
        { "fp8kv: rows, topk d=4, split", 2, 64, 33, Fill::topk, 4, 400, true },
    };
    for (size_t i = 0; i < fp8_cases.size(); ++i) {
        if (filter && fp8_cases[i].name.find(filter) == std::string::npos) continue;
        std::mt19937 rng(20261028u + (unsigned)i);
        fails += run_fp8kv(fp8_cases[i], rng);
    }
    if (!filter || std::strstr("rowwise", filter)) {
        std::mt19937 rng(20261022u);
        fails += run_rowwise(rng);
    }
    /* T 45 over eight ranks: runs of 6, the last of 3; 37 over four: 10 and 7; 48: whole runs. */
    const struct {
        const char* name;
        int world, fresh;
    } splits[] = { { "split: heads vs rows, 8 ranks", 8, 15 },
                   { "split: heads vs rows, 8 ranks, T 300", 8, 270 },
                   { "split: heads vs rows, 4 ranks", 4, 7 },
                   { "split: heads vs rows, whole runs", 8, 18 } };
    for (size_t i = 0; i < sizeof splits / sizeof splits[0]; ++i) {
        if (filter && !std::strstr(splits[i].name, filter)) continue;
        std::mt19937 rng(20261021u + (unsigned)i);
        fails += run_split(splits[i].name, splits[i].world, splits[i].fresh, rng);
    }
    /* Eight heads is a TP4 rank of Ling-3.0, 32 one card; a context of 40 is under the old
     * clamp by table length, 700 past it. */
    if (!filter || std::strstr("paged verify", filter)) {
        std::mt19937 rng(20261031u);
        for (int heads : { 8, 32 })
            for (int context : { 40, 700 }) fails += run_paged_verify(heads, context, rng);
    }
    if (!filter) fails += periodic_report();
    std::printf("%d failure(s)\n", fails);
    return fails ? 1 : 0;
}
