/* kv_autoquant_test.cpp -- AutoQuant's per-layer KV assignment (core/kv_autoquant.h) and its
 * calibration driver, rad-autoquant (tools/rad_autoquant.cpp).
 *
 * The solver is held to exhaustive search over synthetic sensitivities: within the budget, of
 * least summed KL, and the same answer every time. The driver is run as a deployment runs it, but
 * against a stand-in for the engine: a shell script that answers each KL-mode run with a mean KL
 * summed from a per-(layer, layout) table, the model of additive costs the solver assumes. The
 * real engine cannot take the stand-in's place on the host: no plugin the host backend serves
 * declares its cache per layer, and libref has no rows for the packed layouts. So the driver's
 * measuring, solving and recipe are checked here, and what the engine measures is the KL mode's
 * own tests' (kld_test, engine_e2e_test).
 */
#include "rad_test.h"
#include "tiny_model.h"

#include "kv_autoquant.h"
#include "rad_core.h"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace rad;

namespace {

/* The recipe layouts at their prices (bits a group of 32 K and 32 V elements). */
KvCandidate fp8(double kl = 0)   { return { RAD_KVQ_NONE, 512, kl }; }
KvCandidate v4(double kl)        { return { RAD_KVQ_V4,   400, kl }; }
KvCandidate k6v4(double kl)      { return { RAD_KVQ_K6V4, 352, kl }; }
KvCandidate k4h(double kl)       { return { RAD_KVQ_K4H,  288, kl }; }
KvCandidate k4hv2h(double kl)    { return { RAD_KVQ_K4HV2H, 224, kl }; }

/* Every assignment, for the least summed KL within `budget`; -1 when none fits. */
double brute_force(const std::vector<std::vector<KvCandidate>>& layers, int64_t budget) {
    double best = -1;
    std::vector<size_t> at(layers.size(), 0);
    for (;;) {
        int64_t bits = 0;
        double kl = 0;
        for (size_t l = 0; l < layers.size(); ++l) {
            bits += layers[l][at[l]].bits;
            kl += layers[l][at[l]].kl;
        }
        if (bits <= budget && (best < 0 || kl < best)) best = kl;
        size_t l = 0;
        while (l < layers.size() && ++at[l] == layers[l].size()) at[l++] = 0;
        if (l == layers.size()) return best;
    }
}

/* Every assignment, for the fewest bits whose summed KL is within `max_kl`, and the least KL of
 * those; bits -1 when none is. */
std::pair<int64_t, double> brute_force_kl(const std::vector<std::vector<KvCandidate>>& layers,
                                          double max_kl) {
    int64_t best_bits = -1;
    double best_kl = 0;
    std::vector<size_t> at(layers.size(), 0);
    for (;;) {
        int64_t bits = 0;
        double kl = 0;
        for (size_t l = 0; l < layers.size(); ++l) {
            bits += layers[l][at[l]].bits;
            kl += layers[l][at[l]].kl;
        }
        if (kl <= max_kl * (1 + 1e-12) &&
            (best_bits < 0 || bits < best_bits || (bits == best_bits && kl < best_kl))) {
            best_bits = bits;
            best_kl = kl;
        }
        size_t l = 0;
        while (l < layers.size() && ++at[l] == layers[l].size()) at[l++] = 0;
        if (l == layers.size()) return { best_bits, best_kl };
    }
}

std::vector<int> kvq_of(const std::vector<std::vector<KvCandidate>>& layers,
                        const KvAssignment& a) {
    std::vector<int> q;
    for (size_t l = 0; l < layers.size(); ++l) q.push_back(layers[l][(size_t)a.pick[l]].kvq);
    return q;
}

}  /* namespace */

/* ================================================================== the solver */

TEST(the_layout_table_prices_each_recipe_layout_widest_first) {
    CHECK(kv_layouts() == std::vector<int>({ RAD_KVQ_BF16, RAD_KVQ_I8H, RAD_KVQ_NONE, RAD_KVQ_V4,
                                             RAD_KVQ_K6V4, RAD_KVQ_K4H, RAD_KVQ_E2M1H,
                                             RAD_KVQ_K4HV2H }));
    /* fp8's 8 bits an element, and the 25/32, 22/32, 18/32 and 14/32 of it docs/GUIDE.md §4.4
     * states; i8h at 258/256 of it (an f16 scale a 256-dim head is 2 bits a group of 32); bf16 at
     * twice it, 16 bits an element and no scale. */
    CHECK_EQ(kv_layout_group_bits(RAD_KVQ_BF16), 1024);
    CHECK_EQ(kv_layout_group_bits(RAD_KVQ_I8H), 516);
    CHECK_EQ(kv_layout_group_bits(RAD_KVQ_NONE), 512);
    CHECK_EQ(kv_layout_group_bits(RAD_KVQ_V4), 400);
    CHECK_EQ(kv_layout_group_bits(RAD_KVQ_K6V4), 352);
    CHECK_EQ(kv_layout_group_bits(RAD_KVQ_K4H), 288);
    CHECK_EQ(kv_layout_group_bits(RAD_KVQ_K4HV2H), 224);
    /* e2m1h (issue #18's experiment): k4h_v4's 288, an E4M3 byte a 16 where k4h_v4 has an f16 a 32. */
    CHECK_EQ(kv_layout_group_bits(RAD_KVQ_E2M1H), 288);
    CHECK_EQ(kv_layout_group_bits(99), -1);
    for (int q : kv_layouts()) CHECK_EQ(kv_layout_of(kv_layout_name(q)), q);
}

/* Synthetic sensitivities, a thousand instances of up to six layers, each against every budget
 * from the narrowest assignment to past the widest: the solver's bits never exceed the budget and
 * its KL is exhaustive search's, to the rounding of a different summation order. */
TEST(the_solver_finds_the_least_kl_within_the_budget_on_synthetic_sensitivities) {
    std::mt19937 rng(20261102);
    std::uniform_real_distribution<double> u(0.0, 1e-3);
    int cases = 0;
    for (int t = 0; t < 1000; ++t) {
        const size_t n = 1 + (size_t)(rng() % 6);
        std::vector<std::vector<KvCandidate>> layers(n);
        for (auto& c : layers) {
            /* Narrower costs at least as much as wider, as a real layer does; a third of the
             * layers tolerate k6_v4 for free, so ties are exercised too. */
            const double a = u(rng), b = (rng() % 3 == 0) ? 0.0 : a + u(rng);
            c = { fp8(), v4(rng() % 3 == 0 ? 0.0 : a), k6v4(std::max(a, b)) };
        }
        const int64_t lo = 352 * (int64_t)n, hi = 512 * (int64_t)n + 16;
        for (int64_t budget = lo; budget <= hi; budget += 8) {
            KvAssignment a;
            REQUIRE_EQ(kv_autoquant_solve(layers, budget, &a), RAD_OK);
            REQUIRE_EQ(a.pick.size(), n);
            int64_t bits = 0;
            double kl = 0;
            for (size_t l = 0; l < n; ++l) {
                REQUIRE(a.pick[l] >= 0 && a.pick[l] < 3);
                bits += layers[l][(size_t)a.pick[l]].bits;
                kl += layers[l][(size_t)a.pick[l]].kl;
            }
            CHECK_EQ(a.bits, bits);
            CHECK(a.bits <= budget);
            CHECK_EQ(a.kl, kl);
            CHECK_NEAR(a.kl, brute_force(layers, budget), 1e-15);
            ++cases;
        }
    }
    CHECK(cases > 10000);
}

/* Where a greedy exchange (narrow whichever layer costs least per bit saved, until the budget
 * holds) goes wrong. At 864 bits over two layers 160 must go: greedy narrows layer 0 to fp8_v4
 * first (0.0001 for 112 bits), and the 48 still missing then cost a second narrowing, 0.00105 or
 * 0.0010 in all, where layer 1 alone at k6_v4 saves the 160 for 0.00095. */
TEST(the_solver_takes_the_optimum_a_greedy_exchange_misses) {
    const std::vector<std::vector<KvCandidate>> layers = {
        { fp8(), v4(0.0001), k6v4(0.0100) },
        { fp8(), v4(0.0009), k6v4(0.00095) },
    };
    KvAssignment a;
    REQUIRE_EQ(kv_autoquant_solve(layers, 864, &a), RAD_OK);
    CHECK(kvq_of(layers, a) == std::vector<int>({ RAD_KVQ_NONE, RAD_KVQ_K6V4 }));
    CHECK_EQ(a.bits, (int64_t)864);
    CHECK_NEAR(a.kl, 0.00095, 1e-15);
    /* With 112 to give up, layer 0's fp8_v4 is the cheap way. */
    REQUIRE_EQ(kv_autoquant_solve(layers, 512 + 400, &a), RAD_OK);
    CHECK(kvq_of(layers, a) == std::vector<int>({ RAD_KVQ_V4, RAD_KVQ_NONE }));
    CHECK_NEAR(a.kl, 0.0001, 1e-15);
}

/* The budget's ends, the tie-break and determinism. A layer that narrows for free stays wide
 * unless the budget needs its bits: the candidate order (widest first) decides a tie, so a
 * measurement that sees no cost does not narrow a layer for nothing. */
TEST(the_solver_keeps_ties_wide_and_answers_the_same_every_time) {
    const std::vector<std::vector<KvCandidate>> layers = {
        { fp8(), v4(0.0), k6v4(0.0) },
        { fp8(), v4(0.0002), k6v4(0.0009) },
        { fp8(), v4(0.0), k6v4(0.0003) },
    };
    KvAssignment a;
    REQUIRE_EQ(kv_autoquant_solve(layers, 3 * 512, &a), RAD_OK);
    CHECK(kvq_of(layers, a) == std::vector<int>({ RAD_KVQ_NONE, RAD_KVQ_NONE, RAD_KVQ_NONE }));
    CHECK_EQ(a.kl, 0.0);
    REQUIRE_EQ(kv_autoquant_solve(layers, 1000000, &a), RAD_OK);   /* past the widest */
    CHECK_EQ(a.bits, (int64_t)3 * 512);
    REQUIRE_EQ(kv_autoquant_solve(layers, 3 * 352, &a), RAD_OK);   /* only the narrowest fits */
    CHECK(kvq_of(layers, a) == std::vector<int>({ RAD_KVQ_K6V4, RAD_KVQ_K6V4, RAD_KVQ_K6V4 }));
    /* 112 bits to give up: layer 0's k6_v4 is free and enough. */
    REQUIRE_EQ(kv_autoquant_solve(layers, 3 * 512 - 160, &a), RAD_OK);
    CHECK(kvq_of(layers, a) == std::vector<int>({ RAD_KVQ_K6V4, RAD_KVQ_NONE, RAD_KVQ_NONE }));
    CHECK_EQ(a.kl, 0.0);

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0.0, 1e-3);
    std::vector<std::vector<KvCandidate>> big(64);
    for (auto& c : big) c = { fp8(), v4(u(rng)), k6v4(u(rng)) };
    KvAssignment x, y;
    REQUIRE_EQ(kv_autoquant_solve(big, 64 * 384, &x), RAD_OK);
    REQUIRE_EQ(kv_autoquant_solve(big, 64 * 384, &y), RAD_OK);
    CHECK(x.pick == y.pick);
    CHECK_EQ(x.bits, y.bits);
    CHECK_EQ(x.kl, y.kl);
    CHECK(x.bits <= 64 * 384);
}

TEST(the_solver_refuses_what_it_cannot_assign) {
    KvAssignment a;
    a.pick = { 1 };
    CHECK_EQ(kv_autoquant_solve({}, 512, &a), RAD_E_INVAL);
    CHECK(a.pick.empty());
    CHECK_EQ(kv_autoquant_solve({ {} }, 512, &a), RAD_E_INVAL);
    CHECK_EQ(kv_autoquant_solve({ { { RAD_KVQ_NONE, 0, 0.0 } } }, 512, &a), RAD_E_INVAL);
    CHECK_EQ(kv_autoquant_solve({ { fp8(), v4(std::nan("")) } }, 512, &a), RAD_E_INVAL);
    CHECK_EQ(kv_autoquant_solve({ { fp8(), v4(INFINITY) } }, 512, &a), RAD_E_INVAL);
    CHECK_EQ(kv_autoquant_solve({ { fp8(), k6v4(0.1) } }, 351, &a), RAD_E_INVAL);
    CHECK(a.pick.empty());
    CHECK_EQ(kv_autoquant_solve({ { fp8(), k6v4(0.1) } }, 352, &a), RAD_OK);
}

/* The converse question on the same synthetic sensitivities: for limits from nothing to past the
 * narrowest assignment's sum, the fewest bits within the limit and the least KL at those bits,
 * as exhaustive search finds them. */
TEST(the_kl_solver_finds_the_fewest_bits_within_the_limit_on_synthetic_sensitivities) {
    std::mt19937 rng(20261103);
    std::uniform_real_distribution<double> u(0.0, 1e-3);
    int cases = 0;
    for (int t = 0; t < 500; ++t) {
        const size_t n = 1 + (size_t)(rng() % 5);
        std::vector<std::vector<KvCandidate>> layers(n);
        for (auto& c : layers) {
            const double a = u(rng), b = a + u(rng), d = b + u(rng);
            c = { fp8(), v4(rng() % 3 == 0 ? 0.0 : a), k6v4(b), k4h(d), k4hv2h(d + u(rng)) };
        }
        double most = 0;
        for (auto& c : layers) most += c.back().kl;
        for (int k = 0; k <= 40; ++k) {
            const double limit = most * 1.1 * k / 40.0;
            KvAssignment a;
            REQUIRE_EQ(kv_autoquant_solve_kl(layers, limit, &a), RAD_OK);
            REQUIRE_EQ(a.pick.size(), n);
            int64_t bits = 0;
            double kl = 0;
            for (size_t l = 0; l < n; ++l) {
                bits += layers[l][(size_t)a.pick[l]].bits;
                kl += layers[l][(size_t)a.pick[l]].kl;
            }
            CHECK_EQ(a.bits, bits);
            CHECK_EQ(a.kl, kl);
            CHECK(a.kl <= limit * (1 + 1e-12));
            const auto [want_bits, want_kl] = brute_force_kl(layers, limit);
            CHECK_EQ(a.bits, want_bits);
            CHECK_NEAR(a.kl, want_kl, 1e-15);
            ++cases;
        }
    }
    CHECK(cases > 10000);
}

/* The two questions agree: the least KL within B bits, asked back as a limit, is met by B bits or
 * fewer; a limit copied from a solved cost is met by that cost's assignment despite rounding. A
 * limit under every assignment is not found, and a limit that is not a KL is refused. */
TEST(the_kl_solver_agrees_with_the_budget_solver_and_refuses_what_it_cannot_meet) {
    const std::vector<std::vector<KvCandidate>> layers = {
        { fp8(), v4(0.0001), k6v4(0.0100) },
        { fp8(), v4(0.0009), k6v4(0.00095) },
    };
    KvAssignment a, b;
    REQUIRE_EQ(kv_autoquant_solve(layers, 864, &a), RAD_OK);
    REQUIRE_EQ(kv_autoquant_solve_kl(layers, a.kl, &b), RAD_OK);
    CHECK(b.bits <= a.bits);
    CHECK(kvq_of(layers, b) == std::vector<int>({ RAD_KVQ_NONE, RAD_KVQ_K6V4 }));
    /* No room at all: every layer stays wide. All the room: every layer narrowest. */
    REQUIRE_EQ(kv_autoquant_solve_kl(layers, 0.0, &b), RAD_OK);
    CHECK(kvq_of(layers, b) == std::vector<int>({ RAD_KVQ_NONE, RAD_KVQ_NONE }));
    REQUIRE_EQ(kv_autoquant_solve_kl(layers, 1.0, &b), RAD_OK);
    CHECK_EQ(b.bits, (int64_t)2 * 352);

    const std::vector<std::vector<KvCandidate>> costly = { { fp8(0.002), k6v4(0.003) } };
    b.pick = { 1 };
    CHECK_EQ(kv_autoquant_solve_kl(costly, 0.001, &b), RAD_E_NOTFOUND);
    CHECK(b.pick.empty());
    CHECK_EQ(kv_autoquant_solve_kl(costly, 0.002, &b), RAD_OK);
    CHECK_EQ(kv_autoquant_solve_kl(layers, -1e-3, &b), RAD_E_INVAL);
    CHECK_EQ(kv_autoquant_solve_kl(layers, std::nan(""), &b), RAD_E_INVAL);
    CHECK_EQ(kv_autoquant_solve_kl(layers, INFINITY, &b), RAD_E_INVAL);
    CHECK_EQ(kv_autoquant_solve_kl({}, 0.1, &b), RAD_E_INVAL);
}

/* ================================================================== the card's costs */

namespace {

/* What rad-autoquant v1 measured on the card (tests/data/autoquant/flashnext_v1_costs.txt): the
 * all-fp8 base, and each of Flash-Next's 12 attention layers alone in each plain layout no wider
 * than fp8, as the solver's candidates widest first (costs above the base, clamped at 0 as the
 * driver does); and the recipes measured afterwards. i8h had no runs then, so it is no candidate
 * here. */
struct Card {
    double base = -1;
    std::vector<int> layers;
    std::vector<std::vector<KvCandidate>> cand;
    std::map<std::string, double> recipe;
};

Card card() {
    Card c;
    std::map<int, std::map<std::string, double>> kl;
    std::istringstream in(tiny::read_text(std::string(RAD_AUTOQUANT_FIXTURES) +
                                          "/flashnext_v1_costs.txt"));
    for (std::string line; std::getline(in, line);) {
        std::istringstream w(line);
        std::string a, b;
        if (!(w >> a) || a[0] == '#') continue;
        if (a == "base") { w >> c.base; continue; }
        if (a == "recipe") { double bits, m; w >> b >> bits >> m; c.recipe[b] = m; continue; }
        double m = 0;
        w >> b >> m;
        kl[std::stoi(a)][b] = m;
    }
    for (const auto& [l, by] : kl) {
        c.layers.push_back(l);
        std::vector<KvCandidate> v = { fp8() };
        for (int q : kv_layouts())
            if (by.count(kv_layout_name(q)))
                v.push_back({ q, kv_layout_group_bits(q), std::max(0.0, by.at(kv_layout_name(q)) - c.base) });
        c.cand.push_back(v);
    }
    return c;
}

/* Bits a group of 32 K and 32 V elements, summed over n layers, for an average of `b`. */
int64_t budget_of(double b, size_t n) { return (int64_t)std::llround(b * 64 * (double)n); }

/* A measurement that does not add up: the summed costs times 1 + g for every narrowed layer past
 * the first, g fitted to the card's 5.5-bit recipe (10 layers narrowed, 0.0101 measured above the
 * base where the costs summed to 0.00868). */
double non_additive(const std::vector<std::vector<KvCandidate>>& c, const KvAssignment& a, double g) {
    int k = 0;
    for (size_t l = 0; l < c.size(); ++l) k += c[l][(size_t)a.pick[l]].kvq != RAD_KVQ_NONE;
    return a.kl * (1 + g * std::max(0, k - 1));
}

}  /* namespace */

/* The two questions are one table read two ways, held on the card's own costs: at each budget the
 * v1 recipes were solved for, the least-KL assignment's KL, taken as a limit, is met by no more
 * bits, and at those bits by nothing of lower KL. The 5.5-bit one is the recipe v1 wrote, measured
 * 0.0363 against a predicted 0.0349: the error the measured fit exists for. */
TEST(on_the_cards_costs_the_kl_solver_answers_the_bit_budget_backwards) {
    const Card c = card();
    REQUIRE_EQ(c.layers.size(), (size_t)12);
    CHECK_NEAR(c.base, 0.026249, 1e-9);
    for (const auto& l : c.cand) REQUIRE_EQ(l.size(), (size_t)5);
    const size_t n = c.layers.size();
    KvAssignment a;
    REQUIRE_EQ(kv_autoquant_solve(c.cand, budget_of(5.5, n), &a), RAD_OK);
    CHECK(kvq_of(c.cand, a) ==
          std::vector<int>({ RAD_KVQ_K6V4, RAD_KVQ_K6V4, RAD_KVQ_K6V4, RAD_KVQ_NONE, RAD_KVQ_K6V4,
                             RAD_KVQ_K4H, RAD_KVQ_NONE, RAD_KVQ_K6V4, RAD_KVQ_K4H, RAD_KVQ_K4H,
                             RAD_KVQ_K4H, RAD_KVQ_K4H }));
    CHECK_NEAR(c.base + a.kl, 0.0349, 5e-5);
    CHECK(c.recipe.at("aq-b5.5") - (c.base + a.kl) > 0.001);

    for (double b : { 7.0, 6.0, 5.5, 5.0, 4.5, 4.0 }) {
        KvAssignment w, x, y;
        REQUIRE_EQ(kv_autoquant_solve(c.cand, budget_of(b, n), &w), RAD_OK);
        REQUIRE_EQ(kv_autoquant_solve_kl(c.cand, w.kl, &x), RAD_OK);
        CHECK(x.bits <= w.bits);
        CHECK(x.kl <= w.kl * (1 + 1e-12));
        REQUIRE_EQ(kv_autoquant_solve(c.cand, x.bits, &y), RAD_OK);
        CHECK_EQ(x.kl, y.kl);
        /* 16 bits, the layouts' common divisor, fewer: over the limit or not assignable at all. */
        if (kv_autoquant_solve(c.cand, x.bits - 16, &y) == RAD_OK) CHECK(y.kl > w.kl);
    }
    /* Past every cost all layers go to the narrowest; at no room, every layer fp8. */
    REQUIRE_EQ(kv_autoquant_solve_kl(c.cand, 1.0, &a), RAD_OK);
    CHECK_EQ(a.bits, (int64_t)n * 224);
    REQUIRE_EQ(kv_autoquant_solve_kl(c.cand, 0.0, &a), RAD_OK);
    CHECK_EQ(a.bits, (int64_t)n * 512);
    CHECK_EQ(a.kl, 0.0);
    KvAssignment x, y;
    REQUIRE_EQ(kv_autoquant_solve_kl(c.cand, 0.007, &x), RAD_OK);
    REQUIRE_EQ(kv_autoquant_solve_kl(c.cand, 0.007, &y), RAD_OK);
    CHECK(x.pick == y.pick);
    CHECK_EQ(x.kl, y.kl);
}

/* Equal bits and equal KL: the earlier candidate, so a free rotation or V mean is not taken for
 * nothing; the variants' order (plain first) is the tie-break the driver relies on. */
TEST(the_kl_solver_keeps_the_plain_variant_where_rotation_or_the_mean_buy_nothing) {
    const std::vector<std::vector<KvCandidate>> tie = {
        { fp8(), { RAD_KVQ_NONE | (int)RAD_KVQ_ROTATED, 512, 0.0 }, k6v4(0.001),
          { RAD_KVQ_K6V4 | (int)RAD_KVQ_VMEAN, 352, 0.001 } } };
    KvAssignment a;
    REQUIRE_EQ(kv_autoquant_solve_kl(tie, 0.0, &a), RAD_OK);
    CHECK_EQ(a.pick[0], 0);
    REQUIRE_EQ(kv_autoquant_solve_kl(tie, 0.001, &a), RAD_OK);
    CHECK_EQ(a.pick[0], 2);
}

/* ================================================================== the measured fit */

/* The fit on the card's costs, at the limit the 5.5-bit recipe was predicted to meet: the first
 * solution measures over, each round's room is below the last rejected prediction and scaled by
 * allowed / measured, no assignment comes back, the bits rise, and it ends within the limit in a
 * few rounds. With costs that do add up, it is the KL solver's answer in one round. */
TEST(the_fit_tightens_a_non_additive_cost_until_the_measurement_is_within_the_limit) {
    const Card c = card();
    const double g = ((0.0363 - c.base) / 0.00868 - 1) / 9;
    const double limit = 0.00868;
    std::vector<KvRound> trail;
    int calls = 0;
    const KvMeasure measure = [&](const KvAssignment& a, std::vector<KvCheck>* ch) {
        ++calls;
        ch->push_back({ non_additive(c.cand, a, g), limit });
        return (int)RAD_OK;
    };
    REQUIRE_EQ(kv_autoquant_fit(c.cand, limit, 6, measure, &trail), RAD_OK);
    REQUIRE(trail.size() >= 2);
    CHECK_EQ(calls, (int)trail.size());
    CHECK(!trail[0].within);
    CHECK(trail.back().within);
    CHECK(trail.size() <= 4);
    CHECK(non_additive(c.cand, trail.back().pick, g) <= limit);
    for (size_t r = 1; r < trail.size(); ++r) {
        CHECK(trail[r].allowed < trail[r - 1].pick.kl);
        CHECK(trail[r].allowed < trail[r - 1].allowed);
        CHECK(trail[r].pick.bits > trail[r - 1].pick.bits);
        CHECK(trail[r].pick.pick != trail[r - 1].pick.pick);
        /* The rejected prediction times allowed / measured. */
        CHECK_NEAR(trail[r].allowed, trail[r - 1].pick.kl * limit /
                                         non_additive(c.cand, trail[r - 1].pick, g), 1e-15);
    }
    /* Still far below fp8 everywhere: 12 x 8 bits is 6144. */
    CHECK(trail.back().pick.bits < 12 * 512 * 3 / 4);

    REQUIRE_EQ(kv_autoquant_fit(c.cand, limit, 6, [&](const KvAssignment& a, std::vector<KvCheck>* ch) {
        ch->push_back({ a.kl, limit });
        return (int)RAD_OK;
    }, &trail), RAD_OK);
    REQUIRE_EQ(trail.size(), (size_t)1);
    KvAssignment want;
    REQUIRE_EQ(kv_autoquant_solve_kl(c.cand, limit, &want), RAD_OK);
    CHECK(trail[0].pick.pick == want.pick);
    CHECK(trail[0].within);
}

/* Where the fit stops short: after its rounds; where a rejected assignment predicted no cost (the
 * costs have nothing left to give up); before a solution of no fewer bits than the cap; and on a
 * measurement's error. The tightest failing check sets the next room, a top-1 check as much as the
 * KL one. */
TEST(the_fit_stops_after_its_rounds_on_a_free_assignment_at_the_cap_and_on_a_failed_run) {
    const std::vector<std::vector<KvCandidate>> c = {
        { fp8(), v4(0.0), k6v4(0.002) }, { fp8(), v4(0.001), k6v4(0.003) } };
    std::vector<KvRound> trail;
    /* Always twice what is allowed on KL, and four times on top-1. */
    const KvMeasure over = [&](const KvAssignment& a, std::vector<KvCheck>* ch) {
        ch->push_back({ 2 * a.kl + 1e-4, a.kl + 0.5e-4 });
        ch->push_back({ 4e-3, 1e-3 });
        return (int)RAD_OK;
    };
    REQUIRE_EQ(kv_autoquant_fit(c, 0.0055, 10, over, &trail), RAD_OK);
    REQUIRE_EQ(trail.size(), (size_t)3);
    CHECK_NEAR(trail[0].pick.kl, 0.005, 1e-15);
    CHECK_NEAR(trail[1].allowed, 0.005 / 4, 1e-15);
    CHECK(!trail.back().within);
    CHECK_EQ(trail.back().pick.kl, 0.0);   /* layer 0's free fp8_v4: nothing left to give up */
    CHECK(kvq_of(c, trail.back().pick) == std::vector<int>({ RAD_KVQ_V4, RAD_KVQ_NONE }));
    REQUIRE_EQ(kv_autoquant_fit(c, 0.0055, 1, over, &trail), RAD_OK);
    CHECK_EQ(trail.size(), (size_t)1);
    /* A cap at the first solution's bits measures nothing; one past them, the rounds under it. */
    REQUIRE_EQ(kv_autoquant_fit(c, 0.0055, 10, over, &trail, 704), RAD_OK);
    CHECK(trail.empty());
    REQUIRE_EQ(kv_autoquant_fit(c, 0.0055, 10, over, &trail, 801), RAD_OK);
    REQUIRE_EQ(trail.size(), (size_t)2);
    CHECK_EQ(trail[1].pick.bits, (int64_t)800);
    CHECK_EQ(kv_autoquant_fit(c, 0.005, 0, over, &trail), RAD_E_INVAL);
    CHECK_EQ(kv_autoquant_fit(c, -1.0, 3, over, &trail), RAD_E_INVAL);
    CHECK_EQ(kv_autoquant_fit(c, 0.005, 3, [](const KvAssignment&, std::vector<KvCheck>*) {
        return (int)RAD_E_IO;
    }, &trail), RAD_E_IO);
    CHECK(trail.empty());
}

/* ================================================================== the confidence interval */

/* Independent positions: the batch-means half-width is the textbook 1.96 sigma / sqrt(n), to the
 * noise of estimating it from 32 batches. Correlated positions (an AR(1) run, rho 0.9): wider by
 * about sqrt((1 + rho) / (1 - rho)) = 4.4, which the spread of single positions misses. Documents
 * of very different means: the documents are strata, so the half-width is the noise within them,
 * not the spread between them. */
TEST(the_interval_from_the_rows_is_batch_means_with_the_documents_as_strata) {
    std::mt19937 rng(11);
    std::normal_distribution<double> z(0.0, 1.0);
    const size_t n = 32768;
    std::vector<double> iid(n), ar(n), docs(n);
    double p = 0;
    for (size_t i = 0; i < n; ++i) {
        iid[i] = 0.03 + 0.05 * z(rng);
        p = 0.9 * p + std::sqrt(1 - 0.81) * z(rng);
        ar[i] = 0.03 + 0.05 * p;
        docs[i] = (i < n / 2 ? 0.01 : 0.09) + 0.05 * z(rng);
    }
    const double textbook = 1.96 * 0.05 / std::sqrt((double)n);
    KvInterval a;
    REQUIRE_EQ(kv_autoquant_interval(iid, {}, &a), RAD_OK);
    CHECK_EQ(a.df, (int64_t)31);
    CHECK_NEAR(a.mean, 0.03, 4 * textbook);
    CHECK(a.half > 0.7 * textbook && a.half < 1.3 * textbook);
    REQUIRE_EQ(kv_autoquant_interval(ar, {}, &a), RAD_OK);
    CHECK(a.half > 3 * textbook && a.half < 6 * textbook);
    REQUIRE_EQ(kv_autoquant_interval(docs, { (int64_t)n / 2, (int64_t)n / 2 }, &a), RAD_OK);
    CHECK_EQ(a.df, (int64_t)30);
    CHECK_NEAR(a.mean, 0.05, 4 * textbook);
    CHECK(a.half > 0.7 * textbook && a.half < 1.3 * textbook);
    /* As one document, the two halves' offset is all the spread there is. */
    REQUIRE_EQ(kv_autoquant_interval(docs, {}, &a), RAD_OK);
    CHECK(a.half > 10 * textbook);

    CHECK_NEAR(kv_autoquant_t975(1), 12.706, 1e-3);
    CHECK_NEAR(kv_autoquant_t975(11), 2.201, 1e-3);
    CHECK_NEAR(kv_autoquant_t975(28), 2.048, 1e-3);
    CHECK_NEAR(kv_autoquant_t975(31), 2.040, 1e-3);
    CHECK_NEAR(kv_autoquant_t975(120), 1.980, 1e-3);
    CHECK_NEAR(kv_autoquant_t975(1000000), 1.960, 1e-3);

    CHECK_EQ(kv_autoquant_interval(iid, { 100, 200 }, &a), RAD_E_INVAL);
    CHECK_EQ(kv_autoquant_interval({ 0.1 }, {}, &a), RAD_E_INVAL);
    CHECK_EQ(kv_autoquant_interval({ 0.1, NAN, 0.2 }, {}, &a), RAD_E_INVAL);
    CHECK_EQ(kv_autoquant_interval({}, {}, &a), RAD_E_INVAL);
    REQUIRE_EQ(kv_autoquant_interval({ 0.1, 0.3 }, {}, &a), RAD_OK);
    CHECK_EQ(a.mean, 0.2);
    CHECK_EQ(a.df, (int64_t)1);
}

/* ================================================================== the variants */

/* The candidates come from the layout table and the recipe parser: each layout plain, `vmean`,
 * `rotated` and both, where the parser takes the words and they make another cache. bf16 takes
 * neither word, so it is one variant; fp8 has no integer V to centre; i8h, k4h_v4 and k4h_v2h are
 * rotated by their names. Each is a recipe line
 * config_parse loads, with its means, as the flags the variant names. */
/* e2m1h is opt-in: without opt_in no variant of it exists (so rad-autoquant's default candidates,
 * and every test of them below, are the table's other layouts as before); with it, `e2m1h` (rotated
 * by its name) and `e2m1h vmean` join, right after k4h_v4's two, at k4h_v4's bits. */
TEST(e2m1h_is_a_variant_only_when_asked_for) {
    const int R = RAD_KVQ_ROTATED, M = RAD_KVQ_VMEAN;
    for (const KvVariant& v : kv_autoquant_variants(true))
        CHECK((v.kvq & RAD_KVQ_LAYOUT) != RAD_KVQ_E2M1H);
    const std::vector<KvVariant> all = kv_autoquant_variants(true, true);
    REQUIRE_EQ(all.size(), (size_t)19);
    CHECK_EQ(all[15].kvq, (int32_t)(RAD_KVQ_E2M1H | R));
    CHECK_EQ(all[15].words, std::string("e2m1h"));
    CHECK_EQ(all[15].bits, 288);
    CHECK_EQ(all[16].kvq, (int32_t)(RAD_KVQ_E2M1H | R | M));
    CHECK_EQ(all[16].words, std::string("e2m1h vmean"));
}

TEST(every_variant_is_a_recipe_line_the_server_loads_as_that_variant) {
    const int R = RAD_KVQ_ROTATED, M = RAD_KVQ_VMEAN;
    std::vector<int32_t> plain, centred;
    for (const KvVariant& v : kv_autoquant_variants(false)) plain.push_back(v.kvq);
    CHECK(plain == std::vector<int32_t>({ RAD_KVQ_BF16, RAD_KVQ_I8H | R, RAD_KVQ_NONE, RAD_KVQ_NONE | R,
                                          RAD_KVQ_V4, RAD_KVQ_V4 | R, RAD_KVQ_K6V4,
                                          RAD_KVQ_K6V4 | R, RAD_KVQ_K4H | R, RAD_KVQ_K4HV2H | R }));
    const std::vector<KvVariant> all = kv_autoquant_variants(true);
    for (const KvVariant& v : all) centred.push_back(v.kvq);
    CHECK(centred == std::vector<int32_t>({
              RAD_KVQ_BF16, RAD_KVQ_I8H | R, RAD_KVQ_I8H | R | M, RAD_KVQ_NONE, RAD_KVQ_NONE | R, RAD_KVQ_V4,
              RAD_KVQ_V4 | M, RAD_KVQ_V4 | R, RAD_KVQ_V4 | R | M, RAD_KVQ_K6V4, RAD_KVQ_K6V4 | M,
              RAD_KVQ_K6V4 | R, RAD_KVQ_K6V4 | R | M, RAD_KVQ_K4H | R, RAD_KVQ_K4H | R | M,
              RAD_KVQ_K4HV2H | R, RAD_KVQ_K4HV2H | R | M }));
    REQUIRE_EQ(all.size(), (size_t)17);
    CHECK_EQ(all[0].words, std::string("bf16"));
    CHECK_EQ(all[1].words, std::string("i8h"));
    CHECK_EQ(all[8].words, std::string("fp8_v4 rotated vmean"));
    CHECK_EQ(all[13].words, std::string("k4h_v4"));
    CHECK_EQ(all[14].words, std::string("k4h_v4 vmean"));

    tiny::TempDir dir("autoquant-variants");
    REQUIRE(dir.ok());
    REQUIRE(tiny::write_text(dir / "m.kvmean", "kv_vmean 1 head_dim 2 kv_heads 1 rows 9\n"
                                               "layer 3 head 0 0.5 -0.25\nlayer 7 head 0 1 2\n"));
    for (const KvVariant& v : all) {
        CHECK_EQ(v.bits, kv_layout_group_bits(v.kvq & RAD_KVQ_LAYOUT));
        const std::string recipe = dir / "r.kvrecipe";
        REQUIRE(tiny::write_text(recipe, std::string(v.kvq & M ? "vmean m.kvmean\n" : "") +
                                             "layer 3 " + v.words + "\nlayer 7 fp8\n"));
        Config c;
        Argv a({ "--model", "/m/x.rad", "--kv-cache-dtype", "fp8", "--kv-recipe", recipe });
        const int rc = config_parse(a.argc(), a.argv.data(), &c);
        CHECK_EQ(rc, RAD_OK);
        if (rc != RAD_OK) { fprintf(stderr, "    variant '%s'\n", v.words.c_str()); continue; }
        REQUIRE_EQ(c.kv_recipe.size(), (size_t)8);
        CHECK_EQ(c.kv_recipe[3], v.kvq);
        CHECK_EQ(c.kv_recipe[7], (int32_t)RAD_KVQ_NONE);
    }
}

/* ================================================================== the driver */

namespace {

/* The stand-in engine: it answers a KL-mode run (`--kv-recipe R ... --kld-out O`) with a report
 * whose mean KL is 0.04 plus, for each recipe line, the table's cost of that layer in that variant
 * (`3 k6_v4+vmean 0.001`, the words joined by '+'; a variant the table does not name costs what
 * its plain layout does, and a layout it does not name nothing), and a top-1 agreement of 0.95
 * less twice that; it appends its arguments to `calls`. Table lines change the model: `fail N V`
 * makes a mixed run (not every layer in one variant) with layer N in V exit 1; `pair P` adds P for
 * every layer narrower than fp8 past the first, costs that do not add up when layouts are mixed,
 * as the card measures them; `all V KL` is what a recipe of every layer in V costs, whatever its
 * layers' costs sum to. With `scored` positions a document the reference scores them (score_from)
 * and each run writes its .rows: the mean KL as 0.5 and 1.5 times it in alternating stretches of
 * 37 positions, so the positions are correlated and the interval has a width, and a top-1 column
 * agreeing on that share of every 100 positions.
 *
 * The noise floor and the knobs (#16): `flag WORD KL [DNLL]` adds KL (and an NLL change) to a run
 * whose arguments hold the word WORD (`--max-num-batched-tokens`, the noise run's; `i8` and `f16`,
 * the knobs' values); `refuse WORD` makes such a run exit 1, as an engine that does not serve a
 * knob refuses it at startup. A layer's cost adds to the run's KL whatever the flags, so a recipe
 * measures the base's 0.04 plus its layers' costs and can sit far under a noise run's 0.04 + KL,
 * as the card measures it (the floor perturbs every op, a recipe only the cache). A cost line's
 * fourth field is that entry's NLL
 * change (default 0): the report's perplexity ratio is exp of the summed changes, and the .rows'
 * candidate NLL is 1 + the change in the KL's stretches, the reference's 1. Its log carries the
 * engine's KV report: the attention cache at 64 elements a layer and token (a layout's group
 * bits / 2 bytes a 4-token page), the block keys at 3.00 KiB a page (1.59 KiB with
 * `--qsa-bkey-dtype i8`) and the delta-net state at 54.00 MiB a slot (27.00 MiB with
 * `--state-cache-dtype f16`). */
struct Bench {
    tiny::TempDir dir{ "autoquant" };
    std::string engine = dir / "engine.sh", table = dir / "costs", calls = dir / "calls",
                ref = dir / "ref", out = dir / "model.kvrecipe", nrows = dir / "nrows";

    explicit Bench(const std::vector<int64_t>& doc_tokens = { 40000, 65536 }, int64_t scored = 0) {
        tiny::write_text(engine,
            "#!/bin/sh\n"
            "echo \"$*\" >> '" + calls + "'\n"
            "args=\"$*\"\n"
            "recipe=; out=\n"
            "while [ $# -gt 0 ]; do\n"
            "  case \"$1\" in --kv-recipe) recipe=$2; shift;; --kld-out) out=$2; shift;; esac\n"
            "  shift\n"
            "done\n"
            "r=$(awk -v t='" + table + "' -v args=\"$args\" 'BEGIN { na = split(args, x, \" \");\n"
            "    for (i = 1; i <= na; ++i) fl[x[i]] = 1;\n"
            "    b[\"bf16\"] = 1024; b[\"i8h\"] = 516; b[\"fp8\"] = 512; b[\"fp8_v4\"] = 400;\n"
            "    b[\"k6_v4\"] = 352; b[\"k4h_v4\"] = 288; b[\"k4h_v2h\"] = 224;\n"
            "    while ((getline l < t) > 0) { split(l, a, \" \");\n"
            "    if (a[1] == \"fail\") f[a[2] \" \" a[3]] = 1; else if (a[1] == \"pair\") p = a[2];\n"
            "    else if (a[1] == \"all\") u[a[2]] = a[3];\n"
            "    else if (a[1] == \"flag\") { if (a[2] in fl) { g += a[3]; gd += a[4] } }\n"
            "    else if (a[1] == \"refuse\") { if (a[2] in fl) no = 1 }\n"
            "    else { c[a[1] \" \" a[2]] = a[3]; d[a[1] \" \" a[2]] = a[4] } } }\n"
            "  $1 == \"layer\" { w = $3; for (i = 4; i <= NF; ++i) w = w \"+\" $i; k = $2 \" \" w;\n"
            "    s += (k in c) ? c[k] : c[$2 \" \" $3]; e += (k in c) ? d[k] : d[$2 \" \" $3];\n"
            "    page += b[$3] / 2; nl2++;\n"
            "    if ($3 != \"fp8\" && $3 != \"i8h\" && $3 != \"bf16\") n++;\n"
            "    if (f[k]) bad = 1; if (!nl++) first = w; else if (w != first) mixed = 1 }\n"
            "  END { if (no) { print \"stand-in: refused the flag\" > \"/dev/stderr\"; exit 1 }\n"
            "        if (bad && mixed) { print \"stand-in: refused\" > \"/dev/stderr\"; exit 1 }\n"
            "        if (n > 1) s += p * (n - 1); if (!mixed && (first in u)) s = u[first]\n"
            "        s += g; e += gd\n"
            "        printf \"%.9f %.9f %.12f %d %d %.9f\\n\", 0.04 + s, 0.95 - 2 * s, exp(e), page, nl2, e }' \"$recipe\") || exit 1\n"
            "set -- $r\n"
            "printf '{\"all\": {\"kld\": {\"mean\": %s}, \"top1_agreement\": %s, \"ppl\": {\"candidate\": %s, "
            "\"reference\": 1, \"ratio\": %s}}}\\n' $1 $2 $3 $3 > \"$out\"\n"
            "bkey='3.00 KiB'; state='54.00 MiB'\n"
            "case \"$args\" in *'--qsa-bkey-dtype i8'*) bkey='1.59 KiB';; esac\n"
            "case \"$args\" in *'--state-cache-dtype f16'*) state='27.00 MiB';; esac\n"
            "echo \"    kv_attn        full    $5 layers  block    4 tok  page $4 B    10 blocks = 0 B  (40 tokens)\"\n"
            "echo \"    kv_gdn_state   linear  36 layers  state $state        2 slots = 0 B\"\n"
            "echo \"    kv_qsa_bkey    full    $5 layers  block    4 tok  page $bkey    10 blocks = 0 B  (40 tokens)\"\n"
            "[ -f '" + nrows + "' ] && perl -e '($n, $m, $t, $d) = @ARGV; for $i (0 .. $n - 1) {\n"
            "  $f = int($i / 37) % 2 ? 1.5 : 0.5; print pack(\"f<4\", $m * $f, 1 + $d * $f, 1, $i % 100 < 100 * $t ? 1 : 0) }'"
            " $(cat '" + nrows + "') $1 $2 $6 > \"$out.rows\"\n"
            "exit 0\n");
        ::chmod(engine.c_str(), 0755);
        ::mkdir(ref.c_str(), 0755);
        std::string docs;
        for (int64_t n : doc_tokens)
            docs += (docs.empty() ? "" : ", ") + std::string("{\"tokens\": ") + std::to_string(n) +
                    ", \"score_from\": " + std::to_string(scored ? n - 1 - scored : 0) + "}";
        tiny::write_text(ref + "/kld.json", "{\"format\": \"radiance-kld-1\", \"docs\": [" + docs + "]}\n");
        if (scored) tiny::write_text(nrows, std::to_string(scored * (int64_t)doc_tokens.size()) + "\n");
    }

    int run(const std::vector<std::string>& more, std::string* log) const {
        std::vector<std::string> a = { RAD_AUTOQUANT, "--kld-ref", ref, "-o", out,
                                       "--radiance", engine };
        a.insert(a.end(), more.begin(), more.end());
        return tiny::run(a, {}, log, 60);
    }
    int lines(const std::string& path) const {
        const std::string s = tiny::read_text(path);
        return (int)std::count(s.begin(), s.end(), '\n');
    }
};

}  /* namespace */

/* Four attention layers with costs from a table: one run with every layer in each variant, fp8 the
 * base (i8h measures no better here and costs more bits), one for each layer in each variant
 * screening keeps, the costs the runs' means above the base, the assignment the solver's on those
 * costs, and a recipe the server's own --kv-recipe parse loads -- a k4h_v4 layer, written as
 * `layer 7 k4h_v4`, loaded rotated. */
TEST(the_driver_measures_each_layer_and_writes_a_recipe_the_server_loads) {
    Bench b;
    REQUIRE(b.dir.ok());
    const std::map<std::pair<int, int>, double> cost = {
        { { 3, RAD_KVQ_V4 }, 0.0040 },  { { 3, RAD_KVQ_K6V4 }, 0.0090 },
        { { 7, RAD_KVQ_V4 }, 0.0001 },  { { 7, RAD_KVQ_K6V4 }, 0.0002 },
        { { 11, RAD_KVQ_V4 }, 0.0003 }, { { 11, RAD_KVQ_K6V4 }, 0.0030 },
        { { 15, RAD_KVQ_V4 }, 0.0000 }, { { 15, RAD_KVQ_K6V4 }, 0.0004 },
        { { 3, RAD_KVQ_K4H }, 0.0500 },  { { 7, RAD_KVQ_K4H }, 0.0001 },
        { { 11, RAD_KVQ_K4H }, 0.0500 }, { { 15, RAD_KVQ_K4H }, 0.0500 },
        { { 3, RAD_KVQ_K4HV2H }, 0.0900 },  { { 7, RAD_KVQ_K4HV2H }, 0.0900 },
        { { 11, RAD_KVQ_K4HV2H }, 0.0900 }, { { 15, RAD_KVQ_K4HV2H }, 0.0900 },
    };
    std::string t;
    for (const auto& [k, v] : cost)
        t += std::to_string(k.first) + " " + kv_layout_name(k.second) + " " + std::to_string(v) + "\n";
    REQUIRE(tiny::write_text(b.table, t));

    std::string log;
    const int rc = b.run({ "--layers", "3,7,11,15", "--bits", "6.5", "--", "--model", "/m/x.rad",
                           "--tp", "2" }, &log);
    if (rc != 0) fprintf(stderr, "    exited %d:\n%s\n", rc, log.c_str());
    REQUIRE_EQ(rc, 0);

    /* 10 uniform runs (bf16, i8h, fp8, fp8 rotated, fp8_v4, fp8_v4 rotated, k6_v4, k6_v4 rotated,
     * k4h_v4, k4h_v2h), then 4 layers x the 4 narrower layouts: screening drops bf16 and i8h (the
     * stand-in prices neither above fp8, so a narrower one measures as well for fewer bits) and
     * the rotated ones (priced as their plain layouts, and an exact tie keeps the earlier). Each with the engine's flags and the driver's. */
    CHECK_EQ(b.lines(b.calls), 26);
    CHECK(has(log, "screening drops bf16 (0.040000 at 16.00 bits; i8h: 0.040000 at 8.06)"));
    CHECK(has(log, "screening drops i8h (0.040000 at 8.06 bits; fp8: 0.040000 at 8.00)"));
    CHECK(has(log, "screening drops fp8_v4 rotated ("));
    CHECK(has(log, "the base, the most accurate: every layer fp8\n"));
    const std::string calls = tiny::read_text(b.calls);
    CHECK(has(calls, "--model /m/x.rad --tp 2 --kv-cache-dtype fp8 --kv-recipe "));
    CHECK(has(calls, "--kld-ref " + b.ref + " --kld-out "));

    /* The solver on the table's costs, in the driver's units, is what was written. */
    std::vector<std::vector<KvCandidate>> layers;
    for (int l : { 3, 7, 11, 15 })
        layers.push_back({ fp8(), v4(cost.at({ l, RAD_KVQ_V4 })), k6v4(cost.at({ l, RAD_KVQ_K6V4 })),
                           k4h(cost.at({ l, RAD_KVQ_K4H })), k4hv2h(cost.at({ l, RAD_KVQ_K4HV2H })) });
    KvAssignment want;
    REQUIRE_EQ(kv_autoquant_solve(layers, 1664, &want), RAD_OK);
    /* Layer 3 is the sensitive one: it keeps fp8, and the rest give up at least 384 bits, layer 7
     * to k4h_v4, which costs it less than k6_v4 (0.0004 against 0.0005 in all). */
    CHECK(kvq_of(layers, want) ==
          std::vector<int>({ RAD_KVQ_NONE, RAD_KVQ_K4H, RAD_KVQ_V4, RAD_KVQ_V4 }));
    CHECK_NEAR(want.kl, 0.0004, 1e-12);

    Config c;
    Argv a({ "--model", "/m/x.rad", "--kv-cache-dtype", "fp8", "--kv-recipe", b.out });
    REQUIRE_EQ(config_parse(a.argc(), a.argv.data(), &c), RAD_OK);
    const std::vector<int32_t> got = c.kv_recipe;
    REQUIRE_EQ(got.size(), (size_t)16);
    const std::vector<int> picked = kvq_of(layers, want);
    for (size_t i = 0; i < 4; ++i) CHECK_EQ(got[(size_t)(3 + 4 * i)] & RAD_KVQ_LAYOUT, picked[i]);
    CHECK_EQ(got[7], (int32_t)(RAD_KVQ_K4H | RAD_KVQ_ROTATED));
    CHECK(has(tiny::read_text(b.out), "layer 7 k4h_v4\n"));
    for (size_t l = 0; l < got.size(); ++l)
        if (l % 4 != 3) CHECK_EQ(got[l], -1);

    /* The measurements are in the recipe's comments, and every run's files in the work dir. */
    const std::string recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "# every layer fp8: mean KL 0.040000, top-1 0.9500, dNLL +0.0000 (the base)\n"));
    CHECK(has(recipe, "# no per-layer runs: k6_v4 rotated ("));
    CHECK(has(recipe, "#       3    4.000e-03    9.000e-03"));
    CHECK(has(recipe, "# chosen: 6.2500 bits an element"));
    for (const char* f : { "uniform_fp8.kvrecipe", "uniform_fp8.json", "uniform_fp8.log",
                           "uniform_k6_v4+rotated.json", "layer3_k6_v4.json" }) {
        struct stat st{};
        CHECK(::stat((b.out + ".work/" + f).c_str(), &st) == 0);
    }
    CHECK(has(log, "model.kvrecipe: 6.2500 bits an element"));

    /* --layouts narrows the set to those layouts' variants; fp8 is the base of what is left, and
     * the room for one fp8 layer goes to layer 3, whose fp8_v4 costs most. 4 uniform runs (fp8,
     * fp8_v4, each plain and rotated) and 4 per-layer ones. */
    std::remove(b.calls.c_str());
    log.clear();
    REQUIRE_EQ(b.run({ "--layers", "3,7,11,15", "--bits", "6.75", "--layouts", "fp8,fp8_v4", "--",
                       "--model", "/m/x.rad" }, &log), 0);
    CHECK_EQ(b.lines(b.calls), 8);
    Config d;
    Argv a2({ "--model", "/m/x.rad", "--kv-cache-dtype", "fp8", "--kv-recipe", b.out });
    REQUIRE_EQ(config_parse(a2.argc(), a2.argv.data(), &d), RAD_OK);
    CHECK(d.kv_recipe == std::vector<int32_t>({ -1, -1, -1, RAD_KVQ_NONE, -1, -1, -1, RAD_KVQ_V4,
                                                -1, -1, -1, RAD_KVQ_V4, -1, -1, -1, RAD_KVQ_V4 }));
}

/* The long-context requirement: a reference with a document under --min-context (32768 by
 * default) is refused before anything runs, naming the document and the bound; lowering the bound
 * is the explicit override. */
TEST(the_driver_refuses_a_short_reference_unless_the_bound_is_lowered) {
    Bench b({ 40000, 4096, 50000 });
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, "3 k6_v4 0.001\n"));
    const std::vector<std::string> args = { "--layers", "3,7", "--bits", "6", "--", "--model", "m" };
    std::string log;
    CHECK(b.run(args, &log) != 0);
    CHECK(has(log, "document 1 holds 4096 tokens, under --min-context 32768"));
    CHECK_EQ(b.lines(b.calls), 0);
    struct stat st{};
    CHECK(::stat(b.out.c_str(), &st) != 0);

    std::vector<std::string> low = { "--min-context", "4097" };
    low.insert(low.end(), args.begin(), args.end());
    log.clear();
    CHECK(b.run(low, &log) != 0);
    CHECK(has(log, "under --min-context 4097"));

    low[1] = "4096";
    log.clear();
    CHECK_EQ(b.run(low, &log), 0);
    CHECK(::stat(b.out.c_str(), &st) == 0);
}

/* What the driver refuses up front, and a run that fails stopping it with the run's log: no
 * recipe is written from a partial measurement. */
TEST(the_driver_refuses_bad_arguments_and_stops_at_a_failed_run) {
    Bench b;
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, "7 k6_v4 0.001\nfail 7 k6_v4\n"));
    auto refused = [&](const std::vector<std::string>& args, const std::string& why) {
        std::string log;
        const int rc = b.run(args, &log);
        CHECK(rc != 0);
        CHECK(has(log, why));
        if (rc == 0 || !has(log, why)) fprintf(stderr, "    wanted '%s' in:\n%s\n", why.c_str(), log.c_str());
    };
    refused({ "--layers", "3,7", "--bits", "6", "--", "--model", "m", "--kv-cache-dtype", "bf16" },
            "--kv-cache-dtype is set by rad-autoquant");
    refused({ "--layers", "3,7", "--bits", "6", "--", "--model", "m", "--kld-out=x" },
            "--kld-out is set by rad-autoquant");
    refused({ "--layers", "3,7", "--bits", "6" }, "no engine flags after --");
    refused({ "--layers", "3,3", "--bits", "6", "--", "--model", "m" }, "names layer 3 twice");
    refused({ "--layers", "3,x", "--bits", "6", "--", "--model", "m" }, "'x' is not a layer number");
    refused({ "--layers", "3,7", "--bits", "3", "--", "--model", "m" },
            "--bits 3 is under the narrowest layout's 3.50");
    refused({ "--layers", "3,7", "--bits", "5.5", "--layouts", "fp8_v4", "--", "--model", "m" },
            "under the narrowest layout's 6.25");
    refused({ "--layers", "3,7", "--bits", "6", "--layouts", "k4_v2", "--", "--model", "m" },
            "'k4_v2' is not a KV layout");
    refused({ "--layers", "3,7", "--bits", "6", "--layouts", "k4h_v4", "--variants", "fp8_v4",
              "--", "--model", "m" }, "--layouts and --variants leave no variant");
    refused({ "--layers", "3,7", "--bits", "6", "--variants", "k4_v2", "--", "--model", "m" },
            "--variants: 'k4_v2' is not a variant; they are bf16, i8h, i8h+vmean, fp8, fp8+rotated, "
            "fp8_v4, fp8_v4+vmean, fp8_v4+rotated, fp8_v4+rotated+vmean,");
    refused({ "--layers", "3,7", "--bits", "6", "--variants", "fp8+vmean", "--", "--model", "m" },
            "'fp8+vmean' is not a variant");
    refused({ "--layers", "3,7", "--bits", "6", "--variants", "k6_v4+vmean", "--", "--model", "m" },
            "k6_v4+vmean subtracts a V mean, and no --vmean FILE names one");
    refused({ "--layers", "3,7", "--bits", "6", "--vmean", b.dir / "none.txt", "--", "--model", "m" },
            "--vmean " + b.dir / "none.txt" + ": ");
    refused({ "--layers", "3,7", "--bits", "6", "--vmean", b.table, "--", "--model", "m" },
            "is not a V mean file");
    refused({ "--layers", "3,7", "--bits", "6", "--strict", "--", "--model", "m" },
            "go with --max-kl");
    refused({ "--layers", "3,7", "--bits", "6", "--min-top1", "0.9", "--", "--model", "m" },
            "go with --max-kl");
    refused({ "--layers", "3,7", "--max-kl", "0.05", "--min-top1", "2", "--", "--model", "m" },
            "--min-top1 '2' is not an agreement from 0 to 1");
    refused({ "--layers", "3,7", "--", "--model", "m" }, "are required");
    CHECK_EQ(b.lines(b.calls), 0);

    /* Every variant measures 0.04 but k6_v4's 0.041, so screening would keep only fp8 and
     * k4h_v2h; --keep-dominated runs layer 7 in k6_v4 all the same, and that run fails. */
    refused({ "--layers", "3,7", "--bits", "6", "--keep-dominated", "--", "--model", "m" },
            "run layer7_k6_v4 failed; its log, ");
    std::string log;
    b.run({ "--layers", "3,7", "--bits", "6", "--keep-dominated", "--", "--model", "m" }, &log);
    CHECK(has(log, "stand-in: refused"));
    struct stat st{};
    CHECK(::stat(b.out.c_str(), &st) != 0);
}

namespace {

/* The four-layer table of the first driver test, as the stand-in reads it and the solver takes it. */
const std::map<std::pair<int, int>, double> kCosts = {
    { { 3, RAD_KVQ_V4 }, 0.0040 },  { { 3, RAD_KVQ_K6V4 }, 0.0090 },
    { { 7, RAD_KVQ_V4 }, 0.0001 },  { { 7, RAD_KVQ_K6V4 }, 0.0002 },
    { { 11, RAD_KVQ_V4 }, 0.0003 }, { { 11, RAD_KVQ_K6V4 }, 0.0030 },
    { { 15, RAD_KVQ_V4 }, 0.0000 }, { { 15, RAD_KVQ_K6V4 }, 0.0004 },
    { { 3, RAD_KVQ_K4H }, 0.0500 },  { { 7, RAD_KVQ_K4H }, 0.0001 },
    { { 11, RAD_KVQ_K4H }, 0.0500 }, { { 15, RAD_KVQ_K4H }, 0.0500 },
    { { 3, RAD_KVQ_K4HV2H }, 0.0900 },  { { 7, RAD_KVQ_K4HV2H }, 0.0900 },
    { { 11, RAD_KVQ_K4HV2H }, 0.0900 }, { { 15, RAD_KVQ_K4HV2H }, 0.0900 },
};

std::string cost_table(const std::string& more = "") {
    std::string t;
    for (const auto& [k, v] : kCosts)
        t += std::to_string(k.first) + " " + kv_layout_name(k.second) + " " + std::to_string(v) + "\n";
    return t + more;
}

std::vector<std::vector<KvCandidate>> cost_layers() {
    std::vector<std::vector<KvCandidate>> layers;
    for (int l : { 3, 7, 11, 15 })
        layers.push_back({ fp8(), v4(kCosts.at({ l, RAD_KVQ_V4 })), k6v4(kCosts.at({ l, RAD_KVQ_K6V4 })),
                           k4h(kCosts.at({ l, RAD_KVQ_K4H })), k4hv2h(kCosts.at({ l, RAD_KVQ_K4HV2H })) });
    return layers;
}

/* The recipe the driver wrote, as the server loads it: the layout of each of the four layers. */
std::vector<int> written(const Bench& b) {
    Config c;
    Argv a({ "--model", "/m/x.rad", "--kv-cache-dtype", "fp8", "--kv-recipe", b.out });
    if (config_parse(a.argc(), a.argv.data(), &c) != RAD_OK || c.kv_recipe.size() != 16) return {};
    std::vector<int> q;
    for (int l : { 3, 7, 11, 15 }) q.push_back(c.kv_recipe[(size_t)l] & RAD_KVQ_LAYOUT);
    return q;
}

}  /* namespace */

/* --max-kl with costs that add up: the per-layer runs, every layout uniform, and the solver's one
 * answer measured, which lands within the limit and is what was written -- the fewest bits the
 * KL solver finds on the table's costs. */
TEST(the_driver_writes_the_fewest_bits_within_a_kl_limit) {
    Bench b;
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, cost_table()));
    std::string log;
    const int rc = b.run({ "--layers", "3,7,11,15", "--max-kl", "0.0412", "--", "--model",
                           "/m/x.rad" }, &log);
    if (rc != 0) fprintf(stderr, "    exited %d:\n%s\n", rc, log.c_str());
    REQUIRE_EQ(rc, 0);
    /* 10 uniform, the noise floor, 16 per-layer runs, 1 verified, and the knobs on the answer:
     * each alone and both, which the stand-in prices at nothing. */
    CHECK_EQ(b.lines(b.calls), 27 + 1 + 3);

    KvAssignment want;
    REQUIRE_EQ(kv_autoquant_solve_kl(cost_layers(), 0.0012, &want), RAD_OK);
    CHECK(written(b) == kvq_of(cost_layers(), want));
    const std::string recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "the fewest bits measured within a mean KL of 0.0412"));
    CHECK(has(recipe, "verify 1"));
    CHECK(has(recipe, "# every layer k6_v4: mean KL 0.052600, top-1 0.9248, +0.012600 above the "
                      "floor, dNLL +0.0000\n"));
    CHECK(has(recipe, "# solved 1: room 0.001200 above the base, predicted mean KL 0.040800, "
                      "measured 6.0625 bits, mean KL 0.040800, top-1 0.9484, +0.000800 above the "
                      "floor, dNLL +0.0000 (verify 1)\n"));
    CHECK(has(recipe, "(verify 1), the fewest bits measured within 0.0412"));
    CHECK(has(log, "within the limit"));
    CHECK(!has(log, "above the limit\n"));
}

/* Costs that do not add up: the first answer measures above the limit, the predicted limit
 * tightens, and the recipe written is one whose own run measured within the limit, with fewer
 * bits than every layer fp8. */
TEST(the_driver_tightens_until_a_measured_recipe_is_within_the_limit) {
    Bench b;
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, cost_table("pair 0.0004\n")));
    std::string log;
    const int rc = b.run({ "--layers", "3,7,11,15", "--max-kl", "0.0412", "--", "--model",
                           "/m/x.rad" }, &log);
    if (rc != 0) fprintf(stderr, "    exited %d:\n%s\n", rc, log.c_str());
    REQUIRE_EQ(rc, 0);
    CHECK(has(log, "above the limit"));
    const std::string recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "above the limit"));
    /* What was written measures within the limit under the stand-in's own sum. */
    const std::vector<int> q = written(b);
    REQUIRE_EQ(q.size(), (size_t)4);
    double kl = 0.04;
    int narrowed = 0;
    int64_t bits = 0;
    for (size_t i = 0; i < 4; ++i) {
        const int l = 3 + 4 * (int)i;
        if (q[i] != RAD_KVQ_NONE) { kl += kCosts.at({ l, q[i] }); ++narrowed; }
        bits += kv_layout_group_bits(q[i]);
    }
    if (narrowed > 1) kl += 0.0004 * (narrowed - 1);
    CHECK(kl <= 0.0412 + 1e-12);
    CHECK(bits < 4 * 512);
    if (kl > 0.0412 + 1e-12 || bits >= 4 * 512) fprintf(stderr, "    %s\n%s\n", log.c_str(), recipe.c_str());
}

/* --reuse: a second search with the same flags runs nothing, and writes the same recipe; a change
 * to the engine's flags runs everything again. */
TEST(the_driver_reuses_runs_whose_recipe_and_flags_are_unchanged) {
    Bench b;
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, cost_table("pair 0.0004\n")));
    const std::vector<std::string> args = { "--reuse", "--layers", "3,7,11,15", "--max-kl", "0.0412",
                                            "--", "--model", "/m/x.rad" };
    std::string log;
    REQUIRE_EQ(b.run(args, &log), 0);
    const int first = b.lines(b.calls);
    const std::string recipe = tiny::read_text(b.out);
    log.clear();
    REQUIRE_EQ(b.run(args, &log), 0);
    CHECK_EQ(b.lines(b.calls), first);
    CHECK(has(log, "uniform_fp8: reused from "));
    CHECK_EQ(tiny::read_text(b.out), recipe);
    /* A looser limit re-solves from the same measurements: only new recipes run. */
    std::vector<std::string> loose = args;
    loose[4] = "0.0450";
    log.clear();
    REQUIRE_EQ(b.run(loose, &log), 0);
    CHECK(b.lines(b.calls) - first <= 6);
    std::vector<std::string> other = args;
    other.push_back("--tp");
    other.push_back("2");
    log.clear();
    REQUIRE_EQ(b.run(other, &log), 0);
    CHECK(b.lines(b.calls) >= 2 * first);
}

TEST(the_driver_refuses_a_kl_limit_it_cannot_serve) {
    Bench b;
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, cost_table()));
    std::string log;
    CHECK(b.run({ "--layers", "3,7", "--max-kl", "0.03", "--", "--model", "m" }, &log) != 0);
    CHECK(has(log, "no recipe measured is within the limits: the most accurate, every layer fp8, "
                   "measures a mean KL of 0.040000, above --max-kl 0.03"));
    /* The uniform runs and the noise floor, and no per-layer run: below the base the costs have
     * nothing to give. */
    CHECK_EQ(b.lines(b.calls), 10 + 1);
    struct stat st{};
    CHECK(::stat(b.out.c_str(), &st) != 0);
    log.clear();
    CHECK(b.run({ "--layers", "3,7", "--max-kl", "0.05", "--bits", "6", "--", "--model", "m" }, &log) != 0);
    CHECK(has(log, "--bits and --max-kl ask opposite questions"));
    log.clear();
    CHECK(b.run({ "--layers", "3,7", "--max-kl", "-1", "--", "--model", "m" }, &log) != 0);
    CHECK(has(log, "is not a positive mean KL"));
    log.clear();
    CHECK(b.run({ "--layers", "3,7", "--max-kl", "0.05", "--verify", "0", "--", "--model", "m" }, &log) != 0);
    CHECK(has(log, "--verify '0'"));
}

/* ================================================================== every variant a candidate */

namespace {

/* What the stand-in reports for the four layers of `written` under kCosts: 0.04 plus the costs
 * (fp8 free), and the bits they take. */
std::pair<double, int64_t> stand_in(const std::vector<int>& q, double pair = 0) {
    double s = 0;
    int64_t bits = 0;
    int narrowed = 0;
    for (size_t i = 0; i < q.size(); ++i) {
        const int l = 3 + 4 * (int)i;
        if (q[i] != RAD_KVQ_NONE) { s += kCosts.at({ l, q[i] }); ++narrowed; }
        bits += kv_layout_group_bits(q[i]);
    }
    if (narrowed > 1) s += pair * (narrowed - 1);
    return { s, bits };
}

}  /* namespace */

/* The base is the variant whose uniform run measures least, not fp8: here i8h, with fp8 costing
 * 0.002 a layer. Every cost is measured above i8h, so a limit under every-layer-fp8's 0.048 is
 * reachable: layer 7 alone in k6_v4, the rest i8h, measures 0.0402 for fewer bits than all-i8h,
 * and is written without a run of its own, as the per-layer run already measured that recipe. */
TEST(the_driver_takes_the_most_accurate_variant_as_its_base_and_meets_a_limit_under_fp8s) {
    Bench b;
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, "3 fp8 0.002\n7 fp8 0.002\n11 fp8 0.002\n15 fp8 0.002\n"
                                      "3 k6_v4 0.003\n7 k6_v4 0.0002\n11 k6_v4 0.003\n15 k6_v4 0.003\n"));
    std::string log;
    const int rc = b.run({ "--layers", "3,7,11,15", "--max-kl", "0.0405", "--layouts",
                           "i8h,fp8,k6_v4", "--", "--model", "/m/x.rad" }, &log);
    if (rc != 0) fprintf(stderr, "    exited %d:\n%s\n", rc, log.c_str());
    REQUIRE_EQ(rc, 0);
    CHECK(has(log, "the base, the most accurate: every layer i8h\n"));
    /* 5 uniform (i8h, fp8, fp8 rotated, k6_v4, k6_v4 rotated); the rotated ones tie with their
     * plain layouts and are dropped; the noise floor; 4 layers x fp8 and k6_v4, the rest i8h;
     * the three knob runs. */
    CHECK_EQ(b.lines(b.calls), 5 + 1 + 8 + 3);
    const std::string recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "# every layer i8h: mean KL 0.040000, top-1 0.9500, +0.000000 above the "
                      "floor, dNLL +0.0000 (the base)\n"));
    CHECK(has(recipe, "# every layer fp8: mean KL 0.048000"));
    CHECK(has(recipe, "\nlayer 3 i8h\nlayer 7 k6_v4\nlayer 11 i8h\nlayer 15 i8h\n"));
    CHECK(has(recipe, "(layer 7 k6_v4), the fewest bits measured within 0.0405"));
    CHECK(has(tiny::read_text(b.calls), "/layer7_fp8.kvrecipe"));
    CHECK(has(tiny::read_text(b.out + ".work/layer7_fp8.kvrecipe"),
              "layer 3 i8h\nlayer 7 fp8\nlayer 11 i8h\nlayer 15 i8h\n"));
    Config c;
    Argv a({ "--model", "/m/x.rad", "--kv-cache-dtype", "fp8", "--kv-recipe", b.out });
    REQUIRE_EQ(config_parse(a.argc(), a.argv.data(), &c), RAD_OK);
    CHECK_EQ(c.kv_recipe[3], (int32_t)RAD_KVQ_I8H | (int32_t)RAD_KVQ_ROTATED);
    CHECK_EQ(c.kv_recipe[7], (int32_t)RAD_KVQ_K6V4);
}

/* bf16 joins from the layout table as the most accurate variant, so it is the base and a limit
 * under every 8-bit layout's is met: here i8h costs 0.001 a layer (0.044 uniform) and fp8 more,
 * and --max-kl 0.0402 is under both. Layer 7 alone in fp8 (0.04015, the rest bf16) is the fewest
 * bits within it, written without a run of its own; 4 uniform runs (bf16, i8h, fp8, fp8 rotated,
 * the last dropped as fp8's tie), the noise floor, 4 layers x i8h and fp8, and the three knob
 * runs. */
TEST(the_driver_takes_bf16_as_its_base_and_meets_a_limit_under_i8hs) {
    Bench b;
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, "3 i8h 0.001\n7 i8h 0.001\n11 i8h 0.001\n15 i8h 0.001\n"
                                      "3 fp8 0.002\n7 fp8 0.00015\n11 fp8 0.002\n15 fp8 0.002\n"));
    std::string log;
    const int rc = b.run({ "--layers", "3,7,11,15", "--max-kl", "0.0402", "--layouts",
                           "bf16,i8h,fp8", "--", "--model", "/m/x.rad" }, &log);
    if (rc != 0) fprintf(stderr, "    exited %d:\n%s\n", rc, log.c_str());
    REQUIRE_EQ(rc, 0);
    CHECK(has(log, "the base, the most accurate: every layer bf16\n"));
    CHECK_EQ(b.lines(b.calls), 4 + 1 + 8 + 3);
    const std::string recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "# every layer bf16: mean KL 0.040000, top-1 0.9500, +0.000000 above the "
                      "floor, dNLL +0.0000 (the base)\n"));
    CHECK(has(recipe, "# every layer i8h: mean KL 0.044000"));
    CHECK(has(recipe, "\nlayer 3 bf16\nlayer 7 fp8\nlayer 11 bf16\nlayer 15 bf16\n"));
    CHECK(has(recipe, "(layer 7 fp8), the fewest bits measured within 0.0402"));
    Config c;
    Argv a({ "--model", "/m/x.rad", "--kv-cache-dtype", "fp8", "--kv-recipe", b.out });
    REQUIRE_EQ(config_parse(a.argc(), a.argv.data(), &c), RAD_OK);
    CHECK_EQ(c.kv_recipe[3], (int32_t)RAD_KVQ_BF16);
    CHECK_EQ(c.kv_recipe[7], (int32_t)RAD_KVQ_NONE);
}

/* Screening: a variant beaten on its uniform run by one of no more bits gets no per-layer runs
 * (here bf16, i8h and the rotated variants, which tie with fp8 and their plain layouts); with
 * --keep-dominated every variant has them, 10 + 4 x 9 runs. The predicted KL is the same, and a
 * tie keeps the earlier variant: never a rotated one, which costs the same bits as its plain
 * layout, but i8h where the budget has 4 bits to spare for a layer the stand-in prices as fp8
 * (the solver keeps ties wide). */
TEST(the_driver_screens_out_dominated_variants_unless_asked_to_keep_them) {
    Bench b;
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, cost_table()));
    const std::vector<std::string> args = { "--layers", "3,7,11,15", "--bits", "6.5", "--",
                                            "--model", "/m/x.rad" };
    std::string log;
    REQUIRE_EQ(b.run(args, &log), 0);
    CHECK_EQ(b.lines(b.calls), 10 + 16);
    const std::vector<int> screened = written(b);
    const std::string recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "# no per-layer runs: i8h (0.040000 at 8.06 bits; fp8: 0.040000 at 8.00)"));
    CHECK(!has(tiny::read_text(b.calls), "/layer3_k6_v4+rotated.kvrecipe"));

    std::vector<std::string> keep = args;
    keep.insert(keep.begin(), "--keep-dominated");
    std::remove(b.calls.c_str());
    log.clear();
    REQUIRE_EQ(b.run(keep, &log), 0);
    CHECK_EQ(b.lines(b.calls), 10 + 4 * 9);
    CHECK(has(tiny::read_text(b.calls), "/layer3_k6_v4+rotated.kvrecipe"));
    CHECK(!has(log, "screening drops i8h"));
    const std::string kept = tiny::read_text(b.out);
    CHECK(has(recipe, "predicted mean KL 0.040400 "));
    CHECK(has(kept, "predicted mean KL 0.040400 "));
    CHECK(!has(kept, " rotated\n"));
    const std::vector<int> all = written(b);
    REQUIRE_EQ(all.size(), screened.size());
    for (size_t i = 0; i < all.size(); ++i)
        CHECK(all[i] == screened[i] || (all[i] == RAD_KVQ_I8H && screened[i] == RAD_KVQ_NONE));
}

/* A uniform recipe that measures far better than its layers' costs sum to (as rotation and the V
 * mean do together on the card) is the answer when no solved recipe has as few bits: the solver's
 * first answer has more bits than uniform k6_v4, already within, so it is not even measured. And
 * fp8_v4, which uniform k6_v4 now beats for fewer bits, gets no per-layer runs. */
TEST(the_driver_keeps_a_uniform_recipe_that_measures_cheaper_than_any_solved_one) {
    Bench b;
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, cost_table("all k6_v4 0.0005\n")));
    std::string log;
    const int rc = b.run({ "--layers", "3,7,11,15", "--max-kl", "0.0412", "--", "--model",
                           "/m/x.rad" }, &log);
    if (rc != 0) fprintf(stderr, "    exited %d:\n%s\n", rc, log.c_str());
    REQUIRE_EQ(rc, 0);
    CHECK(written(b) == std::vector<int>(4, RAD_KVQ_K6V4));
    CHECK(has(log, "screening drops fp8_v4 (0.044400 at 6.25 bits; k6_v4: 0.040500 at 5.50)"));
    /* 10 uniform, the noise floor, 4 layers x k6_v4, k4h_v4, k4h_v2h; no verification; the
     * three knob runs. */
    CHECK_EQ(b.lines(b.calls), 10 + 1 + 12 + 3);
    CHECK(has(log, "the next solution has no fewer bits than every layer k6_v4"));
    CHECK(has(tiny::read_text(b.out), "(every layer k6_v4), the fewest bits measured within 0.0412"));
}

/* --min-top1: the stand-in's agreement falls by twice the KL it adds, so a floor of 0.9485 allows
 * 0.00075 above the base where --max-kl 0.0412 allows 0.0012. The KL limit's own answer (0.0008)
 * measures under the floor; the top-1 check tightens the room and the recipe written keeps it.
 * A floor over the base's agreement is met by nothing. */
TEST(the_driver_keeps_a_top1_floor_beside_the_kl_limit) {
    Bench b;
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, cost_table()));
    std::string log;
    int rc = b.run({ "--layers", "3,7,11,15", "--max-kl", "0.0412", "--", "--model", "/m/x.rad" },
                   &log);
    REQUIRE_EQ(rc, 0);
    const auto loose = stand_in(written(b));
    CHECK_NEAR(loose.first, 0.0008, 1e-12);

    log.clear();
    rc = b.run({ "--layers", "3,7,11,15", "--max-kl", "0.0412", "--min-top1", "0.9485", "--",
                 "--model", "/m/x.rad" }, &log);
    if (rc != 0) fprintf(stderr, "    exited %d:\n%s\n", rc, log.c_str());
    REQUIRE_EQ(rc, 0);
    const auto floor = stand_in(written(b));
    CHECK(0.95 - 2 * floor.first >= 0.9485 - 1e-9);
    CHECK(floor.second > loose.second);
    CHECK(floor.second < 4 * 512);
    CHECK(has(log, "above the limit"));
    const std::string recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "within a mean KL of 0.0412 and a top-1 agreement of 0.9485\n"));
    CHECK(has(recipe, "the fewest bits measured within 0.0412 and top-1 0.9485\n"));

    log.clear();
    CHECK(b.run({ "--layers", "3,7,11,15", "--max-kl", "0.0412", "--min-top1", "0.96", "--",
                  "--model", "/m/x.rad" }, &log) != 0);
    CHECK(has(log, "no recipe measured is within the limits"));
    CHECK(has(log, "or under --min-top1 0.96"));
}

/* --strict: the limits hold for the 95 % bound of each run's mean, from its .rows with each
 * document a stratum (the stand-in's rows are 999 positions a document, the mean KL in stretches
 * of half and one and a half times it). What is written meets the limit at mean + half-width, the
 * interval as the driver states it, and takes no fewer bits than the means alone would. Without
 * .rows, --strict has nothing to check and refuses. */
TEST(the_driver_under_strict_checks_the_limit_at_the_upper_bound_from_the_rows) {
    Bench b({ 40000, 40000 }, 999);
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, cost_table()));
    const std::vector<std::string> args = { "--layers", "3,7,11,15", "--max-kl", "0.046", "--",
                                            "--model", "/m/x.rad" };
    std::string log;
    REQUIRE_EQ(b.run(args, &log), 0);
    const auto loose = stand_in(written(b));

    std::vector<std::string> strict = args;
    strict.insert(strict.begin(), "--strict");
    log.clear();
    const int rc = b.run(strict, &log);
    if (rc != 0) fprintf(stderr, "    exited %d:\n%s\n", rc, log.c_str());
    REQUIRE_EQ(rc, 0);
    const auto got = stand_in(written(b));
    /* The chosen run's rows, as the stand-in writes them, and their interval. */
    char printed[32];
    std::snprintf(printed, sizeof printed, "%.9f", 0.04 + got.first);
    const float m = (float)std::stod(printed);
    std::vector<double> kl;
    for (int i = 0; i < 1998; ++i) kl.push_back((double)(m * ((i / 37) % 2 ? 1.5f : 0.5f)));
    KvInterval ci;
    REQUIRE_EQ(kv_autoquant_interval(kl, { 999, 999 }, &ci), RAD_OK);
    CHECK(ci.half > 1e-4);
    CHECK(0.04 + got.first + ci.half <= 0.046 + 1e-6);
    CHECK(got.second >= loose.second);
    const std::string recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "within a mean KL of 0.046 (95% bounds)\n"));
    CHECK(has(recipe, "# every layer fp8: mean KL 0.040000 +- 0.00"));
    CHECK(has(recipe, " at the 95% bounds\n"));
    char want[64];
    std::snprintf(want, sizeof want, " +- %.6f", ci.half);
    CHECK(has(recipe.substr(recipe.find("# chosen: ")), want));

    Bench plain;
    REQUIRE(plain.dir.ok());
    REQUIRE(tiny::write_text(plain.table, cost_table()));
    log.clear();
    CHECK(plain.run(strict, &log) != 0);
    CHECK(has(log, "run uniform_bf16 left no .rows of the reference's positions, which --strict needs"));
}

/* ================================================================== the noise floor and the knobs */

namespace {

/* The stand-in's model of the card: the noise run (the base with --max-num-batched-tokens 64)
 * measures 0.024 above the base, as recomputing in another order does on Flash-Next, while a
 * recipe measures the base's 0.04 plus its layers' costs, however far under the floor that is. */
const std::string kFloor = "flag --max-num-batched-tokens 0.024\n";

/* The 95 % half-width the driver finds in a stand-in run's .rows over two documents of 999
 * scored positions: of the KL column for a run of mean KL `kl`, or of the NLL change for a run of
 * NLL change `dnll` -- the values as the stand-in prints them (%.9f) and the f32 rows hold them. */
double rows_half(double kl, double dnll = NAN) {
    char printed[32];
    std::snprintf(printed, sizeof printed, "%.9f", std::isnan(dnll) ? kl : dnll);
    const double v = std::stod(printed);
    std::vector<double> x;
    for (int i = 0; i < 1998; ++i) {
        const double f = (i / 37) % 2 ? 1.5 : 0.5;
        x.push_back(std::isnan(dnll) ? (double)((float)v * (float)f)
                                     : (double)(float)(1 + v * f) - 1.0);
    }
    KvInterval ci;
    if (kv_autoquant_interval(x, { 999, 999 }, &ci) != RAD_OK) return NAN;
    return ci.half;
}

}  /* namespace */

/* The noise floor: the base (fp8 here: the stand-in prices it and bf16 alike, and screening keeps
 * the fewer bits) once more with --max-num-batched-tokens 64, measuring the stand-in's floor of
 * 0.064 where the base measures 0.04. The per-layer costs are measured above the base, never
 * above the floor, so they are the table's; --max-dkl 0.0012 is a limit of 0.0652, which leaves
 * 0.0252 above the base, and the answer is the solver's on the table within that room, measured
 * under the floor. An absolute --max-kl under the floor (0.05) is an ordinary search: the room is
 * 0.01 above the base, and the answer measures under the floor too. */
TEST(the_driver_measures_a_noise_floor_and_meets_a_kl_limit_above_it) {
    Bench b;
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, cost_table(kFloor)));
    std::string log;
    int rc = b.run({ "--layers", "3,7,11,15", "--max-dkl", "0.0012", "--no-knobs", "--",
                     "--model", "/m/x.rad" }, &log);
    if (rc != 0) fprintf(stderr, "    exited %d:\n%s\n", rc, log.c_str());
    REQUIRE_EQ(rc, 0);
    CHECK(has(log, "the base, the most accurate: every layer fp8\n"));
    CHECK(has(log, "the noise floor: mean KL 0.064000; every layer fp8 is -0.024000 above it\n"));
    CHECK(has(log, "every layer k6_v4: -0.011400 above the floor\n"));
    CHECK(has(log, "not a bound for KV-only changes; costs are measured above the base)\n"));
    /* 10 uniform, the noise run, 4 layers x fp8_v4, k6_v4, k4h_v4 and k4h_v2h (bf16 and i8h are
     * screened out by fp8, the rotated ones by their plain layouts), 1 solved. */
    CHECK_EQ(b.lines(b.calls), 10 + 1 + 16 + 1);
    const std::string calls = tiny::read_text(b.calls);
    CHECK(has(calls, "--model /m/x.rad --max-num-batched-tokens 64 --kv-cache-dtype fp8 --kv-recipe " +
                         b.out + ".work/noise_fp8_"));

    KvAssignment want;
    REQUIRE_EQ(kv_autoquant_solve_kl(cost_layers(), 0.0252, &want), RAD_OK);
    CHECK(written(b) == kvq_of(cost_layers(), want));
    CHECK(0.04 + stand_in(written(b)).first < 0.064);
    const std::string recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "the fewest bits measured within a mean KL of 0.0012 above the noise floor\n"));
    CHECK(has(recipe, "# noise floor, every layer fp8 with --max-num-batched-tokens 64: mean KL "
                      "0.064000, top-1 0.9020, dNLL +0.0000\n"));
    CHECK(has(recipe, "# each layer alone in a variant, the rest the base: mean KL above the base\n"));
    CHECK(has(recipe, "#       3    4.000e-03    9.000e-03    5.000e-02"));
    CHECK(has(recipe, "# solved 1: room 0.025200 above the base, predicted mean KL 0.052500, "
                      "measured 5.2500 bits, mean KL 0.052500, top-1 0.9250, -0.011500 above the "
                      "floor, dNLL +0.0000 (verify 1)\n"));
    CHECK(has(recipe, "(verify 1), the fewest bits measured within floor + 0.0012\n"));

    std::remove(b.calls.c_str());
    log.clear();
    rc = b.run({ "--layers", "3,7,11,15", "--max-kl", "0.05", "--no-knobs", "--", "--model",
                 "/m/x.rad" }, &log);
    REQUIRE_EQ(rc, 0);
    CHECK(!has(log, "is under the noise floor"));
    CHECK_EQ(b.lines(b.calls), 10 + 1 + 16 + 1);
    REQUIRE_EQ(kv_autoquant_solve_kl(cost_layers(), 0.01, &want), RAD_OK);
    CHECK(written(b) == kvq_of(cost_layers(), want));
    CHECK(0.04 + stand_in(written(b)).first <= 0.05 + 1e-12);
    CHECK(has(tiny::read_text(b.out), "# solved 1: room 0.010000 above the base"));
}

/* The card's case (Flash-Next --tp 2: layers 3-23 bf16 + 27-47 i8h measure 0.0061 where the
 * noise floor is 0.0243 and every layer i8h 0.0236): the early layers carry the KL, so bf16 early
 * and the narrow layout late measures far under the floor. Here bf16 is the base (0.04), the noise
 * run 0.064, every layer i8h 0.0635, just under the floor as on the card, and i8h costs 0.011,
 * 0.008, 0.003 and 0.0015 in layers 3, 7, 11 and 15. An absolute --max-kl of 0.046, under the
 * floor, leaves 0.006 above the base: layers 11 and 15 in i8h (0.0045), the rest bf16, found from
 * costs counted from the base, measured at 0.0445 -- 0.0195 under the floor -- and accepted. */
TEST(the_driver_finds_a_recipe_under_the_noise_floor_within_an_absolute_kl_limit_below_it) {
    Bench b;
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, "3 i8h 0.011\n7 i8h 0.008\n11 i8h 0.003\n15 i8h 0.0015\n" + kFloor));
    std::string log;
    const int rc = b.run({ "--layers", "3,7,11,15", "--max-kl", "0.046", "--layouts", "bf16,i8h",
                           "--no-knobs", "--", "--model", "/m/x.rad" }, &log);
    if (rc != 0) fprintf(stderr, "    exited %d:\n%s\n", rc, log.c_str());
    REQUIRE_EQ(rc, 0);
    CHECK(has(log, "the base, the most accurate: every layer bf16\n"));
    CHECK(has(log, "the noise floor: mean KL 0.064000; every layer bf16 is -0.024000 above it\n"));
    CHECK(has(log, "every layer i8h: -0.000500 above the floor\n"));
    CHECK(!has(log, "is under the noise floor"));
    /* 2 uniform, the noise run, 4 layers x i8h, 1 solved. */
    CHECK_EQ(b.lines(b.calls), 2 + 1 + 4 + 1);
    CHECK(written(b) == std::vector<int>({ RAD_KVQ_BF16, RAD_KVQ_BF16, RAD_KVQ_I8H, RAD_KVQ_I8H }));
    const std::string recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "# each layer alone in a variant, the rest the base: mean KL above the base\n"));
    CHECK(has(recipe, "#      15    1.500e-03"));
    CHECK(has(recipe, "# solved 1: room 0.006000 above the base, predicted mean KL 0.044500, "
                      "measured 12.0312 bits, mean KL 0.044500, top-1 0.9410, -0.019500 above the "
                      "floor, dNLL +0.0000 (verify 1)\n"));
    CHECK(has(recipe, "(verify 1), the fewest bits measured within 0.046\n"));
}

/* --max-dkl under --strict: the run's upper 95 % bound less the floor's lower bound is at most the
 * limit. The noise run here measures 0.0004 above the base (0.0404; at the card's 0.024 every
 * recipe of the table would fit either way). The intervals are about 3 % of the mean on the
 * stand-in's rows, so the floor's lower bound is near 0.0392 and a recipe's upper bound must stay
 * under 0.0442 where the mean rule lets its mean reach 0.0454: what is written meets that rule,
 * takes more bits than the mean rule's answer, and the mean rule's answer would not have met it.
 * The recipes' KL is the base's 0.04 plus their costs, not the floor's. */
TEST(the_driver_under_strict_checks_kl_above_the_floor_at_the_upper_less_the_floors_lower_bound) {
    Bench b({ 40000, 40000 }, 999);
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, cost_table("flag --max-num-batched-tokens 0.0004\n")));
    const std::vector<std::string> args = { "--layers", "3,7,11,15", "--max-dkl", "0.005",
                                            "--no-knobs", "--", "--model", "/m/x.rad" };
    std::string log;
    REQUIRE_EQ(b.run(args, &log), 0);
    const auto mean = stand_in(written(b));
    std::vector<std::string> strict = args;
    strict.insert(strict.begin(), "--strict");
    log.clear();
    const int rc = b.run(strict, &log);
    if (rc != 0) fprintf(stderr, "    exited %d:\n%s\n", rc, log.c_str());
    REQUIRE_EQ(rc, 0);
    const auto got = stand_in(written(b));
    const double floor = 0.0404, floor_lo = floor - rows_half(floor);
    auto upper = [](double kl) { return kl + rows_half(kl); };
    CHECK(rows_half(floor) > 1e-3);
    CHECK(upper(0.04 + got.first) - floor_lo <= 0.005 + 1e-6);
    CHECK(upper(0.04 + mean.first) - floor_lo > 0.005);
    CHECK(0.04 + mean.first <= floor + 0.005 + 1e-9);
    CHECK(got.second > mean.second);
    const std::string recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "within a mean KL of 0.005 above the noise floor (95% bounds)\n"));
    CHECK(has(recipe, "# noise floor, every layer fp8 with --max-num-batched-tokens 64: mean KL "
                      "0.040400 +- 0.00"));
    CHECK(has(recipe, "the fewest bits measured within floor + 0.005 at the 95% bounds\n"));
}

namespace {

/* Layers 7 and 15 change the NLL when narrowed (stand-in table lines with a fourth field):
 * layer 7 by 0.004 in every narrower layout, which its tiny KL costs never show, and layer 15 by
 * 0.00295, within a limit of 0.003 at the mean but not at its upper bound. */
const std::string kNll = "7 fp8_v4 0.0001 0.004\n7 k6_v4 0.0002 0.004\n7 k4h_v4 0.0001 0.004\n"
                         "7 k4h_v2h 0.09 0.004\n15 fp8_v4 0.0000 0.00295\n15 k6_v4 0.0004 0.00295\n"
                         "15 k4h_v4 0.05 0.00295\n15 k4h_v2h 0.09 0.00295\n";

/* The NLL change the stand-in gives the four layers of `written` under kNll. */
double stand_in_dnll(const std::vector<int>& q) {
    double d = 0;
    if (q.size() == 4 && q[1] != RAD_KVQ_NONE && q[1] != RAD_KVQ_I8H && q[1] != RAD_KVQ_BF16) d += 0.004;
    if (q.size() == 4 && q[3] != RAD_KVQ_NONE && q[3] != RAD_KVQ_I8H && q[3] != RAD_KVQ_BF16) d += 0.00295;
    return d;
}

}  /* namespace */

/* --max-dnll: the KL limit alone narrows layer 7, whose NLL change (0.004) the limit of 0.003
 * refuses; with it the recipe keeps layer 7 at fp8 and measures within both. The NLL change is a
 * check on measured recipes, as top-1 is, not a cost the solver sums, so the answer here is a
 * single-layer run: layer 15 in k6_v4, NLL change 0.00295. Under --strict that is over the limit
 * at its upper bound (+ about 3 %), so layer 15 stays fp8 and another layer narrows instead. The
 * NLL change is stated for every run, with its interval. An NLL limit every recipe is over is met
 * by nothing, and the refusal names it. */
TEST(the_driver_keeps_an_nll_limit_at_the_mean_and_under_strict_at_the_upper_bound) {
    Bench b({ 40000, 40000 }, 999);
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, cost_table(kNll)));
    std::string log;
    REQUIRE_EQ(b.run({ "--layers", "3,7,11,15", "--max-kl", "0.0412", "--no-knobs", "--",
                       "--model", "/m/x.rad" }, &log), 0);
    const std::vector<int> kl_only = written(b);
    REQUIRE_EQ(kl_only.size(), (size_t)4);
    CHECK(kl_only[1] != RAD_KVQ_NONE);

    const std::vector<std::string> args = { "--layers", "3,7,11,15", "--max-kl", "0.046",
                                            "--max-dnll", "0.003", "--no-knobs", "--", "--model",
                                            "/m/x.rad" };
    log.clear();
    int rc = b.run(args, &log);
    if (rc != 0) fprintf(stderr, "    exited %d:\n%s\n", rc, log.c_str());
    REQUIRE_EQ(rc, 0);
    const std::vector<int> mean = written(b);
    REQUIRE_EQ(mean.size(), (size_t)4);
    CHECK_EQ(mean[1], RAD_KVQ_NONE);
    CHECK(mean[3] != RAD_KVQ_NONE);
    CHECK(stand_in_dnll(mean) <= 0.003);
    CHECK(0.04 + stand_in(mean).first <= 0.046 + 1e-12);
    CHECK(stand_in(mean).second < 4 * 512);
    std::string recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "within a mean KL of 0.046 and a mean NLL change of 0.003 nats a token\n"));
    CHECK(has(recipe, "the fewest bits measured within 0.046 and dNLL 0.003\n"));
    CHECK(has(recipe, ", dNLL +0.0000 +- 0.0000"));
    char want[160];
    std::snprintf(want, sizeof want, "rad-autoquant: layer 7 fp8_v4: 7.5625 bits, mean KL 0.040100 "
                  "+- %.6f, top-1 0.9498 +- 0.0132, +0.000100 above the floor, dNLL +0.0040 +- "
                  "%.4f\n", rows_half(0.0401), rows_half(0, 0.004));
    CHECK(has(log, want));
    if (!has(log, want)) fprintf(stderr, "    wanted '%s'\n", want);

    std::vector<std::string> strict = args;
    strict.insert(strict.begin(), "--strict");
    log.clear();
    rc = b.run(strict, &log);
    if (rc != 0) fprintf(stderr, "    exited %d:\n%s\n", rc, log.c_str());
    REQUIRE_EQ(rc, 0);
    const std::vector<int> got = written(b);
    REQUIRE_EQ(got.size(), (size_t)4);
    CHECK(0.00295 + rows_half(0, 0.00295) > 0.003);
    CHECK_EQ(got[1], RAD_KVQ_NONE);
    CHECK_EQ(got[3], RAD_KVQ_NONE);
    CHECK(stand_in(got).second >= stand_in(mean).second);
    CHECK(stand_in(got).second < 4 * 512);
    recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "the fewest bits measured within 0.046 and dNLL 0.003 at the 95% bounds\n"));

    REQUIRE(tiny::write_text(b.table, "3 bf16 0 0.01\n3 i8h 0 0.01\n3 fp8 0 0.01\n3 fp8_v4 0.004 0.01\n"
                                      "3 k6_v4 0.009 0.01\n3 k4h_v4 0.05 0.01\n3 k4h_v2h 0.09 0.01\n"));
    log.clear();
    CHECK(b.run({ "--layers", "3,7", "--max-dnll", "0.005", "--no-knobs", "--", "--model",
                  "/m/x.rad" }, &log) != 0);
    CHECK(has(log, "no recipe measured is within the limits: the most accurate, every layer fp8, "
                   "measures a mean KL of 0.040000 and dNLL +0.0100, above --max-dnll 0.005"));
}

/* The knobs on the answer of the first --max-kl test (6.0625 bits, 0.0408): int8 block keys add
 * 0.0001 and stay within 0.0412, the f16 state adds 0.002 and does not, so both together are not
 * run; the block keys are kept, for fewer bytes a token (the stand-in's attention pages, 776 bytes
 * a 4-token page, beside block keys of 3.00 KiB, or 1.59 KiB in int8), and the serve flags say so.
 * A knob that breaks only the NLL limit is not kept either, nor one whose run the engine refuses,
 * which stops nothing. Both within: both kept, the state at half. --knobs restricts the search,
 * and a knob it leaves out may be fixed in the engine's flags; --no-knobs runs none. */
TEST(the_driver_keeps_a_knob_only_where_the_recipe_stays_within_every_limit) {
    Bench b;
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, cost_table("flag i8 0.0001\nflag f16 0.002\n")));
    const std::vector<std::string> args = { "--layers", "3,7,11,15", "--max-kl", "0.0412", "--",
                                            "--model", "/m/x.rad" };
    std::string log;
    int rc = b.run(args, &log);
    if (rc != 0) fprintf(stderr, "    exited %d:\n%s\n", rc, log.c_str());
    REQUIRE_EQ(rc, 0);
    /* The first --max-kl test's 27 runs, the noise floor, and one run a knob. */
    CHECK_EQ(b.lines(b.calls), 27 + 1 + 2);
    std::string recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "# its cache: 0.94 KiB a token, state 54.00 MiB a sequence (rank 0's, from its log)\n"));
    CHECK(has(recipe, "# with --qsa-bkey-dtype i8: 6.0625 bits, mean KL 0.040900, top-1 0.9482, "
                      "+0.000900 above the floor, dNLL +0.0000, 0.59 KiB a token, state 54.00 MiB "
                      "a sequence\n"));
    CHECK(has(recipe, "# with --state-cache-dtype f16: 6.0625 bits, mean KL 0.042800"));
    CHECK(has(recipe, "state 27.00 MiB a sequence, above the limits\n"));
    CHECK(has(recipe, "# knobs: --qsa-bkey-dtype i8, the fewest bytes a token within every limit"));
    CHECK(has(recipe, "# serve with: --kv-cache-dtype fp8 --kv-recipe " + b.out +
                      " --qsa-bkey-dtype i8\n"));
    CHECK(has(log, "mean KL 0.040900"));
    CHECK(has(tiny::read_text(b.calls), "--model /m/x.rad --qsa-bkey-dtype i8 --kv-cache-dtype fp8"));

    /* The engine refuses int8 block keys; the f16 state is free in KL but moves the NLL by 0.01,
     * over --max-dnll 0.005. Neither is kept, and the recipe is still written. */
    REQUIRE(tiny::write_text(b.table, cost_table("refuse i8\nflag f16 0 0.01\n")));
    log.clear();
    rc = b.run({ "--layers", "3,7,11,15", "--max-kl", "0.0412", "--max-dnll", "0.005", "--",
                 "--model", "/m/x.rad" }, &log);
    if (rc != 0) fprintf(stderr, "    exited %d:\n%s\n", rc, log.c_str());
    REQUIRE_EQ(rc, 0);
    CHECK(has(log, "stand-in: refused the flag"));
    CHECK(has(log, "with --qsa-bkey-dtype i8 the run failed: not kept"));
    recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "# with --qsa-bkey-dtype i8: the run failed, not kept\n"));
    CHECK(has(recipe, "dNLL +0.0100, 0.94 KiB a token, state 27.00 MiB a sequence, above the limits\n"));
    CHECK(has(recipe, "# knobs: none kept"));
    CHECK(has(recipe, "# serve with: --kv-cache-dtype fp8 --kv-recipe " + b.out + "\n"));

    /* Both free: each alone, then both, kept. */
    REQUIRE(tiny::write_text(b.table, cost_table()));
    std::remove(b.calls.c_str());
    log.clear();
    REQUIRE_EQ(b.run(args, &log), 0);
    CHECK_EQ(b.lines(b.calls), 27 + 1 + 3);
    recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "# with --qsa-bkey-dtype i8 --state-cache-dtype f16: "));
    CHECK(has(recipe, "# serve with: --kv-cache-dtype fp8 --kv-recipe " + b.out +
                      " --qsa-bkey-dtype i8 --state-cache-dtype f16\n"));
    CHECK(has(recipe, "0.59 KiB a token, state 27.00 MiB a sequence\n"));

    /* --knobs state_f16: one knob run; the block keys fixed at i8 in the engine's flags are run
     * that way throughout. Searched, the flag is refused there; --no-knobs runs no knob. */
    std::remove(b.calls.c_str());
    log.clear();
    REQUIRE_EQ(b.run({ "--layers", "3,7,11,15", "--max-kl", "0.0412", "--knobs", "state_f16", "--",
                       "--model", "/m/x.rad", "--qsa-bkey-dtype", "i8" }, &log), 0);
    CHECK_EQ(b.lines(b.calls), 27 + 1 + 1);
    CHECK(has(tiny::read_text(b.out), "--kv-recipe " + b.out + " --state-cache-dtype f16\n"));
    std::remove(b.calls.c_str());
    log.clear();
    REQUIRE_EQ(b.run({ "--layers", "3,7,11,15", "--max-kl", "0.0412", "--no-knobs", "--",
                       "--model", "/m/x.rad" }, &log), 0);
    CHECK_EQ(b.lines(b.calls), 27 + 1);
    CHECK(!has(tiny::read_text(b.out), "# knobs:"));

    auto refused = [&](const std::vector<std::string>& more, const std::string& why) {
        std::vector<std::string> a = { "--layers", "3,7" };
        a.insert(a.end(), more.begin(), more.end());
        std::string l;
        CHECK(b.run(a, &l) != 0);
        CHECK(has(l, why));
        if (!has(l, why)) fprintf(stderr, "    wanted '%s' in:\n%s\n", why.c_str(), l.c_str());
    };
    std::remove(b.calls.c_str());
    refused({ "--max-kl", "0.05", "--", "--model", "m", "--qsa-bkey-dtype", "i8" },
            "--qsa-bkey-dtype is the knob bkey_i8, which rad-autoquant tries on the answer");
    refused({ "--max-kl", "0.05", "--knobs", "bkey", "--", "--model", "m" },
            "--knobs: 'bkey' is not a knob; they are bkey_i8, state_f16");
    refused({ "--max-kl", "0.05", "--knobs", "bkey_i8", "--no-knobs", "--", "--model", "m" },
            "--knobs and --no-knobs contradict each other");
    refused({ "--max-dkl", "-0.01", "--", "--model", "m" }, "--max-dkl '-0.01' is not a mean KL");
    refused({ "--max-dnll", "x", "--", "--model", "m" }, "--max-dnll 'x' is not a mean NLL change");
    refused({ "--max-dkl", "0.01", "--noise-flags", " ", "--", "--model", "m" },
            "--noise-flags names no engine flag");
    refused({ "--max-dkl", "0.01", "--noise-flags", "--kld-out x", "--", "--model", "m" },
            "--noise-flags: --kld-out is set by rad-autoquant");
    refused({ "--bits", "6", "--max-dkl", "0.01", "--", "--model", "m" },
            "--bits and --max-dkl ask opposite questions");
    refused({ "--bits", "6", "--no-knobs", "--", "--model", "m" }, "go with --max-kl, --max-dkl");
    CHECK_EQ(b.lines(b.calls), 0);
}

/* --reuse reads the noise run and the knob runs back like every other: a second search with the
 * same flags runs nothing and writes the same recipe. The noise run is named after the base and
 * --noise-flags (noise_fp8_HASH), so other noise flags are one run of their own, and here, where
 * the stand-in's floor does not depend on them, the only one. The f16 state costs 0.02, past the
 * room the answer leaves under floor + 0.0012, so both knobs together are not run. */
TEST(the_driver_reuses_the_noise_floor_and_the_knob_runs_from_the_work_dir) {
    Bench b;
    REQUIRE(b.dir.ok());
    REQUIRE(tiny::write_text(b.table, cost_table(kFloor + "flag i8 0.0001\nflag f16 0.02\n")));
    std::vector<std::string> args = { "--reuse", "--layers", "3,7,11,15", "--max-dkl", "0.0012",
                                      "--", "--model", "/m/x.rad" };
    std::string log;
    REQUIRE_EQ(b.run(args, &log), 0);
    const int first = b.lines(b.calls);
    CHECK_EQ(first, 10 + 1 + 16 + 1 + 2);
    const std::string recipe = tiny::read_text(b.out);
    CHECK(has(recipe, " --qsa-bkey-dtype i8\n"));
    log.clear();
    REQUIRE_EQ(b.run(args, &log), 0);
    CHECK_EQ(b.lines(b.calls), first);
    CHECK(has(log, "rad-autoquant: noise_fp8_"));
    CHECK(has(log, "rad-autoquant: knob_bkey_i8_"));
    CHECK(!has(log, "rad-autoquant: knob_bkey_i8+state_f16_"));
    CHECK_EQ(tiny::read_text(b.out), recipe);

    args.insert(args.begin(), { "--noise-flags", "--max-num-batched-tokens 32" });
    log.clear();
    REQUIRE_EQ(b.run(args, &log), 0);
    CHECK_EQ(b.lines(b.calls), first + 1);
    CHECK(has(tiny::read_text(b.calls), "--max-num-batched-tokens 32 --kv-cache-dtype fp8"));
    CHECK(has(tiny::read_text(b.out), "every layer fp8 with --max-num-batched-tokens 32: "));
}

RAD_TEST_MAIN()
