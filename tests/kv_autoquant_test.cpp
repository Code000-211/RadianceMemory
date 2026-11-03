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
#include <map>
#include <random>
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
    CHECK(kv_layouts() == std::vector<int>({ RAD_KVQ_I8H, RAD_KVQ_NONE, RAD_KVQ_V4, RAD_KVQ_K6V4,
                                             RAD_KVQ_K4H, RAD_KVQ_K4HV2H }));
    /* fp8's 8 bits an element, and the 25/32, 22/32, 18/32 and 14/32 of it docs/GUIDE.md §4.4
     * states; i8h at 258/256 of it (an f16 scale a 256-dim head is 2 bits a group of 32). */
    CHECK_EQ(kv_layout_group_bits(RAD_KVQ_I8H), 516);
    CHECK_EQ(kv_layout_group_bits(RAD_KVQ_NONE), 512);
    CHECK_EQ(kv_layout_group_bits(RAD_KVQ_V4), 400);
    CHECK_EQ(kv_layout_group_bits(RAD_KVQ_K6V4), 352);
    CHECK_EQ(kv_layout_group_bits(RAD_KVQ_K4H), 288);
    CHECK_EQ(kv_layout_group_bits(RAD_KVQ_K4HV2H), 224);
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

/* ================================================================== the driver */

namespace {

/* The stand-in engine: it answers a KL-mode run (`--kv-recipe R ... --kld-out O`) with a report
 * whose mean KL is 0.04 plus, for each recipe line, the table's cost of that layer in that layout
 * (fp8 costs nothing), and appends its arguments to `calls`. A table line `fail N L` makes the run
 * of layer N in layout L exit 1. A table line `pair P` adds P for every narrowed layer past the
 * first: costs that do not add up when layouts are mixed, as the card measures them. */
struct Bench {
    tiny::TempDir dir{ "autoquant" };
    std::string engine = dir / "engine.sh", table = dir / "costs", calls = dir / "calls",
                ref = dir / "ref", out = dir / "model.kvrecipe";

    explicit Bench(const std::vector<int64_t>& doc_tokens = { 40000, 65536 }) {
        tiny::write_text(engine,
            "#!/bin/sh\n"
            "echo \"$*\" >> '" + calls + "'\n"
            "recipe=; out=\n"
            "while [ $# -gt 0 ]; do\n"
            "  case \"$1\" in --kv-recipe) recipe=$2; shift;; --kld-out) out=$2; shift;; esac\n"
            "  shift\n"
            "done\n"
            "awk -v t='" + table + "' 'BEGIN { while ((getline l < t) > 0) { split(l, a, \" \");\n"
            "    if (a[1] == \"fail\") f[a[2] \" \" a[3]] = 1; else if (a[1] == \"pair\") p = a[2];\n"
            "    else c[a[1] \" \" a[2]] = a[3] } }\n"
            "  $1 == \"layer\" { s += c[$2 \" \" $3]; if ($3 != \"fp8\") n++; if (f[$2 \" \" $3]) bad = 1 }\n"
            "  END { if (bad) { print \"stand-in: refused\" > \"/dev/stderr\"; exit 1 }\n"
            "        if (n > 1) s += p * (n - 1)\n"
            "        printf \"{\\\"all\\\": {\\\"kld\\\": {\\\"mean\\\": %.9f}}}\\n\", 0.04 + s }'"
            " \"$recipe\" > \"$out\" || { rm -f \"$out\"; exit 1; }\n");
        ::chmod(engine.c_str(), 0755);
        ::mkdir(ref.c_str(), 0755);
        std::string docs;
        for (int64_t n : doc_tokens)
            docs += (docs.empty() ? "" : ", ") + std::string("{\"tokens\": ") + std::to_string(n) +
                    ", \"score_from\": 0}";
        tiny::write_text(ref + "/kld.json", "{\"format\": \"radiance-kld-1\", \"docs\": [" + docs + "]}\n");
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

/* Four attention layers with costs from a table: one run with every layer fp8 and one for each
 * layer in each narrower layout, the costs the runs' means above the base, the assignment the
 * solver's on those costs, and a recipe the server's own --kv-recipe parse loads -- a k4h_v4
 * layer, written as `layer 7 k4h_v4`, loaded rotated. */
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

    /* 1 + 4 layers x 4 narrower layouts (i8h, wider than the base, is no candidate), each with the
     * engine's flags and the driver's. */
    CHECK_EQ(b.lines(b.calls), 17);
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
    CHECK(has(recipe, "# every layer fp8: mean KL 0.040000"));
    CHECK(has(recipe, "#       3    4.000e-03    9.000e-03"));
    CHECK(has(recipe, "# chosen: 6.2500 bits an element"));
    for (const char* f : { "base.kvrecipe", "base.json", "base.log", "layer3_k6_v4.json" }) {
        struct stat st{};
        CHECK(::stat((b.out + ".work/" + f).c_str(), &st) == 0);
    }
    CHECK(has(log, "model.kvrecipe: 6.2500 bits an element"));

    /* --layouts narrows the set; fp8 stays as the base, and the room for one fp8 layer goes to
     * layer 3, whose fp8_v4 costs most. */
    std::remove(b.calls.c_str());
    log.clear();
    REQUIRE_EQ(b.run({ "--layers", "3,7,11,15", "--bits", "6.75", "--layouts", "fp8_v4", "--",
                       "--model", "/m/x.rad" }, &log), 0);
    CHECK_EQ(b.lines(b.calls), 5);
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
    /* A cost is KL above the fp8 base, clamped at zero, so a wider layout cannot be priced. */
    refused({ "--layers", "3,7", "--bits", "8.5", "--layouts", "i8h", "--", "--model", "m" },
            "--layouts: i8h is wider than the fp8 base");
    refused({ "--layers", "3,7", "--", "--model", "m" }, "are required");
    CHECK_EQ(b.lines(b.calls), 0);

    refused({ "--layers", "3,7", "--bits", "6", "--", "--model", "m" },
            "run layer7_k6_v4 failed; its log, ");
    std::string log;
    b.run({ "--layers", "3,7", "--bits", "6", "--", "--model", "m" }, &log);
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
    /* 1 + 16 per-layer runs, 4 uniform, 1 verified. */
    CHECK_EQ(b.lines(b.calls), 22);

    KvAssignment want;
    REQUIRE_EQ(kv_autoquant_solve_kl(cost_layers(), 0.0012, &want), RAD_OK);
    CHECK(written(b) == kvq_of(cost_layers(), want));
    const std::string recipe = tiny::read_text(b.out);
    CHECK(has(recipe, "the fewest bits measured within a mean KL of 0.0412"));
    CHECK(has(recipe, "verify 1"));
    CHECK(has(recipe, "uniform k6_v4"));
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
    CHECK(has(log, "base: reused from "));
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
    CHECK(has(log, "every layer fp8 already measures a mean KL of 0.040000, above --max-kl 0.03"));
    CHECK_EQ(b.lines(b.calls), 1);
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

RAD_TEST_MAIN()
