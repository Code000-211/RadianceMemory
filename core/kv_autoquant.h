/* kv_autoquant.h -- AutoQuant's assignment: which KV layout each attention layer takes, so the
 * model's output moves least for the bits the deployment pays.
 *
 * rad-autoquant (tools/rad_autoquant.cpp) measures, layer by layer, what each layout costs: the
 * mean KL against an unquantised cache when that layer alone takes the layout. With those costs
 * taken as adding up across layers, the recipe is a multiple-choice knapsack: one candidate a
 * layer, the candidates' bits summed within the budget, the summed KL least. The budget is a
 * whole number of bits and every candidate's cost a small one (fp8's 512 a group of 32 K and 32 V
 * elements, docs/GUIDE.md §4.4), so the knapsack is solved exactly by dynamic programming over the
 * bits spent, not approximated by a greedy exchange that can miss the optimum.
 *
 * The same table answers the converse question: the fewest bits whose summed KL stays within a
 * limit (kv_autoquant_solve_kl), which is what `rad-autoquant --max-kl` asks.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace rad {

/* One layout a layer can take: its RAD_KVQ_* (with RAD_KVQ_ROTATED / RAD_KVQ_VMEAN as the recipe
 * parser sets them, kv_autoquant_variants), its bits (any whole unit, the same for every
 * candidate of every layer; rad-autoquant's is a group of 32 K and 32 V elements) and its cost. */
struct KvCandidate {
    int     kvq  = 0;
    int64_t bits = 0;
    double  kl   = 0;
};

struct KvAssignment {
    std::vector<int> pick;     /* per layer, the index of its chosen candidate */
    int64_t          bits = 0; /* the chosen candidates' bits, summed */
    double           kl   = 0; /* and their costs */
};

/* The assignment of least summed KL whose bits sum to at most `budget`. `layers[l]` lists layer l's
 * candidates; the order is the tie-break, as a candidate replaces an earlier one only at a strictly
 * smaller sum, so listing the widest first keeps a layer wide where narrowing it would buy nothing
 * measurable. The same input gives the same assignment, bit for bit.
 *
 * RAD_E_INVAL, with `out` empty, for no layer, a layer without a candidate, a candidate whose bits
 * are not positive or whose cost is not finite, or a budget below the narrowest assignment. */
int kv_autoquant_solve(const std::vector<std::vector<KvCandidate>>& layers, int64_t budget,
                       KvAssignment* out);

/* The assignment of fewest summed bits whose summed KL is at most `max_kl`; of those, the one of
 * least summed KL, with the same tie-break as kv_autoquant_solve. A sum within a relative 1e-12 of
 * `max_kl` counts as within it, so a limit copied from a solved cost is met by that assignment.
 *
 * RAD_E_INVAL, with `out` empty, for the inputs kv_autoquant_solve refuses or a `max_kl` that is
 * negative or not finite; RAD_E_NOTFOUND, with `out` empty, when even the assignment of least
 * summed KL is above `max_kl`. */
int kv_autoquant_solve_kl(const std::vector<std::vector<KvCandidate>>& layers, double max_kl,
                          KvAssignment* out);

/* A candidate a recipe can name: a row of the layout table (core/config.cpp, kKvLayouts) with
 * the words a recipe line may add after it, `rotated` and `vmean`. `kvq` is what kv_recipe_parse
 * makes of `layer N <words>`, `bits` the row's (rotation and the V mean cost none). */
struct KvVariant {
    int32_t     kvq  = 0;
    int         bits = 0;
    std::string words;   /* "fp8_v4 rotated vmean" */
};

/* Every variant: the table's layouts widest first, each plain, `vmean`, `vpred`, `rotated`,
 * `rotated vmean` and `rotated vpred`; with `vmean` (`vpred`) false none that subtracts a V mean
 * (a prediction of V from K). Found by asking the recipe parser, not by rules kept here: a
 * combination it refuses (`fp8 vmean`, `k6_v4 vpred`) is no variant, and one it reads as another
 * (`k4h_v4 rotated` is `k4h_v4`, rotated by its name) is listed once, in its fewest words. So a
 * layout or recipe word added to the table and the parser is a candidate with no change here or
 * in rad-autoquant. A layout the table marks opt-in (kv_layout_opt_in) is left out unless
 * `opt_in` is set. */
std::vector<KvVariant> kv_autoquant_variants(bool vmean, bool opt_in = false, bool vpred = false);

/* The variant order of rad-autoquant's second pass: of the candidates `v` (each a variant's bits
 * and its uniform recipe's measured KL; kvq is not read), the indices on the lower convex hull of
 * KL over bits, by bits from the most accurate down to the narrowest. A layer steps "one
 * narrower" to the next entry with fewer bits than its own. Of equal bits the lower KL stands (an
 * exact tie: the earlier); past the least KL a wider variant is no step down; and a variant above
 * the straight line between its neighbours is skipped, since mixing those two reaches its bits
 * at less KL, so stepping onto it spends KL for almost no bits (Flash-Next: fp8 at 8.00 bits and
 * 0.0262, beside i8h at 8.06 and 0.0236, saves 0.06 bits a step where k6_v4 rotated vmean saves
 * 2.56 for 0.0309; i8h itself lies 0.0002 above the line from bf16 to k6_v4 rotated vmean, so a
 * layer in either steps to k6_v4 rotated vmean). Collinear points stay. Empty for no candidate. */
std::vector<size_t> kv_autoquant_hull(const std::vector<KvCandidate>& v);

/* One limit a measured assignment is checked against, both sides measured from the base (the
 * most accurate recipe): for the mean KL the run's minus the base's and the limit minus the
 * base's; for top-1 agreement, which falls as the cache narrows, the base's minus the run's and
 * the base's minus the floor. Within when excess <= allowed. */
struct KvCheck {
    double excess  = 0;
    double allowed = 0;
};
struct KvRound {
    double               allowed = 0;  /* the summed cost the solver was asked to stay within */
    KvAssignment         pick;
    std::vector<KvCheck> checks;
    bool                 within = false;
};
/* Measures an assignment and states its checks; a non-zero return stops the fit with it. */
using KvMeasure = std::function<int(const KvAssignment&, std::vector<KvCheck>* checks)>;

/* Solve, measure, tighten: the loop `rad-autoquant --max-kl` runs because per-layer costs do not
 * add up when layouts are mixed. Each round asks kv_autoquant_solve_kl for the fewest bits within
 * `allowed` (the first round: `room`), measures that, and stops when every check is within. A
 * check over scales the room by how far the prediction fell short: the next `allowed` is the
 * rejected assignment's predicted cost times the least allowed / excess of its failing checks,
 * and strictly under that prediction, so no rejected assignment comes back, the room falls every
 * round and the bits rise. Stops after `rounds`; where the room would fall below zero (a rejected
 * assignment that predicted no cost leaves the costs nothing to give up); or before measuring a
 * solution of `bits_cap` bits or more, since a recipe already measured within the limits has as
 * few. `trail` gets every measured round, the last one deciding; empty where the first solution
 * reached the cap. RAD_E_INVAL for rounds < 1 or what kv_autoquant_solve_kl refuses; the
 * measurement's own error as it returned it. */
int kv_autoquant_fit(const std::vector<std::vector<KvCandidate>>& layers, double room, int rounds,
                     const KvMeasure& measure, std::vector<KvRound>* trail,
                     int64_t bits_cap = INT64_MAX);

/* The mean of a run's per-position values (a column of the KL mode's `--kld-out` .rows) and its
 * 95 % confidence half-width. Neighbouring positions of a document are correlated (one hard
 * passage raises a run of them), so the spread of single positions understates the noise; batch
 * means over contiguous stretches do not. The documents are the strata: a limit is stated on this
 * corpus, so which documents it holds is fixed and only where in them the positions fell is
 * sampled. Each document of `doc_rows[d]` positions (in order, summing to x's length; empty: one
 * document) is cut into b = max(2, ceil(32 / documents)) contiguous batches; the variance of the
 * mean is the sum over documents of (n_d / n)^2 times the variance of its batch means over b, and
 * the half-width Student's t at the pooled degrees of freedom times its root. RAD_E_INVAL where
 * the rows do not add up, a value is not finite, or no document holds two positions. */
struct KvInterval {
    double  mean = 0;
    double  half = 0;
    int64_t df   = 0;
};
int kv_autoquant_interval(const std::vector<double>& x, const std::vector<int64_t>& doc_rows,
                          KvInterval* out);
/* Student's t at 97.5 %, `df` degrees of freedom (the table to 3 decimals up to 10, a
 * Cornish-Fisher expansion past it, within 1e-3). */
double kv_autoquant_t975(int64_t df);

}  /* namespace rad */
