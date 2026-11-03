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
#include <vector>

namespace rad {

/* One layout a layer can take: its RAD_KVQ_*, its bits (any whole unit, the same for every
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

}  /* namespace rad */
