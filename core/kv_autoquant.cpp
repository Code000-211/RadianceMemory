/* kv_autoquant.cpp -- the per-layer KV layout assignment under a bit budget or a KL limit
 * (kv_autoquant.h). */
#include "kv_autoquant.h"

#include "rad_core.h"

#include <cmath>
#include <limits>
#include <numeric>

namespace rad {

namespace {

constexpr double kNone = std::numeric_limits<double>::infinity();

/* The knapsack's table, shared by both questions. best[w]: the least summed KL of every layer
 * within w units, infinite where they cannot fit; choice[l][w]: layer l's candidate in that sum.
 * Each layer is the previous table plus each of its candidates, so every assignment within W
 * units is considered once. best[] does not rise with w: a sum that fits w units fits w + 1. */
struct Table {
    int64_t unit = 0, least = 0, most = 0;
    std::vector<double> best;
    std::vector<std::vector<int>> choice;
};

/* The common divisor of the candidates' bits and the narrowest and widest assignments' sums, or
 * RAD_E_INVAL for the inputs both solvers refuse. */
int measure(const std::vector<std::vector<KvCandidate>>& layers, Table* t) {
    if (layers.empty()) return RAD_E_INVAL;
    /* Every cost in units of the candidates' common divisor, so the table is as short as the
     * candidates allow (fp8, fp8_v4 and k6_v4 share 16 bits: 32, 25 and 22 units a layer). */
    for (const std::vector<KvCandidate>& c : layers) {
        if (c.empty()) return RAD_E_INVAL;
        int64_t lo = std::numeric_limits<int64_t>::max(), hi = 0;
        for (const KvCandidate& k : c) {
            if (k.bits <= 0 || !std::isfinite(k.kl)) return RAD_E_INVAL;
            t->unit = std::gcd(t->unit, k.bits);
            lo = std::min(lo, k.bits);
            hi = std::max(hi, k.bits);
        }
        t->least += lo;
        t->most += hi;
    }
    return RAD_OK;
}

void fill(const std::vector<std::vector<KvCandidate>>& layers, int64_t W, Table* t) {
    t->best.assign((size_t)W + 1, 0.0);
    std::vector<double> next((size_t)W + 1);
    t->choice.assign(layers.size(), std::vector<int>((size_t)W + 1, -1));
    for (size_t l = 0; l < layers.size(); ++l) {
        std::fill(next.begin(), next.end(), kNone);
        for (int64_t w = 0; w <= W; ++w)
            for (size_t o = 0; o < layers[l].size(); ++o) {
                const int64_t u = layers[l][o].bits / t->unit;
                if (u > w || t->best[(size_t)(w - u)] == kNone) continue;
                const double s = t->best[(size_t)(w - u)] + layers[l][o].kl;
                if (s < next[(size_t)w]) { next[(size_t)w] = s; t->choice[l][(size_t)w] = (int)o; }
            }
        t->best.swap(next);
    }
}

/* The assignment the table holds at w units, back from the last layer. */
void pick(const std::vector<std::vector<KvCandidate>>& layers, const Table& t, int64_t w,
          KvAssignment* out) {
    out->pick.assign(layers.size(), -1);
    for (size_t l = layers.size(); l-- > 0;) {
        const int o = t.choice[l][(size_t)w];
        out->pick[l] = o;
        out->bits += layers[l][(size_t)o].bits;
        w -= layers[l][(size_t)o].bits / t.unit;
    }
    /* Summed in layer order, so the reported cost does not depend on the table's path. */
    for (size_t l = 0; l < layers.size(); ++l) out->kl += layers[l][(size_t)out->pick[l]].kl;
}

}  /* namespace */

int kv_autoquant_solve(const std::vector<std::vector<KvCandidate>>& layers, int64_t budget,
                       KvAssignment* out) {
    *out = KvAssignment{};
    if (budget < 0) return RAD_E_INVAL;
    Table t;
    if (const int rc = measure(layers, &t); rc != RAD_OK) return rc;
    if (t.least > budget) return RAD_E_INVAL;
    /* Past the widest assignment more budget buys nothing, so the table stops there. */
    const int64_t W = std::min(budget, t.most) / t.unit;
    fill(layers, W, &t);
    pick(layers, t, W, out);
    return RAD_OK;
}

int kv_autoquant_solve_kl(const std::vector<std::vector<KvCandidate>>& layers, double max_kl,
                          KvAssignment* out) {
    *out = KvAssignment{};
    if (!std::isfinite(max_kl) || max_kl < 0) return RAD_E_INVAL;
    Table t;
    if (const int rc = measure(layers, &t); rc != RAD_OK) return rc;
    const int64_t W = t.most / t.unit;
    fill(layers, W, &t);
    /* The fewest units whose least sum is within the limit. Below it no assignment fits, so the
     * one the table holds there spends exactly those units: the fewest bits, and of those the
     * least KL. */
    const double limit = max_kl + 1e-12 * std::max(1.0, std::fabs(max_kl));
    for (int64_t w = t.least / t.unit; w <= W; ++w)
        if (t.best[(size_t)w] <= limit) {
            pick(layers, t, w, out);
            return RAD_OK;
        }
    return RAD_E_NOTFOUND;
}

}  /* namespace rad */
