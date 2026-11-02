/* kv_autoquant.cpp -- the per-layer KV layout assignment under a bit budget (kv_autoquant.h). */
#include "kv_autoquant.h"

#include "rad_core.h"

#include <cmath>
#include <limits>
#include <numeric>

namespace rad {

int kv_autoquant_solve(const std::vector<std::vector<KvCandidate>>& layers, int64_t budget,
                       KvAssignment* out) {
    *out = KvAssignment{};
    if (layers.empty() || budget < 0) return RAD_E_INVAL;
    /* Every cost in units of the candidates' common divisor, so the table is as short as the
     * candidates allow (fp8, fp8_v4 and k6_v4 share 16 bits: 32, 25 and 22 units a layer). */
    int64_t unit = 0, least = 0, most = 0;
    for (const std::vector<KvCandidate>& c : layers) {
        if (c.empty()) return RAD_E_INVAL;
        int64_t lo = std::numeric_limits<int64_t>::max(), hi = 0;
        for (const KvCandidate& k : c) {
            if (k.bits <= 0 || !std::isfinite(k.kl)) return RAD_E_INVAL;
            unit = std::gcd(unit, k.bits);
            lo = std::min(lo, k.bits);
            hi = std::max(hi, k.bits);
        }
        least += lo;
        most += hi;
    }
    if (least > budget) return RAD_E_INVAL;
    /* Past the widest assignment more budget buys nothing, so the table stops there. */
    const int64_t W = std::min(budget, most) / unit;

    /* best[w]: the least summed KL of the layers so far within w units, infinite where they cannot
     * fit; choice[l][w]: layer l's candidate in that sum. Each layer is the previous table plus
     * each of its candidates, so every assignment within the budget is considered once. */
    constexpr double kNone = std::numeric_limits<double>::infinity();
    std::vector<double> best((size_t)W + 1, 0.0), next((size_t)W + 1);
    std::vector<std::vector<int>> choice(layers.size(), std::vector<int>((size_t)W + 1, -1));
    for (size_t l = 0; l < layers.size(); ++l) {
        std::fill(next.begin(), next.end(), kNone);
        for (int64_t w = 0; w <= W; ++w)
            for (size_t o = 0; o < layers[l].size(); ++o) {
                const int64_t u = layers[l][o].bits / unit;
                if (u > w || best[(size_t)(w - u)] == kNone) continue;
                const double s = best[(size_t)(w - u)] + layers[l][o].kl;
                if (s < next[(size_t)w]) { next[(size_t)w] = s; choice[l][(size_t)w] = (int)o; }
            }
        best.swap(next);
    }

    out->pick.assign(layers.size(), -1);
    int64_t w = W;
    for (size_t l = layers.size(); l-- > 0;) {
        const int o = choice[l][(size_t)w];
        out->pick[l] = o;
        out->bits += layers[l][(size_t)o].bits;
        w -= layers[l][(size_t)o].bits / unit;
    }
    /* Summed in layer order, so the reported cost does not depend on the table's path. */
    for (size_t l = 0; l < layers.size(); ++l) out->kl += layers[l][(size_t)out->pick[l]].kl;
    return RAD_OK;
}

}  /* namespace rad */
