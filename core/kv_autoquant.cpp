/* kv_autoquant.cpp -- the per-layer KV layout assignment under a bit budget or a KL limit
 * (kv_autoquant.h). */
#include "kv_autoquant.h"

#include "rad_core.h"
#include "rad_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>

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

std::vector<KvVariant> kv_autoquant_variants(bool vmean, bool opt_in) {
    std::vector<KvVariant> v;
    std::set<int32_t> seen;
    /* The parser's refusals are the answer here, not errors to print. */
    const Log was = log_level();
    log_set_level((Log)-1);
    for (int q : kv_layouts()) {
        if (!opt_in && kv_layout_opt_in(q)) continue;
        for (bool rot : { false, true })
            for (bool vm : { false, true }) {
                if (vm && !vmean) continue;
                const std::string words = std::string(kv_layout_name(q)) +
                                          (rot ? " rotated" : "") + (vm ? " vmean" : "");
                std::vector<int32_t> r;
                if (kv_recipe_parse((vm ? "vmean m\n" : "") + std::string("layer 0 ") + words,
                                    "variant", &r) != RAD_OK || r.size() != 1)
                    continue;
                if (seen.insert(r[0]).second) v.push_back({ r[0], kv_layout_group_bits(q), words });
            }
    }
    log_set_level(was);
    return v;
}

int kv_autoquant_fit(const std::vector<std::vector<KvCandidate>>& layers, double room, int rounds,
                     const KvMeasure& measure, std::vector<KvRound>* trail, int64_t bits_cap) {
    trail->clear();
    if (rounds < 1) return RAD_E_INVAL;
    double allowed = room;
    for (int r = 0; r < rounds; ++r) {
        KvRound k;
        k.allowed = allowed;
        if (kv_autoquant_solve_kl(layers, allowed, &k.pick) != RAD_OK) return RAD_E_INVAL;
        if (k.pick.bits >= bits_cap) break;
        if (const int rc = measure(k.pick, &k.checks); rc != RAD_OK) return rc;
        /* The tightest failing check scales the room: allowed / excess < 1 there. One that is not
         * a number, or measured nothing above the base and still failed, leaves nothing. */
        double f = 1;
        k.within = true;
        for (const KvCheck& c : k.checks)
            if (!(c.excess <= c.allowed)) {
                k.within = false;
                f = std::min(f, c.excess > 0 && c.allowed > 0 && std::isfinite(c.excess)
                                    ? c.allowed / c.excess : 0.0);
            }
        trail->push_back(k);
        if (k.within) break;
        /* Under the rejected prediction, so the solver cannot answer it again. */
        const double next = std::min(k.pick.kl * f, k.pick.kl - std::max(1e-12, 1e-9 * k.pick.kl));
        if (next < 0) break;
        allowed = next;
    }
    return RAD_OK;
}

double kv_autoquant_t975(int64_t df) {
    static const double kSmall[] = { 12.706, 4.303, 3.182, 2.776, 2.571,
                                     2.447,  2.365, 2.306, 2.262, 2.228 };
    if (df < 1) return kNone;
    if (df <= 10) return kSmall[df - 1];
    /* Cornish-Fisher in 1/df around the normal quantile (Abramowitz & Stegun 26.7.5). */
    const double z = 1.959963984540054, n = (double)df, z2 = z * z;
    const double g1 = (z2 + 1) * z / 4;
    const double g2 = ((5 * z2 + 16) * z2 + 3) * z / 96;
    const double g3 = (((3 * z2 + 19) * z2 + 17) * z2 - 15) * z / 384;
    const double g4 = ((((79 * z2 + 776) * z2 + 1482) * z2 - 1920) * z2 - 945) * z / 92160;
    return z + g1 / n + g2 / (n * n) + g3 / (n * n * n) + g4 / (n * n * n * n);
}

int kv_autoquant_interval(const std::vector<double>& x, const std::vector<int64_t>& doc_rows,
                          KvInterval* out) {
    *out = KvInterval{};
    std::vector<int64_t> docs = doc_rows;
    if (docs.empty()) docs.push_back((int64_t)x.size());
    int64_t n = 0;
    for (int64_t d : docs) {
        if (d < 0) return RAD_E_INVAL;
        n += d;
    }
    if (n != (int64_t)x.size() || n == 0) return RAD_E_INVAL;
    double sum = 0;
    for (double v : x) {
        if (!std::isfinite(v)) return RAD_E_INVAL;
        sum += v;
    }
    const int64_t b = std::max<int64_t>(2, (32 + (int64_t)docs.size() - 1) / (int64_t)docs.size());
    double var = 0;
    int64_t df = 0, at = 0;
    for (int64_t nd : docs) {
        /* A document too short for b batches takes one a position; one of a single position adds
         * its weight to the mean and nothing to what is known of the spread. */
        const int64_t bd = std::min(b, nd);
        if (bd >= 2) {
            std::vector<double> m((size_t)bd, 0.0);
            for (int64_t k = 0; k < bd; ++k) {
                const int64_t lo = at + k * nd / bd, hi = at + (k + 1) * nd / bd;
                for (int64_t i = lo; i < hi; ++i) m[(size_t)k] += x[(size_t)i];
                m[(size_t)k] /= (double)(hi - lo);
            }
            const double mm = std::accumulate(m.begin(), m.end(), 0.0) / (double)bd;
            double s2 = 0;
            for (double v : m) s2 += (v - mm) * (v - mm);
            s2 /= (double)(bd - 1);
            const double w = (double)nd / (double)n;
            var += w * w * s2 / (double)bd;
            df += bd - 1;
        }
        at += nd;
    }
    if (df < 1) return RAD_E_INVAL;
    out->mean = sum / (double)n;
    out->half = kv_autoquant_t975(df) * std::sqrt(var);
    out->df = df;
    return RAD_OK;
}

}  /* namespace rad */
