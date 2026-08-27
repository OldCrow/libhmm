#include "libhmm/distributions/binomial_distribution.h"
#include "libhmm/io/json_utils.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <numeric>   // For std::accumulate (not in common.h)
#include <algorithm> // For std::for_each, std::max_element (exists in common.h, included for clarity)

using namespace libhmm::constants;

namespace libhmm {

/**
 * Computes the probability mass function for the Binomial distribution.
 *
 * For discrete distributions, this returns the exact probability mass
 * P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
 *
 * @param value The value at which to evaluate the PMF (rounded to nearest integer)
 * @return Probability mass for the given value
 */
double BinomialDistribution::getProbability(double value) const {
    // Validate input - discrete distributions only accept non-negative integer values
    if (std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }

    // Round to nearest integer and check if it's in valid range.
    // #88: the range check against n_ must happen on the rounded double,
    // before the cast to int -- casting an out-of-range finite double is UB
    // ([conv.fpint]) and is ISA-dependent.
    const double rounded = std::round(value);
    if (rounded < 0.0 || rounded > static_cast<double>(n_)) {
        return math::ZERO_DOUBLE;
    }
    auto k = static_cast<int>(rounded);

    // Handle edge cases
    if (p_ == math::ZERO_DOUBLE) {
        return (k == 0) ? math::ONE : math::ZERO_DOUBLE;
    }
    if (p_ == math::ONE) {
        return (k == n_) ? math::ONE : math::ZERO_DOUBLE;
    }

    // Ensure cache is valid
    ensureCache();
    const double logCoeff = logBinomialCoefficient(n_, k);
    const double logProb =
        logCoeff + static_cast<double>(k) * logP_ + static_cast<double>(n_ - k) * log1MinusP_;
    const double prob = std::exp(logProb);
    if (std::isnan(prob) || prob < math::ZERO_DOUBLE)
        return math::ZERO_DOUBLE;
    return std::min(prob, math::ONE);
}

/**
 * Fits p (exact MLE) and estimates n from data.
 *
 * Given n, the MLE for p is exact: p̂ = k̄ / n.
 *
 * n cannot be estimated via MLE in closed form when unknown: the joint MLE
 * for (n, p) is not guaranteed to exist — the profile likelihood in n is
 * non-decreasing for many datasets, giving no finite maximum. The maximum
 * observed value is used as a lower bound (the smallest n consistent with
 * all observations) and keeps p̂ ∈ [0,1]. In EM this is acceptable: n is
 * effectively fixed between steps and only p is re-estimated.
 *
 * @param data Observed non-negative integer counts.
 */
double BinomialDistribution::sample(std::mt19937_64 &rng) const {
    std::binomial_distribution<int> dist(n_, p_);
    return static_cast<double>(dist(rng));
}

void BinomialDistribution::fit(std::span<const double> data) {
    if (data.empty()) {
        reset();
        return;
    }
    int maxObs = 0;
    double sum = 0.0;
    std::size_t validCount = 0;
    for (const double val : data) {
        // #88: bound-check the rounded value against INT_MAX before casting
        // -- n_ is not yet known during unweighted fit, so INT_MAX is the
        // only available domain bound here.
        const double rounded = std::round(val);
        if (val >= 0.0 && std::isfinite(val) &&
            rounded <= static_cast<double>(std::numeric_limits<int>::max())) {
            const auto intVal = static_cast<int>(rounded);
            maxObs = std::max(maxObs, intVal);
            sum += static_cast<double>(intVal);
            ++validCount;
        }
    }
    if (validCount == 0) {
        reset();
        return;
    }
    if (maxObs == 0) {
        n_ = 1;
        p_ = math::ZERO_DOUBLE;
        invalidateCache();
        return;
    }
    n_ = maxObs;
    p_ = std::max(math::ZERO_DOUBLE, std::min(math::ONE, (sum / static_cast<double>(validCount)) /
                                                             static_cast<double>(n_)));
    invalidateCache();
}

void BinomialDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    // Guard: keep current parameters when effective weight is near zero.
    // Calling reset() would destroy valid parameters and cause state collapse in EM.
    if (sumW < precision::ZERO || std::isnan(sumW))
        return;
    int maxObs = 0;
    double sumWX = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        // #88: bound-check before casting -- see the unweighted fit() above.
        const double rounded = std::round(data[i]);
        if (data[i] >= 0.0 && std::isfinite(data[i]) && weights[i] > 0.0 &&
            rounded <= static_cast<double>(std::numeric_limits<int>::max())) {
            const auto intVal = static_cast<int>(rounded);
            maxObs = std::max(maxObs, intVal);
            sumWX += weights[i] * static_cast<double>(intVal);
        }
    }
    if (maxObs == 0) {
        n_ = 1;
        p_ = math::ZERO_DOUBLE;
        invalidateCache();
        return;
    }
    n_ = maxObs;
    p_ = std::max(math::ZERO_DOUBLE, std::min(math::ONE, (sumWX / sumW) / static_cast<double>(n_)));
    invalidateCache();
}

/**
 * Resets the distribution to default parameters (n = 10, p = 0.5).
 * This corresponds to a balanced binomial distribution with moderate number of trials.
 */
void BinomialDistribution::reset() noexcept {
    n_ = 10;
    p_ = math::HALF;
    invalidateCache();
}

/**
 * Returns a string representation of the distribution following the standardized format.
 *
 * @return String describing the distribution parameters and statistics
 */
std::string BinomialDistribution::toString() const {
    std::ostringstream oss{};
    oss << std::fixed << std::setprecision(6);
    oss << "Binomial Distribution:\n";
    oss << "      n (trials) = " << n_ << "\n";
    oss << "      p (success probability) = " << p_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

double BinomialDistribution::getLogProbability(double value) const noexcept {
    // Validate input - discrete distributions only accept non-negative integer values
    if (std::isnan(value) || std::isinf(value)) {
        return -std::numeric_limits<double>::infinity();
    }

    // Round to nearest integer and check if it's in valid range.
    // #88: bound-check before casting -- see getProbability().
    const double rounded = std::round(value);
    if (rounded < 0.0 || rounded > static_cast<double>(n_)) {
        return -std::numeric_limits<double>::infinity();
    }
    auto k = static_cast<int>(rounded);

    // Handle edge cases
    if (p_ == math::ZERO_DOUBLE) {
        return (k == 0) ? math::ZERO_DOUBLE : -std::numeric_limits<double>::infinity();
    }
    if (p_ == math::ONE) {
        return (k == n_) ? math::ZERO_DOUBLE : -std::numeric_limits<double>::infinity();
    }

    // Ensure cache is valid
    ensureCache();
    const double logCoeff = logBinomialCoefficient(n_, k);
    return logCoeff + static_cast<double>(k) * logP_ + static_cast<double>(n_ - k) * log1MinusP_;
}

double BinomialDistribution::getCumulativeProbability(double value) const noexcept {
    // Validate input
    if (std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }

    // #88: bound-check the floored value before casting -- see
    // getProbability(). Any value at or beyond n_ (including one too large
    // to represent as int) has cumulative probability 1.0, so it is safe to
    // resolve that case before the cast.
    const double floored = std::floor(value);
    if (floored < 0.0) {
        return math::ZERO_DOUBLE;
    }
    if (floored >= static_cast<double>(n_)) {
        return math::ONE;
    }
    auto k = static_cast<int>(floored);

    // Compute CDF as cumulative sum: P(X <= k) = sum_{i=0}^{k} P(X = i)
    double cdf = math::ZERO_DOUBLE;
    for (int i = 0; i <= k; ++i) {
        cdf += getProbability(static_cast<double>(i));
    }

    return std::min(math::ONE, cdf);
}

bool BinomialDistribution::operator==(const BinomialDistribution &other) const {
    const double tolerance = 1e-10;
    return (n_ == other.n_) && (std::abs(p_ - other.p_) < tolerance);
}

std::ostream &operator<<(std::ostream &os, const libhmm::BinomialDistribution &distribution) {
    os << distribution.toString();
    return os;
}

// Parses the format produced by toString() / operator<<:
//   Binomial Distribution:
//     n (trials) = VALUE
//     p (success probability) = VALUE
//     Mean = VALUE
//     Variance = VALUE
std::istream &operator>>(std::istream &is, libhmm::BinomialDistribution &distribution) {
    try {
        std::string s, t;
        is >> s >> s;           // "Binomial" "Distribution:"
        is >> s >> s >> s >> t; // "n" "(trials)" "=" VALUE
        const int n = static_cast<int>(std::stod(t));
        is >> s >> s >> s >> s >> t; // "p" "(success" "probability)" "=" VALUE
        const double p = std::stod(t);
        is >> s >> s >> t;
        is >> s >> s >> t; // skip Mean, Variance
        if (is.good())
            distribution.setParameters(n, p);
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

void BinomialDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                    std::span<double> out) const {
    // Tier 1 — concrete non-virtual loop; compiler auto-vectorizes the arithmetic
    // terms under -march=native. Index loop preserved: a std::ranges::transform
    // lambda would add an indirect call boundary that inhibits auto-vectorisation.
    // Tier 2 upgrade does NOT need lgamma — this distribution never calls it.
    // logBinomialCoefficient is three lookups into logFactorialCache_ (all
    // three arguments are integers), so the blocker is the gather, exactly as
    // for Poisson and Discrete. The old claim that this "uses lgamma
    // internally" was wrong about its own code; corrected 2026-08-16.
    checkBatchSpans(observations.size(), out.size());
    ensureCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        out[i] = BinomialDistribution::getLogProbability(observations[i]);
    }
}

std::string BinomialDistribution::to_json() const {
    return json::write_distribution("Binomial", {{"n", static_cast<double>(n_)}, {"p", p_}});
}
std::unique_ptr<EmissionDistribution> BinomialDistribution::from_json(json::Reader &r) {
    r.read_key();
    const int n = static_cast<int>(r.read_double());
    r.read_key();
    const double p = r.read_double();
    r.consume('}');
    return std::make_unique<BinomialDistribution>(n, p);
}

} // namespace libhmm
