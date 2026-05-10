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

    // Round to nearest integer and check if it's in valid range
    auto k = static_cast<int>(std::round(value));
    if (k < 0 || k > n_) {
        return math::ZERO_DOUBLE;
    }

    // Handle edge cases
    if (p_ == math::ZERO_DOUBLE) {
        return (k == 0) ? math::ONE : math::ZERO_DOUBLE;
    }
    if (p_ == math::ONE) {
        return (k == n_) ? math::ONE : math::ZERO_DOUBLE;
    }

    // Ensure cache is valid
    if (!isCacheValid())
        updateCache();
    const double logCoeff = logBinomialCoefficient(n_, k);
    const double logProb =
        logCoeff + static_cast<double>(k) * logP_ + static_cast<double>(n_ - k) * log1MinusP_;
    const double prob = std::exp(logProb);
    if (std::isnan(prob) || prob < math::ZERO_DOUBLE)
        return math::ZERO_DOUBLE;
    return std::min(prob, math::ONE);
}

/**
 * Fits the distribution parameters to the given data using maximum likelihood estimation.
 *
 * For Binomial distribution with known n, the MLE of p is:
 * p̂ = sample_mean / n
 *
 * If n is unknown, we estimate it as the maximum observed value, then fit p.
 * This is a common approach when the number of trials is not known a priori.
 *
 * @param values Vector of observed data points
 */
void BinomialDistribution::fit(std::span<const double> data) {
    if (data.empty()) {
        reset();
        return;
    }
    int maxObs = 0;
    double sum = 0.0;
    std::size_t validCount = 0;
    for (const double val : data) {
        if (val >= 0.0 && std::isfinite(val)) {
            const auto intVal = static_cast<int>(std::round(val));
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
    if (sumW < precision::ZERO || std::isnan(sumW)) {
        reset();
        return;
    }
    int maxObs = 0;
    double sumWX = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i] >= 0.0 && std::isfinite(data[i]) && weights[i] > 0.0) {
            const auto intVal = static_cast<int>(std::round(data[i]));
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

    // Round to nearest integer and check if it's in valid range
    auto k = static_cast<int>(std::round(value));
    if (k < 0 || k > n_) {
        return -std::numeric_limits<double>::infinity();
    }

    // Handle edge cases
    if (p_ == math::ZERO_DOUBLE) {
        return (k == 0) ? math::ZERO_DOUBLE : -std::numeric_limits<double>::infinity();
    }
    if (p_ == math::ONE) {
        return (k == n_) ? math::ZERO_DOUBLE : -std::numeric_limits<double>::infinity();
    }

    // Ensure cache is valid
    if (!isCacheValid())
        updateCache();
    const double logCoeff = logBinomialCoefficient(n_, k);
    return logCoeff + static_cast<double>(k) * logP_ + static_cast<double>(n_ - k) * log1MinusP_;
}

double BinomialDistribution::getCumulativeProbability(double value) const noexcept {
    // Validate input
    if (std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }

    auto k = static_cast<int>(std::floor(value));

    // Handle boundary cases
    if (k < 0) {
        return math::ZERO_DOUBLE;
    }
    if (k >= n_) {
        return math::ONE;
    }

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
    // terms under -march=native / /arch:AVX512.
    // Tier 2 upgrade requires vectorised log-binomial-coefficient (uses lgamma internally):
    // available via Intel SVML or platform-specific math libraries, but not
    // portably available without a dedicated math-library dependency.
    if (!isCacheValid())
        updateCache();
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
