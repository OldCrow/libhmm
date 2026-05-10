#include "libhmm/distributions/negative_binomial_distribution.h"
#include "libhmm/io/json_utils.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <numeric>   // For std::accumulate (not in common.h)
#include <algorithm> // For std::for_each (exists in common.h, included for clarity)

using namespace libhmm::constants;

namespace libhmm {

/**
 * Computes the probability mass function for the Negative Binomial distribution.
 *
 * For discrete distributions, this returns the exact probability mass
 * P(X = k) = C(k+r-1, k) * p^r * (1-p)^k
 *
 * @param value The value at which to evaluate the PMF (rounded to nearest integer)
 * @return Probability mass for the given value
 */
double NegativeBinomialDistribution::getProbability(double value) const {
    // Validate input - discrete distributions only accept non-negative integer values
    if (std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }

    // Round to nearest integer and check if it's in valid range
    auto k = static_cast<int>(std::round(value));
    if (k < 0) {
        return math::ZERO_DOUBLE;
    }

    // Handle edge cases
    if (p_ == math::ONE) {
        return (k == 0) ? math::ONE : math::ZERO_DOUBLE;
    }

    if (!isCacheValid())
        updateCache();
    const double logCoeff = logGeneralizedBinomialCoefficient(k);
    const double logProb = logCoeff + r_ * logP_ + static_cast<double>(k) * log1MinusP_;
    const double prob = std::exp(logProb);
    if (std::isnan(prob) || prob < math::ZERO_DOUBLE)
        return math::ZERO_DOUBLE;
    return std::min(prob, math::ONE);
}

/**
 * Fits the distribution parameters to the given data using method of moments.
 *
 * For Negative Binomial distribution, the method of moments estimators are:
 * p̂ = mean / variance (if variance > mean)
 * r̂ = mean² / (variance - mean) (if variance > mean)
 *
 * If variance ≤ mean, the negative binomial model is not appropriate
 * (indicates under-dispersion), so we fall back to default parameters.
 *
 * @param values Vector of observed data points
 */
void NegativeBinomialDistribution::fit(std::span<const double> data) {
    if (data.size() < 2) {
        reset();
        return;
    }
    double mean = 0.0, m2 = 0.0;
    std::size_t count = 0;
    for (const double val : data) {
        if (val >= 0.0 && std::isfinite(val)) {
            ++count;
            const double delta = val - mean;
            mean += delta / static_cast<double>(count);
            m2 += delta * (val - mean);
        }
    }
    if (count < 2) {
        reset();
        return;
    }
    const double var = m2 / static_cast<double>(count - 1);
    if (var <= mean || mean <= math::ZERO_DOUBLE) {
        reset();
        return;
    }
    const double pHat = mean / var;
    const double rHat = (mean * mean) / (var - mean);
    if (!std::isfinite(pHat) || !std::isfinite(rHat) || pHat <= math::ZERO_DOUBLE ||
        pHat > math::ONE || rHat <= math::ZERO_DOUBLE) {
        reset();
        return;
    }
    p_ = pHat;
    r_ = rHat;
    invalidateCache();
}

void NegativeBinomialDistribution::fit(std::span<const double> data,
                                       std::span<const double> weights) {
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (sumW < precision::ZERO || std::isnan(sumW)) {
        reset();
        return;
    }
    double mean = 0.0, cumW = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i] >= 0.0 && std::isfinite(data[i]) && weights[i] > 0.0) {
            cumW += weights[i];
            mean += (weights[i] / cumW) * (data[i] - mean);
        }
    }
    double var = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] >= 0.0 && std::isfinite(data[i]) && weights[i] > 0.0)
            var += weights[i] * (data[i] - mean) * (data[i] - mean);
    var /= sumW;
    if (var <= mean || mean <= math::ZERO_DOUBLE) {
        reset();
        return;
    }
    const double pHat = mean / var, rHat = (mean * mean) / (var - mean);
    if (!std::isfinite(pHat) || !std::isfinite(rHat) || pHat <= math::ZERO_DOUBLE ||
        pHat > math::ONE || rHat <= math::ZERO_DOUBLE) {
        reset();
        return;
    }
    p_ = pHat;
    r_ = rHat;
    invalidateCache();
}

/**
 * Resets the distribution to default parameters (r = 5.0, p = 0.5).
 * This corresponds to a moderate negative binomial distribution.
 */
void NegativeBinomialDistribution::reset() noexcept {
    r_ = 5.0;
    p_ = math::HALF;
    invalidateCache();
}

/**
 * Returns a string representation of the distribution following the standardized format.
 *
 * @return String describing the distribution parameters and statistics
 */
std::string NegativeBinomialDistribution::toString() const {
    std::ostringstream oss{};
    oss << std::fixed << std::setprecision(6);
    oss << "Negative Binomial Distribution:\n";
    oss << "      r (successes) = " << r_ << "\n";
    oss << "      p (success probability) = " << p_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

double NegativeBinomialDistribution::getLogProbability(double value) const noexcept {
    // Validate input - discrete distributions only accept non-negative integer values
    if (std::isnan(value) || std::isinf(value)) {
        return -std::numeric_limits<double>::infinity();
    }

    // Round to nearest integer and check if it's in valid range
    auto k = static_cast<int>(std::round(value));
    if (k < 0) {
        return -std::numeric_limits<double>::infinity();
    }

    // Handle edge cases
    if (p_ == math::ONE) {
        return (k == 0) ? math::ZERO_DOUBLE : -std::numeric_limits<double>::infinity();
    }

    if (!isCacheValid())
        updateCache();
    const double logCoeff = logGeneralizedBinomialCoefficient(k);
    return logCoeff + r_ * logP_ + static_cast<double>(k) * log1MinusP_;
}

double NegativeBinomialDistribution::getCumulativeProbability(double value) const noexcept {
    // Validate input
    if (std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }

    auto k = static_cast<int>(std::floor(value));

    // Handle boundary cases
    if (k < 0) {
        return math::ZERO_DOUBLE;
    }

    // Compute CDF as cumulative sum: P(X <= k) = sum_{i=0}^{k} P(X = i)
    // For efficiency, we limit computation to reasonable range
    const int maxK = std::min(k, 1000); // Practical upper limit for computation

    double cdf = math::ZERO_DOUBLE;
    for (int i = 0; i <= maxK; ++i) {
        cdf += getProbability(static_cast<double>(i));
    }

    return std::min(math::ONE, cdf);
}

bool NegativeBinomialDistribution::operator==(const NegativeBinomialDistribution &other) const {
    const double tolerance = 1e-10;
    return (std::abs(r_ - other.r_) < tolerance) && (std::abs(p_ - other.p_) < tolerance);
}

// Parses the format produced by toString() / operator<<:
//   Negative Binomial Distribution:
//     r (successes) = VALUE
//     p (success probability) = VALUE
//     Mean = VALUE
//     Variance = VALUE
std::istream &operator>>(std::istream &is, libhmm::NegativeBinomialDistribution &distribution) {
    try {
        std::string s, t;
        is >> s >> s >> s;      // "Negative" "Binomial" "Distribution:"
        is >> s >> s >> s >> t; // "r" "(successes)" "=" VALUE
        const double r = std::stod(t);
        is >> s >> s >> s >> s >> t; // "p" "(success" "probability)" "=" VALUE
        const double p = std::stod(t);
        is >> s >> s >> t;
        is >> s >> s >> t; // skip Mean, Variance
        if (is.good())
            distribution.setParameters(r, p);
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

std::ostream &operator<<(std::ostream &os,
                         const libhmm::NegativeBinomialDistribution &distribution) {
    os << distribution.toString();
    return os;
}

void NegativeBinomialDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                            std::span<double> out) const {
    // Tier 1 — concrete non-virtual loop; compiler auto-vectorizes the arithmetic
    // terms under -march=native. Index loop preserved: a std::ranges::transform
    // lambda would add an indirect call boundary that inhibits auto-vectorisation.
    // Tier 2 upgrade requires vectorised generalised log-binomial-coefficient
    // (uses lgamma internally): available via Intel SVML or platform-specific
    // math libraries, but not portably without a math-library dependency.
    if (!isCacheValid())
        updateCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        out[i] = NegativeBinomialDistribution::getLogProbability(observations[i]);
    }
}

std::string NegativeBinomialDistribution::to_json() const {
    return json::write_distribution("NegativeBinomial", {{"r", r_}, {"p", p_}});
}
std::unique_ptr<EmissionDistribution> NegativeBinomialDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double rv = r.read_double();
    r.read_key();
    const double p = r.read_double();
    r.consume('}');
    return std::make_unique<NegativeBinomialDistribution>(rv, p);
}

} // namespace libhmm
