#include "libhmm/distributions/poisson_distribution.h"
#include "libhmm/math/weighted_stats.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <span>

using namespace libhmm::constants;

namespace libhmm {

/*
 * Computes log(k!) with exact arithmetic:
 * - Pre-computed cached values for small k (k <= 12)
 * - std::lgamma(k + 1) for larger k
 */
double PoissonDistribution::logFactorial(int k) const noexcept {
    if (k < 0)
        return -std::numeric_limits<double>::infinity();

    if (!isCacheValid())
        updateCache();
    if (k <= 12) {
        return std::log(smallFactorials_[k]);
    }
    return std::lgamma(static_cast<double>(k) + 1.0);
}

/*
 * Computes the Poisson PMF: P(X = k) = (λ^k * e^(-λ)) / k!
 * Uses logarithms for numerical stability: log(P) = k*log(λ) - λ - log(k!)
 */
double PoissonDistribution::getProbability(double value) const {
    if (!isValidCount(value))
        return 0.0;
    const auto k = static_cast<int>(value);
    if (!isCacheValid())
        updateCache();

    // Handle edge cases - use cached exp(-lambda) for efficiency
    if (k == 0) {
        return expNegLambda_;
    }

    // For very large lambda or k, check for potential overflow/underflow
    if (lambda_ > 700.0 || k > 700) {
        // Use log-space computation to avoid overflow
        const double logProb = k * logLambda_ - lambda_ - logFactorial(k);

        // Check for underflow
        if (logProb < -700.0) {
            return 0.0;
        }

        return std::exp(logProb);
    }

    // Standard computation for moderate values
    const double logProb = k * logLambda_ - lambda_ - logFactorial(k);
    return std::exp(logProb);
}

/*
 * Fits the Poisson distribution to data using Maximum Likelihood Estimation.
 * For Poisson, MLE of λ is simply the sample mean: λ̂ = (1/n) * Σ(x_i)
 * Uses single-pass algorithm for efficiency.
 */
void PoissonDistribution::fit(std::span<const double> data) {
    if (data.empty()) {
        reset();
        return;
    }
    double sum = 0.0;
    for (const double val : data) {
        if (val < 0.0 || !std::isfinite(val))
            throw std::invalid_argument("Poisson fit: requires non-negative finite values");
        sum += val;
    }
    lambda_ = std::max(sum / static_cast<double>(data.size()), precision::ZERO);
    invalidateCache();
}

void PoissonDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    // Weighted MLE: λ = weighted mean
    const auto mean = detail::compute_weighted_mean(data, weights);
    if (!mean) {
        reset();
        return;
    }
    lambda_ = std::max(*mean, precision::ZERO);
    invalidateCache();
}

/*
 * Resets the distribution to default parameters.
 */
void PoissonDistribution::reset() noexcept {
    lambda_ = 1.0;
    invalidateCache();
}

/*
 * Creates a string representation of the Poisson distribution.
 */
std::string PoissonDistribution::toString() const {
    std::ostringstream oss{};
    oss << "Poisson Distribution:\n";
    oss << "      λ (rate parameter) = " << std::fixed << std::setprecision(6) << lambda_ << "\n";
    oss << "      Mean = " << std::fixed << std::setprecision(6) << getMean() << "\n";
    oss << "      Variance = " << std::fixed << std::setprecision(6) << getVariance() << "\n";

    return oss.str();
}

/*
 * Stream output operator implementation.
 */
std::ostream &operator<<(std::ostream &os, const libhmm::PoissonDistribution &distribution) {
    os << distribution.toString();
    return os;
}

/*
 * Evaluates the logarithm of the probability mass function
 * Formula: log P(X = k) = k*log(λ) - λ - log(k!)
 * More numerically stable for small probabilities
 */
double PoissonDistribution::getLogProbability(double value) const noexcept {
    // Validate input - must be non-negative integer
    if (!isValidCount(value)) {
        return -std::numeric_limits<double>::infinity();
    }

    const auto k = static_cast<int>(value);

    if (!isCacheValid())
        updateCache();
    const double logProb = k * logLambda_ - lambda_ - logFactorial(k);

    return logProb;
}

/*
 * Evaluates the CDF at k using cumulative sum approach
 * For large k, uses asymptotic approximation for efficiency
 */
double PoissonDistribution::getCumulativeProbability(double k) const noexcept {
    // Validate input
    if (std::isnan(k) || std::isinf(k)) {
        return math::ZERO_DOUBLE;
    }

    if (k < math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }

    const auto kInt = static_cast<int>(std::floor(k));

    // For very large k or lambda, the cumulative sum becomes computationally expensive
    // and numerically unstable. In such cases, use normal approximation.
    if (kInt > 100 && lambda_ > 100.0) {
        if (!isCacheValid())
            updateCache();
        // Normal approximation with continuity correction
        // Use cached sqrt(lambda) for efficiency
        const double z = (static_cast<double>(kInt) + 0.5 - lambda_) * invSqrtLambda_;
        return 0.5 * (1.0 + std::erf(z / math::SQRT_2));
    }

    // For moderate values, compute CDF as cumulative sum: P(X ≤ k) = Σ(i=0 to k) P(X = i)
    double cdf = math::ZERO_DOUBLE;
    for (int i = 0; i <= kInt; ++i) {
        cdf += getProbability(static_cast<double>(i));

        // Early termination if we've accumulated essentially all probability
        if (cdf >= 0.999999) {
            break;
        }
    }

    return std::min(math::ONE, cdf);
}

/*
 * Equality comparison operator with numerical tolerance
 */
bool PoissonDistribution::operator==(const PoissonDistribution &other) const {
    const double tolerance = 1e-10;
    return std::abs(lambda_ - other.lambda_) < tolerance;
}

/*
 * Stream input operator implementation.
 * Expects format: "Poisson Distribution: λ = <value>"
 */
std::istream &operator>>(std::istream &is, libhmm::PoissonDistribution &distribution) {
    try {
        std::string token;
        double lambda = 0.0;
        // Skip "Poisson Distribution: λ ="
        std::string lambda_str;
        is >> token >> token >> token >> token >> lambda_str;
        lambda = std::stod(lambda_str);

        if (is.good()) {
            distribution.setLambda(lambda);
        }

    } catch (const std::exception &) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }

    return is;
}

void PoissonDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                   std::span<double> out) const {
    // Tier 1 — concrete non-virtual loop; compiler auto-vectorizes the arithmetic
    // terms under -march=native / /arch:AVX512.
    // Tier 2 upgrade requires vectorised log-factorial (or lgamma(k+1)): available
    // via Intel SVML or platform-specific math libraries, but not portably
    // without a math-library dependency. A small-k lookup table (k ≤ 20) could
    // serve as a portable partial optimisation.
    if (!isCacheValid())
        updateCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        out[i] = PoissonDistribution::getLogProbability(observations[i]);
    }
}

} // namespace libhmm
