#include "libhmm/distributions/poisson_distribution.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <algorithm>   // For std::max (exists in common.h, included for clarity)
#include <numeric>     // For std::accumulate (not in common.h)
#include <limits>      // For std::numeric_limits (exists in common.h via <climits>)

using namespace libhmm::constants;

namespace libhmm
{

/*
 * Computes log(k!) efficiently using:
 * - Pre-computed cached values for small k (k <= 12)
 * - Stirling's approximation for large k: log(k!) ≈ k*log(k) - k + 0.5*log(2πk)
 */
double PoissonDistribution::logFactorial(int k) const noexcept {
    if (k < 0) return -std::numeric_limits<double>::infinity();
    
    // Ensure cache is valid
    if (!cacheValid_) {
        updateCache();
    }
    
    // For small k, use pre-computed cached values (fastest)
    if (k <= 12) {
        return std::log(smallFactorials_[k]);
    }
    
    // For large k, use Stirling's approximation
    const auto kd = static_cast<double>(k);
    return kd * std::log(kd) - kd + 0.5 * std::log(2.0 * math::PI * kd);
}

/*
 * Computes the Poisson PMF: P(X = k) = (λ^k * e^(-λ)) / k!
 * Uses logarithms for numerical stability: log(P) = k*log(λ) - λ - log(k!)
 */
double PoissonDistribution::getProbability(double value) {
    // Validate input - must be non-negative integer
    if (!isValidCount(value)) {
        return 0.0;
    }
    
    const auto k = static_cast<int>(value);
    
    // Update cache if needed
    if (!cacheValid_) {
        updateCache();
    }
    
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
void PoissonDistribution::fit(const std::vector<Observation>& values) {
    if (values.empty()) {
        reset();
        return;
    }
    
    // Single-pass validation and computation for efficiency
    double sum = math::ZERO_DOUBLE;
    std::size_t validCount = 0;
    
    for (const auto& val : values) {
        // Validate input
        if (val < math::ZERO_DOUBLE || !std::isfinite(val)) {
            throw std::invalid_argument("Poisson distribution requires non-negative finite values");
        }
        if (std::floor(val) != val) {
            throw std::invalid_argument("Poisson distribution requires integer count values");
        }
        
        // Accumulate sum in single pass
        sum += val;
        ++validCount;
    }
    
    // Compute MLE estimate: λ̂ = sample_mean
    const double sampleMean = sum / static_cast<double>(validCount);
    
    // Ensure lambda is positive (handle edge case of all zeros)
    lambda_ = std::max(sampleMean, precision::ZERO);
    cacheValid_ = false;
}

/*
 * Resets the distribution to default parameters.
 */
void PoissonDistribution::reset() noexcept {
    lambda_ = 1.0;
    cacheValid_ = false;
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
std::ostream& operator<<(std::ostream& os, const libhmm::PoissonDistribution& distribution) {
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
    
    // Update cache if needed
    if (!cacheValid_) {
        updateCache();
    }
    
    // Compute log probability: log P(X = k) = k*log(λ) - λ - log(k!)
    const double logProb = k * logLambda_ - lambda_ - logFactorial(k);
    
    return logProb;
}

/*
 * Evaluates the CDF at k using cumulative sum approach
 * For large k, uses asymptotic approximation for efficiency
 */
double PoissonDistribution::getCumulativeProbability(double value) noexcept {
    // Validate input
    if (std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }
    
    if (value < math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }
    
    const auto k = static_cast<int>(std::floor(value));
    
    // For very large k or lambda, the cumulative sum becomes computationally expensive
    // and numerically unstable. In such cases, use normal approximation.
    if (k > 100 && lambda_ > 100.0) {
        // Ensure cache is valid
        if (!cacheValid_) {
            updateCache();
        }
        
        // Normal approximation with continuity correction: P(X ≤ k) ≈ Φ((k + 0.5 - λ) / √λ)
        // Use cached sqrt(lambda) for efficiency
        const double z = (static_cast<double>(k) + 0.5 - lambda_) * invSqrtLambda_;
        return 0.5 * (1.0 + std::erf(z / math::SQRT_2));
    }
    
    // For moderate values, compute CDF as cumulative sum: P(X ≤ k) = Σ(i=0 to k) P(X = i)
    double cdf = math::ZERO_DOUBLE;
    for (int i = 0; i <= k; ++i) {
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
bool PoissonDistribution::operator==(const PoissonDistribution& other) const {
    const double tolerance = 1e-10;
    return std::abs(lambda_ - other.lambda_) < tolerance;
}

/*
 * Stream input operator implementation.
 * Expects format: "Poisson Distribution: λ = <value>"
 */
std::istream& operator>>(std::istream& is, libhmm::PoissonDistribution& distribution) {
    std::string token;
    double lambda = 0.0;
    
    try {
        // Skip "Poisson Distribution: λ ="
        std::string lambda_str;
        is >> token >> token >> token >> token >> lambda_str;
        lambda = std::stod(lambda_str);
        
        if (is.good()) {
            distribution.setLambda(lambda);
        }
        
    } catch (const std::exception& e) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }
    
    return is;
}

} // namespace libhmm
