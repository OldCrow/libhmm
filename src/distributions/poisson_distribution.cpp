#include "libhmm/distributions/poisson_distribution.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iomanip>

namespace libhmm
{

/*
 * Computes log(k!) efficiently using:
 * - Direct computation for small k (k <= 12)
 * - Stirling's approximation for large k: log(k!) ≈ k*log(k) - k + 0.5*log(2πk)
 */
double PoissonDistribution::logFactorial(int k) const noexcept {
    if (k < 0) return -std::numeric_limits<double>::infinity();
    if (k == 0 || k == 1) return 0.0;
    
    // For small k, use direct computation (more accurate)
    if (k <= 12) {
        double result = 0.0;
        for (int i = 2; i <= k; ++i) {
            result += std::log(static_cast<double>(i));
        }
        return result;
    }
    
    // For large k, use Stirling's approximation
    const double kd = static_cast<double>(k);
    return kd * std::log(kd) - kd + 0.5 * std::log(2.0 * PI * kd);
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
    
    const int k = static_cast<int>(value);
    
    // Update cache if needed
    if (!cacheValid_) {
        updateCache();
    }
    
    // Handle edge cases
    if (k == 0) {
        return std::exp(-lambda_);
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
 */
void PoissonDistribution::fit(const std::vector<Observation>& values) {
    if (values.empty()) {
        reset();
        return;
    }
    
    // Validate that all values are non-negative counts
    for (const auto& val : values) {
        if (val < 0.0 || !std::isfinite(val)) {
            throw std::invalid_argument("Poisson distribution requires non-negative finite values");
        }
        if (std::floor(val) != val) {
            throw std::invalid_argument("Poisson distribution requires integer count values");
        }
    }
    
    // Compute sample mean (MLE estimator for λ)
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const double sampleMean = sum / static_cast<double>(values.size());
    
    // Ensure lambda is positive (handle edge case of all zeros)
    lambda_ = std::max(sampleMean, ZERO);
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
    std::ostringstream oss;
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
 * Stream input operator implementation.
 * Expects format: "Poisson Distribution: λ = <value>"
 */
std::istream& operator>>(std::istream& is, libhmm::PoissonDistribution& distribution) {
    std::string word;
    double lambda;
    
    // Skip "Poisson Distribution: λ ="
    is >> word >> word >> word >> word >> lambda;
    
    if (is.fail()) {
        throw std::runtime_error("Failed to parse Poisson distribution from stream");
    }
    
    try {
        distribution.setLambda(lambda);
    } catch (const std::exception& e) {
        is.setstate(std::ios::failbit);
        throw std::runtime_error("Invalid lambda value in Poisson distribution input: " + std::string(e.what()));
    }
    
    return is;
}

} // namespace libhmm
