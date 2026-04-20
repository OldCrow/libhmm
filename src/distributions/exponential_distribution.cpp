#include "libhmm/distributions/exponential_distribution.h"
#include <limits>
#include <numeric>
#include <span>

using namespace libhmm::constants;

namespace libhmm
{

/**
 * Computes the probability density function for the Exponential distribution.
 * 
 * For continuous distributions in discrete sampling contexts, we approximate
 * the probability as P(x - ε <= X <= x) = F(x) - F(x - ε) where ε is a small tolerance.
 * 
 * This provides a numerically stable approximation of the PDF scaled by the tolerance,
 * which is appropriate for discrete sampling of continuous distributions.
 * 
 * @param x The value at which to evaluate the probability
 * @return Approximated probability for discrete sampling
 */            
double ExponentialDistribution::getProbability(double value) const {
    // Exponential distribution has support [0, ∞)
    if (value < math::ZERO_DOUBLE || std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }
    
    // For continuous distributions, we return the actual PDF value
    // This is more mathematically correct than the old discrete approximation
    if (value == math::ZERO_DOUBLE) {
        // At x=0, PDF equals λ (the rate parameter)
        return lambda_;
    }
    
    if (!isCacheValid()) updateCache();
    return lambda_ * std::exp(negLambda_ * value);
}

/**
 * Computes the logarithm of the probability density function for numerical stability.
 * 
 * For exponential distribution: log(f(x)) = log(λ) - λx for x ≥ 0
 * 
 * @param x The value at which to evaluate the log-PDF
 * @return Natural logarithm of the probability density, or -∞ for invalid values
 */
double ExponentialDistribution::getLogProbability(double value) const noexcept {
    // Exponential distribution has support [0, ∞)
    if (value < math::ZERO_DOUBLE || std::isnan(value) || std::isinf(value)) {
        return -std::numeric_limits<double>::infinity();
    }
    
    if (!isCacheValid()) updateCache();
    return logLambda_ - lambda_ * value;
}

/**
 * Evaluates the CDF for the Exponential distribution at x.
 * 
 * Formula: F(x) = 1 - exp(-λx) for x ≥ 0
 * 
 * @param x The value at which to evaluate the CDF
 * @return Cumulative probability P(X ≤ x)
 */
double ExponentialDistribution::getCumulativeProbability(double x) const noexcept {
    const double y = math::ONE - std::exp(-lambda_ * x);
    assert(y >= math::ZERO_DOUBLE);
    return y;
}

/**
 * Fits the distribution parameters to the given data using maximum likelihood estimation.
 * 
 * For the Exponential distribution, the MLE of the rate parameter is:
 * λ = 1 / sample_mean
 *
 * The sample mean is calculated as:
 * mean = Σ(x_i) / N for i = 1 to N
 *
 * @param values Vector of observed data points
 */                   
void ExponentialDistribution::fit(std::span<const double> data) {
    if (data.size() <= 1) { reset(); return; }

    double mean = 0.0;
    std::size_t count = 0;
    for (const double val : data) {
        if (val < 0.0 || std::isnan(val) || std::isinf(val)) { reset(); return; }
        ++count;
        mean += (val - mean) / static_cast<double>(count);
    }

    if (mean <= 0.0 || !std::isfinite(mean)) { reset(); return; }
    const double lam = 1.0 / mean;
    if (!std::isfinite(lam) || lam <= 0.0) { reset(); return; }
    lambda_ = lam;
    invalidateCache();
}

void ExponentialDistribution::fit(std::span<const double> data,
                                   std::span<const double> weights) {
    // Weighted MLE: λ = 1 / weighted_mean
    double sumW = 0.0, sumWX = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        sumW  += weights[i];
        sumWX += weights[i] * data[i];
    }
    if (sumW < precision::ZERO || std::isnan(sumW) || sumWX <= 0.0) {
        reset(); return;
    }
    const double weightedMean = sumWX / sumW;
    if (weightedMean <= 0.0 || !std::isfinite(weightedMean)) { reset(); return; }
    const double lam = 1.0 / weightedMean;
    if (!std::isfinite(lam) || lam <= 0.0) { reset(); return; }
    lambda_ = lam;
    invalidateCache();
}

/**
 * Resets the distribution to default parameters (λ = 1.0).
 * This corresponds to the standard exponential distribution.
 */
void ExponentialDistribution::reset() noexcept {
    lambda_ = math::ONE;
    invalidateCache();
}

std::string ExponentialDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Exponential Distribution:\n";
    oss << "      λ (rate parameter) = " << lambda_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    return oss.str();
}

std::ostream& operator<<( std::ostream& os, 
        const libhmm::ExponentialDistribution& distribution ){
    os << "Exponential Distribution: " << std::endl;
    os << "    Rate parameter = " << distribution.getLambda( ) << std::endl;
    os << std::endl;
    
    return os;
}

std::istream& operator>>( std::istream& is,
        libhmm::ExponentialDistribution& distribution ){
    std::string token, lambda_str;
    
    try {
        is >> token; // "Rate"
        is >> token; // "parameter"
        is >> token; // "="
        is >> lambda_str;
        double lambda = std::stod(lambda_str);
        
        // Use setLambda for validation
        distribution.setLambda(lambda);
        
    } catch (const std::exception& e) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }

    return is;
}

bool ExponentialDistribution::operator==(const ExponentialDistribution& other) const {
    using namespace libhmm::constants;
    return std::abs(lambda_ - other.lambda_) < precision::LIMIT_TOLERANCE;
}

}
