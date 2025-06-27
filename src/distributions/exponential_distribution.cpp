#include "libhmm/distributions/exponential_distribution.h"
#include <iostream>
#include <numeric>
#include <limits>

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
double ExponentialDistribution::getProbability(double value) {
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
    
    // Ensure cache is valid
    if (!cacheValid_) {
        updateCache();
    }
    
    // Optimized PDF calculation: f(x) = λ * exp(-λx)
    // Use cached -λ value for efficiency
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
    
    // Ensure cache is valid
    if (!cacheValid_) {
        updateCache();
    }
    
    // For exponential: log(f(x)) = log(λ) - λx
    return logLambda_ - lambda_ * value;
}

/*
 * Evaluates the CDF for the Normal distribution at x.  The CDF is defined as
 *
 *   F(x) = 1 - exp( -lambda * x )
 */
double ExponentialDistribution::CDF(double x) const noexcept {
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
void ExponentialDistribution::fit(const std::vector<Observation>& values) {
    // Handle edge cases: empty data or single data point
    if (values.empty()) {
        reset();
        return;
    }
    
    if (values.size() == 1) {
        // For a single data point, MLE is not well-defined.
        // Following the library's convention, reset to default parameters
        reset();
        return;
    }

    // Use Welford's single-pass algorithm for numerical stability and performance
    // This algorithm provides better numerical accuracy than naive sum-and-divide
    double mean = math::ZERO_DOUBLE;
    double n = math::ZERO_DOUBLE;
    
    for (const auto& val : values) {
        // Validate each data point during iteration (fail-fast approach)
        if (val < math::ZERO_DOUBLE || std::isnan(val) || std::isinf(val)) {
            reset(); // Invalid data for exponential distribution
            return;
        }
        
        // Welford's algorithm update
        n += math::ONE;
        double delta = val - mean;
        mean += delta / n;
    }
    
    // Validate that mean is positive and reasonable
    if (mean <= math::ZERO_DOUBLE || !std::isfinite(mean)) {
        reset(); // Fall back to default if computed mean is invalid
        return;
    }
    
    // Set MLE estimate: λ = 1/mean
    // For exponential distribution, this is the optimal estimator
    lambda_ = math::ONE / mean;
    
    // Validate the resulting lambda parameter
    if (!std::isfinite(lambda_) || lambda_ <= math::ZERO_DOUBLE) {
        reset(); // Fallback if lambda computation fails
        return;
    }
    
    cacheValid_ = false; // Invalidate cache since parameters changed
}

/**
 * Resets the distribution to default parameters (λ = 1.0).
 * This corresponds to the standard exponential distribution.
 */
void ExponentialDistribution::reset() noexcept {
    lambda_ = math::ONE;
    cacheValid_ = false; // Invalidate cache since parameters changed
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
