#include "libhmm/distributions/exponential_distribution.h"
#include <iostream>
#include <numeric>

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
double ExponentialDistribution::getProbability(double x) {
    // Exponential distribution has support [0, ∞)
    if (x < 0) {
        return 0.0;
    }
    
    if (x == 0.0) {
        return 0.0;
    }
    
    // Ensure cache is valid
    if (!cacheValid_) {
        updateCache();
    }
    
    double p = 0.0;
    if (x > LIMIT_TOLERANCE) {
        p = CDF(x) - CDF(x - LIMIT_TOLERANCE);
    } else if (x > 0 && x < LIMIT_TOLERANCE) {
        // For very small positive values, use the PDF scaled by tolerance
        // This provides better numerical stability than CDF differences
        p = lambda_ * LIMIT_TOLERANCE * std::exp(-lambda_ * x);
    }

    // Ensure numerical stability
    if (std::isnan(p) || p < 0.0) {
        p = ZERO;
    }
    
    assert(p <= 1.0);
    return p;
}

/*
 * Evaluates the CDF for the Normal distribution at x.  The CDF is defined as
 *
 *   F(x) = 1 - exp( -lambda * x )
 */
double ExponentialDistribution::CDF(double x) noexcept {
    const double y = 1 - std::exp(-lambda_ * x);
    assert(y >= 0);
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

    // Calculate sample mean
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const double mean = sum / static_cast<double>(values.size());
    
    // Validate that mean is positive (required for exponential distribution)
    if (mean <= 0.0) {
        reset(); // Fall back to default if data is invalid
        return;
    }
    
    // Set MLE estimate: λ = 1/mean
    lambda_ = 1.0 / mean;
    cacheValid_ = false; // Invalidate cache since parameters changed
}

/**
 * Resets the distribution to default parameters (λ = 1.0).
 * This corresponds to the standard exponential distribution.
 */
void ExponentialDistribution::reset() noexcept {
    lambda_ = 1.0;
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
    std::string s, t;
    is >> s; //" Rate"
    is >> s; // "parameter"
    is >> s; // "="
    is >> t;
    distribution.setLambda(std::stod(t));

    return is;
}


}
