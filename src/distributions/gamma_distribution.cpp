#include "libhmm/distributions/gamma_distribution.h"
#include <iostream>
#include <cfloat>

namespace libhmm
{

/**
 * Computes the probability density function for the Gamma distribution.
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
double GammaDistribution::getProbability(double x) {
    // Gamma distribution has support [0, ∞)
    if (std::isnan(x) || std::isinf(x) || x < 0.0) {
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
        // For very small positive values, use a small approximation
        // to avoid numerical issues with CDF differences
        p = LIMIT_TOLERANCE * std::pow(x / theta_, kMinus1_) * std::exp(-x / theta_) / 
            (std::exp(logGammaK_) * std::pow(theta_, k_));
    }
    
    // Ensure numerical stability
    if (std::isnan(p) || p < 0.0) {
        p = ZERO;
    }
    
    assert(p <= 1.0);
    return p;
}

/*
 * Returns the value of the CDF of the gamma distribution.
 *
 * The CDF is given as 
 *
 *          ligamma( k, x / theta )
 *   F(x) = -----------------------
 *                gamma( k )
 *
 * We have P( a, x ) and loggamma( a ).
 */
double GammaDistribution::CDF(double x) noexcept {
    assert(x >= 0);
    if(x == 0) return 0;

    double i = gammap(k_, x / theta_);
    if(std::isnan(i) || i < ZERO) i = ZERO;
    
    double j = loggamma(k_);
    if(std::isnan(j) || j < ZERO) j = ZERO;
    
    double y = std::exp(std::log(i) - 2 * j);
    if(std::isnan(y) || y < ZERO) y = ZERO; 
    
    assert(y <= 1.0);
    return y;
}

/*
 * Returns the value of the LOWER INCOMPLETE gamma function given a and x.
 */
double GammaDistribution::ligamma(double a, double x) noexcept {
    return std::exp(std::log(gammap(a, x)) + loggamma(a));
}


/*
 * Sets k and theta such that the resulting Gamma distribution fits the data.
 *
 * Wikipedia states that there is no closed form value of k, but we can
 * approximate to 1.5% if we use the value
 *
 *   s = ln( sum( x_i, i = 1..N ) / N ) - sum( ln( x_i ), i = 1..N ) / N
 *
 * where x_i is a data  point to use in setting the PDF and N is the total
 * number of values of x.  k can then be computed as 
 *
 *   k ~= 3 - s + sqrt( (s - 3)^2 + 24s )
 *        ---------------------------------
 *                        12s
 * 
 * Using k, theta is defined as
 *
 *   theta = sum( x_i, i = 1..N )
 *           --------------------
 *                   kN
 * 
 * Note that the presence of a zero in the list of values will tend to screw
 * with things because log( 0 ) is undefined (it approaches -infinity).  I've
 * written an approximation that if there is a zero, to add libhmm::ZERO to the
 * sum and logsum values, but that's something that should really be solved with
 * some more intensive numerical methods.  Perhaps the logsum value should be
 * increased by -REALLY_BIG_NUMBER.                  
 */                   
/**
 * Fits the distribution parameters to the given data using maximum likelihood estimation
 * approximation for the Gamma distribution.
 * 
 * Uses the method described in literature with approximation:
 * s = ln(sample_mean) - mean(ln(x_i))
 * k ≈ (3 - s + sqrt((s-3)² + 24s)) / (12s)
 * θ = sample_mean / k
 * 
 * This provides approximately 1.5% accuracy for the shape parameter.
 * 
 * @param values Vector of observed data points
 */
void GammaDistribution::fit(const std::vector<Observation>& values) {
    const auto N = values.size();

    // Handle edge cases: empty data or single data point
    if (N == 0) {
        // Legacy behavior: set parameters to ZERO for empty clusters
        k_ = ZERO;
        theta_ = ZERO;
        cacheValid_ = false;
        return;
    }
    
    if (N == 1) {
        reset(); // MLE is not well-defined for single point
        return;
    }

    double sum = 0.0;
    double logsum = 0.0;
    bool hasValidData = false;

    for (const auto& val : values) {
        if (val > 0.0) { // Only positive values are valid for Gamma distribution
            sum += val;
            logsum += std::log(val);
            hasValidData = true;
        }
        // Skip zero or negative values as they're not in the support of Gamma distribution
    }
    
    if (!hasValidData) {
        reset(); // Fall back to default if no valid data
        return;
    }

    const double s = std::log(sum / N) - logsum / N;
    
    // Validate that s is positive for numerical stability
    if (s <= 0.0 || std::isnan(s) || std::isinf(s)) {
        reset(); // Fall back to default
        return;
    }

    // Use the approximation formula for shape parameter
    k_ = (3.0 - s + std::sqrt(std::pow((s - 3.0), 2.0) + 24.0 * s)) / (12.0 * s);
    
    // Validate computed shape parameter
    if (std::isnan(k_) || std::isinf(k_) || k_ <= 0.0) {
        reset(); // Fall back to default
        return;
    }

    // Compute scale parameter
    theta_ = sum / (k_ * N);
    
    // Validate computed scale parameter
    if (std::isnan(theta_) || std::isinf(theta_) || theta_ <= 0.0) {
        reset(); // Fall back to default
        return;
    }
    
    cacheValid_ = false; // Invalidate cache since parameters changed
}

/**
 * Resets the distribution to default parameters (k = 1.0, θ = 1.0).
 * This corresponds to the standard exponential distribution.
 */
void GammaDistribution::reset() noexcept {
    k_ = 1.0;
    theta_ = 1.0;
    cacheValid_ = false; // Invalidate cache since parameters changed
}

std::string GammaDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Gamma Distribution:\n";
    oss << "      k (shape parameter) = " << k_ << "\n";
    oss << "      θ (scale parameter) = " << theta_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

std::ostream& operator<<( std::ostream& os, 
        const libhmm::GammaDistribution& distribution ){

    os << "Gamma Distribution: " << std::endl;
    os << "    k = " << distribution.getK( ) << std::endl;
    os << "    theta = " << distribution.getTheta( ) << std::endl;

    return os;
}


}//namespace
