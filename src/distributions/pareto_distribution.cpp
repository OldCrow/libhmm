#include "libhmm/distributions/pareto_distribution.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cfloat>

using namespace libhmm::constants;

namespace libhmm
{

/**
 * Computes the probability density function for the Pareto distribution.
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
double ParetoDistribution::getProbability(double x) {
    // Pareto distribution has support [x_m, ∞)
    if (std::isnan(x) || std::isinf(x) || x < xm_) {
        return math::ZERO_DOUBLE;
    }
    
    if (x == xm_) {
        return math::ZERO_DOUBLE;
    }
    
    // Ensure cache is valid
    if (!cacheValid_) {
        updateCache();
    }
    
    double p = 0.0;
    if (x > xm_ + precision::LIMIT_TOLERANCE) {
        p = CDF(x) - CDF(x - precision::LIMIT_TOLERANCE);
    } else if (x > xm_ && x < xm_ + precision::LIMIT_TOLERANCE) {
        // For values very close to x_m, use the PDF scaled by tolerance
        // to avoid numerical issues with CDF differences
        p = precision::LIMIT_TOLERANCE * (k_ * std::pow(xm_, k_)) / std::pow(x, kPlus1_);
    }
    
    // Ensure numerical stability
    if (std::isnan(p) || p < math::ZERO_DOUBLE) {
        p = precision::ZERO;
    }
    
    assert(p <= 1.0);
    return p;
}

/*
 * Evaluates the CDF for the Pareto distribution at x.  The CDF is defined as
 *
 *   F(x) = 1 - (xm/x)^k
 */
double ParetoDistribution::CDF(double x) noexcept {
    const double y = math::ONE - std::pow(xm_ / x, k_);
    assert(y >= math::ZERO_DOUBLE);
    return y;
}

/**
 * Fits the distribution parameters to the given data using maximum likelihood estimation.
 * 
 * For Pareto distribution, the MLE estimators are:
 * x_m = min(x_i) for all i
 * k = n / Σ(ln(x_i) - ln(x_m)) for i = 1 to n
 * 
 * @param values Vector of observed data
 */                   
void ParetoDistribution::fit(const std::vector<Observation>& values) {
    // Handle edge cases: empty data or single data point
    if (values.empty()) {
        reset();
        return;
    }
    
    if (values.size() == 1) {
        reset(); // MLE is not well-defined for single point
        return;
    }

    // Validate that all values are positive (required for Pareto distribution)
    auto minValue = *std::min_element(values.begin(), values.end());
    if (minValue <= math::ZERO_DOUBLE) {
        reset(); // Fall back to default if data contains non-positive values
        return;
    }
    
    // Set scale parameter to minimum value
    xm_ = minValue;
    
    // Calculate sum of log differences for shape parameter
    double sum = math::ZERO_DOUBLE;
    for (const auto& val : values) {
        if (val > math::ZERO_DOUBLE) {
            sum += std::log(val) - std::log(xm_);
        }
    }
    
    // Validate that sum is positive for numerical stability
    if (sum <= math::ZERO_DOUBLE) {
        reset(); // Fall back to default
        return;
    }
    
    // Calculate MLE estimate for shape parameter
    k_ = static_cast<double>(values.size()) / sum;
    
    // Validate computed shape parameter
    if (std::isnan(k_) || std::isinf(k_) || k_ <= math::ZERO_DOUBLE) {
        reset(); // Fall back to default
        return;
    }
    
    cacheValid_ = false; // Invalidate cache since parameters changed
}

/**
 * Resets the distribution to default parameters (k = 1.0, x_m = 1.0).
 * This corresponds to a standard Pareto distribution.
 */
void ParetoDistribution::reset() noexcept {
    k_ = math::ONE;
    xm_ = math::ONE;
    cacheValid_ = false; // Invalidate cache since parameters changed
}

std::string ParetoDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Pareto Distribution:\n";
    oss << "      k (shape parameter) = " << k_ << "\n";
    oss << "      x_m (scale parameter) = " << xm_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

std::ostream& operator<<( std::ostream& os, 
        const libhmm::ParetoDistribution& distribution ){
    os << "Pareto Distribution: " << std::endl;
    os << "    k = " << distribution.getK( ) << std::endl;
    os << "    xm = " << distribution.getXm( ) << std::endl;
    os << std::endl;
    
    return os;
}

std::istream& operator>>( std::istream& is,
        libhmm::ParetoDistribution& distribution ){
    std::string token, k_str, xm_str;
    
    try {
        is >> token; //" k"
        is >> token; // "="
        is >> k_str;
        double k = std::stod(k_str);

        is >> token; // " xm"
        is >> token; // " ="
        is >> xm_str;
        double xm = std::stod(xm_str);
        
        if (is.good()) {
            distribution.setK(k);
            distribution.setXm(xm);
        }
        
    } catch (const std::exception& e) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }

    return is;
}


}
