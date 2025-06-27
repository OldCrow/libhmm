#include "libhmm/distributions/beta_distribution.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <limits>

using namespace libhmm::constants;

namespace libhmm {

double BetaDistribution::getProbability(double value) {
    // Beta distribution is only defined on [0,1]
    if (value < math::ZERO_DOUBLE || value > math::ONE || std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }
    
    // Update cache if needed
    if (!cacheValid_) {
        updateCache();
    }
    
    // Handle boundary cases - use cached invBeta_ for efficiency
    if (value == math::ZERO_DOUBLE) {
        return (alpha_ == math::ONE) ? invBeta_ : math::ZERO_DOUBLE;
    }
    if (value == math::ONE) {
        return (beta_ == math::ONE) ? invBeta_ : math::ZERO_DOUBLE;
    }
    
    // For efficiency, use direct computation when both exponents are integers
    // and small, otherwise use optimized log-space computation
    if (alphaMinus1_ == std::floor(alphaMinus1_) && betaMinus1_ == std::floor(betaMinus1_) &&
        alphaMinus1_ >= precision::ZERO && betaMinus1_ >= precision::ZERO && 
        alphaMinus1_ <= math::FOUR && betaMinus1_ <= math::FOUR) {
        
        // Direct computation for small integer exponents (faster than log/exp)
        double result = invBeta_;
        
        // Use efficient integer power computation
        int alphaExp = static_cast<int>(alphaMinus1_);
        int betaExp = static_cast<int>(betaMinus1_);
        
        // Compute x^(α-1) efficiently
        double xPower = math::ONE;
        for (int i = 0; i < alphaExp; ++i) {
            xPower *= value;
        }
        
        // Compute (1-x)^(β-1) efficiently
        double oneMinusX = math::ONE - value;
        double oneMinusXPower = math::ONE;
        for (int i = 0; i < betaExp; ++i) {
            oneMinusXPower *= oneMinusX;
        }
        
        return result * xPower * oneMinusXPower;
    } else {
        // Use optimized log-space computation for general case
        // f(x) = x^(α-1) * (1-x)^(β-1) / B(α,β)
        // Use cached values to avoid repeated calculations
        return invBeta_ * std::pow(value, alphaMinus1_) * std::pow(math::ONE - value, betaMinus1_);
    }
}

/**
 * Computes the logarithm of the probability density function for numerical stability.
 * 
 * For Beta distribution: log(f(x)) = (α-1)log(x) + (β-1)log(1-x) - log(B(α,β))
 * 
 * @param value The value at which to evaluate the log-PDF (should be in [0,1])
 * @return Natural logarithm of the probability density, or -∞ for invalid values
 */
double BetaDistribution::getLogProbability(double value) const noexcept {
    // Beta distribution is only defined on [0,1]
    if (value < math::ZERO_DOUBLE || value > math::ONE || std::isnan(value) || std::isinf(value)) {
        return -std::numeric_limits<double>::infinity();
    }
    
    // Update cache if needed
    if (!cacheValid_) {
        updateCache();
    }
    
    // Handle boundary cases carefully
    if (value == 0.0) {
        if (alpha_ == 1.0) {
            // log(f(0)) = -log(B(1,β)) = -log(Γ(β)) = -logBeta_
            return -logBeta_;
        } else if (alpha_ > 1.0) {
            // f(0) = 0 since x^(α-1) → 0 as x → 0 for α > 1
            return -std::numeric_limits<double>::infinity();
        } else {
            // α < 1: f(0) → +∞, which should be avoided
            return -std::numeric_limits<double>::infinity();
        }
    }
    
    if (value == 1.0) {
        if (beta_ == 1.0) {
            // log(f(1)) = -log(B(α,1)) = -log(Γ(α)) = -logBeta_
            return -logBeta_;
        } else if (beta_ > 1.0) {
            // f(1) = 0 since (1-x)^(β-1) → 0 as x → 1 for β > 1
            return -std::numeric_limits<double>::infinity();
        } else {
            // β < 1: f(1) → +∞, which should be avoided
            return -std::numeric_limits<double>::infinity();
        }
    }
    
    // For interior points: log(f(x)) = (α-1)log(x) + (β-1)log(1-x) - log(B(α,β))
    // Use cached values for maximum efficiency
    return alphaMinus1_ * std::log(value) + 
           betaMinus1_ * std::log(1.0 - value) - 
           logBeta_;
}

void BetaDistribution::fit(const std::vector<Observation>& values) {
    if (values.empty()) {
        reset();
        return;
    }
    
    // Validate that all values are in [0,1]
    for (const auto& val : values) {
        if (val < 0.0 || val > 1.0 || std::isnan(val) || std::isinf(val)) {
            throw std::invalid_argument("Beta distribution fitting requires all values to be in [0,1]");
        }
    }
    
    // For single data point, set to default (insufficient data)
    if (values.size() == 1) {
        reset();
        return;
    }
    
    // Use single-pass Welford's algorithm for numerical stability and performance
    double mean = 0.0;
    double M2 = 0.0;  // Sum of squared differences from current mean
    double n = 0.0;
    
    for (const auto& val : values) {
        n += 1.0;
        double delta = val - mean;
        mean += delta / n;
        double delta2 = val - mean;
        M2 += delta * delta2;
    }
    
    double variance = (n > 1.0) ? M2 / (n - 1.0) : 0.0;
    
    // Method of moments estimators
    // α = μ * (μ(1-μ)/σ² - 1)
    // β = (1-μ) * (μ(1-μ)/σ² - 1)
    
    // Avoid division by zero and ensure valid parameters
    if (variance <= 0.0 || mean <= 0.0 || mean >= 1.0) {
        reset();
        return;
    }
    
    double factor = mean * (1.0 - mean) / variance - 1.0;
    if (factor <= 0.0) {
        reset();
        return;
    }
    
    double newAlpha = mean * factor;
    double newBeta = (1.0 - mean) * factor;
    
    // Ensure parameters are positive and reasonable
    if (newAlpha > 0.0 && newBeta > 0.0 && 
        newAlpha < 1000.0 && newBeta < 1000.0) {
        alpha_ = newAlpha;
        beta_ = newBeta;
        cacheValid_ = false;
    } else {
        reset();
    }
}

void BetaDistribution::reset() noexcept {
    alpha_ = 1.0;
    beta_ = 1.0;
    cacheValid_ = false;
}

std::string BetaDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Beta Distribution:\n";
    oss << "      α (alpha) = " << alpha_ << "\n";
    oss << "      β (beta) = " << beta_ << "\n";
    return oss.str();
}

double BetaDistribution::CDF(double x) noexcept {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    
    // Use the incomplete Beta function
    // CDF(x) = I_x(α, β) = B(x; α, β) / B(α, β)
    // where B(x; α, β) is the incomplete Beta function
    
    // For implementation, we use the relationship with the regularized 
    // incomplete Gamma function that's already available in the base class
    return gammap(alpha_, -std::log(1.0 - x) * alpha_);
}

std::ostream& operator<<(std::ostream& os, const BetaDistribution& distribution) {
    return os << distribution.toString();
}

std::istream& operator>>(std::istream& is, BetaDistribution& distribution) {
    std::string token;
    double alpha = 1.0;
    double beta = 1.0;
    
    try {
        // Expected format: "Beta Distribution: α (alpha) = <value> β (beta) = <value>"
        is >> token >> token;  // "Beta" "Distribution:"
        is >> token >> token >> token >> token;  // "α" "(alpha)" "=" <alpha_str>
        alpha = std::stod(token);
        is >> token >> token >> token >> token;   // "β" "(beta)" "=" <beta_str>
        beta = std::stod(token);
        
        if (is.good()) {
            distribution = BetaDistribution(alpha, beta);
        }
        
    } catch (const std::exception& e) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }
    
    return is;
}
} // namespace libhmm
