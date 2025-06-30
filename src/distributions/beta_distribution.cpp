#include "libhmm/distributions/beta_distribution.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <algorithm>   // For std::for_each (exists in common.h, included for clarity)
#include <numeric>     // For std::accumulate (not in common.h)
#include <limits>      // For std::numeric_limits (exists in common.h via <climits>)

// SIMD support now provided via simd_platform.h if needed
// #include "libhmm/performance/simd_platform.h"  // Uncomment if using SIMD operations

using namespace libhmm::constants;

namespace libhmm {

/**
 * Computes the probability density function for the Beta distribution.
 * 
 * @param value The value at which to evaluate the PDF (should be in [0,1])
 * @return Probability density, or 0.0 if value is outside [0,1]
 */
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
        
        // Use efficient binary exponentiation for integer powers
        int alphaExp = static_cast<int>(alphaMinus1_);
        int betaExp = static_cast<int>(betaMinus1_);
        
        // Binary exponentiation for x^(α-1)
        auto fastPower = [](double base, int exp) -> double {
            if (exp == 0) return 1.0;
            if (exp == 1) return base;
            if (exp == 2) return base * base;
            if (exp == 3) return base * base * base;
            if (exp == 4) { double sq = base * base; return sq * sq; }
            
            // For larger powers, use binary exponentiation
            double result = 1.0;
            double currentPower = base;
            while (exp > 0) {
                if (exp & 1) result *= currentPower;
                currentPower *= currentPower;
                exp >>= 1;
            }
            return result;
        };
        
        double xPower = fastPower(value, alphaExp);
        double oneMinusXPower = fastPower(math::ONE - value, betaExp);
        
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


void BetaDistribution::getProbabilityBatch(const std::vector<double>& values, std::vector<double>& results) {
    if (!cacheValid_) {
        updateCache();
    }
    
    // Resize the results vector if needed
    if (results.size() != values.size()) {
        results.resize(values.size());
    }
    
    auto it = results.begin();
    for (const double& value : values) {
        *it++ = getProbability(value);
    }
}

void BetaDistribution::getLogProbabilityBatch(const std::vector<double>& values, std::vector<double>& results) const {
    if (!cacheValid_) {
        updateCache();
    }
    
    // Resize the results vector if needed
    if (results.size() != values.size()) {
        results.resize(values.size());
    }
    
    auto it = results.begin();
    for (const double& value : values) {
        *it++ = getLogProbability(value);
    }
}

void BetaDistribution::fit(const std::vector<Observation>& values) {
    if (values.empty()) {
        reset();
        return;
    }
    
    // Single-pass Welford's algorithm with validation
    double mean = math::ZERO_DOUBLE;
    double M2 = math::ZERO_DOUBLE;  // Sum of squared differences from current mean
    std::size_t validCount = 0;
    
    for (const auto& val : values) {
        // Validate in the loop to avoid extra pass
        if (val < math::ZERO_DOUBLE || val > math::ONE || std::isnan(val) || std::isinf(val)) {
            throw std::invalid_argument("Beta distribution fitting requires all values to be in [0,1]");
        }
        
        ++validCount;
        const double delta = val - mean;
        mean += delta / static_cast<double>(validCount);
        const double delta2 = val - mean;
        M2 += delta * delta2;
    }
    
    // For single data point, set to default (insufficient data)
    if (validCount < 2) {
        reset();
        return;
    }
    
    const double variance = M2 / static_cast<double>(validCount - 1);
    
    // Method of moments estimators with cached constants
    // α = μ * (μ(1-μ)/σ² - 1), β = (1-μ) * (μ(1-μ)/σ² - 1)
    
    // Avoid division by zero and ensure valid parameters
    if (variance <= precision::ZERO || mean <= precision::ZERO || mean >= math::ONE) {
        reset();
        return;
    }
    
    const double oneMinusMean = math::ONE - mean;
    const double factor = mean * oneMinusMean / variance - math::ONE;
    if (factor <= precision::ZERO) {
        reset();
        return;
    }
    
    const double newAlpha = mean * factor;
    const double newBeta = oneMinusMean * factor;
    
    // Ensure parameters are positive and reasonable using constants
    if (newAlpha > precision::ZERO && newBeta > precision::ZERO && 
        newAlpha < thresholds::MAX_DISTRIBUTION_PARAMETER && newBeta < thresholds::MAX_DISTRIBUTION_PARAMETER) {
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
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

double BetaDistribution::getCumulativeProbability(double value) const noexcept {
    // Handle boundary cases
    if (value <= 0.0) return 0.0;
    if (value >= 1.0) return 1.0;
    if (std::isnan(value) || std::isinf(value)) return 0.0;
    
    // Use the incomplete Beta function I_x(α, β)
    // CDF(x) = I_x(α, β) 
    return incompleteBeta(value, alpha_, beta_);
}

double BetaDistribution::incompleteBeta(double x, double a, double b) const noexcept {
    // Implementation of regularized incomplete beta function I_x(a,b)
    // Using continued fraction for better numerical stability
    
    // Handle edge cases
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    if (a <= 0.0 || b <= 0.0) return 0.0;
    
    // For better numerical stability, use symmetry relation if needed:
    // I_x(a,b) = 1 - I_{1-x}(b,a)
    bool use_symmetry = false;
    double result_x = x;
    double result_a = a;
    double result_b = b;
    
    if (x > (a + 1.0) / (a + b + 2.0)) {
        // Use symmetry for better convergence
        use_symmetry = true;
        result_x = 1.0 - x;
        result_a = b;
        result_b = a;
    }
    
    // Calculate log(B(a,b)) = log(Γ(a)) + log(Γ(b)) - log(Γ(a+b))
    const double logBetaAB = std::lgamma(result_a) + std::lgamma(result_b) - std::lgamma(result_a + result_b);
    
    // Calculate prefix: x^a * (1-x)^b / (a * B(a,b))
    const double logPrefix = result_a * std::log(result_x) + result_b * std::log(1.0 - result_x) - std::log(result_a) - logBetaAB;
    const double prefix = std::exp(logPrefix);
    
    // Continued fraction evaluation
    const int maxIter = 200;
    const double tolerance = 1e-12;
    
    double cf = 1.0;
    double c = 1.0;
    double d = 1.0 - (result_a + result_b) * result_x / (result_a + 1.0);
    if (std::abs(d) < 1e-30) d = 1e-30;
    d = 1.0 / d;
    cf = d;
    
    for (int m = 1; m <= maxIter; ++m) {
        // Even step (2m)
        double numerator = m * (result_b - m) * result_x / ((result_a + 2.0 * m - 1.0) * (result_a + 2.0 * m));
        d = 1.0 + numerator * d;
        if (std::abs(d) < 1e-30) d = 1e-30;
        c = 1.0 + numerator / c;
        if (std::abs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        cf *= d * c;
        
        // Odd step (2m+1)
        numerator = -(result_a + m) * (result_a + result_b + m) * result_x / ((result_a + 2.0 * m) * (result_a + 2.0 * m + 1.0));
        d = 1.0 + numerator * d;
        if (std::abs(d) < 1e-30) d = 1e-30;
        c = 1.0 + numerator / c;
        if (std::abs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        const double delta = d * c;
        cf *= delta;
        
        if (std::abs(delta - 1.0) < tolerance) break;
    }
    
    double result = prefix * cf;
    
    // Apply symmetry if used
    if (use_symmetry) {
        result = 1.0 - result;
    }
    
    // Ensure result is in valid range
    return std::max(0.0, std::min(1.0, result));
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
