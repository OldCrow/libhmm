#include "libhmm/distributions/weibull_distribution.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <algorithm>   // For std::max, std::min (exists in common.h, included for clarity)
#include <numeric>     // For std::accumulate (not in common.h)

using namespace libhmm::constants;

namespace libhmm {

double WeibullDistribution::getProbability(double value) {
    // Weibull distribution is only defined for x ≥ 0
    if (value < math::ZERO_DOUBLE || std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }
    
    // Update cache if needed
    if (!cacheValid_) {
        updateCache();
    }
    
    // Handle boundary case
    if (value == math::ZERO_DOUBLE) {
        return (k_ == math::ONE) ? kOverLambda_ : math::ZERO_DOUBLE;
    }
    
    // Use optimized log-space calculation then exponentiate
    // This avoids numerical issues with very small/large values
    const double logPdf = getLogProbability(value);
    if (logPdf == -std::numeric_limits<double>::infinity()) {
        return math::ZERO_DOUBLE;
    }
    
    return std::exp(logPdf);
}

double WeibullDistribution::getLogProbability(double value) const noexcept {
    // Weibull distribution is only defined for x ≥ 0
    if (value < math::ZERO_DOUBLE || std::isnan(value) || std::isinf(value)) {
        return -std::numeric_limits<double>::infinity();
    }
    
    // Update cache if needed
    if (!cacheValid_) {
        updateCache();
    }
    
    // Handle boundary case
    if (value == math::ZERO_DOUBLE) {
        if (k_ == math::ONE) {
            return logK_ - logLambda_;  // log(k/λ)
        } else {
            return -std::numeric_limits<double>::infinity();
        }
    }
    
    // Optimized log PDF computation using cached values:
    // log(f(x)) = log(k) - log(λ) + (k-1)*log(x) - (k-1)*log(λ) - (x/λ)^k
    //           = log(k) - k*log(λ) + (k-1)*log(x) - (x*invλ)^k
    
    const double logX = std::log(value);
    const double xTimesInvLambda = value * invLambda_;  // Use cached reciprocal
    
    // Use efficient power calculation for common k values
    double powerTerm;
    if (k_ == math::ONE) {
        powerTerm = xTimesInvLambda;  // Linear case
    } else if (k_ == math::TWO) {
        powerTerm = xTimesInvLambda * xTimesInvLambda;  // Quadratic case (Rayleigh)
    } else {
        powerTerm = std::pow(xTimesInvLambda, k_);  // General case
    }
    
    const double logPdf = logK_ - k_ * logLambda_ + kMinus1_ * logX - powerTerm;
    
    return logPdf;
}

void WeibullDistribution::fit(const std::vector<Observation>& values) {
    if (values.empty()) {
        reset();
        return;
    }
    
    // Validate that all values are non-negative
    for (const auto& val : values) {
        if (val < math::ZERO_DOUBLE || std::isnan(val) || std::isinf(val)) {
            throw std::invalid_argument("Weibull distribution fitting requires all values to be non-negative");
        }
    }
    
    // For single data point, set to default (insufficient data)
    if (values.size() == 1) {
        reset();
        return;
    }
    
    // Use Welford's algorithm for numerically stable mean and variance calculation
    // This is more cache-friendly and numerically stable than two-pass methods
    const auto n = static_cast<double>(values.size());
    double mean = math::ZERO_DOUBLE;
    double m2 = math::ZERO_DOUBLE;  // Sum of squared differences from current mean
    
    std::size_t count = 0;
    for (const auto& val : values) {
        ++count;
        const double delta = val - mean;
        mean += delta / static_cast<double>(count);
        const double delta2 = val - mean;
        m2 += delta * delta2;
    }
    
    // Sample variance uses Bessel's correction (N-1)
    const double sampleVariance = m2 / (n - math::ONE);
    
    // Avoid degenerate cases
    if (sampleVariance <= precision::ZERO || mean <= precision::ZERO) {
        reset();
        return;
    }
    
    // Method of moments approximation for Weibull parameters
    // We use the coefficient of variation to estimate k
    const double cv = std::sqrt(sampleVariance) / mean;  // coefficient of variation
    
    // For Weibull, CV = sqrt(Γ(1+2/k)/Γ(1+1/k)² - 1)
    // We approximate k using an empirical relationship
    double k_est;
    if (cv < 0.2) {
        // High k (> 5), shape approaches normal
        k_est = math::ONE / (cv * cv * math::TEN * 0.6);  // Optimized: 6.0 = 10 * 0.6
    } else if (cv < math::ONE) {
        // Medium k (1-5), use approximation
        k_est = std::pow(1.2 / cv, 1.086);
    } else {
        // Low k (< 1), use simpler approximation
        k_est = math::ONE / cv;
    }
    
    // Ensure k is reasonable using standardized bounds
    k_est = std::max(thresholds::MIN_DISTRIBUTION_PARAMETER, 
                     std::min(k_est, thresholds::MAX_DISTRIBUTION_PARAMETER));
    
    // Estimate λ using the relationship: mean = λ * Γ(1 + 1/k)
    const double gamma_term = std::exp(std::lgamma(math::ONE + math::ONE/k_est));
    const double lambda_est = mean / gamma_term;
    
    // Ensure λ is positive and reasonable using standardized bounds
    if (lambda_est > precision::ZERO && lambda_est < thresholds::MAX_DISTRIBUTION_PARAMETER) {
        k_ = k_est;
        lambda_ = lambda_est;
        cacheValid_ = false;
    } else {
        reset();
    }
}

void WeibullDistribution::reset() noexcept {
    k_ = math::ONE;
    lambda_ = math::ONE;
    cacheValid_ = false;
}

std::string WeibullDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Weibull Distribution:\n";
    oss << "      k (shape) = " << k_ << "\n";
    oss << "      λ (scale) = " << lambda_ << "\n";
    return oss.str();
}

double WeibullDistribution::CDF(double x) const noexcept {
    if (x <= math::ZERO_DOUBLE) return math::ZERO_DOUBLE;
    
    // Update cache if needed
    if (!cacheValid_) {
        updateCache();
    }
    
    // CDF(x) = 1 - exp(-(x/λ)^k) = 1 - exp(-(x*invλ)^k)
    const double xTimesInvLambda = x * invLambda_;  // Use cached reciprocal
    
    // Use efficient power calculation for common k values
    double powerTerm;
    if (k_ == math::ONE) {
        powerTerm = xTimesInvLambda;  // Linear case (Exponential)
    } else if (k_ == math::TWO) {
        powerTerm = xTimesInvLambda * xTimesInvLambda;  // Quadratic case (Rayleigh)
    } else {
        powerTerm = std::pow(xTimesInvLambda, k_);  // General case
    }
    
    return math::ONE - std::exp(-powerTerm);
}

bool WeibullDistribution::operator==(const WeibullDistribution& other) const noexcept {
    // Use tolerance for floating-point comparison
    const double tolerance = precision::ZERO;
    return std::abs(k_ - other.k_) < tolerance && 
           std::abs(lambda_ - other.lambda_) < tolerance;
}

std::ostream& operator<<(std::ostream& os, const WeibullDistribution& distribution) {
    return os << distribution.toString();
}

std::istream& operator>>(std::istream& is, WeibullDistribution& distribution) {
    std::string token;
    double k, lambda;
    
    try {
        // Expected format: "Weibull Distribution: k (shape) = <value> λ (scale) = <value>"
        std::string k_str, lambda_str;
        is >> token >> token;  // "Weibull" "Distribution:"
        is >> token >> token >> token >> k_str;      // "k" "(shape)" "=" <k_str>
        is >> token >> token >> token >> lambda_str; // "λ" "(scale)" "=" <lambda_str>
        k = std::stod(k_str);
        lambda = std::stod(lambda_str);
        
        if (is.good()) {
            distribution = WeibullDistribution(k, lambda);
        }
        
    } catch (const std::exception& e) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }
    
    return is;
}

} // namespace libhmm
