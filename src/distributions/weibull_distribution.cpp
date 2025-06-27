#include "libhmm/distributions/weibull_distribution.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

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
        return (k_ == math::ONE) ? (k_ / lambda_) : math::ZERO_DOUBLE;
    }
    
    // Compute PDF: f(x) = (k/λ) * (x/λ)^(k-1) * exp(-(x/λ)^k)
    // In log space: log(f(x)) = log(k) - log(λ) + (k-1)*log(x/λ) - (x/λ)^k
    
    double x_over_lambda = value / lambda_;
    double logPdf = logK_ - logLambda_ + 
                    (k_ - math::ONE) * (std::log(value) - logLambda_) - 
                    std::pow(x_over_lambda, k_);
    
    return std::exp(logPdf);
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
    
    // Compute sample statistics
    double sum = std::accumulate(values.begin(), values.end(), math::ZERO_DOUBLE);
    double mean = sum / values.size();
    
    double sumSq = std::accumulate(values.begin(), values.end(), math::ZERO_DOUBLE,
        [mean](double acc, double val) {
            double diff = val - mean;
            return acc + diff * diff;
        });
    double variance = sumSq / (values.size() - 1);
    
    // Avoid degenerate cases
    if (variance <= math::ZERO_DOUBLE || mean <= math::ZERO_DOUBLE) {
        reset();
        return;
    }
    
    // Method of moments approximation for Weibull parameters
    // We use the coefficient of variation to estimate k
    double cv = std::sqrt(variance) / mean;  // coefficient of variation
    
    // For Weibull, CV = sqrt(Γ(1+2/k)/Γ(1+1/k)² - 1)
    // We approximate k using an empirical relationship
    double k_est;
    if (cv < 0.2) {
        // High k (> 5), shape approaches normal
        k_est = math::ONE / (cv * cv * 6.0);
    } else if (cv < math::ONE) {
        // Medium k (1-5), use approximation
        k_est = std::pow(1.2 / cv, 1.086);
    } else {
        // Low k (< 1), use simpler approximation
        k_est = math::ONE / cv;
    }
    
    // Ensure k is reasonable
    k_est = std::max(0.1, std::min(k_est, 10.0));
    
    // Estimate λ using the relationship: mean = λ * Γ(1 + 1/k)
    double gamma_term = std::exp(loggamma(math::ONE + math::ONE/k_est));
    double lambda_est = mean / gamma_term;
    
    // Ensure λ is positive and reasonable
    if (lambda_est > math::ZERO_DOUBLE && lambda_est < 1e6) {
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

double WeibullDistribution::CDF(double x) noexcept {
    if (x <= math::ZERO_DOUBLE) return math::ZERO_DOUBLE;
    
    // CDF(x) = 1 - exp(-(x/λ)^k)
    double x_over_lambda = x / lambda_;
    return math::ONE - std::exp(-std::pow(x_over_lambda, k_));
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
