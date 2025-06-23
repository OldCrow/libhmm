#include "libhmm/distributions/beta_distribution.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

namespace libhmm {

double BetaDistribution::getProbability(double value) {
    // Beta distribution is only defined on [0,1]
    if (value < 0.0 || value > 1.0 || std::isnan(value) || std::isinf(value)) {
        return 0.0;
    }
    
    // Update cache if needed
    if (!cacheValid_) {
        updateCache();
    }
    
    // Handle boundary cases
    if (value == 0.0) {
        return (alpha_ == 1.0) ? std::exp(-logBeta_) : 0.0;
    }
    if (value == 1.0) {
        return (beta_ == 1.0) ? std::exp(-logBeta_) : 0.0;
    }
    
    // Compute PDF: f(x) = x^(α-1) * (1-x)^(β-1) / B(α,β)
    // In log space: log(f(x)) = (α-1)log(x) + (β-1)log(1-x) - log(B(α,β))
    double logPdf = (alpha_ - 1.0) * std::log(value) + 
                    (beta_ - 1.0) * std::log(1.0 - value) - 
                    logBeta_;
    
    return std::exp(logPdf);
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
    
    // Compute sample mean and variance
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double mean = sum / values.size();
    
    double sumSq = std::accumulate(values.begin(), values.end(), 0.0,
        [mean](double acc, double val) {
            double diff = val - mean;
            return acc + diff * diff;
        });
    double variance = sumSq / (values.size() - 1);
    
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
    double alpha, beta;
    
    // Expected format: "Beta Distribution: α (alpha) = <value> β (beta) = <value>"
    is >> token >> token;  // "Beta" "Distribution:"
    is >> token >> token >> token >> alpha;  // "α" "(alpha)" "=" <value>
    is >> token >> token >> token >> beta;   // "β" "(beta)" "=" <value>
    
    distribution = BetaDistribution(alpha, beta);
    return is;
}

} // namespace libhmm
