#include "libhmm/distributions/uniform_distribution.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

namespace libhmm {

void UniformDistribution::validateParameters(double a, double b) const {
    // Check for NaN or infinity
    if (std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b)) {
        throw std::invalid_argument("Uniform distribution parameters cannot be NaN or infinite");
    }
    
    // Check that a < b
    if (a >= b) {
        throw std::invalid_argument("Uniform distribution requires a < b (lower bound must be less than upper bound)");
    }
}

UniformDistribution::UniformDistribution() : a_(0.0), b_(1.0) {
    // Default: standard uniform distribution on [0, 1]
}

UniformDistribution::UniformDistribution(double a, double b) : a_(a), b_(b) {
    validateParameters(a, b);
}

double UniformDistribution::getProbability(Observation val) {
    // Handle invalid inputs
    if (std::isnan(val) || std::isinf(val)) {
        return 0.0;
    }
    
    // Uniform PDF: f(x) = 1/(b-a) for a ≤ x ≤ b, 0 otherwise
    if (val >= a_ && val <= b_) {
        return 1.0 / (b_ - a_);
    }
    
    return 0.0;
}

void UniformDistribution::fit(const std::vector<Observation>& data) {
    if (data.empty()) {
        // Reset to default if no data
        reset();
        return;
    }
    
    if (data.size() < 2) {
        // Need at least 2 points to determine bounds
        reset();
        return;
    }
    
    // Validate data
    for (const auto& obs : data) {
        if (std::isnan(obs) || std::isinf(obs)) {
            throw std::invalid_argument("Uniform distribution fitting: data contains NaN or infinite values");
        }
    }
    
    // For uniform distribution, the most straightforward approach is to use
    // the sample minimum and maximum as estimates for a and b
    auto minmax = std::minmax_element(data.begin(), data.end());
    double min_val = *minmax.first;
    double max_val = *minmax.second;
    
    // Add small padding to ensure all data points are within [a, b]
    // This accounts for the fact that sample min/max are estimates of true bounds
    double range = max_val - min_val;
    if (range == 0.0) {
        // All values are the same - create a small interval around the value
        double padding = std::max(std::abs(min_val) * 1e-6, 1e-6);
        a_ = min_val - padding;
        b_ = max_val + padding;
    } else {
        // Add 5% padding on each side
        double padding = range * 0.05;
        a_ = min_val - padding;
        b_ = max_val + padding;
    }
    
    // Ensure we still have valid parameters
    validateParameters(a_, b_);
}

void UniformDistribution::reset() noexcept {
    a_ = 0.0;
    b_ = 1.0;
}

std::string UniformDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Uniform Distribution:\n";
    oss << "      a (lower bound) = " << a_ << "\n";
    oss << "      b (upper bound) = " << b_ << "\n";
    return oss.str();
}

void UniformDistribution::setA(double a) {
    validateParameters(a, b_);
    a_ = a;
}

void UniformDistribution::setB(double b) {
    validateParameters(a_, b);
    b_ = b;
}

void UniformDistribution::setParameters(double a, double b) {
    validateParameters(a, b);
    a_ = a;
    b_ = b;
}

double UniformDistribution::getMean() const {
    // Mean of uniform distribution: μ = (a + b) / 2
    return (a_ + b_) / 2.0;
}

double UniformDistribution::getVariance() const {
    // Variance of uniform distribution: σ² = (b - a)² / 12
    double range = b_ - a_;
    return (range * range) / 12.0;
}

double UniformDistribution::getStandardDeviation() const {
    // Standard deviation: σ = (b - a) / √12
    return std::sqrt(getVariance());
}

bool UniformDistribution::isApproximatelyEqual(const UniformDistribution& other, double tolerance) const {
    return std::abs(a_ - other.a_) < tolerance && 
           std::abs(b_ - other.b_) < tolerance;
}

} // namespace libhmm
