#include "libhmm/distributions/uniform_distribution.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <algorithm>   // For std::minmax_element (exists in common.h, included for clarity)
#include <numeric>     // For std::accumulate (not in common.h)
#include <limits>      // For std::numeric_limits (exists in common.h via <climits>)

using namespace libhmm::constants;

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

UniformDistribution::UniformDistribution() : a_(math::ZERO_DOUBLE), b_(math::ONE), cache_valid_(false) {
    // Default: standard uniform distribution on [0, 1]
}

UniformDistribution::UniformDistribution(double a, double b) : a_(a), b_(b), cache_valid_(false) {
    validateParameters(a, b);
}

void UniformDistribution::updateCache() const {
    if (!cache_valid_) {
        cached_range_ = b_ - a_;
        cached_inv_range_ = math::ONE / cached_range_;
        cached_pdf_ = cached_inv_range_;
        cached_log_pdf_ = -std::log(cached_range_);
        cached_mean_ = (a_ + b_) * math::HALF;
        cached_variance_ = (cached_range_ * cached_range_) / (math::THREE * math::FOUR); // /12
        cached_std_dev_ = cached_range_ / std::sqrt(math::THREE * math::FOUR); // /√12
        cache_valid_ = true;
    }
}

double UniformDistribution::getProbability(Observation val) {
    // Handle invalid inputs
    if (std::isnan(val) || std::isinf(val)) {
        return math::ZERO_DOUBLE;
    }
    
    // Uniform PDF: f(x) = 1/(b-a) for a ≤ x ≤ b, 0 otherwise
    if (val >= a_ && val <= b_) {
        if (!cache_valid_) {
            updateCache();
        }
        return cached_pdf_;
    }
    
    return math::ZERO_DOUBLE;
}

double UniformDistribution::getLogProbability(Observation val) const noexcept {
    // Handle invalid inputs
    if (std::isnan(val) || std::isinf(val)) {
        return -std::numeric_limits<double>::infinity();
    }
    
    // Uniform log PDF: log(f(x)) = -log(b-a) for a ≤ x ≤ b, -∞ otherwise
    if (val >= a_ && val <= b_) {
        if (!cache_valid_) {
            updateCache();
        }
        return cached_log_pdf_;
    }
    
    return -std::numeric_limits<double>::infinity();
}

double UniformDistribution::CDF(double x) const {
    // Handle invalid inputs
    if (std::isnan(x) || std::isinf(x)) {
        return std::isnan(x) ? std::numeric_limits<double>::quiet_NaN() : 
               (x > math::ZERO_DOUBLE ? math::ONE : math::ZERO_DOUBLE);
    }
    
    // Uniform CDF: F(x) = 0 for x < a, (x-a)/(b-a) for a ≤ x ≤ b, 1 for x > b
    if (x < a_) {
        return math::ZERO_DOUBLE;
    } else if (x > b_) {
        return math::ONE;
    } else {
        return (x - a_) / (b_ - a_);
    }
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
    if (range == math::ZERO_DOUBLE) {
        // All values are the same - create a small interval around the value
        double padding = std::max(std::abs(min_val) * thresholds::MIN_DISTRIBUTION_PARAMETER, 
                                 thresholds::MIN_DISTRIBUTION_PARAMETER);
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
    
    // Invalidate cache since parameters changed
    cache_valid_ = false;
}

void UniformDistribution::reset() noexcept {
    a_ = math::ZERO_DOUBLE;
    b_ = math::ONE;
    cache_valid_ = false;
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
    cache_valid_ = false;
}

void UniformDistribution::setB(double b) {
    validateParameters(a_, b);
    b_ = b;
    cache_valid_ = false;
}

void UniformDistribution::setParameters(double a, double b) {
    validateParameters(a, b);
    a_ = a;
    b_ = b;
    cache_valid_ = false;
}

double UniformDistribution::getMean() const {
    if (!cache_valid_) {
        updateCache();
    }
    return cached_mean_;
}

double UniformDistribution::getVariance() const {
    if (!cache_valid_) {
        updateCache();
    }
    return cached_variance_;
}

double UniformDistribution::getStandardDeviation() const {
    if (!cache_valid_) {
        updateCache();
    }
    return cached_std_dev_;
}

bool UniformDistribution::isApproximatelyEqual(const UniformDistribution& other, double tolerance) const {
    return std::abs(a_ - other.a_) < tolerance && 
           std::abs(b_ - other.b_) < tolerance;
}

bool UniformDistribution::operator==(const UniformDistribution& other) const {
    return isApproximatelyEqual(other, precision::LIMIT_TOLERANCE);
}

std::ostream& operator<<(std::ostream& os, const UniformDistribution& dist) {
    os << std::fixed << std::setprecision(6);
    os << "Uniform Distribution: a = " << dist.getA() << ", b = " << dist.getB();
    return os;
}

std::istream& operator>>(std::istream& is, UniformDistribution& dist) {
    std::string token;
    double a, b;
    
    try {
        // Expected format: "Uniform Distribution: a = <value>, b = <value>"
        std::string a_str, b_str;
        is >> token >> token >> token >> token >> a_str >> token >> token >> token >> b_str;  // "Uniform" "Distribution:" "a" "=" <a_str> "," "b" "=" <b_str>
        a = std::stod(a_str);
        b = std::stod(b_str);
        
        if (is.good()) {
            dist.setParameters(a, b);
        }
        
    } catch (const std::exception& e) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }
    
    return is;
}

} // namespace libhmm
