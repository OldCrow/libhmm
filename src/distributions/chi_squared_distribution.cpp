#include "libhmm/distributions/chi_squared_distribution.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <algorithm>   // For std::remove_if, std::max, std::min (exists in common.h, included for clarity)
#include <numeric>     // For std::accumulate (not in common.h)
#include <limits>      // For std::numeric_limits (exists in common.h via <climits>)

using namespace libhmm::constants;

namespace libhmm {

double ChiSquaredDistribution::getProbability(Observation value) {
    if (!cache_valid_) {
        updateCache();
    }
    
    auto x = static_cast<double>(value);
    
    // Handle invalid inputs
    if (!std::isfinite(x)) {
        return math::ZERO_DOUBLE;
    }
    
    // Return 0 for negative values (outside support)
    if (x < math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }
    
    // Handle special case for x = 0
    if (x == math::ZERO_DOUBLE) {
        if (degrees_of_freedom_ < math::TWO) {
            return std::numeric_limits<double>::infinity();
        } else if (degrees_of_freedom_ == math::TWO) {
            return std::exp(cached_log_normalization_);  // exp(-log(2)) = 0.5
        } else {
            return math::ZERO_DOUBLE;
        }
    }
    
    // Log PDF: log(f(x|k)) = log_normalization + (k/2-1)*log(x) - x/2
    double log_prob = cached_log_normalization_ + cached_half_k_minus_one_ * std::log(x) - math::HALF * x;
    
    return std::exp(log_prob);
}

double ChiSquaredDistribution::getLogProbability(Observation value) const noexcept {
    if (!cache_valid_) {
        updateCache();
    }
    
    auto x = static_cast<double>(value);
    
    // Handle invalid inputs
    if (!std::isfinite(x)) {
        return -std::numeric_limits<double>::infinity();
    }
    
    // Return -∞ for negative values (outside support)
    if (x < math::ZERO_DOUBLE) {
        return -std::numeric_limits<double>::infinity();
    }
    
    // Handle special case for x = 0
    if (x == math::ZERO_DOUBLE) {
        if (degrees_of_freedom_ < math::TWO) {
            return std::numeric_limits<double>::infinity();
        } else if (degrees_of_freedom_ == math::TWO) {
            return cached_log_normalization_;  // log(0.5) = -log(2)
        } else {
            return -std::numeric_limits<double>::infinity();
        }
    }
    
    // Log PDF: log(f(x|k)) = log_normalization + (k/2-1)*log(x) - x/2
    return cached_log_normalization_ + cached_half_k_minus_one_ * std::log(x) - math::HALF * x;
}

double ChiSquaredDistribution::getCumulativeProbability(double x) {
    // Handle invalid inputs
    if (std::isnan(x)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    // Return 0 for negative values (outside support)
    if (x <= math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }
    
    // For positive x, use the relationship to incomplete gamma function:
    // CDF(x) = γ(k/2, x/2) / Γ(k/2) = P(k/2, x/2)
    // where γ is the lower incomplete gamma function and P is the regularized gamma function
    double half_k = math::HALF * degrees_of_freedom_;
    double half_x = math::HALF * x;
    
    // Use the gamma probability function from the base class
    return gammap(half_k, half_x);
}

void ChiSquaredDistribution::fit(const std::vector<Observation>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }
    
    // Check for invalid values and ensure non-negative
    for (Observation obs : values) {
        auto val = static_cast<double>(obs);
        if (!std::isfinite(val) || val < math::ZERO_DOUBLE) {
            throw std::invalid_argument("Chi-squared distribution requires non-negative finite values");
        }
    }
    
    // Method of moments estimation for degrees of freedom
    // For Chi-squared distribution: E[X] = k
    // Therefore: k̂ = sample_mean
    
    size_t n = values.size();
    
    // Convert to double and calculate sample mean
    std::vector<double> double_values{};
    double_values.reserve(n);
    for (Observation obs : values) {
        double_values.push_back(static_cast<double>(obs));
    }
    
    double mean = std::accumulate(double_values.begin(), double_values.end(), math::ZERO_DOUBLE) / n;
    
    // Estimate degrees of freedom as sample mean
    double estimated_df = mean;
    
    // Clamp to reasonable bounds
    estimated_df = std::max(thresholds::MIN_DEGREES_OF_FREEDOM, 
                           std::min(thresholds::MAX_DEGREES_OF_FREEDOM, estimated_df));
    
    setDegreesOfFreedom(estimated_df);
}

void ChiSquaredDistribution::reset() noexcept {
    degrees_of_freedom_ = math::ONE;
    cache_valid_ = false;
}

std::string ChiSquaredDistribution::toString() const {
    std::ostringstream oss{};
    oss << "ChiSquared Distribution:\n";
    oss << "  k (degrees of freedom) = " << std::fixed << std::setprecision(6) << degrees_of_freedom_;
    return oss.str();
}

ChiSquaredDistribution ChiSquaredDistribution::fromString(const std::string& str) {
    // Expected format: "ChiSquared(k=value)" or "ChiSquared(df=value)"
    std::string::size_type start = str.find('(');
    std::string::size_type end = str.find(')', start);
    
    if (start == std::string::npos || end == std::string::npos) {
        throw std::invalid_argument("Invalid ChiSquared distribution string format");
    }
    
    std::string params = str.substr(start + 1, end - start - 1);
    
    // Look for parameter patterns
    std::string::size_type eq_pos = params.find('=');
    if (eq_pos == std::string::npos) {
        throw std::invalid_argument("Invalid ChiSquared parameter format");
    }
    
    std::string param_name = params.substr(0, eq_pos);
    std::string param_value = params.substr(eq_pos + 1);
    
    // Remove whitespace
    param_name.erase(std::remove_if(param_name.begin(), param_name.end(), ::isspace), param_name.end());
    param_value.erase(std::remove_if(param_value.begin(), param_value.end(), ::isspace), param_value.end());
    
    if (param_name == "k" || param_name == "df") {
        double df = std::stod(param_value);
        return ChiSquaredDistribution(df);
    } else {
        throw std::invalid_argument("Unknown ChiSquared parameter: " + param_name);
    }
}

bool ChiSquaredDistribution::operator==(const ChiSquaredDistribution& other) const {
    return std::abs(degrees_of_freedom_ - other.degrees_of_freedom_) < PARAMETER_TOLERANCE;
}

std::ostream& operator<<(std::ostream& os, const ChiSquaredDistribution& dist) {
    os << std::fixed << std::setprecision(6);
    os << "ChiSquared Distribution: k = " << dist.getDegreesOfFreedom();
    return os;
}

std::istream& operator>>(std::istream& is, ChiSquaredDistribution& dist) {
    std::string token;
    double k = 0.0;
    
    try {
        // Expected format: "ChiSquared Distribution: k = <value>"
        std::string k_str;
        is >> token >> token >> token >> token >> k_str;  // "ChiSquared" "Distribution:" "k" "=" <k_str>
        k = std::stod(k_str);
        
        if (is.good()) {
            dist.setDegreesOfFreedom(k);
        }
        
    } catch (const std::exception& e) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }
    
    return is;
}

} // namespace libhmm
