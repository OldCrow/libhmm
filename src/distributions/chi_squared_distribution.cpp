#include "libhmm/distributions/chi_squared_distribution.h"
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace libhmm {

double ChiSquaredDistribution::getProbability(Observation value) {
    if (!cache_valid_) {
        updateCache();
    }
    
    double x = static_cast<double>(value);
    
    // Handle invalid inputs
    if (!std::isfinite(x)) {
        return 0.0;
    }
    
    // Return 0 for negative values (outside support)
    if (x < 0.0) {
        return 0.0;
    }
    
    // Handle special case for x = 0
    if (x == 0.0) {
        if (degrees_of_freedom_ < 2.0) {
            return std::numeric_limits<double>::infinity();
        } else if (degrees_of_freedom_ == 2.0) {
            return std::exp(cached_log_normalization_);  // exp(-log(2)) = 0.5
        } else {
            return 0.0;
        }
    }
    
    // Log PDF: log(f(x|k)) = log_normalization + (k/2-1)*log(x) - x/2
    double log_prob = cached_log_normalization_ + cached_half_k_minus_one_ * std::log(x) - 0.5 * x;
    
    return std::exp(log_prob);
}

void ChiSquaredDistribution::fit(const std::vector<Observation>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }
    
    // Check for invalid values and ensure non-negative
    for (Observation obs : values) {
        double val = static_cast<double>(obs);
        if (!std::isfinite(val) || val < 0.0) {
            throw std::invalid_argument("Chi-squared distribution requires non-negative finite values");
        }
    }
    
    // Method of moments estimation for degrees of freedom
    // For Chi-squared distribution: E[X] = k
    // Therefore: k̂ = sample_mean
    
    size_t n = values.size();
    
    // Convert to double and calculate sample mean
    std::vector<double> double_values;
    double_values.reserve(n);
    for (Observation obs : values) {
        double_values.push_back(static_cast<double>(obs));
    }
    
    double mean = std::accumulate(double_values.begin(), double_values.end(), 0.0) / n;
    
    // Estimate degrees of freedom as sample mean
    double estimated_df = mean;
    
    // Clamp to reasonable bounds
    estimated_df = std::max(MIN_DEGREES_OF_FREEDOM, 
                           std::min(MAX_DEGREES_OF_FREEDOM, estimated_df));
    
    setDegreesOfFreedom(estimated_df);
}

void ChiSquaredDistribution::reset() noexcept {
    degrees_of_freedom_ = 1.0;
    cache_valid_ = false;
}

std::string ChiSquaredDistribution::toString() const {
    std::ostringstream oss;
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

} // namespace libhmm
