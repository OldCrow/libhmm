#include "libhmm/distributions/chi_squared_distribution.h"
#include <algorithm>
#include <span>

using namespace libhmm::constants;

namespace libhmm {

double ChiSquaredDistribution::getProbability(double value) const {
    if (!isCacheValid()) updateCache();
    const double x = value;
    if (!std::isfinite(x)) return math::ZERO_DOUBLE;
    
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

double ChiSquaredDistribution::getLogProbability(double value) const noexcept {
    if (!isCacheValid()) updateCache();
    const double x = value;
    
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

double ChiSquaredDistribution::getCumulativeProbability(double x) const noexcept {
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

void ChiSquaredDistribution::fit(std::span<const double> data) {
    if (data.empty()) throw std::invalid_argument("Cannot fit distribution to empty data");
    double sum = 0.0;
    for (const double v : data) {
        if (!std::isfinite(v) || v < 0.0)
            throw std::invalid_argument("Chi-squared distribution requires non-negative finite values");
        sum += v;
    }
    double est = std::max(MIN_DEGREES_OF_FREEDOM,
                          std::min(MAX_DEGREES_OF_FREEDOM, sum / static_cast<double>(data.size())));
    setDegreesOfFreedom(est);
}

void ChiSquaredDistribution::fit(std::span<const double> data,
                                 std::span<const double> weights) {
    double sumW = 0.0, sumWX = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) { sumW += weights[i]; sumWX += weights[i] * data[i]; }
    if (sumW < precision::ZERO || std::isnan(sumW)) { reset(); return; }
    double est = std::max(MIN_DEGREES_OF_FREEDOM, std::min(MAX_DEGREES_OF_FREEDOM, sumWX / sumW));
    setDegreesOfFreedom(est);
}

void ChiSquaredDistribution::reset() noexcept {
    degrees_of_freedom_ = math::ONE;
    invalidateCache();
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
