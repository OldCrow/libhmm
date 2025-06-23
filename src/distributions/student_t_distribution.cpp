#include "libhmm/distributions/student_t_distribution.h"
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace libhmm {

StudentTDistribution::StudentTDistribution()
    : degrees_of_freedom_(1.0), location_(0.0), scale_(1.0), cache_valid_(false) {
}

StudentTDistribution::StudentTDistribution(double degrees_of_freedom)
    : degrees_of_freedom_(degrees_of_freedom), location_(0.0), scale_(1.0), cache_valid_(false) {
    validateParameters(degrees_of_freedom);
    updateCache();
}

StudentTDistribution::StudentTDistribution(double degrees_of_freedom, double location, double scale)
    : degrees_of_freedom_(degrees_of_freedom), location_(location), scale_(scale), cache_valid_(false) {
    validateParameters(degrees_of_freedom);
    if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) {
        throw std::invalid_argument("Scale parameter must be a positive finite number");
    }
    updateCache();
}

StudentTDistribution::StudentTDistribution(const StudentTDistribution& other)
    : degrees_of_freedom_(other.degrees_of_freedom_),
      location_(other.location_),
      scale_(other.scale_),
      cached_log_gamma_half_nu_plus_one_(other.cached_log_gamma_half_nu_plus_one_),
      cached_log_gamma_half_nu_(other.cached_log_gamma_half_nu_),
      cached_log_normalization_(other.cached_log_normalization_),
      cache_valid_(other.cache_valid_) {
}

StudentTDistribution& StudentTDistribution::operator=(const StudentTDistribution& other) {
    if (this != &other) {
        degrees_of_freedom_ = other.degrees_of_freedom_;
        location_ = other.location_;
        scale_ = other.scale_;
        cached_log_gamma_half_nu_plus_one_ = other.cached_log_gamma_half_nu_plus_one_;
        cached_log_gamma_half_nu_ = other.cached_log_gamma_half_nu_;
        cached_log_normalization_ = other.cached_log_normalization_;
        cache_valid_ = other.cache_valid_;
    }
    return *this;
}

double StudentTDistribution::getProbability(Observation value) {
    if (!cache_valid_) {
        updateCache();
    }
    
    double x = static_cast<double>(value);
    
    // Handle invalid inputs
    if (!std::isfinite(x)) {
        return 0.0;
    }
    
    // Standardize with location and scale: z = (x - μ) / σ
    double z = (x - location_) / scale_;
    
    // Log PDF: log(f(x|ν,μ,σ)) = log_normalization - log(σ) - ((ν+1)/2) * log(1 + z²/ν)
    double z_squared_over_nu = (z * z) / degrees_of_freedom_;
    double log_denominator_term = ((degrees_of_freedom_ + 1.0) / 2.0) * std::log(1.0 + z_squared_over_nu);
    
    double log_prob = cached_log_normalization_ - std::log(scale_) - log_denominator_term;
    return std::exp(log_prob);
}

void StudentTDistribution::fit(const std::vector<Observation>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }
    
    // Check for invalid values
    for (Observation obs : values) {
        double val = static_cast<double>(obs);
        if (!std::isfinite(val)) {
            throw std::invalid_argument("Observations contain non-finite values");
        }
    }
    
    // Method of moments estimation for degrees of freedom
    // For t-distribution: Var[X] = ν/(ν-2) for ν > 2
    // Solving: sample_variance = ν/(ν-2) gives ν = 2*sample_variance/(sample_variance-1)
    
    size_t n = values.size();
    if (n < 2) {
        // Not enough data for variance estimation, use default
        degrees_of_freedom_ = 1.0;
        cache_valid_ = false;
        return;
    }
    
    // Convert to double and calculate sample mean and variance
    std::vector<double> double_values;
    double_values.reserve(n);
    for (Observation obs : values) {
        double_values.push_back(static_cast<double>(obs));
    }
    
    double mean = std::accumulate(double_values.begin(), double_values.end(), 0.0) / n;
    double variance = 0.0;
    for (double val : double_values) {
        double diff = val - mean;
        variance += diff * diff;
    }
    variance /= (n - 1);  // Sample variance
    
    // Estimate degrees of freedom using method of moments
    if (variance > 1.0) {
        double estimated_df = 2.0 * variance / (variance - 1.0);
        
        // Clamp to reasonable bounds
        estimated_df = std::max(MIN_DEGREES_OF_FREEDOM, 
                               std::min(MAX_DEGREES_OF_FREEDOM, estimated_df));
        
        setDegreesOfFreedom(estimated_df);
    } else {
        // Variance too small for reliable estimation
        setDegreesOfFreedom(3.0);  // Default reasonable value
    }
}

void StudentTDistribution::reset() noexcept {
    degrees_of_freedom_ = 1.0;
    location_ = 0.0;
    scale_ = 1.0;
    cache_valid_ = false;
}

void StudentTDistribution::setDegreesOfFreedom(double degrees_of_freedom) {
    validateParameters(degrees_of_freedom);
    degrees_of_freedom_ = degrees_of_freedom;
    cache_valid_ = false;
}

void StudentTDistribution::setScale(double scale) {
    if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) {
        throw std::invalid_argument("Scale parameter must be a positive finite number");
    }
    scale_ = scale;
    // Note: Scale doesn't affect cached normalization constant (only affects PDF calculation)
}

double StudentTDistribution::getMean() const {
    if (degrees_of_freedom_ > 1.0) {
        return location_;  // For generalized t-distribution, mean = location parameter
    } else {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

double StudentTDistribution::getVariance() const {
    if (degrees_of_freedom_ > 2.0) {
        // For generalized t-distribution: Var = σ² * ν/(ν-2)
        return scale_ * scale_ * degrees_of_freedom_ / (degrees_of_freedom_ - 2.0);
    } else if (degrees_of_freedom_ > 1.0) {
        return std::numeric_limits<double>::infinity();
    } else {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

double StudentTDistribution::getStandardDeviation() const {
    double variance = getVariance();
    if (std::isfinite(variance)) {
        return std::sqrt(variance);
    } else {
        return variance;  // NaN or infinity
    }
}

std::string StudentTDistribution::toString() const {
    std::ostringstream oss;
    oss << "StudentT Distribution:\n";
    oss << "  nu (degrees of freedom) = " << std::fixed << std::setprecision(6) << degrees_of_freedom_ << "\n";
    oss << "  mu (location) = " << std::fixed << std::setprecision(6) << location_ << "\n";
    oss << "  sigma (scale) = " << std::fixed << std::setprecision(6) << scale_;
    return oss.str();
}

StudentTDistribution StudentTDistribution::fromString(const std::string& str) {
    // Expected format: "StudentT(ν=value)" or "StudentT(nu=value)" or "StudentT(df=value)"
    std::string::size_type start = str.find('(');
    std::string::size_type end = str.find(')', start);
    
    if (start == std::string::npos || end == std::string::npos) {
        throw std::invalid_argument("Invalid StudentT distribution string format");
    }
    
    std::string params = str.substr(start + 1, end - start - 1);
    
    // Look for parameter patterns
    std::string::size_type eq_pos = params.find('=');
    if (eq_pos == std::string::npos) {
        throw std::invalid_argument("Invalid StudentT parameter format");
    }
    
    std::string param_name = params.substr(0, eq_pos);
    std::string param_value = params.substr(eq_pos + 1);
    
    // Remove whitespace
    param_name.erase(std::remove_if(param_name.begin(), param_name.end(), ::isspace), param_name.end());
    param_value.erase(std::remove_if(param_value.begin(), param_value.end(), ::isspace), param_value.end());
    
    if (param_name == "ν" || param_name == "nu" || param_name == "df") {
        double df = std::stod(param_value);
        return StudentTDistribution(df);
    } else {
        throw std::invalid_argument("Unknown StudentT parameter: " + param_name);
    }
}

bool StudentTDistribution::operator==(const StudentTDistribution& other) const {
    return std::abs(degrees_of_freedom_ - other.degrees_of_freedom_) < PARAMETER_TOLERANCE &&
           std::abs(location_ - other.location_) < PARAMETER_TOLERANCE &&
           std::abs(scale_ - other.scale_) < PARAMETER_TOLERANCE;
}

} // namespace libhmm
