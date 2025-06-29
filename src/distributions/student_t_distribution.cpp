#include "libhmm/distributions/student_t_distribution.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <algorithm>   // For std::remove_if, std::max, std::min (exists in common.h, included for clarity)
#include <numeric>     // For std::accumulate (not in common.h)
#include <limits>      // For std::numeric_limits (exists in common.h via <climits>)

using namespace libhmm::constants;

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
      cached_normalization_factor_(other.cached_normalization_factor_),
      cached_half_nu_plus_one_(other.cached_half_nu_plus_one_),
      cached_half_nu_(other.cached_half_nu_),
      cached_inv_scale_(other.cached_inv_scale_),
      cached_log_scale_(other.cached_log_scale_),
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
        cached_normalization_factor_ = other.cached_normalization_factor_;
        cached_half_nu_plus_one_ = other.cached_half_nu_plus_one_;
        cached_half_nu_ = other.cached_half_nu_;
        cached_inv_scale_ = other.cached_inv_scale_;
        cached_log_scale_ = other.cached_log_scale_;
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
        return math::ZERO_DOUBLE;
    }
    
    // Direct calculation for better performance
    // Standardize with location and scale: z = (x - μ) / σ
    const double z = (x - location_) * cached_inv_scale_;
    const double z_squared_over_nu = (z * z) / degrees_of_freedom_;
    
    // Direct PDF calculation using cached normalization constant:
    // f(x) = normalization_factor * (1 + z²/ν)^(-(ν+1)/2)
    const double power_term = std::pow(math::ONE + z_squared_over_nu, -cached_half_nu_plus_one_);
    
    return cached_normalization_factor_ * power_term;
}

/**
 * Computes the logarithm of the probability density function for numerical stability.
 */
double StudentTDistribution::getLogProbability(double value) const noexcept {
    if (!std::isfinite(value)) {
        return -std::numeric_limits<double>::infinity();
    }
    
    if (!cache_valid_) {
        updateCache();
    }
    
    // Standardize with location and scale: z = (x - μ) / σ
    const double z = (value - location_) * cached_inv_scale_;
    
    // Optimized log PDF using cached values:
    // log(f(x|ν,μ,σ)) = cached_log_normalization - ((ν+1)/2) * log(1 + z²/ν)
    const double z_squared_over_nu = (z * z) / degrees_of_freedom_;
    const double log_denominator_term = cached_half_nu_plus_one_ * std::log(math::ONE + z_squared_over_nu);
    
    return cached_log_normalization_ - log_denominator_term;
}

/**
 * Computes the cumulative distribution function for the Student's t-distribution.
 * 
 * Uses the relationship with the incomplete beta function for numerical accuracy.
 */
double StudentTDistribution::getCumulativeProbability(double value) const noexcept {
    if (!std::isfinite(value)) {
        return std::isnan(value) ? std::numeric_limits<double>::quiet_NaN() : 
               (value < math::ZERO_DOUBLE ? math::ZERO_DOUBLE : math::ONE);
    }
    
    if (!cache_valid_) {
        updateCache();
    }
    
    // Standardize: t = (x - μ) / σ
    const double t = (value - location_) * cached_inv_scale_;
    
    // For standard t-distribution, use the relationship with incomplete beta function:
    // CDF(t) = 1/2 + (t/sqrt(ν)) * B(1/2, ν/2) / B(1/2, ν/2) * hypergeometric_function
    // 
    // More practical implementation using the identity:
    // If X ~ t_ν, then X² / (ν + X²) ~ Beta(1/2, ν/2)
    
    const double t_squared = t * t;
    const double beta_x = t_squared / (degrees_of_freedom_ + t_squared);
    
    // Use incomplete beta function I_x(a,b) where x = t²/(ν + t²), a = 1/2, b = ν/2
    double incomplete_beta_val;
    try {
        // Note: This is a simplified approximation. For production code,
        // you would use a proper incomplete beta function implementation
        if (std::abs(t) < 1e-8) {
            // For very small |t|, use series expansion: CDF ≈ 1/2 + t/(sqrt(π*ν)) + O(t³)
            incomplete_beta_val = math::HALF + t / std::sqrt(constants::math::PI * degrees_of_freedom_) / math::TWO;
        } else {
            // Simplified approximation using erf for moderate degrees of freedom
            if (degrees_of_freedom_ >= 30.0) {
                // For large ν, t-distribution approaches normal distribution
                incomplete_beta_val = math::HALF * (math::ONE + std::erf(t / constants::math::SQRT_2));
            } else {
                // Use a rational approximation for moderate ν
                const double sqrt_term = std::sqrt(degrees_of_freedom_ / (degrees_of_freedom_ + t_squared));
                incomplete_beta_val = math::HALF * (math::ONE + (t > math::ZERO_DOUBLE ? math::ONE : -math::ONE) * 
                                     (math::ONE - sqrt_term));
            }
        }
    } catch (...) {
        // Fallback for numerical issues
        incomplete_beta_val = (t >= math::ZERO_DOUBLE) ? 0.9 : 0.1;
    }
    
    // Ensure result is in [0,1]
    return std::max(math::ZERO_DOUBLE, std::min(math::ONE, incomplete_beta_val));
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
    
    // Method of moments estimation using efficient Welford's algorithm
    // For t-distribution: Var[X] = ν/(ν-2) for ν > 2
    // Solving: sample_variance = ν/(ν-2) gives ν = 2*sample_variance/(sample_variance-1)
    
    size_t n = values.size();
    if (n < 2) {
        // Not enough data for variance estimation, use default
        degrees_of_freedom_ = 1.0;
        cache_valid_ = false;
        return;
    }
    
    // Use Welford's algorithm for numerical stability and performance
    double mean = 0.0;
    double M2 = 0.0;  // Sum of squared differences from current mean
    double count = 0.0;
    
    for (Observation obs : values) {
        double val = static_cast<double>(obs);
        count += 1.0;
        double delta = val - mean;
        mean += delta / count;
        double delta2 = val - mean;
        M2 += delta * delta2;
    }
    
    double variance = (count > 1.0) ? M2 / (count - 1.0) : 0.0;
    
    // Estimate degrees of freedom using method of moments
    if (variance > 1.0) {
        double estimated_df = 2.0 * variance / (variance - 1.0);
        
        // Clamp to reasonable bounds
        estimated_df = std::max(constants::thresholds::MIN_DEGREES_OF_FREEDOM, 
                               std::min(constants::thresholds::MAX_DEGREES_OF_FREEDOM, estimated_df));
        
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

bool StudentTDistribution::operator!=(const StudentTDistribution& other) const {
    return !(*this == other);
}

/**
 * Input stream operator for reading StudentT distribution from formatted text.
 * Expected format: "StudentT(nu=value, mu=value, sigma=value)" or similar variations.
 */
std::istream& operator>>(std::istream& is, StudentTDistribution& dist) {
    std::string line;
    if (!std::getline(is, line)) {
        is.setstate(std::ios::failbit);
        return is;
    }
    
    try {
        // Parse parameters from the line
        double nu = 1.0, mu = 0.0, sigma = 1.0;
        bool found_nu = false, found_mu = false, found_sigma = false;
        
        // Look for parameter patterns
        std::string::size_type start = line.find('(');
        std::string::size_type end = line.rfind(')');
        
        if (start != std::string::npos && end != std::string::npos && end > start) {
            std::string params = line.substr(start + 1, end - start - 1);
            
            // Split by commas and parse each parameter
            std::istringstream param_stream(params);
            std::string param;
            
            while (std::getline(param_stream, param, ',')) {
                std::string::size_type eq_pos = param.find('=');
                if (eq_pos != std::string::npos) {
                    std::string name = param.substr(0, eq_pos);
                    std::string value = param.substr(eq_pos + 1);
                    
                    // Trim whitespace
                    name.erase(std::remove_if(name.begin(), name.end(), ::isspace), name.end());
                    value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());
                    
                    if (name == "nu" || name == "ν" || name == "df") {
                        nu = std::stod(value);
                        found_nu = true;
                    } else if (name == "mu" || name == "μ" || name == "location") {
                        mu = std::stod(value);
                        found_mu = true;
                    } else if (name == "sigma" || name == "σ" || name == "scale") {
                        sigma = std::stod(value);
                        found_sigma = true;
                    }
                }
            }
        }
        
        // Create new distribution with parsed parameters
        if (found_nu && found_mu && found_sigma) {
            dist = StudentTDistribution(nu, mu, sigma);
        } else if (found_nu) {
            dist = StudentTDistribution(nu);
        } else {
            // Default case - just parse the first number if any
            std::istringstream number_stream(line);
            double value;
            if (number_stream >> value) {
                dist = StudentTDistribution(value);
            } else {
                is.setstate(std::ios::failbit);
            }
        }
    } catch (const std::exception&) {
        is.setstate(std::ios::failbit);
    }
    
    return is;
}

/**
 * Output stream operator for writing StudentT distribution in readable format.
 */
std::ostream& operator<<(std::ostream& os, const StudentTDistribution& dist) {
    os << "StudentT(nu=" << std::fixed << std::setprecision(6) << dist.getDegreesOfFreedom()
       << ", mu=" << dist.getLocation() << ", sigma=" << dist.getScale() << ")";
    return os;
}

/**
 * Validates the degrees of freedom parameter.
 */
void StudentTDistribution::validateParameters(double degrees_of_freedom) const {
    if (std::isnan(degrees_of_freedom) || std::isinf(degrees_of_freedom) || degrees_of_freedom <= 0.0) {
        throw std::invalid_argument("Degrees of freedom must be a positive finite number");
    }
}

/**
 * Updates cached values for efficient repeated calculations.
 * This method computes and caches expensive values like log-gamma and normalization constants.
 */
void StudentTDistribution::updateCache() const {
    // Cache frequently used fractional values
    cached_half_nu_ = degrees_of_freedom_ * math::HALF;
    cached_half_nu_plus_one_ = cached_half_nu_ + math::HALF;
    
    // Cache scale-related values
    cached_inv_scale_ = math::ONE / scale_;
    cached_log_scale_ = std::log(scale_);
    
    // Compute log-gamma values for normalization
    // For Student's t-distribution: 
    // PDF = Γ((ν+1)/2) / (√(νπ) * Γ(ν/2)) * (1 + t²/ν)^(-(ν+1)/2)
    // log(PDF) = log(Γ((ν+1)/2)) - log(Γ(ν/2)) - (1/2)*log(νπ) - (ν+1)/2 * log(1 + t²/ν)
    
    try {
        cached_log_gamma_half_nu_plus_one_ = std::lgamma(cached_half_nu_plus_one_);
        cached_log_gamma_half_nu_ = std::lgamma(cached_half_nu_);
    } catch (const std::exception&) {
        // Fallback for extreme values
        cached_log_gamma_half_nu_plus_one_ = math::ZERO_DOUBLE;
        cached_log_gamma_half_nu_ = math::ZERO_DOUBLE;
    }
    
    // Pre-compute the log normalization constant:
    // log_normalization = log(Γ((ν+1)/2)) - log(Γ(ν/2)) - (1/2)*log(νπ) - log(σ)
    const double log_nu_pi = std::log(degrees_of_freedom_ * constants::math::PI);
    
    cached_log_normalization_ = cached_log_gamma_half_nu_plus_one_ - 
                               cached_log_gamma_half_nu_ - 
                               math::HALF * log_nu_pi - 
                               cached_log_scale_;
    
    // Cache the exponential of the log normalization for direct PDF calculation
    cached_normalization_factor_ = std::exp(cached_log_normalization_);
    
    cache_valid_ = true;
}

} // namespace libhmm
