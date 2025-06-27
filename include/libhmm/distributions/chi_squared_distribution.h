#ifndef CHI_SQUARED_DISTRIBUTION_H_
#define CHI_SQUARED_DISTRIBUTION_H_

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"

namespace libhmm {

/**
 * Chi-squared distribution for modeling sums of squared standard normal variables.
 * 
 * The Chi-squared distribution is a continuous probability distribution with support
 * on non-negative real numbers. It is a special case of the Gamma distribution and
 * arises frequently in statistical inference, particularly in hypothesis testing.
 * 
 * Mathematical properties:
 * - PDF: f(x; k) = (1/(2^(k/2) * Γ(k/2))) * x^(k/2-1) * e^(-x/2)
 * - Support: x ∈ [0, ∞)
 * - Parameters: k > 0 (degrees of freedom)
 * - Mean: k
 * - Variance: 2k
 * - Relation to Gamma: χ²(k) = Gamma(k/2, 2)
 * 
 * Applications:
 * - Goodness-of-fit tests
 * - Tests of independence in contingency tables
 * - Confidence intervals for variance
 * - Distribution of sample variance in normal populations
 * - Model selection criteria (AIC, BIC)
 * - Likelihood ratio tests
 */
class ChiSquaredDistribution : public ProbabilityDistribution {
private:
    /**
     * Degrees of freedom parameter k - must be positive
     */
    double degrees_of_freedom_{1.0};
    
    /**
     * Cached value of log(Γ(k/2)) for efficiency in probability calculations
     */
    mutable double cached_log_gamma_half_k_{0.0};
    
    /**
     * Cached log normalization constant: -k/2 * log(2) - log(Γ(k/2))
     */
    mutable double cached_log_normalization_{0.0};
    
    /**
     * Cached value of (k/2 - 1) for efficiency in probability calculations
     */
    mutable double cached_half_k_minus_one_{0.0};
    
    /**
     * Flag to track if cached values need updating
     */
    mutable bool cache_valid_{false};

    /**
     * Updates cached values when parameters change
     */
    void updateCache() const noexcept {
        double half_k = 0.5 * degrees_of_freedom_;
        
        cached_log_gamma_half_k_ = loggamma(half_k);
        cached_half_k_minus_one_ = half_k - 1.0;
        
        // Log normalization: -log(2^(k/2) * Γ(k/2)) = -k/2 * log(2) - log(Γ(k/2))
        cached_log_normalization_ = -half_k * std::log(2.0) - cached_log_gamma_half_k_;
        
        cache_valid_ = true;
    }
    
    /**
     * Validates parameters for the Chi-squared distribution
     * @param degrees_of_freedom Degrees of freedom parameter (must be positive and finite)
     * @throws std::invalid_argument if parameter is invalid
     */
    void validateParameters(double degrees_of_freedom) const {
        if (std::isnan(degrees_of_freedom) || std::isinf(degrees_of_freedom) || degrees_of_freedom <= 0.0) {
            throw std::invalid_argument("Degrees of freedom must be a positive finite number");
        }
    }

public:
    /**
     * Constructs a Chi-squared distribution with given degrees of freedom.
     * 
     * @param degrees_of_freedom Degrees of freedom k (must be positive)
     * @throws std::invalid_argument if degrees_of_freedom <= 0
     */
    ChiSquaredDistribution(double degrees_of_freedom = 1.0)
        : degrees_of_freedom_(degrees_of_freedom) {
        validateParameters(degrees_of_freedom);
        updateCache();
    }
    
    /**
     * Copy constructor
     */
    ChiSquaredDistribution(const ChiSquaredDistribution& other) 
        : degrees_of_freedom_(other.degrees_of_freedom_),
          cached_log_gamma_half_k_(other.cached_log_gamma_half_k_),
          cached_log_normalization_(other.cached_log_normalization_),
          cached_half_k_minus_one_(other.cached_half_k_minus_one_),
          cache_valid_(other.cache_valid_) {}
    
    /**
     * Copy assignment operator
     */
    ChiSquaredDistribution& operator=(const ChiSquaredDistribution& other) {
        if (this != &other) {
            degrees_of_freedom_ = other.degrees_of_freedom_;
            cached_log_gamma_half_k_ = other.cached_log_gamma_half_k_;
            cached_log_normalization_ = other.cached_log_normalization_;
            cached_half_k_minus_one_ = other.cached_half_k_minus_one_;
            cache_valid_ = other.cache_valid_;
        }
        return *this;
    }
    
    /**
     * Move constructor
     */
    ChiSquaredDistribution(ChiSquaredDistribution&& other) noexcept
        : degrees_of_freedom_(other.degrees_of_freedom_),
          cached_log_gamma_half_k_(other.cached_log_gamma_half_k_),
          cached_log_normalization_(other.cached_log_normalization_),
          cached_half_k_minus_one_(other.cached_half_k_minus_one_),
          cache_valid_(other.cache_valid_) {}
    
    /**
     * Move assignment operator
     */
    ChiSquaredDistribution& operator=(ChiSquaredDistribution&& other) noexcept {
        if (this != &other) {
            degrees_of_freedom_ = other.degrees_of_freedom_;
            cached_log_gamma_half_k_ = other.cached_log_gamma_half_k_;
            cached_log_normalization_ = other.cached_log_normalization_;
            cached_half_k_minus_one_ = other.cached_half_k_minus_one_;
            cache_valid_ = other.cache_valid_;
        }
        return *this;
    }
    
    /**
     * Destructor - explicitly defaulted to satisfy Rule of Five
     */
    ~ChiSquaredDistribution() override = default;

    /**
     * Computes the probability density function for the Chi-squared distribution.
     * 
     * @param value The value at which to evaluate the PDF (should be non-negative)
     * @return Probability density f(value|k), or 0.0 if value < 0
     */
    double getProbability(Observation value) override;
    
    /**
     * Computes the log probability density function for numerical stability.
     * 
     * @param value The value at which to evaluate the log PDF (should be non-negative)
     * @return Log probability density log(f(value|k)), or -∞ if value < 0
     */
    double getLogProbability(Observation value) const;
    
    /**
     * Computes the cumulative distribution function.
     * 
     * @param x The value at which to evaluate the CDF
     * @return P(X ≤ x) for the Chi-squared distribution
     */
    double CDF(double x);

    /**
     * Fits the distribution parameters to the given data using method of moments estimation.
     * 
     * Method of moments for Chi-squared distribution:
     * Given sample mean μ, estimate k from: μ = k
     * Therefore: k̂ = sample_mean
     * 
     * @param values Vector of observed data (should be non-negative)
     */
    void fit(const std::vector<Observation>& values) override;

    /**
     * Resets the distribution to default parameters (k = 1.0).
     */
    void reset() noexcept override;

    /**
     * Returns a string representation of the distribution.
     * 
     * @return String describing the distribution parameters
     */
    std::string toString() const override;

    /**
     * Gets the degrees of freedom parameter.
     * 
     * @return Current degrees of freedom value
     */
    double getDegreesOfFreedom() const noexcept { return degrees_of_freedom_; }
    
    /**
     * Sets the degrees of freedom parameter.
     * 
     * @param degrees_of_freedom New degrees of freedom parameter (must be positive)
     * @throws std::invalid_argument if degrees_of_freedom <= 0 or is not finite
     */
    void setDegreesOfFreedom(double degrees_of_freedom) {
        validateParameters(degrees_of_freedom);
        degrees_of_freedom_ = degrees_of_freedom;
        cache_valid_ = false;
    }

    /**
     * Gets the mean of the distribution.
     * 
     * @return Mean (k)
     */
    double getMean() const noexcept { return degrees_of_freedom_; }

    /**
     * Gets the variance of the distribution.
     * 
     * @return Variance (2k)
     */
    double getVariance() const noexcept { return 2.0 * degrees_of_freedom_; }

    /**
     * Gets the standard deviation of the distribution.
     * 
     * @return Standard deviation (√(2k))
     */
    double getStandardDeviation() const noexcept { return std::sqrt(2.0 * degrees_of_freedom_); }

    /**
     * Gets the mode of the distribution.
     * 
     * @return Mode (max(0, k-2))
     */
    double getMode() const noexcept { return std::max(0.0, degrees_of_freedom_ - 2.0); }

    /**
     * Create distribution from string representation
     * @param str String representation
     * @return ChiSquaredDistribution object
     * @throws std::invalid_argument if string format is invalid
     */
    static ChiSquaredDistribution fromString(const std::string& str);

    /**
     * Equality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are equal within tolerance
     */
    bool operator==(const ChiSquaredDistribution& other) const;

    /**
     * Inequality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are not equal
     */
    bool operator!=(const ChiSquaredDistribution& other) const { return !(*this == other); }

private:
    static constexpr double PARAMETER_TOLERANCE = 1e-10;  ///< Tolerance for parameter comparison
    static constexpr double MIN_DEGREES_OF_FREEDOM = 1e-10;  ///< Minimum degrees of freedom
    static constexpr double MAX_DEGREES_OF_FREEDOM = 1e6;    ///< Maximum degrees of freedom for numerical stability
};

/**
 * Stream output operator for Chi-squared distribution
 * @param os Output stream
 * @param dist Distribution to output
 * @return Reference to the output stream
 */
std::ostream& operator<<(std::ostream& os, const ChiSquaredDistribution& dist);

/**
 * Stream input operator for Chi-squared distribution
 * @param is Input stream
 * @param dist Distribution to input
 * @return Reference to the input stream
 */
std::istream& operator>>(std::istream& is, ChiSquaredDistribution& dist);

} // namespace libhmm

#endif // CHI_SQUARED_DISTRIBUTION_H_
