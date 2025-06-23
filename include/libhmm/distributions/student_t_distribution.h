#ifndef STUDENT_T_DISTRIBUTION_H_
#define STUDENT_T_DISTRIBUTION_H_

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"

namespace libhmm {

/**
 * @brief Student's t-distribution implementation
 * 
 * The Student's t-distribution is a probability distribution used in statistics,
 * particularly for small sample sizes or when the population variance is unknown.
 * It approaches the normal distribution as degrees of freedom increase.
 * 
 * Mathematical properties:
 * - PDF: f(x|ν) = Γ((ν+1)/2) / (√(νπ) * Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)
 * - Support: x ∈ (-∞, +∞)
 * - Parameters: ν > 0 (degrees of freedom)
 * - Mean: 0 (for ν > 1), undefined otherwise
 * - Variance: ν/(ν-2) (for ν > 2), infinite for 1 < ν ≤ 2, undefined for ν ≤ 1
 * 
 * Applications:
 * - Statistical hypothesis testing (t-tests)
 * - Confidence intervals for unknown variance
 * - Small sample statistical inference
 * - Bayesian analysis with unknown precision
 * - Financial modeling (fat-tailed distributions)
 * - Robust regression analysis
 */
class StudentTDistribution : public ProbabilityDistribution {
private:
    /**
     * Degrees of freedom parameter ν - must be positive
     */
    double degrees_of_freedom_{1.0};
    
    /**
     * Location parameter μ (mean when ν > 1)
     */
    double location_{0.0};
    
    /**
     * Scale parameter σ - must be positive
     */
    double scale_{1.0};
    
    /**
     * Cached value of log(Γ((ν+1)/2)) for efficiency in probability calculations
     */
    mutable double cached_log_gamma_half_nu_plus_one_{0.0};
    
    /**
     * Cached value of log(Γ(ν/2)) for efficiency in probability calculations
     */
    mutable double cached_log_gamma_half_nu_{0.0};
    
    /**
     * Cached log normalization constant
     */
    mutable double cached_log_normalization_{0.0};
    
    /**
     * Flag to track if cached values need updating
     */
    mutable bool cache_valid_{false};

    /**
     * Updates cached values when parameters change
     */
    void updateCache() const noexcept {
        double half_nu = 0.5 * degrees_of_freedom_;
        double half_nu_plus_one = half_nu + 0.5;
        
        cached_log_gamma_half_nu_plus_one_ = loggamma(half_nu_plus_one);
        cached_log_gamma_half_nu_ = loggamma(half_nu);
        
        // Log normalization: log(Γ((ν+1)/2) / (√(νπ) * Γ(ν/2)))
        cached_log_normalization_ = cached_log_gamma_half_nu_plus_one_ 
                                   - cached_log_gamma_half_nu_ 
                                   - 0.5 * (std::log(degrees_of_freedom_) + std::log(M_PI));
        
        cache_valid_ = true;
    }
    
    /**
     * Validates parameters for the Student's t-distribution
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
     * @brief Default constructor with degrees of freedom = 1
     */
    StudentTDistribution();

    /**
     * @brief Constructor with specified degrees of freedom
     * @param degrees_of_freedom Degrees of freedom parameter (ν > 0)
     * @throws std::invalid_argument if degrees_of_freedom <= 0
     */
    explicit StudentTDistribution(double degrees_of_freedom);
    
    /**
     * @brief Constructor with degrees of freedom, location, and scale
     * @param degrees_of_freedom Degrees of freedom parameter (ν > 0)
     * @param location Location parameter (μ)
     * @param scale Scale parameter (σ > 0)
     * @throws std::invalid_argument if parameters are invalid
     */
    StudentTDistribution(double degrees_of_freedom, double location, double scale);

    /**
     * @brief Copy constructor
     */
    StudentTDistribution(const StudentTDistribution& other);

    /**
     * @brief Assignment operator
     */
    StudentTDistribution& operator=(const StudentTDistribution& other);

    /**
     * @brief Destructor
     */
    virtual ~StudentTDistribution() = default;

    /**
     * Computes the probability density function for the Student's t-distribution.
     * 
     * @param value The value at which to evaluate the PDF
     * @return Probability density f(value|ν)
     */
    double getProbability(Observation value) override;

    /**
     * Fits the distribution parameters to the given data using method of moments estimation.
     * 
     * Method of moments for t-distribution:
     * Given sample variance s², estimate ν from: s² = ν/(ν-2)
     * Solving: ν = 2s²/(s²-1) for s² > 1
     * 
     * @param values Vector of observed data
     */
    void fit(const std::vector<Observation>& values) override;

    /**
     * Resets the distribution to default parameters (ν = 1.0).
     * This corresponds to the standard Cauchy distribution.
     */
    void reset() noexcept override;

    /**
     * @brief Get the degrees of freedom parameter
     * @return Degrees of freedom (ν)
     */
    double getDegreesOfFreedom() const { return degrees_of_freedom_; }

    /**
     * @brief Set the degrees of freedom parameter
     * @param degrees_of_freedom New degrees of freedom (ν > 0)
     * @throws std::invalid_argument if degrees_of_freedom <= 0
     */
    void setDegreesOfFreedom(double degrees_of_freedom);
    
    /**
     * @brief Get the location parameter
     * @return Location parameter (μ)
     */
    double getLocation() const { return location_; }
    
    /**
     * @brief Set the location parameter
     * @param location New location parameter (μ)
     */
    void setLocation(double location) { location_ = location; }
    
    /**
     * @brief Get the scale parameter
     * @return Scale parameter (σ)
     */
    double getScale() const { return scale_; }
    
    /**
     * @brief Set the scale parameter
     * @param scale New scale parameter (σ > 0)
     * @throws std::invalid_argument if scale <= 0
     */
    void setScale(double scale);

    /**
     * @brief Get the mean of the distribution
     * @return Mean (0 for ν > 1, NaN otherwise)
     */
    double getMean() const;

    /**
     * @brief Get the variance of the distribution
     * @return Variance (ν/(ν-2) for ν > 2, infinity for 1 < ν ≤ 2, NaN for ν ≤ 1)
     */
    double getVariance() const;

    /**
     * @brief Get the standard deviation of the distribution
     * @return Standard deviation (sqrt(variance))
     */
    double getStandardDeviation() const;

    /**
     * @brief Check if the distribution has finite mean
     * @return true if ν > 1, false otherwise
     */
    bool hasFiniteMean() const { return degrees_of_freedom_ > 1.0; }

    /**
     * @brief Check if the distribution has finite variance
     * @return true if ν > 2, false otherwise
     */
    bool hasFiniteVariance() const { return degrees_of_freedom_ > 2.0; }

    /**
     * Returns a string representation of the distribution.
     * 
     * @return String describing the distribution parameters
     */
    std::string toString() const override;

    /**
     * @brief Create distribution from string representation
     * @param str String representation
     * @return StudentTDistribution object
     * @throws std::invalid_argument if string format is invalid
     */
    static StudentTDistribution fromString(const std::string& str);

    /**
     * @brief Equality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are equal within tolerance
     */
    bool operator==(const StudentTDistribution& other) const;

    /**
     * @brief Inequality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are not equal
     */
    bool operator!=(const StudentTDistribution& other) const { return !(*this == other); }

private:
    static constexpr double PARAMETER_TOLERANCE = 1e-10;  ///< Tolerance for parameter comparison
    static constexpr double MIN_DEGREES_OF_FREEDOM = 1e-10;  ///< Minimum degrees of freedom
    static constexpr double MAX_DEGREES_OF_FREEDOM = 1e6;    ///< Maximum degrees of freedom for numerical stability
};

} // namespace libhmm

#endif // LIBHMM_STUDENT_T_DISTRIBUTION_H
