#pragma once

#include "probability_distribution.h"
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <limits>

namespace libhmm {

/**
 * @brief Uniform Distribution
 * 
 * The uniform distribution is a continuous probability distribution where all values
 * within a specified interval [a, b] have equal probability density.
 * 
 * Probability Density Function:
 * f(x) = 1/(b-a) for a ≤ x ≤ b, 0 otherwise
 * 
 * Parameters:
 * - a: Lower bound (minimum value)
 * - b: Upper bound (maximum value)
 * 
 * Properties:
 * - Mean: μ = (a + b) / 2
 * - Variance: σ² = (b - a)² / 12
 * - Support: x ∈ [a, b]
 */
class UniformDistribution : public ProbabilityDistribution {
private:
    double a_;  ///< Lower bound
    double b_;  ///< Upper bound
    
    // Cached values for performance
    mutable double cached_pdf_;         ///< Cached PDF value: 1/(b-a)
    mutable double cached_log_pdf_;     ///< Cached log PDF value: log(1/(b-a))
    mutable bool cache_valid_;          ///< Flag indicating if cache is valid
    
    /**
     * @brief Update cached values
     */
    void updateCache() const;
    
    /**
     * @brief Validate parameters
     * @param a Lower bound
     * @param b Upper bound
     * @throws std::invalid_argument if parameters are invalid
     */
    void validateParameters(double a, double b) const;

public:
    /**
     * @brief Default constructor
     * Creates a uniform distribution on [0, 1]
     */
    UniformDistribution();
    
    /**
     * @brief Parameterized constructor
     * @param a Lower bound
     * @param b Upper bound
     * @throws std::invalid_argument if a >= b or parameters are invalid
     */
    UniformDistribution(double a, double b);
    
    /**
     * @brief Copy constructor
     */
    UniformDistribution(const UniformDistribution& other) = default;
    
    /**
     * @brief Move constructor
     */
    UniformDistribution(UniformDistribution&& other) noexcept = default;
    
    /**
     * @brief Copy assignment operator
     */
    UniformDistribution& operator=(const UniformDistribution& other) = default;
    
    /**
     * @brief Move assignment operator
     */
    UniformDistribution& operator=(UniformDistribution&& other) noexcept = default;
    
    /**
     * @brief Destructor
     */
    virtual ~UniformDistribution() = default;
    
    /**
     * @brief Calculate probability density at x
     * @param val Value to evaluate
     * @return Probability density at x
     */
    double getProbability(Observation val) override;
    
    /**
     * @brief Calculate log probability density at x for numerical stability
     * @param val Value to evaluate
     * @return Log probability density at x
     */
    double getLogProbability(Observation val) const;
    
    /**
     * @brief Calculate cumulative distribution function at x
     * @param x Value to evaluate
     * @return CDF at x: P(X <= x)
     */
    double CDF(double x) const;
    
    /**
     * @brief Fit distribution parameters to data using method of moments
     * @param data Vector of observations
     * @throws std::invalid_argument if data contains invalid values
     */
    void fit(const std::vector<Observation>& data) override;
    
    /**
     * @brief Reset distribution to default parameters [0, 1]
     */
    void reset() noexcept override;
    
    /**
     * @brief Get string representation of the distribution
     * @return String description
     */
    std::string toString() const override;
    
    /**
     * @brief Get the lower bound parameter
     * @return Lower bound a
     */
    double getA() const { return a_; }
    
    /**
     * @brief Get the upper bound parameter
     * @return Upper bound b
     */
    double getB() const { return b_; }
    
    /**
     * @brief Get the lower bound parameter (alternative name)
     * @return Lower bound a
     */
    double getMin() const { return a_; }
    
    /**
     * @brief Get the upper bound parameter (alternative name)
     * @return Upper bound b
     */
    double getMax() const { return b_; }
    
    /**
     * @brief Set the lower bound parameter
     * @param a New lower bound
     * @throws std::invalid_argument if a >= current b or a is invalid
     */
    void setA(double a);
    
    /**
     * @brief Set the upper bound parameter
     * @param b New upper bound
     * @throws std::invalid_argument if b <= current a or b is invalid
     */
    void setB(double b);
    
    /**
     * @brief Set both parameters
     * @param a Lower bound
     * @param b Upper bound
     * @throws std::invalid_argument if a >= b or parameters are invalid
     */
    void setParameters(double a, double b);
    
    /**
     * @brief Get the mean of the distribution
     * @return Mean μ = (a + b) / 2
     */
    double getMean() const;
    
    /**
     * @brief Get the variance of the distribution
     * @return Variance σ² = (b - a)² / 12
     */
    double getVariance() const;
    
    /**
     * @brief Get the standard deviation of the distribution
     * @return Standard deviation σ = (b - a) / √12
     */
    double getStandardDeviation() const;
    
    /**
     * @brief Check if two distributions are approximately equal
     * @param other Other distribution to compare
     * @param tolerance Tolerance for floating point comparison
     * @return True if distributions are approximately equal
     */
    bool isApproximatelyEqual(const UniformDistribution& other, double tolerance = 1e-9) const;
    
    /**
     * @brief Equality operator
     * @param other Other distribution to compare
     * @return True if distributions are approximately equal
     */
    bool operator==(const UniformDistribution& other) const;
};

/**
 * @brief Stream output operator
 * @param os Output stream
 * @param dist Distribution to output
 * @return Reference to the output stream
 */
std::ostream& operator<<(std::ostream& os, const UniformDistribution& dist);

/**
 * @brief Stream input operator
 * @param is Input stream
 * @param dist Distribution to input
 * @return Reference to the input stream
 */
std::istream& operator>>(std::istream& is, UniformDistribution& dist);

} // namespace libhmm
