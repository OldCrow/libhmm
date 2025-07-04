#ifndef POISSONDISTRIBUTION_H_
#define POISSONDISTRIBUTION_H_

#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"
#include <array>
// Common.h already includes: <iostream>, <cmath>, <cassert>, <stdexcept>, <sstream>, <iomanip>

namespace libhmm{

/**
 * Modern C++17 Poisson distribution for modeling count data and rare events.
 * 
 * The Poisson distribution models the number of events occurring in a fixed 
 * interval of time or space, given that these events occur with a known 
 * constant mean rate and independently of the time since the last event.
 * 
 * PMF: P(X = k) = (λ^k * e^(-λ)) / k!  for k = 0, 1, 2, ...
 * where λ (lambda) is the rate parameter (mean number of events per interval)
 */
class PoissonDistribution : public ProbabilityDistribution
{   
private:
    /**
     * Rate parameter λ (lambda) - average number of events per interval.
     * Must be positive. Also equals both mean and variance of the distribution.
     */
    double lambda_{1.0};

    /**
     * Comprehensive cached values for efficiency in probability calculations
     */
    mutable double logLambda_{0.0};         // log(λ) - used in log probability calculations
    mutable double expNegLambda_{1.0};      // exp(-λ) - used frequently in PMF calculations
    mutable double sqrtLambda_{1.0};        // sqrt(λ) - used in standard deviation and CDF
    mutable double invSqrtLambda_{1.0};     // 1/sqrt(λ) - used in normal approximation
    mutable double sqrtTwoPiLambda_{1.0};   // sqrt(2πλ) - used in normal approximation
    
    /**
     * Pre-computed small factorial values for efficiency (0! through 12!)
     * Computing factorials directly for small values is faster than Stirling's approximation
     */
    mutable std::array<double, 13> smallFactorials_;
    
    /**
     * Flag to track if cached values need updating
     */
    mutable bool cacheValid_{false};
    
    /**
     * Updates all cached values when lambda changes
     * Pre-computes expensive calculations to avoid repeated computation
     */
    void updateCache() const noexcept {
        logLambda_ = std::log(lambda_);
        expNegLambda_ = std::exp(-lambda_);
        sqrtLambda_ = std::sqrt(lambda_);
        invSqrtLambda_ = 1.0 / sqrtLambda_;
        sqrtTwoPiLambda_ = std::sqrt(2.0 * constants::math::PI * lambda_);
        
        // Pre-compute small factorials for performance
        smallFactorials_[0] = 1.0;  // 0! = 1
        smallFactorials_[1] = 1.0;  // 1! = 1
        for (int i = 2; i <= 12; ++i) {
            smallFactorials_[i] = smallFactorials_[i-1] * static_cast<double>(i);
        }
        
        cacheValid_ = true;
    }
    
    /**
     * Computes log(k!) using Stirling's approximation for large k,
     * exact computation for small k.
     * 
     * @param k Non-negative integer
     * @return log(k!)
     */
    double logFactorial(int k) const noexcept;
    
    /**
     * Validates parameters for the Poisson distribution
     * @param lambda Rate parameter (must be positive and finite)
     * @throws std::invalid_argument if lambda is invalid
     */
    void validateParameters(double lambda) const {
        if (std::isnan(lambda) || std::isinf(lambda) || lambda <= 0.0) {
            throw std::invalid_argument("Lambda must be a positive finite number");
        }
    }
    
    /**
     * Validates that k is a valid count (non-negative integer)
     * 
     * @param k Value to validate
     * @return true if k is a valid count, false otherwise
     */
    bool isValidCount(double k) const noexcept {
        return k >= 0.0 && std::floor(k) == k && std::isfinite(k);
    }

    friend std::istream& operator>>(std::istream& is,
            libhmm::PoissonDistribution& distribution);

public:
    /**
     * Constructs a Poisson distribution with given rate parameter.
     * 
     * @param lambda Rate parameter (must be positive)
     * @throws std::invalid_argument if lambda <= 0 or is not finite
     */
    explicit PoissonDistribution(double lambda = 1.0) : lambda_{lambda} {
        validateParameters(lambda);
        updateCache();
    }

    /**
     * Copy constructor
     */
    PoissonDistribution(const PoissonDistribution& other) 
        : lambda_{other.lambda_}, 
          logLambda_{other.logLambda_}, expNegLambda_{other.expNegLambda_},
          sqrtLambda_{other.sqrtLambda_}, invSqrtLambda_{other.invSqrtLambda_},
          sqrtTwoPiLambda_{other.sqrtTwoPiLambda_}, 
          smallFactorials_{other.smallFactorials_}, cacheValid_{other.cacheValid_} {}
    
    /**
     * Copy assignment operator
     */
    PoissonDistribution& operator=(const PoissonDistribution& other) {
        if (this != &other) {
            lambda_ = other.lambda_;
            logLambda_ = other.logLambda_;
            expNegLambda_ = other.expNegLambda_;
            sqrtLambda_ = other.sqrtLambda_;
            invSqrtLambda_ = other.invSqrtLambda_;
            sqrtTwoPiLambda_ = other.sqrtTwoPiLambda_;
            smallFactorials_ = other.smallFactorials_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }
    
    /**
     * Move constructor
     */
    PoissonDistribution(PoissonDistribution&& other) noexcept
        : lambda_{other.lambda_}, 
          logLambda_{other.logLambda_}, expNegLambda_{other.expNegLambda_},
          sqrtLambda_{other.sqrtLambda_}, invSqrtLambda_{other.invSqrtLambda_},
          sqrtTwoPiLambda_{other.sqrtTwoPiLambda_}, 
          smallFactorials_{std::move(other.smallFactorials_)}, cacheValid_{other.cacheValid_} {}
    
    /**
     * Move assignment operator
     */
    PoissonDistribution& operator=(PoissonDistribution&& other) noexcept {
        if (this != &other) {
            lambda_ = other.lambda_;
            logLambda_ = other.logLambda_;
            expNegLambda_ = other.expNegLambda_;
            sqrtLambda_ = other.sqrtLambda_;
            invSqrtLambda_ = other.invSqrtLambda_;
            sqrtTwoPiLambda_ = other.sqrtTwoPiLambda_;
            smallFactorials_ = std::move(other.smallFactorials_);
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }
    
    /**
     * Destructor - explicitly defaulted to satisfy Rule of Five
     */
    ~PoissonDistribution() override = default;

    /**
     * Computes the probability mass function P(X = k) for the Poisson distribution.
     * 
     * @param value The count value k (must be non-negative integer)
     * @return Probability P(X = k), or 0.0 if value is invalid
     */
    double getProbability(double value) override;

    /**
     * Fits the distribution parameters to the given data using maximum likelihood estimation.
     * For Poisson distribution, the MLE of λ is simply the sample mean.
     * 
     * @param values Vector of observed count data
     * @throws std::invalid_argument if values contain negative numbers
     */
    void fit(const std::vector<Observation>& values) override;

    /**
     * Resets the distribution to default parameters (λ = 1.0).
     */
    void reset() noexcept override;

    /**
     * Returns a string representation of the distribution.
     * 
     * @return String describing the distribution parameters
     */
    std::string toString() const override;

    /**
     * Gets the rate parameter λ.
     * 
     * @return Current lambda value
     */
    double getLambda() const noexcept { return lambda_; }
    
    /**
     * Sets the rate parameter λ.
     * 
     * @param lambda New rate parameter (must be positive)
     * @throws std::invalid_argument if lambda <= 0 or is not finite
     */
    void setLambda(double lambda) {
        validateParameters(lambda);
        lambda_ = lambda;
        cacheValid_ = false;
    }
    
    /**
     * Gets the mean of the distribution (equal to λ).
     * 
     * @return Mean value
     */
    double getMean() const noexcept { return lambda_; }
    
    /**
     * Gets the variance of the distribution (equal to λ).
     * 
     * @return Variance value
     */
    double getVariance() const noexcept { return lambda_; }
    
    /**
     * Gets the standard deviation of the distribution (sqrt(λ)).
     * Uses cached value for efficiency.
     * 
     * @return Standard deviation
     */
    double getStandardDeviation() const noexcept { 
        if (!cacheValid_) {
            updateCache();
        }
        return sqrtLambda_; 
    }
    
    /**
     * Evaluates the logarithm of the probability mass function
     * More numerically stable for small probabilities
     * 
     * @param value The count value k at which to evaluate the log PMF
     * @return Log probability mass
     */
    [[nodiscard]] double getLogProbability(double value) const noexcept override;
    
    /**
     * Evaluates the CDF at k using cumulative sum approach
     * Formula: CDF(k) = ∑(i=0 to k) P(X = i)
     * 
     * @param k The value at which to evaluate the CDF
     * @return Cumulative probability P(X ≤ k)
     */
    [[nodiscard]] double getCumulativeProbability(double k) noexcept;
    
    /**
     * Equality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are equal within tolerance
     */
    bool operator==(const PoissonDistribution& other) const;
    
    /**
     * Inequality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are not equal
     */
    bool operator!=(const PoissonDistribution& other) const { return !(*this == other); }
};

/**
 * Stream output operator for Poisson distribution.
 */
std::ostream& operator<<(std::ostream& os, const libhmm::PoissonDistribution& distribution);

} // namespace libhmm

#endif // POISSONDISTRIBUTION_H_
