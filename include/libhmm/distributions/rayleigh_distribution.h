#ifndef RAYLEIGHDISTRIBUTION_H_
#define RAYLEIGHDISTRIBUTION_H_

#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"
// Common.h already includes: <iostream>, <ccmath>, <cassert>, <stdexcept>, <sstream>, <iomanip>

namespace libhmm{

/**
 * Modern C++17 Rayleigh distribution for modeling magnitudes and speeds.
 * 
 * The Rayleigh distribution is a continuous probability distribution that arises
 * when modeling the magnitude of a 2D random vector whose components are independent,
 * identically distributed, zero-mean Gaussian random variables.
 * 
 * This is a special case of the Weibull distribution with shape parameter k = 2,
 * but implemented as a standalone class for maximum efficiency.
 * 
 * PDF: f(x) = (x/σ²) * exp(-x²/(2σ²)) for x ≥ 0, 0 otherwise
 * CDF: F(x) = 1 - exp(-x²/(2σ²)) for x ≥ 0, 0 otherwise
 * where σ is the scale parameter (σ > 0)
 * 
 * Properties:
 * - Mean: σ * √(π/2) ≈ 1.253 * σ
 * - Variance: σ² * (4-π)/2 ≈ 0.429 * σ²
 * - Mode: σ
 * - Support: x ∈ [0, ∞)
 * 
 * Applications:
 * - Wind speed modeling
 * - Wave height analysis
 * - Signal processing (magnitude of complex Gaussian noise)
 * - Materials science (fiber strength)
 * - Communications (fading channel modeling)
 */
class RayleighDistribution : public ProbabilityDistribution
{   
private:
    /**
     * Scale parameter σ (sigma) - must be positive
     * Controls the spread and scale of the distribution
     */
    double sigma_{1.0};
    
    /**
     * Cached value of ln(σ) for efficiency in log probability calculations
     */
    mutable double logSigma_{0.0};
    
    /**
     * Cached value of 1/σ for efficiency (multiply instead of divide)
     */
    mutable double invSigma_{1.0};
    
    /**
     * Cached value of 1/σ² for efficiency in PDF and CDF calculations
     */
    mutable double invSigmaSquared_{1.0};
    
    /**
     * Cached value of -1/(2σ²) for CDF and log-PDF calculations
     * This eliminates the need for division and negation in hot paths
     */
    mutable double negHalfInvSigmaSquared_{-0.5};
    
    /**
     * Cached value of σ² for variance and other calculations
     */
    mutable double sigmaSquared_{1.0};
    
    /**
     * Cached value of σ * √(π/2) for mean calculation
     * Mean = σ * √(π/2) ≈ 1.2533141373 * σ
     */
    mutable double mean_{constants::math::SQRT_PI_OVER_TWO};
    
    /**
     * Cached value of σ² * (4-π)/2 for variance calculation  
     * Variance = σ² * (4-π)/2 ≈ 0.4292036732 * σ²
     */
    mutable double variance_{constants::math::FOUR_MINUS_PI_OVER_TWO};
    
    /**
     * Flag to track if cached values need updating
     */
    mutable bool cacheValid_{false};
    
    /**
     * Updates cached values when parameters change
     * Computes all derived values to eliminate divisions and operations in hot paths
     */
    void updateCache() const noexcept {
        logSigma_ = std::log(sigma_);
        invSigma_ = constants::math::ONE / sigma_;
        sigmaSquared_ = sigma_ * sigma_;
        invSigmaSquared_ = invSigma_ * invSigma_;
        negHalfInvSigmaSquared_ = -constants::math::HALF * invSigmaSquared_;
        mean_ = sigma_ * constants::math::SQRT_PI_OVER_TWO;
        variance_ = sigmaSquared_ * constants::math::FOUR_MINUS_PI_OVER_TWO;
        cacheValid_ = true;
    }
    
    /**
     * Validates parameters for the Rayleigh distribution
     * @param sigma Scale parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    static void validateParameters(double sigma) {
        if (std::isnan(sigma) || std::isinf(sigma) || sigma <= 0.0) {
            throw std::invalid_argument("Sigma (scale parameter) must be a positive finite number");
        }
    }

    friend std::istream& operator>>(std::istream& is,
            libhmm::RayleighDistribution& distribution);

public:
    /**
     * Constructs a Rayleigh distribution with given scale parameter.
     * 
     * @param sigma Scale parameter σ (must be positive)
     * @throws std::invalid_argument if sigma is invalid
     */
    explicit RayleighDistribution(double sigma = 1.0)
        : sigma_{sigma} {
        validateParameters(sigma);
        updateCache();
    }
    
    /**
     * Copy constructor
     */
    RayleighDistribution(const RayleighDistribution& other) 
        : sigma_{other.sigma_}, logSigma_{other.logSigma_}, 
          invSigma_{other.invSigma_}, invSigmaSquared_{other.invSigmaSquared_},
          negHalfInvSigmaSquared_{other.negHalfInvSigmaSquared_}, 
          sigmaSquared_{other.sigmaSquared_}, mean_{other.mean_}, 
          variance_{other.variance_}, cacheValid_{other.cacheValid_} {}
    
    /**
     * Copy assignment operator
     */
    RayleighDistribution& operator=(const RayleighDistribution& other) {
        if (this != &other) {
            sigma_ = other.sigma_;
            logSigma_ = other.logSigma_;
            invSigma_ = other.invSigma_;
            invSigmaSquared_ = other.invSigmaSquared_;
            negHalfInvSigmaSquared_ = other.negHalfInvSigmaSquared_;
            sigmaSquared_ = other.sigmaSquared_;
            mean_ = other.mean_;
            variance_ = other.variance_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }
    
    /**
     * Move constructor
     */
    RayleighDistribution(RayleighDistribution&& other) noexcept
        : sigma_{other.sigma_}, logSigma_{other.logSigma_}, 
          invSigma_{other.invSigma_}, invSigmaSquared_{other.invSigmaSquared_},
          negHalfInvSigmaSquared_{other.negHalfInvSigmaSquared_}, 
          sigmaSquared_{other.sigmaSquared_}, mean_{other.mean_}, 
          variance_{other.variance_}, cacheValid_{other.cacheValid_} {}
    
    /**
     * Move assignment operator
     */
    RayleighDistribution& operator=(RayleighDistribution&& other) noexcept {
        if (this != &other) {
            sigma_ = other.sigma_;
            logSigma_ = other.logSigma_;
            invSigma_ = other.invSigma_;
            invSigmaSquared_ = other.invSigmaSquared_;
            negHalfInvSigmaSquared_ = other.negHalfInvSigmaSquared_;
            sigmaSquared_ = other.sigmaSquared_;
            mean_ = other.mean_;
            variance_ = other.variance_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }

    /**
     * Destructor
     */
    ~RayleighDistribution() = default;

    /**
     * Computes the probability density function for the Rayleigh distribution.
     * 
     * PDF: f(x) = (x/σ²) * exp(-x²/(2σ²)) for x ≥ 0
     * 
     * @param value The value at which to evaluate the PDF
     * @return Probability density (or approximated probability for discrete sampling)
     */
    double getProbability(double value) override;

    /**
     * Computes the logarithm of the probability density function for numerical stability.
     * 
     * For Rayleigh distribution: log(f(x)) = log(x) - 2*log(σ) - x²/(2σ²) for x > 0
     * 
     * @param value The value at which to evaluate the log-PDF
     * @return Natural logarithm of the probability density, or -∞ for invalid values
     */
    double getLogProbability(double value) const noexcept override;

    /**
     * Computes the cumulative distribution function for the Rayleigh distribution.
     * 
     * CDF: F(x) = 1 - exp(-x²/(2σ²)) for x ≥ 0
     * 
     * @param value The value at which to evaluate the CDF
     * @return Cumulative probability, or 0.0 for negative values
     */
    double getCumulativeProbability(double value) const noexcept;

    /**
     * Fits the distribution parameters to the given data using maximum likelihood estimation.
     * For Rayleigh distribution, MLE gives σ = √(Σx²/(2n)).
     * 
     * @param values Vector of observed data
     */
    void fit(const std::vector<Observation>& values) override;

    /**
     * Resets the distribution to default parameters (σ = 1.0).
     * This corresponds to the standard Rayleigh distribution.
     */
    void reset() noexcept override;

    /**
     * Returns a string representation of the distribution.
     * 
     * @return String describing the distribution parameters
     */
    std::string toString() const override;

    /**
     * Gets the scale parameter σ.
     * 
     * @return Current scale parameter value
     */
    double getSigma() const noexcept { return sigma_; }
    
    /**
     * Sets the scale parameter σ.
     * 
     * @param sigma New scale parameter (must be positive)
     * @throws std::invalid_argument if sigma is invalid
     */
    void setSigma(double sigma) {
        validateParameters(sigma);
        sigma_ = sigma;
        cacheValid_ = false;
    }
    
    /**
     * Gets the mean of the distribution.
     * Mean = σ * √(π/2)
     * 
     * @return Mean value
     */
    double getMean() const noexcept {
        if (!cacheValid_) updateCache();
        return mean_;
    }
    
    /**
     * Gets the variance of the distribution.
     * Variance = σ² * (4-π)/2
     * 
     * @return Variance value
     */
    double getVariance() const noexcept {
        if (!cacheValid_) updateCache();
        return variance_;
    }
    
    /**
     * Gets the standard deviation of the distribution.
     * 
     * @return Standard deviation (square root of variance)
     */
    double getStandardDeviation() const noexcept {
        return std::sqrt(getVariance());
    }
    
    /**
     * Gets the mode of the distribution.
     * Mode = σ
     * 
     * @return Mode value
     */
    double getMode() const noexcept {
        return sigma_;
    }
    
    /**
     * Gets the median of the distribution.
     * Median = σ * √(2 * ln(2)) ≈ 1.177 * σ
     * 
     * @return Median value
     */
    double getMedian() const noexcept {
        return sigma_ * constants::math::SQRT_TWO_LN_TWO;
    }

    /**
     * Equality operator
     */
    bool operator==(const RayleighDistribution& other) const noexcept {
        return std::abs(sigma_ - other.sigma_) < constants::precision::ULTRA_HIGH_PRECISION_TOLERANCE;
    }

    /**
     * Inequality operator
     */
    bool operator!=(const RayleighDistribution& other) const noexcept {
        return !(*this == other);
    }
};

}  // namespace libhmm

#endif  // RAYLEIGHDISTRIBUTION_H_
