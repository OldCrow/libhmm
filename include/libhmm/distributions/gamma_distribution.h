#ifndef GAMMADISTRIBUTION_H_
#define GAMMADISTRIBUTION_H_

#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"
// Common.h already includes: <iostream>, <cmath>, <cassert>, <stdexcept>, <sstream>, <iomanip>

namespace libhmm{

/**
 * Modern C++17 Gamma distribution for modeling continuous non-negative data.
 * 
 * The Gamma distribution is a versatile continuous probability distribution
 * commonly used to model waiting times, failure rates, and size distributions.
 * It generalizes the exponential distribution and is the conjugate prior for
 * the precision of a normal distribution.
 * 
 * PDF: f(x) = (1/(Γ(k)θ^k)) * x^(k-1) * exp(-x/θ) for x ≥ 0
 * where k is the shape parameter (k > 0) and θ is the scale parameter (θ > 0)
 * Γ(k) is the gamma function
 * 
 * Alternative parameterization uses rate parameter β = 1/θ:
 * PDF: f(x) = (β^k/Γ(k)) * x^(k-1) * exp(-βx)
 * 
 * Properties:
 * - Mean: k*θ (or k/β)
 * - Variance: k*θ² (or k/β²)
 * - Support: x ∈ [0, ∞)
 * - Special cases: k=1 gives exponential distribution, k→∞ approaches normal
 */
class GammaDistribution : public ProbabilityDistribution
{    
private:
    /**
     * Shape parameter k - must be positive
     * Controls the "shape" of the distribution:
     * - k < 1: decreasing PDF with vertical asymptote at x=0
     * - k = 1: exponential distribution
     * - k > 1: unimodal with mode at (k-1)*θ
     */
    double k_{1.0};

    /**
     * Scale parameter θ - must be positive
     * Controls the "scale" or spread of the distribution
     * Larger θ spreads the distribution to the right
     */
    double theta_{1.0};
    
    /**
     * Cached value of ln(Γ(k)) for efficiency in probability calculations
     * Updated when shape parameter changes
     */
    mutable double logGammaK_{0.0};
    
    /**
     * Cached value of k * ln(θ) for efficiency in probability calculations
     * Updated when parameters change
     */
    mutable double kLogTheta_{0.0};
    
    /**
     * Cached value of (k-1) for efficiency in probability calculations
     * Updated when shape parameter changes
     */
    mutable double kMinus1_{0.0};
    
    /**
     * Flag to track if cached values need updating
     */
    mutable bool cacheValid_{false};
    
    /**
     * Updates cached values when parameters change
     */
    void updateCache() const noexcept {
        logGammaK_ = std::lgamma(k_);
        kLogTheta_ = k_ * std::log(theta_);
        kMinus1_ = k_ - 1.0;
        cacheValid_ = true;
    }
    
    /**
     * Validates parameters for the Gamma distribution
     * @param k Shape parameter (must be positive and finite)
     * @param theta Scale parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    static void validateParameters(double k, double theta) {
        if (std::isnan(k) || std::isinf(k) || k <= 0.0) {
            throw std::invalid_argument("Shape parameter k must be a positive finite number");
        }
        if (std::isnan(theta) || std::isinf(theta) || theta <= 0.0) {
            throw std::invalid_argument("Scale parameter theta must be a positive finite number");
        }
    }

    /**
     * Evaluates the LOWER INCOMPLETE gamma function at x
     * Uses numerical approximation for computational efficiency
     */
    double ligamma(double a, double x) noexcept;

public:
    /**
     * Constructs a Gamma distribution with given parameters.
     * 
     * @param k Shape parameter k (must be positive)
     * @param theta Scale parameter θ (must be positive)
     * @throws std::invalid_argument if parameters are invalid
     */
    explicit GammaDistribution(double k = 1.0, double theta = 1.0)
        : k_{k}, theta_{theta}, logGammaK_{0.0}, kLogTheta_{0.0}, 
          kMinus1_{0.0}, cacheValid_{false} {
        validateParameters(k, theta);
        updateCache();
    }
    
    /**
     * Copy constructor
     */
    GammaDistribution(const GammaDistribution& other) 
        : k_{other.k_}, theta_{other.theta_}, logGammaK_{other.logGammaK_}, 
          kLogTheta_{other.kLogTheta_}, kMinus1_{other.kMinus1_}, 
          cacheValid_{other.cacheValid_} {}
    
    /**
     * Copy assignment operator
     */
    GammaDistribution& operator=(const GammaDistribution& other) {
        if (this != &other) {
            k_ = other.k_;
            theta_ = other.theta_;
            logGammaK_ = other.logGammaK_;
            kLogTheta_ = other.kLogTheta_;
            kMinus1_ = other.kMinus1_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }
    
    /**
     * Move constructor
     */
    GammaDistribution(GammaDistribution&& other) noexcept
        : k_{other.k_}, theta_{other.theta_}, logGammaK_{other.logGammaK_}, 
          kLogTheta_{other.kLogTheta_}, kMinus1_{other.kMinus1_}, 
          cacheValid_{other.cacheValid_} {}
    
    /**
     * Move assignment operator
     */
    GammaDistribution& operator=(GammaDistribution&& other) noexcept {
        if (this != &other) {
            k_ = other.k_;
            theta_ = other.theta_;
            logGammaK_ = other.logGammaK_;
            kLogTheta_ = other.kLogTheta_;
            kMinus1_ = other.kMinus1_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }

    /**
     * Destructor - explicitly defaulted to satisfy Rule of Five
     */
    ~GammaDistribution() override = default;

    /**
     * Computes the probability density function for the Gamma distribution.
     * 
     * @param value The value at which to evaluate the PDF
     * @return Probability density (or approximated probability for discrete sampling)
     */
    [[nodiscard]] double getProbability(double x) override;
    
    /**
     * Evaluates the logarithm of the probability density function
     * Formula: log PDF(x) = (k-1)*ln(x) - x/θ - k*ln(θ) - ln(Γ(k))
     * More numerically stable for small probabilities
     * 
     * @param x The value at which to evaluate the log PDF
     * @return Log probability density
     */
    [[nodiscard]] double getLogProbability(double x) const noexcept override;

    /**
     * Evaluates the CDF at x using the incomplete gamma function
     * Formula: CDF(x) = P(k, x/θ) = γ(k, x/θ) / Γ(k)
     * where P is the regularized incomplete gamma function
     * 
     * @param x The value at which to evaluate the CDF
     * @return Cumulative probability P(X ≤ x)
     */
    [[nodiscard]] double getCumulativeProbability(double x) noexcept;

    /**
     * Fits the distribution parameters to the given data using method of moments estimation.
     * More sophisticated MLE methods could be implemented but require iterative algorithms.
     * 
     * Method of moments:
     * - sample_mean = k*θ
     * - sample_variance = k*θ²
     * - Solving: θ = sample_variance/sample_mean, k = sample_mean²/sample_variance
     * 
     * @param values Vector of observed data
     */
    void fit(const std::vector<Observation>& values) override;

    /**
     * Resets the distribution to default parameters (k = 1.0, θ = 1.0).
     * This corresponds to the standard exponential distribution.
     */
    void reset() noexcept override;

    /**
     * Returns a string representation of the distribution.
     * 
     * @return String describing the distribution parameters
     */
    [[nodiscard]] std::string toString() const override;

    /**
     * Gets the shape parameter k.
     * 
     * @return Current shape parameter value
     */
    [[nodiscard]] double getK() const noexcept { return k_; }
    
    /**
     * Gets the scale parameter θ.
     * 
     * @return Current scale parameter value
     */
    [[nodiscard]] double getTheta() const noexcept { return theta_; }
    
    /**
     * Sets the shape parameter k.
     * 
     * @param k New shape parameter (must be positive)
     * @throws std::invalid_argument if k <= 0 or is not finite
     */
    void setK(double k) {
        validateParameters(k, theta_);
        k_ = k;
        cacheValid_ = false;
    }
    
    /**
     * Sets the scale parameter θ.
     * 
     * @param theta New scale parameter (must be positive)
     * @throws std::invalid_argument if theta <= 0 or is not finite
     */
    void setTheta(double theta) {
        validateParameters(k_, theta);
        theta_ = theta;
        cacheValid_ = false;
    }
    
    /**
     * Sets both parameters simultaneously.
     * 
     * @param k New shape parameter
     * @param theta New scale parameter
     * @throws std::invalid_argument if parameters are invalid
     */
    void setParameters(double k, double theta) {
        validateParameters(k, theta);
        k_ = k;
        theta_ = theta;
        cacheValid_ = false;
    }
    
    /**
     * Gets the mean of the distribution.
     * For Gamma distribution, mean = k*θ
     * 
     * @return Mean value
     */
    [[nodiscard]] double getMean() const noexcept { 
        return k_ * theta_; 
    }
    
    /**
     * Gets the variance of the distribution.
     * For Gamma distribution, variance = k*θ²
     * 
     * @return Variance value
     */
    [[nodiscard]] double getVariance() const noexcept { 
        return k_ * theta_ * theta_; 
    }
    
    /**
     * Gets the standard deviation of the distribution.
     * For Gamma distribution, std_dev = θ*√k
     * 
     * @return Standard deviation value
     */
    [[nodiscard]] double getStandardDeviation() const noexcept { 
        return theta_ * std::sqrt(k_); 
    }
    
    /**
     * Gets the mode of the distribution.
     * For Gamma distribution with k > 1, mode = (k-1)*θ
     * For k ≤ 1, the mode is at x = 0 (but PDF may be infinite there)
     * 
     * @return Mode value
     */
    [[nodiscard]] double getMode() const noexcept {
        return (k_ > 1.0) ? (k_ - 1.0) * theta_ : 0.0;
    }
    
    /**
     * Gets the rate parameter β = 1/θ (alternative parameterization).
     * 
     * @return Rate parameter (1/θ)
     */
    [[nodiscard]] double getRate() const noexcept {
        return 1.0 / theta_;
    }
    
    /**
     * Equality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are equal within tolerance
     */
    bool operator==(const GammaDistribution& other) const;
    
    /**
     * Inequality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are not equal
     */
    bool operator!=(const GammaDistribution& other) const { return !(*this == other); }
};

std::ostream& operator<<( std::ostream&, 
        const libhmm::GammaDistribution& );
std::istream& operator>>( std::istream&,
        libhmm::GammaDistribution& );

} // namespace
#endif
