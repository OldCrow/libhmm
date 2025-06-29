#ifndef PARETODISTRIBUTION_H_
#define PARETODISTRIBUTION_H_

#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"
// Common.h already includes: <iostream>, <cmath>, <cassert>, <stdexcept>, <sstream>, <iomanip>

namespace libhmm{

/**
 * Modern C++17 Pareto distribution for modeling power-law phenomena.
 * 
 * The Pareto distribution is a continuous probability distribution commonly
 * used to model income distribution, city population sizes, stock price
 * fluctuations, and other phenomena that follow the "80-20 rule" or
 * Pareto principle.
 * 
 * PDF: f(x) = (k * x_m^k) / x^(k+1) for x ≥ x_m, 0 otherwise
 * CDF: F(x) = 1 - (x_m/x)^k for x ≥ x_m, 0 otherwise
 * where k is the shape parameter (k > 0) and x_m is the scale parameter (x_m > 0)
 * 
 * Properties:
 * - Mean: k*x_m/(k-1) for k > 1, undefined for k ≤ 1
 * - Variance: (k*x_m²)/((k-1)²*(k-2)) for k > 2, undefined for k ≤ 2
 * - Mode: x_m (always at the scale parameter)
 * - Support: x ∈ [x_m, ∞)
 * - Heavy-tailed distribution (polynomial decay)
 */
class ParetoDistribution : public ProbabilityDistribution
{   
private:
    /**
     * Shape parameter k - must be positive
     * Controls the "tail heaviness" of the distribution:
     * - Smaller k: heavier tail, more extreme values
     * - Larger k: lighter tail, more moderate values
     */
    double k_{1.0};

    /**
     * Scale parameter x_m - must be positive
     * Represents the minimum possible value in the distribution
     * All observations must be ≥ x_m
     */
    double xm_{1.0};
    
    /**
     * Cached value of ln(k) for efficiency in probability calculations
     */
    mutable double logK_{0.0};
    
    /**
     * Cached value of k * ln(x_m) for efficiency in probability calculations
     */
    mutable double kLogXm_{0.0};
    
    /**
     * Cached value of (k+1) for efficiency in probability calculations
     */
    mutable double kPlus1_{2.0};
    
    /**
     * Cached value of k * x_m^k for efficiency in PDF calculations
     */
    mutable double kXmPowK_{1.0};
    
    /**
     * Cached value of -k for efficiency in CDF calculations
     */
    mutable double negK_{-1.0};
    
    /**
     * Cached value of log(x_m) for efficiency in log probability calculations
     */
    mutable double logXm_{0.0};
    
    /**
     * Flag to track if cached values need updating
     */
    mutable bool cacheValid_{false};
    
    /**
     * Updates cached values when parameters change
     */
    void updateCache() const noexcept {
        logK_ = std::log(k_);
        logXm_ = std::log(xm_);
        kLogXm_ = k_ * logXm_;
        kPlus1_ = k_ + constants::math::ONE;
        kXmPowK_ = k_ * std::pow(xm_, k_);
        negK_ = -k_;
        cacheValid_ = true;
    }
    
    /**
     * Validates parameters for the Pareto distribution
     * @param k Shape parameter (must be positive and finite)
     * @param xm Scale parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    void validateParameters(double k, double xm) const {
        if (std::isnan(k) || std::isinf(k) || k <= 0.0) {
            throw std::invalid_argument("Shape parameter k must be a positive finite number");
        }
        if (std::isnan(xm) || std::isinf(xm) || xm <= 0.0) {
            throw std::invalid_argument("Scale parameter xm must be a positive finite number");
        }
    }

    /**
     * Evaluates the CDF at x using the standard Pareto CDF formula
     */
    double CDF(double x) const noexcept;

    friend std::istream& operator>>(std::istream& is,
            libhmm::ParetoDistribution& distribution);

public:
    /**
     * Constructs a Pareto distribution with given parameters.
     * 
     * @param k Shape parameter k (must be positive)
     * @param xm Scale parameter x_m (must be positive)
     * @throws std::invalid_argument if parameters are invalid
     */
    ParetoDistribution(double k = 1.0, double xm = 1.0)
        : k_{k}, xm_{xm}, logK_{0.0}, kLogXm_{0.0}, kPlus1_{2.0}, cacheValid_{false} {
        validateParameters(k, xm);
        updateCache();
    }
    
    /**
     * Copy constructor
     */
    ParetoDistribution(const ParetoDistribution& other) 
        : k_{other.k_}, xm_{other.xm_}, logK_{other.logK_}, 
          kLogXm_{other.kLogXm_}, kPlus1_{other.kPlus1_}, 
          kXmPowK_{other.kXmPowK_}, negK_{other.negK_}, logXm_{other.logXm_},
          cacheValid_{other.cacheValid_} {}
    
    /**
     * Copy assignment operator
     */
    ParetoDistribution& operator=(const ParetoDistribution& other) {
        if (this != &other) {
            k_ = other.k_;
            xm_ = other.xm_;
            logK_ = other.logK_;
            kLogXm_ = other.kLogXm_;
            kPlus1_ = other.kPlus1_;
            kXmPowK_ = other.kXmPowK_;
            negK_ = other.negK_;
            logXm_ = other.logXm_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }
    
    /**
     * Move constructor
     */
    ParetoDistribution(ParetoDistribution&& other) noexcept
        : k_{other.k_}, xm_{other.xm_}, logK_{other.logK_}, 
          kLogXm_{other.kLogXm_}, kPlus1_{other.kPlus1_}, 
          kXmPowK_{other.kXmPowK_}, negK_{other.negK_}, logXm_{other.logXm_},
          cacheValid_{other.cacheValid_} {}
    
    /**
     * Move assignment operator
     */
    ParetoDistribution& operator=(ParetoDistribution&& other) noexcept {
        if (this != &other) {
            k_ = other.k_;
            xm_ = other.xm_;
            logK_ = other.logK_;
            kLogXm_ = other.kLogXm_;
            kPlus1_ = other.kPlus1_;
            kXmPowK_ = other.kXmPowK_;
            negK_ = other.negK_;
            logXm_ = other.logXm_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }

    /**
     * Computes the probability density function for the Pareto distribution.
     * 
     * @param value The value at which to evaluate the PDF
     * @return Probability density (or approximated probability for discrete sampling)
     */
    double getProbability(double value) override;

    /**
     * Computes the logarithm of the probability density function for numerical stability.
     * 
     * For Pareto distribution: log(f(x)) = log(k) + k*log(x_m) - (k+1)*log(x) for x ≥ x_m
     * 
     * @param value The value at which to evaluate the log-PDF
     * @return Natural logarithm of the probability density, or -∞ for invalid values
     */
    double getLogProbability(double value) const noexcept override;

    /**
     * Computes the cumulative distribution function for the Pareto distribution.
     * 
     * CDF: F(x) = 1 - (x_m/x)^k for x ≥ x_m
     * 
     * @param value The value at which to evaluate the CDF
     * @return Cumulative probability, or 0.0 for values below x_m
     */
    double getCumulativeProbability(double value) const noexcept;

    /**
     * Fits the distribution parameters to the given data using maximum likelihood estimation.
     * 
     * For Pareto distribution, the MLE estimators are:
     * x_m = min(x_i) for all i
     * k = n / Σ(ln(x_i) - ln(x_m)) for i = 1 to n
     * 
     * @param values Vector of observed data
     */
    void fit(const std::vector<Observation>& values) override;

    /**
     * Resets the distribution to default parameters (k = 1.0, x_m = 1.0).
     * This corresponds to a standard Pareto distribution.
     */
    void reset() noexcept override;

    /**
     * Returns a string representation of the distribution.
     * 
     * @return String describing the distribution parameters
     */
    std::string toString() const override;

    /**
     * Gets the shape parameter k.
     * 
     * @return Current shape parameter value
     */
    double getK() const noexcept { return k_; }
    
    /**
     * Sets the shape parameter k.
     * 
     * @param k New shape parameter (must be positive)
     * @throws std::invalid_argument if k <= 0 or is not finite
     */
    void setK(double k) {
        validateParameters(k, xm_);
        k_ = k;
        cacheValid_ = false;
    }

    /**
     * Gets the scale parameter x_m.
     * 
     * @return Current scale parameter value
     */
    double getXm() const noexcept { return xm_; }
    
    /**
     * Sets the scale parameter x_m.
     * 
     * @param xm New scale parameter (must be positive)
     * @throws std::invalid_argument if xm <= 0 or is not finite
     */
    void setXm(double xm) {
        validateParameters(k_, xm);
        xm_ = xm;
        cacheValid_ = false;
    }
    
    /**
     * Sets both parameters simultaneously.
     * 
     * @param k New shape parameter
     * @param xm New scale parameter
     * @throws std::invalid_argument if parameters are invalid
     */
    void setParameters(double k, double xm) {
        validateParameters(k, xm);
        k_ = k;
        xm_ = xm;
        cacheValid_ = false;
    }
    
    /**
     * Gets the mean of the Pareto distribution.
     * For Pareto distribution, mean = k*x_m/(k-1) if k > 1, undefined otherwise
     * 
     * @return Mean value if k > 1, otherwise returns infinity
     */
    double getMean() const noexcept { 
        return (k_ > 1.0) ? (k_ * xm_) / (k_ - 1.0) : std::numeric_limits<double>::infinity(); 
    }
    
    /**
     * Gets the variance of the Pareto distribution.
     * For Pareto distribution, variance = (k*x_m²)/((k-1)²*(k-2)) if k > 2, undefined otherwise
     * 
     * @return Variance value if k > 2, otherwise returns infinity
     */
    double getVariance() const noexcept { 
        if (k_ > 2.0) {
            double kMinus1 = k_ - 1.0;
            return (k_ * xm_ * xm_) / (kMinus1 * kMinus1 * (k_ - 2.0));
        }
        return std::numeric_limits<double>::infinity();
    }
    
    /**
     * Gets the standard deviation of the Pareto distribution.
     * 
     * @return Standard deviation if k > 2, otherwise returns infinity
     */
    double getStandardDeviation() const noexcept { 
        double var = getVariance();
        return std::isinf(var) ? var : std::sqrt(var);
    }
    
    /**
     * Gets the mode of the Pareto distribution.
     * For Pareto distribution, mode = x_m (always at the scale parameter)
     * 
     * @return Mode value (equals x_m)
     */
    double getMode() const noexcept {
        return xm_;
    }
    
    /**
     * Gets the median of the Pareto distribution.
     * For Pareto distribution, median = x_m * 2^(1/k)
     * 
     * @return Median value
     */
    double getMedian() const noexcept {
        return xm_ * std::pow(constants::math::TWO, constants::math::ONE / k_);
    }

    /**
     * Equality operator with tolerance for floating-point comparison
     */
    bool operator==(const ParetoDistribution& other) const noexcept {
        return std::abs(k_ - other.k_) < constants::precision::ULTRA_HIGH_PRECISION_TOLERANCE &&
               std::abs(xm_ - other.xm_) < constants::precision::ULTRA_HIGH_PRECISION_TOLERANCE;
    }

    /**
     * Inequality operator
     */
    bool operator!=(const ParetoDistribution& other) const noexcept {
        return !(*this == other);
    }

};

std::ostream& operator<<( std::ostream&, 
        const libhmm::ParetoDistribution& );
std::istream& operator>>( std::istream&,
        libhmm::ParetoDistribution& );
} // namespace
#endif
