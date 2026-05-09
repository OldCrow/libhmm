#pragma once

#include "libhmm/distributions/distribution_base.h"
#include "libhmm/common/common.h"
#include <span>

namespace libhmm {

/**
 * Modern C++20 Pareto distribution for modeling power-law phenomena.
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
class ParetoDistribution : public DistributionBase {
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
     * Cached value of log(x_m) for efficiency in log probability calculations
     */
    mutable double logXm_{0.0};

    void updateCache() const noexcept {
        logK_ = std::log(k_);
        logXm_ = std::log(xm_);
        kLogXm_ = k_ * logXm_;
        kPlus1_ = k_ + constants::math::ONE;
        kXmPowK_ = k_ * std::pow(xm_, k_);
        markCacheValid();
    }

    /**
     * Validates parameters for the Pareto distribution
     * @param k Shape parameter (must be positive and finite)
     * @param xm Scale parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    static void validateParameters(double k, double xm) {
        if (std::isnan(k) || std::isinf(k) || k <= 0.0) {
            throw std::invalid_argument("Shape parameter k must be a positive finite number");
        }
        if (std::isnan(xm) || std::isinf(xm) || xm <= 0.0) {
            throw std::invalid_argument("Scale parameter xm must be a positive finite number");
        }
    }

    friend std::istream &operator>>(std::istream &is, libhmm::ParetoDistribution &distribution);

public:
    /**
     * Constructs a Pareto distribution with given parameters.
     *
     * @param k Shape parameter k (must be positive)
     * @param xm Scale parameter x_m (must be positive)
     * @throws std::invalid_argument if parameters are invalid
     */
    explicit ParetoDistribution(double k = 1.0, double xm = 1.0) : k_{k}, xm_{xm} {
        validateParameters(k, xm);
        updateCache();
    }

    ParetoDistribution(const ParetoDistribution &other) = default;
    ParetoDistribution &operator=(const ParetoDistribution &other) = default;
    ParetoDistribution(ParetoDistribution &&other) noexcept = default;
    ParetoDistribution &operator=(ParetoDistribution &&other) noexcept = default;
    ~ParetoDistribution() override = default;

    /**
     * Computes the probability density function for the Pareto distribution.
     *
     * @param value The value at which to evaluate the PDF
     * @return Probability density (or approximated probability for discrete sampling)
     */
    [[nodiscard]] double getProbability(double x) const override;
    [[nodiscard]] double getLogProbability(double value) const noexcept override;

    /// Concrete non-virtual batch log-PDF. Eliminates per-element virtual dispatch.
    /// Precondition: observations.size() == out.size()
    void getBatchLogProbabilities(std::span<const double> observations,
                                  std::span<double> out) const override;
    [[nodiscard]] double getCumulativeProbability(double value) const noexcept;

    /** MLE: x_m = min(x_i), k̂ = n / Σ(ln(x_i/x_m)). */
    void fit(std::span<const double> data) override;
    /** Weighted MLE: x_m = min(x_i), k̂ = Σw_i / Σ(w_i * ln(x_i/x_m)). */
    void fit(std::span<const double> data, std::span<const double> weights) override;

    /** Returns false — Pareto is a continuous distribution. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

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
    [[nodiscard]] std::string to_json() const override;
    /// @internal JSON factory — called by the distribution registry in src/io/hmm_json.cpp.
    static std::unique_ptr<EmissionDistribution> from_json(json::Reader &r);

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
        invalidateCache();
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
        invalidateCache();
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
        invalidateCache();
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
    double getMode() const noexcept { return xm_; }

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
    bool operator==(const ParetoDistribution &other) const noexcept {
        return std::abs(k_ - other.k_) < constants::precision::ULTRA_HIGH_PRECISION_TOLERANCE &&
               std::abs(xm_ - other.xm_) < constants::precision::ULTRA_HIGH_PRECISION_TOLERANCE;
    }

    /**
     * Inequality operator
     */
    bool operator!=(const ParetoDistribution &other) const noexcept { return !(*this == other); }
};

std::ostream &operator<<(std::ostream &, const libhmm::ParetoDistribution &);
std::istream &operator>>(std::istream &, libhmm::ParetoDistribution &);
} // namespace libhmm
