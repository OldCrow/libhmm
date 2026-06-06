#pragma once

#include "libhmm/distributions/distribution_base.h"
#include "libhmm/common/common.h"
#include <span>

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
class UniformDistribution : public DistributionBase<UniformDistribution> {
private:
    double a_; ///< Lower bound
    double b_; ///< Upper bound

    mutable double cached_pdf_{0.0};
    mutable double cached_log_pdf_{0.0};
    mutable double cached_range_{1.0};
    mutable double cached_inv_range_{1.0};
    mutable double cached_mean_{0.5};
    mutable double cached_variance_{1.0 / 12.0};
    mutable double cached_std_dev_{1.0 / std::sqrt(12.0)};

    void updateCache() const noexcept;
    static void validateParameters(double a, double b);

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

    UniformDistribution(const UniformDistribution &other) = default;
    UniformDistribution &operator=(const UniformDistribution &other) = default;
    UniformDistribution(UniformDistribution &&other) noexcept = default;
    UniformDistribution &operator=(UniformDistribution &&other) noexcept = default;
    ~UniformDistribution() override = default;

    [[nodiscard]] double getProbability(double val) const override;
    [[nodiscard]] double getLogProbability(double val) const noexcept override;

    /// Concrete non-virtual batch log-PDF (constant inside support, -Inf outside).
    /// Eliminates per-element virtual dispatch.
    /// Precondition: observations.size() == out.size()
    void getBatchLogProbabilities(std::span<const double> observations,
                                  std::span<double> out) const override;
    double CDF(double x) const;

    /** Fit [a, b] to unweighted data using sample min/max with padding. */
    void fit(std::span<const double> data) override;

    /**
     * Weighted fit using method of moments on weighted mean/variance.
     * For Uniform(a,b): mean = (a+b)/2, var = (b-a)²/12.
     * Solve: a = μ - √(3σ²), b = μ + √(3σ²).
     * Falls back to reset() if sumW is near zero or variance is zero.
     */
    void fit(std::span<const double> data, std::span<const double> weights) override;

    /** Returns false — Uniform is a continuous distribution. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }
    [[nodiscard]] std::size_t getNumParameters() const noexcept override { return 2; }

    /**
     * @brief Reset distribution to default parameters [0, 1]
     */
    void reset() noexcept override;

    /**
     * @brief Get string representation of the distribution
     * @return String description
     */
    std::string toString() const override;
    [[nodiscard]] std::string to_json() const override;
    /// @internal JSON factory — called by the distribution registry in src/io/hmm_json.cpp.
    static std::unique_ptr<EmissionDistribution> from_json(json::Reader &r);

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
    void setB(double b);
    void setParameters(double a, double b);
    double getMean() const;
    double getVariance() const;
    double getStandardDeviation() const;

    /**
     * @brief Check if two distributions are approximately equal
     * @param other Other distribution to compare
     * @param tolerance Tolerance for floating point comparison
     * @return True if distributions are approximately equal
     */
    bool isApproximatelyEqual(const UniformDistribution &other, double tolerance = 1e-9) const;

    /**
     * @brief Equality operator
     * @param other Other distribution to compare
     * @return True if distributions are approximately equal
     */
    bool operator==(const UniformDistribution &other) const;
};

/**
 * @brief Stream output operator
 * @param os Output stream
 * @param dist Distribution to output
 * @return Reference to the output stream
 */
std::ostream &operator<<(std::ostream &os, const UniformDistribution &dist);

/**
 * @brief Stream input operator
 * @param is Input stream
 * @param dist Distribution to input
 * @return Reference to the input stream
 */
std::istream &operator>>(std::istream &is, UniformDistribution &dist);

} // namespace libhmm
