#pragma once

#include "libhmm/distributions/distribution_base.h"
#include "libhmm/common/common.h"
#include <span>

namespace libhmm {

/**
 * Beta distribution for modeling probabilities and proportions.
 *
 * The Beta distribution is a continuous probability distribution defined
 * on the interval [0,1] and parameterized by two positive shape parameters
 * α (alpha) and β (beta).
 *
 * PDF: f(x; α, β) = (x^(α-1) * (1-x)^(β-1)) / B(α, β)
 * where B(α, β) is the Beta function: B(α, β) = Γ(α)Γ(β)/Γ(α+β)
 *
 * Special cases:
 * - α = β = 1: Uniform distribution on [0,1]
 * - α = β: Symmetric around 0.5
 * - α < β: Skewed toward 0
 * - α > β: Skewed toward 1
 */
class BetaDistribution : public DistributionBase {
private:
    /**
     * Shape parameter α (alpha) - must be positive
     */
    double alpha_{1.0};

    /**
     * Shape parameter β (beta) - must be positive
     */
    double beta_{1.0};

    /**
     * Cached value of log(B(α, β)) for efficiency in probability calculations
     */
    mutable double logBeta_{0.0};

    /**
     * Cached value of (α-1) for efficiency - used in every PDF calculation
     */
    mutable double alphaMinus1_{0.0};

    /**
     * Cached value of (β-1) for efficiency - used in every PDF calculation
     */
    mutable double betaMinus1_{0.0};

    /**
     * Cached value of 1/B(α,β) for direct PDF calculation efficiency
     */
    mutable double invBeta_{1.0};

    void updateCache() const noexcept {
        logBeta_ = std::lgamma(alpha_) + std::lgamma(beta_) - std::lgamma(alpha_ + beta_);
        invBeta_ = std::exp(-logBeta_);
        alphaMinus1_ = alpha_ - 1.0;
        betaMinus1_ = beta_ - 1.0;
        markCacheValid();
    }

    /**
     * Validates parameters for the Beta distribution
     * @param alpha Alpha parameter (must be positive and finite)
     * @param beta Beta parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    static void validateParameters(double alpha, double beta) {
        if (std::isnan(alpha) || std::isinf(alpha) || alpha <= 0.0) {
            throw std::invalid_argument("Alpha parameter must be a positive finite number");
        }
        if (std::isnan(beta) || std::isinf(beta) || beta <= 0.0) {
            throw std::invalid_argument("Beta parameter must be a positive finite number");
        }
    }

    friend std::istream &operator>>(std::istream &is, libhmm::BetaDistribution &distribution);

public:
    /**
     * Constructs a Beta distribution with given shape parameters.
     *
     * @param alpha Shape parameter α (must be positive)
     * @param beta Shape parameter β (must be positive)
     * @throws std::invalid_argument if parameters are not positive finite numbers
     */
    explicit BetaDistribution(double alpha = 1.0, double beta = 1.0) : alpha_{alpha}, beta_{beta} {
        validateParameters(alpha, beta);
        updateCache();
    }

    BetaDistribution(const BetaDistribution &other) = default;
    BetaDistribution &operator=(const BetaDistribution &other) = default;
    BetaDistribution(BetaDistribution &&other) noexcept = default;
    BetaDistribution &operator=(BetaDistribution &&other) noexcept = default;
    ~BetaDistribution() override = default;

    /**
     * Computes the probability density function for the Beta distribution.
     *
     * @param value The value at which to evaluate the PDF (should be in [0,1])
     * @return Probability density, or 0.0 if value is outside [0,1]
     */
    [[nodiscard]] double getProbability(double value) const override;
    [[nodiscard]] double getLogProbability(double value) const noexcept override;

    /// Concrete non-virtual batch log-PDF. Eliminates per-element virtual dispatch;
    /// enables compiler auto-vectorization under -march=native or /arch:AVX2.
    /// Precondition: observations.size() == out.size()
    void getBatchLogProbabilities(std::span<const double> observations,
                                  std::span<double> out) const override;
    [[nodiscard]] double sample(std::mt19937_64 &rng) const override;

    /**
     * Computes the cumulative distribution function for the Beta distribution.
     *
     * Uses the regularized incomplete beta function I_x(α,β)
     *
     * @param value The value at which to evaluate the CDF
     * @return Cumulative probability P(X ≤ value)
     */
    double getCumulativeProbability(double value) const noexcept;

    /// MLE fit: Newton–Raphson on ψ(α)−ψ(α+β)=ḡ and ψ(β)−ψ(α+β)=h̄,
    /// where ḡ=mean(log x) and h̄=mean(log(1−x)); initialised from MoM.
    void fit(std::span<const double> data) override;
    /// Weighted MLE: same Newton–Raphson system with weighted log-means;
    /// initialised from weighted MoM. Near-zero total weight retains current params.
    void fit(std::span<const double> data, std::span<const double> weights) override;
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }
    [[nodiscard]] std::size_t getNumParameters() const noexcept override { return 2; }

    /**
     * Resets the distribution to default parameters (α = 1.0, β = 1.0).
     * This corresponds to a uniform distribution on [0,1].
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
     * Gets the alpha (α) shape parameter.
     *
     * @return Current alpha value
     */
    double getAlpha() const noexcept { return alpha_; }

    /**
     * Sets the alpha (α) shape parameter.
     *
     * @param alpha New alpha parameter (must be positive)
     * @throws std::invalid_argument if alpha <= 0 or is not finite
     */
    void setAlpha(double alpha) {
        validateParameters(alpha, beta_);
        alpha_ = alpha;
        invalidateCache();
    }

    /**
     * Gets the beta (β) shape parameter.
     *
     * @return Current beta value
     */
    double getBeta() const noexcept { return beta_; }

    /**
     * Sets the beta (β) shape parameter.
     *
     * @param beta New beta parameter (must be positive)
     * @throws std::invalid_argument if beta <= 0 or is not finite
     */
    void setBeta(double beta) {
        validateParameters(alpha_, beta);
        beta_ = beta;
        invalidateCache();
    }

    /**
     * Gets the mean of the distribution.
     * For Beta(α, β), mean = α/(α+β)
     *
     * @return Mean value
     */
    double getMean() const noexcept { return alpha_ / (alpha_ + beta_); }

    /**
     * Gets the variance of the distribution.
     * For Beta(α, β), variance = αβ/((α+β)²(α+β+1))
     *
     * @return Variance value
     */
    double getVariance() const noexcept {
        double sum = alpha_ + beta_;
        return (alpha_ * beta_) / (sum * sum * (sum + 1.0));
    }

    /**
     * Gets the standard deviation of the distribution.
     *
     * @return Standard deviation
     */
    double getStandardDeviation() const noexcept { return std::sqrt(getVariance()); }

    /**
     * Equality operator with tolerance for floating-point comparison
     */
    bool operator==(const BetaDistribution &other) const noexcept {
        const double tolerance = 1e-10;
        return std::abs(alpha_ - other.alpha_) < tolerance &&
               std::abs(beta_ - other.beta_) < tolerance;
    }

    /**
     * Inequality operator
     */
    bool operator!=(const BetaDistribution &other) const noexcept { return !(*this == other); }
};

/**
 * Stream output operator for Beta distribution.
 */
std::ostream &operator<<(std::ostream &os, const libhmm::BetaDistribution &distribution);

} // namespace libhmm
