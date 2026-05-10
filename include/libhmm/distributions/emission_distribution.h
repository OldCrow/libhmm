#pragma once

#include <span>
#include <string>

namespace libhmm {

/**
 * @brief Abstract interface for HMM emission distributions.
 *
 * Replaces ProbabilityDistribution. Narrower than a general statistical
 * interface — exposes only what the HMM engine actually calls:
 *   - Const-correct scalar PDF/logPDF evaluation
 *   - In-place batch logPDF over an observation sequence (SIMD-ready)
 *   - Unweighted fit (initialization, Viterbi training)
 *   - Weighted fit (Baum-Welch M-step, unnormalized γ weights)
 *   - Reset and metadata
 *
 * All observation values are double. Discrete observations are
 * integer-encoded as double (e.g., {0,1,2,3} for a 4-symbol alphabet).
 */
class EmissionDistribution {
public:
    virtual ~EmissionDistribution() = default;

    EmissionDistribution() = default;
    EmissionDistribution(const EmissionDistribution &) = default;
    EmissionDistribution &operator=(const EmissionDistribution &) = default;
    EmissionDistribution(EmissionDistribution &&) = default;
    EmissionDistribution &operator=(EmissionDistribution &&) = default;

    // =========================================================================
    // Scalar evaluation
    // =========================================================================

    /**
     * @brief Probability density/mass at x.
     * Used by non-SIMD calculators and for small models.
     */
    [[nodiscard]] virtual double getProbability(double x) const = 0;

    /**
     * @brief Log probability density/mass at x.
     * Numerically preferred in log-space calculator variants.
     */
    [[nodiscard]] virtual double getLogProbability(double x) const = 0;

    // =========================================================================
    // Batch evaluation (in-place, no allocation)
    // =========================================================================

    /**
     * @brief Compute log probabilities for a sequence of observations.
     *
     * Writes log p(observations[i]) into out[i] for all i.
     * The default implementation is a scalar loop calling getLogProbability().
     * Override for SIMD vectorization.
     *
     * Precondition: observations.size() == out.size()
     */
    virtual void getBatchLogProbabilities(std::span<const double> observations,
                                          std::span<double> out) const = 0;

    // =========================================================================
    // Parameter estimation
    // =========================================================================

    /**
     * @brief Fit parameters to unweighted data (MLE).
     * Used for initialization and Viterbi training.
     */
    virtual void fit(std::span<const double> data) = 0;

    /**
     * @brief Fit parameters to weighted data (weighted MLE).
     * Used by the Baum-Welch M-step.
     *
     * Weights are unnormalized γ values from the E-step. Each distribution
     * normalizes by sum(weights) internally. If sum(weights) is near zero
     * (state rarely visited), the implementation should call reset().
     *
     * Precondition: data.size() == weights.size()
     */
    virtual void fit(std::span<const double> data, std::span<const double> weights) = 0;

    /**
     * @brief Reset parameters to distribution-specific defaults.
     */
    virtual void reset() noexcept = 0;

    // =========================================================================
    // Metadata
    // =========================================================================

    /** @brief Human-readable string representation.  Delegates to to_json(). */
    [[nodiscard]] virtual std::string toString() const { return to_json(); }

    /**
     * @brief Serialise to a compact JSON object string.
     *
     * Must produce output that round-trips exactly through the matching
     * static from_json() factory registered in src/io/hmm_json.cpp.
     * Use json::write_distribution() from libhmm/io/json_utils.h.
     */
    [[nodiscard]] virtual std::string to_json() const = 0;

    /** @brief Returns true for discrete (PMF) distributions, false for continuous (PDF). */
    [[nodiscard]] virtual bool isDiscrete() const noexcept = 0;

    /**
     * @brief Number of free parameters in this distribution.
     *
     * Used by model-selection utilities (AIC, BIC, AICc) to compute the
     * total parameter count for an HMM. Each concrete distribution reports
     * the number of independently estimated scalar parameters; for
     * DiscreteDistribution with K symbols this is K-1 (simplex constraint).
     */
    [[nodiscard]] virtual std::size_t getNumParameters() const noexcept = 0;
};

} // namespace libhmm
