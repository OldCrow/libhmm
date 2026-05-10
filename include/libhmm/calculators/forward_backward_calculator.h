#pragma once

#include "libhmm/calculators/calculator.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/performance/fb_recurrence_policy.h"
#include <limits>
#include <optional>
#include <vector>

namespace libhmm {

/**
 * @brief Log-space Forward-Backward calculator.
 *
 * Computes forward (alpha) and backward (beta) variables entirely in log-space,
 * eliminating overflow for arbitrarily long observation sequences.
 *
 * Emission probabilities are obtained via EmissionDistribution::getBatchLogProbabilities(),
 * which allows per-distribution SIMD acceleration without any changes here.
 *
 * Log-transition matrices are precomputed once at construction and reused across
 * compute() calls, amortising the log() cost when the same calculator is reused
 * for multiple observation sequences on the same HMM.
 *
 * Thread safety: const methods (getLogProbability, getLogForwardVariables,
 * getLogBackwardVariables) are safe to call concurrently after compute().
 * compute() itself is not reentrant.
 */
class ForwardBackwardCalculator : public Calculator {
public:
    /**
     * @brief Construct and immediately run the forward-backward passes.
     * @param hmm    The HMM (must be validated).
     * @param observations  Observation sequence (must be non-empty).
     * @throws std::invalid_argument if observations is empty.
     */
    ForwardBackwardCalculator(const Hmm &hmm, const ObservationSet &observations);

    /** Legacy pointer constructor for backward compatibility. */
    ForwardBackwardCalculator(Hmm *hmm, const ObservationSet &observations);

    // Non-copyable, movable.
    ForwardBackwardCalculator(const ForwardBackwardCalculator &) = delete;
    ForwardBackwardCalculator &operator=(const ForwardBackwardCalculator &) = delete;
    ForwardBackwardCalculator(ForwardBackwardCalculator &&) = default;
    ForwardBackwardCalculator &operator=(ForwardBackwardCalculator &&) = default;
    ~ForwardBackwardCalculator() override = default;

    /**
     * @brief Re-run computation with a new observation sequence.
     * Reuses the precomputed log-transition matrix.
     */
    void compute(const ObservationSet &observations);

    /** Re-run with the current observation sequence (e.g. after HMM parameters change). */
    void compute();

    // -------------------------------------------------------------------------
    // Results (all in log-space)
    // -------------------------------------------------------------------------

    /**
     * @brief Log-probability of the observation sequence: log P(O | λ).
     * Returns -infinity if compute() has not been called or the sequence has
     * zero probability.
     */
    [[nodiscard]] double getLogProbability() const noexcept { return logProbability_; }

    /**
     * @brief Probability of the observation sequence P(O | λ).
     * May underflow to 0.0 for long sequences; prefer getLogProbability().
     */
    [[nodiscard]] double probability() const noexcept { return std::exp(logProbability_); }

    /**
     * @brief Log forward variables: logAlpha(t, i) = log P(O_1..O_t, q_t=i | λ).
     * Matrix dimensions: T x N (observation-length by number of states).
     */
    [[nodiscard]] const Matrix &getLogForwardVariables() const noexcept { return logAlpha_; }

    /**
     * @brief Log backward variables: logBeta(t, i) = log P(O_{t+1}..O_T | q_t=i, λ).
     * Matrix dimensions: T x N.
     */
    [[nodiscard]] const Matrix &getLogBackwardVariables() const noexcept { return logBeta_; }

    /** Number of HMM states used by this calculator. */
    [[nodiscard]] std::size_t getNumStates() const noexcept { return numStates_; }

    /**
     * @brief Posterior (marginal) decoding: most probable state at each time step.
     *
     * Returns argmax_i γ_t(i) for each t, where
     *   γ_t(i) = P(q_t=i | O, λ) ∝ exp(logAlpha_(t,i) + logBeta_(t,i)).
     *
     * Unlike Viterbi, this maximises the marginal state probability at each
     * time step independently rather than the joint sequence probability.
     * The result is optimal in terms of per-state error rate.
     *
     * Requires compute() to have been called first.
     */
    [[nodiscard]] StateSequence decodePosterior() const;

    /**
     * @brief Force a specific recurrence kernel for subsequent compute() calls.
     *
     * Pass `std::nullopt` to clear the override and return to adaptive policy.
     * The override takes precedence over the static policy bins, but is itself
     * superseded by the compile-time `LIBHMM_EXPERIMENT_FB_MAX_REDUCE` and
     * `LIBHMM_EXPERIMENT_FB_ADAPTIVE_SELECTOR` forcers when those are defined.
     */
    void setRecurrenceModeOverride(std::optional<FbRecurrenceMode> mode) noexcept {
        modeOverride_ = mode;
    }

    /** Currently active recurrence-mode override, if any. */
    [[nodiscard]] std::optional<FbRecurrenceMode> getRecurrenceModeOverride() const noexcept {
        return modeOverride_;
    }

    /** Recurrence mode resolved on the most recent compute() call. */
    [[nodiscard]] FbRecurrenceMode getRecurrenceMode() const noexcept { return currentMode_; }

private:
    std::size_t numStates_{0};

    // Precomputed log-transition matrix [N x N]: logTrans_(i,j) = log a_{ij}
    Matrix logTrans_;
    // Transposed transition matrix [N x N]: logTransT_(j,i) = log a_{ij}
    Matrix logTransT_;

    // Results
    Matrix logAlpha_; // T x N
    Matrix logBeta_;  // T x N
    double logProbability_{-std::numeric_limits<double>::infinity()};

    // State-major log-emission buffer: logEmitBuf_[i * T + t] = log b_i(O_t).
    // Filled directly by getBatchLogProbabilities per state.
    std::vector<double> logEmitBuf_;
    // Time-major emission buffer: logEmitByTime_[t * N + i] = log b_i(O_t).
    // Derived from logEmitBuf_ for contiguous per-time access in recurrences.
    std::vector<double> logEmitByTime_;
    // Recurrence kernel resolved by the policy + override pipeline on the most
    // recent compute() call. Defaults to Pairwise (the comparator-safe choice).
    FbRecurrenceMode currentMode_{FbRecurrenceMode::Pairwise};
    // Optional per-instance override (Phase A4). Set via setRecurrenceModeOverride().
    std::optional<FbRecurrenceMode> modeOverride_;

    [[nodiscard]] FbRecurrenceMode resolveRecurrenceMode(std::size_t numStates,
                                                         std::size_t sequenceLength) const noexcept;
    void precomputeLogTransitions();
    void computeLogForward();
    void computeLogBackward();
    void computeLogForwardPairwise();
    void computeLogForwardMaxReduce();
    void computeLogBackwardPairwise();
    void computeLogBackwardMaxReduce();

    /** log-sum-exp of two log-space values: log(exp(a) + exp(b)). */
    static double logSumExp(double a, double b) noexcept;
};

} // namespace libhmm
