#pragma once

#include "libhmm/calculators/calculator.h"
#include <limits>
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

private:
    std::size_t numStates_{0};

    // Precomputed log-transition matrix [N x N]: logTrans_(i,j) = log a_{ij}
    Matrix logTrans_;

    // Results
    Matrix logAlpha_; // T x N
    Matrix logBeta_;  // T x N
    double logProbability_{-std::numeric_limits<double>::infinity()};

    // Per-state log-emission buffer reused each timestep [T x N, row-major].
    // Allocated once; filled by getBatchLogProbabilities per state.
    mutable std::vector<double> logEmitBuf_;
    bool useMaxReduceRecurrence_{false};

    [[nodiscard]] static bool shouldUseMaxReduceRecurrence(std::size_t numStates,
                                                           std::size_t sequenceLength) noexcept;
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