#pragma once

#include "libhmm/calculators/calculator.h"
#include <limits>
#include <vector>

namespace libhmm {

/**
 * @brief Log-space Viterbi calculator.
 *
 * Finds the most likely state sequence for an observation sequence using the
 * Viterbi algorithm in log-space, eliminating overflow for long sequences.
 *
 * Emission probabilities are obtained via EmissionDistribution::getBatchLogProbabilities(),
 * allowing per-distribution SIMD acceleration transparently.
 *
 * Log-transition matrices are precomputed at construction and reused across
 * decode() calls on the same HMM.
 */
class ViterbiCalculator : public Calculator {
public:
    /**
     * @brief Construct and immediately run Viterbi decoding.
     * @param hmm          The HMM (must be validated).
     * @param observations Observation sequence (must be non-empty).
     * @throws std::invalid_argument if observations is empty.
     */
    ViterbiCalculator(const Hmm& hmm, const ObservationSet& observations);

    /** Legacy pointer constructor for backward compatibility. */
    ViterbiCalculator(Hmm* hmm, const ObservationSet& observations);

    ViterbiCalculator(const ViterbiCalculator&) = delete;
    ViterbiCalculator& operator=(const ViterbiCalculator&) = delete;
    ViterbiCalculator(ViterbiCalculator&&) = default;
    ViterbiCalculator& operator=(ViterbiCalculator&&) = default;
    ~ViterbiCalculator() override = default;

    /**
     * @brief Decode the most likely state sequence.
     * Runs the Viterbi algorithm if it hasn't been run yet, or re-runs after
     * a new observation sequence is set via setObservations().
     * @return Most likely state sequence (length T).
     */
    [[nodiscard]] StateSequence decode();

    /**
     * @brief Log-probability of the most likely path: log P(O, q* | λ).
     * Valid after decode() has been called.
     */
    [[nodiscard]] double getLogProbability() const noexcept { return logProbability_; }

    /**
     * @brief The decoded state sequence (same as returned by decode()).
     * Valid after decode() has been called.
     */
    [[nodiscard]] const StateSequence& getStateSequence() const noexcept {
        return sequence_;
    }

    /** Number of HMM states. */
    [[nodiscard]] std::size_t getNumStates() const noexcept { return numStates_; }

private:
    std::size_t numStates_{0};

    // Precomputed log-transition matrix [N x N]
    Matrix logTrans_;

    // Viterbi trellis: logDelta(t,i) = max log-prob path ending at state i at time t
    Matrix logDelta_;

    // Backtrack pointers: psi(t,i) = arg max_j [logDelta(t-1,j) + logTrans(j,i)]
    std::vector<std::vector<int>> psi_;

    // Result
    StateSequence sequence_;
    double logProbability_{-std::numeric_limits<double>::infinity()};

    // Per-state emission buffer
    mutable std::vector<double> logEmitBuf_;

    void precomputeLogTransitions();
    void runViterbi();
    void backtrack();
};

} // namespace libhmm
