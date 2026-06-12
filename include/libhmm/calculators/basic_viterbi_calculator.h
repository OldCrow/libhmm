#pragma once

#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "libhmm/calculators/basic_calculator.h"
#include "libhmm/linalg/linalg_types.h"

namespace libhmm {

/**
 * @brief Log-space Viterbi calculator, parameterised on observation type.
 *
 * @tparam Obs  `double` (scalar, v3-compatible) or `ObservationVectorView` (v4 MV).
 *
 * Finds the most likely state sequence via the Viterbi algorithm in log-space.
 * The scalar path calls getBatchLogProbabilities() for SIMD-accelerated emission
 * fills.  The multivariate path calls getLogProbability(row_view(obs, t)) per
 * (state, timestep) pair.  The Viterbi DP recurrence and backtrack are
 * observation-type-independent.
 *
 * Two explicit instantiations are provided:
 *   - src/calculators/viterbi_calculator.cpp    → BasicViterbiCalculator<double>
 *   - src/calculators/viterbi_calculator_mv.cpp → BasicViterbiCalculator<ObservationVectorView>
 */
template <typename Obs>
class BasicViterbiCalculator : public BasicCalculator<Obs> {
public:
    using Base = BasicCalculator<Obs>;
    using HmmType = typename Base::HmmType;
    using SeqType = typename Base::SeqType;

    /**
     * @brief Construct and immediately run Viterbi decoding.
     * @param hmm          The HMM (must be validated).
     * @param observations Observation sequence (must be non-empty).
     * @throws std::invalid_argument if observations is empty.
     */
    BasicViterbiCalculator(const HmmType &hmm, const SeqType &observations);

    /** @brief Legacy pointer constructor for backward compatibility. */
    BasicViterbiCalculator(HmmType *hmm, const SeqType &observations);

    // Non-copyable, movable.
    BasicViterbiCalculator(const BasicViterbiCalculator &) = delete;
    BasicViterbiCalculator &operator=(const BasicViterbiCalculator &) = delete;
    BasicViterbiCalculator(BasicViterbiCalculator &&) = default;
    BasicViterbiCalculator &operator=(BasicViterbiCalculator &&) = default;
    ~BasicViterbiCalculator() override = default;

    /**
     * @brief Run (or re-run) Viterbi decoding on the current observation sequence.
     * @return The most likely state sequence (length T).
     */
    [[nodiscard]] StateSequence decode();

    /** @brief log P(O, q* | λ) — log-probability of the best path. Valid after decode(). */
    [[nodiscard]] double getLogProbability() const noexcept { return logProbability_; }

    /** @brief The decoded state sequence. Valid after decode(). */
    [[nodiscard]] const StateSequence &getStateSequence() const noexcept { return sequence_; }

    /** @brief Number of HMM states. */
    [[nodiscard]] std::size_t getNumStates() const noexcept { return numStates_; }

private:
    static constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();

    std::size_t numStates_{0};

    // Precomputed log-transition matrices [N×N].
    Matrix logTrans_;  ///< logTrans_(i,j)  = log a_{ij}
    Matrix logTransT_; ///< logTransT_(j,i) = log a_{ij}  (transposed; used in DP)

    // Viterbi trellis: logDelta(t,i) = max log-prob path ending at state i at time t.
    Matrix logDelta_;

    // Backtrack pointers (time-major contiguous):
    // psi_[t * N + j] = argmax_i [logDelta(t-1,i) + logTrans(i,j)]
    std::vector<int> psi_;

    // Decoded result.
    StateSequence sequence_;
    double logProbability_{LOG_ZERO};

    // Emission buffers (resized on each decode() call).
    std::vector<double> logEmitBuf_;    ///< state-major: [i*T + t] = log b_i(O_t)
    std::vector<double> logEmitByTime_; ///< time-major:  [t*N + i] = log b_i(O_t)

    void precomputeLogTransitions();

    /**
     * @brief Fill logEmitBuf_ (state-major) and logEmitByTime_ (time-major) for @p obs.
     * Scalar path: getBatchLogProbabilities() per state (potentially SIMD-accelerated).
     * MV path:     getLogProbability(row_view(obs, t)) per (state, timestep) pair.
     */
    void fillLogEmissions(const SeqType &obs, std::size_t T);

    /// Viterbi DP forward pass; sets logDelta_, psi_, sequence_(T-1), logProbability_.
    void runViterbi(std::size_t T);

    /// Backtrack through psi_ to fill sequence_(0..T-2).
    void backtrack(std::size_t T);
};

// =============================================================================
// Inline method definitions
// =============================================================================

template <typename Obs>
BasicViterbiCalculator<Obs>::BasicViterbiCalculator(const HmmType &hmm, const SeqType &observations)
    : Base(hmm, observations), numStates_(static_cast<std::size_t>(hmm.getNumStates())) {
    if (ObsSeqTraits<Obs>::sequence_length(observations) == 0) {
        throw std::invalid_argument("Observation sequence cannot be empty");
    }
    precomputeLogTransitions();
    static_cast<void>(decode());
}

template <typename Obs>
BasicViterbiCalculator<Obs>::BasicViterbiCalculator(HmmType *hmm, const SeqType &observations)
    : BasicViterbiCalculator(hmm ? *hmm : throw std::invalid_argument("HMM pointer cannot be null"),
                             observations) {}

template <typename Obs>
StateSequence BasicViterbiCalculator<Obs>::decode() {
    const SeqType &obs = this->getObservations();
    const std::size_t T = ObsSeqTraits<Obs>::sequence_length(obs);
    if (T == 0) {
        logProbability_ = LOG_ZERO;
        sequence_.resize(0);
        return sequence_;
    }

    logEmitBuf_.resize(T * numStates_);
    logEmitByTime_.resize(T * numStates_);

    fillLogEmissions(obs, T);
    runViterbi(T);
    backtrack(T);
    return sequence_;
}

template <typename Obs>
void BasicViterbiCalculator<Obs>::fillLogEmissions(const SeqType &obs, std::size_t T) {
    const HmmType &hmm = this->getHmmRef();
    if constexpr (std::is_same_v<Obs, double>) {
        const std::span<const double> obsSpan(obs.data(), T);
        for (std::size_t i = 0; i < numStates_; ++i) {
            hmm.getDistribution(i).getBatchLogProbabilities(
                obsSpan, std::span<double>(logEmitBuf_.data() + i * T, T));
        }
    } else {
        for (std::size_t i = 0; i < numStates_; ++i) {
            for (std::size_t t = 0; t < T; ++t) {
                logEmitBuf_[i * T + t] = hmm.getDistribution(i).getLogProbability(row_view(obs, t));
            }
        }
    }
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double *stateRow = logEmitBuf_.data() + i * T;
        for (std::size_t t = 0; t < T; ++t) {
            logEmitByTime_[t * numStates_ + i] = stateRow[t];
        }
    }
}

template <typename Obs>
void BasicViterbiCalculator<Obs>::precomputeLogTransitions() {
    this->precompute_log_transitions(this->getHmmRef(), numStates_, logTrans_, logTransT_);
}

template <typename Obs>
void BasicViterbiCalculator<Obs>::runViterbi(std::size_t T) {
    const HmmType &hmm = this->getHmmRef();
    const Vector &pi = hmm.getPi();

    logDelta_.resize(T, numStates_);
    psi_.assign(T * numStates_, 0);

    const double *logTransTData = logTransT_.data();
    const double *logEmitByTimeData = logEmitByTime_.data();
    double *logDeltaData = logDelta_.data();
    const std::size_t N = numStates_;

    // t = 0: initialise from π and first emission row.
    const double *emitRow0 = logEmitByTimeData;
    for (std::size_t i = 0; i < N; ++i) {
        const double logPi = (pi(i) > 0.0) ? std::log(pi(i)) : LOG_ZERO;
        logDeltaData[i] = logPi + emitRow0[i];
    }

    // t > 0: DP recursion.
    for (std::size_t t = 1; t < T; ++t) {
        const double *prevRow = logDeltaData + (t - 1) * N;
        double *curRow = logDeltaData + t * N;
        const double *emitRow = logEmitByTimeData + t * N;
        for (std::size_t j = 0; j < N; ++j) {
            double bestVal = LOG_ZERO;
            int bestFrom = 0;
            const double *transCol = logTransTData + j * N;
            for (std::size_t i = 0; i < N; ++i) {
                const double val = prevRow[i] + transCol[i];
                if (val > bestVal) {
                    bestVal = val;
                    bestFrom = static_cast<int>(i);
                }
            }
            curRow[j] = bestVal + emitRow[j];
            psi_[t * N + j] = bestFrom;
        }
    }

    // Termination: best state at T-1.
    double bestVal = LOG_ZERO;
    int bestLast = 0;
    const double *finalRow = logDeltaData + (T - 1) * N;
    for (std::size_t i = 0; i < N; ++i) {
        if (finalRow[i] > bestVal) {
            bestVal = finalRow[i];
            bestLast = static_cast<int>(i);
        }
    }
    logProbability_ = bestVal;
    sequence_.resize(T);
    sequence_(T - 1) = bestLast;
}

template <typename Obs>
void BasicViterbiCalculator<Obs>::backtrack(std::size_t T) {
    if (T <= 1)
        return;
    const std::size_t N = numStates_;
    for (std::size_t t = T - 2;; --t) {
        sequence_(t) = psi_[(t + 1) * N + static_cast<std::size_t>(sequence_(t + 1))];
        if (t == 0)
            break;
    }
}

// =============================================================================
// Explicit instantiation declarations.
// =============================================================================
extern template class BasicViterbiCalculator<double>;
extern template class BasicViterbiCalculator<ObservationVectorView>;

/// @brief Scalar alias (v3-compatible name).
using ViterbiCalculator = BasicViterbiCalculator<double>;

} // namespace libhmm
