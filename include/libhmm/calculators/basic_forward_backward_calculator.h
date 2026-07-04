#pragma once

#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "libhmm/calculators/basic_calculator.h"
#include "libhmm/detail/log_utils.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/performance/transcendental_kernels.h"
#include "libhmm/performance/fb_recurrence_policy.h"

namespace libhmm {

/**
 * @brief Log-space Forward-Backward calculator, parameterised on observation type.
 *
 * @tparam Obs  `double` (scalar, v3-compatible) or `ObservationVectorView` (v4 MV).
 *
 * The scalar path calls getBatchLogProbabilities() per state for SIMD-accelerated
 * emission fills.  The multivariate path calls getLogProbability(row_view(obs, t))
 * per (state, timestep) pair.  Transition recurrence (forward/backward sweeps) is
 * identical for both observation types.
 *
 * The precomputed log-emission time-major buffer is exposed via getLogEmitByTime()
 * so BaumWelchTrainer can reuse it without a second emission computation pass.
 *
 * Two explicit instantiations are compiled with LIBHMM_BEST_SIMD_FLAGS:
 *   - src/calculators/forward_backward_calculator.cpp    → BasicForwardBackwardCalculator<double>
 *   - src/calculators/forward_backward_calculator_mv.cpp → BasicForwardBackwardCalculator<ObservationVectorView>
 *
 * The `extern template` declarations below suppress implicit instantiation in
 * consumer TUs.  Include this header and link against the library — do not rely
 * on the compiler generating its own instantiation.
 *
 * Thread safety: const accessor methods (getLogProbability, getLogForwardVariables,
 * getLogBackwardVariables) are safe to call concurrently after compute().
 * compute() itself is not reentrant.
 */
template <typename Obs>
class BasicForwardBackwardCalculator : public BasicCalculator<Obs> {
public:
    using Base = BasicCalculator<Obs>;
    using HmmType = typename Base::HmmType;
    using SeqType = typename Base::SeqType;

    /**
     * @brief Construct and immediately run the forward-backward passes.
     * @param hmm           The HMM (must be validated, states > 0).
     * @param observations  Observation sequence (must be non-empty).
     * @throws std::invalid_argument if observations is empty.
     */
    BasicForwardBackwardCalculator(const HmmType &hmm, const SeqType &observations);

    /** @brief Legacy pointer constructor for backward compatibility. */
    BasicForwardBackwardCalculator(HmmType *hmm, const SeqType &observations);

    // Non-copyable, movable.
    BasicForwardBackwardCalculator(const BasicForwardBackwardCalculator &) = delete;
    BasicForwardBackwardCalculator &operator=(const BasicForwardBackwardCalculator &) = delete;
    BasicForwardBackwardCalculator(BasicForwardBackwardCalculator &&) = default;
    BasicForwardBackwardCalculator &operator=(BasicForwardBackwardCalculator &&) = default;
    ~BasicForwardBackwardCalculator() override = default;

    /// Deleted: passing a temporary observation sequence is UB (dangling reference).
    /// The base-class deletion is bypassed when a temporary binds to `const SeqType&`
    /// in a derived constructor; this overload closes that gap on the derived class.
    BasicForwardBackwardCalculator(const HmmType &, SeqType &&) = delete;
    BasicForwardBackwardCalculator(HmmType *, SeqType &&) = delete;

    /**
     * @brief Re-run with a new observation sequence.
     * Reuses the precomputed log-transition matrices.
     */
    void compute(const SeqType &observations);

    /** @brief Re-run with the current observation sequence. */
    void compute();

    // -------------------------------------------------------------------------
    // Results (log-space)
    // -------------------------------------------------------------------------

    /** @brief log P(O|λ). Returns -∞ before compute() or when P(O|λ) = 0. */
    [[nodiscard]] double getLogProbability() const noexcept { return logProbability_; }

    /** @brief P(O|λ). May underflow to 0 for long sequences; prefer getLogProbability(). */
    [[nodiscard]] double probability() const noexcept { return std::exp(logProbability_); }

    /** @brief logAlpha(t,i) = log P(O₁…Oₜ, qₜ=i|λ). Dimensions: T×N. */
    [[nodiscard]] const Matrix &getLogForwardVariables() const noexcept { return logAlpha_; }

    /** @brief logBeta(t,i)  = log P(Oₜ₊₁…Oₜ|qₜ=i, λ).  Dimensions: T×N. */
    [[nodiscard]] const Matrix &getLogBackwardVariables() const noexcept { return logBeta_; }

    /** @brief Number of HMM states N used by this calculator. */
    [[nodiscard]] std::size_t getNumStates() const noexcept { return numStates_; }

    /**
     * @brief Time-major log-emission buffer produced by the most recent compute().
     *
     * logEmitByTime[t * N + i] = log b_i(O_t).
     * Exposed so BaumWelchTrainer can reuse the buffer without a second emission
     * computation pass over the same observation sequence.
     */
    [[nodiscard]] const std::vector<double> &getLogEmitByTime() const noexcept {
        return logEmitByTime_;
    }

    /**
     * @brief Posterior (marginal) decoding: argmax_i γₜ(i) for each t.
     *
     * Unlike Viterbi, this maximises the marginal state probability at each
     * time step independently.  Requires compute() to have been called.
     */
    [[nodiscard]] StateSequence decodePosterior() const;

    /**
     * @brief Force a specific recurrence kernel for subsequent compute() calls.
     * Pass std::nullopt to restore adaptive policy selection.
     */
    void setRecurrenceModeOverride(std::optional<FbRecurrenceMode> mode) noexcept {
        modeOverride_ = mode;
    }

    /** @brief Currently active recurrence-mode override, if any. */
    [[nodiscard]] std::optional<FbRecurrenceMode> getRecurrenceModeOverride() const noexcept {
        return modeOverride_;
    }

    /** @brief Recurrence mode resolved on the most recent compute() call. */
    [[nodiscard]] FbRecurrenceMode getRecurrenceMode() const noexcept { return currentMode_; }

private:
    static constexpr double LOG_ZERO = detail::LOG_ZERO;

    std::size_t numStates_{0};

    // Precomputed log-transition matrices [N×N], built once at construction.
    Matrix logTrans_;  ///< logTrans_(i,j)  = log a_{ij}
    Matrix logTransT_; ///< logTransT_(j,i) = log a_{ij}  (transposed; used in forward pass)

    // Per-compute() results.
    Matrix logAlpha_; ///< T×N forward variables
    Matrix logBeta_;  ///< T×N backward variables
    double logProbability_{LOG_ZERO};

    // Emission buffers (resized on each compute() call).
    std::vector<double> logEmitBuf_;    ///< state-major: [i*T + t] = log b_i(O_t)
    std::vector<double> logEmitByTime_; ///< time-major:  [t*N + i] = log b_i(O_t)

    FbRecurrenceMode currentMode_{FbRecurrenceMode::Pairwise};
    std::optional<FbRecurrenceMode> modeOverride_;

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    [[nodiscard]] FbRecurrenceMode resolveRecurrenceMode(std::size_t numStates,
                                                         std::size_t seqLen) const noexcept;

    void precomputeLogTransitions();

    /**
     * @brief Fill logEmitBuf_ (state-major) and logEmitByTime_ (time-major) for @p obs.
     * Scalar path: getBatchLogProbabilities() per state (potentially SIMD-accelerated).
     * MV path:     getLogProbability(row_view(obs, t)) per (state, timestep) pair.
     */
    void fillLogEmissions(const SeqType &obs, std::size_t T);

    void computeLogForward(std::size_t T);
    void computeLogBackward(std::size_t T);
    void computeLogForwardPairwise(std::size_t T);
    void computeLogForwardMaxReduce(std::size_t T);
    void computeLogBackwardPairwise(std::size_t T);
    void computeLogBackwardMaxReduce(std::size_t T);

    /// Initialise log-α at t=0: logAlpha[i] = log π_i + log b_i(O_0).
    static void init_log_forward(double *alphaData, const Vector &pi, const double *emitRow0,
                                 std::size_t N) noexcept;

    /// Set log-β at t=T-1 to 0 (log 1 = 0: terminal backward condition).
    static void init_log_backward(double *betaData, std::size_t T, std::size_t N) noexcept;

    /**
     * @brief Shared forward-pass loop.
     * @p inner receives (prevAlphaRow, transposedTransCol, N) and returns the
     * log-sum term for state j at time t.
     */
    template <typename InnerFn>
    static void compute_forward_impl(const HmmType &hmm, std::size_t N, std::size_t T,
                                     const double *logTransT, const double *emitByTime,
                                     Matrix &logAlpha, InnerFn inner);

    /**
     * @brief Shared backward-pass loop.
     * @p inner receives (transRow, emitNextRow, nextBetaRow, N) and returns the
     * log-sum term for state i at time t.
     */
    template <typename InnerFn>
    static void compute_backward_impl(std::size_t N, std::size_t T, const double *logTrans,
                                      const double *emitByTime, Matrix &logBeta, InnerFn inner);

    /** @brief log(exp(a) + exp(b)) — numerically stable. */
    [[nodiscard]] static double logSumExp(double a, double b) noexcept;
};

// =============================================================================
// Inline method definitions
// (All definitions must be visible where the explicit instantiation is provided.)
// =============================================================================

template <typename Obs>
BasicForwardBackwardCalculator<Obs>::BasicForwardBackwardCalculator(const HmmType &hmm,
                                                                    const SeqType &observations)
    : Base(hmm, observations), numStates_(hmm.getNumStatesModern()) {
    if (ObsSeqTraits<Obs>::sequence_length(observations) == 0) {
        throw std::invalid_argument("Observation sequence cannot be empty");
    }
    precomputeLogTransitions();
    compute();
}

template <typename Obs>
BasicForwardBackwardCalculator<Obs>::BasicForwardBackwardCalculator(HmmType *hmm,
                                                                    const SeqType &observations)
    : BasicForwardBackwardCalculator(
          hmm ? *hmm : throw std::invalid_argument("HMM pointer cannot be null"), observations) {}

template <typename Obs>
void BasicForwardBackwardCalculator<Obs>::compute(const SeqType &observations) {
    this->setObservations(observations);
    compute();
}

template <typename Obs>
void BasicForwardBackwardCalculator<Obs>::compute() {
    const SeqType &obs = this->getObservations();
    const std::size_t T = ObsSeqTraits<Obs>::sequence_length(obs);
    if (T == 0) {
        logProbability_ = LOG_ZERO;
        return;
    }

    logAlpha_.resize(T, numStates_);
    logBeta_.resize(T, numStates_);
    logEmitBuf_.resize(T * numStates_);
    logEmitByTime_.resize(T * numStates_);

    fillLogEmissions(obs, T);

    currentMode_ = resolveRecurrenceMode(numStates_, T);
    computeLogForward(T);
    computeLogBackward(T);

    // log P(O|λ) = log-sum-exp over states at the final timestep.
    double lp = LOG_ZERO;
    for (std::size_t i = 0; i < numStates_; ++i) {
        lp = logSumExp(lp, logAlpha_(T - 1, i));
    }
    logProbability_ = lp;
}

template <typename Obs>
void BasicForwardBackwardCalculator<Obs>::fillLogEmissions(const SeqType &obs, std::size_t T) {
    const HmmType &hmm = this->getHmmRef();
    if constexpr (std::is_same_v<Obs, double>) {
        // Scalar: one batch call per state — allows per-distribution SIMD acceleration.
        const std::span<const double> obsSpan(obs.data(), T);
        for (std::size_t i = 0; i < numStates_; ++i) {
            hmm.getDistribution(i).getBatchLogProbabilities(
                obsSpan, std::span<double>(logEmitBuf_.data() + i * T, T));
        }
    } else {
        // Multivariate: per-timestep evaluation for each state via row views.
        for (std::size_t i = 0; i < numStates_; ++i) {
            for (std::size_t t = 0; t < T; ++t) {
                logEmitBuf_[i * T + t] = hmm.getDistribution(i).getLogProbability(row_view(obs, t));
            }
        }
    }
    // Transpose from state-major to time-major layout for contiguous
    // per-time access in the recurrence loops.
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double *stateRow = logEmitBuf_.data() + i * T;
        for (std::size_t t = 0; t < T; ++t) {
            logEmitByTime_[t * numStates_ + i] = stateRow[t];
        }
    }
}

template <typename Obs>
FbRecurrenceMode
BasicForwardBackwardCalculator<Obs>::resolveRecurrenceMode(std::size_t numStates,
                                                           std::size_t seqLen) const noexcept {
#if defined(LIBHMM_EXPERIMENT_FB_MAX_REDUCE)
    (void)numStates;
    (void)seqLen;
    return FbRecurrenceMode::MaxReduce;
#elif defined(LIBHMM_EXPERIMENT_FB_ADAPTIVE_SELECTOR)
    (void)seqLen;
    return (numStates > 2) ? FbRecurrenceMode::MaxReduce : FbRecurrenceMode::Pairwise;
#else
    if (modeOverride_.has_value())
        return *modeOverride_;
    return selectFbRecurrenceMode(numStates, seqLen);
#endif
}

template <typename Obs>
void BasicForwardBackwardCalculator<Obs>::precomputeLogTransitions() {
    this->precompute_log_transitions(this->getHmmRef(), numStates_, logTrans_, logTransT_);
}

template <typename Obs>
void BasicForwardBackwardCalculator<Obs>::init_log_forward(double *alphaData, const Vector &pi,
                                                           const double *emitRow0,
                                                           std::size_t N) noexcept {
    for (std::size_t i = 0; i < N; ++i) {
        alphaData[i] = (pi(i) > 0.0) ? std::log(pi(i)) + emitRow0[i] : LOG_ZERO;
    }
}

template <typename Obs>
void BasicForwardBackwardCalculator<Obs>::init_log_backward(double *betaData, std::size_t T,
                                                            std::size_t N) noexcept {
    double *finalRow = betaData + (T - 1) * N;
    for (std::size_t i = 0; i < N; ++i)
        finalRow[i] = 0.0;
}

template <typename Obs>
template <typename InnerFn>
void BasicForwardBackwardCalculator<Obs>::compute_forward_impl(const HmmType &hmm, std::size_t N,
                                                               std::size_t T,
                                                               const double *logTransT,
                                                               const double *emitByTime,
                                                               Matrix &logAlpha, InnerFn inner) {
    const Vector &pi = hmm.getPi();
    double *alphaData = logAlpha.data();
    init_log_forward(alphaData, pi, emitByTime, N);
    for (std::size_t t = 1; t < T; ++t) {
        const double *prevRow = alphaData + (t - 1) * N;
        double *curRow = alphaData + t * N;
        const double *emitRow = emitByTime + t * N;
        for (std::size_t j = 0; j < N; ++j) {
            curRow[j] = emitRow[j] + inner(prevRow, logTransT + j * N, N);
        }
    }
}

template <typename Obs>
template <typename InnerFn>
void BasicForwardBackwardCalculator<Obs>::compute_backward_impl(std::size_t N, std::size_t T,
                                                                const double *logTrans,
                                                                const double *emitByTime,
                                                                Matrix &logBeta, InnerFn inner) {
    double *betaData = logBeta.data();
    init_log_backward(betaData, T, N);
    if (T > 1) {
        for (std::size_t t = T - 2;; --t) {
            double *betaRow = betaData + t * N;
            const double *nextBetaRow = betaData + (t + 1) * N;
            const double *emitNextRow = emitByTime + (t + 1) * N;
            for (std::size_t i = 0; i < N; ++i) {
                betaRow[i] = inner(logTrans + i * N, emitNextRow, nextBetaRow, N);
            }
            if (t == 0)
                break;
        }
    }
}

template <typename Obs>
void BasicForwardBackwardCalculator<Obs>::computeLogForward(std::size_t T) {
    if (currentMode_ == FbRecurrenceMode::MaxReduce) {
        computeLogForwardMaxReduce(T);
    } else {
        computeLogForwardPairwise(T);
    }
}

template <typename Obs>
void BasicForwardBackwardCalculator<Obs>::computeLogForwardPairwise(std::size_t T) {
    compute_forward_impl(this->getHmmRef(), numStates_, T, logTransT_.data(), logEmitByTime_.data(),
                         logAlpha_,
                         [](const double *prev, const double *transCol, std::size_t n) noexcept {
                             double s = LOG_ZERO;
                             for (std::size_t i = 0; i < n; ++i)
                                 s = logSumExp(s, prev[i] + transCol[i]);
                             return s;
                         });
}

template <typename Obs>
void BasicForwardBackwardCalculator<Obs>::computeLogForwardMaxReduce(std::size_t T) {
    using TK = performance::detail::TranscendentalKernels;
    const std::size_t N = numStates_;
    const Vector &pi = this->getHmmRef().getPi();
    double *alphaData = logAlpha_.data();
    init_log_forward(alphaData, pi, logEmitByTime_.data(), N);

    std::vector<double> maxTerms(N);
    std::vector<double> log1pTerms(N);

    for (std::size_t t = 1; t < T; ++t) {
        const double *prevRow = alphaData + (t - 1) * N;
        double *curRow = alphaData + t * N;
        const double *emitRow = logEmitByTime_.data() + t * N;

        for (std::size_t j = 0; j < N; ++j) {
            const double *transCol = logTransT_.data() + j * N;
            const double m = TK::reduce_max_sum2(prevRow, transCol, N);
            maxTerms[j] = m;
            if (std::isfinite(m)) {
                const double s = TK::sum_exp_sum2_minus_max(prevRow, transCol, N, m);
                log1pTerms[j] = (s > 1.0) ? (s - 1.0) : 0.0;
            } else {
                log1pTerms[j] = 0.0;
            }
        }

        TK::log1p_inplace(std::span<double>(log1pTerms.data(), log1pTerms.size()));

        for (std::size_t j = 0; j < N; ++j) {
            curRow[j] =
                std::isfinite(maxTerms[j]) ? emitRow[j] + maxTerms[j] + log1pTerms[j] : LOG_ZERO;
        }
    }
}

template <typename Obs>
void BasicForwardBackwardCalculator<Obs>::computeLogBackward(std::size_t T) {
    if (currentMode_ == FbRecurrenceMode::MaxReduce) {
        computeLogBackwardMaxReduce(T);
    } else {
        computeLogBackwardPairwise(T);
    }
}

template <typename Obs>
void BasicForwardBackwardCalculator<Obs>::computeLogBackwardPairwise(std::size_t T) {
    compute_backward_impl(numStates_, T, logTrans_.data(), logEmitByTime_.data(), logBeta_,
                          [](const double *transRow, const double *emitNext, const double *nextBeta,
                             std::size_t n) noexcept {
                              double s = LOG_ZERO;
                              for (std::size_t j = 0; j < n; ++j)
                                  s = logSumExp(s, transRow[j] + emitNext[j] + nextBeta[j]);
                              return s;
                          });
}

template <typename Obs>
void BasicForwardBackwardCalculator<Obs>::computeLogBackwardMaxReduce(std::size_t T) {
    using TK = performance::detail::TranscendentalKernels;
    const std::size_t N = numStates_;
    double *betaData = logBeta_.data();
    init_log_backward(betaData, T, N);
    if (T <= 1) {
        return;
    }

    std::vector<double> maxTerms(N);
    std::vector<double> log1pTerms(N);

    for (std::size_t t = T - 2;; --t) {
        double *betaRow = betaData + t * N;
        const double *nextBetaRow = betaData + (t + 1) * N;
        const double *emitNextRow = logEmitByTime_.data() + (t + 1) * N;

        for (std::size_t i = 0; i < N; ++i) {
            const double *transRow = logTrans_.data() + i * N;
            const double m = TK::reduce_max_sum3(transRow, emitNextRow, nextBetaRow, N);
            maxTerms[i] = m;
            if (std::isfinite(m)) {
                const double s =
                    TK::sum_exp_sum3_minus_max(transRow, emitNextRow, nextBetaRow, N, m);
                log1pTerms[i] = (s > 1.0) ? (s - 1.0) : 0.0;
            } else {
                log1pTerms[i] = 0.0;
            }
        }

        TK::log1p_inplace(std::span<double>(log1pTerms.data(), log1pTerms.size()));

        for (std::size_t i = 0; i < N; ++i) {
            betaRow[i] = std::isfinite(maxTerms[i]) ? maxTerms[i] + log1pTerms[i] : LOG_ZERO;
        }

        if (t == 0)
            break;
    }
}

template <typename Obs>
StateSequence BasicForwardBackwardCalculator<Obs>::decodePosterior() const {
    if (!std::isfinite(logProbability_)) {
        throw std::runtime_error("decodePosterior: sequence has zero probability under this model "
                                 "(logP = -inf); posterior decoding is undefined");
    }
    const std::size_t T = logAlpha_.size1();
    StateSequence result(T);
    // Subtract logProbability_ before the argmax.  The constant cancels in the
    // comparison for well-behaved models, but removing it avoids misorderings
    // in degenerate models where some log-posteriors are -inf.
    const double logP = logProbability_;
    for (std::size_t t = 0; t < T; ++t) {
        std::size_t best = 0;
        double bestScore = logAlpha_(t, 0) + logBeta_(t, 0) - logP;
        for (std::size_t i = 1; i < numStates_; ++i) {
            const double score = logAlpha_(t, i) + logBeta_(t, i) - logP;
            if (score > bestScore) {
                bestScore = score;
                best = i;
            }
        }
        result(t) = static_cast<StateIndex>(best);
    }
    return result;
}

template <typename Obs>
double BasicForwardBackwardCalculator<Obs>::logSumExp(double a, double b) noexcept {
    return detail::logSumExp(a, b);
}

// =============================================================================
// Explicit instantiation declarations.
// Suppress implicit instantiation in every consumer TU.
// Definitions are in forward_backward_calculator.cpp (scalar) and
// forward_backward_calculator_mv.cpp (multivariate).
// =============================================================================
extern template class BasicForwardBackwardCalculator<double>;
extern template class BasicForwardBackwardCalculator<ObservationVectorView>;

/// @brief Scalar alias (v3-compatible name).
using ForwardBackwardCalculator = BasicForwardBackwardCalculator<double>;

} // namespace libhmm
