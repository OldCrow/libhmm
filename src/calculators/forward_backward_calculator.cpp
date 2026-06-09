#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/hmm.h"
#include "libhmm/performance/transcendental_kernels.h"
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

namespace libhmm {

namespace {
constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();

/// Initialises log-α at t=0 from log-π and the first row of log-emissions.
void init_log_forward(double *alphaData, const Vector &pi, const double *emitRow0,
                      std::size_t N) noexcept {
    for (std::size_t i = 0; i < N; ++i)
        alphaData[i] = (pi(i) > 0.0) ? std::log(pi(i)) + emitRow0[i] : LOG_ZERO;
}

/// Sets log-β at t=T-1 to 0 (backward terminal condition: log(1) = 0).
void init_log_backward(double *betaData, std::size_t T, std::size_t N) noexcept {
    double *finalRow = betaData + (T - 1) * N;
    for (std::size_t i = 0; i < N; ++i)
        finalRow[i] = 0.0;
}

/// Shared forward-pass loop. Sets up pointers, calls init_log_forward, then
/// iterates t=1..T-1. The per-j inner computation is delegated to @p inner,
/// which receives (prevAlphaRow, transCol, N) and returns the log-sum term.
///
/// This template is instantiated twice — once for Pairwise, once for
/// MaxReduce — in the same translation unit, so no explicit instantiation
/// or header declaration is required.
template <typename InnerFn>
void compute_forward(const Hmm &hmm, std::size_t N, const ObservationSet &observations,
                     const double *logTransT, const double *emitByTime, Matrix &logAlpha,
                     InnerFn inner) {
    const Vector &pi = hmm.getPi();
    const std::size_t T = observations.size();
    double *alphaData = logAlpha.data();
    init_log_forward(alphaData, pi, emitByTime, N);
    for (std::size_t t = 1; t < T; ++t) {
        const double *prevRow = alphaData + (t - 1) * N;
        double *row = alphaData + t * N;
        const double *emitRow = emitByTime + t * N;
        for (std::size_t j = 0; j < N; ++j)
            row[j] = emitRow[j] + inner(prevRow, logTransT + j * N, N);
    }
}

/// Shared backward-pass loop. Sets up pointers, calls init_log_backward,
/// then iterates t=T-2..0. The per-i inner computation is delegated to
/// @p inner, which receives (transRow, emitNextRow, nextBetaRow, N) and
/// returns the log-sum term.
template <typename InnerFn>
void compute_backward(std::size_t N, const ObservationSet &observations, const double *logTrans,
                      const double *emitByTime, Matrix &logBeta, InnerFn inner) {
    const std::size_t T = observations.size();
    double *betaData = logBeta.data();
    init_log_backward(betaData, T, N);
    if (T > 1) {
        for (std::size_t t = T - 2;; --t) {
            double *betaRow = betaData + t * N;
            const double *nextBetaRow = betaData + (t + 1) * N;
            const double *emitNextRow = emitByTime + (t + 1) * N;
            for (std::size_t i = 0; i < N; ++i)
                betaRow[i] = inner(logTrans + i * N, emitNextRow, nextBetaRow, N);
            if (t == 0)
                break;
        }
    }
}

} // namespace

FbRecurrenceMode
ForwardBackwardCalculator::resolveRecurrenceMode(const std::size_t numStates,
                                                 const std::size_t sequenceLength) const noexcept {
#if defined(LIBHMM_EXPERIMENT_FB_MAX_REDUCE)
    // Compile-time forcer: highest priority. Preserves benchmark-build contract.
    (void)numStates;
    (void)sequenceLength;
    return FbRecurrenceMode::MaxReduce;
#elif defined(LIBHMM_EXPERIMENT_FB_ADAPTIVE_SELECTOR)
    // Legacy adaptive forcer: simple N>2 cutoff. Preserves benchmark-build contract.
    (void)sequenceLength;
    return (numStates > 2) ? FbRecurrenceMode::MaxReduce : FbRecurrenceMode::Pairwise;
#else
    if (modeOverride_.has_value()) {
        return *modeOverride_;
    }
    return selectFbRecurrenceMode(numStates, sequenceLength);
#endif
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

ForwardBackwardCalculator::ForwardBackwardCalculator(const Hmm &hmm,
                                                     const ObservationSet &observations)
    : Calculator(hmm, observations), numStates_(static_cast<std::size_t>(hmm.getNumStates())) {
    if (observations.empty()) {
        throw std::invalid_argument("Observation sequence cannot be empty");
    }
    precomputeLogTransitions();
    compute();
}

ForwardBackwardCalculator::ForwardBackwardCalculator(Hmm *hmm, const ObservationSet &observations)
    : ForwardBackwardCalculator(
          hmm ? *hmm : throw std::invalid_argument("HMM pointer cannot be null"), observations) {}

// ---------------------------------------------------------------------------
// Public compute interface
// ---------------------------------------------------------------------------

void ForwardBackwardCalculator::compute(const ObservationSet &observations) {
    setObservations(observations);  // rebind reference
    compute();
}

void ForwardBackwardCalculator::compute() {
    const ObservationSet &obs = getObservations();
    const std::size_t T = obs.size();
    if (T == 0) {
        logProbability_ = LOG_ZERO;
        return;
    }

    // Allocate/resize result matrices.
    logAlpha_.resize(T, numStates_);
    logBeta_.resize(T, numStates_);

    // Build state-major log-emission buffer: logEmitBuf_[i * T + t] = log b_i(O_t).
    // Then derive shared time-major layout: logEmitByTime_[t * N + i] = log b_i(O_t).
    logEmitBuf_.resize(T * numStates_);
    logEmitByTime_.resize(T * numStates_);
    const std::span<const double> obsSpan(obs.data(), T);

    const Hmm &hmm = getHmmRef();
    for (std::size_t i = 0; i < numStates_; ++i) {
        hmm.getDistribution(i).getBatchLogProbabilities(
            obsSpan, std::span<double>(logEmitBuf_.data() + i * T, T));
    }
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double *stateRow = logEmitBuf_.data() + i * T;
        for (std::size_t t = 0; t < T; ++t) {
            logEmitByTime_[t * numStates_ + i] = stateRow[t];
        }
    }

    // Resolve recurrence mode per the compile-time forcer / instance override /
    // static policy pipeline.
    currentMode_ = resolveRecurrenceMode(numStates_, T);

    computeLogForward();
    computeLogBackward();

    // log P(O|lambda) = log-sum-exp over states at final timestep.
    double lp = LOG_ZERO;
    for (std::size_t i = 0; i < numStates_; ++i) {
        lp = logSumExp(lp, logAlpha_(T - 1, i));
    }
    logProbability_ = lp;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void ForwardBackwardCalculator::precomputeLogTransitions() {
    precompute_log_transitions(getHmmRef(), numStates_, logTrans_, logTransT_);
}

void ForwardBackwardCalculator::computeLogForward() {
    if (currentMode_ == FbRecurrenceMode::MaxReduce) {
        computeLogForwardMaxReduce();
        return;
    }
    computeLogForwardPairwise();
}

void ForwardBackwardCalculator::computeLogForwardPairwise() {
    compute_forward(getHmmRef(), numStates_, getObservations(), logTransT_.data(),
                    logEmitByTime_.data(), logAlpha_,
                    [](const double *prev, const double *transCol, std::size_t n) {
                        double s = LOG_ZERO;
                        for (std::size_t i = 0; i < n; ++i)
                            s = logSumExp(s, prev[i] + transCol[i]);
                        return s;
                    });
}

void ForwardBackwardCalculator::computeLogForwardMaxReduce() {
    using TK = performance::detail::TranscendentalKernels;
    compute_forward(getHmmRef(), numStates_, getObservations(), logTransT_.data(),
                    logEmitByTime_.data(), logAlpha_,
                    [](const double *prev, const double *transCol, std::size_t n) {
                        const double m = TK::reduce_max_sum2(prev, transCol, n);
                        if (!std::isfinite(m))
                            return LOG_ZERO;
                        const double s = TK::sum_exp_sum2_minus_max(prev, transCol, n, m);
                        return (s > 0.0) ? m + std::log(s) : LOG_ZERO;
                    });
}

void ForwardBackwardCalculator::computeLogBackward() {
    if (currentMode_ == FbRecurrenceMode::MaxReduce) {
        computeLogBackwardMaxReduce();
        return;
    }
    computeLogBackwardPairwise();
}

void ForwardBackwardCalculator::computeLogBackwardPairwise() {
    compute_backward(
        numStates_, getObservations(), logTrans_.data(), logEmitByTime_.data(), logBeta_,
        [](const double *transRow, const double *emitNext, const double *nextBeta, std::size_t n) {
            double s = LOG_ZERO;
            for (std::size_t j = 0; j < n; ++j)
                s = logSumExp(s, transRow[j] + emitNext[j] + nextBeta[j]);
            return s;
        });
}

void ForwardBackwardCalculator::computeLogBackwardMaxReduce() {
    using TK = performance::detail::TranscendentalKernels;
    compute_backward(
        numStates_, getObservations(), logTrans_.data(), logEmitByTime_.data(), logBeta_,
        [](const double *transRow, const double *emitNext, const double *nextBeta, std::size_t n) {
            const double m = TK::reduce_max_sum3(transRow, emitNext, nextBeta, n);
            if (!std::isfinite(m))
                return LOG_ZERO;
            const double s = TK::sum_exp_sum3_minus_max(transRow, emitNext, nextBeta, n, m);
            return (s > 0.0) ? m + std::log(s) : LOG_ZERO;
        });
}

StateSequence ForwardBackwardCalculator::decodePosterior() const {
    const std::size_t T = logAlpha_.size1();
    StateSequence result(T);
    for (std::size_t t = 0; t < T; ++t) {
        std::size_t best = 0;
        double bestScore = logAlpha_(t, 0) + logBeta_(t, 0);
        for (std::size_t i = 1; i < numStates_; ++i) {
            const double score = logAlpha_(t, i) + logBeta_(t, i);
            if (score > bestScore) {
                bestScore = score;
                best = i;
            }
        }
        result(t) = static_cast<StateIndex>(best);
    }
    return result;
}

// Numerically stable log(exp(a) + exp(b)).
double ForwardBackwardCalculator::logSumExp(double a, double b) noexcept {
    if (a == LOG_ZERO) {
        return b;
    }
    if (b == LOG_ZERO) {
        return a;
    }
    if (a > b) {
        return a + std::log1p(std::exp(b - a));
    }
    return b + std::log1p(std::exp(a - b));
}

} // namespace libhmm
