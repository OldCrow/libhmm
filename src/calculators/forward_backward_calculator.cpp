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
    observations_ = observations;
    compute();
}

void ForwardBackwardCalculator::compute() {
    const std::size_t T = observations_.size();
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
    const std::span<const double> obsSpan(observations_.data(), T);

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
    const Hmm &hmm = getHmmRef();
    const Vector &pi = hmm.getPi();
    const std::size_t T = observations_.size();
    const std::size_t N = numStates_;
    const double *logTransTData = logTransT_.data();
    const double *emitByTimeData = logEmitByTime_.data();
    double *alphaData = logAlpha_.data();

    init_log_forward(alphaData, pi, emitByTimeData, N);

    // t > 0.
    for (std::size_t t = 1; t < T; ++t) {
        const double *prevAlphaRow = alphaData + (t - 1) * N;
        double *alphaRow = alphaData + t * N;
        const double *emitRow = emitByTimeData + t * N;
        for (std::size_t j = 0; j < N; ++j) {
            const double *transCol = logTransTData + j * N;
            double logSum = LOG_ZERO;
            for (std::size_t i = 0; i < N; ++i) {
                logSum = logSumExp(logSum, prevAlphaRow[i] + transCol[i]);
            }
            alphaRow[j] = emitRow[j] + logSum;
        }
    }
}

void ForwardBackwardCalculator::computeLogForwardMaxReduce() {
    const Hmm &hmm = getHmmRef();
    const Vector &pi = hmm.getPi();
    const std::size_t T = observations_.size();
    const std::size_t N = numStates_;
    const double *logTransTData = logTransT_.data();
    const double *emitByTimeData = logEmitByTime_.data();
    double *alphaData = logAlpha_.data();

    init_log_forward(alphaData, pi, emitByTimeData, N);

    // t > 0.
    for (std::size_t t = 1; t < T; ++t) {
        const double *prevAlphaRow = alphaData + (t - 1) * N;
        double *alphaRow = alphaData + t * N;
        const double *emitRow = emitByTimeData + t * N;
        for (std::size_t j = 0; j < N; ++j) {
            const double *transCol = logTransTData + j * N;
            const double maxTerm = performance::detail::TranscendentalKernels::reduce_max_sum2(
                prevAlphaRow, transCol, N);

            double logSum = LOG_ZERO;
            if (std::isfinite(maxTerm)) {
                const double scaledSum =
                    performance::detail::TranscendentalKernels::sum_exp_sum2_minus_max(
                        prevAlphaRow, transCol, N, maxTerm);
                if (scaledSum > 0.0) {
                    logSum = maxTerm + std::log(scaledSum);
                }
            }
            alphaRow[j] = emitRow[j] + logSum;
        }
    }
}

void ForwardBackwardCalculator::computeLogBackward() {
    if (currentMode_ == FbRecurrenceMode::MaxReduce) {
        computeLogBackwardMaxReduce();
        return;
    }
    computeLogBackwardPairwise();
}

void ForwardBackwardCalculator::computeLogBackwardPairwise() {
    const std::size_t T = observations_.size();
    const std::size_t N = numStates_;
    const double *logTransData = logTrans_.data();
    const double *emitByTimeData = logEmitByTime_.data();
    double *betaData = logBeta_.data();

    init_log_backward(betaData, T, N);

    // t < T - 1.
    if (T > 1) {
        for (std::size_t t = T - 2;; --t) {
            double *betaRow = betaData + t * N;
            const double *nextBetaRow = betaData + (t + 1) * N;
            const double *emitNextRow = emitByTimeData + (t + 1) * N;
            for (std::size_t i = 0; i < N; ++i) {
                const double *transRow = logTransData + i * N;
                double logSum = LOG_ZERO;
                for (std::size_t j = 0; j < N; ++j) {
                    logSum = logSumExp(logSum, transRow[j] + emitNextRow[j] + nextBetaRow[j]);
                }
                betaRow[i] = logSum;
            }
            if (t == 0) {
                break;
            }
        }
    }
}

void ForwardBackwardCalculator::computeLogBackwardMaxReduce() {
    const std::size_t T = observations_.size();
    const std::size_t N = numStates_;
    const double *logTransData = logTrans_.data();
    const double *emitByTimeData = logEmitByTime_.data();
    double *betaData = logBeta_.data();

    init_log_backward(betaData, T, N);

    // t < T - 1.
    if (T > 1) {
        for (std::size_t t = T - 2;; --t) {
            double *betaRow = betaData + t * N;
            const double *nextBetaRow = betaData + (t + 1) * N;
            const double *emitNextRow = emitByTimeData + (t + 1) * N;
            for (std::size_t i = 0; i < N; ++i) {
                const double *transRow = logTransData + i * N;
                const double maxTerm = performance::detail::TranscendentalKernels::reduce_max_sum3(
                    transRow, emitNextRow, nextBetaRow, N);

                double logSum = LOG_ZERO;
                if (std::isfinite(maxTerm)) {
                    const double scaledSum =
                        performance::detail::TranscendentalKernels::sum_exp_sum3_minus_max(
                            transRow, emitNextRow, nextBetaRow, N, maxTerm);
                    if (scaledSum > 0.0) {
                        logSum = maxTerm + std::log(scaledSum);
                    }
                }
                betaRow[i] = logSum;
            }
            if (t == 0) {
                break;
            }
        }
    }
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
