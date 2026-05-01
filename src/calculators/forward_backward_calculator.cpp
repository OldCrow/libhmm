#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/hmm.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace libhmm {

namespace {
constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();
constexpr std::size_t FB_MAX_REDUCE_FORCE_PAIRWISE_MAX_STATES = 2;
constexpr int kProbeRounds = 3;

// One-shot read of LIBHMM_FB_MODE. Returns std::nullopt unless the value
// resolves to a known mode keyword. "auto" or any unknown value is treated
// as "no override" so the static policy + probe path remains active.
std::optional<FbRecurrenceMode> readEnvRecurrenceModeOverride() noexcept {
    static const std::optional<FbRecurrenceMode> kCached =
        []() -> std::optional<FbRecurrenceMode> {
        // std::getenv is the portable C++ choice. MSVC emits C4996 here
        // suggesting _dupenv_s; suppress narrowly because this single read
        // is one-shot at static init and the value is not retained as a
        // string.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        const char *raw = std::getenv("LIBHMM_FB_MODE");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
        if (raw == nullptr) {
            return std::nullopt;
        }
        const std::string_view value(raw);
        if (value == "pairwise") {
            return FbRecurrenceMode::Pairwise;
        }
        if (value == "max-reduce" || value == "maxreduce") {
            return FbRecurrenceMode::MaxReduce;
        }
        return std::nullopt;
    }();
    return kCached;
}

// Thread-local LRU cache mapping N -> probed FbRecurrenceMode. Bounded
// capacity prevents unbounded growth in long-lived processes that touch
// many distinct N values.
class FbProbeCache {
public:
    static constexpr std::size_t kCapacity = 32;

    [[nodiscard]] std::optional<FbRecurrenceMode> get(std::size_t numStates) const noexcept {
        for (const auto &entry : entries_) {
            if (entry.first == numStates) {
                return entry.second;
            }
        }
        return std::nullopt;
    }

    void put(std::size_t numStates, FbRecurrenceMode mode) noexcept {
        for (auto &entry : entries_) {
            if (entry.first == numStates) {
                entry.second = mode;
                return;
            }
        }
        if (entries_.size() < kCapacity) {
            entries_.emplace_back(numStates, mode);
            return;
        }
        entries_[evictIdx_] = {numStates, mode};
        evictIdx_ = (evictIdx_ + 1) % kCapacity;
    }

private:
    std::vector<std::pair<std::size_t, FbRecurrenceMode>> entries_;
    std::size_t evictIdx_{0};
};

thread_local FbProbeCache g_fbProbeCache;
} // namespace

FbRecurrenceMode ForwardBackwardCalculator::resolveRecurrenceMode(
    const std::size_t numStates, const std::size_t sequenceLength) const noexcept {
#if defined(LIBHMM_EXPERIMENT_FB_MAX_REDUCE)
    // Compile-time forcer: highest priority. Preserves benchmark-build contract.
    (void)numStates;
    (void)sequenceLength;
    return FbRecurrenceMode::MaxReduce;
#elif defined(LIBHMM_EXPERIMENT_FB_ADAPTIVE_SELECTOR)
    // Legacy adaptive forcer: simple N>2 cutoff. Preserves benchmark-build contract.
    (void)sequenceLength;
    return (numStates > FB_MAX_REDUCE_FORCE_PAIRWISE_MAX_STATES)
               ? FbRecurrenceMode::MaxReduce
               : FbRecurrenceMode::Pairwise;
#else
    if (modeOverride_.has_value()) {
        return *modeOverride_;
    }
    if (const auto envMode = readEnvRecurrenceModeOverride(); envMode.has_value()) {
        return *envMode;
    }
    constexpr FbHostProfile profile = makeFbHostProfile();
    if (isFbBoundaryPoint(numStates, sequenceLength, profile)) {
        if (const auto cached = g_fbProbeCache.get(numStates); cached.has_value()) {
            return *cached;
        }
        // The actual probe runs in compute() once buffers are populated. Until
        // then we fall back to the static bin so callers can still resolve a
        // valid mode without observation data.
    }
    return selectFbRecurrenceMode(numStates, sequenceLength, profile);
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
    // env var / boundary cache / static policy pipeline.
    currentMode_ = resolveRecurrenceMode(numStates_, T);

#if !defined(LIBHMM_EXPERIMENT_FB_MAX_REDUCE) && !defined(LIBHMM_EXPERIMENT_FB_ADAPTIVE_SELECTOR)
    // Boundary refinement (Phase A3): if no override path applies and we are
    // in a boundary region with no thread-local cache hit yet, probe both
    // kernels on a single timestep using populated buffers and cache the
    // winner for subsequent compute() calls in this thread.
    if (!modeOverride_.has_value() &&
        !readEnvRecurrenceModeOverride().has_value() && T >= 2) {
        constexpr FbHostProfile profile = makeFbHostProfile();
        if (isFbBoundaryPoint(numStates_, T, profile) &&
            !g_fbProbeCache.get(numStates_).has_value()) {
            const Vector &pi = hmm.getPi();
            std::vector<double> probeAlpha0(numStates_);
            const double *emitRow0 = logEmitByTime_.data();
            for (std::size_t i = 0; i < numStates_; ++i) {
                const double logPi = (pi(i) > 0.0) ? std::log(pi(i)) : LOG_ZERO;
                probeAlpha0[i] = logPi + emitRow0[i];
            }
            const double *emitRow1 = logEmitByTime_.data() + numStates_;
            const FbRecurrenceMode probed = probeRecurrenceMode(
                numStates_, probeAlpha0.data(), emitRow1, logTransT_.data());
            g_fbProbeCache.put(numStates_, probed);
            currentMode_ = probed;
        }
    }
#endif

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
    const Hmm &hmm = getHmmRef();
    const Matrix &trans = hmm.getTrans();
    logTrans_.resize(numStates_, numStates_);
    logTransT_.resize(numStates_, numStates_);
    for (std::size_t i = 0; i < numStates_; ++i) {
        for (std::size_t j = 0; j < numStates_; ++j) {
            const double a = trans(i, j);
            const double logA = (a > 0.0) ? std::log(a) : LOG_ZERO;
            logTrans_(i, j) = logA;
            logTransT_(j, i) = logA;
        }
    }
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

    // t = 0.
    const double *emitRow0 = emitByTimeData;
    for (std::size_t i = 0; i < N; ++i) {
        const double logPi = (pi(i) > 0.0) ? std::log(pi(i)) : LOG_ZERO;
        alphaData[i] = logPi + emitRow0[i];
    }

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

    // t = 0.
    const double *emitRow0 = emitByTimeData;
    for (std::size_t i = 0; i < N; ++i) {
        const double logPi = (pi(i) > 0.0) ? std::log(pi(i)) : LOG_ZERO;
        alphaData[i] = logPi + emitRow0[i];
    }

    // t > 0.
    for (std::size_t t = 1; t < T; ++t) {
        const double *prevAlphaRow = alphaData + (t - 1) * N;
        double *alphaRow = alphaData + t * N;
        const double *emitRow = emitByTimeData + t * N;
        for (std::size_t j = 0; j < N; ++j) {
            const double *transCol = logTransTData + j * N;
            double maxTerm = LOG_ZERO;
            for (std::size_t i = 0; i < N; ++i) {
                const double term = prevAlphaRow[i] + transCol[i];
                if (term > maxTerm) {
                    maxTerm = term;
                }
            }

            double logSum = LOG_ZERO;
            if (std::isfinite(maxTerm)) {
                double scaledSum = 0.0;
                for (std::size_t i = 0; i < N; ++i) {
                    const double term = prevAlphaRow[i] + transCol[i];
                    if (std::isfinite(term)) {
                        scaledSum += std::exp(term - maxTerm);
                    }
                }
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

    // t = T - 1.
    double *finalBetaRow = betaData + (T - 1) * N;
    for (std::size_t i = 0; i < N; ++i) {
        finalBetaRow[i] = 0.0;
    }

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

    // t = T - 1.
    double *finalBetaRow = betaData + (T - 1) * N;
    for (std::size_t i = 0; i < N; ++i) {
        finalBetaRow[i] = 0.0;
    }

    // t < T - 1.
    if (T > 1) {
        for (std::size_t t = T - 2;; --t) {
            double *betaRow = betaData + t * N;
            const double *nextBetaRow = betaData + (t + 1) * N;
            const double *emitNextRow = emitByTimeData + (t + 1) * N;
            for (std::size_t i = 0; i < N; ++i) {
                const double *transRow = logTransData + i * N;
                double maxTerm = LOG_ZERO;
                for (std::size_t j = 0; j < N; ++j) {
                    const double term = transRow[j] + emitNextRow[j] + nextBetaRow[j];
                    if (term > maxTerm) {
                        maxTerm = term;
                    }
                }

                double logSum = LOG_ZERO;
                if (std::isfinite(maxTerm)) {
                    double scaledSum = 0.0;
                    for (std::size_t j = 0; j < N; ++j) {
                        const double term = transRow[j] + emitNextRow[j] + nextBetaRow[j];
                        if (std::isfinite(term)) {
                            scaledSum += std::exp(term - maxTerm);
                        }
                    }
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

FbRecurrenceMode ForwardBackwardCalculator::probeRecurrenceMode(
    const std::size_t N, const double *prevAlphaRow, const double *emitRow,
    const double *logTransTData) noexcept {
    using Clock = std::chrono::steady_clock;
    std::vector<double> outPair(N);
    std::vector<double> outMax(N);

    auto runPair = [&]() {
        for (std::size_t j = 0; j < N; ++j) {
            const double *transCol = logTransTData + j * N;
            double sum = LOG_ZERO;
            for (std::size_t i = 0; i < N; ++i) {
                const double term = prevAlphaRow[i] + transCol[i];
                sum = logSumExp(sum, term);
            }
            outPair[j] = emitRow[j] + sum;
        }
    };

    auto runMax = [&]() {
        for (std::size_t j = 0; j < N; ++j) {
            const double *transCol = logTransTData + j * N;
            double maxTerm = LOG_ZERO;
            for (std::size_t i = 0; i < N; ++i) {
                const double term = prevAlphaRow[i] + transCol[i];
                if (term > maxTerm) {
                    maxTerm = term;
                }
            }
            double logSum = LOG_ZERO;
            if (std::isfinite(maxTerm)) {
                double scaledSum = 0.0;
                for (std::size_t i = 0; i < N; ++i) {
                    const double term = prevAlphaRow[i] + transCol[i];
                    if (std::isfinite(term)) {
                        scaledSum += std::exp(term - maxTerm);
                    }
                }
                if (scaledSum > 0.0) {
                    logSum = maxTerm + std::log(scaledSum);
                }
            }
            outMax[j] = emitRow[j] + logSum;
        }
    };

    std::array<Clock::duration, kProbeRounds> pairTimes{};
    std::array<Clock::duration, kProbeRounds> maxTimes{};
    // Warm-up: discard first run so cache effects do not bias the median.
    runPair();
    runMax();
    for (int r = 0; r < kProbeRounds; ++r) {
        const auto t0 = Clock::now();
        runPair();
        const auto t1 = Clock::now();
        runMax();
        const auto t2 = Clock::now();
        pairTimes[r] = t1 - t0;
        maxTimes[r] = t2 - t1;
    }
    std::sort(pairTimes.begin(), pairTimes.end());
    std::sort(maxTimes.begin(), maxTimes.end());
    const auto pairMedian = pairTimes[kProbeRounds / 2];
    const auto maxMedian = maxTimes[kProbeRounds / 2];
    return (maxMedian < pairMedian) ? FbRecurrenceMode::MaxReduce
                                    : FbRecurrenceMode::Pairwise;
}

} // namespace libhmm
