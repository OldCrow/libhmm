#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/hmm.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

namespace libhmm {

namespace {
constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

ViterbiCalculator::ViterbiCalculator(const Hmm &hmm, const ObservationSet &observations)
    : Calculator(hmm, observations), numStates_(static_cast<std::size_t>(hmm.getNumStates())) {
    if (observations.empty()) {
        throw std::invalid_argument("Observation sequence cannot be empty");
    }
    precomputeLogTransitions();
    static_cast<void>(decode());
}

ViterbiCalculator::ViterbiCalculator(Hmm *hmm, const ObservationSet &observations)
    : ViterbiCalculator(hmm ? *hmm : throw std::invalid_argument("HMM pointer cannot be null"),
                        observations) {}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

StateSequence ViterbiCalculator::decode() {
    const std::size_t T = observations_.size();
    if (T == 0) {
        logProbability_ = LOG_ZERO;
        sequence_.resize(0);
        return sequence_;
    }

    // Fill log-emission buffer: logEmitBuf_[i * T + t] = log b_i(O_t)
    logEmitBuf_.resize(T * numStates_);
    const Hmm &hmm = getHmmRef();
    const std::span<const double> obsSpan(observations_.data(), T);

    for (std::size_t i = 0; i < numStates_; ++i) {
        hmm.getDistribution(i).getBatchLogProbabilities(
            obsSpan,
            std::span<double>(logEmitBuf_.data() + i * T, T));
    }
    // Build time-major emission buffer once for locality in dynamic programming.
    logEmitByTime_.resize(T * numStates_);
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double *stateRow = logEmitBuf_.data() + i * T;
        for (std::size_t t = 0; t < T; ++t) {
            logEmitByTime_[t * numStates_ + i] = stateRow[t];
        }
    }

    runViterbi();
    backtrack();
    return sequence_;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void ViterbiCalculator::precomputeLogTransitions() {
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

void ViterbiCalculator::runViterbi() {
    const Hmm &hmm = getHmmRef();
    const Vector &pi = hmm.getPi();
    const std::size_t T = observations_.size();

    logDelta_.resize(T, numStates_);
    psi_.assign(T * numStates_, 0);

    const double *logTransTData = logTransT_.data();
    const double *logEmitByTimeData = logEmitByTime_.data();
    double *logDeltaData = logDelta_.data();
    const std::size_t N = numStates_;

    // t = 0: initialise
    const double *emitRow0 = logEmitByTimeData;
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double logPi = (pi(i) > 0.0) ? std::log(pi(i)) : LOG_ZERO;
        logDeltaData[i] = logPi + emitRow0[i];
    }

    // t > 0: recursion
    for (std::size_t t = 1; t < T; ++t) {
        const double *prevDeltaRow = logDeltaData + (t - 1) * N;
        double *deltaRow = logDeltaData + t * N;
        const double *emitRow = logEmitByTimeData + t * N;
        for (std::size_t j = 0; j < numStates_; ++j) {
            double maxVal = LOG_ZERO;
            int maxFrom = 0;
            const double *transCol = logTransTData + j * N;
            for (std::size_t i = 0; i < numStates_; ++i) {
                const double val = prevDeltaRow[i] + transCol[i];
                if (val > maxVal) {
                    maxVal = val;
                    maxFrom = static_cast<int>(i);
                }
            }
            deltaRow[j] = maxVal + emitRow[j];
            psi_[t * N + j] = maxFrom;
        }
    }

    // Termination: best last state
    double bestVal = LOG_ZERO;
    int bestLast = 0;
    const double *finalDeltaRow = logDeltaData + (T - 1) * N;
    for (std::size_t i = 0; i < numStates_; ++i) {
        if (finalDeltaRow[i] > bestVal) {
            bestVal = finalDeltaRow[i];
            bestLast = static_cast<int>(i);
        }
    }
    logProbability_ = bestVal;

    sequence_.resize(T);
    sequence_(T - 1) = bestLast;
}

void ViterbiCalculator::backtrack() {
    const std::size_t T = observations_.size();
    if (T <= 1)
        return;
    const std::size_t N = numStates_;

    for (std::size_t t = T - 2;; --t) {
        sequence_(t) = psi_[(t + 1) * N + static_cast<std::size_t>(sequence_(t + 1))];
        if (t == 0)
            break;
    }
}

} // namespace libhmm
