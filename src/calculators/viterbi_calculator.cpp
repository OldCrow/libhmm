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

ViterbiCalculator::ViterbiCalculator(
        const Hmm& hmm, const ObservationSet& observations)
    : Calculator(hmm, observations)
    , numStates_(static_cast<std::size_t>(hmm.getNumStates()))
{
    if (observations.empty()) {
        throw std::invalid_argument("Observation sequence cannot be empty");
    }
    precomputeLogTransitions();
    static_cast<void>(decode());
}

ViterbiCalculator::ViterbiCalculator(
        Hmm* hmm, const ObservationSet& observations)
    : ViterbiCalculator(
        hmm ? *hmm : throw std::invalid_argument("HMM pointer cannot be null"),
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
    const Hmm& hmm = getHmmRef();

    std::vector<double> obsVec(T);
    for (std::size_t t = 0; t < T; ++t) obsVec[t] = observations_(t);

    for (std::size_t i = 0; i < numStates_; ++i) {
        hmm.getDistribution(i).getBatchLogProbabilities(
            std::span<const double>(obsVec.data(), T),
            std::span<double>(logEmitBuf_.data() + i * T, T));
    }

    runViterbi();
    backtrack();
    return sequence_;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void ViterbiCalculator::precomputeLogTransitions() {
    const Hmm& hmm = getHmmRef();
    const Matrix& trans = hmm.getTrans();
    logTrans_.resize(numStates_, numStates_);
    for (std::size_t i = 0; i < numStates_; ++i) {
        for (std::size_t j = 0; j < numStates_; ++j) {
            const double a = trans(i, j);
            logTrans_(i, j) = (a > 0.0) ? std::log(a) : LOG_ZERO;
        }
    }
}

void ViterbiCalculator::runViterbi() {
    const Hmm& hmm = getHmmRef();
    const Vector& pi = hmm.getPi();
    const std::size_t T = observations_.size();

    logDelta_.resize(T, numStates_);
    psi_.assign(T, std::vector<int>(numStates_, 0));

    // t = 0: initialise
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double logPi = (pi(i) > 0.0) ? std::log(pi(i)) : LOG_ZERO;
        logDelta_(0, i) = logPi + logEmitBuf_[i * T + 0];
    }

    // t > 0: recursion
    for (std::size_t t = 1; t < T; ++t) {
        for (std::size_t j = 0; j < numStates_; ++j) {
            double maxVal  = LOG_ZERO;
            int    maxFrom = 0;
            for (std::size_t i = 0; i < numStates_; ++i) {
                const double val = logDelta_(t - 1, i) + logTrans_(i, j);
                if (val > maxVal) { maxVal = val; maxFrom = static_cast<int>(i); }
            }
            logDelta_(t, j) = maxVal + logEmitBuf_[j * T + t];
            psi_[t][j] = maxFrom;
        }
    }

    // Termination: best last state
    double bestVal  = LOG_ZERO;
    int    bestLast = 0;
    for (std::size_t i = 0; i < numStates_; ++i) {
        if (logDelta_(T - 1, i) > bestVal) {
            bestVal  = logDelta_(T - 1, i);
            bestLast = static_cast<int>(i);
        }
    }
    logProbability_ = bestVal;

    sequence_.resize(T);
    sequence_(T - 1) = bestLast;
}

void ViterbiCalculator::backtrack() {
    const std::size_t T = observations_.size();
    if (T <= 1) return;

    for (std::size_t t = T - 2; ; --t) {
        sequence_(t) = psi_[t + 1][static_cast<std::size_t>(sequence_(t + 1))];
        if (t == 0) break;
    }
}

} // namespace libhmm
