#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/hmm.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <span>

namespace libhmm {

namespace {
    constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

ForwardBackwardCalculator::ForwardBackwardCalculator(
        const Hmm& hmm, const ObservationSet& observations)
    : Calculator(hmm, observations)
    , numStates_(static_cast<std::size_t>(hmm.getNumStates()))
{
    if (observations.empty()) {
        throw std::invalid_argument("Observation sequence cannot be empty");
    }
    precomputeLogTransitions();
    compute();
}

ForwardBackwardCalculator::ForwardBackwardCalculator(
        Hmm* hmm, const ObservationSet& observations)
    : ForwardBackwardCalculator(
        hmm ? *hmm : throw std::invalid_argument("HMM pointer cannot be null"),
        observations) {}

// ---------------------------------------------------------------------------
// Public compute interface
// ---------------------------------------------------------------------------

void ForwardBackwardCalculator::compute(const ObservationSet& observations) {
    observations_ = observations;
    compute();
}

void ForwardBackwardCalculator::compute() {
    const std::size_t T = observations_.size();
    if (T == 0) {
        logProbability_ = LOG_ZERO;
        return;
    }

    // Allocate/resize result matrices
    logAlpha_.resize(T, numStates_);
    logBeta_.resize(T, numStates_);

    // Pre-fill the log-emission buffer: logEmitBuf_[i * T + t] = log b_i(O_t)
    // Build observation span once; reuse across all N states.
    logEmitBuf_.resize(T * numStates_);
    std::vector<double> obsVec(T);
    for (std::size_t t = 0; t < T; ++t) obsVec[t] = observations_(t);
    const std::span<const double> obsSpan(obsVec.data(), T);

    const Hmm& hmm = getHmmRef();
    for (std::size_t i = 0; i < numStates_; ++i) {
        hmm.getDistribution(i).getBatchLogProbabilities(
            obsSpan,
            std::span<double>(logEmitBuf_.data() + i * T, T));
    }

    computeLogForward();
    computeLogBackward();

    // log P(O|λ) = log-sum-exp over states at final timestep
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

void ForwardBackwardCalculator::computeLogForward() {
    const Hmm& hmm = getHmmRef();
    const Vector& pi = hmm.getPi();
    const std::size_t T = observations_.size();

    // t = 0: log alpha(0, i) = log pi_i + log b_i(O_0)
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double logPi = (pi(i) > 0.0) ? std::log(pi(i)) : LOG_ZERO;
        logAlpha_(0, i) = logPi + logEmitBuf_[i * T + 0];
    }

    // t > 0
    for (std::size_t t = 1; t < T; ++t) {
        for (std::size_t j = 0; j < numStates_; ++j) {
            double logSum = LOG_ZERO;
            for (std::size_t i = 0; i < numStates_; ++i) {
                logSum = logSumExp(logSum, logAlpha_(t - 1, i) + logTrans_(i, j));
            }
            logAlpha_(t, j) = logEmitBuf_[j * T + t] + logSum;
        }
    }
}

void ForwardBackwardCalculator::computeLogBackward() {
    const std::size_t T = observations_.size();

    // t = T-1: log beta(T-1, i) = log(1) = 0
    for (std::size_t i = 0; i < numStates_; ++i) {
        logBeta_(T - 1, i) = 0.0;
    }

    // t < T-1, working backwards
    if (T > 1) {
        for (std::size_t t = T - 2; ; --t) {
            for (std::size_t i = 0; i < numStates_; ++i) {
                double logSum = LOG_ZERO;
                for (std::size_t j = 0; j < numStates_; ++j) {
                    logSum = logSumExp(logSum,
                        logTrans_(i, j) +
                        logEmitBuf_[j * T + (t + 1)] +
                        logBeta_(t + 1, j));
                }
                logBeta_(t, i) = logSum;
            }
            if (t == 0) break;
        }
    }
}

// Numerically stable log(exp(a) + exp(b))
double ForwardBackwardCalculator::logSumExp(double a, double b) noexcept {
    if (a == LOG_ZERO) return b;
    if (b == LOG_ZERO) return a;
    if (a > b) return a + std::log1p(std::exp(b - a));
    return b + std::log1p(std::exp(a - b));
}

} // namespace libhmm
