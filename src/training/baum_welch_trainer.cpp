#include "libhmm/training/baum_welch_trainer.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/hmm.h"
#include "libhmm/performance/transcendental_kernels.h"
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace libhmm {

namespace {
constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();
}

BaumWelchTrainer::BaumWelchTrainer(Hmm &hmm, const ObservationLists &obsLists)
    : Trainer(hmm, obsLists) {}

BaumWelchTrainer::BaumWelchTrainer(Hmm *hmm, const ObservationLists &obsLists)
    : Trainer(hmm, obsLists) {}

// ---------------------------------------------------------------------------
// train() — one full EM pass over all sequences
// ---------------------------------------------------------------------------

void BaumWelchTrainer::train() {
    Hmm &hmm = hmm_ref_.get();
    const std::size_t N = static_cast<std::size_t>(hmm.getNumStates());
    std::size_t totalExpectedLength = 0;
    for (const auto &obs : obsLists_) {
        totalExpectedLength += obs.size();
    }

    // Accumulators (linear space, summed across all sequences)
    std::vector<double> piNum(N, 0.0);
    std::vector<double> transDen(N, 0.0);
    // Column-major accumulation: transNumT[j * N + i] stores the expected count
    // for transition i->j. This matches the t/j/i xi loop for contiguous reads
    // from the transposed log-transition matrix.
    std::vector<double> transNumT(N * N, 0.0);

    // Per-state emission data/weights accumulated across sequences
    std::vector<std::vector<double>> emisData(N);
    std::vector<std::vector<double>> emisWts(N);
    for (std::size_t i = 0; i < N; ++i) {
        emisData[i].reserve(totalExpectedLength);
        emisWts[i].reserve(totalExpectedLength);
    }

    // Precompute transposed log-transition matrix from the current model:
    // logTransT[j * N + i] = log a_{ij}
    const Matrix &curTrans = hmm.getTrans();
    std::vector<double> logTransT(N * N);
    bool hasZeroTransitions = false;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            const double a = curTrans(i, j);
            if (a > 0.0) {
                logTransT[j * N + i] = std::log(a);
            } else {
                logTransT[j * N + i] = LOG_ZERO;
                hasZeroTransitions = true;
            }
        }
    }

    std::size_t validSeqs = 0;

    for (const auto &obs : obsLists_) {
        const std::size_t T = obs.size();
        if (T == 0)
            continue;

        ForwardBackwardCalculator fbc(hmm, obs);
        const double logP = fbc.getLogProbability();
        if (!std::isfinite(logP))
            continue;

        const Matrix &logAlpha = fbc.getLogForwardVariables();
        const Matrix &logBeta = fbc.getLogBackwardVariables();
        const double *logAlphaData = logAlpha.data();
        const double *logBetaData = logBeta.data();

        // Precompute log-emissions for this sequence, then relayout to time-major:
        // logEmitByTime[t * N + j] = log b_j(O_t)
        std::vector<double> logEmitStateMajor(N * T);
        std::vector<double> logEmitByTime(N * T);
        const std::span<const double> obsSpan(obs.data(), T);
        for (std::size_t i = 0; i < N; ++i) {
            hmm.getDistribution(i).getBatchLogProbabilities(
                obsSpan, std::span<double>(logEmitStateMajor.data() + i * T, T));
        }
        for (std::size_t i = 0; i < N; ++i) {
            const double *stateRow = logEmitStateMajor.data() + i * T;
            for (std::size_t t = 0; t < T; ++t) {
                logEmitByTime[t * N + i] = stateRow[t];
            }
        }

        // Accumulate gamma (per timestep per state) and pi/trans denominators
        for (std::size_t t = 0; t < T; ++t) {
            const double *alphaRow = logAlphaData + t * N;
            const double *betaRow = logBetaData + t * N;
            const double obsVal = obs(t);
            for (std::size_t i = 0; i < N; ++i) {
                const double g = std::exp(alphaRow[i] + betaRow[i] - logP);
                emisData[i].push_back(obsVal);
                emisWts[i].push_back(g);
                if (t == 0)
                    piNum[i] += g;
                if (t < T - 1)
                    transDen[i] += g;
            }
        }

        // Accumulate xi (transition counts). Dense models take a branch-free
        // path; sparse models keep the zero-transition skip.
        if (hasZeroTransitions) {
            for (std::size_t t = 0; t + 1 < T; ++t) {
                const double *alphaRow = logAlphaData + t * N;
                const double *betaNextRow = logBetaData + (t + 1) * N;
                const double *emitNextRow = logEmitByTime.data() + (t + 1) * N;
                for (std::size_t j = 0; j < N; ++j) {
                    const double emitBetaNext = emitNextRow[j] + betaNextRow[j] - logP;
                    const double *transCol = logTransT.data() + j * N;
                    double *transNumCol = transNumT.data() + j * N;
                    for (std::size_t i = 0; i < N; ++i) {
                        if (transCol[i] == LOG_ZERO) {
                            continue;
                        }
                        const double logXi = alphaRow[i] + transCol[i] + emitBetaNext;
                        transNumCol[i] += std::exp(logXi);
                    }
                }
            }
        } else {
            for (std::size_t t = 0; t + 1 < T; ++t) {
                const double *alphaRow = logAlphaData + t * N;
                const double *betaNextRow = logBetaData + (t + 1) * N;
                const double *emitNextRow = logEmitByTime.data() + (t + 1) * N;
                for (std::size_t j = 0; j < N; ++j) {
                    const double emitBetaNext = emitNextRow[j] + betaNextRow[j] - logP;
                    const double *transCol = logTransT.data() + j * N;
                    double *transNumCol = transNumT.data() + j * N;
                    performance::detail::TranscendentalKernels::accumulate_exp_sum2_bias(
                        transNumCol, alphaRow, transCol, N, emitBetaNext);
                }
            }
        }

        ++validSeqs;
    }

    if (validSeqs == 0) {
        throw std::runtime_error("BaumWelchTrainer: no valid observation sequences "
                                 "(all had zero probability under the current model)");
    }

    // ---- M-step: pi ----
    {
        double piSum = 0.0;
        for (std::size_t i = 0; i < N; ++i)
            piSum += piNum[i];
        Vector pi(N);
        for (std::size_t i = 0; i < N; ++i) {
            pi(i) = (piSum > 0.0) ? piNum[i] / piSum : 1.0 / static_cast<double>(N);
        }
        hmm.setPi(pi);
    }

    // ---- M-step: transition matrix ----
    {
        Matrix newTrans(N, N);
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                newTrans(i, j) =
                    (transDen[i] > 0.0) ? transNumT[j * N + i] / transDen[i]
                                        : 1.0 / static_cast<double>(N);
            }
        }
        hmm.setTrans(newTrans);
    }

    // ---- M-step: emission distributions ----
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t M = emisData[i].size();
        if (M == 0) {
            hmm.getDistribution(i).reset();
            continue;
        }
        hmm.getDistribution(i).fit(std::span<const double>(emisData[i].data(), M),
                                   std::span<const double>(emisWts[i].data(), M));
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

double BaumWelchTrainer::logSumExp(double a, double b) noexcept {
    if (a == LOG_ZERO)
        return b;
    if (b == LOG_ZERO)
        return a;
    if (a > b)
        return a + std::log1p(std::exp(b - a));
    return b + std::log1p(std::exp(a - b));
}

} // namespace libhmm
