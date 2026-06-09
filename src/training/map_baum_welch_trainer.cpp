#include "libhmm/training/map_baum_welch_trainer.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/hmm.h"
#include "libhmm/performance/transcendental_kernels.h"
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace libhmm {

namespace {
// Per-TU sentinel — same convention as baum_welch_trainer.cpp,
// forward_backward_calculator.cpp, and viterbi_calculator.cpp.
constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------------------
// E-step helpers
//
// Identical in structure to the private statics in BaumWelchTrainer.
// Duplicated here to avoid coupling to BaumWelchTrainer's private interface;
// a future refactor could promote them to Trainer as protected statics.
// ---------------------------------------------------------------------------

/// Accumulates γ statistics for one observation sequence.
void accumulate_gamma(const Matrix &logAlpha, const Matrix &logBeta, const ObservationSet &obs,
                      double logP, std::size_t N, std::vector<std::vector<double>> &emisData,
                      std::vector<std::vector<double>> &emisWts, std::vector<double> &piNum,
                      std::vector<double> &transDen) noexcept {
    const std::size_t T = obs.size();
    const double *logAlphaData = logAlpha.data();
    const double *logBetaData = logBeta.data();
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
}

/// Accumulates ξ statistics (expected transition counts) for one sequence.
void accumulate_xi(const double *logAlphaData, const double *logBetaData,
                   const std::vector<double> &logEmitByTime, const std::vector<double> &logTransT,
                   double logP, std::size_t T, std::size_t N, bool hasZeroTransitions,
                   std::vector<double> &transNumT) noexcept {
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
                    if (transCol[i] == LOG_ZERO)
                        continue;
                    transNumCol[i] += std::exp(alphaRow[i] + transCol[i] + emitBetaNext);
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
}

} // namespace

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

MapBaumWelchTrainer::MapBaumWelchTrainer(Hmm &hmm, const ObservationLists &obsLists,
                                         double pseudo_count)
    : Trainer(hmm, obsLists), pseudo_count_(pseudo_count) {
    if (pseudo_count < 0.0) {
        throw std::invalid_argument("MapBaumWelchTrainer: pseudo_count must be >= 0 "
                                    "(sparse-inducing priors require Variational Bayes)");
    }
}

MapBaumWelchTrainer::MapBaumWelchTrainer(Hmm *hmm, const ObservationLists &obsLists,
                                         double pseudo_count)
    : MapBaumWelchTrainer(hmm ? *hmm : throw std::invalid_argument("HMM pointer cannot be null"),
                          obsLists, pseudo_count) {}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

void MapBaumWelchTrainer::setPseudoCount(double c) {
    if (c < 0.0)
        throw std::invalid_argument("MapBaumWelchTrainer: pseudo_count must be >= 0");
    pseudo_count_ = c;
}

double MapBaumWelchTrainer::computeLogPrior() const {
    if (pseudo_count_ == 0.0)
        return 0.0;

    const Hmm &hmm = hmm_ref_.get();
    const std::size_t N = static_cast<std::size_t>(hmm.getNumStates());
    const double c = pseudo_count_;
    double lp = 0.0;

    // c · Σ_i Σ_j log A(i,j)
    const Matrix &A = hmm.getTrans();
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            lp += c * (A(i, j) > 0.0 ? std::log(A(i, j)) : LOG_ZERO);

    // c · Σ_i log π_i
    const Vector &pi = hmm.getPi();
    for (std::size_t i = 0; i < N; ++i)
        lp += c * (pi(i) > 0.0 ? std::log(pi(i)) : LOG_ZERO);

    // c · Σ_{discrete i} Σ_k log B(i,k)
    for (std::size_t i = 0; i < N; ++i) {
        const EmissionDistribution &dist = hmm.getDistribution(i);
        if (!dist.isDiscrete())
            continue;
        const auto &dd = static_cast<const DiscreteDistribution &>(dist);
        const std::size_t K = dd.getNumSymbols();
        for (std::size_t k = 0; k < K; ++k) {
            const double bk = dd.getSymbolProbability(k);
            lp += c * (bk > 0.0 ? std::log(bk) : LOG_ZERO);
        }
    }

    return lp;
}

// ---------------------------------------------------------------------------
// train() — one full MAP-EM pass
// ---------------------------------------------------------------------------

void MapBaumWelchTrainer::train() {
    Hmm &hmm = hmm_ref_.get();
    const std::size_t N = static_cast<std::size_t>(hmm.getNumStates());
    const double c = pseudo_count_;

    std::size_t totalExpectedLength = 0;
    for (const auto &obs : getObservationLists())
        totalExpectedLength += obs.size();

    // Accumulators (linear space, summed across all sequences)
    std::vector<double> piNum(N, 0.0);
    std::vector<double> transDen(N, 0.0);
    // Column-major: transNumT[j*N+i] = ξ(i→j)
    std::vector<double> transNumT(N * N, 0.0);

    std::vector<std::vector<double>> emisData(N);
    std::vector<std::vector<double>> emisWts(N);
    for (std::size_t i = 0; i < N; ++i) {
        emisData[i].reserve(totalExpectedLength);
        emisWts[i].reserve(totalExpectedLength);
    }

    // Precompute transposed log-transition matrix
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

    for (const auto &obs : getObservationLists()) {
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

        // Time-major log-emission buffer
        std::vector<double> logEmitStateMajor(N * T);
        std::vector<double> logEmitByTime(N * T);
        const std::span<const double> obsSpan(obs.data(), T);
        for (std::size_t i = 0; i < N; ++i) {
            hmm.getDistribution(i).getBatchLogProbabilities(
                obsSpan, std::span<double>(logEmitStateMajor.data() + i * T, T));
        }
        for (std::size_t i = 0; i < N; ++i) {
            const double *stateRow = logEmitStateMajor.data() + i * T;
            for (std::size_t t = 0; t < T; ++t)
                logEmitByTime[t * N + i] = stateRow[t];
        }

        accumulate_gamma(logAlpha, logBeta, obs, logP, N, emisData, emisWts, piNum, transDen);
        accumulate_xi(logAlphaData, logBetaData, logEmitByTime, logTransT, logP, T, N,
                      hasZeroTransitions, transNumT);
        ++validSeqs;
    }

    if (validSeqs == 0) {
        throw std::runtime_error("MapBaumWelchTrainer: no valid observation sequences "
                                 "(all had zero probability under the current model)");
    }

    m_step_pi_map(hmm, N, piNum, c);
    m_step_transitions_map(hmm, N, transNumT, transDen, c);
    apply_emission_fits_map(hmm, N, emisData, emisWts, c);
}

// ---------------------------------------------------------------------------
// MAP M-step helpers
// ---------------------------------------------------------------------------

void MapBaumWelchTrainer::m_step_pi_map(Hmm &hmm, std::size_t N, const std::vector<double> &piNum,
                                        double c) {
    const double piSum = std::accumulate(piNum.begin(), piNum.end(), 0.0);
    const double denom = piSum + static_cast<double>(N) * c;
    Vector pi(N);
    for (std::size_t i = 0; i < N; ++i)
        pi(i) = (denom > 0.0) ? (piNum[i] + c) / denom : 1.0 / static_cast<double>(N);
    hmm.setPi(pi);
}

void MapBaumWelchTrainer::m_step_transitions_map(Hmm &hmm, std::size_t N,
                                                 const std::vector<double> &transNumT,
                                                 const std::vector<double> &transDen, double c) {
    const double Nc = static_cast<double>(N) * c;
    Matrix newTrans(N, N);
    for (std::size_t i = 0; i < N; ++i) {
        const double denom = transDen[i] + Nc;
        for (std::size_t j = 0; j < N; ++j) {
            newTrans(i, j) =
                (denom > 0.0) ? (transNumT[j * N + i] + c) / denom : 1.0 / static_cast<double>(N);
        }
    }
    hmm.setTrans(newTrans);
}

void MapBaumWelchTrainer::apply_emission_fits_map(Hmm &hmm, std::size_t N,
                                                  const std::vector<std::vector<double>> &emisData,
                                                  const std::vector<std::vector<double>> &emisWts,
                                                  double c) {
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t M = emisData[i].size();
        EmissionDistribution &dist = hmm.getDistribution(i);

        if (M == 0) {
            dist.reset();
            continue;
        }

        // Weighted MLE fit for all distribution types.
        dist.fit(std::span<const double>(emisData[i].data(), M),
                 std::span<const double>(emisWts[i].data(), M));

        // For discrete distributions, apply Dirichlet smoothing on top of MLE.
        // B_map(i,k) = (B_mle(i,k)·sumW + c) / (sumW + K·c)
        if (c > 0.0 && dist.isDiscrete()) {
            auto &dd = static_cast<DiscreteDistribution &>(dist);
            const std::size_t K = dd.getNumSymbols();
            const double sumW = std::accumulate(emisWts[i].begin(), emisWts[i].end(), 0.0);
            if (sumW > 0.0) {
                const double denom = sumW + static_cast<double>(K) * c;
                for (std::size_t k = 0; k < K; ++k) {
                    dd.setProbability(static_cast<double>(k),
                                      (dd.getSymbolProbability(k) * sumW + c) / denom);
                }
            }
        }
    }
}

} // namespace libhmm
