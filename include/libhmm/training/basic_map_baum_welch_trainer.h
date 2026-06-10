#pragma once

#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/performance/transcendental_kernels.h"
#include "libhmm/training/basic_trainer.h"

namespace libhmm {

/**
 * @brief MAP-EM Baum-Welch trainer with symmetric Dirichlet priors,
 *        parameterised on observation type.
 *
 * @tparam Obs  `double` (scalar, v3-compatible) or `ObservationVectorView` (v4 MV).
 *
 * Extends standard Baum-Welch by placing symmetric Dirichlet priors on π and the
 * transition matrix rows.  For scalar HMMs with DiscreteDistribution emissions the
 * same pseudo-count is also applied to emission symbol probabilities.  All continuous
 * and all multivariate emission distributions are fitted by MLE (Dirichlet smoothing
 * is only meaningful for discrete/multinomial emissions).
 *
 * The pseudo-count c ≥ 0 is the Dirichlet concentration minus 1:
 *   c = 0 → standard MLE (identical to BaumWelchTrainer)
 *   c = 1 → Laplace smoothing
 *
 * computeLogPrior() returns the parameter-dependent part of log P(λ|c) for use as
 * the convergence criterion:  mapObjective = logLikelihood + computeLogPrior().
 *
 * Two explicit instantiations compiled with LIBHMM_BEST_SIMD_FLAGS:
 *   - src/training/map_baum_welch_trainer.cpp    → BasicMapBaumWelchTrainer<double>
 *   - src/training/map_baum_welch_trainer_mv.cpp → BasicMapBaumWelchTrainer<ObservationVectorView>
 */
template<typename Obs>
class BasicMapBaumWelchTrainer : public BasicTrainer<Obs> {
public:
    using Base     = BasicTrainer<Obs>;
    using HmmType  = typename Base::HmmType;
    using ListType = typename Base::ListType;
    using SeqType  = typename ObsSeqTraits<Obs>::SeqType;

    /**
     * @param hmm          HMM to train in place.
     * @param obsLists     Observation sequences (must be non-empty).
     * @param pseudo_count Dirichlet pseudo-count c ≥ 0.
     * @throws std::invalid_argument if pseudo_count < 0.
     */
    BasicMapBaumWelchTrainer(HmmType& hmm, const ListType& obsLists,
                              double pseudo_count = 1.0);

    /** @brief Legacy pointer constructor. */
    BasicMapBaumWelchTrainer(HmmType* hmm, const ListType& obsLists,
                              double pseudo_count = 1.0);

    ~BasicMapBaumWelchTrainer() override = default;

    BasicMapBaumWelchTrainer(const BasicMapBaumWelchTrainer&) = delete;
    BasicMapBaumWelchTrainer& operator=(const BasicMapBaumWelchTrainer&) = delete;
    BasicMapBaumWelchTrainer(BasicMapBaumWelchTrainer&&) = default;
    BasicMapBaumWelchTrainer& operator=(BasicMapBaumWelchTrainer&&) = default;

    /** @brief One MAP-EM pass. @throws std::runtime_error if no valid sequences. */
    void train() override;

    /** @param c New pseudo-count. @throws std::invalid_argument if c < 0. */
    void setPseudoCount(double c);

    /** @return Current pseudo-count c. */
    [[nodiscard]] double getPseudoCount() const noexcept { return pseudo_count_; }

    /**
     * @brief Unnormalised log-prior of current HMM parameters.
     *
     * Returns  c · (Σ_{i,j} log A(i,j) + Σ_i log π_i)
     *         + c · Σ_{discrete i} Σ_k log B(i,k)   [scalar only]
     *
     * Normalising constants are omitted (invariant to λ).  Returns 0 when c = 0.
     * MAP convergence criterion:  logLikelihood + computeLogPrior().
     */
    [[nodiscard]] double computeLogPrior() const;

private:
    double pseudo_count_;

    static constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();

    /// Expected transition counts (xi) for one sequence — same as BaumWelchTrainer.
    static void accumulate_xi(const double* logAlphaData, const double* logBetaData,
                               const std::vector<double>& logEmitByTime,
                               const std::vector<double>& logTransT,
                               double logP, std::size_t T, std::size_t N,
                               bool hasZeroTransitions,
                               std::vector<double>& transNumT) noexcept;

    /// MAP M-step for π: add c to each numerator.
    static void m_step_pi_map(HmmType& hmm, std::size_t N,
                               const std::vector<double>& piNum, double c);

    /// MAP M-step for A: add c to each xi count.
    static void m_step_transitions_map(HmmType& hmm, std::size_t N,
                                        const std::vector<double>& transNumT,
                                        const std::vector<double>& transDen, double c);
};

// =============================================================================
// Inline method definitions
// =============================================================================

template<typename Obs>
BasicMapBaumWelchTrainer<Obs>::BasicMapBaumWelchTrainer(
        HmmType& hmm, const ListType& obsLists, double pseudo_count)
    : Base(hmm, obsLists), pseudo_count_(pseudo_count)
{
    if (pseudo_count < 0.0) {
        throw std::invalid_argument(
            "MapBaumWelchTrainer: pseudo_count must be >= 0 "
            "(sparse-inducing priors require Variational Bayes)");
    }
}

template<typename Obs>
BasicMapBaumWelchTrainer<Obs>::BasicMapBaumWelchTrainer(
        HmmType* hmm, const ListType& obsLists, double pseudo_count)
    : BasicMapBaumWelchTrainer(
          hmm ? *hmm : throw std::invalid_argument("HMM pointer cannot be null"),
          obsLists, pseudo_count) {}

template<typename Obs>
void BasicMapBaumWelchTrainer<Obs>::setPseudoCount(double c) {
    if (c < 0.0)
        throw std::invalid_argument("MapBaumWelchTrainer: pseudo_count must be >= 0");
    pseudo_count_ = c;
}

// ---------------------------------------------------------------------------
// computeLogPrior
// ---------------------------------------------------------------------------

template<typename Obs>
double BasicMapBaumWelchTrainer<Obs>::computeLogPrior() const {
    if (pseudo_count_ == 0.0) return 0.0;

    const HmmType& hmm = this->getHmmRef();
    const std::size_t N = static_cast<std::size_t>(hmm.getNumStates());
    const double c = pseudo_count_;
    double lp = 0.0;

    const Matrix& A = hmm.getTrans();
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            lp += c * (A(i, j) > 0.0 ? std::log(A(i, j)) : LOG_ZERO);

    const Vector& pi = hmm.getPi();
    for (std::size_t i = 0; i < N; ++i)
        lp += c * (pi(i) > 0.0 ? std::log(pi(i)) : LOG_ZERO);

    // Discrete emission prior — scalar path only (MV emissions are never discrete).
    if constexpr (std::is_same_v<Obs, double>) {
        for (std::size_t i = 0; i < N; ++i) {
            const auto& dist = hmm.getDistribution(i);
            if (!dist.isDiscrete()) continue;
            const auto& dd = static_cast<const DiscreteDistribution&>(dist);
            const std::size_t K = dd.getNumSymbols();
            for (std::size_t k = 0; k < K; ++k) {
                const double bk = dd.getSymbolProbability(k);
                lp += c * (bk > 0.0 ? std::log(bk) : LOG_ZERO);
            }
        }
    }
    return lp;
}

// ---------------------------------------------------------------------------
// train() — one full MAP-EM pass
// ---------------------------------------------------------------------------

template<typename Obs>
void BasicMapBaumWelchTrainer<Obs>::train() {
    HmmType& hmm = this->getHmmRef();
    const std::size_t N = static_cast<std::size_t>(hmm.getNumStates());
    const double c = pseudo_count_;

    const Matrix& curTrans = hmm.getTrans();
    std::vector<double> logTransT(N * N);
    bool hasZeroTransitions = false;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            const double a = curTrans(i, j);
            logTransT[j * N + i] = (a > 0.0) ? std::log(a) : LOG_ZERO;
            if (a <= 0.0) hasZeroTransitions = true;
        }
    }

    std::vector<double> piNum(N, 0.0);
    std::vector<double> transDen(N, 0.0);
    std::vector<double> transNumT(N * N, 0.0);
    std::size_t validSeqs = 0;

    if constexpr (std::is_same_v<Obs, double>) {
        // ----------------------------------------------------------------
        // Scalar path
        // ----------------------------------------------------------------
        std::size_t totalLen = 0;
        for (const auto& obs : this->getObservationLists()) totalLen += obs.size();

        std::vector<std::vector<double>> emisData(N);
        std::vector<std::vector<double>> emisWts(N);
        for (std::size_t i = 0; i < N; ++i) {
            emisData[i].reserve(totalLen);
            emisWts[i].reserve(totalLen);
        }

        for (const auto& obs : this->getObservationLists()) {
            const std::size_t T = obs.size();
            if (T == 0) continue;

            BasicForwardBackwardCalculator<double> fbc(hmm, obs);
            const double logP = fbc.getLogProbability();
            if (!std::isfinite(logP)) continue;

            const Matrix& logAlpha = fbc.getLogForwardVariables();
            const Matrix& logBeta  = fbc.getLogBackwardVariables();
            const double* alphaData = logAlpha.data();
            const double* betaData  = logBeta.data();

            for (std::size_t t = 0; t < T; ++t) {
                const double* aRow = alphaData + t * N;
                const double* bRow = betaData  + t * N;
                const double  oval = obs(t);
                for (std::size_t i = 0; i < N; ++i) {
                    const double g = std::exp(aRow[i] + bRow[i] - logP);
                    emisData[i].push_back(oval);
                    emisWts[i].push_back(g);
                    if (t == 0)     piNum[i]    += g;
                    if (t < T - 1)  transDen[i] += g;
                }
            }

            accumulate_xi(alphaData, betaData, fbc.getLogEmitByTime(), logTransT,
                          logP, T, N, hasZeroTransitions, transNumT);
            ++validSeqs;
        }

        if (validSeqs == 0) {
            throw std::runtime_error(
                "MapBaumWelchTrainer: no valid observation sequences "
                "(all had zero probability under the current model)");
        }

        m_step_pi_map(hmm, N, piNum, c);
        m_step_transitions_map(hmm, N, transNumT, transDen, c);

        // Emission M-step — MLE fit for all, then Dirichlet smoothing for discrete.
        for (std::size_t i = 0; i < N; ++i) {
            const std::size_t M = emisData[i].size();
            auto& dist = hmm.getDistribution(i);
            if (M == 0) { dist.reset(); continue; }
            dist.fit(std::span<const double>(emisData[i].data(), M),
                     std::span<const double>(emisWts[i].data(), M));
            // Dirichlet smoothing for discrete distributions.
            if (c > 0.0 && dist.isDiscrete()) {
                auto& dd = static_cast<DiscreteDistribution&>(dist);
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
    } else {
        // ----------------------------------------------------------------
        // Multivariate path — identical structure to BasicBaumWelchTrainer MV,
        // but with MAP M-step for π and A instead of plain MLE.
        // Continuous MV distributions are always fitted by MLE (no Dirichlet on emissions).
        // ----------------------------------------------------------------
        std::vector<std::vector<ObservationVectorView>> emisViews(N);
        std::vector<std::vector<double>> emisWts(N);

        for (const auto& obs : this->getObservationLists()) {
            const std::size_t T = ObsSeqTraits<Obs>::sequence_length(obs);
            if (T == 0) continue;

            BasicForwardBackwardCalculator<Obs> fbc(hmm, obs);
            const double logP = fbc.getLogProbability();
            if (!std::isfinite(logP)) continue;

            const Matrix& logAlpha = fbc.getLogForwardVariables();
            const Matrix& logBeta  = fbc.getLogBackwardVariables();
            const double* alphaData = logAlpha.data();
            const double* betaData  = logBeta.data();

            for (std::size_t t = 0; t < T; ++t) {
                const double*         aRow = alphaData + t * N;
                const double*         bRow = betaData  + t * N;
                ObservationVectorView view = row_view(obs, t);
                for (std::size_t i = 0; i < N; ++i) {
                    const double g = std::exp(aRow[i] + bRow[i] - logP);
                    emisViews[i].push_back(view);
                    emisWts[i].push_back(g);
                    if (t == 0)     piNum[i]    += g;
                    if (t < T - 1)  transDen[i] += g;
                }
            }

            accumulate_xi(alphaData, betaData, fbc.getLogEmitByTime(), logTransT,
                          logP, T, N, hasZeroTransitions, transNumT);
            ++validSeqs;
        }

        if (validSeqs == 0) {
            throw std::runtime_error(
                "MapBaumWelchTrainer: no valid observation sequences "
                "(all had zero probability under the current model)");
        }

        m_step_pi_map(hmm, N, piNum, c);
        m_step_transitions_map(hmm, N, transNumT, transDen, c);

        // Emission M-step — MLE for all MV distributions (no discrete smoothing).
        for (std::size_t i = 0; i < N; ++i) {
            const std::size_t M = emisViews[i].size();
            if (M == 0) { hmm.getDistribution(i).reset(); continue; }
            hmm.getDistribution(i).fit(
                std::span<const ObservationVectorView>(emisViews[i].data(), M),
                std::span<const double>(emisWts[i].data(), M));
        }
    }
}

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

template<typename Obs>
void BasicMapBaumWelchTrainer<Obs>::accumulate_xi(
        const double* logAlphaData, const double* logBetaData,
        const std::vector<double>& logEmitByTime,
        const std::vector<double>& logTransT,
        double logP, std::size_t T, std::size_t N,
        bool hasZeroTransitions,
        std::vector<double>& transNumT) noexcept
{
    if (hasZeroTransitions) {
        for (std::size_t t = 0; t + 1 < T; ++t) {
            const double* alphaRow    = logAlphaData + t * N;
            const double* betaNextRow = logBetaData  + (t + 1) * N;
            const double* emitNextRow = logEmitByTime.data() + (t + 1) * N;
            for (std::size_t j = 0; j < N; ++j) {
                const double emitBetaNext = emitNextRow[j] + betaNextRow[j] - logP;
                const double* transCol    = logTransT.data() + j * N;
                double*       transNumCol = transNumT.data() + j * N;
                for (std::size_t i = 0; i < N; ++i) {
                    if (transCol[i] == LOG_ZERO) continue;
                    transNumCol[i] += std::exp(alphaRow[i] + transCol[i] + emitBetaNext);
                }
            }
        }
    } else {
        for (std::size_t t = 0; t + 1 < T; ++t) {
            const double* alphaRow    = logAlphaData + t * N;
            const double* betaNextRow = logBetaData  + (t + 1) * N;
            const double* emitNextRow = logEmitByTime.data() + (t + 1) * N;
            for (std::size_t j = 0; j < N; ++j) {
                const double emitBetaNext = emitNextRow[j] + betaNextRow[j] - logP;
                const double* transCol    = logTransT.data() + j * N;
                double*       transNumCol = transNumT.data() + j * N;
                performance::detail::TranscendentalKernels::accumulate_exp_sum2_bias(
                    transNumCol, alphaRow, transCol, N, emitBetaNext);
            }
        }
    }
}

template<typename Obs>
void BasicMapBaumWelchTrainer<Obs>::m_step_pi_map(
        HmmType& hmm, std::size_t N,
        const std::vector<double>& piNum, double c)
{
    const double piSum = std::accumulate(piNum.begin(), piNum.end(), 0.0);
    const double denom = piSum + static_cast<double>(N) * c;
    Vector pi(N);
    for (std::size_t i = 0; i < N; ++i)
        pi(i) = (denom > 0.0) ? (piNum[i] + c) / denom : 1.0 / static_cast<double>(N);
    hmm.setPi(pi);
}

template<typename Obs>
void BasicMapBaumWelchTrainer<Obs>::m_step_transitions_map(
        HmmType& hmm, std::size_t N,
        const std::vector<double>& transNumT,
        const std::vector<double>& transDen, double c)
{
    const double Nc = static_cast<double>(N) * c;
    Matrix newTrans(N, N);
    for (std::size_t i = 0; i < N; ++i) {
        const double denom = transDen[i] + Nc;
        for (std::size_t j = 0; j < N; ++j) {
            newTrans(i, j) = (denom > 0.0)
                ? (transNumT[j * N + i] + c) / denom
                : 1.0 / static_cast<double>(N);
        }
    }
    hmm.setTrans(newTrans);
}

// =============================================================================
// Explicit instantiation declarations.
// =============================================================================
extern template class BasicMapBaumWelchTrainer<double>;
extern template class BasicMapBaumWelchTrainer<ObservationVectorView>;

/// @brief Scalar alias (v3-compatible name).
using MapBaumWelchTrainer = BasicMapBaumWelchTrainer<double>;

} // namespace libhmm
