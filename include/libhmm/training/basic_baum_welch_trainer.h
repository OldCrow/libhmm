#pragma once

#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/performance/transcendental_kernels.h"
#include "libhmm/training/basic_trainer.h"

namespace libhmm {

/**
 * @brief Log-space Baum-Welch (EM) trainer, parameterised on observation type.
 *
 * @tparam Obs  `double` (scalar, v3-compatible) or `ObservationVectorView` (v4 MV).
 *
 * Performs one EM iteration per train() call:
 *   E-step: forward-backward via BasicForwardBackwardCalculator<Obs>.
 *   M-step: update π, transition matrix, and call fit() on each emission distribution.
 *
 * Scalar path (Obs=double):
 *   Accumulates (observation_value, gamma) pairs per state and calls the weighted
 *   scalar fit() overload.  The log-emission buffer from the FBC is reused for the
 *   xi accumulation pass, avoiding a second emission evaluation.
 *
 * Multivariate path (Obs=ObservationVectorView):
 *   Accumulates (ObservationVectorView, gamma) pairs per state.  Views are non-owning
 *   spans into the original ObservationMatrix data; no observation data is copied.
 *   At M-step time the per-state span vectors are passed directly to fit().
 *
 * @throws std::runtime_error from train() if no sequence has finite log-probability
 *         under the current model (all sequences have probability zero).
 *
 * Two explicit instantiations are compiled with LIBHMM_BEST_SIMD_FLAGS:
 *   - src/training/baum_welch_trainer.cpp    → BasicBaumWelchTrainer<double>
 *   - src/training/baum_welch_trainer_mv.cpp → BasicBaumWelchTrainer<ObservationVectorView>
 */
template<typename Obs>
class BasicBaumWelchTrainer : public BasicTrainer<Obs> {
public:
    using Base    = BasicTrainer<Obs>;
    using HmmType = typename Base::HmmType;
    using ListType = typename Base::ListType;
    using SeqType  = typename ObsSeqTraits<Obs>::SeqType;

    BasicBaumWelchTrainer(HmmType& hmm, const ListType& obsLists);

    /** @brief Legacy pointer constructor for backward compatibility. */
    BasicBaumWelchTrainer(HmmType* hmm, const ListType& obsLists);

    ~BasicBaumWelchTrainer() override = default;

    /** @brief Execute one full EM pass, updating the HMM in place. */
    void train() override;

private:
    static constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();

    // -------------------------------------------------------------------------
    // Shared helpers (both Obs specialisations use these)
    // -------------------------------------------------------------------------

    /**
     * @brief Accumulate expected transition counts (xi) for one sequence.
     *
     * Dispatches to a branch-free SIMD path (dense model) or a zero-skip path
     * (sparse model) based on @p hasZeroTransitions.
     */
    static void accumulate_xi(const double* logAlphaData, const double* logBetaData,
                               const std::vector<double>& logEmitByTime,
                               const std::vector<double>& logTransT,
                               double logP, std::size_t T, std::size_t N,
                               bool hasZeroTransitions,
                               std::vector<double>& transNumT) noexcept;

    /// M-step: normalise piNum and update the HMM initial-state distribution.
    static void m_step_pi(HmmType& hmm, std::size_t N, const std::vector<double>& piNum);

    /// M-step: normalise transNumT / transDen and update the transition matrix.
    static void m_step_transitions(HmmType& hmm, std::size_t N,
                                    const std::vector<double>& transNumT,
                                    const std::vector<double>& transDen);

    /** @brief log(exp(a) + exp(b)) — numerically stable. */
    [[nodiscard]] static double logSumExp(double a, double b) noexcept;
};

// =============================================================================
// Inline method definitions
// =============================================================================

template<typename Obs>
BasicBaumWelchTrainer<Obs>::BasicBaumWelchTrainer(HmmType& hmm, const ListType& obsLists)
    : Base(hmm, obsLists) {}

template<typename Obs>
BasicBaumWelchTrainer<Obs>::BasicBaumWelchTrainer(HmmType* hmm, const ListType& obsLists)
    : Base(hmm ? *hmm : throw std::invalid_argument("HMM pointer cannot be null"), obsLists) {}

// ---------------------------------------------------------------------------
// train() — one full EM pass over all observation sequences
// ---------------------------------------------------------------------------

template<typename Obs>
void BasicBaumWelchTrainer<Obs>::train() {
    HmmType& hmm = this->getHmmRef();
    const std::size_t N = static_cast<std::size_t>(hmm.getNumStates());

    // Precompute column-major transposed log-transition matrix:
    // logTransT[j * N + i] = log a_{ij}
    // Column-major storage matches the xi inner loop for contiguous reads.
    const Matrix& curTrans = hmm.getTrans();
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

    // Accumulators for π and transition parameters (shared for both paths).
    std::vector<double> piNum(N, 0.0);
    std::vector<double> transDen(N, 0.0);
    // Column-major: transNumT[j * N + i] = expected count for transition i→j.
    std::vector<double> transNumT(N * N, 0.0);

    std::size_t validSeqs = 0;

    if constexpr (std::is_same_v<Obs, double>) {
        // ----------------------------------------------------------------
        // Scalar path — accumulate (observation_value, gamma) per state.
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

            // Accumulate gamma: emission data/weights, pi numerator, transition denominator.
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

            // Reuse emission buffer from FBC — avoids a second evaluation pass.
            accumulate_xi(alphaData, betaData, fbc.getLogEmitByTime(), logTransT,
                          logP, T, N, hasZeroTransitions, transNumT);
            ++validSeqs;
        }

        if (validSeqs == 0) {
            throw std::runtime_error(
                "BaumWelchTrainer: no valid observation sequences "
                "(all had zero probability under the current model)");
        }

        m_step_pi(hmm, N, piNum);
        m_step_transitions(hmm, N, transNumT, transDen);

        // Emission M-step — weighted scalar fit.
        for (std::size_t i = 0; i < N; ++i) {
            const std::size_t M = emisData[i].size();
            if (M == 0) { hmm.getDistribution(i).reset(); continue; }
            hmm.getDistribution(i).fit(
                std::span<const double>(emisData[i].data(), M),
                std::span<const double>(emisWts[i].data(), M));
        }
    } else {
        // ----------------------------------------------------------------
        // Multivariate path — accumulate (ObservationVectorView, gamma) per state.
        // Views are non-owning spans; observation data is never copied.
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

            // Accumulate gamma with row-view references into the observation matrix.
            for (std::size_t t = 0; t < T; ++t) {
                const double*          aRow = alphaData + t * N;
                const double*          bRow = betaData  + t * N;
                ObservationVectorView  view = row_view(obs, t);
                for (std::size_t i = 0; i < N; ++i) {
                    const double g = std::exp(aRow[i] + bRow[i] - logP);
                    emisViews[i].push_back(view);
                    emisWts[i].push_back(g);
                    if (t == 0)     piNum[i]    += g;
                    if (t < T - 1)  transDen[i] += g;
                }
            }

            // Reuse emission buffer from FBC.
            accumulate_xi(alphaData, betaData, fbc.getLogEmitByTime(), logTransT,
                          logP, T, N, hasZeroTransitions, transNumT);
            ++validSeqs;
        }

        if (validSeqs == 0) {
            throw std::runtime_error(
                "BaumWelchTrainer: no valid observation sequences "
                "(all had zero probability under the current model)");
        }

        m_step_pi(hmm, N, piNum);
        m_step_transitions(hmm, N, transNumT, transDen);

        // Emission M-step — weighted multivariate fit.
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
// Shared static helpers
// ---------------------------------------------------------------------------

template<typename Obs>
void BasicBaumWelchTrainer<Obs>::accumulate_xi(
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
void BasicBaumWelchTrainer<Obs>::m_step_pi(
        HmmType& hmm, std::size_t N, const std::vector<double>& piNum)
{
    double piSum = 0.0;
    for (std::size_t i = 0; i < N; ++i) piSum += piNum[i];
    Vector pi(N);
    for (std::size_t i = 0; i < N; ++i) {
        pi(i) = (piSum > 0.0) ? piNum[i] / piSum : 1.0 / static_cast<double>(N);
    }
    hmm.setPi(pi);
}

template<typename Obs>
void BasicBaumWelchTrainer<Obs>::m_step_transitions(
        HmmType& hmm, std::size_t N,
        const std::vector<double>& transNumT,
        const std::vector<double>& transDen)
{
    Matrix newTrans(N, N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            newTrans(i, j) = (transDen[i] > 0.0)
                ? transNumT[j * N + i] / transDen[i]
                : 1.0 / static_cast<double>(N);
        }
    }
    hmm.setTrans(newTrans);
}

template<typename Obs>
double BasicBaumWelchTrainer<Obs>::logSumExp(double a, double b) noexcept {
    if (a == LOG_ZERO) return b;
    if (b == LOG_ZERO) return a;
    if (a > b) return a + std::log1p(std::exp(b - a));
    return b + std::log1p(std::exp(a - b));
}

// =============================================================================
// Explicit instantiation declarations.
// Suppress implicit instantiation in consumer TUs.
// Definitions are in baum_welch_trainer.cpp (scalar) and
// baum_welch_trainer_mv.cpp (multivariate).
// =============================================================================
extern template class BasicBaumWelchTrainer<double>;
extern template class BasicBaumWelchTrainer<ObservationVectorView>;

/// @brief Scalar alias (v3-compatible name).
using BaumWelchTrainer = BasicBaumWelchTrainer<double>;

} // namespace libhmm
