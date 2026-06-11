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
    // Emission accumulator types
    //
    // EmisElem adapts to the observation type:
    //   Obs=double             → double              (scalar observation value)
    //   Obs=ObservationVectorView → ObservationVectorView (non-owning row span)
    // EmisAccumType is a per-state vector-of-vectors of these elements.
    // -------------------------------------------------------------------------
    using EmisElem = std::conditional_t<std::is_same_v<Obs, double>,
                                         double, ObservationVectorView>;
    using EmisAccumType = std::vector<std::vector<EmisElem>>;

    /// Mutable accumulators passed through the E-step as a single aggregate.
    struct EStepBuffers {
        EmisAccumType&                    emisAccum;
        std::vector<std::vector<double>>& emisWts;
        std::vector<double>&              piNum;
        std::vector<double>&              transDen;
        std::vector<double>&              transNumT;
    };

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /**
     * @brief Run the E-step for one observation sequence.
     *
     * Runs BasicForwardBackwardCalculator, accumulates gamma statistics
     * (emission data/weights, π numerator, transition denominator) and
     * xi statistics (expected transition counts) into @p bufs.
     *
     * @return true if the sequence contributed (finite log-probability);
     *         false if the sequence was skipped (empty or zero probability).
     */
    [[nodiscard]] static bool accum_one_sequence(const HmmType& hmm, const SeqType& obs,
                                    std::size_t N,
                                    const std::vector<double>& logTransT,
                                    bool hasZeroTransitions,
                                    EStepBuffers& bufs);

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

    std::vector<double> logTransT(N * N);
    bool hasZeroTransitions = false;
    this->precompute_log_trans_flat(hmm, N, logTransT, hasZeroTransitions);

    // Accumulators for π, transitions (shared), and emissions (type-specific).
    std::vector<double> piNum(N, 0.0);
    std::vector<double> transDen(N, 0.0);
    std::vector<double> transNumT(N * N, 0.0);
    EmisAccumType       emisAccum(N);
    std::vector<std::vector<double>> emisWts(N);

    // Pre-allocate emission buffers on the scalar path to avoid repeated
    // reallocations when accumulating over many sequences.
    if constexpr (std::is_same_v<Obs, double>) {
        std::size_t totalLen = 0;
        for (const auto& obs : this->getObservationLists()) totalLen += obs.size();
        for (std::size_t i = 0; i < N; ++i) {
            emisAccum[i].reserve(totalLen);
            emisWts[i].reserve(totalLen);
        }
    }

    EStepBuffers bufs{emisAccum, emisWts, piNum, transDen, transNumT};
    std::size_t validSeqs = 0;
    for (const auto& obs : this->getObservationLists()) {
        if (accum_one_sequence(hmm, obs, N, logTransT, hasZeroTransitions, bufs))
            ++validSeqs;
    }

    if (validSeqs == 0) {
        throw std::runtime_error(
            "BaumWelchTrainer: no valid observation sequences "
            "(all had zero probability under the current model)");
    }

    m_step_pi(hmm, N, piNum);
    m_step_transitions(hmm, N, transNumT, transDen);

    // Emission M-step: EmisElem resolves to double (scalar) or
    // ObservationVectorView (MV), selecting the correct fit() overload.
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t M = emisAccum[i].size();
        if (M == 0) { hmm.getDistribution(i).reset(); continue; }
        hmm.getDistribution(i).fit(
            std::span<const EmisElem>(emisAccum[i].data(), M),
            std::span<const double>(emisWts[i].data(), M));
    }
}

// ---------------------------------------------------------------------------
// accum_one_sequence — E-step for a single observation sequence
// ---------------------------------------------------------------------------

template<typename Obs>
bool BasicBaumWelchTrainer<Obs>::accum_one_sequence(
        const HmmType& hmm, const SeqType& obs, std::size_t N,
        const std::vector<double>& logTransT, bool hasZeroTransitions,
        EStepBuffers& bufs)
{
    const std::size_t T = ObsSeqTraits<Obs>::sequence_length(obs);
    if (T == 0) return false;

    BasicForwardBackwardCalculator<Obs> fbc(hmm, obs);
    const double logP = fbc.getLogProbability();
    if (!std::isfinite(logP)) return false;

    const double* alphaData = fbc.getLogForwardVariables().data();
    const double* betaData  = fbc.getLogBackwardVariables().data();

    for (std::size_t t = 0; t < T; ++t) {
        const double* aRow = alphaData + t * N;
        const double* bRow = betaData  + t * N;
        // Compute observation value/view once per timestep, outside the state loop.
        const EmisElem obs_t = [&]() -> EmisElem {
            if constexpr (std::is_same_v<Obs, double>) return obs(t);
            else return row_view(obs, t);
        }();
        for (std::size_t i = 0; i < N; ++i) {
            const double g = std::exp(aRow[i] + bRow[i] - logP);
            bufs.emisAccum[i].push_back(obs_t);
            bufs.emisWts[i].push_back(g);
            if (t == 0)    bufs.piNum[i]    += g;
            if (t < T - 1) bufs.transDen[i] += g;
        }
    }

    // Reuse the FBC's emission buffer — avoids a second emission evaluation.
    accumulate_xi(alphaData, betaData, fbc.getLogEmitByTime(), logTransT,
                  logP, T, N, hasZeroTransitions, bufs.transNumT);
    return true;
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
