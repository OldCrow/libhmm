#pragma once

/**
 * @file bw_estep.h
 * @brief Shared Baum-Welch E-step types and functions.
 *
 * Extracted from BasicBaumWelchTrainer and BasicMapBaumWelchTrainer to
 * eliminate the ~150-line near-verbatim duplication that caused drift
 * between the two (issue #55 / audit Finding 4).
 *
 * Finding 5 (redundant per-state observation copies) is also addressed here:
 * BwEStepBuffers holds a single shared observation vector rather than N
 * identical per-state copies.  Only the per-state gamma weights differ.
 *
 * Internal header — include only from trainer implementations.
 */

#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/detail/log_utils.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/performance/transcendental_kernels.h"

namespace libhmm {

// =============================================================================
// E-step accumulator types
// =============================================================================

/**
 * @brief Observation element type for the Baum-Welch E-step.
 *
 * `double` on the scalar path; `ObservationVectorView` on the MV path.
 */
template <typename Obs>
using BwEmisElem = std::conditional_t<std::is_same_v<Obs, double>, double, ObservationVectorView>;

/**
 * @brief Mutable E-step accumulators, shared by BaumWelchTrainer and
 *        MapBaumWelchTrainer.
 *
 * Finding 5 fix: @p emisObs holds one copy of all observations concatenated
 * across all processed sequences.  @p emisWts[i] holds the matching per-state
 * gamma weights at each position.  The M-step passes @p emisObs as the data
 * span and @p emisWts[i] as the weight span for each state's fit() call.
 * Previously, N identical copies of @p emisObs were stored — one per state.
 */
template <typename Obs>
struct BwEStepBuffers {
    using EmisElem = BwEmisElem<Obs>;

    std::vector<EmisElem> &emisObs;            ///< Shared observation sequence (one copy)
    std::vector<std::vector<double>> &emisWts; ///< Per-state gamma weights
    std::vector<double> &piNum;                ///< π numerator (γ at t=0 per state)
    std::vector<double> &transDen;             ///< Transition denominator (γ_{t<T-1} per state)
    std::vector<double> &transNumT;            ///< Expected transition counts, column-major N×N
};

// =============================================================================
// Shared E-step functions
// =============================================================================

/**
 * @brief Accumulate expected transition counts (xi) for one sequence.
 *
 * Observation-type-independent: operates entirely on pre-computed log-space
 * buffers.  Dispatches to a zero-skip path (sparse transitions) or the
 * branch-free SIMD path (dense model).
 *
 * @param logAlphaData  Row-major T×N forward variables (log-space).
 * @param logBetaData   Row-major T×N backward variables (log-space).
 * @param logEmitByTime Time-major T×N log-emission buffer from the FBC.
 * @param logTransT     Column-major N×N log-transition matrix.
 * @param logP          log P(O|λ) for this sequence.
 * @param T             Sequence length.
 * @param N             Number of states.
 * @param hasZeroTransitions  True if any A(i,j) = 0 (enables zero-skip path).
 * @param transNumT     [in/out] Column-major N×N expected transition counts.
 */
inline void bw_accumulate_xi(const double *logAlphaData, const double *logBetaData,
                             const std::vector<double> &logEmitByTime,
                             const std::vector<double> &logTransT, double logP, std::size_t T,
                             std::size_t N, bool hasZeroTransitions,
                             std::vector<double> &transNumT) noexcept {
    constexpr double LOG_ZERO = detail::LOG_ZERO;
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

/**
 * @brief Run the E-step for one observation sequence.
 *
 * Runs BasicForwardBackwardCalculator<Obs>, accumulates gamma statistics
 * (one shared observation entry per timestep, per-state gamma weights,
 * π numerator, transition denominator) and xi statistics (expected
 * transition counts) into @p bufs.
 *
 * The FBC's log-emission buffer is reused for the xi accumulation pass,
 * avoiding a second emission evaluation.
 *
 * @return Finite log P(O|λ) if the sequence contributed; −∞ if skipped
 *         (empty sequence or zero probability under the current model).
 */
template <typename Obs>
[[nodiscard]] double bw_accum_one_sequence(const BasicHmm<Obs> &hmm,
                                           const typename ObsSeqTraits<Obs>::SeqType &obs,
                                           std::size_t N, const std::vector<double> &logTransT,
                                           bool hasZeroTransitions, BwEStepBuffers<Obs> &bufs) {
    using EmisElem = BwEmisElem<Obs>;

    const std::size_t T = ObsSeqTraits<Obs>::sequence_length(obs);
    if (T == 0)
        return -std::numeric_limits<double>::infinity();

    BasicForwardBackwardCalculator<Obs> fbc(hmm, obs);
    const double logP = fbc.getLogProbability();
    if (!std::isfinite(logP))
        return -std::numeric_limits<double>::infinity();

    const double *alphaData = fbc.getLogForwardVariables().data();
    const double *betaData = fbc.getLogBackwardVariables().data();

    for (std::size_t t = 0; t < T; ++t) {
        const double *aRow = alphaData + t * N;
        const double *bRow = betaData + t * N;
        // One observation entry per timestep — shared across all states.
        const EmisElem obs_t = [&]() -> EmisElem {
            if constexpr (std::is_same_v<Obs, double>)
                return obs(t);
            else
                return row_view(obs, t);
        }();
        bufs.emisObs.push_back(obs_t);
        for (std::size_t i = 0; i < N; ++i) {
            const double g = std::exp(aRow[i] + bRow[i] - logP);
            bufs.emisWts[i].push_back(g);
            if (t == 0)
                bufs.piNum[i] += g;
            if (t < T - 1)
                bufs.transDen[i] += g;
        }
    }

    // Reuse the FBC's emission buffer — avoids a second emission evaluation.
    bw_accumulate_xi(alphaData, betaData, fbc.getLogEmitByTime(), logTransT, logP, T, N,
                     hasZeroTransitions, bufs.transNumT);
    return logP;
}

} // namespace libhmm
