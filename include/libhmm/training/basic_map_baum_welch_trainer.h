#pragma once

#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/detail/log_utils.h"
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
template <typename Obs>
class BasicMapBaumWelchTrainer : public BasicTrainer<Obs> {
public:
    using Base = BasicTrainer<Obs>;
    using HmmType = typename Base::HmmType;
    using ListType = typename Base::ListType;
    using SeqType = typename ObsSeqTraits<Obs>::SeqType;

    /**
     * @param hmm          HMM to train in place.
     * @param obsLists     Observation sequences (must be non-empty).
     * @param pseudo_count Dirichlet pseudo-count c ≥ 0.
     * @throws std::invalid_argument if pseudo_count < 0.
     */
    BasicMapBaumWelchTrainer(HmmType &hmm, const ListType &obsLists, double pseudo_count = 1.0);

    /** @brief Legacy pointer constructor. */
    BasicMapBaumWelchTrainer(HmmType *hmm, const ListType &obsLists, double pseudo_count = 1.0);

    ~BasicMapBaumWelchTrainer() override = default;

    BasicMapBaumWelchTrainer(const BasicMapBaumWelchTrainer &) = delete;
    BasicMapBaumWelchTrainer &operator=(const BasicMapBaumWelchTrainer &) = delete;
    BasicMapBaumWelchTrainer(BasicMapBaumWelchTrainer &&) = default;
    BasicMapBaumWelchTrainer &operator=(BasicMapBaumWelchTrainer &&) = default;

    /** @brief One MAP-EM pass. @throws std::runtime_error if no valid sequences. */
    void train() override;

    /** @param c New pseudo-count. @throws std::invalid_argument if c < 0. */
    void setPseudoCount(double c);

    /** @return Current pseudo-count c. */
    [[nodiscard]] double getPseudoCount() const noexcept { return pseudo_count_; }

    /**
     * @return Total finite E-step log-probability from the last train() call:
     *         Σ log P(O_k | λ) over all sequences with finite probability.
     *         Reset to −∞ before each train() call; remains −∞ if no valid
     *         sequences were found.
     */
    [[nodiscard]] double getLastLogProbability() const noexcept { return lastLogProb_; }

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
    double lastLogProb_{-std::numeric_limits<double>::infinity()};

    static constexpr double LOG_ZERO = detail::LOG_ZERO;

    // Same type-adapting accumulator pattern as BasicBaumWelchTrainer.
    using EmisElem = std::conditional_t<std::is_same_v<Obs, double>, double, ObservationVectorView>;
    using EmisAccumType = std::vector<std::vector<EmisElem>>;

    /// Mutable accumulators passed through the E-step as a single aggregate.
    struct EStepBuffers {
        EmisAccumType &emisAccum;
        std::vector<std::vector<double>> &emisWts;
        std::vector<double> &piNum;
        std::vector<double> &transDen;
        std::vector<double> &transNumT;
    };

    /**
     * @brief Run the E-step for one sequence (same logic as BaumWelchTrainer).
     * @return Finite log P(O|λ) if the sequence contributed; −∞ if skipped.
     */
    [[nodiscard]] static double accum_one_sequence(const HmmType &hmm, const SeqType &obs,
                                                   std::size_t N,
                                                   const std::vector<double> &logTransT,
                                                   bool hasZeroTransitions, EStepBuffers &bufs);

    /**
     * @brief Dirichlet-smoothed log-prior over discrete emission distributions.
     * Only compiled and called for the scalar specialisation.
     */
    [[nodiscard]] static double discrete_emission_log_prior(const HmmType &hmm, std::size_t N,
                                                            double c);

    /**
     * @brief Apply Dirichlet smoothing to a discrete emission distribution
     * after MLE fitting.  Only called for the scalar specialisation.
     */
    static void apply_discrete_smoothing(HmmType &hmm, std::size_t state,
                                         const std::vector<double> &wts, double c);

    /// Expected transition counts (xi) for one sequence.
    static void accumulate_xi(const double *logAlphaData, const double *logBetaData,
                              const std::vector<double> &logEmitByTime,
                              const std::vector<double> &logTransT, double logP, std::size_t T,
                              std::size_t N, bool hasZeroTransitions,
                              std::vector<double> &transNumT) noexcept;

    /// MAP M-step for π: add c to each numerator.
    static void m_step_pi_map(HmmType &hmm, std::size_t N, const std::vector<double> &piNum,
                              double c);

    /// MAP M-step for A: add c to each xi count.
    static void m_step_transitions_map(HmmType &hmm, std::size_t N,
                                       const std::vector<double> &transNumT,
                                       const std::vector<double> &transDen, double c);
};

// =============================================================================
// Inline method definitions
// =============================================================================

template <typename Obs>
BasicMapBaumWelchTrainer<Obs>::BasicMapBaumWelchTrainer(HmmType &hmm, const ListType &obsLists,
                                                        double pseudo_count)
    : Base(hmm, obsLists), pseudo_count_(pseudo_count) {
    // Use !(>= 0) so that NaN also fails the check (NaN < 0 is false).
    if (!(pseudo_count >= 0.0)) {
        throw std::invalid_argument("MapBaumWelchTrainer: pseudo_count must be >= 0 "
                                    "(sparse-inducing priors require Variational Bayes)");
    }
}

template <typename Obs>
BasicMapBaumWelchTrainer<Obs>::BasicMapBaumWelchTrainer(HmmType *hmm, const ListType &obsLists,
                                                        double pseudo_count)
    : BasicMapBaumWelchTrainer(hmm ? *hmm
                                   : throw std::invalid_argument("HMM pointer cannot be null"),
                               obsLists, pseudo_count) {}

template <typename Obs>
void BasicMapBaumWelchTrainer<Obs>::setPseudoCount(double c) {
    // Use !(>= 0) so that NaN also fails the check (NaN < 0 is false).
    if (!(c >= 0.0))
        throw std::invalid_argument("MapBaumWelchTrainer: pseudo_count must be >= 0");
    pseudo_count_ = c;
}

// ---------------------------------------------------------------------------
// computeLogPrior
// ---------------------------------------------------------------------------

template <typename Obs>
double BasicMapBaumWelchTrainer<Obs>::discrete_emission_log_prior(const HmmType &hmm, std::size_t N,
                                                                  double c) {
    // DiscreteDistribution only exists on the scalar path; MV emissions are
    // always continuous.  The if constexpr keeps the downcast from being
    // instantiated for Obs=ObservationVectorView.
    if constexpr (std::is_same_v<Obs, double>) {
        double lp = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            const auto &dist = hmm.getDistribution(i);
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
    return 0.0;
}

template <typename Obs>
double BasicMapBaumWelchTrainer<Obs>::computeLogPrior() const {
    if (pseudo_count_ == 0.0)
        return 0.0;

    const HmmType &hmm = this->getHmmRef();
    const std::size_t N = hmm.getNumStatesModern();
    const double c = pseudo_count_;
    double lp = 0.0;

    const Matrix &A = hmm.getTrans();
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            lp += c * (A(i, j) > 0.0 ? std::log(A(i, j)) : LOG_ZERO);

    const Vector &pi = hmm.getPi();
    for (std::size_t i = 0; i < N; ++i)
        lp += c * (pi(i) > 0.0 ? std::log(pi(i)) : LOG_ZERO);

    // Discrete emission prior is only applicable on the scalar path.
    if constexpr (std::is_same_v<Obs, double>)
        lp += discrete_emission_log_prior(hmm, N, c);

    return lp;
}

// ---------------------------------------------------------------------------
// train() — one full MAP-EM pass
// ---------------------------------------------------------------------------

template <typename Obs>
void BasicMapBaumWelchTrainer<Obs>::apply_discrete_smoothing(HmmType &hmm, std::size_t state,
                                                             const std::vector<double> &wts,
                                                             double c) {
    if constexpr (std::is_same_v<Obs, double>) {
        if (c <= 0.0)
            return; // no smoothing when pseudo-count is zero
        auto &dist = hmm.getDistribution(state);
        if (!dist.isDiscrete())
            return;
        auto &dd = static_cast<DiscreteDistribution &>(dist);
        const std::size_t K = dd.getNumSymbols();
        const double sumW = std::accumulate(wts.begin(), wts.end(), 0.0);
        if (sumW <= 0.0)
            return;
        const double denom = sumW + static_cast<double>(K) * c;
        for (std::size_t k = 0; k < K; ++k) {
            dd.setProbability(static_cast<double>(k),
                              (dd.getSymbolProbability(k) * sumW + c) / denom);
        }
        // Re-normalize: guards against un-normalized probabilities when
        // DiscreteDistribution::fit() used an inflated sumW denominator
        // (e.g. out-of-range observations counted in weight sum but not in
        // per-symbol bins), which leaves the pre-smoothing pdf summing to < 1.
        double total = 0.0;
        for (std::size_t k = 0; k < K; ++k)
            total += dd.getSymbolProbability(k);
        if (total > 0.0 && std::isfinite(total)) {
            const double inv_total = 1.0 / total;
            for (std::size_t k = 0; k < K; ++k)
                dd.setProbability(static_cast<double>(k), dd.getSymbolProbability(k) * inv_total);
        }
    }
}

template <typename Obs>
double BasicMapBaumWelchTrainer<Obs>::accum_one_sequence(const HmmType &hmm, const SeqType &obs,
                                                         std::size_t N,
                                                         const std::vector<double> &logTransT,
                                                         bool hasZeroTransitions,
                                                         EStepBuffers &bufs) {
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
        const EmisElem obs_t = [&]() -> EmisElem {
            if constexpr (std::is_same_v<Obs, double>)
                return obs(t);
            else
                return row_view(obs, t);
        }();
        for (std::size_t i = 0; i < N; ++i) {
            const double g = std::exp(aRow[i] + bRow[i] - logP);
            bufs.emisAccum[i].push_back(obs_t);
            bufs.emisWts[i].push_back(g);
            if (t == 0)
                bufs.piNum[i] += g;
            if (t < T - 1)
                bufs.transDen[i] += g;
        }
    }
    accumulate_xi(alphaData, betaData, fbc.getLogEmitByTime(), logTransT, logP, T, N,
                  hasZeroTransitions, bufs.transNumT);
    return logP;
}

template <typename Obs>
void BasicMapBaumWelchTrainer<Obs>::train() {
    HmmType &hmm = this->getHmmRef();
    const std::size_t N = hmm.getNumStatesModern();
    const double c = pseudo_count_;

    std::vector<double> logTransT(N * N);
    bool hasZeroTransitions = false;
    this->precompute_log_trans_flat(hmm, N, logTransT, hasZeroTransitions);

    std::vector<double> piNum(N, 0.0);
    std::vector<double> transDen(N, 0.0);
    std::vector<double> transNumT(N * N, 0.0);
    EmisAccumType emisAccum(N);
    std::vector<std::vector<double>> emisWts(N);

    if constexpr (std::is_same_v<Obs, double>) {
        const std::size_t totalLen = std::accumulate(
            this->getObservationLists().begin(), this->getObservationLists().end(), std::size_t{0},
            [](std::size_t s, const auto &obs) { return s + obs.size(); });
        for (std::size_t i = 0; i < N; ++i) {
            emisAccum[i].reserve(totalLen);
            emisWts[i].reserve(totalLen);
        }
    }

    EStepBuffers bufs{emisAccum, emisWts, piNum, transDen, transNumT};
    std::size_t validSeqs = 0;
    lastLogProb_ = -std::numeric_limits<double>::infinity();
    double totalLogProb = 0.0;
    for (const auto &obs : this->getObservationLists()) {
        const double logP = accum_one_sequence(hmm, obs, N, logTransT, hasZeroTransitions, bufs);
        if (std::isfinite(logP)) {
            totalLogProb += logP;
            ++validSeqs;
        }
    }

    if (validSeqs == 0) {
        throw std::runtime_error("MapBaumWelchTrainer: no valid observation sequences "
                                 "(all had zero probability under the current model)");
    }
    lastLogProb_ = totalLogProb;

    m_step_pi_map(hmm, N, piNum, c);
    m_step_transitions_map(hmm, N, transNumT, transDen, c);

    // Emission M-step: EmisElem selects the correct fit() overload.
    // On the scalar path, apply Dirichlet smoothing to discrete distributions.
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t M = emisAccum[i].size();
        auto &dist = hmm.getDistribution(i);
        if (M == 0) {
            dist.reset();
            continue;
        }
        dist.fit(std::span<const EmisElem>(emisAccum[i].data(), M),
                 std::span<const double>(emisWts[i].data(), M));
        // apply_discrete_smoothing is a no-op when c==0 or dist is not discrete.
        if constexpr (std::is_same_v<Obs, double>)
            apply_discrete_smoothing(hmm, i, emisWts[i], c);
    }
}

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

template <typename Obs>
void BasicMapBaumWelchTrainer<Obs>::accumulate_xi(
    const double *logAlphaData, const double *logBetaData, const std::vector<double> &logEmitByTime,
    const std::vector<double> &logTransT, double logP, std::size_t T, std::size_t N,
    bool hasZeroTransitions, std::vector<double> &transNumT) noexcept {
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

template <typename Obs>
void BasicMapBaumWelchTrainer<Obs>::m_step_pi_map(HmmType &hmm, std::size_t N,
                                                  const std::vector<double> &piNum, double c) {
    const double piSum = std::accumulate(piNum.begin(), piNum.end(), 0.0);
    const double denom = piSum + static_cast<double>(N) * c;
    Vector pi(N);
    for (std::size_t i = 0; i < N; ++i)
        pi(i) = (denom > 0.0) ? (piNum[i] + c) / denom : 1.0 / static_cast<double>(N);
    hmm.setPi(pi);
}

template <typename Obs>
void BasicMapBaumWelchTrainer<Obs>::m_step_transitions_map(HmmType &hmm, std::size_t N,
                                                           const std::vector<double> &transNumT,
                                                           const std::vector<double> &transDen,
                                                           double c) {
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

// =============================================================================
// Explicit instantiation declarations.
// =============================================================================
extern template class BasicMapBaumWelchTrainer<double>;
extern template class BasicMapBaumWelchTrainer<ObservationVectorView>;

/// @brief Scalar alias (v3-compatible name).
using MapBaumWelchTrainer = BasicMapBaumWelchTrainer<double>;

} // namespace libhmm
