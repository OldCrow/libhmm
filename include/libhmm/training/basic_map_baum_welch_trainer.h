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
#include "libhmm/training/basic_trainer.h"
#include "libhmm/training/detail/bw_estep.h"

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

    using EmisElem = BwEmisElem<Obs>;

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
    }
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
    std::vector<EmisElem> emisObs;
    std::vector<std::vector<double>> emisWts(N);

    const std::size_t totalLen = std::accumulate(
        this->getObservationLists().begin(), this->getObservationLists().end(), std::size_t{0},
        [](std::size_t s, const auto &seq) { return s + ObsSeqTraits<Obs>::sequence_length(seq); });
    emisObs.reserve(totalLen);
    for (std::size_t i = 0; i < N; ++i)
        emisWts[i].reserve(totalLen);

    BwEStepBuffers<Obs> bufs{emisObs, emisWts, piNum, transDen, transNumT};
    std::size_t validSeqs = 0;
    lastLogProb_ = -std::numeric_limits<double>::infinity();
    double totalLogProb = 0.0;
    for (const auto &obs : this->getObservationLists()) {
        const double logP = bw_accum_one_sequence(hmm, obs, N, logTransT, hasZeroTransitions, bufs);
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

    // Emission M-step: emisObs is the shared observation buffer; emisWts[i]
    // holds the matching per-state gamma weights.
    const std::size_t M = emisObs.size();
    for (std::size_t i = 0; i < N; ++i) {
        auto &dist = hmm.getDistribution(i);
        if (M == 0) {
            dist.reset();
            continue;
        }
        dist.fit(std::span<const EmisElem>(emisObs.data(), M),
                 std::span<const double>(emisWts[i].data(), M));
        // apply_discrete_smoothing is a no-op when c==0 or dist is not discrete.
        if constexpr (std::is_same_v<Obs, double>)
            apply_discrete_smoothing(hmm, i, emisWts[i], c);
    }
}

// ---------------------------------------------------------------------------
// Static helpers (MAP-specific; E-step functions are in detail/bw_estep.h)
// ---------------------------------------------------------------------------

template <typename Obs>
void BasicMapBaumWelchTrainer<Obs>::m_step_pi_map(HmmType &hmm, std::size_t N,
                                                  const std::vector<double> &piNum, double c) {
    const double piSum = std::accumulate(piNum.begin(), piNum.end(), 0.0);
    const double denom = piSum + static_cast<double>(N) * c;
    Vector pi(N);
    for (std::size_t i = 0; i < N; ++i)
        pi(i) = (denom >= constants::precision::ZERO) ? (piNum[i] + c) / denom
                                                      : 1.0 / static_cast<double>(N);
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
            newTrans(i, j) = (denom >= constants::precision::ZERO)
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
