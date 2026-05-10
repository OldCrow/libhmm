#pragma once

#include "libhmm/training/trainer.h"
#include <vector>

namespace libhmm {

/**
 * @brief MAP-EM Baum-Welch trainer with symmetric Dirichlet priors.
 *
 * Extends standard Baum-Welch (see BaumWelchTrainer) by placing symmetric
 * Dirichlet priors on the transition matrix rows, the initial distribution,
 * and — for DiscreteDistribution emissions — the per-state emission vectors.
 * All other emission distributions are fitted by MLE.
 *
 * Parameterisation
 * ----------------
 * The prior is expressed as a pseudo-count c ≥ 0, related to the Dirichlet
 * concentration parameter α by c = α − 1.  In the MAP M-step:
 *
 *   A(i,j) = (ξ(i,j) + c) / (Σ_k ξ(i,k) + N·c)
 *   π_i    = (γ₀(i)  + c) / (Σ_k γ₀(k) + N·c)
 *
 * For DiscreteDistribution with K symbols per state, the same c is added to
 * each symbol's expected count before normalising:
 *
 *   B(i,k) = (Σ_t γ_t(i)·I[x_t=k] + c) / (Σ_t γ_t(i) + K·c)
 *
 * c = 0  → standard MLE, identical to BaumWelchTrainer.
 * c = 1  → Laplace smoothing (one virtual observation per outcome).
 *
 * Sparse-inducing priors (α < 1, i.e. c < 0) are not supported; the MAP
 * estimate is undefined when adjusted numerators go negative on short
 * sequences.  Users requiring sparsity should use Variational Bayes.
 *
 * Convergence
 * -----------
 * MAP-EM guarantees monotonic increase of the full MAP objective:
 *
 *   log P(O | λ) + log P(λ | c)
 *
 * The likelihood term log P(O | λ) alone is NOT guaranteed to be monotone.
 * On short sequences with non-zero c the likelihood can decrease while the
 * MAP objective increases.  Use computeLogPrior() to form the correct
 * convergence criterion:
 *
 *   ForwardBackwardCalculator(hmm, obs).getLogProbability()
 *     + trainer.computeLogPrior()
 *
 * Known limitations
 * -----------------
 * - Sparse-inducing priors (c < 0) unsupported; use VB for that regime.
 * - A single c is applied uniformly; per-row asymmetric priors unsupported.
 * - No conjugate priors on continuous distributions; they use MLE regardless.
 * - No state pruning during training.  To identify near-zero-occupancy states
 *   after training, compute posterior marginals via ForwardBackwardCalculator
 *   and remove states manually if desired.
 * - No hyperparameter optimisation; c is fixed at construction time.
 */
class MapBaumWelchTrainer : public Trainer {
public:
    /**
     * @brief Construct with an HMM reference, observation sequences, and pseudo-count.
     * @param hmm          HMM to train in place (non-owning reference).
     * @param obsLists     One or more observation sequences (must be non-empty).
     * @param pseudo_count Dirichlet pseudo-count c ≥ 0.  c = 0 → standard MLE.
     * @throws std::invalid_argument if obsLists is empty or pseudo_count < 0.
     */
    MapBaumWelchTrainer(Hmm &hmm, const ObservationLists &obsLists, double pseudo_count = 1.0);

    /** Legacy pointer constructor. @throws std::invalid_argument if hmm is null. */
    MapBaumWelchTrainer(Hmm *hmm, const ObservationLists &obsLists, double pseudo_count = 1.0);

    ~MapBaumWelchTrainer() override = default;

    MapBaumWelchTrainer(const MapBaumWelchTrainer &) = delete;
    MapBaumWelchTrainer &operator=(const MapBaumWelchTrainer &) = delete;
    MapBaumWelchTrainer(MapBaumWelchTrainer &&) = default;
    MapBaumWelchTrainer &operator=(MapBaumWelchTrainer &&) = default;

    /**
     * @brief Execute one MAP-EM pass (E-step + MAP M-step).
     * @throws std::runtime_error if no sequence has finite log-probability
     *         under the current model.
     */
    void train() override;

    /**
     * @brief Set the pseudo-count c for subsequent train() calls.
     * @throws std::invalid_argument if c < 0.
     */
    void setPseudoCount(double c);

    /** @return Current pseudo-count c. */
    [[nodiscard]] double getPseudoCount() const noexcept { return pseudo_count_; }

    /**
     * @brief Unnormalised log-prior of the current HMM parameters.
     *
     * Returns the parameter-dependent part of log P(λ | c):
     *
     *   c · (Σ_i Σ_j log A(i,j) + Σ_i log π_i)
     *     + c · Σ_{discrete states i} Σ_k log B(i,k)
     *
     * The Dirichlet normalising constants are omitted; they are invariant to
     * the HMM parameters and cancel when checking convergence.
     *
     * For the correct MAP convergence criterion combine this with
     * ForwardBackwardCalculator::getLogProbability():
     *
     *   mapObjective = logLikelihood + trainer.computeLogPrior()
     *
     * If c = 0 this returns 0.0.
     * Transitions or π entries equal to 0 contribute −∞ to the sum.
     */
    [[nodiscard]] double computeLogPrior() const;

private:
    double pseudo_count_;

    /// MAP M-step for π: add c to each numerator; normalise by piSum + N·c.
    static void m_step_pi_map(Hmm &hmm, std::size_t N, const std::vector<double> &piNum, double c);

    /// MAP M-step for A: add c to each xi count; normalise by transDen[i] + N·c.
    static void m_step_transitions_map(Hmm &hmm, std::size_t N,
                                       const std::vector<double> &transNumT,
                                       const std::vector<double> &transDen, double c);

    /// MAP M-step for emissions: MLE for continuous; Dirichlet-smoothed for discrete.
    static void apply_emission_fits_map(Hmm &hmm, std::size_t N,
                                        const std::vector<std::vector<double>> &emisData,
                                        const std::vector<std::vector<double>> &emisWts, double c);
};

} // namespace libhmm
