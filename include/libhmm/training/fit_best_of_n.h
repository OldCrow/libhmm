#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "libhmm/basic_hmm.h"
#include "libhmm/calculators/basic_forward_backward_calculator.h"
#include "libhmm/hmm.h"
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/math/constants.h"
#include "libhmm/training/basic_baum_welch_trainer.h"
#include "libhmm/training/kmeans_init.h"

namespace libhmm {

namespace detail {

/**
 * @brief Sum log P(O_k | λ) over all sequences via forward-backward.
 *
 * Zero-length sequences are skipped.  A sequence with zero probability under
 * the model contributes -inf, which propagates to the total — a model that
 * cannot explain part of the data loses the best-of-n comparison naturally.
 */
template <typename Obs>
[[nodiscard]] double total_log_likelihood(const BasicHmm<Obs> &hmm,
                                          const typename ObsSeqTraits<Obs>::ListType &obsLists) {
    double total = 0.0;
    for (const auto &seq : obsLists) {
        if (ObsSeqTraits<Obs>::sequence_length(seq) == 0)
            continue;
        BasicForwardBackwardCalculator<Obs> fbc(hmm, seq);
        total += fbc.getLogProbability();
    }
    return total;
}

/**
 * @brief Re-initialise a scalar HMM's emissions from small random subsamples.
 *
 * For each state, draws m observations with replacement from the pool of all
 * observation values and refits the state's emission via the existing
 * unweighted fit().  The subsample is kept deliberately small
 * (m = clamp(pool/(4N), 2, 32)): small samples have high variance, and that
 * variance is what produces genuinely diverse EM starting points.  Refitting
 * to large subsamples would converge every state to near-identical
 * pooled-data fits and defeat the restart.
 *
 * This is the settled interpretation of the issue's "randomise emission
 * parameters from prior": no prior machinery exists, and this construction
 * needs only the fit() surface every distribution family already implements.
 *
 * π and the transition matrix are left at their cloned values; the first
 * EM iteration re-estimates both once the emissions differ.
 */
inline void randomise_emissions_scalar(Hmm &hmm, const std::vector<double> &pool,
                                       std::mt19937_64 &rng) {
    const std::size_t N = hmm.getNumStatesModern();
    if (pool.size() < 2 || N == 0)
        return; // nothing to subsample; restart proceeds from cloned params
    const std::size_t m = std::clamp<std::size_t>(pool.size() / (4 * N), 2, 32);
    std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
    std::vector<double> sub(m);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < m; ++j)
            sub[j] = pool[pick(rng)];
        hmm.getDistribution(i).fit(std::span<const double>(sub.data(), m));
    }
}

} // namespace detail

/**
 * @brief Multi-restart Baum-Welch: train from n independent starts, keep the best.
 *
 * EM converges to a local optimum that depends strongly on initialisation.
 * fit_best_of_n() runs @p n_restarts independent Baum-Welch trainings and
 * copies the model with the highest total log-likelihood back into @p hmm.
 *
 * Restart 0 trains from the caller's current parameters, unrandomised.
 * Restarts 1..n-1 clone the HMM and re-initialise its emissions:
 *   - Scalar path (Hmm): each state's emission is refit to a small random
 *     subsample of the pooled observations (see detail::randomise_emissions_scalar).
 *   - Multivariate path (HmmMV): kmeans_init() with fresh k-means++ seeding.
 *
 * Because restart 0 is the unrandomised single run, the returned model is
 * BY CONSTRUCTION at least as good as a single training run with the same
 * iteration policy — not merely with high probability.
 *
 * Each restart runs Baum-Welch until the E-step log-likelihood changes by
 * less than a relative tolerance (constants::precision::BW_TOLERANCE) or
 * @p max_iters iterations, then scores the result by summing forward-backward
 * log-probabilities over all sequences.  A restart that throws (e.g. a
 * degenerate refit or all-zero-probability sequences) is discarded; if every
 * restart fails, the last exception is rethrown.
 *
 * Restarts are independent and serial here; parallel execution is issue #47.
 *
 * @param hmm         HMM to train.  Mutated in place with the best result.
 * @param obsLists    Training sequences (must outlive the call; not copied).
 * @param n_restarts  Total number of starts, including the unrandomised
 *                    restart 0.  Must be >= 1.
 * @param rng         Seeded RNG; consumed sequentially across restarts, so a
 *                    fixed seed gives a reproducible restart set per platform.
 * @param max_iters   Baum-Welch iteration cap per restart.
 * @returns           Total log-likelihood of the best model found.
 *
 * @throws std::invalid_argument if obsLists is empty or n_restarts == 0.
 * @throws std::runtime_error if every restart failed to train.
 */
template <typename Obs>
[[nodiscard]] double
fit_best_of_n(BasicHmm<Obs> &hmm, const typename ObsSeqTraits<Obs>::ListType &obsLists,
              std::size_t n_restarts, std::mt19937_64 &rng, std::size_t max_iters = 500) {
    if (obsLists.empty())
        throw std::invalid_argument("fit_best_of_n: observation lists cannot be empty");
    if (n_restarts == 0)
        throw std::invalid_argument("fit_best_of_n: n_restarts must be >= 1");

    // Scalar path: pool all observation values once, shared across restarts.
    std::vector<double> pool;
    if constexpr (std::is_same_v<Obs, double>) {
        std::size_t total = 0;
        for (const auto &seq : obsLists)
            total += seq.size();
        pool.reserve(total);
        for (const auto &seq : obsLists)
            for (std::size_t t = 0; t < seq.size(); ++t)
                pool.push_back(seq(t));
    }

    double bestLogL = -std::numeric_limits<double>::infinity();
    BasicHmm<Obs> best(hmm.getNumStatesModern());
    bool haveBest = false;
    std::exception_ptr lastError;

    for (std::size_t r = 0; r < n_restarts; ++r) {
        BasicHmm<Obs> candidate = hmm.clone();
        try {
            if (r > 0) {
                if constexpr (std::is_same_v<Obs, ObservationVectorView>) {
                    kmeans_init(candidate, obsLists, rng);
                } else {
                    detail::randomise_emissions_scalar(candidate, pool, rng);
                }
            }

            BasicBaumWelchTrainer<Obs> trainer(candidate, obsLists);
            double prev = -std::numeric_limits<double>::infinity();
            for (std::size_t it = 0; it < max_iters; ++it) {
                trainer.train();
                const double cur = trainer.getLastLogProbability();
                if (std::isfinite(prev) &&
                    std::abs(cur - prev) <=
                        constants::precision::BW_TOLERANCE * (std::abs(cur) + 1.0))
                    break;
                prev = cur;
            }

            const double logL = detail::total_log_likelihood(candidate, obsLists);
            if (std::isnan(logL)) {
                // A restart whose final M-step produced a non-finite parameter scores
                // NaN here. Without this guard restart 0 would install it via
                // `!haveBest` and no later restart could displace it (x > NaN is
                // false). Treat it like a throwing restart: record and move on.
                lastError = std::make_exception_ptr(
                    std::runtime_error("fit_best_of_n: restart produced a NaN log-likelihood "
                                       "(non-finite emission parameters after the M-step?)"));
                continue;
            }
            if (!haveBest || logL > bestLogL) {
                bestLogL = logL;
                best = std::move(candidate);
                haveBest = true;
            }
        } catch (...) {
            lastError = std::current_exception();
        }
    }

    if (!haveBest) {
        if (lastError)
            std::rethrow_exception(lastError);
        throw std::runtime_error("fit_best_of_n: no restart produced a trained model");
    }

    hmm = std::move(best);
    return bestLogL;
}

} // namespace libhmm
