#pragma once

#include "libhmm/basic_hmm.h"
#include "libhmm/hmm.h"
#include <cstddef>
#include <limits>

namespace libhmm {

/**
 * @brief AIC, BIC, and AICc for a fitted HMM.
 *
 * All criteria follow the "lower is better" convention:
 *   AIC  = 2k − 2 logL
 *   BIC  = k ln(n) − 2 logL
 *   AICc = AIC + 2k(k+1)/(n−k−1)   (+∞ when n ≤ k+1)
 *
 * Use evaluate_model() to fill this struct from an HMM and a log-likelihood.
 */
struct HmmModelCriteria {
    double aic{0.0};
    double bic{0.0};
    double aicc{0.0};
};

/**
 * @brief Count the free parameters of a fitted HMM.
 *
 * Free parameters = N*(N-1)  (transition matrix: N simplex rows, each N-1 free)
 *                 + (N-1)    (initial distribution simplex)
 *                 + Σ dist_i.getNumParameters()  (one emission per state)
 *
 * The template overload works for both scalar (Hmm) and multivariate (HmmMV)
 * HMMs; the formula is identical since transitions and pi have the same shape.
 *
 * @param hmm  The HMM to inspect.
 * @return Total free parameter count.
 */
template <typename Obs>
[[nodiscard]] std::size_t count_free_parameters(const BasicHmm<Obs> &hmm) {
    const std::size_t N = hmm.getNumStatesModern();
    std::size_t k = N * (N - 1) + (N - 1); // transition rows + pi simplex
    for (std::size_t i = 0; i < N; ++i)
        k += hmm.getDistribution(i).getNumParameters();
    return k;
}

/// @brief Scalar specialisation — delegates to the template.
[[nodiscard]] inline std::size_t count_free_parameters(const Hmm &hmm) {
    return count_free_parameters<double>(hmm);
}

/**
 * @brief AIC = 2k − 2 logL.
 * @param logL  Log-likelihood of the fitted model.
 * @param k     Number of free parameters.
 */
[[nodiscard]] double compute_aic(double logL, std::size_t k) noexcept;

/**
 * @brief BIC = k ln(n) − 2 logL.
 * @param logL  Log-likelihood of the fitted model.
 * @param k     Number of free parameters.
 * @param n     Number of observations (sequence length).
 */
[[nodiscard]] double compute_bic(double logL, std::size_t k, std::size_t n) noexcept;

/**
 * @brief AICc = AIC + 2k(k+1)/(n−k−1) (corrected AIC for small samples).
 *
 * Returns positive infinity when n ≤ k+1 (correction term is undefined).
 *
 * @param logL  Log-likelihood of the fitted model.
 * @param k     Number of free parameters.
 * @param n     Number of observations (sequence length).
 */
[[nodiscard]] double compute_aicc(double logL, std::size_t k, std::size_t n) noexcept;

/**
 * @brief Compute AIC, BIC, and AICc for a fitted HMM.
 *
 * Convenience wrapper: derives k from count_free_parameters(hmm), then
 * calls compute_aic(), compute_bic(), and compute_aicc().
 *
 * @param hmm             The fitted HMM.
 * @param logLikelihood   log P(O | λ) from the ForwardBackwardCalculator.
 * @param sequenceLength  Number of observations T.
 * @return HmmModelCriteria with all three criteria populated.
 */
[[nodiscard]] HmmModelCriteria evaluate_model(const Hmm &hmm, double logLikelihood,
                                              std::size_t sequenceLength);

} // namespace libhmm
