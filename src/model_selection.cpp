#include "libhmm/training/model_selection.h"
#include <cmath>
#include <limits>

namespace libhmm {

std::size_t count_free_parameters(const Hmm &hmm) {
    const std::size_t N = hmm.getNumStatesModern();
    // Transition matrix: N rows, each a simplex with N-1 free parameters.
    std::size_t k = N * (N - 1u);
    // Initial distribution: simplex with N-1 free parameters.
    if (N > 0u) {
        k += N - 1u;
    }
    // Emission distributions: each state contributes its own parameter count.
    for (std::size_t i = 0; i < N; ++i) {
        k += hmm.getDistribution(i).getNumParameters();
    }
    return k;
}

double compute_aic(double logL, std::size_t k) noexcept {
    return 2.0 * static_cast<double>(k) - 2.0 * logL;
}

double compute_bic(double logL, std::size_t k, std::size_t n) noexcept {
    return static_cast<double>(k) * std::log(static_cast<double>(n)) - 2.0 * logL;
}

double compute_aicc(double logL, std::size_t k, std::size_t n) noexcept {
    const double kd = static_cast<double>(k);
    const double nd = static_cast<double>(n);
    // Correction term 2k(k+1)/(n-k-1) is undefined when n <= k+1.
    if (nd <= kd + 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    const double aic = compute_aic(logL, k);
    return aic + 2.0 * kd * (kd + 1.0) / (nd - kd - 1.0);
}

HmmModelCriteria evaluate_model(const Hmm &hmm, double logLikelihood, std::size_t sequenceLength) {
    const std::size_t k = count_free_parameters(hmm);
    return HmmModelCriteria{compute_aic(logLikelihood, k),
                            compute_bic(logLikelihood, k, sequenceLength),
                            compute_aicc(logLikelihood, k, sequenceLength)};
}

} // namespace libhmm
