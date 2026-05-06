#include "libhmm/math/numerical_stability.h"
#include <algorithm>
#include <numeric>

namespace libhmm {
namespace numerical {

void NumericalSafety::checkFinite(double value, const std::string &name) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("Non-finite value detected in " + name + ": " +
                                 std::to_string(value));
    }
}

double NumericalSafety::clampProbability(double prob) noexcept {
    if (!std::isfinite(prob)) {
        return NumericalConstants::MIN_PROBABILITY;
    }
    return std::clamp(prob, NumericalConstants::MIN_PROBABILITY,
                      NumericalConstants::MAX_PROBABILITY);
}

bool NumericalSafety::normalizeProbabilities(Vector &probs) noexcept {
    // Clamp each probability to a valid range.
    for (std::size_t i = 0; i < probs.size(); ++i) {
        probs(i) = clampProbability(probs(i));
    }

    const double sum = std::accumulate(probs.begin(), probs.end(), 0.0);

    if (sum < 1e-15) {
        // All probabilities effectively zero — fall back to uniform.
        const double uniform = 1.0 / static_cast<double>(probs.size());
        for (std::size_t i = 0; i < probs.size(); ++i) {
            probs(i) = uniform;
        }
        return false;
    }

    for (std::size_t i = 0; i < probs.size(); ++i) {
        probs(i) /= sum;
    }
    return true;
}

} // namespace numerical
} // namespace libhmm
