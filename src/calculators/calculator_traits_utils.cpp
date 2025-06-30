#include "libhmm/calculators/calculator_traits_utils.h"
#include <algorithm>
#include <cmath>

namespace libhmm {
namespace calculator_traits {

double calculateSIMDBenefit(std::size_t numStates, std::size_t seqLength) noexcept {
    // SIMD benefit increases with both dimensions but with diminishing returns
    const double stateFactor = 1.0 + std::log2(std::max(std::size_t(1), numStates / 4)) * 0.2;
    const double seqFactor = 1.0 + std::log2(std::max(std::size_t(1), seqLength / 50)) * 0.1;
    return stateFactor * seqFactor;
}

double calculateMemoryImpact(std::size_t overhead, double budget) noexcept {
    if (budget <= 0.0) {
        return 1.0; // No budget constraint
    }
    
    const double ratio = static_cast<double>(overhead) / budget;
    if (ratio >= 1.0) {
        return 0.1; // Heavy penalty for exceeding budget
    }
    
    return 1.0 - ratio * 0.5; // Linear impact scaling
}

double calculateStabilityNeed(std::size_t seqLength) noexcept {
    // Stability importance grows with sequence length
    return std::log2(std::max(std::size_t(1), seqLength / 100)) * 0.2;
}

} // namespace calculator_traits
} // namespace libhmm
