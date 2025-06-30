#ifndef CALCULATOR_TRAITS_UTILS_H_
#define CALCULATOR_TRAITS_UTILS_H_

#include <cstddef>
#include "libhmm/common/common.h"
#include "libhmm/performance/simd_support.h"

namespace libhmm {
namespace calculator_traits {

/**
 * @brief Calculate SIMD benefit factor based on problem size
 * @param numStates Number of states
 * @param seqLength Sequence length
 * @return SIMD benefit multiplier
 */
double calculateSIMDBenefit(std::size_t numStates, std::size_t seqLength) noexcept;

/**
 * @brief Calculate memory overhead impact
 * @param overhead Memory overhead in bytes
 * @param budget Available memory budget
 * @return Memory impact factor (0.0 to 1.0)
 */
double calculateMemoryImpact(std::size_t overhead, double budget) noexcept;

/**
 * @brief Calculate numerical stability requirements
 * @param seqLength Sequence length
 * @return Stability requirement score
 */
double calculateStabilityNeed(std::size_t seqLength) noexcept;

} // namespace calculator_traits
} // namespace libhmm

#endif // CALCULATOR_TRAITS_UTILS_H_
