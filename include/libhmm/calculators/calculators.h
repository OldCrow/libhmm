#ifndef LIBHMM_CALCULATORS_H_
#define LIBHMM_CALCULATORS_H_

/**
 * @file calculators.h
 * @brief Convenience header that includes all libhmm HMM calculators
 * 
 * This header provides a single include point for all calculator implementations
 * available in libhmm. It follows the standard library convention of providing
 * umbrella headers for related functionality.
 * 
 * Usage:
 * @code
 * #include "libhmm/calculators/calculators.h"
 * 
 * // All calculators are now available:
 * ForwardBackwardCalculator calc(hmm, observations);
 * LogForwardBackwardCalculator logCalc(hmm, observations);
 * ViterbiCalculator viterbi(hmm, observations);
 * @endcode
 * 
 * @note For better compilation times, consider including only the specific
 *       calculator headers you need in performance-critical applications.
 */

// Base calculator interface
#include "libhmm/calculators/calculator.h"

// Forward-Backward calculators
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/scaled_forward_backward_calculator.h"
#include "libhmm/calculators/log_forward_backward_calculator.h"
#include "libhmm/calculators/optimized_forward_backward_calculator.h"
#include "libhmm/calculators/log_simd_forward_backward_calculator.h"
#include "libhmm/calculators/scaled_simd_forward_backward_calculator.h"

// Viterbi calculator
#include "libhmm/calculators/viterbi_calculator.h"

// Calculator selection and optimization
#include "libhmm/calculators/calculator_traits.h"

/**
 * @namespace libhmm
 * @brief All calculators are available in the libhmm namespace
 * 
 * After including this header, all calculator classes are available:
 * 
 * **Forward-Backward Calculators:**
 * - ForwardBackwardCalculator: Standard forward-backward algorithm
 * - ScaledForwardBackwardCalculator: Rabiner's scaling for numerical stability
 * - LogForwardBackwardCalculator: Log-space computation for very long sequences
 * - OptimizedForwardBackwardCalculator: SIMD-optimized (numerical issues)
 * - LogSIMDForwardBackwardCalculator: Log-space + SIMD optimizations
 * - ScaledSIMDForwardBackwardCalculator: Scaled computation + SIMD optimizations
 * 
 * **Path Calculators:**
 * - ViterbiCalculator: Most likely state sequence computation
 * 
 * **Calculator Selection:**
 * - calculators::CalculatorSelector: Automatic optimal calculator selection
 * - calculators::CalculatorTraits: Performance characteristics and selection
 * - calculators::AutoCalculator: RAII wrapper with automatic selection
 */

// Calculator count for compile-time verification
namespace libhmm {
    namespace detail {
        /// Total number of concrete calculator types (excluding base class)
        inline constexpr std::size_t CALCULATOR_COUNT = 7;
        
        /// Number of forward-backward calculator variants
        inline constexpr std::size_t FORWARD_BACKWARD_CALCULATOR_COUNT = 6;
        
        /// Number of path calculator types
        inline constexpr std::size_t PATH_CALCULATOR_COUNT = 1;
        
        static_assert(FORWARD_BACKWARD_CALCULATOR_COUNT + PATH_CALCULATOR_COUNT == CALCULATOR_COUNT,
                     "Calculator counts must be consistent");
    }
}

#endif // LIBHMM_CALCULATORS_H_
