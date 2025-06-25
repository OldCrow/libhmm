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
#include "libhmm/calculators/log_simd_forward_backward_calculator.h"
#include "libhmm/calculators/scaled_simd_forward_backward_calculator.h"

// Viterbi calculators
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/calculators/scaled_simd_viterbi_calculator.h"
#include "libhmm/calculators/log_simd_viterbi_calculator.h"

// Calculator selection and optimization
#include "libhmm/calculators/forward_backward_traits.h"
#include "libhmm/calculators/viterbi_traits.h"

/**
 * @namespace libhmm
 * @brief All calculators are available in the libhmm namespace
 * 
 * After including this header, all calculator classes are available:
 * 
 * **Forward-Backward Calculators:**
 * - ForwardBackwardCalculator: Standard forward-backward algorithm
 * - LogSIMDForwardBackwardCalculator: Log-space + SIMD optimizations with fallback
 * - ScaledSIMDForwardBackwardCalculator: Scaled computation + SIMD optimizations with fallback
 * 
 * **Viterbi Calculators:**
 * - ViterbiCalculator: Standard Viterbi algorithm for most likely state sequence
 * - ScaledSIMDViterbiCalculator: Scaled Viterbi with SIMD optimization and fallback
 * - LogSIMDViterbiCalculator: Log-space Viterbi with SIMD optimization and fallback
 * 
 * **Calculator Selection:**
 * - calculators::CalculatorSelector: Automatic optimal Forward-Backward calculator selection
 * - calculators::CalculatorTraits: Forward-Backward performance characteristics and selection
 * - calculators::AutoCalculator: RAII wrapper with automatic Forward-Backward selection
 * - viterbi::CalculatorSelector: Automatic optimal Viterbi calculator selection
 * - viterbi::CalculatorTraits: Viterbi performance characteristics and selection
 * - viterbi::AutoCalculator: RAII wrapper with automatic Viterbi selection
 */

// Calculator count for compile-time verification
namespace libhmm {
    namespace detail {
        /// Total number of concrete calculator types (excluding base class)
        inline constexpr std::size_t CALCULATOR_COUNT = 6;
        
        /// Number of forward-backward calculator variants
        inline constexpr std::size_t FORWARD_BACKWARD_CALCULATOR_COUNT = 3;
        
        /// Number of Viterbi calculator variants
        inline constexpr std::size_t VITERBI_CALCULATOR_COUNT = 3;
        
        static_assert(FORWARD_BACKWARD_CALCULATOR_COUNT + VITERBI_CALCULATOR_COUNT == CALCULATOR_COUNT,
                     "Calculator counts must be consistent");
    }
}

#endif // LIBHMM_CALCULATORS_H_
