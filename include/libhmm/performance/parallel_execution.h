#ifndef LIBHMM_PERFORMANCE_PARALLEL_EXECUTION_H_
#define LIBHMM_PERFORMANCE_PARALLEL_EXECUTION_H_

/**
 * @file parallel_execution.h  
 * @brief C++17 parallel execution policy detection and utilities
 * 
 * This header centralizes all C++17 parallel execution support detection
 * and provides utilities for using std::execution policies safely across
 * different compilers and standard library implementations.
 * 
 * Key features:
 * - Automatic detection of C++17 parallel execution support
 * - Safe fallback macros for unsupported platforms
 * - Compile-time utilities for parallel execution decisions
 * - Cross-platform compatibility for execution policies
 * 
 * @section usage Usage Examples
 * @code
 * #include "libhmm/performance/parallel_execution.h"
 * 
 * // Use the detection macro
 * #if LIBHMM_HAS_PARALLEL_EXECUTION
 *     std::sort(std::execution::par_unseq, vec.begin(), vec.end());
 * #else
 *     std::sort(vec.begin(), vec.end());
 * #endif
 * 
 * // Or use the utility functions
 * if constexpr (libhmm::performance::parallel::has_execution_policies()) {
 *     // Use parallel algorithms
 * } else {
 *     // Use serial algorithms
 * }
 * @endcode
 * 
 * @version 2.8.0
 * @note This header is automatically included by parallel_constants.h
 */

#include <cstddef>

//==============================================================================
// C++17 PARALLEL EXECUTION POLICY DETECTION
//==============================================================================

/// Check for C++17 parallel execution support
/// This macro is used throughout libhmm for conditional compilation
#ifdef __cpp_lib_execution
    #include <execution>
    #define LIBHMM_HAS_PARALLEL_EXECUTION 1
#else
    #define LIBHMM_HAS_PARALLEL_EXECUTION 0
#endif

//==============================================================================
// PARALLEL EXECUTION UTILITIES  
//==============================================================================

namespace libhmm {
namespace performance {
namespace parallel {

/**
 * @brief Compile-time check for parallel execution policy support
 * @return true if std::execution policies are available
 */
constexpr bool has_execution_policies() noexcept {
#if LIBHMM_HAS_PARALLEL_EXECUTION
    return true;
#else
    return false;
#endif
}

/**
 * @brief Get a human-readable description of parallel execution support
 * @return String describing parallel execution capabilities
 */
inline const char* execution_support_string() noexcept {
#if LIBHMM_HAS_PARALLEL_EXECUTION
    return "C++17 std::execution policies available";
#else
    return "C++17 std::execution policies not available (fallback to serial)";
#endif
}

/**
 * @brief Check if a problem size is large enough to benefit from parallel execution
 * @param problem_size Total number of elements or operations
 * @param threshold Minimum size to consider parallel execution (default: 1000)
 * @return true if parallel execution is likely beneficial
 */
constexpr bool should_use_parallel(std::size_t problem_size, std::size_t threshold = 1000) noexcept {
    return has_execution_policies() && (problem_size >= threshold);
}

//==============================================================================
// SAFE PARALLEL EXECUTION MACROS
//==============================================================================

/**
 * @brief Safely execute parallel algorithms with automatic fallback
 * 
 * These macros provide a clean way to use parallel algorithms with automatic
 * fallback to serial algorithms when parallel execution is not available.
 * They eliminate the need for #ifdef blocks throughout the codebase.
 */

/// Execute with parallel unseq policy if available, otherwise serial
#if LIBHMM_HAS_PARALLEL_EXECUTION
    #define LIBHMM_PAR_UNSEQ std::execution::par_unseq,
    #define LIBHMM_PAR std::execution::par,
    #define LIBHMM_SEQ std::execution::seq,
#else
    #define LIBHMM_PAR_UNSEQ
    #define LIBHMM_PAR  
    #define LIBHMM_SEQ
#endif

//==============================================================================
// PARALLEL ALGORITHM WRAPPERS
//==============================================================================

/**
 * @brief Safe wrappers for common parallel algorithms
 * 
 * These provide a consistent API that automatically uses parallel execution
 * when available and falls back to serial execution otherwise.
 */

/// Safe parallel fill operation
template<typename Iterator, typename T>
void safe_fill(Iterator first, Iterator last, const T& value) {
#if LIBHMM_HAS_PARALLEL_EXECUTION
    if (std::distance(first, last) > 1000) {
        std::fill(std::execution::par_unseq, first, last, value);
    } else {
        std::fill(first, last, value);
    }
#else
    std::fill(first, last, value);
#endif
}

/// Safe parallel transform operation  
template<typename Iterator1, typename Iterator2, typename UnaryOp>
void safe_transform(Iterator1 first1, Iterator1 last1, Iterator2 first2, UnaryOp op) {
#if LIBHMM_HAS_PARALLEL_EXECUTION
    if (std::distance(first1, last1) > 1000) {
        std::transform(std::execution::par_unseq, first1, last1, first2, op);
    } else {
        std::transform(first1, last1, first2, op);
    }
#else
    std::transform(first1, last1, first2, op);
#endif
}

/// Safe parallel reduce operation (returns sum-like result)
template<typename Iterator, typename T>
T safe_reduce(Iterator first, Iterator last, T init) {
#if LIBHMM_HAS_PARALLEL_EXECUTION
    if (std::distance(first, last) > 1000) {
        return std::reduce(std::execution::par_unseq, first, last, init);
    } else {
        return std::accumulate(first, last, init);
    }
#else
    return std::accumulate(first, last, init);
#endif
}

/// Safe parallel for_each operation
template<typename Iterator, typename UnaryFunction>
void safe_for_each(Iterator first, Iterator last, UnaryFunction f) {
#if LIBHMM_HAS_PARALLEL_EXECUTION
    if (std::distance(first, last) > 1000) {
        std::for_each(std::execution::par_unseq, first, last, f);
    } else {
        std::for_each(first, last, f);
    }
#else
    std::for_each(first, last, f);
#endif
}

} // namespace parallel
} // namespace performance  
} // namespace libhmm

//==============================================================================
// COMPILER COMPATIBILITY NOTES
//==============================================================================

/**
 * @section compatibility Compiler Compatibility
 * 
 * **Supported Compilers with Parallel Execution:**
 * - GCC 9.1+ with Intel TBB or libstdc++ parallel mode
 * - Clang 9.0+ with Intel TBB  
 * - MSVC 2019+ (Visual Studio 16.0+)
 * - Intel C++ Compiler 19.0+
 * 
 * **Automatic Fallback for:**
 * - Older compiler versions
 * - Systems without TBB
 * - Embedded or resource-constrained platforms
 * 
 * **Detection Method:**
 * Uses the standard `__cpp_lib_execution` feature test macro as defined
 * in the C++17 standard (ISO/IEC 14882:2017).
 */

#endif // LIBHMM_PERFORMANCE_PARALLEL_EXECUTION_H_
