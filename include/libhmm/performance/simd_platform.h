#ifndef LIBHMM_SIMD_PLATFORM_H_
#define LIBHMM_SIMD_PLATFORM_H_

/**
 * @file simd_platform.h
 * @brief Platform-specific SIMD intrinsics and feature detection
 * 
 * This header centralizes all SIMD-related includes and platform detection logic.
 * It provides a clean interface for SIMD functionality across different architectures
 * while handling platform-specific quirks and compatibility issues.
 * 
 * DESIGN DECISION RATIONALE:
 * 
 * Previously, SIMD intrinsics were scattered across multiple files with inconsistent
 * platform detection logic. This led to several problems:
 * 
 * 1. DUPLICATE CODE: Multiple files had similar #ifdef blocks for SIMD detection
 * 2. INCONSISTENT LOGIC: Different files used different platform detection approaches
 * 3. MAINTENANCE BURDEN: Changes to platform support required updates in many files
 * 4. SEPARATION OF CONCERNS: common.h contained SIMD logic mixed with basic types
 * 
 * This header solves these issues by:
 * 
 * - CENTRALIZED DETECTION: All SIMD platform logic in one location
 * - CONSISTENT API: Unified feature macros (LIBHMM_HAS_*) across all files
 * - SEPARATION OF CONCERNS: SIMD functionality separate from general utilities
 * - SINGLE RESPONSIBILITY: This header only handles SIMD platform concerns
 * - EXTENSIBILITY: Easy to add new SIMD instruction sets or platforms
 * 
 * FILES THAT INCLUDE THIS HEADER:
 * - src/calculators/log_simd_viterbi_calculator.cpp
 * - src/calculators/scaled_simd_viterbi_calculator.cpp
 * - src/performance/log_space_ops.cpp
 * - include/libhmm/performance/simd_support.h
 * - include/libhmm/common/optimized_matrix.h (via simd_support.h)
 * - include/libhmm/common/optimized_vector.h (via simd_support.h)
 * 
 * Features:
 * - Cross-platform SIMD intrinsics inclusion
 * - Apple Silicon vs Intel chipset detection on macOS
 * - Feature detection macros with standardized naming
 * - Safe fallback for unsupported platforms
 * - Compile-time utility functions for SIMD capabilities
 * 
 * Usage:
 *   #include "libhmm/performance/simd_platform.h"
 *   
 *   #ifdef LIBHMM_HAS_AVX
 *       // Use AVX instructions
 *   #elif defined(LIBHMM_HAS_SSE2)
 *       // Use SSE2 instructions
 *   #elif defined(LIBHMM_HAS_NEON)
 *       // Use ARM NEON instructions
 *   #else
 *       // Use scalar fallback
 *   #endif
 * 
 * Or use the utility functions:
 *   if constexpr (libhmm::performance::simd::has_simd_support()) {
 *       // SIMD path
 *   } else {
 *       // Scalar path
 *   }
 */

//==============================================================================
// Platform Detection and SIMD Intrinsics
//==============================================================================

// Microsoft Visual C++ - Windows
#ifdef _MSC_VER
    #include <intrin.h>
    #define LIBHMM_HAS_SSE2
    #if defined(__AVX__)
        #define LIBHMM_HAS_AVX
    #endif
    #if defined(__AVX2__)
        #define LIBHMM_HAS_AVX2
    #endif

// GCC/Clang - x86/x64 platforms (Intel/AMD)
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
    #include <immintrin.h>
    #include <x86intrin.h>
    
    #define LIBHMM_HAS_SSE2  // Available on all modern x86_64
    
    #if defined(__SSE4_1__)
        #define LIBHMM_HAS_SSE4_1
    #endif
    
    #if defined(__AVX__)
        #define LIBHMM_HAS_AVX
    #endif
    
    #if defined(__AVX2__)
        #define LIBHMM_HAS_AVX2
    #endif
    
    #if defined(__AVX512F__)
        #define LIBHMM_HAS_AVX512
    #endif

// ARM platforms (Apple Silicon, ARM servers, embedded ARM)
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define LIBHMM_HAS_NEON
    
    // Apple Silicon specific optimizations
    #if defined(__APPLE__) && defined(__aarch64__)
        #define LIBHMM_APPLE_SILICON
    #endif

// Fallback - No SIMD support detected
#else
    #warning "No SIMD support detected - using scalar fallback implementations"
#endif

//==============================================================================
// Feature Detection Utilities
//==============================================================================

namespace libhmm {
namespace performance {
namespace simd {

/**
 * @brief Compile-time SIMD capability detection
 * @return true if any SIMD extensions are available at compile time
 */
constexpr bool has_simd_support() noexcept {
#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_SSE2) || defined(LIBHMM_HAS_NEON)
    return true;
#else
    return false;
#endif
}

/**
 * @brief Get the SIMD vector width for double precision operations
 * @return Number of double-precision values that fit in a SIMD register
 */
constexpr std::size_t double_vector_width() noexcept {
#if defined(LIBHMM_HAS_AVX512)
    return 8;  // AVX-512 can handle 8 doubles
#elif defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)
    return 4;  // AVX can handle 4 doubles
#elif defined(LIBHMM_HAS_SSE2)
    return 2;  // SSE2 can handle 2 doubles
#elif defined(LIBHMM_HAS_NEON)
    return 2;  // ARM NEON can handle 2 doubles (64-bit elements)
#else
    return 1;  // Scalar fallback
#endif
}

/**
 * @brief Get the SIMD vector width for single precision operations
 * @return Number of single-precision values that fit in a SIMD register
 */
constexpr std::size_t float_vector_width() noexcept {
#if defined(LIBHMM_HAS_AVX512)
    return 16; // AVX-512 can handle 16 floats
#elif defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)
    return 8;  // AVX can handle 8 floats
#elif defined(LIBHMM_HAS_SSE2)
    return 4;  // SSE2 can handle 4 floats
#elif defined(LIBHMM_HAS_NEON)
    return 4;  // ARM NEON can handle 4 floats (32-bit elements)
#else
    return 1;  // Scalar fallback
#endif
}

/**
 * @brief Get the optimal memory alignment for SIMD operations
 * @return Alignment in bytes for optimal SIMD performance
 */
constexpr std::size_t optimal_alignment() noexcept {
#if defined(LIBHMM_HAS_AVX512)
    return 64; // AVX-512 benefits from 64-byte alignment
#elif defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_AVX2)
    return 32; // AVX requires 32-byte alignment
#elif defined(LIBHMM_HAS_SSE2)
    return 16; // SSE2 requires 16-byte alignment
#elif defined(LIBHMM_HAS_NEON)
    return 16; // ARM NEON benefits from 16-byte alignment
#else
    return 8;  // Basic double alignment
#endif
}

/**
 * @brief Get a human-readable description of available SIMD features
 * @return String describing the detected SIMD capabilities
 */
inline const char* feature_string() noexcept {
#if defined(LIBHMM_HAS_AVX512)
    return "AVX-512";
#elif defined(LIBHMM_HAS_AVX2)
    return "AVX2";
#elif defined(LIBHMM_HAS_AVX)
    return "AVX";
#elif defined(LIBHMM_HAS_SSE4_1)
    return "SSE4.1";
#elif defined(LIBHMM_HAS_SSE2)
    return "SSE2";
#elif defined(LIBHMM_HAS_NEON)
    #if defined(LIBHMM_APPLE_SILICON)
        return "ARM NEON (Apple Silicon)";
    #else
        return "ARM NEON";
    #endif
#else
    return "Scalar (No SIMD)";
#endif
}

/**
 * @brief Check if the current build supports vectorized operations
 * @return true if SIMD is available and beneficial for performance
 */
constexpr bool supports_vectorization() noexcept {
    return has_simd_support() && double_vector_width() >= 2;
}

//==============================================================================
// Platform-Adaptive Constants
//==============================================================================

/**
 * @brief Optimal SIMD alignment based on detected platform capabilities
 * 
 * This constant adapts to the actual SIMD capabilities detected at compile time,
 * ensuring optimal memory alignment for the available instruction set.
 */
static constexpr std::size_t SIMD_ALIGNMENT = optimal_alignment();

/**
 * @brief SIMD vector width for double precision based on detected platform
 * 
 * This constant adapts to the actual SIMD capabilities detected at compile time,
 * providing the correct vector width for double precision operations.
 */
static constexpr std::size_t DOUBLE_SIMD_WIDTH = double_vector_width();

/**
 * @brief SIMD vector width for single precision based on detected platform
 * 
 * This constant adapts to the actual SIMD capabilities detected at compile time,
 * providing the correct vector width for single precision operations.
 */
static constexpr std::size_t FLOAT_SIMD_WIDTH = float_vector_width();

} // namespace simd
} // namespace performance
} // namespace libhmm


#endif // LIBHMM_SIMD_PLATFORM_H_
