#ifndef LIBHMM_SIMD_SUPPORT_H_
#define LIBHMM_SIMD_SUPPORT_H_

#include <cstddef>
#include <memory>
#include <type_traits>

// Import all SIMD platform detection and intrinsics
#include "libhmm/performance/simd_platform.h"

namespace libhmm {
namespace performance {

// Import platform-adaptive SIMD constants from simd_platform.h
using simd::SIMD_ALIGNMENT;
using simd::DOUBLE_SIMD_WIDTH;
using simd::FLOAT_SIMD_WIDTH;

/// Check if SIMD is available at compile time
constexpr bool simd_available() noexcept {
#if defined(LIBHMM_HAS_AVX) || defined(LIBHMM_HAS_SSE2) || defined(LIBHMM_HAS_NEON)
    return true;
#else
    return false;
#endif
}

/// Aligned memory allocator for SIMD operations
template<typename T>
class aligned_allocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<typename U>
    struct rebind {
        using other = aligned_allocator<U>;
    };

    aligned_allocator() = default;
    
    template<typename U>
    aligned_allocator(const aligned_allocator<U>&) noexcept {}

    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        
        size_type bytes = n * sizeof(T);
        void* ptr = nullptr;
        
#if defined(_WIN32)
        ptr = _aligned_malloc(bytes, SIMD_ALIGNMENT);
        if (!ptr) throw std::bad_alloc();
#elif defined(__APPLE__) || defined(__linux__)
        if (posix_memalign(&ptr, SIMD_ALIGNMENT, bytes) != 0) {
            throw std::bad_alloc();
        }
#else
        ptr = std::aligned_alloc(SIMD_ALIGNMENT, bytes);
        if (!ptr) throw std::bad_alloc();
#endif
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        if (p) {
#if defined(_WIN32)
            _aligned_free(p);
#else
            std::free(p);
#endif
        }
    }

    bool operator==(const aligned_allocator&) const noexcept { return true; }
    bool operator!=(const aligned_allocator&) const noexcept { return false; }
};

/// SIMD-optimized vector operations
class SIMDOps {
public:
    /// Vectorized dot product
    /// @param a First vector
    /// @param b Second vector
    /// @param size Vector size
    /// @return Dot product result
    static double dot_product(const double* a, const double* b, std::size_t size) noexcept;
    
    /// Vectorized vector addition
    /// @param a First vector
    /// @param b Second vector
    /// @param result Output vector (a + b)
    /// @param size Vector size
    static void vector_add(const double* a, const double* b, double* result, std::size_t size) noexcept;
    
    /// Vectorized vector multiplication
    /// @param a First vector
    /// @param b Second vector
    /// @param result Output vector (a * b element-wise)
    /// @param size Vector size
    static void vector_multiply(const double* a, const double* b, double* result, std::size_t size) noexcept;
    
    /// Vectorized scalar multiplication
    /// @param a Input vector
    /// @param scalar Scalar value
    /// @param result Output vector (a * scalar)
    /// @param size Vector size
    static void scalar_multiply(const double* a, double scalar, double* result, std::size_t size) noexcept;
    
    /// Vectorized matrix-vector multiplication (optimized)
    /// @param matrix Matrix data (row-major)
    /// @param vector Vector data
    /// @param result Output vector
    /// @param rows Number of matrix rows
    /// @param cols Number of matrix columns
    static void matrix_vector_multiply(const double* matrix, const double* vector, 
                                       double* result, std::size_t rows, std::size_t cols) noexcept;

private:
    /// Fallback implementations for non-SIMD systems
    static double dot_product_fallback(const double* a, const double* b, std::size_t size) noexcept;
    static void vector_add_fallback(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_multiply_fallback(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void scalar_multiply_fallback(const double* a, double scalar, double* result, std::size_t size) noexcept;
    static void matrix_vector_multiply_fallback(const double* matrix, const double* vector, 
                                                double* result, std::size_t rows, std::size_t cols) noexcept;

#ifdef LIBHMM_HAS_AVX
    /// AVX implementations
    static double dot_product_avx(const double* a, const double* b, std::size_t size) noexcept;
    static void vector_add_avx(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_multiply_avx(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void scalar_multiply_avx(const double* a, double scalar, double* result, std::size_t size) noexcept;
#endif

#ifdef LIBHMM_HAS_SSE2
    /// SSE2 implementations
    static double dot_product_sse2(const double* a, const double* b, std::size_t size) noexcept;
    static void vector_add_sse2(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_multiply_sse2(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void scalar_multiply_sse2(const double* a, double scalar, double* result, std::size_t size) noexcept;
#endif

#ifdef LIBHMM_HAS_NEON
    /// ARM NEON implementations
    static double dot_product_neon(const double* a, const double* b, std::size_t size) noexcept;
    static void vector_add_neon(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_multiply_neon(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void scalar_multiply_neon(const double* a, double scalar, double* result, std::size_t size) noexcept;
#endif
};

/// Memory prefetching hints for better cache performance
inline void prefetch_read(const void* addr) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 0, 3); // Read, high temporal locality
#elif defined(_MSC_VER)
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
#endif
}

inline void prefetch_write(const void* addr) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 1, 3); // Write, high temporal locality
#elif defined(_MSC_VER)
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
#endif
}

/// Cache line alignment utility
template<typename T>
constexpr std::size_t cache_aligned_size(std::size_t size) noexcept {
    constexpr std::size_t cache_line_size = 64; // Common cache line size
    const std::size_t total_bytes = size * sizeof(T);
    return ((total_bytes + cache_line_size - 1) / cache_line_size) * cache_line_size / sizeof(T);
}

} // namespace performance
} // namespace libhmm

#endif // LIBHMM_SIMD_SUPPORT_H_
