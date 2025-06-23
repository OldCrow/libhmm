#include "libhmm/performance/simd_support.h"
#include <cstring>
#include <cstdlib>

namespace libhmm {
namespace performance {

//========== Public Interface Implementations ==========

double SIMDOps::dot_product(const double* a, const double* b, std::size_t size) noexcept {
#ifdef LIBHMM_HAS_AVX
    if (simd_available() && size >= DOUBLE_SIMD_WIDTH) {
        return dot_product_avx(a, b, size);
    }
#endif
#ifdef LIBHMM_HAS_SSE2
    if (simd_available() && size >= 2) {
        return dot_product_sse2(a, b, size);
    }
#endif
#ifdef LIBHMM_HAS_NEON
    if (simd_available() && size >= 2) {
        return dot_product_neon(a, b, size);
    }
#endif
    return dot_product_fallback(a, b, size);
}

void SIMDOps::vector_add(const double* a, const double* b, double* result, std::size_t size) noexcept {
#ifdef LIBHMM_HAS_AVX
    if (simd_available() && size >= DOUBLE_SIMD_WIDTH) {
        vector_add_avx(a, b, result, size);
        return;
    }
#endif
#ifdef LIBHMM_HAS_SSE2
    if (simd_available() && size >= 2) {
        vector_add_sse2(a, b, result, size);
        return;
    }
#endif
#ifdef LIBHMM_HAS_NEON
    if (simd_available() && size >= 2) {
        vector_add_neon(a, b, result, size);
        return;
    }
#endif
    vector_add_fallback(a, b, result, size);
}

void SIMDOps::vector_multiply(const double* a, const double* b, double* result, std::size_t size) noexcept {
#ifdef LIBHMM_HAS_AVX
    if (simd_available() && size >= DOUBLE_SIMD_WIDTH) {
        vector_multiply_avx(a, b, result, size);
        return;
    }
#endif
#ifdef LIBHMM_HAS_SSE2
    if (simd_available() && size >= 2) {
        vector_multiply_sse2(a, b, result, size);
        return;
    }
#endif
#ifdef LIBHMM_HAS_NEON
    if (simd_available() && size >= 2) {
        vector_multiply_neon(a, b, result, size);
        return;
    }
#endif
    vector_multiply_fallback(a, b, result, size);
}

void SIMDOps::scalar_multiply(const double* a, double scalar, double* result, std::size_t size) noexcept {
#ifdef LIBHMM_HAS_AVX
    if (simd_available() && size >= DOUBLE_SIMD_WIDTH) {
        scalar_multiply_avx(a, scalar, result, size);
        return;
    }
#endif
#ifdef LIBHMM_HAS_SSE2
    if (simd_available() && size >= 2) {
        scalar_multiply_sse2(a, scalar, result, size);
        return;
    }
#endif
#ifdef LIBHMM_HAS_NEON
    if (simd_available() && size >= 2) {
        scalar_multiply_neon(a, scalar, result, size);
        return;
    }
#endif
    scalar_multiply_fallback(a, scalar, result, size);
}

void SIMDOps::matrix_vector_multiply(const double* matrix, const double* vector, 
                                     double* result, std::size_t rows, std::size_t cols) noexcept {
    // Use SIMD-optimized dot products for each row
    for (std::size_t i = 0; i < rows; ++i) {
        result[i] = dot_product(&matrix[i * cols], vector, cols);
    }
}

//========== Fallback Implementations ==========

double SIMDOps::dot_product_fallback(const double* a, const double* b, std::size_t size) noexcept {
    double result = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

void SIMDOps::vector_add_fallback(const double* a, const double* b, double* result, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDOps::vector_multiply_fallback(const double* a, const double* b, double* result, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SIMDOps::scalar_multiply_fallback(const double* a, double scalar, double* result, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void SIMDOps::matrix_vector_multiply_fallback(const double* matrix, const double* vector, 
                                              double* result, std::size_t rows, std::size_t cols) noexcept {
    for (std::size_t i = 0; i < rows; ++i) {
        result[i] = 0.0;
        for (std::size_t j = 0; j < cols; ++j) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

//========== AVX Implementations ==========

#ifdef LIBHMM_HAS_AVX

double SIMDOps::dot_product_avx(const double* a, const double* b, std::size_t size) noexcept {
    __m256d sum = _mm256_setzero_pd();
    std::size_t simd_end = (size / DOUBLE_SIMD_WIDTH) * DOUBLE_SIMD_WIDTH;
    
    // Process SIMD blocks
    for (std::size_t i = 0; i < simd_end; i += DOUBLE_SIMD_WIDTH) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d prod = _mm256_mul_pd(va, vb);
        sum = _mm256_add_pd(sum, prod);
    }
    
    // Extract horizontal sum
    __m128d sum_low = _mm256_castpd256_pd128(sum);
    __m128d sum_high = _mm256_extractf128_pd(sum, 1);
    __m128d sum_combined = _mm_add_pd(sum_low, sum_high);
    __m128d sum_final = _mm_hadd_pd(sum_combined, sum_combined);
    
    double result;
    _mm_store_sd(&result, sum_final);
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

void SIMDOps::vector_add_avx(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / DOUBLE_SIMD_WIDTH) * DOUBLE_SIMD_WIDTH;
    
    // Process SIMD blocks
    for (std::size_t i = 0; i < simd_end; i += DOUBLE_SIMD_WIDTH) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vresult = _mm256_add_pd(va, vb);
        _mm256_store_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDOps::vector_multiply_avx(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / DOUBLE_SIMD_WIDTH) * DOUBLE_SIMD_WIDTH;
    
    // Process SIMD blocks
    for (std::size_t i = 0; i < simd_end; i += DOUBLE_SIMD_WIDTH) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vresult = _mm256_mul_pd(va, vb);
        _mm256_store_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SIMDOps::scalar_multiply_avx(const double* a, double scalar, double* result, std::size_t size) noexcept {
    __m256d vscalar = _mm256_set1_pd(scalar);
    std::size_t simd_end = (size / DOUBLE_SIMD_WIDTH) * DOUBLE_SIMD_WIDTH;
    
    // Process SIMD blocks
    for (std::size_t i = 0; i < simd_end; i += DOUBLE_SIMD_WIDTH) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vresult = _mm256_mul_pd(va, vscalar);
        _mm256_store_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

#endif // LIBHMM_HAS_AVX

//========== SSE2 Implementations ==========

#ifdef LIBHMM_HAS_SSE2

double SIMDOps::dot_product_sse2(const double* a, const double* b, std::size_t size) noexcept {
    __m128d sum = _mm_setzero_pd();
    std::size_t simd_end = (size / 2) * 2;
    
    // Process SIMD blocks (2 doubles at a time)
    for (std::size_t i = 0; i < simd_end; i += 2) {
        __m128d va = _mm_load_pd(&a[i]);
        __m128d vb = _mm_load_pd(&b[i]);
        __m128d prod = _mm_mul_pd(va, vb);
        sum = _mm_add_pd(sum, prod);
    }
    
    // Extract horizontal sum
    __m128d sum_shuf = _mm_shuffle_pd(sum, sum, 1);
    sum = _mm_add_sd(sum, sum_shuf);
    
    double result;
    _mm_store_sd(&result, sum);
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

void SIMDOps::vector_add_sse2(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / 2) * 2;
    
    // Process SIMD blocks
    for (std::size_t i = 0; i < simd_end; i += 2) {
        __m128d va = _mm_load_pd(&a[i]);
        __m128d vb = _mm_load_pd(&b[i]);
        __m128d vresult = _mm_add_pd(va, vb);
        _mm_store_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDOps::vector_multiply_sse2(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / 2) * 2;
    
    // Process SIMD blocks
    for (std::size_t i = 0; i < simd_end; i += 2) {
        __m128d va = _mm_load_pd(&a[i]);
        __m128d vb = _mm_load_pd(&b[i]);
        __m128d vresult = _mm_mul_pd(va, vb);
        _mm_store_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SIMDOps::scalar_multiply_sse2(const double* a, double scalar, double* result, std::size_t size) noexcept {
    __m128d vscalar = _mm_set1_pd(scalar);
    std::size_t simd_end = (size / 2) * 2;
    
    // Process SIMD blocks
    for (std::size_t i = 0; i < simd_end; i += 2) {
        __m128d va = _mm_load_pd(&a[i]);
        __m128d vresult = _mm_mul_pd(va, vscalar);
        _mm_store_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

#endif // LIBHMM_HAS_SSE2

//========== ARM NEON Implementations ==========

#ifdef LIBHMM_HAS_NEON

double SIMDOps::dot_product_neon(const double* a, const double* b, std::size_t size) noexcept {
    float64x2_t sum = vdupq_n_f64(0.0);
    std::size_t simd_end = (size / 2) * 2;
    
    // Process NEON blocks (2 doubles at a time)
    for (std::size_t i = 0; i < simd_end; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t prod = vmulq_f64(va, vb);
        sum = vaddq_f64(sum, prod);
    }
    
    // Extract horizontal sum
    double result = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

void SIMDOps::vector_add_neon(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / 2) * 2;
    
    // Process NEON blocks
    for (std::size_t i = 0; i < simd_end; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vresult = vaddq_f64(va, vb);
        vst1q_f64(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDOps::vector_multiply_neon(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / 2) * 2;
    
    // Process NEON blocks
    for (std::size_t i = 0; i < simd_end; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vresult = vmulq_f64(va, vb);
        vst1q_f64(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SIMDOps::scalar_multiply_neon(const double* a, double scalar, double* result, std::size_t size) noexcept {
    float64x2_t vscalar = vdupq_n_f64(scalar);
    std::size_t simd_end = (size / 2) * 2;
    
    // Process NEON blocks
    for (std::size_t i = 0; i < simd_end; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vresult = vmulq_f64(va, vscalar);
        vst1q_f64(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

#endif // LIBHMM_HAS_NEON

} // namespace performance
} // namespace libhmm
