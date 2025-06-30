#include "libhmm/common/optimized_matrix.h"

namespace libhmm {

// Placeholder SIMD implementations for OptimizedMatrix<double>
// These will fall back to serial implementations until platform-specific SIMD code is added

template<>
void OptimizedMatrix<double>::add_simd(const OptimizedMatrix<double>& other) {
    add_serial(other);
}

template<>
void OptimizedMatrix<double>::subtract_simd(const OptimizedMatrix<double>& other) {
    subtract_serial(other);
}

template<>
void OptimizedMatrix<double>::scale_simd(const double& scalar) {
    scale_serial(scalar);
}

template<>
void OptimizedMatrix<double>::multiply_vector_simd(const OptimizedVector<double>& vec, OptimizedVector<double>& result) const {
    multiply_vector_serial(vec, result);
}

template<>
void OptimizedMatrix<double>::hadamard_simd(const OptimizedMatrix<double>& other) {
    hadamard_serial(other);
}

template<>
double OptimizedMatrix<double>::sum_simd() const {
    return sum_serial();
}

template<>
double OptimizedMatrix<double>::norm_simd() const {
    return norm_serial();
}

// Placeholder SIMD implementations for OptimizedMatrix<float>

template<>
void OptimizedMatrix<float>::add_simd(const OptimizedMatrix<float>& other) {
    add_serial(other);
}

template<>
void OptimizedMatrix<float>::subtract_simd(const OptimizedMatrix<float>& other) {
    subtract_serial(other);
}

template<>
void OptimizedMatrix<float>::scale_simd(const float& scalar) {
    scale_serial(scalar);
}

template<>
void OptimizedMatrix<float>::multiply_vector_simd(const OptimizedVector<float>& vec, OptimizedVector<float>& result) const {
    multiply_vector_serial(vec, result);
}

template<>
void OptimizedMatrix<float>::hadamard_simd(const OptimizedMatrix<float>& other) {
    hadamard_serial(other);
}

template<>
float OptimizedMatrix<float>::sum_simd() const {
    return sum_serial();
}

template<>
float OptimizedMatrix<float>::norm_simd() const {
    return norm_serial();
}

} // namespace libhmm
