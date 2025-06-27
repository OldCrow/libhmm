#include "libhmm/common/optimized_vector.h"
#include "libhmm/common/optimized_matrix.h"

// Temporary SIMD implementation stubs that fall back to serial operations
// TODO: Replace with actual platform-specific SIMD implementations

namespace libhmm {

// OptimizedVector SIMD stubs
template<>
void OptimizedVector<double>::add_simd(const OptimizedVector<double>& other) {
    add_serial(other);
}

template<>
void OptimizedVector<double>::subtract_simd(const OptimizedVector<double>& other) {
    subtract_serial(other);
}

template<>
void OptimizedVector<double>::scale_simd(const double& scalar) {
    scale_serial(scalar);
}

template<>
double OptimizedVector<double>::sum_simd() const {
    return sum_serial();
}

template<>
double OptimizedVector<double>::product_simd() const {
    return product_serial();
}

template<>
double OptimizedVector<double>::dot_simd(const OptimizedVector<double>& other) const {
    return dot_serial(other);
}

template<>
double OptimizedVector<double>::norm_simd() const {
    return norm_serial();
}

template<>
void OptimizedVector<double>::hadamard_simd(const OptimizedVector<double>& other) {
    hadamard_serial(other);
}

template<>
void OptimizedVector<double>::element_divide_simd(const OptimizedVector<double>& other) {
    element_divide_serial(other);
}

template<>
std::pair<double, OptimizedVector<double>::size_type> OptimizedVector<double>::max_element_simd() const {
    return max_element_serial();
}

template<>
std::pair<double, OptimizedVector<double>::size_type> OptimizedVector<double>::min_element_simd() const {
    return min_element_serial();
}

// OptimizedMatrix SIMD stubs
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

// Float versions (if needed)
template<>
void OptimizedVector<float>::add_simd(const OptimizedVector<float>& other) {
    add_serial(other);
}

template<>
void OptimizedVector<float>::subtract_simd(const OptimizedVector<float>& other) {
    subtract_serial(other);
}

template<>
void OptimizedVector<float>::scale_simd(const float& scalar) {
    scale_serial(scalar);
}

template<>
float OptimizedVector<float>::sum_simd() const {
    return sum_serial();
}

template<>
float OptimizedVector<float>::product_simd() const {
    return product_serial();
}

template<>
float OptimizedVector<float>::dot_simd(const OptimizedVector<float>& other) const {
    return dot_serial(other);
}

template<>
float OptimizedVector<float>::norm_simd() const {
    return norm_serial();
}

template<>
void OptimizedVector<float>::hadamard_simd(const OptimizedVector<float>& other) {
    hadamard_serial(other);
}

template<>
void OptimizedVector<float>::element_divide_simd(const OptimizedVector<float>& other) {
    element_divide_serial(other);
}

template<>
std::pair<float, OptimizedVector<float>::size_type> OptimizedVector<float>::max_element_simd() const {
    return max_element_serial();
}

template<>
std::pair<float, OptimizedVector<float>::size_type> OptimizedVector<float>::min_element_simd() const {
    return min_element_serial();
}

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
