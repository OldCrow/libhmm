#include "libhmm/common/optimized_vector.h"

namespace libhmm {

// Placeholder SIMD implementations for OptimizedVector<double>
// These will fall back to serial implementations until platform-specific SIMD code is added

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
std::pair<double, std::size_t> OptimizedVector<double>::max_element_simd() const {
    return max_element_serial();
}

template<>
std::pair<double, std::size_t> OptimizedVector<double>::min_element_simd() const {
    return min_element_serial();
}

// Placeholder SIMD implementations for OptimizedVector<float>

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
std::pair<float, std::size_t> OptimizedVector<float>::max_element_simd() const {
    return max_element_serial();
}

template<>
std::pair<float, std::size_t> OptimizedVector<float>::min_element_simd() const {
    return min_element_serial();
}

} // namespace libhmm
