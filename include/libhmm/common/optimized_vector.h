#ifndef LIBHMM_OPTIMIZED_VECTOR_H_
#define LIBHMM_OPTIMIZED_VECTOR_H_

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <numeric>
#include <cmath>

// Use robust performance infrastructure (includes parallel execution detection)
#include "libhmm/performance/simd_support.h"
#include "libhmm/performance/parallel_constants.h"
#include "common.h"
#include "basic_vector.h"  // Forward declaration for conversion constructor

namespace libhmm {

/**
 * High-performance Vector class with SIMD optimizations and parallel execution
 * Designed for numerical computing workloads in HMM algorithms
 * 
 * Features:
 * - SIMD-optimized mathematical operations
 * - Parallel execution for large vectors
 * - Cache-friendly memory layout
 * - Backward compatibility with BasicVector API
 * - Zero external dependencies (pure C++17)
 */
template<typename T>
class OptimizedVector {
private:
    std::vector<T> data_;
    
    /// SIMD optimization parameters using robust performance infrastructure
    static constexpr std::size_t SIMD_BLOCK_SIZE = constants::simd::DEFAULT_BLOCK_SIZE;
    static constexpr std::size_t PARALLEL_THRESHOLD = performance::parallel::MIN_WORK_PER_THREAD; // Use parallel ops for large vectors
    static constexpr std::size_t SIMD_ALIGNMENT = performance::simd::optimal_alignment();

public:
    // Type aliases for compatibility
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using const_reference = const T&;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    // Constructors
    OptimizedVector() = default;
    
    explicit OptimizedVector(size_type size) : data_(size) {}
    
    OptimizedVector(size_type size, const T& value) : data_(size, value) {}
    
    OptimizedVector(const std::vector<T>& vec) : data_(vec) {}
    OptimizedVector(std::vector<T>&& vec) noexcept : data_(std::move(vec)) {}
    
    OptimizedVector(std::initializer_list<T> init) : data_(init) {}
    
    // Conversion constructor from BasicVector (enables dynamic upgrading)
    explicit OptimizedVector(const BasicVector<T>& basic_vec) 
        : data_(basic_vec.get_data()) {
        // Copy data from BasicVector to enable seamless transition to optimized operations
    }
    
    // Default copy/move operations
    OptimizedVector(const OptimizedVector&) = default;
    OptimizedVector(OptimizedVector&&) noexcept = default;
    OptimizedVector& operator=(const OptimizedVector&) = default;
    OptimizedVector& operator=(OptimizedVector&&) noexcept = default;

    // Element access (compatible with BasicVector)
    reference operator[](size_type index) noexcept { return data_[index]; }
    const_reference operator[](size_type index) const noexcept { return data_[index]; }
    
    reference operator()(size_type index) noexcept { return data_[index]; }
    const_reference operator()(size_type index) const noexcept { return data_[index]; }

    reference at(size_type index) { return data_.at(index); }
    const_reference at(size_type index) const { return data_.at(index); }

    // Size and capacity
    size_type size() const noexcept { return data_.size(); }
    size_type capacity() const noexcept { return data_.capacity(); }
    bool empty() const noexcept { return data_.empty(); }

    // Resize operations
    void resize(size_type size) { data_.resize(size); }
    void resize(size_type size, const T& value) { data_.resize(size, value); }
    void reserve(size_type capacity) { data_.reserve(capacity); }

    // Clear vector (optimized)
    void clear() {
        if (data_.size() > PARALLEL_THRESHOLD) {
            clear_parallel();
        } else {
            clear_serial();
        }
    }

    // Push/pop operations
    void push_back(const T& value) { data_.push_back(value); }
    void push_back(T&& value) { data_.push_back(std::move(value)); }
    void pop_back() { data_.pop_back(); }

    // Raw data access
    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    // Iterator support
    iterator begin() noexcept { return data_.begin(); }
    iterator end() noexcept { return data_.end(); }
    const_iterator begin() const noexcept { return data_.begin(); }
    const_iterator end() const noexcept { return data_.end(); }
    const_iterator cbegin() const noexcept { return data_.cbegin(); }
    const_iterator cend() const noexcept { return data_.cend(); }

    // SIMD-optimized vector operations
    OptimizedVector& operator+=(const OptimizedVector& other) {
        if (size() != other.size()) {
            throw std::invalid_argument("Vector dimensions must match for addition");
        }
        
        // Temporarily use only serial implementation until SIMD infrastructure is complete
        add_serial(other);
        return *this;
    }

    OptimizedVector& operator-=(const OptimizedVector& other) {
        if (size() != other.size()) {
            throw std::invalid_argument("Vector dimensions must match for subtraction");
        }
        
        // Temporarily use only serial implementation until SIMD infrastructure is complete
        subtract_serial(other);
        return *this;
    }

    OptimizedVector& operator*=(const T& scalar) {
        // Temporarily use only serial implementation until SIMD infrastructure is complete
        scale_serial(scalar);
        return *this;
    }

    OptimizedVector& operator/=(const T& scalar) {
        return *this *= (T{1} / scalar);
    }

    // Comparison operators
    bool operator==(const OptimizedVector& other) const {
        return data_ == other.data_;
    }

    bool operator!=(const OptimizedVector& other) const {
        return !(*this == other);
    }

    // Mathematical operations (SIMD-optimized)
    
    /// Sum of all elements (parallel for large vectors)
    T sum() const {
        if (data_.size() > PARALLEL_THRESHOLD) {
            return sum_parallel();
        } else {
            return sum_serial();
        }
    }

    /// Product of all elements
    T product() const {
        // Temporarily use only serial implementation until SIMD infrastructure is complete
        return product_serial();
    }

    /// Dot product with another vector (SIMD-optimized)
    T dot(const OptimizedVector& other) const {
        if (size() != other.size()) {
            throw std::invalid_argument("Vector dimensions must match for dot product");
        }
        
        // Temporarily use only serial implementation until SIMD infrastructure is complete
        return dot_serial(other);
    }

    /// L2 norm (Euclidean norm) - SIMD optimized
    T norm() const {
        // Temporarily use only serial implementation until SIMD infrastructure is complete
        return norm_serial();
    }

    /// Normalize vector to unit length
    OptimizedVector& normalize() {
        T n = norm();
        if (n > constants::precision::ZERO) {
            *this *= (T{1} / n);
        }
        return *this;
    }

    /// Element-wise multiplication (Hadamard product)
    OptimizedVector& elementwise_multiply(const OptimizedVector& other) {
        if (size() != other.size()) {
            throw std::invalid_argument("Vector dimensions must match for element-wise multiplication");
        }
        
        // Temporarily use only serial implementation until SIMD infrastructure is complete
        hadamard_serial(other);
        return *this;
    }
    
    /// Element-wise multiplication (alias for compatibility with BasicVector)
    OptimizedVector& element_multiply(const OptimizedVector& other) {
        return elementwise_multiply(other);
    }
    
    /// Element-wise division
    OptimizedVector& element_divide(const OptimizedVector& other) {
        if (size() != other.size()) {
            throw std::invalid_argument("Vector dimensions must match for element-wise division");
        }
        
        // Temporarily use only serial implementation until SIMD infrastructure is complete
        element_divide_serial(other);
        return *this;
    }

    /// Find maximum element and its index
    std::pair<T, size_type> max_element() const {
        if (empty()) {
            throw std::runtime_error("Cannot find maximum of empty vector");
        }
        
        // Temporarily use only serial implementation until SIMD infrastructure is complete
        return max_element_serial();
    }

    /// Find minimum element and its index
    std::pair<T, size_type> min_element() const {
        if (empty()) {
            throw std::runtime_error("Cannot find minimum of empty vector");
        }
        
        // Temporarily use only serial implementation until SIMD infrastructure is complete
        return min_element_serial();
    }

    /// Fill with value (parallel for large vectors)
    void fill(const T& value) {
        if (data_.size() > PARALLEL_THRESHOLD) {
            fill_parallel(value);
        } else {
            std::fill(data_.begin(), data_.end(), value);
        }
    }

    /// Apply function to all elements (parallel when beneficial)
    template<typename Func>
    OptimizedVector& apply(Func func) {
        if (data_.size() > PARALLEL_THRESHOLD) {
            apply_parallel(func);
        } else {
            apply_serial(func);
        }
        return *this;
    }

private:
    // Serial implementations (fallback)
    void clear_serial() {
        std::fill(data_.begin(), data_.end(), T{});
    }

    void add_serial(const OptimizedVector& other) {
        for (size_type i = 0; i < size(); ++i) {
            data_[i] += other.data_[i];
        }
    }

    void subtract_serial(const OptimizedVector& other) {
        for (size_type i = 0; i < size(); ++i) {
            data_[i] -= other.data_[i];
        }
    }

    void scale_serial(const T& scalar) {
        for (auto& element : data_) {
            element *= scalar;
        }
    }

    T sum_serial() const {
        return std::accumulate(data_.begin(), data_.end(), T{});
    }

    T product_serial() const {
        return std::accumulate(data_.begin(), data_.end(), T{1}, std::multiplies<T>());
    }

    T dot_serial(const OptimizedVector& other) const {
        T result = T{};
        for (size_type i = 0; i < size(); ++i) {
            result += data_[i] * other.data_[i];
        }
        return result;
    }

    T norm_serial() const {
        T sum_of_squares = T{};
        for (const auto& element : data_) {
            sum_of_squares += element * element;
        }
        return std::sqrt(sum_of_squares);
    }

    void hadamard_serial(const OptimizedVector& other) {
        for (size_type i = 0; i < size(); ++i) {
            data_[i] *= other.data_[i];
        }
    }

    std::pair<T, size_type> max_element_serial() const {
        auto it = std::max_element(data_.begin(), data_.end());
        return {*it, static_cast<size_type>(std::distance(data_.begin(), it))};
    }

    std::pair<T, size_type> min_element_serial() const {
        auto it = std::min_element(data_.begin(), data_.end());
        return {*it, static_cast<size_type>(std::distance(data_.begin(), it))};
    }
    
    void element_divide_serial(const OptimizedVector& other) {
        for (size_type i = 0; i < size(); ++i) {
            data_[i] /= other.data_[i];
        }
    }

    // Parallel implementations (for large vectors)
    void clear_parallel() {
#if LIBHMM_HAS_PARALLEL_EXECUTION
        std::fill(std::execution::par_unseq, data_.begin(), data_.end(), T{});
#else
        clear_serial();
#endif
    }

    T sum_parallel() const {
#if LIBHMM_HAS_PARALLEL_EXECUTION
        return std::reduce(std::execution::par_unseq, data_.begin(), data_.end(), T{});
#else
        return sum_serial();
#endif
    }

    void fill_parallel(const T& value) {
#if LIBHMM_HAS_PARALLEL_EXECUTION
        std::fill(std::execution::par_unseq, data_.begin(), data_.end(), value);
#else
        std::fill(data_.begin(), data_.end(), value);
#endif
    }

    template<typename Func>
    void apply_parallel(Func func) {
#if LIBHMM_HAS_PARALLEL_EXECUTION
        std::for_each(std::execution::par_unseq, data_.begin(), data_.end(), func);
#else
        apply_serial(func);
#endif
    }

    template<typename Func>
    void apply_serial(Func func) {
        std::for_each(data_.begin(), data_.end(), func);
    }

    // SIMD implementations (platform-specific)
    void add_simd(const OptimizedVector& other);
    void subtract_simd(const OptimizedVector& other);
    void scale_simd(const T& scalar);
    T sum_simd() const;
    T product_simd() const;
    T dot_simd(const OptimizedVector& other) const;
    T norm_simd() const;
    void hadamard_simd(const OptimizedVector& other);
    void element_divide_simd(const OptimizedVector& other);
    std::pair<T, size_type> max_element_simd() const;
    std::pair<T, size_type> min_element_simd() const;
};

// Binary arithmetic operators (compatible with BasicVector)
template<typename T>
OptimizedVector<T> operator+(const OptimizedVector<T>& lhs, const OptimizedVector<T>& rhs) {
    OptimizedVector<T> result = lhs;
    result += rhs;
    return result;
}

template<typename T>
OptimizedVector<T> operator-(const OptimizedVector<T>& lhs, const OptimizedVector<T>& rhs) {
    OptimizedVector<T> result = lhs;
    result -= rhs;
    return result;
}

template<typename T>
OptimizedVector<T> operator*(const OptimizedVector<T>& vector, const T& scalar) {
    OptimizedVector<T> result = vector;
    result *= scalar;
    return result;
}

template<typename T>
OptimizedVector<T> operator*(const T& scalar, const OptimizedVector<T>& vector) {
    return vector * scalar;
}

template<typename T>
OptimizedVector<T> operator/(const OptimizedVector<T>& vector, const T& scalar) {
    OptimizedVector<T> result = vector;
    result /= scalar;
    return result;
}

// Mathematical functions for uBLAS compatibility

/// Element-wise multiplication (Hadamard product) - compatible with BasicVector
template<typename T>
OptimizedVector<T> element_prod(const OptimizedVector<T>& lhs, const OptimizedVector<T>& rhs) {
    OptimizedVector<T> result = lhs;
    result.element_multiply(rhs);
    return result;
}

/// Element-wise division - compatible with BasicVector
template<typename T>
OptimizedVector<T> element_div(const OptimizedVector<T>& lhs, const OptimizedVector<T>& rhs) {
    OptimizedVector<T> result = lhs;
    result.element_divide(rhs);
    return result;
}

/// Inner product (dot product) - uBLAS compatibility
/// Note: This is the central definition for OptimizedVector inner_prod operations
/// (mirrors the pattern used in basic classes where matrix.h contains the definitive inner_prod)
template<typename T>
T inner_prod(const OptimizedVector<T>& lhs, const OptimizedVector<T>& rhs) {
    return lhs.dot(rhs);
}

/// Stream output operator
template<typename T>
std::ostream& operator<<(std::ostream& os, const OptimizedVector<T>& vector) {
    os << "[";
    for (std::size_t i = 0; i < vector.size(); ++i) {
        os << std::setprecision(6) << vector[i];
        if (i < vector.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}

// Type aliases for common use cases
using OptimizedVectorF = OptimizedVector<float>;
using OptimizedVectorD = OptimizedVector<double>;
using OptimizedVectorI = OptimizedVector<int>;

// Factory function with type deduction
template<typename T>
auto make_optimized_vector(std::size_t size, const T& init_value = T{}) {
    return OptimizedVector<T>(size, init_value);
}

} // namespace libhmm

#endif // LIBHMM_OPTIMIZED_VECTOR_H_
