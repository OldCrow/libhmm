#ifndef LIBHMM_VECTOR_H_
#define LIBHMM_VECTOR_H_

#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace libhmm {

// Fixed version with BasicVector class naming to avoid conflicts

/**
 * Lightweight Vector class designed to replace boost::numeric::ublas::vector
 * with better performance and SIMD-friendly operations.
 * 
 * Features:
 * - Based on std::vector for optimal standard library integration
 * - SIMD-friendly contiguous memory layout
 * - Compatible API with existing uBLAS usage patterns
 * - Enhanced mathematical operations for HMM computations
 * - Zero external dependencies (pure C++17)
 */
template<typename T>
class BasicVector {
private:
    std::vector<T> data_;

public:
    // Type aliases for compatibility
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using const_reference = const T&;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    // Default constructor
    BasicVector() = default;
    
    // Constructor with size
    explicit BasicVector(size_type size) : data_(size) {}
        
    // Constructor with size and default value
    BasicVector(size_type size, const T& value) : data_(size, value) {}
        
    // Constructor from std::vector
    explicit BasicVector(const std::vector<T>& vec) : data_(vec) {}
    explicit BasicVector(std::vector<T>&& vec) : data_(std::move(vec)) {}
        
    // Copy constructor
    BasicVector(const BasicVector& other) : data_(other.data_) {}
        
    // Move constructor
    BasicVector(BasicVector&& other) noexcept : data_(std::move(other.data_)) {}
    
    // Initializer list constructor
    BasicVector(std::initializer_list<T> init) : data_(init) {}
    
    // Copy assignment
    BasicVector& operator=(const BasicVector& other) {
        if (this != &other) {
            data_ = other.data_;
        }
        return *this;
    }
    
    // Move assignment
    BasicVector& operator=(BasicVector&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
        }
        return *this;
    }

    // Element access
    reference operator[](size_type index) { return data_[index]; }
    const_reference operator[](size_type index) const { return data_[index]; }
    
    reference operator()(size_type index) { return data_[index]; }
    const_reference operator()(size_type index) const { return data_[index]; }

    // Bounds-checked element access
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

    // Clear vector (set all elements to zero)
    void clear() {
        std::fill(data_.begin(), data_.end(), T{});
    }

    // Push/pop operations
    void push_back(const T& value) { data_.push_back(value); }
    void push_back(T&& value) { data_.push_back(std::move(value)); }
    void pop_back() { data_.pop_back(); }

    // Raw data access for SIMD operations
    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    // Iterator support
    iterator begin() noexcept { return data_.begin(); }
    iterator end() noexcept { return data_.end(); }
    const_iterator begin() const noexcept { return data_.begin(); }
    const_iterator end() const noexcept { return data_.end(); }
    const_iterator cbegin() const noexcept { return data_.cbegin(); }
    const_iterator cend() const noexcept { return data_.cend(); }

    // Vector operations
    BasicVector& operator+=(const BasicVector& other) {
        if (size() != other.size()) {
            throw std::invalid_argument("Vector dimensions must match for addition");
        }
        for (size_type i = 0; i < size(); ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    BasicVector& operator-=(const BasicVector& other) {
        if (size() != other.size()) {
            throw std::invalid_argument("Vector dimensions must match for subtraction");
        }
        for (size_type i = 0; i < size(); ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    BasicVector& operator*=(const T& scalar) {
        for (auto& element : data_) {
            element *= scalar;
        }
        return *this;
    }

    BasicVector& operator/=(const T& scalar) {
        for (auto& element : data_) {
            element /= scalar;
        }
        return *this;
    }

    // Comparison operators
    bool operator==(const BasicVector& other) const {
        return data_ == other.data_;
    }

    bool operator!=(const BasicVector& other) const {
        return !(*this == other);
    }

    // Mathematical operations specific to HMM computations
    
    // Sum of all elements
    T sum() const {
        return std::accumulate(data_.begin(), data_.end(), T{});
    }
    
    // Product of all elements
    T product() const {
        return std::accumulate(data_.begin(), data_.end(), T{1}, std::multiplies<T>());
    }
    
    // Dot product with another vector
    T dot(const BasicVector& other) const {
        if (size() != other.size()) {
            throw std::invalid_argument("Vector dimensions must match for dot product");
        }
        T result = T{};
        for (size_type i = 0; i < size(); ++i) {
            result += data_[i] * other.data_[i];
        }
        return result;
    }
    
    // L2 norm (Euclidean norm)
    T norm() const {
        T sum_of_squares = T{};
        for (const auto& element : data_) {
            sum_of_squares += element * element;
        }
        return std::sqrt(sum_of_squares);
    }
    
    // Normalize vector to unit length
    BasicVector& normalize() {
        T n = norm();
        if (n > T{}) {
            *this /= n;
        }
        return *this;
    }
    
    // Element-wise multiplication (Hadamard product)
    BasicVector& element_multiply(const BasicVector& other) {
        if (size() != other.size()) {
            throw std::invalid_argument("Vector dimensions must match for element-wise multiplication");
        }
        for (size_type i = 0; i < size(); ++i) {
            data_[i] *= other.data_[i];
        }
        return *this;
    }
    
    // Element-wise division
    BasicVector& element_divide(const BasicVector& other) {
        if (size() != other.size()) {
            throw std::invalid_argument("Vector dimensions must match for element-wise division");
        }
        for (size_type i = 0; i < size(); ++i) {
            data_[i] /= other.data_[i];
        }
        return *this;
    }

    // Access to underlying std::vector
    std::vector<T>& get_data() { return data_; }
    const std::vector<T>& get_data() const { return data_; }
};

// Binary arithmetic operators
template<typename T>
BasicVector<T> operator+(const BasicVector<T>& lhs, const BasicVector<T>& rhs) {
    BasicVector<T> result = lhs;
    result += rhs;
    return result;
}

template<typename T>
BasicVector<T> operator-(const BasicVector<T>& lhs, const BasicVector<T>& rhs) {
    BasicVector<T> result = lhs;
    result -= rhs;
    return result;
}

template<typename T>
BasicVector<T> operator*(const BasicVector<T>& vector, const T& scalar) {
    BasicVector<T> result = vector;
    result *= scalar;
    return result;
}

template<typename T>
BasicVector<T> operator*(const T& scalar, const BasicVector<T>& vector) {
    return vector * scalar;
}

template<typename T>
BasicVector<T> operator/(const BasicVector<T>& vector, const T& scalar) {
    BasicVector<T> result = vector;
    result /= scalar;
    return result;
}

// Stream output operator
template<typename T>
std::ostream& operator<<(std::ostream& os, const BasicVector<T>& vector) {
    os << "[";
    for (std::size_t i = 0; i < vector.size(); ++i) {
        os << std::setprecision(6) << vector[i];
        if (i < vector.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}

// Mathematical functions for vectors

// Element-wise multiplication (Hadamard product)
template<typename T>
BasicVector<T> element_prod(const BasicVector<T>& lhs, const BasicVector<T>& rhs) {
    BasicVector<T> result = lhs;
    result.element_multiply(rhs);
    return result;
}

// Element-wise division
template<typename T>
BasicVector<T> element_div(const BasicVector<T>& lhs, const BasicVector<T>& rhs) {
    BasicVector<T> result = lhs;
    result.element_divide(rhs);
    return result;
}

// Dot product
template<typename T>
T inner_prod(const BasicVector<T>& lhs, const BasicVector<T>& rhs) {
    return lhs.dot(rhs);
}

} // namespace libhmm

#endif // LIBHMM_VECTOR_H_
