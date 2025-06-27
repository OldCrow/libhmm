#ifndef LIBHMM_OPTIMIZED_MATRIX_H_
#define LIBHMM_OPTIMIZED_MATRIX_H_

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <numeric>
#include <iostream>
#include <iomanip>

// Check for C++17 parallel execution support
#ifdef __cpp_lib_execution
#include <execution>
#define LIBHMM_HAS_PARALLEL_EXECUTION 1
#else
#define LIBHMM_HAS_PARALLEL_EXECUTION 0
#endif

// Platform and system headers for SIMD
#ifdef _MSC_VER
    #include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
    #include <immintrin.h>
#include <x86intrin.h>
#endif

#include "common.h"

namespace libhmm {

// Forward declaration for compatibility
template<typename T> class OptimizedVector;

/**
 * High-performance Matrix class with SIMD optimizations and parallel execution
 * Designed for numerical computing workloads in HMM algorithms
 * 
 * Features:
 * - Contiguous memory storage for optimal cache performance
 * - Row-major ordering for better CPU cache utilization
 * - SIMD-optimized mathematical operations
 * - Parallel execution for large matrices
 * - Backward compatibility with BasicMatrix API
 * - Zero external dependencies (pure C++17)
 */
template<typename T>
class OptimizedMatrix {
private:
    std::vector<T> data_;
    std::size_t rows_;
    std::size_t cols_;
    
    /// SIMD optimization parameters
    static constexpr std::size_t SIMD_BLOCK_SIZE = constants::simd::DEFAULT_BLOCK_SIZE;
    static constexpr std::size_t PARALLEL_THRESHOLD = 10000; // Use parallel ops for matrices > 10k elements
    static constexpr std::size_t CACHE_BLOCK_SIZE = constants::simd::MAX_BLOCK_SIZE;

public:
    // Type aliases for compatibility
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using const_reference = const T&;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    // Constructors
    OptimizedMatrix() : rows_(0), cols_(0) {}
    
    OptimizedMatrix(size_type rows, size_type cols) 
        : data_(rows * cols), rows_(rows), cols_(cols) {}
        
    OptimizedMatrix(size_type rows, size_type cols, const T& value)
        : data_(rows * cols, value), rows_(rows), cols_(cols) {}
        
    // Default copy/move operations
    OptimizedMatrix(const OptimizedMatrix&) = default;
    OptimizedMatrix(OptimizedMatrix&&) noexcept = default;
    OptimizedMatrix& operator=(const OptimizedMatrix&) = default;
    OptimizedMatrix& operator=(OptimizedMatrix&&) noexcept = default;

    // Element access (row, col) - compatible with BasicMatrix
    reference operator()(size_type row, size_type col) noexcept {
        return data_[row * cols_ + col];
    }
    
    const_reference operator()(size_type row, size_type col) const noexcept {
        return data_[row * cols_ + col];
    }

    // Bounds-checked element access
    reference at(size_type row, size_type col) {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Matrix index out of bounds");
        }
        return data_[row * cols_ + col];
    }
    
    const_reference at(size_type row, size_type col) const {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Matrix index out of bounds");
        }
        return data_[row * cols_ + col];
    }

    // Dimension accessors (compatible with BasicMatrix)
    size_type size1() const noexcept { return rows_; }
    size_type size2() const noexcept { return cols_; }
    size_type rows() const noexcept { return rows_; }
    size_type cols() const noexcept { return cols_; }
    size_type size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    // Resize operations
    void resize(size_type rows, size_type cols) {
        data_.resize(rows * cols);
        rows_ = rows;
        cols_ = cols;
    }
    
    void resize(size_type rows, size_type cols, const T& value) {
        data_.resize(rows * cols, value);
        rows_ = rows;
        cols_ = cols;
    }

    // Clear matrix (optimized for large matrices)
    void clear() {
        if (data_.size() > PARALLEL_THRESHOLD) {
            clear_parallel();
        } else {
            clear_serial();
        }
    }

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

    // SIMD-optimized matrix operations
    OptimizedMatrix& operator+=(const OptimizedMatrix& other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }
        
        if (data_.size() > SIMD_BLOCK_SIZE) {
            add_simd(other);
        } else {
            add_serial(other);
        }
        return *this;
    }

    OptimizedMatrix& operator-=(const OptimizedMatrix& other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }
        
        if (data_.size() > SIMD_BLOCK_SIZE) {
            subtract_simd(other);
        } else {
            subtract_serial(other);
        }
        return *this;
    }

    OptimizedMatrix& operator*=(const T& scalar) {
        if (data_.size() > SIMD_BLOCK_SIZE) {
            scale_simd(scalar);
        } else {
            scale_serial(scalar);
        }
        return *this;
    }

    OptimizedMatrix& operator/=(const T& scalar) {
        return *this *= (T{1} / scalar);
    }

    // Comparison operators
    bool operator==(const OptimizedMatrix& other) const {
        return rows_ == other.rows_ && cols_ == other.cols_ && data_ == other.data_;
    }

    bool operator!=(const OptimizedMatrix& other) const {
        return !(*this == other);
    }

    // Advanced mathematical operations

    /// Matrix-vector multiplication (optimized)
    OptimizedVector<T> multiply_vector(const OptimizedVector<T>& vec) const {
        if (cols_ != vec.size()) {
            throw std::invalid_argument("Matrix columns must match vector size for multiplication");
        }
        
        OptimizedVector<T> result(rows_);
        
        if (data_.size() > SIMD_BLOCK_SIZE) {
            multiply_vector_simd(vec, result);
        } else {
            multiply_vector_serial(vec, result);
        }
        
        return result;
    }

    /// Matrix-matrix multiplication (cache-blocked for large matrices)
    OptimizedMatrix multiply(const OptimizedMatrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        }
        
        OptimizedMatrix result(rows_, other.cols_);
        
        if (data_.size() > PARALLEL_THRESHOLD || other.data_.size() > PARALLEL_THRESHOLD) {
            multiply_blocked(other, result);
        } else {
            multiply_serial(other, result);
        }
        
        return result;
    }

    /// Transpose (cache-optimized)
    OptimizedMatrix transpose() const {
        OptimizedMatrix result(cols_, rows_);
        
        if (data_.size() > CACHE_BLOCK_SIZE) {
            transpose_blocked(result);
        } else {
            transpose_serial(result);
        }
        
        return result;
    }

    /// Get a row as an OptimizedVector (copies data)
    OptimizedVector<T> row(size_type row_index) const {
        if (row_index >= rows_) {
            throw std::out_of_range("Row index out of bounds");
        }
        
        OptimizedVector<T> result(cols_);
        const T* row_start = data_.data() + row_index * cols_;
        std::copy(row_start, row_start + cols_, result.data());
        return result;
    }

    /// Get a column as an OptimizedVector (copies data)
    OptimizedVector<T> column(size_type col_index) const {
        if (col_index >= cols_) {
            throw std::out_of_range("Column index out of bounds");
        }
        
        OptimizedVector<T> result(rows_);
        for (size_type i = 0; i < rows_; ++i) {
            result[i] = data_[i * cols_ + col_index];
        }
        return result;
    }

    /// Set a row from an OptimizedVector
    void set_row(size_type row_index, const OptimizedVector<T>& vec) {
        if (row_index >= rows_) {
            throw std::out_of_range("Row index out of bounds");
        }
        if (vec.size() != cols_) {
            throw std::invalid_argument("Vector size must match matrix columns");
        }
        
        T* row_start = data_.data() + row_index * cols_;
        std::copy(vec.data(), vec.data() + cols_, row_start);
    }

    /// Set a column from an OptimizedVector
    void set_column(size_type col_index, const OptimizedVector<T>& vec) {
        if (col_index >= cols_) {
            throw std::out_of_range("Column index out of bounds");
        }
        if (vec.size() != rows_) {
            throw std::invalid_argument("Vector size must match matrix rows");
        }
        
        for (size_type i = 0; i < rows_; ++i) {
            data_[i * cols_ + col_index] = vec[i];
        }
    }

    /// Element-wise operations (Hadamard)
    OptimizedMatrix& hadamard_multiply(const OptimizedMatrix& other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");
        }
        
        if (data_.size() > SIMD_BLOCK_SIZE) {
            hadamard_simd(other);
        } else {
            hadamard_serial(other);
        }
        return *this;
    }

    /// Fill with value (parallel for large matrices)
    void fill(const T& value) {
        if (data_.size() > PARALLEL_THRESHOLD) {
            fill_parallel(value);
        } else {
            std::fill(data_.begin(), data_.end(), value);
        }
    }

    /// Sum of all elements
    T sum() const {
        if (data_.size() > PARALLEL_THRESHOLD) {
            return sum_parallel();
        } else if (data_.size() > SIMD_BLOCK_SIZE) {
            return sum_simd();
        } else {
            return sum_serial();
        }
    }

    /// Frobenius norm (matrix norm)
    T norm() const {
        if (data_.size() > SIMD_BLOCK_SIZE) {
            return norm_simd();
        } else {
            return norm_serial();
        }
    }

    /// Apply function to all elements
    template<typename Func>
    OptimizedMatrix& apply(Func func) {
        if (data_.size() > PARALLEL_THRESHOLD) {
            apply_parallel(func);
        } else {
            apply_serial(func);
        }
        return *this;
    }
    
    /// Row sums - returns vector of row sums (critical for probability normalization)
    OptimizedVector<T> row_sums() const {
        OptimizedVector<T> result(rows_);
        for (size_type i = 0; i < rows_; ++i) {
            T sum = T{};
            for (size_type j = 0; j < cols_; ++j) {
                sum += (*this)(i, j);
            }
            result[i] = sum;
        }
        return result;
    }
    
    /// Column sums - returns vector of column sums (critical for probability normalization)
    OptimizedVector<T> column_sums() const {
        OptimizedVector<T> result(cols_);
        for (size_type j = 0; j < cols_; ++j) {
            T sum = T{};
            for (size_type i = 0; i < rows_; ++i) {
                sum += (*this)(i, j);
            }
            result[j] = sum;
        }
        return result;
    }
    
    /// Normalize rows to sum to 1.0 (essential for stochastic matrices in HMM)
    OptimizedMatrix& normalize_rows() {
        for (size_type i = 0; i < rows_; ++i) {
            T row_sum = T{};
            for (size_type j = 0; j < cols_; ++j) {
                row_sum += (*this)(i, j);
            }
            if (row_sum > T{}) {
                for (size_type j = 0; j < cols_; ++j) {
                    (*this)(i, j) /= row_sum;
                }
            }
        }
        return *this;
    }
    
    /// Normalize columns to sum to 1.0 (essential for emission matrices in HMM)
    OptimizedMatrix& normalize_columns() {
        for (size_type j = 0; j < cols_; ++j) {
            T col_sum = T{};
            for (size_type i = 0; i < rows_; ++i) {
                col_sum += (*this)(i, j);
            }
            if (col_sum > T{}) {
                for (size_type i = 0; i < rows_; ++i) {
                    (*this)(i, j) /= col_sum;
                }
            }
        }
        return *this;
    }
    
    /// In-place transpose for square matrices (more efficient for square matrices)
    OptimizedMatrix& transpose_inplace() {
        if (rows_ != cols_) {
            throw std::invalid_argument("In-place transpose only supported for square matrices");
        }
        for (size_type i = 0; i < rows_; ++i) {
            for (size_type j = i + 1; j < cols_; ++j) {
                std::swap((*this)(i, j), (*this)(j, i));
            }
        }
        return *this;
    }

private:
    // Serial implementations
    void clear_serial() {
        std::fill(data_.begin(), data_.end(), T{});
    }

    void add_serial(const OptimizedMatrix& other) {
        for (size_type i = 0; i < data_.size(); ++i) {
            data_[i] += other.data_[i];
        }
    }

    void subtract_serial(const OptimizedMatrix& other) {
        for (size_type i = 0; i < data_.size(); ++i) {
            data_[i] -= other.data_[i];
        }
    }

    void scale_serial(const T& scalar) {
        for (auto& element : data_) {
            element *= scalar;
        }
    }

    void multiply_vector_serial(const OptimizedVector<T>& vec, OptimizedVector<T>& result) const {
        for (size_type i = 0; i < rows_; ++i) {
            T sum = T{};
            for (size_type j = 0; j < cols_; ++j) {
                sum += (*this)(i, j) * vec[j];
            }
            result[i] = sum;
        }
    }

    void multiply_serial(const OptimizedMatrix& other, OptimizedMatrix& result) const {
        for (size_type i = 0; i < rows_; ++i) {
            for (size_type j = 0; j < other.cols_; ++j) {
                T sum = T{};
                for (size_type k = 0; k < cols_; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
    }

    void transpose_serial(OptimizedMatrix& result) const {
        for (size_type i = 0; i < rows_; ++i) {
            for (size_type j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
    }

    void hadamard_serial(const OptimizedMatrix& other) {
        for (size_type i = 0; i < data_.size(); ++i) {
            data_[i] *= other.data_[i];
        }
    }

    T sum_serial() const {
        return std::accumulate(data_.begin(), data_.end(), T{});
    }

    T norm_serial() const {
        T sum_of_squares = T{};
        for (const auto& element : data_) {
            sum_of_squares += element * element;
        }
        return std::sqrt(sum_of_squares);
    }

    // Parallel implementations
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

    // Cache-blocked implementations for large matrices
    void multiply_blocked(const OptimizedMatrix& other, OptimizedMatrix& result) const {
        const size_type block_size = CACHE_BLOCK_SIZE;
        
        // Initialize result to zero
        result.clear();
        
        // Cache-blocked matrix multiplication
        for (size_type ii = 0; ii < rows_; ii += block_size) {
            for (size_type jj = 0; jj < other.cols_; jj += block_size) {
                for (size_type kk = 0; kk < cols_; kk += block_size) {
                    // Block boundaries
                    size_type i_end = std::min(ii + block_size, rows_);
                    size_type j_end = std::min(jj + block_size, other.cols_);
                    size_type k_end = std::min(kk + block_size, cols_);
                    
                    // Block multiplication
                    for (size_type i = ii; i < i_end; ++i) {
                        for (size_type j = jj; j < j_end; ++j) {
                            T sum = T{};
                            for (size_type k = kk; k < k_end; ++k) {
                                sum += (*this)(i, k) * other(k, j);
                            }
                            result(i, j) += sum;
                        }
                    }
                }
            }
        }
    }

    void transpose_blocked(OptimizedMatrix& result) const {
        const size_type block_size = CACHE_BLOCK_SIZE;
        
        for (size_type ii = 0; ii < rows_; ii += block_size) {
            for (size_type jj = 0; jj < cols_; jj += block_size) {
                size_type i_end = std::min(ii + block_size, rows_);
                size_type j_end = std::min(jj + block_size, cols_);
                
                for (size_type i = ii; i < i_end; ++i) {
                    for (size_type j = jj; j < j_end; ++j) {
                        result(j, i) = (*this)(i, j);
                    }
                }
            }
        }
    }

    // SIMD implementations (platform-specific)
    void add_simd(const OptimizedMatrix& other);
    void subtract_simd(const OptimizedMatrix& other);
    void scale_simd(const T& scalar);
    void multiply_vector_simd(const OptimizedVector<T>& vec, OptimizedVector<T>& result) const;
    void hadamard_simd(const OptimizedMatrix& other);
    T sum_simd() const;
    T norm_simd() const;
};

// Type aliases for common use cases
using OptimizedMatrixF = OptimizedMatrix<float>;
using OptimizedMatrixD = OptimizedMatrix<double>;
using OptimizedMatrixI = OptimizedMatrix<int>;

// Binary arithmetic operators (compatible with BasicMatrix)
template<typename T>
OptimizedMatrix<T> operator+(const OptimizedMatrix<T>& lhs, const OptimizedMatrix<T>& rhs) {
    OptimizedMatrix<T> result = lhs;
    result += rhs;
    return result;
}

template<typename T>
OptimizedMatrix<T> operator-(const OptimizedMatrix<T>& lhs, const OptimizedMatrix<T>& rhs) {
    OptimizedMatrix<T> result = lhs;
    result -= rhs;
    return result;
}

template<typename T>
OptimizedMatrix<T> operator*(const OptimizedMatrix<T>& matrix, const T& scalar) {
    OptimizedMatrix<T> result = matrix;
    result *= scalar;
    return result;
}

template<typename T>
OptimizedMatrix<T> operator*(const T& scalar, const OptimizedMatrix<T>& matrix) {
    return matrix * scalar;
}

template<typename T>
OptimizedMatrix<T> operator/(const OptimizedMatrix<T>& matrix, const T& scalar) {
    OptimizedMatrix<T> result = matrix;
    result /= scalar;
    return result;
}

// Stream output operator
template<typename T>
std::ostream& operator<<(std::ostream& os, const OptimizedMatrix<T>& matrix) {
    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        os << "[";
        for (std::size_t j = 0; j < matrix.cols(); ++j) {
            os << std::setw(10) << std::setprecision(6) << matrix(i, j);
            if (j < matrix.cols() - 1) os << ", ";
        }
        os << "]";
        if (i < matrix.rows() - 1) os << "\n";
    }
    return os;
}

// Linear algebra functions for uBLAS compatibility

/// Get a row from a matrix (compatible with boost::numeric::ublas::row)
template<typename T>
OptimizedVector<T> row(const OptimizedMatrix<T>& matrix, typename OptimizedMatrix<T>::size_type row_index) {
    return matrix.row(row_index);
}

/// Get a column from a matrix (compatible with boost::numeric::ublas::column)
template<typename T>
OptimizedVector<T> column(const OptimizedMatrix<T>& matrix, typename OptimizedMatrix<T>::size_type col_index) {
    return matrix.column(col_index);
}

/// Element-wise multiplication (Hadamard product)
template<typename T>
OptimizedMatrix<T> element_prod(const OptimizedMatrix<T>& lhs, const OptimizedMatrix<T>& rhs) {
    OptimizedMatrix<T> result = lhs;
    result.hadamard_multiply(rhs);
    return result;
}

/// Matrix transpose function (uBLAS compatibility)
template<typename T>
OptimizedMatrix<T> trans(const OptimizedMatrix<T>& matrix) {
    return matrix.transpose();
}

/// Matrix-vector product (uBLAS compatibility)
template<typename T>
OptimizedVector<T> prod(const OptimizedMatrix<T>& matrix, const OptimizedVector<T>& vector) {
    return matrix.multiply_vector(vector);
}

/// Matrix-matrix product (uBLAS compatibility)
template<typename T>
OptimizedMatrix<T> prod(const OptimizedMatrix<T>& lhs, const OptimizedMatrix<T>& rhs) {
    return lhs.multiply(rhs);
}

// Factory function with type deduction
template<typename T>
auto make_optimized_matrix(std::size_t rows, std::size_t cols, const T& init_value = T{}) {
    return OptimizedMatrix<T>(rows, cols, init_value);
}

} // namespace libhmm

#endif // LIBHMM_OPTIMIZED_MATRIX_H_
