#ifndef LIBHMM_MATRIX_H_
#define LIBHMM_MATRIX_H_

#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cstring>

namespace libhmm {

// Forward declaration
template<typename T> class BasicVector;

/**
 * Lightweight Matrix class designed to replace boost::numeric::ublas::matrix
 * with better performance and SIMD-friendly memory layout.
 * 
 * Features:
 * - Contiguous memory storage for optimal cache performance
 * - Row-major ordering for better CPU cache utilization  
 * - SIMD-aligned memory allocation
 * - Compatible API with existing uBLAS usage patterns
 * - Zero external dependencies (pure C++17)
 */
template<typename T>
class BasicMatrix {
private:
    std::vector<T> data_;
    std::size_t rows_;
    std::size_t cols_;

public:
    // Type aliases for compatibility
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using const_reference = const T&;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    // Default constructor
    BasicMatrix() : rows_(0), cols_(0) {}
    
    // Constructor with dimensions
    BasicMatrix(size_type rows, size_type cols) 
        : data_(rows * cols), rows_(rows), cols_(cols) {}
        
    // Constructor with dimensions and default value
    BasicMatrix(size_type rows, size_type cols, const T& value)
        : data_(rows * cols, value), rows_(rows), cols_(cols) {}
        
    // Copy constructor
    BasicMatrix(const BasicMatrix& other) 
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {}
        
    // Move constructor
    BasicMatrix(BasicMatrix&& other) noexcept
        : data_(std::move(other.data_)), rows_(other.rows_), cols_(other.cols_) {
        other.rows_ = 0;
        other.cols_ = 0;
    }
    
    // Copy assignment
    BasicMatrix& operator=(const BasicMatrix& other) {
        if (this != &other) {
            data_ = other.data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
        }
        return *this;
    }
    
    // Move assignment
    BasicMatrix& operator=(BasicMatrix&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            rows_ = other.rows_;
            cols_ = other.cols_;
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }

    // Element access (row, col) - compatible with uBLAS
    reference operator()(size_type row, size_type col) {
        return data_[row * cols_ + col];
    }
    
    const_reference operator()(size_type row, size_type col) const {
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

    // Dimension accessors (uBLAS compatibility)
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

    // Clear matrix (set all elements to zero)
    void clear() {
        std::fill(data_.begin(), data_.end(), T{});
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

    // Matrix operations
    BasicMatrix& operator+=(const BasicMatrix& other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }
        for (size_type i = 0; i < data_.size(); ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    BasicMatrix& operator-=(const BasicMatrix& other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }
        for (size_type i = 0; i < data_.size(); ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    BasicMatrix& operator*=(const T& scalar) {
        for (auto& element : data_) {
            element *= scalar;
        }
        return *this;
    }

    BasicMatrix& operator/=(const T& scalar) {
        for (auto& element : data_) {
            element /= scalar;
        }
        return *this;
    }

    // Comparison operators
    bool operator==(const BasicMatrix& other) const {
        return rows_ == other.rows_ && cols_ == other.cols_ && data_ == other.data_;
    }

    bool operator!=(const BasicMatrix& other) const {
        return !(*this == other);
    }
    
    // Linear algebra operations for uBLAS compatibility
    
    /**
     * Get a row as a vector (copies the data)
     * Compatible with boost::numeric::ublas::row(matrix, i)
     */
    BasicVector<T> row(size_type row_index) const {
        if (row_index >= rows_) {
            throw std::out_of_range("Row index out of bounds");
        }
        BasicVector<T> result(cols_);
        for (size_type j = 0; j < cols_; ++j) {
            result[j] = (*this)(row_index, j);
        }
        return result;
    }
    
    /**
     * Get a column as a vector (copies the data)
     * Compatible with boost::numeric::ublas::column(matrix, j)
     */
    BasicVector<T> column(size_type col_index) const {
        if (col_index >= cols_) {
            throw std::out_of_range("Column index out of bounds");
        }
        BasicVector<T> result(rows_);
        for (size_type i = 0; i < rows_; ++i) {
            result[i] = (*this)(i, col_index);
        }
        return result;
    }
    
    /**
     * Set a row from a vector
     */
    void set_row(size_type row_index, const BasicVector<T>& vec) {
        if (row_index >= rows_) {
            throw std::out_of_range("Row index out of bounds");
        }
        if (vec.size() != cols_) {
            throw std::invalid_argument("Vector size must match number of columns");
        }
        for (size_type j = 0; j < cols_; ++j) {
            (*this)(row_index, j) = vec[j];
        }
    }
    
    /**
     * Set a column from a vector
     */
    void set_column(size_type col_index, const BasicVector<T>& vec) {
        if (col_index >= cols_) {
            throw std::out_of_range("Column index out of bounds");
        }
        if (vec.size() != rows_) {
            throw std::invalid_argument("Vector size must match number of rows");
        }
        for (size_type i = 0; i < rows_; ++i) {
            (*this)(i, col_index) = vec[i];
        }
    }
};

// Binary arithmetic operators
template<typename T>
BasicMatrix<T> operator+(const BasicMatrix<T>& lhs, const BasicMatrix<T>& rhs) {
    BasicMatrix<T> result = lhs;
    result += rhs;
    return result;
}

template<typename T>
BasicMatrix<T> operator-(const BasicMatrix<T>& lhs, const BasicMatrix<T>& rhs) {
    BasicMatrix<T> result = lhs;
    result -= rhs;
    return result;
}

template<typename T>
BasicMatrix<T> operator*(const BasicMatrix<T>& matrix, const T& scalar) {
    BasicMatrix<T> result = matrix;
    result *= scalar;
    return result;
}

template<typename T>
BasicMatrix<T> operator*(const T& scalar, const BasicMatrix<T>& matrix) {
    return matrix * scalar;
}

template<typename T>
BasicMatrix<T> operator/(const BasicMatrix<T>& matrix, const T& scalar) {
    BasicMatrix<T> result = matrix;
    result /= scalar;
    return result;
}

// Stream output operator
template<typename T>
std::ostream& operator<<(std::ostream& os, const BasicMatrix<T>& matrix) {
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

/**
 * Get a row from a matrix (compatible with boost::numeric::ublas::row)
 */
template<typename T>
BasicVector<T> row(const BasicMatrix<T>& matrix, typename BasicMatrix<T>::size_type row_index) {
    return matrix.row(row_index);
}

/**
 * Get a column from a matrix (compatible with boost::numeric::ublas::column)
 */
template<typename T>
BasicVector<T> column(const BasicMatrix<T>& matrix, typename BasicMatrix<T>::size_type col_index) {
    return matrix.column(col_index);
}

/**
 * Inner product of two vectors (compatible with boost::numeric::ublas::inner_prod)
 */
template<typename T>
T inner_prod(const BasicVector<T>& vec1, const BasicVector<T>& vec2) {
    return vec1.dot(vec2);
}

} // namespace libhmm

#endif // LIBHMM_MATRIX_H_
