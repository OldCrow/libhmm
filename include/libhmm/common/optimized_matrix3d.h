#ifndef OPTIMIZED_MATRIX3D_H_
#define OPTIMIZED_MATRIX3D_H_

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <numeric>

// Use robust performance infrastructure (includes parallel execution detection)
#include "libhmm/performance/simd_support.h"
#include "libhmm/performance/parallel_constants.h"
#include "common.h"
#include "basic_matrix3d.h"  // Forward declaration for conversion constructor

namespace libhmm {

/// High-performance 3D matrix with flat memory layout for optimal cache performance
/// Uses row-major ordering: data[i][j][k] = flat_data[i*y_*z_ + j*z_ + k]
template<typename T>
class OptimizedMatrix3D
{
private:
    std::vector<T> data_;           ///< Flat contiguous storage
    std::size_t x_, y_, z_;         ///< Dimensions
    std::size_t yz_stride_;         ///< Pre-computed y*z for faster indexing
    
    /// Convert 3D indices to flat index
    /// @param i First dimension index
    /// @param j Second dimension index  
    /// @param k Third dimension index
    /// @return Flat array index
    constexpr std::size_t flatten_index(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return i * yz_stride_ + j * z_ + k;
    }

public:
    /// Constructor with dimensions and optional initial value
    /// @param x First dimension size
    /// @param y Second dimension size
    /// @param z Third dimension size
    /// @param init_value Initial value for all elements (default: zero)
    /// @throws std::invalid_argument if any dimension is zero
    OptimizedMatrix3D(std::size_t x, std::size_t y, std::size_t z, const T& init_value = T{0})
        : x_{x}, y_{y}, z_{z}, yz_stride_{y * z}
    {
        if (x == 0 || y == 0 || z == 0) {
            throw std::invalid_argument("OptimizedMatrix3D dimensions must be greater than zero");
        }
        
        const std::size_t total_size = x * y * z;
        
        // Check for overflow
        if (total_size / x / y != z) {
            throw std::invalid_argument("OptimizedMatrix3D dimensions would cause overflow");
        }
        
        data_.assign(total_size, init_value);
    }
    
    /// Conversion constructor from BasicMatrix3D (enables dynamic upgrading)
    explicit OptimizedMatrix3D(const BasicMatrix3D<T>& basic_matrix3d)
        : x_{basic_matrix3d.getXDimensionSize()}, 
          y_{basic_matrix3d.getYDimensionSize()}, 
          z_{basic_matrix3d.getZDimensionSize()},
          yz_stride_{y_ * z_} {
        
        const std::size_t total_size = x_ * y_ * z_;
        data_.resize(total_size);
        
        // Copy data from BasicMatrix3D to enable seamless transition to optimized operations
        for (std::size_t i = 0; i < x_; ++i) {
            for (std::size_t j = 0; j < y_; ++j) {
                for (std::size_t k = 0; k < z_; ++k) {
                    data_[flatten_index(i, j, k)] = basic_matrix3d(i, j, k);
                }
            }
        }
    }

    /// Default destructor
    ~OptimizedMatrix3D() = default;

    /// Copy constructor  
    OptimizedMatrix3D(const OptimizedMatrix3D&) = default;
    
    /// Move constructor
    OptimizedMatrix3D(OptimizedMatrix3D&&) noexcept = default;
    
    /// Copy assignment
    OptimizedMatrix3D& operator=(const OptimizedMatrix3D&) = default;
    
    /// Move assignment
    OptimizedMatrix3D& operator=(OptimizedMatrix3D&&) noexcept = default;

    /// Fast unchecked element access (for performance-critical code)
    /// @param i First dimension index
    /// @param j Second dimension index
    /// @param k Third dimension index
    /// @return Reference to the element
    T& operator()(std::size_t i, std::size_t j, std::size_t k) noexcept {
        return data_[flatten_index(i, j, k)];
    }

    /// Fast unchecked const element access
    /// @param i First dimension index
    /// @param j Second dimension index
    /// @param k Third dimension index
    /// @return Const reference to the element
    const T& operator()(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return data_[flatten_index(i, j, k)];
    }

    /// Bounds-checked element access (for safety when needed)
    /// @param i First dimension index
    /// @param j Second dimension index
    /// @param k Third dimension index
    /// @return Reference to the element
    /// @throws std::out_of_range if indices are invalid
    T& at(std::size_t i, std::size_t j, std::size_t k) {
        if (i >= x_ || j >= y_ || k >= z_) {
            throw std::out_of_range("OptimizedMatrix3D index out of bounds");
        }
        return data_[flatten_index(i, j, k)];
    }

    /// Bounds-checked const element access
    /// @param i First dimension index
    /// @param j Second dimension index
    /// @param k Third dimension index
    /// @return Const reference to the element
    /// @throws std::out_of_range if indices are invalid
    const T& at(std::size_t i, std::size_t j, std::size_t k) const {
        if (i >= x_ || j >= y_ || k >= z_) {
            throw std::out_of_range("OptimizedMatrix3D index out of bounds");
        }
        return data_[flatten_index(i, j, k)];
    }

    /// Set element value with bounds checking
    /// @param i First dimension index
    /// @param j Second dimension index
    /// @param k Third dimension index
    /// @param value Value to set
    /// @throws std::out_of_range if indices are invalid
    void Set(std::size_t i, std::size_t j, std::size_t k, const T& value) {
        at(i, j, k) = value;
    }

    /// Get first dimension size
    /// @return Size of first dimension
    constexpr std::size_t getXDimensionSize() const noexcept { return x_; }

    /// Get second dimension size
    /// @return Size of second dimension
    constexpr std::size_t getYDimensionSize() const noexcept { return y_; }

    /// Get third dimension size
    /// @return Size of third dimension
    constexpr std::size_t getZDimensionSize() const noexcept { return z_; }

    /// Get total number of elements
    /// @return Total element count
    constexpr std::size_t size() const noexcept { return data_.size(); }

    /// Check if matrix is empty
    /// @return True if empty
    constexpr bool empty() const noexcept { return data_.empty(); }

    /// Fill all elements with a value (serial version)
    /// @param value Value to fill with
    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    /// Fill all elements with a value (parallel version for large matrices)
    /// @param value Value to fill with
    void fill_parallel(const T& value) {
#if LIBHMM_HAS_PARALLEL_EXECUTION
        if (data_.size() > 10000) {  // Only use parallel for large matrices
            std::fill(std::execution::par_unseq, data_.begin(), data_.end(), value);
        } else {
            std::fill(data_.begin(), data_.end(), value);
        }
#else
        std::fill(data_.begin(), data_.end(), value);
#endif
    }

    /// Clear all elements to zero
    void clear() {
        fill(T{0});
    }

    /// Get raw data pointer (for C-style APIs or advanced optimizations)
    /// @return Pointer to underlying data
    T* data() noexcept { return data_.data(); }

    /// Get const raw data pointer
    /// @return Const pointer to underlying data
    const T* data() const noexcept { return data_.data(); }

    /// Get a 2D slice at fixed first dimension
    /// Returns a view-like object that allows 2D access: slice[j][k]
    class Matrix2DSlice {
    private:
        T* data_ptr_;
        std::size_t y_, z_;
        
    public:
        Matrix2DSlice(T* data_ptr, std::size_t y, std::size_t z) 
            : data_ptr_(data_ptr), y_(y), z_(z) {}
            
        class RowSlice {
        private:
            T* row_ptr_;
            
        public:
            RowSlice(T* row_ptr) : row_ptr_(row_ptr) {}
            
            T& operator[](std::size_t k) noexcept { return row_ptr_[k]; }
            const T& operator[](std::size_t k) const noexcept { return row_ptr_[k]; }
        };
        
        RowSlice operator[](std::size_t j) noexcept {
            return RowSlice(data_ptr_ + j * z_);
        }
        
        T& operator()(std::size_t j, std::size_t k) noexcept {
            return data_ptr_[j * z_ + k];
        }
        
        const T& operator()(std::size_t j, std::size_t k) const noexcept {
            return data_ptr_[j * z_ + k];
        }
    };

    /// Get a 2D slice at fixed first dimension
    /// @param i First dimension index
    /// @return 2D slice object
    Matrix2DSlice slice(std::size_t i) noexcept {
        return Matrix2DSlice(data_.data() + i * yz_stride_, y_, z_);
    }

    /// Element-wise addition (parallel when beneficial)
    /// @param other Matrix to add
    /// @return Reference to this matrix
    /// @throws std::invalid_argument if dimensions don't match
    OptimizedMatrix3D& operator+=(const OptimizedMatrix3D& other) {
        if (x_ != other.x_ || y_ != other.y_ || z_ != other.z_) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }
        
#if LIBHMM_HAS_PARALLEL_EXECUTION
        if (data_.size() > 10000) {
            std::transform(std::execution::par_unseq, 
                          data_.begin(), data_.end(), 
                          other.data_.begin(), 
                          data_.begin(),
                          std::plus<T>());
        } else {
            std::transform(data_.begin(), data_.end(), 
                          other.data_.begin(), 
                          data_.begin(),
                          std::plus<T>());
        }
#else
        std::transform(data_.begin(), data_.end(), 
                      other.data_.begin(), 
                      data_.begin(),
                      std::plus<T>());
#endif
        return *this;
    }

    /// Element-wise multiplication by scalar
    /// @param scalar Scalar value to multiply by
    /// @return Reference to this matrix
    OptimizedMatrix3D& operator*=(const T& scalar) {
#if LIBHMM_HAS_PARALLEL_EXECUTION
        if (data_.size() > 10000) {
            std::transform(std::execution::par_unseq, 
                          data_.begin(), data_.end(), 
                          data_.begin(),
                          [scalar](const T& val) { return val * scalar; });
        } else {
            std::transform(data_.begin(), data_.end(), 
                          data_.begin(),
                          [scalar](const T& val) { return val * scalar; });
        }
#else
        std::transform(data_.begin(), data_.end(), 
                      data_.begin(),
                      [scalar](const T& val) { return val * scalar; });
#endif
        return *this;
    }

    /// Compute sum of all elements (parallel when beneficial)
    /// @return Sum of all elements
    T sum() const {
#if LIBHMM_HAS_PARALLEL_EXECUTION
        if (data_.size() > 10000) {
            return std::reduce(std::execution::par_unseq, data_.begin(), data_.end(), T{0});
        } else {
            return std::accumulate(data_.begin(), data_.end(), T{0});
        }
#else
        return std::accumulate(data_.begin(), data_.end(), T{0});
#endif
    }

    /// Legacy compatibility methods (deprecated but maintained for transition)
    [[deprecated("Use getXDimensionSize() instead")]]
    int GetXDimensionSize() const { return static_cast<int>(x_); }
    
    [[deprecated("Use getYDimensionSize() instead")]]
    int GetYDimensionSize() const { return static_cast<int>(y_); }
    
    [[deprecated("Use getZDimensionSize() instead")]]  
    int GetZDimensionSize() const { return static_cast<int>(z_); }

}; // class OptimizedMatrix3D

/// Factory function for creating OptimizedMatrix3D with type deduction
/// @param x First dimension size
/// @param y Second dimension size
/// @param z Third dimension size
/// @param init_value Initial value
/// @return New OptimizedMatrix3D instance
template<typename T>
auto make_matrix3d(std::size_t x, std::size_t y, std::size_t z, const T& init_value = T{0}) {
    return OptimizedMatrix3D<T>(x, y, z, init_value);
}

} // namespace libhmm

#endif // OPTIMIZED_MATRIX3D_H_
