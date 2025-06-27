#ifndef BASIC_MATRIX3D_H_
#define BASIC_MATRIX3D_H_

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace libhmm{

// Forward declaration for BasicMatrix
template<typename T> class BasicMatrix;

/// Improved 3D matrix template class for HMM training algorithms
/// Uses contiguous memory layout for better cache performance
/// Provides type-safe, bounds-checked access to 3D data structures
template<typename T>
class BasicMatrix3D
{
private:
    std::vector<T> data_;           // Flat contiguous storage
    std::size_t x_, y_, z_;         // Dimensions
    std::size_t yz_stride_;         // Pre-computed y*z for faster indexing
    
    /// Convert 3D indices to 1D flat index
    /// Uses row-major order: [i][j][k] -> i*(y*z) + j*z + k
    constexpr std::size_t index(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return i * yz_stride_ + j * z_ + k;
    }

public:
    /// Constructor with dimensions
    /// @param x First dimension size
    /// @param y Second dimension size  
    /// @param z Third dimension size
    /// @throws std::invalid_argument if any dimension is zero
    BasicMatrix3D(std::size_t x, std::size_t y, std::size_t z)
        : x_{x}, y_{y}, z_{z}, yz_stride_{y * z} {
        if (x == 0 || y == 0 || z == 0) {
            throw std::invalid_argument("BasicMatrix3D dimensions must be greater than zero");
        }
        
        // Check for potential overflow
        if (yz_stride_ < y || yz_stride_ < z) {
            throw std::invalid_argument("BasicMatrix3D dimensions too large (overflow)");
        }
        
        std::size_t total_size = x * yz_stride_;
        if (total_size < x || total_size < yz_stride_) {
            throw std::invalid_argument("BasicMatrix3D total size too large (overflow)");
        }
        
        // Initialize contiguous storage with zero-fill
        data_.resize(total_size, T{0});
    }
    
    /// Constructor with dimensions and initial value
    /// @param x First dimension size
    /// @param y Second dimension size  
    /// @param z Third dimension size
    /// @param init_value Initial value for all elements
    /// @throws std::invalid_argument if any dimension is zero
    BasicMatrix3D(std::size_t x, std::size_t y, std::size_t z, const T& init_value)
        : x_{x}, y_{y}, z_{z}, yz_stride_{y * z} {
        if (x == 0 || y == 0 || z == 0) {
            throw std::invalid_argument("BasicMatrix3D dimensions must be greater than zero");
        }
        
        // Check for potential overflow
        if (yz_stride_ < y || yz_stride_ < z) {
            throw std::invalid_argument("BasicMatrix3D dimensions too large (overflow)");
        }
        
        std::size_t total_size = x * yz_stride_;
        if (total_size < x || total_size < yz_stride_) {
            throw std::invalid_argument("BasicMatrix3D total size too large (overflow)");
        }
        
        // Initialize contiguous storage with specified value
        data_.resize(total_size, init_value);
    }

    /// Default destructor
    ~BasicMatrix3D() = default;

    /// Copy constructor
    BasicMatrix3D(const BasicMatrix3D&) = default;
    
    /// Move constructor
    BasicMatrix3D(BasicMatrix3D&&) = default;
    
    /// Copy assignment
    BasicMatrix3D& operator=(const BasicMatrix3D&) = default;
    
    /// Move assignment
    BasicMatrix3D& operator=(BasicMatrix3D&&) = default;

    /// Bounds-checked element access
    /// @param i First dimension index
    /// @param j Second dimension index
    /// @param k Third dimension index
    /// @return Reference to the element
    /// @throws std::out_of_range if indices are invalid
    T& operator()(std::size_t i, std::size_t j, std::size_t k) {
        if (i >= x_ || j >= y_ || k >= z_) {
            throw std::out_of_range("BasicMatrix3D index out of bounds");
        }
        return data_[index(i, j, k)];
    }

    /// Const bounds-checked element access
    /// @param i First dimension index
    /// @param j Second dimension index
    /// @param k Third dimension index
    /// @return Const reference to the element
    /// @throws std::out_of_range if indices are invalid
    const T& operator()(std::size_t i, std::size_t j, std::size_t k) const {
        if (i >= x_ || j >= y_ || k >= z_) {
            throw std::out_of_range("BasicMatrix3D index out of bounds");
        }
        return data_[index(i, j, k)];
    }

    /// Set element value with bounds checking
    /// @param i First dimension index
    /// @param j Second dimension index
    /// @param k Third dimension index
    /// @param value Value to set
    /// @throws std::out_of_range if indices are invalid
    void Set(std::size_t i, std::size_t j, std::size_t k, const T& value) {
        if (i >= x_ || j >= y_ || k >= z_) {
            throw std::out_of_range("BasicMatrix3D index out of bounds");
        }
        data_[index(i, j, k)] = value;
    }

    /// Get first dimension size
    /// @return Size of first dimension
    std::size_t getXDimensionSize() const noexcept { return x_; }

    /// Get second dimension size
    /// @return Size of second dimension
    std::size_t getYDimensionSize() const noexcept { return y_; }

    /// Get third dimension size
    /// @return Size of third dimension
    std::size_t getZDimensionSize() const noexcept { return z_; }

    /// Fill all elements with a value
    /// @param value Value to fill with
    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    /// Clear all elements to zero
    void clear() {
        fill(T{0});
    }
    
    /// Get total number of elements
    /// @return Total size (x * y * z)
    std::size_t size() const noexcept {
        return data_.size();
    }
    
    /// Get raw pointer to underlying data (for interoperability)
    /// @return Pointer to contiguous data array
    T* data() noexcept {
        return data_.data();
    }
    
    /// Get const raw pointer to underlying data
    /// @return Const pointer to contiguous data array
    const T* data() const noexcept {
        return data_.data();
    }
    
    /// Check if matrix is empty
    /// @return True if any dimension is 0
    bool empty() const noexcept {
        return data_.empty();
    }
    
    /// Sum of all elements in the 3D matrix
    /// @return Sum of all elements
    T sum() const {
        return std::accumulate(data_.begin(), data_.end(), T{});
    }
    
    /// Get a 2D slice from the 3D matrix (fixes first dimension)
    /// @param i First dimension index (fixed)
    /// @return 2D matrix representing the slice
    template<typename MatrixType = BasicMatrix<T>>
    MatrixType slice(std::size_t i) const {
        if (i >= x_) {
            throw std::out_of_range("Slice index out of bounds");
        }
        MatrixType result(y_, z_);
        for (std::size_t j = 0; j < y_; ++j) {
            for (std::size_t k = 0; k < z_; ++k) {
                result(j, k) = (*this)(i, j, k);
            }
        }
        return result;
    }
    
    /// Set a 2D slice in the 3D matrix (fixes first dimension)
    /// @param i First dimension index (fixed)
    /// @param slice_matrix 2D matrix to copy into the slice
    template<typename MatrixType>
    void set_slice(std::size_t i, const MatrixType& slice_matrix) {
        if (i >= x_) {
            throw std::out_of_range("Slice index out of bounds");
        }
        if (slice_matrix.rows() != y_ || slice_matrix.cols() != z_) {
            throw std::invalid_argument("Slice matrix dimensions must match y and z dimensions");
        }
        for (std::size_t j = 0; j < y_; ++j) {
            for (std::size_t k = 0; k < z_; ++k) {
                (*this)(i, j, k) = slice_matrix(j, k);
            }
        }
    }
    
    /// Element-wise multiplication with another 3D matrix
    /// @param other Matrix to multiply with
    /// @return Reference to this matrix
    BasicMatrix3D& element_multiply(const BasicMatrix3D& other) {
        if (x_ != other.x_ || y_ != other.y_ || z_ != other.z_) {
            throw std::invalid_argument("BasicMatrix3D dimensions must match for element-wise multiplication");
        }
        for (std::size_t i = 0; i < data_.size(); ++i) {
            data_[i] *= other.data_[i];
        }
        return *this;
    }
    
    /// Element-wise addition with another 3D matrix
    /// @param other Matrix to add
    /// @return Reference to this matrix
    BasicMatrix3D& operator+=(const BasicMatrix3D& other) {
        if (x_ != other.x_ || y_ != other.y_ || z_ != other.z_) {
            throw std::invalid_argument("BasicMatrix3D dimensions must match for addition");
        }
        for (std::size_t i = 0; i < data_.size(); ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }
    
    /// Element-wise subtraction with another 3D matrix
    /// @param other Matrix to subtract
    /// @return Reference to this matrix
    BasicMatrix3D& operator-=(const BasicMatrix3D& other) {
        if (x_ != other.x_ || y_ != other.y_ || z_ != other.z_) {
            throw std::invalid_argument("BasicMatrix3D dimensions must match for subtraction");
        }
        for (std::size_t i = 0; i < data_.size(); ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }
    
    /// Scalar multiplication
    /// @param scalar Value to multiply all elements by
    /// @return Reference to this matrix
    BasicMatrix3D& operator*=(const T& scalar) {
        for (auto& element : data_) {
            element *= scalar;
        }
        return *this;
    }
    
    /// Scalar division
    /// @param scalar Value to divide all elements by
    /// @return Reference to this matrix
    BasicMatrix3D& operator/=(const T& scalar) {
        for (auto& element : data_) {
            element /= scalar;
        }
        return *this;
    }

    // Legacy compatibility methods (deprecated)
    [[deprecated("Use getXDimensionSize() instead")]]
    int GetXDimensionSize() { return static_cast<int>(x_); }
    
    [[deprecated("Use getYDimensionSize() instead")]]
    int GetYDimensionSize() { return static_cast<int>(y_); }
    
    [[deprecated("Use getZDimensionSize() instead")]]
    int GetZDimensionSize() { return static_cast<int>(z_); }

};  //end BasicMatrix3D template

} //namespace

#endif // BASIC_MATRIX3D_H_



