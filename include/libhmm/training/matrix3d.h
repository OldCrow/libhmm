#ifndef MATRIX3D_H_
#define MATRIX3D_H_

#include <vector>
#include <stdexcept>
#include <algorithm>

namespace libhmm{

/// Modern 3D matrix template class for HMM training algorithms
/// Provides type-safe, bounds-checked access to 3D data structures
template<typename T>
class Matrix3D
{
private:
    std::vector<std::vector<std::vector<T>>> matrix_;
    std::size_t x_, y_, z_;

public:
    /// Constructor with dimensions
    /// @param x First dimension size
    /// @param y Second dimension size  
    /// @param z Third dimension size
    /// @throws std::invalid_argument if any dimension is zero
    Matrix3D(std::size_t x, std::size_t y, std::size_t z)
        : x_{x}, y_{y}, z_{z} {
        if (x == 0 || y == 0 || z == 0) {
            throw std::invalid_argument("Matrix3D dimensions must be greater than zero");
        }
        
        // Initialize with proper sizing and zero-fill
        matrix_.resize(x_);
        for (auto& plane : matrix_) {
            plane.resize(y_);
            for (auto& row : plane) {
                row.resize(z_, T{0});
            }
        }
    }

    /// Default destructor
    ~Matrix3D() = default;

    /// Copy constructor
    Matrix3D(const Matrix3D&) = default;
    
    /// Move constructor
    Matrix3D(Matrix3D&&) = default;
    
    /// Copy assignment
    Matrix3D& operator=(const Matrix3D&) = default;
    
    /// Move assignment
    Matrix3D& operator=(Matrix3D&&) = default;

    /// Bounds-checked element access
    /// @param i First dimension index
    /// @param j Second dimension index
    /// @param k Third dimension index
    /// @return Reference to the element
    /// @throws std::out_of_range if indices are invalid
    T& operator()(std::size_t i, std::size_t j, std::size_t k) {
        if (i >= x_ || j >= y_ || k >= z_) {
            throw std::out_of_range("Matrix3D index out of bounds");
        }
        return matrix_[i][j][k];
    }

    /// Const bounds-checked element access
    /// @param i First dimension index
    /// @param j Second dimension index
    /// @param k Third dimension index
    /// @return Const reference to the element
    /// @throws std::out_of_range if indices are invalid
    const T& operator()(std::size_t i, std::size_t j, std::size_t k) const {
        if (i >= x_ || j >= y_ || k >= z_) {
            throw std::out_of_range("Matrix3D index out of bounds");
        }
        return matrix_[i][j][k];
    }

    /// Set element value with bounds checking
    /// @param i First dimension index
    /// @param j Second dimension index
    /// @param k Third dimension index
    /// @param value Value to set
    /// @throws std::out_of_range if indices are invalid
    void Set(std::size_t i, std::size_t j, std::size_t k, const T& value) {
        if (i >= x_ || j >= y_ || k >= z_) {
            throw std::out_of_range("Matrix3D index out of bounds");
        }
        matrix_[i][j][k] = value;
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
        for (auto& plane : matrix_) {
            for (auto& row : plane) {
                std::fill(row.begin(), row.end(), value);
            }
        }
    }

    /// Clear all elements to zero
    void clear() {
        fill(T{0});
    }

    // Legacy compatibility methods (deprecated)
    [[deprecated("Use getXDimensionSize() instead")]]
    int GetXDimensionSize() { return static_cast<int>(x_); }
    
    [[deprecated("Use getYDimensionSize() instead")]]
    int GetYDimensionSize() { return static_cast<int>(y_); }
    
    [[deprecated("Use getZDimensionSize() instead")]]
    int GetZDimensionSize() { return static_cast<int>(z_); }

};  //end Matrix3D template

} //namespace

#endif // MATRIX3D_H_



