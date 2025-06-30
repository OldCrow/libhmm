#ifndef LIBHMM_LINEAR_ALGEBRA_H_
#define LIBHMM_LINEAR_ALGEBRA_H_

/**
 * @file linear_algebra.h
 * @brief Complete linear algebra header for libhmm
 * 
 * This header provides access to both basic and optimized linear algebra classes.
 * It's designed as a convenience header for users who want the full set of
 * linear algebra capabilities without needing to include individual headers.
 * 
 * @section usage Usage Examples
 * @code
 * #include "libhmm/common/linear_algebra.h"
 * 
 * using namespace libhmm;
 * 
 * // Basic classes (always available)
 * BasicVector<double> basic_vec(1000, 1.0);
 * BasicMatrix<double> basic_mat(100, 100, 0.5);
 * BasicMatrix3D<double> basic_3d(10, 10, 10);
 * 
 * // Optimized classes (for performance-critical code)
 * OptimizedVector<double> opt_vec(1000, 1.0);
 * OptimizedMatrix<double> opt_mat(100, 100, 0.5);
 * OptimizedMatrix3D<double> opt_3d(10, 10, 10);
 * 
 * // Seamless conversion from basic to optimized
 * OptimizedVector<double> upgraded_vec(basic_vec);
 * OptimizedMatrix<double> upgraded_mat(basic_mat);
 * OptimizedMatrix3D<double> upgraded_3d(basic_3d);
 * @endcode
 * 
 * @section design Design Philosophy
 * - **Basic classes**: Simple, safe, API-compatible implementations
 * - **Optimized classes**: High-performance SIMD and parallel implementations  
 * - **Conversion constructors**: Enable dynamic performance scaling
 * - **Unified API**: Both families provide identical functionality
 * 
 * @section performance Performance Guidelines
 * - Use basic classes for small data (\< 100 elements) or development/debugging
 * - Use optimized classes for large data (\> 100 elements) or production performance
 * - Conversion constructors allow upgrading basic to optimized at runtime
 * 
 * @version 2.8.0
 */

//==============================================================================
// BASIC LINEAR ALGEBRA CLASSES
//==============================================================================

/// Fundamental linear algebra classes (always included via common.h)
/// These are already included, but we document them here for completeness:
/// - BasicVector<T>: Simple vector operations with bounds checking
/// - BasicMatrix<T>: Simple matrix operations with row-major layout  
/// - BasicMatrix3D<T>: Simple 3D tensor operations with contiguous memory

//==============================================================================
// OPTIMIZED LINEAR ALGEBRA CLASSES  
//==============================================================================

/// High-performance vector class with SIMD optimizations
#include "libhmm/common/optimized_vector.h"

/// High-performance matrix class with cache-blocking and SIMD
#include "libhmm/common/optimized_matrix.h"

/// High-performance 3D matrix class with parallel execution
#include "libhmm/common/optimized_matrix3d.h"

//==============================================================================
// CONVENIENCE TYPE ALIASES
//==============================================================================

namespace libhmm {

/// Type aliases for common optimized linear algebra usage patterns
namespace opt {
    /// Common optimized types for numerical computing
    using VectorD = OptimizedVector<double>;
    using VectorF = OptimizedVector<float>;
    using VectorI = OptimizedVector<int>;
    
    using MatrixD = OptimizedMatrix<double>;
    using MatrixF = OptimizedMatrix<float>;
    using MatrixI = OptimizedMatrix<int>;
    
    using Matrix3DD = OptimizedMatrix3D<double>;
    using Matrix3DF = OptimizedMatrix3D<float>;
    
    /// HMM-specific optimized types
    using ObservationVector = OptimizedVector<Observation>;
    using ObservationMatrix = OptimizedMatrix<Observation>;
    using ObservationMatrix3D = OptimizedMatrix3D<Observation>;
    using OptimizedStateSequence = OptimizedVector<StateIndex>;
}

/// Type aliases for basic linear algebra (these are already in common.h)
namespace basic {
    /// Common basic types for development and small data
    using VectorD = BasicVector<double>;
    using VectorF = BasicVector<float>;
    using VectorI = BasicVector<int>;
    
    using MatrixD = BasicMatrix<double>;
    using MatrixF = BasicMatrix<float>;
    using MatrixI = BasicMatrix<int>;
    
    using Matrix3DD = BasicMatrix3D<double>;
    using Matrix3DF = BasicMatrix3D<float>;
}

//==============================================================================
// FACTORY FUNCTIONS FOR TYPE DEDUCTION
//==============================================================================

/// Factory functions that choose appropriate class based on size hints
namespace factory {
    
    /// Create vector with automatic basic/optimized selection based on size
    template<typename T>
    auto make_vector(std::size_t size, const T& init_value = T{}, bool prefer_optimized = true) {
        if (prefer_optimized || size > 100) {
            return OptimizedVector<T>(size, init_value);
        } else {
            return BasicVector<T>(size, init_value);
        }
    }
    
    /// Create matrix with automatic basic/optimized selection based on size
    template<typename T>
    auto make_matrix(std::size_t rows, std::size_t cols, const T& init_value = T{}, bool prefer_optimized = true) {
        if (prefer_optimized || (rows * cols) > 1000) {
            return OptimizedMatrix<T>(rows, cols, init_value);
        } else {
            return BasicMatrix<T>(rows, cols, init_value);
        }
    }
    
    /// Create 3D matrix with automatic basic/optimized selection based on size
    template<typename T>
    auto make_matrix3d(std::size_t x, std::size_t y, std::size_t z, const T& init_value = T{}, bool prefer_optimized = true) {
        if (prefer_optimized || (x * y * z) > 1000) {
            return OptimizedMatrix3D<T>(x, y, z, init_value);
        } else {
            return BasicMatrix3D<T>(x, y, z, init_value);
        }
    }
}

//==============================================================================
// PERFORMANCE UPGRADE UTILITIES
//==============================================================================

/// Utilities for converting between basic and optimized classes
namespace upgrade {
    
    /// Convert basic vector to optimized (using conversion constructor)
    template<typename T>
    OptimizedVector<T> to_optimized(const BasicVector<T>& basic_vec) {
        return OptimizedVector<T>(basic_vec);
    }
    
    /// Convert basic matrix to optimized (using conversion constructor)
    template<typename T>
    OptimizedMatrix<T> to_optimized(const BasicMatrix<T>& basic_mat) {
        return OptimizedMatrix<T>(basic_mat);
    }
    
    /// Convert basic 3D matrix to optimized (using conversion constructor)
    template<typename T>
    OptimizedMatrix3D<T> to_optimized(const BasicMatrix3D<T>& basic_mat3d) {
        return OptimizedMatrix3D<T>(basic_mat3d);
    }
    
    /// Check if a problem size would benefit from optimized classes
    constexpr bool should_use_optimized(std::size_t total_elements) {
        return total_elements > 100;  // Threshold where optimizations typically help
    }
}

} // namespace libhmm

//==============================================================================
// FEATURE SUMMARY
//==============================================================================

/**
 * @brief Available linear algebra classes summary
 * 
 * **Basic Classes** (included via common.h):
 * - BasicVector<T>: Vector operations with bounds checking
 * - BasicMatrix<T>: Matrix operations with row-major layout
 * - BasicMatrix3D<T>: 3D tensor operations with contiguous memory
 * 
 * **Optimized Classes** (included via this header):
 * - OptimizedVector<T>: SIMD-optimized vector with parallel execution
 * - OptimizedMatrix<T>: Cache-blocked matrix with SIMD optimization
 * - OptimizedMatrix3D<T>: Parallel 3D tensor with contiguous memory
 * 
 * **Key Features**:
 * - Conversion constructors enable seamless basic -> optimized upgrades
 * - Identical APIs ensure drop-in compatibility
 * - Automatic optimization selection based on problem size
 * - Performance benefits for large data (> 100-1000 elements)
 * - Maintains numerical accuracy and stability
 */

#endif // LIBHMM_LINEAR_ALGEBRA_H_
