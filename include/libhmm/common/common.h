#ifndef COMMON_H_
#define COMMON_H_

/*
 * Standard Library includes - C++17 only, no external dependencies
 * Consolidated to reduce duplicate includes across the project
 */

// Core C++ standard library headers
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <utility>
#include <type_traits>

// Mathematical and numerical headers
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <limits>

// Platform and system headers - SIMD support
// Only include x86 intrinsics on x86/x64 platforms to avoid Apple Silicon issues
#ifdef _MSC_VER
    #include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
    #include <immintrin.h>
    #include <x86intrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
#endif

namespace libhmm
{

/* 
 * If we make all observations as double, we can truncate as needed and still
 * retain tons of precision for most of the operations.
 */
typedef double Observation;

/* 
 * We know that a state is an integral value, so...
 */
typedef int StateIndex;

}//namespace

/*
 * Custom libhmm Matrix and Vector classes (C++17 standard library only)
 * Must be included after basic typedefs to avoid naming conflicts
 */
#include "libhmm/common/basic_matrix.h"
#include "libhmm/common/basic_vector.h"
#include "libhmm/common/basic_matrix3d.h"

namespace libhmm
{

/*
 * Type aliases - Replace boost::numeric::ublas types with custom implementations
 * These provide drop-in replacement for existing Boost uBLAS usage
 */
using Matrix = BasicMatrix<Observation>;
using Vector = BasicVector<Observation>;
using ObservationSet = BasicVector<Observation>;

/*
 * 3D Matrix type alias for HMM training algorithms
 * Provides efficient 3D matrix operations for xi and gamma calculations
 */
template<typename T>
using Matrix3DTemplate = BasicMatrix3D<T>;

// Convenient type alias for the most common case
using ObservationMatrix3D = BasicMatrix3D<Observation>;

/*
 * Viterbi decode requires a fixed size vector for state sequences
 */
using StateSequence = BasicVector<StateIndex>;

/*
 * Training requires creating a list of all the observation sets.
 * We can't make a Vector of Vectors, but we can use std::list or std::vector
 * for things like this.
 */
typedef std::vector<ObservationSet> ObservationLists;

//==============================================================================
// CONSOLIDATED CONSTANTS FOR LIBHMM
// 
// This section defines all numerical constants used throughout the library.
// Constants are organized by category and consistently named for easy reference.
// These replace scattered constant definitions throughout the codebase.
//==============================================================================

/// Core numerical precision constants
namespace constants {
    
    /// Basic precision and tolerance values
    namespace precision {
        /// Minimum value considered non-zero throughout libhmm
        /// Used as the effective zero threshold for numerical computations
        inline constexpr double ZERO = 1.0e-30;
        
        /// Default convergence tolerance for training algorithms
        /// Used by Baum-Welch, Viterbi training, and gamma functions
        inline constexpr double DEFAULT_CONVERGENCE_TOLERANCE = 1.0e-8;
        
        /// Legacy Baum-Welch tolerance for backward compatibility
        /// Note: New code should use DEFAULT_CONVERGENCE_TOLERANCE
        inline constexpr double BW_TOLERANCE = 3.0e-7;
        
        /// Tolerance for continuous distribution range validation
        /// Used to determine valid probability ranges in continuous distributions
        inline constexpr double LIMIT_TOLERANCE = 1.0e-6;
        
        /// High precision tolerance for critical numerical operations
        /// Used in SIMD calculations and numerical stability checks
        inline constexpr double HIGH_PRECISION_TOLERANCE = 1.0e-12;
        
        /// Ultra-high precision for research and validation purposes
        /// Used when maximum numerical accuracy is required
        inline constexpr double ULTRA_HIGH_PRECISION_TOLERANCE = 1.0e-15;
        
        /// Minimum standard deviation for distribution parameters
        /// Prevents degenerate distributions and numerical instability
        inline constexpr double MIN_STD_DEV = 1.0e-6;
    }
    
    /// Probability bounds and safety limits
    namespace probability {
        /// Minimum probability value to prevent underflow
        /// Used throughout probability calculations to maintain numerical stability
        inline constexpr double MIN_PROBABILITY = 1.0e-300;
        
        /// Maximum probability value to prevent overflow
        /// Slightly less than 1.0 to avoid floating-point edge cases
        inline constexpr double MAX_PROBABILITY = 1.0 - 1.0e-15;
        
        /// Minimum log probability to prevent -infinity
        /// Approximately log(MIN_PROBABILITY)
        inline constexpr double MIN_LOG_PROBABILITY = -700.0;
        
        /// Maximum log probability (log(1.0))
        inline constexpr double MAX_LOG_PROBABILITY = 0.0;
        
        /// Threshold for scaling operations to prevent underflow
        /// Used in scaled Forward-Backward algorithms
        inline constexpr double SCALING_THRESHOLD = 1.0e-100;
        
        /// Log-space threshold for scaling operations
        /// When log probabilities fall below this, scaling is applied
        inline constexpr double LOG_SCALING_THRESHOLD = -230.0; // Approximately log(1e-100)
    }
    
    /// Iteration limits for training algorithms
    namespace iterations {
        /// Maximum iterations for Viterbi training
        /// Conservative limit for robust convergence
        inline constexpr std::size_t MAX_VITERBI_ITERATIONS = 500;
        
        /// Maximum iterations for Baum-Welch training
        /// Higher limit due to algorithm characteristics
        inline constexpr std::size_t MAX_BAUM_WELCH_ITERATIONS = 1000;
        
        /// Maximum iterations for gamma-related functions
        /// Used in statistical distribution computations
        inline constexpr std::size_t ITMAX = 10000;
        
        /// Default maximum iterations for general numerical algorithms
        /// Used by convergence detectors and adaptive algorithms
        inline constexpr std::size_t DEFAULT_MAX_ITERATIONS = 10000;
        
        /// Maximum iterations for real-time applications
        /// Reduced limit for time-critical operations
        inline constexpr std::size_t REALTIME_MAX_ITERATIONS = 100;
    }
    
    /// SIMD optimization parameters
    namespace simd {
        /// Default SIMD block size for vectorized operations
        /// Optimized for most modern processors (SSE2/AVX)
        inline constexpr std::size_t DEFAULT_BLOCK_SIZE = 8;
        
        /// Minimum problem size to benefit from SIMD
        /// Below this threshold, scalar code may be faster
        inline constexpr std::size_t MIN_SIMD_SIZE = 4;
        
        /// Maximum block size for cache optimization
        /// Prevents cache thrashing in large matrix operations
        inline constexpr std::size_t MAX_BLOCK_SIZE = 64;
        
        /// SIMD alignment requirement (bytes)
        /// Required for efficient memory access
        inline constexpr std::size_t SIMD_ALIGNMENT = 32;
    }
    
    /// Mathematical constants
    namespace math {
        /// High-precision value of π
        /// Sufficient precision for all libhmm calculations
        inline constexpr double PI = 3.141592653589793238462643383279502884;
        
        /// Natural logarithm of 2
        /// Used in information-theoretic calculations
        inline constexpr double LN2 = 0.6931471805599453094172321214581766;
        
        /// Euler's number (e)
        /// Used in exponential calculations
        inline constexpr double E = 2.7182818284590452353602874713526625;
        
        /// Square root of 2π (used in Gaussian calculations)
        /// Precomputed for efficiency in normal distribution
        inline constexpr double SQRT_2PI = 2.5066282746310005024157652848110453;
        
        /// Natural logarithm of 2π (used in log-space Gaussian calculations)
        /// Precomputed for efficiency in log-space normal distribution
        inline constexpr double LN_2PI = 1.8378770664093454835606594728112353;
        
        /// Square root of 2 (used in Gaussian CDF calculations)
        /// Precomputed for efficiency in error function calculations
        inline constexpr double SQRT_2 = 1.4142135623730950488016887242096981;
        
        /// Half of ln(2π) (used in log-space Gaussian calculations)
        /// Precomputed for efficiency: 0.5 * ln(2π)
        inline constexpr double HALF_LN_2PI = 0.9189385332046727417803297364056176;
        
        /// Euler-Mascheroni constant (γ)
        /// Used in various statistical distributions and special functions
        inline constexpr double EULER_MASCHERONI = 0.5772156649015328606065120900824024;
        
        /// Golden ratio (φ = (1 + √5)/2)
        /// Occasionally used in optimization and special calculations
        inline constexpr double GOLDEN_RATIO = 1.6180339887498948482045868343656381;
        
        /// Commonly used fractional constants
        inline constexpr double HALF = 0.5;
        inline constexpr double QUARTER = 0.25;
        inline constexpr double THREE_QUARTERS = 0.75;
        
        /// Commonly used integer constants as doubles
        inline constexpr double ZERO_DOUBLE = 0.0;
        inline constexpr double ONE = 1.0;
        inline constexpr double TWO = 2.0;
        inline constexpr double THREE = 3.0;
        inline constexpr double FOUR = 4.0;
        inline constexpr double FIVE = 5.0;
        inline constexpr double TEN = 10.0;
        inline constexpr double HUNDRED = 100.0;
        inline constexpr double THOUSAND = 1000.0;
    }
    
    /// Algorithm-specific thresholds
    namespace thresholds {
        /// Minimum scale factor for scaled algorithms
        /// Prevents numerical instability in scaling operations
        inline constexpr double MIN_SCALE_FACTOR = 1.0e-100;
        
        /// Maximum scale factor for scaled algorithms
        /// Prevents overflow in scaling operations
        inline constexpr double MAX_SCALE_FACTOR = 1.0e100;
        
        /// Threshold for switching to log-space computation
        /// When probabilities fall below this, use log-space
        inline constexpr double LOG_SPACE_THRESHOLD = 1.0e-50;
        
        /// Convergence window size for iterative algorithms
        /// Number of iterations to look back for convergence detection
        inline constexpr std::size_t CONVERGENCE_WINDOW = 5;
        
        /// Minimum cluster size for K-means training
        /// Prevents degenerate clusters in segmented training
        inline constexpr std::size_t MIN_CLUSTER_SIZE = 1;
        
        /// Minimum degrees of freedom for Student's t-distribution
        /// Prevents numerical instability in t-distribution calculations
        inline constexpr double MIN_DEGREES_OF_FREEDOM = 0.1;
        
        /// Maximum degrees of freedom for Student's t-distribution
        /// Prevents overflow in t-distribution calculations
        inline constexpr double MAX_DEGREES_OF_FREEDOM = 1000.0;
        
        /// Maximum parameter value for distribution fitting
        /// Used to prevent extreme parameter values that could cause numerical issues
        inline constexpr double MAX_DISTRIBUTION_PARAMETER = 1.0e6;
        
        /// Minimum parameter value for distribution fitting
        /// Used to prevent extremely small parameter values
        inline constexpr double MIN_DISTRIBUTION_PARAMETER = 1.0e-6;
    }
}

/// Legacy constants for backward compatibility
/// These maintain the original names while pointing to the new consolidated values
inline constexpr double BW_TOLERANCE = constants::precision::BW_TOLERANCE;
inline constexpr double ZERO = constants::precision::ZERO;
inline constexpr double LIMIT_TOLERANCE = constants::precision::LIMIT_TOLERANCE;
inline constexpr std::size_t MAX_VITERBI_ITERATIONS = constants::iterations::MAX_VITERBI_ITERATIONS;
inline constexpr std::size_t ITMAX = constants::iterations::ITMAX;
inline constexpr double PI = constants::math::PI;
 
// Custom Matrix and Vector classes initialize to zero on construction,
// but provide explicit clear functions for API compatibility
void clear_matrix( Matrix& m );
void clear_vector( Vector& v );
void clear_vector( StateSequence& v );

}//namespace

/*
 * Custom C++17 serialization functions for matrices and vectors.
 * These replace the Boost serialization functionality with lightweight,
 * standards-compliant serialization using simple XML format.
 * 
 * Design Goals:
 * - Zero external dependencies (pure C++17)
 * - Backward compatibility with existing XML format where possible
 * - Simple, human-readable XML output
 * - Efficient parsing for HMM-specific use cases
 */
namespace libhmm {
namespace serialization {

/**
 * Simple XML serialization for BasicMatrix objects
 * Replaces boost::serialization with lightweight implementation
 */
template<typename T>
class MatrixSerializer {
public:
    // Save matrix to XML format
    static void save(std::ostream& os, const BasicMatrix<T>& matrix, const std::string& name = "matrix") {
        os << "<" << name << ">\n";
        os << "  <rows>" << matrix.size1() << "</rows>\n";
        os << "  <cols>" << matrix.size2() << "</cols>\n";
        os << "  <data>\n";
        
        for (std::size_t i = 0; i < matrix.size1(); ++i) {
            os << "    <row>";
            for (std::size_t j = 0; j < matrix.size2(); ++j) {
                os << matrix(i, j);
                if (j < matrix.size2() - 1) os << " ";
            }
            os << "</row>\n";
        }
        
        os << "  </data>\n";
        os << "</" << name << ">\n";
    }
    
    // Load matrix from XML format
    static void load(std::istream& is, BasicMatrix<T>& matrix, const std::string& name = "matrix") {
        std::string line;
        std::size_t rows = 0, cols = 0;
        
        // Simple XML parsing - find opening tag
        while (std::getline(is, line)) {
            if (line.find("<" + name + ">") != std::string::npos) {
                break;
            }
        }
        
        // Read dimensions
        if (std::getline(is, line)) {
            std::size_t start = line.find("<rows>") + 6;
            std::size_t end = line.find("</rows>");
            if (start != std::string::npos && end != std::string::npos) {
                rows = std::stoull(line.substr(start, end - start));
            }
        }
        
        if (std::getline(is, line)) {
            std::size_t start = line.find("<cols>") + 6;
            std::size_t end = line.find("</cols>");
            if (start != std::string::npos && end != std::string::npos) {
                cols = std::stoull(line.substr(start, end - start));
            }
        }
        
        // Resize matrix
        matrix.resize(rows, cols);
        
        // Skip <data> tag
        std::getline(is, line);
        
        // Read matrix data
        for (std::size_t i = 0; i < rows; ++i) {
            if (std::getline(is, line)) {
                std::size_t start = line.find("<row>") + 5;
                std::size_t end = line.find("</row>");
                if (start != std::string::npos && end != std::string::npos) {
                    std::string data_str = line.substr(start, end - start);
                    std::istringstream row_stream(data_str);
                    
                    for (std::size_t j = 0; j < cols; ++j) {
                        T value;
                        row_stream >> value;
                        matrix(i, j) = value;
                    }
                }
            }
        }
    }
};

/**
 * Simple XML serialization for BasicVector objects
 * Replaces boost::serialization with lightweight implementation
 */
template<typename T>
class VectorSerializer {
public:
    // Save vector to XML format
    static void save(std::ostream& os, const BasicVector<T>& vector, const std::string& name = "vector") {
        os << "<" << name << ">\n";
        os << "  <size>" << vector.size() << "</size>\n";
        os << "  <data>";
        
        for (std::size_t i = 0; i < vector.size(); ++i) {
            os << vector[i];
            if (i < vector.size() - 1) os << " ";
        }
        
        os << "</data>\n";
        os << "</" << name << ">\n";
    }
    
    // Load vector from XML format  
    static void load(std::istream& is, BasicVector<T>& vector, const std::string& name = "vector") {
        std::string line;
        std::size_t size = 0;
        
        // Simple XML parsing - find opening tag
        while (std::getline(is, line)) {
            if (line.find("<" + name + ">") != std::string::npos) {
                break;
            }
        }
        
        // Read size
        if (std::getline(is, line)) {
            std::size_t start = line.find("<size>") + 6;
            std::size_t end = line.find("</size>");
            if (start != std::string::npos && end != std::string::npos) {
                size = std::stoull(line.substr(start, end - start));
            }
        }
        
        // Resize vector
        vector.resize(size);
        
        // Read data
        if (std::getline(is, line)) {
            std::size_t start = line.find("<data>") + 6;
            std::size_t end = line.find("</data>");
            if (start != std::string::npos && end != std::string::npos) {
                std::string data_str = line.substr(start, end - start);
                std::istringstream data_stream(data_str);
                
                for (std::size_t i = 0; i < size; ++i) {
                    T value;
                    data_stream >> value;
                    vector[i] = value;
                }
            }
        }
    }
};

/**
 * Convenience functions to match the old boost::serialization style
 */

// Matrix serialization convenience functions
template<typename Archive, typename T>
void save(Archive& ar, const BasicMatrix<T>& matrix, const std::string& name = "matrix") {
    MatrixSerializer<T>::save(ar, matrix, name);
}

template<typename Archive, typename T>
void load(Archive& ar, BasicMatrix<T>& matrix, const std::string& name = "matrix") {
    MatrixSerializer<T>::load(ar, matrix, name);
}

// Vector serialization convenience functions
template<typename Archive, typename T>
void save(Archive& ar, const BasicVector<T>& vector, const std::string& name = "vector") {
    VectorSerializer<T>::save(ar, vector, name);
}

template<typename Archive, typename T>
void load(Archive& ar, BasicVector<T>& vector, const std::string& name = "vector") {
    VectorSerializer<T>::load(ar, vector, name);
}

} // namespace serialization
} // namespace libhmm

#endif
