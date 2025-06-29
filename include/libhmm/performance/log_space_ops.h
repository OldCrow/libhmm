#ifndef LIBHMM_PERFORMANCE_LOG_SPACE_OPS_H_
#define LIBHMM_PERFORMANCE_LOG_SPACE_OPS_H_

#include "libhmm/common/common.h"
#include "libhmm/performance/simd_support.h"
#include <cmath>
#include <limits>
#include <array>

namespace libhmm {
namespace performance {

/**
 * @brief High-performance log-space arithmetic operations
 * 
 * This class provides optimized implementations of log-space arithmetic
 * operations commonly used in HMM calculations. Key optimizations include:
 * - Precomputed lookup tables for frequently used values
 * - SIMD-vectorized operations
 * - Numerically stable log-sum-exp implementations
 * - Efficient handling of log(0) cases
 */
class LogSpaceOps {
public:
    /// Log-space representation of zero (negative infinity)
    static constexpr double LOG_ZERO = constants::probability::MIN_LOG_PROBABILITY;
    
    /// Threshold below which exp() terms are considered negligible
    static constexpr double LOG_SUM_THRESHOLD = -50.0;
    
    /// Size of precomputed lookup tables
    static constexpr std::size_t LOOKUP_TABLE_SIZE = 1024;
    
    /**
     * @brief Initialize precomputed lookup tables
     * Call this once at program startup for optimal performance
     */
    static void initialize();
    
    /**
     * @brief Numerically stable log-sum-exp: log(exp(a) + exp(b))
     * 
     * Highly optimized version using lookup tables and avoiding
     * expensive exp/log operations when possible.
     * 
     * @param logA First log value
     * @param logB Second log value  
     * @return log(exp(logA) + exp(logB))
     */
    static double logSumExp(double logA, double logB) noexcept;
    
    /**
     * @brief Fast log-sum-exp for arrays using SIMD
     * 
     * @param logValues Array of log values
     * @param size Number of values
     * @return log(sum(exp(logValues[i])))
     */
    static double logSumExpArray(const double* logValues, std::size_t size) noexcept;
    
    /**
     * @brief Precompute log values for transition matrix
     * 
     * Converts probability matrix to log-space once and caches results.
     * Much faster than repeated log() calls during computation.
     * 
     * @param probMatrix Input probability matrix
     * @param logMatrix Output log matrix (must be pre-allocated)
     * @param rows Number of rows
     * @param cols Number of columns
     */
    static void precomputeLogMatrix(const double* probMatrix, double* logMatrix, 
                                   std::size_t rows, std::size_t cols) noexcept;
    
    /**
     * @brief SIMD-optimized log-space matrix-vector multiplication
     * 
     * Performs: result[i] = logSumExp_j(logMatrix[i*cols + j] + logVector[j])
     * 
     * @param logMatrix Log-space matrix (row-major)
     * @param logVector Log-space vector
     * @param result Output log-space vector
     * @param rows Number of matrix rows
     * @param cols Number of matrix columns
     */
    static void logMatrixVectorMultiply(const double* logMatrix, const double* logVector,
                                       double* result, std::size_t rows, std::size_t cols) noexcept;
    
    /**
     * @brief SIMD-optimized transposed log-space matrix-vector multiplication
     * 
     * Performs: result[j] = logSumExp_i(logMatrix[i*cols + j] + logVector[i])
     * 
     * @param logMatrix Log-space matrix (row-major)
     * @param logVector Log-space vector
     * @param result Output log-space vector
     * @param rows Number of matrix rows
     * @param cols Number of matrix columns
     */
    static void logMatrixVectorMultiplyTransposed(const double* logMatrix, const double* logVector,
                                                 double* result, std::size_t rows, std::size_t cols) noexcept;
    
    /**
     * @brief Check if log value represents zero (is LOG_ZERO or NaN)
     */
    static bool isLogZero(double logValue) noexcept {
        return std::isnan(logValue) || logValue <= LOG_ZERO;
    }
    
    /**
     * @brief Safe conversion from probability to log-space
     */
    static double safeLog(double prob) noexcept {
        return (prob > 0.0) ? std::log(prob) : LOG_ZERO;
    }

private:
    /// Precomputed lookup table for log(1 + exp(x)) for x in [-50, 0]
    static std::array<double, LOOKUP_TABLE_SIZE> logOnePlusExpTable_;
    static bool initialized_;
    
    /// Internal helper for lookup table access
    static double lookupLogOnePlusExp(double x) noexcept;
    
    /// SIMD implementations
    static double logSumExpArraySIMD(const double* logValues, std::size_t size) noexcept;
    static double logSumExpArrayScalar(const double* logValues, std::size_t size) noexcept;
};

/**
 * @brief RAII class to automatically initialize log-space operations
 * 
 * Create one instance of this at program startup to ensure
 * lookup tables are properly initialized.
 */
class LogSpaceInitializer {
public:
    LogSpaceInitializer() {
        LogSpaceOps::initialize();
    }
};

/// Global initializer - ensures tables are ready when library is loaded
static LogSpaceInitializer globalLogSpaceInit;

} // namespace performance
} // namespace libhmm

#endif // LIBHMM_PERFORMANCE_LOG_SPACE_OPS_H_
