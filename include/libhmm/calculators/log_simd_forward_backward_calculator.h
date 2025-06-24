#ifndef LIBHMM_LOG_SIMD_FORWARD_BACKWARD_CALCULATOR_H_
#define LIBHMM_LOG_SIMD_FORWARD_BACKWARD_CALCULATOR_H_

#include "libhmm/calculators/log_forward_backward_calculator.h"
#include "libhmm/performance/simd_support.h"
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

namespace libhmm {

/// High-performance log-space forward-backward calculator using SIMD optimizations
/// Combines numerical stability of log-space arithmetic with SIMD performance
/// This addresses the critical gap where OptimizedForwardBackwardCalculator lacks numerical stability
class LogSIMDForwardBackwardCalculator : public LogForwardBackwardCalculator {
private:
    /// Aligned storage for SIMD operations
    using AlignedVector = std::vector<double, performance::aligned_allocator<double>>;
    
    /// Log-space constants for SIMD operations
    static constexpr double LOGZERO = std::numeric_limits<double>::quiet_NaN();
    static constexpr double LOG_MIN_PROBABILITY = -700.0;  // Prevents -infinity
    
    /// Cached aligned matrices for efficient SIMD operations
    mutable std::unique_ptr<AlignedVector> alignedForward_;
    mutable std::unique_ptr<AlignedVector> alignedBackward_;
    mutable std::unique_ptr<AlignedVector> alignedLogTrans_;
    mutable std::unique_ptr<AlignedVector> alignedLogPi_;
    
    /// Matrix dimensions for cache efficiency
    std::size_t numStates_;
    std::size_t obsSize_;
    std::size_t alignedStateSize_;  // Padded to SIMD alignment
    
    /// Performance optimization flags
    bool useBlockedComputation_;
    std::size_t blockSize_;
    
    /// Initialize aligned storage and copy data for SIMD operations
    void initializeAlignedStorage();
    
    /// Copy matrix data to aligned storage with padding, converting to log-space
    void copyToAlignedLogStorage(const Matrix& source, AlignedVector& dest, 
                                 std::size_t rows, std::size_t cols, std::size_t alignedCols);
    
    /// Copy aligned log-space storage back to boost matrix
    void copyFromAlignedLogStorage(const AlignedVector& source, Matrix& dest,
                                   std::size_t rows, std::size_t cols, std::size_t alignedCols);
    
    /// SIMD-optimized log-space emission probability computation
    void computeLogEmissionProbabilities(std::size_t t, AlignedVector& logEmissions) const;
    
    /// SIMD-optimized extended logarithm function (vectorized eln)
    /// Handles zero values by converting to LOGZERO
    void vectorizedEln(const double* input, double* output, std::size_t size) const;
    
    /// SIMD-optimized extended logarithm sum (vectorized elnsum)
    /// log(exp(x) + exp(y)) with numerical stability
    void vectorizedElnSum(const double* x, const double* y, double* result, std::size_t size) const;
    
    /// SIMD-optimized extended logarithm product (vectorized elnproduct)
    /// Simply x + y in log space
    void vectorizedElnProduct(const double* x, const double* y, double* result, std::size_t size) const;
    
    /// SIMD-optimized log-space matrix-vector multiplication with elnsum
    /// Computes result[j] = elnsum_i(trans_log[i,j] + vector_log[i])
    void logMatrixVectorMultiplyTransposed(const double* logMatrix, const double* logVector,
                                           double* logResult, std::size_t rows, std::size_t cols) const;
    
    /// Blocked log-space matrix-vector multiplication for large matrices
    void blockedLogMatrixVectorMultiplyTransposed(const double* logMatrix, const double* logVector,
                                                  double* logResult, std::size_t rows, std::size_t cols,
                                                  std::size_t blockSize) const;
    
    /// SIMD-optimized standard log-space matrix-vector multiplication
    void logMatrixVectorMultiply(const double* logMatrix, const double* logVector,
                                 double* logResult, std::size_t rows, std::size_t cols) const;
    
    /// Blocked version of standard log-space matrix-vector multiplication
    void blockedLogMatrixVectorMultiply(const double* logMatrix, const double* logVector,
                                        double* logResult, std::size_t rows, std::size_t cols,
                                        std::size_t blockSize) const;

protected:
    /// SIMD-optimized log-space forward algorithm implementation
    void forward() override;
    
    /// SIMD-optimized log-space backward algorithm implementation  
    void backward() override;

public:
    /// Constructor with HMM and observations
    /// @param hmm Pointer to the HMM (must not be null)
    /// @param observations The observation set to process
    /// @param useBlocking Enable blocked computation for large matrices
    /// @param blockSize Block size for cache optimization (0 = auto-detect)
    /// @throws std::invalid_argument if hmm is null
    LogSIMDForwardBackwardCalculator(Hmm* hmm, const ObservationSet& observations,
                                     bool useBlocking = true, std::size_t blockSize = 0);
    
    /// Virtual destructor
    virtual ~LogSIMDForwardBackwardCalculator() = default;
    
    /// Get performance information
    /// @return String describing optimizations used
    std::string getOptimizationInfo() const;
    
    /// Check if SIMD optimizations are available
    /// @return True if SIMD is available and being used
    bool isSIMDOptimized() const noexcept {
        return performance::simd_available();
    }
    
    /// Get recommended block size for this system
    /// @param numStates Number of HMM states
    /// @return Optimal block size for cache efficiency
    static std::size_t getRecommendedBlockSize(std::size_t numStates) noexcept;
    
    /// Calculate log probability with SIMD optimization
    /// @return The log probability value
    double logProbability() override;
};

/// Helper class for SIMD log-space operations
class SIMDLogOps {
public:
    /// SIMD-optimized elnsum operation
    /// Computes log(exp(x) + exp(y)) for arrays
    static void elnsum_array(const double* x, const double* y, double* result, std::size_t size);
    
    /// SIMD-optimized elnproduct operation
    /// Computes x + y for log-space arrays
    static void elnproduct_array(const double* x, const double* y, double* result, std::size_t size);
    
    /// SIMD-optimized eln operation
    /// Converts linear to log-space for arrays
    static void eln_array(const double* x, double* result, std::size_t size);
    
    /// SIMD-optimized reduction sum in log space
    /// Computes elnsum of all elements in array
    static double elnsum_reduce(const double* array, std::size_t size);
    
    /// Check if a value is LOGZERO (NaN in our implementation)
    static bool is_logzero(double x) noexcept {
        return std::isnan(x);
    }
    
    /// Safe log-space value clamping
    static void clamp_logspace(double* array, std::size_t size) noexcept;
};

} // namespace libhmm

#endif // LIBHMM_LOG_SIMD_FORWARD_BACKWARD_CALCULATOR_H_
