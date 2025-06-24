#ifndef LIBHMM_SCALED_SIMD_FORWARD_BACKWARD_CALCULATOR_H_
#define LIBHMM_SCALED_SIMD_FORWARD_BACKWARD_CALCULATOR_H_

#include "libhmm/calculators/scaled_forward_backward_calculator.h"
#include "libhmm/performance/simd_support.h"
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <immintrin.h>

namespace libhmm {

/// High-performance scaled forward-backward calculator using SIMD optimizations
/// Combines numerical stability of Rabiner's scaling with true SIMD vectorization
/// This follows HMMLib's approach of vectorizing scaled arithmetic rather than log arithmetic
class ScaledSIMDForwardBackwardCalculator : public ScaledForwardBackwardCalculator {
private:
    /// Aligned storage for SIMD operations
    using AlignedVector = std::vector<double, performance::aligned_allocator<double>>;
    
    /// Scaling constants for SIMD operations
    static constexpr double MIN_SCALE_FACTOR = 1e-100;
    static constexpr double MAX_SCALE_FACTOR = 1e100;
    
    /// Cached aligned matrices for efficient SIMD operations
    mutable std::unique_ptr<AlignedVector> alignedForward_;
    mutable std::unique_ptr<AlignedVector> alignedBackward_;
    mutable std::unique_ptr<AlignedVector> alignedTrans_;
    mutable std::unique_ptr<AlignedVector> alignedPi_;
    mutable std::unique_ptr<AlignedVector> alignedScales_;
    
    /// Matrix dimensions for cache efficiency
    std::size_t numStates_;
    std::size_t obsSize_;
    std::size_t alignedStateSize_;  // Padded to SIMD alignment (multiple of 4 for AVX)
    
    /// Performance optimization flags
    bool useBlockedComputation_;
    std::size_t blockSize_;
    
    /// Initialize aligned storage and copy data for SIMD operations
    void initializeAlignedStorage();
    
    /// Copy matrix data to aligned storage with padding
    void copyToAlignedStorage(const Matrix& source, AlignedVector& dest, 
                              std::size_t rows, std::size_t cols, std::size_t alignedCols);
    
    /// Copy aligned storage back to boost matrix
    void copyFromAlignedStorage(const AlignedVector& source, Matrix& dest,
                                std::size_t rows, std::size_t cols, std::size_t alignedCols);
    
    /// SIMD-optimized emission probability computation
    void computeEmissionProbabilities(std::size_t t, AlignedVector& emissions) const;
    
    /// SIMD-optimized vector operations
    void simdVectorMultiply(const double* a, const double* b, double* result, std::size_t size) const;
    void simdVectorAdd(const double* a, const double* b, double* result, std::size_t size) const;
    void simdVectorScale(const double* input, double scale, double* result, std::size_t size) const;
    double simdVectorSum(const double* vector, std::size_t size) const;
    
    /// SIMD-optimized transposed matrix-vector multiplication
    /// Computes result[j] = sum_i(matrix[i * cols + j] * vector[i])
    void simdMatrixVectorMultiplyTransposed(const double* matrix, const double* vector,
                                            double* result, std::size_t rows, std::size_t cols) const;
    
    /// Blocked transposed matrix-vector multiplication for large matrices
    void blockedMatrixVectorMultiplyTransposed(const double* matrix, const double* vector,
                                               double* result, std::size_t rows, std::size_t cols,
                                               std::size_t blockSize) const;
    
    /// SIMD-optimized standard matrix-vector multiplication
    void simdMatrixVectorMultiply(const double* matrix, const double* vector,
                                  double* result, std::size_t rows, std::size_t cols) const;
    
    /// Blocked version of standard matrix-vector multiplication
    void blockedMatrixVectorMultiply(const double* matrix, const double* vector,
                                     double* result, std::size_t rows, std::size_t cols,
                                     std::size_t blockSize) const;
    
    /// SIMD-optimized scaling operations
    double computeScalingFactor(const double* array, std::size_t size) const;
    void applyScaling(double* array, double scaleFactor, std::size_t size) const;

protected:
    /// SIMD-optimized scaled forward algorithm implementation
    void forward() override;
    
    /// SIMD-optimized scaled backward algorithm implementation  
    void backward() override;

public:
    /// Constructor with HMM and observations
    /// @param hmm Pointer to the HMM (must not be null)
    /// @param observations The observation set to process
    /// @param useBlocking Enable blocked computation for large matrices
    /// @param blockSize Block size for cache optimization (0 = auto-detect)
    /// @throws std::invalid_argument if hmm is null
    ScaledSIMDForwardBackwardCalculator(Hmm* hmm, const ObservationSet& observations,
                                        bool useBlocking = true, std::size_t blockSize = 0);
    
    /// Virtual destructor
    virtual ~ScaledSIMDForwardBackwardCalculator() = default;
    
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
    
    /// Get scaling factors
    /// @return Vector of scaling factors used in computation
    Vector getScalingFactors() const;
};

/// Helper class for SIMD scaled operations
class SIMDScaledOps {
public:
    /// SIMD-optimized element-wise multiplication
    /// Uses AVX when available, falls back to SSE2
    static void multiply_arrays(const double* a, const double* b, double* result, std::size_t size);
    
    /// SIMD-optimized element-wise addition
    static void add_arrays(const double* a, const double* b, double* result, std::size_t size);
    
    /// SIMD-optimized scaling (multiply by scalar)
    static void scale_array(const double* input, double scale, double* result, std::size_t size);
    
    /// SIMD-optimized horizontal sum (reduction)
    static double sum_array(const double* array, std::size_t size);
    
    /// SIMD-optimized dot product
    static double dot_product(const double* a, const double* b, std::size_t size);
    
    /// SIMD-optimized maximum element finding
    static double max_element(const double* array, std::size_t size);
    
    /// Check CPU capabilities for SIMD optimization selection
    static bool hasAVX() noexcept;
    static bool hasSSE2() noexcept;
    
private:
    /// AVX implementation (when available)
    static void multiply_arrays_avx(const double* a, const double* b, double* result, std::size_t size);
    static void add_arrays_avx(const double* a, const double* b, double* result, std::size_t size);
    static void scale_array_avx(const double* input, double scale, double* result, std::size_t size);
    static double sum_array_avx(const double* array, std::size_t size);
    
    /// SSE2 implementation (fallback)
    static void multiply_arrays_sse2(const double* a, const double* b, double* result, std::size_t size);
    static void add_arrays_sse2(const double* a, const double* b, double* result, std::size_t size);
    static void scale_array_sse2(const double* input, double scale, double* result, std::size_t size);
    static double sum_array_sse2(const double* array, std::size_t size);
    
    /// Scalar implementation (ultimate fallback)
    static void multiply_arrays_scalar(const double* a, const double* b, double* result, std::size_t size);
    static void add_arrays_scalar(const double* a, const double* b, double* result, std::size_t size);
    static void scale_array_scalar(const double* input, double scale, double* result, std::size_t size);
    static double sum_array_scalar(const double* array, std::size_t size);
};

} // namespace libhmm

#endif // LIBHMM_SCALED_SIMD_FORWARD_BACKWARD_CALCULATOR_H_
