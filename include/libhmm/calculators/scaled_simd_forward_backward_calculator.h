#ifndef LIBHMM_SCALED_SIMD_FORWARD_BACKWARD_CALCULATOR_H_
#define LIBHMM_SCALED_SIMD_FORWARD_BACKWARD_CALCULATOR_H_

#include "libhmm/hmm.h"
#include "libhmm/calculators/calculator.h"
#include "libhmm/performance/simd_support.h"
#include "libhmm/common/optimized_matrix.h"
#include <cfloat>
#include <cmath>
#include <vector>
#include <memory>
// SIMD headers are now conditionally included in common.h based on platform

namespace libhmm {

/**
 * @brief SIMD-optimized scaled Forward-Backward algorithm implementation
 * 
 * This calculator combines Rabiner-style scaling for numerical stability
 * with SIMD vectorization for high performance. The scaling prevents
 * underflow issues while SIMD acceleration provides substantial speed
 * improvements for the critical inner loops.
 * 
 * Key features:
 * - Numerical stability through scaling factors
 * - SIMD vectorization of probability computations
 * - Cache-friendly memory access patterns
 * - Automatic fallback to scalar implementation
 * 
 * The algorithm implements the scaled Forward-Backward approach where
 * forward and backward variables are scaled at each time step to prevent
 * underflow, and the final probability is reconstructed from the scaling factors.
 */
class ScaledSIMDForwardBackwardCalculator : public Calculator {
private:
    // Core matrices (aligned for SIMD)
    std::vector<double, performance::aligned_allocator<double>> forwardVariables_;
    std::vector<double, performance::aligned_allocator<double>> backwardVariables_;
    std::vector<double> scalingFactors_;
    
    // Results
    double probability_;
    double logProbability_;
    
    // Problem dimensions
    std::size_t numStates_;
    std::size_t seqLength_;
    
    // SIMD optimization parameters (using consolidated constants)
    static constexpr std::size_t SIMD_BLOCK_SIZE = constants::simd::DEFAULT_BLOCK_SIZE;
    static constexpr double SCALING_THRESHOLD = constants::probability::SCALING_THRESHOLD;
    static constexpr double MIN_SCALE_FACTOR = constants::thresholds::MIN_SCALE_FACTOR;
    static constexpr double MAX_SCALE_FACTOR = constants::thresholds::MAX_SCALE_FACTOR;
    
    // Temporary SIMD-aligned vectors for computations
    mutable std::vector<double, performance::aligned_allocator<double>> tempEmissions_;
    mutable std::vector<double, performance::aligned_allocator<double>> tempProbs_;
    
    // Performance optimization flags
    bool useBlockedComputation_;
    std::size_t blockSize_;
    std::size_t alignedStateSize_;  // Padded to SIMD alignment

public:
    /**
     * @brief Constructor with HMM and observations
     * @param hmm Pointer to the HMM (must not be null)
     * @param observations The observation set to process
     * @param useBlocking Enable blocked computation for large matrices
     * @param blockSize Block size for cache optimization (0 = auto-detect)
     * @throws std::invalid_argument if hmm is null
     */
    ScaledSIMDForwardBackwardCalculator(Hmm* hmm, const ObservationSet& observations,
                                        bool useBlocking = true, std::size_t blockSize = 0);
    
    /**
     * @brief Compute forward and backward variables using scaled SIMD algorithm
     * 
     * Performs the complete Forward-Backward algorithm with numerical scaling
     * and SIMD acceleration. Computes both forward and backward variables.
     * 
     * @throws std::runtime_error if computation fails
     */
    void compute();
    
    /**
     * @brief Get the probability of the observation sequence
     * 
     * Returns the probability of the observation sequence given the HMM,
     * properly reconstructed from the scaling factors.
     * 
     * @return The probability value
     */
    double getProbability() const noexcept {
        return probability_;
    }
    
    /**
     * @brief Get the log probability of the observation sequence
     * 
     * Returns the log probability of the observation sequence,
     * reconstructed from the scaling factors.
     * 
     * @return The log probability value
     */
    double getLogProbability() const noexcept {
        return logProbability_;
    }
    
    /**
     * @brief Calculate probability (required by traits system)
     * 
     * This method is required for compatibility with the calculator
     * traits system and automatic selection.
     * 
     * @return The probability value
     */
    double probability() {
        return getProbability();
    }
    
    /**
     * @brief Get the forward variables matrix
     * @return The forward variables as an OptimizedMatrix
     */
    OptimizedMatrix<double> getForwardVariables() const;
    
    /**
     * @brief Get the backward variables matrix
     * @return The backward variables as an OptimizedMatrix
     */
    OptimizedMatrix<double> getBackwardVariables() const;
    
    /**
     * @brief Get the forward variables as basic Matrix (for compatibility)
     * @return The forward variables as a Matrix
     */
    Matrix getForwardVariablesCompat() const;
    
    /**
     * @brief Get the backward variables as basic Matrix (for compatibility)
     * @return The backward variables as a Matrix
     */
    Matrix getBackwardVariablesCompat() const;
    
    /**
     * @brief Get the scaling factors used during computation
     * 
     * Useful for debugging and understanding numerical behavior.
     * 
     * @return Vector of scaling factors for each time step
     */
    std::vector<double> getScalingFactors() const {
        return scalingFactors_;
    }
    
    /**
     * @brief Check if SIMD optimization is being used
     * @return True if SIMD is available and being used
     */
    static bool isSIMDEnabled() noexcept {
        return performance::simd_available();
    }
    
    /**
     * @brief Get performance information
     * @return String describing optimizations used
     */
    std::string getOptimizationInfo() const;
    
    /**
     * @brief Get recommended block size for this system
     * @param numStates Number of HMM states
     * @return Optimal block size for cache efficiency
     */
    static std::size_t getRecommendedBlockSize(std::size_t numStates) noexcept;

private:
    /**
     * @brief Initialize matrices and prepare for computation
     */
    void initializeMatrices();
    
    /**
     * @brief Perform SIMD-optimized forward pass
     */
    void computeForward();
    
    /**
     * @brief Perform SIMD-optimized backward pass
     */
    void computeBackward();
    
    /**
     * @brief Initialize forward variables with first observation
     */
    void initializeForwardStep();
    
    /**
     * @brief Perform SIMD-optimized forward step at time t
     * @param t Time step index
     */
    void computeForwardStepSIMD(std::size_t t);
    
    /**
     * @brief Fallback scalar computation for forward step
     * @param t Time step index
     */
    void computeForwardStepScalar(std::size_t t);
    
    /**
     * @brief Initialize backward variables
     */
    void initializeBackwardStep();
    
    /**
     * @brief Perform SIMD-optimized backward step at time t
     * @param t Time step index
     */
    void computeBackwardStepSIMD(std::size_t t);
    
    /**
     * @brief Fallback scalar computation for backward step
     * @param t Time step index
     */
    void computeBackwardStepScalar(std::size_t t);
    
    /**
     * @brief Apply scaling to variables at time step t
     * @param t Time step index
     * @param variables Pointer to variables to scale
     * @return Scaling factor applied
     */
    double applyScaling(std::size_t t, double* variables);
    
    /**
     * @brief Reconstruct probability from scaling factors
     */
    void reconstructProbability();
    
    /**
     * @brief SIMD-optimized computation of emission probabilities
     * @param observation Current observation
     * @param emisProbs Output emission probabilities (aligned)
     */
    void computeEmissionProbabilities(Observation observation, double* emisProbs) const;
    
    /**
     * @brief SIMD-optimized vector operations
     */
    void simdVectorMultiply(const double* a, const double* b, double* result, std::size_t size) const;
    void simdVectorAdd(const double* a, const double* b, double* result, std::size_t size) const;
    void simdVectorScale(const double* input, double scale, double* result, std::size_t size) const;
    double simdVectorSum(const double* vector, std::size_t size) const;
    
    /**
     * @brief SIMD-optimized matrix-vector multiplication
     */
    void simdMatrixVectorMultiply(const double* matrix, const double* vector,
                                  double* result, std::size_t rows, std::size_t cols) const;
    
    /**
     * @brief SIMD-optimized transposed matrix-vector multiplication
     */
    void simdMatrixVectorMultiplyTransposed(const double* matrix, const double* vector,
                                            double* result, std::size_t rows, std::size_t cols) const;
    
    /**
     * @brief Check if variables need scaling
     * @param t Time step index
     * @param variables Pointer to variables to check
     * @return True if scaling is needed
     */
    bool needsScaling(std::size_t t, const double* variables) const;
    
    /**
     * @brief Get matrix index for forward/backward matrices
     * @param t Time step
     * @param state State index
     * @return Linear index in matrix
     */
    std::size_t getMatrixIndex(std::size_t t, std::size_t state) const {
        return t * numStates_ + state;
    }
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
