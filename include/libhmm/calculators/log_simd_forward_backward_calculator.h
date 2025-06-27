#ifndef LIBHMM_LOG_SIMD_FORWARD_BACKWARD_CALCULATOR_H_
#define LIBHMM_LOG_SIMD_FORWARD_BACKWARD_CALCULATOR_H_

#include "libhmm/hmm.h"
#include "libhmm/calculators/calculator.h"
#include "libhmm/performance/simd_support.h"
#include "libhmm/common/optimized_matrix.h"
#include <cfloat>
#include <cmath>
#include <vector>
#include <memory>
#include <limits>

namespace libhmm {

/**
 * @brief SIMD-optimized log-space Forward-Backward algorithm implementation
 * 
 * This calculator combines log-space arithmetic for numerical stability
 * with SIMD vectorization for high performance. The log-space approach
 * prevents underflow issues while SIMD acceleration provides substantial
 * speed improvements for the critical inner loops.
 * 
 * Key features:
 * - Numerical stability through log-space arithmetic
 * - SIMD vectorization of probability computations
 * - Cache-friendly memory access patterns
 * - Automatic fallback to scalar implementation
 * 
 * The algorithm implements log-space Forward-Backward where all
 * probabilities are maintained in log space throughout computation,
 * eliminating the need for scaling factors.
 */
class LogSIMDForwardBackwardCalculator : public Calculator {
private:
    // Core matrices (aligned for SIMD)
    std::vector<double, performance::aligned_allocator<double>> logForwardVariables_;
    std::vector<double, performance::aligned_allocator<double>> logBackwardVariables_;
    
    // Results
    double logProbability_;
    
    // Problem dimensions
    std::size_t numStates_;
    std::size_t seqLength_;
    
    // SIMD optimization parameters (using consolidated constants)
    static constexpr std::size_t SIMD_BLOCK_SIZE = constants::simd::DEFAULT_BLOCK_SIZE;
    static constexpr double LOGZERO = -std::numeric_limits<double>::infinity();
    static constexpr double LOG_MIN_PROBABILITY = constants::probability::MIN_LOG_PROBABILITY;
    
    // Temporary SIMD-aligned vectors for computations
    mutable std::vector<double, performance::aligned_allocator<double>> tempLogEmissions_;
    mutable std::vector<double, performance::aligned_allocator<double>> tempLogProbs_;
    
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
    LogSIMDForwardBackwardCalculator(Hmm* hmm, const ObservationSet& observations,
                                     bool useBlocking = true, std::size_t blockSize = 0);
    
    /**
     * @brief Compute forward and backward variables using log-space SIMD algorithm
     * 
     * Performs the complete Forward-Backward algorithm in log space with
     * SIMD acceleration. Computes both forward and backward variables.
     * 
     * @throws std::runtime_error if computation fails
     */
    void compute();
    
    /**
     * @brief Get the log probability of the observation sequence
     * 
     * Returns the log probability of the observation sequence given the HMM.
     * 
     * @return The log probability value
     */
    double getLogProbability() const noexcept {
        return logProbability_;
    }
    
    /**
     * @brief Get the probability of the observation sequence
     * 
     * Returns the probability by exponentiating the log probability.
     * Note: This may underflow for very low probabilities.
     * 
     * @return The probability value
     */
    double getProbability() const noexcept {
        return std::exp(logProbability_);
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
     * @brief Get the log forward variables matrix
     * @return The log forward variables as an OptimizedMatrix
     */
    OptimizedMatrix<double> getLogForwardVariables() const;
    
    /**
     * @brief Get the log backward variables matrix
     * @return The log backward variables as an OptimizedMatrix
     */
    OptimizedMatrix<double> getLogBackwardVariables() const;
    
    /**
     * @brief Get the forward variables matrix (converted from log space)
     * @return The forward variables as an OptimizedMatrix
     */
    OptimizedMatrix<double> getForwardVariables() const;
    
    /**
     * @brief Get the backward variables matrix (converted from log space)
     * @return The backward variables as an OptimizedMatrix
     */
    OptimizedMatrix<double> getBackwardVariables() const;
    
    /**
     * @brief Get the log forward variables as basic Matrix (for compatibility)
     * @return The log forward variables as a Matrix
     */
    Matrix getLogForwardVariablesCompat() const;
    
    /**
     * @brief Get the log backward variables as basic Matrix (for compatibility)
     * @return The log backward variables as a Matrix
     */
    Matrix getLogBackwardVariablesCompat() const;
    
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
     * @brief Perform SIMD-optimized log-space forward pass
     */
    void computeLogForward();
    
    /**
     * @brief Perform SIMD-optimized log-space backward pass
     */
    void computeLogBackward();
    
    /**
     * @brief Initialize log forward variables with first observation
     */
    void initializeLogForwardStep();
    
    /**
     * @brief Perform SIMD-optimized log forward step at time t
     * @param t Time step index
     */
    void computeLogForwardStepSIMD(std::size_t t);
    
    /**
     * @brief Fallback scalar computation for log forward step
     * @param t Time step index
     */
    void computeLogForwardStepScalar(std::size_t t);
    
    /**
     * @brief Initialize log backward variables
     */
    void initializeLogBackwardStep();
    
    /**
     * @brief Perform SIMD-optimized log backward step at time t
     * @param t Time step index
     */
    void computeLogBackwardStepSIMD(std::size_t t);
    
    /**
     * @brief Fallback scalar computation for log backward step
     * @param t Time step index
     */
    void computeLogBackwardStepScalar(std::size_t t);
    
    /**
     * @brief Compute final log probability from log forward variables
     */
    void computeFinalLogProbability();
    
    /**
     * @brief SIMD-optimized computation of log emission probabilities
     * @param observation Current observation
     * @param logEmisProbs Output log emission probabilities (aligned)
     */
    void computeLogEmissionProbabilities(Observation observation, double* logEmisProbs) const;
    
    /**
     * @brief SIMD-optimized extended logarithm operations
     */
    void vectorizedEln(const double* input, double* output, std::size_t size) const;
    void vectorizedElnSum(const double* x, const double* y, double* result, std::size_t size) const;
    void vectorizedElnProduct(const double* x, const double* y, double* result, std::size_t size) const;
    
    /**
     * @brief SIMD-optimized log-space matrix-vector multiplication
     */
    void logMatrixVectorMultiply(const double* logMatrix, const double* logVector,
                                 double* logResult, std::size_t rows, std::size_t cols) const;
    
    /**
     * @brief SIMD-optimized log-space transposed matrix-vector multiplication
     */
    void logMatrixVectorMultiplyTransposed(const double* logMatrix, const double* logVector,
                                           double* logResult, std::size_t rows, std::size_t cols) const;
    
    /**
     * @brief Check if log values are valid (not NaN or -inf in inappropriate places)
     * @param t Time step index
     * @param logVariables Pointer to log variables to check
     * @return True if values are valid
     */
    bool areLogValuesValid(std::size_t t, const double* logVariables) const;
    
    /**
     * @brief Get matrix index for log forward/backward matrices
     * @param t Time step
     * @param state State index
     * @return Linear index in matrix
     */
    std::size_t getMatrixIndex(std::size_t t, std::size_t state) const {
        return t * numStates_ + state;
    }
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
