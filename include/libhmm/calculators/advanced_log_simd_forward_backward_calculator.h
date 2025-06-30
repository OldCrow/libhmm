#ifndef LIBHMM_ADVANCED_LOG_SIMD_FORWARD_BACKWARD_CALCULATOR_H_
#define LIBHMM_ADVANCED_LOG_SIMD_FORWARD_BACKWARD_CALCULATOR_H_

#include "libhmm/hmm.h"
#include "libhmm/calculators/calculator.h"
#include "libhmm/performance/simd_support.h"
#include "libhmm/performance/work_stealing_pool.h"
#include "libhmm/performance/log_space_ops.h"
#include "libhmm/common/optimized_matrix.h"
#include <memory>
#include <vector>
#include <chrono>

namespace libhmm {

/**
 * @brief Advanced log-SIMD Forward-Backward calculator with maximum optimizations
 * 
 * This calculator represents the state-of-the-art in HMM forward-backward computation,
 * incorporating all available optimization techniques:
 * 
 * Performance Optimizations:
 * - Pure log-space arithmetic with lookup tables
 * - SIMD vectorization for all critical paths
 * - Work-stealing thread pool for dynamic load balancing
 * - Precomputed transition matrices
 * - Cache-friendly memory layout and access patterns
 * - Eliminated divisions and expensive operations
 * - Optimized loop structures and data dependencies
 * 
 * Numerical Optimizations:
 * - Numerically stable log-sum-exp operations
 * - Efficient handling of log(0) cases
 * - Minimal precision loss throughout computation
 * - No scaling factors needed (pure log-space)
 * 
 * Memory Optimizations:
 * - SIMD-aligned memory allocation
 * - Efficient matrix storage and access
 * - Minimized memory allocations during computation
 * - Cache-friendly data structures
 */
class AdvancedLogSIMDForwardBackwardCalculator : public Calculator {
public:
    /**
     * @brief Constructor with advanced optimization options
     * @param hmm Reference to the HMM (const for immutability)
     * @param observations The observation set to process
     * @param useWorkStealing Enable work-stealing thread pool (default: auto)
     * @param precomputeMatrices Precompute log matrices for speed (default: true)
     * @throws std::invalid_argument if hmm is invalid
     */
    AdvancedLogSIMDForwardBackwardCalculator(const Hmm& hmm, const ObservationSet& observations,
                                            bool useWorkStealing = true, bool precomputeMatrices = true);
    
    /**
     * @brief Legacy constructor for compatibility with existing pointer-based API
     * @param hmm Pointer to the HMM (must not be null)
     * @param observations The observation set to process
     * @param useWorkStealing Enable work-stealing thread pool (default: auto)
     * @param precomputeMatrices Precompute log matrices for speed (default: true)
     * @throws std::invalid_argument if hmm is null
     * @deprecated Use const reference constructor for better type safety
     */
    [[deprecated("Use const reference constructor for better type safety")]]
    AdvancedLogSIMDForwardBackwardCalculator(Hmm* hmm, const ObservationSet& observations,
                                            bool useWorkStealing = true, bool precomputeMatrices = true);
    
    /**
     * @brief Compute forward and backward variables using all optimizations
     * 
     * This method represents the fastest possible forward-backward computation
     * available in the library, utilizing all performance optimizations.
     * 
     * @throws std::runtime_error if computation fails
     */
    void compute();
    
    /**
     * @brief Get the log probability of the observation sequence
     * @return The log probability value
     */
    double getLogProbability() const noexcept { return logProbability_; }
    
    /**
     * @brief Get the probability of the observation sequence
     * @return The probability value (may underflow for very small values)
     */
    double getProbability() const noexcept { return std::exp(logProbability_); }
    
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
     * @brief Get performance statistics for this computation
     */
    struct PerformanceStats {
        double computationTimeMs;
        std::size_t simdOperations;
        std::size_t workStealingTasks;
        double workStealingEfficiency;
        std::size_t cacheHits;
        std::size_t totalOperations;
    };
    
    PerformanceStats getPerformanceStats() const { return stats_; }
    
    /**
     * @brief Check if this calculator uses maximum optimizations
     * @return Always true for AdvancedLogSIMDForwardBackwardCalculator
     */
    static constexpr bool isFullyOptimized() noexcept { return true; }
    
    /**
     * @brief Get optimization information string
     */
    std::string getOptimizationInfo() const;

private:
    // Performance configuration
    bool useWorkStealing_;
    bool precomputeMatrices_;
    
    // Problem dimensions
    std::size_t numStates_;
    std::size_t seqLength_;
    std::size_t alignedStateSize_;  // Padded for SIMD alignment
    
    // Core computation matrices (SIMD-aligned)
    std::vector<double, performance::aligned_allocator<double>> logForwardVariables_;
    std::vector<double, performance::aligned_allocator<double>> logBackwardVariables_;
    
    // Precomputed matrices for maximum speed
    std::vector<double, performance::aligned_allocator<double>> logTransitionMatrix_;
    std::vector<double, performance::aligned_allocator<double>> logInitialStateProbs_;
    
    // Temporary computation buffers (SIMD-aligned)
    mutable std::vector<double, performance::aligned_allocator<double>> tempLogEmissions_;
    mutable std::vector<double, performance::aligned_allocator<double>> tempLogProbs_;
    mutable std::vector<double, performance::aligned_allocator<double>> tempWorkBuffer_;
    
    // Results
    double logProbability_;
    
    // Performance tracking
    mutable PerformanceStats stats_;
    mutable std::chrono::high_resolution_clock::time_point computeStartTime_;
    
    /**
     * @brief Initialize all matrices and precomputed data
     */
    void initializeMatrices();
    
    /**
     * @brief Precompute log-space transition matrix for maximum efficiency
     */
    void precomputeLogTransitions();
    
    /**
     * @brief Precompute log-space initial state probabilities
     */
    void precomputeLogInitialStates();
    
    /**
     * @brief Optimized log forward pass using all available optimizations
     */
    void computeOptimizedLogForward();
    
    /**
     * @brief Optimized log backward pass using all available optimizations
     */
    void computeOptimizedLogBackward();
    
    /**
     * @brief Initialize log forward variables with first observation
     */
    void initializeLogForwardStep();
    
    /**
     * @brief Advanced forward step with work-stealing and SIMD
     * @param t Time step index
     */
    void computeAdvancedForwardStep(std::size_t t);
    
    /**
     * @brief Advanced backward step with work-stealing and SIMD
     * @param t Time step index
     */
    void computeAdvancedBackwardStep(std::size_t t);
    
    /**
     * @brief Compute final log probability using optimized summation
     */
    void computeFinalLogProbability();
    
    /**
     * @brief Optimized emission probability computation
     * @param observation Current observation
     * @param logEmisProbs Output buffer (must be aligned)
     */
    void computeOptimizedLogEmissions(Observation observation, double* logEmisProbs) const;
    
    /**
     * @brief High-performance log-space matrix-vector multiplication
     * Uses precomputed matrices and optimized SIMD operations
     */
    void optimizedLogMatrixVectorMultiply(const double* logMatrix, const double* logVector,
                                         double* result, std::size_t rows, std::size_t cols) const;
    
    /**
     * @brief Work-stealing parallel computation for forward/backward steps
     */
    template<typename Func>
    void parallelStateComputation(std::size_t numStates, Func func) const;
    
    /**
     * @brief SIMD-optimized operations on state vectors
     */
    void simdVectorOperations(const double* input1, const double* input2, 
                             double* output, std::size_t size, char operation) const;
    
    /**
     * @brief Update performance statistics
     */
    void updateStats(const std::string& operation, std::size_t operations) const;
    
    /**
     * @brief Get matrix index with proper alignment
     */
    std::size_t getAlignedIndex(std::size_t t, std::size_t state) const noexcept {
        return t * alignedStateSize_ + state;
    }
};


} // namespace libhmm

#endif // LIBHMM_ADVANCED_LOG_SIMD_FORWARD_BACKWARD_CALCULATOR_H_
