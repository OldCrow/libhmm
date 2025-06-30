#ifndef LIBHMM_ADVANCED_LOG_SIMD_VITERBI_CALCULATOR_H_
#define LIBHMM_ADVANCED_LOG_SIMD_VITERBI_CALCULATOR_H_

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
 * @brief Advanced log-SIMD Viterbi calculator with maximum optimizations
 *
 * This calculator implements the most efficient Viterbi algorithm using
 * state-of-the-art optimization techniques:
 *
 * Performance Features:
 * - Pure log-space arithmetic for stability and performance
 * - SIMD vectorization for critical performance enhancement
 * - Work-stealing thread pool to optimize core usage
 * - Precomputed matrices for faster calculations
 * - Reduced divisions and expensive operations
 * - Loop and memory access optimizations
 *
 * Numerical Features:
 * - Stable log-sum-exp calculations
 * - Safe handling of log(0) edge cases
 * - Precision maintenance throughout calculations
 * - Fully log-space without scaling factors
 *
 * Memory Features:
 * - Aligned memory allocation for SIMD efficiency
 * - Optimized matrix structures for speed
 * - Minimized memory footprint
 * - Cache-aware data access patterns
 */
class AdvancedLogSIMDViterbiCalculator : public Calculator {
public:
    /**
     * @brief Constructor with advanced optimization options (modern, type-safe)
     * @param hmm Reference to the HMM (const for immutability)
     * @param observations The observation set to process
     * @param useWorkStealing Enable work-stealing thread pool (default: auto)
     * @param precomputeMatrices Precompute log matrices for speed (default: true)
     * @throws std::invalid_argument if hmm is invalid
     */
    AdvancedLogSIMDViterbiCalculator(const Hmm& hmm, const ObservationSet& observations,
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
    AdvancedLogSIMDViterbiCalculator(Hmm* hmm, const ObservationSet& observations,
                             bool useWorkStealing = true, bool precomputeMatrices = true);
    
    /**
     * @brief Compute the most likely state sequence using all optimizations
     *
     * Utilizes all available optimizations for the fastest Viterbi path calculation
     * in the library.
     *
     * @throws std::runtime_error if computation fails
     */
    StateSequence decode();
    
    /**
     * @brief Get the log probability of the most likely path
     * @return The log probability value
     */
    double getLogProbability() const noexcept { return logProbability_; }
    
    /**
     * @brief Get the optimal state sequence
     * @return The optimal state sequence
     */
    StateSequence getStateSequence() const noexcept { return sequence_; }
    
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
     * @return Always true for AdvancedLogSIMDViterbiCalculator
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
    std::vector<double, performance::aligned_allocator<double>> logDelta_;
    std::vector<int> psi_;
    
    // Precomputed matrices and initializations
    std::vector<double, performance::aligned_allocator<double>> logTransitionMatrix_;
    std::vector<double, performance::aligned_allocator<double>> logInitialStateProbs_;
    
    // Temporary computation buffers (SIMD-aligned)
    mutable std::vector<double, performance::aligned_allocator<double>> tempLogScores_;
    mutable std::vector<double, performance::aligned_allocator<double>> tempLogEmissions_;
    
    // Results
    StateSequence sequence_;
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
     * @brief Optimized Viterbi pass using all available optimizations
     */
    void computeOptimizedViterbi();
    
    /**
     * @brief Initialize log delta with first observation
     */
    void initializeLogDelta();
    
    /**
     * @brief Advanced forward step with work-stealing and SIMD
     * @param t Time step index
     */
    void computeAdvancedForwardStep(std::size_t t);
    
    /**
     * @brief Compute final termination step using optimized summation
     */
    void computeTermination();
    
    /**
     * @brief Backtrack to find optimal state sequence
     */
    void backtrackPath();
    
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
     * @brief Work-stealing parallel computation for forward steps
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

#endif // LIBHMM_ADVANCED_LOG_SIMD_VITERBI_CALCULATOR_H_  

