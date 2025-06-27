#ifndef LIBHMM_LOG_SIMD_VITERBI_CALCULATOR_H_
#define LIBHMM_LOG_SIMD_VITERBI_CALCULATOR_H_

#include "libhmm/hmm.h"
#include "libhmm/calculators/calculator.h"
#include "libhmm/performance/simd_support.h"
#include <cfloat>
#include <cmath>
#include <vector>
#include <memory>

namespace libhmm {

/**
 * @brief SIMD-optimized log-space Viterbi algorithm implementation
 * 
 * This calculator performs the entire Viterbi computation in log space
 * to provide excellent numerical stability while using SIMD vectorization
 * for high performance. Working entirely in log space eliminates the need
 * for explicit scaling while providing superior numerical accuracy.
 * 
 * Key features:
 * - Pure log-space computation for maximum numerical stability
 * - SIMD vectorization of log probability operations
 * - Cache-friendly memory layout and access patterns
 * - Automatic fallback to scalar implementation
 * - Efficient handling of log(0) and numerical edge cases
 * 
 * The algorithm avoids any probability-space operations, working entirely
 * with log probabilities to maintain precision across long sequences and
 * models with extreme probability ranges.
 */
class LogSIMDViterbiCalculator : public Calculator {
private:
    // Core matrices (aligned for SIMD) - all in log space
    std::vector<double, performance::aligned_allocator<double>> logDelta_;
    std::vector<int> psi_;
    
    // State sequence and probability
    StateSequence sequence_;
    double logProbability_;
    
    // Problem dimensions
    std::size_t numStates_;
    std::size_t seqLength_;
    
    // SIMD optimization parameters (using consolidated constants)
    static constexpr std::size_t SIMD_BLOCK_SIZE = constants::simd::DEFAULT_BLOCK_SIZE;
    static constexpr double LOG_ZERO = constants::probability::MIN_LOG_PROBABILITY; // Representation of log(0)
    static constexpr double MIN_LOG_PROB = constants::probability::MIN_LOG_PROBABILITY; // Minimum meaningful log probability
    
    // Temporary SIMD-aligned vectors for computations
    mutable std::vector<double, performance::aligned_allocator<double>> tempLogScores_;
    mutable std::vector<double, performance::aligned_allocator<double>> tempLogEmisProbs_;
    mutable std::vector<double, performance::aligned_allocator<double>> tempLogTransProbs_;

public:
    /**
     * @brief Constructor with HMM and observations
     * @param hmm Pointer to the HMM (must not be null)
     * @param observations The observation set to process
     * @throws std::invalid_argument if hmm is null
     */
    LogSIMDViterbiCalculator(Hmm* hmm, const ObservationSet& observations);
    
    /**
     * @brief Compute the optimal state sequence using log-space SIMD Viterbi
     * 
     * Performs the complete Viterbi algorithm entirely in log space with
     * SIMD acceleration. Provides excellent numerical stability and performance.
     * 
     * @return The optimal state sequence
     * @throws std::runtime_error if computation fails
     */
    StateSequence decode();
    
    /**
     * @brief Get the log probability of the optimal path
     * 
     * Returns the log probability of the most likely state sequence.
     * Since computation is entirely in log space, this is directly available.
     * 
     * @return The log probability value
     */
    double getLogProbability() const noexcept {
        return logProbability_;
    }
    
    /**
     * @brief Get the computed state sequence
     * @return The optimal state sequence
     */
    StateSequence getStateSequence() const noexcept {
        return sequence_;
    }
    
    /**
     * @brief Check if SIMD optimization is being used
     * @return True if SIMD is available and being used
     */
    static bool isSIMDEnabled() noexcept {
        return performance::simd_available();
    }
    
    /**
     * @brief Get the minimum log probability representation
     * 
     * Useful for understanding the numerical limits of the calculator.
     * 
     * @return The minimum log probability value used
     */
    static double getLogZero() noexcept {
        return LOG_ZERO;
    }

private:
    /**
     * @brief Initialize log delta matrix with first observation
     * Handles the initialization step in log space
     */
    void initializeFirstStep();
    
    /**
     * @brief Perform SIMD-optimized forward pass in log space
     * @param t Time step index
     */
    void computeForwardStepSIMD(std::size_t t);
    
    /**
     * @brief Fallback scalar computation for forward step in log space
     * @param t Time step index
     */
    void computeForwardStepScalar(std::size_t t);
    
    /**
     * @brief Find optimal final state and compute termination
     */
    void computeTermination();
    
    /**
     * @brief Backtrack to find optimal state sequence
     */
    void backtrackPath();
    
    /**
     * @brief SIMD-optimized computation of log transition scores
     * @param fromState Source state
     * @param logTransitionsStart Pointer to log transition probabilities
     * @param prevLogDelta Previous log delta values
     * @param results Output log scores
     * @param numStates Number of states to process
     */
    void computeLogTransitionScoresSIMD(
        std::size_t fromState,
        const double* logTransitionsStart,
        const double* prevLogDelta,
        double* results,
        std::size_t numStates) const;
    
    /**
     * @brief Find maximum value and its index in SIMD-friendly way
     * @param values Array of log values
     * @param size Number of values
     * @param maxValue Output maximum value
     * @param maxIndex Output index of maximum
     */
    void findMaxSIMD(const double* values, std::size_t size, 
                     double& maxValue, std::size_t& maxIndex) const;
    
    /**
     * @brief Compute log emission probabilities for all states at once
     * @param observation Current observation
     * @param logEmisProbs Output log emission probabilities (aligned)
     */
    void computeLogEmissionProbabilities(Observation observation, double* logEmisProbs) const;
    
    /**
     * @brief Precompute log transition probabilities for efficient access
     * @param logTransProbs Output log transition matrix (aligned)
     */
    void precomputeLogTransitionProbabilities(double* logTransProbs) const;
    
    /**
     * @brief Safe log probability computation
     * @param prob Probability value
     * @return Log probability, or LOG_ZERO for zero/negative probabilities
     */
    static double safeLog(double prob) noexcept {
        return (prob > 0.0) ? std::log(prob) : LOG_ZERO;
    }
    
    /**
     * @brief SIMD-optimized log addition for numerical stability
     * @param logA First log value
     * @param logB Second log value
     * @return log(exp(logA) + exp(logB)) computed stably
     */
    static double logAdd(double logA, double logB) noexcept;
    
    /**
     * @brief Get matrix index for logDelta/psi matrices
     * @param t Time step
     * @param state State index
     * @return Linear index for matrix access
     */
    std::size_t getMatrixIndex(std::size_t t, std::size_t state) const noexcept {
        return t * numStates_ + state;
    }
};

} // namespace libhmm

#endif // LIBHMM_LOG_SIMD_VITERBI_CALCULATOR_H_
