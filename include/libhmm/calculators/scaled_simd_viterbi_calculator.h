#ifndef LIBHMM_SCALED_SIMD_VITERBI_CALCULATOR_H_
#define LIBHMM_SCALED_SIMD_VITERBI_CALCULATOR_H_

#include "libhmm/hmm.h"
#include "libhmm/calculators/calculator.h"
#include "libhmm/performance/simd_support.h"
#include <cfloat>
#include <cmath>
#include <vector>
#include <memory>

namespace libhmm {

/**
 * @brief SIMD-optimized scaled Viterbi algorithm implementation
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
 * The algorithm implements the scaled Viterbi approach where delta values
 * are scaled at each time step to prevent underflow, and the final
 * log probability is reconstructed from the scaling factors.
 */
class ScaledSIMDViterbiCalculator : public Calculator {
private:
    // Core matrices (aligned for SIMD)
    std::vector<double, performance::aligned_allocator<double>> delta_;
    std::vector<double, performance::aligned_allocator<double>> scaledDelta_;
    std::vector<int> psi_;
    std::vector<double> scalingFactors_;
    
    // State sequence and probability
    StateSequence sequence_;
    double logProbability_;
    
    // Problem dimensions
    std::size_t numStates_;
    std::size_t seqLength_;
    
    // SIMD optimization parameters
    static constexpr std::size_t SIMD_BLOCK_SIZE = 8;
    static constexpr double SCALING_THRESHOLD = 1e-100;
    static constexpr double LOG_SCALING_THRESHOLD = -230.0; // log(1e-100)
    
    // Temporary SIMD-aligned vectors for computations
    mutable std::vector<double, performance::aligned_allocator<double>> tempScores_;
    mutable std::vector<double, performance::aligned_allocator<double>> tempProbs_;

public:
    /**
     * @brief Constructor with HMM and observations
     * @param hmm Pointer to the HMM (must not be null)
     * @param observations The observation set to process
     * @throws std::invalid_argument if hmm is null
     */
    ScaledSIMDViterbiCalculator(Hmm* hmm, const ObservationSet& observations);
    
    /**
     * @brief Compute the optimal state sequence using scaled SIMD Viterbi
     * 
     * Performs the complete Viterbi algorithm with numerical scaling
     * and SIMD acceleration. Returns the most likely state sequence.
     * 
     * @return The optimal state sequence
     * @throws std::runtime_error if computation fails
     */
    StateSequence decode();
    
    /**
     * @brief Get the log probability of the optimal path
     * 
     * Returns the log probability of the most likely state sequence,
     * properly reconstructed from the scaling factors.
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

private:
    /**
     * @brief Initialize delta matrix with first observation
     * Handles the initialization step with proper scaling
     */
    void initializeFirstStep();
    
    /**
     * @brief Perform SIMD-optimized forward pass
     * @param t Time step index
     */
    void computeForwardStepSIMD(std::size_t t);
    
    /**
     * @brief Fallback scalar computation for forward step
     * @param t Time step index
     */
    void computeForwardStepScalar(std::size_t t);
    
    /**
     * @brief Apply scaling to delta values at time step t
     * @param t Time step index
     * @return Scaling factor applied
     */
    double applyScaling(std::size_t t);
    
    /**
     * @brief Find optimal final state and compute termination
     */
    void computeTermination();
    
    /**
     * @brief Backtrack to find optimal state sequence
     */
    void backtrackPath();
    
    /**
     * @brief Reconstruct log probability from scaling factors
     */
    void reconstructLogProbability();
    
    /**
     * @brief SIMD-optimized computation of transition scores
     * @param fromState Source state
     * @param toStatesStart Pointer to first target state data
     * @param transitionsStart Pointer to transition probabilities
     * @param numStates Number of states to process
     * @param prevDelta Previous delta values
     * @param results Output scores
     */
    void computeTransitionScoresSIMD(
        std::size_t fromState,
        const double* transitionsStart,
        const double* prevDelta,
        double* results,
        std::size_t numStates) const;
    
    /**
     * @brief Find minimum value and its index in SIMD-friendly way
     * @param values Array of values
     * @param size Number of values
     * @param minValue Output minimum value
     * @param minIndex Output index of minimum
     */
    void findMinSIMD(const double* values, std::size_t size, 
                     double& minValue, std::size_t& minIndex) const;
    
    /**
     * @brief Compute emission probabilities for all states at once
     * @param observation Current observation
     * @param emisProbs Output emission probabilities (aligned)
     */
    void computeEmissionProbabilities(Observation observation, double* emisProbs) const;
    
    /**
     * @brief Check if current delta values need scaling
     * @param t Time step index
     * @return True if scaling is needed
     */
    bool needsScaling(std::size_t t) const;
    
    /**
     * @brief Get matrix index for delta/psi matrices
     * @param t Time step
     * @param state State index
     * @return Linear index for matrix access
     */
    std::size_t getMatrixIndex(std::size_t t, std::size_t state) const noexcept {
        return t * numStates_ + state;
    }
};

} // namespace libhmm

#endif // LIBHMM_SCALED_SIMD_VITERBI_CALCULATOR_H_
