#ifndef LIBHMM_FORWARD_BACKWARD_CALCULATOR_H_
#define LIBHMM_FORWARD_BACKWARD_CALCULATOR_H_

/**
 * @file forward_backward_calculator.h
 * @brief Standard Forward-Backward algorithm implementation for HMMs
 * 
 * This file contains the ForwardBackwardCalculator class, which implements the
 * standard Forward-Backward algorithm for Hidden Markov Models. The algorithm
 * computes forward and backward probabilities for a given observation sequence,
 * enabling probability calculations and parameter estimation.
 * 
 * The Forward-Backward algorithm is fundamental to HMM computations and forms
 * the basis for:
 * - Observation sequence likelihood calculation
 * - Baum-Welch parameter estimation
 * - Posterior state probability computation
 * 
 * This implementation focuses on correctness and clarity, serving as the baseline
 * calculator. For numerical stability and performance optimizations, consider
 * the SIMD or scaled variants.
 * 
 * @author libhmm team
 * @see ScaledSIMDForwardBackwardCalculator for numerical stability
 * @see LogSIMDForwardBackwardCalculator for log-space computation
 */

// Core dependencies (organized by category)
#include "libhmm/common/common.h"           // Basic types, constants, and matrix classes
#include "libhmm/calculators/calculator.h"   // Base calculator interface
#include "libhmm/hmm.h"                      // HMM class definition

// Standard library dependencies (minimal set)
#include <cmath>                              // Mathematical functions

namespace libhmm {

/**
 * @class ForwardBackwardCalculator
 * @brief Standard implementation of the Forward-Backward algorithm
 * 
 * The ForwardBackwardCalculator implements the classic Forward-Backward algorithm
 * as described in Rabiner's tutorial. It computes:
 * 
 * **Forward Variables (α):**
 * - α_t(i) = P(O_1, O_2, ..., O_t, q_t = i | λ)
 * - Probability of partial observation sequence and being in state i at time t
 * 
 * **Backward Variables (β):**
 * - β_t(i) = P(O_{t+1}, O_{t+2}, ..., O_T | q_t = i, λ)
 * - Probability of future observations given state i at time t
 * 
 * **Algorithm Characteristics:**
 * - Time Complexity: O(T × N²) where T = sequence length, N = number of states
 * - Space Complexity: O(T × N) for forward and backward matrices
 * - Numerical Stability: Limited - may underflow for long sequences
 * - Performance: Baseline implementation, no vectorization
 * 
 * **Use Cases:**
 * - Short to medium observation sequences (< 100 observations)
 * - Educational and reference implementation
 * - Baseline for performance comparisons
 * - Simple HMM analysis where numerical stability is not critical
 * 
 * **Limitations:**
 * - No scaling: susceptible to numerical underflow
 * - No SIMD optimization
 * - Limited parallelization
 * 
 * For production use with long sequences or numerical stability requirements,
 * consider ScaledSIMDForwardBackwardCalculator or LogSIMDForwardBackwardCalculator.
 * 
 * @code
 * // Basic usage
 * ForwardBackwardCalculator calc(hmm, observations);
 * double probability = calc.probability();
 * Matrix forward = calc.getForwardVariables();
 * Matrix backward = calc.getBackwardVariables();
 * 
 * // Numerically stable log probability
 * double logProb = calc.getLogProbability();
 * @endcode
 */
class ForwardBackwardCalculator: public Calculator
{
protected:
    /// Computes forward variables using the forward algorithm
    virtual void forward();

    /// Computes backward variables using the backward algorithm
    virtual void backward();

    Matrix forwardVariables_;
    Matrix backwardVariables_;
    
public:
    /// Default constructor
    ForwardBackwardCalculator() = default;

    /// Modern C++17 type-safe constructor (preferred)
    /// @param hmm Const reference to HMM (immutable, lifetime-safe)
    /// @param observations The observation set to process
    ForwardBackwardCalculator(const Hmm& hmm, const ObservationSet& observations)
        : Calculator(hmm, observations) {
        forward();
        backward();
    }
    
    /// Legacy pointer-based constructor (backward compatibility)
    /// @param hmm Pointer to the HMM (must not be null)
    /// @param observations The observation set to process
    /// @throws std::invalid_argument if hmm is null
    /// @deprecated Use const reference constructor for better type safety
    [[deprecated("Use const reference constructor for better type safety")]]
    ForwardBackwardCalculator(Hmm* hmm, const ObservationSet& observations)
        : Calculator(hmm, observations) {
        forward();
        backward();
    }
    
    /// Virtual destructor
    virtual ~ForwardBackwardCalculator() = default;
    
    /// Get the forward variables matrix
    /// @return The forward variables matrix
    virtual Matrix getForwardVariables() const noexcept { return forwardVariables_; }

    /// Get the backward variables matrix  
    /// @return The backward variables matrix
    virtual Matrix getBackwardVariables() const noexcept { return backwardVariables_; }
    
    /// Calculates the probability of the observation set given the HMM
    /// @return The probability value
    virtual double probability();
    
    /// Get log probability for numerical stability (derived from forward variables)
    /// @return The log probability value
    virtual double getLogProbability() const;
    
    /// Check if computation has been performed
    /// @return True if forward and backward variables are computed
    bool isComputed() const noexcept {
        return forwardVariables_.size1() == observations_.size() && 
               backwardVariables_.size1() == observations_.size();
    }
};

}

#endif
