#ifndef LIBHMM_VITERBI_CALCULATOR_H_
#define LIBHMM_VITERBI_CALCULATOR_H_

/**
 * @file viterbi_calculator.h
 * @brief Standard Viterbi algorithm implementation for HMMs
 * 
 * This file contains the ViterbiCalculator class, which implements the
 * standard Viterbi algorithm for Hidden Markov Models. The algorithm
 * finds the most likely state sequence given an observation sequence,
 * solving the decoding problem in HMM analysis.
 * 
 * The Viterbi algorithm is essential for:
 * - Finding the most probable hidden state sequence
 * - HMM-based sequence alignment
 * - Speech recognition and natural language processing
 * - Bioinformatics sequence analysis
 * - Pattern recognition in time series
 * 
 * This implementation provides the baseline Viterbi algorithm with log-space
 * computation for numerical stability. For performance optimizations, consider
 * the SIMD variants.
 * 
 * @author libhmm team
 * @see ScaledSIMDViterbiCalculator for performance optimization
 * @see LogSIMDViterbiCalculator for enhanced numerical stability
 */

// Core dependencies (organized by category)
#include "libhmm/common/common.h"           // Basic types, constants, and matrix classes
#include "libhmm/calculators/calculator.h"   // Base calculator interface
#include "libhmm/hmm.h"                      // HMM class definition

// Standard library dependencies (minimal set)
#include <cfloat>                             // Floating-point limits
#include <cmath>                              // Mathematical functions

namespace libhmm {

/**
 * @class ViterbiCalculator
 * @brief Standard implementation of the Viterbi algorithm for optimal state sequence decoding
 * 
 * The ViterbiCalculator implements the classic Viterbi algorithm for finding the most
 * likely state sequence given an observation sequence. The algorithm uses dynamic
 * programming to efficiently compute:
 * 
 * **Most Likely State Sequence:**
 * - q* = argmax P(Q | O, λ) where Q is a state sequence
 * - Solves the "decoding problem" in HMM analysis
 * 
 * **Path Probability:**
 * - P* = max P(Q, O | λ) over all possible state sequences Q
 * - Log probability of the optimal path for numerical stability
 * 
 * **Algorithm Characteristics:**
 * - Time Complexity: O(T × N²) where T = sequence length, N = number of states
 * - Space Complexity: O(T × N) for delta and psi matrices
 * - Numerical Stability: Good - uses log-space computation
 * - Performance: Baseline implementation with cached log probabilities
 * 
 * **Implementation Details:**
 * - Uses log-space arithmetic throughout to prevent underflow
 * - Caches log transition and emission probabilities for efficiency
 * - Employs the standard three-step Viterbi procedure:
 *   1. Initialization: δ₁(i) = log π_i + log b_i(O₁)
 *   2. Recursion: δₜ(j) = max[δₜ₋₁(i) + log a_ij] + log b_j(Oₜ)
 *   3. Termination: Backtrack using ψ pointers
 * 
 * **Use Cases:**
 * - Sequence decoding in speech recognition
 * - Gene finding and sequence annotation
 * - Financial time series state identification
 * - Pattern recognition in sensor data
 * - Any application requiring most likely hidden state sequence
 * 
 * **Advantages:**
 * - Numerically stable log-space computation
 * - Optimal solution guaranteed
 * - Efficient dynamic programming approach
 * - Clear separation of concerns with caching
 * 
 * For enhanced performance with large state spaces or long sequences,
 * consider ScaledSIMDViterbiCalculator or LogSIMDViterbiCalculator.
 * 
 * @code
 * // Basic usage
 * ViterbiCalculator calc(hmm, observations);
 * StateSequence sequence = calc.decode();
 * double logProb = calc.getLogProbability();
 * 
 * // Access computed sequence multiple times
 * StateSequence samePath = calc.getStateSequence();
 * @endcode
 */
class ViterbiCalculator : public Calculator
{
private:
    Matrix delta_;
    Matrix psi_;
    StateSequence sequence_;
    double logProbability_;

public:
    /// Modern C++17 type-safe constructor (preferred)
    /// @param hmm Const reference to HMM (immutable, lifetime-safe)
    /// @param observations The observation set to process
    ViterbiCalculator(const Hmm& hmm, const ObservationSet& observations)
        : Calculator(hmm, observations),
          delta_(observations.size(), hmm.getNumStates()),
          psi_(observations.size(), hmm.getNumStates()),
          sequence_(observations.size()),
          logProbability_(0.0) {
        
        clear_matrix(delta_);
        clear_matrix(psi_);
        clear_vector(sequence_);
    }
    
    /// Legacy pointer-based constructor (backward compatibility)
    /// @param hmm Pointer to the HMM (must not be null)
    /// @param observations The observation set to process
    /// @throws std::invalid_argument if hmm is null
    /// @deprecated Use const reference constructor for better type safety
    [[deprecated("Use const reference constructor for better type safety")]]
    ViterbiCalculator(Hmm* hmm, const ObservationSet& observations)
        : Calculator(hmm, observations),
          delta_(observations.size(), hmm->getNumStates()),
          psi_(observations.size(), hmm->getNumStates()),
          sequence_(observations.size()),
          logProbability_(0.0) {
        
        clear_matrix(delta_);
        clear_matrix(psi_);
        clear_vector(sequence_);
    }

    /// Begins the process of actually computing the optimal state sequence.
    /// This function is a reorganization from the ViterbiCalculator class in 
    /// JAHMM...the constructor did this work in that class.
    /// @return The optimal state sequence
    StateSequence decode();

    /// Get the log probability of the optimal path
    /// @return The log probability value
    double getLogProbability() const noexcept {
        return logProbability_;
    }

    /// Returns the state sequence already computed.
    /// @return The computed state sequence
    StateSequence getStateSequence() const noexcept {
        return sequence_;
    }
};

} //namespace

#endif
