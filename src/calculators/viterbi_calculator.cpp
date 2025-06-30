/**
 * @file viterbi_calculator.cpp
 * @brief Implementation of the standard Viterbi algorithm
 * 
 * This file implements the ViterbiCalculator class, providing the
 * classic Viterbi algorithm for finding the most likely state sequence
 * in Hidden Markov Models. The implementation uses log-space computation
 * for numerical stability and caches probabilities for efficiency.
 */

#include "libhmm/calculators/viterbi_calculator.h"

// Standard library dependencies (organized)
#include <algorithm>      // For std::max
#include <cfloat>         // For DBL_MAX
#include <cmath>          // For std::log
#include <vector>         // For std::vector

namespace libhmm{

/*
 * Finds the optimal state sequence for an ObservationSet.
 */    
StateSequence ViterbiCalculator::decode() {
    const Hmm& hmm = getHmmRef();  // Modern type-safe access
    const auto numStates = static_cast<std::size_t>(hmm.getNumStates());
    const auto obsSize = observations_.size();
    
    // Cache const references to avoid repeated function calls
    const Matrix& trans = hmm.getTrans();
    const Vector& pi = hmm.getPi();
    
    // Pre-compute log transition probabilities for better performance
    std::vector<std::vector<double>> logTrans(numStates, std::vector<double>(numStates));
    for (std::size_t i = 0; i < numStates; ++i) {
        for (std::size_t j = 0; j < numStates; ++j) {
            const double transProb = trans(i, j);
            logTrans[i][j] = (transProb > constants::math::ZERO_DOUBLE) ? std::log(transProb) : constants::probability::MIN_LOG_PROBABILITY;
        }
    }
    
    // Pre-compute log initial probabilities
    std::vector<double> logPi(numStates);
    for (std::size_t i = 0; i < numStates; ++i) {
        const double piProb = pi(i);
        logPi[i] = (piProb > constants::math::ZERO_DOUBLE) ? std::log(piProb) : constants::probability::MIN_LOG_PROBABILITY;
    }
    
    // Cache emission log probabilities for better performance
    std::vector<double> logEmissionProbs(numStates);

    // Step 1: Initialization
    // delta(0, i) = ln(pi(i)) + ln(b_i(O_0))
    // psi(0, i) = 0
    for (std::size_t i = 0; i < numStates; ++i) {
        logEmissionProbs[i] = hmm.getProbabilityDistribution(static_cast<int>(i))->getLogProbability(observations_(0));
        delta_(0, i) = logPi[i] + logEmissionProbs[i];
        psi_(0, i) = 0;
    }

    // Step 2: Recursive computation
    // For 2 <= t <= T, 1 <= i <= N
    // Maximum likelihood formulation:
    //
    // for 2 <= t <= T
    //   for 1 <= j <= N
    //     delta( t, j ) = max(delta(t-1,i) + ln(a(i,j)),i=1..N) + ln( b(j,O_t))
    //     psi( t, j ) = arg max(delta(t-1,i) + ln(a(i,j)),i=1..N)
    //
    for (std::size_t t = 1; t < obsSize; ++t) {
        // Cache emission probabilities for current observation
        for (std::size_t j = 0; j < numStates; ++j) {
            logEmissionProbs[j] = hmm.getProbabilityDistribution(static_cast<int>(j))->getLogProbability(observations_(t));
        }
        
        for (std::size_t j = 0; j < numStates; ++j) {
        // Find the maximum log probability using cached values
        // Use constant instead of DBL_MAX for consistency
        double maxValue = constants::probability::MIN_LOG_PROBABILITY;
        std::size_t maxIndex = 0;
            
            for (std::size_t i = 0; i < numStates; ++i) {
                const double temp = delta_(t - 1, i) + logTrans[i][j];
                
                if (maxValue < temp) {
                    maxValue = temp;
                    maxIndex = i;
                }
            }

            // Store the best previous state
            psi_(t, j) = static_cast<int>(maxIndex);
            
            // Use cached emission log probability
            delta_(t, j) = maxValue + logEmissionProbs[j];
        }
    }

    // Step 3: Termination
    //  P* = max( delta( T, i ), i=1..N )
    //  q* = arg max( delta( T, i ), i = 1..N )
    //  q is the last state
    logProbability_ = constants::probability::MIN_LOG_PROBABILITY;
    const auto lastIndex = obsSize - 1;
    
    for (std::size_t i = 0; i < numStates; ++i) {
        if (logProbability_ < delta_(lastIndex, i)) {
            logProbability_ = delta_(lastIndex, i);
            sequence_(lastIndex) = static_cast<int>(i);
        }
    }

    // Backtrack to find the optimal path
    for (std::size_t i = obsSize - 1; i > 0; --i) {
        const auto currentIndex = i - 1;
        sequence_(currentIndex) = static_cast<int>(psi_(i, sequence_(i)));
    }

    return sequence_;
}

} // namespace libhmm
