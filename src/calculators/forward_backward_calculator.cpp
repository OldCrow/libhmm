/**
 * @file forward_backward_calculator.cpp
 * @brief Implementation of the standard Forward-Backward algorithm
 * 
 * This file implements the ForwardBackwardCalculator class, providing the
 * classic Forward-Backward algorithm for HMM probability computation.
 * The implementation focuses on clarity and correctness while leveraging
 * modern C++17 features and performance optimizations.
 */

#include "libhmm/calculators/forward_backward_calculator.h"

// Standard library dependencies (organized)
#include <algorithm>      // For std::max
#include <cmath>          // For std::log, std::exp, std::isnan
#include <iostream>       // For error output
#include <limits>         // For std::numeric_limits
#include <stdexcept>      // For std::runtime_error
#include <vector>         // For std::vector

namespace libhmm
{

void ForwardBackwardCalculator::forward()
{
    const Hmm& hmm = getHmmRef();  // Modern type-safe access
    const auto numStates = static_cast<std::size_t>(hmm.getNumStates());
    const auto obsSize = observations_.size();
    
    // Resize matrices once to avoid reallocation
    forwardVariables_.resize(obsSize, numStates);
    
    // Cache const references to avoid repeated function calls
    const Matrix& trans = hmm.getTrans();
    const Vector& pi = hmm.getPi();
    
    // Cache emission probabilities for better performance
    std::vector<double> emissionProbs(numStates);
    
    // Initialization step: alpha(0, i) = pi(i) * b_i(O_0)
    for (std::size_t i = 0; i < numStates; ++i) {
        emissionProbs[i] = hmm.getProbabilityDistribution(static_cast<int>(i))->getProbability(observations_(0));
        forwardVariables_(0, i) = pi(i) * emissionProbs[i];
        
        // Clamp to avoid numerical issues using consolidated constants
        if (forwardVariables_(0, i) < constants::precision::ZERO || std::isnan(forwardVariables_(0, i))) {
            forwardVariables_(0, i) = constants::precision::ZERO;
        }
    }
    
    // Forward recursion with optimized memory access patterns
    for (std::size_t t = 1; t < obsSize; ++t) {
        // Cache emission probabilities for current observation
        for (std::size_t j = 0; j < numStates; ++j) {
            emissionProbs[j] = hmm.getProbabilityDistribution(static_cast<int>(j))->getProbability(observations_(t));
        }
        
        // Compute forward variables with better cache locality
        for (std::size_t j = 0; j < numStates; ++j) {
            double sum = 0.0;
            
            // Manual inner product for better performance
            for (std::size_t i = 0; i < numStates; ++i) {
                sum += forwardVariables_(t - 1, i) * trans(i, j);
            }
            
            forwardVariables_(t, j) = emissionProbs[j] * sum;
            
            // Clamp to avoid numerical issues using consolidated constants
            if (forwardVariables_(t, j) < constants::precision::ZERO || std::isnan(forwardVariables_(t, j))) {
                forwardVariables_(t, j) = constants::precision::ZERO;
            }
        }
    }
}

// Implements the Backward algorithm
//  This is NOT scaled!!
void ForwardBackwardCalculator::backward() {
    const Hmm& hmm = getHmmRef();  // Modern type-safe access
    const auto numStates = static_cast<std::size_t>(hmm.getNumStates());
    const auto obsSize = observations_.size();
    
    // Resize matrices once to avoid reallocation
    backwardVariables_.resize(obsSize, numStates);
    
    // Cache const reference to avoid repeated function calls
    const Matrix& trans = hmm.getTrans();
    
    // Cache emission probabilities for better performance
    std::vector<double> emissionProbs(numStates);

    // Initialize: beta_T(i) = 1 for all i (Rabiner's convention)
    for (std::size_t i = 0; i < numStates; ++i) {
        backwardVariables_(obsSize - 1, i) = constants::math::ONE;
    }

    // Backward recursion: beta_t(i) = sum(a[i][j] * b_j(O_{t+1}) * beta_{t+1}(j))
    for (std::size_t t = obsSize - 1; t > 0; --t) {
        const std::size_t currentT = t - 1;
        
        // Cache emission probabilities for current observation
        for (std::size_t j = 0; j < numStates; ++j) {
            emissionProbs[j] = hmm.getProbabilityDistribution(static_cast<int>(j))->getProbability(observations_(t));
        }
        
        // Compute backward variables with better cache locality
        for (std::size_t i = 0; i < numStates; ++i) {
            double sum = 0.0;
            
            // Manual summation for better performance
            for (std::size_t j = 0; j < numStates; ++j) {
                sum += trans(i, j) * emissionProbs[j] * backwardVariables_(t, j);
            }
            
            backwardVariables_(currentT, i) = sum;
        }
    }
}

double ForwardBackwardCalculator::probability() {
    const Hmm& hmm = getHmmRef();  // Modern type-safe access
    double p = 0.0;
    const auto numStates = static_cast<std::size_t>(hmm.getNumStates());
    const auto lastObsIndex = observations_.size() - 1;

    // Probability of the entire sequence is given by the sum of the elements 
    // in the last column
    for (std::size_t i = 0; i < numStates; ++i) {
        p += forwardVariables_(lastObsIndex, i);
    }
    
    if (p > constants::math::ONE) {
        std::cerr << "ForwardBackwardCalculator: Numeric Underflow occurred!" << std::endl;
        std::cerr << "Probability: " << p << std::endl;
        std::cerr << "Observations: " << observations_ << std::endl;
        std::cerr << "HMM: " << hmm << std::endl;
        std::cerr << "Forward variables:" << forwardVariables_ << std::endl;
    }

    assert(p <= constants::math::ONE);
    return p;    
}

double ForwardBackwardCalculator::getLogProbability() const {
    if (!isComputed()) {
        throw std::runtime_error("Forward-backward computation not performed");
    }
    
    const auto numStates = forwardVariables_.size2();
    const auto lastObsIndex = observations_.size() - 1;
    
    // Use log-sum-exp for numerical stability
    double logSum = -std::numeric_limits<double>::infinity();
    
    for (std::size_t i = 0; i < numStates; ++i) {
        const double forwardVal = forwardVariables_(lastObsIndex, i);
        if (forwardVal > 0.0) {
            const double logVal = std::log(forwardVal);
            
            if (std::isinf(logSum)) {
                logSum = logVal;
            } else {
                // Optimized numerically stable log-sum-exp using precomputed constants
                const double maxVal = std::max(logSum, logVal);
                const double diff1 = logSum - maxVal;
                const double diff2 = logVal - maxVal;
                
                // Avoid expensive exp() operations when differences are too small
                if (diff1 < constants::probability::MIN_LOG_PROBABILITY) {
                    logSum = maxVal + diff2;  // logSum contribution negligible
                } else if (diff2 < constants::probability::MIN_LOG_PROBABILITY) {
                    logSum = maxVal + diff1;  // logVal contribution negligible  
                } else {
                    // Standard log-sum-exp when both terms are significant
                    logSum = maxVal + std::log(std::exp(diff1) + std::exp(diff2));
                }
            }
        }
    }
    
    return logSum;
}

} // namespace libhmm
