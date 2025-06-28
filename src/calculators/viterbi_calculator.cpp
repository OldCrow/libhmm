#include "libhmm/calculators/viterbi_calculator.h"
#include <cfloat>
#include <cmath>
#include <algorithm>

namespace libhmm{

/*
 * Finds the optimal state sequence for an ObservationSet.
 */    
StateSequence ViterbiCalculator::decode() {
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    const auto obsSize = observations_.size();
    const Matrix a = hmm_->getTrans();

    // Step 1: Initialization
    // delta( 0, i ) = ln( pi( i ) ) + ln( b( i, O_1 ) )
    // psi( 0, i ) = 0
    for (std::size_t i = 0; i < numStates; ++i) {
        const double piProb = hmm_->getPi()(i);
        
        // Use optimized getLogProbability for better performance and numerical stability
        const double logEmis = hmm_->getProbabilityDistribution(static_cast<int>(i))->getLogProbability(observations_(0));
        
        // Handle zero probabilities to avoid log(0)
        const double logPi = (piProb > 0.0) ? std::log(piProb) : std::log(ZERO);
        
        delta_(0, i) = logPi + logEmis;
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
        const Observation o = observations_(t);
        
        for (std::size_t j = 0; j < numStates; ++j) {
            // Find the maximum log probability
            double maxValue = -DBL_MAX;
            std::size_t maxIndex = 0;
            
            for (std::size_t i = 0; i < numStates; ++i) {
                const double transProb = a(i, j);
                const double logTrans = (transProb > 0.0) ? std::log(transProb) : std::log(ZERO);
                const double temp = delta_(t - 1, i) + logTrans;
                
                if (maxValue < temp) {
                    maxValue = temp;
                    maxIndex = i;
                }
            }

            // Store the best previous state
            psi_(t, j) = static_cast<int>(maxIndex);
            
            // Use optimized getLogProbability for better performance and numerical stability
            const double logEmis = hmm_->getProbabilityDistribution(static_cast<int>(j))->getLogProbability(o);
            
            delta_(t, j) = maxValue + logEmis;
        }
    }

    // Step 3: Termination
    //  P* = max( delta( T, i ), i=1..N )
    //  q* = arg max( delta( T, i ), i = 1..N )
    //  q is the last state
    logProbability_ = -DBL_MAX;
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

}// namespace
