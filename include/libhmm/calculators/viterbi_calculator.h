#ifndef VITERBICALCULATOR_H_
#define VITERBICALCULATOR_H_

#include "libhmm/hmm.h"
#include "libhmm/calculators/calculator.h"
#include <cfloat>
#include <cmath>

namespace libhmm{

/**
 * Somewhat based on ViterbiCalculator in JAHMM, this class implements the
 * Viterbi algorithm to determine the optimal state sequence through an HMM
 * given a series of observations.  This is *my* implementation, although
 * the source code looks very similar to the JAHMM code.
 *
 * As part of the Segmented k-Means Algorithm, the Viterbi algorithm is used 
 * to determine the best way to optimize a HMM such that the optimal state
 * sequence is maximized, given the series of observations.
 *
 * In essence, Segmented k-Means tells us how to get an HMM that maximizes
 * P(O, I | lambda).  Viterbi gives us I.
 */
class ViterbiCalculator : public Calculator
{
private:
    Matrix delta_;
    Matrix psi_;
    StateSequence sequence_;
    double logProbability_;

public:
    /// Constructor with HMM and observations
    /// @param hmm Pointer to the HMM (must not be null)
    /// @param observations The observation set to process
    /// @throws std::invalid_argument if hmm is null
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
