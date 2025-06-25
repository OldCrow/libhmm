#ifndef FORWARDBACKWARDCALCULATOR_H_
#define FORWARDBACKWARDCALCULATOR_H_

// Removed boost include - using custom Matrix/Vector classes
#include <cmath>

#include "libhmm/common/common.h"
#include "libhmm/calculators/calculator.h"
#include "libhmm/hmm.h"

namespace libhmm
{

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

    /// Constructor with HMM and observations
    /// @param hmm Pointer to the HMM (must not be null)
    /// @param observations The observation set to process
    /// @throws std::invalid_argument if hmm is null
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
};

}

#endif
