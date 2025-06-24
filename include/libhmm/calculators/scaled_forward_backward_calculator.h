#ifndef SCALEDFORWARDBACKWARDCALCULATOR_H_
#define SCALEDFORWARDBACKWARDCALCULATOR_H_

#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <cmath>

#include "libhmm/common/common.h"
#include "libhmm/hmm.h"
#include "libhmm/calculators/forward_backward_calculator.h"

namespace libhmm
{

class ScaledForwardBackwardCalculator : public ForwardBackwardCalculator
{
protected:
    // Scaling factors
    // c is the *lowercase* c mentioned by Rahimi
    Vector c_;

protected:    
    /// Runs the forward algorithm on a set of observations...only scaled
    virtual void forward() override;

    /// Runs the backward algorithm on a set of observations...scaled
    virtual void backward() override;  
        
public:
    /// Constructor with HMM and observations
    /// @param hmm Pointer to the HMM (must not be null)
    /// @param observations The observation set to process
    /// @throws std::invalid_argument if hmm is null
    ScaledForwardBackwardCalculator(Hmm* hmm, const ObservationSet& observations)
        : ForwardBackwardCalculator(hmm, observations),
          c_(observations.size()) {
        clear_vector(c_);
        forward();
        backward();
    }
    
    /// Virtual destructor
    virtual ~ScaledForwardBackwardCalculator() = default;
        
    /// Calculates the log10 probability of the observation set given the HMM
    /// @return The log10 probability value
    virtual double logProbability();     
    
    /// Calculates the probability of the observation set given the HMM
    /// @return The probability value
    virtual double probability() override; 
};

}

#endif /*SCALEDFORWARDBACKWARDCALCULATOR_H_*/
