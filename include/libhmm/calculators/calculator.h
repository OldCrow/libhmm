#ifndef CALCULATOR_H_
#define CALCULATOR_H_

#include "libhmm/common/common.h"
#include "libhmm/hmm.h"
#include <stdexcept>

namespace libhmm
{

/*
 * Base class for a Calculator - an object that takes an HMM and an 
 * ObservationSet and performs *some* operation on them.
 */
class Calculator
{
protected:
    Hmm* hmm_{nullptr};
    ObservationSet observations_;
    
public:
    /// Default constructor
    Calculator() = default;
    
    /// Constructor with HMM and observations
    /// @param hmm Pointer to the HMM (must not be null)
    /// @param observations The observation set to process
    /// @throws std::invalid_argument if hmm is null
    Calculator(Hmm* hmm, const ObservationSet& observations)
        : hmm_{hmm}, observations_{observations} {
        if (!hmm) {
            throw std::invalid_argument("HMM pointer cannot be null");
        }
    }
    
    /// Virtual destructor for proper inheritance
    virtual ~Calculator() = default;
    
    /// Non-copyable but movable
    Calculator(const Calculator&) = delete;
    Calculator& operator=(const Calculator&) = delete;
    Calculator(Calculator&&) = default;
    Calculator& operator=(Calculator&&) = default;
    
    /// Get the HMM pointer
    /// @return Pointer to the HMM
    Hmm* getHmm() const noexcept { return hmm_; }
    
    /// Get the observations
    /// @return The observation set
    const ObservationSet& getObservations() const noexcept { return observations_; }
    
    /// Set new observations
    /// @param observations The new observation set
    void setObservations(const ObservationSet& observations) {
        observations_ = observations;
    }
}; //class Calculator

}//namespace

#endif //CALCULATOR_H_
