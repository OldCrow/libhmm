#ifndef BAUMWELCHTRAINER_H_
#define BAUMWELCHTRAINER_H_

#include <cmath>
#include "libhmm/common/common.h"
#include "libhmm/training/hmm_trainer.h"
#include "libhmm/common/basic_matrix3d.h"
#include "libhmm/distributions/distributions.h"
#include "libhmm/calculators/forward_backward_calculator.h"

namespace libhmm
{

/// Baum-Welch training algorithm implementation
/// Uses the forward-backward algorithm to iteratively improve HMM parameters
class BaumWelchTrainer : public HmmTrainer
{
public:
    /// Constructor with HMM and observation lists
    /// @param hmm Pointer to the HMM to train (must not be null)
    /// @param obsLists List of observation sequences for training
    /// @throws std::invalid_argument if hmm is null or obsLists is empty
    BaumWelchTrainer(Hmm* hmm, const ObservationLists& obsLists)
        : HmmTrainer(hmm, obsLists) {
    }
    
    /// Virtual destructor
    virtual ~BaumWelchTrainer() = default;
    
    /// Execute Baum-Welch training algorithm
    /// Updates HMM parameters to maximize likelihood of observation sequences
    virtual void train() override;

private:
    /// Calculates xi (transition probabilities) 3D matrix
    /// @param observations Single observation sequence
    /// @return 3D matrix of xi values
    /// @throws std::runtime_error if calculation fails
    BasicMatrix3D<double> calculateXi(const ObservationSet& observations);
    
    /// Calculates gamma (state probabilities) matrix
    /// @param observations Single observation sequence
    /// @param xi Precomputed xi matrix
    /// @return Matrix of gamma values
    Matrix calculateGamma(const ObservationSet& observations, const BasicMatrix3D<double>& xi);
    
    /// Validates that the HMM has discrete distributions for training
    /// @throws std::runtime_error if HMM doesn't have discrete distributions
    void validateDiscreteDistributions() const;
};

}

#endif /*BAUMWELCHTRAINER_H_*/
