#ifndef SCALEDBAUMWELCHTRAINER_H_
#define SCALEDBAUMWELCHTRAINER_H_

#include <cmath>
#include "libhmm/common/common.h"
#include "libhmm/training/hmm_trainer.h"
#include "libhmm/training/matrix3d.h"
#include "libhmm/distributions/distributions.h"
#include "libhmm/calculators/scaled_forward_backward_calculator.h"

namespace libhmm
{

/// Scaled Baum-Welch training algorithm implementation
/// Uses scaled forward-backward algorithm to prevent numerical underflow
class ScaledBaumWelchTrainer : public HmmTrainer
{
public:
    /// Constructor with HMM and observation lists
    /// @param hmm Pointer to the HMM to train (must not be null)
    /// @param obsLists List of observation sequences for training
    /// @throws std::invalid_argument if hmm is null or obsLists is empty
    ScaledBaumWelchTrainer(Hmm* hmm, const ObservationLists& obsLists)
        : HmmTrainer(hmm, obsLists) {
    }
    
    /// Virtual destructor
    virtual ~ScaledBaumWelchTrainer() = default;
    
    /// Execute scaled Baum-Welch training algorithm
    /// Updates HMM parameters using numerically stable scaled computations
    virtual void train() override;

private:
    /// Calculates xi (transition probabilities) 3D matrix using scaled computation
    /// @param observations Single observation sequence
    /// @return 3D matrix of xi values
    /// @throws std::runtime_error if calculation fails
    Matrix3D<double> calculateXi(const ObservationSet& observations);
    
    /// Calculates gamma (state probabilities) matrix
    /// @param observations Single observation sequence
    /// @param xi Precomputed xi matrix
    /// @return Matrix of gamma values
    Matrix calculateGamma(const ObservationSet& observations, const Matrix3D<double>& xi);
    
    /// Validates that the HMM has discrete distributions for training
    /// @throws std::runtime_error if HMM doesn't have discrete distributions
    void validateDiscreteDistributions() const;
};

}

#endif /*SCALEDBAUMWELCHTRAINER_H_*/
