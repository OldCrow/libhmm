#include "libhmm/calculators/scaled_forward_backward_calculator.h"
#include <cmath>

namespace libhmm
{

void ScaledForwardBackwardCalculator::forward()
{
    
    // Scaled forward variable
    Matrix alpha(observations_.size(), hmm_->getNumStates());
    
    // Rabiner does not go into great detail about how to calculate the scaled
    // forward variables.  This *is* described in detail by Ali Rahimi at 
    // http://alumni.media.mit.edu/~rahimi/rabiner/rabiner-errata/rabiner-errata.html
    Matrix alpha_bar(observations_.size(), hmm_->getNumStates());
    
    const Matrix trans = hmm_->getTrans();
    const Vector pi = hmm_->getPi();
    
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    const auto obsSize = observations_.size();
   
    clear_matrix(alpha_bar);
    clear_matrix(alpha);
    
    // The forward algorithm: Initialization
    for (std::size_t i = 0; i < numStates; ++i) {
        alpha_bar(0, i) = pi(i) * hmm_->getProbabilityDistribution(static_cast<int>(i))->getProbability(observations_(0));
        if (alpha_bar(0, i) < ZERO) {
            alpha_bar(0, i) = ZERO;
        }
        c_(0) += alpha_bar(0, i);
    }
    
    for (std::size_t i = 0; i < numStates; ++i) {
        alpha(0, i) = alpha_bar(0, i) / c_(0);
    }
    
    // The forward algorithm: Induction
    // Move forward through the list of observations, skipping the first one 
    // since we already took care of it.
    for (std::size_t t = 1; t < obsSize; ++t) {
        for (std::size_t j = 0; j < numStates; ++j) {
            alpha_bar(t, j) = hmm_->getProbabilityDistribution(static_cast<int>(j))->getProbability(observations_(t)) * 
                             boost::numeric::ublas::inner_prod(boost::numeric::ublas::row(alpha, t - 1), boost::numeric::ublas::column(trans, j));
            if (alpha_bar(t, j) < ZERO) {
                alpha_bar(t, j) = ZERO;
            }
            c_(t) += alpha_bar(t, j);
        }
        
        for (std::size_t j = 0; j < numStates; ++j) {
            alpha(t, j) = alpha_bar(t, j) / c_(t);
        }
    }
    
    //std::cout << "    alpha_bar: " << alpha_bar << std::endl;
    //std::cout << "    c: " << c << std::endl;
    //std::cout << "    alpha: " << alpha << std::endl;

    forwardVariables_ = alpha;    
}

// Implements the Backward algorithm
// Note: This function depends on the C variable...Forward() must have 
// been run before Backward()
//  This is NOT scaled!!
void ScaledForwardBackwardCalculator::backward() {
    
    // Rabiner does not go into great detail about how to calculate the scaled
    // forward variables.  This *is* described in detail by Ali Rahimi at 
    // http://alumni.media.mit.edu/~rahimi/rabiner/rabiner-errata/rabiner-errata.html
    Matrix beta_bar(observations_.size(), hmm_->getNumStates());
    Matrix beta(observations_.size(), hmm_->getNumStates());
    const Matrix trans = hmm_->getTrans();
    
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    const auto obsSize = observations_.size();

    // Zero the beta matrix
    clear_matrix(beta_bar);
    clear_matrix(beta);
    
    // Rabiner defines beta_T( i ) to be 1, but does so arbitrarily and admits
    // it freely.
    for (std::size_t i = 0; i < numStates; ++i) {
        beta_bar(obsSize - 1, i) = 1.0;
        beta(obsSize - 1, i) = 1.0 / c_(obsSize - 1);
    }

    // Walk backward through the observation sequence
    // beta_t(i) = sum( a[i][j]*b_j( O_(t+1) )*beta_(t+1)(j), j=1..N )
    for (std::size_t t = obsSize - 1; t > 0; --t) {
        const std::size_t currentT = t - 1;
        
        // beta_bar computation
        for (std::size_t i = 0; i < numStates; ++i) {
            // We need to sum over all states j to get a value for state i
            for (std::size_t j = 0; j < numStates; ++j) {
                beta_bar(currentT, i) += trans(i, j) * 
                    hmm_->getProbabilityDistribution(static_cast<int>(j))->getProbability(observations_(t)) * 
                    beta(t, j);
            }
            beta(currentT, i) = beta_bar(currentT, i) / c_(currentT);
            //std::cout << "beta["<<currentT<<"]["<<i<<"]="<<beta(currentT,i)<<std::endl;            
        }
    }

    //std::cout << "    c: " << c << std::endl;
    //std::cout << "    beta_bar: " << beta_bar << std::endl;
    //std::cout << "    beta: " << beta << std::endl;
    backwardVariables_ = beta;
}


// Returns P( observations | hmm )
// Considering that the process is scaled, the answer here is likely to be
// different than if the ForwardBackwardCalculator was used.
// This function takes the result from LogProbability() and uses it 
// as an exponent to the base 10.
double ScaledForwardBackwardCalculator::probability()
{
    const double p = logProbability();
    return std::pow(10.0, p);
}

// Calculates the *log* of P( observations | hmm )
// Note that this is a log because its highly unlikely that this probability
// is within the range of a double.
double ScaledForwardBackwardCalculator::logProbability() {
    double p = 0.0;
    
    // Rabiner says on p. 273 that 
    // log[ P( O | lambda ) = -sum( log c_t, t = 1 .. T )
    for (std::size_t i = 0; i < observations_.size(); ++i) {
        p += std::log10(c_(i));
    }
    
    return p;
}

} // namespace
