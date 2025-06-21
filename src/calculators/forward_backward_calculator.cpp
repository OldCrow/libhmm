#include "libhmm/calculators/forward_backward_calculator.h"
#include <iostream>

namespace libhmm
{

void ForwardBackwardCalculator::forward()
{
    Matrix alpha(observations_.size(), hmm_->getNumStates());
    const Matrix trans = hmm_->getTrans();
    const Vector pi = hmm_->getPi();
    
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    const auto obsSize = observations_.size();

    // The forward algorithm
    // Initialization step:
    //   alpha( 0, i ) = pi( i ) * b_i( O_1 )
    // or: initial probability of state i times probability that state i 
    // emitted observation 1
    for (std::size_t i = 0; i < numStates; ++i) {
        alpha(0, i) = pi(i) * hmm_->getProbabilityDistribution(static_cast<int>(i))->getProbability(observations_(0));
        if (alpha(0, i) < ZERO || std::isnan(alpha(0, i))) {
            alpha(0, i) = ZERO;
        }
    }
    
    // Move forward through the list of observations, skipping the first one 
    // since we already took care of it.
    for (std::size_t t = 1; t < obsSize; ++t) {
        for (std::size_t j = 0; j < numStates; ++j) {
            alpha(t, j) = hmm_->getProbabilityDistribution(static_cast<int>(j))->getProbability(observations_(t)) * 
                         boost::numeric::ublas::inner_prod(boost::numeric::ublas::row(alpha, t - 1), boost::numeric::ublas::column(trans, j));
            if (alpha(t, j) < ZERO || std::isnan(alpha(t, j))) {
                alpha(t, j) = ZERO;
            }
        }
    }
    
    //std::cout << "    alpha: " << alpha << std::endl;

    forwardVariables_ = alpha;
}

// Implements the Backward algorithm
//  This is NOT scaled!!
void ForwardBackwardCalculator::backward() {
    Matrix beta(observations_.size(), hmm_->getNumStates());
    const Matrix trans = hmm_->getTrans();
    
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    const auto obsSize = observations_.size();

    // Zero the beta matrix
    clear_matrix(beta);

    // Rabiner defines beta_T( i ) to be 1, but does so arbitrarily and admits
    // it freely.
    for (std::size_t i = 0; i < numStates; ++i) {
        beta(obsSize - 1, i) = 1.0;
    }

    // Walk backward through the observation sequence
    // beta_t(i) = sum( a[i][j]*b_j( O_(t+1) )*beta_(t+1)(j), j=1..N )
    for (std::size_t t = obsSize - 1; t > 0; --t) { // Use size_t and avoid underflow
        const std::size_t currentT = t - 1;
        for (std::size_t i = 0; i < numStates; ++i) {
            // We need to sum over all states j to get a value for state i
            for (std::size_t j = 0; j < numStates; ++j) {
                beta(currentT, i) += trans(i, j) * 
                    hmm_->getProbabilityDistribution(static_cast<int>(j))->getProbability(observations_(t)) * 
                    beta(t, j);
            }
        }
    }

    //std::cout << "    beta: " << beta << std::endl;
    backwardVariables_ = beta;
}

double ForwardBackwardCalculator::probability() {
    double p = 0.0;
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    const auto lastObsIndex = observations_.size() - 1;

    // Probability of the entire sequence is given by the sum of the elements 
    // in the last column
    for (std::size_t i = 0; i < numStates; ++i) {
        p += forwardVariables_(lastObsIndex, i);
    }
    
    if (p > 1.0) {
        std::cerr << "ForwardBackwardCalculator: Numeric Underflow occurred!" << std::endl;
        std::cerr << "Probability: " << p << std::endl;
        std::cerr << "Observations: " << observations_ << std::endl;
        std::cerr << "HMM: " << *hmm_ << std::endl;
        std::cerr << "Forward variables:" << forwardVariables_ << std::endl;
    }

    assert(p <= 1.0);
    return p;    
}

}// namespace
