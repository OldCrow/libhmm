#include "libhmm/calculators/log_forward_backward_calculator.h"
#include <iostream>

namespace libhmm
{

void LogForwardBackwardCalculator::forward()
{
    Matrix alpha(observations_.size(), hmm_->getNumStates());
    const Matrix trans = hmm_->getTrans();
    const Vector pi = hmm_->getPi();
    
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    const auto obsSize = observations_.size();

    // The forward algorithm
    for (std::size_t i = 0; i < numStates; ++i) {
        alpha(0, i) = elnproduct(eln(pi(i)), 
                eln(hmm_->getProbabilityDistribution(static_cast<int>(i))->getProbability(observations_(0))));
    }
    
    // Move forward through the list of observations, skipping the first one 
    // since we already took care of it.
    for (std::size_t t = 1; t < obsSize; ++t) {
        for (std::size_t j = 0; j < numStates; ++j) {
            double logalpha = LOGZERO;
            for (std::size_t i = 0; i < numStates; ++i) {
                logalpha = elnsum(logalpha, elnproduct(alpha(t - 1, i), eln(trans(i, j))));
            }
            alpha(t, j) = elnproduct(logalpha, 
                    eln(hmm_->getProbabilityDistribution(static_cast<int>(j))->getProbability(observations_(t))));
        }
    }
    
    //std::cout << "    alpha: " << alpha << std::endl;

    forwardVariables_ = alpha;
}

// Implements the Backward algorithm
void LogForwardBackwardCalculator::backward() {
    Matrix beta(observations_.size(), hmm_->getNumStates());
    const Matrix trans = hmm_->getTrans();
    
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    const auto obsSize = observations_.size();

    // Zero the beta matrix
    clear_matrix(beta);

    // Rabiner defines beta_T( i ) to be 1, but does so arbitrarily and admits
    // it freely.  Mann(2006) says to start with 0 in log space.
    for (std::size_t i = 0; i < numStates; ++i) {
        beta(obsSize - 1, i) = 0.0;
    }

    // Walk backward through the observation sequence
    for (std::size_t t = obsSize - 1; t > 0; --t) {
        const std::size_t currentT = t - 1;
        for (std::size_t i = 0; i < numStates; ++i) {
            // We need to sum over all states j to get a value for state i
            beta(currentT, i) = LOGZERO;
            for (std::size_t j = 0; j < numStates; ++j) {
                beta(currentT, i) = elnsum(beta(currentT, i), elnproduct(eln(trans(i, j)), 
                    elnproduct(eln(hmm_->getProbabilityDistribution(static_cast<int>(j))->getProbability(observations_(t))),
                    beta(t, j))));
            }
        }
    }

    //std::cout << "    beta: " << beta << std::endl;
    backwardVariables_ = beta;
}

double LogForwardBackwardCalculator::probability() {
    return std::exp(logProbability());
}

double LogForwardBackwardCalculator::logProbability() {
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    const auto lastIndex = observations_.size() - 1;
    
    if (numStates == 0) {
        return LOGZERO;
    }
    
    double p = forwardVariables_(lastIndex, 0);
    if (numStates > 1) {
        p = elnsum(p, forwardVariables_(lastIndex, 1));
    }

    // Probability of the entire sequence is given by the sum of the elements 
    // in the last column
    for (std::size_t i = 2; i < numStates; ++i) {
        p = elnsum(p, forwardVariables_(lastIndex, i));
    }
    
    if (p > 1.0) {
        std::cerr << "LogForwardBackwardCalculator: Numeric Underflow occurred!" << std::endl;
        std::cerr << "Probability: " << p << std::endl;
        std::cerr << "Observations: " << observations_ << std::endl;
        std::cerr << "HMM: " << *hmm_ << std::endl;
        std::cerr << "Forward variables:" << forwardVariables_ << std::endl;
    }

    return p;    
}

}// namespace
