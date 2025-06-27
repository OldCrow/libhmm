#include "libhmm/training/scaled_baum_welch_trainer.h"
#include "libhmm/calculators/scaled_simd_forward_backward_calculator.h"
#include <iostream>
#include <cassert>

namespace libhmm
{

// Calculates xi for a given HMM and observation.  (Maybe this should be a 
// calculator?
//
// xi_t( i, j ) is the probability of being in state i at time t and
// state j at time t + 1, or better written as
// xi_t( i, j ) = P( q_t = S_i, q_(t+1) = S_j ) | O, lambda )
BasicMatrix3D<double> ScaledBaumWelchTrainer::calculateXi(const ObservationSet& observations)
{
    if (observations.size() < 2) {
        throw std::runtime_error("Observation sequence too short for xi calculation");
    }
    
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    const auto obsSize = observations.size();
BasicMatrix3D<double> xi(obsSize - 1, numStates, numStates);
    const Matrix trans = hmm_->getTrans();
    
    // Get Forward (alpha) and Backward (beta) variables
    ScaledSIMDForwardBackwardCalculator fbc(hmm_, observations);
    const auto alpha = fbc.getForwardVariables();
    const auto beta = fbc.getBackwardVariables();
    
    // Get the probability of seeing these observations
    const double probability = fbc.probability();
    if (probability <= 0.0) {
        throw std::runtime_error("Invalid probability in xi calculation");
    }
    
    for (std::size_t t = 0; t < obsSize - 1; ++t) {
        for (std::size_t i = 0; i < numStates; ++i) {
            for (std::size_t j = 0; j < numStates; ++j) {
                const double emisProb = hmm_->getProbabilityDistribution(static_cast<int>(j))->getProbability(observations(t + 1));
                const double xiValue = (alpha(t, i) * trans(i, j) * emisProb * beta(t + 1, j)) / probability;
                xi.Set(t, i, j, xiValue);
            }
        }
    }
    
    return xi;
}

// Calculates Gamma
//
// gamma_t( i ) is the probability of being in state i at time t 
// given observation sequence O and HMM lambda, or:
// gamma_t( i ) = P( q_t = S_i | O, lambda );
Matrix ScaledBaumWelchTrainer::calculateGamma(const ObservationSet& observations, const BasicMatrix3D<double>& xi)
{
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    const auto obsSize = observations.size();
    Matrix gamma(obsSize, numStates);
    clear_matrix(gamma);
    
    for (std::size_t t = 0; t < obsSize - 1; ++t) {
        for (std::size_t i = 0; i < numStates; ++i) {
            for (std::size_t j = 0; j < numStates; ++j) {
                gamma(t, i) += xi(t, i, j);
            }
        }
    }
    
    // Handle the last time step separately
    if (obsSize > 0) {
        const std::size_t lastT = obsSize - 1;
        for (std::size_t i = 0; i < numStates; ++i) {
            for (std::size_t j = 0; j < numStates; ++j) {
                if (lastT > 0) {
                    gamma(lastT, i) += xi(lastT - 1, j, i);
                }
            }
        }
    }

    return gamma;
}

void ScaledBaumWelchTrainer::validateDiscreteDistributions() const {
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    for (std::size_t i = 0; i < numStates; ++i) {
        const auto* dist = dynamic_cast<const DiscreteDistribution*>(hmm_->getProbabilityDistribution(static_cast<int>(i)));
        if (!dist) {
            throw std::runtime_error("ScaledBaumWelchTrainer requires all states to have DiscreteDistribution");
        }
    }
}

// Uses the scaled Baum Welch algorithm to train an HMM.
void ScaledBaumWelchTrainer::train() {
    validateDiscreteDistributions();
    
    // HMM properties
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    const auto* firstDist = dynamic_cast<const DiscreteDistribution*>(hmm_->getProbabilityDistribution(0));
    const auto numSymbols = firstDist->getNumSymbols();

    // Initialize matrices for parameter estimation
    Matrix newTrans(numStates, numStates);
    Vector newPi(numStates);
    
    // Zero the matrices and vectors
    clear_matrix(newTrans);
    clear_vector(newPi);
    
    // Update transition matrix using scaled Baum-Welch re-estimation
    const Matrix currentTrans = hmm_->getTrans();
    for (std::size_t i = 0; i < numStates; ++i) {
        for (std::size_t j = 0; j < numStates; ++j) {
            double numerator = 0.0;
            double denominator = 0.0;

            for (const auto& observations : obsLists_) {
                if (observations.size() < 2) continue; // Skip too-short sequences
                
                ScaledSIMDForwardBackwardCalculator fbc(hmm_, observations);
                const auto alpha = fbc.getForwardVariables();
                const auto beta = fbc.getBackwardVariables();
                const double logProb = fbc.getLogProbability();
                
                if (std::isnan(logProb) || std::isinf(logProb)) continue; // Skip invalid sequences

                double xiSum = 0.0;
                double gammaSum = 0.0;
                
                for (std::size_t t = 0; t < observations.size() - 1; ++t) {
                    const double emisProb = hmm_->getProbabilityDistribution(static_cast<int>(j))->getProbability(observations(t + 1));
                    xiSum += alpha(t, i) * currentTrans(i, j) * emisProb * beta(t + 1, j);
                    gammaSum += alpha(t, i) * beta(t, i);
                }
                
                if (!std::isnan(xiSum) && !std::isinf(xiSum)) {
                    numerator += xiSum;
                }
                if (!std::isnan(gammaSum) && !std::isinf(gammaSum)) {
                    denominator += gammaSum;
                }
            }

            if (denominator > ZERO) {
                newTrans(i, j) = numerator / denominator;
            } else {
                newTrans(i, j) = ZERO;
            }
            
            if (std::isnan(newTrans(i, j)) || std::isinf(newTrans(i, j))) {
                newTrans(i, j) = ZERO;
            }
        }
    }

    // Update emission probabilities for discrete distributions
    for (std::size_t j = 0; j < numStates; ++j) {
        auto* dist = dynamic_cast<DiscreteDistribution*>(hmm_->getProbabilityDistribution(static_cast<int>(j)));
        
        for (std::size_t symbol = 0; symbol < numSymbols; ++symbol) {
            double numerator = 0.0;
            double denominator = 0.0;

            for (const auto& observations : obsLists_) {
                ScaledSIMDForwardBackwardCalculator fbc(hmm_, observations);
                const auto alpha = fbc.getForwardVariables();
                const auto beta = fbc.getBackwardVariables();
                const double logProb = fbc.getLogProbability();
                
                if (std::isnan(logProb) || std::isinf(logProb)) continue;

                double symbolSum = 0.0;
                double totalSum = 0.0;
                
                for (std::size_t t = 0; t < observations.size(); ++t) {
                    const double gamma = alpha(t, j) * beta(t, j);
                    if (static_cast<std::size_t>(observations(t)) == symbol) {
                        symbolSum += gamma;
                    }
                    totalSum += gamma;
                }
                
                if (!std::isnan(symbolSum) && !std::isinf(symbolSum)) {
                    numerator += symbolSum;
                }
                if (!std::isnan(totalSum) && !std::isinf(totalSum)) {
                    denominator += totalSum;
                }
            }

            if (denominator > ZERO) {
                const double prob = numerator / denominator;
                if (!std::isnan(prob) && !std::isinf(prob) && prob >= 0.0 && prob <= 1.0) {
                    dist->setProbability(static_cast<int>(symbol), prob);
                } else {
                    dist->setProbability(static_cast<int>(symbol), 1.0 / static_cast<double>(numSymbols));
                }
            } else {
                dist->setProbability(static_cast<int>(symbol), 1.0 / static_cast<double>(numSymbols));
            }
        }
    }

    // Update initial state probabilities (pi vector)
    // Note: Some implementations keep pi fixed as in Rabiner, but we'll update it here
    for (std::size_t i = 0; i < numStates; ++i) {
        double piSum = 0.0;
        
        for (const auto& observations : obsLists_) {
            if (observations.empty()) continue;
            
            ScaledSIMDForwardBackwardCalculator fbc(hmm_, observations);
            const auto alpha = fbc.getForwardVariables();
            const auto beta = fbc.getBackwardVariables();
            const double logProb = fbc.getLogProbability();
            
            if (!std::isnan(logProb) && !std::isinf(logProb)) {
                const double gamma = alpha(0, i) * beta(0, i);
                if (!std::isnan(gamma) && !std::isinf(gamma)) {
                    piSum += gamma;
                }
            }
        }
        
        newPi(i) = piSum / static_cast<double>(obsLists_.size());
        if (std::isnan(newPi(i)) || std::isinf(newPi(i))) {
            newPi(i) = 1.0 / static_cast<double>(numStates);
        }
    }
    
    // Apply updates to the HMM
    hmm_->setTrans(newTrans);
    hmm_->setPi(newPi);

}

} // namespace
