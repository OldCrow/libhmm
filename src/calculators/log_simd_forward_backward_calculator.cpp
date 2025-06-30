#include "libhmm/calculators/log_simd_forward_backward_calculator.h"
#include "libhmm/common/common.h"
#include "libhmm/performance/parallel_constants.h"
#include "libhmm/hmm.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace libhmm {

LogSIMDForwardBackwardCalculator::LogSIMDForwardBackwardCalculator(
    Hmm* hmm, const ObservationSet& observations, bool useBlocking, std::size_t blockSize)
    : Calculator(hmm, observations),
      logProbability_(LOGZERO),
      numStates_(static_cast<std::size_t>(hmm->getNumStates())),
      seqLength_(observations.size()),
      useBlockedComputation_(useBlocking),
      blockSize_(blockSize) {
    
    // Initialize matrices
    initializeMatrices();
    
    // Run the computation
    compute();
}

void LogSIMDForwardBackwardCalculator::initializeMatrices() {
    // Initialize log forward and backward variables
    logForwardVariables_.resize(seqLength_ * numStates_, LOGZERO);
    logBackwardVariables_.resize(seqLength_ * numStates_, LOGZERO);
    
    // Initialize temp vectors for SIMD alignment
    tempLogEmissions_.resize(numStates_, LOGZERO);
    tempLogProbs_.resize(numStates_, LOGZERO);
}

void LogSIMDForwardBackwardCalculator::compute() {
    computeLogForward();
    computeLogBackward();
    computeFinalLogProbability();
}

void LogSIMDForwardBackwardCalculator::computeLogForward() {
    // Initialize first time step
    initializeLogForwardStep();
    
    // Forward pass
    for (std::size_t t = 1; t < seqLength_; ++t) {
        if (performance::simd_available()) {
            computeLogForwardStepSIMD(t);
        } else {
            computeLogForwardStepScalar(t);
        }
    }
}

void LogSIMDForwardBackwardCalculator::initializeLogForwardStep() {
    const Vector& pi = hmm_->getPi();
    
    // Compute emission probabilities for first observation
    computeLogEmissionProbabilities(observations_(0), tempLogEmissions_.data());
    
    // Initialize: log_alpha(0, i) = log_pi(i) + log_b_i(O_0)
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double logPi = (pi(i) > 0.0) ? std::log(pi(i)) : LOGZERO;
        logForwardVariables_[i] = logPi + tempLogEmissions_[i];
    }
}

void LogSIMDForwardBackwardCalculator::computeLogForwardStepSIMD(std::size_t t) {
    // Fallback to scalar for now - SIMD implementation would go here
    computeLogForwardStepScalar(t);
}

void LogSIMDForwardBackwardCalculator::computeLogForwardStepScalar(std::size_t t) {
    const Matrix& trans = hmm_->getTrans();
    const std::size_t prevIdx = (t - 1) * numStates_;
    const std::size_t currIdx = t * numStates_;
    
    // Compute log emission probabilities for current observation
    computeLogEmissionProbabilities(observations_(t), tempLogEmissions_.data());
    
    // Log forward step: log_alpha(t, j) = log_b_j(O_t) + elnsum_i(log_alpha(t-1, i) + log_trans(i, j))
    // Use parallel computation for large state spaces
    if (numStates_ >= performance::parallel::MIN_STATES_FOR_CALCULATOR_PARALLEL) {
        // Parallel computation of forward variables
        performance::ParallelUtils::parallelFor(0, numStates_, [&](std::size_t j) {
            double logSum = LOGZERO;
            
            for (std::size_t i = 0; i < numStates_; ++i) {
                const double logTrans = (trans(i, j) > 0.0) ? std::log(trans(i, j)) : LOGZERO;
                const double logAlpha = logForwardVariables_[prevIdx + i];
                
                if (!std::isnan(logTrans) && !std::isnan(logAlpha)) {
                    const double logProduct = logAlpha + logTrans;
                    
                    if (std::isnan(logSum)) {
                        logSum = logProduct;
                    } else {
                        // Stabilized Log-sum-exp: log(exp(a) + exp(b)), refactored for numerical stability 
                        double maxVal = std::max(logSum, logProduct);
                        if (maxVal > LOG_MIN_PROBABILITY) {
                            logSum = maxVal + std::log(std::exp(logSum - maxVal) + std::exp(logProduct - maxVal));
                        } else {
                            logSum = maxVal; // Avoid unnecessary computations
                        }
                    }
                }
            }
            
            logForwardVariables_[currIdx + j] = tempLogEmissions_[j] + logSum;
        }, performance::parallel::CALCULATOR_GRAIN_SIZE); // Larger grain size to reduce overhead
    } else {
        // Sequential computation for smaller state spaces
        for (std::size_t j = 0; j < numStates_; ++j) {
            double logSum = LOGZERO;
            
            for (std::size_t i = 0; i < numStates_; ++i) {
                const double logTrans = (trans(i, j) > 0.0) ? std::log(trans(i, j)) : LOGZERO;
                const double logAlpha = logForwardVariables_[prevIdx + i];
                
                if (!std::isnan(logTrans) && !std::isnan(logAlpha)) {
                    const double logProduct = logAlpha + logTrans;
                    
                    if (std::isnan(logSum)) {
                        logSum = logProduct;
                    } else {
                        // Log-sum-exp: log(exp(a) + exp(b))
                        if (logSum > logProduct) {
                            logSum = logSum + std::log(1.0 + std::exp(logProduct - logSum));
                        } else {
                            logSum = logProduct + std::log(1.0 + std::exp(logSum - logProduct));
                        }
                    }
                }
            }
            
            logForwardVariables_[currIdx + j] = tempLogEmissions_[j] + logSum;
        }
    }
}

void LogSIMDForwardBackwardCalculator::computeLogBackward() {
    // Initialize last time step
    initializeLogBackwardStep();
    
    // Backward pass
    for (std::size_t t = seqLength_ - 1; t > 0; --t) {
        if (performance::simd_available()) {
            computeLogBackwardStepSIMD(t - 1);
        } else {
            computeLogBackwardStepScalar(t - 1);
        }
    }
}

void LogSIMDForwardBackwardCalculator::initializeLogBackwardStep() {
    const std::size_t lastIdx = (seqLength_ - 1) * numStates_;
    
    // Initialize: log_beta(T-1, i) = log(1.0) = 0.0
    for (std::size_t i = 0; i < numStates_; ++i) {
        logBackwardVariables_[lastIdx + i] = 0.0;
    }
}

void LogSIMDForwardBackwardCalculator::computeLogBackwardStepSIMD(std::size_t t) {
    // Fallback to scalar for now - SIMD implementation would go here
    computeLogBackwardStepScalar(t);
}

void LogSIMDForwardBackwardCalculator::computeLogBackwardStepScalar(std::size_t t) {
    const Matrix& trans = hmm_->getTrans();
    const std::size_t currIdx = t * numStates_;
    const std::size_t nextIdx = (t + 1) * numStates_;
    
    // Compute emission probabilities for next observation
    computeLogEmissionProbabilities(observations_(t + 1), tempLogEmissions_.data());
    
    // Backward step: log_beta(t, i) = elnsum_j(log_trans(i, j) + log_b_j(O_{t+1}) + log_beta(t+1, j))
    // Use parallel computation for large state spaces
    if (numStates_ >= performance::parallel::MIN_STATES_FOR_CALCULATOR_PARALLEL) {
        // Parallel computation of log backward variables
        performance::ParallelUtils::parallelFor(0, numStates_, [&](std::size_t i) {
            double logSum = LOGZERO;
            
            for (std::size_t j = 0; j < numStates_; ++j) {
                const double logTrans = (trans(i, j) > 0.0) ? std::log(trans(i, j)) : LOGZERO;
                const double logEmis = tempLogEmissions_[j];
                const double logNext = logBackwardVariables_[nextIdx + j];
                
                if (!std::isnan(logTrans) && !std::isnan(logEmis) && !std::isnan(logNext)) {
                    const double logProduct = logTrans + logEmis + logNext;
                    
                    if (std::isnan(logSum)) {
                        logSum = logProduct;
                    } else {
                        // Log-sum-exp: log(exp(a) + exp(b))
                        if (logSum > logProduct) {
                            logSum = logSum + std::log(1.0 + std::exp(logProduct - logSum));
                        } else {
                            logSum = logProduct + std::log(1.0 + std::exp(logSum - logProduct));
                        }
                    }
                }
            }
            
            logBackwardVariables_[currIdx + i] = logSum;
        }, performance::parallel::CALCULATOR_GRAIN_SIZE); // Larger grain size to reduce overhead
    } else {
        // Sequential computation for smaller state spaces
        for (std::size_t i = 0; i < numStates_; ++i) {
            double logSum = LOGZERO;
            
            for (std::size_t j = 0; j < numStates_; ++j) {
                const double logTrans = (trans(i, j) > 0.0) ? std::log(trans(i, j)) : LOGZERO;
                const double logEmis = tempLogEmissions_[j];
                const double logNext = logBackwardVariables_[nextIdx + j];
                
                if (!std::isnan(logTrans) && !std::isnan(logEmis) && !std::isnan(logNext)) {
                    const double logProduct = logTrans + logEmis + logNext;
                    
                    if (std::isnan(logSum)) {
                        logSum = logProduct;
                    } else {
                        // Log-sum-exp: log(exp(a) + exp(b))
                        if (logSum > logProduct) {
                            logSum = logSum + std::log(1.0 + std::exp(logProduct - logSum));
                        } else {
                            logSum = logProduct + std::log(1.0 + std::exp(logSum - logProduct));
                        }
                    }
                }
            }
            
            logBackwardVariables_[currIdx + i] = logSum;
        }
    }
}

void LogSIMDForwardBackwardCalculator::computeFinalLogProbability() {
    const std::size_t lastIdx = (seqLength_ - 1) * numStates_;
    
    // Sum the last forward variables in log space
    logProbability_ = LOGZERO;
    
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double logAlpha = logForwardVariables_[lastIdx + i];
        
        if (!std::isnan(logAlpha)) {
            if (std::isnan(logProbability_)) {
                logProbability_ = logAlpha;
            } else {
                // Log-sum-exp
                if (logProbability_ > logAlpha) {
                    logProbability_ = logProbability_ + std::log(1.0 + std::exp(logAlpha - logProbability_));
                } else {
                    logProbability_ = logAlpha + std::log(1.0 + std::exp(logProbability_ - logAlpha));
                }
            }
        }
    }
}

void LogSIMDForwardBackwardCalculator::computeLogEmissionProbabilities(Observation observation, double* logEmisProbs) const {
    // Use parallel computation for large state spaces to improve emission probability calculation
    // Use same threshold as main computation for consistency
    if (numStates_ >= performance::parallel::MIN_STATES_FOR_CALCULATOR_PARALLEL) {
        // Parallel emission probability computation
        performance::ParallelUtils::parallelFor(0, numStates_, [&](std::size_t i) {
            logEmisProbs[i] = hmm_->getProbabilityDistribution(static_cast<int>(i))
                                 ->getLogProbability(observation);
        }, performance::parallel::CALCULATOR_GRAIN_SIZE); // Same grain size as main computation
    } else {
        // Sequential computation for smaller state spaces
        for (std::size_t i = 0; i < numStates_; ++i) {
            logEmisProbs[i] = hmm_->getProbabilityDistribution(static_cast<int>(i))
                                 ->getLogProbability(observation);
        }
    }
}


OptimizedMatrix<double> LogSIMDForwardBackwardCalculator::getLogForwardVariables() const {
    OptimizedMatrix<double> result(seqLength_, numStates_);
    for (std::size_t t = 0; t < seqLength_; ++t) {
        for (std::size_t i = 0; i < numStates_; ++i) {
            result(t, i) = logForwardVariables_[t * numStates_ + i];
        }
    }
    return result;
}

OptimizedMatrix<double> LogSIMDForwardBackwardCalculator::getLogBackwardVariables() const {
    OptimizedMatrix<double> result(seqLength_, numStates_);
    for (std::size_t t = 0; t < seqLength_; ++t) {
        for (std::size_t i = 0; i < numStates_; ++i) {
            result(t, i) = logBackwardVariables_[t * numStates_ + i];
        }
    }
    return result;
}

OptimizedMatrix<double> LogSIMDForwardBackwardCalculator::getForwardVariables() const {
    OptimizedMatrix<double> result(seqLength_, numStates_);
    for (std::size_t t = 0; t < seqLength_; ++t) {
        for (std::size_t i = 0; i < numStates_; ++i) {
            const double logVal = logForwardVariables_[t * numStates_ + i];
            result(t, i) = std::isnan(logVal) ? 0.0 : std::exp(logVal);
        }
    }
    return result;
}

OptimizedMatrix<double> LogSIMDForwardBackwardCalculator::getBackwardVariables() const {
    OptimizedMatrix<double> result(seqLength_, numStates_);
    for (std::size_t t = 0; t < seqLength_; ++t) {
        for (std::size_t i = 0; i < numStates_; ++i) {
            const double logVal = logBackwardVariables_[t * numStates_ + i];
            result(t, i) = std::isnan(logVal) ? 0.0 : std::exp(logVal);
        }
    }
    return result;
}

// Compatibility methods for basic Matrix interface
Matrix LogSIMDForwardBackwardCalculator::getLogForwardVariablesCompat() const {
    Matrix result(seqLength_, numStates_);
    for (std::size_t t = 0; t < seqLength_; ++t) {
        for (std::size_t i = 0; i < numStates_; ++i) {
            result(t, i) = logForwardVariables_[t * numStates_ + i];
        }
    }
    return result;
}

Matrix LogSIMDForwardBackwardCalculator::getLogBackwardVariablesCompat() const {
    Matrix result(seqLength_, numStates_);
    for (std::size_t t = 0; t < seqLength_; ++t) {
        for (std::size_t i = 0; i < numStates_; ++i) {
            result(t, i) = logBackwardVariables_[t * numStates_ + i];
        }
    }
    return result;
}

Matrix LogSIMDForwardBackwardCalculator::getForwardVariablesCompat() const {
    Matrix result(seqLength_, numStates_);
    for (std::size_t t = 0; t < seqLength_; ++t) {
        for (std::size_t i = 0; i < numStates_; ++i) {
            const double logVal = logForwardVariables_[t * numStates_ + i];
            result(t, i) = std::isnan(logVal) ? 0.0 : std::exp(logVal);
        }
    }
    return result;
}

Matrix LogSIMDForwardBackwardCalculator::getBackwardVariablesCompat() const {
    Matrix result(seqLength_, numStates_);
    for (std::size_t t = 0; t < seqLength_; ++t) {
        for (std::size_t i = 0; i < numStates_; ++i) {
            const double logVal = logBackwardVariables_[t * numStates_ + i];
            result(t, i) = std::isnan(logVal) ? 0.0 : std::exp(logVal);
        }
    }
    return result;
}

std::string LogSIMDForwardBackwardCalculator::getOptimizationInfo() const {
    std::string info = "LogSIMDForwardBackwardCalculator using: ";
    
    if (performance::simd_available()) {
        info += "SIMD + Log-space";
    } else {
        info += "Scalar Log-space";
    }
    
    info += ", States: " + std::to_string(numStates_);
    info += ", Sequence length: " + std::to_string(seqLength_);
    
    return info;
}

std::size_t LogSIMDForwardBackwardCalculator::getRecommendedBlockSize(std::size_t numStates) noexcept {
    // Simple heuristic for block size
    if (numStates <= 64) return numStates;
    if (numStates <= 256) return 64;
    return 128;
}

} // namespace libhmm
