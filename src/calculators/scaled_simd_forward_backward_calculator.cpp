#include "libhmm/calculators/scaled_simd_forward_backward_calculator.h"
#include "libhmm/common/common.h"
#include "libhmm/hmm.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace libhmm {

ScaledSIMDForwardBackwardCalculator::ScaledSIMDForwardBackwardCalculator(
    Hmm* hmm, const ObservationSet& observations, bool useBlocking, std::size_t blockSize)
    : Calculator(hmm, observations),
      probability_(0.0),
      logProbability_(0.0),
      numStates_(static_cast<std::size_t>(hmm->getNumStates())),
      seqLength_(observations.size()),
      useBlockedComputation_(useBlocking),
      blockSize_(blockSize) {
    
    if (!hmm) {
        throw std::invalid_argument("HMM cannot be null");
    }
    
    if (observations.empty()) {
        throw std::invalid_argument("Observation sequence cannot be empty");
    }
    
    // Calculate aligned size for SIMD operations (must be multiple of 4 for AVX)
    alignedStateSize_ = ((numStates_ + 3) / 4) * 4;
    
    // Auto-detect block size if not specified
    if (blockSize_ == 0) {
        blockSize_ = getRecommendedBlockSize(numStates_);
    }
    
    // Disable blocking for small matrices
    if (numStates_ <= blockSize_) {
        useBlockedComputation_ = false;
    }
    
    // Initialize matrices
    initializeMatrices();
    
    // Run the computation
    compute();
}

void ScaledSIMDForwardBackwardCalculator::initializeMatrices() {
    // Initialize forward and backward variables
    forwardVariables_.resize(seqLength_ * alignedStateSize_, 0.0);
    backwardVariables_.resize(seqLength_ * alignedStateSize_, 0.0);
    scalingFactors_.resize(seqLength_, 1.0);
    
    // Initialize temp vectors for SIMD alignment
    tempEmissions_.resize(alignedStateSize_, 0.0);
    tempProbs_.resize(alignedStateSize_, 0.0);
}

void ScaledSIMDForwardBackwardCalculator::compute() {
    computeForward();
    computeBackward();
    
    // Compute final probability from scaling factors
    logProbability_ = 0.0;
    for (std::size_t t = 0; t < seqLength_; ++t) {
        if (scalingFactors_[t] > 0.0) {
            logProbability_ += std::log(scalingFactors_[t]);
        }
    }
    
    probability_ = std::exp(logProbability_);
}

void ScaledSIMDForwardBackwardCalculator::computeForward() {
    // Initialize first time step
    initializeForwardStep();
    
    // Forward pass
    for (std::size_t t = 1; t < seqLength_; ++t) {
        if (performance::simd_available() && numStates_ >= 4) {
            computeForwardStepSIMD(t);
        } else {
            computeForwardStepScalar(t);
        }
    }
}

void ScaledSIMDForwardBackwardCalculator::initializeForwardStep() {
    const Vector& pi = hmm_->getPi();
    
    // Compute emission probabilities for first observation
    computeEmissionProbabilities(observations_(0), tempEmissions_.data());
    
    // Initialize: alpha(0, i) = pi(i) * b_i(O_0)
    double scaleFactor = 0.0;
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double value = pi(i) * tempEmissions_[i];
        forwardVariables_[i] = value;
        scaleFactor += value;
    }
    
    // Apply scaling
    if (scaleFactor > SCALING_THRESHOLD) {
        scalingFactors_[0] = scaleFactor;
        const double invScale = 1.0 / scaleFactor;
        for (std::size_t i = 0; i < numStates_; ++i) {
            forwardVariables_[i] *= invScale;
        }
    }
    
    // Zero-pad for alignment
    for (std::size_t i = numStates_; i < alignedStateSize_; ++i) {
        forwardVariables_[i] = 0.0;
    }
}

void ScaledSIMDForwardBackwardCalculator::computeForwardStepSIMD(std::size_t t) {
    // Fallback to scalar for now - SIMD implementation would go here
    computeForwardStepScalar(t);
}

void ScaledSIMDForwardBackwardCalculator::computeForwardStepScalar(std::size_t t) {
    const Matrix& trans = hmm_->getTrans();
    const std::size_t prevIdx = (t - 1) * alignedStateSize_;
    const std::size_t currIdx = t * alignedStateSize_;
    
    // Compute emission probabilities for current observation
    computeEmissionProbabilities(observations_(t), tempEmissions_.data());
    
    // Forward step: alpha(t, j) = b_j(O_t) * sum_i(alpha(t-1, i) * trans(i, j))
    double scaleFactor = 0.0;
    for (std::size_t j = 0; j < numStates_; ++j) {
        double sum = 0.0;
        
        for (std::size_t i = 0; i < numStates_; ++i) {
            sum += forwardVariables_[prevIdx + i] * trans(i, j);
        }
        
        const double value = tempEmissions_[j] * sum;
        forwardVariables_[currIdx + j] = value;
        scaleFactor += value;
    }
    
    // Apply scaling
    if (scaleFactor > SCALING_THRESHOLD) {
        scalingFactors_[t] = scaleFactor;
        const double invScale = 1.0 / scaleFactor;
        for (std::size_t j = 0; j < numStates_; ++j) {
            forwardVariables_[currIdx + j] *= invScale;
        }
    }
    
    // Zero-pad for alignment
    for (std::size_t j = numStates_; j < alignedStateSize_; ++j) {
        forwardVariables_[currIdx + j] = 0.0;
    }
}

void ScaledSIMDForwardBackwardCalculator::computeBackward() {
    // Initialize last time step
    initializeBackwardStep();
    
    // Backward pass
    for (std::size_t t = seqLength_ - 1; t > 0; --t) {
        if (performance::simd_available() && numStates_ >= 4) {
            computeBackwardStepSIMD(t - 1);
        } else {
            computeBackwardStepScalar(t - 1);
        }
    }
}

void ScaledSIMDForwardBackwardCalculator::initializeBackwardStep() {
    const std::size_t lastIdx = (seqLength_ - 1) * alignedStateSize_;
    
    // Initialize: beta(T-1, i) = 1.0 / scaling_factor(T-1)
    const double invScale = (scalingFactors_[seqLength_ - 1] > 0.0) ? 
                           1.0 / scalingFactors_[seqLength_ - 1] : 1.0;
    
    for (std::size_t i = 0; i < numStates_; ++i) {
        backwardVariables_[lastIdx + i] = invScale;
    }
    
    // Zero-pad for alignment
    for (std::size_t i = numStates_; i < alignedStateSize_; ++i) {
        backwardVariables_[lastIdx + i] = 0.0;
    }
}

void ScaledSIMDForwardBackwardCalculator::computeBackwardStepSIMD(std::size_t t) {
    // Fallback to scalar for now - SIMD implementation would go here
    computeBackwardStepScalar(t);
}

void ScaledSIMDForwardBackwardCalculator::computeBackwardStepScalar(std::size_t t) {
    const Matrix& trans = hmm_->getTrans();
    const std::size_t currIdx = t * alignedStateSize_;
    const std::size_t nextIdx = (t + 1) * alignedStateSize_;
    
    // Compute emission probabilities for next observation
    computeEmissionProbabilities(observations_(t + 1), tempEmissions_.data());
    
    // Backward step: beta(t, i) = (1/c(t)) * sum_j(trans(i, j) * b_j(O_{t+1}) * beta(t+1, j))
    for (std::size_t i = 0; i < numStates_; ++i) {
        double sum = 0.0;
        
        for (std::size_t j = 0; j < numStates_; ++j) {
            sum += trans(i, j) * tempEmissions_[j] * backwardVariables_[nextIdx + j];
        }
        
        // Apply scaling
        const double invScale = (scalingFactors_[t] > 0.0) ? 1.0 / scalingFactors_[t] : 1.0;
        backwardVariables_[currIdx + i] = sum * invScale;
    }
    
    // Zero-pad for alignment
    for (std::size_t i = numStates_; i < alignedStateSize_; ++i) {
        backwardVariables_[currIdx + i] = 0.0;
    }
}

void ScaledSIMDForwardBackwardCalculator::computeEmissionProbabilities(
    Observation observation, double* emisProbs) const {
    
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double prob = hmm_->getProbabilityDistribution(static_cast<int>(i))
                               ->getProbability(observation);
        emisProbs[i] = prob;
    }
    
    // Zero-pad for alignment
    for (std::size_t i = numStates_; i < alignedStateSize_; ++i) {
        emisProbs[i] = 0.0;
    }
}


OptimizedMatrix<double> ScaledSIMDForwardBackwardCalculator::getForwardVariables() const {
    OptimizedMatrix<double> result(seqLength_, numStates_);
    for (std::size_t t = 0; t < seqLength_; ++t) {
        for (std::size_t i = 0; i < numStates_; ++i) {
            result(t, i) = forwardVariables_[t * alignedStateSize_ + i];
        }
    }
    return result;
}

OptimizedMatrix<double> ScaledSIMDForwardBackwardCalculator::getBackwardVariables() const {
    OptimizedMatrix<double> result(seqLength_, numStates_);
    for (std::size_t t = 0; t < seqLength_; ++t) {
        for (std::size_t i = 0; i < numStates_; ++i) {
            result(t, i) = backwardVariables_[t * alignedStateSize_ + i];
        }
    }
    return result;
}

// Compatibility methods for basic Matrix interface
Matrix ScaledSIMDForwardBackwardCalculator::getForwardVariablesCompat() const {
    Matrix result(seqLength_, numStates_);
    for (std::size_t t = 0; t < seqLength_; ++t) {
        for (std::size_t i = 0; i < numStates_; ++i) {
            result(t, i) = forwardVariables_[t * alignedStateSize_ + i];
        }
    }
    return result;
}

Matrix ScaledSIMDForwardBackwardCalculator::getBackwardVariablesCompat() const {
    Matrix result(seqLength_, numStates_);
    for (std::size_t t = 0; t < seqLength_; ++t) {
        for (std::size_t i = 0; i < numStates_; ++i) {
            result(t, i) = backwardVariables_[t * alignedStateSize_ + i];
        }
    }
    return result;
}

std::string ScaledSIMDForwardBackwardCalculator::getOptimizationInfo() const {
    std::string info = "ScaledSIMDForwardBackwardCalculator using: ";
    
    if (performance::simd_available()) {
        info += "SIMD + Scaling";
    } else {
        info += "Scalar + Scaling";
    }
    
    info += ", States: " + std::to_string(numStates_);
    info += ", Sequence length: " + std::to_string(seqLength_);
    info += ", Aligned size: " + std::to_string(alignedStateSize_);
    
    return info;
}

std::size_t ScaledSIMDForwardBackwardCalculator::getRecommendedBlockSize(std::size_t numStates) noexcept {
    // Simple heuristic for block size
    if (numStates <= 64) return numStates;
    if (numStates <= 256) return 64;
    return 128;
}

} // namespace libhmm
