#include "libhmm/calculators/scaled_simd_viterbi_calculator.h"
#include "libhmm/common/common.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>

#ifdef LIBHMM_HAS_AVX
#include <immintrin.h>
#elif defined(LIBHMM_HAS_SSE2)
#include <emmintrin.h>
#endif

namespace libhmm {

ScaledSIMDViterbiCalculator::ScaledSIMDViterbiCalculator(Hmm* hmm, const ObservationSet& observations)
    : Calculator(hmm, observations),
      logProbability_(0.0),
      numStates_(static_cast<std::size_t>(hmm->getNumStates())),
      seqLength_(observations.size()) {
    
    if (!hmm) {
        throw std::invalid_argument("HMM cannot be null");
    }
    
    if (observations.empty()) {
        throw std::invalid_argument("Observation sequence cannot be empty");
    }
    
    // Initialize matrices with proper alignment for SIMD
    const std::size_t matrixSize = seqLength_ * numStates_;
    
    // Ensure SIMD-aligned allocation
    const std::size_t alignedSize = performance::cache_aligned_size<double>(matrixSize);
    
    delta_.resize(alignedSize, 0.0);
    scaledDelta_.resize(alignedSize, 0.0);
    psi_.resize(matrixSize, 0);
    scalingFactors_.resize(seqLength_, 1.0);
    sequence_.resize(seqLength_, 0);
    
    // Temporary vectors for SIMD computations
    const std::size_t tempSize = performance::cache_aligned_size<double>(numStates_);
    tempScores_.resize(tempSize, 0.0);
    tempProbs_.resize(tempSize, 0.0);
}

StateSequence ScaledSIMDViterbiCalculator::decode() {
    try {
        // Step 1: Initialize with first observation
        initializeFirstStep();
        
        // Step 2: Forward pass with scaling and SIMD optimization
        for (std::size_t t = 1; t < seqLength_; ++t) {
            if (performance::simd_available() && numStates_ >= 4) {
                computeForwardStepSIMD(t);
            } else {
                computeForwardStepScalar(t);
            }
            
            // Apply scaling if needed
            if (needsScaling(t)) {
                scalingFactors_[t] = applyScaling(t);
            }
        }
        
        // Step 3: Termination
        computeTermination();
        
        // Step 4: Backtrack
        backtrackPath();
        
        // Step 5: Reconstruct log probability
        reconstructLogProbability();
        
        return sequence_;
        
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("ScaledSIMDViterbiCalculator::decode failed: ") + e.what());
    }
}

void ScaledSIMDViterbiCalculator::initializeFirstStep() {
    const Vector& pi = hmm_->getPi();
    const Observation firstObs = observations_(0);
    
    // Compute emission probabilities for all states
    computeEmissionProbabilities(firstObs, tempProbs_.data());
    
    double maxValue = -DBL_MAX;
    
    // Initialize delta and find maximum for numerical stability
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double piProb = pi(i);
        const double emisProb = tempProbs_[i];
        
        // Use log probabilities to avoid underflow
        const double logPi = (piProb > ZERO) ? std::log(piProb) : std::log(ZERO);
        const double logEmis = (emisProb > ZERO) ? std::log(emisProb) : std::log(ZERO);
        
        delta_[getMatrixIndex(0, i)] = logPi + logEmis;
        psi_[getMatrixIndex(0, i)] = 0; // No previous state
        
        maxValue = std::max(maxValue, delta_[getMatrixIndex(0, i)]);
    }
    
    // Apply initial scaling if needed
    if (maxValue < LOG_SCALING_THRESHOLD) {
        scalingFactors_[0] = applyScaling(0);
    }
}

void ScaledSIMDViterbiCalculator::computeForwardStepSIMD(std::size_t t) {
    const Matrix& trans = hmm_->getTrans();
    const Observation obs = observations_(t);
    
    // Compute emission probabilities for all states
    computeEmissionProbabilities(obs, tempProbs_.data());
    
    // Prefetch the previous delta values
    const double* prevDelta = &delta_[getMatrixIndex(t - 1, 0)];
    
    // Process states - for now use scalar logic but with SIMD-optimized finding
    for (std::size_t j = 0; j < numStates_; ++j) {
        // Compute transition scores from all previous states to state j
        for (std::size_t i = 0; i < numStates_; ++i) {
            const double transProb = trans(i, j);
            const double logTrans = (transProb > ZERO) ? std::log(transProb) : std::log(ZERO);
            tempScores_[i] = prevDelta[i] + logTrans;
        }
        
        // Find the maximum (best) transition using SIMD-optimized search
        double maxScore = -DBL_MAX;
        std::size_t bestPrevState = 0;
        
        // Simple scalar max finding for now - SIMD version was buggy
        for (std::size_t i = 0; i < numStates_; ++i) {
            if (tempScores_[i] > maxScore) {
                maxScore = tempScores_[i];
                bestPrevState = i;
            }
        }
        
        // Add emission probability
        const double emisProb = tempProbs_[j];
        const double logEmis = (emisProb > ZERO) ? std::log(emisProb) : std::log(ZERO);
        
        delta_[getMatrixIndex(t, j)] = maxScore + logEmis;
        psi_[getMatrixIndex(t, j)] = static_cast<int>(bestPrevState);
    }
}

void ScaledSIMDViterbiCalculator::computeForwardStepScalar(std::size_t t) {
    const Matrix& trans = hmm_->getTrans();
    const Observation obs = observations_(t);
    
    // Compute emission probabilities for all states
    computeEmissionProbabilities(obs, tempProbs_.data());
    
    for (std::size_t j = 0; j < numStates_; ++j) {
        double maxScore = -DBL_MAX;
        std::size_t bestPrevState = 0;
        
        // Find best previous state
        for (std::size_t i = 0; i < numStates_; ++i) {
            const double transProb = trans(i, j);
            const double logTrans = (transProb > ZERO) ? std::log(transProb) : std::log(ZERO);
            const double score = delta_[getMatrixIndex(t - 1, i)] + logTrans;
            
            if (score > maxScore) {
                maxScore = score;
                bestPrevState = i;
            }
        }
        
        // Add emission probability
        const double emisProb = tempProbs_[j];
        const double logEmis = (emisProb > ZERO) ? std::log(emisProb) : std::log(ZERO);
        
        delta_[getMatrixIndex(t, j)] = maxScore + logEmis;
        psi_[getMatrixIndex(t, j)] = static_cast<int>(bestPrevState);
    }
}

void ScaledSIMDViterbiCalculator::computeTransitionScoresSIMD(
    std::size_t /*fromState*/,
    const double* transitionsStart,
    const double* prevDelta,
    double* results,
    std::size_t numStates) const {
    
#ifdef LIBHMM_HAS_AVX
    const std::size_t simdWidth = 4; // AVX processes 4 doubles at once
    const std::size_t simdBlocks = numStates / simdWidth;
    const std::size_t remainder = numStates % simdWidth;
    
    // Process SIMD blocks
    for (std::size_t block = 0; block < simdBlocks; ++block) {
        const std::size_t baseIdx = block * simdWidth;
        
        // Load transition probabilities (unaligned for safety)
        __m256d trans = _mm256_loadu_pd(&transitionsStart[baseIdx]);
        
        // Load previous delta values (unaligned for safety)
        __m256d delta = _mm256_loadu_pd(&prevDelta[baseIdx]);
        
        // Convert to log space (assuming small probabilities, use approximation)
        // For better accuracy, we'd need a vectorized log function
        __m256d logTrans = _mm256_set1_pd(std::log(ZERO)); // Default for zero probs
        
        // Simplified: assume non-zero probabilities for demonstration
        // In production, we'd need proper handling of log(0)
        for (int i = 0; i < 4; ++i) {
            double t = transitionsStart[baseIdx + i];
            if (t > ZERO) {
                reinterpret_cast<double*>(&logTrans)[i] = std::log(t);
            }
        }
        
        // Compute scores: delta + log(trans)
        __m256d scores = _mm256_add_pd(delta, logTrans);
        
        // Store results (unaligned for safety)
        _mm256_storeu_pd(&results[baseIdx], scores);
    }
    
    // Handle remainder elements
    for (std::size_t i = simdBlocks * simdWidth; i < numStates; ++i) {
        const double transProb = transitionsStart[i];
        const double logTrans = (transProb > ZERO) ? std::log(transProb) : std::log(ZERO);
        results[i] = prevDelta[i] + logTrans;
    }
    
#elif defined(LIBHMM_HAS_SSE2)
    const std::size_t simdWidth = 2; // SSE2 processes 2 doubles at once
    const std::size_t simdBlocks = numStates / simdWidth;
    const std::size_t remainder = numStates % simdWidth;
    
    // Process SSE2 blocks
    for (std::size_t block = 0; block < simdBlocks; ++block) {
        const std::size_t baseIdx = block * simdWidth;
        
        __m128d trans = _mm_loadu_pd(&transitionsStart[baseIdx]);
        __m128d delta = _mm_loadu_pd(&prevDelta[baseIdx]);
        
        // Simplified log computation (similar to AVX)
        __m128d logTrans = _mm_set1_pd(std::log(ZERO));
        for (int i = 0; i < 2; ++i) {
            double t = transitionsStart[baseIdx + i];
            if (t > ZERO) {
                reinterpret_cast<double*>(&logTrans)[i] = std::log(t);
            }
        }
        
        __m128d scores = _mm_add_pd(delta, logTrans);
        _mm_storeu_pd(&results[baseIdx], scores);
    }
    
    // Handle remainder
    for (std::size_t i = simdBlocks * simdWidth; i < numStates; ++i) {
        const double transProb = transitionsStart[i];
        const double logTrans = (transProb > ZERO) ? std::log(transProb) : std::log(ZERO);
        results[i] = prevDelta[i] + logTrans;
    }
    
#else
    // Fallback scalar implementation
    for (std::size_t i = 0; i < numStates; ++i) {
        const double transProb = transitionsStart[i];
        const double logTrans = (transProb > ZERO) ? std::log(transProb) : std::log(ZERO);
        results[i] = prevDelta[i] + logTrans;
    }
#endif
}

void ScaledSIMDViterbiCalculator::findMinSIMD(const double* values, std::size_t size, 
                                              double& maxValue, std::size_t& maxIndex) const {
    maxValue = -DBL_MAX;
    maxIndex = 0;
    
    // Simple scalar implementation - the SIMD version was causing issues
    // TODO: Implement proper SIMD max finding with correct alignment and logic
    for (std::size_t i = 0; i < size; ++i) {
        if (values[i] > maxValue) {
            maxValue = values[i];
            maxIndex = i;
        }
    }
}

void ScaledSIMDViterbiCalculator::computeEmissionProbabilities(Observation observation, double* emisProbs) const {
    for (std::size_t i = 0; i < numStates_; ++i) {
        emisProbs[i] = hmm_->getProbabilityDistribution(static_cast<int>(i))->getProbability(observation);
    }
}

bool ScaledSIMDViterbiCalculator::needsScaling(std::size_t t) const {
    const double* deltaRow = &delta_[getMatrixIndex(t, 0)];
    
    // Check if any value is below threshold
    for (std::size_t i = 0; i < numStates_; ++i) {
        if (deltaRow[i] < LOG_SCALING_THRESHOLD) {
            return true;
        }
    }
    return false;
}

double ScaledSIMDViterbiCalculator::applyScaling(std::size_t t) {
    double* deltaRow = &delta_[getMatrixIndex(t, 0)];
    
    // Find the maximum value for stable scaling
    double maxVal = -DBL_MAX;
    for (std::size_t i = 0; i < numStates_; ++i) {
        maxVal = std::max(maxVal, deltaRow[i]);
    }
    
    // Scale all values by subtracting the maximum
    const double scaleFactor = maxVal;
    for (std::size_t i = 0; i < numStates_; ++i) {
        deltaRow[i] -= scaleFactor;
    }
    
    return scaleFactor;
}

void ScaledSIMDViterbiCalculator::computeTermination() {
    const auto lastTimeIndex = seqLength_ - 1;
    const double* lastDelta = &delta_[getMatrixIndex(lastTimeIndex, 0)];
    
    double maxValue = -DBL_MAX;
    std::size_t bestFinalState = 0;
    
    // Find the best final state
    for (std::size_t i = 0; i < numStates_; ++i) {
        if (lastDelta[i] > maxValue) {
            maxValue = lastDelta[i];
            bestFinalState = i;
        }
    }
    
    sequence_[lastTimeIndex] = static_cast<int>(bestFinalState);
    logProbability_ = maxValue; // Will be adjusted with scaling factors
}

void ScaledSIMDViterbiCalculator::backtrackPath() {
    // Backtrack from the end to find the optimal path
    for (std::size_t t = seqLength_ - 1; t > 0; --t) {
        const int currentState = sequence_[t];
        const int prevState = psi_[getMatrixIndex(t, currentState)];
        sequence_[t - 1] = prevState;
    }
}

void ScaledSIMDViterbiCalculator::reconstructLogProbability() {
    // Add back all the scaling factors to get the true log probability
    double totalLogScaling = 0.0;
    for (std::size_t t = 0; t < seqLength_; ++t) {
        if (scalingFactors_[t] != 1.0) {
            totalLogScaling += scalingFactors_[t]; // scaling factors are already in log space
        }
    }
    
    logProbability_ += totalLogScaling;
}

} // namespace libhmm
