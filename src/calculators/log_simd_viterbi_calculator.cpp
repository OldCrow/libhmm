#include "libhmm/calculators/log_simd_viterbi_calculator.h"
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

LogSIMDViterbiCalculator::LogSIMDViterbiCalculator(Hmm* hmm, const ObservationSet& observations)
    : Calculator(hmm, observations),
      logProbability_(LOG_ZERO),
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
    const std::size_t alignedMatrixSize = performance::cache_aligned_size<double>(matrixSize);
    
    logDelta_.resize(alignedMatrixSize, LOG_ZERO);
    psi_.resize(matrixSize, 0);
    sequence_.resize(seqLength_, 0);
    
    // Temporary vectors for SIMD computations
    const std::size_t tempSize = performance::cache_aligned_size<double>(numStates_);
    tempLogScores_.resize(tempSize, LOG_ZERO);
    tempLogEmisProbs_.resize(tempSize, LOG_ZERO);
    
    // Pre-allocate space for transition probabilities matrix
    const std::size_t transSize = performance::cache_aligned_size<double>(numStates_ * numStates_);
    tempLogTransProbs_.resize(transSize, LOG_ZERO);
}

StateSequence LogSIMDViterbiCalculator::decode() {
    try {
        // Precompute log transition probabilities for efficiency
        precomputeLogTransitionProbabilities(tempLogTransProbs_.data());
        
        // Step 1: Initialize with first observation
        initializeFirstStep();
        
        // Step 2: Forward pass with SIMD optimization
        for (std::size_t t = 1; t < seqLength_; ++t) {
            if (performance::simd_available() && numStates_ >= 4) {
                computeForwardStepSIMD(t);
            } else {
                computeForwardStepScalar(t);
            }
        }
        
        // Step 3: Termination
        computeTermination();
        
        // Step 4: Backtrack
        backtrackPath();
        
        return sequence_;
        
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("LogSIMDViterbiCalculator::decode failed: ") + e.what());
    }
}

void LogSIMDViterbiCalculator::initializeFirstStep() {
    const Vector& pi = hmm_->getPi();
    const Observation firstObs = observations_(0);
    
    // Compute log emission probabilities for all states
    computeLogEmissionProbabilities(firstObs, tempLogEmisProbs_.data());
    
    // Initialize log delta values
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double piProb = pi(i);
        const double logPi = safeLog(piProb);
        const double logEmis = tempLogEmisProbs_[i];
        
        logDelta_[getMatrixIndex(0, i)] = logPi + logEmis;
        psi_[getMatrixIndex(0, i)] = 0; // No previous state
    }
}

void LogSIMDViterbiCalculator::computeForwardStepSIMD(std::size_t t) {
    const Observation obs = observations_(t);
    
    // Compute log emission probabilities for all states
    computeLogEmissionProbabilities(obs, tempLogEmisProbs_.data());
    
    // Prefetch the previous log delta values
    const double* prevLogDelta = &logDelta_[getMatrixIndex(t - 1, 0)];
    performance::prefetch_read(prevLogDelta);
    
    // Process each target state
    for (std::size_t j = 0; j < numStates_; ++j) {
        // Get pointer to log transition probabilities for all source states -> state j
        const double* logTransToJ = &tempLogTransProbs_[j * numStates_];
        
        // SIMD-optimized computation of log transition scores
        computeLogTransitionScoresSIMD(j, logTransToJ, prevLogDelta, tempLogScores_.data(), numStates_);
        
        // Find the maximum (best) log transition score
        double maxLogScore = LOG_ZERO;
        std::size_t bestPrevState = 0;
        
        findMaxSIMD(tempLogScores_.data(), numStates_, maxLogScore, bestPrevState);
        
        // Add log emission probability
        const double logEmis = tempLogEmisProbs_[j];
        
        logDelta_[getMatrixIndex(t, j)] = maxLogScore + logEmis;
        psi_[getMatrixIndex(t, j)] = static_cast<int>(bestPrevState);
    }
}

void LogSIMDViterbiCalculator::computeForwardStepScalar(std::size_t t) {
    const Observation obs = observations_(t);
    
    // Compute log emission probabilities for all states
    computeLogEmissionProbabilities(obs, tempLogEmisProbs_.data());
    
    for (std::size_t j = 0; j < numStates_; ++j) {
        double maxLogScore = LOG_ZERO;
        std::size_t bestPrevState = 0;
        
        // Find best previous state
        for (std::size_t i = 0; i < numStates_; ++i) {
            const double logTrans = tempLogTransProbs_[j * numStates_ + i];
            const double logScore = logDelta_[getMatrixIndex(t - 1, i)] + logTrans;
            
            if (logScore > maxLogScore) {
                maxLogScore = logScore;
                bestPrevState = i;
            }
        }
        
        // Add log emission probability
        const double logEmis = tempLogEmisProbs_[j];
        
        logDelta_[getMatrixIndex(t, j)] = maxLogScore + logEmis;
        psi_[getMatrixIndex(t, j)] = static_cast<int>(bestPrevState);
    }
}

void LogSIMDViterbiCalculator::computeLogTransitionScoresSIMD(
    std::size_t /*fromState*/,
    const double* logTransitionsStart,
    const double* prevLogDelta,
    double* results,
    std::size_t numStates) const {
    
#ifdef LIBHMM_HAS_AVX
    const std::size_t simdWidth = 4; // AVX processes 4 doubles at once
    const std::size_t simdBlocks = numStates / simdWidth;
    
    // Process SIMD blocks
    for (std::size_t block = 0; block < simdBlocks; ++block) {
        const std::size_t baseIdx = block * simdWidth;
        
        // Load log transition probabilities (unaligned for safety)
        __m256d logTrans = _mm256_loadu_pd(&logTransitionsStart[baseIdx]);
        
        // Load previous log delta values (unaligned for safety)
        __m256d logDelta = _mm256_loadu_pd(&prevLogDelta[baseIdx]);
        
        // Compute log scores: logDelta + logTrans
        __m256d logScores = _mm256_add_pd(logDelta, logTrans);
        
        // Store results (unaligned for safety)
        _mm256_storeu_pd(&results[baseIdx], logScores);
    }
    
    // Handle remainder elements
    for (std::size_t i = simdBlocks * simdWidth; i < numStates; ++i) {
        results[i] = prevLogDelta[i] + logTransitionsStart[i];
    }
    
#elif defined(LIBHMM_HAS_SSE2)
    const std::size_t simdWidth = 2; // SSE2 processes 2 doubles at once
    const std::size_t simdBlocks = numStates / simdWidth;
    
    // Process SSE2 blocks
    for (std::size_t block = 0; block < simdBlocks; ++block) {
        const std::size_t baseIdx = block * simdWidth;
        
        // Use unaligned loads/stores for safety
        __m128d logTrans = _mm_loadu_pd(&logTransitionsStart[baseIdx]);
        __m128d logDelta = _mm_loadu_pd(&prevLogDelta[baseIdx]);
        
        __m128d logScores = _mm_add_pd(logDelta, logTrans);
        _mm_storeu_pd(&results[baseIdx], logScores);
    }
    
    // Handle remainder
    for (std::size_t i = simdBlocks * simdWidth; i < numStates; ++i) {
        results[i] = prevLogDelta[i] + logTransitionsStart[i];
    }
    
#else
    // Fallback scalar implementation
    for (std::size_t i = 0; i < numStates; ++i) {
        results[i] = prevLogDelta[i] + logTransitionsStart[i];
    }
#endif
}

void LogSIMDViterbiCalculator::findMaxSIMD(const double* values, std::size_t size, 
                                           double& maxValue, std::size_t& maxIndex) const {
    maxValue = LOG_ZERO;
    maxIndex = 0;
    
#ifdef LIBHMM_HAS_AVX
    if (size >= 4) {
        __m256d maxVec = _mm256_set1_pd(LOG_ZERO);
        __m256d indices = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
        __m256d maxIndices = _mm256_setzero_pd();
        
        const std::size_t simdBlocks = size / 4;
        
        for (std::size_t block = 0; block < simdBlocks; ++block) {
            const std::size_t baseIdx = block * 4;
            __m256d vals = _mm256_loadu_pd(&values[baseIdx]);
            
            // Compare and update maximums
            __m256d mask = _mm256_cmp_pd(vals, maxVec, _CMP_GT_OQ);
            maxVec = _mm256_max_pd(maxVec, vals);
            
            // Update indices where we found new maximums
            __m256d blockIndices = _mm256_add_pd(indices, _mm256_set1_pd(static_cast<double>(baseIdx)));
            maxIndices = _mm256_blendv_pd(maxIndices, blockIndices, mask);
            
            indices = _mm256_add_pd(indices, _mm256_set1_pd(4.0));
        }
        
        // Extract maximum and index from SIMD register
        alignas(32) double maxValues[4];
        alignas(32) double maxIndicesArray[4];
        _mm256_store_pd(maxValues, maxVec);
        _mm256_store_pd(maxIndicesArray, maxIndices);
        
        for (int i = 0; i < 4; ++i) {
            if (maxValues[i] > maxValue) {
                maxValue = maxValues[i];
                maxIndex = static_cast<std::size_t>(maxIndicesArray[i]);
            }
        }
        
        // Handle remainder
        for (std::size_t i = simdBlocks * 4; i < size; ++i) {
            if (values[i] > maxValue) {
                maxValue = values[i];
                maxIndex = i;
            }
        }
        return;
    }
#endif
    
    // Fallback scalar implementation
    for (std::size_t i = 0; i < size; ++i) {
        if (values[i] > maxValue) {
            maxValue = values[i];
            maxIndex = i;
        }
    }
}

void LogSIMDViterbiCalculator::computeLogEmissionProbabilities(Observation observation, double* logEmisProbs) const {
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double emisProb = hmm_->getProbabilityDistribution(static_cast<int>(i))->getProbability(observation);
        logEmisProbs[i] = safeLog(emisProb);
    }
}

void LogSIMDViterbiCalculator::precomputeLogTransitionProbabilities(double* logTransProbs) const {
    const Matrix& trans = hmm_->getTrans();
    
    // Convert transition matrix to log space and store in row-major order
    // logTransProbs[j * numStates_ + i] = log(P(i -> j))
    for (std::size_t j = 0; j < numStates_; ++j) {
        for (std::size_t i = 0; i < numStates_; ++i) {
            const double transProb = trans(i, j);
            logTransProbs[j * numStates_ + i] = safeLog(transProb);
        }
    }
}

void LogSIMDViterbiCalculator::computeTermination() {
    const auto lastTimeIndex = seqLength_ - 1;
    const double* lastLogDelta = &logDelta_[getMatrixIndex(lastTimeIndex, 0)];
    
    double maxLogValue = LOG_ZERO;
    std::size_t bestFinalState = 0;
    
    // Find the best final state
    for (std::size_t i = 0; i < numStates_; ++i) {
        if (lastLogDelta[i] > maxLogValue) {
            maxLogValue = lastLogDelta[i];
            bestFinalState = i;
        }
    }
    
    sequence_[lastTimeIndex] = static_cast<int>(bestFinalState);
    logProbability_ = maxLogValue;
}

void LogSIMDViterbiCalculator::backtrackPath() {
    // Backtrack from the end to find the optimal path
    for (std::size_t t = seqLength_ - 1; t > 0; --t) {
        const int currentState = sequence_[t];
        const int prevState = psi_[getMatrixIndex(t, currentState)];
        sequence_[t - 1] = prevState;
    }
}

double LogSIMDViterbiCalculator::logAdd(double logA, double logB) noexcept {
    // Numerically stable computation of log(exp(logA) + exp(logB))
    if (logA == LOG_ZERO && logB == LOG_ZERO) {
        return LOG_ZERO;
    }
    
    if (logA == LOG_ZERO) {
        return logB;
    }
    
    if (logB == LOG_ZERO) {
        return logA;
    }
    
    // Ensure logA >= logB for numerical stability
    if (logA < logB) {
        std::swap(logA, logB);
    }
    
    const double diff = logB - logA;
    
    // If the difference is too large, the smaller term is negligible
    if (diff < -50.0) {
        return logA;
    }
    
    return logA + std::log1p(std::exp(diff));
}

} // namespace libhmm
