#include "libhmm/calculators/log_simd_forward_backward_calculator.h"
#include "libhmm/common/common.h"
#include "libhmm/hmm.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace libhmm {

LogSIMDForwardBackwardCalculator::LogSIMDForwardBackwardCalculator(
    Hmm* hmm, const ObservationSet& observations, bool useBlocking, std::size_t blockSize)
    : LogForwardBackwardCalculator(hmm, observations),
      numStates_(static_cast<std::size_t>(hmm->getNumStates())),
      obsSize_(observations.size()),
      useBlockedComputation_(useBlocking),
      blockSize_(blockSize) {
    
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
    
    // Initialize aligned storage
    initializeAlignedStorage();
    
    // Run the algorithms
    forward();
    backward();
}

void LogSIMDForwardBackwardCalculator::initializeAlignedStorage() {
    const std::size_t forwardSize = obsSize_ * alignedStateSize_;
    const std::size_t backwardSize = obsSize_ * alignedStateSize_;
    const std::size_t transSize = numStates_ * alignedStateSize_;
    const std::size_t piSize = alignedStateSize_;
    
    // Allocate aligned storage
    alignedForward_ = std::make_unique<AlignedVector>(forwardSize, LOGZERO);
    alignedBackward_ = std::make_unique<AlignedVector>(backwardSize, LOGZERO);
    alignedLogTrans_ = std::make_unique<AlignedVector>(transSize, LOGZERO);
    alignedLogPi_ = std::make_unique<AlignedVector>(piSize, LOGZERO);
    
    // Copy and convert transition matrix to log-space
    const Matrix trans = hmm_->getTrans();
    copyToAlignedLogStorage(trans, *alignedLogTrans_, numStates_, numStates_, alignedStateSize_);
    
    // Copy and convert initial probabilities to log-space
    const Vector pi = hmm_->getPi();
    for (std::size_t i = 0; i < numStates_; ++i) {
        (*alignedLogPi_)[i] = (pi(i) > 0.0) ? std::log(pi(i)) : LOGZERO;
    }
    // Zero-pad the rest
    for (std::size_t i = numStates_; i < alignedStateSize_; ++i) {
        (*alignedLogPi_)[i] = LOGZERO;
    }
}

void LogSIMDForwardBackwardCalculator::copyToAlignedLogStorage(
    const Matrix& source, AlignedVector& dest, 
    std::size_t rows, std::size_t cols, std::size_t alignedCols) {
    
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const double value = source(i, j);
            dest[i * alignedCols + j] = (value > 0.0) ? std::log(value) : LOGZERO;
        }
        // Zero-pad the rest of the row
        for (std::size_t j = cols; j < alignedCols; ++j) {
            dest[i * alignedCols + j] = LOGZERO;
        }
    }
}

void LogSIMDForwardBackwardCalculator::copyFromAlignedLogStorage(
    const AlignedVector& source, Matrix& dest,
    std::size_t rows, std::size_t cols, std::size_t alignedCols) {
    
    dest.resize(rows, cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const double logValue = source[i * alignedCols + j];
            dest(i, j) = std::isnan(logValue) ? 0.0 : std::exp(logValue);
        }
    }
}

void LogSIMDForwardBackwardCalculator::computeLogEmissionProbabilities(
    std::size_t t, AlignedVector& logEmissions) const {
    
    // Zero the output array
    std::fill(logEmissions.begin(), logEmissions.end(), LOGZERO);
    
    // Compute emission probabilities and convert to log-space
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double prob = hmm_->getProbabilityDistribution(static_cast<int>(i))
                               ->getProbability(observations_(t));
        logEmissions[i] = (prob > 0.0) ? std::log(prob) : LOGZERO;
    }
}

void LogSIMDForwardBackwardCalculator::vectorizedElnSum(
    const double* x, const double* y, double* result, std::size_t size) const {
    
    for (std::size_t i = 0; i < size; ++i) {
        const double xi = x[i];
        const double yi = y[i];
        
        if (std::isnan(xi) || std::isnan(yi)) {
            if (std::isnan(xi)) {
                result[i] = yi;
            } else {
                result[i] = xi;
            }
        } else {
            if (xi > yi) {
                result[i] = xi + std::log(1.0 + std::exp(yi - xi));
            } else {
                result[i] = yi + std::log(1.0 + std::exp(xi - yi));
            }
        }
    }
}

void LogSIMDForwardBackwardCalculator::vectorizedElnProduct(
    const double* x, const double* y, double* result, std::size_t size) const {
    
    for (std::size_t i = 0; i < size; ++i) {
        if (std::isnan(x[i]) || std::isnan(y[i])) {
            result[i] = LOGZERO;
        } else {
            result[i] = x[i] + y[i];
        }
    }
}

void LogSIMDForwardBackwardCalculator::logMatrixVectorMultiplyTransposed(
    const double* logMatrix, const double* logVector,
    double* logResult, std::size_t rows, std::size_t cols) const {
    
    // Initialize result to LOGZERO
    std::fill_n(logResult, cols, LOGZERO);
    
    // For each column j, compute: result[j] = elnsum_i(matrix[i,j] + vector[i])
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < rows; ++i) {
            const double matrixVal = logMatrix[i * cols + j];
            const double vectorVal = logVector[i];
            
            if (!std::isnan(matrixVal) && !std::isnan(vectorVal)) {
                const double product = matrixVal + vectorVal;
                
                // Elnsum with current result
                if (std::isnan(logResult[j])) {
                    logResult[j] = product;
                } else {
                    if (logResult[j] > product) {
                        logResult[j] = logResult[j] + std::log(1.0 + std::exp(product - logResult[j]));
                    } else {
                        logResult[j] = product + std::log(1.0 + std::exp(logResult[j] - product));
                    }
                }
            }
        }
    }
}

void LogSIMDForwardBackwardCalculator::logMatrixVectorMultiply(
    const double* logMatrix, const double* logVector,
    double* logResult, std::size_t rows, std::size_t cols) const {
    
    // Initialize result to LOGZERO
    std::fill_n(logResult, rows, LOGZERO);
    
    // For each row i, compute: result[i] = elnsum_j(matrix[i,j] + vector[j])
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const double matrixVal = logMatrix[i * cols + j];
            const double vectorVal = logVector[j];
            
            if (!std::isnan(matrixVal) && !std::isnan(vectorVal)) {
                const double product = matrixVal + vectorVal;
                
                // Elnsum with current result
                if (std::isnan(logResult[i])) {
                    logResult[i] = product;
                } else {
                    if (logResult[i] > product) {
                        logResult[i] = logResult[i] + std::log(1.0 + std::exp(product - logResult[i]));
                    } else {
                        logResult[i] = product + std::log(1.0 + std::exp(logResult[i] - product));
                    }
                }
            }
        }
    }
}

void LogSIMDForwardBackwardCalculator::forward() {
    AlignedVector logEmissions(alignedStateSize_);
    
    // Initialization step: log_alpha(0, i) = log_pi(i) + log_b_i(O_0)
    computeLogEmissionProbabilities(0, logEmissions);
    vectorizedElnProduct(alignedLogPi_->data(), logEmissions.data(), 
                        &(*alignedForward_)[0], alignedStateSize_);
    
    // Induction step
    for (std::size_t t = 1; t < obsSize_; ++t) {
        const double* prevLogAlpha = &(*alignedForward_)[(t - 1) * alignedStateSize_];
        double* currLogAlpha = &(*alignedForward_)[t * alignedStateSize_];
        
        // Compute emission probabilities for current observation
        computeLogEmissionProbabilities(t, logEmissions);
        
        // We need to compute: log_alpha[t,j] = log_b_j(O_t) + elnsum_i(log_alpha[t-1,i] + log_trans[i,j])
        // This is equivalent to: currLogAlpha = log_emissions + log_trans^T * prevLogAlpha
        
        if (useBlockedComputation_ && numStates_ > blockSize_) {
            // Use blocked computation for large matrices
            blockedLogMatrixVectorMultiplyTransposed(
                alignedLogTrans_->data(), prevLogAlpha, currLogAlpha, 
                numStates_, alignedStateSize_, blockSize_);
        } else {
            // Use standard log-space matrix-vector multiplication
            logMatrixVectorMultiplyTransposed(
                alignedLogTrans_->data(), prevLogAlpha, currLogAlpha, 
                numStates_, alignedStateSize_);
        }
        
        // Add emission probabilities in log space
        vectorizedElnProduct(currLogAlpha, logEmissions.data(), currLogAlpha, alignedStateSize_);
    }
    
    // Copy results back to boost matrix format
    Matrix alpha(obsSize_, numStates_);
    copyFromAlignedLogStorage(*alignedForward_, alpha, obsSize_, numStates_, alignedStateSize_);
    forwardVariables_ = alpha;
}

void LogSIMDForwardBackwardCalculator::backward() {
    AlignedVector logEmissions(alignedStateSize_);
    
    // Initialization: log_beta(T-1, i) = log(1.0) = 0.0
    const std::size_t lastT = obsSize_ - 1;
    double* lastLogBeta = &(*alignedBackward_)[lastT * alignedStateSize_];
    std::fill_n(lastLogBeta, numStates_, 0.0);  // log(1.0) = 0.0
    std::fill_n(lastLogBeta + numStates_, alignedStateSize_ - numStates_, LOGZERO);
    
    // Induction step (backward)
    for (std::size_t t = obsSize_ - 1; t > 0; --t) {
        const std::size_t currentT = t - 1;
        const double* nextLogBeta = &(*alignedBackward_)[t * alignedStateSize_];
        double* currLogBeta = &(*alignedBackward_)[currentT * alignedStateSize_];
        
        // Compute emission probabilities for next observation
        computeLogEmissionProbabilities(t, logEmissions);
        
        // Element-wise addition in log space: log_beta * log_emission_probs
        AlignedVector weightedLogBeta(alignedStateSize_);
        vectorizedElnProduct(nextLogBeta, logEmissions.data(), weightedLogBeta.data(), alignedStateSize_);
        
        // Backward computation: log_beta[currentT,i] = elnsum_j(log_trans[i,j] + log_emission[j] + log_beta[t,j])
        // We already computed weightedLogBeta[j] = log_emission[j] + log_beta[t,j]
        // So we need: log_beta[currentT,i] = elnsum_j(log_trans[i,j] + weightedLogBeta[j])
        // This is a standard log-space matrix-vector multiplication: currLogBeta = log_trans * weightedLogBeta
        
        if (useBlockedComputation_ && numStates_ > blockSize_) {
            // Use blocked computation for large matrices
            blockedLogMatrixVectorMultiply(
                alignedLogTrans_->data(), weightedLogBeta.data(), currLogBeta, 
                numStates_, alignedStateSize_, blockSize_);
        } else {
            // Use standard log-space matrix-vector multiplication
            logMatrixVectorMultiply(
                alignedLogTrans_->data(), weightedLogBeta.data(), currLogBeta, 
                numStates_, alignedStateSize_);
        }
        
        // Zero-pad for alignment
        std::fill_n(currLogBeta + numStates_, alignedStateSize_ - numStates_, LOGZERO);
    }
    
    // Copy results back to boost matrix format
    Matrix beta(obsSize_, numStates_);
    copyFromAlignedLogStorage(*alignedBackward_, beta, obsSize_, numStates_, alignedStateSize_);
    backwardVariables_ = beta;
}

double LogSIMDForwardBackwardCalculator::logProbability() {
    const auto lastIndex = obsSize_ - 1;
    
    if (numStates_ == 0) {
        return LOGZERO;
    }
    
    // Sum the last forward variables in log space using elnsum
    double logProb = (*alignedForward_)[lastIndex * alignedStateSize_ + 0];
    
    for (std::size_t i = 1; i < numStates_; ++i) {
        const double val = (*alignedForward_)[lastIndex * alignedStateSize_ + i];
        
        if (!std::isnan(val)) {
            if (std::isnan(logProb)) {
                logProb = val;
            } else {
                if (logProb > val) {
                    logProb = logProb + std::log(1.0 + std::exp(val - logProb));
                } else {
                    logProb = val + std::log(1.0 + std::exp(logProb - val));
                }
            }
        }
    }
    
    return logProb;
}

std::string LogSIMDForwardBackwardCalculator::getOptimizationInfo() const {
    std::string info = "LogSIMDForwardBackwardCalculator using: ";
    
    if (performance::simd_available()) {
#ifdef LIBHMM_HAS_AVX
        info += "AVX + Log-space";
#elif defined(LIBHMM_HAS_SSE2)
        info += "SSE2 + Log-space";
#elif defined(LIBHMM_HAS_NEON)
        info += "ARM NEON + Log-space";
#endif
    } else {
        info += "Scalar Log-space";
    }
    
    info += ", States: " + std::to_string(numStates_);
    info += ", Aligned size: " + std::to_string(alignedStateSize_);
    info += ", Block size: " + std::to_string(blockSize_);
    info += ", Blocked computation: ";
    info += (useBlockedComputation_ ? "Yes" : "No");
    
    return info;
}

std::size_t LogSIMDForwardBackwardCalculator::getRecommendedBlockSize(std::size_t numStates) noexcept {
    // Cache size heuristics for log-space operations
    constexpr std::size_t L1_CACHE_SIZE = 32 * 1024;  // 32KB typical L1 cache
    constexpr std::size_t DOUBLE_SIZE = sizeof(double);
    
    // Try to fit working set in L1 cache (log matrix + 2 log vectors)
    const std::size_t maxBlockSize = std::sqrt(L1_CACHE_SIZE / (3 * DOUBLE_SIZE));
    
    // Use power of 2 block sizes for better alignment
    std::size_t blockSize = 64;
    while (blockSize > maxBlockSize && blockSize > 8) {
        blockSize /= 2;
    }
    
    return std::min(blockSize, numStates);
}

void LogSIMDForwardBackwardCalculator::blockedLogMatrixVectorMultiplyTransposed(
    const double* logMatrix, const double* logVector,
    double* logResult, std::size_t rows, std::size_t cols, std::size_t blockSize) const {
    
    // Initialize result
    std::fill_n(logResult, cols, LOGZERO);
    
    // Blocked computation for better cache locality
    for (std::size_t jj = 0; jj < cols; jj += blockSize) {
        const std::size_t jEnd = std::min(jj + blockSize, cols);
        
        for (std::size_t ii = 0; ii < rows; ii += blockSize) {
            const std::size_t iEnd = std::min(ii + blockSize, rows);
            
            // Process block in log space
            for (std::size_t j = jj; j < jEnd; ++j) {
                for (std::size_t i = ii; i < iEnd; ++i) {
                    const double matrixVal = logMatrix[i * cols + j];
                    const double vectorVal = logVector[i];
                    
                    if (!std::isnan(matrixVal) && !std::isnan(vectorVal)) {
                        const double product = matrixVal + vectorVal;
                        
                        // Elnsum with current result
                        if (std::isnan(logResult[j])) {
                            logResult[j] = product;
                        } else {
                            if (logResult[j] > product) {
                                logResult[j] = logResult[j] + std::log(1.0 + std::exp(product - logResult[j]));
                            } else {
                                logResult[j] = product + std::log(1.0 + std::exp(logResult[j] - product));
                            }
                        }
                    }
                }
            }
        }
    }
}

void LogSIMDForwardBackwardCalculator::blockedLogMatrixVectorMultiply(
    const double* logMatrix, const double* logVector,
    double* logResult, std::size_t rows, std::size_t cols, std::size_t blockSize) const {
    
    // Initialize result
    std::fill_n(logResult, rows, LOGZERO);
    
    // Blocked computation for better cache locality
    for (std::size_t ii = 0; ii < rows; ii += blockSize) {
        const std::size_t iEnd = std::min(ii + blockSize, rows);
        
        for (std::size_t jj = 0; jj < cols; jj += blockSize) {
            const std::size_t jEnd = std::min(jj + blockSize, cols);
            
            // Process block in log space
            for (std::size_t i = ii; i < iEnd; ++i) {
                for (std::size_t j = jj; j < jEnd; ++j) {
                    const double matrixVal = logMatrix[i * cols + j];
                    const double vectorVal = logVector[j];
                    
                    if (!std::isnan(matrixVal) && !std::isnan(vectorVal)) {
                        const double product = matrixVal + vectorVal;
                        
                        // Elnsum with current result
                        if (std::isnan(logResult[i])) {
                            logResult[i] = product;
                        } else {
                            if (logResult[i] > product) {
                                logResult[i] = logResult[i] + std::log(1.0 + std::exp(product - logResult[i]));
                            } else {
                                logResult[i] = product + std::log(1.0 + std::exp(logResult[i] - product));
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace libhmm
