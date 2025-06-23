#include "libhmm/calculators/optimized_forward_backward_calculator.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace libhmm {

//========== OptimizedForwardBackwardCalculator Implementation ==========

OptimizedForwardBackwardCalculator::OptimizedForwardBackwardCalculator(
    Hmm* hmm, const ObservationSet& observations, bool useBlocking, std::size_t blockSize)
    : ForwardBackwardCalculator(hmm, observations),
      numStates_(static_cast<std::size_t>(hmm->getNumStates())),
      obsSize_(observations.size()),
      useBlockedComputation_(useBlocking),
      blockSize_(blockSize == 0 ? getRecommendedBlockSize(numStates_) : blockSize) {
    
    // Calculate aligned state size for SIMD operations
    alignedStateSize_ = performance::cache_aligned_size<double>(numStates_);
    
    // Initialize aligned storage
    initializeAlignedStorage();
    
    // Run optimized algorithms
    forward();
    backward();
}

void OptimizedForwardBackwardCalculator::initializeAlignedStorage() {
    const std::size_t forwardBackwardSize = obsSize_ * alignedStateSize_;
    const std::size_t transSize = alignedStateSize_ * alignedStateSize_;
    const std::size_t piSize = alignedStateSize_;
    
    // Allocate aligned storage
    alignedForward_ = std::make_unique<AlignedVector>(forwardBackwardSize, 0.0);
    alignedBackward_ = std::make_unique<AlignedVector>(forwardBackwardSize, 0.0);
    alignedTrans_ = std::make_unique<AlignedVector>(transSize, 0.0);
    alignedPi_ = std::make_unique<AlignedVector>(piSize, 0.0);
    
    // Copy transition matrix to aligned storage
    const Matrix trans = hmm_->getTrans();
    copyToAlignedStorage(trans, *alignedTrans_, numStates_, numStates_, alignedStateSize_);
    
    // Copy pi vector to aligned storage  
    const Vector pi = hmm_->getPi();
    for (std::size_t i = 0; i < numStates_; ++i) {
        (*alignedPi_)[i] = pi(i);
    }
    
    // Zero-pad the pi vector
    for (std::size_t i = numStates_; i < alignedStateSize_; ++i) {
        (*alignedPi_)[i] = 0.0;
    }
}

void OptimizedForwardBackwardCalculator::copyToAlignedStorage(
    const Matrix& source, AlignedVector& dest, std::size_t rows, std::size_t cols, std::size_t alignedCols) {
    
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            dest[i * alignedCols + j] = source(i, j);
        }
        // Zero-pad the row
        for (std::size_t j = cols; j < alignedCols; ++j) {
            dest[i * alignedCols + j] = 0.0;
        }
    }
}

void OptimizedForwardBackwardCalculator::copyFromAlignedStorage(
    const AlignedVector& source, Matrix& dest, std::size_t rows, std::size_t cols, std::size_t alignedCols) {
    
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            dest(i, j) = source[i * alignedCols + j];
        }
    }
}

void OptimizedForwardBackwardCalculator::computeEmissionProbabilities(
    std::size_t t, AlignedVector& emissions) const {
    
    const Observation obs = observations_(t);
    
    // Compute emission probabilities for all states
    for (std::size_t i = 0; i < numStates_; ++i) {
        emissions[i] = hmm_->getProbabilityDistribution(static_cast<int>(i))->getProbability(obs);
        if (emissions[i] < ZERO || std::isnan(emissions[i])) {
            emissions[i] = ZERO;
        }
    }
    
    // Zero-pad for SIMD alignment
    for (std::size_t i = numStates_; i < alignedStateSize_; ++i) {
        emissions[i] = 0.0;
    }
}

void OptimizedForwardBackwardCalculator::blockedMatrixVectorMultiply(
    const double* matrix, const double* vector, double* result, 
    std::size_t rows, std::size_t cols, std::size_t blockSize) const {
    
    // Initialize result vector
    std::fill_n(result, rows, 0.0);
    
    // Blocked computation for better cache locality
    for (std::size_t ii = 0; ii < rows; ii += blockSize) {
        const std::size_t iEnd = std::min(ii + blockSize, rows);
        
        for (std::size_t jj = 0; jj < cols; jj += blockSize) {
            const std::size_t jEnd = std::min(jj + blockSize, cols);
            
            // Process block
            for (std::size_t i = ii; i < iEnd; ++i) {
                performance::prefetch_read(&matrix[(i + 1) * cols + jj]);
                
                // Use SIMD dot product for the block
                const std::size_t blockCols = jEnd - jj;
                const double partial = performance::SIMDOps::dot_product(
                    &matrix[i * cols + jj], &vector[jj], blockCols);
                result[i] += partial;
            }
        }
    }
}

void OptimizedForwardBackwardCalculator::matrixVectorMultiplyTransposed(
    const double* matrix, const double* vector, double* result, 
    std::size_t rows, std::size_t cols) const {
    
    // Initialize result vector
    std::fill_n(result, cols, 0.0);
    
    // Compute result[j] = sum_i(matrix[i * cols + j] * vector[i])
    // This is the transpose of the normal matrix-vector multiplication
    
    for (std::size_t j = 0; j < cols; ++j) {
        // Extract column j from the matrix for vectorized computation
        AlignedVector column(rows);
        for (std::size_t i = 0; i < rows; ++i) {
            column[i] = matrix[i * cols + j];
        }
        
        // Use SIMD dot product
        result[j] = performance::SIMDOps::dot_product(column.data(), vector, rows);
    }
}

void OptimizedForwardBackwardCalculator::blockedMatrixVectorMultiplyTransposed(
    const double* matrix, const double* vector, double* result, 
    std::size_t rows, std::size_t cols, std::size_t blockSize) const {
    
    // Initialize result vector
    std::fill_n(result, cols, 0.0);
    
    // Blocked computation for better cache locality
    for (std::size_t jj = 0; jj < cols; jj += blockSize) {
        const std::size_t jEnd = std::min(jj + blockSize, cols);
        
        for (std::size_t ii = 0; ii < rows; ii += blockSize) {
            const std::size_t iEnd = std::min(ii + blockSize, rows);
            
            // Process block
            for (std::size_t j = jj; j < jEnd; ++j) {
                performance::prefetch_read(&matrix[(ii + blockSize) * cols + j]);
                
                // Extract partial column for this block
                AlignedVector partialColumn(iEnd - ii);
                for (std::size_t i = ii; i < iEnd; ++i) {
                    partialColumn[i - ii] = matrix[i * cols + j];
                }
                
                // Use SIMD dot product for the block
                const double partial = performance::SIMDOps::dot_product(
                    partialColumn.data(), &vector[ii], iEnd - ii);
                result[j] += partial;
            }
        }
    }
}

void OptimizedForwardBackwardCalculator::forward() {
    AlignedVector emissions(alignedStateSize_);
    
    // Initialization step: alpha(0, i) = pi(i) * b_i(O_0)
    computeEmissionProbabilities(0, emissions);
    performance::SIMDOps::vector_multiply(
        alignedPi_->data(), emissions.data(), 
        &(*alignedForward_)[0], alignedStateSize_);
    
    // Induction step
    for (std::size_t t = 1; t < obsSize_; ++t) {
        const double* prevAlpha = &(*alignedForward_)[(t - 1) * alignedStateSize_];
        double* currAlpha = &(*alignedForward_)[t * alignedStateSize_];
        
        // Prefetch next observation data
        if (t + 1 < obsSize_) {
            performance::prefetch_read(&observations_(t + 1));
        }
        
        // Compute emission probabilities for current observation
        computeEmissionProbabilities(t, emissions);
        
        // We need to compute: currAlpha[j] = sum_i(prevAlpha[i] * trans[i,j])
        // This is equivalent to: currAlpha = trans^T * prevAlpha
        
        if (useBlockedComputation_ && numStates_ > blockSize_) {
            // Use blocked SIMD computation for large matrices
            blockedMatrixVectorMultiplyTransposed(
                alignedTrans_->data(), prevAlpha, currAlpha, 
                numStates_, alignedStateSize_, blockSize_);
        } else {
            // Use SIMD-optimized transposed matrix-vector multiplication
            matrixVectorMultiplyTransposed(
                alignedTrans_->data(), prevAlpha, currAlpha, 
                numStates_, alignedStateSize_);
        }
        
        // Element-wise multiplication with emission probabilities
        performance::SIMDOps::vector_multiply(
            currAlpha, emissions.data(), currAlpha, alignedStateSize_);
        
        // Numerical stability check
        for (std::size_t i = 0; i < numStates_; ++i) {
            if (currAlpha[i] < ZERO || std::isnan(currAlpha[i])) {
                currAlpha[i] = ZERO;
            }
        }
    }
    
    // Copy results back to boost matrix format
    Matrix alpha(obsSize_, numStates_);
    copyFromAlignedStorage(*alignedForward_, alpha, obsSize_, numStates_, alignedStateSize_);
    forwardVariables_ = alpha;
}

void OptimizedForwardBackwardCalculator::backward() {
    AlignedVector emissions(alignedStateSize_);
    
    // Initialization: beta(T-1, i) = 1.0
    const std::size_t lastT = obsSize_ - 1;
    double* lastBeta = &(*alignedBackward_)[lastT * alignedStateSize_];
    std::fill_n(lastBeta, numStates_, 1.0);
    std::fill_n(lastBeta + numStates_, alignedStateSize_ - numStates_, 0.0);
    
    // Induction step (backward)
    for (std::size_t t = obsSize_ - 1; t > 0; --t) {
        const std::size_t currentT = t - 1;
        const double* nextBeta = &(*alignedBackward_)[t * alignedStateSize_];
        double* currBeta = &(*alignedBackward_)[currentT * alignedStateSize_];
        
        // Prefetch previous observation data
        if (currentT > 0) {
            performance::prefetch_read(&observations_(currentT - 1));
        }
        
        // Compute emission probabilities for next observation
        computeEmissionProbabilities(t, emissions);
        
        // Element-wise multiplication: beta * emission_probs
        AlignedVector weightedBeta(alignedStateSize_);
        performance::SIMDOps::vector_multiply(
            nextBeta, emissions.data(), weightedBeta.data(), alignedStateSize_);
        
        // Backward computation: beta[currentT,i] = sum_j(trans[i,j] * emission[j] * beta[t,j])
        // We already computed weightedBeta[j] = emission[j] * beta[t,j]
        // So we need: beta[currentT,i] = sum_j(trans[i,j] * weightedBeta[j])
        // This is a standard matrix-vector multiplication: currBeta = trans * weightedBeta
        
        if (useBlockedComputation_ && numStates_ > blockSize_) {
            // Use blocked SIMD computation for large matrices
            blockedMatrixVectorMultiply(
                alignedTrans_->data(), weightedBeta.data(), currBeta, 
                numStates_, alignedStateSize_, blockSize_);
        } else {
            // Use SIMD-optimized matrix-vector multiplication
            performance::SIMDOps::matrix_vector_multiply(
                alignedTrans_->data(), weightedBeta.data(), currBeta, 
                numStates_, alignedStateSize_);
        }
        
        // Zero-pad for alignment
        std::fill_n(currBeta + numStates_, alignedStateSize_ - numStates_, 0.0);
    }
    
    // Copy results back to boost matrix format
    Matrix beta(obsSize_, numStates_);
    copyFromAlignedStorage(*alignedBackward_, beta, obsSize_, numStates_, alignedStateSize_);
    backwardVariables_ = beta;
}

std::string OptimizedForwardBackwardCalculator::getOptimizationInfo() const {
    std::string info = "OptimizedForwardBackwardCalculator using: ";
    
    if (performance::simd_available()) {
#ifdef LIBHMM_HAS_AVX
        info += "AVX";
#elif defined(LIBHMM_HAS_SSE2)
        info += "SSE2";
#elif defined(LIBHMM_HAS_NEON)
        info += "ARM NEON";
#endif
    } else {
        info += "Scalar fallback";
    }
    
    info += ", States: " + std::to_string(numStates_);
    info += ", Aligned size: " + std::to_string(alignedStateSize_);
    info += ", Block size: " + std::to_string(blockSize_);
    info += ", Blocked computation: ";
    info += (useBlockedComputation_ ? "Yes" : "No");
    
    return info;
}

std::size_t OptimizedForwardBackwardCalculator::getRecommendedBlockSize(std::size_t numStates) noexcept {
    // Cache size heuristics
    constexpr std::size_t L1_CACHE_SIZE = 32 * 1024;  // 32KB typical L1 cache
    constexpr std::size_t DOUBLE_SIZE = sizeof(double);
    
    // Try to fit working set in L1 cache
    const std::size_t maxBlockSize = std::sqrt(L1_CACHE_SIZE / (3 * DOUBLE_SIZE)); // matrix + 2 vectors
    
    // Use power of 2 block sizes for better alignment
    std::size_t blockSize = 64;
    while (blockSize > maxBlockSize && blockSize > 8) {
        blockSize /= 2;
    }
    
    return std::min(blockSize, numStates);
}

//========== CalculatorMemoryPool Implementation ==========

double* CalculatorMemoryPool::acquire(std::size_t size) {
    std::lock_guard<std::mutex> lock(poolMutex_);
    
    // Look for existing block of sufficient size
    for (auto& block : blocks_) {
        if (!block->inUse && block->size >= size) {
            block->inUse = true;
            return block->memory[0];
        }
    }
    
    // Create new block
    auto newBlock = std::make_unique<PoolBlock>(size);
    double* ptr = newBlock->memory[0];
    newBlock->inUse = true;
    blocks_.push_back(std::move(newBlock));
    
    return ptr;
}

void CalculatorMemoryPool::release(double* ptr) noexcept {
    std::lock_guard<std::mutex> lock(poolMutex_);
    
    for (auto& block : blocks_) {
        if (block->memory[0] == ptr) {
            block->inUse = false;
            return;
        }
    }
}

void CalculatorMemoryPool::clear() {
    std::lock_guard<std::mutex> lock(poolMutex_);
    blocks_.clear();
}

} // namespace libhmm
