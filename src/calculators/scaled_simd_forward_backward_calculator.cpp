#include "libhmm/calculators/scaled_simd_forward_backward_calculator.h"
#include "libhmm/common/common.h"
#include "libhmm/hmm.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace libhmm {

ScaledSIMDForwardBackwardCalculator::ScaledSIMDForwardBackwardCalculator(
    Hmm* hmm, const ObservationSet& observations, bool useBlocking, std::size_t blockSize)
    : ScaledForwardBackwardCalculator(hmm, observations),
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
    
    // Run the scaled SIMD algorithms
    forward();
    backward();
}

void ScaledSIMDForwardBackwardCalculator::initializeAlignedStorage() {
    const std::size_t forwardSize = obsSize_ * alignedStateSize_;
    const std::size_t backwardSize = obsSize_ * alignedStateSize_;
    const std::size_t transSize = numStates_ * alignedStateSize_;
    const std::size_t piSize = alignedStateSize_;
    const std::size_t scalesSize = obsSize_;
    
    // Allocate aligned storage
    alignedForward_ = std::make_unique<AlignedVector>(forwardSize, 0.0);
    alignedBackward_ = std::make_unique<AlignedVector>(backwardSize, 0.0);
    alignedTrans_ = std::make_unique<AlignedVector>(transSize, 0.0);
    alignedPi_ = std::make_unique<AlignedVector>(piSize, 0.0);
    alignedScales_ = std::make_unique<AlignedVector>(scalesSize, 1.0);
    
    // Copy transition matrix
    const Matrix trans = hmm_->getTrans();
    copyToAlignedStorage(trans, *alignedTrans_, numStates_, numStates_, alignedStateSize_);
    
    // Copy initial probabilities
    const Vector pi = hmm_->getPi();
    for (std::size_t i = 0; i < numStates_; ++i) {
        (*alignedPi_)[i] = pi(i);
    }
    // Zero-pad the rest
    for (std::size_t i = numStates_; i < alignedStateSize_; ++i) {
        (*alignedPi_)[i] = 0.0;
    }
}

void ScaledSIMDForwardBackwardCalculator::copyToAlignedStorage(
    const Matrix& source, AlignedVector& dest, 
    std::size_t rows, std::size_t cols, std::size_t alignedCols) {
    
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            dest[i * alignedCols + j] = source(i, j);
        }
        // Zero-pad the rest of the row
        for (std::size_t j = cols; j < alignedCols; ++j) {
            dest[i * alignedCols + j] = 0.0;
        }
    }
}

void ScaledSIMDForwardBackwardCalculator::copyFromAlignedStorage(
    const AlignedVector& source, Matrix& dest,
    std::size_t rows, std::size_t cols, std::size_t alignedCols) {
    
    dest.resize(rows, cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            dest(i, j) = source[i * alignedCols + j];
        }
    }
}

void ScaledSIMDForwardBackwardCalculator::computeEmissionProbabilities(
    std::size_t t, AlignedVector& emissions) const {
    
    // Zero the output array
    std::fill(emissions.begin(), emissions.end(), 0.0);
    
    // Compute emission probabilities
    for (std::size_t i = 0; i < numStates_; ++i) {
        const double prob = hmm_->getProbabilityDistribution(static_cast<int>(i))
                               ->getProbability(observations_(t));
        emissions[i] = (prob > ZERO) ? prob : ZERO;
    }
}

void ScaledSIMDForwardBackwardCalculator::simdVectorMultiply(
    const double* a, const double* b, double* result, std::size_t size) const {
    
    SIMDScaledOps::multiply_arrays(a, b, result, size);
}

void ScaledSIMDForwardBackwardCalculator::simdVectorAdd(
    const double* a, const double* b, double* result, std::size_t size) const {
    
    SIMDScaledOps::add_arrays(a, b, result, size);
}

void ScaledSIMDForwardBackwardCalculator::simdVectorScale(
    const double* input, double scale, double* result, std::size_t size) const {
    
    SIMDScaledOps::scale_array(input, scale, result, size);
}

double ScaledSIMDForwardBackwardCalculator::simdVectorSum(
    const double* vector, std::size_t size) const {
    
    return SIMDScaledOps::sum_array(vector, size);
}

void ScaledSIMDForwardBackwardCalculator::simdMatrixVectorMultiplyTransposed(
    const double* matrix, const double* vector,
    double* result, std::size_t rows, std::size_t cols) const {
    
    // Initialize result to zero
    std::fill_n(result, cols, 0.0);
    
    // For each column j, compute: result[j] = sum_i(matrix[i * cols + j] * vector[i])
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < rows; ++i) {
            result[j] += matrix[i * cols + j] * vector[i];
        }
    }
}

void ScaledSIMDForwardBackwardCalculator::blockedMatrixVectorMultiplyTransposed(
    const double* matrix, const double* vector,
    double* result, std::size_t rows, std::size_t cols, std::size_t blockSize) const {
    
    // Initialize result
    std::fill_n(result, cols, 0.0);
    
    // Blocked computation for better cache locality
    for (std::size_t jj = 0; jj < cols; jj += blockSize) {
        const std::size_t jEnd = std::min(jj + blockSize, cols);
        
        for (std::size_t ii = 0; ii < rows; ii += blockSize) {
            const std::size_t iEnd = std::min(ii + blockSize, rows);
            
            // Process block
            for (std::size_t j = jj; j < jEnd; ++j) {
                for (std::size_t i = ii; i < iEnd; ++i) {
                    result[j] += matrix[i * cols + j] * vector[i];
                }
            }
        }
    }
}

void ScaledSIMDForwardBackwardCalculator::simdMatrixVectorMultiply(
    const double* matrix, const double* vector,
    double* result, std::size_t rows, std::size_t cols) const {
    
    // Initialize result to zero
    std::fill_n(result, rows, 0.0);
    
    // For each row i, compute: result[i] = sum_j(matrix[i * cols + j] * vector[j])
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

void ScaledSIMDForwardBackwardCalculator::blockedMatrixVectorMultiply(
    const double* matrix, const double* vector,
    double* result, std::size_t rows, std::size_t cols, std::size_t blockSize) const {
    
    // Initialize result
    std::fill_n(result, rows, 0.0);
    
    // Blocked computation for better cache locality
    for (std::size_t ii = 0; ii < rows; ii += blockSize) {
        const std::size_t iEnd = std::min(ii + blockSize, rows);
        
        for (std::size_t jj = 0; jj < cols; jj += blockSize) {
            const std::size_t jEnd = std::min(jj + blockSize, cols);
            
            // Process block
            for (std::size_t i = ii; i < iEnd; ++i) {
                for (std::size_t j = jj; j < jEnd; ++j) {
                    result[i] += matrix[i * cols + j] * vector[j];
                }
            }
        }
    }
}

double ScaledSIMDForwardBackwardCalculator::computeScalingFactor(
    const double* array, std::size_t size) const {
    
    double sum = simdVectorSum(array, size);
    return (sum > MIN_SCALE_FACTOR && sum < MAX_SCALE_FACTOR) ? sum : 1.0;
}

void ScaledSIMDForwardBackwardCalculator::applyScaling(
    double* array, double scaleFactor, std::size_t size) const {
    
    if (scaleFactor > MIN_SCALE_FACTOR) {
        const double invScale = 1.0 / scaleFactor;
        simdVectorScale(array, invScale, array, size);
    }
}

void ScaledSIMDForwardBackwardCalculator::forward() {
    AlignedVector emissions(alignedStateSize_);
    AlignedVector alphaBar(alignedStateSize_);
    
    // Initialization step
    computeEmissionProbabilities(0, emissions);
    simdVectorMultiply(alignedPi_->data(), emissions.data(), alphaBar.data(), alignedStateSize_);
    
    // Compute scaling factor for first time step
    double scaleFactor = computeScalingFactor(alphaBar.data(), numStates_);
    (*alignedScales_)[0] = scaleFactor;
    
    // Scale and store alpha[0]
    applyScaling(alphaBar.data(), scaleFactor, alignedStateSize_);
    double* alpha0 = &(*alignedForward_)[0];
    std::copy_n(alphaBar.data(), alignedStateSize_, alpha0);
    
    // Induction step
    for (std::size_t t = 1; t < obsSize_; ++t) {
        const double* prevAlpha = &(*alignedForward_)[(t - 1) * alignedStateSize_];
        double* currAlpha = &(*alignedForward_)[t * alignedStateSize_];
        
        // Compute emission probabilities for current observation
        computeEmissionProbabilities(t, emissions);
        
        // Matrix-vector multiplication: alpha_bar[t,j] = sum_i(alpha[t-1,i] * trans[i,j])
        if (useBlockedComputation_ && numStates_ > blockSize_) {
            blockedMatrixVectorMultiplyTransposed(
                alignedTrans_->data(), prevAlpha, alphaBar.data(), 
                numStates_, alignedStateSize_, blockSize_);
        } else {
            simdMatrixVectorMultiplyTransposed(
                alignedTrans_->data(), prevAlpha, alphaBar.data(), 
                numStates_, alignedStateSize_);
        }
        
        // Multiply by emission probabilities
        simdVectorMultiply(alphaBar.data(), emissions.data(), alphaBar.data(), alignedStateSize_);
        
        // Compute and apply scaling
        scaleFactor = computeScalingFactor(alphaBar.data(), numStates_);
        (*alignedScales_)[t] = scaleFactor;
        applyScaling(alphaBar.data(), scaleFactor, alignedStateSize_);
        
        // Store scaled alpha[t]
        std::copy_n(alphaBar.data(), alignedStateSize_, currAlpha);
    }
    
    // Copy results back to boost matrix format
    Matrix alpha(obsSize_, numStates_);
    copyFromAlignedStorage(*alignedForward_, alpha, obsSize_, numStates_, alignedStateSize_);
    forwardVariables_ = alpha;
    
    // Update scaling factors in base class
    for (std::size_t t = 0; t < obsSize_; ++t) {
        c_(t) = (*alignedScales_)[t];
    }
}

void ScaledSIMDForwardBackwardCalculator::backward() {
    AlignedVector emissions(alignedStateSize_);
    AlignedVector weightedBeta(alignedStateSize_);
    
    // Initialization: beta[T-1, i] = 1.0 / c[T-1]
    const std::size_t lastT = obsSize_ - 1;
    double* lastBeta = &(*alignedBackward_)[lastT * alignedStateSize_];
    const double invLastScale = ((*alignedScales_)[lastT] > MIN_SCALE_FACTOR) ? 
                                1.0 / (*alignedScales_)[lastT] : 1.0;
    
    std::fill_n(lastBeta, numStates_, invLastScale);
    std::fill_n(lastBeta + numStates_, alignedStateSize_ - numStates_, 0.0);
    
    // Induction step (backward)
    for (std::size_t t = obsSize_ - 1; t > 0; --t) {
        const std::size_t currentT = t - 1;
        const double* nextBeta = &(*alignedBackward_)[t * alignedStateSize_];
        double* currBeta = &(*alignedBackward_)[currentT * alignedStateSize_];
        
        // Compute emission probabilities for next observation
        computeEmissionProbabilities(t, emissions);
        
        // Element-wise multiplication: weighted_beta[j] = beta[t,j] * emission[j]
        simdVectorMultiply(nextBeta, emissions.data(), weightedBeta.data(), alignedStateSize_);
        
        // Matrix-vector multiplication: beta[currentT,i] = sum_j(trans[i,j] * weighted_beta[j])
        if (useBlockedComputation_ && numStates_ > blockSize_) {
            blockedMatrixVectorMultiply(
                alignedTrans_->data(), weightedBeta.data(), currBeta, 
                numStates_, alignedStateSize_, blockSize_);
        } else {
            simdMatrixVectorMultiply(
                alignedTrans_->data(), weightedBeta.data(), currBeta, 
                numStates_, alignedStateSize_);
        }
        
        // Apply scaling: beta[currentT,i] = beta[currentT,i] / c[currentT]
        const double invScale = ((*alignedScales_)[currentT] > MIN_SCALE_FACTOR) ? 
                               1.0 / (*alignedScales_)[currentT] : 1.0;
        simdVectorScale(currBeta, invScale, currBeta, alignedStateSize_);
        
        // Zero-pad for alignment
        std::fill_n(currBeta + numStates_, alignedStateSize_ - numStates_, 0.0);
    }
    
    // Copy results back to boost matrix format
    Matrix beta(obsSize_, numStates_);
    copyFromAlignedStorage(*alignedBackward_, beta, obsSize_, numStates_, alignedStateSize_);
    backwardVariables_ = beta;
}

std::string ScaledSIMDForwardBackwardCalculator::getOptimizationInfo() const {
    std::string info = "ScaledSIMDForwardBackwardCalculator using: ";
    
    if (performance::simd_available()) {
#ifdef LIBHMM_HAS_AVX
        info += "AVX + Scaling";
#elif defined(LIBHMM_HAS_SSE2)
        info += "SSE2 + Scaling";
#elif defined(LIBHMM_HAS_NEON)
        info += "ARM NEON + Scaling";
#endif
    } else {
        info += "Scalar + Scaling";
    }
    
    info += ", States: " + std::to_string(numStates_);
    info += ", Aligned size: " + std::to_string(alignedStateSize_);
    info += ", Block size: " + std::to_string(blockSize_);
    info += ", Blocked computation: ";
    info += (useBlockedComputation_ ? "Yes" : "No");
    
    return info;
}

std::size_t ScaledSIMDForwardBackwardCalculator::getRecommendedBlockSize(std::size_t numStates) noexcept {
    // Cache size heuristics for scaled operations
    constexpr std::size_t L1_CACHE_SIZE = 32 * 1024;  // 32KB typical L1 cache
    constexpr std::size_t DOUBLE_SIZE = sizeof(double);
    
    // Try to fit working set in L1 cache (matrix + 2 vectors + scaling array)
    const std::size_t maxBlockSize = std::sqrt(L1_CACHE_SIZE / (4 * DOUBLE_SIZE));
    
    // Use power of 2 block sizes for better alignment
    std::size_t blockSize = 64;
    while (blockSize > maxBlockSize && blockSize > 8) {
        blockSize /= 2;
    }
    
    return std::min(blockSize, numStates);
}

Vector ScaledSIMDForwardBackwardCalculator::getScalingFactors() const {
    Vector scales(obsSize_);
    for (std::size_t i = 0; i < obsSize_; ++i) {
        scales(i) = (*alignedScales_)[i];
    }
    return scales;
}

// SIMDScaledOps implementations
void SIMDScaledOps::multiply_arrays(const double* a, const double* b, double* result, std::size_t size) {
    if (hasAVX()) {
        multiply_arrays_avx(a, b, result, size);
    } else if (hasSSE2()) {
        multiply_arrays_sse2(a, b, result, size);
    } else {
        multiply_arrays_scalar(a, b, result, size);
    }
}

void SIMDScaledOps::add_arrays(const double* a, const double* b, double* result, std::size_t size) {
    if (hasAVX()) {
        add_arrays_avx(a, b, result, size);
    } else if (hasSSE2()) {
        add_arrays_sse2(a, b, result, size);
    } else {
        add_arrays_scalar(a, b, result, size);
    }
}

void SIMDScaledOps::scale_array(const double* input, double scale, double* result, std::size_t size) {
    if (hasAVX()) {
        scale_array_avx(input, scale, result, size);
    } else if (hasSSE2()) {
        scale_array_sse2(input, scale, result, size);
    } else {
        scale_array_scalar(input, scale, result, size);
    }
}

double SIMDScaledOps::sum_array(const double* array, std::size_t size) {
    if (hasAVX()) {
        return sum_array_avx(array, size);
    } else if (hasSSE2()) {
        return sum_array_sse2(array, size);
    } else {
        return sum_array_scalar(array, size);
    }
}

double SIMDScaledOps::dot_product(const double* a, const double* b, std::size_t size) {
    double result = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

double SIMDScaledOps::max_element(const double* array, std::size_t size) {
    double maxVal = array[0];
    for (std::size_t i = 1; i < size; ++i) {
        if (array[i] > maxVal) {
            maxVal = array[i];
        }
    }
    return maxVal;
}

bool SIMDScaledOps::hasAVX() noexcept {
#ifdef LIBHMM_HAS_AVX
    return true;
#else
    return false;
#endif
}

bool SIMDScaledOps::hasSSE2() noexcept {
#ifdef LIBHMM_HAS_SSE2
    return true;
#else
    return false;
#endif
}

// AVX implementations
#ifdef LIBHMM_HAS_AVX
void SIMDScaledOps::multiply_arrays_avx(const double* a, const double* b, double* result, std::size_t size) {
    std::size_t simdSize = (size / 4) * 4;
    
    for (std::size_t i = 0; i < simdSize; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vr = _mm256_mul_pd(va, vb);
        _mm256_store_pd(&result[i], vr);
    }
    
    // Handle remaining elements
    for (std::size_t i = simdSize; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SIMDScaledOps::add_arrays_avx(const double* a, const double* b, double* result, std::size_t size) {
    std::size_t simdSize = (size / 4) * 4;
    
    for (std::size_t i = 0; i < simdSize; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vr = _mm256_add_pd(va, vb);
        _mm256_store_pd(&result[i], vr);
    }
    
    // Handle remaining elements
    for (std::size_t i = simdSize; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDScaledOps::scale_array_avx(const double* input, double scale, double* result, std::size_t size) {
    std::size_t simdSize = (size / 4) * 4;
    __m256d vscale = _mm256_set1_pd(scale);
    
    for (std::size_t i = 0; i < simdSize; i += 4) {
        __m256d vin = _mm256_load_pd(&input[i]);
        __m256d vr = _mm256_mul_pd(vin, vscale);
        _mm256_store_pd(&result[i], vr);
    }
    
    // Handle remaining elements
    for (std::size_t i = simdSize; i < size; ++i) {
        result[i] = input[i] * scale;
    }
}

double SIMDScaledOps::sum_array_avx(const double* array, std::size_t size) {
    std::size_t simdSize = (size / 4) * 4;
    __m256d vsum = _mm256_setzero_pd();
    
    for (std::size_t i = 0; i < simdSize; i += 4) {
        __m256d va = _mm256_load_pd(&array[i]);
        vsum = _mm256_add_pd(vsum, va);
    }
    
    // Horizontal sum
    __m128d vlow = _mm256_castpd256_pd128(vsum);
    __m128d vhigh = _mm256_extractf128_pd(vsum, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    double result = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
    
    // Handle remaining elements
    for (std::size_t i = simdSize; i < size; ++i) {
        result += array[i];
    }
    
    return result;
}
#endif

// SSE2 implementations
#ifdef LIBHMM_HAS_SSE2
void SIMDScaledOps::multiply_arrays_sse2(const double* a, const double* b, double* result, std::size_t size) {
    std::size_t simdSize = (size / 2) * 2;
    
    for (std::size_t i = 0; i < simdSize; i += 2) {
        __m128d va = _mm_load_pd(&a[i]);
        __m128d vb = _mm_load_pd(&b[i]);
        __m128d vr = _mm_mul_pd(va, vb);
        _mm_store_pd(&result[i], vr);
    }
    
    // Handle remaining elements
    for (std::size_t i = simdSize; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SIMDScaledOps::add_arrays_sse2(const double* a, const double* b, double* result, std::size_t size) {
    std::size_t simdSize = (size / 2) * 2;
    
    for (std::size_t i = 0; i < simdSize; i += 2) {
        __m128d va = _mm_load_pd(&a[i]);
        __m128d vb = _mm_load_pd(&b[i]);
        __m128d vr = _mm_add_pd(va, vb);
        _mm_store_pd(&result[i], vr);
    }
    
    // Handle remaining elements
    for (std::size_t i = simdSize; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDScaledOps::scale_array_sse2(const double* input, double scale, double* result, std::size_t size) {
    std::size_t simdSize = (size / 2) * 2;
    __m128d vscale = _mm_set1_pd(scale);
    
    for (std::size_t i = 0; i < simdSize; i += 2) {
        __m128d vin = _mm_load_pd(&input[i]);
        __m128d vr = _mm_mul_pd(vin, vscale);
        _mm_store_pd(&result[i], vr);
    }
    
    // Handle remaining elements
    for (std::size_t i = simdSize; i < size; ++i) {
        result[i] = input[i] * scale;
    }
}

double SIMDScaledOps::sum_array_sse2(const double* array, std::size_t size) {
    std::size_t simdSize = (size / 2) * 2;
    __m128d vsum = _mm_setzero_pd();
    
    for (std::size_t i = 0; i < simdSize; i += 2) {
        __m128d va = _mm_load_pd(&array[i]);
        vsum = _mm_add_pd(vsum, va);
    }
    
    // Horizontal sum
    __m128d high64 = _mm_unpackhi_pd(vsum, vsum);
    double result = _mm_cvtsd_f64(_mm_add_sd(vsum, high64));
    
    // Handle remaining elements
    for (std::size_t i = simdSize; i < size; ++i) {
        result += array[i];
    }
    
    return result;
}
#endif

// Scalar fallback implementations
void SIMDScaledOps::multiply_arrays_scalar(const double* a, const double* b, double* result, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SIMDScaledOps::add_arrays_scalar(const double* a, const double* b, double* result, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDScaledOps::scale_array_scalar(const double* input, double scale, double* result, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = input[i] * scale;
    }
}

double SIMDScaledOps::sum_array_scalar(const double* array, std::size_t size) {
    double sum = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        sum += array[i];
    }
    return sum;
}

} // namespace libhmm
