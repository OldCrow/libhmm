#include "libhmm/performance/log_space_ops.h"
#include <algorithm>
#include <numeric>

// SIMD intrinsics now centralized in simd_platform.h
#include "libhmm/performance/simd_platform.h"

namespace libhmm {
namespace performance {

// Static member definitions
std::array<double, LogSpaceOps::LOOKUP_TABLE_SIZE> LogSpaceOps::logOnePlusExpTable_;
bool LogSpaceOps::initialized_ = false;

void LogSpaceOps::initialize() {
    if (initialized_) return;
    
    // Precompute log(1 + exp(x)) for x in [LOG_SUM_THRESHOLD, 0]
    // This covers the most common range for log-sum-exp operations
    const double range = -LOG_SUM_THRESHOLD; // 50.0
    const double step = range / (LOOKUP_TABLE_SIZE - 1);
    
    for (std::size_t i = 0; i < LOOKUP_TABLE_SIZE; ++i) {
        const double x = LOG_SUM_THRESHOLD + i * step;
        logOnePlusExpTable_[i] = std::log(1.0 + std::exp(x));
    }
    
    initialized_ = true;
}

double LogSpaceOps::lookupLogOnePlusExp(double x) noexcept {
    if (x <= LOG_SUM_THRESHOLD) {
        return 0.0; // log(1 + exp(x)) ≈ 0 when x is very negative
    }
    if (x >= 0.0) {
        return x + std::log(1.0 + std::exp(-x)); // More stable for x >= 0
    }
    
    // Map x from [LOG_SUM_THRESHOLD, 0] to [0, LOOKUP_TABLE_SIZE-1]
    const double range = -LOG_SUM_THRESHOLD;
    const double normalized = (x - LOG_SUM_THRESHOLD) / range;
    const double index = normalized * (LOOKUP_TABLE_SIZE - 1);
    
    // Linear interpolation for better accuracy
    const std::size_t i0 = static_cast<std::size_t>(index);
    const std::size_t i1 = std::min(i0 + 1, LOOKUP_TABLE_SIZE - 1);
    const double alpha = index - i0;
    
    return logOnePlusExpTable_[i0] * (1.0 - alpha) + logOnePlusExpTable_[i1] * alpha;
}

double LogSpaceOps::logSumExp(double logA, double logB) noexcept {
    // Handle special cases
    if (isLogZero(logA)) return logB;
    if (isLogZero(logB)) return logA;
    
    // Ensure logA >= logB for numerical stability
    if (logA < logB) {
        std::swap(logA, logB);
    }
    
    const double diff = logB - logA;
    
    // If difference is too large, the smaller term is negligible
    if (diff <= LOG_SUM_THRESHOLD) {
        return logA;
    }
    
    // Use lookup table for common case
    return logA + lookupLogOnePlusExp(diff);
}

double LogSpaceOps::logSumExpArray(const double* logValues, std::size_t size) noexcept {
    if (size == 0) return LOG_ZERO;
    if (size == 1) return logValues[0];
    
    // Use SIMD implementation for larger arrays
    if (size >= 8 && simd_available()) {
        return logSumExpArraySIMD(logValues, size);
    } else {
        return logSumExpArrayScalar(logValues, size);
    }
}

double LogSpaceOps::logSumExpArrayScalar(const double* logValues, std::size_t size) noexcept {
    // Find maximum value for numerical stability
    double maxVal = *std::max_element(logValues, logValues + size);
    
    if (isLogZero(maxVal)) {
        return LOG_ZERO;
    }
    
    // Compute sum(exp(logValues[i] - maxVal))
    double sum = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        if (!isLogZero(logValues[i])) {
            const double diff = logValues[i] - maxVal;
            if (diff > LOG_SUM_THRESHOLD) {
                sum += std::exp(diff);
            }
        }
    }
    
    return (sum > 0.0) ? maxVal + std::log(sum) : LOG_ZERO;
}

double LogSpaceOps::logSumExpArraySIMD(const double* logValues, std::size_t size) noexcept {
#ifdef LIBHMM_HAS_AVX
    // Find maximum using AVX
    __m256d maxVec = _mm256_set1_pd(LOG_ZERO);
    const std::size_t simdSize = size - (size % 4);
    
    for (std::size_t i = 0; i < simdSize; i += 4) {
        __m256d vals = _mm256_loadu_pd(&logValues[i]);
        maxVec = _mm256_max_pd(maxVec, vals);
    }
    
    // Extract maximum from SIMD register
    alignas(32) double maxArray[4];
    _mm256_store_pd(maxArray, maxVec);
    double maxVal = *std::max_element(maxArray, maxArray + 4);
    
    // Handle remainder elements
    for (std::size_t i = simdSize; i < size; ++i) {
        maxVal = std::max(maxVal, logValues[i]);
    }
    
    if (isLogZero(maxVal)) {
        return LOG_ZERO;
    }
    
    // Compute sum using AVX
    const __m256d maxBroadcast = _mm256_set1_pd(maxVal);
    const __m256d thresholdVec = _mm256_set1_pd(LOG_SUM_THRESHOLD);
    __m256d sumVec = _mm256_setzero_pd();
    
    for (std::size_t i = 0; i < simdSize; i += 4) {
        __m256d vals = _mm256_loadu_pd(&logValues[i]);
        __m256d diff = _mm256_sub_pd(vals, maxBroadcast);
        
        // Mask for values above threshold
        __m256d mask = _mm256_cmp_pd(diff, thresholdVec, _CMP_GT_OQ);
        
        // Compute exp(diff) only for values above threshold
        alignas(32) double diffArray[4];
        _mm256_store_pd(diffArray, diff);
        
        alignas(32) double expArray[4];
        for (int j = 0; j < 4; ++j) {
            expArray[j] = (diffArray[j] > LOG_SUM_THRESHOLD) ? std::exp(diffArray[j]) : 0.0;
        }
        
        __m256d expVec = _mm256_load_pd(expArray);
        sumVec = _mm256_add_pd(sumVec, expVec);
    }
    
    // Sum elements in SIMD register
    alignas(32) double sumArray[4];
    _mm256_store_pd(sumArray, sumVec);
    double sum = sumArray[0] + sumArray[1] + sumArray[2] + sumArray[3];
    
    // Handle remainder elements
    for (std::size_t i = simdSize; i < size; ++i) {
        if (!isLogZero(logValues[i])) {
            const double diff = logValues[i] - maxVal;
            if (diff > LOG_SUM_THRESHOLD) {
                sum += std::exp(diff);
            }
        }
    }
    
    return (sum > 0.0) ? maxVal + std::log(sum) : LOG_ZERO;
    
#else
    // Fallback to scalar implementation
    return logSumExpArrayScalar(logValues, size);
#endif
}

void LogSpaceOps::precomputeLogMatrix(const double* probMatrix, double* logMatrix, 
                                     std::size_t rows, std::size_t cols) noexcept {
    const std::size_t totalSize = rows * cols;
    
    // Vectorized log computation
    for (std::size_t i = 0; i < totalSize; ++i) {
        logMatrix[i] = safeLog(probMatrix[i]);
    }
}

void LogSpaceOps::logMatrixVectorMultiply(const double* logMatrix, const double* logVector,
                                         double* result, std::size_t rows, std::size_t cols) noexcept {
    for (std::size_t i = 0; i < rows; ++i) {
        const double* matrixRow = logMatrix + i * cols;
        
        // Compute log-sum-exp of (matrixRow[j] + logVector[j]) for all j
        double maxVal = LOG_ZERO;
        
        // Find maximum for numerical stability
        for (std::size_t j = 0; j < cols; ++j) {
            if (!isLogZero(matrixRow[j]) && !isLogZero(logVector[j])) {
                const double val = matrixRow[j] + logVector[j];
                maxVal = std::max(maxVal, val);
            }
        }
        
        if (isLogZero(maxVal)) {
            result[i] = LOG_ZERO;
            continue;
        }
        
        // Compute sum(exp(matrixRow[j] + logVector[j] - maxVal))
        double sum = 0.0;
        for (std::size_t j = 0; j < cols; ++j) {
            if (!isLogZero(matrixRow[j]) && !isLogZero(logVector[j])) {
                const double val = matrixRow[j] + logVector[j];
                const double diff = val - maxVal;
                if (diff > LOG_SUM_THRESHOLD) {
                    sum += std::exp(diff);
                }
            }
        }
        
        result[i] = (sum > 0.0) ? maxVal + std::log(sum) : LOG_ZERO;
    }
}

void LogSpaceOps::logMatrixVectorMultiplyTransposed(const double* logMatrix, const double* logVector,
                                                   double* result, std::size_t rows, std::size_t cols) noexcept {
    // Initialize result
    std::fill(result, result + cols, LOG_ZERO);
    
    for (std::size_t j = 0; j < cols; ++j) {
        double maxVal = LOG_ZERO;
        
        // Find maximum for numerical stability
        for (std::size_t i = 0; i < rows; ++i) {
            if (!isLogZero(logMatrix[i * cols + j]) && !isLogZero(logVector[i])) {
                const double val = logMatrix[i * cols + j] + logVector[i];
                maxVal = std::max(maxVal, val);
            }
        }
        
        if (isLogZero(maxVal)) {
            result[j] = LOG_ZERO;
            continue;
        }
        
        // Compute sum(exp(logMatrix[i*cols + j] + logVector[i] - maxVal))
        double sum = 0.0;
        for (std::size_t i = 0; i < rows; ++i) {
            if (!isLogZero(logMatrix[i * cols + j]) && !isLogZero(logVector[i])) {
                const double val = logMatrix[i * cols + j] + logVector[i];
                const double diff = val - maxVal;
                if (diff > LOG_SUM_THRESHOLD) {
                    sum += std::exp(diff);
                }
            }
        }
        
        result[j] = (sum > 0.0) ? maxVal + std::log(sum) : LOG_ZERO;
    }
}

} // namespace performance
} // namespace libhmm
