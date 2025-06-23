#ifndef LIBHMM_OPTIMIZED_FORWARD_BACKWARD_CALCULATOR_H_
#define LIBHMM_OPTIMIZED_FORWARD_BACKWARD_CALCULATOR_H_

#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/performance/simd_support.h"
#include <memory>
#include <vector>
#include <string>
#include <mutex>

namespace libhmm {

/// High-performance forward-backward calculator using SIMD optimizations
/// Provides significant speedup for larger HMMs through vectorized operations
class OptimizedForwardBackwardCalculator : public ForwardBackwardCalculator {
private:
    /// Aligned storage for SIMD operations
    using AlignedVector = std::vector<double, performance::aligned_allocator<double>>;
    
    /// Cached aligned matrices for efficient SIMD operations
    mutable std::unique_ptr<AlignedVector> alignedForward_;
    mutable std::unique_ptr<AlignedVector> alignedBackward_;
    mutable std::unique_ptr<AlignedVector> alignedTrans_;
    mutable std::unique_ptr<AlignedVector> alignedPi_;
    
    /// Matrix dimensions for cache efficiency
    std::size_t numStates_;
    std::size_t obsSize_;
    std::size_t alignedStateSize_;  // Padded to SIMD alignment
    
    /// Performance optimization flags
    bool useBlockedComputation_;
    std::size_t blockSize_;
    
    /// Initialize aligned storage and copy data for SIMD operations
    void initializeAlignedStorage();
    
    /// Copy matrix data to aligned storage with padding
    void copyToAlignedStorage(const Matrix& source, AlignedVector& dest, 
                              std::size_t rows, std::size_t cols, std::size_t alignedCols);
    
    /// Copy aligned storage back to boost matrix
    void copyFromAlignedStorage(const AlignedVector& source, Matrix& dest,
                                std::size_t rows, std::size_t cols, std::size_t alignedCols);
    
    /// SIMD-optimized emission probability computation
    void computeEmissionProbabilities(std::size_t t, AlignedVector& emissions) const;
    
    /// Blocked matrix-vector multiplication for better cache performance
    void blockedMatrixVectorMultiply(const double* matrix, const double* vector,
                                     double* result, std::size_t rows, std::size_t cols,
                                     std::size_t blockSize) const;
    
    /// SIMD-optimized transposed matrix-vector multiplication
    /// Computes result[j] = sum_i(matrix[i * cols + j] * vector[i])
    void matrixVectorMultiplyTransposed(const double* matrix, const double* vector,
                                        double* result, std::size_t rows, std::size_t cols) const;
    
    /// Blocked transposed matrix-vector multiplication for large matrices
    void blockedMatrixVectorMultiplyTransposed(const double* matrix, const double* vector,
                                               double* result, std::size_t rows, std::size_t cols,
                                               std::size_t blockSize) const;

protected:
    /// SIMD-optimized forward algorithm implementation
    void forward() override;
    
    /// SIMD-optimized backward algorithm implementation  
    void backward() override;

public:
    /// Constructor with HMM and observations
    /// @param hmm Pointer to the HMM (must not be null)
    /// @param observations The observation set to process
    /// @param useBlocking Enable blocked computation for large matrices
    /// @param blockSize Block size for cache optimization (0 = auto-detect)
    /// @throws std::invalid_argument if hmm is null
    OptimizedForwardBackwardCalculator(Hmm* hmm, const ObservationSet& observations,
                                       bool useBlocking = true, std::size_t blockSize = 0);
    
    /// Virtual destructor
    virtual ~OptimizedForwardBackwardCalculator() = default;
    
    /// Get performance information
    /// @return String describing optimizations used
    std::string getOptimizationInfo() const;
    
    /// Check if SIMD optimizations are available
    /// @return True if SIMD is available and being used
    bool isSIMDOptimized() const noexcept {
        return performance::simd_available();
    }
    
    /// Get recommended block size for this system
    /// @param numStates Number of HMM states
    /// @return Optimal block size for cache efficiency
    static std::size_t getRecommendedBlockSize(std::size_t numStates) noexcept;
};

/// High-performance memory pool for frequent allocations
class CalculatorMemoryPool {
private:
    struct PoolBlock {
        std::unique_ptr<performance::aligned_allocator<double>::pointer[]> memory;
        std::size_t size;
        bool inUse;
        
        PoolBlock(std::size_t sz) : size(sz), inUse(false) {
            performance::aligned_allocator<double> alloc;
            memory.reset(new performance::aligned_allocator<double>::pointer[1]);
            memory[0] = alloc.allocate(sz);
        }
        
        ~PoolBlock() {
            if (memory && memory[0]) {
                performance::aligned_allocator<double> alloc;
                alloc.deallocate(memory[0], size);
            }
        }
    };
    
    std::vector<std::unique_ptr<PoolBlock>> blocks_;
    mutable std::mutex poolMutex_;
    
public:
    /// Get memory block of specified size
    /// @param size Required size in doubles
    /// @return Pointer to aligned memory block
    double* acquire(std::size_t size);
    
    /// Release memory block back to pool
    /// @param ptr Pointer to release
    void release(double* ptr) noexcept;
    
    /// Get singleton instance
    static CalculatorMemoryPool& getInstance() {
        static CalculatorMemoryPool instance;
        return instance;
    }
    
    /// Clear all cached memory (for cleanup)
    void clear();
};

} // namespace libhmm

#endif // LIBHMM_OPTIMIZED_FORWARD_BACKWARD_CALCULATOR_H_
