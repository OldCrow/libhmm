#include <gtest/gtest.h>
#include "libhmm/performance/simd_support.h"
#include "libhmm/performance/thread_pool.h"
#include "libhmm/calculators/optimized_forward_backward_calculator.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/two_state_hmm.h"
#include <memory>
#include <vector>
#include <random>
#include <cmath>

using namespace libhmm;
using namespace libhmm::performance;

class PerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test HMM
        hmm_ = createTwoStateHmm();
        
        // Create test observation sequence
        observations_ = ObservationSet(50);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 5);
        
        for (std::size_t i = 0; i < observations_.size(); ++i) {
            observations_(i) = dis(gen);
        }
        
        // Create aligned test vectors for SIMD tests
        alignedA_ = AlignedVector(64, 1.0);
        alignedB_ = AlignedVector(64, 2.0);
        alignedResult_ = AlignedVector(64, 0.0);
        
        // Fill with test data
        for (std::size_t i = 0; i < alignedA_.size(); ++i) {
            alignedA_[i] = static_cast<double>(i + 1);
            alignedB_[i] = static_cast<double>(i + 2);
        }
    }

    std::unique_ptr<Hmm> hmm_;
    ObservationSet observations_;
    
    using AlignedVector = std::vector<double, aligned_allocator<double>>;
    AlignedVector alignedA_;
    AlignedVector alignedB_;
    AlignedVector alignedResult_;
};

// SIMD Support Tests
TEST_F(PerformanceTest, SIMDAvailabilityCheck) {
    // This test will pass regardless of platform, but provides useful info
    const bool available = simd_available();
    std::cout << "SIMD Support Available: " << (available ? "Yes" : "No") << std::endl;
    
    if (available) {
#ifdef LIBHMM_HAS_AVX
        std::cout << "AVX Support: Available" << std::endl;
#elif defined(LIBHMM_HAS_SSE2)
        std::cout << "SSE2 Support: Available" << std::endl;
#elif defined(LIBHMM_HAS_NEON)
        std::cout << "ARM NEON Support: Available" << std::endl;
#endif
    }
}

TEST_F(PerformanceTest, SIMDDotProduct) {
    const std::size_t size = alignedA_.size();
    
    // Calculate expected result
    double expected = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        expected += alignedA_[i] * alignedB_[i];
    }
    
    // Test SIMD implementation
    const double result = SIMDOps::dot_product(alignedA_.data(), alignedB_.data(), size);
    
    EXPECT_NEAR(result, expected, 1e-10);
}

TEST_F(PerformanceTest, SIMDVectorOperations) {
    const std::size_t size = alignedA_.size();
    
    // Test vector addition
    SIMDOps::vector_add(alignedA_.data(), alignedB_.data(), alignedResult_.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(alignedResult_[i], alignedA_[i] + alignedB_[i], 1e-10);
    }
    
    // Test vector multiplication
    SIMDOps::vector_multiply(alignedA_.data(), alignedB_.data(), alignedResult_.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(alignedResult_[i], alignedA_[i] * alignedB_[i], 1e-10);
    }
    
    // Test scalar multiplication
    const double scalar = 3.5;
    SIMDOps::scalar_multiply(alignedA_.data(), scalar, alignedResult_.data(), size);
    for (std::size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(alignedResult_[i], alignedA_[i] * scalar, 1e-10);
    }
}

TEST_F(PerformanceTest, AlignedAllocator) {
    // Test aligned allocator
    aligned_allocator<double> alloc;
    
    const std::size_t count = 100;
    double* ptr = alloc.allocate(count);
    ASSERT_NE(ptr, nullptr);
    
    // Check alignment (should be aligned to SIMD_ALIGNMENT)
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(ptr);
    EXPECT_EQ(address % SIMD_ALIGNMENT, 0);
    
    // Test basic usage
    for (std::size_t i = 0; i < count; ++i) {
        ptr[i] = static_cast<double>(i);
    }
    
    for (std::size_t i = 0; i < count; ++i) {
        EXPECT_DOUBLE_EQ(ptr[i], static_cast<double>(i));
    }
    
    alloc.deallocate(ptr, count);
}

// Thread Pool Tests
TEST_F(PerformanceTest, ThreadPoolBasicFunctionality) {
    ThreadPool pool(2); // Use 2 threads for testing
    
    EXPECT_EQ(pool.getNumThreads(), 2);
    
    // Test simple task submission
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < 10; ++i) {
        auto future = pool.submit([&counter]() {
            counter++;
        });
        futures.push_back(std::move(future));
    }
    
    // Wait for all tasks
    for (auto& future : futures) {
        future.wait();
    }
    
    EXPECT_EQ(counter.load(), 10);
}

TEST_F(PerformanceTest, ThreadPoolWithReturnValues) {
    ThreadPool pool(2);
    
    // Test task with return value
    auto future = pool.submit([]() -> int {
        return 42;
    });
    
    EXPECT_EQ(future.get(), 42);
}

TEST_F(PerformanceTest, ParallelForExecution) {
    std::vector<int> data(1000, 0);
    
    // Parallel initialization
    ParallelUtils::parallelFor(0, data.size(), [&data](std::size_t i) {
        data[i] = static_cast<int>(i * 2);
    });
    
    // Verify results
    for (std::size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(data[i], static_cast<int>(i * 2));
    }
}

TEST_F(PerformanceTest, ParallelReduction) {
    const std::size_t count = 1000;
    
    // Calculate sum using parallel reduction
    const int result = ParallelUtils::parallelReduce<int>(
        0, count, 0,
        [](std::size_t i) -> int { return static_cast<int>(i); },
        [](int a, int b) -> int { return a + b; }
    );
    
    // Expected sum: 0 + 1 + 2 + ... + (count-1) = count*(count-1)/2
    const int expected = static_cast<int>(count * (count - 1) / 2);
    EXPECT_EQ(result, expected);
}

// CPU Info Tests
TEST_F(PerformanceTest, CpuInfoDetection) {
    const auto features = CpuInfo::getCpuFeatures();
    const auto cpuInfo = CpuInfo::getCpuInfoString();
    
    // These tests mainly verify the functions don't crash
    EXPECT_GE(CpuInfo::getLogicalCores(), 1);
    EXPECT_GE(CpuInfo::getPhysicalCores(), 1);
    EXPECT_GT(CpuInfo::getL1CacheSize(), 0);
    EXPECT_GT(CpuInfo::getL2CacheSize(), 0);
    EXPECT_GT(CpuInfo::getL3CacheSize(), 0);
    EXPECT_GT(CpuInfo::getCacheLineSize(), 0);
    EXPECT_FALSE(cpuInfo.empty());
    
    // Use the features variable to avoid unused variable warning
    // (Features is a complex type so we just verify it's accessible)
    EXPECT_NO_THROW(features.size());
    
    std::cout << "CPU Info:\n" << cpuInfo << std::endl;
}

// Optimized Calculator Tests
TEST_F(PerformanceTest, OptimizedForwardBackwardCalculator) {
    // Test optimized calculator
    OptimizedForwardBackwardCalculator optimizedCalc(hmm_.get(), observations_);
    
    // Test basic functionality
    EXPECT_NO_THROW(optimizedCalc.probability());
    
    const Matrix optimizedForward = optimizedCalc.getForwardVariables();
    const Matrix optimizedBackward = optimizedCalc.getBackwardVariables();
    
    EXPECT_EQ(optimizedForward.size1(), observations_.size());
    EXPECT_EQ(optimizedForward.size2(), static_cast<std::size_t>(hmm_->getNumStates()));
    EXPECT_EQ(optimizedBackward.size1(), observations_.size());
    EXPECT_EQ(optimizedBackward.size2(), static_cast<std::size_t>(hmm_->getNumStates()));
    
    // Compare with standard calculator
    ForwardBackwardCalculator standardCalc(hmm_.get(), observations_);
    
    const double optimizedProb = optimizedCalc.probability();
    const double standardProb = standardCalc.probability();
    
    EXPECT_NEAR(optimizedProb, standardProb, 1e-10);
    
    // Compare forward variables
    const Matrix standardForward = standardCalc.getForwardVariables();
    for (std::size_t i = 0; i < optimizedForward.size1(); ++i) {
        for (std::size_t j = 0; j < optimizedForward.size2(); ++j) {
            EXPECT_NEAR(optimizedForward(i, j), standardForward(i, j), 1e-10)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
    
    // Compare backward variables
    const Matrix standardBackward = standardCalc.getBackwardVariables();
    for (std::size_t i = 0; i < optimizedBackward.size1(); ++i) {
        for (std::size_t j = 0; j < optimizedBackward.size2(); ++j) {
            EXPECT_NEAR(optimizedBackward(i, j), standardBackward(i, j), 1e-10)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST_F(PerformanceTest, OptimizedCalculatorConfiguration) {
    OptimizedForwardBackwardCalculator calc(hmm_.get(), observations_, true, 32);
    
    EXPECT_TRUE(calc.isSIMDOptimized() == simd_available());
    
    const std::string info = calc.getOptimizationInfo();
    EXPECT_FALSE(info.empty());
    std::cout << "Optimization Info: " << info << std::endl;
    
    // Test recommended block size calculation
    const std::size_t blockSize = OptimizedForwardBackwardCalculator::getRecommendedBlockSize(10);
    EXPECT_GT(blockSize, 0);
    EXPECT_LE(blockSize, 10);
}

// Memory Pool Tests
TEST_F(PerformanceTest, CalculatorMemoryPool) {
    CalculatorMemoryPool& pool = CalculatorMemoryPool::getInstance();
    
    // Test acquisition and release
    const std::size_t size = 1000;
    double* ptr1 = pool.acquire(size);
    ASSERT_NE(ptr1, nullptr);
    
    double* ptr2 = pool.acquire(size * 2);
    ASSERT_NE(ptr2, nullptr);
    EXPECT_NE(ptr1, ptr2);
    
    // Test usage
    for (std::size_t i = 0; i < size; ++i) {
        ptr1[i] = static_cast<double>(i);
    }
    
    for (std::size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(ptr1[i], static_cast<double>(i));
    }
    
    pool.release(ptr1);
    pool.release(ptr2);
    
    // Test reuse (same size should reuse the block)
    double* ptr3 = pool.acquire(size);
    EXPECT_EQ(ptr1, ptr3); // Should reuse the first block
    
    pool.release(ptr3);
}

// Performance Comparison Test (informational)
TEST_F(PerformanceTest, DISABLED_PerformanceComparison) {
    // This test is disabled by default as it's for manual performance analysis
    
    const std::size_t numRuns = 100;
    const auto startStandard = std::chrono::high_resolution_clock::now();
    
    // Standard calculator benchmark
    for (std::size_t i = 0; i < numRuns; ++i) {
        ForwardBackwardCalculator calc(hmm_.get(), observations_);
        volatile double prob = calc.probability(); // volatile to prevent optimization
        (void)prob;
    }
    
    const auto endStandard = std::chrono::high_resolution_clock::now();
    const auto standardTime = std::chrono::duration<double>(endStandard - startStandard).count();
    
    const auto startOptimized = std::chrono::high_resolution_clock::now();
    
    // Optimized calculator benchmark
    for (std::size_t i = 0; i < numRuns; ++i) {
        OptimizedForwardBackwardCalculator calc(hmm_.get(), observations_);
        volatile double prob = calc.probability(); // volatile to prevent optimization
        (void)prob;
    }
    
    const auto endOptimized = std::chrono::high_resolution_clock::now();
    const auto optimizedTime = std::chrono::duration<double>(endOptimized - startOptimized).count();
    
    const double speedup = standardTime / optimizedTime;
    
    std::cout << "Performance Comparison (" << numRuns << " runs):" << std::endl;
    std::cout << "Standard Calculator: " << standardTime << "s" << std::endl;
    std::cout << "Optimized Calculator: " << optimizedTime << "s" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    
    // The optimized version should at least not be significantly slower
    EXPECT_LE(optimizedTime, standardTime * 1.5); // Allow 50% slower in worst case
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
