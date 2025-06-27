#include <gtest/gtest.h>
#include "libhmm/performance/simd_support.h"
#include "libhmm/performance/thread_pool.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/scaled_simd_forward_backward_calculator.h"
#include "libhmm/calculators/log_simd_forward_backward_calculator.h"
#include "libhmm/two_state_hmm.h"
#include <memory>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <atomic>
#include <future>

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

// SIMD Operation Tests
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

// Basic performance tests for available functionality

// SIMD Calculator Tests
TEST_F(PerformanceTest, ScaledSIMDForwardBackwardCalculator) {
    // Test SIMD calculator
    ScaledSIMDForwardBackwardCalculator simdCalc(hmm_.get(), observations_);
    
    // Test basic functionality
    EXPECT_NO_THROW(simdCalc.getProbability());
    
    const Matrix simdForward = simdCalc.getForwardVariablesCompat();
    const Matrix simdBackward = simdCalc.getBackwardVariablesCompat();
    
    EXPECT_EQ(simdForward.size1(), observations_.size());
    EXPECT_EQ(simdForward.size2(), static_cast<std::size_t>(hmm_->getNumStates()));
    EXPECT_EQ(simdBackward.size1(), observations_.size());
    EXPECT_EQ(simdBackward.size2(), static_cast<std::size_t>(hmm_->getNumStates()));
    
    // Test probability computation (should be valid)
    const double simdProb = simdCalc.getProbability();
    EXPECT_GT(simdProb, 0.0);
    EXPECT_LE(simdProb, 1.0);
    
    // Test log probability computation
    const double simdLogProb = simdCalc.getLogProbability();
    EXPECT_LE(simdLogProb, 0.0); // Log probability should be <= 0
    EXPECT_NEAR(std::log(simdProb), simdLogProb, 1e-10);
    
    // Test that forward and backward variables have reasonable values
    // (All should be positive and finite)
    for (std::size_t i = 0; i < simdForward.size1(); ++i) {
        for (std::size_t j = 0; j < simdForward.size2(); ++j) {
            EXPECT_GT(simdForward(i, j), 0.0) << "Forward variable negative at (" << i << ", " << j << ")";
            EXPECT_TRUE(std::isfinite(simdForward(i, j))) << "Forward variable not finite at (" << i << ", " << j << ")";
            
            EXPECT_GT(simdBackward(i, j), 0.0) << "Backward variable negative at (" << i << ", " << j << ")";
            EXPECT_TRUE(std::isfinite(simdBackward(i, j))) << "Backward variable not finite at (" << i << ", " << j << ")";
        }
    }
}

TEST_F(PerformanceTest, SIMDCalculatorConfiguration) {
    ScaledSIMDForwardBackwardCalculator calc(hmm_.get(), observations_, true, 32);
    
    EXPECT_TRUE(calc.isSIMDEnabled());
    
    const std::string info = calc.getOptimizationInfo();
    EXPECT_FALSE(info.empty());
    std::cout << "Optimization Info: " << info << std::endl;
    
    // Test recommended block size calculation
    const std::size_t blockSize = ScaledSIMDForwardBackwardCalculator::getRecommendedBlockSize(10);
    EXPECT_GT(blockSize, 0);
    EXPECT_LE(blockSize, 10);
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
    
    const auto startSIMD = std::chrono::high_resolution_clock::now();
    
    // SIMD calculator benchmark
    for (std::size_t i = 0; i < numRuns; ++i) {
        ScaledSIMDForwardBackwardCalculator calc(hmm_.get(), observations_);
        volatile double prob = calc.getProbability(); // volatile to prevent optimization
        (void)prob;
    }
    
    const auto endSIMD = std::chrono::high_resolution_clock::now();
    const auto simdTime = std::chrono::duration<double>(endSIMD - startSIMD).count();
    
    const double speedup = standardTime / simdTime;
    
    std::cout << "Performance Comparison (" << numRuns << " runs):" << std::endl;
    std::cout << "Standard Calculator: " << standardTime << "s" << std::endl;
    std::cout << "SIMD Calculator: " << simdTime << "s" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    
    // The SIMD version should at least not be significantly slower
    EXPECT_LE(simdTime, standardTime * 1.5); // Allow 50% slower in worst case
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
