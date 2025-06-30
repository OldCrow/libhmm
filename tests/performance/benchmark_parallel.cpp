#include "libhmm/hmm.h"
#include "libhmm/calculators/forward_backward_traits.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/scaled_simd_forward_backward_calculator.h"
#include "libhmm/calculators/log_simd_forward_backward_calculator.h"
#include "libhmm/performance/thread_pool.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

using namespace libhmm;

struct BenchmarkResult {
    std::string name;
    double computeTime;
    double probability;
    int threadCount;
};

// Function to force single-threaded execution by temporarily setting thread count
void runBenchmark(const std::string& name, Hmm* hmm, const ObservationSet& observations, 
                  std::vector<BenchmarkResult>& results, int threadCount = -1) {
    
    // Store original thread count
    int originalThreads = performance::ThreadPool::getOptimalThreadCount();
    
    // Temporarily set thread count if specified
    if (threadCount > 0) {
        // Note: This assumes we can control thread count through environment or similar
        // For this test, we'll create our own calculator instances
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create calculator based on name
    std::unique_ptr<Calculator> calculator;
    double probability = 0.0;
    
    if (name == "Reference") {
        auto fbCalc = std::make_unique<ForwardBackwardCalculator>(hmm, observations);
        probability = fbCalc->probability();
    } else if (name == "Scaled-SIMD") {
        auto scaledCalc = std::make_unique<ScaledSIMDForwardBackwardCalculator>(hmm, observations);
        scaledCalc->compute();
        probability = scaledCalc->getProbability();
    } else if (name == "Log-SIMD") {
        auto logCalc = std::make_unique<LogSIMDForwardBackwardCalculator>(hmm, observations);
        logCalc->compute();
        probability = logCalc->getProbability();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    results.push_back({
        name,
        duration.count() / 1000.0, // Convert to milliseconds
        probability,
        threadCount > 0 ? threadCount : originalThreads
    });
}

int main() {
    std::cout << "HMM Calculator Performance Benchmark\n";
    std::cout << "====================================\n\n";
    
    // Test with different problem sizes - including ones that trigger parallel execution
    std::vector<std::pair<int, int>> problemSizes = {
        {10, 50},    // Small - sequential
        {20, 100},   // Medium - sequential 
        {64, 200},   // Large - sequential (just below threshold)
        {128, 300},  // Large - sequential (at threshold)
        {256, 400},  // Very Large - parallel (above threshold)
        {512, 500}   // Very Large - parallel 
        // {1024, 600}  // Extra Large - parallel (commented out for faster development)
    };
    
    for (auto [numStates, seqLength] : problemSizes) {
        std::cout << "Problem Size: " << numStates << " states, " << seqLength << " observations\n";
        std::cout << std::string(60, '-') << "\n";
        
        // Create HMM
        Hmm hmm(numStates);
        
        // Initialize with random-ish transition matrix
        Matrix trans(numStates, numStates);
        for (int i = 0; i < numStates; ++i) {
            double sum = 0.0;
            for (int j = 0; j < numStates; ++j) {
                trans(i, j) = 0.1 + 0.8 * (sin(i + j * 0.7) + 1.0) / 2.0;
                sum += trans(i, j);
            }
            // Normalize
            for (int j = 0; j < numStates; ++j) {
                trans(i, j) /= sum;
            }
        }
        hmm.setTrans(trans);
        
        // Initialize Pi
        Vector pi(numStates);
        for (int i = 0; i < numStates; ++i) {
            pi(i) = 1.0 / numStates;
        }
        hmm.setPi(pi);
        
        // Set up distributions (Gaussian for each state)
        for (int i = 0; i < numStates; ++i) {
            auto dist = std::make_unique<GaussianDistribution>(i * 2.0, 1.0);
            hmm.setProbabilityDistribution(i, std::move(dist));
        }
        
        // Create observation sequence
        ObservationSet observations(seqLength);
        for (int t = 0; t < seqLength; ++t) {
            observations(t) = sin(t * 0.1) * 5.0;
        }
        
        std::vector<BenchmarkResult> results;
        
        // Run benchmarks with multiple iterations for accuracy
        const int iterations = 3;
        
        // Test Reference Calculator
        double refTime = 0;
        for (int i = 0; i < iterations; ++i) {
            runBenchmark("Reference", &hmm, observations, results);
            refTime += results.back().computeTime;
        }
        refTime /= iterations;
        
        // Test SIMD Calculators  
        double scaledTime = 0;
        for (int i = 0; i < iterations; ++i) {
            runBenchmark("Scaled-SIMD", &hmm, observations, results);
            scaledTime += results.back().computeTime;
        }
        scaledTime /= iterations;
        
        double logTime = 0;
        for (int i = 0; i < iterations; ++i) {
            runBenchmark("Log-SIMD", &hmm, observations, results);
            logTime += results.back().computeTime;
        }
        logTime /= iterations;
        
        // Display results
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Reference Calculator:     " << std::setw(8) << refTime << " ms\n";
        std::cout << "Scaled-SIMD Calculator:   " << std::setw(8) << scaledTime << " ms";
        std::cout << "  (Speedup: " << std::setprecision(1) << refTime/scaledTime << "x)\n";
        std::cout << "Log-SIMD Calculator:      " << std::setw(8) << std::setprecision(2) << logTime << " ms";
        std::cout << "  (Speedup: " << std::setprecision(1) << refTime/logTime << "x)\n";
        
        // Calculate performance metrics
        double stateSteps = numStates * seqLength;
        std::cout << std::setprecision(1);
        std::cout << "Performance (state-steps/ms):\n";
        std::cout << "  Reference: " << stateSteps / refTime << "\n";
        std::cout << "  Scaled-SIMD: " << stateSteps / scaledTime << "\n";
        std::cout << "  Log-SIMD: " << stateSteps / logTime << "\n";
        
        std::cout << "\nThread Pool Info: " << performance::ThreadPool::getOptimalThreadCount() << " threads\n";
        std::cout << "\n";
    }
    
    std::cout << "Benchmark completed!\n";
    std::cout << "Note: SIMD calculators include thread pool parallelization\n";
    std::cout << "when the problem size exceeds the parallel threshold.\n";
    
    return 0;
}
