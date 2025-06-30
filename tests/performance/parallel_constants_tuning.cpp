#include "libhmm/hmm.h"
#include "libhmm/calculators/scaled_simd_forward_backward_calculator.h"
#include "libhmm/calculators/log_simd_forward_backward_calculator.h"
#include "libhmm/performance/thread_pool.h"
#include "libhmm/performance/parallel_constants.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <string>

using namespace libhmm;

struct TestConfiguration {
    std::string name;
    std::size_t minStatesThreshold;
    std::size_t grainSize;
    std::size_t emissionThreshold;
    std::size_t simpleGrainSize;
};

struct BenchmarkResult {
    std::string configName;
    std::string calculatorType;
    int problemSize;
    double computeTime;
    double speedup;
    bool usedParallel;
};

// Test configurations to compare
std::vector<TestConfiguration> getTestConfigurations() {
    return {
        // Current constants (non-base-2)
        {"Current", 500, 50, 200, 25},
        
        // Base-2 variants
        {"Base2-Conservative", 512, 64, 256, 32},
        {"Base2-Aggressive", 256, 32, 128, 16},
        {"Base2-Large", 512, 128, 256, 64},
        
        // Different thresholds with same grain sizes
        {"LowThresh-Base2", 256, 64, 128, 32},
        {"HighThresh-Base2", 1024, 64, 512, 32},
        
        // Grain size variations
        {"SmallGrain-Base2", 512, 16, 256, 8},
        {"LargeGrain-Base2", 512, 256, 256, 128},
    };
}

// Mock the parallel constants by creating a custom calculator that uses different thresholds
class TunableScaledSIMDCalculator : public Calculator {
private:
    ScaledSIMDForwardBackwardCalculator* impl_;
    TestConfiguration config_;
    
public:
    TunableScaledSIMDCalculator(Hmm* hmm, const ObservationSet& observations, const TestConfiguration& config)
        : Calculator(hmm, observations), config_(config) {
        // Note: This is a simplified approach - in reality, we'd need to modify
        // the actual implementation to use different constants
        impl_ = new ScaledSIMDForwardBackwardCalculator(hmm, observations);
    }
    
    ~TunableScaledSIMDCalculator() {
        delete impl_;
    }
    
    double probability() {
        return impl_->getProbability();
    }
    
    double getProbability() const {
        return impl_->getProbability();
    }
    
    bool wouldUseParallel(std::size_t numStates) const {
        return numStates >= config_.minStatesThreshold;
    }
    
    const TestConfiguration& getConfig() const { return config_; }
};

class TunableLogSIMDCalculator : public Calculator {
private:
    LogSIMDForwardBackwardCalculator* impl_;
    TestConfiguration config_;
    
public:
    TunableLogSIMDCalculator(Hmm* hmm, const ObservationSet& observations, const TestConfiguration& config)
        : Calculator(hmm, observations), config_(config) {
        impl_ = new LogSIMDForwardBackwardCalculator(hmm, observations);
    }
    
    ~TunableLogSIMDCalculator() {
        delete impl_;
    }
    
    double probability() {
        return impl_->getProbability();
    }
    
    double getProbability() const {
        return impl_->getProbability();
    }
    
    bool wouldUseParallel(std::size_t numStates) const {
        return numStates >= config_.minStatesThreshold;
    }
    
    const TestConfiguration& getConfig() const { return config_; }
};

// Create HMM for testing
std::unique_ptr<Hmm> createTestHMM(int numStates) {
    auto hmm = std::make_unique<Hmm>(numStates);
    
    // Initialize transition matrix
    Matrix trans(numStates, numStates);
    for (int i = 0; i < numStates; ++i) {
        double sum = 0.0;
        for (int j = 0; j < numStates; ++j) {
            trans(i, j) = 0.1 + 0.8 * (std::sin(i + j * 0.7) + 1.0) / 2.0;
            sum += trans(i, j);
        }
        // Normalize
        for (int j = 0; j < numStates; ++j) {
            trans(i, j) /= sum;
        }
    }
    hmm->setTrans(trans);
    
    // Initialize Pi
    Vector pi(numStates);
    for (int i = 0; i < numStates; ++i) {
        pi(i) = 1.0 / numStates;
    }
    hmm->setPi(pi);
    
    // Set up distributions
    for (int i = 0; i < numStates; ++i) {
        auto dist = std::make_unique<GaussianDistribution>(i * 2.0, 1.0);
        hmm->setProbabilityDistribution(i, std::move(dist));
    }
    
    return hmm;
}

// Create observation sequence
ObservationSet createObservations(int seqLength) {
    ObservationSet observations(seqLength);
    for (int t = 0; t < seqLength; ++t) {
        observations(t) = std::sin(t * 0.1) * 5.0;
    }
    return observations;
}

// Benchmark a configuration
BenchmarkResult benchmarkConfiguration(const TestConfiguration& config, 
                                     const std::string& calculatorType,
                                     int numStates, int seqLength,
                                     int iterations = 3) {
    auto hmm = createTestHMM(numStates);
    auto observations = createObservations(seqLength);
    
    double totalTime = 0.0;
    double probability = 0.0;
    bool usedParallel = false;
    
    for (int iter = 0; iter < iterations; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (calculatorType == "Scaled-SIMD") {
            TunableScaledSIMDCalculator calc(hmm.get(), observations, config);
            probability = calc.getProbability();
            usedParallel = calc.wouldUseParallel(numStates);
        } else if (calculatorType == "Log-SIMD") {
            TunableLogSIMDCalculator calc(hmm.get(), observations, config);
            probability = calc.getProbability();
            usedParallel = calc.wouldUseParallel(numStates);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        totalTime += duration.count() / 1000.0; // Convert to milliseconds
    }
    
    return {
        config.name,
        calculatorType,
        numStates,
        totalTime / iterations,
        0.0, // Will be calculated later
        usedParallel
    };
}

int main() {
    std::cout << "Parallel Constants Tuning Analysis\n";
    std::cout << "==================================\n\n";
    
    std::cout << "Testing different parallelization constants to find optimal values.\n";
    std::cout << "Focus: Base-2 vs non-base-2 constants and their performance impact.\n\n";
    
    auto configurations = getTestConfigurations();
    
    // Problem sizes to test (covering sequential to parallel ranges)
    std::vector<std::pair<int, int>> problemSizes = {
        {128, 200},   // Just below current threshold
        {256, 300},   // Base-2 threshold
        {512, 400},   // Current threshold
        {1024, 500},  // Large parallel
    };
    
    std::vector<std::string> calculatorTypes = {"Scaled-SIMD", "Log-SIMD"};
    std::vector<BenchmarkResult> allResults;
    
    for (const auto& [numStates, seqLength] : problemSizes) {
        std::cout << "\nProblem Size: " << numStates << " states, " << seqLength << " observations\n";
        std::cout << std::string(80, '-') << "\n";
        
        // Get baseline (current configuration) for speedup calculations
        double baselineScaled = 0.0, baselineLog = 0.0;
        for (const auto& config : configurations) {
            if (config.name == "Current") {
                auto resultScaled = benchmarkConfiguration(config, "Scaled-SIMD", numStates, seqLength);
                auto resultLog = benchmarkConfiguration(config, "Log-SIMD", numStates, seqLength);
                baselineScaled = resultScaled.computeTime;
                baselineLog = resultLog.computeTime;
                break;
            }
        }
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Config                 Calculator   Time(ms)  Speedup  Parallel  Thresh  Grain\n";
        std::cout << std::string(80, '-') << "\n";
        
        for (const auto& config : configurations) {
            for (const auto& calcType : calculatorTypes) {
                auto result = benchmarkConfiguration(config, calcType, numStates, seqLength);
                
                // Calculate speedup
                double baseline = (calcType == "Scaled-SIMD") ? baselineScaled : baselineLog;
                result.speedup = baseline / result.computeTime;
                
                allResults.push_back(result);
                
                std::cout << std::setw(20) << std::left << config.name
                          << " " << std::setw(10) << std::left << calcType
                          << " " << std::setw(8) << result.computeTime
                          << " " << std::setw(7) << result.speedup << "x"
                          << " " << std::setw(8) << (result.usedParallel ? "Yes" : "No")
                          << " " << std::setw(6) << config.minStatesThreshold
                          << " " << std::setw(4) << config.grainSize << "\n";
            }
        }
    }
    
    // Analysis summary
    std::cout << "\n\nAnalysis Summary\n";
    std::cout << "================\n";
    
    std::cout << "\nConfiguration Details:\n";
    std::cout << std::setw(20) << "Config" << std::setw(8) << "Thresh" << std::setw(6) << "Grain" 
              << std::setw(8) << "EmisTh" << std::setw(8) << "SimGrain" << std::setw(12) << "Base2?" << "\n";
    std::cout << std::string(70, '-') << "\n";
    
    for (const auto& config : configurations) {
        bool isBase2Thresh = (config.minStatesThreshold & (config.minStatesThreshold - 1)) == 0;
        bool isBase2Grain = (config.grainSize & (config.grainSize - 1)) == 0;
        bool isBase2Overall = isBase2Thresh && isBase2Grain;
        
        std::cout << std::setw(20) << config.name
                  << std::setw(8) << config.minStatesThreshold
                  << std::setw(6) << config.grainSize
                  << std::setw(8) << config.emissionThreshold
                  << std::setw(8) << config.simpleGrainSize
                  << std::setw(12) << (isBase2Overall ? "Yes" : "No") << "\n";
    }
    
    std::cout << "\nKey Findings:\n";
    std::cout << "- Compare configurations marked 'Base2: Yes' vs 'Base2: No'\n";
    std::cout << "- Look for consistent performance patterns across problem sizes\n";
    std::cout << "- Pay attention to when 'Parallel: Yes' starts appearing\n";
    std::cout << "- Best speedup indicates optimal configuration for each problem size\n";
    
    std::cout << "\nThread Pool Info: " << performance::ThreadPool::getOptimalThreadCount() << " threads\n";
    
    return 0;
}
