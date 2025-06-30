#include "libhmm/hmm.h"
#include "libhmm/calculators/forward_backward_traits.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/scaled_simd_forward_backward_calculator.h"
#include "libhmm/calculators/log_simd_forward_backward_calculator.h"
#include "libhmm/performance/thread_pool.h"
#include <iostream>
#include <chrono>
#include <vector>

using namespace libhmm;

int main() {
    std::cout << "Testing Parallel Calculator Performance\n";
    std::cout << "========================================\n\n";
    
    // Create a larger HMM for testing parallelization
    const int numStates = 20;  // Large enough to trigger parallel computation
    const int seqLength = 100; // Long enough to see benefits
    
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
    
    std::cout << "Problem characteristics:\n";
    std::cout << "  States: " << numStates << "\n";
    std::cout << "  Sequence length: " << seqLength << "\n";
    std::cout << "  Thread pool threads: " << performance::ThreadPool::getOptimalThreadCount() << "\n\n";
    
    // Test automatic calculator selection
    auto characteristics = forwardbackward::ProblemCharacteristics(&hmm, observations);
    auto selectedType = forwardbackward::CalculatorSelector::selectOptimal(characteristics);
    auto traits = forwardbackward::CalculatorSelector::getTraits(selectedType);
    
    std::cout << "Automatic selection chose: " << traits.name << "\n";
    std::cout << "  Supports parallel: " << (traits.supportsParallel ? "Yes" : "No") << "\n";
    std::cout << "  Uses SIMD: " << (traits.usesSIMD ? "Yes" : "No") << "\n";
    std::cout << "  Numerically stable: " << (traits.numericallyStable ? "Yes" : "No") << "\n\n";
    
    // Create and time the selected calculator
    std::cout << "Running performance test...\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    auto calculator = forwardbackward::CalculatorSelector::createOptimal(&hmm, observations);
    
    // Cast to get probability method - different calculators have different interfaces
    double probability = 0.0;
    if (auto fbCalc = dynamic_cast<ForwardBackwardCalculator*>(calculator.get())) {
        probability = fbCalc->probability();
    } else if (auto scaledCalc = dynamic_cast<ScaledSIMDForwardBackwardCalculator*>(calculator.get())) {
        probability = scaledCalc->getProbability();
    } else if (auto logCalc = dynamic_cast<LogSIMDForwardBackwardCalculator*>(calculator.get())) {
        probability = logCalc->getProbability();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Results:\n";
    std::cout << "  Probability: " << probability << "\n";
    std::cout << "  Computation time: " << duration.count() << " microseconds\n";
    std::cout << "  Performance: " << (numStates * seqLength * 1000.0 / duration.count()) << " state-steps/ms\n\n";
    
    // Test parallelization benefits by comparing traits
    std::cout << "Performance predictions for all calculator types:\n";
    std::cout << forwardbackward::CalculatorSelector::getPerformanceComparison(characteristics) << "\n";
    
    std::cout << "Parallel calculator implementation completed successfully!\n";
    
    return 0;
}
