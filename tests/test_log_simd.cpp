#include "libhmm/calculators/log_simd_forward_backward_calculator.h"
#include "libhmm/calculators/log_forward_backward_calculator.h"
#include "libhmm/hmm.h"
#include "libhmm/distributions/discrete_distribution.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>

using namespace libhmm;

void createDishonestCasinoHMM(std::unique_ptr<Hmm>& hmm) {
    // Create 2-state discrete HMM for dishonest casino
    hmm = std::make_unique<Hmm>(2);
    
    // Set initial probabilities: start with fair die
    Vector pi(2);
    pi(0) = 0.5; // Fair die
    pi(1) = 0.5; // Loaded die
    hmm->setPi(pi);
    
    // Set transition matrix
    Matrix trans(2, 2);
    trans(0, 0) = 0.95; trans(0, 1) = 0.05; // Fair to fair/loaded
    trans(1, 0) = 0.10; trans(1, 1) = 0.90; // Loaded to fair/loaded
    hmm->setTrans(trans);
    
    // Fair die (state 0): uniform distribution - create with 6 symbols
    auto fairDist = std::make_unique<DiscreteDistribution>(6);
    for (int i = 0; i < 6; ++i) {
        fairDist->setProbability(i, 1.0/6.0);
    }
    hmm->setProbabilityDistribution(0, std::move(fairDist));
    
    // Loaded die (state 1): favors 6
    auto loadedDist = std::make_unique<DiscreteDistribution>(6);
    std::vector<double> loadedProbs = {0.1, 0.1, 0.1, 0.1, 0.1, 0.5};
    for (int i = 0; i < 6; ++i) {
        loadedDist->setProbability(i, loadedProbs[i]);
    }
    hmm->setProbabilityDistribution(1, std::move(loadedDist));
}

int main() {
    std::cout << "Testing LogSIMDForwardBackwardCalculator\n";
    std::cout << "==========================================\n\n";
    
    // Create HMM
    std::unique_ptr<Hmm> hmm;
    createDishonestCasinoHMM(hmm);
    
    // Create observation sequence: 1, 6, 6, 1, 1 (0-indexed)
    ObservationSet observations(5);
    observations(0) = 0; observations(1) = 5; observations(2) = 5; 
    observations(3) = 0; observations(4) = 0; // Dice faces - 1
    
    std::cout << "HMM: 2 states (Fair/Loaded die), " << observations.size() << " observations\n";
    std::cout << "Observations: ";
    for (size_t i = 0; i < observations.size(); ++i) {
        std::cout << (observations(i) + 1) << " "; // Convert back to 1-6
    }
    std::cout << "\n\n";
    
    try {
        // Test LogForwardBackwardCalculator (reference)
        std::cout << "1. Testing LogForwardBackwardCalculator (reference):\n";
        LogForwardBackwardCalculator logCalc(hmm.get(), observations);
        const double logProbRef = logCalc.logProbability();
        const double probRef = logCalc.probability();
        
        std::cout << "   Log probability: " << logProbRef << "\n";
        std::cout << "   Raw probability: " << probRef << "\n";
        std::cout << "   Forward variables shape: " << logCalc.getForwardVariables().size1() 
                  << "x" << logCalc.getForwardVariables().size2() << "\n\n";
        
        // Test LogSIMDForwardBackwardCalculator
        std::cout << "2. Testing LogSIMDForwardBackwardCalculator (new):\n";
        LogSIMDForwardBackwardCalculator simdCalc(hmm.get(), observations);
        const double logProbSIMD = simdCalc.logProbability();
        const double probSIMD = simdCalc.probability();
        
        std::cout << "   Log probability: " << logProbSIMD << "\n";
        std::cout << "   Raw probability: " << probSIMD << "\n";
        std::cout << "   Forward variables shape: " << simdCalc.getForwardVariables().size1() 
                  << "x" << simdCalc.getForwardVariables().size2() << "\n";
        std::cout << "   Optimization info: " << simdCalc.getOptimizationInfo() << "\n\n";
        
        // Compare results
        std::cout << "3. Accuracy Comparison:\n";
        const double logDiff = std::abs(logProbRef - logProbSIMD);
        const double probDiff = std::abs(probRef - probSIMD);
        const double logRelError = logDiff / std::abs(logProbRef);
        
        std::cout << "   Log probability difference: " << logDiff << "\n";
        std::cout << "   Raw probability difference: " << probDiff << "\n";
        std::cout << "   Log relative error: " << logRelError << "\n";
        
        const double tolerance = 1e-12;
        if (logRelError < tolerance) {
            std::cout << "   ✅ NUMERICAL ACCURACY: EXCELLENT (< 1e-12 relative error)\n";
        } else if (logRelError < 1e-10) {
            std::cout << "   ✅ NUMERICAL ACCURACY: GOOD (< 1e-10 relative error)\n";
        } else if (logRelError < 1e-8) {
            std::cout << "   ⚠️  NUMERICAL ACCURACY: ACCEPTABLE (< 1e-8 relative error)\n";
        } else {
            std::cout << "   ❌ NUMERICAL ACCURACY: POOR (>= 1e-8 relative error)\n";
        }
        
        std::cout << "\n4. Forward Variables Comparison:\n";
        const Matrix refForward = logCalc.getForwardVariables();
        const Matrix simdForward = simdCalc.getForwardVariables();
        
        double maxForwardDiff = 0.0;
        for (size_t t = 0; t < refForward.size1(); ++t) {
            for (size_t i = 0; i < refForward.size2(); ++i) {
                const double diff = std::abs(refForward(t, i) - simdForward(t, i));
                maxForwardDiff = std::max(maxForwardDiff, diff);
            }
        }
        
        std::cout << "   Maximum forward variable difference: " << maxForwardDiff << "\n";
        if (maxForwardDiff < 1e-12) {
            std::cout << "   ✅ FORWARD VARIABLES: PERFECT MATCH\n";
        } else {
            std::cout << "   ⚠️  FORWARD VARIABLES: Some differences detected\n";
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ ERROR: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "\n==========================================\n";
    std::cout << "LogSIMD Calculator Test Complete!\n";
    
    return 0;
}
