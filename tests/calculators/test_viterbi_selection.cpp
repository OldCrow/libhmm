#include <iostream>
#include "libhmm/libhmm.h"
#include "libhmm/calculators/viterbi_traits.h"

int main() {
    std::cout << "Testing Calculator Selection System\n\n";
    
    // Create a simple HMM for testing
    auto hmm = std::make_unique<libhmm::Hmm>(5);
    
    // Initialize with random probabilities
    libhmm::Vector pi(5);
    libhmm::Matrix trans(5, 5);
    
    for (size_t i = 0; i < 5; ++i) {
        pi(i) = 0.2;  // Equal probabilities
        for (size_t j = 0; j < 5; ++j) {
            trans(i, j) = 0.2;  // Equal probabilities
        }
    }
    
    hmm->setPi(pi);
    hmm->setTrans(trans);
    
    // Set discrete emission distributions
    for (size_t i = 0; i < 5; ++i) {
        auto dist = std::make_unique<libhmm::DiscreteDistribution>(4);
        for (size_t j = 0; j < 4; ++j) {
            dist->setProbability(j, 0.25);  // Equal probabilities
        }
        hmm->setProbabilityDistribution(i, std::move(dist));
    }
    
    // Test different sequence lengths to see which calculator gets selected
    std::vector<size_t> sequence_lengths = {10, 50, 100, 200, 500, 1000};
    
    for (size_t seq_len : sequence_lengths) {
        // Create observation sequence
        libhmm::ObservationSet obs(seq_len);
        for (size_t i = 0; i < seq_len; ++i) {
            obs(i) = i % 4;  // Simple pattern
        }
        
        // Get problem characteristics
        libhmm::viterbi::ProblemCharacteristics characteristics(hmm.get(), obs);
        
        // Get performance comparison
        std::string comparison = libhmm::viterbi::CalculatorSelector::getPerformanceComparison(characteristics);
        
        std::cout << "=== Sequence Length: " << seq_len << " ===\n";
        std::cout << comparison << "\n";
        
        // Debug: print all calculator types and their enum values
        std::cout << "Debug - Calculator enum values:\n";
        std::cout << "STANDARD: " << static_cast<int>(libhmm::viterbi::CalculatorType::STANDARD) << "\n";
        std::cout << "SCALED_SIMD: " << static_cast<int>(libhmm::viterbi::CalculatorType::SCALED_SIMD) << "\n";
        std::cout << "LOG_SIMD: " << static_cast<int>(libhmm::viterbi::CalculatorType::LOG_SIMD) << "\n";
        std::cout << "AUTO: " << static_cast<int>(libhmm::viterbi::CalculatorType::AUTO) << "\n";
        std::cout << "\n";
        
        // Debug: Test SCALED_SIMD specifically
        auto scaledSimdTraits = libhmm::viterbi::CalculatorSelector::getTraits(libhmm::viterbi::CalculatorType::SCALED_SIMD);
        auto scaledSimdPerf = libhmm::viterbi::CalculatorSelector::predictPerformance(libhmm::viterbi::CalculatorType::SCALED_SIMD, characteristics);
        std::cout << "SCALED_SIMD traits name: '" << scaledSimdTraits.name << "'\n";
        std::cout << "SCALED_SIMD predicted performance: " << scaledSimdPerf << "x\n";
        
        // Test AutoCalculator selection
        libhmm::viterbi::AutoCalculator calc(hmm.get(), obs);
        std::cout << "Selected Type: " << static_cast<int>(calc.getSelectedType()) << "\n";
        std::cout << "Rationale: " << calc.getSelectionRationale() << "\n\n";
    }
    
    return 0;
}
