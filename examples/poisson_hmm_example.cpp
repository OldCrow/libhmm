#include <iostream>
#include <vector>
#include <memory>
#include "libhmm/libhmm.h"

using libhmm::Hmm;
using libhmm::PoissonDistribution;
using libhmm::ViterbiCalculator;
using libhmm::ViterbiTrainer;
using libhmm::ObservationSet;
using libhmm::ObservationLists;
using libhmm::Vector;
using libhmm::Matrix;

/**
 * Example: Modeling Website Traffic with Poisson HMM
 * 
 * This example demonstrates using Poisson distributions in an HMM to model
 * website traffic patterns. We'll model two hidden states:
 * - State 0: "Normal Traffic" (λ = 10 requests/minute)
 * - State 1: "High Traffic" (λ = 50 requests/minute)
 */
int main() {
    std::cout << "=== Poisson HMM Example: Website Traffic Modeling ===" << std::endl;
    std::cout << std::endl;
    
    // Create 2-state HMM
    auto hmm = std::make_unique<Hmm>(2);
    
    // Set up transition matrix
    // State 0 (Normal) -> State 0: 0.9, State 1: 0.1
    // State 1 (High)   -> State 0: 0.3, State 1: 0.7
    Matrix trans(2, 2);
    trans(0, 0) = 0.9; trans(0, 1) = 0.1;
    trans(1, 0) = 0.3; trans(1, 1) = 0.7;
    hmm->setTrans(trans);
    
    // Set initial probabilities (start in normal traffic state)
    Vector pi(2);
    pi(0) = 0.8;  // 80% chance of starting in normal state
    pi(1) = 0.2;  // 20% chance of starting in high traffic state
    hmm->setPi(pi);
    
    // Set up Poisson emission distributions
    hmm->setProbabilityDistribution(0, std::make_unique<PoissonDistribution>(10.0));  // Normal: λ=10
    hmm->setProbabilityDistribution(1, std::make_unique<PoissonDistribution>(50.0));  // High: λ=50
    
    std::cout << "Initial HMM Configuration:" << std::endl;
    std::cout << *hmm << std::endl;
    
    // Demonstrate probability calculations
    std::cout << "Probability Examples:" << std::endl;
    std::cout << "P(15 requests | Normal state) = " 
              << hmm->getProbabilityDistribution(0)->getProbability(15) << std::endl;
    std::cout << "P(15 requests | High state) = " 
              << hmm->getProbabilityDistribution(1)->getProbability(15) << std::endl;
    std::cout << "P(45 requests | Normal state) = " 
              << hmm->getProbabilityDistribution(0)->getProbability(45) << std::endl;
    std::cout << "P(45 requests | High state) = " 
              << hmm->getProbabilityDistribution(1)->getProbability(45) << std::endl;
    std::cout << std::endl;
    
    // Create sample observation sequences (website requests per minute)
    std::cout << "=== Viterbi Decoding Example ===" << std::endl;
    
    // Sequence 1: Likely normal traffic
    ObservationSet normalSequence(5);
    normalSequence(0) = 8;   // 8 requests
    normalSequence(1) = 12;  // 12 requests  
    normalSequence(2) = 9;   // 9 requests
    normalSequence(3) = 11;  // 11 requests
    normalSequence(4) = 7;   // 7 requests
    
    ViterbiCalculator viterbi1(hmm.get(), normalSequence);
    auto states1 = viterbi1.decode();
    
    std::cout << "Normal traffic sequence: ";
    for (size_t i = 0; i < normalSequence.size(); ++i) {
        std::cout << normalSequence(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Most likely states:      ";
    for (size_t i = 0; i < states1.size(); ++i) {
        std::cout << states1(i) << " ";
    }
    std::cout << " (0=Normal, 1=High)" << std::endl;
    std::cout << std::endl;
    
    // Sequence 2: Mixed traffic (transition from normal to high)
    ObservationSet mixedSequence(6);
    mixedSequence(0) = 10;  // Normal
    mixedSequence(1) = 12;  // Normal
    mixedSequence(2) = 38;  // Transition to high
    mixedSequence(3) = 52;  // High
    mixedSequence(4) = 48;  // High
    mixedSequence(5) = 15;  // Back to normal
    
    ViterbiCalculator viterbi2(hmm.get(), mixedSequence);
    auto states2 = viterbi2.decode();
    
    std::cout << "Mixed traffic sequence:  ";
    for (size_t i = 0; i < mixedSequence.size(); ++i) {
        std::cout << mixedSequence(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Most likely states:      ";
    for (size_t i = 0; i < states2.size(); ++i) {
        std::cout << states2(i) << " ";
    }
    std::cout << " (0=Normal, 1=High)" << std::endl;
    std::cout << std::endl;
    
    // Training example with synthetic data
    std::cout << "=== Training Example ===" << std::endl;
    
    // Create training data: mix of normal and high traffic periods
    ObservationLists trainingData;
    
    // Add normal traffic sequences
    for (int i = 0; i < 10; ++i) {
        ObservationSet seq(4);
        seq(0) = 8 + (i % 5);   // 8-12 requests
        seq(1) = 9 + (i % 4);   // 9-12 requests
        seq(2) = 7 + (i % 6);   // 7-12 requests
        seq(3) = 10 + (i % 3);  // 10-12 requests
        trainingData.push_back(seq);
    }
    
    // Add high traffic sequences  
    for (int i = 0; i < 10; ++i) {
        ObservationSet seq(4);
        seq(0) = 45 + (i % 10);  // 45-54 requests
        seq(1) = 48 + (i % 8);   // 48-55 requests  
        seq(2) = 42 + (i % 12);  // 42-53 requests
        seq(3) = 50 + (i % 6);   // 50-55 requests
        trainingData.push_back(seq);
    }
    
    // Create fresh HMM for training
    auto trainHmm = std::make_unique<Hmm>(2);
    trainHmm->setProbabilityDistribution(0, std::make_unique<PoissonDistribution>(15.0));  // Initial guess
    trainHmm->setProbabilityDistribution(1, std::make_unique<PoissonDistribution>(40.0));  // Initial guess
    
    std::cout << "Before training:" << std::endl;
    std::cout << "State 0 (Normal): " << trainHmm->getProbabilityDistribution(0)->toString() << std::endl;
    std::cout << "State 1 (High):   " << trainHmm->getProbabilityDistribution(1)->toString() << std::endl;
    
    // Train with Viterbi (note: Baum-Welch currently only supports discrete distributions)
    ViterbiTrainer trainer(trainHmm.get(), trainingData);
    trainer.train();
    
    std::cout << std::endl << "After training:" << std::endl;
    std::cout << "State 0 (Normal): " << trainHmm->getProbabilityDistribution(0)->toString() << std::endl;
    std::cout << "State 1 (High):   " << trainHmm->getProbabilityDistribution(1)->toString() << std::endl;
    std::cout << std::endl;
    
    std::cout << "Training should have learned parameters close to:" << std::endl;
    std::cout << "- Normal state: λ ≈ 10 (actual training data mean)" << std::endl;
    std::cout << "- High state:   λ ≈ 50 (actual training data mean)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Applications of Poisson HMMs ===" << std::endl;
    std::cout << "• Web traffic analysis and anomaly detection" << std::endl;
    std::cout << "• Call center volume modeling" << std::endl;
    std::cout << "• Network packet arrival modeling" << std::endl;
    std::cout << "• Quality control in manufacturing (defect counts)" << std::endl;
    std::cout << "• Biological modeling (gene expression, mutations)" << std::endl;
    std::cout << "• Financial modeling (transaction counts, events)" << std::endl;
    std::cout << "• Epidemiology (disease outbreak detection)" << std::endl;
    
    return 0;
}
