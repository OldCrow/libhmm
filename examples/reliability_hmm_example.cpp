#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include "libhmm/libhmm.h"

using libhmm::Hmm;
using libhmm::WeibullDistribution;
using libhmm::ExponentialDistribution;
using libhmm::ViterbiCalculator;
using libhmm::ViterbiTrainer;
using libhmm::ObservationSet;
using libhmm::ObservationLists;
using libhmm::Vector;
using libhmm::Matrix;

/**
 * Example: Reliability Engineering with Weibull and Exponential HMM
 * 
 * This example demonstrates modeling system reliability using:
 * - Weibull distribution for component lifetimes (flexible hazard rates)
 * - Exponential distribution for memoryless failure times
 * 
 * Hidden States:
 * - State 0: "Normal Operation" (low failure rate)
 * - State 1: "Degraded State" (higher failure rate)
 * - State 2: "Critical State" (very high failure rate)
 */
int main() {
    std::cout << "=== Reliability Engineering HMM Example ===\n\n";
    
    // Create 3-state HMM for system reliability
    auto hmm = std::make_unique<Hmm>(3);
    
    // Set up transition matrix (degradation process)
    Matrix trans(3, 3);
    // Normal -> Normal: 0.9, Degraded: 0.08, Critical: 0.02
    trans(0, 0) = 0.90; trans(0, 1) = 0.08; trans(0, 2) = 0.02;
    // Degraded -> Normal: 0.1, Degraded: 0.7, Critical: 0.2
    trans(1, 0) = 0.10; trans(1, 1) = 0.70; trans(1, 2) = 0.20;
    // Critical -> Normal: 0.05, Degraded: 0.25, Critical: 0.7
    trans(2, 0) = 0.05; trans(2, 1) = 0.25; trans(2, 2) = 0.70;
    hmm->setTrans(trans);
    
    // Initial state probabilities (start in normal operation)
    Vector pi(3);
    pi(0) = 0.8;  // 80% chance of starting in normal state
    pi(1) = 0.15; // 15% chance of starting in degraded state
    pi(2) = 0.05; // 5% chance of starting in critical state
    hmm->setPi(pi);
    
    std::cout << "=== Component Lifetime Modeling with Weibull Distribution ===\n";
    
    // Weibull distributions for component lifetimes (time to failure)
    // Normal: Weibull(shape=2.5, scale=1000) - moderate hazard rate
    // Degraded: Weibull(shape=1.8, scale=500) - increasing hazard
    // Critical: Weibull(shape=1.2, scale=200) - high early failure rate
    hmm->setProbabilityDistribution(0, std::make_unique<WeibullDistribution>(2.5, 1000.0));  // Normal
    hmm->setProbabilityDistribution(1, std::make_unique<WeibullDistribution>(1.8, 500.0));   // Degraded  
    hmm->setProbabilityDistribution(2, std::make_unique<WeibullDistribution>(1.2, 200.0));   // Critical
    
    std::cout << "Reliability HMM Configuration:\n";
    std::cout << *hmm << std::endl;
    
    // Demonstrate lifetime probability calculations
    std::cout << "Component Lifetime Probability Examples (hours):\n";
    std::cout << "P(lifetime=100 | Normal)   = " 
              << hmm->getProbabilityDistribution(0)->getProbability(100) << std::endl;
    std::cout << "P(lifetime=100 | Degraded) = " 
              << hmm->getProbabilityDistribution(1)->getProbability(100) << std::endl;
    std::cout << "P(lifetime=100 | Critical) = " 
              << hmm->getProbabilityDistribution(2)->getProbability(100) << std::endl;
    std::cout << "P(lifetime=500 | Normal)   = " 
              << hmm->getProbabilityDistribution(0)->getProbability(500) << std::endl;
    std::cout << "P(lifetime=500 | Degraded) = " 
              << hmm->getProbabilityDistribution(1)->getProbability(500) << std::endl;
    std::cout << "P(lifetime=500 | Critical) = " 
              << hmm->getProbabilityDistribution(2)->getProbability(500) << std::endl;
    std::cout << std::endl;
    
    // Create lifetime observation sequence (component failure times in hours)
    ObservationSet lifetimeSequence(10);
    lifetimeSequence(0) = 850;   // Long lifetime (normal state)
    lifetimeSequence(1) = 920;   // Long lifetime
    lifetimeSequence(2) = 420;   // Moderate lifetime (degrading)
    lifetimeSequence(3) = 380;   // Shorter lifetime
    lifetimeSequence(4) = 150;   // Short lifetime (critical state)
    lifetimeSequence(5) = 95;    // Very short lifetime
    lifetimeSequence(6) = 180;   // Short lifetime
    lifetimeSequence(7) = 320;   // Recovering
    lifetimeSequence(8) = 650;   // Better lifetime
    lifetimeSequence(9) = 750;   // Good lifetime
    
    ViterbiCalculator viterbiLifetime(hmm.get(), lifetimeSequence);
    auto lifetimeStates = viterbiLifetime.decode();
    
    std::cout << "Component lifetimes:     ";
    for (size_t i = 0; i < lifetimeSequence.size(); ++i) {
        std::cout << lifetimeSequence(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Most likely states:      ";
    for (size_t i = 0; i < lifetimeStates.size(); ++i) {
        std::cout << lifetimeStates(i) << " ";
    }
    std::cout << " (0=Normal, 1=Degraded, 2=Critical)\n\n";
    
    // ===== Exponential Failure Rate Model =====
    std::cout << "=== Failure Rate Modeling with Exponential Distribution ===\n";
    
    // Create 2-state HMM for failure rates (memoryless process)
    auto failureHmm = std::make_unique<Hmm>(2);
    
    // Simpler transition matrix for failure rates
    Matrix failureTrans(2, 2);
    failureTrans(0, 0) = 0.85; failureTrans(0, 1) = 0.15;  // Normal -> High failure
    failureTrans(1, 0) = 0.40; failureTrans(1, 1) = 0.60;  // High -> Normal failure
    failureHmm->setTrans(failureTrans);
    
    Vector failurePi(2);
    failurePi(0) = 0.7;  // Start in low failure rate
    failurePi(1) = 0.3;  // Start in high failure rate
    failureHmm->setPi(failurePi);
    
    // Exponential distributions for time between failures
    // Low failure rate: λ = 0.001 (mean time = 1000 hours)
    // High failure rate: λ = 0.01 (mean time = 100 hours)
    failureHmm->setProbabilityDistribution(0, std::make_unique<ExponentialDistribution>(0.001));  // Low rate
    failureHmm->setProbabilityDistribution(1, std::make_unique<ExponentialDistribution>(0.01));   // High rate
    
    std::cout << "Failure Rate HMM Configuration:\n";
    std::cout << *failureHmm << std::endl;
    
    // Demonstrate failure rate probability calculations
    std::cout << "Time Between Failures Probability Examples (hours):\n";
    std::cout << "P(time=50 | Low rate)  = " 
              << failureHmm->getProbabilityDistribution(0)->getProbability(50) << std::endl;
    std::cout << "P(time=50 | High rate) = " 
              << failureHmm->getProbabilityDistribution(1)->getProbability(50) << std::endl;
    std::cout << "P(time=200 | Low rate)  = " 
              << failureHmm->getProbabilityDistribution(0)->getProbability(200) << std::endl;
    std::cout << "P(time=200 | High rate) = " 
              << failureHmm->getProbabilityDistribution(1)->getProbability(200) << std::endl;
    std::cout << std::endl;
    
    // Create failure time observation sequence (time between failures)
    ObservationSet failureSequence(8);
    failureSequence(0) = 180;  // Moderate time (low failure rate)
    failureSequence(1) = 250;  // Good time
    failureSequence(2) = 45;   // Short time (high failure rate)
    failureSequence(3) = 30;   // Very short time
    failureSequence(4) = 55;   // Short time
    failureSequence(5) = 120;  // Improving
    failureSequence(6) = 200;  // Much better
    failureSequence(7) = 300;  // Good reliability
    
    ViterbiCalculator viterbiFailure(failureHmm.get(), failureSequence);
    auto failureStates = viterbiFailure.decode();
    
    std::cout << "Time between failures:   ";
    for (size_t i = 0; i < failureSequence.size(); ++i) {
        std::cout << failureSequence(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Most likely states:      ";
    for (size_t i = 0; i < failureStates.size(); ++i) {
        std::cout << failureStates(i) << " ";
    }
    std::cout << " (0=Low Rate, 1=High Rate)\n\n";
    
    // Training example with synthetic reliability data
    std::cout << "=== Training Example with Synthetic Reliability Data ===\n";
    
    ObservationLists trainingData;
    
    // Generate normal operation periods (long lifetimes)
    for (int i = 0; i < 20; ++i) {
        ObservationSet seq(4);
        seq(0) = 800 + (i * 10) % 400;   // 800-1200 hours
        seq(1) = 750 + (i * 15) % 450;   // 750-1200 hours
        seq(2) = 900 + (i * 8) % 300;    // 900-1200 hours
        seq(3) = 850 + (i * 12) % 350;   // 850-1200 hours
        trainingData.push_back(seq);
    }
    
    // Generate degraded periods (moderate lifetimes)
    for (int i = 0; i < 15; ++i) {
        ObservationSet seq(4);
        seq(0) = 300 + (i * 20) % 300;   // 300-600 hours
        seq(1) = 250 + (i * 25) % 350;   // 250-600 hours
        seq(2) = 400 + (i * 15) % 200;   // 400-600 hours
        seq(3) = 350 + (i * 18) % 250;   // 350-600 hours
        trainingData.push_back(seq);
    }
    
    // Generate critical periods (short lifetimes)
    for (int i = 0; i < 10; ++i) {
        ObservationSet seq(4);
        seq(0) = 50 + (i * 15) % 150;    // 50-200 hours
        seq(1) = 75 + (i * 12) % 125;    // 75-200 hours
        seq(2) = 100 + (i * 10) % 100;   // 100-200 hours
        seq(3) = 60 + (i * 20) % 140;    // 60-200 hours
        trainingData.push_back(seq);
    }
    
    // Create fresh HMM for training
    auto trainHmm = std::make_unique<Hmm>(3);
    trainHmm->setProbabilityDistribution(0, std::make_unique<WeibullDistribution>(2.0, 800.0));   // Initial guess
    trainHmm->setProbabilityDistribution(1, std::make_unique<WeibullDistribution>(1.5, 400.0));   // Initial guess
    trainHmm->setProbabilityDistribution(2, std::make_unique<WeibullDistribution>(1.0, 150.0));   // Initial guess
    
    std::cout << "Before training:\n";
    std::cout << "Normal state:   " << trainHmm->getProbabilityDistribution(0)->toString() << std::endl;
    std::cout << "Degraded state: " << trainHmm->getProbabilityDistribution(1)->toString() << std::endl;
    std::cout << "Critical state: " << trainHmm->getProbabilityDistribution(2)->toString() << std::endl;
    
    ViterbiTrainer trainer(trainHmm.get(), trainingData);
    trainer.train();
    
    std::cout << "\nAfter training:\n";
    std::cout << "Normal state:   " << trainHmm->getProbabilityDistribution(0)->toString() << std::endl;
    std::cout << "Degraded state: " << trainHmm->getProbabilityDistribution(1)->toString() << std::endl;
    std::cout << "Critical state: " << trainHmm->getProbabilityDistribution(2)->toString() << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Reliability Analysis Insights ===\n";
    std::cout << "• Weibull shape parameter β interpretation:\n";
    std::cout << "  - β < 1: Decreasing hazard rate (early failures)\n";
    std::cout << "  - β = 1: Constant hazard rate (random failures)\n";
    std::cout << "  - β > 1: Increasing hazard rate (wear-out failures)\n";
    std::cout << "• Scale parameter η represents characteristic lifetime\n";
    std::cout << "• Exponential distribution assumes memoryless failures\n";
    std::cout << std::endl;
    
    std::cout << "=== Applications of Reliability HMMs ===\n";
    std::cout << "• Predictive maintenance scheduling\n";
    std::cout << "• System health monitoring and diagnostics\n";
    std::cout << "• Warranty analysis and cost estimation\n";
    std::cout << "• Quality control in manufacturing\n";
    std::cout << "• Infrastructure asset management\n";
    std::cout << "• Safety-critical system design\n";
    std::cout << "• Spare parts inventory optimization\n";
    
    return 0;
}
