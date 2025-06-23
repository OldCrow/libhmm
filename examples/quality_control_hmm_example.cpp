#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include "libhmm/libhmm.h"

using libhmm::Hmm;
using libhmm::BinomialDistribution;
using libhmm::UniformDistribution;
using libhmm::ViterbiCalculator;
using libhmm::ViterbiTrainer;
using libhmm::ObservationSet;
using libhmm::ObservationLists;
using libhmm::Vector;
using libhmm::Matrix;

/**
 * Example: Quality Control Process Monitoring with Binomial and Uniform HMM
 * 
 * This example demonstrates modeling quality control processes using:
 * - Binomial distribution for defect counts in batches
 * - Uniform distribution for measurement tolerances
 * 
 * Hidden States:
 * - State 0: "In Control" (low defect rate, tight tolerances)
 * - State 1: "Out of Control" (high defect rate, loose tolerances)
 */
int main() {
    std::cout << "=== Quality Control HMM Example ===\n\n";
    
    // Create 2-state HMM for quality control process
    auto hmm = std::make_unique<Hmm>(2);
    
    // Set up transition matrix (process control stability)
    Matrix trans(2, 2);
    trans(0, 0) = 0.92; trans(0, 1) = 0.08;  // In control tends to stay in control
    trans(1, 0) = 0.35; trans(1, 1) = 0.65;  // Out of control can persist
    hmm->setTrans(trans);
    
    // Initial state probabilities (assume process starts in control)
    Vector pi(2);
    pi(0) = 0.85;  // 85% chance of starting in control
    pi(1) = 0.15;  // 15% chance of starting out of control
    hmm->setPi(pi);
    
    std::cout << "=== Defect Count Modeling with Binomial Distribution ===\n";
    
    // Binomial distributions for defect counts (n=100 items per batch)
    // In control: Binomial(n=100, p=0.02) - 2% defect rate
    // Out of control: Binomial(n=100, p=0.12) - 12% defect rate
    hmm->setProbabilityDistribution(0, std::make_unique<BinomialDistribution>(100, 0.02));  // In control
    hmm->setProbabilityDistribution(1, std::make_unique<BinomialDistribution>(100, 0.12));  // Out of control
    
    std::cout << "Quality Control HMM Configuration:\n";
    std::cout << *hmm << std::endl;
    
    // Demonstrate defect probability calculations
    std::cout << "Defect Count Probability Examples (out of 100 items):\n";
    std::cout << "P(1 defect | In control)     = " 
              << hmm->getProbabilityDistribution(0)->getProbability(1) << std::endl;
    std::cout << "P(1 defect | Out of control) = " 
              << hmm->getProbabilityDistribution(1)->getProbability(1) << std::endl;
    std::cout << "P(5 defects | In control)    = " 
              << hmm->getProbabilityDistribution(0)->getProbability(5) << std::endl;
    std::cout << "P(5 defects | Out of control) = " 
              << hmm->getProbabilityDistribution(1)->getProbability(5) << std::endl;
    std::cout << "P(15 defects | In control)   = " 
              << hmm->getProbabilityDistribution(0)->getProbability(15) << std::endl;
    std::cout << "P(15 defects | Out of control) = " 
              << hmm->getProbabilityDistribution(1)->getProbability(15) << std::endl;
    std::cout << std::endl;
    
    // Create defect count observation sequence (defects per 100-item batch)
    ObservationSet defectSequence(12);
    defectSequence(0) = 1;   // Low defect count (in control)
    defectSequence(1) = 3;   // Still reasonable
    defectSequence(2) = 2;   // Good quality
    defectSequence(3) = 8;   // Increasing defects (process shifting)
    defectSequence(4) = 12;  // High defect count (out of control)
    defectSequence(5) = 15;  // Very high defects
    defectSequence(6) = 11;  // Still high
    defectSequence(7) = 14;  // Continuing issues
    defectSequence(8) = 6;   // Improving (corrective action?)
    defectSequence(9) = 4;   // Better
    defectSequence(10) = 2;  // Back in control
    defectSequence(11) = 1;  // Good quality
    
    ViterbiCalculator viterbiDefects(hmm.get(), defectSequence);
    auto defectStates = viterbiDefects.decode();
    
    std::cout << "Defect counts per batch:     ";
    for (size_t i = 0; i < defectSequence.size(); ++i) {
        std::cout << defectSequence(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Most likely process states:  ";
    for (size_t i = 0; i < defectStates.size(); ++i) {
        std::cout << defectStates(i) << " ";
    }
    std::cout << " (0=In Control, 1=Out of Control)\n\n";
    
    // ===== Measurement Tolerance Model =====
    std::cout << "=== Measurement Tolerance Modeling with Uniform Distribution ===\n";
    
    // Create new HMM for measurement deviations
    auto toleranceHmm = std::make_unique<Hmm>(2);
    toleranceHmm->setTrans(trans);  // Same transition structure
    toleranceHmm->setPi(pi);        // Same initial probabilities
    
    // Uniform distributions for measurement deviations from target
    // In control: Uniform(-0.5, 0.5) - tight tolerance, ±0.5 units
    // Out of control: Uniform(-2.0, 2.0) - loose tolerance, ±2.0 units
    toleranceHmm->setProbabilityDistribution(0, std::make_unique<UniformDistribution>(-0.5, 0.5));   // Tight
    toleranceHmm->setProbabilityDistribution(1, std::make_unique<UniformDistribution>(-2.0, 2.0));   // Loose
    
    std::cout << "Tolerance HMM Configuration:\n";
    std::cout << *toleranceHmm << std::endl;
    
    // Demonstrate tolerance probability calculations
    std::cout << "Measurement Deviation Probability Examples:\n";
    std::cout << "P(deviation=0.1 | In control)     = " 
              << toleranceHmm->getProbabilityDistribution(0)->getProbability(0.1) << std::endl;
    std::cout << "P(deviation=0.1 | Out of control) = " 
              << toleranceHmm->getProbabilityDistribution(1)->getProbability(0.1) << std::endl;
    std::cout << "P(deviation=1.5 | In control)     = " 
              << toleranceHmm->getProbabilityDistribution(0)->getProbability(1.5) << std::endl;
    std::cout << "P(deviation=1.5 | Out of control) = " 
              << toleranceHmm->getProbabilityDistribution(1)->getProbability(1.5) << std::endl;
    std::cout << std::endl;
    
    // Create measurement deviation sequence (deviations from target specification)
    ObservationSet toleranceSequence(10);
    toleranceSequence(0) = 0.15;   // Small deviation (in control)
    toleranceSequence(1) = -0.22;  // Small negative deviation
    toleranceSequence(2) = 0.08;   // Very small deviation
    toleranceSequence(3) = 0.95;   // Larger deviation (going out of control)
    toleranceSequence(4) = 1.75;   // Large deviation (out of control)
    toleranceSequence(5) = -1.42;  // Large negative deviation
    toleranceSequence(6) = 1.88;   // Very large deviation
    toleranceSequence(7) = 0.85;   // Moderate deviation (improving)
    toleranceSequence(8) = 0.35;   // Getting better
    toleranceSequence(9) = -0.18;  // Back in control
    
    ViterbiCalculator viterbiTolerance(toleranceHmm.get(), toleranceSequence);
    auto toleranceStates = viterbiTolerance.decode();
    
    std::cout << "Measurement deviations:      ";
    for (size_t i = 0; i < toleranceSequence.size(); ++i) {
        std::cout << toleranceSequence(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Most likely process states:  ";
    for (size_t i = 0; i < toleranceStates.size(); ++i) {
        std::cout << toleranceStates(i) << " ";
    }
    std::cout << " (0=In Control, 1=Out of Control)\n\n";
    
    // Statistical Process Control Insights
    std::cout << "=== Statistical Process Control Insights ===\n";
    std::cout << "Control Charts Analysis:\n";
    
    // Calculate control limits for defect counts (assuming in-control state)
    // Using known parameters (n=100, p=0.02) for control limit calculation
    double mean = 100 * 0.02;  // n * p = 100 * 0.02 = 2
    double variance = 100 * 0.02 * (1 - 0.02);  // n * p * (1-p)
    double stddev = std::sqrt(variance);
    
    std::cout << "Defect Count Control Limits (Binomial n=100, p=0.02):\n";
    std::cout << "  Mean (CL):  " << mean << " defects\n";
    std::cout << "  UCL (+3σ):  " << mean + 3*stddev << " defects\n";
    std::cout << "  LCL (-3σ):  " << std::max(0.0, mean - 3*stddev) << " defects\n";
    std::cout << std::endl;
    
    std::cout << "Measurement Control Limits (Uniform [-0.5, 0.5]):\n";
    std::cout << "  Range:      [-0.5, 0.5] units\n";
    std::cout << "  Outside range indicates out-of-control state\n";
    std::cout << std::endl;
    
    // Training example with synthetic quality data
    std::cout << "=== Training Example with Synthetic Quality Data ===\n";
    
    ObservationLists trainingData;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Generate in-control periods (low defect counts)
    std::binomial_distribution<int> inControlDefects(100, 0.02);
    for (int i = 0; i < 25; ++i) {
        ObservationSet seq(4);
        for (size_t j = 0; j < seq.size(); ++j) {
            seq(j) = inControlDefects(gen);
        }
        trainingData.push_back(seq);
    }
    
    // Generate out-of-control periods (high defect counts)
    std::binomial_distribution<int> outOfControlDefects(100, 0.12);
    for (int i = 0; i < 15; ++i) {
        ObservationSet seq(4);
        for (size_t j = 0; j < seq.size(); ++j) {
            seq(j) = outOfControlDefects(gen);
        }
        trainingData.push_back(seq);
    }
    
    // Create fresh HMM for training
    auto trainHmm = std::make_unique<Hmm>(2);
    trainHmm->setProbabilityDistribution(0, std::make_unique<BinomialDistribution>(100, 0.05));  // Initial guess
    trainHmm->setProbabilityDistribution(1, std::make_unique<BinomialDistribution>(100, 0.15));  // Initial guess
    
    std::cout << "Before training:\n";
    std::cout << "In control state:     " << trainHmm->getProbabilityDistribution(0)->toString() << std::endl;
    std::cout << "Out of control state: " << trainHmm->getProbabilityDistribution(1)->toString() << std::endl;
    
    ViterbiTrainer trainer(trainHmm.get(), trainingData);
    trainer.train();
    
    std::cout << "\nAfter training:\n";
    std::cout << "In control state:     " << trainHmm->getProbabilityDistribution(0)->toString() << std::endl;
    std::cout << "Out of control state: " << trainHmm->getProbabilityDistribution(1)->toString() << std::endl;
    std::cout << std::endl;
    
    std::cout << "Expected parameters after training:\n";
    std::cout << "• In control: Binomial(100, ~0.02) - close to 2% defect rate\n";
    std::cout << "• Out of control: Binomial(100, ~0.12) - close to 12% defect rate\n";
    std::cout << std::endl;
    
    std::cout << "=== Quality Control Decision Rules ===\n";
    std::cout << "Process should be considered OUT OF CONTROL if:\n";
    std::cout << "• Single point exceeds UCL (>7 defects for our example)\n";
    std::cout << "• 2 out of 3 consecutive points exceed 2σ limit\n";
    std::cout << "• 4 out of 5 consecutive points exceed 1σ limit\n";
    std::cout << "• 8 consecutive points on same side of centerline\n";
    std::cout << "• Measurement deviations exceed tolerance range\n";
    std::cout << std::endl;
    
    std::cout << "=== Applications of Quality Control HMMs ===\n";
    std::cout << "• Manufacturing process monitoring\n";
    std::cout << "• Automated quality inspection systems\n";
    std::cout << "• Supply chain quality assessment\n";
    std::cout << "• Medical device manufacturing\n";
    std::cout << "• Food safety and quality control\n";
    std::cout << "• Pharmaceutical batch testing\n";
    std::cout << "• Software defect prediction\n";
    std::cout << "• Service quality monitoring\n";
    
    return 0;
}
