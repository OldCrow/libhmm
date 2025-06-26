#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <iomanip>
#include "libhmm/libhmm.h"
#include "libhmm/calculators/forward_backward_traits.h"

using libhmm::Hmm;
using libhmm::ChiSquaredDistribution;
using libhmm::GaussianDistribution;
using libhmm::ExponentialDistribution;
using libhmm::ViterbiCalculator;
using libhmm::ForwardBackwardCalculator;
using libhmm::ViterbiTrainer;
using libhmm::ObservationSet;
using libhmm::ObservationLists;
using libhmm::Vector;
using libhmm::Matrix;

/**
 * Example: Statistical Process Control with Chi-squared Distribution HMM
 * 
 * This example demonstrates quality control monitoring using:
 * - Chi-squared distribution for test statistics and variance measures
 * - Gaussian distribution for measurement errors
 * - Exponential distribution for time-between-failures
 * 
 * Hidden States:
 * - State 0: "In Control" (process operating normally)
 * - State 1: "Warning" (process showing signs of deviation)
 * - State 2: "Out of Control" (process requires intervention)
 * 
 * Key applications of Chi-squared in quality control:
 * - Goodness-of-fit testing for process capability
 * - Variance monitoring and control charts
 * - Statistical hypothesis testing for quality metrics
 * - Multivariate process control
 */
int main() {
    std::cout << "=== Statistical Process Control with Chi-squared Distribution HMM ===\n\n";
    
    // Create 3-state HMM for process control states
    auto hmm = std::make_unique<Hmm>(3);
    
    // Set up transition matrix (process degradation model)
    Matrix trans(3, 3);
    // In Control transitions
    trans(0, 0) = 0.90; trans(0, 1) = 0.08; trans(0, 2) = 0.02;  // Mostly stable
    // Warning transitions
    trans(1, 0) = 0.20; trans(1, 1) = 0.60; trans(1, 2) = 0.20;  // Can improve or degrade
    // Out of Control transitions
    trans(2, 0) = 0.10; trans(2, 1) = 0.30; trans(2, 2) = 0.60;  // Tends to persist until fixed
    hmm->setTrans(trans);
    
    // Initial state probabilities (assume process starts in control)
    Vector pi(3);
    pi(0) = 0.85;  // 85% chance of starting in control
    pi(1) = 0.12;  // 12% chance of starting in warning
    pi(2) = 0.03;  // 3% chance of starting out of control
    hmm->setPi(pi);
    
    std::cout << "=== Chi-squared Test Statistics Model ===\n";
    
    // Chi-squared distributions for goodness-of-fit test statistics
    // In Control: χ²(df=2) - low test statistic values (good fit)
    // Warning: χ²(df=4) - moderate test statistic values (marginal fit)
    // Out of Control: χ²(df=8) - high test statistic values (poor fit)
    hmm->setProbabilityDistribution(0, std::make_unique<ChiSquaredDistribution>(2.0));  // In Control
    hmm->setProbabilityDistribution(1, std::make_unique<ChiSquaredDistribution>(4.0));  // Warning
    hmm->setProbabilityDistribution(2, std::make_unique<ChiSquaredDistribution>(8.0));  // Out of Control
    
    std::cout << "Process Control HMM Configuration:\n";
    std::cout << *hmm << std::endl;
    
    // Demonstrate test statistic probability analysis
    std::cout << "=== Process Capability Analysis (Chi-squared Test Statistics) ===\n";
    std::cout << std::fixed << std::setprecision(4);
    
    std::vector<double> testStats = {0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 15.0};
    
    std::cout << "Test Statistic | In Control | Warning | Out of Control\n";
    std::cout << "---------------+------------+---------+---------------\n";
    
    for (double stat : testStats) {
        std::cout << std::setw(13) << stat << " | ";
        for (int state = 0; state < 3; ++state) {
            double prob = hmm->getProbabilityDistribution(state)->getProbability(stat);
            std::cout << std::setw(10) << prob << " | ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    // Critical values analysis
    std::cout << "=== Critical Values and Control Limits ===\n";
    
    for (int state = 0; state < 3; ++state) {
        const auto* dist = dynamic_cast<const ChiSquaredDistribution*>(hmm->getProbabilityDistribution(state));
        std::vector<std::string> stateNames = {"In Control", "Warning", "Out of Control"};
        
        std::cout << stateNames[state] << " State (χ²(" << dist->getDegreesOfFreedom() << ")):\n";
        std::cout << "  Mean: " << dist->getMean() << "\n";
        std::cout << "  Std Dev: " << dist->getStandardDeviation() << "\n";
        std::cout << "  Variance: " << dist->getVariance() << "\n";
        std::cout << "  Mode: " << ((dist->getDegreesOfFreedom() >= 2) ? (dist->getDegreesOfFreedom() - 2) : 0) << "\n";
        std::cout << std::endl;
    }
    
    // Process monitoring example with test statistic sequence
    std::cout << "=== Process Monitoring Example ===\n";
    ObservationSet testStatSequence(15);
    
    // Simulate a process degradation sequence (increasing test statistics)
    double stats[] = {
        1.2, 0.8, 1.5, 2.1, 1.9,    // In control period
        3.2, 4.1, 3.8, 5.2, 4.7,    // Warning period (increasing variance)
        8.5, 12.3, 9.8, 15.2, 11.6  // Out of control period
    };
    
    for (size_t i = 0; i < testStatSequence.size(); ++i) {
        testStatSequence(i) = stats[i];
    }
    
    // Decode most likely process states
    ViterbiCalculator viterbi(hmm.get(), testStatSequence);
    auto states = viterbi.decode();
    
    std::cout << "Test Statistics:     ";
    for (size_t i = 0; i < testStatSequence.size(); ++i) {
        std::cout << std::setw(6) << std::setprecision(1) << testStatSequence(i) << " ";
    }
    std::cout << "\nProcess States:      ";
    for (size_t i = 0; i < states.size(); ++i) {
        std::cout << std::setw(6) << states(i) << " ";
    }
    std::cout << "\n                     (0=InControl, 1=Warning, 2=OutOfControl)\n\n";
    
    // Calculate overall process likelihood using AutoCalculator
    libhmm::forwardbackward::AutoCalculator fb(hmm.get(), testStatSequence);
    double likelihood = fb.probability();
    double logLikelihood = fb.getLogProbability();
    
    std::cout << "=== Process Quality Assessment ===\n";
    std::cout << "Sequence likelihood: " << std::scientific << std::setprecision(3) << likelihood << "\n";
    std::cout << "Log-likelihood: " << std::fixed << std::setprecision(2) << logLikelihood << "\n\n";
    
    std::cout << "=== Decoded Process State Sequence ===\n";
    std::cout << "Sample | Test Stat | Most Likely State\n";
    std::cout << "-------+-----------+------------------\n";
    
    std::vector<std::string> stateLabels = {"In Control", "Warning", "Out of Control"};
    for (size_t i = 0; i < testStatSequence.size(); ++i) {
        std::cout << std::setw(6) << i+1 << " | " 
                  << std::setw(9) << std::setprecision(1) << testStatSequence(i) << " | "
                  << std::setw(16) << stateLabels[states(i)] << "\n";
    }
    std::cout << std::endl;
    
    // ===== Multi-variate Process Control Example =====
    std::cout << "=== Multivariate Process Control Example ===\n";
    
    // Create HMM for monitoring multiple quality characteristics
    auto multivarHmm = std::make_unique<Hmm>(3);
    multivarHmm->setTrans(trans);
    multivarHmm->setPi(pi);
    
    // Hotelling's T² statistics follow scaled chi-squared distributions
    // Different degrees of freedom for different numbers of quality characteristics
    multivarHmm->setProbabilityDistribution(0, std::make_unique<ChiSquaredDistribution>(3.0));   // 3 characteristics
    multivarHmm->setProbabilityDistribution(1, std::make_unique<ChiSquaredDistribution>(5.0));   // 5 characteristics
    multivarHmm->setProbabilityDistribution(2, std::make_unique<ChiSquaredDistribution>(10.0));  // 10 characteristics
    
    std::cout << "Multivariate Control Chart Analysis:\n";
    std::cout << "- Monitoring multiple quality characteristics simultaneously\n";
    std::cout << "- Using Hotelling's T² statistic (follows chi-squared distribution)\n";
    std::cout << "- Higher degrees of freedom = more characteristics monitored\n\n";
    
    // Simulate multivariate control data
    ObservationSet multivarStats(12);
    double multivarData[] = {2.1, 3.5, 2.8, 4.2, 6.8, 8.5, 7.2, 12.3, 15.8, 18.2, 16.5, 14.1};
    
    for (size_t i = 0; i < multivarStats.size(); ++i) {
        multivarStats(i) = multivarData[i];
    }
    
    ViterbiCalculator multivarViterbi(multivarHmm.get(), multivarStats);
    auto multivarStates = multivarViterbi.decode();
    
    std::cout << "T² Statistics:       ";
    for (size_t i = 0; i < multivarStats.size(); ++i) {
        std::cout << std::setw(6) << std::setprecision(1) << multivarStats(i) << " ";
    }
    std::cout << "\nControl States:      ";
    for (size_t i = 0; i < multivarStates.size(); ++i) {
        std::cout << std::setw(6) << multivarStates(i) << " ";
    }
    std::cout << std::endl << std::endl;
    
    // ===== Variance Monitoring Example =====
    std::cout << "=== Process Variance Monitoring ===\n";
    
    // Create HMM specifically for variance monitoring
    auto varianceHmm = std::make_unique<Hmm>(3);
    varianceHmm->setTrans(trans);
    varianceHmm->setPi(pi);
    
    // Sample variance scaled by degrees of freedom follows chi-squared
    // Different expected variances for different control states
    varianceHmm->setProbabilityDistribution(0, std::make_unique<ChiSquaredDistribution>(6.0));   // Normal variance
    varianceHmm->setProbabilityDistribution(1, std::make_unique<ChiSquaredDistribution>(10.0));  // Increased variance
    varianceHmm->setProbabilityDistribution(2, std::make_unique<ChiSquaredDistribution>(16.0));  // High variance
    
    std::cout << "Variance Control Chart Configuration:\n";
    for (int state = 0; state < 3; ++state) {
        const auto* dist = dynamic_cast<const ChiSquaredDistribution*>(varianceHmm->getProbabilityDistribution(state));
        std::vector<std::string> labels = {"Normal", "Increased", "High"};
        std::cout << labels[state] << " Variance: χ²(" << dist->getDegreesOfFreedom() 
                  << "), Expected = " << dist->getMean() << std::endl;
    }
    std::cout << std::endl;
    
    // ===== Training with Quality Control Data =====
    std::cout << "=== Model Training with Historical Quality Data ===\n";
    
    ObservationLists qualityData;
    std::random_device rd;
    std::mt19937 gen(123);  // Fixed seed for reproducibility
    
    // Generate training data for different control states
    std::chi_squared_distribution<double> inControlDist(2.0);
    std::chi_squared_distribution<double> warningDist(4.0);
    std::chi_squared_distribution<double> outControlDist(8.0);
    
    // In control sequences
    for (int i = 0; i < 25; ++i) {
        ObservationSet seq(8);
        for (size_t j = 0; j < seq.size(); ++j) {
            seq(j) = inControlDist(gen);
        }
        qualityData.push_back(seq);
    }
    
    // Warning sequences
    for (int i = 0; i < 15; ++i) {
        ObservationSet seq(6);
        for (size_t j = 0; j < seq.size(); ++j) {
            seq(j) = warningDist(gen);
        }
        qualityData.push_back(seq);
    }
    
    // Out of control sequences
    for (int i = 0; i < 10; ++i) {
        ObservationSet seq(5);
        for (size_t j = 0; j < seq.size(); ++j) {
            seq(j) = outControlDist(gen);
        }
        qualityData.push_back(seq);
    }
    
    // Create training model
    auto trainHmm = std::make_unique<Hmm>(3);
    trainHmm->setProbabilityDistribution(0, std::make_unique<ChiSquaredDistribution>(3.0));  // Initial guess
    trainHmm->setProbabilityDistribution(1, std::make_unique<ChiSquaredDistribution>(5.0));  // Initial guess
    trainHmm->setProbabilityDistribution(2, std::make_unique<ChiSquaredDistribution>(7.0));  // Initial guess
    
    std::cout << "Training with " << qualityData.size() << " quality control sequences...\n";
    
    ViterbiTrainer trainer(trainHmm.get(), qualityData);
    trainer.train();
    
    std::cout << "\nTrained Model Parameters:\n";
    for (int state = 0; state < 3; ++state) {
        const auto* dist = dynamic_cast<const ChiSquaredDistribution*>(trainHmm->getProbabilityDistribution(state));
        std::vector<std::string> stateLabels = {"In Control", "Warning", "Out of Control"};
        std::cout << stateLabels[state] << ": df=" << std::setprecision(2) << dist->getDegreesOfFreedom()
                  << ", Mean=" << dist->getMean() << ", Std=" << dist->getStandardDeviation() << "\n";
    }
    std::cout << std::endl;
    
    // ===== Control Chart Limits Calculation =====
    std::cout << "=== Statistical Control Chart Limits ===\n";
    
    // Calculate control limits based on chi-squared critical values
    std::vector<double> alphaLevels = {0.95, 0.99, 0.999};  // Confidence levels
    
    for (int state = 0; state < 3; ++state) {
        const auto* dist = dynamic_cast<const ChiSquaredDistribution*>(hmm->getProbabilityDistribution(state));
        std::vector<std::string> stateNames = {"In Control", "Warning", "Out of Control"};
        
        std::cout << stateNames[state] << " State Control Limits:\n";
        std::cout << "  Center Line (Mean): " << std::setprecision(2) << dist->getMean() << "\n";
        
        // Approximate control limits (exact calculation would require inverse CDF)
        double ucl_95 = dist->getMean() + 2.0 * dist->getStandardDeviation();
        double ucl_99 = dist->getMean() + 3.0 * dist->getStandardDeviation();
        
        std::cout << "  UCL (95%): " << ucl_95 << "\n";
        std::cout << "  UCL (99%): " << ucl_99 << "\n";
        std::cout << std::endl;
    }
    
    // Real-world application scenarios
    std::cout << "=== Real-World Applications ===\n";
    std::cout << "• Manufacturing quality control (Six Sigma)\n";
    std::cout << "• Software testing and defect prediction\n";
    std::cout << "• Healthcare process monitoring\n";
    std::cout << "• Financial risk management and compliance\n";
    std::cout << "• Environmental monitoring and control\n";
    std::cout << "• Supply chain quality assurance\n";
    std::cout << "• Clinical trial monitoring\n";
    std::cout << "• Service level agreement monitoring\n\n";
    
    std::cout << "=== Key Advantages of Chi-squared Distribution in SPC ===\n";
    std::cout << "• Natural fit for test statistics and variance measures\n";
    std::cout << "• Well-established critical values and tables\n";
    std::cout << "• Appropriate for goodness-of-fit testing\n";
    std::cout << "• Handles multivariate control charts (Hotelling's T²)\n";
    std::cout << "• Robust theoretical foundation for hypothesis testing\n";
    std::cout << "• Integrates well with existing SPC methodologies\n";
    
    return 0;
}
