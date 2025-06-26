#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <iomanip>
#include "libhmm/libhmm.h"
#include "libhmm/calculators/forward_backward_traits.h"

using libhmm::Hmm;
using libhmm::StudentTDistribution;
using libhmm::GaussianDistribution;
using libhmm::ViterbiCalculator;
using libhmm::ForwardBackwardCalculator;
using libhmm::ViterbiTrainer;
using libhmm::ObservationSet;
using libhmm::ObservationLists;
using libhmm::Vector;
using libhmm::Matrix;

/**
 * Example: Robust Financial Risk Modeling with Student's t-Distribution HMM
 * 
 * This example demonstrates modeling financial returns with heavy tails using:
 * - Student's t-distribution for capturing extreme market events
 * - Gaussian distribution for comparison with traditional models
 * 
 * Hidden States:
 * - State 0: "Normal Market" (low volatility, near-normal returns)
 * - State 1: "Stressed Market" (high volatility, heavy-tailed returns)
 * - State 2: "Crisis Market" (extreme volatility, very heavy tails)
 * 
 * Key advantages of t-distribution in finance:
 * - Captures fat tails (extreme events more likely than normal distribution)
 * - Better fits for daily financial returns
 * - More robust to outliers
 */
int main() {
    std::cout << "=== Robust Financial Risk Modeling with Student's t-Distribution HMM ===\n\n";
    
    // Create 3-state HMM for market regimes
    auto hmm = std::make_unique<Hmm>(3);
    
    // Set up transition matrix (regime persistence with crisis feedback)
    Matrix trans(3, 3);
    // Normal market transitions
    trans(0, 0) = 0.92; trans(0, 1) = 0.07; trans(0, 2) = 0.01;  // Normal mostly persists
    // Stressed market transitions  
    trans(1, 0) = 0.15; trans(1, 1) = 0.75; trans(1, 2) = 0.10;  // Stress can escalate
    // Crisis market transitions
    trans(2, 0) = 0.05; trans(2, 1) = 0.25; trans(2, 2) = 0.70;  // Crisis tends to persist
    hmm->setTrans(trans);
    
    // Initial state probabilities (usually start in normal conditions)
    Vector pi(3);
    pi(0) = 0.80;  // 80% chance of starting in normal market
    pi(1) = 0.15;  // 15% chance of starting in stressed market
    pi(2) = 0.05;  // 5% chance of starting in crisis
    hmm->setPi(pi);
    
    std::cout << "=== Student's t-Distribution Risk Model ===\n";
    
    // Student's t-distributions with different tail heaviness
    // Normal market: t(df=30, μ=0.05, σ=0.8) - close to normal, slight positive drift
    // Stressed market: t(df=5, μ=-0.2, σ=1.5) - heavy tails, negative drift  
    // Crisis market: t(df=3, μ=-1.0, σ=2.5) - very heavy tails, strong negative drift
    hmm->setProbabilityDistribution(0, std::make_unique<StudentTDistribution>(30.0, 0.05, 0.8));  // Normal
    hmm->setProbabilityDistribution(1, std::make_unique<StudentTDistribution>(5.0, -0.2, 1.5));   // Stressed
    hmm->setProbabilityDistribution(2, std::make_unique<StudentTDistribution>(3.0, -1.0, 2.5));   // Crisis
    
    std::cout << "Market Risk HMM Configuration:\n";
    std::cout << *hmm << std::endl;
    
    // Demonstrate tail behavior comparison
    std::cout << "=== Tail Risk Analysis (Extreme Event Probabilities) ===\n";
    std::cout << std::fixed << std::setprecision(6);
    
    std::vector<double> extremeReturns = {-5.0, -3.0, -2.0, 2.0, 3.0, 5.0};
    
    for (double ret : extremeReturns) {
        std::cout << "Return = " << std::setw(5) << ret << "%: ";
        for (int state = 0; state < 3; ++state) {
            double prob = hmm->getProbabilityDistribution(state)->getProbability(ret);
            std::cout << "State" << state << "=" << std::scientific << std::setprecision(2) << prob << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::fixed << std::setprecision(2) << std::endl;
    
    // Create realistic financial returns sequence (daily returns in %)
    std::cout << "=== Market Regime Detection Example ===\n";
    ObservationSet returnsSequence(20);
    
    // Simulate a crisis event sequence
    double returns[] = {
        0.2, 0.1, -0.3, 0.4, 0.1,      // Normal market conditions
        -0.8, -1.2, 0.5, -1.5, -0.9,   // Market stress beginning
        -3.2, -2.8, -4.1, -2.5, 1.8,   // Crisis period (extreme moves)
        -1.1, 0.6, 0.3, 0.8, 0.4       // Recovery period
    };
    
    for (size_t i = 0; i < returnsSequence.size(); ++i) {
        returnsSequence(i) = returns[i];
    }
    
    // Decode most likely market regimes
    ViterbiCalculator viterbi(hmm.get(), returnsSequence);
    auto states = viterbi.decode();
    
    std::cout << "Daily Returns (%):   ";
    for (size_t i = 0; i < returnsSequence.size(); ++i) {
        std::cout << std::setw(6) << returnsSequence(i) << " ";
    }
    std::cout << "\nMarket Regimes:      ";
    for (size_t i = 0; i < states.size(); ++i) {
        std::cout << std::setw(6) << states(i) << " ";
    }
    std::cout << "\n                     (0=Normal, 1=Stressed, 2=Crisis)\n\n";
    
    // Calculate overall likelihood for the sequence using AutoCalculator
    libhmm::forwardbackward::AutoCalculator fb(hmm.get(), returnsSequence);
    double likelihood = fb.probability();
    double logLikelihood = fb.getLogProbability();
    
    std::cout << "=== Sequence Analysis ===\n";
    std::cout << "Total sequence likelihood: " << std::scientific << std::setprecision(3) << likelihood << "\n";
    std::cout << "Log-likelihood: " << std::fixed << std::setprecision(2) << logLikelihood << "\n\n";
    
    std::cout << "=== Decoded Regime Sequence ===\n";
    std::cout << "Day | Return | Most Likely Regime\n";
    std::cout << "----+--------+-------------------\n";
    
    std::vector<std::string> regimeLabels = {"Normal", "Stressed", "Crisis"};
    for (size_t i = 0; i < returnsSequence.size(); ++i) {
        std::cout << std::setw(3) << i+1 << " | " 
                  << std::setw(6) << returnsSequence(i) << " | "
                  << std::setw(17) << regimeLabels[states(i)] << "\n";
    }
    std::cout << std::endl;
    
    // Risk metrics calculation
    std::cout << "=== Risk Metrics Analysis ===\n";
    
    // Calculate Value at Risk (VaR) for each regime
    std::vector<std::string> regimeNames = {"Normal", "Stressed", "Crisis"};
    
    for (int state = 0; state < 3; ++state) {
        const auto* dist = dynamic_cast<const StudentTDistribution*>(hmm->getProbabilityDistribution(state));
        
        std::cout << regimeNames[state] << " Market Regime:\n";
        std::cout << "  Degrees of Freedom: " << dist->getDegreesOfFreedom() << "\n";
        std::cout << "  Location (μ): " << dist->getLocation() << "%\n";
        std::cout << "  Scale (σ): " << dist->getScale() << "%\n";
        
        if (dist->hasFiniteMean()) {
            std::cout << "  Expected Return: " << dist->getMean() << "%\n";
        }
        if (dist->hasFiniteVariance()) {
            std::cout << "  Volatility: " << dist->getStandardDeviation() << "%\n";
        } else {
            std::cout << "  Volatility: Infinite (heavy tails)\n";
        }
        std::cout << std::endl;
    }
    
    // Comparison with Gaussian model
    std::cout << "=== Comparison with Traditional Gaussian Model ===\n";
    
    auto gaussianHmm = std::make_unique<Hmm>(3);
    gaussianHmm->setTrans(trans);
    gaussianHmm->setPi(pi);
    
    // Equivalent Gaussian distributions (matched means and variances where possible)
    gaussianHmm->setProbabilityDistribution(0, std::make_unique<GaussianDistribution>(0.05, 0.8));
    gaussianHmm->setProbabilityDistribution(1, std::make_unique<GaussianDistribution>(-0.2, 1.5));
    gaussianHmm->setProbabilityDistribution(2, std::make_unique<GaussianDistribution>(-1.0, 2.5));
    
    std::cout << "Extreme Event Probability Comparison:\n";
    std::cout << "Return | t-Distribution Model | Gaussian Model | Ratio (t/Gaussian)\n";
    std::cout << "-------+---------------------+----------------+-------------------\n";
    
    for (double extremeRet : {-4.0, -3.0, -2.0, 3.0, 4.0}) {
        double tProb = 0.0, gaussProb = 0.0;
        
        // Weight by state probabilities (assume uniform for this comparison)
        for (int state = 0; state < 3; ++state) {
            tProb += hmm->getProbabilityDistribution(state)->getProbability(extremeRet) / 3.0;
            gaussProb += gaussianHmm->getProbabilityDistribution(state)->getProbability(extremeRet) / 3.0;
        }
        
        double ratio = (gaussProb > 0) ? tProb / gaussProb : std::numeric_limits<double>::infinity();
        
        std::cout << std::setw(6) << extremeRet << " | " 
                  << std::scientific << std::setprecision(3) << std::setw(19) << tProb << " | "
                  << std::setw(14) << gaussProb << " | "
                  << std::fixed << std::setprecision(1) << std::setw(17) << ratio << "\n";
    }
    std::cout << std::endl;
    
    // Training example with simulated crisis data
    std::cout << "=== Model Training with Historical Crisis Data ===\n";
    
    ObservationLists trainingData;
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    
    // Generate synthetic training sequences
    std::student_t_distribution<double> normalReturns(30.0);
    std::student_t_distribution<double> stressedReturns(5.0);
    std::student_t_distribution<double> crisisReturns(3.0);
    
    // Normal market sequences
    for (int i = 0; i < 20; ++i) {
        ObservationSet seq(10);
        for (size_t j = 0; j < seq.size(); ++j) {
            seq(j) = 0.05 + 0.8 * normalReturns(gen);
        }
        trainingData.push_back(seq);
    }
    
    // Stressed market sequences  
    for (int i = 0; i < 10; ++i) {
        ObservationSet seq(8);
        for (size_t j = 0; j < seq.size(); ++j) {
            seq(j) = -0.2 + 1.5 * stressedReturns(gen);
        }
        trainingData.push_back(seq);
    }
    
    // Crisis market sequences
    for (int i = 0; i < 5; ++i) {
        ObservationSet seq(6);
        for (size_t j = 0; j < seq.size(); ++j) {
            seq(j) = -1.0 + 2.5 * crisisReturns(gen);
        }
        trainingData.push_back(seq);
    }
    
    // Create model for training
    auto trainHmm = std::make_unique<Hmm>(3);
    trainHmm->setProbabilityDistribution(0, std::make_unique<StudentTDistribution>(10.0, 0.0, 1.0));  // Initial guess
    trainHmm->setProbabilityDistribution(1, std::make_unique<StudentTDistribution>(8.0, 0.0, 1.5));   // Initial guess  
    trainHmm->setProbabilityDistribution(2, std::make_unique<StudentTDistribution>(5.0, 0.0, 2.0));   // Initial guess
    
    std::cout << "Training model with " << trainingData.size() << " sequences...\n";
    
    ViterbiTrainer trainer(trainHmm.get(), trainingData);
    trainer.train();
    
    std::cout << "\nTrained Model Parameters:\n";
    for (int state = 0; state < 3; ++state) {
        const auto* dist = dynamic_cast<const StudentTDistribution*>(trainHmm->getProbabilityDistribution(state));
        std::cout << "State " << state << ": df=" << std::setprecision(2) << dist->getDegreesOfFreedom()
                  << ", μ=" << dist->getLocation() << ", σ=" << dist->getScale() << "\n";
    }
    std::cout << std::endl;
    
    std::cout << "=== Applications of Robust Financial Risk Models ===\n";
    std::cout << "• Crisis detection and early warning systems\n";
    std::cout << "• Stress testing and scenario analysis\n";
    std::cout << "• Risk-adjusted portfolio optimization\n"; 
    std::cout << "• Regulatory capital calculation (Basel III)\n";
    std::cout << "• High-frequency trading risk management\n";
    std::cout << "• Derivatives pricing under extreme market conditions\n";
    std::cout << "• Insurance and catastrophe modeling\n\n";
    
    std::cout << "=== Key Advantages of Student's t-Distribution ===\n";
    std::cout << "• Better captures extreme market events (fat tails)\n";
    std::cout << "• More robust parameter estimation\n";
    std::cout << "• Realistic modeling of financial return distributions\n";
    std::cout << "• Improved Value-at-Risk and Expected Shortfall estimates\n";
    std::cout << "• Better fits for high-frequency financial data\n";
    
    return 0;
}
