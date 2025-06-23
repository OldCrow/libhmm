#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include "libhmm/libhmm.h"

using libhmm::Hmm;
using libhmm::BetaDistribution;
using libhmm::LogNormalDistribution;
using libhmm::ViterbiCalculator;
using libhmm::ViterbiTrainer;
using libhmm::ObservationSet;
using libhmm::ObservationLists;
using libhmm::Vector;
using libhmm::Matrix;

/**
 * Example: Financial Market Volatility Modeling with Beta and Log-Normal HMM
 * 
 * This example demonstrates modeling financial market states using:
 * - Beta distribution for volatility measures (bounded between 0 and 1)
 * - Log-Normal distribution for asset returns (always positive)
 * 
 * Hidden States:
 * - State 0: "Low Volatility" (stable market conditions)
 * - State 1: "High Volatility" (turbulent market conditions)
 */
int main() {
    std::cout << "=== Financial Market HMM Example ===\n\n";
    
    // Create 2-state HMM for market volatility
    auto hmm = std::make_unique<Hmm>(2);
    
    // Set up transition matrix (market regime persistence)
    Matrix trans(2, 2);
    trans(0, 0) = 0.85; trans(0, 1) = 0.15;  // Low vol tends to persist
    trans(1, 0) = 0.25; trans(1, 1) = 0.75;  // High vol is somewhat persistent
    hmm->setTrans(trans);
    
    // Initial state probabilities (start in low volatility)
    Vector pi(2);
    pi(0) = 0.7;  // 70% chance of starting in low volatility
    pi(1) = 0.3;  // 30% chance of starting in high volatility
    hmm->setPi(pi);
    
    std::cout << "=== Volatility Modeling with Beta Distribution ===\n";
    
    // Beta distributions for volatility (scaled to [0,1])
    // Low volatility: Beta(5, 20) - concentrated near 0
    // High volatility: Beta(2, 3) - more spread out, higher values
    hmm->setProbabilityDistribution(0, std::make_unique<BetaDistribution>(5.0, 20.0));  // Low vol
    hmm->setProbabilityDistribution(1, std::make_unique<BetaDistribution>(2.0, 3.0));   // High vol
    
    std::cout << "Volatility HMM Configuration:\n";
    std::cout << *hmm << std::endl;
    
    // Demonstrate volatility probability calculations
    std::cout << "Volatility Probability Examples (scaled to [0,1]):\n";
    std::cout << "P(vol=0.1 | Low regime)  = " 
              << hmm->getProbabilityDistribution(0)->getProbability(0.1) << std::endl;
    std::cout << "P(vol=0.1 | High regime) = " 
              << hmm->getProbabilityDistribution(1)->getProbability(0.1) << std::endl;
    std::cout << "P(vol=0.5 | Low regime)  = " 
              << hmm->getProbabilityDistribution(0)->getProbability(0.5) << std::endl;
    std::cout << "P(vol=0.5 | High regime) = " 
              << hmm->getProbabilityDistribution(1)->getProbability(0.5) << std::endl;
    std::cout << std::endl;
    
    // Create volatility observation sequence (VIX-like scaled to [0,1])
    ObservationSet volSequence(10);
    volSequence(0) = 0.12;  // Low volatility period
    volSequence(1) = 0.08;
    volSequence(2) = 0.15;
    volSequence(3) = 0.35;  // Transition to high volatility
    volSequence(4) = 0.62;
    volSequence(5) = 0.58;
    volSequence(6) = 0.71;  // High volatility period
    volSequence(7) = 0.45;  // Transitioning back
    volSequence(8) = 0.23;
    volSequence(9) = 0.18;
    
    ViterbiCalculator viterbiVol(hmm.get(), volSequence);
    auto volStates = viterbiVol.decode();
    
    std::cout << "Volatility sequence:     ";
    for (size_t i = 0; i < volSequence.size(); ++i) {
        std::cout << volSequence(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Most likely regimes:     ";
    for (size_t i = 0; i < volStates.size(); ++i) {
        std::cout << volStates(i) << " ";
    }
    std::cout << " (0=Low, 1=High)\n\n";
    
    // ===== Log-Normal Returns Model =====
    std::cout << "=== Returns Modeling with Log-Normal Distribution ===\n";
    
    // Create new HMM for returns (always positive due to log-normal)
    auto returnsHmm = std::make_unique<Hmm>(2);
    returnsHmm->setTrans(trans);  // Same transition structure
    returnsHmm->setPi(pi);        // Same initial probabilities
    
    // Log-Normal distributions for returns
    // Bull market: LogNormal(μ=0.1, σ=0.2) - higher expected returns
    // Bear market: LogNormal(μ=-0.05, σ=0.3) - lower returns, higher variance
    returnsHmm->setProbabilityDistribution(0, std::make_unique<LogNormalDistribution>(0.1, 0.2));   // Bull
    returnsHmm->setProbabilityDistribution(1, std::make_unique<LogNormalDistribution>(-0.05, 0.3)); // Bear
    
    std::cout << "Returns HMM Configuration:\n";
    std::cout << *returnsHmm << std::endl;
    
    // Demonstrate returns probability calculations
    std::cout << "Returns Probability Examples:\n";
    std::cout << "P(return=1.05 | Bull market) = " 
              << returnsHmm->getProbabilityDistribution(0)->getProbability(1.05) << std::endl;
    std::cout << "P(return=1.05 | Bear market) = " 
              << returnsHmm->getProbabilityDistribution(1)->getProbability(1.05) << std::endl;
    std::cout << "P(return=0.95 | Bull market) = " 
              << returnsHmm->getProbabilityDistribution(0)->getProbability(0.95) << std::endl;
    std::cout << "P(return=0.95 | Bear market) = " 
              << returnsHmm->getProbabilityDistribution(1)->getProbability(0.95) << std::endl;
    std::cout << std::endl;
    
    // Create returns observation sequence (daily returns as multipliers)
    ObservationSet returnsSequence(8);
    returnsSequence(0) = 1.02;  // +2% return (bull market)
    returnsSequence(1) = 1.01;  // +1% return
    returnsSequence(2) = 1.03;  // +3% return
    returnsSequence(3) = 0.98;  // -2% return (entering bear)
    returnsSequence(4) = 0.95;  // -5% return
    returnsSequence(5) = 0.92;  // -8% return (bear market)
    returnsSequence(6) = 0.97;  // -3% return
    returnsSequence(7) = 1.01;  // +1% return (recovery)
    
    ViterbiCalculator viterbiReturns(returnsHmm.get(), returnsSequence);
    auto returnStates = viterbiReturns.decode();
    
    std::cout << "Returns sequence:        ";
    for (size_t i = 0; i < returnsSequence.size(); ++i) {
        std::cout << returnsSequence(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Most likely regimes:     ";
    for (size_t i = 0; i < returnStates.size(); ++i) {
        std::cout << returnStates(i) << " ";
    }
    std::cout << " (0=Bull, 1=Bear)\n\n";
    
    // Training example with synthetic data
    std::cout << "=== Training Example with Synthetic Market Data ===\n";
    
    ObservationLists trainingData;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Generate bull market periods (low volatility, positive returns)
    for (int i = 0; i < 15; ++i) {
        ObservationSet seq(5);
        std::uniform_real_distribution<double> bullVol(0.05, 0.25);
        for (size_t j = 0; j < seq.size(); ++j) {
            seq(j) = bullVol(gen);
        }
        trainingData.push_back(seq);
    }
    
    // Generate bear market periods (high volatility)
    for (int i = 0; i < 10; ++i) {
        ObservationSet seq(5);
        std::uniform_real_distribution<double> bearVol(0.35, 0.75);
        for (size_t j = 0; j < seq.size(); ++j) {
            seq(j) = bearVol(gen);
        }
        trainingData.push_back(seq);
    }
    
    // Create fresh HMM for training
    auto trainHmm = std::make_unique<Hmm>(2);
    trainHmm->setProbabilityDistribution(0, std::make_unique<BetaDistribution>(2.0, 5.0));  // Initial guess
    trainHmm->setProbabilityDistribution(1, std::make_unique<BetaDistribution>(3.0, 2.0));  // Initial guess
    
    std::cout << "Before training:\n";
    std::cout << "Low vol state:  " << trainHmm->getProbabilityDistribution(0)->toString() << std::endl;
    std::cout << "High vol state: " << trainHmm->getProbabilityDistribution(1)->toString() << std::endl;
    
    ViterbiTrainer trainer(trainHmm.get(), trainingData);
    trainer.train();
    
    std::cout << "\nAfter training:\n";
    std::cout << "Low vol state:  " << trainHmm->getProbabilityDistribution(0)->toString() << std::endl;
    std::cout << "High vol state: " << trainHmm->getProbabilityDistribution(1)->toString() << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Applications of Financial HMMs ===\n";
    std::cout << "• Market regime detection (bull/bear identification)\n";
    std::cout << "• Volatility forecasting and risk management\n";
    std::cout << "• Portfolio optimization under regime uncertainty\n";
    std::cout << "• Options pricing with stochastic volatility\n";
    std::cout << "• Algorithmic trading strategy development\n";
    std::cout << "• Economic recession prediction\n";
    std::cout << "• Asset allocation and hedging strategies\n";
    
    return 0;
}
