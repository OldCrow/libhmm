#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include "libhmm/libhmm.h"

using libhmm::Hmm;
using libhmm::NegativeBinomialDistribution;
using libhmm::ParetoDistribution;
using libhmm::ViterbiCalculator;
using libhmm::ViterbiTrainer;
using libhmm::ObservationSet;
using libhmm::ObservationLists;
using libhmm::Vector;
using libhmm::Matrix;

/**
 * Example: Economic and Social Science Modeling with Negative Binomial and Pareto HMM
 * 
 * This example demonstrates modeling economic phenomena using:
 * - Negative Binomial distribution for overdispersed count data (customer purchases, accidents)
 * - Pareto distribution for power-law phenomena (income, wealth, city sizes)
 * 
 * Hidden States for Customer Behavior:
 * - State 0: "Low Activity" (few purchases, occasional high-value items)
 * - State 1: "High Activity" (many purchases, frequent transactions)
 * 
 * Hidden States for Economic Regimes:
 * - State 0: "Normal Economy" (typical income distribution)
 * - State 1: "Crisis Economy" (more extreme inequality)
 */
int main() {
    std::cout << "=== Economics and Social Science HMM Example ===\n\n";
    
    // ===== Customer Purchase Behavior with Negative Binomial =====
    std::cout << "=== Customer Purchase Modeling with Negative Binomial Distribution ===\n";
    
    // Create 2-state HMM for customer behavior
    auto customerHmm = std::make_unique<Hmm>(2);
    
    // Set up transition matrix (customer behavior persistence)
    Matrix customerTrans(2, 2);
    customerTrans(0, 0) = 0.80; customerTrans(0, 1) = 0.20;  // Low activity can shift to high
    customerTrans(1, 0) = 0.30; customerTrans(1, 1) = 0.70;  // High activity tends to persist
    customerHmm->setTrans(customerTrans);
    
    // Initial state probabilities (most customers start with low activity)
    Vector customerPi(2);
    customerPi(0) = 0.75;  // 75% chance of starting in low activity
    customerPi(1) = 0.25;  // 25% chance of starting in high activity
    customerHmm->setPi(customerPi);
    
    // Negative Binomial distributions for monthly purchase counts
    // Low activity: NegBin(r=2, p=0.3) - mean≈4.7, variance≈15.6 (overdispersed)
    // High activity: NegBin(r=8, p=0.4) - mean≈12, variance≈30 (higher activity, still overdispersed)
    customerHmm->setProbabilityDistribution(0, std::make_unique<NegativeBinomialDistribution>(2, 0.3));   // Low activity
    customerHmm->setProbabilityDistribution(1, std::make_unique<NegativeBinomialDistribution>(8, 0.4));   // High activity
    
    std::cout << "Customer Behavior HMM Configuration:\n";
    std::cout << *customerHmm << std::endl;
    
    // Demonstrate purchase probability calculations
    std::cout << "Monthly Purchase Count Probability Examples:\n";
    std::cout << "P(3 purchases | Low activity)  = " 
              << customerHmm->getProbabilityDistribution(0)->getProbability(3) << std::endl;
    std::cout << "P(3 purchases | High activity) = " 
              << customerHmm->getProbabilityDistribution(1)->getProbability(3) << std::endl;
    std::cout << "P(10 purchases | Low activity)  = " 
              << customerHmm->getProbabilityDistribution(0)->getProbability(10) << std::endl;
    std::cout << "P(10 purchases | High activity) = " 
              << customerHmm->getProbabilityDistribution(1)->getProbability(10) << std::endl;
    std::cout << "P(20 purchases | Low activity)  = " 
              << customerHmm->getProbabilityDistribution(0)->getProbability(20) << std::endl;
    std::cout << "P(20 purchases | High activity) = " 
              << customerHmm->getProbabilityDistribution(1)->getProbability(20) << std::endl;
    std::cout << std::endl;
    
    // Create customer purchase observation sequence (monthly purchase counts)
    ObservationSet purchaseSequence(12);
    purchaseSequence(0) = 2;   // Low activity period
    purchaseSequence(1) = 4;   // Still low
    purchaseSequence(2) = 1;   // Very low
    purchaseSequence(3) = 8;   // Transitioning to high activity
    purchaseSequence(4) = 15;  // High activity
    purchaseSequence(5) = 12;  // Sustained high activity
    purchaseSequence(6) = 18;  // Very high activity
    purchaseSequence(7) = 11;  // Still high
    purchaseSequence(8) = 6;   // Declining activity
    purchaseSequence(9) = 3;   // Back to low
    purchaseSequence(10) = 5;  // Low activity
    purchaseSequence(11) = 2;  // Very low
    
    ViterbiCalculator viterbiPurchases(customerHmm.get(), purchaseSequence);
    auto purchaseStates = viterbiPurchases.decode();
    
    std::cout << "Monthly purchase counts:     ";
    for (size_t i = 0; i < purchaseSequence.size(); ++i) {
        std::cout << purchaseSequence(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Most likely activity states: ";
    for (size_t i = 0; i < purchaseStates.size(); ++i) {
        std::cout << purchaseStates(i) << " ";
    }
    std::cout << " (0=Low Activity, 1=High Activity)\n\n";
    
    // ===== Income Distribution with Pareto =====
    std::cout << "=== Income Distribution Modeling with Pareto Distribution ===\n";
    
    // Create 2-state HMM for economic regimes
    auto economicHmm = std::make_unique<Hmm>(2);
    
    // Set up transition matrix (economic regime persistence)
    Matrix economicTrans(2, 2);
    economicTrans(0, 0) = 0.95; economicTrans(0, 1) = 0.05;  // Normal economy is stable
    economicTrans(1, 0) = 0.15; economicTrans(1, 1) = 0.85;  // Crisis can persist
    economicHmm->setTrans(economicTrans);
    
    // Initial state probabilities (assume normal economic conditions)
    Vector economicPi(2);
    economicPi(0) = 0.85;  // 85% chance of starting in normal economy
    economicPi(1) = 0.15;  // 15% chance of starting in crisis
    economicHmm->setPi(economicPi);
    
    // Pareto distributions for income (in thousands of dollars)
    // Normal economy: Pareto(scale=30, shape=1.8) - moderate inequality
    // Crisis economy: Pareto(scale=25, shape=1.2) - extreme inequality (lower shape = more inequality)
    economicHmm->setProbabilityDistribution(0, std::make_unique<ParetoDistribution>(30.0, 1.8));  // Normal
    economicHmm->setProbabilityDistribution(1, std::make_unique<ParetoDistribution>(25.0, 1.2));  // Crisis
    
    std::cout << "Economic Regime HMM Configuration:\n";
    std::cout << *economicHmm << std::endl;
    
    // Demonstrate income probability calculations
    std::cout << "Income Distribution Probability Examples (in thousands $):\n";
    std::cout << "P(income=50k | Normal economy) = " 
              << economicHmm->getProbabilityDistribution(0)->getProbability(50) << std::endl;
    std::cout << "P(income=50k | Crisis economy) = " 
              << economicHmm->getProbabilityDistribution(1)->getProbability(50) << std::endl;
    std::cout << "P(income=100k | Normal economy) = " 
              << economicHmm->getProbabilityDistribution(0)->getProbability(100) << std::endl;
    std::cout << "P(income=100k | Crisis economy) = " 
              << economicHmm->getProbabilityDistribution(1)->getProbability(100) << std::endl;
    std::cout << "P(income=200k | Normal economy) = " 
              << economicHmm->getProbabilityDistribution(0)->getProbability(200) << std::endl;
    std::cout << "P(income=200k | Crisis economy) = " 
              << economicHmm->getProbabilityDistribution(1)->getProbability(200) << std::endl;
    std::cout << std::endl;
    
    // Create income observation sequence (sampled incomes in thousands)
    ObservationSet incomeSequence(10);
    incomeSequence(0) = 45;   // Normal economy period
    incomeSequence(1) = 52;   // Typical incomes
    incomeSequence(2) = 38;   // Normal range
    incomeSequence(3) = 85;   // Higher income (still normal)
    incomeSequence(4) = 150;  // Very high income (entering crisis - extreme inequality)
    incomeSequence(5) = 35;   // Low income (crisis effect)
    incomeSequence(6) = 280;  // Extremely high income (crisis inequality)
    incomeSequence(7) = 42;   // More typical income
    incomeSequence(8) = 65;   // Recovering
    incomeSequence(9) = 48;   // Back to normal
    
    ViterbiCalculator viterbiIncome(economicHmm.get(), incomeSequence);
    auto incomeStates = viterbiIncome.decode();
    
    std::cout << "Observed incomes (k$):       ";
    for (size_t i = 0; i < incomeSequence.size(); ++i) {
        std::cout << incomeSequence(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Most likely economic states: ";
    for (size_t i = 0; i < incomeStates.size(); ++i) {
        std::cout << incomeStates(i) << " ";
    }
    std::cout << " (0=Normal, 1=Crisis)\n\n";
    
    // Statistical insights about the distributions
    std::cout << "=== Distribution Properties and Economic Insights ===\n";
    
    // Negative Binomial insights
    std::cout << "Negative Binomial Distribution Insights:\n";
    std::cout << "• Models overdispersed count data (variance > mean)\n";
    std::cout << "• r parameter: number of 'successes' needed to stop\n";
    std::cout << "• p parameter: probability of success on each trial\n";
    std::cout << "• Mean = r(1-p)/p, Variance = r(1-p)/p²\n";
    std::cout << "• Useful for: customer transactions, accidents, defects\n";
    std::cout << std::endl;
    
    // Pareto insights
    std::cout << "Pareto Distribution Insights:\n";
    std::cout << "• Models power-law phenomena (80/20 rule)\n";
    std::cout << "• Lower shape parameter α → more inequality\n";
    std::cout << "• Scale parameter xₘ → minimum value\n";
    std::cout << "• PDF: f(x) = (α·xₘᵅ)/x^(α+1) for x ≥ xₘ\n";
    std::cout << "• Useful for: income, wealth, city sizes, word frequency\n";
    std::cout << std::endl;
    
    // Training example with synthetic data
    std::cout << "=== Training Example with Synthetic Economic Data ===\n";
    
    ObservationLists trainingData;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Generate low activity customer periods (fewer purchases)
    std::negative_binomial_distribution<int> lowActivity(2, 0.3);
    for (int i = 0; i < 20; ++i) {
        ObservationSet seq(4);
        for (size_t j = 0; j < seq.size(); ++j) {
            seq(j) = lowActivity(gen);
        }
        trainingData.push_back(seq);
    }
    
    // Generate high activity customer periods (more purchases)
    std::negative_binomial_distribution<int> highActivity(8, 0.4);
    for (int i = 0; i < 15; ++i) {
        ObservationSet seq(4);
        for (size_t j = 0; j < seq.size(); ++j) {
            seq(j) = highActivity(gen);
        }
        trainingData.push_back(seq);
    }
    
    // Create fresh HMM for training
    auto trainHmm = std::make_unique<Hmm>(2);
    trainHmm->setProbabilityDistribution(0, std::make_unique<NegativeBinomialDistribution>(3, 0.4));  // Initial guess
    trainHmm->setProbabilityDistribution(1, std::make_unique<NegativeBinomialDistribution>(6, 0.5));  // Initial guess
    
    std::cout << "Before training:\n";
    std::cout << "Low activity state:  " << trainHmm->getProbabilityDistribution(0)->toString() << std::endl;
    std::cout << "High activity state: " << trainHmm->getProbabilityDistribution(1)->toString() << std::endl;
    
    ViterbiTrainer trainer(trainHmm.get(), trainingData);
    trainer.train();
    
    std::cout << "\nAfter training:\n";
    std::cout << "Low activity state:  " << trainHmm->getProbabilityDistribution(0)->toString() << std::endl;
    std::cout << "High activity state: " << trainHmm->getProbabilityDistribution(1)->toString() << std::endl;
    std::cout << std::endl;
    
    std::cout << "Expected parameters after training:\n";
    std::cout << "• Low activity: NegBin(r≈2, p≈0.3) - fewer purchases, more variability\n";
    std::cout << "• High activity: NegBin(r≈8, p≈0.4) - more purchases, sustained engagement\n";
    std::cout << std::endl;
    
    std::cout << "=== Economic Theory Connections ===\n";
    std::cout << "Negative Binomial Applications:\n";
    std::cout << "• Customer lifetime value modeling\n";
    std::cout << "• Insurance claim frequency\n";
    std::cout << "• Patent citation counts\n";
    std::cout << "• Social media engagement\n";
    std::cout << "• Accident and safety modeling\n";
    std::cout << std::endl;
    
    std::cout << "Pareto Distribution Applications:\n";
    std::cout << "• Income and wealth inequality (Pareto principle)\n";
    std::cout << "• City population sizes (Zipf's law)\n";
    std::cout << "• Stock price movements\n";
    std::cout << "• Internet traffic and file sizes\n";
    std::cout << "• Scientific citation analysis\n";
    std::cout << "• Natural disaster magnitudes\n";
    std::cout << std::endl;
    
    std::cout << "=== Policy and Business Applications ===\n";
    std::cout << "Customer Behavior Analysis:\n";
    std::cout << "• Personalized marketing campaigns\n";
    std::cout << "• Churn prediction and retention\n";
    std::cout << "• Inventory management and demand forecasting\n";
    std::cout << "• Customer segmentation strategies\n";
    std::cout << std::endl;
    
    std::cout << "Economic Policy Analysis:\n";
    std::cout << "• Income inequality measurement and monitoring\n";
    std::cout << "• Tax policy impact assessment\n";
    std::cout << "• Social welfare program design\n";
    std::cout << "• Economic crisis detection and response\n";
    std::cout << "• Urban planning and resource allocation\n";
    
    return 0;
}
