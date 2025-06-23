#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <iomanip>
#include "libhmm/libhmm.h"

using libhmm::Hmm;
using libhmm::PoissonDistribution;
using libhmm::ExponentialDistribution;
using libhmm::GammaDistribution;
using libhmm::ViterbiCalculator;
using libhmm::ViterbiTrainer;
using libhmm::ObservationSet;
using libhmm::ObservationLists;
using libhmm::Vector;
using libhmm::Matrix;

/**
 * Example: Queuing Theory and Service Systems with HMM
 * 
 * This example demonstrates modeling service systems using HMMs to represent:
 * - Customer arrival patterns (Poisson arrivals)
 * - Service time distributions (Exponential, Gamma)
 * - System state transitions (load levels, server availability)
 * 
 * Service System States:
 * - State 0: "Low Load" (few customers, fast service)
 * - State 1: "Medium Load" (moderate queue, normal service)
 * - State 2: "High Load" (long queue, slow service)
 * 
 * Models Demonstrated:
 * 1. M/M/1 Queue (Poisson arrivals, Exponential service)
 * 2. M/G/1 Queue (Poisson arrivals, Gamma service times)
 * 3. Call Center Dynamics with varying load states
 */
int main() {
    std::cout << "=== Queuing Theory HMM Example ===\n\n";
    
    // ===== Customer Arrival Modeling =====
    std::cout << "=== Customer Arrival Modeling (Poisson Process) ===\n";
    
    // Create 3-state HMM for service system load
    auto arrivalHmm = std::make_unique<Hmm>(3);
    
    // Set up transition matrix (load state transitions)
    // Models realistic service system dynamics
    Matrix arrivalTrans(3, 3);
    // Low Load: tends to stay low, can go to medium
    arrivalTrans(0, 0) = 0.75; arrivalTrans(0, 1) = 0.20; arrivalTrans(0, 2) = 0.05;
    // Medium Load: can go either way
    arrivalTrans(1, 0) = 0.15; arrivalTrans(1, 1) = 0.65; arrivalTrans(1, 2) = 0.20;
    // High Load: tends to persist, difficult to recover
    arrivalTrans(2, 0) = 0.05; arrivalTrans(2, 1) = 0.25; arrivalTrans(2, 2) = 0.70;
    arrivalHmm->setTrans(arrivalTrans);
    
    // Initial state probabilities (assume system starts in low load)
    Vector arrivalPi(3);
    arrivalPi(0) = 0.6;  // 60% chance of starting in low load
    arrivalPi(1) = 0.3;  // 30% chance of starting in medium load
    arrivalPi(2) = 0.1;  // 10% chance of starting in high load
    arrivalHmm->setPi(arrivalPi);
    
    // Poisson distributions for customer arrivals per time period
    // Low load: λ = 3 customers/hour
    // Medium load: λ = 8 customers/hour  
    // High load: λ = 15 customers/hour
    arrivalHmm->setProbabilityDistribution(0, std::make_unique<PoissonDistribution>(3.0));   // Low load
    arrivalHmm->setProbabilityDistribution(1, std::make_unique<PoissonDistribution>(8.0));   // Medium load
    arrivalHmm->setProbabilityDistribution(2, std::make_unique<PoissonDistribution>(15.0));  // High load
    
    std::cout << "Customer Arrival HMM Configuration:\n";
    std::cout << *arrivalHmm << std::endl;
    
    // Demonstrate arrival probability calculations
    std::cout << "Customer Arrival Probability Examples (customers per hour):\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "P(5 arrivals | Low load)    = " 
              << arrivalHmm->getProbabilityDistribution(0)->getProbability(5) << std::endl;
    std::cout << "P(5 arrivals | Medium load) = " 
              << arrivalHmm->getProbabilityDistribution(1)->getProbability(5) << std::endl;
    std::cout << "P(5 arrivals | High load)   = " 
              << arrivalHmm->getProbabilityDistribution(2)->getProbability(5) << std::endl;
    std::cout << "P(12 arrivals | Low load)   = " 
              << arrivalHmm->getProbabilityDistribution(0)->getProbability(12) << std::endl;
    std::cout << "P(12 arrivals | Medium load) = " 
              << arrivalHmm->getProbabilityDistribution(1)->getProbability(12) << std::endl;
    std::cout << "P(12 arrivals | High load)  = " 
              << arrivalHmm->getProbabilityDistribution(2)->getProbability(12) << std::endl;
    std::cout << std::endl;
    
    // Create arrival observation sequence (hourly customer counts)
    ObservationSet arrivalSequence(24);  // 24-hour period
    // Simulate a day: low morning, buildup to peak, evening decline
    int arrivals[] = {2, 1, 3, 4, 6, 8, 12, 15, 18, 16, 14, 12, 
                     10, 13, 15, 17, 14, 11, 8, 6, 4, 3, 2, 1};
    for (size_t i = 0; i < arrivalSequence.size(); ++i) {
        arrivalSequence(i) = arrivals[i];
    }
    
    ViterbiCalculator viterbiArrivals(arrivalHmm.get(), arrivalSequence);
    auto arrivalStates = viterbiArrivals.decode();
    
    std::cout << "Hourly customer arrivals:    ";
    for (size_t i = 0; i < arrivalSequence.size(); ++i) {
        std::cout << std::setw(2) << arrivalSequence(i) << " ";
        if ((i + 1) % 12 == 0) std::cout << std::endl << "                             ";
    }
    std::cout << std::endl;
    std::cout << "Most likely load states:     ";
    for (size_t i = 0; i < arrivalStates.size(); ++i) {
        std::cout << std::setw(2) << arrivalStates(i) << " ";
        if ((i + 1) % 12 == 0) std::cout << std::endl << "                             ";
    }
    std::cout << " (0=Low, 1=Medium, 2=High)\n\n";
    
    // ===== Service Time Modeling =====
    std::cout << "=== Service Time Modeling (M/M/1 and M/G/1 Queues) ===\n";
    
    // Create 2-state HMM for service efficiency
    auto serviceHmm = std::make_unique<Hmm>(2);
    
    // Service state transitions (efficiency levels)
    Matrix serviceTrans(2, 2);
    serviceTrans(0, 0) = 0.90; serviceTrans(0, 1) = 0.10;  // Efficient service tends to persist
    serviceTrans(1, 0) = 0.30; serviceTrans(1, 1) = 0.70;  // Slow service can persist due to queue buildup
    serviceHmm->setTrans(serviceTrans);
    
    Vector servicePi(2);
    servicePi(0) = 0.8;  // 80% chance of starting efficient
    servicePi(1) = 0.2;  // 20% chance of starting slow
    serviceHmm->setPi(servicePi);
    
    std::cout << "M/M/1 Queue Model (Exponential Service Times):\n";
    
    // Exponential distributions for service times (M/M/1 model)
    // Efficient: μ = 0.2 (mean service time = 5 minutes)
    // Slow: μ = 0.1 (mean service time = 10 minutes)
    serviceHmm->setProbabilityDistribution(0, std::make_unique<ExponentialDistribution>(0.2));  // Efficient
    serviceHmm->setProbabilityDistribution(1, std::make_unique<ExponentialDistribution>(0.1));  // Slow
    
    std::cout << *serviceHmm << std::endl;
    
    // Demonstrate service time probabilities
    std::cout << "Service Time Probability Examples (minutes):\n";
    std::cout << "P(service=3min | Efficient) = " 
              << serviceHmm->getProbabilityDistribution(0)->getProbability(3) << std::endl;
    std::cout << "P(service=3min | Slow)      = " 
              << serviceHmm->getProbabilityDistribution(1)->getProbability(3) << std::endl;
    std::cout << "P(service=8min | Efficient) = " 
              << serviceHmm->getProbabilityDistribution(0)->getProbability(8) << std::endl;
    std::cout << "P(service=8min | Slow)      = " 
              << serviceHmm->getProbabilityDistribution(1)->getProbability(8) << std::endl;
    std::cout << std::endl;
    
    // Create service time observation sequence
    ObservationSet serviceSequence(15);
    double serviceTimes[] = {4.2, 3.8, 5.1, 2.9, 4.5, 8.7, 12.3, 9.8, 11.2, 6.5, 4.8, 3.7, 5.2, 4.1, 3.9};
    for (size_t i = 0; i < serviceSequence.size(); ++i) {
        serviceSequence(i) = serviceTimes[i];
    }
    
    ViterbiCalculator viterbiService(serviceHmm.get(), serviceSequence);
    auto serviceStates = viterbiService.decode();
    
    std::cout << "Service times (minutes):     ";
    for (size_t i = 0; i < serviceSequence.size(); ++i) {
        std::cout << std::setprecision(1) << std::setw(4) << serviceSequence(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Most likely efficiency:      ";
    for (size_t i = 0; i < serviceStates.size(); ++i) {
        std::cout << std::setw(4) << serviceStates(i) << " ";
    }
    std::cout << " (0=Efficient, 1=Slow)\n\n";
    
    // ===== M/G/1 Queue with Gamma Service Times =====
    std::cout << "M/G/1 Queue Model (Gamma Service Times):\n";
    
    auto mgOneHmm = std::make_unique<Hmm>(2);
    mgOneHmm->setTrans(serviceTrans);
    mgOneHmm->setPi(servicePi);
    
    // Gamma distributions for more realistic service times
    // Efficient: Gamma(shape=4, scale=1.25) - mean=5, more consistent
    // Slow: Gamma(shape=2, scale=5) - mean=10, more variable
    mgOneHmm->setProbabilityDistribution(0, std::make_unique<GammaDistribution>(4.0, 1.25));  // Efficient, consistent
    mgOneHmm->setProbabilityDistribution(1, std::make_unique<GammaDistribution>(2.0, 5.0));   // Slow, variable
    
    std::cout << *mgOneHmm << std::endl;
    
    ViterbiCalculator viterbiMG1(mgOneHmm.get(), serviceSequence);
    auto mg1States = viterbiMG1.decode();
    
    std::cout << "M/G/1 Analysis (same data):  ";
    for (size_t i = 0; i < mg1States.size(); ++i) {
        std::cout << std::setw(4) << mg1States(i) << " ";
    }
    std::cout << " (0=Efficient, 1=Slow)\n\n";
    
    // ===== Queuing Theory Insights =====
    std::cout << "=== Queuing Theory Performance Metrics ===\n";
    
    // Calculate theoretical performance metrics
    double lambda_low = 3.0, lambda_med = 8.0, lambda_high = 15.0;
    double mu_efficient = 0.2, mu_slow = 0.1;
    double service_rate_efficient = 1.0 / mu_efficient;  // customers per minute
    double service_rate_slow = 1.0 / mu_slow;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "M/M/1 Queue Analysis:\n";
    std::cout << "Arrival Rates: Low=" << lambda_low << ", Medium=" << lambda_med << ", High=" << lambda_high << " customers/hour\n";
    std::cout << "Service Rates: Efficient=" << service_rate_efficient*60 << ", Slow=" << service_rate_slow*60 << " customers/hour\n\n";
    
    // Traffic intensity (ρ = λ/μ)
    std::cout << "Traffic Intensity (ρ = λ/μ):\n";
    std::cout << "Low Load + Efficient:   ρ = " << lambda_low / (service_rate_efficient * 60) << std::endl;
    std::cout << "Medium Load + Efficient: ρ = " << lambda_med / (service_rate_efficient * 60) << std::endl;
    std::cout << "High Load + Efficient:  ρ = " << lambda_high / (service_rate_efficient * 60) << std::endl;
    std::cout << "Medium Load + Slow:     ρ = " << lambda_med / (service_rate_slow * 60) << std::endl;
    std::cout << "High Load + Slow:       ρ = " << lambda_high / (service_rate_slow * 60) << " (UNSTABLE!)" << std::endl;
    std::cout << std::endl;
    
    // Average queue length (L = ρ/(1-ρ) for M/M/1)
    auto calcQueueLength = [](double rho) -> double {
        return rho < 1.0 ? rho / (1.0 - rho) : std::numeric_limits<double>::infinity();
    };
    
    double rho1 = lambda_med / (service_rate_efficient * 60);
    double rho2 = lambda_high / (service_rate_efficient * 60);
    double rho3 = lambda_med / (service_rate_slow * 60);
    
    std::cout << "Average Queue Length (L = ρ/(1-ρ)):\n";
    std::cout << "Medium Load + Efficient: L = " << calcQueueLength(rho1) << " customers\n";
    std::cout << "High Load + Efficient:   L = " << calcQueueLength(rho2) << " customers\n";
    std::cout << "Medium Load + Slow:      L = " << calcQueueLength(rho3) << " customers\n";
    std::cout << std::endl;
    
    // Training example with synthetic queuing data
    std::cout << "=== Training Example with Synthetic Queuing Data ===\n";
    
    ObservationLists trainingData;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Generate low load periods (few arrivals)
    std::poisson_distribution<int> lowLoadArrivals(3);
    for (int i = 0; i < 20; ++i) {
        ObservationSet seq(6);  // 6-hour periods
        for (size_t j = 0; j < seq.size(); ++j) {
            seq(j) = lowLoadArrivals(gen);
        }
        trainingData.push_back(seq);
    }
    
    // Generate medium load periods
    std::poisson_distribution<int> mediumLoadArrivals(8);
    for (int i = 0; i < 25; ++i) {
        ObservationSet seq(6);
        for (size_t j = 0; j < seq.size(); ++j) {
            seq(j) = mediumLoadArrivals(gen);
        }
        trainingData.push_back(seq);
    }
    
    // Generate high load periods
    std::poisson_distribution<int> highLoadArrivals(15);
    for (int i = 0; i < 15; ++i) {
        ObservationSet seq(6);
        for (size_t j = 0; j < seq.size(); ++j) {
            seq(j) = highLoadArrivals(gen);
        }
        trainingData.push_back(seq);
    }
    
    // Create fresh HMM for training
    auto trainHmm = std::make_unique<Hmm>(3);
    trainHmm->setProbabilityDistribution(0, std::make_unique<PoissonDistribution>(4.0));   // Initial guess
    trainHmm->setProbabilityDistribution(1, std::make_unique<PoissonDistribution>(9.0));   // Initial guess
    trainHmm->setProbabilityDistribution(2, std::make_unique<PoissonDistribution>(12.0));  // Initial guess
    
    std::cout << "Before training:\n";
    std::cout << "Low load state:    " << trainHmm->getProbabilityDistribution(0)->toString() << std::endl;
    std::cout << "Medium load state: " << trainHmm->getProbabilityDistribution(1)->toString() << std::endl;
    std::cout << "High load state:   " << trainHmm->getProbabilityDistribution(2)->toString() << std::endl;
    
    ViterbiTrainer trainer(trainHmm.get(), trainingData);
    trainer.train();
    
    std::cout << "\nAfter training:\n";
    std::cout << "Low load state:    " << trainHmm->getProbabilityDistribution(0)->toString() << std::endl;
    std::cout << "Medium load state: " << trainHmm->getProbabilityDistribution(1)->toString() << std::endl;
    std::cout << "High load state:   " << trainHmm->getProbabilityDistribution(2)->toString() << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Queuing Theory Models and Applications ===\n";
    std::cout << "Common Queuing Models:\n";
    std::cout << "• M/M/1: Poisson arrivals, Exponential service, 1 server\n";
    std::cout << "• M/M/c: Poisson arrivals, Exponential service, c servers\n";
    std::cout << "• M/G/1: Poisson arrivals, General service distribution\n";
    std::cout << "• G/G/1: General arrivals, General service times\n";
    std::cout << "• M/M/1/K: Finite capacity system (K customers max)\n";
    std::cout << std::endl;
    
    std::cout << "Key Performance Metrics:\n";
    std::cout << "• Traffic Intensity (ρ): Utilization ratio λ/μ\n";
    std::cout << "• Average Queue Length (L): Expected customers in system\n";
    std::cout << "• Average Wait Time (W): Expected time in system (Little's Law: L = λW)\n";
    std::cout << "• Server Utilization: Fraction of time server is busy\n";
    std::cout << "• Throughput: Rate of completed services\n";
    std::cout << std::endl;
    
    std::cout << "=== Applications of Queuing Theory HMMs ===\n";
    std::cout << "Service Industry:\n";
    std::cout << "• Call center staffing and performance optimization\n";
    std::cout << "• Hospital emergency department management\n";
    std::cout << "• Bank teller and ATM queue analysis\n";
    std::cout << "• Restaurant and retail service optimization\n";
    std::cout << std::endl;
    
    std::cout << "Technology Systems:\n";
    std::cout << "• Computer system performance modeling\n";
    std::cout << "• Network traffic analysis and capacity planning\n";
    std::cout << "• Cloud computing resource allocation\n";
    std::cout << "• Database query optimization\n";
    std::cout << std::endl;
    
    std::cout << "Transportation:\n";
    std::cout << "• Traffic light timing optimization\n";
    std::cout << "• Airport check-in and security queue management\n";
    std::cout << "• Public transportation scheduling\n";
    std::cout << "• Logistics and supply chain optimization\n";
    std::cout << std::endl;
    
    std::cout << "Manufacturing:\n";
    std::cout << "• Production line bottleneck analysis\n";
    std::cout << "• Inventory management and buffer sizing\n";
    std::cout << "• Quality control station design\n";
    std::cout << "• Machine maintenance scheduling\n";
    
    return 0;
}
