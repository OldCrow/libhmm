#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>

// libhmm includes
#include "libhmm/hmm.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/viterbi_calculator.h"

// HMMLib includes
#include "HMMlib/hmm.hpp"
#include "HMMlib/hmm_vector.hpp"
#include "HMMlib/hmm_matrix.hpp"

using namespace std;

void demonstrateLibHmm() {
    cout << "=== libhmm Example: Simple 2-State Weather Model ===" << endl;
    
    try {
        // Create 2-state HMM: Sunny(0), Rainy(1)
        // Observations: Hot(0), Cold(1)
        auto hmm = make_unique<libhmm::Hmm>(2);
        
        // Set initial probabilities: P(Sunny) = 0.6, P(Rainy) = 0.4
        libhmm::Vector pi(2);
        pi(0) = 0.6; // Sunny
        pi(1) = 0.4; // Rainy
        hmm->setPi(pi);
        
        // Set transition matrix
        // From Sunny: P(Sunny->Sunny) = 0.7, P(Sunny->Rainy) = 0.3
        // From Rainy: P(Rainy->Sunny) = 0.4, P(Rainy->Rainy) = 0.6
        libhmm::Matrix trans(2, 2);
        trans(0, 0) = 0.7; trans(0, 1) = 0.3;  // From Sunny
        trans(1, 0) = 0.4; trans(1, 1) = 0.6;  // From Rainy
        hmm->setTrans(trans);
        
        // Set emission distributions
        // Sunny state: P(Hot|Sunny) = 0.8, P(Cold|Sunny) = 0.2
        auto sunny_dist = make_unique<libhmm::DiscreteDistribution>(2);
        sunny_dist->setProbability(0, 0.8);  // Hot given Sunny
        sunny_dist->setProbability(1, 0.2);  // Cold given Sunny
        hmm->setProbabilityDistribution(0, sunny_dist.release());
        
        // Rainy state: P(Hot|Rainy) = 0.3, P(Cold|Rainy) = 0.7
        auto rainy_dist = make_unique<libhmm::DiscreteDistribution>(2);
        rainy_dist->setProbability(0, 0.3);  // Hot given Rainy
        rainy_dist->setProbability(1, 0.7);  // Cold given Rainy
        hmm->setProbabilityDistribution(1, rainy_dist.release());
        
        // Create observation sequence: Hot, Hot, Cold, Hot
        libhmm::ObservationSet obs(4);
        obs(0) = 0; // Hot
        obs(1) = 0; // Hot
        obs(2) = 1; // Cold
        obs(3) = 0; // Hot
        
        cout << "Observation sequence: Hot, Hot, Cold, Hot" << endl;
        
        // Run forward-backward algorithm
        libhmm::ForwardBackwardCalculator fb_calc(hmm.get(), obs);
        double likelihood = fb_calc.probability();
        double log_likelihood = log(likelihood);
        
        cout << "Forward-Backward Results:" << endl;
        cout << "  Likelihood: " << scientific << setprecision(6) << likelihood << endl;
        cout << "  Log-likelihood: " << log_likelihood << endl;
        
        // Run Viterbi algorithm
        libhmm::ViterbiCalculator viterbi_calc(hmm.get(), obs);
        libhmm::StateSequence best_path = viterbi_calc.decode();
        double viterbi_log_prob = viterbi_calc.getLogProbability();
        
        cout << "Viterbi Results:" << endl;
        cout << "  Viterbi log-probability: " << viterbi_log_prob << endl;
        cout << "  Most likely state sequence: ";
        for (size_t i = 0; i < best_path.size(); ++i) {
            cout << (best_path[i] == 0 ? "Sunny" : "Rainy");
            if (i < best_path.size() - 1) cout << " -> ";
        }
        cout << endl;
        
    } catch (const exception& e) {
        cout << "libhmm error: " << e.what() << endl;
    }
}

void demonstrateHmmLib() {
    cout << "\n=== HMMLib Example: Same 2-State Weather Model ===" << endl;
    
    try {
        // Create HMMLib matrices and vectors
        auto pi_ptr = boost::shared_ptr<hmmlib::HMMVector<double>>(
            new hmmlib::HMMVector<double>(2));
        auto T_ptr = boost::shared_ptr<hmmlib::HMMMatrix<double>>(
            new hmmlib::HMMMatrix<double>(2, 2));
        auto E_ptr = boost::shared_ptr<hmmlib::HMMMatrix<double>>(
            new hmmlib::HMMMatrix<double>(2, 2));  // 2 observations, 2 states
        
        // Set initial probabilities
        (*pi_ptr)(0) = 0.6; // Sunny
        (*pi_ptr)(1) = 0.4; // Rainy
        
        // Set transition matrix
        (*T_ptr)(0, 0) = 0.7; (*T_ptr)(0, 1) = 0.3;  // From Sunny
        (*T_ptr)(1, 0) = 0.4; (*T_ptr)(1, 1) = 0.6;  // From Rainy
        
        // Set emission matrix (note: HMMLib uses E(observation, state) indexing)
        (*E_ptr)(0, 0) = 0.8; (*E_ptr)(0, 1) = 0.3;  // P(Hot | Sunny), P(Hot | Rainy)
        (*E_ptr)(1, 0) = 0.2; (*E_ptr)(1, 1) = 0.7;  // P(Cold | Sunny), P(Cold | Rainy)
        
        // Create HMM
        hmmlib::HMM<double> hmm(pi_ptr, T_ptr, E_ptr);
        
        // Same observation sequence: Hot, Hot, Cold, Hot
        vector<unsigned int> obs_sequence = {0, 0, 1, 0};
        
        cout << "Observation sequence: Hot, Hot, Cold, Hot" << endl;
        
        // Prepare matrices for forward-backward
        hmmlib::HMMMatrix<double> F(obs_sequence.size(), 2);
        hmmlib::HMMMatrix<double> B(obs_sequence.size(), 2);
        hmmlib::HMMVector<double> scales(obs_sequence.size());
        
        // Run forward-backward algorithm
        hmm.forward(obs_sequence, scales, F);
        hmm.backward(obs_sequence, scales, B);
        double log_likelihood = hmm.likelihood(scales);
        
        cout << "Forward-Backward Results:" << endl;
        cout << "  Log-likelihood: " << scientific << setprecision(6) << log_likelihood << endl;
        
        // Run Viterbi algorithm
        vector<unsigned int> best_path(obs_sequence.size());
        double viterbi_log_likelihood = hmm.viterbi(obs_sequence, best_path);
        
        cout << "Viterbi Results:" << endl;
        cout << "  Viterbi log-likelihood: " << viterbi_log_likelihood << endl;
        cout << "  Most likely state sequence: ";
        for (size_t i = 0; i < best_path.size(); ++i) {
            cout << (best_path[i] == 0 ? "Sunny" : "Rainy");
            if (i < best_path.size() - 1) cout << " -> ";
        }
        cout << endl;
        
    } catch (const exception& e) {
        cout << "HMMLib error: " << e.what() << endl;
    }
}

void compareResults() {
    cout << "\n=== Comparison Notes ===" << endl;
    cout << "Both libraries should produce similar log-likelihood values" << endl;
    cout << "for the same HMM configuration and observation sequence." << endl;
    cout << "Small differences may occur due to:" << endl;
    cout << "  - Different numerical precision handling" << endl;
    cout << "  - Different scaling strategies" << endl;
    cout << "  - Different underlying algorithms" << endl;
    cout << "\nIf results differ significantly, check:" << endl;
    cout << "  - Matrix/vector indexing conventions" << endl;
    cout << "  - Probability normalization" << endl;
    cout << "  - Observation encoding" << endl;
}

int main() {
    cout << "Library Usage Examples: Proper HMM Setup and Comparison" << endl;
    cout << "=======================================================" << endl;
    
    demonstrateLibHmm();
    demonstrateHmmLib();
    compareResults();
    
    return 0;
}
