#include <iostream>
#include <vector>
#include <iomanip>
#include <memory>
#include <random>

// libhmm includes
#include "libhmm/hmm.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/calculators/calculators.h"

// HMMLib includes  
#include "HMMlib/hmm_table.hpp"
#include "HMMlib/hmm_vector.hpp"
#include "HMMlib/hmm_matrix.hpp"
#include "HMMlib/hmm.hpp"

using namespace std;

// Fixed seed and sequence for reproducible testing
random_device rd;
mt19937 gen(42);

void testNumericalAccuracy() {
    cout << "=== NUMERICAL ACCURACY INVESTIGATION ===" << endl;
    
    // Simple 2x2 casino problem
    vector<double> initial_probs = {0.5, 0.5};  // Fair, Loaded
    vector<vector<double>> transition_matrix = {
        {0.95, 0.05},  // Fair -> {Fair, Loaded}
        {0.10, 0.90}   // Loaded -> {Fair, Loaded}
    };
    vector<vector<double>> emission_matrix = {
        // Emissions: 0, 1, 2, 3, 4, 5 (dice faces 1-6 as 0-5)
        {1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6},  // Fair die
        {0.10, 0.10, 0.10, 0.10, 0.10, 0.50}          // Loaded die (5 is more likely)
    };
    
    // Generate simple test sequence
    uniform_int_distribution<unsigned int> dist(0, 5);
    vector<unsigned int> obs_sequence(10);  // Just 10 observations for debugging
    for (int i = 0; i < 10; ++i) {
        obs_sequence[i] = dist(gen);
    }
    
    cout << "Test sequence: ";
    for (auto obs : obs_sequence) {
        cout << obs << " ";
    }
    cout << endl;
    
    // === libhmm test ===
    cout << "\n--- libhmm computation ---" << endl;
    try {
        auto hmm = make_unique<libhmm::Hmm>(2);
        
        // Set up transition matrix
        libhmm::Matrix trans_matrix(2, 2);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                trans_matrix(i, j) = transition_matrix[i][j];
            }
        }
        hmm->setTrans(trans_matrix);
        
        // Set up initial probabilities
        libhmm::Vector pi_vector(2);
        for (int i = 0; i < 2; ++i) {
            pi_vector(i) = initial_probs[i];
        }
        hmm->setPi(pi_vector);
        
        // Set discrete emission distributions
        for (int i = 0; i < 2; ++i) {
            auto discrete_dist = make_unique<libhmm::DiscreteDistribution>(6);
            for (int j = 0; j < 6; ++j) {
                discrete_dist->setProbability(static_cast<libhmm::Observation>(j), emission_matrix[i][j]);
            }
            hmm->setProbabilityDistribution(i, discrete_dist.release());
        }
        
        // Convert observation sequence to libhmm format
        libhmm::ObservationSet libhmm_obs(obs_sequence.size());
        for (size_t i = 0; i < obs_sequence.size(); ++i) {
            libhmm_obs(i) = static_cast<libhmm::Observation>(obs_sequence[i]);
        }
        
        // Test different calculators
        cout << "Testing different libhmm calculators:" << endl;
        
        // Auto calculator
        libhmm::forwardbackward::AutoCalculator auto_calc(hmm.get(), libhmm_obs);
        double auto_likelihood = auto_calc.getLogProbability();
        cout << "  AutoCalculator result: " << scientific << setprecision(10) << auto_likelihood << endl;
        cout << "  Selected: " << auto_calc.getSelectionRationale() << endl;
        
        // Manual scaled calculator for comparison
        libhmm::ScaledSIMDForwardBackwardCalculator scaled_calc(hmm.get(), libhmm_obs);
        scaled_calc.compute();
        double scaled_likelihood = scaled_calc.getLogProbability();
        cout << "  ScaledSIMD result: " << scientific << setprecision(10) << scaled_likelihood << endl;
        
        // Manual unscaled calculator for comparison
        libhmm::ForwardBackwardCalculator unscaled_calc(hmm.get(), libhmm_obs);
        double unscaled_prob = unscaled_calc.probability();
        double unscaled_likelihood = log(unscaled_prob);
        cout << "  UnscaledSIMD prob: " << scientific << setprecision(10) << unscaled_prob << endl;
        cout << "  UnscaledSIMD log: " << scientific << setprecision(10) << unscaled_likelihood << endl;
        
    } catch (const exception& e) {
        cout << "libhmm error: " << e.what() << endl;
    }
    
    // === HMMLib test ===
    cout << "\n--- HMMLib computation ---" << endl;
    try {
        auto pi_ptr = boost::shared_ptr<hmmlib::HMMVector<double>>(
            new hmmlib::HMMVector<double>(2));
        auto T_ptr = boost::shared_ptr<hmmlib::HMMMatrix<double>>(
            new hmmlib::HMMMatrix<double>(2, 2));
        auto E_ptr = boost::shared_ptr<hmmlib::HMMMatrix<double>>(
            new hmmlib::HMMMatrix<double>(6, 2));
        
        // Set initial probabilities
        for (int i = 0; i < 2; ++i) {
            (*pi_ptr)(i) = initial_probs[i];
        }
        
        // Set transition matrix
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                (*T_ptr)(i, j) = transition_matrix[i][j];
            }
        }
        
        // Set emission matrix (note: HMMLib uses E(symbol, state) indexing)
        for (int symbol = 0; symbol < 6; ++symbol) {
            for (int state = 0; state < 2; ++state) {
                (*E_ptr)(symbol, state) = emission_matrix[state][symbol];
            }
        }
        
        // Create HMMLib HMM
        hmmlib::HMM<double> hmm(pi_ptr, T_ptr, E_ptr);
        
        // Prepare data structures for algorithms
        hmmlib::HMMMatrix<double> F(obs_sequence.size(), 2);
        hmmlib::HMMMatrix<double> B(obs_sequence.size(), 2);
        hmmlib::HMMVector<double> scales(obs_sequence.size());
        
        // Forward-backward computation
        hmm.forward(obs_sequence, scales, F);
        hmm.backward(obs_sequence, scales, B);
        double hmmlib_likelihood = hmm.likelihood(scales);
        
        cout << "HMMLib likelihood: " << scientific << setprecision(10) << hmmlib_likelihood << endl;
        
        // Debug: print the scales to see scaling behavior
        cout << "HMMLib scales: ";
        for (size_t i = 0; i < obs_sequence.size(); ++i) {
            cout << scientific << setprecision(3) << scales(i) << " ";
        }
        cout << endl;
        
    } catch (const exception& e) {
        cout << "HMMLib error: " << e.what() << endl;
    }
    
    cout << "\n=== Analysis ===" << endl;
    cout << "If results differ significantly, this indicates a systematic" << endl;
    cout << "difference in how the libraries compute or scale probabilities." << endl;
}

int main() {
    testNumericalAccuracy();
    return 0;
}
