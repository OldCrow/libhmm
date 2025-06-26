#include <iostream>
#include <vector>
#include <iomanip>
#include <memory>
#include <random>
#include <cmath>

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

struct NumericalAnalysisResult {
    string description;
    double libhmm_value;
    double hmmlib_value;
    double absolute_difference;
    double relative_difference;
    
    void print() const {
        cout << "=== " << description << " ===" << endl;
        cout << "  libhmm:     " << scientific << setprecision(15) << libhmm_value << endl;
        cout << "  HMMLib:     " << scientific << setprecision(15) << hmmlib_value << endl;
        cout << "  Abs diff:   " << scientific << setprecision(6) << absolute_difference << endl;
        cout << "  Rel diff:   " << fixed << setprecision(6) << (relative_difference * 100) << "%" << endl;
        cout << endl;
    }
};

class NumericalAccuracyInvestigator {
private:
    // Fixed seed for reproducibility
    mt19937 gen;
    
    // Simple 2x2 casino problem parameters
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
    
public:
    NumericalAccuracyInvestigator() : gen(42) {}
    
    vector<unsigned int> generateSequence(size_t length) {
        uniform_int_distribution<unsigned int> dist(0, 5);
        vector<unsigned int> sequence(length);
        for (size_t i = 0; i < length; ++i) {
            sequence[i] = dist(gen);
        }
        return sequence;
    }
    
    pair<unique_ptr<libhmm::Hmm>, hmmlib::HMM<double>> setupModels() {
        // === Setup libhmm model ===
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
        
        // === Setup HMMLib model ===
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
        
        hmmlib::HMM<double> hmmlib_hmm(pi_ptr, T_ptr, E_ptr);
        
        return {move(hmm), move(hmmlib_hmm)};
    }
    
    void analyzeSequenceLengthDependence() {
        cout << "NUMERICAL ACCURACY ANALYSIS: SEQUENCE LENGTH DEPENDENCE" << endl;
        cout << "=========================================================" << endl << endl;
        
        vector<size_t> test_lengths = {10, 50, 100, 200, 500, 1000, 2000};
        vector<NumericalAnalysisResult> results;
        
        for (size_t length : test_lengths) {
            cout << "Testing sequence length: " << length << endl;
            
            auto sequence = generateSequence(length);
            auto [libhmm_hmm, hmmlib_hmm] = setupModels();
            
            // Convert to libhmm format
            libhmm::ObservationSet libhmm_obs(sequence.size());
            for (size_t i = 0; i < sequence.size(); ++i) {
                libhmm_obs(i) = static_cast<libhmm::Observation>(sequence[i]);
            }
            
            // === libhmm computation ===
            libhmm::forwardbackward::AutoCalculator auto_calc(libhmm_hmm.get(), libhmm_obs);
            double libhmm_likelihood = auto_calc.getLogProbability();
            
            // === HMMLib computation ===
            hmmlib::HMMMatrix<double> F(sequence.size(), 2);
            hmmlib::HMMMatrix<double> B(sequence.size(), 2);
            hmmlib::HMMVector<double> scales(sequence.size());
            
            hmmlib_hmm.forward(sequence, scales, F);
            hmmlib_hmm.backward(sequence, scales, B);
            double hmmlib_likelihood = hmmlib_hmm.likelihood(scales);
            
            // Calculate differences
            double abs_diff = abs(libhmm_likelihood - hmmlib_likelihood);
            double rel_diff = abs_diff / abs(hmmlib_likelihood);
            
            NumericalAnalysisResult result = {
                "Length " + to_string(length),
                libhmm_likelihood,
                hmmlib_likelihood,
                abs_diff,
                rel_diff
            };
            
            result.print();
            results.push_back(result);
        }
        
        // Analyze trend
        cout << "TREND ANALYSIS:" << endl;
        cout << "===============" << endl;
        cout << "Length\t\tRel Error (%)" << endl;
        for (const auto& result : results) {
            cout << result.description << "\t\t" << fixed << setprecision(6) 
                 << (result.relative_difference * 100) << "%" << endl;
        }
        cout << endl;
    }
    
    void analyzeStepByStepComputation() {
        cout << "STEP-BY-STEP FORWARD ALGORITHM ANALYSIS" << endl;
        cout << "=======================================" << endl << endl;
        
        // Use a very short sequence for detailed analysis
        vector<unsigned int> sequence = {3, 4, 2};  // Fixed short sequence
        cout << "Test sequence: ";
        for (auto obs : sequence) {
            cout << obs << " ";
        }
        cout << endl << endl;
        
        auto [libhmm_hmm, hmmlib_hmm] = setupModels();
        
        // Convert to libhmm format
        libhmm::ObservationSet libhmm_obs(sequence.size());
        for (size_t i = 0; i < sequence.size(); ++i) {
            libhmm_obs(i) = static_cast<libhmm::Observation>(sequence[i]);
        }
        
        // === libhmm detailed computation ===
        cout << "=== libhmm ScaledSIMD Calculator ===" << endl;
        libhmm::ScaledSIMDForwardBackwardCalculator libhmm_calc(libhmm_hmm.get(), libhmm_obs);
        libhmm_calc.compute();
        
        auto libhmm_forward = libhmm_calc.getForwardVariables();
        auto libhmm_scaling = libhmm_calc.getScalingFactors();
        double libhmm_log_prob = libhmm_calc.getLogProbability();
        
        cout << "Forward variables:" << endl;
        for (size_t t = 0; t < sequence.size(); ++t) {
            cout << "t=" << t << ": ";
            for (size_t s = 0; s < 2; ++s) {
                cout << scientific << setprecision(6) << libhmm_forward(t, s) << " ";
            }
            cout << endl;
        }
        
        cout << "Scaling factors: ";
        for (size_t t = 0; t < sequence.size(); ++t) {
            cout << scientific << setprecision(6) << libhmm_scaling[t] << " ";
        }
        cout << endl;
        cout << "Log probability: " << scientific << setprecision(15) << libhmm_log_prob << endl << endl;
        
        // === HMMLib detailed computation ===
        cout << "=== HMMLib Calculator ===" << endl;
        hmmlib::HMMMatrix<double> F(sequence.size(), 2);
        hmmlib::HMMMatrix<double> B(sequence.size(), 2);
        hmmlib::HMMVector<double> scales(sequence.size());
        
        hmmlib_hmm.forward(sequence, scales, F);
        double hmmlib_log_prob = hmmlib_hmm.likelihood(scales);
        
        cout << "Forward variables:" << endl;
        for (size_t t = 0; t < sequence.size(); ++t) {
            cout << "t=" << t << ": ";
            for (size_t s = 0; s < 2; ++s) {
                cout << scientific << setprecision(6) << F(t, s) << " ";
            }
            cout << endl;
        }
        
        cout << "Scaling factors: ";
        for (size_t t = 0; t < sequence.size(); ++t) {
            cout << scientific << setprecision(6) << scales(t) << " ";
        }
        cout << endl;
        cout << "Log probability: " << scientific << setprecision(15) << hmmlib_log_prob << endl << endl;
        
        // === Compare step by step ===
        cout << "=== Step-by-step comparison ===" << endl;
        
        // Compare forward variables
        for (size_t t = 0; t < sequence.size(); ++t) {
            for (size_t s = 0; s < 2; ++s) {
                double diff = abs(libhmm_forward(t, s) - F(t, s));
                double rel_diff = diff / abs(F(t, s));
                cout << "F[" << t << "," << s << "] diff: " << scientific << setprecision(3) 
                     << diff << " (" << fixed << setprecision(3) << (rel_diff * 100) << "%)" << endl;
            }
        }
        
        // Compare scaling factors
        for (size_t t = 0; t < sequence.size(); ++t) {
            double diff = abs(libhmm_scaling[t] - scales(t));
            double rel_diff = diff / abs(scales(t));
            cout << "Scale[" << t << "] diff: " << scientific << setprecision(3) 
                 << diff << " (" << fixed << setprecision(3) << (rel_diff * 100) << "%)" << endl;
        }
        
        // Final probability comparison
        double final_diff = abs(libhmm_log_prob - hmmlib_log_prob);
        double final_rel_diff = final_diff / abs(hmmlib_log_prob);
        cout << "Final log-prob diff: " << scientific << setprecision(6) << final_diff 
             << " (" << fixed << setprecision(6) << (final_rel_diff * 100) << "%)" << endl << endl;
    }
    
    void investigateScalingDifferences() {
        cout << "SCALING STRATEGY INVESTIGATION" << endl;
        cout << "==============================" << endl << endl;
        
        // Test different libhmm calculators
        vector<unsigned int> sequence = {3, 4, 2, 1, 5, 0, 3, 4, 2, 1};  // 10 observations
        cout << "Test sequence: ";
        for (auto obs : sequence) {
            cout << obs << " ";
        }
        cout << endl << endl;
        
        auto [libhmm_hmm, hmmlib_hmm] = setupModels();
        
        // Convert to libhmm format
        libhmm::ObservationSet libhmm_obs(sequence.size());
        for (size_t i = 0; i < sequence.size(); ++i) {
            libhmm_obs(i) = static_cast<libhmm::Observation>(sequence[i]);
        }
        
        // Test AutoCalculator
        cout << "=== AutoCalculator ===" << endl;
        libhmm::forwardbackward::AutoCalculator auto_calc(libhmm_hmm.get(), libhmm_obs);
        double auto_result = auto_calc.getLogProbability();
        cout << "Selected: " << auto_calc.getSelectionRationale() << endl;
        cout << "Result: " << scientific << setprecision(15) << auto_result << endl << endl;
        
        // Test ScaledSIMD Calculator
        cout << "=== ScaledSIMD Calculator ===" << endl;
        libhmm::ScaledSIMDForwardBackwardCalculator scaled_calc(libhmm_hmm.get(), libhmm_obs);
        scaled_calc.compute();
        double scaled_result = scaled_calc.getLogProbability();
        auto scaling_factors = scaled_calc.getScalingFactors();
        cout << "Result: " << scientific << setprecision(15) << scaled_result << endl;
        cout << "Scaling factors: ";
        for (double sf : scaling_factors) {
            cout << scientific << setprecision(6) << sf << " ";
        }
        cout << endl << endl;
        
        // Test unscaled calculator
        cout << "=== Unscaled Calculator ===" << endl;
        libhmm::ForwardBackwardCalculator unscaled_calc(libhmm_hmm.get(), libhmm_obs);
        double unscaled_prob = unscaled_calc.probability();
        double unscaled_result = (unscaled_prob > 0) ? log(unscaled_prob) : -numeric_limits<double>::infinity();
        cout << "Raw probability: " << scientific << setprecision(15) << unscaled_prob << endl;
        cout << "Log result: " << scientific << setprecision(15) << unscaled_result << endl << endl;
        
        // HMMLib reference
        cout << "=== HMMLib Reference ===" << endl;
        hmmlib::HMMMatrix<double> F(sequence.size(), 2);
        hmmlib::HMMMatrix<double> B(sequence.size(), 2);
        hmmlib::HMMVector<double> scales(sequence.size());
        
        hmmlib_hmm.forward(sequence, scales, F);
        double hmmlib_result = hmmlib_hmm.likelihood(scales);
        cout << "Result: " << scientific << setprecision(15) << hmmlib_result << endl;
        cout << "Scaling factors: ";
        for (size_t i = 0; i < sequence.size(); ++i) {
            cout << scientific << setprecision(6) << scales(i) << " ";
        }
        cout << endl << endl;
        
        // Compare all results
        cout << "=== Comparison ===" << endl;
        vector<pair<string, double>> results = {
            {"AutoCalculator", auto_result},
            {"ScaledSIMD", scaled_result},
            {"Unscaled", unscaled_result},
            {"HMMLib", hmmlib_result}
        };
        
        for (const auto& result : results) {
            double diff = abs(result.second - hmmlib_result);
            double rel_diff = diff / abs(hmmlib_result);
            cout << result.first << " vs HMMLib: " << scientific << setprecision(6) << diff
                 << " (" << fixed << setprecision(6) << (rel_diff * 100) << "%)" << endl;
        }
    }
};

int main() {
    NumericalAccuracyInvestigator investigator;
    
    cout << "DEEP NUMERICAL ACCURACY INVESTIGATION" << endl;
    cout << "=====================================" << endl << endl;
    
    // Test 1: Sequence length dependence
    investigator.analyzeSequenceLengthDependence();
    
    cout << "\n" << string(80, '=') << "\n" << endl;
    
    // Test 2: Step-by-step analysis
    investigator.analyzeStepByStepComputation();
    
    cout << "\n" << string(80, '=') << "\n" << endl;
    
    // Test 3: Scaling strategy investigation
    investigator.investigateScalingDifferences();
    
    return 0;
}
