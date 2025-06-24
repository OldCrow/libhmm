#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <random>
#include <functional>
#include "../include/libhmm/libhmm.h"
#include "../include/libhmm/calculators/log_simd_forward_backward_calculator.h"
#include "../include/libhmm/calculators/scaled_simd_forward_backward_calculator.h"
#include "../include/libhmm/calculators/log_forward_backward_calculator.h"
#include "../include/libhmm/calculators/scaled_forward_backward_calculator.h"

using namespace std::chrono;

class PerformanceTimer {
private:
    high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(end_time - start_time);
        return duration.count() / 1000000.0; // Return milliseconds
    }
};

struct PerformanceResult {
    std::string calculator_name;
    std::size_t num_states;
    std::size_t seq_length;
    double time_ms;
    double log_prob;
    
    void print() const {
        std::cout << std::setw(20) << calculator_name 
                  << std::setw(8) << num_states
                  << std::setw(12) << seq_length  
                  << std::setw(12) << std::fixed << std::setprecision(3) << time_ms << "ms"
                  << std::setw(18) << std::scientific << std::setprecision(3) << log_prob
                  << std::endl;
    }
};

std::unique_ptr<libhmm::Hmm> createTestHMM(std::size_t num_states, std::size_t alphabet_size) {
    auto hmm = std::make_unique<libhmm::Hmm>(num_states);
    
    // Initialize with reasonable probabilities
    libhmm::Vector pi(num_states);
    libhmm::Matrix trans(num_states, num_states);
    
    // Equal initial probabilities
    for (std::size_t i = 0; i < num_states; ++i) {
        pi(i) = 1.0 / num_states;
    }
    
    // Random-ish but reproducible transition matrix
    std::mt19937 gen(42); // Fixed seed
    std::uniform_real_distribution<double> dist(0.1, 1.0);
    
    for (std::size_t i = 0; i < num_states; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < num_states; ++j) {
            trans(i, j) = dist(gen);
            sum += trans(i, j);
        }
        // Normalize
        for (std::size_t j = 0; j < num_states; ++j) {
            trans(i, j) /= sum;
        }
    }
    
    hmm->setPi(pi);
    hmm->setTrans(trans);
    
    // Set discrete emission distributions
    for (std::size_t i = 0; i < num_states; ++i) {
        auto dist_ptr = std::make_unique<libhmm::DiscreteDistribution>(alphabet_size);
        libhmm::Vector probs(alphabet_size);
        double sum = 0.0;
        for (std::size_t j = 0; j < alphabet_size; ++j) {
            probs(j) = dist(gen);
            sum += probs(j);
        }
        // Normalize and set
        for (std::size_t j = 0; j < alphabet_size; ++j) {
            probs(j) /= sum;
            dist_ptr->setProbability(j, probs(j));
        }
        hmm->setProbabilityDistribution(i, std::move(dist_ptr));
    }
    
    return hmm;
}

libhmm::ObservationSet createTestObservations(std::size_t seq_length, std::size_t alphabet_size) {
    libhmm::ObservationSet obs(seq_length);
    std::mt19937 gen(123); // Different seed from HMM
    std::uniform_int_distribution<unsigned int> dist(0, alphabet_size - 1);
    
    for (std::size_t i = 0; i < seq_length; ++i) {
        obs(i) = dist(gen);
    }
    
    return obs;
}

PerformanceResult testCalculator(const std::string& name, 
                                std::function<std::unique_ptr<libhmm::ForwardBackwardCalculator>(libhmm::Hmm*, const libhmm::ObservationSet&)> factory,
                                libhmm::Hmm* hmm, const libhmm::ObservationSet& obs, std::size_t iterations = 5) {
    
    PerformanceTimer timer;
    double total_time = 0.0;
    double log_prob = 0.0;
    
    // Warmup
    auto warmup_calc = factory(hmm, obs);
    
    // Get log probability depending on calculator type
    if (name == "Scaled" || name == "Scaled-SIMD") {
        if (auto scaled_calc = dynamic_cast<libhmm::ScaledForwardBackwardCalculator*>(warmup_calc.get())) {
            log_prob = scaled_calc->logProbability();
        }
    } else if (name == "Log-Space" || name == "Log-SIMD") {
        if (auto log_calc = dynamic_cast<libhmm::LogForwardBackwardCalculator*>(warmup_calc.get())) {
            log_prob = log_calc->logProbability();
        }
    }
    
    // Benchmark
    for (std::size_t i = 0; i < iterations; ++i) {
        timer.start();
        auto calc = factory(hmm, obs);
        volatile double prob = calc->probability(); // Use base method to ensure all work
        total_time += timer.stop();
        (void)prob; // Suppress unused variable warning
    }
    
    return {
        name,
        static_cast<std::size_t>(hmm->getNumStates()),
        obs.size(),
        total_time / iterations,
        log_prob
    };
}

int main() {
    std::cout << "=== Direct Calculator Performance Comparison ===\n\n";
    
    std::cout << std::setw(20) << "Calculator" 
              << std::setw(8) << "States"
              << std::setw(12) << "Seq Length"
              << std::setw(12) << "Time"
              << std::setw(18) << "Log Probability"
              << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    // Test configurations
    std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> configs = {
        {5, 4, 50},      // Small  
        {5, 4, 100},     // Medium
        {10, 6, 200},    // Large
        {20, 8, 500},    // Very Large
    };
    
    for (const auto& config : configs) {
        std::size_t num_states, alphabet_size, seq_length;
        std::tie(num_states, alphabet_size, seq_length) = config;
        
        std::cout << "\n--- Configuration: " << num_states << " states, " 
                  << alphabet_size << " symbols, " << seq_length << " observations ---\n";
        
        auto hmm = createTestHMM(num_states, alphabet_size);
        auto obs = createTestObservations(seq_length, alphabet_size);
        
        // Test different calculators
        auto results = std::vector<PerformanceResult>{
            testCalculator("Scaled", 
                [](libhmm::Hmm* h, const libhmm::ObservationSet& o) -> std::unique_ptr<libhmm::ForwardBackwardCalculator> {
                    return std::make_unique<libhmm::ScaledForwardBackwardCalculator>(h, o);
                }, hmm.get(), obs),
                
            testCalculator("Log-Space", 
                [](libhmm::Hmm* h, const libhmm::ObservationSet& o) -> std::unique_ptr<libhmm::ForwardBackwardCalculator> {
                    return std::make_unique<libhmm::LogForwardBackwardCalculator>(h, o);
                }, hmm.get(), obs),
                
            testCalculator("Log-SIMD", 
                [](libhmm::Hmm* h, const libhmm::ObservationSet& o) -> std::unique_ptr<libhmm::ForwardBackwardCalculator> {
                    return std::make_unique<libhmm::LogSIMDForwardBackwardCalculator>(h, o);
                }, hmm.get(), obs),
                
            testCalculator("Scaled-SIMD", 
                [](libhmm::Hmm* h, const libhmm::ObservationSet& o) -> std::unique_ptr<libhmm::ForwardBackwardCalculator> {
                    return std::make_unique<libhmm::ScaledSIMDForwardBackwardCalculator>(h, o);
                }, hmm.get(), obs)
        };
        
        // Print results
        for (const auto& result : results) {
            result.print();
        }
        
        // Calculate speedups
        double scaled_time = results[0].time_ms;
        std::cout << "\nSpeedups vs Scaled:\n";
        for (std::size_t i = 1; i < results.size(); ++i) {
            double speedup = scaled_time / results[i].time_ms;
            std::cout << "  " << results[i].calculator_name << ": " 
                      << std::fixed << std::setprecision(2) << speedup << "x\n";
        }
        
        // Numerical accuracy check
        std::cout << "\nNumerical Accuracy (log probability differences from Scaled):\n";
        double reference_prob = results[0].log_prob;
        for (std::size_t i = 1; i < results.size(); ++i) {
            double diff = std::abs(results[i].log_prob - reference_prob);
            std::cout << "  " << results[i].calculator_name << ": " 
                      << std::scientific << std::setprecision(2) << diff << "\n";
        }
    }
    
    return 0;
}
