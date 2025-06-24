#include "libhmm/libhmm.h"
#include <iostream>
#include <vector>
#include <limits>

using namespace libhmm;

int main() {
    std::cout << "=== RobustViterbiTrainer Demonstration ===" << std::endl;
    
    try {
        // Create a 2-state HMM
        Hmm hmm(2);
        
        // Set initial probabilities
        Vector pi(2);
        pi(0) = 0.6;
        pi(1) = 0.4;
        hmm.setPi(pi);
        
        // Set transition matrix
        Matrix trans(2, 2);
        trans(0, 0) = 0.7;  trans(0, 1) = 0.3;
        trans(1, 0) = 0.4;  trans(1, 1) = 0.6;
        hmm.setTrans(trans);
        
        // Set emission distributions
        hmm.setProbabilityDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
        hmm.setProbabilityDistribution(1, std::make_unique<GaussianDistribution>(3.0, 1.5));
        
        // Create normal observation data
        ObservationLists normalData;
        ObservationSet seq1(10);
        std::vector<double> vals1 = {-0.5, 0.2, -0.1, 0.8, -0.3, 0.1, -0.7, 0.4, -0.2, 0.6};
        for (size_t i = 0; i < vals1.size(); ++i) {
            seq1(i) = vals1[i];
        }
        normalData.push_back(seq1);
        
        ObservationSet seq2(8);
        std::vector<double> vals2 = {2.8, 3.2, 3.5, 2.6, 3.8, 2.9, 3.1, 3.4};
        for (size_t i = 0; i < vals2.size(); ++i) {
            seq2(i) = vals2[i];
        }
        normalData.push_back(seq2);
        
        // Create problematic observation data
        ObservationLists problematicData;
        ObservationSet seqProb(6);
        seqProb(0) = 1.0;
        seqProb(1) = std::numeric_limits<double>::quiet_NaN();
        seqProb(2) = 2.0;
        seqProb(3) = std::numeric_limits<double>::infinity();
        seqProb(4) = 3.0;
        seqProb(5) = -std::numeric_limits<double>::infinity();
        problematicData.push_back(seqProb);
        
        std::cout << "\n1. Training with NORMAL data using different presets:" << std::endl;
        
        // Test with different presets
        std::vector<std::string> presets = {"conservative", "balanced", "aggressive", "realtime", "high_precision"};
        
        for (const auto& preset : presets) {
            std::cout << "\n--- Using " << preset << " preset ---" << std::endl;
            
            auto trainer = RobustViterbiTrainer::createWithRecommendedSettings(&hmm, normalData, preset);
            
            // Show configuration
            auto config = trainer->getConfig();
            std::cout << "Configuration:" << std::endl;
            std::cout << "  Max iterations: " << config.maxIterations << std::endl;
            std::cout << "  Convergence tolerance: " << config.convergenceTolerance << std::endl;
            std::cout << "  Adaptive precision: " << (config.enableAdaptivePrecision ? "enabled" : "disabled") << std::endl;
            std::cout << "  Error recovery: " << (config.enableErrorRecovery ? "enabled" : "disabled") << std::endl;
            
            trainer->train();
            
            std::cout << "Results:" << std::endl;
            std::cout << "  Converged: " << (trainer->hasConverged() ? "Yes" : "No") << std::endl;
            std::cout << "  Max iterations reached: " << (trainer->reachedMaxIterations() ? "Yes" : "No") << std::endl;
            
            auto recommendations = trainer->getPerformanceRecommendations();
            if (!recommendations.empty()) {
                std::cout << "  Recommendations:" << std::endl;
                for (const auto& rec : recommendations) {
                    std::cout << "    - " << rec << std::endl;
                }
            }
        }
        
        std::cout << "\n\n2. Training with PROBLEMATIC data (error recovery):" << std::endl;
        
        // Test with problematic data and error recovery
        auto robustConfig = training_presets::balanced();
        robustConfig.enableProgressReporting = true;
        robustConfig.enableErrorRecovery = true;
        robustConfig.enableDiagnostics = true;
        robustConfig.maxIterations = 50;
        
        std::cout << "\n--- Robust training with NaN and Inf values ---" << std::endl;
        std::cout << "Data contains: NaN, +Inf, -Inf values" << std::endl;
        
        RobustViterbiTrainer robustTrainer(&hmm, problematicData, robustConfig);
        robustTrainer.train();
        
        std::cout << "\nTraining Results:" << std::endl;
        std::cout << robustTrainer.getConvergenceReport() << std::endl;
        
        std::cout << "\nNumerical Health:" << std::endl;
        std::cout << robustTrainer.getNumericalHealthReport() << std::endl;
        
        std::cout << "\n\n3. Adaptive precision demonstration:" << std::endl;
        
        // Test adaptive precision
        auto adaptiveConfig = training_presets::balanced();
        adaptiveConfig.enableProgressReporting = true;
        adaptiveConfig.enableAdaptivePrecision = true;
        adaptiveConfig.convergenceTolerance = 1e-10; // Very strict initially
        adaptiveConfig.maxIterations = 100;
        
        RobustViterbiTrainer adaptiveTrainer(&hmm, normalData, adaptiveConfig);
        
        std::cout << "Initial tolerance: " << adaptiveTrainer.getCurrentTolerance() << std::endl;
        
        adaptiveTrainer.train();
        
        std::cout << "Final tolerance: " << adaptiveTrainer.getCurrentTolerance() << std::endl;
        std::cout << "Tolerance adapted: " << (adaptiveTrainer.getCurrentTolerance() != adaptiveConfig.convergenceTolerance ? "Yes" : "No") << std::endl;
        
        std::cout << "\n\n4. Error recovery strategies:" << std::endl;
        
        // Test different recovery strategies
        std::vector<std::pair<std::string, numerical::ErrorRecovery::RecoveryStrategy>> strategies = {
            {"GRACEFUL", numerical::ErrorRecovery::RecoveryStrategy::GRACEFUL},
            {"ROBUST", numerical::ErrorRecovery::RecoveryStrategy::ROBUST},
            {"ADAPTIVE", numerical::ErrorRecovery::RecoveryStrategy::ADAPTIVE}
        };
        
        for (const auto& [name, strategy] : strategies) {
            std::cout << "\n--- Testing " << name << " recovery strategy ---" << std::endl;
            
            auto strategyConfig = training_presets::balanced();
            strategyConfig.enableProgressReporting = true;
            strategyConfig.maxIterations = 30;
            
            RobustViterbiTrainer strategyTrainer(&hmm, problematicData, strategyConfig);
            strategyTrainer.setRecoveryStrategy(strategy);
            
            std::cout << "Strategy: " << name << std::endl;
            strategyTrainer.train();
            std::cout << "Training completed successfully with " << name << " strategy" << std::endl;
        }
        
        std::cout << "\n=== RobustViterbiTrainer demonstration completed successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
