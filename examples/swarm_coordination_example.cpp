/**
 * @file swarm_coordination_example.cpp
 * @brief Discrete State Swarm Coordination Example using libhmm
 * 
 * This example demonstrates how to use Hidden Markov Models for coordinating
 * a drone swarm through different formation states and mission phases.
 * 
 * Key Features:
 * - Discrete state space modeling (formation types, mission phases)
 * - Multi-dimensional discrete observations (altitude, speed, threats)
 * - Automatic calculator selection with SIMD optimization
 * - Real-time state prediction and formation coordination
 * - Fault detection and recovery mechanisms
 * 
 * Applications:
 * - Autonomous drone swarm coordination
 * - Multi-robot formation control
 * - Mission state management
 * - System health monitoring
 * 
 * @author libhmm development team
 * @version 2.5.0
 */

#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <iomanip>
#include <chrono>
#include <map>
#include <string>

#include "libhmm/libhmm.h"

/**
 * @brief Swarm formation states for coordinated flight
 */
enum class FormationState {
    STANDBY = 0,        // Ground standby, engines off
    TAKEOFF = 1,        // Individual takeoff sequence
    CRUISE = 2,         // Loose formation, transit
    FORMATION_TIGHT = 3,// Tight formation flying
    SEARCH_PATTERN = 4, // Distributed search deployment
    EVASIVE = 5,        // Threat response, evasive maneuvers
    RETURN_TO_BASE = 6, // Mission completion, return
    EMERGENCY = 7       // System failure, emergency procedures
};

/**
 * @brief Multi-dimensional discrete observations for swarm state
 */
struct SwarmObservation {
    int altitude_zone;      // 0=low(<100m), 1=medium(100-500m), 2=high(>500m)
    int speed_category;     // 0=slow(<10m/s), 1=medium(10-25m/s), 2=fast(>25m/s)
    int threat_level;       // 0=none, 1=low, 2=medium, 3=high
    int formation_quality;  // 0=poor, 1=acceptable, 2=good, 3=excellent
    int communication;      // 0=poor, 1=degraded, 2=good, 3=excellent
    
    /**
     * @brief Convert multi-dimensional observation to single discrete value
     * @return Combined observation value for HMM
     */
    unsigned int toDiscreteValue() const {
        // Pack multiple discrete observations into single value
        // This is a simple encoding - could use more sophisticated methods
        return altitude_zone * 81 + speed_category * 27 + threat_level * 9 + 
               formation_quality * 3 + communication;
    }
    
    /**
     * @brief Get maximum possible observation value
     * @return Maximum discrete observation value
     */
    static unsigned int getMaxValue() {
        return 3 * 81 + 3 * 27 + 4 * 9 + 4 * 3 + 4; // 243 possible combinations
    }
};

/**
 * @brief Drone Swarm Coordinator using HMM for state management
 */
class DroneSwarmController {
private:
    std::unique_ptr<libhmm::Hmm> formationHMM_;
    std::mt19937 randomGen_;
    std::vector<FormationState> stateHistory_;
    std::vector<SwarmObservation> observationHistory_;
    
    const std::map<FormationState, std::string> stateNames_ = {
        {FormationState::STANDBY, "STANDBY"},
        {FormationState::TAKEOFF, "TAKEOFF"},
        {FormationState::CRUISE, "CRUISE"},
        {FormationState::FORMATION_TIGHT, "FORMATION_TIGHT"},
        {FormationState::SEARCH_PATTERN, "SEARCH_PATTERN"},
        {FormationState::EVASIVE, "EVASIVE"},
        {FormationState::RETURN_TO_BASE, "RETURN_TO_BASE"},
        {FormationState::EMERGENCY, "EMERGENCY"}
    };

public:
    /**
     * @brief Constructor - sets up HMM for swarm coordination
     */
    DroneSwarmController() : randomGen_(std::chrono::steady_clock::now().time_since_epoch().count()) {
        // Create HMM with 8 formation states
        formationHMM_ = std::make_unique<libhmm::Hmm>(8);
        
        setupTransitionMatrix();
        setupObservationModels();
        setupInitialProbabilities();
        
        std::cout << "🚁 Drone Swarm Controller initialized with 8 formation states\n";
        std::cout << "📊 Using discrete observation space with " << SwarmObservation::getMaxValue() 
                  << " possible observation combinations\n\n";
    }
    
    /**
     * @brief Set up state transition probabilities
     */
    void setupTransitionMatrix() {
        libhmm::Matrix transitions(8, 8);
        
        // Initialize with small probabilities
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                transitions(i, j) = 0.01; // Small baseline probability
            }
        }
        
        // Define logical state transitions with higher probabilities
        
        // STANDBY transitions
        transitions(0, 0) = 0.6;  // Stay in standby
        transitions(0, 1) = 0.35; // Go to takeoff
        transitions(0, 7) = 0.04; // Emergency
        
        // TAKEOFF transitions  
        transitions(1, 1) = 0.3;  // Continue takeoff
        transitions(1, 2) = 0.65; // Go to cruise
        transitions(1, 7) = 0.04; // Emergency
        
        // CRUISE transitions
        transitions(2, 2) = 0.4;  // Stay in cruise
        transitions(2, 3) = 0.25; // Go to tight formation
        transitions(2, 4) = 0.25; // Go to search pattern
        transitions(2, 6) = 0.05; // Return to base
        transitions(2, 5) = 0.02; // Evasive
        transitions(2, 7) = 0.03; // Emergency
        
        // FORMATION_TIGHT transitions
        transitions(3, 3) = 0.5;  // Stay in tight formation
        transitions(3, 2) = 0.2;  // Back to cruise
        transitions(3, 4) = 0.15; // Go to search
        transitions(3, 5) = 0.08; // Evasive maneuvers
        transitions(3, 6) = 0.04; // Return to base
        transitions(3, 7) = 0.03; // Emergency
        
        // SEARCH_PATTERN transitions
        transitions(4, 4) = 0.45; // Continue search
        transitions(4, 3) = 0.15; // Reform tight formation
        transitions(4, 2) = 0.2;  // Back to cruise
        transitions(4, 5) = 0.05; // Evasive
        transitions(4, 6) = 0.12; // Mission complete, RTB
        transitions(4, 7) = 0.03; // Emergency
        
        // EVASIVE transitions
        transitions(5, 5) = 0.3;  // Continue evasive
        transitions(5, 2) = 0.4;  // Back to cruise (threat passed)
        transitions(5, 3) = 0.15; // Reform formation
        transitions(5, 6) = 0.1;  // Abort mission, RTB
        transitions(5, 7) = 0.05; // Emergency
        
        // RETURN_TO_BASE transitions
        transitions(6, 6) = 0.8;  // Continue RTB
        transitions(6, 0) = 0.15; // Land and standby
        transitions(6, 7) = 0.05; // Emergency
        
        // EMERGENCY transitions
        transitions(7, 7) = 0.6;  // Stay in emergency
        transitions(7, 6) = 0.3;  // Emergency RTB
        transitions(7, 0) = 0.1;  // Emergency landing
        
        formationHMM_->setTrans(transitions);
        
        std::cout << "✅ Transition matrix configured with mission-logical probabilities\n";
    }
    
    /**
     * @brief Set up observation probability distributions for each state
     */
    void setupObservationModels() {
        const int numObservations = SwarmObservation::getMaxValue();
        
        for (int state = 0; state < 8; ++state) {
            auto distribution = std::make_unique<libhmm::DiscreteDistribution>(numObservations);
            
            // Set probabilities based on expected observations for each state
            setupStateObservationProbabilities(static_cast<FormationState>(state), *distribution);
            
            formationHMM_->setProbabilityDistribution(state, std::move(distribution));
        }
        
        std::cout << "✅ Observation models configured for all " << numObservations << " observation combinations\n";
    }
    
    /**
     * @brief Configure observation probabilities for a specific state
     */
    void setupStateObservationProbabilities(FormationState state, libhmm::DiscreteDistribution& dist) {
        const int numObs = SwarmObservation::getMaxValue();
        
        // Initialize with uniform low probability
        for (int i = 0; i < numObs; ++i) {
            dist.setProbability(i, 1.0 / numObs);
        }
        
        // Boost probabilities for observations typical of this state
        std::vector<SwarmObservation> typicalObservations = getTypicalObservations(state);
        
        double boost = 10.0; // Increase probability by this factor
        double totalBoost = 0.0;
        
        for (const auto& obs : typicalObservations) {
            unsigned int obsValue = obs.toDiscreteValue();
            if (obsValue < static_cast<unsigned int>(numObs)) {
                dist.setProbability(obsValue, dist.getProbability(obsValue) * boost);
                totalBoost += dist.getProbability(obsValue) * (boost - 1);
            }
        }
        
        // Normalize to ensure probabilities sum to 1
        double sum = 0.0;
        for (int i = 0; i < numObs; ++i) {
            sum += dist.getProbability(i);
        }
        
        for (int i = 0; i < numObs; ++i) {
            dist.setProbability(i, dist.getProbability(i) / sum);
        }
    }
    
    /**
     * @brief Get typical observations for each formation state
     */
    std::vector<SwarmObservation> getTypicalObservations(FormationState state) {
        std::vector<SwarmObservation> observations;
        
        switch (state) {
            case FormationState::STANDBY:
                observations.push_back({0, 0, 0, 0, 2}); // Low alt, no speed, no threat, poor formation, good comms
                observations.push_back({0, 0, 0, 0, 3}); // Variant with excellent comms
                break;
                
            case FormationState::TAKEOFF:
                observations.push_back({0, 1, 0, 1, 2}); // Low->medium alt, medium speed, acceptable formation
                observations.push_back({1, 1, 0, 2, 2}); // Medium alt, good formation developing
                break;
                
            case FormationState::CRUISE:
                observations.push_back({1, 1, 0, 2, 3}); // Medium alt, medium speed, good formation
                observations.push_back({2, 1, 0, 2, 3}); // High alt variant
                observations.push_back({1, 2, 0, 1, 2}); // Faster variant, looser formation
                break;
                
            case FormationState::FORMATION_TIGHT:
                observations.push_back({1, 1, 0, 3, 3}); // Medium alt, excellent formation & comms
                observations.push_back({2, 1, 0, 3, 3}); // High alt variant
                observations.push_back({1, 2, 0, 3, 3}); // Fast variant
                break;
                
            case FormationState::SEARCH_PATTERN:
                observations.push_back({1, 1, 0, 1, 2}); // Spread out, acceptable formation
                observations.push_back({2, 1, 0, 0, 2}); // High alt, very spread out
                observations.push_back({1, 2, 0, 1, 1}); // Fast search, degraded comms
                break;
                
            case FormationState::EVASIVE:
                observations.push_back({1, 2, 2, 0, 1}); // Fast, medium threat, poor formation
                observations.push_back({2, 2, 3, 0, 1}); // High speed/threat, poor formation
                observations.push_back({0, 2, 2, 0, 2}); // Low alt evasion
                break;
                
            case FormationState::RETURN_TO_BASE:
                observations.push_back({1, 1, 0, 2, 3}); // Orderly return, good formation
                observations.push_back({1, 2, 1, 1, 2}); // Faster return, some threat
                observations.push_back({2, 1, 0, 3, 3}); // High alt return
                break;
                
            case FormationState::EMERGENCY:
                observations.push_back({0, 0, 3, 0, 0}); // Emergency landing, high threat, poor everything
                observations.push_back({1, 2, 2, 0, 1}); // Emergency escape, fast, medium threat
                observations.push_back({2, 1, 1, 0, 1}); // High alt emergency
                break;
        }
        
        return observations;
    }
    
    /**
     * @brief Set up initial state probabilities
     */
    void setupInitialProbabilities() {
        libhmm::Vector initialProbs(8);
        
        // Most missions start in STANDBY
        initialProbs(0) = 0.7;  // STANDBY
        initialProbs(1) = 0.1;  // TAKEOFF  
        initialProbs(2) = 0.15; // CRUISE (ongoing mission)
        initialProbs(3) = 0.02; // FORMATION_TIGHT
        initialProbs(4) = 0.02; // SEARCH_PATTERN
        initialProbs(5) = 0.005;// EVASIVE
        initialProbs(6) = 0.01; // RETURN_TO_BASE
        initialProbs(7) = 0.005;// EMERGENCY
        
        formationHMM_->setPi(initialProbs);
        
        std::cout << "✅ Initial state probabilities set (70% start in STANDBY)\n";
    }
    
    /**
     * @brief Simulate swarm observations based on current conditions
     */
    SwarmObservation simulateObservation(FormationState actualState, double noiseLevel = 0.1) {
        SwarmObservation obs;
        
        // Base observations on actual state with some noise
        auto typical = getTypicalObservations(actualState);
        if (!typical.empty()) {
            obs = typical[randomGen_() % typical.size()];
            
            // Add noise
            if (std::uniform_real_distribution<>(0.0, 1.0)(randomGen_) < noiseLevel) {
                // Randomly perturb one observation dimension
                int dimension = randomGen_() % 5;
                switch (dimension) {
                    case 0: obs.altitude_zone = std::min(2, std::max(0, obs.altitude_zone + static_cast<int>(randomGen_() % 3) - 1)); break;
                    case 1: obs.speed_category = std::min(2, std::max(0, obs.speed_category + static_cast<int>(randomGen_() % 3) - 1)); break;
                    case 2: obs.threat_level = std::min(3, std::max(0, obs.threat_level + static_cast<int>(randomGen_() % 3) - 1)); break;
                    case 3: obs.formation_quality = std::min(3, std::max(0, obs.formation_quality + static_cast<int>(randomGen_() % 3) - 1)); break;
                    case 4: obs.communication = std::min(3, std::max(0, obs.communication + static_cast<int>(randomGen_() % 3) - 1)); break;
                }
            }
        } else {
            // Fallback random observation
            obs = {
                static_cast<int>(randomGen_() % 3), 
                static_cast<int>(randomGen_() % 3), 
                static_cast<int>(randomGen_() % 4), 
                static_cast<int>(randomGen_() % 4), 
                static_cast<int>(randomGen_() % 4)
            };
        }
        
        return obs;
    }
    
    /**
     * @brief Predict next formation state based on observation history
     */
    FormationState predictNextFormation(const std::vector<SwarmObservation>& observations) {
        if (observations.empty()) {
            return FormationState::STANDBY;
        }
        
        // Convert observations to discrete values
        std::vector<unsigned int> obsValues;
        for (const auto& obs : observations) {
            obsValues.push_back(obs.toDiscreteValue());
        }
        
        // Create observation set for libhmm
        libhmm::ObservationSet obsSet(obsValues.size());
        for (size_t i = 0; i < obsValues.size(); ++i) {
            obsSet(i) = obsValues[i];
        }
        
        // Use Viterbi algorithm to find most likely state sequence
        libhmm::viterbi::AutoCalculator viterbi(formationHMM_.get(), obsSet);
        auto stateSequence = viterbi.decode();
        
        if (stateSequence.size() > 0) {
            return static_cast<FormationState>(stateSequence(stateSequence.size() - 1));
        }
        
        return FormationState::STANDBY;
    }
    
    /**
     * @brief Calculate confidence in current state prediction
     */
    double calculateStateConfidence(const std::vector<SwarmObservation>& observations) {
        if (observations.empty()) {
            return 0.0;
        }
        
        // Convert observations
        libhmm::ObservationSet obsSet(observations.size());
        for (size_t i = 0; i < observations.size(); ++i) {
            obsSet(i) = observations[i].toDiscreteValue();
        }
        
        // Use Forward-Backward to get probability
        libhmm::forwardbackward::AutoCalculator fb(formationHMM_.get(), obsSet);
        double probability = fb.probability();
        
        // Convert to confidence score (0-100%)
        return std::min(100.0, std::max(0.0, probability * 1000)); // Scale for display
    }
    
    /**
     * @brief Run a complete mission simulation
     */
    void runMissionSimulation() {
        std::cout << "🚀 Starting Mission Simulation\n";
        std::cout << "=" << std::string(80, '=') << "\n\n";
        
        // Simulate mission phases
        std::vector<FormationState> missionPhases = {
            FormationState::STANDBY,
            FormationState::TAKEOFF,
            FormationState::CRUISE,
            FormationState::FORMATION_TIGHT,
            FormationState::SEARCH_PATTERN,
            FormationState::FORMATION_TIGHT,
            FormationState::RETURN_TO_BASE,
            FormationState::STANDBY
        };
        
        std::vector<SwarmObservation> observationWindow;
        const size_t windowSize = 8; // Use last 8 observations for prediction
        
        const size_t stepsPerPhase = 8; // Changed from 3 to 8 steps per phase
        
        for (size_t step = 0; step < missionPhases.size() * stepsPerPhase; ++step) {
            // Current actual state (with some state persistence)
            FormationState actualState = missionPhases[std::min(step / stepsPerPhase, missionPhases.size() - 1)];
            
            // Simulate observation
            SwarmObservation obs = simulateObservation(actualState, 0.15);
            observationWindow.push_back(obs);
            
            // Keep window size manageable
            if (observationWindow.size() > windowSize) {
                observationWindow.erase(observationWindow.begin());
            }
            
            // Predict state
            FormationState predictedState = predictNextFormation(observationWindow);
            double confidence = calculateStateConfidence(observationWindow);
            
            // Display results
            std::cout << "Step " << std::setw(2) << step + 1 << ": ";
            std::cout << "Actual: " << std::setw(15) << stateNames_.at(actualState);
            std::cout << " | Predicted: " << std::setw(15) << stateNames_.at(predictedState);
            std::cout << " | Confidence: " << std::setw(6) << std::fixed << std::setprecision(1) << confidence << "%";
            
            // Show observation details
            std::cout << " | Obs: [Alt:" << obs.altitude_zone << " Spd:" << obs.speed_category 
                      << " Thr:" << obs.threat_level << " Fmt:" << obs.formation_quality 
                      << " Com:" << obs.communication << "]";
            
            if (actualState == predictedState) {
                std::cout << " ✅";
            } else {
                std::cout << " ❌";
            }
            std::cout << "\n";
            
            // Store history
            stateHistory_.push_back(actualState);
            observationHistory_.push_back(obs);
        }
        
        // Calculate overall accuracy
        size_t correct = 0;
        for (size_t i = windowSize; i < stateHistory_.size(); ++i) {
            std::vector<SwarmObservation> window(
                observationHistory_.begin() + i - windowSize + 1,
                observationHistory_.begin() + i + 1
            );
            FormationState predicted = predictNextFormation(window);
            if (predicted == stateHistory_[i]) {
                correct++;
            }
        }
        
        double accuracy = 100.0 * correct / (stateHistory_.size() - windowSize + 1);
        
        std::cout << "\n📊 Mission Simulation Results:\n";
        std::cout << "Total Steps: " << stateHistory_.size() << "\n";
        std::cout << "Prediction Accuracy: " << std::fixed << std::setprecision(1) << accuracy << "%\n";
        std::cout << "Window Size: " << windowSize << " observations\n";
    }
    
    /**
     * @brief Demonstrate real-time state monitoring
     */
    void demonstrateRealTimeMonitoring() {
        std::cout << "\n🔄 Real-Time State Monitoring Demo\n";
        std::cout << "=" << std::string(50, '=') << "\n\n";
        
        std::vector<SwarmObservation> realtimeWindow;
        
        // Simulate real-time observations
        std::vector<FormationState> scenario = {
            FormationState::CRUISE,
            FormationState::CRUISE,
            FormationState::EVASIVE,      // Threat detected!
            FormationState::EVASIVE,
            FormationState::FORMATION_TIGHT, // Regroup
            FormationState::RETURN_TO_BASE   // Mission abort
        };
        
        for (size_t i = 0; i < scenario.size(); ++i) {
            SwarmObservation obs = simulateObservation(scenario[i], 0.1);
            realtimeWindow.push_back(obs);
            
            FormationState predicted = predictNextFormation(realtimeWindow);
            double confidence = calculateStateConfidence(realtimeWindow);
            
            std::cout << "Time T+" << std::setw(2) << i + 1 << ": ";
            std::cout << stateNames_.at(predicted) << " (";
            std::cout << std::fixed << std::setprecision(1) << confidence << "% confidence)";
            
            if (scenario[i] == FormationState::EVASIVE && i == 2) {
                std::cout << " ⚠️ THREAT DETECTED!";
            }
            std::cout << "\n";
        }
    }
    
    /**
     * @brief Get system status and diagnostics
     */
    void printSystemDiagnostics() {
        std::cout << "\n🔧 System Diagnostics\n";
        std::cout << "=" << std::string(30, '=') << "\n";
        std::cout << "HMM States: " << formationHMM_->getNumStates() << "\n";
        std::cout << "Observation Space: " << SwarmObservation::getMaxValue() << " combinations\n";
        std::cout << "State History: " << stateHistory_.size() << " entries\n";
        std::cout << "Observation History: " << observationHistory_.size() << " entries\n";
        std::cout << "Calculator: AutoCalculator with SIMD optimization\n";
    }
};

/**
 * @brief Main function demonstrating swarm coordination
 */
int main() {
    std::cout << "🚁 libhmm Discrete State Swarm Coordination Example\n";
    std::cout << "==================================================\n\n";
    
    std::cout << "This example demonstrates:\n";
    std::cout << "✓ Multi-state swarm formation control\n";
    std::cout << "✓ Discrete observation space modeling\n";  
    std::cout << "✓ Real-time state prediction\n";
    std::cout << "✓ Mission phase coordination\n";
    std::cout << "✓ Automatic SIMD optimization\n\n";
    
    try {
        // Create swarm controller
        DroneSwarmController controller;
        
        // Run mission simulation
        controller.runMissionSimulation();
        
        // Demonstrate real-time monitoring
        controller.demonstrateRealTimeMonitoring();
        
        // Show system diagnostics
        controller.printSystemDiagnostics();
        
        std::cout << "\n✅ Swarm coordination example completed successfully!\n";
        std::cout << "\n💡 Key Takeaways:\n";
        std::cout << "• HMMs excel at discrete state coordination\n";
        std::cout << "• Multi-dimensional observations can be encoded effectively\n";
        std::cout << "• Real-time prediction enables proactive coordination\n";
        std::cout << "• SIMD optimization provides performance benefits\n";
        std::cout << "• Current libhmm capabilities are well-suited for swarm applications\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
