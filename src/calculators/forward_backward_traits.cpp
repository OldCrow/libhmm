#include "libhmm/calculators/forward_backward_traits.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/log_simd_forward_backward_calculator.h"
#include "libhmm/calculators/scaled_simd_forward_backward_calculator.h"
// #include "libhmm/calculators/advanced_log_simd_forward_backward_calculator.h"  // Temporarily disabled
#include "libhmm/performance/thread_pool.h"
#include "libhmm/performance/simd_support.h"
#include "libhmm/hmm.h"
#include "libhmm/calculators/calculator_traits_utils.h"
#include <algorithm>
#include <sstream>
#include <chrono>
#include <limits>
#include <iostream>

namespace libhmm {
namespace forwardbackward {

//========== ProblemCharacteristics Implementation ==========

ProblemCharacteristics::ProblemCharacteristics(
    const Hmm* hmm, const ObservationSet& observations,
    bool requiresStability, bool isRT, double memBudget)
    : numStates(static_cast<std::size_t>(hmm->getNumStates())),
      sequenceLength(observations.size()),
      numSequences(1),
      requiresNumericalStability(requiresStability),
      isRealTime(isRT),
      memoryBudget(memBudget) {
    
    // Auto-detect stability requirement for very long sequences
    if (!requiresStability && sequenceLength > 1000) {
        requiresNumericalStability = true;
    }
    
    // Set default memory budget if not specified
    if (memoryBudget == 0.0) {
        // Estimate based on problem size (rough heuristic)
        const std::size_t basicMemory = numStates * sequenceLength * sizeof(double) * 4; // Forward + backward + working
        memoryBudget = static_cast<double>(basicMemory * 10); // Allow 10x overhead
    }
}

//========== CalculatorSelector Implementation ==========

CalculatorTraits CalculatorSelector::getTraits(CalculatorType type) noexcept {
    switch (type) {
        case CalculatorType::STANDARD:
            return {
                "Standard", 
                false,  // supportsParallel
                false,  // usesSIMD
                false,  // usesBlocking
                0,      // memoryOverhead
                1,      // minStatesForBenefit
                1,      // minObsForBenefit
                1.0,    // scalingFactor
                false   // numericallyStable (can have underflow)
            };
            
        case CalculatorType::SCALED_SIMD:
            return {
                "Scaled-SIMD",
                true,   // supportsParallel
                true,   // usesSIMD
                true,   // usesBlocking
                sizeof(double) * 1200, // memoryOverhead (aligned storage + scaling factors)
                4,      // minStatesForBenefit (SIMD benefits 4+ elements)
                50,     // minObsForBenefit (scaling benefits + SIMD setup)
                2.5,    // scalingFactor (excellent SIMD benefits for scaled arithmetic)
                true    // numericallyStable (scaled arithmetic)
            };
            
        case CalculatorType::LOG_SIMD:
            return {
                "Log-SIMD",
                true,   // supportsParallel
                true,   // usesSIMD
                true,   // usesBlocking
                sizeof(double) * 1500, // memoryOverhead (aligned + log storage)
                8,      // minStatesForBenefit (log-space SIMD benefits larger problems)
                100,    // minObsForBenefit (log-space excellent for long sequences)
                2.0,    // scalingFactor (good SIMD benefits for log-space)
                true    // numericallyStable (log-space arithmetic)
            };
            
        case CalculatorType::ADVANCED_LOG_SIMD:
            // Temporarily disabled - return standard calculator traits
            return {
                "Standard", 
                false,  // supportsParallel
                false,  // usesSIMD
                false,  // usesBlocking
                0,      // memoryOverhead
                1,      // minStatesForBenefit
                1,      // minObsForBenefit
                1.0,    // scalingFactor
                false   // numericallyStable (can have underflow)
            };
            
        case CalculatorType::AUTO:
        default:
            return {"Auto", false, false, false, 0, 1, 1, 1.0, false};
    }
}

double CalculatorSelector::predictPerformance(
    CalculatorType type, const ProblemCharacteristics& characteristics) noexcept {
    
    const CalculatorTraits traits = getTraits(type);
    double performance = traits.scalingFactor;
    
    // Check minimum requirements
    if (characteristics.numStates < traits.minStatesForBenefit ||
        characteristics.sequenceLength < traits.minObsForBenefit) {
        performance *= 0.5; // Penalty for small problems
    }
    
    // SIMD benefit calculation
    if (traits.usesSIMD) {
        const double simdBenefit = calculateSIMDBenefit(
            characteristics.numStates, characteristics.sequenceLength);
        performance *= simdBenefit;
    }
    
    // Memory overhead impact
    if (traits.memoryOverhead > 0) {
        const double memoryImpact = calculateMemoryImpact(
            traits.memoryOverhead, characteristics.memoryBudget);
        performance *= memoryImpact;
    }
    
    // Numerical stability requirements
    if (characteristics.requiresNumericalStability && !traits.numericallyStable) {
        performance *= 0.1; // Heavy penalty for unstable calculators when stability needed
    }
    
    // Real-time processing penalty for complex calculators
    if (characteristics.isRealTime && traits.memoryOverhead > 0) {
        performance *= 0.8; // Slight penalty for setup overhead
    }
    
    // Sequence length scaling
    const double stabilityNeed = calculateStabilityNeed(characteristics.sequenceLength);
    if (traits.numericallyStable) {
        performance *= (1.0 + stabilityNeed * 0.5); // Bonus for stability when needed
    }
    
    return std::max(0.1, performance); // Minimum performance floor
}

CalculatorType CalculatorSelector::selectOptimal(const ProblemCharacteristics& characteristics) noexcept {
    double bestPerformance = 0.0;
    CalculatorType bestType = CalculatorType::STANDARD;
    
    const std::vector<CalculatorType> candidates = {
        CalculatorType::STANDARD,
        CalculatorType::SCALED_SIMD,
        CalculatorType::LOG_SIMD,
        CalculatorType::ADVANCED_LOG_SIMD
    };
    
    for (CalculatorType type : candidates) {
        const double performance = predictPerformance(type, characteristics);
        if (performance > bestPerformance) {
            bestPerformance = performance;
            bestType = type;
        }
    }
    
    return bestType;
}

std::unique_ptr<Calculator> CalculatorSelector::create(
    CalculatorType type, Hmm* hmm, const ObservationSet& observations) {
    
    switch (type) {
        case CalculatorType::STANDARD:
            return std::make_unique<ForwardBackwardCalculator>(hmm, observations);
            
        case CalculatorType::LOG_SIMD:
        {
            auto calc = std::make_unique<LogSIMDForwardBackwardCalculator>(hmm, observations);
            calc->compute(); // SIMD calculators need explicit computation
            return calc;
        }
            
        case CalculatorType::SCALED_SIMD:
        {
            auto calc = std::make_unique<ScaledSIMDForwardBackwardCalculator>(hmm, observations);
            calc->compute(); // SIMD calculators need explicit computation
            return calc;
        }
            
        case CalculatorType::ADVANCED_LOG_SIMD:
        {
            // Temporarily disabled - return standard calculator instead
            auto calc = std::make_unique<ForwardBackwardCalculator>(hmm, observations);
            return calc;
        }
            
        case CalculatorType::AUTO:
        {
            const ProblemCharacteristics characteristics(hmm, observations);
            const CalculatorType optimalType = selectOptimal(characteristics);
            return create(optimalType, hmm, observations);
        }
        
        default:
            return std::make_unique<ForwardBackwardCalculator>(hmm, observations);
    }
}

std::unique_ptr<Calculator> CalculatorSelector::createOptimal(
    Hmm* hmm, const ObservationSet& observations,
    bool requiresStability, bool isRealTime, double memoryBudget) {
    
    const ProblemCharacteristics characteristics(
        hmm, observations, requiresStability, isRealTime, memoryBudget);
    const CalculatorType optimalType = selectOptimal(characteristics);
    return create(optimalType, hmm, observations);
}

std::string CalculatorSelector::getPerformanceComparison(const ProblemCharacteristics& characteristics) {
    std::ostringstream oss;
    oss << "Calculator Performance Comparison:\n";
    oss << "Problem: " << characteristics.numStates << " states, " 
        << characteristics.sequenceLength << " observations\n";
    oss << "Stability required: " << (characteristics.requiresNumericalStability ? "Yes" : "No") << "\n";
    oss << "Real-time: " << (characteristics.isRealTime ? "Yes" : "No") << "\n\n";
    
    const std::vector<CalculatorType> types = {
        CalculatorType::STANDARD,
        CalculatorType::LOG_SIMD,
        CalculatorType::SCALED_SIMD,
        CalculatorType::ADVANCED_LOG_SIMD
    };
    
    CalculatorType bestType = CalculatorType::STANDARD;
    double bestPerformance = 0.0;
    
    for (CalculatorType type : types) {
        const CalculatorTraits traits = getTraits(type);
        const double performance = predictPerformance(type, characteristics);
        
        oss << traits.name << ": " << performance << "x";
        if (performance > bestPerformance) {
            bestPerformance = performance;
            bestType = type;
            oss << " (BEST)";
        }
        oss << "\n";
    }
    
    oss << "\nSelected: " << getTraits(bestType).name;
    return oss.str();
}

//========== Private Helper Methods ==========

double CalculatorSelector::calculateSIMDBenefit(std::size_t numStates, std::size_t seqLength) noexcept {
    return calculator_traits::calculateSIMDBenefit(numStates, seqLength);
}

double CalculatorSelector::calculateMemoryImpact(std::size_t overhead, double budget) noexcept {
    return calculator_traits::calculateMemoryImpact(overhead, budget);
}

double CalculatorSelector::calculateStabilityNeed(std::size_t seqLength) noexcept {
    return calculator_traits::calculateStabilityNeed(seqLength);
}

//========== AutoCalculator Implementation ==========

AutoCalculator::AutoCalculator(Hmm* hmm, const ObservationSet& observations,
                               bool requiresStability, bool isRealTime, double memoryBudget)
    : characteristics_(hmm, observations, requiresStability, isRealTime, memoryBudget) {
    
    selectedType_ = CalculatorSelector::selectOptimal(characteristics_);
    calculator_ = CalculatorSelector::create(selectedType_, hmm, observations);
}

std::string AutoCalculator::getSelectionRationale() const {
    const CalculatorTraits traits = CalculatorSelector::getTraits(selectedType_);
    std::ostringstream oss;
    
    oss << "Selected " << traits.name << " calculator because:\n";
    
    if (characteristics_.requiresNumericalStability && traits.numericallyStable) {
        oss << "- Provides numerical stability for long sequences\n";
    }
    
    if (traits.usesSIMD && characteristics_.numStates >= traits.minStatesForBenefit) {
        oss << "- SIMD optimizations benefit this problem size\n";
    }
    
    if (characteristics_.isRealTime && traits.memoryOverhead == 0) {
        oss << "- Low memory overhead suitable for real-time processing\n";
    }
    
    const double performance = CalculatorSelector::predictPerformance(selectedType_, characteristics_);
    oss << "- Predicted performance: " << performance << "x baseline";
    
    return oss.str();
}

double AutoCalculator::probability() {
    // We need to cast to appropriate Forward-Backward calculator type to call probability()
    // This follows the same pattern as the Viterbi traits system
    if (selectedType_ == CalculatorType::STANDARD) {
        if (auto fb = dynamic_cast<ForwardBackwardCalculator*>(calculator_.get())) {
            return fb->probability();
        }
    } else if (selectedType_ == CalculatorType::SCALED_SIMD) {
        if (auto scaled = dynamic_cast<ScaledSIMDForwardBackwardCalculator*>(calculator_.get())) {
            return scaled->probability();
        }
    } else if (selectedType_ == CalculatorType::LOG_SIMD) {
        if (auto log = dynamic_cast<LogSIMDForwardBackwardCalculator*>(calculator_.get())) {
            return log->probability();
        }
    } else if (selectedType_ == CalculatorType::ADVANCED_LOG_SIMD) {
        // Temporarily disabled - fallback to standard calculator
        if (auto fb = dynamic_cast<ForwardBackwardCalculator*>(calculator_.get())) {
            return fb->probability();
        }
    }
    throw std::runtime_error("Unable to compute probability with selected calculator type");
}

double AutoCalculator::getLogProbability() {
    // Get log probability using selected calculator with numerical stability
    if (selectedType_ == CalculatorType::STANDARD) {
        if (auto fb = dynamic_cast<ForwardBackwardCalculator*>(calculator_.get())) {
            // Standard calculator returns raw probability - take log carefully
            double prob = fb->probability();
            if (prob <= 0.0) {
                return -std::numeric_limits<double>::infinity();
            }
            return std::log(prob);
        }
    } else if (selectedType_ == CalculatorType::SCALED_SIMD) {
        if (auto scaled = dynamic_cast<ScaledSIMDForwardBackwardCalculator*>(calculator_.get())) {
            // Scaled calculator has getLogProbability() method for numerical stability
            return scaled->getLogProbability();
        }
    } else if (selectedType_ == CalculatorType::LOG_SIMD) {
        if (auto log = dynamic_cast<LogSIMDForwardBackwardCalculator*>(calculator_.get())) {
            // Log calculator has getLogProbability() method for numerical stability
            return log->getLogProbability();
        }
    } else if (selectedType_ == CalculatorType::ADVANCED_LOG_SIMD) {
        // Temporarily disabled - fallback to standard calculator
        if (auto fb = dynamic_cast<ForwardBackwardCalculator*>(calculator_.get())) {
            double prob = fb->probability();
            if (prob <= 0.0) {
                return -std::numeric_limits<double>::infinity();
            }
            return std::log(prob);
        }
    }
    throw std::runtime_error("Unable to compute log probability with selected calculator type");
}

Matrix AutoCalculator::getForwardVariables() const {
    // Use dynamic casting to call getForwardVariables() based on calculator type
    if (selectedType_ == CalculatorType::STANDARD) {
        if (auto fb = dynamic_cast<ForwardBackwardCalculator*>(calculator_.get())) {
            return fb->getForwardVariables();
        }
    } else if (selectedType_ == CalculatorType::SCALED_SIMD) {
        if (auto scaled = dynamic_cast<ScaledSIMDForwardBackwardCalculator*>(calculator_.get())) {
            return scaled->getForwardVariablesCompat();
        }
    } else if (selectedType_ == CalculatorType::LOG_SIMD) {
        if (auto log = dynamic_cast<LogSIMDForwardBackwardCalculator*>(calculator_.get())) {
            return log->getForwardVariablesCompat();
        }
    } else if (selectedType_ == CalculatorType::ADVANCED_LOG_SIMD) {
        // Temporarily disabled - fallback to standard calculator
        if (auto fb = dynamic_cast<ForwardBackwardCalculator*>(calculator_.get())) {
            return fb->getForwardVariables();
        }
    }
    throw std::runtime_error("Unable to get forward variables with selected calculator type");
}

Matrix AutoCalculator::getBackwardVariables() const {
    // Use dynamic casting to call getBackwardVariables() based on calculator type
    if (selectedType_ == CalculatorType::STANDARD) {
        if (auto fb = dynamic_cast<ForwardBackwardCalculator*>(calculator_.get())) {
            return fb->getBackwardVariables();
        }
    } else if (selectedType_ == CalculatorType::SCALED_SIMD) {
        if (auto scaled = dynamic_cast<ScaledSIMDForwardBackwardCalculator*>(calculator_.get())) {
            return scaled->getBackwardVariablesCompat();
        }
    } else if (selectedType_ == CalculatorType::LOG_SIMD) {
        if (auto log = dynamic_cast<LogSIMDForwardBackwardCalculator*>(calculator_.get())) {
            return log->getBackwardVariablesCompat();
        }
    } else if (selectedType_ == CalculatorType::ADVANCED_LOG_SIMD) {
        // Temporarily disabled - fallback to standard calculator
        if (auto fb = dynamic_cast<ForwardBackwardCalculator*>(calculator_.get())) {
            return fb->getBackwardVariables();
        }
    }
    throw std::runtime_error("Unable to get backward variables with selected calculator type");
}

//========== CalculatorBenchmark Implementation ==========

std::map<CalculatorType, double> CalculatorBenchmark::benchmarkAll(
    Hmm* hmm, const ObservationSet& observations, std::size_t iterations) {
    
    std::map<CalculatorType, double> results;
    
    const std::vector<CalculatorType> types = {
        CalculatorType::STANDARD,
        CalculatorType::SCALED_SIMD,
        CalculatorType::LOG_SIMD,
        CalculatorType::ADVANCED_LOG_SIMD
    };
    
    for (CalculatorType type : types) {
        double totalTime = 0.0;
        
        for (std::size_t i = 0; i < iterations; ++i) {
            const auto start = std::chrono::high_resolution_clock::now();
            
            try {
                auto calculator = CalculatorSelector::create(type, hmm, observations);
                
                // Use dynamic casting to call probability() based on calculator type
                volatile double prob = 0.0;
                if (type == CalculatorType::STANDARD) {
                    if (auto fb = dynamic_cast<ForwardBackwardCalculator*>(calculator.get())) {
                        prob = fb->probability();
                    }
                } else if (type == CalculatorType::SCALED_SIMD) {
                    if (auto scaled = dynamic_cast<ScaledSIMDForwardBackwardCalculator*>(calculator.get())) {
                        prob = scaled->probability();
                    }
                } else if (type == CalculatorType::LOG_SIMD) {
                    if (auto log = dynamic_cast<LogSIMDForwardBackwardCalculator*>(calculator.get())) {
                        prob = log->probability();
                    }
                } else if (type == CalculatorType::ADVANCED_LOG_SIMD) {
                    // Temporarily disabled - fallback to standard calculator
                    if (auto fb = dynamic_cast<ForwardBackwardCalculator*>(calculator.get())) {
                        prob = fb->probability();
                    }
                }
                (void)prob;
            } catch (const std::exception&) {
                // Skip failed calculators
                totalTime = std::numeric_limits<double>::max();
                break;
            }
            
            const auto end = std::chrono::high_resolution_clock::now();
            const auto duration = std::chrono::duration<double>(end - start);
            totalTime += duration.count();
        }
        
        results[type] = totalTime / static_cast<double>(iterations);
    }
    
    // Convert to relative performance (inverse of time, normalized to standard)
    const double standardTime = results[CalculatorType::STANDARD];
    if (standardTime > 0.0) {
        for (auto& pair : results) {
            pair.second = standardTime / pair.second; // Higher is better
        }
    }
    
    return results;
}

void CalculatorBenchmark::updatePerformanceModel(
    const ProblemCharacteristics& characteristics,
    const std::map<CalculatorType, double>& results) {
    // This would update machine learning models in a full implementation
    // For now, we just log the results
    std::cout << "Benchmark results for " << characteristics.numStates 
              << " states, " << characteristics.sequenceLength << " observations:\n";
    
    for (const auto& pair : results) {
        const CalculatorTraits traits = CalculatorSelector::getTraits(pair.first);
        std::cout << "  " << traits.name << ": " << pair.second << "x\n";
    }
}

bool CalculatorBenchmark::validateSelection(Hmm* hmm, const ObservationSet& observations) {
    const ProblemCharacteristics characteristics(hmm, observations);
    const CalculatorType predicted = CalculatorSelector::selectOptimal(characteristics);
    
    const auto results = benchmarkAll(hmm, observations, 5);
    
    // Find actual best performer
    CalculatorType actualBest = CalculatorType::STANDARD;
    double bestPerformance = 0.0;
    
    for (const auto& pair : results) {
        if (pair.second > bestPerformance) {
            bestPerformance = pair.second;
            actualBest = pair.first;
        }
    }
    
    return predicted == actualBest;
}

} // namespace forwardbackward
} // namespace libhmm
