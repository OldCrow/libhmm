#include "libhmm/calculators/calculator_traits.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/scaled_forward_backward_calculator.h"
#include "libhmm/calculators/log_forward_backward_calculator.h"
#include "libhmm/calculators/optimized_forward_backward_calculator.h"
#include "libhmm/performance/thread_pool.h"
#include "libhmm/performance/simd_support.h"
#include "libhmm/hmm.h"
#include <algorithm>
#include <sstream>
#include <chrono>
#include <limits>
#include <iostream>

namespace libhmm {
namespace calculators {

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
            
        case CalculatorType::SCALED:
            return {
                "Scaled",
                false,  // supportsParallel
                false,  // usesSIMD
                false,  // usesBlocking
                sizeof(double) * 1000, // memoryOverhead (scaling factors)
                1,      // minStatesForBenefit
                100,    // minObsForBenefit (benefits longer sequences)
                0.95,   // scalingFactor (slight overhead)
                true    // numericallyStable
            };
            
        case CalculatorType::LOG_SPACE:
            return {
                "LogSpace",
                false,  // supportsParallel
                false,  // usesSIMD
                false,  // usesBlocking
                0,      // memoryOverhead
                1,      // minStatesForBenefit
                500,    // minObsForBenefit (mainly for very long sequences)
                0.85,   // scalingFactor (logarithm overhead)
                true    // numericallyStable
            };
            
        case CalculatorType::OPTIMIZED:
            return {
                "SIMD-Optimized",
                true,   // supportsParallel
                true,   // usesSIMD
                true,   // usesBlocking
                sizeof(double) * 1000, // memoryOverhead (aligned storage)
                4,      // minStatesForBenefit (SIMD benefits 4+ elements)
                10,     // minObsForBenefit
                2.5,    // scalingFactor (potential 2.5x speedup)
                false   // numericallyStable (same as standard)
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
        CalculatorType::SCALED,
        CalculatorType::LOG_SPACE,
        CalculatorType::OPTIMIZED
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

std::unique_ptr<ForwardBackwardCalculator> CalculatorSelector::create(
    CalculatorType type, Hmm* hmm, const ObservationSet& observations) {
    
    switch (type) {
        case CalculatorType::STANDARD:
            return std::make_unique<ForwardBackwardCalculator>(hmm, observations);
            
        case CalculatorType::SCALED:
            return std::make_unique<ScaledForwardBackwardCalculator>(hmm, observations);
            
        case CalculatorType::LOG_SPACE:
            return std::make_unique<LogForwardBackwardCalculator>(hmm, observations);
            
        case CalculatorType::OPTIMIZED:
            return std::make_unique<OptimizedForwardBackwardCalculator>(hmm, observations);
            
        case CalculatorType::AUTO: {
            const ProblemCharacteristics characteristics(hmm, observations);
            const CalculatorType optimalType = selectOptimal(characteristics);
            return create(optimalType, hmm, observations);
        }
        
        default:
            return std::make_unique<ForwardBackwardCalculator>(hmm, observations);
    }
}

std::unique_ptr<ForwardBackwardCalculator> CalculatorSelector::createOptimal(
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
        CalculatorType::SCALED,
        CalculatorType::LOG_SPACE,
        CalculatorType::OPTIMIZED
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
    if (!performance::simd_available()) {
        return 0.8; // Slight penalty for overhead without SIMD
    }
    
    // SIMD benefits scale with problem size
    const std::size_t problemSize = numStates * seqLength;
    
    if (problemSize < 100) {
        return 0.9; // Small problems have overhead
    } else if (problemSize < 1000) {
        return 1.2; // Modest benefit
    } else if (problemSize < 10000) {
        return 2.0; // Good benefit
    } else {
        return 2.5; // Excellent benefit for large problems
    }
}

double CalculatorSelector::calculateMemoryImpact(std::size_t overhead, double budget) noexcept {
    if (budget <= 0.0) {
        return 1.0; // No budget constraint
    }
    
    const double overheadRatio = static_cast<double>(overhead) / budget;
    
    if (overheadRatio > 0.5) {
        return 0.3; // Heavy penalty for excessive memory use
    } else if (overheadRatio > 0.2) {
        return 0.7; // Moderate penalty
    } else {
        return 1.0; // No penalty for reasonable memory use
    }
}

double CalculatorSelector::calculateStabilityNeed(std::size_t seqLength) noexcept {
    if (seqLength < 100) {
        return 0.0; // No stability concerns
    } else if (seqLength < 500) {
        return 0.2; // Minor stability concerns
    } else if (seqLength < 1000) {
        return 0.5; // Moderate stability concerns
    } else {
        return 1.0; // High stability concerns
    }
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
    return calculator_->probability();
}

Matrix AutoCalculator::getForwardVariables() const {
    return calculator_->getForwardVariables();
}

Matrix AutoCalculator::getBackwardVariables() const {
    return calculator_->getBackwardVariables();
}

//========== CalculatorBenchmark Implementation ==========

std::map<CalculatorType, double> CalculatorBenchmark::benchmarkAll(
    Hmm* hmm, const ObservationSet& observations, std::size_t iterations) {
    
    std::map<CalculatorType, double> results;
    
    const std::vector<CalculatorType> types = {
        CalculatorType::STANDARD,
        CalculatorType::SCALED,
        CalculatorType::LOG_SPACE,
        CalculatorType::OPTIMIZED
    };
    
    for (CalculatorType type : types) {
        double totalTime = 0.0;
        
        for (std::size_t i = 0; i < iterations; ++i) {
            const auto start = std::chrono::high_resolution_clock::now();
            
            try {
                auto calculator = CalculatorSelector::create(type, hmm, observations);
                volatile double prob = calculator->probability(); // Prevent optimization
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

} // namespace calculators
} // namespace libhmm
