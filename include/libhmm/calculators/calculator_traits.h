#ifndef LIBHMM_CALCULATOR_TRAITS_H_
#define LIBHMM_CALCULATOR_TRAITS_H_

#include <memory>
#include <string>
#include <cstddef>
#include <map>
#include "libhmm/common/common.h"

namespace libhmm {

// Forward declarations
class Hmm;
class ForwardBackwardCalculator;

namespace calculators {

/// Calculator performance characteristics
struct CalculatorTraits {
    std::string name;                    ///< Calculator name
    bool supportsParallel = false;       ///< Supports parallel processing
    bool usesSIMD = false;               ///< Uses SIMD optimizations
    bool usesBlocking = false;           ///< Uses cache-blocking
    std::size_t memoryOverhead = 0;      ///< Additional memory overhead (bytes)
    std::size_t minStatesForBenefit = 1; ///< Minimum states to show benefit
    std::size_t minObsForBenefit = 1;    ///< Minimum observations to show benefit
    double scalingFactor = 1.0;          ///< Expected performance scaling
    bool numericallyStable = true;       ///< Numerically stable for long sequences
};

/// Calculator type enumeration
enum class CalculatorType {
    STANDARD,           ///< Standard forward-backward calculator
    SCALED,            ///< Scaled forward-backward calculator  
    LOG_SPACE,         ///< Log-space forward-backward calculator
    OPTIMIZED,         ///< SIMD-optimized calculator
    AUTO               ///< Automatic selection based on problem characteristics
};

/// HMM problem characteristics for calculator selection
struct ProblemCharacteristics {
    std::size_t numStates;           ///< Number of HMM states
    std::size_t sequenceLength;      ///< Length of observation sequence
    std::size_t numSequences;        ///< Number of observation sequences (for training)
    bool requiresNumericalStability; ///< Long sequences requiring stability
    bool isRealTime;                 ///< Real-time processing requirement
    double memoryBudget;             ///< Available memory budget (bytes)
    
    /// Constructor from HMM and observations
    /// @param hmm HMM to analyze
    /// @param observations Observation sequence
    /// @param requiresStability Force numerical stability
    /// @param isRT Real-time processing flag
    /// @param memBudget Available memory in bytes
    ProblemCharacteristics(const Hmm* hmm, const ObservationSet& observations,
                          bool requiresStability = false, bool isRT = false,
                          double memBudget = 0.0);
};

/// Calculator performance predictor and selector
class CalculatorSelector {
public:
    /// Get traits for a specific calculator type
    /// @param type Calculator type
    /// @return Calculator traits
    static CalculatorTraits getTraits(CalculatorType type) noexcept;
    
    /// Predict relative performance for a calculator on a problem
    /// @param type Calculator type
    /// @param characteristics Problem characteristics
    /// @return Predicted relative performance (higher is better)
    static double predictPerformance(CalculatorType type, 
                                   const ProblemCharacteristics& characteristics) noexcept;
    
    /// Select optimal calculator type for given problem
    /// @param characteristics Problem characteristics
    /// @return Optimal calculator type
    static CalculatorType selectOptimal(const ProblemCharacteristics& characteristics) noexcept;
    
    /// Create calculator instance of specified type
    /// @param type Calculator type
    /// @param hmm HMM for calculations
    /// @param observations Observation sequence
    /// @return Calculator instance
    static std::unique_ptr<ForwardBackwardCalculator> create(
        CalculatorType type, Hmm* hmm, const ObservationSet& observations);
    
    /// Create optimal calculator instance (convenience method)
    /// @param hmm HMM for calculations
    /// @param observations Observation sequence
    /// @param requiresStability Force numerical stability
    /// @param isRealTime Real-time processing flag
    /// @param memoryBudget Available memory budget
    /// @return Optimal calculator instance
    static std::unique_ptr<ForwardBackwardCalculator> createOptimal(
        Hmm* hmm, const ObservationSet& observations,
        bool requiresStability = false, bool isRealTime = false,
        double memoryBudget = 0.0);
    
    /// Get performance comparison for all calculator types
    /// @param characteristics Problem characteristics
    /// @return String with performance predictions
    static std::string getPerformanceComparison(const ProblemCharacteristics& characteristics);

private:
    /// Calculate SIMD benefit factor based on problem size
    /// @param numStates Number of states
    /// @param seqLength Sequence length
    /// @return SIMD benefit multiplier
    static double calculateSIMDBenefit(std::size_t numStates, std::size_t seqLength) noexcept;
    
    /// Calculate memory overhead impact
    /// @param overhead Memory overhead in bytes
    /// @param budget Available memory budget
    /// @return Memory impact factor (0.0 to 1.0)
    static double calculateMemoryImpact(std::size_t overhead, double budget) noexcept;
    
    /// Calculate numerical stability requirements
    /// @param seqLength Sequence length
    /// @return Stability requirement score
    static double calculateStabilityNeed(std::size_t seqLength) noexcept;
};

/// RAII helper for automatic calculator selection
class AutoCalculator {
private:
    std::unique_ptr<ForwardBackwardCalculator> calculator_;
    CalculatorType selectedType_;
    ProblemCharacteristics characteristics_;
    
public:
    /// Constructor with automatic selection
    /// @param hmm HMM for calculations
    /// @param observations Observation sequence
    /// @param requiresStability Force numerical stability
    /// @param isRealTime Real-time processing flag
    /// @param memoryBudget Available memory budget
    AutoCalculator(Hmm* hmm, const ObservationSet& observations,
                   bool requiresStability = false, bool isRealTime = false,
                   double memoryBudget = 0.0);
    
    /// Get the underlying calculator
    /// @return Reference to calculator
    ForwardBackwardCalculator& getCalculator() { return *calculator_; }
    
    /// Get the underlying calculator (const)
    /// @return Const reference to calculator
    const ForwardBackwardCalculator& getCalculator() const { return *calculator_; }
    
    /// Get selected calculator type
    /// @return Calculator type that was selected
    CalculatorType getSelectedType() const noexcept { return selectedType_; }
    
    /// Get problem characteristics
    /// @return Problem characteristics used for selection
    const ProblemCharacteristics& getCharacteristics() const noexcept { return characteristics_; }
    
    /// Get selection rationale
    /// @return String explaining why this calculator was chosen
    std::string getSelectionRationale() const;
    
    /// Calculate probability using selected calculator
    /// @return Probability of observation sequence
    double probability();
    
    /// Get forward variables
    /// @return Forward variables matrix
    Matrix getForwardVariables() const;
    
    /// Get backward variables
    /// @return Backward variables matrix
    Matrix getBackwardVariables() const;
};

/// Performance benchmarking for calculator selection tuning
class CalculatorBenchmark {
public:
    /// Benchmark all calculators on a specific problem
    /// @param hmm HMM to benchmark
    /// @param observations Observation sequence
    /// @param iterations Number of benchmark iterations
    /// @return Performance results for each calculator type
    static std::map<CalculatorType, double> benchmarkAll(
        Hmm* hmm, const ObservationSet& observations, std::size_t iterations = 10);
    
    /// Update performance model based on benchmark results
    /// @param characteristics Problem characteristics
    /// @param results Benchmark results
    static void updatePerformanceModel(const ProblemCharacteristics& characteristics,
                                      const std::map<CalculatorType, double>& results);
    
    /// Validate calculator selection accuracy
    /// @param hmm HMM to test
    /// @param observations Observation sequence
    /// @return True if automatic selection matches best benchmark result
    static bool validateSelection(Hmm* hmm, const ObservationSet& observations);
};

} // namespace calculators
} // namespace libhmm

#endif // LIBHMM_CALCULATOR_TRAITS_H_
