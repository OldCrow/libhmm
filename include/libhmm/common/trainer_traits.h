#ifndef LIBHMM_TRAINER_TRAITS_H_
#define LIBHMM_TRAINER_TRAITS_H_

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <chrono>
#include "libhmm/common/common.h"
#include "libhmm/common/numerical_stability.h"

namespace libhmm {

// Forward declarations
class Hmm;
class HmmTrainer;
class ViterbiTrainer;
class BaumWelchTrainer;
class ScaledBaumWelchTrainer;
class RobustViterbiTrainer;

namespace trainer_traits {

/// Enumeration of available trainer types
enum class TrainerType {
    VITERBI,                    ///< Standard Viterbi (k-means) trainer
    ROBUST_VITERBI,            ///< Enhanced Viterbi with robust edge case handling
    BAUM_WELCH,                ///< Standard Baum-Welch trainer
    SCALED_BAUM_WELCH,         ///< Scaled Baum-Welch for numerical stability
    AUTO                       ///< Automatic selection based on problem characteristics
};

/// Problem complexity classification for trainer selection
enum class ProblemComplexity {
    SIMPLE,                    ///< Well-conditioned, small-scale problems
    MODERATE,                  ///< Medium-scale problems with standard characteristics
    COMPLEX,                   ///< Large-scale or ill-conditioned problems
    EXTREME                    ///< Very challenging problems requiring maximum robustness
};

/// Training objectives that influence trainer selection
enum class TrainingObjective {
    SPEED,                     ///< Prioritize training speed
    ACCURACY,                  ///< Prioritize training accuracy
    ROBUSTNESS,                ///< Prioritize stability and reliability
    BALANCED                   ///< Balance between speed, accuracy, and robustness
};

/// Data quality assessment for trainer selection
struct DataCharacteristics {
    std::size_t numSequences = 0;           ///< Number of observation sequences
    std::size_t avgSequenceLength = 0;      ///< Average sequence length
    std::size_t maxSequenceLength = 0;      ///< Maximum sequence length
    std::size_t numStates = 0;              ///< Number of HMM states
    double dataRange = 0.0;                 ///< Range of observation values
    double dataMean = 0.0;                  ///< Mean of observations
    double dataStdDev = 0.0;                ///< Standard deviation of observations
    bool hasOutliers = false;               ///< Presence of outlier values
    bool hasRepeatedSequences = false;     ///< Presence of identical sequences
    bool isDiscrete = false;                ///< Whether data appears discrete
    double sparsity = 0.0;                  ///< Sparsity measure (0=dense, 1=sparse)
    ProblemComplexity complexity = ProblemComplexity::MODERATE;
    
    /// Constructor from observation data
    /// @param obsLists Observation sequences to analyze
    /// @param hmm HMM for context (number of states, etc.)
    DataCharacteristics(const ObservationLists& obsLists, const Hmm* hmm);
    
    /// Default constructor
    DataCharacteristics() = default;
};

/// Performance characteristics and capabilities of a trainer
struct TrainerCapabilities {
    std::string name;                       ///< Human-readable trainer name
    TrainerType type;                       ///< Trainer type identifier
    
    // Capability flags
    bool supportsDiscreteDistributions = false;     ///< Can train discrete distributions
    bool supportsContinuousDistributions = false;   ///< Can train continuous distributions
    bool handlesEmptyClusters = false;              ///< Robust to empty clusters
    bool isNumericallyStable = false;               ///< Uses numerical stability techniques
    bool supportsEarlyTermination = false;          ///< Can terminate early
    bool providesConvergenceInfo = false;           ///< Provides convergence diagnostics
    
    // Performance characteristics
    double speedFactor = 1.0;               ///< Relative speed (1.0 = baseline)
    double accuracyFactor = 1.0;            ///< Relative accuracy (1.0 = baseline)
    double robustnessFactor = 1.0;          ///< Relative robustness (1.0 = baseline)
    double memoryOverhead = 1.0;            ///< Memory usage multiplier
    
    // Problem size recommendations
    std::size_t minRecommendedStates = 1;   ///< Minimum states for effectiveness
    std::size_t maxRecommendedStates = 1000; ///< Maximum states before degradation
    std::size_t minRecommendedSequences = 1; ///< Minimum sequences needed
    std::size_t minSequenceLength = 1;      ///< Minimum effective sequence length
    
    // Numerical stability characteristics
    double convergenceTolerance = 1e-8;     ///< Default convergence tolerance
    std::size_t maxIterations = 1000;       ///< Default maximum iterations
    numerical::ErrorRecovery::RecoveryStrategy recommendedRecoveryStrategy = 
        numerical::ErrorRecovery::RecoveryStrategy::GRACEFUL;
};

/// Training configuration optimized for specific scenarios
struct TrainingConfiguration {
    TrainerType trainerType = TrainerType::AUTO;
    TrainingObjective objective = TrainingObjective::BALANCED;
    
    // Numerical parameters
    double convergenceTolerance = 1e-8;
    std::size_t maxIterations = 1000;
    bool enableAdaptivePrecision = true;
    bool enableRobustErrorRecovery = true;
    
    // Performance parameters
    bool enableEarlyTermination = true;
    bool enableProgressReporting = false;
    double memoryBudget = 0.0; ///< Memory budget in bytes (0 = unlimited)
    
    // Recovery strategy
    numerical::ErrorRecovery::RecoveryStrategy recoveryStrategy = 
        numerical::ErrorRecovery::RecoveryStrategy::ADAPTIVE;
    
    /// Constructor with automatic configuration
    /// @param characteristics Data characteristics for optimization
    /// @param objective Training objective
    TrainingConfiguration(const DataCharacteristics& characteristics,
                         TrainingObjective objective = TrainingObjective::BALANCED);
    
    /// Default constructor
    TrainingConfiguration() = default;
};

/// Performance prediction for trainer selection
struct PerformancePrediction {
    TrainerType trainerType;
    double expectedTrainingTime = 0.0;      ///< Predicted training time (seconds)
    double expectedAccuracy = 0.0;          ///< Predicted accuracy score (0-1)
    double expectedRobustness = 0.0;        ///< Predicted robustness score (0-1)
    double expectedMemoryUsage = 0.0;       ///< Predicted memory usage (bytes)
    double confidenceScore = 0.0;           ///< Prediction confidence (0-1)
    std::string rationale;                  ///< Explanation of choice
    std::vector<std::string> warnings;     ///< Potential issues or concerns
    std::vector<std::string> recommendations; ///< Optimization suggestions
};

/// Core trainer traits and capability detection system
class TrainerTraits {
public:
    /// Get capabilities for a specific trainer type
    /// @param type Trainer type to query
    /// @return Trainer capabilities structure
    static TrainerCapabilities getCapabilities(TrainerType type) noexcept;
    
    /// Get all available trainer types and their capabilities
    /// @return Map of trainer types to their capabilities
    static std::map<TrainerType, TrainerCapabilities> getAllCapabilities() noexcept;
    
    /// Check if a trainer supports specific distribution types
    /// @param type Trainer type to check
    /// @param isDiscrete Whether distributions are discrete
    /// @return True if trainer supports the distribution type
    static bool supportsDistributionType(TrainerType type, bool isDiscrete) noexcept;
    
    /// Get recommended trainers for given data characteristics
    /// @param characteristics Data characteristics
    /// @param objective Training objective
    /// @return Vector of recommended trainer types (ordered by preference)
    static std::vector<TrainerType> getRecommendedTrainers(
        const DataCharacteristics& characteristics,
        TrainingObjective objective = TrainingObjective::BALANCED) noexcept;
    
    /// Predict performance for a trainer on specific data
    /// @param type Trainer type
    /// @param characteristics Data characteristics
    /// @param objective Training objective
    /// @return Performance prediction
    static PerformancePrediction predictPerformance(
        TrainerType type,
        const DataCharacteristics& characteristics,
        TrainingObjective objective = TrainingObjective::BALANCED) noexcept;
    
    /// Select optimal trainer automatically
    /// @param characteristics Data characteristics
    /// @param objective Training objective
    /// @return Optimal trainer type
    static TrainerType selectOptimalTrainer(
        const DataCharacteristics& characteristics,
        TrainingObjective objective = TrainingObjective::BALANCED) noexcept;
    
    /// Generate performance comparison report
    /// @param characteristics Data characteristics
    /// @param objective Training objective
    /// @return Formatted comparison report
    static std::string generatePerformanceComparison(
        const DataCharacteristics& characteristics,
        TrainingObjective objective = TrainingObjective::BALANCED);
    
    /// Validate trainer compatibility with HMM
    /// @param type Trainer type
    /// @param hmm HMM to validate against
    /// @return True if trainer is compatible
    static bool isCompatible(TrainerType type, const Hmm* hmm) noexcept;

private:
    /// Calculate complexity score for data characteristics
    /// @param characteristics Data characteristics
    /// @return Complexity score (0-1, higher = more complex)
    static double calculateComplexityScore(const DataCharacteristics& characteristics) noexcept;
    
    /// Calculate problem scale factor
    /// @param characteristics Data characteristics
    /// @return Scale factor (1.0 = baseline)
    static double calculateScaleFactor(const DataCharacteristics& characteristics) noexcept;
    
    /// Estimate training time for a trainer
    /// @param type Trainer type
    /// @param characteristics Data characteristics
    /// @return Estimated training time in seconds
    static double estimateTrainingTime(TrainerType type, const DataCharacteristics& characteristics) noexcept;
    
    /// Calculate accuracy expectation
    /// @param type Trainer type
    /// @param characteristics Data characteristics
    /// @return Expected accuracy score (0-1)
    static double calculateAccuracyExpectation(TrainerType type, const DataCharacteristics& characteristics) noexcept;
    
    /// Calculate robustness expectation
    /// @param type Trainer type
    /// @param characteristics Data characteristics
    /// @return Expected robustness score (0-1)
    static double calculateRobustnessExpectation(TrainerType type, const DataCharacteristics& characteristics) noexcept;
};

/// Factory for creating trainers with optimal configuration
class TrainerFactory {
public:
    /// Create trainer with automatic selection and configuration
    /// @param hmm HMM to train
    /// @param obsLists Observation sequences
    /// @param objective Training objective
    /// @return Configured trainer instance
    static std::unique_ptr<HmmTrainer> createOptimalTrainer(
        Hmm* hmm,
        const ObservationLists& obsLists,
        TrainingObjective objective = TrainingObjective::BALANCED);
    
    /// Create trainer of specific type with optimal configuration
    /// @param type Trainer type
    /// @param hmm HMM to train
    /// @param obsLists Observation sequences
    /// @param config Training configuration (optional)
    /// @return Configured trainer instance
    static std::unique_ptr<HmmTrainer> createTrainer(
        TrainerType type,
        Hmm* hmm,
        const ObservationLists& obsLists,
        const TrainingConfiguration& config = TrainingConfiguration{});
    
    /// Create trainer with performance monitoring
    /// @param type Trainer type
    /// @param hmm HMM to train
    /// @param obsLists Observation sequences
    /// @param config Training configuration
    /// @return Configured trainer with monitoring enabled
    static std::unique_ptr<HmmTrainer> createMonitoredTrainer(
        TrainerType type,
        Hmm* hmm,
        const ObservationLists& obsLists,
        const TrainingConfiguration& config = TrainingConfiguration{});
    
    /// Get recommended configuration for specific trainer and data
    /// @param type Trainer type
    /// @param characteristics Data characteristics
    /// @param objective Training objective
    /// @return Optimized training configuration
    static TrainingConfiguration getRecommendedConfiguration(
        TrainerType type,
        const DataCharacteristics& characteristics,
        TrainingObjective objective = TrainingObjective::BALANCED);

private:
    /// Configure trainer with optimal parameters
    /// @param trainer Trainer to configure
    /// @param config Training configuration
    static void configureTrainer(HmmTrainer* trainer, const TrainingConfiguration& config);
};

/// RAII helper for automatic trainer selection and execution
class AutoTrainer {
private:
    std::unique_ptr<HmmTrainer> trainer_;
    TrainerType selectedType_;
    DataCharacteristics characteristics_;
    TrainingConfiguration config_;
    std::chrono::steady_clock::time_point startTime_;
    
public:
    /// Constructor with automatic trainer selection
    /// @param hmm HMM to train
    /// @param obsLists Observation sequences
    /// @param objective Training objective
    AutoTrainer(Hmm* hmm, const ObservationLists& obsLists,
                TrainingObjective objective = TrainingObjective::BALANCED);
    
    /// Execute training with monitoring
    /// @return True if training completed successfully
    bool train();
    
    /// Get selected trainer type
    /// @return Trainer type that was selected
    TrainerType getSelectedType() const noexcept { return selectedType_; }
    
    /// Get data characteristics
    /// @return Data characteristics used for selection
    const DataCharacteristics& getCharacteristics() const noexcept { return characteristics_; }
    
    /// Get training configuration
    /// @return Training configuration used
    const TrainingConfiguration& getConfiguration() const noexcept { return config_; }
    
    /// Get selection rationale
    /// @return String explaining why this trainer was chosen
    std::string getSelectionRationale() const;
    
    /// Get training performance report
    /// @return Detailed performance report
    std::string getPerformanceReport() const;
    
    /// Get underlying trainer
    /// @return Reference to the trainer (may be null if training failed)
    HmmTrainer* getTrainer() const noexcept { return trainer_.get(); }
    
    /// Check if training completed successfully
    /// @return True if training was successful
    bool isTrainingSuccessful() const noexcept;
    
    /// Get training duration
    /// @return Training time in seconds
    double getTrainingDuration() const noexcept;
};

/// Performance benchmarking for trainer comparison and tuning
class TrainerBenchmark {
public:
    /// Benchmark result for a single trainer
    struct BenchmarkResult {
        TrainerType trainerType;
        double trainingTime = 0.0;          ///< Actual training time (seconds)
        double finalLogLikelihood = 0.0;    ///< Final log-likelihood achieved
        std::size_t iterations = 0;         ///< Number of iterations used
        bool converged = false;             ///< Whether training converged
        double memoryUsage = 0.0;           ///< Peak memory usage (bytes)
        std::string errorMessage;           ///< Error message if training failed
        std::vector<std::string> warnings;  ///< Warnings during training
    };
    
    /// Benchmark all applicable trainers on given data
    /// @param hmm HMM to train
    /// @param obsLists Observation sequences
    /// @param iterations Number of benchmark runs per trainer
    /// @return Benchmark results for each trainer
    static std::map<TrainerType, BenchmarkResult> benchmarkAll(
        Hmm* hmm,
        const ObservationLists& obsLists,
        std::size_t iterations = 3);
    
    /// Benchmark specific trainer
    /// @param type Trainer type to benchmark
    /// @param hmm HMM to train
    /// @param obsLists Observation sequences
    /// @param iterations Number of benchmark runs
    /// @return Average benchmark result
    static BenchmarkResult benchmarkTrainer(
        TrainerType type,
        Hmm* hmm,
        const ObservationLists& obsLists,
        std::size_t iterations = 3);
    
    /// Generate benchmark report
    /// @param results Benchmark results
    /// @return Formatted benchmark report
    static std::string generateBenchmarkReport(const std::map<TrainerType, BenchmarkResult>& results);
    
    /// Validate trainer selection accuracy
    /// @param hmm HMM to test
    /// @param obsLists Observation sequences
    /// @return True if automatic selection matches best benchmark result
    static bool validateSelection(Hmm* hmm, const ObservationLists& obsLists);
    
    /// Update performance models based on benchmark results
    /// @param characteristics Data characteristics
    /// @param results Benchmark results
    static void updatePerformanceModels(const DataCharacteristics& characteristics,
                                       const std::map<TrainerType, BenchmarkResult>& results);
};

/// Configuration presets for common training scenarios
namespace presets {
    /// Configuration for real-time applications (prioritize speed)
    TrainingConfiguration realtime();
    
    /// Configuration for high-accuracy research applications
    TrainingConfiguration highAccuracy();
    
    /// Configuration for maximum robustness (handle any data)
    TrainingConfiguration maxRobustness();
    
    /// Configuration for large-scale problems
    TrainingConfiguration largScale();
    
    /// Configuration for small, well-behaved problems
    TrainingConfiguration simple();
    
    /// Configuration balancing all factors
    TrainingConfiguration balanced();
}

} // namespace trainer_traits
} // namespace libhmm

#endif // LIBHMM_TRAINER_TRAITS_H_
