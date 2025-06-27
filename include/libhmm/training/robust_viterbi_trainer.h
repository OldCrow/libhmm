#ifndef LIBHMM_ROBUST_VITERBI_TRAINER_H_
#define LIBHMM_ROBUST_VITERBI_TRAINER_H_

#include "libhmm/training/viterbi_trainer.h"
#include "libhmm/common/numerical_stability.h"
#include <memory>
#include <string>

namespace libhmm {

/// Enhanced ViterbiTrainer with comprehensive robust edge case handling
/// Extends the base ViterbiTrainer with numerical stability, adaptive convergence,
/// and error recovery capabilities for reliable training on challenging datasets
class RobustViterbiTrainer : public ViterbiTrainer {
public:
    /// Training configuration
    struct RobustTrainingConfig {
        bool enableAdaptivePrecision = true;
        bool enableErrorRecovery = true;
        bool enableDiagnostics = true;
        bool enableProgressReporting = true;
        double convergenceTolerance = constants::precision::DEFAULT_CONVERGENCE_TOLERANCE;
        std::size_t maxIterations = constants::iterations::DEFAULT_MAX_ITERATIONS;
        std::size_t convergenceWindow = 5;
    };

private:
    /// Training configuration instance
    RobustTrainingConfig config_;
    
    /// Numerical stability and convergence management
    std::unique_ptr<numerical::ConvergenceDetector> convergenceDetector_;
    std::unique_ptr<numerical::AdaptivePrecision> adaptivePrecision_;
    
    /// Error recovery strategy
    numerical::ErrorRecovery::RecoveryStrategy recoveryStrategy_;
    
    /// Diagnostic tracking
    mutable std::vector<numerical::NumericalDiagnostics::DiagnosticReport> diagnosticReports_;
    
    /// Validate and preprocess observation data
    /// @param obsLists Observation lists to validate
    /// @throws std::runtime_error if data has irrecoverable issues
    void validateAndPreprocessData(const ObservationLists& obsLists);
    
    /// Analyze data characteristics for adaptive precision
    /// @param obsLists Observation lists to analyze
    /// @return Data range and characteristics
    std::pair<double, std::size_t> analyzeDataCharacteristics(const ObservationLists& obsLists) const;
    
    /// Enhanced cluster fitting with robust error handling
    /// @param clusterIndex Index of cluster to fit
    /// @return True if fitting was successful
    bool robustClusterFitting(std::size_t clusterIndex);
    
    /// Validate HMM state after training iteration
    /// @return True if HMM state is valid
    bool validateHmmState() const;
    
    /// Generate comprehensive training report
    /// @return Detailed training report string
    std::string generateTrainingReport() const;
    
    /// Handle numerical issues during training
    /// @param issueDescription Description of the issue
    /// @param canRecover Whether recovery is possible
    /// @return True if issue was resolved
    bool handleNumericalIssue(const std::string& issueDescription, bool canRecover = true);

protected:
    /// Enhanced training loop with robust error handling
    void robustTrainingLoop();
    
    /// Validate cluster state before fitting
    /// @param clusterIndex Index of cluster to validate
    /// @return True if cluster is ready for fitting
    bool validateClusterForFitting(std::size_t clusterIndex) const;
    
    /// Robust probability matrix calculations with error recovery
    void robustCalculatePi();
    void robustCalculateTrans();

public:
    /// Constructor with enhanced configuration options
    /// @param hmm Pointer to the HMM to train (must not be null)
    /// @param obsLists List of observation sequences for training
    /// @param config Training configuration
    /// @throws std::invalid_argument if hmm is null or obsLists is empty
    RobustViterbiTrainer(Hmm* hmm, const ObservationLists& obsLists,
                        const RobustTrainingConfig& config);
                        
    /// Constructor with default configuration
    /// @param hmm Pointer to the HMM to train (must not be null)
    /// @param obsLists List of observation sequences for training
    /// @throws std::invalid_argument if hmm is null or obsLists is empty
    RobustViterbiTrainer(Hmm* hmm, const ObservationLists& obsLists);
    
    /// Virtual destructor
    virtual ~RobustViterbiTrainer() = default;
    
    /// Execute robust Viterbi training with comprehensive error handling
    /// Updates HMM parameters using numerically stable algorithms
    virtual void train() override;
    
    /// Get convergence information
    /// @return Convergence report string
    std::string getConvergenceReport() const;
    
    /// Get numerical health report
    /// @return Comprehensive numerical diagnostics
    std::string getNumericalHealthReport() const;
    
    /// Check if training converged successfully
    /// @return True if training converged within tolerance
    bool hasConverged() const;
    
    /// Check if maximum iterations were reached
    /// @return True if training stopped due to iteration limit
    bool reachedMaxIterations() const;
    
    /// Get training configuration
    /// @return Current training configuration
    const RobustTrainingConfig& getConfig() const noexcept;
    
    /// Update training configuration
    /// @param config New training configuration
    void setConfig(const RobustTrainingConfig& config);
    
    /// Set recovery strategy for numerical issues
    /// @param strategy Error recovery strategy to use
    void setRecoveryStrategy(numerical::ErrorRecovery::RecoveryStrategy strategy);
    
    /// Get current recovery strategy
    /// @return Current error recovery strategy
    numerical::ErrorRecovery::RecoveryStrategy getRecoveryStrategy() const noexcept;
    
    /// Enable or disable adaptive precision
    /// @param enable Whether to use adaptive precision
    void setAdaptivePrecisionEnabled(bool enable);
    
    /// Check if adaptive precision is enabled
    /// @return True if adaptive precision is enabled
    bool isAdaptivePrecisionEnabled() const noexcept;
    
    /// Get current numerical tolerance
    /// @return Current convergence tolerance
    double getCurrentTolerance() const;
    
    /// Force early termination of training
    /// Useful for interactive applications or timeout scenarios
    void requestEarlyTermination();
    
    /// Reset training state for a new training run
    /// Preserves configuration but clears convergence history
    void resetTrainingState();
    
    /// Factory method for creating trainer with recommended settings
    /// @param hmm HMM to train
    /// @param obsLists Observation sequences
    /// @param problemType Problem characteristics for optimal configuration
    /// @return Configured robust trainer
    static std::unique_ptr<RobustViterbiTrainer> createWithRecommendedSettings(
        Hmm* hmm, const ObservationLists& obsLists, const std::string& problemType = "general");
    
    /// Validate that the trainer is properly configured
    /// @throws std::runtime_error if configuration is invalid
    void validateConfiguration() const;
    
    /// Get recommendations for improving training performance
    /// @return Vector of recommendation strings
    std::vector<std::string> getPerformanceRecommendations() const;
};

/// Training configuration presets for common scenarios
namespace training_presets {
    /// Conservative settings for maximum stability
    RobustViterbiTrainer::RobustTrainingConfig conservative();
    
    /// Balanced settings for general use
    RobustViterbiTrainer::RobustTrainingConfig balanced();
    
    /// Aggressive settings for speed (may sacrifice some stability)
    RobustViterbiTrainer::RobustTrainingConfig aggressive();
    
    /// Real-time settings for interactive applications
    RobustViterbiTrainer::RobustTrainingConfig realtime();
    
    /// High-precision settings for research applications
    RobustViterbiTrainer::RobustTrainingConfig highPrecision();
}

} // namespace libhmm

#endif // LIBHMM_ROBUST_VITERBI_TRAINER_H_
