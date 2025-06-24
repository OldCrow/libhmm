#ifndef LIBHMM_NUMERICAL_STABILITY_H_
#define LIBHMM_NUMERICAL_STABILITY_H_

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "libhmm/common/common.h"

namespace libhmm {
namespace numerical {

/// Numerical stability constants and utilities
struct NumericalConstants {
    /// Minimum probability value to prevent underflow
    static constexpr double MIN_PROBABILITY = 1e-300;
    
    /// Maximum probability value to prevent overflow  
    static constexpr double MAX_PROBABILITY = 1.0 - 1e-15;
    
    /// Tolerance for convergence detection
    static constexpr double DEFAULT_CONVERGENCE_TOLERANCE = 1e-8;
    
    /// Maximum iterations before forced termination
    static constexpr std::size_t DEFAULT_MAX_ITERATIONS = 10000;
    
    /// Minimum log probability to prevent -infinity
    static constexpr double MIN_LOG_PROBABILITY = -700.0; // approx log(1e-300)
    
    /// Maximum log probability
    static constexpr double MAX_LOG_PROBABILITY = 0.0;
    
    /// Small value for numerical differentiation
    static constexpr double NUMERICAL_EPSILON = 1e-12;
    
    /// Scaling threshold for numerical stability
    static constexpr double SCALING_THRESHOLD = 1e-100;
};

/// Numerical safety checks and corrections
class NumericalSafety {
public:
    /// Check if a value is finite and valid
    /// @param value Value to check
    /// @param name Variable name for error messages
    /// @throws std::runtime_error if value is not finite
    static void checkFinite(double value, const std::string& name = "value");
    
    /// Check if a probability is in valid range [0, 1]
    /// @param prob Probability value to check
    /// @param name Variable name for error messages
    /// @throws std::runtime_error if probability is invalid
    static void checkProbability(double prob, const std::string& name = "probability");
    
    /// Check if a log probability is in valid range
    /// @param logProb Log probability value to check
    /// @param name Variable name for error messages
    /// @throws std::runtime_error if log probability is invalid
    static void checkLogProbability(double logProb, const std::string& name = "log_probability");
    
    /// Clamp probability to valid range [MIN_PROBABILITY, MAX_PROBABILITY]
    /// @param prob Probability to clamp
    /// @return Clamped probability
    static double clampProbability(double prob) noexcept;
    
    /// Clamp log probability to valid range
    /// @param logProb Log probability to clamp
    /// @return Clamped log probability
    static double clampLogProbability(double logProb) noexcept;
    
    /// Safe logarithm that handles edge cases
    /// @param value Value to take logarithm of
    /// @return Safe logarithm, clamped to valid range
    static double safeLog(double value) noexcept;
    
    /// Safe exponential that handles edge cases
    /// @param value Value to exponentiate
    /// @return Safe exponential, clamped to valid range
    static double safeExp(double value) noexcept;
    
    /// Check if a matrix contains only finite values
    /// @param matrix Matrix to check
    /// @param name Matrix name for error messages
    /// @throws std::runtime_error if matrix contains non-finite values
    static void checkMatrixFinite(const Matrix& matrix, const std::string& name = "matrix");
    
    /// Check if a vector contains only finite values
    /// @param vector Vector to check
    /// @param name Vector name for error messages
    /// @throws std::runtime_error if vector contains non-finite values
    static void checkVectorFinite(const Vector& vector, const std::string& name = "vector");
    
    /// Normalize probabilities to sum to 1, handling edge cases
    /// @param probs Probability vector to normalize (modified in place)
    /// @return True if normalization was successful, false if all probabilities were zero
    static bool normalizeProbabilities(Vector& probs) noexcept;
    
    /// Check if probabilities sum to approximately 1
    /// @param probs Probability vector to check
    /// @param tolerance Tolerance for sum check
    /// @return True if probabilities are properly normalized
    static bool isProbabilityDistribution(const Vector& probs, 
                                        double tolerance = NumericalConstants::DEFAULT_CONVERGENCE_TOLERANCE) noexcept;
};

/// Convergence detection for iterative algorithms
class ConvergenceDetector {
private:
    std::vector<double> history_;
    double tolerance_;
    std::size_t maxIterations_;
    std::size_t windowSize_;
    std::size_t currentIteration_;
    
public:
    /// Constructor with convergence parameters
    /// @param tolerance Convergence tolerance
    /// @param maxIterations Maximum iterations before forced termination
    /// @param windowSize Number of previous values to consider for convergence
    explicit ConvergenceDetector(double tolerance = NumericalConstants::DEFAULT_CONVERGENCE_TOLERANCE,
                                std::size_t maxIterations = NumericalConstants::DEFAULT_MAX_ITERATIONS,
                                std::size_t windowSize = 5);
    
    /// Add a new value and check convergence
    /// @param value New value (e.g., log-likelihood)
    /// @return True if algorithm has converged
    bool addValue(double value);
    
    /// Check if maximum iterations reached
    /// @return True if max iterations exceeded
    bool isMaxIterationsReached() const noexcept;
    
    /// Get current iteration count
    /// @return Current iteration number
    std::size_t getCurrentIteration() const noexcept;
    
    /// Get convergence history
    /// @return Vector of recent values
    const std::vector<double>& getHistory() const noexcept;
    
    /// Reset the detector for a new training run
    void reset() noexcept;
    
    /// Get convergence statistics
    /// @return String describing convergence status
    std::string getConvergenceReport() const;
    
    /// Check for oscillation (values bouncing back and forth)
    /// @return True if oscillation detected
    bool isOscillating() const noexcept;
    
    /// Check for stagnation (values not changing significantly)
    /// @return True if stagnation detected
    bool isStagnating() const noexcept;
};

/// Adaptive precision management for numerical computations
class AdaptivePrecision {
private:
    double baseTolerance_;
    double currentTolerance_;
    std::size_t problemSize_;  // Used for future problem-size-dependent scaling
    bool useAdaptive_;
    
public:
    /// Constructor with base tolerance and problem characteristics
    /// @param baseTolerance Base numerical tolerance
    /// @param problemSize Size of the problem (affects precision requirements)
    /// @param useAdaptive Whether to use adaptive precision scaling
    explicit AdaptivePrecision(double baseTolerance = NumericalConstants::DEFAULT_CONVERGENCE_TOLERANCE,
                              std::size_t problemSize = 1,
                              bool useAdaptive = true);
    
    /// Get current tolerance based on problem characteristics
    /// @return Current adaptive tolerance
    double getCurrentTolerance() const noexcept;
    
    /// Update tolerance based on iteration progress
    /// @param iteration Current iteration number
    /// @param convergenceRate Recent convergence rate
    void updateTolerance(std::size_t iteration, double convergenceRate = 1.0);
    
    /// Scale tolerance based on data characteristics
    /// @param dataRange Range of data values
    /// @param sequenceLength Length of observation sequences
    void scaleForData(double dataRange, std::size_t sequenceLength);
    
    /// Get recommended scaling threshold for the current precision
    /// @return Scaling threshold
    double getScalingThreshold() const noexcept;
};

/// Error recovery strategies for numerical failures
class ErrorRecovery {
public:
    /// Strategy for handling numerical failures
    enum class RecoveryStrategy {
        STRICT,         ///< Throw exception on any numerical issue
        GRACEFUL,       ///< Try to recover with degraded precision
        ROBUST,         ///< Aggressively recover, may sacrifice some accuracy
        ADAPTIVE        ///< Choose strategy based on problem characteristics
    };
    
    /// Attempt to recover from underflow in probability calculations
    /// @param probs Probability vector with potential underflow
    /// @param strategy Recovery strategy to use
    /// @return True if recovery was successful
    static bool recoverFromUnderflow(Vector& probs, RecoveryStrategy strategy = RecoveryStrategy::GRACEFUL);
    
    /// Attempt to recover from overflow in exponential calculations
    /// @param values Values that may have caused overflow
    /// @param strategy Recovery strategy to use
    /// @return True if recovery was successful
    static bool recoverFromOverflow(Vector& values, RecoveryStrategy strategy = RecoveryStrategy::GRACEFUL);
    
    /// Handle NaN values in computations
    /// @param values Vector containing potential NaN values
    /// @param strategy Recovery strategy to use
    /// @return Number of NaN values recovered
    static std::size_t handleNaNValues(Vector& values, RecoveryStrategy strategy = RecoveryStrategy::GRACEFUL);
    
    /// Attempt to fix a degenerate probability matrix
    /// @param matrix Matrix with potential issues
    /// @param strategy Recovery strategy to use
    /// @return True if matrix was successfully repaired
    static bool repairDegenerateMatrix(Matrix& matrix, RecoveryStrategy strategy = RecoveryStrategy::GRACEFUL);
    
    /// Get default recovery strategy for given problem characteristics
    /// @param sequenceLength Length of observation sequences
    /// @param numStates Number of HMM states
    /// @param isRealTime Whether this is real-time processing
    /// @return Recommended recovery strategy
    static RecoveryStrategy getRecommendedStrategy(std::size_t sequenceLength, 
                                                 std::size_t numStates, 
                                                 bool isRealTime = false) noexcept;
};

/// Numerical diagnostics for debugging and monitoring
class NumericalDiagnostics {
public:
    /// Diagnostic information about numerical stability
    struct DiagnosticReport {
        bool hasUnderflow = false;
        bool hasOverflow = false;
        bool hasNaN = false;
        bool hasInfinity = false;
        double minValue = std::numeric_limits<double>::max();
        double maxValue = std::numeric_limits<double>::lowest();
        double conditionNumber = 1.0;
        std::size_t problemSize = 0;
        std::string recommendations;
    };
    
    /// Analyze vector for numerical issues
    /// @param vector Vector to analyze
    /// @param name Name for reporting
    /// @return Diagnostic report
    static DiagnosticReport analyzeVector(const Vector& vector, const std::string& name = "vector");
    
    /// Analyze matrix for numerical issues
    /// @param matrix Matrix to analyze
    /// @param name Name for reporting
    /// @return Diagnostic report
    static DiagnosticReport analyzeMatrix(const Matrix& matrix, const std::string& name = "matrix");
    
    /// Generate comprehensive numerical health report
    /// @param reports Collection of diagnostic reports
    /// @return Formatted health report string
    static std::string generateHealthReport(const std::vector<DiagnosticReport>& reports);
    
    /// Estimate condition number of a matrix (simplified)
    /// @param matrix Matrix to analyze
    /// @return Estimated condition number
    static double estimateConditionNumber(const Matrix& matrix) noexcept;
};

} // namespace numerical
} // namespace libhmm

#endif // LIBHMM_NUMERICAL_STABILITY_H_
