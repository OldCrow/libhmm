#include "libhmm/common/numerical_stability.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <numeric>

namespace libhmm {
namespace numerical {

// NumericalSafety implementation
void NumericalSafety::checkFinite(double value, const std::string& name) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("Non-finite value detected in " + name + 
                                ": " + std::to_string(value));
    }
}

void NumericalSafety::checkProbability(double prob, const std::string& name) {
    checkFinite(prob, name);
    if (prob < 0.0 || prob > 1.0) {
        throw std::runtime_error("Invalid probability value in " + name + 
                                ": " + std::to_string(prob) + " (must be in [0,1])");
    }
}

void NumericalSafety::checkLogProbability(double logProb, const std::string& name) {
    checkFinite(logProb, name);
    if (logProb > NumericalConstants::MAX_LOG_PROBABILITY) {
        throw std::runtime_error("Invalid log probability value in " + name + 
                                ": " + std::to_string(logProb) + " (must be <= 0)");
    }
}

double NumericalSafety::clampProbability(double prob) noexcept {
    if (!std::isfinite(prob)) {
        return NumericalConstants::MIN_PROBABILITY;
    }
    return std::clamp(prob, NumericalConstants::MIN_PROBABILITY, NumericalConstants::MAX_PROBABILITY);
}

double NumericalSafety::clampLogProbability(double logProb) noexcept {
    if (!std::isfinite(logProb)) {
        return NumericalConstants::MIN_LOG_PROBABILITY;
    }
    return std::clamp(logProb, NumericalConstants::MIN_LOG_PROBABILITY, NumericalConstants::MAX_LOG_PROBABILITY);
}

double NumericalSafety::safeLog(double value) noexcept {
    if (value <= 0.0 || !std::isfinite(value)) {
        return NumericalConstants::MIN_LOG_PROBABILITY;
    }
    return std::max(std::log(value), NumericalConstants::MIN_LOG_PROBABILITY);
}

double NumericalSafety::safeExp(double value) noexcept {
    if (!std::isfinite(value)) {
        return NumericalConstants::MIN_PROBABILITY;
    }
    if (value < NumericalConstants::MIN_LOG_PROBABILITY) {
        return NumericalConstants::MIN_PROBABILITY;
    }
    if (value > 700.0) { // Log of very large number, prevent overflow
        return NumericalConstants::MAX_PROBABILITY;
    }
    double result = std::exp(value);
    return clampProbability(result);
}

void NumericalSafety::checkMatrixFinite(const Matrix& matrix, const std::string& name) {
    for (std::size_t i = 0; i < matrix.size1(); ++i) {
        for (std::size_t j = 0; j < matrix.size2(); ++j) {
            if (!std::isfinite(matrix(i, j))) {
                throw std::runtime_error("Non-finite value detected in " + name + 
                                        " at position (" + std::to_string(i) + "," + std::to_string(j) + 
                                        "): " + std::to_string(matrix(i, j)));
            }
        }
    }
}

void NumericalSafety::checkVectorFinite(const Vector& vector, const std::string& name) {
    for (std::size_t i = 0; i < vector.size(); ++i) {
        if (!std::isfinite(vector(i))) {
            throw std::runtime_error("Non-finite value detected in " + name + 
                                    " at position " + std::to_string(i) + 
                                    ": " + std::to_string(vector(i)));
        }
    }
}

bool NumericalSafety::normalizeProbabilities(Vector& probs) noexcept {
    // First, clamp all probabilities to valid range
    for (std::size_t i = 0; i < probs.size(); ++i) {
        probs(i) = clampProbability(probs(i));
    }
    
    // Calculate sum
    double sum = 0.0;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        sum += probs(i);
    }
    
    // Handle degenerate case where all probabilities are effectively zero
    if (sum < 1e-15) {
        // Set uniform distribution
        const double uniformProb = 1.0 / static_cast<double>(probs.size());
        for (std::size_t i = 0; i < probs.size(); ++i) {
            probs(i) = uniformProb;
        }
        return false; // Indicate that we had to fix the distribution
    }
    
    // Normalize
    for (std::size_t i = 0; i < probs.size(); ++i) {
        probs(i) /= sum;
    }
    
    return true;
}

bool NumericalSafety::isProbabilityDistribution(const Vector& probs, double tolerance) noexcept {
    // Check individual probabilities
    for (std::size_t i = 0; i < probs.size(); ++i) {
        if (!std::isfinite(probs(i)) || probs(i) < 0.0 || probs(i) > 1.0) {
            return false;
        }
    }
    
    // Check sum
    double sum = 0.0;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        sum += probs(i);
    }
    
    return std::abs(sum - 1.0) <= tolerance;
}

// ConvergenceDetector implementation
ConvergenceDetector::ConvergenceDetector(double tolerance, std::size_t maxIterations, std::size_t windowSize)
    : tolerance_(tolerance), maxIterations_(maxIterations), windowSize_(windowSize), currentIteration_(0) {
    history_.reserve(windowSize);
}

bool ConvergenceDetector::addValue(double value) {
    NumericalSafety::checkFinite(value, "convergence_value");
    
    history_.push_back(value);
    if (history_.size() > windowSize_) {
        history_.erase(history_.begin());
    }
    
    ++currentIteration_;
    
    // Need at least 2 values to check convergence
    if (history_.size() < 2) {
        return false;
    }
    
    // Check for convergence based on recent changes
    if (history_.size() >= windowSize_) {
        double maxChange = 0.0;
        for (std::size_t i = 1; i < history_.size(); ++i) {
            double change = std::abs(history_[i] - history_[i-1]);
            maxChange = std::max(maxChange, change);
        }
        
        if (maxChange < tolerance_) {
            return true;
        }
    }
    
    return false;
}

bool ConvergenceDetector::isMaxIterationsReached() const noexcept {
    return currentIteration_ >= maxIterations_;
}

std::size_t ConvergenceDetector::getCurrentIteration() const noexcept {
    return currentIteration_;
}

const std::vector<double>& ConvergenceDetector::getHistory() const noexcept {
    return history_;
}

void ConvergenceDetector::reset() noexcept {
    history_.clear();
    currentIteration_ = 0;
}

std::string ConvergenceDetector::getConvergenceReport() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Convergence Report:\n";
    oss << "  Iteration: " << currentIteration_ << "/" << maxIterations_ << "\n";
    oss << "  Tolerance: " << tolerance_ << "\n";
    
    if (!history_.empty()) {
        oss << "  Current value: " << history_.back() << "\n";
        
        if (history_.size() >= 2) {
            double lastChange = std::abs(history_.back() - history_[history_.size()-2]);
            oss << "  Last change: " << lastChange << "\n";
        }
        
        if (isOscillating()) {
            oss << "  Status: Oscillating\n";
        } else if (isStagnating()) {
            oss << "  Status: Stagnating\n";
        } else {
            oss << "  Status: Converging\n";
        }
    }
    
    return oss.str();
}

bool ConvergenceDetector::isOscillating() const noexcept {
    if (history_.size() < 4) return false;
    
    // Check for alternating increases and decreases
    int changes = 0;
    for (std::size_t i = 2; i < history_.size(); ++i) {
        double change1 = history_[i-1] - history_[i-2];
        double change2 = history_[i] - history_[i-1];
        
        if ((change1 > 0 && change2 < 0) || (change1 < 0 && change2 > 0)) {
            changes++;
        }
    }
    
    // If most changes are oscillating, consider it oscillation
    return changes >= static_cast<int>(history_.size() - 2) * 0.7;
}

bool ConvergenceDetector::isStagnating() const noexcept {
    if (history_.size() < windowSize_) return false;
    
    // Check if all recent changes are very small
    double maxChange = 0.0;
    for (std::size_t i = 1; i < history_.size(); ++i) {
        maxChange = std::max(maxChange, std::abs(history_[i] - history_[i-1]));
    }
    
    return maxChange < tolerance_ * 0.1; // Much smaller than tolerance
}

// AdaptivePrecision implementation
AdaptivePrecision::AdaptivePrecision(double baseTolerance, std::size_t problemSize, bool useAdaptive)
    : baseTolerance_(baseTolerance), currentTolerance_(baseTolerance), 
      problemSize_(problemSize), useAdaptive_(useAdaptive) {
}

double AdaptivePrecision::getCurrentTolerance() const noexcept {
    return currentTolerance_;
}

void AdaptivePrecision::updateTolerance(std::size_t iteration, double convergenceRate) {
    if (!useAdaptive_) {
        return;
    }
    
    // Relax tolerance as iterations progress (slower convergence may need looser tolerance)
    double iterationFactor = 1.0 + static_cast<double>(iteration) / 1000.0;
    
    // Adjust based on convergence rate
    double rateFactor = std::clamp(convergenceRate, 0.1, 10.0);
    
    currentTolerance_ = baseTolerance_ * iterationFactor * rateFactor;
    
    // Keep within reasonable bounds
    currentTolerance_ = std::clamp(currentTolerance_, 
                                  baseTolerance_ * 0.1, 
                                  baseTolerance_ * 100.0);
}

void AdaptivePrecision::scaleForData(double dataRange, std::size_t sequenceLength) {
    if (!useAdaptive_) {
        return;
    }
    
    // Scale tolerance based on data range
    double rangeFactor = std::max(1.0, std::log10(std::max(dataRange, 1.0)));
    
    // Scale based on sequence length (longer sequences may need looser tolerance)
    double lengthFactor = 1.0 + std::log10(static_cast<double>(sequenceLength)) / 10.0;
    
    currentTolerance_ = baseTolerance_ * rangeFactor * lengthFactor;
    
    // Keep within reasonable bounds
    currentTolerance_ = std::clamp(currentTolerance_, 
                                  baseTolerance_ * 0.01, 
                                  baseTolerance_ * 1000.0);
}

double AdaptivePrecision::getScalingThreshold() const noexcept {
    return NumericalConstants::SCALING_THRESHOLD * (currentTolerance_ / baseTolerance_);
}

// ErrorRecovery implementation
bool ErrorRecovery::recoverFromUnderflow(Vector& probs, RecoveryStrategy strategy) {
    bool hasUnderflow = false;
    
    // Detect underflow
    for (std::size_t i = 0; i < probs.size(); ++i) {
        if (probs(i) < NumericalConstants::MIN_PROBABILITY && probs(i) > 0.0) {
            hasUnderflow = true;
            break;
        }
    }
    
    if (!hasUnderflow) {
        return true;
    }
    
    switch (strategy) {
        case RecoveryStrategy::STRICT:
            return false; // Don't recover, let caller handle
            
        case RecoveryStrategy::GRACEFUL:
            // Clamp small values to minimum
            for (std::size_t i = 0; i < probs.size(); ++i) {
                if (probs(i) < NumericalConstants::MIN_PROBABILITY) {
                    probs(i) = NumericalConstants::MIN_PROBABILITY;
                }
            }
            break;
            
        case RecoveryStrategy::ROBUST:
            // More aggressive recovery with renormalization
            for (std::size_t i = 0; i < probs.size(); ++i) {
                probs(i) = NumericalSafety::clampProbability(probs(i));
            }
            NumericalSafety::normalizeProbabilities(probs);
            break;
            
        case RecoveryStrategy::ADAPTIVE:
            // Choose strategy based on severity
            double minVal = *std::min_element(probs.begin(), probs.end());
            if (minVal < NumericalConstants::MIN_PROBABILITY * 1e-10) {
                return recoverFromUnderflow(probs, RecoveryStrategy::ROBUST);
            } else {
                return recoverFromUnderflow(probs, RecoveryStrategy::GRACEFUL);
            }
    }
    
    return true;
}

bool ErrorRecovery::recoverFromOverflow(Vector& values, RecoveryStrategy strategy) {
    bool hasOverflow = false;
    
    // Detect overflow
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values(i)) || values(i) > 1e100) {
            hasOverflow = true;
            break;
        }
    }
    
    if (!hasOverflow) {
        return true;
    }
    
    switch (strategy) {
        case RecoveryStrategy::STRICT:
            return false;
            
        case RecoveryStrategy::GRACEFUL:
        case RecoveryStrategy::ROBUST:
        case RecoveryStrategy::ADAPTIVE:
            // Clamp large values
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (!std::isfinite(values(i)) || values(i) > 1e100) {
                    values(i) = 1e100;
                }
                if (values(i) < -1e100) {
                    values(i) = -1e100;
                }
            }
            break;
    }
    
    return true;
}

std::size_t ErrorRecovery::handleNaNValues(Vector& values, RecoveryStrategy strategy) {
    std::size_t nanCount = 0;
    
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::isnan(values(i))) {
            nanCount++;
            
            switch (strategy) {
                case RecoveryStrategy::STRICT:
                    return nanCount; // Don't fix, return count
                    
                case RecoveryStrategy::GRACEFUL:
                    values(i) = 0.0; // Replace with zero
                    break;
                    
                case RecoveryStrategy::ROBUST:
                    // Replace with average of neighbors if possible
                    if (i > 0 && i < values.size() - 1 && 
                        std::isfinite(values(i-1)) && std::isfinite(values(i+1))) {
                        values(i) = (values(i-1) + values(i+1)) / 2.0;
                    } else {
                        values(i) = 0.0;
                    }
                    break;
                    
                case RecoveryStrategy::ADAPTIVE:
                    values(i) = 0.0;
                    break;
            }
        }
    }
    
    return nanCount;
}

bool ErrorRecovery::repairDegenerateMatrix(Matrix& matrix, RecoveryStrategy strategy) {
    bool hasIssues = false;
    
    // Check for problems
    for (std::size_t i = 0; i < matrix.size1(); ++i) {
        for (std::size_t j = 0; j < matrix.size2(); ++j) {
            if (!std::isfinite(matrix(i, j))) {
                hasIssues = true;
                break;
            }
        }
        if (hasIssues) break;
    }
    
    if (!hasIssues) {
        return true;
    }
    
    switch (strategy) {
        case RecoveryStrategy::STRICT:
            return false;
            
        case RecoveryStrategy::GRACEFUL:
        case RecoveryStrategy::ROBUST:
        case RecoveryStrategy::ADAPTIVE:
            // Replace non-finite values with small positive values
            for (std::size_t i = 0; i < matrix.size1(); ++i) {
                for (std::size_t j = 0; j < matrix.size2(); ++j) {
                    if (!std::isfinite(matrix(i, j))) {
                        matrix(i, j) = NumericalConstants::MIN_PROBABILITY;
                    }
                }
            }
            break;
    }
    
    return true;
}

ErrorRecovery::RecoveryStrategy ErrorRecovery::getRecommendedStrategy(std::size_t sequenceLength, 
                                                                     std::size_t numStates, 
                                                                     bool isRealTime) noexcept {
    if (isRealTime) {
        return RecoveryStrategy::ROBUST; // Fast recovery for real-time
    }
    
    if (sequenceLength > 1000 || numStates > 50) {
        return RecoveryStrategy::GRACEFUL; // Large problems need stability
    }
    
    return RecoveryStrategy::ADAPTIVE; // Default to adaptive
}

// NumericalDiagnostics implementation
NumericalDiagnostics::DiagnosticReport NumericalDiagnostics::analyzeVector(const Vector& vector, const std::string& name) {
    DiagnosticReport report;
    report.problemSize = vector.size();
    
    for (std::size_t i = 0; i < vector.size(); ++i) {
        double val = vector(i);
        
        if (std::isnan(val)) {
            report.hasNaN = true;
        } else if (std::isinf(val)) {
            report.hasInfinity = true;
        } else if (std::isfinite(val)) {
            report.minValue = std::min(report.minValue, val);
            report.maxValue = std::max(report.maxValue, val);
            
            if (val < NumericalConstants::MIN_PROBABILITY && val > 0.0) {
                report.hasUnderflow = true;
            }
            if (val > 1e100) {
                report.hasOverflow = true;
            }
        }
    }
    
    // Generate recommendations
    std::ostringstream oss;
    if (report.hasNaN) {
        oss << "NaN values detected in " << name << ". ";
    }
    if (report.hasInfinity) {
        oss << "Infinite values detected in " << name << ". ";
    }
    if (report.hasUnderflow) {
        oss << "Underflow detected in " << name << ". Consider using scaled arithmetic. ";
    }
    if (report.hasOverflow) {
        oss << "Overflow detected in " << name << ". Consider using log-space arithmetic. ";
    }
    
    report.recommendations = oss.str();
    
    return report;
}

NumericalDiagnostics::DiagnosticReport NumericalDiagnostics::analyzeMatrix(const Matrix& matrix, const std::string& name) {
    DiagnosticReport report;
    report.problemSize = matrix.size1() * matrix.size2();
    
    for (std::size_t i = 0; i < matrix.size1(); ++i) {
        for (std::size_t j = 0; j < matrix.size2(); ++j) {
            double val = matrix(i, j);
            
            if (std::isnan(val)) {
                report.hasNaN = true;
            } else if (std::isinf(val)) {
                report.hasInfinity = true;
            } else if (std::isfinite(val)) {
                report.minValue = std::min(report.minValue, val);
                report.maxValue = std::max(report.maxValue, val);
                
                if (val < NumericalConstants::MIN_PROBABILITY && val > 0.0) {
                    report.hasUnderflow = true;
                }
                if (val > 1e100) {
                    report.hasOverflow = true;
                }
            }
        }
    }
    
    // Estimate condition number (simplified)
    report.conditionNumber = estimateConditionNumber(matrix);
    
    // Generate recommendations
    std::ostringstream oss;
    if (report.hasNaN) {
        oss << "NaN values detected in " << name << ". ";
    }
    if (report.hasInfinity) {
        oss << "Infinite values detected in " << name << ". ";
    }
    if (report.hasUnderflow) {
        oss << "Underflow detected in " << name << ". Consider using scaled arithmetic. ";
    }
    if (report.hasOverflow) {
        oss << "Overflow detected in " << name << ". Consider using log-space arithmetic. ";
    }
    if (report.conditionNumber > 1e12) {
        oss << "High condition number (" << report.conditionNumber << ") in " << name << ". Matrix may be ill-conditioned. ";
    }
    
    report.recommendations = oss.str();
    
    return report;
}

std::string NumericalDiagnostics::generateHealthReport(const std::vector<DiagnosticReport>& reports) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "=== Numerical Health Report ===\n\n";
    
    bool hasIssues = false;
    for (const auto& report : reports) {
        if (report.hasNaN || report.hasInfinity || report.hasUnderflow || 
            report.hasOverflow || report.conditionNumber > 1e12) {
            hasIssues = true;
            break;
        }
    }
    
    if (!hasIssues) {
        oss << "✅ All systems healthy - no numerical issues detected.\n";
    } else {
        oss << "⚠️  Numerical issues detected:\n\n";
        
        for (const auto& report : reports) {
            if (!report.recommendations.empty()) {
                oss << "• " << report.recommendations << "\n";
            }
        }
    }
    
    oss << "\nOverall Statistics:\n";
    oss << "  Total components analyzed: " << reports.size() << "\n";
    
    double avgCondition = 0.0;
    std::size_t totalSize = 0;
    for (const auto& report : reports) {
        avgCondition += report.conditionNumber;
        totalSize += report.problemSize;
    }
    if (!reports.empty()) {
        avgCondition /= reports.size();
    }
    
    oss << "  Average condition number: " << avgCondition << "\n";
    oss << "  Total problem size: " << totalSize << "\n";
    
    return oss.str();
}

double NumericalDiagnostics::estimateConditionNumber(const Matrix& matrix) noexcept {
    if (matrix.size1() != matrix.size2()) {
        return 1.0; // Not applicable for non-square matrices
    }
    
    // Simple estimate: ratio of max to min diagonal elements
    double maxDiag = 0.0;
    double minDiag = std::numeric_limits<double>::max();
    
    for (std::size_t i = 0; i < matrix.size1(); ++i) {
        double val = std::abs(matrix(i, i));
        if (val > 0.0) {
            maxDiag = std::max(maxDiag, val);
            minDiag = std::min(minDiag, val);
        }
    }
    
    if (minDiag == 0.0 || !std::isfinite(minDiag) || !std::isfinite(maxDiag)) {
        return std::numeric_limits<double>::infinity();
    }
    
    return maxDiag / minDiag;
}

} // namespace numerical
} // namespace libhmm
