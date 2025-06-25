#include "libhmm/training/robust_viterbi_trainer.h"
#include "libhmm/calculators/scaled_simd_forward_backward_calculator.h"
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

namespace libhmm {

// Constructor with config
RobustViterbiTrainer::RobustViterbiTrainer(Hmm* hmm, 
                                          const ObservationLists& obsLists,
                                          const RobustTrainingConfig& config)
    : ViterbiTrainer(hmm, obsLists), 
      config_(config),
      recoveryStrategy_(config.enableErrorRecovery ?
        numerical::ErrorRecovery::RecoveryStrategy::GRACEFUL :
        numerical::ErrorRecovery::RecoveryStrategy::STRICT) {
    
    // Validate inputs
    if (!hmm) {
        throw std::invalid_argument("HMM pointer cannot be null");
    }
    if (obsLists.empty()) {
        throw std::invalid_argument("Observation lists cannot be empty");
    }
    
    validateConfiguration();
    validateAndPreprocessData(obsLists);
    
    // Initialize numerical stability components
    convergenceDetector_ = std::make_unique<numerical::ConvergenceDetector>(
        config_.convergenceTolerance,
        config_.maxIterations,
        config_.convergenceWindow
    );
    
    if (config_.enableAdaptivePrecision) {
        auto [dataRange, maxSequenceLength] = analyzeDataCharacteristics(obsLists);
        adaptivePrecision_ = std::make_unique<numerical::AdaptivePrecision>(
            config_.convergenceTolerance,
            maxSequenceLength,
            true
        );
        adaptivePrecision_->scaleForData(dataRange, maxSequenceLength);
    }
    
    // Clear diagnostic reports
    diagnosticReports_.clear();
}

// Constructor with default config
RobustViterbiTrainer::RobustViterbiTrainer(Hmm* hmm, const ObservationLists& obsLists)
    : RobustViterbiTrainer(hmm, obsLists, training_presets::balanced()) {
}

void RobustViterbiTrainer::validateConfiguration() const {
    if (config_.convergenceTolerance <= 0.0 || config_.convergenceTolerance >= 1.0) {
        throw std::runtime_error("Convergence tolerance must be in range (0, 1)");
    }
    if (config_.maxIterations == 0) {
        throw std::runtime_error("Maximum iterations must be greater than 0");
    }
    if (config_.convergenceWindow == 0) {
        throw std::runtime_error("Convergence window must be greater than 0");
    }
}

void RobustViterbiTrainer::validateAndPreprocessData(const ObservationLists& obsLists) {
    std::size_t totalObservations = 0;
    double minValue = std::numeric_limits<double>::max();
    double maxValue = std::numeric_limits<double>::lowest();
    bool hasInvalidValues = false;
    
    for (const auto& obsSet : obsLists) {
        if (obsSet.size() == 0) {
            throw std::runtime_error("Empty observation sequence detected");
        }
        
        totalObservations += obsSet.size();
        
        for (std::size_t i = 0; i < obsSet.size(); ++i) {
            double value = obsSet(i);
            
            if (!std::isfinite(value)) {
                hasInvalidValues = true;
                if (config_.enableErrorRecovery) {
                    if (std::isnan(value)) {
                        std::cerr << "Warning: NaN value detected at sequence " << obsLists.size() 
                                  << ", position " << i << ". Will attempt recovery." << std::endl;
                    } else {
                        std::cerr << "Warning: Infinite value detected at sequence " << obsLists.size() 
                                  << ", position " << i << ". Will attempt recovery." << std::endl;
                    }
                } else {
                    throw std::runtime_error("Invalid observation value detected (NaN or Inf)");
                }
            } else {
                minValue = std::min(minValue, value);
                maxValue = std::max(maxValue, value);
            }
        }
    }
    
    if (totalObservations < static_cast<std::size_t>(hmm_->getNumStates())) {
        throw std::runtime_error("Total observations must be at least equal to number of states");
    }
    
    if (hasInvalidValues && !config_.enableErrorRecovery) {
        throw std::runtime_error("Invalid values detected in observation data");
    }
    
    // Log data characteristics
    if (config_.enableProgressReporting) {
        std::cout << "Data validation complete:" << std::endl;
        std::cout << "  Total observations: " << totalObservations << std::endl;
        std::cout << "  Value range: [" << minValue << ", " << maxValue << "]" << std::endl;
        if (hasInvalidValues) {
            std::cout << "  Warning: Invalid values detected (recovery enabled)" << std::endl;
        }
    }
}

std::pair<double, std::size_t> RobustViterbiTrainer::analyzeDataCharacteristics(const ObservationLists& obsLists) const {
    double minValue = std::numeric_limits<double>::max();
    double maxValue = std::numeric_limits<double>::lowest();
    std::size_t maxSequenceLength = 0;
    std::size_t validObservations = 0;
    
    for (const auto& obsSet : obsLists) {
        maxSequenceLength = std::max(maxSequenceLength, obsSet.size());
        
        for (std::size_t i = 0; i < obsSet.size(); ++i) {
            double value = obsSet(i);
            if (std::isfinite(value)) {
                minValue = std::min(minValue, value);
                maxValue = std::max(maxValue, value);
                validObservations++;
            }
        }
    }
    
    double dataRange = (validObservations > 0) ? (maxValue - minValue) : 1.0;
    return std::make_pair(dataRange, maxSequenceLength);
}

void RobustViterbiTrainer::train() {
    auto startTime = std::chrono::steady_clock::now();
    
    try {
        if (config_.enableProgressReporting) {
            std::cout << "Starting robust Viterbi training..." << std::endl;
            std::cout << "Configuration:" << std::endl;
            std::cout << "  Convergence tolerance: " << config_.convergenceTolerance << std::endl;
            std::cout << "  Maximum iterations: " << config_.maxIterations << std::endl;
            std::cout << "  Adaptive precision: " << (config_.enableAdaptivePrecision ? "enabled" : "disabled") << std::endl;
            std::cout << "  Error recovery: " << (config_.enableErrorRecovery ? "enabled" : "disabled") << std::endl;
        }
        
        // Reset convergence detector for this training run
        convergenceDetector_->reset();
        diagnosticReports_.clear();
        
        // Call the enhanced training loop
        robustTrainingLoop();
        
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        if (config_.enableProgressReporting) {
            std::cout << "\nTraining completed in " << duration.count() << "ms" << std::endl;
            std::cout << getConvergenceReport() << std::endl;
            
            if (config_.enableDiagnostics) {
                std::cout << "\nNumerical health report:" << std::endl;
                std::cout << getNumericalHealthReport() << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        if (config_.enableErrorRecovery) {
            std::cerr << "Training error: " << e.what() << std::endl;
            handleNumericalIssue("Training failure", true);
        } else {
            throw;
        }
    }
}

void RobustViterbiTrainer::robustTrainingLoop() {
    // Step 1: Initialize (delegate to parent class for basic setup)
    ObservationSet s = flattenObservationLists();
    const auto numStates = static_cast<std::size_t>(hmm_->getNumStates());
    
    // Validate flattened observations
    if (config_.enableDiagnostics) {
        // Create vector from observation data for diagnostics
        Vector obsVector(s.size());
        for (std::size_t i = 0; i < s.size(); ++i) {
            obsVector(i) = s(i);
        }
        
        auto vectorReport = numerical::NumericalDiagnostics::analyzeVector(obsVector, "observations");
        diagnosticReports_.push_back(vectorReport);
        
        if (vectorReport.hasNaN || vectorReport.hasInfinity) {
            if (!handleNumericalIssue("Invalid values in observations", config_.enableErrorRecovery)) {
                throw std::runtime_error("Cannot proceed with invalid observation values");
            }
        }
    }
    
    std::size_t iteration = 0;
    double previousLogLikelihood = -std::numeric_limits<double>::infinity();
    
    if (config_.enableProgressReporting) {
        std::cout << "Training for " << s.size() << " observations across " 
                  << numStates << " states" << std::endl;
    }
    
    // Perform initial clustering (similar to parent, but with robust handling)
    // Step 2: Initialize clusters
    if (config_.enableProgressReporting) {
        std::cout << "Initializing clusters..." << std::endl;
    }
    
    quickSort(s); // Use parent's sorting for now
    
    for (std::size_t i = 0; i < numStates; ++i) {
        const auto j = ((i + 1) * (s.size() - 1)) / (numStates + 1);
        clusters_[i].init(s(j));
        
        if (config_.enableProgressReporting) {
            std::cout << "  Cluster " << i << ": centroid = " << clusters_[i].getCentroidValue() << std::endl;
        }
    }
    
    // Step 3: Initial assignment
    if (config_.enableProgressReporting) {
        std::cout << "Performing initial cluster assignments..." << std::endl;
    }
    
    for (std::size_t i = 0; i < obsLists_.size(); ++i) {
        const ObservationSet& os = obsLists_.at(i);
        for (std::size_t l = 0; l < os.size(); ++l) {
            auto clusterIndex = findClosestCluster(os(l));
            clusters_[clusterIndex].batchAdd(os(l));
            associateObservation(i, l, clusterIndex);
        }
    }
    
    // Recalculate centroids
    for (std::size_t i = 0; i < numStates; ++i) {
        clusters_[i].recalculateCentroid();
    }
    
    // Main training loop
    do {
        iteration++;
        auto iterationStart = std::chrono::steady_clock::now();
        
        // Update adaptive precision if enabled
        if (config_.enableAdaptivePrecision && adaptivePrecision_) {
            double convergenceRate = iteration > 1 ? 
                std::abs(previousLogLikelihood - (-std::numeric_limits<double>::infinity())) : 1.0;
            adaptivePrecision_->updateTolerance(iteration, convergenceRate);
        }
        
        // Steps 4 and 5: Calculate pi and transition matrices with robust handling
        if (config_.enableProgressReporting) {
            std::cout << "\rIteration " << iteration << ": Computing parameters...";
            std::cout.flush();
        }
        
        robustCalculatePi();
        robustCalculateTrans();
        
        // Step 6: Robust cluster fitting
        bool fittingSuccessful = true;
        for (std::size_t i = 0; i < numStates; ++i) {
            if (!robustClusterFitting(i)) {
                fittingSuccessful = false;
                if (!config_.enableErrorRecovery) {
                    throw std::runtime_error("Cluster fitting failed for cluster " + std::to_string(i));
                }
            }
        }
        
        if (!fittingSuccessful && config_.enableErrorRecovery) {
            if (!handleNumericalIssue("Cluster fitting issues", true)) {
                break; // Stop training if recovery fails
            }
        }
        
        // Step 7: Viterbi decoding and reassignment
        numChanges_ = 0;
        std::vector<double> probabilities;
        
        for (std::size_t i = 0; i < obsLists_.size(); ++i) {
            const ObservationSet& os = obsLists_.at(i);
            
            try {
                ViterbiCalculator vc(hmm_, os);
                StateSequence sequence = vc.decode();
                compareSequence(sequence, i);
                
                ScaledSIMDForwardBackwardCalculator sfbc(hmm_, os);
                double logProbability = sfbc.getLogProbability();
                
                // Validate log probability
                if (std::isfinite(logProbability)) {
                    probabilities.push_back(logProbability);
                } else if (config_.enableErrorRecovery) {
                    std::cerr << "Warning: Invalid log probability for sequence " << i 
                              << ", using previous value" << std::endl;
                    if (!probabilities.empty()) {
                        probabilities.push_back(probabilities.back());
                    } else {
                        probabilities.push_back(-1000.0); // Fallback value
                    }
                }
            } catch (const std::exception& e) {
                if (config_.enableErrorRecovery) {
                    std::cerr << "Warning: Error processing sequence " << i << ": " << e.what() << std::endl;
                    probabilities.push_back(-1000.0); // Fallback value
                } else {
                    throw;
                }
            }
        }
        
        // Calculate average log likelihood
        double currentLogLikelihood = -std::numeric_limits<double>::infinity();
        if (!probabilities.empty()) {
            try {
                GaussianDistribution gd;
                gd.fit(probabilities);
                currentLogLikelihood = gd.getMean();
            } catch (const std::exception& e) {
                if (config_.enableErrorRecovery) {
                    // Calculate simple mean as fallback
                    currentLogLikelihood = std::accumulate(probabilities.begin(), probabilities.end(), 0.0) / probabilities.size();
                    std::cerr << "Warning: Gaussian fitting failed, using arithmetic mean: " << e.what() << std::endl;
                } else {
                    throw;
                }
            }
        }
        
        // Record timing
        auto iterationEnd = std::chrono::steady_clock::now();
        auto iterationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(iterationEnd - iterationStart);
        
        // Progress reporting
        if (config_.enableProgressReporting) {
            std::cout << "\rIteration " << std::setw(3) << iteration 
                      << ": Changes=" << std::setw(4) << numChanges_
                      << ", LogL=" << std::setw(8) << std::fixed << std::setprecision(2) << currentLogLikelihood
                      << ", Time=" << std::setw(4) << iterationDuration.count() << "ms";
            if (config_.enableAdaptivePrecision && adaptivePrecision_) {
                std::cout << ", Tol=" << std::scientific << std::setprecision(1) << adaptivePrecision_->getCurrentTolerance();
            }
            std::cout.flush();
        }
        
        // Check convergence
        bool converged = convergenceDetector_->addValue(currentLogLikelihood);
        
        // Validate HMM state
        if (config_.enableDiagnostics && !validateHmmState()) {
            if (!handleNumericalIssue("Invalid HMM state", config_.enableErrorRecovery)) {
                break;
            }
        }
        
        // Check termination conditions
        terminated_ = (numChanges_ == 0) || converged;
        
        if (convergenceDetector_->isMaxIterationsReached()) {
            if (config_.enableProgressReporting) {
                std::cout << "\nMaximum iterations reached." << std::endl;
            }
            break;
        }
        
        previousLogLikelihood = currentLogLikelihood;
        
    } while (!terminated_);
    
    if (config_.enableProgressReporting) {
        std::cout << std::endl;
    }
}

bool RobustViterbiTrainer::robustClusterFitting(std::size_t clusterIndex) {
    if (!validateClusterForFitting(clusterIndex)) {
        return false;
    }
    
    std::vector<Observation> clusterObservations = clusters_[clusterIndex].getObservations();
    
    try {
        // Validate observations before fitting
        bool hasValidObservations = false;
        for (auto& obs : clusterObservations) {
            if (std::isfinite(obs)) {
                hasValidObservations = true;
            } else if (config_.enableErrorRecovery) {
                // Replace invalid observations with cluster centroid
                obs = clusters_[clusterIndex].getCentroidValue();
            }
        }
        
        if (!hasValidObservations) {
            if (config_.enableErrorRecovery) {
                std::cerr << "Warning: No valid observations in cluster " << clusterIndex 
                          << ", skipping fitting." << std::endl;
                return false;
            } else {
                throw std::runtime_error("No valid observations in cluster");
            }
        }
        
        ProbabilityDistribution& distribution = 
            *hmm_->getProbabilityDistribution(static_cast<int>(clusterIndex));
        distribution.fit(clusterObservations);
        
        // Validate fitted distribution
        if (config_.enableDiagnostics) {
            // Add basic validation - more sophisticated validation could be added
            auto* gaussDist = dynamic_cast<GaussianDistribution*>(&distribution);
            if (gaussDist) {
                double mean = gaussDist->getMean();
                double stddev = gaussDist->getStandardDeviation();
                
                if (!std::isfinite(mean) || !std::isfinite(stddev) || stddev <= 0) {
                    if (config_.enableErrorRecovery) {
                        std::cerr << "Warning: Invalid distribution parameters for cluster " 
                                  << clusterIndex << std::endl;
                        return false;
                    } else {
                        throw std::runtime_error("Invalid distribution parameters");
                    }
                }
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        if (config_.enableErrorRecovery) {
            std::cerr << "Error fitting distribution for cluster " << clusterIndex 
                      << ": " << e.what() << std::endl;
            return false;
        } else {
            throw;
        }
    }
}

bool RobustViterbiTrainer::validateClusterForFitting(std::size_t clusterIndex) const {
    if (clusterIndex >= static_cast<std::size_t>(hmm_->getNumStates())) {
        return false;
    }
    
    if (clusters_[clusterIndex].size() == 0) {
        if (config_.enableProgressReporting) {
            std::cerr << "Warning: Cluster " << clusterIndex << " is empty." << std::endl;
        }
        return false;
    }
    
    return true;
}

void RobustViterbiTrainer::robustCalculatePi() {
    try {
        calculatePi(); // Delegate to parent implementation
        
        if (config_.enableDiagnostics) {
            // Validate pi vector
            Vector pi = hmm_->getPi();
            auto piReport = numerical::NumericalDiagnostics::analyzeVector(pi, "pi_vector");
            diagnosticReports_.push_back(piReport);
            
            if (piReport.hasNaN || piReport.hasInfinity) {
                if (!handleNumericalIssue("Invalid values in pi vector", config_.enableErrorRecovery)) {
                    throw std::runtime_error("Pi vector contains invalid values");
                }
            }
            
            // Check if probabilities sum to 1
            if (!numerical::NumericalSafety::isProbabilityDistribution(pi)) {
                if (config_.enableErrorRecovery) {
                    Vector piCopy = pi;
                    if (numerical::NumericalSafety::normalizeProbabilities(piCopy)) {
                        hmm_->setPi(piCopy);
                        std::cerr << "Warning: Pi vector was not properly normalized, fixed." << std::endl;
                    }
                } else {
                    throw std::runtime_error("Pi vector is not a valid probability distribution");
                }
            }
        }
        
    } catch (const std::exception& e) {
        if (config_.enableErrorRecovery) {
            handleNumericalIssue("Error calculating pi vector: " + std::string(e.what()), true);
        } else {
            throw;
        }
    }
}

void RobustViterbiTrainer::robustCalculateTrans() {
    try {
        calculateTrans(); // Delegate to parent implementation
        
        if (config_.enableDiagnostics) {
            // Validate transition matrix
            Matrix trans = hmm_->getTrans();
            auto transReport = numerical::NumericalDiagnostics::analyzeMatrix(trans, "transition_matrix");
            diagnosticReports_.push_back(transReport);
            
            if (transReport.hasNaN || transReport.hasInfinity) {
                if (!handleNumericalIssue("Invalid values in transition matrix", config_.enableErrorRecovery)) {
                    throw std::runtime_error("Transition matrix contains invalid values");
                }
            }
            
            // Check if each row sums to 1
            for (std::size_t i = 0; i < trans.size1(); ++i) {
                Vector row(trans.size2());
                for (std::size_t j = 0; j < trans.size2(); ++j) {
                    row(j) = trans(i, j);
                }
                
                if (!numerical::NumericalSafety::isProbabilityDistribution(row)) {
                    if (config_.enableErrorRecovery) {
                        if (numerical::NumericalSafety::normalizeProbabilities(row)) {
                            for (std::size_t j = 0; j < trans.size2(); ++j) {
                                trans(i, j) = row(j);
                            }
                            std::cerr << "Warning: Transition matrix row " << i 
                                      << " was not properly normalized, fixed." << std::endl;
                        }
                    } else {
                        throw std::runtime_error("Transition matrix row " + std::to_string(i) + 
                                                " is not a valid probability distribution");
                    }
                }
            }
            
            hmm_->setTrans(trans);
        }
        
    } catch (const std::exception& e) {
        if (config_.enableErrorRecovery) {
            handleNumericalIssue("Error calculating transition matrix: " + std::string(e.what()), true);
        } else {
            throw;
        }
    }
}

bool RobustViterbiTrainer::validateHmmState() const {
    try {
        hmm_->validate(); // Use HMM's built-in validation
        return true;
    } catch (const std::exception& e) {
        if (config_.enableProgressReporting) {
            std::cerr << "HMM validation failed: " << e.what() << std::endl;
        }
        return false;
    }
}

bool RobustViterbiTrainer::handleNumericalIssue(const std::string& issueDescription, bool canRecover) {
    if (config_.enableProgressReporting) {
        std::cerr << "Numerical issue detected: " << issueDescription << std::endl;
    }
    
    if (!canRecover || recoveryStrategy_ == numerical::ErrorRecovery::RecoveryStrategy::STRICT) {
        return false;
    }
    
    // Attempt recovery based on strategy
    switch (recoveryStrategy_) {
        case numerical::ErrorRecovery::RecoveryStrategy::GRACEFUL:
            if (config_.enableProgressReporting) {
                std::cerr << "Attempting graceful recovery..." << std::endl;
            }
            // Could implement specific recovery actions here
            return true;
            
        case numerical::ErrorRecovery::RecoveryStrategy::ROBUST:
            if (config_.enableProgressReporting) {
                std::cerr << "Attempting robust recovery..." << std::endl;
            }
            // Could implement more aggressive recovery actions here
            return true;
            
        case numerical::ErrorRecovery::RecoveryStrategy::ADAPTIVE:
            // Choose strategy based on problem characteristics
            // For ADAPTIVE, default to GRACEFUL recovery to avoid infinite recursion
            if (config_.enableProgressReporting) {
                std::cerr << "Attempting adaptive recovery (using graceful strategy)..." << std::endl;
            }
            return true;
            
        default:
            return false;
    }
}

// Public interface methods

std::string RobustViterbiTrainer::getConvergenceReport() const {
    if (!convergenceDetector_) {
        return "Convergence detector not initialized";
    }
    
    std::ostringstream report;
    report << "Convergence Report:" << std::endl;
    report << "  Iterations: " << convergenceDetector_->getCurrentIteration() << std::endl;
    report << "  Max iterations reached: " << (convergenceDetector_->isMaxIterationsReached() ? "Yes" : "No") << std::endl;
    report << "  Converged: " << (hasConverged() ? "Yes" : "No") << std::endl;
    
    if (config_.enableAdaptivePrecision && adaptivePrecision_) {
        report << "  Final tolerance: " << adaptivePrecision_->getCurrentTolerance() << std::endl;
    }
    
    report << convergenceDetector_->getConvergenceReport();
    
    return report.str();
}

std::string RobustViterbiTrainer::getNumericalHealthReport() const {
    if (diagnosticReports_.empty()) {
        return "No diagnostic information available";
    }
    
    return numerical::NumericalDiagnostics::generateHealthReport(diagnosticReports_);
}

bool RobustViterbiTrainer::hasConverged() const {
    return convergenceDetector_ && 
           convergenceDetector_->getCurrentIteration() > 0 && 
           !convergenceDetector_->isMaxIterationsReached() &&
           terminated_;
}

bool RobustViterbiTrainer::reachedMaxIterations() const {
    return convergenceDetector_ && convergenceDetector_->isMaxIterationsReached();
}

const RobustViterbiTrainer::RobustTrainingConfig& RobustViterbiTrainer::getConfig() const noexcept {
    return config_;
}

void RobustViterbiTrainer::setConfig(const RobustTrainingConfig& config) {
    config_ = config;
    validateConfiguration();
    
    // Reinitialize convergence detector with new parameters
    if (convergenceDetector_) {
        convergenceDetector_ = std::make_unique<numerical::ConvergenceDetector>(
            config_.convergenceTolerance,
            config_.maxIterations,
            config_.convergenceWindow
        );
    }
}

void RobustViterbiTrainer::setRecoveryStrategy(numerical::ErrorRecovery::RecoveryStrategy strategy) {
    recoveryStrategy_ = strategy;
    config_.enableErrorRecovery = (strategy != numerical::ErrorRecovery::RecoveryStrategy::STRICT);
}

numerical::ErrorRecovery::RecoveryStrategy RobustViterbiTrainer::getRecoveryStrategy() const noexcept {
    return recoveryStrategy_;
}

void RobustViterbiTrainer::setAdaptivePrecisionEnabled(bool enable) {
    config_.enableAdaptivePrecision = enable;
    
    if (enable && !adaptivePrecision_) {
        // Initialize adaptive precision if not already done
        adaptivePrecision_ = std::make_unique<numerical::AdaptivePrecision>(
            config_.convergenceTolerance, 1, true);
    }
}

bool RobustViterbiTrainer::isAdaptivePrecisionEnabled() const noexcept {
    return config_.enableAdaptivePrecision;
}

double RobustViterbiTrainer::getCurrentTolerance() const {
    if (config_.enableAdaptivePrecision && adaptivePrecision_) {
        return adaptivePrecision_->getCurrentTolerance();
    }
    return config_.convergenceTolerance;
}

void RobustViterbiTrainer::requestEarlyTermination() {
    terminated_ = true;
}

void RobustViterbiTrainer::resetTrainingState() {
    terminated_ = false;
    if (convergenceDetector_) {
        convergenceDetector_->reset();
    }
    diagnosticReports_.clear();
}

std::unique_ptr<RobustViterbiTrainer> RobustViterbiTrainer::createWithRecommendedSettings(
    Hmm* hmm, const ObservationLists& obsLists, const std::string& problemType) {
    
    RobustTrainingConfig config;
    
    if (problemType == "conservative") {
        config = training_presets::conservative();
    } else if (problemType == "aggressive") {
        config = training_presets::aggressive();
    } else if (problemType == "realtime") {
        config = training_presets::realtime();
    } else if (problemType == "high_precision") {
        config = training_presets::highPrecision();
    } else {
        config = training_presets::balanced();
    }
    
    return std::make_unique<RobustViterbiTrainer>(hmm, obsLists, config);
}

std::vector<std::string> RobustViterbiTrainer::getPerformanceRecommendations() const {
    std::vector<std::string> recommendations;
    
    if (reachedMaxIterations()) {
        recommendations.push_back("Consider increasing maximum iterations or relaxing convergence tolerance");
    }
    
    if (!hasConverged()) {
        recommendations.push_back("Training did not converge - consider different initialization or data preprocessing");
    }
    
    if (config_.enableAdaptivePrecision && adaptivePrecision_) {
        double finalTolerance = adaptivePrecision_->getCurrentTolerance();
        if (finalTolerance > config_.convergenceTolerance * 10) {
            recommendations.push_back("Adaptive precision significantly relaxed tolerance - consider data quality improvement");
        }
    }
    
    if (!diagnosticReports_.empty()) {
        bool hasNumericalIssues = std::any_of(diagnosticReports_.begin(), diagnosticReports_.end(),
            [](const auto& report) { return report.hasNaN || report.hasInfinity || report.hasUnderflow; });
        
        if (hasNumericalIssues) {
            recommendations.push_back("Numerical issues detected - consider data normalization or different algorithms");
        }
    }
    
    return recommendations;
}

// Training presets implementation
namespace training_presets {

RobustViterbiTrainer::RobustTrainingConfig conservative() {
    RobustViterbiTrainer::RobustTrainingConfig config;
    config.enableAdaptivePrecision = true;
    config.enableErrorRecovery = true;
    config.enableDiagnostics = true;
    config.enableProgressReporting = true;
    config.convergenceTolerance = 1e-10;
    config.maxIterations = 5000;
    config.convergenceWindow = 10;
    return config;
}

RobustViterbiTrainer::RobustTrainingConfig balanced() {
    RobustViterbiTrainer::RobustTrainingConfig config;
    config.enableAdaptivePrecision = true;
    config.enableErrorRecovery = true;
    config.enableDiagnostics = true;
    config.enableProgressReporting = true;
    config.convergenceTolerance = 1e-8;
    config.maxIterations = 1000;
    config.convergenceWindow = 5;
    return config;
}

RobustViterbiTrainer::RobustTrainingConfig aggressive() {
    RobustViterbiTrainer::RobustTrainingConfig config;
    config.enableAdaptivePrecision = false;
    config.enableErrorRecovery = true;
    config.enableDiagnostics = false;
    config.enableProgressReporting = false;
    config.convergenceTolerance = 1e-6;
    config.maxIterations = 500;
    config.convergenceWindow = 3;
    return config;
}

RobustViterbiTrainer::RobustTrainingConfig realtime() {
    RobustViterbiTrainer::RobustTrainingConfig config;
    config.enableAdaptivePrecision = false;
    config.enableErrorRecovery = true;
    config.enableDiagnostics = false;
    config.enableProgressReporting = false;
    config.convergenceTolerance = 1e-4;
    config.maxIterations = 100;
    config.convergenceWindow = 3;
    return config;
}

RobustViterbiTrainer::RobustTrainingConfig highPrecision() {
    RobustViterbiTrainer::RobustTrainingConfig config;
    config.enableAdaptivePrecision = true;
    config.enableErrorRecovery = true;
    config.enableDiagnostics = true;
    config.enableProgressReporting = true;
    config.convergenceTolerance = 1e-12;
    config.maxIterations = 10000;
    config.convergenceWindow = 15;
    return config;
}

} // namespace training_presets

} // namespace libhmm
