#include <gtest/gtest.h>
#include "libhmm/common/numerical_stability.h"
#include <limits>
#include <cmath>

using namespace libhmm;
using namespace libhmm::numerical;

class NumericalStabilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test vectors and matrices
        testVector_ = Vector(5);
        testVector_(0) = 1.0;
        testVector_(1) = 0.5;
        testVector_(2) = 0.3;
        testVector_(3) = 0.1;
        testVector_(4) = 0.1;
        
        testMatrix_ = Matrix(3, 3);
        testMatrix_(0, 0) = 0.7; testMatrix_(0, 1) = 0.2; testMatrix_(0, 2) = 0.1;
        testMatrix_(1, 0) = 0.3; testMatrix_(1, 1) = 0.4; testMatrix_(1, 2) = 0.3;
        testMatrix_(2, 0) = 0.1; testMatrix_(2, 1) = 0.3; testMatrix_(2, 2) = 0.6;
    }
    
    Vector testVector_;
    Matrix testMatrix_;
};

// Test NumericalSafety class
TEST_F(NumericalStabilityTest, NumericalSafetyBasicChecks) {
    // Test finite value checking
    EXPECT_NO_THROW(NumericalSafety::checkFinite(1.0));
    EXPECT_NO_THROW(NumericalSafety::checkFinite(-1.0));
    EXPECT_NO_THROW(NumericalSafety::checkFinite(0.0));
    
    EXPECT_THROW(NumericalSafety::checkFinite(std::numeric_limits<double>::infinity()), std::runtime_error);
    EXPECT_THROW(NumericalSafety::checkFinite(-std::numeric_limits<double>::infinity()), std::runtime_error);
    EXPECT_THROW(NumericalSafety::checkFinite(std::numeric_limits<double>::quiet_NaN()), std::runtime_error);
}

TEST_F(NumericalStabilityTest, ProbabilityValidation) {
    // Valid probabilities
    EXPECT_NO_THROW(NumericalSafety::checkProbability(0.0));
    EXPECT_NO_THROW(NumericalSafety::checkProbability(0.5));
    EXPECT_NO_THROW(NumericalSafety::checkProbability(1.0));
    
    // Invalid probabilities
    EXPECT_THROW(NumericalSafety::checkProbability(-0.1), std::runtime_error);
    EXPECT_THROW(NumericalSafety::checkProbability(1.1), std::runtime_error);
    EXPECT_THROW(NumericalSafety::checkProbability(std::numeric_limits<double>::quiet_NaN()), std::runtime_error);
}

TEST_F(NumericalStabilityTest, LogProbabilityValidation) {
    // Valid log probabilities
    EXPECT_NO_THROW(NumericalSafety::checkLogProbability(0.0));
    EXPECT_NO_THROW(NumericalSafety::checkLogProbability(-1.0));
    EXPECT_NO_THROW(NumericalSafety::checkLogProbability(-100.0));
    
    // Invalid log probabilities
    EXPECT_THROW(NumericalSafety::checkLogProbability(0.1), std::runtime_error);
    EXPECT_THROW(NumericalSafety::checkLogProbability(std::numeric_limits<double>::quiet_NaN()), std::runtime_error);
}

TEST_F(NumericalStabilityTest, ProbabilityClamping) {
    // Test probability clamping
    EXPECT_EQ(NumericalSafety::clampProbability(0.5), 0.5);
    EXPECT_EQ(NumericalSafety::clampProbability(1.5), NumericalConstants::MAX_PROBABILITY);
    EXPECT_EQ(NumericalSafety::clampProbability(-0.5), NumericalConstants::MIN_PROBABILITY);
    EXPECT_EQ(NumericalSafety::clampProbability(std::numeric_limits<double>::quiet_NaN()), 
              NumericalConstants::MIN_PROBABILITY);
}

TEST_F(NumericalStabilityTest, SafeLogarithmAndExponential) {
    // Test safe logarithm
    EXPECT_NEAR(NumericalSafety::safeLog(1.0), 0.0, 1e-15);
    EXPECT_NEAR(NumericalSafety::safeLog(std::exp(1.0)), 1.0, 1e-15);
    EXPECT_EQ(NumericalSafety::safeLog(0.0), NumericalConstants::MIN_LOG_PROBABILITY);
    EXPECT_EQ(NumericalSafety::safeLog(-1.0), NumericalConstants::MIN_LOG_PROBABILITY);
    
    // Test safe exponential
    EXPECT_NEAR(NumericalSafety::safeExp(0.0), 1.0, 1e-15);
    // For values that could result in probability > 1, they get clamped
    double exp1 = NumericalSafety::safeExp(1.0);
    EXPECT_GT(exp1, 0.0);
    EXPECT_LE(exp1, NumericalConstants::MAX_PROBABILITY);
    EXPECT_EQ(NumericalSafety::safeExp(1000.0), NumericalConstants::MAX_PROBABILITY);
    EXPECT_EQ(NumericalSafety::safeExp(-1000.0), NumericalConstants::MIN_PROBABILITY);
}

TEST_F(NumericalStabilityTest, VectorFiniteCheck) {
    // Valid vector should pass
    EXPECT_NO_THROW(NumericalSafety::checkVectorFinite(testVector_));
    
    // Create vector with NaN
    Vector nanVector = testVector_;
    nanVector(2) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(NumericalSafety::checkVectorFinite(nanVector), std::runtime_error);
    
    // Create vector with infinity
    Vector infVector = testVector_;
    infVector(2) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(NumericalSafety::checkVectorFinite(infVector), std::runtime_error);
}

TEST_F(NumericalStabilityTest, MatrixFiniteCheck) {
    // Valid matrix should pass
    EXPECT_NO_THROW(NumericalSafety::checkMatrixFinite(testMatrix_));
    
    // Create matrix with NaN
    Matrix nanMatrix = testMatrix_;
    nanMatrix(1, 1) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(NumericalSafety::checkMatrixFinite(nanMatrix), std::runtime_error);
}

TEST_F(NumericalStabilityTest, ProbabilityNormalization) {
    // Test normalization of valid probabilities
    Vector probs = testVector_;
    EXPECT_TRUE(NumericalSafety::normalizeProbabilities(probs));
    
    // Check that probabilities sum to 1
    double sum = 0.0;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        sum += probs(i);
    }
    EXPECT_NEAR(sum, 1.0, 1e-15);
    
    // Test normalization of zero probabilities
    Vector zeroProbs(3);
    zeroProbs(0) = 0.0; zeroProbs(1) = 0.0; zeroProbs(2) = 0.0;
    EXPECT_FALSE(NumericalSafety::normalizeProbabilities(zeroProbs));
    
    // Should be converted to uniform distribution
    for (std::size_t i = 0; i < zeroProbs.size(); ++i) {
        EXPECT_NEAR(zeroProbs(i), 1.0/3.0, 1e-15);
    }
}

TEST_F(NumericalStabilityTest, ProbabilityDistributionCheck) {
    // Test valid probability distribution
    Vector validProbs(3);
    validProbs(0) = 0.3; validProbs(1) = 0.3; validProbs(2) = 0.4;
    EXPECT_TRUE(NumericalSafety::isProbabilityDistribution(validProbs));
    
    // Test invalid probability distribution (doesn't sum to 1)
    Vector invalidProbs(3);
    invalidProbs(0) = 0.3; invalidProbs(1) = 0.3; invalidProbs(2) = 0.3;
    EXPECT_FALSE(NumericalSafety::isProbabilityDistribution(invalidProbs));
    
    // Test with negative probability
    Vector negativeProbs(3);
    negativeProbs(0) = -0.1; negativeProbs(1) = 0.5; negativeProbs(2) = 0.6;
    EXPECT_FALSE(NumericalSafety::isProbabilityDistribution(negativeProbs));
}

// Test ConvergenceDetector class
TEST_F(NumericalStabilityTest, ConvergenceDetectorBasicFunctionality) {
    ConvergenceDetector detector(1e-3, 100, 3);  // Use looser tolerance for more reliable test
    
    // Should not converge with first value
    EXPECT_FALSE(detector.addValue(10.0));
    
    // Should not converge with second value (significant change)
    EXPECT_FALSE(detector.addValue(9.0));
    
    // Add more values with decreasing changes
    EXPECT_FALSE(detector.addValue(8.5));
    
    // Should converge when changes become small enough
    detector.addValue(8.4999);
    detector.addValue(8.4999);
    EXPECT_TRUE(detector.addValue(8.4999));
}

TEST_F(NumericalStabilityTest, ConvergenceDetectorMaxIterations) {
    ConvergenceDetector detector(1e-12, 5, 3); // Very tight tolerance, few iterations
    
    // Add values that don't converge
    for (int i = 0; i < 6; ++i) {
        detector.addValue(static_cast<double>(i));
    }
    
    EXPECT_TRUE(detector.isMaxIterationsReached());
}

TEST_F(NumericalStabilityTest, ConvergenceDetectorReset) {
    ConvergenceDetector detector;
    
    detector.addValue(1.0);
    detector.addValue(2.0);
    EXPECT_EQ(detector.getCurrentIteration(), 2);
    
    detector.reset();
    EXPECT_EQ(detector.getCurrentIteration(), 0);
    EXPECT_TRUE(detector.getHistory().empty());
}

TEST_F(NumericalStabilityTest, ConvergenceDetectorOscillationDetection) {
    ConvergenceDetector detector(1e-6, 100, 5);
    
    // Create oscillating sequence
    detector.addValue(1.0);
    detector.addValue(2.0);
    detector.addValue(1.0);
    detector.addValue(2.0);
    detector.addValue(1.0);
    
    EXPECT_TRUE(detector.isOscillating());
}

// Test AdaptivePrecision class
TEST_F(NumericalStabilityTest, AdaptivePrecisionBasicFunctionality) {
    AdaptivePrecision precision(1e-8, 100, true);
    
    EXPECT_EQ(precision.getCurrentTolerance(), 1e-8);
    
    // Update tolerance based on iteration
    precision.updateTolerance(100, 1.0);
    EXPECT_GT(precision.getCurrentTolerance(), 1e-8); // Should be relaxed
    
    // Scale for data characteristics
    precision.scaleForData(1000.0, 1000);
    EXPECT_GT(precision.getCurrentTolerance(), 1e-8); // Should be further relaxed
}

TEST_F(NumericalStabilityTest, AdaptivePrecisionDisabled) {
    AdaptivePrecision precision(1e-8, 100, false);
    
    double initialTolerance = precision.getCurrentTolerance();
    
    // Updates should have no effect when disabled
    precision.updateTolerance(100, 1.0);
    precision.scaleForData(1000.0, 1000);
    
    EXPECT_EQ(precision.getCurrentTolerance(), initialTolerance);
}

// Test ErrorRecovery class
TEST_F(NumericalStabilityTest, ErrorRecoveryUnderflow) {
    Vector probs(3);
    probs(0) = 1e-301; // Below MIN_PROBABILITY
    probs(1) = 0.5;
    probs(2) = 0.5;
    
    EXPECT_TRUE(ErrorRecovery::recoverFromUnderflow(probs, ErrorRecovery::RecoveryStrategy::GRACEFUL));
    EXPECT_GE(probs(0), NumericalConstants::MIN_PROBABILITY);
}

TEST_F(NumericalStabilityTest, ErrorRecoveryOverflow) {
    Vector values(3);
    values(0) = 1e200;
    values(1) = std::numeric_limits<double>::infinity();
    values(2) = -1e200;
    
    EXPECT_TRUE(ErrorRecovery::recoverFromOverflow(values, ErrorRecovery::RecoveryStrategy::GRACEFUL));
    EXPECT_TRUE(std::isfinite(values(0)));
    EXPECT_TRUE(std::isfinite(values(1)));
    EXPECT_TRUE(std::isfinite(values(2)));
}

TEST_F(NumericalStabilityTest, ErrorRecoveryNaNHandling) {
    Vector values(4);
    values(0) = 1.0;
    values(1) = std::numeric_limits<double>::quiet_NaN();
    values(2) = 2.0;
    values(3) = std::numeric_limits<double>::quiet_NaN();
    
    std::size_t nanCount = ErrorRecovery::handleNaNValues(values, ErrorRecovery::RecoveryStrategy::GRACEFUL);
    EXPECT_EQ(nanCount, 2);
    
    // NaN values should be replaced
    EXPECT_FALSE(std::isnan(values(1)));
    EXPECT_FALSE(std::isnan(values(3)));
}

TEST_F(NumericalStabilityTest, ErrorRecoveryStrategyRecommendation) {
    // Real-time scenario should recommend ROBUST
    auto strategy1 = ErrorRecovery::getRecommendedStrategy(100, 10, true);
    EXPECT_EQ(strategy1, ErrorRecovery::RecoveryStrategy::ROBUST);
    
    // Large problem should recommend GRACEFUL
    auto strategy2 = ErrorRecovery::getRecommendedStrategy(2000, 100, false);
    EXPECT_EQ(strategy2, ErrorRecovery::RecoveryStrategy::GRACEFUL);
    
    // Normal problem should recommend ADAPTIVE
    auto strategy3 = ErrorRecovery::getRecommendedStrategy(100, 10, false);
    EXPECT_EQ(strategy3, ErrorRecovery::RecoveryStrategy::ADAPTIVE);
}

// Test NumericalDiagnostics class
TEST_F(NumericalStabilityTest, NumericalDiagnosticsVector) {
    auto report = NumericalDiagnostics::analyzeVector(testVector_, "test_vector");
    
    EXPECT_FALSE(report.hasNaN);
    EXPECT_FALSE(report.hasInfinity);
    EXPECT_FALSE(report.hasUnderflow);
    EXPECT_FALSE(report.hasOverflow);
    EXPECT_EQ(report.problemSize, testVector_.size());
}

TEST_F(NumericalStabilityTest, NumericalDiagnosticsVectorWithIssues) {
    Vector problematicVector(4);
    problematicVector(0) = std::numeric_limits<double>::quiet_NaN();
    problematicVector(1) = std::numeric_limits<double>::infinity();
    problematicVector(2) = 1e-301; // Underflow
    problematicVector(3) = 1e200;  // Potential overflow
    
    auto report = NumericalDiagnostics::analyzeVector(problematicVector, "problematic_vector");
    
    EXPECT_TRUE(report.hasNaN);
    EXPECT_TRUE(report.hasInfinity);
    EXPECT_TRUE(report.hasUnderflow);
    EXPECT_TRUE(report.hasOverflow);
    EXPECT_FALSE(report.recommendations.empty());
}

TEST_F(NumericalStabilityTest, NumericalDiagnosticsMatrix) {
    auto report = NumericalDiagnostics::analyzeMatrix(testMatrix_, "test_matrix");
    
    EXPECT_FALSE(report.hasNaN);
    EXPECT_FALSE(report.hasInfinity);
    EXPECT_FALSE(report.hasUnderflow);
    EXPECT_FALSE(report.hasOverflow);
    EXPECT_GT(report.conditionNumber, 0.0);
    EXPECT_EQ(report.problemSize, testMatrix_.size1() * testMatrix_.size2());
}

TEST_F(NumericalStabilityTest, NumericalDiagnosticsHealthReport) {
    std::vector<NumericalDiagnostics::DiagnosticReport> reports;
    
    // Add healthy report
    reports.push_back(NumericalDiagnostics::analyzeVector(testVector_, "healthy_vector"));
    
    // Add problematic report
    Vector problematicVector(2);
    problematicVector(0) = std::numeric_limits<double>::quiet_NaN();
    problematicVector(1) = 1.0;
    reports.push_back(NumericalDiagnostics::analyzeVector(problematicVector, "problematic_vector"));
    
    std::string healthReport = NumericalDiagnostics::generateHealthReport(reports);
    
    EXPECT_FALSE(healthReport.empty());
    EXPECT_NE(healthReport.find("Numerical issues detected"), std::string::npos);
    EXPECT_NE(healthReport.find("Total components analyzed: 2"), std::string::npos);
}

// Integration test for multiple components
TEST_F(NumericalStabilityTest, IntegratedNumericalStabilityWorkflow) {
    // Create a workflow that uses multiple components together
    
    // 1. Create adaptive precision manager
    AdaptivePrecision precision(1e-8);
    
    // 2. Create convergence detector
    ConvergenceDetector detector(precision.getCurrentTolerance());
    
    // 3. Simulate training loop with numerical issues
    Vector probs(3);
    probs(0) = 0.33; probs(1) = 0.33; probs(2) = 0.34;
    
    bool converged = false;
    for (int iter = 0; iter < 50 && !converged; ++iter) {
        // Simulate some numerical degradation
        if (iter > 20) {
            probs(0) *= 0.99999; // Slight change
            probs(1) = 1e-301;   // Introduce underflow
        }
        
        // Apply error recovery
        ErrorRecovery::recoverFromUnderflow(probs, ErrorRecovery::RecoveryStrategy::GRACEFUL);
        
        // Normalize probabilities
        NumericalSafety::normalizeProbabilities(probs);
        
        // Update precision based on iteration
        precision.updateTolerance(iter, 1.0);
        
        // Check convergence
        double sum = 0.0;
        for (std::size_t i = 0; i < probs.size(); ++i) {
            sum += probs(i);
        }
        converged = detector.addValue(sum);
        
        if (detector.isMaxIterationsReached()) {
            break;
        }
    }
    
    // Generate diagnostic report
    auto report = NumericalDiagnostics::analyzeVector(probs, "final_probabilities");
    
    // Verify that the workflow completed successfully
    EXPECT_TRUE(NumericalSafety::isProbabilityDistribution(probs));
    EXPECT_FALSE(report.hasNaN);
    EXPECT_FALSE(report.hasInfinity);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
