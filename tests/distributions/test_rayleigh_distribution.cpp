#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <climits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "libhmm/distributions/rayleigh_distribution.h"
#include <gtest/gtest.h>

using libhmm::Observation;
using libhmm::RayleighDistribution;

TEST(RayleighDistributionTest, BasicFunctionality) {

    // Test default constructor
    RayleighDistribution rayleigh;
    EXPECT_EQ(rayleigh.getSigma(), 1.0);

    // Test parameterized constructor
    RayleighDistribution rayleigh2(2.5);
    EXPECT_EQ(rayleigh2.getSigma(), 2.5);
}

TEST(RayleighDistributionTest, Probabilities) {

    RayleighDistribution rayleigh(1.0); // σ=1

    // Test that probability is zero for negative and zero values
    EXPECT_EQ(rayleigh.getProbability(-0.1), 0.0);
    EXPECT_EQ(rayleigh.getProbability(0.0), 0.0);

    // Test that probability is positive for positive values
    double prob1 = rayleigh.getProbability(0.5);
    double prob2 = rayleigh.getProbability(1.0);
    double prob3 = rayleigh.getProbability(2.0);

    EXPECT_GT(prob1, 0.0);
    EXPECT_GT(prob2, 0.0);
    EXPECT_GT(prob3, 0.0);

    // Test boundary value at x=0
    double probAt0 = rayleigh.getProbability(0.0);
    EXPECT_EQ(probAt0, 0.0);
}

TEST(RayleighDistributionTest, Fitting) {

    RayleighDistribution rayleigh;

    // Test with known data
    std::vector<Observation> data = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    rayleigh.fit(data);
    EXPECT_GT(rayleigh.getSigma(), 0.0);

    // Test with empty data (should reset to default)
    std::vector<Observation> emptyData;
    rayleigh.fit(emptyData);
    EXPECT_EQ(rayleigh.getSigma(), 1.0);
}

TEST(RayleighDistributionTest, ParameterValidation) {

    // Test invalid constructor parameters
    EXPECT_THROW(RayleighDistribution rayleigh(0.0), std::invalid_argument);

    EXPECT_THROW(RayleighDistribution rayleigh(-1.0), std::invalid_argument);

    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();

    EXPECT_THROW(RayleighDistribution rayleigh(nan_val), std::invalid_argument);

    EXPECT_THROW(RayleighDistribution rayleigh(inf_val), std::invalid_argument);
}

TEST(RayleighDistributionTest, Statistics) {

    RayleighDistribution rayleigh(1.0);

    // Statistical properties
    double mean = rayleigh.getMean();
    double variance = rayleigh.getVariance();
    double stddev = rayleigh.getStandardDeviation();

    EXPECT_TRUE(mean > 1.25 && mean < 1.26);         // σ * √(π/2) ≈ 1.253
    EXPECT_TRUE(variance > 0.42 && variance < 0.43); // σ² * (4-π)/2 ≈ 0.429
    EXPECT_NEAR(stddev, std::sqrt(variance), 1e-10);
}

TEST(RayleighDistributionTest, StatisticalMoments) {

    RayleighDistribution rayleigh(1.0);

    double expectedMean = libhmm::constants::math::SQRT_PI_OVER_TWO;
    double expectedVar = libhmm::constants::math::FOUR_MINUS_PI_OVER_TWO;

    EXPECT_NEAR(rayleigh.getMean(), expectedMean, 1e-10);
    EXPECT_NEAR(rayleigh.getVariance(), expectedVar, 1e-10);
    EXPECT_NEAR(rayleigh.getStandardDeviation(), std::sqrt(expectedVar), 1e-10);
}

TEST(RayleighDistributionTest, StringRepresentation) {

    RayleighDistribution rayleigh(1.0);
    std::string str = rayleigh.toString();

    EXPECT_NE(str.find("Rayleigh"), std::string::npos);
    EXPECT_NE(str.find("Distribution"), std::string::npos);
    EXPECT_NE(str.find("1.0"), std::string::npos);

    std::cout << "String representation: " << str << std::endl;
}

TEST(RayleighDistributionTest, CopyMoveSemantics) {

    RayleighDistribution original(2.0);

    RayleighDistribution copied(original);
    EXPECT_EQ(copied.getSigma(), original.getSigma());

    RayleighDistribution assigned;
    assigned = original;
    EXPECT_EQ(assigned.getSigma(), original.getSigma());

    RayleighDistribution moved(std::move(original));
    EXPECT_EQ(moved.getSigma(), 2.0);
}

TEST(RayleighDistributionTest, EqualityAndIO) {

    RayleighDistribution r1(1.0);
    RayleighDistribution r2(1.0);
    RayleighDistribution r3(2.0);

    EXPECT_EQ(r1, r2);
    EXPECT_EQ(r2, r1);
    EXPECT_FALSE(r1 == r3);
    EXPECT_NE(r1, r3);

    std::ostringstream oss;
    oss << r1.toString();
    std::string output = oss.str();
    EXPECT_NE(output.find("Rayleigh"), std::string::npos);
    EXPECT_NE(output.find("1.0"), std::string::npos);

    std::cout << "Stream output: " << output << std::endl;

    std::istringstream iss("Scale parameter = 3.14");
    RayleighDistribution inputDist;
    iss >> inputDist;
    if (!iss.fail()) {
        EXPECT_NEAR(inputDist.getSigma(), 3.14, 1e-10);
    }
}

TEST(RayleighDistributionTest, Caching) {

    RayleighDistribution rayleigh(1.0);

    double prob1 = rayleigh.getProbability(1.0);
    rayleigh.setSigma(2.0);
    double prob2 = rayleigh.getProbability(1.0);

    EXPECT_NE(prob1, prob2);

    rayleigh.reset(); // Reset back to sigma=1.0
    double prob3 = rayleigh.getProbability(1.0);
    EXPECT_NEAR(prob1, prob3, 1e-10); // Should be the same as original

    // Test that cache invalidation works correctly
    rayleigh.setSigma(3.0);
    double prob4 = rayleigh.getProbability(1.0);
    EXPECT_NE(prob1, prob4);
    EXPECT_NE(prob2, prob4);
}

TEST(RayleighDistributionTest, CDFCalculations) {

    RayleighDistribution rayleigh(1.0);

    // Test boundary values
    EXPECT_EQ(rayleigh.getCumulativeProbability(-1.0), 0.0);
    EXPECT_EQ(rayleigh.getCumulativeProbability(0.0), 0.0);

    // Test known values
    double cdf_at_sigma = rayleigh.getCumulativeProbability(1.0); // CDF at x = σ
    double expected_cdf_at_sigma = 1.0 - std::exp(-0.5);          // 1 - exp(-1²/(2*1²))
    EXPECT_NEAR(cdf_at_sigma, expected_cdf_at_sigma, 1e-10);

    // Test monotonicity
    double cdf1 = rayleigh.getCumulativeProbability(0.5);
    double cdf2 = rayleigh.getCumulativeProbability(1.0);
    double cdf3 = rayleigh.getCumulativeProbability(2.0);
    EXPECT_LT(cdf1, cdf2);
    EXPECT_LT(cdf2, cdf3);

    // Test that CDF approaches 1 for large values
    double cdf_large = rayleigh.getCumulativeProbability(10.0);
    EXPECT_GT(cdf_large, 0.99);
}

TEST(RayleighDistributionTest, NumericalStability) {

    RayleighDistribution smallSigma(0.001);
    RayleighDistribution largeSigma(1000.0);

    double probSmall = smallSigma.getProbability(0.0001);
    double probLarge = largeSigma.getProbability(500.0);

    EXPECT_TRUE(probSmall > 0.0 && std::isfinite(probSmall));
    EXPECT_TRUE(probLarge > 0.0 && std::isfinite(probLarge));

    // Test CDF stability
    double cdfSmall = smallSigma.getCumulativeProbability(0.0001);
    double cdfLarge = largeSigma.getCumulativeProbability(500.0);

    EXPECT_TRUE(cdfSmall >= 0.0 && cdfSmall <= 1.0 && std::isfinite(cdfSmall));
    EXPECT_TRUE(cdfLarge >= 0.0 && cdfLarge <= 1.0 && std::isfinite(cdfLarge));
}

TEST(RayleighDistributionTest, PerformanceCharacteristics) {

    RayleighDistribution rayleigh(1.0);

    // Test PDF timing
    auto start = std::chrono::high_resolution_clock::now();
    const int pdfIterations = 100000;
    volatile double sum = 0.0; // volatile to prevent optimization

    for (int i = 0; i < pdfIterations; ++i) {
        double x = static_cast<double>(i + 1) / 10000.0; // Range 0.0001 to 10
        sum += rayleigh.getProbability(x);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto pdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double pdfTimePerCall = static_cast<double>(pdfDuration.count()) / pdfIterations;

    // Test Log PDF timing
    start = std::chrono::high_resolution_clock::now();
    volatile double logSum = 0.0;

    for (int i = 0; i < pdfIterations; ++i) {
        double x = static_cast<double>(i + 1) / 10000.0; // Range 0.0001 to 10
        logSum += rayleigh.getLogProbability(x);
    }

    end = std::chrono::high_resolution_clock::now();
    auto logPdfDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double logPdfTimePerCall = static_cast<double>(logPdfDuration.count()) / pdfIterations;

    // Test fitting timing
    std::vector<Observation> fitData(5000);
    for (size_t i = 0; i < fitData.size(); ++i) {
        fitData[i] = static_cast<double>(i + 1) / 1000.0; // Positive values
    }

    start = std::chrono::high_resolution_clock::now();
    rayleigh.fit(fitData);
    end = std::chrono::high_resolution_clock::now();
    auto fitDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double fitTimePerPoint = static_cast<double>(fitDuration.count()) / fitData.size();

    std::cout << "  PDF timing:       " << std::fixed << std::setprecision(3) << pdfTimePerCall
              << " μs/call (" << pdfIterations << " calls)" << std::endl;
    std::cout << "  Log PDF timing:   " << std::fixed << std::setprecision(3) << logPdfTimePerCall
              << " μs/call (" << pdfIterations << " calls)" << std::endl;
    std::cout << "  Fit timing:       " << std::fixed << std::setprecision(3) << fitTimePerPoint
              << " μs/point (" << fitData.size() << " points)" << std::endl;

    // Performance requirements (should be fast)
    EXPECT_LT(pdfTimePerCall, 1.0);    // Less than 1 μs per PDF call
    EXPECT_LT(logPdfTimePerCall, 1.0); // Less than 1 μs per log PDF call
    EXPECT_LT(fitTimePerPoint, 0.1);   // Less than 0.1 μs per data point for fitting
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
