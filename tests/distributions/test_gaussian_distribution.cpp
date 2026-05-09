#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "libhmm/distributions/gaussian_distribution.h"

using libhmm::GaussianDistribution;
using libhmm::Observation;
using namespace libhmm::constants;

TEST(GaussianDistributionTest, BasicFunctionality) {
    GaussianDistribution gaussian;
    EXPECT_EQ(gaussian.getMean(), 0.0);
    EXPECT_EQ(gaussian.getStandardDeviation(), 1.0);

    GaussianDistribution gaussian2(5.0, 2.5);
    EXPECT_EQ(gaussian2.getMean(), 5.0);
    EXPECT_EQ(gaussian2.getStandardDeviation(), 2.5);
}

TEST(GaussianDistributionTest, Probabilities) {
    GaussianDistribution gaussian(0.0, 1.0);

    double probAtMean = gaussian.getProbability(0.0);
    EXPECT_GT(probAtMean, 0.0);
    EXPECT_NEAR(probAtMean, 1.0 / math::SQRT_2PI, 1e-10);

    double probAt1 = gaussian.getProbability(1.0);
    EXPECT_NEAR(probAt1, gaussian.getProbability(-1.0), 1e-10);
    EXPECT_GT(probAtMean, probAt1);
    EXPECT_LT(probAtMean, 1.0);
    EXPECT_GT(probAtMean, 1e-10);
}

TEST(GaussianDistributionTest, Fitting) {
    GaussianDistribution gaussian;

    std::vector<Observation> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    gaussian.fit(data);
    EXPECT_NEAR(gaussian.getMean(), 3.0, 1e-10);
    EXPECT_GT(gaussian.getStandardDeviation(), 0.0);

    gaussian.fit(std::vector<Observation>{});
    EXPECT_EQ(gaussian.getMean(), 0.0);
    EXPECT_EQ(gaussian.getStandardDeviation(), 1.0);

    gaussian.fit(std::vector<Observation>{7.5});
    EXPECT_EQ(gaussian.getMean(), 0.0);
    EXPECT_EQ(gaussian.getStandardDeviation(), 1.0);
}

TEST(GaussianDistributionTest, ParameterValidation) {
    EXPECT_THROW(GaussianDistribution(0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(GaussianDistribution(0.0, -1.0), std::invalid_argument);
    EXPECT_THROW(GaussianDistribution(std::numeric_limits<double>::quiet_NaN(), 1.0),
                 std::invalid_argument);
    EXPECT_THROW(GaussianDistribution(std::numeric_limits<double>::infinity(), 1.0),
                 std::invalid_argument);

    GaussianDistribution gaussian(0.0, 1.0);
    EXPECT_THROW(gaussian.setMean(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(gaussian.setStandardDeviation(0.0), std::invalid_argument);
    EXPECT_THROW(gaussian.setStandardDeviation(-1.0), std::invalid_argument);
}

TEST(GaussianDistributionTest, StringRepresentation) {
    GaussianDistribution gaussian(2.5, 1.5);
    std::string str = gaussian.toString();
    EXPECT_NE(str.find("Gaussian"), std::string::npos);
    EXPECT_NE(str.find("Distribution"), std::string::npos);
    EXPECT_NE(str.find("2.5"), std::string::npos);
    EXPECT_NE(str.find("1.5"), std::string::npos);
    EXPECT_NE(str.find("Mean"), std::string::npos);
    EXPECT_NE(str.find("std. deviation"), std::string::npos);
}

TEST(GaussianDistributionTest, CopyMoveSemantics) {
    GaussianDistribution original(3.14, 2.71);

    GaussianDistribution copied(original);
    EXPECT_EQ(copied.getMean(), original.getMean());
    EXPECT_EQ(copied.getStandardDeviation(), original.getStandardDeviation());

    GaussianDistribution assigned;
    assigned = original;
    EXPECT_EQ(assigned.getMean(), original.getMean());
    EXPECT_EQ(assigned.getStandardDeviation(), original.getStandardDeviation());

    GaussianDistribution moved(std::move(original));
    EXPECT_EQ(moved.getMean(), 3.14);
    EXPECT_EQ(moved.getStandardDeviation(), 2.71);

    GaussianDistribution moveAssigned;
    GaussianDistribution temp(1.41, 1.73);
    moveAssigned = std::move(temp);
    EXPECT_EQ(moveAssigned.getMean(), 1.41);
    EXPECT_EQ(moveAssigned.getStandardDeviation(), 1.73);
}

TEST(GaussianDistributionTest, InvalidInputHandling) {
    GaussianDistribution gaussian(0.0, 1.0);
    EXPECT_EQ(gaussian.getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
    EXPECT_EQ(gaussian.getProbability(std::numeric_limits<double>::infinity()), 0.0);
    EXPECT_EQ(gaussian.getProbability(-std::numeric_limits<double>::infinity()), 0.0);
}

TEST(GaussianDistributionTest, LogProbability) {
    GaussianDistribution gaussian(0.0, 1.0);

    EXPECT_NEAR(gaussian.getLogProbability(0.0), std::log(1.0 / math::SQRT_2PI), 1e-10);
    EXPECT_NEAR(std::exp(gaussian.getLogProbability(0.0)), gaussian.getProbability(0.0), 1e-10);
    EXPECT_NEAR(gaussian.getLogProbability(1.0), gaussian.getLogProbability(-1.0), 1e-10);

    EXPECT_TRUE(std::isinf(gaussian.getLogProbability(std::numeric_limits<double>::quiet_NaN())));
    EXPECT_TRUE(std::isinf(gaussian.getLogProbability(std::numeric_limits<double>::infinity())));
}

TEST(GaussianDistributionTest, AdditionalGettersSetters) {
    GaussianDistribution gaussian(3.0, 2.0);
    EXPECT_NEAR(gaussian.getVariance(), 4.0, 1e-10);

    gaussian.setParameters(1.5, 0.5);
    EXPECT_NEAR(gaussian.getMean(), 1.5, 1e-10);
    EXPECT_NEAR(gaussian.getStandardDeviation(), 0.5, 1e-10);
    EXPECT_NEAR(gaussian.getVariance(), 0.25, 1e-10);

    EXPECT_THROW(gaussian.setParameters(1.0, -1.0), std::invalid_argument);
}

TEST(GaussianDistributionTest, MathematicalCorrectness) {
    GaussianDistribution gaussian1(0.0, 1.0);
    GaussianDistribution gaussian2(5.0, 2.0);

    EXPECT_NEAR(gaussian1.getProbability(0.0), 0.3989422804, 1e-8);
    EXPECT_NEAR(gaussian1.getProbability(1.0), 0.2419707245, 1e-8);
    EXPECT_NEAR(gaussian2.getProbability(5.0), 1.0 / (2.0 * math::SQRT_2PI), 1e-8);
}

TEST(GaussianDistributionTest, ResetFunctionality) {
    GaussianDistribution gaussian(10.0, 5.0);
    gaussian.reset();
    EXPECT_EQ(gaussian.getMean(), 0.0);
    EXPECT_EQ(gaussian.getStandardDeviation(), 1.0);
}

TEST(GaussianDistributionTest, CDF) {
    GaussianDistribution gaussian(0.0, 1.0);

    EXPECT_NEAR(gaussian.getCumulativeProbability(0.0), 0.5, 1e-6);

    double cdfAt1 = gaussian.getCumulativeProbability(1.0);
    double cdfAtNeg1 = gaussian.getCumulativeProbability(-1.0);
    EXPECT_NEAR(cdfAt1 + cdfAtNeg1, 1.0, 1e-6);

    EXPECT_LT(gaussian.getCumulativeProbability(-2.0), gaussian.getCumulativeProbability(-1.0));
    EXPECT_LT(gaussian.getCumulativeProbability(-1.0), gaussian.getCumulativeProbability(0.0));
    EXPECT_LT(gaussian.getCumulativeProbability(0.0), gaussian.getCumulativeProbability(1.0));
    EXPECT_LT(gaussian.getCumulativeProbability(1.0), gaussian.getCumulativeProbability(2.0));

    EXPECT_GE(gaussian.getCumulativeProbability(-5.0), 0.0);
    EXPECT_LE(gaussian.getCumulativeProbability(-5.0), 1.0);
    EXPECT_LT(gaussian.getCumulativeProbability(-10.0), 0.001);
    EXPECT_GT(gaussian.getCumulativeProbability(10.0), 0.999);

    GaussianDistribution gaussian2(2.0, 0.5);
    EXPECT_NEAR(gaussian2.getCumulativeProbability(2.0), 0.5, 1e-6);

    EXPECT_EQ(gaussian.getCumulativeProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
}

TEST(GaussianDistributionTest, EqualityAndIO) {
    GaussianDistribution g1(2.5, 1.5);
    GaussianDistribution g2(2.5, 1.5);
    GaussianDistribution g3(2.5, 1.6);
    GaussianDistribution g4(2.6, 1.5);

    EXPECT_TRUE(g1 == g2);
    EXPECT_TRUE(g2 == g1);
    EXPECT_FALSE(g1 == g3);
    EXPECT_FALSE(g1 == g4);
    EXPECT_TRUE(g1 != g3);
    EXPECT_TRUE(g1 == g1);
    EXPECT_TRUE(g1 == GaussianDistribution(2.5, 1.5 + 1e-15));
    EXPECT_FALSE(g1 == GaussianDistribution(2.5, 1.5 + 1e-5));

    std::ostringstream oss;
    oss << g1;
    std::string output = oss.str();
    EXPECT_NE(output.find("Gaussian Distribution"), std::string::npos);
    EXPECT_NE(output.find("2.5"), std::string::npos);
    EXPECT_NE(output.find("1.5"), std::string::npos);

    std::istringstream invalid_iss("invalid data format");
    GaussianDistribution invalid_test;
    invalid_iss >> invalid_test;
    EXPECT_TRUE(invalid_iss.fail());
}

TEST(GaussianDistributionTest, Caching) {
    GaussianDistribution gaussian(1.0, 2.0);

    double prob1 = gaussian.getProbability(1.0);
    double logProb1 = gaussian.getLogProbability(1.0);
    EXPECT_NEAR(prob1, std::exp(logProb1), 1e-10);

    gaussian.setMean(3.0);
    double prob2 = gaussian.getProbability(1.0);
    EXPECT_GT(std::abs(prob1 - prob2), 1e-6);
    EXPECT_NEAR(prob2, std::exp(gaussian.getLogProbability(1.0)), 1e-10);

    double prob3 = gaussian.getProbability(3.0);
    gaussian.setStandardDeviation(1.0);
    EXPECT_GT(std::abs(prob3 - gaussian.getProbability(3.0)), 1e-6);

    double prob5 = gaussian.getProbability(3.0);
    gaussian.setParameters(0.0, 1.0);
    EXPECT_GT(std::abs(prob5 - gaussian.getProbability(3.0)), 1e-6);

    gaussian.setMean(5.0);
    double prob_before_reset = gaussian.getProbability(0.0);
    gaussian.reset();
    EXPECT_GT(std::abs(prob_before_reset - gaussian.getProbability(0.0)), 1e-6);
}

TEST(GaussianDistributionTest, Performance) {
    using namespace std::chrono;
    GaussianDistribution gaussian(0.0, 1.0);

    const int pdf_iterations = 100000;
    const int fit_datapoints = 5000;

    std::vector<double> testValues;
    testValues.reserve(pdf_iterations);
    for (int i = 0; i < pdf_iterations; ++i) {
        testValues.push_back(-3.0 + (6.0 * i) / pdf_iterations);
    }

    auto start = high_resolution_clock::now();
    double sum_pdf = 0.0;
    for (const auto &val : testValues) {
        sum_pdf += gaussian.getProbability(val);
    }
    auto end = high_resolution_clock::now();
    double pdf_per_call =
        static_cast<double>(duration_cast<microseconds>(end - start).count()) / pdf_iterations;

    start = high_resolution_clock::now();
    double sum_log_pdf = 0.0;
    for (const auto &val : testValues) {
        sum_log_pdf += gaussian.getLogProbability(val);
    }
    end = high_resolution_clock::now();
    double log_pdf_per_call =
        static_cast<double>(duration_cast<microseconds>(end - start).count()) / pdf_iterations;

    std::vector<double> fit_data(fit_datapoints);
    for (int i = 0; i < fit_datapoints; ++i) {
        fit_data[i] = i * 0.001;
    }
    start = high_resolution_clock::now();
    gaussian.fit(fit_data);
    end = high_resolution_clock::now();
    double fit_per_point =
        static_cast<double>(duration_cast<microseconds>(end - start).count()) / fit_datapoints;

    std::cout << std::fixed << std::setprecision(3) << "  PDF: " << pdf_per_call << " μs/call"
              << "  LogPDF: " << log_pdf_per_call << " μs/call"
              << "  Fit: " << fit_per_point << " μs/point\n";

    EXPECT_LT(pdf_per_call, 1.0);
    EXPECT_LT(log_pdf_per_call, 1.0);
    EXPECT_LT(fit_per_point, 10.0);
    EXPECT_GT(sum_pdf, 0.0);
    EXPECT_LT(sum_log_pdf, 0.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
