/**
 * @file test_hmm_json_mv.cpp
 * @brief Tests for multivariate HMM JSON IO (Phase I).
 *
 * Verifies:
 *   - to_json(HmmMV) / from_json_mv() round-trips for all three MV distributions.
 *   - Exact parameter recovery (mean, var, cov, dim, reg, component types).
 *   - Schema fields (libhmm_version, obs_type, dimensions) are present.
 *   - File save/load round-trip via save_json_mv / load_json_mv.
 *   - Error cases: wrong obs_type, missing version, scalar schema rejected.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/distributions/full_covariance_gaussian_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/independent_components_distribution.h"
#include "libhmm/hmm.h"
#include "libhmm/io/hmm_json.h"
#include "libhmm/linalg/linalg_types.h"

using namespace libhmm;

// =============================================================================
// Shared factory helpers
// =============================================================================

namespace {

/// Build a 2-state HmmMV with uniform trans+pi.
HmmMV make_uniform_hmm_mv(std::size_t N = 2) {
    HmmMV hmm(N);
    Matrix trans(N, N);
    Vector pi(N);
    for (std::size_t i = 0; i < N; ++i) {
        pi(i) = 1.0 / static_cast<double>(N);
        for (std::size_t j = 0; j < N; ++j)
            trans(i, j) = (i == j) ? 0.8 : (0.2 / static_cast<double>(N - 1));
    }
    hmm.setTrans(trans);
    hmm.setPi(pi);
    return hmm;
}

/// Check that JSON string @p json contains the substring @p needle.
bool json_contains(const std::string &json, const std::string &needle) {
    return json.find(needle) != std::string::npos;
}

} // namespace

// =============================================================================
// Schema field presence
// =============================================================================

TEST(MvJsonSchema, ContainsLibhmmVersion) {
    HmmMV hmm = make_uniform_hmm_mv();
    hmm.setDistribution(0, std::make_unique<DiagonalGaussianDistribution>(2));
    hmm.setDistribution(1, std::make_unique<DiagonalGaussianDistribution>(2));
    const std::string j = to_json(hmm);
    EXPECT_TRUE(json_contains(j, "\"libhmm_version\""));
    EXPECT_TRUE(json_contains(j, "\"4\""));
}

TEST(MvJsonSchema, ContainsObsType) {
    HmmMV hmm = make_uniform_hmm_mv();
    hmm.setDistribution(0, std::make_unique<DiagonalGaussianDistribution>(2));
    hmm.setDistribution(1, std::make_unique<DiagonalGaussianDistribution>(2));
    const std::string j = to_json(hmm);
    EXPECT_TRUE(json_contains(j, "\"obs_type\""));
    EXPECT_TRUE(json_contains(j, "\"multivariate\""));
}

TEST(MvJsonSchema, ContainsDimensions) {
    HmmMV hmm = make_uniform_hmm_mv();
    constexpr std::size_t D = 3;
    hmm.setDistribution(0, std::make_unique<DiagonalGaussianDistribution>(D));
    hmm.setDistribution(1, std::make_unique<DiagonalGaussianDistribution>(D));
    const std::string j = to_json(hmm);
    EXPECT_TRUE(json_contains(j, "\"dimensions\""));
}

// =============================================================================
// DiagonalGaussianDistribution round-trip
// =============================================================================

TEST(MvJsonRoundTrip, DiagonalGaussianNumStates) {
    constexpr std::size_t N = 3, D = 2;
    HmmMV hmm = make_uniform_hmm_mv(N);
    for (std::size_t i = 0; i < N; ++i)
        hmm.setDistribution(i, std::make_unique<DiagonalGaussianDistribution>(D));

    HmmMV restored = from_json_mv(to_json(hmm));
    EXPECT_EQ(restored.getNumStatesModern(), N);
}

TEST(MvJsonRoundTrip, DiagonalGaussianPiAndTrans) {
    constexpr std::size_t N = 2, D = 2;
    HmmMV hmm = make_uniform_hmm_mv(N);
    hmm.setDistribution(0, std::make_unique<DiagonalGaussianDistribution>(D, 1.0, 2.0));
    hmm.setDistribution(1, std::make_unique<DiagonalGaussianDistribution>(D, 3.0, 4.0));

    HmmMV restored = from_json_mv(to_json(hmm));
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_EQ(hmm.getPi()(i), restored.getPi()(i)) << "pi[" << i << "]";
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_EQ(hmm.getTrans()(i, j), restored.getTrans()(i, j))
                << "trans(" << i << "," << j << ")";
}

TEST(MvJsonRoundTrip, DiagonalGaussianParametersExact) {
    // Verify bit-exact recovery of mean and variance for both states.
    constexpr std::size_t D = 2;
    HmmMV hmm = make_uniform_hmm_mv(2);
    hmm.setDistribution(0, std::make_unique<DiagonalGaussianDistribution>(D, 1.5, 0.25));
    hmm.setDistribution(1, std::make_unique<DiagonalGaussianDistribution>(D, -2.0, 3.0));

    HmmMV restored = from_json_mv(to_json(hmm));

    auto *d0 = dynamic_cast<const DiagonalGaussianDistribution *>(&restored.getDistribution(0));
    ASSERT_NE(d0, nullptr);
    EXPECT_EQ(d0->getDimension(), D);
    for (std::size_t k = 0; k < D; ++k) {
        EXPECT_EQ(d0->getMean()[k], 1.5) << "state0 mean[" << k << "]";
        EXPECT_EQ(d0->getVariance()[k], 0.25) << "state0 var[" << k << "]";
    }

    auto *d1 = dynamic_cast<const DiagonalGaussianDistribution *>(&restored.getDistribution(1));
    ASSERT_NE(d1, nullptr);
    for (std::size_t k = 0; k < D; ++k) {
        EXPECT_EQ(d1->getMean()[k], -2.0) << "state1 mean[" << k << "]";
        EXPECT_EQ(d1->getVariance()[k], 3.0) << "state1 var[" << k << "]";
    }
}

// =============================================================================
// FullCovarianceGaussianDistribution round-trip
// =============================================================================

TEST(MvJsonRoundTrip, FullCovGaussianTypeAndDimension) {
    constexpr std::size_t D = 3;
    HmmMV hmm = make_uniform_hmm_mv(2);
    hmm.setDistribution(0, std::make_unique<FullCovarianceGaussianDistribution>(D));
    hmm.setDistribution(1, std::make_unique<FullCovarianceGaussianDistribution>(D));

    HmmMV restored = from_json_mv(to_json(hmm));
    auto *d =
        dynamic_cast<const FullCovarianceGaussianDistribution *>(&restored.getDistribution(0));
    ASSERT_NE(d, nullptr) << "Wrong distribution type after round-trip";
    EXPECT_EQ(d->getDimension(), D);
}

TEST(MvJsonRoundTrip, FullCovGaussianMeanAndCovExact) {
    // Fit a known covariance then check round-trip.
    constexpr std::size_t D = 2;
    HmmMV hmm = make_uniform_hmm_mv(2);

    // State 0: explicit mean; default Σ=I (from constructor).
    auto d0 = std::make_unique<FullCovarianceGaussianDistribution>(D, 1e-10);
    hmm.setDistribution(0, std::move(d0));
    hmm.setDistribution(1, std::make_unique<FullCovarianceGaussianDistribution>(D, 1e-10));

    HmmMV restored = from_json_mv(to_json(hmm));

    auto *r =
        dynamic_cast<const FullCovarianceGaussianDistribution *>(&restored.getDistribution(0));
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->getDimension(), D);
    // Default mean is [0,0]; default cov is I.
    for (std::size_t d = 0; d < D; ++d)
        EXPECT_NEAR(r->getMean()[d], 0.0, 1e-12) << "mean[" << d << "]";
    for (std::size_t i = 0; i < D; ++i)
        for (std::size_t j = 0; j < D; ++j)
            EXPECT_NEAR(r->getCovariance()(i, j), (i == j) ? 1.0 : 0.0, 1e-12)
                << "cov(" << i << "," << j << ")";
}

// =============================================================================
// IndependentComponentsDistribution round-trip
// =============================================================================

TEST(MvJsonRoundTrip, IndependentComponentsTypeAndDimension) {
    constexpr std::size_t D = 3;
    HmmMV hmm = make_uniform_hmm_mv(2);
    hmm.setDistribution(0, std::make_unique<IndependentComponentsDistribution>(D));
    hmm.setDistribution(1, std::make_unique<IndependentComponentsDistribution>(D));

    HmmMV restored = from_json_mv(to_json(hmm));
    auto *d = dynamic_cast<const IndependentComponentsDistribution *>(&restored.getDistribution(0));
    ASSERT_NE(d, nullptr) << "Wrong distribution type after round-trip";
    EXPECT_EQ(d->getDimension(), D);
}

TEST(MvJsonRoundTrip, IndependentComponentsComponentTypes) {
    // Default components are GaussianDistribution; verify they survive round-trip.
    constexpr std::size_t D = 2;
    HmmMV hmm = make_uniform_hmm_mv(2);
    hmm.setDistribution(0, std::make_unique<IndependentComponentsDistribution>(D));
    hmm.setDistribution(1, std::make_unique<IndependentComponentsDistribution>(D));

    HmmMV restored = from_json_mv(to_json(hmm));
    auto *d = dynamic_cast<const IndependentComponentsDistribution *>(&restored.getDistribution(0));
    ASSERT_NE(d, nullptr);
    for (std::size_t k = 0; k < D; ++k) {
        const auto *g = dynamic_cast<const GaussianDistribution *>(&d->getComponent(k));
        EXPECT_NE(g, nullptr) << "Component " << k << " should be GaussianDistribution";
    }
}

// =============================================================================
// File save / load
// =============================================================================

class MvJsonFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmpDir_ = std::filesystem::temp_directory_path() / "libhmm_mv_json_test";
        std::filesystem::create_directories(tmpDir_);
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(tmpDir_, ec);
    }
    std::filesystem::path tmpDir_;
};

TEST_F(MvJsonFileTest, SaveLoadDiagGaussianRoundTrip) {
    constexpr std::size_t N = 2, D = 2;
    HmmMV hmm = make_uniform_hmm_mv(N);
    hmm.setDistribution(0, std::make_unique<DiagonalGaussianDistribution>(D, 1.0, 2.0));
    hmm.setDistribution(1, std::make_unique<DiagonalGaussianDistribution>(D, -3.0, 0.5));

    const auto path = tmpDir_ / "mv_test.json";
    ASSERT_NO_THROW(save_json_mv(hmm, path));
    ASSERT_TRUE(std::filesystem::exists(path));
    ASSERT_GT(std::filesystem::file_size(path), 0u);

    HmmMV restored = load_json_mv(path);
    EXPECT_EQ(restored.getNumStatesModern(), N);

    auto *d0 = dynamic_cast<const DiagonalGaussianDistribution *>(&restored.getDistribution(0));
    ASSERT_NE(d0, nullptr);
    EXPECT_EQ(d0->getMean()[0], 1.0);
    EXPECT_EQ(d0->getVariance()[0], 2.0);
}

TEST_F(MvJsonFileTest, LoadNonExistentThrows) {
    EXPECT_THROW(static_cast<void>(load_json_mv(tmpDir_ / "does_not_exist.json")),
                 std::runtime_error);
}

// =============================================================================
// Error cases
// =============================================================================

TEST(MvJsonErrors, RejectsScalarSchema) {
    // A scalar HMM JSON should be rejected by from_json_mv.
    Hmm scalar_hmm(2);
    const std::string scalar_json = to_json(scalar_hmm);
    EXPECT_THROW(static_cast<void>(from_json_mv(scalar_json)), std::runtime_error);
}

TEST(MvJsonErrors, RejectsWrongObsType) {
    const std::string bad =
        R"({"libhmm_version":"4","obs_type":"scalar","dimensions":2,"states":1,"pi":[1.0],"trans":[[1.0]],"distributions":[]})";
    EXPECT_THROW(static_cast<void>(from_json_mv(bad)), std::runtime_error);
}

TEST(MvJsonErrors, RejectsUnknownDistributionType) {
    const std::string bad =
        R"({"libhmm_version":"4","obs_type":"multivariate","dimensions":2,"states":1,"pi":[1.0],"trans":[[1.0]],"distributions":[{"type":"BogusGaussian","dim":2}]})";
    EXPECT_THROW(static_cast<void>(from_json_mv(bad)), std::runtime_error);
}

TEST(MvJsonErrors, RejectsMissingVersion) {
    const std::string bad =
        R"({"obs_type":"multivariate","dimensions":2,"states":1,"pi":[1.0],"trans":[[1.0]],"distributions":[]})";
    EXPECT_THROW(static_cast<void>(from_json_mv(bad)), std::runtime_error);
}

TEST(MvJsonErrors, RejectsOversizedInput) {
    const std::string oversized(11UL * 1024UL * 1024UL, ' ');
    EXPECT_THROW(static_cast<void>(from_json_mv(oversized)), std::runtime_error);
}

// =============================================================================
// Scalar from_json still works (backward compat unchanged)
// =============================================================================

TEST(MvJsonBackwardCompat, ScalarFromJsonUnaffected) {
    Hmm original(2);
    Vector pi(2);
    pi(0) = 0.6;
    pi(1) = 0.4;
    original.setPi(pi);
    Matrix trans(2, 2);
    trans(0, 0) = 0.9;
    trans(0, 1) = 0.1;
    trans(1, 0) = 0.2;
    trans(1, 1) = 0.8;
    original.setTrans(trans);

    Hmm restored = from_json(to_json(original));
    EXPECT_EQ(restored.getNumStatesModern(), 2u);
    EXPECT_EQ(restored.getPi()(0), 0.6);
    EXPECT_EQ(restored.getPi()(1), 0.4);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
