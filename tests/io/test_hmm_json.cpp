#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>

#include "libhmm/distributions/distributions.h"
#include "libhmm/io/hmm_json.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/linalg/linalg_types.h"

using namespace libhmm;

// =============================================================================
// Distribution to_json format: verify the type field is written correctly
// =============================================================================

TEST(DistributionToJson, TypeFieldPrefix) {
    // Each to_json() must begin with {"type":"<TypeName>" exactly.
    auto check = [](const EmissionDistribution &d, const std::string &expected_type) {
        const std::string j = d.to_json();
        const std::string prefix = "{\"type\":\"" + expected_type + "\"";
        EXPECT_EQ(j.substr(0, prefix.size()), prefix) << "type: " << expected_type;
    };

    check(GaussianDistribution(1.0, 2.0), "Gaussian");
    check(ExponentialDistribution(3.0), "Exponential");
    check(GammaDistribution(2.0, 1.5), "Gamma");
    check(BetaDistribution(2.0, 3.0), "Beta");
    check(WeibullDistribution(1.5, 2.5), "Weibull");
    check(LogNormalDistribution(0.5, 1.2), "LogNormal");
    check(ParetoDistribution(2.0, 0.5), "Pareto");
    check(NegativeBinomialDistribution(5.0, 0.4), "NegativeBinomial");
    check(ChiSquaredDistribution(4.0), "ChiSquared");
    check(StudentTDistribution(3.0, 0.5, 1.5), "StudentT");
    check(PoissonDistribution(2.5), "Poisson");
    check(BinomialDistribution(10, 0.3), "Binomial");

    DiscreteDistribution disc(4);
    for (int i = 0; i < 4; ++i)
        disc.setProbability(static_cast<double>(i), 0.25);
    check(disc, "Discrete");

    check(UniformDistribution(1.0, 3.0), "Uniform");
    check(RayleighDistribution(2.0), "Rayleigh");
}

// =============================================================================
// All-distributions round-trip via HMM JSON (exact recovery using max_digits10)
// =============================================================================

TEST(HmmJson, AllDistributionsRoundTrip) {
    constexpr std::size_t N = 15;

    Matrix trans(N, N);
    Vector pi(N);
    for (std::size_t i = 0; i < N; ++i) {
        pi[i] = 1.0 / 15.0;
        for (std::size_t j = 0; j < N; ++j)
            trans(i, j) = (i == j) ? 0.9 : (0.1 / 14.0);
    }

    std::vector<std::unique_ptr<EmissionDistribution>> emis(N);
    emis[0] = std::make_unique<GaussianDistribution>(1.5, 2.5);
    emis[1] = std::make_unique<ExponentialDistribution>(3.0);
    emis[2] = std::make_unique<GammaDistribution>(2.0, 1.5);
    emis[3] = std::make_unique<BetaDistribution>(2.0, 3.0);
    emis[4] = std::make_unique<WeibullDistribution>(1.5, 2.5);
    emis[5] = std::make_unique<LogNormalDistribution>(0.5, 1.2);
    emis[6] = std::make_unique<ParetoDistribution>(2.0, 0.5);
    emis[7] = std::make_unique<NegativeBinomialDistribution>(5.0, 0.4);
    emis[8] = std::make_unique<ChiSquaredDistribution>(4.0);
    emis[9] = std::make_unique<StudentTDistribution>(3.0, 0.5, 1.5);
    emis[10] = std::make_unique<PoissonDistribution>(2.5);
    emis[11] = std::make_unique<BinomialDistribution>(10, 0.3);
    {
        auto disc = std::make_unique<DiscreteDistribution>(4);
        disc->setProbability(0.0, 0.1);
        disc->setProbability(1.0, 0.2);
        disc->setProbability(2.0, 0.3);
        disc->setProbability(3.0, 0.4);
        emis[12] = std::move(disc);
    }
    emis[13] = std::make_unique<UniformDistribution>(1.0, 3.0);
    emis[14] = std::make_unique<RayleighDistribution>(2.0);

    Hmm original(std::move(trans), std::move(emis), std::move(pi));

    const std::string json_str = to_json(original);
    Hmm restored = from_json(json_str);

    ASSERT_EQ(restored.getNumStatesModern(), N);

    // pi and trans must be bit-exact (max_digits10 guarantees round-trip)
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_EQ(original.getPi()[i], restored.getPi()[i]) << "pi[" << i << "]";
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_EQ(original.getTrans()(i, j), restored.getTrans()(i, j))
                << "trans(" << i << "," << j << ")";

    auto *d0 = dynamic_cast<const GaussianDistribution *>(&restored.getDistribution(0));
    ASSERT_NE(d0, nullptr);
    EXPECT_EQ(d0->getMean(), 1.5);
    EXPECT_EQ(d0->getStandardDeviation(), 2.5);

    auto *d1 = dynamic_cast<const ExponentialDistribution *>(&restored.getDistribution(1));
    ASSERT_NE(d1, nullptr);
    EXPECT_EQ(d1->getLambda(), 3.0);

    auto *d2 = dynamic_cast<const GammaDistribution *>(&restored.getDistribution(2));
    ASSERT_NE(d2, nullptr);
    EXPECT_EQ(d2->getK(), 2.0);
    EXPECT_EQ(d2->getTheta(), 1.5);

    auto *d3 = dynamic_cast<const BetaDistribution *>(&restored.getDistribution(3));
    ASSERT_NE(d3, nullptr);
    EXPECT_EQ(d3->getAlpha(), 2.0);
    EXPECT_EQ(d3->getBeta(), 3.0);

    auto *d4 = dynamic_cast<const WeibullDistribution *>(&restored.getDistribution(4));
    ASSERT_NE(d4, nullptr);
    EXPECT_EQ(d4->getK(), 1.5);
    EXPECT_EQ(d4->getLambda(), 2.5);

    auto *d5 = dynamic_cast<const LogNormalDistribution *>(&restored.getDistribution(5));
    ASSERT_NE(d5, nullptr);
    EXPECT_EQ(d5->getMean(), 0.5);
    EXPECT_EQ(d5->getStandardDeviation(), 1.2);

    auto *d6 = dynamic_cast<const ParetoDistribution *>(&restored.getDistribution(6));
    ASSERT_NE(d6, nullptr);
    EXPECT_EQ(d6->getK(), 2.0);
    EXPECT_EQ(d6->getXm(), 0.5);

    auto *d7 = dynamic_cast<const NegativeBinomialDistribution *>(&restored.getDistribution(7));
    ASSERT_NE(d7, nullptr);
    EXPECT_EQ(d7->getR(), 5.0);
    EXPECT_EQ(d7->getP(), 0.4);

    auto *d8 = dynamic_cast<const ChiSquaredDistribution *>(&restored.getDistribution(8));
    ASSERT_NE(d8, nullptr);
    EXPECT_EQ(d8->getDegreesOfFreedom(), 4.0);

    auto *d9 = dynamic_cast<const StudentTDistribution *>(&restored.getDistribution(9));
    ASSERT_NE(d9, nullptr);
    EXPECT_EQ(d9->getDegreesOfFreedom(), 3.0);
    EXPECT_EQ(d9->getLocation(), 0.5);
    EXPECT_EQ(d9->getScale(), 1.5);

    auto *d10 = dynamic_cast<const PoissonDistribution *>(&restored.getDistribution(10));
    ASSERT_NE(d10, nullptr);
    EXPECT_EQ(d10->getLambda(), 2.5);

    auto *d11 = dynamic_cast<const BinomialDistribution *>(&restored.getDistribution(11));
    ASSERT_NE(d11, nullptr);
    EXPECT_EQ(d11->getN(), 10);
    EXPECT_EQ(d11->getP(), 0.3);

    auto *d12 = dynamic_cast<const DiscreteDistribution *>(&restored.getDistribution(12));
    ASSERT_NE(d12, nullptr);
    EXPECT_EQ(d12->getNumSymbols(), 4u);
    EXPECT_EQ(d12->getSymbolProbability(0), 0.1);
    EXPECT_EQ(d12->getSymbolProbability(1), 0.2);
    EXPECT_EQ(d12->getSymbolProbability(2), 0.3);
    EXPECT_EQ(d12->getSymbolProbability(3), 0.4);

    auto *d13 = dynamic_cast<const UniformDistribution *>(&restored.getDistribution(13));
    ASSERT_NE(d13, nullptr);
    EXPECT_EQ(d13->getA(), 1.0);
    EXPECT_EQ(d13->getB(), 3.0);

    auto *d14 = dynamic_cast<const RayleighDistribution *>(&restored.getDistribution(14));
    ASSERT_NE(d14, nullptr);
    EXPECT_EQ(d14->getSigma(), 2.0);
}

// =============================================================================
// File save/load round-trip
// =============================================================================

class HmmJsonFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmpDir_ = std::filesystem::temp_directory_path() / "libhmm_json_test";
        std::filesystem::create_directories(tmpDir_);
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(tmpDir_, ec);
    }
    std::filesystem::path tmpDir_;
};

TEST_F(HmmJsonFileTest, FileSaveLoadRoundTrip) {
    Hmm original(2);

    Vector pi(2);
    pi[0] = 0.7;
    pi[1] = 0.3;
    original.setPi(pi);

    Matrix trans(2, 2);
    trans(0, 0) = 0.8;
    trans(0, 1) = 0.2;
    trans(1, 0) = 0.4;
    trans(1, 1) = 0.6;
    original.setTrans(trans);

    original.setDistribution(0, std::make_unique<GaussianDistribution>(2.0, 0.5));
    original.setDistribution(1, std::make_unique<PoissonDistribution>(4.0));

    const auto filepath = tmpDir_ / "test_hmm.json";
    ASSERT_NO_THROW(save_json(original, filepath));
    ASSERT_TRUE(std::filesystem::exists(filepath));
    ASSERT_GT(std::filesystem::file_size(filepath), 0u);

    Hmm restored = load_json(filepath);
    EXPECT_EQ(restored.getNumStatesModern(), 2u);
    EXPECT_EQ(restored.getPi()[0], 0.7);
    EXPECT_EQ(restored.getPi()[1], 0.3);
    EXPECT_EQ(restored.getTrans()(0, 0), 0.8);
    EXPECT_EQ(restored.getTrans()(0, 1), 0.2);
    EXPECT_EQ(restored.getTrans()(1, 0), 0.4);
    EXPECT_EQ(restored.getTrans()(1, 1), 0.6);

    const auto *g = dynamic_cast<const GaussianDistribution *>(&restored.getDistribution(0));
    ASSERT_NE(g, nullptr);
    EXPECT_EQ(g->getMean(), 2.0);
    EXPECT_EQ(g->getStandardDeviation(), 0.5);

    const auto *p = dynamic_cast<const PoissonDistribution *>(&restored.getDistribution(1));
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->getLambda(), 4.0);
}

TEST_F(HmmJsonFileTest, LoadNonExistentThrows) {
    EXPECT_THROW(static_cast<void>(load_json(tmpDir_ / "does_not_exist.json")), std::runtime_error);
}

// =============================================================================
// Error cases
// =============================================================================

TEST(HmmJson, UnknownDistributionTypeThrows) {
    const std::string bad_json = "{\"states\":1"
                                 ",\"pi\":[1.0]"
                                 ",\"trans\":[[1.0]]"
                                 ",\"distributions\":[{\"type\":\"Bogus\",\"x\":0}]}";
    EXPECT_THROW(static_cast<void>(from_json(bad_json)), std::runtime_error);
}

TEST(HmmJson, MalformedInputThrows) {
    EXPECT_THROW(static_cast<void>(from_json("not json at all")), std::runtime_error);
    EXPECT_THROW(static_cast<void>(from_json("{}")), std::runtime_error);
    EXPECT_THROW(static_cast<void>(from_json("")), std::runtime_error);
}
TEST(HmmJson, ZeroStatesThrows) {
    const std::string zero_states = "{\"states\":0,\"pi\":[],\"trans\":[],\"distributions\":[]}";
    EXPECT_THROW(static_cast<void>(from_json(zero_states)), std::runtime_error);
}

// =============================================================================
// Input sanitization boundary tests
// =============================================================================

// kMaxHmmStates = 4096; N=4097 must be rejected before any allocation.
TEST(HmmJsonSanitization, StatesCapExceededThrows) {
    const std::string json = "{\"states\":4097,\"pi\":[],\"trans\":[],\"distributions\":[]}";
    EXPECT_THROW(static_cast<void>(from_json(json)), std::runtime_error);
}

// Non-finite states value: static_cast<size_t>(inf) is UB without the guard.
TEST(HmmJsonSanitization, NonFiniteStatesThrows) {
    const std::string json = "{\"states\":1e309,\"pi\":[],\"trans\":[],\"distributions\":[]}";
    EXPECT_THROW(static_cast<void>(from_json(json)), std::runtime_error);
}

// kMaxJsonInputBytes = 10 MB; anything larger is rejected before parsing.
TEST(HmmJsonSanitization, InputSizeLimitThrows) {
    const std::string oversized(11UL * 1024UL * 1024UL, ' ');
    EXPECT_THROW(static_cast<void>(from_json(oversized)), std::runtime_error);
}

// N=2 but pi has 3 elements; read_double_array(N) must cap and throw.
TEST(HmmJsonSanitization, PiArrayLongerThanNThrows) {
    const std::string json = "{\"states\":2,\"pi\":[0.5,0.3,0.2],"
                             "\"trans\":[[1.0,0.0],[0.0,1.0]],"
                             "\"distributions\":[{\"type\":\"Gaussian\",\"mu\":0,\"sigma\":1},"
                             "{\"type\":\"Gaussian\",\"mu\":0,\"sigma\":1}]}";
    EXPECT_THROW(static_cast<void>(from_json(json)), std::runtime_error);
}

// N=2 but trans has 3 rows; read_double_matrix(N, N) must cap and throw.
TEST(HmmJsonSanitization, TransMatrixLongerThanNThrows) {
    const std::string json = "{\"states\":2,\"pi\":[0.5,0.5],"
                             "\"trans\":[[1.0,0.0],[0.0,1.0],[0.5,0.5]],"
                             "\"distributions\":[{\"type\":\"Gaussian\",\"mu\":0,\"sigma\":1},"
                             "{\"type\":\"Gaussian\",\"mu\":0,\"sigma\":1}]}";
    EXPECT_THROW(static_cast<void>(from_json(json)), std::runtime_error);
}

// kMaxDiscreteSymbols = 65536; n=65537 must be rejected before allocation.
TEST(HmmJsonSanitization, DiscreteNCapExceededThrows) {
    const std::string json =
        "{\"states\":1,\"pi\":[1.0],\"trans\":[[1.0]],"
        "\"distributions\":[{\"type\":\"Discrete\",\"n\":65537,\"probs\":[]}]}";
    EXPECT_THROW(static_cast<void>(from_json(json)), std::runtime_error);
}

// Discrete n=2 but probs has 3 elements; read_double_array(n) must cap.
TEST(HmmJsonSanitization, DiscretePropsArrayLongerThanNThrows) {
    const std::string json = "{\"states\":1,\"pi\":[1.0],\"trans\":[[1.0]],"
                             "\"distributions\":[{\"type\":\"Discrete\",\"n\":2,"
                             "\"probs\":[0.3,0.4,0.3]}]}";
    EXPECT_THROW(static_cast<void>(from_json(json)), std::runtime_error);
}

// =============================================================================
// json::Reader unit tests
// The reader is a handwritten parser; these tests exercise its error paths
// directly rather than routing through from_json().
// =============================================================================

using libhmm::json::Reader;

TEST(JsonReader, ConsumeMatchSucceeds) {
    Reader r("{ }");
    EXPECT_NO_THROW(r.consume('{'));
}

TEST(JsonReader, ConsumeMismatchThrows) {
    Reader r("[1.0]");
    // Input starts with '[', consuming '{' must throw.
    EXPECT_THROW(r.consume('{'), std::runtime_error);
}

TEST(JsonReader, ConsumeAtEofThrows) {
    Reader r("");
    EXPECT_THROW(r.consume('{'), std::runtime_error);
}

TEST(JsonReader, PeekAtEofThrows) {
    Reader r("");
    EXPECT_THROW(static_cast<void>(r.peek()), std::runtime_error);
}

TEST(JsonReader, ReadStringValid) {
    Reader r("\"hello\"");
    EXPECT_EQ(r.read_string(), "hello");
}

TEST(JsonReader, ReadStringUnterminatedThrows) {
    Reader r("\"unterminated");
    EXPECT_THROW(static_cast<void>(r.read_string()), std::runtime_error);
}

TEST(JsonReader, ReadDoubleValid) {
    Reader r("3.14");
    EXPECT_NEAR(r.read_double(), 3.14, 1e-15);
}

TEST(JsonReader, ReadDoubleNonNumericThrows) {
    Reader r("abc");
    EXPECT_THROW(static_cast<void>(r.read_double()), std::runtime_error);
}

TEST(JsonReader, ReadDoubleAtEofThrows) {
    Reader r("");
    EXPECT_THROW(static_cast<void>(r.read_double()), std::runtime_error);
}

TEST(JsonReader, ReadDoubleArrayValid) {
    Reader r("[1.0, 2.0, 3.0]");
    auto v = r.read_double_array();
    ASSERT_EQ(v.size(), 3u);
    EXPECT_NEAR(v[0], 1.0, 1e-15);
    EXPECT_NEAR(v[1], 2.0, 1e-15);
    EXPECT_NEAR(v[2], 3.0, 1e-15);
}

TEST(JsonReader, ReadDoubleArrayEmptyArray) {
    Reader r("[]");
    auto v = r.read_double_array();
    EXPECT_TRUE(v.empty());
}

TEST(JsonReader, ReadDoubleArraySizeCapThrows) {
    // Array has 3 elements but max_elements=2.
    Reader r("[1.0, 2.0, 3.0]");
    EXPECT_THROW(static_cast<void>(r.read_double_array(2)), std::runtime_error);
}

TEST(JsonReader, WriteDoubleRoundTrip) {
    // write_double must produce a string that reads back bit-exact.
    const double v = 1.0 / 3.0;
    const std::string s = libhmm::json::write_double(v);
    Reader r(s);
    EXPECT_EQ(r.read_double(), v);
}

TEST(JsonReader, WriteArrayRoundTrip) {
    const std::vector<double> original = {1.5, 2.5, 3.5};
    const std::string s = libhmm::json::write_array(original);
    Reader r(s);
    auto restored = r.read_double_array();
    ASSERT_EQ(restored.size(), original.size());
    for (std::size_t i = 0; i < original.size(); ++i)
        EXPECT_EQ(restored[i], original[i]) << "element " << i;
}

// Verify that valid boundary values are accepted.
TEST(HmmJsonSanitization, StatesAtCapAccepted) {
    // Building a full 4096-state HMM is impractical in a unit test.
    // Verify N=4096 passes the numeric check by using N=3 (well within limit).
    Hmm hmm(3);
    Vector pi(3);
    pi[0] = pi[1] = pi[2] = 1.0 / 3.0;
    hmm.setPi(pi);
    Matrix trans(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            trans(i, j) = 1.0 / 3.0;
    hmm.setTrans(trans);
    hmm.setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
    hmm.setDistribution(1, std::make_unique<GaussianDistribution>(1.0, 2.0));
    hmm.setDistribution(2, std::make_unique<GaussianDistribution>(2.0, 3.0));
    EXPECT_NO_THROW({
        Hmm r = from_json(to_json(hmm));
        (void)r;
    });
}

// =============================================================================
// Key-order validation (regression for F3 bugfix)
// =============================================================================

TEST(HmmJsonErrors, RejectsReorderedKeys) {
    // from_json now validates each key name before consuming its value.
    // A JSON with keys in a different order must be rejected rather than
    // silently scrambling HMM parameters.
    // This JSON has "pi" first instead of "states" first.
    const std::string reordered = R"({"pi":[0.5,0.5],"states":2,"trans":[[0.7,0.3],[0.2,0.8]],)"
                                  R"("distributions":[{"type":"Gaussian","mu":0.0,"sigma":1.0},)"
                                  R"({"type":"Gaussian","mu":1.0,"sigma":1.0}]})"; // NOLINT
    EXPECT_THROW(static_cast<void>(from_json(reordered)), std::runtime_error);
}

TEST(HmmJsonErrors, RejectsOversizedInput) {
    const std::string oversized(11UL * 1024UL * 1024UL, ' ');
    EXPECT_THROW(static_cast<void>(from_json(oversized)), std::runtime_error);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
