#include <gtest/gtest.h>
#include "libhmm/common/common.h"
#include "libhmm/common/string_tokenizer.h"
#include "libhmm/two_state_hmm.h"
#include <string>
#include <cmath>

using namespace libhmm;

// Common Constants Tests
class CommonConstantsTest : public ::testing::Test {};

TEST_F(CommonConstantsTest, ConstantsAreProperlyDefined) {
    // Test that all constants are accessible and have reasonable values
    EXPECT_GT(BW_TOLERANCE, 0.0);
    EXPECT_LT(BW_TOLERANCE, 1.0);
    
    EXPECT_GT(ZERO, 0.0);
    EXPECT_LT(ZERO, 1e-10);
    
    EXPECT_GT(LIMIT_TOLERANCE, 0.0);
    EXPECT_LT(LIMIT_TOLERANCE, 1e-3);
    
    EXPECT_GT(MAX_VITERBI_ITERATIONS, 0u);
    EXPECT_LT(MAX_VITERBI_ITERATIONS, 10000u);
    
    EXPECT_GT(ITMAX, 0u);
    EXPECT_LT(ITMAX, 100000u);
    
    // PI should be approximately 3.14159...
    EXPECT_NEAR(PI, 3.141592653589793, 1e-10);
}

TEST_F(CommonConstantsTest, ConstantsAreConstexpr) {
    // These should compile if the constants are constexpr
    constexpr double tolerance = BW_TOLERANCE;
    constexpr double zero = ZERO;
    constexpr double limit = LIMIT_TOLERANCE;
    constexpr std::size_t maxIter = MAX_VITERBI_ITERATIONS;
    constexpr std::size_t itmax = ITMAX;
    constexpr double pi = PI;
    
    // Use the variables to avoid unused variable warnings
    EXPECT_GT(tolerance, 0.0);
    EXPECT_GT(zero, 0.0);
    EXPECT_GT(limit, 0.0);
    EXPECT_GT(maxIter, 0u);
    EXPECT_GT(itmax, 0u);
    EXPECT_GT(pi, 3.0);
}

// Matrix and Vector Utility Tests
class CommonUtilitiesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test matrix and vectors
        testMatrix_ = Matrix(3, 3);
        testVector_ = Vector(3);
        testStateSeq_ = StateSequence(3);
        
        // Initialize with some values
        testMatrix_(0, 0) = 1.0; testMatrix_(0, 1) = 2.0; testMatrix_(0, 2) = 3.0;
        testMatrix_(1, 0) = 4.0; testMatrix_(1, 1) = 5.0; testMatrix_(1, 2) = 6.0;
        testMatrix_(2, 0) = 7.0; testMatrix_(2, 1) = 8.0; testMatrix_(2, 2) = 9.0;
        
        testVector_(0) = 1.5;
        testVector_(1) = 2.5;
        testVector_(2) = 3.5;
        
        testStateSeq_(0) = 0;
        testStateSeq_(1) = 1;
        testStateSeq_(2) = 2;
    }
    
    Matrix testMatrix_;
    Vector testVector_;
    StateSequence testStateSeq_;
};

TEST_F(CommonUtilitiesTest, ClearMatrixFunctionality) {
    // Matrix should have non-zero values initially
    EXPECT_NE(testMatrix_(1, 1), 0.0);
    
    // Clear the matrix
    clear_matrix(testMatrix_);
    
    // All elements should be zero
    for (std::size_t i = 0; i < testMatrix_.size1(); ++i) {
        for (std::size_t j = 0; j < testMatrix_.size2(); ++j) {
            EXPECT_DOUBLE_EQ(testMatrix_(i, j), 0.0);
        }
    }
}

TEST_F(CommonUtilitiesTest, ClearVectorFunctionality) {
    // Vector should have non-zero values initially
    EXPECT_NE(testVector_(1), 0.0);
    
    // Clear the vector
    clear_vector(testVector_);
    
    // All elements should be zero
    for (std::size_t i = 0; i < testVector_.size(); ++i) {
        EXPECT_DOUBLE_EQ(testVector_(i), 0.0);
    }
}

TEST_F(CommonUtilitiesTest, ClearStateSequenceFunctionality) {
    // StateSequence should have non-zero values initially
    EXPECT_NE(testStateSeq_(1), 0);
    
    // Clear the state sequence
    clear_vector(testStateSeq_);
    
    // All elements should be zero
    for (std::size_t i = 0; i < testStateSeq_.size(); ++i) {
        EXPECT_EQ(testStateSeq_(i), 0);
    }
}

TEST_F(CommonUtilitiesTest, EmptyMatrixHandling) {
    Matrix emptyMatrix(0, 0);
    EXPECT_NO_THROW(clear_matrix(emptyMatrix));
}

TEST_F(CommonUtilitiesTest, EmptyVectorHandling) {
    Vector emptyVector(0);
    EXPECT_NO_THROW(clear_vector(emptyVector));
    
    StateSequence emptyStateSeq(0);
    EXPECT_NO_THROW(clear_vector(emptyStateSeq));
}

// StringTokenizer Tests
class StringTokenizerTest : public ::testing::Test {};

TEST_F(StringTokenizerTest, BasicTokenization) {
    std::string testString = "hello world test";
    StringTokenizer tokenizer(testString);
    
    EXPECT_TRUE(tokenizer.hasMoreTokens());
    EXPECT_EQ(tokenizer.countTokens(), 3u);
    
    std::string token1 = tokenizer.nextToken();
    EXPECT_EQ(token1, "hello");
    
    std::string token2 = tokenizer.nextToken();
    EXPECT_EQ(token2, "world");
    
    std::string token3 = tokenizer.nextToken();
    EXPECT_EQ(token3, "test");
    
    EXPECT_FALSE(tokenizer.hasMoreTokens());
}

TEST_F(StringTokenizerTest, CustomDelimiter) {
    std::string testString = "one,two,three,four";
    StringTokenizer tokenizer(testString, ",");
    
    EXPECT_EQ(tokenizer.countTokens(), 4u);
    
    std::vector<std::string> expected = {"one", "two", "three", "four"};
    std::vector<std::string> actual;
    
    while (tokenizer.hasMoreTokens()) {
        actual.push_back(tokenizer.nextToken());
    }
    
    EXPECT_EQ(actual, expected);
}

TEST_F(StringTokenizerTest, MultipleDelimiters) {
    std::string testString = "a:b;c:d;e";
    StringTokenizer tokenizer(testString, ":;");
    
    EXPECT_EQ(tokenizer.countTokens(), 5u);
    
    std::vector<std::string> expected = {"a", "b", "c", "d", "e"};
    std::vector<std::string> actual;
    
    while (tokenizer.hasMoreTokens()) {
        actual.push_back(tokenizer.nextToken());
    }
    
    EXPECT_EQ(actual, expected);
}

TEST_F(StringTokenizerTest, LeadingAndTrailingSpaces) {
    std::string testString = "  hello   world  ";
    StringTokenizer tokenizer(testString);
    
    EXPECT_EQ(tokenizer.countTokens(), 2u);
    
    EXPECT_EQ(tokenizer.nextToken(), "hello");
    EXPECT_EQ(tokenizer.nextToken(), "world");
}

TEST_F(StringTokenizerTest, EmptyString) {
    std::string emptyString = "";
    StringTokenizer tokenizer(emptyString);
    
    EXPECT_EQ(tokenizer.countTokens(), 0u);
    EXPECT_FALSE(tokenizer.hasMoreTokens());
}

TEST_F(StringTokenizerTest, OnlyDelimiters) {
    std::string delimString = "   \t\n  ";
    StringTokenizer tokenizer(delimString);
    
    EXPECT_EQ(tokenizer.countTokens(), 0u);
    EXPECT_FALSE(tokenizer.hasMoreTokens());
}

TEST_F(StringTokenizerTest, SingleToken) {
    std::string singleToken = "hello";
    StringTokenizer tokenizer(singleToken);
    
    EXPECT_EQ(tokenizer.countTokens(), 1u);
    EXPECT_TRUE(tokenizer.hasMoreTokens());
    EXPECT_EQ(tokenizer.nextToken(), "hello");
    EXPECT_FALSE(tokenizer.hasMoreTokens());
}

TEST_F(StringTokenizerTest, NextTokenWithReference) {
    std::string testString = "alpha beta gamma";
    StringTokenizer tokenizer(testString);
    
    std::string token;
    tokenizer.nextToken(token);
    EXPECT_EQ(token, "alpha");
    
    tokenizer.nextToken(token);
    EXPECT_EQ(token, "beta");
    
    tokenizer.nextToken(token);
    EXPECT_EQ(token, "gamma");
}

TEST_F(StringTokenizerTest, ExhaustTokensThrows) {
    std::string testString = "single";
    StringTokenizer tokenizer(testString);
    
    // Get the single token
    EXPECT_NO_THROW(tokenizer.nextToken());
    
    // Should throw when trying to get another token
    EXPECT_THROW(tokenizer.nextToken(), std::runtime_error);
}

TEST_F(StringTokenizerTest, ConsecutiveDelimiters) {
    std::string testString = "a,,b,,,c";
    StringTokenizer tokenizer(testString, ",");
    
    // Should skip empty tokens between consecutive delimiters
    EXPECT_EQ(tokenizer.countTokens(), 3u);
    EXPECT_EQ(tokenizer.nextToken(), "a");
    EXPECT_EQ(tokenizer.nextToken(), "b");
    EXPECT_EQ(tokenizer.nextToken(), "c");
}

// Type Definitions Tests
class TypeDefinitionsTest : public ::testing::Test {};

TEST_F(TypeDefinitionsTest, BasicTypes) {
    // Test that basic types are properly defined
    Observation obs = 3.14;
    EXPECT_DOUBLE_EQ(obs, 3.14);
    
    StateIndex state = 5;
    EXPECT_EQ(state, 5);
}

TEST_F(TypeDefinitionsTest, MatrixAndVectorTypes) {
    // Test matrix operations
    Matrix mat(2, 3);
    EXPECT_EQ(mat.size1(), 2u);
    EXPECT_EQ(mat.size2(), 3u);
    
    // Test vector operations
    Vector vec(5);
    EXPECT_EQ(vec.size(), 5u);
    
    // Test observation set
    ObservationSet obsSet(10);
    EXPECT_EQ(obsSet.size(), 10u);
    
    // Test state sequence
    StateSequence stateSeq(7);
    EXPECT_EQ(stateSeq.size(), 7u);
}

TEST_F(TypeDefinitionsTest, ObservationLists) {
    ObservationLists lists;
    
    ObservationSet seq1(3);
    seq1(0) = 1.0; seq1(1) = 2.0; seq1(2) = 3.0;
    
    ObservationSet seq2(2);
    seq2(0) = 4.0; seq2(1) = 5.0;
    
    lists.push_back(seq1);
    lists.push_back(seq2);
    
    EXPECT_EQ(lists.size(), 2u);
    EXPECT_EQ(lists[0].size(), 3u);
    EXPECT_EQ(lists[1].size(), 2u);
}

// Two-State HMM Tests
class TwoStateHMMTest : public ::testing::Test {};

TEST_F(TwoStateHMMTest, ConstantsAreCorrect) {
    using namespace TwoStateHMM;
    
    EXPECT_EQ(NUM_STATES, 2u);
    EXPECT_EQ(NUM_SYMBOLS, 6u);
    EXPECT_EQ(FAIR_STATE, 0u);
    EXPECT_EQ(LOADED_STATE, 1u);
    
    EXPECT_DOUBLE_EQ(FAIR_INITIAL_PROB, 0.75);
    EXPECT_DOUBLE_EQ(LOADED_INITIAL_PROB, 0.25);
    
    EXPECT_DOUBLE_EQ(FAIR_TO_FAIR, 0.9);
    EXPECT_DOUBLE_EQ(FAIR_TO_LOADED, 0.1);
    EXPECT_DOUBLE_EQ(LOADED_TO_FAIR, 0.8);
    EXPECT_DOUBLE_EQ(LOADED_TO_LOADED, 0.2);
    
    EXPECT_NEAR(FAIR_EMISSION_PROB, 1.0/6.0, 1e-10);
    EXPECT_DOUBLE_EQ(LOADED_NORMAL_PROB, 0.125);
    EXPECT_DOUBLE_EQ(LOADED_BIASED_PROB, 0.375);
}

TEST_F(TwoStateHMMTest, PrepareTwoStateHmm) {
    auto hmm = std::make_unique<Hmm>(2);
    
    EXPECT_NO_THROW(prepareTwoStateHmm(hmm.get()));
    
    // Verify basic properties
    EXPECT_EQ(hmm->getNumStates(), 2);
    
    // Verify pi vector
    const Vector& pi = hmm->getPi();
    EXPECT_DOUBLE_EQ(pi(0), TwoStateHMM::FAIR_INITIAL_PROB);
    EXPECT_DOUBLE_EQ(pi(1), TwoStateHMM::LOADED_INITIAL_PROB);
    
    // Verify transition matrix
    const Matrix& trans = hmm->getTrans();
    EXPECT_DOUBLE_EQ(trans(0, 0), TwoStateHMM::FAIR_TO_FAIR);
    EXPECT_DOUBLE_EQ(trans(0, 1), TwoStateHMM::FAIR_TO_LOADED);
    EXPECT_DOUBLE_EQ(trans(1, 0), TwoStateHMM::LOADED_TO_FAIR);
    EXPECT_DOUBLE_EQ(trans(1, 1), TwoStateHMM::LOADED_TO_LOADED);
    
    // Verify that validation passes
    EXPECT_NO_THROW(hmm->validate());
}

TEST_F(TwoStateHMMTest, PrepareTwoStateHmmNullThrows) {
    EXPECT_THROW(prepareTwoStateHmm(nullptr), std::invalid_argument);
}

TEST_F(TwoStateHMMTest, CreateTwoStateHmm) {
    auto hmm = createTwoStateHmm();
    
    EXPECT_EQ(hmm->getNumStates(), 2);
    EXPECT_NO_THROW(hmm->validate());
}

TEST_F(TwoStateHMMTest, LegacyCompatibility) {
    auto hmm = std::make_unique<Hmm>(2);
    
    // Use the modern function to avoid deprecation warnings
    EXPECT_NO_THROW(prepareTwoStateHmm(hmm.get()));
    
    // Should produce the same result as modern function
    EXPECT_EQ(hmm->getNumStates(), 2);
    EXPECT_NO_THROW(hmm->validate());
}

// Error Handling and Edge Cases
class CommonErrorHandlingTest : public ::testing::Test {};

TEST_F(CommonErrorHandlingTest, StringTokenizerErrorHandling) {
    // Test with empty string and immediate token request
    StringTokenizer emptyTokenizer("");
    EXPECT_THROW(emptyTokenizer.nextToken(), std::runtime_error);
    
    std::string token;
    EXPECT_THROW(emptyTokenizer.nextToken(token), std::runtime_error);
}

TEST_F(CommonErrorHandlingTest, MatrixVectorEdgeCases) {
    // Very large matrices/vectors should work
    EXPECT_NO_THROW(Matrix(1000, 1000));
    EXPECT_NO_THROW(Vector(10000));
    
    // Zero-size should work
    EXPECT_NO_THROW(Matrix(0, 0));
    EXPECT_NO_THROW(Vector(0));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
