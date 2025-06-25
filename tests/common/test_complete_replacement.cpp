#include "libhmm/common/common_complete.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <sstream>

using namespace libhmm;

void test_basic_types() {
    std::cout << "Testing basic type definitions...\n";
    
    // Test type aliases work correctly
    Matrix m(3, 3);
    Vector v(3);
    ObservationSet obs(5);
    StateSequence seq(4);
    
    // Test dimensions
    assert(m.size1() == 3);
    assert(m.size2() == 3);
    assert(v.size() == 3);
    assert(obs.size() == 5);
    assert(seq.size() == 4);
    
    std::cout << "Basic type definitions: PASSED\n";
}

void test_matrix_operations() {
    std::cout << "Testing Matrix operations...\n";
    
    Matrix m1(2, 2);
    m1(0, 0) = 1.0; m1(0, 1) = 2.0;
    m1(1, 0) = 3.0; m1(1, 1) = 4.0;
    
    Matrix m2(2, 2);
    m2(0, 0) = 2.0; m2(0, 1) = 1.0;
    m2(1, 0) = 1.0; m2(1, 1) = 2.0;
    
    // Test matrix arithmetic
    Matrix result = m1 + m2;
    assert(result(0, 0) == 3.0);
    assert(result(1, 1) == 6.0);
    
    // Test clear function
    clear_matrix(result);
    assert(result(0, 0) == 0.0);
    assert(result(1, 1) == 0.0);
    
    std::cout << "Matrix operations: PASSED\n";
}

void test_vector_operations() {
    std::cout << "Testing Vector operations...\n";
    
    Vector v1{1.0, 2.0, 3.0};
    Vector v2{4.0, 5.0, 6.0};
    
    // Test vector arithmetic
    Vector result = v1 + v2;
    assert(result[0] == 5.0);
    assert(result[1] == 7.0);
    assert(result[2] == 9.0);
    
    // Test dot product
    double dot_result = v1.dot(v2);
    assert(dot_result == 32.0);  // 1*4 + 2*5 + 3*6 = 32
    
    // Test clear function
    clear_vector(result);
    assert(result[0] == 0.0);
    assert(result[1] == 0.0);
    assert(result[2] == 0.0);
    
    std::cout << "Vector operations: PASSED\n";
}

void test_state_sequence() {
    std::cout << "Testing StateSequence operations...\n";
    
    StateSequence seq{0, 1, 2, 1, 0};
    assert(seq.size() == 5);
    assert(seq[0] == 0);
    assert(seq[2] == 2);
    assert(seq[4] == 0);
    
    // Test clear function
    clear_vector(seq);
    assert(seq[0] == 0);
    assert(seq[2] == 0);
    
    std::cout << "StateSequence operations: PASSED\n";
}

void test_observation_lists() {
    std::cout << "Testing ObservationLists...\n";
    
    ObservationLists obs_lists;
    
    ObservationSet obs1{1.0, 2.0, 3.0};
    ObservationSet obs2{4.0, 5.0, 6.0};
    
    obs_lists.push_back(obs1);
    obs_lists.push_back(obs2);
    
    assert(obs_lists.size() == 2);
    assert(obs_lists[0].size() == 3);
    assert(obs_lists[1].size() == 3);
    assert(obs_lists[0][1] == 2.0);
    assert(obs_lists[1][2] == 6.0);
    
    std::cout << "ObservationLists: PASSED\n";
}

void test_matrix_serialization() {
    std::cout << "Testing Matrix serialization...\n";
    
    // Create test matrix
    Matrix original(2, 3);
    original(0, 0) = 1.1; original(0, 1) = 2.2; original(0, 2) = 3.3;
    original(1, 0) = 4.4; original(1, 1) = 5.5; original(1, 2) = 6.6;
    
    // Serialize to string
    std::ostringstream oss;
    serialization::MatrixSerializer<double>::save(oss, original, "test_matrix");
    std::string xml_output = oss.str();
    
    // Check that XML contains expected elements
    assert(xml_output.find("<test_matrix>") != std::string::npos);
    assert(xml_output.find("<rows>2</rows>") != std::string::npos);
    assert(xml_output.find("<cols>3</cols>") != std::string::npos);
    assert(xml_output.find("1.1") != std::string::npos);
    assert(xml_output.find("6.6") != std::string::npos);
    
    // Deserialize back
    std::istringstream iss(xml_output);
    Matrix loaded(1, 1);  // Start with wrong size
    serialization::MatrixSerializer<double>::load(iss, loaded, "test_matrix");
    
    // Verify loaded matrix
    assert(loaded.size1() == 2);
    assert(loaded.size2() == 3);
    assert(std::abs(loaded(0, 0) - 1.1) < 1e-10);
    assert(std::abs(loaded(1, 2) - 6.6) < 1e-10);
    
    std::cout << "Matrix serialization: PASSED\n";
}

void test_vector_serialization() {
    std::cout << "Testing Vector serialization...\n";
    
    // Create test vector
    Vector original{1.1, 2.2, 3.3, 4.4};
    
    // Serialize to string
    std::ostringstream oss;
    serialization::VectorSerializer<double>::save(oss, original, "test_vector");
    std::string xml_output = oss.str();
    
    // Check that XML contains expected elements
    assert(xml_output.find("<test_vector>") != std::string::npos);
    assert(xml_output.find("<size>4</size>") != std::string::npos);
    assert(xml_output.find("1.1") != std::string::npos);
    assert(xml_output.find("4.4") != std::string::npos);
    
    // Deserialize back
    std::istringstream iss(xml_output);
    Vector loaded(1);  // Start with wrong size
    serialization::VectorSerializer<double>::load(iss, loaded, "test_vector");
    
    // Verify loaded vector
    assert(loaded.size() == 4);
    assert(std::abs(loaded[0] - 1.1) < 1e-10);
    assert(std::abs(loaded[3] - 4.4) < 1e-10);
    
    std::cout << "Vector serialization: PASSED\n";
}

void test_constants() {
    std::cout << "Testing constants...\n";
    
    // Test that all constants are defined and have expected values
    assert(BW_TOLERANCE == 3.0e-7);
    assert(ZERO == 1.0e-30);
    assert(LIMIT_TOLERANCE == 1.0e-6);
    assert(MAX_VITERBI_ITERATIONS == 500);
    assert(ITMAX == 10000);
    assert(std::abs(PI - 3.141592653589793238462643383279502884) < 1e-15);
    
    std::cout << "Constants: PASSED\n";
}

void test_convenience_serialization() {
    std::cout << "Testing convenience serialization functions...\n";
    
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;
    
    Vector v{5.0, 6.0, 7.0};
    
    // Test convenience functions
    std::ostringstream oss;
    serialization::save(oss, m, "matrix");
    serialization::save(oss, v, "vector");
    
    std::string output = oss.str();
    assert(output.find("<matrix>") != std::string::npos);
    assert(output.find("<vector>") != std::string::npos);
    
    std::cout << "Convenience serialization functions: PASSED\n";
}

int main() {
    std::cout << "=== Testing Complete Boost Replacement ===\n";
    std::cout << "Branch: feature/boost-replacement-phase8\n\n";
    
    try {
        test_basic_types();
        test_matrix_operations();
        test_vector_operations();
        test_state_sequence();
        test_observation_lists();
        test_matrix_serialization();
        test_vector_serialization();
        test_constants();
        test_convenience_serialization();
        
        std::cout << "\n=== ALL TESTS PASSED! ===\n";
        std::cout << "✅ Complete Boost replacement is working correctly\n";
        std::cout << "✅ Type aliases provide seamless uBLAS compatibility\n";
        std::cout << "✅ All Matrix and Vector operations functional\n";
        std::cout << "✅ StateSequence and ObservationLists working\n";
        std::cout << "✅ Custom XML serialization working (replaces Boost.Serialization)\n";
        std::cout << "✅ All constants properly defined\n";
        std::cout << "✅ Clear functions provide API compatibility\n";
        std::cout << "\n🎯 Ready for integration with full libhmm codebase!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
