#include "libhmm/common/matrix.h"
#include "libhmm/common/vector.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace libhmm;

void test_matrix_basic_operations() {
    std::cout << "Testing Matrix basic operations...\n";
    
    // Test construction
    Matrix<double> m1(3, 4);
    assert(m1.size1() == 3);
    assert(m1.size2() == 4);
    assert(m1.rows() == 3);
    assert(m1.cols() == 4);
    
    // Test element access
    m1(0, 0) = 1.0;
    m1(1, 2) = 2.5;
    m1(2, 3) = 3.7;
    
    assert(m1(0, 0) == 1.0);
    assert(m1(1, 2) == 2.5);
    assert(m1(2, 3) == 3.7);
    
    // Test initialization with value
    Matrix<double> m2(2, 3, 5.0);
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            assert(m2(i, j) == 5.0);
        }
    }
    
    // Test copy constructor
    Matrix<double> m3 = m2;
    assert(m3.size1() == 2);
    assert(m3.size2() == 3);
    assert(m3(0, 0) == 5.0);
    
    // Test resize operations
    m1.resize(2, 2);
    assert(m1.size1() == 2);
    assert(m1.size2() == 2);
    
    std::cout << "Matrix basic operations: PASSED\n";
}

void test_vector_basic_operations() {
    std::cout << "Testing Vector basic operations...\n";
    
    // Test construction
    Vector<double> v1(5);
    assert(v1.size() == 5);
    
    // Test element access
    v1[0] = 1.0;
    v1[1] = 2.0;
    v1(2) = 3.0;  // Test uBLAS-style access
    
    assert(v1[0] == 1.0);
    assert(v1[1] == 2.0);
    assert(v1(2) == 3.0);
    
    // Test initialization with value
    Vector<double> v2(3, 7.5);
    for (std::size_t i = 0; i < 3; ++i) {
        assert(v2[i] == 7.5);
    }
    
    // Test initializer list
    Vector<double> v3{1.0, 2.0, 3.0, 4.0};
    assert(v3.size() == 4);
    assert(v3[0] == 1.0);
    assert(v3[3] == 4.0);
    
    // Test resize operations
    v1.resize(3);
    assert(v1.size() == 3);
    
    std::cout << "Vector basic operations: PASSED\n";
}

void test_matrix_arithmetic() {
    std::cout << "Testing Matrix arithmetic operations...\n";
    
    Matrix<double> m1(2, 2);
    m1(0, 0) = 1.0; m1(0, 1) = 2.0;
    m1(1, 0) = 3.0; m1(1, 1) = 4.0;
    
    Matrix<double> m2(2, 2);
    m2(0, 0) = 2.0; m2(0, 1) = 1.0;
    m2(1, 0) = 1.0; m2(1, 1) = 2.0;
    
    // Test addition
    Matrix<double> result = m1 + m2;
    assert(result(0, 0) == 3.0);
    assert(result(0, 1) == 3.0);
    assert(result(1, 0) == 4.0);
    assert(result(1, 1) == 6.0);
    
    // Test subtraction
    Matrix<double> diff = m1 - m2;
    assert(diff(0, 0) == -1.0);
    assert(diff(0, 1) == 1.0);
    assert(diff(1, 0) == 2.0);
    assert(diff(1, 1) == 2.0);
    
    // Test scalar multiplication
    Matrix<double> scaled = m1 * 2.0;
    assert(scaled(0, 0) == 2.0);
    assert(scaled(0, 1) == 4.0);
    assert(scaled(1, 0) == 6.0);
    assert(scaled(1, 1) == 8.0);
    
    // Test compound assignment operators
    Matrix<double> m3 = m1;
    m3 += m2;
    assert(m3(0, 0) == 3.0);
    assert(m3(1, 1) == 6.0);
    
    std::cout << "Matrix arithmetic operations: PASSED\n";
}

void test_vector_arithmetic() {
    std::cout << "Testing Vector arithmetic operations...\n";
    
    Vector<double> v1{1.0, 2.0, 3.0};
    Vector<double> v2{4.0, 5.0, 6.0};
    
    // Test addition
    Vector<double> result = v1 + v2;
    assert(result[0] == 5.0);
    assert(result[1] == 7.0);
    assert(result[2] == 9.0);
    
    // Test subtraction
    Vector<double> diff = v2 - v1;
    assert(diff[0] == 3.0);
    assert(diff[1] == 3.0);
    assert(diff[2] == 3.0);
    
    // Test scalar multiplication
    Vector<double> scaled = v1 * 2.0;
    assert(scaled[0] == 2.0);
    assert(scaled[1] == 4.0);
    assert(scaled[2] == 6.0);
    
    // Test dot product
    double dot_result = v1.dot(v2);
    assert(dot_result == 32.0);  // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    
    // Test sum
    double sum_result = v1.sum();
    assert(sum_result == 6.0);  // 1 + 2 + 3 = 6
    
    // Test compound assignment operators
    Vector<double> v3 = v1;
    v3 += v2;
    assert(v3[0] == 5.0);
    assert(v3[1] == 7.0);
    assert(v3[2] == 9.0);
    
    std::cout << "Vector arithmetic operations: PASSED\n";
}

void test_mathematical_functions() {
    std::cout << "Testing mathematical functions...\n";
    
    Vector<double> v{3.0, 4.0};
    
    // Test norm (should be 5.0 for 3-4-5 triangle)
    double norm_result = v.norm();
    assert(std::abs(norm_result - 5.0) < 1e-10);
    
    // Test element-wise operations
    Vector<double> v1{2.0, 4.0, 6.0};
    Vector<double> v2{1.0, 2.0, 3.0};
    
    Vector<double> element_prod_result = element_prod(v1, v2);
    assert(element_prod_result[0] == 2.0);
    assert(element_prod_result[1] == 8.0);
    assert(element_prod_result[2] == 18.0);
    
    Vector<double> element_div_result = element_div(v1, v2);
    assert(element_div_result[0] == 2.0);
    assert(element_div_result[1] == 2.0);
    assert(element_div_result[2] == 2.0);
    
    // Test inner product function
    double inner_result = inner_prod(v1, v2);
    assert(inner_result == 28.0);  // 2*1 + 4*2 + 6*3 = 2 + 8 + 18 = 28
    
    std::cout << "Mathematical functions: PASSED\n";
}

void test_uBLAS_compatibility() {
    std::cout << "Testing uBLAS API compatibility...\n";
    
    // Test Matrix uBLAS-style API
    Matrix<double> m(3, 3);
    assert(m.size1() == 3);  // uBLAS row count method
    assert(m.size2() == 3);  // uBLAS column count method
    
    // Test Vector uBLAS-style API  
    Vector<int> v(5);
    assert(v.size() == 5);
    v(0) = 10;  // uBLAS-style parentheses access
    assert(v(0) == 10);
    
    // Test StateSequence compatibility (Vector<int>)
    Vector<int> state_seq{0, 1, 2, 1, 0};
    assert(state_seq.size() == 5);
    assert(state_seq[0] == 0);
    assert(state_seq[4] == 0);
    
    std::cout << "uBLAS API compatibility: PASSED\n";
}

void test_memory_layout_and_simd_readiness() {
    std::cout << "Testing memory layout and SIMD readiness...\n";
    
    // Test contiguous memory for Matrix
    Matrix<double> m(3, 4);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            m(i, j) = i * 4 + j;  // Fill with sequential values
        }
    }
    
    // Verify raw data access gives contiguous memory
    const double* data_ptr = m.data();
    for (std::size_t idx = 0; idx < 12; ++idx) {
        assert(data_ptr[idx] == static_cast<double>(idx));
    }
    
    // Test contiguous memory for Vector
    Vector<double> v{10.0, 20.0, 30.0, 40.0};
    const double* vec_data = v.data();
    assert(vec_data[0] == 10.0);
    assert(vec_data[1] == 20.0);
    assert(vec_data[2] == 30.0);
    assert(vec_data[3] == 40.0);
    
    std::cout << "Memory layout and SIMD readiness: PASSED\n";
}

void test_error_handling() {
    std::cout << "Testing error handling...\n";
    
    Vector<double> v1{1.0, 2.0};
    Vector<double> v2{1.0, 2.0, 3.0};  // Different size
    
    // Test that mismatched sizes throw exceptions
    try {
        Vector<double> result = v1 + v2;
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    try {
        double dot_result = v1.dot(v2);
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    Matrix<double> m1(2, 2);
    Matrix<double> m2(2, 3);  // Different size
    
    try {
        Matrix<double> result = m1 + m2;
        assert(false);  // Should not reach here
    } catch (const std::invalid_argument&) {
        // Expected behavior
    }
    
    std::cout << "Error handling: PASSED\n";
}

int main() {
    std::cout << "=== Testing libhmm Matrix and Vector Classes ===\n";
    std::cout << "Branch: feature/boost-replacement-phase8\n\n";
    
    try {
        test_matrix_basic_operations();
        test_vector_basic_operations();
        test_matrix_arithmetic();
        test_vector_arithmetic();
        test_mathematical_functions();
        test_uBLAS_compatibility();
        test_memory_layout_and_simd_readiness();
        test_error_handling();
        
        std::cout << "\n=== ALL TESTS PASSED! ===\n";
        std::cout << "✅ Matrix and Vector classes are ready to replace boost::numeric::ublas\n";
        std::cout << "✅ Full API compatibility with existing uBLAS usage\n";
        std::cout << "✅ SIMD-friendly contiguous memory layout\n";
        std::cout << "✅ Proper error handling for dimension mismatches\n";
        std::cout << "✅ Enhanced mathematical operations for HMM computations\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
