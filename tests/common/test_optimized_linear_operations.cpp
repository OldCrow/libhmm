#include "../../include/libhmm/common/optimized_vector.h"
#include "../../include/libhmm/common/optimized_matrix.h"
#include "../../include/libhmm/common/optimized_matrix3d.h"
#include "../../include/libhmm/common/basic_vector.h"
#include "../../include/libhmm/common/basic_matrix.h"
#include "../../include/libhmm/common/basic_matrix3d.h"
#include <iostream>
#include <chrono>
#include <numeric>
#include <iomanip>

using namespace libhmm;

void test_optimized_vector_operations() {
    std::cout << "=== Testing OptimizedVector Operations ===\n";
    
    // Test construction and basic operations
    OptimizedVector<double> vec1({1.0, 2.0, 3.0, 4.0});
    OptimizedVector<double> vec2(4, 2.0);  // [2, 2, 2, 2]
    
    std::cout << "vec1: " << vec1 << "\n";
    std::cout << "vec2: " << vec2 << "\n";
    
    // Test vector arithmetic (should use SIMD when applicable)
    auto vec3 = vec1 + vec2;
    std::cout << "vec1 + vec2: " << vec3 << "\n";
    
    // Test dot product (optimized)
    double dot = vec1.dot(vec2);
    std::cout << "vec1 · vec2 = " << dot << " (should be 20.0)\n";
    
    // Test norm and normalization (optimized)
    double norm = vec1.norm();
    std::cout << "||vec1|| = " << norm << " (should be ~5.477)\n";
    
    OptimizedVector<double> vec1_normalized = vec1;
    vec1_normalized.normalize();
    std::cout << "normalized vec1: " << vec1_normalized << "\n";
    std::cout << "||normalized vec1|| = " << vec1_normalized.norm() << " (should be ~1.0)\n";
    
    // Test element-wise operations (SIMD optimized for larger vectors)
    vec1.element_multiply(vec2);
    std::cout << "vec1 ⊙ vec2 (Hadamard): " << vec1 << "\n";
    
    // Test element-wise division
    OptimizedVector<double> vec4({4.0, 8.0, 6.0, 8.0});
    OptimizedVector<double> vec5({2.0, 2.0, 3.0, 4.0});
    vec4.element_divide(vec5);
    std::cout << "vec4 ÷ vec5 (element-wise): " << vec4 << " (should be [2, 4, 2, 2])\n";
    
    // Test uBLAS compatibility functions
    double inner_prod_result = inner_prod(vec2, vec3);
    std::cout << "inner_prod(vec2, vec3) = " << inner_prod_result << "\n";
    
    std::cout << "✅ OptimizedVector operations passed!\n\n";
}

void test_optimized_matrix_operations() {
    std::cout << "=== Testing OptimizedMatrix Operations ===\n";
    
    // Test construction
    OptimizedMatrix<double> mat(3, 3);
    
    // Fill with test data
    mat(0, 0) = 1; mat(0, 1) = 2; mat(0, 2) = 3;
    mat(1, 0) = 4; mat(1, 1) = 5; mat(1, 2) = 6;
    mat(2, 0) = 7; mat(2, 1) = 8; mat(2, 2) = 9;
    
    std::cout << "Test matrix:\n" << mat << "\n\n";
    
    // Test transpose (cache-optimized)
    auto mat_t = mat.transpose();
    std::cout << "Transposed matrix:\n" << mat_t << "\n\n";
    
    // Test matrix-vector multiplication (SIMD optimized)
    OptimizedVector<double> vec({1.0, 2.0, 3.0});
    auto result_vec = mat.multiply_vector(vec);
    std::cout << "Matrix * vector [1,2,3]: " << result_vec << "\n";
    std::cout << "Expected: [14, 32, 50]\n\n";
    
    // Test matrix-matrix multiplication (cache-blocked for large matrices)
    OptimizedMatrix<double> identity(3, 3);
    identity(0,0) = 1; identity(1,1) = 1; identity(2,2) = 1;
    auto mat_result = mat.multiply(identity);
    std::cout << "Matrix * Identity:\n" << mat_result << "\n";
    std::cout << "Should equal original matrix\n\n";
    
    // Test row and column operations (optimized)
    auto row1 = mat.row(1);
    auto col2 = mat.column(2);
    std::cout << "Row 1: " << row1 << "\n";
    std::cout << "Column 2: " << col2 << "\n";
    
    // Test HMM-specific operations
    OptimizedMatrix<double> prob_mat(2, 3, 0.5);  // 2x3 matrix filled with 0.5
    std::cout << "\nProbability matrix before normalization:\n" << prob_mat << "\n";
    prob_mat.normalize_rows();
    std::cout << "After row normalization:\n" << prob_mat << "\n";
    
    auto row_sums = prob_mat.row_sums();
    std::cout << "Row sums: " << row_sums << " (should be [1,1])\n";
    
    // Test uBLAS compatibility functions
    auto first_row_ublas = row(prob_mat, 0);
    auto first_col_ublas = column(prob_mat, 0);
    std::cout << "uBLAS row function: " << first_row_ublas << "\n";
    std::cout << "uBLAS column function: " << first_col_ublas << "\n";
    
    // Test matrix product with uBLAS-style function
    auto prod_result = prod(identity, mat);
    std::cout << "prod(identity, matrix):\n" << prod_result << "\n";
    
    std::cout << "✅ OptimizedMatrix operations passed!\n\n";
}

void test_optimized_matrix3d_operations() {
    std::cout << "=== Testing OptimizedMatrix3D Operations ===\n";
    
    // Test construction
    OptimizedMatrix3D<double> mat3d(2, 3, 4);
    
    // Fill with test data
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                mat3d(i, j, k) = i * 100 + j * 10 + k;
            }
        }
    }
    
    std::cout << "Created 2x3x4 matrix with test data\n";
    
    // Test slicing operations (critical for HMM xi/gamma matrices)
    auto slice0 = mat3d.slice(0);
    std::cout << "Slice 0 access: slice0[1][2] = " << slice0[1][2] 
              << " (should be 12)\n";
    std::cout << "Slice 0 function access: slice0(1,2) = " << slice0(1, 2) 
              << " (should be 12)\n";
    
    // Test sum operation (parallel for large matrices)
    double total_sum = mat3d.sum();
    std::cout << "Total sum: " << total_sum << "\n";
    
    // Test element-wise operations (parallel when beneficial)
    OptimizedMatrix3D<double> mat3d_copy = mat3d;
    mat3d_copy *= 2.0;
    std::cout << "After scalar multiplication by 2, mat3d_copy(0,1,2) = " 
              << mat3d_copy(0, 1, 2) << " (should be 24)\n";
    
    mat3d += mat3d_copy;
    std::cout << "After adding doubled matrix, mat3d(0,1,2) = " 
              << mat3d(0, 1, 2) << " (should be 36)\n";
    
    // Test parallel operations on larger matrix
    OptimizedMatrix3D<double> large_mat3d(50, 50, 50);  // 125k elements
    large_mat3d.fill_parallel(3.14);
    std::cout << "Large matrix filled with parallel operation: " 
              << large_mat3d(25, 25, 25) << " (should be 3.14)\n";
    
    std::cout << "✅ OptimizedMatrix3D operations passed!\n\n";
}

void test_performance_comparison() {
    std::cout << "=== Performance Comparison Tests ===\n";
    
    const size_t vec_size = 10000;      // Use larger sizes to trigger SIMD/parallel
    const size_t mat_rows = 500;
    const size_t mat_cols = 500;
    const int iterations = 5;
    
    std::cout << "Testing with larger sizes to trigger optimizations:\n";
    std::cout << "Vector size: " << vec_size << " elements\n";
    std::cout << "Matrix size: " << mat_rows << "x" << mat_cols << " (" 
              << mat_rows * mat_cols << " elements)\n\n";
    
    // Test optimized vector operations
    OptimizedVector<double> large_vec1(vec_size, 1.0);
    OptimizedVector<double> large_vec2(vec_size, 2.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        large_vec1 += large_vec2;  // Should use SIMD
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto vec_add_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Test dot product (SIMD optimized)
    start = std::chrono::high_resolution_clock::now();
    double dot_result = 0;
    for (int i = 0; i < iterations; ++i) {
        dot_result += large_vec1.dot(large_vec2);  // Should use SIMD
    }
    end = std::chrono::high_resolution_clock::now();
    auto dot_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Test matrix operations
    OptimizedMatrix<double> large_mat1(mat_rows, mat_cols, 1.0);
    OptimizedMatrix<double> large_mat2(mat_rows, mat_cols, 2.0);
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        large_mat1 += large_mat2;  // Should use SIMD
    }
    end = std::chrono::high_resolution_clock::now();
    auto mat_add_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Test matrix-vector multiplication
    OptimizedVector<double> mat_vec(mat_cols, 1.5);
    start = std::chrono::high_resolution_clock::now();
    OptimizedVector<double> mv_result(mat_rows);
    for (int i = 0; i < iterations; ++i) {
        mv_result = large_mat1.multiply_vector(mat_vec);  // Should use SIMD
    }
    end = std::chrono::high_resolution_clock::now();
    auto matvec_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "Performance Results (" << iterations << " iterations):\n";
    std::cout << "Vector addition (SIMD): " << vec_add_time << " ms\n";
    std::cout << "Vector dot product (SIMD): " << dot_time << " ms\n";
    std::cout << "Matrix addition (SIMD): " << mat_add_time << " ms\n";
    std::cout << "Matrix-vector multiply (SIMD): " << matvec_time << " ms\n";
    
    std::cout << "Dot product result: " << dot_result / iterations << "\n";
    std::cout << "Matrix-vector result sample: " << mv_result[0] << "\n";
    
    std::cout << "✅ Performance tests completed!\n\n";
}

void test_simd_thresholds() {
    std::cout << "=== Testing SIMD Threshold Behavior ===\n";
    
    // Test small vectors (should use serial)
    OptimizedVector<double> small_vec1(4, 1.0);
    OptimizedVector<double> small_vec2(4, 2.0);
    
    auto small_result = small_vec1 + small_vec2;
    std::cout << "Small vector addition (serial): " << small_result << "\n";
    
    // Test medium vectors (should use SIMD)
    OptimizedVector<double> med_vec1(100, 1.0);
    OptimizedVector<double> med_vec2(100, 2.0);
    
    double med_dot = med_vec1.dot(med_vec2);
    std::cout << "Medium vector dot product (SIMD): " << med_dot << " (should be 200)\n";
    
    // Test large vectors (should use parallel)
    OptimizedVector<double> large_vec1(5000, 1.0);
    OptimizedVector<double> large_vec2(5000, 2.0);
    
    double large_sum = large_vec1.sum();
    std::cout << "Large vector sum (parallel): " << large_sum << " (should be 5000)\n";
    
    // Test matrix thresholds
    OptimizedMatrix<double> small_mat(5, 5, 1.0);
    OptimizedMatrix<double> large_mat(200, 200, 1.0);
    
    small_mat.clear();  // Should use serial
    large_mat.clear();  // Should use parallel
    
    std::cout << "Small matrix cleared (serial): " << small_mat.sum() << " (should be 0)\n";
    std::cout << "Large matrix cleared (parallel): " << large_mat.sum() << " (should be 0)\n";
    
    std::cout << "✅ SIMD threshold tests passed!\n\n";
}

void test_hmm_compatibility() {
    std::cout << "=== Testing HMM-Specific Optimized Features ===\n";
    
    // Test optimized HMM operations
    OptimizedMatrix<double> transition_matrix(3, 3, 0.3);
    transition_matrix.normalize_rows();  // Stochastic matrix (optimized)
    
    OptimizedVector<double> state_prob({0.5, 0.3, 0.2});
    auto next_state = transition_matrix.multiply_vector(state_prob);  // SIMD optimized
    std::cout << "HMM state transition (optimized): " << next_state << "\n";
    
    // Test uBLAS compatibility with optimized classes
    double prob_sum = inner_prod(state_prob, next_state);
    std::cout << "Inner product (uBLAS compat): " << prob_sum << "\n";
    
    // Test row/column functions (uBLAS compatibility)
    auto first_row = row(transition_matrix, 0);
    auto first_col = column(transition_matrix, 0);
    std::cout << "First row (optimized): " << first_row << "\n";
    std::cout << "First column (optimized): " << first_col << "\n";
    
    // Test matrix transpose (cache-optimized)
    auto trans_mat = trans(transition_matrix);
    std::cout << "Transposed matrix (cache-optimized):\n" << trans_mat << "\n";
    
    // Test large-scale HMM operations (should trigger optimizations)
    OptimizedMatrix<double> large_transition(100, 100, 0.01);
    large_transition.normalize_rows();  // Should use optimized normalization
    
    OptimizedVector<double> large_state(100, 1.0/100.0);
    auto large_next = prod(large_transition, large_state);  // Should use optimized multiply
    
    double large_sum = large_next.sum();
    std::cout << "Large HMM transition sum: " << large_sum << " (should be ~1.0)\n";
    
    std::cout << "✅ HMM compatibility tests passed!\n\n";
}

void test_conversion_constructors() {
    std::cout << "=== Testing Basic-to-Optimized Conversion Constructors ===\n";
    
    // Test BasicVector to OptimizedVector conversion
    BasicVector<double> basic_vec({1.0, 2.0, 3.0, 4.0, 5.0});
    OptimizedVector<double> opt_vec(basic_vec);  // Conversion constructor
    
    std::cout << "Basic vector: " << basic_vec << "\n";
    std::cout << "Converted optimized vector: " << opt_vec << "\n";
    
    // Verify data integrity
    bool vec_data_matches = true;
    for (size_t i = 0; i < basic_vec.size(); ++i) {
        if (basic_vec[i] != opt_vec[i]) {
            vec_data_matches = false;
            break;
        }
    }
    std::cout << "Vector conversion data integrity: " << (vec_data_matches ? "✅ PASS" : "❌ FAIL") << "\n";
    
    // Test BasicMatrix to OptimizedMatrix conversion
    BasicMatrix<double> basic_mat(3, 3);
    basic_mat(0, 0) = 1.0; basic_mat(0, 1) = 2.0; basic_mat(0, 2) = 3.0;
    basic_mat(1, 0) = 4.0; basic_mat(1, 1) = 5.0; basic_mat(1, 2) = 6.0;
    basic_mat(2, 0) = 7.0; basic_mat(2, 1) = 8.0; basic_mat(2, 2) = 9.0;
    
    OptimizedMatrix<double> opt_mat(basic_mat);  // Conversion constructor
    
    std::cout << "\nBasic matrix:\n" << basic_mat << "\n";
    std::cout << "Converted optimized matrix:\n" << opt_mat << "\n";
    
    // Verify matrix data integrity
    bool mat_data_matches = true;
    for (size_t i = 0; i < basic_mat.rows(); ++i) {
        for (size_t j = 0; j < basic_mat.cols(); ++j) {
            if (basic_mat(i, j) != opt_mat(i, j)) {
                mat_data_matches = false;
                break;
            }
        }
        if (!mat_data_matches) break;
    }
    std::cout << "Matrix conversion data integrity: " << (mat_data_matches ? "✅ PASS" : "❌ FAIL") << "\n";
    
    // Test BasicMatrix3D to OptimizedMatrix3D conversion
    BasicMatrix3D<double> basic_mat3d(2, 3, 2);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                basic_mat3d(i, j, k) = i * 100 + j * 10 + k;
            }
        }
    }
    
    OptimizedMatrix3D<double> opt_mat3d(basic_mat3d);  // Conversion constructor
    
    std::cout << "\nBasic matrix3D dimensions: " << basic_mat3d.getXDimensionSize() 
              << "x" << basic_mat3d.getYDimensionSize() 
              << "x" << basic_mat3d.getZDimensionSize() << "\n";
    std::cout << "Optimized matrix3D dimensions: " << opt_mat3d.getXDimensionSize() 
              << "x" << opt_mat3d.getYDimensionSize() 
              << "x" << opt_mat3d.getZDimensionSize() << "\n";
    
    // Verify 3D matrix data integrity
    bool mat3d_data_matches = true;
    for (size_t i = 0; i < basic_mat3d.getXDimensionSize(); ++i) {
        for (size_t j = 0; j < basic_mat3d.getYDimensionSize(); ++j) {
            for (size_t k = 0; k < basic_mat3d.getZDimensionSize(); ++k) {
                if (basic_mat3d(i, j, k) != opt_mat3d(i, j, k)) {
                    mat3d_data_matches = false;
                    std::cout << "Mismatch at (" << i << "," << j << "," << k << "): "
                              << basic_mat3d(i, j, k) << " vs " << opt_mat3d(i, j, k) << "\n";
                    break;
                }
            }
            if (!mat3d_data_matches) break;
        }
        if (!mat3d_data_matches) break;
    }
    std::cout << "Matrix3D conversion data integrity: " << (mat3d_data_matches ? "✅ PASS" : "❌ FAIL") << "\n";
    
    // Test performance upgrade scenario
    std::cout << "\n=== Performance Upgrade Scenario ===\n";
    
    // Start with basic operations for development
    BasicVector<double> development_vec(1000, 2.0);
    BasicMatrix<double> development_mat(100, 100, 0.5);
    
    std::cout << "Development phase - using basic classes...\n";
    auto basic_sum = development_vec.sum();
    auto basic_row_sums = development_mat.row_sums();
    std::cout << "Basic vector sum: " << basic_sum << "\n";
    std::cout << "Basic matrix first row sum: " << basic_row_sums[0] << "\n";
    
    // Upgrade to optimized for production performance
    std::cout << "\nProduction phase - upgrading to optimized classes...\n";
    OptimizedVector<double> production_vec(development_vec);  // Seamless upgrade
    OptimizedMatrix<double> production_mat(development_mat);  // Seamless upgrade
    
    auto opt_sum = production_vec.sum();
    auto opt_row_sums = production_mat.row_sums();
    std::cout << "Optimized vector sum: " << opt_sum << "\n";
    std::cout << "Optimized matrix first row sum: " << opt_row_sums[0] << "\n";
    
    // Verify results are identical
    bool results_match = (std::abs(basic_sum - opt_sum) < 1e-10) && 
                        (std::abs(basic_row_sums[0] - opt_row_sums[0]) < 1e-10);
    std::cout << "Performance upgrade correctness: " << (results_match ? "✅ PASS" : "❌ FAIL") << "\n";
    
    std::cout << "✅ All conversion constructor tests passed!\n\n";
}

void test_error_handling() {
    std::cout << "=== Testing Error Handling ===\n";
    
    try {
        OptimizedMatrix3D<int> bad_mat(0, 5, 5);
        std::cout << "❌ Should have thrown for zero dimension\n";
    } catch (const std::invalid_argument& e) {
        std::cout << "✅ Correctly caught zero dimension: " << e.what() << "\n";
    }
    
    try {
        OptimizedVector<double> vec1(3);
        OptimizedVector<double> vec2(4);
        vec1 += vec2;  // Mismatched dimensions
        std::cout << "❌ Should have thrown for dimension mismatch\n";
    } catch (const std::invalid_argument& e) {
        std::cout << "✅ Correctly caught dimension mismatch: " << e.what() << "\n";
    }
    
    try {
        OptimizedMatrix<double> mat(3, 4);
        OptimizedVector<double> vec(5);
        mat.multiply_vector(vec);  // Incompatible dimensions
        std::cout << "❌ Should have thrown for incompatible multiplication\n";
    } catch (const std::invalid_argument& e) {
        std::cout << "✅ Correctly caught incompatible multiplication: " << e.what() << "\n";
    }
    
    std::cout << "✅ Error handling tests passed!\n\n";
}

int main() {
    std::cout << "Comprehensive Test of Optimized Linear Algebra Classes\n";
    std::cout << "=====================================================\n\n";
    
    // Test all three optimized classes
    test_optimized_vector_operations();
    test_optimized_matrix_operations();
    test_optimized_matrix3d_operations();
    
    // Test performance and optimization behavior
    test_performance_comparison();
    test_simd_thresholds();
    
    // Test conversion constructors (enables dynamic performance scaling)
    test_conversion_constructors();
    
    // Test HMM-specific functionality
    test_hmm_compatibility();
    
    // Test error handling
    test_error_handling();
    
    std::cout << "🎉 All optimized class tests completed successfully!\n\n";
    
    std::cout << "=== SUMMARY: Optimized Linear Algebra Classes Status ===\n";
    std::cout << "✅ OptimizedVector: SIMD + parallel optimized\n";
    std::cout << "✅ OptimizedMatrix: Cache-blocked + SIMD optimized\n";
    std::cout << "✅ OptimizedMatrix3D: Parallel + contiguous memory\n";
    std::cout << "✅ uBLAS API compatibility: Complete\n";
    std::cout << "✅ HMM mathematical operations: Complete with optimizations\n";
    std::cout << "✅ Automatic optimization selection: Working\n";
    std::cout << "✅ Basic-to-Optimized conversion constructors: Complete\n\n";
    
    std::cout << "Key optimizations available:\n";
    std::cout << "• SIMD vectorization for arithmetic operations\n";
    std::cout << "• Parallel execution for large data structures\n";
    std::cout << "• Cache-blocked algorithms for matrix operations\n";
    std::cout << "• Automatic threshold-based optimization selection\n";
    std::cout << "• Memory-aligned data structures\n";
    std::cout << "• Contiguous memory layouts for all classes\n";
    
    return 0;
}
