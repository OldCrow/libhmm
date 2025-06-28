#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>

#include "libhmm/common/basic_matrix3d.h"
#include "libhmm/common/optimized_matrix3d.h"

using namespace libhmm;

class OptimizedMatrix3DTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test dimensions
        small_x = 10; small_y = 10; small_z = 10;
        medium_x = 50; medium_y = 50; medium_z = 20;
        large_x = 100; large_y = 100; large_z = 50;
        
        // Initialize random number generator
        generator.seed(42); // Fixed seed for reproducible tests
    }
    
    std::size_t small_x, small_y, small_z;
    std::size_t medium_x, medium_y, medium_z;
    std::size_t large_x, large_y, large_z;
    std::mt19937 generator;
};

// Test basic functionality
TEST_F(OptimizedMatrix3DTest, BasicFunctionality) {
    OptimizedMatrix3D<double> matrix(small_x, small_y, small_z);
    
    // Test dimensions
    EXPECT_EQ(matrix.getXDimensionSize(), small_x);
    EXPECT_EQ(matrix.getYDimensionSize(), small_y);
    EXPECT_EQ(matrix.getZDimensionSize(), small_z);
    EXPECT_EQ(matrix.size(), small_x * small_y * small_z);
    EXPECT_FALSE(matrix.empty());
    
    // Test element access
    matrix(5, 5, 5) = 3.14;
    EXPECT_DOUBLE_EQ(matrix(5, 5, 5), 3.14);
    
    // Test bounds checking
    EXPECT_NO_THROW(matrix.at(9, 9, 9));
    EXPECT_THROW(matrix.at(10, 9, 9), std::out_of_range);
    EXPECT_THROW(matrix.at(9, 10, 9), std::out_of_range);
    EXPECT_THROW(matrix.at(9, 9, 10), std::out_of_range);
}

// Test constructor validation
TEST_F(OptimizedMatrix3DTest, ConstructorValidation) {
    // Valid construction
    EXPECT_NO_THROW(OptimizedMatrix3D<int>(1, 1, 1));
    EXPECT_NO_THROW(OptimizedMatrix3D<double>(10, 20, 30, 5.0));
    
    // Invalid dimensions
    EXPECT_THROW(OptimizedMatrix3D<int>(0, 1, 1), std::invalid_argument);
    EXPECT_THROW(OptimizedMatrix3D<int>(1, 0, 1), std::invalid_argument);
    EXPECT_THROW(OptimizedMatrix3D<int>(1, 1, 0), std::invalid_argument);
}

// Test initialization values
TEST_F(OptimizedMatrix3DTest, InitializationValues) {
    // Default initialization (zero)
    OptimizedMatrix3D<double> matrix1(5, 5, 5);
    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            for (std::size_t k = 0; k < 5; ++k) {
                EXPECT_DOUBLE_EQ(matrix1(i, j, k), 0.0);
            }
        }
    }
    
    // Custom initialization
    OptimizedMatrix3D<double> matrix2(3, 3, 3, 2.5);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 3; ++k) {
                EXPECT_DOUBLE_EQ(matrix2(i, j, k), 2.5);
            }
        }
    }
}

// Test fill operations
TEST_F(OptimizedMatrix3DTest, FillOperations) {
    OptimizedMatrix3D<double> matrix(small_x, small_y, small_z);
    
    // Fill with specific value
    matrix.fill(7.5);
    for (std::size_t i = 0; i < small_x; ++i) {
        for (std::size_t j = 0; j < small_y; ++j) {
            for (std::size_t k = 0; k < small_z; ++k) {
                EXPECT_DOUBLE_EQ(matrix(i, j, k), 7.5);
            }
        }
    }
    
    // Clear to zero
    matrix.clear();
    for (std::size_t i = 0; i < small_x; ++i) {
        for (std::size_t j = 0; j < small_y; ++j) {
            for (std::size_t k = 0; k < small_z; ++k) {
                EXPECT_DOUBLE_EQ(matrix(i, j, k), 0.0);
            }
        }
    }
}

// Test 2D slice functionality
TEST_F(OptimizedMatrix3DTest, SliceFunctionality) {
    OptimizedMatrix3D<double> matrix(5, 4, 3);
    
    // Fill matrix with known pattern
    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            for (std::size_t k = 0; k < 3; ++k) {
                matrix(i, j, k) = i * 100 + j * 10 + k;
            }
        }
    }
    
    // Test slice access
    auto slice = matrix.slice(2);
    for (std::size_t j = 0; j < 4; ++j) {
        for (std::size_t k = 0; k < 3; ++k) {
            EXPECT_DOUBLE_EQ(slice(j, k), 200 + j * 10 + k);
            EXPECT_DOUBLE_EQ(slice[j][k], 200 + j * 10 + k);
        }
    }
}

// Test arithmetic operations
TEST_F(OptimizedMatrix3DTest, ArithmeticOperations) {
    OptimizedMatrix3D<double> matrix1(3, 3, 3, 2.0);
    OptimizedMatrix3D<double> matrix2(3, 3, 3, 3.0);
    
    // Test addition
    matrix1 += matrix2;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 3; ++k) {
                EXPECT_DOUBLE_EQ(matrix1(i, j, k), 5.0);
            }
        }
    }
    
    // Test scalar multiplication
    matrix1 *= 2.0;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 3; ++k) {
                EXPECT_DOUBLE_EQ(matrix1(i, j, k), 10.0);
            }
        }
    }
    
    // Test sum operation
    EXPECT_DOUBLE_EQ(matrix1.sum(), 27 * 10.0); // 3*3*3 * 10.0
    
    // Test dimension mismatch
    OptimizedMatrix3D<double> matrix3(2, 2, 2);
    EXPECT_THROW(matrix1 += matrix3, std::invalid_argument);
}

// Test move semantics
TEST_F(OptimizedMatrix3DTest, MoveSemantics) {
    OptimizedMatrix3D<double> matrix1(small_x, small_y, small_z, 5.0);
    
    // Move constructor
    OptimizedMatrix3D<double> matrix2 = std::move(matrix1);
    EXPECT_EQ(matrix2.getXDimensionSize(), small_x);
    EXPECT_DOUBLE_EQ(matrix2(0, 0, 0), 5.0);
    
    // Move assignment
    OptimizedMatrix3D<double> matrix3(1, 1, 1);
    matrix3 = std::move(matrix2);
    EXPECT_EQ(matrix3.getXDimensionSize(), small_x);
    EXPECT_DOUBLE_EQ(matrix3(0, 0, 0), 5.0);
}

// Test factory function
TEST_F(OptimizedMatrix3DTest, FactoryFunction) {
    auto matrix = make_matrix3d(5, 6, 7, 3.14);
    
    EXPECT_EQ(matrix.getXDimensionSize(), 5);
    EXPECT_EQ(matrix.getYDimensionSize(), 6);
    EXPECT_EQ(matrix.getZDimensionSize(), 7);
    EXPECT_DOUBLE_EQ(matrix(0, 0, 0), 3.14);
}

// Test legacy compatibility
TEST_F(OptimizedMatrix3DTest, LegacyCompatibility) {
    OptimizedMatrix3D<double> matrix(small_x, small_y, small_z);
    
    // Legacy methods should still work (though deprecated)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    EXPECT_EQ(matrix.GetXDimensionSize(), static_cast<int>(small_x));
    EXPECT_EQ(matrix.GetYDimensionSize(), static_cast<int>(small_y));
    EXPECT_EQ(matrix.GetZDimensionSize(), static_cast<int>(small_z));
    #pragma GCC diagnostic pop
}

// Performance benchmark comparing old vs new implementation
TEST_F(OptimizedMatrix3DTest, PerformanceBenchmark) {
    const std::size_t iterations = 100;  // Reduced from 1000 to 100
    const std::size_t x = 25, y = 25, z = 10; // Reduced from 50x50x20 to 25x25x10
    
    std::cout << "\n=== Performance Benchmark ===\n";
    std::cout << "Matrix size: " << x << "x" << y << "x" << z << " (" << x*y*z << " elements)\n";
    std::cout << "Iterations: " << iterations << "\n\n";
    
    // Benchmark original Matrix3D
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (std::size_t iter = 0; iter < iterations; ++iter) {
        BasicMatrix3D<double> matrix(x, y, z);
            
            // Fill with random values
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (std::size_t i = 0; i < x; ++i) {
                for (std::size_t j = 0; j < y; ++j) {
                    for (std::size_t k = 0; k < z; ++k) {
                        matrix.Set(i, j, k, dist(generator));
                    }
                }
            }
            
            // Access all elements
            double sum = 0.0;
            for (std::size_t i = 0; i < x; ++i) {
                for (std::size_t j = 0; j < y; ++j) {
                    for (std::size_t k = 0; k < z; ++k) {
                        sum += matrix(i, j, k);
                    }
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Original Matrix3D: " << duration.count() << " ms\n";
    }
    
    // Benchmark OptimizedMatrix3D
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (std::size_t iter = 0; iter < iterations; ++iter) {
            OptimizedMatrix3D<double> matrix(x, y, z);
            
            // Fill with random values
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (std::size_t i = 0; i < x; ++i) {
                for (std::size_t j = 0; j < y; ++j) {
                    for (std::size_t k = 0; k < z; ++k) {
                        matrix(i, j, k) = dist(generator);
                    }
                }
            }
            
            // Use optimized sum operation
            double sum = matrix.sum();
            (void)sum; // Suppress unused variable warning
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "OptimizedMatrix3D: " << duration.count() << " ms\n";
    }
    
    std::cout << "\n";
}

// Test correctness against original implementation
TEST_F(OptimizedMatrix3DTest, CorrectnessComparison) {
    const std::size_t x = 10, y = 8, z = 6;
    
    BasicMatrix3D<double> original(x, y, z);
    OptimizedMatrix3D<double> optimized(x, y, z);
    
    // Fill both with same pattern
    for (std::size_t i = 0; i < x; ++i) {
        for (std::size_t j = 0; j < y; ++j) {
            for (std::size_t k = 0; k < z; ++k) {
                double value = i * 1000 + j * 100 + k;
                original.Set(i, j, k, value);
                optimized(i, j, k) = value;
            }
        }
    }
    
    // Verify all elements are identical
    for (std::size_t i = 0; i < x; ++i) {
        for (std::size_t j = 0; j < y; ++j) {
            for (std::size_t k = 0; k < z; ++k) {
                EXPECT_DOUBLE_EQ(original(i, j, k), optimized(i, j, k));
            }
        }
    }
    
    // Test dimension getters
    EXPECT_EQ(original.getXDimensionSize(), optimized.getXDimensionSize());
    EXPECT_EQ(original.getYDimensionSize(), optimized.getYDimensionSize());
    EXPECT_EQ(original.getZDimensionSize(), optimized.getZDimensionSize());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
