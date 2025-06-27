# Test configuration for libhmm
cmake_minimum_required(VERSION 3.14)

# Find required packages
find_package(GTest QUIET)
find_package(Threads REQUIRED)

# Detect if using LLVM compiler and set appropriate linking flags
# NOTE: LLVM-specific linking temporarily commented out due to runtime issues
#       on macOS 10.15 (2012 MacBook Pro). Using system libc++ for stability.
# if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER MATCHES "llvm")
#     message(STATUS "Detected LLVM Clang compiler - configuring libc++ linking")
#     set(LIBCXX_LINK_FLAGS "-stdlib=libc++")
#     # Add explicit path to LLVM's libc++ if needed
#     get_filename_component(LLVM_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
#     get_filename_component(LLVM_ROOT_DIR ${LLVM_BIN_DIR} DIRECTORY)
#     set(LLVM_LIBCXX_PATH "${LLVM_ROOT_DIR}/lib/c++")
#     if(EXISTS "${LLVM_LIBCXX_PATH}")
#         set(LIBCXX_LINK_FLAGS "${LIBCXX_LINK_FLAGS} -L${LLVM_LIBCXX_PATH} -Wl,-rpath,${LLVM_LIBCXX_PATH}")
#         message(STATUS "Using LLVM libc++ from: ${LLVM_LIBCXX_PATH}")
#     endif()
# else()
#     set(LIBCXX_LINK_FLAGS "")
# endif()

# Use system libc++ for stability
set(LIBCXX_LINK_FLAGS "")

# Check if Google Test is available
if(GTest_FOUND OR TARGET gtest)
    message(STATUS "Google Test found - building comprehensive test suite")
    
    # Include directories
    include_directories(${CMAKE_SOURCE_DIR}/include)
    if(GTest_FOUND)
        include_directories(${GTEST_INCLUDE_DIRS})
    endif()
    
    # Common test libraries and dependencies
    if(TARGET gtest)
        set(TEST_LIBRARIES gtest gtest_main Threads::Threads hmm)
    else()
        set(TEST_LIBRARIES ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} Threads::Threads hmm)
    endif()
    
    # Apply LLVM-specific linking flags if needed
    if(LIBCXX_LINK_FLAGS)
        list(APPEND TEST_LIBRARIES "${LIBCXX_LINK_FLAGS}")
    endif()
    
    # Main integration tests (in tests/ root directory)
    set(MAIN_TESTS
        test_hmm_core.cpp
    )
    
    # Common module tests (in common/ subdirectory)
    set(COMMON_TESTS
        common/test_common.cpp
        common/test_numerical_stability.cpp
        common/test_optimized_matrix3d.cpp
        common/test_modern_constants.cpp
    )
    
    # Calculator tests (in calculators/ subdirectory)
    set(CALCULATOR_TESTS
        calculators/test_forwardbackward_calculators.cpp
        calculators/test_forwardbackward_performance.cpp
        calculators/test_forwardbackward_selection.cpp
        calculators/test_forwardbackward_traits.cpp
        calculators/test_simd_forward_backward_calculators.cpp
        calculators/test_simd_viterbi_calculators.cpp
        calculators/test_viterbi_calculators.cpp
        calculators/test_viterbi_performance.cpp
        calculators/test_viterbi_selection.cpp
        calculators/test_viterbi_traits.cpp
    )
    
    # Distribution tests (in distributions/ subdirectory)
    set(DISTRIBUTION_TESTS
        distributions/test_distribution_traits.cpp
        distributions/test_distributions_header.cpp
        distributions/test_distributions.cpp
        distributions/test_beta_distribution.cpp
        distributions/test_binomial_distribution.cpp
        distributions/test_chi_squared_distribution.cpp
        distributions/test_discrete_distribution.cpp
        distributions/test_exponential_distribution.cpp
        distributions/test_gamma_distribution.cpp
        distributions/test_gaussian_distribution.cpp
        distributions/test_log_normal_distribution.cpp
        distributions/test_negative_binomial_distribution.cpp
        distributions/test_pareto_distribution.cpp
        distributions/test_poisson_distribution.cpp
        distributions/test_student_t_distribution.cpp
        distributions/test_uniform_distribution.cpp
        distributions/test_weibull_distribution.cpp
        distributions/test_distribution_type_safety.cpp
    )
    
    # IO tests (in io/ subdirectory)
    set(IO_TESTS
        io/test_xml_file_io.cpp
        io/test_hmm_stream_io.cpp
    )
    
    # Performance tests (in performance/ subdirectory)
    set(PERFORMANCE_TESTS
        performance/test_performance.cpp
    )

    # Training tests (in training/ subdirectory)
    set(TRAINING_TESTS
        training/test_training.cpp
        training/test_training_edge_cases.cpp
        training/test_robust_viterbi_trainer.cpp
        training/test_trainer_traits.cpp
    )

    # Combine all tests
    set(COMPREHENSIVE_TESTS ${MAIN_TESTS} ${COMMON_TESTS} ${CALCULATOR_TESTS} ${DISTRIBUTION_TESTS} ${IO_TESTS} ${PERFORMANCE_TESTS} ${TRAINING_TESTS})
    
    # Create individual test executables
    foreach(TEST_SOURCE ${COMPREHENSIVE_TESTS})
        # Get the test name from the source file name
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
        
        # Create executable
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        
        # Link libraries
        target_link_libraries(${TEST_NAME} ${TEST_LIBRARIES})
        
        # Apply LLVM-specific linking flags if needed
        if(LIBCXX_LINK_FLAGS)
            set_target_properties(${TEST_NAME} PROPERTIES
                LINK_FLAGS "${LIBCXX_LINK_FLAGS}"
            )
        endif()
        
        # Set C++17 standard
        set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD 17)
        set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
        
        # Add to test suite
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
        
        # Set test properties
        set_tests_properties(${TEST_NAME} PROPERTIES
            TIMEOUT 300
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endforeach()
    
    # Note: Skipping combined test executable due to multiple main functions
    # Each test file has its own main() function, so they can't be combined
    
    # Test discovery for modern CMake versions
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.10 AND TARGET gtest_discover_tests)
        include(GoogleTest)
        foreach(TEST_SOURCE ${COMPREHENSIVE_TESTS})
            get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
            gtest_discover_tests(${TEST_NAME})
        endforeach()
    endif()
    
    # Create list of test target names (without subdirectory paths)
    set(ALL_TEST_TARGETS)
    foreach(TEST_SOURCE ${COMPREHENSIVE_TESTS})
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
        list(APPEND ALL_TEST_TARGETS ${TEST_NAME})
    endforeach()
else()
    message(STATUS "Google Test not found - building basic test suite only")
    set(ALL_TEST_TARGETS "")
endif()


# Original/Legacy test executables (always built)
# Note: test_type_safety is now included in the comprehensive test suite above

# Custom target to run all tests
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose --parallel 4
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running all libhmm tests"
)

# Install test executables
set(LEGACY_TARGETS)
# Note: All tests are now in the comprehensive test suite

install(TARGETS ${ALL_TEST_TARGETS}
    RUNTIME DESTINATION bin/tests
    COMPONENT tests
)

# Print summary
message(STATUS "Test configuration summary:")
if(GTest_FOUND OR TARGET gtest)
    message(STATUS "  Comprehensive tests: ${COMPREHENSIVE_TESTS}")
else()
    message(STATUS "  Comprehensive tests: DISABLED (Google Test not found)")
endif()
message(STATUS "  Legacy tests: ${LEGACY_TARGETS}")
message(STATUS "  All test targets: ${ALL_TEST_TARGETS}")
