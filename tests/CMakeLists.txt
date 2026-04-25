# Test configuration for libhmm
cmake_minimum_required(VERSION 3.20)

# Find required packages with architecture-aware paths
if(APPLE)
    # Use the same architecture detection as the main CMakeLists.txt
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE APPLE_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(APPLE_ARCH STREQUAL "arm64")
        set(HOMEBREW_PREFIX "/opt/homebrew")
    else()
        set(HOMEBREW_PREFIX "/usr/local")
    endif()
    
    # Set GTest search paths to avoid mixing Intel and ARM libraries
    set(CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}" ${CMAKE_PREFIX_PATH})
endif()

find_package(GTest QUIET)
find_package(Threads REQUIRED)

# FetchContent fallback: pull GTest from source when not found locally.
# This makes the test suite self-contained on CI and machines without vcpkg.
# On machines with GTest installed (find_package above succeeded), this is skipped.
if(NOT GTest_FOUND AND NOT TARGET gtest)
    message(STATUS "Google Test not found locally — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
        GIT_SHALLOW    TRUE
    )
    # On Windows, prevent GTest from overriding the project's CRT choice
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

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

    include_directories(${CMAKE_SOURCE_DIR}/include)
    if(GTest_FOUND)
        include_directories(${GTEST_INCLUDE_DIRS})
    endif()

    if(TARGET GTest::gtest AND TARGET GTest::gtest_main)
        set(TEST_LIBRARIES GTest::gtest_main GTest::gtest Threads::Threads hmm_static)
        message(STATUS "Using modern GTest targets")
    elseif(TARGET gtest AND TARGET gtest_main)
        set(TEST_LIBRARIES gtest_main gtest Threads::Threads hmm_static)
        message(STATUS "Using legacy GTest targets")
    else()
        set(TEST_LIBRARIES ${GTEST_MAIN_LIBRARIES} ${GTEST_LIBRARIES} Threads::Threads hmm_static)
        message(STATUS "Using GTest library variables")
    endif()

    if(LIBCXX_LINK_FLAGS)
        list(APPEND TEST_LIBRARIES "${LIBCXX_LINK_FLAGS}")
    endif()

    # =========================================================================
    # add_hmm_test(target source [LABELS label1 label2 ...])
    # Creates a test executable, links it, and registers it with CTest.
    # Appends the target name to ALL_TEST_TARGETS in the parent scope.
    #
    # Labels:
    #   known_broken  Pre-existing failure; excluded from CI via -LE known_broken
    #   benchmark     On-demand only; excluded from standard runs
    #   timing        Has timing-sensitive assertions; run serially for accuracy
    # =========================================================================
    function(add_hmm_test TARGET_NAME SOURCE_FILE)
        cmake_parse_arguments(_T "" "" "LABELS" ${ARGN})

        add_executable(${TARGET_NAME} ${SOURCE_FILE})
        target_link_libraries(${TARGET_NAME} ${TEST_LIBRARIES})
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 20)
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
        # Apply same warning level as the library so test code is held to the
        # same standard. GTest itself is excluded via SYSTEM includes.
        if(MSVC)
            target_compile_options(${TARGET_NAME} PRIVATE /W4 /permissive-)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        if(LIBCXX_LINK_FLAGS)
            set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LIBCXX_LINK_FLAGS}")
        endif()

        add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
        set_tests_properties(${TARGET_NAME} PROPERTIES
            TIMEOUT 300
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

        if(_T_LABELS)
            set_tests_properties(${TARGET_NAME} PROPERTIES LABELS "${_T_LABELS}")
        endif()

        set(ALL_TEST_TARGETS ${ALL_TEST_TARGETS} ${TARGET_NAME} PARENT_SCOPE)
    endfunction()

    set(ALL_TEST_TARGETS "")
    
    # =========================================================================
    # Level 0: Platform
    # No tests yet. test_simd_platform will be added in Phase 4.5.2 (tools)
    # and referenced here when a portable SIMD-capability test is available.
    # =========================================================================

    # =========================================================================
    # Level 1: Math & Numerics
    # =========================================================================
    add_hmm_test(test_modern_constants    common/test_modern_constants.cpp)
    add_hmm_test(test_numerical_stability common/test_numerical_stability.cpp)
    add_hmm_test(test_common              common/test_common.cpp)

    # =========================================================================
    # Level 2: Linear Algebra
    # =========================================================================
    add_hmm_test(test_optimized_linear_operations common/test_optimized_linear_operations.cpp)
    add_hmm_test(test_linear_operations_classes   common/test_linear_operations_classes.cpp)
    add_hmm_test(test_optimized_matrix3d          common/test_optimized_matrix3d.cpp)

    # =========================================================================
    # Level 3: Distributions
    # =========================================================================
    add_hmm_test(test_distribution_traits       distributions/test_distribution_traits.cpp)
    add_hmm_test(test_distributions_header      distributions/test_distributions_header.cpp)
    add_hmm_test(test_distributions             distributions/test_distributions.cpp)
    add_hmm_test(test_distribution_type_safety  distributions/test_distribution_type_safety.cpp)
    add_hmm_test(test_beta_distribution              distributions/test_beta_distribution.cpp)
    add_hmm_test(test_binomial_distribution          distributions/test_binomial_distribution.cpp)
    add_hmm_test(test_chi_squared_distribution       distributions/test_chi_squared_distribution.cpp)
    add_hmm_test(test_discrete_distribution          distributions/test_discrete_distribution.cpp)
    add_hmm_test(test_exponential_distribution       distributions/test_exponential_distribution.cpp)
    add_hmm_test(test_gamma_distribution             distributions/test_gamma_distribution.cpp)
    add_hmm_test(test_gaussian_distribution          distributions/test_gaussian_distribution.cpp)
    add_hmm_test(test_log_normal_distribution        distributions/test_log_normal_distribution.cpp)
    add_hmm_test(test_negative_binomial_distribution distributions/test_negative_binomial_distribution.cpp)
    add_hmm_test(test_pareto_distribution            distributions/test_pareto_distribution.cpp)
    add_hmm_test(test_poisson_distribution           distributions/test_poisson_distribution.cpp)
    add_hmm_test(test_rayleigh_distribution          distributions/test_rayleigh_distribution.cpp)
    add_hmm_test(test_student_t_distribution         distributions/test_student_t_distribution.cpp)
    add_hmm_test(test_uniform_distribution           distributions/test_uniform_distribution.cpp)
    add_hmm_test(test_weibull_distribution           distributions/test_weibull_distribution.cpp)

    # =========================================================================
    # Level 4: Core HMM
    # =========================================================================
    add_hmm_test(test_hmm_core test_hmm_core.cpp)

    # =========================================================================
    # Level 5: Calculators
    # =========================================================================
    add_hmm_test(test_canonical_calculators calculators/test_canonical_calculators.cpp)
    add_hmm_test(test_calculator_continuous calculators/test_calculator_continuous.cpp)
    add_hmm_test(test_calculator_edge_cases calculators/test_calculator_edge_cases.cpp)

    # =========================================================================
    # Level 6: Trainers
    # =========================================================================
    add_hmm_test(test_canonical_training     training/test_canonical_training.cpp)
    add_hmm_test(test_training               training/test_training.cpp)
    add_hmm_test(test_training_edge_cases    training/test_training_edge_cases.cpp)
    add_hmm_test(test_baum_welch_convergence training/test_baum_welch_convergence.cpp)

    # =========================================================================
    # Level 7: IO & Integration
    # =========================================================================
    add_hmm_test(test_xml_file_io   io/test_xml_file_io.cpp)
    add_hmm_test(test_hmm_stream_io io/test_hmm_stream_io.cpp)
    add_hmm_test(test_end_to_end    integration/test_end_to_end.cpp)

else()
    message(STATUS "Google Test not found - building basic test suite only")
    set(ALL_TEST_TARGETS "")
endif()

# =============================================================================
# Custom targets
# =============================================================================
# check: correctness suite, safe to run in parallel.
# Excludes known_broken (pre-existing failures) and benchmark (on-demand).
# This is the standard CI target. Named 'check' (not 'run_tests') to avoid
# clashing with cmake's built-in RUN_TESTS target on Windows VS generator.
add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -LE "known_broken|benchmark"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running libhmm correctness suite (excludes known_broken, benchmark)"
)

# check_timing: run serially for accurate timing measurement.
add_custom_target(check_timing
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j1
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running libhmm tests serially (for timing accuracy)"
)

# run_all_tests: runs everything in parallel (includes known_broken).
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --parallel 4
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running all libhmm tests"
)

# Install test executables
install(TARGETS ${ALL_TEST_TARGETS}
    RUNTIME DESTINATION bin/tests
    COMPONENT tests
)

message(STATUS "Test configuration: ${ALL_TEST_TARGETS}")
