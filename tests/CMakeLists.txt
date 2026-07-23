# Test configuration for libhmm
cmake_minimum_required(VERSION 3.20)

# Find required packages with architecture-aware paths
if(APPLE)
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
    # Suppress Apple Clang's -Wcharacter-conversion warning in GTest's own headers.
    # gtest-printers.h:528 casts char8_t to char32_t, which is valid C++ but triggers
    # this diagnostic on AppleClang + C++20. Not a libhmm issue; suppressed here so
    # the build log stays clean for warnings we actually own.
    foreach(_gt_target gtest gtest_main gmock gmock_main)
        if(TARGET ${_gt_target})
        target_compile_options(${_gt_target} PRIVATE
                # -Wno-unknown-warning-option must come first: AppleClang (which lags
                # upstream Clang) does not recognise -Wno-character-conversion and
                # would itself warn about the unknown flag without this prefix.
                $<$<CXX_COMPILER_ID:AppleClang>:-Wno-unknown-warning-option;-Wno-character-conversion>)
        endif()
    endforeach()
endif()

# Check if Google Test is available
if(GTest_FOUND OR TARGET gtest)
    message(STATUS "Google Test found - building comprehensive test suite")

    include_directories(${CMAKE_SOURCE_DIR}/include)

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
    set(LIBHMM_GTEST_SYSTEM_INCLUDE_DIRS ${GTEST_INCLUDE_DIRS})
    foreach(_GTEST_TARGET GTest::gtest GTest::gtest_main gtest gtest_main)
        if(TARGET ${_GTEST_TARGET})
            get_target_property(_GTEST_TARGET_INCLUDES
                ${_GTEST_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
            if(_GTEST_TARGET_INCLUDES)
                list(APPEND LIBHMM_GTEST_SYSTEM_INCLUDE_DIRS ${_GTEST_TARGET_INCLUDES})
            endif()
        endif()
    endforeach()
    if(LIBHMM_GTEST_SYSTEM_INCLUDE_DIRS)
        list(REMOVE_DUPLICATES LIBHMM_GTEST_SYSTEM_INCLUDE_DIRS)
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
        if(LIBHMM_GTEST_SYSTEM_INCLUDE_DIRS)
            target_include_directories(${TARGET_NAME} SYSTEM
                PRIVATE ${LIBHMM_GTEST_SYSTEM_INCLUDE_DIRS})
        endif()
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 20)
        set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
        # Apply same warning level as the library so test code is held to the
        # same standard. GTest itself is excluded via SYSTEM includes.
        target_compile_options(${TARGET_NAME} PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/W4;/permissive->
            $<$<OR:$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:Clang>>:
                -Wall;-Wextra;-Wpedantic;-Wno-sign-compare>
            $<$<OR:$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:Clang>>:
                -Wno-unknown-warning-option;-Wno-character-conversion>
            $<$<CXX_COMPILER_ID:GNU>:-Wall;-Wextra;-Wpedantic>)
        # LIBHMM_WARN_FLAGS (top-level CMakeLists.txt) is a target-scope
        # variable, not directory-scope add_compile_options — apply explicitly
        # so tests pick up -Wpointer-arith and (when LIBHMM_WERROR) -Werror.
        target_compile_options(${TARGET_NAME} PRIVATE ${LIBHMM_WARN_FLAGS})
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
    # Platform Capabilities
    # =========================================================================
    add_hmm_test(test_simd_platform platform/test_simd_platform.cpp)

    # =========================================================================
    # Math & Numerics
    # =========================================================================
    add_hmm_test(test_modern_constants common/test_modern_constants.cpp)
    add_hmm_test(test_common           common/test_common.cpp)

    # =========================================================================
    # Linalg helpers
    # =========================================================================
    add_hmm_test(test_cholesky linalg/test_cholesky.cpp)

    # =========================================================================
    # Performance Primitives
    # Cross-cutting SIMD kernels consumed by both calculators and trainers.
    # Compiled with LIBHMM_BEST_SIMD_FLAGS so the active SIMD path matches
    # the production library -- parity is checked against std::exp.
    # =========================================================================
    add_hmm_test(test_transcendental_kernels performance/test_transcendental_kernels.cpp)
    if(LIBHMM_BEST_SIMD_FLAGS)
        set_source_files_properties(
            performance/test_transcendental_kernels.cpp
            PROPERTIES COMPILE_FLAGS "${LIBHMM_BEST_SIMD_FLAGS}")
    endif()

    # =========================================================================
    # Distributions
    # =========================================================================
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
    add_hmm_test(test_von_mises_distribution         distributions/test_von_mises_distribution.cpp)
    add_hmm_test(test_distribution_fitting           distributions/test_distribution_fitting.cpp)
    add_hmm_test(test_multivariate_distributions     distributions/test_multivariate_distributions.cpp)

    # =========================================================================
    # Multivariate calculators (Phase K)
    # =========================================================================
    add_hmm_test(test_mv_calculator calculators/test_mv_calculator.cpp)

    # =========================================================================
    # Core HMM
    # =========================================================================
    add_hmm_test(test_hmm_core test_hmm_core.cpp)

    # =========================================================================
    # Calculators
    # =========================================================================
    add_hmm_test(test_canonical_calculators calculators/test_canonical_calculators.cpp)
    add_hmm_test(test_calculator_continuous calculators/test_calculator_continuous.cpp)
    add_hmm_test(test_calculator_edge_cases calculators/test_calculator_edge_cases.cpp)
    add_hmm_test(test_fb_mode_parity        calculators/test_fb_mode_parity.cpp)
    add_hmm_test(test_posterior_decoding    calculators/test_posterior_decoding.cpp)

    # =========================================================================
    # Trainers
    # =========================================================================
    add_hmm_test(test_canonical_training     training/test_canonical_training.cpp)
    add_hmm_test(test_training               training/test_training.cpp)
    add_hmm_test(test_training_edge_cases    training/test_training_edge_cases.cpp)
    add_hmm_test(test_baum_welch_convergence training/test_baum_welch_convergence.cpp)
    add_hmm_test(test_bw_parity              training/test_bw_parity.cpp)
    add_hmm_test(test_map_baum_welch         training/test_map_baum_welch.cpp)

    # =========================================================================
    # Multivariate training and kmeans_init (Phase K)
    # =========================================================================
    add_hmm_test(test_mv_training          training/test_mv_training.cpp)
    add_hmm_test(test_segmental_kmeans_mv  training/test_segmental_kmeans_mv.cpp)

    # =========================================================================
    # IO & Integration
    # =========================================================================
    add_hmm_test(test_xml_file_io       io/test_xml_file_io.cpp)
    add_hmm_test(test_hmm_stream_io     io/test_hmm_stream_io.cpp)
    add_hmm_test(test_hmm_json          io/test_hmm_json.cpp)
    add_hmm_test(test_hmm_json_mv       io/test_hmm_json_mv.cpp)
    add_hmm_test(test_end_to_end        integration/test_end_to_end.cpp)
    add_hmm_test(test_model_selection   test_model_selection.cpp)

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
    VERBATIM
)

# check_timing: run serially for accurate timing measurement.
add_custom_target(check_timing
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j1
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running libhmm tests serially (for timing accuracy)"
    VERBATIM
)

# run_all_tests: runs everything in parallel (includes known_broken).
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --parallel 4
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running all libhmm tests"
    VERBATIM
)


message(STATUS "Test configuration: ${ALL_TEST_TARGETS}")
