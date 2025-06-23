# Test configuration for libhmm
cmake_minimum_required(VERSION 3.14)

# Find required packages
find_package(GTest QUIET)
find_package(Threads REQUIRED)

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
        set(TEST_LIBRARIES gtest gtest_main Threads::Threads hmm ${Boost_LIBRARIES} m)
    else()
        set(TEST_LIBRARIES ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} Threads::Threads hmm ${Boost_LIBRARIES} m)
    endif()
    
    # Integration test executables (GoogleTest framework)
    set(GTEST_SOURCES
        test_hmm_core.cpp
        test_calculators.cpp
        test_training.cpp
        test_io.cpp
        test_distributions.cpp
        test_common.cpp
    )
    
    # Create individual test executables
    foreach(TEST_SOURCE ${GTEST_SOURCES})
        # Get the test name from the source file name
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
        
        # Create executable
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        
        # Link libraries
        target_link_libraries(${TEST_NAME} ${TEST_LIBRARIES})
        
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
        foreach(TEST_SOURCE ${GTEST_SOURCES})
            get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
            gtest_discover_tests(${TEST_NAME})
        endforeach()
    endif()
    
    set(ALL_TEST_TARGETS ${GTEST_SOURCES})
    string(REPLACE ".cpp" "" ALL_TEST_TARGETS "${ALL_TEST_TARGETS}")
else()
    message(STATUS "Google Test not found - building basic test suite only")
    set(ALL_TEST_TARGETS "")
endif()

# Unit tests (standalone, fast individual tests)
set(UNIT_TEST_SOURCES
    unit/test_poisson_distribution.cpp
    unit/test_gaussian_distribution.cpp
    unit/test_gamma_distribution.cpp
    unit/test_exponential_distribution.cpp
    unit/test_log_normal_distribution.cpp
    unit/test_pareto_distribution.cpp
    unit/test_discrete_distribution.cpp
    unit/test_beta_distribution.cpp
    unit/test_weibull_distribution.cpp
    unit/test_uniform_distribution.cpp
    unit/test_binomial_distribution.cpp
    unit/test_negative_binomial_distribution.cpp
)

set(UNIT_TEST_TARGETS "")
foreach(UNIT_TEST_SOURCE ${UNIT_TEST_SOURCES})
    get_filename_component(UNIT_TEST_NAME ${UNIT_TEST_SOURCE} NAME_WE)
    
    add_executable(${UNIT_TEST_NAME} ${UNIT_TEST_SOURCE})
    target_link_libraries(${UNIT_TEST_NAME} hmm ${Boost_LIBRARIES} m)
    target_include_directories(${UNIT_TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_property(TARGET ${UNIT_TEST_NAME} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${UNIT_TEST_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
    
    # Add to test suite with "unit_" prefix
    add_test(NAME unit_${UNIT_TEST_NAME} COMMAND ${UNIT_TEST_NAME})
    
    list(APPEND UNIT_TEST_TARGETS ${UNIT_TEST_NAME})
endforeach()

# Original/Legacy test executables (always built)
add_executable(test_type_safety test_type_safety.cpp)
target_link_libraries(test_type_safety hmm ${Boost_LIBRARIES} m)
target_include_directories(test_type_safety PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_property(TARGET test_type_safety PROPERTY CXX_STANDARD 17)
set_property(TARGET test_type_safety PROPERTY CXX_STANDARD_REQUIRED ON)

# Register legacy tests with CTest
add_test(NAME TypeSafetyTest COMMAND test_type_safety)

# Custom target to run all tests
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose --parallel 4
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running all libhmm tests"
)

# Install test executables
set(LEGACY_TARGETS test_type_safety)
list(APPEND ALL_TEST_TARGETS ${LEGACY_TARGETS})

install(TARGETS ${ALL_TEST_TARGETS}
    RUNTIME DESTINATION bin/tests
    COMPONENT tests
)

# Print summary
message(STATUS "Test configuration summary:")
if(GTest_FOUND OR TARGET gtest)
    message(STATUS "  Comprehensive tests: ${GTEST_SOURCES}")
else()
    message(STATUS "  Comprehensive tests: DISABLED (Google Test not found)")
endif()
message(STATUS "  Legacy tests: ${LEGACY_TARGETS}")
message(STATUS "  All test targets: ${ALL_TEST_TARGETS}")
