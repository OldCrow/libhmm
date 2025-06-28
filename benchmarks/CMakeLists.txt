cmake_minimum_required(VERSION 3.15)
project(HMMBenchmarks)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
# NOTE: Boost is ONLY required for building third-party HMM libraries (HMMlib, etc.)
# that we benchmark libhmm against. libhmm itself has ZERO Boost dependencies.
find_package(Boost REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/HMMlib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/StochHMM/source/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/GHMM/install/include)
include_directories(${Boost_INCLUDE_DIRS})

# Platform-specific compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
    if(APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    elseif(LINUX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()
endif()

# Find libhmm library (assuming it's built in the parent directory)
find_library(LIBHMM_LIBRARY 
    NAMES hmm
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../build
    NO_DEFAULT_PATH
)

if(NOT LIBHMM_LIBRARY)
    message(FATAL_ERROR "libhmm library not found. Please build libhmm first.")
endif()

# Find StochHMM library
find_library(STOCHHMM_LIBRARY 
    NAMES stochhmm
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/StochHMM/install/lib
    NO_DEFAULT_PATH
)

if(NOT STOCHHMM_LIBRARY)
    message(WARNING "StochHMM library not found. StochHMM benchmarks will be disabled.")
endif()

# Find GHMM library
find_library(GHMM_LIBRARY 
    NAMES ghmm
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/GHMM/install/lib
    NO_DEFAULT_PATH
)

if(NOT GHMM_LIBRARY)
    message(WARNING "GHMM library not found. GHMM benchmarks will be disabled.")
endif()

# Create benchmark executables
add_executable(algorithm_performance_benchmark src/algorithm_performance_benchmark.cpp)
add_executable(simple_benchmark src/simple_benchmark.cpp)
add_executable(classic_problems_benchmark src/classic_problems_benchmark.cpp)
add_executable(libhmm_discrete_benchmark src/libhmm_discrete_benchmark.cpp)
add_executable(libhmm_vs_stochhmm_benchmark src/libhmm_vs_stochhmm_benchmark.cpp)
add_executable(libhmm_vs_hmmlib_benchmark src/libhmm_vs_hmmlib_benchmark.cpp)
add_executable(numerical_accuracy_test src/numerical_accuracy_test.cpp)
add_executable(deep_numerical_analysis src/deep_numerical_analysis.cpp)
add_executable(libhmm_vs_ghmm_benchmark src/libhmm_vs_ghmm_benchmark.cpp)
add_executable(libhmm_vs_htk_benchmark src/libhmm_vs_htk_benchmark.cpp)
add_executable(threeway_benchmark src/threeway_benchmark.cpp)
add_executable(libhmm_vs_ghmm_continuous_benchmark src/libhmm_vs_ghmm_continuous_benchmark.cpp)
add_executable(libhmm_vs_stochhmm_continuous_benchmark src/libhmm_vs_stochhmm_continuous_benchmark.cpp)
add_executable(gaussian_distribution_comparison src/gaussian_distribution_comparison.cpp)
add_executable(stochhmm_api_test src/stochhmm_api_test.cpp)
add_executable(libhmm_vs_htk_continuous_benchmark src/libhmm_vs_htk_continuous_benchmark.cpp)

# Link libraries (Boost needed for HMMlib benchmarks)
if(APPLE)
    target_link_libraries(algorithm_performance_benchmark ${LIBHMM_LIBRARY} ${Boost_LIBRARIES})
    target_link_libraries(simple_benchmark ${LIBHMM_LIBRARY})
    target_link_libraries(classic_problems_benchmark ${LIBHMM_LIBRARY} ${Boost_LIBRARIES})
    target_link_libraries(libhmm_discrete_benchmark ${LIBHMM_LIBRARY})
    target_link_libraries(libhmm_vs_hmmlib_benchmark ${LIBHMM_LIBRARY} ${Boost_LIBRARIES})
    target_link_libraries(numerical_accuracy_test ${LIBHMM_LIBRARY} ${Boost_LIBRARIES})
    target_link_libraries(deep_numerical_analysis ${LIBHMM_LIBRARY} ${Boost_LIBRARIES})
    if(GHMM_LIBRARY)
        target_link_libraries(libhmm_vs_ghmm_benchmark ${LIBHMM_LIBRARY} ${GHMM_LIBRARY})
    else()
        target_link_libraries(libhmm_vs_ghmm_benchmark ${LIBHMM_LIBRARY})
    endif()
    target_link_libraries(libhmm_vs_htk_benchmark ${LIBHMM_LIBRARY})
    target_link_libraries(libhmm_vs_htk_continuous_benchmark ${LIBHMM_LIBRARY})
    if(STOCHHMM_LIBRARY)
        target_link_libraries(libhmm_vs_stochhmm_benchmark ${LIBHMM_LIBRARY} ${STOCHHMM_LIBRARY})
        target_link_libraries(libhmm_vs_stochhmm_continuous_benchmark ${LIBHMM_LIBRARY} ${STOCHHMM_LIBRARY})
        target_link_libraries(threeway_benchmark ${LIBHMM_LIBRARY} ${STOCHHMM_LIBRARY} ${Boost_LIBRARIES})
    endif()
    if(GHMM_LIBRARY)
        target_link_libraries(libhmm_vs_ghmm_continuous_benchmark ${LIBHMM_LIBRARY} ${GHMM_LIBRARY})
    else()
        target_link_libraries(libhmm_vs_ghmm_continuous_benchmark ${LIBHMM_LIBRARY})
    endif()
    # Link Gaussian distribution comparison with all available libraries
    if(GHMM_LIBRARY AND STOCHHMM_LIBRARY)
        target_link_libraries(gaussian_distribution_comparison ${LIBHMM_LIBRARY} ${GHMM_LIBRARY} ${STOCHHMM_LIBRARY})
    elseif(GHMM_LIBRARY)
        target_link_libraries(gaussian_distribution_comparison ${LIBHMM_LIBRARY} ${GHMM_LIBRARY})
    elseif(STOCHHMM_LIBRARY)
        target_link_libraries(gaussian_distribution_comparison ${LIBHMM_LIBRARY} ${STOCHHMM_LIBRARY})
    else()
        target_link_libraries(gaussian_distribution_comparison ${LIBHMM_LIBRARY})
    endif()
    # Link StochHMM API test
    if(STOCHHMM_LIBRARY)
        target_link_libraries(stochhmm_api_test ${STOCHHMM_LIBRARY})
    endif()
else()
    target_link_libraries(algorithm_performance_benchmark ${LIBHMM_LIBRARY} ${Boost_LIBRARIES} m)
    target_link_libraries(simple_benchmark ${LIBHMM_LIBRARY} m)
    target_link_libraries(classic_problems_benchmark ${LIBHMM_LIBRARY} ${Boost_LIBRARIES} m)
    target_link_libraries(libhmm_discrete_benchmark ${LIBHMM_LIBRARY} m)
    target_link_libraries(libhmm_vs_hmmlib_benchmark ${LIBHMM_LIBRARY} ${Boost_LIBRARIES} m)
    target_link_libraries(numerical_accuracy_test ${LIBHMM_LIBRARY} ${Boost_LIBRARIES} m)
    target_link_libraries(deep_numerical_analysis ${LIBHMM_LIBRARY} ${Boost_LIBRARIES} m)
    if(GHMM_LIBRARY)
        target_link_libraries(libhmm_vs_ghmm_benchmark ${LIBHMM_LIBRARY} ${GHMM_LIBRARY} m)
    else()
        target_link_libraries(libhmm_vs_ghmm_benchmark ${LIBHMM_LIBRARY} m)
    endif()
    target_link_libraries(libhmm_vs_htk_benchmark ${LIBHMM_LIBRARY} m)
    target_link_libraries(libhmm_vs_htk_continuous_benchmark ${LIBHMM_LIBRARY} m)
    if(STOCHHMM_LIBRARY)
        target_link_libraries(libhmm_vs_stochhmm_benchmark ${LIBHMM_LIBRARY} ${STOCHHMM_LIBRARY} m)
        target_link_libraries(threeway_benchmark ${LIBHMM_LIBRARY} ${STOCHHMM_LIBRARY} ${Boost_LIBRARIES} m)
    endif()
endif()

# Set target properties
set_target_properties(algorithm_performance_benchmark PROPERTIES
    OUTPUT_NAME "algorithm_performance_benchmark"
)
set_target_properties(classic_problems_benchmark PROPERTIES
    OUTPUT_NAME "classic_problems_benchmark"
)
set_target_properties(simple_benchmark PROPERTIES
    OUTPUT_NAME "simple_benchmark"
)

# Create test targets
add_custom_target(run_algorithm_benchmarks
    COMMAND algorithm_performance_benchmark
    DEPENDS algorithm_performance_benchmark
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running algorithm performance benchmarks"
)

add_custom_target(run_classic_benchmarks
    COMMAND classic_problems_benchmark
    DEPENDS classic_problems_benchmark
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running classic problems benchmarks"
)

add_custom_target(run_all_benchmarks
    DEPENDS run_algorithm_benchmarks run_classic_benchmarks
    COMMENT "Running all HMM library benchmarks"
)

# Add informational message
message(STATUS "Benchmark configuration:")
message(STATUS "  libhmm library: ${LIBHMM_LIBRARY}")
message(STATUS "  Boost include: ${Boost_INCLUDE_DIRS}")
message(STATUS "  Boost libraries: ${Boost_LIBRARIES}")
message(STATUS "  Note: libhmm itself has zero external dependencies")
message(STATUS "  Note: Boost only required for HMMlib comparison benchmarks")
