
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(HMMBenchmarks LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3)
endif()

set(LIBHMM_LINK_TARGET hmm)
if(TARGET hmm)
    message(STATUS "benchmarks: linking against in-tree target 'hmm'")
else()
    find_library(LIBHMM_LIBRARY
        NAMES hmm
        PATHS
            ${CMAKE_CURRENT_SOURCE_DIR}/../build
            ${CMAKE_CURRENT_SOURCE_DIR}/../build/lib
            ${CMAKE_CURRENT_SOURCE_DIR}/../build/Release
        NO_DEFAULT_PATH
    )
    if(NOT LIBHMM_LIBRARY)
        message(FATAL_ERROR "libhmm library not found. Build libhmm first or configure from repo root with -DBUILD_BENCHMARKS=ON.")
    endif()
    set(LIBHMM_LINK_TARGET ${LIBHMM_LIBRARY})
    message(STATUS "benchmarks: linking against prebuilt libhmm: ${LIBHMM_LIBRARY}")
endif()

set(LIBHMM_BENCHMARK_DEPS_ROOT "$ENV{HOME}/Development" CACHE PATH "Root directory containing external benchmark dependencies")
set(HMMLIB_DIR "${LIBHMM_BENCHMARK_DEPS_ROOT}/HMMLib" CACHE PATH "Path to HMMLib root directory")
set(GHMM_DIR "${LIBHMM_BENCHMARK_DEPS_ROOT}/GHMM" CACHE PATH "Path to GHMM root directory")
set(STOCHHMM_DIR "${LIBHMM_BENCHMARK_DEPS_ROOT}/StochHMM" CACHE PATH "Path to StochHMM root directory")
set(HTK_DIR "${LIBHMM_BENCHMARK_DEPS_ROOT}/HTK" CACHE PATH "Path to HTK root directory")
set(JAHMM_DIR "${LIBHMM_BENCHMARK_DEPS_ROOT}/Jahmm" CACHE PATH "Path to Jahmm root directory")
set(LAMP_DIR "${LIBHMM_BENCHMARK_DEPS_ROOT}/LAMP" CACHE PATH "Path to LAMP root directory")
set(KALDI_DIR "${LIBHMM_BENCHMARK_DEPS_ROOT}/Kaldi" CACHE PATH "Path to Kaldi root directory")
set(JAHMM_ANT_DIR "${JAHMM_DIR}/apache-ant-1.10.15" CACHE PATH "Path to Apache Ant used for Jahmm builds")

set(HMMLIB_READY OFF)
if(EXISTS "${HMMLIB_DIR}")
    find_package(Boost QUIET)
    if(Boost_FOUND)
        set(HMMLIB_READY ON)
    else()
        message(WARNING "HMMLib directory found at ${HMMLIB_DIR}, but Boost was not found. HMMLib-dependent benchmarks will be skipped.")
    endif()
else()
    message(WARNING "HMMLib directory not found at ${HMMLIB_DIR}. HMMLib-dependent benchmarks will be skipped.")
endif()

set(STOCHHMM_INCLUDE_DIR "${STOCHHMM_DIR}/source/src")
if(NOT EXISTS "${STOCHHMM_INCLUDE_DIR}")
    set(STOCHHMM_INCLUDE_DIR "")
endif()
find_library(STOCHHMM_LIBRARY
    NAMES stochhmm
    HINTS
        ${STOCHHMM_DIR}/install/lib
        ${STOCHHMM_DIR}/lib
)
set(STOCHHMM_READY OFF)
if(STOCHHMM_INCLUDE_DIR AND STOCHHMM_LIBRARY)
    set(STOCHHMM_READY ON)
else()
    message(WARNING "StochHMM include/library not fully available from ${STOCHHMM_DIR}. StochHMM-dependent benchmarks will be skipped.")
endif()

set(GHMM_INCLUDE_DIR "${GHMM_DIR}/install/include")
if(NOT EXISTS "${GHMM_INCLUDE_DIR}")
    set(GHMM_INCLUDE_DIR "")
endif()
find_library(GHMM_LIBRARY
    NAMES ghmm
    HINTS
        ${GHMM_DIR}/install/lib
        ${GHMM_DIR}/lib
)
set(GHMM_READY OFF)
if(GHMM_INCLUDE_DIR AND GHMM_LIBRARY)
    set(GHMM_READY ON)
else()
    message(WARNING "GHMM include/library not fully available from ${GHMM_DIR}. GHMM-dependent benchmarks will be skipped.")
endif()

# On macOS, GHMM builds often retain an absolute install_name from the build location
# where the library was compiled. When GHMM is moved to a new directory, dependent
# benchmark executables can fail at runtime with dyld "Library not loaded".
# Normalize GHMM's install_name during configure so newly linked benchmarks resolve
# to the external canonical GHMM location under ${GHMM_DIR}.
if(APPLE AND GHMM_READY)
    find_program(INSTALL_NAME_TOOL_EXECUTABLE install_name_tool)
    if(INSTALL_NAME_TOOL_EXECUTABLE)
        set(GHMM_DYLIB_ID_TARGET "${GHMM_LIBRARY}")
        if(EXISTS "${GHMM_DIR}/install/lib/libghmm.1.dylib")
            set(GHMM_DYLIB_ID_TARGET "${GHMM_DIR}/install/lib/libghmm.1.dylib")
        endif()
        execute_process(
            COMMAND ${INSTALL_NAME_TOOL_EXECUTABLE} -id "${GHMM_DYLIB_ID_TARGET}" "${GHMM_DYLIB_ID_TARGET}"
            RESULT_VARIABLE GHMM_INSTALL_NAME_RESULT
            ERROR_VARIABLE GHMM_INSTALL_NAME_ERROR
            OUTPUT_QUIET
        )
        if(NOT GHMM_INSTALL_NAME_RESULT EQUAL 0)
            message(WARNING "Could not normalize GHMM install_name for ${GHMM_DYLIB_ID_TARGET}: ${GHMM_INSTALL_NAME_ERROR}")
        else()
            message(STATUS "Normalized GHMM install_name: ${GHMM_DYLIB_ID_TARGET}")
        endif()
    else()
        message(WARNING "install_name_tool not found; GHMM runtime path normalization skipped.")
    endif()
endif()

function(add_libhmm_benchmark target_name source_file)
    add_executable(${target_name} ${source_file})
    target_include_directories(${target_name}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
    if(LIBHMM_LEGACY_WARNING_SUPPRESSIONS)
        target_compile_options(${target_name} PRIVATE ${LIBHMM_LEGACY_WARNING_SUPPRESSIONS})
    endif()
    target_link_libraries(${target_name} PRIVATE ${LIBHMM_LINK_TARGET})
    if(NOT APPLE)
        target_link_libraries(${target_name} PRIVATE m)
    endif()
endfunction()

function(enable_hmmlib target_name)
    target_include_directories(${target_name}
        SYSTEM PRIVATE
            ${HMMLIB_DIR}
            ${Boost_INCLUDE_DIRS}
    )
    if(Boost_LIBRARIES)
        target_link_libraries(${target_name} PRIVATE ${Boost_LIBRARIES})
    endif()
endfunction()

function(enable_stochhmm target_name)
    target_include_directories(${target_name} SYSTEM PRIVATE ${STOCHHMM_INCLUDE_DIR})
    target_link_libraries(${target_name} PRIVATE ${STOCHHMM_LIBRARY})
endfunction()

function(enable_ghmm target_name)
    target_include_directories(${target_name} SYSTEM PRIVATE ${GHMM_INCLUDE_DIR})
    target_link_libraries(${target_name} PRIVATE ${GHMM_LIBRARY})
    if(APPLE)
        set_property(TARGET ${target_name} APPEND PROPERTY BUILD_RPATH "${GHMM_DIR}/install/lib")
    endif()
endfunction()

# Core benchmarks (libhmm-only)
add_libhmm_benchmark(simple_benchmark src/simple_benchmark.cpp)
add_libhmm_benchmark(libhmm_discrete_benchmark src/libhmm_discrete_benchmark.cpp)
add_libhmm_benchmark(diagnostic_accuracy_test src/diagnostic_accuracy_test.cpp)
add_libhmm_benchmark(libhmm_vs_htk_benchmark src/libhmm_vs_htk_benchmark.cpp)
add_libhmm_benchmark(libhmm_vs_htk_continuous_benchmark src/libhmm_vs_htk_continuous_benchmark.cpp)
add_libhmm_benchmark(libhmm_vs_jahmm_benchmark src/libhmm_vs_jahmm_benchmark.cpp)
add_libhmm_benchmark(libhmm_vs_lamp_benchmark src/libhmm_vs_lamp_benchmark.cpp)
target_compile_definitions(libhmm_vs_jahmm_benchmark PRIVATE LIBHMM_BENCH_JAHMM_DIR="${JAHMM_DIR}")
target_compile_definitions(libhmm_vs_lamp_benchmark PRIVATE LIBHMM_BENCH_LAMP_DIR="${LAMP_DIR}")

# HMMLib-dependent benchmarks
if(HMMLIB_READY)
    add_libhmm_benchmark(algorithm_performance_benchmark src/algorithm_performance_benchmark.cpp)
    enable_hmmlib(algorithm_performance_benchmark)

    add_libhmm_benchmark(classic_problems_benchmark src/classic_problems_benchmark.cpp)
    enable_hmmlib(classic_problems_benchmark)

    add_libhmm_benchmark(libhmm_vs_hmmlib_benchmark src/libhmm_vs_hmmlib_benchmark.cpp)
    enable_hmmlib(libhmm_vs_hmmlib_benchmark)

    add_libhmm_benchmark(numerical_accuracy_test src/numerical_accuracy_test.cpp)
    enable_hmmlib(numerical_accuracy_test)

    add_libhmm_benchmark(deep_numerical_analysis src/deep_numerical_analysis.cpp)
    enable_hmmlib(deep_numerical_analysis)
else()
    message(STATUS "Skipping HMMLib-dependent benchmark targets.")
endif()

# StochHMM-dependent benchmarks
if(STOCHHMM_READY)
    add_libhmm_benchmark(libhmm_vs_stochhmm_benchmark src/libhmm_vs_stochhmm_benchmark.cpp)
    enable_stochhmm(libhmm_vs_stochhmm_benchmark)

    add_libhmm_benchmark(libhmm_vs_stochhmm_continuous_benchmark src/libhmm_vs_stochhmm_continuous_benchmark.cpp)
    enable_stochhmm(libhmm_vs_stochhmm_continuous_benchmark)
else()
    message(STATUS "Skipping StochHMM-dependent benchmark targets.")
endif()

# GHMM-dependent benchmarks
if(GHMM_READY)
    add_libhmm_benchmark(libhmm_vs_ghmm_benchmark src/libhmm_vs_ghmm_benchmark.cpp)
    enable_ghmm(libhmm_vs_ghmm_benchmark)

    add_libhmm_benchmark(libhmm_vs_ghmm_continuous_benchmark src/libhmm_vs_ghmm_continuous_benchmark.cpp)
    enable_ghmm(libhmm_vs_ghmm_continuous_benchmark)
else()
    message(STATUS "Skipping GHMM-dependent benchmark targets.")
endif()

# Combined external-dependency benchmarks
if(HMMLIB_READY AND STOCHHMM_READY)
    add_libhmm_benchmark(threeway_benchmark src/threeway_benchmark.cpp)
    enable_hmmlib(threeway_benchmark)
    enable_stochhmm(threeway_benchmark)
else()
    message(STATUS "Skipping threeway_benchmark (requires HMMLib + StochHMM).")
endif()

if(GHMM_READY AND STOCHHMM_READY)
    add_libhmm_benchmark(gaussian_distribution_comparison src/gaussian_distribution_comparison.cpp)
    enable_ghmm(gaussian_distribution_comparison)
    enable_stochhmm(gaussian_distribution_comparison)
else()
    message(STATUS "Skipping gaussian_distribution_comparison (requires GHMM + StochHMM).")
endif()

if(TARGET algorithm_performance_benchmark)
    add_custom_target(run_algorithm_benchmarks
        COMMAND algorithm_performance_benchmark
        DEPENDS algorithm_performance_benchmark
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running algorithm performance benchmarks"
    )
endif()

if(TARGET classic_problems_benchmark)
    add_custom_target(run_classic_benchmarks
        COMMAND classic_problems_benchmark
        DEPENDS classic_problems_benchmark
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running classic problems benchmarks"
    )
endif()

set(_run_benchmark_targets "")
if(TARGET run_algorithm_benchmarks)
    list(APPEND _run_benchmark_targets run_algorithm_benchmarks)
endif()
if(TARGET run_classic_benchmarks)
    list(APPEND _run_benchmark_targets run_classic_benchmarks)
endif()

if(_run_benchmark_targets)
    add_custom_target(run_all_benchmarks
        DEPENDS ${_run_benchmark_targets}
        COMMENT "Running selected benchmark suites"
    )
endif()

message(STATUS "Benchmark dependency configuration:")
message(STATUS "  deps root: ${LIBHMM_BENCHMARK_DEPS_ROOT}")
message(STATUS "  HMMLib dir: ${HMMLIB_DIR} (ready: ${HMMLIB_READY})")
message(STATUS "  GHMM dir: ${GHMM_DIR} (ready: ${GHMM_READY})")
message(STATUS "  StochHMM dir: ${STOCHHMM_DIR} (ready: ${STOCHHMM_READY})")
message(STATUS "  HTK dir: ${HTK_DIR}")
message(STATUS "  Jahmm dir: ${JAHMM_DIR}")
message(STATUS "  LAMP dir: ${LAMP_DIR}")
message(STATUS "  Kaldi dir: ${KALDI_DIR}")
message(STATUS "  Jahmm Ant dir: ${JAHMM_ANT_DIR}")
