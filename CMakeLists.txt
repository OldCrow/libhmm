cmake_minimum_required(VERSION 3.20)

# Set policy for FindBoost to suppress deprecation warning
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# Configure C/C++ compilers with LLVM preference on macOS
# NOTE: LLVM detection temporarily commented out due to runtime segfault issues
#       on macOS 10.15 (2012 MacBook Pro). Compilation works but runtime fails.
# TODO: Investigate LLVM 19/20 runtime compatibility issues for future use
# if(APPLE AND NOT CMAKE_CXX_COMPILER)
#     # Try to find LLVM compilers in order of preference: 20 -> 19 -> Apple Clang
#     find_program(LLVM20_CXX_COMPILER
#         NAMES clang++
#         PATHS /usr/local/Cellar/llvm/20.1.7/bin
#         NO_DEFAULT_PATH
#     )
#     
#     find_program(LLVM20_C_COMPILER
#         NAMES clang
#         PATHS /usr/local/Cellar/llvm/20.1.7/bin
#         NO_DEFAULT_PATH
#     )
#     
#     find_program(LLVM19_CXX_COMPILER
#         NAMES clang++
#         PATHS /usr/local/Cellar/llvm@19/19.1.7/bin
#         NO_DEFAULT_PATH
#     )
#     
#     find_program(LLVM19_C_COMPILER
#         NAMES clang
#         PATHS /usr/local/Cellar/llvm@19/19.1.7/bin
#         NO_DEFAULT_PATH
#     )
#     
#     if(LLVM20_CXX_COMPILER AND LLVM20_C_COMPILER)
#         set(CMAKE_CXX_COMPILER ${LLVM20_CXX_COMPILER})
#         set(CMAKE_C_COMPILER ${LLVM20_C_COMPILER})
#         message(STATUS "Using LLVM 20 compilers:")
#         message(STATUS "  C++: ${CMAKE_CXX_COMPILER}")
#         message(STATUS "  C:   ${CMAKE_C_COMPILER}")
#     elseif(LLVM19_CXX_COMPILER AND LLVM19_C_COMPILER)
#         set(CMAKE_CXX_COMPILER ${LLVM19_CXX_COMPILER})
#         set(CMAKE_C_COMPILER ${LLVM19_C_COMPILER})
#         message(STATUS "Using LLVM 19 compilers:")
#         message(STATUS "  C++: ${CMAKE_CXX_COMPILER}")
#         message(STATUS "  C:   ${CMAKE_C_COMPILER}")
#     else()
#         message(STATUS "LLVM not found, falling back to system compiler")
#     endif()
# endif()

# Default to system compiler (Apple Clang on macOS)
if(APPLE AND NOT CMAKE_CXX_COMPILER)
    message(STATUS "Using system compiler (Apple Clang) for stability")
endif()

project(libhmm
    VERSION 2.9.1
    DESCRIPTION "Modern C++20 Hidden Markov Model Library"
    LANGUAGES CXX
)

# C++20 standard — required for std::span, concepts, and constexpr improvements
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build test programs" ON)
option(ENABLE_STATIC_ANALYSIS "Enable static analysis with clang-tidy" ON)
option(ENABLE_CLANG_TIDY "Enable clang-tidy analysis during build" OFF)
option(ENABLE_CPPCHECK "Enable cppcheck analysis" ON)

# Platform detection and setup
if(APPLE)
    set(LIBHMM_PLATFORM "macOS")
    message(STATUS "Building for macOS")
elseif(WIN32)
    set(LIBHMM_PLATFORM "Windows")
    message(STATUS "Building for Windows")
elseif(LINUX)
    set(LIBHMM_PLATFORM "Linux")
    message(STATUS "Building for Linux")
elseif(UNIX)
    set(LIBHMM_PLATFORM "Unix-like")
    message(STATUS "Building for Unix-like system (not macOS or Linux)")
else()
    set(LIBHMM_PLATFORM "Unknown")
    message(WARNING "Building for unknown platform")
endif()

# Find required packages with platform-specific handling
if(APPLE)
    # Architecture-aware Homebrew path detection for macOS
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE APPLE_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(APPLE_ARCH STREQUAL "arm64")
        # Apple Silicon M1/M2/M3 - Homebrew is in /opt/homebrew
        set(HOMEBREW_PREFIX "/opt/homebrew")
        message(STATUS "Detected Apple Silicon (${APPLE_ARCH}) - using Homebrew at ${HOMEBREW_PREFIX}")
    else()
        # Intel Mac - Homebrew is in /usr/local
        set(HOMEBREW_PREFIX "/usr/local")
        message(STATUS "Detected Intel Mac (${APPLE_ARCH}) - using Homebrew at ${HOMEBREW_PREFIX}")
    endif()
    
    # Set architecture-specific CMAKE_PREFIX_PATH
    set(CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}" ${CMAKE_PREFIX_PATH})
    
elseif(LINUX)
    # On Linux, check common package manager locations
    set(CMAKE_PREFIX_PATH "/usr" "/usr/local" ${CMAKE_PREFIX_PATH})
endif()

# find_package(Boost REQUIRED)  # REMOVED: No longer needed after Phase 8.3 Boost replacement

# Include directories - use modern target-based approach
include_directories(include)
# include_directories(${Boost_INCLUDE_DIRS})  # REMOVED: No longer needed

# Platform-specific compiler flags
if(MSVC)
    # /W4: strong warnings. /wd4267 silence size_t narrowing, /wd4996 silence deprecated-constructor
    # warnings that are pre-existing and will be addressed during the refactor.
    # /permissive- enables stricter C++ standards conformance.
    add_compile_options(/W4 /wd4267 /wd4996 /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wpointer-arith -Wno-deprecated-declarations -Wno-defaulted-function-deleted")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

    # Position-independent code for shared libraries on Unix platforms
    if(APPLE OR LINUX OR UNIX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()

    if(APPLE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")
    elseif(LINUX)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
    endif()
endif()

# =============================================================================
# SIMD DETECTION — compiler capability, NOT build machine CPU
#
# Each SIMD ISA is checked against the compiler using check_cxx_compiler_flag.
# If the compiler can compile the flag, the corresponding source TU is included
# with that flag set per-TU via set_source_files_properties.
#
# This ensures a binary built on any x86-64 machine includes all ISA variants
# the compiler supports and dispatches at runtime — the build machine's CPU
# is irrelevant to which variants are compiled in.
#
# The dispatch TU (src/performance/simd_support.cpp) is compiled with NO SIMD
# flags so that the runtime CPUID code cannot itself emit SIMD instructions.
#
# NOTE: Per-TU SIMD source files will be added in Phase 3 when the batch
# distribution interface (getBatchLogProbabilities) is implemented.
# This scaffolding is here to document the pattern and make Phase 3 simpler.
# =============================================================================
include(CheckCXXCompilerFlag)

if(NOT MSVC)
    check_cxx_compiler_flag("-mavx"   COMPILER_SUPPORTS_AVX)
    check_cxx_compiler_flag("-mavx2"  COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)
else()
    # MSVC uses /arch: flags; check for AVX support
    check_cxx_compiler_flag("/arch:AVX"  COMPILER_SUPPORTS_AVX)
    check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
    set(COMPILER_SUPPORTS_SSE42 ON)  # SSE4.2 is guaranteed on x64 MSVC
endif()

message(STATUS "SIMD compiler support — AVX: ${COMPILER_SUPPORTS_AVX}  AVX2: ${COMPILER_SUPPORTS_AVX2}")
# Phase 3: use COMPILER_SUPPORTS_AVX/AVX2 to conditionally add per-TU SIMD sources

# Source files
set(LIBHMM_SOURCES
    src/hmm.cpp
    src/common/common.cpp
    src/common/string_tokenizer.cpp
    src/common/numerical_stability.cpp
    src/common/optimized_vector_simd.cpp
    src/common/optimized_matrix_simd.cpp
    src/training/trainer_traits.cpp
    src/distributions/probability_distribution.cpp
    src/distributions/discrete_distribution.cpp
    src/distributions/gaussian_distribution.cpp
    src/distributions/gamma_distribution.cpp
    src/distributions/exponential_distribution.cpp
    src/distributions/log_normal_distribution.cpp
    src/distributions/pareto_distribution.cpp
    src/distributions/poisson_distribution.cpp
    src/distributions/beta_distribution.cpp
    src/distributions/weibull_distribution.cpp
    src/distributions/rayleigh_distribution.cpp
    src/distributions/uniform_distribution.cpp
    src/distributions/binomial_distribution.cpp
    src/distributions/negative_binomial_distribution.cpp
    src/distributions/student_t_distribution.cpp
    src/distributions/chi_squared_distribution.cpp
    src/calculators/forward_backward_calculator.cpp
    src/calculators/log_simd_forward_backward_calculator.cpp
    src/calculators/scaled_simd_forward_backward_calculator.cpp
    src/calculators/viterbi_calculator.cpp
    src/calculators/scaled_simd_viterbi_calculator.cpp
    src/calculators/log_simd_viterbi_calculator.cpp
    src/training/viterbi_trainer.cpp
    src/training/robust_viterbi_trainer.cpp
    src/training/segmented_kmeans_trainer.cpp
    src/training/baum_welch_trainer.cpp
    src/training/scaled_baum_welch_trainer.cpp
    src/io/file_io_manager.cpp
    src/io/xml_file_reader.cpp
    src/io/xml_file_writer.cpp
    src/performance/simd_support.cpp
    src/performance/thread_pool.cpp
    src/performance/log_space_ops.cpp
    src/calculators/forward_backward_traits.cpp
    src/calculators/viterbi_traits.cpp
    src/calculators/calculator_traits_utils.cpp
)

# Create the library
add_library(hmm ${LIBHMM_SOURCES})

# Enable clang-tidy if requested
if(ENABLE_CLANG_TIDY)
    # Try to find clang-tidy in common Homebrew locations
    find_program(CLANG_TIDY_EXE 
        NAMES clang-tidy
        PATHS 
            /usr/local/Cellar/llvm/20.1.7/bin
            /usr/local/Cellar/llvm@19/19.1.7/bin
            /opt/homebrew/bin
            /usr/local/bin
        NO_DEFAULT_PATH
    )
    
    if(CLANG_TIDY_EXE)
        message(STATUS "Found clang-tidy: ${CLANG_TIDY_EXE}")
        set_target_properties(
            hmm PROPERTIES
            CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-quiet"
        )
    else()
        message(WARNING "clang-tidy requested but not found - skipping static analysis")
    endif()
endif()

# Set library properties with cross-platform considerations
set_target_properties(hmm PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER include/libhmm/libhmm.h
    OUTPUT_NAME "hmm"
    CLEAN_DIRECT_OUTPUT 1
)

# Platform-specific library properties
if(APPLE)
    set_target_properties(hmm PROPERTIES
        MACOSX_RPATH TRUE
        INSTALL_NAME_DIR "@rpath"
    )
elseif(LINUX)
    set_target_properties(hmm PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# Link libraries with platform-specific handling
if(LINUX)
    # On Linux, explicitly link math library
    target_link_libraries(hmm m)  # Boost libraries removed
elseif(APPLE)
    # On macOS, math functions are part of the system library
    target_link_libraries(hmm)  # Boost libraries removed
elseif(WIN32)
    target_link_libraries(hmm)  # Math functions are part of the MSVC CRT; no separate m.lib exists
else()
    # Fallback for other Unix systems
    target_link_libraries(hmm m)  # Boost libraries removed
endif()

# Install targets
install(TARGETS hmm
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/libhmm
)

install(DIRECTORY include/libhmm DESTINATION include)

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Development Benchmarking and Debug Tests
# These executables are placed in build/tests/ directory for better organization
add_executable(analyze_overhead tests/performance/analyze_overhead.cpp)
set_target_properties(analyze_overhead PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
target_link_libraries(analyze_overhead hmm)

add_executable(benchmark_parallel tests/performance/benchmark_parallel.cpp)
set_target_properties(benchmark_parallel PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
target_link_libraries(benchmark_parallel hmm)

add_executable(debug_parallel tests/performance/debug_parallel.cpp)
set_target_properties(debug_parallel PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
target_link_libraries(debug_parallel hmm)

add_executable(parallel_calculator_assessment tests/performance/parallel_calculator_assessment.cpp)
set_target_properties(parallel_calculator_assessment PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
target_link_libraries(parallel_calculator_assessment hmm)

add_executable(parallel_constants_tuning tests/performance/parallel_constants_tuning.cpp)
set_target_properties(parallel_constants_tuning PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
target_link_libraries(parallel_constants_tuning hmm)
# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libhmmConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libhmmConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libhmmConfig.cmake"
    INSTALL_DESTINATION lib/cmake/libhmm
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/libhmmConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/libhmmConfigVersion.cmake"
    DESTINATION lib/cmake/libhmm
)

# Platform and configuration summary
message(STATUS "")
message(STATUS "=== libhmm Configuration Summary ===")
message(STATUS "Platform: ${LIBHMM_PLATFORM}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Shared Libraries: ${BUILD_SHARED_LIBS}")
if(APPLE)
    message(STATUS "Library Type: .dylib (macOS dynamic library)")
elseif(WIN32)
    message(STATUS "Library Type: .lib/.dll (Windows)")
elseif(LINUX)
    message(STATUS "Library Type: .so (Linux shared object)")
elseif(UNIX)
    message(STATUS "Library Type: Unix shared library")
else()
    message(STATUS "Library Type: Platform default")
endif()
message(STATUS "Examples: ${BUILD_EXAMPLES}")
message(STATUS "Tests: ${BUILD_TESTS}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "=")
