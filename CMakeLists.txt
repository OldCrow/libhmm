cmake_minimum_required(VERSION 3.15)

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
    VERSION 2.7.0
    DESCRIPTION "Modern C++17 Hidden Markov Model Library"
    LANGUAGES CXX
)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
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
    # On macOS, prefer Homebrew installations
    set(CMAKE_PREFIX_PATH "/usr/local" "/opt/homebrew" ${CMAKE_PREFIX_PATH})
elseif(LINUX)
    # On Linux, check common package manager locations
    set(CMAKE_PREFIX_PATH "/usr" "/usr/local" ${CMAKE_PREFIX_PATH})
endif()

# find_package(Boost REQUIRED)  # REMOVED: No longer needed after Phase 8.3 Boost replacement

# Include directories - use modern target-based approach
include_directories(include)
# include_directories(${Boost_INCLUDE_DIRS})  # REMOVED: No longer needed

# Platform-specific compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wpointer-arith")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    
    # Platform-specific flags
    if(APPLE)
        # macOS specific flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
        # macOS handles undefined symbols dynamically
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")
    elseif(LINUX)
        # Linux specific flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
        # Linux requires all symbols to be defined at link time
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
    elseif(UNIX)
        # Generic Unix flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()
endif()

# Source files
set(LIBHMM_SOURCES
    src/hmm.cpp
    src/common/common.cpp
    src/common/string_tokenizer.cpp
    src/common/numerical_stability.cpp
    src/common/trainer_traits.cpp
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
    src/calculators/forward_backward_traits.cpp
    src/calculators/viterbi_traits.cpp
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
