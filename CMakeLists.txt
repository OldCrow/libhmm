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
    VERSION 3.4.0
    DESCRIPTION "Modern C++20 Hidden Markov Model Library"
    LANGUAGES CXX
)

# C++20 standard — required for std::span, concepts, and constexpr improvements
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
# Both static and shared libraries are always built (sources compiled once via
# an OBJECT library). BUILD_SHARED_LIBS is retained for CMake convention
# compatibility but has no effect — hmm (shared) and hmm_static (static) are
# always produced regardless of this flag.
option(BUILD_SHARED_LIBS "Build shared library in addition to static" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_TOOLS      "Build performance inspection and validation tools" ON)
option(BUILD_BENCHMARKS "Build comparison benchmarks (requires external HMM libraries)" OFF)
option(ENABLE_STATIC_ANALYSIS "Enable static analysis with clang-tidy" ON)
option(ENABLE_CLANG_TIDY "Enable clang-tidy analysis during build" OFF)
option(ENABLE_CPPCHECK "Enable cppcheck analysis" ON)
option(LIBHMM_ALLOW_UNSUPPORTED_CATALINA_HOMEBREW_LIBCXX
    "Allow Homebrew libc++ paths on macOS Catalina (unsafe; may crash at runtime)"
    OFF
)

# Catalina guard/probe status (reported in configure summary)
set(LIBHMM_CATALINA_GUARD_ACTIVE OFF)
set(LIBHMM_CATALINA_CONTAMINATION_CHECKS_RAN OFF)
set(LIBHMM_CATALINA_CONTAMINATION_CHECKS_CLEAN "N/A")
set(LIBHMM_CATALINA_CXX20_FEATURE_PROBES_RAN OFF)
set(LIBHMM_CATALINA_CXX20_FEATURE_PROBES_PASSED "N/A")

include(CheckCXXSourceCompiles)

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
    # Guard against unsupported runtime combinations on macOS Catalina.
    # Homebrew libc++ on 10.15 has shown widespread runtime segfaults in tests.
    set(LIBHMM_IS_CATALINA OFF)
    set(LIBHMM_IS_PRE_VENTURA OFF)
    set(LIBHMM_HOST_MACOS_VERSION "")
    set(LIBHMM_DARWIN_VERSION "${CMAKE_SYSTEM_VERSION}")
    execute_process(
        COMMAND sw_vers -productVersion
        OUTPUT_VARIABLE LIBHMM_HOST_MACOS_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE LIBHMM_SW_VERS_STATUS
    )
    if(LIBHMM_HOST_MACOS_VERSION MATCHES "^([0-9]+)\\.([0-9]+)")
        set(LIBHMM_HOST_MACOS_VERSION_MAJOR "${CMAKE_MATCH_1}")
        if(LIBHMM_HOST_MACOS_VERSION_MAJOR LESS 13)
            set(LIBHMM_IS_PRE_VENTURA ON)
        endif()
    endif()

    if(CMAKE_SYSTEM_VERSION MATCHES "^([0-9]+)")
        set(LIBHMM_DARWIN_VERSION "${CMAKE_MATCH_1}")
        if(LIBHMM_DARWIN_VERSION LESS 22)
            set(LIBHMM_IS_PRE_VENTURA ON)
        endif()
    endif()

    if(LIBHMM_SW_VERS_STATUS EQUAL 0 AND LIBHMM_HOST_MACOS_VERSION MATCHES "^10\\.15(\\..*)?$")
        set(LIBHMM_IS_CATALINA ON)
    elseif(CMAKE_SYSTEM_VERSION MATCHES "^19(\\..*)?$")
        set(LIBHMM_IS_CATALINA ON)
    elseif(CMAKE_OSX_DEPLOYMENT_TARGET MATCHES "^10\\.15(\\..*)?$")
        set(LIBHMM_IS_CATALINA ON)
    elseif(CMAKE_OSX_SYSROOT MATCHES "MacOSX10\\.15\\.sdk")
        set(LIBHMM_IS_CATALINA ON)
    endif()
    if(LIBHMM_IS_CATALINA)
        set(LIBHMM_CATALINA_GUARD_ACTIVE ON)
        set(LIBHMM_CATALINA_CONTAMINATION_CHECKS_RAN ON)
    endif()

    set(LIBHMM_HOMEBREW_LIBCXX_HINT_REGEX
        "(/usr/local|/opt/homebrew)[^; ]*(lib/c\\+\\+|include/c\\+\\+/v1|opt/llvm|Cellar/llvm|Cellar/llvm@)"
    )

    macro(libhmm_collect_homebrew_libcxx_hint hint_name hint_value)
        if(NOT "${hint_value}" STREQUAL "")
            if("${hint_value}" MATCHES "${LIBHMM_HOMEBREW_LIBCXX_HINT_REGEX}")
                list(APPEND LIBHMM_HOMEBREW_LIBCXX_HINT_SOURCES "${hint_name}")
            endif()
        endif()
    endmacro()

    set(LIBHMM_HOMEBREW_LIBCXX_HINT_SOURCES "")
    libhmm_collect_homebrew_libcxx_hint("CMAKE_CXX_COMPILER" "${CMAKE_CXX_COMPILER}")
    libhmm_collect_homebrew_libcxx_hint("CMAKE_C_COMPILER" "${CMAKE_C_COMPILER}")
    if(CMAKE_CXX_COMPILER AND EXISTS "${CMAKE_CXX_COMPILER}")
        file(REAL_PATH "${CMAKE_CXX_COMPILER}" LIBHMM_CXX_COMPILER_REALPATH)
        libhmm_collect_homebrew_libcxx_hint("CMAKE_CXX_COMPILER(realpath)" "${LIBHMM_CXX_COMPILER_REALPATH}")
    endif()
    if(CMAKE_C_COMPILER AND EXISTS "${CMAKE_C_COMPILER}")
        file(REAL_PATH "${CMAKE_C_COMPILER}" LIBHMM_C_COMPILER_REALPATH)
        libhmm_collect_homebrew_libcxx_hint("CMAKE_C_COMPILER(realpath)" "${LIBHMM_C_COMPILER_REALPATH}")
    endif()
    libhmm_collect_homebrew_libcxx_hint("env:CC" "$ENV{CC}")
    libhmm_collect_homebrew_libcxx_hint("env:CXX" "$ENV{CXX}")
    libhmm_collect_homebrew_libcxx_hint("env:LDFLAGS" "$ENV{LDFLAGS}")
    libhmm_collect_homebrew_libcxx_hint("env:CPPFLAGS" "$ENV{CPPFLAGS}")
    libhmm_collect_homebrew_libcxx_hint("env:CXXFLAGS" "$ENV{CXXFLAGS}")
    libhmm_collect_homebrew_libcxx_hint("env:SDKROOT" "$ENV{SDKROOT}")
    libhmm_collect_homebrew_libcxx_hint("env:MACOSX_DEPLOYMENT_TARGET" "$ENV{MACOSX_DEPLOYMENT_TARGET}")
    libhmm_collect_homebrew_libcxx_hint("env:CPATH" "$ENV{CPATH}")
    libhmm_collect_homebrew_libcxx_hint("env:CPLUS_INCLUDE_PATH" "$ENV{CPLUS_INCLUDE_PATH}")
    libhmm_collect_homebrew_libcxx_hint("env:LIBRARY_PATH" "$ENV{LIBRARY_PATH}")
    libhmm_collect_homebrew_libcxx_hint("env:DYLD_LIBRARY_PATH" "$ENV{DYLD_LIBRARY_PATH}")
    libhmm_collect_homebrew_libcxx_hint("env:PKG_CONFIG_PATH" "$ENV{PKG_CONFIG_PATH}")
    libhmm_collect_homebrew_libcxx_hint("CMAKE_EXE_LINKER_FLAGS" "${CMAKE_EXE_LINKER_FLAGS}")
    libhmm_collect_homebrew_libcxx_hint("CMAKE_SHARED_LINKER_FLAGS" "${CMAKE_SHARED_LINKER_FLAGS}")
    libhmm_collect_homebrew_libcxx_hint("CMAKE_MODULE_LINKER_FLAGS" "${CMAKE_MODULE_LINKER_FLAGS}")
    libhmm_collect_homebrew_libcxx_hint("CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")
    libhmm_collect_homebrew_libcxx_hint("CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES" "${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}")
    libhmm_collect_homebrew_libcxx_hint("CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES" "${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES}")
    list(REMOVE_DUPLICATES LIBHMM_HOMEBREW_LIBCXX_HINT_SOURCES)

    if(LIBHMM_IS_CATALINA AND LIBHMM_HOMEBREW_LIBCXX_HINT_SOURCES AND NOT LIBHMM_ALLOW_UNSUPPORTED_CATALINA_HOMEBREW_LIBCXX)
        string(REPLACE ";" ", " LIBHMM_HOMEBREW_LIBCXX_HINTS_TEXT "${LIBHMM_HOMEBREW_LIBCXX_HINT_SOURCES}")
        message(FATAL_ERROR
            "Detected Homebrew LLVM/libc++ hints on macOS Catalina (10.15). "
            "This combination is known to cause runtime segfaults in libhmm tests.\n"
            "Detected in: ${LIBHMM_HOMEBREW_LIBCXX_HINTS_TEXT}\n"
            "Configure with the Catalina-safe helper script:\n"
            "  ./scripts/configure_catalina.sh build\n"
            "Or configure manually with a sanitized environment:\n"
            "  env -u CC -u CXX -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS -u SDKROOT -u MACOSX_DEPLOYMENT_TARGET -u CPATH -u CPLUS_INCLUDE_PATH -u LIBRARY_PATH -u DYLD_LIBRARY_PATH -u PKG_CONFIG_PATH cmake -S . -B build\n"
            "To bypass this guard temporarily, pass:\n"
            "  -DLIBHMM_ALLOW_UNSUPPORTED_CATALINA_HOMEBREW_LIBCXX=ON"
        )
    endif()
    if(LIBHMM_IS_CATALINA)
        if(LIBHMM_HOMEBREW_LIBCXX_HINT_SOURCES)
            if(LIBHMM_ALLOW_UNSUPPORTED_CATALINA_HOMEBREW_LIBCXX)
                set(LIBHMM_CATALINA_CONTAMINATION_CHECKS_CLEAN "OVERRIDDEN")
            else()
                set(LIBHMM_CATALINA_CONTAMINATION_CHECKS_CLEAN "NO")
            endif()
        else()
            set(LIBHMM_CATALINA_CONTAMINATION_CHECKS_CLEAN "YES")
        endif()
    endif()

    if(LIBHMM_IS_CATALINA AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        set(LIBHMM_CATALINA_CXX20_FEATURE_PROBES_RAN ON)
        set(_libhmm_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
        set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -std=c++20")

        check_cxx_source_compiles(
            "#include <array>
#include <span>
int main() {
    std::array<double, 3> values{1.0, 2.0, 3.0};
    std::span<const double> view(values);
    return static_cast<int>(view.size());
}"
            LIBHMM_CATALINA_HAS_STD_SPAN
        )

        set(LIBHMM_CORE_REQUIRES_CONCEPTS OFF)
        file(GLOB_RECURSE LIBHMM_CONCEPT_SCAN_FILES
            "${CMAKE_SOURCE_DIR}/include/libhmm/*.h"
            "${CMAKE_SOURCE_DIR}/include/libhmm/*.hpp"
            "${CMAKE_SOURCE_DIR}/src/*.cpp"
        )
        foreach(_scan_file IN LISTS LIBHMM_CONCEPT_SCAN_FILES)
            file(STRINGS "${_scan_file}" _concept_decl_lines
                REGEX "^[ \t]*concept[ \t]+[A-Za-z_]"
                LIMIT_COUNT 1
            )
            if(_concept_decl_lines)
                set(LIBHMM_CORE_REQUIRES_CONCEPTS ON)
                break()
            endif()
        endforeach()

        if(LIBHMM_CORE_REQUIRES_CONCEPTS)
            check_cxx_source_compiles(
                "#include <concepts>
template <typename T>
concept Arithmetic = std::integral<T> || std::floating_point<T>;
template <Arithmetic T>
constexpr T add(T a, T b) {
    return a + b;
}
int main() {
    return add(1, 2) == 3 ? 0 : 1;
}"
                LIBHMM_CATALINA_HAS_CONCEPTS
            )
        else()
            set(LIBHMM_CATALINA_HAS_CONCEPTS ON)
            message(STATUS "Catalina concepts probe skipped: core sources do not declare concepts")
        endif()

        set(CMAKE_REQUIRED_FLAGS "${_libhmm_saved_required_flags}")

        if(NOT LIBHMM_CATALINA_HAS_STD_SPAN OR NOT LIBHMM_CATALINA_HAS_CONCEPTS)
            set(LIBHMM_CATALINA_CXX20_FEATURE_PROBES_PASSED "NO")
            message(FATAL_ERROR
                "AppleClang on macOS Catalina failed required C++20 feature probes (std::span and/or concepts).\n"
                "Install or update Xcode Command Line Tools, then reconfigure with:\n"
                "  ./scripts/configure_catalina.sh build"
            )
        endif()

        set(LIBHMM_CATALINA_CXX20_FEATURE_PROBES_PASSED "YES")
    endif()

    set(LIBHMM_LEGACY_WARNING_SUPPRESSIONS "")
    if(LIBHMM_IS_PRE_VENTURA AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(LIBHMM_LEGACY_WARNING_SUPPRESSIONS
            -Wno-unknown-warning-option
            -Wno-deprecated-volatile
        )
        if(LIBHMM_HOST_MACOS_VERSION)
            message(STATUS "Pre-Ventura macOS (${LIBHMM_HOST_MACOS_VERSION}) detected - enabling legacy warning suppressions for non-core targets")
        else()
            message(STATUS "Pre-Ventura macOS detected - enabling legacy warning suppressions for non-core targets")
        endif()
    endif()
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

    # On Catalina, keep core toolchain/library resolution system-first and rely
    # on explicit paths for optional external dependencies.
    if(LIBHMM_IS_CATALINA)
        message(STATUS "Catalina guard active - skipping global Homebrew CMAKE_PREFIX_PATH injection")
    else()
        # Set architecture-specific CMAKE_PREFIX_PATH
        set(CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}" ${CMAKE_PREFIX_PATH})
    endif()

elseif(LINUX)
    # On Linux, check common package manager locations
    set(CMAKE_PREFIX_PATH "/usr" "/usr/local" ${CMAKE_PREFIX_PATH})
endif()

# find_package(Boost REQUIRED)  # REMOVED: No longer needed after Phase 8.3 Boost replacement

# Include directories - use modern target-based approach
include_directories(include)
# include_directories(${Boost_INCLUDE_DIRS})  # REMOVED: No longer needed

# Platform-specific compiler flags
# NOTE: MSVC flags are applied per-target (see after add_library below) so that
# FetchContent dependencies like GTest are not compiled with /W4.
#
# /utf-8 is the exception: it specifies source and execution character sets as
# UTF-8 globally. This suppresses C4566 warnings that occur when source files
# contain non-ASCII Unicode characters (Greek letters, math symbols, etc.) on
# Windows systems with a non-UTF-8 ANSI code page (e.g. CP1252). It is safe
# to apply to all targets including GTest because it only affects how MSVC
# interprets source encoding — it has no effect on ASCII-only code.
if(MSVC)
    add_compile_options(/utf-8)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wpointer-arith")
    # Note: -Wno-deprecated-declarations and -Wno-defaulted-function-deleted removed
    # to surface all warnings. Re-add with justification if specific cases cannot be fixed.
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
# SIMD DETECTION — compile-time dispatch, build machine CPU matters
#
# libhmm uses compile-time SIMD dispatch: each machine builds a binary optimised
# for its own CPU. LIBHMM_BEST_SIMD_FLAGS is applied per-TU to LIBHMM_SIMD_SOURCES
# only (not globally) so non-SIMD code paths stay at the baseline ISA.
#
# GCC/Clang: -march=native asks the compiler to use everything the build
# machine's CPU supports. This is both compiler and CPU aware.
#
# MSVC: no -march=native equivalent. check_cxx_compiler_flag only tests whether
# the compiler ACCEPTS a flag (always true for /arch:AVX512 on modern MSVC),
# NOT whether the build machine's CPU can execute the resulting instructions.
# check_cxx_source_runs compiles and runs a tiny test, verifying both.
# =============================================================================
include(CheckCXXCompilerFlag)

# x86 ISA extension flags are only meaningful on x86/x86_64 targets. Running
# them on AArch64 (Apple Silicon, ARM servers) produces misleading results
# because Clang silently accepts x86 flags without enabling those ISAs.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    # AArch64: NEON is the mandatory ISA baseline — always present, no flag needed.
    set(COMPILER_SUPPORTS_NEON ON)
    message(STATUS "SIMD compiler support — NEON: ON (AArch64 baseline)")
elseif(NOT MSVC)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
    check_cxx_compiler_flag("-mavx"    COMPILER_SUPPORTS_AVX)
    check_cxx_compiler_flag("-mavx2"   COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)
    message(STATUS "SIMD compiler support — AVX-512: ${COMPILER_SUPPORTS_AVX512F}  AVX2: ${COMPILER_SUPPORTS_AVX2}  AVX: ${COMPILER_SUPPORTS_AVX}")
else()
    # MSVC x86/x64: informational only — actual flag selected via check_cxx_source_runs below.
    check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512F)
    check_cxx_compiler_flag("/arch:AVX2"   COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("/arch:AVX"    COMPILER_SUPPORTS_AVX)
    set(COMPILER_SUPPORTS_SSE42 ON)  # SSE4.2 is guaranteed on x64 MSVC
    message(STATUS "SIMD compiler support — AVX-512: ${COMPILER_SUPPORTS_AVX512F}  AVX2: ${COMPILER_SUPPORTS_AVX2}  AVX: ${COMPILER_SUPPORTS_AVX}")
endif()

# =============================================================================
# Select best SIMD flag for this compiler + CPU combination.
#
# GCC/Clang: -march=native is both compiler-aware and CPU-aware.
# MSVC: use check_cxx_source_runs to compile AND execute a minimal test,
# mirroring what -march=native does automatically on GCC/Clang.
# Falls back to SSE2 baseline in cross-compilation (can't run tests).
# =============================================================================
if(MSVC AND NOT CMAKE_CROSSCOMPILING)
    include(CheckCXXSourceRuns)
    set(_saved_req_flags "${CMAKE_REQUIRED_FLAGS}")

    set(CMAKE_REQUIRED_FLAGS "/arch:AVX512")
    check_cxx_source_runs(
        "#include <intrin.h>\nint main(){__m512d x=_mm512_setzero_pd();(void)x;return 0;}"
        CPU_SUPPORTS_AVX512F)

    set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
    check_cxx_source_runs(
        "#include <intrin.h>\nint main(){__m256d x=_mm256_setzero_pd();(void)x;return 0;}"
        CPU_SUPPORTS_AVX2)

    set(CMAKE_REQUIRED_FLAGS "/arch:AVX")
    check_cxx_source_runs(
        "#include <intrin.h>\nint main(){__m256d x=_mm256_setzero_pd();(void)x;return 0;}"
        CPU_SUPPORTS_AVX)

    set(CMAKE_REQUIRED_FLAGS "${_saved_req_flags}")

    if(CPU_SUPPORTS_AVX512F)
        set(LIBHMM_BEST_SIMD_FLAGS "/arch:AVX512")
        message(STATUS "SIMD optimization: /arch:AVX512 (CPU verified)")
    elseif(CPU_SUPPORTS_AVX2)
        set(LIBHMM_BEST_SIMD_FLAGS "/arch:AVX2")
        message(STATUS "SIMD optimization: /arch:AVX2 (CPU verified)")
    elseif(CPU_SUPPORTS_AVX)
        set(LIBHMM_BEST_SIMD_FLAGS "/arch:AVX")
        message(STATUS "SIMD optimization: /arch:AVX (CPU verified)")
    else()
        set(LIBHMM_BEST_SIMD_FLAGS "")
        message(STATUS "SIMD optimization: (SSE2 baseline)")
    endif()
elseif(MSVC)
    # Cross-compilation: cannot run tests; fall back to SSE2 baseline.
    set(LIBHMM_BEST_SIMD_FLAGS "")
    message(STATUS "SIMD optimization: (cross-compilation — SSE2 baseline)")
else()
    # GCC/Clang on all platforms: -march=native selects the best ISA for the
    # build machine (NEON on AArch64; AVX-512/AVX2/AVX/SSE2 on x86_64).
    set(LIBHMM_BEST_SIMD_FLAGS "-march=native")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        message(STATUS "SIMD optimization: -march=native (AArch64/NEON)")
    else()
        message(STATUS "SIMD optimization: -march=native")
    endif()
endif()

# All distribution sources with getBatchLogProbabilities() overrides.
# Gaussian has explicit SIMD intrinsics (tier 2); all others have concrete
# non-virtual loops that the compiler can auto-vectorize (tier 1).
set(LIBHMM_SIMD_SOURCES
    src/distributions/beta_distribution.cpp
    src/distributions/binomial_distribution.cpp
    src/distributions/chi_squared_distribution.cpp
    src/distributions/discrete_distribution.cpp
    src/distributions/exponential_distribution.cpp
    src/distributions/gamma_distribution.cpp
    src/distributions/gaussian_distribution.cpp
    src/distributions/log_normal_distribution.cpp
    src/distributions/negative_binomial_distribution.cpp
    src/distributions/pareto_distribution.cpp
    src/distributions/poisson_distribution.cpp
    src/distributions/rayleigh_distribution.cpp
    src/distributions/student_t_distribution.cpp
    src/distributions/uniform_distribution.cpp
    src/distributions/weibull_distribution.cpp
)

# Additional TUs that include simd_kernels_internal.h or transcendental_kernels.h
# and therefore need LIBHMM_BEST_SIMD_FLAGS to activate the #if LIBHMM_HAS_* cascade.
# (log_normal and pareto are already in LIBHMM_SIMD_SOURCES above.)
list(APPEND LIBHMM_SIMD_SOURCES
    src/performance/transcendental_kernels.cpp
    src/calculators/forward_backward_calculator.cpp
    src/training/baum_welch_trainer.cpp
)

if(LIBHMM_BEST_SIMD_FLAGS)
    foreach(simd_src ${LIBHMM_SIMD_SOURCES})
        set_source_files_properties(
            ${simd_src}
            PROPERTIES COMPILE_FLAGS "${LIBHMM_BEST_SIMD_FLAGS}"
        )
    endforeach()
    message(STATUS "Applied SIMD flags to ${CMAKE_CURRENT_SOURCE_DIR}")
endif()
# Note: MSVC with /arch:AVX512 emits 2 × C4244 from <memory> internal templates
# when AVX-512 register types are instantiated. /external:anglebrackets /external:W0
# (already on the hmm target) does not suppress them — a known MSVC limitation in
# template specialization warning paths. Accepted as external-code noise.

# Source files
set(LIBHMM_SOURCES
    src/hmm.cpp
    src/common/common.cpp
    src/common/string_tokenizer.cpp
    src/common/numerical_stability.cpp
    src/common/weighted_stats.cpp
    src/performance/transcendental_kernels.cpp
    src/distributions/distribution_base.cpp
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
    # Canonical calculators (Phase 3)
    src/calculators/forward_backward_calculator.cpp
    src/calculators/viterbi_calculator.cpp
    # Canonical trainers (Phase 4)
    src/training/baum_welch_trainer.cpp
    src/training/viterbi_trainer.cpp
    src/training/segmental_kmeans_trainer.cpp
    src/io/file_io_manager.cpp
    src/io/json_utils.cpp
    src/io/hmm_json.cpp
    src/io/xml_file_reader.cpp
    src/io/xml_file_writer.cpp
)

# ── Library targets ───────────────────────────────────────────────────────────
# Sources are compiled once into an OBJECT library, then linked into both a
# SHARED and a STATIC target. This avoids compiling the source tree twice and
# ensures the two library variants are always bit-identical.

add_library(hmm_objects OBJECT ${LIBHMM_SOURCES})

# MSVC compiler flags — applied to the OBJECT library so they affect all
# compiled TUs regardless of which library type is ultimately consumed.
# /W4: strong warnings. /permissive-: strict standards conformance.
# /external:anglebrackets /external:W0 suppresses warnings from system headers
# (e.g. C4244 from <memory> template instantiations — MSVC limitation).
if(MSVC)
    target_compile_options(hmm_objects PRIVATE
        /W4 /permissive- /external:anglebrackets /external:W0)
endif()

# PIC is required for objects that go into the shared library; harmless for static.
set_property(TARGET hmm_objects PROPERTY POSITION_INDEPENDENT_CODE ON)

# Shared library
add_library(hmm SHARED $<TARGET_OBJECTS:hmm_objects>)

# MSVC does not generate an import .lib for a DLL unless symbols are explicitly
# exported or WINDOWS_EXPORT_ALL_SYMBOLS is set. The latter is the pragmatic
# choice here: it instructs CMake to auto-generate a .def file at link time,
# which causes MSVC to emit both hmm.dll and hmm.lib (the import library).
if(MSVC)
    set_target_properties(hmm PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

# Static library (compiled from the same objects — no recompilation)
add_library(hmm_static STATIC $<TARGET_OBJECTS:hmm_objects>)
# On Windows the shared library already produces an import library called hmm.lib
# alongside hmm.dll. Naming the static archive hmm.lib too would cause a collision
# in the output directory. Use hmm_static.lib on Windows (a common convention:
# e.g. zlib.lib/zlibstatic.lib). On all other platforms the .a/.lib suffixes
# distinguish the two types unambiguously, so -lhmm suffices.
if(WIN32)
    set_target_properties(hmm_static PROPERTIES OUTPUT_NAME "hmm_static")
else()
    set_target_properties(hmm_static PROPERTIES OUTPUT_NAME "hmm")
endif()
set_target_properties(hmm_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

# Enable clang-tidy if requested — applied to the OBJECT library so analysis
# runs once on all compiled TUs regardless of which library type is built.
if(ENABLE_CLANG_TIDY)
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
        set_target_properties(hmm_objects PROPERTIES
            CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-quiet")
    else()
        message(WARNING "clang-tidy requested but not found - skipping static analysis")
    endif()
endif()

# ── Shared library properties ─────────────────────────────────────────────────
set_target_properties(hmm PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER include/libhmm/libhmm.h
    OUTPUT_NAME "hmm"
    CLEAN_DIRECT_OUTPUT 1
)

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

# ── Link dependencies ─────────────────────────────────────────────────────────
# On Linux the math library is a separate DSO; link it into both targets.
# On macOS and Windows math is part of the system/CRT — no explicit link needed.
if(LINUX)
    target_link_libraries(hmm        PUBLIC m)
    target_link_libraries(hmm_static INTERFACE m)  # propagated to static consumers
elseif(UNIX AND NOT APPLE)
    target_link_libraries(hmm        PUBLIC m)
    target_link_libraries(hmm_static INTERFACE m)
endif()

# ── Install targets ────────────────────────────────────────────────────────────
# Install both library types. RUNTIME covers the .dll on Windows (placed in bin/
# by convention so it is found at runtime without PATH manipulation).
install(TARGETS hmm hmm_static
    LIBRARY DESTINATION lib          # .so / .dylib
    ARCHIVE DESTINATION lib          # .a / .lib (static archive + Windows import lib)
    RUNTIME DESTINATION bin          # .dll (Windows)
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
# Benchmarks (comparative and internal performance suites)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Performance inspection and validation tools
if(BUILD_TOOLS)
    add_subdirectory(tools)
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
message(STATUS "Library targets: hmm (shared) + hmm_static (static)")
if(APPLE)
    message(STATUS "  Shared:  libhmm.${PROJECT_VERSION}.dylib")
    message(STATUS "  Static:  libhmm.a")
elseif(WIN32)
    message(STATUS "  Shared:  hmm.dll + hmm.lib (import library for DLL consumers)")
    message(STATUS "  Static:  hmm_static.lib (static archive for fully-static builds)")
elseif(LINUX)
    message(STATUS "  Shared:  libhmm.so.${PROJECT_VERSION}")
    message(STATUS "  Static:  libhmm.a")
else()
    message(STATUS "  Shared:  platform-default shared library")
    message(STATUS "  Static:  platform-default static library")
endif()
message(STATUS "Examples:     ${BUILD_EXAMPLES}")
message(STATUS "Tests:        ${BUILD_TESTS}")
message(STATUS "Tools:        ${BUILD_TOOLS}")
message(STATUS "Benchmarks:   ${BUILD_BENCHMARKS}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
if(APPLE)
    message(STATUS "Catalina guard active: ${LIBHMM_CATALINA_GUARD_ACTIVE}")
    message(STATUS "Catalina contamination checks clean: ${LIBHMM_CATALINA_CONTAMINATION_CHECKS_CLEAN}")
    if(LIBHMM_CATALINA_CXX20_FEATURE_PROBES_RAN)
        message(STATUS "Catalina C++20 feature probes passed: ${LIBHMM_CATALINA_CXX20_FEATURE_PROBES_PASSED}")
    endif()
endif()
message(STATUS "=")
