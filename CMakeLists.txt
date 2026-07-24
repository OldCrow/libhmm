cmake_minimum_required(VERSION 3.25)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# libhmm v4.x requires macOS 13 Ventura or newer.  Do not force every local
# build to target the minimum: when CMAKE_OSX_DEPLOYMENT_TARGET is unset, let
# AppleClang/CMake use the host/compiler default.  This avoids linker warnings
# when local dependencies such as Homebrew GTest were built for the host macOS
# version (for example 15.0).  Packagers who need Ventura-compatible binaries
# should set -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 explicitly.
set(LIBHMM_MIN_MACOS_DEPLOYMENT_TARGET "13.0")
if(APPLE)
    if(DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND NOT "${CMAKE_OSX_DEPLOYMENT_TARGET}" STREQUAL "")
        if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS LIBHMM_MIN_MACOS_DEPLOYMENT_TARGET)
            message(FATAL_ERROR
                "libhmm v4.x requires CMAKE_OSX_DEPLOYMENT_TARGET >= "
                "${LIBHMM_MIN_MACOS_DEPLOYMENT_TARGET}; got ${CMAKE_OSX_DEPLOYMENT_TARGET}")
        endif()
        message(STATUS "macOS deployment target: ${CMAKE_OSX_DEPLOYMENT_TARGET} "
                       "(minimum supported: ${LIBHMM_MIN_MACOS_DEPLOYMENT_TARGET})")
    else()
        message(STATUS "macOS deployment target: compiler default "
                       "(minimum supported: ${LIBHMM_MIN_MACOS_DEPLOYMENT_TARGET}; "
                       "set -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 for Ventura-compatible binaries)")
    endif()
endif()

project(libhmm
    VERSION 4.2.5
    DESCRIPTION "Modern C++20 Hidden Markov Model Library"
    LANGUAGES CXX
)

# GNUInstallDirs gives platform-correct, packager-overridable destinations
# (e.g. lib64 on some Linux distros) instead of hardcoded lib/bin/include.
# Included here (rather than down by the install() rules) because the
# $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> genexes on the hmm/
# hmm_static targets substitute CMAKE_INSTALL_INCLUDEDIR as a plain variable
# at the target_include_directories() call site — it must already be set by
# then, or it silently expands to an empty install interface.
include(GNUInstallDirs)

# C++20 standard — required for std::span, concepts, and constexpr improvements
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# compile_commands.json for clangd/IDEs; top-level only (house style §10).
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# =============================================================================
# Minimum compiler version enforcement.
# v4.0.0 uses C++20 concepts, std::ranges, and std::span without fallback.
# All three platforms must provide a complete C++20 standard library.
# =============================================================================
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0")
        message(FATAL_ERROR
            "libhmm v4.0.0 requires Apple Clang 14 (Xcode 14) or later.\n"
            "Detected: Apple Clang ${CMAKE_CXX_COMPILER_VERSION}.\n"
            "Install Xcode 14 or later from the App Store or developer.apple.com."
        )
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12.0")
        message(FATAL_ERROR
            "libhmm v4.0.0 requires GCC 12 or later.\n"
            "Detected: GCC ${CMAKE_CXX_COMPILER_VERSION}."
        )
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0")
        message(FATAL_ERROR
            "libhmm v4.0.0 requires Clang 14 or later.\n"
            "Detected: Clang ${CMAKE_CXX_COMPILER_VERSION}."
        )
    endif()
elseif(MSVC)
    if(MSVC_VERSION LESS 1930)
        message(FATAL_ERROR
            "libhmm v4.0.0 requires MSVC 2022 (Visual Studio 2022 17.0, _MSC_VER >= 1930).\n"
            "Detected: _MSC_VER ${MSVC_VERSION}."
        )
    endif()
endif()

# =============================================================================
# Deprecated option-name shim.
# libhmm renamed its unprefixed BUILD_*/ENABLE_* options to LIBHMM_*-prefixed
# names (unprefixed names collide with sibling options in superprojects that
# add_subdirectory/FetchContent this repo — see CMAKE-HOUSE-STYLE.md §2). Old
# spellings are honored for one release with a deprecation warning.
# REMOVE THIS WHOLE SHIM BLOCK at the next minor release (v4.3.0).
# =============================================================================
if(DEFINED BUILD_EXAMPLES AND NOT DEFINED LIBHMM_BUILD_EXAMPLES)
    message(DEPRECATION "BUILD_EXAMPLES is deprecated; use LIBHMM_BUILD_EXAMPLES instead")
    set(LIBHMM_BUILD_EXAMPLES ${BUILD_EXAMPLES} CACHE BOOL "Build example programs")
endif()
if(DEFINED BUILD_TESTS AND NOT DEFINED LIBHMM_BUILD_TESTS)
    message(DEPRECATION "BUILD_TESTS is deprecated; use LIBHMM_BUILD_TESTS instead")
    set(LIBHMM_BUILD_TESTS ${BUILD_TESTS} CACHE BOOL "Build test programs")
endif()
if(DEFINED BUILD_TOOLS AND NOT DEFINED LIBHMM_BUILD_TOOLS)
    message(DEPRECATION "BUILD_TOOLS is deprecated; use LIBHMM_BUILD_TOOLS instead")
    set(LIBHMM_BUILD_TOOLS ${BUILD_TOOLS} CACHE BOOL "Build performance inspection and validation tools")
endif()
if(DEFINED BUILD_BENCHMARKS AND NOT DEFINED LIBHMM_BUILD_BENCHMARKS)
    message(DEPRECATION "BUILD_BENCHMARKS is deprecated; use LIBHMM_BUILD_BENCHMARKS instead")
    set(LIBHMM_BUILD_BENCHMARKS ${BUILD_BENCHMARKS} CACHE BOOL "Build comparison benchmarks (requires external HMM libraries)")
endif()
if(DEFINED ENABLE_CLANG_TIDY AND NOT DEFINED LIBHMM_ENABLE_CLANG_TIDY)
    message(DEPRECATION "ENABLE_CLANG_TIDY is deprecated; use LIBHMM_ENABLE_CLANG_TIDY instead")
    set(LIBHMM_ENABLE_CLANG_TIDY ${ENABLE_CLANG_TIDY} CACHE BOOL "Enable clang-tidy analysis during build")
endif()
# ENABLE_STATIC_ANALYSIS / ENABLE_CPPCHECK were declared but never consumed by
# any CMakeLists in this repo (cppcheck runs from CI/scripts directly) — dead
# options, deleted rather than renamed. Old spellings map to nothing; a stale
# -D flag surfaces via this message instead of silently no-op'ing.
if(DEFINED ENABLE_STATIC_ANALYSIS)
    message(DEPRECATION "ENABLE_STATIC_ANALYSIS was removed (never consumed); cppcheck runs from CI/scripts")
endif()
if(DEFINED ENABLE_CPPCHECK)
    message(DEPRECATION "ENABLE_CPPCHECK was removed (never consumed); cppcheck runs from CI/scripts")
endif()

# Build options
# Both static and shared libraries are always built (sources compiled once via
# an OBJECT library) — there is no BUILD_SHARED_LIBS toggle; see the hmm /
# hmm_static target definitions below.
option(LIBHMM_BUILD_EXAMPLES "Build example programs" ${PROJECT_IS_TOP_LEVEL})
option(LIBHMM_BUILD_TESTS "Build test programs" ${PROJECT_IS_TOP_LEVEL})
option(LIBHMM_BUILD_TOOLS "Build performance inspection and validation tools" ${PROJECT_IS_TOP_LEVEL})
option(LIBHMM_BUILD_BENCHMARKS "Build comparison benchmarks (requires external HMM libraries)" OFF)
option(LIBHMM_ENABLE_CLANG_TIDY "Enable clang-tidy analysis during build" OFF)
option(LIBHMM_WERROR "Treat compiler warnings as errors (-Werror / /WX)" OFF)
# LIBHMM_PORTABLE: swap -march=native for a portable baseline ISA on LIBHMM_SIMD_SOURCES.
# Use this when building distributable wheels so they do not SIGILL on machines
# with a lower ISA than the build host.  Runtime-dispatched tier-2 distribution
# kernels (DoubleVecOps) are unaffected and continue to dispatch at full speed.
# Local builds should leave this OFF.  See GitHub issue #58 for the long-term
# plan to extend tier-2 dispatch to the affected TUs (no performance trade-off).
option(LIBHMM_PORTABLE "Portable baseline ISA for LIBHMM_SIMD_SOURCES (for distributable wheels; OFF by default)" OFF)
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
    # Architecture-aware Homebrew prefix for macOS
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE APPLE_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(APPLE_ARCH STREQUAL "arm64")
        set(HOMEBREW_PREFIX "/opt/homebrew")
        message(STATUS "Detected Apple Silicon (${APPLE_ARCH}) — Homebrew at ${HOMEBREW_PREFIX}")
    else()
        set(HOMEBREW_PREFIX "/usr/local")
        message(STATUS "Detected Intel Mac (${APPLE_ARCH}) — Homebrew at ${HOMEBREW_PREFIX}")
    endif()

    set(CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}" ${CMAKE_PREFIX_PATH})

elseif(LINUX)
    # On Linux, check common package manager locations
    set(CMAKE_PREFIX_PATH "/usr" "/usr/local" ${CMAKE_PREFIX_PATH})
endif()

# find_package(Boost REQUIRED)  # REMOVED: No longer needed after Phase 8.3 Boost replacement

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

# LIBHMM_WARN_FLAGS / LIBHMM_SHARED_LINK_OPTIONS are applied PRIVATE, per
# target (hmm_objects below; hmm for the link options; tests/tools/examples/
# benchmarks in their own subdir CMakeLists apply LIBHMM_WARN_FLAGS too)
# rather than via directory-scope add_compile_options/CMAKE_SHARED_LINKER_FLAGS,
# so a consumer embedding libhmm via add_subdirectory/FetchContent never
# inherits our warnings or link flags. The warning set is gated on
# PROJECT_IS_TOP_LEVEL for the same reason.
set(LIBHMM_WARN_FLAGS "")
set(LIBHMM_SHARED_LINK_OPTIONS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(PROJECT_IS_TOP_LEVEL)
        set(LIBHMM_WARN_FLAGS -Wall -Wextra -Wpedantic -Wpointer-arith)
        # Note: -Wno-deprecated-declarations and -Wno-defaulted-function-deleted removed
        # to surface all warnings. Re-add with justification if specific cases cannot be fixed.
        # GCC/Clang build-type defaults (Debug: -g; Release: -O3 -DNDEBUG) are used as-is.
        if(LIBHMM_WERROR)
            list(APPEND LIBHMM_WARN_FLAGS -Werror)
        endif()
    endif()

    # Position-independent code comes from the POSITION_INDEPENDENT_CODE
    # target property on hmm_objects (below), not a global -fPIC flag.

    if(APPLE)
        set(LIBHMM_SHARED_LINK_OPTIONS -undefined dynamic_lookup)
    elseif(LINUX)
        set(LIBHMM_SHARED_LINK_OPTIONS -Wl,--no-undefined)
    endif()
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/SimdDispatch.cmake")

# Source files
set(LIBHMM_SOURCES
    src/hmm.cpp
    src/common/common.cpp
    src/common/string_tokenizer.cpp
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
    src/distributions/von_mises_distribution.cpp
    # Canonical calculators (Phase H: scalar + MV explicit instantiations)
    src/calculators/forward_backward_calculator.cpp
    src/calculators/forward_backward_calculator_mv.cpp
    src/calculators/viterbi_calculator.cpp
    src/calculators/viterbi_calculator_mv.cpp
    # Canonical trainers (Phase H: scalar + MV explicit instantiations)
    src/training/baum_welch_trainer.cpp
    src/training/baum_welch_trainer_mv.cpp
    src/training/map_baum_welch_trainer.cpp
    src/training/map_baum_welch_trainer_mv.cpp
    src/training/viterbi_trainer.cpp
    src/training/viterbi_trainer_mv.cpp
    src/training/segmental_kmeans_trainer.cpp
    src/training/segmental_kmeans_trainer_mv.cpp
    src/training/kmeans_init.cpp
    src/io/file_io_manager.cpp
    src/io/json_utils.cpp
    src/io/hmm_json.cpp
    src/io/xml_file_reader.cpp
    src/io/xml_file_writer.cpp
    src/model_selection.cpp
    # Linalg helpers (Phase F+)
    src/linalg/cholesky.cpp
    # Multivariate distributions (Phase G)
    src/distributions/independent_components_distribution.cpp
    src/distributions/diagonal_gaussian_distribution.cpp
    src/distributions/full_covariance_gaussian_distribution.cpp
)

# Append runtime dispatch TUs (computed before set() reset above).
list(APPEND LIBHMM_SOURCES ${LIBHMM_DISPATCH_SOURCES})

# ── Library targets ───────────────────────────────────────────────────────────
# Sources are compiled once into an OBJECT library, then linked into both a
# SHARED and a STATIC target. This avoids compiling the source tree twice and
# ensures the two library variants are always bit-identical.

add_library(hmm_objects OBJECT ${LIBHMM_SOURCES})

# Target-scope include dir for the library's own TUs — final libs (hmm,
# hmm_static) get their own PUBLIC BUILD_INTERFACE/INSTALL_INTERFACE include
# dirs below; this one is PRIVATE and only for compiling the object library.
target_include_directories(hmm_objects PRIVATE include)

# MSVC compiler flags — applied to the OBJECT library so they affect all
# compiled TUs regardless of which library type is ultimately consumed.
# /W4: strong warnings. /permissive-: strict standards conformance.
# /external:anglebrackets /external:W0 suppresses warnings from system headers
# (e.g. C4244 from <memory> template instantiations — MSVC limitation).
if(MSVC)
    target_compile_options(hmm_objects PRIVATE
        /W4 /permissive- /external:anglebrackets /external:W0)
    if(LIBHMM_WERROR)
        target_compile_options(hmm_objects PRIVATE /WX)
    endif()
endif()
target_compile_options(hmm_objects PRIVATE ${LIBHMM_WARN_FLAGS})

# =============================================================================
# Probe for C++17 std::cyl_bessel_i (modified Bessel function of the first kind).
#
# Available on GCC 6.1+ and MSVC 2017 15.5+. NOT available on AppleClang/libc++
# as of Xcode 15 — Apple's libc++ has not implemented the C++17 special math
# functions. When absent, VonMisesDistribution falls back to the Abramowitz &
# Stegun §9.8.1-9.8.4 polynomial approximations in include/libhmm/math/bessel.h.
# =============================================================================
check_cxx_source_compiles(
    "#include <cmath>
int main() {
    double r = std::cyl_bessel_i(0.0, 1.0);
    return r > 0.0 ? 0 : 1;
}"
    LIBHMM_HAS_CXX17_BESSEL
)
if(LIBHMM_HAS_CXX17_BESSEL)
    target_compile_definitions(hmm_objects PRIVATE LIBHMM_HAS_CXX17_BESSEL)
    message(STATUS "C++17 std::cyl_bessel_i: available")
else()
    message(STATUS "C++17 std::cyl_bessel_i: not available (A&S polynomial fallback active)")
endif()

# PIC is required for objects that go into the shared library; harmless for static.
set_property(TARGET hmm_objects PROPERTY POSITION_INDEPENDENT_CODE ON)

# Shared library
add_library(hmm SHARED $<TARGET_OBJECTS:hmm_objects>)
add_library(libhmm::hmm ALIAS hmm)
# PUBLIC include dirs are required directly on hmm (not just hmm_objects):
# install(EXPORT) only captures properties set directly on the exported
# target — without this, the exported libhmm::hmm target carries no include
# path at all and downstream find_package(libhmm) consumers fail to compile.
target_include_directories(hmm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
# Shared-library link flags — see LIBHMM_SHARED_LINK_OPTIONS definition above
# (deferred symbol resolution on macOS; explicit --no-undefined on Linux so a
# missing symbol fails the link rather than only surfacing at load time).
target_link_options(hmm PRIVATE ${LIBHMM_SHARED_LINK_OPTIONS})

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
add_library(libhmm::hmm_static ALIAS hmm_static)
target_include_directories(hmm_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Enable clang-tidy if requested — applied to the OBJECT library so analysis
# runs once on all compiled TUs regardless of which library type is built.
if(LIBHMM_ENABLE_CLANG_TIDY)
    # Search standard locations; do not pin a version-specific Homebrew Cellar
    # path — that breaks on CI Linux, Windows, and any macOS with a different
    # LLVM version.  cmake --preset release -DLIBHMM_ENABLE_CLANG_TIDY=ON discovers
    # whatever clang-tidy is first on PATH.
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)
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
    EXPORT libhmm-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}          # .so / .dylib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}          # .a / .lib (static archive + Windows import lib)
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}          # .dll (Windows)
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libhmm
)

install(EXPORT libhmm-targets
    NAMESPACE libhmm::
    FILE libhmm-targets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libhmm
)

install(DIRECTORY include/libhmm DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# pkg-config support (UNIX, including APPLE — mirrors libstats).
if(UNIX)
    if(LINUX)
        set(LIBHMM_PC_LIBS_PRIVATE "-lm")
    else()
        set(LIBHMM_PC_LIBS_PRIVATE "")
    endif()
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libhmm.pc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/libhmm.pc"
        @ONLY
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libhmm.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    )
endif()

# Examples
if(LIBHMM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(LIBHMM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
# Benchmarks (comparative and internal performance suites)
if(LIBHMM_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Performance inspection and validation tools
if(LIBHMM_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libhmm-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# PATH_VARS INCLUDE_INSTALL_DIR is required so @PACKAGE_INCLUDE_INSTALL_DIR@ in
# the template expands to a real, install-relocatable path instead of being
# left unexpanded (configure_package_config_file only substitutes variables
# named in PATH_VARS).
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libhmm-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libhmm-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libhmm
    PATH_VARS INCLUDE_INSTALL_DIR
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/libhmm-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/libhmm-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libhmm
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
message(STATUS "Examples:     ${LIBHMM_BUILD_EXAMPLES}")
message(STATUS "Tests:        ${LIBHMM_BUILD_TESTS}")
message(STATUS "Tools:        ${LIBHMM_BUILD_TOOLS}")
message(STATUS "Benchmarks:   ${LIBHMM_BUILD_BENCHMARKS}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "=")
