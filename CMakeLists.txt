cmake_minimum_required(VERSION 3.20)

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
    VERSION 4.2.4
    DESCRIPTION "Modern C++20 Hidden Markov Model Library"
    LANGUAGES CXX
)

# C++20 standard — required for std::span, concepts, and constexpr improvements
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
    add_compile_options(-Wall -Wextra -Wpedantic -Wpointer-arith)
    # Note: -Wno-deprecated-declarations and -Wno-defaulted-function-deleted removed
    # to surface all warnings. Re-add with justification if specific cases cannot be fixed.
    # GCC/Clang build-type defaults (Debug: -g; Release: -O3 -DNDEBUG) are used as-is.

    # Position-independent code for shared libraries on Unix platforms
    if(APPLE OR LINUX OR UNIX)
        add_compile_options(-fPIC)
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
#
# LIBHMM_PORTABLE=ON: skip CPU probing and use a portable baseline ISA instead
# of -march=native, so LIBHMM_SIMD_SOURCES TUs are safe to distribute.
# GCC/Clang: -msse2 (x86) or -march=armv8-a (AArch64).
# MSVC: empty (SSE2 is the x64 ABI baseline; no further /arch flag).
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
if(MSVC)
    if(LIBHMM_PORTABLE OR CMAKE_CROSSCOMPILING)
        # Portable or cross-compilation: use SSE2 baseline (x64 ABI minimum).
        set(LIBHMM_BEST_SIMD_FLAGS "")
        if(LIBHMM_PORTABLE)
            message(STATUS "SIMD optimization: (SSE2 baseline — portable mode)")
        else()
            message(STATUS "SIMD optimization: (cross-compilation — SSE2 baseline)")
        endif()
    else()
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
    endif()
else()
    if(LIBHMM_PORTABLE)
        # Portable mode: use a stable baseline ISA so the built TUs are safe
        # to distribute.  Runtime-dispatched tier-2 kernels are unaffected.
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
            set(LIBHMM_BEST_SIMD_FLAGS "-march=armv8-a")
            message(STATUS "SIMD optimization: -march=armv8-a (AArch64 — portable mode)")
        else()
            set(LIBHMM_BEST_SIMD_FLAGS "-msse2")
            message(STATUS "SIMD optimization: -msse2 (portable mode)")
        endif()
    else()
        # GCC/Clang: -march=native selects the best ISA for the build machine
        # (NEON on AArch64; AVX-512/AVX2/AVX/SSE2 on x86_64).
        set(LIBHMM_BEST_SIMD_FLAGS "-march=native")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
            message(STATUS "SIMD optimization: -march=native (AArch64/NEON)")
        else()
            message(STATUS "SIMD optimization: -march=native")
        endif()
    endif()
endif()

# Distribution sources with getBatchLogProbabilities() overrides that still
# compile with LIBHMM_BEST_SIMD_FLAGS for compiler auto-vectorization or
# recurrence helper use. Runtime-dispatched tier-2 distribution kernels live
# in the per-ISA simd_double_ops_*.cpp TUs below.
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
    src/distributions/von_mises_distribution.cpp
)

# Additional TUs that include transcendental_kernels.h and therefore need
# LIBHMM_BEST_SIMD_FLAGS to activate the #if LIBHMM_HAS_* cascade.
# Both scalar and MV explicit-instantiation TUs are compiled with SIMD flags so the
# shared transition recurrence and xi accumulation kernel benefit from vectorisation.
list(APPEND LIBHMM_SIMD_SOURCES
    src/performance/transcendental_kernels.cpp
    src/calculators/forward_backward_calculator.cpp
    src/calculators/forward_backward_calculator_mv.cpp
    src/training/baum_welch_trainer.cpp
    src/training/baum_welch_trainer_mv.cpp
    # MapBaumWelchTrainer calls accumulate_exp_sum2_bias — needs SIMD flags.
    src/training/map_baum_welch_trainer.cpp
    src/training/map_baum_welch_trainer_mv.cpp
)

# Runtime-dispatched distribution wrappers no longer contain SIMD intrinsics;
# they call through the DoubleVecOps table. Remove them from LIBHMM_SIMD_SOURCES
# so -march=native is not applied to those wrapper TUs.
list(REMOVE_ITEM LIBHMM_SIMD_SOURCES
    src/distributions/gaussian_distribution.cpp
    src/distributions/exponential_distribution.cpp)

if(LIBHMM_BEST_SIMD_FLAGS)
    foreach(simd_src ${LIBHMM_SIMD_SOURCES})
        # COMPILE_OPTIONS replaces the deprecated COMPILE_FLAGS property:
        # it handles lists correctly and supports generator expressions.
        set_source_files_properties(
            ${simd_src}
            PROPERTIES COMPILE_OPTIONS "${LIBHMM_BEST_SIMD_FLAGS}"
        )
    endforeach()
    message(STATUS "Applied SIMD flags to ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# =============================================================================
# Runtime SIMD dispatch infrastructure (H-3)
#
# Each ISA tier is compiled into its own TU with a targeted per-file flag
# (-mavx512f, -mavx2, etc.) so the resulting binary contains code for every
# compiled tier without requiring -march=native on the consuming machine.
#
# simd_dispatch.cpp receives COMPILE_DEFINITIONS (LIBHMM_BUILD_*_KERNEL)
# indicating which per-ISA TUs were compiled in; it forward-declares and
# references only those symbols, avoiding linker errors on unavailable tiers.
# =============================================================================

set(LIBHMM_DISPATCH_DEFINES "")
# LIBHMM_DISPATCH_SOURCES accumulates new TUs here; appended to LIBHMM_SOURCES
# AFTER the set(LIBHMM_SOURCES ...) block resets it below.
set(LIBHMM_DISPATCH_SOURCES
    src/platform/cpu_detection.cpp
    src/performance/simd_double_ops_scalar.cpp
    src/performance/simd_dispatch.cpp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    # AArch64: NEON is the mandatory baseline — no extra compile flag needed.
    list(APPEND LIBHMM_DISPATCH_SOURCES src/performance/simd_double_ops_neon.cpp)
    list(APPEND LIBHMM_DISPATCH_DEFINES "LIBHMM_BUILD_NEON_KERNEL")
    message(STATUS "Runtime dispatch: NEON (AArch64 baseline)")
else()
    # x86/x64: add each ISA tier if the compiler supports it.
    # SSE2 is the x86-64 ABI baseline — always available.
    list(APPEND LIBHMM_DISPATCH_SOURCES src/performance/simd_double_ops_sse2.cpp)
    list(APPEND LIBHMM_DISPATCH_DEFINES "LIBHMM_BUILD_SSE2_KERNEL")
    set_source_files_properties(src/performance/simd_double_ops_sse2.cpp
        PROPERTIES COMPILE_OPTIONS
            "$<IF:$<CXX_COMPILER_ID:MSVC>,,-msse2>")

    if(COMPILER_SUPPORTS_AVX2)
        list(APPEND LIBHMM_DISPATCH_SOURCES src/performance/simd_double_ops_avx2.cpp)
        list(APPEND LIBHMM_DISPATCH_DEFINES "LIBHMM_BUILD_AVX2_KERNEL")
        set_source_files_properties(src/performance/simd_double_ops_avx2.cpp
            PROPERTIES COMPILE_OPTIONS
                "$<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX2,-mavx2;-mfma>")
    endif()

    if(COMPILER_SUPPORTS_AVX512F)
        list(APPEND LIBHMM_DISPATCH_SOURCES src/performance/simd_double_ops_avx512.cpp)
        list(APPEND LIBHMM_DISPATCH_DEFINES "LIBHMM_BUILD_AVX512_KERNEL")
        set_source_files_properties(src/performance/simd_double_ops_avx512.cpp
            PROPERTIES COMPILE_OPTIONS
                "$<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX512,-mavx512f;-mavx512dq>")
    endif()

    message(STATUS "Runtime dispatch tiers: ${LIBHMM_DISPATCH_DEFINES}")
endif()

# Communicate compiled-in tiers to simd_dispatch.cpp only (not globally).
if(LIBHMM_DISPATCH_DEFINES)
    set_source_files_properties(src/performance/simd_dispatch.cpp
        PROPERTIES COMPILE_DEFINITIONS "${LIBHMM_DISPATCH_DEFINES}")
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

# MSVC compiler flags — applied to the OBJECT library so they affect all
# compiled TUs regardless of which library type is ultimately consumed.
# /W4: strong warnings. /permissive-: strict standards conformance.
# /external:anglebrackets /external:W0 suppresses warnings from system headers
# (e.g. C4244 from <memory> template instantiations — MSVC limitation).
if(MSVC)
    target_compile_options(hmm_objects PRIVATE
        /W4 /permissive- /external:anglebrackets /external:W0)
endif()

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
    # Search standard locations; do not pin a version-specific Homebrew Cellar
    # path — that breaks on CI Linux, Windows, and any macOS with a different
    # LLVM version.  cmake --preset release -DENABLE_CLANG_TIDY=ON discovers
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
    LIBRARY DESTINATION lib          # .so / .dylib
    ARCHIVE DESTINATION lib          # .a / .lib (static archive + Windows import lib)
    RUNTIME DESTINATION bin          # .dll (Windows)
    PUBLIC_HEADER DESTINATION include/libhmm
)

install(DIRECTORY include/libhmm DESTINATION include
    PATTERN "detail" EXCLUDE)

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
message(STATUS "=")
