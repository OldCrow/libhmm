# SimdDispatch.cmake - Compile-time SIMD flag selection and runtime dispatch TU wiring
#
# Extracted verbatim from the top-level CMakeLists.txt SIMD block. Include this module at the exact
# original location (after LIBHMM_PORTABLE is declared, before LIBHMM_SOURCES is set) — it relies on
# CMAKE_CURRENT_SOURCE_DIR referring to the top-level source directory (true for include(), unlike
# add_subdirectory()) and it sets plain variables (LIBHMM_BEST_SIMD_FLAGS, LIBHMM_SIMD_SOURCES,
# LIBHMM_DISPATCH_SOURCES, LIBHMM_DISPATCH_DEFINES, COMPILER_SUPPORTS_*) that the includer's later
# code and tests/CMakeLists.txt / tools/CMakeLists.txt depend on.

# =============================================================================
# SIMD DETECTION — compile-time dispatch, build machine CPU matters
#
# libhmm uses compile-time SIMD dispatch: each machine builds a binary optimised for its own CPU.
# LIBHMM_BEST_SIMD_FLAGS is applied per-TU to LIBHMM_SIMD_SOURCES only (not globally) so non-SIMD
# code paths stay at the baseline ISA.
#
# GCC/Clang: -march=native asks the compiler to use everything the build machine's CPU supports.
# This is both compiler and CPU aware.
#
# MSVC: no -march=native equivalent. check_cxx_compiler_flag only tests whether the compiler ACCEPTS
# a flag (always true for /arch:AVX512 on modern MSVC), NOT whether the build machine's CPU can
# execute the resulting instructions. check_cxx_source_runs compiles and runs a tiny test, verifying
# both.
#
# LIBHMM_PORTABLE=ON: skip CPU probing and use a portable baseline ISA instead of -march=native, so
# LIBHMM_SIMD_SOURCES TUs are safe to distribute. GCC/Clang: -msse2 (x86) or -march=armv8-a
# (AArch64). MSVC: empty (SSE2 is the x64 ABI baseline; no further /arch flag).
# =============================================================================
include(CheckCXXCompilerFlag)

# x86 ISA extension flags are only meaningful on x86/x86_64 targets. Running them on AArch64 (Apple
# Silicon, ARM servers) produces misleading results because Clang silently accepts x86 flags without
# enabling those ISAs.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    # AArch64: NEON is the mandatory ISA baseline — always present, no flag needed.
    set(COMPILER_SUPPORTS_NEON ON)
    message(STATUS "SIMD compiler support — NEON: ON (AArch64 baseline)")
elseif(NOT MSVC)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
    check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)
    message(
        STATUS
            "SIMD compiler support — AVX-512: ${COMPILER_SUPPORTS_AVX512F}  AVX2: ${COMPILER_SUPPORTS_AVX2}  AVX: ${COMPILER_SUPPORTS_AVX}"
    )
else()
    # MSVC x86/x64: informational only — actual flag selected via check_cxx_source_runs below.
    check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512F)
    check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("/arch:AVX" COMPILER_SUPPORTS_AVX)
    set(COMPILER_SUPPORTS_SSE42 ON) # SSE4.2 is guaranteed on x64 MSVC
    message(
        STATUS
            "SIMD compiler support — AVX-512: ${COMPILER_SUPPORTS_AVX512F}  AVX2: ${COMPILER_SUPPORTS_AVX2}  AVX: ${COMPILER_SUPPORTS_AVX}"
    )
endif()

# =============================================================================
# Select best SIMD flag for this compiler + CPU combination.
#
# GCC/Clang: -march=native is both compiler-aware and CPU-aware. MSVC: use check_cxx_source_runs to
# compile AND execute a minimal test, mirroring what -march=native does automatically on GCC/Clang.
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
        # Portable mode: use a stable baseline ISA so the built TUs are safe to distribute.
        # Runtime-dispatched tier-2 kernels are unaffected.
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
            set(LIBHMM_BEST_SIMD_FLAGS "-march=armv8-a")
            message(STATUS "SIMD optimization: -march=armv8-a (AArch64 — portable mode)")
        else()
            set(LIBHMM_BEST_SIMD_FLAGS "-msse2")
            message(STATUS "SIMD optimization: -msse2 (portable mode)")
        endif()
    else()
        # GCC/Clang: -march=native selects the best ISA for the build machine (NEON on AArch64;
        # AVX-512/AVX2/AVX/SSE2 on x86_64).
        set(LIBHMM_BEST_SIMD_FLAGS "-march=native")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
            message(STATUS "SIMD optimization: -march=native (AArch64/NEON)")
        else()
            message(STATUS "SIMD optimization: -march=native")
        endif()
    endif()
endif()

# Distribution sources with getBatchLogProbabilities() overrides that still compile with
# LIBHMM_BEST_SIMD_FLAGS for compiler auto-vectorization or recurrence helper use.
# Runtime-dispatched tier-2 distribution kernels live in the per-ISA simd_double_ops_*.cpp TUs
# below.
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
    src/distributions/von_mises_distribution.cpp)

# Additional TUs that include transcendental_kernels.h and therefore need LIBHMM_BEST_SIMD_FLAGS to
# activate the #if LIBHMM_HAS_* cascade. Both scalar and MV explicit-instantiation TUs are compiled
# with SIMD flags so the shared transition recurrence and xi accumulation kernel benefit from
# vectorisation.
list(
    APPEND
    LIBHMM_SIMD_SOURCES
    src/performance/transcendental_kernels.cpp
    src/calculators/forward_backward_calculator.cpp
    src/calculators/forward_backward_calculator_mv.cpp
    src/training/baum_welch_trainer.cpp
    src/training/baum_welch_trainer_mv.cpp
    # MapBaumWelchTrainer calls accumulate_exp_sum2_bias — needs SIMD flags.
    src/training/map_baum_welch_trainer.cpp
    src/training/map_baum_welch_trainer_mv.cpp)

# Runtime-dispatched distribution wrappers no longer contain SIMD intrinsics; they call through the
# DoubleVecOps table. Remove them from LIBHMM_SIMD_SOURCES so -march=native is not applied to those
# wrapper TUs.
list(REMOVE_ITEM LIBHMM_SIMD_SOURCES src/distributions/gaussian_distribution.cpp
     src/distributions/exponential_distribution.cpp)

if(LIBHMM_BEST_SIMD_FLAGS)
    foreach(simd_src ${LIBHMM_SIMD_SOURCES})
        # COMPILE_OPTIONS replaces the deprecated COMPILE_FLAGS property: it handles lists correctly
        # and supports generator expressions.
        set_source_files_properties(${simd_src} PROPERTIES COMPILE_OPTIONS
                                                           "${LIBHMM_BEST_SIMD_FLAGS}")
    endforeach()
    message(STATUS "Applied SIMD flags to ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# =============================================================================
# Runtime SIMD dispatch infrastructure (H-3)
#
# Each ISA tier is compiled into its own TU with a targeted per-file flag (-mavx512f, -mavx2, etc.)
# so the resulting binary contains code for every compiled tier without requiring -march=native on
# the consuming machine.
#
# simd_dispatch.cpp receives COMPILE_DEFINITIONS (LIBHMM_BUILD_*_KERNEL) indicating which per-ISA
# TUs were compiled in; it forward-declares and references only those symbols, avoiding linker
# errors on unavailable tiers.
# =============================================================================

set(LIBHMM_DISPATCH_DEFINES "")
# LIBHMM_DISPATCH_SOURCES accumulates new TUs here; appended to LIBHMM_SOURCES AFTER the
# set(LIBHMM_SOURCES ...) block resets it below.
set(LIBHMM_DISPATCH_SOURCES
    src/platform/cpu_detection.cpp src/performance/simd_double_ops_scalar.cpp
    src/performance/simd_dispatch.cpp)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    # AArch64: NEON is the mandatory baseline — no extra compile flag needed.
    list(APPEND LIBHMM_DISPATCH_SOURCES src/performance/simd_double_ops_neon.cpp)
    list(APPEND LIBHMM_DISPATCH_DEFINES "LIBHMM_BUILD_NEON_KERNEL")
    message(STATUS "Runtime dispatch: NEON (AArch64 baseline)")
else()
    # x86/x64: add each ISA tier if the compiler supports it. SSE2 is the x86-64 ABI baseline —
    # always available.
    list(APPEND LIBHMM_DISPATCH_SOURCES src/performance/simd_double_ops_sse2.cpp)
    list(APPEND LIBHMM_DISPATCH_DEFINES "LIBHMM_BUILD_SSE2_KERNEL")
    set_source_files_properties(src/performance/simd_double_ops_sse2.cpp
                                PROPERTIES COMPILE_OPTIONS "$<IF:$<CXX_COMPILER_ID:MSVC>,,-msse2>")

    if(COMPILER_SUPPORTS_AVX2)
        list(APPEND LIBHMM_DISPATCH_SOURCES src/performance/simd_double_ops_avx2.cpp)
        list(APPEND LIBHMM_DISPATCH_DEFINES "LIBHMM_BUILD_AVX2_KERNEL")
        set_source_files_properties(
            src/performance/simd_double_ops_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "$<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX2,-mavx2;-mfma>")
    endif()

    if(COMPILER_SUPPORTS_AVX512F)
        list(APPEND LIBHMM_DISPATCH_SOURCES src/performance/simd_double_ops_avx512.cpp)
        list(APPEND LIBHMM_DISPATCH_DEFINES "LIBHMM_BUILD_AVX512_KERNEL")
        set_source_files_properties(
            src/performance/simd_double_ops_avx512.cpp
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
# Note: MSVC with /arch:AVX512 emits 2 × C4244 from <memory> internal templates when AVX-512
# register types are instantiated. /external:anglebrackets /external:W0 (already on the hmm target)
# does not suppress them — a known MSVC limitation in template specialization warning paths.
# Accepted as external-code noise.
