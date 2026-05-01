#pragma once

/**
 * @file fb_recurrence_policy.h
 * @brief Architecture/compiler-aware policy for Forward-Backward recurrence kernel selection.
 *
 * The Forward-Backward recurrence has two semantically equivalent kernels:
 *   - Pairwise: repeated stable two-argument log-sum-exp.
 *   - MaxReduce: max-then-reduce (find max, then sum exp differences).
 *
 * Empirical contour evidence shows the crossover between these kernels depends on
 * compiler and ISA more than on raw architecture. This header centralizes the
 * policy used to choose between them, grounded in the "policy-defining evidence"
 * subsections of the plan's Appendix A.
 *
 * Design constraints:
 *   - Pure compile-time policy here (constexpr); runtime overrides and probing
 *     live in the calculator implementation.
 *   - Log-space semantics are preserved by either kernel.
 *   - Default to Pairwise in any unknown configuration to protect comparator
 *     low-state behavior.
 */

#include <cstddef>

namespace libhmm {

/// Selectable recurrence kernel for Forward-Backward.
enum class FbRecurrenceMode {
    Pairwise,
    MaxReduce,
};

/// Compiler identification used for policy bins.
/// Order of detection matters: clang-cl defines both `_MSC_VER` and `__clang__`,
/// and must be checked first.
enum class FbCompiler {
    Unknown,
    Msvc,
    ClangCl,
    Clang,
    Gcc,
};

/// Host profile derived entirely from compile-time predefined macros.
struct FbHostProfile {
    FbCompiler compiler;
};

/// Build the host profile for the current translation unit.
/// Compiler identity is the primary policy axis; architecture-specific
/// branches within each compiler case use preprocessor checks directly.
constexpr FbHostProfile makeFbHostProfile() noexcept {
    FbCompiler c = FbCompiler::Unknown;
#if defined(__clang__) && defined(_MSC_VER)
    c = FbCompiler::ClangCl;
#elif defined(_MSC_VER)
    c = FbCompiler::Msvc;
#elif defined(__clang__)
    c = FbCompiler::Clang;
#elif defined(__GNUC__)
    c = FbCompiler::Gcc;
#endif
    return FbHostProfile{c};
}

/// Convenience: profile of the current translation unit.
inline constexpr FbHostProfile kFbCurrentHostProfile = makeFbHostProfile();

/**
 * @brief Static recurrence-mode selection from compiler/ISA evidence.
 *
 * Bins are derived from the plan's Appendix A "policy-defining evidence"
 * subsections. The default in unknown profiles is `Pairwise` to protect
 * comparator-facing low-state workloads.
 *
 * @param numStates       Number of HMM states (`N`).
 * @param sequenceLength  Observation length (`T`). Currently unused except for
 *                         signature stability; reserved for future T-aware bins.
 * @param profile         Host profile (compiler identity).
 */
constexpr FbRecurrenceMode selectFbRecurrenceMode(std::size_t numStates,
                                                  std::size_t sequenceLength,
                                                  FbHostProfile profile) noexcept {
    (void)sequenceLength;
    if (numStates < 2) {
        return FbRecurrenceMode::Pairwise;
    }
    switch (profile.compiler) {
    case FbCompiler::Msvc:
        // Windows / Ryzen / MSVC: pairwise N<=4, max-reduce N>=5.
        return (numStates >= 5) ? FbRecurrenceMode::MaxReduce
                                : FbRecurrenceMode::Pairwise;
    case FbCompiler::ClangCl:
        // Windows / Ryzen / ClangCL with /O2: pairwise N<=3, max-reduce N>=4.
        return (numStates >= 4) ? FbRecurrenceMode::MaxReduce
                                : FbRecurrenceMode::Pairwise;
    case FbCompiler::Gcc:
        // Windows / Ryzen / MinGW GCC and Linux GCC: boundary across N=3..6,
        // favor max-reduce only from N>=7 to keep low-N comparator behavior.
        return (numStates >= 7) ? FbRecurrenceMode::MaxReduce
                                : FbRecurrenceMode::Pairwise;
    case FbCompiler::Clang:
        // Clang split by ISA family:
        //   * arm64 (Apple Silicon): pairwise N<=3, max-reduce N>=4.
        //   * x86_64: Kaby Lake AppleClang shows weak/inconsistent crossover,
        //     so use pairwise as a conservative static default and rely on
        //     boundary probing at runtime for refinement.
#if defined(__aarch64__) || defined(_M_ARM64)
        return (numStates >= 4) ? FbRecurrenceMode::MaxReduce
                                : FbRecurrenceMode::Pairwise;
#else
        return FbRecurrenceMode::Pairwise;
#endif
    case FbCompiler::Unknown:
        return FbRecurrenceMode::Pairwise;
    }
    return FbRecurrenceMode::Pairwise;
}

/**
 * @brief Whether `(N, T)` falls in a region where Stage-2 runtime probing should
 *        refine the static choice.
 *
 * Boundary regions are approximate per-compiler envelopes around the published
 * crossover bins. Stage-1 selection above is still safe to use without probing;
 * Stage-2 probing simply reduces sensitivity to noise near the crossover.
 */
constexpr bool isFbBoundaryPoint(std::size_t numStates,
                                 std::size_t sequenceLength,
                                 FbHostProfile profile) noexcept {
    (void)sequenceLength;
    if (numStates < 2) {
        return false;
    }
    switch (profile.compiler) {
    case FbCompiler::Msvc:
        return numStates >= 3 && numStates <= 5;
    case FbCompiler::ClangCl:
        return numStates >= 3 && numStates <= 4;
    case FbCompiler::Gcc:
        return numStates >= 3 && numStates <= 6;
    case FbCompiler::Clang:
        return numStates >= 3 && numStates <= 6;
    case FbCompiler::Unknown:
        return numStates >= 3 && numStates <= 6;
    }
    return false;
}

/// Human-readable name for a recurrence mode.
constexpr const char *toString(FbRecurrenceMode mode) noexcept {
    switch (mode) {
    case FbRecurrenceMode::Pairwise:
        return "pairwise";
    case FbRecurrenceMode::MaxReduce:
        return "max-reduce";
    }
    return "unknown";
}

/// Human-readable name for a compiler tag.
constexpr const char *toString(FbCompiler compiler) noexcept {
    switch (compiler) {
    case FbCompiler::Msvc:
        return "msvc";
    case FbCompiler::ClangCl:
        return "clang-cl";
    case FbCompiler::Clang:
        return "clang";
    case FbCompiler::Gcc:
        return "gcc";
    case FbCompiler::Unknown:
        return "unknown";
    }
    return "unknown";
}

} // namespace libhmm
