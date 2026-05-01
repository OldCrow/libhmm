#pragma once

/**
 * @file fb_recurrence_policy.h
 * @brief Minimal ISA-aware policy for Forward-Backward recurrence selection.
 *
 * The two recurrence kernels are semantically equivalent in log-space:
 *   - Pairwise: repeated two-argument log-sum-exp
 *   - MaxReduce: max-then-reduce
 *
 * The only policy decision retained here is a conservative ISA-family cutoff:
 *   - arm64: switch at N>=4
 *   - x86/x64: switch at N>=5
 *
 * This keeps the useful large-N reduction in exp/log1p traffic without the
 * previous per-compiler and runtime-probing complexity.
 */

#include <cstddef>

namespace libhmm {

/// Selectable recurrence kernel for Forward-Backward.
enum class FbRecurrenceMode {
    Pairwise,
    MaxReduce,
};


/**
 * @brief Static recurrence-mode selection from ISA-family evidence.
 *
 * @param numStates       Number of HMM states (`N`).
 * @param sequenceLength  Observation length (`T`). Currently unused except for
 *                         signature stability; reserved for future T-aware bins.
 */
constexpr FbRecurrenceMode selectFbRecurrenceMode(std::size_t numStates,
                                                  std::size_t sequenceLength) noexcept {
    (void)sequenceLength;
    if (numStates < 2) {
        return FbRecurrenceMode::Pairwise;
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    return (numStates >= 4) ? FbRecurrenceMode::MaxReduce
                            : FbRecurrenceMode::Pairwise;
#else
    return (numStates >= 5) ? FbRecurrenceMode::MaxReduce
                            : FbRecurrenceMode::Pairwise;
#endif
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


} // namespace libhmm
