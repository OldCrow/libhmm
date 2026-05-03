#pragma once

/**
 * @file fb_recurrence_policy.h
 * @brief Minimal ISA-aware policy for Forward-Backward recurrence selection.
 *
 * The two recurrence kernels are semantically equivalent in log-space:
 *   - Pairwise: repeated two-argument log-sum-exp
 *   - MaxReduce: max-then-reduce
 *
 * The only policy decision retained here is an ISA-family cutoff:
 *   - arm64: switch at N>=4
 *   - x86/x64: switch at N>=4
 *
 * Threshold calibrated by fb_crossover_sweep on Zen 4 / MSVC / AVX-512
 * (Ryzen 7 7745HX, T=1000, median 8 runs):
 *   N=2: MaxReduce 2.1x slower (Pairwise wins)
 *   N=3: MaxReduce 1.1x slower (Pairwise wins)
 *   N=4: MaxReduce 1.7x faster -- crossover
 *   N=8: MaxReduce 5.0x faster
 *   N=32: MaxReduce 15x faster
 * Previous x86 threshold was N>=5; N=4 was incorrectly left on the slower
 * Pairwise path before the TranscendentalKernels SIMD backends landed.
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
    return (numStates >= 4) ? FbRecurrenceMode::MaxReduce : FbRecurrenceMode::Pairwise;
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
