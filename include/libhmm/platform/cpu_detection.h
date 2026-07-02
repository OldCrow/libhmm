#pragma once

/// @file cpu_detection.h
/// @brief Runtime CPU ISA feature detection for libhmm's dispatch infrastructure.
///
/// Each function is backed by a function-local static bool, initialized exactly
/// once (thread-safe per C++11). The implementation (cpu_detection.cpp) is
/// compiled WITHOUT any SIMD flags so the detection code itself runs on every
/// CPU and never triggers an illegal-instruction exception.
///
/// On AArch64, NEON is mandatory per the architecture spec; supports_neon()
/// returns true unconditionally. All x86 functions return false.

namespace libhmm::platform {

/// @returns true if the runtime CPU + OS support AVX-512F/DQ.
/// Always false on non-x86 platforms.
bool supports_avx512() noexcept;

/// @returns true if the runtime CPU + OS support AVX2 and FMA.
/// Always false on non-x86 platforms.
bool supports_avx2() noexcept;

/// @returns true if the runtime CPU supports SSE2.
/// Always true on x86-64 (SSE2 is the mandatory baseline ISA).
/// Always false on non-x86 platforms.
bool supports_sse2() noexcept;

/// @returns true if ARM NEON is available.
/// Always true on AArch64 (NEON is the mandatory baseline ISA).
/// Always false on x86 platforms.
bool supports_neon() noexcept;

} // namespace libhmm::platform
