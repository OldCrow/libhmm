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

/// Bitmask on CPUID leaf 7 EBX required for libhmm's AVX-512 tier: F (bit 16),
/// DQ (bit 17), BW (bit 30), VL (bit 31) -- exactly what /arch:AVX512 licenses
/// and what simd_double_ops_avx512.cpp's -mavx512f -mavx512dq compile flags
/// require (issue #83: F alone is not enough -- the AVX-512DQ intrinsics used
/// there, e.g. _mm512_cvtepi64_pd, SIGILL on an F-without-DQ part such as
/// Knights Landing/Mill). Exposed so tests/platform/test_simd_platform.cpp can
/// regression-guard the exact value: a one-bit perturbation here must fail
/// that test.
inline constexpr unsigned kAvx512RequiredMask =
    (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31); // F | DQ | BW | VL

/// @returns true if the runtime CPU + OS support AVX-512 F/DQ/BW/VL
/// (kAvx512RequiredMask).
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
