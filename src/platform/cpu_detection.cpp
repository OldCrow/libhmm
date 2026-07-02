// cpu_detection.cpp — runtime ISA feature detection via CPUID.
//
// CRITICAL: compiled WITHOUT any SIMD flags (-mavx2, -march=native, etc.).
// The CPUID instruction itself is always safe to execute; the SIMD intrinsics
// that it guards are not. Mixing the two in one TU would be circular.

#include "libhmm/platform/cpu_detection.h"

// ============================================================================
// Platform setup
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#define LIBHMM_CPU_X86
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define LIBHMM_CPU_ARM64
#endif

#ifdef LIBHMM_CPU_X86
#if defined(_MSC_VER)
#include <intrin.h> // __cpuidex, _xgetbv
#else
#include <cpuid.h> // __cpuid_count
#include <cstdint>
#endif
#endif

namespace libhmm::platform {

namespace {

// ============================================================================
// x86 CPUID helpers
// ============================================================================

#ifdef LIBHMM_CPU_X86

void run_cpuid(int leaf, int subleaf, int out[4]) noexcept {
#if defined(_MSC_VER)
    __cpuidex(out, leaf, subleaf);
#else
    __cpuid_count(leaf, subleaf, out[0], out[1], out[2], out[3]);
#endif
}

unsigned long long read_xcr0() noexcept {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned int lo = 0, hi = 0;
    // xgetbv: ECX=0 → EAX:EDX = XCR0. Requires OSXSAVE bit in ECX of CPUID(1).
    asm volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0U));
    return (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
}

// Does the OS + CPU support at least AVX (YMM register save)?
// Precondition: CPUID is available (leaf ≥ 1).
bool os_and_cpu_support_avx() noexcept {
    int info[4] = {};
    run_cpuid(1, 0, info);
    const bool osxsave = (info[2] >> 27) & 1; // ECX bit 27: OSXSAVE
    const bool avx_cpu = (info[2] >> 28) & 1; // ECX bit 28: AVX
    if (!osxsave || !avx_cpu)
        return false;
    // XCR0 bits 1+2 must be set: OS saves XMM (bit1) and YMM (bit2) state.
    const auto xcr0 = read_xcr0();
    return (xcr0 & 0x6ULL) == 0x6ULL;
}

#if !defined(__x86_64__) && !defined(_M_X64)
bool detect_sse2() noexcept {
    int info[4] = {};
    run_cpuid(1, 0, info);
    return (info[3] >> 26) & 1; // EDX bit 26: SSE2
}
#endif

bool detect_avx2() noexcept {
    if (!os_and_cpu_support_avx())
        return false;
    int info[4] = {};
    run_cpuid(7, 0, info);
    return (info[1] >> 5) & 1; // EBX bit 5: AVX2
}

bool detect_avx512() noexcept {
    if (!os_and_cpu_support_avx())
        return false;
    // OS must also save ZMM registers: XCR0 bits 5 (opmask), 6 (ZMMhi256), 7 (ZMMhi16).
    // Combined with YMM bits (1+2): mask = 0b11100110 = 0xE6.
    const auto xcr0 = read_xcr0();
    if ((xcr0 & 0xE6ULL) != 0xE6ULL)
        return false;
    int info[4] = {};
    run_cpuid(7, 0, info);
    return (info[1] >> 16) & 1; // EBX bit 16: AVX-512F
}

#endif // LIBHMM_CPU_X86

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

bool supports_sse2() noexcept {
    // SSE2 is mandated by the x86-64 ABI — always present on 64-bit x86.
    // Return true directly so static analysis tools can see the invariant.
#if defined(__x86_64__) || defined(_M_X64)
    return true;
#elif defined(LIBHMM_CPU_X86)
    static const bool v = detect_sse2();
    return v;
#else
    return false;
#endif
}

bool supports_avx2() noexcept {
#ifdef LIBHMM_CPU_X86
    static const bool v = detect_avx2();
    return v;
#else
    return false;
#endif
}

bool supports_avx512() noexcept {
#ifdef LIBHMM_CPU_X86
    static const bool v = detect_avx512();
    return v;
#else
    return false;
#endif
}

bool supports_neon() noexcept {
#ifdef LIBHMM_CPU_ARM64
    return true; // NEON is mandatory on AArch64
#else
    return false;
#endif
}

} // namespace libhmm::platform
