/**
 * simd_inspection — libhmm SIMD capability report and smoke test.
 *
 * Reports:
 *   - Processor architecture (compile-time)
 *   - SIMD ISA macros visible to this TU (compiled without SIMD flags)
 *   - Whether distribution TUs use a higher ISA (see cmake output for
 *     LIBHMM_BEST_SIMD_FLAGS — the actual ISA baked into getBatchLogProbabilities)
 *   - Functional smoke test of GaussianDistribution::getBatchLogProbabilities
 *     and ForwardBackwardCalculator (both exercise the SIMD paths indirectly)
 */
#include "libhmm/hmm.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/platform/simd_platform.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

using namespace libhmm;

static std::unique_ptr<Hmm> make_test_hmm(int N) {
    auto hmm = std::make_unique<Hmm>(N);
    Matrix trans(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            trans(i, j) = (i == j) ? 0.9 : 0.1 / (N - 1);
    }
    hmm->setTrans(trans);
    Vector pi(N);
    for (int i = 0; i < N; ++i)
        pi(i) = 1.0 / N;
    hmm->setPi(pi);
    for (int i = 0; i < N; ++i)
        hmm->setDistribution(i, std::make_unique<GaussianDistribution>(i * 3.0, 1.0));
    return hmm;
}

int main() {
    std::cout << "libhmm SIMD Inspection\n";
    std::cout << "======================\n\n";

    // -------------------------------------------------------------------------
    // Architecture
    // -------------------------------------------------------------------------
    std::cout << "Architecture: ";
#if defined(__aarch64__) || defined(_M_ARM64)
    std::cout << "AArch64 (ARM — NEON always present)\n";
#elif defined(__x86_64__) || defined(_M_X64)
    std::cout << "x86_64\n";
#elif defined(__i386__) || defined(_M_IX86)
    std::cout << "x86 (32-bit)\n";
#else
    std::cout << "Unknown\n";
#endif

    // -------------------------------------------------------------------------
    // SIMD macros visible in this TU
    // NOTE: tools/ is compiled without LIBHMM_BEST_SIMD_FLAGS so only the
    // baseline ISA is visible here. The distribution TUs (in LIBHMM_SIMD_SOURCES)
    // are compiled with the best available ISA reported at cmake configuration.
    // -------------------------------------------------------------------------
    std::cout << "\nISA macros (compiled with LIBHMM_BEST_SIMD_FLAGS —\n";
    std::cout << "  same flags used by distribution getBatchLogProbabilities TUs):\n";
#ifdef LIBHMM_HAS_AVX512
    std::cout << "  LIBHMM_HAS_AVX512 : YES\n";
#else
    std::cout << "  LIBHMM_HAS_AVX512 : -\n";
#endif
#ifdef LIBHMM_HAS_AVX2
    std::cout << "  LIBHMM_HAS_AVX2   : YES\n";
#else
    std::cout << "  LIBHMM_HAS_AVX2   : -\n";
#endif
#ifdef LIBHMM_HAS_AVX
    std::cout << "  LIBHMM_HAS_AVX    : YES\n";
#else
    std::cout << "  LIBHMM_HAS_AVX    : -\n";
#endif
#ifdef LIBHMM_HAS_SSE2
    std::cout << "  LIBHMM_HAS_SSE2   : YES (x86_64 baseline)\n";
#else
    std::cout << "  LIBHMM_HAS_SSE2   : -\n";
#endif
#ifdef LIBHMM_HAS_NEON
    std::cout << "  LIBHMM_HAS_NEON   : YES (AArch64 baseline)\n";
#else
    std::cout << "  LIBHMM_HAS_NEON   : -\n";
#endif
    std::cout << "  Vector width (double) = " << performance::simd::double_vector_width()
              << " lanes\n";

    // -------------------------------------------------------------------------
    // Functional smoke tests
    // -------------------------------------------------------------------------
    std::cout << "\nFunctional smoke tests:\n";
    int pass = 0, fail = 0;
    auto check = [&](const char *name, bool ok) {
        std::cout << "  " << std::left << std::setw(52) << name << (ok ? "PASS" : "FAIL") << "\n";
        if (ok)
            ++pass;
        else
            ++fail;
    };

    // 1. GaussianDistribution::getBatchLogProbabilities — 8 obs (SIMD main loop)
    {
        GaussianDistribution dist(0.0, 1.0);
        std::vector<double> obs = {-1.0, -0.5, 0.0, 0.3, 0.5, 0.8, 1.0, 1.5};
        std::vector<double> out(8);
        dist.getBatchLogProbabilities(std::span<const double>(obs.data(), obs.size()),
                                      std::span<double>(out.data(), out.size()));
        bool ok = true;
        for (double v : out)
            if (!std::isfinite(v)) {
                ok = false;
                break;
            }
        check("getBatchLogProbabilities: 8 obs (SIMD main loop)", ok);
    }

    // 2. Scalar tail — 9 obs (not a multiple of 8/4/2)
    {
        GaussianDistribution dist(2.0, 0.5);
        std::vector<double> obs = {1.5, 1.8, 2.0, 2.1, 2.3, 2.5, 1.9, 2.2, 2.4};
        std::vector<double> out(9);
        dist.getBatchLogProbabilities(std::span<const double>(obs.data(), obs.size()),
                                      std::span<double>(out.data(), out.size()));
        bool ok = true;
        for (double v : out)
            if (!std::isfinite(v)) {
                ok = false;
                break;
            }
        check("getBatchLogProbabilities: 9 obs (tests scalar tail)", ok);
    }

    // 3. NaN input → -Inf output (NaN masking)
    {
        GaussianDistribution dist(0.0, 1.0);
        std::vector<double> obs = {0.0, std::numeric_limits<double>::quiet_NaN(), 1.0};
        std::vector<double> out(3);
        dist.getBatchLogProbabilities(std::span<const double>(obs.data(), obs.size()),
                                      std::span<double>(out.data(), out.size()));
        bool ok = std::isfinite(out[0]) && out[1] == -std::numeric_limits<double>::infinity() &&
                  std::isfinite(out[2]);
        check("getBatchLogProbabilities: NaN input → -Inf", ok);
    }

    // 4. ForwardBackwardCalculator — 2-state, 100 obs
    {
        auto hmm = make_test_hmm(2);
        ObservationSet obs(100);
        for (std::size_t t = 0; t < 100; ++t)
            obs(t) = (t < 50) ? 0.0 : 3.0;
        ForwardBackwardCalculator fbc(*hmm, obs);
        check("ForwardBackwardCalculator: 2 states, 100 obs",
              std::isfinite(fbc.getLogProbability()));
    }

    // 5. ViterbiCalculator — 2-state, 100 obs
    {
        auto hmm = make_test_hmm(2);
        ObservationSet obs(100);
        for (std::size_t t = 0; t < 100; ++t)
            obs(t) = (t < 50) ? 0.0 : 3.0;
        ViterbiCalculator vc(*hmm, obs);
        auto seq = vc.decode();
        check("ViterbiCalculator: 2 states, 100 obs", seq.size() == 100u);
    }

    // 6. Large Gaussian HMM — exercises SIMD with many states
    {
        auto hmm = make_test_hmm(16);
        ObservationSet obs(200);
        for (std::size_t t = 0; t < 200; ++t)
            obs(t) = static_cast<double>(t % 16) * 3.0;
        ForwardBackwardCalculator fbc(*hmm, obs);
        check("ForwardBackwardCalculator: 16 states, 200 obs",
              std::isfinite(fbc.getLogProbability()));
    }

    std::cout << "\nResult: " << pass << " passed, " << fail << " failed\n";
    return fail > 0 ? 1 : 0;
}
