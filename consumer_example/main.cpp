/// @file main.cpp
/// @brief Minimal consumer example for libhmm via find_package.
///
/// Verifies that an installed libhmm can be found, linked, and used
/// by an external project.

#include "libhmm/libhmm.h"
#include "libhmm/math/bessel.h"

#include <cmath>
#include <iostream>
#include <memory>

using libhmm::GaussianDistribution;
using libhmm::Hmm;

int main() {
    // Construct a 2-state HMM and give each state a Gaussian emission
    // distribution — the same pattern as examples/basic_hmm_example.cpp.
    Hmm hmm(2);
    hmm.setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.0));
    hmm.setDistribution(1, std::make_unique<GaussianDistribution>(5.0, 1.0));

    std::cout << "libhmm consumer example\n";
    std::cout << "=======================\n";
    std::cout << "States: " << hmm.getNumStatesModern() << "\n";

    // Query the emission distribution for state 0 at its own mean: PDF(0) for
    // N(0,1) = 1/sqrt(2*pi) ~= 0.3989.
    double pdf_at_zero = hmm.getDistribution(0).getProbability(0.0);
    double expected = 1.0 / std::sqrt(2.0 * M_PI);
    std::cout << "  State 0 PDF(0) = " << pdf_at_zero << "\n";

    if (std::abs(pdf_at_zero - expected) > 1e-10) {
        std::cerr << "Verification failed: PDF(0) = " << pdf_at_zero << ", expected " << expected
                  << "\n";
        return 1;
    }

    // The installed tree must carry libhmm/config.h, so that a consumer
    // compiling libhmm/math/bessel.h selects the SAME Bessel tier the library
    // was built with. It did not before issue #75: the tier came from a
    // PRIVATE compile definition that no consumer could see, so every consumer
    // silently got the 1.6e-7 A&S fallback plus an ODR violation against the
    // library's own inline definitions.
    //
    // Decide independently whether this compiler has the C++17 special math
    // functions, then require the installed config header to agree. A one-
    // sided check would pass on a broken install, so this is deliberately
    // two-sided.
#if defined(__cpp_lib_math_special_functions)
    constexpr bool compiler_has_it = true;
#else
    constexpr bool compiler_has_it = false;
#endif
#if defined(LIBHMM_HAS_CXX17_BESSEL)
    constexpr bool installed_says_tier1 = true;
#else
    constexpr bool installed_says_tier1 = false;
#endif

    std::cout << "  Bessel tier: " << (installed_says_tier1 ? "1 (std::cyl_bessel_i)" : "2 (A&S)")
              << "\n";

    if (compiler_has_it && !installed_says_tier1) {
        std::cerr << "Verification failed: this compiler provides std::cyl_bessel_i but the "
                     "installed libhmm/config.h did not define LIBHMM_HAS_CXX17_BESSEL. The "
                     "installed tree is selecting a different Bessel tier than the library was "
                     "built with (issue #75).\n";
        return 1;
    }

    // I0(1) = 1.2660658777520084 (mpmath, dps 50). Tier 1 must be far tighter
    // than the A&S fallback's documented 1.6e-7.
    const double i0_at_one = libhmm::detail::bessel_i0(1.0);
    const double i0_tolerance = installed_says_tier1 ? 1e-14 : 1.6e-7;
    if (std::abs(i0_at_one - 1.2660658777520084) > i0_tolerance) {
        std::cerr << "Verification failed: bessel_i0(1) = " << i0_at_one << ", outside the "
                  << i0_tolerance << " tolerance for the selected tier\n";
        return 1;
    }

    std::cout << "\nVerification passed.\n";
    return 0;
}
