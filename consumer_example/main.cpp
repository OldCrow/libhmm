/// @file main.cpp
/// @brief Minimal consumer example for libhmm via find_package.
///
/// Verifies that an installed libhmm can be found, linked, and used
/// by an external project.

#include "libhmm/libhmm.h"

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

    std::cout << "\nVerification passed.\n";
    return 0;
}
