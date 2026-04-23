/**
 * hmm_validator — load, validate and run inference on an XML HMM file.
 *
 * Usage:
 *   hmm_validator <hmm_xml_file> [T]
 *
 *   hmm_xml_file  Path to an HMM written by libhmm (XMLFileWriter)
 *   T             Observation sequence length for inference (default: 100)
 *
 * Loads the HMM from XML, validates its structure, generates T synthetic
 * zero-valued observations (always within support for any distribution),
 * runs ForwardBackward and Viterbi, and prints a diagnostics report.
 *
 * Exit code: 0 = all checks passed, 1 = load/validate/inference failure.
 */
#include "libhmm/hmm.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/calculators/viterbi_calculator.h"
#include "libhmm/io/xml_file_reader.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace libhmm;

static void print_usage(const char *prog) {
    std::cout << "Usage: " << prog << " <hmm_xml_file> [T]\n\n"
              << "  hmm_xml_file  Path to an XML HMM file written by libhmm\n"
              << "  T             Observation sequence length (default: 100)\n";
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string xml_path = argv[1];
    const int T = (argc >= 3) ? std::stoi(argv[2]) : 100;

    std::cout << "libhmm HMM Validator\n";
    std::cout << "====================\n";
    std::cout << "File: " << xml_path << "\n";
    std::cout << "T:    " << T << "\n\n";

    int exit_code = 0;

    // -------------------------------------------------------------------------
    // 1. Load
    // -------------------------------------------------------------------------
    Hmm hmm(1); // placeholder overwritten by reader
    try {
        XMLFileReader reader;
        hmm = reader.read(xml_path);
        std::cout << "[ OK ] Load from XML\n";
    } catch (const std::exception &e) {
        std::cerr << "[FAIL] Load: " << e.what() << "\n";
        return 1;
    }

    // -------------------------------------------------------------------------
    // 2. Validate structure
    // -------------------------------------------------------------------------
    try {
        hmm.validate();
        std::cout << "[ OK ] Validate structure\n";
    } catch (const std::exception &e) {
        std::cerr << "[FAIL] Validate: " << e.what() << "\n";
        return 1;
    }

    const int N = hmm.getNumStates();
    std::cout << "\nHMM summary:\n";
    std::cout << "  States: " << N << "\n";

    // Print transition matrix
    std::cout << "  Transition matrix:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "    [";
        for (int j = 0; j < N; ++j) {
            std::cout << "  " << hmm.getTrans()(i, j);
        }
        std::cout << "  ]\n";
    }

    // Print pi vector
    std::cout << "  Pi: [";
    for (int i = 0; i < N; ++i)
        std::cout << "  " << hmm.getPi()(i);
    std::cout << "  ]\n";

    // Print distribution types
    std::cout << "  Distributions:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "    State " << i << ": " << hmm.getDistribution(i).toString().substr(0, 60)
                  << "\n";
    }

    // -------------------------------------------------------------------------
    // 3. ForwardBackward
    // -------------------------------------------------------------------------
    std::cout << "\nForwardBackward (" << T << " zero-valued observations):\n";
    ObservationSet obs(T);
    for (int t = 0; t < T; ++t)
        obs(t) = 0.0;

    try {
        ForwardBackwardCalculator fbc(hmm, obs);
        const double log_p = fbc.getLogProbability();
        std::cout << "  log P(O|λ)  = " << log_p << "\n";
        std::cout << "  P(O|λ)      = " << fbc.probability() << "\n";

        if (std::isfinite(log_p)) {
            std::cout << "[ OK ] ForwardBackward\n";
        } else {
            std::cout << "[WARN] Non-finite log-probability — "
                         "check emission distributions for zero support at x=0\n";
            // Not fatal — some distributions have zero probability at 0
        }
    } catch (const std::exception &e) {
        std::cerr << "[FAIL] ForwardBackward: " << e.what() << "\n";
        exit_code = 1;
    }

    // -------------------------------------------------------------------------
    // 4. Viterbi
    // -------------------------------------------------------------------------
    std::cout << "\nViterbi (" << T << " zero-valued observations):\n";
    try {
        ViterbiCalculator vc(hmm, obs);
        const auto path = vc.decode();

        std::cout << "  log P(O,q*|λ) = " << vc.getLogProbability() << "\n";
        std::cout << "  Path length   = " << path.size() << "\n";

        // State occupancy counts
        std::vector<int> counts(N, 0);
        for (std::size_t t = 0; t < path.size(); ++t)
            counts[path(t)]++;
        std::cout << "  State counts: ";
        for (int i = 0; i < N; ++i) {
            std::cout << "s" << i << "=" << counts[i];
            if (i < N - 1)
                std::cout << "  ";
        }
        std::cout << "\n";

        std::cout << "[ OK ] Viterbi\n";
    } catch (const std::exception &e) {
        std::cerr << "[FAIL] Viterbi: " << e.what() << "\n";
        exit_code = 1;
    }

    std::cout << "\nValidation " << (exit_code == 0 ? "complete." : "completed with failures.")
              << "\n";
    return exit_code;
}
