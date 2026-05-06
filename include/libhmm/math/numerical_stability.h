#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "libhmm/linalg/linalg_types.h"

namespace libhmm {
namespace numerical {

/// Numerical stability constants.
/// Aliases the shared constants from common.h for use within this module.
struct NumericalConstants {
    static constexpr double MIN_PROBABILITY =
        constants::probability::MIN_PROBABILITY;
    static constexpr double MAX_PROBABILITY =
        constants::probability::MAX_PROBABILITY;
    static constexpr double MIN_LOG_PROBABILITY =
        constants::probability::MIN_LOG_PROBABILITY;
    static constexpr double MAX_LOG_PROBABILITY =
        constants::probability::MAX_LOG_PROBABILITY;
    static constexpr double DEFAULT_CONVERGENCE_TOLERANCE =
        constants::precision::DEFAULT_CONVERGENCE_TOLERANCE;
    static constexpr std::size_t DEFAULT_MAX_ITERATIONS =
        constants::iterations::DEFAULT_MAX_ITERATIONS;
    static constexpr double NUMERICAL_EPSILON =
        constants::precision::HIGH_PRECISION_TOLERANCE;
    static constexpr double SCALING_THRESHOLD =
        constants::probability::SCALING_THRESHOLD;
};

/// Numerical safety utilities used by the V3 log-space architecture.
///
/// Only the three methods below are wired into the library. The broader
/// NumericalSafety / ConvergenceDetector / AdaptivePrecision / ErrorRecovery /
/// NumericalDiagnostics framework was removed in the V3 refactor when scaled
/// (non-log-space) calculators and trainers were dropped in favour of the
/// current log-space-only design.
class NumericalSafety {
public:
    /// Throws std::runtime_error if @p value is NaN or infinite.
    static void checkFinite(double value, const std::string &name = "value");

    /// Clamps @p prob to [MIN_PROBABILITY, MAX_PROBABILITY].
    /// Returns MIN_PROBABILITY for NaN input.
    static double clampProbability(double prob) noexcept;

    /// Normalises @p probs in place so they sum to 1.
    /// If all values are effectively zero, replaces with a uniform distribution.
    /// Returns false in that case, true on successful normalisation.
    // cppcheck-suppress constParameterReference
    static bool normalizeProbabilities(Vector &probs) noexcept;
};

} // namespace numerical
} // namespace libhmm
