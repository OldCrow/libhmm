#pragma once

#include <cstddef> // std::size_t

namespace libhmm {

//==============================================================================
// CONSOLIDATED CONSTANTS FOR LIBHMM
//
// All numerical constants used throughout the library, organized by category.
// Previously defined inline in common/common.h; extracted to this standalone
// header in Phase 1 refactor so components can include only what they need.
//==============================================================================

/// Core numerical precision constants
namespace constants {

/// Basic precision and tolerance values
namespace precision {
/// Minimum value considered non-zero throughout libhmm
inline constexpr double ZERO = 1.0e-30;

/// Default convergence tolerance for training algorithms
inline constexpr double DEFAULT_CONVERGENCE_TOLERANCE = 1.0e-8;

/// Legacy Baum-Welch tolerance for backward compatibility
inline constexpr double BW_TOLERANCE = 3.0e-7;

/// Tolerance for continuous distribution range validation
inline constexpr double LIMIT_TOLERANCE = 1.0e-6;

/// High precision tolerance for critical numerical operations
inline constexpr double HIGH_PRECISION_TOLERANCE = 1.0e-12;

/// Ultra-high precision for research and validation purposes
inline constexpr double ULTRA_HIGH_PRECISION_TOLERANCE = 1.0e-15;

/// Minimum standard deviation for distribution parameters
inline constexpr double MIN_STD_DEV = 1.0e-6;
} // namespace precision

/// Probability bounds and safety limits
namespace probability {
/// Minimum probability value to prevent underflow
inline constexpr double MIN_PROBABILITY = 1.0e-300;

/// Maximum probability value to prevent overflow
inline constexpr double MAX_PROBABILITY = 1.0 - 1.0e-15;

/// Minimum log probability to prevent -infinity (≈ log(MIN_PROBABILITY))
inline constexpr double MIN_LOG_PROBABILITY = -700.0;

/// Maximum log probability (log(1.0))
inline constexpr double MAX_LOG_PROBABILITY = 0.0;

/// Threshold for scaling operations to prevent underflow
inline constexpr double SCALING_THRESHOLD = 1.0e-100;

/// Log-space threshold for scaling (≈ log(1e-100))
inline constexpr double LOG_SCALING_THRESHOLD = -230.0;
} // namespace probability

/// Iteration limits for training algorithms
namespace iterations {
/// Maximum iterations for Viterbi training
inline constexpr std::size_t MAX_VITERBI_ITERATIONS = 500;

/// Maximum iterations for Baum-Welch training
inline constexpr std::size_t MAX_BAUM_WELCH_ITERATIONS = 1000;

/// Maximum iterations for gamma-related functions
inline constexpr std::size_t ITMAX = 10000;

/// Default maximum iterations for general numerical algorithms
inline constexpr std::size_t DEFAULT_MAX_ITERATIONS = 10000;

/// Maximum iterations for real-time applications
inline constexpr std::size_t REALTIME_MAX_ITERATIONS = 100;
} // namespace iterations

/// SIMD optimization parameters
namespace simd {
/// Default SIMD block size for vectorized operations
inline constexpr std::size_t DEFAULT_BLOCK_SIZE = 8;

/// Minimum problem size to benefit from SIMD
inline constexpr std::size_t MIN_SIMD_SIZE = 4;

/// Maximum block size for cache optimization
inline constexpr std::size_t MAX_BLOCK_SIZE = 64;

/// SIMD alignment requirement (bytes)
inline constexpr std::size_t SIMD_ALIGNMENT = 32;
} // namespace simd

/// Mathematical constants
namespace math {
// Provide M_PI for code that tests for it (non-standard but widely expected).
// Defined before PI so both are available in the same translation unit.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
inline constexpr double PI = 3.141592653589793238462643383279502884;
inline constexpr double LN2 = 0.6931471805599453094172321214581766;
inline constexpr double LN_10 = 2.302585092994046;
inline constexpr double LN_HALF = -0.6931471805599453;
inline constexpr double LOG10_E = 0.4342944819032518;
inline constexpr double E = 2.7182818284590452353602874713526625;
inline constexpr double SQRT_2PI = 2.5066282746310005024157652848110453;
inline constexpr double LN_2PI = 1.8378770664093454835606594728112353;
inline constexpr double SQRT_2 = 1.4142135623730950488016887242096981;
inline constexpr double SQRT_3 = 1.7320508075688772;
inline constexpr double SQRT_5 = 2.2360679774997897;
inline constexpr double SQRT_10 = 3.1622776601683795;
inline constexpr double HALF_LN_2PI = 0.9189385332046727417803297364056176;
inline constexpr double EULER_MASCHERONI = 0.5772156649015328606065120900824024;
inline constexpr double GOLDEN_RATIO = 1.6180339887498948482045868343656381;

inline constexpr double HALF = 0.5;
inline constexpr double QUARTER = 0.25;
inline constexpr double THREE_QUARTERS = 0.75;
inline constexpr double ONE_THIRD = 1.0 / 3.0;
inline constexpr double ONE_FIFTH = 0.2;
inline constexpr double ONE_SIXTH = 1.0 / 6.0;
inline constexpr double ONE_TENTH = 0.1;
inline constexpr double ONE_TWELFTH = 1.0 / 12.0;

inline constexpr double ZERO_DOUBLE = 0.0;
inline constexpr double ONE = 1.0;
inline constexpr double TWO = 2.0;
inline constexpr double THREE = 3.0;
inline constexpr double FOUR = 4.0;
inline constexpr double FIVE = 5.0;
inline constexpr double TEN = 10.0;
inline constexpr double HUNDRED = 100.0;
inline constexpr double THOUSAND = 1000.0;

/// Rayleigh distribution constants
inline constexpr double SQRT_PI_OVER_TWO = 1.2533141373155003;       // √(π/2)
inline constexpr double FOUR_MINUS_PI_OVER_TWO = 0.4292036732051033; // (4-π)/2
inline constexpr double SQRT_TWO_LN_TWO = 1.1774100225154747;        // √(2·ln2)

/// Derived constants (eliminate repeated runtime computation)
inline constexpr double INV_SQRT_2PI = 1.0 / SQRT_2PI;
inline constexpr double TWO_PI = 2.0 * PI;
inline constexpr double PI_OVER_2 = PI / 2.0;
inline constexpr double PI_OVER_4 = PI / 4.0;
inline constexpr double NEG_HALF_LN_2PI = -0.5 * LN_2PI;
} // namespace math

/// Algorithm-specific thresholds
namespace thresholds {
inline constexpr double MIN_SCALE_FACTOR = 1.0e-100;
inline constexpr double MAX_SCALE_FACTOR = 1.0e100;
inline constexpr double LOG_SPACE_THRESHOLD = 1.0e-50;
inline constexpr std::size_t CONVERGENCE_WINDOW = 5;
inline constexpr std::size_t MIN_CLUSTER_SIZE = 1;
inline constexpr double MIN_DEGREES_OF_FREEDOM = 0.1;
inline constexpr double MAX_DEGREES_OF_FREEDOM = 1000.0;
inline constexpr double MAX_DISTRIBUTION_PARAMETER = 1.0e6;
inline constexpr double MIN_DISTRIBUTION_PARAMETER = 1.0e-6;
} // namespace thresholds

} // namespace constants

/// Legacy constants — maintain original unqualified names for backward compatibility
inline constexpr double BW_TOLERANCE = constants::precision::BW_TOLERANCE;
inline constexpr double ZERO = constants::precision::ZERO;
inline constexpr double LIMIT_TOLERANCE = constants::precision::LIMIT_TOLERANCE;
inline constexpr std::size_t MAX_VITERBI_ITERATIONS = constants::iterations::MAX_VITERBI_ITERATIONS;
inline constexpr std::size_t ITMAX = constants::iterations::ITMAX;
inline constexpr double PI = constants::math::PI;

} // namespace libhmm
