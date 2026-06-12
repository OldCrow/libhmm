#pragma once

/**
 * @file distributions.h
 * @brief Convenience header that includes all libhmm probability distributions.
 *
 * Includes 16 scalar (Obs=double) distributions and 3 multivariate
 * (Obs=ObservationVectorView) distributions introduced in v4.
 *
 * Usage:
 * @code
 * #include "libhmm/distributions/distributions.h"
 *
 * // Scalar distributions (v3-compatible)
 * GaussianDistribution gauss(0.0, 1.0);
 * PoissonDistribution  poisson(2.5);
 *
 * // Multivariate distributions (v4)
 * DiagonalGaussianDistribution  diag(3);      // 3-dimensional diagonal Gaussian
 * FullCovarianceGaussianDistribution full(3); // 3-dimensional full-covariance Gaussian
 * @endcode
 *
 * @note For better compilation times, consider including only the specific
 *       distribution headers you need in performance-critical applications.
 */

// v4 parameterised interface and C++20 concepts (canonical)
#include "libhmm/distributions/basic_emission_distribution.h"
#include "libhmm/distributions/emission_concepts.h"

// v3 compatibility alias and concrete base
#include "libhmm/distributions/emission_distribution.h"
#include "libhmm/distributions/distribution_base.h"

// Distribution type traits (retained for backward compatibility; see emission_concepts.h)
#include "libhmm/distributions/distribution_traits.h"

// Discrete distributions
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/binomial_distribution.h"
#include "libhmm/distributions/negative_binomial_distribution.h"
#include "libhmm/distributions/poisson_distribution.h"

// Multivariate distributions (Phase G)
// Obs = ObservationVectorView = std::span<const double>
#include "libhmm/distributions/independent_components_distribution.h"
#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/distributions/full_covariance_gaussian_distribution.h"

// Continuous distributions
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/exponential_distribution.h"
#include "libhmm/distributions/gamma_distribution.h"
#include "libhmm/distributions/log_normal_distribution.h"
#include "libhmm/distributions/pareto_distribution.h"
#include "libhmm/distributions/beta_distribution.h"
#include "libhmm/distributions/uniform_distribution.h"
#include "libhmm/distributions/weibull_distribution.h"
#include "libhmm/distributions/rayleigh_distribution.h"
#include "libhmm/distributions/student_t_distribution.h"
#include "libhmm/distributions/chi_squared_distribution.h"
#include "libhmm/distributions/von_mises_distribution.h"

/**
 * @namespace libhmm
 * @brief All distributions are available in the libhmm namespace
 *
 * After including this header, all distribution classes are available:
 *
 * **Discrete Distributions (Obs=double):**
 * - DiscreteDistribution: General discrete distribution
 * - BinomialDistribution: Binomial distribution B(n,p)
 * - NegativeBinomialDistribution: Negative binomial distribution
 * - PoissonDistribution: Poisson distribution P(λ)
 *
 * **Continuous Distributions (Obs=double):**
 * - GaussianDistribution: Normal distribution N(μ,σ²)
 * - ExponentialDistribution: Exponential distribution Exp(λ)
 * - GammaDistribution: Gamma distribution Γ(k,θ)
 * - LogNormalDistribution: Log-normal distribution
 * - ParetoDistribution: Pareto distribution
 * - BetaDistribution: Beta distribution B(α,β)
 * - UniformDistribution: Uniform distribution U(a,b)
 * - WeibullDistribution: Weibull distribution
 * - StudentTDistribution: Student's t-distribution
 * - ChiSquaredDistribution: Chi-squared distribution χ²(k)
 * - VonMisesDistribution: Von Mises circular distribution
 * - RayleighDistribution: Rayleigh distribution
 *
 * **Multivariate Distributions (Obs=ObservationVectorView, v4):**
 * - IndependentComponentsDistribution: D independent scalar emission components
 * - DiagonalGaussianDistribution: Multivariate Gaussian with diagonal covariance
 * - FullCovarianceGaussianDistribution: Multivariate Gaussian with full covariance
 */

// Distribution count for compile-time verification
namespace libhmm {
namespace detail {
/// Total concrete distribution types (16 scalar + 3 multivariate)
inline constexpr std::size_t DISTRIBUTION_COUNT = 19;

/// Number of scalar discrete distribution types
inline constexpr std::size_t DISCRETE_DISTRIBUTION_COUNT = 4;

/// Number of scalar continuous distribution types
inline constexpr std::size_t CONTINUOUS_DISTRIBUTION_COUNT = 12;

/// Number of multivariate distribution types (Obs = ObservationVectorView, v4)
inline constexpr std::size_t MULTIVARIATE_DISTRIBUTION_COUNT = 3;

static_assert(DISCRETE_DISTRIBUTION_COUNT + CONTINUOUS_DISTRIBUTION_COUNT +
                      MULTIVARIATE_DISTRIBUTION_COUNT ==
                  DISTRIBUTION_COUNT,
              "Distribution counts must be consistent");
} // namespace detail
} // namespace libhmm
