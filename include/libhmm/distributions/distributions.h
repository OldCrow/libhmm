#ifndef LIBHMM_DISTRIBUTIONS_H_
#define LIBHMM_DISTRIBUTIONS_H_

/**
 * @file distributions.h
 * @brief Convenience header that includes all libhmm probability distributions
 * 
 * This header provides a single include point for all probability distributions
 * available in libhmm. It follows the standard library convention of providing
 * umbrella headers for related functionality.
 * 
 * Usage:
 * @code
 * #include "libhmm/distributions/distributions.h"
 * 
 * // All distributions are now available:
 * GaussianDistribution gauss(0.0, 1.0);
 * PoissonDistribution poisson(2.5);
 * DiscreteDistribution discrete(6);
 * @endcode
 * 
 * @note For better compilation times, consider including only the specific
 *       distribution headers you need in performance-critical applications.
 */

// Base distribution interface
#include "libhmm/distributions/probability_distribution.h"

// Discrete distributions
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/binomial_distribution.h"
#include "libhmm/distributions/negative_binomial_distribution.h"
#include "libhmm/distributions/poisson_distribution.h"

// Continuous distributions
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/exponential_distribution.h"
#include "libhmm/distributions/gamma_distribution.h"
#include "libhmm/distributions/log_normal_distribution.h"
#include "libhmm/distributions/pareto_distribution.h"
#include "libhmm/distributions/beta_distribution.h"
#include "libhmm/distributions/uniform_distribution.h"
#include "libhmm/distributions/weibull_distribution.h"
#include "libhmm/distributions/student_t_distribution.h"
#include "libhmm/distributions/chi_squared_distribution.h"

/**
 * @namespace libhmm
 * @brief All distributions are available in the libhmm namespace
 * 
 * After including this header, all distribution classes are available:
 * 
 * **Discrete Distributions:**
 * - DiscreteDistribution: General discrete distribution
 * - BinomialDistribution: Binomial distribution B(n,p)
 * - NegativeBinomialDistribution: Negative binomial distribution
 * - PoissonDistribution: Poisson distribution P(λ)
 * 
 * **Continuous Distributions:**
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
 */

// Distribution count for compile-time verification
namespace libhmm {
    namespace detail {
        /// Total number of concrete distribution types (excluding base class)
        inline constexpr std::size_t DISTRIBUTION_COUNT = 14;
        
        /// Number of discrete distribution types
        inline constexpr std::size_t DISCRETE_DISTRIBUTION_COUNT = 4;
        
        /// Number of continuous distribution types
        inline constexpr std::size_t CONTINUOUS_DISTRIBUTION_COUNT = 10;
        
        static_assert(DISCRETE_DISTRIBUTION_COUNT + CONTINUOUS_DISTRIBUTION_COUNT == DISTRIBUTION_COUNT,
                     "Distribution counts must be consistent");
    }
}

#endif // LIBHMM_DISTRIBUTIONS_H_
