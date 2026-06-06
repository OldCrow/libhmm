#pragma once

#include <concepts>
#include <cstddef>
#include <memory>
#include <span>
#include <string>

#include "libhmm/distributions/basic_emission_distribution.h"

namespace libhmm {

/**
 * @brief Concept satisfied by any emission distribution for observation type Obs.
 *
 * All concrete distributions that derive from DistributionBase<Derived, Obs>
 * satisfy this concept automatically. It can also be used to constrain
 * template parameters independently of the class hierarchy.
 *
 * @tparam D    Distribution type to check.
 * @tparam Obs  Observation type (default: double).
 */
template<typename D, typename Obs = double>
concept EmissionDistributionFor =
    requires(D& d, const D& cd,
             typename BasicEmissionDistribution<Obs>::obs_param_t x,
             std::span<const Obs> data,
             std::span<const double> weights,
             std::span<double> out) {
        { cd.getLogProbability(x) }        -> std::convertible_to<double>;
        { cd.getBatchLogProbabilities(data, out) };
        { d.fit(data) };
        { d.fit(data, weights) };
        { d.reset() } noexcept;
        { cd.isDiscrete() }                -> std::convertible_to<bool>;
        { cd.getNumParameters() }          -> std::convertible_to<std::size_t>;
        { cd.to_json() }                   -> std::convertible_to<std::string>;
        { cd.clone() }
            -> std::same_as<std::unique_ptr<BasicEmissionDistribution<Obs>>>;
    };

/// @brief Concept for scalar (Obs = double) emission distributions.
template<typename D>
concept ScalarEmission = EmissionDistributionFor<D, double>;

/// @brief Helper variable template: true if D satisfies ScalarEmission.
template<typename D>
inline constexpr bool is_scalar_emission_v = ScalarEmission<D>;

} // namespace libhmm
