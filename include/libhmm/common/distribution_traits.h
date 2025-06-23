#ifndef DISTRIBUTION_TRAITS_H_
#define DISTRIBUTION_TRAITS_H_

#include <type_traits>
#include <utility>

namespace libhmm {

// Forward declarations for distribution types
class DiscreteDistribution;
class GaussianDistribution;
class PoissonDistribution;
class GammaDistribution;
class ExponentialDistribution;
class LogNormalDistribution;
class ParetoDistribution;
class BetaDistribution;
class WeibullDistribution;
class UniformDistribution;
class BinomialDistribution;
class NegativeBinomialDistribution;
class StudentTDistribution;
class ChiSquaredDistribution;

/// Type trait to detect if a type is a discrete distribution
template<typename T>
struct is_discrete_distribution : std::false_type {};

// Specializations for discrete distributions
template<>
struct is_discrete_distribution<DiscreteDistribution> : std::true_type {};

template<>
struct is_discrete_distribution<PoissonDistribution> : std::true_type {};

template<>
struct is_discrete_distribution<BinomialDistribution> : std::true_type {};

template<>
struct is_discrete_distribution<NegativeBinomialDistribution> : std::true_type {};

/// Type trait to detect if a type is a continuous distribution
template<typename T>
struct is_continuous_distribution : std::false_type {};

// Specializations for continuous distributions
template<>
struct is_continuous_distribution<GaussianDistribution> : std::true_type {};

template<>
struct is_continuous_distribution<GammaDistribution> : std::true_type {};

template<>
struct is_continuous_distribution<ExponentialDistribution> : std::true_type {};

template<>
struct is_continuous_distribution<LogNormalDistribution> : std::true_type {};

template<>
struct is_continuous_distribution<ParetoDistribution> : std::true_type {};

template<>
struct is_continuous_distribution<BetaDistribution> : std::true_type {};

template<>
struct is_continuous_distribution<WeibullDistribution> : std::true_type {};

template<>
struct is_continuous_distribution<UniformDistribution> : std::true_type {};

template<>
struct is_continuous_distribution<StudentTDistribution> : std::true_type {};

template<>
struct is_continuous_distribution<ChiSquaredDistribution> : std::true_type {};

/// Type trait to detect if a type supports fitting
template<typename T, typename = void>
struct supports_fitting : std::false_type {};

template<typename T>
struct supports_fitting<T, std::void_t<decltype(std::declval<T>().fit(std::declval<std::vector<double>&>()))>>
    : std::true_type {};

/// Type trait to detect if a type has parameter estimation
template<typename T, typename = void>
struct has_parameter_estimation : std::false_type {};

template<typename T>
struct has_parameter_estimation<T, std::void_t<
    decltype(std::declval<T>().getMean()),
    decltype(std::declval<T>().getVariance())
>> : std::true_type {};

/// Type trait to check if a distribution is suitable for Baum-Welch training
template<typename T>
struct is_baum_welch_compatible : std::bool_constant<
    is_discrete_distribution<T>::value && supports_fitting<T>::value
> {};

/// Type trait to check if a distribution is suitable for Viterbi training
template<typename T>
struct is_viterbi_compatible : std::bool_constant<
    (is_discrete_distribution<T>::value || is_continuous_distribution<T>::value) && 
    supports_fitting<T>::value
> {};

/// C++17 variable templates for convenience
template<typename T>
inline constexpr bool is_discrete_distribution_v = is_discrete_distribution<T>::value;

template<typename T>
inline constexpr bool is_continuous_distribution_v = is_continuous_distribution<T>::value;

template<typename T>
inline constexpr bool supports_fitting_v = supports_fitting<T>::value;

template<typename T>
inline constexpr bool has_parameter_estimation_v = has_parameter_estimation<T>::value;

template<typename T>
inline constexpr bool is_baum_welch_compatible_v = is_baum_welch_compatible<T>::value;

template<typename T>
inline constexpr bool is_viterbi_compatible_v = is_viterbi_compatible<T>::value;

/// SFINAE helpers for enabling/disabling template specializations
template<typename T>
using enable_if_discrete_t = std::enable_if_t<is_discrete_distribution_v<T>>;

template<typename T>
using enable_if_continuous_t = std::enable_if_t<is_continuous_distribution_v<T>>;

template<typename T>
using enable_if_baum_welch_compatible_t = std::enable_if_t<is_baum_welch_compatible_v<T>>;

template<typename T>
using enable_if_viterbi_compatible_t = std::enable_if_t<is_viterbi_compatible_v<T>>;

/// Distribution category enum for runtime checks
enum class DistributionCategory {
    Discrete,
    Continuous,
    Unknown
};

/// Runtime function to get distribution category
template<typename T>
constexpr DistributionCategory get_distribution_category() {
    if constexpr (is_discrete_distribution_v<T>) {
        return DistributionCategory::Discrete;
    } else if constexpr (is_continuous_distribution_v<T>) {
        return DistributionCategory::Continuous;
    } else {
        return DistributionCategory::Unknown;
    }
}

/// Compile-time validation helpers
namespace detail {
    template<typename T>
    struct distribution_validator {
        static_assert(is_discrete_distribution_v<T> || is_continuous_distribution_v<T>,
                     "Type must be a recognized distribution type");
        static_assert(supports_fitting_v<T>,
                     "Distribution must support fitting operation");
    };
}

/// Macro for compile-time distribution validation
#define LIBHMM_VALIDATE_DISTRIBUTION(T) \
    static_assert(libhmm::is_discrete_distribution_v<T> || libhmm::is_continuous_distribution_v<T>, \
                  #T " must be a recognized distribution type"); \
    static_assert(libhmm::supports_fitting_v<T>, \
                  #T " must support fitting operation")

/// Factory trait to determine appropriate trainer for a distribution type
template<typename DistType>
struct trainer_selector {
    static_assert(is_discrete_distribution_v<DistType> || is_continuous_distribution_v<DistType>,
                  "Distribution type must be discrete or continuous");
                  
    using type = std::conditional_t<
        is_discrete_distribution_v<DistType>,
        std::integral_constant<int, 0>,  // Baum-Welch compatible
        std::integral_constant<int, 1>   // Viterbi compatible
    >;
    
    static constexpr int value = type::value;
};

template<typename DistType>
inline constexpr int trainer_selector_v = trainer_selector<DistType>::value;

} // namespace libhmm

#endif // DISTRIBUTION_TRAITS_H_
