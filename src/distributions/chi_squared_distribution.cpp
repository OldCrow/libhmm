#include "libhmm/distributions/chi_squared_distribution.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/math/weighted_stats.h"
#include <algorithm>
#include <span>

using namespace libhmm::constants;

namespace libhmm {

double ChiSquaredDistribution::getProbability(double value) const {
    if (!isCacheValid())
        updateCache();
    const double x = value;
    if (!std::isfinite(x))
        return math::ZERO_DOUBLE;

    // Return 0 for negative values (outside support)
    if (x < math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }

    // Handle special case for x = 0
    if (x == math::ZERO_DOUBLE) {
        if (degrees_of_freedom_ < math::TWO) {
            return std::numeric_limits<double>::infinity();
        } else if (degrees_of_freedom_ == math::TWO) {
            return std::exp(cached_log_normalization_); // exp(-log(2)) = 0.5
        } else {
            return math::ZERO_DOUBLE;
        }
    }

    // Log PDF: log(f(x|k)) = log_normalization + (k/2-1)*log(x) - x/2
    double log_prob =
        cached_log_normalization_ + cached_half_k_minus_one_ * std::log(x) - math::HALF * x;

    return std::exp(log_prob);
}

double ChiSquaredDistribution::getLogProbability(double value) const noexcept {
    if (!isCacheValid())
        updateCache();
    const double x = value;

    // Handle invalid inputs
    if (!std::isfinite(x)) {
        return -std::numeric_limits<double>::infinity();
    }

    // Return -∞ for negative values (outside support)
    if (x < math::ZERO_DOUBLE) {
        return -std::numeric_limits<double>::infinity();
    }

    // Handle special case for x = 0
    if (x == math::ZERO_DOUBLE) {
        if (degrees_of_freedom_ < math::TWO) {
            return std::numeric_limits<double>::infinity();
        } else if (degrees_of_freedom_ == math::TWO) {
            return cached_log_normalization_; // log(0.5) = -log(2)
        } else {
            return -std::numeric_limits<double>::infinity();
        }
    }

    // Log PDF: log(f(x|k)) = log_normalization + (k/2-1)*log(x) - x/2
    return cached_log_normalization_ + cached_half_k_minus_one_ * std::log(x) - math::HALF * x;
}

double ChiSquaredDistribution::getCumulativeProbability(double x) const noexcept {
    // Handle invalid inputs
    if (std::isnan(x)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Return 0 for negative values (outside support)
    if (x <= math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }

    // For positive x, use the relationship to incomplete gamma function:
    // CDF(x) = γ(k/2, x/2) / Γ(k/2) = P(k/2, x/2)
    // where γ is the lower incomplete gamma function and P is the regularized gamma function
    double half_k = math::HALF * degrees_of_freedom_;
    double half_x = math::HALF * x;

    // Use the gamma probability function from the base class
    return gammap(half_k, half_x);
}

void ChiSquaredDistribution::fit(std::span<const double> data) {
    if (data.empty())
        throw std::invalid_argument("Cannot fit distribution to empty data");
    double sum = 0.0;
    for (const double v : data) {
        if (!std::isfinite(v) || v < 0.0)
            throw std::invalid_argument(
                "Chi-squared distribution requires non-negative finite values");
        sum += v;
    }
    double est = std::max(MIN_DEGREES_OF_FREEDOM,
                          std::min(MAX_DEGREES_OF_FREEDOM, sum / static_cast<double>(data.size())));
    setDegreesOfFreedom(est);
}

void ChiSquaredDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    const auto mean = detail::compute_weighted_mean(data, weights);
    // Guard: near-zero weight → keep current parameters (not reset).
    if (!mean)
        return;
    double est = std::max(MIN_DEGREES_OF_FREEDOM, std::min(MAX_DEGREES_OF_FREEDOM, *mean));
    setDegreesOfFreedom(est);
}

void ChiSquaredDistribution::reset() noexcept {
    degrees_of_freedom_ = math::ONE;
    invalidateCache();
}

std::string ChiSquaredDistribution::toString() const {
    std::ostringstream oss{};
    oss << "ChiSquared Distribution:\n";
    oss << "  k (degrees of freedom) = " << std::fixed << std::setprecision(6)
        << degrees_of_freedom_;
    return oss.str();
}

bool ChiSquaredDistribution::operator==(const ChiSquaredDistribution &other) const {
    return std::abs(degrees_of_freedom_ - other.degrees_of_freedom_) < PARAMETER_TOLERANCE;
}

std::ostream &operator<<(std::ostream &os, const ChiSquaredDistribution &dist) {
    os << dist.toString();
    return os;
}

// Parses the format produced by toString() / operator<<:
//   ChiSquared Distribution:
//     k (degrees of freedom) = VALUE
std::istream &operator>>(std::istream &is, ChiSquaredDistribution &dist) {
    try {
        std::string s, t;
        is >> s >> s;                     // "ChiSquared" "Distribution:"
        is >> s >> s >> s >> s >> s >> t; // "k" "(degrees" "of" "freedom)" "=" VALUE
        if (is.good())
            dist.setDegreesOfFreedom(std::stod(t));
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

void ChiSquaredDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                      std::span<double> out) const {
    // Tier 1 — concrete non-virtual loop; compiler auto-vectorizes the arithmetic
    // terms under -march=native. Index loop preserved: a std::ranges::transform
    // lambda would add an indirect call boundary that inhibits auto-vectorisation.
    // Tier 2 upgrade requires vectorised lgamma (the log-normalisation constant
    // lgamma(k/2) is precomputed in the cache, but the per-element (k/2-1)*log(x)
    // term needs vectorised log(x)): available via Intel SVML or platform-specific
    // math libraries, but not portably available without a math-library dependency.
    if (!isCacheValid())
        updateCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        out[i] = ChiSquaredDistribution::getLogProbability(observations[i]);
    }
}

std::string ChiSquaredDistribution::to_json() const {
    return json::write_distribution("ChiSquared", {{"k", degrees_of_freedom_}});
}
std::unique_ptr<EmissionDistribution> ChiSquaredDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double k = r.read_double();
    r.consume('}');
    return std::make_unique<ChiSquaredDistribution>(k);
}

} // namespace libhmm
