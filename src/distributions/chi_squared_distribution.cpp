#include "libhmm/distributions/chi_squared_distribution.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/math/psi_functions.h"
#include "libhmm/math/weighted_stats.h"
#include "libhmm/performance/simd_double_ops.h" // runtime dispatch
#include <algorithm>
#include <span>

using namespace libhmm::constants;

namespace libhmm {

double ChiSquaredDistribution::getProbability(double value) const {
    ensureCache();
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
    ensureCache();
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

double ChiSquaredDistribution::sample(std::mt19937_64 &rng) const {
    std::chi_squared_distribution<double> dist(degrees_of_freedom_);
    return dist(rng);
}

namespace {

/// MLE for Chi-squared(k) via Newton–Raphson on the score equation:
///   ψ(γ) = s,  γ = k/2,  s = mean_log_x − log(2)
/// (follows directly from χ²(k) = Gamma(k/2, 2))
/// Starting estimate: γ₀ = mean_x/2  (MOM seed).
/// Returns unclamped k = 2·γ; caller applies distribution bounds.
[[nodiscard]] double chi_squared_mle_solve(double mean_x, double mean_log_x) noexcept {
    const double s = mean_log_x - std::log(2.0);
    double gamma = std::max(mean_x * 0.5, 0.5); // MOM seed for γ = k/2
    for (int i = 0; i < 20; ++i) {
        const double f = detail::digamma(gamma) - s;
        const double fp = detail::trigamma(gamma);
        if (std::fabs(fp) < 1e-15)
            break;
        const double dg = f / fp;
        gamma -= dg;
        if (gamma <= 0.0)
            gamma = 1e-10;
        if (std::fabs(dg) < 1e-10 * gamma)
            break;
    }
    return 2.0 * gamma;
}

} // anonymous namespace

void ChiSquaredDistribution::fit(std::span<const double> data) {
    if (data.empty())
        throw std::invalid_argument("Cannot fit distribution to empty data");
    double sum = 0.0, sum_log = 0.0;
    std::size_t count = 0;
    for (const double v : data) {
        if (!std::isfinite(v) || v < 0.0)
            throw std::invalid_argument(
                "Chi-squared distribution requires non-negative finite values");
        if (v > 0.0) {
            sum += v;
            sum_log += std::log(v);
            ++count;
        }
    }
    if (count == 0) {
        reset();
        return;
    }
    const double n = static_cast<double>(count);
    const double k = chi_squared_mle_solve(sum / n, sum_log / n);
    setDegreesOfFreedom(std::clamp(k, MIN_DEGREES_OF_FREEDOM, MAX_DEGREES_OF_FREEDOM));
}

void ChiSquaredDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    double sumW = 0.0, sum_wx = 0.0, sum_wlogx = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        const double v = data[i];
        const double w = weights[i];
        if (!std::isfinite(v) || v <= 0.0 || !std::isfinite(w) || w <= 0.0)
            continue;
        sumW += w;
        sum_wx += w * v;
        sum_wlogx += w * std::log(v);
    }
    // Guard: keep current parameters when effective weight is near zero.
    if (sumW < precision::ZERO || std::isnan(sumW))
        return;
    const double k = chi_squared_mle_solve(sum_wx / sumW, sum_wlogx / sumW);
    setDegreesOfFreedom(std::clamp(k, MIN_DEGREES_OF_FREEDOM, MAX_DEGREES_OF_FREEDOM));
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
    ensureCache();
    performance::get_double_vec_ops().chisq_batch(observations.data(), out.data(),
                                                  observations.size(), cached_half_k_minus_one_,
                                                  cached_log_normalization_);
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
