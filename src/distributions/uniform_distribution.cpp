#include "libhmm/distributions/uniform_distribution.h"
#include "libhmm/math/weighted_stats.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

using namespace libhmm::constants;

namespace libhmm {

void UniformDistribution::validateParameters(double a, double b) {
    // Check for NaN or infinity
    if (std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b)) {
        throw std::invalid_argument("Uniform distribution parameters cannot be NaN or infinite");
    }

    // Check that a < b
    if (a >= b) {
        throw std::invalid_argument(
            "Uniform distribution requires a < b (lower bound must be less than upper bound)");
    }
}

UniformDistribution::UniformDistribution() : a_(math::ZERO_DOUBLE), b_(math::ONE) {}

UniformDistribution::UniformDistribution(double a, double b) : a_(a), b_(b) {
    validateParameters(a, b);
}

void UniformDistribution::updateCache() const noexcept {
    cached_range_ = b_ - a_;
    cached_inv_range_ = math::ONE / cached_range_;
    cached_pdf_ = cached_inv_range_;
    cached_log_pdf_ = -std::log(cached_range_);
    cached_mean_ = (a_ + b_) * math::HALF;
    cached_variance_ = (cached_range_ * cached_range_) / (math::THREE * math::FOUR);
    cached_std_dev_ = cached_range_ / std::sqrt(math::THREE * math::FOUR);
    markCacheValid();
}

double UniformDistribution::getProbability(double val) const {
    if (std::isnan(val) || std::isinf(val))
        return math::ZERO_DOUBLE;
    if (val >= a_ && val <= b_) {
        if (!isCacheValid())
            updateCache();
        return cached_pdf_;
    }
    return math::ZERO_DOUBLE;
}

double UniformDistribution::getLogProbability(double val) const noexcept {
    if (std::isnan(val) || std::isinf(val))
        return -std::numeric_limits<double>::infinity();
    if (val >= a_ && val <= b_) {
        if (!isCacheValid())
            updateCache();
        return cached_log_pdf_;
    }
    return -std::numeric_limits<double>::infinity();
}

double UniformDistribution::CDF(double x) const {
    // Handle invalid inputs
    if (std::isnan(x) || std::isinf(x)) {
        return std::isnan(x) ? std::numeric_limits<double>::quiet_NaN()
                             : (x > math::ZERO_DOUBLE ? math::ONE : math::ZERO_DOUBLE);
    }

    // Uniform CDF: F(x) = 0 for x < a, (x-a)/(b-a) for a ≤ x ≤ b, 1 for x > b
    if (x < a_) {
        return math::ZERO_DOUBLE;
    } else if (x > b_) {
        return math::ONE;
    } else {
        return (x - a_) / (b_ - a_);
    }
}

void UniformDistribution::fit(std::span<const double> data) {
    if (data.size() < 2) {
        reset();
        return;
    }

    double minVal = std::numeric_limits<double>::max();
    double maxVal = std::numeric_limits<double>::lowest();
    for (const double x : data) {
        if (std::isnan(x) || std::isinf(x))
            throw std::invalid_argument(
                "Uniform distribution fit: data contains NaN or infinite values");
        if (x < minVal)
            minVal = x;
        if (x > maxVal)
            maxVal = x;
    }

    const double range = maxVal - minVal;
    if (range == 0.0) {
        const double pad = std::max(std::abs(minVal) * thresholds::MIN_DISTRIBUTION_PARAMETER,
                                    thresholds::MIN_DISTRIBUTION_PARAMETER);
        a_ = minVal - pad;
        b_ = maxVal + pad;
    } else {
        const double pad = range * 0.05;
        a_ = minVal - pad;
        b_ = maxVal + pad;
    }
    validateParameters(a_, b_);
    invalidateCache();
}

void UniformDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    // Method-of-moments weighted fit.
    // For Uniform(a,b): mean = (a+b)/2, var = (b-a)²/12.
    // Solve: half_range = √(3*var), a = mean - half_range, b = mean + half_range.
    const auto stats = detail::compute_weighted_stats(data, weights);
    if (!stats) {
        reset();
        return;
    }
    const double halfRange = std::sqrt(3.0 * stats->variance);
    if (halfRange < thresholds::MIN_DISTRIBUTION_PARAMETER || !std::isfinite(halfRange)) {
        reset();
        return;
    }
    a_ = stats->mean - halfRange;
    b_ = stats->mean + halfRange;
    invalidateCache();
}

void UniformDistribution::reset() noexcept {
    a_ = math::ZERO_DOUBLE;
    b_ = math::ONE;
    invalidateCache();
}

std::string UniformDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Uniform Distribution:\n";
    oss << "      a (lower bound) = " << a_ << "\n";
    oss << "      b (upper bound) = " << b_ << "\n";
    return oss.str();
}

void UniformDistribution::setA(double a) {
    validateParameters(a, b_);
    a_ = a;
    invalidateCache();
}

void UniformDistribution::setB(double b) {
    validateParameters(a_, b);
    b_ = b;
    invalidateCache();
}

void UniformDistribution::setParameters(double a, double b) {
    validateParameters(a, b);
    a_ = a;
    b_ = b;
    invalidateCache();
}

double UniformDistribution::getMean() const {
    if (!isCacheValid())
        updateCache();
    return cached_mean_;
}

double UniformDistribution::getVariance() const {
    if (!isCacheValid())
        updateCache();
    return cached_variance_;
}

double UniformDistribution::getStandardDeviation() const {
    if (!isCacheValid())
        updateCache();
    return cached_std_dev_;
}

bool UniformDistribution::isApproximatelyEqual(const UniformDistribution &other,
                                               double tolerance) const {
    return std::abs(a_ - other.a_) < tolerance && std::abs(b_ - other.b_) < tolerance;
}

bool UniformDistribution::operator==(const UniformDistribution &other) const {
    return isApproximatelyEqual(other, precision::LIMIT_TOLERANCE);
}

std::ostream &operator<<(std::ostream &os, const UniformDistribution &dist) {
    os << std::fixed << std::setprecision(6);
    os << "Uniform Distribution: a = " << dist.getA() << ", b = " << dist.getB();
    return os;
}

std::istream &operator>>(std::istream &is, UniformDistribution &dist) {
    try {
        std::string token;
        double a = 0.0, b = 0.0;
        // Expected format: "Uniform Distribution: a = <value>, b = <value>"
        std::string a_str, b_str;
        is >> token >> token >> token >> token >> a_str >> token >> token >> token >>
            b_str; // "Uniform" "Distribution:" "a" "=" <a_str> "," "b" "=" <b_str>
        a = std::stod(a_str);
        b = std::stod(b_str);

        if (is.good()) {
            dist.setParameters(a, b);
        }

    } catch (const std::exception &) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }

    return is;
}

void UniformDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                   std::span<double> out) const {
    // Tier 1 — concrete non-virtual loop; the valid-input path reduces to a
    // constant (-logRange_), so the compiler auto-vectorizes to a compare +
    // blend under -march=native / /arch:AVX512.
    // Tier 2 with explicit intrinsics would replicate that same compare/blend
    // pattern for marginal gain; the auto-vectorized version is near-optimal.
    if (!isCacheValid())
        updateCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        out[i] = UniformDistribution::getLogProbability(observations[i]);
    }
}

} // namespace libhmm
