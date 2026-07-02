#include "libhmm/distributions/exponential_distribution.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/math/weighted_stats.h"
#include "libhmm/performance/simd_double_ops.h" // runtime dispatch
#include <limits>
#include <numeric>
#include <span>

using namespace libhmm::constants;

namespace libhmm {

/**
 * Computes the probability density function for the Exponential distribution.
 *
 * For continuous distributions in discrete sampling contexts, we approximate
 * the probability as P(x - ε <= X <= x) = F(x) - F(x - ε) where ε is a small tolerance.
 *
 * This provides a numerically stable approximation of the PDF scaled by the tolerance,
 * which is appropriate for discrete sampling of continuous distributions.
 *
 * @param x The value at which to evaluate the probability
 * @return Approximated probability for discrete sampling
 */
double ExponentialDistribution::getProbability(double value) const {
    // Exponential distribution has support [0, ∞)
    if (value < math::ZERO_DOUBLE || std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }

    // For continuous distributions, we return the actual PDF value
    // This is more mathematically correct than the old discrete approximation
    if (value == math::ZERO_DOUBLE) {
        // At x=0, PDF equals λ (the rate parameter)
        return lambda_;
    }

    ensureCache();
    return lambda_ * std::exp(negLambda_ * value);
}

/**
 * Computes the logarithm of the probability density function for numerical stability.
 *
 * For exponential distribution: log(f(x)) = log(λ) - λx for x ≥ 0
 *
 * @param x The value at which to evaluate the log-PDF
 * @return Natural logarithm of the probability density, or -∞ for invalid values
 */
double ExponentialDistribution::getLogProbability(double value) const noexcept {
    // Exponential distribution has support [0, ∞)
    if (value < math::ZERO_DOUBLE || std::isnan(value) || std::isinf(value)) {
        return -std::numeric_limits<double>::infinity();
    }

    ensureCache();
    return logLambda_ - lambda_ * value;
}

/**
 * Evaluates the CDF for the Exponential distribution at x.
 *
 * Formula: F(x) = 1 - exp(-λx) for x ≥ 0
 *
 * @param x The value at which to evaluate the CDF
 * @return Cumulative probability P(X ≤ x)
 */
double ExponentialDistribution::getCumulativeProbability(double x) const noexcept {
    const double y = math::ONE - std::exp(-lambda_ * x);
    assert(y >= math::ZERO_DOUBLE);
    return y;
}

/**
 * Fits the distribution parameters to the given data using maximum likelihood estimation.
 *
 * For the Exponential distribution, the MLE of the rate parameter is:
 * λ = 1 / sample_mean
 *
 * The sample mean is calculated as:
 * mean = Σ(x_i) / N for i = 1 to N
 *
 * @param values Vector of observed data points
 */
double ExponentialDistribution::sample(std::mt19937_64 &rng) const {
    std::exponential_distribution<double> dist(lambda_);
    return dist(rng);
}

void ExponentialDistribution::fit(std::span<const double> data) {
    if (data.size() <= 1) {
        reset();
        return;
    }

    double mean = 0.0;
    std::size_t count = 0;
    for (const double val : data) {
        if (val < 0.0 || std::isnan(val) || std::isinf(val)) {
            reset();
            return;
        }
        ++count;
        mean += (val - mean) / static_cast<double>(count);
    }

    if (mean <= 0.0 || !std::isfinite(mean)) {
        reset();
        return;
    }
    const double lam = 1.0 / mean;
    if (!std::isfinite(lam) || lam <= 0.0) {
        reset();
        return;
    }
    lambda_ = lam;
    invalidateCache();
}

void ExponentialDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    // Weighted MLE: λ = 1 / weighted_mean
    const auto mean = detail::compute_weighted_mean(data, weights);
    // Guard: near-zero weight → keep current parameters (not reset).
    if (!mean)
        return;
    if (*mean <= 0.0 || !std::isfinite(*mean)) {
        reset();
        return;
    }
    const double lam = 1.0 / *mean;
    if (!std::isfinite(lam) || lam <= 0.0) {
        reset();
        return;
    }
    lambda_ = lam;
    invalidateCache();
}

/**
 * Resets the distribution to default parameters (λ = 1.0).
 * This corresponds to the standard exponential distribution.
 */
void ExponentialDistribution::reset() noexcept {
    lambda_ = math::ONE;
    invalidateCache();
}

std::string ExponentialDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Exponential Distribution:\n";
    oss << "      λ (rate parameter) = " << lambda_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const libhmm::ExponentialDistribution &distribution) {
    os << distribution.toString();
    return os;
}

// Parses the format produced by toString() / operator<<:
//   Exponential Distribution:
//     \u03bb (rate parameter) = VALUE
//     Mean = VALUE
std::istream &operator>>(std::istream &is, libhmm::ExponentialDistribution &distribution) {
    try {
        std::string s, t;
        is >> s >> s;                // "Exponential" "Distribution:"
        is >> s >> s >> s >> s >> t; // "\u03bb" "(rate" "parameter)" "=" VALUE
        const double lambda = std::stod(t);
        is >> s >> s >> t; // skip Mean
        if (is.good())
            distribution.setLambda(lambda);
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

bool ExponentialDistribution::operator==(const ExponentialDistribution &other) const {
    using namespace libhmm::constants;
    return std::abs(lambda_ - other.lambda_) < precision::LIMIT_TOLERANCE;
}

// =============================================================================
// Batch log-PDF — dispatched at runtime to the best ISA kernel.
// Kernels extracted to simd_double_ops_*.cpp; selected once at startup via CPUID.
// =============================================================================

void ExponentialDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                       std::span<double> out) const {
    ensureCache();
    performance::get_double_vec_ops().exponential_batch(
        observations.data(), out.data(), observations.size(), logLambda_, negLambda_);
}

std::string ExponentialDistribution::to_json() const {
    return json::write_distribution("Exponential", {{"lambda", lambda_}});
}
std::unique_ptr<EmissionDistribution> ExponentialDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double lambda = r.read_double();
    r.consume('}');
    return std::make_unique<ExponentialDistribution>(lambda);
}

} // namespace libhmm
