#include "libhmm/distributions/poisson_distribution.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/math/weighted_stats.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <span>

using namespace libhmm::constants;

namespace libhmm {

/*
 * log(k!) from the shared table (math/log_factorial.h): exact for k <= 18,
 * <= 1 ULP up to k = 1023, lgamma above that.
 *
 * The old k < 0 guard returned -inf, which was sign-wrong for the one way
 * this value is used -- `... - logFactorial(k)` would have yielded +inf, i.e.
 * a log-probability above zero. It was unreachable (every call site is behind
 * isValidCount), and the shared helper returns +inf, matching the poles of
 * Gamma and giving -inf for the log-probability.
 */
double PoissonDistribution::logFactorial(int k) noexcept {
    return detail::log_factorial(k);
}

/*
 * Computes the Poisson PMF: P(X = k) = (λ^k * e^(-λ)) / k!
 * Uses logarithms for numerical stability: log(P) = k*log(λ) - λ - log(k!)
 */
double PoissonDistribution::getProbability(double value) const {
    if (!isValidCount(value))
        return 0.0;
    const auto k = static_cast<int>(value);
    ensureCache();

    // Handle edge cases - use cached exp(-lambda) for efficiency
    if (k == 0) {
        return expNegLambda_;
    }

    // For very large lambda or k, check for potential overflow/underflow
    if (lambda_ > 700.0 || k > 700) {
        // Use log-space computation to avoid overflow
        const double logProb = k * logLambda_ - lambda_ - logFactorial(k);

        // Check for underflow
        if (logProb < -700.0) {
            return 0.0;
        }

        return std::exp(logProb);
    }

    // Standard computation for moderate values
    const double logProb = k * logLambda_ - lambda_ - logFactorial(k);
    return std::exp(logProb);
}

/*
 * Fits the Poisson distribution to data using Maximum Likelihood Estimation.
 * For Poisson, MLE of λ is simply the sample mean: λ̂ = (1/n) * Σ(x_i)
 * Uses single-pass algorithm for efficiency.
 */
double PoissonDistribution::sample(std::mt19937_64 &rng) const {
    std::poisson_distribution<int> dist(lambda_);
    return static_cast<double>(dist(rng));
}

void PoissonDistribution::fit(std::span<const double> data) {
    if (data.empty()) {
        reset();
        return;
    }
    double sum = 0.0;
    for (const double val : data) {
        if (val < 0.0 || !std::isfinite(val))
            throw std::invalid_argument("Poisson fit: requires non-negative finite values");
        sum += val;
    }
    lambda_ = std::max(sum / static_cast<double>(data.size()), precision::ZERO);
    invalidateCache();
}

void PoissonDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    // Weighted MLE: λ = weighted mean
    const auto mean = detail::compute_weighted_mean(data, weights);
    // Guard: near-zero weight → keep current parameters (not reset).
    if (!mean)
        return;
    lambda_ = std::max(*mean, precision::ZERO);
    invalidateCache();
}

/*
 * Resets the distribution to default parameters.
 */
void PoissonDistribution::reset() noexcept {
    lambda_ = 1.0;
    invalidateCache();
}

/*
 * Creates a string representation of the Poisson distribution.
 */
std::string PoissonDistribution::toString() const {
    std::ostringstream oss{};
    oss << "Poisson Distribution:\n";
    oss << "      λ (rate parameter) = " << std::fixed << std::setprecision(6) << lambda_ << "\n";
    oss << "      Mean = " << std::fixed << std::setprecision(6) << getMean() << "\n";
    oss << "      Variance = " << std::fixed << std::setprecision(6) << getVariance() << "\n";

    return oss.str();
}

/*
 * Stream output operator implementation.
 */
std::ostream &operator<<(std::ostream &os, const libhmm::PoissonDistribution &distribution) {
    os << distribution.toString();
    return os;
}

/*
 * Evaluates the logarithm of the probability mass function
 * Formula: log P(X = k) = k*log(λ) - λ - log(k!)
 * More numerically stable for small probabilities
 */
double PoissonDistribution::getLogProbability(double value) const noexcept {
    // Validate input - must be non-negative integer
    if (!isValidCount(value)) {
        return -std::numeric_limits<double>::infinity();
    }

    const auto k = static_cast<int>(value);

    ensureCache();
    const double logProb = k * logLambda_ - lambda_ - logFactorial(k);

    return logProb;
}

/*
 * Evaluates the CDF at k using cumulative sum approach
 * For large k, uses asymptotic approximation for efficiency
 */
double PoissonDistribution::getCumulativeProbability(double k) const noexcept {
    // Validate input
    if (std::isnan(k) || std::isinf(k)) {
        return math::ZERO_DOUBLE;
    }

    if (k < math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }

    const auto kInt = static_cast<int>(std::floor(k));

    // For very large k or lambda, the cumulative sum becomes computationally expensive
    // and numerically unstable. In such cases, use normal approximation.
    if (kInt > 100 && lambda_ > 100.0) {
        ensureCache();
        // Normal approximation with continuity correction
        // Use cached sqrt(lambda) for efficiency
        const double z = (static_cast<double>(kInt) + 0.5 - lambda_) * invSqrtLambda_;
        return 0.5 * (1.0 + std::erf(z / math::SQRT_2));
    }

    // For moderate values, compute CDF as cumulative sum: P(X ≤ k) = Σ(i=0 to k) P(X = i)
    double cdf = math::ZERO_DOUBLE;
    for (int i = 0; i <= kInt; ++i) {
        cdf += getProbability(static_cast<double>(i));

        // Early termination if we've accumulated essentially all probability
        if (cdf >= 0.999999) {
            break;
        }
    }

    return std::min(math::ONE, cdf);
}

/*
 * Equality comparison operator with numerical tolerance
 */
bool PoissonDistribution::operator==(const PoissonDistribution &other) const {
    const double tolerance = 1e-10;
    return std::abs(lambda_ - other.lambda_) < tolerance;
}

/*
 * Stream input operator implementation.
 * Expects format: "Poisson Distribution: λ = <value>"
 */
// Parses the format produced by toString() / operator<<:
//   Poisson Distribution:
//     \u03bb (rate parameter) = VALUE
//     Mean = VALUE
//     Variance = VALUE
std::istream &operator>>(std::istream &is, libhmm::PoissonDistribution &distribution) {
    try {
        std::string s, t;
        is >> s >> s;                // "Poisson" "Distribution:"
        is >> s >> s >> s >> s >> t; // "\u03bb" "(rate" "parameter)" "=" VALUE
        const double lambda = std::stod(t);
        is >> s >> s >> t;
        is >> s >> s >> t; // skip Mean, Variance
        if (is.good())
            distribution.setLambda(lambda);
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

void PoissonDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                   std::span<double> out) const {
    // Tier 1 — concrete non-virtual loop; compiler auto-vectorizes the arithmetic
    // terms under -march=native. Index loop preserved: a std::ranges::transform
    // lambda would add an indirect call boundary that inhibits auto-vectorisation.
    // Tier 2 upgrade does NOT need a vectorised lgamma, contrary to what this
    // comment claimed until 2026-08-16. k is an integer count, so log(k!) is a
    // table lookup (math/log_factorial.h, k ≤ 1023). What blocks tier 2 here is
    // the gather to index that table by k — the same blocker as Discrete, and
    // libstats settled empirically that x86 hardware gather is too expensive to
    // pay for (its #33; table kernels are a NEON technique, not an x86 one).
    checkBatchSpans(observations.size(), out.size());
    ensureCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        out[i] = PoissonDistribution::getLogProbability(observations[i]);
    }
}

std::string PoissonDistribution::to_json() const {
    return json::write_distribution("Poisson", {{"lambda", lambda_}});
}
std::unique_ptr<EmissionDistribution> PoissonDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double lambda = r.read_double();
    r.consume('}');
    return std::make_unique<PoissonDistribution>(lambda);
}

} // namespace libhmm
