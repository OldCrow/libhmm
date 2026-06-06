#include "libhmm/distributions/gamma_distribution.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/math/psi_functions.h"
#include <numeric>
#include <span>

using namespace libhmm::constants;

namespace libhmm {

/**
 * Computes the probability density function for the Gamma distribution.
 * PDF: f(x) = (1/(Γ(k)θ^k)) * x^(k-1) * exp(-x/θ) for x ≥ 0
 *
 * @param x The value at which to evaluate the probability
 * @return Probability density
 */
double GammaDistribution::getProbability(double x) const {
    if (std::isnan(x) || std::isinf(x) || x < 0.0)
        return 0.0;
    if (x == 0.0)
        return (k_ < 1.0) ? std::numeric_limits<double>::infinity() : 0.0;
    if (!isCacheValid())
        updateCache();

    // Use log space for numerical stability then exponentiate
    const double logPdf = getLogProbability(x);
    if (logPdf == -std::numeric_limits<double>::infinity()) {
        return 0.0;
    }

    return std::exp(logPdf);
}

/**
 * Evaluates the logarithm of the probability density function for numerical stability.
 * Formula: log PDF(x) = (k-1)*ln(x) - x/θ - k*ln(θ) - ln(Γ(k))
 *
 * @param x The value at which to evaluate the log PDF
 * @return Log probability density
 */
double GammaDistribution::getLogProbability(double x) const noexcept {
    // Gamma distribution has support [0, ∞)
    if (std::isnan(x) || std::isinf(x) || x < 0.0) {
        return -std::numeric_limits<double>::infinity();
    }

    if (x == 0.0) {
        // For x=0: log(PDF) = -∞ unless k < 1 and we're at the boundary
        return (k_ < 1.0) ? std::numeric_limits<double>::infinity()
                          : -std::numeric_limits<double>::infinity();
    }

    if (!isCacheValid())
        updateCache();
    // log PDF(x) = (k-1)*ln(x) - x/θ - k*ln(θ) - ln(Γ(k))
    const double logPdf = kMinus1_ * std::log(x) - x / theta_ - kLogTheta_ - logGammaK_;

    return logPdf;
}

/**
 * Evaluates the CDF at x using the incomplete gamma function
 * Formula: CDF(x) = P(k, x/θ) = γ(k, x/θ) / Γ(k)
 * where P is the regularized incomplete gamma function
 *
 * @param x The value at which to evaluate the CDF
 * @return Cumulative probability P(X ≤ x)
 */
double GammaDistribution::getCumulativeProbability(double x) const noexcept {
    if (x <= 0)
        return 0.0;

    double i = gammap(k_, x / theta_);
    if (std::isnan(i) || i < 0.0)
        i = 0.0;

    // Clamp to valid probability range
    if (i > 1.0)
        i = 1.0;

    return i;
}

namespace {

/// Solve log(k) − ψ(k) = s for k > 0 via Newton–Raphson.
/// s = log(x̄) − Ē[log x] is the Gamma MLE sufficient-statistic contrast
/// (always > 0 for positive Gamma data; Jensen: log concave ⇒ E[log X] < log E[X]).
/// Initial estimate: Cheng & Feast (1979), converges in ≤ 10 iterations typically.
/// Returns (k, θ) with θ = x̄/k.
[[nodiscard]] std::pair<double, double> gamma_mle_solve(double mean_x, double mean_log_x) noexcept {
    const double s = std::log(mean_x) - mean_log_x;
    if (s <= 0.0)
        return {1.0, mean_x}; // degenerate input — return sensible fallback

    // Cheng & Feast (1979) starting estimate for k
    double k = (3.0 - s + std::sqrt((s - 3.0) * (s - 3.0) + 24.0 * s)) / (12.0 * s);

    // Newton–Raphson: f(k) = log k − ψ(k) − s = 0,  f'(k) = 1/k − ψ'(k)
    for (int i = 0; i < 20; ++i) {
        const double f = std::log(k) - detail::digamma(k) - s;
        const double fp = 1.0 / k - detail::trigamma(k);
        if (std::fabs(fp) < 1e-15)
            break;
        const double dk = f / fp;
        k -= dk;
        if (k <= 0.0)
            k = 1e-10;
        if (std::fabs(dk) < 1e-11 * k)
            break;
    }
    return {k, mean_x / k};
}

} // anonymous namespace

double GammaDistribution::sample(std::mt19937_64& rng) const {
    // std::gamma_distribution<double>(alpha, beta) with alpha=shape, beta=scale.
    std::gamma_distribution<double> dist(k_, theta_);
    return dist(rng);
}

void GammaDistribution::fit(std::span<const double> data) {
    if (data.size() < 2) {
        reset();
        return;
    }
    double sum_x = 0.0, sum_log_x = 0.0;
    std::size_t count = 0;
    for (const double val : data) {
        if (val > 0.0 && std::isfinite(val)) {
            ++count;
            sum_x += val;
            sum_log_x += std::log(val);
        }
    }
    if (count < 2) {
        reset();
        return;
    }
    const double n = static_cast<double>(count);
    auto [k, theta] = gamma_mle_solve(sum_x / n, sum_log_x / n);
    if (!std::isfinite(k) || !std::isfinite(theta) || k <= 0.0 || theta <= 0.0) {
        reset();
        return;
    }
    k_ = k;
    theta_ = theta;
    invalidateCache();
}

void GammaDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    // Guard: keep current parameters when effective weight is near zero.
    // Calling reset() would destroy valid parameters and cause state collapse in EM.
    if (sumW < precision::ZERO || std::isnan(sumW))
        return;
    double sum_wx = 0.0, sum_wlx = 0.0, cumW = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i] > 0.0 && std::isfinite(data[i]) && weights[i] > 0.0) {
            cumW += weights[i];
            sum_wx += weights[i] * data[i];
            sum_wlx += weights[i] * std::log(data[i]);
        }
    }
    if (cumW < precision::ZERO) {
        reset();
        return;
    }
    auto [k, theta] = gamma_mle_solve(sum_wx / cumW, sum_wlx / cumW);
    if (!std::isfinite(k) || !std::isfinite(theta) || k <= 0.0 || theta <= 0.0) {
        reset();
        return;
    }
    k_ = k;
    theta_ = theta;
    invalidateCache();
}

/**
 * Resets the distribution to default parameters (k = 1.0, θ = 1.0).
 * This corresponds to the standard exponential distribution.
 */
void GammaDistribution::reset() noexcept {
    k_ = 1.0;
    theta_ = 1.0;
    invalidateCache();
}

std::string GammaDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Gamma Distribution:\n";
    oss << "      k (shape parameter) = " << k_ << "\n";
    oss << "      θ (scale parameter) = " << theta_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const libhmm::GammaDistribution &distribution) {
    os << distribution.toString();
    return os;
}

// Parses the format produced by toString() / operator<<:
//   Gamma Distribution:
//     k (shape parameter) = VALUE
//     \u03b8 (scale parameter) = VALUE
//     Mean = VALUE
//     Variance = VALUE
std::istream &operator>>(std::istream &is, libhmm::GammaDistribution &distribution) {
    try {
        std::string s, t;
        is >> s >> s;                // "Gamma" "Distribution:"
        is >> s >> s >> s >> s >> t; // "k" "(shape" "parameter)" "=" VALUE
        const double k = std::stod(t);
        is >> s >> s >> s >> s >> t; // "\u03b8" "(scale" "parameter)" "=" VALUE
        const double theta = std::stod(t);
        is >> s >> s >> t;
        is >> s >> s >> t; // skip Mean, Variance
        if (is.good())
            distribution.setParameters(k, theta);
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

bool GammaDistribution::operator==(const GammaDistribution &other) const {
    using namespace libhmm::constants;
    return std::abs(k_ - other.k_) < precision::LIMIT_TOLERANCE &&
           std::abs(theta_ - other.theta_) < precision::LIMIT_TOLERANCE;
}

void GammaDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                 std::span<double> out) const {
    // Tier 1 — concrete non-virtual loop; compiler auto-vectorizes the arithmetic
    // terms under -march=native. Index loop preserved: a std::ranges::transform
    // lambda would add an indirect call boundary that inhibits auto-vectorisation.
    // Tier 2 upgrade requires vectorised log(x): the inner loop contains
    // (k-1)*log(x) - x/θ, which needs a vectorised log — available via Intel SVML,
    // GNU libmvec, or Apple Accelerate vvlog, but not portably without a
    // math-library dependency.
    if (!isCacheValid())
        updateCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        out[i] = GammaDistribution::getLogProbability(observations[i]);
    }
}

std::string GammaDistribution::to_json() const {
    return json::write_distribution("Gamma", {{"k", k_}, {"theta", theta_}});
}
std::unique_ptr<EmissionDistribution> GammaDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double k = r.read_double();
    r.read_key();
    const double theta = r.read_double();
    r.consume('}');
    return std::make_unique<GammaDistribution>(k, theta);
}

} // namespace libhmm
