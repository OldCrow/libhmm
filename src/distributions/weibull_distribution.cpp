#include "libhmm/distributions/weibull_distribution.h"
#include "libhmm/io/json_utils.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <algorithm> // For std::max, std::min (exists in common.h, included for clarity)
#include <numeric>   // For std::accumulate (not in common.h)
#include <vector>    // For precomputed log arrays used in MLE Newton iterations

using namespace libhmm::constants;

namespace libhmm {

double WeibullDistribution::getProbability(double value) const {
    if (value < math::ZERO_DOUBLE || std::isnan(value) || std::isinf(value))
        return math::ZERO_DOUBLE;
    ensureCache();

    // Handle boundary case
    if (value == math::ZERO_DOUBLE) {
        return (k_ == math::ONE) ? kOverLambda_ : math::ZERO_DOUBLE;
    }

    // Use optimized log-space calculation then exponentiate
    // This avoids numerical issues with very small/large values
    const double logPdf = getLogProbability(value);
    if (logPdf == -std::numeric_limits<double>::infinity()) {
        return math::ZERO_DOUBLE;
    }

    return std::exp(logPdf);
}

double WeibullDistribution::getLogProbability(double value) const noexcept {
    // Weibull distribution is only defined for x ≥ 0
    if (value < math::ZERO_DOUBLE || std::isnan(value) || std::isinf(value)) {
        return -std::numeric_limits<double>::infinity();
    }

    ensureCache();
    if (value == math::ZERO_DOUBLE)
        return (k_ == math::ONE) ? logK_ - logLambda_ : -std::numeric_limits<double>::infinity();

    // Optimized log PDF computation using cached values:
    // log(f(x)) = log(k) - log(λ) + (k-1)*log(x) - (k-1)*log(λ) - (x/λ)^k
    //           = log(k) - k*log(λ) + (k-1)*log(x) - (x*invλ)^k

    const double logX = std::log(value);
    const double xTimesInvLambda = value * invLambda_; // Use cached reciprocal

    // Use efficient power calculation for common k values
    double powerTerm = 0.0;
    if (k_ == math::ONE) {
        powerTerm = xTimesInvLambda; // Linear case
    } else if (k_ == math::TWO) {
        powerTerm = xTimesInvLambda * xTimesInvLambda; // Quadratic case (Rayleigh)
    } else {
        powerTerm = std::pow(xTimesInvLambda, k_); // General case
    }

    const double logPdf = logK_ - k_ * logLambda_ + kMinus1_ * logX - powerTerm;

    return logPdf;
}

namespace {

/// MoM seed for k (coefficient-of-variation approximation).
void weibull_mom_init(double mean, double var, double &k_out, double &lambda_out) noexcept {
    const double cv = std::sqrt(var) / mean;
    double k_est = 0.0;
    if (cv < 0.2)
        k_est = 1.0 / (cv * cv * 6.0);
    else if (cv < 1.0)
        k_est = std::pow(1.2 / cv, 1.086);
    else
        k_est = 1.0 / cv;
    k_est = std::max(thresholds::MIN_DISTRIBUTION_PARAMETER,
                     std::min(k_est, thresholds::MAX_DISTRIBUTION_PARAMETER));
    k_out = k_est;
    lambda_out = mean / std::exp(std::lgamma(1.0 + 1.0 / k_est));
}

/// Weibull MLE via Newton–Raphson on the profile score for k.
///
/// Profile score: g(k) = E_k[log x] − 1/k − s̄ = 0,
///   where E_k[·] weights observations by w_i·x_i^k and s̄ = Σw_i log(x_i)/sumW.
/// Derivative: g'(k) = Var_k[log x] + 1/k² > 0 (monotone, Newton always converges).
/// After convergence: λ = (Σw_i x_i^k / sumW)^(1/k).
///
/// Inputs (valid positive observations only, precomputed log values):
///   log_x[i]  = log(x_i),  log_x2[i] = (log x_i)²,  w[i] = weight_i.
///   If w is empty, unit weights are assumed.
[[nodiscard]] std::pair<double, double> weibull_mle_solve(std::span<const double> log_x,
                                                          std::span<const double> log_x2,
                                                          std::span<const double> w, double sumW,
                                                          double s_bar, double init_k) noexcept {
    const std::size_t n = log_x.size();
    const bool unit_w = w.empty();
    double k = init_k;

    for (int iter = 0; iter < 100; ++iter) {
        double s0 = 0.0; // Σ w_i x_i^k
        double s1 = 0.0; // Σ w_i x_i^k log(x_i)
        double s2 = 0.0; // Σ w_i x_i^k (log x_i)^2
        for (std::size_t i = 0; i < n; ++i) {
            // Clamp exponent to avoid std::exp overflow to inf on outlier data.
            const double exponent = std::clamp(k * log_x[i], -700.0, 700.0);
            const double wxk = (unit_w ? 1.0 : w[i]) * std::exp(exponent);
            s0 += wxk;
            s1 += wxk * log_x[i];
            s2 += wxk * log_x2[i];
        }
        if (!std::isfinite(s0) || s0 <= 0.0)
            break;

        const double c1 = s1 / s0;               // E_k[log x]
        const double var_k = s2 / s0 - c1 * c1;  // Var_k[log x] ≥ 0
        const double g = c1 - 1.0 / k - s_bar;   // profile score
        const double gp = var_k + 1.0 / (k * k); // always > 0
        if (gp <= 0.0)
            break;

        const double dk = g / gp;
        k -= dk;
        if (k <= 0.0)
            k = 1e-8;
        if (std::fabs(dk) < 1e-11 * k)
            break;
    }

    // Fall back to the MoM seed rather than returning a garbage k.
    if (!std::isfinite(k) || k <= 0.0)
        k = init_k;

    // λ = (Σ w_i x_i^k / sumW)^(1/k)
    double s0f = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double exponent = std::clamp(k * log_x[i], -700.0, 700.0);
        s0f += (unit_w ? 1.0 : w[i]) * std::exp(exponent);
    }
    const double lambda = (s0f > 0.0 && sumW > 0.0) ? std::exp(std::log(s0f / sumW) / k) : 1.0;
    return {k, lambda};
}

} // anonymous namespace

double WeibullDistribution::sample(std::mt19937_64 &rng) const {
    // std::weibull_distribution<double>(a, b): a = shape k, b = scale lambda.
    std::weibull_distribution<double> dist(k_, lambda_);
    return dist(rng);
}

void WeibullDistribution::fit(std::span<const double> data) {
    if (data.size() < 2) {
        reset();
        return;
    }

    std::vector<double> lx, lx2;
    lx.reserve(data.size());
    lx2.reserve(data.size());

    double mean = 0.0, M2 = 0.0, sum_log_x = 0.0;
    std::size_t count = 0;

    for (const double val : data) {
        if (val <= 0.0 || !std::isfinite(val))
            throw std::invalid_argument("Weibull fitting requires strictly positive values");
        ++count;
        const double delta = val - mean;
        mean += delta / static_cast<double>(count);
        M2 += delta * (val - mean);
        const double l = std::log(val);
        sum_log_x += l;
        lx.push_back(l);
        lx2.push_back(l * l);
    }

    if (count < 2) {
        reset();
        return;
    }

    const double n = static_cast<double>(count);
    const double variance = M2 / (n - 1.0);
    const double s_bar = sum_log_x / n;

    double k_init = 1.0;
    if (variance > precision::ZERO && mean > precision::ZERO) {
        double lambda_tmp = 0.0;
        weibull_mom_init(mean, variance, k_init, lambda_tmp);
    }

    const auto [k, lambda] = weibull_mle_solve(lx, lx2, {}, n, s_bar, k_init);

    if (std::isfinite(k) && std::isfinite(lambda) && k > precision::ZERO &&
        lambda > precision::ZERO && k < thresholds::MAX_DISTRIBUTION_PARAMETER &&
        lambda < thresholds::MAX_DISTRIBUTION_PARAMETER) {
        k_ = k;
        lambda_ = lambda;
        invalidateCache();
    } else {
        reset();
    }
}

void WeibullDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    // Guard: keep current parameters when effective weight is near zero.
    // Calling reset() would destroy valid parameters and cause state collapse in EM.
    if (sumW < precision::ZERO || std::isnan(sumW))
        return;

    std::vector<double> lx, lx2, wt;
    lx.reserve(data.size());
    lx2.reserve(data.size());
    wt.reserve(data.size());

    double sum_wx = 0.0, sum_wx2 = 0.0, sum_wlog_x = 0.0, cumW = 0.0;

    for (std::size_t i = 0; i < data.size(); ++i) {
        const double val = data[i];
        const double weight = weights[i];
        if (val <= 0.0 || !std::isfinite(val) || !std::isfinite(weight) || weight <= 0.0)
            continue;
        cumW += weight;
        sum_wx += weight * val;
        sum_wx2 += weight * val * val;
        const double l = std::log(val);
        sum_wlog_x += weight * l;
        lx.push_back(l);
        lx2.push_back(l * l);
        wt.push_back(weight);
    }

    if (cumW < precision::ZERO || lx.empty()) {
        reset();
        return;
    }

    const double mean = sum_wx / cumW;
    const double variance = sum_wx2 / cumW - mean * mean;
    const double s_bar = sum_wlog_x / cumW;

    double k_init = 1.0;
    if (variance > precision::ZERO && mean > precision::ZERO) {
        double lambda_tmp = 0.0;
        weibull_mom_init(mean, variance, k_init, lambda_tmp);
    }

    const auto [k, lambda] = weibull_mle_solve(lx, lx2, wt, cumW, s_bar, k_init);

    if (std::isfinite(k) && std::isfinite(lambda) && k > precision::ZERO &&
        lambda > precision::ZERO && k < thresholds::MAX_DISTRIBUTION_PARAMETER &&
        lambda < thresholds::MAX_DISTRIBUTION_PARAMETER) {
        k_ = k;
        lambda_ = lambda;
        invalidateCache();
    } else {
        reset();
    }
}

void WeibullDistribution::reset() noexcept {
    k_ = math::ONE;
    lambda_ = math::ONE;
    invalidateCache();
}

std::string WeibullDistribution::toString() const {
    std::ostringstream oss{};
    oss << std::fixed << std::setprecision(6);
    oss << "Weibull Distribution:\n";
    oss << "      k (shape) = " << k_ << "\n";
    oss << "      λ (scale) = " << lambda_ << "\n";
    return oss.str();
}

double WeibullDistribution::CDF(double x) const noexcept {
    if (x <= math::ZERO_DOUBLE)
        return math::ZERO_DOUBLE;

    ensureCache();
    // CDF(x) = 1 - exp(-(x/λ)^k)
    const double xTimesInvLambda = x * invLambda_; // Use cached reciprocal

    // Use efficient power calculation for common k values
    double powerTerm = 0.0;
    if (k_ == math::ONE) {
        powerTerm = xTimesInvLambda; // Linear case (Exponential)
    } else if (k_ == math::TWO) {
        powerTerm = xTimesInvLambda * xTimesInvLambda; // Quadratic case (Rayleigh)
    } else {
        powerTerm = std::pow(xTimesInvLambda, k_); // General case
    }

    return math::ONE - std::exp(-powerTerm);
}

bool WeibullDistribution::operator==(const WeibullDistribution &other) const noexcept {
    // Use tolerance for floating-point comparison
    const double tolerance = precision::ZERO;
    return std::abs(k_ - other.k_) < tolerance && std::abs(lambda_ - other.lambda_) < tolerance;
}

std::ostream &operator<<(std::ostream &os, const WeibullDistribution &distribution) {
    return os << distribution.toString();
}

std::istream &operator>>(std::istream &is, WeibullDistribution &distribution) {
    try {
        std::string token;
        double k = 0.0, lambda = 0.0;
        // Expected format: "Weibull Distribution: k (shape) = <value> λ (scale) = <value>"
        std::string k_str, lambda_str;
        is >> token >> token;                        // "Weibull" "Distribution:"
        is >> token >> token >> token >> k_str;      // "k" "(shape)" "=" <k_str>
        is >> token >> token >> token >> lambda_str; // "λ" "(scale)" "=" <lambda_str>
        k = std::stod(k_str);
        lambda = std::stod(lambda_str);

        if (is.good()) {
            distribution = WeibullDistribution(k, lambda);
        }

    } catch (const std::exception &) {
        // Set error state on stream if parsing fails
        is.setstate(std::ios::failbit);
    }

    return is;
}

void WeibullDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                   std::span<double> out) const {
    // Tier 1 — concrete non-virtual loop; compiler auto-vectorizes the arithmetic
    // terms under -march=native. Index loop preserved: a std::ranges::transform
    // lambda would add an indirect call boundary that inhibits auto-vectorisation.
    // Tier 2 upgrade requires both vectorised log(x) and vectorised pow(x, k):
    // inner loop is log(k) - k*log(λ) + (k-1)*log(x) - (x/λ)^k. Available via
    // Intel SVML (_mm512_log_pd + _mm512_pow_pd), but not portably without a
    // math-library dependency. The k=1 (exponential) and k=2 (Rayleigh) special
    // cases eliminate pow and could be handled without SVML.
    ensureCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        out[i] = WeibullDistribution::getLogProbability(observations[i]);
    }
}

std::string WeibullDistribution::to_json() const {
    return json::write_distribution("Weibull", {{"k", k_}, {"lambda", lambda_}});
}
std::unique_ptr<EmissionDistribution> WeibullDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double k = r.read_double();
    r.read_key();
    const double lambda = r.read_double();
    r.consume('}');
    return std::make_unique<WeibullDistribution>(k, lambda);
}

} // namespace libhmm
