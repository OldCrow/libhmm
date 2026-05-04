#include "libhmm/distributions/weibull_distribution.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <algorithm> // For std::max, std::min (exists in common.h, included for clarity)
#include <numeric>   // For std::accumulate (not in common.h)

using namespace libhmm::constants;

namespace libhmm {

double WeibullDistribution::getProbability(double value) const {
    if (value < math::ZERO_DOUBLE || std::isnan(value) || std::isinf(value))
        return math::ZERO_DOUBLE;
    if (!isCacheValid())
        updateCache();

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

    if (!isCacheValid())
        updateCache();
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

static void weibull_mom_fit(double mean, double var, double &k_out, double &lambda_out) {
    // MOM approximation using coefficient of variation
    const double cv = std::sqrt(var) / mean;
    double k_est;
    if (cv < 0.2)
        k_est = libhmm::constants::math::ONE / (cv * cv * libhmm::constants::math::TEN * 0.6);
    else if (cv < libhmm::constants::math::ONE)
        k_est = std::pow(1.2 / cv, 1.086);
    else
        k_est = libhmm::constants::math::ONE / cv;
    k_est = std::max(libhmm::constants::thresholds::MIN_DISTRIBUTION_PARAMETER,
                     std::min(k_est, libhmm::constants::thresholds::MAX_DISTRIBUTION_PARAMETER));
    const double gamma_term =
        std::exp(std::lgamma(libhmm::constants::math::ONE + libhmm::constants::math::ONE / k_est));
    k_out = k_est;
    lambda_out = mean / gamma_term;
}

void WeibullDistribution::apply_fit_params(double mean, double var) {
    if (var <= precision::ZERO || mean <= precision::ZERO) {
        reset();
        return;
    }
    double k_est, lambda_est;
    weibull_mom_fit(mean, var, k_est, lambda_est);
    if (lambda_est > precision::ZERO && lambda_est < thresholds::MAX_DISTRIBUTION_PARAMETER) {
        k_ = k_est;
        lambda_ = lambda_est;
        invalidateCache();
    } else {
        reset();
    }
}

void WeibullDistribution::fit(std::span<const double> data) {
    if (data.size() < 2) {
        reset();
        return;
    }
    for (const double val : data)
        if (val < math::ZERO_DOUBLE || std::isnan(val) || std::isinf(val))
            throw std::invalid_argument("Weibull fitting requires non-negative values");
    const auto n = static_cast<double>(data.size());
    double mean = 0.0, m2 = 0.0;
    std::size_t count = 0;
    for (const double val : data) {
        ++count;
        const double delta = val - mean;
        mean += delta / static_cast<double>(count);
        m2 += delta * (val - mean);
    }
    apply_fit_params(mean, m2 / (n - math::ONE));
}

void WeibullDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    double sumW = 0.0;
    for (const double w : weights)
        sumW += w;
    if (sumW < precision::ZERO || std::isnan(sumW)) {
        reset();
        return;
    }
    double mean = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] >= 0.0 && std::isfinite(data[i]) && weights[i] > 0.0)
            mean += (weights[i] / sumW) * data[i];
    double var = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] >= 0.0 && std::isfinite(data[i]) && weights[i] > 0.0)
            var += weights[i] * (data[i] - mean) * (data[i] - mean);
    apply_fit_params(mean, var / sumW);
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

    if (!isCacheValid())
        updateCache();
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
    // terms under -march=native / /arch:AVX512.
    // Tier 2 upgrade requires both vectorised log(x) and vectorised pow(x, k):
    // inner loop is log(k) - k*log(λ) + (k-1)*log(x) - (x/λ)^k. Available via
    // Intel SVML (_mm512_log_pd + _mm512_pow_pd), but not portably without a
    // math-library dependency. The k=1 (exponential) and k=2 (Rayleigh) special
    // cases eliminate pow and could be handled without SVML.
    if (!isCacheValid())
        updateCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        out[i] = WeibullDistribution::getLogProbability(observations[i]);
    }
}

} // namespace libhmm
