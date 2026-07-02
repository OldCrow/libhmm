#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/performance/simd_double_ops.h" // runtime dispatch
#include <limits>
#include <numeric>
#include <span>

using namespace libhmm::constants;

namespace libhmm {
/**
 * Returns the probability density function value for the Gaussian distribution.
 *
 * Formula: PDF(x) = (1/σ√(2π)) * exp(-½((x-μ)/σ)²)
 */
double GaussianDistribution::getProbability(double x) const {
    if (std::isnan(x) || std::isinf(x)) {
        return 0.0;
    }
    ensureCache();

    const double exponent = (x - mean_) * (x - mean_) * negHalfSigmaSquaredInv_;
    return normalizationConstant_ * std::exp(exponent);
}

/**
 * Returns the log probability density function value for numerical stability.
 * Formula: log PDF(x) = -½log(2π) - log(σ) - ½((x-μ)/σ)²
 */
double GaussianDistribution::getLogProbability(double x) const noexcept {
    // Validate input
    if (std::isnan(x) || std::isinf(x)) {
        return -std::numeric_limits<double>::infinity();
    }

    ensureCache();
    // Use cached values for maximum performance
    const double z = (x - mean_) * invStandardDeviation_;
    const double logPdf = -0.5 * math::LN_2PI - logStandardDeviation_ - 0.5 * z * z;

    return logPdf;
}

/**
 * Evaluates the CDF for the Normal distribution at x.  The CDF is defined as
 *
 *          1             x - mean
 *   F(x) = -( 1 + erf(---------------) )
 *          2           sigma*sqrt(2)
 */
double GaussianDistribution::getCumulativeProbability(double x) const noexcept {
    // Handle problematic inputs
    if (std::isnan(x) || std::isnan(mean_) || std::isnan(standardDeviation_)) {
        return 0.0;
    }
    if (std::isinf(x)) {
        return (x > 0) ? 1.0 : 0.0;
    }
    if (standardDeviation_ <= 0.0) {
        return (x >= mean_) ? 1.0 : 0.0;
    }

    ensureCache();
    // Use cached sigma*sqrt(2) for efficiency
    const double y = 0.5 * (1 + std::erf((x - mean_) / sigmaSqrt2_));

    // Ensure valid probability range
    if (std::isnan(y) || y < 0.0) {
        return 0.0;
    }
    if (y > 1.0) {
        return 1.0;
    }

    return y;
}

/*
 * Fits the distribution parameters using maximum likelihood estimation with optimized algorithm.
 *
 * Uses single-pass Welford's algorithm for numerically stable variance calculation:
 * - Better cache locality than two-pass algorithm
 * - Numerically stable for extreme values
 * - O(n) time complexity with single data traversal
 */
void GaussianDistribution::fit(std::span<const double> data) {
    if (data.size() <= 1) {
        reset();
        return;
    }

    // Welford's online algorithm: single-pass, numerically stable
    double mean = 0.0;
    double m2 = 0.0;
    std::size_t count = 0;
    for (const double val : data) {
        ++count;
        const double delta = val - mean;
        mean += delta / static_cast<double>(count);
        const double delta2 = val - mean;
        m2 += delta * delta2;
    }

    mean_ = mean;
    // MLE variance: biased N denominator, not N-1, for true log-likelihood maximisation.
    double sd = std::sqrt(m2 / static_cast<double>(count));
    if (sd <= 0.0 || std::isnan(sd) || std::isinf(sd) || sd < precision::MIN_STD_DEV) {
        sd = precision::MIN_STD_DEV;
    }
    standardDeviation_ = sd;
    invalidateCache();
}

void GaussianDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    // Weighted Gaussian MLE for Baum-Welch M-step.
    // Weights are unnormalized γ values; we normalize by their sum.
    const double sumW = [&] {
        double s = 0.0;
        for (const double w : weights)
            s += w;
        return s;
    }();

    // Guard: keep current parameters when effective weight is near zero.
    // Calling reset() would destroy valid parameters and cause state collapse in EM.
    if (sumW < precision::ZERO || std::isnan(sumW))
        return;

    // Weighted Welford for numerical stability
    double mean = 0.0;
    double m2 = 0.0;
    double cumW = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        cumW += weights[i];
        const double delta = data[i] - mean;
        mean += (weights[i] / cumW) * delta;
        const double delta2 = data[i] - mean;
        m2 += weights[i] * delta * delta2;
    }

    mean_ = mean;
    // Population-weighted variance (no Bessel correction for EM)
    double sd = std::sqrt(m2 / sumW);
    if (sd <= 0.0 || std::isnan(sd) || std::isinf(sd) || sd < precision::MIN_STD_DEV) {
        sd = precision::MIN_STD_DEV;
    }
    standardDeviation_ = sd;
    invalidateCache();
}

/**
 * Resets the distribution to default parameters (μ = 0.0, σ = 1.0).
 * This corresponds to the standard normal distribution.
 */
void GaussianDistribution::reset() noexcept {
    mean_ = 0.0;
    standardDeviation_ = 1.0;
    invalidateCache();
}

std::string GaussianDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Gaussian Distribution:\n";
    oss << "      μ (mean) = " << mean_ << "\n";
    oss << "      σ (std. deviation) = " << standardDeviation_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const libhmm::GaussianDistribution &distribution) {
    os << distribution.toString();
    return os;
}

// Parses the format produced by toString() / operator<<:
//   Gaussian Distribution:
//     \u03bc (mean) = VALUE
//     \u03c3 (std. deviation) = VALUE
//     Mean = VALUE
//     Variance = VALUE
std::istream &operator>>(std::istream &is, libhmm::GaussianDistribution &distribution) {
    try {
        std::string s, t;
        is >> s >> s;           // "Gaussian" "Distribution:"
        is >> s >> s >> s >> t; // "\u03bc" "(mean)" "=" VALUE
        const double mean = std::stod(t);
        is >> s >> s >> s >> s >> t; // "\u03c3" "(std." "deviation)" "=" VALUE
        const double sd = std::stod(t);
        is >> s >> s >> t;
        is >> s >> s >> t; // skip Mean, Variance
        if (is.good())
            distribution.setParameters(mean, sd);
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

bool GaussianDistribution::operator==(const GaussianDistribution &other) const {
    using namespace libhmm::constants;
    return std::abs(mean_ - other.mean_) < precision::LIMIT_TOLERANCE &&
           std::abs(standardDeviation_ - other.standardDeviation_) < precision::LIMIT_TOLERANCE;
}

// =============================================================================
// Batch log-PDF — dispatched at runtime to the best ISA kernel.
// The free function and its #if chain have been extracted to the five
// simd_double_ops_*.cpp TUs and a dispatch table in simd_dispatch.cpp.
// get_double_vec_ops().gaussian_batch is selected once at startup via CPUID.
// =============================================================================
namespace detail {

// Retained for callers that use the old detail:: name (internal only).
void gaussian_logpdf_batch(const double *obs, double *out, std::size_t n, double mean,
                           double neg_half_inv_sigma_sq, double log_norm) noexcept {
    performance::get_double_vec_ops().gaussian_batch(obs, out, n, mean, neg_half_inv_sigma_sq,
                                                     log_norm);
}

} // namespace detail

double GaussianDistribution::sample(std::mt19937_64 &rng) const {
    std::normal_distribution<double> dist(mean_, standardDeviation_);
    return dist(rng);
}

void GaussianDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                    std::span<double> out) const {
    ensureCache();
    const double log_norm = -0.5 * math::LN_2PI - logStandardDeviation_;
    performance::get_double_vec_ops().gaussian_batch(observations.data(), out.data(),
                                                     observations.size(), mean_,
                                                     negHalfSigmaSquaredInv_, log_norm);
}

std::string GaussianDistribution::to_json() const {
    return json::write_distribution("Gaussian", {{"mu", mean_}, {"sigma", standardDeviation_}});
}
std::unique_ptr<EmissionDistribution> GaussianDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double mu = r.read_double();
    r.read_key();
    const double sigma = r.read_double();
    r.consume('}');
    return std::make_unique<GaussianDistribution>(mu, sigma);
}

} // namespace libhmm
