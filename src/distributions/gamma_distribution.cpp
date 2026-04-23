#include "libhmm/distributions/gamma_distribution.h"
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

/**
 * Returns the value of the LOWER INCOMPLETE gamma function given a and x.
 */
double GammaDistribution::ligamma(double a, double x) noexcept {
    return std::exp(std::log(gammap(a, x)) + std::lgamma(a));
}

/**
 * Fits the distribution parameters to the given data using method of moments estimation.
 * 
 * Method of moments uses:
 * sample_mean = k*θ
 * sample_variance = k*θ²
 * 
 * Solving: θ = sample_variance/sample_mean, k = sample_mean²/sample_variance
 * 
 * This is more numerically stable than MLE approximations for the Gamma distribution.
 * 
 * @param values Vector of observed data points
 */
void GammaDistribution::fit(std::span<const double> data) {
    if (data.size() < 2) {
        reset();
        return;
    }
    double mean = 0.0, m2 = 0.0;
    std::size_t count = 0;
    for (const double val : data) {
        if (val > 0.0 && std::isfinite(val)) {
            ++count;
            const double delta = val - mean;
            mean += delta / static_cast<double>(count);
            m2 += delta * (val - mean);
        }
    }
    if (count < 2) {
        reset();
        return;
    }
    const double var = m2 / (static_cast<double>(count) - 1.0);
    if (mean <= precision::ZERO || var <= precision::ZERO) {
        reset();
        return;
    }
    const double newTheta = var / mean, newK = (mean * mean) / var;
    if (!std::isfinite(newK) || !std::isfinite(newTheta) || newK <= 0.0 || newTheta <= 0.0) {
        reset();
        return;
    }
    theta_ = newTheta;
    k_ = newK;
    invalidateCache();
}

void GammaDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    double sumW = 0.0;
    for (const double w : weights)
        sumW += w;
    if (sumW < precision::ZERO || std::isnan(sumW)) {
        reset();
        return;
    }
    double mean = 0.0, m2 = 0.0, cumW = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i] > 0.0 && std::isfinite(data[i]) && weights[i] > 0.0) {
            cumW += weights[i];
            const double delta = data[i] - mean;
            mean += (weights[i] / cumW) * delta;
            m2 += weights[i] * delta * (data[i] - mean);
        }
    }
    if (cumW < precision::ZERO) {
        reset();
        return;
    }
    const double var = m2 / cumW;
    if (mean <= precision::ZERO || var <= precision::ZERO) {
        reset();
        return;
    }
    const double newTheta = var / mean, newK = (mean * mean) / var;
    if (!std::isfinite(newK) || !std::isfinite(newTheta) || newK <= 0.0 || newTheta <= 0.0) {
        reset();
        return;
    }
    theta_ = newTheta;
    k_ = newK;
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
    os << "Gamma Distribution: " << std::endl;
    os << "    k (shape) = " << distribution.getK() << std::endl;
    os << "    theta (scale) = " << distribution.getTheta() << std::endl;
    os << "    Mean = " << distribution.getMean() << std::endl;
    os << "    Variance = " << distribution.getVariance() << std::endl;

    return os;
}

std::istream &operator>>(std::istream &is, libhmm::GammaDistribution &distribution) {
    try {
        std::string token, k_str, theta_str;
        is >> token; // "k"
        is >> token; // "(shape)"
        is >> token; // "="
        is >> k_str;
        double k = std::stod(k_str);

        is >> token; // "theta"
        is >> token; // "(scale)"
        is >> token; // "="
        is >> theta_str;
        double theta = std::stod(theta_str);

        // Use setParameters for validation
        distribution.setParameters(k, theta);

    } catch (const std::exception &) {
        // Set error state on stream if parsing fails
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
    // terms under -march=native / /arch:AVX512.
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

} // namespace libhmm
