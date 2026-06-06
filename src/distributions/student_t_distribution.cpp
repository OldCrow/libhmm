#include "libhmm/distributions/student_t_distribution.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/math/psi_functions.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <span>

using namespace libhmm::constants;

namespace libhmm {

StudentTDistribution::StudentTDistribution()
    : degrees_of_freedom_(1.0), location_(0.0), scale_(1.0) {}

StudentTDistribution::StudentTDistribution(double degrees_of_freedom)
    : degrees_of_freedom_(degrees_of_freedom), location_(0.0), scale_(1.0) {
    validateParameters(degrees_of_freedom);
    updateCache();
}

StudentTDistribution::StudentTDistribution(double degrees_of_freedom, double location, double scale)
    : degrees_of_freedom_(degrees_of_freedom), location_(location), scale_(scale) {
    validateParameters(degrees_of_freedom);
    if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0)
        throw std::invalid_argument("Scale parameter must be a positive finite number");
    updateCache();
}

double StudentTDistribution::getProbability(double value) const {
    if (!isCacheValid())
        updateCache();
    if (!std::isfinite(value))
        return math::ZERO_DOUBLE;
    const double x = value;

    // Direct calculation for better performance
    // Standardize with location and scale: z = (x - μ) / σ
    const double z = (x - location_) * cached_inv_scale_;
    const double z_squared_over_nu = (z * z) / degrees_of_freedom_;

    // Direct PDF calculation using cached normalization constant:
    // f(x) = normalization_factor * (1 + z²/ν)^(-(ν+1)/2)
    const double power_term = std::pow(math::ONE + z_squared_over_nu, -cached_half_nu_plus_one_);

    return cached_normalization_factor_ * power_term;
}

/**
 * Computes the logarithm of the probability density function for numerical stability.
 */
double StudentTDistribution::getLogProbability(double value) const noexcept {
    if (!std::isfinite(value))
        return -std::numeric_limits<double>::infinity();
    if (!isCacheValid())
        updateCache();
    const double z = (value - location_) * cached_inv_scale_;
    // Optimized log PDF
    // log(f(x|ν,μ,σ)) = cached_log_normalization - ((ν+1)/2) * log(1 + z²/ν)
    const double z_squared_over_nu = (z * z) / degrees_of_freedom_;
    const double log_denominator_term =
        cached_half_nu_plus_one_ * std::log(math::ONE + z_squared_over_nu);

    return cached_log_normalization_ - log_denominator_term;
}

/**
 * Computes the cumulative distribution function for the Student's t-distribution.
 *
 * Uses the relationship with the incomplete beta function for numerical accuracy.
 */
double StudentTDistribution::getCumulativeProbability(double value) const noexcept {
    if (!std::isfinite(value)) {
        return std::isnan(value) ? std::numeric_limits<double>::quiet_NaN()
                                 : (value < math::ZERO_DOUBLE ? math::ZERO_DOUBLE : math::ONE);
    }

    if (!isCacheValid())
        updateCache();
    const double t = (value - location_) * cached_inv_scale_;

    // Exact CDF via regularised incomplete beta:
    //
    //   P(T ≤ t; ν) = 1 − ½·I_{x_b}(ν/2, 1/2)   for t > 0
    //   P(T ≤ t; ν) =     ½·I_{x_b}(ν/2, 1/2)   for t < 0
    //
    // where x_b = ν / (ν + t²) and I_x is the regularised incomplete beta.
    // The formula holds for all ν > 0 without special-casing moderate vs large ν.
    if (std::abs(t) < 1e-15)
        return math::HALF;

    const double x_b = degrees_of_freedom_ / (degrees_of_freedom_ + t * t);
    const double ib  = incompleteBeta(x_b, math::HALF * degrees_of_freedom_, math::HALF);
    return t > math::ZERO_DOUBLE ? math::ONE - math::HALF * ib : math::HALF * ib;
}

void StudentTDistribution::fit(std::span<const double> data) {
    if (data.size() < 2) {
        reset();
        return;
    }

    const double nu = degrees_of_freedom_;
    const double mu = location_;
    const double sig = scale_;
    const double n = static_cast<double>(data.size());

    // ECME E-step: e_i = (ν+1) / (ν + z_i²) with current (μ, σ, ν).
    // Sufficient statistics for all three CM-steps — one pass.
    //
    // For the ν CM-step we need mean[E[log U_i]] where U_i|x_i ~ Gamma((ν+1)/2, (ν+z_i²)/2):
    //   E[log U_i] = ψ((ν+1)/2) − log((ν+z_i²)/2)
    // We accumulate sum_logz = Σ log((ν+z_i²)/2); the ψ term is constant over i.
    // NOTE: log(e_i) ≠ E[log U_i] — Jensen's inequality makes log(E[U]) ≥ E[log U],
    // so using log(e_i) drives ν → ∞ unconditionally.
    double sum_e = 0.0, sum_ex = 0.0, sum_exx = 0.0, sum_logz = 0.0;
    for (const double val : data) {
        if (!std::isfinite(val))
            throw std::invalid_argument("Observations contain non-finite values");
        const double z = (val - mu) / sig;
        const double zz = z * z;
        const double e = (nu + 1.0) / (nu + zz);
        sum_e += e;
        sum_ex += e * val;
        sum_exx += e * val * val;
        sum_logz += std::log(0.5 * (nu + zz)); // log((ν+z²)/2)
    }
    if (sum_e < precision::ZERO) {
        reset();
        return;
    }

    // CM-step 1: μ (WLS with weights e_i)
    const double mu_new = sum_ex / sum_e;

    // CM-step 2: σ (König–Huygens; denominator is n not sum_e)
    const double sig2_new = (sum_exx - sum_ex * sum_ex / sum_e) / n;
    if (!std::isfinite(sig2_new) || sig2_new <= 0.0) {
        reset();
        return;
    }
    const double sig_new = std::sqrt(sig2_new);

    // CM-step 3: ν via Newton–Raphson on the ECM score equation:
    // f(ν) = log(ν/2) + 1 − ψ(ν/2) + c = 0
    // c = mean[E[log U_i]] − mean[e_i]
    //   = ψ((ν_old+1)/2) − mean[log((ν_old+z_i²)/2)] − mean[e_i]
    // f'(ν) = 1/ν − (1/2) ψ'(ν/2)
    const double c = detail::digamma(0.5 * (nu + 1.0)) - sum_logz / n - sum_e / n;
    double nu_new = nu;
    for (int i = 0; i < 30; ++i) {
        const double half_nu = 0.5 * nu_new;
        const double f = std::log(half_nu) + 1.0 - detail::digamma(half_nu) + c;
        const double fp = 1.0 / nu_new - 0.5 * detail::trigamma(half_nu);
        if (std::fabs(fp) < 1e-15)
            break;
        const double dnu = f / fp;
        nu_new -= dnu;
        if (nu_new <= 0.0)
            nu_new = 0.1;
        if (std::fabs(dnu) < 1e-10 * nu_new)
            break;
    }
    nu_new = std::clamp(nu_new, constants::thresholds::MIN_DEGREES_OF_FREEDOM,
                        constants::thresholds::MAX_DEGREES_OF_FREEDOM);

    location_ = mu_new;
    scale_ = sig_new;
    degrees_of_freedom_ = nu_new;
    invalidateCache();
}

void StudentTDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    // Guard: keep current parameters when effective weight is near zero.
    // Resetting would destroy valid parameters and cause state collapse in EM.
    if (sumW < precision::ZERO || std::isnan(sumW))
        return;

    const double nu = degrees_of_freedom_;
    const double mu = location_;
    const double sig = scale_;

    // ECME E-step: e_i = (ν+1) / (ν + z_i²) with current (μ, σ, ν).
    // Sufficient statistics for all three CM-steps — one pass.
    //
    // For the ν CM-step we need mean_w[E[log U_i]] where U_i|x_i ~ Gamma((ν+1)/2, (ν+z_i²)/2):
    //   E[log U_i] = ψ((ν+1)/2) − log((ν+z_i²)/2)
    // We accumulate sum_wlogz = Σ w_i·log((ν+z_i²)/2); the ψ term is constant over i.
    // NOTE: log(e_i) ≠ E[log U_i] — Jensen's inequality makes log(E[U]) ≥ E[log U],
    // so using log(e_i) would make c ≤ −1 always, driving ν → ∞ unconditionally.
    double sum_we = 0.0, sum_wex = 0.0, sum_wexx = 0.0, sum_wlogz = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        const double w = weights[i];
        if (w <= 0.0 || !std::isfinite(data[i]))
            continue;
        const double z = (data[i] - mu) / sig;
        const double zz = z * z;
        const double e = (nu + 1.0) / (nu + zz);
        const double we = w * e;
        sum_we += we;
        sum_wex += we * data[i];
        sum_wexx += we * data[i] * data[i];
        sum_wlogz += w * std::log(0.5 * (nu + zz)); // log((ν+z²)/2)
    }
    if (sum_we < precision::ZERO)
        return;

    // CM-step 1: μ (WLS with combined weights w_i · e_i)
    const double mu_new = sum_wex / sum_we;

    // CM-step 2: σ (König–Huygens; denominator is sumW not sum_we)
    const double sig2_new = (sum_wexx - sum_wex * sum_wex / sum_we) / sumW;
    if (!std::isfinite(sig2_new) || sig2_new <= 0.0)
        return;
    const double sig_new = std::sqrt(sig2_new);

    // CM-step 3: ν via Newton–Raphson on the ECM score equation:
    // f(ν) = log(ν/2) + 1 − ψ(ν/2) + c = 0
    // c = mean_w[E[log U_i]] − mean_w[e_i]
    //   = ψ((ν_old+1)/2) − mean_w[log((ν_old+z_i²)/2)] − mean_w[e_i]
    // f'(ν) = 1/ν − (1/2) ψ'(ν/2)
    const double c = detail::digamma(0.5 * (nu + 1.0)) - sum_wlogz / sumW - sum_we / sumW;
    double nu_new = nu;
    for (int i = 0; i < 30; ++i) {
        const double half_nu = 0.5 * nu_new;
        const double f = std::log(half_nu) + 1.0 - detail::digamma(half_nu) + c;
        const double fp = 1.0 / nu_new - 0.5 * detail::trigamma(half_nu);
        if (std::fabs(fp) < 1e-15)
            break;
        const double dnu = f / fp;
        nu_new -= dnu;
        if (nu_new <= 0.0)
            nu_new = 0.1;
        if (std::fabs(dnu) < 1e-10 * nu_new)
            break;
    }
    nu_new = std::clamp(nu_new, constants::thresholds::MIN_DEGREES_OF_FREEDOM,
                        constants::thresholds::MAX_DEGREES_OF_FREEDOM);

    location_ = mu_new;
    scale_ = sig_new;
    degrees_of_freedom_ = nu_new;
    invalidateCache();
}

void StudentTDistribution::reset() noexcept {
    degrees_of_freedom_ = 1.0;
    location_ = 0.0;
    scale_ = 1.0;
    invalidateCache();
}

void StudentTDistribution::setDegreesOfFreedom(double degrees_of_freedom) {
    validateParameters(degrees_of_freedom);
    degrees_of_freedom_ = degrees_of_freedom;
    invalidateCache();
}

void StudentTDistribution::setScale(double scale) {
    if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) {
        throw std::invalid_argument("Scale parameter must be a positive finite number");
    }
    scale_ = scale;
    invalidateCache(); // cached_inv_scale_, cached_log_scale_, cached_log_normalization_, cached_normalization_factor_ all depend on scale_
}

double StudentTDistribution::getMean() const {
    if (degrees_of_freedom_ > 1.0) {
        return location_; // For generalized t-distribution, mean = location parameter
    } else {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

double StudentTDistribution::getVariance() const {
    if (degrees_of_freedom_ > 2.0) {
        // For generalized t-distribution: Var = σ² * ν/(ν-2)
        return scale_ * scale_ * degrees_of_freedom_ / (degrees_of_freedom_ - 2.0);
    } else if (degrees_of_freedom_ > 1.0) {
        return std::numeric_limits<double>::infinity();
    } else {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

double StudentTDistribution::getStandardDeviation() const {
    double variance = getVariance();
    if (std::isfinite(variance)) {
        return std::sqrt(variance);
    } else {
        return variance; // NaN or infinity
    }
}

std::string StudentTDistribution::toString() const {
    std::ostringstream oss;
    oss << "StudentT Distribution:\n";
    oss << "  nu (degrees of freedom) = " << std::fixed << std::setprecision(6)
        << degrees_of_freedom_ << "\n";
    oss << "  mu (location) = " << std::fixed << std::setprecision(6) << location_ << "\n";
    oss << "  sigma (scale) = " << std::fixed << std::setprecision(6) << scale_;
    return oss.str();
}

bool StudentTDistribution::operator==(const StudentTDistribution &other) const {
    return std::abs(degrees_of_freedom_ - other.degrees_of_freedom_) < PARAMETER_TOLERANCE &&
           std::abs(location_ - other.location_) < PARAMETER_TOLERANCE &&
           std::abs(scale_ - other.scale_) < PARAMETER_TOLERANCE;
}

bool StudentTDistribution::operator!=(const StudentTDistribution &other) const {
    return !(*this == other);
}

// Parses the format produced by toString() / operator<<:
//   StudentT Distribution:
//     nu (degrees of freedom) = VALUE
//     mu (location) = VALUE
//     sigma (scale) = VALUE
std::istream &operator>>(std::istream &is, StudentTDistribution &dist) {
    try {
        std::string s, t;
        is >> s >> s;                     // "StudentT" "Distribution:"
        is >> s >> s >> s >> s >> s >> t; // "nu" "(degrees" "of" "freedom)" "=" VALUE
        const double nu = std::stod(t);
        is >> s >> s >> s >> t; // "mu" "(location)" "=" VALUE
        const double mu = std::stod(t);
        is >> s >> s >> s >> t; // "sigma" "(scale)" "=" VALUE
        if (is.good())
            dist = StudentTDistribution(nu, mu, std::stod(t));
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

// Delegates to toString() — consistent with other distributions.
std::ostream &operator<<(std::ostream &os, const StudentTDistribution &dist) {
    os << dist.toString();
    return os;
}

/**
 * Validates the degrees of freedom parameter.
 */
void StudentTDistribution::validateParameters(double degrees_of_freedom) {
    if (std::isnan(degrees_of_freedom) || std::isinf(degrees_of_freedom) ||
        degrees_of_freedom <= 0.0) {
        throw std::invalid_argument("Degrees of freedom must be a positive finite number");
    }
}

/// Updates cached values for efficient repeated calculations.
/// std::lgamma and all arithmetic here are noexcept (C++17 §26.8),
/// so this method is correctly marked noexcept.
void StudentTDistribution::updateCache() const noexcept {
    // Cache frequently used fractional values
    cached_half_nu_ = degrees_of_freedom_ * math::HALF;
    cached_half_nu_plus_one_ = cached_half_nu_ + math::HALF;

    // Cache scale-related values
    cached_inv_scale_ = math::ONE / scale_;
    cached_log_scale_ = std::log(scale_);

    // log(PDF) = log(Γ((ν+1)/2)) - log(Γ(ν/2)) - (1/2)*log(νπ) - (ν+1)/2 * log(1 + t²/ν)
    cached_log_gamma_half_nu_plus_one_ = std::lgamma(cached_half_nu_plus_one_);
    cached_log_gamma_half_nu_ = std::lgamma(cached_half_nu_);

    // Pre-compute the log normalization constant:
    // log_normalization = log(Γ((ν+1)/2)) - log(Γ(ν/2)) - (1/2)*log(νπ) - log(σ)
    const double log_nu_pi = std::log(degrees_of_freedom_ * constants::math::PI);

    cached_log_normalization_ = cached_log_gamma_half_nu_plus_one_ - cached_log_gamma_half_nu_ -
                                math::HALF * log_nu_pi - cached_log_scale_;

    // Cache the exponential of the log normalization for direct PDF calculation
    cached_normalization_factor_ = std::exp(cached_log_normalization_);

    markCacheValid();
}

void StudentTDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                    std::span<double> out) const {
    // Tier 1 — concrete non-virtual loop; compiler auto-vectorizes the arithmetic
    // terms under -march=native / /arch:AVX512.
    // Tier 2 upgrade: the log-normalisation constant is precomputed in the cache,
    // so the per-element work is log(1 + t²/ν) — requires vectorised log.
    // Available via Intel SVML, GNU libmvec, or Apple Accelerate vvlog, but
    // not portably without a math-library dependency.
    if (!isCacheValid())
        updateCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        out[i] = StudentTDistribution::getLogProbability(observations[i]);
    }
}

std::string StudentTDistribution::to_json() const {
    return json::write_distribution(
        "StudentT", {{"df", degrees_of_freedom_}, {"mu", location_}, {"sigma", scale_}});
}
std::unique_ptr<EmissionDistribution> StudentTDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double df = r.read_double();
    r.read_key();
    const double mu = r.read_double();
    r.read_key();
    const double sigma = r.read_double();
    r.consume('}');
    return std::make_unique<StudentTDistribution>(df, mu, sigma);
}

} // namespace libhmm
