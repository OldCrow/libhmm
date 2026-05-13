#include "libhmm/distributions/von_mises_distribution.h"
#include "libhmm/math/bessel.h"
#include "libhmm/math/constants.h"
#include "libhmm/io/json_utils.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace libhmm {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

/// Mardia-Jupp approximation for κ given mean resultant length R̄ ∈ [0, 1).
/// Mardia & Jupp (2000), Directional Statistics, p. 85–86, §A.2.
/// Error < 0.003 on the whole range; refined by one Newton step if needed.
[[nodiscard]] double kappa_from_r_bar(double R_bar) noexcept {
    if (R_bar <= 0.0)
        return 0.0;
    if (R_bar >= 1.0)
        return 1.0e6; // effectively point mass

    double kappa;
    if (R_bar < 0.53) {
        kappa = 2.0 * R_bar + R_bar * R_bar * R_bar +
                (5.0 / 6.0) * R_bar * R_bar * R_bar * R_bar * R_bar;
    } else if (R_bar < 0.85) {
        kappa = -0.4 + 1.39 * R_bar + 0.43 / (1.0 - R_bar);
    } else {
        const double r = R_bar;
        kappa = 1.0 / (r * r * r - 4.0 * r * r + 3.0 * r);
    }

    if (kappa < 0.0)
        kappa = 0.0; // guard against rounding near edges

    // One Newton step to tighten the approximation.
    // A(κ) = I₁(κ)/I₀(κ); A'(κ) = 1 − A(κ)² − A(κ)/κ
    // Newton: κ ← κ − (A(κ) − R̄) / A'(κ)
    if (kappa > 0.0) {
        const double i0 = detail::bessel_i0(kappa);
        const double i1 = detail::bessel_i1(kappa);
        if (i0 > 0.0) {
            const double A = i1 / i0;
            const double Ap = 1.0 - A * A - A / kappa;
            if (std::fabs(Ap) > 1e-12)
                kappa -= (A - R_bar) / Ap;
            if (kappa < 0.0)
                kappa = 0.0;
        }
    }
    return kappa;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Construction and cache
// ---------------------------------------------------------------------------

VonMisesDistribution::VonMisesDistribution(double mu, double kappa)
    : mu_{wrap_angle(mu)}, kappa_{kappa} {
    validateParameters(mu_, kappa_);
    updateCache();
}

double VonMisesDistribution::wrap_angle(double x) noexcept {
    // Wrap to (−π, π]
    if (!std::isfinite(x))
        return x;
    const double twopi = constants::math::TWO_PI;
    x = std::fmod(x, twopi);
    if (x <= -constants::math::PI)
        x += twopi;
    if (x > constants::math::PI)
        x -= twopi;
    return x;
}

void VonMisesDistribution::validateParameters(double mu, double kappa) {
    if (std::isnan(mu) || std::isinf(mu))
        throw std::invalid_argument("VonMisesDistribution: mu must be a finite real number");
    if (std::isnan(kappa) || std::isinf(kappa) || kappa < 0.0)
        throw std::invalid_argument(
            "VonMisesDistribution: kappa must be a non-negative finite number");
}

void VonMisesDistribution::updateCache() const noexcept {
    // logNormaliser = log(2π) + log I₀(κ)
    logNormaliser_ = constants::math::LN_2PI + detail::log_bessel_i0(kappa_);

    // Circular variance = 1 − I₁(κ)/I₀(κ)
    // For κ = 0: I₁(0)=0, I₀(0)=1, variance = 1 (uniform).
    // For large κ: both bessel calls converge to exp(κ)/√(2πκ); ratio → 1.
    if (kappa_ < 1e-10) {
        circularVariance_ = 1.0;
    } else {
        const double i0 = detail::bessel_i0(kappa_);
        const double i1 = detail::bessel_i1(kappa_);
        circularVariance_ = (i0 > 0.0) ? 1.0 - i1 / i0 : 1.0;
    }
    markCacheValid();
}

// ---------------------------------------------------------------------------
// Probability / log-probability
// ---------------------------------------------------------------------------

double VonMisesDistribution::getProbability(double value) const {
    if (std::isnan(value) || std::isinf(value))
        return 0.0;
    if (!isCacheValid())
        updateCache();
    return std::exp(kappa_ * std::cos(value - mu_) - logNormaliser_);
}

double VonMisesDistribution::getLogProbability(double value) const noexcept {
    if (std::isnan(value) || std::isinf(value))
        return -std::numeric_limits<double>::infinity();
    if (!isCacheValid())
        updateCache();
    return kappa_ * std::cos(value - mu_) - logNormaliser_;
}

void VonMisesDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                    std::span<double> out) const {
    // Tier 1 — concrete non-virtual loop; inner cos() is the bottleneck.
    // The hot path is: kappa * cos(x - mu) - logNormaliser
    // Both kappa and mu are loop-invariant; compilers can hoist them.
    if (!isCacheValid())
        updateCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        const double x = observations[i];
        out[i] = (std::isnan(x) || std::isinf(x)) ? -std::numeric_limits<double>::infinity()
                                                  : kappa_ * std::cos(x - mu_) - logNormaliser_;
    }
}

// ---------------------------------------------------------------------------
// CDF (numerical integration, trapezoidal rule)
// ---------------------------------------------------------------------------

double VonMisesDistribution::getCumulativeProbability(double value) const noexcept {
    if (!std::isfinite(value))
        return std::isnan(value) ? 0.0 : 1.0;
    const double v = wrap_angle(value);
    if (!isCacheValid())
        updateCache();

    // Integrate f(x) from -π to v using trapezoidal rule with 512 steps.
    constexpr int N = 512;
    const double a = -constants::math::PI;
    const double h = (v - a) / static_cast<double>(N);
    if (std::fabs(h) < 1e-15)
        return 0.0;

    double sum = 0.5 * (getProbability(a) + getProbability(v));
    for (int i = 1; i < N; ++i)
        sum += getProbability(a + i * h);
    return std::clamp(sum * h, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Fitting
// ---------------------------------------------------------------------------

void VonMisesDistribution::fit(std::span<const double> data) {
    if (data.empty()) {
        reset();
        return;
    }

    double S = 0.0, C = 0.0;
    for (const double x : data) {
        S += std::sin(x);
        C += std::cos(x);
    }
    const double n = static_cast<double>(data.size());
    mu_ = wrap_angle(std::atan2(S / n, C / n));
    const double R_bar = std::sqrt(S * S + C * C) / n;
    kappa_ = kappa_from_r_bar(R_bar);
    invalidateCache();
}

void VonMisesDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (sumW < constants::precision::ZERO || std::isnan(sumW)) {
        reset();
        return;
    }

    double S = 0.0, C = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (weights[i] > 0.0 && std::isfinite(data[i])) {
            S += weights[i] * std::sin(data[i]);
            C += weights[i] * std::cos(data[i]);
        }
    }

    mu_ = wrap_angle(std::atan2(S / sumW, C / sumW));
    const double R_bar = std::sqrt(S * S + C * C) / sumW;
    kappa_ = kappa_from_r_bar(R_bar);
    invalidateCache();
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

void VonMisesDistribution::reset() noexcept {
    mu_ = 0.0;
    kappa_ = 1.0;
    invalidateCache();
}

void VonMisesDistribution::setMu(double mu) {
    validateParameters(mu, kappa_);
    mu_ = wrap_angle(mu);
    invalidateCache();
}

void VonMisesDistribution::setKappa(double kappa) {
    validateParameters(mu_, kappa);
    kappa_ = kappa;
    invalidateCache();
}

bool VonMisesDistribution::operator==(const VonMisesDistribution &o) const noexcept {
    return std::fabs(mu_ - o.mu_) < constants::precision::ULTRA_HIGH_PRECISION_TOLERANCE &&
           std::fabs(kappa_ - o.kappa_) < constants::precision::ULTRA_HIGH_PRECISION_TOLERANCE;
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------

std::string VonMisesDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Von Mises Distribution:\n";
    oss << "      μ (mean direction) = " << mu_ << "\n";
    oss << "      κ (concentration) = " << kappa_ << "\n";
    oss << "      Circular variance = " << getCircularVariance() << "\n";
    oss << "      Log normaliser = " << logNormaliser_ << "\n";
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const VonMisesDistribution &d) {
    os << d.toString();
    return os;
}

// Parses the format produced by toString():
//   Von Mises Distribution:
//     μ (mean direction) = VALUE
//     κ (concentration) = VALUE
//     Circular variance = VALUE
//     Log normaliser = VALUE
std::istream &operator>>(std::istream &is, VonMisesDistribution &d) {
    try {
        std::string s, t;
        is >> s >> s >> s;           // "Von" "Mises" "Distribution:"
        is >> s >> s >> s >> s >> t; // "μ" "(mean" "direction)" "=" VALUE
        const double mu = std::stod(t);
        is >> s >> s >> s >> t; // "κ" "(concentration)" "=" VALUE
        const double kappa = std::stod(t);
        // skip Circular variance, Log normaliser
        is >> s >> s >> s >> t;
        is >> s >> s >> s >> t;
        if (is.good())
            d = VonMisesDistribution(mu, kappa);
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

std::string VonMisesDistribution::to_json() const {
    return json::write_distribution("VonMises", {{"mu", mu_}, {"kappa", kappa_}});
}

std::unique_ptr<EmissionDistribution> VonMisesDistribution::from_json(json::Reader &r) {
    r.read_key(); // "mu"
    const double mu = r.read_double();
    r.read_key(); // "kappa"
    const double kappa = r.read_double();
    r.consume('}');
    return std::make_unique<VonMisesDistribution>(mu, kappa);
}

} // namespace libhmm
