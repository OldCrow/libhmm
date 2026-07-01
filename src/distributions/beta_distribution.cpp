#include "libhmm/distributions/beta_distribution.h"
#include "libhmm/io/json_utils.h"
#include "libhmm/math/psi_functions.h"
#include "libhmm/math/weighted_stats.h"
#include <algorithm>
#include <numeric>
#include <span>

using namespace libhmm::constants;

namespace libhmm {

/**
 * Computes the probability density function for the Beta distribution.
 *
 * @param value The value at which to evaluate the PDF (should be in [0,1])
 * @return Probability density, or 0.0 if value is outside [0,1]
 */
double BetaDistribution::getProbability(double value) const {
    // Beta distribution is only defined on [0,1]
    if (value < math::ZERO_DOUBLE || value > math::ONE || std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }

    // Update cache if needed
    ensureCache();
    // Handle boundary cases - use cached invBeta_
    if (value == math::ZERO_DOUBLE) {
        return (alpha_ == math::ONE) ? invBeta_ : math::ZERO_DOUBLE;
    }
    if (value == math::ONE) {
        return (beta_ == math::ONE) ? invBeta_ : math::ZERO_DOUBLE;
    }

    // For efficiency, use direct computation when both exponents are integers
    // and small, otherwise use optimized log-space computation
    if (alphaMinus1_ == std::floor(alphaMinus1_) && betaMinus1_ == std::floor(betaMinus1_) &&
        alphaMinus1_ >= precision::ZERO && betaMinus1_ >= precision::ZERO &&
        alphaMinus1_ <= math::FOUR && betaMinus1_ <= math::FOUR) {

        // Direct computation for small integer exponents (faster than log/exp)
        double result = invBeta_;

        // Use efficient binary exponentiation for integer powers
        int alphaExp = static_cast<int>(alphaMinus1_);
        int betaExp = static_cast<int>(betaMinus1_);

        // Binary exponentiation for x^(α-1)
        auto fastPower = [](double base, int exp) -> double {
            if (exp == 0)
                return 1.0;
            if (exp == 1)
                return base;
            if (exp == 2)
                return base * base;
            if (exp == 3)
                return base * base * base;
            if (exp == 4) {
                double sq = base * base;
                return sq * sq;
            }

            // For larger powers, use binary exponentiation
            double result = 1.0;
            double currentPower = base;
            while (exp > 0) {
                if (exp & 1)
                    result *= currentPower;
                currentPower *= currentPower;
                exp >>= 1;
            }
            return result;
        };

        double xPower = fastPower(value, alphaExp);
        double oneMinusXPower = fastPower(math::ONE - value, betaExp);

        return result * xPower * oneMinusXPower;
    } else {
        // Use optimized log-space computation for general case
        // f(x) = x^(α-1) * (1-x)^(β-1) / B(α,β)
        // Use cached values to avoid repeated calculations
        return invBeta_ * std::pow(value, alphaMinus1_) * std::pow(math::ONE - value, betaMinus1_);
    }
}

/**
 * Computes the logarithm of the probability density function for numerical stability.
 *
 * For Beta distribution: log(f(x)) = (α-1)log(x) + (β-1)log(1-x) - log(B(α,β))
 *
 * @param value The value at which to evaluate the log-PDF (should be in [0,1])
 * @return Natural logarithm of the probability density, or -∞ for invalid values
 */
double BetaDistribution::getLogProbability(double value) const noexcept {
    // Beta distribution is only defined on [0,1]
    if (value < math::ZERO_DOUBLE || value > math::ONE || std::isnan(value) || std::isinf(value)) {
        return -std::numeric_limits<double>::infinity();
    }

    // Update cache if needed
    ensureCache();
    // Handle boundary cases carefully
    if (value == 0.0) {
        if (alpha_ == 1.0) {
            // log(f(0)) = -log(B(1,β)) = -log(Γ(β)) = -logBeta_
            return -logBeta_;
        } else if (alpha_ > 1.0) {
            // f(0) = 0 since x^(α-1) → 0 as x → 0 for α > 1
            return -std::numeric_limits<double>::infinity();
        } else {
            // α < 1: f(0) → +∞, which should be avoided
            return -std::numeric_limits<double>::infinity();
        }
    }

    if (value == 1.0) {
        if (beta_ == 1.0) {
            // log(f(1)) = -log(B(α,1)) = -log(Γ(α)) = -logBeta_
            return -logBeta_;
        } else if (beta_ > 1.0) {
            // f(1) = 0 since (1-x)^(β-1) → 0 as x → 1 for β > 1
            return -std::numeric_limits<double>::infinity();
        } else {
            // β < 1: f(1) → +∞, which should be avoided
            return -std::numeric_limits<double>::infinity();
        }
    }

    // For interior points: log(f(x)) = (α-1)log(x) + (β-1)log(1-x) - log(B(α,β))
    // Use cached values for maximum efficiency
    return alphaMinus1_ * std::log(value) + betaMinus1_ * std::log(1.0 - value) - logBeta_;
}

namespace {

[[nodiscard]] std::pair<double, double> beta_mle_solve(double mean_log_x,
                                                       double mean_log_one_minus_x,
                                                       double init_alpha,
                                                       double init_beta) noexcept {
    double alpha = init_alpha;
    double beta = init_beta;

    for (int iter = 0; iter < 200; ++iter) {
        const double psi_sum = detail::digamma(alpha + beta);
        const double trigamma_sum = detail::trigamma(alpha + beta);

        const double grad_alpha = detail::digamma(alpha) - psi_sum - mean_log_x;
        const double grad_beta = detail::digamma(beta) - psi_sum - mean_log_one_minus_x;
        const double h_alpha = detail::trigamma(alpha) - trigamma_sum;
        const double h_beta = detail::trigamma(beta) - trigamma_sum;

        if (h_alpha <= 0.0 || h_beta <= 0.0)
            break;

        const double next_alpha = std::max(alpha - grad_alpha / h_alpha, 1e-10);
        const double next_beta = std::max(beta - grad_beta / h_beta, 1e-10);
        const double delta = std::fabs(next_alpha - alpha) + std::fabs(next_beta - beta);

        alpha = next_alpha;
        beta = next_beta;

        if (delta < 1e-10 * (alpha + beta))
            break;
    }

    return {alpha, beta};
}

} // anonymous namespace

double BetaDistribution::sample(std::mt19937_64 &rng) const {
    // Beta(alpha, beta) via the Gamma-ratio method:
    // X ~ Gamma(alpha, 1), Y ~ Gamma(beta, 1),  result = X / (X + Y).
    std::gamma_distribution<double> gx(alpha_, 1.0);
    std::gamma_distribution<double> gy(beta_, 1.0);
    const double x = gx(rng);
    const double y = gy(rng);
    const double s = x + y;
    return (s > 0.0) ? x / s : 0.5;
}

void BetaDistribution::fit(std::span<const double> data) {
    if (data.size() < 2) {
        reset();
        return;
    }

    double mean = 0.0;
    double M2 = 0.0;
    double sum_log_x = 0.0;
    double sum_log_one_minus_x = 0.0;
    std::size_t count = 0;

    for (const double val : data) {
        if (val < 0.0 || val > 1.0 || std::isnan(val) || std::isinf(val))
            throw std::invalid_argument(
                "Beta distribution fitting requires all values to be in [0,1]");

        ++count;
        const double delta = val - mean;
        mean += delta / static_cast<double>(count);
        M2 += delta * (val - mean);

        const double clamped = std::clamp(val, 1e-15, 1.0 - 1e-15);
        sum_log_x += std::log(clamped);
        sum_log_one_minus_x += std::log(1.0 - clamped);
    }

    if (count < 2) {
        reset();
        return;
    }

    const double n = static_cast<double>(count);
    const double variance = M2 / (n - 1.0);
    if (variance <= precision::ZERO || mean <= precision::ZERO || mean >= math::ONE) {
        reset();
        return;
    }

    const double factor = mean * (math::ONE - mean) / variance - math::ONE;
    if (factor <= precision::ZERO) {
        reset();
        return;
    }

    const auto [new_alpha, new_beta] = beta_mle_solve(sum_log_x / n, sum_log_one_minus_x / n,
                                                      mean * factor, (math::ONE - mean) * factor);

    if (std::isfinite(new_alpha) && std::isfinite(new_beta) && new_alpha > precision::ZERO &&
        new_beta > precision::ZERO && new_alpha < thresholds::MAX_DISTRIBUTION_PARAMETER &&
        new_beta < thresholds::MAX_DISTRIBUTION_PARAMETER) {
        alpha_ = new_alpha;
        beta_ = new_beta;
        invalidateCache();
    } else {
        reset();
    }
}

void BetaDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    // Guard: near-zero total weight → keep current parameters (not reset).
    if (sumW < precision::ZERO || std::isnan(sumW))
        return;

    double sum_wx = 0.0;
    double sum_wx2 = 0.0;
    double sum_wlog_x = 0.0;
    double sum_wlog_one_minus_x = 0.0;
    double cumW = 0.0;

    for (std::size_t i = 0; i < data.size(); ++i) {
        const double value = data[i];
        const double weight = weights[i];
        if (!std::isfinite(value) || value < 0.0 || value > 1.0 || !std::isfinite(weight) ||
            weight <= 0.0) {
            continue;
        }

        cumW += weight;
        sum_wx += weight * value;
        sum_wx2 += weight * value * value;

        const double clamped = std::clamp(value, 1e-15, 1.0 - 1e-15);
        sum_wlog_x += weight * std::log(clamped);
        sum_wlog_one_minus_x += weight * std::log(1.0 - clamped);
    }

    if (cumW < precision::ZERO) {
        reset();
        return;
    }

    const double mean = sum_wx / cumW;
    const double variance = sum_wx2 / cumW - mean * mean;
    if (variance <= precision::ZERO || mean <= precision::ZERO || mean >= math::ONE) {
        reset();
        return;
    }

    const double factor = mean * (math::ONE - mean) / variance - math::ONE;
    if (factor <= precision::ZERO) {
        reset();
        return;
    }

    const auto [new_alpha, new_beta] = beta_mle_solve(
        sum_wlog_x / cumW, sum_wlog_one_minus_x / cumW, mean * factor, (math::ONE - mean) * factor);

    if (std::isfinite(new_alpha) && std::isfinite(new_beta) && new_alpha > precision::ZERO &&
        new_beta > precision::ZERO && new_alpha < thresholds::MAX_DISTRIBUTION_PARAMETER &&
        new_beta < thresholds::MAX_DISTRIBUTION_PARAMETER) {
        alpha_ = new_alpha;
        beta_ = new_beta;
        invalidateCache();
    } else {
        reset();
    }
}

void BetaDistribution::reset() noexcept {
    alpha_ = 1.0;
    beta_ = 1.0;
    invalidateCache();
}

std::string BetaDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Beta Distribution:\n";
    oss << "      α (alpha) = " << alpha_ << "\n";
    oss << "      β (beta) = " << beta_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

double BetaDistribution::getCumulativeProbability(double value) const noexcept {
    // Handle boundary cases
    if (value <= 0.0)
        return 0.0;
    if (value >= 1.0)
        return 1.0;
    if (std::isnan(value) || std::isinf(value))
        return 0.0;

    // Use the incomplete Beta function I_x(α, β)
    // CDF(x) = I_x(α, β)
    return incompleteBeta(value, alpha_, beta_);
}

std::ostream &operator<<(std::ostream &os, const BetaDistribution &distribution) {
    return os << distribution.toString();
}

// Parses the format produced by toString() / operator<<:
//   Beta Distribution:
//     \u03b1 (alpha) = VALUE
//     \u03b2 (beta) = VALUE
//     Mean = VALUE
//     Variance = VALUE
std::istream &operator>>(std::istream &is, BetaDistribution &distribution) {
    try {
        std::string s, t;
        is >> s >> s;           // "Beta" "Distribution:"
        is >> s >> s >> s >> t; // "\u03b1" "(alpha)" "=" VALUE
        const double alpha = std::stod(t);
        is >> s >> s >> s >> t; // "\u03b2" "(beta)" "=" VALUE
        const double beta = std::stod(t);
        is >> s >> s >> t;
        is >> s >> s >> t; // skip Mean, Variance
        if (is.good())
            distribution = BetaDistribution(alpha, beta);
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

void BetaDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                std::span<double> out) const {
    // Tier 1 — concrete non-virtual loop; compiler auto-vectorizes the arithmetic
    // terms under -march=native. Index loop preserved: a std::ranges::transform
    // lambda would add an indirect call boundary that inhibits auto-vectorisation.
    // Tier 2 upgrade requires vectorised lgamma (log B(α,β) = lgamma(α)+lgamma(β)-lgamma(α+β)):
    // available via Intel SVML or platform-specific math libraries, but not
    // portably available without a dedicated math-library dependency.
    ensureCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        out[i] = BetaDistribution::getLogProbability(observations[i]);
    }
}

std::string BetaDistribution::to_json() const {
    return json::write_distribution("Beta", {{"alpha", alpha_}, {"beta", beta_}});
}
std::unique_ptr<EmissionDistribution> BetaDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double alpha = r.read_double();
    r.read_key();
    const double beta = r.read_double();
    r.consume('}');
    return std::make_unique<BetaDistribution>(alpha, beta);
}

} // namespace libhmm
