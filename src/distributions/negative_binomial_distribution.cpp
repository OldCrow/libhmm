#include "libhmm/distributions/negative_binomial_distribution.h"
#include "libhmm/io/json_utils.h"
// Header already includes: <iostream>, <sstream>, <iomanip>, <cmath>, <cassert>, <stdexcept> via common.h
#include <numeric>   // For std::accumulate (not in common.h)
#include <algorithm> // For std::for_each (exists in common.h, included for clarity)
#include <vector>    // For collecting valid k-values and weights
#include "libhmm/math/psi_functions.h" // digamma/trigamma for MLE Newton solver

using namespace libhmm::constants;

namespace libhmm {

/**
 * Computes the probability mass function for the Negative Binomial distribution.
 *
 * For discrete distributions, this returns the exact probability mass
 * P(X = k) = C(k+r-1, k) * p^r * (1-p)^k
 *
 * @param value The value at which to evaluate the PMF (rounded to nearest integer)
 * @return Probability mass for the given value
 */
double NegativeBinomialDistribution::getProbability(double value) const {
    // Validate input - discrete distributions only accept non-negative integer values
    if (std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }

    // Round to nearest integer and check if it's in valid range
    auto k = static_cast<int>(std::round(value));
    if (k < 0) {
        return math::ZERO_DOUBLE;
    }

    // Handle edge cases
    if (p_ == math::ONE) {
        return (k == 0) ? math::ONE : math::ZERO_DOUBLE;
    }

    if (!isCacheValid())
        updateCache();
    const double logCoeff = logGeneralizedBinomialCoefficient(k);
    const double logProb = logCoeff + r_ * logP_ + static_cast<double>(k) * log1MinusP_;
    const double prob = std::exp(logProb);
    if (std::isnan(prob) || prob < math::ZERO_DOUBLE)
        return math::ZERO_DOUBLE;
    return std::min(prob, math::ONE);
}

namespace {

/// Solve NB MLE: Newton–Raphson on the profile score for r; p = r/(r+k̄) is closed form.
///
/// Profile score: f(r) = (1/W) Σ w_i [ψ(k_i+r) − ψ(r)] + log r − log(r + k̄) = 0
/// Derivative:    f'(r) = (1/W) Σ w_i [ψ'(k_i+r) − ψ'(r)] + k̄/(r(r+k̄))
///
/// k_vals[i] = round(data[i]) ≥ 0, stored as double.
/// weights[i] = w_i; empty span → unit weights (sumW = n).
/// sumW = Σ w_i.  k_bar = Σ w_i k_vals[i] / sumW.
[[nodiscard]] std::pair<double, double> nb_mle_solve(std::span<const double> k_vals,
                                                     std::span<const double> weights, double sumW,
                                                     double k_bar, double init_r) noexcept {
    const std::size_t n = k_vals.size();
    const bool unit_w = weights.empty();
    double r = init_r;

    for (int iter = 0; iter < 200; ++iter) {
        const double psi_r = detail::digamma(r);
        const double tpsi_r = detail::trigamma(r);

        double sum_f = 0.0, sum_fp = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double w = unit_w ? 1.0 : weights[i];
            const double kr = k_vals[i] + r;
            sum_f += w * (detail::digamma(kr) - psi_r);
            sum_fp += w * (detail::trigamma(kr) - tpsi_r);
        }

        const double f = sum_f / sumW + std::log(r) - std::log(r + k_bar);
        const double fp = sum_fp / sumW + k_bar / (r * (r + k_bar));
        if (std::fabs(fp) < 1e-15)
            break;

        const double dr = f / fp;
        r -= dr;
        if (r <= 0.0)
            r = 1e-6;
        if (std::fabs(dr) < 1e-11 * r)
            break;
    }

    return {r, r / (r + k_bar)};
}

} // anonymous namespace

double NegativeBinomialDistribution::sample(std::mt19937_64 &rng) const {
    // NegBin(r, p) via Gamma-Poisson mixture; supports real-valued r.
    // lambda ~ Gamma(r, (1-p)/p)  then  k ~ Poisson(lambda).
    std::gamma_distribution<double> gamma_dist(r_, (1.0 - p_) / p_);
    const double lambda = gamma_dist(rng);
    std::poisson_distribution<int> poisson_dist(lambda);
    return static_cast<double>(poisson_dist(rng));
}

void NegativeBinomialDistribution::fit(std::span<const double> data) {
    if (data.size() < 2) {
        reset();
        return;
    }

    std::vector<double> k_vals;
    k_vals.reserve(data.size());

    double mean = 0.0, M2 = 0.0;
    std::size_t count = 0;

    for (const double val : data) {
        if (val >= 0.0 && std::isfinite(val)) {
            const double k = std::round(val);
            ++count;
            const double delta = k - mean;
            mean += delta / static_cast<double>(count);
            M2 += delta * (k - mean);
            k_vals.push_back(k);
        }
    }

    if (count < 2) {
        reset();
        return;
    }

    const double n = static_cast<double>(count);
    const double var = M2 / (n - 1.0);
    const double k_bar = mean;

    if (var <= k_bar || k_bar <= math::ZERO_DOUBLE) {
        reset();
        return;
    }

    const double r_init = (k_bar * k_bar) / (var - k_bar); // MoM seed
    const auto [r, p] = nb_mle_solve(k_vals, {}, n, k_bar, r_init);

    if (std::isfinite(r) && std::isfinite(p) && r > precision::ZERO && p > precision::ZERO &&
        p <= math::ONE) {
        r_ = r;
        p_ = p;
        invalidateCache();
    } else {
        reset();
    }
}

void NegativeBinomialDistribution::fit(std::span<const double> data,
                                       std::span<const double> weights) {
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    // Guard: keep current parameters when effective weight is near zero.
    // Calling reset() would destroy valid parameters and cause state collapse in EM.
    if (sumW < precision::ZERO || std::isnan(sumW))
        return;

    std::vector<double> k_vals, wt;
    k_vals.reserve(data.size());
    wt.reserve(data.size());

    double sum_wk = 0.0, sum_wk2 = 0.0, cumW = 0.0;

    for (std::size_t i = 0; i < data.size(); ++i) {
        const double val = data[i];
        const double weight = weights[i];
        if (val < 0.0 || !std::isfinite(val) || !std::isfinite(weight) || weight <= 0.0)
            continue;
        const double k = std::round(val);
        cumW += weight;
        sum_wk += weight * k;
        sum_wk2 += weight * k * k;
        k_vals.push_back(k);
        wt.push_back(weight);
    }

    if (cumW < precision::ZERO || k_vals.empty()) {
        reset();
        return;
    }

    const double k_bar = sum_wk / cumW;
    const double var = sum_wk2 / cumW - k_bar * k_bar;

    if (var <= k_bar || k_bar <= math::ZERO_DOUBLE) {
        reset();
        return;
    }

    const double r_init = (k_bar * k_bar) / (var - k_bar); // MoM seed
    const auto [r, p] = nb_mle_solve(k_vals, wt, cumW, k_bar, r_init);

    if (std::isfinite(r) && std::isfinite(p) && r > precision::ZERO && p > precision::ZERO &&
        p <= math::ONE) {
        r_ = r;
        p_ = p;
        invalidateCache();
    } else {
        reset();
    }
}

/**
 * Resets the distribution to default parameters (r = 5.0, p = 0.5).
 * This corresponds to a moderate negative binomial distribution.
 */
void NegativeBinomialDistribution::reset() noexcept {
    r_ = 5.0;
    p_ = math::HALF;
    invalidateCache();
}

/**
 * Returns a string representation of the distribution following the standardized format.
 *
 * @return String describing the distribution parameters and statistics
 */
std::string NegativeBinomialDistribution::toString() const {
    std::ostringstream oss{};
    oss << std::fixed << std::setprecision(6);
    oss << "NegativeBinomial Distribution:\n";
    oss << "      r (successes) = " << r_ << "\n";
    oss << "      p (success probability) = " << p_ << "\n";
    oss << "      Mean = " << getMean() << "\n";
    oss << "      Variance = " << getVariance() << "\n";
    return oss.str();
}

double NegativeBinomialDistribution::getLogProbability(double value) const noexcept {
    // Validate input - discrete distributions only accept non-negative integer values
    if (std::isnan(value) || std::isinf(value)) {
        return -std::numeric_limits<double>::infinity();
    }

    // Round to nearest integer and check if it's in valid range
    auto k = static_cast<int>(std::round(value));
    if (k < 0) {
        return -std::numeric_limits<double>::infinity();
    }

    // Handle edge cases
    if (p_ == math::ONE) {
        return (k == 0) ? math::ZERO_DOUBLE : -std::numeric_limits<double>::infinity();
    }

    if (!isCacheValid())
        updateCache();
    const double logCoeff = logGeneralizedBinomialCoefficient(k);
    return logCoeff + r_ * logP_ + static_cast<double>(k) * log1MinusP_;
}

double NegativeBinomialDistribution::getCumulativeProbability(double value) const noexcept {
    // Validate input
    if (std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }

    auto k = static_cast<int>(std::floor(value));

    // Handle boundary cases
    if (k < 0) {
        return math::ZERO_DOUBLE;
    }

    // Compute CDF as cumulative sum: P(X <= k) = sum_{i=0}^{k} P(X = i)
    // For efficiency, we limit computation to reasonable range
    const int maxK = std::min(k, 1000); // Practical upper limit for computation

    double cdf = math::ZERO_DOUBLE;
    for (int i = 0; i <= maxK; ++i) {
        cdf += getProbability(static_cast<double>(i));
    }

    return std::min(math::ONE, cdf);
}

bool NegativeBinomialDistribution::operator==(const NegativeBinomialDistribution &other) const {
    const double tolerance = 1e-10;
    return (std::abs(r_ - other.r_) < tolerance) && (std::abs(p_ - other.p_) < tolerance);
}

// Parses the format produced by toString() / operator<<:
//   Negative Binomial Distribution:
//     r (successes) = VALUE
//     p (success probability) = VALUE
//     Mean = VALUE
//     Variance = VALUE
std::istream &operator>>(std::istream &is, libhmm::NegativeBinomialDistribution &distribution) {
    try {
        std::string s, t;
        is >> s >> s;           // "NegativeBinomial" "Distribution:"
        is >> s >> s >> s >> t; // "r" "(successes)" "=" VALUE
        const double r = std::stod(t);
        is >> s >> s >> s >> s >> t; // "p" "(success" "probability)" "=" VALUE
        const double p = std::stod(t);
        is >> s >> s >> t;
        is >> s >> s >> t; // skip Mean, Variance
        if (is.good())
            distribution.setParameters(r, p);
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

std::ostream &operator<<(std::ostream &os,
                         const libhmm::NegativeBinomialDistribution &distribution) {
    os << distribution.toString();
    return os;
}

void NegativeBinomialDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                            std::span<double> out) const {
    // Tier 1 — concrete non-virtual loop; compiler auto-vectorizes the arithmetic
    // terms under -march=native. Index loop preserved: a std::ranges::transform
    // lambda would add an indirect call boundary that inhibits auto-vectorisation.
    // Tier 2 upgrade requires vectorised generalised log-binomial-coefficient
    // (uses lgamma internally): available via Intel SVML or platform-specific
    // math libraries, but not portably without a math-library dependency.
    if (!isCacheValid())
        updateCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        out[i] = NegativeBinomialDistribution::getLogProbability(observations[i]);
    }
}

std::string NegativeBinomialDistribution::to_json() const {
    return json::write_distribution("NegativeBinomial", {{"r", r_}, {"p", p_}});
}
std::unique_ptr<EmissionDistribution> NegativeBinomialDistribution::from_json(json::Reader &r) {
    r.read_key();
    const double rv = r.read_double();
    r.read_key();
    const double p = r.read_double();
    r.consume('}');
    return std::make_unique<NegativeBinomialDistribution>(rv, p);
}

} // namespace libhmm
