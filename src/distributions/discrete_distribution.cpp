#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/io/json_utils.h"
#include <numeric>
#include <span>

using namespace libhmm::constants;

namespace libhmm {

/**
 * Gets the probability mass function value for a discrete observation.
 *
 * @param x The discrete value (will be cast to integer index)
 * @return Probability mass for the given value, 0.0 if out of range
 */
double DiscreteDistribution::getProbability(double x) const {
    if (std::isnan(x) || std::isinf(x) || x < math::ZERO_DOUBLE)
        return math::ZERO_DOUBLE;
    const auto index = static_cast<std::size_t>(x);
    if (!isValidIndex(index))
        return math::ZERO_DOUBLE;
    assert(pdf_[index] <= 1.0 && pdf_[index] >= 0.0);
    return pdf_[index];
}
// setProbability is now inline in the header.

/**
 * Fits the distribution to observed data using maximum likelihood estimation.
 * Computes empirical probabilities: P(X = k) = count(k) / total_count
 *
 * @param values Vector of observed discrete values
 */
void DiscreteDistribution::fit(std::span<const double> data) {
    if (data.empty()) {
        reset();
        return;
    }

    std::fill(pdf_.begin(), pdf_.end(), 0.0);
    std::size_t validCount = 0;
    for (const double val : data) {
        if (val >= 0.0) {
            const auto index = static_cast<std::size_t>(val);
            if (isValidIndex(index)) {
                pdf_[index]++;
                ++validCount;
            }
        }
    }
    if (validCount == 0) {
        reset();
        return;
    }
    const double inv = 1.0 / static_cast<double>(validCount);
    for (double &p : pdf_)
        p *= inv;
    invalidateCache();
}

void DiscreteDistribution::fit(std::span<const double> data, std::span<const double> weights) {
    // Weighted empirical probabilities: P(X=k) = Σ(w_i for x_i=k) / Σ(w_i)
    const double sumW = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (sumW < precision::ZERO || std::isnan(sumW)) {
        reset();
        return;
    }

    std::fill(pdf_.begin(), pdf_.end(), 0.0);
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i] >= 0.0) {
            const auto index = static_cast<std::size_t>(data[i]);
            if (isValidIndex(index))
                pdf_[index] += weights[i];
        }
    }
    for (double &p : pdf_)
        p /= sumW;
    invalidateCache();
}

/**
 * Resets the distribution to uniform probabilities.
 * Each symbol gets probability 1/numSymbols
 */
void DiscreteDistribution::reset() noexcept {
    init_uniform();
}

/**
 * Returns a string representation of the distribution.
 *
 * @return String showing all symbol probabilities
 */
std::string DiscreteDistribution::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Discrete Distribution:\n";
    oss << "      Number of symbols = " << numSymbols_ << "\n";
    for (std::size_t i = 0; i < numSymbols_; ++i) {
        oss << "      P(" << i << ") = " << pdf_[i] << "\n";
    }
    return oss.str();
}

/**
 * Evaluates the logarithm of the probability mass function
 * Uses cached log probabilities for maximum performance
 */
double DiscreteDistribution::getLogProbability(double value) const noexcept {
    // Validate input - discrete distributions only accept non-negative integer values
    if (std::isnan(value) || std::isinf(value) || value < math::ZERO_DOUBLE) {
        return -std::numeric_limits<double>::infinity();
    }

    // Convert to integer index
    const auto index = static_cast<std::size_t>(value);
    if (!isValidIndex(index)) {
        return -std::numeric_limits<double>::infinity();
    }

    if (!isCacheValid())
        updateCache();
    return cachedLogProbs_[index];
}

/**
 * Evaluates the CDF at k using pre-computed cached values
 * O(1) lookup for maximum performance
 */
double DiscreteDistribution::getCumulativeProbability(double value) const noexcept {
    // Validate input
    if (std::isnan(value) || std::isinf(value)) {
        return math::ZERO_DOUBLE;
    }

    if (value < math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }

    const auto k = static_cast<std::size_t>(std::floor(value));

    // If k is beyond our range, CDF = 1.0
    if (k >= numSymbols_) {
        return math::ONE;
    }

    if (!isCacheValid())
        updateCache();
    return cachedCDF_[k];
}

/**
 * Equality comparison operator with numerical tolerance
 */
bool DiscreteDistribution::operator==(const DiscreteDistribution &other) const {
    if (numSymbols_ != other.numSymbols_) {
        return false;
    }

    const double tolerance = 1e-10;
    for (std::size_t i = 0; i < numSymbols_; ++i) {
        if (std::abs(pdf_[i] - other.pdf_[i]) > tolerance) {
            return false;
        }
    }

    return true;
}

// Parses the format produced by toString() / operator<<:
//   Discrete Distribution:
//     Number of symbols = N
//     P(0) = VALUE
//     ...
//     P(N-1) = VALUE
std::istream &operator>>(std::istream &is, libhmm::DiscreteDistribution &distribution) {
    try {
        std::string s, t;
        is >> s >> s;                // "Discrete" "Distribution:"
        is >> s >> s >> s >> s >> t; // "Number" "of" "symbols" "=" N
        const auto n = std::stoull(t);
        if (n == 0) {
            is.setstate(std::ios::failbit);
            return is;
        }
        DiscreteDistribution newDist(static_cast<int>(n));
        for (std::size_t i = 0; i < n; ++i) {
            is >> s >> s >> t; // "P(i)" "=" VALUE
            newDist.setProbability(static_cast<double>(i), std::stod(t));
        }
        distribution = std::move(newDist);
    } catch (const std::exception &) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

std::ostream &operator<<(std::ostream &os, const libhmm::DiscreteDistribution &distribution) {

    os << distribution.toString();
    return os;
}

void DiscreteDistribution::getBatchLogProbabilities(std::span<const double> observations,
                                                    std::span<double> out) const {
    // Tier 1 — concrete non-virtual loop with O(1) cached-table lookup.
    // Tier 2 upgrade: SIMD gather instructions (_mm512_i64gather_pd) could batch
    // the index lookups, but the per-element index-validation branch limits
    // vectorization benefit. The cached log-probability table is already optimal
    // for the scalar case.
    if (!isCacheValid())
        updateCache();
    for (std::size_t i = 0; i < observations.size(); ++i) {
        out[i] = DiscreteDistribution::getLogProbability(observations[i]);
    }
}

std::string DiscreteDistribution::to_json() const {
    return json::write_distribution_with_array("Discrete",
                                               {{"n", static_cast<double>(numSymbols_)}}, "probs",
                                               std::span<const double>(pdf_.data(), numSymbols_));
}
std::unique_ptr<EmissionDistribution> DiscreteDistribution::from_json(json::Reader &r) {
    // Maximum symbol count accepted during deserialization.
    // 65536 symbols × 8 bytes = 512 KB per distribution — generous for any
    // practical use. Values above this cap indicate corrupted or adversarial input.
    // The guard also prevents static_cast<int> UB when n is non-finite or huge.
    static constexpr int kMaxDiscreteSymbols = 65536;

    r.read_key(); // "n"
    const double n_raw = r.read_double();
    if (!std::isfinite(n_raw) || n_raw < 1.0 || n_raw > static_cast<double>(kMaxDiscreteSymbols))
        throw std::runtime_error("DiscreteDistribution JSON: n must be an integer in [1, " +
                                 std::to_string(kMaxDiscreteSymbols) + "]");
    const int n = static_cast<int>(n_raw);

    r.read_key(); // "probs"
    // Cap array read to n elements — a longer array is malformed and must not
    // be allowed to grow the heap before the distribution constructor fires.
    const auto probs = r.read_double_array(static_cast<std::size_t>(n));
    r.consume('}');
    auto dist = std::make_unique<DiscreteDistribution>(n);
    for (std::size_t i = 0; i < probs.size(); ++i)
        dist->setProbability(static_cast<double>(i), probs[i]);
    return dist;
}

} // namespace libhmm
