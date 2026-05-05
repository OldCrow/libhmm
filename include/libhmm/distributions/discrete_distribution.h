#pragma once

#include "libhmm/distributions/distribution_base.h"
#include "libhmm/common/common.h"
#include <numeric>
#include <span>

namespace libhmm {

/**
 * Modern C++20 Discrete distribution for modeling categorical data.
 *
 * The Discrete distribution (also known as Categorical distribution) is a
 * discrete probability distribution that generalizes the Bernoulli distribution.
 * It describes the possible results of a random variable that can take on
 * one of K possible categories, with the probability of each category separately specified.
 *
 * PMF: P(X = k) = p_k for k ∈ {0, 1, 2, ..., K-1}
 * where p_k is the probability of category k and ∑p_k = 1
 *
 * Properties:
 * - Support: {0, 1, 2, ..., numSymbols-1}
 * - Probability mass function defined for each discrete symbol
 * - All probabilities must sum to 1.0
 * - Each probability must be in [0, 1]
 *
 * Applications:
 * - Hidden Markov Models with discrete observations
 * - Classification problems
 * - Multinomial experiments
 * - Any scenario with discrete, mutually exclusive outcomes
 */
class DiscreteDistribution : public DistributionBase {
private:
    /**
     * Number of discrete symbols/categories
     * Must be > 0, typically small (e.g., 2-100)
     */
    std::size_t numSymbols_;

    /**
     * Contains probabilities for discrete observations
     * pdf_[i] = P(X = i) for i ∈ {0, 1, ..., numSymbols-1}
     * Using std::vector for modern C++ performance and semantics
     */
    std::vector<double> pdf_;

    /**
     * Comprehensive cached values for maximum performance
     */
    mutable double cachedSum_{1.0};                   // Sum of all probabilities (validation)
    mutable double cachedEntropy_{0.0};               // Shannon entropy: H(X) = -∑p_i*log(p_i)
    mutable std::vector<double> cachedLogProbs_;      // Pre-computed log probabilities
    mutable std::vector<double> cachedCDF_;           // Pre-computed cumulative distribution
    mutable std::size_t cachedMode_{0};               // Index of most probable symbol
    mutable double cachedMaxProb_{0.0};               // Maximum probability value
    mutable std::vector<std::size_t> nonZeroIndices_; // Indices with non-zero probabilities

    void updateCache() const noexcept {
        // Basic statistics
        cachedSum_ = std::accumulate(pdf_.begin(), pdf_.end(), 0.0);

        // Pre-compute log probabilities and entropy
        cachedLogProbs_.resize(numSymbols_);
        cachedEntropy_ = 0.0;
        cachedMaxProb_ = 0.0;
        cachedMode_ = 0;
        nonZeroIndices_.clear();

        for (std::size_t i = 0; i < numSymbols_; ++i) {
            const double p = pdf_[i];

            // Cache log probabilities
            if (p > 0.0) {
                cachedLogProbs_[i] = std::log(p);
                cachedEntropy_ -= p * cachedLogProbs_[i];
                nonZeroIndices_.push_back(i);

                // Track mode (most probable symbol)
                if (p > cachedMaxProb_) {
                    cachedMaxProb_ = p;
                    cachedMode_ = i;
                }
            } else {
                cachedLogProbs_[i] = -std::numeric_limits<double>::infinity();
            }
        }

        // Pre-compute CDF
        cachedCDF_.resize(numSymbols_);
        cachedCDF_[0] = pdf_[0];
        for (std::size_t i = 1; i < numSymbols_; ++i) {
            cachedCDF_[i] = cachedCDF_[i - 1] + pdf_[i];
        }

        markCacheValid();
    }

    /**
     * Validates that an observation index is within valid range
     */
    bool isValidIndex(std::size_t index) const noexcept { return index < numSymbols_; }

    static std::size_t validateSymbols(int symbols) {
        if (symbols <= 0)
            throw std::invalid_argument("Number of symbols must be greater than 0");
        return static_cast<std::size_t>(symbols);
    }

    /** Validates a probability value is in [0, 1]. */
    static void validateProbabilityValue(double value) {
        if (std::isnan(value) || std::isinf(value) || value < 0.0 || value > 1.0)
            throw std::invalid_argument("Probability value must be between 0 and 1");
    }

public:
    /**
     * Constructs a Discrete distribution with given number of symbols.
     * Initializes to uniform distribution.
     *
     * @param symbols Number of discrete symbols/categories (must be > 0)
     * @throws std::invalid_argument if symbols <= 0
     */
    explicit DiscreteDistribution(int symbols = 10)
        : DistributionBase{}, numSymbols_{validateSymbols(symbols)}, pdf_(numSymbols_),
          cachedSum_{1.0}, cachedEntropy_{0.0} {
        reset();
    }

    DiscreteDistribution(const DiscreteDistribution &other) = default;
    DiscreteDistribution &operator=(const DiscreteDistribution &other) = default;
    DiscreteDistribution(DiscreteDistribution &&other) noexcept = default;
    DiscreteDistribution &operator=(DiscreteDistribution &&other) noexcept = default;
    ~DiscreteDistribution() override = default;

    /**
     * Gets the probability mass function value for a discrete observation.
     *
     * @param value The discrete value (will be cast to integer index)
     * @return Probability mass for the given value, 0.0 if out of range
     */
    [[nodiscard]] double getProbability(double x) const override;

    /** Empirical probabilities: P(X=k) = count(k) / N. */
    void fit(std::span<const double> data) override;

    /**
     * Weighted empirical probabilities: P(X=k) = Σ(w_i for x_i=k) / Σ(w_i).
     * Falls back to reset() if sum(weights) is near zero.
     */
    void fit(std::span<const double> data, std::span<const double> weights) override;

    /** Returns true — Discrete is a discrete distribution. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return true; }

    /**
     * Resets the distribution to uniform probabilities.
     * Each symbol gets probability 1/numSymbols
     */
    void reset() noexcept override;

    /**
     * Sets the probability for a specific discrete observation.
     *
     * @param o The discrete observation (symbol index)
     * @param value The probability value (must be in [0,1])
     * @throws std::invalid_argument if value is not a valid probability
     * @throws std::out_of_range if observation index is out of range
     */
    /**
     * Sets the probability for a specific symbol.
     * Now inline, consistent with setters in all other ported distributions.
     */
    void setProbability(double o, double value) {
        validateProbabilityValue(value);
        // Guard before cast: negative float → size_t is UB
        if (std::isnan(o) || std::isinf(o) || o < 0.0)
            throw std::out_of_range("Observation index out of range");
        const auto index = static_cast<std::size_t>(o);
        if (!isValidIndex(index))
            throw std::out_of_range("Observation index out of range");
        pdf_[index] = value;
        invalidateCache();
    }

    /**
     * Returns a string representation of the distribution.
     *
     * @return String showing all symbol probabilities
     */
    std::string toString() const override;
    [[nodiscard]] std::string to_json() const override;
    /// @internal JSON factory — called by the distribution registry in src/io/hmm_json.cpp.
    static std::unique_ptr<EmissionDistribution> from_json(json::Reader &r);

    /**
     * Gets the number of discrete symbols
     *
     * @return Number of symbols/categories
     */
    std::size_t getNumSymbols() const noexcept { return numSymbols_; }

    /**
     * Gets the probability for a specific symbol.
     *
     * @param index Symbol index (must be < numSymbols)
     * @return Probability for the symbol
     * @throws std::out_of_range if index is out of range
     */
    /** Get probability by direct symbol index (throws std::out_of_range if out of bounds). */
    double getSymbolProbability(std::size_t index) const {
        if (!isValidIndex(index))
            throw std::out_of_range("Symbol index out of range");
        return pdf_[index];
    }

    /**
     * Gets the sum of all probabilities (should be approximately 1.0).
     *
     * @return Sum of all probabilities
     */
    double getProbabilitySum() const {
        if (!isCacheValid())
            updateCache();
        return cachedSum_;
    }
    double getEntropy() const {
        if (!isCacheValid())
            updateCache();
        return cachedEntropy_;
    }
    std::size_t getMode() const {
        if (!isCacheValid())
            updateCache();
        return cachedMode_;
    }

    /**
     * Gets the mean of the distribution.
     * For discrete distribution, mean = ∑(i * p_i) for i = 0 to numSymbols-1
     *
     * @return Mean value
     */
    double getMean() const noexcept {
        double mean = 0.0;
        for (std::size_t i = 0; i < numSymbols_; ++i) {
            mean += static_cast<double>(i) * pdf_[i];
        }
        return mean;
    }

    /**
     * Gets the variance of the distribution.
     * For discrete distribution, variance = ∑(i² * p_i) - mean²
     *
     * @return Variance value
     */
    double getVariance() const noexcept {
        const double mean = getMean();
        double secondMoment = 0.0;
        for (std::size_t i = 0; i < numSymbols_; ++i) {
            const double iDouble = static_cast<double>(i);
            secondMoment += iDouble * iDouble * pdf_[i];
        }
        return secondMoment - mean * mean;
    }

    /**
     * Gets the standard deviation of the distribution.
     *
     * @return Standard deviation value
     */
    double getStandardDeviation() const noexcept { return std::sqrt(getVariance()); }

    /**
     * Normalizes the distribution so probabilities sum to 1.0.
     * Useful after manual probability modifications.
     */
    void normalize() {
        if (!isCacheValid())
            updateCache();
        if (cachedSum_ > 0.0) {
            for (double &p : pdf_)
                p /= cachedSum_;
            invalidateCache();
        }
    }

    /**
     * Evaluates the logarithm of the probability mass function
     * More numerically stable for small probabilities
     *
     * @param value The discrete value (will be cast to integer index)
     * @return Log probability mass, -infinity if out of range or probability is 0
     */
    [[nodiscard]] double getLogProbability(double value) const noexcept override;

    /// Concrete non-virtual batch log-PMF (table lookup). Eliminates per-element virtual dispatch.
    /// Precondition: observations.size() == out.size()
    void getBatchLogProbabilities(std::span<const double> observations,
                                  std::span<double> out) const override;

    /**
     * Evaluates the CDF at k using cumulative sum approach
     * Formula: CDF(k) = ∑(i=0 to k) P(X = i)
     *
     * @param value The value at which to evaluate the CDF
     * @return Cumulative probability P(X ≤ value)
     */
    [[nodiscard]] double getCumulativeProbability(double value) const noexcept;

    /**
     * Equality comparison operator
     * @param other Other distribution to compare with
     * @return true if distributions are equal within tolerance
     */
    bool operator==(const DiscreteDistribution &other) const;

    /**
     * Inequality comparison operator
     * @param other Other distribution to compare with
     * @return true if distributions are not equal
     */
    bool operator!=(const DiscreteDistribution &other) const { return !(*this == other); }

private:
    friend std::istream &operator>>(std::istream &is, libhmm::DiscreteDistribution &distribution);
};

std::ostream &operator<<(std::ostream &, const libhmm::DiscreteDistribution &);

} // namespace libhmm
