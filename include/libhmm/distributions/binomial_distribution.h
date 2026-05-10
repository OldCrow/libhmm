#pragma once

#include "libhmm/distributions/distribution_base.h"
#include "libhmm/common/common.h"
#include <span>

namespace libhmm {

/**
 * Binomial distribution for modeling discrete count data.
 *
 * PMF: P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
 * where C(n,k) is the binomial coefficient "n choose k"
 *
 * Properties:
 * - Mean: n * p
 * - Variance: n * p * (1-p)
 * - Support: k ∈ {0, 1, 2, ..., n}
 */
class BinomialDistribution : public DistributionBase {
private:
    /** Number of trials n - must be a positive integer */
    int n_{10};
    /** Success probability p - must be in [0,1] */
    double p_{0.5};
    /** Cached log factorial values for efficiency */
    mutable std::vector<double> logFactorialCache_;
    /** Cached log(p) and log(1-p) */
    mutable double logP_{0.0};
    mutable double log1MinusP_{0.0};

    void updateCache() const noexcept {
        logP_ = std::log(p_);
        log1MinusP_ = std::log(1.0 - p_);
        logFactorialCache_.resize(static_cast<std::size_t>(n_) + 1);
        logFactorialCache_[0] = 0.0;
        for (int i = 1; i <= n_; ++i)
            logFactorialCache_[static_cast<std::size_t>(i)] =
                logFactorialCache_[static_cast<std::size_t>(i - 1)] +
                std::log(static_cast<double>(i));
        markCacheValid();
    }

    static void validateParameters(int n, double p) {
        if (n <= 0)
            throw std::invalid_argument("Number of trials must be positive");
        if (std::isnan(p) || std::isinf(p) || p < 0.0 || p > 1.0)
            throw std::invalid_argument("Success probability must be in [0,1]");
    }

    /** Computes log C(n,k) = log(n!/(k!(n-k)!)) using cached log-factorials. */
    double logBinomialCoefficient(int n, int k) const {
        if (k < 0 || k > n)
            return -std::numeric_limits<double>::infinity();
        if (!isCacheValid())
            updateCache();
        const auto un = static_cast<std::size_t>(n);
        const auto uk = static_cast<std::size_t>(k);
        return logFactorialCache_[un] - logFactorialCache_[uk] - logFactorialCache_[un - uk];
    }

    friend std::istream &operator>>(std::istream &is, libhmm::BinomialDistribution &distribution);

public:
    explicit BinomialDistribution(int n = 10, double p = 0.5) : n_{n}, p_{p} {
        validateParameters(n, p);
        updateCache();
    }

    BinomialDistribution(const BinomialDistribution &other) = default;
    BinomialDistribution &operator=(const BinomialDistribution &other) = default;
    BinomialDistribution(BinomialDistribution &&other) noexcept = default;
    BinomialDistribution &operator=(BinomialDistribution &&other) noexcept = default;
    ~BinomialDistribution() override = default;

    [[nodiscard]] double getProbability(double value) const override;
    [[nodiscard]] double getLogProbability(double value) const noexcept override;

    /// Concrete non-virtual batch log-PMF. Eliminates per-element virtual dispatch.
    /// Precondition: observations.size() == out.size()
    void getBatchLogProbabilities(std::span<const double> observations,
                                  std::span<double> out) const override;
    [[nodiscard]] double getCumulativeProbability(double value) const noexcept;

    /** Fit p̂ = sample_mean / n (n estimated as max observed value). */
    void fit(std::span<const double> data) override;
    /** Weighted fit: p̂ = weighted_mean / n. */
    void fit(std::span<const double> data, std::span<const double> weights) override;

    /** Returns true — Binomial is a discrete distribution. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return true; }
    [[nodiscard]] std::size_t getNumParameters() const noexcept override { return 2; }

    void reset() noexcept override;
    std::string toString() const override;
    [[nodiscard]] std::string to_json() const override;
    /// @internal JSON factory — called by the distribution registry in src/io/hmm_json.cpp.
    static std::unique_ptr<EmissionDistribution> from_json(json::Reader &r);

    [[nodiscard]] int getN() const noexcept { return n_; }
    [[nodiscard]] double getP() const noexcept { return p_; }

    void setN(int n) {
        validateParameters(n, p_);
        n_ = n;
        invalidateCache();
    }
    void setP(double p) {
        validateParameters(n_, p);
        p_ = p;
        invalidateCache();
    }
    void setParameters(int n, double p) {
        validateParameters(n, p);
        n_ = n;
        p_ = p;
        invalidateCache();
    }

    [[nodiscard]] double getMean() const noexcept { return static_cast<double>(n_) * p_; }
    [[nodiscard]] double getVariance() const noexcept {
        return static_cast<double>(n_) * p_ * (1.0 - p_);
    }
    [[nodiscard]] double getStandardDeviation() const noexcept { return std::sqrt(getVariance()); }

    bool operator==(const BinomialDistribution &other) const;
    bool operator!=(const BinomialDistribution &other) const { return !(*this == other); }
};

std::ostream &operator<<(std::ostream &, const libhmm::BinomialDistribution &);

} // namespace libhmm
