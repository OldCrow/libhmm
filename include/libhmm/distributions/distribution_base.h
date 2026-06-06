#pragma once

#include <atomic>
#include <cstddef>
#include <span>
#include "libhmm/distributions/emission_distribution.h"

namespace libhmm {

// Forward declaration — concrete distributions declare from_json(json::Reader&)
// as a static factory method; implementation is in src/io/hmm_json.cpp.
namespace json {
class Reader;
} // namespace json

/**
 * @brief Shared implementation base for all emission distributions.
 *
 * Inherits EmissionDistribution and provides:
 *   - std::atomic<bool> cache flag with thread-safe load/store helpers
 *   - Default getBatchLogProbabilities() scalar loop (override for SIMD)
 *   - Protected math helpers: incomplete gamma, inverse error function
 *
 * Concrete distributions implement the EmissionDistribution pure virtuals.
 */
class DistributionBase : public EmissionDistribution {
public:
    ~DistributionBase() override = default;

    DistributionBase();
    DistributionBase(const DistributionBase &other);
    DistributionBase &operator=(const DistributionBase &other);
    DistributionBase(DistributionBase &&other) noexcept;
    DistributionBase &operator=(DistributionBase &&other) noexcept;

    // Default batch implementation (scalar loop).
    // Override in concrete distributions for SIMD vectorization.
    void getBatchLogProbabilities(std::span<const double> observations,
                                  std::span<double> out) const override;

protected:
    // =========================================================================
    // Thread-safe cache management
    //
    // Use std::atomic<bool> rather than plain mutable bool because the
    // calculator thread pool can trigger concurrent const reads of the same
    // distribution. Plain bool would be a data race under concurrent reads
    // that cause a cache fill.
    //
    // Memory ordering:
    //   - load: acquire  — ensures cached values are visible before the flag
    //   - store(true): release — ensures updateCache() writes are visible
    //   - store(false): relaxed — invalidation, no ordering needed
    // =========================================================================
    mutable std::atomic<bool> cacheValid_{false};

    /** Mark cache as stale. Call from setters and fit(). */
    void invalidateCache() noexcept { cacheValid_.store(false, std::memory_order_relaxed); }

    /** Returns true if the cache is current. */
    [[nodiscard]] bool isCacheValid() const noexcept {
        return cacheValid_.load(std::memory_order_acquire);
    }

    /** Mark cache as valid. Call at end of updateCache(). */
    void markCacheValid() const noexcept { cacheValid_.store(true, std::memory_order_release); }

    // =========================================================================
    // Shared math helpers — static, available to all distributions.
    // Implementations in src/distributions/distribution_base.cpp.
    // =========================================================================

    /** Regularized incomplete gamma P(a, x). */
    static double gammap(double a, double x) noexcept;

    /**
     * @brief Regularized incomplete beta function I_x(a, b).
     *
     * Uses a continued-fraction algorithm with symmetry relation for
     * numerical stability across the full (0,1) domain.  Used by
     * BetaDistribution::getCumulativeProbability() and
     * StudentTDistribution::getCumulativeProbability().
     */
    static double incompleteBeta(double x, double a, double b) noexcept;

    /** Inverse error function erfinv(y). */
    static double errorf_inv(double y) noexcept;

private:
    /** Incomplete gamma via continued fraction (used by gammap). */
    static void gcf(double &gammcf, double a, double x, double &gln) noexcept;

    /** Incomplete gamma via series representation (used by gammap). */
    static void gser(double &gamser, double a, double x, double &gln) noexcept;
};

} // namespace libhmm
