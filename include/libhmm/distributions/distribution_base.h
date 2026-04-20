#pragma once

#include <atomic>
#include <cstddef>
#include <vector>
#include "libhmm/distributions/emission_distribution.h"
#include "libhmm/distributions/probability_distribution.h"  // backward compat during Phase 2

namespace libhmm {

/**
 * @brief Shared implementation base for all ported emission distributions.
 *
 * Inherits both:
 *   - EmissionDistribution — the new const-correct interface
 *   - ProbabilityDistribution — for backward compatibility with existing HMM,
 *     calculator, and training code during Phase 2
 *
 * Provides:
 *   - std::atomic<bool> cache flag with thread-safe load/store helpers
 *   - Default getBatchLogProbabilities() (scalar loop; override for SIMD)
 *   - Shims so old callers (using ProbabilityDistribution*) work unchanged:
 *       getProbability(Observation) → delegates to getProbability(double) const
 *       fit(vector<Observation>&)   → delegates to fit(span<const double>)
 *   - Protected math helpers shared across distributions
 *     (incomplete gamma, inverse error function)
 *
 * Concrete distributions implement the EmissionDistribution pure virtuals.
 * The ProbabilityDistribution shims are marked final here so derived classes
 * cannot accidentally re-implement the old non-const interface.
 *
 * Phase 3: once HMM/calculators are updated to use EmissionDistribution*,
 * the ProbabilityDistribution base and shims can be removed.
 */
class DistributionBase : public ProbabilityDistribution,
                         public EmissionDistribution {
public:
    virtual ~DistributionBase() = default;

    DistributionBase();
    DistributionBase(const DistributionBase& other);
    DistributionBase& operator=(const DistributionBase& other);
    DistributionBase(DistributionBase&& other) noexcept;
    DistributionBase& operator=(DistributionBase&& other) noexcept;

    // =========================================================================
    // EmissionDistribution: default batch implementation (scalar loop)
    // Override in concrete distributions for SIMD vectorization.
    // =========================================================================
    void getBatchLogProbabilities(
        std::span<const double> observations,
        std::span<double> out) const override;

    // =========================================================================
    // ProbabilityDistribution shims — backward compatibility, marked final.
    // Derived classes implement the EmissionDistribution const interface;
    // these shims forward old callers to the new implementations.
    // =========================================================================

    /** Shim: calls const getProbability(double) via the new interface. */
    double getProbability(Observation val) final {
        return static_cast<const DistributionBase*>(this)
                   ->getProbability(static_cast<double>(val));
    }

    /** Shim: converts vector to span and calls new fit(span). Marked final so derived
     *  classes cannot accidentally re-implement the old vector interface. The span
     *  overloads below are separately declared pure virtual so MSVC knows they are
     *  distinct from this final overload and CAN be overridden. */
    void fit(const std::vector<Observation>& values) final {
        fit(std::span<const double>(values.data(), values.size()));
    }

    // Explicitly re-declare EmissionDistribution pure virtuals here so that:
    // (a) name hiding by the vector shim above is resolved, and
    // (b) MSVC can see that these overloads are NOT final and CAN be overridden
    //     by concrete distributions.
    // (Using 'using EmissionDistribution::fit' interacts poorly with 'final' on
    //  the vector overload in MSVC — it treats final as applying to all fit overloads.)
    using EmissionDistribution::getProbability;    // unhides getProbability(double) const
    using EmissionDistribution::getLogProbability; // resolves ambiguity (same sig in both bases)
    virtual void fit(std::span<const double> data) override = 0;
    virtual void fit(std::span<const double> data,
                     std::span<const double> weights) override = 0;

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
    void invalidateCache() noexcept {
        cacheValid_.store(false, std::memory_order_relaxed);
    }

    /** Returns true if the cache is current. */
    [[nodiscard]] bool isCacheValid() const noexcept {
        return cacheValid_.load(std::memory_order_acquire);
    }

    /** Mark cache as valid. Call at end of updateCache(). */
    void markCacheValid() const noexcept {
        cacheValid_.store(true, std::memory_order_release);
    }

    // =========================================================================
    // Shared math helpers — static, available to all distributions.
    // Implementations in src/distributions/distribution_base.cpp.
    // =========================================================================

    /** Regularized incomplete gamma P(a, x). */
    static double gammap(double a, double x) noexcept;

    /** Inverse error function erfinv(y). */
    static double errorf_inv(double y) noexcept;

private:
    /** Incomplete gamma via continued fraction (used by gammap). */
    static void gcf(double& gammcf, double a, double x, double& gln) noexcept;

    /** Incomplete gamma via series representation (used by gammap). */
    static void gser(double& gamser, double a, double x, double& gln) noexcept;
};

} // namespace libhmm
