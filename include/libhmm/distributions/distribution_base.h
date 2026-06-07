#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <span>
#include "libhmm/distributions/basic_emission_distribution.h"
#include "libhmm/distributions/emission_distribution.h"

namespace libhmm {

// Forward declaration for JSON Reader (used by concrete distribution factories).
namespace json {
class Reader;
} // namespace json

// =============================================================================
// Non-template base: shared math helpers used by concrete distributions.
//
// Keeping these in a concrete (non-template) class means their definitions
// live in src/distributions/distribution_base.cpp and are compiled exactly
// once, regardless of how many DistributionBase<Derived, Obs> are instantiated.
// =============================================================================
class DistributionMathHelper {
protected:
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
    static void gcf(double& gammcf, double a, double x, double& gln) noexcept;

    /** Incomplete gamma via series representation (used by gammap). */
    static void gser(double& gamser, double a, double x, double& gln) noexcept;
};

// =============================================================================
// CRTP distribution base for all emission distributions.
//
// Inherits from BasicEmissionDistribution<Obs> (the abstract interface) and
// DistributionMathHelper (non-template shared math, compiled once).
//
// Provides:
//   - Thread-safe cache management via std::atomic<bool>
//   - Default getBatchLogProbabilities() scalar loop (override for SIMD)
//   - CRTP clone() — polymorphic copy with no per-class boilerplate
//
// @tparam Derived  Concrete distribution type (CRTP parameter).
// @tparam Obs      Observation type; default double preserves v3 API.
// =============================================================================
template<typename Derived, typename Obs = double>
class DistributionBase : public BasicEmissionDistribution<Obs>,
                         protected DistributionMathHelper {
public:
    ~DistributionBase() override = default;

    // =========================================================================
    // Rule of Five
    //
    // std::atomic<bool> is neither copyable nor movable, so copy/move
    // operations are defined explicitly to load and store the atomic flag.
    //
    // Memory ordering:
    //   - load:         acquire — cached values visible before the flag
    //   - store(true):  release — updateCache() writes visible to readers
    //   - store(false): relaxed — invalidation, no ordering needed
    // =========================================================================

    DistributionBase() : cacheValid_{false} {}

    DistributionBase(const DistributionBase& other)
        : cacheValid_{other.cacheValid_.load(std::memory_order_acquire)} {}

    DistributionBase& operator=(const DistributionBase& other) {
        if (this != &other) {
            cacheValid_.store(other.cacheValid_.load(std::memory_order_acquire),
                              std::memory_order_release);
        }
        return *this;
    }

    DistributionBase(DistributionBase&& other) noexcept
        : cacheValid_{other.cacheValid_.load(std::memory_order_acquire)} {
        // Leave moved-from object in a determinate (cache-invalid) state.
        other.cacheValid_.store(false, std::memory_order_relaxed);
    }

    DistributionBase& operator=(DistributionBase&& other) noexcept {
        if (this != &other) {
            cacheValid_.store(other.cacheValid_.load(std::memory_order_acquire),
                              std::memory_order_release);
            other.cacheValid_.store(false, std::memory_order_relaxed);
        }
        return *this;
    }

    // =========================================================================
    // CRTP clone()
    //
    // Returns a heap-allocated copy of the concrete distribution.
    // Requires Derived to be copy-constructible (all 16 distributions are).
    // =========================================================================

    [[nodiscard]] std::unique_ptr<BasicEmissionDistribution<Obs>>
    clone() const override {
        return std::make_unique<Derived>(static_cast<const Derived&>(*this));
    }

    // =========================================================================
    // Default batch log-probability (scalar loop)
    //
    // Concrete distributions override this for SIMD vectorization.
    // The virtual getLogProbability() call is intentionally explicit to make
    // the virtual dispatch visible at the call site.
    // =========================================================================

    void getBatchLogProbabilities(std::span<const Obs> observations,
                                  std::span<double> out) const override {
        assert(observations.size() == out.size());
        for (std::size_t i = 0; i < observations.size(); ++i) {
            out[i] = this->getLogProbability(observations[i]);
        }
    }

protected:
    // =========================================================================
    // Thread-safe cache management
    //
    // Use std::atomic<bool> rather than plain mutable bool because the
    // calculator thread pool can trigger concurrent const reads of the same
    // distribution. Plain bool would be a data race under concurrent reads
    // that cause a cache fill.
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
};

} // namespace libhmm
