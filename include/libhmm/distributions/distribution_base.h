#pragma once

#include <atomic>
#include <cassert>
#include <mutex>
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
    static void gcf(double &gammcf, double a, double x, double &gln) noexcept;

    /** Incomplete gamma via series representation (used by gammap). */
    static void gser(double &gamser, double a, double x, double &gln) noexcept;
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
template <typename Derived, typename Obs = double>
class DistributionBase : public BasicEmissionDistribution<Obs>, protected DistributionMathHelper {
public:
    ~DistributionBase() override = default;

    // =========================================================================
    // CRTP clone()
    //
    // Returns a heap-allocated copy of the concrete distribution.
    // Requires Derived to be copy-constructible (all concrete distributions are).
    // =========================================================================

    [[nodiscard]] std::unique_ptr<BasicEmissionDistribution<Obs>> clone() const override {
        return std::make_unique<Derived>(static_cast<const Derived &>(*this));
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
    // Rule of Five
    //
    // Constructors and assignment operators are protected: DistributionBase is
    // a CRTP intermediate and should never be instantiated directly.  Only
    // derived classes construct and assign through these operations.
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

    DistributionBase(const DistributionBase &other)
        : cacheValid_{other.cacheValid_.load(std::memory_order_acquire)} {}

    DistributionBase &operator=(const DistributionBase &other) {
        if (this != &other) {
            cacheValid_.store(other.cacheValid_.load(std::memory_order_acquire),
                              std::memory_order_release);
        }
        return *this;
    }

    DistributionBase(DistributionBase &&other) noexcept
        : cacheValid_{other.cacheValid_.load(std::memory_order_acquire)} {
        // Leave moved-from object in a determinate (cache-invalid) state.
        other.cacheValid_.store(false, std::memory_order_relaxed);
    }

    DistributionBase &operator=(DistributionBase &&other) noexcept {
        if (this != &other) {
            cacheValid_.store(other.cacheValid_.load(std::memory_order_acquire),
                              std::memory_order_release);
            other.cacheValid_.store(false, std::memory_order_relaxed);
        }
        return *this;
    }

    // =========================================================================
    // Thread-safe cache management
    //
    // cacheValid_ is a std::atomic<bool> (not a plain bool) so that the
    // fast-path check in ensureCache() is a lock-free acquire load visible
    // to all threads. cacheMutex_ serializes concurrent first-fills: two
    // threads racing on a cold cache both pass the atomic check, then one
    // acquires the mutex and fills while the other blocks; the second
    // re-checks under the lock and skips the fill. After the fill,
    // markCacheValid() does a release store so subsequent acquire loads
    // from any thread see all cached fields written by updateCache().
    //
    // Memory ordering:
    //   - isCacheValid() load: acquire — cached values visible before flag
    //   - markCacheValid() store: release — updateCache() writes visible to readers
    //   - invalidateCache() store: relaxed — invalidation needs no ordering
    //
    // cacheMutex_ is not copied or moved: each object constructs its own
    // independent mutex via the explicit copy/move constructors below.
    // =========================================================================
    mutable std::atomic<bool> cacheValid_{false};
    mutable std::mutex cacheMutex_;

    /** Mark cache as stale. Call from setters and fit(). */
    void invalidateCache() noexcept { cacheValid_.store(false, std::memory_order_relaxed); }

    /** Returns true if the cache is current. */
    [[nodiscard]] bool isCacheValid() const noexcept {
        return cacheValid_.load(std::memory_order_acquire);
    }

    /** Mark cache as valid. Call at end of updateCache(). */
    void markCacheValid() const noexcept { cacheValid_.store(true, std::memory_order_release); }

    // =========================================================================
    // ensureCache() — CRTP double-checked-locking fill helper
    //
    // Replace every `if (!isCacheValid()) updateCache();` in derived-class
    // const methods with a single `ensureCache();` call. The atomic fast
    // path avoids lock acquisition once the cache is warm; the mutex
    // serializes concurrent first-fills so cached fields are never written
    // by two threads simultaneously.
    //
    // Derived-class updateCache() must call markCacheValid() at the end and
    // must not call ensureCache() itself (would deadlock on cacheMutex_).
    // =========================================================================
    void ensureCache() const noexcept {
        if (!cacheValid_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(cacheMutex_);
            if (!cacheValid_.load(std::memory_order_acquire)) {
                static_cast<const Derived *>(this)->updateCache();
            }
        }
    }
};

} // namespace libhmm
