#pragma once

#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

#include "libhmm/basic_hmm.h"
#include "libhmm/linalg/linalg_types.h"

namespace libhmm {

/**
 * @brief Template base class for HMM training algorithms.
 *
 * @tparam Obs  Observation type.  `double` = scalar path (v3-compatible).
 *              `ObservationVectorView` = multivariate path (v4).
 *
 * The HMM is held by non-owning mutable reference (mutated in place by train()).
 * The observation list is held by non-owning const reference — the caller must
 * keep the data alive for the trainer's lifetime.  This follows the same
 * non-ownership convention used for the HMM reference and avoids copying
 * large observation datasets.
 *
 * v4 breaking change from v3: obsLists is now a const reference, not a copy.
 * Code that passed a temporary will not compile; use a named variable instead.
 *
 * For the scalar alias:
 *   using Trainer = BasicTrainer<double>;
 */
template <typename Obs>
class BasicTrainer {
public:
    using HmmType = BasicHmm<Obs>;
    using ListType = typename ObsSeqTraits<Obs>::ListType;

protected:
    std::reference_wrapper<HmmType> hmm_ref_;
    std::reference_wrapper<const ListType> obsLists_ref_;

public:
    /// Primary constructor (lvalue const-ref — preferred).
    /// @throws std::invalid_argument if obsLists is empty.
    BasicTrainer(HmmType &hmm, const ListType &obsLists) : hmm_ref_{hmm}, obsLists_ref_{obsLists} {
        if (obsLists.empty()) {
            throw std::invalid_argument("Observation lists cannot be empty");
        }
    }

    /// Deleted rvalue overload: passing a temporary is UB (dangling reference).
    /// Store the observation list in a named variable with sufficient lifetime.
    BasicTrainer(HmmType &, ListType &&) = delete;

    virtual ~BasicTrainer() = default;

    BasicTrainer(const BasicTrainer &) = delete;
    BasicTrainer &operator=(const BasicTrainer &) = delete;
    BasicTrainer(BasicTrainer &&) = default;
    BasicTrainer &operator=(BasicTrainer &&) = default;

    /// Execute one full training pass, updating the HMM in place.
    virtual void train() = 0;

    [[nodiscard]] HmmType &getHmmRef() const noexcept { return hmm_ref_.get(); }
    [[nodiscard]] HmmType &getHmm() const noexcept { return hmm_ref_.get(); }
    [[nodiscard]] const ListType &getObservationLists() const noexcept {
        return obsLists_ref_.get();
    }

protected:
    /**
     * @brief Build a column-major flat log-transition vector from the current HMM.
     *
     * logTransT[j * N + i] = log a_{ij}.  Column-major layout matches the
     * contiguous read pattern in the xi inner loop.
     *
     * @param[out] logTransT         Flat vector sized N*N (pre-allocated by caller).
     * @param[out] hasZeroTransitions Set to true if any A(i,j) = 0 (sparse model).
     */
    static void precompute_log_trans_flat(const HmmType &hmm, std::size_t N,
                                          std::vector<double> &logTransT,
                                          bool &hasZeroTransitions) noexcept {
        constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();
        const Matrix &A = hmm.getTrans();
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                const double a = A(i, j);
                logTransT[j * N + i] = (a > 0.0) ? std::log(a) : LOG_ZERO;
                if (a <= 0.0)
                    hasZeroTransitions = true;
            }
        }
    }

    // =========================================================================
    // apply_emission_fits — scalar path (Obs=double)
    //
    // Refits each emission distribution from accumulated (data, weights) pairs.
    // States with no observations are reset to defaults.
    // If weights is non-empty the weighted fit overload is used (Baum-Welch);
    // otherwise the unweighted overload is used (Viterbi hard assignment).
    //
    // The multivariate path provides its own M-step logic in the concrete
    // trainer via a different accumulation strategy (index pairs).
    // =========================================================================
    static void apply_emission_fits(HmmType &hmm, std::size_t numStates,
                                    const std::vector<std::vector<double>> &data,
                                    const std::vector<std::vector<double>> &weights = {}) {
        for (std::size_t i = 0; i < numStates; ++i) {
            const std::size_t M = data[i].size();
            if (M == 0) {
                hmm.getDistribution(i).reset();
                continue;
            }
            if (weights.empty()) {
                hmm.getDistribution(i).fit(std::span<const double>(data[i].data(), M));
            } else {
                hmm.getDistribution(i).fit(std::span<const double>(data[i].data(), M),
                                           std::span<const double>(weights[i].data(), M));
            }
        }
    }
};

} // namespace libhmm
