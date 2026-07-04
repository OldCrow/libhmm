#pragma once

#include <functional>
#include <limits>
#include <stdexcept>

#include "libhmm/basic_hmm.h"
#include "libhmm/linalg/linalg_types.h"

namespace libhmm {

/**
 * @brief Template base class for HMM calculators.
 *
 * @tparam Obs  Observation type.  `double` = scalar path (v3-compatible).
 *              `ObservationVectorView` = multivariate path (v4).
 *
 * Holds a const non-owning reference to the HMM and the observation sequence.
 * Derived classes implement the actual algorithm (forward-backward, Viterbi).
 *
 * For the scalar alias:
 *   using Calculator = BasicCalculator<double>;
 *
 * @note Passing a temporary observation sequence to any derived calculator is
 *       a compile error: both this base class and each derived class delete the
 *       `SeqType&&` constructor overload.  The base-class deletion alone is
 *       insufficient because a temporary binds to `const SeqType&` in the
 *       derived constructor and reaches the base as a named lvalue, bypassing
 *       the deleted overload.  Bind the sequence to a named variable with
 *       sufficient lifetime before constructing a calculator; the reference
 *       must remain valid for the calculator's entire lifetime.
 */
template <typename Obs>
class BasicCalculator {
public:
    using HmmType = BasicHmm<Obs>;
    using SeqType = typename ObsSeqTraits<Obs>::SeqType;

protected:
    std::reference_wrapper<const HmmType> hmm_ref_;
    std::reference_wrapper<const SeqType> obsRef_;

    /// Primary constructor (lvalue const-ref — preferred).
    BasicCalculator(const HmmType &hmm, const SeqType &observations)
        : hmm_ref_{hmm}, obsRef_{observations} {}

    BasicCalculator() = delete;
    virtual ~BasicCalculator() = default;

    BasicCalculator(const BasicCalculator &) = delete;
    BasicCalculator &operator=(const BasicCalculator &) = delete;
    BasicCalculator(BasicCalculator &&) = default;
    BasicCalculator &operator=(BasicCalculator &&) = default;

    /// Deleted rvalue overload: passing a temporary sequence is UB (dangling reference).
    /// Store the observation sequence in a named variable with sufficient lifetime.
    BasicCalculator(const HmmType &, SeqType &&) = delete;

    [[nodiscard]] const HmmType &getHmmRef() const noexcept { return hmm_ref_.get(); }
    [[nodiscard]] const SeqType &getObservations() const noexcept { return obsRef_.get(); }

    /// Rebind to a new observation sequence (caller must ensure lifetime).
    void setObservations(const SeqType &obs) noexcept { obsRef_ = std::cref(obs); }

    /// Precompute log-transition matrix logTrans[i,j] = log a_{ij} and its
    /// transpose logTransT[j,i] = log a_{ij}.  Common to all calculators.
    static void precompute_log_transitions(const HmmType &hmm, std::size_t N, Matrix &logTrans,
                                           Matrix &logTransT) noexcept {
        const Matrix &trans = hmm.getTrans();
        logTrans.resize(N, N);
        logTransT.resize(N, N);
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                const double a = trans(i, j);
                const double logA =
                    (a > 0.0) ? std::log(a) : -std::numeric_limits<double>::infinity();
                logTrans(i, j) = logA;
                logTransT(j, i) = logA;
            }
        }
    }
};

} // namespace libhmm
