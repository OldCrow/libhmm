#pragma once

#include "libhmm/linalg/linalg_types.h"
#include "libhmm/hmm.h"
#include <stdexcept>
#include <memory>
#include <functional>

namespace libhmm {

/**
 * @brief Base class for HMM calculators.
 *
 * Holds a const reference to the HMM and the observation sequence.
 * Derived classes implement the actual algorithm (forward-backward, Viterbi).
 */
class Calculator {
protected:
    std::reference_wrapper<const Hmm> hmm_ref_;
    ObservationSet observations_;

    Calculator(std::reference_wrapper<const Hmm> hmm_ref, const ObservationSet &observations)
        : hmm_ref_(hmm_ref), observations_(observations) {}

    /// Default constructor (disabled - calculators must have HMM)
    Calculator() = delete;

    /**
     * @brief Modern C++17 type-safe constructor (preferred)
     * @param hmm Const reference to HMM (immutable, lifetime-safe)
     * @param observations Observation sequence
     */
    Calculator(const Hmm &hmm, const ObservationSet &observations)
        : Calculator(std::cref(hmm), observations) {}

    /**
     * @brief Legacy pointer-based constructor (deprecated)
     * @param hmm Pointer to the HMM (must not be null)
     * @param observations The observation set to process
     * @throws std::invalid_argument if hmm is null
     * @deprecated Use const reference constructor for better type safety
     */
    [[deprecated("Use const reference constructor for better type safety")]]
    Calculator(Hmm *hmm, const ObservationSet &observations)
        : Calculator(hmm ? std::cref(*hmm)
                         : throw std::invalid_argument("HMM pointer cannot be null"),
                     observations) {}

    /// Virtual destructor for proper inheritance
    virtual ~Calculator() = default;

    /// Non-copyable but movable
    Calculator(const Calculator &) = delete;
    Calculator &operator=(const Calculator &) = delete;
    Calculator(Calculator &&) = default;
    Calculator &operator=(Calculator &&) = default;

    /**
     * @brief Get const reference to HMM (modern, type-safe)
     * @return Const reference to the HMM
     */
    const Hmm &getHmmRef() const noexcept { return hmm_ref_.get(); }

    /**
     * @brief Get const reference to observations
     * @return The observation set
     */
    const ObservationSet &getObservations() const noexcept { return observations_; }

    /**
     * @brief Set new observations
     * @param observations The new observation set
     */
    void setObservations(const ObservationSet &observations) { observations_ = observations; }

    /// Fills logTrans (row-major) and logTransT (column-major/transposed) from the
    /// HMM transition matrix. Resizes both matrices to numStates×numStates.
    static void precompute_log_transitions(const Hmm &hmm, std::size_t numStates, Matrix &logTrans,
                                           Matrix &logTransT) noexcept {
        const Matrix &trans = hmm.getTrans();
        logTrans.resize(numStates, numStates);
        logTransT.resize(numStates, numStates);
        for (std::size_t i = 0; i < numStates; ++i) {
            for (std::size_t j = 0; j < numStates; ++j) {
                const double a = trans(i, j);
                const double logA =
                    (a > 0.0) ? std::log(a) : -std::numeric_limits<double>::infinity();
                logTrans(i, j) = logA;
                logTransT(j, i) = logA;
            }
        }
    }
}; //class Calculator

} // namespace libhmm
