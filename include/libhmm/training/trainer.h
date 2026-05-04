#pragma once

#include "libhmm/linalg/linalg_types.h"
#include "libhmm/hmm.h"
#include <functional>
#include <stdexcept>

namespace libhmm {

/**
 * @brief Abstract base for HMM training algorithms.
 *
 * Holds a non-owning reference to the HMM being trained and the observation
 * sequences. The HMM is mutated in place by train().
 *
 * Prefer the Hmm& constructor. The Hmm* overload is provided for legacy
 * subclasses whose initialiser lists forward a raw pointer.
 */
class Trainer {
protected:
    std::reference_wrapper<Hmm> hmm_ref_;
    ObservationLists obsLists_;

public:
    /// Primary constructor.
    /// @throws std::invalid_argument if obsLists is empty.
    Trainer(Hmm &hmm, const ObservationLists &obsLists) : hmm_ref_{hmm}, obsLists_{obsLists} {
        if (obsLists.empty()) {
            throw std::invalid_argument("Observation lists cannot be empty");
        }
    }

    /// Legacy constructor for subclasses that forward a raw pointer.
    /// @throws std::invalid_argument if hmm is null or obsLists is empty.
    Trainer(Hmm *hmm, const ObservationLists &obsLists)
        : Trainer(hmm ? *hmm : throw std::invalid_argument("HMM pointer cannot be null"),
                  obsLists) {}

    virtual ~Trainer() = default;

    Trainer(const Trainer &) = delete;
    Trainer &operator=(const Trainer &) = delete;
    Trainer(Trainer &&) = default;
    Trainer &operator=(Trainer &&) = default;

    /// Execute one full training pass, updating the HMM in place.
    virtual void train() = 0;

protected:
    /// Refits emission distributions for all states after an E-step.
    /// States with no observations are reset to defaults.
    /// If weights is non-empty the weighted fit overload is used (Baum-Welch);
    /// otherwise the unweighted overload is used (Viterbi hard assignment).
    static void apply_emission_fits(Hmm &hmm, std::size_t numStates,
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

public:
    /// @return Reference to the HMM under training.
    [[nodiscard]] Hmm &getHmmRef() const noexcept { return hmm_ref_.get(); }

    /// @return Reference to the HMM under training.
    [[nodiscard]] Hmm &getHmm() const noexcept { return getHmmRef(); }

    /// @return The observation sequences supplied at construction.
    [[nodiscard]] const ObservationLists &getObservationLists() const noexcept { return obsLists_; }
}; // class Trainer

} // namespace libhmm
