#ifndef HMMTRAINER_H_
#define HMMTRAINER_H_

#include "libhmm/common/common.h"
#include "libhmm/hmm.h"
#include <stdexcept>

namespace libhmm
{

/// Base class for HMM training algorithms
/// Provides common interface and functionality for all HMM trainers
class HmmTrainer
{
protected:
    Hmm* hmm_{nullptr};
    ObservationLists obsLists_;

public:
    /// Virtual destructor for proper inheritance
    virtual ~HmmTrainer() = default;

    /// Constructor with HMM and observation lists
    /// @param hmm Pointer to the HMM to train (must not be null)
    /// @param obsLists List of observation sequences for training
    /// @throws std::invalid_argument if hmm is null or obsLists is empty
    HmmTrainer(Hmm* hmm, const ObservationLists& obsLists)
        : hmm_{hmm}, obsLists_{obsLists} {
        if (!hmm) {
            throw std::invalid_argument("HMM pointer cannot be null");
        }
        if (obsLists.empty()) {
            throw std::invalid_argument("Observation lists cannot be empty");
        }
    }

    /// Non-copyable but movable
    HmmTrainer(const HmmTrainer&) = delete;
    HmmTrainer& operator=(const HmmTrainer&) = delete;
    HmmTrainer(HmmTrainer&&) = default;
    HmmTrainer& operator=(HmmTrainer&&) = default;

    /// Get the HMM being trained
    /// @return Pointer to the HMM
    Hmm* getHmm() const noexcept { return hmm_; }

    /// Get the observation lists
    /// @return Reference to the observation lists
    const ObservationLists& getObservationLists() const noexcept { return obsLists_; }

    /// Train the HMM using the provided observation sequences
    /// Subclasses must implement this function with their specific algorithm
    virtual void train() = 0;
}; //class HmmTrainer

}

#endif

