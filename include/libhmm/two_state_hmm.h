#ifndef TWO_STATE_HMM_H_
#define TWO_STATE_HMM_H_

#include <memory>
#include <stdexcept>
#include "libhmm/libhmm.h"

namespace libhmm {

/// Modern C++17 constants for two-state HMM configuration
namespace TwoStateHMM {
    /// Number of states in the two-state HMM
    inline constexpr std::size_t NUM_STATES = 2;
    
    /// Number of symbols for discrete distribution (dice faces)
    inline constexpr std::size_t NUM_SYMBOLS = 6;
    
    /// Fair die state index
    inline constexpr std::size_t FAIR_STATE = 0;
    
    /// Loaded die state index  
    inline constexpr std::size_t LOADED_STATE = 1;
    
    /// Initial probability for fair state
    inline constexpr double FAIR_INITIAL_PROB = 0.75;
    
    /// Initial probability for loaded state
    inline constexpr double LOADED_INITIAL_PROB = 0.25;
    
    /// Transition probabilities
    inline constexpr double FAIR_TO_FAIR = 0.9;
    inline constexpr double FAIR_TO_LOADED = 0.1;
    inline constexpr double LOADED_TO_FAIR = 0.8;
    inline constexpr double LOADED_TO_LOADED = 0.2;
    
    /// Emission probabilities for fair die (uniform)
    inline constexpr double FAIR_EMISSION_PROB = 1.0 / NUM_SYMBOLS; // 0.166667
    
    /// Emission probabilities for loaded die
    inline constexpr double LOADED_NORMAL_PROB = 0.125; // faces 1-5
    inline constexpr double LOADED_BIASED_PROB = 0.375; // face 6
}

/**
 * Creates a modern, type-safe HMM based on the "Occasionally Dishonest Casino" example.
 * This function sets up a two-state HMM where:
 * - State 0: Fair die (uniform distribution over 6 faces)
 * - State 1: Loaded die (biased toward face 6)
 * 
 * @param hmm Pointer to HMM to configure (must not be null)
 * @throws std::invalid_argument if hmm is null
 * @throws std::runtime_error if HMM setup fails
 */
inline void prepareTwoStateHmm(Hmm* hmm) {
    if (!hmm) {
        throw std::invalid_argument("HMM pointer cannot be null");
    }
    
    try {
        // Set up initial state probabilities
        Vector pi(TwoStateHMM::NUM_STATES);
        pi(TwoStateHMM::FAIR_STATE) = TwoStateHMM::FAIR_INITIAL_PROB;
        pi(TwoStateHMM::LOADED_STATE) = TwoStateHMM::LOADED_INITIAL_PROB;
        
        // Set up transition matrix
        Matrix trans(TwoStateHMM::NUM_STATES, TwoStateHMM::NUM_STATES);
        trans(TwoStateHMM::FAIR_STATE, TwoStateHMM::FAIR_STATE) = TwoStateHMM::FAIR_TO_FAIR;
        trans(TwoStateHMM::FAIR_STATE, TwoStateHMM::LOADED_STATE) = TwoStateHMM::FAIR_TO_LOADED;
        trans(TwoStateHMM::LOADED_STATE, TwoStateHMM::FAIR_STATE) = TwoStateHMM::LOADED_TO_FAIR;
        trans(TwoStateHMM::LOADED_STATE, TwoStateHMM::LOADED_STATE) = TwoStateHMM::LOADED_TO_LOADED;
        
        hmm->setPi(pi);
        hmm->setTrans(trans);
        
        // Set up emission distributions
        // Fair die: uniform distribution
        auto fairDist = std::make_unique<DiscreteDistribution>(TwoStateHMM::NUM_SYMBOLS);
        for (std::size_t i = 0; i < TwoStateHMM::NUM_SYMBOLS; ++i) {
            fairDist->setProbability(static_cast<int>(i), TwoStateHMM::FAIR_EMISSION_PROB);
        }
        hmm->setProbabilityDistribution(TwoStateHMM::FAIR_STATE, std::move(fairDist));
        
        // Loaded die: biased toward 6
        auto loadedDist = std::make_unique<DiscreteDistribution>(TwoStateHMM::NUM_SYMBOLS);
        for (std::size_t i = 0; i < TwoStateHMM::NUM_SYMBOLS - 1; ++i) {
            loadedDist->setProbability(static_cast<int>(i), TwoStateHMM::LOADED_NORMAL_PROB);
        }
        loadedDist->setProbability(static_cast<int>(TwoStateHMM::NUM_SYMBOLS - 1), TwoStateHMM::LOADED_BIASED_PROB);
        hmm->setProbabilityDistribution(TwoStateHMM::LOADED_STATE, std::move(loadedDist));
        
        // Validate the HMM is properly configured
        hmm->validate();
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to prepare two-state HMM: " + std::string(e.what()));
    }
}

/**
 * Legacy function for backward compatibility.
 * @deprecated Use prepareTwoStateHmm() instead
 */
[[deprecated("Use prepareTwoStateHmm() instead for better type safety")]]
inline void prepare_hmm(Hmm* hmm) {
    prepareTwoStateHmm(hmm);
}

/**
 * Creates and returns a pre-configured two-state HMM.
 * @return Unique pointer to configured HMM
 * @throws std::runtime_error if HMM creation fails
 */
inline std::unique_ptr<Hmm> createTwoStateHmm() {
    auto hmm = std::make_unique<Hmm>(TwoStateHMM::NUM_STATES);
    prepareTwoStateHmm(hmm.get());
    return hmm;
}

} // namespace libhmm

#endif // TWO_STATE_HMM_H_
