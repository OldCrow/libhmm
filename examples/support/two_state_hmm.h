#pragma once

#include "libhmm/hmm.h"
#include "libhmm/distributions/discrete_distribution.h"
#include <memory>
#include <stdexcept>

namespace libhmm::examples {

namespace dishonest_casino {
inline constexpr std::size_t num_states = 2;
inline constexpr std::size_t num_symbols = 6;
inline constexpr std::size_t fair_state = 0;
inline constexpr std::size_t loaded_state = 1;
inline constexpr double fair_initial_prob = 0.75;
inline constexpr double loaded_initial_prob = 0.25;
inline constexpr double fair_to_fair = 0.9;
inline constexpr double fair_to_loaded = 0.1;
inline constexpr double loaded_to_fair = 0.8;
inline constexpr double loaded_to_loaded = 0.2;
inline constexpr double fair_emission_prob = 1.0 / num_symbols;
inline constexpr double loaded_normal_prob = 0.125;
inline constexpr double loaded_biased_prob = 0.375;
} // namespace dishonest_casino

inline void prepare_two_state_hmm(Hmm &hmm) {
    if (hmm.getNumStatesModern() != dishonest_casino::num_states) {
        throw std::invalid_argument("Dishonest casino helper expects a 2-state HMM");
    }

    Vector pi(dishonest_casino::num_states);
    pi(dishonest_casino::fair_state) = dishonest_casino::fair_initial_prob;
    pi(dishonest_casino::loaded_state) = dishonest_casino::loaded_initial_prob;

    Matrix trans(dishonest_casino::num_states, dishonest_casino::num_states);
    trans(dishonest_casino::fair_state, dishonest_casino::fair_state) =
        dishonest_casino::fair_to_fair;
    trans(dishonest_casino::fair_state, dishonest_casino::loaded_state) =
        dishonest_casino::fair_to_loaded;
    trans(dishonest_casino::loaded_state, dishonest_casino::fair_state) =
        dishonest_casino::loaded_to_fair;
    trans(dishonest_casino::loaded_state, dishonest_casino::loaded_state) =
        dishonest_casino::loaded_to_loaded;

    hmm.setPi(pi);
    hmm.setTrans(trans);

    auto fairDist = std::make_unique<DiscreteDistribution>(dishonest_casino::num_symbols);
    for (std::size_t i = 0; i < dishonest_casino::num_symbols; ++i) {
        fairDist->setProbability(static_cast<int>(i), dishonest_casino::fair_emission_prob);
    }
    hmm.setDistribution(dishonest_casino::fair_state, std::move(fairDist));

    auto loadedDist = std::make_unique<DiscreteDistribution>(dishonest_casino::num_symbols);
    for (std::size_t i = 0; i < dishonest_casino::num_symbols - 1; ++i) {
        loadedDist->setProbability(static_cast<int>(i), dishonest_casino::loaded_normal_prob);
    }
    loadedDist->setProbability(static_cast<int>(dishonest_casino::num_symbols - 1),
                               dishonest_casino::loaded_biased_prob);
    hmm.setDistribution(dishonest_casino::loaded_state, std::move(loadedDist));

    hmm.validate();
}

inline std::unique_ptr<Hmm> create_two_state_hmm() {
    auto hmm = std::make_unique<Hmm>(dishonest_casino::num_states);
    prepare_two_state_hmm(*hmm);
    return hmm;
}

} // namespace libhmm::examples
