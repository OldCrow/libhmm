#pragma once

// v4 compatibility header.
//
// Hmm is now a type alias for BasicHmm<double>.  All v3 code that uses
// Hmm continues to compile unchanged: constructors, setters, getters,
// validate(), JSON I/O, and stream operators are all unaffected.

#include "libhmm/basic_hmm.h"
#include <iostream> // for stream operator signatures
#include <random>   // sample() (#44)
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace libhmm {

/// @brief Scalar HMM type alias (v3 compatibility).
///
/// Hmm is BasicHmm<double>. All v3 code that constructs, trains, scores,
/// or serialises an Hmm continues to compile unchanged.
using Hmm = BasicHmm<double>;

/// @brief Multivariate HMM alias.
///
/// Each observation is a non-owning row view (ObservationVectorView =
/// std::span<const double>) into an ObservationMatrix sequence.
/// Emission distributions must be set explicitly via setDistribution();
/// the default constructor leaves emission slots null for non-scalar Obs.
using HmmMV = BasicHmm<ObservationVectorView>;

/// @brief Explicit deep copy of a scalar HMM — see BasicHmm::clone() (#43).
[[nodiscard]] inline Hmm clone_hmm(const Hmm &h) {
    return h.clone();
}

/// @brief Explicit deep copy of a multivariate HMM — see BasicHmm::clone() (#43).
[[nodiscard]] inline HmmMV clone_hmm(const HmmMV &h) {
    return h.clone();
}

namespace detail {

/// Draw an index from an unnormalised non-negative weight row (inverse CDF).
/// Throws if the row does not sum to a positive value — sampling from an
/// HMM whose π or transition rows are still zero (default-constructed and
/// never set) is a caller error that must not fail silently.
template <typename RowAccess>
[[nodiscard]] inline StateIndex draw_categorical(RowAccess weight, std::size_t n,
                                                 std::mt19937_64 &rng) {
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        total += weight(i);
    if (!(total > 0.0)) {
        throw std::runtime_error(
            "sample: probability row sums to zero — HMM parameters not initialised");
    }
    const double u = std::uniform_real_distribution<double>(0.0, total)(rng);
    double cum = 0.0;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        cum += weight(i);
        if (u < cum)
            return static_cast<StateIndex>(i);
    }
    return static_cast<StateIndex>(n - 1); // absorbs FP rounding at the top end
}

} // namespace detail

/**
 * @brief Sample one observation sequence of length T from a scalar HMM (#44).
 *
 * s0 ~ Categorical(pi); per step, o_t ~ emission(s_t) and
 * s_{t+1} ~ Categorical(A[s_t, .]). Rows are treated as unnormalised
 * weights; a zero pi or transition row throws.
 *
 * @return {observations (length T), state sequence (length T)}.
 */
[[nodiscard]] inline std::pair<ObservationSet, StateSequence> sample(const Hmm &hmm, std::size_t T,
                                                                     std::mt19937_64 &rng) {
    const std::size_t n = hmm.getNumStatesModern();
    ObservationSet obs(T);
    StateSequence states(T);
    if (T == 0)
        return {obs, states};

    const Matrix &trans = hmm.getTrans();
    const Vector &pi = hmm.getPi();
    StateIndex s = detail::draw_categorical([&pi](std::size_t i) { return pi(i); }, n, rng);
    for (std::size_t t = 0; t < T; ++t) {
        states(t) = s;
        obs(t) = hmm.getDistribution(static_cast<std::size_t>(s)).sample(rng);
        if (t + 1 < T) {
            s = detail::draw_categorical([&trans, s](std::size_t j) { return trans(s, j); }, n,
                                         rng);
        }
    }
    return {obs, states};
}

/**
 * @brief Sample one observation sequence of length T from a multivariate HMM (#44).
 *
 * Same chain as the scalar overload; each observation is one T-by-D matrix
 * row drawn via the emission's sample_mv(). All states must have
 * distributions set (getDistribution throws on a null slot) and agree on
 * getDimension().
 *
 * @return {observations (T-by-D matrix), state sequence (length T)}.
 */
[[nodiscard]] inline std::pair<ObservationMatrix, StateSequence>
sample(const HmmMV &hmm, std::size_t T, std::mt19937_64 &rng) {
    const std::size_t n = hmm.getNumStatesModern();
    if (T == 0)
        return {ObservationMatrix(), StateSequence()};

    const std::size_t d = hmm.getDistribution(0).getDimension();
    ObservationMatrix obs(T, d);
    StateSequence states(T);

    const Matrix &trans = hmm.getTrans();
    const Vector &pi = hmm.getPi();
    StateIndex s = detail::draw_categorical([&pi](std::size_t i) { return pi(i); }, n, rng);
    for (std::size_t t = 0; t < T; ++t) {
        states(t) = s;
        const std::vector<double> x =
            hmm.getDistribution(static_cast<std::size_t>(s)).sample_mv(rng);
        if (x.size() != d) {
            throw std::runtime_error("sample: state dimensions disagree (state " +
                                     std::to_string(s) + " drew " + std::to_string(x.size()) +
                                     ", expected " + std::to_string(d) + ")");
        }
        for (std::size_t k = 0; k < d; ++k)
            obs(t, k) = x[k];
        if (t + 1 < T) {
            s = detail::draw_categorical([&trans, s](std::size_t j) { return trans(s, j); }, n,
                                         rng);
        }
    }
    return {obs, states};
}

/// Legacy stream I/O operators (scalar HMM only).
/// Prefer JSON I/O (hmm_json.h) for new code.
std::ostream &operator<<(std::ostream &, const Hmm &);
std::istream &operator>>(std::istream &, Hmm &);

} // namespace libhmm
