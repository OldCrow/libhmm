#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "libhmm/linalg/linalg_types.h"
#include "libhmm/distributions/distributions.h"
#include "libhmm/common/string_tokenizer.h"

namespace libhmm {

/**
 * @brief Hidden Markov Model parameterised on observation type.
 *
 * @tparam Obs  Observation type.  Default `double` produces the scalar HMM
 *              that is binary-identical to v3's `Hmm` class through the
 *              `using Hmm = BasicHmm<double>` alias in hmm.h.
 *
 * Transition matrix and initial-state vector are always `double` (Matrix /
 * Vector) because state probabilities are independent of observation
 * dimensionality.
 *
 * Emission distributions are owned exclusively via
 * `unique_ptr<BasicEmissionDistribution<Obs>>`.  Tied-parameter HMMs
 * (shared distributions across states) are not currently supported; if
 * needed in future, change storage to `shared_ptr`.
 *
 * Default construction (Obs = double only) initialises each state with a
 * GaussianDistribution.  Non-scalar HMMs leave emission slots null until
 * setDistribution() is called.
 */
template <typename Obs = double>
class BasicHmm {
protected:
    Matrix trans_;
    std::vector<std::unique_ptr<BasicEmissionDistribution<Obs>>> emis_;
    Vector pi_;
    std::size_t states_;

    /// Validates state index bounds.
    void validateStateIndex(std::size_t state) const {
        if (state >= states_) {
            throw std::out_of_range("State index " + std::to_string(state) +
                                    " is out of range [0, " + std::to_string(states_) + ")");
        }
    }

    /// Initialises transition matrix, pi vector, and emission slots.
    /// For Obs = double: fills each emission slot with a default GaussianDistribution.
    /// For other Obs types: leaves emission slots null (setDistribution() required).
    void initializeMatrices() {
        trans_ = Matrix(states_, states_);
        pi_ = Vector(states_);
        emis_.resize(states_);

        clear_matrix(trans_);
        clear_vector(pi_);

        // Only create default Gaussian emissions on the scalar path.
        // Multivariate specialisations must set distributions explicitly.
        if constexpr (std::is_same_v<Obs, double>) {
            for (std::size_t i = 0; i < states_; ++i) {
                emis_[i] = std::make_unique<GaussianDistribution>();
            }
        }
    }

public:
    /// Default constructor — creates a 4-state HMM.
    BasicHmm() : states_{4} { initializeMatrices(); }

    /// Constructor with specified number of states.
    /// @throws std::invalid_argument if numStates is 0.
    explicit BasicHmm(std::size_t numStates) : states_{numStates} {
        if (numStates == 0) {
            throw std::invalid_argument("Number of states must be greater than 0");
        }
        initializeMatrices();
    }

    /// Legacy constructor for backward compatibility.
    /// @throws std::invalid_argument if numStates <= 0.
    explicit BasicHmm(int numStates) {
        if (numStates <= 0) {
            throw std::invalid_argument("Number of states must be greater than 0");
        }
        states_ = static_cast<std::size_t>(numStates);
        initializeMatrices();
    }

    /// Virtual destructor for proper inheritance.
    virtual ~BasicHmm() = default;

    /// Constructor with given transition matrix, emissions, and initial probabilities.
    /// @throws std::invalid_argument if dimensions don't match.
    BasicHmm(Matrix trans, std::vector<std::unique_ptr<BasicEmissionDistribution<Obs>>> emis,
             Vector pi)
        : trans_{std::move(trans)}, emis_{std::move(emis)}, pi_{std::move(pi)} {

        if (trans_.size1() != trans_.size2()) {
            throw std::invalid_argument("Transition matrix must be square");
        }

        states_ = trans_.size1();

        if (emis_.size() != states_ || pi_.size() != states_) {
            throw std::invalid_argument(
                "Emission distributions and pi vector must match number of states");
        }

        if (states_ == 0) {
            throw std::invalid_argument("Number of states must be greater than 0");
        }
    }

    /// Non-copyable but movable.
    BasicHmm(const BasicHmm &) = delete;
    BasicHmm &operator=(const BasicHmm &) = delete;
    BasicHmm(BasicHmm &&) = default;
    BasicHmm &operator=(BasicHmm &&) = default;

    // =========================================================================
    // Setters
    // =========================================================================

    /// Sets the initial state probability vector.
    /// @throws std::invalid_argument if size doesn't match number of states.
    void setPi(const Vector &pi) {
        if (pi.size() != states_) {
            throw std::invalid_argument("Pi vector size must match number of states");
        }
        pi_ = pi;
    }

    /// Sets the transition matrix.
    /// @throws std::invalid_argument if dimensions don't match.
    void setTrans(const Matrix &trans) {
        if (trans.size1() != states_ || trans.size2() != states_) {
            throw std::invalid_argument("Transition matrix dimensions must match number of states");
        }
        trans_ = trans;
    }

    /// Sets the emission distribution for a specific state.
    /// @throws std::out_of_range if state index is invalid.
    /// @throws std::invalid_argument if distribution is null.
    void setDistribution(std::size_t state,
                         std::unique_ptr<BasicEmissionDistribution<Obs>> distribution) {
        validateStateIndex(state);
        if (!distribution) {
            throw std::invalid_argument("Emission distribution cannot be null");
        }
        emis_[state] = std::move(distribution);
    }

    // =========================================================================
    // Getters
    // =========================================================================

    /// Gets the emission distribution for a state (non-const — for trainers).
    /// @throws std::out_of_range if state index is invalid.
    /// @throws std::runtime_error if the emission slot is null (MV HMM requires setDistribution).
    BasicEmissionDistribution<Obs> &getDistribution(std::size_t state) {
        validateStateIndex(state);
        if (!emis_[state])
            throw std::runtime_error("getDistribution: emission distribution for state " +
                                     std::to_string(state) +
                                     " is null; call setDistribution() before use");
        return *emis_[state];
    }

    /// Gets the emission distribution for a state (const).
    /// @throws std::out_of_range if state index is invalid.
    /// @throws std::runtime_error if the emission slot is null (MV HMM requires setDistribution).
    [[nodiscard]] const BasicEmissionDistribution<Obs> &getDistribution(std::size_t state) const {
        validateStateIndex(state);
        if (!emis_[state])
            throw std::runtime_error("getDistribution: emission distribution for state " +
                                     std::to_string(state) +
                                     " is null; call setDistribution() before use");
        return *emis_[state];
    }

    /// Gets the transition matrix.
    [[nodiscard]] const Matrix &getTrans() const noexcept { return trans_; }

    /// Gets the initial state probability vector.
    [[nodiscard]] const Vector &getPi() const noexcept { return pi_; }

    /// Gets the number of states (legacy int interface for backward compatibility).
    [[nodiscard]] int getNumStates() const noexcept { return static_cast<int>(states_); }

    /// Gets the number of states.
    [[nodiscard]] std::size_t getNumStatesModern() const noexcept { return states_; }

    // =========================================================================
    // Validation
    // =========================================================================

    /// Validates HMM consistency.
    /// @throws std::runtime_error if HMM is in invalid state.
    void validate() const {
        if (states_ == 0) {
            throw std::runtime_error("HMM must have at least one state");
        }

        if (trans_.size1() != states_ || trans_.size2() != states_) {
            throw std::runtime_error("Transition matrix dimensions are inconsistent");
        }

        if (pi_.size() != states_) {
            throw std::runtime_error("Pi vector size is inconsistent");
        }

        if (emis_.size() != states_) {
            throw std::runtime_error("Emission distributions size is inconsistent");
        }

        for (std::size_t i = 0; i < states_; ++i) {
            if (!emis_[i]) {
                throw std::runtime_error("Emission distribution for state " + std::to_string(i) +
                                         " is null");
            }
        }
    }
};

} // namespace libhmm
