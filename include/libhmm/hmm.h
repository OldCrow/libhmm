#ifndef HMM_H_
#define HMM_H_

#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "libhmm/common/common.h"
#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/distributions/gamma_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/distributions/discrete_distribution.h"
#include "libhmm/distributions/poisson_distribution.h"
#include "libhmm/distributions/exponential_distribution.h"
#include "libhmm/distributions/log_normal_distribution.h"
#include "libhmm/distributions/pareto_distribution.h"
#include "libhmm/distributions/beta_distribution.h"
#include "libhmm/common/string_tokenizer.h"

namespace libhmm{

/**
 * Modern C++17 Hidden Markov Model class with enhanced type safety,
 * memory management, and performance optimizations.
 */
class Hmm
{
protected:
    Matrix trans_;
    std::vector<std::unique_ptr<ProbabilityDistribution>> emis_;
    Vector pi_;
    std::size_t states_;

    /// Validates state index bounds
    /// @param state State index to validate
    /// @throws std::out_of_range if state index is invalid
    void validateStateIndex(std::size_t state) const {
        if (state >= states_) {
            throw std::out_of_range("State index " + std::to_string(state) + 
                                   " is out of range [0, " + std::to_string(states_) + ")");
        }
    }

    /// Initializes matrices and vectors to proper sizes
    void initializeMatrices() {
        trans_ = Matrix(states_, states_);
        pi_ = Vector(states_);
        emis_.resize(states_);
        
        clear_matrix(trans_);
        clear_vector(pi_);
        
        // Initialize with default Gaussian distributions
        for (std::size_t i = 0; i < states_; ++i) {
            emis_[i] = std::make_unique<GaussianDistribution>();
        }
    }

public:
    /// Default constructor - creates 4-state HMM
    Hmm() : states_{4} { 
        initializeMatrices();
    }

    /// Constructor with specified number of states
    /// @param numStates Number of states (must be > 0)
    /// @throws std::invalid_argument if numStates is 0
    explicit Hmm(std::size_t numStates) : states_{numStates} {
        if (numStates == 0) {
            throw std::invalid_argument("Number of states must be greater than 0");
        }
        initializeMatrices();
    }

    /// Legacy constructor for backward compatibility
    /// @param numStates Number of states
    /// @throws std::invalid_argument if numStates <= 0
    explicit Hmm(int numStates) {
        if (numStates <= 0) {
            throw std::invalid_argument("Number of states must be greater than 0");
        }
        states_ = static_cast<std::size_t>(numStates);
        initializeMatrices();
    }

    /// Virtual destructor for proper inheritance
    virtual ~Hmm() = default;

    /// Constructor with given transition matrix, emissions, and initial probabilities
    /// @param trans Transition matrix
    /// @param emis Vector of emission distribution unique_ptrs
    /// @param pi Initial state probabilities
    /// @throws std::invalid_argument if dimensions don't match
    Hmm(Matrix trans, std::vector<std::unique_ptr<ProbabilityDistribution>> emis, Vector pi)
        : trans_{std::move(trans)}, emis_{std::move(emis)}, pi_{std::move(pi)} {
        
        if (trans_.size1() != trans_.size2()) {
            throw std::invalid_argument("Transition matrix must be square");
        }
        
        states_ = trans_.size1();
        
        if (emis_.size() != states_ || pi_.size() != states_) {
            throw std::invalid_argument("Emission distributions and pi vector must match number of states");
        }
        
        if (states_ == 0) {
            throw std::invalid_argument("Number of states must be greater than 0");
        }
    }
    
    /// Non-copyable but movable for performance
    Hmm(const Hmm&) = delete;
    Hmm& operator=(const Hmm&) = delete;
    Hmm(Hmm&&) = default;
    Hmm& operator=(Hmm&&) = default;
    
    /// Sets the initial state probability vector
    /// @param pi Initial state probabilities
    /// @throws std::invalid_argument if size doesn't match number of states
    void setPi(const Vector& pi) {
        if (pi.size() != states_) {
            throw std::invalid_argument("Pi vector size must match number of states");
        }
        pi_ = pi;
    }
    
    /// Sets the transition matrix
    /// @param trans Transition matrix
    /// @throws std::invalid_argument if dimensions don't match
    void setTrans(const Matrix& trans) {
        if (trans.size1() != states_ || trans.size2() != states_) {
            throw std::invalid_argument("Transition matrix dimensions must match number of states");
        }
        trans_ = trans;
    }
    
    /// Sets the probability distribution for a specific state
    /// @param state State index
    /// @param distribution Unique pointer to probability distribution
    /// @throws std::out_of_range if state index is invalid
    /// @throws std::invalid_argument if distribution is null
    void setProbabilityDistribution(std::size_t state, std::unique_ptr<ProbabilityDistribution> distribution) {
        validateStateIndex(state);
        if (!distribution) {
            throw std::invalid_argument("Probability distribution cannot be null");
        }
        emis_[state] = std::move(distribution);
    }
    
    /// Legacy setter for backward compatibility
    /// @param state State index
    /// @param distribution Raw pointer to probability distribution (ownership transferred)
    /// @throws std::out_of_range if state index is invalid
    /// @throws std::invalid_argument if distribution is null or state < 0
    void setProbabilityDistribution(int state, ProbabilityDistribution* distribution) {
        if (state < 0) {
            throw std::invalid_argument("State index cannot be negative");
        }
        if (!distribution) {
            throw std::invalid_argument("Probability distribution cannot be null");
        }
        setProbabilityDistribution(static_cast<std::size_t>(state), 
                                 std::unique_ptr<ProbabilityDistribution>(distribution));
    }
    
    /// Gets the probability distribution for a state
    /// @param state State index
    /// @return Pointer to the probability distribution
    /// @throws std::out_of_range if state index is invalid
    const ProbabilityDistribution* getProbabilityDistribution(std::size_t state) const {
        validateStateIndex(state);
        return emis_[state].get();
    }
    
    /// Legacy getter for backward compatibility
    /// @param state State index
    /// @return Pointer to the probability distribution
    /// @throws std::out_of_range if state index is invalid
    /// @throws std::invalid_argument if state < 0
    ProbabilityDistribution* getProbabilityDistribution(int state) const {
        if (state < 0) {
            throw std::invalid_argument("State index cannot be negative");
        }
        return const_cast<ProbabilityDistribution*>(getProbabilityDistribution(static_cast<std::size_t>(state)));
    }
    
    /// Gets the transition matrix
    /// @return The transition matrix
    const Matrix& getTrans() const noexcept { return trans_; }
    
    /// Gets the initial state probability vector
    /// @return The initial state probability vector
    const Vector& getPi() const noexcept { return pi_; }
    
    /// Gets the number of states in the HMM (legacy interface)
    /// @return Number of states as int for backward compatibility
    int getNumStates() const noexcept { return static_cast<int>(states_); }
    
    /// Modern getter for number of states
    /// @return Number of states as size_t
    std::size_t getNumStatesModern() const noexcept { return states_; }
    
    /// Validates HMM consistency
    /// @throws std::runtime_error if HMM is in invalid state
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
                throw std::runtime_error("Emission distribution for state " + std::to_string(i) + " is null");
            }
        }
    }
    
friend std::istream& operator>>(std::istream&, libhmm::Hmm&);
};

// Overload operator<<() to allow direct output to cout
// This shouldn't be a member function for a variety of reasons
// that I don't really understand
//
// WHY THESE OPERATORS SHOULD NOT BE MEMBER FUNCTIONS
// If the stream operators were member functions, you would have to put the Hmm
// object on the left when you called the functions. For example:
//	Hmm h;
//	h >> cin;
//	h << cout;
// While legal, it is contrary to convention and is confusing. For a full
// explanation, see Scott Meyers' book 'Effective C++,' 2nd edition,
// Addison-Wesley, 1998, pages 84-89.
//	- L. Bell
std::ostream& operator<<( std::ostream&, const libhmm::Hmm& );
}

#endif /*HMM_H_*/
