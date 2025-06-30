#ifndef CALCULATOR_H_
#define CALCULATOR_H_

#include "libhmm/common/common.h"
#include "libhmm/hmm.h"
#include <stdexcept>
#include <memory>
#include <functional>

namespace libhmm
{

/**
 * @brief Modern C++17 type-safe base class for HMM calculators
 * 
 * This class provides both legacy pointer-based and modern reference-based
 * interfaces for maximum flexibility and type safety. The implementation
 * internally uses std::reference_wrapper for type safety while maintaining
 * backward compatibility with existing pointer-based APIs.
 * 
 * @note New code should prefer the const reference constructors for better
 *       type safety and lifetime management.
 */
class Calculator
{
protected:
    // Modern C++17 implementation using reference wrapper for type safety
    std::reference_wrapper<const Hmm> hmm_ref_;
    ObservationSet observations_;
    
    // Legacy pointer for backward compatibility (derived from reference)
    mutable Hmm* hmm_legacy_ptr_;
    
    /**
     * @brief Internal constructor for reference-based initialization
     * @param hmm_ref Reference wrapper to HMM
     * @param observations Observation sequence
     */
    Calculator(std::reference_wrapper<const Hmm> hmm_ref, const ObservationSet& observations)
        : hmm_ref_(hmm_ref), observations_(observations), 
          hmm_legacy_ptr_(const_cast<Hmm*>(&hmm_ref_.get())) {}
    
public:
    // Legacy protected member access (for existing derived classes)
    // TODO: Remove in v3.0
    [[deprecated("Use getHmmRef() for type safety")]]
    Hmm*& hmm_ = hmm_legacy_ptr_;
    
    /// Default constructor (disabled - calculators must have HMM)
    Calculator() = delete;
    
    /**
     * @brief Modern C++17 type-safe constructor (preferred)
     * @param hmm Const reference to HMM (immutable, lifetime-safe)
     * @param observations Observation sequence
     */
    Calculator(const Hmm& hmm, const ObservationSet& observations)
        : Calculator(std::cref(hmm), observations) {}
    
    /**
     * @brief Legacy pointer-based constructor (deprecated)
     * @param hmm Pointer to the HMM (must not be null)
     * @param observations The observation set to process
     * @throws std::invalid_argument if hmm is null
     * @deprecated Use const reference constructor for better type safety
     */
    [[deprecated("Use const reference constructor for better type safety")]]
    Calculator(Hmm* hmm, const ObservationSet& observations)
        : Calculator(hmm ? std::cref(*hmm) : throw std::invalid_argument("HMM pointer cannot be null"), observations) {}
    
    /// Virtual destructor for proper inheritance
    virtual ~Calculator() = default;
    
    /// Non-copyable but movable
    Calculator(const Calculator&) = delete;
    Calculator& operator=(const Calculator&) = delete;
    Calculator(Calculator&&) = default;
    Calculator& operator=(Calculator&&) = default;
    
    /**
     * @brief Get const reference to HMM (modern, type-safe)
     * @return Const reference to the HMM
     */
    const Hmm& getHmmRef() const noexcept { return hmm_ref_.get(); }
    
    /**
     * @brief Get HMM pointer (legacy compatibility)
     * @return Pointer to the HMM
     * @deprecated Use getHmmRef() for better type safety
     */
    [[deprecated("Use getHmmRef() for better type safety")]]
    Hmm* getHmm() const noexcept { return hmm_legacy_ptr_; }
    
    /**
     * @brief Get const reference to observations
     * @return The observation set
     */
    const ObservationSet& getObservations() const noexcept { return observations_; }
    
    /**
     * @brief Set new observations
     * @param observations The new observation set
     */
    void setObservations(const ObservationSet& observations) {
        observations_ = observations;
    }
}; //class Calculator

}//namespace

#endif //CALCULATOR_H_
