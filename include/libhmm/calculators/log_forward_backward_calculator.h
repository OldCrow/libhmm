#ifndef LOGFORWARDBACKWARDCALCULATOR_H_
#define LOGFORWARDBACKWARDCALCULATOR_H_

#include <cmath>
#include <cfloat>
#include <limits>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include "libhmm/common/common.h"
#include "libhmm/calculators/forward_backward_calculator.h"
#include "libhmm/hmm.h"

namespace libhmm
{
static constexpr double LOGZERO = std::numeric_limits<double>::quiet_NaN();

class LogForwardBackwardCalculator: public ForwardBackwardCalculator
{
private:
    /// Extended exponential function that handles NaN values
    /// @param x Input value
    /// @return Exponential of x, or 0 if x is NaN
    double eexp(double x) const noexcept {
        if (std::isnan(x)) {
            return 0.0;
        }
        return std::exp(x);
    }

    /// Extended logarithm function that handles zero values
    /// @param x Input value
    /// @return Natural log of x, or LOGZERO if x is 0
    double eln(double x) const noexcept {
        if (x == 0.0) {
            return LOGZERO;
        }
        return std::log(x);
    }

    /// Extended log sum function for numerical stability
    /// @param x First value
    /// @param y Second value  
    /// @return Log of (exp(x) + exp(y))
    double elnsum(double x, double y) const noexcept {
        if (std::isnan(x) || std::isnan(y)) {
            if (std::isnan(x)) {
                return y;
            }
            return x;
        }
        
        if (x > y) {
            return x + eln(1.0 + std::exp(y - x));
        }
        return y + eln(1.0 + std::exp(x - y));
    }

    /// Extended log product function
    /// @param x First value
    /// @param y Second value
    /// @return Log of (exp(x) * exp(y)) = x + y
    double elnproduct(double x, double y) const noexcept {
        if (std::isnan(x) || std::isnan(y)) {
            return LOGZERO;
        }
        return x + y;
    }

protected:
    /// Computes forward variables using log-space forward algorithm
    virtual void forward() override;

    /// Computes backward variables using log-space backward algorithm
    virtual void backward() override;

public:
    /// Default constructor
    LogForwardBackwardCalculator() = default;

    /// Constructor with HMM and observations
    /// @param hmm Pointer to the HMM (must not be null)
    /// @param observations The observation set to process
    /// @throws std::invalid_argument if hmm is null
    LogForwardBackwardCalculator(Hmm* hmm, const ObservationSet& observations)
        : ForwardBackwardCalculator(hmm, observations) {
        forward();
        backward();
    }
    
    /// Virtual destructor
    virtual ~LogForwardBackwardCalculator() = default;
    
    /// Get the forward variables matrix
    /// @return The forward variables matrix
    Matrix getForwardVariables() const noexcept override { return forwardVariables_; }

    /// Get the backward variables matrix
    /// @return The backward variables matrix
    Matrix getBackwardVariables() const noexcept override { return backwardVariables_; }
    
    /// Calculates the probability of the observation set given the HMM
    /// @return The probability value
    virtual double probability() override; 
    
    /// Calculates the log probability of the observation set given the HMM
    /// @return The log probability value
    virtual double logProbability();
};

}

#endif
