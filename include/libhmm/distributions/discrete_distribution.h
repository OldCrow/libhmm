#ifndef DISCRETEDISTRIBUTION_H_
#define DISCRETEDISTRIBUTION_H_

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstddef>
#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"

namespace libhmm{

class DiscreteDistribution : public ProbabilityDistribution
{    
private:
    /*
     * Number of discrete symbols
     */
    std::size_t numSymbols_;
    
    /*
     * Contains probabilities for discrete observations
     */
    Vector pdf_;
    
    /*
     * Validates that an observation index is within valid range
     */
    bool isValidIndex(std::size_t index) const noexcept {
        return index < numSymbols_;
    }
    
public:    
    explicit DiscreteDistribution(std::size_t symbols = 10)
        : numSymbols_{symbols}, pdf_{Vector(numSymbols_)} {
        if (symbols == 0) {
            throw std::invalid_argument("Number of symbols must be greater than 0");
        }
        clear_vector(pdf_);
        reset();
    }

    ~DiscreteDistribution() = default;

    double getProbability(double value) override;

    void fit(const std::vector<Observation>& values) override;

    void reset() noexcept override;

    void setProbability(Observation o, double value);

    std::string toString() const override;

    std::size_t getNumSymbols() const noexcept { return numSymbols_; }
};

std::ostream& operator<<( std::ostream&, 
        const libhmm::DiscreteDistribution& );

} // namespace
#endif
