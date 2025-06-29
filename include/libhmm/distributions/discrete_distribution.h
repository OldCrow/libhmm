#ifndef DISCRETEDISTRIBUTION_H_
#define DISCRETEDISTRIBUTION_H_

#include "libhmm/distributions/probability_distribution.h"
#include "libhmm/common/common.h"
// Common.h already includes: <iostream>, <cmath>, <stdexcept>, <cstddef>, <cassert>, <vector>, <algorithm>, <sstream>, <iomanip>
#include <numeric>     // For std::accumulate (not in common.h)

namespace libhmm{

/**
 * Modern C++17 Discrete distribution for modeling categorical data.
 * 
 * The Discrete distribution (also known as Categorical distribution) is a
 * discrete probability distribution that generalizes the Bernoulli distribution.
 * It describes the possible results of a random variable that can take on
 * one of K possible categories, with the probability of each category separately specified.
 * 
 * PMF: P(X = k) = p_k for k ∈ {0, 1, 2, ..., K-1}
 * where p_k is the probability of category k and ∑p_k = 1
 * 
 * Properties:
 * - Support: {0, 1, 2, ..., numSymbols-1}
 * - Probability mass function defined for each discrete symbol
 * - All probabilities must sum to 1.0
 * - Each probability must be in [0, 1]
 * 
 * Applications:
 * - Hidden Markov Models with discrete observations
 * - Classification problems
 * - Multinomial experiments
 * - Any scenario with discrete, mutually exclusive outcomes
 */
class DiscreteDistribution : public ProbabilityDistribution
{    
private:
    /**
     * Number of discrete symbols/categories
     * Must be > 0, typically small (e.g., 2-100)
     */
    std::size_t numSymbols_;
    
    /**
     * Contains probabilities for discrete observations
     * pdf_[i] = P(X = i) for i ∈ {0, 1, ..., numSymbols-1}
     * Using std::vector for modern C++ performance and semantics
     */
    std::vector<double> pdf_;
    
    /**
     * Comprehensive cached values for maximum performance
     */
    mutable double cachedSum_{1.0};              // Sum of all probabilities (validation)
    mutable double cachedEntropy_{0.0};          // Shannon entropy: H(X) = -∑p_i*log(p_i)
    mutable std::vector<double> cachedLogProbs_; // Pre-computed log probabilities
    mutable std::vector<double> cachedCDF_;      // Pre-computed cumulative distribution
    mutable std::size_t cachedMode_{0};          // Index of most probable symbol
    mutable double cachedMaxProb_{0.0};          // Maximum probability value
    mutable std::vector<std::size_t> nonZeroIndices_; // Indices with non-zero probabilities
    
    /**
     * Flag to track if cached values need updating
     */
    mutable bool cacheValid_{false};
    
    /**
     * Updates all cached values when probabilities change
     * Comprehensive caching for maximum performance
     */
    void updateCache() const noexcept {
        // Basic statistics
        cachedSum_ = std::accumulate(pdf_.begin(), pdf_.end(), 0.0);
        
        // Pre-compute log probabilities and entropy
        cachedLogProbs_.resize(numSymbols_);
        cachedEntropy_ = 0.0;
        cachedMaxProb_ = 0.0;
        cachedMode_ = 0;
        nonZeroIndices_.clear();
        
        for (std::size_t i = 0; i < numSymbols_; ++i) {
            const double p = pdf_[i];
            
            // Cache log probabilities
            if (p > 0.0) {
                cachedLogProbs_[i] = std::log(p);
                cachedEntropy_ -= p * cachedLogProbs_[i];
                nonZeroIndices_.push_back(i);
                
                // Track mode (most probable symbol)
                if (p > cachedMaxProb_) {
                    cachedMaxProb_ = p;
                    cachedMode_ = i;
                }
            } else {
                cachedLogProbs_[i] = -std::numeric_limits<double>::infinity();
            }
        }
        
        // Pre-compute CDF
        cachedCDF_.resize(numSymbols_);
        cachedCDF_[0] = pdf_[0];
        for (std::size_t i = 1; i < numSymbols_; ++i) {
            cachedCDF_[i] = cachedCDF_[i-1] + pdf_[i];
        }
        
        cacheValid_ = true;
    }
    
    /**
     * Validates that an observation index is within valid range
     */
    bool isValidIndex(std::size_t index) const noexcept {
        return index < numSymbols_;
    }
    
    /**
     * Validates parameters for the Discrete distribution
     * @param symbols Number of symbols (must be > 0)
     * @throws std::invalid_argument if parameters are invalid
     */
    void validateParameters(std::size_t symbols) const {
        if (symbols == 0) {
            throw std::invalid_argument("Number of symbols must be greater than 0");
        }
    }
    
public:    
    /**
     * Constructs a Discrete distribution with given number of symbols.
     * Initializes to uniform distribution.
     * 
     * @param symbols Number of discrete symbols/categories (must be > 0)
     * @throws std::invalid_argument if symbols == 0
     */
    explicit DiscreteDistribution(std::size_t symbols = 10)
        : numSymbols_{symbols}, pdf_(numSymbols_), 
          cachedSum_{1.0}, cachedEntropy_{0.0}, cacheValid_{false} {
        validateParameters(symbols);
        reset();
    }
    
    /**
     * Copy constructor
     */
    DiscreteDistribution(const DiscreteDistribution& other) 
        : numSymbols_{other.numSymbols_}, pdf_{other.pdf_}, 
          cachedSum_{other.cachedSum_}, cachedEntropy_{other.cachedEntropy_}, 
          cachedLogProbs_{other.cachedLogProbs_}, cachedCDF_{other.cachedCDF_},
          cachedMode_{other.cachedMode_}, cachedMaxProb_{other.cachedMaxProb_},
          nonZeroIndices_{other.nonZeroIndices_}, cacheValid_{other.cacheValid_} {}
    
    /**
     * Copy assignment operator
     */
    DiscreteDistribution& operator=(const DiscreteDistribution& other) {
        if (this != &other) {
            numSymbols_ = other.numSymbols_;
            pdf_ = other.pdf_;
            cachedSum_ = other.cachedSum_;
            cachedEntropy_ = other.cachedEntropy_;
            cachedLogProbs_ = other.cachedLogProbs_;
            cachedCDF_ = other.cachedCDF_;
            cachedMode_ = other.cachedMode_;
            cachedMaxProb_ = other.cachedMaxProb_;
            nonZeroIndices_ = other.nonZeroIndices_;
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }
    
    /**
     * Move constructor
     */
    DiscreteDistribution(DiscreteDistribution&& other) noexcept
        : numSymbols_{other.numSymbols_}, pdf_{std::move(other.pdf_)}, 
          cachedSum_{other.cachedSum_}, cachedEntropy_{other.cachedEntropy_}, 
          cachedLogProbs_{std::move(other.cachedLogProbs_)}, 
          cachedCDF_{std::move(other.cachedCDF_)},
          cachedMode_{other.cachedMode_}, cachedMaxProb_{other.cachedMaxProb_},
          nonZeroIndices_{std::move(other.nonZeroIndices_)}, cacheValid_{other.cacheValid_} {}
    
    /**
     * Move assignment operator
     */
    DiscreteDistribution& operator=(DiscreteDistribution&& other) noexcept {
        if (this != &other) {
            numSymbols_ = other.numSymbols_;
            pdf_ = std::move(other.pdf_);
            cachedSum_ = other.cachedSum_;
            cachedEntropy_ = other.cachedEntropy_;
            cachedLogProbs_ = std::move(other.cachedLogProbs_);
            cachedCDF_ = std::move(other.cachedCDF_);
            cachedMode_ = other.cachedMode_;
            cachedMaxProb_ = other.cachedMaxProb_;
            nonZeroIndices_ = std::move(other.nonZeroIndices_);
            cacheValid_ = other.cacheValid_;
        }
        return *this;
    }

    /**
     * Destructor - explicitly defaulted to satisfy Rule of Five
     */
    ~DiscreteDistribution() override = default;

    /**
     * Gets the probability mass function value for a discrete observation.
     * 
     * @param value The discrete value (will be cast to integer index)
     * @return Probability mass for the given value, 0.0 if out of range
     */
    double getProbability(double x) override;

    /**
     * Fits the distribution to observed data using maximum likelihood estimation.
     * Computes empirical probabilities: P(X = k) = count(k) / total_count
     * 
     * @param values Vector of observed discrete values
     */
    void fit(const std::vector<Observation>& values) override;

    /**
     * Resets the distribution to uniform probabilities.
     * Each symbol gets probability 1/numSymbols
     */
    void reset() noexcept override;

    /**
     * Sets the probability for a specific discrete observation.
     * 
     * @param o The discrete observation (symbol index)
     * @param value The probability value (must be in [0,1])
     * @throws std::invalid_argument if value is not a valid probability
     * @throws std::out_of_range if observation index is out of range
     */
    void setProbability(Observation o, double value);

    /**
     * Returns a string representation of the distribution.
     * 
     * @return String showing all symbol probabilities
     */
    std::string toString() const override;

    /**
     * Gets the number of discrete symbols in the distribution.
     * 
     * @return Number of symbols/categories
     */
    std::size_t getNumSymbols() const noexcept { return numSymbols_; }
    
    /**
     * Gets the probability for a specific symbol.
     * 
     * @param index Symbol index (must be < numSymbols)
     * @return Probability for the symbol
     * @throws std::out_of_range if index is out of range
     */
    double getProbability(std::size_t index) const {
        if (!isValidIndex(index)) {
            throw std::out_of_range("Symbol index out of range");
        }
        return pdf_[index];
    }
    
    /**
     * Gets the sum of all probabilities (should be approximately 1.0).
     * 
     * @return Sum of all probabilities
     */
    double getProbabilitySum() const {
        if (!cacheValid_) {
            updateCache();
        }
        return cachedSum_;
    }
    
    /**
     * Gets the entropy of the distribution: H(X) = -∑p_i*log(p_i).
     * Higher entropy indicates more uncertainty/randomness.
     * 
     * @return Entropy in nats (natural logarithm base)
     */
    double getEntropy() const {
        if (!cacheValid_) {
            updateCache();
        }
        return cachedEntropy_;
    }
    
    /**
     * Gets the mode of the distribution (most likely symbol).
     * Uses cached value for O(1) performance.
     * 
     * @return Index of the symbol with highest probability
     */
    std::size_t getMode() const {
        if (!cacheValid_) {
            updateCache();
        }
        return cachedMode_;
    }
    
    /**
     * Gets the mean of the distribution.
     * For discrete distribution, mean = ∑(i * p_i) for i = 0 to numSymbols-1
     * 
     * @return Mean value
     */
    double getMean() const noexcept {
        double mean = 0.0;
        for (std::size_t i = 0; i < numSymbols_; ++i) {
            mean += static_cast<double>(i) * pdf_[i];
        }
        return mean;
    }
    
    /**
     * Gets the variance of the distribution.
     * For discrete distribution, variance = ∑(i² * p_i) - mean²
     * 
     * @return Variance value
     */
    double getVariance() const noexcept {
        const double mean = getMean();
        double secondMoment = 0.0;
        for (std::size_t i = 0; i < numSymbols_; ++i) {
            const double iDouble = static_cast<double>(i);
            secondMoment += iDouble * iDouble * pdf_[i];
        }
        return secondMoment - mean * mean;
    }
    
    /**
     * Gets the standard deviation of the distribution.
     * 
     * @return Standard deviation value
     */
    double getStandardDeviation() const noexcept {
        return std::sqrt(getVariance());
    }
    
    /**
     * Normalizes the distribution so probabilities sum to 1.0.
     * Useful after manual probability modifications.
     */
    void normalize() {
        if (!cacheValid_) {
            updateCache();
        }
        
        if (cachedSum_ > 0.0) {
            for (double& p : pdf_) {
                p /= cachedSum_;
            }
            cacheValid_ = false; // Need to recalculate after normalization
        }
    }
    
    /**
     * Evaluates the logarithm of the probability mass function
     * More numerically stable for small probabilities
     * 
     * @param value The discrete value (will be cast to integer index)
     * @return Log probability mass, -infinity if out of range or probability is 0
     */
    [[nodiscard]] double getLogProbability(double value) const noexcept override;
    
    /**
     * Evaluates the CDF at k using cumulative sum approach
     * Formula: CDF(k) = ∑(i=0 to k) P(X = i)
     * 
     * @param value The value at which to evaluate the CDF
     * @return Cumulative probability P(X ≤ value)
     */
    [[nodiscard]] double getCumulativeProbability(double value) noexcept;
    
    /**
     * Equality comparison operator
     * @param other Other distribution to compare with
     * @return true if distributions are equal within tolerance
     */
    bool operator==(const DiscreteDistribution& other) const;
    
    /**
     * Inequality comparison operator
     * @param other Other distribution to compare with
     * @return true if distributions are not equal
     */
    bool operator!=(const DiscreteDistribution& other) const { return !(*this == other); }

private:
    friend std::istream& operator>>(std::istream& is,
            libhmm::DiscreteDistribution& distribution);
};

std::ostream& operator<<( std::ostream&, 
        const libhmm::DiscreteDistribution& );

} // namespace
#endif
