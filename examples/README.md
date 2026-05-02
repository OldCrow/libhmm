# libhmm Examples

This directory contains 13 comprehensive examples demonstrating real-world applications of Hidden Markov Models using the libhmm library. Each example showcases different probability distributions and modeling scenarios, covering major application domains from finance to manufacturing.

## Building Examples

```bash
cmake -B build -DBUILD_EXAMPLES=ON
cmake --build build --config Release

# Run a specific example
./build/examples/Release/basic_hmm_example
./build/examples/Release/baum_welch_example
./build/examples/Release/viterbi_trainer_example
./build/examples/Release/segmental_kmeans_example
./build/examples/Release/student_t_hmm_example
./build/examples/Release/poisson_hmm_example
./build/examples/Release/financial_hmm_example
./build/examples/Release/reliability_hmm_example
./build/examples/Release/quality_control_hmm_example
./build/examples/Release/economics_hmm_example
./build/examples/Release/queuing_theory_hmm_example
./build/examples/Release/statistical_process_control_hmm_example
./build/examples/Release/swarm_coordination_example
```

## Example Overview

### 🎲 [basic_hmm_example.cpp](basic_hmm_example.cpp)
**Topic**: Fundamental HMM operations and distribution testing
**Distributions**: Gaussian, Gamma, Log-Normal, Exponential, Poisson
**Concepts**: Basic HMM construction, probability calculations, Viterbi training
**Use Case**: Learning HMM fundamentals and distribution behavior

**Key Features:**
- Classic "Occasionally Dishonest Casino" example
- Multiple distribution demonstrations
- Forward-backward algorithm comparisons
- Modern C++17 coding patterns

---

### 🎯 [segmental_kmeans_example.cpp](segmental_kmeans_example.cpp)
**Topic**: Hard-assignment training for discrete-emission HMMs
**Distributions**: Discrete
**Concepts**: K-means style training, Baum-Welch warm-start pattern, discrete-only constraint
**Use Case**: Fast initialisation of discrete HMMs before EM refinement

**Key Features:**
- Path A: SegmentalKMeansTrainer used standalone on a 2-state biased-die HMM
- Path B: segmental k-means warm-start followed by Baum-Welch refinement
- Constraint demonstration: trainer rejects non-discrete distributions with `std::runtime_error`
- Log-likelihood comparison across the two training paths

---

### 📊 [poisson_hmm_example.cpp](poisson_hmm_example.cpp)
**Topic**: Count data modeling and event frequency analysis
**Distributions**: Poisson
**Concepts**: Discrete count modeling, state inference, parameter estimation
**Use Case**: Website traffic analysis, call center modeling, rare event detection

**Key Features:**
- Website traffic state detection (normal vs. high traffic)
- Real-world observation sequences
- Training with synthetic count data
- Comprehensive application examples

**Applications:**
- Web traffic analysis and anomaly detection
- Call center volume modeling
- Network packet arrival modeling
- Quality control (defect counting)

---

### 💰 [financial_hmm_example.cpp](financial_hmm_example.cpp)
**Topic**: Financial market regime detection and volatility modeling
**Distributions**: Beta, Log-Normal
**Concepts**: Market state transitions, volatility regimes, returns modeling
**Use Case**: Algorithmic trading, risk management, market analysis

**Key Features:**
- Volatility modeling with Beta distribution (bounded [0,1])
- Returns modeling with Log-Normal distribution (always positive)
- Bull/bear market detection
- Regime persistence analysis

**Applications:**
- Market regime detection (bull/bear identification)
- Volatility forecasting and risk management
- Portfolio optimization under uncertainty
- Options pricing with stochastic volatility

---

### 🔧 [reliability_hmm_example.cpp](reliability_hmm_example.cpp)
**Topic**: System reliability and failure analysis
**Distributions**: Weibull, Exponential
**Concepts**: Hazard rates, lifetime modeling, degradation processes
**Use Case**: Predictive maintenance, quality control, safety analysis

**Key Features:**
- Component lifetime modeling with Weibull distribution
- Failure rate analysis with Exponential distribution
- Three-state degradation model (Normal → Degraded → Critical)
- Hazard rate interpretation (β parameter analysis)

**Applications:**
- Predictive maintenance scheduling
- System health monitoring and diagnostics
- Warranty analysis and cost estimation
- Infrastructure asset management

---

### 🏭 [quality_control_hmm_example.cpp](quality_control_hmm_example.cpp)
**Topic**: Manufacturing quality control and statistical process control
**Distributions**: Binomial, Uniform
**Concepts**: Defect counting, tolerance analysis, process monitoring
**Use Case**: Manufacturing quality assurance, process control

**Key Features:**
- Defect count modeling with Binomial distribution
- Measurement tolerance analysis with Uniform distribution
- Statistical process control (SPC) concepts
- Control chart limit calculations

**Applications:**
- Manufacturing process monitoring
- Automated quality inspection systems
- Statistical process control (SPC)
- Supply chain quality assessment

---

### 📈 [economics_hmm_example.cpp](economics_hmm_example.cpp)
**Topic**: Economic and social science modeling
**Distributions**: Negative Binomial, Pareto
**Concepts**: Overdispersed count data, power-law phenomena, inequality analysis
**Use Case**: Customer behavior analysis, economic regime detection, social science research

**Key Features:**
- Customer purchase behavior with Negative Binomial distribution (overdispersed counts)
- Income distribution modeling with Pareto distribution (power-law, inequality)
- Economic regime detection (normal vs. crisis economics)
- Statistical insights into distribution properties

**Applications:**
- Customer lifetime value modeling
- Income inequality measurement and monitoring
- Social media engagement analysis
- Economic crisis detection and response

---

### ⏱️ [queuing_theory_hmm_example.cpp](queuing_theory_hmm_example.cpp)
**Topic**: Service systems and queuing theory analysis
**Distributions**: Poisson, Exponential, Gamma
**Concepts**: Service load modeling, arrival processes, performance metrics
**Use Case**: Call center optimization, system performance analysis, service design

**Key Features:**
- Customer arrival modeling with Poisson distribution (load states)
- Service time analysis with Exponential distribution (M/M/1 queues)
- Advanced service modeling with Gamma distribution (M/G/1 queues)
- Performance metrics calculation (traffic intensity, queue length)
- 24-hour service pattern analysis

**Applications:**
- Call center staffing and performance optimization
- Computer system performance modeling
- Hospital emergency department management
- Network traffic analysis and capacity planning

---

### 💪 [student_t_hmm_example.cpp](student_t_hmm_example.cpp)
**Topic**: Heavy-tailed financial modeling and outlier-robust analysis
**Distributions**: Student's t-distribution
**Concepts**: Heavy tails, robust statistics, financial outliers, location-scale modeling
**Use Case**: Robust financial modeling, risk management with extreme events

**Key Features:**
- Student's t-distribution for heavy-tailed financial returns
- Location (μ), scale (σ), and degrees of freedom (ν) parameter interpretation
- Outlier-robust modeling compared to Gaussian distributions
- Financial crisis and normal market regime detection
- Fat tail analysis and extreme event modeling

**Applications:**
- Robust portfolio optimization under heavy-tailed returns
- Financial risk management with extreme events
- Credit risk modeling with default clustering
- Commodity price modeling with supply shocks
- Insurance claims modeling with catastrophic events

---

### 🔬 [statistical_process_control_hmm_example.cpp](statistical_process_control_hmm_example.cpp)
**Topic**: Advanced statistical process control and quality monitoring
**Distributions**: Chi-squared distribution
**Concepts**: Goodness-of-fit testing, categorical analysis, process variation monitoring
**Use Case**: Advanced quality control, hypothesis testing, process capability analysis

**Key Features:**
- Chi-squared distribution for goodness-of-fit testing
- Degrees of freedom parameter interpretation
- Statistical hypothesis testing integration
- Process capability monitoring
- Categorical data analysis for quality attributes

**Applications:**
- Advanced statistical process control (SPC)
- Hypothesis testing in manufacturing
- Categorical quality attribute analysis
- Process capability studies
- Six Sigma quality improvement projects

---

## Common Patterns

All examples follow consistent patterns:

1. **HMM Configuration**: State setup, transition matrices, initial probabilities
2. **Distribution Assignment**: Appropriate distributions for each state
3. **Probability Demonstrations**: Calculate and display probabilities for different observations
4. **Viterbi Decoding**: Find most likely state sequences for observation data
5. **Training Examples**: Learn parameters from synthetic or real data
6. **Real-World Applications**: Practical use cases and interpretations

## Mathematical Insights

Each example includes:
- **Parameter Interpretation**: What distribution parameters mean in context
- **Statistical Analysis**: Control limits, confidence intervals, significance tests
- **Domain Knowledge**: Industry-specific insights and best practices
- **Decision Rules**: When to take action based on state inference

## Getting Started

1. **Start with `basic_hmm_example.cpp`** to understand fundamental concepts
2. **Choose domain-specific examples** based on your application:
   - Count data → `poisson_hmm_example.cpp`
   - Financial data → `financial_hmm_example.cpp` or `student_t_hmm_example.cpp`
   - Reliability data → `reliability_hmm_example.cpp`
   - Quality data → `quality_control_hmm_example.cpp` or `statistical_process_control_hmm_example.cpp`
   - Economics → `economics_hmm_example.cpp`
   - Service systems → `queuing_theory_hmm_example.cpp`
3. **Adapt examples** to your specific data and requirements
4. **Combine concepts** from multiple examples as needed

## Distribution Selection Guide

| Data Type | Distribution | Example |
|-----------|-------------|---------|
| Count data (0, 1, 2, ...) | Poisson, Binomial | Website hits, defects |
| Continuous positive | Exponential, Weibull, Gamma | Lifetimes, waiting times |
| Bounded [0,1] | Beta, Uniform | Probabilities, proportions |
| Unbounded continuous | Gaussian, Log-Normal | Measurements, returns |
| Categorical | Discrete | Symbols, classes |

## Next Steps

- Modify examples with your own data
- Experiment with different state numbers
- Try other distributions from the library
- Combine multiple observation types
- Implement custom training algorithms
