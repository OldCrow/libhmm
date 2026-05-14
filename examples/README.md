# libhmm Examples

20 examples in two categories: algorithm and distribution demonstrations using
synthetic data, and real-world benchmarks against published datasets and
established R packages.

---

## Building

```bash
cmake -B build -DBUILD_EXAMPLES=ON
cmake --build build --config Release

# Run any example
./build/examples/basic_hmm_example
./build/examples/elk_movement_example /tmp    # data-dependent examples need a data dir
./build/examples/earthquake_example           # data is embedded — no download needed
```

Data preparation scripts for the benchmark examples are in `scripts/`:

```bash
Rscript scripts/prepare_elk_data.R      # elk movement   → /tmp/elk_*_obs.csv
Rscript scripts/prepare_dax_data.R      # DAX + S&P 500  → /tmp/dax_logreturns.csv, sp500_logreturns.csv
Rscript scripts/prepare_wind_data.R     # NOAA wind data → /tmp/ohare_wind_2015.csv
```

---

## Part I — Algorithm and Distribution Demonstrations

Synthetic or illustrative data. Start here to learn the library.

### Training algorithms

#### [basic_hmm_example.cpp](basic_hmm_example.cpp)
Classic "Occasionally Dishonest Casino" HMM. Covers construction, probability
calculations, and Viterbi training. Entry point for learning the API.
**Distributions:** Gaussian, Gamma, Log-Normal, Exponential, Poisson

#### [baum_welch_example.cpp](baum_welch_example.cpp)
Baum-Welch EM on synthetic two-cluster data. Prints log-likelihood at each
iteration, verifies monotonic improvement, and contrasts with Viterbi training.
**Distributions:** Gaussian

#### [viterbi_trainer_example.cpp](viterbi_trainer_example.cpp)
Hard-assignment training with `TrainingConfig` presets (`fast`, `balanced`,
`precise`) and a custom config. Reports convergence and max-iteration flags.
**Distributions:** Gaussian, Discrete

#### [segmental_kmeans_example.cpp](segmental_kmeans_example.cpp)
`SegmentalKMeansTrainer` standalone and as a Baum-Welch warm-start.
Demonstrates the discrete-only constraint.
**Distributions:** Discrete

#### [posterior_decoding_example.cpp](posterior_decoding_example.cpp)
`decodePosterior()` vs `ViterbiCalculator::decode()` on the casino HMM.
Shows time steps where the strategies diverge. Also demonstrates
`evaluate_model()` for AIC / BIC / AICc.
**Distributions:** Discrete
- Use `decodePosterior()` when per-step annotation accuracy matters (gene prediction)
- Use Viterbi when whole-sequence coherence is required (speech alignment)

#### [map_baum_welch_example.cpp](map_baum_welch_example.cpp)
MAP-EM Baum-Welch with Dirichlet priors. Contrasts `c = 0` (MLE) with
`c = 1` (Laplace smoothing). Shows that `logL + computeLogPrior()` is the
correct convergence criterion when `c > 0`.
**Distributions:** Discrete

---

### Domain applications (synthetic data)

| Example | Distributions | Domain |
|---|---|---|
| [poisson_hmm_example.cpp](poisson_hmm_example.cpp) | Poisson | Website traffic, call centers, rare events |
| [financial_hmm_example.cpp](financial_hmm_example.cpp) | Beta, Log-Normal | Volatility regimes, options pricing |
| [student_t_hmm_example.cpp](student_t_hmm_example.cpp) | Student-t | Heavy-tailed returns, financial crises |
| [reliability_hmm_example.cpp](reliability_hmm_example.cpp) | Weibull, Exponential | Predictive maintenance, failure analysis |
| [quality_control_hmm_example.cpp](quality_control_hmm_example.cpp) | Binomial, Uniform | SPC, defect counting, tolerance analysis |
| [economics_hmm_example.cpp](economics_hmm_example.cpp) | Negative Binomial, Pareto | Overdispersion, power-law phenomena |
| [queuing_theory_hmm_example.cpp](queuing_theory_hmm_example.cpp) | Poisson, Exponential, Gamma | M/M/1 and M/G/1 queues, 24-hour patterns |
| [statistical_process_control_hmm_example.cpp](statistical_process_control_hmm_example.cpp) | Chi-squared | Goodness-of-fit, Six Sigma |
| [swarm_coordination_example.cpp](swarm_coordination_example.cpp) | Discrete (243 symbols) | Drone formation control, mission states |

---

## Part II — Real-World Benchmarks

Published datasets with known results. Each example fits libhmm against an
established R reference package on the same data, reporting parameter
estimates, log-likelihood, and wall time.

### Ecology

#### [elk_movement_example.cpp](elk_movement_example.cpp)
**Joint Gamma + von Mises HMM on elk GPS tracks**

Fits behavioral states (encamped / travelling) to step lengths and turning
angles from 4 elk (Morales et al. 2004), the canonical dataset for the
`moveHMM` R package. Uses a custom joint Baum-Welch EM with conditional
independence of step length and angle given state.

| | libhmm | moveHMM |
|---|---|---|
| Encamped step mean | 377 m | 374 m |
| Travelling step mean | 3189 m | 3247 m |
| Encamped angle κ | 0.595 | 0.592 |
| Wall time | **99 ms** | ~2000 ms |

**Data:** `Rscript scripts/prepare_elk_data.R`
**Reference:** Michelot et al. (2016), *Methods in Ecology and Evolution*

---

### Finance

#### [dax_regime_example.cpp](dax_regime_example.cpp)
**3-state Student-t HMM on DAX daily log-returns, 2000–2022**

Fits bearish / neutral / bullish market regimes using StudentTDistribution
with ECME (scale-mixture EM for ν, μ, σ). Direct comparison to the `fHMM`
R package reference fit.

| | libhmm | fHMM |
|---|---|---|
| Bearish σ | 0.02628 | 0.02629 |
| Bearish ν | 11.14 | 11.16 |
| Bullish ν | 5.35 | 5.316 |
| Log-likelihood | **17487.2** | 17485.7 |
| Wall time | **~2 s** | ~1360 s |

**Data:** `Rscript scripts/prepare_dax_data.R`
**Reference:** Oelschläger et al. (2024), *J. Statistical Software*

---

#### [sp500_regime_example.cpp](sp500_regime_example.cpp)
**Same 3-state Student-t model on S&P 500, 2000–2022**

Cross-market comparison using the identical model. S&P 500 shows lower
bearish σ (0.023 vs DAX 0.026) and lighter tails in the bearish state
(ν ≈ 6 vs DAX ν ≈ 11), reflecting structural differences in US vs German
equity risk and liquidity.

**Data:** generated by `prepare_dax_data.R` alongside the DAX file.

---

### Earth science

#### [earthquake_example.cpp](earthquake_example.cpp)
**2-state Poisson HMM on annual major earthquake counts, 1900–2006**

The canonical running example from Zucchini & MacDonald (2009), used
throughout chapters 3–7 of their textbook. The 107 annual counts are
embedded in the source — no download required. Results match the
`HiddenMarkov` R package to four significant figures.

| | libhmm | HiddenMarkov |
|---|---|---|
| λ low (quiet) | 15.419 | 15.418 |
| λ high (active) | 26.015 | 26.013 |
| Log-likelihood | −341.879 | −341.879 |
| Wall time | **4 ms** | ~20 ms |

**Data:** embedded in source (no download needed)
**Reference:** Zucchini & MacDonald (2009), *Hidden Markov Models for Time Series*

---

#### [wind_direction_example.cpp](wind_direction_example.cpp)
**2-state VonMisesDistribution HMM on hourly wind directions, O'Hare 2015**

Demonstrates why VonMisesDistribution is the correct distribution for
circular data. The `HiddenMarkov` R package uses a Normal approximation that
fails at the 0°/360° boundary. This example runs both models and quantifies
the error empirically.

**Measured disagreement between Normal and VonMises models on 11,894 hours:**

| Direction bin | Disagreement rate |
|---|---|
| 0°–300° (all directions) | 0% |
| 300°–330° (NNW) | 31% |
| **330°–360° (NNW/N)** | **100%** |

990 NNW-to-N wind hours are misclassified 100% of the time by the Normal
model. A direction 19° from the NNE state mean is 11.2 standard deviations
away under Normal (log-likelihood = −61.9, effectively zero probability).
VonMisesDistribution evaluates cos(−19°) = 0.75 — and assigns all 990
correctly.

**Data:** `Rscript scripts/prepare_wind_data.R`
**Reference:** NOAA NCEI Integrated Surface Database; Zucchini et al. (2017),
*Hidden Markov Models for Time Series, 2nd ed.* (Ch. 10: Wind direction)

---

## Getting Started

**Learning the API:** `basic_hmm_example` → `baum_welch_example`

**Choosing a distribution:**

| Data type | Distribution | Example |
|---|---|---|
| Count data (0, 1, 2 …) | Poisson, Binomial | `poisson_hmm_example`, `earthquake_example` |
| Continuous positive | Gamma, Weibull, Exponential, Rayleigh | `reliability_hmm_example`, `elk_movement_example` |
| Bounded [0, 1] | Beta, Uniform | `financial_hmm_example` |
| Unbounded continuous | Gaussian, Log-Normal, Student-t | `baum_welch_example`, `dax_regime_example` |
| **Circular / directional** | **VonMises** | **`wind_direction_example`**, `elk_movement_example` |
| Categorical | Discrete | `basic_hmm_example`, `swarm_coordination_example` |

**Choosing a trainer:**

| Situation | Trainer |
|---|---|
| Standard fitting | `BaumWelchTrainer` |
| Sparse data / prevent zero transitions | `MapBaumWelchTrainer` |
| Fast convergence, well-separated states | `ViterbiTrainer` |
| Discrete HMM initialisation | `SegmentalKMeansTrainer` |

**Choosing a decoder:**

| Goal | Method |
|---|---|
| Minimise per-step error (annotation) | `ForwardBackwardCalculator::decodePosterior()` |
| Globally coherent path (segmentation) | `ViterbiCalculator::decode()` |

**Adapting benchmark examples to your own data:**
- Movement / GPS tracks → adapt `elk_movement_example`
- Financial time series → adapt `dax_regime_example` (any index via `prepare_dax_data.R`)
- Count time series → adapt `earthquake_example` (data is embedded; swap the array)
- Circular / directional data → adapt `wind_direction_example`
