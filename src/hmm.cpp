#include "libhmm/hmm.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace libhmm {

// All the constructors and core methods are now implemented in the header
// using modern C++17 features. Only the I/O operators need implementation here.

std::ostream &operator<<(std::ostream &os, const libhmm::Hmm &h) {
    const auto numStates = h.getNumStatesModern();

    os << "Hidden Markov Model parameters" << std::endl;
    os << "  States: " << numStates << std::endl;

    os << "  Pi: [";
    for (std::size_t i = 0; i < numStates; ++i) {
        os << " " << std::setw(12) << h.getPi()(i);
    }
    os << " ]" << std::endl;

    os << "  Transmission matrix: " << std::endl;
    for (std::size_t i = 0; i < numStates; ++i) {
        os << "   [";
        for (std::size_t j = 0; j < numStates; ++j) {
            os << std::setw(12) << h.getTrans()(i, j);
        }
        os << " ]" << std::endl;
    }

    os << "  Emissions: " << std::endl;
    for (std::size_t i = 0; i < numStates; ++i) {
        os << "   State " << i << ": ";
        os << h.getDistribution(i).toString() << std::endl;
    }

    return os;
} //operator<<()

// =============================================================================
// operator>> stream-format parsers
//
// operator>>(istream, Hmm) reads the text produced by operator<<(ostream, Hmm).
// It is retained for backward compatibility with the CDATA-wrapped XML format
// written by XMLFileWriter. Both the XML classes and this operator are
// deprecated; prefer hmm_json.h for new code.
//
// Each parse_* function is called with the stream positioned immediately after
// the distribution type keyword (e.g. "Gaussian", "Poisson") has been read.
// The branching complexity of each parser is now measured independently by
// static analysis tools rather than being folded into operator>>'s CC.
//
// NegativeBinomial round-trip: toString() now emits "NegativeBinomial Distribution:"
// (single-token keyword) so the dispatch table can match it directly.  For
// backward compatibility with existing XML files that contain the old two-word
// form "Negative Binomial", the legacy key "Negative" is also registered.
// =============================================================================

namespace {

using StreamParserFn = std::unique_ptr<EmissionDistribution> (*)(std::istream &);

// parse_gaussian: reads all of GaussianDistribution::toString() after the type keyword.
std::unique_ptr<EmissionDistribution> parse_gaussian(std::istream &is) {
    std::string s, t;
    is >> s;                // "Distribution:"
    is >> s >> s >> s >> t; // "\u03bc" "(mean)" "=" VALUE
    const double mean = std::stod(t);
    is >> s >> s >> s >> s >> t; // "\u03c3" "(std." "deviation)" "=" VALUE
    const double sd = std::stod(t);
    is >> s >> s >> t;
    is >> s >> s >> t; // skip Mean, Variance
    return std::make_unique<GaussianDistribution>(mean, sd);
}

std::unique_ptr<EmissionDistribution> parse_discrete(std::istream &is) {
    std::string s, t;
    is >> s; // "Distribution:"
    if (!is)
        throw std::runtime_error("Failed to parse Discrete distribution header");

    is >> t;
    if (!is)
        throw std::runtime_error("Failed to parse Discrete distribution data");

    // Modern format: "Number of symbols = N" followed by "P(i) = value" lines.
    if (t == "Number") {
        std::string of_tok, sym_tok, eq_tok, n_tok;
        is >> of_tok >> sym_tok >> eq_tok >> n_tok;
        if (of_tok != "of" || sym_tok != "symbols" || eq_tok != "=")
            throw std::runtime_error("Malformed Discrete distribution symbol header");
        const auto n = std::stoull(n_tok);
        if (n == 0)
            throw std::runtime_error("Discrete distribution must have at least one symbol");
        // Guard against uint64 → int narrowing: values above INT_MAX are
        // implementation-defined on cast and would corrupt the symbol count.
        constexpr auto kIntMax = static_cast<unsigned long long>(std::numeric_limits<int>::max());
        if (n > kIntMax)
            throw std::runtime_error("Discrete distribution symbol count exceeds INT_MAX");
        auto dist = std::make_unique<DiscreteDistribution>(static_cast<int>(n));
        for (std::size_t k = 0; k < n; ++k) {
            std::string label, eq, val;
            is >> label >> eq >> val;
            if (eq != "=")
                throw std::runtime_error("Malformed Discrete distribution probability entry");
            dist->setProbability(static_cast<double>(k), std::stod(val));
        }
        return dist;
    }

    // Legacy fallback: bare probability list (11 symbols).
    constexpr std::size_t kLegacySymbols = 11;
    std::vector<double> probs(kLegacySymbols);
    probs[0] = std::stod(t);
    for (std::size_t k = 1; k < kLegacySymbols; ++k) {
        is >> t;
        probs[k] = std::stod(t);
    }
    auto dist = std::make_unique<DiscreteDistribution>(static_cast<int>(kLegacySymbols));
    for (std::size_t k = 0; k < kLegacySymbols; ++k)
        dist->setProbability(static_cast<double>(k), probs[k]);
    return dist;
}

// parse_gamma: "k (shape parameter) = V\n\u03b8 (scale parameter) = V\nMean = V\nVariance = V"
std::unique_ptr<EmissionDistribution> parse_gamma(std::istream &is) {
    std::string s, t;
    is >> s;                     // "Distribution:"
    is >> s >> s >> s >> s >> t; // "k" "(shape" "parameter)" "=" VALUE
    const double k = std::stod(t);
    is >> s >> s >> s >> s >> t; // "\u03b8" "(scale" "parameter)" "=" VALUE
    const double theta = std::stod(t);
    is >> s >> s >> t;
    is >> s >> s >> t; // skip Mean, Variance
    return std::make_unique<GammaDistribution>(k, theta);
}

// parse_exponential: "\u03bb (rate parameter) = VALUE\nMean = VALUE"
std::unique_ptr<EmissionDistribution> parse_exponential(std::istream &is) {
    std::string s, t;
    is >> s;                     // "Distribution:"
    is >> s >> s >> s >> s >> t; // "\u03bb" "(rate" "parameter)" "=" VALUE
    const double lambda = std::stod(t);
    is >> s >> s >> t; // skip Mean
    return std::make_unique<ExponentialDistribution>(lambda);
}

// parse_log_normal: "\u03bc (log mean) = V\n\u03c3 (log std. deviation) = V\nMean = V\nVariance = V"
std::unique_ptr<EmissionDistribution> parse_log_normal(std::istream &is) {
    std::string s, t;
    is >> s;                     // "Distribution:"
    is >> s >> s >> s >> s >> t; // "\u03bc" "(log" "mean)" "=" VALUE
    const double mean = std::stod(t);
    is >> s >> s >> s >> s >> s >> t; // "\u03c3" "(log" "std." "deviation)" "=" VALUE
    const double sd = std::stod(t);
    is >> s >> s >> t;
    is >> s >> s >> t; // skip Mean, Variance
    return std::make_unique<LogNormalDistribution>(mean, sd);
}

// parse_pareto: "k (shape parameter) = V\nx_m (scale parameter) = V\nMean = V\nVariance = V"
std::unique_ptr<EmissionDistribution> parse_pareto(std::istream &is) {
    std::string s, t;
    is >> s;                     // "Distribution:"
    is >> s >> s >> s >> s >> t; // "k" "(shape" "parameter)" "=" VALUE
    const double k = std::stod(t);
    is >> s >> s >> s >> s >> t; // "x_m" "(scale" "parameter)" "=" VALUE
    const double xm = std::stod(t);
    is >> s >> s >> t;
    is >> s >> s >> t; // skip Mean, Variance
    return std::make_unique<ParetoDistribution>(k, xm);
}

// parse_poisson: "\u03bb (rate parameter) = V\nMean = V\nVariance = V"
std::unique_ptr<EmissionDistribution> parse_poisson(std::istream &is) {
    std::string s, t;
    is >> s;                     // "Distribution:"
    is >> s >> s >> s >> s >> t; // "\u03bb" "(rate" "parameter)" "=" VALUE
    const double lambda = std::stod(t);
    is >> s >> s >> t;
    is >> s >> s >> t; // skip Mean, Variance
    return std::make_unique<PoissonDistribution>(lambda);
}

// parse_beta: "\u03b1 (alpha) = V\n\u03b2 (beta) = V\nMean = V\nVariance = V"
std::unique_ptr<EmissionDistribution> parse_beta(std::istream &is) {
    std::string s, t;
    is >> s >> s >> s >> s >> t; // "Distribution:" "\u03b1" "(alpha)" "=" VALUE
    const double alpha = std::stod(t);
    is >> s >> s >> s >> t; // "\u03b2" "(beta)" "=" VALUE
    const double beta = std::stod(t);
    is >> s >> s >> t;
    is >> s >> s >> t; // skip Mean, Variance
    return std::make_unique<BetaDistribution>(alpha, beta);
}

std::unique_ptr<EmissionDistribution> parse_weibull(std::istream &is) {
    std::string s, t;
    is >> s >> s >> s >> s >> t; // "Distribution:" "k" "(shape)" "=" value
    const double k = std::stod(t);
    is >> s >> s >> s >> t; // "\u03bb" "(scale)" "=" value
    return std::make_unique<WeibullDistribution>(k, std::stod(t));
}

// parse_uniform: "a (lower bound) = V\nb (upper bound) = V"
std::unique_ptr<EmissionDistribution> parse_uniform(std::istream &is) {
    std::string s, t;
    is >> s;                     // "Distribution:"
    is >> s >> s >> s >> s >> t; // "a" "(lower" "bound)" "=" VALUE
    const double a = std::stod(t);
    is >> s >> s >> s >> s >> t; // "b" "(upper" "bound)" "=" VALUE
    return std::make_unique<UniformDistribution>(a, std::stod(t));
}

std::unique_ptr<EmissionDistribution> parse_student_t(std::istream &is) {
    std::string s, t;
    is >> s;                          // "Distribution:"
    is >> s >> s >> s >> s >> s >> t; // "nu" "(degrees" "of" "freedom)" "=" value
    const double nu = std::stod(t);
    is >> s >> s >> s >> t; // "mu" "(location)" "=" value
    const double mu = std::stod(t);
    is >> s >> s >> s >> t; // "sigma" "(scale)" "=" value
    return std::make_unique<StudentTDistribution>(nu, mu, std::stod(t));
}

std::unique_ptr<EmissionDistribution> parse_chi_squared(std::istream &is) {
    std::string s, t;
    is >> s;                          // "Distribution:"
    is >> s >> s >> s >> s >> s >> t; // "k" "(degrees" "of" "freedom)" "=" value
    return std::make_unique<ChiSquaredDistribution>(std::stod(t));
}

// parse_binomial: "n (trials) = V\np (success probability) = V\nMean = V\nVariance = V"
std::unique_ptr<EmissionDistribution> parse_binomial(std::istream &is) {
    std::string s, t;
    is >> s;                // "Distribution:"
    is >> s >> s >> s >> t; // "n" "(trials)" "=" VALUE
    const int n = static_cast<int>(std::stod(t));
    is >> s >> s >> s >> s >> t; // "p" "(success" "probability)" "=" VALUE
    const double p = std::stod(t);
    is >> s >> s >> t;
    is >> s >> s >> t; // skip Mean, Variance
    return std::make_unique<BinomialDistribution>(n, p);
}

// parse_rayleigh: "\u03c3 (scale parameter) = V\nMean = V\nVariance = V\nMedian = V\nMode = V"
std::unique_ptr<EmissionDistribution> parse_rayleigh(std::istream &is) {
    std::string s, t;
    is >> s;                     // "Distribution:"
    is >> s >> s >> s >> s >> t; // "\u03c3" "(scale" "parameter)" "=" VALUE
    const double sigma = std::stod(t);
    // skip Mean, Variance, Median, Mode
    is >> s >> s >> t;
    is >> s >> s >> t;
    is >> s >> s >> t;
    is >> s >> s >> t;
    return std::make_unique<RayleighDistribution>(sigma);
}

// NegativeBinomial body reader (called after the type keyword has been consumed).
std::unique_ptr<EmissionDistribution> parse_negative_binomial(std::istream &is) {
    std::string s, t;
    is >> s;                // "Distribution:"
    is >> s >> s >> s >> t; // "r" "(successes)" "=" VALUE
    const double r = std::stod(t);
    is >> s >> s >> s >> s >> t; // "p" "(success" "probability)" "=" VALUE
    const double p = std::stod(t);
    is >> s >> s >> t; // skip Mean
    is >> s >> s >> t; // skip Variance
    return std::make_unique<NegativeBinomialDistribution>(r, p);
}

// Legacy fallback: old toString() emitted "Negative Binomial Distribution:",
// so the dispatch key was "Negative". Consume the second word first.
std::unique_ptr<EmissionDistribution> parse_negative_binomial_legacy(std::istream &is) {
    std::string s;
    is >> s; // consume "Binomial" (second word of the type name)
    return parse_negative_binomial(is);
}

/// Returns the stream-parser dispatch map.
/// Function-local static avoids initialization-order issues.
const std::unordered_map<std::string, StreamParserFn> &stream_parsers() {
    static const std::unordered_map<std::string, StreamParserFn> s = {
        {"Gaussian", parse_gaussian},
        {"Discrete", parse_discrete},
        {"Gamma", parse_gamma},
        {"Exponential", parse_exponential},
        {"LogNormal", parse_log_normal},
        {"Pareto", parse_pareto},
        {"Poisson", parse_poisson},
        {"Beta", parse_beta},
        {"Weibull", parse_weibull},
        {"Uniform", parse_uniform},
        {"Binomial", parse_binomial},
        {"Rayleigh", parse_rayleigh},
        {"StudentT", parse_student_t},
        {"ChiSquared", parse_chi_squared},
        // NegativeBinomial: single-token key (current toString() output)
        {"NegativeBinomial", parse_negative_binomial},
        // Legacy: old toString() emitted "Negative Binomial" (two words); the
        // HMM operator>> reads one token as the type key, so old XML files
        // arrive here as "Negative" and need to consume "Binomial" themselves.
        {"Negative", parse_negative_binomial_legacy},
    };
    return s;
}

} // anonymous namespace

std::istream &operator>>(std::istream &is, libhmm::Hmm &hmm) {
    std::string s, t;
    std::size_t states = 0;

    is >> s >> s >> s >> s; // "Hidden Markov Model parameters"
    is >> s >> s;           // "States:"
    states = std::stoull(s);
    if (states == 0)
        throw std::runtime_error("Invalid number of states in HMM input");

    hmm = Hmm(states);
    Vector pi(states);
    Matrix trans(states, states);

    is >> s >> s; // "Pi:" "["
    for (std::size_t i = 0; i < states; ++i) {
        is >> t;
        pi(i) = std::stod(t);
    }
    is >> s; // "]"

    is >> s >> s; // "Transmission" "matrix:"
    for (std::size_t i = 0; i < states; ++i) {
        is >> s; // "["
        for (std::size_t j = 0; j < states; ++j) {
            is >> t;
            trans(i, j) = std::stod(t);
        }
        is >> s; // "]"
    }

    is >> s; // "Emissions:"
    for (std::size_t i = 0; i < states; ++i) {
        is >> s >> s >> t; // "State" "N:" type-keyword
        const auto it = stream_parsers().find(t);
        if (it == stream_parsers().end())
            throw std::runtime_error("Unknown distribution type in stream: " + t);
        hmm.setDistribution(i, it->second(is));
    }

    hmm.setPi(pi);
    hmm.setTrans(trans);
    return is;
} // operator>>()

} // namespace libhmm
