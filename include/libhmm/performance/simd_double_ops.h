#pragma once

#include <cstddef>

namespace libhmm::performance {

/// @brief Runtime-dispatched vectorized kernel table for double-precision
/// distribution batch evaluation.
///
/// The table is built exactly once at program startup by get_double_vec_ops()
/// (function-local static, thread-safe per C++11) using CPUID to select the
/// highest ISA tier available on the runtime CPU. After the first call, every
/// subsequent call is a trivial reference return with no locking or branching.
///
/// ## Adding a new primitive for tier-1 uplift
/// 1. Declare the function-pointer member here (uncommenting the template line).
/// 2. Implement the function in each of the five simd_double_ops_*.cpp TUs.
/// 3. Register the pointers in build_table() in simd_dispatch.cpp.
/// Zero new TUs required; the five existing ISA files each grow by one function.
struct DoubleVecOps {
    // -------------------------------------------------------------------------
    // Tier-2 named kernels (initial population)
    //
    // gaussian_batch: log_norm + neg_half_inv_sq * (x - mean)^2
    //   x=NaN or x=Inf → -Inf. x=finite → normal Gaussian log-PDF.
    // -------------------------------------------------------------------------
    void (*gaussian_batch)(const double *obs, double *out, std::size_t n, double mean,
                           double neg_half_inv_sq, double log_norm) noexcept;

    // exponential_batch: log_lambda + neg_lambda * x, for x >= 0
    //   x < 0 or x=NaN → -Inf. x=+Inf → -Inf naturally from the formula.
    void (*exponential_batch)(const double *obs, double *out, std::size_t n, double log_lambda,
                              double neg_lambda) noexcept;

    // -------------------------------------------------------------------------
    // Generic math primitives
    // -------------------------------------------------------------------------
    /// log(x): x ≤ 0 → −∞, NaN → NaN. SLEEF xlog_u1 core, < 1 ULP.
    void (*log_batch)(const double *in, double *out, std::size_t n) noexcept;
    /// exp(x): clamped to [−708, 709.8]. SLEEF-inspired, < 1 ULP.
    void (*exp_batch)(const double *in, double *out, std::size_t n) noexcept;
    /// cos(x): all finite x, 7-term Horner, |error| < 2×10⁻¹⁰.
    void (*cos_batch)(const double *in, double *out, std::size_t n) noexcept;
    /// log(1+x): accurate for |x| ≪ 1. Implemented as SIMD(1+x) then log core.
    void (*log1p_batch)(const double *in, double *out, std::size_t n) noexcept;

    // -------------------------------------------------------------------------
    // Tier-1 distribution kernels (single-pass, inline primitives, no temp buf)
    // -------------------------------------------------------------------------
    /// LogNormal: −logNormConst + neg_half_inv_sq*(log(x)−mean_log)²; x≤0 → −∞.
    void (*lognormal_batch)(const double *obs, double *out, std::size_t n, double mean_log,
                            double neg_half_inv_sq, double log_norm_const) noexcept;
    /// Gamma: const_term + (k−1)*log(x) − x/θ; x≤0 → −∞.
    void (*gamma_batch)(const double *obs, double *out, std::size_t n, double k_minus_1,
                        double inv_theta, double const_term) noexcept;
    /// Chi-squared: const_term + (k/2−1)*log(x) − x/2; x≤0 → −∞.
    void (*chisq_batch)(const double *obs, double *out, std::size_t n, double half_k_minus_1,
                        double const_term) noexcept;
    /// Rayleigh: log_norm + log(x) − x²*inv2sigma_sq; x≤0 → −∞.
    void (*rayleigh_batch)(const double *obs, double *out, std::size_t n, double inv2sigma_sq,
                           double log_norm) noexcept;
    /// Pareto: log_norm − (k+1)*log(x); x<xm → −∞.
    void (*pareto_batch)(const double *obs, double *out, std::size_t n, double k_plus_1, double xm,
                         double log_norm) noexcept;
    /// Weibull: log_norm + (k−1)*log(x) − exp(k*log(x)+neg_k_log_lambda); x≤0 → −∞.
    void (*weibull_batch)(const double *obs, double *out, std::size_t n, double k_minus_1, double k,
                          double log_norm, double neg_k_log_lambda) noexcept;
    /// Beta: neg_log_beta + (α−1)*log(x) + (β−1)*log1p(−x); x∉(0,1) → −∞.
    void (*beta_batch)(const double *obs, double *out, std::size_t n, double alpha_minus_1,
                       double beta_minus_1, double neg_log_beta) noexcept;
    /// StudentT: log_norm − half_nu_plus_1*log1p(((x−loc)*inv_scale)²*inv_nu).
    void (*student_t_batch)(const double *obs, double *out, std::size_t n, double location,
                            double inv_scale, double half_nu_plus_1, double log_norm,
                            double inv_nu) noexcept;
    /// VonMises: κ*cos(x−μ) − log_normaliser; NaN/Inf → −∞.
    void (*vonmises_batch)(const double *obs, double *out, std::size_t n, double mu, double kappa,
                           double log_normaliser) noexcept;
};

/// @brief Returns the runtime-selected dispatch table.
///
/// The first call builds the table (CPUID + function-pointer selection) and
/// stores the result in a function-local static. All subsequent calls return
/// the cached reference without any synchronization overhead.
const DoubleVecOps &get_double_vec_ops() noexcept;

} // namespace libhmm::performance
