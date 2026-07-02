// simd_double_ops_scalar.cpp — pure C++ scalar fallback kernels.
// Compiled without any SIMD flags; selected by the dispatch table on CPUs
// without SSE2/NEON or as the scalar tail in other ISA TUs.
// Future tier-1 uplift: add one function here and mirror it in the ISA TUs.

#include <cmath>
#include <cstddef>
#include <limits>

namespace libhmm::performance::detail {

void gaussian_batch_scalar(const double *obs, double *out, std::size_t n, double mean,
                           double neg_half_inv_sq, double log_norm) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        if (std::isnan(x) || std::isinf(x)) {
            out[i] = neg_inf;
        } else {
            const double d = x - mean;
            out[i] = log_norm + neg_half_inv_sq * d * d;
        }
    }
}

void exponential_batch_scalar(const double *obs, double *out, std::size_t n, double log_lambda,
                              double neg_lambda) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        out[i] = (x < 0.0 || std::isnan(x)) ? neg_inf : log_lambda + neg_lambda * x;
    }
}

// ---- Generic math primitives ----

void log_batch_scalar(const double *in, double *out, std::size_t n) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = in[i];
        // x <= 0 → -Inf per spec; NaN <= 0 is false → std::log(NaN) = NaN per spec
        out[i] = (x <= 0.0) ? neg_inf : std::log(x);
    }
}

void exp_batch_scalar(const double *in, double *out, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = std::exp(in[i]);
}

void cos_batch_scalar(const double *in, double *out, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = std::cos(in[i]);
}

void log1p_batch_scalar(const double *in, double *out, std::size_t n) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = in[i];
        // x < -1 → 1+x < 0 → -Inf; x == -1 → std::log1p(-1) == -Inf naturally
        out[i] = (x < -1.0) ? neg_inf : std::log1p(x);
    }
}

// ---- Tier-1 distribution kernels ----

void lognormal_batch_scalar(const double *obs, double *out, std::size_t n, double mean_log,
                            double neg_half_inv_sq, double log_norm_const) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        if (!std::isfinite(x) || x <= 0.0) {
            out[i] = neg_inf;
        } else {
            const double lx = std::log(x);
            const double d = lx - mean_log;
            out[i] = -lx - log_norm_const + neg_half_inv_sq * d * d;
        }
    }
}

void gamma_batch_scalar(const double *obs, double *out, std::size_t n, double k_minus_1,
                        double inv_theta, double const_term) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        out[i] = (!std::isfinite(x) || x <= 0.0)
                     ? neg_inf
                     : const_term + k_minus_1 * std::log(x) - x * inv_theta;
    }
}

void chisq_batch_scalar(const double *obs, double *out, std::size_t n, double half_k_minus_1,
                        double const_term) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        out[i] = (!std::isfinite(x) || x <= 0.0)
                     ? neg_inf
                     : const_term + half_k_minus_1 * std::log(x) - x * 0.5;
    }
}

void rayleigh_batch_scalar(const double *obs, double *out, std::size_t n, double inv2sigma_sq,
                           double log_norm) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        out[i] = (!std::isfinite(x) || x <= 0.0) ? neg_inf
                                                 : log_norm + std::log(x) - x * x * inv2sigma_sq;
    }
}

void pareto_batch_scalar(const double *obs, double *out, std::size_t n, double k_plus_1, double xm,
                         double log_norm) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        out[i] = (!std::isfinite(x) || x < xm) ? neg_inf : log_norm - k_plus_1 * std::log(x);
    }
}

void weibull_batch_scalar(const double *obs, double *out, std::size_t n, double k_minus_1, double k,
                          double log_norm, double neg_k_log_lambda) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        if (!std::isfinite(x) || x <= 0.0) {
            out[i] = neg_inf;
        } else {
            const double lx = std::log(x);
            out[i] = log_norm + k_minus_1 * lx - std::exp(k * lx + neg_k_log_lambda);
        }
    }
}

void beta_batch_scalar(const double *obs, double *out, std::size_t n, double alpha_minus_1,
                       double beta_minus_1, double neg_log_beta) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        if (!std::isfinite(x) || x <= 0.0 || x >= 1.0) {
            out[i] = neg_inf;
        } else {
            out[i] = neg_log_beta + alpha_minus_1 * std::log(x) + beta_minus_1 * std::log1p(-x);
        }
    }
}

void student_t_batch_scalar(const double *obs, double *out, std::size_t n, double location,
                            double inv_scale, double half_nu_plus_1, double log_norm,
                            double inv_nu) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        if (!std::isfinite(x)) {
            out[i] = neg_inf;
        } else {
            const double z = (x - location) * inv_scale;
            out[i] = log_norm - half_nu_plus_1 * std::log1p(z * z * inv_nu);
        }
    }
}

void vonmises_batch_scalar(const double *obs, double *out, std::size_t n, double mu, double kappa,
                           double log_normaliser) noexcept {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = obs[i];
        out[i] = (!std::isfinite(x)) ? neg_inf : kappa * std::cos(x - mu) - log_normaliser;
    }
}

} // namespace libhmm::performance::detail
