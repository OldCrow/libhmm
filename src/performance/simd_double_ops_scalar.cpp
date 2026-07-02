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

} // namespace libhmm::performance::detail
