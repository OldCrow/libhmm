#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

namespace libhmm {
namespace performance {
namespace detail {

/**
 * @brief Internal backend tag for explicit transcendental-vector kernels.
 *
 * Current implementation is scalar-only. The enum and helper boundaries exist
 * so AVX2 / NEON implementations can replace these scalar loops without
 * another structural rewrite of FB max-reduce and BW dense-xi call sites.
 */
enum class TranscendentalBackend {
    Scalar,
    Avx2,
    Neon,
};

[[nodiscard]] constexpr TranscendentalBackend currentTranscendentalBackend() noexcept {
#if defined(LIBHMM_HAS_AVX2)
    return TranscendentalBackend::Avx2;
#elif defined(LIBHMM_HAS_NEON)
    return TranscendentalBackend::Neon;
#else
    return TranscendentalBackend::Scalar;
#endif
}

[[nodiscard]] constexpr std::size_t currentTranscendentalLaneCount() noexcept {
    switch (currentTranscendentalBackend()) {
    case TranscendentalBackend::Avx2:
        return 4;
    case TranscendentalBackend::Neon:
        return 2;
    case TranscendentalBackend::Scalar:
        return 1;
    }
    return 1;
}

[[nodiscard]] constexpr const char *toString(TranscendentalBackend backend) noexcept {
    switch (backend) {
    case TranscendentalBackend::Scalar:
        return "scalar";
    case TranscendentalBackend::Avx2:
        return "avx2";
    case TranscendentalBackend::Neon:
        return "neon";
    }
    return "unknown";
}

class TranscendentalKernels {
public:
    [[nodiscard]] static inline double reduce_max_sum2(const double *a, const double *b,
                                                       std::size_t size) noexcept {
        return reduce_max_sum2_scalar(a, b, size);
    }

    [[nodiscard]] static inline double sum_exp_sum2_minus_max(const double *a, const double *b,
                                                              std::size_t size,
                                                              double maxVal) noexcept {
        return sum_exp_sum2_minus_max_scalar(a, b, size, maxVal);
    }

    [[nodiscard]] static inline double reduce_max_sum3(const double *a, const double *b,
                                                       const double *c,
                                                       std::size_t size) noexcept {
        return reduce_max_sum3_scalar(a, b, c, size);
    }

    [[nodiscard]] static inline double sum_exp_sum3_minus_max(const double *a, const double *b,
                                                              const double *c,
                                                              std::size_t size,
                                                              double maxVal) noexcept {
        return sum_exp_sum3_minus_max_scalar(a, b, c, size, maxVal);
    }

    static inline void accumulate_exp_sum2_bias(double *dst, const double *a, const double *b,
                                                std::size_t size, double bias) noexcept {
        accumulate_exp_sum2_bias_scalar(dst, a, b, size, bias);
    }

private:
    [[nodiscard]] static inline double reduce_max_sum2_scalar(const double *a, const double *b,
                                                              std::size_t size) noexcept {
        double maxVal = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < size; ++i) {
            const double term = a[i] + b[i];
            if (term > maxVal) {
                maxVal = term;
            }
        }
        return maxVal;
    }

    [[nodiscard]] static inline double
    sum_exp_sum2_minus_max_scalar(const double *a, const double *b, std::size_t size,
                                  double maxVal) noexcept {
        if (!std::isfinite(maxVal)) {
            return 0.0;
        }
        double sum = 0.0;
        for (std::size_t i = 0; i < size; ++i) {
            const double term = a[i] + b[i];
            if (std::isfinite(term)) {
                sum += std::exp(term - maxVal);
            }
        }
        return sum;
    }

    [[nodiscard]] static inline double reduce_max_sum3_scalar(const double *a, const double *b,
                                                              const double *c,
                                                              std::size_t size) noexcept {
        double maxVal = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < size; ++i) {
            const double term = a[i] + b[i] + c[i];
            if (term > maxVal) {
                maxVal = term;
            }
        }
        return maxVal;
    }

    [[nodiscard]] static inline double
    sum_exp_sum3_minus_max_scalar(const double *a, const double *b, const double *c,
                                  std::size_t size, double maxVal) noexcept {
        if (!std::isfinite(maxVal)) {
            return 0.0;
        }
        double sum = 0.0;
        for (std::size_t i = 0; i < size; ++i) {
            const double term = a[i] + b[i] + c[i];
            if (std::isfinite(term)) {
                sum += std::exp(term - maxVal);
            }
        }
        return sum;
    }

    static inline void accumulate_exp_sum2_bias_scalar(double *dst, const double *a,
                                                       const double *b, std::size_t size,
                                                       double bias) noexcept {
        for (std::size_t i = 0; i < size; ++i) {
            dst[i] += std::exp(a[i] + b[i] + bias);
        }
    }
};

} // namespace detail
} // namespace performance
} // namespace libhmm
