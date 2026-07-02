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
    // Generic math primitives — add here as tier-1 distributions are uplifted.
    // Each new entry requires one function per existing ISA file; no new TUs.
    // -------------------------------------------------------------------------
    // void (*log_batch)(const double *in, double *out, std::size_t n) noexcept;
    // void (*exp_batch)(const double *in, double *out, std::size_t n) noexcept;
};

/// @brief Returns the runtime-selected dispatch table.
///
/// The first call builds the table (CPUID + function-pointer selection) and
/// stores the result in a function-local static. All subsequent calls return
/// the cached reference without any synchronization overhead.
const DoubleVecOps &get_double_vec_ops() noexcept;

} // namespace libhmm::performance
