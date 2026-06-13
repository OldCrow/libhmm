#include "libhmm/linalg/cholesky.h"

#include <cassert>
#include <cmath>
#include <limits>

namespace libhmm {
namespace chol {

// =============================================================================
// factorize — Cholesky-Banachiewicz (column-by-column)
//
// For each pivot column j:
//   L[j,j] = sqrt(A[j,j] - sum_{k<j} L[j,k]^2)
//   L[i,j] = (A[i,j] - sum_{k<j} L[i,k]*L[j,k]) / L[j,j]   for i > j
//
// The upper triangle of L is left at 0 (zero-initialised).
// =============================================================================

CholeskyResult factorize(const BasicMatrix<double> &A) noexcept {
    const std::size_t n = A.rows();
    CholeskyResult result;
    result.L = BasicMatrix<double>(n, n, 0.0);

    for (std::size_t j = 0; j < n; ++j) {
        // Diagonal entry
        double s = A(j, j);
        for (std::size_t k = 0; k < j; ++k) {
            s -= result.L(j, k) * result.L(j, k);
        }
        if (s <= 0.0) {
            result.success = false;
            return result;
        }
        result.L(j, j) = std::sqrt(s);

        // Sub-diagonal entries in column j
        const double inv_ljj = 1.0 / result.L(j, j);
        for (std::size_t i = j + 1; i < n; ++i) {
            double t = A(i, j);
            for (std::size_t k = 0; k < j; ++k) {
                t -= result.L(i, k) * result.L(j, k);
            }
            result.L(i, j) = t * inv_ljj;
        }
    }

    result.success = true;
    return result;
}

// =============================================================================
// log_det — from Cholesky factor diagonal
//
// log(det(A)) = log(det(L)^2) = 2 * sum_i log(L[i,i])
// =============================================================================

double log_det(const BasicMatrix<double> &L) noexcept {
    double logd = 0.0;
    for (std::size_t i = 0; i < L.rows(); ++i) {
        const double lii = L(i, i);
        if (lii <= 0.0) {
            return -std::numeric_limits<double>::infinity();
        }
        logd += std::log(lii);
    }
    return 2.0 * logd;
}

// =============================================================================
// solve_lower — forward substitution for L * x = b
//
// x[i] = (b[i] - sum_{j<i} L[i,j] * x[j]) / L[i,i]
// =============================================================================

BasicVector<double> solve_lower(const BasicMatrix<double> &L, std::span<const double> b) noexcept {
    const std::size_t n = b.size();
    assert(L.rows() == n && L.cols() == n);

    BasicVector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (std::size_t j = 0; j < i; ++j) {
            s -= L(i, j) * x[j];
        }
        x[i] = s / L(i, i);
    }
    return x;
}

// =============================================================================
// inv_quad_form — Mahalanobis squared distance xᵀ A⁻¹ x
//
// z = L⁻¹ x  (forward substitution)
// result = z · z = ‖z‖²
// =============================================================================

double inv_quad_form(const BasicMatrix<double> &L, std::span<const double> x) noexcept {
    const auto z = solve_lower(L, x);
    double q = 0.0;
    for (std::size_t i = 0; i < z.size(); ++i) {
        q += z[i] * z[i];
    }
    return q;
}

// =============================================================================
// inv_quad_form_mv — (x−μ)ᵀ Σ⁻¹ (x−μ) with thread_local scratch
//
// Residual and forward substitution are performed in-place in a thread_local
// buffer, avoiding any heap allocation in steady state.
// =============================================================================

double inv_quad_form_mv(const BasicMatrix<double> &L, const std::vector<double> &mu,
                        std::span<const double> x) noexcept {
    const std::size_t n = mu.size();
    // One allocation per thread; buffer only grows, never shrinks.
    thread_local std::vector<double> v;
    if (v.size() < n)
        v.resize(n);

    // Compute residual x − μ, then solve L·v = residual in-place.
    for (std::size_t i = 0; i < n; ++i)
        v[i] = x[i] - mu[i];
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j)
            v[i] -= L(i, j) * v[j];
        v[i] /= L(i, i);
    }
    double q = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        q += v[i] * v[i];
    return q;
}

} // namespace chol
} // namespace libhmm
