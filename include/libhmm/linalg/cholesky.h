#pragma once

/**
 * @file cholesky.h
 * @brief Cholesky factorization and derived linear algebra helpers.
 *
 * Provides the numerical primitives required by FullCovarianceGaussianDistribution:
 *
 *   factorize(A)          — compute L such that A = L·Lᵀ (Cholesky-Banachiewicz)
 *   log_det(L)            — log(det(A)) = 2·Σᵢ log(Lᵢᵢ)
 *   solve_lower(L, b)     — forward substitution: solve L·x = b
 *   inv_quad_form(L, x)   — xᵀ·A⁻¹·x = ‖L⁻¹·x‖²
 *
 * Uses BasicMatrix<double> and BasicVector<double> only — no external deps.
 * All functions are noexcept; errors are communicated through CholeskyResult::success.
 */

#include <cstddef>
#include <span>

#include "libhmm/linalg/linalg_types.h"

namespace libhmm {
namespace chol {

// =============================================================================
// Result type
// =============================================================================

/**
 * @brief Result of a Cholesky factorization.
 *
 * L is meaningful only when success == true.  On failure (matrix not
 * symmetric positive-definite) L is left in a partial or zeroed state.
 */
struct CholeskyResult {
    BasicMatrix<double> L{};    ///< Lower triangular factor such that A = L·Lᵀ
    bool success{false};        ///< false when A is not positive-definite
};

// =============================================================================
// Core operations
// =============================================================================

/**
 * @brief Cholesky-Banachiewicz factorization: A = L·Lᵀ.
 *
 * A must be symmetric positive-definite (SPD).  The algorithm proceeds
 * column by column and fails early if any diagonal entry of L would be
 * non-positive (indicating A is not SPD).
 *
 * Complexity: O(D³/3).
 *
 * @param A  Input matrix (D×D, must be SPD).
 * @return   CholeskyResult with success=true and the lower triangular L on
 *           success; success=false otherwise.
 */
[[nodiscard]] CholeskyResult factorize(const BasicMatrix<double>& A) noexcept;

/**
 * @brief Log-determinant of A from its Cholesky factor L.
 *
 * log(det(A)) = log(det(L)²) = 2·Σᵢ log(Lᵢᵢ)
 *
 * @param L  Lower triangular factor from a successful factorize() call.
 * @return   log(det(A)).  Result is -∞ if any diagonal entry is ≤ 0.
 */
[[nodiscard]] double log_det(const BasicMatrix<double>& L) noexcept;

/**
 * @brief Forward substitution: solve L·x = b for x.
 *
 * L must be lower triangular with non-zero diagonal entries.
 * Precondition: L.rows() == L.cols() == b.size().
 *
 * @param L  Lower triangular matrix (D×D).
 * @param b  Right-hand side (D).  Accepts std::span<const double> directly,
 *           which includes ObservationVectorView and std::vector<double>.
 * @return   Solution vector x (D).
 */
[[nodiscard]] BasicVector<double> solve_lower(const BasicMatrix<double>& L,
                                              std::span<const double> b) noexcept;

/**
 * @brief Inverse quadratic form: xᵀ·A⁻¹·x.
 *
 * Computed as ‖z‖² where z = L⁻¹·x (forward substitution).
 * This is the Mahalanobis squared distance used in the multivariate
 * Gaussian log-probability.
 *
 * Precondition: L.rows() == L.cols() == x.size().
 *
 * @param L  Cholesky factor of A (lower triangular, D×D).
 * @param x  Query vector (D).  Accepts ObservationVectorView directly.
 * @return   xᵀ·A⁻¹·x ≥ 0.
 */
[[nodiscard]] double inv_quad_form(const BasicMatrix<double>& L,
                                   std::span<const double> x) noexcept;

} // namespace chol
} // namespace libhmm
