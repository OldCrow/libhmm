/**
 * @file test_cholesky.cpp
 * @brief Tests for libhmm::chol — Cholesky factorization and derived helpers.
 *
 * All reference values are derived analytically from first principles so that
 * any numerical regression in the implementation is immediately visible.
 *
 * Reference matrices:
 *
 *   2D:  A = [[4, 2], [2, 3]]   det = 8
 *        L = [[2, 0], [1, √2]]
 *
 *   3D:  A = [[9, 3, 1], [3, 5, 2], [1, 2, 4]]   det = 115
 *        L = [[3, 0, 0], [1, 2, 0], [1/3, 5/6, √115/6]]
 *
 *   Non-SPD: A = [[1, 2], [2, 1]]   det = -3  (must fail)
 */

#include <gtest/gtest.h>
#include "libhmm/linalg/cholesky.h"
#include <cmath>
#include <vector>

using namespace libhmm;
using namespace libhmm::chol;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static BasicMatrix<double> make2d()
{
    BasicMatrix<double> A(2, 2);
    A(0, 0) = 4.0;  A(0, 1) = 2.0;
    A(1, 0) = 2.0;  A(1, 1) = 3.0;
    return A;
}

static BasicMatrix<double> make3d()
{
    BasicMatrix<double> A(3, 3);
    A(0, 0) = 9.0;  A(0, 1) = 3.0;  A(0, 2) = 1.0;
    A(1, 0) = 3.0;  A(1, 1) = 5.0;  A(1, 2) = 2.0;
    A(2, 0) = 1.0;  A(2, 1) = 2.0;  A(2, 2) = 4.0;
    return A;
}

// ---------------------------------------------------------------------------
// factorize
// ---------------------------------------------------------------------------

TEST(CholeskyTest, Factorize2DKnownFactor)
{
    // A = [[4,2],[2,3]], L = [[2,0],[1,√2]]
    const auto [L, ok] = factorize(make2d());
    ASSERT_TRUE(ok);
    EXPECT_NEAR(L(0, 0), 2.0,             1e-12);
    EXPECT_NEAR(L(0, 1), 0.0,             1e-12);  // upper triangle = 0
    EXPECT_NEAR(L(1, 0), 1.0,             1e-12);
    EXPECT_NEAR(L(1, 1), std::sqrt(2.0),  1e-12);
}

TEST(CholeskyTest, Factorize3DRoundtrip)
{
    // Verify L * Lᵀ recovers A for the 3D case.
    const auto A    = make3d();
    const auto [L, ok] = factorize(A);
    ASSERT_TRUE(ok);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            // (L * Lᵀ)[i,j] = sum_k L[i,k] * L[j,k]
            double val = 0.0;
            for (std::size_t k = 0; k <= std::min(i, j); ++k) {
                val += L(i, k) * L(j, k);
            }
            EXPECT_NEAR(val, A(i, j), 1e-10)
                << "mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST(CholeskyTest, Factorize1D)
{
    // Trivial 1×1 case: L = [[√a]]
    BasicMatrix<double> A(1, 1);
    A(0, 0) = 9.0;
    const auto [L, ok] = factorize(A);
    ASSERT_TRUE(ok);
    EXPECT_NEAR(L(0, 0), 3.0, 1e-12);
}

TEST(CholeskyTest, NonSPDFails)
{
    // A = [[1,2],[2,1]] is not SPD (det = -3).
    // The factorization must fail at the second diagonal entry:
    //   L[0,0] = 1, L[1,0] = 2, s = 1 - 4 = -3 → failure.
    BasicMatrix<double> A(2, 2);
    A(0, 0) = 1.0;  A(0, 1) = 2.0;
    A(1, 0) = 2.0;  A(1, 1) = 1.0;
    const auto [L, ok] = factorize(A);
    EXPECT_FALSE(ok);
}

TEST(CholeskyTest, NegativeDiagonalFails)
{
    BasicMatrix<double> A(2, 2);
    A(0, 0) = -1.0;  A(0, 1) = 0.0;
    A(1, 0) = 0.0;   A(1, 1) = 1.0;
    EXPECT_FALSE(factorize(A).success);
}

// ---------------------------------------------------------------------------
// log_det
// ---------------------------------------------------------------------------

TEST(CholeskyTest, LogDet2D)
{
    // log(det([[4,2],[2,3]])) = log(8)
    const auto [L, ok] = factorize(make2d());
    ASSERT_TRUE(ok);
    EXPECT_NEAR(log_det(L), std::log(8.0), 1e-12);
}

TEST(CholeskyTest, LogDet3D)
{
    // log(det([[9,3,1],[3,5,2],[1,2,4]])) = log(115)
    const auto [L, ok] = factorize(make3d());
    ASSERT_TRUE(ok);
    EXPECT_NEAR(log_det(L), std::log(115.0), 1e-10);
}

TEST(CholeskyTest, LogDet1D)
{
    BasicMatrix<double> A(1, 1);
    A(0, 0) = 4.0;
    const auto [L, ok] = factorize(A);
    ASSERT_TRUE(ok);
    EXPECT_NEAR(log_det(L), std::log(4.0), 1e-12);
}

// ---------------------------------------------------------------------------
// solve_lower
// ---------------------------------------------------------------------------

TEST(CholeskyTest, SolveLower2D)
{
    // L = [[2,0],[1,√2]], b = [2, 1+√2]
    // Forward sub: x[0] = 2/2 = 1, x[1] = (1+√2 - 1)/√2 = 1
    const auto [L, ok] = factorize(make2d());
    ASSERT_TRUE(ok);

    const double s2 = std::sqrt(2.0);
    const std::vector<double> b = {2.0, 1.0 + s2};
    const auto x = solve_lower(L, b);

    EXPECT_NEAR(x[0], 1.0, 1e-12);
    EXPECT_NEAR(x[1], 1.0, 1e-12);
}

TEST(CholeskyTest, SolveLower1D)
{
    BasicMatrix<double> A(1, 1);
    A(0, 0) = 5.0;
    const auto [L, ok] = factorize(A);
    ASSERT_TRUE(ok);

    const std::vector<double> b = {10.0};
    const auto x = solve_lower(L, b);
    EXPECT_NEAR(x[0], 10.0 / std::sqrt(5.0), 1e-12);
}

// ---------------------------------------------------------------------------
// inv_quad_form
// ---------------------------------------------------------------------------

TEST(CholeskyTest, InvQuadForm2D)
{
    // A = [[4,2],[2,3]], x = [1,1]
    // A⁻¹ = (1/8) * [[3,-2],[-2,4]]
    // xᵀ A⁻¹ x = (1/8)*(3-2-2+4) = 3/8
    const auto [L, ok] = factorize(make2d());
    ASSERT_TRUE(ok);

    const std::vector<double> x = {1.0, 1.0};
    EXPECT_NEAR(inv_quad_form(L, x), 3.0 / 8.0, 1e-12);
}

TEST(CholeskyTest, InvQuadFormIdentity)
{
    // A = I_3 → xᵀ A⁻¹ x = ‖x‖²
    BasicMatrix<double> I(3, 3, 0.0);
    I(0, 0) = 1.0;  I(1, 1) = 1.0;  I(2, 2) = 1.0;
    const auto [L, ok] = factorize(I);
    ASSERT_TRUE(ok);

    const std::vector<double> x = {1.0, 2.0, 3.0};
    EXPECT_NEAR(inv_quad_form(L, x), 1.0 + 4.0 + 9.0, 1e-12);
}

TEST(CholeskyTest, InvQuadFormScalar)
{
    // 1D: xᵀ A⁻¹ x = x² / a
    BasicMatrix<double> A(1, 1);
    A(0, 0) = 4.0;
    const auto [L, ok] = factorize(A);
    ASSERT_TRUE(ok);

    const std::vector<double> x = {6.0};
    EXPECT_NEAR(inv_quad_form(L, x), 36.0 / 4.0, 1e-12);
}

TEST(CholeskyTest, InvQuadFormUsesObservationVectorView)
{
    // Verify the function accepts ObservationVectorView (= span<const double>)
    // directly, as required by Phase G multivariate distributions.
    const auto [L, ok] = factorize(make2d());
    ASSERT_TRUE(ok);

    const std::vector<double> storage = {1.0, 1.0};
    const libhmm::ObservationVectorView view(storage);
    EXPECT_NEAR(inv_quad_form(L, view), 3.0 / 8.0, 1e-12);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
