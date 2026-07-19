#include "libhmm/distributions/distribution_base.h"
#include "libhmm/math/constants.h"
#include <cmath>
#include <limits>

namespace libhmm {

// =============================================================================
// DistributionMathHelper — shared math utilities for all emission distributions.
//
// Rule of Five and getBatchLogProbabilities() are now inline in the
// DistributionBase<Derived, Obs> template header.
// =============================================================================

double DistributionMathHelper::gammap(double a, double x) noexcept {
    using namespace libhmm::constants;
    // Regularized lower incomplete gamma P(a, x) = γ(a, x) / Γ(a).
    //
    // Reference: Abramowitz & Stegun, "Handbook of Mathematical Functions" §6.5;
    // NIST DLMF §8.  Evaluation splits at x = a+1, where each branch converges
    // fastest:
    //   x < a+1 : power series for γ(a, x)                 (DLMF 8.7.1)
    //   x >= a+1: continued fraction for Q(a, x) = 1 − P   (DLMF 8.9.2)
    if (!(x > math::ZERO_DOUBLE) || !(a > math::ZERO_DOUBLE)) {
        // Domain guard (also rejects NaN): a > 0 required and P(a, 0) = 0.
        return math::ZERO_DOUBLE;
    }
    if (x < a + math::ONE) {
        return lowerGammaSeries(a, x);
    }
    return math::ONE - upperGammaContinuedFraction(a, x);
}

double DistributionMathHelper::lowerGammaSeries(double a, double x) noexcept {
    using namespace libhmm::constants;
    // Power series (DLMF 8.7.1):
    //   P(a, x) = e^{-x} x^a / Γ(a) · Σ_{n>=0} x^n / (a)_{n+1}
    // The bracketed sum starts at 1/a; each term multiplies the previous by
    // x/(a+n).  Converges rapidly for x < a+1.
    const double logPrefactor = a * std::log(x) - x - std::lgamma(a);
    double term = math::ONE / a;
    double sum = term;
    double denom = a;
    for (std::size_t n = 1; n <= iterations::ITMAX; ++n) {
        denom += math::ONE;
        term *= x / denom;
        sum += term;
        if (std::abs(term) < std::abs(sum) * precision::BW_TOLERANCE) {
            break;
        }
    }
    return sum * std::exp(logPrefactor);
}

double DistributionMathHelper::upperGammaContinuedFraction(double a, double x) noexcept {
    using namespace libhmm::constants;
    // Continued fraction for Q(a, x) = Γ(a, x) / Γ(a) (DLMF 8.9.2), evaluated
    // with the modified Lentz algorithm (W. J. Lentz, Appl. Opt. 15, 1976).
    // Returns Q; the caller forms P = 1 − Q.  Used for x >= a+1.
    const double logPrefactor = a * std::log(x) - x - std::lgamma(a);
    double bTerm = x + math::ONE - a;            // b_1
    double cLentz = math::ONE / precision::ZERO; // C_1 (large; precision::ZERO is Lentz's tiny)
    double dLentz = math::ONE / bTerm;           // D_1
    double frac = dLentz;                        // running fraction value
    for (std::size_t n = 1; n <= iterations::ITMAX; ++n) {
        const double aTerm = -static_cast<double>(n) * (static_cast<double>(n) - a);
        bTerm += math::TWO;
        dLentz = aTerm * dLentz + bTerm;
        if (std::abs(dLentz) < precision::ZERO)
            dLentz = precision::ZERO;
        cLentz = bTerm + aTerm / cLentz;
        if (std::abs(cLentz) < precision::ZERO)
            cLentz = precision::ZERO;
        dLentz = math::ONE / dLentz;
        const double delta = dLentz * cLentz;
        frac *= delta;
        if (std::abs(delta - math::ONE) < precision::BW_TOLERANCE)
            break;
    }
    return std::exp(logPrefactor) * frac;
}

double DistributionMathHelper::incompleteBeta(double x, double a, double b) noexcept {
    // Regularized incomplete beta function I_x(a, b) = B(x; a, b) / B(a, b).
    //
    // Reference: Abramowitz & Stegun §6.6 / §26.5.8; NIST DLMF §8.17.  The
    // continued fraction (DLMF 8.17.22) is evaluated with the modified Lentz
    // algorithm (Lentz, 1976).  The symmetry relation I_x(a,b) = 1 − I_{1−x}(b,a)
    // (DLMF 8.17.4) keeps the fraction in its fast-converging regime.
    if (x <= 0.0)
        return 0.0;
    if (x >= 1.0)
        return 1.0;
    if (a <= 0.0 || b <= 0.0)
        return 0.0;

    // Evaluate on whichever side of the symmetry relation converges faster.
    const bool flip = (x > (a + 1.0) / (a + b + 2.0));
    const double xx = flip ? 1.0 - x : x;
    const double aa = flip ? b : a;
    const double bb = flip ? a : b;

    // Prefactor x^a (1−x)^b / (a · B(a, b)), formed in log space for stability.
    const double logBeta = std::lgamma(aa) + std::lgamma(bb) - std::lgamma(aa + bb);
    const double logPrefactor =
        aa * std::log(xx) + bb * std::log(1.0 - xx) - std::log(aa) - logBeta;
    const double prefactor = std::exp(logPrefactor);

    constexpr int kMaxIter = 200;
    constexpr double kTolerance = 1.0e-12;
    constexpr double kTiny = 1.0e-30;

    // Modified Lentz evaluation with coefficients from DLMF 8.17.22.
    double cLentz = 1.0;
    double dLentz = 1.0 - (aa + bb) * xx / (aa + 1.0);
    if (std::abs(dLentz) < kTiny)
        dLentz = kTiny;
    dLentz = 1.0 / dLentz;
    double frac = dLentz;

    for (int m = 1; m <= kMaxIter; ++m) {
        const double m2 = 2.0 * m;
        // Even-indexed coefficient d_{2m}.
        double coeff = m * (bb - m) * xx / ((aa + m2 - 1.0) * (aa + m2));
        dLentz = 1.0 + coeff * dLentz;
        if (std::abs(dLentz) < kTiny)
            dLentz = kTiny;
        cLentz = 1.0 + coeff / cLentz;
        if (std::abs(cLentz) < kTiny)
            cLentz = kTiny;
        dLentz = 1.0 / dLentz;
        frac *= dLentz * cLentz;

        // Odd-indexed coefficient d_{2m+1}.
        coeff = -(aa + m) * (aa + bb + m) * xx / ((aa + m2) * (aa + m2 + 1.0));
        dLentz = 1.0 + coeff * dLentz;
        if (std::abs(dLentz) < kTiny)
            dLentz = kTiny;
        cLentz = 1.0 + coeff / cLentz;
        if (std::abs(cLentz) < kTiny)
            cLentz = kTiny;
        dLentz = 1.0 / dLentz;
        const double delta = dLentz * cLentz;
        frac *= delta;
        if (std::abs(delta - 1.0) < kTolerance)
            break;
    }

    const double result = prefactor * frac;
    return flip ? 1.0 - result : result;
}

double DistributionMathHelper::errorf_inv(double y) noexcept {
    using namespace libhmm::constants;
    // Inverse error function erf⁻¹(y) for y ∈ (−1, 1).
    //
    // Initial estimate: Winitzki, S. (2008), "A handy approximation for the
    // error function and its inverse" — a closed form accurate to a few ×10⁻³.
    // It is then refined to full double precision with two Newton iterations on
    // f(x) = erf(x) − y, using f'(x) = (2/√π) e^{−x²} and std::erf (C++11).
    if (std::isnan(y))
        return std::numeric_limits<double>::quiet_NaN();
    if (y >= math::ONE)
        return std::numeric_limits<double>::infinity();
    if (y <= -math::ONE)
        return -std::numeric_limits<double>::infinity();
    if (y == math::ZERO_DOUBLE)
        return math::ZERO_DOUBLE;

    // Winitzki closed-form initial approximation.
    constexpr double kShape = 0.147; // fitting constant a
    const double ln1my2 = std::log(math::ONE - y * y);
    const double t = math::TWO / (math::PI * kShape) + math::HALF * ln1my2;
    double x = std::copysign(std::sqrt(std::sqrt(t * t - ln1my2 / kShape) - t), y);

    // Newton refinement; two steps reach machine precision from a ~10⁻³ start.
    constexpr double kTwoOverSqrtPi = 1.1283791670955126158634; // 2/√π
    for (int i = 0; i < 2; ++i) {
        const double deriv = kTwoOverSqrtPi * std::exp(-x * x);
        if (deriv < precision::ZERO)
            break; // derivative underflow near saturation; keep current estimate
        x -= (std::erf(x) - y) / deriv;
    }
    return x;
}

} // namespace libhmm
