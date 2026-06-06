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
    if (x < math::ZERO_DOUBLE || a <= math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }

    if (x < (a + math::ONE)) {
        double gamser = 0.0, gln = 0.0;
        gser(gamser, a, x, gln);
        return gamser;
    } else {
        double gammcf = 0.0, gln = 0.0;
        gcf(gammcf, a, x, gln);
        return math::ONE - gammcf;
    }
}

void DistributionMathHelper::gcf(double& gammcf, double a, double x, double& gln) noexcept {
    using namespace libhmm::constants;

    gln = std::lgamma(a);
    double b = x + math::ONE - a;
    double c = math::ONE / precision::ZERO;
    double d = math::ONE / b;
    double h = d;

    for (std::size_t i = 1; i <= iterations::ITMAX; ++i) {
        const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += math::TWO;
        d = an * d + b;
        if (std::abs(d) < precision::ZERO)
            d = precision::ZERO;
        c = b + an / c;
        if (std::abs(c) < precision::ZERO)
            c = precision::ZERO;
        d = math::ONE / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - math::ONE) < precision::BW_TOLERANCE)
            break;
    }

    gammcf = std::exp(-x + a * std::log(x) - gln) * h;
}

void DistributionMathHelper::gser(double& gamser, double a, double x, double& gln) noexcept {
    using namespace libhmm::constants;

    gln = std::lgamma(a);

    if (x <= math::ZERO_DOUBLE) {
        gamser = math::ZERO_DOUBLE;
        return;
    }

    double ap = a;
    double sum = math::ONE / a;
    double del = sum;

    for (std::size_t n = 1; n <= iterations::ITMAX; ++n) {
        ++ap;
        del *= x / ap;
        sum += del;
        if (std::abs(del) < std::abs(sum) * precision::BW_TOLERANCE) {
            gamser = sum * std::exp(-x + a * std::log(x) - gln);
            return;
        }
    }
    // Convergence not reached — return best estimate
    gamser = sum * std::exp(-x + a * std::log(x) - gln);
}

double DistributionMathHelper::errorf_inv(double y) noexcept {
    using namespace libhmm::constants;

    if (y == math::ZERO_DOUBLE)
        return math::ZERO_DOUBLE;
    if (y > math::ONE)
        return std::numeric_limits<double>::infinity();
    if (y < -math::ONE)
        return -std::numeric_limits<double>::infinity();

    const double k = y;
    if (y < 0)
        y = -y;

    const double z = math::ONE - y;
    const double w = 0.916461398268964 - std::log(z);
    const double u = std::sqrt(w);
    const double s = (std::log(u) + 0.488826640273108) / w;
    double t = math::ONE / (u + 0.231729200323405);
    double x = u * (math::ONE - s * (s * 0.124610454613712 + math::HALF)) -
               ((((-0.0728846765585675 * t + 0.269999308670029) * t + 0.150689047360223) * t +
                 0.116065025341614) *
                    t +
                0.499999303439796) *
                   t;

    t = 3.97886080735226 / (x + 3.97886080735226);
    const double u2 = t - math::HALF;
    double sv = (((((((((0.00112648096188977922 * u2 + 1.05739299623423047e-4) * u2 -
                        0.00351287146129100025) *
                           u2 -
                       7.71708358954120939e-4) *
                          u2 +
                      0.00685649426074558612) *
                         u2 +
                     0.00339721910367775861) *
                        u2 -
                    0.011274916933250487) *
                       u2 -
                   0.0118598117047771104) *
                      u2 +
                  0.0142961988697898018) *
                     u2 +
                 0.0346494207789099922) *
                    u2 +
                0.00220995927012179067;
    sv = ((((((((((((sv * u2 - 0.0743424357241784861) * u2 - 0.105872177941595488) * u2 +
                   0.0147297938331485121) *
                      u2 +
                  0.316847638520135944) *
                     u2 +
                 0.713657635868730364) *
                    u2 +
                1.05375024970847138) *
                   u2 +
               1.21448730779995237) *
                  u2 +
              1.16374581931560831) *
                 u2 +
             0.956464974744799006) *
                u2 +
            0.686265948274097816) *
               u2 +
           0.434397492331430115) *
              u2 +
          0.244044510593190935) *
             t -
         z * std::exp(x * x - 0.120782237635245222);

    x += sv * (x * sv + math::ONE);
    return k < 0 ? -x : x;
}

} // namespace libhmm
