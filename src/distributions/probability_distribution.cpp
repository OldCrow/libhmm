#include "libhmm/distributions/probability_distribution.h"
#include <limits>

using namespace libhmm::constants;

namespace libhmm
{

double ProbabilityDistribution::loggamma(double xx) const noexcept {
    double x, y, tmp, ser;
    static constexpr double cof[6] = { 76.18009172947146,
                                      -86.50532032941677,
                                       24.01409824083091,
                                      -1.231739572450155,
                                   0.1208650973866179e-2,
                                     -0.5395239384953e-5 };
   
    y = x = xx;
    tmp = x + 5.5;
    tmp -= (x + math::HALF)*std::log(tmp);
    ser = 1.000000000190015;
    for(int j = 0; j <= 5; j++) {
        ser += cof[j]/++y;
    }
    return -tmp + std::log(2.5066282746310005*ser/x);
}
   

double ProbabilityDistribution::gammap(double a, double x) noexcept {
    double gamser, gammcf, gln;

    if(x < math::ZERO_DOUBLE || a <= math::ZERO_DOUBLE) {
        std::cerr << "Invalid arguments in gammap" << std::endl;
        return math::ZERO_DOUBLE;
    }
    
    if(x < (a + math::ONE)) {
        // Use the series representation
        gser(gamser, a, x, gln);
        return gamser;
    } else {
        //  Use the continued fraction representation and take its complement
        gcf(gammcf, a, x, gln);
        return math::ONE - gammcf;
    }
}

void ProbabilityDistribution::gcf(double& gammcf, double a, double x, double& gln) noexcept {
    double an, b, c, d, del, h;

    gln = loggamma(a);
    b = x + math::ONE - a;
    c = math::ONE / precision::ZERO;
    d = math::ONE / b;
    h = d;

    // Iterate to convergence
    for(std::size_t i = 1; i <= ITMAX; i++) {
        an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += math::TWO;
        d = an * d + b;
        if(std::abs(d) < precision::ZERO) d = precision::ZERO;
        c = b + an / c;
        if(std::abs(c) < precision::ZERO) c = precision::ZERO;
        d = math::ONE / d;
        del = d * c;
        h *= del;
        if(std::abs(del - math::ONE) < precision::BW_TOLERANCE) break;
    }

    // Put factors in front
    gammcf = std::exp(-x + a * std::log(x) - gln) * h;
}

void ProbabilityDistribution::gser(double& gamser, double a, double x, double& gln) noexcept {
    double sum, del, ap;

    gln = loggamma(a);

    if(x <= math::ZERO_DOUBLE) {
        if(x < math::ZERO_DOUBLE)
            std::cerr << "x less than 0 in gser" << std::endl;

        gamser = math::ZERO_DOUBLE;
        return;
    }
    
    ap = a;
    del = sum = math::ONE / a;
    for(std::size_t n = 1; n <= ITMAX; n++) {
        ++ap;
        del *= x / ap;
        sum += del;
        if(std::abs(del) < std::abs(sum) * precision::BW_TOLERANCE) {
            gamser = sum * std::exp(-x + a * std::log(x) - gln);
            return;
        }
    }
    std::cerr << "a too large, ITMAX too small in gser" << std::endl;
}

double ProbabilityDistribution::errorf(double x) noexcept {
    return x < math::ZERO_DOUBLE ? -gammap(math::HALF, x * x) : gammap(math::HALF, x * x);
}

/* This code is adapted from 
 * http://www.codecogs.com/d-ox/maths/special/errorfn_inv.php
 */
double ProbabilityDistribution::errorf_inv(double y) noexcept {
    double s, t, u, w, x, z;

    const double k = y; // store y before switching its sign

    if(y == math::ZERO_DOUBLE) {
        return math::ZERO_DOUBLE;
    }

    if(y > math::ONE) {
        return std::numeric_limits<double>::infinity();
    }

    if(y < -math::ONE) {
        return -std::numeric_limits<double>::infinity();
    }

    if(y < 0)
        y = -y; // switch the sign of y if it's negative

    z = math::ONE - y;
    w = 0.916461398268964 - std::log(z);
    u = std::sqrt(w);
    s = (std::log(u) + 0.488826640273108) / w;
    t = math::ONE / (u + 0.231729200323405);
    x = u * (math::ONE - s * (s * 0.124610454613712 + math::HALF)) -
        ((((-0.0728846765585675 * t + 0.269999308670029) * t +
        0.150689047360223) * t + 0.116065025341614) * t +
        0.499999303439796) * t;
    t = 3.97886080735226 / (x + 3.97886080735226);
    u = t - math::HALF;
    s = (((((((((0.00112648096188977922 * u +
        1.05739299623423047e-4) * u - 0.00351287146129100025) * u -
        7.71708358954120939e-4) * u + 0.00685649426074558612) * u +
        0.00339721910367775861) * u - 0.011274916933250487) * u -
        0.0118598117047771104) * u + 0.0142961988697898018) * u +
        0.0346494207789099922) * u + 0.00220995927012179067;
    s = ((((((((((((s * u - 0.0743424357241784861) * u -
        0.105872177941595488) * u + 0.0147297938331485121) * u +
        0.316847638520135944) * u + 0.713657635868730364) * u +
        1.05375024970847138) * u + 1.21448730779995237) * u +
        1.16374581931560831) * u + 0.956464974744799006) * u +
        0.686265948274097816) * u + 0.434397492331430115) * u +
        0.244044510593190935) * t -
        z * std::exp(x * x - 0.120782237635245222);

    x += s * (x * s + math::ONE);

    return k < 0 ? -x : x; // function is symmetric about the origin
}


std::ostream& operator<<( std::ostream& os, 
        const libhmm::ProbabilityDistribution& /* p */ ){

    os << "Base ProbabilityDistribution class " << std::endl;
    return os;
}

}
