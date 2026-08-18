#!/usr/bin/env python3
"""Generate include/libhmm/detail/trig_cleanroom_data.inc -- constants for the
clean-room quadrant-reduction cos_pd/sin_pd kernels in simd_math_helpers.h.

Ported from libstats' scripts/gen_neon_trig_cleanroom_table.py (same owner,
MIT, clean-room derived, no third-party source) with ISA-neutral constant
names: this table backs cos_pd/sin_pd on every ISA tier (SSE2/AVX2/AVX-512/
NEON), not just NEON, so the libstats "kTrigNeon*" names are renamed
"kTrig*" here. The math is byte-for-byte identical to the libstats generator
-- same precision, same splits, same fits -- so the emitted constants are
expected to be bit-identical to libstats' src/neon_trig_cleanroom_data.inc.

Clean-room derived (2026-07-19, libstats): the reduction scheme, split
constants, and polynomial coefficients were authored independently from
first principles in an isolated child-agent workspace, from a functional
specification only, with zero access to any existing trigonometric
implementation. See (in libstats) docs/NEON_TRIG_DERIVATION.md for the
mathematics and docs/NEON_TRIG_DIVERGENCE_AUDIT.md for the point-by-point
comparison against ARM optimized-routines advsimd sin.c/cos.c confirming no
shared expression. No third-party source.

Design (libstats DERIVATION.md sections 1-4):
    Quadrant reduction x = n*(pi/2) + r, n = round(x * 2/pi), |r| <= pi/4;
    quadrant q = n mod 4 selects/negates the two parity cores
        sin(r) = r + r*(u*P(u)),  cos(r) = 1 + u*Q(u),  u = r^2.
    pi/2 is split into 4 parts with 30 significant bits each (last part full
    precision), so every product n*p_k is EXACT for |n| < 2^(53-30) = 2^23
    (exact-product lemma), giving supported domain D_MAX = 2^23
    (n_max = round(2^23 * 2/pi) = 5,340,354 < 2^23). The reduction is
    compensated: each inexact subtraction's residual is recovered exactly
    (rounding and large-cancellation are mutually exclusive per step), so the
    (r, rlo) pair carries the reduced argument to far better than double
    precision -- this is what holds the near-k*pi/2 stress error at 0.50 ULP.
    P and Q are degree-6 near-minimax fits (mpmath chebyfit at 320-bit
    precision) on u in [0, (pi/4 * (1+1e-6))^2]; degree 6 measured as the
    plateau for sin and the sweet spot for cos (plain Taylor at this degree
    cannot reach <1 ULP for cos).

Usage:  ./.venv/Scripts/python.exe scripts/gen_trig_cleanroom_table.py
Writes include/libhmm/detail/trig_cleanroom_data.inc relative to the repo root.
"""

import math
import os

import mpmath as mp

mp.mp.prec = 320

SIG_BITS = 30       # per-part significant bits -> n*part exact for |n| < 2^23
N_PARTS = 4          # 3 parts measured to fail (1.6 ULP stress); 2 catastrophic
D_MAX = float(2**23)
DEG = 6             # degree of both P and Q (u-polynomials)
FIT_PAD = 1.0 + 1e-6

PIO2 = mp.pi / 2
TWO_OVER_PI = float(mp.mpf(2) / mp.pi)
U_MAX = float((mp.pi / 4 * mp.mpf(FIT_PAD)) ** 2)


def as_hex(x: float) -> str:
    if x == 0.0 and math.copysign(1.0, x) < 0:
        return "-0x0p+0"
    return float(x).hex()


# --- split pi/2: exact-product parts --------------------------------------
def split_pio2(nparts: int, sig: int):
    parts = []
    R = mp.mpf(PIO2)
    for _ in range(nparts - 1):
        _, e = mp.frexp(R)
        g = int(e) - sig
        q = int(mp.nint(R / mp.mpf(2) ** g))
        assert abs(q) < 2 ** (sig + 1)
        p = float(mp.mpf(q) * mp.mpf(2) ** g)
        assert mp.mpf(p) == mp.mpf(q) * mp.mpf(2) ** g
        assert abs(q) * (2 ** (53 - sig) - 1) < 2**53 or q == 0
        parts.append(p)
        R = R - mp.mpf(p)
    parts.append(float(R))
    resid = abs(R - mp.mpf(float(R)))
    return parts, resid


PARTS, RESID = split_pio2(N_PARTS, SIG_BITS)
# truncation bound: |n|*delta must sit far below one ulp of the smallest
# reachable |r| near k*pi/2 (measured |r|_min ~ 2^-60.5; libstats DERIVATION.md sec. 3)
assert RESID < mp.mpf(2) ** -140, ("split truncation too large", RESID)
NMAX = int(mp.nint(mp.mpf(D_MAX) * 2 / mp.pi))
assert NMAX < 2 ** (53 - SIG_BITS), ("n_max breaks exact-product lemma", NMAX)


# --- polynomial cores ------------------------------------------------------
def h_sin(u):
    u = mp.mpf(u)
    if u < mp.mpf(2) ** -80:
        return mp.mpf(-1) / 6 + u / 120 - u**2 / 5040
    r = mp.sqrt(u)
    return (mp.sin(r) - r) / (u * r)


def h_cos(u):
    u = mp.mpf(u)
    if u < mp.mpf(2) ** -80:
        return mp.mpf(-1) / 2 + u / 24 - u**2 / 720
    return (mp.cos(mp.sqrt(u)) - 1) / u


def fit_poly(h, deg):
    coeffs = mp.chebyfit(h, [0, U_MAX], deg + 1)  # highest-degree first
    return list(reversed([float(c) for c in coeffs]))


def poly_mp(cs, u):
    acc = mp.mpf(0)
    for c in reversed(cs):
        acc = acc * u + mp.mpf(c)
    return acc


SIN_C = fit_poly(h_sin, DEG)
COS_C = fit_poly(h_cos, DEG)

# verify the double-rounded coefficient sets against 320-bit references
worst_s = mp.mpf(0)
worst_c = mp.mpf(0)
for i in range(2001):
    u = mp.mpf(U_MAX) * i / 2000
    r = mp.sqrt(u)
    if r > 0:
        s_approx = r + r * u * poly_mp(SIN_C, u)
        worst_s = max(worst_s, abs(s_approx / mp.sin(r) - 1))
    c_approx = 1 + u * poly_mp(COS_C, u)
    worst_c = max(worst_c, abs(c_approx / mp.cos(r) - 1))
assert worst_s < mp.mpf(2) ** -56, ("sin fit too weak", worst_s)
assert worst_c < mp.mpf(2) ** -56, ("cos fit too weak", worst_c)
# the kernel's exact 1 - u/2 head/tail split requires c0 == -1/2 exactly
assert COS_C[0] == -0.5, COS_C[0]

# --- emit ------------------------------------------------------------------
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "include", "libhmm", "detail", "trig_cleanroom_data.inc")
with open(out, "w") as f:
    f.write("// Auto-generated by scripts/gen_trig_cleanroom_table.py -- do not edit.\n")
    f.write("// Clean-room derived constants for cos_pd/sin_pd (issue #74); ported from\n")
    f.write("// libstats' scripts/gen_neon_trig_cleanroom_table.py (same owner, MIT,\n")
    f.write("// clean-room derived, no third-party source). See libstats'\n")
    f.write("// docs/NEON_TRIG_DERIVATION.md and docs/NEON_TRIG_DIVERGENCE_AUDIT.md for\n")
    f.write("// the mathematics and the point-by-point comparison against ARM\n")
    f.write("// optimized-routines advsimd sin.c/cos.c confirming no shared expression.\n\n")
    f.write(f"static constexpr double kTrigDMax = {as_hex(D_MAX)};  // 2^23\n")
    f.write(f"static constexpr double kTrigTwoOverPi = {as_hex(TWO_OVER_PI)};\n\n")
    f.write("// pi/2 split into exact-product parts: 30/30/30 significant bits + full\n")
    f.write("// tail; n*part exact for |n| < 2^23. Subtract in order with FMS.\n")
    f.write(f"static constexpr double kTrigPio2[{N_PARTS}] = {{\n")
    for p in PARTS:
        f.write(f"    {as_hex(p)},\n")
    f.write("};\n\n")
    f.write("// degree-6 near-minimax cores on u = r^2 in [0, (pi/4)^2 (padded)]:\n")
    f.write("// sin(r) = r + r*(u*P(u)); cos(r) = 1 + u*Q(u), Q[0] == -1/2 exactly\n")
    f.write("// (required by the kernel's exact 1 - u/2 head/tail split).\n")
    f.write(f"static constexpr double kTrigSinC[{DEG + 1}] = {{\n")
    for c in SIN_C:
        f.write(f"    {as_hex(c)},\n")
    f.write("};\n")
    f.write(f"static constexpr double kTrigCosC[{DEG + 1}] = {{\n")
    for c in COS_C:
        f.write(f"    {as_hex(c)},\n")
    f.write("};\n")

print(f"wrote {out}: {N_PARTS}-part split (resid 2^{mp.nstr(mp.log(RESID, 2), 5)}), "
      f"deg {DEG}/{DEG}, sin fit 2^{mp.nstr(mp.log(worst_s, 2), 5)}, "
      f"cos fit 2^{mp.nstr(mp.log(worst_c, 2), 5)}")
