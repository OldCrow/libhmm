#!/usr/bin/env python3
"""Generate tests/performance/trig_ulp_vectors.inc -- correctly-rounded
cos()/sin() reference vectors for the clean-room cos_pd/sin_pd SIMD kernels
(issue #74, commit 61e7347).

Backs tests/performance/test_trig_ulp_gates.cpp (the per-tier ULP gate for
cos_batch_<tier>/sin_batch_<tier>). Each entry is (input_bits, cos_bits,
sin_bits): cos/sin evaluated at 320-bit precision with mpmath, each rounded
once to nearest double. Architecture-neutral (pure mathematics) -- the same
reference set gates every ISA tier.

Buckets (main set, all inside the kernel's vectorized domain |x| <= 2^23 ==
kTrigDMax, per include/libhmm/detail/trig_cleanroom_data.inc):
  - uniform in [-2pi, 2pi] (the von Mises range)
  - uniform in [-1e4, 1e4]
  - log-uniform magnitude out to +/-2^23 (domain-wide coverage)
  - near k*pi/2 stress walk: for random k up to floor(2^23 * 2/pi), the
    nearest double to k*pi/2 plus a few nextafter neighbours either side.
    Both odd k (cos ~ 0, sin ~ +/-1) and even k (sin ~ 0, cos ~ +/-1) are
    included so reduction error is stressed for both functions' near-zero
    outputs.

Plus a separate small kTrigUlpSpecials array (same struct, outside the main
gate budget): +/-0.0, +/-kTrigDMax exactly, a few values just above
kTrigDMax (exercising the batch wrappers' scalar-libm fixup path), and
+/-inf, NaN (reference bits: cos/sin of +/-inf and NaN are NaN, so those
four entries are hardcoded rather than routed through mpmath).

Usage:  python scripts/gen_trig_ulp_vectors.py   (any Python 3 with mpmath)
Writes tests/performance/trig_ulp_vectors.inc relative to the repo root.
"""

import math
import os
import random
import struct

import mpmath as mp

mp.mp.prec = 320
SEED = 20260818  # documented fixed seed
random.seed(SEED)

D_MAX = float(2**23)  # must match kTrigDMax (trig_cleanroom_data.inc)

D = struct.Struct("<d")
Q = struct.Struct("<Q")


def bits(x: float) -> int:
    return Q.unpack(D.pack(x))[0]


def from_bits(b: int) -> float:
    return D.unpack(Q.pack(b))[0]


def cr_cos(x: float) -> float:
    return float(mp.cos(mp.mpf(x)))


def cr_sin(x: float) -> float:
    return float(mp.sin(mp.mpf(x)))


def vec(x: float) -> tuple:
    return (bits(x), bits(cr_cos(x)), bits(cr_sin(x)))


# ---------------------------------------------------------------------------
# Main buckets
# ---------------------------------------------------------------------------

bucket_counts = []
pts = []

# uniform [-2pi, 2pi]
n_before = len(pts)
for _ in range(1500):
    pts.append(random.uniform(-2 * math.pi, 2 * math.pi))
bucket_counts.append(("uniform_2pi", len(pts) - n_before))

# uniform [-1e4, 1e4]
n_before = len(pts)
for _ in range(1000):
    pts.append(random.uniform(-1e4, 1e4))
bucket_counts.append(("uniform_1e4", len(pts) - n_before))

# log-uniform magnitude out to +/-2^23
n_before = len(pts)
for _ in range(1000):
    mag = 2.0 ** random.uniform(-3, 23)
    pts.append(mag if random.random() < 0.5 else -mag)
bucket_counts.append(("log_uniform_domain", len(pts) - n_before))

# near k*pi/2 stress walk: both odd and even k, nearest double plus a few
# nextafter neighbours either side. K_MAX kept slightly below the true
# floor(2^23 * 2/pi) so that a handful of nextafter steps outward cannot
# push a generated point past kTrigDMax.
K_MAX = int(mp.floor(mp.mpf(D_MAX) * 2 / mp.pi)) - 8
OFFSETS = (-2, -1, 0, 1, 2)
n_before = len(pts)
stress_target = 1500
attempts = 0
while len(pts) - n_before < stress_target and attempts < stress_target * 20:
    attempts += 1
    parity = attempts % 2  # alternate odd/even k so both are well represented
    k = random.randint(0, K_MAX) * 2 + parity
    if random.random() < 0.5:
        k = -k
    xk = mp.mpf(k) * mp.pi / 2
    xk_d = float(xk)
    for off in OFFSETS:
        if len(pts) - n_before >= stress_target:
            break
        xoff = xk_d
        for _ in range(abs(off)):
            xoff = math.nextafter(xoff, math.inf if off > 0 else -math.inf)
        if abs(xoff) > D_MAX:
            continue
        pts.append(xoff)
bucket_counts.append(("near_k_pi_2_stress", len(pts) - n_before))

# self-check: every main-bucket point must sit inside the vectorized domain
for x in pts:
    assert abs(x) <= D_MAX, ("main-bucket point outside kTrigDMax", x)

main_vecs = [vec(x) for x in pts]

# ---------------------------------------------------------------------------
# Specials (outside the main gate budget; scalar-libm fixup + edge cases)
# ---------------------------------------------------------------------------

nan_bits = bits(math.nan)
pinf = math.inf
ninf = -math.inf

specials_finite = [
    0.0,
    -0.0,
    D_MAX,
    -D_MAX,
    math.nextafter(D_MAX, math.inf),
    2.0 * D_MAX,  # 2^24
    1e9,
    1e300,
]
# +/-inf and NaN FIRST (lanes 0-2): cos/sin are NaN. mpmath cannot evaluate
# these directly, so the reference bits are hardcoded to the NaN encoding.
# They lead the table so that every tier — including the 8-wide one, whose
# vector body covers lanes 0-7 of the 11 specials — evaluates them inside the
# SIMD kernel rather than in the scalar tail (which would be libm's answer).
specials = [(bits(x), nan_bits, nan_bits) for x in (pinf, ninf, math.nan)]
specials += [vec(x) for x in specials_finite]
specials_labels = [pinf, ninf, math.nan] + specials_finite

# self-check: NaN/Inf encodings are exactly what IEEE-754 double predicts
for (xb, cb, sb), xv in zip(specials[:3], (pinf, ninf, math.nan)):
    assert math.isnan(from_bits(cb)) and math.isnan(from_bits(sb)), (
        "specials NaN/Inf reference must be NaN",
        xv,
    )
    if math.isinf(xv):
        assert math.isinf(from_bits(xb)) and math.copysign(1.0, from_bits(xb)) == math.copysign(
            1.0, xv
        ), ("specials Inf encoding mismatch", xv)
    else:
        assert math.isnan(from_bits(xb)), ("specials NaN encoding mismatch", xv)

# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "tests", "performance", "trig_ulp_vectors.inc")
with open(out, "w") as f:
    f.write("// Auto-generated correctly-rounded cos()/sin() reference vectors.\n")
    f.write("// {input_bits, cos_bits, sin_bits}; cos/sin evaluated at 320-bit precision\n")
    f.write("// (mpmath) then each rounded once to nearest double. Gates the clean-room\n")
    f.write("// cos_pd/sin_pd SIMD kernels (issue #74) at a per-tier ULP budget.\n")
    f.write(f"// Fixed seed {SEED}. DO NOT EDIT -- regenerate with\n")
    f.write("// scripts/gen_trig_ulp_vectors.py.\n")
    f.write(
        "struct TrigUlpVector { std::uint64_t x_bits; std::uint64_t cos_bits; "
        "std::uint64_t sin_bits; };\n"
    )
    f.write(f"static constexpr TrigUlpVector kTrigUlpVectors[{len(main_vecs)}] = {{\n")
    for xb, cb, sb in main_vecs:
        f.write(f"    {{0x{xb:016x}ULL, 0x{cb:016x}ULL, 0x{sb:016x}ULL}},\n")
    f.write("};\n\n")
    f.write("// Specials: outside the main gate budget. Beyond-kTrigDMax finite points\n")
    f.write("// exercise the batch wrappers' per-lane scalar-libm fixup path; +/-inf and\n")
    f.write("// NaN must produce NaN exactly.\n")
    f.write(f"static constexpr TrigUlpVector kTrigUlpSpecials[{len(specials)}] = {{\n")
    for xb, cb, sb in specials:
        f.write(f"    {{0x{xb:016x}ULL, 0x{cb:016x}ULL, 0x{sb:016x}ULL}},\n")
    f.write("};\n")

print(f"wrote {out}: {len(main_vecs)} main vectors, {len(specials)} specials")
print("bucket counts: " + ", ".join(f"{name}={count}" for name, count in bucket_counts))
