#!/usr/bin/env python3
"""
generate_figures.py — Generate figures for the libhmm JOSS paper.

Produces three PDF figures in the same directory as this script:
  figure1_speedup.pdf       — Wall-time comparison: libhmm vs R packages
  figure2_convergence.pdf   — ECME EM log-likelihood convergence on DAX data
  figure3_wind_boundary.pdf — VonMises vs Normal boundary failure (wind data)

Figure 3 requires ohare_wind_2015.csv. Generate it with:
  python scripts/prepare_wind_data.py [output_dir]
Then pass the directory as an argument:
  python paper/figures/generate_figures.py [wind_data_dir]

All other figures are self-contained (data embedded below).

Dependencies: matplotlib, numpy
Figure 3 additionally: no extra dependencies (VonMises LL computed in numpy)

Data sources and references
----------------------------
DAX benchmark:   OldCrow/libhmm examples/dax_regime_example.cpp
                 Oelschläger et al. (2024), J. Stat. Softw. 109(9)
Elk benchmark:   OldCrow/libhmm examples/elk_movement_example.cpp
                 Michelot et al. (2016), J. Anim. Ecol. 85(4)
Earthquake:      OldCrow/libhmm examples/earthquake_example.cpp
                 Leroux & Puterman (1992), Biometrics 48(2)
Wind data:       NOAA NCEI Global Surface Hourly, station 725300-14819
                 Zucchini et al. (2017), Hidden Markov Models for Time Series
"""

import sys
import os
import math
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Output directory (same directory as this script)
# ---------------------------------------------------------------------------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Wind CSV location: optional first argument, else $TEMP / /tmp
WIND_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEMP", "/tmp")
WIND_CSV = os.path.join(WIND_DIR, "ohare_wind_2015.csv")

# ---------------------------------------------------------------------------
# Matplotlib style — minimal, publication-quality
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   1.5,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
RED    = "#d62728"
GRAY   = "#888888"
GREEN  = "#2ca02c"

# ===========================================================================
# Figure 1 — Wall-time comparison: libhmm vs R packages
# ===========================================================================
# Benchmark data from examples/README.md and dax_regime_example.cpp
# All times are median wall-clock on comparable hardware.
# DAX times: libhmm on Windows Ryzen (AVX-512); fHMM on Intel Ivy Bridge.
# Ratios are conservative (different hardware; fHMM hardware is faster).

benchmarks = [
    # label,              libhmm_ms,  r_ms,     r_package
    ("Earthquake\n(N=107)", 4,        20,       "HiddenMarkov"),
    ("Elk movement\n(N=14,394)", 99,  2_000,    "moveHMM"),
    ("DAX regimes\n(N=5,838)", 2_000, 1_360_000, "fHMM"),
]

fig1, ax1 = plt.subplots(figsize=(5.5, 3.0))

labels     = [b[0] for b in benchmarks]
libhmm_ms  = np.array([b[1] for b in benchmarks], dtype=float)
r_ms       = np.array([b[2] for b in benchmarks], dtype=float)
r_packages = [b[3] for b in benchmarks]

y      = np.arange(len(labels))
height = 0.35

bars_r = ax1.barh(y + height / 2, r_ms,       height, color=ORANGE, label="R package", zorder=3)
bars_l = ax1.barh(y - height / 2, libhmm_ms,  height, color=BLUE,   label="libhmm (C++)", zorder=3)

ax1.set_xscale("log")
ax1.set_xlabel("Wall time (ms, log scale)")
ax1.set_yticks(y)
ax1.set_yticklabels(labels)
ax1.set_title("Figure 1  Wall-time comparison: libhmm vs R packages")

# Annotate R package names and speedup factors
for i, (bm, lms, rms, rpkg) in enumerate(zip(labels, libhmm_ms, r_ms, r_packages)):
    speedup = rms / lms
    ax1.text(rms * 1.4, i + height / 2,
             f"{rpkg}\n({speedup:.0f}× slower)", va="center", fontsize=7, color=ORANGE)

ax1.legend(loc="lower right", framealpha=0.8)
ax1.set_xlim(right=ax1.get_xlim()[1] * 50)  # room for annotations
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, "figure1_speedup.pdf"), bbox_inches="tight")
fig1.savefig(os.path.join(OUT_DIR, "figure1_speedup.png"), bbox_inches="tight", dpi=150)
plt.close(fig1)
print("Saved figure1_speedup.pdf")

# ===========================================================================
# Figure 2 — ECME EM log-likelihood convergence on DAX 2000-2022
# Convergence data: dax_regime_example (libhmm v3.7.0, Windows/MSVC/AVX-512,
# 5838 log-returns). Reference values from prior runs noted in comments.
# ===========================================================================

# All 201 LL values (iteration 0 = initial, iterations 1-200 = EM steps)
ECME_LL = [
    17417.9816, 17461.9139, 17477.4168, 17483.1849, 17485.2242,
    17486.0142, 17486.3638, 17486.5420, 17486.6449, 17486.7103,
    17486.7550, 17486.7873, 17486.8115, 17486.8305, 17486.8457,
    17486.8583, 17486.8690, 17486.8782, 17486.8863, 17486.8936,
    17486.9002, 17486.9063, 17486.9119, 17486.9172, 17486.9221,
    17486.9268, 17486.9313, 17486.9355, 17486.9396, 17486.9435,
    17486.9473, 17486.9509, 17486.9544, 17486.9578, 17486.9611,
    17486.9642, 17486.9673, 17486.9703, 17486.9732, 17486.9761,
    17486.9788, 17486.9815, 17486.9842, 17486.9867, 17486.9892,
    17486.9917, 17486.9941, 17486.9965, 17486.9988, 17487.0010,
    17487.0032, 17487.0054, 17487.0076, 17487.0097, 17487.0117,
    17487.0138, 17487.0158, 17487.0177, 17487.0197, 17487.0216,
    17487.0235, 17487.0254, 17487.0272, 17487.0290, 17487.0308,
    17487.0326, 17487.0343, 17487.0360, 17487.0378, 17487.0394,
    17487.0411, 17487.0428, 17487.0444, 17487.0460, 17487.0476,
    17487.0492, 17487.0508, 17487.0524, 17487.0539, 17487.0554,
    17487.0569, 17487.0585, 17487.0599, 17487.0614, 17487.0629,
    17487.0644, 17487.0658, 17487.0672, 17487.0687, 17487.0701,
    17487.0715, 17487.0729, 17487.0743, 17487.0756, 17487.0770,
    17487.0784, 17487.0797, 17487.0811, 17487.0824, 17487.0837,
    17487.0850, 17487.0863, 17487.0876, 17487.0889, 17487.0902,
    17487.0915, 17487.0927, 17487.0940, 17487.0953, 17487.0965,
    17487.0977, 17487.0990, 17487.1002, 17487.1014, 17487.1026,
    17487.1038, 17487.1050, 17487.1062, 17487.1074, 17487.1086,
    17487.1098, 17487.1109, 17487.1121, 17487.1133, 17487.1144,
    17487.1156, 17487.1167, 17487.1178, 17487.1190, 17487.1201,
    17487.1212, 17487.1223, 17487.1234, 17487.1245, 17487.1256,
    17487.1267, 17487.1278, 17487.1289, 17487.1300, 17487.1310,
    17487.1321, 17487.1332, 17487.1342, 17487.1353, 17487.1363,
    17487.1374, 17487.1384, 17487.1394, 17487.1405, 17487.1415,
    17487.1425, 17487.1435, 17487.1445, 17487.1455, 17487.1465,
    17487.1475, 17487.1485, 17487.1495, 17487.1505, 17487.1515,
    17487.1525, 17487.1534, 17487.1544, 17487.1554, 17487.1563,
    17487.1573, 17487.1582, 17487.1592, 17487.1601, 17487.1611,
    17487.1620, 17487.1629, 17487.1638, 17487.1648, 17487.1657,
    17487.1666, 17487.1675, 17487.1684, 17487.1693, 17487.1702,
    17487.1711, 17487.1720, 17487.1729, 17487.1738, 17487.1747,
    17487.1756, 17487.1765, 17487.1773, 17487.1782, 17487.1791,
    17487.1799, 17487.1808, 17487.1816, 17487.1825, 17487.1833,
    17487.1842, 17487.1850, 17487.1859, 17487.1867, 17487.1875,
    17487.1884,
]

# Reference values
MOM_LL  = 17334.9   # kurtosis MOM M-step (libhmm v3.6.0, same data/init)
FHMM_LL = 17485.7   # fHMM 1.2.0 reference (nlm optimizer, Intel Ivy Bridge)

iters = np.arange(len(ECME_LL))
ll    = np.array(ECME_LL)

# Find where ECME first exceeds fHMM
surpass_iter = int(np.argmax(ll > FHMM_LL))

fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))

ax2.plot(iters, ll, color=BLUE, zorder=4, label="libhmm ECME (v3.7.0)")

# Reference lines
ax2.axhline(MOM_LL,  color=RED,  linestyle="--", linewidth=1.2,
            label=f"kurtosis MOM v3.6.0  (LL = {MOM_LL:.1f})", zorder=3)
ax2.axhline(FHMM_LL, color=GRAY, linestyle=":",  linewidth=1.2,
            label=f"fHMM 1.2.0 reference  (LL = {FHMM_LL:.1f})", zorder=3)

# Annotate where fHMM is surpassed
if surpass_iter > 0:
    ax2.axvline(surpass_iter, color=GRAY, linestyle=":", linewidth=0.8, zorder=2)
    ax2.annotate(f"Surpasses fHMM\nat iter {surpass_iter}",
                 xy=(surpass_iter, FHMM_LL),
                 xytext=(surpass_iter + 15, FHMM_LL - 25),
                 fontsize=7, color=GRAY,
                 arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))

# Delta annotation
delta = ll[-1] - MOM_LL
ax2.annotate(f"Δ = {delta:.1f} nats\nvs MOM",
             xy=(195, (ll[-1] + MOM_LL) / 2),
             fontsize=7, color=RED, ha="right", va="center",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

ax2.set_xlabel("EM iteration")
ax2.set_ylabel("Log-likelihood")
ax2.set_title("Figure 2  ECME convergence on DAX 2000–2022 "
              "(3-state Student-t HMM, N = 5,838)")
ax2.legend(loc="lower right", framealpha=0.8)
ax2.set_xlim(0, 200)
ax2.set_ylim(MOM_LL - 20, ll[-1] + 10)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "figure2_convergence.pdf"), bbox_inches="tight")
fig2.savefig(os.path.join(OUT_DIR, "figure2_convergence.png"), bbox_inches="tight", dpi=150)
plt.close(fig2)
print("Saved figure2_convergence.pdf")

# ===========================================================================
# Figure 3 — VonMises vs Normal boundary failure
# Wind directions, Chicago O'Hare 2015 (NOAA ISD)
# Fitted parameters from wind_direction_example (libhmm v3.7.0)
# ===========================================================================

# Fitted VonMises parameters from wind_direction_example output
VM_PREVAILING_MU    =  0.5436   # rad (31.1° NNE, concentrated state)
VM_PREVAILING_KAPPA =  2.3824
VM_VARIABLE_MU      = -2.3583   # rad (~225° SW, dispersed state; periodic equiv 3.9249)
VM_VARIABLE_KAPPA   =  1.8950

# Normal approximation parameters from HiddenMarkov R reference (embedded in wind example)
NORM_MU_PREV  = 0.8649   # rad (~49.6°)
NORM_SD_PREV  = 0.4684
NORM_MU_VAR   = 4.1726   # rad (~239.1°)
NORM_SD_VAR   = 1.0931

# Disagreement rate by 30° bin (from wind_direction_example empirical boundary analysis)
BIN_LABELS = [
    "0–30°\n(N/NNE)", "30–60°\n(NNE/NE)", "60–90°\n(NE/E)",
    "90–120°\n(E/SE)", "120–150°\n(SE)", "150–180°\n(S/SSW)",
    "180–210°\n(SW)", "210–240°\n(W/SW)", "240–270°\n(W/WNW)",
    "270–300°\n(NW/WNW)", "300–330°\n(NNW)", "330–360°\n(N/NNW)",
]
DISAGREE_RATE = [0.0, 0.0, 0.0, 15.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 47.7, 100.0]


def vonmises_logp(x, mu, kappa):
    """Log-likelihood of VonMises(mu, kappa) at angle x (radians)."""
    # log p = kappa*cos(x - mu) - log(2*pi*I0(kappa))
    # I0 via modified Bessel approximation (Abramowitz & Stegun 9.8.1)
    # Accurate to ~1e-7 for all kappa >= 0
    t = kappa / 3.75
    if kappa < 3.75:
        i0 = (1.0 + t**2 * (3.5156229 + t**2 * (3.0899424 + t**2 *
              (1.2067492 + t**2 * (0.2659732 + t**2 *
              (0.0360768 + t**2 * 0.0045813))))))
    else:
        i0 = (math.exp(kappa) / math.sqrt(kappa) *
              (0.39894228 + (1/t) * (0.01328592 + (1/t) *
              (0.00225319 + (1/t) * (-0.00157565 + (1/t) *
              (0.00916281 + (1/t) * (-0.02057706 + (1/t) *
              (0.02635537 + (1/t) * (-0.01647633 + (1/t) * 0.00392377)))))))))
    return kappa * math.cos(x - mu) - math.log(2 * math.pi * i0)


def norm_logp(x, mu, sd):
    z = (x - mu) / sd
    return -0.5 * z * z - math.log(sd)


if not os.path.isfile(WIND_CSV):
    print(f"Wind CSV not found at {WIND_CSV} — skipping Figure 3.")
    print("Generate it with: python scripts/prepare_wind_data.py [output_dir]")
else:
    # Load wind directions
    directions = []
    with open(WIND_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            directions.append(float(row["direction_rad"]))

    n = len(directions)
    print(f"Loaded {n} wind directions for Figure 3")

    # Assign VonMises and Normal states per observation
    vm_state   = []  # 0 = prevailing, 1 = variable
    norm_state = []  # 0 = prevailing, 1 = variable

    for x in directions:
        lp_prev_vm = vonmises_logp(x, VM_PREVAILING_MU, VM_PREVAILING_KAPPA)
        lp_var_vm  = vonmises_logp(x, VM_VARIABLE_MU,  VM_VARIABLE_KAPPA)
        vm_state.append(0 if lp_prev_vm > lp_var_vm else 1)

        lp_prev_n = norm_logp(x, NORM_MU_PREV, NORM_SD_PREV)
        lp_var_n  = norm_logp(x, NORM_MU_VAR,  NORM_SD_VAR)
        norm_state.append(0 if lp_prev_n > lp_var_n else 1)

    # Build 12-bin histograms of observations coloured by VonMises state
    n_bins = 12
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_prev  = np.zeros(n_bins, dtype=int)  # VonMises → prevailing
    bin_var   = np.zeros(n_bins, dtype=int)  # VonMises → variable

    for x, vs in zip(directions, vm_state):
        deg = math.degrees(x) % 360
        b   = min(int(deg / 30), 11)
        if vs == 0:
            bin_prev[b] += 1
        else:
            bin_var[b] += 1

    bin_total = bin_prev + bin_var
    theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False) + np.pi / n_bins

    # -----------------------------------------------------------------------
    # Two-panel figure
    # -----------------------------------------------------------------------
    fig3, (ax_rose, ax_bar) = plt.subplots(
        1, 2,
        figsize=(7.0, 3.5),
        subplot_kw=dict(polar=False),
        gridspec_kw=dict(width_ratios=[1.1, 1.4]),
    )
    plt.close(fig3)

    fig3 = plt.figure(figsize=(7.0, 3.5))
    ax_rose = fig3.add_subplot(121, projection="polar")
    ax_bar  = fig3.add_subplot(122)

    # --- Rose diagram (polar histogram) ---
    width = 2 * np.pi / n_bins

    # Bars go N→NE→E→S→W→N (meteorological convention: 0° = N clockwise)
    # matplotlib polar: 0 = East, counterclockwise. Convert:
    #   met_deg = 0° (N) → math_rad = π/2
    # Shift theta: math_theta = π/2 - met_theta
    # Bin 0: 0-30° met → math theta centred at π/2
    theta_plot = np.pi / 2 - theta  # clockwise from North → counterclockwise from East

    ax_rose.set_theta_zero_location("N")
    ax_rose.set_theta_direction(-1)   # clockwise

    bars_var  = ax_rose.bar(theta + np.pi / n_bins, bin_var,  width=width,
                            bottom=0, color=ORANGE, alpha=0.75, label="Variable (SW)")
    bars_prev = ax_rose.bar(theta + np.pi / n_bins, bin_prev, width=width,
                            bottom=bin_var, color=BLUE, alpha=0.85, label="Prevailing (NNE)")

    ax_rose.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax_rose.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=7)
    ax_rose.set_yticks([])
    ax_rose.set_title("(a) Wind rose by VonMises state\n(O'Hare 2015, N=11,894)", fontsize=8)
    ax_rose.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.15), fontsize=7)

    # Highlight the 330-360° problematic zone
    problem_theta = np.radians(345)  # centre of 330-360° bin
    ax_rose.annotate("",
                     xy=(problem_theta, bin_total[11] * 0.9),
                     xytext=(problem_theta, bin_total[11] * 1.35),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax_rose.text(problem_theta, bin_total[11] * 1.5, "100%\ndisagree",
                 ha="center", va="bottom", fontsize=6.5, color=RED)

    # --- Disagreement rate bar chart ---
    x_pos   = np.arange(n_bins)
    colours = [RED if r > 40 else (ORANGE if r > 10 else BLUE)
               for r in DISAGREE_RATE]
    ax_bar.bar(x_pos, DISAGREE_RATE, color=colours, zorder=3, width=0.75)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(
        [f"{i*30}°" for i in range(n_bins)],
        rotation=45, ha="right", fontsize=7,
    )
    ax_bar.set_ylabel("Disagreement rate (%)")
    ax_bar.set_ylim(0, 115)
    ax_bar.set_title("(b) VonMises vs Normal assignment\ndisagreement rate per 30° bin", fontsize=8)
    ax_bar.axhline(100, color=RED, linestyle="--", linewidth=0.8, alpha=0.5)

    # Annotate the 330-360° bar
    ax_bar.annotate("730 h, 100%\nmisclassified\nby Normal",
                    xy=(11, 100), xytext=(9.2, 72),
                    fontsize=6.5, color=RED, ha="center", va="top",
                    arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

    legend_handles = [
        mpatches.Patch(color=RED,    label="≥40% disagreement"),
        mpatches.Patch(color=ORANGE, label="10–40%"),
        mpatches.Patch(color=BLUE,   label="<10%"),
    ]
    ax_bar.legend(handles=legend_handles, fontsize=7, loc="upper left")

    fig3.suptitle(
        "Figure 3  VonMisesDistribution vs Normal approximation — boundary failure "
        "at 0°/360°",
        fontsize=9, y=1.01,
    )
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUT_DIR, "figure3_wind_boundary.pdf"), bbox_inches="tight")
    fig3.savefig(os.path.join(OUT_DIR, "figure3_wind_boundary.png"), bbox_inches="tight", dpi=150)
    plt.close(fig3)
    print("Saved figure3_wind_boundary.pdf")

print("\nDone. Figures written to:", OUT_DIR)
