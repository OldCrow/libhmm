#!/usr/bin/env python3
"""
generate_throughput_figure.py — Figure 4 for libhmm arXiv paper.

Plots forward-backward throughput (observations/ms) vs sequence length T
for libhmm, HMMLib, and StochHMM on discrete HMMs (Dishonest Casino, 2-state).

Data source: benchmark-analysis/ logs (adaptive SIMD build, Windows Ryzen 7,
MSVC Release). HMMLib uses scaled forward-backward; libhmm uses log-space.
StochHMM uses unscaled recursive computation.

Output: figure4_throughput.pdf / figure4_throughput.png
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Benchmark data — from benchmark-analysis/adaptive_libhmm_vs_hmmlib.log
# and benchmark-analysis/adaptive_libhmm_vs_stochhmm.log
# Hardware: Windows Ryzen 7, AVX-512, MSVC Release
# Model: Dishonest Casino (2-state, 2-symbol discrete HMM)
# Metric: forward-backward throughput (observations / ms)
# Note: HMMLib uses scaled FB; libhmm uses full log-space FB;
#       StochHMM uses unscaled recursive computation.
# ---------------------------------------------------------------------------

T_vals = [100, 500, 1000, 2000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000]

# libhmm throughput (obs/ms) — adaptive (SIMD) build, Casino problem
libhmm = [5263.2, 8333.3, 9090.9, 8849.6, 9920.6, 10929.0, 11027.8,
          10915.8, 10900.8, 11789.8]

# HMMLib throughput (obs/ms) — adaptive build, Casino problem
# HMMLib uses scaled forward-backward (not log-space)
hmmlib = [33333.3, 31250.0, 27777.8, 30769.2, 30303.0, 30303.0, 29868.6,
          30120.5, 29327.2, 30165.9]

# StochHMM throughput (obs/ms) — Casino problem
stochhmm = [2500.0, 2976.2, 4444.4, 3853.6, 4664.2, 4529.0, 4486.3,
            4589.7, 4489.3, 4449.8]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "lines.linewidth":  1.6,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

BLUE   = "#1f77b4"   # libhmm
ORANGE = "#ff7f0e"   # HMMLib
RED    = "#d62728"   # StochHMM

fig, ax = plt.subplots(figsize=(5.5, 3.5))

ax.semilogx(T_vals, libhmm,   color=BLUE,   marker="o",  ms=4,
            label="libhmm (log-space FB)")
ax.semilogx(T_vals, hmmlib,   color=ORANGE, marker="s",  ms=4,
            label="HMMLib (scaled FB, SSE)")
ax.semilogx(T_vals, stochhmm, color=RED,    marker="^",  ms=4,
            label="StochHMM (unscaled)")

ax.set_xlabel("Sequence length $T$")
ax.set_ylabel("Throughput (observations / ms)")
ax.set_title("Figure 4  Forward-backward throughput: libhmm vs HMMLib vs StochHMM\n"
             "(Dishonest Casino, 2-state discrete HMM, Windows Ryzen 7 / AVX-512 / MSVC)")

ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{int(x):,}"
))
ax.set_ylim(bottom=0)
ax.legend(loc="center right", framealpha=0.85)

# Annotate the throughput ratio at T=100000
ratio_hmmlib  = hmmlib[7]  / libhmm[7]
ratio_stochhmm = libhmm[7] / stochhmm[7]
ax.annotate(f"HMMLib: {ratio_hmmlib:.1f}× faster",
            xy=(100_000, hmmlib[7]), xytext=(100_000, hmmlib[7] + 3500),
            fontsize=7, color=ORANGE, ha="center",
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
ax.annotate(f"libhmm: {ratio_stochhmm:.1f}× faster",
            xy=(100_000, stochhmm[7]), xytext=(8_000, stochhmm[7] + 2500),
            fontsize=7, color=RED, ha="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figure4_throughput.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "figure4_throughput.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("Saved figure4_throughput.pdf")
print(f"\nKey ratios at T=100,000:")
print(f"  HMMLib / libhmm:  {ratio_hmmlib:.2f}x  (HMMLib faster — scaled FB vs log-space)")
print(f"  libhmm / StochHMM: {ratio_stochhmm:.2f}x  (libhmm faster)")
