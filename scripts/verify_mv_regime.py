#!/usr/bin/env python3
"""
verify_mv_regime.py — hmmlearn reference fit for mv_regime_example validation.

Fits a 3-state GaussianHMM (diagonal and full covariance) to SPY + QQQ monthly
log-returns and prints log-likelihoods and recovered state parameters for direct
comparison against libhmm's mv_regime_example output.

Usage:
    /tmp/libhmm_hmmlearn_venv/bin/python3 scripts/verify_mv_regime.py [data_dir]

  data_dir  Directory containing spy_qqq_monthly.csv (default: /tmp).
            Produce this file with: Rscript scripts/prepare_mv_regime_data.R

Setup (one-time):
    python3 -m venv /tmp/libhmm_hmmlearn_venv
    /tmp/libhmm_hmmlearn_venv/bin/pip install hmmlearn

Reference:
    Weiss, R., Seifert, Q., Adam, T., Cetinkaya, H., Oelschläger, L., & Zucchini, W.
    (2024). hmmlearn 0.3.3. https://hmmlearn.readthedocs.io
"""

import sys
import csv
import math
import os

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    import numpy as np
    from hmmlearn import hmm
except ImportError:
    print("hmmlearn not found. Install with:")
    print("  python3 -m venv /tmp/libhmm_hmmlearn_venv")
    print("  /tmp/libhmm_hmmlearn_venv/bin/pip install hmmlearn")
    print("Then run with:")
    print("  /tmp/libhmm_hmmlearn_venv/bin/python3 scripts/verify_mv_regime.py")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data_dir  = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
data_path = os.path.join(data_dir, "spy_qqq_monthly.csv")

if not os.path.exists(data_path):
    print(f"Data file not found: {data_path}")
    print("Run: Rscript scripts/prepare_mv_regime_data.R")
    sys.exit(1)

spy_vals, qqq_vals = [], []
with open(data_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        spy_vals.append(float(row["spy_logret"]))
        qqq_vals.append(float(row["qqq_logret"]))

X = np.column_stack([spy_vals, qqq_vals])
N = len(X)

print("SPY + QQQ Monthly Log-Returns — hmmlearn Reference Fit")
print("=======================================================")
print(f"Observations: {N}")
print(f"SPY: mean={np.mean(X[:,0]):+.3f}%  sd={np.std(X[:,0]):.3f}%")
print(f"QQQ: mean={np.mean(X[:,1]):+.3f}%  sd={np.std(X[:,1]):.3f}%")
print(f"Overall correlation: {np.corrcoef(X[:,0], X[:,1])[0,1]:.3f}\n")

N_STATES   = 3
N_RESTARTS = 20   # multiple restarts guard against local optima

# ---------------------------------------------------------------------------
# Fit helper — returns best model over N_RESTARTS random inits
# ---------------------------------------------------------------------------
def best_fit(cov_type: str) -> tuple:
    """Return (best_model, best_log_likelihood)."""
    best_ll  = -math.inf
    best_mod = None
    for seed in range(N_RESTARTS):
        try:
            model = hmm.GaussianHMM(
                n_components=N_STATES,
                covariance_type=cov_type,
                n_iter=500,
                tol=1e-4,
                random_state=seed,
                init_params="stmc",   # let hmmlearn init means, trans, priors, cov
                params="stmc",
            )
            model.fit(X)
            ll = model.score(X)
            if ll > best_ll:
                best_ll  = ll
                best_mod = model
        except Exception:
            continue
    return best_mod, best_ll


# ---------------------------------------------------------------------------
# Model A — diagonal covariance
# ---------------------------------------------------------------------------
print("--- Model A: GaussianHMM (covariance_type='diag') ---\n")
mod_a, ll_a = best_fit("diag")

# Sort states by mean SPY return (ascending: crisis → bear → bull)
order_a = np.argsort(mod_a.means_[:, 0])
labels  = ["Crisis", "Bear  ", "Bull  "]
for rank, s in enumerate(order_a):
    mu  = mod_a.means_[s]
    # hmmlearn 0.3+ stores diag covars as full (n, n) matrices with zeros
    # off-diagonal; np.diag() extracts the actual per-feature variances.
    std = np.sqrt(np.diag(np.asarray(mod_a.covars_[s])))
    print(f"State {s} ({labels[rank]}): "
          f"mu=[{float(mu[0]):+.3f}, {float(mu[1]):+.3f}]  "
          f"sd=[{float(std[0]):.3f}, {float(std[1]):.3f}]")
k_a = N_STATES * 2 + N_STATES * 2 + N_STATES * (N_STATES - 1) + (N_STATES - 1)
bic_a = -2 * ll_a + k_a * math.log(N)
print(f"Log-likelihood: {ll_a:.4f}")
print(f"BIC: {bic_a:.2f}  (k={k_a})\n")

# ---------------------------------------------------------------------------
# Model B — full covariance
# ---------------------------------------------------------------------------
print("--- Model B: GaussianHMM (covariance_type='full') ---\n")
mod_b, ll_b = best_fit("full")

order_b = np.argsort(mod_b.means_[:, 0])
for rank, s in enumerate(order_b):
    mu  = mod_b.means_[s]
    cov = mod_b.covars_[s]            # shape (2,2) for full
    std = np.sqrt(np.diag(cov))
    rho = cov[0, 1] / (std[0] * std[1])
    print(f"State {s} ({labels[rank]}): "
          f"mu=[{mu[0]:+.3f}, {mu[1]:+.3f}]  "
          f"sd=[{std[0]:.3f}, {std[1]:.3f}]  "
          f"rho={rho:.3f}")
k_b = N_STATES * 2 + N_STATES * (2 * 3 // 2) + N_STATES * (N_STATES - 1) + (N_STATES - 1)
bic_b = -2 * ll_b + k_b * math.log(N)
print(f"Log-likelihood: {ll_b:.4f}")
print(f"BIC: {bic_b:.2f}  (k={k_b})\n")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=== hmmlearn summary ===\n")
print(f"{'':30s} {'Model A (diag)':>16} {'Model B (full)':>16}")
print("-" * 62)
print(f"{'Log-likelihood':30s} {ll_a:>16.2f} {ll_b:>16.2f}")
print(f"{'Free parameters k':30s} {k_a:>16d} {k_b:>16d}")
print(f"{'BIC (lower = better)':30s} {bic_a:>16.2f} {bic_b:>16.2f}")
print()
if bic_b < bic_a:
    print("-> Model B wins: cross-sector correlation is informative.")
    print(f"   BIC improvement: {bic_a - bic_b:.1f} units.")
else:
    print("-> Model A wins: diagonal approximation is sufficient.")

print()
print("Run libhmm for comparison:")
print("  ./build/examples/mv_regime_example", data_dir)
print("Model B (full) log-likelihoods should agree to < 0.1 nat.")
print("Model A (diag) may differ by a few nats: hmmlearn uses multiple")
print("random restarts; libhmm uses fixed initialisation (single run).")
