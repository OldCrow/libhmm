#!/usr/bin/env python3
"""
prepare_dax_data.py — Download DAX and S&P 500 log-returns using yfinance.

Python equivalent of prepare_dax_data.R for systems without R/quantmod.

Usage:
    python scripts/prepare_dax_data.py [output_dir]

Output:
    dax_logreturns.csv   — single column: logreturn
    dax_2000_2022.csv    — Date, Close, logreturn
    sp500_logreturns.csv — single column: logreturn
"""

import sys
import os
import math
import yfinance as yf
import csv

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("TEMP", "/tmp"))


def fetch_logreturns(ticker: str, start: str, end: str) -> list[tuple]:
    """Return list of (date, close, logreturn) tuples, NA rows dropped."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    # yfinance >=0.2 returns MultiIndex columns; squeeze to a Series
    close_col = df["Close"]
    if hasattr(close_col, "squeeze"):
        close_col = close_col.squeeze()
    closes = close_col.dropna()
    rows = []
    prev = None
    for date, close in closes.items():
        if prev is not None:
            lr = math.log(float(close)) - math.log(float(prev))
            rows.append((date.strftime("%Y-%m-%d"), float(close), lr))
        prev = close
    return rows


print("Downloading ^GDAXI (DAX, 2000-01-01 to 2022-12-31)...")
dax_rows = fetch_logreturns("^GDAXI", "2000-01-01", "2022-12-31")
print(f"Downloaded {len(dax_rows)} daily log-returns")
print(f"Date range: {dax_rows[0][0]} to {dax_rows[-1][0]}")

dax_full_path = os.path.join(out_dir, "dax_2000_2022.csv")
dax_lr_path   = os.path.join(out_dir, "dax_logreturns.csv")

with open(dax_full_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Date", "Close", "logreturn"])
    w.writerows(dax_rows)

with open(dax_lr_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["logreturn"])
    for _, _, lr in dax_rows:
        w.writerow([lr])

print(f"\nExported:\n  {dax_lr_path} ({len(dax_rows)} rows)\n  {dax_full_path} (full data)")

print("\nDownloading ^GSPC (S&P 500, 2000-01-01 to 2022-12-31)...")
sp_rows = fetch_logreturns("^GSPC", "2000-01-01", "2022-12-31")
sp_path = os.path.join(out_dir, "sp500_logreturns.csv")
with open(sp_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["logreturn"])
    for _, _, lr in sp_rows:
        w.writerow([lr])
print(f"Exported {len(sp_rows)} S&P 500 log-returns -> {sp_path}")

print(f"\nDone. Run: dax_regime_example {out_dir}")
print(f"       or: sp500_regime_example {out_dir}")
