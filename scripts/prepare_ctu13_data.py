#!/usr/bin/env python3
"""
prepare_ctu13_data.py — Pre-process CTU-13 Scenario 1 (Neris botnet) for
the zeek_anomaly_poc libhmm example.

Downloads the labeled binetflow from the CTU Malware Capture Facility if the
file is not already present in the output directory.

Dataset:
  CTU-Malware-Capture-Botnet-42 (maps to CTU-13 Scenario 1, Neris botnet)
  URL: https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/
       detailed-bidirectional-flow-labels/capture20110810.binetflow

  Garcia, S. et al. (2014). An empirical comparison of botnet detection
  methods. Computers and Security, 45, 100-123.

Output (written to <output_dir>/):
  ctu13_train.csv   — benign-only per-key observation sequences (training)
  ctu13_test.csv    — all per-key sequences (benign + botnet, testing)
  ctu13_labels.csv  — per-key ground-truth labels

Observation vector per row (log1p-transformed):
  log_inter_arrival, log_dur, log_tot_bytes, log_src_bytes

A key is the 4-tuple (src_ip, dst_ip, dst_port, proto).
Keys with fewer than MIN_SEQ_LEN observations are dropped.
Keys containing any Botnet-labeled flow are marked botnet=1.

Usage:
  python3 scripts/prepare_ctu13_data.py [output_dir] [binetflow_path]

  output_dir      Directory for output CSVs (default: /tmp)
  binetflow_path  Path to downloaded binetflow (default: /tmp/capture20110810.binetflow)
                  Downloaded automatically if absent.
"""

import csv
import math
import os
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

BINETFLOW_URL = (
    "https://mcfp.felk.cvut.cz/publicDatasets/"
    "CTU-Malware-Capture-Botnet-42/"
    "detailed-bidirectional-flow-labels/"
    "capture20110810.binetflow"
)
DEFAULT_BINETFLOW = "/tmp/capture20110810.binetflow"

# Minimum number of observations per key (i.e. flows - 1, since first flow
# has no inter-arrival time).
MIN_SEQ_LEN = 4   # ≥5 flows per key

# =============================================================================
# Helpers
# =============================================================================

def parse_ts(s: str) -> float:
    """Parse binetflow timestamp '2011/08/10 09:46:53.047277' → Unix float."""
    return datetime.strptime(s, "%Y/%m/%d %H:%M:%S.%f").timestamp()

def log1p(x: float) -> float:
    return math.log1p(max(x, 0.0))

# =============================================================================
# Load and group flows
# =============================================================================

def load_flows(binetflow_path: str):
    """
    Read binetflow, skip Background, group by (src, dst, dport, proto) key.
    Returns dict: key → list of (ts, dur, tot_bytes, src_bytes, is_botnet).
    """
    keys = defaultdict(list)
    n_skip = 0

    print(f"Reading {binetflow_path} …")
    with open(binetflow_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            lbl = row["Label"]
            if "Background" in lbl:
                n_skip += 1
                continue
            try:
                ts       = parse_ts(row["StartTime"])
                dur      = float(row["Dur"])
                tot_b    = float(row["TotBytes"])
                src_b    = float(row["SrcBytes"])
                proto    = row["Proto"]
                src_addr = row["SrcAddr"]
                dst_addr = row["DstAddr"]
                dport    = row["Dport"]
            except (ValueError, KeyError):
                continue
            is_bot = 1 if "Botnet" in lbl else 0
            key = (src_addr, dst_addr, dport, proto)
            keys[key].append((ts, dur, tot_b, src_b, is_bot))

    print(f"  Skipped {n_skip:,} Background flows.")
    print(f"  Loaded {sum(len(v) for v in keys.values()):,} Normal/Botnet flows "
          f"across {len(keys):,} keys.")
    return keys

# =============================================================================
# Build observation sequences
# =============================================================================

def build_sequences(keys):
    """
    For each key, sort flows by timestamp and compute per-observation vectors:
      [log1p(inter_arrival), log1p(dur), log1p(tot_bytes), log1p(src_bytes)]

    The first flow in each key has no inter-arrival time and is dropped.
    Keys with fewer than MIN_SEQ_LEN resulting observations are excluded.

    Returns: list of (key_str, obs_list, is_botnet)
      obs_list: list of [f0, f1, f2, f3]
    """
    sequences = []
    n_short = 0

    for key, flows in keys.items():
        flows_sorted = sorted(flows, key=lambda x: x[0])
        ts_list, dur_list, tot_list, src_list, bot_flags = zip(*flows_sorted)

        is_botnet = int(any(b == 1 for b in bot_flags))

        obs = []
        for i in range(1, len(flows_sorted)):
            ia = ts_list[i] - ts_list[i - 1]
            obs.append([
                log1p(ia),
                log1p(dur_list[i]),
                log1p(tot_list[i]),
                log1p(src_list[i]),
            ])

        if len(obs) < MIN_SEQ_LEN:
            n_short += 1
            continue

        key_str = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}"
        sequences.append((key_str, obs, is_botnet))

    n_benign = sum(1 for _, _, b in sequences if b == 0)
    n_botnet = sum(1 for _, _, b in sequences if b == 1)
    print(f"  After MIN_SEQ_LEN={MIN_SEQ_LEN} filter: "
          f"{n_benign} benign, {n_botnet} botnet, {n_short} too short → dropped.")
    return sequences

# =============================================================================
# Write output CSVs
# =============================================================================

HEADER = "key,obs_idx,log_inter_arrival,log_dur,log_tot_bytes,log_src_bytes\n"

def write_csv(path: str, sequences):
    with open(path, "w") as f:
        f.write(HEADER)
        for key_str, obs, _ in sequences:
            for i, row in enumerate(obs):
                f.write(f"{key_str},{i},{row[0]:.6f},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f}\n")
    print(f"  Written: {path}  ({os.path.getsize(path)//1024} KB)")

def write_labels(path: str, sequences):
    with open(path, "w") as f:
        f.write("key,is_botnet,n_obs\n")
        for key_str, obs, is_botnet in sequences:
            f.write(f"{key_str},{is_botnet},{len(obs)}\n")
    print(f"  Written: {path}")

# =============================================================================
# main
# =============================================================================

def main():
    out_dir      = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    binetflow    = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BINETFLOW

    os.makedirs(out_dir, exist_ok=True)

    # Download if absent
    if not os.path.exists(binetflow):
        print(f"Downloading binetflow to {binetflow} …")
        print(f"  URL: {BINETFLOW_URL}")
        urllib.request.urlretrieve(BINETFLOW_URL, binetflow)
        print(f"  Done ({os.path.getsize(binetflow)//1024//1024} MB)")

    # Load and build sequences
    keys      = load_flows(binetflow)
    sequences = build_sequences(keys)

    benign = [(k, o, b) for k, o, b in sequences if b == 0]
    all_   = sequences   # benign + botnet

    # Print summary
    total_benign_obs = sum(len(o) for _, o, _ in benign)
    total_botnet_obs = sum(len(o) for _, o, b in all_ if b == 1)
    print(f"\nSummary:")
    print(f"  Training keys (benign only): {len(benign):,}  "
          f"({total_benign_obs:,} observations)")
    print(f"  Botnet keys (test):          "
          f"{sum(1 for _,_,b in all_ if b==1):,}  "
          f"({total_botnet_obs:,} observations)")

    # Write output files
    print("\nWriting output files …")
    train_path  = os.path.join(out_dir, "ctu13_train.csv")
    test_path   = os.path.join(out_dir, "ctu13_test.csv")
    labels_path = os.path.join(out_dir, "ctu13_labels.csv")

    write_csv(train_path,  benign)
    write_csv(test_path,   all_)
    write_labels(labels_path, all_)

    print(f"\nRun the libhmm example:")
    print(f"  ./build/examples/zeek_anomaly_poc {out_dir}")

if __name__ == "__main__":
    main()
