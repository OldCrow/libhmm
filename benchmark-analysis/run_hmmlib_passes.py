import argparse
import csv
import os
import pathlib
import re
import statistics
import subprocess


LIBHMM_RE = re.compile(r"libhmm average throughput:\s*([0-9]+(?:\.[0-9]+)?)\s+observations/ms")
HMMLIB_RE = re.compile(r"HMMLib average throughput:\s*([0-9]+(?:\.[0-9]+)?)\s+observations/ms")
RATIO_RE = re.compile(r"Overall performance ratio:\s*([0-9]+(?:\.[0-9]+)?)x\s+\(HMMLib/libhmm\)")


def parse_summary(output: str) -> dict:
    m_libhmm = LIBHMM_RE.search(output)
    m_hmmlib = HMMLIB_RE.search(output)
    m_ratio = RATIO_RE.search(output)
    if not (m_libhmm and m_hmmlib and m_ratio):
        raise RuntimeError("Could not parse benchmark summary lines from comparator output")
    return {
        "libhmm_avg_obs_ms": float(m_libhmm.group(1)),
        "hmmlib_avg_obs_ms": float(m_hmmlib.group(1)),
        "ratio_hmmlib_over_libhmm": float(m_ratio.group(1)),
    }


def median(values: list[float]) -> float:
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--dll-dir", required=True)
    parser.add_argument("--passes", type=int, default=9)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    exe = pathlib.Path(args.exe)
    if not exe.exists():
        raise FileNotFoundError(f"Missing executable: {exe}")
    dll_dir = pathlib.Path(args.dll_dir)
    if not dll_dir.exists():
        raise FileNotFoundError(f"Missing DLL directory: {dll_dir}")

    env = os.environ.copy()
    env["PATH"] = f"{dll_dir};{env.get('PATH', '')}"

    rows = []
    for run_idx in range(1, args.passes + 1):
        proc = subprocess.run(
            [str(exe)],
            cwd=str(exe.parent),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        parsed = parse_summary(proc.stdout)
        row = {"label": args.label, "pass": run_idx}
        row.update(parsed)
        rows.append(row)
        print(
            f"{args.label} pass {run_idx}/{args.passes}: "
            f"libhmm={parsed['libhmm_avg_obs_ms']:.1f} "
            f"hmmlib={parsed['hmmlib_avg_obs_ms']:.1f} "
            f"ratio={parsed['ratio_hmmlib_over_libhmm']:.3f}"
        )

    out_csv = pathlib.Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "pass", "libhmm_avg_obs_ms", "hmmlib_avg_obs_ms", "ratio_hmmlib_over_libhmm"],
        )
        writer.writeheader()
        writer.writerows(rows)

    lib_vals = [row["libhmm_avg_obs_ms"] for row in rows]
    hm_vals = [row["hmmlib_avg_obs_ms"] for row in rows]
    ratio_vals = [row["ratio_hmmlib_over_libhmm"] for row in rows]
    print(
        f"{args.label} medians: "
        f"libhmm={median(lib_vals):.1f} hmmlib={median(hm_vals):.1f} "
        f"ratio={median(ratio_vals):.3f}"
    )
    print(f"wrote: {out_csv}")


if __name__ == "__main__":
    main()
