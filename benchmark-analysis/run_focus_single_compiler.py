import argparse
import csv
import pathlib
import re
import statistics
import subprocess


COMPILERS = {
    "msvc": {
        "pair_build": pathlib.Path(r"C:\Users\gdwol\Development\libhmm\build-focus-pairwise-ryzen-msvc"),
        "max_build": pathlib.Path(r"C:\Users\gdwol\Development\libhmm\build-focus-max-ryzen-msvc"),
        "out_dir": pathlib.Path(r"C:\Users\gdwol\Development\libhmm\benchmark-analysis\focus-n2-8-ryzen-windows-msvc-rerun"),
    },
    "clangcl": {
        "pair_build": pathlib.Path(r"C:\Users\gdwol\Development\libhmm\build-focus-pairwise-ryzen-clangcl"),
        "max_build": pathlib.Path(r"C:\Users\gdwol\Development\libhmm\build-focus-max-ryzen-clangcl"),
        "out_dir": pathlib.Path(r"C:\Users\gdwol\Development\libhmm\benchmark-analysis\focus-n2-8-ryzen-windows-clangcl-rerun"),
    },
    "mingw": {
        "pair_build": pathlib.Path(r"C:\Users\gdwol\Development\libhmm\build-focus-pairwise-ryzen-mingw"),
        "max_build": pathlib.Path(r"C:\Users\gdwol\Development\libhmm\build-focus-max-ryzen-mingw"),
        "out_dir": pathlib.Path(r"C:\Users\gdwol\Development\libhmm\benchmark-analysis\focus-n2-8-ryzen-windows-mingw-rerun"),
    },
}

N_VALUES = list(range(2, 9))
T_VALUES = [500, 1000, 2000, 5000, 10000, 100000]

FB_BLOCK_RE = re.compile(r"Forward-Backward phase breakdown:(.*?)Viterbi phase breakdown:", re.S)
NUM_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")


def parse_output(text: str) -> dict:
    block_match = FB_BLOCK_RE.search(text)
    if not block_match:
        raise RuntimeError("Could not find Forward-Backward breakdown block")
    block = block_match.group(1)

    def metric(label: str) -> float:
        for line in block.splitlines():
            if label in line:
                nums = NUM_RE.findall(line)
                if nums:
                    return float(nums[0])
        raise RuntimeError(f"Missing metric line for {label}")

    total_line = None
    for line in block.splitlines():
        if line.strip().startswith("TOTAL"):
            total_line = line
            break
    if total_line is None:
        raise RuntimeError("Missing TOTAL line in Forward-Backward block")

    total_nums = NUM_RE.findall(total_line)
    if not total_nums:
        raise RuntimeError("Missing TOTAL numeric value in Forward-Backward block")

    return {
        "fb_total_ms": float(total_nums[0]),
        "forward_ms": metric("Forward recursion"),
        "backward_ms": metric("Backward recursion"),
    }


def run_grid(build_dir: pathlib.Path, mode: str, runs: int, warmup: int) -> list:
    exe = build_dir / "tools" / "hotspot_breakdown.exe"
    if not exe.exists():
        raise FileNotFoundError(f"Missing executable: {exe}")
    rows = []
    for n in N_VALUES:
        for t in T_VALUES:
            proc = subprocess.run(
                [str(exe), str(n), str(t), str(runs), str(warmup)],
                cwd=str(build_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            parsed = parse_output(proc.stdout)
            rows.append(
                {
                    "mode": mode,
                    "n": n,
                    "t": t,
                    "runs": runs,
                    "warmup": warmup,
                    "fb_total_ms": parsed["fb_total_ms"],
                    "forward_ms": parsed["forward_ms"],
                    "backward_ms": parsed["backward_ms"],
                }
            )
    return rows


def write_csv(path: pathlib.Path, rows: list) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", choices=sorted(COMPILERS.keys()), required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    cfg = COMPILERS[args.compiler]
    out_dir = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = run_grid(cfg["pair_build"], "pairwise", args.runs, args.warmup)
    max_rows = run_grid(cfg["max_build"], "max_reduce", args.runs, args.warmup)

    pair_csv = out_dir / "focused_pairwise_n2_8.csv"
    max_csv = out_dir / "focused_max_reduce_n2_8.csv"
    cmp_csv = out_dir / "focused_pairwise_vs_max_reduce_n2_8.csv"

    write_csv(pair_csv, pair_rows)
    write_csv(max_csv, max_rows)

    pair_map = {(r["n"], r["t"]): r for r in pair_rows}
    cmp_rows = []
    for mr in max_rows:
        pr = pair_map[(mr["n"], mr["t"])]
        speedup = pr["fb_total_ms"] / mr["fb_total_ms"]
        cmp_rows.append(
            {
                "n": mr["n"],
                "t": mr["t"],
                "pairwise_fb_total_ms": pr["fb_total_ms"],
                "max_reduce_fb_total_ms": mr["fb_total_ms"],
                "speedup_max_over_pair": speedup,
                "winner": "max_reduce" if speedup > 1.0 else "pairwise",
            }
        )

    cmp_rows.sort(key=lambda row: (row["n"], row["t"]))
    write_csv(cmp_csv, cmp_rows)

    speedups = [row["speedup_max_over_pair"] for row in cmp_rows]
    max_wins = sum(1 for row in cmp_rows if row["winner"] == "max_reduce")
    pair_wins = len(cmp_rows) - max_wins
    print(
        f"{args.compiler}: points={len(cmp_rows)} max_wins={max_wins} "
        f"pair_wins={pair_wins} median={statistics.median(speedups):.6f}"
    )
    print(f"wrote: {pair_csv}")
    print(f"wrote: {max_csv}")
    print(f"wrote: {cmp_csv}")


if __name__ == "__main__":
    main()
