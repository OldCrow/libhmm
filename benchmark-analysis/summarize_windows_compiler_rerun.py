import csv
import math
import pathlib
import statistics


ROOT = pathlib.Path(r"C:\Users\gdwol\Development\libhmm\benchmark-analysis")

FOCUS = {
    "msvc": ROOT / "focus-n2-8-ryzen-windows-msvc-rerun" / "focused_pairwise_vs_max_reduce_n2_8.csv",
    "clangcl": ROOT / "focus-n2-8-ryzen-windows-clangcl-rerun" / "focused_pairwise_vs_max_reduce_n2_8.csv",
    "mingw": ROOT / "focus-n2-8-ryzen-windows-mingw-rerun" / "focused_pairwise_vs_max_reduce_n2_8.csv",
}

HMMLIB = {
    "msvc_control": ROOT / "hmmlib-9pass-ryzen-windows-msvc-rerun" / "control_passes.csv",
    "msvc_adaptive": ROOT / "hmmlib-9pass-ryzen-windows-msvc-rerun" / "adaptive_passes.csv",
    "mingw_control": ROOT / "hmmlib-9pass-ryzen-windows-mingw-rerun" / "control_passes.csv",
    "mingw_adaptive": ROOT / "hmmlib-9pass-ryzen-windows-mingw-rerun" / "adaptive_passes.csv",
    "clangcl_control": ROOT / "hmmlib-9pass-ryzen-windows-clangcl-rerun-o2" / "control_passes.csv",
    "clangcl_adaptive": ROOT / "hmmlib-9pass-ryzen-windows-clangcl-rerun-o2" / "adaptive_passes.csv",
}


def geomean(vals: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize_focus() -> None:
    print("FOCUSED_SWEEP_SUMMARY")
    for compiler, path in FOCUS.items():
        rows = read_csv(path)
        speedups = [float(r["speedup_max_over_pair"]) for r in rows]
        max_wins = sum(1 for r in rows if r["winner"] == "max_reduce")
        pair_wins = len(rows) - max_wins
        pair_vals = [float(r["pairwise_fb_total_ms"]) for r in rows]
        max_vals = [float(r["max_reduce_fb_total_ms"]) for r in rows]
        print(
            f"{compiler}: points={len(rows)} max_wins={max_wins} pair_wins={pair_wins} "
            f"median_speedup={statistics.median(speedups):.6f} "
            f"geomean_pair_ms={geomean(pair_vals):.6f} geomean_max_ms={geomean(max_vals):.6f}"
        )
        for n in range(2, 9):
            nrows = [r for r in rows if int(r["n"]) == n]
            n_max = sum(1 for r in nrows if r["winner"] == "max_reduce")
            print(f"  n={n}: max_wins={n_max}/{len(nrows)}")


def summarize_hmmlib() -> None:
    print("HMMLIB_9PASS_SUMMARY")
    med = {}
    for label, path in HMMLIB.items():
        rows = read_csv(path)
        lib_vals = [float(r["libhmm_avg_obs_ms"]) for r in rows]
        hm_vals = [float(r["hmmlib_avg_obs_ms"]) for r in rows]
        ratio_vals = [float(r["ratio_hmmlib_over_libhmm"]) for r in rows]
        med[label] = {
            "lib": statistics.median(lib_vals),
            "hm": statistics.median(hm_vals),
            "ratio": statistics.median(ratio_vals),
        }
        print(
            f"{label}: passes={len(rows)} med_libhmm={med[label]['lib']:.4f} "
            f"med_hmmlib={med[label]['hm']:.4f} med_ratio={med[label]['ratio']:.6f}"
        )

    msvc_delta = (med["msvc_adaptive"]["lib"] / med["msvc_control"]["lib"] - 1.0) * 100.0
    mingw_delta = (med["mingw_adaptive"]["lib"] / med["mingw_control"]["lib"] - 1.0) * 100.0
    clangcl_delta = (med["clangcl_adaptive"]["lib"] / med["clangcl_control"]["lib"] - 1.0) * 100.0
    print(f"msvc adaptive_vs_control delta_libhmm_pct={msvc_delta:.6f}")
    print(f"mingw adaptive_vs_control delta_libhmm_pct={mingw_delta:.6f}")
    print(f"clangcl adaptive_vs_control delta_libhmm_pct={clangcl_delta:.6f}")
    ctrl_mingw_vs_msvc = (med["mingw_control"]["lib"] / med["msvc_control"]["lib"] - 1.0) * 100.0
    adapt_mingw_vs_msvc = (med["mingw_adaptive"]["lib"] / med["msvc_adaptive"]["lib"] - 1.0) * 100.0
    ctrl_clangcl_vs_msvc = (med["clangcl_control"]["lib"] / med["msvc_control"]["lib"] - 1.0) * 100.0
    adapt_clangcl_vs_msvc = (med["clangcl_adaptive"]["lib"] / med["msvc_adaptive"]["lib"] - 1.0) * 100.0
    ctrl_clangcl_vs_mingw = (med["clangcl_control"]["lib"] / med["mingw_control"]["lib"] - 1.0) * 100.0
    adapt_clangcl_vs_mingw = (med["clangcl_adaptive"]["lib"] / med["mingw_adaptive"]["lib"] - 1.0) * 100.0
    print(f"mingw_vs_msvc control_libhmm_pct={ctrl_mingw_vs_msvc:.6f}")
    print(f"mingw_vs_msvc adaptive_libhmm_pct={adapt_mingw_vs_msvc:.6f}")
    print(f"clangcl_vs_msvc control_libhmm_pct={ctrl_clangcl_vs_msvc:.6f}")
    print(f"clangcl_vs_msvc adaptive_libhmm_pct={adapt_clangcl_vs_msvc:.6f}")
    print(f"clangcl_vs_mingw control_libhmm_pct={ctrl_clangcl_vs_mingw:.6f}")
    print(f"clangcl_vs_mingw adaptive_libhmm_pct={adapt_clangcl_vs_mingw:.6f}")


if __name__ == "__main__":
    summarize_focus()
    summarize_hmmlib()
