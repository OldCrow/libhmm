import csv
import pathlib
import re
import subprocess
import statistics

compilers = {
    'msvc': {
        'pair_exe': pathlib.Path(r'C:\Users\gdwol\Development\libhmm\build-focus-pairwise-ryzen-msvc\tools\hotspot_breakdown.exe'),
        'max_exe': pathlib.Path(r'C:\Users\gdwol\Development\libhmm\build-focus-max-ryzen-msvc\tools\hotspot_breakdown.exe'),
        'out_dir': pathlib.Path(r'C:\Users\gdwol\Development\libhmm\benchmark-analysis\focus-n2-8-ryzen-windows-msvc-rerun'),
    },
    'clangcl': {
        'pair_exe': pathlib.Path(r'C:\Users\gdwol\Development\libhmm\build-focus-pairwise-ryzen-clangcl\tools\hotspot_breakdown.exe'),
        'max_exe': pathlib.Path(r'C:\Users\gdwol\Development\libhmm\build-focus-max-ryzen-clangcl\tools\hotspot_breakdown.exe'),
        'out_dir': pathlib.Path(r'C:\Users\gdwol\Development\libhmm\benchmark-analysis\focus-n2-8-ryzen-windows-clangcl-rerun'),
    },
    'mingw': {
        'pair_exe': pathlib.Path(r'C:\Users\gdwol\Development\libhmm\build-focus-pairwise-ryzen-mingw\tools\hotspot_breakdown.exe'),
        'max_exe': pathlib.Path(r'C:\Users\gdwol\Development\libhmm\build-focus-max-ryzen-mingw\tools\hotspot_breakdown.exe'),
        'out_dir': pathlib.Path(r'C:\Users\gdwol\Development\libhmm\benchmark-analysis\focus-n2-8-ryzen-windows-mingw-rerun'),
    },
}

n_vals = list(range(2, 9))
t_vals = [500, 1000, 2000, 5000, 10000, 100000]
runs = 5
warmup = 2

fb_block_re = re.compile(r'Forward-Backward phase breakdown:(.*?)Viterbi phase breakdown:', re.S)
num_re = re.compile(r'([0-9]+(?:\.[0-9]+)?)')

def parse_hotspot_output(text: str):
    m = fb_block_re.search(text)
    if not m:
        raise RuntimeError('Could not find FB breakdown block')
    block = m.group(1)

    def find_metric(label: str):
        for candidate in block.splitlines():
            if label in candidate:
                nums = num_re.findall(candidate)
                if nums:
                    return float(nums[0])
        raise RuntimeError(f'Missing metric line for {label}')

    total_line = None
    for candidate in block.splitlines():
        if candidate.strip().startswith('TOTAL'):
            total_line = candidate
            break
    if total_line is None:
        raise RuntimeError('Missing TOTAL line in FB block')

    total_nums = num_re.findall(total_line)
    if not total_nums:
        raise RuntimeError('No TOTAL numeric value in FB block')

    return {
        'fb_total_ms': float(total_nums[0]),
        'forward_ms': find_metric('Forward recursion'),
        'backward_ms': find_metric('Backward recursion'),
    }

def run_grid(exe: pathlib.Path, mode: str):
    rows = []
    for n in n_vals:
        for t in t_vals:
            proc = subprocess.run(
                [str(exe), str(n), str(t), str(runs), str(warmup)],
                capture_output=True,
                text=True,
                check=True,
            )
            metrics = parse_hotspot_output(proc.stdout)
            rows.append({
                'mode': mode,
                'n': n,
                't': t,
                'runs': runs,
                'warmup': warmup,
                'fb_total_ms': metrics['fb_total_ms'],
                'forward_ms': metrics['forward_ms'],
                'backward_ms': metrics['backward_ms'],
            })
    return rows

for compiler, cfg in compilers.items():
    out_dir = cfg['out_dir']
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = run_grid(cfg['pair_exe'], 'pairwise')
    max_rows = run_grid(cfg['max_exe'], 'max_reduce')

    pair_csv = out_dir / 'focused_pairwise_n2_8.csv'
    max_csv = out_dir / 'focused_max_reduce_n2_8.csv'
    cmp_csv = out_dir / 'focused_pairwise_vs_max_reduce_n2_8.csv'

    with pair_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)

    with max_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(max_rows[0].keys()))
        w.writeheader()
        w.writerows(max_rows)

    pair_map = {(r['n'], r['t']): r for r in pair_rows}
    cmp_rows = []
    for mr in max_rows:
        key = (mr['n'], mr['t'])
        pr = pair_map[key]
        speedup = pr['fb_total_ms'] / mr['fb_total_ms']
        cmp_rows.append({
            'n': mr['n'],
            't': mr['t'],
            'pairwise_fb_total_ms': pr['fb_total_ms'],
            'max_reduce_fb_total_ms': mr['fb_total_ms'],
            'speedup_max_over_pair': speedup,
            'winner': 'max_reduce' if speedup > 1.0 else 'pairwise',
        })

    with cmp_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(cmp_rows, key=lambda r: (r['n'], r['t'])))

    vals = [r['speedup_max_over_pair'] for r in cmp_rows]
    max_wins = sum(1 for r in cmp_rows if r['winner'] == 'max_reduce')
    pair_wins = len(cmp_rows) - max_wins
    print(f"{compiler}: points={len(cmp_rows)} max_wins={max_wins} pair_wins={pair_wins} median={statistics.median(vals):.6f}")

print('DONE')
