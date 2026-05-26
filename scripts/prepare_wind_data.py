#!/usr/bin/env python3
"""
prepare_wind_data.py — Download NOAA ISD wind data for wind_direction_example.

Python equivalent of prepare_wind_data.R for systems without R.

Downloads hourly wind observations from Chicago O'Hare International
Airport (NOAA ISD station 725300-14819) for 2015, extracts wind direction
and speed, and exports to a CSV file.

Usage:
    python scripts/prepare_wind_data.py [output_dir]

Output:
    ohare_wind_2015.csv — direction_rad, speed_ms columns

Data source:
    NOAA NCEI (2001): Global Surface Hourly [ISD]. NCEI.
    https://www.ncei.noaa.gov/data/global-hourly/
    Station: Chicago O'Hare (USAF 725300, WBAN 14819)
"""

import sys
import os
import csv
import math
import urllib.request
import tempfile

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEMP", "/tmp")

URL = "https://www.ncei.noaa.gov/data/global-hourly/access/2015/72530094846.csv"

print("Downloading NOAA ISD data for Chicago O'Hare (2015)...")
tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
tmp.close()  # close before writing via urlretrieve (required on Windows)
try:
    urllib.request.urlretrieve(URL, tmp.name)
    print(f"Downloaded to {tmp.name}")

    rows = []
    with open(tmp.name, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wnd = row.get("WND", "")
            parts = wnd.split(",")
            if len(parts) < 4:
                continue
            try:
                dir_deg = float(parts[0])
                spd_raw = float(parts[3])
            except ValueError:
                continue
            # 999 = missing direction, 9999 = missing speed, 0 = calm
            if dir_deg < 0 or dir_deg > 360:
                continue
            if spd_raw >= 9000 or spd_raw <= 0:
                continue
            dir_rad = dir_deg * math.pi / 180.0
            speed_ms = spd_raw / 10.0
            rows.append((dir_rad, speed_ms))

    print(f"Valid wind observations: {len(rows)}")
    speeds = [r[1] for r in rows]
    print(f"Speed range: {min(speeds):.1f} to {max(speeds):.1f} m/s")

    out_path = os.path.join(out_dir, "ohare_wind_2015.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["direction_rad", "speed_ms"])
        w.writerows(rows)

    print(f"\nExported: {out_path} ({len(rows)} rows)")
    print(f"\nDone. Run: wind_direction_example {out_dir}")
finally:
    os.unlink(tmp.name)
