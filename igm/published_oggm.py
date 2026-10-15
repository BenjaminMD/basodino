"""Basodino (RGI60-11.01987) rows of OGGM v1.6.1 standard projections (per-glacier files, one GCM and SSP each).

Usage: python3 published_oggm.py OUT.csv FILE.nc ...  Appends gcm, ssp, year, volume_km3, area_km2 (state on
1 January of the year) to OUT.csv; writes the header if OUT.csv is new.
"""
import csv
import sys
from pathlib import Path

import h5py

RGI = "RGI60-11.01987"

out = Path(sys.argv[1])
new = not out.exists()
with out.open("a", newline="") as fo:
    w = csv.writer(fo)
    if new:
        w.writerow(["gcm", "ssp", "year", "volume_km3", "area_km2"])
    for path in sys.argv[2:]:
        gcm, ssp = Path(path).name.split("_gcm_merged_")[1].split("_bc_")[0].rsplit("_", 1)
        with h5py.File(path) as f:
            ids = [s.decode() if isinstance(s, bytes) else s for s in f["rgi_id"][:]]
            i = ids.index(RGI)
            years, vol, area = f["time"][:], f["volume"][:, i], f["area"][:, i]
        w.writerows((gcm, ssp, int(y), f"{v / 1e9:.6g}", f"{a / 1e6:.6g}") for y, v, a in zip(years, vol, area))
