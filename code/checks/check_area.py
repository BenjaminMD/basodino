"""Independent check of step 3 (area_clock.py), central case, written without reading area_clock.csv.

Start: H(u) = max(z_2021 - bed + u sigma_plus, 0) on the 2021 mask where the map has a bed (u >= 0). Melt per cell:
k x the sum of the GLAMOS bin balances (observation periods from 10 Sep 2021, 100 m bin of the 2021 elevation,
nearest bin outside the range) x 1000/900, minus 0.3 m; 2025/26 = the 2024/25 profile shifted to -3.503 m w.e.
(GLAMOS bin areas). A cell keeps ice where H > melt. Gate: u = 0, zero melt -> 1.379 km2. Then the u at which the
predicted area meets the observed one (and its -+ sigma) for 6 Sep 2025 and 29 Sep 2026, for k = 0.856 and 0.77, and
with an emergence of +0.33 m a-1 on every cell (the largest band value of the conventions, the map's best case).

Inputs: data/course/{dem,mask}_basodino_2021.asc, data/thickness/C14-10_{GlacierBed,IceThicknessUncertaintyPlus}.tif,
        data/glamos_basodino_massbalance_observation_elevationbins.csv, code/outline_2026.csv (observed areas)
Output: code/checks/check_area.csv
usage:  python code/checks/check_area.py
"""
import csv
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
asc = ROOT / "data/course/dem_basodino_2021.asc"
hd = dict(l.split()[:2] for l in open(asc).readlines()[:6])
z = np.loadtxt(asc, skiprows=6)
z[z == -9999] = np.nan
mask = np.loadtxt(ROOT / "data/course/mask_basodino_2021.asc", skiprows=6) == 1
cs, x0 = float(hd["cellsize"]), float(hd["xllcorner"])
y1 = float(hd["yllcorner"]) + z.shape[0] * cs
E, N = np.meshgrid(x0 + (np.arange(z.shape[1]) + .5) * cs, y1 - (np.arange(z.shape[0]) + .5) * cs)


def sample(name):
    with rasterio.open(ROOT / f"data/thickness/C14-10_{name}.tif") as src:
        v = np.array(list(src.sample(np.c_[E.ravel(), N.ravel()])))[:, 0].astype(float)
        v[v == src.nodata] = np.nan
    return v.reshape(z.shape)


bed, plus = sample("GlacierBed"), sample("IceThicknessUncertaintyPlus")
c = mask & np.isfinite(bed)
h0, sp, zc = (z - bed)[c], plus[c], z[c]

bins = {}
for r in csv.reader(open(ROOT / "data/glamos_basodino_massbalance_observation_elevationbins.csv", encoding="utf-8")):
    if len(r) > 10 and r[1] == "C14-10" and r[2] >= "2021-09-10":
        bins.setdefault((r[2], r[4]), []).append((float(r[9]), float(r[7]) / 1000, float(r[8])))
per = sorted(bins)
assert [p[0] for p in per[1:]] == [p[1] for p in per[:-1]] and per[-1][1] == "2025-09-06", per


def bal(b, zz):
    lo = np.array([x[0] for x in sorted(b)])
    return np.array([x[1] for x in sorted(b)])[np.clip(np.searchsorted(lo, zz, side="right") - 1, 0, len(lo) - 1)]


b25 = sorted(bins[per[-1]])
shift = -3.503 - sum(x[1] * x[2] for x in b25) / sum(x[2] for x in b25)
raw = {"2025-09-06": -sum(bal(bins[p], zc) for p in per) / 0.9}
raw["2026-09-29"] = raw["2025-09-06"] - bal([(x[0], x[1] + shift, x[2]) for x in b25], zc) / 0.9
years = {"2025-09-06": 4, "2026-09-29": 5}
obs = {r["scene"][:10]: (float(r["area_m2"]) / 1e6, float(r["pixel_x_perimeter_m2"]) / 1e6)
       for r in csv.DictReader(open(ROOT / "code/outline_2026.csv")) if r["scene"][:4] in ("2025", "2026")}
obs = {"2025-09-06": obs["2025-09-07"], "2026-09-29": obs["2026-09-27"]}

U = np.arange(0, 3.0001, 0.01)
area = lambda u, m: ((h0 + u * sp > m) & (h0 + u * sp > 0)).sum() * cs * cs / 1e6
rows = [("case", "end", "value", "low", "high", "note"),
        ("gate u=0 no melt", "2021-09-30", round(area(0, 0), 4), "", "", "km2, expected 1.379"),
        ("volume u=0", "2021-09-30", round(np.clip(h0, 0, None).sum() * 100 / 1e9, 6), "", "", "km3")]
for k, emerg in ((0.856, 0.0), (0.77, 0.0), (0.856, 0.33)):
    for end, m in raw.items():
        melt = np.maximum(k * m - 0.3 - emerg * years[end], 0)
        a = np.array([area(u, melt) for u in U])
        um = [round(float(U[np.argmax(a >= t)]), 2) if a[-1] >= t else ">3" for t in (obs[end][0], obs[end][0] - obs[end][1],
                                                                    obs[end][0] + obs[end][1])]
        name = f"k {k}, emergence {emerg}"
        rows += [(f"area u=0, {name}", end, round(a[0], 3), "", "", f"observed {obs[end][0]:.3f} +- {obs[end][1]:.3f}"),
                 (f"u_match, {name}", end, *um, "")]
        if isinstance(um[0], float):
            rows.append((f"volume at u_match, {name}", end,
                         round(np.clip(h0 + um[0] * sp, 0, None).sum() * 100 / 1e9, 4), "", "", "km3"))
for r in rows:
    print(*r, sep="\t")
csv.writer(open(ROOT / "code/checks/check_area.csv", "w", newline="")).writerows(rows)
