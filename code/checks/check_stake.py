"""Independent check of step 2 (stake_clock.py): melt at the GLAMOS stake sites after 30 Sep 2021 against the largest
mapped 2021 thickness within 30 m, written without reading stake_clock.py's code or CSV.

Melt = -mb_raw (cm of ice) summed over the readings from 10 Sep 2021, stakes of one site and period averaged, a series
broken where the stake moved more than 60 m or a period is missing; minus 0.3 m (10-30 Sep 2021) for series from
10 Sep 2021. 2025/26: a third transcription of Fig. A.1 (code/checks/fig_a1_third.csv), placed with the scale bar
(north up) and the translation that matches most triangles to 2024/25 stakes within 30 m; a live series takes the
nearest triangle within 60 m; melt = -b x 1000/900. Map: max(z_2021 - bed, 0) on the 2021 mask (0 elsewhere), the
largest over cell centres within 30 m of any position of the series; +1 sigma with the bed lowered by the plus grid.
Emergence test: excess minus 0.33 m a-1 x the years of the series (the largest band value of the conventions).

Inputs: data/glamos_point_2025/basodino_annual.dat, code/checks/fig_a1_third.csv, data/course/{dem,mask}_basodino_2021.asc,
        data/thickness/C14-10_{GlacierBed,IceThicknessUncertaintyPlus}.tif, data/outline_2026/outline_2026-09-27.geojson
Output: code/checks/check_stake.csv
usage:  python code/checks/check_stake.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from shapely.geometry import Point, shape

ROOT = Path(__file__).resolve().parents[2]
OFF = np.array([2e6, 1e6])                              # LV03 -> LV95, plain offset

recs = [l.split() for l in open(ROOT / "data/glamos_point_2025/basodino_annual.dat", encoding="latin-1")
        if not l.startswith("#") and l.strip()]
recent = [r for r in recs if r[1] >= "20210910"]
print(f"{len(recs)} readings, {len(recent)} since 10 Sep 2021")
per = {}
for r in recent:
    per.setdefault((r[0].rstrip("b")[-2:], r[1]), []).append(r)
sites = {}
for (s, d0), rs in sorted(per.items()):
    sites.setdefault(s, []).append({"d0": d0, "days": float(rs[0][5]), "melt": -np.mean([float(r[11]) for r in rs]) / 100,
                                    "xy": np.mean([[float(r[7]), float(r[8])] for r in rs], 0) + OFF})
series = {}
for s, ps in sites.items():
    runs = [[ps[0]]]
    for a, b in zip(ps, ps[1:]):
        nxt = int(a["d0"][:4]) + 1 == int(b["d0"][:4])
        (runs[-1].append(b) if nxt and np.hypot(*(a["xy"] - b["xy"])) <= 60 else runs.append([b]))
    for i, run in enumerate(runs):
        series[s + ("ab"[i] if len(runs) > 1 else "")] = run

# Fig. A.1, third reading
tri = list(csv.DictReader(open(ROOT / "code/checks/fig_a1_third.csv")))
px = np.array([[float(t["column_px"]), -float(t["row_px"])] for t in tri]) * 600 / (698.5 - 522.5)
live = {k: v[-1]["xy"] for k, v in series.items() if v[-1]["d0"] == "20240903"}
ref = np.array(list(live.values()))


def matches(shift):
    d = np.hypot(*((px + shift)[:, None] - ref[None]).transpose(2, 0, 1))
    return d.argmin(1), d.min(1)


shift = max((ref[j] - px[i] for i in range(len(px)) for j in range(len(ref))),
            key=lambda t: (matches(t)[1] <= 30).sum())
j, d = matches(shift)
shift = shift + np.mean(ref[j[d <= 30]] - (px + shift)[d <= 30], 0)
j, d = matches(shift)
print(f"Fig. A.1: {(d <= 30).sum()} triangles within 30 m, residuals {np.round(d[d <= 30], 1)}")
link = {}
for i in np.argsort(d):
    name = list(live)[j[i]]
    if d[i] <= 60 and name not in link and tri[i]["balance_mwe"]:
        link[name] = (i, -float(tri[i]["balance_mwe"]) / 0.9, (px[i] + shift))

# map on the course lattice
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
ok = mask & np.isfinite(bed)
H = np.where(ok, np.clip(z - bed, 0, None), 0)
Hp = np.where(ok, np.clip(z - bed + plus, 0, None), 0)
outline = shape(json.load(open(ROOT / "data/outline_2026/outline_2026-09-27.geojson"))["features"][0]["geometry"])

rows = [("series", "end", "readings", "melt_after_m", "map_max_m", "map_plus_sigma_m", "excess_m",
         "excess_minus_emergence_m", "inside_2026", "fig_a1_triangle", "fig_a1_offset_m")]
for name, run in sorted(series.items()):
    for prov in (False, True):
        if prov and name not in link:
            continue
        pos = [p["xy"] for p in run] + ([link[name][2]] if prov else [])
        melt = sum(p["melt"] for p in run) + (link[name][1] if prov else 0)
        melt -= 0.3 if run[0]["d0"] == "20210910" else 0
        near = np.zeros(z.shape, bool)
        for e, n in pos:
            near |= np.hypot(E - e, N - n) <= 30
        years = len(run) + prov
        hm, hpm = H[near].max(), Hp[near].max()
        rows.append((name, "2026-09-29" if prov else str(int(run[-1]["d0"][:4]) + 1), years, round(melt, 2),
                     round(hm, 2), round(hpm, 2), round(melt - hm, 2), round(melt - hm - 0.33 * years, 2),
                     outline.contains(Point(*pos[-1])), tri[link[name][0]]["triangle"] if prov else "",
                     round(float(np.hypot(*(pos[-1] - pos[-2]))), 1) if prov else ""))
for r in rows:
    print(*r, sep="\t")
csv.writer(open(ROOT / "code/checks/check_stake.csv", "w", newline="")).writerows(rows)
