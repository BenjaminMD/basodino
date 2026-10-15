"""Observed values of the hindcast validation (decision 11, DESIGN.md step 5) and the flowline of the length change, on
the 10 m lattice of inputs/igm_static.npz (rows south to north). No model numbers.

  flowline  the SGI2016 centreline of C14-10 (derived/vectors.gpkg sgi_2016_centerlines, 1.255 km, top to tongue)
            sampled every 10 m, continued down the 1850 flowline: from the cell of its lower end, steepest descent (8
            neighbours, drop over distance) on usurf_2021 smoothed by a 3 x 3 cell mean (30 m; unsmoothed, the descent
            stops in a pit after 526 m), only on SGI1850 cells, no cell twice, until no neighbour is lower; asserted to
            end on or next to the lowest SGI1850 cell (then that cell is appended). line_cell (flat cell index), line_s (m from the top), line_e, line_n
  areas     ice area, all cells, at the inventory dates: SGI1931 1929-08-13 3.202 km2 (sigma: perimeter x 28 m map NMAD,
            0.23), SGI1973 2.320 (1973; 1 October save), SGI2010 1.897 (2008-2011; the 2010 save; inventory
            +-2.6 %), SGI2016 1.641 (acquired 2018; the 2018 save; inventory +-2.3 %) (TARGETS.md section 1)
  volume    GLAMOS all-pairs file (derived/glamos_basodino_volumechange.csv), the six survey pairs of decision 11, dV
            km3; sigma = |dV| sigma_B / |B_geod| (sigma of B_geod in m w.e. a-1 times the years times the mean area,
            as the file's dV); unknown survey days: 1907 -> 1907-10-01 (save), 1929 -> 1929-08-13 (the SGI1931
            image date)
  tongue    GLAMOS tongue elevation h_min (derived/glamos_basodino_lengthchange.csv), 2000-2024, at the 1 October save
            nearest the observation date
  length    cumulative GLAMOS length change since 1899-09-01 at the end dates nearest 1929, 1961, 1986 and 2024 (the
            decision-11 value is 1899-2024; the others are shown, not counted), at the nearest 1 October saves
  surfaces  mean surface over the observed cells: 1991-09-18 usurf_1994 on mask_1994; 2012-09-01 the 2012 surface of
            Grab et al. (2021) + 0.556 m (as targets_step4.py) on the SGI2016 cells (the outline of their grid); 2021-09-30 usurf_2021 on mask_2021
Expected misses, stated in decision 11 before any hindcast run (Huss et al. 2009, constant melt factors): the model
loses too little in 1929-1949 (1940s radiation high) and too much in 1949-1985 (dimming): column 'expected' of the
volume rows (1929-1949 'too little loss'; 1949-1961, 1961-1971, 1971-1985 'too much loss'); no other row has one.
Outputs: inputs/step5_targets.npz, igm/results/step5_targets.csv (kind, name, date_start, date_end, observed, sigma,
unit, expected, note), igm/results/step5_flowline.csv (s, E, N, cell, z_2021, part).
usage (s02, agl env):  python igm/targets_step5.py
"""
import csv
from datetime import date

import geopandas as gpd
import numpy as np
from scipy.ndimage import uniform_filter

from inputs import DER, DX, E0, N0, NX, NY, OUT, RES, tif_grid

DZ_STABLE_MEAN = 0.55649                                       # code/thickness.csv, as targets_step4.py
AREAS = (("SGI1931", "sgi1931", "1929-08-13", 3.202, 0.23, "map outline: perimeter 8205 m x NMAD 28 m (Mannerfelt et "
          "al. 2022); +-10 m shift 0.082"),
         ("SGI1973", "sgi1973", "1973-10-01", 2.320, 0.103, "aerial images 1973, day unknown: 1 October save; "
          "uncertainty not quantified, sigma = +-10 m shift"),
         ("SGI2010", "sgi2010", "2010-10-01", 1.897, 0.049, "images 2008-2011: the 2010 save; inventory total +-2.6 %"),
         ("SGI2016", "sgi2016", "2018-10-01", 1.641, 0.038, "acquired 2018: 1 October save; inventory total +-2.3 %"))
PAIRS = (("19079999", "19299999", "1907-10-01", "1929-08-13", ""),
         ("19299999", "19491007", "1929-08-13", "1949-10-07", "too little loss"),
         ("19491007", "19610919", "1949-10-07", "1961-09-19", "too much loss"),
         ("19610919", "19710903", "1961-09-19", "1971-09-03", "too much loss"),
         ("19710903", "19850910", "1971-09-03", "1985-09-10", "too much loss"),
         ("19850910", "19910918", "1985-09-10", "1991-09-18", ""))


def save_date(d):
    """The 1 October save nearest the date d (ISO string) as an ISO string."""
    d = date.fromisoformat(d)
    a, b = date(d.year - 1, 10, 1), date(d.year, 10, 1)
    return str(b if abs((b - d).days) <= abs((d - a).days) else a)


def flowline(g):
    c = gpd.read_file(DER / "vectors.gpkg", layer="sgi_2016_centerlines")
    line = c.loc[c["sgi-id"] == "C14-10", "geometry"].iloc[0]
    s_top = np.arange(0.0, line.length, DX)
    pts = [line.interpolate(s) for s in s_top]
    e, n = np.array([p.x for p in pts]), np.array([p.y for p in pts])
    x, y = g["x"].astype(float), g["y"].astype(float)
    col, row = np.floor((e - E0) / DX).astype(int), np.floor((n - N0) / DX).astype(int)
    cells, s, part = list(row * NX + col), list(s_top), [0] * len(s_top)
    z = uniform_filter(g["usurf_2021"].astype(float), 3)
    m = g["sgi1850"] > 0
    end = line.interpolate(line.length)
    r, k = int((end.y - N0) // DX), int((end.x - E0) // DX)
    seen, dist = {(r, k)}, float(line.length)
    cells.append(r * NX + k), s.append(dist), part.append(1)
    while True:
        best = None
        for dr in (-1, 0, 1):
            for dk in (-1, 0, 1):
                rr, kk = r + dr, k + dk
                if (dr, dk) == (0, 0) or not m[rr, kk] or (rr, kk) in seen:
                    continue
                slope = (z[r, k] - z[rr, kk]) / np.hypot(dr, dk)
                if best is None or slope > best[0]:
                    best = (slope, rr, kk)
        if best is None or best[0] <= 0:
            break
        dist += DX * np.hypot(best[1] - r, best[2] - k)
        r, k = best[1], best[2]
        seen.add((r, k))
        cells.append(r * NX + k), s.append(dist), part.append(1)
    low = np.argmin(np.where(m, g["usurf_2021"], np.inf))
    lr, lk = divmod(int(low), NX)
    assert max(abs(lr - r), abs(lk - k)) <= 1, (cells[-1], low)       # the smoothed minimum may sit one cell off
    if (lr, lk) != (r, k):
        dist += DX * np.hypot(lr - r, lk - k)
        cells.append(int(low)), s.append(dist), part.append(1)
    cells = np.array(cells)
    return cells, np.array(s), x[cells % NX], y[cells // NX], np.array(part)


def main():
    g = dict(np.load(OUT / "igm_static.npz"))
    t, rows = {}, []
    t["line_cell"], t["line_s"], t["line_e"], t["line_n"], part = flowline(g)
    zf = g["usurf_2021"].ravel()[t["line_cell"]]
    with open(RES / "step5_flowline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s_m", "E", "N", "cell", "z_2021", "part"])
        w.writerows([f"{a:.1f}", f"{b:.1f}", f"{c:.1f}", d, f"{zz:.1f}", ("centreline 2016", "1850 descent")[p]]
                    for a, b, c, d, zz, p in zip(t["line_s"], t["line_e"], t["line_n"], t["line_cell"], zf, part))

    for name, key, d, obs, sig, note in AREAS:
        rows.append(("area", name, "", d, obs, sig, "km2", "", note + f"; lattice {(g[key] > 0).sum() * 1e-4:.4f}"))

    vc = {(r[3], r[4]): r for r in csv.reader(open(DER / "glamos_basodino_volumechange.csv")) if len(r) > 12}
    for a, b, da, db, exp in PAIRS:
        r = vc[(a, b)]
        dv, B, sB = float(r[9]), float(r[11]), float(r[12])
        rows.append(("volume", f"{da[:4]}-{db[:4]}", da, db, dv, round(abs(dv) * sB / abs(B), 6), "km3", exp,
                     f"GLAMOS {r[1]} to {r[2]}, B_geod {B:+.3f} +- {sB:.3f} m w.e. a-1, areas {r[5]} to {r[7]} km2"
                     + ("; series file (online) -0.0051 with start area 2.595" if a == "19491007" else "")))

    lc = [r for r in csv.reader(open(DER / "glamos_basodino_lengthchange.csv")) if len(r) > 7 and r[1] == "C14-10"]
    for r in lc:
        if r[7] and r[4] >= "2000":
            rows.append(("tongue", f"tongue {r[4][:4]}", "", save_date(r[4]), float(r[7]), "", "m", "",
                         f"GLAMOS h_min, observed {r[4]}"))
    cum, ends = 0.0, {}
    for r in lc:
        cum += float(r[6])
        ends[r[4]] = cum
    for target in ("1929", "1961", "1986", "2024"):
        d = min(ends, key=lambda e: abs(int(e[:4]) - int(target)) + (e[:4] != target) * 0.5)
        rows.append(("length", f"length 1899-{d[:4]}", save_date("1899-09-01"), save_date(d), round(ends[d], 1), "",
                     "m", "", "decision 11" if target == "2024" else "shown, not counted (decision 11: 1899-2024)"))

    s12 = tif_grid(DER / "thickness/C14-10_SwissALTI3D_r2019.tif")[::-1] + DZ_STABLE_MEAN
    t["s12"] = np.where(np.isfinite(s12), s12, 0.0).astype(np.float32)
    t["has12"] = np.isfinite(s12) & (g["sgi2016"] > 0)
    for d, z, cells, note in (("1991-09-18", g["usurf_1994"], g["mask_1994"] == 1, "usurf_1994 on mask_1994"),
                              ("2012-09-01", t["s12"], t["has12"], "2012 surface (Grab et al. 2021) + 0.556 m on sgi2016"),
                              ("2021-09-30", g["usurf_2021"], g["mask_2021"] == 1, "usurf_2021 on mask_2021")):
        rows.append(("surface", f"surface {d[:4]}", "", d, round(float(z[cells].mean()), 3), "", "m", "",
                     f"mean over {cells.sum()} cells, {note}"))
    np.savez_compressed(OUT / "step5_targets.npz", **t)
    with open(RES / "step5_targets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "name", "date_start", "date_end", "observed", "sigma", "unit", "expected", "note"])
        w.writerows(rows)
    for r in rows:
        print(*r, sep=" | ")
    print(f"flowline: {len(t['line_s'])} samples, {t['line_s'][-1]:.0f} m, centreline {(part == 0).sum()} samples")


if __name__ == "__main__":
    main()
