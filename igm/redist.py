"""Snow redistribution factor D of decision 4 (igm/DESIGN.md) on the lattice of inputs.py, and the fit of k.

  D = exp(k kappa) s, scaled to mean 1 on the 2021 mask
  kappa  Laplacian of swissALTI3D 2022 (2 m) after a 50 m x 50 m moving mean (25 x 25 cells), m-1, positive in
         hollows, averaged to the lattice (5 x 5 cells)
  s      1 below 30 deg, linear to 0 at 60 deg, slope of usurf_2021 (as posbal in inputs.py)
Fit: GLAMOS winter probings and snow pits of Basòdino (point mass balance release 2025, winter/basodino_winter.dat,
1127 values, LV03 -> LV95 with pyproj EPSG:21781 -> 2056). Modelled winter balance with D = 1: accumulation of smb.py
(snow below 0.5 degC, rain above 2.5, linear between) from 1 October to the probing date (29 February read as
28 February), at the probing height z_pos, from weather.py's OTL day with decision 5's transfer: T = T_OTL + g_m (z -
367 m), g_m per calendar month from the ROE - OTL daily means 1991-2025 over 1898 - 367 m; P = P_OTL (1 + 0.02 (z - 367
m) / 100 m), c_prec dropped (it is one of the yearly intercepts). Melt before the probing date is not modelled (its
parameters come from the calibration of step 2). Least squares: log(probed / modelled) = a_year + k kappa, one
intercept per hydrological year; probings with a balance <= 0 or off the lattice left out; standard error of k from the residual variance.
Variance shares (within-year, from the same intercept-only fit): log space, 1 - SSR(a_year + k kappa) / SSR(a_year);
and with the full D (s included, scaled) at the probings, intercepts refitted, in log and in mm w.e.
Inputs: s02 raw/glamos/point_2025/winter/basodino_winter.dat (unzipped from raw/glamos/massbalance_point_2025_r2025.zip),
igm/inputs/weather_otl_daily.csv, raw/climate/obs/ogd-smn_{otl,roe}_d_{historical,recent}.csv, derived/dem_2m.tif,
igm/inputs/usurf_2021.tif and mask_2021.tif (inputs.py).
Outputs (s02): igm/inputs/kappa.tif, slope_factor.tif, D.tif; igm/results/redist_fit.csv (k, SE, n, variance shares,
g_m), igm/results/redist_probings.csv (per probing: year, date, E, N, z, probed and modelled mm w.e., kappa, D).
usage (s02, agl env):  python igm/redist.py
"""
import csv
from datetime import date

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import uniform_filter

from inputs import DER, DX, E0, E1, N0, N1, NX, NY, OUT, RES, ROOT, read, slope_deg, write

POINTS = ROOT / "raw/glamos/point_2025/winter/basodino_winter.dat"
OBS = ROOT / "raw/climate/obs"
Z_OTL, Z_ROE, B_PREC, T_SNOW, T_RAIN, MARGIN = 367.0, 1898.0, 0.02 / 100, 0.5, 2.5, 100


def curvature():
    """kappa on the lattice from the 2 m model: 50 m moving mean, 5-point Laplacian at 2 m, 5 x 5 mean."""
    with rasterio.open(DER / "dem_2m.tif") as r:
        w = rasterio.windows.from_bounds(E0 - MARGIN, N0 - MARGIN, E1 + MARGIN, N1 + MARGIN, r.transform)
        z = r.read(1, window=w).astype(float)
        assert r.res == (2.0, 2.0) and not (z == r.nodata).any()
    z = uniform_filter(z, 25, mode="nearest")
    lap = np.zeros_like(z)
    lap[1:-1, 1:-1] = (z[2:, 1:-1] + z[:-2, 1:-1] + z[1:-1, 2:] + z[1:-1, :-2] - 4 * z[1:-1, 1:-1]) / 4.0
    m = MARGIN // 2
    lap = lap[m:-m, m:-m]
    assert lap.shape == (NY * 5, NX * 5)
    return lap.reshape(NY, 5, NX, 5).mean(axis=(1, 3))


def gradients():
    """g_m, degC m-1 per calendar month: mean daily ROE - OTL 1991-2025 over the height difference."""
    t = {}
    for st in ("otl", "roe"):
        for part in ("historical", "recent"):
            with open(OBS / f"ogd-smn_{st}_d_{part}.csv", encoding="latin-1") as f:
                for r in csv.DictReader(f, delimiter=";"):
                    d, v = r["reference_timestamp"][:10], r["tre200d0"]
                    if v and 1991 <= int(d[6:10]) <= 2025:
                        t.setdefault(d, {})[st] = float(v)
    g = np.zeros(13)
    for m in range(1, 13):
        dd = [v["roe"] - v["otl"] for d, v in t.items() if int(d[3:5]) == m and len(v) == 2]
        g[m] = np.mean(dd) / (Z_ROE - Z_OTL)
    return g


def probings():
    """Rows of the winter file: (date1, E, N, z, mb_we mm)."""
    lv95 = Transformer.from_crs(21781, 2056, always_xy=True)
    out = []
    for line in open(POINTS, encoding="latin-1"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        e, n = lv95.transform(float(f[7]), float(f[8]))
        out.append((date(int(f[3][:4]), int(f[3][4:6]), int(f[3][6:8])), e, n, float(f[9]), float(f[14])))
    return out


def lsq(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return beta, r


def main():
    kappa = curvature()
    z21, m21 = read("usurf_2021").astype(float), read("mask_2021") == 1
    s = np.clip((60.0 - slope_deg(z21)) / 30.0, 0.0, 1.0)
    g = gradients()

    days = {}
    with open(OUT / "weather_otl_daily.csv") as f:
        for r in csv.DictReader(f):
            days.setdefault(int(r["hydro_year"]), []).append((r["date"], float(r["T_degC"]), float(r["P_mm"])))
    rows = []
    for d, e, n, z, bw in probings():
        hy = d.year + (d.month >= 10)
        dd = date(d.year, 2, 28) if (d.month, d.day) == (2, 29) else d
        yr = days[hy]
        k = [x[0] for x in yr].index(dd.isoformat())
        months = np.array([int(x[0][5:7]) for x in yr[:k + 1]])
        T = np.array([x[1] for x in yr[:k + 1]]) + g[months] * (z - Z_OTL)
        P = np.array([x[2] for x in yr[:k + 1]]) * (1 + B_PREC * (z - Z_OTL))
        acc = np.where(T <= T_SNOW, P, np.where(T >= T_RAIN, 0.0, P * (T_RAIN - T) / (T_RAIN - T_SNOW))).sum()
        c, r_ = int((e - E0) // DX), int((N1 - n) // DX)
        if not (0 <= c < NX and 0 <= r_ < NY):                         # 2019-05-07 pit "S": N 151139 (LV03), 10 km off
            rows.append([hy, dd.isoformat(), e, n, z, bw, acc, np.nan, np.nan])
            continue
        rows.append([hy, dd.isoformat(), e, n, z, bw, acc, kappa[r_, c], s[r_, c]])
    a = np.array([r[2:] for r in rows])
    years = np.array([r[0] for r in rows])
    ok = (a[:, 3] > 0) & (a[:, 4] > 0) & ~np.isnan(a[:, 5])
    y, kap, sl, yrs = np.log(a[ok, 3] / a[ok, 4]), a[ok, 5], a[ok, 6], years[ok]
    Y = (yrs[:, None] == np.unique(yrs)[None, :]).astype(float)
    _, r0 = lsq(y, Y)
    beta, r1 = lsq(y, np.column_stack([Y, kap]))
    k = beta[-1]
    dof = len(y) - Y.shape[1] - 1
    se = np.sqrt(r1 @ r1 / dof / np.sum((kap - Y @ lsq(kap, Y)[0]) ** 2))

    D = np.exp(k * kappa) * s
    D /= D[m21].mean()
    write("kappa", kappa), write("slope_factor", s), write("D", D)
    Dp = np.exp(k * kap) * sl / (np.exp(k * kappa) * s)[m21].mean()       # D at the probings, scaled as the grid
    _, r2 = lsq(y - np.log(np.maximum(Dp, 1e-6)), Y)                     # full D at the probings, intercepts refitted
    # mm w.e.: probed against exp(a_year) x modelled (x D)
    a0, _ = lsq(y, Y)
    aD, _ = lsq(y - np.log(np.maximum(Dp, 1e-6)), Y)
    obs, mod = a[ok, 3], a[ok, 4]
    lin0 = obs - np.exp(Y @ a0) * mod
    linD = obs - np.exp(Y @ aD) * mod * Dp
    lin_tot = obs - np.array([obs[yrs == v].mean() for v in yrs])

    res = [("probings in the file", len(rows)), ("probings outside the lattice", int(np.isnan(a[:, 5]).sum())),
           ("probings with probed <= 0", int((a[:, 3] <= 0).sum())), ("probings used", int(ok.sum())),
           ("hydrological years (intercepts)", Y.shape[1]), ("k, m", f"{k:.4g}"), ("standard error of k, m", f"{se:.3g}"),
           ("within-year variance of log(probed/modelled) explained by k kappa", f"{1 - r1 @ r1 / (r0 @ r0):.4f}"),
           ("same, with the full D (slope factor, scaling) at the probings", f"{1 - r2 @ r2 / (r0 @ r0):.4f}"),
           ("within-year variance of probed mm w.e. explained, intercepts x modelled", f"{1 - lin0 @ lin0 / (lin_tot @ lin_tot):.4f}"),
           ("same, intercepts x modelled x D", f"{1 - linD @ linD / (lin_tot @ lin_tot):.4f}"),
           ("kappa on the 2021 mask, 5 / 50 / 95 percent, m-1", " ".join(f"{v:.4g}" for v in np.percentile(kappa[m21], [5, 50, 95]))),
           ("kappa at the probings, 5 / 50 / 95 percent, m-1", " ".join(f"{v:.4g}" for v in np.percentile(kap, [5, 50, 95]))),
           ("D on the 2021 mask, 5 / 50 / 95 percent", " ".join(f"{v:.4g}" for v in np.percentile(D[m21], [5, 50, 95]))),
           ("D on posbal, min / max", f"{D[read('posbal') == 1].min():.4g} {D[read('posbal') == 1].max():.4g}"),
           ("g_m Oct..Sep, K per 100 m", " ".join(f"{100 * g[m]:.3f}" for m in (10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9)))]
    with open(RES / "redist_fit.csv", "w", newline="") as f:
        csv.writer(f).writerows([["quantity", "value"], *res])
    with open(RES / "redist_probings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hydro_year", "date", "E", "N", "z", "probed_mmwe", "modelled_D1_mmwe", "kappa", "slope_factor", "used"])
        for r, u in zip(rows, ok):
            w.writerow([r[0], r[1], f"{r[2]:.1f}", f"{r[3]:.1f}", r[4], r[5], f"{r[6]:.1f}", f"{r[7]:.5g}", f"{r[8]:.3f}", int(u)])
    for q, v in res:
        print(q, v, sep="\t")


if __name__ == "__main__":
    main()
