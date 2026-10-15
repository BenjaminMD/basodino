"""Observed targets of the start-state and flow fit (decision 9 as amended, DESIGN.md step 4), on the 10 m lattice of
inputs/igm_static.npz (rows south to north). No model numbers: observations, their sigma and the cells they refer to.

  areas   ice area by each target's definition at its date, km2; sigma = the area change for a 10 m shift of the
          outline: 2021-09-30 course mask 1.6065 (all cells; +-0.109, TARGETS.md section 1), 2024-08-23 swissTLM3D
          all polygons 1.3801 (+-0.112, same table), 2025-09-07 and 2026-09-27 Sentinel-2 main body 1.2311 and 1.0210
          (code/outline_2026.csv area_main_body_m2; sigma 10 m x the main-body perimeter perimeter_m, 8413 and 9250 m)
  dh      mean surface change per 50 m band, 1991-09-18 to 2012-09-01 (bands of usurf_1994) and 2012-09-01 to
          2021-09-30 (bands of the 2012 surface), m; cells: inside mask_1994 (the only cells whose surface the model
          can change) where the 2012 surface of Grab et al. (2021) (derived/thickness/C14-10_SwissALTI3D_r2019.tif)
          has a value; the 2012
          surface raised by 0.556 m, the mean of z_2021 - z_2012 on ice-free ground (code/thickness.csv dz_stable_mean;
          usurf_1994 equals usurf_2021 off the glacier, TARGETS.md section 2); bands with at least MIN_CELLS cells;
          sigma 2.93 m per band = the standard deviation of z_2021 - z_2012 on that ground (dz_stable_std), taken as
          fully correlated within a band
  radar   the 2006 profiles 1-6 of inputs/radar_2006.csv (points whose bedrock lies above usurf_2021 left out): mean
          observed thickness per profile, sigma 5.3 m (decision 9); the lattice cell of every kept point
  stakes  one-sided: the modelled 2021 thickness, largest over the cells whose centre lies within 30 m of any position
          of the series (code/stake_positions.csv, written by code/stake_clock.py), at least melt_after_30sep2021_m of
          code/stake_clock.csv; one bound per series: the provisional 2026 row (Fig. A.1 of the GLAMOS 2026 report)
          where it exists ('with 2026'), else its release row; the release rows alone are kept as the variant
          'release' (2025/26 values provisional); sigma STAKE_SIGMA = 1 m (the margin of code/stake_clock.py)
  speed   ITS_LIVE bound of TARGETS.md section 7 (inputs.ITS_BOUND, ITS_SIGMA): median 2014-2021 surface speed over
          the 96 pixels <= 1 m a-1, sigma 0.5 m a-1 above (the model side is track.py its_median)
Outputs: inputs/step4_targets.npz (arrays named below), results/step4_targets.csv (kind, name, date(s), observed,
sigma, cells, note).
usage (s02, agl env):  python igm/targets_step4.py   (reads igm/inputs/stake_positions.csv and stake_clock.csv,
                       copies of code/ rsynced from the Mac)
"""
import csv

import numpy as np

from inputs import DER, DX, E0, N0, NX, NY, OUT, RES, ITS_BOUND, ITS_SIGMA, tif_grid

DZ_STABLE_MEAN, DZ_STABLE_STD = 0.55649, 2.93239        # code/thickness.csv
MIN_CELLS, STAKE_SIGMA, DISC = 20, 1.0, 30.0
AREAS = (("2021-09-30", "all", 1.6065, 0.109, "course mask 2021 (TARGETS.md section 1, +-10 m)"),
         ("2024-08-23", "all", 1.38008, 0.112, "swissTLM3D 2024, all polygons (TARGETS.md section 1, +-10 m)"),
         ("2025-09-07", "body", 1.231086, 0.08413, "Sentinel-2 main body (code/outline_2026.csv; 10 m x 8413 m)"),
         ("2026-09-27", "body", 1.021007, 0.09250, "Sentinel-2 main body (code/outline_2026.csv; 10 m x 9250 m)"))


def bands(z, cells, dh):
    """Band index grid (-1 off), lower edges, observed means and cell counts, 50 m bands of z on cells."""
    low = np.floor(np.where(cells, z, np.nan) / 50) * 50
    edges = [e for e in np.unique(low[cells]) if (low == e).sum() >= MIN_CELLS]
    grid = np.full(z.shape, -1, np.int16)
    for k, e in enumerate(edges):
        grid[low == e] = k
    obs = np.array([dh[grid == k].mean() for k in range(len(edges))])
    n = np.array([(grid == k).sum() for k in range(len(edges))])
    return grid, np.array(edges), obs, n


def main():
    g = dict(np.load(OUT / "igm_static.npz"))
    s94, s21, m94 = g["usurf_1994"].astype(float), g["usurf_2021"].astype(float), g["mask_1994"] == 1
    s12 = tif_grid(DER / "thickness/C14-10_SwissALTI3D_r2019.tif")[::-1] + DZ_STABLE_MEAN    # rows south to north
    has12 = np.isfinite(s12)
    t, rows = {}, []
    t["dh1_band"], t["dh1_edges"], t["dh1_obs"], t["dh1_n"] = bands(s94, has12 & m94, s12 - s94)
    t["dh2_band"], t["dh2_edges"], t["dh2_obs"], t["dh2_n"] = bands(s12, has12 & m94, s21 - s12)
    for p, d in (("dh1", "1991-09-18 2012-09-01"), ("dh2", "2012-09-01 2021-09-30")):
        rows += [("dh", f"{p} {e:.0f}-{e + 50:.0f} m", d, f"{o:.3f}", f"{DZ_STABLE_STD:.3f}", n, "")
                 for e, o, n in zip(t[f"{p}_edges"], t[f"{p}_obs"], t[f"{p}_n"])]

    rad = list(csv.DictReader(open(OUT / "radar_2006.csv")))
    keep = [r for r in rad if r["above_2021"] == "0"]
    t["radar_cell"] = np.array([(NY - 1 - int(r["row"])) * NX + int(r["col"]) for r in keep])
    t["radar_profile"] = np.array([int(r["profile"]) for r in keep])
    t["radar_obs"] = np.array([float(r["thickness"]) for r in keep])
    t["radar_profile_obs"] = np.array([t["radar_obs"][t["radar_profile"] == p].mean() for p in range(1, 7)])
    rows += [("radar", f"profile {p}", "2006", f"{o:.3f}", "5.3", int((t["radar_profile"] == p).sum()),
              f"{sum(r['profile'] == str(p) for r in rad)} points, bedrock above usurf_2021 left out")
             for p, o in zip(range(1, 7), t["radar_profile_obs"])]

    clock = {(r["series"], r["end"]): r for r in csv.DictReader(open(OUT / "stake_clock.csv"))}
    pos = {}
    for r in csv.DictReader(open(OUT / "stake_positions.csv")):
        pos.setdefault((r["series"], r["end"]), []).append((float(r["easting"]), float(r["northing"])))
    X, Y = np.meshgrid(E0 + DX * (np.arange(NX) + 0.5), N0 + DX * (np.arange(NY) + 0.5))
    series = sorted({s for s, _ in clock})
    names, bound, prov, release, cells = [], [], [], [], []
    for s in series:
        ends = sorted(e for s_, e in clock if s_ == s)
        rel = [e for e in ends if clock[(s, e)]["provisional"] == "no"][-1]
        for e in ends:
            disc = np.zeros(X.shape, bool)
            for x, y in pos[(s, e)]:
                disc |= np.hypot(X - x, Y - y) <= DISC
            names.append(f"{s} {e}")
            bound.append(float(clock[(s, e)]["melt_after_30sep2021_m"]))
            prov.append(clock[(s, e)]["provisional"] == "yes")
            release.append(e == rel)
            cells.append(np.flatnonzero(disc.ravel()))
            rows.append(("stake", f"{s} to {e}", f"{clock[(s, e)]['start']} {e}", f"{bound[-1]:.2f}", STAKE_SIGMA,
                         disc.sum(), ("provisional (Fig. A.1); " if prov[-1] else "") +
                         ("used in 'with 2026'" if e == ends[-1] else "") +
                         ("; " if e == ends[-1] and e == rel else "") + ("used in 'release'" if e == rel else "")))
    last = {n.split()[0]: k for k, n in enumerate(names)}
    t["stake_name"], t["stake_bound"] = np.array(names), np.array(bound)
    t["stake_provisional"], t["stake_release"] = np.array(prov), np.array(release)
    t["stake_with2026"] = np.isin(np.arange(len(names)), list(last.values()))
    t["stake_cells"] = np.full((len(names), max(len(c) for c in cells)), -1)
    for k, c in enumerate(cells):
        t["stake_cells"][k, :len(c)] = c

    t["area_date"] = np.array([a[0] for a in AREAS])
    t["area_kind"] = np.array([a[1] for a in AREAS])
    t["area_obs"] = np.array([a[2] for a in AREAS])
    t["area_sigma"] = np.array([a[3] for a in AREAS])
    rows += [("area", f"{a[1]} {a[0]}", a[0], a[2], a[3], "", a[4]) for a in AREAS]
    t["speed_bound"], t["speed_sigma"] = np.array(ITS_BOUND), np.array(ITS_SIGMA)
    rows.append(("speed", "ITS_LIVE median 2014-2021, 96 pixels", "2014 2021", ITS_BOUND, ITS_SIGMA, 96,
                 "one-sided: model at most the bound"))
    np.savez_compressed(OUT / "step4_targets.npz", **t)
    with open(RES / "step4_targets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "name", "dates", "observed", "sigma", "cells", "note"])
        w.writerows(rows)
    for r in rows:
        print(*r, sep=" | ")


if __name__ == "__main__":
    main()
