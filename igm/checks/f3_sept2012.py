"""F3 check: the balance of 1-30 September 2012, which the observed dh 2012-2021 holds (survey 2012-09-01) but the _12
segments of fit_f3.py leave out (they start on 1 October 2012). Per 50 m band of the dh2 term (inputs/step4_targets.npz
dh2_band), the mean balance of days 335-364 of hydrological year 2012 (smb.py on the 2012 surface s12 of
inputs/step5_targets.npz, observed weather inputs/forcing_obs.npz, D and ipot of igm_static.npz, transfer of
clim_otl), converted to m ice (/ 0.91), for the F1 MAP draw and the F1 kept draws' 5 / 95 % of the band values.
Output: results/f3_sept2012.csv (band, cells, observed dh2, MAP and 5 / 95 % in m ice, the same in units of the dh
sigma 2.93 m).
usage (s03, env.sh):  python checks/f3_sept2012.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IGM))
from smb import ICE, Z_OTL, balance, transfer  # noqa: E402

DAY0, YEAR, SIGMA = 335, 2012, 2.93239


def main():
    sys.path.insert(0, str(IGM / "checks"))
    from f2_spinup import f1_map
    g = np.load(IGM / "inputs/igm_static.npz")
    t = np.load(IGM / "inputs/step4_targets.npz")
    s12 = np.load(IGM / "inputs/step5_targets.npz")["s12"].astype(np.float64)
    f = np.load(IGM / "inputs/forcing_obs.npz")
    k = int(np.flatnonzero(f["hydro_year"] == YEAR)[0])
    band = t["dh2_band"]
    cells = np.flatnonzero(band.ravel() >= 0)
    z = np.where(s12 > 0, s12, g["usurf_2021"]).ravel()[cells]
    D, I = g["D"].ravel()[cells], g["ipot"].reshape(365, -1)[:, cells]
    rows = [r for r in csv.DictReader(open(IGM / "results/calib_f1_draws.csv"))]
    m = f1_map()
    par = [(m["Fm"], m["c_prec"], m["dT"], m["dg"], m["b"])] + [
        (float(r["Fm_m_ice_per_day_degC"]), float(r["c_prec"]), float(r["dT_degC"]), float(r["dg_K_per_km"]) / 1e3,
         float(r["b_percent_per_100m"]) / 1e4) for r in rows]
    out = []
    for Fm, cp, dT, dg, b in par:
        T, P = transfer(f["t"][k][:, None], f["p"][k][:, None], f["gm"][:, None], z[None] - Z_OTL, dg, b)
        bd = balance((T.astype(np.float32) + dT).astype(np.float32), P.astype(np.float32), I, cp * D, Fm, 0.012 * Fm)
        out.append(bd[0, DAY0:].sum(0) / ICE)                     # m ice, (cell,)
    out = np.array(out)
    bc = band.ravel()[cells]
    res = []
    for j in range(len(t["dh2_obs"])):
        v = out[:, bc == j].mean(1)
        lo, hi = np.percentile(v[1:], [5, 95])
        res.append([f"{t['dh2_edges'][j]:.0f}-{t['dh2_edges'][j] + 50:.0f}", int((bc == j).sum()),
                    f"{t['dh2_obs'][j]:.3f}", f"{v[0]:.3f}", f"{lo:.3f}", f"{hi:.3f}", f"{v[0] / SIGMA:+.3f}"])
    with open(IGM / "results/f3_sept2012.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["band_m", "cells", "dh2_observed_m", "sept2012_MAP_m_ice", "sept2012_F1_p05", "sept2012_F1_p95",
                    "MAP_in_dh_sigma"])
        w.writerows(res)
    for r in res:
        print(*r, sep=" | ")


if __name__ == "__main__":
    main()
