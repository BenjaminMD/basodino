"""Step 2 (DESIGN.md, decision 7): size of the steps the calibration geometry makes at its cuts. The ice mask of a
hydrological year is the outline nearest in time to 1 April of that year (calib_smb.OUTLINES), so it switches between
two years; the surface is linear in time between the 1991, 2012 and 2021 surfaces (no step, a kink in 2012).
Per switch year i (outline of year i-1 -> outline of year i), on every cell (stride 1), with the MAP draw of a run in
numpy float64 (smb.step): area of both masks, the year-i hydrological-year balance (1 October - 30 September)
averaged over each mask, and the year-i volume change over each mask (km3 ice, 850 kg m-3) against the GLAMOS sigma of
the geodetic period holding the year (per year: sigma_obs / years). Surface: largest |z(i) - z(i-1)| on the lattice per
year, around the 2012 kink and elsewhere.
Output: igm/results/calib_cuts{_TAG}.csv (quantity, value).
usage (s02, agl env, from basodino/):  python igm/checks/calib_cuts.py [TAG]     (TAG as calib_step2_check.py)
"""
import csv
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IGM))
import calib_smb as c  # noqa: E402
from smb import step, transfer  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
SFX = f"_{TAG}" if TAG else ""
RES = IGM / "results"
out = []


def put(k, v):
    out.append((k, v))
    print(k, "\t", v, flush=True)


run = sorted((RES / f"calib{SFX}").glob("round*.npz"))[-1]
z = np.load(run)
th = np.r_[z["theta"][int(np.argmax(z["weight"]))], 0.0, c.B_PREC][:6]       # earlier runs: dg = 0, b = B_PREC
put(f"MAP draw of {run.parent.name}/{run.name} ({', '.join(c.NAMES)})", " ".join(f"{v:.5g}" for v in th))

X = dict(np.load(IGM / "inputs/calib_stride1.npz"))
years, near, nc, G = X["years"], X["near"], int(X["ncell"]), X["groups"]
kind, pc = X["kind"], X["pieces"]
area_g = {int(i): int(g) for j, g, i, a, e in pc if kind[j] == "dv"}           # year index -> mask group (area)
dv_span = {}                                                                   # year index -> geodetic target
for j, g, i, a, e in pc:
    if kind[j] == "dv":
        dv_span[int(i)] = j
cell_km2 = 1e-4                                                                # stride 1: 10 m cells
put("hydrological years and outline nearest in time", " ".join(f"{y}:{n}" for y, n in zip(years, near)))

Fm, r = np.full((1, 1), th[0]), np.full((1, 1), th[1])
D = th[2] * X["D"][None, :]
for i in range(1, len(years)):
    if near[i] == near[i - 1] or i not in area_g or i - 1 not in area_g:
        continue
    T, P = transfer(X["t_otl"][i][:, None], X["p_otl"][i][:, None], X["gm"][:, None], X["z"][i][None, :] - c.Z_OTL,
                    th[4], th[5])
    T = T + th[3]
    S, ba = np.zeros((1, T.shape[1])), np.zeros(T.shape[1])
    for k in range(365):
        S, b = step(S, T[k], P[k], X["ipot"][k], D, Fm, r, 0.75 * r)
        ba += b[0]
    m0, m1 = G[area_g[i - 1]][:nc] > 0, G[area_g[i]][:nc] > 0
    a0, a1 = m0.sum() * cell_km2, m1.sum() * cell_km2
    v0, v1 = (ba[:nc][m].sum() * cell_km2 * 1e-3 / c.RHO for m in (m0, m1))
    j = dv_span[i]
    yrs = (np.datetime64(X["d1"][j]) - np.datetime64(X["d0"][j])).astype(int) / 365.25
    put(f"switch {years[i - 1]}->{years[i]} ({near[i - 1]} -> {near[i]}): area km2 old / new; year-{years[i]} "
        "glacier-mean Ba m w.e. old / new mask; year volume change km3 old / new / difference; GLAMOS sigma per year of "
        f"its period {X['d0'][j]} to {X['d1'][j]}",
        f"{a0:.4f} {a1:.4f}; {ba[:nc][m0].mean():+.3f} {ba[:nc][m1].mean():+.3f}; {v0:+.5f} {v1:+.5f} {v1 - v0:+.5f}; "
        f"{X['sig_d'][list(X['idx_d']).index(j)] / yrs:.5f}")

dz = np.abs(np.diff(X["z"][:, :nc], axis=0)).max(1)
put("surface: largest |z(i) - z(i-1)| on the lattice per year m", " ".join(f"{y}:{d:.2f}" for y, d in zip(years[1:], dz)))

with open(RES / f"calib_cuts{SFX}.csv", "w", newline="") as f:
    csv.writer(f).writerows([("quantity", "value"), *out])
