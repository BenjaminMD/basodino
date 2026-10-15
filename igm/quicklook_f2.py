"""Quick look at F2 of the fix round (preview, not a report figure): basin confinement in the spin-up of the F1 MAP
draw (u = 0, f = 1). (a) cycle-mean ice area inside and outside SGI1850 against the 1850 temperature offset (scan and
secant runs), the 1850 target band; (b) ice thickness at the end of the accepted secant run and (c) of the coldest
scan run, with the SGI1850 outline (black) and the basin edge (grey: the cells where ice may stay); (d) the yearly ice
volume melted outside the basin in both runs.
Inputs (s03, env.sh): results/f2_spinup.csv (rsynced from the Mac), runs/f2_*/*/fields.npz and series.csv,
inputs/igm_static.npz.
usage (s03):  python quicklook_f2.py      output: results/quicklook_f2.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IGM = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(IGM / "results/f2_spinup.csv")))
g = np.load(IGM / "inputs/igm_static.npz")
x, y = g["x"], g["y"]
acc = next(r for r in rows if r["accepted"] == "True")
cold = min(rows, key=lambda r: float(r["dT_1850"]))

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(2, 2, 1)
scan = sorted((r for r in rows if r["stage"] == "f2_scan"), key=lambda r: float(r["dT_1850"]))
off = [float(r["dT_1850"]) for r in scan]
for k, lab in (("Ain_last", "inside SGI1850"), ("Aout_last", "outside SGI1850, inside the basin"), ("A_last", "total")):
    ax.plot(off, [float(r[k]) for r in scan], "o-", ms=3, label=lab)
ax.plot(float(acc["dT_1850"]), float(acc["Ain_last"]), "k*", ms=12, label="accepted secant run")
ax.axhspan(4.008 - 0.25, 4.008 + 0.25, color="0.85", label="SGI1850 area 4.008 +- 0.25 km2")
ax.set_xlabel("temperature offset 1850, on top of the pre-1991 offset (K)")
ax.set_ylabel("ice area, mean of the last 30-year cycle (km2)")
ax.legend(fontsize=8)
ax.set_title("Spin-up scan, F1 maximum a posteriori balance")
for i, (r, name) in enumerate(((acc, "accepted"), (cold, "coldest scan"))):
    ax = fig.add_subplot(2, 2, 2 + i)
    thk = np.load(IGM / f"runs/{r['stage']}/{r['name']}/fields.npz")["thk_end"]
    im = ax.pcolormesh(x, y, np.where(thk >= 1, thk, np.nan), cmap="Blues", vmin=0, vmax=150, shading="auto")
    ax.contour(x, y, g["sgi1850"], [0.5], colors="k", linewidths=0.8)
    ax.contour(x, y, g["posbal"], [0.5], colors="0.5", linewidths=0.6)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{name} run, offset {float(r['dT_1850']):+.2f} K: inside {float(r['Ain_last']):.2f}, "
                 f"outside SGI1850 {float(r['Aout_last']):.2f} km2", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="ice thickness, end of spin-up (m)")
ax = fig.add_subplot(2, 2, 4)
for r, name in ((acc, "accepted"), (cold, "coldest scan")):
    s = list(csv.DictReader(open(IGM / f"runs/{r['stage']}/{r['name']}/series.csv")))[1:]
    ax.plot([int(v["year"]) for v in s], [float(v["basin_melt_km3"]) * 1e3 for v in s], lw=0.8, label=name)
ax.set_xlabel("model year")
ax.set_ylabel("ice melted outside the basin (10^-3 km3 per year)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(IGM / "results/quicklook_f2.png", dpi=110)
print("results/quicklook_f2.png")
