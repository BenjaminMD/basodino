"""Quick look at the dT_1850 scan of step 5 (preview, not a report figure): ice area (thk >= 1 m, mean of the last
30-year cycle) inside and outside the SGI1850 outline against dT_1850 for the sample of the largest weight, the
1850 target band, and the end-state thickness of six scan runs with the SGI1850 outline (black) and the cells where
decision 2 allows positive balance (grey line).
Inputs (s03, env.sh): results/step5_runs_s5_scan2.csv, runs/s5_scan2/scan_*/fields.npz, inputs/igm_static.npz.
usage (s03):  python quicklook_step5_scan.py      output: results/quicklook_step5_scan.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IGM = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(IGM / "results/step5_runs_s5_scan2.csv")))
off = -1.4 + 0.2 * np.arange(len(rows))
g = np.load(IGM / "inputs/igm_static.npz")
x, y = g["x"], g["y"]

fig = plt.figure(figsize=(17, 9))
ax = fig.add_subplot(2, 4, 1)
for k, lab in (("Ain_last", "inside SGI1850"), ("Aout_last", "outside SGI1850"), ("A_last", "total")):
    ax.plot(off, [float(r[k]) for r in rows], "o-", ms=3, label=lab)
ax.axhspan(4.008 - 0.25, 4.008 + 0.25, color="0.85", label="SGI1850 4.008 +- 0.25 km2")
ax.set_xlabel("dT_1850 (K)")
ax.set_ylabel("ice area, last 30-year cycle (km2)")
ax.legend(fontsize=7)
ax.set_title("scan, sample r2_0936")
for i, k in enumerate((5, 6, 7, 8, 9, 10)):
    ax = fig.add_subplot(2, 4, i + 2 + (i >= 3))
    thk = np.load(IGM / f"runs/s5_scan2/scan_{k:02d}/fields.npz")["thk_end"]
    im = ax.pcolormesh(x, y, np.where(thk >= 1, thk, np.nan), cmap="Blues", vmin=0, vmax=150, shading="auto")
    ax.contour(x, y, g["sgi1850"], [0.5], colors="k", linewidths=0.8)
    ax.contour(x, y, g["posbal"], [0.5], colors="0.5", linewidths=0.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    r = rows[k]
    ax.set_title(f"dT_1850 {off[k]:+.1f} K: inside {float(r['Ain_end']):.2f}, outside {float(r['Aout_end']):.2f} km2",
                 fontsize=9)
fig.colorbar(im, ax=fig.axes[1:], shrink=0.6, label="ice thickness 1850 (m)")
fig.savefig(IGM / "results/quicklook_step5_scan.png", dpi=110)
print("results/quicklook_step5_scan.png")
