"""Quick look at the thickness-map test: the last year each 10 m cell held ice (code/last_ice.tif, Sentinel-2
2018-2026), the stake sites (code/stake_clock.csv), and the area the map predicts against the observed area
(code/area_clock.csv, k = 0.856, no melt spread, no taper). Preview only, not a report figure.
Output: tex/preview/mass_test_quicklook.png.  usage: uv run --with rasterio --with matplotlib python code/mass_test_quicklook.py
"""
import csv

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from common import ROOT

r = rasterio.open(ROOT / "code/last_ice.tif")
a = r.read(1).astype(float)
b = r.bounds
vals = np.unique(a)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
m = np.where((a >= 2018) & (a <= 2025), a, np.nan)
im = ax.imshow(m, extent=(b.left, b.right, b.bottom, b.top), cmap="plasma", vmin=2018, vmax=2025)
still = np.where(a > 2025, 1.0, np.nan)
ax.imshow(still, extent=(b.left, b.right, b.bottom, b.top), cmap="Blues", vmin=0, vmax=1.3, alpha=0.8)
plt.colorbar(im, ax=ax, label="Last year with ice")
for row in csv.DictReader(open(ROOT / "code/stake_clock.csv")):
    e, n, x = float(row["easting"]), float(row["northing"]), float(row["excess_m"])
    ax.plot(e, n, "o", ms=7, mfc="red" if x > 1 else "white", mec="k")
    ax.annotate(f"{row['series']} {x:+.1f} m", (e, n), xytext=(4, 4), textcoords="offset points", fontsize=7)
ax.set_title("Year each cell lost its ice (blue: still ice in 2026); stakes: melt minus mapped thickness")
ax.set_xlim(2678900, 2681600), ax.set_ylim(1140300, 1141600)
pred = {}
for row in csv.DictReader(open(ROOT / "code/area_clock.csv")):
    if row["quantity"] == "area_predicted" and row["k"].startswith("0.85") and row["s"] == "0" and row["taper"] == "none":
        pred.setdefault(float(row["u"]), []).append((row["date"], float(row["value"])))
obs = {"2021": 1.578, "2024": 1.380, "2025": 1.237, "2026": 1.025}
for u in (0, 0.5, 1.0):
    if u in pred:
        d = sorted(pred[u])
        bx.plot([int(x[:4]) + 0.7 for x, _ in d], [v for _, v in d], "-o", label=f"map, u = {u}")
bx.errorbar([int(y) + 0.7 for y in obs], list(obs.values()), yerr=0.09, fmt="ks", label="observed")
bx.set_ylabel("Ice area (km2)"), bx.legend(), bx.set_title("Area the map predicts vs observed")
fig.tight_layout()
fig.savefig(ROOT / "tex/preview/mass_test_quicklook.png", dpi=130)
print(vals)
