"""Quick look at the hindcast 1850-2026 of step 5 (preview, not a report figure): the 128 draws (median line, 5-95 %
band) against the observations of the validation table: ice area (all ice and inside SGI1850) with the inventory
areas, volume change since 1907 with the GLAMOS survey pairs chained, tongue elevation with GLAMOS, and the length
along the flowline with the GLAMOS length change since 1899 added to the model's 1899 length.
Inputs: results/P_series_HIND.csv, results/P_targets.csv, data/glamos_basodino_lengthchange.csv; P, HIND = step5,
s5_hind (first ensemble) or f5, f5_hind (fix round: the first ensemble's median drawn grey, dashed, for comparison).
usage (Mac):  python quicklook_step5.py [f5]      output: results/quicklook_step5.png | quicklook_f5.png
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IGM = Path(__file__).resolve().parent
R = IGM / "results"
P, HIND = ("f5", "f5_hind") if sys.argv[1:] == ["f5"] else ("step5", "s5_hind")


def load(path):
    out = defaultdict(list)
    for r in csv.DictReader(open(path)):
        out[r["name"]].append(r)
    return out


S = load(R / f"{P}_series_{HIND}.csv")
S0 = load(R / "step5_series_s5_hind.csv") if P == "f5" else None
years = np.array([int(r["year"]) for r in next(iter(S.values()))])


def col(c, S=S):
    return np.array([[float(r[c]) if r[c] not in ("", "nan") else np.nan for r in rs] for rs in S.values()])


def band(ax, z, label, color):
    ax.fill_between(years, *np.nanpercentile(z, [5, 95], axis=0), color=color, alpha=0.25, lw=0)
    ax.plot(years, np.nanmedian(z, axis=0), color=color, lw=1.2, label=label)


def first(ax, c, ref=None):
    """The first ensemble's median of column c (minus its value in year ref), grey dashed."""
    if S0 is None:
        return
    z = col(c, S0)
    if ref is not None:
        z = z - z[:, [list(years).index(ref)]]
    ax.plot(years, np.nanmedian(z, axis=0), color="0.4", ls="--", lw=1, label="first ensemble, median")


T = list(csv.DictReader(open(R / f"{P}_targets.csv")))
fig, axs = plt.subplots(2, 2, figsize=(14, 9))
ax = axs[0, 0]
band(ax, col("A1_km2"), "model, all ice", "C0")
band(ax, col("A1_in1850_km2"), "model, inside SGI1850", "C2")
first(ax, "A1_km2")
for t in T:
    if t["kind"] == "area":
        ax.errorbar(int(t["date_end"][:4]), float(t["observed"]), float(t["sigma"]), fmt="ko", ms=4)
ax.errorbar(1850, 4.008, 0.25, fmt="ks", ms=4, label="inventories (SGI1850 square)")
ax.set_ylabel("ice area (km2)")
ax.legend(fontsize=8)
ax = axs[0, 1]
V = col("V_km3")
band(ax, V - V[:, [list(years).index(1907)]], "model, from 1907-10-01", "C0")
first(ax, "V_km3", 1907)
cum, pts = 0.0, [(1907, 0.0)]
for t in T:
    if t["kind"] == "volume":
        cum += float(t["observed"])
        pts.append((int(t["date_end"][:4]), cum))
ax.plot(*zip(*pts), "ko-", ms=4, label="GLAMOS survey pairs, chained")
ax.set_xlim(1900, 2026)
ax.set_ylabel("volume change since 1907 (km3)")
ax.legend(fontsize=8)
ax = axs[1, 0]
band(ax, col("tongue_m"), "model, lowest ice of the largest body", "C0")
first(ax, "tongue_m")
tg = [(int(t["date_end"][:4]), float(t["observed"])) for t in T if t["kind"] == "tongue"]
ax.plot(*zip(*tg), "ko", ms=4, label="GLAMOS tongue elevation")
ax.axhline(2069, color="0.5", ls=":", label="lowest point of SGI1850")
ax.set_ylabel("tongue elevation (m)")
ax.legend(fontsize=8)
ax = axs[1, 1]
L = col("length_m")
band(ax, L, "model, along the flowline", "C0")
first(ax, "length_m")
L99 = np.nanmedian(L[:, list(years).index(1899)])
obs, cum = [], 0.0
for r in csv.reader(open(IGM.parent / "data/glamos_basodino_lengthchange.csv")):
    if len(r) > 7 and r[1] == "C14-10":
        cum += float(r[6])
        obs.append((int(r[4][:4]), L99 + cum))
ax.plot(*zip(*obs), "k.", ms=4, label="GLAMOS length change since 1899, from the model median in 1899")
ax.axhline(3037, color="0.5", ls=":", label="end of the flowline (lowest point of SGI1850)")
ax.set_ylabel("length (m)")
ax.legend(fontsize=8)
for ax in axs.flat:
    ax.set_xlabel("year")
fig.suptitle(f"Hindcast 1850-2026, {len(S)} draws (median, 5-95 %)")
fig.tight_layout()
out = "quicklook_step5.png" if P == "step5" else "quicklook_f5.png"
fig.savefig(R / out, dpi=110)
print(f"results/{out}")
