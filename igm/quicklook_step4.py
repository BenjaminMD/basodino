"""Quick look at the start-state and flow fit (DESIGN.md step 4; preview, not a report figure): every sample in start
thickness factor u against flow factor f (grey; the weighted ones in colour, area by weight), the modelled ITS_LIVE
median surface speed 2014-2021 against f with its 1 m per year bound, and the modelled ice area at the four target
dates against u with the observed areas.
Inputs: results/step4_weights.csv, results/step4_runs_s4_r*.csv, results/step4_targets.npz (weights.py).
usage (Mac):  python quicklook_step4.py      output: results/quicklook_step4.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent / "results"
W = list(csv.DictReader(open(R / "step4_weights.csv")))
runs = {}
for f in sorted(R.glob("step4_runs_s4_r*.csv")):
    for r in csv.DictReader(open(f)):
        runs[r["name"]] = r
t = np.load(R / "step4_targets.npz")
u = np.array([float(r["u"]) for r in W])
f = np.array([float(r["f"]) for r in W])
w = np.array([float(r["w_main"]) for r in W])
speed = np.array([float(runs[r["name"] + "_91"]["speed"]) for r in W])
big = w > 1e-3 * w.max()

fig, axs = plt.subplots(1, 3, figsize=(16, 5))
ax = axs[0]
ax.scatter(u, f, s=4, c="0.75")
ax.scatter(u[big], f[big], s=10 + 400 * w[big] / w.max(), c="tab:blue", alpha=0.7)
ax.set_yscale("log")
ax.set_xlabel("start thickness factor u")
ax.set_ylabel("flow factor f")
ax.set_title(f"{len(W)} samples; weight above 1/1000 of the largest: {big.sum()}")
ax = axs[1]
ax.scatter(f, speed, s=4, c="0.75")
ax.scatter(f[big], speed[big], s=10 + 400 * w[big] / w.max(), c="tab:blue", alpha=0.7)
ax.axhline(float(t["speed_bound"]), color="tab:red", lw=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("flow factor f")
ax.set_ylabel("median surface speed 2014-2021, metres per year")
ax.set_title("Modelled speed on the 96 satellite pixels; red: the bound")
ax = axs[2]
for k, (d, kind, obs) in enumerate(zip(t["area_date"], t["area_kind"], t["area_obs"])):
    key = {"2021-09-30": "A2021", "2024-08-23": "A2024", "2025-09-07": "A2025", "2026-09-27": "A2026"}[str(d)]
    seg = "_91" if key == "A2021" else "_21"
    a = np.array([float(runs[r["name"] + seg][f"{key}_1.0"]) for r in W])
    c = f"C{k}"
    ax.scatter(u, a, s=3, c=c, alpha=0.3)
    ax.axhline(obs, color=c, lw=1, label=f"{d} ({'all ice' if kind == 'all' else 'main body'}), observed {obs:.2f}")
ax.set_xlabel("start thickness factor u")
ax.set_ylabel("ice area, square kilometres")
ax.set_title("Modelled area at the target dates (dots), observed (lines)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(R / "quicklook_step4.png", dpi=110)
print(R / "quicklook_step4.png")
