"""Quick look at F3 of the fix round (preview, not a report figure): (a) the samples in (u, f), size and colour by
weight; (b) modelled ice area against u (colour: f) with the observed 2026 main body; (c) observed areas 2024, 2025,
2026 against the posterior 5-95 % range and median, and the prior range; (d) the modelled 2014-2021 surface speed
(median over the ITS_LIVE pixels) against f with the upper bound.
Inputs (Mac): results/f3_weights.csv, f3_runs_f3_r*.csv, f3_samples_r*.csv (fit_f3.load), step4_targets.npz.
usage (Mac):  python quicklook_f3.py      output: results/quicklook_f3.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fit_f3 import AREAS, RES, load, read_csv, wq

W = read_csv(RES / "f3_weights.csv")
rounds = sorted({int(r["round"]) for r in W})
S, R = load(rounds)
w = np.array([float(r["w_main"]) for r in W])
assert [r["name"] for r in W] == [s["name"] for s in S]
u = np.array([float(s["u"]) for s in S])
f = np.array([float(s["f"]) for s in S])
t = dict(np.load(RES / "step4_targets.npz"))
A = {d: np.array([float(r[f"A{d}_1.0"]) for r in R]) for d, _, _ in AREAS}
sp = np.array([float(r["speed"]) for r in R])
top = np.argsort(w)

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
a = ax[0, 0]
a.scatter(u, f, s=3, c="0.75")
sc = a.scatter(u[top], f[top], s=2 + 400 * w[top] / w.max(), c=w[top], cmap="viridis")
a.set_yscale("log")
a.set_xlabel("start thickness parameter u")
a.set_ylabel("flow factor f")
a.set_title(f"Samples (grey) and weights, {len(S)} samples, rounds {rounds}")
fig.colorbar(sc, ax=a, label="normalised weight")
a = ax[0, 1]
sc = a.scatter(u, A["2026"], s=4, c=np.log10(f), cmap="coolwarm")
a.axhspan(t["area_obs"][3] - t["area_sigma"][3], t["area_obs"][3] + t["area_sigma"][3], color="0.85", zorder=0)
a.axhline(t["area_obs"][3], color="k", lw=0.8)
a.set_xlabel("start thickness parameter u")
a.set_ylabel("main body ice area 2026-09-27 (km2)")
a.set_title("Modelled 2026 main body; observed with one sigma (grey)")
fig.colorbar(sc, ax=a, label="log10 flow factor")
a = ax[1, 0]
for k, (d, _, _) in enumerate(AREAS):
    q = wq(A[d], w, [0.05, 0.5, 0.95])
    p = np.percentile(A[d], [5, 95])
    a.plot([k - 0.1] * 2, p, color="0.6", lw=6, solid_capstyle="butt",
           label="all samples (rounds pooled, unweighted) 5-95 %" if k == 0 else None)
    a.plot([k + 0.1] * 2, [q[0], q[2]], color="C0", lw=6, solid_capstyle="butt",
           label="posterior 5-95 %" if k == 0 else None)
    a.plot(k + 0.1, q[1], "wo", ms=4)
    a.errorbar(k + 0.3, t["area_obs"][k + 1], t["area_sigma"][k + 1], fmt="ks", ms=5,
               label="observed, one sigma" if k == 0 else None)
a.set_xticks(range(3), ["2024-08-23\nall polygons", "2025-09-07\nmain body", "2026-09-27\nmain body"])
a.set_ylabel("ice area (km2)")
a.legend(fontsize=8)
a.set_title("Areas: observed against the fit")
a = ax[1, 1]
a.scatter(f, sp, s=3, c="0.75")
a.scatter(f[top], sp[top], s=2 + 400 * w[top] / w.max(), c=w[top], cmap="viridis")
a.axhline(float(t["speed_bound"]), color="k", lw=0.8)
a.set_xscale("log")
a.set_yscale("log")
a.set_xlabel("flow factor f")
a.set_ylabel("surface speed 2014-2021, median over the ITS_LIVE pixels (m per year)")
a.set_title("Speed against the ITS_LIVE upper bound (line)")
fig.tight_layout()
fig.savefig(RES / "quicklook_f3.png", dpi=110)
print("results/quicklook_f3.png")
