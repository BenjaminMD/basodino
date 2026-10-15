"""Diagnostic for step 2 (DESIGN.md): which likelihood term sets the melt factor. From one stored round of
calib_smb.py (Sobol draws over the round's box, marginal log-likelihood per term a, b, c, d with the model-error SD
integrated out), the melt factor Fm is cut into 20 equal bins of the box; per bin the draw with the largest total
log-likelihood (the profile over the other free parameters) is listed with its c_prec, dT and each term's
log-likelihood minus that term's value at the overall best draw, and, per term, the bin where that term alone is
largest. No new model runs.
Output: igm/results/calib_term_profile_{tag}.csv.
usage (s03 or s02):  python checks/calib_term_profile.py results/calib_2par/round1.npz 2par
"""
import csv
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
z = np.load(IGM / sys.argv[1])
theta, L = z["theta"], z["loglik"]
tot = L.sum(1)
best = int(np.argmax(tot))
lo, hi = theta[:, 0].min(), theta[:, 0].max()
edges = np.linspace(lo, hi, 21)
b = np.clip(np.digitize(theta[:, 0], edges) - 1, 0, 19)
rows = [("Fm bin centre m ice d-1 degC-1", "draws", "c_prec", "dT degC",
         "total minus best", "a minus best", "b minus best", "c minus best", "d minus best")]
for i in range(20):
    m = np.nonzero(b == i)[0]
    j = m[np.argmax(tot[m])]
    rows.append((f"{(edges[i] + edges[i + 1]) / 2:.5f}", len(m), f"{theta[j, 2]:.3f}", f"{theta[j, 3]:.2f}",
                 f"{tot[j] - tot[best]:.1f}", *(f"{L[j, k] - L[best, k]:.1f}" for k in range(4))))
rows.append(("best draw Fm, c_prec, dT", "", f"{theta[best, 0]:.5f} {theta[best, 2]:.3f} {theta[best, 3]:.2f}"))
for k, name in enumerate("abcd"):
    j = int(np.argmax(L[:, k]))
    rows.append((f"term {name} alone largest at Fm, c_prec, dT", "", f"{theta[j, 0]:.5f} {theta[j, 2]:.3f} {theta[j, 3]:.2f}"))
with open(IGM / f"results/calib_term_profile_{sys.argv[2]}.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
for r in rows:
    print(*r, sep="\t")
