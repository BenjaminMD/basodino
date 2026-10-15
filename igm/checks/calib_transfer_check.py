"""Check of the third step-2 attempt (DESIGN.md, step 2): calib_smb.simulate with the station gradients as parameters
(smb.transfer on the GPU in float64) reproduces the previous simulate (host transfer of decision 5, git 413df36) when
dg = 0 and b = B_PREC (2 % per 100 m), value for value; also the same draws with dT, and the theta layout of the
previous attempts (4 columns, missing dg and b at their defaults).
Draws: the 8 of checks/calib_store_check.py (r_ice = 0.012 Fm), stride 2.
Input: the previous module as exported from git (Mac):  git show 413df36:basodino/igm/calib_smb.py > previous copy,
rsynced to s03 tmp/calib_smb_413df36.py (tmp/ is emptied after the stage).
Output: igm/results/calib_transfer_check.csv (kind, values, largest |current - previous| for 6-column and 4-column
theta, largest |previous|).
usage (s03, env.sh, one GPU):  python checks/calib_transfer_check.py tmp/calib_smb_413df36.py
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IGM))
import calib_smb as c  # noqa: E402

spec = importlib.util.spec_from_file_location("calib_smb_previous", IGM / sys.argv[1])
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

X = dict(np.load(IGM / "inputs/calib_stride2.npz"))
fm = np.array([0.5, 1.5, 3, 5, 0.5, 1.5, 3, 5]) * 1e-3
cp = np.array([0.8, 1.2, 1.6, 2.4, 1.0, 1.4, 2.0, 2.8])
dt = np.array([0, 0, 0, 0, 1.5, 1.5, 1.5, 1.5])
th4 = np.stack([fm, c.RATIO * fm, cp, dt], 1)
th6 = np.concatenate([th4, np.zeros((8, 1)), np.full((8, 1), c.B_PREC)], 1)
prev = old.simulate(X, th4, 8)
new6, new4 = c.simulate(X, th6, 8), c.simulate(X, th4, 8)
rows = []
for k in dict.fromkeys(X["kind"]):
    m = X["kind"] == k
    rows.append((k, int(m.sum()), f"{np.abs(new6[:, m] - prev[:, m]).max():.3e}",
                 f"{np.abs(new4[:, m] - prev[:, m]).max():.3e}", f"{np.abs(prev[:, m]).max():.4f}"))
    print(*rows[-1], sep="\t")
rows.append(("all", len(X["kind"]), f"{np.abs(new6 - prev).max():.3e}", f"{np.abs(new4 - prev).max():.3e}",
             f"{np.abs(prev).max():.4f}"))
print(*rows[-1], sep="\t")
with open(IGM / "results/calib_transfer_check.csv", "w", newline="") as f:
    csv.writer(f).writerows([("kind", "values", "largest |current - previous|, theta with dg = 0, b = B_PREC",
                              "the same, 4-column theta", "largest |previous|"), *rows])
