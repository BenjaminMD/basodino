"""Check of calib_smb.simulate (step 2, DESIGN.md) after the winter term became the snow store at date_end_winter:
the TensorFlow float32 values of every target against an independent numpy float64 loop over smb.step, for 16 draws
spread over the first-round prior box (r_ice = 0.012 Fm; dT 0 and +1.5 degC), on the stride-2 lattice: draws 1-8 with
the gradients of decision 5 (dg = 0, b = 2 % per 100 m), draws 9-16 the same with dg and b varied over their priors
(third attempt, the transfer written out here: T = T_OTL + (g_m + dg) dz, P = P_OTL (1 + b dz)).
The numpy loop sums the daily balance of each piece over its days and, for the winter balances of the observation
periods (calib_smb.STORE), takes the column store at the end day of the year holding date1, written here without
the piece filtering of simulate().
Output: igm/results/calib_store_check.csv (kind, values, largest |TF - numpy| m w.e. or km3 for draws 1-8 and 9-16,
largest |numpy|).
usage (s03, env.sh, one GPU):  python checks/calib_store_check.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IGM))
import calib_smb as c  # noqa: E402
from smb import step  # noqa: E402

X = dict(np.load(IGM / "inputs/calib_stride2.npz"))
fm = np.tile([0.5, 1.5, 3, 5, 0.5, 1.5, 3, 5], 2) * 1e-3
cp = np.tile([0.8, 1.2, 1.6, 2.4, 1.0, 1.4, 2.0, 2.8], 2)
dt = np.tile([0, 0, 0, 0, 1.5, 1.5, 1.5, 1.5], 2)
dg = np.r_[np.zeros(8), np.array([-2.5, -1, 1, 2.5, 2, -2, 0.5, -0.5]) * 1e-3]
bp = np.r_[np.full(8, c.B_PREC), np.array([-1.5, 0, 4, 9, 6, 1, -1, 8]) * 1e-4]
theta = np.stack([fm, c.RATIO * fm, cp, dt, dg, bp], 1)
tf_vals = c.simulate(X, theta, 16)

kind, pc, G = X["kind"], X["pieces"], X["groups"].astype(np.float64)
nT, n = len(X["obs"]), len(theta)
store = np.isin(kind, c.STORE)
ref = np.zeros((n, nT))
D = theta[:, 2:3] * X["D"][None, :]
Fm, r = theta[:, 0:1], theta[:, 1:2]
last = {}
for j, g, y, a, e in pc:
    if j not in last or y > last[j][0]:
        last[j] = (y, g, e)
for i in range(len(X["years"])):
    dz = X["z"][i] - c.Z_OTL
    S = np.zeros((n, len(dz)))
    C, Sh = [np.zeros_like(S)], [np.zeros_like(S)]
    for k in range(365):
        T = X["t_otl"][i][k] + (X["gm"][k] + theta[:, 4:5]) * dz[None, :] + theta[:, 3:4]
        P = X["p_otl"][i][k] * (1 + theta[:, 5:6] * dz[None, :])
        S, b = step(S, T, P, X["ipot"][k], D, Fm, r, 0.75 * r)
        C.append(C[-1] + b)
        Sh.append(S)
    for j, g, y, a, e in pc[pc[:, 2] == i]:
        if not store[j]:
            ref[:, j] += (C[e] - C[a]) @ G[g]
    for j, (y, g, e) in last.items():
        if store[j] and y == i:
            ref[:, j] = Sh[e] @ G[g]

rows = []
for k in dict.fromkeys(kind):
    m = kind == k
    d = np.abs(tf_vals[:, m] - ref[:, m])
    rows.append((k, int(m.sum()), f"{d[:8].max():.2e}", f"{d[8:].max():.2e}", f"{np.abs(ref[:, m]).max():.4f}"))
    print(*rows[-1], sep="\t")
with open(IGM / "results/calib_store_check.csv", "w", newline="") as f:
    csv.writer(f).writerows([("kind", "values", "largest |TF float32 - numpy float64|, draws 1-8 (decision 5 gradients)",
                              "the same, draws 9-16 (dg, b varied)", "largest |numpy|"), *rows])
