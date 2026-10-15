"""Members without ice flow (decision 13, DESIGN.md): per-cell balance and thinning, the model of the IGM runs with the
flow taken out, all members of a file at once (member axis, TensorFlow on the GPU when one is visible).

Per hydrological year Y (1 October Y - 1 to 30 September Y), as clim_otl and smb_ti in IGM: on the surface z = topg +
thk of 1 October, T = t_OTL + (g_m + dg)(z - 367 m) (float64, cast to float32) + dT + offset, P = c_prec p_OTL (1 + b
(z - 367 m)); smb.step day by day (float32), C = sum of the balance, at most 0 where posbal = 0 (decision 2);
thk = max(thk + C / 0.91, 0). Areas at a date d inside year Y: max(thk(1 October) + C_n / 0.91, 0), C_n the balance
of the days before d (as track.py, without the flow term). Start state: geometry.start of the IGM input module (dem
1994 | 2021 | none, u); start ice outside the basin (posbal = 0, F2 of the fix round) set to 0, as the basin process
melts it in IGM's first step. Only the cells of posbal or with ice at the start are computed: no other cell can gain
ice without flow.
Stop rule (decision 14, as track.py stops IGM; fix round F6): a member stops at the first 1 October after the start
without a cell >= 1 m; that row is its last (the first ensemble's no-flow members ran to the end and 4340 regrew).
Members file (csv): Fm, r_ice (empty: 0.012 Fm), c_prec, dT, dg (K m-1), b (m-1), u, offset (empty: 0); one row
per member, the parameters of the IGM run it stands for. Forcing: inputs/forcing_obs.npz (observed weather).
Projections (DESIGN.md step 6): DEM restart with a column restart (npz of topg, thk_end: the member's start state), and
columns station, chain, ssp, level, seed: each member its own scenario weather (projection.weather, as clim_otl
with a scenario); then also OUT with _last_ice.npz for .csv: keep (the computed cells, y, x), last_ice (member, kept
cell) int16, the last hydrological year with ice (thk >= 1 m) at a 1 October, -1 never (as track.py).
Output (csv): member, date (YYYY-10-01 = end of hydrological year YYYY - 1 ... ; the start state first; then the
dates asked for), V_km3, A01_km2, A1_km2, A2_km2 (thk >= 0.1, 1, 2 m), Ba (m w.e., over the cells with >= 1 m at the
start of the year; empty for the start row and the dates).
usage (s03, env.sh):  python noflow.py MEMBERS DEM START_YEAR END_YEAR OUT [YYYY-MM-DD ...]
        START_YEAR, END_YEAR: hydrological years; the run goes from 1 October START_YEAR to 1 October END_YEAR
        (e.g. 2021 2026 = 2021-10-01 to 2026-10-01, the IGM run 2021.75-2026.75)
"""
import csv
import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import tensorflow as tf

import projection
from smb import ICE, Z_OTL, step, transfer

IGM = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("geometry", IGM / "user/code/inputs/geometry/geometry.py")
geometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geometry)
_spec = importlib.util.spec_from_file_location("track", IGM / "user/code/outputs/track/track.py")
track = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(track)
NSNAP, CHUNK = 6, 256


@tf.function(jit_compile=True)
def year(t, p, gm, dz, I, D, Fm, r_ice, r_snow, dT, dg, b, c_prec, snap):
    """One hydrological year for (member, cell): C, snapshots of C at the starts of the days snap."""
    C = tf.zeros(tf.shape(dz), tf.float32)
    S = tf.zeros_like(C)
    snaps = tf.zeros(tf.concat([[NSNAP], tf.shape(dz)], 0), tf.float32)
    for k in tf.range(365):
        T, P = transfer(t[k], p[k], gm[k], dz, dg, b)
        S, bk = step(S, tf.cast(T, tf.float32) + dT, tf.cast(c_prec * P, tf.float32), I[k], D, Fm, r_ice, r_snow,
                     tf)
        C = C + bk
        snaps = tf.where(tf.equal(snap, k + 1)[:, None, None], C[None], snaps)
    return C, snaps


def main():
    members_file, dem, y0, y1, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    dates = [date.fromisoformat(s) for s in sys.argv[6:]]
    g = dict(np.load(IGM / "inputs/igm_static.npz"))
    f = dict(np.load(IGM / "inputs/forcing_obs.npz"))
    index = {int(y): k for k, y in enumerate(f["hydro_year"])}
    M = list(csv.DictReader(open(members_file)))
    num = lambda r, k, d: float(r[k]) if r.get(k, "") != "" else d
    par = np.array([[num(r, "Fm", 0), num(r, "r_ice", 0.012 * num(r, "Fm", 0)), num(r, "c_prec", 1), num(r, "dT", 0),
                     num(r, "dg", 0), num(r, "b", 0), num(r, "offset", 0), num(r, "u", 0)] for r in M])
    cache = {}
    for r in M:
        if r.get("restart") and r["restart"] not in cache:
            cache[r["restart"]] = geometry.start(g, dem, 0.0, IGM / r["restart"])
    starts = [cache[r["restart"]] if r.get("restart") else geometry.start(g, dem, u) for r, u in zip(M, par[:, 7])]
    scen = bool(M[0].get("ssp"))
    if scen:
        F = dict(np.load(IGM / projection.SCEN))
        chains = list(F["chains"])
        ix = [np.array([L.index(r[k]) for r in M]) for k, L in (("station", projection.STATIONS), ("chain", chains),
                                                                ("ssp", projection.SSPS), ("level", projection.LEVELS))]
        order = np.array([projection.year_order(int(r["seed"])) for r in M])
    keep = (g["posbal"] == 1) | np.logical_or.reduce([th > 0 for _, th in starts])
    topg = np.array([tp[keep] for tp, _ in starts])
    thk = np.array([th[keep] for _, th in starts])
    out_basin = (thk > 0) & (g["posbal"][keep] != 1)[None]
    if out_basin.any():
        print(f"start ice outside the basin set to 0: {out_basin.sum()} member cells, "
              f"{thk[out_basin].sum() * 1e-7:.6f} km3 summed over the members")
    thk[out_basin] = 0.0
    I, D = tf.constant(g["ipot"][:, keep]), tf.constant(g["D"][keep])
    on = tf.constant(g["posbal"][keep] == 1)
    gm = tf.constant(f["gm"])
    a = 100.0 / 1e6                                          # km2 per cell
    snaps_of = {}
    for d in dates:
        y, n = track.day_index(d)
        snaps_of.setdefault(y, []).append((n, d))
    rows = []

    alive = np.ones(len(M), bool)                            # stop rule: rows only while a member runs

    def state_rows(th, when, ba=None):
        for m in np.flatnonzero(alive):
            rows.append([m, when, f"{th[m].sum() * a / 1e3:.6f}", *(f"{(th[m] >= h).sum() * a:.4f}" for h in
                                                                 (0.1, 1.0, 2.0)), "" if ba is None else f"{ba[m]:.5f}"])

    state_rows(thk, f"{y0}-10-01")
    last = np.where(thk >= 1.0, y0, -1).astype(np.int16)
    for Y in range(y0 + 1, y1 + 1):
        if scen:
            tY, pY = projection.weather(F, *ix, order, Y)          # (member, 365)
        else:
            k = index[Y]
        snap_days = [n for n, _ in snaps_of.get(Y, []) if n > 0]
        snap = tf.constant(snap_days + [-1] * (NSNAP - len(snap_days)), tf.int32)
        C = np.zeros_like(thk, np.float32)
        Cn = np.zeros((NSNAP,) + thk.shape, np.float32) if snap_days else None
        for c0 in range(0, len(M), CHUNK):
            s = slice(c0, c0 + CHUNK)
            p = par[s]
            col = lambda j, dt=tf.float32: tf.constant(p[:, j:j + 1], dt)
            dz = tf.constant(topg[s] + thk[s].astype(np.float64) - Z_OTL, tf.float64)
            t_, p_ = (tY[s].T[:, :, None], pY[s].T[:, :, None]) if scen else (f["t"][k], f["p"][k])
            c, cn = year(tf.constant(t_), tf.constant(p_), gm, dz, I, D, col(0), col(1),
                         col(1) * 0.75, col(3) + col(6), col(4, tf.float64), col(5, tf.float64), col(2, tf.float64),
                         snap)
            C[s] = tf.where(on, c, tf.minimum(c, 0.0)).numpy()
            if snap_days:
                Cn[:, s] = tf.where(on, cn, tf.minimum(cn, 0.0)).numpy()
        for n, d in snaps_of.get(Y, []):
            if n > 0:
                state_rows(np.maximum(thk + Cn[snap_days.index(n)] / ICE, 0.0), d.isoformat())
        ice = thk >= 1.0
        ba = np.array([C[m][ice[m]].mean() if ice[m].any() else np.nan for m in range(len(M))])
        thk = np.maximum(thk + C / ICE, 0.0).astype(np.float32)
        last = np.where(alive[:, None] & (thk >= 1.0), Y, last).astype(np.int16)
        state_rows(thk, f"{Y}-10-01", ba)
        for n, d in snaps_of.get(Y + 1, []):
            if n == 0:
                state_rows(thk, d.isoformat())
        alive &= (thk >= 1.0).any(1)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["member", "date", "V_km3", "A01_km2", "A1_km2", "A2_km2", "Ba"])
        w.writerows(rows)
    if scen:
        np.savez_compressed(out.replace(".csv", "_last_ice.npz"), keep=keep, last_ice=last)
    print(f"{len(M)} members, {keep.sum()} cells, hydrological years {y0 + 1}-{y1}, {(~alive).sum()} stopped (no ice): "
          f"{out}")


if __name__ == "__main__":
    main()
