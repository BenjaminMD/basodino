"""IGM process smb_ti: the temperature-index balance of decision 3 (DESIGN.md), smb.step day by day on every cell,
once per hydrological year, from the daily fields of clim_otl (run it right after clim_otl).

  per day k = 0 (1 October) ... 364: snow store S and balance b of smb.step with Fm, r_ice (default 0.012 Fm, the
  ratio fixed in step 2), r_snow = 0.75 r_ice, accumulation factor D (redist.py); C = sum of b (m w.e.)
  decision 2: where posbal = 0 the balance is at most 0 (C and every snapshot of C)
  state.smb = C / 0.91 m ice a-1, applied as a rate through the year (thk process)
Timing as clim_otl (decision 3): the first update falls on the start, then one at each 1 October (Y.75), on the
surface of that moment; the +1.10 m placeholder of the reference never exists here (state.smb = 0 until the first
update, which happens before thk's first step).
Kept for the track output (state, per year): smb_ba = C, smb_snaps = C at the start of day 212 (1 May: the
fixed-date winter balance, as calib_smb.py bw_fixdate) and at the day indices state.track_snaps[Y] that track asks
for (start of the day: the balance of the days before it, the balance accumulated since 1 October of decision 3 for
the areas at the outline dates); smb_years keeps (C, snapshots, days) of this year and the one before, so track finds
year Y at its save even when a time step ended within the 1e-3 a tolerance of hydro_year before Y.75 and year Y + 1
was loaded one step early (run r1_0463_91 of step 4 lost its 2012 row that way).
Parameters per run (cfg): Fm m ice d-1 degC-1, r_ice m ice m2 W-1 d-1 degC-1 (null: 0.012 Fm), float32 on the GPU
(the calibration ran the balance in float32 too).
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))     # igm/: smb.py
from smb import ICE, hydro_year, step  # noqa: E402

BW_DAY, NSNAP, RATIO = 212, 6, 0.012     # start of 1 May; snapshots per year (1 May + up to 5 dates); r_ice / Fm


@tf.function(jit_compile=True)
def year(T, P, I, D, Fm, r_ice, r_snow, snap):
    """C (y, x) and its snapshots at the starts of the days snap (NSNAP, y, x)."""
    S = tf.zeros_like(D)
    C = tf.zeros_like(D)
    snaps = tf.zeros((NSNAP,) + tuple(D.shape), D.dtype)
    for k in tf.range(365):
        S, b = step(S, T[k], P[k], I[k], D, Fm, r_ice, r_snow, tf)
        C = C + b
        snaps = tf.where(tf.equal(snap, k + 1)[:, None, None], C[None], snaps)
    return C, snaps


def initialize(cfg, state):
    c = cfg.processes.smb_ti
    Fm = float(c.Fm)
    r_ice = RATIO * Fm if c.r_ice is None else float(c.r_ice)
    state.smb_par = tuple(tf.constant(v, tf.float32) for v in (Fm, r_ice, 0.75 * r_ice))
    state.smb = tf.zeros_like(state.thk)
    state.smb_year = None
    state.tlast_smb = tf.Variable(cfg.processes.time.start - 1.0)


def update(cfg, state):
    Y = hydro_year(state.t)
    if Y == state.smb_year:
        return
    if state.smb_year is not None and Y >= hydro_year(cfg.processes.time.end):
        return          # as clim_otl: the year that begins at time.end is not computed
    assert state.clim_year == Y, (state.clim_year, Y)
    days = [BW_DAY] + list(getattr(state, "track_snaps", {}).get(Y, []))
    assert len(days) <= NSNAP, days
    snap = tf.constant(days + [-1] * (NSNAP - len(days)), tf.int32)
    C, snaps = year(state.air_temp, state.precipitation, state.ipot, state.D, *state.smb_par, snap)
    on = state.posbal > 0.5
    C = tf.where(on, C, tf.minimum(C, 0.0))
    state.smb_snaps = tf.where(on[None], snaps, tf.minimum(snaps, 0.0))
    state.smb_snap_days = days
    state.smb_ba = C
    state.smb_years = {y: v for y, v in getattr(state, "smb_years", {}).items() if y == Y - 1}
    state.smb_years[Y] = (C, state.smb_snaps, days)
    state.smb = C / ICE
    state.smb_year = Y
    state.tlast_smb.assign(state.t)


def finalize(cfg, state):
    pass
