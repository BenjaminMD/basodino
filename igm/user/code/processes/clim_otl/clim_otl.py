"""IGM process clim_otl: daily air temperature and precipitation of one hydrological year on every cell, from the
Locarno-Monti (OTL) day values with the calibrated station transfer (decision 5 with the step-2 parameters,
DESIGN.md; smb.transfer as in calib_smb.py):
  T = t_OTL + (g_m + dg)(z - 367 m) + dT [+ dT_pre] + offset   degC
  P = c_prec p_OTL (1 + b (z - 367 m))                      m w.e. d-1
z the current surface (usurf), g_m the monthly Robiei-Locarno gradient of the day, dT, dg, b, c_prec one posterior
draw of results/calib_draws.csv (dg in K m-1, b in m-1), offset an extra temperature change (dT_1850 of the
spin-up, decision 10; 0 otherwise); dT_pre (F1 of the fix round, results/calib_f1_draws.csv) is added in the model's
hydrological years up to PRE_END = 1991 (calib_smb.py PRE_END; in a spin-up every model year, so dT_1850 = offset acts
on top of it). The transfer runs in float64 and is cast to float32 before dT is added, as in
the calibration (there c_prec multiplies D instead of P: the same product to float32 rounding).
Timing (decision 3): the first update falls on the start (last update = start - 1); afterwards one update per
hydrological year: at t = Y.75 (1 October) hydrological year Y + 1 = smb.hydro_year(t) is loaded on the surface of
that moment. An update is due when hydro_year(t) differs from the year loaded last (for t on Y.75 within IGM's
float32 rounding this is t - t_last >= 1 of the decision; it cannot miss a year when the time step lands a hair
below Y.75). The year that begins at time.end is not loaded (F3, 2026-10-09: two of 2048 runs ended a hair below
2026.75 and asked for hydrological year 2027, which the observed forcing does not hold).
Forcing file (cfg file, npz): hydro_year, t (year, 365) degC, p (year, 365) m w.e. d-1, gm (365,) K m-1
(pack_inputs.py: forcing_obs.npz). cycle [y0, y1]: hydrological year Y of the run takes year y0 + (Y - Y_start) mod
(y1 - y0 + 1) of the file (spin-up, decision 10); null: year Y itself.
Scenario (decision 12, step 6): with scenario.file (inputs/forcing_scen.npz) the days of 2027-2100 are the member's
scenario weather (projection.member_forcing: station, chain, ssp, level, seed); gm stays the file's.
Sets state.air_temp, state.precipitation (365, y, x) float32 and state.clim_year.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))     # igm/: smb.py
from smb import Z_OTL, hydro_year, transfer  # noqa: E402

PRE_END = 1991                     # dT_pre acts on the hydrological years up to this one (calib_smb.py PRE_END)
from projection import member_forcing  # noqa: E402


def initialize(cfg, state):
    c = cfg.processes.clim_otl
    with np.load(Path(state.original_cwd) / c.file) as f:
        state.clim_index = {int(y): k for k, y in enumerate(f["hydro_year"])}
        state.clim_t, state.clim_p, state.clim_gm = f["t"], f["p"], f["gm"]
    if c.scenario.file:
        state.clim_index, state.clim_t, state.clim_p = member_forcing(Path(state.original_cwd) / c.scenario.file,
                                                                      c.scenario)
    state.clim_year = None
    state.clim_first = hydro_year(cfg.processes.time.start)
    state.tlast_clim_otl = tf.Variable(cfg.processes.time.start - 1.0)


def forcing_year(c, first, Y):
    """The file's hydrological year that forces model year Y."""
    if c.cycle is None:
        return Y
    y0, y1 = int(c.cycle[0]), int(c.cycle[1])
    return y0 + (Y - first) % (y1 - y0 + 1)


@tf.function(jit_compile=True)
def fields(t, p, gm, usurf, dg, b, dT, c_prec):
    """T, P (365, y, x) float32; one fused XLA kernel, so no float64 (365, y, x) array is ever stored."""
    dz = tf.cast(usurf, tf.float64)[None] - Z_OTL
    T, P = transfer(t[:, None, None], p[:, None, None], gm[:, None, None], dz, dg, b)
    return tf.cast(T, tf.float32) + dT, tf.cast(c_prec * P, tf.float32)       # dT after the cast, as calib


def update(cfg, state):
    Y = hydro_year(state.t)
    if Y == state.clim_year:
        return
    if state.clim_year is not None and Y >= hydro_year(cfg.processes.time.end):
        return          # a last step ending within hydro_year's 1e-3 a of time.end: the run ends, no year to load
    c = cfg.processes.clim_otl
    k = state.clim_index[forcing_year(c, state.clim_first, Y)]
    f64 = lambda v: tf.constant(v, tf.float64)
    state.air_temp, state.precipitation = fields(
        f64(state.clim_t[k]), f64(state.clim_p[k]), f64(state.clim_gm), state.usurf, f64(float(c.dg)),
        f64(float(c.b)), tf.constant(float(c.dT) + float(c.offset) + (float(c.get("dT_pre", 0.0)) if Y <= PRE_END
                                                                         else 0.0), tf.float32),
        f64(float(c.c_prec)))
    state.clim_year = Y
    state.tlast_clim_otl.assign(state.t)


def finalize(cfg, state):
    pass
