"""IGM output track: what the ensemble keeps of a run, written at IGM's save hook (state.saveresult; experiments
use time.save = 1 from a start on Y.75, so every save falls on 1 October).

At each save, t = Y.75, the end of hydrological year Y (the first save, iteration 0, is the start state):
  series.csv  one row per save: year Y, t, iteration, wall s since the process started; volume km3 and area km2
              at thk >= 0.1, 1, 2 m (decision 14: ice = thk >= 1 m); glacier-wide Ba and fixed-date Bw (1 October to
              30 April) and Bs = Ba - Bw, m w.e., over the cells with ice at the start of year Y (smb_ti; empty on
              the start row); tongue = lowest surface of the ice cells of the largest 4-connected ice body, m;
              surface speed over the ice cells: median, p90 and maximum, m a-1, and the median over the 96 ITS_LIVE
              pixels of their cells' mean speed (decision 9; 0 on ice-free cells); ice area (thk >= 1 m) inside and
              outside the SGI1850 outline, km2 (decision 10); length_m, the distance along the flowline of cfg
              flowline (npz: line_cell, flat cell index of each 10 m sample; line_s, its distance from the top, m) of
              the farthest sample whose cell is an ice cell of the largest body (empty without flowline or ice);
              basin_melt_km3, the ice volume the basin process (F2) removed outside the basin since the last save,
              km3 ice (empty without basin)
  stop        decision 14: a run stops at the first save after the start without ice (no cell >= 1 m); not with
              stop_no_ice false (spin-ups from ice-free, step 5: one year of snow can stay below 1 m)
  fields.npz  written when the run ends (compressed): topg, thk_end, last_ice (int16, the last hydrological year a cell held
              ice at a save, -1 never; ice-free year = last_ice + 1), and thk_YYYYMMDD at every date of cfg dates
At a date d inside hydrological year Y (decision 3): thk_d = max(thk(Y - 1 save) + (n / 365) F + C_n / 0.91, 0),
n the 365-day index of d (start of the day; 29 February read as 28 February), C_n the balance accumulated since
1 October (smb_ti snapshot, decision 2 applied) and F the year's flow thickness change, the sum of -divflux dt over
its steps minus the ice the basin process removed (F2), spread evenly through the year. A date on 1 October is the
save itself.
Optional: save_net (path relative to the start directory): the iceflow network weights after iteration 0 (cold
training on the start geometry; the input normaliser left out), npz read by the warmstart process (decision 1);
yearly: true keeps thk at every save as thk_yYYYY in fields.npz (checks), and the surface velocity as us_yYYYY, vs_yYYYY;
every: N > 0 keeps thk_yYYYY at the saves of the years divisible by N (step 5: 10); topg: false leaves topg out of
fields.npz (step 6: the start file holds it).
"""
import csv
import sys
import time
from calendar import isleap
from datetime import date
from pathlib import Path

import numpy as np
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))     # igm/: smb.py
from smb import ICE, hydro_year  # noqa: E402

FIELDS = ("year", "t", "it", "wall_s", "V_km3", "A01_km2", "A1_km2", "A2_km2", "Ba", "Bw", "Bs",
          "tongue_m", "v_med", "v_p90", "v_max", "its_median", "A1_in1850_km2", "A1_out1850_km2", "length_m",
          "basin_melt_km3")


def day_index(d):
    """(hydrological year, 365-day index of the start of d from 1 October), as the rows of weather.py."""
    y = d.year + (d.month >= 10)
    if (d.month, d.day) == (2, 29):
        d = date(d.year, 2, 28)
    n = (d - date(y - 1, 10, 1)).days
    if isleap(y) and d.year == y and d.month > 2:                 # 29 February of y lies before d
        n -= 1
    return y, n


def initialize(cfg, state):
    c = cfg.outputs.track
    k = {}
    k["dates"] = [date.fromisoformat(str(s)) for s in c.dates]
    snaps = {}
    for d in k["dates"]:
        y, n = day_index(d)
        if n > 0:
            snaps.setdefault(y, []).append(n)
    state.track_snaps = {y: sorted(set(v)) for y, v in snaps.items()}
    with np.load(Path(state.original_cwd) / cfg.inputs.geometry.static) as g:
        pix = g["itslive_pixel"].astype(int)
        k["in1850"] = g["sgi1850"] > 0
    k["pix"] = np.where(pix >= 0, pix, 96).ravel()
    k["pix_n"] = np.bincount(k["pix"], minlength=97)[:96]
    k["save_net"], k["yearly"], k["every"] = str(c.save_net), bool(c.yearly), int(c.get("every", 0))
    k["stop"] = bool(c.get("stop_no_ice", True))
    k["topg"] = bool(c.get("topg", True))
    k["line"] = None
    if c.get("flowline", ""):
        with np.load(Path(state.original_cwd) / c.flowline) as f:
            k["line"] = (f["line_cell"].astype(int), f["line_s"].astype(float))
    k["fields"], k["F"], k["it_prev"] = {}, 0.0, 0
    k["file"] = open("series.csv", "w", newline="")
    k["csv"] = csv.writer(k["file"])
    k["csv"].writerow(FIELDS)
    k["t0"] = state.start_time.timestamp() if hasattr(state, "start_time") else time.time()
    state.track = k


def speeds(state, ice):
    if not hasattr(state, "uvelsurf") or not ice.any():
        return [np.nan] * 4
    v = np.hypot(state.uvelsurf.numpy(), state.vvelsurf.numpy())
    vi = v[ice]
    v = np.where(ice, v, 0.0).ravel()
    s = np.bincount(state.track["pix"], weights=v, minlength=97)[:96]
    return [np.median(vi), np.percentile(vi, 90), vi.max(), np.median(s / state.track["pix_n"])]


def write_fields(state, reason):
    k = state.track
    topg = {"topg": state.topg.numpy()} if k["topg"] else {}
    np.savez_compressed("fields.npz", thk_end=state.thk.numpy(), last_ice=k["last_ice"], **topg, **k["fields"])
    with open("track_end.txt", "w") as f:
        f.write(f"{reason} t={float(state.t):.4f} wall_s={time.time() - k['t0']:.1f}\n")
    k["file"].close()


def run(cfg, state):
    k = state.track
    if state.it == 0 and k["save_net"]:
        norm = {id(v) for v in getattr(state.iceflow_model.input_normalizer, "weights", [])}
        vs = [v for v in state.iceflow_model.weights if id(v) not in norm]     # as warmstart.network_weights
        path = Path(state.original_cwd) / k["save_net"]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, names=np.array([v.path for v in vs]), **{f"w{i}": v.numpy() for i, v in enumerate(vs)})
    if hasattr(state, "divflux") and state.it > 0:
        k["F"] = k["F"] - float(state.dt) * state.divflux.numpy()
        if hasattr(state, "basin_removed"):                      # F2: ice that flowed out of the basin and melted
            k["F"] = k["F"] - state.basin_removed.numpy()
    if not state.saveresult:
        return
    t = float(state.t)
    Y = hydro_year(t) - 1
    thk = state.thk.numpy()
    ice = thk >= 1.0
    a = float(state.dx) ** 2 / 1e6
    row = [Y, f"{t:.4f}", state.it, f"{time.time() - k['t0']:.1f}", f"{thk.sum() * a / 1e3:.6f}",
           *(f"{(thk >= h).sum() * a:.4f}" for h in (0.1, 1.0, 2.0))]
    year = getattr(state, "smb_years", {}).get(Y)      # (C, snapshots, their days) of year Y (smb_ti)
    if state.it > 0 and year is not None and k["ice_prev"].any():
        m = k["ice_prev"]
        ba = float(year[0].numpy()[m].mean())
        bw = float(year[1][0].numpy()[m].mean())
        row += [f"{ba:.5f}", f"{bw:.5f}", f"{ba - bw:.5f}"]
        for d in k["dates"]:
            y, n = day_index(d)
            if y == Y and n > 0:
                C = year[1][year[2].index(n)].numpy()
                k["fields"][f"thk_{d:%Y%m%d}"] = np.maximum(k["thk_prev"] + n / 365 * k["F"] + C / ICE, 0.0)
    else:
        row += [""] * 3
    if k["yearly"] or (k["every"] > 0 and Y % k["every"] == 0):
        k["fields"][f"thk_y{Y}"] = thk.copy()
    if k["yearly"]:
        if hasattr(state, "uvelsurf"):                           # surface velocity, m a-1 (east, north)
            k["fields"][f"us_y{Y}"] = state.uvelsurf.numpy().astype(np.float32)
            k["fields"][f"vs_y{Y}"] = state.vvelsurf.numpy().astype(np.float32)
    for d in k["dates"]:
        if day_index(d) == (Y + 1, 0):
            k["fields"][f"thk_{d:%Y%m%d}"] = thk.copy()
    if ice.any():
        lab, _ = label(ice)
        big = lab == np.argmax(np.bincount(lab[ice]))
        tongue = float(state.usurf.numpy()[big].min())
    else:
        tongue, big = np.nan, ice
    length = ""
    if k["line"] is not None:
        on = big.ravel()[k["line"][0]]
        length = f"{k['line'][1][on].max():.1f}" if on.any() else ""
    row += [f"{tongue:.1f}", *(f"{v:.3f}" for v in speeds(state, ice)), f"{(ice & k['in1850']).sum() * a:.4f}",
            f"{(ice & ~k['in1850']).sum() * a:.4f}", length,
            f"{state.basin_melt * 1e-9:.6f}" if hasattr(state, "basin_melt") else ""]
    if hasattr(state, "basin_melt"):
        state.basin_melt = 0.0
    k["csv"].writerow(row)
    k["file"].flush()
    k["last_ice"] = np.where(ice, Y, k.get("last_ice", np.full(thk.shape, -1, np.int16))).astype(np.int16)
    k["thk_prev"], k["ice_prev"], k["F"] = thk, ice, 0.0
    if state.it > 0 and not ice.any() and k["stop"]:
        state.continue_run = False
        write_fields(state, "no ice")
    elif not state.continue_run:
        write_fields(state, "end time")
