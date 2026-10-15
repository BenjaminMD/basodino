"""Gate of decision 3 timing (DESIGN.md step 1): the first model year runs on the real balance, never on the
reference's +1.10 m placeholder.

IGM's own loop (update_modules) with its time and thk processes runs two model years with ice flow off
(ubar = vbar = 0, so thk = max(thk + dt smb, 0)) on the Aletsch example (225 x 132 cells of 100 m, input.nc), twice:
  decision 3  a stand-in for clim_otl and smb_ti built from smb.py only: t_last = start - 1; an update when
              t - t_last >= 1; at t = Y.75 the hydrological year Y + 1 = smb.hydro_year(t), 1 October Y to
              30 September Y + 1 of temp_prec.dat moved to the current surface with clim_aletsch's lapse rates;
              smb.balance with the Aletsch inputs of checks/smb_reference.py; rate sum / 0.91 m ice a-1. Start 1880.75.
  reference   clim_aletsch and smb_accmelt unchanged (t_last = start; smb = 1000/910 until the first update).
              Start 1880.0.
A recorder in the output slot keeps t, the last update time, smb and thk after every iteration.
Pass: decision 3 updates at the start (iteration 0) and one year later; the balance in force through the first year
equals smb.balance of hydrological year 1881 on the start surface on every cell and nowhere the placeholder; the
first-year thickness change is max(thk0 + B, 0) - thk0 on every cell, so ice-free cells with B < 0 gain nothing. The
reference run is the control: its placeholder is in force through the first year and adds 1000/910 m on every cell.

Inputs: s03 igm-examples/aletsch/ (as checks/smb_reference.py), IGM 3.2.0 (update_modules, time, thk), ../smb.py.
Output: results/smb_first_year.csv, one row per run: start, updates (iteration, time), cells, cells on which the
balance in force through the first year is the run's own first computed balance B (for decision 3 also checked
against smb.balance recomputed here) or the placeholder, first-year thickness change (minimum, maximum), cells where
it equals max(thk0 + B, 0) - thk0, ice-free cells with B < 0 and how many of them gain ice, verdict.
usage (s03, CPU):  source env.sh && cd $IGM_HOME && CUDA_VISIBLE_DEVICES= nrun nice -n 10 taskset -c 0-1 \
                     python checks/smb_first_year.py
"""
import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from netCDF4 import Dataset
from omegaconf import OmegaConf

import igm.processes.thk as thk_process
import igm.processes.time as time_process
from igm.common import State
from igm.common.runner.modules.src import update_modules

IGM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IGM))
from smb import ICE, balance, hydro_year  # noqa: E402

ALETSCH = IGM / "igm-examples/aletsch"
CONF = ALETSCH / "user/conf/processes"
PLACEHOLDER = np.float32(1000.0 / 910.0)
ZWS, DT_DZ, DP_DZ = 2766.0, -0.00552, 0.00035            # clim_aletsch: station elevation (m), lapse rates
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def load(name):
    """A user process module of the example, imported from its file as IGM does."""
    spec = importlib.util.spec_from_file_location(name, ALETSCH / f"user/code/processes/{name}/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(start):
    weights = OmegaConf.load(ALETSCH / "experiment/params_A_step3.yaml").processes.smb_accmelt
    return OmegaConf.create({
        "core": {"print_comp": False, "print_info": False, "logging": False, "hardware": {"profile": False}},
        "processes": {
            "clim_aletsch": OmegaConf.load(CONF / "clim_aletsch.yaml").clim_aletsch,
            "smb_accmelt": OmegaConf.merge(OmegaConf.load(CONF / "smb_accmelt.yaml").smb_accmelt, weights),
            "time": {"start": start, "end": start + 2.0, "save": 1.0, "cfl": 0.3, "step_max": 1.0,
                     "cfl_percentile": 100.0},
            "thk": {"slope_type": "superbee", "ratio_density": 0.91}}})


with Dataset(ALETSCH / "data/input.nc") as nc:
    x, y, usurf, thk, icemask = (np.asarray(nc[v][:], np.float32) for v in ("x", "y", "usurf", "thk", "icemask"))
shape, n = thk.shape, thk.size


def new_state():
    state = State()
    state.original_cwd = ALETSCH
    state.x, state.y, state.dx = tf.constant(x), tf.constant(y), tf.constant(float(x[1] - x[0]))
    state.topg, state.thk, state.icemask = tf.constant(usurf - thk), tf.constant(thk), tf.constant(icemask)
    state.ubar = state.vbar = tf.zeros_like(state.thk)        # ice flow off
    return state


clim, ref = load("clim_aletsch"), load("smb_accmelt")
data = new_state()                                            # the Aletsch inputs as the reference reads them
clim.initialize(config(1880.0), data)
ref.initialize(config(1880.0), data)
p = config(1880.0).processes.smb_accmelt
split = int(365 * p.shift_hydro_year)                         # calendar day index of 1 October
order = (np.arange(365) + split) % 365                        # hydrological day -> calendar day
I = data.direct_radiation.numpy()[order].reshape(365, n)
D = (data.snow_redistribution[0].numpy() * np.float32(p.weight_accumulation)).reshape(n)


def rate(year, z):
    """smb.balance of hydrological year `year` (1 October year - 1 to 30 September year) on surface z, m ice a-1."""
    a, b = (np.flatnonzero(data.year == v)[0] for v in (year - 1, year))
    temp = np.concatenate([data.temp[split:, a], data.temp[:split, b]])[:, None]
    prec = np.concatenate([data.prec[split:, a], data.prec[:split, b]])[:, None]
    T = temp + np.float32(DT_DZ) * (z - np.float32(ZWS))
    P = np.clip(prec * (1 + np.float32(DP_DZ) * (z - np.float32(ZWS))), 0, 1e10) / np.float32(365 * 1000)
    row = data.mb_parameters.numpy()[data.IMB.numpy()[year - 1880]]
    par = [np.float64(row[i] * np.float32(s)) * p.weight_ablation for i, s in ((2, 1e-3), (3, 1e-5), (4, 1e-5))]
    return balance(T, P, I, D, *par).sum(axis=1)[0] / ICE


def initialize(cfg, state):
    state.tlast_smb = tf.Variable(cfg.processes.time.start - 1.0)       # decision 3: the first update at the start


def update(cfg, state):
    if state.t - state.tlast_smb >= 1:
        state.smb = tf.constant(rate(hydro_year(state.t), state.usurf.numpy().reshape(n)).reshape(shape))
        state.tlast_smb.assign(state.t)


decision3 = SimpleNamespace(__name__="smb_decision3", initialize=initialize, update=update,
                            finalize=lambda cfg, state: None)


def run(start, processes):
    """Two model years through IGM's loop -> the recorder's rows and the initial last-update time."""
    cfg, state, log = config(start), new_state(), []

    def record(cfg, state):
        log.append({"it": state.it, "t": float(np.asarray(state.t)), "tlast": float(np.asarray(state.tlast_smb)),
                    "smb": np.asarray(state.smb).reshape(n).copy(), "thk": state.thk.numpy().reshape(n).copy()})

    for module in processes:
        module.initialize(cfg, state)
    tlast0 = float(np.asarray(state.tlast_smb))
    update_modules(processes, [SimpleNamespace(__name__="recorder", run=record)], cfg, state)
    return log, tlast0


B = rate(1881, ((usurf - thk) + thk).reshape(n))             # on the start surface as thk.initialize makes it
rows = []
for name, start, processes in (("decision 3", 1880.75, [decision3, time_process, thk_process]),
                               ("reference", 1880.0, [clim, ref, time_process, thk_process])):
    log, tlast0 = run(start, processes)
    updates, previous = [], tlast0
    for r in log:
        if r["tlast"] != previous:
            updates.append(f"iteration {r['it']} at {r['tlast']:g}")
            previous = r["tlast"]
    first, one = log[0], log[1]                               # t = start (dt 0) and t = start + 1
    assert (first["t"], one["t"]) == (start, start + 1.0)
    own = next(r for r in log if r["tlast"] != tlast0)        # right after the run's first update
    real = int((one["smb"] == own["smb"]).sum())             # its first computed balance, in force in year one
    placeholder = int((one["smb"] == PLACEHOLDER).sum())
    dthk = one["thk"] - first["thk"]
    as_real = int((one["thk"] == np.maximum(first["thk"] + own["smb"], 0)).sum())
    free = (first["thk"] == 0) & (own["smb"] < 0)
    gaining = int((one["thk"][free] > 0).sum())
    if name == "decision 3":
        ok = updates[:1] == [f"iteration 0 at {start:g}"] and (own["smb"] == B).all() and real == n \
            and placeholder == 0 and as_real == n and gaining == 0
        verdict = "passed" if ok else "NOT passed"
    else:
        ok = placeholder == n and np.abs(dthk - PLACEHOLDER).max() < 1e-4
        verdict = "placeholder detected" if ok else "placeholder NOT detected"
    rows.append([name, f"{start:g}", "; ".join(updates), n, real, placeholder, f"{dthk.min():.6f}", f"{dthk.max():.6f}",
                 as_real, int(free.sum()), gaining, verdict])
    print(*rows[-1], sep=" | ", flush=True)

header = ["run", "start", "updates", "cells", "cells_own_first_balance_in_force_year_one",
          "cells_placeholder_in_force_year_one", "year_one_thickness_change_min_m", "year_one_thickness_change_max_m",
          "cells_thickness_as_own_first_balance", "ice_free_cells_negative_balance", "of_them_gaining_ice", "verdict"]
(IGM / "results").mkdir(exist_ok=True)
with open(IGM / "results/smb_first_year.csv", "w", newline="") as f:
    csv.writer(f).writerows([header] + rows)
print("gate", "passed" if rows[0][-1] == "passed" and rows[1][-1] == "placeholder detected" else "NOT passed")
