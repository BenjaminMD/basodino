"""Gate of decision 3 (DESIGN.md step 1): smb.balance reproduces the reference smb_accmelt of the IGM example aletsch
to 1e-5 m, the differences being rounding only.

The reference modules clim_aletsch and smb_accmelt run unchanged (their initialize and update, TensorFlow, float32)
on the Aletsch inputs with the weights of the example's step 3 (weight_accumulation 1.84, weight_ablation 2.01, read
from experiment/params_A_step3.yaml; basin weights 1), on the surface of input.nc, for every year 1880-2100 of
mbparameter.dat: once without the ice mask (the equations alone) and once with it (positive values outside the mask
set to -10 m ice a-1). smb.balance gets the fields smb_accmelt used: air_temp and precipitation as clim_aletsch made
them and the daily direct_radiation, in the reference calendar alignment (loop day kk is calendar day
(kk + int(365 x 0.75)) mod 365, so day 0 is day 274 = 1 October of the same calendar year), precipitation divided by
365 x 1000 (to m w.e. d-1), D = 1.84 x snow_redistribution, and the year's Fm, r_ice, r_snow times 2.01. The sum over
the year times 1000/910 (m ice a-1) is compared cell by cell, with smb.balance evaluated twice: in float64, close to
exact, so the difference is the reference's own float32 rounding; and in float32 like the reference, where the two
roundings add. The ice-mask rule is applied as the reference does.
Switches: the reference picks r_ice where its float32 snow store is exactly 0. On a melt-out day where melt and store
agree to within rounding, one evaluation clips the store to 0 and the other keeps a few 1e-9 m, so the next day melts
with r_ice in one and r_snow in the other: max(T, 0) (r_ice - r_snow) I in m ice for that cell and year, up to about
2e-2 m. Every cell-year above 1e-4 m (all rounding stays below 3.1e-5 m) is checked to be such a switch from
smb.balance's own daily balance (switch_day): the day whose switch amount matches the difference, and the store margin
the melt of the day before left (near 0). The other differences are reported apart, the float32 ones also in float32
spacings of the reference value (of 1 m below 1 m). For the first year of each parameter period: the TensorFlow path
of smb.balance against its numpy path in float32 and in float64, and one call with three members (melt factors and D
varied) against three single calls (daily values, float32).
Gate (Technical settings of DESIGN.md): float64, on the ice-mask cells, every difference but the listed S == 0
switches within 1e-5 m; the TensorFlow path agrees with the numpy path.

Inputs: s03 igm-examples/aletsch/ (data/, experiment/params_A_step3.yaml, user/conf and user/code/processes of
clim_aletsch and smb_accmelt), ../smb.py.
Output (full range only; a shorter one prints):
  results/smb_reference.csv            one row per year: melt parameters (reference units times their scale), the
      reference's mean over the ice mask and its lowest cell (m ice a-1); the largest absolute differences (m ice a-1)
      in float64 (all cells, with the mask rule, ice-mask cells) and float32 (all cells, ice-mask cells); switches
      (float64, float32); the largest other differences (float64, float32, float32 in spacings, float64 on the mask);
      the TensorFlow path in float32 and float64 and the member axis (daily, m w.e. d-1, blank if not run); cells
      above 1e-5 m
  results/smb_reference_switches.csv   one row per switch: year, cell (flat, row, column), on the mask, evaluation,
      difference and reference value (m ice a-1), the switch day (loop index from 1 October, calendar day), its
      switch amount and the residual (m ice a-1), the store margin of the day before (m w.e.)
usage (s03, CPU):  source env.sh && cd $IGM_HOME && CUDA_VISIBLE_DEVICES= nrun nice -n 10 taskset -c 0-1 \
                     python checks/smb_reference.py [first last]
"""
import csv
import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import tensorflow as tf
from netCDF4 import Dataset
from omegaconf import OmegaConf

IGM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IGM))
from smb import ICE, balance  # noqa: E402

ALETSCH = IGM / "igm-examples/aletsch"
GATE = 1e-5                                                  # m
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def load(name):
    """A user process module of the example, imported from its file as IGM does."""
    spec = importlib.util.spec_from_file_location(name, ALETSCH / f"user/code/processes/{name}/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class State:
    pass


clim, ref = load("clim_aletsch"), load("smb_accmelt")
weights = OmegaConf.load(ALETSCH / "experiment/params_A_step3.yaml").processes.smb_accmelt
cfg = OmegaConf.create({"processes": {
    "clim_aletsch": OmegaConf.load(ALETSCH / "user/conf/processes/clim_aletsch.yaml").clim_aletsch,
    "smb_accmelt": OmegaConf.merge(OmegaConf.load(ALETSCH / "user/conf/processes/smb_accmelt.yaml").smb_accmelt,
                                   weights),
    "time": {"start": 1880.0}}})
p = cfg.processes.smb_accmelt
w_acc, w_abl = p.weight_accumulation, p.weight_ablation

state = State()
state.original_cwd = ALETSCH
with Dataset(ALETSCH / "data/input.nc") as nc:
    state.x, state.y = (tf.constant(np.asarray(nc[v][:], np.float32)) for v in ("x", "y"))
    usurf, thk, icemask = (np.asarray(nc[v][:], np.float32) for v in ("usurf", "thk", "icemask"))
state.usurf, state.topg = tf.constant(usurf), tf.constant(usurf - thk)
clim.initialize(cfg, state)
ref.initialize(cfg, state)

shape, n = usurf.shape, usurf.size
order = (np.arange(365) + int(365 * p.shift_hydro_year)) % 365          # reference loop day -> calendar day
I = state.direct_radiation.numpy()[order].reshape(365, n)
D = (state.snow_redistribution[0].numpy() * np.float32(w_acc)).reshape(n)
mb, imb = state.mb_parameters.numpy(), state.IMB.numpy()
glacier = icemask.reshape(n) > 0.5
period_first = {int(row[0]) for row in mb}

SWITCH = 1e-4                                                # m: above every rounding difference (3.1e-5 seen)


def switch_day(b, t, i, r_ice, r_snow, diff):
    """The day whose melt factor, switched between r_ice and r_snow, explains the difference diff (m ice a-1) at one
    cell, from smb.balance's own daily balance b (m w.e.) of that cell: the store follows as S = max(S + b, 0), and a
    switch on day k moves the year by max(T, 0) (r_ice - r_snow) I in m ice (the 0.91 cancels). Returns the best day,
    its switch amount, the residual, and the store left by the melt of the day before, before clipping (near 0 at the
    knife edge of the reference's S == 0 test)."""
    S, margin = 0.0, np.empty(len(b))
    for k, bk in enumerate(np.asarray(b, np.float64)):
        margin[k] = S + bk
        S = max(margin[k], 0.0)
    amount = np.maximum(t, 0) * (r_ice - r_snow) * i
    k = int(np.argmin(np.abs(np.abs(diff) - amount)))
    return k, amount[k], abs(abs(diff) - amount[k]), margin[k - 1]


first, last = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (1880, 2100)
rows, switches = [], []
for year in range(first, last + 1):
    state.t = tf.Variable(float(year))
    state.tlast_clim_aletsch.assign(year - 1.0)                   # make the yearly update due now
    clim.update(cfg, state)
    if hasattr(state, "icemask"):
        del state.icemask
    state.tlast_smb.assign(year - 1.0)
    ref.update(cfg, state)
    raw = state.smb.numpy().reshape(n)
    state.icemask = tf.constant(icemask)
    state.tlast_smb.assign(year - 1.0)
    ref.update(cfg, state)
    masked = state.smb.numpy().reshape(n)

    T = state.air_temp.numpy()[order].reshape(365, n)
    P = (state.precipitation.numpy()[order] / np.float32(365 * 1000)).reshape(365, n)
    row = mb[imb[year - 1880]]
    Fm, ri, rs = row[2] * np.float32(1e-3), row[3] * np.float32(1e-5), row[4] * np.float32(1e-5)
    par = [np.float64(v) * w_abl for v in (Fm, ri, rs)]
    b32 = balance(T, P, I, D, *par)
    b64 = balance(T.astype(np.float64), P.astype(np.float64), I, D, *par)
    a32 = b32.sum(axis=1)[0] * (1000.0 / 910.0)
    a64 = b64.sum(axis=1)[0] / ICE
    d32, d64 = np.abs(a32 - raw), np.abs(a64 - raw)
    m64 = np.abs(np.where((a64 < 0) | glacier, a64, -10.0) - masked)
    for name, b, a, dd in (("float64", b64, a64, d64), ("float32", b32, a32, d32)):
        for j in np.flatnonzero(dd > SWITCH):
            k, amount, residual, margin = switch_day(b[0, :, j], T[:, j], I[:, j], par[1], par[2], a[j] - raw[j])
            switches.append([year, j, j // shape[1], j % shape[1], int(glacier[j]), name, f"{a[j] - raw[j]:.6g}",
                             f"{raw[j]:.6f}", k, (date(2001, 1, 1) + timedelta(int(order[k]))).strftime("%m-%d"),
                             f"{amount:.6g}", f"{residual:.3g}", f"{margin:.3g}"])
            print("switch", *switches[-1], flush=True)
    o32, o64 = d32 <= SWITCH, d64 <= SWITCH                  # all cells but the switched ones
    spacings = d32 / np.spacing(np.maximum(np.abs(raw), np.float32(1)))   # float32 spacings of the value (of 1 m)
    dtf = dtf64 = dmember = ""
    if year in period_first or year == first:
        btf = balance(tf.constant(T), tf.constant(P), tf.constant(I), tf.constant(D), *par).numpy()
        dtf = f"{np.abs(btf - b32).max():.3g}"
        btf = balance(tf.constant(T.astype(np.float64)), tf.constant(P.astype(np.float64)), tf.constant(I),
                      tf.constant(D), *par)
        assert btf.dtype == tf.float64
        dtf64 = f"{np.abs(btf.numpy() - b64).max():.3g}"
        factor = np.array([[1.0, 0.8, 1.2], [1.0, 1.1, 0.9], [1.0, 1.1, 0.9]])     # three members: Fm, r_ice, r_snow
        pm = [v * fv for v, fv in zip(par, factor)]
        Dm = D * np.array([1.0, 0.9, 1.3], np.float32)[:, None]                      # (member, cell)
        bm = balance(T, P, I, Dm, *pm)
        single = [balance(T, P, I, Dm[m], *(v[m] for v in pm))[0] for m in range(3)]
        dmember = f"{max(np.abs(bm[m] - single[m]).max() for m in range(3)):.3g}"
    rows.append([year, f"{Fm:.6g}", f"{ri:.6g}", f"{rs:.6g}", f"{raw[glacier].mean():.6f}", f"{raw.min():.4f}",
                 f"{d64.max():.3g}", f"{m64.max():.3g}", f"{d64[glacier].max():.3g}", f"{d32.max():.3g}",
                 f"{d32[glacier].max():.3g}", int((~o64).sum()), int((~o32).sum()), f"{d64[o64].max():.3g}",
                 f"{d32[o32].max():.3g}", f"{spacings[o32].max():.3g}", f"{d64[o64 & glacier].max():.3g}", dtf,
                 dtf64, dmember, int((d64 > GATE).sum()), int((d32 > GATE).sum())])
    print(*rows[-1], flush=True)

header = ["year", "Fm_m_ice_per_day_degC", "r_ice", "r_snow", "reference_mean_on_mask_m_ice", "reference_min_m_ice",
          "max_diff_float64_m_ice", "max_diff_float64_mask_rule_m_ice", "max_diff_float64_on_mask_m_ice",
          "max_diff_float32_m_ice", "max_diff_float32_on_mask_m_ice", "switches_float64", "switches_float32",
          "max_diff_float64_other_m_ice", "max_diff_float32_other_m_ice", "max_spacings_float32_other",
          "max_diff_float64_other_on_mask_m_ice", "max_diff_tensorflow_daily_m_we",
          "max_diff_tensorflow_float64_daily_m_we", "max_diff_member_axis_daily_m_we",
          "cells_above_1e-5_float64", "cells_above_1e-5_float32"]
switch_header = ["year", "cell", "row", "column", "on_mask", "evaluation", "difference_m_ice", "reference_m_ice",
                 "switch_day_of_loop", "calendar_day", "switch_amount_m_ice", "residual_m_ice",
                 "store_margin_day_before_m_we"]
if (first, last) == (1880, 2100):                            # a shorter range only prints
    (IGM / "results").mkdir(exist_ok=True)
    with open(IGM / "results/smb_reference.csv", "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    with open(IGM / "results/smb_reference_switches.csv", "w", newline="") as f:
        csv.writer(f).writerows([switch_header] + switches)
top = {h: max(float(r[i]) for r in rows) for i, h in enumerate(header) if 6 <= i <= 16}
explained = sum(float(s[11]) <= 5e-5 and abs(float(s[12])) <= 1e-5 for s in switches)
print(f"years {first}-{last}: {n * len(rows)} cell-years, {glacier.sum()} of {n} cells on the ice mask")
print(f"switches of the S == 0 test: float64 {sum(r[11] for r in rows)}, float32 {sum(r[12] for r in rows)}; "
      f"{explained} of {len(switches)} explained by one day (residual <= 5e-5 m, store margin <= 1e-5 m)")
print(f"max |difference| (m ice a-1), all cells: float64 {top['max_diff_float64_m_ice']:.3g}, float32 "
      f"{top['max_diff_float32_m_ice']:.3g}, float64 with the mask rule {top['max_diff_float64_mask_rule_m_ice']:.3g}")
print(f"without the switches: float64 {top['max_diff_float64_other_m_ice']:.3g} (on the mask "
      f"{top['max_diff_float64_other_on_mask_m_ice']:.3g}), float32 {top['max_diff_float32_other_m_ice']:.3g} "
      f"(at most {top['max_spacings_float32_other']:.3g} float32 spacings)")
tfd = [float(r[i]) for r in rows for i in (17, 18) if r[i] != ""]
mask_switches = sum(s[4] == 1 and s[5] == "float64" for s in switches)
print(f"TensorFlow minus numpy path, daily, m w.e.: float32 {max(float(r[17]) for r in rows if r[17] != ''):.3g}, "
      f"float64 {max(float(r[18]) for r in rows if r[18] != ''):.3g} ({len(tfd) // 2} years)")
print(f"gate {GATE:g} m, float64, ice-mask cells, switches apart ({mask_switches} listed on the mask): "
      f"{'passed' if top['max_diff_float64_other_on_mask_m_ice'] <= GATE else 'NOT passed'}")
print(f"old gate {GATE:g} m on every cell-year: float64 "
      f"{'passed' if top['max_diff_float64_m_ice'] <= GATE else 'NOT passed'}, float32 "
      f"{'passed' if top['max_diff_float32_m_ice'] <= GATE else 'NOT passed'}")
