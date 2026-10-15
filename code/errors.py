"""Uncertainty of every headline number of tex/min_si.tex, from the uncertainties of its inputs.

Linear propagation where the result is linear in the input (Question 1, 5.1, 5.2); the model re-run with each input
moved by one standard deviation where it is not (Questions 3 and 4: the degree-day factor is recalibrated, then the
91 years are reconstructed again). Sources are taken as independent and added in quadrature. Known one-sided biases
are listed separately and are not part of the standard deviation.

Standard deviations of the inputs (one sigma):
  elevation models   1994 +-2 m, 2021 +-1 m, as one offset over the whole glacier (aerial photogrammetry over snow
                     and ice; the models are identical off the glacier, so the stable terrain cannot test this)
  density            +-60 kg m-3 (Huss 2013)
  glacier masks      half a cell (5 m) along the outline, as one offset
  model dates        anywhere in October, each: +-15 d uniform
  lapse rate         +-0.5e-3 K m-1
  snow threshold     +-1 K
  precipitation      +-25 per cent at the glacier (gauge undercatch, elevation gradient)
  scaling constant   +-34 per cent of the volume (Bahr 2015); exponent +-0.025 with c in km units
  yearly balances    the model's root mean square error against the 34 measured years (empirical)
Inputs: the course grids, task2_meteo.csv, task1_geodetic.csv, task4_balance.csv, the GLAMOS files.
Output: code/errors.csv (quantity, value, sigma, unit, definition) and the budget on the terminal.
usage:  python3 code/errors.py
"""
import csv
from collections import defaultdict

import numpy as np
from scipy import stats

from common import (CALIB, COURSE, LAPSE, METEO, ND, RHO_DV, RHO_ICE, RHO_W, ROOT, STATION, T0, TM, YEARS, read_csv,
                    read_esri_ascii, read_glamos, write_csv)

S_Z94, S_Z21, S_RHO, S_CELL, S_DATE = 2.0, 1.0, 60.0, 5.0, 15 / np.sqrt(3)   # m, m, kg m-3, m, d
S_LAPSE, S_T0, S_P = 0.5e-3, 1.0, 0.25                                        # K m-1, K, fraction
S_C, S_GAMMA, C, GAMMA = 0.34, 0.025, 0.034, 1.375                           # fraction of V; km^(3-2 gamma)
YEAR = 365.25

# --- Question 1: geodetic balance, linear in every input
z94, hd = read_esri_ascii(COURSE / "dem_basodino_1994.asc")
z21, _ = read_esri_ascii(COURSE / "dem_basodino_2021.asc")
g94 = read_esri_ascii(COURSE / "mask_basodino_1994.asc")[0] == 1
g21 = read_esri_ascii(COURSE / "mask_basodino_2021.asc")[0] == 1
ice, cell = g94 | g21, hd["cellsize"] ** 2
dv = (z21 - z94)[ice].sum() * cell                                            # m3
a94, a21, a_union = g94.sum() * cell, g21.sum() * cell, ice.sum() * cell
a_mean, dt = (a94 + a21) / 2, 27.0


def perimeter(m):
    """Length of the outline of a mask, m: cell edges between ice and no ice."""
    return (np.sum(m[1:] != m[:-1]) + np.sum(m[:, 1:] != m[:, :-1]) + m[0].sum() + m[-1].sum() + m[:, 0].sum()
            + m[:, -1].sum()) * hd["cellsize"]


s_dv = np.hypot(S_Z94, S_Z21) * a_union                                       # one offset over the glacier
s_a94, s_a21 = perimeter(g94) * S_CELL, perimeter(g21) * S_CELL
s_amean = np.hypot(s_a94, s_a21) / 2
s_dt = np.hypot(S_DATE, S_DATE) / YEAR
b_rate = RHO_DV * dv / (a_mean * dt)                                          # kg m-2 a-1
rel1 = {"elevation models": s_dv / abs(dv), "density": S_RHO / RHO_DV, "area": s_amean / a_mean, "dates": s_dt / dt}
s_rel1 = np.sqrt(sum(v ** 2 for v in rel1.values()))
print("Question 1: relative standard deviation of the rate")
for k, v in rel1.items():
    print(f"  {k:18s} {100 * v:5.1f} %")
print(f"  {'in quadrature':18s} {100 * s_rel1:5.1f} %   rate {b_rate / 1000:.3f} +- {s_rel1 * abs(b_rate) / 1000:.3f} m w.e. a-1")

# --- the model of Questions 3 and 4, with the inputs that are perturbed as arguments
z = z21[g21]
z_station = {r["station_abbr"]: float(r["station_height_masl"]) for r in
             csv.DictReader(open(METEO / "ogd-nbcn_meta_stations.csv", encoding="latin-1"), delimiter=";")}[STATION]
meteo = defaultdict(list)
for r in read_csv("task2_meteo.csv"):
    if r["station"] == STATION:
        meteo[int(r["hydro_year"])].append((float(r["T_degC"]), float(r["P_mm"])))
tp = {y: np.array(meteo[y]) for y in YEARS}


def model(lapse=LAPSE, t0=T0, p_factor=1.0, b_total=b_rate * dt):
    """Degree-day factor (mm w.e. per K d) calibrated on b_total (kg m-2 over the 27 years), and the balance of every
    hydrological year 1935-2025 (m w.e.) on the 2021 surface."""
    c, s = {}, {}
    for y in YEARS:
        t = tp[y][:, :1] - lapse * (z[None, :] - z_station)
        c[y] = np.where(t < t0, p_factor * tp[y][:, 1:] / 1000, 0.0).sum(axis=0).mean()
        s[y] = (ND * np.maximum(t - TM, 0.0)).sum(axis=0).mean()
    k = (sum(c[y] for y in CALIB) - b_total / 1000) / sum(s[y] for y in CALIB)
    return k * 1000, {y: c[y] - k * s[y] for y in YEARS}


def summary(k, b):
    """The headline numbers of Questions 3, 4 and 5.2 from one model run."""
    v = np.array([b[y] for y in YEARS])
    yrs = np.array(list(YEARS))
    out = {"F": k, "cumulative": v.sum(), "mean": v.mean()}
    for a, e in ((1935, 1965), (1966, 1995), (1996, 2025)):
        out[f"mean_{a}_{e}"] = v[(yrs >= a) & (yrs <= e)].mean()
    out["trend_1986_2025"] = stats.linregress(yrs[yrs >= 1986], v[yrs >= 1986]).slope * 10
    out["B_2022_2025"] = sum(b[y] for y in range(2022, 2026))
    out["B_future"] = out["mean_1996_2025"]
    v21 = C * (a21 / 1e6) ** GAMMA * 1e9                                          # m3
    v25 = v21 + a21 * out["B_2022_2025"] * RHO_W / RHO_ICE
    out["t_half"] = 2025 + (v25 - v21 / 2) / (a21 * -out["B_future"] * RHO_W / RHO_ICE)   # end of that hydrological year
    return out


nominal = summary(*model())
runs = {"lapse rate": (model(lapse=LAPSE - S_LAPSE), model(lapse=LAPSE + S_LAPSE)),
        "snow threshold": (model(t0=T0 - S_T0), model(t0=T0 + S_T0)),
        "precipitation": (model(p_factor=1 - S_P), model(p_factor=1 + S_P)),
        "geodetic balance": (model(b_total=b_rate * dt * (1 - s_rel1)), model(b_total=b_rate * dt * (1 + s_rel1)))}
effects = {src: {q: (summary(*hi)[q] - summary(*lo)[q]) / 2 for q in nominal} for src, (lo, hi) in runs.items()}
sigma = {q: np.sqrt(sum(effects[src][q] ** 2 for src in runs)) for q in nominal}

# --- Question 5: the scaling volume and the year, the terms that the model runs do not hold
v21 = C * (a21 / 1e6) ** GAMMA * 1e9
rel_v = {"scaling constant": S_C, "exponent": np.log(a21 / 1e6) * S_GAMMA, "area": GAMMA * s_a21 / a21}
s_rel_v = np.sqrt(sum(v ** 2 for v in rel_v.values()))
melt_rate = a21 * -nominal["B_future"] * RHO_W / RHO_ICE                        # m3 a-1
effects["scaling volume"] = {q: 0.0 for q in nominal} | {"t_half": 0.5 * s_rel_v * v21 / melt_rate}
sigma["t_half"] = np.hypot(sigma["t_half"], effects["scaling volume"]["t_half"])

# --- the empirical checks: the measured years and the geodetic chain of GLAMOS
glamos = read_glamos()
b = {int(r["hydro_year"]): float(r["B_mwe"]) for r in read_csv("task4_balance.csv") if r["station"] == STATION}
gy = [y for y in sorted(glamos) if y in YEARS]
d = np.array([b[y] - glamos[y][2] for y in gy])
r = np.corrcoef([b[y] for y in gy], [glamos[y][2] for y in gy])[0, 1]
r_lo, r_hi = np.tanh(np.arctanh(r) + np.array([-1.96, 1.96]) / np.sqrt(len(gy) - 3))
with open(ROOT / "data/glamos_basodino_volumechange.csv", encoding="utf-8") as f:
    vc = {(r_[3], r_[4]): (float(r_[11]), float(r_[12]), float(r_[9])) for r_ in csv.reader(f)
          if len(r_) > 13 and r_[0] == "C14-10"}
CHAIN = ("19299999", "19491007", "19610919", "19710903", "19850910", "19910918", "20020916", "20080829", "20140923",
         "20190920")
chain = [(max(int(a[:4]) + 1, 1935), int(e[:4])) + vc[(a, e)] for a, e in zip(CHAIN, CHAIN[1:])]
chain += [(2020, 2021, np.mean([glamos[2020][2], glamos[2021][2]]), 0.3, np.nan)]        # no geodetic pair 2019-2021
chain += [(2022, 2024) + vc[("20219999", "20240823")], (2025, 2025, glamos[2025][2], 0.3, np.nan)]
cum_meas = sum(rate * (e - a + 1) for a, e, rate, s, _ in chain)
s_cum_meas = np.sqrt(sum((s * (e - a + 1)) ** 2 for a, e, rate, s, _ in chain))
cum_model_pre = sum(b[y] for y in range(1935, 1962))
cum_meas_pre = sum(rate * (e - a + 1) for a, e, rate, s, _ in chain if e <= 1961)
OLD = ("19079999", "19299999", "19491007", "19610919", "19710903", "19850910", "19910918")
dv_1907 = sum(vc[(a, e)][2] for a, e in zip(OLD, OLD[1:]))
areas = {"19079999": 3.2020, "19299999": 3.2020, "19491007": 0.9968, "19610919": 2.5948, "19710903": 2.3287,
         "19850910": 2.3378, "19910918": 2.3361}
s_dv_1907 = np.sqrt(sum((vc[(a, e)][1] * (int(e[:4]) - int(a[:4])) * (areas[a] + areas[e]) / 2 * 1000 / 850 / 1000) ** 2
                        for a, e in zip(OLD, OLD[1:])))

# --- output
rows = [("B_rate", b_rate / 1000, s_rel1 * abs(b_rate) / 1000, "m w.e. a-1", "Question 1, mean area; quadrature of " +
         ", ".join(f"{k} {100 * v:.1f} %" for k, v in rel1.items())),
        ("dV", dv / 1e9, s_dv / 1e9, "km3", "elevation models alone"),
        ("A_1994", a94 / 1e6, s_a94 / 1e6, "km2", f"half a cell along {perimeter(g94) / 1e3:.2f} km of outline"),
        ("A_2021", a21 / 1e6, s_a21 / 1e6, "km2", f"half a cell along {perimeter(g21) / 1e3:.2f} km of outline"),
        ("B_total", b_rate * dt / 1000, s_rel1 * abs(b_rate) * dt / 1000, "m w.e.", "27 years"),
        ("glamos_1991_2021", vc[("19910918", "20219999")][0], vc[("19910918", "20219999")][1], "m w.e. a-1",
         "GLAMOS, its own uncertainty, for comparison")]
for q, unit, what in (("F", "mm w.e. (K d)-1", "Question 3"), ("cumulative", "m w.e.", "Question 4 (i)"),
                      ("mean", "m w.e. a-1", "Question 4 (ii)"), ("mean_1935_1965", "m w.e. a-1", "Question 4"),
                      ("mean_1966_1995", "m w.e. a-1", "Question 4"), ("mean_1996_2025", "m w.e. a-1", "Question 4, 5.2"),
                      ("trend_1986_2025", "m w.e. a-1 per decade", "Question 4"),
                      ("B_2022_2025", "m w.e.", "Question 5.2"), ("t_half", "year", "Question 5.2, area constant")):
    rows.append((q, nominal[q], sigma[q], unit, what + "; " + ", ".join(
        f"{src} {abs(effects[src][q]):.3g}" for src in effects if effects[src][q])))
yrs = np.array(list(YEARS))
v = np.array([b[y] for y in YEARS])
rows += [("trend_1986_2025_se", nominal["trend_1986_2025"],
          stats.linregress(yrs[yrs >= 1986], v[yrs >= 1986]).stderr * 10, "m w.e. a-1 per decade",
          "standard error of the fitted slope alone"),
         ("V_2021", v21 / 1e9, s_rel_v * v21 / 1e9, "km3", "Question 5.1; " + ", ".join(
             f"{k} {100 * v_:.1f} %" for k, v_ in rel_v.items())),
         ("H_2021", v21 / a21, s_rel_v * v21 / a21, "m", "mean thickness from scaling"),
         ("year_rmse", np.sqrt((d ** 2).mean()), d.std(ddof=1) / np.sqrt(2 * (len(d) - 1)), "m w.e.",
          f"root mean square of model minus measured over the {len(gy)} years {gy[0]}-{gy[-1]}; bias {d.mean():+.3f}"),
         ("r_measured", r, (r_hi - r_lo) / 2 / 1.96, "", f"correlation with the measured years, 95 % interval {r_lo:.2f} to {r_hi:.2f}"),
         ("cumulative_measured", cum_meas, s_cum_meas, "m w.e.",
          "GLAMOS geodetic chain 1935-2025, glaciological 2020, 2021, 2025; model " + f"{nominal['cumulative']:.1f}"),
         ("bias_before_1962", (cum_model_pre - cum_meas_pre) / 27, np.nan, "m w.e. a-1",
          f"model minus measured, 1935-1961: model {cum_model_pre:.1f}, measured {cum_meas_pre:.1f} m w.e."),
         ("dV_1907_1991_measured", dv_1907, s_dv_1907, "km3", "GLAMOS chain of six pairs, supplement S3")]
write_csv("errors.csv", ("quantity", "value", "sigma", "unit", "definition"), rows)
print("\nbudget (effect of one standard deviation of each source)")
print(f"{'':18s}" + "".join(f"{q[:14]:>15s}" for q in nominal))
for src in effects:
    print(f"{src:18s}" + "".join(f"{effects[src][q]:15.3f}" for q in nominal))
print(f"{'quadrature':18s}" + "".join(f"{sigma[q]:15.3f}" for q in nominal))
print(f"{'value':18s}" + "".join(f"{nominal[q]:15.3f}" for q in nominal))
print("\nbiases, one-sided: precipitation x 1.67 (measured winter balance) gives F "
      f"{model(p_factor=1.67)[0]:.2f}; stations: see task3_ddf.csv")
for q, val, s, u, what in rows:
    print(f"{q:24s} {val:10.4g} +- {s:8.3g} {u:22s} {what[:90]}")
