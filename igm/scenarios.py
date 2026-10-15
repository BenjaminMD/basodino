"""Monthly climate changes 2027-2100 per CH2025 chain, SSP and 2026 level: decision 12 of igm/DESIGN.md.

Method:
  chain change   per station (OTL, ROE), chain and calendar month, the 30-year mean of each warming-level window
                 (GWL1.5, 2.0, 2.5, 3.0) against the chain's own 1991-2020: dT added (degC), P ratio (multiplied)
  global warming per SSP piecewise linear in time through +0.9 degC at the centre of 1991-2020 (2006.0) and the AR6
                 SPM.1 best estimates at the centres of 2021-2040, 2041-2060, 2081-2100 (2031.0, 2051.0, 2091.0),
                 the last segment extended to 2100; evaluated at the centre of hydrological year Y (1 April, Y.25)
  in warming     the chain change linear between +0.9 (no change) and its windows; beyond the last window extrapolated
                 from the last two (2.5-3.0, or 2.0-2.5 for the chains without GWL3.0): dT linearly, the P ratio
                 linearly in log space (positive; equal to the last window's ratio at its level, so continuous there)
  2026 level     native: as above. anchored: one constant dT offset per station, chain, SSP and season (DJF, MAM,
                 JJA, SON) that puts hydrological year 2026 on the observed 1991-2026 OTL trend of that season
                 (least-squares line through the hydrological-year means, its 2026 value minus the 1991-2020 mean).
                 P is not anchored. Seasons and years are day-weighted means of months (365-day calendar).
Observed OTL T is the homogenised monthly mean (ths200m0); the forcing of weather.py has these monthly means in
1991-2026 (Technical settings of DESIGN.md), so the anchor is the one of the forcing.
Inputs, s02: raw/climate/ch2025/ogd-climate-scenarios-ch2025_{otl,roe}_{tas,pr}_{ref91-20,gwl*}.csv,
  raw/climate/obs/ogd-nbcn_otl_m.csv, derived/climate_summary.csv (checked against, scripts/climate_summary.py)
Outputs:
  igm/inputs/scenarios_monthly.csv  station, chain, ssp, level, hydro_year, month, global_warming_degC, dT_degC,
                                    P_ratio; 2 x 26 x 3 x 2 x 74 years x 12 months, months in hydrological order
  igm/results/scenarios_check.csv   check, station, variable, ssp, level, period, season, n, median, min, max, unit:
                                    reproduction of climate_summary.csv, observed trend, 2026 levels, scenario minus
                                    observed 2016-2025 and 2021-2025, anchor error, P ratio range, 2081-2100 change,
                                    jumps at the cuts (warming levels, AR6 centres), anchor offset steps between seasons
usage (s02, agl env):  python igm/scenarios.py
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]                      # s02:/data/befahl/projects/AGL/basodino
CH, OBS, IGM = ROOT / "raw/climate/ch2025", ROOT / "raw/climate/obs", ROOT / "igm"
DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])          # January to December
SEASONS = {"year": range(1, 13), "DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
HYDRO = (10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9)
GWL, REF_GW = ("1.5", "2.0", "2.5", "3.0"), 0.9
CENTRES = np.array([2006.0, 2031.0, 2051.0, 2091.0])
SSP = {"SSP1-2.6": (1.5, 1.7, 1.8), "SSP2-4.5": (1.5, 2.0, 2.7), "SSP5-8.5": (1.6, 2.4, 4.4)}
YEARS, ANCHOR, BASE = range(2027, 2101), range(1991, 2027), range(1991, 2021)


def warming(ssp, year):
    """Global warming (degC above 1850-1900) at the centre of hydrological year(s) `year`."""
    t, w = np.asarray(year, float) + 0.25, np.array((REF_GW, *SSP[ssp]))
    return np.interp(t, CENTRES, w) + np.maximum(t - CENTRES[-1], 0) * (w[-1] - w[-2]) / (CENTRES[-1] - CENTRES[-2])


def at(x, y, w, log=False):
    """Values y (points, 12) given at warming x, linear in warming at w (n,), outside x the end segments extended;
    with log, beyond the last point the last segment is extended in log y (y[-2] (y[-1] / y[-2])^f, f = 1 at x[-1])."""
    i = np.clip(np.searchsorted(x, w) - 1, 0, len(x) - 2)
    v = y[i] + ((w - x[i]) / (x[i + 1] - x[i]))[:, None] * (y[i + 1] - y[i])
    if log:
        beyond = w > x[-1]
        v[beyond] = y[-2] * (y[-1] / y[-2]) ** ((w[beyond] - x[-2]) / (x[-1] - x[-2]))[:, None]
    return v


def season(monthly, s):
    """Day-weighted mean over the months of season s; monthly (..., 12) January to December."""
    m = np.array(SEASONS[s]) - 1
    return (monthly[..., m] * DAYS[m]).sum(-1) / DAYS[m].sum()


def chain_means(st, var, period):
    """CH2025 file -> DataFrame calendar month (1-12) x chain of the daily mean."""
    x = pd.read_csv(CH / f"ogd-climate-scenarios-ch2025_{st}_{var}_{period}.csv", sep=";", skiprows=23,
                    na_values="NA")
    return x.drop(columns="DATE").groupby(x.DATE.str[5:7].astype(int)).mean()


# chain changes: change[st][var] = (chains, {chain: (warming points, values (points, 12))})
change, means = {}, {}
for st in ("otl", "roe"):
    for var in ("tas", "pr"):
        ref = chain_means(st, var, "ref91-20")
        new = {g: chain_means(st, var, f"gwl{g}") for g in GWL}
        means[st, var] = ref, new
        pts = {}
        for c in ref.columns:
            gs = [g for g in GWL if c in new[g].columns]
            d = [new[g][c].values - ref[c].values if var == "tas" else new[g][c].values / ref[c].values for g in gs]
            pts[c] = (np.array([REF_GW, *map(float, gs)]), np.array([np.zeros(12) if var == "tas" else np.ones(12), *d]))
        change[st, var] = pts
chains = list(means["otl", "tas"][0].columns)
assert len(chains) == 26 and all(list(means[k][0].columns) == chains for k in means)
n3 = sum(c in means["otl", "tas"][1]["3.0"].columns for c in chains)

# observed OTL: hydrological-year monthly T, trend 1991-2026 per season
m = pd.read_csv(OBS / "ogd-nbcn_otl_m.csv", sep=";", encoding="latin-1")
T = {(int(r[6:10]), int(r[3:5])): v for r, v in zip(m.reference_timestamp, m.ths200m0)}
obs = np.array([[T[y - (k >= 10), k] for k in range(1, 13)] for y in ANCHOR])           # (hydro year, Jan..Dec)
assert np.isfinite(obs).all()
anomaly = {s: season(obs, s) - season(obs[:len(BASE)], s).mean() for s in SEASONS}       # vs 1991-2020
trend = {s: np.polyval(np.polyfit(list(ANCHOR), anomaly[s], 1), 2026) for s in SEASONS}

# monthly changes per station, chain, SSP and level; dT[st, c, ssp, level] (years 2016..2100, 12)
span = np.arange(2016, 2101)
dT, Pr = {}, {}
for st in ("otl", "roe"):
    for c in chains:
        for ssp in SSP:
            w = warming(ssp, span)
            native = at(*change[st, "tas"][c], w)
            Pr[st, c, ssp] = at(*change[st, "pr"][c], w, log=True)
            i26 = list(span).index(2026)
            offset = np.zeros(12)
            for s in ("DJF", "MAM", "JJA", "SON"):
                offset[np.array(SEASONS[s]) - 1] = trend[s] - season(native[i26], s)
            dT[st, c, ssp, "native"], dT[st, c, ssp, "anchored"] = native, native + offset

out = list(span).index(YEARS[0])
(IGM / "inputs").mkdir(exist_ok=True)
with open(IGM / "inputs/scenarios_monthly.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(("station", "chain", "ssp", "level", "hydro_year", "month", "global_warming_degC", "dT_degC",
                 "P_ratio"))
    for (st, c, ssp, lev), d in dT.items():
        w = warming(ssp, span)
        for i in range(out, len(span)):
            wr.writerows((st.upper(), c, ssp, lev, span[i], k, f"{w[i]:.4f}", f"{d[i, k - 1]:.4f}",
                          f"{Pr[st, c, ssp][i, k - 1]:.5f}") for k in HYDRO)

# checks
rows = []


def stat(check, st, var, ssp, lev, period, s, x, unit):
    x = np.asarray(x, float)
    rows.append((check, st, var, ssp, lev, period, s, len(x), *(f"{v:.4f}" for v in (np.median(x), x.min(), x.max())),
                 unit))


# 1 chain changes against derived/climate_summary.csv (its season rule: tas plain mean of months, pr total)
summ = pd.read_csv(ROOT / "derived/climate_summary.csv")
summ = summ[summ.source == "CH2025 DAILY-LOCAL"]
for (st, var), (ref, new) in means.items():
    diffs = []
    for g in GWL:
        for s in ("year", "DJF", "JJA"):
            ms = list(SEASONS[s])
            if var == "tas":
                v = new[g].loc[ms].mean() - ref.loc[ms].mean()
            else:
                v = 100 * ((new[g].loc[ms].mul(DAYS[np.array(ms) - 1], axis=0)).sum()
                           / (ref.loc[ms].mul(DAYS[np.array(ms) - 1], axis=0)).sum() - 1)
            v = v.dropna()
            sel = summ[(summ.station == st.upper()) & (summ.variable == var) & (summ.scenario == f"GWL{g}")
                       & (summ.season == s)].set_index("member").value
            assert set(v.index) <= set(sel.index), (st, var, g, s)
            diffs += list(np.abs(v.values - sel[v.index].astype(float).values))
            diffs.append(abs(v.median() - float(sel["median"])))
    stat("chain change minus climate_summary.csv, absolute", st.upper(), var, "", "", "GWL windows", "year DJF JJA",
         diffs, "degC" if var == "tas" else "percent")

# 2 observed trend, 2026 levels, anchor error, scenario minus observed
for s in SEASONS:
    rows.append(("observed OTL trend 1991-2026 at 2026 minus 1991-2020", "OTL", "tas", "", "", "hydrological years",
                 s, len(ANCHOR), f"{trend[s]:.4f}", "", "", "degC"))
i26 = list(span).index(2026)
for st in ("otl", "roe"):
    for ssp in SSP:
        for lev in ("native", "anchored"):
            for s in ("year", "JJA"):
                lv = [season(dT[st, c, ssp, lev][i26], s) for c in chains]
                stat("2026 level", st.upper(), "tas", ssp, lev, "2026", s, lv, "degC")
                if lev == "anchored":
                    stat("anchored 2026 minus observed trend, absolute", st.upper(), "tas", ssp, lev, "2026", s,
                         np.abs(np.array(lv) - trend[s]), "degC")
                for a, b in ((2016, 2025), (2021, 2025)):
                    k = slice(a - 2016, b - 2016 + 1)
                    o = anomaly[s][a - ANCHOR[0]:b - ANCHOR[0] + 1].mean()
                    stat("scenario minus observed OTL", st.upper(), "tas", ssp, lev, f"{a}-{b}", s,
                         [season(dT[st, c, ssp, lev][k], s).mean() - o for c in chains], "degC")
            for s in ("year", "JJA"):
                stat("change 2081-2100", st.upper(), "tas", ssp, lev, "2081-2100", s,
                     [season(dT[st, c, ssp, lev][2081 - 2016:], s).mean() for c in chains], "degC")
        stat("P ratio, all months 2027-2100", st.upper(), "pr", ssp, "", "2027-2100", "month",
             np.array([Pr[st, c, ssp][out:] for c in chains]).ravel(), "ratio")
# 3 continuity: limit from above minus limit from below at every cut of the paths, absolute
for (st, var), pts in change.items():
    jumps = []
    for x, y in pts.values():
        assert (y > 0).all() or var == "tas"
        for xj in x[1:]:
            lo, hi = (at(x, y, np.array([xj + e]), log=var == "pr")[0] for e in (-1e-9, 1e-9))
            jumps += list(np.abs(hi - lo))
    stat("jump at the chain's warming levels (+-1e-9 degC), absolute", st.upper(), var, "", "", "GWL windows",
         "month", jumps, "degC" if var == "tas" else "ratio")
for ssp in SSP:
    t = np.concatenate([CENTRES[1:] - 0.25 - 1e-9, CENTRES[1:] - 0.25 + 1e-9])
    g = warming(ssp, t)
    stat("jump of global warming at the AR6 centres (+-1e-9 a), absolute", "", "", ssp, "", "2031 2051 2091", "year",
         np.abs(g[3:] - g[:3]), "degC")
    for st in ("otl", "roe"):
        steps = []
        for c in chains:
            off = dT[st, c, ssp, "anchored"][0] - dT[st, c, ssp, "native"][0]
            steps += [abs(off[a - 1] - off[b - 1]) for a, b in ((11, 12), (2, 3), (5, 6), (8, 9))]
        stat("anchor offset step between adjacent seasons (Nov-Dec, Feb-Mar, May-Jun, Aug-Sep), absolute",
             st.upper(), "tas", ssp, "anchored", "all years", "month", steps, "degC")
for s in ("year", "JJA"):
    o = [anomaly[s][a - ANCHOR[0]:b - ANCHOR[0] + 1].mean() for a, b in ((2016, 2025), (2021, 2025))]
    rows.append(("observed OTL minus 1991-2020", "OTL", "tas", "", "", "2016-2025, 2021-2025", s, 2,
                 f"{o[0]:.4f}", f"{o[1]:.4f}", "", "degC"))
for ssp in SSP:
    for y in (2026, 2050, 2100):
        rows.append(("global warming", "", "", ssp, "", y, "year", 1, f"{float(warming(ssp, y)):.4f}", "", "",
                     "degC above 1850-1900"))

(IGM / "results").mkdir(exist_ok=True)
with open(IGM / "results/scenarios_check.csv", "w", newline="") as f:
    csv.writer(f).writerows([("check", "station", "variable", "ssp", "level", "period", "season", "n", "median",
                              "min", "max", "unit"), *rows])
print(f"{len(chains)} chains ({n3} with GWL3.0); {len(dT) * len(YEARS) * 12} rows to igm/inputs/scenarios_monthly.csv")
for r in rows:
    print(" | ".join(map(str, r)))
