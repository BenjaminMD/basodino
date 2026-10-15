"""Change signals of the station-scale climate scenarios in raw/climate/ (see scripts/fetch_climate.sh), as one CSV.

Inputs, under raw/climate/:
  ch2018/climate-ch2018-climate-normals-data_{tas,pr}_{OTL,GRH,CIM}.csv   CH2018 web atlas, monthly 30-year means
  ch2025/ogd-climate-scenarios-ch2025_{otl,roe}_{tas,pr}_{ref91-20,gwl*}.csv   CH2025 DAILY-LOCAL, 30 years per period
  obs/ogd-nbcn_otl_m.csv        homogenised monthly series of Locarno / Monti (OTL)
  obs/ogd-smn_roe_d_historical.csv   measured daily series of Robiei (ROE), from 1991
Output: derived/climate_summary.csv with columns
  source, station, variable, scenario, period, season, member, value, unit, reference
  CH2018: change of the 30-year mean against the observed 1981-2010 mean, member lower / medium / upper estimate (5, 50,
    95 % of the simulations). Seasons are means of the monthly values, so their lower and upper are approximate.
  CH2025: change of each chain's 30-year mean against the same chain's 1991-2020 (model reference), member = chain,
    and n, median, min, max over the chains; for OTL also median_vs_1981-2010 = median plus the observed shift of the
    OTL mean from 1981-2010 to 1991-2020 (homogenised monthly series).
  obs and ref: absolute 30-year means (observed, and the CH2025 chain median of 1991-2020) as a bias check.
  tas: degC, changes in degC; pr: mm per year or season, changes in percent.
usage:  ~/miniconda3/envs/agl/bin/python scripts/climate_summary.py   (from the project directory)
"""
import csv
from pathlib import Path

import pandas as pd

R, OUT = Path("raw/climate"), Path("derived/climate_summary.csv")
SEASONS = {"year": range(1, 13), "DJF": (12, 1, 2), "JJA": (6, 7, 8)}
DAYS = dict(zip(range(1, 13), (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)))
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
rows = []


def change(var, new, ref):
    return new - ref if var == "tas" else 100 * (new / ref - 1)


def season_value(var, monthly):  # monthly: {month: mean daily value}; tas -> mean, pr -> total mm
    return lambda ms: (sum(monthly[m] for m in ms) / len(ms) if var == "tas"
                       else sum(monthly[m] * DAYS[m] for m in ms))


# CH2018 web atlas: 12 month rows, column 1 observations 1981-2010, then period x RCP x estimate
for f in sorted(R.glob("ch2018/*.csv")):
    var, st = f.stem.split("_")[1:]
    lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(";observations"))
    per, rcp, est = (lines[i + j].strip().split(";") for j in range(3))
    vals = {l.split(";")[0].strip('"'): [float(x) for x in l.split(";")[1:]] for l in lines[i + 3:i + 15]}
    obs = season_value(var, {MONTHS.index(m) + 1: v[0] for m, v in vals.items()})
    for c in range(2, len(per)):
        new = season_value(var, {MONTHS.index(m) + 1: v[c - 1] for m, v in vals.items()})
        for s, ms in SEASONS.items():
            rows.append(("CH2018 web atlas", st, var, rcp[c], per[c], s, est[c].split()[0], change(var, new(ms), obs(ms)),
                         "degC" if var == "tas" else "percent", "obs 1981-2010"))
    for s, ms in SEASONS.items():
        rows.append(("CH2018 web atlas", st, var, "obs", "1981-2010", s, "obs", obs(ms),
                     "degC" if var == "tas" else "mm", "-"))

# observed shift of the OTL mean 1981-2010 -> 1991-2020, homogenised monthly series
m = pd.read_csv(R / "obs/ogd-nbcn_otl_m.csv", sep=";", encoding="latin-1")
m["year"], m["month"] = m.reference_timestamp.str[6:10].astype(int), m.reference_timestamp.str[3:5].astype(int)
shift = {}
for var, col in (("tas", "ths200m0"), ("pr", "rhs150m0")):
    for s, ms in SEASONS.items():
        clim = {}
        for a, b in ((1981, 2010), (1991, 2020)):
            x = m[m.year.between(a, b) & m.month.isin(ms)][col]
            clim[a] = x.mean() if var == "tas" else x.mean() * len(ms)
            rows.append(("obs ogd-nbcn", "OTL", var, "obs", f"{a}-{b}", s, "obs", clim[a], "degC" if var == "tas" else "mm", "-"))
        shift[var, s] = clim[1991] - clim[1981] if var == "tas" else clim[1991] / clim[1981]

# observed ROE 1991-2020, measured daily series
d = pd.read_csv(R / "obs/ogd-smn_roe_d_historical.csv", sep=";", encoding="latin-1")
d["year"], d["month"] = d.reference_timestamp.str[6:10].astype(int), d.reference_timestamp.str[3:5].astype(int)
d = d[d.year.between(1991, 2020)]
for var, col in (("tas", "tre200d0"), ("pr", "rre150d0")):
    mon = d.groupby("month")[col].mean()
    v = season_value(var, mon.to_dict())
    for s, ms in SEASONS.items():
        rows.append(("obs ogd-smn", "ROE", var, "obs", "1991-2020", s, "obs", v(ms), "degC" if var == "tas" else "mm", "-"))

# CH2025 DAILY-LOCAL: per chain, change against its own 1991-2020
for st in ("otl", "roe"):
    for var in ("tas", "pr"):
        read = lambda p: pd.read_csv(R / f"ch2025/ogd-climate-scenarios-ch2025_{st}_{var}_{p}.csv", sep=";",
                                     skiprows=23, na_values="NA")
        ref = read("ref91-20")
        ref["month"] = ref.DATE.str[5:7].astype(int)
        refm = ref.groupby("month").mean(numeric_only=True)
        for s, ms in SEASONS.items():
            r = refm.apply(lambda col: season_value(var, col.to_dict())(ms))
            rows.append(("CH2025 DAILY-LOCAL", st.upper(), var, "ref91-20", "1991-2020 (model)", s, "median", r.median(),
                         "degC" if var == "tas" else "mm", "-"))
        for g in ("gwl1.5", "gwl2.0", "gwl2.5", "gwl3.0"):
            x = read(g)
            x["month"] = x.DATE.str[5:7].astype(int)
            xm = x.groupby("month").mean(numeric_only=True)
            for s, ms in SEASONS.items():
                new = xm.apply(lambda col: season_value(var, col.to_dict())(ms))
                old = refm[new.index].apply(lambda col: season_value(var, col.to_dict())(ms))
                dc = change(var, new, old)
                base = ("CH2025 DAILY-LOCAL", st.upper(), var, g.upper(), "30 years at the GWL", s)
                unit, ref_txt = ("degC" if var == "tas" else "percent"), "model 1991-2020 (same chain)"
                rows += [base + (c, dc[c], unit, ref_txt) for c in dc.index]
                rows += [base + (k, v, unit, ref_txt) for k, v in (("n", len(dc)), ("median", dc.median()),
                                                                   ("min", dc.min()), ("max", dc.max()))]
                if st == "otl":
                    med = (dc.median() + shift[var, s] if var == "tas"
                           else 100 * ((1 + dc.median() / 100) * shift[var, s] - 1))
                    rows.append(base + ("median_vs_1981-2010", med, unit, "model 1991-2020 + observed shift"))

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(("source", "station", "variable", "scenario", "period", "season", "member", "value", "unit", "reference"))
    w.writerows(r[:7] + (round(float(r[7]), 3),) + r[8:] for r in rows)
print(f"{OUT}: {len(rows)} rows")
