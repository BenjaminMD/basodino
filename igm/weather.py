"""Daily weather at Locarno-Monti (OTL, 367 m), hydrological years 1851-2026: decision 6 of igm/DESIGN.md.

Month by month from October 1850 to September 2026, 29 February left out first (365-day years):
  T  1882-12 to 2024-12  homogenised daily mean (ths200d0) shifted to the homogenised monthly mean (ths200m0)
     2025-01 to 2026-09  measured daily mean (tre200d0) shifted to the homogenised monthly mean
     earlier             monthly mean rebuilt from Lugano (1864-01 to 1882-11) or Grand St-Bernard (1850-10 to 1863-12)
  P  1900-10 to 2026-09  measured daily total (rre150d0, 6 to 6 UTC) scaled to the homogenised monthly total (rhs150m0)
     1883-01 to 1900-09  homogenised monthly total
     1864-01 to 1882-12  monthly total rebuilt from Lugano
     earlier             the months of a hydrological year 1865-1894 of this series, the 14 years 1851-1864 drawn
                         without replacement
Rebuilt month: per calendar month, the homogenised monthly OTL values fitted by least squares on the station's over
1883-1930, OTL = mean_OTL + slope (station - mean_station); a negative P would be set to 0.
A month without measured daily values takes the daily anomalies of the same month in a random year 1901-1930 with
precipitation in that month: homogenised daily T minus its monthly mean, added to the month's mean; measured daily P
over its monthly total, times the month's total. T and P of a month share the year. Draws: numpy default_rng(SEED),
the 14 hydrological years for P first, then the months in time order.
Inputs, s02 raw/climate/obs (MeteoSwiss open data, fetched by igm/fetch_climate.sh): ogd-nbcn_{otl,lug,gsb}_m.csv,
ogd-nbcn_otl_d_historical.csv (to 2024-12-31), ogd-smn_otl_d_{historical,recent}.csv.
Outputs:
  igm/inputs/weather_otl_daily.csv  hydro_year, day (1 = 1 October, 365 = 30 September), date, T_degC, P_mm (per day);
                                    176 x 365 rows in time order, so a column reshapes to (year, day)
  igm/results/weather_monthly.csv   per month: mean T, total P, where each came from, the random hydrological year
                                    lending P (before 1864) and the random year lending the daily anomalies, the shift
                                    of the daily T onto the monthly mean (from 1882-12) and the scale factor of the
                                    measured P
  igm/results/weather_fits.csv      per station, variable and calendar month: n, slope, r and RMSE of the rebuilt OTL
                                    1883-1930, the two means
usage (s02, agl env):  python igm/weather.py
"""
import csv
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]                      # s02:/data/befahl/projects/AGL/basodino
OBS, IGM = ROOT / "raw/climate/obs", ROOT / "igm"
YEARS = range(1851, 2027)                                       # hydrological year y: 1 October y-1 to 30 September y
MONTHS = (10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9)
FIT, PATTERN, P_YEARS = range(1883, 1931), range(1901, 1931), range(1865, 1895)
SEED = 1850


def read(name, *cols):
    """MeteoSwiss open-data file -> {(year, month, day): [values of cols]}, empty cells NaN."""
    with open(OBS / name, encoding="latin-1") as f:
        return {tuple(map(int, r["reference_timestamp"][:10].split(".")[::-1])):
                [float(r[c]) if r[c] else np.nan for c in cols] for r in csv.DictReader(f, delimiter=";")}


def days(y, m):
    """The days of a month as (year, month, day), 29 February left out."""
    n = (date(y + m // 12, m % 12 + 1, 1) - date(y, m, 1)).days
    return [(y, m, d) for d in range(1, n + 1) if (m, d) != (2, 29)]


def write(name, header, rows):
    with open(IGM / name, "w", newline="") as f:
        csv.writer(f).writerows([header, *rows])


monthly = {st: {k[:2]: v for k, v in read(f"ogd-nbcn_{st.lower()}_m.csv", "ths200m0", "rhs150m0").items()}
           for st in ("OTL", "LUG", "GSB")}                     # (year, month) -> [T degC, P mm]
hom = read("ogd-nbcn_otl_d_historical.csv", "ths200d0")
meas = read("ogd-smn_otl_d_historical.csv", "tre200d0", "rre150d0") | read("ogd-smn_otl_d_recent.csv", "tre200d0",
                                                                           "rre150d0")

fits, fit_rows = {}, []                                         # (station, variable, month) -> (mean OTL, slope, mean)
for st, var in (("LUG", "T"), ("GSB", "T"), ("LUG", "P")):
    i = "TP".index(var)
    for m in range(1, 13):
        x, o = (np.array([monthly[s][y, m][i] for y in FIT]) for s in (st, "OTL"))
        b = ((x - x.mean()) * (o - o.mean())).mean() / x.var()
        fits[st, var, m] = (o.mean(), b, x.mean())
        rmse = np.sqrt(((o.mean() + b * (x - x.mean()) - o) ** 2).mean())
        fit_rows.append((st, var, m, len(x), f"{b:.4f}", f"{np.corrcoef(x, o)[0, 1]:.4f}", f"{rmse:.4f}",
                         f"{o.mean():.3f}", f"{x.mean():.3f}", "degC" if var == "T" else "mm per month"))


def rebuilt(st, var, y, m):
    o, b, x = fits[st, var, m]
    return o + b * (monthly[st][y, m]["TP".index(var)] - x)


def p_month(y, m):
    """Monthly P at OTL from 1864-01: homogenised from 1883-01, before rebuilt from Lugano."""
    return monthly["OTL"][y, m][1] if y >= 1883 else max(rebuilt("LUG", "P", y, m), 0.0)


wet = {m: [s for s in PATTERN if sum(meas[k][1] for k in days(s, m)) > 0] for m in range(1, 13)}
rng = np.random.default_rng(SEED)
p_year = dict(zip(range(YEARS[0], 1865), map(int, rng.choice(P_YEARS, 1865 - YEARS[0], replace=False))))
daily, rows, dry, clipped = [], [], 0, 0
for hy in YEARS:
    for m in MONTHS:
        y = hy - (m >= 10)
        src = int(rng.choice(wet[m])) if (y, m) < (1900, 10) else ""       # lends its daily anomalies
        shift = factor = sy = ""
        if (y, m) < (1882, 12):
            a = np.array([hom[k][0] for k in days(src, m)])
            st = "LUG" if y >= 1864 else "GSB"
            t, t_from = rebuilt(st, "T", y, m) + a - a.mean(), f"{st.lower()}_fit"
        else:
            a = np.array([(hom if y < 2025 else meas)[k][0] for k in days(y, m)])
            shift = monthly["OTL"][y, m][0] - a.mean()
            t = a + shift
            t_from = "otl_daily_homogenised_shifted" if y < 2025 else "otl_daily_measured_shifted"
        if src == "":
            a = np.array([meas[k][1] for k in days(y, m)])
            total = monthly["OTL"][y, m][1]
            assert a.sum() > 0 or total == 0, f"{y}-{m}: measured P 0, homogenised {total}"
            factor = total / a.sum() if a.sum() > 0 else 1.0
            p, p_from = a * factor, "otl_daily_measured_scaled"
        else:
            if y < 1864:
                sy = p_year[hy]
                total, p_from = p_month(sy - (m >= 10), m), "random_year"
            else:
                total, p_from = p_month(y, m), "otl_monthly_homogenised" if y >= 1883 else "lug_fit"
                clipped += y < 1883 and rebuilt("LUG", "P", y, m) < 0
            a = np.array([meas[k][1] for k in days(src, m)])
            p = total * a / a.sum()
        assert np.ptp(t) > 0 and (np.ptp(p) > 0 or p.sum() == 0), f"{y}-{m} spread evenly"
        dry += p.sum() == 0
        daily += [(hy, date(*k).isoformat(), f"{tv:.3f}", f"{pv:.3f}") for k, tv, pv in zip(days(y, m), t, p)]
        rows.append((hy, y, m, f"{t.mean():.3f}", f"{p.sum():.2f}", t_from, p_from, sy, src,
                     shift if shift == "" else f"{shift:.3f}", factor if factor == "" else f"{factor:.4f}"))

dates = [r[1] for r in daily]
assert set(Counter(r[0] for r in daily).values()) == {365} and len(set(dates)) == len(dates) == 365 * len(YEARS)
assert (dates[0], dates[-1]) == ("1850-10-01", "2026-09-30") and not any(d.endswith("-02-29") for d in dates)
T, P = (np.array([float(r[i]) for r in daily]).reshape(len(YEARS), 365) for i in (2, 3))
assert np.isfinite(T).all() and np.isfinite(P).all() and (P >= 0).all()
for d in ("inputs", "results"):
    (IGM / d).mkdir(exist_ok=True)
write("inputs/weather_otl_daily.csv", ("hydro_year", "day", "date", "T_degC", "P_mm"),
      [(hy, i % 365 + 1, *r) for i, (hy, *r) in enumerate(daily)])
write("results/weather_monthly.csv", ("hydro_year", "year", "month", "T_degC", "P_mm", "T_from", "P_from",
                                      "P_hydro_year", "anomaly_year", "T_shift_degC", "P_factor"), rows)
write("results/weather_fits.csv", ("station", "variable", "month", "n", "slope", "r", "rmse", "mean_otl",
                                   "mean_station", "unit"), fit_rows)

print(f"{len(daily)} days, {dates[0]} to {dates[-1]}, no 29 February; no month spread evenly; {dry} months without "
      f"precipitation; rebuilt P below 0 set to 0 in {clipped} months")
for st, var in (("LUG", "T"), ("GSB", "T"), ("LUG", "P")):
    f = np.array([r[4:7] for r in fit_rows if r[:2] == (st, var)], float)
    print(f"fit {st} {var} 1883-1930: slope {f[:, 0].min():.2f} to {f[:, 0].max():.2f}, r {f[:, 1].min():.3f} to "
          f"{f[:, 1].max():.3f}, RMSE {f[:, 2].min():.2f} to {f[:, 2].max():.2f} {'degC' if var == 'T' else 'mm'}")
for a, b, what in ((1882, 2024, "homogenised"), (2025, 2026, "measured")):
    s = np.array([float(r[9]) for r in rows if r[9] != "" and a <= r[1] <= b])
    print(f"{what} daily T {a}-{b}: shifted onto the homogenised monthly mean by {s.min():+.3f} to {s.max():+.3f} "
          f"degC (mean {s.mean():+.3f}), {np.sum(np.abs(s) > 0.05)} of {len(s)} months beyond 0.05")
f = [float(r[10]) for r in rows if r[1] >= 2025]
print(f"2025-01 to 2026-09: measured P scaled by {min(f):.4f} to {max(f):.4f}")
pool = P[P_YEARS[0] - YEARS[0]:P_YEARS[-1] - YEARS[0] + 1].sum(1)
print(f"P of 1851-1864 from the hydrological years {[p_year[hy] for hy in sorted(p_year)]}; the pool 1865-1894 has "
      f"{pool.mean():.0f} mm per year, standard deviation {pool.std(ddof=1):.0f}")
for a, b in ((1851, 1863), (1864, 1882), (1883, 1930), (1931, 1990), (1991, 2020), (2021, 2026)):
    k = slice(a - YEARS[0], b - YEARS[0] + 1)
    print(f"hydrological years {a}-{b}: T {T[k].mean():5.2f} degC, P {P[k].sum(1).mean():5.0f} mm per year")
