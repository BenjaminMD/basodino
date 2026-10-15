"""Check for F1 (DESIGN.md 'Fix round'): does the station transfer carry a trend error? The transfer (decision 5) adds a
constant monthly gradient to the Locarno-Monti (OTL) day values, so the glacier warms as fast as OTL. Robiei (ROE,
1898 m, 4 km from the glacier) measures the warming nearer the glacier height.

Per hydrological year 1991-2025 (October to September) and per season (DJF, MAM, JJA, SON): mean daily ROE, OTL
and ROE - OTL over the days both stations have, least-squares trend per decade and its standard error (ordinary,
independent years), and the number of day pairs. Precipitation the same way (decision 5 scales P_OTL by a constant
factor): the ratio of the ROE and OTL sums over the days both stations have, for the winter October to April and the
year, trend in per cent of its mean per decade.
Inputs (s02): raw/climate/obs/ogd-smn_{otl,roe}_d_{historical,recent}.csv (tre200d0 as redist.gradients, rre150d0).
Output: igm/results/f1_station_trend.csv (season, quantity, trend per decade (K, or per cent of the mean ratio),
standard error, years, mean pairs per year).
usage (s02, agl env):  python igm/checks/f1_station_trend.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redist import OBS  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
SEASONS = {"year": range(1, 13), "DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def main():
    t, p = {}, {}
    for st in ("otl", "roe"):
        for part in ("historical", "recent"):
            with open(OBS / f"ogd-smn_{st}_d_{part}.csv", encoding="latin-1") as f:
                for r in csv.DictReader(f, delimiter=";"):
                    d, v, q = r["reference_timestamp"][:10], r["tre200d0"], r["rre150d0"]
                    if v:
                        t.setdefault(d, {})[st] = float(v)
                    if q:
                        p.setdefault(d, {})[st] = float(q)
    rows = []
    for name, months in SEASONS.items():
        per = {}
        for d, v in t.items():
            y, m = int(d[6:10]), int(d[3:5])
            hy = y + (m >= 10)
            if len(v) == 2 and m in months and 1991 <= hy <= 2025:
                per.setdefault(hy, []).append((v["roe"], v["otl"]))
        ys = np.array(sorted(per))
        a = np.array([np.mean(per[y], axis=0) for y in ys])
        n = np.mean([len(per[y]) for y in ys])
        for q, series in (("ROE", a[:, 0]), ("OTL", a[:, 1]), ("ROE - OTL", a[:, 0] - a[:, 1])):
            X = np.c_[np.ones(len(ys)), ys - ys.mean()]
            coef, res = np.linalg.lstsq(X, series, rcond=None)[:2]
            se = np.sqrt(res[0] / (len(ys) - 2) / ((ys - ys.mean()) ** 2).sum())
            rows.append((name, q, f"{coef[1] * 10:+.3f}", f"{se * 10:.3f}", len(ys), f"{n:.0f}"))
    for name, months in (("winter October to April", (10, 11, 12, 1, 2, 3, 4)), ("year", range(1, 13))):
        per = {}
        for d, v in p.items():
            y, m = int(d[6:10]), int(d[3:5])
            hy = y + (m >= 10)
            if len(v) == 2 and m in months and 1991 <= hy <= 2025:
                per.setdefault(hy, []).append((v["roe"], v["otl"]))
        ys = np.array(sorted(per))
        s = np.array([np.sum(per[y], axis=0) for y in ys])
        ratio = s[:, 0] / s[:, 1]
        X = np.c_[np.ones(len(ys)), ys - ys.mean()]
        coef, res = np.linalg.lstsq(X, ratio, rcond=None)[:2]
        se = np.sqrt(res[0] / (len(ys) - 2) / ((ys - ys.mean()) ** 2).sum())
        rows.append((f"P {name}", f"ROE / OTL sum, mean {ratio.mean():.3f}", f"{coef[1] * 10 / ratio.mean() * 100:+.2f}",
                     f"{se * 10 / ratio.mean() * 100:.2f}", len(ys), f"{np.mean([len(per[y]) for y in ys]):.0f}"))
    with open(RES / "f1_station_trend.csv", "w", newline="") as f:
        csv.writer(f).writerows([("season", "quantity", "trend_per_decade", "standard_error", "years",
                                  "mean_day_pairs_per_year"), *rows])
    for r in rows:
        print(*r, sep="\t")


if __name__ == "__main__":
    main()
