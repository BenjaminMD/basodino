"""Daily against monthly mean temperatures: the two numbers of tex/minimal.tex that need daily data.

The homogenised daily mean temperature of Locarno-Monti (ths200d0), extrapolated to the glacier with the assignment's
decrease of 6.5 K per kilometre, against the monthly means of the same days:
  ratio   degree-days from daily temperatures over those from monthly means, mean over the cells of the 2021 glacier,
          hydrological years 1995-2021, each month counted with the assignment's 365/12 days (Question 3.3)
  june    melt at the top of the 2021 glacier (3177 m) with the calibrated factor, June 1936 and 1942: from the
          homogenised monthly mean (the model of Question 3.1) and from the daily temperatures of that June (Question 3.2)
Inputs: data/meteoswiss/ogd-nbcn_otl_d_historical.csv (MeteoSwiss open data, to 31 December 2024), task2_meteo.csv,
task3_ddf.csv, the 2021 grids. Output: code/daily_temperatures.csv.
usage:  python3 code/daily_temperatures.py
"""
import csv
from collections import defaultdict

import numpy as np

from common import CALIB, COURSE, LAPSE, METEO, ND, STATION, TM, read_csv, read_esri_ascii, write_csv

z21, _ = read_esri_ascii(COURSE / "dem_basodino_2021.asc")
z = z21[read_esri_ascii(COURSE / "mask_basodino_2021.asc")[0] == 1]
z_st = next(float(r["station_height_masl"]) for r in csv.DictReader(open(METEO / "ogd-nbcn_meta_stations.csv",
            encoding="latin-1"), delimiter=";") if r["station_abbr"] == STATION)
f_mm = next(float(r["DDF_mm_per_degCd"]) for r in read_csv("task3_ddf.csv")
            if (r["station"], r["approach"], r["target"]) == (STATION, "distributed", "specific"))
t_month = {(int(r["year"]), int(r["month"])): float(r["T_degC"]) for r in read_csv("task2_meteo.csv")
           if r["station"] == STATION}

days = defaultdict(list)                                         # (calendar year, month) -> daily means, degC
for r in csv.DictReader(open(METEO / f"ogd-nbcn_{STATION.lower()}_d_historical.csv"), delimiter=";"):
    if r["ths200d0"]:
        d, m, y = r["reference_timestamp"][:10].split(".")
        days[int(y), int(m)].append(float(r["ths200d0"]))

s_day = s_month = 0.0
for hy in CALIB:
    for y, m in [(hy - 1, m) for m in range(10, 13)] + [(hy, m) for m in range(1, 10)]:
        v = np.array(days[y, m])
        t = v[:, None] - LAPSE * (z[None, :] - z_st)                 # day x cell
        s_day += (ND / len(v) * np.maximum(t - TM, 0).sum(0)).mean()
        s_month += (ND * np.maximum(v.mean() - LAPSE * (z - z_st) - TM, 0)).mean()
ratio = s_day / s_month
print(f"degree-days {CALIB.start}-{CALIB.stop - 1}, 2021 glacier mean: daily {s_day / len(CALIB):.1f}, "
      f"monthly {s_month / len(CALIB):.1f} per year, ratio {ratio:.4f}")

rows = [("ratio_daily_monthly", "", ratio, "")]
dz = LAPSE * (z.max() - z_st)
for y in (1936, 1942):
    v = np.array(days[y, 6])
    melt_month = f_mm * ND * max(t_month[y, 6] - dz - TM, 0)
    melt_day = f_mm * np.maximum(v - dz - TM, 0).sum()
    print(f"June {y} at {z.max():.0f} m: monthly mean {t_month[y, 6] - dz + 273.15:.3f} K (daily data "
          f"{v.mean() - dz + 273.15:.3f} K), melt {melt_month:.1f} kg m-2 monthly, {melt_day:.0f} kg m-2 daily, "
          f"{(v - dz > TM).sum()} of {len(v)} days above the threshold")
    rows += [(f"june{y}_melt_monthly", y, melt_month, "kg m-2"), (f"june{y}_melt_daily", y, melt_day, "kg m-2")]
write_csv("daily_temperatures.csv", ("quantity", "year", "value", "unit"), rows)
