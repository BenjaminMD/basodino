"""June at the top of the glacier, 1935-2025: the example of Question 3.2 against the degree-day factor.

Modelled June mean air temperature at the highest cell of the 2021 glacier (3177 m above sea level), from the
Locarno-Monti series with the assignment's decrease of 6.5 K per kilometre, and the June melt the calibrated model
gives there, F x tau x max(T - 273.15 K, 0). Months at or below 273.15 K melt nothing, whatever the sun does.
Inputs: task2_meteo.csv, task3_ddf.csv, the 2021 grids. Output: tex/figures/june_summit_si.pdf (150 x 34 mm, kelvin);
the count of Junes without melt and a few June melts are printed.
usage:  python3 code/june_summit.py
"""
import csv

import numpy as np

from common import (ACCENT, COURSE, GLACIER, GREY, LAB, LAPSE, METEO, MM, ND, RULE, STATION, YEARS, figure_style,
                    read_csv, read_esri_ascii, save)

T_MELT = 273.15                                                  # K
z21, _ = read_esri_ascii(COURSE / "dem_basodino_2021.asc")
top = z21[read_esri_ascii(COURSE / "mask_basodino_2021.asc")[0] == 1].max()
z_st = next(float(r["station_height_masl"]) for r in csv.DictReader(open(METEO / "ogd-nbcn_meta_stations.csv",
            encoding="latin-1"), delimiter=";") if r["station_abbr"] == STATION)
f_mm = next(float(r["DDF_mm_per_degCd"]) for r in read_csv("task3_ddf.csv")
            if (r["station"], r["approach"], r["target"]) == (STATION, "distributed", "specific"))
t_june = {int(r["year"]): float(r["T_degC"]) + T_MELT - LAPSE * (top - z_st) for r in read_csv("task2_meteo.csv")
          if r["station"] == STATION and r["month"] == "6"}
years = np.array(list(YEARS))
t = np.array([t_june[y] for y in years])
melt = f_mm * ND * np.maximum(t - T_MELT, 0.0)                  # mm w.e. = kg m-2
cold = t <= T_MELT
print(f"top {top:.0f} m: {cold.sum()} of {len(years)} Junes at or below {T_MELT} K; mean June melt {melt.mean():.0f} kg m-2; "
      f"melt per kelvin above the threshold {f_mm * ND:.0f} kg m-2")
for y in (1936, 1942, 2003):
    print(f"June {y}: {t_june[y]:.3f} K, melt {f_mm * ND * max(t_june[y] - T_MELT, 0):.1f} kg m-2")

plt = figure_style()
W, H = 150.0, 34.0
fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white")
ax = fig.add_axes([16 / W, 1 - 26 / H, 132 / W, 24 / H])
ax.axhline(T_MELT, color=GREY, lw=0.5, ls=(0, (3, 1.5)), zorder=1)
ax.plot(years, t, color=RULE, lw=0.5, zorder=2)
ax.plot(years[~cold], t[~cold], "o", ms=2.5, color=ACCENT, lw=0, zorder=3)
ax.plot(years[cold], t[cold], "o", ms=2.5, color=GLACIER, lw=0, zorder=3)
ax.text(2025.5, T_MELT - 0.2, "no melt", ha="right", va="top", color=GLACIER, fontsize=LAB)
ax.text(2025.5, T_MELT + 0.2, "melt", ha="right", va="bottom", color=ACCENT, fontsize=LAB)
ax.set_ylabel("Temperature (K)")
ax.set_xlabel("Year", labelpad=1.5)
ax.set_xlim(1933.5, 2026.5), ax.set_ylim(t.min() - 0.5, t.max() + 0.5)
save(fig, "june_summit_si", "june_summit.py")
