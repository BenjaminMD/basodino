"""Task 2: homogenised monthly temperature and precipitation 1935-2025 of the candidate stations, and their fit to the glacier.

Inputs: data/meteoswiss/ogd-nbcn_{grh,ant,otl}_m.csv (MeteoSwiss Open Government Data, homogeneous monthly series of
the Swiss National Basic Climatological Network, downloaded 2026-09-23 from https://data.geo.admin.ch/, path
ch.meteoschweiz.ogd-nbcn/<abbr>/ogd-nbcn_<abbr>_m.csv; ths200m0 = air temperature 2 m,
homogeneous monthly mean, degC; rhs150m0 = precipitation, homogeneous monthly total, mm) and ogd-nbcn_meta_stations.csv;
the glaciological balances of GLAMOS (common.read_glamos) for the check.
Outputs:
  code/task2_meteo.csv       station, hydrological year, calendar year, month, T (degC), P (mm), October 1934 to September 2025
  code/task2_stations.csv    per station: elevation, distance and side of the main Alpine divide, 1991-2020 means, and the
                             correlation of its winter precipitation (October-April) with the GLAMOS winter balance and of
                             its May-September temperature with the GLAMOS summer balance, 1992-2025
  tex/figures/meteo_1935_2025.pdf   the station of the report (common.STATION): monthly and annual temperature,
                             annual precipitation; 150 x 50 mm; _si in kelvin and kg m-2

usage:  python3 code/task2_meteo.py
"""
import csv

import numpy as np

from common import (INK, METEO, MM, PALE_BLUE, RULE, STATION, STATIONS, YEARS, figure_style, panel_letter,
                    read_glamos, save, write_csv)

GLACIER_EN = (2679622, 1141059)                                     # LV95, glacier centre
SIDE = {"GRH": "north", "ANT": "north", "OTL": "south"}            # of the main Alpine divide; the glacier lies south

meta = {r["station_abbr"]: r for r in csv.DictReader(open(METEO / "ogd-nbcn_meta_stations.csv", encoding="latin-1"),
                                                      delimiter=";")}
series, rows = {}, []
for st in STATIONS:
    data = {}
    with open(METEO / f"ogd-nbcn_{st.lower()}_m.csv", encoding="latin-1") as f:
        for r in csv.DictReader(f, delimiter=";"):
            _, month, year = r["reference_timestamp"].split()[0].split(".")
            year, month = int(year), int(month)
            hy = year + (month >= 10)                                 # hydrological year
            if hy in YEARS:
                data[(hy, year, month)] = (float(r["ths200m0"]), float(r["rhs150m0"]))
    assert len(data) == 12 * len(YEARS), f"{st}: {12 * len(YEARS) - len(data)} months missing"
    series[st] = data
    rows += [(st, hy, y, m, t, p) for (hy, y, m), (t, p) in sorted(data.items())]
write_csv("task2_meteo.csv", ("station", "hydro_year", "year", "month", "T_degC", "P_mm"), rows)

glamos = read_glamos()
table = []
for st in STATIONS:
    d, m = series[st], meta[st]
    e, n = float(m["station_coordinates_lv95_east"]), float(m["station_coordinates_lv95_north"])
    yrs = sorted(y for y in glamos if y in YEARS)
    p_win = [sum(p for (hy, _, mo), (t, p) in d.items() if hy == y and mo in (10, 11, 12, 1, 2, 3, 4)) for y in yrs]
    t_sum = [np.mean([t for (hy, _, mo), (t, p) in d.items() if hy == y and 5 <= mo <= 9]) for y in yrs]
    norm = [(t, p) for (hy, _, _), (t, p) in d.items() if 1991 <= hy <= 2020]
    table.append((st, m["station_name"], float(m["station_height_masl"]), round(np.hypot(e - GLACIER_EN[0],
                  n - GLACIER_EN[1]) / 1e3, 1), SIDE[st], np.mean([t for t, _ in norm]), sum(p for _, p in norm) / 30,
                  np.corrcoef(p_win, [glamos[y][0] for y in yrs])[0, 1],
                  np.corrcoef(t_sum, [glamos[y][1] for y in yrs])[0, 1], len(yrs)))
write_csv("task2_stations.csv", ("station", "name", "elevation_m", "distance_km", "side", "T_1991_2020_degC",
                                 "P_1991_2020_mm_per_a", "r_Pwinter_Bw", "r_Tsummer_Bs", "n_glamos_years"), table)
for r in table:
    print("{:4} {:16} {:5.0f} m {:5.1f} km {:5}  T {:5.2f}  P {:5.0f}  r(Pw,Bw) {:5.2f}  r(Ts,Bs) {:5.2f}  n {}".format(*r))

# --- figure: the report's station, in the report's units and in SI (kelvin, kg m-2)
plt = figure_style()
d = series[STATION]
months = sorted(d)
yrs = np.array(list(YEARS))
t_year = np.array([np.mean([d[k][0] for k in months if k[0] == y]) for y in YEARS])
p_year = np.array([sum(d[k][1] for k in months if k[0] == y) for y in YEARS])
run = lambda v: np.convolve(v, np.ones(11) / 11, mode="valid")    # centred 11-year running mean, 1940-2020  # noqa: E731

for name, dt, t_label, p_label in (("meteo_1935_2025", 0.0, "Temperature (°C)", "Precipitation\n(mm a⁻¹)"),
                                   ("meteo_1935_2025_si", 273.15, "Temperature (K)", "Precipitation\n(kg m⁻²)")):
    W, H = 150.0, 50.0
    fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white")
    ax1 = fig.add_axes([16 / W, 1 - 22 / H, 132 / W, 20 / H])
    ax2 = fig.add_axes([16 / W, 1 - 43 / H, 132 / W, 17 / H], sharex=ax1)
    ax1.plot(yrs, t_year + dt, color=RULE, lw=0.6)
    ax1.plot(yrs[5:-5], run(t_year) + dt, color=INK, lw=1.1)
    ax1.set_ylim(np.floor(t_year.min() + dt) - 0.2, np.ceil(t_year.max() + dt) + 0.2)
    ax1.set_ylabel(t_label)
    ax1.tick_params(labelbottom=False)
    ax2.bar(yrs, p_year, width=0.72, color=PALE_BLUE, lw=0)
    ax2.plot(yrs[5:-5], run(p_year), color=INK, lw=1.1)
    ax2.set_yticks(range(0, 4000, 1000))
    ax2.set_ylabel(p_label)
    ax2.set_xlabel("Hydrological year", labelpad=1.5)
    ax2.set_xlim(1933.5, 2026.5)
    for ax in (ax1, ax2):
        ax.yaxis.set_label_coords(-0.058, 0.5)
    for ax, letter in ((ax1, "a"), (ax2, "b")):
        panel_letter(fig, ax, letter, W)
    save(fig, name, "task2_meteo.py")
print(f"{STATION}: T {t_year.mean():.2f} degC, P {p_year.mean():.0f} mm a-1 (1935-2025); "
      f"T trend {np.polyfit(list(YEARS), t_year, 1)[0] * 10:.2f} K per decade")
