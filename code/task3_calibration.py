"""Task 3: calibrate the degree-day factor (DDF) so that the model reproduces the geodetic balance 1994-2021 of Task 1.

Model (assignment, main.tex eq. model and balance), monthly, on the fixed 2021 surface (dem_basodino_2021.asc inside
mask_basodino_2021.asc): T(z) = T_station - 6.5 degC/km (z - z_station); the month's precipitation is snow where
T(z) < 2 degC and is the same at every elevation (block model, no correction factor); melt = DDF x 30.42 d x max(T(z), 0).
Per hydrological year the glacier-wide accumulation C (m w.e.) and positive degree-day sum S (degC d) are averages over
the glacier, so the balance is B = C - DDF x S, linear in the DDF. Calibration over the 27 years October 1994 to
September 2021: sum(B) = B_total of Task 1 gives DDF = (sum C - B_total) / sum S, exactly, without a search.
Two approaches (Hint 3): "distributed", every 10 m cell; "lumped", one point at the mean glacier elevation.
Two targets: "specific", the rate of Task 1 (-0.78 m w.e./a over the mean area) as a balance of the 2021 glacier;
"mass", the same mass change spread over the 2021 area (rate x A_mean / A_2021).

Inputs: task2_meteo.csv, task1_geodetic.csv, the course grids. Outputs:
  code/task3_terms.csv   station, approach, hydrological year, C (m w.e.), S (degC d), 1935-2025
  code/task3_ddf.csv     station, approach, target, DDF (mm w.e. per degC d), ELA and AAR of the calibration period
  tex/figures/ddf_calibration{,_si}.pdf   (_si: kg m-2 s-1 K-1, kg m-2 s-1, m2) a: mean balance 1995-2021 against the DDF for the three stations; b: balance profile
                         of the report's station at its DDF; c: area of the 2021 glacier per 50 m band; 150 x 56 mm
usage:  python3 code/task3_calibration.py
"""
import csv
from collections import defaultdict

import numpy as np


from common import (ACCENT, CALIB, COURSE, GLACIER, GREY, INK, LAB, LAPSE, METEO, MM, ND, PALE_BLUE, RULE, STATION, STATIONS,
                    T0,
                    TM, YEARS, figure_style, panel_letter, read_csv, read_esri_ascii, save, write_csv)

z21, _ = read_esri_ascii(COURSE / "dem_basodino_2021.asc")
mask = read_esri_ascii(COURSE / "mask_basodino_2021.asc")[0] == 1
z = z21[mask]                                                      # 16065 glacier cells, 10 m
q1 = {r["quantity"]: float(r["value"]) for r in read_csv("task1_geodetic.csv")}
TARGET = {"specific": q1["B_rate"] * len(CALIB),                  # m w.e. over the 27 years
          "mass": q1["B_rate"] * q1["A_mean"] / q1["A_2021"] * len(CALIB)}
APPROACH = {"distributed": z, "lumped": np.array([z.mean()])}
z_station = {r["station_abbr"]: float(r["station_height_masl"]) for r in
             csv.DictReader(open(METEO / "ogd-nbcn_meta_stations.csv", encoding="latin-1"), delimiter=";")}

meteo = defaultdict(list)                                          # (station, hydro year) -> [(T, P), ...] Oct..Sep
for r in read_csv("task2_meteo.csv"):
    meteo[(r["station"], int(r["hydro_year"]))].append((float(r["T_degC"]), float(r["P_mm"])))


def terms(station, year, zz):
    """Accumulation (m w.e.) and positive degree days (degC d) of one hydrological year at elevations zz."""
    tp = np.array(meteo[(station, year)])                          # 12 x 2
    t = tp[:, :1] - LAPSE * (zz[None, :] - z_station[station])     # 12 x cells
    c = np.where(t < T0, tp[:, 1:] / 1000, 0.0).sum(axis=0)
    s = (ND * np.maximum(t - TM, 0.0)).sum(axis=0)
    return c, s


rows, ddf_rows, ddf = [], [], {}
for st in STATIONS:
    for ap, zz in APPROACH.items():
        cs = {y: terms(st, y, zz) for y in YEARS}
        rows += [(st, ap, y, c.mean(), s.mean()) for y, (c, s) in cs.items()]
        sum_c = sum(cs[y][0].mean() for y in CALIB)
        sum_s = sum(cs[y][1].mean() for y in CALIB)
        for target, b_total in TARGET.items():
            k = (sum_c - b_total) / sum_s                          # m w.e. per degC d
            ddf[(st, ap, target)] = k
            if ap == "distributed":                                # ELA, AAR of the mean balance profile 1995-2021
                b = sum(cs[y][0] - k * cs[y][1] for y in CALIB) / len(CALIB)
                order = np.argsort(zz)
                ela = np.interp(0.0, np.maximum.accumulate(b[order]), zz[order])
                aar = (b > 0).mean()
            else:
                ela, aar = np.nan, np.nan
            ddf_rows.append((st, ap, target, k * 1000, sum_c / len(CALIB), sum_s / len(CALIB), ela, aar))
write_csv("task3_terms.csv", ("station", "approach", "hydro_year", "C_mwe", "S_degCd"), rows)
write_csv("task3_ddf.csv", ("station", "approach", "target", "DDF_mm_per_degCd", "C_mean_mwe", "S_mean_degCd",
                            "ELA_m", "AAR"), ddf_rows)
for r in ddf_rows:
    print("{:4} {:11} {:8} DDF {:6.2f} mm/(degC d)  C {:5.2f} m w.e./a  S {:6.1f} degC d/a  ELA {:6.0f}  AAR {:4.2f}".format(*r))

# --- figure
plt = figure_style()
RATE = 1000 / (365.25 * 86400) * 1e5          # m w.e. a-1 -> 1e-5 kg m-2 s-1
VARIANTS = (("ddf_calibration", 1.0, 1.0, 1.0,
             {"x": "Degree-day factor (mm w.e. (°C d)⁻¹)", "y": "Balance (m w.e. a⁻¹)",
              "b": "Balance\n(m w.e. a⁻¹)", "z": "Elevation (m a.s.l.)", "a": "Area (km²)"}),
            ("ddf_calibration_si", 1e5 / 86400, RATE, 10.0,
             {"x": "Degree-day factor (10⁻⁵ kg m⁻² s⁻¹ K⁻¹)", "y": "Balance rate (10⁻⁵ kg m⁻² s⁻¹)",
              "b": "Balance rate\n(10⁻⁵ kg m⁻² s⁻¹)", "z": "Elevation (m)", "a": "Area (10⁵ m²)"}))
k = ddf[(STATION, "distributed", "specific")]
c_cal = sum(terms(STATION, y, z)[0] for y in CALIB) / len(CALIB)
s_cal = sum(terms(STATION, y, z)[1] for y in CALIB) / len(CALIB)
b_cell = c_cal - k * s_cal
bands = np.arange(np.floor(z.min() / 50) * 50, z.max() + 50, 50)
mid = (bands[:-1] + bands[1:]) / 2
idx = np.digitize(z, bands) - 1
b_band = np.array([b_cell[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(mid))])
area = np.array([(idx == i).sum() * 100 / 1e6 for i in range(len(mid))])
z_band = np.array([z[idx == i].mean() for i in range(len(mid))])   # mean elevation of the cells in each band
from matplotlib.ticker import MaxNLocator  # noqa: E402

colour = {"GRH": INK, "ANT": GLACIER, "OTL": ACCENT}
station_name = {"GRH": "Grimsel Hospiz", "ANT": "Andermatt", "OTL": "Locarno-Monti"}
ela = next(r[6] for r in ddf_rows if r[:3] == (STATION, "distributed", "specific"))
k_grid = np.linspace(0, 10, 50)                                       # mm w.e. per degC d
for name, fx, fy, fa, lab in VARIANTS:
    W, H = 150.0, 56.0
    fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white")
    axa = fig.add_axes([16 / W, 10 / H, 59 / W, 43 / H])
    axb = fig.add_axes([92 / W, 10 / H, 33 / W, 43 / H])
    axc = fig.add_axes([128 / W, 10 / H, 20 / W, 43 / H], sharey=axb)
    axa.set_xlim(0, 10 * fx), axa.set_ylim(-4.6 * fy, 2.0 * fy)
    axa.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    axa.axhline(fy * q1["B_rate"], color=GREY, lw=0.5, ls=(0, (3, 1.5)), zorder=1)
    axa.text(0.15 * fx, fy * q1["B_rate"], "geodetic rate", ha="left", va="top", color=GREY, fontsize=LAB)
    for st in STATIONS:
        r = next(r for r in ddf_rows if r[:3] == (st, "distributed", "specific"))
        line = lambda kk: fy * (r[4] - kk / 1000 * r[5])  # noqa: E731
        axa.plot(fx * k_grid, line(k_grid), color=colour[st], lw=1.1, zorder=2)
        axa.plot([fx * r[3]], [fy * q1["B_rate"]], "o", ms=3.4, color=colour[st], mec="white", mew=0.6, zorder=4)
        x0 = 8.4
        ang = np.degrees(np.arctan2(line(x0 + 0.5) - line(x0), fx * 0.5))
        axa.text(fx * x0, line(x0) - 0.12 * fy, station_name[st], color=colour[st], ha="center", va="top",
                 rotation=ang, rotation_mode="anchor", transform_rotates_text=True, fontsize=LAB)
    axa.set_xlabel(lab["x"], labelpad=1.5)
    axa.set_ylabel(lab["y"])

    axb.axvline(0, color=RULE, lw=0.4)
    axb.plot(fy * b_band, z_band, color=colour[STATION], lw=1.1)
    axb.axhline(ela, color=GREY, lw=0.5, ls=(0, (3, 1.5)))
    axb.text(axb.get_xlim()[0], ela, f"  zero balance, {ela:.0f} m", ha="left", va="bottom", color=GREY,
             fontsize=LAB)
    axb.xaxis.set_major_locator(MaxNLocator(4))
    axb.set_xlabel(lab["b"], labelpad=1.5)
    axb.set_ylabel(lab["z"])
    axc.barh(mid, fa * area, height=42, color=PALE_BLUE, lw=0)
    axc.xaxis.set_major_locator(MaxNLocator(3))
    axc.set_xlabel(lab["a"], labelpad=1.5)
    axc.tick_params(labelleft=False)
    panel_letter(fig, axa, "a", W)
    panel_letter(fig, axb, "b", W, 79.0)
    panel_letter(fig, axc, "c", W, 125.6)
    save(fig, name, "task3_calibration.py")
