"""Every number of tex/minimal.tex in SI units, next to the same number in the units of the assignment.

SI: masses per area in kg m-2, rates per second, temperature differences in K, lengths, areas and volumes in m, m2
and m3, the degree-day factor as a mass flux per kelvin (kg m-2 s-1 K-1) and as a heat-transfer coefficient (W m-2 K-1,
times the latent heat of fusion). The assignment's units: m (or mm) water equivalent, per year (365.25 days), per day,
degree-days, km and km3. Water equivalent: 1 m = 1000 kg m-2 (density of water 1000 kg m-3).

Inputs: task1_geodetic.csv, task2_stations.csv, task3_ddf.csv, task3_terms.csv, task4_summary.csv, task4_balance.csv,
task5_projection.csv, all from the task scripts. Output: code/si_units.csv (quantity, SI value, SI unit, value, unit).
usage:  python3 code/si_units.py
"""
import numpy as np

from common import STATION, read_csv, read_glamos, write_csv

RHO_W, L_F = 1000.0, 3.34e5                       # kg m-3, J kg-1
DAY, YEAR = 86400.0, 365.25 * 86400.0             # s
MONTH = 365 / 12 * DAY                            # s, the assignment's month

q1 = {r["quantity"]: float(r["value"]) for r in read_csv("task1_geodetic.csv")}
q5 = {r["quantity"]: float(r["value"]) for r in read_csv("task5_projection.csv")}
q4 = {r["quantity"]: r["value"] for r in read_csv("task4_summary.csv") if r["station"] == STATION}
ddf = {r["station"]: float(r["DDF_mm_per_degCd"]) for r in read_csv("task3_ddf.csv")
       if (r["approach"], r["target"]) == ("distributed", "specific")}
cal = next(r for r in read_csv("task3_ddf.csv") if (r["station"], r["approach"], r["target"]) == (STATION, "distributed", "specific"))
grh = next(r for r in read_csv("task3_ddf.csv") if (r["station"], r["approach"], r["target"]) == ("GRH", "distributed", "specific"))
b = {int(r["hydro_year"]): float(r["B_mwe"]) for r in read_csv("task4_balance.csv") if r["station"] == STATION}
glamos = read_glamos()

rate = lambda mwea: mwea * RHO_W / YEAR           # m w.e. a-1 -> kg m-2 s-1  # noqa: E731
mass = lambda mwe: mwe * RHO_W                    # m w.e. -> kg m-2  # noqa: E731
flux = lambda mm_cd: mm_cd / DAY                  # mm w.e. (degC d)-1 -> kg m-2 s-1 K-1  # noqa: E731

rows = [
    # Question 1
    ("dV", q1["dV"] * 1e9, "m3", q1["dV"], "km3"),
    ("A_1994", q1["A_1994"] * 1e6, "m2", q1["A_1994"], "km2"),
    ("A_2021", q1["A_2021"] * 1e6, "m2", q1["A_2021"], "km2"),
    ("A_mean", q1["A_mean"] * 1e6, "m2", q1["A_mean"], "km2"),
    ("dt", 27 * YEAR, "s", 27, "a"),
    ("B_rate", rate(q1["B_rate"]), "kg m-2 s-1", q1["B_rate"], "m w.e. a-1"),
    ("B_total", mass(q1["B_total"]), "kg m-2", q1["B_total"], "m w.e."),
    ("B_rate_A1994", rate(q1["B_rate_A1994"]), "kg m-2 s-1", q1["B_rate_A1994"], "m w.e. a-1"),
    ("B_rate_A2021", rate(q1["B_rate_A2021"]), "kg m-2 s-1", q1["B_rate_A2021"], "m w.e. a-1"),
    ("glamos_B_rate", rate(q1["glamos_B_rate"]), "kg m-2 s-1", q1["glamos_B_rate"], "m w.e. a-1"),
    # Question 2
    ("lapse_rate", 6.5e-3, "K m-1", 6.5, "degC km-1"),
    ("P_mean_station", 1887.1 / YEAR, "kg m-2 s-1", 1887.1, "mm a-1"),
    # Question 3
    ("month", MONTH, "s", 365 / 12, "d"),
    ("F", flux(ddf[STATION]), "kg m-2 s-1 K-1", ddf[STATION], "mm w.e. (degC d)-1"),
    ("h", flux(ddf[STATION]) * L_F, "W m-2 K-1", ddf[STATION], "mm w.e. (degC d)-1"),
    ("F_GRH", flux(ddf["GRH"]), "kg m-2 s-1 K-1", ddf["GRH"], "mm w.e. (degC d)-1"),
    ("h_GRH", flux(ddf["GRH"]) * L_F, "W m-2 K-1", ddf["GRH"], "mm w.e. (degC d)-1"),
    ("F_ANT", flux(ddf["ANT"]), "kg m-2 s-1 K-1", ddf["ANT"], "mm w.e. (degC d)-1"),
    ("h_ANT", flux(ddf["ANT"]) * L_F, "W m-2 K-1", ddf["ANT"], "mm w.e. (degC d)-1"),
    ("C_mean_year", mass(float(cal["C_mean_mwe"])), "kg m-2", float(cal["C_mean_mwe"]), "m w.e."),
    ("S_mean_year", float(cal["S_mean_degCd"]) * DAY, "K s", float(cal["S_mean_degCd"]), "degC d"),
    ("T_excess_OTL", float(cal["S_mean_degCd"]) / 365.25, "K", float(cal["S_mean_degCd"]), "degC d a-1"),
    ("T_excess_GRH", float(grh["S_mean_degCd"]) / 365.25, "K", float(grh["S_mean_degCd"]), "degC d a-1"),
] + [(f"lit_{name}", flux(v), "kg m-2 s-1 K-1", v, "mm w.e. (degC d)-1") for name, v in
     (("aletsch_snow", 5.3), ("aletsch_ice", 11.7), ("braithwaite_low", 3.5), ("braithwaite_high", 4.6),
      ("braithwaite_sd", 1.4))] + [(f"lit_{name}_h", flux(v) * L_F, "W m-2 K-1", v, "mm w.e. (degC d)-1") for name, v in
     (("aletsch_snow", 5.3), ("aletsch_ice", 11.7), ("braithwaite_low", 3.5), ("braithwaite_high", 4.6),
      ("braithwaite_sd", 1.4))] + [
    ("snow_winter_model", 880.0, "kg m-2", 0.88, "m w.e."),
    ("snow_winter_glamos", mass(np.mean([v[0] for y, v in glamos.items() if 1992 <= y <= 2025])), "kg m-2",
     np.mean([v[0] for y, v in glamos.items() if 1992 <= y <= 2025]), "m w.e."),
    ("june1942_melt", ddf[STATION] * 365 / 12 * 0.04, "kg m-2", ddf[STATION] * 365 / 12 * 0.04 / 1000, "m w.e."),
    ("B_2022_model", mass(b[2022]), "kg m-2", b[2022], "m w.e."),
    ("B_2022_glamos", mass(glamos[2022][2]), "kg m-2", glamos[2022][2], "m w.e."),
    # Question 4
    ("cumulative", mass(float(q4["cumulative"])), "kg m-2", float(q4["cumulative"]), "m w.e."),
    ("mean", rate(float(q4["mean"])), "kg m-2 s-1", float(q4["mean"]), "m w.e. a-1"),
] + [(f"mean_{p}", rate(float(q4[f"mean_{p}"])), "kg m-2 s-1", float(q4[f"mean_{p}"]), "m w.e. a-1")
     for p in ("1935_1965", "1966_1995", "1996_2025")] + [
    ("trend", rate(float(q4["trend"])) / (10 * YEAR), "kg m-2 s-2", float(q4["trend"]), "m w.e. a-1 per decade"),
    ("B_1977_model", mass(b[1977]), "kg m-2", b[1977], "m w.e."),
    ("geodetic_1950_1961_model", rate(float(q4["geodetic_1950_1961_model"])), "kg m-2 s-1",
     float(q4["geodetic_1950_1961_model"]), "m w.e. a-1"),
    ("geodetic_1950_1961_glamos", rate(float(q4["geodetic_1950_1961_glamos"])), "kg m-2 s-1",
     float(q4["geodetic_1950_1961_glamos"]), "m w.e. a-1"),
    # Question 5
    ("c", 0.034 * 1000 ** (3 - 2 * 1.375), "m^(3-2 gamma)", 0.034, "km^(3-2 gamma)"),
    ("V_2021", q5["V_2021"] * 1e9, "m3", q5["V_2021"], "km3"),
    ("B_2022_2025", mass(q5["B_2022_2025"]), "kg m-2", q5["B_2022_2025"], "m w.e."),
    ("dV_2021_2025", (q5["V_2025"] - q5["V_2021"]) * 1e9, "m3", q5["V_2025"] - q5["V_2021"], "km3"),
    ("V_2025", q5["V_2025"] * 1e9, "m3", q5["V_2025"], "km3"),
    ("B_future", rate(q5["B_future"]), "kg m-2 s-1", q5["B_future"], "m w.e. a-1"),
    ("B_recent", rate(q5["B_recent_2015_2025"]), "kg m-2 s-1", q5["B_recent_2015_2025"], "m w.e. a-1"),
    ("dV_per_time", q5["A_2021"] * 1e6 * rate(q5["B_future"]) / 900.0, "m3 s-1",
     q5["A_2021"] * q5["B_future"] / 1000 * 1000 / 900, "km3 a-1"),
    ("thinning", rate(q5["B_future"]) / 900.0, "m s-1", q5["dH_future"], "m ice a-1"),
]
write_csv("si_units.csv", ("quantity", "SI", "SI_unit", "assignment", "assignment_unit"), rows)
for q, v, u, w, wu in rows:
    print(f"{q:28s} {v:13.4g} {u:16s} {w:10.4g} {wu}")
