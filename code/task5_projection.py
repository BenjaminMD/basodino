"""Task 5: glacier volume 2021 from volume-area scaling, and the year in which half of it is gone.

V = c A^gamma with gamma = 1.375 (Bahr et al. 1997) and c = 0.034 km^(3 - 2 gamma), the worldwide mean of Bahr et al.
(2015), A = A_2021 of Task 1. Hydrological years 2022-2025 take the reconstructed balances of Task 4; after 2025 the
balance stays at the mean of the 30 years October 1995 to September 2025 (hydrological years 1996-2025). Volume change
per year = A x B x rho_w / rho_ice (900 kg m-3, Hint 4).
Two area rules: "constant", A = A_2021 for ever (Hint 1); "scaling", the area follows the volume each year,
A = (V / c)^(1 / gamma), one way to avoid Hint 1 (Question 5.3). For the record, the constant-area projection with
the balance of the last eleven years (2015-2025) instead of the 30-year mean.
Outputs:
  code/task5_projection.csv           quantity, value, unit, definition
  tex/figures/volume_projection{,_si}.pdf   volume 2021-2100 under both area rules; 150 x 52 mm; _si in m3
usage:  python3 code/task5_projection.py
"""
import numpy as np

from common import (GLACIER, GREY, ICE, INK, LAB, MM, RHO_ICE, RHO_W, STATION, figure_style, read_csv, save, write_csv)

C, GAMMA = 0.034, 1.375                                       # km^(3 - 2 gamma), dimensionless
FUTURE = range(1996, 2026)                                    # the 30 years of the future balance
q1 = {r["quantity"]: float(r["value"]) for r in read_csv("task1_geodetic.csv")}
b = {int(r["hydro_year"]): float(r["B_mwe"]) for r in read_csv("task4_balance.csv") if r["station"] == STATION}

a21 = q1["A_2021"]                                            # km2
v21 = C * a21 ** GAMMA                                        # km3
b_future = np.mean([b[y] for y in FUTURE])
b_future_31 = np.mean([b[y] for y in range(1995, 2026)])     # the same with 1995 included, for the record


b_recent = np.mean([b[y] for y in range(2015, 2026)])


def project(rule, b_after=b_future, years=range(2022, 2201)):
    """Volume at the end of each hydrological year, from V_2021."""
    v, a, out = v21, a21, {2021: v21}
    for y in years:
        v = max(v + a * (b[y] if y in b else b_after) / 1000 * RHO_W / RHO_ICE, 0.0)
        a = a21 if rule == "constant" else (v / C) ** (1 / GAMMA)
        out[y] = v
    return out


def half_year(series):
    return next(y for y, v in sorted(series.items()) if v <= v21 / 2)


runs = {rule: project(rule) for rule in ("constant", "scaling")}
numbers = [("A_2021", a21, "km2", "Task 1"), ("V_2021", v21, "km3", "c A_2021^gamma"),
           ("H_2021", v21 / a21 * 1000, "m", "mean ice thickness V_2021 / A_2021"),
           ("B_2022_2025", sum(b[y] for y in range(2022, 2026)), "m w.e.", f"sum of Task 4, {STATION}"),
           ("V_2025", runs["constant"][2025], "km3", "V_2021 + A_2021 x B_2022_2025 x rho_w / rho_ice"),
           ("B_future", b_future, "m w.e. a-1", "mean of hydrological years 1996-2025"),
           ("B_future_1995_2025", b_future_31, "m w.e. a-1", "the same over 31 years 1995-2025"),
           ("dH_future", b_future * RHO_W / RHO_ICE, "m ice a-1", "thinning rate after 2025")]
for rule, run in runs.items():
    numbers += [(f"half_volume_year_{rule}", half_year(run), "", f"first year with V <= V_2021 / 2, area {rule}"),
                (f"V_2050_{rule}", run[2050], "km3", ""), (f"V_2100_{rule}", run[2100], "km3", "")]
    if rule == "constant":
        numbers.append(("zero_volume_year_constant", next(y for y, v in sorted(run.items()) if v == 0), "", ""))
numbers += [("B_recent_2015_2025", b_recent, "m w.e. a-1", "mean of hydrological years 2015-2025"),
            ("half_volume_year_constant_recent", half_year(project("constant", b_recent)), "",
             "area constant, balance of 2015-2025 after 2025")]
write_csv("task5_projection.csv", ("quantity", "value", "unit", "definition"), numbers)
for q, v, u, _ in numbers:
    print(f"{q:28s} {v:10.4f} {u}" if isinstance(v, float) else f"{q:28s} {v:10d}")

# --- figure: in km3 and in m3; each volume is drawn at the end of its hydrological year (30 September, year + 0.75)
plt = figure_style()
E = 0.75
for name, fv, v_label in (("volume_projection", 1.0, "Volume (km³)"),
                          ("volume_projection_si", 100.0, "Volume (10⁷ m³)")):
    W, H = 150.0, 52.0
    fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white")
    ax = fig.add_axes([16 / W, 9 / H, 130 / W, 41 / H])
    ax.set_xlim(2020, 2100), ax.set_ylim(0, fv * v21 * 1.08)
    ax.axvspan(2021 + E, 2025 + E, color=ICE, lw=0, zorder=0)
    ax.axhline(fv * v21 / 2, color=GREY, lw=0.5, ls=(0, (3, 1.5)), zorder=1)
    ax.text(2099.5, fv * v21 / 2, "half the 2021 volume", ha="right", va="bottom", color=GREY, fontsize=LAB)
    for rule, colour, offset, ha in (("constant", INK, (-4, -8), "right"), ("scaling", GLACIER, (4, 3), "left")):
        yrs = [y for y in sorted(runs[rule]) if y <= 2100 and (y == 2021 or runs[rule][y - 1] > 0)]
        ax.plot([y + E for y in yrs], [fv * runs[rule][y] for y in yrs], color=colour, lw=1.3, zorder=3)
        yh = half_year(runs[rule])
        ax.plot([yh + E], [fv * runs[rule][yh]], "o", ms=3.4, color=colour, mec="white", mew=0.6, zorder=4)
        ax.annotate(str(yh), (yh + E, fv * runs[rule][yh]), xytext=offset, textcoords="offset points", ha=ha,
                    color=colour, fontweight=600, fontsize=LAB)
    y_end = next(y for y in sorted(runs["constant"]) if runs["constant"][y] == 0)
    ax.text(y_end + E + 1.0, fv * v21 * 0.035, "area constant at 2021", ha="left", va="bottom", color=INK,
            fontsize=LAB)
    x0 = 2082
    ax.text(x0 + E, fv * runs["scaling"][x0] + fv * v21 * 0.03, "area following the volume", ha="left",
            va="bottom", color=GLACIER, fontsize=LAB)
    ax.set_xlabel("Year", labelpad=1.5)
    ax.set_ylabel(v_label)
    save(fig, name, "task5_projection.py")
