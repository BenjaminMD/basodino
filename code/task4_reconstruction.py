"""Task 4: annual mass balance 1935-2025 with the calibrated model, hydrological years (1 October to 30 September).

B(y) = C(y) - DDF x S(y) from task3_terms.csv and task3_ddf.csv (distributed model, specific target), on the fixed 2021
surface (Hint 2). Every station is reconstructed; each is checked against the GLAMOS glaciological annual balances
1992-2025, which the calibration did not use, and against the GLAMOS geodetic balances of the survey intervals since
1949 (the model mean over the hydrological years that end after the first and up to the second survey).
Outputs:
  code/task4_balance.csv    hydrological year, station, B (m w.e.), cumulative B since 1935 (m w.e.)
  code/task4_summary.csv    per station: cumulative and mean balance, three 30-year means, linear trend with its p value,
                            the five most negative and most positive years, GLAMOS check (r, bias, RMSE), geodetic
                            intervals (GLAMOS and model, m w.e. a-1)
  tex/figures/mb_1935_2025{,_si}.pdf   the report's station, in m water equivalent and in kg m-2: a annual balance with the GLAMOS values and the 11-year mean,
                            b cumulative balance; 150 x 64 mm
usage:  python3 code/task4_reconstruction.py
"""
import csv

import numpy as np
from scipy import stats

from common import (GLACIER, GREY, INK, LAB, MM, PALE_BLUE, PALE_RED, ROOT, STATION, STATIONS, YEARS, figure_style,
                    panel_letter, read_csv,
                    read_glamos, save, write_csv)

PERIODS = ((1935, 1965), (1966, 1995), (1996, 2025))
SURVEYS = ("19299999", "19491007", "19610919", "19710903", "19850910", "19910918", "20020916", "20080829", "20140923", "20190920",
           None, "20219999", "20240823")                       # GLAMOS survey chain; it has no pair 2019 to 2021
with open(ROOT / "data/glamos_basodino_volumechange.csv", encoding="utf-8") as f:
    vc = {(r[3], r[4]): float(r[11]) for r in csv.reader(f) if len(r) > 13 and r[0] == "C14-10"}
GEODETIC = [(int(a[:4]) + 1, int(e[:4]), vc[(a, e)]) for a, e in zip(SURVEYS, SURVEYS[1:]) if a and e]
ddf = {r["station"]: float(r["DDF_mm_per_degCd"]) / 1000 for r in read_csv("task3_ddf.csv")
       if (r["approach"], r["target"]) == ("distributed", "specific")}
b = {st: {} for st in STATIONS}
for r in read_csv("task3_terms.csv"):
    if r["approach"] == "distributed":
        b[r["station"]][int(r["hydro_year"])] = float(r["C_mwe"]) - ddf[r["station"]] * float(r["S_degCd"])
glamos = read_glamos()
gy = [y for y in sorted(glamos) if y in YEARS]

rows, summary = [], []
for st in STATIONS:
    v = np.array([b[st][y] for y in YEARS])
    rows += [(y, st, x, c) for y, x, c in zip(YEARS, v, np.cumsum(v))]
    fit = stats.linregress(list(YEARS), v)
    order = np.argsort(v)
    g_mod, g_obs = np.array([b[st][y] for y in gy]), np.array([glamos[y][2] for y in gy])
    s = [("cumulative", v.sum(), "m w.e."), ("mean", v.mean(), "m w.e. a-1"), ("std", v.std(ddof=1), "m w.e. a-1"),
         ("positive_years", int((v > 0).sum()), "")]
    s += [(f"mean_{a}_{e}", v[(np.array(YEARS) >= a) & (np.array(YEARS) <= e)].mean(), "m w.e. a-1") for a, e in PERIODS]
    s += [("trend", fit.slope * 10, "m w.e. a-1 per decade"), ("trend_p", fit.pvalue, ""),
          ("trend_1966_2025", stats.linregress(list(YEARS)[31:], v[31:]).slope * 10, "m w.e. a-1 per decade")]
    s += [(f"lowest_{i + 1}", f"{YEARS[j]}: {v[j]:.2f}", "m w.e.") for i, j in enumerate(order[:5])]
    s += [(f"highest_{i + 1}", f"{YEARS[j]}: {v[j]:.2f}", "m w.e.") for i, j in enumerate(order[::-1][:5])]
    s += [("glamos_n", len(gy), "years"), ("glamos_r", np.corrcoef(g_mod, g_obs)[0, 1], ""),
          ("glamos_bias", (g_mod - g_obs).mean(), "m w.e. a-1"),
          ("glamos_rmse", np.sqrt(((g_mod - g_obs) ** 2).mean()), "m w.e. a-1"),
          ("glamos_mean_obs", g_obs.mean(), "m w.e. a-1"), ("glamos_mean_mod", g_mod.mean(), "m w.e. a-1"),
          ("glamos_std_obs", g_obs.std(ddof=1), "m w.e. a-1"), ("glamos_std_mod", g_mod.std(ddof=1), "m w.e. a-1")]
    s += [(f"geodetic_{y0}_{y1}_{w}", x, "m w.e. a-1") for y0, y1, obs in GEODETIC
          for w, x in (("glamos", obs), ("model", np.mean([b[st][y] for y in range(max(y0, 1935), y1 + 1)])))]
    summary += [(st, q, x, u) for q, x, u in s]
write_csv("task4_balance.csv", ("hydro_year", "station", "B_mwe", "B_cumulative_mwe"), rows)
write_csv("task4_summary.csv", ("station", "quantity", "value", "unit"), summary)
for st in STATIONS:
    print(st, "  ".join(f"{q} {x:.3g}" if isinstance(x, float) else f"{q} {x}" for s_, q, x, _ in summary if s_ == st))

# --- figure: the report's station; both variants have the same numbers (1 m water equivalent = 10^3 kg m-2)
plt = figure_style()
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

x = np.array(YEARS)
v = np.array([b[STATION][y] for y in YEARS])
cum = np.cumsum(v)
c0 = cum[x == 1949][0]                                         # the model's cumulative balance at the 1949 survey
later = [e for e in SURVEYS if e and e > "19491007"]
meas_x = [1949] + [int(e[:4]) for e in later]
meas_y = [c0] + [c0 + vc[("19491007", e)] * (int(e[:4]) - 1949) for e in later]
for name, unit in (("mb_1935_2025", "m water equivalent"), ("mb_1935_2025_si", "10³ kg m⁻²")):
    W, H = 150.0, 64.0
    fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white")
    ax1 = fig.add_axes([16 / W, 1 - 33 / H, 121 / W, 26 / H])
    ax2 = fig.add_axes([16 / W, 1 - 57 / H, 121 / W, 21 / H], sharex=ax1)
    ax1.axhline(0, color=GREY, lw=0.4, zorder=1)
    ax1.bar(x, v, width=0.72, color=np.where(v < 0, PALE_RED, PALE_BLUE), lw=0, zorder=2)
    for y0, y1, obs in GEODETIC:
        ax1.plot([y0 - 0.5, y1 + 0.5], [obs, obs], color=GLACIER, lw=1.8, solid_capstyle="butt", zorder=2.5)
    ax1.plot(x[5:-5], np.convolve(v, np.ones(11) / 11, mode="valid"), color=INK, lw=1.1, zorder=3)
    ax1.plot(gy, [glamos[y][2] for y in gy], "o", ms=2.3, mfc="white", mec=INK, mew=0.6, zorder=5)
    ax1.set_ylabel(f"Balance\n({unit})")
    ax1.tick_params(labelbottom=False)
    keys = [(Patch(facecolor=PALE_BLUE), Patch(facecolor=PALE_RED)), Line2D([], [], color=INK, lw=1.1),
            Line2D([], [], ls="", marker="o", ms=2.3, mfc="white", mec=INK, mew=0.6),
            Line2D([], [], color=GLACIER, lw=1.8)]
    fig.legend(keys, ["Model, each year", "Model, 11-year mean", "Measured, each year", "Measured, survey periods"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc="upper left",
               bbox_to_anchor=(16 / W, 1 - 0.6 / H), ncols=4, handlelength=1.6, handleheight=0.7, handletextpad=0.5,
               columnspacing=1.6, borderaxespad=0, borderpad=0)
    ax2.axhline(0, color=GREY, lw=0.4)
    ax2.plot(x, cum, color=INK, lw=1.0)
    ax2.plot(meas_x, meas_y, "-o", color=GLACIER, lw=0.9, ms=2.3)
    ax2.text(2027.5, cum[-1], "model", ha="left", va="center", color=INK, fontsize=LAB, clip_on=False)
    ax2.text(2027.5, meas_y[-1], "measured", ha="left", va="center", color=GLACIER, fontsize=LAB, clip_on=False)
    ax2.yaxis.set_major_locator(MultipleLocator(10))
    ax2.set_ylabel(f"Cumulative\nbalance ({unit})")
    ax2.set_xlabel("Hydrological year", labelpad=1.5)
    ax2.set_xlim(1933.5, 2026.5)
    for ax in (ax1, ax2):
        ax.yaxis.set_label_coords(-0.063, 0.5)
    for ax, letter in ((ax1, "a"), (ax2, "b")):
        panel_letter(fig, ax, letter, W)
    save(fig, name, "task4_reconstruction.py")
