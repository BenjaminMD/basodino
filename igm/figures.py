"""Report figures of the fixed IGM ensemble (DESIGN.md step 7 and fix round), house style of code/common.py: Source
Sans 3, drawn at print size 150 mm wide, graphics only (tex/SIDECARS.md: no titles, no sentences; the caption text and
every number are in igm/RESULTS.md), no abbreviations, SI units. PDFs to tex/figures/. Time axes of projections end 3
years after the latest 95th-percentile disappearance year of the ensembles shown (author's rule, DESIGN.md fix round).
The first ensemble's version of this script is in the git history, its PDFs in results/first_ensemble/figures/.

  igm_hindcast        hindcast 1850-2026 of the 128 draws (median, 5-95 %) against the validation data of F5: area,
                      the 11 volume changes 1907-2024 with the expected misses stated before the runs, tongue
                      elevation, length change (results/f5_series_f5_hind.csv, f5_validation.csv,
                      data/glamos_basodino_lengthchange.csv)
  igm_calibration     balance calibration F1 (results/calib_f1_bands.csv, calib_f1_draws.csv) and start-state fit F3
                      (results/f3_terms.csv, f3_weights.csv, f3_runs_f3_r{1,2}.csv, posterior_f3_128.csv):
                      glacier-wide balance, the temperature offsets, the 11 volume changes, ice area at the target
                      dates, elevation change per band 2012-2021, the speed against the ITS_LIVE bound, the
                      standardised misfit of every start-state term
  igm_volume_area     volume and ice area 2021 to the end of the time axis per SSP, 2026 level anchored (bands) and
                      native (lines), with the observed areas 2021-2026 (results/f7_series_quantiles.csv,
                      f7_years.csv, f3_terms.csv)
  igm_years           half-volume year (against V_2026) and disappearance years (0.01, 0.1 km2) per SSP and level;
                      share of their spread per factor; with against without ice flow (results/f7_years.csv,
                      f7_spread.csv)
  igm_half_volume_u   V_2021 and the half-volume year against V_2021 as a function of the start-thickness factor u
                      (results/f7_half2021_samples.csv, members.csv)
  igm_icefree_map     median ice-free year per cell and its 90 % range, anchored level, three SSPs pooled, with the
                      outlines 1850-2026 (results/f7_icefree_maps.npz, data/vectors.gpkg,
                      data/outline_2026/outline_2026-09-27.gpkg)
  igm_published       this ensemble against the published projections of igm/PUBLISHED_PROJECTIONS.md: volume
                      2000-2100 under SSP2-4.5, the disappearance years and the half-volume years per scenario
                      (results/f7_series_quantiles.csv, f7_years.csv, data/published/basodino_volume.csv,
                      basodino_members.csv, vantricht2025/basodino_extinction.csv; the statements of the table
                      'Other projections and statements')
usage (Mac):  python igm/figures.py [name ...]
"""
import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely import wkb

IGM = Path(__file__).resolve().parent
sys.path.insert(0, str(IGM.parent / "code"))
from common import (ACCENT, FIG, GLACIER, GREY, ICE, INK, LAB, MM, NUM, PALE_BLUE, PALE_RED, ROOT, RULE,  # noqa: E402
                    figure_style, panel_letter)

RES = IGM / "results"
PUB = ROOT / "data/published"
W = 150.0
SSPS = ("SSP1-2.6", "SSP2-4.5", "SSP5-8.5")
plt = figure_style()


def model_handle(c=GLACIER):
    """Legend entry of a model bar (5-95 %) with its median (white dot)."""
    return (plt.Line2D([], [], color=c, lw=2.4),
            plt.Line2D([], [], color=c, marker="o", ms=2.6, mfc="white", mew=0.6, ls=""))


def rows(name):
    return list(csv.DictReader(open(RES / name)))


def num(v):
    return float(v) if v not in ("", "nan") else np.nan


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", metadata={"Title": name, "Creator": "igm/figures.py"})
    plt.close(fig)
    print(f"tex/figures/{name}.pdf")


def axes(fig, H, x, y, w, h):
    """Axes at x, y (bottom left), w, h in mm."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def band(ax, x, z, color, pale, label=None, lw=1.0):
    lo, med, hi = np.nanpercentile(z, [5, 50, 95], axis=0)
    ax.fill_between(x, lo, hi, color=pale, lw=0, zorder=1)
    ax.plot(x, med, color=color, lw=lw, zorder=2, label=label)


def gpkg(path, table, where=""):
    """GeoPackage geometry blobs -> shapely (as code/area_clock.py)."""
    con = sqlite3.connect(path)
    blobs = [b for (b,) in con.execute(f"SELECT geom FROM {table} {where}")]
    con.close()
    return [wkb.loads(bytes(b[8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(b[3] >> 1) & 7]:])) for b in blobs]


def period_bars(ax, R, lo, med, hi, obs, sig):
    """Model bar (lo-hi, median) beside the observation with its sigma, one pair per row of R."""
    x = np.arange(len(R))
    for i, r in enumerate(R):
        ax.plot([i + 0.12] * 2, [num(r[lo]), num(r[hi])], color=GLACIER, lw=2.4, solid_capstyle="butt")
        ax.plot([i + 0.12], [num(r[med])], "o", ms=2.6, mfc="white", mec=GLACIER, mew=0.6)
    ax.errorbar(x - 0.12, [num(r[obs]) for r in R], [num(r[sig]) for r in R], fmt="o", ms=3.0, color=INK,
                elinewidth=0.6, capsize=0)
    ax.axhline(0, color=RULE, lw=0.4, zorder=0)
    ax.set_xlim(-0.6, len(R) - 0.4)
    return x


# --- hindcast ---------------------------------------------------------------------------------------------------------

def hindcast():
    S = defaultdict(list)
    for r in rows("f5_series_f5_hind.csv"):
        S[r["name"]].append(r)
    yr = np.array([int(r["year"]) for r in next(iter(S.values()))])
    col = lambda c: np.array([[num(r[c]) for r in s] for s in S.values()])
    V = rows("f5_validation.csv")
    H = 140.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    # a: area
    ax = axes(fig, H, 14, 100, 132, 36)
    band(ax, yr, col("A1_km2"), GLACIER, PALE_BLUE, "Model, all ice, median and 5–95 %")
    ax.plot(yr, np.nanmedian(col("A1_in1850_km2"), 0), color=GREY, lw=0.8, ls=(0, (3, 1.5)), zorder=2,
            label="Model, inside the 1850 outline, median")
    ax.errorbar([1850], [4.008], [0.25], fmt="s", ms=3.2, mfc="white", mec=INK, ecolor=INK, elinewidth=0.6,
                capsize=0, zorder=4, label="Spin-up target 1850")
    obs = [r for r in V if r["kind"] == "area"]
    ax.errorbar([int(r["date_end"][:4]) + int(r["date_end"][5:7]) / 12 for r in obs],
                [float(r["observed"]) for r in obs], [float(r["sigma"]) for r in obs], fmt="o", ms=3.0, color=INK,
                elinewidth=0.6, capsize=0, zorder=4, label="Inventory area")
    ax.set_xlim(1845, 2030), ax.set_ylim(0, 5.0)
    ax.set_ylabel("Ice area (km²)")
    ax.legend(loc="lower left", fontsize=NUM, handlelength=1.6, borderaxespad=0.2, ncol=2, columnspacing=1.2)
    panel_letter(fig, ax, "a", W)
    # b: volume change per survey period, expected misses (stated before the runs)
    ax = axes(fig, H, 14, 54, 132, 34)
    vol = [r for r in V if r["kind"] == "volume"]
    x = period_bars(ax, vol, "p5", "p50", "p95", "observed", "sigma")
    for i, r in enumerate(vol):
        if r["expected"]:
            y = max(float(r["observed"]) + float(r["sigma"]), float(r["p95"])) + 0.008
            ax.plot([i], [y], marker="v" if r["expected"] == "too much loss" else "^", ms=4.2, color=ACCENT,
                    mfc=ACCENT if r["expected_met"] == "yes" else "white", mew=0.7)
    ax.set_xticks(x, [f"{r['date_start'][:4]}–{r['date_end'][:4]}" for r in vol])
    ax.set_ylim(-0.15, 0.045)
    ax.set_ylabel("Volume change (km³)")
    h, lab = [plt.Line2D([], [], color=INK, marker="o", ms=3, ls=""), model_handle()], ["Survey",
                                                                                       "Model, median and 5–95 %"]
    for e, m in (("too little loss", "^"), ("too much loss", "v")):
        for met in ("yes", "no"):
            if any(r["expected"] == e and r["expected_met"] == met for r in vol):
                h.append(plt.Line2D([], [], color=ACCENT, marker=m, ms=4, ls="", mfc=ACCENT if met == "yes" else
                                    "white"))
                lab.append(f"Expected: {e}, {'met' if met == 'yes' else 'not met'}")
    ax.legend(handles=h, labels=lab, loc="lower right", fontsize=NUM, handlelength=1.2, borderaxespad=0.2, ncol=2,
              columnspacing=1.0)
    panel_letter(fig, ax, "b", W)
    # c: tongue elevation
    ax = axes(fig, H, 14, 10, 58, 32)
    k = yr >= 1990
    band(ax, yr[k], col("tongue_m")[:, k], GLACIER, PALE_BLUE, "Model")
    tg = [r for r in V if r["kind"] == "tongue"]
    ax.plot([int(r["date_end"][:4]) for r in tg], [float(r["observed"]) for r in tg], "o", ms=2.6, color=INK,
            label="Measured")
    ax.set_xlim(1989, 2027)
    ax.set_xlabel("Hydrological year", labelpad=1.5)
    ax.set_ylabel("Lowest ice surface (m)")
    ax.legend(loc="lower right", fontsize=NUM, borderaxespad=0.2)
    panel_letter(fig, ax, "c", W)
    # d: length change since 1899
    ax = axes(fig, H, 90, 10, 56, 32)
    L = col("length_m")
    k99 = list(yr).index(1899)
    k = (yr >= 1899) & np.isfinite(L).all(0)          # years in which every draw has ice on the flowline
    band(ax, yr[k], (L - L[:, [k99]])[:, k], GLACIER, PALE_BLUE, "Model")
    cum, pts = 0.0, [(1899, 0.0)]
    for r in csv.reader(open(ROOT / "data/glamos_basodino_lengthchange.csv", encoding="utf-8")):
        if len(r) > 7 and r[1] == "C14-10":
            cum += float(r[6])
            pts.append((int(r[4][:4]), cum))
    ax.plot(*zip(*pts), "o", ms=1.8, color=INK, label="Measured")
    ax.axhline(0, color=RULE, lw=0.4, zorder=0)
    ax.set_xlim(1895, 2027)
    ax.set_xlabel("Hydrological year", labelpad=1.5)
    ax.set_ylabel("Length change since 1899 (m)")
    ax.legend(loc="lower left", fontsize=NUM, borderaxespad=0.2)
    panel_letter(fig, ax, "d", W, x_mm=73.5)
    save(fig, "igm_hindcast")


# --- calibration and start state --------------------------------------------------------------------------------------

def calibration():
    B = rows("calib_f1_bands.csv")
    T = rows("f3_terms.csv")
    H = 168.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    # a: glacier-wide balance of the observation periods (not fitted)
    ax = axes(fig, H, 14, 134, 58, 30)
    for kind, c, pale, name in (("bw_glacier", GLACIER, PALE_BLUE, "winter"), ("ba_glacier", ACCENT, PALE_RED,
                                                                                "annual")):
        d = [r for r in B if r["kind"] == kind]
        y = [int(r["label"]) for r in d]
        ax.fill_between(y, [num(r["model_p5"]) for r in d], [num(r["model_p95"]) for r in d], color=pale, lw=0)
        ax.plot(y, [num(r["model_p50"]) for r in d], color=c, lw=0.9, label=f"Model, {name}")
        ax.plot(y, [num(r["observed"]) for r in d], "o", ms=2.4, mfc="white", mec=c, mew=0.6,
                label=f"Measured, {name}")
    ax.axhline(0, color=RULE, lw=0.4, zorder=0)
    ax.set_xlim(1991, 2026), ax.set_ylim(-4.2, 3.6)
    ax.set_xlabel("Hydrological year", labelpad=1.5)
    ax.set_ylabel("Balance (m water equivalent)")
    ax.legend(loc="lower left", fontsize=NUM, ncol=2, columnspacing=0.8, handlelength=1.2, borderaxespad=0.1)
    panel_letter(fig, ax, "a", W)
    # b: the two temperature offsets of the balance posterior and the 128 ensemble draws; prior -3 to 3 K
    ax = axes(fig, H, 88, 134, 58, 30)
    K = rows("calib_f1_draws.csv")
    w = np.array([float(r["weight"]) for r in K])
    P = rows("posterior_f3_128.csv")
    for key, pkey, c, pale, name in (("dT_degC", "dT", GLACIER, PALE_BLUE, "Station offset, all years"),
                                     ("dT_pre_degC", "dT_pre", ACCENT, PALE_RED, "Added before 1991")):
        z = np.array([float(r[key]) for r in K])
        ax.hist(z, bins=np.arange(-3.0, 3.01, 0.1), weights=w / w.sum(), color=pale, lw=0, label=name)
        ax.plot([float(r[pkey]) for r in P], [-0.012 if c == GLACIER else -0.024] * len(P), "|", ms=3.5, color=c,
                mew=0.5)
    ax.axvline(-3.0, color=GREY, lw=0.6)
    ax.axvline(3.0, color=GREY, lw=0.6)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(bottom=-0.034)
    ax.set_xlabel("Temperature offset (K)", labelpad=1.5)
    ax.set_ylabel("Share of weight")
    h = [plt.matplotlib.patches.Patch(color=PALE_BLUE, lw=0), plt.matplotlib.patches.Patch(color=PALE_RED, lw=0),
         plt.Line2D([], [], color=INK, marker="|", ms=4, ls="", mew=0.6), plt.Line2D([], [], color=GREY, lw=0.6)]
    ax.legend(handles=h, labels=["Station offset, all years", "Added before 1991", "Ensemble draws",
                                 "Prior limit"], loc="upper left", fontsize=NUM, borderaxespad=0.1)
    panel_letter(fig, ax, "b", W, x_mm=76)
    # c: geodetic volume change, 11 periods (binding)
    ax = axes(fig, H, 14, 94, 132, 28)
    d = [r for r in B if r["kind"] == "dv"]
    x = period_bars(ax, d, "predictive_p5", "predictive_p50", "predictive_p95", "observed", "sigma_obs")
    ax.set_xticks(x, [f"{r['date0'][:4]}–{r['date1'][:4]}" for r in d])
    ax.set_ylabel("Volume change (km³)")
    h = [plt.Line2D([], [], color=INK, marker="o", ms=3, ls=""), model_handle()]
    ax.legend(handles=h, labels=["Survey", "Model, median and 5–95 % with model error"], loc="lower right",
              fontsize=NUM, borderaxespad=0.2)
    panel_letter(fig, ax, "c", W)
    # d: ice area at the target dates
    ax = axes(fig, H, 14, 52, 58, 30)
    A = [r for r in T if r["term"] == "area"]
    for i, r in enumerate(A):
        ax.plot([i + 0.2] * 2, [float(r["samples_model_p05"]), float(r["samples_model_p95"])], color=RULE, lw=2.4,
                solid_capstyle="butt")
        ax.plot([i + 0.05] * 2, [float(r["model_p05"]), float(r["model_p95"])], color=GLACIER, lw=2.4,
                solid_capstyle="butt")
        ax.plot([i + 0.05], [float(r["model_p50"])], "o", ms=2.6, mfc="white", mec=GLACIER, mew=0.6)
    ax.errorbar(np.arange(len(A)) - 0.15, [float(r["observed"]) for r in A], [float(r["sigma"]) for r in A],
                fmt="o", ms=3, color=INK, elinewidth=0.6, capsize=0)
    ax.set_xticks(range(len(A)), [r["value"].split()[1][:4] + ("\nall ice" if "all" in r["value"] else
                                                               "\nmain body") for r in A])
    ax.set_xlim(-0.6, len(A) - 0.4), ax.set_ylim(0, 1.8)
    ax.set_ylabel("Ice area (km²)")
    h = [plt.Line2D([], [], color=INK, marker="o", ms=3, ls=""), model_handle(),
         plt.Line2D([], [], color=RULE, lw=2.4)]
    ax.legend(handles=h, labels=["Observed", "Posterior, median and 5–95 %", "All samples, 5–95 %"],
              loc="lower left", fontsize=NUM, borderaxespad=0.1)
    panel_letter(fig, ax, "d", W)
    # e: elevation change per 50 m band 2012-2021
    ax = axes(fig, H, 88, 52, 58, 30)
    d = [r for r in T if r["term"] == "dh2"]
    z = np.array([float(r["value"].split()[-2].split("-")[0]) + 25 for r in d])
    ax.fill_betweenx(z, [float(r["model_p05"]) for r in d], [float(r["model_p95"]) for r in d], color=PALE_BLUE,
                     lw=0)
    ax.plot([float(r["model_p50"]) for r in d], z, color=GLACIER, lw=0.9, label="Model")
    ax.errorbar([float(r["observed"]) for r in d], z, xerr=[float(r["sigma"]) for r in d], fmt="o", ms=2.4,
                mfc="white", mec=INK, ecolor=INK, elinewidth=0.5, capsize=0, label="Survey")
    ax.axvline(0, color=RULE, lw=0.4, zorder=0)
    ax.set_xlabel("Elevation change 2012–2021 (m)", labelpad=1.5)
    ax.set_ylabel("Elevation (m)")
    ax.legend(loc="lower left", fontsize=NUM, handlelength=1.2, borderaxespad=0.1)
    ax.set_ylim(2490, 3210)
    panel_letter(fig, ax, "e", W, x_mm=76)
    # f: median surface speed 2014-2021 on the ITS_LIVE pixels, posterior against all samples, and the bound
    ax = axes(fig, H, 14, 10, 58, 30)
    Wt = {r["name"]: float(r["w_main"]) for r in rows("f3_weights.csv")}
    sp = {}
    for f in ("f3_runs_f3_r1.csv", "f3_runs_f3_r2.csv", "f3_runs_f3_r1fix.csv"):
        for r in rows(f):
            if r["name"].endswith("_12") and r["speed"]:
                sp[r["name"][:-3]] = float(r["speed"])
    names = sorted(set(sp) & set(Wt))
    s = np.array([sp[n] for n in names])
    w = np.array([Wt[n] for n in names])
    bins = np.logspace(-2, 2, 41)
    ax.hist(s, bins=bins, weights=np.full(len(s), 1 / len(s)), color="#d9d9d9", lw=0, label="All samples")
    ax.hist(s, bins=bins, weights=w / w.sum(), color=PALE_BLUE, lw=0, alpha=0.9, label="Posterior")
    ax.axvline(1.0, color=ACCENT, lw=0.8, label="Bound, 1 m per year")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 100)
    ax.set_xlabel("Median surface speed 2014–2021 (m per year)", labelpad=1.5)
    ax.set_ylabel("Share of weight")
    ax.legend(loc="upper right", fontsize=NUM, borderaxespad=0.1)
    panel_letter(fig, ax, "f", W)
    # g: standardised misfit of every start-state term
    ax = axes(fig, H, 118, 10, 28, 30)
    groups = (("area", "Ice area"), ("dh2", "Elevation change"), ("radar", "Radar bedrock 2006"),
              ("stake", "Stake melt bound"), ("speed", "Surface speed bound"))
    ax.axvspan(-2, 2, color=ICE, lw=0, zorder=0)
    for j, (g, name) in enumerate(groups):
        zz = np.array([float(r["median_minus_observed_in_sigma"]) for r in T if r["term"] == g])
        yy = j + np.linspace(-0.15, 0.15, len(zz))
        ax.plot(zz[zz <= 6], yy[zz <= 6], "o", ms=2.4, mfc="white", mec=GLACIER, mew=0.6)
        ax.plot([6.3] * int((zz > 6).sum()), yy[zz > 6], ">", ms=2.6, color=GLACIER, mew=0)   # bound met by far
    ax.axvline(0, color=RULE, lw=0.4)
    ax.set_yticks(range(len(groups)), [n for _, n in groups])
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.set_xlim(-6, 6.8)
    ax.set_xlabel("Model minus observed\n(standard deviations)", labelpad=1.5)
    panel_letter(fig, ax, "g", W, x_mm=76)
    save(fig, "igm_calibration")


# --- volume and area --------------------------------------------------------------------------------------------------

BREAK, SQUEEZE = 2050.0, 0.3        # the time axis runs at 0.3 of its scale after 2050 (marked by a thin line)


def squeeze(ax, end):
    f = lambda x: np.where(x <= BREAK, x, BREAK + (x - BREAK) * SQUEEZE)
    g = lambda x: np.where(x <= BREAK, x, BREAK + (x - BREAK) / SQUEEZE)
    ax.set_xscale("function", functions=(f, g))
    if end > BREAK:
        ax.axvline(BREAK, color=RULE, lw=0.4, zorder=0)
    ax.set_xticks([t for t in (2030, 2040, 2050, 2075, 2100) if t <= end])
    ax.set_xticks([t for t in (2025, 2035, 2045) if t <= end], minor=True)


def axis_end(Y, ssps, levels):
    """End of a projection time axis: latest 95th-percentile disappearance year (0.01 km2) of the groups shown + 3, as a
    1 October date; 'not by 2100' (2101) caps it at 2100.75."""
    p95 = max(float(Y[("with flow", "dis_001", s, l)]["p95"]) for s in ssps for l in levels)
    return min(np.ceil(p95) + 3, 2100) + 0.75


def volume_area():
    Q = defaultdict(list)
    for r in rows("f7_series_quantiles.csv"):
        Q[(r["ssp"], r["level"], r["variable"])].append(r)
    Y = {(r["run"], r["quantity"], r["ssp"], r["level"]): r for r in rows("f7_years.csv")}
    T = {r["value"]: r for r in rows("f3_terms.csv") if r["term"] == "area"}
    H = 100.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    g = lambda rs, p: np.array([float(r[p]) for r in rs])
    for j, ssp in enumerate(SSPS):
        end = axis_end(Y, [ssp], ("anchored", "native"))
        for i, (var, lab, top) in enumerate((("V", "Volume (km³)", 0.03), ("A1", "Ice area (km²)", 1.75))):
            ax = axes(fig, H, 14 + j * 45, 52 - i * 42, 40, 34)
            a = Q[(ssp, "anchored", var)]
            y = np.array([int(r["year"]) for r in a]) + 0.75
            k = y <= end
            ax.fill_between(y[k], g(a, "p5")[k], g(a, "p95")[k], color=ICE, lw=0)
            ax.fill_between(y[k], g(a, "p17")[k], g(a, "p83")[k], color=PALE_BLUE, lw=0)
            ax.plot(y[k], g(a, "p50")[k], color=GLACIER, lw=1.1)
            n = Q[(ssp, "native", var)]
            ax.plot(y[k], g(n, "p50")[k], color=INK, lw=0.8, ls=(0, (3, 1.5)))
            ax.plot(y[k], g(n, "p95")[k], color=GREY, lw=0.5, ls=(0, (1, 1.2)))
            if var == "A1":
                ax.errorbar([2021.75], [1.6065], [0.0], fmt="o", ms=2.6, mfc="white", mec=INK, zorder=5)
                for key, d in (("area 2024-08-23 (all)", 2024.64), ("area 2025-09-07 (body)", 2025.68),
                               ("area 2026-09-27 (body)", 2026.74)):
                    ax.errorbar([d], [float(T[key]["observed"])], [float(T[key]["sigma"])], fmt="o", ms=2.6,
                                color=INK, elinewidth=0.5, capsize=0, zorder=5)
            squeeze(ax, end)
            ax.set_xlim(2021, end), ax.set_ylim(0, top)
            if j == 0:
                ax.set_ylabel(lab)
            else:
                ax.set_yticklabels([])
            if i == 1:
                ax.set_xlabel("Hydrological year", labelpad=1.5)
            else:
                ax.set_xticklabels([])
                ax.text(0.5, 1.04, ssp, transform=ax.transAxes, ha="center", va="bottom", fontsize=LAB, color=INK)
    h = [plt.matplotlib.patches.Patch(color=PALE_BLUE, lw=0, label="Anchored 2026, 17–83 %"),
         plt.matplotlib.patches.Patch(color=ICE, lw=0, label="Anchored 2026, 5–95 %"),
         plt.Line2D([], [], color=GLACIER, lw=1.1, label="Anchored 2026, median"),
         plt.Line2D([], [], color=INK, lw=0.8, ls=(0, (3, 1.5)), label="Native 2026, median"),
         plt.Line2D([], [], color=GREY, lw=0.5, ls=(0, (1, 1.2)), label="Native 2026, 95 %"),
         plt.Line2D([], [], color=INK, marker="o", ms=2.6, ls="", label="Observed area"),
         plt.Line2D([], [], color=INK, marker="o", ms=2.6, mfc="white", ls="", label="Start 2021, mapped")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=NUM,
               columnspacing=1.2, handlelength=1.8)
    save(fig, "igm_volume_area")


# --- years ------------------------------------------------------------------------------------------------------------

def interval(ax, r, y, c, ms=3.2):
    v = [float(r[f"p{x}"]) for x in (5, 17, 50, 83, 95)]
    ax.plot([v[0], v[4]], [y] * 2, color=c, lw=0.6, solid_capstyle="butt")
    ax.plot([v[1], v[3]], [y] * 2, color=c, lw=2.4, solid_capstyle="butt")
    ax.plot([v[2]], [y], "o", ms=ms, color="white", mec=c, mew=0.9)


def years():
    Y = {(r["run"], r["quantity"], r["ssp"], r["level"]): r for r in rows("f7_years.csv")}
    S = rows("f7_spread.csv")
    H = 122.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    quants = (("half_2026", "Half the 2026 volume"), ("dis_01", "Area below 0.1 km²"),
              ("dis_001", "Area below 0.01 km²"))
    # a: intervals per quantity, SSP and level
    ax = axes(fig, H, 22, 62, 124, 56)
    pos, labels, p = [], [], 0.0
    for q, qname in quants:
        ax.text(2025.4, p - 0.75, qname, ha="left", va="center", fontsize=LAB, fontweight=600, color=INK)
        for ssp in SSPS:
            for lev, c, off in (("anchored", GLACIER, -0.17), ("native", GREY, 0.17)):
                interval(ax, Y[("with flow", q, ssp, lev)], p + off, c)
            pos.append(p)
            labels.append(ssp)
            p += 1
        p += 1.0
    ax.set_yticks(pos, labels)
    ax.set_ylim(p - 1.3, -1.3)
    ax.set_xlim(2025, 2102)
    ax.set_xticks([2030, 2040, 2050, 2060, 2070, 2080, 2090, 2101], ["2030", "2040", "2050", "2060", "2070",
                                                                      "2080", "2090", "After\n2100"])
    ax.set_xlabel("Hydrological year", labelpad=1.5)
    h = [plt.Line2D([], [], color=GLACIER, lw=2.4, label="Anchored 2026"),
         plt.Line2D([], [], color=GREY, lw=2.4, label="Native 2026")]
    ax.legend(handles=h, loc="lower right", fontsize=NUM, borderaxespad=0.2)
    panel_letter(fig, ax, "a", W)
    # b: share of the spread per factor (all 9984 members)
    ax = axes(fig, H, 32, 10, 52, 32)
    cats = (("Emission scenario", ("SSP",), "#08306b"),
            ("Climate model chain", ("chain",), "#2171b5"),
            ("2026 level", ("  of which 2026 level",), "#6baed6"),
            ("Station", ("  of which station", "  of which 2026 level x station"), "#c6dbef"),
            ("Balance, thickness, flow", ("  of which draws within a level and station (balance, u, f)",), "#a50f15"),
            ("Draw × chain", ("draw x chain",), "#fb6a4a"),
            ("Other two-way", ("draw x SSP", "chain x SSP"), "#fcbba1"),
            ("Three-way and weather", ("draw x chain x SSP and weather order",), "#bdbdbd"))
    for k, (q, qname) in enumerate(quants):
        left = 0.0
        for name, keys, c in cats:
            v = sum(float(r["share_of_variance"]) for r in S if r["members"] == "all" and r["quantity"] == q
                    and r["factor"] in keys)
            ax.barh(k, v * 100, left=left, color=c, height=0.62, lw=0, label=name if k == 0 else None)
            left += v * 100
    ax.set_yticks(range(len(quants)), [n for _, n in quants])
    ax.set_ylim(len(quants) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the variance (%)", labelpad=1.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=NUM, ncol=1, handlelength=1.0,
              borderaxespad=0)
    panel_letter(fig, ax, "b", W)
    # c: with against without ice flow, all 9984 members: share whose year is earlier or later with flow (the rest:
    # the same year)
    ax = axes(fig, H, 124, 10, 22, 32)
    for k, (q, qname) in enumerate(quants):
        r = Y[("IGM minus no-flow", q, "all SSPs", "both")]
        e, la = float(r["share_earlier"]) * 100, float(r["share_later"]) * 100
        ax.barh(k, e, color=GLACIER, height=0.62, lw=0)
        ax.barh(k, la, left=e, color=ACCENT, height=0.62, lw=0)
    ax.set_yticks(range(len(quants)), [])
    ax.set_ylim(len(quants) - 0.5, -0.5)
    ax.set_xlim(0, 6)
    ax.set_xticks([0, 2, 4, 6])
    ax.set_xlabel("Members with another\nyear without flow (%)", labelpad=1.5)
    h = [plt.matplotlib.patches.Patch(color=GLACIER, lw=0), plt.matplotlib.patches.Patch(color=ACCENT, lw=0)]
    ax.legend(handles=h, labels=["Earlier with flow", "Later with flow"], loc="lower left",
              bbox_to_anchor=(-0.1, 1.02), fontsize=NUM, handlelength=1.0, borderaxespad=0)
    panel_letter(fig, ax, "c", W, x_mm=118)
    save(fig, "igm_years")


# --- half-volume year against V_2021 as a function of u ---------------------------------------------------------------

def half_volume_u():
    Sm = rows("f7_half2021_samples.csv")
    M = [m for m in rows("members.csv") if m["map_u0"] == "0"]
    H = 60.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    u = np.array([float(r["u"]) for r in Sm])
    v21 = np.array([float(r["V_2021"]) for r in Sm])
    hy = np.array([int(r["half_2021_by_2026"]) if r["half_2021_by_2026"] else 2027 for r in Sm])
    by = defaultdict(list)
    for m in M:
        by[int(m["draw"])].append(m)
    du = np.array([float(v[0]["u"]) for v in by.values()])
    dv = np.array([float(v[0]["V_2021"]) for v in by.values()])
    ax = axes(fig, H, 14, 10, 58, 44)
    ax.plot(u, v21, "o", ms=1.4, color=RULE, mew=0, label="Start-state samples")
    ax.plot(du, dv, "o", ms=3.0, mfc="white", mec=GLACIER, mew=0.7, label="Ensemble draws")
    ax.set_xlabel("Start-thickness factor u", labelpad=1.5)
    ax.set_ylabel("Volume 2021 (km³)")
    ax.set_xlim(-1.05, 2.05), ax.set_ylim(0, None)
    ax.legend(loc="upper left", fontsize=NUM, borderaxespad=0.2)
    panel_letter(fig, ax, "a", W)
    ax = axes(fig, H, 88, 10, 58, 44)
    rng = np.random.default_rng(0)
    ax.plot(u, hy + rng.uniform(-0.3, 0.3, len(hy)), "o", ms=1.4, color=RULE, mew=0, label="Start-state samples")
    for k, v in enumerate(by.values()):
        z = np.array([int(m["half_2021"]) if m["half_2021"] else 2101 for m in v])
        lo, med, hi = np.percentile(z, [5, 50, 95])
        uu = float(v[0]["u"])
        ax.plot([uu, uu], [lo, hi], color=GLACIER, lw=0.6)
        ax.plot([uu], [med], "o", ms=3.0, mfc="white", mec=GLACIER, mew=0.7, label="Ensemble draws" if k == 0
                else None)
    ax.set_yticks([2022, 2023, 2024, 2025, 2026, 2027], ["2022", "2023", "2024", "2025", "2026", "After 2026"])
    ax.axhspan(2026.5, 2027.5, color=ICE, lw=0, zorder=0)
    ax.set_xlabel("Start-thickness factor u", labelpad=1.5)
    ax.set_ylabel("Half the 2021 volume (year)")
    ax.set_xlim(-1.05, 2.05), ax.set_ylim(2021.5, 2027.6)
    ax.legend(loc="lower right", fontsize=NUM, borderaxespad=0.2)
    panel_letter(fig, ax, "b", W, x_mm=76)
    save(fig, "igm_half_volume_u")


# --- ice-free year per cell -------------------------------------------------------------------------------------------

def icefree_map():
    from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap
    Z = np.load(RES / "f7_icefree_maps.npz")
    x, y = Z["x"].astype(float), Z["y"].astype(float)
    key = "all SSPs|anchored"
    p5, p50, p95 = Z[key + "|flow"]
    share = Z[key + "|ice2026"]
    db = ROOT / "data/vectors.gpkg"
    outl = [("1850", gpkg(db, "sgi_1850", "WHERE SGI = 'C14-10'"), INK, (0, (4, 1.5)), 0.6),
            ("1929", gpkg(db, "sgi_1931", "WHERE SGI = 'C14-10'"), GREY, (0, (4, 1, 1, 1)), 0.5),
            ("1973", gpkg(db, "sgi_1973", "WHERE SGI = 'C14-10'"), GREY, (0, (1, 1)), 0.6),
            ("2010", gpkg(db, "sgi_2010", "WHERE SGI = 'C14-10'"), INK, (0, (2, 1)), 0.45),
            ("2018", gpkg(db, "sgi_2016", "WHERE \"sgi-id\" = 'C14-10'"), INK, "-", 0.45),
            ("2024", gpkg(db, "glacier_2024", "WHERE sgi = 'C14-10'"), GLACIER, "-", 0.6),
            ("2026", gpkg(ROOT / "data/outline_2026/outline_2026-09-27.gpkg", "outline_2026"), ACCENT, "-", 0.75)]
    e0, e1, n0, n1 = 2678950, 2681750, 1140250, 1142450
    hs = LightSource(azdeg=315, altdeg=40).hillshade(Z["usurf_2021"][::-1], dx=10, dy=10)
    ext = (x[0] - 5, x[-1] + 5, y[0] - 5, y[-1] + 5)
    show = share > 0
    mw = 62.0
    mh = mw * (n1 - n0) / (e1 - e0)
    H = mh + 34
    fig = plt.figure(figsize=(W * MM, H * MM))
    bounds = [2026.5, 2028.5, 2030.5, 2032.5, 2034.5, 2036.5, 2038.5, 2040.5, 2045.5, 2101.5]
    cm = ListedColormap(plt.get_cmap("viridis_r")(np.linspace(0.05, 0.95, len(bounds) - 1)))
    rbounds = [-0.5, 2.5, 5.5, 10.5, 20.5, 40.5, 80]
    rcm = ListedColormap(plt.get_cmap("magma_r")(np.linspace(0.08, 0.85, len(rbounds) - 1)))
    for k, (z, cmap, bnd, letter) in enumerate(((p50, cm, bounds, "a"), (p95 - p5, rcm, rbounds, "b"))):
        x0 = 14 + k * (mw + 6)
        ax = axes(fig, H, x0, 22, mw, mh)
        ax.imshow(hs, extent=ext, cmap="gray", vmin=-0.3, vmax=1.3, interpolation="bilinear", zorder=0)
        zz = np.where(show & (p50 >= 2027) if k == 0 else show, z, np.nan)
        im = ax.imshow(zz[::-1], extent=ext, cmap=cmap, norm=BoundaryNorm(bnd, cmap.N), interpolation="nearest",
                       alpha=0.92, zorder=1)
        if k == 0:
            g = np.where(show & (p50 < 2027), 1.0, np.nan)
            ax.imshow(g[::-1], extent=ext, cmap=ListedColormap(["#ffffff"]), alpha=0.8, interpolation="nearest",
                      zorder=1)
        for lab, geoms, c, st, lw in outl:
            for gm in geoms:
                for part in getattr(gm, "geoms", [gm]):
                    for ring in [part.exterior, *part.interiors]:
                        ax.plot(*ring.xy, color=c, ls=st, lw=lw, zorder=3)
        ax.set_xlim(e0, e1), ax.set_ylim(n0, n1)
        ax.set_xticks([2679000, 2680000, 2681000], ["2679", "2680", "2681"])
        ax.set_yticks([1140500, 1141000, 1141500, 1142000], ["1140.5", "1141.0", "1141.5", "1142.0"])
        ax.set_xlabel("Easting (km)", labelpad=1.5)
        if k == 0:
            ax.set_ylabel("Northing (km)")
        else:
            ax.set_yticklabels([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(True)
        cax = axes(fig, H, x0 + 2, 8, mw - 4, 2.2)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_linewidth(0.4)
        if k == 0:
            cb.set_ticks(bounds[:-1], labels=["2027", "2029", "2031", "2033", "2035", "2037", "2039", "2041",
                                              "2046"])
            cb.set_label("Median ice-free year (first year of its bin)", labelpad=1.5)
        else:
            cb.set_ticks([1, 4, 8, 15.5, 30.5, 60], labels=["0–2", "3–5", "6–10", "11–20", "21–40", "Over 40"])
            cb.set_label("Range of 90 % of members (years)", labelpad=1.5)
        panel_letter(fig, ax, letter, W, x_mm=1.2 + k * (mw + 6) + (8 if k else 0))
    h = [plt.Line2D([], [], color=c, ls=st, lw=lw + 0.2, label=lab) for lab, _, c, st, lw in outl]
    h.append(plt.matplotlib.patches.Patch(facecolor="white", edgecolor=GREY, lw=0.4, label="Ice-free by 2026"))
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=8, fontsize=NUM, columnspacing=0.9,
               handlelength=2.0)
    save(fig, "igm_icefree_map")


# --- against the published projections --------------------------------------------------------------------------------

STATEMENTS = (("Huss 2026, interview", 2036, 2046), ("Canton Ticino 2023, fact sheet", 2043, 2043),
              ("Kappenberger 2025, interview", 2030, 2030))    # PUBLISHED_PROJECTIONS.md, 'Other projections'
VT_LEVELS = (("15", "+1.5 K"), ("20", "+2.0 K"), ("27", "+2.7 K"), ("40", "+4.0 K"))
SCEN = {"SSP1-2.6": ("ssp126", "rcp26", "RCP2.6"), "SSP2-4.5": ("ssp245", "rcp45", "RCP4.5"),
        "SSP5-8.5": ("ssp585", "rcp85", "RCP8.5")}


def published_stats(P, source, scen, key):
    """Median (lower middle for an even count), smallest and largest year over the members of a published series;
    an empty value (not reached in the file) counts as after 2100 (2101)."""
    z = sorted(int(r[key]) if r[key] else 2101 for r in P if r["source"] == source and r["scenario"] == scen)
    return (z[(len(z) - 1) // 2], z[0], z[-1]) if z else None


def published():
    Q = defaultdict(list)
    for r in rows("f7_series_quantiles.csv"):
        Q[(r["ssp"], r["level"], r["variable"])].append(r)
    Y = {(r["run"], r["quantity"], r["ssp"], r["level"]): r for r in rows("f7_years.csv")}
    P = list(csv.DictReader(open(PUB / "basodino_members.csv")))
    Vp = defaultdict(dict)
    for r in csv.DictReader(open(PUB / "basodino_volume.csv")):
        Vp[(r["source"], r["scenario"], r["member"])][int(r["year"])] = float(r["volume_km3"])
    VT = next(csv.DictReader(open(PUB / "vantricht2025/basodino_extinction.csv")))
    H = 150.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    # a: volume 2000-2100, SSP2-4.5 (Zekollari: RCP4.5); published values are 1 January states, ours 30 September
    ax = axes(fig, H, 14, 106, 132, 28)
    for source, c, pale, name in (("oggm2023", INK, "#d9d9d9", "Open Global Glacier Model 2023"),
                                  ("rounce2023", ACCENT, PALE_RED, "Rounce 2023")):
        S = [v for (s, sc, m), v in Vp.items() if s == source and sc == "ssp245"]
        yrs = sorted(set.intersection(*[set(v) for v in S]))
        z = np.array([[v[y] for y in yrs] for v in S])
        ax.fill_between(yrs, z.min(0), z.max(0), color=pale, lw=0, alpha=0.7)
        ax.plot(yrs, np.median(z, 0), color=c, lw=0.9, label=f"{name}, median and range of the climate models")
    zk = Vp[("zekollari2019", "rcp45", "mean")]
    ax.plot(sorted(zk), [zk[y] for y in sorted(zk)], color=GREY, lw=0.9, ls=(0, (3, 1.5)),
            label="Zekollari 2019, RCP4.5, mean of the climate models")
    a = Q[("SSP2-4.5", "anchored", "V")]
    y = np.array([int(r["year"]) for r in a]) + 0.75
    lo, med, hi = (np.array([float(r[p]) for r in a]) for p in ("p5", "p50", "p95"))
    ax.fill_between(y, np.maximum(lo, 1e-5), np.maximum(hi, 1e-5), color=PALE_BLUE, lw=0)
    ax.plot(y, np.where(med > 0, med, np.nan), color=GLACIER, lw=1.1, label="This ensemble, anchored 2026, median "
                                                                           "and 5–95 %")
    n = Q[("SSP2-4.5", "native", "V")]
    nm = np.array([float(r["p50"]) for r in n])
    ax.plot(y, np.where(nm > 0, nm, np.nan), color=GLACIER, lw=0.8, ls=(0, (3, 1.5)),
            label="This ensemble, native 2026, median")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 0.2)
    ax.set_xlim(2000, 2100.75)
    ax.set_xlabel("Year", labelpad=1.5)
    ax.set_ylabel("Volume (km³)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=NUM, borderaxespad=0, ncol=2, columnspacing=1.0,
              handlelength=1.8)
    panel_letter(fig, ax, "a", W)
    # b, c: years per scenario; rows: this ensemble (anchored, native), the three models
    rows_b, labels, p = [], [], 0.0
    srcs = (("oggm2023", "Open Global Glacier Model 2023", "area_below_0.01", INK),
            ("rounce2023", "Rounce 2023 (volume below 1 %)", "below_1pct_of_first", ACCENT),
            ("zekollari2019", "Zekollari 2019 (volume below 1 %)", "below_1pct_of_first", GREY))
    heads = []
    for ssp in SSPS:
        heads.append((p - 0.8, ssp))
        for lev in ("anchored", "native"):
            rows_b.append((p, ("ours", ssp, lev)))
            labels.append(f"This ensemble, {lev} 2026")
            p += 1
        for source, name, key, c in srcs:
            scen = SCEN[ssp][1] if source == "zekollari2019" else SCEN[ssp][0]
            rows_b.append((p, (source, scen, key, c)))
            labels.append(name.replace(" (", f", {SCEN[ssp][2]} (") if source == "zekollari2019" else name)
            p += 1
        p += 0.9
    heads.append((p - 0.8, "Warming levels"))
    for k, lab in VT_LEVELS:
        rows_b.append((p, ("vt", k)))
        labels.append(f"Van Tricht 2025, {lab} (area or volume)")
        p += 1
    p += 0.9
    heads.append((p - 0.8, "Statements"))
    for name, a0, a1 in STATEMENTS:
        rows_b.append((p, ("st", a0, a1)))
        labels.append(name)
        p += 1
    top = p - 0.3
    for letter, quantity, xlim, x0, wd, pkey in (("b", "dis_001", (2025, 2102.5), 56, 62, None),
                                                  ("c", "half_2026", (2026, 2062), 122, 24, "half_of_2026")):
        cap = 2101.5 if letter == "b" else xlim[1] - 0.5
        ax = axes(fig, H, x0, 10, wd, 88)
        for yy, spec in rows_b:
            if spec[0] == "ours":
                interval(ax, Y[("with flow", quantity, spec[1], spec[2])], yy,
                         GLACIER if spec[2] == "anchored" else GREY, ms=3.0)
            elif spec[0] in ("oggm2023", "rounce2023", "zekollari2019"):
                st = published_stats(P, spec[0], spec[1], pkey or spec[2])
                if st:
                    m, a0, a1 = st
                    ax.plot([a0, min(a1, cap)], [yy] * 2, color=spec[3], lw=0.8, solid_capstyle="butt")
                    if m <= cap:
                        ax.plot([m], [yy], "o", ms=3.0, color=spec[3], mew=0)
                    if a1 > cap:
                        ax.plot([cap + 0.3], [yy], marker=">", ms=2.6, color=spec[3], mew=0)
            elif spec[0] == "vt" and letter == "b":
                m, q1, q3 = (num(VT[f"{s}_{spec[1]}"]) for s in ("median", "Q1", "Q3"))
                q3 = 2101 if np.isnan(q3) else q3
                ax.plot([q1, min(q3, 2101.5)], [yy] * 2, color=GREY, lw=2.0, solid_capstyle="butt")
                ax.plot([m], [yy], "o", ms=3.0, color="white", mec=GREY, mew=0.8)
                if q3 >= 2101:
                    ax.plot([2101.8], [yy], marker=">", ms=2.6, color=GREY, mew=0)
            elif spec[0] == "st" and letter == "b":
                a0, a1 = spec[1], spec[2]
                if a1 > a0:
                    ax.plot([a0, a1], [yy] * 2, color=ACCENT, lw=2.0, solid_capstyle="butt", alpha=0.5)
                ax.plot([(a0 + a1) / 2], [yy], "D", ms=2.6, color=ACCENT, mew=0)
        ax.set_ylim(top, -1.4)
        ax.set_xlim(*xlim)
        if letter == "b":
            ax.set_yticks([yy for yy, _ in rows_b], labels)
            ax.tick_params(axis="y", length=0)
            for yy, h in heads:
                ax.text(-0.02, yy, h, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=LAB,
                        fontweight=600, color=INK)
            ax.set_xticks([2030, 2040, 2050, 2060, 2070, 2080, 2090, 2101.8],
                          ["2030", "2040", "2050", "2060", "2070", "2080", "2090", "After\n2100"])
            ax.set_xlabel("Disappearance year", labelpad=1.5)
        else:
            ax.set_yticks([yy for yy, _ in rows_b], [])
            ax.tick_params(axis="y", length=0)
            ax.set_xticks([2030, 2040, 2050, 2060])
            ax.set_xlabel("Half the 2026\nvolume (year)", labelpad=1.5)
        for yy, _ in heads:
            ax.axhline(yy + 0.35, color=RULE, lw=0.3, zorder=0)
        panel_letter(fig, ax, letter, W, x_mm=1.2 if letter == "b" else 119)
    save(fig, "igm_published")


if __name__ == "__main__":
    ALL = {"igm_hindcast": hindcast, "igm_calibration": calibration, "igm_volume_area": volume_area,
           "igm_years": years, "igm_half_volume_u": half_volume_u, "igm_icefree_map": icefree_map,
           "igm_published": published}
    for n in sys.argv[1:] or ALL:
        ALL[n]()
