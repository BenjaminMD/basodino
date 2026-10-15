"""Flow of the modelled glacier (the film member of RESULTS.md section 13), house style of code/common.py, drawn at
150 mm, graphics only (tex/SIDECARS.md). Reads results/f7/film_states.npz (animate.py states: yearly 1 October
thickness and surface velocity 1850-2042, the bed, the hindcast's own 2021 state), data/thickness/sections.csv
(section A, summit to tongue, of code/thickness.py) and data/vectors.gpkg (the 1850 outline). Writes
tex/figures/igm_flow.pdf.
  a  the 1850 state: surface speed in steps of 5 m per year and streamlines of the surface velocity, on a hillshade
     of the bed; red: the 1850 outline; dashed: section A
  b  ice thickness along section A (the bed is the base line; its elevation on the top axis): the hindcast ice lost
     between 1850, 1900, 2000 and 2021 as grey bands labelled with their years, the hindcast ice of 2021 (pale blue)
     and the ice of the 2021 start from the measured thickness (red)
  c  surface speed against time, median and 5-95 % of the ice-covered cells weighted by thickness: the hindcast
     1850-2021 and the run restarted in 2021 from the measured thickness, to the year the ice area falls below
     0.01 km²
usage (Mac): python igm/flow_figure.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

IGM = Path(__file__).resolve().parent
sys.path.insert(0, str(IGM))
from figures import (ACCENT, GLACIER, INK, LAB, NUM, PALE_BLUE, PALE_RED, RULE, W, axes, gpkg,  # noqa: E402
                     panel_letter, plt, save)

ROOT = IGM.parent
E0, N0, DX = 2678705.0, 1140005.0, 10.0                 # cell centres of the 10 m domain (row 0 is south)
BANDS = (1850, 1900, 2000, 2021)                        # hindcast surfaces bounding the bands of b
LEVELS = np.arange(0, 31, 5)                            # m per year, the speed steps of a

Z = dict(np.load(IGM / "results/f7/film_states.npz"))
topg, thk, us, vs, years = Z["topg"], Z["thk"], Z["us"], Z["vs"], Z["years"]
ny, nx = topg.shape
x_km, y_km = (E0 + DX * np.arange(nx)) / 1e3, (N0 + DX * np.arange(ny)) / 1e3
cmap = plt.get_cmap("Blues")


def sample(field, e, n):
    """Bilinear value of a grid at easting e, northing n (arrays)."""
    return map_coordinates(field, [(n - N0) / DX, (e - E0) / DX], order=1, mode="nearest")


def hillshade(z, az=315, alt=45):
    gy, gx = np.gradient(z, DX)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, b = np.radians(az), np.radians(alt)
    return np.clip(np.sin(b) * np.cos(slope) + np.cos(b) * np.sin(slope) * np.cos(a - aspect), 0, 1)


def speed_quantiles(h, u, v):
    """5, 50, 95 % of the surface speed over the cells with at least 1 m of ice, each cell weighted by its thickness
    (so that stagnant snow patches of a few metres do not set the lower bound); NaN below 0.01 km² of ice."""
    ice = h >= 1
    if ice.sum() * DX * DX < 1e4:
        return np.full(3, np.nan)
    o = np.argsort(np.hypot(u, v)[ice])
    c = np.cumsum(h[ice][o])
    return np.interp([0.05, 0.5, 0.95], c / c[-1], np.hypot(u, v)[ice][o])


rows = [r for r in csv.DictReader(open(ROOT / "data/thickness/sections.csv")) if r["section"] == "A"]
dA, eA, nA = (np.array([float(r[k]) for r in rows]) for k in ("distance_m", "easting_m", "northing_m"))
bed = sample(topg, eA, nA)
hind = {y: thk[y - 1850] for y in BANDS[:-1]} | {2021: Z["thk_hind_2021"]}
thick = {y: np.where(sample(h, eA, nA) >= 1, sample(h, eA, nA), 0) for y, h in hind.items()}
H21 = np.where(sample(thk[171], eA, nA) >= 1, sample(thk[171], eA, nA), 0)
bed_mono = np.minimum.accumulate(bed)                    # the bed, made monotone for the elevation axis of b

ice = thk[0] >= 1
rows_i, cols_i = np.where(ice)
e_lim = (x_km[cols_i.min()] - 0.12, x_km[cols_i.max()] + 0.15)
n_lim = (y_km[rows_i.min()] - 0.12, y_km[rows_i.max()] + 0.12)
w_a, h_b = 64.0, 38.0
h_a = w_a * (n_lim[1] - n_lim[0]) / (e_lim[1] - e_lim[0])
y_b = 11
y_a = y_b + h_b + 27
H = y_a + h_a + 4
fig = plt.figure(figsize=(W / 25.4, H / 25.4))

# --- a: the 1850 state, speed steps and streamlines
ax = axes(fig, H, 9, y_a, w_a, h_a)
ax.imshow(hillshade(topg), cmap="gray", vmin=-0.3, vmax=1.3, extent=(x_km[0] - 0.005, x_km[-1] + 0.005,
          y_km[0] - 0.005, y_km[-1] + 0.005), origin="lower", interpolation="bilinear")
speed = np.where(ice, np.hypot(us[0], vs[0]), np.nan)
cf = ax.contourf(x_km, y_km, speed, levels=LEVELS, colors=cmap(np.linspace(0.15, 0.95, len(LEVELS) - 1)), alpha=0.9)
ax.contour(x_km, y_km, speed, levels=LEVELS[1:], colors="white", linewidths=0.3)
ax.streamplot(x_km, y_km, np.ma.masked_where(~ice, us[0]), np.ma.masked_where(~ice, vs[0]), density=0.75,
              color=INK, linewidth=0.35, arrowsize=0.5, arrowstyle="->", maxlength=0.6)
for g in gpkg(ROOT / "data/vectors.gpkg", "sgi_1850", "WHERE SGI = 'C14-10'"):
    for p in getattr(g, "geoms", [g]):
        ax.plot(*(np.array(p.exterior.coords).T / 1e3), color=ACCENT, lw=0.6)
ax.plot(eA / 1e3, nA / 1e3, color=INK, lw=0.5, ls=(0, (3, 2)))
ax.text(eA[0] / 1e3 - 0.01, nA[0] / 1e3 - 0.01, "A", fontsize=LAB, fontweight=600, ha="right", va="top", color=INK)
ax.text(eA[-1] / 1e3 - 0.01, nA[-1] / 1e3 + 0.02, "A′", fontsize=LAB, fontweight=600, ha="right", va="bottom",
        color=INK)
ax.set_xlim(*e_lim), ax.set_ylim(*n_lim)
ax.set_xticks([2679, 2680, 2681]), ax.set_yticks([1140.5, 1141, 1141.5, 1142, 1142.5])
ax.set_xlabel("Easting (km)", labelpad=1.5)
ax.set_ylabel("Northing (km)")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(True)
cax = axes(fig, H, 13, y_a - 11, w_a - 8, 2.2)
cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=LEVELS)
cb.set_label("Surface speed 1850 (m per year)", fontsize=LAB, labelpad=1.5)
cb.outline.set_linewidth(0.4)
panel_letter(fig, ax, "a", W)

# --- c: surface speed against time, hindcast and the restart
ax = axes(fig, H, 90, y_a - 11, 56, h_a + 11)
q_h = np.array([speed_quantiles(thk[k], us[k], vs[k]) for k in range(171)] +
               [speed_quantiles(Z["thk_hind_2021"], Z["us_hind_2021"], Z["vs_hind_2021"])])
q_r = np.array([speed_quantiles(thk[k], us[k], vs[k]) for k in range(171, len(years))])
for yr, q, c, pale in ((np.arange(1850, 2022), q_h, GLACIER, PALE_BLUE), (years[171:], q_r, ACCENT, PALE_RED)):
    ax.fill_between(yr, q[:, 0], q[:, 2], color=pale, lw=0)
    ax.plot(yr, q[:, 1], color=c, lw=1.0)
ax.axvline(2021, color=RULE, lw=0.4, zorder=0)
ax.text(1853, 30, "Hindcast", fontsize=LAB, color=GLACIER, ha="left", va="bottom")
ax.text(2017, 0.12, "Restart 2021,\nmeasured thickness", fontsize=LAB, color=ACCENT, ha="right", va="center",
        linespacing=1.1)
ax.set_yscale("log")
ax.set_ylim(0.005, 50), ax.set_xlim(1845, 2045)
ax.set_yticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
ax.set_xticks([1850, 1900, 1950, 2000, 2042])
ax.set_xlabel("Year", labelpad=1.5)
ax.set_ylabel("Surface speed (m per year)")
panel_letter(fig, ax, "c", W, x_mm=78)

# --- b: thickness along section A, the bands of the hindcast ice and the 2021 start
ax = axes(fig, H, 13, y_b, 128, h_b)
for (y0, y1), c in zip(zip(BANDS[:-1], BANDS[1:]), ("#e4e4e4", "#c6c6c6", "#a6a6a6")):
    ax.fill_between(dA, thick[y1], thick[y0], where=thick[y0] > thick[y1], color=c, lw=0)
ax.fill_between(dA, 0, thick[2021], color=PALE_BLUE, lw=0)
ax.fill_between(dA, 0, H21, color=PALE_RED, lw=0)
ax.plot(dA, np.where(H21 > 0, H21, np.nan), color=ACCENT, lw=0.7)
for y in BANDS:
    ax.plot(dA, np.where(thick[y] > 0, thick[y], np.nan), color="white", lw=0.3)
for d, z, lab, c in ((1250, 57, "1850–1900", INK), (1830, 33, "1900–2000", INK), (1150, 38, "2000–2021", INK),
                     (1560, 22, "2021, hindcast", INK), (700, 11, "2021, measured", ACCENT)):
    ax.text(d, z, lab, fontsize=NUM, color=c, ha="center", va="center")
ax.axhline(0, color=INK, lw=0.7)
ax.set_xlim(0, dA[-1] + 20), ax.set_ylim(0, 75)
ax.set_xlabel("Distance along A (m)", labelpad=1.5)
ax.set_ylabel("Ice thickness (m)")
top = ax.secondary_xaxis("top", functions=(lambda d: np.interp(d, dA, bed_mono),
                                           lambda z: np.interp(z, bed_mono[::-1], dA[::-1])))
top.set_xticks([3200, 3100, 3000, 2900, 2800, 2700, 2600])
ax.annotate("Bed elevation (m)", (0.5, 1), xycoords="axes fraction", xytext=(0, 13), textcoords="offset points",
            ha="center", va="bottom", fontsize=LAB, color=INK)
for xy, lab, ha in ((dA[0] + 10, "A", "left"), (dA[-1] - 10, "A′", "right")):
    ax.text(xy, 73, lab, fontsize=LAB, fontweight=600, ha=ha, va="top", color=INK)
panel_letter(fig, ax, "b", W)
print(f"figure {W:.0f} x {H:.0f} mm; a {w_a:.0f} x {h_a:.0f} mm, b {h_b:.0f} mm")
save(fig, "igm_flow")
