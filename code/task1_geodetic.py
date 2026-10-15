"""Task 1: geodetic mass balance 1994-2021 from the course elevation models and glacier masks.

Inputs: data/course/dem_basodino_{1994,2021}.asc and mask_basodino_{1994,2021}.asc (10 m, LV95, both October);
data/glamos_basodino_volumechange.csv for the published geodetic pair closest to the course period.
Outputs:
  code/task1_geodetic.csv              every number of Question 1: quantity, value, unit, definition
  tex/figures/dh_1994_2021.pdf         elevation change on the area glaciated in 1994 or 2021
  tex/figures/outlines_1994_2021.pdf   the two masks over the 100 m contours of 2021
The figures hold graphics only (tex/SIDECARS.md) and are drawn at print size, 72.75 x 55 mm each, set side by side
in the 150 mm text width.

Method (main.tex, eq. geodetic): dh = z2021 - z1994 on every cell glaciated in 1994 or 2021, dV = sum(dh) x cell area,
B = rho_dV / rho_w x dV / (A_mean x dt), A_mean = (A1994 + A2021) / 2, rho_dV = 850 kg m-3, dt = 27 a (October to
October). Written next to it: the rate with A1994 or A2021 alone, and dh on the terrain outside both masks (a mean
and spread near zero mean the two models need no co-registration).

usage:  python3 code/task1_geodetic.py
"""
import csv

import numpy as np

from common import (ACCENT, COURSE, GREY, ICE, INK, LAB, MM, NUM, PALE_RED, RHO_DV, RHO_W, ROOT, figure_style,
                    panel_letter, read_esri_ascii, save, write_csv)

DT = 27.0                                            # a, October 1994 to October 2021
GLAMOS_PAIR = ("19910918", "20219999")              # aerial survey 1991 to swissALTI3D 2021, the closest GLAMOS pair
CSV_GLAMOS = ROOT / "data/glamos_basodino_volumechange.csv"


z94, hd = read_esri_ascii(COURSE / "dem_basodino_1994.asc")
z21, _ = read_esri_ascii(COURSE / "dem_basodino_2021.asc")
g94 = read_esri_ascii(COURSE / "mask_basodino_1994.asc")[0] == 1
g21 = read_esri_ascii(COURSE / "mask_basodino_2021.asc")[0] == 1
ice = g94 | g21
dh = z21 - z94
assert np.isfinite(dh[ice]).all(), "gaps in the elevation models on the glacier: fill them before summing"

cell = hd["cellsize"] ** 2                           # m2
a94, a21 = g94.sum() * cell, g21.sum() * cell
a_mean = (a94 + a21) / 2
dv = dh[ice].sum() * cell                            # m3
rate = {name: RHO_DV / RHO_W * dv / (a * DT) for name, a in (("mean", a_mean), ("1994", a94), ("2021", a21))}
stable = dh[~ice]

with open(CSV_GLAMOS, encoding="utf-8") as f:
    rows = [r for r in csv.reader(f) if len(r) > 13 and (r[3], r[4]) == GLAMOS_PAIR]
glamos = rows[0]                                     # columns: see the header of the file

numbers = [
    ("A_1994", a94 / 1e6, "km2", "cells of mask 1994 x 100 m2"),
    ("A_2021", a21 / 1e6, "km2", "cells of mask 2021 x 100 m2"),
    ("A_mean", a_mean / 1e6, "km2", "(A_1994 + A_2021) / 2"),
    ("cells_2021_outside_1994", int((g21 & ~g94).sum()), "", "glaciated in 2021, not in 1994"),
    ("dh_mean_glacier", dh[ice].mean(), "m", "mean dh on the cells glaciated in 1994 or 2021"),
    ("dh_mean_2021_area", dh[g21].mean(), "m", "mean dh on the 2021 mask"),
    ("dh_mean_lost_area", dh[g94 & ~g21].mean(), "m", "mean dh where the ice of 1994 had gone by 2021"),
    ("dh_min_glacier", dh[ice].min(), "m", "largest lowering"),
    ("dV", dv / 1e9, "km3", "sum of dh x cell area on the cells glaciated in 1994 or 2021"),
    ("dt", DT, "a", "October 1994 to October 2021"),
    ("rho_dV", RHO_DV, "kg m-3", "prescribed by the assignment"),
    ("B_rate", rate["mean"], "m w.e. a-1", "rho_dV / rho_w x dV / (A_mean x dt)"),
    ("B_rate_A1994", rate["1994"], "m w.e. a-1", "the same with A_1994"),
    ("B_rate_A2021", rate["2021"], "m w.e. a-1", "the same with A_2021"),
    ("B_total", rate["mean"] * DT, "m w.e.", "B_rate x dt"),
    ("dh_stable_mean", stable.mean(), "m", "mean dh outside both masks"),
    ("dh_stable_std", stable.std(), "m", "standard deviation of dh outside both masks"),
    ("z_2021_min", z21[g21].min(), "m a.s.l.", "lowest cell of the 2021 mask"),
    ("z_2021_max", z21[g21].max(), "m a.s.l.", "highest cell of the 2021 mask"),
    ("glamos_B_rate", float(glamos[11]), "m w.e. a-1", f"GLAMOS {glamos[1]} {glamos[3]} to {glamos[2]} {glamos[4]}"),
    ("glamos_sigma", float(glamos[12]), "m w.e.", "GLAMOS uncertainty of that pair"),
]
write_csv("task1_geodetic.csv", ("quantity", "value", "unit", "definition"), numbers)
for q, v, u, _ in numbers:
    print(f"{q:24s} {v:12.4f} {u}" if isinstance(v, float) else f"{q:24s} {v:12d}")

# --- figures
plt = figure_style()
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
W, H = 72.75, 55.0
x0, y0, cs = hd["xllcorner"], hd["yllcorner"], hd["cellsize"]
ny, nx = dh.shape
EXTENT = [x0 / 1e3, (x0 + nx * cs) / 1e3, y0 / 1e3, (y0 + ny * cs) / 1e3]          # km
X, Y = np.meshgrid((x0 + (np.arange(nx) + 0.5) * cs) / 1e3, (y0 + ny * cs - (np.arange(ny) + 0.5) * cs) / 1e3)
LEFT, TOP, MAP_W = 11.0, 1.5, W - 11.0 - 1.5         # mm
MAP_H = MAP_W * ny / nx
DASH = [(0, (2.5, 1.2))]


def map_axes(fig, letter):
    ax = fig.add_axes([LEFT / W, 1 - (TOP + MAP_H) / H, MAP_W / W, MAP_H / H])
    ax.spines[:].set_visible(True)
    ax.set_xlim(EXTENT[:2]), ax.set_ylim(EXTENT[2:])
    ax.set_xticks([2679, 2680, 2681]), ax.set_yticks([1140.5, 1141, 1141.5])
    ax.set_xlabel("Easting (km)", labelpad=1.5)
    ax.set_ylabel("Northing (km)", labelpad=2)
    panel_letter(fig, ax, letter, W)
    return ax


def outline(ax, mask, **kw):
    return ax.contour(X, Y, mask.astype(float), [0.5], linewidths=0.6, **kw)


# elevation change: dark red for the largest lowering, light at 0 m; ice-free cells left white
fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white")
ax = map_axes(fig, "a")
ramp = LinearSegmentedColormap.from_list("lowering", [(0, "#3d0607"), (0.38, ACCENT), (1, "#fbe9e7")])
im = ax.imshow(np.where(ice, dh, np.nan), extent=EXTENT, cmap=ramp, vmin=-60, vmax=0, interpolation="none")
outline(ax, g94, colors=GREY, linestyles=DASH)
outline(ax, g21, colors=INK).set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])
cax = fig.add_axes([LEFT / W, 1 - (TOP + MAP_H + 10.5) / H, MAP_W / W, 2.0 / H])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-60, -50, -40, -30, -20, -10, 0])
cb.outline.set_linewidth(0.4)
cb.set_label("Elevation change 1994–2021 (m)", labelpad=1.5)
save(fig, "dh_1994_2021", "task1_geodetic.py")

# masks: ice of 2021, ice lost since 1994, 100 m contours of the 2021 surface
fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white")
ax = map_axes(fig, "b")
rgb = np.ones(dh.shape + (3,))
rgb[g94 & ~g21] = to_rgb(PALE_RED)
rgb[g21] = to_rgb(ICE)
ax.imshow(rgb, extent=EXTENT, interpolation="none")
cn = ax.contour(X, Y, z21, np.arange(2400, 3300, 100), colors=GREY, linewidths=0.25)
ax.clabel(cn, manual=[(2680.49, 1141.57), (2680.45, 1141.02), (2679.57, 1140.94)], fontsize=NUM, colors=GREY,
          inline_spacing=1, fmt="%d")
outline(ax, g94, colors=GREY, linestyles=DASH)
outline(ax, g21, colors=INK)
keys = [Patch(facecolor=ICE, edgecolor=INK, lw=0.6, label="Ice 2021"),
        Patch(facecolor=PALE_RED, edgecolor="none", label="Ice lost 1994–2021"),
        Line2D([], [], color=GREY, lw=0.6, ls=DASH[0], label="Outline 1994"),
        Line2D([], [], color=GREY, lw=0.25, label="Contours, 100 m")]
fig.legend(handles=keys, loc="upper left", bbox_to_anchor=(LEFT / W, 1 - (TOP + MAP_H + 8.0) / H), ncols=2,
           frameon=False, fontsize=LAB, handlelength=1.6, handleheight=0.8, columnspacing=1.5, borderaxespad=0,
           borderpad=0, labelspacing=0.4)
save(fig, "outlines_1994_2021", "task1_geodetic.py")
