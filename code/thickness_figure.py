"""Figure for Question 5.1: the published ice thickness of Ghiacciaio del Basòdino on a map, and two sections.

Inputs:
  data/thickness/C14-10_IceThickness.tif   ice thickness of the reference year 2012 (Grab and fifteen co-authors 2021),
                                           10 m cells, Swiss coordinates (EPSG:2056); provenance in its README.md
  data/course/mask_basodino_2021.asc       course glacier mask of 2021, drawn as the 2021 outline
  data/thickness/sections.csv              sections A and B, written by code/thickness.py
Output:
  tex/figures/thickness_sections_si.pdf    150 x 142 mm, drawn at print size, graphics only (tex/SIDECARS.md)
  tex/preview/thickness_sections_si.png    the PDF rendered at 150 ppi (poppler pdftoppm)
Drawing:
  a     the thickness at 1:25 000 (colour scale 0 to 25.3 m; the thickest cell has 25.2 m), the 2021 outline, the
        section lines from their first to their last sample with the end letters, a 500 m scale bar.
  b, c  the sections at 1:20 000 horizontally: A at true scale (1:20 000 vertically too), B with the vertical scale
        exaggerated three times (1:6667). Rock (grey) below the bed, and below the 2021 surface off the outline of
        the data; the ice of 2021 (ice tone) between the bed and the 2021 surface where the 2021 mask has ice and the
        surface lies above the bed; the 2021 surface over the 2021 mask and the 1994 surface over the 1994 mask, each
        one sample further to meet the ground. A red mark where the other section crosses; direct labels in c only.
usage:  uv run --with rasterio --with numpy --with matplotlib python code/thickness_figure.py
"""
import csv
import subprocess

import numpy as np
import rasterio

from common import (ACCENT, COURSE, FIG, GLACIER, GREY, ICE, INK, LAB, MM, NUM, PALE_BLUE, ROOT, figure_style,
                    panel_letter, read_esri_ascii, save)

DATA = ROOT / "data/thickness"
MAP_SCALE, SECTION_SCALE = 25000, 20000                     # map, sections horizontally
VE = {"A": 1, "B": 3}                                       # vertical exaggeration of each section
VIEW = (2678950, 1140300, 2681650, 1141600)                 # map extent: west, south, east, north (m)
YLIM = {"A": (2570, 3345), "B": (2890, 3040)}               # elevation range of each section (m)
YTICKS = {"A": [2600, 2800, 3000, 3200], "B": [2900, 2950, 3000]}
HMAX = 25.3                                                 # m, top of the colour scale, above the thickest cell
ROCK, NAVY = "#e0e0e0", "#0b2f52"
DASH = (0, (2.5, 1.2))

with rasterio.open(DATA / "C14-10_IceThickness.tif") as src:
    assert src.crs.to_epsg() == 2056
    h = src.read(1, masked=True).astype(float).filled(np.nan)
    bounds = src.bounds
assert np.nanmax(h) <= HMAX
g21, hd = read_esri_ascii(COURSE / "mask_basodino_2021.asc")
X, Y = np.meshgrid(hd["xllcorner"] + (np.arange(round(hd["ncols"])) + 0.5) * hd["cellsize"],
                   hd["yllcorner"] + (hd["nrows"] - np.arange(round(hd["nrows"])) - 0.5) * hd["cellsize"])
with open(DATA / "sections.csv") as f:
    rows = list(csv.DictReader(f))
sec = {name: {k: np.array([float(r[k]) if r[k] else np.nan for r in rows if r["section"] == name])
              for k in rows[0] if k != "section"} for name in VE}
ends = {name: np.array([[s["easting_m"][0], s["northing_m"][0]], [s["easting_m"][-1], s["northing_m"][-1]]])
        for name, s in sec.items()}

# where the two lines cross: distance along each
(pa, qa), (pb, qb) = ends["A"], ends["B"]
t, u = np.linalg.solve(np.c_[qa - pa, pb - qb], pb - pa)
cross = {"A": ("B", t * np.hypot(*(qa - pa))), "B": ("A", u * np.hypot(*(qb - pb)))}

# --- layout, mm from the top-left corner
W, LEFT = 150.0, 11.0
MAP_W, MAP_H = (VIEW[2] - VIEW[0]) / MAP_SCALE * 1e3, (VIEW[3] - VIEW[1]) / MAP_SCALE * 1e3
SEC_W = {k: s["distance_m"][-1] / SECTION_SCALE * 1e3 for k, s in sec.items()}
SEC_H = {k: (YLIM[k][1] - YLIM[k][0]) * VE[k] / SECTION_SCALE * 1e3 for k in VE}
TOP_MAP = 1.5
TOP = {"A": TOP_MAP + MAP_H + 11.5}
TOP["B"] = TOP["A"] + SEC_H["A"] + 7.5
H = TOP["B"] + SEC_H["B"] + 8.5

plt = figure_style()
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HALO = [pe.withStroke(linewidth=1.6, foreground="white")]
fig = plt.figure(figsize=(W * MM, H * MM), dpi=600, facecolor="white")    # 600 ppi: the colour bar's raster fill


def axes(left, top, width, height):
    return fig.add_axes([left / W, 1 - (top + height) / H, width / W, height / H])


# --- a: map
ax = axes(LEFT, TOP_MAP, MAP_W, MAP_H)
ramp = LinearSegmentedColormap.from_list("thickness", [(0, ICE), (0.21, PALE_BLUE), (0.59, GLACIER), (1, NAVY)])
im = ax.imshow(h, extent=np.array([bounds.left, bounds.right, bounds.bottom, bounds.top]) / 1e3, cmap=ramp, vmin=0,
               vmax=HMAX, interpolation="none")
ax.contour(X / 1e3, Y / 1e3, g21, [0.5], colors=INK, linewidths=0.6).set_path_effects(HALO)
ax.text(2679.47, 1140.465, "Outline 2021", ha="left", va="top", fontsize=LAB, color=INK)   # below the south margin
for name, (p, q) in ends.items():
    ax.plot(*np.c_[p, q] / 1e3, color=ACCENT, lw=0.8, solid_capstyle="butt")
    step = (q - p) / np.hypot(*(q - p)) * 2.2 * MAP_SCALE / 1e3       # 2.2 mm beyond each end, in m
    for xy, letter in ((p - step, name), (q + step, name + "′")):
        ax.text(*xy / 1e3, letter, ha="center", va="center", fontsize=LAB, fontweight=600, color=ACCENT,
                path_effects=HALO)
ax.set_xlim(VIEW[0] / 1e3, VIEW[2] / 1e3), ax.set_ylim(VIEW[1] / 1e3, VIEW[3] / 1e3)
ax.spines[:].set_visible(True)
ax.set_xticks([2679, 2680, 2681]), ax.set_yticks([1140.5, 1141, 1141.5])
ax.set_xlabel("Easting (km)", labelpad=1.5)
ax.set_ylabel("Northing (km)", labelpad=1.0)                          # 2 would push its ink past the page edge
panel_letter(fig, ax, "a", W)
x0, y0, bar = (VIEW[2] - 200) / 1e3 - 0.5, (VIEW[1] + 90) / 1e3, 1.0 * MAP_SCALE / 1e6    # scale bar 0-500 m, km
for i, colour in enumerate(("white", INK)):
    ax.add_patch(plt.Rectangle((x0 + 0.25 * i, y0), 0.25, bar, facecolor=colour, edgecolor=INK, lw=0.4))
for dx, txt in ((0, "0"), (0.25, "250"), (0.5, "500 m")):
    ax.text(x0 + dx, y0 + bar + 0.025, txt, ha="left" if dx == 0.5 else "center", va="bottom", fontsize=NUM)
cax = axes(LEFT + MAP_W + 4.0, TOP_MAP + MAP_H - 30.0, 2.2, 30.0)
cb = fig.colorbar(im, cax=cax, ticks=[0, 5, 10, 15, 20, 25])
cb.outline.set_linewidth(0.4)
cb.set_label("Ice thickness 2012 (m)", labelpad=2)


# --- b, c: sections
def grow(m):
    """The mask and one sample on either side, so that a surface line meets the ground at the margin."""
    return m | np.r_[m[1:], False] | np.r_[False, m[:-1]]


for name, letter in (("A", "b"), ("B", "c")):
    s = sec[name]
    d, bed, z94, z21 = s["distance_m"], s["bed_m"], s["surface_1994_m"], s["surface_2021_m"]
    i94, i21 = s["ice_1994"] == 1, s["ice_2021"] == 1
    rock = np.where(np.isfinite(bed), bed, z21)
    ax = axes(LEFT, TOP[name], SEC_W[name], SEC_H[name])
    ax.fill_between(d, YLIM[name][0], rock, color=ROCK, lw=0)
    ax.fill_between(d, rock, z21, where=i21 & (z21 > rock), interpolate=True, color=ICE, lw=0)
    ax.plot(d, rock, color=INK, lw=0.7)
    ax.plot(d, np.where(grow(i21), z21, np.nan), color=GLACIER, lw=0.7)
    ax.plot(d, np.where(grow(i94), z94, np.nan), color=GREY, lw=0.5, ls=DASH)
    other, at = cross[name]
    k = np.argmin(np.abs(d - at))
    top = np.nanmax([z94[k], z21[k]])
    per_mm = SECTION_SCALE / 1e3 / VE[name]                            # m of elevation per mm
    ax.plot([at, at], [top + 1.0 * per_mm, top + 3.0 * per_mm], color=ACCENT, lw=0.5)
    ax.text(at, top + 3.4 * per_mm, other, ha="center", va="bottom", fontsize=LAB, fontweight=600, color=ACCENT)
    if name == "B":                                    # direct labels, once, where the lines lie furthest apart (m)
        ax.text(700, np.interp(700, d, z94) + 0.9 * per_mm, "Surface 1994", ha="left", va="baseline",
                fontsize=LAB, color=GREY)
        cap = 0.66 * LAB / 72 * 25.4 * per_mm                          # cap height of the words, in m of elevation
        span = np.abs(d - 1260) <= 4.2 * SECTION_SCALE / 1e3           # under the label, 8.4 mm wide (m along B)
        ax.text(1260, (np.nanmax(bed[span]) + np.nanmin(z21[span]) - cap) / 2, "Ice 2021", ha="center",
                va="baseline", fontsize=LAB, color=GLACIER)            # centred between the lines under it
        ax.text(560, np.interp(560, d, bed) - 0.9 * per_mm, "Bed", ha="left", va="top", fontsize=LAB, color=INK)
    for x, txt, ha in ((0.8 * SECTION_SCALE / 1e3, name, "left"), (d[-1], name + "′", "right")):  # 0.8 mm off the axis
        ax.text(x, YLIM[name][1], txt, ha=ha, va="top", fontsize=LAB, fontweight=600, color=ACCENT)
    ax.set_xlim(0, d[-1]), ax.set_ylim(*YLIM[name])
    ax.set_xticks(np.arange(0, d[-1] + 1, 500)), ax.set_yticks(YTICKS[name])
    if name == "B":
        ax.set_xlabel("Distance (m)", labelpad=1.5)
    ax.set_ylabel("Elevation (m)", labelpad=2)
    panel_letter(fig, ax, letter, W)

save(fig, "thickness_sections_si", "thickness_figure.py")
pdf = FIG / "thickness_sections_si.pdf"
subprocess.run(["pdftoppm", "-r", "150", "-png", "-singlefile", str(pdf),
                str(ROOT / "tex/preview/thickness_sections_si")], check=True)
print(f"{pdf.relative_to(ROOT)}: {W:.1f} x {H:.1f} mm; map 1:{MAP_SCALE}, sections 1:{SECTION_SCALE} horizontally, "
      f"vertical exaggeration A {VE['A']}, B {VE['B']}; crossing at A {cross['A'][1]:.0f} m, B {cross['B'][1]:.0f} m")
