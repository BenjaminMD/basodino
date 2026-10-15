"""The end of every summer 2016-2026 on Ghiacciaio del Basòdino in true colour: PlanetScope (3 m) for the years with a
clear scene in the Planet archive (2018-2023, 2025, 2026), Sentinel-2 (10 m) for 2016, 2017 and 2024. The tile window,
stretch, outline and key are those of code/satellite_year_planet.py, the tiles 2 mm shorter so that eleven of them and
the key fit on one page.

Scenes (SCENES below): PlanetScope ids of data/planet/search.csv, one per year, the clearest scene from 10 September
on, ordered clipped to the 10 km2 box by code/planet_order.py (2024 has no PlanetScope scene covering the box with 80
per cent clear pixels); Sentinel-2 tags of the archive data/satellite_archive/sentinel2: the scene of
code/satellite_record.py for that year (2017, 2024) or of the time-lapse code/timelapse.py (2016).
PlanetScope: bundle analytic_sr_udm2, surface reflectance = DN x 1e-4 (bands blue, green, red, near infrared).
Sentinel-2 archive: scenes/<year>/<tag>_10m.tif, B02 B03 B04 B08, reflectance = value / 10000, 0 = no data, LV95 grid.
Output: tex/figures/satellite_summers.pdf, 150 x 214 mm, 2 columns x 6 rows of 74 x 34 mm tiles (2960 x 1360 m at
  1:40 000, E 2678700-2681660, N 1140283-1141643), the last slot holds the key; tex/preview/satellite_summers.png.
usage, from basodino/: uv run --with rasterio --with matplotlib --with scikit-image python code/satellite_summers.py
"""
import json
import subprocess
from glob import glob

import numpy as np
import rasterio
from matplotlib import patheffects
from rasterio.transform import from_bounds as grid_from_bounds
from rasterio.warp import Resampling, reproject
from skimage.color import lab2rgb, rgb2lab

from common import ACCENT, FIG, GREY, INK, LAB, MM, NUM, ROOT, figure_style, save

S2, PREVIEW = ROOT / "data/satellite_archive/sentinel2/scenes", ROOT / "tex/preview"
SCENES = [("29 September 2016", "Sentinel-2", "2016-09-29_S2A_R065"),
          ("7 September 2017", "Sentinel-2", "2017-09-07_S2A_R108"),
          ("28 September 2018", "PlanetScope", "20180928_095118_1014"),
          ("21 September 2019", "PlanetScope", "20190921_100219_103d"),
          ("17 September 2020", "PlanetScope", "20200917_094522_46_106b"),
          ("17 September 2021", "PlanetScope", "20210917_092946_61_2465"),
          ("20 September 2022", "PlanetScope", "20220920_101209_76_2426"),
          ("11 September 2023", "PlanetScope", "20230911_093005_28_24ab"),
          ("10 September 2024", "Sentinel-2", "2024-09-10_S2A_R108"),
          ("18 September 2025", "PlanetScope", "20250918_104912_00_253c"),
          ("13 September 2026", "PlanetScope", "20260913_105647_59_254a")]
STYLE = json.load(open(ROOT / "maps/style.json"))["imagery"]["sentinel2"]
WHITE, CHROMA, KNEE = STYLE["tone_white"], STYLE["tone_chroma"], 0.6

W, SCALE, PPI = 150.0, 40000, 400                        # mm, scale denominator, pixels per inch
TILE_W, TILE_H, GAP = 74.0, 34.0, 2.0                    # mm
ROWS = (len(SCENES) + 2) // 2
H = ROWS * TILE_H + (ROWS - 1) * GAP
E0, N0 = 2678700, 1140283                                # tile window, m
E1, N1 = E0 + TILE_W * SCALE / 1000, N0 + TILE_H * SCALE / 1000
NX, NY = round(TILE_W / 25.4 * PPI), round(TILE_H / 25.4 * PPI)
GRID = grid_from_bounds(E0, N0, E1, N1, NX, NY)
LINE, CASING, FRAME = 0.5, 0.3, 0.4                      # pt: outline, white casing beyond it on each side, tile frame
STAMP = dict(text=INK, halo="white", width=0.8, alpha=0.75)


def reflectance(sensor, ident):
    """(bands (3, rows, cols) red, green, blue as reflectance, transform, crs) of one scene on its native grid."""
    if sensor == "PlanetScope":
        (path,) = glob(str(ROOT / f"data/planet/*/*/PSScene/{ident}_3B_AnalyticMS_SR_clip.tif"))
        with rasterio.open(path) as src:
            dn, t, crs = src.read([3, 2, 1]).astype(np.float32), src.transform, src.crs
        return dn * 1e-4, t, crs
    with rasterio.open(S2 / ident[:4] / f"{ident}_10m.tif") as src:
        dn, t, crs = src.read([3, 2, 1]).astype(np.float32), src.transform, src.crs
    return np.where(dn == 0, 0.0, dn * 1e-4).astype(np.float32), t, crs


def tile(sensor, ident):
    """Reflectance (rows, cols, 3) of one scene on the tile grid."""
    refl, t, crs = reflectance(sensor, ident)
    out = np.full((3, NY, NX), np.nan, np.float32)
    for k in range(3):
        reproject(refl[k], out[k], src_transform=t, src_crs=crs, dst_transform=GRID, dst_crs="EPSG:2056",
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
    assert np.isfinite(out).all(), f"{ident}: tile not covered"
    return out.transpose(1, 2, 0)


def tone(r):
    """Display colours (uint8) of reflectance: the curve of satellite_year.py, sRGB encoding, chroma x CHROMA."""
    r = np.maximum(r, 0.0)
    t = np.where(r <= KNEE, r / WHITE,
                 KNEE / WHITE + (1 - KNEE / WHITE) * (1 - np.exp(-(r - KNEE) / (WHITE - KNEE))))
    s = np.where(t <= 0.0031308, 12.92 * t, 1.055 * np.power(t, 1 / 2.4) - 0.055)
    lab = rgb2lab(s)
    lab[..., 1:] *= CHROMA
    return np.clip(np.rint(np.clip(lab2rgb(lab), 0, 1) * 255), 0, 255).astype(np.uint8)


def width_mm(s, size, weight=400):
    """Advance width of a string in Source Sans 3, mm."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath
    w, _, _ = TextToPath().get_text_width_height_descent(s, FontProperties(family="Source Sans 3", size=size,
                                                                            weight=weight), ismath=False)
    return w / 72 * 25.4


def rings():
    """Rings (n, 2) of the 2026 outline, every part with its holes, Swiss coordinates."""
    gj = json.load(open(ROOT / "data/outline_2026/outline_2026-09-27.geojson"))
    (feature,) = gj["features"]
    g = feature["geometry"]
    parts = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    return [np.asarray(r) for part in parts for r in part]


# --- data
tiles = []
for date, sensor, ident in SCENES:
    refl = tile(sensor, ident)
    peak = refl.max(axis=2)
    print(f"{date:18s} {sensor:12s} {ident}: tile above {KNEE}: {np.mean(peak > KNEE):5.1%}, 99.9th percentile "
          f"{np.percentile(peak, 99.9):.2f}")
    tiles.append((date, sensor, tone(refl)))
outline = rings()

# --- figure
plt = figure_style()
fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white")
case = [patheffects.Stroke(linewidth=LINE + 2 * CASING, foreground="white", alpha=0.6), patheffects.Normal()]
cap = 0.66 * LAB / 72 * 25.4                              # cap height of the 7 pt stamp, mm
inset = FRAME / 2 / 72 * 25.4                             # centre line of the frame inside the tile edge, mm
halo = [patheffects.Stroke(linewidth=2 * STAMP["width"], foreground=STAMP["halo"], alpha=STAMP["alpha"],
                           joinstyle="round")]
for i, (date, sensor, rgb) in enumerate(tiles):
    row, col = divmod(i, 2)
    x, top = col * (TILE_W + GAP), row * (TILE_H + GAP)
    ax = fig.add_axes((x / W, 1 - (top + TILE_H) / H, TILE_W / W, TILE_H / H))
    ax.imshow(rgb, extent=(E0, E1, N0, N1), interpolation="none")
    for xy in outline:
        ax.plot(xy[:, 0], xy[:, 1], color=ACCENT, lw=LINE, solid_joinstyle="round", path_effects=case)
    ax.set_xlim(E0, E1), ax.set_ylim(N0, N1), ax.set_axis_off()
    ax.add_patch(plt.Rectangle((inset / TILE_W, inset / TILE_H), 1 - 2 * inset / TILE_W, 1 - 2 * inset / TILE_H,
                               transform=ax.transAxes, fill=False, lw=FRAME, edgecolor=GREY, zorder=5))
    kw = dict(transform=ax.transAxes, va="baseline", fontsize=LAB)
    for text, xx, ha, weight in ((date, 1.3 / TILE_W, "left", 600), (sensor, 1 - 1.3 / TILE_W, "right", 400)):
        y = 1 - (1.3 + cap) / TILE_H
        ax.text(xx, y, text, color=STAMP["halo"], path_effects=halo, ha=ha, fontweight=weight, **kw)
        ax.text(xx, y, text, color=STAMP["text"], ha=ha, fontweight=weight, **kw)

# key, north arrow and scale bar in the last slot (bottom right), on one baseline
row, col = divmod(len(tiles), 2)
x, top = col * (TILE_W + GAP), row * (TILE_H + GAP)
fx = fig.add_axes((x / W, 1 - (top + TILE_H) / H, TILE_W / W, TILE_H / H))
fx.set_xlim(0, TILE_W), fx.set_ylim(0, TILE_H), fx.set_axis_off()
base, hb = TILE_H - 6.0, 1.4                              # outline key baseline; scale bar height, mm
fx.plot([0.2, 7.0], [base + 0.6] * 2, color=ACCENT, lw=LINE, solid_capstyle="butt")
fx.text(8.5, base, "Glacier, 27 September 2026", ha="left", va="baseline", fontsize=LAB, color=INK)
k = SCALE / 1e6                                           # km per mm
unit = " km"
base, yb = base - 7.0, base - 9.1                         # second line: the scale bar and the north arrow
x0 = TILE_W - 1 / k - width_mm("1", NUM) / 2 - width_mm(unit, NUM)
for j in range(4):
    fx.add_patch(plt.Rectangle((x0 + j * 0.25 / k, yb), 0.25 / k, hb, facecolor=INK if j % 2 == 0 else "white",
                               edgecolor=INK, lw=0.4))
for km in (0, 1):
    fx.text(x0 + km / k, base, str(km), ha="center", va="baseline", fontsize=NUM, color=INK)
fx.text(x0 + 1 / k + width_mm("1", NUM) / 2, base, unit, ha="left", va="baseline", fontsize=NUM, color=INK)
xa = x0 - 6.0 - width_mm("North", LAB, 600)
fx.plot([xa, xa], [yb, base + cap - 1.2], color=INK, lw=0.5, solid_capstyle="butt")
fx.add_patch(plt.Polygon([(xa - 0.6, base + cap - 1.4), (xa, base + cap), (xa + 0.6, base + cap - 1.4)],
                         closed=True, facecolor=INK, lw=0))
fx.text(xa + 1.1, base, "North", ha="left", va="baseline", fontsize=LAB, fontweight=600, color=INK)

save(fig, "satellite_summers", "satellite_summers.py")
pdf = FIG / "satellite_summers.pdf"
PREVIEW.mkdir(exist_ok=True)
subprocess.run(["pdftoppm", "-r", "150", "-png", "-singlefile", str(pdf), str(PREVIEW / "satellite_summers")],
               check=True)
print(f"{pdf}: {W:.0f} x {H:.0f} mm, {pdf.stat().st_size / 1e6:.1f} MB")
