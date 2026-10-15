"""The end of every summer with an image, 1938-2026: one tile per year on the record window E 2678800-2681800,
N 1140200-1142200 (the glacier of 1931 with its margins), the best source of the year.
Selection: tex/build/record_files.csv, written by code/satellite_record_select.py from the archive catalogue (the
  rules are in its docstring: the window 10 August to 10 October, the cloud, shadow and no-data limits, Landsat tier 1
  only, the source order aerial orthophoto, Sentinel-2, Landsat; the latest scene near the year's snow minimum).
  PlanetScope (3 m, the scenes of data/planet ordered for code/satellite_summers.py) replaces Sentinel-2 here where
  a clipped scene exists. Years without a usable image: 1939-1945, 1947-1960, 1962-1980 (the 1976 flight has cloud
  over the glacier), 1982-1983, 1995, 1998, 2005, 2008.
Inputs: data/satellite_record/<year>_<date>_<platform>.tif (code/satellite_record_cut.py on s02: red, green, blue on
  the window; float32 reflectance for the satellites, Landsat MSS in false colour near infrared, red, green; uint8
  display values at 2 m for the aerial orthophotos), data/planet/*/*/PSScene/<id>_3B_AnalyticMS_SR_clip.tif.
Tone: satellites with the curve of code/satellite_year.py (reflectance / 0.9 to 0.6, then the shoulder, sRGB, chroma
  x 1.1), one curve for all; aerial orthophotos are display values, each stretched between its 1st and 99th
  percentile over the window (the flights differ in film and exposure). Pixels: average from the native grid to the
  tile grid (400 ppi at 1:83 333, 5.3 m) for the aerial images, bilinear for the satellites.
Output: tex/figures/satellite_record_1.pdf (32 tiles) and satellite_record_2.pdf (the rest and
  the key), 4 columns of 36 x 24 mm tiles with 2 mm gaps, the date stamped top left and the source bottom right, the 27 September
  2026 outline in red on every tile; tex/preview/satellite_record_{1,2}.png.
usage, from basodino/: uv run --with rasterio --with matplotlib --with scikit-image python code/satellite_record.py
"""
import csv
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

REC, PREVIEW = ROOT / "data/satellite_record", ROOT / "tex/preview"
PLANET = {2019: "20190921_100219_103d", 2020: "20200917_094522_46_106b", 2022: "20220920_101209_76_2426",
          2023: "20230911_093005_28_24ab", 2025: "20250918_104912_00_253c", 2026: "20260913_105647_59_254a"}
SATURATED = 0.8                                          # reflectance: near-saturated Landsat pixels drawn grey
STYLE = json.load(open(ROOT / "maps/style.json"))["imagery"]["sentinel2"]
WHITE, CHROMA, KNEE = STYLE["tone_white"], STYLE["tone_chroma"], 0.6
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")

W, SCALE, PPI = 150.0, 83333, 400                        # mm, scale denominator, pixels per inch
COLS, TILE_W, TILE_H, GAP = 4, 36.0, 24.0, 2.0           # mm
PER_PAGE = 32
E0, N0, E1, N1 = 2678800, 1140200, 2681800, 1142200      # window, m
NX, NY = round(TILE_W / 25.4 * PPI), round(TILE_H / 25.4 * PPI)
GRID = grid_from_bounds(E0, N0, E1, N1, NX, NY)
LINE, CASING, FRAME = 0.4, 0.3, 0.4                      # pt: outline, white casing beyond it on each side, tile frame
STAMP = dict(text=INK, halo="white", width=0.8, alpha=0.75)


def read_window(path, how):
    with rasterio.open(path) as src:
        arr, t, crs, dtype = src.read().astype(np.float32), src.transform, src.crs, src.dtypes[0]
        tags = src.tags()
    out = np.full((3, NY, NX), np.nan, np.float32)
    for k in range(3):
        reproject(arr[k], out[k], src_transform=t, src_crs=crs, src_nodata=np.nan, dst_transform=GRID,
                  dst_crs="EPSG:2056", dst_nodata=np.nan, resampling=how)
    return out.transpose(1, 2, 0), dtype, tags


def fill(r):
    """Saturated or missing bands (NaN): a pixel with some bands left takes their brightest value in the missing
    ones (as the archive quicklooks); a pixel with no band left is drawn white. A pixel with a band at or above
    SATURATED is near saturation (Landsat TM over snow: the detectors compress before they clip, so blue drops and
    the snow turns yellow in stripes): it is drawn grey at its brightest band."""
    bad = np.isnan(r)
    best = np.nanmax(np.where(bad, -np.inf, r), axis=2)
    best = np.where(np.isfinite(best), best, 1.0)
    out = np.where(bad, best[..., None], r)
    near = (np.nan_to_num(r) >= SATURATED).any(axis=2)
    return np.where(near[..., None], best[..., None], out)


def tone(r):
    """Display colours (uint8) of reflectance: the curve of satellite_year.py, sRGB encoding, chroma x CHROMA."""
    r = np.maximum(fill(r), 0.0)
    t = np.where(r <= KNEE, r / WHITE,
                 KNEE / WHITE + (1 - KNEE / WHITE) * (1 - np.exp(-(r - KNEE) / (WHITE - KNEE))))
    s = np.where(t <= 0.0031308, 12.92 * t, 1.055 * np.power(t, 1 / 2.4) - 0.055)
    lab = rgb2lab(s)
    lab[..., 1:] *= CHROMA
    return np.clip(np.rint(np.clip(lab2rgb(lab), 0, 1) * 255), 0, 255).astype(np.uint8)


def width_mm(s, size, weight=400):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath
    w, _, _ = TextToPath().get_text_width_height_descent(s, FontProperties(family="Source Sans 3", size=size,
                                                                            weight=weight), ismath=False)
    return w / 72 * 25.4


def rings():
    gj = json.load(open(ROOT / "data/outline_2026/outline_2026-09-27.geojson"))
    (feature,) = gj["features"]
    g = feature["geometry"]
    parts = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    return [np.asarray(r) for part in parts for r in part]


# --- data
tiles = []
for r in csv.DictReader(open(ROOT / "tex/build/record_files.csv")):
    year, date, platform = int(r["year"]), r["date"], r["platform"]
    d = date.split("-")
    planet = glob(str(ROOT / f"data/planet/*/*/PSScene/{PLANET.get(year, 'none')}_3B_AnalyticMS_SR_clip.tif"))
    if planet and platform != "aircraft":
        with rasterio.open(planet[0]) as src:
            arr, t, crs = src.read([3, 2, 1]).astype(np.float32) * 1e-4, src.transform, src.crs
        out = np.full((3, NY, NX), np.nan, np.float32)
        for k in range(3):
            reproject(arr[k], out[k], src_transform=t, src_crs=crs, dst_transform=GRID, dst_crs="EPSG:2056",
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
        pid = PLANET[year]
        date, platform, src_name = f"{pid[:4]}-{pid[4:6]}-{pid[6:8]}", "PlanetScope", "PlanetScope"
        rgb = tone(out.transpose(1, 2, 0))
    else:
        (path,) = glob(str(REC / f"{year}_{date}_*.tif"))
        img, dtype, tags = read_window(path, Resampling.average if platform == "aircraft" else Resampling.bilinear)
        if platform == "aircraft":
            lo, hi = np.nanpercentile(img, [1, 99])
            rgb = np.clip(np.rint((img - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
            src_name = "Aerial, swisstopo"
        else:
            if np.isnan(img).all(axis=2).mean() > 0.5 or np.nanstd(img, axis=(0, 1)).max() < 1e-3:
                print(f"{year} {date}: no data over the window, left out")
                continue
            rgb = tone(img)
            if "Landsat" in platform:
                src_name = platform + (", false colour" if "false colour" in tags.get("encoding", "") else "")
            else:
                src_name = "Sentinel-2"
    d = date.split("-")
    tiles.append((f"{int(d[2])} {MONTHS[int(d[1]) - 1]} {d[0]}", src_name, rgb))
    print(f"{year} {date} {src_name}")
outline = rings()

# --- figures
plt = figure_style()
case = [patheffects.Stroke(linewidth=LINE + 2 * CASING, foreground="white", alpha=0.6), patheffects.Normal()]
halo = [patheffects.Stroke(linewidth=2 * STAMP["width"], foreground=STAMP["halo"], alpha=STAMP["alpha"],
                           joinstyle="round")]
cap = 0.66 * (LAB - 1) / 72 * 25.4
inset = FRAME / 2 / 72 * 25.4
pages = [tiles[i:i + PER_PAGE] for i in range(0, len(tiles), PER_PAGE)]
for p, page in enumerate(pages, 1):
    slots = len(page) + (1 if p == len(pages) else 0)
    rows = -(-slots // COLS)
    H = rows * TILE_H + (rows - 1) * GAP
    fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white")
    for i, (stamp, src_name, rgb) in enumerate(page):
        row, col = divmod(i, COLS)
        x, top = col * (TILE_W + GAP), row * (TILE_H + GAP)
        ax = fig.add_axes((x / W, 1 - (top + TILE_H) / H, TILE_W / W, TILE_H / H))
        ax.imshow(rgb, extent=(E0, E1, N0, N1), interpolation="none")
        for xy in outline:
            ax.plot(xy[:, 0], xy[:, 1], color=ACCENT, lw=LINE, solid_joinstyle="round", path_effects=case)
        ax.set_xlim(E0, E1), ax.set_ylim(N0, N1), ax.set_axis_off()
        ax.add_patch(plt.Rectangle((inset / TILE_W, inset / TILE_H), 1 - 2 * inset / TILE_W, 1 - 2 * inset / TILE_H,
                                   transform=ax.transAxes, fill=False, lw=FRAME, edgecolor=GREY, zorder=5))
        kw = dict(transform=ax.transAxes, va="baseline", fontsize=LAB - 1)
        for text, xx, y, ha, weight in ((stamp, 1.0 / TILE_W, 1 - (1.0 + cap) / TILE_H, "left", 600),
                                        (src_name, 1 - 1.0 / TILE_W, 1.1 / TILE_H, "right", 400)):
            ax.text(xx, y, text, color=STAMP["halo"], path_effects=halo, ha=ha, fontweight=weight, **kw)
            ax.text(xx, y, text, color=STAMP["text"], ha=ha, fontweight=weight, **kw)
    if p == len(pages):                                   # key in the slot after the last tile
        row, col = divmod(len(page), COLS)
        x, top = col * (TILE_W + GAP), row * (TILE_H + GAP)
        fx = fig.add_axes((x / W, 1 - (top + TILE_H) / H, TILE_W / W, TILE_H / H))
        fx.set_xlim(0, TILE_W), fx.set_ylim(0, TILE_H), fx.set_axis_off()
        base = TILE_H - 5.0
        fx.plot([1.0, 6.0], [base + 0.6] * 2, color=ACCENT, lw=LINE, solid_capstyle="butt")
        fx.text(7.0, base, "Glacier, 27 September 2026", ha="left", va="baseline", fontsize=LAB - 1, color=INK)
        k = SCALE / 1e6                                   # km per mm
        x0, yb, hb = 1.0, base - 6.5, 1.2
        for j in range(4):
            fx.add_patch(plt.Rectangle((x0 + j * 0.25 / k, yb), 0.25 / k, hb,
                                       facecolor=INK if j % 2 == 0 else "white", edgecolor=INK, lw=0.4))
        for km in (0, 1):
            fx.text(x0 + km / k, yb + hb + 0.6, str(km), ha="center", va="baseline", fontsize=NUM - 1, color=INK)
        fx.text(x0 + 1 / k + width_mm("1", NUM - 1) / 2, yb + hb + 0.6, " km", ha="left", va="baseline",
                fontsize=NUM - 1, color=INK)
        xa = x0 + 1 / k + 9.0
        fx.plot([xa, xa], [yb, yb + 3.2], color=INK, lw=0.5, solid_capstyle="butt")
        fx.add_patch(plt.Polygon([(xa - 0.6, yb + 3.0), (xa, yb + 4.2), (xa + 0.6, yb + 3.0)], closed=True,
                                 facecolor=INK, lw=0))
        fx.text(xa + 1.1, yb + 0.9, "North", ha="left", va="baseline", fontsize=LAB - 1, fontweight=600, color=INK)
    save(fig, f"satellite_record_{p}", "satellite_record.py")
    pdf = FIG / f"satellite_record_{p}.pdf"
    PREVIEW.mkdir(exist_ok=True)
    subprocess.run(["pdftoppm", "-r", "150", "-png", "-singlefile", str(pdf), str(PREVIEW / f"satellite_record_{p}")],
                   check=True)
    print(f"{pdf}: {W:.0f} x {H:.0f} mm, {len(page)} tiles, {pdf.stat().st_size / 1e6:.1f} MB")
