"""The melt season of the hydrological year 2026 on Ghiacciaio del Basòdino: six Sentinel-2 true-colour tiles,
7 October 2025 to 22 September 2026.

The glacier is white from the first days of the hydrological year until early July. In the three July tiles, ten days
apart from the same satellite and orbit, bare ice spreads over its northern part until snow is left only along the
southern margin; on 22 September the glacier is bare ice.

Scenes (SCENES: tags of data/satellite_archive/sentinel2/catalog.csv, which also gives the Earth Search item; Level-2A,
collection sentinel-2-c1-l2a; all relative orbit 108, 10:38 UTC; clear over the tile on the quicklooks):
  7 October 2025     Sentinel-2A   first clear orbit-108 scene of the hydrological year: snow on the whole glacier,
                                   blue where the ridge along its southern margin (crest about 3030 m, 100 m south
                                   of the margin; data/dem_2m.tif) casts its shadow at a sun 37 deg above the horizon
  23 May 2026        Sentinel-2C   the only clear orbit-108 scene of May; the slopes below the glacier snow-free
  2 July 2026        Sentinel-2C   glacier white, first bare patches
  12 July 2026       Sentinel-2C   bare ice over the northern half
  22 July 2026       Sentinel-2C   snow only along the southern margin
  22 September 2026  Sentinel-2A   last clear orbit-108 scene before the end of the hydrological year
  No June or August scene of orbit 108 is free of cloud or cloud shadow over the glacier (quicklooks: haze over the
  western glacier on 14 June, cloud shadow on 22 and 24 June and on 11 and 13 August, the others cloudy). Orbit 65 has
  a nearly clear 14 June (2 % of the glacier classified as cloud), left out only to keep one orbit, and a 13 August
  with cumulus south of the glacier and cloud shadow over its western end. 22 September rather than the outline scene
  of 27 September (orbit 65) keeps one orbit; code/outline_2026.csv puts its outline 3.8 m from the 27 September one.
  The scenes agree in position within 5 m, 0.1 mm on paper (phase correlation of gradient images against 22
  September; largest 4.3 m, 2 July), so none is shifted. Alternative winter tile: 19 March 2026 (Sentinel-2B, orbit
  108, cached), which on the glacier looks like 7 October.
Snow share (printed per scene): snowice_g2024 - bareice_g2024 of catalog.csv, the share of the 2024 glacier area
  (swissTLM3D, 1.380 km2) that is snow: clear pixels with snow index above 0.4 (green above 0.10, near infrared above
  0.11) and broadband albedo of at least 0.40 (src/archive.py). Snow in deep shadow falls below that albedo and counts
  as ice.

Cache: data/satellite_year/<tag>_truecolour_utm32n_10m.tif (not in git: *.tif). Bands B04, B03, B02 of each item as
  stored in its cloud-optimised files (UTM 32N, 10 m; reflectance = DN x 0.0001 - 0.1, DN 0 = no data), read once over
  the plate-3 square plus 100 m (E 2678500-2682200, N 1139500-1143200) so that the tile window can move inside it
  without a new fetch; the item's metadata are in the file tags. Delete a file to fetch it again (about 7 s).

Pixels: each band in one Lanczos-3 step from its native grid to the tile grid (Swiss coordinates, 400 ppi at
  1:40 000, 2.54 m), clamped to the minimum and maximum of the 3 x 3 source pixels around each output pixel so that
  snow edges do not ring (as plate 4). DN 0 occurs as single-band dropouts in deep cast shadow east of the glacier (a
  few per scene, neighbours near reflectance 0) and is drawn as reflectance 0.

Stretch, one fixed curve for all six tiles, nothing adjusted per tile: reflectance r -> t = r / 0.9 (the report tone,
  maps/style.json imagery.sentinel2) up to r = 0.6, above it the shoulder t = 2/3 + 1/3 (1 - exp(-(r - 0.6) / 0.3)),
  continuous in value and slope, which reaches white only asymptotically; then the sRGB encoding and chroma x 1.1 in
  CIELAB (style.json). Rock, ice and firn (r < 0.6) therefore look as on plates 4 and 4b, and snow keeps its relief:
  sunlit snow reaches r = 1.5 (99.9th percentile of 7 October), and the plain report tone would print 27 % of the
  7 October tile and 46 % of the 23 May tile as flat paper white; with the shoulder no pixel reaches it. The July snow
  (r 0.6-0.9) changes little.

Outline: data/outline_2026/outline_2026-09-27.geojson (code/outline_2026.py), our glacier outline from Sentinel-2 of 27
  September 2026 (the glacier body with its holes and the detached blocks), the same in every tile: red 0.5 pt with a
  white casing, as in that script's figure. Rerun this script whenever the outline changes.

Output: tex/figures/satellite_year_2026.pdf, 150 x 120 mm, placed at 100 %: 2 columns x 3 rows of 74 x 36.7 mm tiles
  (2960 x 1468 m at 1:40 000, E 2678700-2681660, N 1140140-1141608, grid north up; the outline 6 mm from the top and
  bottom edges, 10 mm from the sides) with 2 mm gaps and a 0.4 pt grey frame inside each tile, so that no part of it is
  cut at the edge of the figure; under them the key, a grid-north arrow and a 0-1 km scale bar in 250 m segments
  (style.json draws 0-2 km at 1:40 000, for the 175 mm plate frames). Graphics only (tex/SIDECARS.md): date stamps in
  dark lettering on a white halo (style.json labels.halo.relief_plates), since most tile corners are snow. Imagery
  embedded losslessly at 400 ppi (about 3.9 MB): a 10 m pixel is 0.25 mm on paper, so 400 ppi oversamples it 3.6
  times; 600 ppi looked the same and took 6.8 MB.
  tex/preview/satellite_year_2026.png: the PDF rendered at 150 ppi (poppler pdftoppm).

usage, from basodino/: uv run --with rasterio --with matplotlib --with scikit-image python code/satellite_year.py
"""
import csv
import datetime
import json
import subprocess
import urllib.request

import numpy as np
import rasterio
from matplotlib import patheffects
from rasterio.transform import from_bounds as grid_from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from scipy.ndimage import maximum_filter, minimum_filter
from skimage.color import lab2rgb, rgb2lab

from common import ACCENT, FIG, GREY, INK, LAB, MM, NUM, ROOT, figure_style, save

CACHE, PREVIEW = ROOT / "data/satellite_year", ROOT / "tex/preview"
STAC = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a/items/"
SCENES = ["2025-10-07_S2A_R108", "2026-05-23_S2C_R108", "2026-07-02_S2C_R108", "2026-07-12_S2C_R108",
          "2026-07-22_S2C_R108", "2026-09-22_S2A_R108"]
CATALOG = {r["tag"]: r for r in csv.DictReader(open(ROOT / "data/satellite_archive/sentinel2/catalog.csv"))}
BOX = (2678500, 1139500, 2682200, 1143200)               # fetched: plate-3 square + 100 m, Swiss coordinates, m
STYLE = json.load(open(ROOT / "maps/style.json"))["imagery"]["sentinel2"]
WHITE, CHROMA, KNEE = STYLE["tone_white"], STYLE["tone_chroma"], 0.6
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")

W, SCALE, PPI = 150.0, 40000, 400                        # mm, scale denominator, pixels per inch
TILE_W, TILE_H, GAP, STRIP = 74.0, 36.7, 2.0, 5.9        # mm
H = 3 * TILE_H + 2 * GAP + STRIP
E0, N0 = 2678700, 1140140                                # tile window, m
E1, N1 = E0 + TILE_W * SCALE / 1000, N0 + TILE_H * SCALE / 1000
NX, NY = round(TILE_W / 25.4 * PPI), round(TILE_H / 25.4 * PPI)
GRID = grid_from_bounds(E0, N0, E1, N1, NX, NY)
LINE, CASING, FRAME = 0.5, 0.3, 0.4                      # pt: outline, white casing beyond it on each side, tile frame
STAMP = dict(text=INK, halo="white", width=0.8, alpha=0.75)   # style.json labels.halo.relief_plates


def fetch(tag):
    """Bands 4, 3, 2 of one Level-2A item on their native UTM 32N 10 m grid over BOX, as stored (DN), with the item's
    metadata as tags. Run once per scene; the file is the cache."""
    item_id = CATALOG[tag]["id"]
    with urllib.request.urlopen(STAC + item_id, timeout=60) as r:
        item = json.load(r)
    p, bands = item["properties"], []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                      GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="5"):
        for name in ("red", "green", "blue"):
            asset = item["assets"][name]
            rb = asset["raster:bands"][0]
            assert (rb["scale"], rb["offset"], rb["nodata"]) == (0.0001, -0.1, 0), f"{item_id} {name}: {rb}"
            with rasterio.open(asset["href"]) as src:
                win = from_bounds(*transform_bounds("EPSG:2056", src.crs, *BOX), transform=src.transform)
                win = win.round_offsets().round_lengths()
                bands.append(src.read(1, window=win))
                profile = dict(driver="GTiff", dtype="uint16", count=3, width=win.width, height=win.height,
                               crs=src.crs, transform=src.window_transform(win), nodata=0, compress="deflate",
                               predictor=2)
    with rasterio.open(CACHE / f"{tag}_truecolour_utm32n_10m.tif", "w", **profile) as dst:
        dst.write(np.stack(bands))
        dst.update_tags(item=item_id, source=STAC + item_id, product=p["s2:product_uri"], datetime=p["datetime"],
                        platform=p["platform"], sun_elevation=p["view:sun_elevation"],
                        sun_azimuth=p["view:sun_azimuth"], encoding="reflectance = DN x 0.0001 - 0.1; DN 0 = no data")
        for i, d in enumerate(("B04 red", "B03 green", "B02 blue"), 1):
            dst.set_band_description(i, d)


def tile(tag):
    """Reflectance (rows, cols, 3) of one scene on the tile grid."""
    with rasterio.open(CACHE / f"{tag}_truecolour_utm32n_10m.tif") as src:
        dn, t, crs = src.read().astype(np.float32), src.transform, src.crs
    if (dn == 0).mean() > 1e-4:
        raise ValueError(f"{tag}: {int((dn == 0).sum())} values without data")
    refl = np.where(dn == 0, 0.0, dn * 1e-4 - 0.1).astype(np.float32)

    def warp(a, how):
        out = np.full((NY, NX), np.nan, np.float32)
        reproject(a, out, src_transform=t, src_crs=crs, src_nodata=np.nan, dst_transform=GRID, dst_crs="EPSG:2056",
                  dst_nodata=np.nan, resampling=how)
        return out

    rgb = np.dstack([np.clip(warp(b, Resampling.lanczos), warp(minimum_filter(b, 3), Resampling.nearest),
                             warp(maximum_filter(b, 3), Resampling.nearest)) for b in refl])
    assert np.isfinite(rgb).all(), f"{tag}: tile not covered"
    return rgb


def tone(r):
    """Display colours (uint8) of reflectance: the curve of the docstring, sRGB encoding, chroma x CHROMA."""
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
    """Rings (n, 2) of the 2026 outline, every part (glacier body and detached blocks) with its holes, Swiss
    coordinates."""
    gj = json.load(open(ROOT / "data/outline_2026/outline_2026-09-27.geojson"))
    assert gj["crs"]["properties"]["name"].endswith("EPSG::2056")
    (feature,) = gj["features"]
    g = feature["geometry"]
    assert g["type"] in ("Polygon", "MultiPolygon")
    parts = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    return [np.asarray(r) for part in parts for r in part]


# --- data
CACHE.mkdir(exist_ok=True)
for tag in SCENES:
    if not (CACHE / f"{tag}_truecolour_utm32n_10m.tif").is_file():
        fetch(tag)
        print("fetched", tag)
tiles = []
for tag in SCENES:
    refl, c = tile(tag), CATALOG[tag]
    d = datetime.datetime.fromisoformat(c["datetime"].replace("Z", "+00:00"))
    snow = float(c["snowice_g2024"]) - float(c["bareice_g2024"])
    peak = refl.max(axis=2)
    print(f"{tag}: sun {float(c['sun_elevation']):4.1f} deg, snow on {snow:4.0%} of the 2024 glacier area; tile "
          f"above {KNEE}: {np.mean(peak > KNEE):5.1%}, above {WHITE}: {np.mean(peak > WHITE):5.1%}, 99.9th "
          f"percentile {np.percentile(peak, 99.9):.2f}")
    tiles.append((f"{d.day} {MONTHS[d.month - 1]} {d.year}", tone(refl)))
outline = rings()

# --- figure
plt = figure_style()
fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white")
case = [patheffects.Stroke(linewidth=LINE + 2 * CASING, foreground="white", alpha=0.6), patheffects.Normal()]
cap = 0.66 * LAB / 72 * 25.4                              # cap height of the 7 pt stamp, mm
inset = FRAME / 2 / 72 * 25.4                             # centre line of the frame inside the tile edge, mm
for i, (stamp, rgb) in enumerate(tiles):
    row, col = divmod(i, 2)
    x, top = col * (TILE_W + GAP), row * (TILE_H + GAP)
    ax = fig.add_axes((x / W, 1 - (top + TILE_H) / H, TILE_W / W, TILE_H / H))
    ax.imshow(rgb, extent=(E0, E1, N0, N1), interpolation="none")
    for xy in outline:
        ax.plot(xy[:, 0], xy[:, 1], color=ACCENT, lw=LINE, solid_joinstyle="round", path_effects=case)
    ax.set_xlim(E0, E1), ax.set_ylim(N0, N1), ax.set_axis_off()
    ax.add_patch(plt.Rectangle((inset / TILE_W, inset / TILE_H), 1 - 2 * inset / TILE_W, 1 - 2 * inset / TILE_H,
                               transform=ax.transAxes, fill=False, lw=FRAME, edgecolor=GREY, zorder=5))
    kw = dict(transform=ax.transAxes, ha="left", va="baseline", fontsize=LAB, fontweight=600)
    xy = (1.3 / TILE_W, 1 - (1.3 + cap) / TILE_H)
    ax.text(*xy, stamp, color=STAMP["halo"], path_effects=[patheffects.Stroke(
        linewidth=2 * STAMP["width"], foreground=STAMP["halo"], alpha=STAMP["alpha"], joinstyle="round")], **kw)
    ax.text(*xy, stamp, color=STAMP["text"], **kw)              # live text over its halo

# key, north arrow and scale bar under the tiles, in mm from the bottom-left corner of the figure; all lettering on
# one baseline, the scale bar and the foot of the arrow below it
fx = fig.add_axes((0, 0, 1, STRIP / H))
fx.set_xlim(0, W), fx.set_ylim(0, STRIP), fx.set_axis_off()
base, yb, hb = 2.25, 0.15, 1.4                            # lettering baseline, scale bar bottom and height, mm
fx.plot([0.2, 7.0], [base + 0.6] * 2, color=ACCENT, lw=LINE, solid_capstyle="butt")      # x-height centre
fx.text(8.5, base, "Glacier, 27 September 2026", ha="left", va="baseline", fontsize=LAB, color=INK)
k = SCALE / 1e6                                           # km per mm
unit = " km"
x0 = W - 1 / k - width_mm("1", NUM) / 2 - width_mm(unit, NUM)    # bar start: the unit ends at the right edge
for j in range(4):                                        # 0-1 km in 250 m segments
    fx.add_patch(plt.Rectangle((x0 + j * 0.25 / k, yb), 0.25 / k, hb, facecolor=INK if j % 2 == 0 else "white",
                               edgecolor=INK, lw=0.4))
for km in (0, 1):
    fx.text(x0 + km / k, base, str(km), ha="center", va="baseline", fontsize=NUM, color=INK)
fx.text(x0 + 1 / k + width_mm("1", NUM) / 2, base, unit, ha="left", va="baseline", fontsize=NUM, color=INK)
xa = x0 - 6.0 - width_mm("North", LAB, 600)              # grid-north arrow, then its word, left of the bar
fx.plot([xa, xa], [yb, base + cap - 1.2], color=INK, lw=0.5, solid_capstyle="butt")
fx.add_patch(plt.Polygon([(xa - 0.6, base + cap - 1.4), (xa, base + cap), (xa + 0.6, base + cap - 1.4)],
                         closed=True, facecolor=INK, lw=0))
fx.text(xa + 1.1, base, "North", ha="left", va="baseline", fontsize=LAB, fontweight=600, color=INK)

save(fig, "satellite_year_2026", "satellite_year.py")
pdf = FIG / "satellite_year_2026.pdf"
PREVIEW.mkdir(exist_ok=True)
subprocess.run(["pdftoppm", "-r", "150", "-png", "-singlefile", str(pdf), str(PREVIEW / "satellite_year_2026")],
               check=True)
print(f"{pdf}: {W:.0f} x {H:.0f} mm, {pdf.stat().st_size / 1e6:.1f} MB")
