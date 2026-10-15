"""Time-lapse of Ghiacciaio del Basòdino (SGI C14-10), 2016-2026: one Sentinel-2 scene at the end of each summer,
true colour at 10 m, dissolving from year to year, with the end-of-summer ice of each year as a red line and the
lines of the earlier years left behind, faint; an end card with all lines.

Inputs (paths below basodino/; in the s02 mirror data/ is the project's derived/, see usage):
  data/satellite_archive/sentinel2/catalog.csv, scenes/<year>/   the archive (src/archive.py): every acquisition up
                                     to 29 September 2026; surface reflectance on the 10 m and 20 m LV95 grids, the
                                     processing-baseline offset already removed (DN / 10000 - 0.1 for baseline 04.00
                                     or later, else DN / 10000, as in code/last_ice.py), and the scene classification
  data/vectors.gpkg, layer sgi_2016  outline of C14-10 in the Swiss Glacier Inventory 2016 (imagery of 2018)
  code/last_ice.tif                  the last year each 10 m cell held ice, 2018-2026 (code/last_ice.py)
  data/outline_2026/outline_2026-09-27.gpkg   our outline of 27 September 2026 (code/outline_2026.py), for the core
  code/outline_2026.py, its code above '# --- the outline and its checks'   the terrain's cast shadow (its step 3,
                                     from data/dem_2m.tif), run here unchanged in a namespace of its own
  data/satellite_archive/sentinel1/scripts/fonts/SourceSans3-*.ttf   the report's typeface

Method:
  1. Frame: E 2677800-2682600, N 1139610-1142310, 480 x 270 cells of 10 m (16:9), 8 pixels a cell at 3840 x 2160.
  2. Scenes: the Level-2A scenes of SEASON (1 August to 30 September, the end of the hydrological year; code/
     last_ice.py reaches to 31 October, but the October scenes have the low sun and long shadows) of every year (the
     one Level-1C scene, top-of-atmosphere, has no scene classification). A scene is a candidate when
       data    every band B02, B03, B04 has data over the frame plus the registration margin, except single-band
               dropouts in deep shadow (one band missing, the others below DARK; drawn as 0, as in code/
               satellite_year.py), and the scene classification has no cell without data or defective (0, 1) over
               the frame
       cloud   cells that the scene classification calls cloud or cirrus (8, 9, 10) and that are bright (B02 > 0.3)
               on at most CLOUD_MAX of the frame. The brightness test is needed for baseline 02.12 (2015-2018),
               whose classification calls sunlit rock cloud (B02 0.13-0.25)
       shadow  cells of the region (the SGI 2016 outline plus 100 m) outside the terrain's cast shadow (code/
               outline_2026.py, step 3, for the sun of the scene, widened by 20 m) that the classification calls
               cloud shadow (3) or that are dark (B08 < 0.08) on at most SHADOW_MAX of the region, and cloud shadow
               (3) on at most SHADOW_MAX of the frame
       ice     cells of the core (our 27 September 2026 outline shrunk by 20 m, ice in every year) that are bright
               (B03 > 0.3) but not snow or ice (NDSI < 0.4) on at most ICE_MAX: cloud over the ice, bright in
               short-wave infrared where snow and ice absorb
       veil    median snow index NDSI = (B03 - B11) / (B03 + B11) over the core at least VEIL_MIN: thin cloud and
               haze lower it (code/last_ice.py takes 0.7; its hazy 20 September 2026 is 0.72)
       bright  cells bright in blue but not snow (B02 > 0.35 and NDSI < 0.5) on at most BRIGHT_MAX of the frame
       snow    snow (NDSI > 0.4, B03 > 0.10, B08 > 0.11, the archive's test) on at most SNOW_MAX of the frame outside
               the region: fresh snow, or the winter's snow not yet gone
       registration   the shift estimate of step 3 consistent: its four quarters scatter by at most SPREAD_MAX cell
     Of a year's candidates the one with the least snow outside the region (the end of the melt season); one of
     relative orbit ORBIT instead when its snow is within ORBIT_SLACK of that least, since the two orbits see the
     valley floor 0.5-0.9 cell apart, a parallax that one shift per scene cannot remove.
  3. Co-registration to the scene REF: phase correlation of the gradient magnitude of B04 over 5.12 km around the
     frame (Hann window), the peak refined to 1/UP cell by a matrix Fourier transform (Guizar-Sicairos, Thurman and
     Fienup 2008, Optics Letters 33, 156); the same estimate on the four quarters of that window gives its spread.
     code/checks/check_timelapse_registration.py recovers known shifts of REF with this estimate.
  4. Colour: B04, B03, B02 as reflectance r -> r / 0.9 up to r = 0.6, above it the shoulder 2/3 + 1/3 (1 - exp(-(r -
     0.6) / 0.3)) (code/satellite_year.py, without its chroma step), then the sRGB transfer curve; one fixed curve for
     every frame. Sunlit snow keeps its relief instead of clipping to flat white.
  5. Upscaling to 3840 x 2160: Lanczos (3 lobes) on the encoded values, the window shifted by the co-registration
     (sub-cell), then clamped per channel to the range of the 3 x 3 source neighbourhood (bilinear minimum and
     maximum filters), so the Lanczos overshoot at sharp edges leaves no halo.
  6. Lines: for year y (2018-2026) the cells of code/last_ice.tif with last year y or later, smoothed by a Gaussian of
     SMOOTH cell and contoured at 0.5 between cell centres; rings enclosing less than MIN_RING m2 dropped. Every year,
     2026 included, is drawn the same way. The scene of year y shows line y in red and lines 2018 to y - 1 thin white
     at half opacity; 2016 and 2017 have none. Drawn at twice the size and averaged down (anti-aliased).
  7. Text, Source Sans 3: the name of the glacier at the top left, a 1 km scale bar at the top right, the year large
     and the day small at the bottom left, the key and the credit at the bottom right. End card: the last scene
     darkened, every line from pale (2018) to red (2026), and its key.
  8. Timing: each scene HOLD s, dissolves of FADE s (smoothstep), to the end card END_FADE s, the end card END s, at
     FPS frames per second. H.264, CRF 16, yuv420p, BT.709; 1920 x 1080 by area averaging of the same frames. The GIF
     holds the key frames only (no dissolves), GIF_WIDTH px wide.
  Run of 8 October 2026: 287 Level-2A scenes in the seasons of 2015-2026; shown 29 September 2016, 20 August 2017,
  24 September 2018, 16 September 2020, 17 August 2021, 24 August 2022, 24 August 2023, 10 September 2024,
  7 September 2025, 27 September 2026 (orbit 108 in 2020, 2021, 2024). No scene for 2015 (every scene cloudy over the
  frame or, 25 September, under fresh snow) nor 2019 (3 August, the only clear one, has winter snow on 16 % of the
  frame outside the region; 4 September has cloud over the eastern third). 10 September 2024 keeps faint haze south-
  east of the tongue, below every test. 20.4 s.

Outputs:
  tex/animation/timelapse_2016_2026.mp4         3840 x 2160
  tex/animation/timelapse_2016_2026_1080p.mp4   1920 x 1080
  tex/animation/timelapse_2016_2026.gif         GIF_WIDTH px wide, the key frames
  code/timelapse_2016_2026_scenes.csv           one row per Level-2A scene of the season: date, platform, orbit,
                              baseline, sun elevation, the test values, the snow outside the region, the shift
                              (columns east, rows south, cells) and its spread, the decision ('shown' for the scene of
                              its year)
  stdout: the scenes shown, the reasons for the others, the years without a scene

usage, on s02 where the scenes are (env agl), from the mirror of this tree, a few minutes on WORKERS cores:
  rsync -a code/{common,outline_2026,timelapse}.py code/last_ice.tif s02:/data/befahl/projects/AGL/basodino/report/code/
  ssh s02 'source ~/miniconda3/bin/activate agl && cd /data/befahl/projects/AGL/basodino/report &&
           ~/.local/bin/nrun python code/timelapse.py'
  then copy tex/animation/timelapse_2016_2026* and code/timelapse_2016_2026_scenes.csv back
"""
import csv
import datetime
import os
import subprocess
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                         # one thread per worker
import contourpy  # noqa: E402
import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from scipy.ndimage import binary_dilation, gaussian_filter, maximum_filter, minimum_filter

from common import ROOT, write_csv

S2, ANIM = ROOT / "data/satellite_archive/sentinel2", ROOT / "tex/animation"
FONTS = ROOT / "data/satellite_archive/sentinel1/scripts/fonts"
NAME = "timelapse_2016_2026"
E0, N0, E1, N1 = 2677800, 1139610, 2682600, 1142310       # frame, LV95 m
CELL, W, H = 10.0, 3840, 2160                              # m; output pixels
PX = W / ((E1 - E0) / CELL)                                # pixels a cell
REG = 512                                                  # registration window, cells, about the frame
RE0, RN1 = 2677640, 1143520                                # its north-west corner, on the 20 m lattice
REF, ORBIT = "2025-08-08_S2B_R065", "R065"                 # reference scene of the co-registration; preferred orbit
SEASON = ("08-01", "09-30")
CLOUD_MAX, SHADOW_MAX, ICE_MAX, VEIL_MIN, BRIGHT_MAX, SPREAD_MAX, UP = 0.01, 0.02, 0.001, 0.75, 0.03, 0.5, 20
SNOW_MAX, ORBIT_SLACK = 0.1, 0.01
DARK = 0.05                                                # reflectance: single-band dropouts in deep shadow
BUFFER, SMOOTH, MIN_RING = 100.0, 1.0, 2000.0
FPS, HOLD, FADE, END_FADE, END, CRF, GIF_WIDTH, WORKERS = 30, 1.0, 0.6, 1.0, 4.0, 16, 1280, 8
LINE, OLD_LINE = 4.0, 2.5                                  # px at 4K: the year's line, the earlier years'
RED, PALE = np.array((197, 20, 21)), np.array((242, 196, 192))   # ACCENT and PALE_RED of the report palette

O = {}                                                     # code/outline_2026.py: cast shadow on its window
exec((ROOT / "code/outline_2026.py").read_text(encoding="utf-8").split("# --- the outline and its checks")[0], O)
O_ROWS = slice(round((RN1 - O["N1"]) / CELL), round((RN1 - O["N0"]) / CELL))
O_COLS = slice(round((O["E0"] - RE0) / CELL), round((O["E1"] - RE0) / CELL))
CAT = {r["tag"]: r for r in csv.DictReader(open(S2 / "catalog.csv")) if r["level"] == "L2A"}
T_REG = from_origin(RE0, RN1, CELL, CELL)
G16 = gpd.read_file(ROOT / "data/vectors.gpkg", layer="sgi_2016").query("`sgi-id` == 'C14-10'").union_all()
G26 = gpd.read_file(ROOT / "data/outline_2026/outline_2026-09-27.gpkg").union_all()
REGION = rasterize([G16.buffer(BUFFER)], (REG, REG), transform=T_REG).astype(bool)
CORE = rasterize([G26.buffer(-20.0)], (REG, REG), transform=T_REG).astype(bool)
FRAME = (slice(round((RN1 - N1) / CELL), round((RN1 - N0) / CELL)),
         slice(round((E0 - RE0) / CELL), round((E1 - RE0) / CELL)))   # the frame in the registration window
IN_FRAME = np.zeros((REG, REG), bool)
IN_FRAME[FRAME] = True


def read(path, res, bands):
    """Bands of an archive file over the registration window, float32, 0 -> NaN, value / 10000 unless SCL."""
    with rasterio.open(S2 / path) as src:
        w = from_bounds(RE0, RN1 - REG * CELL, RE0 + REG * CELL, RN1, src.transform).round_offsets().round_lengths()
        v = src.read(bands, window=w).astype(np.float32)
    v = v.repeat(round(res / CELL), axis=1).repeat(round(res / CELL), axis=2)   # 20 m -> 10 m, each cell 2 x 2
    return v if bands == [1] and path.endswith("_scl.tif") else np.where(v == 0, np.nan, v / 1e4)


def gradient(b04):
    """Gradient magnitude of B04, NaN as the mean, under a Hann window: the image that is registered."""
    gy, gx = np.gradient(np.where(np.isfinite(b04), b04, np.nanmean(b04)))
    g = np.hypot(gx, gy)
    return (g - g.mean()) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))


def shift(ref, img):
    """Sub-cell shift (rows, columns) d with img(x + d) = ref(x): phase correlation, peak refined to 1/UP cell by a
    matrix Fourier transform over 1.5 cells around it."""
    f = np.fft.fft2(img) * np.fft.fft2(ref).conj()
    f /= np.abs(f) + 1e-12
    c = np.fft.ifft2(f).real
    p = np.array(np.unravel_index(c.argmax(), c.shape), float)
    p = np.where(p > np.array(c.shape) / 2, p - c.shape, p)
    u = (np.arange(round(1.5 * UP)) - round(0.75 * UP)) / UP
    rows, cols = p[0] + u, p[1] + u
    er = np.exp(2j * np.pi * np.outer(rows, np.fft.fftfreq(c.shape[0])))
    ec = np.exp(2j * np.pi * np.outer(np.fft.fftfreq(c.shape[1]), cols))
    i, j = np.unravel_index((er @ f @ ec).real.argmax(), (len(rows), len(cols)))
    return rows[i], cols[j]


def survey(tag):
    """Test values, snow and shift of one scene (step 2 and 3 of the docstring)."""
    c = CAT[tag]
    b02, b03, b04, b08 = read(c["file_10m"], 10, [1, 2, 3, 4])
    b11 = read(c["file_20m"], 20, [1])[0]
    scl = read(c["file_scl"], 20, [1])[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        ndsi = (b03 - b11) / (b03 + b11)
    terrain = np.zeros((REG, REG), bool)
    terrain[O_ROWS, O_COLS] = binary_dilation(O["cast_shadow"](float(c["sun_elevation"]), float(c["sun_azimuth"])),
                                              iterations=2)
    margin = (slice(FRAME[0].start - 5, FRAME[0].stop + 5), slice(FRAME[1].start - 5, FRAME[1].stop + 5))
    with np.errstate(invalid="ignore"):
        rgb = np.stack([b02[margin], b03[margin], b04[margin]])
        gaps = ~np.isfinite(rgb)
        dropout = (gaps.sum(axis=0) == 1) & (np.nanmax(rgb, axis=0) < DARK)
        r = dict(nodata=int((gaps.any(axis=0) & ~dropout).sum()) + int((scl[IN_FRAME] < 2).sum()),
                 cloud=float((np.isin(scl, (8, 9, 10)) & (b02 > 0.3))[IN_FRAME].mean()),
                 shadow=float((((scl == 3) | (b08 < 0.08)) & ~terrain)[REGION].mean()),
                 shadow_frame=float((scl == 3)[IN_FRAME].mean()),
                 ice=float(((b03 > 0.3) & (ndsi < 0.4))[CORE].mean()), veil=float(np.nanmedian(ndsi[CORE])),
                 bright=float(((b02 > 0.35) & (ndsi < 0.5))[IN_FRAME].mean()),
                 snow=float(((ndsi > 0.4) & (b03 > 0.10) & (b08 > 0.11))[IN_FRAME & ~REGION].mean()))
    if r["nodata"]:
        return r
    r["dy"], r["dx"] = shift(GREF, gradient(b04))
    h = REG // 2
    q = np.array([shift(gradient(REF_B04[i:i + h, j:j + h]), gradient(b04[i:i + h, j:j + h]))
                  for i in (0, h) for j in (0, h)])
    r["spread"] = float(np.sqrt(((q - [r["dy"], r["dx"]]) ** 2).sum(axis=1).mean()))
    return r


REF_B04 = read(CAT[REF]["file_10m"], 10, [3])[0]
GREF = gradient(REF_B04)


def decide(r):
    """The first test of step 2 that the scene fails, or 'candidate'."""
    for test, fails in (("no data", r["nodata"] > 0), ("cloud", r["cloud"] > CLOUD_MAX),
                        ("shadow", max(r["shadow"], r["shadow_frame"]) > SHADOW_MAX), ("ice", r["ice"] > ICE_MAX),
                        ("veil", r["veil"] < VEIL_MIN), ("bright", r["bright"] > BRIGHT_MAX),
                        ("snow", r["snow"] > SNOW_MAX), ("registration", r.get("spread", 0) > SPREAD_MAX)):
        if fails:
            return test
    return "candidate"


# --- lines: anti-aliased alpha masks at 4K
def to_px(e, n):
    """LV95 metres -> pixel coordinates of the image at twice the size (pixel centres at whole numbers)."""
    return 2 * W * (np.asarray(e) - E0) / (E1 - E0) - 0.5, 2 * H * (N1 - np.asarray(n)) / (N1 - N0) - 0.5


def line_mask(lines, width):
    """Alpha (0-1, H x W x 1) of the lines (lists of LV95 (E, N) vertices), drawn at twice the size, averaged down."""
    img = Image.new("L", (2 * W, 2 * H), 0)
    d = ImageDraw.Draw(img)
    for xy in lines:
        x, y = to_px(xy[:, 0], xy[:, 1])
        d.line(list(zip(x, y)), fill=255, width=round(2 * width), joint="curve")
    return np.asarray(img.reduce(2), np.float32)[..., None] / 255


def ring_area(xy):
    """Area enclosed by a closed line, m2 (shoelace)."""
    x, y = xy[:, 0], xy[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


with rasterio.open(ROOT / "code/last_ice.tif") as src:
    LAST, _t = src.read(1), src.transform
_xs, _ys = _t.c + CELL * (np.arange(LAST.shape[1]) + 0.5), _t.f - CELL * (np.arange(LAST.shape[0]) + 0.5)
YEARS = range(2018, 2027)
LINES = {y: [xy for xy in contourpy.contour_generator(_xs, _ys, gaussian_filter((LAST >= y).astype(float), SMOOTH))
             .lines(0.5) if ring_area(xy) >= MIN_RING] for y in YEARS}

# --- text
FONT_TITLE = ImageFont.truetype(str(FONTS / "SourceSans3-Semibold.ttf"), 60)
FONT_YEAR = ImageFont.truetype(str(FONTS / "SourceSans3-Semibold.ttf"), 132)
FONT_TEXT = ImageFont.truetype(str(FONTS / "SourceSans3-Regular.ttf"), 44)
FONT_SMALL = ImageFont.truetype(str(FONTS / "SourceSans3-Regular.ttf"), 36)
CREDIT, PAD = "contains modified Copernicus Sentinel data 2016-2026", 80


def text(img, xy, s, font, anchor, fill=(255, 255, 255)):
    """Text with a soft shadow: the letters widened by 4 px, blurred, black at up to 60 %, so it reads on snow."""
    shadow = Image.new("L", img.size, 0)
    ImageDraw.Draw(shadow).text(xy, s, font=font, anchor=anchor, fill=153, stroke_width=4, stroke_fill=153)
    img.paste(Image.new("RGB", img.size, 0), mask=shadow.filter(ImageFilter.GaussianBlur(7)))
    ImageDraw.Draw(img).text(xy, s, font=font, anchor=anchor, fill=fill)


def bar(img, box, fill):
    """A filled rectangle with the same soft shadow as the text."""
    shadow = Image.new("L", img.size, 0)
    ImageDraw.Draw(shadow).rectangle([box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4], fill=153)
    img.paste(Image.new("RGB", img.size, 0), mask=shadow.filter(ImageFilter.GaussianBlur(7)))
    ImageDraw.Draw(img).rectangle(box, fill=fill)


def furniture(img):
    """Name of the glacier, top left; 1 km scale bar, top right; credit, bottom right."""
    text(img, (PAD, PAD), "Ghiacciaio del Basòdino", FONT_TITLE, "lt")
    x1, x0 = W - PAD, W - PAD - 1000 / CELL * PX
    bar(img, (x0, PAD + 50, x1, PAD + 58), (255, 255, 255))
    text(img, ((x0 + x1) / 2, PAD + 34), "1 kilometre", FONT_SMALL, "ms")
    text(img, (W - PAD, H - PAD), CREDIT, FONT_SMALL, "rs", (235, 235, 235))


def key(img, rows):
    """Key above the credit, bottom right: rows of (colour, width px, label), the last row lowest."""
    y = H - PAD - 64
    for colour, width, label in reversed(rows):
        text(img, (W - PAD, y), label, FONT_SMALL, "rs")
        lx = W - PAD - FONT_SMALL.getlength(label) - 24
        bar(img, (lx - 72, y - 13 - width / 2, lx, y - 13 + width / 2), tuple(int(v) for v in colour))
        y -= 56


# --- frames
def tone(r):
    """Reflectance -> display value 0-1: the curve of step 4."""
    r = np.maximum(np.nan_to_num(r), 0.0)
    t = np.where(r <= 0.6, r / 0.9, 2 / 3 + 1 / 3 * (1 - np.exp(-(r - 0.6) / 0.3)))
    return np.where(t <= 0.0031308, 12.92 * t, 1.055 * np.power(np.maximum(t, 0.0031308), 1 / 2.4) - 0.055)


def upscale(v, box):
    """One channel (float, 0-1) to W x H by Lanczos over box, clamped to the 3 x 3 neighbourhood's range."""
    def resize(a, f):
        return np.asarray(Image.fromarray(a.astype(np.float32), "F").resize((W, H), f, box=box))
    return np.clip(resize(v, Image.LANCZOS), resize(minimum_filter(v, 3), Image.BILINEAR),
                   resize(maximum_filter(v, 3), Image.BILINEAR))


def picture(s):
    """The scene, toned and upscaled, float H x W x 3 in 0-1."""
    v = tone(read(CAT[s["tag"]]["file_10m"], 10, [3, 2, 1]))
    top, left = FRAME[0].start + s["dy"], FRAME[1].start + s["dx"]
    return np.dstack([upscale(c, (left, top, left + (E1 - E0) / CELL, top + (N1 - N0) / CELL)) for c in v])


def over(rgb, alpha, colour, opacity=1.0):
    return rgb * (1 - opacity * alpha) + np.asarray(colour) / 255 * opacity * alpha


def year_frame(s):
    """Key frame of one year: the scene, the lines up to its year, the text. RGB uint8 H x W x 3."""
    rgb, year = picture(s), int(s["tag"][:4])
    for y in YEARS:
        if y < year:
            rgb = over(rgb, line_mask(LINES[y], OLD_LINE), (255, 255, 255), 0.5)
    if year in YEARS:
        rgb = over(rgb, line_mask(LINES[year], LINE), RED)
    img = Image.fromarray(np.round(255 * np.clip(rgb, 0, 1)).astype(np.uint8))
    furniture(img)
    date = datetime.date.fromisoformat(s["tag"][:10])
    text(img, (PAD, H - PAD), str(year), FONT_YEAR, "ls")
    text(img, (PAD + FONT_YEAR.getlength(str(year)) + 28, H - PAD), f"{date.day} {date:%B}", FONT_TEXT, "ls")
    rows = [(RED, LINE + 2, "ice at the end of summer")] if year in YEARS else []
    if year > YEARS[0]:
        rows.append(((200, 200, 200), OLD_LINE + 2, "earlier years, from 2018"))
    key(img, rows)
    return np.asarray(img)


def end_frame(s):
    """End card: the last scene darkened, every line from pale (2018) to red (2026), its key."""
    rgb = 0.7 * picture(s)
    ramp = {y: PALE + (RED - PALE) * i / (len(YEARS) - 1) for i, y in enumerate(YEARS)}
    for y in YEARS:
        rgb = over(rgb, line_mask(LINES[y], LINE if y == YEARS[-1] else OLD_LINE + 0.5), ramp[y])
    img = Image.fromarray(np.round(255 * np.clip(rgb, 0, 1)).astype(np.uint8))
    furniture(img)
    text(img, (PAD, H - PAD - 96), "ice at the end of summer", FONT_TEXT, "ls")
    x = PAD
    for y in YEARS:
        bar(img, (x, H - PAD - 52, x + 52, H - PAD - 42), tuple(int(v) for v in ramp[y]))
        x += 64
    text(img, (PAD, H - PAD), "2018", FONT_SMALL, "ls")
    text(img, (x - 12, H - PAD), "2026", FONT_SMALL, "rs")
    return np.asarray(img)


def render(job):
    kind, s = job
    return year_frame(s) if kind == "year" else end_frame(s)


if __name__ == "__main__":
    tags = sorted((t for t in CAT if SEASON[0] <= t[5:10] <= SEASON[1]), key=lambda k: CAT[k]["datetime"])
    with Pool(WORKERS) as pool:
        stats = dict(zip(tags, pool.map(survey, tags, chunksize=4)))
    shown = []
    for t in tags:
        stats[t]["decision"] = decide(stats[t])
    for year in sorted({t[:4] for t in tags}):
        cand = [t for t in tags if t[:4] == year and stats[t]["decision"] == "candidate"]
        if cand:
            least = min(stats[t]["snow"] for t in cand)
            cand = [t for t in cand if CAT[t]["orbit"] == ORBIT and stats[t]["snow"] <= least + ORBIT_SLACK] or cand
            t = min(cand, key=lambda k: stats[k]["snow"])
            stats[t]["decision"] = "shown"
            shown.append(dict(tag=t, dy=stats[t]["dy"], dx=stats[t]["dx"]))

    def num(v, digits):
        return "" if v is None or not np.isfinite(v) else round(float(v), digits)
    write_csv(f"{NAME}_scenes.csv", ["scene", "datetime", "platform", "orbit", "processing_baseline", "source",
                                     "sun_elevation_deg", "nodata_cells", "cloud_frame", "shadow_region",
                                     "shadow_frame", "cloud_core", "veil_core_ndsi", "bright_frame",
                                     "snow_outside_region", "shift_east_cells", "shift_south_cells",
                                     "shift_spread_cells", "decision"],
              [[t, CAT[t]["datetime"], CAT[t]["platform"], CAT[t]["orbit"], CAT[t]["processing_baseline"],
                CAT[t]["source"], CAT[t]["sun_elevation"], stats[t]["nodata"], num(stats[t]["cloud"], 4),
                num(stats[t]["shadow"], 4), num(stats[t]["shadow_frame"], 4), num(stats[t]["ice"], 4),
                num(stats[t]["veil"], 3), num(stats[t]["bright"], 4), num(stats[t]["snow"], 4),
                num(stats[t].get("dx"), 3), num(stats[t].get("dy"), 3), num(stats[t].get("spread"), 3),
                stats[t]["decision"]] for t in tags])
    print("shown: " + ", ".join(s["tag"] for s in shown))
    print("decisions: " + ", ".join(f"{d} {sum(r['decision'] == d for r in stats.values())}"
                                    for d in ("shown", "candidate", "no data", "cloud", "shadow", "ice", "veil",
                                              "bright", "snow", "registration")))
    print("years without a scene: " + ", ".join(y for y in sorted({t[:4] for t in tags})
                                                if all(s["tag"][:4] != y for s in shown)))

    with Pool(min(WORKERS, len(shown) + 1)) as pool:
        keys = pool.map(render, [("year", s) for s in shown] + [("end", shown[-1])], chunksize=1)

    ANIM.mkdir(parents=True, exist_ok=True)
    x264 = ["-c:v", "libx264", "-preset", "slow", "-crf", str(CRF), "-pix_fmt", "yuv420p", "-colorspace", "bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart", "-threads", "16"]
    graph = (f"[0]split[a][b];[a]scale=out_color_matrix=bt709:out_range=tv[uhd];"
             f"[b]scale={W // 2}:{H // 2}:flags=area:out_color_matrix=bt709:out_range=tv[hd]")
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                            f"{W}x{H}", "-r", str(FPS), "-i", "-", "-filter_complex", graph,
                            "-map", "[uhd]", *x264, str(ANIM / f"{NAME}.mp4"), "-map", "[hd]", *x264,
                            str(ANIM / f"{NAME}_1080p.mp4")], stdin=subprocess.PIPE)
    n = 0
    for i, k in enumerate(keys):
        last = i == len(keys) - 1
        for _ in range(round((END if last else HOLD) * FPS)):
            enc.stdin.write(k.tobytes())
            n += 1
        if last:
            break
        m = round((END_FADE if i == len(keys) - 2 else FADE) * FPS)
        a0, a1 = k.astype(np.float32), keys[i + 1].astype(np.float32)
        for j in range(1, m + 1):
            a = j / (m + 1)
            a = a * a * (3 - 2 * a)                                    # smoothstep
            enc.stdin.write(np.round(a0 + a * (a1 - a0)).astype(np.uint8).tobytes())
            n += 1
    enc.stdin.close()
    assert enc.wait() == 0, "ffmpeg failed"
    gif = [Image.fromarray(k).resize((GIF_WIDTH, GIF_WIDTH * H // W), Image.BOX) for k in keys]
    gif[0].save(ANIM / f"{NAME}.gif", save_all=True, append_images=gif[1:], loop=0, optimize=True,
                duration=[round(1000 * (HOLD + FADE))] * (len(gif) - 1) + [round(1000 * END)])
    print(f"{len(shown)} scenes, {n} frames, {n / FPS:.1f} s; written tex/animation/{NAME}.mp4, _1080p.mp4, .gif, "
          f"code/{NAME}_scenes.csv")
