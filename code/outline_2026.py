"""Our outline of Ghiacciaio del Basòdino (SGI C14-10) for September 2026 from Sentinel-2, its checks and its figure.

Scene: 27 September 2026, Sentinel-2C, relative orbit 65, 10:28 UTC, sun 40.6 deg above the horizon, the latest scene
clear over the glacier and its surroundings. On 29 September (Sentinel-2A, orbit 65) cloud covers 26 % of the 10 km
square and 9 % of the 1850 extent and touches the eastern margin. No scene of 20-29 September has fresh snow outside
the glaciers (snow and ice on 1.1 % of the square, catalog.csv snowice_square; quicklooks).

Inputs (paths below basodino/; in the s02 mirror data/ is the project's derived/, see usage):
  data/satellite_archive/sentinel2/catalog.csv       scene list, sun elevation and azimuth (src/archive.py)
  data/satellite_archive/sentinel2/scenes/<year>/<tag>_10m.tif, _20m.tif
                                                     surface reflectance on the 10 m and 20 m LV95 grids
  data/vectors.gpkg, layer glacier_2024              swissTLM3D 2024 outline of C14-10 (3 polygons, 1.380 km2)
  data/dem_2m.tif                                    swissALTI3D 2 m, for the cast shadows
  data/course/mask_basodino_2021.asc                 course mask, October 2021, 10 m cells
  data/glamos_basodino_massbalance_observation.csv   GLAMOS glacier areas (2024: 1.489 km2, 2025: 1.354 km2)
  data/ortho_1m.tif                                  SWISSIMAGE 2024 (21 and 29 August), for the check images

Method, the same for every scene:
  1. NDSI = (B03 - B11) / (B03 + B11) on the 10 m grid: B03 at 10 m, B11 interpolated bilinearly from 20 m.
  2. Region: the reference outline (2024, all three polygons) buffered by 50 m, so that separate snowfields and
     Ghiacciaio di Basòdino Nord (98 m away) do not count.
  3. Cast shadow: terrain shadow for the sun of the scene on the 10 m grid (2 m elevation model averaged 5 x 5): a
     pixel is in shadow where the terrain within 2 km towards the sun rises above the sun's ray. The grid
     convergence (0.8 deg) is neglected; the elevation model holds the glacier surface of September 2021.
  4. Thresholds from the scene's own histogram, Otsu's method (400 bins on -1 to 1): one for the pixels of the region
     outside the cast shadow, one for those in it inside the reference outline; in shadow, ice is accepted only inside
     the reference outline (the glacier did not advance). Shadowed ice keeps an NDSI near 0.2 (sunlit 0.88), shadowed
     rock lies near -0.4, so the sunlit threshold drops the ice under the south wall of the tongue, and a threshold
     taken over all shadowed pixels of the region (-0.23 on 27 September) adds the shadowed rock walls (+0.02 km2;
     both found against 13 August 2026, sun 55 deg, when no cast shadow fell on the glacier).
     27 September: 0.335 sunlit, -0.140 in shadow (the other scenes: code/outline_2026.csv).
  5. Ice: the iso-line NDSI = threshold between pixel centres (marching squares) bounds the ice polygons. Narrow
     strips of ice are joined: B11 has 20 m pixels, so across a strip narrower than about 20 m it mixes the ice with
     the rock beside it and the NDSI falls below the threshold where the 10 m bands still see ice (the strip at
     E 2680345, N 1141055, about 15 m wide: 67-94 % ice by the 10 m bands, 20-52 % by B11). Candidates are sunlit
     pixels (step 6) whose NDSI lies at most DROP = 0.15 below the threshold and whose green and near-infrared index
     GNI = (B03 - B08) / (B03 + B08), all 10 m, is at or above its own Otsu threshold over the pixels of the region
     outside the cast shadow (27 September: 0.020; bare ice 0.12, rock -0.07). A 4-connected group of candidates (the
     iso-line may cut a diagonal step) that touches two separate ice polygons joins them: its pixels are raised by
     DROP and the iso-line is redrawn. A group that would only close a bay of one polygon is not joined (that would
     change no late-September scene, and close bays of 2100 and 18000 m2 on 12 September and 13 August); but any
     polygon counts as a side, also a speck (6 to 27 m2 on 25 and 22 September, outline_2026_review.py; on 27
     September every side has 4 or 5 ice votes). DROP: a 10 m pixel all ice whose B11 sees a quarter ice lies 0.15
     below the threshold (27 September, medians of the sunlit region: ice B03 0.37, B11 0.02; rock B03 0.22, B11
     0.33). The two strips of 27 September hold only for DROP of 0.134 (place 3, E 2680345, N 1141055) and 0.136
     (gully, E 2679800, N 1141330) or more; at 0.133 neither joins (1023365 m2 against 1025235 m2 at 0.15;
     code/outline_2026_drop.py).
  6. Glacier: the largest polygon is the glacier body; its holes are rock or debris, the other polygons detached
     blocks of ice, counted only when at least half inside the reference outline. The five late-September scenes,
     each outlined by steps 1-5, vote per pixel: a scene votes where it sees the pixel sunlit (outside the cast
     shadow, and B02 or B08 at least DARK = 0.06), for rock or for ice. A hole stays when one of its pixels has
     VOTES = 3 of the five for rock, a block when one of its pixels has 3 for ice. So a rock spot of a few pixels
     stays when it persists and a one-scene speck does not; nor does a hole or block seen only in shadow, where the
     NDSI is noise (deep shadow: the hole of 1833 m2 at the tongue's south wall reads -0.29 to 0.30 on 27 September,
     0.5 to 0.8 sunlit on 12 September and 13 August) or rests on the shadow threshold (the polygon of 490 m2 at
     E 2679146, N 1140797: ice by it in three scenes, rock on 13 August). Every other scene (one each, no vote)
     keeps holes and blocks of MIN_PART = 1000 m2 (10 pixels) or more.
  Debris-covered ice: the NDSI sees it as rock. The Swiss Glacier Inventory 2016 maps 1378 m2 of debris on C14-10;
  the margins left out inside the 2024 outline were compared with SWISSIMAGE 2024 (crevassed bare ice in August
  2024) and with the scenes of September 2025 and August 2026, in which rock knobs emerge and join (check images).
  Band B08 (10 m) enters only through GNI and the deep-shadow test. Other uses, tested on the five late-September
  scenes (2026-10-06; the first three with the rules of commit 10a3d87), are not adopted: sharpening B11 to 10 m
  (B11 x B08 / B08 of the 20 m pixel) raised the area by 0.8 % and the mean boundary distance between the scenes
  from 3.8 to 4.3 m; excluding water (B08 below 0.11) found none (the dark pixels above the threshold are shadowed
  ice at the margins); widening the terrain shadow by 30 m where B08 is below 0.11 changed the area spread of the
  scenes from 0.015 to 0.013 km2 and left the mean boundary distance at 3.8 m; adding the deep shadow (B02 and B08
  below 0.06) to the cast shadow fills the hole at the tongue's south wall but makes 40 pixels ice on 27 September
  that are not ice on 13 August (sun 55 deg).
Checks: the same method on four more late-September scenes (Sentinel-2B 25th and 2A 22nd, orbit 108; 2B 22nd,
  orbit 65; 2C 20th, orbit 108, hazy): area spread and mean boundary distance to the 27 September outline
  (symmetric mean of point-to-line distances, points every 2 m; the five share the votes of step 6, so these do not
  test which holes and blocks stay). The two relative orbits pass 10 minutes apart, so their cast shadows coincide;
  shadow is told from ice with 12 September and 13 August 2026 (sun 46 and 55 deg): ice inside the cast shadow of
  27 September, and the agreement of the shadowed pixels with 13 August. The method against reference outlines of
  the same summer: 10 September 2024 against swissTLM3D 2024, 1 September 2021 against the course mask (region from
  that mask). 7 September 2025 against the GLAMOS area of 2025. These single scenes test steps 1-5 and the size rule
  of step 6 (MIN_PART), not its vote. 2024 cannot vote: 10 September is its only clear scene without fresh snow
  (catalog.csv: 15 September snow or ice on 15 % of the 10 km square; 21, 23 and 31 August and 12 September cloud
  on 18-35 % of the 2024 glacier).
  Area uncertainty: pixel size (10 m) x perimeter, and the spread of the five late-September scenes.

Outputs:
  data/outline_2026/outline_2026-09-27.gpkg, .geojson   the outline: the glacier body and the detached blocks, one
                                 MultiPolygon (body first), EPSG:2056
  code/outline_2026.csv          one row per scene and reference: thresholds, area (body and blocks), area of the
                                 main body (the largest polygon), perimeter, pixel x perimeter, ice in the cast shadow
                                 of 27 September, differences to 2024 and 2021, mean boundary distance to the outline
                                 named in compared_with; last row the late-September spread
  tex/figures/outline_2026_si.pdf   150 x 95 mm, map at 1:20 000, graphics only (tex/SIDECARS.md)
  tex/preview/outline_2026_check_{overview,tongue,upper,north,west}.png   checks at full resolution: Sentinel-2 of
                                 27 September (10 m pixels) and SWISSIMAGE 2024 with all outlines and the shadow
  stdout: the table, the joined pixels per scene, the holes and polygons of 27 September with their votes, the
          shadow agreement, and per quadrant (seen from the glacier centre) how far the 2024 and 2021 outlines lie
          from the 2026 outline and how much of their ice is ice-free in 2026

usage, on s02 where the scenes are (env agl), from the mirror of this tree: report/data -> ../derived, and
derived/course/ holds the course masks (copied from data/course/ on 2026-10-06, as on s04):
  rsync -a code/common.py code/outline_2026.py s02:/data/befahl/projects/AGL/basodino/report/code/
  ssh s02 'source ~/miniconda3/bin/activate agl && cd /data/befahl/projects/AGL/basodino/report &&
           ~/.local/bin/nrun python code/outline_2026.py'
  then copy code/outline_2026.csv, data/outline_2026/, tex/figures/outline_2026_si.pdf and the checks back
"""
import csv

import contourpy
import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from scipy.ndimage import binary_dilation, label, map_coordinates, zoom

from common import ACCENT, COURSE, GLACIER, LAB, MM, NUM, ROOT, figure_style, read_esri_ascii, save, write_csv

S2, OUT, PREVIEW = ROOT / "data/satellite_archive/sentinel2", ROOT / "data/outline_2026", ROOT / "tex/preview"
MAIN = "2026-09-27_S2C_R065"
LATE = ["2026-09-25_S2B_R108", "2026-09-22_S2A_R108", "2026-09-22_S2B_R065", "2026-09-20_S2C_R108"]
HIGH_SUN = ["2026-09-12_S2B_R065", "2026-08-13_S2B_R065"]    # 13 August S2A (orbit 108): cloud shadow on the glacier
YEAR_BEFORE, CHECK_2024, CHECK_2021 = "2025-09-07_S2B_R065", "2024-09-10_S2A_R108", "2021-09-01_S2B_R108"
E0, N0, E1, N1 = 2678600, 1139900, 2681900, 1142000     # analysis window, LV95 m, edges on the 20 m grid
PIXEL, BUFFER, MIN_PART = 10.0, 50.0, 1000.0             # m, m, m2
DROP, DARK, VOTES = 0.15, 0.06, 3                        # NDSI; reflectance; scenes of the five of late September
REACH, MARGIN = 2000.0, 2000.0                           # m: terrain searched towards the sun; elevation model margin
EPOCH_2024 = "#0c2f5d"                                   # maps/style.json glacier_epochs 2024
CENTRE = (2679622, 1141059)                              # glacier centre, data/README_swiss_geodata.md

X = E0 + PIXEL / 2 + PIXEL * np.arange(round((E1 - E0) / PIXEL))     # pixel centres
Y = N1 - PIXEL / 2 - PIXEL * np.arange(round((N1 - N0) / PIXEL))
T10 = from_origin(E0, N1, PIXEL, PIXEL)
CAT = {r["tag"]: r for r in csv.DictReader(open(S2 / "catalog.csv"))}
G24 = gpd.read_file(ROOT / "data/vectors.gpkg", layer="glacier_2024").query("sgi == 'C14-10'").union_all()
M21, H21 = read_esri_ascii(COURSE / "mask_basodino_2021.asc")
M21 = (M21 == 1).astype(np.uint8)
T21 = from_origin(H21["xllcorner"], H21["yllcorner"] + H21["nrows"] * H21["cellsize"], H21["cellsize"], H21["cellsize"])
G21 = shapely.union_all([shapely.geometry.shape(g) for g, _ in shapes(M21, mask=M21 == 1, transform=T21)])


def read(path, box=(E0, N0, E1, N1), band=None):
    with rasterio.open(path) as src:
        win = from_bounds(*box, src.transform).round_offsets().round_lengths()
        return src.read(band, window=win).astype(np.float32)


def raster(geom):
    """True on the pixels of the window whose centres lie in geom."""
    return rasterize([geom], (len(Y), len(X)), transform=T10).astype(bool)


with rasterio.open(ROOT / "data/dem_2m.tif") as src:
    win = from_bounds(E0 - MARGIN, N0 - MARGIN, E1 + MARGIN, N1 + MARGIN, src.transform).round_offsets().round_lengths()
    z = src.read(1, window=win)
    z = np.where(z == src.nodata, np.nan, z)
DEM = z.reshape(z.shape[0] // 5, 5, z.shape[1] // 5, 5).mean(axis=(1, 3))     # 10 m, the window plus MARGIN


def cast_shadow(elevation, azimuth):
    """True where the terrain within REACH towards the sun rises above the sun's ray (10 m grid of the window)."""
    e, a, m = np.radians(elevation), np.radians(azimuth), round(MARGIN / PIXEL)
    row, col = np.mgrid[m:m + len(Y), m:m + len(X)].astype(np.float32)
    z0, shade = DEM[m:m + len(Y), m:m + len(X)], np.zeros((len(Y), len(X)), bool)
    for d in np.arange(PIXEL / 2, REACH, PIXEL / 2):
        ahead = map_coordinates(DEM, [row - d * np.cos(a) / PIXEL, col + d * np.sin(a) / PIXEL], order=1)
        shade |= ahead > z0 + d * np.tan(e)
    return shade


def otsu(v):
    """Threshold between the two classes of the histogram of v with the largest between-class variance."""
    h, edges = np.histogram(v, 400, (-1.0, 1.0))
    c = (edges[:-1] + edges[1:]) / 2
    n0, m0 = np.cumsum(h), np.cumsum(h * c)
    n1, m1 = n0[-1] - n0, m0[-1] - m0
    return edges[1 + np.argmax(n0 * n1 * (m0 / np.maximum(n0, 1) - m1 / np.maximum(n1, 1)) ** 2)]


def scene(tag):
    """Reflectance of a scene on the 10 m grid of the window (B11 bilinear from 20 m, 0 = no data) and its sun."""
    c = CAT[tag]
    v10, v20 = read(S2 / c["file_10m"]), read(S2 / c["file_20m"], band=1)
    b02, b03, b04, b08 = np.where(v10 == 0, np.nan, v10 / 1e4)
    b11 = zoom(np.where(v20 == 0, np.nan, v20 / 1e4), 2, order=1, mode="nearest", grid_mode=True)
    return dict(B02=b02, B03=b03, B04=b04, B08=b08, B11=b11, sun=(float(c["sun_elevation"]), float(c["sun_azimuth"])))


def polygons(z):
    """Polygons where z > 0, bounded by the iso-line z = 0 between pixel centres."""
    pts, offs = contourpy.contour_generator(X, Y, z, fill_type="OuterOffset").filled(0.0, np.inf)
    return [shapely.Polygon(p[o[0]:o[1]], [p[o[i]:o[i + 1]] for i in range(1, len(o) - 1)]) for p, o in zip(pts, offs)]


def outline(tag, reference=G24):
    """Ice of one scene (steps 1-5 of the docstring): its polygons after the joins, largest first; z = NDSI minus
    threshold; the sunlit pixels of step 6 and the joined ones; thresholds and cast shadow."""
    s = scene(tag)
    ndsi = (s["B03"] - s["B11"]) / (s["B03"] + s["B11"])
    gni = (s["B03"] - s["B08"]) / (s["B03"] + s["B08"])
    region, inside, shade = raster(reference.buffer(BUFFER)), raster(reference), cast_shadow(*s["sun"])
    assert np.isfinite(ndsi[region]).all(), f"{tag}: no data in the region"
    t_sun = otsu(ndsi[region & ~shade])
    t_shade = otsu(ndsi[inside & shade]) if (inside & shade).sum() >= 100 else np.nan    # too few: no shadow class
    z = ndsi - np.where(shade & np.isfinite(t_shade), t_shade, t_sun)
    z[~region | (shade & ~inside & np.isfinite(t_shade))] = -1.0
    sunlit = ~shade & ((s["B02"] >= DARK) | (s["B08"] >= DARK))
    part = rasterize([(p, i) for i, p in enumerate(polygons(z), 1)], z.shape, transform=T10, dtype="int32")
    groups, n = label(sunlit & (z <= 0) & (z >= -DROP) & (gni >= otsu(gni[region & ~shade])))    # 4-connected
    joined = np.zeros(z.shape, bool)
    for group in (groups == i for i in range(1, n + 1)):
        if len(set(part[binary_dilation(group) & ~group]) - {0}) >= 2:      # touches two ice polygons
            joined |= group
    z[joined] += DROP
    return dict(parts=sorted(polygons(z), key=lambda p: -p.area), z=z, sunlit=sunlit, joined=joined,
                reference=reference, t_sun=t_sun, t_shade=t_shade, shade=shade, bands=s)


def glacier(r, ice=None, rock=None):
    """Step 6: the body (largest polygon) with the holes that stay, and the detached blocks that stay, as one
    MultiPolygon, body first. ice, rock: per pixel the late-September scenes that see it sunlit as ice, as rock;
    without them the size rule."""
    body, *rest = r["parts"]
    if ice is None:
        holes = [h for h in body.interiors if shapely.Polygon(h).area >= MIN_PART]
        blocks = [p for p in rest if p.area >= MIN_PART]
    else:
        holes = [h for h in body.interiors if rock[raster(shapely.Polygon(h))].max(initial=0) >= VOTES]
        blocks = [p for p in rest if ice[raster(p)].max(initial=0) >= VOTES]
    blocks = [p for p in blocks if p.intersection(r["reference"]).area >= p.area / 2]
    return shapely.MultiPolygon([shapely.Polygon(body.exterior, holes), *blocks])


def boundary_distance(a, b):
    """Symmetric mean distance (m) between the outlines of polygons a and b, points every 2 m."""
    def one_way(p, q):
        return shapely.distance(shapely.points(shapely.get_coordinates(shapely.segmentize(p.boundary, 2.0))),
                                q.boundary).mean()
    return (one_way(a, b) + one_way(b, a)) / 2


def margin_shift(before, after, name):
    """Print per quadrant seen from CENTRE how far the outline `before` lies outside `after` (points every 10 m)
    and how much of its area is ice-free in `after`."""
    pts = shapely.get_coordinates([shapely.segmentize(p.exterior, 10.0) for p in shapely.get_parts(before)])
    d = shapely.distance(shapely.points(pts), after)
    bearing = np.degrees(np.arctan2(pts[:, 0] - CENTRE[0], pts[:, 1] - CENTRE[1]))
    lost = shapely.difference(before, after)
    (w0, _, e0, _), (w1, _, e1, _) = shapely.bounds(before), shapely.bounds(after)
    print(f"{name}: {lost.area / 1e6:.3f} km2 ice-free in 2026; outline within 20 m of 2026 on {np.mean(d <= 20):.0%}; "
          f"west end {w0:.0f} -> {w1:.0f}, east end {e0:.0f} -> {e1:.0f}")
    for quadrant, b in (("north", 0), ("east", 90), ("south", 180), ("west", 270)):
        m = np.abs((bearing - b + 180) % 360 - 180) < 45
        wedge = shapely.Polygon([CENTRE] + [(CENTRE[0] + 1e4 * np.sin(np.radians(a)), CENTRE[1] + 1e4 * np.cos(
            np.radians(a))) for a in np.linspace(b - 45, b + 45, 19)])
        print(f"   {quadrant:5s} median {np.median(d[m]):3.0f} m, largest {d[m].max():3.0f} m, "
              f"ice-free {shapely.intersection(lost, wedge).area / 1e6:.3f} km2")


# --- the outline and its checks
res = {tag: outline(tag) for tag in [MAIN] + LATE + HIGH_SUN + [YEAR_BEFORE, CHECK_2024]}
res[CHECK_2021] = outline(CHECK_2021, G21)
ICE = sum((res[t]["z"] > 0) & res[t]["sunlit"] for t in [MAIN] + LATE)      # step 6, votes per pixel
ROCK = sum((res[t]["z"] <= 0) & res[t]["sunlit"] for t in [MAIN] + LATE)
for tag, r in res.items():
    r["glacier"] = glacier(r, *((ICE, ROCK) if tag in [MAIN] + LATE else ()))
main, shade_main = res[MAIN]["glacier"], res[MAIN]["shade"]
a24, a21 = G24.area, M21.sum() * H21["cellsize"] ** 2
with open(ROOT / "data/glamos_basodino_massbalance_observation.csv", encoding="utf-8") as f:
    glamos = {int(r[4][:4]): float(r[10]) * 1e6 for r in csv.reader(f) if len(r) > 10 and r[1] == "C14-10"}


def row(name, role, geom=None, area=None, tag=None, compared=("", None)):
    c, r = CAT.get(tag, {}), res.get(tag, {})
    area = geom.area if area is None else area
    return [name, role, c.get("platform", ""), c.get("orbit", ""), round(r["bands"]["sun"][0], 1) if r else "",
            round(float(r["t_sun"]), 3) if r else "",
            round(float(r["t_shade"]), 3) if r and np.isfinite(r["t_shade"]) else "", round(area),
            round(max(p.area for p in shapely.get_parts(geom))) if geom else "",
            round(geom.length) if geom else "", round(PIXEL * geom.length) if r else "",
            round((raster(geom) & shade_main).sum() * PIXEL ** 2) if geom else "", round(area - a24),
            round(area - a21), compared[0], round(boundary_distance(compared[1], geom), 1) if compared[1] else ""]


late = [res[t]["glacier"] for t in [MAIN] + LATE]
spread = max(p.area for p in late) - min(p.area for p in late)
pairs = [boundary_distance(p, q) for i, p in enumerate(late) for q in late[i + 1:]]
to_main = ("2026-09-27", main)
rows = ([row(MAIN, "outline", main, tag=MAIN)]
        + [row(t, "late September", res[t]["glacier"], tag=t, compared=to_main) for t in LATE]
        + [row(t, "higher sun", res[t]["glacier"], tag=t, compared=to_main) for t in HIGH_SUN]
        + [row(YEAR_BEFORE, "year before", res[YEAR_BEFORE]["glacier"], tag=YEAR_BEFORE, compared=to_main),
           row(CHECK_2024, "method check", res[CHECK_2024]["glacier"], tag=CHECK_2024,
               compared=("swissTLM3D 2024", G24)),
           row(CHECK_2021, "method check", res[CHECK_2021]["glacier"], tag=CHECK_2021,
               compared=("course mask 2021", G21)),
           row("swissTLM3D 2024", "reference", G24, compared=to_main),
           row("course mask 2021", "reference", G21, area=a21, compared=to_main),
           row("GLAMOS 2024", "reference", area=glamos[2024]), row("GLAMOS 2025", "reference", area=glamos[2025]),
           ["late September, 5 scenes", "spread", "", "", "", "", "", round(spread), "", "", "", "", "", "",
            "each other", round(float(np.mean(pairs)), 1)]])
write_csv("outline_2026.csv", ["scene", "role", "platform", "orbit", "sun_elevation_deg", "threshold_sunlit",
                               "threshold_shadow", "area_m2", "area_main_body_m2", "perimeter_m",
                               "pixel_x_perimeter_m2", "ice_in_shadow_of_27_september_m2", "minus_2024_m2",
                               "minus_2021_m2", "compared_with", "boundary_distance_m"], rows)
for r in rows:
    print(*r, sep="  ")
for t, r in res.items():
    groups, n = label(r["joined"])
    print(f"{t}: joined {r['joined'].sum()} px in {n} groups at " + ", ".join(
        f"{X[c].mean():.0f} {Y[rr].mean():.0f}" for rr, c in (np.nonzero(groups == i) for i in range(1, n + 1))))
body, *rest = res[MAIN]["parts"]
for kind, geoms, votes in (("hole", [shapely.Polygon(h) for h in body.interiors], ROCK), ("polygon", rest, ICE)):
    for g in geoms:
        p = g.representative_point()
        print(f"27 September {kind} {g.area:.0f} m2 at {p.x:.0f} {p.y:.0f}: {votes[raster(g)].max(initial=0)} of 5 "
              f"scenes, {g.intersection(G24).area / g.area:.0%} inside 2024, "
              f"{'stays' if main.contains(p) == (kind == 'polygon') else 'goes'}")
ref = raster(res[HIGH_SUN[1]]["glacier"])
for t in [MAIN] + LATE:
    zone = raster(G24.buffer(BUFFER)) & res[t]["shade"]
    ours = raster(res[t]["glacier"])
    print(f"{t}: cast shadow {zone.sum()} px, ice {ours[zone].sum()} px, 13 August {ref[zone].sum()} px, "
          f"agreement {np.mean(ours[zone] == ref[zone]):.3f}")
print(f"late September: spread {spread:.0f} m2, mean boundary distance between the scenes {np.mean(pairs):.1f} m "
      f"(largest {max(pairs):.1f} m); 27 September: main body {main.geoms[0].area:.0f} m2, "
      f"{len(main.geoms) - 1} detached blocks {main.area - main.geoms[0].area:.0f} m2")
margin_shift(G24, main, "2024 swissTLM3D")
margin_shift(G21, main, "2021 course mask")
print(f"2026 ice outside the 2024 outline: {shapely.difference(main, G24).area:.0f} m2")

OUT.mkdir(exist_ok=True)
gdf = gpd.GeoDataFrame([dict(name="Ghiacciaio del Basòdino", sgi="C14-10", date=MAIN[:10],
                             scene=CAT[MAIN]["product"],
                             method="normalised difference snow index iso-line, Otsu thresholds sunlit and in shadow; "
                                    "narrow strips joined by the 10 m bands; holes and detached blocks by the votes "
                                    "of five late-September scenes",
                             threshold_sunlit=round(float(res[MAIN]["t_sun"]), 3),
                             threshold_shadow=round(float(res[MAIN]["t_shade"]), 3), area_m2=round(main.area),
                             area_main_body_m2=round(main.geoms[0].area), detached_blocks=len(main.geoms) - 1,
                             perimeter_m=round(main.length), geometry=main)], crs="EPSG:2056")
for ext in ("gpkg", "geojson"):
    (OUT / f"outline_{MAIN[:10]}.{ext}").unlink(missing_ok=True)
gdf.to_file(OUT / f"outline_{MAIN[:10]}.gpkg", layer="outline_2026", driver="GPKG")
gdf.to_file(OUT / f"outline_{MAIN[:10]}.geojson", driver="GeoJSON", COORDINATE_PRECISION=2)


# --- figure: true colour of 27 September with the three outlines; 150 x 95 mm, map at 1:20 000
def true_colour(s, white=0.6):
    """sRGB encoding of reflectance / white (no snow in late September 2026: bare ice stays below 0.6)."""
    x = np.clip(np.nan_to_num(np.dstack([s["B04"], s["B03"], s["B02"]])) / white, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def rings(geom):
    for p in shapely.get_parts(geom):
        for ring in [p.exterior, *p.interiors]:
            yield np.asarray(ring.coords)


plt = figure_style()
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

W, H, SCALE = 150.0, 95.0, 20000
LEFT, TOP, MAP_W, MAP_H = 11.0, 1.5, 137.5, 80.5                       # mm
VIEW = (2678840, 1140140, 2678840 + MAP_W * SCALE / 1e3, 1140140 + MAP_H * SCALE / 1e3)   # m
X21 = H21["xllcorner"] + H21["cellsize"] * (np.arange(round(H21["ncols"])) + 0.5)
Y21 = H21["yllcorner"] + H21["cellsize"] * (H21["nrows"] - np.arange(round(H21["nrows"])) - 0.5)
STYLES = {"ours": dict(color=ACCENT, lw=0.9, label="September 2026, ours"),
          "2024": dict(color=EPOCH_2024, lw=0.6, label="2024, swissTLM3D"),
          "2021": dict(color=GLACIER, lw=0.6, ls=(0, (4, 2)), label="2021, course mask")}
CASE = {k: [patheffects.Stroke(linewidth=v["lw"] + (0.8 if k == "ours" else 0.5), foreground="white",
                               alpha=0.85 if k == "ours" else 0.6), patheffects.Normal()] for k, v in STYLES.items()}
HALO = [patheffects.withStroke(linewidth=1.0, foreground=(0, 0, 0, 0.55))]

fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white", dpi=300)
ax = fig.add_axes([LEFT / W, 1 - (TOP + MAP_H) / H, MAP_W / W, MAP_H / H])
ax.imshow(true_colour(res[MAIN]["bands"]), extent=[E0 / 1e3, E1 / 1e3, N0 / 1e3, N1 / 1e3], interpolation="lanczos")
ax.contour(X21 / 1e3, Y21 / 1e3, M21.astype(float), [0.5], colors=GLACIER, linewidths=STYLES["2021"]["lw"],
           linestyles=[STYLES["2021"]["ls"]]).set_path_effects(CASE["2021"])
for key, geom in (("2024", G24), ("ours", main)):
    for xy in rings(geom):
        ax.plot(xy[:, 0] / 1e3, xy[:, 1] / 1e3, color=STYLES[key]["color"], lw=STYLES[key]["lw"],
                solid_joinstyle="round", path_effects=CASE[key])
ax.set_xlim(VIEW[0] / 1e3, VIEW[2] / 1e3), ax.set_ylim(VIEW[1] / 1e3, VIEW[3] / 1e3)
ax.spines[:].set_visible(True)
ax.set_xticks([2679, 2680, 2681]), ax.set_yticks([1140.5, 1141, 1141.5])
ax.set_xlabel("East, Swiss coordinates (km)", labelpad=1.5)
ax.set_ylabel("North, Swiss coordinates (km)", labelpad=2)
x0, y0, h = (VIEW[2] - 640) / 1e3, (VIEW[1] + 80) / 1e3, 1.4 * SCALE / 1e6     # scale bar, 0-500 m, km
for i, colour in enumerate(("white", "#1a1a1a")):
    ax.add_patch(plt.Rectangle((x0 + 0.25 * i, y0), 0.25, h, facecolor=colour, edgecolor="#1a1a1a", lw=0.4))
for d, txt in ((0, "0"), (0.25, "250"), (0.5, "500 m")):
    ax.text(x0 + d, y0 + h + 0.02, txt, ha="left" if d == 0.5 else "center", va="bottom", fontsize=NUM,
            color="white", path_effects=HALO)
xa, ya = (VIEW[2] - 140) / 1e3, (VIEW[3] - 240) / 1e3                       # grid-north arrow, km
ax.annotate("", xy=(xa, ya + 0.11), xytext=(xa, ya), arrowprops=dict(
    arrowstyle="-|>,head_length=0.45,head_width=0.2", color="white", lw=0.8, shrinkA=0, shrinkB=0,
    path_effects=HALO))
ax.text(xa, ya + 0.125, "North", ha="center", va="bottom", fontsize=LAB, fontweight=600, color="white",
        path_effects=HALO)
fig.legend(handles=[Line2D([], [], **STYLES[k]) for k in ("ours", "2024", "2021")], loc="lower left",
           bbox_to_anchor=(LEFT / W, 0.6 / H), ncols=3, frameon=False, fontsize=LAB, handlelength=2.2,
           columnspacing=2.0, borderaxespad=0, borderpad=0, handletextpad=0.6)
save(fig, "outline_2026_si", "outline_2026.py")


# --- check images at full resolution (not for the report)
def check(name, box, scale):
    """Sentinel-2 of 27 September (10 m pixels) and SWISSIMAGE 2024 (1 m), outlines and cast shadow."""
    e0, n0, e1, n1 = box
    f, axs = plt.subplots(1, 2, figsize=(2 * (e1 - e0) / PIXEL * scale / 100, (n1 - n0) / PIXEL * scale / 100), dpi=100)
    f.subplots_adjust(0, 0, 1, 1, 0.01, 0)
    axs[0].imshow(true_colour(res[MAIN]["bands"]), extent=[E0, E1, N0, N1], interpolation="nearest")
    axs[1].imshow(np.moveaxis(read(ROOT / "data/ortho_1m.tif", box).astype(np.uint8), 0, -1), extent=[e0, e1, n0, n1])
    for a in axs:
        a.contour(X, Y, shade_main.astype(float), [0.5], colors="#ffd400", linewidths=0.8, linestyles=":")
        a.contour(X21, Y21, M21.astype(float), [0.5], colors="#00a0ff", linewidths=0.8, linestyles="--")
        for geom, colour, lw in ([(res[t]["glacier"], "#40e0d0", 0.5) for t in LATE]
                                 + [(G24, "#ff8c00", 0.9), (main, "#ff1010", 1.0)]):
            for xy in rings(geom):
                a.plot(xy[:, 0], xy[:, 1], color=colour, lw=lw)
        a.set_xlim(e0, e1), a.set_ylim(n0, n1), a.set_axis_off()
    f.savefig(PREVIEW / f"outline_2026_check_{name}.png")
    plt.close(f)


for name, box, scale in (("overview", (2678900, 1140300, 2681500, 1141600), 4),
                         ("tongue", (2680500, 1140340, 2681500, 1141100), 5),
                         ("upper", (2678900, 1140300, 2680100, 1140900), 5),
                         ("north", (2679400, 1140800, 2680800, 1141560), 5),
                         ("west", (2678900, 1140600, 2679900, 1141300), 5)):
    check(name, box, scale)
