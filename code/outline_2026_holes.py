"""The dark spot inside the ice of the September 2026 outline (place 1 of the author's review, 2026-10-06, near
E 2680680, N 1140890) and every other hole that the 1000 m2 rule of code/outline_2026.py (MIN_HOLE) fills.
A check of outline_2026.py, not part of the report chain.

Status: written for and run with the outline_2026.py of commit 10a3d87 (MIN_HOLE; the keys 'body' and 'ndsi' of
outline()). It does not run on the outline_2026.py of the rule change of 2026-10-06 (joins, votes, MIN_PART), and its
images tex/preview/outline_2026_place1_*.png show the 10a3d87 outline.

Method: steps 1-5 of outline_2026.py run unchanged (its code above "# --- the outline and its checks" is executed
here and writes nothing), with MIN_HOLE set to 0, so that the glacier body of every scene keeps all its holes.
Inputs: those of outline_2026.py, and data/ortho_0p25m.tif (SWISSIMAGE 2024), data/dem_0p5m.tif (swissALTI3D, glacier
surface of 30 September 2021), data/thickness/ (Grab et al. 2021: bed and thickness under the surface of 2012; radar
points of 2006), SWISSIMAGE 10 cm of 2018, 2021 and 2024 over the central 100 m, read from the swisstopo STAC
(tile 2680-1140; image strips over the spot of 20 and 25 September 2018, 30 September 2021, 29 August 2024).

stdout
  1. per scene (the five late-September 2026 scenes, 12 September and 13 August 2026, 7 September 2025, 10 September
     2024, and clear late-summer scenes of 2021 to 2025): thresholds; holes of the iso-line within 30 m of the spot;
     the index of the 10 x 10 pixels (100 m) around the spot, * below the threshold (27 September also B11);
     reflectance of the darkest pixel, of the sunlit ice 60-100 m around the spot and of the sunlit rock of the
     large outcrop (hole of 2 ha at E 2679807, N 1141055), with B08/B03 (open water lowers it below that of ice);
     the rock-equivalent area within 30 and 50 m of the spot by linear unmixing of B11 between that ice and that rock
     (in B11 ice, water and shadow are all dark, rock and debris bright; shaded rock counts less, so a lower bound)
  2. every hole of the five late-September bodies, grouped by place: area per scene, number of scenes showing it,
     lowest index on 27 September, distance to the outer margin; what candidate rules keep
  3. the 0.5 m elevation model at the spot: slope and direct sunlight of the 20 m smoothed surface, relief (surface
     minus its 20 m Gaussian smoothing); bed and thickness of Grab et al. (2021) and the nearest radar point; the
     GLAMOS balance of the 2800-2900 m band since the surface of 2021
tex/preview/outline_2026_place1_truecolour.png, _index.png   the 300 m box around the spot in every scene, 10 m
                     pixels: true colour (white ticks on the frame point at the spot); index minus the scene's
                     threshold (-0.6 to 0.6) with the scene's iso-line (black, every hole) and the 27 September outline
                     (red)
tex/preview/outline_2026_place1_aerial.png   SWISSIMAGE 2018, 2021, 2024 of the central 100 m with the 10 m grid of
                     Sentinel-2 (yellow) and the 27 September iso-line (red); SWISSIMAGE 2024 of the 300 m box; the 0.5 m
                     elevation model of 2021 as shading and as relief (-1 to +1 m)
tex/preview/outline_2026_place1_holes.png    27 September true colour of the glacier, every hole of the five
                     late-September scenes numbered as in table 2 (yellow: in 3 or more scenes)

usage, on s02 from the mirror of this tree (as outline_2026.py; under a minute):
  rsync -a code/common.py code/outline_2026.py code/outline_2026_holes.py \
      s02:/data/befahl/projects/AGL/basodino/report/code/
  ssh s02 'source ~/miniconda3/bin/activate agl && cd /data/befahl/projects/AGL/basodino/report &&
           python code/outline_2026_holes.py'
  then copy tex/preview/outline_2026_place1_*.png back
"""
from pathlib import Path

exec((Path(__file__).resolve().parent / "outline_2026.py").read_text(encoding="utf-8")
     .split("# --- the outline and its checks")[0])                   # noqa: S102 (the method, unchanged)

from scipy.ndimage import gaussian_filter  # noqa: E402

from common import RHO_ICE  # noqa: E402

MIN_HOLE = 0.0                    # read by outline() at call time: every hole of the glacier body is kept
HISTORY = ["2025-08-26_S2C_R108", "2024-08-31_S2A_R108", "2023-09-08_S2B_R065", "2022-08-24_S2B_R065", CHECK_2021]
SCENES = [MAIN] + LATE + HIGH_SUN + [YEAR_BEFORE, CHECK_2024] + HISTORY
SPOT = (2680680, 1140890)                                # the dark patch of the screenshot, a pixel corner
BOX = (SPOT[0] - 150, SPOT[1] - 150, SPOT[0] + 150, SPOT[1] + 150)
NEAR = 30.0                                              # m around the spot: holes and unmixing
OUTCROP = shapely.Point(2679807, 1141055)                # inside the 2 ha hole of 27 September: sunlit rock
BANDS = ("B02", "B03", "B04", "B08", "B11")
STAC = ("https://data.geo.admin.ch/ch.swisstopo.swissimage-dop10/swissimage-dop10_{y}_2680-1140/"
        "swissimage-dop10_{y}_2680-1140_0.1_2056.tif")
FLIGHT = {2018: "20 or 25 September 2018", 2021: "30 September 2021", 2024: "29 August 2024"}

res = {t: outline(t, G21 if t == CHECK_2021 else G24) for t in SCENES}
for r in res.values():
    s = r["bands"]
    r["ndsi"] = (s["B03"] - s["B11"]) / (s["B03"] + s["B11"])
    r["z"] = r["ndsi"] - np.where(r["shade"] & np.isfinite(r["t_shade"]), r["t_shade"], r["t_sun"])
    r["holes"] = [shapely.Polygon(h) for h in r["body"].interiors]
spot = shapely.Point(SPOT)
c0, r0 = round((SPOT[0] - E0) / PIXEL) - 5, round((N1 - SPOT[1]) / PIXEL) - 5     # 10 x 10 pixels around the spot
W = np.s_[r0:r0 + 10, c0:c0 + 10]
ZONE, ANNULUS = raster(spot.buffer(NEAR)), raster(spot.buffer(100)) & ~raster(spot.buffer(60))
OUTCROP_PX = raster(OUTCROP.buffer(60))
ROCK11 = {}                       # B11 of the sunlit outcrop rock, scenes with 20 or more such pixels
for t, r in res.items():
    m = OUTCROP_PX & ~r["shade"] & (r["z"] < -0.1)
    if m.sum() >= 20:
        ROCK11[t] = float(np.median(r["bands"]["B11"][m]))
ROCK11_2026 = float(np.median([v for t, v in ROCK11.items() if t.startswith("2026")]))


def name(tag):
    d = np.datetime64(tag[:10]).astype(object)
    return f"{d.day} {d:%B} {d.year}, Sentinel-{tag[12:14]}, orbit {tag[-3:]}"


# --- 1. the spot in every scene
print(f"1. THE SPOT, scene by scene (pixels within {NEAR:.0f} m: {ZONE.sum()})")
for t in SCENES:
    r, s = res[t], res[t]["bands"]
    nd, zz = r["ndsi"][W], r["z"][W]
    near = [h for h in r["holes"] if h.distance(spot) <= NEAR]
    print(f"\n{name(t)}: sun {s['sun'][0]:.1f} deg, threshold sunlit {r['t_sun']:.3f}, in shadow {r['t_shade']:.3f};"
          f" pixels in cast shadow within {NEAR:.0f} m: {int((ZONE & r['shade']).sum())}")
    print(f"  holes within {NEAR:.0f} m: " + (", ".join(f"{h.area:.0f} m2 at {h.centroid.x:.0f} {h.centroid.y:.0f}"
                                                for h in near) or "none")
          + f"; 100 m window: lowest index {np.nanmin(nd):.3f}, {int((zz < 0).sum())} pixels below the threshold")
    print(f"  index x 100 ({'E':s} of the pixel centres across, N down)\n  {'':9s}" + "".join(f"{e:7.0f}" for e in X[c0:c0 + 10]))
    for i in range(10):
        print(f"  {Y[r0 + i]:9.0f}" + "".join(f"{100 * nd[i, j]:6.0f}{'*' if zz[i, j] < 0 else ' '}" for j in range(10)))
    if t == MAIN:
        print("  B11 x 1000 (shortwave infrared: ice about 20, sunlit rock about 290, water about 10)")
        for i in range(10):
            print(f"  {Y[r0 + i]:9.0f}" + "".join(f"{1000 * s['B11'][W][i, j]:7.0f}" for j in range(10)))
    ice = ANNULUS & raster(r["body"]) & ~r["shade"] & (r["z"] > 0)
    rock = OUTCROP_PX & ~r["shade"] & (r["z"] < -0.1)
    k = np.nanargmin(np.where(ZONE, r["ndsi"], np.nan))
    dark = np.zeros_like(ZONE)
    dark.flat[k] = True
    for label, m in ((f"darkest {X[k % len(X)]:.0f} {Y[k // len(X)]:.0f}", dark), ("ice 60-100 m around", ice),
                     ("outcrop rock", rock)):
        med = {b: np.nanmedian(s[b][m]) if m.any() else np.nan for b in BANDS}
        print(f"  {label:24s} {int(m.sum()):4d} px  index {np.nanmedian(r['ndsi'][m]) if m.any() else np.nan:6.3f}  "
              + "  ".join(f"{b} {med[b]:.3f}" for b in BANDS) + f"  B08/B03 {med['B08'] / med['B03']:.2f}")
    i11, r11 = float(np.median(s["B11"][ice])), ROCK11.get(t, ROCK11_2026)
    print(f"  rock-equivalent area by unmixing B11 between that ice ({i11:.3f}) and sunlit rock ({r11:.3f}"
          f"{'' if t in ROCK11 else ', median of the 2026 scenes'}): " + ", ".join(
              f"{PIXEL ** 2 * ((s['B11'][m] - i11) / (r11 - i11)).sum():.0f} m2 within {d:.0f} m"
              for d, m in ((d, raster(spot.buffer(d)) & ~r["shade"]) for d in (NEAR, 50.0))))


# --- 2. every hole of the five late-September bodies
LATE5 = [MAIN] + LATE
print("\n2. HOLES OF THE FIVE LATE-SEPTEMBER BODIES, grouped by place (m2; scenes in the order of the columns)")
allholes = [(i, h) for i, t in enumerate(LATE5) for h in res[t]["holes"]]
groups = sorted(shapely.get_parts(shapely.union_all([h.buffer(5) for _, h in allholes])),
                key=lambda g: (g.centroid.x, g.centroid.y))
table = []
for g in groups:
    area = [sum(h.area for i, h in allholes if i == k and h.intersects(g)) for k in range(len(LATE5))]
    m = raster(g)
    table.append(dict(g=g, area=area, n=sum(a > 0 for a in area),
                      low=float(np.nanmin(res[MAIN]["ndsi"][m])) if m.any() else np.nan,
                      shaded=bool(res[MAIN]["shade"][m].any()), margin=res[MAIN]["body"].exterior.distance(g)))
print(" no  centre E  centre N  " + "".join(f"{t[5:10]}{t[12:14]:>4s} " for t in LATE5)
      + " scenes  lowest index 27 Sep  shadow  to outer margin m")
for k, d in enumerate(table, 1):
    print(f"{k:3d}  {d['g'].centroid.x:8.0f}  {d['g'].centroid.y:8.0f}  " + "".join(f"{a:9.0f} " for a in d["area"])
          + f"  {d['n']:5d}  {d['low']:19.3f}  {'yes' if d['shaded'] else 'no':>6s}  {d['margin']:8.0f}")
RULES = {"now: 1000 m2 or more on 27 September": lambda d: d["area"][0] >= 1000,
         "any size, in 3 or more of the 5 scenes": lambda d: d["n"] >= 3,
         "1000 m2, or in 3 or more of the 5 scenes": lambda d: d["area"][0] >= 1000 or d["n"] >= 3,
         "200 m2 or more on 27 September": lambda d: d["area"][0] >= 200}
for label, rule in RULES.items():
    kept = [k for k, d in enumerate(table, 1) if d["area"][0] > 0 and rule(d)]
    print(f"  {label:42s} keeps {len(kept):2d} holes of 27 September, "
          f"{sum(table[k - 1]['area'][0] for k in kept):6.0f} m2; numbers {kept}")


# --- 3. elevation, bed and thickness at the spot
print("\n3. ELEVATION MODEL, BED AND THICKNESS AT THE SPOT")
pad = 100                                                  # m, removed again after the smoothing
dem = read(ROOT / "data/dem_0p5m.tif", (BOX[0] - pad, BOX[1] - pad, BOX[2] + pad, BOX[3] + pad), band=1)
dem = gaussian_filter(dem, 2)                              # sigma 1 m: the striping of the 0.5 m model over ice
smooth = gaussian_filter(dem, 40)                          # sigma 20 m
relief = (dem - smooth)[2 * pad:-2 * pad, 2 * pad:-2 * pad]
dem, smooth = dem[2 * pad:-2 * pad, 2 * pad:-2 * pad], smooth[2 * pad:-2 * pad, 2 * pad:-2 * pad]
gy, gx = np.gradient(dem, 0.5)                             # per m: towards south (rows), towards east (columns)
sy, sx = np.gradient(smooth, 0.5)
ri, ci = round((BOX[3] - SPOT[1]) / 0.5), round((SPOT[0] - BOX[0]) / 0.5)
slope, aspect = np.arctan(np.hypot(sx[ri, ci], sy[ri, ci])), np.arctan2(-sx[ri, ci], sy[ri, ci])  # aspect: downhill
e, a = np.radians(res[MAIN]["bands"]["sun"])
light = np.cos(slope) * np.sin(e) + np.sin(slope) * np.cos(e) * np.cos(a - aspect)
print(f"surface of 30 September 2021 at the spot {dem[ri, ci]:.1f} m; the 20 m smoothed surface slopes "
      f"{np.degrees(slope):.0f} deg towards {np.degrees(aspect) % 360:.0f} deg from north: direct sunlight on 27 September "
      f"{light / np.sin(e):.2f} of that on level ground")
w = np.s_[ri - 60:ri + 60, ci - 60:ci + 60]                # 30 m around the spot
lo = np.unravel_index(relief[w].argmin(), relief[w].shape)
print(f"relief at the spot {relief[ri, ci]:+.2f} m; within 30 m lowest {relief[w].min():+.2f} m at "
      f"E {BOX[0] + 0.5 * (ci - 60 + lo[1]):.0f} N {BOX[3] - 0.5 * (ri - 60 + lo[0]):.0f}, highest {relief[w].max():+.2f} m")
TH = ROOT / "data/thickness"
cells = {}
for f in ("C14-10_SwissALTI3D_r2019.tif", "C14-10_GlacierBed.tif", "C14-10_IceThickness.tif"):
    with rasterio.open(TH / f) as src:
        v = src.read(1, window=from_bounds(SPOT[0] - 50, SPOT[1] - 50, SPOT[0] + 50, SPOT[1] + 50, src.transform))
        cells[f] = np.where(v == src.nodata, np.nan, v)
    print(f"{f}: the 4 cells at the spot {np.round(cells[f][4:6, 4:6].ravel(), 1).tolist()}; "
          f"100 m window {np.nanmin(cells[f]):.1f} to {np.nanmax(cells[f]):.1f}")
d21 = read(ROOT / "data/dem_2m.tif", (SPOT[0] - 50, SPOT[1] - 50, SPOT[0] + 50, SPOT[1] + 50), band=1)
d21 = d21.reshape(10, 5, 10, 5).mean(axis=(1, 3))
h21 = d21 - cells["C14-10_GlacierBed.tif"]
lo = np.unravel_index(np.nanargmin(h21), h21.shape)
print(f"thickness 2021 = surface 2021 - bed, 4 cells {np.round(h21[4:6, 4:6].ravel(), 1).tolist()} m; 100 m window "
      f"{np.nanmin(h21):.1f} m (cell centre E {SPOT[0] - 45 + 10 * lo[1]:.0f} N {SPOT[1] + 45 - 10 * lo[0]:.0f}) to "
      f"{np.nanmax(h21):.1f} m; surface 2021 - 2012 at the 4 cells "
      f"{np.round((d21 - cells['C14-10_SwissALTI3D_r2019.tif'])[4:6, 4:6].ravel(), 1).tolist()} m")
pts = [r.split("\t") for r in open(TH / "C14-10_basodino_GlaThiDa.txt") if not r.startswith("#")]
near = sorted((np.hypot(float(p[3]) - SPOT[0], float(p[4]) - SPOT[1]), float(p[3]), float(p[4]), float(p[8]))
              for p in pts)[0]
print(f"nearest radar point of 2006: {near[0]:.0f} m away (E {near[1]:.0f}, N {near[2]:.0f}), thickness {near[3]:.0f} m")
with open(ROOT / "data/glamos_basodino_massbalance_observation_elevationbins.csv", encoding="utf-8") as f:
    ba = {r[4]: float(r[7]) / 1000 for r in csv.reader(f)
          if len(r) > 10 and r[1] == "C14-10" and r[9] == "2800" and "2022" <= r[4][:4] <= "2025"}
print("GLAMOS annual balance of the 2800-2900 m band: " + ", ".join(f"to {d} {v:+.2f}" for d, v in ba.items())
      + f" m w.e.; sum {sum(ba.values()):+.2f} m w.e. = {1000 * sum(ba.values()) / RHO_ICE:+.1f} m of ice")


# --- previews
def true_colour(s, white=0.6):
    """sRGB encoding of reflectance / white, as outline_2026.py."""
    x = np.clip(np.nan_to_num(np.dstack([s["B04"], s["B03"], s["B02"]])) / white, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def rings(geom):
    for p in shapely.get_parts(geom):
        for ring in [p.exterior, *p.interiors]:
            yield np.asarray(ring.coords)


def ticks(a, box, colour):
    e0, n0, e1, n1 = box
    for xy in (((SPOT[0], n0), (SPOT[0], n0 + 20)), ((SPOT[0], n1), (SPOT[0], n1 - 20)),
               ((e0, SPOT[1]), (e0 + 20, SPOT[1])), ((e1, SPOT[1]), (e1 - 20, SPOT[1]))):
        a.plot(*zip(*xy), color=colour, lw=1.5)


plt = figure_style()
outline27 = gpd.read_file(OUT / f"outline_{MAIN[:10]}.gpkg").geometry.iloc[0]
for kind in ("truecolour", "index"):
    f, axs = plt.subplots(3, 5, figsize=(15, 9.6), dpi=100)
    f.subplots_adjust(0.005, 0.005, 0.995, 0.97, 0.02, 0.09)
    for a, t in zip(axs.ravel(), SCENES + [None]):
        a.set_axis_off()
        if t is None:
            continue
        r = res[t]
        if kind == "truecolour":
            a.imshow(true_colour(r["bands"]), extent=[E0, E1, N0, N1], interpolation="nearest")
            ticks(a, BOX, "white")
        else:
            a.imshow(r["z"], extent=[E0, E1, N0, N1], interpolation="nearest", cmap="RdBu", vmin=-0.6, vmax=0.6)
            for xy in rings(r["body"]):
                a.plot(xy[:, 0], xy[:, 1], color="black", lw=0.9)
            for xy in rings(outline27):
                a.plot(xy[:, 0], xy[:, 1], color="#ff1010", lw=0.8)
            ticks(a, BOX, "black")
        a.set_xlim(BOX[0], BOX[2]), a.set_ylim(BOX[1], BOX[3])
        a.set_title(name(t), fontsize=8)
    f.savefig(PREVIEW / f"outline_2026_place1_{kind}.png")
    plt.close(f)

C = (SPOT[0] - 50, SPOT[1] - 50, SPOT[0] + 50, SPOT[1] + 50)                     # the central 100 m
air = {}
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
    for y in FLIGHT:
        with rasterio.open("/vsicurl/" + STAC.format(y=y)) as src:
            air[y] = np.moveaxis(src.read((1, 2, 3), window=from_bounds(*C, src.transform)), 0, -1)
ortho = np.moveaxis(read(ROOT / "data/ortho_0p25m.tif", BOX), 0, -1)[..., :3].astype(np.uint8)
light = (-0.5, 0.5, 0.5 ** 0.5)                            # towards the light: north-west, 45 deg up
shading = (-gx * light[0] + gy * light[1] + light[2]) / np.sqrt(gx ** 2 + gy ** 2 + 1)
panels = ([(f"SWISSIMAGE, {FLIGHT[y]}, central 100 m", air[y], C, {}) for y in FLIGHT]
          + [("SWISSIMAGE, 29 August 2024, 300 m", ortho, BOX, {}),
             ("elevation model, 30 September 2021, shading", shading, BOX, dict(cmap="gray")),
             ("elevation model, 30 September 2021, relief -1 to +1 m", relief, BOX, dict(cmap="RdBu_r", vmin=-1, vmax=1))])
f, axs = plt.subplots(2, 3, figsize=(15, 10.3), dpi=100)
f.subplots_adjust(0.005, 0.005, 0.995, 0.97, 0.01, 0.05)
for a, (title, img, box, kw) in zip(axs.ravel(), panels):
    a.set_axis_off()
    a.set_title(title, fontsize=9)
    a.imshow(img, extent=[box[0], box[2], box[1], box[3]], **kw)
    a.contour(X, Y, res[MAIN]["z"], [0.0], colors="#ff1010", linewidths=0.8)
    if box == C:
        for e in np.arange(C[0], C[2] + 1, PIXEL):
            a.axvline(e, color="yellow", lw=0.3)
        for n in np.arange(C[1], C[3] + 1, PIXEL):
            a.axhline(n, color="yellow", lw=0.3)
    else:
        ticks(a, box, "yellow")
    a.set_xlim(box[0], box[2]), a.set_ylim(box[1], box[3])
f.savefig(PREVIEW / "outline_2026_place1_aerial.png")
plt.close(f)

view = (2678900, 1140300, 2681600, 1141600)
f, a = plt.subplots(figsize=(13.5, 6.5), dpi=100)
f.subplots_adjust(0, 0, 1, 1)
a.imshow(true_colour(res[MAIN]["bands"]), extent=[E0, E1, N0, N1], interpolation="nearest")
for xy in rings(outline27):
    a.plot(xy[:, 0], xy[:, 1], color="#ff1010", lw=0.6)
for k, d in enumerate(table, 1):
    colour = "yellow" if d["n"] >= 3 else "cyan"
    a.plot(d["g"].centroid.x, d["g"].centroid.y, "o", ms=9, mfc="none", mec=colour, mew=1.2)
    a.text(d["g"].centroid.x + 18, d["g"].centroid.y + 18, str(k), color=colour, fontsize=8)
a.set_xlim(view[0], view[2]), a.set_ylim(view[1], view[3]), a.set_axis_off()
f.savefig(PREVIEW / "outline_2026_place1_holes.png")
plt.close(f)
