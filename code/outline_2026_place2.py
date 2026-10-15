"""Check of place 2 of the September 2026 outline: an ice block on the northern margin, apart from the glacier body,
that the outline drops (author's screenshot of the 2026-10-06 review: a crop of tex/figures/outline_2026_si.pdf,
E 2680050-2680760, N 1140900-1141280 by matching it on the full figure). Not part of the report chain.

Status: written for and run with the outline_2026.py of commit 10a3d87 (MIN_HOLE; the keys 'body' and 'ndsi' of
outline()). It does not run on the outline_2026.py of the rule change of 2026-10-06 (joins, votes, MIN_PART), and its
outputs tex/preview/outline_2026_place2_*.png and _stdout.txt show the 10a3d87 outline.

Method: that of code/outline_2026.py; this script executes its part above '# --- the outline and its checks'
(constants, readers, cast shadow, Otsu, iso-line polygons, outline()) and keeps every iso-line polygon of a scene, not
only the largest (the glacier body). The block is the largest polygon of 27 September apart from the body inside the
screenshot. Per scene (the five of late September 2026, 12 September and 13 August 2026, 7 September 2025,
10 September 2024):
  the polygons at the block, their area, whether the scene's body reaches the block, the gap between them;
  the saddle: the highest level of snow index minus threshold at which a path of edge-neighbouring 10 m pixels joins
    the block to the glacier body of 27 September (pixels more than 100 m from the block);
  median reflectances of the block's pixels and their scene classification (Sen2Cor, 20 m; 11 = snow or ice).
On 27 September the same reflectances for references in the 300 m box: sunlit pixels of the glacier body; the strip
around the block (inside the 2024 outline, below the threshold now); the rock outcrops of the 2024 outline (holes);
the terrain more than 20 m outside the 2024 outline.
Ice: the thickness model of Grab et al. (2021) for 2012 plus the surface change 2012 to 30 September 2021 (their
swissALTI3D release 2019 at 10 m against data/dem_0p5m.tif averaged to 10 m), the bed held fixed; then the GLAMOS
balances 2021/22-2024/25 of the 100 m elevation band (ice at 900 kg m-3) to September 2025. At the block, in the strip
between block and body (and along the line joining them), and on the 2024 rock outcrops (a check of the model there).
Other cases: every polygon of 27 September apart from the body, and the detached parts of the 2024 outline, with the
late-September scenes that show ice there and whether the body reaches it. Rule test: per scene, the polygons apart
from the body of MIN_HOLE (1000 m2) or more that lie at least half inside the 2024 outline, and the area they add.

Inputs: those of outline_2026.py, plus data/ortho_0p25m.tif (SWISSIMAGE 2024, 21 and 29 August), data/dem_0p5m.tif
(swissALTI3D, glacier surface of 30 September 2021), data/thickness/C14-10_{IceThickness,SwissALTI3D_r2019}.tif,
data/glamos_basodino_massbalance_observation_elevationbins.csv, data/vectors.gpkg (sgi_2016) and the scene
classification files of data/satellite_archive/sentinel2 (catalog.csv file_scl).
Outputs (not for the report): tex/preview/outline_2026_place2_{true_colour,snow_index,pixels,ortho_relief,
other_cases}.png;
stdout, the numbers.

usage, on s02 like outline_2026.py:
  rsync -a code/common.py code/outline_2026.py code/outline_2026_place2.py \
      s02:/data/befahl/projects/AGL/basodino/report/code/
  ssh s02 'source ~/miniconda3/bin/activate agl && cd /data/befahl/projects/AGL/basodino/report &&
           ~/.local/bin/nrun python code/outline_2026_place2.py > tex/preview/outline_2026_place2_stdout.txt'
  then copy tex/preview/outline_2026_place2_* back
"""
from pathlib import Path

exec((Path(__file__).resolve().parent / "outline_2026.py").read_text(encoding="utf-8")
     .split("# --- the outline and its checks")[0])                 # noqa: S102 (the method, unchanged)

from scipy.ndimage import label  # noqa: E402
from shapely.ops import nearest_points  # noqa: E402

SHOT = shapely.box(2680050, 1140900, 2680760, 1141280)               # the author's screenshot, LV95 m
SCENES = [MAIN] + LATE + HIGH_SUN + [YEAR_BEFORE, CHECK_2024]
NAMES = {"S2A": "Sentinel-2A", "S2B": "Sentinel-2B", "S2C": "Sentinel-2C"}
MONTHS = {"08": "August", "09": "September"}
REGION, INSIDE = raster(G24.buffer(BUFFER)), raster(G24)
RHO_ICE = 900.0                                                      # kg m-3


def title(tag):
    y, m, d = tag[:10].split("-")
    return f"{int(d)} {MONTHS[m]} {y}, {NAMES[tag[11:14]]}, orbit {int(tag[16:])}"


def ice(tag):
    """outline() of a scene with every iso-line polygon kept: z = snow index minus threshold as in outline()."""
    r = outline(tag)
    b = r["bands"]
    ndsi = (b["B03"] - b["B11"]) / (b["B03"] + b["B11"])
    z = ndsi - np.where(r["shade"] & np.isfinite(r["t_shade"]), r["t_shade"], r["t_sun"])
    z[~REGION | (r["shade"] & ~INSIDE & np.isfinite(r["t_shade"]))] = -1.0
    return dict(r, ndsi=ndsi, z=z, parts=sorted(polygons(z), key=lambda p: -p.area))


def saddle(z, a, b):
    """Highest level L at which pixels a and pixels b share a 4-connected region of z > L."""
    for level in np.arange(np.max(z[a]) - 1e-6, -1.0, -0.005):
        lab = label(z > level)[0]
        if np.intersect1d(lab[a & (z > level)], lab[b & (z > level)]).size:
            return level
    return -1.0


S = {t: ice(t) for t in SCENES}
M = S[MAIN]
BODY = M["parts"][0]
BLOCK = max((p for p in M["parts"][1:] if p.intersects(SHOT)), key=lambda p: p.area)
cx, cy = round(BLOCK.centroid.x / 10) * 10, round(BLOCK.centroid.y / 10) * 10
BOX = (cx - 150, cy - 150, cx + 150, cy + 150)                       # 300 m, edges on the 10 m grid
COL, ROW = (X > BOX[0]) & (X < BOX[2]), (Y > BOX[1]) & (Y < BOX[3])
BPX = raster(BLOCK)
FAR = raster(BODY) & ~raster(BLOCK.buffer(100.0)) & (M["z"] > 0)     # the glacier body away from the block
INBOX = np.zeros_like(BPX)
INBOX[np.ix_(ROW, COL)] = True
print(f"block (27 September): {BLOCK.area:.0f} m2, {BPX.sum()} pixel centres, centroid E {BLOCK.centroid.x:.0f} "
      f"N {BLOCK.centroid.y:.0f}, extent E {BLOCK.bounds[0]:.0f}-{BLOCK.bounds[2]:.0f} N {BLOCK.bounds[1]:.0f}-"
      f"{BLOCK.bounds[3]:.0f}, gap to the body {BLOCK.distance(BODY):.1f} m; box {BOX}")
print(f"  inside: swissTLM3D 2024 {BLOCK.intersection(G24).area / BLOCK.area:.2f}, course mask 2021 "
      f"{BLOCK.intersection(G21).area / BLOCK.area:.2f}, SGI 2016 ", end="")
SGI16 = gpd.read_file(ROOT / "data/vectors.gpkg", layer="sgi_2016").query("`sgi-id` == 'C14-10'").union_all()
print(f"{BLOCK.intersection(SGI16).area / BLOCK.area:.2f}; 2024 parts of C14-10: "
      + ", ".join(f"{p.area:.0f} m2 at E {p.centroid.x:.0f} N {p.centroid.y:.0f}" for p in shapely.get_parts(G24)))

# --- the block in every scene
BANDS = ["B02", "B03", "B04", "B08", "B11"]
print("\nscene  sun  t_sun  t_shade | block: pixels > threshold of n, in shadow, median and largest snow index | "
      "polygons there m2, body reaches it, gap m | saddle (snow index minus threshold, snow index) | median "
      "reflectance " + " ".join(BANDS) + " | scene classification 11 of n")
for t in SCENES:
    s = S[t]
    body = s["parts"][0]
    here = [p for p in s["parts"][1:] if p.intersects(BLOCK)]
    joined = body.intersects(BLOCK)
    gap = min((p.distance(body) for p in here), default=np.nan)
    lev = saddle(s["z"], BPX, FAR)
    thr = s["t_sun"]
    scl = np.zeros_like(BPX, dtype=np.uint8)
    if CAT[t].get("file_scl"):
        with rasterio.open(S2 / CAT[t]["file_scl"]) as src:
            win = from_bounds(E0, N0, E1, N1, src.transform).round_offsets().round_lengths()
            scl = np.repeat(np.repeat(src.read(1, window=win), 2, 0), 2, 1)
    refl = " ".join(f"{np.nanmedian(s['bands'][b][BPX]):.3f}" for b in BANDS)
    print(f"{t}  {s['bands']['sun'][0]:.1f}  {thr:.3f}  {s['t_shade']:.3f} | {(s['z'][BPX] > 0).sum()} of {BPX.sum()}, "
          f"{(s['shade'] & BPX).sum()}, {np.nanmedian(s['ndsi'][BPX]):.3f} {np.nanmax(s['ndsi'][BPX]):.3f} | "
          f"{sum(p.area for p in here):.0f}, {'yes' if joined else 'no'}, {gap:.1f} | {lev:+.3f} {thr + lev:.3f} | "
          f"{refl} | {(scl[BPX] == 11).sum()} of {BPX.sum()} "
          f"{dict(zip(*[v.tolist() for v in np.unique(scl[BPX], return_counts=True)]))}")

# --- references on 27 September in the box, sunlit
b27, lit = M["bands"], ~M["shade"]
holes24 = shapely.union_all([shapely.Polygon(h) for p in shapely.get_parts(G24) for h in p.interiors])
ANY27 = raster(shapely.union_all(M["parts"]))
REFS = {"block": BPX,
        "glacier body": raster(BODY) & (M["z"] > 0) & INBOX & lit,
        "strip, 2024 glacier below the threshold": INSIDE & ~ANY27 & INBOX & lit,
        "rock outcrops of 2024": raster(holes24) & INBOX & lit,
        "terrain 20 m or more outside 2024": ~raster(G24.buffer(20.0)) & INBOX & lit}
print("\n27 September, medians: reference, pixels, snow index, " + ", ".join(BANDS) + ", near infrared / green")
for k, px in REFS.items():
    print(f"  {k}: {px.sum()}, {np.nanmedian(M['ndsi'][px]):.3f}, "
          + ", ".join(f"{np.nanmedian(b27[b][px]):.3f}" for b in BANDS)
          + f", {np.nanmedian(b27['B08'][px] / b27['B03'][px]):.2f}")

# --- ice thickness: model for 2012, surface change to 2021, GLAMOS balances to 2025
def grid10(path):
    with rasterio.open(path) as src:
        v = src.read(1, window=from_bounds(*BOX, src.transform).round_offsets().round_lengths()).astype(float)
        return np.where(v == src.nodata, np.nan, v)


h12 = grid10(ROOT / "data/thickness/C14-10_IceThickness.tif")
s12 = grid10(ROOT / "data/thickness/C14-10_SwissALTI3D_r2019.tif")
with rasterio.open(ROOT / "data/dem_0p5m.tif") as src:
    d = src.read(1, window=from_bounds(*BOX, src.transform).round_offsets().round_lengths()).astype(float)
s21 = d.reshape(30, 20, 30, 20).mean(axis=(1, 3))
h21 = np.where(np.isfinite(h12), np.maximum(h12 + (s21 - s12), 0.0), np.nan)
zb = float(np.nanmedian(s21[BPX[np.ix_(ROW, COL)]]))
band = int(zb // 100 * 100)
loss = {}
with open(ROOT / "data/glamos_basodino_massbalance_observation_elevationbins.csv", encoding="utf-8") as f:
    for r in csv.reader(f):
        if len(r) > 10 and r[1] == "C14-10" and r[4][:4] in ("2022", "2023", "2024", "2025") and int(r[9]) == band:
            loss[r[4]] = float(r[7]) / 1000
ice_loss = -sum(loss.values()) * 1000.0 / RHO_ICE
print(f"\nGLAMOS annual balance of the band {band}-{band + 100} m (m w.e.): {loss}; ice lost to September 2025: "
      f"{ice_loss:.1f} m")
print("ice thickness (m): reference, pixels with a model value, 2012 model median, surface change 2012-2021 median, "
      "2021 median and range, 2025 median (2021 minus the band loss)")
for k, px in REFS.items():
    p = px[np.ix_(ROW, COL)] & np.isfinite(h12)
    if p.sum():
        print(f"  {k}: {p.sum()}, {np.median(h12[p]):.1f}, {np.median(s21[p] - s12[p]):.1f}, {np.median(h21[p]):.1f} "
              f"({h21[p].min():.1f}-{h21[p].max():.1f}), {np.median(h21[p]) - ice_loss:.1f}")
slope = np.degrees(np.arctan(np.hypot(*np.gradient(d, 0.5)))).reshape(30, 20, 30, 20).mean(axis=(1, 3))
print(f"  block: elevation 2021 median {zb:.0f} m, slope (0.5 m model, mean over 10 m) median "
      f"{np.median(slope[BPX[np.ix_(ROW, COL)]]):.0f} deg; 27 September outline {M['body'].area:.0f} m2, the block "
      f"adds {BLOCK.area / M['body'].area:.2%}")
_, near = nearest_points(BLOCK, BODY)
line = shapely.LineString([BLOCK.centroid, near])
print(f"\nline block centroid -> nearest point of the body ({near.x:.0f}, {near.y:.0f}), every 10 m: distance m, "
      f"snow index, shortwave infrared, green, elevation 2021 m, thickness 2021 m (model)")
for dist in np.arange(0, line.length + 10, 10):
    pt = line.interpolate(min(dist, line.length))
    j, i = int((pt.x - E0) // PIXEL), int((N1 - pt.y) // PIXEL)
    jj, ii = int((pt.x - BOX[0]) // PIXEL), int((BOX[3] - pt.y) // PIXEL)
    print(f"  {dist:5.0f}  {M['ndsi'][i, j]:.3f}  {b27['B11'][i, j]:.3f}  {b27['B03'][i, j]:.3f}  {s21[ii, jj]:.0f}  "
          f"{h21[ii, jj]:.1f}")

# --- other cases: polygons of 27 September apart from the body, and the detached parts of the 2024 outline
print("\nother cases: E, N, area 27 September m2, gap m, late-September scenes with ice there (of 5), of them joined "
      "to the body, ice/joined on 12 September 2026, 13 August 2026, 7 September 2025, 10 September 2024, inside 2024")
def seen(t, p):
    """Ice of scene t on polygon p (25 m2 or more), and whether that ice is the scene's glacier body."""
    hit = [q for q in S[t]["parts"] if q.intersection(p).area >= 25]
    return bool(hit), any(q is S[t]["parts"][0] for q in hit)


parts24 = shapely.get_parts(G24)
cases = [("27 September part", p) for p in M["parts"][1:]] + [
    ("2024 detached part", p) for p in parts24 if p.area < 0.5 * max(q.area for q in parts24)]
for kind, p in cases:
    late = [seen(t, p) for t in [MAIN] + LATE]
    hs = ["/".join("yes" if v else "no" for v in seen(t, p)) for t in HIGH_SUN + [YEAR_BEFORE, CHECK_2024]]
    area27 = shapely.intersection(shapely.union_all(M["parts"]), p).area if kind.startswith("2024") else p.area
    print(f"  {kind}: {p.centroid.x:.0f} {p.centroid.y:.0f}, {area27:.0f}, {p.distance(BODY):.1f}, "
          f"{sum(v[0] for v in late)}, {sum(v[1] for v in late)}, {', '.join(hs)}, "
          f"{p.intersection(G24).area / p.area:.2f}")

print("\nrule test (keep detached polygons of MIN_HOLE or more lying at least half inside the 2024 outline): scene, "
      "outline area now m2, parts kept, their area m2, outline plus parts m2; each part: area, E, N, late-September "
      "scenes with ice there")
for t in SCENES:
    keep = [p for p in S[t]["parts"][1:] if p.area >= MIN_HOLE and p.intersection(G24).area >= 0.5 * p.area]
    add = sum(p.area for p in keep)
    print(f"  {t}: {S[t]['body'].area:.0f}, {len(keep)}, {add:.0f}, {S[t]['body'].area + add:.0f}; "
          + "; ".join(f"{p.area:.0f} E {p.centroid.x:.0f} N {p.centroid.y:.0f} "
                      f"{sum(seen(u, p)[0] for u in [MAIN] + LATE)}/5" for p in keep))


# --- check images
plt = figure_style()


def true_colour(s, white=0.6):
    x = np.clip(np.nan_to_num(np.dstack([s["B04"], s["B03"], s["B02"]])) / white, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def rings(geom):
    for p in shapely.get_parts(geom):
        for ring in [p.exterior, *p.interiors]:
            yield np.asarray(ring.coords)


def lines(a, s=None, box=BOX):
    """2024 outline orange, 27 September outline red, block of 27 September magenta, the scene's polygons
    turquoise dashed, its cast shadow yellow dotted, scale bar 100 m."""
    if s is not None:
        a.contour(X, Y, s["shade"].astype(float), [0.5], colors="#ffd400", linewidths=0.8, linestyles=":")
        for xy in rings(shapely.union_all(s["parts"])):
            a.plot(xy[:, 0], xy[:, 1], color="#40e0d0", lw=0.9, ls=(0, (3, 2)))
    for geom, colour, lw in ((G24, "#ff8c00", 0.9), (M["body"], "#ff1010", 1.0), (BLOCK, "#ff00ff", 0.7)):
        for xy in rings(geom):
            a.plot(xy[:, 0], xy[:, 1], color=colour, lw=lw)
    a.plot([box[0] + 15, box[0] + 115], [box[1] + 15, box[1] + 15], color="white", lw=2)
    a.text(box[0] + 65, box[1] + 20, "100 m", color="white", ha="center", va="bottom", fontsize=7)
    a.set_xlim(box[0], box[2]), a.set_ylim(box[1], box[3]), a.set_axis_off()


KEY = ("orange: 2024 swissTLM3D; red: 27 September 2026 outline (glacier body); magenta: the block on 27 September;\n"
       "turquoise dashed: the panel's own iso-lines, every polygon; yellow dotted: the panel's cast shadow")
for name, draw in (("true_colour", lambda a, s: a.imshow(true_colour(s["bands"]), extent=[E0, E1, N0, N1],
                                                         interpolation="nearest")),
                   ("snow_index", lambda a, s: a.imshow(s["z"], extent=[E0, E1, N0, N1], interpolation="nearest",
                                                        cmap="RdBu", vmin=-0.6, vmax=0.6))):
    f, axs = plt.subplots(3, 3, figsize=(10, 10.9), dpi=100)
    f.subplots_adjust(0.01, 0.04, 0.99, 0.97, 0.03, 0.12)
    for a, t in zip(axs.ravel(), SCENES):
        s = S[t]
        draw(a, s)
        lines(a, s)
        shade = f", in shadow {s['t_shade']:.3f}" if np.isfinite(s["t_shade"]) else ""
        a.set_title(f"{title(t)}\nthreshold {s['t_sun']:.3f}{shade}", fontsize=7.5)
    f.text(0.5, 0.008, KEY + ("" if name == "true_colour" else
                             "; colour: snow index minus the threshold, -0.6 red to +0.6 blue"),
           ha="center", fontsize=6.5)
    f.savefig(PREVIEW / f"outline_2026_place2_{name}.png")
    plt.close(f)

# per-pixel values of 27 September around the block and the strip towards the body
ZOOM = (cx - 100, near.y - 40, cx + 100, cy + 60)
ZOOM = tuple(round(v / 10) * 10 for v in ZOOM)
zc, zr = (X > ZOOM[0]) & (X < ZOOM[2]), (Y > ZOOM[1]) & (Y < ZOOM[3])
f, axs = plt.subplots(1, 2, figsize=(2 * (ZOOM[2] - ZOOM[0]) / 25, (ZOOM[3] - ZOOM[1]) / 25 + 0.5), dpi=100)
f.subplots_adjust(0.005, 0.06, 0.995, 0.94, 0.02, 0)
for a, (v, cmap, lim, what) in zip(axs, ((M["ndsi"], "RdBu", (-0.6, 1.0), "snow index x 100"),
                                         (b27["B11"], "magma", (0, 0.35), "shortwave infrared reflectance x 100"))):
    a.imshow(v, extent=[E0, E1, N0, N1], interpolation="nearest", cmap=cmap, vmin=lim[0], vmax=lim[1])
    for i in np.flatnonzero(zr):
        for j in np.flatnonzero(zc):
            u = (v[i, j] - lim[0]) / (lim[1] - lim[0])           # position on the colour scale, 0-1
            dark = abs(u - 0.5) > 0.3 if cmap == "RdBu" else u < 0.55
            a.text(X[j], Y[i], f"{v[i, j] * 100:.0f}", ha="center", va="center", fontsize=6.5,
                   color="white" if dark else "black")
    lines(a, None, ZOOM)
    a.set_title(f"27 September 2026, {what}", fontsize=8)
f.text(0.5, 0.01, "orange: 2024 swissTLM3D; red: 27 September outline; magenta: the block; threshold 0.335 (33 on the "
       "left)", ha="center", fontsize=7)
f.savefig(PREVIEW / "outline_2026_place2_pixels.png")
plt.close(f)

# SWISSIMAGE 2024, relief of the 2021 surface, ice thickness 2021 (model), 27 September true colour
ortho = np.moveaxis(read(ROOT / "data/ortho_0p25m.tif", BOX).astype(np.uint8), 0, -1)
gy, gx = np.gradient(d, 0.5)                                 # d/d(row, southwards), d/d(column, eastwards)
az, alt = np.radians(315), np.radians(45)
shade_rel = (-gx * np.sin(az) * np.cos(alt) + gy * np.cos(az) * np.cos(alt) + np.sin(alt)) / np.sqrt(1 + gx**2 + gy**2)
f, axs = plt.subplots(2, 2, figsize=(10, 10.6), dpi=100)
f.subplots_adjust(0.01, 0.04, 0.99, 0.97, 0.03, 0.07)
ext = [BOX[0], BOX[2], BOX[1], BOX[3]]
axs[0, 0].imshow(ortho, extent=ext)
axs[0, 1].imshow(np.clip(shade_rel, 0, 1), extent=ext, cmap="gray", vmin=0, vmax=1)
cs = axs[0, 1].contour(np.linspace(BOX[0] + 0.25, BOX[2] - 0.25, d.shape[1]),
                       np.linspace(BOX[3] - 0.25, BOX[1] + 0.25, d.shape[0]), d,
                       np.arange(np.floor(np.nanmin(d) / 5) * 5, np.nanmax(d), 5), colors="#3050ff", linewidths=0.4)
axs[0, 1].clabel(cs, cs.levels[::4], fontsize=6, fmt="%.0f")
im = axs[1, 0].imshow(h21, extent=ext, cmap="viridis", vmin=0, vmax=25, interpolation="nearest")
for i in range(30):
    for j in range(30):
        if np.isfinite(h21[i, j]) and i % 2 == 0 and j % 2 == 0:
            axs[1, 0].text(BOX[0] + 10 * j + 5, BOX[3] - 10 * i - 5, f"{h21[i, j]:.0f}", fontsize=5.5, ha="center",
                           va="center", color="white" if h21[i, j] < 12 else "black")
axs[1, 1].imshow(true_colour(b27), extent=[E0, E1, N0, N1], interpolation="nearest")
for a, ttl in zip(axs.ravel(), ("SWISSIMAGE 2024 (21 and 29 August 2024), 0.25 m",
                                "relief of the 0.5 m elevation model (glacier surface 30 September 2021), "
                                "contours every 5 m",
                                "ice thickness on 30 September 2021 (m): 2012 model plus surface change, every second "
                                "cell labelled",
                                "27 September 2026, Sentinel-2C, 10 m pixels")):
    lines(a, M if a is axs[1, 1] else None)
    a.set_title(ttl, fontsize=7.5)
f.text(0.5, 0.012, KEY, ha="center", fontsize=6.5)
f.savefig(PREVIEW / "outline_2026_place2_ortho_relief.png")
plt.close(f)

# the other cases: 200 m around each, 27 September true colour and snow index, 13 August, 2025, SWISSIMAGE 2024
CASES = [(kind, p) for kind, p in cases if p.area >= 200 and p is not BLOCK]
COLS = [("27 September 2026", MAIN, "rgb"), ("27 September, snow index", MAIN, "z"), ("13 August 2026", HIGH_SUN[1], "rgb"),
        ("7 September 2025", YEAR_BEFORE, "rgb"), ("SWISSIMAGE 2024", None, "ortho")]
f, axs = plt.subplots(len(CASES), len(COLS), figsize=(2.3 * len(COLS), 2.45 * len(CASES)), dpi=110, squeeze=False)
f.subplots_adjust(0.005, 0.03, 0.995, 0.97, 0.03, 0.2)
for row, (kind, p) in zip(axs, CASES):
    ex, ny = round(p.centroid.x / 10) * 10, round(p.centroid.y / 10) * 10
    box = (ex - 100, ny - 100, ex + 100, ny + 100)
    for a, (ttl, t, what) in zip(row, COLS):
        if what == "ortho":
            a.imshow(np.moveaxis(read(ROOT / "data/ortho_0p25m.tif", box).astype(np.uint8), 0, -1),
                     extent=[box[0], box[2], box[1], box[3]])
        elif what == "z":
            a.imshow(S[t]["z"], extent=[E0, E1, N0, N1], interpolation="nearest", cmap="RdBu", vmin=-0.6, vmax=0.6)
        else:
            a.imshow(true_colour(S[t]["bands"]), extent=[E0, E1, N0, N1], interpolation="nearest")
        lines(a, S[t] if t else None, box)
        for xy in rings(p):
            a.plot(xy[:, 0], xy[:, 1], color="#ff00ff", lw=0.8)
        a.set_title(f"{ttl}\n{kind}, E {p.centroid.x:.0f}, N {p.centroid.y:.0f}" if a is row[0] else ttl, fontsize=6.5)
f.text(0.5, 0.005, "magenta: the case; orange: 2024 swissTLM3D; red: 27 September outline; turquoise dashed: the panel's "
       "iso-lines; colour of the snow index: minus the threshold, -0.6 red to +0.6 blue", ha="center", fontsize=6)
f.savefig(PREVIEW / "outline_2026_place2_other_cases.png")
plt.close(f)
