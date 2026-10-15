"""Check of place 3 of the September 2026 outline (review of 2026-10-06): a narrow strip of ice on the northern margin
that joins the glacier body south-west of the rock bay at E 2680320, N 1140990 to the lobe north-east of it, and that
the outline cuts. A check of outline_2026.py, not part of the report chain.

Status: written for and run with the outline_2026.py of commit 10a3d87 (MIN_HOLE; the key 'body' of outline()). It
does not run on the outline_2026.py of the rule change of 2026-10-06 (joins, votes, MIN_PART), and its images
tex/preview/outline_2026_place3_*.png show the 10a3d87 outline.

Method: that of code/outline_2026.py, executed from its own source above '# --- the outline and its checks'
(constants, readers, cast shadow, Otsu thresholds, iso-line polygons, outline()); nothing of it is changed here.
Scenes: the five of late September 2026, 12 September and 13 August 2026 (higher sun), 7 September 2025 and
10 September 2024, each with its own thresholds and cast shadow; SWISSIMAGE 2024 (21 and 29 August, 0.25 m) and the
0.5 m elevation model (swissALTI3D, glacier surface of 30 September 2021): data/ortho_0p25m.tif, data/dem_0p5m.tif.

Per scene, inside CORRIDOR (south of it the two sides meet round the bay; north of it lies the block of place 2):
  join     the highest level at which pixels at or above it join the seed WEST to the seed EAST: 4-connected,
           8-connected, and by the iso-line of outline_2026.py (marching squares between pixel centres); for the snow
           index and for the green and near-infrared index GNI = (B03 - B08) / (B03 + B08), all 10 m (bare ice about
           +0.18; rock below 0, bright rock too, where B08 >= B03), GNI threshold by Otsu as the snow index's
  chain    the shortest 8-connected chain at the snow-index join of 27 September: pixels, values, length outside
           the outline; its principal axis gives the profile stations (every 10 m, 15 m clear of the ends)
  width    ice fraction of each pixel: (a) 10 m, least squares of B02 B03 B04 B08 on three end members (ice, dark
           rock, bright rock) with the fractions summing to one; (b) 20 m, (B11 rock - B11) / (B11 rock - B11 ice)
           on the B11 that outline_2026.py uses. End members: medians of the sunlit pixels with snow index above
           0.8 in BOX (ice; the top tenth if fewer than 20) and of the corridor pixels that both indices call rock
           (dark: the darker half in B03, bright: the brightest quarter; the box if fewer than 20). The integral of
           the fraction along the normal of the axis (+-25 m, bilinear between pixel centres) is the equivalent ice
           width at a station; linear resampling (the cubic projection to LV95, the bilinear B11) keeps it
  B11      27 September with B11 by nearest neighbour, on the native 20 m grid (B03 averaged 2 x 2), sharpened with
           B08 (B11 x B08 / B08 of the 20 m pixel, the test of outline_2026.py's docstring), and sharpened with
           D = B03 - B08 (B11 + k (D - D averaged over 20 m and interpolated like B11), k = (B11 ice - B11 rock) /
           (D ice - D rock) over the region's sunlit pixels with snow index above 0.8 and below 0); each with its
           own Otsu thresholds; the strip's join, the glacier area. The D-sharpened index is also run on every scene
  model    a straight strip of ice of width w through rock between two ice fields, at the axis angle and 64
           positions against the 20 m grid: B03 and B08 averaged over 10 m pixels, B11 over 20 m pixels (without
           and with a Gaussian point spread of 0.4 pixel), B11 bilinear to 10 m as outline_2026.py, end members of
           27 September (ice and all rock); join levels of the snow index and of GNI (4-connected) against the
           thresholds of 27 September
  rules    two candidate rules on every scene, against outline_2026.py: the D-sharpened snow index (above), and a
           join rule (join_rule(): 8-connected sunlit pixels with GNI >= its threshold and snow index >= the sunlit
           threshold minus 0.15, or >= 0, kept where they touch two separate parts of the ice or close a non-ice
           area of at least MIN_HOLE); glacier area, holes, the strip, mean boundary distance between the five
           late-September scenes; also the sunlit threshold of 27 September lowered to the strip's join
  surface  the 2021 surface across the strip (elevation and departure from a 15 m smoothing) on three profiles
  others   the same joins for two more places of the northern margin (case()): the gully at E 2679800,
           N 1141330 and the bay east of the strip at E 2680600, N 1141065
Outputs: tex/preview/outline_2026_place3_{true_colour,snow_index,values,ortho_relief,model,rule,gully,east_bay}.png
         (not for the report); numbers on stdout.

usage, on s02 like outline_2026.py (its docstring):
  rsync -a code/common.py code/outline_2026.py code/outline_2026_place3.py \
      s02:/data/befahl/projects/AGL/basodino/report/code/
  ssh s02 'source ~/miniconda3/bin/activate agl && cd /data/befahl/projects/AGL/basodino/report &&
           ~/.local/bin/nrun python code/outline_2026_place3.py'
  then copy tex/preview/outline_2026_place3_*.png back
"""
from pathlib import Path

exec((Path(__file__).resolve().parent / "outline_2026.py").read_text(encoding="utf-8")
     .split("# --- the outline and its checks")[0])

from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from scipy.ndimage import gaussian_filter, label  # noqa: E402

BOX = (2680170, 1140900, 2680470, 1141200)                 # 300 m around the strip, LV95 m
CORRIDOR = (2680130, 1140990, 2680470, 1141120)            # pixel centres inside count
WEST, EAST = (2680245, 1140995), (2680435, 1141075)        # pixel centres in the body either side of the bay
TAGS = [MAIN] + LATE + HIGH_SUN + [YEAR_BEFORE, CHECK_2024]
FOUR, EIGHT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), np.ones((3, 3), int)
BANDS = ("B02", "B03", "B04", "B08")


def at(p):
    return int(np.argmin(np.abs(Y - p[1]))), int(np.argmin(np.abs(X - p[0])))


def window(box):
    e0, n0, e1, n1 = box
    return np.outer((Y > n0) & (Y < n1), (X > e0) & (X < e1))


CORR, BOXM, IW, IE = window(CORRIDOR), window(BOX), at(WEST), at(EAST)
REGION = raster(G24.buffer(BUFFER))


def join_level(v, test, mask=CORR, seeds=(IW, IE)):
    """Highest value t of v in mask at which test(v_masked, t, seeds) holds (bisection; joins are monotone in t)."""
    vm = np.where(mask, np.nan_to_num(v, nan=-9.0), -9.0)
    vals = np.unique(vm[mask])[::-1]
    lo, hi = 0, len(vals) - 1
    if not test(vm, vals[hi], seeds):
        return np.nan
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if test(vm, vals[mid], seeds) else (mid + 1, hi)
    return float(vals[lo])


def pixels_join(structure):
    def test(vm, t, seeds):
        lab, _ = label(vm >= t, structure)
        return lab[seeds[0]] != 0 and lab[seeds[0]] == lab[seeds[1]]
    return test


def isoline_join(vm, t, seeds):
    w, e = (shapely.Point(X[s[1]], Y[s[0]]) for s in seeds)
    return any(p.contains(w) and p.contains(e) for p in polygons(vm - t))


def joins(v):
    return [join_level(v, test) for test in (pixels_join(FOUR), pixels_join(EIGHT), isoline_join)]


def chain(v, t):
    """Shortest 8-connected chain of corridor pixels with v >= t from WEST to EAST (breadth first)."""
    ok = CORR & (np.nan_to_num(v, nan=-9.0) >= t)
    prev, todo = {IW: None}, [IW]
    while todo:
        nxt = []
        for p in todo:
            for q in ((p[0] + a, p[1] + c) for a in (-1, 0, 1) for c in (-1, 0, 1)):
                if q not in prev and 0 <= q[0] < len(Y) and 0 <= q[1] < len(X) and ok[q]:
                    prev[q] = p
                    nxt.append(q)
        todo = nxt
    path, p = [], IE
    while p is not None:
        path.append(p)
        p = prev[p]
    return path[::-1]


def outline_of(index, shade, reference=G24):
    """Steps 4-5 of outline_2026.py's outline() for any index: Otsu thresholds, iso-line, largest polygon, holes."""
    region, inside = raster(reference.buffer(BUFFER)), raster(reference)
    t_sun = otsu(index[region & ~shade])
    t_shade = otsu(index[inside & shade]) if (inside & shade).sum() >= 100 else np.nan
    z = index - np.where(shade & np.isfinite(t_shade), t_shade, t_sun)
    z[~region | (shade & ~inside & np.isfinite(t_shade))] = -1.0
    body = max(polygons(z), key=lambda p: p.area)
    return shapely.Polygon(body.exterior, [h for h in body.interiors if shapely.Polygon(h).area >= MIN_HOLE]), t_sun


def block(a):
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean(axis=(1, 3))


def nearest(a):
    return np.repeat(np.repeat(a, 2, 0), 2, 1)


def bilinear(a):
    return zoom(a, 2, order=1, mode="nearest", grid_mode=True)       # as scene() of outline_2026.py


def ndsi(b, b11=None):
    b11 = b["B11"] if b11 is None else b11
    return (b["B03"] - b11) / (b["B03"] + b11)


def d_sharpened(b, shade):
    """Snow index with B11 + k (D - D low-passed like B11), D = B03 - B08, k from the region's ice and rock."""
    d, v = b["B03"] - b["B08"], ndsi(b)
    ice, rock = REGION & ~shade & (v > 0.8), REGION & ~shade & (v < 0.0)
    k = (np.median(b["B11"][ice]) - np.median(b["B11"][rock])) / (np.median(d[ice]) - np.median(d[rock]))
    return ndsi(b, np.maximum(b["B11"] + k * (d - bilinear(block(d))), 0.005)), float(k)


def members(b, v, g, t_gni, sun):
    """Median spectra (B02 B03 B04 B08 B11) of ice, all rock, dark rock and bright rock near the strip."""
    ice = BOXM & sun & (v > 0.8)
    if ice.sum() < 20:
        ice = BOXM & sun & (v >= np.percentile(v[BOXM & sun], 90))
    rock = CORR & sun & (v < 0.0) & (g < t_gni)
    if rock.sum() < 20:
        rock = BOXM & sun & (v < 0.0) & (g < t_gni)
    dark = rock & (b["B03"] <= np.percentile(b["B03"][rock], 50))
    bright = rock & (b["B03"] >= np.percentile(b["B03"][rock], 75))
    return {m: {k: float(np.median(b[k][mask])) for k in BANDS + ("B11",)} | {"n": int(mask.sum())}
            for m, mask in (("ice", ice), ("rock", rock), ("dark", dark), ("bright", bright))}


def ice_fraction(b, em):
    """Ice fraction per pixel: least squares of B02 B03 B04 B08 on ice, dark and bright rock, fractions sum to one."""
    a = np.vstack([np.array([[em[m][k] for k in BANDS] for m in ("ice", "dark", "bright")]).T, np.full(3, 10.0)])
    obs = np.stack([np.nan_to_num(b[k]) for k in BANDS] + [np.full(b["B03"].shape, 10.0)])
    return np.linalg.lstsq(a, obs.reshape(5, -1), rcond=None)[0][0].reshape(b["B03"].shape)


def true_colour(s, white=0.6):
    """sRGB encoding of reflectance / white, as the figure of outline_2026.py."""
    x = np.clip(np.nan_to_num(np.dstack([s["B04"], s["B03"], s["B02"]])) / white, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def rings(geom):
    for p in shapely.get_parts(geom):
        for ring in [p.exterior, *p.interiors]:
            yield np.asarray(ring.coords)


def name_of(tag):
    c = CAT[tag]
    return (f"{tag[:10]}, Sentinel-{c['platform'][-2:].upper()}, orbit {int(c['orbit'][1:])}, "
            f"sun {float(c['sun_elevation']):.0f}°")


# --- every scene: thresholds, joins, end members
runs = {t: outline(t) for t in TAGS}
assert outline_of(ndsi(runs[MAIN]["bands"]), runs[MAIN]["shade"])[0].equals(runs[MAIN]["body"])
S = {}
for t in TAGS:
    r, b = runs[t], runs[t]["bands"]
    v, g, sun = ndsi(b), (b["B03"] - b["B08"]) / (b["B03"] + b["B08"]), ~r["shade"]
    t_gni = otsu(g[REGION & sun])
    em = members(b, v, g, t_gni, sun)
    S[t] = dict(v=v, g=g, em=em, t_gni=t_gni, jv=joins(v), jg=joins(g), f10=ice_fraction(b, em),
                f11=(em["rock"]["B11"] - b["B11"]) / (em["rock"]["B11"] - em["ice"]["B11"]))
    print(f"{t}: thresholds {r['t_sun']:.3f} sunlit, {r['t_shade']:.3f} shadow; GNI {t_gni:.3f}; shadow in corridor "
          f"{int((r['shade'] & CORR).sum())} px; seeds in body {r['body'].contains(shapely.Point(WEST))} "
          f"{r['body'].contains(shapely.Point(EAST))}")
    for m in ("ice", "dark", "bright", "rock"):
        print(f"   {m:6s} n {em[m]['n']:3d}  " + "  ".join(f"{k} {em[m][k]:.3f}" for k in BANDS + ("B11",)))
    print(f"   snow index join: 4-connected {S[t]['jv'][0]:.3f}, 8-connected {S[t]['jv'][1]:.3f}, iso-line "
          f"{S[t]['jv'][2]:.3f} (threshold minus join {r['t_sun'] - S[t]['jv'][2]:+.3f}: "
          f"{'cut' if S[t]['jv'][2] < r['t_sun'] else 'joined'}); GNI join {S[t]['jg'][0]:.3f} {S[t]['jg'][1]:.3f} "
          f"{S[t]['jg'][2]:.3f} (threshold {t_gni:.3f})")

# --- 27 September: the chain at the snow-index join, its axis and the profile stations
r, b, s = runs[MAIN], runs[MAIN]["bands"], S[MAIN]
inside = raster(r["body"])
path = chain(s["v"], s["jv"][1])
print("27 September chain (8-connected at the join): E N inside snow-index GNI B02 B03 B04 B08 B11 f10 f11")
for p in path:
    print(f"   {X[p[1]]:.0f} {Y[p[0]]:.0f} {int(inside[p])} {s['v'][p]:.3f} {s['g'][p]:.3f} "
          + " ".join(f"{b[k][p]:.3f}" for k in BANDS + ("B11",)) + f" {s['f10'][p]:.2f} {s['f11'][p]:.2f}")
out = np.array([(X[j], Y[i]) for i, j in path if not inside[i, j]])
c0 = out.mean(axis=0)
u = np.linalg.svd(out - c0)[2][0]
u = u if u[0] > 0 else -u
nrm = np.array([-u[1], u[0]])
proj = (out - c0) @ u
ANGLE = float(np.arctan2(u[1], u[0]))
print(f"chain outside the outline: {len(out)} pixels from {out[0]} to {out[-1]}; centre to centre "
      f"{np.hypot(*np.diff(out, axis=0).T).sum():.0f} m along the chain, {np.hypot(*(out[-1] - out[0])):.0f} m straight; "
      f"axis {proj.max() - proj.min():.0f} m at {np.degrees(ANGLE):.0f} deg north of east, centre {c0.round()}")


def across(f, centre, half=25.0):
    """f sampled every metre on the normal through centre, +-half m (bilinear between pixel centres)."""
    o = np.arange(-half, half + 0.5, 1.0)
    pts = centre[None, :] + o[:, None] * nrm[None, :]
    rows, cols = (N1 - PIXEL / 2 - pts[:, 1]) / PIXEL, (pts[:, 0] - E0 - PIXEL / 2) / PIXEL
    return o, map_coordinates(np.nan_to_num(f), [rows, cols], order=1)


STATIONS = [c0 + a * u for a in np.arange(proj.min() + 15, proj.max() - 14, 10.0)]
WIDTH = {}
for t in TAGS:
    w10 = np.array([across(S[t]["f10"], c)[1].sum() for c in STATIONS])
    w11 = np.array([across(S[t]["f11"], c)[1].sum() for c in STATIONS])
    WIDTH[t] = (float(np.median(w10)), float(np.median(w11)))
    print(f"{t}: equivalent ice width, median (range) of {len(STATIONS)} profiles: 10 m bands {np.median(w10):.1f} m "
          f"({w10.min():.1f}-{w10.max():.1f}), B11 {np.median(w11):.1f} m ({w11.min():.1f}-{w11.max():.1f}); "
          f"per station 10 m: " + " ".join(f"{w:.0f}" for w in w10))

# --- 27 September with other B11: nearest, native 20 m, sharpened with B08, sharpened with D
v20 = read(S2 / CAT[MAIN]["file_20m"], band=1)
b11_20 = np.where(v20 == 0, np.nan, v20 / 1e4)
d_index, k27 = d_sharpened(b, r["shade"])
VARIANTS = {"bilinear (outline_2026.py)": s["v"], "nearest neighbour": ndsi(b, nearest(b11_20)),
            "native 20 m": nearest(block(b["B03"]) - b11_20) / nearest(block(b["B03"]) + b11_20),
            "sharpened with B08": ndsi(b, nearest(b11_20) * b["B08"] / nearest(block(b["B08"]))),
            "sharpened with D": d_index}
for name, idx in VARIANTS.items():
    body, t_sun = outline_of(idx, r["shade"])
    j = join_level(idx, isoline_join)
    print(f"27 September, B11 {name}: threshold {t_sun:.3f}, strip joins at {j:.3f} ({'joined' if j >= t_sun else 'cut'}),"
          f" glacier {body.area:.0f} m2 ({body.area - r['body'].area:+.0f})" + (f", k {k27:.2f}" if "D" in name else ""))

# --- the D-sharpened index on every scene: area, strip, scene-to-scene agreement
SHARP = {}
for t in TAGS:
    idx, k = d_sharpened(runs[t]["bands"], runs[t]["shade"])
    body, t_sun = outline_of(idx, runs[t]["shade"])
    j = join_level(idx, isoline_join)
    SHARP[t] = body
    print(f"{t} sharpened with D (k {k:.2f}): threshold {t_sun:.3f}, glacier {body.area:.0f} m2 "
          f"({body.area - runs[t]['body'].area:+.0f}, {100 * (body.area / runs[t]['body'].area - 1):+.1f} %), "
          f"holes {len(body.interiors)} (were {len(runs[t]['body'].interiors)}), strip {'joined' if j >= t_sun else 'cut'} "
          f"(join {j:.3f})")
for name, bodies in (("outline_2026.py", [runs[t]["body"] for t in [MAIN] + LATE]),
                     ("sharpened with D", [SHARP[t] for t in [MAIN] + LATE])):
    pairs = [boundary_distance(p, q) for i, p in enumerate(bodies) for q in bodies[i + 1:]]
    areas = [p.area for p in bodies]
    print(f"late September, {name}: areas {' '.join(f'{a:.0f}' for a in areas)}, spread {max(areas) - min(areas):.0f} m2,"
          f" mean boundary distance between the scenes {np.mean(pairs):.2f} m")
added = shapely.difference(SHARP[MAIN], r["body"])
removed = shapely.difference(r["body"], SHARP[MAIN])
for what, geom in (("added", added), ("removed", removed)):
    parts = sorted(shapely.get_parts(geom), key=lambda p: -p.area)
    print(f"27 September, sharpened with D minus outline_2026.py, {what}: {geom.area:.0f} m2 in {len(parts)} parts; "
          f"from 150 m2: " + "; ".join(f"{p.area:.0f} m2 at {p.centroid.x:.0f} {p.centroid.y:.0f}"
                                         for p in parts if p.area >= 150))


# --- per scene: the detached ice of the strip (the iso-line polygons that the largest-polygon rule drops)
def ice_z(index, shade, reference=G24):
    """z of outline_2026.py's outline() for any index: index minus the threshold of its class, -1 outside."""
    region, inside = raster(reference.buffer(BUFFER)), raster(reference)
    t_sun = otsu(index[region & ~shade])
    t_shade = otsu(index[inside & shade]) if (inside & shade).sum() >= 100 else np.nan
    z = index - np.where(shade & np.isfinite(t_shade), t_shade, t_sun)
    z[~region | (shade & ~inside & np.isfinite(t_shade))] = -1.0
    return z, t_sun


AXIS = shapely.LineString([c0 + proj.min() * u, c0 + proj.max() * u])
for t in TAGS:
    z, _ = ice_z(S[t]["v"], runs[t]["shade"])
    parts = polygons(z)
    body = max(parts, key=lambda p: p.area)
    near = [p for p in parts if p is not body and p.distance(AXIS) < 20]
    print(f"{t}: detached polygons within 20 m of the strip axis: "
          + ("; ".join(f"{p.area:.0f} m2 at {p.centroid.x:.0f} {p.centroid.y:.0f}" for p in near) or "none")
          + f"; body overlaps the axis over {shapely.intersection(body, AXIS).length:.0f} of {AXIS.length:.0f} m")


# --- a targeted rule: chains of 10 m ice that join two parts of the ice or close a rock bay
from scipy.ndimage import binary_dilation, binary_fill_holes  # noqa: E402


def join_rule(t, t_low):
    """outline_2026.py's ice plus the 8-connected components of sunlit region pixels with GNI >= its Otsu threshold
    and snow index >= t_low (not already ice) that touch two separate parts of the ice, or one part and close a
    non-ice area of at least MIN_HOLE; accepted pixels are bounded at the level t_low. Returns body and the joins."""
    r, s = runs[t], S[t]
    z, t_sun = ice_z(s["v"], r["shade"])
    strong = z > 0
    weak = REGION & ~r["shade"] & (s["g"] >= s["t_gni"]) & (s["v"] >= t_low) & ~strong
    lab_s, _ = label(strong, EIGHT)
    lab_w, n_w = label(weak, EIGHT)
    enclosed = binary_fill_holes(strong) & ~strong
    accepted, found = np.zeros_like(strong), []
    for c in range(1, n_w + 1):
        cm = lab_w == c
        touched = set(np.unique(lab_s[binary_dilation(cm, EIGHT) & ~cm])) - {0}
        new_hole = (binary_fill_holes(strong | cm) & ~(strong | cm) & ~enclosed).sum() * PIXEL ** 2
        if len(touched) >= 2 or (len(touched) == 1 and new_hole >= MIN_HOLE):
            accepted |= cm
            i, j = np.argwhere(cm).mean(axis=0)
            found.append((int(cm.sum()), float(np.interp(j, np.arange(len(X)), X)),
                          float(np.interp(i, np.arange(len(Y)), Y)), len(touched), float(new_hole)))
    z2 = np.where(accepted, np.maximum(z, s["v"] - t_low), z)
    parts = polygons(z2)
    body = max(parts, key=lambda p: p.area)
    body = shapely.Polygon(body.exterior, [h for h in body.interiors if shapely.Polygon(h).area >= MIN_HOLE])
    return body, found


RULE = {}
for t_low_name, t_low_of in (("0", lambda t: 0.0), ("threshold - 0.15", lambda t: runs[t]["t_sun"] - 0.15)):
    for t in TAGS:
        body, found = join_rule(t, t_low_of(t))
        RULE[(t_low_name, t)] = body
        strip = body.contains(shapely.Point(c0)) and body.contains(shapely.Point(WEST)) and body.contains(shapely.Point(EAST))
        hole = any(shapely.Polygon(h).contains(shapely.Point(2680330, 1140990)) for h in body.interiors)
        print(f"join rule, snow index >= {t_low_name}, {t}: glacier {body.area:.0f} m2 ({body.area - runs[t]['body'].area:+.0f}), "
              f"strip {'joined' if strip and hole else 'cut'}; joins (pixels, E, N, parts touched, new hole m2): "
              + "; ".join(f"{n} {e:.0f} {n_:.0f} {k} {h:.0f}" for n, e, n_, k, h in found))
    bodies = [RULE[(t_low_name, t)] for t in [MAIN] + LATE]
    pairs = [boundary_distance(p, q) for i, p in enumerate(bodies) for q in bodies[i + 1:]]
    print(f"join rule, snow index >= {t_low_name}: late-September areas {' '.join(f'{p.area:.0f}' for p in bodies)}, "
          f"mean boundary distance between the scenes {np.mean(pairs):.2f} m")


# --- model: a straight strip of width w through rock between two ice fields
def model_join(w, ox, oy, psf, em):
    """Join levels (4-connected) of the snow index and of GNI along the modelled strip, grid shifted by (ox, oy) m."""
    L, H = 400, 240                                             # m, 1 m cells, rows southwards
    yy, xx = np.mgrid[0:H, 0:L] + 0.5
    dist = np.abs((xx - L / 2 - ox) * np.sin(ANGLE) + (yy - H / 2 - oy) * np.cos(ANGLE))
    fine = np.clip(w / 2 - dist + 0.5, 0, 1)
    fine[:, :80], fine[:, -80:] = 1.0, 1.0

    def mean(f, n):
        f = gaussian_filter(f, psf * n) if psf else f
        return f.reshape(H // n, n, L // n, n).mean(axis=(1, 3))

    f10, f20 = mean(fine, 10), mean(fine, 20)
    b03, b08 = (em["rock"][k] + f10 * (em["ice"][k] - em["rock"][k]) for k in ("B03", "B08"))
    b11 = bilinear(em["rock"]["B11"] + f20 * (em["ice"]["B11"] - em["rock"]["B11"]))
    seeds, mask = ((H // 20, 3), (H // 20, L // 10 - 4)), np.ones(f10.shape, bool)
    return [join_level(a, pixels_join(FOUR), mask, seeds) for a in ((b03 - b11) / (b03 + b11), (b03 - b08) / (b03 + b08))]


WIDTHS, PHASES = np.arange(4, 52, 2.0), [(a, c) for a in np.arange(0, 20, 2.5) for c in np.arange(0, 20, 2.5)]
MODEL = {psf: np.array([[model_join(w, ox, oy, psf, S[MAIN]["em"]) for ox, oy in PHASES] for w in WIDTHS])
         for psf in (0.0, 0.4)}                                 # widths x phases x (snow index, GNI)
for psf, m in MODEL.items():
    med = np.median(m, axis=1)
    lo = m.min(axis=1)
    print(f"model, point spread {psf} pixel: width m, snow-index join median (lowest), GNI join median (lowest)")
    for w, a, b_, c, d in zip(WIDTHS, med[:, 0], lo[:, 0], med[:, 1], lo[:, 1]):
        print(f"   {w:4.0f} {a:6.3f} ({b_:6.3f})  {c:6.3f} ({d:6.3f})")
    for name, k, thr in (("snow index", 0, runs[MAIN]["t_sun"]), ("GNI", 1, S[MAIN]["t_gni"])):
        half = WIDTHS[np.argmax(med[:, k] >= thr)] if (med[:, k] >= thr).any() else np.nan
        every = WIDTHS[np.argmax(lo[:, k] >= thr)] if (lo[:, k] >= thr).any() else np.nan
        print(f"   {name}: joined at the threshold of 27 September ({thr:.3f}) from {half:.0f} m (half the positions), "
              f"{every:.0f} m (every position)")
    for t in [MAIN] + LATE:
        print(f"   {t}: observed snow-index join {S[t]['jv'][0]:.3f} (4-connected) = model width "
              f"{np.interp(S[t]['jv'][0], med[:, 0], WIDTHS):.0f} m (median curve)")

# --- the threshold alone: 27 September with the sunlit threshold lowered to the strip's join
for t_low in (S[MAIN]["jv"][2], 0.25):
    z = S[MAIN]["v"] - np.where(r["shade"], r["t_shade"], t_low)
    z[~REGION | (r["shade"] & ~raster(G24))] = -1.0
    body = max(polygons(z), key=lambda p: p.area)
    print(f"27 September, sunlit threshold {t_low:.3f} instead of {r['t_sun']:.3f}: glacier {body.area:.0f} m2 "
          f"({body.area - r['body'].area:+.0f}, {100 * (body.area / r['body'].area - 1):+.1f} %)")

# --- the 2021 surface across the strip: elevation and its departure from a 15 m smoothing, on the middle profile
e0, n0, e1, n1 = BOX
DEM05 = gaussian_filter(read(ROOT / "data/dem_0p5m.tif", (e0 - 100, n0 - 100, e1 + 100, n1 + 100), band=1), 2.0)
LOCAL = (DEM05 - gaussian_filter(DEM05, 30.0))[200:-200, 200:-200]                # sigma 15 m, read with 100 m margin
DEM05 = DEM05[200:-200, 200:-200]                                                  # sigma 1 m against the striping
for k in (len(STATIONS) // 4, len(STATIONS) // 2, 3 * len(STATIONS) // 4):
    o = np.arange(-40, 41, 5.0)
    pts = STATIONS[k][None, :] + o[:, None] * nrm[None, :]
    rows, cols = (n1 - 0.25 - pts[:, 1]) / 0.5, (pts[:, 0] - e0 - 0.25) / 0.5
    zz, ll = (map_coordinates(a, [rows, cols], order=1) for a in (DEM05, LOCAL))
    print(f"2021 surface across the strip at {STATIONS[k].round()} (offset m north-west positive: elevation m, local m): "
          + " ".join(f"{a:+.0f}:{b:.1f}/{c:+.2f}" for a, b, c in zip(o, zz, ll)))


# --- images
plt = figure_style()
ORTHO = np.moveaxis(read(ROOT / "data/ortho_0p25m.tif", BOX).astype(np.uint8), 0, -1)
XD = np.linspace(e0 + 0.25, e1 - 0.25, DEM05.shape[1])
YD = np.linspace(n1 - 0.25, n0 + 0.25, DEM05.shape[0])


def draw_outlines(a, tag=None):
    for xy in rings(G24):
        a.plot(xy[:, 0], xy[:, 1], color="#ff8c00", lw=0.8)
    for xy in rings(runs[MAIN]["body"]):
        a.plot(xy[:, 0], xy[:, 1], color="w", lw=0.9, ls=(0, (3, 2)))
    if tag:
        for xy in rings(runs[tag]["body"]):
            a.plot(xy[:, 0], xy[:, 1], color="#ff1010", lw=1.1)


def caption(a, text, box=BOX):
    a.text(box[0] + 4, box[3] - 4, text, ha="left", va="top", fontsize=7, color="k",
           bbox=dict(facecolor="w", alpha=0.75, lw=0, pad=1.2))


def finish(a, box=BOX):
    a.set_xlim(box[0], box[2]), a.set_ylim(box[1], box[3]), a.set_axis_off()


def relief_panel(a, step=5, labels=4):
    a.imshow(LOCAL, extent=[e0, e1, n0, n1], cmap="BrBG_r", vmin=-1.5, vmax=1.5)
    cs = a.contour(XD, YD, DEM05, np.arange(2700, 3100, step), colors="k", linewidths=0.4)
    a.clabel(cs, cs.levels[::labels], fontsize=6, fmt="%d")


for kind in ("true_colour", "snow_index"):
    f, axs = plt.subplots(2, 6, figsize=(6 * 3.2, 2 * 3.2), dpi=100)
    f.subplots_adjust(0, 0, 1, 1, 0.015, 0.015)
    for a, t in zip(axs.flat, TAGS):
        if kind == "true_colour":
            a.imshow(true_colour(runs[t]["bands"]), extent=[E0, E1, N0, N1], interpolation="nearest")
            caption(a, name_of(t))
        else:
            a.imshow(S[t]["v"], extent=[E0, E1, N0, N1], interpolation="nearest", cmap="RdBu",
                     norm=TwoSlopeNorm(runs[t]["t_sun"], -0.5, 1.0))
            a.contour(X, Y, S[t]["v"], [runs[t]["t_sun"]], colors="k", linewidths=0.7)
            caption(a, f"{t[:10]}: threshold {runs[t]['t_sun']:.3f}, strip joins at {S[t]['jv'][2]:.3f}")
        draw_outlines(a, t)
        finish(a)
    a = axs.flat[9]
    a.imshow(ORTHO, extent=[e0, e1, n0, n1])
    draw_outlines(a)
    caption(a, "orthophoto, 21 and 29 August 2024")
    finish(a)
    a = axs.flat[10]
    relief_panel(a)
    draw_outlines(a)
    caption(a, "surface of 30 September 2021 minus its 15 m smoothing")
    finish(a)
    a = axs.flat[11]
    a.imshow(S[MAIN]["g"], extent=[E0, E1, N0, N1], interpolation="nearest", cmap="RdBu",
             norm=TwoSlopeNorm(S[MAIN]["t_gni"], -0.15, 0.25))
    a.contour(X, Y, S[MAIN]["g"], [S[MAIN]["t_gni"]], colors="k", linewidths=0.7)
    draw_outlines(a, MAIN)
    caption(a, f"27 September, green and near infrared, threshold {S[MAIN]['t_gni']:.3f}")
    finish(a)
    f.savefig(PREVIEW / f"outline_2026_place3_{kind}.png")
    plt.close(f)

# 27 September, pixel values over the strip (x 100)
ZOOM = (2680200, 1140960, 2680470, 1141130)
cz, rz = (X > ZOOM[0]) & (X < ZOOM[2]), (Y > ZOOM[1]) & (Y < ZOOM[3])
f, axs = plt.subplots(3, 1, figsize=((ZOOM[2] - ZOOM[0]) / 10 * 0.42, 3 * (ZOOM[3] - ZOOM[1]) / 10 * 0.42), dpi=100)
f.subplots_adjust(0, 0, 1, 1, 0, 0.01)
for a, (arr, norm, cmap, what) in zip(axs, (
        (S[MAIN]["v"], TwoSlopeNorm(r["t_sun"], -0.5, 1.0), "RdBu", f"snow index x 100, threshold {r['t_sun']:.3f}"),
        (S[MAIN]["g"], TwoSlopeNorm(S[MAIN]["t_gni"], -0.15, 0.25), "RdBu",
         f"green and near infrared x 100, threshold {S[MAIN]['t_gni']:.3f}"),
        (b["B11"], None, "magma", "shortwave infrared B11 x 100, 20 m interpolated to 10 m"))):
    a.imshow(arr, extent=[E0, E1, N0, N1], interpolation="nearest", cmap=cmap, norm=norm,
             **({} if norm else dict(vmin=0, vmax=0.4)))
    sub = arr[np.ix_(rz, cz)]
    for i, y in enumerate(Y[rz]):
        for j, x in enumerate(X[cz]):
            a.text(x, y, f"{sub[i, j] * 100:.0f}", ha="center", va="center", fontsize=6.5,
                   color="k" if cmap == "RdBu" else "w")
    for xy in rings(r["body"]):
        a.plot(xy[:, 0], xy[:, 1], color="#ff1010", lw=1.6)
    a.plot([X[p[1]] for p in path], [Y[p[0]] for p in path], color="#ffd400", lw=1.0, marker="o", ms=2.5)
    caption(a, what, ZOOM)
    finish(a, ZOOM)
f.savefig(PREVIEW / "outline_2026_place3_values.png")
plt.close(f)

# orthophoto 2024, relief of the 2021 surface and 27 September at full resolution, with the chain and the profiles
f, axs = plt.subplots(1, 3, figsize=(3 * 9, 9), dpi=100)
f.subplots_adjust(0, 0, 1, 1, 0.01, 0)
axs[0].imshow(ORTHO, extent=[e0, e1, n0, n1])
relief_panel(axs[1], step=1, labels=5)
axs[2].imshow(true_colour(b), extent=[E0, E1, N0, N1], interpolation="lanczos")
for a in axs:
    draw_outlines(a, MAIN)
    a.plot([X[p[1]] for p in path], [Y[p[0]] for p in path], color="#ffd400", lw=0.8, marker="o", ms=3)
    for c in STATIONS:
        q = np.array([c - 25 * nrm, c + 25 * nrm])
        a.plot(q[:, 0], q[:, 1], color="#00e0ff", lw=0.5)
    finish(a)
f.savefig(PREVIEW / "outline_2026_place3_ortho_relief.png")
plt.close(f)

# model and profiles
f, axs = plt.subplots(1, 3, figsize=(15, 4.2), dpi=120)
f.subplots_adjust(0.05, 0.13, 0.99, 0.93, 0.25)
for a, k, thr, what in ((axs[0], 0, r["t_sun"], "snow index"), (axs[1], 1, S[MAIN]["t_gni"], "green and near infrared")):
    for psf, colour in ((0.0, "#027abe"), (0.4, "#c51415")):
        m = MODEL[psf][:, :, k]
        a.fill_between(WIDTHS, m.min(axis=1), m.max(axis=1), color=colour, alpha=0.2, lw=0)
        a.plot(WIDTHS, np.median(m, axis=1), color=colour, lw=1.2,
               label="pixel boxes" if psf == 0 else "pixel boxes and point spread 0.4 pixel")
    a.axhline(thr, color="k", lw=0.8, ls="--")
    a.text(50, thr, f"threshold of 27 September {thr:.3f}", ha="right", va="bottom", fontsize=7)
    for t in [MAIN] + LATE:
        a.plot(WIDTH[t][0], S[t]["jv" if k == 0 else "jg"][0], "o", ms=3.5, color="k")
        a.plot(WIDTH[t][1], S[t]["jv" if k == 0 else "jg"][0], "s", ms=3.5, mfc="none", color="k")
    a.set_xlabel("width of the strip of ice (m)")
    a.set_ylabel(f"{what}: highest level joining the two sides")
    a.legend(loc="lower right", fontsize=7)
a = axs[2]
for t, colour in ((MAIN, "#c51415"), (HIGH_SUN[0], "#ff8c00"), (HIGH_SUN[1], "#027abe"), (YEAR_BEFORE, "#4d4d4d")):
    prof10 = np.median([across(S[t]["f10"], c, 30.0)[1] for c in STATIONS], axis=0)
    o, _ = across(S[t]["f11"], STATIONS[0], 30.0)
    prof11 = np.median([across(S[t]["f11"], c, 30.0)[1] for c in STATIONS], axis=0)
    a.plot(o, prof10, color=colour, lw=1.2, label=f"{t[:10]}, 10 m bands")
    a.plot(o, prof11, color=colour, lw=1.0, ls="--", label=f"{t[:10]}, B11 20 m")
a.axhline(0, color="k", lw=0.5)
a.set_xlabel("distance across the strip, north-west positive (m)")
a.set_ylabel("ice fraction, median of the profiles")
a.legend(loc="upper right", fontsize=6, ncols=2)
f.savefig(PREVIEW / "outline_2026_place3_model.png")
plt.close(f)

# the rules over the northern margin, 27 September: outline_2026.py, sharpened with D, join rule
WIDE = (2679600, 1140800, 2681300, 1141450)
f, axs = plt.subplots(2, 1, figsize=(17, 2 * 6.5), dpi=100)
f.subplots_adjust(0, 0, 1, 1, 0, 0.01)
axs[0].imshow(true_colour(b), extent=[E0, E1, N0, N1], interpolation="nearest")
axs[1].imshow(S[MAIN]["g"], extent=[E0, E1, N0, N1], interpolation="nearest", cmap="RdBu",
              norm=TwoSlopeNorm(S[MAIN]["t_gni"], -0.15, 0.25))
for a in axs:
    for geom, colour, lw, ls in ((G24, "#ff8c00", 0.8, "-"), (SHARP[MAIN], "#ffd400", 1.0, "-"),
                                 (RULE[("threshold - 0.15", MAIN)], "#00e0ff", 1.0, "-"), (r["body"], "#ff1010", 1.2, "--")):
        for xy in rings(geom):
            a.plot(xy[:, 0], xy[:, 1], color=colour, lw=lw, ls=ls)
    finish(a, WIDE)
caption(axs[0], "27 September: red dashed outline_2026.py, yellow B11 sharpened with green minus near infrared, "
        "cyan join rule (snow index from threshold minus 0.15), orange 2024", WIDE)
caption(axs[1], "27 September, green and near infrared", WIDE)
f.savefig(PREVIEW / "outline_2026_place3_rule.png")
plt.close(f)


# --- other places of the same kind on the northern margin: join levels and panels
def case(name, box, corridor, west, east):
    """Join levels of the snow index and GNI between two seeds inside a corridor, per scene, and panels of BOX-like
    size: 27 September true colour, snow index and GNI; snow index of 12 September, 13 August 2026, 7 September 2025;
    the orthophoto of 2024; the detached ice of 27 September near the seeds."""
    mask, seeds = window(corridor), (at(west), at(east))
    for t in TAGS:
        jv = join_level(S[t]["v"], isoline_join, mask, seeds)
        jg = join_level(S[t]["g"], isoline_join, mask, seeds)
        print(f"{name}, {t}: snow index joins at {jv:.3f} (threshold {runs[t]['t_sun']:.3f}, "
              f"{'joined' if jv >= runs[t]['t_sun'] else 'cut'}), GNI at {jg:.3f} (threshold {S[t]['t_gni']:.3f}, "
              f"{'joined' if jg >= S[t]['t_gni'] else 'cut'}); seeds in body {runs[t]['body'].contains(shapely.Point(west))} "
              f"{runs[t]['body'].contains(shapely.Point(east))}")
    f, axs = plt.subplots(2, 4, figsize=(4 * 3.2, 2 * 3.2 * (box[3] - box[1]) / (box[2] - box[0])), dpi=100)
    f.subplots_adjust(0, 0, 1, 1, 0.015, 0.015)
    ortho = np.moveaxis(read(ROOT / "data/ortho_0p25m.tif", box).astype(np.uint8), 0, -1)
    for a, (t, what) in zip(axs.flat, ((MAIN, "true"), (MAIN, "ndsi"), (MAIN, "gni"), (HIGH_SUN[0], "ndsi"),
                                       (HIGH_SUN[1], "ndsi"), (YEAR_BEFORE, "ndsi"), (None, "ortho"), (LATE[0], "ndsi"))):
        if what == "true":
            a.imshow(true_colour(runs[t]["bands"]), extent=[E0, E1, N0, N1], interpolation="nearest")
        elif what == "ndsi":
            a.imshow(S[t]["v"], extent=[E0, E1, N0, N1], interpolation="nearest", cmap="RdBu",
                     norm=TwoSlopeNorm(runs[t]["t_sun"], -0.5, 1.0))
            a.contour(X, Y, S[t]["v"], [runs[t]["t_sun"]], colors="k", linewidths=0.7)
        elif what == "gni":
            a.imshow(S[t]["g"], extent=[E0, E1, N0, N1], interpolation="nearest", cmap="RdBu",
                     norm=TwoSlopeNorm(S[t]["t_gni"], -0.15, 0.25))
            a.contour(X, Y, S[t]["g"], [S[t]["t_gni"]], colors="k", linewidths=0.7)
        else:
            a.imshow(ortho, extent=[box[0], box[2], box[1], box[3]])
        for xy in rings(G24):
            a.plot(xy[:, 0], xy[:, 1], color="#ff8c00", lw=0.8)
        for xy in rings(runs[t]["body"] if t else r["body"]):
            a.plot(xy[:, 0], xy[:, 1], color="#ff1010", lw=1.1)
        for p in (west, east):
            a.plot(*p, "o", ms=4, mfc="none", color="#ffd400")
        label = {"true": "true colour", "ndsi": "snow index", "gni": "green and near infrared",
                 "ortho": "orthophoto 2024"}[what]
        caption(a, f"{t[:10] + ', ' if t else ''}{label}", box)
        finish(a, box)
    f.savefig(PREVIEW / f"outline_2026_place3_{name}.png")
    plt.close(f)


case("gully", (2679650, 1141200, 2679950, 1141430), (2679740, 1141280, 2679860, 1141390),
     (2679835, 1141285), (2679775, 1141365))
case("east_bay", (2680450, 1140960, 2680750, 1141190), (2680560, 1141000, 2680700, 1141120),
     (2680585, 1141105), (2680625, 1141015))


# --- the two candidate rules at the strip and at the two other places: area added near the axis, joins
NEAR = AXIS.buffer(25.0)
GULLY_TOP, BAY = shapely.Point(2679775, 1141365), shapely.Point(2680600, 1141070)
for t in [MAIN] + LATE + HIGH_SUN:
    rule = RULE[("threshold - 0.15", t)]
    print(f"{t}: added within 25 m of the strip axis: sharpened with D "
          f"{shapely.intersection(shapely.difference(SHARP[t], runs[t]['body']), NEAR).area:.0f} m2, join rule "
          f"{shapely.intersection(shapely.difference(rule, runs[t]['body']), NEAR).area:.0f} m2; gully top in body: "
          f"outline_2026.py {runs[t]['body'].contains(GULLY_TOP)}, sharpened {SHARP[t].contains(GULLY_TOP)}, "
          f"join rule {rule.contains(GULLY_TOP)}; east bay a hole: sharpened "
          f"{any(shapely.Polygon(h).contains(BAY) for h in SHARP[t].interiors)}, join rule "
          f"{any(shapely.Polygon(h).contains(BAY) for h in rule.interiors)}")
