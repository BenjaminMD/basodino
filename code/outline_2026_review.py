"""Adversarial check of the outline rule change of 2026-10-06 (steps 5 and 6 of outline_2026.py): at every place
where the September 2026 outline changed, and at the three places left unchanged on purpose, is a new hole rock, a
kept block ice, a joined strip resolved by Sentinel-2? A check, not part of the report chain.

The five late-September scenes vote in step 6, so they cannot test the votes. The evidence added here:
  hold-out scenes, clear over the glacier and used nowhere in outline_2026.py, each outlined by its steps 1-5:
    29 September (Sentinel-2A, orbit 65; cloud at the eastern margin), 17 September (2C, orbit 65), 15 September
    (2B, orbit 108), 12 September (2A, orbit 108), 7 September (2C, orbit 65), 2 September (2A, orbit 108);
  the higher-sun scenes of outline_2026.py, 12 September (2B, orbit 65) and 13 August 2026 (2B, orbit 65);
  the scene classification of every scene (cloud 8-10, cloud shadow 3; 20 m);
  direct sunlight of 27 September from swissALTI3D 2 m (surface of 2021): the cosine of the incidence angle and the
    share of 2 m cells in shadow of terrain within 300 m, per 10 m pixel (step 3 works on a 10 m model);
  SWISSIMAGE 2024 at 0.25 m (21 and 29 August 2024);
  for stdout only: 13 August 2026 (2A, orbit 108), 11 August, 27 and 24 July 2026.

stdout: the 27 September outline rebuilt here against the GeoJSON; for every joined group of 27 September the
  polygons it connects (area, share inside 2024, ice votes); per place the pixels that changed and, per scene, the
  pixels sunlit / ice and rock among them / candidates of step 5 / joined / cloud, and medians of the index minus
  the threshold (before the joins), of GNI minus its threshold, of B02, B03, B08, B11; per pixel (places of 40
  pixels or fewer) the direct sunlight and one letter per scene: J joined, I ice sunlit, i ice not sunlit,
  c candidate, R rock sunlit, r rock not sunlit, x cloud (ice and rock by the scene's iso-line before its joins);
  validity of the outline; every hole and polygon of 27 September with the votes of the five voters and of the
  clear hold-outs (29, 15, 12 September 2A, 12 September 2B, 13 August 2B); the joined groups of every scene and the
  small polygons they connect (area, share inside 2024); per scene the sunlit bare ice 50 m inside the body (haze
  raises B11 and lowers the index); place 1 against the sunlit ice 40-100 m around it, and the north-western patch
  (reflectance, scene classes); per scene whether its iso-line alone (before its joins) closes the place-3 bay and
  reaches the place-3 block and the gully top; the five voters redone for DROP from 0 to 0.25 (area, joins, spread).
tex/preview/outline_2026_review_<place>.png: row 1 true colour of the five voters and 29 September, row 2 of the
  hold-outs and 13 August (10 m pixels); row 3 the index minus threshold of 27 September, 17 September, 12 September
  (2B) and 13 August with each scene's own iso-line (black), true colour of 27 September brightened four times (shadow),
  SWISSIMAGE 2024. Outline before white dashed, after red, 2024 orange, cast shadow yellow dotted, joined pixels of
  the scene yellow dots.
tex/preview/outline_2026_review_aerial.png: SWISSIMAGE 2024 at 0.25 m, 120 m around six changes, with the 10 m grid.

usage, after outline_2026.py, in the same mirror (its docstring):
  rsync -a code/outline_2026_review.py s02:/data/befahl/projects/AGL/basodino/report/code/
  ssh s02 'source ~/miniconda3/bin/activate agl && cd /data/befahl/projects/AGL/basodino/report &&
           ~/.local/bin/nrun python code/outline_2026_review.py'
  then copy tex/preview/outline_2026_review_*.png back
"""
from pathlib import Path

exec((Path(__file__).resolve().parent / "outline_2026.py").read_text(encoding="utf-8")
     .split("# --- the outline and its checks")[0])                   # noqa: S102 (the method, unchanged)

from matplotlib.colors import TwoSlopeNorm  # noqa: E402

VOTERS = [MAIN] + LATE
HOLD = ["2026-09-29_S2A_R065", "2026-09-17_S2C_R065", "2026-09-15_S2B_R108", "2026-09-12_S2A_R108",
        "2026-09-07_S2C_R065", "2026-09-02_S2A_R108"]
EXTRA = ["2026-08-13_S2A_R108", "2026-08-11_S2C_R108", "2026-07-27_S2B_R108", "2026-07-24_S2A_R108"]  # stdout only
SCENES = VOTERS + HOLD + HIGH_SUN + EXTRA
CLOUD = (3, 8, 9, 10)                                   # scene classification: cloud shadow, cloud, cirrus
PLACES = {"place1": (2680668, 1140883, 120), "place2": (2680415, 1141159, 150), "place3": (2680359, 1141051, 150),
          "gully": (2679809, 1141328, 150), "northwest": (2679235, 1141150, 120),
          "tongue": (2680915, 1140791, 150), "margin": (2680578, 1140457, 120), "ridge": (2679146, 1140797, 120),
          "neck": (2680225, 1141030, 120), "missed": (2679065, 1140840, 120)}      # E, N, half box (m)

res = {t: outline(t) for t in SCENES}
region = raster(G24.buffer(BUFFER))
for t, r in res.items():
    b = r["bands"]
    scl = read(S2 / CAT[t]["file_scl"], band=1)
    r["cloud"] = np.isin(np.repeat(np.repeat(scl, 2, 0), 2, 1), CLOUD)
    r["raw"] = r["z"] - DROP * r["joined"]                              # index minus threshold, before the joins
    gni = (b["B03"] - b["B08"]) / (b["B03"] + b["B08"])
    r["gni"] = gni - otsu(gni[region & ~r["shade"]])
    r["cand"] = r["sunlit"] & (r["raw"] <= 0) & (r["raw"] >= -DROP) & (r["gni"] >= 0)

ICE = sum((res[t]["z"] > 0) & res[t]["sunlit"] for t in VOTERS)
ROCK = sum((res[t]["z"] <= 0) & res[t]["sunlit"] for t in VOTERS)
r = res[MAIN]
new = glacier(r, ICE, ROCK)
written = gpd.read_file(OUT / f"outline_{MAIN[:10]}.geojson").geometry.iloc[0]
old = max(polygons(r["raw"]), key=lambda p: p.area)
old = shapely.Polygon(old.exterior, [h for h in old.interiors if shapely.Polygon(h).area >= 1000.0])
print(f"rebuilt {new.area:.0f} m2, GeoJSON {written.area:.0f} m2, symmetric difference "
      f"{shapely.symmetric_difference(new, written).area:.1f} m2; before {old.area:.0f} m2")

# direct sunlight of 27 September on the 2 m model: incidence and cast shadow within 300 m, per 10 m pixel
with rasterio.open(ROOT / "data/dem_2m.tif") as src:
    z2 = src.read(1, window=from_bounds(E0 - 300, N0 - 300, E1 + 300, N1 + 300, src.transform)
                  .round_offsets().round_lengths()).astype(np.float32)
el, az = (np.radians(v) for v in r["bands"]["sun"])
gy, gx = np.gradient(z2, 2.0)
cos_i = (-gx * np.sin(az) * np.cos(el) + gy * np.cos(az) * np.cos(el) + np.sin(el)) / np.sqrt(gx ** 2 + gy ** 2 + 1)
m2 = 150                                                                # 300 m margin in 2 m cells
rows, cols = np.mgrid[m2:z2.shape[0] - m2, m2:z2.shape[1] - m2].astype(np.float32)
z0, shade2 = z2[m2:-m2, m2:-m2], np.zeros((z2.shape[0] - 2 * m2, z2.shape[1] - 2 * m2), bool)
for d in np.arange(2.0, 300.0, 1.0):
    ahead = map_coordinates(z2, [rows - d * np.cos(az) / 2.0, cols + d * np.sin(az) / 2.0], order=1)
    shade2 |= ahead > z0 + d * np.tan(el)
COS_I = np.clip(cos_i[m2:-m2, m2:-m2], 0, None).reshape(len(Y), 5, len(X), 5).mean(axis=(1, 3))
SHADE2 = shade2.reshape(len(Y), 5, len(X), 5).mean(axis=(1, 3))


def letter(t, i, j):
    q = res[t]
    if q["cloud"][i, j]:
        return "x"
    if q["joined"][i, j]:
        return "J"
    if q["raw"][i, j] > 0:
        return "I" if q["sunlit"][i, j] else "i"
    if q["cand"][i, j]:
        return "c"
    return "R" if q["sunlit"][i, j] else "r"


# --- joined groups of 27 September and the polygons they connect
pre = polygons(r["raw"])
pid = rasterize([(p, k) for k, p in enumerate(pre, 1)], r["z"].shape, transform=T10, dtype="int32")
groups, n = label(r["joined"])
print(f"\n27 September: {n} joined groups, {r['joined'].sum()} pixels; polygons before the joins {len(pre)}")
for g in range(1, n + 1):
    grp = groups == g
    ids = sorted(set(pid[binary_dilation(grp) & ~grp]) - {0})
    ii, jj = np.nonzero(grp)
    print(f"  group {g}: {grp.sum()} px at {X[jj].mean():.0f} {Y[ii].mean():.0f}, connects " + "; ".join(
        f"{pre[k - 1].area:.0f} m2 ({pre[k - 1].intersection(G24).area / pre[k - 1].area:.0%} in 2024, ice votes "
        f"{ICE[raster(pre[k - 1])].max(initial=0)})" for k in ids))

# --- per place
changed = shapely.symmetric_difference(new, old)
plt = figure_style()
for name, (e, n_, half) in PLACES.items():
    near = shapely.Point(e, n_).buffer(25)
    parts = [p for p in shapely.get_parts(changed) if p.intersects(near) and p.area >= 10]
    if parts and name not in ("ridge", "neck", "missed"):
        geom = shapely.union_all(parts)
    else:                       # unchanged places: the 27 September polygon there, else a disc
        cand = [p for p in pre if p.distance(shapely.Point(e, n_)) < 15 and p.area < 5000]
        geom = cand[0] if cand else shapely.Point(e, n_).buffer(12)
    px = raster(geom)
    if not px.any():
        px = raster(geom.buffer(5))
    added = shapely.difference(new, old).intersection(geom).area
    removed = shapely.difference(old, new).intersection(geom).area
    print(f"\n{name} ({e} {n_}): {px.sum()} pixels, added {added:.0f} m2, removed {removed:.0f} m2, "
          f"inside 2024 {geom.intersection(G24).area / geom.area:.0%}; votes ice max {ICE[px].max()}, "
          f"rock max {ROCK[px].max()}; direct sunlight 27 September: cos i median {np.median(COS_I[px]):.2f}, "
          f"2 m shadow share median {np.median(SHADE2[px]):.2f}")
    for t in SCENES:
        q, b = res[t], res[t]["bands"]

        def med(a, m=px):
            return np.nanmedian(a[m]) if m.any() else np.nan
        sl = q["sunlit"] & px & ~q["cloud"]
        print(f"  {t}: sunlit {sl.sum():3d} ice {(sl & (q['z'] > 0)).sum():3d} rock {(sl & (q['z'] <= 0)).sum():3d} "
              f"| not sunlit ice {(~q['sunlit'] & px & (q['z'] > 0)).sum():3d} "
              f"| cand {(q['cand'] & px).sum():3d} joined {(q['joined'] & px).sum():3d} cloud "
              f"{(q['cloud'] & px).sum():3d} | z {med(q['raw']):+.3f} gni {med(q['gni']):+.3f} "
              f"B02 {med(b['B02']):.3f} B03 {med(b['B03']):.3f} B08 {med(b['B08']):.3f} B11 {med(b['B11']):.3f}")
    if px.sum() <= 40:
        print("  per pixel: E N cos_i shadow2m | " + " ".join(t[5:10] + t[13] for t in SCENES))
        for i, j in zip(*np.nonzero(px)):
            print(f"    {X[j]:.0f} {Y[i]:.0f} {COS_I[i, j]:.2f} {SHADE2[i, j]:.2f} | z27 {r['raw'][i, j]:+.3f} "
                  f"gni27 {r['gni'][i, j]:+.3f} B03 {r['bands']['B03'][i, j]:.3f} B08 {r['bands']['B08'][i, j]:.3f} "
                  f"B11 {r['bands']['B11'][i, j]:.3f} | " + " ".join(f"{letter(t, i, j):>6s}" for t in SCENES))

    # figure
    box = (e - half, n_ - half, e + half, n_ + half)
    f, axs = plt.subplots(3, 6, figsize=(6 * 3.0, 3 * 3.0), dpi=100)
    f.subplots_adjust(0, 0, 1, 1, 0.01, 0.01)
    ortho = np.moveaxis(read(ROOT / "data/ortho_0p25m.tif", box).astype(np.uint8), 0, -1)
    panels = ([("tc", t, 0.6) for t in VOTERS + ["2026-09-29_S2A_R065"]]
              + [("tc", t, 0.6) for t in HOLD[1:] + [HIGH_SUN[1]]]
              + [("z", t, None) for t in (MAIN, "2026-09-17_S2C_R065", HIGH_SUN[0], HIGH_SUN[1])]
              + [("tc", MAIN, 0.15), ("ortho", None, None)])
    for a, (kind, t, white) in zip(axs.flat, panels):
        if kind == "tc":
            b = res[t]["bands"]
            x = np.clip(np.nan_to_num(np.dstack([b["B04"], b["B03"], b["B02"]])) / white, 0, 1)
            a.imshow(np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055),
                     extent=[E0, E1, N0, N1], interpolation="nearest")
            label_ = f"{t[:10]} {t[11:14]}" + (" x4" if white < 0.6 else "")
        elif kind == "z":
            a.imshow(res[t]["raw"], extent=[E0, E1, N0, N1], interpolation="nearest", cmap="RdBu",
                     norm=TwoSlopeNorm(0.0, -0.5, 0.5))
            a.contour(X, Y, res[t]["z"], [0.0], colors="k", linewidths=0.8)
            label_ = f"index - threshold {t[:10]}"
        else:
            a.imshow(ortho, extent=[box[0], box[2], box[1], box[3]])
            label_ = "SWISSIMAGE 2024"
        if t is not None:
            a.contour(X, Y, res[t]["shade"].astype(float), [0.5], colors="#ffd400", linewidths=0.8,
                      linestyles=":")
            jj = np.nonzero(res[t]["joined"])
            a.plot(X[jj[1]], Y[jj[0]], ".", color="#ffd400", ms=3)
        for geom_, colour, lw, ls in ((G24, "#ff8c00", 0.8, "-"), (old, "#ffffff", 1.0, (0, (3, 2))),
                                      (new, "#ff1010", 1.0, "-")):
            for p in shapely.get_parts(geom_):
                for ring in [p.exterior, *p.interiors]:
                    xy = np.asarray(ring.coords)
                    a.plot(xy[:, 0], xy[:, 1], color=colour, lw=lw, ls=ls)
        a.text(box[0] + 4, box[3] - 4, label_, ha="left", va="top", fontsize=8, color="k",
               bbox=dict(facecolor="w", alpha=0.8, lw=0, pad=1.2))
        a.set_xlim(box[0], box[2]), a.set_ylim(box[1], box[3]), a.set_axis_off()
    f.savefig(PREVIEW / f"outline_2026_review_{name}.png")
    plt.close(f)


# --- validity, holes and polygons of 27 September with the votes of the voters and of the clear hold-outs
CLEAR = ["2026-09-29_S2A_R065", "2026-09-15_S2B_R108", "2026-09-12_S2A_R108"] + HIGH_SUN
ICE_H = sum((res[t]["raw"] > 0) & res[t]["sunlit"] & ~res[t]["cloud"] for t in CLEAR)
ROCK_H = sum((res[t]["raw"] <= 0) & res[t]["sunlit"] & ~res[t]["cloud"] for t in CLEAR)
print(f"\nvalid {shapely.is_valid(new)} ({shapely.is_valid_reason(new)}); parts {len(new.geoms)}, holes of the body "
      f"{len(new.geoms[0].interiors)}")
body_, *rest_ = r["parts"]
for kind, geoms in (("hole", [shapely.Polygon(h) for h in body_.interiors]), ("polygon", rest_)):
    for g in geoms:
        p, m = g.representative_point(), raster(g)
        print(f"  27 September {kind} {g.area:6.0f} m2 at {p.x:.0f} {p.y:.0f}: voters rock {ROCK[m].max(initial=0)} "
              f"ice {ICE[m].max(initial=0)}; clear hold-outs (of {len(CLEAR)}) rock {ROCK_H[m].max(initial=0)} "
              f"ice {ICE_H[m].max(initial=0)}; {'kept' if new.contains(p) == (kind == 'polygon') else 'not kept'}")

# --- joined groups of every scene: what they connect
for t in SCENES:
    q = res[t]
    pre_t = polygons(q["raw"])
    pid_t = rasterize([(p, k) for k, p in enumerate(pre_t, 1)], q["z"].shape, transform=T10, dtype="int32")
    gr, ng = label(q["joined"])
    out = []
    for g in range(1, ng + 1):
        grp = gr == g
        ids = sorted(set(pid_t[binary_dilation(grp) & ~grp]) - {0})
        ii, jj = np.nonzero(grp)
        small = [pre_t[k - 1] for k in ids if pre_t[k - 1].area < 2e5]
        out.append(f"{grp.sum()} px {X[jj].mean():.0f} {Y[ii].mean():.0f} -> " + ",".join(
            f"{p.area:.0f}({p.intersection(G24).area / p.area:.0%})" for p in small))
    print(f"{t}: {ng} groups; " + "; ".join(out))

# --- haze: sunlit bare ice of the 27 September body, 50 m inside its margin
core = raster(new.geoms[0].buffer(-50)) & ~r["shade"]
print("\nsunlit body ice, medians: scene, B02 B03 B08 B11, index, threshold sunlit")
for t in SCENES:
    b, q = res[t]["bands"], res[t]
    m = core & q["sunlit"] & ~q["cloud"]
    nd = (b["B03"] - b["B11"]) / (b["B03"] + b["B11"])
    print(f"  {t}: {np.nanmedian(b['B02'][m]):.3f} {np.nanmedian(b['B03'][m]):.3f} {np.nanmedian(b['B08'][m]):.3f} "
          f"{np.nanmedian(b['B11'][m]):.3f} | {np.nanmedian(nd[m]):.3f} | {q['t_sun']:.3f}")

# --- place 1: the hole against the sunlit ice 40-100 m around it; north-western patch against the body ice
hole1 = raster(shapely.difference(old, new).intersection(shapely.Point(2680668, 1140883).buffer(40)))
ring1 = raster(shapely.Point(2680668, 1140883).buffer(100)) & ~raster(shapely.Point(2680668, 1140883).buffer(40))
patch = raster(shapely.difference(new, old).intersection(shapely.Point(2679235, 1141150).buffer(40)))
print(f"\nplace 1 hole ({hole1.sum()} px) against ice 40-100 m around: scene, hole B03 B11 index | ice B03 B11 index"
      f" | B11 excess, B03 ratio; north-western patch ({patch.sum()} px): B02 B03 B04 B08 B11 index, classes")
for t in SCENES:
    b, q = res[t]["bands"], res[t]
    nd = (b["B03"] - b["B11"]) / (b["B03"] + b["B11"])
    ice1 = ring1 & q["sunlit"] & (q["raw"] > 0)
    h = [np.nanmedian(b[k][hole1]) for k in ("B03", "B11")] + [np.nanmedian(nd[hole1])]
    i_ = [np.nanmedian(b[k][ice1]) for k in ("B03", "B11")] + [np.nanmedian(nd[ice1])]
    scl = read(S2 / CAT[t]["file_scl"], band=1)
    scl = np.repeat(np.repeat(scl, 2, 0), 2, 1)
    cls = {int(k): int(v) for k, v in zip(*np.unique(scl[patch], return_counts=True))}
    print(f"  {t}: {h[0]:.3f} {h[1]:.3f} {h[2]:+.2f} | {i_[0]:.3f} {i_[1]:.3f} {i_[2]:+.2f} | {h[1] - i_[1]:+.3f} "
          f"{h[0] / i_[0]:.2f} || " + " ".join(f"{np.nanmedian(b[k][patch]):.3f}" for k in ("B02", "B03", "B04", "B08",
                                                                                          "B11"))
          + f" {np.nanmedian(nd[patch]):+.2f} {cls}")

# --- SWISSIMAGE 2024 at 0.25 m, 120 m around the changes, with the 10 m grid of Sentinel-2
ZOOM = {"place1": (2680668, 1140883), "place2": (2680415, 1141159), "place3": (2680330, 1141045),
        "gully": (2679800, 1141330), "northwest": (2679235, 1141150), "margin": (2680578, 1140457)}
f, axs = plt.subplots(2, 3, figsize=(3 * 5.0, 2 * 5.0), dpi=100)
f.subplots_adjust(0, 0, 1, 1, 0.01, 0.01)
for a, (name, (e, n_)) in zip(axs.flat, ZOOM.items()):
    box = (e - 60, n_ - 60, e + 60, n_ + 60)
    a.imshow(np.moveaxis(read(ROOT / "data/ortho_0p25m.tif", box).astype(np.uint8), 0, -1),
             extent=[box[0], box[2], box[1], box[3]])
    for v in np.arange(E0, E1 + 1, PIXEL):
        a.axvline(v, color="#ffd400", lw=0.3, alpha=0.6)
    for v in np.arange(N0, N1 + 1, PIXEL):
        a.axhline(v, color="#ffd400", lw=0.3, alpha=0.6)
    for geom_, colour, lw, ls in ((G24, "#ff8c00", 1.0, "-"), (old, "#ffffff", 1.2, (0, (3, 2))),
                                  (new, "#ff1010", 1.2, "-")):
        for p in shapely.get_parts(geom_):
            for ring in [p.exterior, *p.interiors]:
                xy = np.asarray(ring.coords)
                a.plot(xy[:, 0], xy[:, 1], color=colour, lw=lw, ls=ls)
    a.text(box[0] + 2, box[3] - 2, f"{name}, SWISSIMAGE 2024, 120 m", ha="left", va="top", fontsize=9,
           bbox=dict(facecolor="w", alpha=0.8, lw=0, pad=1.2))
    a.set_xlim(box[0], box[2]), a.set_ylim(box[1], box[3]), a.set_axis_off()
f.savefig(PREVIEW / "outline_2026_review_aerial.png")
plt.close(f)

# --- the joined strips without joins: does each scene's own iso-line (before its joins) already connect them?
print("\nstrips by the iso-line alone (largest polygon before the joins): scene, place-3 bay enclosed as a hole "
      "(its area), place-3 block part of it, gully top part of it")
BAY, BLOCK3, GULLY = shapely.Point(2680336, 1141003), shapely.Point(2680356, 1141050), shapely.Point(2679775, 1141360)
for t in SCENES:
    big = max(polygons(res[t]["raw"]), key=lambda p: p.area)
    bay = [shapely.Polygon(h) for h in big.interiors if shapely.Polygon(h).contains(BAY)]
    print(f"  {t}: bay {'hole ' + format(bay[0].area, '.0f') + ' m2' if bay else 'open'}, "
          f"block {'yes' if big.contains(BLOCK3) else 'no'}, gully {'yes' if big.contains(GULLY) else 'no'}")

# --- sensitivity to DROP (the open decision): the five voters redone, votes and step 6 included
print("\nDROP: 27 September area m2, joined pixels, place-3 bay closed, gully top joined; late-September spread m2, "
      "mean boundary distance m; joined pixels per voter")
for drop in (0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25):
    DROP = drop                                          # read by outline() at call time
    rd = {t: outline(t) for t in VOTERS}
    ice_d = sum((rd[t]["z"] > 0) & rd[t]["sunlit"] for t in VOTERS)
    rock_d = sum((rd[t]["z"] <= 0) & rd[t]["sunlit"] for t in VOTERS)
    gl = [glacier(rd[t], ice_d, rock_d) for t in VOTERS]
    body_d = gl[0].geoms[0]
    bay_d = any(shapely.Polygon(h).contains(BAY) for h in body_d.interiors)
    pairs_d = [boundary_distance(p, q) for i, p in enumerate(gl) for q in gl[i + 1:]]
    print(f"  {drop:.2f}: {gl[0].area:.0f}, {rd[MAIN]['joined'].sum()}, {'yes' if bay_d else 'no'}, "
          f"{'yes' if body_d.contains(GULLY) else 'no'}; {max(g.area for g in gl) - min(g.area for g in gl):.0f}, "
          f"{np.mean(pairs_d):.2f}; " + " ".join(str(rd[t]["joined"].sum()) for t in VOTERS))
DROP = 0.15
