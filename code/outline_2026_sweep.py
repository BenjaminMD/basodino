"""Places where the September 2026 outline disagrees with the five late-September scenes: non-ice holes it fills,
ice blocks it drops, ice bridges it cuts; and the support of the holes it keeps. A check of outline_2026.py, not part
of the report.

Status: written for and run with the outline_2026.py of commit 10a3d87 (MIN_HOLE; the key 'body' of outline()). It
does not run on the outline_2026.py of the rule change of 2026-10-06 (joins, votes, MIN_PART); its outputs
code/outline_2026_sweep.csv and tex/preview/outline_2026_sweep_*.png describe the 10a3d87 outline.

Steps 1-4 of outline_2026.py (normalised difference snow index, cast shadow, the two Otsu thresholds of each scene)
come from that script itself: its code above "# --- the outline and its checks" runs here and writes nothing. The ice
of each scene is outlined as in its step 5 (iso-line between pixel centres), but over the 2024 outline plus 100 m
instead of 50 m and keeping every polygon and every hole; the largest polygon is the scene's glacier body. Compared
with the outline O, data/outline_2026/outline_2026-09-27.geojson (holes of 1000 m2 or more kept):
  hole       a hole of a scene's body inside O's ice that does not touch a hole O keeps
  block      an ice polygon of a scene apart from its body, mostly outside O
  bridge     a hole of a scene's body mostly outside O: that scene's ice closes an embayment of O, at the scene's
             threshold or at the threshold lowered by one of LOWER (0.05 and 0.10 in the index); the other kind of
             bridge joins a block to the body (columns joined, lowered)
  kept hole  each hole O keeps, against the holes of the scenes' bodies
Shapes of the five scenes closer than 10 m are one place. Per place:
  scenes         late-September scenes (of five) that show it at their thresholds: a hole there; ice there, joined
                 to the body or not; the closed embayment; a hole of the body overlapping the kept hole (areas_m2:
                 the areas of those holes)
  lowered        scenes joined at the threshold, at it lowered by 0.05 and by 0.10 (blocks: the best path to the body,
                 link, lies above the lowered threshold; bridges: the scene closes the embayment); in brackets in
                 which the scenes that close it at some level
  joined         blocks: scenes in which the block is part of the body
  area_m2        on 27 September (iso-line), else the median over the scenes that show it; bridges: the strip of ice
                 outside O within 20 m of the closed embayment (median), with enclosed_m2 and width_m (2 area /
                 perimeter); areas_m2 per scene in the order 27, 25, 22 A, 22 B, 20 September; cells_27 pixel
                 centres of 27 September on the wrong side
  margin_27      index minus threshold on 27 September: lowest cell of a hole, highest of a block, median of a strip
                 or a kept hole; margin_median the median over the scenes that show it
  link           blocks: per scene the lowest index minus threshold on the best path to the body (pixel centres, eight
                 neighbours, a diagonal step also takes the mean of its four cells as the iso-line does at a saddle;
                 positive = joined); link_sharp_27 the same with B11 sharpened to 10 m by B08 (B11 of the 20 m pixel x
                 B08 / B08 of the 20 m pixel); cut_m_27 the path below the threshold on 27 September, cut_at its
                 lowest cell
  gap_m          distance to O (holes: to O's margin); in_2024, in_region, other_glacier shares inside the 2024
                 outline, inside it plus 50 m (the region of outline_2026.py), inside another glacier's 2024 outline
                 plus 50 m (Basodino Nord is 98 m away)
  shadow, dark   shares in the cast shadow of 27 September as modelled, and in deep shadow as observed (B02 and B08
                 below 0.06); b08 median near-infrared reflectance (water below 0.11)
  sep12 aug13 y2025 y2024   the same test in 12 September and 13 August 2026, 7 September 2025, 10 September 2024

Then, on stdout only, the five outlines again with the observed deep shadow added to the modelled cast shadow (both
Otsu thresholds recomputed): area and kept holes per scene, to tell holes and gaps tied to the edge of the modelled
shadow from rock.

Outputs: code/outline_2026_sweep.csv, one row per place, and stdout; check images, not for the report, in tex/preview/:
  outline_2026_sweep_map.png    the places numbered on 27 September (holes red, blocks blue, bridges green, kept
                                holes magenta)
  outline_2026_sweep_<n>.png    five places per sheet: 27 September true colour (10 m cells) with the ice of the five
                                scenes (27 September magenta, the others cyan), index minus threshold of 27 September,
                                13 August 2026, 7 September 2025, SWISSIMAGE 2024 at 0.25 m; O red (black on the
                                index), 2024 orange, the place yellow
usage, where outline_2026.py runs (its docstring), from the report mirror on s02 (about 20 s):
  rsync -a code/common.py code/outline_2026.py code/outline_2026_sweep.py \
        s02:/data/befahl/projects/AGL/basodino/report/code/
  ssh s02 'source ~/miniconda3/bin/activate agl && cd /data/befahl/projects/AGL/basodino/report &&
           python code/outline_2026_sweep.py'
  then copy code/outline_2026_sweep.csv and tex/preview/outline_2026_sweep_*.png back
"""
import heapq
from pathlib import Path

exec((Path(__file__).resolve().parent / "outline_2026.py").read_text(encoding="utf-8")
     .split("# --- the outline and its checks")[0])

SWEEP, MERGE, MIN_ICE, NEAR = 100.0, 10.0, 25.0, 200.0   # m, m, m2 of overlap that counts, m searched for a link
LOWER, DARK = (0.05, 0.10), 0.06                         # index; reflectance of observed deep shadow
LATE5, EXTRA = [MAIN] + LATE, HIGH_SUN + [YEAR_BEFORE, CHECK_2024]
O = gpd.read_file(OUT / f"outline_{MAIN[:10]}.geojson").geometry.iloc[0]
FILLED, KEPT = shapely.Polygon(O.exterior), shapely.union_all([shapely.Polygon(h) for h in O.interiors])
NEIGHBOURS = gpd.read_file(ROOT / "data/vectors.gpkg", layer="glacier_2024").query("sgi != 'C14-10'").union_all()
REGION, ZONE = G24.buffer(BUFFER), G24.buffer(SWEEP)
INSIDE, IN_ZONE = raster(G24), raster(ZONE)


def ice(tag):
    """Ice of a scene over the sweep zone: body, other polygons, index minus threshold (also with B11 sharpened)."""
    r = outline(tag)
    b, shade = r["bands"], r["shade"]
    v20 = read(S2 / CAT[tag]["file_20m"], band=1)
    nir20 = b["B08"].reshape(len(Y) // 2, 2, len(X) // 2, 2).mean(axis=(1, 3))
    sharp = np.kron(np.where(v20 == 0, np.nan, v20 / 1e4) / nir20, np.ones((2, 2))) * b["B08"]
    t = np.where(shade & np.isfinite(r["t_shade"]), r["t_shade"], r["t_sun"])
    off = ~IN_ZONE | (shade & ~INSIDE & np.isfinite(r["t_shade"]))
    z, z_sharp = [np.where(off, -1.0, np.nan_to_num((b["B03"] - b11) / (b["B03"] + b11) - t, nan=-1.0))
                  for b11 in (b["B11"], sharp)]
    parts = sorted(polygons(z), key=lambda p: -p.area)
    return dict(body=parts[0], blocks=parts[1:], ice=shapely.union_all(parts), z=z, z_sharp=z_sharp, shade=shade,
                bands=b, method=r["body"])


def holes_of(p):
    return [shapely.Polygon(h) for h in p.interiors]


def body_at(t, d):
    """Scene t's body at its threshold lowered by d."""
    return res[t]["body"] if d == 0 else max(polygons(res[t]["z"] + d), key=lambda p: p.area)


def cells(geom):
    """Cells whose centres lie in geom, else the cell of its representative point."""
    m = raster(geom)
    if not m.any():
        p = geom.representative_point()
        m[int((N1 - p.y) // PIXEL), int((p.x - E0) // PIXEL)] = True
    return m


def shows(t, kind, place, core):
    """Does scene t show the place at its threshold: a hole there (not one O keeps), ice there, a closed embayment."""
    need = min(MIN_ICE, place.area / 2)
    if kind == "block":
        return res[t]["ice"].intersection(place).area >= need
    hs = [h for h in holes_of(res[t]["body"]) if h.intersection(core).area >= need]
    if kind == "hole":
        return any(not h.intersection(KEPT).area for h in hs)
    return any(h.difference(FILLED).area > h.area / 2 for h in hs)


def link(z, start, goal):
    """Path from the cells `start` to the cells `goal` whose lowest z is highest: that z, the path's cells below 0
    and its lowest cell."""
    best, prev = np.full(z.shape, -np.inf), {}
    heap = [(-z[i], i) for i in zip(*np.nonzero(start))]
    for v, i in heap:
        best[i] = -v
    heapq.heapify(heap)
    while heap:
        v, (r, c) = heapq.heappop(heap)
        if -v < best[r, c]:
            continue
        if goal[r, c]:
            break
        for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < z.shape[0] and 0 <= cc < z.shape[1]:
                w = min(-v, z[rr, cc], (z[r, c] + z[rr, cc] + z[r, cc] + z[rr, c]) / 4 if dr and dc else np.inf)
                if w > best[rr, cc]:
                    best[rr, cc], prev[rr, cc] = w, (r, c)
                    heapq.heappush(heap, (-w, (rr, cc)))
    path = [(r, c)]
    while path[-1] in prev:
        path.append(prev[path[-1]])
    return -v, [p for p in path if z[p] <= 0], min(path, key=lambda p: z[p])


def link_near(t, place, key="z"):
    """link() from scene t's body to the place, within NEAR of the place; cells back on the grid of the window."""
    e0, n0, e1, n1 = place.buffer(NEAR).bounds
    rows = slice(max(0, int((N1 - n1) // PIXEL)), int((N1 - n0) // PIXEL) + 1)
    cols = slice(max(0, int((e0 - E0) // PIXEL)), int((e1 - E0) // PIXEL) + 1)
    start, goal = raster(res[t]["body"])[rows, cols], cells(place)[rows, cols]
    if not start.any():
        return np.nan, [], None
    v, cut, low = link(res[t][key][rows, cols], start & ~goal if (start & ~goal).any() else start, goal)
    return v, [(r + rows.start, c + cols.start) for r, c in cut], (low[0] + rows.start, low[1] + cols.start)


def day(t):
    return t[8:10] + (t[13] if t[8:10] == "22" else "")


def describe(kind, place, show, area, margin):
    """The columns every kind of place has."""
    late, c, b = [t for t in LATE5 if show[t]], cells(place), res[MAIN]["bands"]
    wrong = res[MAIN]["z"] > 0 if kind == "block" else res[MAIN]["z"] <= 0
    return dict(kind=kind, east=round(place.representative_point().x), north=round(place.representative_point().y),
                scenes=len(late), which=" ".join(day(t) for t in late), lowered="", joined="",
                area_m2=round(area[MAIN] if show[MAIN] else float(np.median([area[t] for t in late])) if late else 0),
                areas_m2=" ".join(f"{area[t]:.0f}" for t in LATE5), cells_27=int((c & wrong).sum()),
                margin_27=round(margin[MAIN], 2),
                margin_median=round(float(np.median([margin[t] for t in late or LATE5])), 2),
                link="", link_sharp_27="", cut_m_27="", cut_at="", enclosed_m2="", width_m="",
                gap_m=round(place.distance(O if kind == "block" else O.exterior), 1),
                in_2024=round(place.intersection(G24).area / place.area, 2),
                in_region=round(place.intersection(REGION).area / place.area, 2),
                other_glacier=round(place.intersection(NEIGHBOURS.buffer(BUFFER)).area / place.area, 2),
                shadow=round(float(res[MAIN]["shade"][c].mean()), 2),
                dark=round(float(((b["B08"] < DARK) & (b["B02"] < DARK))[c].mean()), 2),
                b08=round(float(np.nanmedian(b["B08"][c])), 3), sep12=int(show[HIGH_SUN[0]]),
                aug13=int(show[HIGH_SUN[1]]), y2025=int(show[YEAR_BEFORE]), y2024=int(show[CHECK_2024]), geom=place)


res = {t: ice(t) for t in LATE5 + EXTRA}
print(f"27 September: outline() against the geojson {shapely.symmetric_difference(res[MAIN]['method'], O).area:.0f} m2")

# --- shapes of the five scenes that disagree with O: (kind, scene, shape, lowering of the threshold)
found = []
for t in LATE5:
    found += [("hole", t, h, 0.0) for h in holes_of(res[t]["body"])
              if not h.intersection(KEPT).area and h.intersection(O).area > h.area / 2]
    found += [("block", t, p, 0.0) for p in res[t]["blocks"] if p.difference(O).area > p.area / 2]
    for d in (0.0, *LOWER):
        found += [("bridge", t, h, d) for h in holes_of(body_at(t, d)) if h.difference(FILLED).area > h.area / 2]

rows = []
for kind in ("hole", "block", "bridge"):
    items = [(t, g, d) for k, t, g, d in found if k == kind]
    for core in shapely.get_parts(shapely.union_all([g.buffer(MERGE / 2) for _, g, _ in items])) if items else []:
        members = [(t, g, d) for t, g, d in items if g.buffer(MERGE / 2).intersects(core)]
        place = shapely.union_all([g for _, g, _ in members])
        show = {t: any(tt == t and d == 0 for tt, _, d in members) or shows(t, kind, place, core)
                for t in LATE5 + EXTRA}
        if kind == "hole":
            area = {t: sum(h.intersection(core).area for h in holes_of(res[t]["body"]) if not h.intersection(KEPT).area)
                    for t in LATE5}
            r = describe(kind, place, show, area, {t: float(res[t]["z"][cells(core)].min()) for t in LATE5})
        elif kind == "block":
            area = {t: res[t]["ice"].intersection(place).area for t in LATE5}
            r = describe(kind, place, show, area, {t: float(res[t]["z"][cells(place)].max()) for t in LATE5})
            links = [link_near(t, place)[0] for t in LATE5]
            _, cut, low = link_near(MAIN, place)
            r.update(joined=sum(res[t]["body"].intersection(place).area >= min(MIN_ICE, place.area / 2) for t in LATE5),
                     lowered=" ".join(str(sum(v > -d for v in links)) for d in (0.0, *LOWER)),
                     link=" ".join(f"{v:+.2f}" for v in links),
                     link_sharp_27=round(link_near(MAIN, place, "z_sharp")[0], 2), cut_m_27=round(PIXEL * len(cut)),
                     cut_at=f"{X[low[1]]:.0f}/{Y[low[0]]:.0f}" if cut else "")
        else:    # the place becomes the strips of ice outside O within 20 m of the closed embayments
            first = {t: min(d for tt, _, d in members if tt == t) for t, _, _ in members}
            strip = {t: shapely.union_all([body_at(t, first[t]).difference(FILLED).intersection(g.buffer(20))
                                           for tt, g, d in members if tt == t and d == first[t]]) for t in first}
            if sum(s.area for s in strip.values()):
                place = shapely.union_all(list(strip.values()))
            area = {t: strip[t].area if t in strip else 0.0 for t in LATE5}
            r = describe(kind, place, show, area, {t: float(np.median(res[t]["z"][cells(place)])) for t in LATE5})
            r.update(lowered=" ".join(str(sum(v <= d for v in first.values())) for d in (0.0, *LOWER)),
                     area_m2=round(float(np.median([s.area for s in strip.values()]))),
                     which=r["which"] + f" ({' '.join(day(t) for t in LATE5 if t in first)})",
                     enclosed_m2=round(float(np.median([g.area for _, g, _ in members]))),
                     width_m=round(float(np.median([2 * s.area / max(s.length, 1) for s in strip.values()])), 1))
        rows.append(r)

for h in holes_of(O):
    there = {t: [x for x in holes_of(res[t]["body"]) if x.intersection(h.buffer(MERGE / 2)).area >= MIN_ICE]
             for t in LATE5 + EXTRA}
    r = describe("kept hole", h, {t: bool(there[t]) for t in there}, {t: sum(x.area for x in there[t]) for t in there},
                 {t: float(np.median(res[t]["z"][cells(h)])) for t in LATE5})
    rows.append(r | dict(area_m2=round(h.area)))

ORDER = {"hole": 0, "block": 1, "bridge": 2, "kept hole": 3}
rows.sort(key=lambda r: (ORDER[r["kind"]], -r["scenes"], [-int(n) for n in r["lowered"].split()], -r["area_m2"]))
for i, r in enumerate(rows):
    r["id"] = i + 1
keys = ["id"] + [k for k in rows[0] if k not in ("id", "geom")]
write_csv("outline_2026_sweep.csv", keys, [[r[k] for k in keys] for r in rows])
print(*keys, sep="  ")
for r in rows:
    print(*[r[k] for k in keys], sep="  ")

# --- the five outlines with the observed deep shadow added to the cast shadow (thresholds recomputed): area, kept holes
region = raster(REGION)
for t in LATE5:
    b = res[t]["bands"]
    for name, shade in (("cast shadow", res[t]["shade"]),
                        ("cast or observed", res[t]["shade"] | ((b["B02"] < DARK) & (b["B08"] < DARK)))):
        ndsi = (b["B03"] - b["B11"]) / (b["B03"] + b["B11"])
        t_sun, t_shade = otsu(ndsi[region & ~shade]), otsu(ndsi[INSIDE & shade])
        z = np.where(~region | (shade & ~INSIDE), -1.0, ndsi - np.where(shade, t_shade, t_sun))
        body = max(polygons(z), key=lambda p: p.area)
        kept = [h for h in holes_of(body) if h.area >= MIN_HOLE]
        print(f"{t} {name}: thresholds {t_sun:.3f} {t_shade:.3f}, area {shapely.Polygon(body.exterior).area - sum(h.area for h in kept):.0f} m2, "
              "kept holes " + ", ".join(f"{h.area:.0f} m2 at {h.representative_point().x:.0f}/{h.representative_point().y:.0f}" for h in kept))


# --- check images, not for the report
def rgb(b):
    return np.clip(np.nan_to_num(np.dstack([b["B04"], b["B03"], b["B02"]])) / 0.6, 0, 1) ** (1 / 2.2)


def draw(a, geom, colour, lw, ls="-"):
    for p in shapely.get_parts(geom):
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            a.plot(xy[:, 0], xy[:, 1], color=colour, lw=lw, ls=ls)


plt = figure_style()
KIND = {"hole": "#ff3030", "block": "#2080ff", "bridge": "#20c040", "kept hole": "#ff40ff"}
f, a = plt.subplots(figsize=(15, 8), dpi=150)
f.subplots_adjust(0, 0, 1, 1)
a.imshow(rgb(res[MAIN]["bands"]), extent=[E0, E1, N0, N1], interpolation="nearest")
draw(a, G24, "#ff8c00", 0.8), draw(a, O, "#ff1010", 0.9), draw(a, ZONE, "#ffffff", 0.6, ":")
for r in rows:
    a.plot(r["east"], r["north"], "o", mfc="none", mec=KIND[r["kind"]], ms=9, mew=1.5)
    a.text(r["east"] + 18, r["north"] + 18, str(r["id"]), color=KIND[r["kind"]], fontsize=9, fontweight=600)
e0, n0, e1, n1 = ZONE.bounds
a.set_xlim(e0, e1), a.set_ylim(n0, n1), a.set_axis_off()
f.savefig(PREVIEW / "outline_2026_sweep_map.png")
plt.close(f)

HALF, PER = 160.0, 5                                                          # m, places per sheet
for page in range(0, len(rows), PER):
    sel = rows[page:page + PER]
    f, axs = plt.subplots(len(sel), 5, figsize=(5 * 2.6, len(sel) * 2.6), dpi=100, squeeze=False)
    f.subplots_adjust(0, 0, 1, 1, 0.02, 0.02)
    for row, r in zip(axs, sel):
        e, n = r["east"], r["north"]
        box = (e - HALF, n - HALF, e + HALF, n + HALF)
        ortho = np.moveaxis(read(ROOT / "data/ortho_0p25m.tif", box)[:3], 0, -1).astype(np.uint8)
        for a, img, kw in ((row[0], rgb(res[MAIN]["bands"]), {}),
                           (row[1], res[MAIN]["z"], dict(cmap="RdBu", vmin=-0.5, vmax=0.5)),
                           (row[2], rgb(res[HIGH_SUN[1]]["bands"]), {}),
                           (row[3], rgb(res[YEAR_BEFORE]["bands"]), {}),
                           (row[4], ortho, {})):
            a.imshow(img, extent=[E0, E1, N0, N1] if a is not row[4] else [box[0], box[2], box[1], box[3]],
                     interpolation="nearest", **kw)
            if a is row[0]:
                for t in LATE:
                    draw(a, res[t]["ice"], "#40e0d0", 0.5)
                draw(a, res[MAIN]["ice"], "#ff40ff", 0.7)
            draw(a, G24, "#ff8c00", 0.8)
            draw(a, O, "#000000" if a is row[1] else "#ff1010", 1.0)
            draw(a, r["geom"], "#ffd400", 0.9, "--")
            a.set_xlim(box[0], box[2]), a.set_ylim(box[1], box[3]), a.set_axis_off()
        label = f"{r['id']} {r['kind']} {r['scenes']}/5" + (f", joined {r['lowered']}" if r["lowered"] else "")
        row[0].text(0.03, 0.97, label, transform=row[0].transAxes, color="white", fontsize=11, va="top",
                    fontweight=600)
    f.savefig(PREVIEW / f"outline_2026_sweep_{page // PER + 1}.png")
    plt.close(f)
