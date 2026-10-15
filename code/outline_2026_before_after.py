"""The September 2026 outline before and after the rule change of 2026-10-06 (steps 5 and 6 of outline_2026.py:
narrow strips joined by the 10 m bands; holes and detached blocks by the votes of the five late-September scenes),
at the three places of the review and the other places that changed. A check, not part of the report chain.

Before: the rules of commit 10a3d87 rebuilt from outline_2026.py's own steps 1-4 (its code above '# --- the outline
and its checks' runs here): the iso-line without the joins, the largest polygon, holes of 1000 m2 or more; checked on
stdout against that commit's area, 1015377 m2. After: data/outline_2026/outline_2026-09-27.geojson as written by
outline_2026.py. stdout: both areas and every part added or removed from 50 m2, with position.

Images (not for the report), tex/preview/outline_2026_fix_<place>.png: 27 September at 10 m pixels, 300 m across:
true colour with the outline before, true colour with the outline after, the snow index (NDSI, white at the sunlit
threshold) and the green and near-infrared index (GNI, white at its threshold) with the outline after; the 2024
outline orange. <place>: place1 (rock spot), place2 (ice block), place3 (ice strip), gully, northwest (ice patch),
tongue (hole at the south wall); overview: the whole glacier, before white dashed, after red.

usage, after outline_2026.py, in the same mirror (its docstring):
  rsync -a code/outline_2026_before_after.py s02:/data/befahl/projects/AGL/basodino/report/code/
  ssh s02 'source ~/miniconda3/bin/activate agl && cd /data/befahl/projects/AGL/basodino/report &&
           python code/outline_2026_before_after.py'
  then copy tex/preview/outline_2026_fix_*.png back
"""
from pathlib import Path

exec((Path(__file__).resolve().parent / "outline_2026.py").read_text(encoding="utf-8")
     .split("# --- the outline and its checks")[0])

from matplotlib.colors import TwoSlopeNorm  # noqa: E402

PLACES = {"place1": (2680670, 1140886), "place2": (2680419, 1141161), "place3": (2680345, 1141055),
          "gully": (2679800, 1141330), "northwest": (2679235, 1141150), "tongue": (2680915, 1140791)}
HALF = 150.0                                                                  # m

r = outline(MAIN)
b = r["bands"]
old = max(polygons(r["z"] - DROP * r["joined"]), key=lambda p: p.area)
old = shapely.Polygon(old.exterior, [h for h in old.interiors if shapely.Polygon(h).area >= 1000.0])
new = gpd.read_file(OUT / f"outline_{MAIN[:10]}.geojson").geometry.iloc[0]
print(f"before {old.area:.0f} m2 (commit 10a3d87: 1015377), after {new.area:.0f} m2, main body "
      f"{max(p.area for p in shapely.get_parts(new)):.0f} m2")
for what, geom in (("added", shapely.difference(new, old)), ("removed", shapely.difference(old, new))):
    for p in sorted(shapely.get_parts(geom), key=lambda p: -p.area):
        if p.area >= 50:
            print(f"   {what} {p.area:6.0f} m2 at {p.representative_point().x:.0f} {p.representative_point().y:.0f}")

ndsi = (b["B03"] - b["B11"]) / (b["B03"] + b["B11"])
gni = (b["B03"] - b["B08"]) / (b["B03"] + b["B08"])
t_gni = otsu(gni[raster(G24.buffer(BUFFER)) & ~r["shade"]])
rgb = np.clip(np.nan_to_num(np.dstack([b["B04"], b["B03"], b["B02"]])) / 0.6, 0, 1) ** (1 / 2.2)


def draw(a, geom, colour, lw, ls="-"):
    for p in shapely.get_parts(geom):
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            a.plot(xy[:, 0], xy[:, 1], color=colour, lw=lw, ls=ls)


plt = figure_style()
for name, (e, n) in PLACES.items():
    box = (e - HALF, n - HALF, e + HALF, n + HALF)
    f, axs = plt.subplots(1, 4, figsize=(4 * 3.6, 3.6), dpi=100)
    f.subplots_adjust(0, 0, 1, 1, 0.01, 0)
    for i, (a, img, kw, geom, label) in enumerate((
            (axs[0], rgb, {}, old, "27 September, before"), (axs[1], rgb, {}, new, "27 September, after"),
            (axs[2], ndsi, dict(cmap="RdBu", norm=TwoSlopeNorm(r["t_sun"], -0.5, 1.0)), new, "snow index"),
            (axs[3], gni, dict(cmap="RdBu", norm=TwoSlopeNorm(t_gni, -0.15, 0.3)), new, "green and near infrared"))):
        a.imshow(img, extent=[E0, E1, N0, N1], interpolation="nearest", **kw)
        draw(a, G24, "#ff8c00", 0.8)
        draw(a, geom, "#ff1010" if i < 2 else "#000000", 1.3)
        a.text(box[0] + 5, box[3] - 5, label, ha="left", va="top", fontsize=9, color="k",
               bbox=dict(facecolor="w", alpha=0.8, lw=0, pad=1.5))
        a.set_xlim(box[0], box[2]), a.set_ylim(box[1], box[3]), a.set_axis_off()
    f.savefig(PREVIEW / f"outline_2026_fix_{name}.png")
    plt.close(f)

box = (2678900, 1140300, 2681500, 1141600)
f, a = plt.subplots(figsize=((box[2] - box[0]) / 200, (box[3] - box[1]) / 200), dpi=100)    # 0.5 px per m
f.subplots_adjust(0, 0, 1, 1)
a.imshow(rgb, extent=[E0, E1, N0, N1], interpolation="nearest")
draw(a, G24, "#ff8c00", 0.8), draw(a, new, "#ff1010", 1.2), draw(a, old, "#ffffff", 1.0, (0, (3, 2)))
for name, (e, n) in PLACES.items():
    a.text(e + 60, n + 60, name, color="#ffd400", fontsize=11, fontweight=600)
a.set_xlim(box[0], box[2]), a.set_ylim(box[1], box[3]), a.set_axis_off()
f.savefig(PREVIEW / "outline_2026_fix_overview.png")
plt.close(f)
