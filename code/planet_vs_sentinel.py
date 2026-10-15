"""PlanetScope (3 m) next to Sentinel-2 (10 m) at Basodino: the PlanetScope test scene of 6 September 2026 and the
two nearest Sentinel-2 scenes without cloud over the glacier (7 September, hazy; 12 September, clear), all warped to
the same Swiss grid (2 m cells, nearest neighbour, so each sensor keeps its own pixel size) and drawn with one shared
stretch (reflectance / 0.9, gamma 2.2), whole glacier and a 300 m zoom on the rock window, with our 27 September
outline in red. Sentinel-2 from Earth Search (sentinel-2-c1-l2a, scale 0.0001, offset -0.1).
usage:  uv run --with rasterio --with numpy --with matplotlib --with pystac-client --with pyproj python code/planet_vs_sentinel.py
output: tex/preview/planet_vs_sentinel_2026-09.png
"""
import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pystac_client import Client
from rasterio.warp import Resampling, reproject
from rasterio.transform import from_origin

from common import ROOT

E0, N0, E1, N1, RES = 2678500, 1139900, 2682500, 1142400, 2.0
W, H = int((E1 - E0) / RES), int((N1 - N0) / RES)
dst_t = from_origin(E0, N1, RES, RES)


def warp(src, bands, scale, offset):
    out = np.zeros((3, H, W))
    for k, bnd in enumerate(bands):
        a = src.read(bnd).astype(float) if isinstance(src, rasterio.io.DatasetReader) else None
        reproject(a, out[k], src_transform=src.transform, src_crs=src.crs, dst_transform=dst_t, dst_crs="EPSG:2056",
                  resampling=Resampling.nearest)
    return out * scale + offset


panels = []
p = rasterio.open(glob.glob(str(ROOT / "data/planet/basodino_*/*/PSScene/*_SR_clip.tif"))[0])
panels.append(("PlanetScope, 6 September 2026, 3 m", warp(p, [3, 2, 1], 1e-4, 0.0)))
cat = Client.open("https://earth-search.aws.element84.com/v1")
for day, label in (("2026-09-07", "Sentinel-2, 7 September 2026 (hazy), 10 m"), ("2026-09-12", "Sentinel-2, 12 September 2026, 10 m")):
    items = list(cat.search(collections=["sentinel-2-c1-l2a"], bbox=[8.46, 46.40, 8.51, 46.44], datetime=day).items())
    it = sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))[0]
    rgb = np.zeros((3, H, W))
    for k, a in enumerate(("red", "green", "blue")):
        with rasterio.open(it.assets[a].href) as s:
            reproject(s.read(1).astype(float), rgb[k], src_transform=s.transform, src_crs=s.crs, dst_transform=dst_t,
                      dst_crs="EPSG:2056", resampling=Resampling.nearest)
    panels.append((label, rgb * 1e-4 - 0.1))

g = json.load(open(ROOT / "data/outline_2026/outline_2026-09-27.geojson"))["features"][0]["geometry"]
polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
fig, axs = plt.subplots(2, 3, figsize=(21, 10.5), gridspec_kw={"height_ratios": [1.25, 1]})
wx, wy = 2680676.5, 1140888.1
for j, (label, rgb) in enumerate(panels):
    img = np.clip(np.clip(rgb, 0, None) / 0.9, 0, 1) ** (1 / 2.2)
    for i, a in enumerate(axs[:, j]):
        a.imshow(img.transpose(1, 2, 0), extent=(E0, E1, N0, N1), interpolation="nearest")
        for poly in polys:
            for ring in poly:
                r = np.array(ring)
                a.plot(r[:, 0], r[:, 1], color="red", lw=0.7 if i == 0 else 1.2)
        a.set_xticks([]), a.set_yticks([])
        if i == 1:
            a.set_xlim(wx - 150, wx + 150), a.set_ylim(wy - 90, wy + 90)
    axs[0, j].set_title(label, fontsize=13)
axs[1, 0].set_ylabel("rock window, 300 m wide", fontsize=12)
fig.suptitle("Same Swiss 2 m grid, nearest neighbour, one stretch for all; red: our outline of 27 September 2026", fontsize=12)
fig.tight_layout()
out = ROOT / "tex/preview/planet_vs_sentinel_2026-09.png"
fig.savefig(out, dpi=130)
print(out)
