"""True-colour preview of a delivered PlanetScope surface-reflectance clip (4 bands: blue, green, red, near infrared),
with our 27 September 2026 outline in red and the rock window marked, plus a 3x zoom of the window.
usage:  uv run --with rasterio --with matplotlib --with numpy --with pyproj python code/planet_preview.py SR_CLIP.tif
output: tex/preview/planet_<scene id>.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer

from common import ROOT

src = rasterio.open(sys.argv[1])
rgb = src.read([3, 2, 1]).astype(float) / 10000.0                 # reflectance
img = np.clip((rgb / 0.9) ** (1 / 2.2), 0, 1).transpose(1, 2, 0)
to = Transformer.from_crs(2056, src.crs, always_xy=True)
g = json.load(open(ROOT / "data/outline_2026/outline_2026-09-27.geojson"))["features"][0]["geometry"]
polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
b = src.bounds
fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.6, 1]})
for a in (ax, bx):
    a.imshow(img, extent=(b.left, b.right, b.bottom, b.top))
    for p in polys:
        for ring in p:
            x, y = to.transform(*np.array(ring).T)
            a.plot(x, y, color="red", lw=0.8)
    a.set_xticks([]), a.set_yticks([])
wx, wy = to.transform(2680676.5, 1140888.1)
bx.set_xlim(wx - 150, wx + 150), bx.set_ylim(wy - 100, wy + 100)
ax.plot(wx, wy, "o", mfc="none", mec="yellow", ms=14)
sid = Path(sys.argv[1]).name.split("_3B")[0]
ax.set_title(f"PlanetScope {sid}, 3 m, true colour; red: our outline of 27 September 2026")
bx.set_title("rock window, 300 m wide")
fig.tight_layout()
out = ROOT / f"tex/preview/planet_{sid}.png"
fig.savefig(out, dpi=150)
print(out, src.res, src.count, src.shape)
