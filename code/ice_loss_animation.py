"""Animation of Ghiacciaio del Basodino shrinking, end of each melt season 2018-2026, from code/last_ice.tif (the
last year each 10 m cell held ice in Sentinel-2 late-season scenes). Each frame shows the ice of that year in white
over a hillshade of the 2021 elevation model; cells that lost their ice that year are red for one frame.
Output: tex/animation/ice_loss_2018_2026.mp4 (and a GIF).
usage: uv run --with rasterio --with matplotlib --with numpy python code/ice_loss_animation.py
"""
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from common import COURSE, ROOT, read_esri_ascii

r = rasterio.open(ROOT / "code/last_ice.tif")
last = r.read(1).astype(int)
b = r.bounds
z, hdr = read_esri_ascii(COURSE / "dem_basodino_2021.asc")
x0, y0, cs = hdr["xllcorner"], hdr["yllcorner"], hdr["cellsize"]
gy, gx = np.gradient(z, cs)
az, el = np.radians(315), np.radians(40)
slope, aspect = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
shade = np.sin(el) * np.cos(slope) + np.cos(el) * np.sin(slope) * np.cos(az - aspect)
ext_dem = (x0, x0 + z.shape[1] * cs, y0, y0 + z.shape[0] * cs)
out = ROOT / "tex/animation/frames"
out.mkdir(parents=True, exist_ok=True)
years = list(range(2018, 2027))
for i, y in enumerate(years):
    fig = plt.figure(figsize=(12, 7), dpi=200, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(shade, extent=ext_dem, cmap="gray", vmin=-0.2, vmax=1.1)
    ice = (last >= y) & (last > 0)
    lost = last == y - 1
    rgba = np.zeros(last.shape + (4,))
    rgba[ice] = (0.86, 0.94, 1.0, 0.92)
    if y > 2018:
        rgba[lost] = (0.85, 0.12, 0.10, 0.95)
    ax.imshow(rgba, extent=(b.left, b.right, b.bottom, b.top), interpolation="nearest")
    ax.set_xlim(2678900, 2681600), ax.set_ylim(1140250, 1141650), ax.axis("off")
    area = ice.sum() * 100 / 1e6
    ax.text(0.02, 0.95, f"{y}", transform=ax.transAxes, color="white", fontsize=34, weight="bold", va="top")
    ax.text(0.02, 0.85, f"{area:.2f} km² of ice", transform=ax.transAxes, color="white", fontsize=18, va="top")
    if y > 2018:
        ax.text(0.02, 0.79, "red: ice gone since the year before", transform=ax.transAxes, color=(1, .45, .4), fontsize=13, va="top")
    ax.text(0.98, 0.03, "Sentinel-2 end of season, contains modified Copernicus Sentinel data 2018-2026",
            transform=ax.transAxes, color="0.75", fontsize=9, ha="right")
    fig.savefig(out / f"f{i:02d}.png", facecolor="black")
    plt.close(fig)
mp4 = ROOT / "tex/animation/ice_loss_2018_2026.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "1", "-i", str(out / "f%02d.png"), "-vf",
                "fps=30,scale=2400:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(mp4)], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "1", "-i", str(out / "f%02d.png"), "-vf",
                "scale=1200:-2", str(mp4.with_suffix(".gif"))], check=True)
print(mp4)
