"""Quick look at the first IGM output of step 3 (preview only, not a result of the ensemble): the spin-up end state
(runs/s3_batch3/spinup_map) and the 1991-2021 run of the calibration's MAP draw with the map thickness unchanged
(u = 0, f = 1; runs/s3_batch3/fit91_map, thickness every 30 September). Panels: spin-up thickness, 1991, 2006, 2021,
plus an MP4 of 1991-2021. Hillshade from bed + ice; thickness in blue; the 1 m ice edge in white.
usage (s03, igm env):  python quicklook_step3.py      outputs: results/quicklook_step3.png, results/quicklook_1991_2021.mp4
"""
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent
run = R / "runs/s3_batch3"
sp = np.load(run / "spinup_map/fields.npz")
fit = np.load(run / "fit91_map/fields.npz")
bed = fit["topg"]


def shade(z):
    gy, gx = np.gradient(z, 10.0)
    sl, asp = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    az, el = np.radians(315), np.radians(40)
    return np.sin(el) * np.cos(sl) + np.cos(el) * np.sin(sl) * np.cos(az - asp)


def panel(ax, thk, title, vmax=60):
    ax.imshow(shade(bed + thk), cmap="gray", vmin=-0.3, vmax=1.1, origin="lower")
    t = np.ma.masked_less(thk, 1.0)
    im = ax.imshow(t, cmap="Blues", vmin=0, vmax=vmax, alpha=0.85, origin="lower")
    ax.contour(thk >= 1.0, levels=[0.5], colors="white", linewidths=0.6, origin="lower")
    area, vol = (thk >= 1).sum() * 100 / 1e6, thk.sum() * 100 / 1e9
    ax.set_title(f"{title}\n{area:.2f} km2 of ice, {vol:.3f} km3, thickest {thk.max():.0f} m", fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])
    return im


fig, axs = plt.subplots(2, 2, figsize=(14, 10))
panel(axs[0, 0], sp["thk_end"], "Spin-up end state (climate of 1865-1894)", vmax=120)
for a, y in zip((axs[0, 1], axs[1, 0], axs[1, 1]), (1991, 2006, 2021)):
    im = panel(a, fit[f"thk_y{y}"], f"Ice-flow run, 30 September {y}")
fig.colorbar(im, ax=axs, shrink=0.6, label="Ice thickness (m)")
fig.savefig(R / "results/quicklook_step3.png", dpi=130)

frames = R / "results/qf"
frames.mkdir(exist_ok=True)
for i, y in enumerate(range(1991, 2022)):
    f, a = plt.subplots(figsize=(12, 7))
    panel(a, fit[f"thk_y{y}"], f"Ghiacciaio del Basodino, ice-flow model, 30 September {y}")
    f.savefig(frames / f"f{i:02d}.png", dpi=110)
    plt.close(f)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "4", "-i", str(frames / "f%02d.png"), "-vf",
                "scale=1320:-2", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18",
                str(R / "results/quicklook_1991_2021.mp4")], check=True)
for p in frames.glob("*.png"):
    p.unlink()
frames.rmdir()
