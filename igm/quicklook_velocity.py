"""Static frames of the preview ice-flow run with surface velocity (preview only: calibration MAP draw, map thickness
unchanged u = 0, default ice softness f = 1; not yet fitted to the observed speed bound). Reads
runs/preview_vel/vel91_map/fields.npz (thickness and surface velocity every 30 September 1991-2021). Each panel:
hillshade of bed + ice, surface speed in colour (m per year) on ice of 1 m or more, arrows every 60 m showing the flow
direction and speed, the 1 m ice edge in white.
usage (s03, igm env):  python quicklook_velocity.py      output: results/quicklook_velocity.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent
d = np.load(R / "runs/preview_vel/vel91_map/fields.npz")
bed = d["topg"]


def shade(z):
    gy, gx = np.gradient(z, 10.0)
    sl, asp = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    az, el = np.radians(315), np.radians(40)
    return np.sin(el) * np.cos(sl) + np.cos(el) * np.sin(sl) * np.cos(az - asp)


years = (1991, 2001, 2011, 2021)
fig, axs = plt.subplots(2, 2, figsize=(15, 11))
step = 6
for ax, y in zip(axs.flat, years):
    thk, u, v = d[f"thk_y{y}"], d[f"us_y{y}"], d[f"vs_y{y}"]
    ice = thk >= 1.0
    sp = np.hypot(u, v)
    ax.imshow(shade(bed + thk), cmap="gray", vmin=-0.3, vmax=1.1, origin="lower")
    im = ax.imshow(np.ma.masked_where(~ice, sp), cmap="magma_r", vmin=0, vmax=8, alpha=0.85, origin="lower")
    ax.contour(ice, levels=[0.5], colors="white", linewidths=0.7, origin="lower")
    yy, xx = np.mgrid[0:thk.shape[0]:step, 0:thk.shape[1]:step]
    m = ice[::step, ::step] & (sp[::step, ::step] > 0.2)
    ax.quiver(xx[m], yy[m], u[::step, ::step][m], v[::step, ::step][m], color="#0b2f52", scale=120, width=0.0022,
              headwidth=3.5, headlength=4)
    ax.set_title(f"30 September {y}: median {np.median(sp[ice]):.1f}, fastest {sp[ice].max():.1f} m per year, "
                 f"{ice.sum() * 100 / 1e6:.2f} km2", fontsize=11)
    ax.set_xticks([]), ax.set_yticks([])
fig.colorbar(im, ax=axs, shrink=0.6, label="Surface speed (m per year)")
fig.suptitle("Ice-flow model preview (not yet fitted to observed speeds): surface speed and flow direction", fontsize=13)
fig.savefig(R / "results/quicklook_velocity.png", dpi=140)
