"""Preview of the step-1 geometry grids: one PNG with six maps on the lattice of inputs.py.

Panels: 2021 surface with the outlines 1850, 1973, 2016, 2024, 2026; ice thickness 2021 at u = 0 (decision 8);
positive-balance cells (decision 2); annual mean potential direct radiation (ipot.py); curvature and the factor D
(redist.py). Map axes in kilometres of LV95.
Inputs: s02 igm/inputs/*.tif. Output: s02 igm/results/preview_geometry.png.
usage (s02, agl env):  python igm/preview_geometry.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from inputs import E0, E1, N0, N1, OUT, RES, read

EXT = (E0 / 1e3, E1 / 1e3, N0 / 1e3, N1 / 1e3)


def main():
    s21, m21 = read("usurf_2021"), read("mask_2021")
    with rasterio.open(OUT / "ipot.tif") as r:
        ip = r.read().mean(axis=0)
    thk = np.where(m21 == 1, s21 - read("bed_u0"), np.nan)
    D = np.where(read("posbal") == 1, read("D"), np.nan)
    panels = [("Surface 2021 and outlines", s21, "gray", "Elevation (m)"),
              ("Thickness 2021, u = 0", thk, "viridis", "Thickness (m)"),
              ("Positive-balance cells", read("posbal").astype(float), "Blues", ""),
              ("Potential direct radiation, annual mean", ip, "magma", "Irradiance (W per square metre)"),
              ("Curvature, 50 metre smoothing", read("kappa"), "PuOr_r", "Curvature (per metre)"),
              ("Redistribution factor on positive-balance cells", D, "cividis", "Factor")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for ax, (title, z, cmap, label) in zip(axs.ravel(), panels):
        lim = np.nanpercentile(np.abs(z), 98) if cmap == "PuOr_r" else None
        im = ax.imshow(z, extent=EXT, cmap=cmap, vmin=-lim if lim else None, vmax=lim)
        for name, color in (("sgi1850", "tab:blue"), ("sgi1973", "tab:green"), ("sgi2016", "tab:orange"),
                            ("tlm2024", "tab:red"), ("s2_2026", "black")):
            ax.contour(np.flipud(read(name)), [0.5], colors=color, linewidths=0.7, extent=EXT, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        if label:
            fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    fig.savefig(RES / "preview_geometry.png", dpi=110)


if __name__ == "__main__":
    main()
