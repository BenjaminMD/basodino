"""Quick-look figures of the relief COGs, one 600 dpi PNG per product: the whole box (E 2676000-2686000, N 1137000-1147000,
averaged to at most 4000 x 4000, 2.5 m pixels) and a 1 km x 1 km crop at the product's native pixels around the glacier
tongue (E 2679700-2680700, N 1141100-1142100; for the provenance the 200 m corner of the data border at E 2678000
N 1145000), as two square panels with one colorbar at 80 % of the panel height.
Shading products grey (1 dark .. 255 light); slope magma, 0-70 degrees; sky-view factor and openness cividis;
elevation viridis; provenance in three flat colours. Nodata would show white (the seamless products have none).
usage: python relief_quicklook.py <relief dir> [font dir]      -> <relief dir>/quicklook/<product>.png"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from osgeo import gdal

gdal.UseExceptions()
BOX = (2676000, 2686000, 1137000, 1147000)              # west, east, south, north
TONGUE = (2679700, 2680700, 1141100, 1142100)
CORNER = (2677900, 2678100, 1144900, 1145100)
SHADE = dict(cmap="gray", vmin=1, vmax=255, label="Shading value (1 dark, 255 light)")
SVF = dict(cmap="cividis", vmin=0.4, vmax=1, label="Sky-view factor (visible fraction of the sky)", extend="min")
OPN = dict(cmap="cividis", vmin=50, vmax=100, label="Positive openness (degrees)", extend="both")
STYLE = {                                                # product name prefix: title, colour scale
    "hillshade_315_45": ("Hillshade, light from 315 degrees at 45 degrees elevation", SHADE),
    "hillshade_multidir": ("Multidirectional hillshade, light from 225 to 360 degrees at 45 degrees elevation", SHADE),
    "relief_composite": ("Composite relief: multidirectional shading, sky-view factor and slope darkening", SHADE),
    "slope": ("Slope", dict(cmap="magma_r", vmin=0, vmax=70, label="Slope (degrees)", extend="max")),
    "svf_r30m": ("Sky-view factor, radius 30 m, 32 directions", SVF),
    "svf_r150m": ("Sky-view factor, radius 150 m, 32 directions", SVF),
    "openness_r30m": ("Positive openness, radius 30 m, 32 directions", OPN),
    "openness_r150m": ("Positive openness, radius 150 m, 32 directions", OPN),
    "seamless_dem": ("Seamless elevation model, swissALTI3D and Regione Piemonte lidar, heights LN02",
                     dict(cmap="viridis", vmin=700, vmax=3300, label="Elevation (metres above sea level)")),
    "seamless_provenance": ("Source of the seamless elevation model", None),
}
PROV = [("swissALTI3D 2022, 0.5 m", "#d9d9d9"), ("Regione Piemonte DTM 5 m, 2009-2011", "#6baed6"),
        ("Blend zone, 25 m", "#f16913")]


def read(path, win, size, alg):
    ds = gdal.Translate("", path, format="MEM", projWin=[win[0], win[3], win[1], win[2]], width=size, height=size,
                        resampleAlg=alg)
    a = ds.ReadAsArray().astype(np.float32)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    if nd is not None:
        a[a == nd] = np.nan
    return a


def figure(path, out):
    name = os.path.basename(path)[:-4]
    title, st = next(v for k, v in STYLE.items() if name.startswith(k))
    res = gdal.Open(path).GetGeoTransform()[1]
    views = [("Whole box, 10 km x 10 km", BOX, min(4000, round(10000 / res))),
             (f"Glacier tongue, 1 km x 1 km, native {res:g} m pixels", TONGUE, round(1000 / res))]
    if st is None:                                       # provenance: zoom on the corner of the data border instead
        views[1] = ("Corner of the data border, 200 m x 200 m, native 0.5 m pixels", CORNER, 400)
    fig = plt.figure(figsize=(12.6, 6.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.22, left=0.06, right=0.92, bottom=0.08, top=0.88)
    for k, (label, win, size) in enumerate(views):
        a = read(path, win, size, "mode" if st is None else "average")
        ax = fig.add_subplot(gs[0, k])
        ext = tuple(v / 1000 for v in win)
        if st is None:
            im = ax.imshow(a, cmap=ListedColormap([c for _, c in PROV]), vmin=0.5, vmax=3.5, extent=ext,
                           interpolation="nearest")
        else:
            cm = plt.get_cmap(st["cmap"]).copy()
            cm.set_bad("white")
            im = ax.imshow(a, cmap=cm, vmin=st["vmin"], vmax=st["vmax"], extent=ext,
                           interpolation="nearest" if k == 1 else "antialiased")
        if k == 0:
            zw = views[1][1]
            ax.add_patch(Rectangle((zw[0] / 1000, zw[2] / 1000), (zw[1] - zw[0]) / 1000, (zw[3] - zw[2]) / 1000,
                                   fill=False, ec="#e41a1c", lw=1.2))
            ax.set_ylabel("North (kilometres)", fontsize=9)
            if st is None:
                ax.legend(handles=[Patch(color=c, label=l) for l, c in PROV], loc="upper right", fontsize=8)
        ax.set_xlabel("East (kilometres)", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.ticklabel_format(useOffset=False, style="plain")
    cax = fig.add_subplot(gs[0, 2])
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + 0.1 * pos.height, pos.width, 0.8 * pos.height])
    if st is None:
        cax.axis("off")
    else:
        cb = fig.colorbar(im, cax=cax, extend=st.get("extend", "neither"))
        cb.set_label(st["label"], fontsize=9)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle(title + (f", {res:g} m pixels" if name.startswith("slope") else ""), fontsize=12, fontweight=600)
    fig.savefig(os.path.join(out, name + ".png"), dpi=600)
    plt.close(fig)
    print(name, flush=True)


def main(d, fonts=None):
    if fonts and os.path.isdir(fonts):
        for f in sorted(glob.glob(os.path.join(fonts, "*.ttf"))):
            font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = "Source Sans 3"
    out = os.path.join(d, "quicklook")
    os.makedirs(out, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(d, "*.tif"))):
        if "_pad" not in os.path.basename(path):
            figure(path, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
