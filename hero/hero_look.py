"""The grey stretch of the quick-look the title-page art is based on, maps/quicklook/svf_r30m_0p5m_tongue_0p5m.png.

usage: python hero_look.py          (env agl on s02; writes checks/look.json and checks/quicklook_repro.png)

That PNG came from derived/relief/scripts/relief_quicklook.py (stage A): the stage A svf_r30m_0p5m.tif read with
gdal.Translate(projWin=TONGUE), then plt.imsave(cmap="gray", vmin=p1, vmax=p99) with p1, p99 = np.nanpercentile of that
2000 x 2000 crop. This script repeats exactly that, checks the result against the PNG pixel for pixel, and records
p1 and p99: every candidate uses this one stretch (linear, gray colormap, 256 levels), whatever its crop."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from osgeo import gdal

gdal.UseExceptions()
P = "/data/befahl/projects/AGL/basodino"
H = f"{P}/hero"
TONGUE = [2679700, 1142100, 2680700, 1141100]                    # ulx uly lrx lry, as relief_quicklook.py


def grey(a, lo, hi):
    """relief_quicklook.py's mapping (plt.imsave, cmap gray, vmin lo, vmax hi) as 8-bit grey."""
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(lo, hi), cmap=plt.get_cmap("gray"))
    return sm.to_rgba(a, bytes=True)[..., 0]


def main():
    src = gdal.Open(f"{P}/derived/relief/svf_r30m_0p5m.tif")
    a = gdal.Translate("", src, format="MEM", projWin=TONGUE).ReadAsArray().astype(np.float32)
    a[a == src.GetRasterBand(1).GetNoDataValue()] = np.nan
    lo, hi = (float(v) for v in np.nanpercentile(a, [1, 99]))
    plt.imsave(f"{H}/checks/quicklook_repro.png", a, cmap="gray", vmin=lo, vmax=hi)
    ref = np.asarray(Image.open(f"{H}/checks/quicklook_original.png"))
    rep = np.asarray(Image.open(f"{H}/checks/quicklook_repro.png"))
    g = grey(a, lo, hi)
    out = {"source": "derived/relief/svf_r30m_0p5m.tif (stage A), window E 2679700-2680700, N 1141100-1142100",
           "stretch": "linear, matplotlib gray, vmin = p1, vmax = p99 of the window", "p1": lo, "p99": hi,
           "shape": list(a.shape), "repro_equals_original_px_share": float((rep == ref).all(axis=-1).mean()),
           "grey_equals_original_red_px_share": float((g == ref[..., 0]).mean()),
           "original_grey_p1_p50_p99": [float(x) for x in np.percentile(ref[..., 0], [1, 50, 99])]}
    json.dump(out, open(f"{H}/checks/look.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
