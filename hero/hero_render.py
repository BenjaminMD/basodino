"""Title-page art candidates: the sky-view factor (radius 30 m, 32 directions) of the destriped, despiked 0.5 m DEM,
cut to the 210 x 88 mm slot of \\herofigure at 600 dpi (4961 x 2079 px), grey stretch of the quick-look.

usage: python hero_render.py <svf.tif> <outdir> [id ...]        (env agl on s02; default: every frame)

Tone    one stretch for every candidate, the quick-look's own (checks/look.json): linear between LO and HI, the 1st and
        99th percentile of the stage A sky-view factor on the quick-look window, through matplotlib's gray colormap
        (256 levels), exactly as maps/quicklook/svf_r30m_0p5m_tongue_0p5m.png was written.
Frames  (west, north) of the image's upper-left corner in LV95 on the 0.5 m grid, ground pixel size, rotation.
        0.5 m: one data pixel per image pixel, 2480.5 x 1039.5 m. 0.75 m: the float sky-view factor of 3720.75 x
        1559.25 m reduced 1.5 x with Lanczos-3 (PIL, kernel widened to the reduction), then the same stretch.
        rot180: turned by 180 degrees (south up), an exact pixel permutation; the sky-view factor has no light
        direction, so the relief does not invert.
Output  <id>.png, 8-bit grey, 600 dpi, no text, no frame."""
import sys
import numpy as np
from PIL import Image
from osgeo import gdal

sys.path.insert(0, "/data/befahl/projects/AGL/basodino/hero/scripts")
from hero_look import grey                                        # noqa: E402

gdal.UseExceptions()
LO, HI = 0.49008177638053896, 0.9237946897745132                 # checks/look.json
PX = (4961, 2079)                                                # 210 x 88 mm at 600 dpi
FRAMES = {  # id: (west, north, metres per pixel, rot180)
    "a_terminus": (2678960.0, 1142120.0, 0.5, False),         # quick-look window widened to the slot, same centre
    "b_margin": (2678990.0, 1141960.0, 0.5, False),           # whole northern margin, NW corner to NE lobe
    "c_front_lakes": (2679700.0, 1141950.0, 0.5, False),      # NE ice front, proglacial lakes, east valley wall
    "d_ridge": (2678700.0, 1141100.0, 0.5, False),            # summit, ridge crest and headwall under the ice
    "e_terminus_south_up": (2678960.0, 1142120.0, 0.5, True),  # a turned 180 degrees
    "f_slabs": (2679420.0, 1142420.0, 0.5, False),            # forefield slab field and its fracture sets
    "g_glacier_0p75m": (2678360.0, 1141740.0, 0.75, False),   # whole glacier, 1.5 x ground scale
    "h_terminus_0p75m": (2678340.0, 1142380.0, 0.75, False),  # a at 1.5 x ground scale, same centre
}


def cut(src, west, north, res):
    g = src.GetGeoTransform()
    x0, y0 = (west - g[0]) / g[1], (g[3] - north) / -g[5]
    if res == g[1]:
        j, i = round(x0), round(y0)
        assert abs(j - x0) < 1e-6 and abs(i - y0) < 1e-6, "frame corner off the 0.5 m grid"
        assert 0 <= j and 0 <= i and j + PX[0] <= src.RasterXSize and i + PX[1] <= src.RasterYSize, "frame outside"
        return src.GetRasterBand(1).ReadAsArray(j, i, *PX)
    k = res / g[1]                                               # reduction factor
    w, h = PX[0] * k, PX[1] * k
    m = 8                                                        # margin for the Lanczos support (3 lobes x 1.5)
    j, i = int(np.floor(x0)) - m, int(np.floor(y0)) - m
    assert 0 <= j and 0 <= i and j + int(w) + 2 * m + 2 <= src.RasterXSize and i + int(h) + 2 * m + 2 <= src.RasterYSize
    a = src.GetRasterBand(1).ReadAsArray(j, i, int(np.ceil(w)) + 2 * m + 2, int(np.ceil(h)) + 2 * m + 2)
    box = (x0 - j, y0 - i, x0 - j + w, y0 - i + h)
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize(PX, Image.LANCZOS, box=box))


def main(svf, outdir, ids):
    src = gdal.Open(svf)
    for fid in ids or FRAMES:
        west, north, res, rot = FRAMES[fid]
        a = cut(src, west, north, res)
        if rot:
            a = a[::-1, ::-1]
        g = grey(a, LO, HI)
        Image.fromarray(g).save(f"{outdir}/{fid}.png", dpi=(600, 600), optimize=True)
        e, s = west + PX[0] * res, north - PX[1] * res
        q = np.percentile(g, [1, 50, 99])
        print(f"{fid}: E {west:.2f}-{e:.2f}, N {s:.2f}-{north:.2f}, {res} m/px{', rotated 180' if rot else ''}; "
              f"grey p1 {q[0]:.0f} median {q[1]:.0f} p99 {q[2]:.0f}, black {100 * (g == 0).mean():.1f} %, "
              f"white {100 * (g == 255).mean():.1f} %", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
