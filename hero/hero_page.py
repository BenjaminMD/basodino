"""Full-page title art: the sky-view factor cut to the whole A4 page (210 x 297 mm at 600 dpi, 4961 x 7016 px), same
tone as hero_render.py. Frames (west, north, metres per pixel) as there.

usage: python hero_page.py <svf.tif> <outdir> [id ...]          (env agl on s02)"""
import sys
import numpy as np
from PIL import Image
from osgeo import gdal

sys.path.insert(0, "/data/befahl/projects/AGL/basodino/hero/scripts")
import hero_render as hr                                          # noqa: E402
from hero_look import grey                                        # noqa: E402

hr.PX = (4961, 7016)
FRAMES = {  # id: (west, north, metres per pixel, rot180)
    "p_full_0p5": (2678960.0, 1143300.0, 0.5, False),         # banner columns, the whole window height
    "p_south_0p5": (2678960.0, 1143000.0, 0.5, False),        # same, 300 m further south
}


def main(svf, outdir, ids):
    src = gdal.Open(svf)
    for fid in ids or FRAMES:
        west, north, res, rot = FRAMES[fid]
        a = hr.cut(src, west, north, res)
        g = grey(a, hr.LO, hr.HI)
        Image.fromarray(g).save(f"{outdir}/{fid}.png", dpi=(600, 600), optimize=True)
        Image.fromarray(g).resize((hr.PX[0] // 6, hr.PX[1] // 6), Image.LANCZOS).save(f"{outdir}/{fid}_small.png")
        e, s = west + hr.PX[0] * res, north - hr.PX[1] * res
        q = np.percentile(g, [1, 50, 99])
        print(f"{fid}: E {west:.2f}-{e:.2f}, N {s:.2f}-{north:.2f}, {res} m/px; grey p1 {q[0]:.0f} median {q[1]:.0f} "
              f"p99 {q[2]:.0f}, white {100 * (g == 255).mean():.1f} %", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
