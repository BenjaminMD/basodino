"""Grey crops of sky-view factor rasters at given LV95 centres, with the quick-look stretch, zoomed by nearest neighbour.
One row per centre, one column per raster (several rasters: paths joined by ':').

usage: python hero_crops.py <svf.tif[:svf2.tif...]> <out.png> <size_px> <zoom> E,N [E,N ...]"""
import sys
import numpy as np
from PIL import Image
from osgeo import gdal

sys.path.insert(0, "/data/befahl/projects/AGL/basodino/hero/scripts")
from hero_look import grey                                        # noqa: E402

LO, HI = 0.49008177638053896, 0.9237946897745132


def main(paths, out, size, zoom, pts):
    srcs = [gdal.Open(p) for p in paths.split(":")]
    s = size * zoom
    W = Image.new("L", (len(srcs) * (s + 8) - 8, len(pts) * (s + 8) - 8), 90)
    for r, p in enumerate(pts):
        E, N = (float(v) for v in p.split(","))
        for c, src in enumerate(srcs):
            g = src.GetGeoTransform()
            j, i = int((E - g[0]) / g[1]) - size // 2, int((g[3] - N) / -g[5]) - size // 2
            a = src.GetRasterBand(1).ReadAsArray(j, i, size, size)
            W.paste(Image.fromarray(grey(a, LO, HI)).resize((s, s), Image.NEAREST), (c * (s + 8), r * (s + 8)))
    W.save(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:])
