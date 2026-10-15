"""Planning overview of the hero window: sky-view factor block-averaged 4 x (2 m pixels) with the quick-look stretch,
the 2024 glacier outline (blue), standing water (cyan), a 250 m grid with LV95 labels every 500 m and optional
candidate frames (red, with ids).

usage: python hero_overview.py <svf.tif> <out.png> [id:W,N,S,E_m_per_px ...]
       frame spec id:west,north,px_per_m... see FRAMES in hero_render.py; here: id:west,north,width_m,height_m"""
import sys
import numpy as np
import geopandas as gpd
from PIL import Image, ImageDraw, ImageFont
from osgeo import gdal

sys.path.insert(0, "/data/befahl/projects/AGL/basodino/hero/scripts")
from hero_look import grey                                        # noqa: E402
from hero_svf import WIN                                          # noqa: E402

LO, HI = 0.49008177638053896, 0.9237946897745132
GPKG = "/data/befahl/projects/AGL/basodino/derived/vectors.gpkg"
K = 4                                                            # block size, px
FONT = "/data/befahl/projects/AGL/basodino/hero/fonts/SourceSans3-Semibold.ttf"


def main(path, out, frames):
    a = gdal.Open(path).ReadAsArray()
    ny, nx = a.shape[0] // K, a.shape[1] // K
    a = a[:ny * K, :nx * K].reshape(ny, K, nx, K).mean(axis=(1, 3))
    im = Image.fromarray(grey(a, LO, HI)).convert("RGB")
    d = ImageDraw.Draw(im)
    m = 0.5 * K                                                   # metres per overview pixel
    xy = lambda E, N: ((E - WIN[0]) / m, (WIN[3] - N) / m)
    try:
        font = ImageFont.truetype(FONT, 22)
    except OSError:
        font = ImageFont.load_default()
    for E in range(WIN[0] - WIN[0] % 250 + 250, WIN[1], 250):
        x = xy(E, 0)[0]
        d.line([(x, 0), (x, im.size[1])], fill=(255, 170, 0) if E % 500 == 0 else (150, 150, 150), width=1)
        if E % 500 == 0:
            d.text((x + 3, 3), f"{E}", fill=(255, 120, 0), font=font)
    for N in range(WIN[2] - WIN[2] % 250 + 250, WIN[3], 250):
        y = xy(0, N)[1]
        d.line([(0, y), (im.size[0], y)], fill=(255, 170, 0) if N % 500 == 0 else (150, 150, 150), width=1)
        if N % 500 == 0:
            d.text((3, y + 3), f"{N}", fill=(255, 120, 0), font=font)
    box = (WIN[0], WIN[2], WIN[1], WIN[3])
    g = gpd.read_file(GPKG, layer="glacier_2024", bbox=box)
    w = gpd.read_file(GPKG, layer="tlm_bb_bodenbedeckung", bbox=box)
    w = w[w.objektart == "Stehende Gewaesser"]
    for geoms, col in ((g.geometry, (40, 110, 255)), (w.geometry, (0, 200, 220))):
        for geom in geoms:
            for poly in getattr(geom, "geoms", [geom]):
                for ring in [poly.exterior, *poly.interiors]:
                    d.line([xy(x, y) for x, y, *_ in ring.coords], fill=col, width=2)
    for f in frames:
        fid, v = f.split(":")
        W, N, wm, hm = (float(t) for t in v.split(","))
        d.rectangle([xy(W, N), xy(W + wm, N - hm)], outline=(230, 0, 0), width=3)
        d.text((xy(W, N)[0] + 6, xy(W, N)[1] + 4), fid, fill=(230, 0, 0), font=font)
    im.save(out)
    print(out, im.size)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
