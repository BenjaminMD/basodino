"""Cut the record window from every image of tex/build/record_files.csv (one end-of-summer image per year, selected
by code/satellite_record_select.py from the archive catalogue) -> products/record/<year>_<date>_<platform>.tif, three
bands red, green, blue on the window E 2678800-2681800, N 1140200-1142200 (LV95), float32 reflectance for the
satellites (Landsat MSS: false colour near infrared, red, green, top-of-atmosphere), uint8 display values at 2 m for
the aerial orthophotos (grey repeated in three bands for black-and-white flights). Runs on s02 in the archive
directory (env agl): python satellite_record_cut.py record_files.csv
"""
import csv
import os
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from rasterio.windows import from_bounds as win

E0, N0, E1, N1 = 2678800, 1140200, 2681800, 1142200
OUT = "products/record"
os.makedirs(OUT, exist_ok=True)
RGB = {"landsat_sr": ("red", "green", "blue"), "landsat_toa_mss": ("nir08", "red", "green")}


def bands(src, names):
    """Band indices by the colour word of the band descriptions ('SR_B3 red ...', 'B6 nir08 ...')."""
    d = [((x or "").split(" ") + ["", ""])[1].upper() for x in src.descriptions]
    idx = []
    for n in names:
        if n.upper() not in d:
            raise KeyError(f"{src.name}: band {n} not in {src.descriptions}")
        idx.append(d.index(n.upper()) + 1)
    return idx


for r in csv.DictReader(open(sys.argv[1])):
    path, enc = r["path"], r["encoding"]
    out = f"{OUT}/{r['year']}_{r['date']}_{r['platform'].replace(' ', '')}.tif"
    with rasterio.open(path) as src:
        if enc == "display":
            res = 2.0
            w, h = int((E1 - E0) / res), int((N1 - N0) / res)
            t = from_bounds(E0, N0, E1, N1, w, h)
            n = min(src.count, 3)
            arr = np.zeros((3, h, w), np.float32)
            for b in range(3):
                reproject(rasterio.band(src, min(b + 1, n)), arr[b], dst_transform=t, dst_crs="EPSG:2056",
                          resampling=Resampling.average)
            data, dtype, tags = np.clip(arr, 0, 255).astype(np.uint8), "uint8", {"encoding": "8-bit display values"}
        else:
            if enc == "s2":
                idx, scale, offset = [3, 2, 1], 1e-4, float(r.get("offset") or 0)
                tags = {"encoding": "reflectance; 0 = no data"}
            elif "_toa" in path:
                idx, scale, offset = bands(src, RGB["landsat_toa_mss"]), 0.0000275, -0.2
                tags = {"encoding": "top-of-atmosphere reflectance, false colour near infrared, red, green"}
            else:
                idx, scale, offset = bands(src, RGB["landsat_sr"]), 0.0000275, -0.2
                tags = {"encoding": "surface reflectance"}
            wd = win(E0, N0, E1, N1, src.transform).round_offsets().round_lengths()
            dn = src.read(idx, window=wd).astype(np.float32)
            data = np.where((dn == 0) | (dn == 65535), np.nan, dn * scale + offset).astype(np.float32)
            t, dtype = src.window_transform(wd), "float32"
            w, h = data.shape[2], data.shape[1]
        with rasterio.open(out, "w", driver="GTiff", width=w, height=h, count=3, dtype=dtype, crs="EPSG:2056",
                           transform=t, compress="deflate", predictor=2 if dtype == "uint8" else 3) as dst:
            dst.write(data)
            dst.update_tags(source=path, date=r["date"], platform=r["platform"], **tags)
    print(out, data.shape, flush=True)
