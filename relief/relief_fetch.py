"""Fetch the latest swissALTI3D 0.5 m tile per 1 km cell of the Basodino box and mosaic them.
usage: python relief_fetch.py <project dir> [pad_m]
writes <dir>/raw/swissalti3d/0.5/*.tif, <dir>/raw/swissalti3d/tiles.csv, <dir>/derived/dem_0p5m.tif
Grid: LV95 (EPSG:2056), E 2676000-2686000, N 1137000-1147000, 0.5 m, 20000 x 20000 float32, nodata -9999.
pad_m > 0 also fetches every cell within pad_m of the box (the ring that pads the relief computation, see
seamless_dem.py); the mosaic stays the box. Cells without a swisstopo tile (Italy) stay nodata.
Tiles are checked against the STAC sha2-256 multihash."""
import csv, hashlib, json, os, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from osgeo import gdal
from pyproj import Transformer

gdal.UseExceptions()

STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"
E0, N0, E1, N1 = 2676000, 1137000, 2686000, 1147000
TE = [str(v) for v in (E0, N0, E1, N1)]


def cells(pad):
    """Lower-left km of every 1 km tile that intersects the box grown by pad metres."""
    km = lambda a, b: range((a - pad) // 1000, -(-(b + pad) // 1000))
    return {f"{e}-{n}" for e in km(E0, E1) for n in km(N0, N1)}


def bbox_wgs84(pad, margin=0.002):
    lon, lat = Transformer.from_crs(2056, 4326, always_xy=True).transform(
        [E0 - pad, E0 - pad, E1 + pad, E1 + pad], [N0 - pad, N1 + pad, N0 - pad, N1 + pad])
    return f"{min(lon) - margin:.5f},{min(lat) - margin:.5f},{max(lon) + margin:.5f},{max(lat) + margin:.5f}"


def latest_tiles(want, bbox):
    url, feats = f"{STAC}?limit=100&bbox={bbox}", []
    while url:
        with urllib.request.urlopen(url, timeout=120) as r:
            page = json.load(r)
        feats += page["features"]
        url = next((l["href"] for l in page["links"] if l["rel"] == "next"), None)
    best = {}
    for f in feats:
        _, year, cell = f["id"].split("_")
        if cell in want and year > best.get(cell, ("0",))[0]:
            a = next(a for k, a in f["assets"].items() if k.endswith("_0.5_2056_5728.tif"))
            best[cell] = (year, a["href"], a["checksum:multihash"][4:].lower())  # strip 1220 (sha2-256, 32 bytes)
    return best


def fetch(href, sha, path):
    for attempt in range(5):
        try:
            if not os.path.exists(path):
                urllib.request.urlretrieve(href, path + ".part")
                os.replace(path + ".part", path)
            with open(path, "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() == sha:
                    return
            os.remove(path)
            print("checksum mismatch, retry", href, flush=True)
        except OSError as e:
            print("retry", attempt, href, e, flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("failed " + href)


def main(d, pad=0):
    tdir = os.path.join(d, "raw/swissalti3d/0.5")
    os.makedirs(tdir, exist_ok=True)
    want = cells(pad)
    best = latest_tiles(want, bbox_wgs84(pad))
    jobs = [(href, sha, os.path.join(tdir, os.path.basename(href))) for _, href, sha in best.values()]
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: fetch(*j), jobs))
    print(f"{len(jobs)} of {len(want)} cells have a tile, years {sorted({v[0] for v in best.values()})}, "
          f"downloaded and verified in {time.time() - t0:.0f} s", flush=True)
    with open(os.path.join(d, "raw/swissalti3d/tiles.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "year", "file", "sha256", "url"])
        for cell in sorted(best):
            year, href, sha = best[cell]
            w.writerow([cell, year, os.path.basename(href), sha, href])
    for _, _, p in jobs:  # every tile must sit on the 0.5 m grid of its 1 km cell
        g = gdal.Open(p)
        e, n = (int(v) * 1000 for v in os.path.basename(p).split("_")[2].split("-"))
        assert g.GetGeoTransform() == (e, 0.5, 0.0, n + 1000, 0.0, -0.5) and g.RasterXSize == g.RasterYSize == 2000, p
    vrt, out = os.path.join(d, "raw/swissalti3d/mosaic_0p5m.vrt"), os.path.join(d, "derived/dem_0p5m.tif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run(["gdalbuildvrt", "-q", "-te", *TE, "-tr", "0.5", "0.5", "-srcnodata", "-9999", "-vrtnodata", "-9999",
                    vrt, *sorted(p for _, _, p in jobs)], check=True)
    subprocess.run(["gdal_translate", "-q", "-of", "COG", "-ot", "Float32", "-a_nodata", "-9999",
                    "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=YES", "-co", "BIGTIFF=YES", "-co", "NUM_THREADS=16",
                    "-co", "OVERVIEWS=IGNORE_EXISTING", "-co", "OVERVIEW_RESAMPLING=AVERAGE", vrt, out], check=True)
    g = gdal.Open(out)
    assert g.GetGeoTransform() == (2676000.0, 0.5, 0.0, 1147000.0, 0.0, -0.5) and g.RasterXSize == g.RasterYSize == 20000
    z = g.GetRasterBand(1).ReadAsArray()
    v = z[z != -9999]
    print(f"{out}: {g.RasterXSize} x {g.RasterYSize}, valid {v.size / z.size:.4f}, "
          f"min {v.min():.2f} max {v.max():.2f} m, total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
