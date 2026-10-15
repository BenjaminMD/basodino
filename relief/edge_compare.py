"""Stage A relief (rays stop at the box edge, 3 x 3 windows run out of data) against relief_v2 (padded seamless model)
where both see the same swissALTI3D heights: 300 m strips along the east and north edges of the box (without the
200 m next to the corners, where the other edge interferes), and an interior window.
usage: python edge_compare.py <project dir>
For every product and band of distance from the edge: mean and maximum |relief_v2 - stage A| and the stage A nodata count.
Interior window E 2681000-2685000, N 1138000-1146000 (> 1 km from every edge, > 3 km from the data border): must be
identical, because relief_v2 changes nothing there."""
import os, sys
import numpy as np
from osgeo import gdal

gdal.UseExceptions()
PRODUCTS = ["svf_r150m_0p5m", "openness_r150m_0p5m", "svf_r30m_0p5m", "openness_r30m_0p5m", "hillshade_315_45_0p5m",
            "hillshade_multidir_0p5m", "slope_0p5m"]
BANDS = [(0, 0.5), (0.5, 10), (10, 50), (50, 150), (150, 300)]      # metres from the edge
RES, N = 0.5, 20000


def read(path, x, y, nx, ny):
    ds = gdal.Open(path)
    a = ds.GetRasterBand(1).ReadAsArray(x, y, nx, ny).astype(np.float64)
    bad = a == ds.GetRasterBand(1).GetNoDataValue()
    a[bad] = np.nan
    return a, bad


def main(d):
    for p in PRODUCTS:
        a_path, b_path = os.path.join(d, "derived/relief", p + ".tif"), os.path.join(d, "relief_v2", p + ".tif")
        k = 600                                                     # 300 m strip
        for edge, win in (("east", (N - k, 400, k, N - 800)), ("north", (400, 0, N - 800, k))):   # 200 m off the corners
            a, bad = read(a_path, *win)
            b, _ = read(b_path, *win)
            dist = ((k - 1 - np.arange(k)) + 0.5) * RES if edge == "east" else (np.arange(k) + 0.5) * RES
            dist = dist[None, :] if edge == "east" else dist[:, None]
            dist = np.broadcast_to(dist, a.shape)
            parts = []
            for lo, hi in BANDS:
                m = (dist >= lo) & (dist < hi)
                dd = np.abs(b[m] - a[m])
                parts.append(f"{lo:g}-{hi:g} m: mean {np.nanmean(dd):.4f} max {np.nanmax(dd):.4f} "
                             f"(stage A nodata {int(bad[m].sum())})" if np.isfinite(dd).any() else
                             f"{lo:g}-{hi:g} m: stage A nodata {int(bad[m].sum())}")
            print(f"{p:26s} {edge:5s} edge | " + " | ".join(parts), flush=True)
        a, _ = read(a_path, 10000, 2000, 8000, 16000)
        b, _ = read(b_path, 10000, 2000, 8000, 16000)
        print(f"{p:26s} interior window identical: {np.array_equal(a, b, equal_nan=True)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
