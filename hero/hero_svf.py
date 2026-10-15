"""Sky-view factor (radius 30 m, 32 directions; Zaksek et al. 2011) for the title-page art, computed with relief_v2's own
kernel (derived/relief_v2/scripts/relief_gpu.py, horizon()) on a window of the 0.5 m grid.

usage: CUDA_VISIBLE_DEVICES=0 python hero_svf.py <dem.tif> <out.tif> [spike_m]     (env agl on s02, RTX A6000)

window  WIN = E 2678100-2682700, N 1139700-1143300: every candidate frame lies inside. The DEM is read with PAD = 40 m
        (80 px) of real terrain on every side, more than the 60 steps + 1 px of a 30 m ray, so no ray of a window pixel
        reaches the edge of the array; the result is cropped back to WIN. Output float32 GeoTIFF, EPSG:2056.
spike_m single-pixel spikes: a DEM pixel more than spike_m above all 8 neighbours, or more than spike_m below all of
        them, takes the median of the 8 neighbours before the horizon search (one pass over the padded window). Such a
        pixel is narrower than any boulder the 0.5 m model resolves (the plates' boulder layer starts at 2 m2 = 8 px);
        in the sky-view factor an upward spike prints as a bright dot in a dark ring ("o"), a pit as a black dot.
        The changed pixels are listed in <out>_despike.csv (E, N, z before, z after)."""
import sys, time
import numpy as np
from osgeo import gdal, gdal_array
from scipy import ndimage as nd

sys.path.insert(0, "/data/befahl/projects/AGL/basodino/derived/relief_v2/scripts")
from relief_gpu import horizon                                   # noqa: E402  the relief_v2 kernel, unchanged

gdal.UseExceptions()
WIN = (2678100, 2682700, 1139700, 1143300)                       # west, east, south, north
PAD, RES, ND = 40, 0.5, -9999.0


def read(path, win, pad):
    src = gdal.Open(path)
    g = src.GetGeoTransform()
    x0, y0 = round((win[0] - pad - g[0]) / RES), round((g[3] - win[3] - pad) / RES)
    nx, ny = round((win[1] - win[0] + 2 * pad) / RES), round((win[3] - win[2] + 2 * pad) / RES)
    assert x0 >= 0 and y0 >= 0 and x0 + nx <= src.RasterXSize and y0 + ny <= src.RasterYSize, "window outside the DEM"
    z = src.GetRasterBand(1).ReadAsArray(x0, y0, nx, ny).astype(np.float32)
    return z, src.GetProjection()


def despike(z, t):
    """Strict single-pixel extrema beyond all 8 neighbours by more than t take the median of the 8 neighbours."""
    ring = np.ones((3, 3), bool)
    ring[1, 1] = False
    mx = nd.maximum_filter(z, footprint=ring, mode="nearest")
    mn = nd.minimum_filter(z, footprint=ring, mode="nearest")
    m = (z - mx > t) | (mn - z > t)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
    i, j = np.nonzero(m)
    nb = np.stack([z[i + di, j + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)])
    out = z.copy()
    out[i, j] = np.median(nb, axis=0)
    return out, i, j


def main(dem, out, spike=None):
    t = time.time()
    z, wkt = read(dem, WIN, PAD)
    nodata = int((z == ND).sum())
    assert nodata == 0, f"{nodata} nodata pixels in the window"
    x0, y1 = WIN[0] - PAD, WIN[3] + PAD                           # corner of the padded array
    if spike is not None:
        z2, i, j = despike(z, spike)
        up = int((z[i, j] > z2[i, j]).sum())
        np.savetxt(out[:-4] + "_despike.csv", np.column_stack([x0 + (j + 0.5) * RES, y1 - (i + 0.5) * RES, z[i, j], z2[i, j]]),
                   fmt=["%.2f", "%.2f", "%.3f", "%.3f"], delimiter=",", header="E,N,z_before,z_after", comments="")
        print(f"despike > {spike} m: {i.size} pixels in the padded window ({up} spikes, {i.size - up} pits), "
              f"largest change {np.abs(z - z2).max():.2f} m", flush=True)
        z = z2
    svf, _, kt = horizon(z, RES, 30.0, ndir=32)
    p = round(PAD / RES)
    svf = np.ascontiguousarray(svf[p:-p, p:-p])
    ds = gdal_array.OpenArray(svf)
    ds.SetGeoTransform((WIN[0], RES, 0.0, WIN[3], 0.0, -RES)); ds.SetProjection(wkt)
    ds.SetMetadata({"source": dem, "code": "hero_svf.py (horizon() of relief_v2/scripts/relief_gpu.py)",
                    "method": "Zaksek et al. (2011), 32 directions, radius 30 m, bilinear steps of 0.5 m",
                    "despike": "none" if spike is None else f"strict single-pixel extrema beyond all 8 neighbours by more "
                                                            f"than {spike} m set to the median of the 8 neighbours",
                    "edges": f"computed on {PAD} m of terrain beyond every edge, then cropped"})
    gdal.GetDriverByName("GTiff").CreateCopy(out, ds, options=["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES",
                                                                 "BIGTIFF=YES", "NUM_THREADS=8"])
    q = np.percentile(svf, [0, 1, 50, 99, 100])
    print(f"{out}: {svf.shape[1]} x {svf.shape[0]}, kernel {kt:.2f} s, wall {time.time() - t:.1f} s, "
          f"min {q[0]:.4f} p1 {q[1]:.4f} median {q[2]:.4f} p99 {q[3]:.4f} max {q[4]:.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None)
