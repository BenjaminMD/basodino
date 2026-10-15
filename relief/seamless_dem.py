"""Seamless 0.5 m DEM of the Basodino box across the Swiss-Italian border (LV95 EPSG:2056, heights LN02).
usage: python seamless_dem.py <project dir> [pad_m]          (pad_m default 200; choose the GPU with CUDA_VISIBLE_DEVICES)

inputs   raw/swissalti3d/tiles.csv, 0.5/*.tif      swissALTI3D 0.5 m tiles of the box and its ring (relief_fetch.py <dir> 200)
         raw/piemonte_dtm5/DTM5_{20,36}.tif        Regione Piemonte DTM 5 m, ICE 2009-2011 lidar, UTM 32N, Genova 1942 heights
         derived/italy/italy_datum_offset.json     Piemonte minus swissALTI3D on stable terrain, +1.913 m +- 0.15 m (stage A)
outputs  relief_v2/seamless_dem_0p5m.tif           20000 x 20000, E 2676000-2686000, N 1137000-1147000, float32, nodata -9999
         relief_v2/seamless_provenance_0p5m.tif    uint8: 1 swissALTI3D, 2 Piemonte, 3 blend zone, 4 filled void (no source)
         relief_v2/seamless_dem_0p5m_pad<pad>m.tif the same model on the box grown by pad_m on every side (relief_gpu.py input)

Piemonte to 0.5 m in one step from the native 5 m lidar grid: every LV95 pixel centre is mapped to UTM 32N (pyproj on a
50 m lattice, bilinear in between) and the interpolating cubic B-spline of the source grid (map_coordinates order 3 with
spline prefilter: passes through the 5 m posts, continuous curvature) is evaluated there; heights minus 1.913 m.
Source nodata is filled with the nearest post before the prefilter, and the footprint is eroded by two posts (10 m),
so filled values cannot reach the model.
Blend: dS, dP = distance (m) from the pixel centre to the edge of the swissALTI3D / Piemonte data (Euclidean distance
transform); wS = s(dS / L), wP = s(dP / L), s(t) = 3t^2 - 2t^3 on [0, 1], L = 25 m;
    g = wS / (wS + (1 - wS) wP),   z = zP + g (zS - zP).
g = 1 wherever swissALTI3D lies at least L inside its own edge (its heights pass bit for bit), g = 0 outside it, and g
and its gradient are continuous: no step at the data edge and no kink at the inner end of the ramp.
The model is built on the padded grid, so the ring tiles and the Piemonte data beyond the box shape the blend exactly
as inside it; the box products are crops of the padded one."""
import csv, json, os, sys, time
import numpy as np, cupy as cp
from cupyx.scipy import ndimage as cnd
from scipy import ndimage
from osgeo import gdal, gdal_array, osr
from pyproj import Transformer

gdal.UseExceptions()
E0, N0, E1, N1 = 2676000, 1137000, 2686000, 1147000
RES, ND, L = 0.5, -9999.0, 25.0
COG = ["COMPRESS=DEFLATE", "PREDICTOR=YES", "BIGTIFF=YES", "NUM_THREADS=16", "STATISTICS=YES"]
SOURCE = ("swissALTI3D 2022 0.5 m (swisstopo) where it exists; elsewhere Regione Piemonte DTM 5 m (ICE 2009-2011 lidar, "
          "CC BY 4.0) shifted to LN02 by -1.913 m, cubic B-spline to 0.5 m; feathered blend over 25 m")


def swiss(d, gt, shape):
    """swissALTI3D tiles listed in tiles.csv, mosaicked without resampling on the grid (gt, shape); nan = no tile."""
    with open(os.path.join(d, "raw/swissalti3d/tiles.csv")) as f:
        tiles = [os.path.join(d, "raw/swissalti3d/0.5", r["file"]) for r in csv.DictReader(f)]
    ny, nx = shape
    vrt = gdal.BuildVRT("/vsimem/swiss.vrt", tiles, outputBounds=(gt[0], gt[3] - ny * RES, gt[0] + nx * RES, gt[3]),
                        xRes=RES, yRes=RES, srcNodata=ND, VRTNodata=ND)
    z = vrt.ReadAsArray()
    z[z == ND] = np.nan
    return z


def piemonte(d, gt, shape, offset, rows=1040):
    """Piemonte DTM heights in LN02 on the grid (gt, shape), float64, nan outside the eroded footprint."""
    src = gdal.BuildVRT("/vsimem/piemonte.vrt", [os.path.join(d, f"raw/piemonte_dtm5/DTM5_{s}.tif") for s in (20, 36)],
                        srcNodata=-99, VRTNodata=-99)
    z, sgt = src.ReadAsArray().astype(np.float64), src.GetGeoTransform()
    valid = z != -99
    z = z[tuple(ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True))]
    coef = cp.asarray(ndimage.spline_filter(z, order=3, mode="mirror"))
    inner = cp.asarray(ndimage.binary_erosion(valid, iterations=2, border_value=0).astype(np.float32))
    ny, nx = shape
    k = 100                                                    # lattice step, pixels (50 m)
    li, lj = np.arange(ny // k + 2) * k, np.arange(nx // k + 2) * k
    ee, nn = np.meshgrid(gt[0] + (lj + 0.5) * RES, gt[3] - (li + 0.5) * RES)
    x, y = Transformer.from_crs(2056, 32632, always_xy=True).transform(ee, nn)
    lat_c = cp.asarray((x - sgt[0]) / sgt[1] - 0.5)            # source column and row of every lattice node
    lat_r = cp.asarray((sgt[3] - y) / -sgt[5] - 0.5)
    out = np.empty(shape, np.float64)
    jj = cp.arange(nx, dtype=cp.float64) / k
    for a in range(0, ny, rows):
        b = min(a + rows, ny)
        ii = cp.arange(a, b, dtype=cp.float64)[:, None] / k
        pos = cp.stack(cp.broadcast_arrays(ii, jj[None, :]))
        c = cnd.map_coordinates(lat_c, pos, order=1, mode="nearest")
        r = cnd.map_coordinates(lat_r, pos, order=1, mode="nearest")
        h = cnd.map_coordinates(coef, cp.stack([r, c]), order=3, mode="mirror", prefilter=False)
        ok = cnd.map_coordinates(inner, cp.stack([r, c]), order=0, mode="constant", cval=0.0) > 0.5
        out[a:b] = cp.where(ok, h - offset, cp.nan).get()
    return out


def smoothstep(t):
    t = cp.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def blend(zs, zp):
    """Seamless heights, provenance and the blend weight g of swissALTI3D (zs) and Piemonte (zp), both nan = no data."""
    s, p = cp.asarray(~np.isnan(zs)), cp.asarray(~np.isnan(zp))
    ws = smoothstep((cnd.distance_transform_edt(s) - 0.5) * RES / L)     # EDT: distance to the nearest pixel without data
    wp = smoothstep((cnd.distance_transform_edt(p) - 0.5) * RES / L)     # minus half a pixel: distance to the data edge
    ws[~s], wp[~p] = 0.0, 0.0
    den = ws + (1.0 - ws) * wp
    g = cp.where(den > 0, ws / cp.where(den > 0, den, 1.0), cp.nan)
    del ws, wp, den
    zs_, zp_ = cp.nan_to_num(cp.asarray(zs, cp.float64)), cp.nan_to_num(cp.asarray(zp))
    z = (g * zs_ + (1.0 - g) * zp_).astype(cp.float32)
    prov = cp.full(z.shape, 3, cp.uint8)
    prov[g == 1.0], prov[g == 0.0], prov[cp.isnan(g)] = 1, 2, 4
    z[cp.isnan(g)] = cp.nan
    return z.get(), prov.get(), g.astype(cp.float32).get()


def fill_voids(z):
    """Fill nan pixels (no source) by GDAL inverse-distance interpolation from the void edge, smoothed."""
    ds = gdal_array.OpenArray(np.where(np.isnan(z), np.float32(ND), z))
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(ND)
    gdal.FillNodata(band, None, maxSearchDist=2000, smoothingIterations=2)
    return band.ReadAsArray()


def write_cog(path, arr, gt, desc, meta, nodata=ND, ovr="AVERAGE"):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2056)
    ds = gdal_array.OpenArray(arr)
    ds.SetGeoTransform(gt)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetMetadata(meta)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.SetDescription(desc)
    gdal.GetDriverByName("COG").CreateCopy(path, ds, options=COG + [f"OVERVIEW_RESAMPLING={ovr}"])


def main(d, pad_m=200):
    t0 = time.time()
    out = os.path.join(d, "relief_v2")
    os.makedirs(out, exist_ok=True)
    p = int(round(pad_m / RES))
    gt = (E0 - pad_m, RES, 0.0, N1 + pad_m, 0.0, -RES)
    shape = (int((N1 - N0) / RES) + 2 * p, int((E1 - E0) / RES) + 2 * p)
    with open(os.path.join(d, "derived/italy/italy_datum_offset.json")) as f:
        offset = json.load(f)["offset_m"]
    print(cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(), "grid", shape, "pad", pad_m, "m",
          "offset", offset, "m", flush=True)

    t = time.time()
    zs = swiss(d, gt, shape)
    print(f"swissALTI3D mosaic: {time.time() - t:.1f} s, valid {np.mean(~np.isnan(zs)):.4f} of the padded grid", flush=True)
    t = time.time()
    zp = piemonte(d, gt, shape, offset)
    cp.cuda.Device().synchronize()
    print(f"Piemonte cubic B-spline to 0.5 m (GPU): {time.time() - t:.1f} s, valid {np.mean(~np.isnan(zp)):.4f}", flush=True)

    t = time.time()
    z, prov, g = blend(zs, zp)
    cp.get_default_memory_pool().free_all_blocks()
    print(f"blend (GPU distance transforms): {time.time() - t:.1f} s", flush=True)
    both = (prov == 3)
    dz = (zs - zp)[both]
    print(f"blend zone: {both.sum() * RES * RES / 1e6:.4f} km2 (padded grid); swissALTI3D minus Piemonte there: "
          f"median {np.median(dz):.3f} m, NMAD {1.4826 * np.median(np.abs(dz - np.median(dz))):.3f} m, "
          f"p1 {np.percentile(dz, 1):.2f} p99 {np.percentile(dz, 99):.2f} min {dz.min():.2f} max {dz.max():.2f} m", flush=True)
    voids = np.isnan(z)
    if voids.any():
        rr, cc = np.nonzero(voids)
        print(f"voids with no source: {voids.sum()} px, E {gt[0] + cc.min() * RES:.1f}-{gt[0] + (cc.max() + 1) * RES:.1f}, "
              f"N {gt[3] - (rr.max() + 1) * RES:.1f}-{gt[3] - rr.min() * RES:.1f}; filled by inverse distance", flush=True)
        z = fill_voids(z)
    else:
        print("voids with no source: none", flush=True)
    s1 = prov == 1
    assert np.array_equal(z[s1], zs[s1]), "swissALTI3D pixels changed"
    del zs, zp

    box = (slice(p, p + shape[0] - 2 * p), slice(p, p + shape[1] - 2 * p))
    gt_box = (float(E0), RES, 0.0, float(N1), 0.0, -RES)
    meta = {"source": SOURCE, "code": "seamless_dem.py", "vertical": "LN02 (m a.s.l.)",
            "provenance": "seamless_provenance_0p5m.tif: 1 swissALTI3D, 2 Piemonte, 3 blend zone, 4 filled void"}
    zb, pb = z[box], prov[box]
    assert zb.shape == (20000, 20000) and not np.isnan(zb).any()
    names = {1: "swissALTI3D", 2: "Piemonte", 3: "blend zone", 4: "filled void"}
    print("box provenance: " + ", ".join(f"{names[k]} {np.mean(pb == k) * 100:.3f} % ({np.sum(pb == k) * RES * RES / 1e6:.4f} km2)"
                                          for k in names), flush=True)
    print(f"box heights: min {zb.min():.2f} m, max {zb.max():.2f} m, no nodata", flush=True)
    t = time.time()
    write_cog(os.path.join(out, "seamless_dem_0p5m.tif"), zb, gt_box, "elevation (m, LN02)", meta)
    write_cog(os.path.join(out, "seamless_provenance_0p5m.tif"), pb, gt_box,
              "1 swissALTI3D, 2 Piemonte, 3 blend zone, 4 filled void", meta, nodata=0, ovr="NEAREST")
    write_cog(os.path.join(out, f"seamless_dem_0p5m_pad{pad_m}m.tif"), z, gt, "elevation (m, LN02)",
              {**meta, "pad": f"{pad_m} m on every side of E {E0}-{E1}, N {N0}-{N1}"})
    print(f"written in {time.time() - t:.1f} s; total {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200)
