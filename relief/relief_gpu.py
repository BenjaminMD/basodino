"""Relief products of a 0.5 m DEM on one GPU, written as COGs.
usage: python relief_gpu.py <dem.tif> <outdir> [pad_px]        (choose the GPU with CUDA_VISIBLE_DEVICES)

pad_px: the DEM carries pad_px pixels of real terrain beyond the product box on every side (seamless_dem.py writes such a
DEM). Every product is computed on the padded DEM and cropped to the box, so 3 x 3 windows, aggregation blocks and horizon
rays at the box edge see terrain instead of an edge, and the products are valid up to the last row and column.
pad_px must be a multiple of 20 (10 m blocks) and larger than 150 m / res + 1 (the 150 m horizon search). pad_px 0 = stage A.

hillshade_315_45, hillshade_multidir: GDAL conventions, Horn (1981) gradient, byte 1 + 254 cos(incidence), 0 = nodata;
    multidirectional = Mark (1992) oblique weighting of azimuths 225/270/315/360, weight cos^2(aspect - azimuth), 45 deg.
svf_r30m, svf_r150m: sky-view factor (Zaksek et al. 2011), 32 directions, bilinear samples every 0.5 m along each ray.
openness_r30m, openness_r150m: positive openness (Yokoyama et al. 2002), degrees, from the same horizon search.
slope_0p5m, slope_2m, slope_10m: Horn slope in degrees; 2 m and 10 m from block means of the 0.5 m DEM.
relief_composite: Swiss-style composite shading, byte 1 + 254 C, 0 = nodata, with
    C = (0.6 M + 0.4 V) cos(S)^0.5
    M  multidirectional shading, reflectance 0..1 (the float behind hillshade_multidir): oblique light, models form
    V  sky-view factor, mean of the 30 m and 150 m radii: diffuse sky light, darkens gullies, couloirs and valley floors
    S  slope at 0.5 m; cos(S)^0.5 is the slope darkening of Swiss manual relief shading, steep rock darker
       (factor 0.93 at 30 deg, 0.84 at 45 deg, 0.71 at 60 deg, 0.42 at 80 deg)
    0.6 M + 0.4 V reads as direct plus diffuse light; flat open ground gets 0.6 sin 45 deg + 0.4 = 0.82 (byte 210).
Nodata: rays stop at nodata and at the DEM edge; any nodata in a 3 x 3 window or aggregation block gives nodata.
The DEM goes through the GPU in row strips with halos wide enough that strip borders cannot show."""
import os, sys, time
import numpy as np, cupy as cp
from osgeo import gdal, gdal_array

gdal.UseExceptions()
ND = -9999.0
ROWS = 1000
COG = ["COMPRESS=DEFLATE", "PREDICTOR=YES", "BIGTIFF=YES", "NUM_THREADS=16", "OVERVIEW_RESAMPLING=AVERAGE", "STATISTICS=YES"]
W_DIRECT, W_SKY, SLOPE_EXP = 0.6, 0.4, 0.5

HORIZON = cp.RawKernel(r'''
extern "C" __global__ void horizon(const float* z, int ny, int nx, int row0, int nrows,
                                   float res, int steps, int ndir, float* svf, float* opn)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= nx || i >= nrows) return;
    int r = row0 + i;                                   // row of this output pixel inside the strip z
    size_t o = (size_t)i * nx + j;
    float z0 = z[(size_t)r * nx + j];
    if (isnan(z0)) { svf[o] = nanf(""); opn[o] = nanf(""); return; }
    float ssin = 0.f, sang = 0.f;
    for (int d = 0; d < ndir; d++) {
        float dx, dy;
        sincospif(2.f * d / ndir, &dx, &dy);            // azimuth clockwise from north: east = sin, north = cos
        dy = -dy;                                       // rows run southwards
        float t = -3e38f;                               // max tan(elevation angle) along the ray
        for (int s = 1; s <= steps; s++) {
            float ox = s * dx, oy = s * dy, fx0 = floorf(ox), fy0 = floorf(oy);
            int x0 = j + (int)fx0, y0 = r + (int)fy0;
            if (x0 < 0 || y0 < 0 || x0 + 1 >= nx || y0 + 1 >= ny) break;
            float fx = ox - fx0, fy = oy - fy0;
            const float* p = z + (size_t)y0 * nx + x0;
            float h = ((p[0] - z0) * (1.f - fx) + (p[1] - z0) * fx) * (1.f - fy)
                    + ((p[nx] - z0) * (1.f - fx) + (p[nx + 1] - z0) * fx) * fy;
            if (isnan(h)) break;
            t = fmaxf(t, h / (s * res));
        }
        float g = t == -3e38f ? 0.f : atanf(t);         // no terrain sampled: open horizon
        ssin += sinf(fmaxf(g, 0.f));
        sang += g;
    }
    svf[o] = 1.f - ssin / ndir;
    opn[o] = 90.f - 57.29577951308232f * sang / ndir;
}
''', 'horizon')


def horizon(z, res, radius, ndir=32, rows=ROWS):
    """Sky-view factor and positive openness (degrees) of the float32 DEM z (nan = nodata). Returns svf, opn, kernel s."""
    steps, (ny, nx) = int(round(radius / res)), z.shape
    halo = steps + 2
    svf, opn, kt = np.empty_like(z), np.empty_like(z), 0.0
    for a in range(0, ny, rows):
        b = min(a + rows, ny)
        s0, s1 = max(a - halo, 0), min(b + halo, ny)
        zs = cp.asarray(z[s0:s1])
        sv, op = cp.empty((b - a, nx), cp.float32), cp.empty((b - a, nx), cp.float32)
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        HORIZON(((nx + 31) // 32, (b - a + 7) // 8), (32, 8),
                (zs, np.int32(s1 - s0), np.int32(nx), np.int32(a - s0), np.int32(b - a),
                 np.float32(res), np.int32(steps), np.int32(ndir), sv, op))
        e1.record(); e1.synchronize()
        kt += cp.cuda.get_elapsed_time(e0, e1) / 1000
        svf[a:b], opn[a:b] = sv.get(), op.get()
    return svf, opn, kt


def gradient(zp, res):
    """Horn (1981) east and north gradients of a nan-padded strip (one extra row and column on every side)."""
    z = cp.asarray(zp, dtype=cp.float64)
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, e, f = z[1:-1, :-2], z[1:-1, 1:-1], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    ge = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * res) + 0 * e   # 0 * e: nodata centre -> nan
    gn = ((a + 2 * b + c) - (g + 2 * h + i)) / (8 * res) + 0 * e   # top row is north
    return ge, gn


def shade(ge, gn, az, alt=45.0):
    az, alt = np.radians(az), np.radians(alt)
    return cp.maximum(0.0, (np.sin(alt) - np.cos(alt) * (np.sin(az) * ge + np.cos(az) * gn)) / cp.sqrt(1 + ge * ge + gn * gn))


def multidirectional(ge, gn, alt=45.0):
    num, w = 0.0, 0.0
    for az in (225, 270, 315, 360):
        wi = (np.sin(np.radians(az)) * ge + np.cos(np.radians(az)) * gn) ** 2   # |grad|^2 cos^2(aspect - az)
        num, w = num + wi * shade(ge, gn, az, alt), w + wi
    return cp.where(w == 0, np.sin(np.radians(alt)), num / w)                   # flat: every azimuth gives sin(alt)


def to_byte(v):
    return cp.where(cp.isnan(v), 0, cp.floor(1 + 254 * v + 0.5)).astype(cp.uint8)


def horn_products(z, res, rows=ROWS):
    """Hillshade 315/45 and multidirectional (uint8, 0 = nodata), slope in degrees and the multidirectional
    reflectance 0..1 (float32, nan = nodata)."""
    zp = np.pad(z, 1, constant_values=np.nan)
    hs, md = np.empty(z.shape, np.uint8), np.empty(z.shape, np.uint8)
    sl, mr = np.empty(z.shape, np.float32), np.empty(z.shape, np.float32)
    for a in range(0, z.shape[0], rows):
        b = min(a + rows, z.shape[0])
        ge, gn = gradient(zp[a:b + 2], res)
        m = multidirectional(ge, gn)
        hs[a:b] = to_byte(shade(ge, gn, 315)).get()
        md[a:b], mr[a:b] = to_byte(m).get(), m.astype(cp.float32).get()
        sl[a:b] = cp.degrees(cp.arctan(cp.hypot(ge, gn))).astype(cp.float32).get()
    return hs, md, sl, mr


def composite(mr, svf30, svf150, slope, rows=ROWS):
    """Swiss-style composite (uint8, 0 = nodata): (W_DIRECT M + W_SKY mean SVF) cos(slope)^SLOPE_EXP."""
    out = np.empty(mr.shape, np.uint8)
    for a in range(0, mr.shape[0], rows):
        b = min(a + rows, mr.shape[0])
        m, v30, v150, s = (cp.asarray(x[a:b], cp.float64) for x in (mr, svf30, svf150, slope))
        out[a:b] = to_byte((W_DIRECT * m + W_SKY * 0.5 * (v30 + v150)) * cp.cos(cp.radians(s)) ** SLOPE_EXP).get()
    return out


def block_mean(z, k):
    ny, nx = z.shape
    return z.reshape(ny // k, k, nx // k, k).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)  # nan if any nodata


def write_cog(path, arr, gt, wkt, desc, meta):
    nodata = 0 if arr.dtype == np.uint8 else ND
    if arr.dtype != np.uint8:
        arr = np.where(np.isnan(arr), np.float32(ND), arr)
    ds = gdal_array.OpenArray(arr)
    ds.SetGeoTransform(gt); ds.SetProjection(wkt); ds.SetMetadata(meta)
    band = ds.GetRasterBand(1); band.SetNoDataValue(nodata); band.SetDescription(desc)
    gdal.GetDriverByName("COG").CreateCopy(path, ds, options=COG)


def main(dem, out, pad=0):
    os.makedirs(out, exist_ok=True)
    src = gdal.Open(dem)
    g0, wkt, res = src.GetGeoTransform(), src.GetProjection(), src.GetGeoTransform()[1]
    assert pad % 20 == 0 and (pad == 0 or pad > round(150 / res) + 1), "pad_px: multiple of 20, longer than the 150 m rays"
    gt = (g0[0] + pad * res, res, 0.0, g0[3] - pad * res, 0.0, -res)
    z = src.GetRasterBand(1).ReadAsArray()
    z[z == ND] = np.nan
    box = lambda a, k=1: np.ascontiguousarray(a[pad // k:a.shape[0] - pad // k, pad // k:a.shape[1] - pad // k])
    meta = {"source": f"{os.path.basename(dem)}: {src.GetMetadataItem('source') or 'swissALTI3D 2022, 0.5 m, LN02'}",
            "code": "relief_gpu.py",
            "edges": f"computed on {pad} px ({pad * res:g} m) of terrain beyond every edge, then cropped" if pad else
                     "computed on the DEM as it is (rays stop at the edge)"}
    print(cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(), "DEM", z.shape, "pad", pad, "px,",
          "products", box(z).shape, flush=True)
    times = {}

    def done(name, arr, desc, gpu_s, g=gt, **m):
        t = time.time()
        write_cog(os.path.join(out, name + ".tif"), arr, g, wkt, desc, {**meta, **m})
        bad = int(np.sum(arr == 0)) if arr.dtype == np.uint8 else int(np.isnan(arr).sum())
        v = arr[arr != 0] if arr.dtype == np.uint8 else arr[~np.isnan(arr)]
        p = np.percentile(v, [1, 50, 99])
        print(f"{name}: gpu {gpu_s:.2f} s, write {time.time() - t:.1f} s, {arr.shape[1]} x {arr.shape[0]}, nodata {bad}, "
              f"min {v.min():.4f} p1 {p[0]:.4f} median {p[1]:.4f} p99 {p[2]:.4f} max {v.max():.4f}", flush=True)

    t0 = time.time()
    t = time.time()
    hs, md, sl, mr = (box(a) for a in horn_products(z, res))
    times["hillshades and slope 0.5 m (Horn, with transfers)"] = tg = time.time() - t
    done("hillshade_315_45_0p5m", hs, "hillshade az 315 alt 45 (1..255, 0 nodata)", tg, method="GDAL hillshade, Horn gradient")
    done("hillshade_multidir_0p5m", md, "multidirectional hillshade alt 45 (1..255, 0 nodata)", tg,
         method="Mark (1992) oblique weighting, azimuths 225 270 315 360, as gdaldem -multidirectional")
    done("slope_0p5m", sl, "slope (degrees)", tg, method="Horn (1981)")
    del hs, md
    for k, name in ((4, "2m"), (20, "10m")):
        t = time.time()
        zk = block_mean(z, k)
        ge, gn = gradient(np.pad(zk, 1, constant_values=np.nan), res * k)
        slk = box(cp.degrees(cp.arctan(cp.hypot(ge, gn))).astype(cp.float32).get(), k)
        times[f"slope {name}"] = time.time() - t
        done(f"slope_{name}", slk, "slope (degrees)", times[f"slope {name}"],
             g=(gt[0], res * k, 0.0, gt[3], 0.0, -res * k), method=f"Horn (1981) on {k} x {k} block means of the 0.5 m DEM")
    svf = {}
    for radius in (30, 150):
        t = time.time()
        s, o, kt = horizon(z, res, radius)
        s, o = box(s), box(o)
        times[f"horizon r {radius} m (kernel)"] = kt
        times[f"horizon r {radius} m (with transfers)"] = time.time() - t
        print(f"horizon r = {radius} m: kernel {kt:.2f} s, with transfers {time.time() - t:.2f} s", flush=True)
        done(f"svf_r{radius}m_0p5m", s, f"sky-view factor, radius {radius} m, 32 directions", kt,
             method=f"Zaksek et al. (2011), 32 directions, radius {radius} m, bilinear steps of 0.5 m")
        done(f"openness_r{radius}m_0p5m", o, f"positive openness (degrees), radius {radius} m", kt,
             method=f"Yokoyama et al. (2002) positive openness, 32 directions, radius {radius} m")
        svf[radius] = s
        del o
        cp.get_default_memory_pool().free_all_blocks()
    t = time.time()
    c = composite(mr, svf[30], svf[150], sl)
    times["composite"] = time.time() - t
    done("relief_composite_0p5m", c, "Swiss-style composite relief (1..255, 0 nodata)", times["composite"],
         method=f"byte 1 + 254 C, C = ({W_DIRECT} M + {W_SKY} V) cos(S)^{SLOPE_EXP}; M multidirectional reflectance "
                f"(hillshade_multidir), V mean sky-view factor of radius 30 m and 150 m, S slope 0.5 m")
    print("GPU times (s): " + ", ".join(f"{k} {v:.2f}" for k, v in times.items()), flush=True)
    print(f"total wall time {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
