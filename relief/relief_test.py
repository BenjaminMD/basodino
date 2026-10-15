"""Checks relief_gpu.py before it runs on the real DEM.   usage: python relief_test.py [dem.tif]
1. Sky-view factor and positive openness on synthetic 0.5 m surfaces, 32 directions, radius 30 m and 150 m, against
   the analytic values: flat plane (SVF 1, openness 90), floor of a V-shaped valley with side slope a
   (SVF 1 - 2a/pi, openness 90 - mean angle), inclined plane with slope b (SVF 1 - b/pi, openness 90) and a cone summit
   (SVF 1, openness 90 + a). 'exact' is the 32-direction sum the kernel must reproduce, 'continuous' the integral.
2. Strip borders: 97-row strips must reproduce a single strip bit for bit (rough surface with a nodata hole).
3. Edges: on a 45 deg inclined plane padded by 320 px (160 m) and cropped, as relief_gpu.py does with pad_px, every box
   pixel including the outermost rows and columns must match the analytic sky-view factor, openness and slope; the same
   box computed without the pad shows the edge error the pad removes (printed, not a check).
4. Composite: flat plane, (0.6 sin 45 deg + 0.4) cos(0)^0.5 -> byte 210.
5. With a DEM: hillshades and slope on a 2000 x 2000 crop against gdaldem (GDAL). gdaldem sums the 3 x 3 window in
   float32 for float32 DEMs (slope up to 0.04 deg off at 3000 m heights); relief_gpu.py uses float64, hence the tolerances."""
import sys
import numpy as np
from osgeo import gdal
from relief_gpu import horizon, horn_products, composite

gdal.UseExceptions()
RES, N, Z0 = 0.5, 801, 2500.0
c = N // 2
i, j = np.mgrid[0:N, 0:N].astype(np.float64)
az = 2 * np.pi * np.arange(32) / 32                          # clockwise from north, as in the kernel
fails = []


def check(name, got_svf, got_opn, angles, svf_cont, tol_opn=2e-4):
    """angles: elevation angle of the horizon per direction (radians) for the analytic surface."""
    svf_x, opn_x = 1 - np.mean(np.sin(np.maximum(angles, 0))), 90 - np.degrees(np.mean(angles))
    e1, e2 = np.abs(got_svf - svf_x).max(), np.abs(got_opn - opn_x).max()
    ok = e1 < 2e-6 and e2 < tol_opn
    fails.extend([] if ok else [name])
    print(f"{name:34s} SVF {got_svf.mean():.6f} exact {svf_x:.6f} (|err| {e1:.1e}) continuous {svf_cont:.6f} | "
          f"openness {got_opn.mean():8.4f} exact {opn_x:8.4f} (|err| {e2:.1e})  {'ok' if ok else 'FAIL'}")


for radius in (30, 150):
    line = (slice(c, c + 1), slice(c - 40, c + 41))           # 81 pixels on the centre row
    s, o, _ = horizon(np.full((N, N), Z0, np.float32), RES, radius)
    check(f"flat plane r {radius} m", s[line], o[line], np.zeros(32), 1.0)
    for t in (0.5, 1.0):                                      # tan of the side slope; heights exact in float32
        a = np.arctan(t)
        z = (Z0 + t * np.abs(i - c) * RES).astype(np.float32)  # valley axis runs east-west along the centre row
        s, o, _ = horizon(z, RES, radius)
        check(f"V valley {np.degrees(a):.1f} deg r {radius} m", s[line], o[line],
              np.arctan(t * np.abs(np.cos(az))), 1 - 2 * a / np.pi)
        z = (Z0 + t * (j - c) * RES).astype(np.float32)        # plane rising towards east
        s, o, _ = horizon(z, RES, radius)
        check(f"inclined plane {np.degrees(a):.1f} deg r {radius} m", s[line], o[line],
              np.arctan(t * np.sin(az)), 1 - a / np.pi)
    a = np.arctan(0.75)
    z = (Z0 - 0.75 * np.hypot(i - c, j - c) * RES).astype(np.float32)
    s, o, _ = horizon(z, RES, radius)
    p = (slice(c, c + 1), slice(c, c + 1))
    check(f"cone summit {np.degrees(a):.1f} deg r {radius} m", s[p], o[p], np.full(32, -a), 1.0, tol_opn=1e-2)

rng = np.random.default_rng(1)
z = (Z0 + np.cumsum(np.cumsum(rng.normal(0, 0.05, (N, N)), 0), 1)).astype(np.float32)
z[300:340, 500:560] = np.nan
for radius in (30, 150):
    s1, o1, _ = horizon(z, RES, radius, rows=N)
    s2, o2, _ = horizon(z, RES, radius, rows=97)
    same = np.array_equal(s1, s2, equal_nan=True) and np.array_equal(o1, o2, equal_nan=True)
    fails.extend([] if same else [f"strips r {radius}"])
    print(f"strips of 97 rows vs one strip, r {radius} m: {'identical' if same else 'DIFFERENT'}")

P = 320                                                       # pad, px; box = N - 2P = 161 px
box = (slice(P, N - P), slice(P, N - P))
z = (Z0 + (j - c) * RES).astype(np.float32)                   # 45 deg plane rising towards east
s, o, _ = horizon(z, RES, 150)
check("padded 45 deg plane r 150 m, whole box", s[box], o[box], np.arctan(np.sin(az)), 0.75)
_, _, sl, _ = horn_products(z, RES)
e = np.abs(sl[box] - 45).max()
fails.extend([] if e < 1e-6 else ["padded slope"])
print(f"padded 45 deg plane, slope over the whole box: max |err| {e:.1e} deg  {'ok' if e < 1e-6 else 'FAIL'}")
s0, o0, _ = horizon(np.ascontiguousarray(z[box]), RES, 150)
print(f"  same box without the pad: SVF error at the box edge up to {np.abs(s0 - s[box]).max():.3f}, "
      f"openness up to {np.abs(o0 - o[box]).max():.1f} deg (what the pad removes)")
flat = np.full((50, 50), Z0, np.float32)
_, _, sl, mr = horn_products(flat, RES)
cb = composite(mr, np.ones_like(mr), np.ones_like(mr), sl)
ok = (cb[1:-1, 1:-1] == 210).all()
fails.extend([] if ok else ["composite flat"])
print(f"composite on a flat plane: {np.unique(cb[1:-1, 1:-1])} (expected 210)  {'ok' if ok else 'FAIL'}")

if len(sys.argv) > 1:
    r0, c0 = 9800, 7400                                       # 2000 x 2000 at the tongue, E 2679700-2680700 N 1141100-1142100
    crop = gdal.Translate("/vsimem/crop.tif", sys.argv[1], srcWin=[c0, r0, 2000, 2000])
    zc = crop.GetRasterBand(1).ReadAsArray()
    zc[zc == -9999] = np.nan
    hs, md, sl, _ = horn_products(zc, RES, rows=333)
    ref = {"hs": gdal.DEMProcessing("/vsimem/hs.tif", crop, "hillshade", azimuth=315, altitude=45).ReadAsArray(),
           "md": gdal.DEMProcessing("/vsimem/md.tif", crop, "hillshade", multiDirectional=True, altitude=45).ReadAsArray(),
           "sl": gdal.DEMProcessing("/vsimem/sl.tif", crop, "slope").ReadAsArray()}
    inner = (slice(1, -1), slice(1, -1))
    for k, mine in (("hs", hs), ("md", md), ("sl", sl)):
        d = np.abs(mine[inner].astype(np.float64) - ref[k][inner].astype(np.float64))
        tol = 0.05 if k == "sl" else 1
        fails.extend([] if d.max() <= tol else [f"gdaldem {k}"])
        print(f"gdaldem {k}: max |diff| {d.max():.2e}, identical pixels {np.mean(d == 0):.4f}, "
              f"edge nodata {'ok' if (hs[0] == 0).all() and np.isnan(sl[:, -1]).all() else 'not ok'}")

print("ALL CHECKS PASSED" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
