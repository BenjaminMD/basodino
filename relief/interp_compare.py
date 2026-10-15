"""Which interpolator puts the Piemonte DTM on the 0.5 m grid most cleanly?   usage: python interp_compare.py <project dir>
Candidates, all in LN02 (-1.913 m), in two windows where swissALTI3D and Piemonte overlap (Italy, > 300 m inside both):
  A  cubic B-spline (interpolating, order 3 with prefilter) on the 2 m LV95 grid italy_dem_2m_ln02.tif, which stage A
     made by GDAL cubic convolution from the 5 m UTM source: two interpolations in a row
  B  GDAL bicubic (cubic convolution, a = -0.5) from the native 5 m UTM grid in one step (gdalwarp -r cubic, exact transform)
  C  cubic B-spline from the native 5 m UTM grid in one step (seamless_dem.piemonte, the method used)
Scores against swissALTI3D 0.5 m (2019/2020 lidar): median, NMAD and RMS of the difference, all pixels and slopes below
30 degrees. Texture: RMS of the 315/45 hillshade (0..1) at wavelengths below 10 m, above the Nyquist frequency of the 5 m
source: the source holds no information there, so that energy is made by the interpolator (swissALTI3D for scale).
Figure: relief_v2/checks/interp_compare.png, 100 m x 100 m at native 0.5 m."""
import os, sys
import numpy as np, cupy as cp
from scipy import ndimage
from osgeo import gdal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from seamless_dem import swiss, piemonte, RES
from relief_gpu import gradient, shade

gdal.UseExceptions()
WINDOWS = {"north-west, E 2676000-2677000 N 1145000-1146000": (2676000, 1146000, 1000),
           "west, E 2678050-2679050 N 1139000-1140000": (2678050, 1140000, 1000)}   # (west, north, size m)
OFFSET = 1.913


def cand_a(d, gt, n):
    ds = gdal.Open(os.path.join(d, "derived/italy/italy_dem_2m_ln02.tif"))
    z = ds.ReadAsArray().astype(np.float64)
    valid = z != -9999
    z = z[tuple(ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True))]
    g2 = ds.GetGeoTransform()
    c = (gt[0] - g2[0]) / 2 + (np.arange(n) + 0.5) / 4 - 0.5
    r = (g2[3] - gt[3]) / 2 + (np.arange(n) + 0.5) / 4 - 0.5
    rr, cc = np.meshgrid(r, c, indexing="ij")
    return ndimage.map_coordinates(z, [rr, cc], order=3, mode="mirror")


def cand_b(d, gt, n):
    src = gdal.BuildVRT("/vsimem/p.vrt", [os.path.join(d, f"raw/piemonte_dtm5/DTM5_{s}.tif") for s in (20, 36)],
                        srcNodata=-99, VRTNodata=-99)
    w = gdal.Warp("/vsimem/b.tif", src, dstSRS="EPSG:2056", outputBounds=(gt[0], gt[3] - n * RES, gt[0] + n * RES, gt[3]),
                  xRes=RES, yRes=RES, resampleAlg="cubic", errorThreshold=0, outputType=gdal.GDT_Float64,
                  srcNodata=-99, dstNodata=np.nan)
    return w.ReadAsArray() - OFFSET


def hillshade(z):
    ge, gn = gradient(np.pad(z, 1, mode="edge"), RES)
    return shade(ge, gn, 315).get()


def highpass(h, cut=0.1):
    """h minus its content at spatial frequencies below cut (cycles per metre), with a Hann taper; returns map and RMS."""
    n = h.shape[0]
    w = np.outer(np.hanning(n), np.hanning(n))
    f = np.fft.fftfreq(n, RES)
    keep = np.hypot(*np.meshgrid(f, f, indexing="ij")) > cut
    F = np.fft.fft2((h - h.mean()) * w)
    rms = np.sqrt(np.sum(np.abs(F[keep]) ** 2) / h.size / np.sum(w ** 2))
    return np.real(np.fft.ifft2(F * keep)), rms


def stats(x):
    m = np.median(x)
    return m, 1.4826 * np.median(np.abs(x - m)), np.sqrt(np.mean((x - m) ** 2))


def main(d):
    os.makedirs(os.path.join(d, "relief_v2/checks"), exist_ok=True)
    shown = {}
    for wname, (w, nrth, size) in WINDOWS.items():
        n = int(size / RES)
        gt = (w, RES, 0.0, nrth, 0.0, -RES)
        zs = swiss(d, gt, (n, n)).astype(np.float64)
        cands = {"A spline on the 2 m grid": cand_a(d, gt, n), "B bicubic convolution from 5 m": cand_b(d, gt, n),
                 "C spline from 5 m": piemonte(d, gt, (n, n), OFFSET)}
        assert not np.isnan(zs).any() and all(not np.isnan(v).any() for v in cands.values())
        ge, gn = gradient(np.pad(zs, 1, mode="edge"), RES)
        gentle = (np.degrees(np.arctan(np.hypot(ge, gn))) < 30).get()
        print(f"window {wname}: {n} x {n} px, slope below 30 degrees on {gentle.mean() * 100:.0f} %")
        _, hf_s = highpass(hillshade(zs))
        print(f"  swissALTI3D itself: hillshade RMS below 10 m wavelength {hf_s:.4f}")
        for name, z in cands.items():
            dz = z - zs
            a, b = stats(dz), stats(dz[gentle])
            _, hf = highpass(hillshade(z))
            print(f"  {name:32s} minus swissALTI3D: median {a[0]:+.3f} NMAD {a[1]:.3f} RMS {a[2]:.3f} m | slope < 30: "
                  f"median {b[0]:+.3f} NMAD {b[1]:.3f} RMS {b[2]:.3f} m | hillshade RMS below 10 m {hf:.4f}")
        for name, z in cands.items():
            for other, zo in cands.items():
                if other > name:
                    print(f"  {name[0]} minus {other[0]}: RMS {np.sqrt(np.mean((z - zo) ** 2)):.3f} m, max |d| {np.abs(z - zo).max():.3f} m")
        if not shown:
            shown = {**cands, "swissALTI3D 0.5 m (reference)": zs}
    # figure: 100 m x 100 m of the first window at native resolution
    sl = (slice(700, 900), slice(700, 900))
    fig = plt.figure(figsize=(12, 6.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.06], wspace=0.08, hspace=0.12)
    for k, (name, z) in enumerate(shown.items()):
        h = hillshade(z)
        hp, _ = highpass(h)
        ax = fig.add_subplot(gs[0, k])
        im0 = ax.imshow(h[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        ax = fig.add_subplot(gs[1, k])
        im1 = ax.imshow(hp[sl], cmap="RdBu_r", vmin=-0.15, vmax=0.15, interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
    for row, im, label in ((0, im0, "Hillshade 315/45 (reflectance)"), (1, im1, "Hillshade part below 10 m wavelength")):
        cax = fig.add_subplot(gs[row, 4])
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 + 0.1 * pos.height, pos.width, 0.8 * pos.height])
        fig.colorbar(im, cax=cax).set_label(label, fontsize=8)
    fig.suptitle("Piemonte DTM 5 m to 0.5 m: 100 m x 100 m near E 2676400 N 1145600, native pixels", fontsize=10)
    fig.savefig(os.path.join(d, "relief_v2/checks/interp_compare.png"), dpi=300, bbox_inches="tight")
    print("figure relief_v2/checks/interp_compare.png")


if __name__ == "__main__":
    main(sys.argv[1])
