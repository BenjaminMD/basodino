"""Checks of the destriped sky-view factor against relief_v2 on the hero window (hero_svf.py WIN).

usage: python hero_compare.py [svf.tif]   (env agl on s02; default svf_r30m_destriped.tif; writes checks/compare_<svf>.json)

1 svf_r30m_seamless.tif (hero_svf.py on relief_v2's own DEM) against derived/relief_v2/svf_r30m_0p5m.tif: the window
  computation must reproduce relief_v2 bit for bit.
2 svf_r30m_destriped.tif against relief_v2 off the ice. The SVF of a pixel depends on the DEM within 61 px (60 steps of
  0.5 m plus the bilinear neighbour), so every pixel farther than 61 px (Chebyshev) from a DEM pixel the destriping changed
  must be bit-identical; the rest are listed by distance from the ice (destripe weight w = 1 on the ice mask).
3 stripes on the ice: 128 m tiles fully on the ice mask; R = power in the stripe band (|kx| 0.035-0.135, |ky| <= 0.012
  cycles/m: north-south stripes) over the same band turned by 90 degrees. Isotropic texture gives R near 1."""
import json, sys
import numpy as np
from osgeo import gdal
from scipy import ndimage as nd

sys.path.insert(0, "/data/befahl/projects/AGL/basodino/hero/scripts")
from hero_svf import WIN, RES                                     # noqa: E402

gdal.UseExceptions()
P = "/data/befahl/projects/AGL/basodino"
H = f"{P}/hero"


def read(path, win=WIN):
    src = gdal.Open(path)
    g = src.GetGeoTransform()
    x0, y0 = round((win[0] - g[0]) / RES), round((g[3] - win[3]) / RES)
    nx, ny = round((win[1] - win[0]) / RES), round((win[3] - win[2]) / RES)
    return src.GetRasterBand(1).ReadAsArray(x0, y0, nx, ny)


def stripe_ratio(a, tiles):
    n = tiles[0][2]
    k = np.fft.fftfreq(n, RES)
    kx, ky = np.abs(k)[None, :], np.abs(k)[:, None]
    band = (kx >= 0.035) & (kx <= 0.135) & (ky <= 0.012)
    hann = np.outer(np.hanning(n), np.hanning(n))
    r = []
    for (i, j, n) in tiles:
        t = a[i:i + n, j:j + n].astype(np.float64)
        f = np.abs(np.fft.fft2((t - t.mean()) * hann)) ** 2
        r.append(f[band].sum() / f[band.T].sum())
    return np.array(r)


def main(name="svf_r30m_destriped.tif"):
    v2 = read(f"{P}/derived/relief_v2/svf_r30m_0p5m.tif")
    rec = read(f"{H}/svf_r30m_seamless.tif")
    ds = read(f"{H}/{name}")
    dz = read(f"{P}/plates_inputs/dem_0p5m_destriped.tif") - read(f"{P}/derived/relief_v2/seamless_dem_0p5m.tif")
    w = read(f"{P}/plates_inputs/destripe_weight_0p5m.tif")
    out = {"window": WIN, "svf": name}
    out["recompute_vs_relief_v2"] = {"pixels": int(v2.size), "differing": int((rec != v2).sum()),
                                     "max_abs": float(np.abs(rec - v2).max())}
    changed = dz != 0
    reach = nd.binary_dilation(changed, structure=np.ones((123, 123), bool))   # 61 px Chebyshev
    ice = w == 255
    off = w == 0
    diff = np.abs(ds - v2)
    dist = nd.distance_transform_edt(~ice) * RES                                # m from the ice mask
    o = {"dem_changed_px": int(changed.sum()), "dem_changed_off_ice_px": int((changed & off).sum()),
         "dem_change_max_abs_cm_off_ice": float(100 * np.abs(dz[off]).max()) if (changed & off).any() else 0.0,
         "outside_ray_reach_px": int((~reach).sum()),
         "outside_ray_reach_differing_px": int((diff[~reach] != 0).sum())}
    offd = off & (diff != 0)
    o["off_ice_px"] = int(off.sum())
    o["off_ice_bit_identical_share"] = float(1 - offd.sum() / off.sum())
    o["off_ice_differing_px"] = int(offd.sum())
    if offd.any():
        dd = dist[offd]
        o["off_ice_differing_max_abs"] = float(diff[offd].max())
        o["off_ice_differing_p99_abs"] = float(np.percentile(diff[offd], 99))
        o["off_ice_differing_distance_from_ice_m_p50_p99_max"] = [float(x) for x in np.percentile(dd, [50, 99, 100])]
        o["off_ice_over_0p01_px"] = int((diff[off] > 0.01).sum())
        o["off_ice_over_0p01_beyond_10m_px"] = int(((diff > 0.01) & off & (dist > 10)).sum())
    out["destriped_vs_relief_v2"] = o
    n = 256
    tiles = [(i, j, n) for i in range(0, ice.shape[0] - n + 1, n // 2) for j in range(0, ice.shape[1] - n + 1, n // 2)
             if ice[i:i + n, j:j + n].all()]
    r0, r1 = stripe_ratio(v2, tiles), stripe_ratio(ds, tiles)
    out["stripes_on_ice"] = {"tiles_128m": len(tiles), "R_relief_v2_p10_p50_p90": [float(x) for x in np.percentile(r0, [10, 50, 90])],
                             "R_destriped_p10_p50_p90": [float(x) for x in np.percentile(r1, [10, 50, 90])],
                             "ice_px": int(ice.sum()), "ice_differing_share": float((diff[ice] != 0).mean()),
                             "ice_abs_diff_p50_p99": [float(x) for x in np.percentile(diff[ice], [50, 99])]}
    json.dump(out, open(f"{H}/checks/compare_{name[:-4]}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
