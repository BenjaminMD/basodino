"""Probe for single-pixel spikes: DEM pixels above (below) all 8 neighbours, and sky-view factor pixels brighter (darker)
than all 8 neighbours, on the hero window. Prints counts and the largest cases; writes checks/spike_probe.npz.

usage: python hero_spike_probe.py          (env agl on s02)"""
import sys
import numpy as np
from osgeo import gdal

sys.path.insert(0, "/data/befahl/projects/AGL/basodino/hero/scripts")
from hero_svf import WIN, RES                                    # noqa: E402
from hero_compare import read                                    # noqa: E402

P = "/data/befahl/projects/AGL/basodino"
H = f"{P}/hero"
LO, HI = 0.49008177638053896, 0.9237946897745132                 # checks/look.json


def n8(a):
    """max, min and median of the 8 neighbours (edge pixels: nan)."""
    ap = np.pad(a, 1, constant_values=np.nan)
    s = [ap[1 + di:ap.shape[0] - 1 + di, 1 + dj:ap.shape[1] - 1 + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)
         if (di, dj) != (0, 0)]
    s = np.stack(s)
    return np.nanmax(s, 0), np.nanmin(s, 0), np.nanmedian(s, 0)


def main():
    z = read(f"{P}/plates_inputs/dem_0p5m_destriped.tif").astype(np.float64)
    s = read(f"{H}/svf_r30m_destriped.tif").astype(np.float64)
    w = read(f"{P}/plates_inputs/destripe_weight_0p5m.tif")
    ice = w == 255
    zmax, zmin, zmed = n8(z)
    up, dn = z - zmax, zmin - z
    print("DEM strict local maxima / minima over all 8 neighbours, by margin (m):")
    for t in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
        print(f"  > {t:4.2f} m: up {int((up > t).sum()):7d} (ice {int((up > t)[ice].sum()):6d})   "
              f"down {int((dn > t).sum()):7d} (ice {int((dn > t)[ice].sum()):6d})")
    smax, smin, smed = n8(s)
    g = (HI - LO) / 255                                           # one grey level of the stretch
    su, sd = (s - smax) / g, (smin - s) / g
    print("SVF pixels brighter / darker than all 8 neighbours, by margin (grey levels of the stretch):")
    for t in (5, 10, 20, 40, 80):
        print(f"  > {t:3d}: bright {int((su > t).sum()):7d} (ice {int((su > t)[ice].sum()):6d})   "
              f"dark {int((sd > t).sum()):7d} (ice {int((sd > t)[ice].sum()):6d})")
    for name, m in (("DEM up", up), ("DEM down", dn), ("SVF dark", sd), ("SVF bright", su)):
        idx = np.argsort(m, axis=None)[::-1][:12]
        print(name, "largest:")
        for k in idx:
            i, j = np.unravel_index(k, m.shape)
            E, N = WIN[0] + (j + 0.5) * RES, WIN[3] - (i + 0.5) * RES
            print(f"   E {E:.2f} N {N:.2f}  margin {m[i, j]:8.3f}  z {z[i, j]:8.2f}  z-med {z[i, j] - zmed[i, j]:7.3f}  "
                  f"svf {s[i, j]:.3f}  ice {bool(ice[i, j])}")
    i, j = np.unravel_index(np.argmin(s), s.shape)
    print("SVF minimum", s[i, j], "at E", WIN[0] + (j + 0.5) * RES, "N", WIN[3] - (i + 0.5) * RES)
    np.savez_compressed(f"{H}/checks/spike_probe.npz", up=up.astype(np.float32), dn=dn.astype(np.float32),
                        su=su.astype(np.float32), sd=sd.astype(np.float32))


if __name__ == "__main__":
    main()
