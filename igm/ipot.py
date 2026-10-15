"""Potential clear-sky direct radiation I_pot on the lattice of inputs.py, daily means of a 365-day hydrological year.

Method (Hock 1999, the I of smb.py and of the reference smb_accmelt, W m-2 averaged over 24 h):
  I = I0 E0 psi^(p / (p0 cos Z)) cos(theta)  when the sun is above the horizon of the cell, else 0
  I0 = 1368 W m-2, E0 = (Rm / R)^2 and declination from Spencer (1971), psi = 0.75, p / p0 = exp(-z / 8434.5 m),
  cos(theta) = cos(beta) cos(Z) + sin(beta) sin(Z) cos(sun azimuth - aspect), negative values set to 0.
  Slope beta and aspect from usurf_2021 (np.gradient, 10 m); aspect and horizon azimuths in the grid frame (clockwise
  from LV95 grid north), the sun's azimuth turned into it by the meridian convergence at the domain centre; latitude of the domain centre; 96 steps of 15 min per day at their midpoints,
  local solar time. Day k = 0 is 1 October (day of year 274), as weather.py.
  Horizon: 72 azimuths of 5 deg; along each, sample points 10 m to 20 km from the cell centre, spacing growing by 4 %;
  elevation from swissALTI3D 2022 at 2 m (derived/dem_2m.tif, its 10 km square) and Copernicus GLO-30 (EGM2008, not
  shifted to LN02: horizon only) reprojected to 30 m LV95 for the Italian gaps and beyond the square; earth curvature
  with refraction coefficient 0.13. The sun's horizon is interpolated linearly between azimuths.
Gate (DESIGN.md step 1): the same code for a flat cell with no horizon and psi = 1 against the closed-form daily mean
extraterrestrial irradiance on a horizontal surface, I0 E0 / pi (cos(phi) cos(delta) sin(ws) + ws sin(phi) sin(delta)),
every day of the year; and the same flat cell with psi = 0.75 against 1-min steps (time discretisation).
Inputs: s02 igm/inputs/usurf_2021.tif (inputs.py), derived/dem_2m.tif, derived/italy/src/copdem/*.tif.
Outputs (s02 igm/inputs/): ipot.tif (365 bands, band k+1 = day k, float32, W m-2), horizon.tif (72 bands, azimuth 5 b
deg clockwise from grid north for band b+1, degrees); igm/results/ipot_gate.csv (day, analytic, model, relative error,
and the psi = 0.75 check), igm/results/ipot_summary.csv.
usage (s02, agl env):  python igm/ipot.py          (about 10 min; run under nrun)
"""
import csv
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Proj, Transformer
from rasterio.warp import Resampling, reproject
from scipy.ndimage import map_coordinates

from inputs import DER, DX, E0, E1, N0, N1, NX, NY, OUT, RES, TRANSFORM, centres, read

I0, PSI, H_SCALE, K_REFR, R_EARTH = 1368.0, 0.75, 8434.5, 0.13, 6.371e6
DOY0, NSTEP, NAZ, REACH = 274, 96, 72, 20000.0
COP = DER / "italy/src/copdem/Copernicus_DSM_COG_10_N46_00_E008_00_DEM.tif"


def sun(doy, nstep=NSTEP):
    """Spencer (1971): eccentricity factor E0, declination (rad); hour angles (rad) at the step midpoints."""
    g = 2 * np.pi * (doy - 1) / 365
    e0 = (1.000110 + 0.034221 * np.cos(g) + 0.001280 * np.sin(g) + 0.000719 * np.cos(2 * g)
          + 0.000077 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    return e0, dec, np.pi * ((np.arange(nstep) + 0.5) / nstep * 2 - 1)


def daily(doy, lat, z, slope, aspect, horizon, psi=PSI, nstep=NSTEP, gamma=0.0):
    """Daily mean I (W m-2) for cells (1-d arrays; horizon (cell, NAZ) in rad) on day of year doy; gamma = grid azimuth
    of true north times -1 (meridian convergence, rad)."""
    e0, dec, hs = sun(doy, nstep)
    sl, cl = np.sin(lat), np.cos(lat)
    tot = np.zeros_like(z)
    pr = np.exp(-z / H_SCALE)
    for h in hs:
        cz = sl * np.sin(dec) + cl * np.cos(dec) * np.cos(h)
        if cz <= 0:
            continue
        sz = np.sqrt(1 - cz * cz)
        az = (np.arctan2(-np.cos(dec) * np.sin(h), np.sin(dec) * cl - np.cos(dec) * sl * np.cos(h)) - gamma) % (2 * np.pi)
        f = az / (2 * np.pi / NAZ)
        k = int(f) % NAZ
        hz = horizon[:, k] + (f - int(f)) * (horizon[:, (k + 1) % NAZ] - horizon[:, k])
        ct = np.maximum(np.cos(slope) * cz + np.sin(slope) * sz * np.cos(az - aspect), 0.0)
        tot += np.where(np.arcsin(cz) > hz, I0 * e0 * psi ** (pr / cz) * ct, 0.0)
    return tot / nstep


def analytic(doy, lat):
    e0, dec, _ = sun(doy)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(dec), -1, 1))
    return I0 * e0 / np.pi * (np.cos(lat) * np.cos(dec) * np.sin(ws) + ws * np.sin(lat) * np.sin(dec))


def elevation():
    """Sampler (E, N) -> elevation: 2 m swissALTI3D where it has data, GLO-30 at 30 m LV95 elsewhere."""
    left, top, c = E0 - REACH, N1 + REACH, 30.0
    nx, ny = int((E1 - E0 + 2 * REACH) / c), int((N1 - N0 + 2 * REACH) / c)
    coarse = np.full((ny, nx), np.nan, "float32")
    with rasterio.open(COP) as r:
        reproject(rasterio.band(r, 1), coarse, dst_transform=rasterio.transform.from_origin(left, top, c, c),
                  dst_crs="EPSG:2056", resampling=Resampling.bilinear, dst_nodata=np.nan)
    assert not np.isnan(coarse).any()

    def at_coarse(e, n):
        return map_coordinates(coarse, [(top - n) / c - 0.5, (e - left) / c - 0.5], order=1, mode="nearest")

    with rasterio.open(DER / "dem_2m.tif") as r:
        fine, (fl, fb, fr, ft) = r.read(1), r.bounds
        gap = fine == r.nodata
    rr, cc = np.nonzero(gap)
    fine[gap] = at_coarse(fl + 2 * (cc + 0.5), ft - 2 * (rr + 0.5))
    stats = {"2 m cells filled from GLO-30, share of the 10 km square": gap.mean()}

    def at(e, n):
        inside = (e > fl + 2) & (e < fr - 2) & (n > fb + 2) & (n < ft - 2)
        out = at_coarse(e, n)
        out[inside] = map_coordinates(fine, [(ft - n[inside]) / 2 - 0.5, (e[inside] - fl) / 2 - 0.5], order=1)
        return out
    return at, stats


def horizons(z, at):
    """Horizon angle (rad), (cell, NAZ), for cells at the lattice centres with elevation z (flattened)."""
    E, N = (v.ravel() for v in centres())
    d = [10.0]
    while d[-1] < REACH:
        d.append(d[-1] * 1.04 + 2.0)
    out = np.full((E.size, NAZ), -np.pi / 2)
    for a in range(NAZ):
        sa, ca = np.sin(np.radians(a * 360 / NAZ)), np.cos(np.radians(a * 360 / NAZ))
        for dist in d:
            zz = at(E + dist * sa, N + dist * ca) - dist * dist * (1 - K_REFR) / (2 * R_EARTH)
            np.maximum(out[:, a], np.arctan((zz - z) / dist), out=out[:, a])
    return out


def main():
    z2 = read("usurf_2021").astype(float)
    E, N = centres()
    ec, nc = (E0 + E1) / 2, (N0 + N1) / 2
    lon, lat = Transformer.from_crs(2056, 4326, always_xy=True).transform(ec, nc)
    gamma = np.radians(Proj("EPSG:2056").get_factors(lon, lat).meridian_convergence)
    lat = np.radians(lat)

    # gate: flat cell, no horizon
    flat = (np.array([0.0]), np.array([0.0]), np.array([0.0]), np.full((1, NAZ), -np.pi / 2))
    rows, worst, worst75 = [], 0.0, 0.0
    for k in range(365):
        doy = (DOY0 - 1 + k) % 365 + 1
        a = analytic(doy, lat)
        m = daily(doy, lat, *flat, psi=1.0)[0]
        m75, f75 = daily(doy, lat, *flat)[0], daily(doy, lat, *flat, nstep=1440)[0]
        worst, worst75 = max(worst, abs(m / a - 1)), max(worst75, abs(m75 / f75 - 1))
        rows.append([k, doy, f"{a:.4f}", f"{m:.4f}", f"{m / a - 1:.2e}", f"{f75:.4f}", f"{m75:.4f}", f"{m75 / f75 - 1:.2e}"])
    with open(RES / "ipot_gate.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "doy", "analytic_psi1", "model_psi1", "rel_err", "fine_psi075", "model_psi075", "rel_err_075"])
        w.writerows(rows)
    print(f"gate: max relative error psi = 1 {worst:.2e}; psi = 0.75, 15 min against 1 min {worst75:.2e}")

    at, stats = elevation()
    hz = horizons(z2.ravel(), at)
    gy, gx = np.gradient(z2, DX)
    gy = -gy                                                        # rows run north to south
    slope = np.arctan(np.hypot(gx, gy)).ravel()
    aspect = (np.arctan2(-gx, -gy) % (2 * np.pi)).ravel()          # downslope direction, clockwise from grid north
    ip = np.stack([daily((DOY0 - 1 + k) % 365 + 1, lat, z2.ravel(), slope, aspect, hz, gamma=gamma) for k in range(365)])

    prof = dict(driver="GTiff", width=NX, height=NY, dtype="float32", crs="EPSG:2056", transform=TRANSFORM,
                compress="deflate")
    with rasterio.open(OUT / "ipot.tif", "w", count=365, **prof) as r:
        r.write(ip.reshape(365, NY, NX).astype("float32"))
    with rasterio.open(OUT / "horizon.tif", "w", count=NAZ, **prof) as r:
        r.write(np.degrees(hz.T.reshape(NAZ, NY, NX)).astype("float32"))
    m21 = read("mask_2021").ravel() == 1
    stats.update({"meridian convergence deg": np.degrees(gamma), "latitude deg": np.degrees(lat),
                  "gate max relative error, psi = 1": worst, "time-step error psi = 0.75": worst75,
                  "annual mean I_pot domain W m-2": ip.mean(), "annual mean I_pot 2021 mask W m-2": ip[:, m21].mean(),
                  "21 December mean on the 2021 mask W m-2": ip[81, m21].mean(),
                  "21 June mean on the 2021 mask W m-2": ip[263, m21].mean(),
                  "median horizon on the 2021 mask deg": np.degrees(np.median(hz[m21]))})
    with open(RES / "ipot_summary.csv", "w", newline="") as f:
        csv.writer(f).writerows([["quantity", "value"], *[[k, f"{v:.6g}"] for k, v in stats.items()]])
    for k, v in stats.items():
        print(k, f"{v:.6g}", sep="\t")


if __name__ == "__main__":
    main()
