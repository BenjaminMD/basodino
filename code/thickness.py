"""Ice thickness and bed of Ghiacciaio del Basòdino on the course grid: volumes, the check of volume-area scaling,
two sections.

Inputs:
  data/thickness/C14-10_*.tif      Swiss Glacier Thickness, release 2020 (Grab and fifteen co-authors 2021; provenance
                                   and licence in data/thickness/README.md): IceThickness, GlacierBed, SwissALTI3D_r2019
                                   (the surface the thickness refers to), IceThicknessUncertainty{Minus,Plus},
                                   ReferenceYear (band 1: most common year of that surface per cell)
  data/course/dem_basodino_{1994,2021}.asc, mask_basodino_{1994,2021}.asc   10 m, Swiss coordinates (EPSG:2056)
Method:
  1. The data grids lie on the 10 m lattice of the course grid (asserted: whole-cell offsets, here 7 columns west and
     16 rows south of its north-west corner); each is pasted into a course-sized array, no resampling; no ice cell
     falls outside (asserted). The published bed is the surface of the reference year minus the thickness (asserted
     to 0.01 m).
  2. Reference year: the published thickness on its own area (the outline of the Swiss Glacier Inventory 2016, imagery
     of 2018; not the glacier area of the reference year).
     Uncertainty range: every cell at its lower or at its upper bound at once, the widest range.
  3. 1994 and 2021: ground = the published bed, but never above the 2021 surface (a bed cannot lie above it), and the
     2021 surface off the outline of the data. Volume = sum over the year's mask of max(surface - ground, 0) x 100 m2,
     mean thickness = volume / mask area. For 2021 this is max(z_2021 - bed, 0), and 0 on mask cells without a bed.
     Where the published bed lies above the 2021 surface, the published 2012 thickness is too small by at least the
     difference; its sum is written out.
  4. Volume-area scaling V = c A^gamma, c = 0.034 km^(1/4) = 0.191 m^(1/4), gamma = 1.375 (Task 5), compared with
     the data on the same area as the ratio data / scaling.
  5. Alignment: d = z_2021 - surface of the reference year on ice-free ground (outside both masks and the outline of
     the data). Least squares d = dz + s_e dz/de + s_n dz/dn gives the displacement (s_e, s_n) of the data's
     surface relative to the course grid; a misplaced lattice (cell edges taken for centres) would show as 5 m.
  6. Sections: straight lines sampled every 10 m from their first end; elevations bilinear between the four nearest
     cell centres (the bed only where all four have one), masks from the cell under the point.
     A: from the summit of Basòdino (highest cell of the course grid) through the lowest cell of the 2021 mask (the
        tongue) to 80 m beyond: the long axis of the glacier, at about 40 degrees to its north-eastward flow.
     B: across the main body, perpendicular to the flow, along the radar profiles 000aac and 000aad of 2006 (the line
        through the first point of 000aad, 2679581 / 1141170, and its bend, 2680422 / 1140612; both profiles stay
        within 26 m of it), extended to 50 m beyond the 1994 margins.
Outputs:
  code/thickness.csv            quantity, value, unit, definition (SI)
  data/thickness/sections.csv   section, distance_m, easting_m, northing_m, bed_m, surface_1994_m, surface_2021_m,
                                surface_2012_m (the reference year), ice_1994, ice_2021 (1 = in that year's mask);
                                bed_m is empty off the outline of the data, where the ground is the 2021 surface
usage:  uv run --with rasterio --with numpy python code/thickness.py
"""
import csv

import numpy as np
import rasterio

from common import COURSE, ROOT, read_esri_ascii, write_csv

DATA = ROOT / "data/thickness"
C, GAMMA = 0.034 * 1000 ** 0.25, 1.375                      # m^(1/4), dimensionless
SECTIONS = {"A": ((2679177, 1140547), (2681520, 1141093)),  # summit to beyond the tongue, through 2681436 / 1141073
            "B": ((2679198, 1141424), (2680648, 1140462))}  # north-west to south-east, along the 2006 radar line

z94, hd = read_esri_ascii(COURSE / "dem_basodino_1994.asc")
z21, _ = read_esri_ascii(COURSE / "dem_basodino_2021.asc")
g94 = read_esri_ascii(COURSE / "mask_basodino_1994.asc")[0] == 1
g21 = read_esri_ascii(COURSE / "mask_basodino_2021.asc")[0] == 1
CS, X0 = hd["cellsize"], hd["xllcorner"]
NY, NX = z21.shape
Y0 = hd["yllcorner"] + NY * CS                              # northern edge
CELL = CS * CS                                              # m2


def cut(name, band=1):
    """A grid of the data set on the course grid, and the number of its valid cells."""
    with rasterio.open(DATA / f"C14-10_{name}.tif") as src:
        assert src.crs.to_epsg() == 2056 and src.res == (CS, CS)
        a = src.read(band, masked=True).astype(float).filled(np.nan)
        c0, r0 = (src.transform.c - X0) / CS, (Y0 - src.transform.f) / CS
    assert c0 == round(c0) and r0 == round(r0), "cell edges off the course lattice"
    c0, r0 = round(c0), round(r0)
    rows, cols = slice(max(r0, 0), min(r0 + a.shape[0], NY)), slice(max(c0, 0), min(c0 + a.shape[1], NX))
    out = np.full((NY, NX), np.nan)
    out[rows, cols] = a[rows.start - r0:rows.stop - r0, cols.start - c0:cols.stop - c0]
    return out, np.isfinite(a).sum()


def centre(field):
    """Swiss coordinates of the cell centre of the largest value."""
    i, j = np.unravel_index(np.nanargmax(field), field.shape)
    return int(X0 + (j + 0.5) * CS), int(Y0 - (i + 0.5) * CS), (i, j)


h, n_ice = cut("IceThickness")
assert np.isfinite(h).sum() == n_ice, "ice cells outside the course grid"
bed, s12 = cut("GlacierBed")[0], cut("SwissALTI3D_r2019")[0]
lo, hi = cut("IceThicknessUncertaintyMinus")[0], cut("IceThicknessUncertaintyPlus")[0]
ice = np.isfinite(h)                                        # outline of the data (Swiss Glacier Inventory 2016)
assert np.nanmax(np.abs(s12 - h - bed)[ice]) < 0.01, "bed is not surface minus thickness"
years = cut("ReferenceYear")[0][ice]
years, counts = np.unique(years[np.isfinite(years)], return_counts=True)
year = int(years[counts.argmax()])

# --- volumes
ground = np.fmin(bed, z21)                                  # fmin: the 2021 surface where there is no bed


def volume(z, mask):
    return np.clip(z - ground, 0, None)[mask].sum() * CELL


a12, v12 = ice.sum() * CELL, np.nansum(h) * CELL
v12_lo, v12_hi = np.nansum(np.clip(h + lo, 0, None)) * CELL, np.nansum(h + hi) * CELL
a21, v21, a94, v94 = g21.sum() * CELL, volume(z21, g21), g94.sum() * CELL, volume(z94, g94)
e12, n12, k12 = centre(h)
h21 = np.where(g21, z21 - ground, np.nan)
e21, n21, _ = centre(h21)
above = np.clip(bed - z21, 0, None)                         # height of the bed above the 2021 surface, NaN off the data


def scaling(area):
    return C * area ** GAMMA


# --- alignment on ice-free ground
stable = ~g94 & ~g21 & ~ice & np.isfinite(s12)
d = (z21 - s12)[stable]
dz_drow, dz_de = np.gradient(z21, CS)                       # rows run south: dz/dnorth = -dz_drow
fit = np.linalg.lstsq(np.c_[np.ones(d.size), dz_de[stable], -dz_drow[stable]], d, rcond=None)[0]

numbers = [
    ("reference_year", year, "", "most common year of the surface the thickness refers to, on the ice cells"),
    ("A_2012", a12, "m2", "cells with a thickness x 100 m2: outline of the Swiss Glacier Inventory 2016 "
                          "(imagery 2018), not the glacier area of 2012"),
    ("V_2012", v12, "m3", "sum of the thickness x 100 m2"),
    ("H_2012_mean", v12 / a12, "m", "V_2012 / A_2012"),
    ("V_2012_low", v12_lo, "m3", "every cell at thickness + uncertainty minus (at least 0)"),
    ("V_2012_high", v12_hi, "m3", "every cell at thickness + uncertainty plus"),
    ("H_2012_max", h[k12], "m", "thickest cell"),
    ("H_2012_max_easting", e12, "m", "its centre, Swiss coordinates"),
    ("H_2012_max_northing", n12, "m", "its centre, Swiss coordinates"),
    ("H_2012_max_surface", s12[k12], "m", "surface elevation of the reference year there"),
    ("A_2021", a21, "m2", "cells of mask 2021 x 100 m2 (Task 1)"),
    ("A_2021_without_bed", (g21 & ~ice).sum() * CELL, "m2", "part of A_2021 outside the outline: thickness 0"),
    ("V_2021", v21, "m3", "sum over mask 2021 of max(z_2021 - bed, 0) x 100 m2"),
    ("H_2021_mean", v21 / a21, "m", "V_2021 / A_2021"),
    ("H_2021_max", np.nanmax(h21), "m", "thickest cell of mask 2021, z_2021 - bed"),
    ("H_2021_max_easting", e21, "m", "its centre, Swiss coordinates"),
    ("H_2021_max_northing", n21, "m", "its centre, Swiss coordinates"),
    ("A_1994", a94, "m2", "cells of mask 1994 x 100 m2 (Task 1)"),
    ("V_1994", v94, "m3", "sum over mask 1994 of max(z_1994 - ground, 0) x 100 m2; ground: bed, at most z_2021, "
                          "z_2021 off the outline"),
    ("H_1994_mean", v94 / a94, "m", "V_1994 / A_1994"),
    ("bed_above_2021_area", (above > 0).sum() * CELL, "m2", "cells with a bed above the 2021 surface x 100 m2"),
    ("bed_above_2021_in_mask", ((above > 0) & g21).sum() * CELL, "m2", "the part of bed_above_2021_area in mask 2021"),
    ("bed_above_2021_max", np.nanmax(above), "m", "largest height of the bed above the 2021 surface"),
    ("bed_above_2021_volume", np.nansum(above) * CELL, "m3",
     "sum of (bed - z_2021) there x 100 m2; V_2012 is too small by at least this"),
    ("V_scaling_2012", scaling(a12), "m3", "c A_2012^gamma"),
    ("ratio_2012", v12 / scaling(a12), "", "V_2012 / V_scaling_2012"),
    ("ratio_2012_high", v12_hi / scaling(a12), "", "V_2012_high / V_scaling_2012"),
    ("V_scaling_2021", scaling(a21), "m3", "c A_2021^gamma (Task 5)"),
    ("ratio_2021", v21 / scaling(a21), "", "V_2021 / V_scaling_2021"),
    ("V_scaling_1994", scaling(a94), "m3", "c A_1994^gamma"),
    ("ratio_1994", v94 / scaling(a94), "", "V_1994 / V_scaling_1994"),
    ("stable_cells", int(stable.sum()), "", "ice-free cells: outside both masks and the outline of the data"),
    ("dz_stable_mean", d.mean(), "m", "mean of z_2021 - surface of the reference year there"),
    ("dz_stable_median", np.median(d), "m", "median of the same"),
    ("dz_stable_std", d.std(), "m", "standard deviation of the same"),
    ("shift_east", fit[1], "m", "displacement of the data's surface, least squares on the ice-free cells"),
    ("shift_north", fit[2], "m", "the same, northward"),
    ("V_2021_offset_removed", volume(z21 - d.mean(), g21), "m3", "V_2021 with z_2021 lowered by dz_stable_mean"),
]
write_csv("thickness.csv", ("quantity", "value", "unit", "definition"), numbers)
for q, v, u, _ in numbers:
    print(f"{q:24s} {v:14.6g} {u}")


# --- sections
def bilinear(z, e, n):
    """z at the points (e, n), linear between the four nearest cell centres; NaN if one of them is NaN."""
    c, r = (e - X0) / CS - 0.5, (Y0 - n) / CS - 0.5
    j, i = np.floor(c).astype(int), np.floor(r).astype(int)
    fc, fr = c - j, r - i
    return (1 - fr) * ((1 - fc) * z[i, j] + fc * z[i, j + 1]) + fr * ((1 - fc) * z[i + 1, j] + fc * z[i + 1, j + 1])


def under(mask, e, n):
    return mask[((Y0 - n) // CS).astype(int), ((e - X0) // CS).astype(int)].astype(int)


with open(DATA / "sections.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(("section", "distance_m", "easting_m", "northing_m", "bed_m", "surface_1994_m", "surface_2021_m",
                f"surface_{year}_m", "ice_1994", "ice_2021"))
    for name, ((ea, na), (eb, nb)) in SECTIONS.items():
        length = np.hypot(eb - ea, nb - na)
        s = np.arange(0, length, 10.0)
        e, n = ea + s / length * (eb - ea), na + s / length * (nb - na)
        z = [bilinear(g, e, n) for g in (bed, z94, z21, s12)]
        m = [under(g, e, n) for g in (g94, g21)]
        for k in range(s.size):
            w.writerow([name, f"{s[k]:.0f}", f"{e[k]:.1f}", f"{n[k]:.1f}"]
                       + ["" if np.isnan(v[k]) else f"{v[k]:.2f}" for v in z] + [m[0][k], m[1][k]])
        print(f"section {name}: {length:.0f} m, {s.size} points, on the 2021 ice {m[1].sum() * 10} m, "
              f"bed above the 2021 surface at {np.sum(z[0] > z[2])} points")
