"""Geometry of the ensemble on the 10 m lattice: decisions 2 and 8 of igm/DESIGN.md and the targets that need cells.

Lattice: E 2678700-2682100, N 1140000-1142900, 340 x 290 cells of 10 m, rows north to south (GeoTIFF order), the
cell edges on the course-grid lattice (multiples of 10 m; asserted for every source grid pasted in).
  usurf_1994, usurf_2021  course DEMs (1994 = the GLAMOS survey of 1991-09-18, answer 4) on their 260 x 153 extent;
                          elsewhere swissALTI3D 2022 (derived/dem_2m.tif) averaged over the 5 x 5 cells of 2 m, for both
                          years (off the glacier the course 1994 grid is the modern terrain, TARGETS.md section 2)
  mask_1994, mask_2021    course masks, 0 outside their extent
  sgi1850, sgi1931, sgi1973, sgi2010, sgi2016, tlm2024, s2_2026, s2_2026_body, c14_11_1850
                          C14-10 outlines (all polygons; s2_2026_body = the first polygon of data/outline_2026), and
                          the 1850 outline of C14-11; a cell belongs to an outline when its centre lies inside
  posbal                  decision 2: sgi1850, or within 100 m of it, slope of usurf_2021 below 30 deg and not c14_11_1850
  bed_grab, sigma_plus, sigma_minus   Grab et al. (2021) bed and the magnitudes of its uncertainty grids (NaN outside)
  taper                   w = min(1, d / 50 m), d = distance of the cell centre inside the 2021 mask edge
                          (Euclidean distance to the nearest cell off the mask minus half a cell), 0 off the mask
  bed_u0                  bed(0) of decision 8 (function bed below), for checks and the preview
  itslive_pixel           index 0-95 of the ITS_LIVE 120 m pixel (EPSG:32632, GeoTransform 400072.5 120 0 5199967.5 0
                          -120) holding the cell, for the 96 pixels whose centre lies inside the 2024 outline; -1 else
bed(u): b = bed_grab - u sigma w (sigma = sigma_plus for u > 0, sigma_minus for u < 0); on the 2021 mask
min(b, usurf_2021 - 1 m), and usurf_2021 - 1 m where the Grab grid has no value; off the 2021 mask the ground usurf_2021.
Bound of TARGETS.md section 7 and decision 9: median 2014-2021 surface speed over the 96 pixels <= 1 m a-1, sigma
0.5 m a-1 above (constants ITS_BOUND, ITS_SIGMA).
Inputs (s02 derived/): course/{dem,mask}_basodino_{1994,2021}.asc, dem_2m.tif, vectors.gpkg (sgi_1850 ... glacier_2024),
outline_2026/outline_2026-09-27.gpkg, thickness/C14-10_{GlacierBed,IceThicknessUncertaintyPlus,Minus}.tif,
thickness/C14-10_basodino_GlaThiDa.txt.
Outputs (s02 igm/inputs/): <name>.tif per grid above (float32, masks uint8, itslive_pixel int16, LV95);
radar_2006.csv (point, profile 1-6 = runs split at gaps over 30 m, E, N, row, col, z_dem 2012, bedrock, thickness,
usurf_2021 of the cell, above_2021 = bedrock above it); itslive_pixels.csv (pixel, x_utm, y_utm, E, N, cells);
igm/results/inputs_summary.csv (area per mask, V_2021 at u = -1, 0, 1, cell counts of the rules, checks).
usage (s02, agl env):  python igm/inputs.py ;  from inputs import read, bed
"""
import csv
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]                      # s02:/data/befahl/projects/AGL/basodino
DER, OUT, RES = ROOT / "derived", ROOT / "igm/inputs", ROOT / "igm/results"
E0, E1, N0, N1, DX = 2678700, 2682100, 1140000, 1142900, 10
NX, NY = (E1 - E0) // DX, (N1 - N0) // DX                      # 340, 290
TRANSFORM = from_origin(E0, N1, DX, DX)
ITS_GT, ITS_BOUND, ITS_SIGMA = (400072.5, 120.0, 5199967.5), 1.0, 0.5
OUTLINES = {"sgi1850": "sgi_1850", "sgi1931": "sgi_1931", "sgi1973": "sgi_1973", "sgi2010": "sgi_2010",
            "sgi2016": "sgi_2016", "tlm2024": "glacier_2024"}


def centres():
    """Cell-centre coordinates (E, N), each (NY, NX)."""
    return np.meshgrid(E0 + DX * (np.arange(NX) + 0.5), N1 - DX * (np.arange(NY) + 0.5))


def paste(z, left, top, dx=DX):
    """Grid z with upper-left corner (left, top) and cell dx = 10 m -> lattice array (NaN outside); asserts alignment."""
    assert dx == DX and (left - E0) % DX == 0 and (N1 - top) % DX == 0, (left, top, dx)
    c, r = int(left - E0) // DX, int(N1 - top) // DX
    out = np.full((NY, NX), np.nan)
    r0, c0 = max(r, 0), max(c, 0)
    r1, c1 = min(r + z.shape[0], NY), min(c + z.shape[1], NX)
    out[r0:r1, c0:c1] = z[r0 - r:r1 - r, c0 - c:c1 - c]
    return out


def ascii_grid(name):
    """Course Esri ASCII grid -> lattice array."""
    path = DER / "course" / name
    with open(path) as f:
        h = {k.lower(): float(v) for k, v in (next(f).split() for _ in range(6))}
    z = np.loadtxt(path, skiprows=6)
    z[z == h["nodata_value"]] = np.nan
    return paste(z, h["xllcorner"], h["yllcorner"] + h["nrows"] * h["cellsize"], h["cellsize"])


def tif_grid(path):
    """10 m GeoTIFF -> lattice array."""
    with rasterio.open(path) as r:
        z = r.read(1).astype(float)
        z[z == r.nodata] = np.nan
        return paste(z, r.bounds.left, r.bounds.top, r.res[0])


def dem10():
    """swissALTI3D 2022 (2 m) averaged to the lattice."""
    with rasterio.open(DER / "dem_2m.tif") as r:
        assert r.res == (2.0, 2.0) and (r.bounds.left - E0) % 2 == 0
        z = r.read(1, window=rasterio.windows.from_bounds(E0, N0, E1, N1, r.transform)).astype(float)
        assert z.shape == (NY * 5, NX * 5) and not (z == r.nodata).any()
    return z.reshape(NY, 5, NX, 5).mean(axis=(1, 3))


def burn(geoms):
    """Polygons -> uint8 lattice mask (cell centre inside)."""
    return rasterize(((g, 1) for g in geoms), out_shape=(NY, NX), transform=TRANSFORM, dtype="uint8")


def slope_deg(z):
    gy, gx = np.gradient(z, DX)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def read(name):
    """A grid written by this script -> array (float grids with NaN)."""
    with rasterio.open(OUT / f"{name}.tif") as r:
        assert r.transform == TRANSFORM and r.shape == (NY, NX), name
        z = r.read(1)
        return np.where(z == r.nodata, np.nan, z) if r.nodata is not None else z


def bed(u, g):
    """Decision 8: bed(u) on the lattice from the grids g (dict, or read() names) for -1 <= u <= 1."""
    sigma = g["sigma_plus"] if u > 0 else g["sigma_minus"]
    b = g["bed_grab"] - u * sigma * g["taper"]
    top = g["usurf_2021"] - 1.0
    on = g["mask_2021"] == 1
    return np.where(on, np.where(np.isnan(b), top, np.minimum(b, top)), g["usurf_2021"])


def write(name, z):
    kind = {"uint8": ("uint8", None), "int16": ("int16", None)}.get(str(z.dtype), ("float32", -9999.0))
    with rasterio.open(OUT / f"{name}.tif", "w", driver="GTiff", width=NX, height=NY, count=1, dtype=kind[0],
                       crs="EPSG:2056", transform=TRANSFORM, nodata=kind[1], compress="deflate") as r:
        r.write(np.where(np.isnan(z), kind[1], z).astype(kind[0]) if kind[0] == "float32" else z, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g, rows = {}, []
    base = dem10()
    for y in (1994, 2021):
        c = ascii_grid(f"dem_basodino_{y}.asc")
        g[f"usurf_{y}"] = np.where(np.isnan(c), base, c)
        m = ascii_grid(f"mask_basodino_{y}.asc")
        g[f"mask_{y}"] = (m == 1).astype("uint8")
    d = (ascii_grid("dem_basodino_2021.asc") - base)[g["mask_2021"] == 1]
    rows.append(("course 2021 minus swissALTI3D 2022 on the 2021 mask, mean and SD m", f"{d.mean():.3f}", f"{d.std():.3f}"))

    vec = DER / "vectors.gpkg"
    for name, layer in OUTLINES.items():
        v = gpd.read_file(vec, layer=layer)
        v = v[v[{"sgi_2016": "sgi-id", "glacier_2024": "sgi"}.get(layer, "SGI")] == "C14-10"]
        g[name] = burn(v.geometry)
    s1850 = gpd.read_file(vec, layer="sgi_1850")
    g["c14_11_1850"] = burn(s1850[s1850.SGI == "C14-11"].geometry)
    o26 = gpd.read_file(DER / "outline_2026/outline_2026-09-27.gpkg").geometry.iloc[0]
    g["s2_2026"], g["s2_2026_body"] = burn([o26]), burn([o26.geoms[0]])

    buf = burn(s1850[s1850.SGI == "C14-10"].geometry.buffer(100.0))
    flat = slope_deg(g["usurf_2021"]) < 30.0
    g["posbal"] = ((g["sgi1850"] == 1) | ((buf == 1) & flat & (g["c14_11_1850"] == 0))).astype("uint8")

    th = DER / "thickness"
    g["bed_grab"] = tif_grid(th / "C14-10_GlacierBed.tif")
    g["sigma_plus"] = np.abs(tif_grid(th / "C14-10_IceThicknessUncertaintyPlus.tif"))
    g["sigma_minus"] = np.abs(tif_grid(th / "C14-10_IceThicknessUncertaintyMinus.tif"))
    on = g["mask_2021"] == 1
    dist = DX * distance_transform_edt(on) - DX / 2
    g["taper"] = np.where(on, np.minimum(1.0, dist / 50.0), 0.0)
    g["bed_u0"] = bed(0.0, g)

    # ITS_LIVE pixels whose centre lies inside the 2024 outline, and the lattice cells they hold
    t24 = gpd.read_file(vec, layer="glacier_2024")
    t24 = t24[t24.sgi == "C14-10"].to_crs(32632).union_all()
    to_utm = Transformer.from_crs(2056, 32632, always_xy=True)
    E, N = centres()
    xu, yu = to_utm.transform(E, N)
    pc, pr = np.floor((xu - ITS_GT[0]) / ITS_GT[1]).astype(int), np.floor((ITS_GT[2] - yu) / ITS_GT[1]).astype(int)
    from shapely import contains_xy
    pix = sorted({(r, c) for r, c in zip(pr.ravel(), pc.ravel())
                  if contains_xy(t24, ITS_GT[0] + ITS_GT[1] * (c + 0.5), ITS_GT[2] - ITS_GT[1] * (r + 0.5))})
    assert len(pix) == 96, len(pix)
    index = {p: k for k, p in enumerate(pix)}
    g["itslive_pixel"] = np.array([index.get(p, -1) for p in zip(pr.ravel(), pc.ravel())], "int16").reshape(NY, NX)
    to_lv95 = Transformer.from_crs(32632, 2056, always_xy=True)
    with open(OUT / "itslive_pixels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pixel", "x_utm", "y_utm", "E", "N", "cells"])
        for k, (r, c) in enumerate(pix):
            x, y = ITS_GT[0] + ITS_GT[1] * (c + 0.5), ITS_GT[2] - ITS_GT[1] * (r + 0.5)
            e, n = to_lv95.transform(x, y)
            w.writerow([k, x, y, f"{e:.1f}", f"{n:.1f}", int((g["itslive_pixel"] == k).sum())])

    # radar 2006: profiles are the runs of points split at gaps over 30 m
    rad = np.genfromtxt(th / "C14-10_basodino_GlaThiDa.txt", comments="#", delimiter="\t", usecols=(3, 4, 5, 7, 8))
    prof = 1 + np.concatenate([[0], np.cumsum(np.hypot(*np.diff(rad[:, :2], axis=0).T) > 30.0)])
    assert len(rad) == 1047 and prof.max() == 6, (len(rad), prof.max())
    col, row = ((rad[:, 0] - E0) // DX).astype(int), ((N1 - rad[:, 1]) // DX).astype(int)
    s21 = g["usurf_2021"][row, col]
    with open(OUT / "radar_2006.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["point", "profile", "E", "N", "row", "col", "z_dem_2012", "bedrock", "thickness", "usurf_2021",
                    "above_2021"])
        for k, (p, (e, n, z, b, h), r, c, s) in enumerate(zip(prof, rad, row, col, s21)):
            w.writerow([k, p, e, n, r, c, z, b, h, f"{s:.2f}", int(b > s)])

    for name, z in g.items():
        assert z.shape == (NY, NX), name
        write(name, z)

    a = DX * DX / 1e6
    for name in ("mask_1994", "mask_2021", *OUTLINES, "s2_2026", "s2_2026_body", "c14_11_1850", "posbal"):
        rows.append((f"area {name} km2", f"{g[name].sum() * a:.4f}", ""))
    for u in (-1.0, 0.0, 1.0):
        v = ((g["usurf_2021"] - bed(u, g)) * on).sum() * a / 1e3
        rows.append((f"V_2021 at u = {u:+.0f}, km3", f"{v:.5f}", ""))
    gr = ~np.isnan(g["bed_grab"])
    rows += [("2021 mask cells without a Grab bed (bed = surface - 1 m)", int((on & ~gr).sum()), ""),
             ("2021 mask cells with the Grab bed less than 1 m below the surface", int((on & gr & (g["bed_grab"] > g["usurf_2021"] - 1)).sum()), ""),
             ("2021 mask cells, total", int(on.sum()), ""),
             ("posbal cells outside sgi1850", int(((g["posbal"] == 1) & (g["sgi1850"] == 0)).sum()), ""),
             ("radar points, of them bedrock above usurf_2021", len(rad), int((rad[:, 3] > s21).sum())),
             ("ITS_LIVE pixels inside the 2024 outline, lattice cells in them", len(pix), int((g["itslive_pixel"] >= 0).sum()))]
    RES.mkdir(exist_ok=True)
    with open(RES / "inputs_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value", "second"])
        w.writerows(rows)
    for r in rows:
        print(*r, sep="\t")


if __name__ == "__main__":
    main()
