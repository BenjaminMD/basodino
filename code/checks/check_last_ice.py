"""Independent check of step 5 (last_ice.py), written without importing it or outline_2026.py.

Re-derives, from the archive's files alone: (a) the scene choice, with my own clearness measures over a fixed zone;
(b) the reflectance offsets, from the catalogue and from the raw products of two sources for the same acquisitions;
(c) the ice of every voter and the yearly vote with my own NDSI / Otsu / cast-shadow code (pixel centres, no joins
of narrow strips), once with last_ice.py's reference of each year and once with the 2024 outline for every year as
in outline_2026.py; (d) the 2026 ice against data/outline_2026; (e) 24 cells where last_ice.tif changes between
neighbouring years (3 per year 2018-2025, seed 0), with the evidence per voter.

Ice rule (outline_2026.py steps 1-5 as its docstring states them): NDSI = (B03 - B11) / (B03 + B11), B11 bilinear
from 20 m; region = reference + 50 m; cast shadow from the 2 m DEM averaged to 10 m, terrain within 2 km towards
the sun above the ray; Otsu (400 bins on -1..1) over the sunlit region and over the shadowed pixels inside the
reference (if >= 100); in shadow ice only inside the reference. Ice of a year: more than half of its voters.

Inputs (report/ on s02, data -> ../derived): data/satellite_archive/sentinel2/{catalog.csv, inventory.csv, scenes/},
  data/dem_2m.tif, data/vectors.gpkg (sgi_2016, glacier_2024), data/course/mask_basodino_2021.asc,
  data/outline_2026/outline_2026-09-27.gpkg, code/outline_2026.csv, code/last_ice.tif, code/last_ice_scenes.csv;
  Earth Search and Planetary Computer STAC (anonymous) for (b) and for the scenes after the archive.
Outputs: code/checks/check_last_ice.csv (one row per voter and per year), code/checks/check_last_ice_cells.csv
  (the inspected cells); stdout: the five checks, kept as code/checks/check_last_ice.log. About 4 min.
usage (s02, env agl): cd /data/befahl/projects/AGL/basodino/report &&
  nrun python code/checks/check_last_ice.py > code/checks/check_last_ice.log; then copy the three outputs back
"""
import csv
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer
import rasterio
import requests
import shapely
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
S2 = ROOT / "data/satellite_archive/sentinel2"
with rasterio.open(ROOT / "code/last_ice.tif") as src:
    LAST, T, (H, W) = src.read(1), src.transform, src.shape                 # the grid of the product
E0, N1 = T.c, T.f
Xc, Yc = E0 + 5 + 10 * np.arange(W), N1 - 5 - 10 * np.arange(H)


def cells(geom):
    return rasterize([geom], (H, W), transform=T).astype(bool)


def window(path, band, res):
    with rasterio.open(path) as src:
        return src.read(band, window=from_bounds(E0, N1 - H * 10, E0 + W * 10, N1, src.transform).round_offsets()
                        .round_lengths()).astype(np.float32)


# --- outlines
g16, g24 = (gpd.read_file(ROOT / "data/vectors.gpkg", layer=k) for k in ("sgi_2016", "glacier_2024"))
G16, G24 = g16[g16["sgi-id"] == "C14-10"].union_all(), g24[g24.sgi == "C14-10"].union_all()
OTHER16, OTHER24 = g16[g16["sgi-id"] != "C14-10"].union_all(), g24[g24.sgi != "C14-10"].union_all()
G26 = gpd.read_file(ROOT / "data/outline_2026/outline_2026-09-27.gpkg").union_all()
with rasterio.open(ROOT / "data/course/mask_basodino_2021.asc") as src:
    m = src.read(1)
    G21 = shapely.union_all([shapely.geometry.shape(g) for g, v in shapes(m, transform=src.transform) if v == 1])
ERA = {y: (G16, OTHER16) if y < 2021 else (G21, OTHER16) if y < 2024 else (G24, OTHER24) for y in range(2018, 2027)}
CORE = cells(G26.buffer(-20))
RING = cells(shapely.union(G16, G24).buffer(300)) & ~cells(shapely.union_all([G16, G24, OTHER16, OTHER24]).buffer(50))

# --- terrain: 10 m DEM on the grid plus 2 km
M = 200
with rasterio.open(ROOT / "data/dem_2m.tif") as src:
    z = src.read(1, window=from_bounds(E0 - 2000, N1 - H * 10 - 2000, E0 + W * 10 + 2000, N1 + 2000, src.transform)
                 .round_offsets().round_lengths())
DEM = z.reshape(z.shape[0] // 5, 5, z.shape[1] // 5, 5).mean(axis=(1, 3))


def shadow(elev, azim):
    """Cast shadow: some terrain within 2 km towards the sun higher than the ray (nearest 10 m cell, 10 m steps)."""
    r, c = np.mgrid[M:M + H, M:M + W]
    out, z0 = np.zeros((H, W), bool), DEM[M:M + H, M:M + W]
    for d in range(10, 2001, 10):
        rr = np.rint(r - d * np.cos(np.radians(azim)) / 10).astype(int)
        cc = np.rint(c + d * np.sin(np.radians(azim)) / 10).astype(int)
        out |= DEM[rr, cc] > z0 + d * np.tan(np.radians(elev))
    return out


def otsu(v):
    h, e = np.histogram(v, 400, (-1, 1))
    mid = (e[1:] + e[:-1]) / 2
    w0, s0 = np.cumsum(h), np.cumsum(h * mid)
    w1, s1 = w0[-1] - w0, s0[-1] - s0
    return e[1 + np.argmax(w0 * w1 * (s0 / np.maximum(w0, 1) - s1 / np.maximum(w1, 1)) ** 2)]


CAT = {r["tag"]: r for r in csv.DictReader(open(S2 / "catalog.csv"))}


def bands(tag):
    c = CAT[tag]
    b02, b03, b04, b08 = (np.where(v == 0, np.nan, v / 1e4) for v in window(S2 / c["file_10m"], None, 10))
    with rasterio.open(S2 / c["file_20m"]) as src:
        b11 = np.full((H, W), np.nan, np.float32)
        v = src.read(1).astype(np.float32)
        reproject(np.where(v == 0, np.nan, v / 1e4), b11, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=T, dst_crs=src.crs, src_nodata=np.nan, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    scl = np.repeat(np.repeat(window(S2 / c["file_scl"], 1, 20), 2, 0), 2, 1) if c["file_scl"] else None
    return dict(B02=b02, B03=b03, B08=b08, B11=b11, ndsi=(b03 - b11) / (b03 + b11), scl=scl,
                shade=shadow(float(c["sun_elevation"]), float(c["sun_azimuth"])))


def ice(b, ref, others):
    region, inside = cells(ref.buffer(50)), cells(ref)
    t_sun = otsu(b["ndsi"][region & ~b["shade"] & np.isfinite(b["ndsi"])])
    sh = inside & b["shade"]
    t_sh = otsu(b["ndsi"][sh]) if sh.sum() >= 100 else np.nan
    t = np.where(b["shade"] & np.isfinite(t_sh), t_sh, t_sun)
    z = b["ndsi"] - t
    ok = region & ~(b["shade"] & ~inside & np.isfinite(t_sh)) & ~(cells(others) & ~inside)
    return (z > 0) & ok, z, t_sun


# --- (a) the scenes: my measures over the 2024 outline + 50 m, the ring and the core
SC = {r["scene"]: r for r in csv.DictReader(open(ROOT / "code/last_ice_scenes.csv")) if r["scene"] != "year"}
VOTERS = {y: [t for t, r in SC.items() if r["decision"].startswith("voter") and t[:4] == str(y)]
          for y in range(2018, 2027)}
ZONE = cells(G24.buffer(50))
print("(a) scenes 1 Aug - 31 Oct 2018-2026: cloud+shadow (SCL 3,8,9,10) in 2024+50 m, snow fraction of the clear "
      "ring, core B03; * voter")
meas = {}
for tag in sorted(t for t in CAT if 2018 <= int(t[:4]) <= 2026 and "08-01" <= t[5:10] <= "10-31"):
    b = bands(tag)
    cl = np.isin(b["scl"], (3, 8, 9, 10))
    clear = RING & ~cl & (b["scl"] > 1)
    snow = (b["ndsi"] > 0.4) & (b["B03"] > 0.1) & (b["B08"] > 0.11)
    meas[tag] = dict(cloud=cl[ZONE].mean(), nodata=int((~np.isfinite(b["ndsi"]) & ZONE).sum()),
                     snow=snow[clear].mean() if clear.sum() > 0.5 * RING.sum() else np.nan,
                     green=np.nanmedian(b["B03"][CORE]), dark=np.nanpercentile(b["B03"], 1))
for y in range(2018, 2027):
    yr = {t: v for t, v in meas.items() if t[:4] == str(y)}
    near = {t: v for t, v in yr.items() if v["cloud"] <= 0.05 and v["nodata"] <= 5}
    smin = min(v["snow"] for v in near.values() if np.isfinite(v["snow"]))
    print(f"  {y}: {len(yr)} scenes, {len(near)} with cloud <= 5 %; ring snow minimum {smin:.3f}")
    for t, v in near.items():
        flag = "*" if t in VOTERS[y] else " "
        print(f"    {flag} {t} cloud {v['cloud']:.3f} nodata {v['nodata']} ring snow {v['snow']:.3f} core B03 "
              f"{v['green']:.3f}  last_ice: {SC[t]['decision'][:45]}")

# --- (b) offsets
rule = [(t, c["processing_baseline"], c["source_offset"]) for t, c in CAT.items() if t in meas
        and float(c["source_offset"]) != (-0.1 if c["processing_baseline"] >= "04.00" else 0.0)]
print(f"(b) catalogue: {len(meas)} scenes in the windows, offset against the baseline rule wrong for {len(rule)} {rule}")
miss = [r for r in SC.values() if float(r["offset"]) != (-0.1 if r["processing_baseline"] >= "04.00" else 0.0)]
print(f"    last_ice_scenes.csv: {len(SC)} rows, offset against the rule wrong for {len(miss)}; voters' first "
      f"percentile B03 {min(meas[t]['dark'] for v in VOTERS.values() for t in v if t in meas):.4f}-"
      f"{max(meas[t]['dark'] for v in VOTERS.values() for t in v if t in meas):.4f}")


def stac(api, coll, date, end=None):
    body = dict(collections=[coll], limit=50, datetime=f"{date}T00:00:00Z/{end or date}T23:59:59Z",
                bbox=list(transform_bounds("EPSG:2056", "EPSG:4326", E0, N1 - H * 10, E0 + W * 10, N1)))
    return [i for i in requests.post(f"{api}/search", json=body, timeout=120).json()["features"]
            if "32TMS" in i["id"] or i["properties"].get("s2:mgrs_tile") == "32TMS"]


ES, PC = "https://earth-search.aws.element84.com/v1", "https://planetarycomputer.microsoft.com/api/stac/v1"
GLAC = cells(G24)


def raw_b03(href):
    """Raw DN of B03 on the product grid around the glacier (no resampling) and its median over the glacier."""
    with rasterio.open(planetary_computer.sign(href) if "windows.net" in href else href) as src, \
            WarpedVRT(src, crs="EPSG:2056", transform=T, width=W, height=H, resampling=Resampling.nearest) as v:
        dn = v.read(1).astype(float)
    return np.median(dn[GLAC & (dn > 0)])


for date in ("2022-10-23", "2022-08-24", "2024-09-10"):
    found = []
    for api, coll, key in ((ES, "sentinel-2-pre-c1-l2a", "green"), (ES, "sentinel-2-c1-l2a", "green"),
                           (PC, "sentinel-2-l2a", "B03")):
        for i in stac(api, coll, date):
            meta = i["assets"][key].get("raster:bands", [{}])[0]
            found.append((coll, i["properties"]["s2:processing_baseline"], meta.get("offset"),
                          i["properties"].get("earthsearch:boa_offset_applied"), raw_b03(i["assets"][key]["href"])))
    tag = next((t for t in CAT if t.startswith(date) and t in meas), None)
    arch = np.nanmedian(np.where(GLAC, bands(tag)["B03"], np.nan)) if tag else np.nan
    print(f"    {date}: archive {tag} ({CAT[tag]['source'] if tag else ''}) median glacier B03 {arch:.4f}; raw DN medians: "
          + "; ".join(f"{c} {b} meta offset {o} applied {a}: DN {d:.0f} -> {d / 1e4 - (0.1 if b >= '04.00' else 0):.4f}"
                      for c, b, o, a, d in found))

for i in stac(ES, "sentinel-2-c1-l2a", "2026-09-30", "2026-10-31"):                # after the archive
    with rasterio.open(i["assets"]["scl"]["href"]) as src, \
            WarpedVRT(src, crs="EPSG:2056", transform=T, width=W, height=H, resampling=Resampling.nearest) as v:
        scl = v.read(1)
    print(f"(a) after the archive: {i['id']} cloud+shadow in 2024+50 m {np.isin(scl, (3, 8, 9, 10))[ZONE].mean():.3f}")

# --- (c) yearly ice with my code, the two reference choices
rows, ICE, Z, B = [], {}, {}, {}
for y in range(2018, 2027):
    votes, votes24 = 0, 0
    for t in VOTERS[y]:
        if t not in CAT:
            continue
        B[t] = b = bands(t)
        a, Z[t], ts = ice(b, *ERA[y])
        a24, _, _ = ice(b, G24, OTHER24)
        votes, votes24 = votes + a, votes24 + a24
        rows.append([y, t, round(ts, 3), SC[t]["threshold_sunlit"], int(a.sum()) * 100, SC[t]["ice_area_m2"],
                     int(a24.sum()) * 100, round(b["shade"][cells(ERA[y][0].buffer(50))].mean(), 3),
                     SC[t]["shadow_region"]])
    n = len(VOTERS[y])
    ICE[y], ice24 = 2 * votes > n, 2 * votes24 > n
    li = int(next(r for r in csv.DictReader(open(ROOT / "code/last_ice_scenes.csv"))
                  if r["year"] == str(y) and r["scene"] == "year")["ice_area_m2"])
    rows.append([y, "year", n, "", int(ICE[y].sum()) * 100, li, int(ice24.sum()) * 100, "", ""])
GATE = {2021: float(next(r for r in csv.DictReader(open(ROOT / "code/outline_2026.csv"))
                         if r["scene"] == "2021-09-01_S2B_R108")["area_m2"]), 2024: G24.area, 2026: G26.area}
print("(c) per voter: my sunlit threshold / last_ice's, my ice / last_ice's / mine with the 2024 reference (m2), "
      "cast-shadow share of the region mine / last_ice's")
for r in rows:
    print("   ", *r)
for y, ref in GATE.items():
    a = ICE[y].sum() * 100
    print(f"    gate {y}: mine {a / 1e6:.4f} km2, reference {ref / 1e6:.4f}, difference {(a - ref) / 1e6:+.4f} "
          f"({'holds' if abs(a - ref) <= 0.09e6 else 'FAILS'})")
last_mine = np.zeros((H, W), int)
for y in range(2018, 2027):
    last_mine[ICE[y]] = y
an = LAST != -9999
print(f"    my last-ice year equals last_ice.tif on {np.mean(last_mine[an] == LAST[an]):.4f} of the analysed cells; "
      f"differ by more than one year on {int((np.abs(last_mine[an] - LAST[an]) > 1).sum())}")

for y in range(2019, 2027):                                       # ice after a year without, and ice for the first time
    prev = np.any([ICE[x] for x in range(2018, y)], axis=0)
    back, new = ICE[y] & ~ICE[y - 1] & prev, ICE[y] & ~prev
    out_prev = ~cells(ERA[y - 1][0].buffer(50))
    g = np.nanmedian(B[VOTERS[y][-1]]["B03"][back]) if back.any() else np.nan
    sh = np.mean([B[t]["shade"][back].mean() for t in VOTERS[y]]) if back.any() else np.nan
    print(f"    {y}: ice again after a year without {int(back.sum())} cells ({int((back & out_prev).sum())} outside "
          f"the region of {y - 1}), median B03 there {g:.2f}, in cast shadow {sh:.2f}; ice never before "
          f"{int(new.sum())} cells "
          f"({int((new & out_prev).sum())} outside the region of {y - 1})")
diff = an & (np.abs(last_mine - LAST) > 1)
print(f"    cells differing by more than a year, by last_ice.tif year: "
      f"{dict(zip(*np.unique(LAST[diff], return_counts=True)))}, mine: {dict(zip(*np.unique(last_mine[diff], return_counts=True)))}")

# --- (d) 2026 against our outline
o = cells(G26)
mine27 = ice(B["2026-09-27_S2C_R065"], G24, OTHER24)[0]
li26 = LAST == 2026
for name, a in (("my 27 September", mine27), ("last_ice.tif 2026 (5 voters)", li26)):
    print(f"(d) {name}: {a.sum() * 100:.0f} m2 against the outline's cells {o.sum() * 100:.0f} m2 (polygon "
          f"{G26.area:.0f}); intersection over union {(a & o).sum() / (a | o).sum():.4f}, "
          f"{int((a & ~o).sum())} cells only here, {int((o & ~a).sum())} only in the outline")

# --- (e) cells where last_ice changes between neighbouring years
rng, out = np.random.default_rng(0), []
for y in range(2018, 2026):
    nb = np.zeros((H, W), bool)
    for s in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb |= np.roll(LAST == y + 1, s, (0, 1))
    rr, cc = np.nonzero((LAST == y) & nb)
    for k in rng.choice(len(rr), 3, replace=False):
        r, c = rr[k], cc[k]
        later = np.isin(LAST, range(y + 1, 2027))
        dist = distance_transform_edt(~later)[r, c] * 10
        ev = []
        for x in (y, y + 1):
            for t in VOTERS[x]:
                b = B[t]
                ev.append(f"{t[:10]} z {Z[t][r, c]:+.2f} B03 {b['B03'][r, c]:.2f} "
                          f"{'shadow' if b['shade'][r, c] else 'sun'} SCL {int(b['scl'][r, c])}")
        series = "".join("I" if ICE[x][r, c] else "." for x in range(2018, 2027))
        out.append([y, int(Xc[c]), int(Yc[r]), series, round(dist), " | ".join(ev)])
        print(f"(e) last {y} at E {Xc[c]:.0f} N {Yc[r]:.0f}: ice 2018-2026 {series}, {dist:.0f} m from ice after "
              f"{y}; " + " | ".join(ev))

with open(ROOT / "code/checks/check_last_ice.csv", "w", newline="") as f:
    csv.writer(f).writerows([["year", "scene", "threshold_sunlit_mine", "threshold_sunlit_last_ice", "ice_m2_mine",
                              "ice_m2_last_ice", "ice_m2_mine_2024_reference", "shadow_region_mine",
                              "shadow_region_last_ice"]] + rows)
with open(ROOT / "code/checks/check_last_ice_cells.csv", "w", newline="") as f:
    csv.writer(f).writerows([["last_year", "E", "N", "ice_2018_2026", "distance_to_later_ice_m", "evidence"]] + out)
