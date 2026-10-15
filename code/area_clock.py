"""Area clock: melt each cell of the thickness map (Grab et al. 2021, release 2020) from the 2021 surface with the
GLAMOS band balances and compare the ice area left with our outlines of 2021-2026 (mass test, step 3).

Inputs:
  data/thickness/C14-10_{GlacierBed,IceThickness,IceThicknessUncertainty{Plus,Minus},SwissALTI3D_r2019}.tif
                                        the map: bed, its outline (SGI2016), sigma towards thicker / thinner ice,
                                        the 2012 surface
  data/thickness/C14-10_basodino_GlaThiDa.txt   the 2006 radar points (LV95)
  data/course/{dem,mask}_basodino_2021.asc      the 2021 surface and mask, 10 m, the start of the clock
  data/glamos_basodino_massbalance_observation_elevationbins.csv   GLAMOS Ba per 100 m bin, observation periods
  B_2026 below                          GLAMOS annual report 2026, Fig. A.1: -3.503 m w.e., 6/9/2025 - 29/9/2026
                                        (glacier-wide only, provisional)
  code/volume_clock.csv                 k (2012 to 2024-08-23) and its sigma, step 1
  code/thickness.csv                    dz_stable_mean, the offset of the 2021 surface on ice-free ground
  code/outline_2026.csv                 observed areas: our Sentinel-2 outlines, swissTLM3D 2024
  data/outline_2026/outline_2026-09-27.gpkg, data/vectors.gpkg layer glacier_2024   the 2026 and 2024 polygons
Method:
  1. Start thickness H(u) = max(z_2021 - bed + u sigma w, 0) on the 2021 mask inside the map's outline, no ice
     elsewhere (0.0152 km2 of the mask has no bed). sigma is the plus grid for u > 0, the minus grid for u < 0;
     w = 1, or the edge taper of igm/DESIGN.md decision 8: w = min(1, d / 50 m), d = distance of the cell centre
     inside the 2021 mask (Euclidean distance to the nearest centre outside, minus half a cell).
  2. Melt M at a cell (m ice) = -k_cell x sum of Ba over the observation periods since 10 September 2021, in the
     100 m bin of the cell's 2021 elevation (nearest bin outside their range), x 1000/900, minus 0.3 m for 10-30
     September 2021 (the upper bound of step 2, on every cell: in favour of the map). Periods end 2024-09-03,
     2025-09-06 and 2026-09-29; the 2025/26 profile is the 2024/25 one shifted to the glacier-wide -3.503 m w.e.
     on the 2024/25 bin areas (Fig. A.1 shows no gradient; the scaled and flat profiles are in the check rows).
     Static 2021 elevations: the bins do not follow the lowering surface.
  3. k_cell: k = 1, step 1's k and k -+ its sigma, 0.77, or k_b per 100 m band of the 2021 surface:
     k_b = (850/1000 x mean dh) / mean B on the cells that were ice in 2012 (map outline) and in 2021 (mask), with
     dh = z_2021 - z_2012 - dz_stable_mean and B the sum of the nine bin balances 2012-09-07 to 2021-09-10. k_b
     carries the emergence velocity of the band, (dh - B x 1000/900) / 9 a.
  4. Melt spread s: the melt at a cell is M (1 + s e), e standard normal; the cell keeps ice with probability
     Phi((H - M) / (s M)) (H > 0 required); the predicted area is the sum of the probabilities x 100 m2. s = 0 is
     the plain test H > M.
  5. Dates: 2021 = zero melt (Sentinel-2 of 1 September 2021, a month before the 2021 surface); official 2024 and
     Sentinel-2 2024-09-10 against the end of the 2023/24 period (2024-09-03); 2025-09-07 against 2025-09-06;
     2026-09-27 against 2026-09-29. The observed areas exclude holes (generous to the map); sigma = pixel x
     perimeter (10 m x perimeter for swissTLM3D). u_match: the u in [-1, 2] (grid 0.01) at which the predicted
     area equals the observed one, low and high at observed -+ sigma; 'beyond 2' when not reached. k_match: the uniform
     k at u = 0, no taper, that does the same.
  6. Where the map is contradicted, for k = step 1's k, s = 0, u = 0, no taper, on the 2024 (glacier_2024) and the
     2026 polygons (cell centre inside): per 100 m band, distance to the map outline's edge and distance to the
     nearest radar point: observed ice area, area observed as ice but predicted gone, area predicted as ice but
     observed gone, median u needed by the contradicted cells (u_cell = (M - z_2021 + bed) / sigma_plus; inf where
     the map has no bed) and the contradicted area with u_cell > 1 (beyond the map's +1 sigma band).
Outputs:
  code/area_clock.csv   quantity, date, k, s, taper, u, group, value, low, high, unit, definition
usage:  uv run --with rasterio --with numpy --with scipy --with shapely python code/area_clock.py
"""
import csv
import sqlite3

import numpy as np
import rasterio
from rasterio import features
from scipy import ndimage, special
from scipy.spatial import cKDTree
from shapely import wkb

from common import COURSE, RHO_DV, RHO_ICE, RHO_W, ROOT, read_csv, read_esri_ascii, write_csv

B_2026 = ("2025-09-06", "2026-09-29", -3.503)          # m w.e., GLAMOS annual report 2026, Fig. A.1, provisional
SEP_2021 = 0.3                                          # m ice, melt 10-30 September 2021 at most (step 2)
TAPER = 50.0                                            # m, igm/DESIGN.md decision 8
U = np.round(np.arange(-1, 2.0001, 0.01), 2)
U_ROWS = np.arange(-1, 2.0001, 0.25)                    # u written out
SPREADS = (0.0, 0.28, 0.4)
ICE = RHO_W / RHO_ICE                                   # m ice per m w.e.

# --- grids on the course lattice
z21, hd = read_esri_ascii(COURSE / "dem_basodino_2021.asc")
g21 = read_esri_ascii(COURSE / "mask_basodino_2021.asc")[0] == 1
CS, X0 = hd["cellsize"], hd["xllcorner"]
NY, NX = z21.shape
Y0 = hd["yllcorner"] + NY * CS
CELL = CS * CS
TRANSFORM = rasterio.transform.from_origin(X0, Y0, CS, CS)


def on_course_grid(name):
    """A grid of the thickness data pasted onto the course grid; cell edges coincide (asserted), no resampling."""
    with rasterio.open(ROOT / f"data/thickness/C14-10_{name}.tif") as src:
        a = src.read(1, masked=True).astype(float).filled(np.nan)
        c0, r0 = (src.transform.c - X0) / CS, (Y0 - src.transform.f) / CS
        assert src.res == (CS, CS) and c0 == round(c0) and r0 == round(r0), "cell edges off the course lattice"
    c0, r0 = round(c0), round(r0)
    rr, cc = slice(max(r0, 0), min(r0 + a.shape[0], NY)), slice(max(c0, 0), min(c0 + a.shape[1], NX))
    out = np.full((NY, NX), np.nan)
    out[rr, cc] = a[rr.start - r0:rr.stop - r0, cc.start - c0:cc.stop - c0]
    return out


def gpkg_polygons(path, table, where=""):
    """GeoPackage geometry blobs -> shapely: 8-byte header, envelope of 0-64 bytes (flag bits 1-3), then WKB."""
    con = sqlite3.connect(path)
    blobs = [b for (b,) in con.execute(f"SELECT geom FROM {table} {where}")]
    con.close()
    return [wkb.loads(bytes(b[8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(b[3] >> 1) & 7]:])) for b in blobs]


bed, h12 = on_course_grid("GlacierBed"), on_course_grid("IceThickness")
plus, minus = on_course_grid("IceThicknessUncertaintyPlus"), -on_course_grid("IceThicknessUncertaintyMinus")
s12 = on_course_grid("SwissALTI3D_r2019")
outline = np.isfinite(h12)                              # the map's outline, SGI2016
cells = g21 & outline                                   # cells that can hold ice: 2021 mask with a bed
MAP = {r["quantity"]: float(r["value"]) for r in read_csv("thickness.csv")}
H21 = (z21 - bed)[cells]                                # may be negative: bed above the 2021 surface
SIG_P, SIG_M = plus[cells], minus[cells]
d_in = ndimage.distance_transform_edt(g21) * CS - CS / 2
W = {"none": np.ones(cells.sum()), "50 m": np.clip(d_in[cells] / TAPER, 0, 1)}
Z = z21[cells]
BAND = np.floor(Z / 100) * 100

# --- GLAMOS bins
BINS = {}                                               # (start, end) -> array of h_min, h_max, Ba (m w.e.), area
with open(ROOT / "data/glamos_basodino_massbalance_observation_elevationbins.csv", encoding="utf-8") as f:
    for r in csv.reader(f):
        if len(r) > 11 and r[1] == "C14-10":
            BINS.setdefault((r[2], r[4]), []).append((float(r[9]), float(r[10]), float(r[7]) / 1000, float(r[8])))
BINS = {p: np.array(sorted(v)) for p, v in BINS.items()}


def bin_balance(b, z):
    """Ba of the bin holding each elevation; the nearest bin outside the range of the bins."""
    i = np.clip(np.searchsorted(b[:, 1], z, side="right"), 0, len(b) - 1)
    return b[i, 2]


def periods(start, end):
    ps = sorted(p for p in BINS if p[0] >= start and p[1] <= end)
    assert all(a[1] == b[0] for a, b in zip(ps, ps[1:])) and ps[0][0] == start and ps[-1][1] == end, "not chained"
    return ps


b25 = BINS["2024-09-03", "2025-09-06"]
shift = B_2026[2] - np.sum(b25[:, 2] * b25[:, 3]) / b25[:, 3].sum()
PROFILES_2026 = {"shifted": b25 + [0, 0, shift, 0], "flat": b25 * [1, 1, 0, 1] + [0, 0, B_2026[2], 0],
                 "scaled": b25 * [1, 1, B_2026[2] / (np.sum(b25[:, 2] * b25[:, 3]) / b25[:, 3].sum()), 1]}
ENDS = ("2024-09-03", "2025-09-06", B_2026[1])


def melt_raw(profile="shifted"):
    """-sum of Ba (m ice) since 10 September 2021 to each end date, per cell; k not yet applied."""
    out, total = {"2021-09-30": np.zeros(Z.size)}, np.zeros(Z.size)
    for p in periods("2021-09-10", ENDS[1]):
        total = total - bin_balance(BINS[p], Z) * ICE
        out[p[1]] = total.copy()
    out[ENDS[2]] = total - bin_balance(PROFILES_2026[profile], Z) * ICE
    return {d: out[d] for d in ("2021-09-30", *ENDS)}


RAW = melt_raw()

# --- k_b, 2012-2021
P12 = periods("2012-09-07", "2021-09-10")
assert len(P12) == 9
assert np.isfinite(s12[cells]).all(), "2012 surface missing"   # cells: ice in 2012 (map outline) and 2021 (mask)
DH = (z21 - s12)[cells] - MAP["dz_stable_mean"]
B12 = sum(bin_balance(BINS[p], Z) for p in P12)
rows = []
KB = {}
for band in np.unique(BAND):
    m = BAND == band
    kb = RHO_DV / RHO_W * DH[m].mean() / B12[m].mean()
    KB[band] = kb
    group = f"{band:.0f}-{band + 100:.0f} m"
    rows += [("dh_2012_2021", "2021-09-30", "", "", "", "", group, DH[m].mean(), "", "", "m",
              f"mean z_2021 - z_2012 - {MAP['dz_stable_mean']:.3f} m on {m.sum() * CELL / 1e6:.4f} km2 (map outline "
              "and 2021 mask)"),
             ("balance_2012_2021", "2021-09-10", "", "", "", "", group, B12[m].mean(), "", "", "m w.e.",
              "mean over the same cells of the sum of the 9 GLAMOS bin balances 2012-09-07 to 2021-09-10"),
             ("k_b", "", "", "", "", "", group, kb, "", "", "", "850/1000 dh_2012_2021 / balance_2012_2021"),
             ("emergence", "", "", "", "", "", group, (DH[m].mean() - B12[m].mean() * ICE) / 9, "", "", "m a-1",
              "(dh_2012_2021 - balance_2012_2021 x 1000/900) / 9 a: thinning minus balance, ice")]
K_B = np.array([KB[b] for b in BAND])
k_dem = RHO_DV / RHO_W * DH.sum() / B12.sum()
VC = [r for r in read_csv("volume_clock.csv") if r["quantity"] == "k"]
K1 = next(r for r in VC if r["start"] == "2012" and r["end"] == "2024-08-23")
K1, SK1 = float(K1["value"]), float(K1["sigma"])
K12 = float(next(r for r in VC if r["start"] == "2012" and r["end"] == "2021")["value"])
rows += [("k_dem_glacier_wide", "", "", "", "", "", "all bands", k_dem, "", "", "",
          f"850/1000 sum dh / sum B over the same cells; check against step 1's k 2012-2021 = {K12:.4f} (GLAMOS "
          "survey on GLAMOS mean areas)"),
         ("k_b_applied_glacier_wide", "", "", "", "", "", "2021 mask", np.sum(K_B * RAW[ENDS[2]]) / RAW[ENDS[2]].sum(),
          "", "", "", "sum of k_b x melt / sum of melt, 2021 to 2026, on the cells that can hold ice")]
KS = {"1": 1.0, f"{K1:.3f}": K1, f"{K1 - SK1:.3f}": K1 - SK1, f"{K1 + SK1:.3f}": K1 + SK1, "0.77": 0.77, "k_b": K_B}


# --- areas
def start_thickness(u, taper):
    """H(u) on the cells, one row per u."""
    u = np.atleast_1d(u)[:, None]
    return np.maximum(H21 + u * np.where(u > 0, SIG_P, SIG_M) * W[taper], 0)


def area(h, melt, s):
    """Predicted ice area (km2) for thickness rows h and melt per cell."""
    if s == 0:
        p = h > melt
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(melt > 0, special.ndtr((h - melt) / (s * melt)), 1.0)
    return np.sum(np.where(h > 0, p, 0), axis=-1) * CELL / 1e6


def melt(k, end, raw=RAW):
    """Melt (m ice) per cell at an end date: k x raw, less the melt of 10-30 September 2021, at least 0."""
    return raw[end] if end == "2021-09-30" else np.maximum(k * raw[end] - SEP_2021, 0)


def crossing(x, y, target):
    """First x where the non-decreasing y reaches target, linear between grid points; 'beyond' the grid's last x
    when y never reaches it, 'before' its first x when y starts above it."""
    if y[-1] < target:
        return f"beyond {x[-1]:g}"
    if y[0] > target:
        return f"before {x[0]:g}"
    i = int(np.argmax(y >= target))
    return float(x[0]) if i == 0 else float(np.interp(target, y[i - 1:i + 1], x[i - 1:i + 1]))


OUT = {r["scene"]: r for r in read_csv("outline_2026.csv")}
TARGETS = [("2021-09-01", "2021-09-30", "Sentinel-2", OUT["2021-09-01_S2B_R108"]),
           ("2024-08", ENDS[0], "swissTLM3D 2024, official", OUT["swissTLM3D 2024"]),
           ("2024-09-10", ENDS[0], "Sentinel-2", OUT["2024-09-10_S2A_R108"]),
           ("2025-09-07", ENDS[1], "Sentinel-2", OUT["2025-09-07_S2B_R065"]),
           ("2026-09-27", ENDS[2], "Sentinel-2", OUT["2026-09-27_S2C_R065"])]
OBSERVED = {}
for day, end, source, r in TARGETS:
    a = float(r["area_m2"]) / 1e6
    sa = float(r["pixel_x_perimeter_m2"] or float(r["perimeter_m"]) * CS) / 1e6
    OBSERVED[day] = (a, sa, end)
    rows.append(("area_observed", day, "", "", "", "", source, a, a - sa, a + sa, "km2",
                 f"total of all polygons, holes excluded; main body {float(r['area_main_body_m2']) / 1e6:.4f}; "
                 f"low/high -+ {'pixel' if r['pixel_x_perimeter_m2'] else '10 m'} x perimeter; predicted at {end}"))

VOLS = {}
for taper in W:
    v = np.sum(start_thickness(U, taper), axis=1) * CELL / 1e9
    rows += [("volume_2021", "2021-09-30", "", "", taper, f"{u:g}", "", v[np.abs(U - u).argmin()], "", "", "km3",
              "sum of H(u) x 100 m2 on the 2021 mask") for u in U_ROWS]
    VOLS[taper] = v
    for klabel, k in KS.items():
        for s in SPREADS:
            for end in ("2021-09-30", *ENDS):
                a = area(start_thickness(U, taper), melt(k, end), s)
                rows += [("area_predicted", end, klabel, s, taper, f"{u:g}", "", a[np.abs(U - u).argmin()], "", "",
                          "km2", "predicted ice area") for u in U_ROWS]
                for day, (obs, sa, e) in OBSERVED.items():
                    if e != end:
                        continue
                    um = [crossing(U, a, t) for t in (obs, obs - sa, obs + sa)]
                    rows.append(("u_match", day, klabel, s, taper, "", "", *um, "",
                                 "u at which the predicted area equals the observed one; low/high: observed -+ sigma"))
                    rows.append(("volume_2021_matched", day, klabel, s, taper, "", "",
                                 *[np.interp(x, U, v) if isinstance(x, float) else "" for x in um], "km3",
                                 "volume_2021 at u_match, low, high"))

KGRID = np.round(np.arange(0, 1.5001, 0.005), 3)
for s in SPREADS:
    for day, (obs, sa, end) in OBSERVED.items():
        if end == "2021-09-30":
            continue
        a = np.array([area(start_thickness(0.0, "none"), melt(k, end), s)[0] for k in KGRID])[::-1]
        km = [crossing(KGRID[::-1], a, t) for t in (obs, obs - sa, obs + sa)]
        rows.append(("k_match", day, "", s, "none", "0", "", *km, "",
                     "uniform k at which the map (u = 0) gives the observed area; low/high at observed -+ sigma"))

for profile in ("flat", "scaled"):
    raw = melt_raw(profile)
    for klabel in (f"{K1:.3f}", "k_b"):
        a = area(start_thickness(0.0, "none"), melt(KS[klabel], ENDS[2], raw), 0.0)[0]
        rows.append(("area_predicted_profile_check", ENDS[2], klabel, 0.0, "none", "0", f"2025/26 {profile}", a, "",
                     "", "km2", f"as area_predicted with the {profile} 2025/26 profile instead of the shifted one"))

# --- where the map is contradicted: k = step 1, s = 0, u = 0, no taper
radar = np.loadtxt(ROOT / "data/thickness/C14-10_basodino_GlaThiDa.txt", comments="#", delimiter="\t", usecols=(3, 4))
ii, jj = np.indices((NY, NX))
centres = np.c_[X0 + (jj.ravel() + 0.5) * CS, Y0 - (ii.ravel() + 0.5) * CS]
d_radar = cKDTree(radar).query(centres)[0].reshape(NY, NX)
d_edge = ndimage.distance_transform_edt(outline) * CS - CS / 2
POLY = {ENDS[0]: gpkg_polygons(ROOT / "data/vectors.gpkg", "glacier_2024", "WHERE sgi = 'C14-10'"),
        ENDS[2]: gpkg_polygons(ROOT / "data/outline_2026/outline_2026-09-27.gpkg", "outline_2026")}
EDGES = (0, 25, 50, 100, 200, np.inf)


def label(x, edges, unit="m"):
    i = np.searchsorted(edges, x, side="right") - 1
    return f"{edges[i]:g}-{edges[i + 1]:g} {unit}" if np.isfinite(edges[i + 1]) else f"over {edges[i]:g} {unit}"


for end, polys in POLY.items():
    observed = features.rasterize(polys, out_shape=(NY, NX), transform=TRANSFORM) == 1
    pred = np.zeros((NY, NX), bool)
    m = melt(K1, end)
    pred[cells] = (H21 > 0) & (H21 > m)
    u_need = np.full((NY, NX), np.inf)
    u_need[cells] = (m - H21) / SIG_P
    rows.append(("area_observed_raster", end, "", "", "", "", "polygons, cell centre inside",
                 observed.sum() * CELL / 1e6, "", "", "km2", "the polygons on the 10 m lattice"))
    every = observed | pred
    groups = {"all": None, "band": np.where(every, np.floor(z21 / 100) * 100, np.nan), "edge": d_edge,
              "radar": d_radar}
    for kind, field in groups.items():
        if kind == "all":
            names = np.full(NY * NX, "all")
        elif kind == "band":
            names = np.array([f"band {v:.0f}-{v + 100:.0f} m" if np.isfinite(v) else "" for v in field.ravel()])
        elif kind == "edge":
            names = np.array([f"edge {label(v, EDGES)}" if o else "edge: outside the map outline"
                              for v, o in zip(field.ravel(), outline.ravel())])
        else:
            names = np.array([f"radar {label(v, EDGES)}" for v in field.ravel()])
        names = names.reshape(NY, NX)
        for g in sorted(set(names[every])):
            sel = (names == g) & every
            bad = sel & observed & ~pred
            rows += [("observed_ice", end, f"{K1:.3f}", 0.0, "none", "0", g, (sel & observed).sum() * CELL / 1e6, "",
                      "", "km2", "observed ice in the group"),
                     ("contradicted", end, f"{K1:.3f}", 0.0, "none", "0", g, bad.sum() * CELL / 1e6, "", "", "km2",
                      "observed ice, predicted gone"),
                     ("too_thick", end, f"{K1:.3f}", 0.0, "none", "0", g, (sel & pred & ~observed).sum() * CELL / 1e6,
                      "", "", "km2", "predicted ice, observed gone"),
                     ("u_needed_median", end, f"{K1:.3f}", 0.0, "none", "0", g,
                      float(np.median(u_need[bad])) if bad.any() else "", "", "", "",
                      "median over the contradicted cells of (M - z_2021 + bed) / sigma_plus"),
                     ("beyond_plus_sigma", end, f"{K1:.3f}", 0.0, "none", "0", g,
                      (bad & (u_need > 1)).sum() * CELL / 1e6, "", "", "km2",
                      "contradicted cells that keep no ice even at u = 1 (their own +1 sigma)")]

# --- gate
a0 = area(start_thickness(0.0, "none"), RAW["2021-09-30"], 0.0)[0]
v0 = VOLS["none"][np.abs(U).argmin()]
rows.insert(0, ("gate_area_u0_no_melt", "2021-09-30", "", 0.0, "none", "0", "", a0, "", "", "km2",
                "u = 0, zero melt: cells of the 2021 mask with z_2021 - bed > 0 (expected 1.379)"))
rows.insert(1, ("gate_volume_u0", "2021-09-30", "", "", "none", "0", "", v0, "", "", "km3",
                f"expected V_2021 of code/thickness.csv, {MAP['V_2021'] / 1e9:.6f}"))
write_csv("area_clock.csv", ("quantity", "date", "k", "s", "taper", "u", "group", "value", "low", "high", "unit",
                             "definition"), rows)

print(f"gate: area at u = 0, no melt {a0:.4f} km2 (1.379); volume {v0:.6f} km3 "
      f"({MAP['V_2021'] / 1e9:.6f}); k from the DEM 2012-2021 {k_dem:.3f} against step 1 {K12:.3f}")
for r in rows:
    if r[0] in ("u_match", "k_match") and r[3] in (0.0,) or r[0] in ("k_b", "emergence"):
        print(" ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r[:10]))
