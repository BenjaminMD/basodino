"""Volume clock: the ice Ghiacciaio del Basòdino lost since the 2012 surface of the thickness map and since
30 September 2021, from GLAMOS surveys and balances alone, against the volumes the map gives (mass test, step 1).

Inputs:
  data/glamos_basodino_volumechange.csv             GLAMOS geodetic survey pairs (release 2025): dV, areas, sigma
  data/glamos_basodino_massbalance_observation.csv  GLAMOS balances of the observation periods (survey to survey)
  B_2026 below                                      GLAMOS annual report 2026 (doi:10.18752/annualrep_2026),
                                                    Fig. A.1, page 15: Ba = -3.503 m w.e., measured period
                                                    6/9/2025 - 29/9/2026, provisional; copy in data/verification/skeptic
  code/outline_2026.csv                             our Sentinel-2 outlines: area and pixel x perimeter per scene
  code/thickness.csv                                the map's volumes V_2021 (course mask 2021) and V_2012 (outline of
                                                    the Swiss Glacier Inventory 2016, imagery 2018); dz_stable_mean
  data/thickness/C14-10_{IceThickness,SwissALTI3D_r2019}.tif, data/course/{dem,mask}_basodino_2021.asc,
  data/vectors.gpkg layer sgi_2010                  for the 2012 ice outside the outline of the map
Method:
  1. GLAMOS rates: B = rho_dV / rho_w dV / (A_mean n), A_mean = (A_start + A_end) / 2, n = end year - start year,
     whole balance years rather than the time between the survey days (checked on all GLAMOS pairs of the glacier).
     Also with the survey days: 2021 = 30 September, 2012 = 19 August or 8 September (the SWISSIMAGE flights over the
     glacier, data/verification/aerial/strips.csv; GLAMOS gives no day).
  2. k = geodetic over glaciological specific balance of one survey pair: (rho_dV / rho_w dV / A_mean) divided by the
     sum of Ba over the observation periods that start in the first and end in the last survey year (n periods,
     chained survey to survey, not hydrological years); sigma_k = k sigma_B n / |geodetic|, the GLAMOS sigma only.
     2012-2024 is used, 2021-2024 is the check; 2012-2021 is the glacier-wide counterpart of the band factors k_b of
     step 3 (summed over the glacier the emergence terms cancel). k_glamos_areas: the 2012-2024 ratio as volumes with
     the GLAMOS balance areas, the part of the 30 per cent that our own areas remove.
  3. Chain without overlap (asserted), each gap listed and not counted: the GLAMOS survey of 30 September 2021 (or of
     2012) to 23 August 2024 as dV, then dV = k Ba rho_w / rho_ice A for 3 September 2024 - 6 September 2025
     (Ba = -1.520 m w.e.) and 6 September 2025 - 29 September 2026 (B_2026), A = mean of our outlines at its ends.
  4. Sigma terms: geodetic sigma_B n A_mean rho_w / rho_dV (sigma_B per year, as the GLAMOS ReadMe defines it; the
     unit row of the file says m w.e.); k: sigma_k / k of the balance part; area: mean pixel x perimeter of the two
     outlines over A, the same sign in both periods; SIGMA_ANNUAL, the random error of each annual balance.
     sigma: the four terms independent. sigma_correlated: the geodetic and the k term added, as they share survey
     models (k 2012-2024 contains the 2024 survey of the 2021 basis and is the very pair of the 2012 basis, where
     it is exact). k 2012-2021 shares the 2021 survey with the 2021 basis with the opposite sign: there sigma is the
     upper one.
  5. Cumulative loss for k = 1, k 2012-2024, 0.77 (approved conventions) and k 2012-2021, over the map's volume (under
     test, so without sigma): V_2021 for the 2021 basis; V_2012, and V_2012 + V_out for the 2012 basis. V_out, the
     2012 ice outside the 2016 outline, ranges from 0 to the sum of (z_2012 - z_2021 + dz_stable_mean) x 100 m2 on the
     cells inside the outline of the inventory 2010 (imagery 2008-2011) and outside the 2016 one: the lowering there is
     the 2012 ice where it was gone by 2021, its thinning where ice was left.
Outputs:
  code/volume_clock.csv   quantity, k, start, end, value, sigma, sigma_correlated, unit, definition
usage:  uv run --with rasterio --with numpy --with shapely python code/volume_clock.py
"""
import csv
import sqlite3
from datetime import date
from itertools import pairwise

import numpy as np
import rasterio
from rasterio import features
from shapely import wkb

from common import COURSE, RHO_DV, RHO_ICE, RHO_W, ROOT, read_csv, read_esri_ascii, write_csv

B_2026 = ("2025-09-06", "2026-09-29", -3.503)         # m w.e., GLAMOS annual report 2026, Fig. A.1, provisional
SIGMA_ANNUAL = 0.40                                   # m w.e., upper end of 0.15-0.40 (Huss, Dhulst and Bauder 2015)
K_LOW = 0.77                                          # approved conventions: GLAMOS glaciological about 1.3 x geodetic
DAYS = {"20129999": ("2012-08-19", "2012-09-08"), "20219999": ("2021-09-30",), "20240823": ("2024-08-23",)}
OUTLINES = ("2024-09-10", "2025-09-07", "2026-09-27")  # our outlines at the ends of the two balance periods
WE, ICE = RHO_DV / RHO_W, RHO_W / RHO_ICE             # m w.e. per m of geodetic change; m of ice per m w.e.

with open(ROOT / "data/glamos_basodino_volumechange.csv", encoding="utf-8") as f:
    PAIRS = {(r[3], r[4]): r for r in csv.reader(f) if len(r) > 14 and r[0] == "C14-10"}
with open(ROOT / "data/glamos_basodino_massbalance_observation.csv", encoding="utf-8") as f:
    OBS = [(r[2], r[4], float(r[7]) / 1000, float(r[10]) * 1e6)  # start, end, Ba (m w.e.), GLAMOS area (m2)
           for r in csv.reader(f) if len(r) > 10 and r[1] == "C14-10"]
MAP = {r["quantity"]: float(r["value"]) for r in read_csv("thickness.csv")}
AREA = {r["scene"][:10]: (float(r["area_m2"]), float(r["pixel_x_perimeter_m2"]))
        for r in read_csv("outline_2026.csv") if r["scene"][:10] in OUTLINES}
assert len(AREA) == len(OUTLINES), "one outline per date expected"


def day(code):
    return code[:4] if code.endswith("9999") else f"{code[:4]}-{code[4:6]}-{code[6:]}"


def days_between(d0, d1):
    return (date.fromisoformat(d1) - date.fromisoformat(d0)).days


def pair(start, end):
    """A GLAMOS survey pair: dV (m3), A_mean (m2), whole years n, published rate and sigma (m w.e. a-1), dV sigma."""
    r = PAIRS[start, end]
    p = {"start": start, "end": end, "dv": float(r[9]) * 1e9, "a": (float(r[5]) + float(r[7])) / 2 * 1e6,
         "n": int(end[:4]) - int(start[:4]), "b": float(r[11]), "sigma_b": float(r[12]), "name": f"{r[1]} to {r[2]}"}
    p["sigma"] = p["sigma_b"] * p["n"] * p["a"] / WE
    return p


def k_factor(start, end):
    """k of one survey pair, with the observation periods it is computed from."""
    p = pair(start, end)
    obs = [o for o in OBS if int(o[0][:4]) >= int(start[:4]) and int(o[1][:4]) <= int(end[:4])]
    assert len(obs) == p["n"] and all(a[1] == b[0] for a, b in pairwise(obs)), "periods do not chain the pair"
    geod, glac = WE * p["dv"] / p["a"], sum(o[2] for o in obs)
    k = geod / glac
    return {"k": k, "sigma": k * p["sigma_b"] * p["n"] / abs(geod), "geod": geod, "glac": glac, "obs": obs, "p": p,
            "glamos_areas": WE * p["dv"] / sum(o[2] * o[3] for o in obs)}


# --- 1 GLAMOS rates
G12, G21 = pair("20129999", "20240823"), pair("20219999", "20240823")
deviation = max(abs(WE * p["dv"] / (p["a"] * p["n"]) - p["b"]) for p in (pair(*codes) for codes in PAIRS))
rows = [("rate_all_pairs_max_deviation", "", "", "", deviation, "", "", "m w.e. a-1",
         f"largest |published - whole-year rate| over all {len(PAIRS)} GLAMOS pairs of C14-10")]
for p in (G12, G21):
    rows += [("rate_published", "", day(p["start"]), day(p["end"]), p["b"], p["sigma_b"], "", "m w.e. a-1",
              f"GLAMOS {p['name']}, per balance year"),
             ("rate_whole_years", "", day(p["start"]), day(p["end"]), WE * p["dv"] / (p["a"] * p["n"]), "", "",
              "m w.e. a-1", (f"850/1000 dV / (A_mean x {p['n']} a), dV {p['dv'] / 1e9:.6f} km3, "
                             f"A_mean {p['a'] / 1e6:.5f} km2"))]
    for d0 in DAYS[p["start"]]:
        t = days_between(d0, DAYS[p["end"]][0]) / 365.25
        rows.append(("rate_survey_days", "", d0, DAYS[p["end"]][0], WE * p["dv"] / (p["a"] * t), "", "",
                     "m w.e. a-1", f"the same over {t:.3f} a between the survey days"))

# --- 2 k
K = {name: k_factor(*codes) for name, codes in (("2012-2024", ("20129999", "20240823")),
                                                ("2021-2024", ("20219999", "20240823")),
                                                ("2012-2021", ("20129999", "20219999")))}
ROLE = {"2012-2024": "used", "2021-2024": "check", "2012-2021": "glacier-wide counterpart of k_b"}
for name, c in K.items():
    o, p = c["obs"], c["p"]
    rows.append(("k", "", day(p["start"]), day(p["end"]), c["k"], c["sigma"], "", "",
                 (f"{ROLE[name]}: geodetic {c['geod']:.3f} m w.e. over the sum of {len(o)} observation periods "
                  f"{o[0][0]} to {o[-1][1]}, {c['glac']:.3f} m w.e.")))
rows.append(("k_glamos_areas", "", "2012", "2024-08-23", K["2012-2024"]["glamos_areas"], "", "", "",
             "850/1000 dV over the sum of Ba x GLAMOS balance area of the same periods"))

# --- 3 and 4 chain and sigma: balance periods after the last survey, our areas
b25 = [o for o in OBS if o[0] == "2024-09-03"]
assert len(b25) == 1 and b25[0][1] == B_2026[0], "the 2024/25 period does not end where the 2025/26 one starts"
PERIODS = []                                          # start, end, Ba (m w.e.), A (m2), sigma A (m2), source
for (s, e, b, source), (o0, o1) in zip(((*b25[0][:3], "GLAMOS observation period, release 2025"),
                                        (*B_2026, "GLAMOS annual report 2026, Fig. A.1, provisional")),
                                       pairwise(OUTLINES)):
    PERIODS.append((s, e, b, (AREA[o0][0] + AREA[o1][0]) / 2, (AREA[o0][1] + AREA[o1][1]) / 2,
                    f"{source}; A = mean of our outlines {o0} and {o1}"))


def balance(k, sigma_k):
    """dV of each balance period (m3) and its terms of sigma: k, area, random error of the annual balance."""
    out = []
    for s, e, b, a, sa, source in PERIODS:
        dv = k * b * ICE * a
        out.append({"start": s, "end": e, "b": b, "a": a, "dv": dv, "source": source, "k": abs(dv) * sigma_k / k,
                    "area": abs(dv) * sa / a, "rand": k * SIGMA_ANNUAL * ICE * a})
    return out


def chain(geo, k, sigma_k):
    """Links in order with the gaps between them; cumulative loss (m3) and both sigmas at each end."""
    links = [{"start": DAYS[geo["start"]][-1], "end": DAYS[geo["end"]][0]}, *balance(k, sigma_k)]
    assert all(days_between(a["end"], b["start"]) >= 0 for a, b in pairwise(links)), "periods overlap"
    loss, kt, at, rt2 = -geo["dv"], 0.0, 0.0, 0.0
    cumulative = [(links[0]["end"], loss, geo["sigma"], geo["sigma"])]
    for t in links[1:]:
        loss, kt, at, rt2 = loss - t["dv"], kt + t["k"], at + t["area"], rt2 + t["rand"] ** 2
        cumulative.append((t["end"], loss, np.sqrt(geo["sigma"] ** 2 + kt ** 2 + at ** 2 + rt2),
                           np.sqrt((geo["sigma"] + kt) ** 2 + at ** 2 + rt2)))
    return cumulative, [(a["end"], b["start"]) for a, b in pairwise(links) if b["start"] != a["end"]]


def survey_row(g, start, role):
    return ("dV", "", start, day(g["end"]), g["dv"] / 1e9, g["sigma"] / 1e9, "", "km3",
            (f"{role}: GLAMOS {g['name']}; sigma {g['sigma_b']} m w.e. a-1 x {g['n']} a x {g['a'] / 1e6:.5f} km2 "
             "x 1000/850"))


k1, s1 = K["2012-2024"]["k"], K["2012-2024"]["sigma"]
_, gaps = chain(G21, k1, s1)
rows.append(survey_row(G21, DAYS[G21["start"]][0], "first link of the 2021 basis"))
for g0, g1 in gaps:
    rows.append(("gap", "", g0, g1, "", "", "", "km3",
                 f"{days_between(g0, g1)} days without a measurement, not counted: melt in them adds to the loss"))
for t in balance(k1, s1):
    rows.append(("dV", k1, t["start"], t["end"], t["dv"] / 1e9,
                 np.sqrt(t["k"] ** 2 + t["area"] ** 2 + t["rand"] ** 2) / 1e9, "", "km3",
                 f"k x {t['b']:.3f} m w.e. x 1000/900 x {t['a'] / 1e6:.4f} km2; {t['source']}"))
rows.append(survey_row(G12, "2012", "first link of the 2012 basis, instead of the 2021 one"))

# --- 5 the map's volumes, the 2012 ice outside its outline, cumulative loss
z21, hd = read_esri_ascii(COURSE / "dem_basodino_2021.asc")
g21 = read_esri_ascii(COURSE / "mask_basodino_2021.asc")[0] == 1
CS, X0 = hd["cellsize"], hd["xllcorner"]
NY, NX = z21.shape
Y0 = hd["yllcorner"] + NY * CS                         # northern edge


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


def gpkg_geometry(blob):
    """GeoPackage geometry blob -> shapely: 8-byte header, envelope of 0-64 bytes (flag bits 1-3), then WKB."""
    return wkb.loads(bytes(blob[8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(blob[3] >> 1) & 7]:]))


con = sqlite3.connect(ROOT / "data/vectors.gpkg")
sgi2010 = [gpkg_geometry(b) for (b,) in con.execute("SELECT geom FROM sgi_2010 WHERE SGI = 'C14-10'")]
con.close()
in2010 = features.rasterize(sgi2010, out_shape=(NY, NX), transform=rasterio.transform.from_origin(X0, Y0, CS, CS))
h, s12 = on_course_grid("IceThickness"), on_course_grid("SwissALTI3D_r2019")
ring = (in2010 == 1) & np.isnan(h)
assert np.isfinite(s12[ring]).all() and np.isfinite(z21[ring]).all(), "surfaces missing on the ring"
v_out = np.sum(s12[ring] - z21[ring] + MAP["dz_stable_mean"]) * CS * CS
V21, V12 = MAP["V_2021"], MAP["V_2012"]
rows += [("V_map_2021", "", "2021-09-30", "", V21 / 1e9, "", "", "km3",
          "the map on the course mask 2021: sum of max(z_2021 - bed, 0) x 100 m2 (code/thickness.csv V_2021)"),
         ("V_map_2012", "", "2012", "", V12 / 1e9, "", "", "km3",
          "the map's own volume on the 2016 outline, 2012 surface (code/thickness.csv V_2012)"),
         ("V_2012_outside", "", "2012", "", v_out / 1e9, "", "", "km3",
          (f"upper end of the range 0 to this: sum of (z_2012 - z_2021 + {MAP['dz_stable_mean']:.3f} m) on the "
           f"{ring.sum() * CS * CS / 1e6:.4f} km2 inside the 2010 and outside the 2016 inventory "
           f"({(ring & g21).sum() * CS * CS / 1e6:.4f} km2 of it still ice in 2021)")),
         ("area_left", "", OUTLINES[-1], "", AREA[OUTLINES[-1]][0] / 1e6, AREA[OUTLINES[-1]][1] / 1e6, "", "km2",
          "our Sentinel-2 outline; sigma one pixel along the perimeter")]

VARIANTS = (("balances as published", 1.0, 0.0), ("k 2012-2024", k1, s1), ("approved conventions", K_LOW, 0.0),
            ("k 2012-2021, glacier-wide k_b", K["2012-2021"]["k"], K["2012-2021"]["sigma"]))
for basis, geo, volume in (("2021", G21, V21), ("2012", G12, V12)):
    start = DAYS[geo["start"]][0] if basis == "2021" else "2012"
    for label, k, sigma_k in VARIANTS:
        cumulative, _ = chain(geo, k, sigma_k)
        for end, loss, sg, sc in cumulative if k == k1 else cumulative[-1:]:
            rows += [(f"loss_since_{basis}", k, start, end, loss / 1e9, sg / 1e9, sc / 1e9, "km3",
                      f"{label}; gap 2024-08-23 to 2024-09-03 not counted"),
                     (f"ratio_to_map_{basis}", k, start, end, loss / volume, sg / volume, sc / volume, "",
                      f"loss_since_{basis} / V_map_{basis}")]
        if basis == "2012":
            end, loss, sg, sc = cumulative[-1]
            rows.append(("ratio_to_map_2012_with_outside", k, start, end, loss / (volume + v_out),
                         sg / (volume + v_out), sc / (volume + v_out), "",
                         "loss_since_2012 / (V_map_2012 + V_2012_outside)"))

write_csv("volume_clock.csv", ("quantity", "k", "start", "end", "value", "sigma", "sigma_correlated", "unit",
                               "definition"), rows)
for r in rows:
    print(f"{r[0]:31s} {r[1] if r[1] == '' else f'{r[1]:.4f}':>6s} {r[2]:>10s} {r[3]:>10s} "
          + " ".join(f"{v:10.5g}" if isinstance(v, float) else f"{v:>10s}" for v in r[4:7]) + f" {r[7]}")

# --- gates
print("\ngates")
for p in (G12, G21):
    rw = WE * p["dv"] / (p["a"] * p["n"])
    by_days = [r[4] for r in rows if r[0] == "rate_survey_days" and r[2][:4] == p["start"][:4]]
    print(f"  rate {day(p['start'])} to {day(p['end'])}: {rw:.4f} with whole years against {p['b']}, "
          f"{'passed' if abs(rw - p['b']) <= 0.01 else 'NOT passed'}; with the survey days "
          + ", ".join(f"{v:.4f} ({'within' if abs(v - p['b']) <= 0.01 else 'NOT within'} 0.01)" for v in by_days))
d = abs(K["2012-2024"]["k"] - K["2021-2024"]["k"])
print(f"  k 2012-2024 {K['2012-2024']['k']:.3f} +- {s1:.3f}, k 2021-2024 {K['2021-2024']['k']:.3f} +- "
      f"{K['2021-2024']['sigma']:.3f}: difference {d:.3f}, "
      f"{'passed' if d <= min(s1, K['2021-2024']['sigma']) else 'NOT passed'}")
print(f"  chain: no overlap (asserted), gaps listed: {gaps}")
