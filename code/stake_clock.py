"""Stake clock: the ice melted at the GLAMOS stake sites of Basòdino after the 2021 surface (30 September 2021),
against the largest mapped 2021 thickness at the same place (Swiss Glacier Thickness, release 2020).

Inputs (provenance and licences in data/glamos_point_2025/README.md and data/thickness/README.md):
  data/glamos_point_2025/basodino_annual.dat   GLAMOS point mass balance, release 2025: annual readings, positions in
                                               Swiss coordinates LV03 (z_pos is the installation height: not used)
  data/glamos_point_2025/fig_a1_basodino.csv   Fig. A.1 of the GLAMOS 2026 annual report, PROVISIONAL: point balances of
                                               6 Sep 2025 - 29 Sep 2026 (m w.e.) read twice, pixel centres of their
                                               triangles, the scale bar
  data/thickness/C14-10_GlacierBed.tif, C14-10_IceThicknessUncertaintyPlus.tif   bed and its +1 sigma (deeper bed)
  data/thickness/C14-10_basodino_GlaThiDa.txt  radar points of 2006 (LV95, bed elevation)
  data/course/dem_basodino_2021.asc, mask_basodino_2021.asc   surface and glacier of 30 September 2021
  data/outline_2026/outline_2026-09-27.geojson                Sentinel-2 outline of 27 September 2026
Method:
  1. Readings from the survey of 10 Sep 2021 on (the one nearest the 2021 surface). Site = last two digits of the stake
     name (2305 and 2205b both stand at site 05); stakes read at one site over the same period are averaged (2023/24:
     sites 05, 09, 10). Melt = -mb_raw, the reading in cm of ice (every reading is ice at 900 kg m-3, asserted);
     where mb_we differs from mb_raw x 9 it is noted, and so is measurement quality 3 (a reconstructed value: stake
     melted out or beyond its range; a melted-out stake makes the melt a lower bound). Period end = date0 + period; where date1 disagrees it is noted
     (site 04, 2022/23: date1 5 Sep 2023, period and next start 5 Oct).
  2. Series: consecutive periods of a site are chained where they join (within a day) and their positions lie within
     60 m; a longer move or a gap starts a new series (letters a, b).
  3. Fig. A.1: the two readings must agree (asserted); a hidden digit (?) gives no value. Triangles to LV03 by a
     similarity transform: scale of the scale bar, north up, the translation that brings most triangles within 30 m
     of a stake read up to 6 Sep 2025 (every triangle-stake pair tried), then least squares on those pairs. The
     nearest triangle within 60 m of such a stake continues its series to 29 Sep 2026: melt = -balance x 1000 / 900.
  4. Melt after 30 Sep 2021: series from 10 Sep 2021 minus 0.3 m, the upper bound of the melt of 10-30 Sep 2021 (from
     the specification, not derived here); series that start later are lower bounds as they stand.
  5. Map, as in thickness.py: 2021 thickness = max(z_2021 - bed, 0) on the 2021 mask, 0 off the outline of the data;
     +1 sigma: the same with the bed lowered by IceThicknessUncertaintyPlus. Largest of each over the cells whose
     centre lies within 30 m of any position of the series. No emergence (ice flow) is added: melt, not thinning.
  6. Radar: the 2006 point nearest to any position of the series, its own thickness (2006) and its 2021 thickness =
     z_2021 of the cell under it minus its bed (negative: the 2021 surface lies below that bed). Outline 2026: the
     last position inside or not, and its distance to the edge (negative outside).
     LV95 = LV03 + (2 000 000, 1 000 000) m, the plain offset.
Outputs:
  code/stake_clock.csv  one row per series and end (release data to Sep 2025; with Fig. A.1 to 29 Sep 2026,
                        provisional = yes): series, start, end, provisional, readings, step_m (largest move between
                        consecutive positions), spread_m (largest distance between any two), melt_m (measured since
                        start), melt_after_30sep2021_m, map_max_m, map_plus_sigma_max_m,
                        excess_m (melt_after - map_max),
                        beyond_map_plus_1m, beyond_plus_sigma, radar_distance_m, radar_h2006_m, radar_h2021_m,
                        easting, northing (LV95, last position), inside_2026, edge_2026_m, fig_a1_triangle,
                        fig_a1_reading_1, fig_a1_reading_2 (m w.e.), fig_a1_offset_m (triangle to the stake of
                        2024/25), note
  code/stake_positions.csv  every position of each row of stake_clock.csv (series, end, provisional, k, easting,
                        northing, LV95): the 30 m discs of the start-state fit (igm/DESIGN.md step 4)
  stdout                placement of Fig. A.1 (scale, rotation, residuals), the table in short
usage:  python3 code/stake_clock.py   (conda base: rasterio, numpy, shapely)
"""
import csv
import json
from datetime import date, timedelta

import numpy as np
import rasterio
from shapely.geometry import Point, shape

from common import COURSE, RHO_ICE, RHO_W, ROOT, read_esri_ascii, write_csv

POINT, THICK = ROOT / "data/glamos_point_2025", ROOT / "data/thickness"
START = date(2021, 9, 10)                      # survey nearest to the 2021 surface of 30 Sep 2021
SEPTEMBER = 0.3                                # m of ice: upper bound of the melt of 10-30 Sep 2021 (specification)
CHAIN, MATCH, DISC = 60.0, 30.0, 30.0          # m: largest move within a series, triangle fit, map disc radius
LV95 = np.array([2_000_000.0, 1_000_000.0])


def day(s):
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def dist(a, b):
    return float(np.hypot(*(np.asarray(a, float) - np.asarray(b, float))))


def yes(b):
    return "yes" if b else "no"


# --- 1. readings since 10 Sep 2021, one per site and period
with open(POINT / "basodino_annual.dat", encoding="latin-1") as f:
    rows = [line.split() for line in f if not line.startswith("#")]
recent = [r for r in rows if day(r[1]) >= START]
assert len(rows) == 344 and len(recent) == 41, (len(rows), len(recent))
assert all(r[12] == "900" for r in recent), "a reading that is not ice"
grouped = {}
for r in recent:
    grouped.setdefault((r[0].rstrip("b")[-2:], r[1]), []).append(r)
sites = {}
for (site, d0), rs in sorted(grouped.items()):
    end, note = day(d0) + timedelta(days=round(float(rs[0][5]))), []
    note += [f"stake {r[0]} {d0[:4]}: date1 {r[3]}, date0 + period {end:%Y%m%d}" for r in rs if day(r[3]) != end]
    melts = [-float(r[11]) / 100 for r in rs]
    note += [f"stake {r[0]} {d0[:4]}: mb_we {r[14]} mm w.e. against mb_raw {r[11]} cm" for r in rs
             if float(r[14]) != float(r[11]) * RHO_ICE / 100]
    note += [f"stake {r[0]} {d0[:4]}: measurement quality 3, reconstructed (stake melted out or beyond its range; "
             "GLAMOS adds 40 cm)" for r in rs if r[15] == "3"]
    if len(rs) > 1:
        note.append(f"{d0[:4]}/{int(d0[2:4]) + 1}: stakes " + ", ".join(f"{r[0]} {m:.2f} m" for r, m in zip(rs, melts))
                    + " averaged")
    sites.setdefault(site, []).append({"start": day(d0), "end": end, "melt": float(np.mean(melts)), "note": note,
                                       "xy": np.mean([[float(r[7]), float(r[8])] for r in rs], 0)})
assert len(sites) == 10

# --- 2. series: chained where the periods join and the stake moved at most 60 m
series, why = {}, {}
for site, ps in sites.items():
    runs, breaks = [[ps[0]]], []
    for p in ps[1:]:
        q = runs[-1][-1]
        move, gap = dist(p["xy"], q["xy"]), abs((p["start"] - q["end"]).days)
        if move <= CHAIN and gap <= 1:
            runs[-1].append(p)
        else:
            runs.append([p])
            breaks.append(f"moved {move:.0f} m" if move > CHAIN else f"{gap} days without reading")
    for k, run in enumerate(runs):
        name = site + ("abcdef"[k] if len(runs) > 1 else "")
        series[name], why[name] = run, []
        if k:
            why[name].append(f"new series: the stake {breaks[k - 1]} in {run[0]['start'].year}")
        if k < len(runs) - 1:
            why[name].append(f"ends: the stake {breaks[k]} in {runs[k + 1][0]['start'].year}")
        if run[0]["start"] > START:
            why[name].append(f"lower bound: not read at these positions from {START} to {run[0]['start']}")

# --- 3. Fig. A.1 (provisional): read twice, placed by a similarity transform
with open(POINT / "fig_a1_basodino.csv", encoding="utf-8") as f:
    fig = {r["feature"]: r for r in csv.DictReader(f)}
assert all(r["reading_1"] == r["reading_2"] for r in fig.values()), "Fig. A.1 read differently twice"
f0, f1 = (date.fromisoformat(fig[f"measured period {k}"]["reading_1"]) for k in ("start", "end"))
bar = [float(fig[f"scale bar {m} m"]["column_px"]) for m in (0, 600)]
scale = 600 / (bar[1] - bar[0])                                         # m per pixel
tri = [r for k, r in fig.items() if k.startswith("triangle")]
px = np.array([[float(r["column_px"]), -float(r["row_px"])] for r in tri])   # y up: north
live = [s for s, run in series.items() if run[-1]["end"] == f0]          # stakes read up to 6 Sep 2025
ref = np.array([series[s][-1]["xy"] for s in live])


def nearest(xy):
    d = np.hypot(*(xy[:, None, :] - ref[None]).transpose(2, 0, 1))
    return d.argmin(1), d.min(1)


shifts = [ref[j] - scale * px[i] for i in range(len(px)) for j in range(len(ref))]
shift = max(shifts, key=lambda t: (nearest(scale * px + t)[1] <= MATCH).sum())
j, d = nearest(scale * px + shift)
m = d <= MATCH
x, y = px[m].T
o, z = np.ones(m.sum()), np.zeros(m.sum())
a, b, ce, cn = np.linalg.lstsq(np.r_[np.c_[x, -y, o, z], np.c_[y, x, z, o]], np.r_[ref[j[m], 0], ref[j[m], 1]],
                               rcond=None)[0]
xy = np.c_[a * px[:, 0] - b * px[:, 1] + ce, b * px[:, 0] + a * px[:, 1] + cn]
j, d = nearest(xy)
fitted = [f"{tri[i]['feature']} {d[i]:.1f} m" for i in np.flatnonzero(m)]
print(f"Fig. A.1: {np.hypot(a, b):.3f} m per pixel (scale bar {scale:.3f}), rotation "
      f"{np.degrees(np.arctan2(b, a)):.2f} degrees; residuals of the {m.sum()} fitted triangles: " + ", ".join(fitted))

extended, linked = {}, {}
for k, s in enumerate(live):
    near = [i for i in range(len(tri)) if j[i] == k and d[i] <= CHAIN]
    if near:
        linked[min(near, key=lambda i: d[i])] = s
for i, s in linked.items():
    if "?" not in tri[i]["reading_1"]:
        melt = -float(tri[i]["reading_1"]) * RHO_W / RHO_ICE
        extended[s] = series[s] + [{"start": f0, "end": f1, "xy": xy[i], "melt": melt, "note": [], "triangle": i}]
last = {s: run[-1] for s, run in series.items()}
for i, r in enumerate(tri):
    if i in linked and linked[i] in extended:
        continue
    s = min(last, key=lambda s: dist(last[s]["xy"], xy[i]))
    gap = dist(last[s]["xy"], xy[i])
    reason = (f"the series ends {last[s]['end']}" if last[s]["end"] != f0 else "farther than 60 m" if gap > CHAIN
              else "a digit is hidden" if "?" in r["reading_1"] else "another triangle is nearer")
    why[s].append(f"Fig. A.1 {r['feature']} ({r['reading_1']} m w.e.) {gap:.0f} m away, not used: {reason}")

# --- 4. map, radar and outline on the course grid
z21, hd = read_esri_ascii(COURSE / "dem_basodino_2021.asc")
g21 = read_esri_ascii(COURSE / "mask_basodino_2021.asc")[0] == 1
CS, (NY, NX) = hd["cellsize"], z21.shape
X0, Y0 = hd["xllcorner"], hd["yllcorner"] + NY * CS                    # west and north edges
E, N = X0 + (np.arange(NX) + 0.5) * CS, Y0 - (np.arange(NY) + 0.5) * CS  # cell centres, LV95


def on_course(name):
    """A thickness grid cell for cell on the course grid (same 10 m lattice, asserted), NaN off the data."""
    with rasterio.open(THICK / f"C14-10_{name}.tif") as src:
        g, t = src.read(1, masked=True).astype(float).filled(np.nan), src.transform
    assert t.a == CS and (t.c - X0) % CS == 0 and (t.f - Y0) % CS == 0, "cell edges off the course lattice"
    c, r = ((E - t.c) // CS).astype(int), ((t.f - N) // CS).astype(int)
    rr, cc = (r >= 0) & (r < g.shape[0]), (c >= 0) & (c < g.shape[1])
    out = np.full((NY, NX), np.nan)
    out[np.ix_(rr, cc)] = g[np.ix_(r[rr], c[cc])]
    return out


bed, plus = on_course("GlacierBed"), on_course("IceThicknessUncertaintyPlus")
on = g21 & np.isfinite(bed)
h21 = np.where(on, np.clip(z21 - bed, 0, None), 0.0)
h21p = np.where(on, np.clip(z21 - bed + plus, 0, None), 0.0)
EE, NN = np.meshgrid(E, N)


def largest(field, positions):
    near = np.zeros(field.shape, bool)
    for e, n in positions:
        near |= np.hypot(EE - e, NN - n) <= DISC
    return float(field[near].max())


radar = np.loadtxt(THICK / "C14-10_basodino_GlaThiDa.txt", usecols=(3, 4, 7, 8))   # easting, northing, bed, thickness
radar_h21 = z21[((Y0 - radar[:, 1]) // CS).astype(int), ((radar[:, 0] - X0) // CS).astype(int)] - radar[:, 2]
with open(ROOT / "data/outline_2026/outline_2026-09-27.geojson", encoding="utf-8") as f:
    outline = shape(json.load(f)["features"][0]["geometry"])

# --- 5. one row per series and end
HEADER = ("series", "start", "end", "provisional", "readings", "step_m", "spread_m", "melt_m", "melt_after_30sep2021_m",
          "map_max_m", "map_plus_sigma_max_m", "excess_m", "beyond_map_plus_1m", "beyond_plus_sigma",
          "radar_distance_m", "radar_h2006_m", "radar_h2021_m", "easting", "northing", "inside_2026", "edge_2026_m",
          "fig_a1_triangle", "fig_a1_reading_1", "fig_a1_reading_2", "fig_a1_offset_m", "note")
out, positions = [], []
for s in sorted(series):
    for periods, provisional in ((series[s], False), (extended.get(s), True)):
        if periods is None:
            continue
        pos = [p["xy"] + LV95 for p in periods]
        positions += [[s, periods[-1]["end"], yes(provisional), k, f"{e:.1f}", f"{n:.1f}"]
                      for k, (e, n) in enumerate(pos)]
        melt = sum(p["melt"] for p in periods)
        after = melt - SEPTEMBER if periods[0]["start"] == START else melt
        hmax, hpmax = largest(h21, pos), largest(h21p, pos)
        dr = np.min([np.hypot(radar[:, 0] - e, radar[:, 1] - n) for e, n in pos], 0)
        k, (e, n) = dr.argmin(), pos[-1]
        inside = outline.contains(Point(e, n))
        edge = outline.boundary.distance(Point(e, n)) * (1 if inside else -1)
        step = max([dist(p, q) for p, q in zip(pos, pos[1:])], default=0.0)
        spread = max(dist(p, q) for p in pos for q in pos)
        t = periods[-1].get("triangle")
        fig_cols = ["", "", "", ""] if t is None else [tri[t]["feature"], tri[t]["reading_1"], tri[t]["reading_2"],
                                                         round(dist(xy[t], periods[-2]["xy"]), 1)]
        note = [n_ for p in periods for n_ in p["note"]] + why[s]
        out.append([s, periods[0]["start"], periods[-1]["end"], yes(provisional), len(periods), round(step, 1),
                    round(spread, 1), round(melt, 2), round(after, 2), round(hmax, 2), round(hpmax, 2),
                    round(after - hmax, 2), yes(after > hmax + 1), yes(after > hpmax), round(float(dr[k]), 1),
                    round(float(radar[k, 3]), 2), round(float(radar_h21[k]), 2), round(e), round(n), yes(inside),
                    round(edge, 1), *fig_cols, "; ".join(note)])
write_csv("stake_clock.csv", HEADER, out)
write_csv("stake_positions.csv", ("series", "end", "provisional", "k", "easting", "northing"), positions)

print(f"{'series':7s}{'end':>11s} prov  melt  after  map  map+1s  excess  radar(m,h)    in2026")
for r in out:
    print(f"{r[0]:7s}{str(r[2]):>11s} {r[3]:4s}{r[7]:6.2f}{r[8]:7.2f}{r[9]:5.1f}{r[10]:7.1f}{r[11]:8.2f}"
          f"  {r[14]:5.1f} {r[16]:5.1f}   {r[19]} ({r[20]} m)")
