"""The last year each 10 m cell of Ghiacciaio del Basòdino (SGI C14-10) held ice, 2018-2026, from Sentinel-2 at the
end of each melt season: step 5 of the mass-conservation test of the Swiss thickness map (step 6 bounds each cell's
2021 thickness with it).

Inputs (paths below basodino/; in the s02 mirror data/ is the project's derived/, see usage):
  code/outline_2026.py, its code above '# --- the outline and its checks'   the band-ratio rule (its steps 1-5), run
                                     here unchanged, with its scene list (catalog.csv), cast shadows and Otsu thresholds
  data/satellite_archive/sentinel2/catalog.csv, scenes/<year>/   the archive: every acquisition up to 29 September
                                     2026, surface reflectance with the processing-baseline offset removed by
                                     src/archive.py (DN / 10000 - 0.1 for baseline 04.00 or later, else DN / 10000)
  data/satellite_archive/sentinel2/inventory.csv   every acquisition in the anonymous catalogues and its product
  data/satellite_archive/sentinel2/src/archive.py  its reader, for the acquisitions after the archive
  Earth Search sentinel-2-c1-l2a, Planetary Computer sentinel-2-l2a   STAC, anonymous; queried at run time
  data/vectors.gpkg, layers sgi_2016 (imagery of 2018) and glacier_2024 (swissTLM3D)   reference outlines, neighbours
  data/course/mask_basodino_2021.asc               course mask, October 2021 (read by outline_2026.py)
  data/outline_2026/outline_2026-09-27.gpkg        our September 2026 outline (1.025 km2): the glacier core, a gate
  code/outline_2026.csv                            the same method on 1 September 2021 (1.578 km2), a gate

Method:
  1. Scenes: every acquisition of 1 August to 31 October 2018-2026. The archive holds all of them up to 29 September
     2026 (inventory.csv: none in these windows unfetched; 2022 from Planetary Computer, baseline 05.10, and Earth
     Search pre-Collection-1, 04.00). Later ones come from the cloud catalogues, Earth Search c1 first: reflectance =
     DN x 0.0001 + offset, the offset from Earth Search's raster:bands metadata (-0.1), for Planetary Computer -0.1 at
     baseline 04.00 or later (its items carry no offset); archive.py's reader resamples them onto the grid and they
     pass through its uint16 encoding, as in the archive. That reader reproduces the archive's B03 and B11 exactly for
     27 September 2026 (Earth Search 05.13 and Planetary Computer 05.13), 24 August 2022 (Planetary Computer 05.10) and
     24 September 2018 (Earth Search 05.00); the 04.00 product of 24 August 2022 differs from the 05.10 one by a median
     of 0.0001 (checked 7 October 2026).
  2. Reference outline of each year, the latest not after it: SGI 2016 (imagery 2018, 1.641 km2) for 2018-2020, the
     course mask (October 2021, 1.6065 km2) for 2021-2023, swissTLM3D 2024 (1.380 km2) for 2024-2026; the glacier did
     not advance, so it bounds the ice of its years (step 4 of outline_2026.py accepts shadowed ice only inside it).
     The region of step 2 (reference plus 50 m) reaches Ghiacciaio di Basòdino Nord (C14-11) while the two touched
     (SGI 2016; the course mask overlaps its 2016 outline by 6666 m2): ice there that lies in another glacier of the
     latest inventory (SGI 2016 for 2018-2023, swissTLM3D 2024 after) and outside the reference is not counted. From
     2024 nothing changes: swissTLM3D 2024 puts Basòdino Nord 98 m away, so the region is that of outline_2026.py.
  3. Tests per scene, on the 10 m grid of outline_2026.py (SCL repeated from 20 m):
       data    no cell of the region without data (outline_2026.py asserts it)
       cloud   scene classification cloud shadow, cloud or cirrus (SCL 3, 8, 9, 10) on at most CLOUD_MAX = 1 % of
               the region
       veil    median snow index (NDSI) over the core, our 2026 outline shrunk by 20 m (ice in every year), at least
               VEIL_MIN = 0.7: thin cloud the SCL misses lowers it (the veiled scenes 0.47-0.68, among them 2, 7 and
               17 September 2026; the voters 0.72-0.94); 20 September 2026, one of outline_2026.py's five late-
               September scenes, is hazy at 0.72 and stays
       seen    at least half of the ring clear by the SCL (ring: 50 to 300 m outside the reference, off every other
               glacier of both inventories plus 50 m)
       snow    fresh snow, or snow left from the winter, against the year's best scene (minimum over the scenes that
               pass the tests above): the ring's snow fraction (the archive's test: NDSI above 0.4, B03 above 0.10,
               B08 above 0.11) at most RING_SNOW = 0.02 above the year's minimum, and the core's median B03 at most
               CORE_GREEN = 0.2 above the year's minimum. The two catch different cases: snow on the glacier and in
               the deglaciated ground inside the reference with a bare ring (11 September 2022, 26 and 28 September
               2023: core +0.33 to +0.41, ring +0.00 to +0.01; their ice 0.05 to 0.10 km2 larger than the bare scenes'),
               and August scenes with winter snow around the glacier (15 August 2018, 17 August 2021: ring +0.04,
               +0.08). Scenes after the lasting snow fail them too.
     Of the scenes that pass, the latest CHOOSE = 5 are the year's voters (the clearest by the tests; the latest are
     nearest the end of the melt season). A year with fewer than 2 is flagged.
  4. Ice per voter: steps 1-5 of outline_2026.py with the year's reference: the iso-line of NDSI minus the Otsu
     threshold (sunlit and in cast shadow) between pixel centres, narrow strips joined by the 10 m bands; a cell is ice
     where its centre lies in the ice polygons (z > 0), less the neighbouring glacier of 2 (39 to 54 cells a voter in
     2018-2020, 14 in 2021, none after). Step 6 (body, holes and blocks) is replaced by the vote.
  5. Vote: a cell is ice in a year where more than half of the year's voters show ice (2 of 2, 2 of 3, 3 of 4, 3 of 5;
     a flagged year's single scene alone).
  6. Last ice: the latest year in which the cell is ice. A cell iced again after a year without ice keeps the later
     year; such cells are counted per year. They lie at the margins (median 14 m from the year before's ice), where
     the index sits near the threshold in the bare years and old snow lifts it in the snow-rich ones: 1020 cells in
     2024 (10 September, the core's B03 0.68, ring snow 0.052) against 22 to 117 in the other years. The error makes
     the last year later, never earlier.
  Gates: yearly ice areas within GATE = 0.09 km2 of the same method on 1 September 2021 (code/outline_2026.csv,
  1.578 km2), of swissTLM3D 2024 (1.380 km2) and of our 2026 outline (1.025 km2).
  Run of 7 October 2026: 353 scenes considered (4 after the archive, all clouded over the glacier); 21 voters; ice
  2018-2026 1.544, 1.612, 1.572, 1.584, 1.391, 1.284, 1.387, 1.222, 1.041 km2; gates +0.006, +0.007, +0.016 km2;
  flagged with one voter each 2019, 2020, 2021, 2024. The 2026 voters are outline_2026.py's five; 27 September alone
  gives 1.0262 km2 of ice cells against its outline's 1.0252 km2.

Outputs:
  code/last_ice.tif         int16, 10 m, EPSG:2056, E 2678700-2682100, N 1140000-1142900 (cell edges on the 10 m
                            lattice of the course grid; 340 x 290 cells): 2018-2025 the last year with ice, 2026 ice
                            still in 2026, 0 no ice in any year 2018-2026, -9999 (nodata) outside the analysed area
                            (outside every year's region: other glaciers, Basòdino Nord included, are not analysed)
  code/last_ice_scenes.csv  one row per scene considered (date order): source, product, processing baseline, the
                            offset removed and by which program, the test values, the decision; for the voters the two
                            thresholds, the cast-shadow share of the region and the ice area (cells x 100 m2); dark_b03
                            and dark_b11, the first percentiles over the window: the voters' lie at 0.006-0.054, with
                            the offset left in they would lie 0.1 higher. Then one row per year: voters, reference,
                            cells iced again, ice area, the gate reference and the difference
  stdout: the voters per year, the yearly areas and the gates, the cells per last year, the voters' offsets

usage, on s02 where the scenes are (env agl), from the mirror of this tree (outline_2026.py's docstring), about 1 min:
  rsync -a code/common.py code/outline_2026.py code/last_ice.py s02:/data/befahl/projects/AGL/basodino/report/code/
  ssh s02 'source ~/miniconda3/bin/activate agl && cd /data/befahl/projects/AGL/basodino/report &&
           ~/.local/bin/nrun python code/last_ice.py'
  then copy code/last_ice.tif and code/last_ice_scenes.csv back
"""
from pathlib import Path

exec((Path(__file__).resolve().parent / "outline_2026.py").read_text(encoding="utf-8")  # noqa: S102
     .split("# --- the outline and its checks")[0])                   # the method, unchanged

import sys  # noqa: E402

import requests  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402

from common import CODE  # noqa: E402

sys.path.insert(0, str(S2 / "src"))
sys.dont_write_bytecode = True                                         # no cache files in the archive's src/
import archive  # noqa: E402  (the archive's reader, offsets and encoding)

YEARS, SEASON = range(2018, 2027), ("08-01", "10-31")                # late season, month-day
OUT_BOX = (2678700, 1140000, 2682100, 1142900)                         # output grid, LV95 m
CLOUD_MAX, VEIL_MIN, RING_SNOW, CORE_GREEN, CHOOSE = 0.01, 0.7, 0.02, 0.2, 5
RING, CORE_IN, GATE, NODATA = 300.0, 20.0, 0.09e6, -9999                # m, m, m2
CLOUDY = (3, 8, 9, 10)                                                 # SCL: cloud shadow, cloud, cirrus
STAC = {"earth-search/sentinel-2-c1-l2a": "https://earth-search.aws.element84.com/v1",
        "planetary-computer/sentinel-2-l2a": "https://planetarycomputer.microsoft.com/api/stac/v1"}
ES_KEYS = dict(B02="blue", B03="green", B04="red", B08="nir", B11="swir16", SCL="scl")

# --- reference outline, region and zones of each year
g16 = gpd.read_file(ROOT / "data/vectors.gpkg", layer="sgi_2016")
g24 = gpd.read_file(ROOT / "data/vectors.gpkg", layer="glacier_2024")
G16 = g16[g16["sgi-id"] == "C14-10"].union_all()
N16, N24 = g16[g16["sgi-id"] != "C14-10"].union_all(), g24.query("sgi != 'C14-10'").union_all()
G26 = gpd.read_file(ROOT / "data/outline_2026/outline_2026-09-27.gpkg").union_all()
OTHERS = shapely.union(N16, N24).buffer(BUFFER)
CORE = raster(G26.buffer(-CORE_IN))
ERAS = []
for start, name, ref, neighbours in ((2018, "SGI 2016 (imagery 2018)", G16, N16),
                                     (2021, "course mask 2021", G21, N16), (2024, "swissTLM3D 2024", G24, N24)):
    region = raster(ref.buffer(BUFFER))                                # step 2 of outline_2026.py
    ERAS.append(dict(start=start, name=name, ref=ref, region=region,
                     keep=region & ~(raster(neighbours) & ~raster(ref)),
                     ring=raster(ref.buffer(RING)) & ~region & ~raster(OTHERS)))
ERA = {y: [e for e in ERAS if e["start"] <= y][-1] for y in YEARS}


def in_season(date):
    return int(date[:4]) in YEARS and SEASON[0] <= date[5:10] <= SEASON[1]


# --- the scenes: the archive, then the cloud catalogues after its last acquisition
with open(S2 / "inventory.csv") as f:
    missing = [r["acquisition"] for r in csv.DictReader(f) if in_season(r["date"]) and not r["fetched_from"]]
assert not missing, f"acquisitions in the windows not in the archive: {missing}"
scenes = [dict(tag=t, date=t[:10], datetime=c["datetime"], platform=c["platform"], orbit=c["orbit"], source=c["source"],
               product=c["product"], baseline=c["processing_baseline"], offset=float(c["source_offset"]),
               offset_by="src/archive.py, baseline rule", sun=(float(c["sun_elevation"]), float(c["sun_azimuth"])))
          for t, c in CAT.items() if in_season(t[:10])]
end = max(c["datetime"] for c in CAT.values())
found = {}
for source, api in STAC.items():                                       # Earth Search first: kept on equal keys
    body = {"collections": [source.split("/")[1]], "limit": 100,
            "bbox": list(transform_bounds("EPSG:2056", "EPSG:4326", E0, N0, E1, N1)),
            "datetime": f"{end}/{max(YEARS)}-{SEASON[1]}T23:59:59Z"}
    for item in requests.post(f"{api}/search", json=body, timeout=120).json()["features"]:
        p = item["properties"]
        if "32TMS" not in (p.get("grid:code") or p.get("s2:mgrs_tile")):
            continue
        sat, _, when, _, orbit = p["s2:product_uri"].split("_")[:5]
        tag, base = f"{when[:4]}-{when[4:6]}-{when[6:8]}_{sat}_{orbit}", p["s2:processing_baseline"]
        if tag in CAT or tag in found and (found[tag]["source"] != source or found[tag]["baseline"] >= base):
            continue
        keys = ES_KEYS if source.startswith("earth") else {b: b for b in ES_KEYS}
        meta = item["assets"][keys["B03"]].get("raster:bands", [{}])[0]
        offset = -0.1 if base >= "04.00" else 0.0
        assert meta.get("offset", offset) == offset and meta.get("scale", 1e-4) == 1e-4, f"{tag}: {meta}"
        found[tag] = dict(tag=tag, date=tag[:10], datetime=p["datetime"], platform=p["platform"].lower(), orbit=orbit,
                          source=source, product=p["s2:product_uri"].removesuffix(".SAFE"), baseline=base,
                          offset=offset, offset_by="last_ice.py, " + ("Earth Search metadata" if "offset" in meta
                                                                       else "baseline rule"),
                          sun=(p["view:sun_elevation"], p["view:sun_azimuth"]) if "view:sun_elevation" in p else
                          (90 - p["s2:mean_solar_zenith"], p["s2:mean_solar_azimuth"]),
                          href={b: item["assets"][k]["href"] for b, k in keys.items()})
scenes = sorted(scenes + list(found.values()), key=lambda s: s["datetime"])


def crop(a, res):
    """The window of outline_2026.py out of the archive's 10 km square."""
    r, c = round((archive.Y0 - N1) / res), round((E0 - archive.X0) / res)
    return a[r:r + round((N1 - N0) / res), c:c + round((E1 - E0) / res)]


def cloud_scene(s):
    """Bands of a scene from a cloud catalogue, as scene() gives them for the archive (read, offset and resampled
    by archive.py's reader, through its uint16 encoding), and its SCL."""
    res = dict(B02=10, B03=10, B04=10, B08=10, B11=20)
    v = {b: crop(archive.encode(archive.read(s["href"][b], r, s["offset"])), r).astype(np.float32)
         for b, r in res.items()}
    bands = {b: np.where(v[b] == 0, np.nan, v[b] / 1e4) for b in ("B02", "B03", "B04", "B08")}
    bands["B11"] = zoom(np.where(v["B11"] == 0, np.nan, v["B11"] / 1e4), 2, order=1, mode="nearest", grid_mode=True)
    return dict(**bands, sun=s["sun"]), crop(archive.read(s["href"]["SCL"], 20), 20)


CLOUD_BANDS, archive_scene = {}, scene


def scene(tag):  # noqa: F811  outline() reads every scene's bands through this name
    return CLOUD_BANDS[tag] if tag in CLOUD_BANDS else archive_scene(tag)


# --- tests per scene
for s in scenes:
    if "href" in s:
        CLOUD_BANDS[s["tag"]], scl = cloud_scene(s)
    else:
        scl = read(S2 / CAT[s["tag"]]["file_scl"], band=1)
    b, e, scl = scene(s["tag"]), ERA[int(s["date"][:4])], np.repeat(np.repeat(scl, 2, 0), 2, 1)
    ndsi = (b["B03"] - b["B11"]) / (b["B03"] + b["B11"])
    seen = e["ring"] & (scl > 1) & ~np.isin(scl, CLOUDY)
    snow = (ndsi > 0.4) & (b["B03"] > 0.10) & (b["B08"] > 0.11)
    s.update(nodata=int((~np.isfinite(ndsi) & e["region"]).sum()), cloud=np.isin(scl, CLOUDY)[e["region"]].mean(),
             veil=np.nanmedian(ndsi[CORE]), green=np.nanmedian(b["B03"][CORE]), seen=seen.sum() / e["ring"].sum(),
             ring_snow=snow[seen].mean() if seen.any() else np.nan, dark_b03=np.nanpercentile(b["B03"], 1),
             dark_b11=np.nanpercentile(b["B11"], 1), decision="")
    for test, fails in (("no data in the region", s["nodata"] > 0), ("cloud", s["cloud"] > CLOUD_MAX),
                        ("veiled", s["veil"] < VEIL_MIN), ("ring not seen", s["seen"] < 0.5)):
        if fails and not s["decision"]:
            s["decision"] = test

# --- choice per year, ice of the voters, vote
ICE, VOTERS = {}, {}
for y in YEARS:
    ok = [s for s in scenes if s["date"][:4] == str(y) and not s["decision"]]
    assert ok, f"{y}: no scene passes the tests"
    ring_min, green_min = min(s["ring_snow"] for s in ok), min(s["green"] for s in ok)
    for s in ok:
        if s["ring_snow"] > ring_min + RING_SNOW:
            s["decision"] = f"snow around the glacier (year's minimum {ring_min:.3f})"
        elif s["green"] > green_min + CORE_GREEN:
            s["decision"] = f"snow on the glacier (year's minimum {green_min:.3f})"
    passed = sorted((s for s in ok if not s["decision"]), key=lambda s: s["datetime"], reverse=True)
    VOTERS[y] = passed[:CHOOSE]
    assert VOTERS[y], f"{y}: no scene passes both snow tests"
    for s in passed[CHOOSE:]:
        s["decision"] = f"passes, not among the latest {CHOOSE}"
    e, votes = ERA[y], 0
    for s in VOTERS[y]:
        r = outline(s["tag"], e["ref"])
        ice = (r["z"] > 0) & e["keep"]
        s.update(decision=f"voter, {len(VOTERS[y])} in {y}", t_sun=float(r["t_sun"]), t_shade=float(r["t_shade"]),
                 shade=float(r["shade"][e["region"]].mean()), area=ice.sum() * PIXEL ** 2,
                 cut=int(((r["z"] > 0) & e["region"] & ~e["keep"]).sum()))
        votes = votes + ice
    ICE[y] = 2 * votes > len(VOTERS[y])

# --- last ice, on the output grid
last = np.zeros((len(Y), len(X)), np.int16)
for y in YEARS:
    last[ICE[y]] = y
last[~np.any([e["keep"] for e in ERAS], axis=0)] = NODATA
w0, s0, e1, n1 = OUT_BOX
grid = np.full((round((n1 - s0) / PIXEL), round((e1 - w0) / PIXEL)), NODATA, np.int16)
west, east, south, north = max(E0, w0), min(E1, e1), max(N0, s0), min(N1, n1)       # the two windows' overlap


def overlap(top, left):
    """Rows and columns of the overlap in a grid with this north-west corner."""
    return (slice(round((top - north) / PIXEL), round((top - south) / PIXEL)),
            slice(round((west - left) / PIXEL), round((east - left) / PIXEL)))


grid[overlap(n1, w0)] = last[overlap(N1, E0)]
assert (last != NODATA).sum() == (grid != NODATA).sum(), "analysed cells outside the output grid"
with rasterio.open(CODE / "last_ice.tif", "w", driver="GTiff", width=grid.shape[1], height=grid.shape[0], count=1,
                   dtype="int16", crs="EPSG:2056", transform=from_origin(w0, n1, PIXEL, PIXEL), nodata=NODATA,
                   compress="deflate") as dst:
    dst.write(grid, 1)
    dst.set_band_description(1, "last year with ice, Ghiacciaio del Basòdino C14-10")
    dst.update_tags(codes="2018-2025 last year with ice; 2026 ice still in 2026; 0 no ice in any year 2018-2026; "
                          "-9999 outside the analysed area", creator="code/last_ice.py",
                    voters="; ".join(f"{y}: " + " ".join(s["tag"] for s in VOTERS[y]) for y in YEARS))

# --- the scene table, the years, the gates
with open(CODE / "outline_2026.csv") as f:
    s21 = next(r for r in csv.DictReader(f) if r["scene"] == "2021-09-01_S2B_R108")
GATES = {2021: ("outline_2026.py on 1 September 2021", float(s21["area_m2"])),
         2024: ("swissTLM3D 2024", G24.area), 2026: ("our outline of 27 September 2026", G26.area)}


def num(v, digits):
    """A number for the table, blank if missing; whole numbers as integers (write_csv keeps 6 digits of floats)."""
    return "" if v == "" or not np.isfinite(v) else round(float(v), digits) if digits else round(float(v))


table = []
for s in scenes:
    v = {k: s.get(k, "") for k in ("t_sun", "t_shade", "shade", "area")}
    table.append([s["date"][:4], s["tag"], s["datetime"], s["platform"], s["orbit"], s["source"], s["product"],
                  s["baseline"], s["offset"], s["offset_by"], round(s["sun"][0], 1), num(s["cloud"], 4), s["nodata"],
                  num(s["veil"], 3), num(s["green"], 3), num(s["seen"], 3), num(s["ring_snow"], 4),
                  num(s["dark_b03"], 4), num(s["dark_b11"], 4), s["decision"], num(v["t_sun"], 3),
                  num(v["t_shade"], 3), num(v["shade"], 3), num(v["area"], 0), "", ""])
regained = {}                                                          # ice again after a year without ice
for y in YEARS:
    area, n = ICE[y].sum() * PIXEL ** 2, len(VOTERS[y])
    name, ref_area = GATES.get(y, ("", None))
    before = [ICE[x] for x in YEARS if x < y - 1]
    regained[y] = int((ICE[y] & ~ICE[y - 1] & np.any(before, axis=0)).sum()) if before else 0
    flag = "flagged: fewer than 2 clear scenes" if n < 2 else f"ice in more than half of {n}"
    table.append([y, "year", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                  f"{flag}; reference {ERA[y]['name']}; {regained[y]} cells ice again after a year without", "",
                  "", "", round(area), name, round(area - ref_area) if ref_area else ""])
write_csv("last_ice_scenes.csv", ["year", "scene", "datetime", "platform", "orbit", "source", "product",
                                  "processing_baseline", "offset", "offset_removed_by", "sun_elevation_deg",
                                  "cloud_region", "nodata_region_cells", "veil_core_ndsi", "core_b03", "ring_seen",
                                  "ring_snow", "dark_b03", "dark_b11", "decision", "threshold_sunlit",
                                  "threshold_shadow", "shadow_region", "ice_area_m2", "compared_with",
                                  "difference_m2"], table)

for y in YEARS:
    print(f"{y} ({ERA[y]['name']}): {ICE[y].sum() * PIXEL ** 2 / 1e6:.3f} km2 from {len(VOTERS[y])}: " + ", ".join(
        f"{s['tag']} {s['area'] / 1e6:.3f}" for s in VOTERS[y]) + (" FLAGGED" if len(VOTERS[y]) < 2 else "")
        + f"; iced again after a year without ice {regained[y]} cells; cut as another glacier's "
        + "/".join(str(s["cut"]) for s in VOTERS[y]))
for y, (name, ref_area) in GATES.items():
    d = ICE[y].sum() * PIXEL ** 2 - ref_area
    print(f"gate {y}: {ICE[y].sum() * PIXEL ** 2 / 1e6:.3f} km2 against {name} {ref_area / 1e6:.3f} km2, "
          f"difference {d / 1e6:+.3f} km2, {'passed' if abs(d) <= GATE else 'NOT passed'}")
print("last ice, cells: " + ", ".join(f"{v} {(grid == v).sum()}" for v in [0, *YEARS]))
voters = [s for y in YEARS for s in VOTERS[y]]
print(f"voters: {len(voters)}, offsets {sorted({(s['source'], s['baseline'], s['offset']) for s in voters})}, "
      f"dark_b03 {min(s['dark_b03'] for s in voters):.4f}-{max(s['dark_b03'] for s in voters):.4f}, "
      f"dark_b11 {min(s['dark_b11'] for s in voters):.4f}-{max(s['dark_b11'] for s in voters):.4f}")
print(f"scenes considered {len(scenes)} ({len(found)} from the cloud catalogues after {end}); "
      f"written code/last_ice.tif, code/last_ice_scenes.csv")
