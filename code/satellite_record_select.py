"""One end-of-summer image per year, 1938-2026, from the archive catalogue -> tex/build/record_files.csv (read by
code/satellite_record_cut.py on s02 and by code/satellite_record.py).
Inputs, copies of the catalogues of s02:/data/befahl/projects/AGL/basodino/derived/satellite_archive in tex/build/:
  catalog_all.csv (every scene of every source with its cloud, shadow, snow and no-data fractions over the 2024
  glacier, the 1850 extent and the 10 km square, and the measured geolocation shift), catalog_other.csv (aerial
  orthophotos and swissEO: file paths), catalog_s2.csv (Sentinel-2 tags and files), catalog_landsat.csv (files).
Rules:
  window        10 August to 10 October
  usable        at most 2 per cent of the 2024 glacier without data; satellites: at most 5 per cent cloud and
                10 per cent cloud shadow over it; Landsat: tier 1, or a checked geolocation with a measured shift
                of at most 60 m (tier 2 scenes are displaced by hundreds of metres here)
  source order  aerial orthophoto (swisstopo), Sentinel-2, Landsat TM, ETM+ and OLI (30 m), Landsat MSS (60 m);
                PlanetScope is substituted later by code/satellite_record.py where a clipped scene exists
  within a source  satellites: the latest scene whose snow in the square is within 1 per cent of the year's
                minimum (the end of the melt season); aerial: the image with the least missing data over the 1850
                extent, then the latest flight
  by eye       REJECT lists scenes the rules accept that the eye rejects, with the reason; a year whose only
                flight is rejected has no tile
usage, from basodino/: python3 code/satellite_record_select.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "tex/build"
rows = list(csv.DictReader(open(B / "catalog_all.csv")))
other = {r["id"]: r for r in csv.DictReader(open(B / "catalog_other.csv"))}
s2 = list(csv.DictReader(open(B / "catalog_s2.csv")))
landsat = {r["id"]: r for r in csv.DictReader(open(B / "catalog_landsat.csv"))}
CODE = {"Sentinel-2A": "S2A", "Sentinel-2B": "S2B", "Sentinel-2C": "S2C"}
REJECT = {"1976-09-06 aircraft": "cloud and cloud shadow over the glacier on every image of the flight"}


def num(r, k):
    try:
        return float(r[k])
    except ValueError:
        return None


def rank(r):
    p = r["platform"]
    if p == "aircraft":
        return 0
    if p.startswith("Sentinel-2") or p.startswith("Sentinel 2"):
        return 2
    if p.startswith("Landsat"):
        return 3 if (num(r, "resolution_m") or 30) <= 30 else 5
    return 9


def usable(r):
    p = r["platform"]
    if r["fetched"] != "1" or (num(r, "nodata_glacier2024") or 0) > 0.02 or f"{r['date']} {p}" in REJECT:
        return False
    if p == "aircraft":
        return r["id"] in other and bool(other[r["id"]]["refl"])
    if (num(r, "cloud_glacier2024") or 0) > 0.05 or (num(r, "shadow_glacier2024") or 0) > 0.10:
        return False
    if p.startswith("Landsat"):
        shift = max(abs(num(r, "shift_east_m") or 0), abs(num(r, "shift_north_m") or 0))
        return r["geolocation"] == "tier 1" or (r["geolocation"] == "checked" and shift <= 60)
    return True


cand = {}
for r in rows:
    if "08-10" <= r["date"][5:] <= "10-10" and rank(r) < 9 and usable(r):
        cand.setdefault((int(r["year"]), rank(r)), []).append(r)
chosen = {}
for (y, k), rs in cand.items():
    if k == 0:
        best = min(rs, key=lambda r: (round(num(r, "nodata_extent1850") or num(r, "nodata_square") or 0, 2),
                                      -int(r["date"].replace("-", ""))))
    else:
        snow = [num(r, "snow_ice_square") for r in rs]
        if all(v is None for v in snow):
            best = max(rs, key=lambda r: r["date"])
        else:
            m = min(v for v in snow if v is not None)
            best = max((r for r in rs if (num(r, "snow_ice_square") or 9) <= m + 0.01), key=lambda r: r["date"])
    if y not in chosen or k < chosen[y][0]:
        chosen[y] = (k, best)

out = []
for y in sorted(chosen):
    k, r = chosen[y]
    p, rid, offset = r["platform"], r["id"], "0"
    if p == "aircraft":
        path, enc = "other/" + other[rid]["refl"], "display"
    elif k == 2:
        hits = [x for x in s2 if x["tag"].startswith(f"{r['date']}_{CODE[p]}") and x["file_10m"]]
        if hits:
            path = "sentinel2/" + hits[0]["file_10m"]
        else:
            path, offset = "other/" + other[rid]["refl"], "-0.1"      # swissEO copy of the same scene
        enc = "s2"
    else:
        path, enc = "landsat/" + landsat[rid]["reflectance_file"], "landsat"
    out.append(dict(year=y, date=r["date"], platform=p, resolution_m=r["resolution_m"], id=rid, path=path,
                    encoding=enc, sun=r["sun_elevation"], offset=offset))
    print(y, r["date"], f"{p:12s}", r["resolution_m"], path)
with open(B / "record_files.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
years = [o["year"] for o in out]
print(len(out), "years; missing since 1972:", [y for y in range(1972, 2027) if y not in years])
