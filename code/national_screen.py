"""National screen: the ice each Swiss glacier lost since the year of its thickness map, over the volume the map gives.

Inputs (provenance and licences in data/national/README.md):
  data/national/01_Summary.zip    Swiss Glacier Thickness, release 2020 (Grab and fifteen co-authors 2021), table
                                  01_Summary/Summary.dbf: per glacier of the Swiss Glacier Inventory 2016 (pk_sgi,
                                  written A10g/02) its area A_sgi2016 (km2), the mapped volume V_year (km3) and its
                                  reference year year_mode (most common year of the swissALTI3D surface the thickness
                                  refers to)
  data/national/volumechange.csv  GLAMOS volume change, release 2025, every pair of elevation models of every glacier
                                  (SGI-ID, written A10g-02): sources, dates (yyyymmdd, 9999 = day unknown), glacier areas
                                  at start and end (km2), dV (km3), Bgeod and its 1 sigma (both m w.e. a-1, 850 kg m-3)
Method:
  1. Start: the GLAMOS survey in the map's reference year (at most one per glacier, asserted). End: the latest survey
     of a pair starting there (unique in its year, asserted). A glacier without such a survey keeps empty columns.
  2. loss_over_map = -dV / V_year, the loss from the reference year to the latest survey (2019 to 2024; later melt is
     not in it). Its sigma = |loss_over_map| x sigma / |Bgeod|: GLAMOS gives the error of the rate, dV carries the same
     relative error. The map's own error is left out; it is what the screen tests.
  3. Area left: GLAMOS's glacier area at the end of the pair (the outline closest to that survey).
  dV covers the glacier of the start year, V_year only the outline of 2016 (imagery 2013 to 2018): ice outside that
  outline counts as loss but not as mapped, so the ratio leans high where the glacier shrank in between
  (area_start_km2 against area_map_km2; Basòdino 1.705 against 1.641 km2). GLAMOS's outlines can also be split
  differently from the inventory's (Zapportgletscher A13n-04: 1.311 km2 mapped, 0.205 left; its eastern part A13n-03:
  0.025 mapped, 0.590 left, loss_over_map 23).
Outputs:
  code/national_screen.csv  one row per glacier of the summary, largest loss_over_map first, glaciers without a pair last
                            (by area): sgi_id, name, area_map_km2, volume_map_km3, year_map, survey_start, date_start,
                            survey_end, date_end, area_start_km2, area_left_km2, dV_km3, loss_over_map,
                            loss_over_map_sigma, B_geodetic_mwe_per_a, sigma_geodetic_mwe_per_a
  stdout                    counts; the glaciers above 0.5 km2 with loss_over_map above 0.8, and Basòdino
usage:  python3 code/national_screen.py
"""
import csv
import struct
import zipfile

from common import ROOT, write_csv

NATIONAL = ROOT / "data/national"
HEADER = ("sgi_id", "name", "area_map_km2", "volume_map_km3", "year_map", "survey_start", "date_start", "survey_end",
          "date_end", "area_start_km2", "area_left_km2", "dV_km3", "loss_over_map", "loss_over_map_sigma",
          "B_geodetic_mwe_per_a", "sigma_geodetic_mwe_per_a")


def read_summary():
    """Summary.dbf (dBASE III: 32-byte header, 32-byte field descriptors, fixed-width ASCII records) -> {id: record}."""
    with zipfile.ZipFile(NATIONAL / "01_Summary.zip") as z:
        b = z.read("01_Summary/Summary.dbf")
    n, start, size = struct.unpack("<IHH", b[4:12])
    fields = [(b[i:i + 11].split(b"\0")[0].decode(), b[i + 11:i + 12], b[i + 16]) for i in range(32, start - 1, 32)]
    out = {}
    for k in range(n):
        pos = start + k * size
        assert b[pos:pos + 1] == b" ", "deleted record"
        rec, pos = {}, pos + 1
        for name, kind, width in fields:
            v = b[pos:pos + width].decode("ascii").strip()
            rec[name], pos = (float(v) if kind == b"N" else v), pos + width
        out[rec["pk_sgi"].replace("/", "-")] = rec
    return out


def read_pairs():
    """volumechange.csv -> {SGI-ID: [pair, ...]}, pairs as dicts; above the header: citation and title, below: units."""
    with open(NATIONAL / "volumechange.csv", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    i = next(i for i, r in enumerate(rows) if r[:1] == ["SGI-ID"])
    out = {}
    for r in rows[i + 2:]:
        p = dict(zip(rows[i], r))
        out.setdefault(p["SGI-ID"], []).append(p)
    return out


summary, pairs = read_summary(), read_pairs()
rows = []
for gid, s in summary.items():
    year = int(s["year_mode"])                                  # -9999 on one glacier: no reference year
    own = [p for p in pairs.get(gid, []) if p["date_start"][:4] == str(year)]
    assert len({(p["source_start"], p["date_start"]) for p in own}) <= 1, f"{gid}: two surveys in {year}"
    row = [gid, pairs[gid][0]["Name"].strip() if gid in pairs else "", s["A_sgi2016"], s["V_year"],
           year if year > 0 else ""]
    if own:
        p = max(own, key=lambda p: p["date_end"])
        assert len({q["date_end"] for q in own if q["date_end"][:4] == p["date_end"][:4]}) == 1, f"{gid}: end"
        dv, b, sigma = float(p["dV"]), float(p["Bgeod"]), float(p["sigma"])
        ratio = -dv / s["V_year"]
        row += [p["source_start"], p["date_start"], p["source_end"], p["date_end"], float(p["A_start"]),
                float(p["A_end"]), dv, ratio, abs(ratio) * sigma / abs(b) if b else "", b, sigma]
    else:
        row += [""] * 11
    rows.append(row)

RATIO, AREA = HEADER.index("loss_over_map"), HEADER.index("area_map_km2")
rows.sort(key=lambda r: (r[RATIO] == "", -(r[RATIO] or 0), -r[AREA]))
write_csv("national_screen.csv", HEADER, rows)

big = [r for r in rows if r[AREA] > 0.5]
screened = [r for r in big if r[RATIO] != ""]
print(f"{len(rows)} glaciers, {sum(r[RATIO] != '' for r in rows)} with a pair from the map's reference year; "
      f"above 0.5 km2: {len(big)}, {len(screened)} with a pair")
show = {r[0] for r in screened if r[RATIO] > 0.8} | {"C14-10"}
for r in [r for r in rows if r[0] in show]:
    print(f"  {r[0]} {r[1]}: {r[5]} {r[6]} to {r[7]} {r[8]}, dV {r[11]:.6f} km3, map {r[3]:.6f} km3 ({r[4]}), "
          f"loss over map {r[RATIO]:.3f} +- {r[RATIO + 1]:.3f}, area {r[AREA]:.4f} km2 mapped, {r[10]:.4f} left")
