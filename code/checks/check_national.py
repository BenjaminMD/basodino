"""Independent check of step 4 (national_screen.py): the Basòdino row and the glaciers above 0.5 km2 that lost more
than 0.8 of their mapped volume, from the two downloads, without reading national_screen.csv.

Summary.dbf is read by field name from its descriptor table; a glacier's loss is -dV of the GLAMOS pair that starts
in the map's reference year and ends latest, over V_year; sigma = loss x sigma / |Bgeod|. The count of glaciers
above 0.5 km2 is given with and without pairs shorter than 6 years (the pre-check said 230).

Inputs: data/national/01_Summary.zip, data/national/volumechange.csv
Output: code/checks/check_national.csv (the glaciers above 0.5 km2 with a ratio above 0.8, and Basòdino)
usage:  python code/checks/check_national.py
"""
import csv
import struct
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data/national"
b = zipfile.ZipFile(ROOT / "01_Summary.zip").read("01_Summary/Summary.dbf")
n, hlen, rlen = struct.unpack("<IHH", b[4:12])
fields, off = [], 1
for i in range(32, hlen - 1, 32):
    fields.append((b[i:i + 11].rstrip(b"\0").decode(), off, b[i + 16]))
    off += b[i + 16]
recs = []
for k in range(n):
    r = b[hlen + k * rlen: hlen + (k + 1) * rlen]
    recs.append({name: r[o:o + w].decode().strip() for name, o, w in fields})

rows = list(csv.reader(open(ROOT / "volumechange.csv", encoding="utf-8")))
h = next(i for i, r in enumerate(rows) if r and r[0] == "SGI-ID")
pairs = {}
for r in rows[h + 2:]:
    pairs.setdefault(r[0], []).append(dict(zip(rows[h], r)))

out, big, big6 = [("sgi_id", "area_km2", "V_year_km3", "year", "end", "dV_km3", "loss_over_map", "sigma")], 0, 0
for s in recs:
    gid, a, v, y = s["pk_sgi"].replace("/", "-"), float(s["A_sgi2016"]), float(s["V_year"]), s["year_mode"]
    own = [p for p in pairs.get(gid, []) if p["date_start"][:4] == y]
    if not own:
        continue
    p = max(own, key=lambda p: p["date_end"])
    ratio = -float(p["dV"]) / v
    sig = abs(ratio) * float(p["sigma"]) / abs(float(p["Bgeod"]))
    if a > 0.5:
        big += 1
        big6 += int(p["date_end"][:4]) - int(y) >= 6
    if (a > 0.5 and ratio > 0.8) or gid == "C14-10":
        out.append((gid, a, v, y, p["date_end"], p["dV"], round(ratio, 4), round(sig, 4)))
out.append(("count above 0.5 km2 with a pair", big, "", "", "", "", "", ""))
out.append(("the same, pairs of at least 6 years", big6, "", "", "", "", "", ""))
for r in out:
    print(*r, sep="\t")
csv.writer(open(Path(__file__).with_suffix(".csv"), "w", newline="")).writerows(out)
