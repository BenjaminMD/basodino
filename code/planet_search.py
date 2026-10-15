"""Free metadata search of PlanetScope scenes over Ghiacciaio del Basodino (Planet Data API quick-search; lists scenes,
downloads nothing, uses no quota). Area of interest: the glacier box E 2678500-2682500, N 1139900-1142400 (Swiss
coordinates, 10 km2), converted to WGS84. Windows: May 2026, the melt season 2026 (1 June to 10 September), its end (11 September
to 15 October) and the end of each season 2018-2025 (15 August to 10 October). Keeps scenes that cover the whole box with at least 80 per cent clear pixels.
The API key is read from the macOS Keychain (service planet-api), never written anywhere.
Output: data/planet/search.csv (one row per scene: id, time, window, satellite, clear percent, cloud cover, sun
elevation, whether our account may download the analytic asset) and a summary on stdout with the download area if each
kept scene is clipped to the 10 km2 box.
usage:  uv run --with requests --with pyproj --with shapely python code/planet_search.py
"""
import csv
import subprocess

import requests
from pyproj import Transformer
from shapely.geometry import box, mapping, shape

from common import ROOT

KEY = subprocess.run(["security", "find-generic-password", "-s", "planet-api", "-w"], capture_output=True,
                     text=True, check=True).stdout.strip()
to_wgs = Transformer.from_crs(2056, 4326, always_xy=True)
E0, N0, E1, N1 = 2678500, 1139900, 2682500, 1142400
corners = [to_wgs.transform(e, n) for e, n in ((E0, N0), (E1, N0), (E1, N1), (E0, N1))]
aoi = box(min(c[0] for c in corners), min(c[1] for c in corners), max(c[0] for c in corners), max(c[1] for c in corners))
AREA_KM2 = (E1 - E0) * (N1 - N0) / 1e6
WINDOWS = [("spring_2026", "2026-05-01", "2026-05-31"), ("melt_2026", "2026-06-01", "2026-09-10"),
           ("end_2026", "2026-09-11", "2026-10-15")] + [(f"end_{y}", f"{y}-08-15", f"{y}-10-10") for y in range(2018, 2026)]

s = requests.Session()
s.auth = (KEY, "")
rows = []
for name, t0, t1 in WINDOWS:
    flt = {"type": "AndFilter", "config": [
        {"type": "GeometryFilter", "field_name": "geometry", "config": mapping(aoi)},
        {"type": "DateRangeFilter", "field_name": "acquired", "config": {"gte": f"{t0}T00:00:00Z", "lte": f"{t1}T23:59:59Z"}},
        {"type": "RangeFilter", "field_name": "clear_percent", "config": {"gte": 80}}]}
    r = s.post("https://api.planet.com/data/v1/quick-search", json={"item_types": ["PSScene"], "filter": flt}, timeout=60)
    r.raise_for_status()
    page = r.json()
    while True:
        for f in page["features"]:
            p = f["properties"]
            if not shape(f["geometry"]).contains(aoi):
                continue
            rows.append((f["id"], p["acquired"], name, p.get("instrument", ""), p.get("clear_percent"),
                         p.get("cloud_cover"), p.get("sun_elevation"),
                         any("assets.ortho_analytic_8b_sr:download" in x or "assets.ortho_analytic_4b_sr:download" in x
                             for x in f.get("_permissions", []))))
        nxt = page["_links"].get("_next")
        if not nxt:
            break
        page = s.get(nxt, timeout=60).json()

out = ROOT / "data/planet/search.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(("id", "acquired", "window", "instrument", "clear_percent", "cloud_cover", "sun_elevation", "downloadable"))
    w.writerows(sorted(rows, key=lambda x: x[1]))
print(f"box {AREA_KM2:.1f} km2; scenes covering it with >= 80 % clear:")
for name, _, _ in WINDOWS:
    sel = [x for x in rows if x[2] == name]
    days = len({x[1][:10] for x in sel})
    print(f"  {name:10s} {len(sel):4d} scenes on {days:3d} days, downloadable {sum(x[7] for x in sel):4d}")
