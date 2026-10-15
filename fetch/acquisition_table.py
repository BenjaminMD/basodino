"""Per-tile acquisition table -> derived/tiles_acquisition.csv.
Joins the downloaded tiles (raw/*/tiles.csv) with geo.admin.ch metadata layers:
  swissSURFACE3D LiDAR division (point-cloud acquisition years, the source of swissALTI3D below ~2000 m),
  SWISSIMAGE 10 cm division (data state year, native ground sampling distance),
  aerial image strips (flight dates of the SWISSIMAGE strips of the tile's year that overlap the tile)."""
import csv, requests
from shapely.geometry import box, shape

API = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
BB = "2676000,1137000,2686000,1147000"

def identify(layer, geometry=False, **extra):
    out, off = [], 0
    while True:
        p = dict(geometryType="esriGeometryEnvelope", geometry=BB, imageDisplay="1000,1000,96", mapExtent=BB,
                 tolerance=0, sr=2056, layers=f"all:{layer}", returnGeometry=str(geometry).lower(),
                 geometryFormat="geojson", offset=off, limit=50, **extra)
        r = requests.get(API, params=p, timeout=60).json()["results"]
        out += r
        if len(r) < 50:
            return out
        off += 50

def by_label(results):  # "2679_1140" -> attributes
    return {f["properties"]["label"].replace("_", "-"): f["properties"] for f in results}

dem = list(csv.DictReader(open("raw/swissalti3d/tiles.csv")))
img = {r["tile"]: r for r in csv.DictReader(open("raw/swissimage-dop10/tiles.csv"))}
lidar = by_label(identify("ch.swisstopo.swisssurface3d.metadata"))
dop = by_label(identify("ch.swisstopo.images-swissimage-dop10.metadata"))
strips = {}
for year in sorted({r["year"] for r in img.values()}):
    strips[year] = [(f["properties"]["flugdatum"], shape(f["geometry"]))
                    for f in identify("ch.swisstopo.lubis-bildstreifen", geometry=True, timeInstant=year)
                    if f["properties"]["goal"] == "SWISSIMAGE"]

rows = []
for d in dem:
    t, o = d["tile"], img[d["tile"]]
    tb = box(int(d["E_min"]), int(d["N_min"]), int(d["E_min"]) + 1000, int(d["N_min"]) + 1000)
    dates = sorted({date for date, g in strips[o["year"]] if g.intersection(tb).area > 0})
    li = lidar.get(t, {})
    rows.append(dict(
        tile=t, E_min=d["E_min"], N_min=d["N_min"],
        swissalti3d_item=d["item"], swissalti3d_year=d["year"], swissalti3d_years_available=d["years_available"],
        swisssurface3d_lidar_years=f"{li.get('gpstime_min', '')}-{li.get('gpstime_max', '')}" if li else "",
        swissalti3d_0p5m_file="raw/swissalti3d/" + d["file_0.5"], swissalti3d_0p5m_bytes=d["bytes_0.5"],
        swissalti3d_2m_file="raw/swissalti3d/" + d["file_2"], swissalti3d_2m_bytes=d["bytes_2"],
        swissimage_item=o["item"], swissimage_year=o["year"], swissimage_years_available=o["years_available"],
        swissimage_flight_dates=" ".join(dates), swissimage_source_gsd=dop.get(t, {}).get("resolution", ""),
        swissimage_0p1m_file="raw/swissimage-dop10/" + o["file_0.1"], swissimage_0p1m_bytes=o["bytes_0.1"],
        swissimage_2m_file="raw/swissimage-dop10/" + o["file_2"], swissimage_2m_bytes=o["bytes_2"]))
with open("derived/tiles_acquisition.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "tiles")
