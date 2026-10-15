"""Order PlanetScope scenes (8-band surface reflectance with the usable-data mask, bundle analytic_sr_udm2), clipped to
the Basodino box E 2678500-2682500, N 1139900-1142400 (10 km2), and download them. Every order is appended to
data/planet/orders.csv (time, ids, clipped km2, order id, state). A hard budget refuses any order that would take the
logged total of the calendar month above BUDGET_KM2. The API key comes from the macOS Keychain (service planet-api).
usage:  uv run --with requests --with pyproj --with shapely python code/planet_order.py ID [ID ...]
outputs: data/planet/<order name>/ (GeoTIFFs and metadata as delivered)
"""
import csv
import datetime as dt
import subprocess
import sys
import time

import requests
from pyproj import Transformer
from shapely.geometry import box, mapping

from common import ROOT

BUDGET_KM2 = 1200.0                      # per calendar month, of the 3000 km2 Education and Research quota
E0, N0, E1, N1 = 2678500, 1139900, 2682500, 1142400
AREA_KM2 = (E1 - E0) * (N1 - N0) / 1e6
LOG = ROOT / "data/planet/orders.csv"

ids = sys.argv[1:]
assert ids, "give scene ids"
month = dt.date.today().strftime("%Y-%m")
used = 0.0
if LOG.exists():
    used = sum(float(r["km2"]) for r in csv.DictReader(open(LOG)) if r["time"].startswith(month))
cost = AREA_KM2 * len(ids)
if used + cost > BUDGET_KM2:
    sys.exit(f"refused: {used:.0f} + {cost:.0f} km2 would exceed the monthly budget of {BUDGET_KM2:.0f} km2")

KEY = subprocess.run(["security", "find-generic-password", "-s", "planet-api", "-w"], capture_output=True,
                     text=True, check=True).stdout.strip()
tw = Transformer.from_crs(2056, 4326, always_xy=True)
cs = [tw.transform(e, n) for e, n in ((E0, N0), (E1, N0), (E1, N1), (E0, N1))]
aoi = box(min(c[0] for c in cs), min(c[1] for c in cs), max(c[0] for c in cs), max(c[1] for c in cs))
name = f"basodino_{dt.datetime.now():%Y%m%d_%H%M%S}"
s = requests.Session()
s.auth = (KEY, "")
req = {"name": name, "products": [{"item_ids": ids, "item_type": "PSScene", "product_bundle": "analytic_sr_udm2"}],
       "tools": [{"clip": {"aoi": mapping(aoi)}}]}
r = s.post("https://api.planet.com/compute/ops/orders/v2", json=req, timeout=60)
r.raise_for_status()
oid = r.json()["id"]
print(f"order {oid}: {len(ids)} scene(s), {cost:.1f} km2 clipped; month total {used + cost:.1f} of {BUDGET_KM2:.0f}")
new = not LOG.exists()
with open(LOG, "a", newline="") as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(("time", "order_id", "name", "ids", "km2", "state"))
    w.writerow((dt.datetime.now().isoformat(timespec="seconds"), oid, name, " ".join(ids), f"{cost:.2f}", "placed"))
while True:
    st = s.get(f"https://api.planet.com/compute/ops/orders/v2/{oid}", timeout=60).json()
    if st["state"] in ("success", "failed", "partial", "cancelled"):
        break
    time.sleep(20)
print("state", st["state"])
out = ROOT / "data/planet" / name
for res in st.get("_links", {}).get("results", []):
    f = out / res["name"]
    f.parent.mkdir(parents=True, exist_ok=True)
    with s.get(res["location"], stream=True, timeout=300) as g, open(f, "wb") as fh:
        for chunk in g.iter_content(1 << 20):
            fh.write(chunk)
    print("got", f.relative_to(ROOT), f.stat().st_size // 1024, "kB")
