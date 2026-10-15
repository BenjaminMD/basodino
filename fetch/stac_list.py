"""List STAC items of a data.geo.admin.ch collection intersecting the Basodino bbox.
Writes raw/<collection>/items.json (all items returned, all years). Stdlib only."""
import json, sys, urllib.request, os

API = "https://data.geo.admin.ch/api/stac/v0.9"
BBOX_WGS84 = (8.42651, 46.38007, 8.55833, 46.46881)
PAD = 0.003  # deg; tiles are filtered by LV95 id afterwards

def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)

def items(coll):
    b = BBOX_WGS84
    url = (f"{API}/collections/{coll}/items?limit=100&bbox="
           f"{b[0]-PAD},{b[1]-PAD},{b[2]+PAD},{b[3]+PAD}")
    out = []
    while url:
        page = get(url)
        out += page["features"]
        url = next((l["href"] for l in page.get("links", []) if l["rel"] == "next"), None)
    return out

if __name__ == "__main__":
    coll, outdir = sys.argv[1], sys.argv[2]
    feats = items(coll)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "items.json"), "w") as f:
        json.dump(feats, f, indent=1)
    print(coll, len(feats), "items")
