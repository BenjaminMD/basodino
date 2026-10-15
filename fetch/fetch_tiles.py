"""Download the latest 1 km tile per location from a STAC items.json (see stac_list.py).
usage: fetch_tiles.py <raw/collection dir> <res> [<res> ...]   e.g. raw/swissalti3d 0.5 2
Keeps tiles with LV95 lower-left km E 2676..2685, N 1137..1146 (bbox 2676000-2686000 / 1137000-1147000).
Verifies each file against the STAC sha2-256 multihash. Writes <dir>/<res>/<file> and <dir>/tiles.csv."""
import csv, hashlib, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import requests

E0, E1, N0, N1 = 2676, 2685, 1137, 1146

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

def fetch(url, path, want):
    if os.path.exists(path) and sha256(path) == want:
        return "cached"
    for attempt in range(5):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(path + ".part", "wb") as f:
                    for b in r.iter_content(1 << 22):
                        f.write(b)
            if sha256(path + ".part") != want:
                raise IOError("checksum mismatch")
            os.replace(path + ".part", path)
            return "ok"
        except Exception as e:
            print("retry", attempt, url, e, flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("failed " + url)

def main(d, resolutions):
    feats = json.load(open(os.path.join(d, "items.json")))
    tiles = {}
    for f in feats:
        yr, e, n = map(int, re.search(r"_(\d{4})_(\d{4})-(\d{4})$", f["id"]).groups())
        if E0 <= e <= E1 and N0 <= n <= N1:
            tiles.setdefault((e, n), []).append((yr, f))
    jobs, rows = [], []
    for (e, n), versions in sorted(tiles.items()):
        yr, f = max(versions, key=lambda v: v[0])
        row = dict(tile=f"{e}-{n}", E_min=e * 1000, N_min=n * 1000, item=f["id"], year=yr,
                   stac_datetime=f["properties"]["datetime"],
                   years_available=" ".join(str(y) for y, _ in sorted(versions, key=lambda v: v[0])))
        for key, a in f["assets"].items():
            m = re.search(r"_(0\.1|0\.5|2)_2056(_5728)?\.tif$", key)
            if m and m.group(1) in resolutions:
                res = m.group(1)
                os.makedirs(os.path.join(d, res), exist_ok=True)
                path = os.path.join(d, res, key)
                jobs.append((a["href"], path, a["checksum:multihash"][4:].lower()))
                row[f"file_{res}"] = os.path.relpath(path, d)
        rows.append(row)
    print(len(rows), "tiles,", len(jobs), "files", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for i, st in enumerate(ex.map(lambda j: fetch(*j), jobs)):
            if i % 20 == 0:
                print(i + 1, "/", len(jobs), st, flush=True)
    for row in rows:
        for res in resolutions:
            p = os.path.join(d, row.get(f"file_{res}", "missing"))
            row[f"bytes_{res}"] = os.path.getsize(p) if os.path.exists(p) else 0
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(d, "tiles.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print("done", sum(r[f"bytes_{res}"] for r in rows for res in resolutions) / 1e9, "GB", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
