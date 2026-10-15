"""Before the s03 run folders are deleted (DESIGN.md step 7): is every file of every s03 run folder in one of the
banked tars on s02, with the same size?

Input: a listing of the s03 run folders, one line per file 'runs/STAGE/NAME/... SIZE' (s03:
find runs -type f -printf '%p %s\\n'), read from stdin; the tars igm/runs_*.tar.gz on s02 (listed with tarfile).
Output: results/step7_bank.csv, per run folder (runs/STAGE): files, files banked with the same size, missing or
different, the tars that hold them.
usage (s02, agl env, from igm/):  ssh s03 'cd ~/dev/AGL/basodino/igm && find runs -type f -printf "%p %s\\n"' |
                                    python checks/step7_bank.py
"""
import csv
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

IGM = Path(__file__).resolve().parents[1]
have = {}
for tar in sorted(IGM.glob("runs_*.tar.gz")):
    with tarfile.open(tar, "r:gz") as t:
        for m in t:
            if m.isfile():
                have[m.name] = (m.size, tar.name)
rows = defaultdict(lambda: [0, 0, [], set()])
for line in sys.stdin:
    path, size = line.rsplit(" ", 1)
    stage = "/".join(path.split("/")[:2])
    r = rows[stage]
    r[0] += 1
    h = have.get(path)
    if h and h[0] == int(size):
        r[1] += 1
        r[3].add(h[1])
    else:
        r[2].append(path)
out = [[s, r[0], r[1], len(r[2]), " ".join(r[2][:3]), " ".join(sorted(r[3]))] for s, r in sorted(rows.items())]
with open(IGM / "results/step7_bank.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["folder", "files", "banked_same_size", "missing_or_different", "first_missing", "tars"])
    w.writerows(out)
for r in out:
    print(" | ".join(map(str, r)))
