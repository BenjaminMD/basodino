"""Frame-to-frame steps of the two films of the fixed ensemble (DESIGN.md step 7, fix round; the first ensemble's films,
tex/animation/first_ensemble/, gave results/first_ensemble/step7_film_steps.csv): each film decoded at 480 x 270, the
mean absolute luma difference between consecutive frames (ffmpeg tblend=difference + signalstats YAVG, 8-bit grey
levels); per film the frame count, the median step, the frames with no change (step < 0.02: holds), and the eight
largest steps with the frame they lead into.
usage (Mac, from basodino/):  python igm/checks/film_steps.py   -> igm/results/f7_film_steps.csv
"""
import csv
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FILMS = ("igm_map_1850_2042.mp4", "igm_3d_1850_2042.mp4")
OUT = ROOT / "igm/results/f7_film_steps.csv"


def steps(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                        "scale=480:270,tblend=all_mode=difference,signalstats,"
                        "metadata=print:key=lavfi.signalstats.YAVG:file=-", "-f", "null", "-"],
                       capture_output=True, text=True, check=True)
    return np.array([float(l.split("=")[1]) for l in r.stdout.splitlines() if "YAVG" in l])


rows = []
for name in FILMS:
    if not (ROOT / "tex/animation" / name).exists():
        continue
    v = steps(ROOT / "tex/animation" / name)
    top = np.argsort(v)[::-1][:8]
    rows.append([name, len(v) + 1, f"{np.median(v):.3f}", int((v < 0.02).sum()),
                 "; ".join(f"{v[i]:.2f} into frame {i + 1}" for i in top)])
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["film", "frames", "median_step_grey_levels", "frames_without_change", "largest_steps"])
    w.writerows(rows)
for r in rows:
    print(*r, sep=" | ")
