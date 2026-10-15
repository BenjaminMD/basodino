"""Height of the ice face at the margin of the film member, per model year (review of the fixed ensemble's films,
DESIGN.md F7 review): the basin process of F2 sets thk = 0 outside the posbal cells at every step, so thick hindcast
ice can end at the basin edge in a face; this measures it.

For each yearly 1 October state of results/f7/film_states.npz: margin cells = ice of at least 1 m with an ice-free
(below 0.5 m) 4-neighbour; face of a margin cell = its thickness minus max(0, neighbour bed - own bed) for its
ice-free neighbours (the ice standing above the neighbour's ground), the largest over them; a face counts as 'at the
basin edge' when that neighbour lies outside posbal (inputs/igm_static.npz). Cell edge 10 m.
usage (s02, agl env, from igm/):  python checks/basin_wall.py   -> results/f7_basin_wall.csv
"""
import csv
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
Z = np.load(IGM / "results/f7/film_states.npz")
outside = ~(np.load(IGM / "inputs/igm_static.npz")["posbal"] > 0)
bed = Z["topg"]
rows = []
for i, year in enumerate(Z["years"]):
    t = Z["thk"][i]
    free, ice = t < 0.5, t >= 1.0
    face_edge, face_in = np.zeros_like(t), np.zeros_like(t)
    for shift in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nf = np.roll(free, shift, (0, 1))
        f = np.where(nf & ice, t - np.maximum(0, np.roll(bed, shift, (0, 1)) - bed), 0)
        no = np.roll(outside, shift, (0, 1))
        face_edge = np.maximum(face_edge, np.where(no, f, 0))
        face_in = np.maximum(face_in, np.where(no, 0, f))
    face = np.maximum(face_edge, face_in)
    rows.append([int(year), int((face > 0).sum()), int((face > 20).sum()), int((face_edge > 20).sum()),
                 f"{face_edge.max():.1f}", f"{face_in.max():.1f}", f"{(face_edge > 20).sum() * 0.01:.2f}"])
with open(IGM / "results/f7_basin_wall.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["year", "margin_cells", "faces_over_20m", "faces_over_20m_at_basin_edge", "largest_face_at_basin_edge_m",
                "largest_face_elsewhere_m", "basin_edge_with_face_over_20m_km"])
    w.writerows(rows)
for r in rows[::10] + rows[-22:-16]:
    print(*r)
