"""From which DROP (step 5 of outline_2026.py) the two strips of 27 September join, and the outline at each DROP.
A check for the open decision on DROP, not part of the report chain.

The method of outline_2026.py runs here unchanged (its code above '# --- the outline and its checks'); DROP is set
before each call of outline(), and the five late-September scenes vote as in its step 6. Per DROP (0.10, 0.11, 0.120
to 0.160 in steps of 0.001, 0.18, 0.20): the 27 September area and main body, the joined pixels of 27 September,
whether the place-3 strip closes the rock bay south of it (bay point E 2680336, N 1141003 inside a hole of the body),
whether the gully strip joins the gully top to the body (E 2679775, N 1141360 inside the body), and the spread of the
five late-September areas. Test points as in outline_2026_review.py.

Output: code/outline_2026_drop.csv (m2, pixels); stdout the same.

usage, after outline_2026.py, in the same mirror (its docstring):
  rsync -a code/outline_2026_drop.py s02:/data/befahl/projects/AGL/basodino/report/code/
  ssh s02 'source ~/miniconda3/bin/activate agl && cd /data/befahl/projects/AGL/basodino/report &&
           ~/.local/bin/nrun python code/outline_2026_drop.py'
  then copy code/outline_2026_drop.csv back
"""
from pathlib import Path

exec((Path(__file__).resolve().parent / "outline_2026.py").read_text(encoding="utf-8")
     .split("# --- the outline and its checks")[0])                   # noqa: S102 (the method, unchanged)

VOTERS = [MAIN] + LATE
BAY, GULLY = shapely.Point(2680336, 1141003), shapely.Point(2679775, 1141360)
rows = []
for drop in [0.10, 0.11] + [round(0.12 + 0.001 * i, 3) for i in range(41)] + [0.18, 0.20]:
    DROP = drop                                                      # read by outline() at call time
    r = {t: outline(t) for t in VOTERS}
    ice = sum((r[t]["z"] > 0) & r[t]["sunlit"] for t in VOTERS)
    rock = sum((r[t]["z"] <= 0) & r[t]["sunlit"] for t in VOTERS)
    late = [glacier(r[t], ice, rock) for t in VOTERS]
    body = late[0].geoms[0]
    rows.append([drop, round(late[0].area), round(body.area), int(r[MAIN]["joined"].sum()),
                 "yes" if any(shapely.Polygon(h).contains(BAY) for h in body.interiors) else "no",
                 "yes" if body.contains(GULLY) else "no",
                 round(max(g.area for g in late) - min(g.area for g in late))])
    print(*rows[-1], sep="  ", flush=True)
write_csv("outline_2026_drop.csv", ["drop", "area_m2", "area_main_body_m2", "joined_pixels_27_september",
                                    "place3_bay_closed", "gully_top_joined", "late_september_spread_m2"], rows)
