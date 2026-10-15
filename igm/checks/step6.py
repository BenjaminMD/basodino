"""Checks of DESIGN.md step 6 on s03 beside the gate of projection.py. Fix round F6: file prefix P6 = f6 (stage
f6_flowoff, results/f6/noflow_check.csv from stages/f6_noflow_check.csv, results/f6_*.csv); the first ensemble used
step6 and s6_flowoff (git history).

  flowoff  scenario weather end to end: IGM without flow (experiment flowoff on CPU, stage P6_flowoff: members
           d000_c00_1 and d000_c00_5 from their 2026 start state, 2026.75-2032.75) against noflow.py on the same
           members (results/P6/noflow_check.csv, stages/P6_noflow_check.csv): largest |difference| of the
           area (thk >= 1 m) km2, volume km3 and Ba m w.e.; the same weather enters both only through
           projection.weather
  stage STAGE  runs, failures, makespan, process-hours, start-up (wall at the first save) and wall s per simulated
           year (median, max), peak GPU memory per run (logs/STAGE/gpu_mem.txt)
  rate STAGE   the Zn CPU campaign (Zn/mesh48; its ph.x share the threads 6-7, 70-71 with the IGM slots): rank-0 cpu
           seconds per wall second from logs/rate_sampler.log (the 'total cpu time' of the newest ph_<i>.out at each
           change), in the 30 min before the stage's first start and during the stage
Output: results/P6_flowoff.csv, P6_stage_STAGE.csv, P6_rate_STAGE.csv (check, quantity, value, threshold,
pass); printed.
usage (s03, env.sh):  python checks/step6.py flowoff | stage STAGE | rate STAGE
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step4 import memory, series, timing  # noqa: E402

IGM = Path(__file__).resolve().parents[1]
RES, LOGS = IGM / "results", IGM / "logs"
P6 = "f6"


def flowoff():
    nf = list(csv.DictReader(open(RES / P6 / "noflow_check.csv")))
    rows = []
    for m, name in enumerate(("d000_c00_1", "d000_c00_5")):
        ref = {r["date"]: r for r in nf if r["member"] == str(m)}
        da = dv = db = 0.0
        s = series(f"{P6}_flowoff", name)
        for r in s:
            n = ref[f"{int(r['year'])}-10-01"]
            da, dv = max(da, abs(float(r["A1_km2"]) - float(n["A1_km2"]))), max(dv, abs(float(r["V_km3"]) - float(n["V_km3"])))
            if r["Ba"] and n["Ba"] not in ("", "nan"):
                db = max(db, abs(float(r["Ba"]) - float(n["Ba"])))
        rows.append(("flowoff", f"{name}: IGM without flow vs noflow.py, {len(s) - 1} years: largest |difference| of "
                     "area km2, volume km3, Ba m w.e.", f"{da:.2e} {dv:.2e} {db:.2e}", "area 0.02 (gate 3d)", da <= 0.02))
    return rows, RES / f"{P6}_flowoff.csv"


def stage(st):
    runs = list(csv.DictReader(open(LOGS / st / "runs.csv")))
    span = max(int(r["end"]) for r in runs) - min(int(r["start"]) for r in runs)
    proc_h = sum(int(r["end"]) - int(r["start"]) for r in runs) / 3600
    bad = [r["name"] for r in runs if r["rc"] != "0" or not (IGM / "runs" / st / r["name"] / "track_end.txt").exists()]
    rows = [("stage", "runs, failed (exit code or no track_end.txt), makespan s, process-hours",
             f"{len(runs)} {len(bad)} {span} {proc_h:.2f}", "0 failed", not bad)]
    ok = [r["name"] for r in runs if r["name"] not in bad]
    tm = timing(st, ok)
    wall = np.array([int(r["end"]) - int(r["start"]) for r in runs if r["name"] in set(ok)])
    rows.append(("timing", "process wall s, start-up s (wall at the first save), wall s per simulated year: median, "
                 "max; simulated years median, max", f"{np.median(wall):.1f} {wall.max()}; {np.median(tm[:, 0]):.1f} "
                 f"{tm[:, 0].max():.1f}; {np.nanmedian(tm[:, 1]):.3f} {np.nanmax(tm[:, 1]):.3f}; "
                 f"{np.median(tm[:, 2]):.0f} {tm[:, 2].max():.0f}", "", ""))
    mem = memory(st)
    rows.append(("memory", "peak GPU memory per run MiB: median, max", f"{np.median(list(mem.values())):.0f} "
                 f"{max(mem.values())}", "3072", max(mem.values()) <= 3072))
    return rows, RES / f"{P6}_stage_{st}.csv"


def rate(st):
    runs = list(csv.DictReader(open(LOGS / st / "runs.csv")))
    t0, t1 = min(int(r["start"]) for r in runs), max(int(r["end"]) for r in runs)
    pts = []                                                     # (epoch, file, cpu s) at each change
    for line in open(LOGS / "rate_sampler.log"):
        part = [p for p in line.split("|") if p.strip().startswith("mesh48 ")]
        if not part or "time" not in part[0]:
            continue
        w = part[0].split()
        e, f, c = int(line.split()[0]), w[1], float(w[w.index("time") + 2])
        if not pts or (pts[-1][1], pts[-1][2]) != (f, c):
            pts.append((e, f, c))
    rows = []
    for label, a, b in (("30 min before the stage", t0 - 1800, t0), ("during the stage", t0, t1)):
        cpu = wall = 0.0
        for (e0, f0, c0), (e1, f1, c1) in zip(pts[:-1], pts[1:]):
            if a <= e0 and e1 <= b and f0 == f1 and c1 > c0:
                cpu, wall = cpu + c1 - c0, wall + e1 - e0
        rows.append(("rate", f"Zn mesh48 rank-0 cpu s per wall s, {label} ({wall / 60:.1f} min between changes)",
                     f"{cpu / wall:.3f}" if wall else "no change", "", ""))
    return rows, RES / f"{P6}_rate_{st}.csv"


def main():
    rows, out = {"flowoff": flowoff, "stage": lambda: stage(sys.argv[2]), "rate": lambda: rate(sys.argv[2])}[
        sys.argv[1]]()
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["check", "quantity", "value", "threshold", "pass"])
        w.writerows(rows)
    for r in rows:
        print(*r, sep=" | ")


if __name__ == "__main__":
    main()
