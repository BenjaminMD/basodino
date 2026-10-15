"""Checks of DESIGN.md step 4 on s03 from the runs of run_stage.sh.

  pilot  stages s4_pilot, s4_pilot2 (stages/step4_pilot*.txt): surface speed median and p90 over the ice cells (thk >= 1
         m) of the start state, online cold (3000 iterations; 6000 and 10000 in s4_pilot2) against the reference L-BFGS
         solve at the corners of the step-4 prior (u -1, 2; f 0.05, 4) on both DEMs, within 10 % as gate 3b; timing of the full segments at the slowest
         and fastest corner (start-up = wall at the first save, wall s per simulated year after it); peak GPU memory
  stage STAGE  runs, failures, makespan, process-hours, start-up and s per year (median, max) per segment, peak GPU
         memory (logs/STAGE/gpu_mem.txt)
Output: results/step4_pilot.csv or results/step4_stage_STAGE.csv (check, quantity, value, threshold, pass); printed.
usage (s03, env.sh):  python checks/step4.py pilot | stage STAGE
"""
import csv
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
RUNS, RES, LOGS = IGM / "runs", IGM / "results", IGM / "logs"


def series(stage, run):
    return list(csv.DictReader(open(RUNS / stage / run / "series.csv")))


def memory(stage):
    pid = {p: n for p, n in (l.split() for l in open(LOGS / stage / "pids.txt"))}
    mem = {}
    for line in open(LOGS / stage / "gpu_mem.txt"):
        _, p, mib = line.split()
        mem[pid[p]] = max(mem.get(pid[p], 0), int(mib))
    return mem


def timing(stage, names):
    out = []
    for n in names:
        s = series(stage, n)
        years = float(s[-1]["t"]) - float(s[0]["t"])
        start = float(s[0]["wall_s"])
        out.append((start, (float(s[-1]["wall_s"]) - start) / years if years > 0 else np.nan, len(s) - 1))
    return np.array(out)


def pilot():
    st, rows = "s4_pilot", []
    runs = [r["name"] for r in csv.DictReader(open(LOGS / st / "runs.csv"))]
    extra = ([("s4_pilot2", r["name"]) for r in csv.DictReader(open(LOGS / "s4_pilot2/runs.csv"))]
             if (LOGS / "s4_pilot2/runs.csv").exists() else [])
    for sc, c in sorted([(st, n) for n in runs if n.startswith("cold_")]) + sorted(extra):
        corner = c.split("_", 1)[1]
        a, b = series(sc, c)[0], series(st, "ref_" + corner)[0]
        va, vb = [float(a[k]) for k in ("v_med", "v_p90")], [float(b[k]) for k in ("v_med", "v_p90")]
        d = [va[i] / vb[i] - 1 if vb[i] > 0 else np.inf for i in (0, 1)]
        tiny = vb[1] < 0.1
        rows.append(("solver", f"{c}: median, p90 m a-1, online | reference; relative differences; absolute p90 "
                     "difference" + (" (reference p90 below 0.1 m a-1: relative difference not meaningful)" if tiny
                                     else ""),
                     f"{va[0]:.3f} {va[1]:.3f} | {vb[0]:.3f} {vb[1]:.3f}; {d[0]:+.3f} {d[1]:+.3f}; "
                     f"{va[1] - vb[1]:+.3f}", "10 %", "" if tiny else max(abs(x) for x in d) <= 0.10))
    for n in sorted(n for n in runs if n.startswith("full")):
        (start, rate, years), = timing(st, [n])
        s = series(st, n)
        rows.append(("timing", f"{n}: start-up s, wall s per simulated year, years; v_max at the start m a-1",
                     f"{start:.1f} {rate:.3f} {years}; {s[0]['v_max']}", "", ""))
    mem = memory(st)
    rows.append(("memory", "peak GPU memory per run MiB: median, max", f"{np.median(list(mem.values())):.0f} "
                 f"{max(mem.values())}", "3072", max(mem.values()) <= 3072))
    return rows, RES / "step4_pilot.csv"


def stage(st):
    rows = []
    runs = list(csv.DictReader(open(LOGS / st / "runs.csv")))
    span = max(int(r["end"]) for r in runs) - min(int(r["start"]) for r in runs)
    proc_h = sum(int(r["end"]) - int(r["start"]) for r in runs) / 3600
    bad = [r["name"] for r in runs if r["rc"] != "0" or not (RUNS / st / r["name"] / "track_end.txt").exists()]
    rows.append(("stage", "runs, failed (exit code or no track_end.txt), makespan s, process-hours",
                 f"{len(runs)} {len(bad)} {span} {proc_h:.2f}", "0 failed", not bad))
    if bad:
        rows.append(("stage", "failed runs", " ".join(bad[:40]), "", ""))
    ok = [r["name"] for r in runs if r["name"] not in bad]
    seg_of = lambda n: n.rsplit("_", 1)[-1] if n.endswith(("_91", "_21")) else n.split("_")[0]
    for seg in sorted({seg_of(n) for n in ok}):
        tm = timing(st, [n for n in ok if seg_of(n) == seg])
        rows.append(("timing", f"segment {seg} ({len(tm)} runs): start-up s median, max; wall s per simulated year "
                     "median, max", f"{np.median(tm[:, 0]):.1f} {tm[:, 0].max():.1f}; {np.nanmedian(tm[:, 1]):.3f} "
                     f"{np.nanmax(tm[:, 1]):.3f}", "", ""))
    mem = memory(st)
    rows.append(("memory", "peak GPU memory per run MiB: median, max", f"{np.median(list(mem.values())):.0f} "
                 f"{max(mem.values())}", "3072", max(mem.values()) <= 3072))
    return rows, RES / f"step4_stage_{st}.csv"


def main():
    rows, out = pilot() if sys.argv[1] == "pilot" else stage(sys.argv[2])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["check", "quantity", "value", "threshold", "pass"])
        w.writerows(rows)
    for r in rows:
        print(*r, sep=" | ")


if __name__ == "__main__":
    main()
