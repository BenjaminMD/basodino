"""F2 of the fix round (DESIGN.md): spin-up test of the basin confinement with the F1 posterior MAP.

Runs (experiment spinup, now with the basin process; u = 0, f = 1 unless given; balance = the MAP draw of
results/calib_f1_summary.csv, dT_pre included, so dT_1850 = clim_otl.offset acts on top of dT_pre):
  scan     stages/f2_scan.txt: dT_1850 = -2.6 to +0.4 K in steps of 0.2 K (16 runs, 300 model years each)
  sec R    stages/f2_sec{R}.txt: the secant of decision 10 / hindcast.py on dT_1850 towards the cycle-mean ice area
           inside SGI1850 T = 4.008 km2, accepted at the first steady run with |A_in - T| <= 0.25 km2; round 1 at
           the scan's dT_1850 for A_in = T - 0.125 km2 (linear between scan points), later rounds through the last two
           runs (or the scan's local slope), steps at most 1 K, at most 5 runs
Reduce (s03): per run the cycle means of hindcast.cycle_means (volume, ice area, inside and outside SGI1850, steady)
and the basin checks: mean yearly ice volume melted outside the basin over the last cycle (track basin_melt_km3),
ice cells (thk >= 1 m) outside the basin and on the outermost row or column of the domain, at the end and at every
decadal save (track every 10), the tongue and speeds at the end.
Outputs: results/f2_runs_{STAGE}.csv (reduce), results/f2_spinup.csv (every run with its role, the accepted one
marked; summary rows at the end).
usage:  Mac:  python checks/f2_spinup.py scan | sec R | table
        s03 (env.sh):  python checks/f2_spinup.py reduce STAGE
"""
import csv
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IGM))
from hindcast import CYCLE, MAX_RUNS, MAX_STEP, SPIN_END, TARGET, TOL, cycle_means  # noqa: E402

RES, STAGES = IGM / "results", IGM / "stages"
SCAN_OFF = -2.6 + 0.2 * np.arange(16)
U, F = 0.0, 1.0
A0, TAU0 = 78.0, 0.2153697218796329


def read_csv(path):
    return list(csv.DictReader(open(path)))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def f1_map():
    """The MAP draw of the F1 posterior (results/calib_f1_summary.csv row 'MAP draw ...': Fm, r_ice, c_prec, dT, dg,
    b, dT_pre; dg K m-1, b m-1 as clim_otl; the MAP is the largest posterior density of all points, not a kept draw)."""
    r = [r for r in read_csv(RES / "calib_f1_summary.csv") if r["quantity"].startswith("MAP draw")][0]
    v = [float(x) for x in r["value"].split()]
    return dict(zip(("Fm", "r_ice", "c_prec", "dT", "dg", "b", "dT_pre"), v))


def line(name, off, u=U, f=F):
    p = f1_map()
    return (f"{name} spinup inputs.geometry.u={u:.6f} processes.clim_otl.dT={p['dT']:.7g} "
            f"processes.clim_otl.dg={p['dg']:.7g} processes.clim_otl.b={p['b']:.7g} "
            f"processes.clim_otl.c_prec={p['c_prec']:.7g} processes.clim_otl.dT_pre={p['dT_pre']:.7g} "
            f"processes.smb_ti.Fm={p['Fm']:.7g} processes.iceflow.physics.viscosity.arrhenius={A0 * f:.6g} "
            f"processes.iceflow.physics.sliding.tau_ref={TAU0 * f ** (-1 / 3):.7g} "
            f"processes.clim_otl.offset={off:.4f} outputs.track.every=10")


def scan():
    lines = [line(f"scan_{k:02d}", off) for k, off in enumerate(SCAN_OFF)]
    head = ("# NAME EXPERIMENT overrides (run_stage.sh); fix round F2: dT_1850 scan of the F1 MAP draw (u = 0, f = 1), "
            "basin on, 300-year spin-ups from ice-free (checks/f2_spinup.py scan)\n")
    (STAGES / "f2_scan.txt").write_text(head + "\n".join(lines) + "\n")
    print(f"stages/f2_scan.txt: {len(lines)} runs, dT_1850 {SCAN_OFF[0]:+.1f} to {SCAN_OFF[-1]:+.1f} K")


def offset_of(name):
    for stage in ["f2_scan"] + [f"f2_sec{R}" for R in range(1, MAX_RUNS + 1)]:
        p = STAGES / f"{stage}.txt"
        if p.exists():
            for l in open(p):
                if l.split()[:1] == [name]:
                    return float(l.split("processes.clim_otl.offset=")[1].split()[0])
    raise KeyError(name)


def runs(prefix):
    out = []
    for p in sorted(RES.glob(f"f2_runs_{prefix}*.csv")):
        for r in read_csv(p):
            out.append({**r, "stage": p.stem[len("f2_runs_"):], "offset": offset_of(r["name"])})
    return out


def ok(r):
    return r["rc"] == "0" and r["Ain_last"] not in ("", None)


def sec(R):
    S = [r for r in runs("f2_scan") if ok(r)]
    off = np.array([r["offset"] for r in S])
    a = np.array([float(r["Ain_last"]) for r in S])
    i = np.argsort(off)
    off, a = off[i], a[i]
    if R == 1:
        x = float(np.interp(TARGET - TOL / 2, a[::-1], off[::-1]))
        note = f"round 1 at the scan's dT_1850 for A_in = {TARGET - TOL / 2:.3f} km2"
    else:
        prev = [r for r in runs("f2_sec") if ok(r)]
        prev.sort(key=lambda r: r["stage"])
        x2, a2 = prev[-1]["offset"], float(prev[-1]["Ain_last"])
        k = int(np.clip(np.searchsorted(off, x2) - 1, 0, len(off) - 2))
        s = min((a[k + 1] - a[k]) / (off[k + 1] - off[k]), -0.05)
        if len(prev) >= 2:
            x1, a1 = prev[-2]["offset"], float(prev[-2]["Ain_last"])
            if x2 != x1 and (a2 - a1) / (x2 - x1) < 0:
                s = (a2 - a1) / (x2 - x1)
        x = x2 + float(np.clip((TARGET - a2) / s, -MAX_STEP, MAX_STEP))
        note = f"round {R}: secant towards {TARGET} km2"
    head = f"# NAME EXPERIMENT overrides (run_stage.sh); fix round F2 secant (checks/f2_spinup.py sec {R}); {note}\n"
    (STAGES / f"f2_sec{R}.txt").write_text(head + line(f"sec_r{R}", x) + "\n")
    print(f"stages/f2_sec{R}.txt: dT_1850 {x:+.4f} K; {note}")


def reduce(stage):
    g = dict(np.load(IGM / "inputs/igm_static.npz"))
    basin = g["posbal"] == 1
    edge = np.zeros(basin.shape, bool)
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    log = {r["name"]: r for r in read_csv(IGM / "logs" / stage / "runs.csv")}
    rows = []
    for name in sorted(log):
        d = IGM / "runs" / stage / name
        row = {"name": name, "rc": log[name]["rc"], "wall_s": int(log[name]["end"]) - int(log[name]["start"])}
        if (d / "series.csv").exists() and (d / "fields.npz").exists():
            s = read_csv(d / "series.csv")
            c = cycle_means(s)
            if c:
                row.update({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in c.items()})
            last = [r for r in s if SPIN_END - CYCLE < int(r["year"]) <= SPIN_END]
            row["basin_melt_km3_per_a"] = f"{np.mean([float(r['basin_melt_km3']) for r in last]):.6f}"
            row["basin_melt_km3_max_year"] = f"{max(float(r['basin_melt_km3']) for r in s[1:]):.6f}"
            F = np.load(d / "fields.npz")
            keys = ["thk_end"] + sorted(k for k in F.files if k.startswith("thk_y"))
            row["saves_checked"] = len(keys)
            row["ice_outside_basin_cells_max"] = max(int(((F[k] >= 1.0) & ~basin).sum()) for k in keys)
            row["ice_edge_cells_max"] = max(int(((F[k] >= 1.0) & edge).sum()) for k in keys)
            row["thk_outside_basin_max_m"] = f"{max(float(F[k][~basin].max()) for k in keys):.3f}"
            e = s[-1]
            row.update(A_end=e["A1_km2"], Ain_end=e["A1_in1850_km2"], Aout_end=e["A1_out1850_km2"], V_end=e["V_km3"],
                       tongue_end=e["tongue_m"], v_med_end=e["v_med"], v_p90_end=e["v_p90"])
        rows.append(row)
    head = ["name", "rc", "wall_s", "V_last", "V_prev", "steady", "A_last", "Ain_last", "Ain_prev", "Aout_last",
            "basin_melt_km3_per_a", "basin_melt_km3_max_year", "saves_checked", "ice_outside_basin_cells_max",
            "ice_edge_cells_max", "thk_outside_basin_max_m", "A_end", "Ain_end", "Aout_end", "V_end", "tongue_end",
            "v_med_end", "v_p90_end"]
    write_csv(RES / f"f2_runs_{stage}.csv", head, [[r.get(k, "") for k in head] for r in rows])
    print(f"{len(rows)} runs: results/f2_runs_{stage}.csv")


def table():
    p = f1_map()
    allr = runs("f2_scan") + runs("f2_sec")
    acc = next((r for r in runs("f2_sec") if ok(r) and r["steady"] == "True"
                and abs(float(r["Ain_last"]) - TARGET) <= TOL), None)
    head = ["stage", "name", "dT_1850", "dT_pre_plus_dT_1850", "rc", "steady", "V_last", "V_prev", "A_last",
            "Ain_last", "Aout_last", "basin_melt_km3_per_a", "ice_outside_basin_cells_max", "ice_edge_cells_max",
            "tongue_end", "v_med_end", "v_p90_end", "wall_s", "accepted"]
    rows = [[r["stage"], r["name"], f"{r['offset']:+.4f}", f"{r['offset'] + p['dT_pre']:+.4f}",
             *(r.get(k, "") for k in head[4:-1]), acc is not None and r["name"] == acc["name"]] for r in allr]
    write_csv(RES / "f2_spinup.csv", head, rows)
    for r in rows:
        print(*r, sep=" | ")
    print("accepted:", acc["name"] if acc else "none", "| F1 MAP:", p)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "scan":
        scan()
    elif cmd == "sec":
        sec(int(sys.argv[2]))
    elif cmd == "reduce":
        reduce(sys.argv[2])
    elif cmd == "table":
        table()
