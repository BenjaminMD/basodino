"""Gates of DESIGN.md step 3 from the runs of stages/step3_*.txt (run_stage.sh) and noflow.py.

  rate     the Zn CPU campaign (Zn/mesh48, its ph.x share the threads 6-7, 70-71) beside the IGM work: rank 0 cpu
           seconds per wall second from the 30 s samples of Zn/mesh48g/campaign.log (read only; consecutive samples of
           the same ph_<i>.out), in four windows: the 30 min before the pause, pause with an IGM run active (any run
           of logs/s3_*/runs.csv), pause without, after the resume (logs/pause_mg.log: last STOP and CONT); and
           the GPU use of all step-3 stages: runs, process-hours (sum of run walls), card-hours (per GPU the union
           of its runs' intervals)
  flowoff  the CPU check before any GPU batch: IGM without flow (experiment flowoff, runs/s3_cpu/flowoff) against
           noflow.py (results/step3_noflow_cpu.csv) and against smb.py in float64
  gates    (a) the MAP draw 1991-2026 in IGM (runs fit91_map, fit21_map): yearly glacier-wide Ba against smb.py
               recomputed in float64 (numpy smb.balance) on the IGM surface of each 1 October and the same cells
               (thk >= 1 m, posbal rule), largest |difference| <= 1e-4 m w.e.; the year of the lowest Ba (2022) and
               Ba 2001 (> 0); ice-free cells with a negative first-year balance that gain ice in year one
           (b) surface speed median and p90 over the ice cells (thk >= 1 m) of the start state: online cold
               against the reference L-BFGS solve (3000 iterations; 1500 shown for convergence) on the MAP spin-up
               end state and on 1991 and 2021 at u = 0, f = 1, within 10 %; warm start (network of u = 0, f = 1)
               against cold at u = 0.5, f = 2 on both DEMs and on the network's own start state, within 10 %; final
               settings cold 3000 / warm 300 iterations (gate_b lists the earlier rounds)
           (c) timing of stage s3_batch3 (8 runs per H200, the final settings): start-up (wall at the first save) and wall per simulated
               year after it, per kind (spin-up, 1991-2021, 2021-2026), against DESIGN.md Cost (1.1, 1.0, 0.7 s per
               year; 25 s warm, 53 s cold start-up) times 1.5; peak GPU memory per run <= 3 GB (3072 MiB)
           (d) area (thk >= 1 m) of fit21_map on 1 October 2022-2026 and at the outline dates 2024-08-23, 2025-09-07,
               2026-09-27 against noflow.py of the same member (results/step3_noflow.csv), |difference| <= 0.02 km2;
               clean stop: run stop21 ends with no ice before 2100 (exit 0, track_end.txt, fields.npz)
Output: results/step3_gates.csv (gate, quantity, value, threshold, pass) or results/step3_flowoff.csv; printed.
usage (s03, env.sh, CPU):  python checks/step3.py flowoff | gates | rate
"""
import csv
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IGM))
from smb import ICE, Z_OTL, balance  # noqa: E402

RUNS, RES = IGM / "runs", IGM / "results"
MAP = dict(Fm=1.047871e-3, c_prec=2.7247, dT=-2.4563, dg=1.5805e-3, b=-1.3046e-4)     # results/calib_summary.csv
COST = {"spinup": (1.1, 53.0), "fit91": (1.0, 25.0), "fit21": (0.7, 25.0)}            # s per year, start-up s


def series(run):
    return list(csv.DictReader(open(RUNS / run / "series.csv")))


def smb_c(topg, thk, Y, g, f, m):
    """Annual balance C (m w.e.) of hydrological year Y by smb.py in float64 on the surface topg + thk, cells m."""
    k = int(np.flatnonzero(f["hydro_year"] == Y)[0])
    dz = (topg + thk).astype(np.float32)[m].astype(np.float64) - Z_OTL
    T = f["t"][k][:, None] + (f["gm"][:, None] + MAP["dg"]) * dz[None] + MAP["dT"]
    P = MAP["c_prec"] * f["p"][k][:, None] * (1 + MAP["b"] * dz[None])
    C = balance(T, P, g["ipot"][:, m].astype(np.float64), g["D"][m].astype(np.float64), MAP["Fm"],
                0.012 * MAP["Fm"])[0].sum(0)
    return np.where(g["posbal"][m] == 1, C, np.minimum(C, 0.0))


def smb_ba(topg, thk, Y, g, f):
    """Glacier-wide Ba of hydrological year Y by smb.py in float64, cells thk >= 1 m."""
    m = thk >= 1.0
    return smb_c(topg, thk, Y, g, f, m).mean(), m


def gate_a(rows, runs, g, f):
    out, worst, ba = [], 0.0, {}
    for run in runs:
        F = np.load(RUNS / run / "fields.npz")
        for r in series(run)[1:]:
            Y = int(r["year"])
            ref, _ = smb_ba(F["topg"], F[f"thk_y{Y - 1}"], Y, g, f)
            worst = max(worst, abs(float(r["Ba"]) - ref))
            ba[Y] = float(r["Ba"])
        y0 = int(series(run)[0]["year"])
        thk0, thk1 = F[f"thk_y{y0}"], F[f"thk_y{y0 + 1}"]
        gain = (thk0 == 0) & (thk1 > 0)
        C = np.zeros_like(thk0)
        C[gain] = smb_c(F["topg"], thk0, y0 + 1, g, f, gain)
        ice = thk0 > 0
        near = np.zeros_like(ice)
        near[1:] |= ice[:-1]; near[:-1] |= ice[1:]; near[:, 1:] |= ice[:, :-1]; near[:, :-1] |= ice[:, 1:]
        neg = gain & (C < 0)
        out.append((f"{run}: ice-free cells at the start; of them gaining ice in year one (largest gain m); of these "
                    "with a negative year-one balance (smb.py), and of those next to a start ice cell (flow)",
                    f"{int((thk0 == 0).sum())}; {int(gain.sum())} ({thk1[gain].max() if gain.any() else 0:.3f}); "
                    f"{int(neg.sum())}, {int((neg & near).sum())}"))
    low = min(ba, key=ba.get)
    low25 = min((y for y in ba if y <= 2025), key=ba.get)
    rows += [("a", "largest |Ba IGM - Ba smb.py float64|, m w.e., years " + f"{min(ba)}-{max(ba)}", f"{worst:.2e}",
              "1e-4", worst <= 1e-4),
             ("a", f"year of the lowest Ba {min(ba)}-{max(ba)}, value; Ba 2022", f"{low} {ba[low]:+.3f}; "
              f"{ba[2022]:+.3f}", "2022", low == 2022),
             ("a", f"year of the lowest Ba {min(ba)}-2025 (the GLAMOS years of the calibration), value",
              f"{low25} {ba[low25]:+.3f}", "2022", low25 == 2022),
             ("a", "Ba 2001", f"{ba.get(2001, np.nan):+.3f}", "> 0", ba.get(2001, -1) > 0)]
    rows += [("a", q, v, "", "") for q, v in out]


def speeds(run):
    r = series(run)[0]
    return float(r["v_med"]), float(r["v_p90"])


def gate_b(rows):
    """Final settings (cold 3000 iterations, warm 300 from the 3000-iteration networks) decide; the rounds before are
    listed as history (cold 300 and 1000, warm 5, 0, 50 from the 300-iteration networks)."""
    final = [("spin-up end state, cold 3000 vs reference", "s3_b2/cold3000_spin", "s3_b_spin/ref_spin"),
             ("1991 u 0 f 1, cold 3000 vs reference", "s3_nets3/net1994", "s3_b/ref_1991"),
             ("2021 u 0 f 1, cold 3000 vs reference", "s3_nets3/net2021", "s3_b/ref_2021"),
             ("1991 u 0.5 f 2, warm 300 vs cold 3000", "s3_b3/warm_1991", "s3_b3/cold_1991"),
             ("2021 u 0.5 f 2, warm 300 vs cold 3000", "s3_b3/warm_2021", "s3_b3/cold_2021"),
             ("1991 u 0 f 1, warm 0 from its own network vs cold 3000", "s3_b3/warmsame_1991", "s3_nets3/net1994"),
             ("2021 u 0 f 1, warm 0 from its own network vs cold 3000", "s3_b3/warmsame_2021", "s3_nets3/net2021")]
    history = [("spin-up, cold 300 vs reference", "s3_b_spin/cold_spin", "s3_b_spin/ref_spin"),
               ("1991, cold 300 vs reference", "s3_nets/net1994", "s3_b/ref_1991"),
               ("2021, cold 300 vs reference", "s3_nets/net2021", "s3_b/ref_2021"),
               ("spin-up, cold 1000 vs reference", "s3_b2/cold1000_spin", "s3_b_spin/ref_spin"),
               ("1991, cold 1000 vs reference", "s3_b2/cold1000_1991", "s3_b/ref_1991"),
               ("2021, cold 1000 vs reference", "s3_b2/cold1000_2021", "s3_b/ref_2021"),
               ("1991, cold 3000 (second network) vs reference", "s3_b2/cold3000_1991", "s3_b/ref_1991"),
               ("2021, cold 3000 (second network) vs reference", "s3_b2/cold3000_2021", "s3_b/ref_2021"),
               ("1991 u 0.5 f 2, warm 5 vs cold 300", "s3_b/warm1994", "s3_b/cold1994"),
               ("2021 u 0.5 f 2, warm 5 vs cold 300", "s3_b/warm2021", "s3_b/cold2021"),
               ("1991 u 0.5 f 2, warm 0 vs cold 300", "s3_b2/warm0_1991", "s3_b/cold1994"),
               ("2021 u 0.5 f 2, warm 0 vs cold 300", "s3_b2/warm0_2021", "s3_b/cold2021"),
               ("1991 u 0.5 f 2, warm 50 vs cold 300", "s3_b2/warm50_1991", "s3_b/cold1994"),
               ("2021 u 0.5 f 2, warm 50 vs cold 300", "s3_b2/warm50_2021", "s3_b/cold2021"),
               ("1991 u 0.5 f 2, warm 300 vs cold 300", "s3_b2/warm300_1991", "s3_b/cold1994"),
               ("2021 u 0.5 f 2, warm 300 vs cold 300", "s3_b2/warm300_2021", "s3_b/cold2021"),
               ("1991 u 0 f 1, warm 5 from its own network vs cold 300", "s3_b/warmsame1994", "s3_nets/net1994"),
               ("2021 u 0 f 1, warm 5 from its own network vs cold 300", "s3_b/warmsame2021", "s3_nets/net2021"),
               ("1991 u 0 f 1, warm 0 from its own network vs cold 300", "s3_b2/warmsame0_1991", "s3_nets/net1994"),
               ("2021 u 0 f 1, warm 0 from its own network vs cold 300", "s3_b2/warmsame0_2021", "s3_nets/net2021")]
    for tag, pairs in (("b", final), ("b history", history)):
        for what, a, b in pairs:
            va, vb = speeds(a), speeds(b)
            d = [va[i] / vb[i] - 1 for i in (0, 1)]
            rows.append((tag, f"{what}: median, p90 m a-1 ({a} | {b}); relative differences",
                         f"{va[0]:.3f} {va[1]:.3f} | {vb[0]:.3f} {vb[1]:.3f}; {d[0]:+.3f} {d[1]:+.3f}", "10 %",
                         max(abs(x) for x in d) <= 0.10))
    for s, st in (("spin", "s3_b_spin"), ("1991", "s3_b"), ("2021", "s3_b")):
        va, vb = speeds(f"{st}/ref_{s}"), speeds(f"{st}/ref1500_{s}")
        rows.append(("b", f"reference convergence {s}: median, p90 at 3000 | 1500 L-BFGS iterations",
                     f"{va[0]:.3f} {va[1]:.3f} | {vb[0]:.3f} {vb[1]:.3f}", "", ""))


def gate_c(rows):
    stage = "s3_batch3"
    runs = list(csv.DictReader(open(IGM / "logs" / stage / "runs.csv")))
    pid = {p: n for p, n in (l.split() for l in open(IGM / "logs" / stage / "pids.txt"))}
    mem = {}
    for line in open(IGM / "logs" / stage / "gpu_mem.txt"):
        _, p, mib = line.split()
        mem[pid[p]] = max(mem.get(pid[p], 0), int(mib))
    per = {}
    for r in runs:
        s = series(f"{stage}/{r['name']}")
        kind = r["name"].split("_")[0]
        years = float(s[-1]["t"]) - float(s[0]["t"])
        start = float(s[0]["wall_s"])
        rate = (float(s[-1]["wall_s"]) - start) / years if years > 0 else np.nan
        per.setdefault(kind, []).append((start, rate, r["name"], int(r["rc"]), years))
    span = max(int(r["end"]) for r in runs) - min(int(r["start"]) for r in runs)
    proc_h = sum(int(r["end"]) - int(r["start"]) for r in runs) / 3600
    rows.append(("c", "stage s3_batch3: runs, exit codes non-zero, makespan s, process-hours",
                 f"{len(runs)} {sum(int(r['rc']) != 0 for r in runs)} {span} {proc_h:.2f}", "", ""))
    for kind, v in per.items():
        if kind not in COST:
            continue
        st, rt = np.array([x[0] for x in v]), np.array([x[1] for x in v])
        cy, cs = COST[kind]
        rows.append(("c", f"{kind} ({len(v)} runs): wall s per simulated year after the first save, median and max",
                     f"{np.nanmedian(rt):.3f} {np.nanmax(rt):.3f}", f"{1.5 * cy:.2f}", np.nanmax(rt) <= 1.5 * cy))
        rows.append(("c", f"{kind}: start-up s (wall at the first save), median and max", f"{np.median(st):.1f} "
                     f"{st.max():.1f}", f"{1.5 * cs:.1f}", st.max() <= 1.5 * cs))
    worst = max(mem.values()) if mem else np.nan
    rows.append(("c", "peak GPU memory per run, MiB: median and max over the runs",
                 f"{np.median(list(mem.values())):.0f} {worst}", "3072", worst <= 3072))


def gate_d(rows):
    s = series("s3_batch3/fit21_map")
    F = np.load(RUNS / "s3_batch3/fit21_map/fields.npz")
    nf = {r["date"]: float(r["A1_km2"]) for r in csv.DictReader(open(RES / "step3_noflow.csv")) if r["member"] == "0"}
    worst = 0.0
    parts = []
    for r in s[1:]:
        d = f"{int(r['year'])}-10-01"
        diff = float(r["A1_km2"]) - nf[d]
        worst = max(worst, abs(diff))
        parts.append(f"{d} {float(r['A1_km2']):.4f}/{nf[d]:.4f}")
    for d in ("2024-08-23", "2025-09-07", "2026-09-27"):
        a = (F[f"thk_{d.replace('-', '')}"] >= 1.0).sum() * 1e-4
        worst = max(worst, abs(a - nf[d]))
        parts.append(f"{d} {a:.4f}/{nf[d]:.4f}")
    rows.append(("d", "area thk >= 1 m km2, IGM/noflow: " + "; ".join(parts), f"largest |difference| {worst:.4f}",
                 "0.02", worst <= 0.02))
    run = RUNS / "s3_batch3/stop21"
    rc = [r["rc"] for r in csv.DictReader(open(IGM / "logs/s3_batch3/runs.csv")) if r["name"] == "stop21"]
    end = (run / "track_end.txt").read_text().strip() if (run / "track_end.txt").exists() else "missing"
    last = series("s3_batch3/stop21")[-1]
    ok = rc == ["0"] and end.startswith("no ice") and (run / "fields.npz").exists() and float(last["A1_km2"]) == 0
    rows.append(("d", "clean stop (stop21: 2021 start, +8 degC): exit code, track_end, last year, last area",
                 f"{rc} '{end}' {last['year']} {last['A1_km2']}", "no ice, exit 0", ok))


def flowoff():
    g, f = dict(np.load(IGM / "inputs/igm_static.npz")), dict(np.load(IGM / "inputs/forcing_obs.npz"))
    rows = []
    run = "s3_cpu/flowoff"
    F = np.load(RUNS / run / "fields.npz")
    nf = {r["date"]: r for r in csv.DictReader(open(RES / "step3_noflow_cpu.csv")) if r["member"] == "0"}
    worst_a = worst_ba = worst_ref = 0.0
    for r in series(run):
        d = f"{int(r['year'])}-10-01"
        worst_a = max(worst_a, abs(float(r["A1_km2"]) - float(nf[d]["A1_km2"])),
                      abs(float(r["V_km3"]) - float(nf[d]["V_km3"])))
        if r["Ba"]:
            Y = int(r["year"])
            worst_ba = max(worst_ba, abs(float(r["Ba"]) - float(nf[d]["Ba"])))
            ref, _ = smb_ba(F["topg"], F[f"thk_y{Y - 1}"], Y, g, f)
            worst_ref = max(worst_ref, abs(float(r["Ba"]) - ref))
    for d in ("2024-08-23", "2025-09-07", "2026-09-27"):
        a = (F[f"thk_{d.replace('-', '')}"] >= 1.0).sum() * 1e-4
        worst_a = max(worst_a, abs(a - float(nf[d]["A1_km2"])))
    rows += [("flowoff", "IGM without flow vs noflow.py: largest |difference| of area km2 and volume km3 (years and "
              "outline dates)", f"{worst_a:.2e}", "", ""),
             ("flowoff", "IGM without flow vs noflow.py: largest |difference| of Ba m w.e.", f"{worst_ba:.2e}", "", ""),
             ("flowoff", "IGM without flow vs smb.py float64: largest |difference| of Ba m w.e.", f"{worst_ref:.2e}",
              "1e-4", worst_ref <= 1e-4)]
    return rows


def rate():
    from datetime import datetime
    ev = [l.split() for l in open(IGM / "logs/pause_mg.log")]
    stop = max(datetime.fromisoformat(e[0]).timestamp() for e in ev if e[1] == "STOP")
    cont = max(datetime.fromisoformat(e[0]).timestamp() for e in ev if e[1] == "CONT")
    active = [(int(r["start"]), int(r["end"])) for f in sorted((IGM / "logs").glob("s3_*/runs.csv"))
              for r in csv.DictReader(open(f))]
    smp = []
    for line in open("/scratch/befahl/Zn/mesh48g/campaign.log"):
        w = line.split()
        if "cpu" in w and "time" in w:
            smp.append((int(w[0]), w[2], float(w[w.index("time") + 2])))
    acc = {}
    for (t0, f0, c0), (t1, f1, c1) in zip(smp[:-1], smp[1:]):
        if f0 != f1 or t1 - t0 > 60 or c1 < c0:
            continue
        m = (t0 + t1) / 2
        if m < stop - 1800:
            continue
        k = ("before pause (30 min)" if m < stop else "after resume" if m > cont else
             "pause, IGM run active" if any(a <= m <= b for a, b in active) else "pause, no IGM run")
        a = acc.setdefault(k, [0.0, 0.0])
        a[0] += c1 - c0
        a[1] += t1 - t0
    runs = [r for f in sorted((IGM / "logs").glob("s3_*/runs.csv")) for r in csv.DictReader(open(f))]
    card = 0.0
    for gpu in ("0", "1"):
        iv = sorted((int(r["start"]), int(r["end"])) for r in runs if r["gpu"] == gpu)
        end = 0
        for a, b in iv:
            card += max(0, b - max(a, end))
            end = max(end, b)
    use = ("use", "step-3 GPU stages: runs (failed), process-hours, card-hours (2 x H200), first start - last end",
           f"{len(runs)} ({sum(r['rc'] != '0' for r in runs)}), {sum(int(r['end']) - int(r['start']) for r in runs) / 3600:.2f}, "
           f"{card / 3600:.2f}, {datetime.fromtimestamp(min(int(r['start']) for r in runs)):%H:%M:%S}-"
           f"{datetime.fromtimestamp(max(int(r['end']) for r in runs)):%H:%M:%S}", "", "")
    rows = [use, ("rate", f"pause window {datetime.fromtimestamp(stop):%H:%M:%S}-{datetime.fromtimestamp(cont):%H:%M:%S}, "
             f"{(cont - stop) / 60:.1f} min", "", "", "")]
    for k, (c, t) in acc.items():
        rows.append(("rate", f"Zn mesh48 rank-0 cpu s per wall s, {k} ({t / 60:.1f} min sampled)", f"{c / t:.3f}", "",
                     ""))
    return rows


def main():
    if sys.argv[1] == "rate":
        rows, out = rate(), RES / "step3_rate.csv"
    elif sys.argv[1] == "flowoff":
        rows, out = flowoff(), RES / "step3_flowoff.csv"
    else:
        g, f = dict(np.load(IGM / "inputs/igm_static.npz")), dict(np.load(IGM / "inputs/forcing_obs.npz"))
        rows, out = [], RES / "step3_gates.csv"
        for gate in (lambda: gate_a(rows, ["s3_batch3/fit91_map", "s3_batch3/fit21_map"], g, f), lambda: gate_b(rows),
                     lambda: gate_c(rows), lambda: gate_d(rows)):
            try:
                gate()
            except (FileNotFoundError, KeyError, IndexError, ValueError) as e:
                rows.append(("?", f"not computed: {type(e).__name__} {e}", "", "", ""))
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gate", "quantity", "value", "threshold", "pass"])
        w.writerows(rows)
    for r in rows:
        print(*r, sep=" | ")


if __name__ == "__main__":
    main()
