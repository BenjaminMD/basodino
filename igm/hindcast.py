"""Spin-up and hindcast of the 128 posterior draws (DESIGN.md step 5, decisions 10 and 11): run lists, the secant on
dT_1850, the reduction of the runs, the validation table and the 2026 start states.
Fix round (DESIGN.md 'F5'): draws of results/posterior_f3_128.csv (F3: balance draw of F1 with dT_pre, u, f), basin
on (F2), dT_pre passed to clim_otl (in a spin-up every model year, so dT_1850 = offset acts on top of it); file prefix
f5 (stages f5_*, results/f5_*.csv, inputs/f5_state2026.npz). The first ensemble used P = "step5", stages s5_*,
results/posterior_128.csv and step5_targets.csv (git history of this file); its result files stay as they are.

Spin-up (experiment spinup, cold online solver, 3000 initial iterations): from ice-free on bed(u) of the draw, the
hydrological years 1865-1894 of the observed forcing in a fixed cycle plus offset = dT_1850, model years 1551-1850
(300 years, the cap of one secant run); balance (dT, dg, b, c_prec, Fm; r_ice = 0.012 Fm), u and f of the draw
(POSTERIOR; A = 78 f, tau_ref = 0.2154 f^(-1/3)); positive balance only on posbal (decision 2,
smb_ti). Per run (reduce): the means over the last two 30-year cycles (model years 1791-1820, 1821-1850) of the
volume and of the ice area (thk >= 1 m) inside and outside SGI1850 (track.py); steady when the last cycle-mean
volume lies within 1 % of the one before.
Secant on dT_1850 (decision 10, per draw, at most 5 runs): target the cycle-mean area inside SGI1850, T = 4.008 km2,
tolerance 0.25 km2; a draw is done at its first steady run with |A_in - T| <= 0.25 (A_in cannot exceed the 4.008 km2
of the outline, so every accepted run lies in 3.758-4.008). Round 1: every draw at the dT_1850 where the scan of the
largest-weight draw (stage SCAN, offsets SCAN_OFF) gives A_in = T - 0.125 (mid band), linear between scan points.
Round R > 1: secant through the draw's last two runs towards T, or with one run (or a non-negative secant slope) the
scan's local slope; steps at most 1 K; with RESTART (f5) the run starts from the end state of the draw's previous run
(topg, thk_end of its fields.npz) instead of ice-free, so a draw's runs form one chain of up to 5 x 300 years. The
hindcast starts from the accepted run.
Hindcast (experiment hindcast, cold, 3000 initial iterations): from the end state of the accepted spin-up (topg,
thk_end of its fields.npz) 1850.75 to 2026.75 with the observed weather of decision 6 (no offset).
Validation (decision 11; observed values, dates and the expected misses: results/P_targets.csv, written before the
hindcast ran; step5_targets.csv by targets_step5.py, f5_targets.csv by 'targets' below): per row the model value of each draw, its 5/50/95 % over the 128 draws
(equal weights: the draws are a resample; n counts the draws with a value: a length is missing when the largest ice
body no longer touches the flowline), observed inside the 5-95 % band or not, (median - observed) / sigma where a
sigma exists (within_2sigma: |median - observed| <= 2 sigma), and for the rows with an expected miss whether the
median misses in the stated direction; with P = f5
also the decision-11 expectation (column expected_d11) and the first ensemble's 5/50/95 % and band of the same row
(results/step5_validation.csv).
targets (P = f5): results/f5_targets.csv = the rows of step5_targets.csv with the volume rows replaced by the 11
geodetic volume changes of F1 (results/calib_f1_bands.csv kind dv: dates, observed, sigma_obs as calibrated, 1907-2024);
column expected: 'too little loss' or 'too much loss' where F1's 90 % predictive band on the prescribed geometry
excludes the observation (direction of F1's model median), else empty; column expected_d11: decision 11's statement
(1929-1949 too little loss; 1949-1961, 1961-1971, 1971-1985 too much loss).
Outputs (results/, P = step5 | f5): P_runs_STAGE.csv (reduce; spin-up: cycle means; hindcast: model value per validation row
and extra values: areas inside SGI1850, volumes, the 2021-2026 outline-date areas, surface RMS, and the contact of
the ice with the domain edge: edge cells with ice in 1850, decadal saves with ice on the edge, the last such year;
f5: the ice volume the basin removed 1850-2026 and the most ice cells outside the basin at any kept field),
P_series_HIND.csv (every save of every hindcast run), P_secant.csv (every spin-up run per draw, the accepted one
marked), P_spinup.csv (per draw: accepted run, dT_1850, dT_pre, areas, steady), P_validation.csv; s03
inputs/P_state2026.npz (states).
usage:  Mac:  python hindcast.py targets | scan | spin R | hind | validate
        s03 (env.sh):  python hindcast.py reduce STAGE | states
"""
import csv
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parent
RES, STAGES, INP = IGM / "results", IGM / "stages", IGM / "inputs"
CYCLE, SPIN_END, STEADY = 30, 1850, 0.01
TARGET, TOL, MAX_RUNS, MAX_STEP = 4.008, 0.25, 5, 1.0
P, POSTERIOR = "f5", "posterior_f3_128.csv"     # first ensemble: "step5", "posterior_128.csv", stages s5_*
SCAN = "f5_scan"                                 # first ensemble: s5_scan2, -1.4 to +1.6 K
SCAN_OFF = -2.6 + 0.2 * np.arange(16)            # as the F2 scan (f2_scan): saturation below about -1.4 K
SPIN = "f5_spin"                                 # stages f5_spin1, f5_spin2, ...
RESTART = True                                   # f5: round R > 1 continues from the end state of round R - 1 (the
                                                 # f5 scan: at f = 0.054 300 years from ice-free are not steady near T)
HIND = "f5_hind"


def read_csv(path):
    return list(csv.DictReader(open(path)))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def draws():
    return read_csv(RES / POSTERIOR)


def overrides(r):
    """hydra overrides of one posterior draw (balance, u, f; dT_pre where the draw has one, F1)."""
    return (f"inputs.geometry.u={r['u']} processes.clim_otl.dT={r['dT']} processes.clim_otl.dg={r['dg']} "
            f"processes.clim_otl.b={r['b']} processes.clim_otl.c_prec={r['c_prec']} processes.smb_ti.Fm={r['Fm']} "
            + (f"processes.clim_otl.dT_pre={r['dT_pre']} " if r.get("dT_pre") else "")
            + f"processes.iceflow.physics.viscosity.arrhenius={r['arrhenius']} "
            f"processes.iceflow.physics.sliding.tau_ref={r['tau_ref']}")


def spin_line(name, r, offset):
    return f"{name} spinup {overrides(r)} processes.clim_otl.offset={offset:.4f}"


# --- spin-up ----------------------------------------------------------------------------------------------------------

def scan():
    """stages/SCAN.txt: the draw of the largest weight at dT_1850 = SCAN_OFF (steps of 0.2 K; the first ensemble's
    first scan, step5_scan.txt, 0 to 3 K, stopped every run above 0.4 K after one year: track's no-ice stop, since
    switched off for spin-ups)."""
    r = max(draws(), key=lambda r: float(r["weight"]))
    lines = [spin_line(f"scan_{k:02d}", r, off) for k, off in enumerate(SCAN_OFF)]
    head = (f"# NAME EXPERIMENT overrides (run_stage.sh); DESIGN.md step 5: dT_1850 scan of sample {r['sample']} "
            f"(largest weight), {SCAN_OFF[0]:+.1f} to {SCAN_OFF[-1]:+.1f} K, 300-year spin-ups from ice-free "
            f"(hindcast.py scan)\n")
    (STAGES / f"{SCAN}.txt").write_text(head + "\n".join(lines) + "\n")
    print(f"stages/{SCAN}.txt: {len(lines)} runs of sample {r['sample']}")


def scan_curve():
    """(dT_1850, A_in) of the scan, A_in falling with dT_1850 (runs of the stage SCANfix replace those of SCAN)."""
    rows = {r["name"]: r for r in read_csv(RES / f"{P}_runs_{SCAN}.csv")}
    if (RES / f"{P}_runs_{SCAN}fix.csv").exists():
        rows.update({r["name"]: r for r in read_csv(RES / f"{P}_runs_{SCAN}fix.csv")})
    return SCAN_OFF, np.array([float(rows[f"scan_{k:02d}"]["Ain_last"]) for k in range(len(SCAN_OFF))])


def scan_slope(dT):
    """dA_in / d dT_1850 of the scan at dT (the segment holding dT; ends: the end segments), km2 K-1, kept < 0."""
    off, a = scan_curve()
    k = int(np.clip(np.searchsorted(off, dT) - 1, 0, len(off) - 2))
    s = (a[k + 1] - a[k]) / (off[k + 1] - off[k])
    return min(s, -0.05)


def spin_runs():
    """Every reduced spin-up run per draw: {draw: [(round, name, offset, row)]}."""
    out = {}
    R = 1
    while (RES / f"{P}_runs_{SPIN}{R}.csv").exists():
        lines = {l.split()[0]: l for l in open(STAGES / f"{SPIN}{R}.txt") if not l.startswith("#")}
        for r in read_csv(RES / f"{P}_runs_{SPIN}{R}.csv"):
            off = float(lines[r["name"]].split("processes.clim_otl.offset=")[1].split()[0])
            out.setdefault(int(r["name"][1:4]), []).append((R, r["name"], off, r))
        R += 1
    return out


def accepted(runs):
    """The first steady run within the tolerance, or None."""
    for run in runs:
        r = run[3]
        if r["rc"] == "0" and r["steady"] == "True" and abs(float(r["Ain_last"]) - TARGET) <= TOL:
            return run
    return None


def spin(R):
    """stages/SPIN{R}.txt: the next spin-up of every draw not yet accepted."""
    D, rest = draws(), {}
    if R == 1:
        off, a = scan_curve()
        dT0 = float(np.interp(TARGET - TOL / 2, a[::-1], off[::-1]))
        nxt = {int(r["draw"]): dT0 for r in D}
        note = f"round 1: dT_1850 {dT0:+.4f} K for every draw (scan {SCAN}: A_in = {TARGET - TOL / 2:.3f} km2)"
    else:
        runs, nxt = spin_runs(), {}
        for d in range(len(D)):
            rs = runs.get(d, [])
            if accepted(rs) or len(rs) >= MAX_RUNS:
                continue
            assert len(rs) == R - 1, (d, len(rs), R)
            ok = [x for x in rs if x[3]["Ain_last"] not in ("", None)]
            assert ok, f"draw {d}: no reduced run"
            x2, a2 = ok[-1][2], float(ok[-1][3]["Ain_last"])
            s = scan_slope(x2)
            if len(ok) >= 2:
                x1, a1 = ok[-2][2], float(ok[-2][3]["Ain_last"])
                if x2 != x1 and (a2 - a1) / (x2 - x1) < 0:
                    s = (a2 - a1) / (x2 - x1)
            nxt[d] = x2 + float(np.clip((TARGET - a2) / s, -MAX_STEP, MAX_STEP))
            if RESTART:
                rest[d] = f"inputs.geometry.dem=restart inputs.geometry.restart=runs/{SPIN}{R - 1}/{rs[-1][1]}/fields.npz"
        note = (f"round {R}: secant towards {TARGET} km2 for the {len(nxt)} draws not yet accepted"
                + (", each from the end state of its previous run" if RESTART else ""))
    lines = [spin_line(f"d{d:03d}_r{R}", D[d], v) + (" " + rest[d] if d in rest else "") for d, v in sorted(nxt.items())]
    head = f"# NAME EXPERIMENT overrides (run_stage.sh); DESIGN.md step 5 spin-up (hindcast.py spin {R}); {note}\n"
    (STAGES / f"{SPIN}{R}.txt").write_text(head + "\n".join(lines) + ("\n" if lines else ""))
    print(f"stages/{SPIN}{R}.txt: {len(lines)} runs; {note}")


def secant_table():
    """results/P_secant.csv and P_spinup.csv from every spin-up round so far."""
    D, runs = draws(), spin_runs()
    rows, per = [], []
    for d in range(len(D)):
        rs = runs.get(d, [])
        acc = accepted(rs)
        for R, name, off, r in rs:
            rows.append([d, D[d]["sample"], R, name, f"{off:.4f}", D[d].get("dT_pre", ""), r["rc"], r["Ain_last"], r["Aout_last"],
                         r["A_last"], r["V_last"], r["V_prev"], r["steady"], acc is not None and acc[1] == name])
        if acc:
            r = acc[3]
            per.append([d, D[d]["sample"], acc[1], len(rs), f"{acc[2]:.4f}", D[d].get("dT_pre", ""), r["Ain_last"], r["Aout_last"],
                        r["A_last"], r["V_last"], f"{float(r['V_last']) / float(r['V_prev']) - 1:+.5f}", r["steady"],
                        r["tongue_end"], r["v_med_end"], r["v_p90_end"], r["wall_s"]])
        else:
            per.append([d, D[d]["sample"], "", len(rs)] + [""] * 12)
    write_csv(RES / f"{P}_secant.csv", ["draw", "sample", "round", "name", "dT_1850", "dT_pre", "rc", "Ain_last",
              "Aout_last", "A_last", "V_last", "V_prev", "steady", "accepted"], rows)
    write_csv(RES / f"{P}_spinup.csv", ["draw", "sample", "run", "runs", "dT_1850", "dT_pre", "Ain_km2", "Aout_km2",
              "A_km2", "V_km3", "V_change_last_cycle", "steady", "tongue_m_1850", "v_med_1850", "v_p90_1850", "wall_s"],
              per)
    n = sum(1 for q in per if q[2])
    print(f"results/{P}_secant.csv ({len(rows)} runs), {P}_spinup.csv: {n} of {len(D)} draws accepted")
    return n


# --- hindcast ---------------------------------------------------------------------------------------------------------

def hind():
    """stages/HIND.txt: every draw from its accepted spin-up end state."""
    assert secant_table() == len(draws()), "not every draw has an accepted spin-up"
    D, runs = draws(), spin_runs()
    lines = []
    for d in range(len(D)):
        R, name, off, _ = accepted(runs[d])
        lines.append(f"h{d:03d} hindcast {overrides(D[d])} "
                     f"inputs.geometry.restart=runs/{SPIN}{R}/{name}/fields.npz")
    head = "# NAME EXPERIMENT overrides (run_stage.sh); DESIGN.md step 5 hindcast 1850.75-2026.75 (hindcast.py hind)\n"
    (STAGES / f"{HIND}.txt").write_text(head + "\n".join(lines) + "\n")
    print(f"stages/{HIND}.txt: {len(lines)} runs")


def key(d):
    return "thk_" + d.replace("-", "")


def body(ice):
    from scipy.ndimage import label
    lab, n = label(ice)
    return (lab == np.argmax(np.bincount(lab[ice]))) if n else ice


def hind_values(F, s, T, g, t):
    """{validation row name or extra: model value} of one hindcast run (F fields.npz, s series rows)."""
    yr = {int(r["year"]): r for r in s}
    out = {}
    V = lambda d: float(F[key(d)].sum()) * 1e-7
    for r in T:
        k, n, a, b = r["kind"], r["name"], r["date_start"], r["date_end"]
        if k == "area":
            out[n] = (F[key(b)] >= 1.0).sum() * 1e-4
            out[n + " inside SGI1850"] = ((F[key(b)] >= 1.0) & (g["sgi1850"] > 0)).sum() * 1e-4
        elif k == "volume":
            out[n] = V(b) - V(a)
        elif k == "tongue":
            out[n] = float(yr[int(b[:4])]["tongue_m"])
        elif k == "length":
            L = [yr[int(x[:4])]["length_m"] for x in (a, b)]       # empty: the largest body is off the flowline
            out[n] = float(L[1]) - float(L[0]) if all(L) else np.nan
        elif k == "surface":
            cells = {"1991": g["mask_1994"] == 1, "2012": t["has12"], "2021": g["mask_2021"] == 1}[b[:4]]
            obs = {"1991": g["usurf_1994"], "2012": t["s12"], "2021": g["usurf_2021"]}[b[:4]]
            z = F["topg"] + F[key(b)]
            out[n] = float(z[cells].mean())
            out[n + " RMS"] = float(np.sqrt(((z - obs)[cells] ** 2).mean()))
    for d, kind in (("2021-09-30", "all"), ("2024-08-23", "all"), ("2025-09-07", "body"), ("2026-09-27", "body")):
        ice = F[key(d)] >= 1.0
        out[f"area {d} {kind}"] = (ice if kind == "all" else body(ice)).sum() * 1e-4
    out["V 1850"], out["V 1991-09-18"] = float(s[0]["V_km3"]), V("1991-09-18")
    out["V 2021-09-30"], out["V 2026-09-30"] = V("2021-09-30"), float(F["thk_end"].sum()) * 1e-7
    out["A_in 1850"], out["A_out 1850"] = float(s[0]["A1_in1850_km2"]), float(s[0]["A1_out1850_km2"])
    out["A_out 1929-08-13"] = ((F[key("1929-08-13")] >= 1.0) & (g["sgi1850"] == 0)).sum() * 1e-4
    out["A 2026-09-30"] = float(s[-1]["A1_km2"])
    edge = np.zeros(F["topg"].shape, bool)                       # the outermost row or column of the domain
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    dec = sorted(k for k in F.files if k.startswith("thk_y"))
    out["edge ice 1850"] = float((F["thk_y1850"][edge] >= 1.0).sum())
    out["edge saves"] = float(sum((F[k][edge] >= 1.0).any() for k in dec))
    out["edge last year"] = float(max([int(k[5:]) for k in dec if (F[k][edge] >= 1.0).any()], default=0))
    out["basin melt 1850-2026 km3"] = sum(float(r["basin_melt_km3"]) for r in s if r.get("basin_melt_km3"))
    kept = [k for k in F.files if k.startswith("thk_")]
    out["ice cells outside basin max"] = float(max(((F[k] >= 1.0) & (g["posbal"] != 1)).sum() for k in kept))
    return out


SERIES = ("year", "V_km3", "A1_km2", "A1_in1850_km2", "A1_out1850_km2", "Ba", "Bw", "Bs", "tongue_m", "length_m",
          "v_med", "v_p90", "its_median", "basin_melt_km3")


def reduce(stage):
    runs = {r["name"]: r for r in read_csv(IGM / "logs" / stage / "runs.csv")}
    hindcast = stage.startswith(HIND)
    if hindcast:
        g = dict(np.load(INP / "igm_static.npz"))
        t = dict(np.load(INP / "step5_targets.npz"))
        T = read_csv(RES / f"{P}_targets.csv")
        ser = []
    rows = []
    for name in sorted(runs):
        d = IGM / "runs" / stage / name
        end = (d / "track_end.txt").read_text().split()[0] if (d / "track_end.txt").exists() else ""
        row = {"name": name, "rc": runs[name]["rc"], "end": end,
               "wall_s": int(runs[name]["end"]) - int(runs[name]["start"])}
        s = read_csv(d / "series.csv") if (d / "series.csv").exists() else []
        if s:                                                    # empty: stopped before its first save
            row["iterations"] = s[-1]["it"]
            if hindcast and end == "end" and runs[name]["rc"] == "0":
                vals = hind_values(np.load(d / "fields.npz"), s, T, g, t)
                row.update({k: f"{v:.6f}" for k, v in vals.items()})
                ser += [[name] + [r.get(c, "") for c in SERIES] for r in s]
            elif not hindcast:
                c = cycle_means(s)
                if c:
                    row.update({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in c.items()})
                    last = s[-1]
                    row.update(V_end=last["V_km3"], A_end=last["A1_km2"], Ain_end=last["A1_in1850_km2"],
                               Aout_end=last["A1_out1850_km2"], tongue_end=last["tongue_m"],
                               v_med_end=last["v_med"], v_p90_end=last["v_p90"])
        rows.append(row)
    if hindcast:
        head = ["name", "rc", "end", "wall_s", "iterations"] + [r["name"] for r in T] + sorted(
            {k for r in rows for k in r} - {"name", "rc", "end", "wall_s", "iterations"} - {r["name"] for r in T})
        write_csv(RES / f"{P}_series_{stage}.csv", ["name"] + list(SERIES), ser)
    else:
        head = ["name", "rc", "end", "wall_s", "iterations", "V_last", "V_prev", "steady", "A_last", "A_prev",
                "Ain_last", "Ain_prev", "Aout_last", "Aout_prev", "V_end", "A_end", "Ain_end", "Aout_end",
                "tongue_end", "v_med_end", "v_p90_end"]
    write_csv(RES / f"{P}_runs_{stage}.csv", head, [[r.get(k, "") for k in head] for r in rows])
    bad = [r["name"] for r in rows if r["rc"] != "0" or r["end"] == ""]
    print(f"{len(rows)} runs reduced, {len(bad)} failed or unfinished {bad[:10]}: results/{P}_runs_{stage}.csv")


def cycle_means(s):
    """Means over the last two 30-year cycles ending at SPIN_END (model years Y of the saves)."""
    out = {}
    for tag, y1 in (("last", SPIN_END), ("prev", SPIN_END - CYCLE)):
        rows = [r for r in s if y1 - CYCLE < int(r["year"]) <= y1]
        if len(rows) != CYCLE:
            return None
        for k, c in (("V", "V_km3"), ("A", "A1_km2"), ("Ain", "A1_in1850_km2"), ("Aout", "A1_out1850_km2")):
            out[f"{k}_{tag}"] = np.mean([float(r[c]) for r in rows])
    out["steady"] = abs(out["V_last"] / out["V_prev"] - 1) <= STEADY if out["V_prev"] > 0 else False
    return out


FIRST = ("p5", "p50", "p95", "band")             # columns of the first ensemble's step5_validation.csv shown beside
D11 = {"1929-1949": "too little loss", "1949-1961": "too much loss", "1961-1971": "too much loss",
       "1971-1985": "too much loss"}           # decision 11, stated before any hindcast (Huss et al. 2009)


def targets():
    """results/f5_targets.csv (docstring 'targets'): step5_targets.csv with its volume rows replaced by the 11 of F1."""
    old = read_csv(RES / "step5_targets.csv")
    code = lambda d: {"1907-10-01": "19079999", "1929-08-13": "19299999", "2018-09-01": "20189999"}.get(
        d, d.replace("-", ""))                                  # GLAMOS survey codes (9999: day unknown)
    vc = {(r[3], r[4]): r for r in csv.reader(open(IGM.parent / "data/glamos_basodino_volumechange.csv"))
          if len(r) > 12}
    rows = []
    for r in read_csv(RES / "calib_f1_bands.csv"):
        if r["kind"] != "dv":
            continue
        name = f"{r['date0'][:4]}-{r['date1'][:4]}"
        v = vc[code(r["date0"]), code(r["date1"])]
        dv, B, sB = float(v[9]), float(v[11]), float(v[12])     # as targets_step5.py and calib_smb.py
        assert abs(dv - float(r["observed"])) < 1e-4, (name, dv, r["observed"])
        obs, lo, hi, med = dv, float(r["predictive_p5"]), float(r["predictive_p95"]), float(r["model_p50"])
        exp = "" if lo <= obs <= hi else ("too little loss" if med > obs else "too much loss")
        rows.append({"kind": "volume", "name": name, "date_start": r["date0"], "date_end": r["date1"],
                     "observed": dv, "sigma": round(abs(dv) * sB / abs(B), 6), "unit": "km3", "expected": exp,
                     "expected_d11": D11.get(name, ""),
                     "note": f"F1 (calib_f1_bands.csv): model 5/50/95 % {r['model_p5']} / {r['model_p50']} / "
                             f"{r['model_p95']}, predictive {r['predictive_p5']} to {r['predictive_p95']}"})
    vol = [i for i, r in enumerate(old) if r["kind"] == "volume"]
    out = old[:vol[0]] + rows + old[vol[-1] + 1:]
    for r in out:
        r.setdefault("expected_d11", "")
        if r["kind"] != "volume":
            r["expected"] = ""
    head = ["kind", "name", "date_start", "date_end", "observed", "sigma", "unit", "expected", "expected_d11", "note"]
    write_csv(RES / "f5_targets.csv", head, [[r[k] for k in head] for r in out])
    for r in rows:
        print(r["name"], r["observed"], r["sigma"], r["expected"] or "-", r["expected_d11"] or "-", sep=" | ")


def validate():
    """results/P_validation.csv from P_targets.csv and P_runs_HIND.csv (f5: with the first ensemble's band per row)."""
    T = read_csv(RES / f"{P}_targets.csv")
    H = [r for r in read_csv(RES / f"{P}_runs_{HIND}.csv") if r["end"] == "end" and r["rc"] == "0"]
    first = {r["name"]: r for r in read_csv(RES / "step5_validation.csv")} if P != "step5" else {}
    out = []
    for r in T + [{"kind": "extra", "name": k} for k in H[0] if k not in {t["name"] for t in T}
                  and k not in ("name", "rc", "end", "wall_s", "iterations")]:
        m = np.array([float(h[r["name"]]) for h in H])
        m = m[np.isfinite(m)]                                    # length: nan where the largest body left the line
        if not len(m):
            out.append([r["kind"], r["name"], r.get("date_start", ""), r.get("date_end", ""), r.get("observed", ""),
                        r.get("sigma", ""), r.get("unit", ""), "", "", "", "no value", "", "", r.get("expected", ""), "",
                        0, r.get("expected_d11", ""), ""] + [first.get(r["name"], {}).get(c, "") for c in FIRST])
            continue
        p5, p50, p95 = np.percentile(m, [5, 50, 95])
        obs = float(r["observed"]) if r.get("observed") else None
        sig = float(r["sigma"]) if r.get("sigma") else None
        inside = "" if obs is None else ("inside" if p5 <= obs <= p95 else "outside")
        z = "" if obs is None or sig is None else f"{(p50 - obs) / sig:+.2f}"
        two = "" if z == "" else ("yes" if abs(p50 - obs) <= 2 * sig else "no")   # median within 2 sigma_obs
        met = lambda e: "" if not e else ("yes" if ((p50 > obs) if e == "too little loss" else (p50 < obs)) else "no")
        exp, e11 = r.get("expected", ""), r.get("expected_d11", "")
        out.append([r["kind"], r["name"], r.get("date_start", ""), r.get("date_end", ""), r.get("observed", ""),
                    r.get("sigma", ""), r.get("unit", ""), f"{p5:.5g}", f"{p50:.5g}", f"{p95:.5g}", inside, z, two,
                    exp, met(exp), len(m), e11, met(e11)] + [first.get(r["name"], {}).get(c, "") for c in FIRST])
    write_csv(RES / f"{P}_validation.csv", ["kind", "name", "date_start", "date_end", "observed", "sigma", "unit",
              "p5", "p50", "p95", "band", "z_median", "within_2sigma", "expected", "expected_met", "n", "expected_d11",
              "expected_d11_met"] + [f"first_{c}" for c in FIRST], out)
    for o in out:
        print(*o, sep=" | ")


def dec8_run(r):
    """Run folder of the draw's decision-8 path (2021.75-2026.75 from the 2021 DEM): f5, the F3 _21 run of its sample
    (the f3_r1fix rerun where one exists); first ensemble, the step-4 run."""
    if P == "step5":
        return IGM / "runs" / f"s4_r{r['round']}" / f"{r['sample']}_21"
    fix = IGM / "runs" / "f3_r1fix" / f"{r['sample']}_21"
    return fix if fix.exists() else IGM / "runs" / f"f3_r{r['round']}" / f"{r['sample']}_21"


def states():
    """s03 inputs/P_state2026.npz: per draw the 2026-09-30 state of the hindcast (thk_end at 2026.75, topg) and of
    the decision-8 path (dec8_run: thk_end, topg)."""
    D = draws()
    out = {k: [] for k in ("thk_hind", "topg_hind", "thk_dec8", "topg_dec8")}
    for d, r in enumerate(D):
        with np.load(IGM / "runs" / HIND / f"h{d:03d}" / "fields.npz") as F:
            out["thk_hind"].append(F["thk_end"]), out["topg_hind"].append(F["topg"])
        with np.load(dec8_run(r) / "fields.npz") as F:
            out["thk_dec8"].append(F["thk_end"]), out["topg_dec8"].append(F["topg"])
    np.savez_compressed(INP / f"{P}_state2026.npz", draw=np.arange(len(D)), sample=np.array([r["sample"] for r in D]),
                        **{k: np.array(v, np.float32) for k, v in out.items()})
    V = lambda a: np.array(a).sum(axis=(1, 2)) * 1e-7
    print(f"inputs/{P}_state2026.npz: V_2026 hindcast median {np.median(V(out['thk_hind'])):.5f} km3, decision 8 "
          f"median {np.median(V(out['thk_dec8'])):.5f} km3")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"targets": targets, "scan": scan, "hind": hind, "validate": validate, "states": states,
     "table": secant_table}.get(
        cmd, lambda: None)()
    if cmd == "spin":
        spin(int(sys.argv[2]))
    elif cmd == "reduce":
        reduce(sys.argv[2])
