"""Independent check of DESIGN.md step 6: rebuilds the step-6 gate from the banked run tar and the raw no-flow outputs;
imports neither projection.py nor track.py nor noflow.py, and compares with the reduced files at the end.
Fix round F6: prefix f6 (results/f6_members.csv, posterior_f3_128.csv, f5_state2026.npz thk_dec8, inputs/f6_start2026,
runs_f6.tar.gz with stages f6_proj, f6_rerun, f6_prep; no extra verify stage); the no-flow members follow the IGM stop
rule (noflow.py), so a no-flow series ends at 2100 or at its first save without ice. The first ensemble's names
(step6, s6_*) and its checks of the uncut no-flow series are in the git history. Below, step6/s6 read f6.

  members  results/step6_members.csv: 9984 = 128 draws x 26 chains x 3 SSPs plus 78 u = 0 members, names unique,
           seeds 2100000 + k unique; 32 draws per (level, station); the draw's parameters of results/posterior_128.csv
  runs     streams runs_s6.tar.gz once; per run of s6_proj:
           .hydra/overrides.yaml against the member (restart, u, balance draw, f, station, chain, ssp, level, seed,
           net of its own start state, 0 iterations); logs/s6_proj/runs.csv one line per run, exit code 0;
           series.csv years 2026..end contiguous; track_end.txt 'no ice' with A1 = 0 at the last save and ice the save
           before, or 'end time' at 2100.75; NaN or inf in every numeric column (tongue, speeds and length may be nan
           or empty only without ice) and every array of fields.npz;
           start V and A1 against the start state (step5_state2026.npz thk_dec8, or s6_prep map_u0_21 thk_end);
           runaway: largest V / V_2026, largest A1, largest thk, largest surface speed;
           ice-free grid (last_ice): values -1 or 2026..end year; for every year Y, cells with last_ice >= Y at least
           A1(Y) / 1e-4 km2, equal at the end year; every start ice cell >= 2026; the same grid as icefree_s6_proj.npz;
           film members: last_ice rebuilt from thk_yYYYY exactly and A1 of every thk_yYYYY against the series;
           disappearance years (0.01, 0.1 km2: the year after the last year at or above, if the last save is below)
           against results/step6_runs_s6_proj.csv
  reruns   the vr_ runs of runs_s6_verify.tar.gz (stage s6_verify, stages/step6_verify.txt) and the rr_ runs of
           s6_rerun against the same members of s6_proj: series every column but wall_s, and every array of
           fields.npz, identical
  noflow   results/step6/noflow_K.csv, noflow_K_last_ice.npz, stages/step6_noflow_K.csv: per member 2026..2100, NaN,
           parameters and start against the IGM run, start V and A1 equal to the IGM run's, ice-free grid (as above on
           the computed cells; equal at 2100), start ice inside the computed cells; members whose A1 falls to 0 and
           comes back; IGM minus no-flow: first-year Ba (same surface, same weather), disappearance year (none =
           2101) paired; against results/step6_noflow.csv
Output: results/step6_verify.csv (part, quantity, value, threshold, pass); printed.
usage (s02, agl env, igm/):  python checks/step6_verify.py
"""
import csv
import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
RES, INP, STAGES = IGM / "results", IGM / "inputs", IGM / "stages"
CELL = 1e-4                        # km2 per 10 m cell
YEARS = np.arange(2026, 2101)
P6, POST, STATE5, START_DIR = "f6", "posterior_f3_128.csv", "f5_state2026.npz", "inputs/f6_start2026"
NUM = ("V_km3", "A01_km2", "A1_km2", "A2_km2", "Ba", "Bw", "Bs", "A1_in1850_km2", "A1_out1850_km2")
ICEONLY = ("tongue_m", "v_med", "v_p90", "v_max", "its_median")   # nan or empty without ice


def rcsv(path):
    return list(csv.DictReader(open(path)))


def fnum(x):
    return np.nan if x in ("", "nan") else float(x)


def dis(Y, A, lim):
    """Year after the last year with A >= lim, if the last value is below lim; None if not by the end."""
    if A[-1] >= lim:
        return None
    up = np.flatnonzero(A >= lim)
    return int(Y[0]) if not len(up) else int(Y[up[-1]]) + 1


def members(out):
    M = rcsv(RES / f"{P6}_members.csv")
    post = {int(r["draw"]): r for r in rcsv(RES / POST)}
    reg = [m for m in M if m["map_u0"] == "0"]
    u0 = [m for m in M if m["map_u0"] == "1"]
    mapd = str(max(post, key=lambda d: float(post[d]["weight"])))    # the largest-weight draw
    trip = {(m["draw"], m["chain_index"], m["ssp"]) for m in reg}
    seeds = [int(m["seed"]) for m in reg]                        # u = 0 members reuse the MAP draw's seeds
    twin = {(m["chain"], m["ssp"]): m["seed"] for m in reg if m["draw"] == mapd}
    seed_u0 = sum(m["seed"] != twin[(m["chain"], m["ssp"])] for m in u0)
    combo = {}
    for m in reg:
        combo.setdefault((m["level"], m["station"]), set()).add(m["draw"])
    bad = sum(1 for m in M if post[int(m["draw"])]["sample"] != m["sample"] or float(post[int(m["draw"])]["f"])
              != float(m["f"]) or float(m["u"]) != (0.0 if m["map_u0"] == "1" else float(post[int(m["draw"])]["u"])))
    bad += sum(1 for m in u0 if m["draw"] != mapd)
    bad += sum(1 for m in u0 if (m["level"], m["station"]) != ("anchored", "OTL"))
    ok = (len(M) == len({m["member"] for m in M}) == 10062 and len(reg) == len(trip) == 128 * 26 * 3 and len(u0) == 78
          and sorted(seeds) == list(range(2100000, 2100000 + 9984)) and seed_u0 == 0 and all(len(v) == 32 for v in combo.values())
          and len(combo) == 4 and bad == 0)
    out.append(("members", "members, (draw, chain, SSP) triples, u = 0 members; seeds 2100000 + k of the 9984, u = 0 "
                "members with a seed other than their MAP-draw twin's; draws per (level, station); parameter mismatches "
                f"against {POST} (MAP draw {mapd}, anchored OTL)",
                f"{len(M)}, {len(trip)}, {len(u0)}; {'unique and contiguous' if sorted(seeds) == list(range(2100000, 2109984)) else 'NO'}, {seed_u0}; "
                f"{sorted(len(v) for v in combo.values())}; {bad}", "10062, 9984, 78; contiguous, 0; 32 each; 0", ok))
    return {m["member"]: m for m in M}, post


def overrides_ok(ov, m, post):
    o = dict(x[2:].split("=", 1) for x in ov.splitlines() if x.startswith("- ") and "=" in x)
    p = post[int(m["draw"])]
    want = {"inputs.geometry.restart": m["start"], "processes.clim_otl.scenario.station": m["station"],
            "processes.clim_otl.scenario.chain": m["chain"], "processes.clim_otl.scenario.ssp": m["ssp"],
            "processes.clim_otl.scenario.level": m["level"], "processes.clim_otl.scenario.seed": m["seed"],
            "processes.warmstart.net": m["net"], "processes.warmstart.nbit": "0",
            "processes.iceflow.unified.nbit_init": "0"}
    bad = [k for k, v in want.items() if o.get(k) != v]
    numw = {"inputs.geometry.u": m["u"], "processes.clim_otl.dT": p["dT"], "processes.clim_otl.dg": p["dg"],
            "processes.clim_otl.b": p["b"], "processes.clim_otl.c_prec": p["c_prec"], "processes.smb_ti.Fm": p["Fm"],
            "processes.iceflow.physics.viscosity.arrhenius": p["arrhenius"],
            "processes.iceflow.physics.sliding.tau_ref": p["tau_ref"], "processes.clim_otl.dT_pre": p["dT_pre"]}
    bad += [k for k, v in numw.items() if k not in o or abs(float(o[k]) - float(v)) > 1e-9 * max(1, abs(float(v)))]
    if m["map_u0"] == "1" and abs(float(o.get("inputs.geometry.u", 1))) > 0:
        bad.append("u0")
    if m["start"] != (f"{START_DIR}/map_u0.npz" if m["map_u0"] == "1" else f"{START_DIR}/d{int(m['draw']):03d}.npz"):
        bad.append("start")
    return bad


def read_run(files):
    s = list(csv.DictReader(io.StringIO(files["series.csv"].decode())))
    F = np.load(io.BytesIO(files["fields.npz"]))
    return s, {k: F[k] for k in F.files}, files["track_end.txt"].decode().split(), files[".hydra/overrides.yaml"].decode()


def check_run(name, s, F, end, ov, m, post, start):
    """Per-run checks; returns (problems, summary)."""
    pr = []
    Y = np.array([int(r["year"]) for r in s])
    if not (Y[0] == 2026 and np.array_equal(Y, np.arange(2026, Y[-1] + 1))):
        pr.append("years")
    V, A = (np.array([fnum(r[c]) for r in s]) for c in ("V_km3", "A1_km2"))
    nan = 0
    for i, r in enumerate(s):
        for c in NUM:
            if c in ("Ba", "Bw", "Bs") and i == 0:
                continue
            nan += not np.isfinite(fnum(r[c]))
        if A[i] > 0:
            nan += sum(not np.isfinite(fnum(r[c])) for c in ICEONLY)
    nan += sum(int((~np.isfinite(v)).sum()) for v in F.values() if v.dtype.kind == "f")
    if nan:
        pr.append(f"nan {nan}")
    reason, t = " ".join(end[:-2]), float(end[-2][2:])
    if reason == "no ice":
        if not (A[-1] == 0 and A[-2] > 0 and abs(t - (Y[-1] + 0.75)) < 1e-6 and Y[-1] <= 2100):
            pr.append("no-ice end")
    elif reason == "end time":
        if not (Y[-1] == 2100 and abs(t - 2100.75) < 1e-6 and A[-1] > 0):
            pr.append("end time")
    else:
        pr.append(f"end {reason}")
    if ov is not None and m is not None:
        b = overrides_ok(ov, m, post)
        if b:
            pr.append("overrides " + ",".join(b))
    li, th = F["last_ice"], F["thk_end"]
    if start is not None:
        if abs((start >= 1).sum() * CELL - A[0]) > 5e-5 or abs(start.sum() * 1e-7 - V[0]) > 1e-6:
            pr.append("start V/A")
        if ((start >= 1) & (li < 2026)).any():
            pr.append("start ice without last_ice")
    if ((li != -1) & ((li < 2026) | (li > Y[-1]))).any():
        pr.append("last_ice range")
    cnt = np.array([(li >= y).sum() for y in Y])
    if (cnt + 0.5 < np.round(A / CELL)).any() or (li == Y[-1]).sum() != round(A[-1] / CELL) or \
            ((th >= 1) != (li == Y[-1])).any():
        pr.append("last_ice vs series")
    film = sorted(k for k in F if k.startswith("thk_y"))
    if film:
        L = np.full(li.shape, -1, np.int16)
        for k in film:
            y = int(k[5:])
            L[F[k] >= 1] = y
            if abs((F[k] >= 1).sum() * CELL - A[Y == y][0]) > 5e-5:
                pr.append(f"film A {y}")
        if not np.array_equal(L, li) or len(film) != len(Y):
            pr.append("film last_ice")
    sm = dict(Y=Y, V=V, A=A, Ba1=fnum(s[1]["Ba"]) if len(s) > 1 else np.nan, reason=reason,
              d001=dis(Y, A, 0.01), d01=dis(Y, A, 0.1), vratio=np.nanmax(V) / V[0], amax=np.nanmax(A),
              thmax=float(th.max()) if not film else max(float(F[k].max()) for k in film),
              vmax=np.nanmax([fnum(r["v_max"]) for r in s]),
              h=hashlib.sha1(li.astype(np.int16).tobytes()).hexdigest(), film=bool(film))
    return pr, sm


def same(a, b):
    sa, Fa = a
    sb, Fb = b
    ser = len(sa) == len(sb) and all(ra[c] == rb[c] for ra, rb in zip(sa, sb) for c in ra if c != "wall_s")
    fld = Fa.keys() == Fb.keys() and all(np.array_equal(Fa[k], Fb[k]) for k in Fa)
    return ser, fld


def stream(path, keep):
    """{(stage, name): {relative file: bytes}} of the run folders of path whose stage is in keep; plus logs."""
    runs, logs = {}, {}
    with tarfile.open(path, "r|gz") as tf:
        for ti in tf:
            if not ti.isfile():
                continue
            p = ti.name.split("/")
            if p[0] == "runs" and p[1] in keep:
                runs.setdefault((p[1], p[2]), {})["/".join(p[3:])] = tf.extractfile(ti).read()
            elif p[0] == "logs" and len(p) == 3 and p[2] == "runs.csv":
                logs[p[1]] = tf.extractfile(ti).read().decode()
    return runs, logs


def runs(out, M, post):
    S = np.load(INP / STATE5)
    thk8 = {int(d): S["thk_dec8"][i] for i, d in enumerate(S["draw"])}
    R, logs = stream(IGM / f"runs_{P6}.tar.gz", {f"{P6}_proj", f"{P6}_rerun", f"{P6}_prep"})
    starts = dict(thk8)
    starts["u0"] = np.load(io.BytesIO(R[(f"{P6}_prep", "map_u0_21")]["fields.npz"]))["thk_end"]
    log = list(csv.DictReader(io.StringIO(logs[f"{P6}_proj"])))
    names = [r["name"] for r in log]
    out.append(("runs", f"runs.csv of {P6}_proj: lines, distinct names, exit code not 0, names not members",
                f"{len(names)}, {len(set(names))}, {sum(r['rc'] != '0' for r in log)}, {len(set(names) - set(M))}",
                "10062, 10062, 0, 0", len(names) == len(set(names)) == 10062 and all(r["rc"] == "0" for r in log)
                and set(names) == set(M)))
    proj = {n: f for (st, n), f in R.items() if st == f"{P6}_proj"}
    miss = [n for n in M if n not in proj or not {"series.csv", "fields.npz", "track_end.txt"} <= proj[n].keys()]
    out.append(("runs", "run folders in the tar with series.csv, fields.npz, track_end.txt", f"{len(M) - len(miss)} of "
                f"{len(M)}", "10062", not miss))
    SM, probs, data = {}, {}, {}
    for n in sorted(proj):
        if n in miss:
            continue
        s, F, end, ov = read_run(proj[n])
        m = M[n]
        st = starts["u0"] if m["map_u0"] == "1" else starts[int(m["draw"])]
        pr, sm = check_run(n, s, F, end, ov, m, post, st)
        SM[n] = sm
        if pr:
            probs[n] = pr
        data[n] = (s, F) if sm["film"] or n in WANT else None
    kinds = {}
    for n, pr in probs.items():
        for p in pr:
            kinds.setdefault(p.split()[0], []).append(n)
    out.append(("runs", "runs with a problem (years, NaN, end reason, overrides, start V/A, ice-free grid, film "
                "fields): kinds and examples", "; ".join(f"{k} {len(v)} ({v[0]})" for k, v in kinds.items()) or "none",
                "none", not probs))
    out.append(("runs", "NaN or inf values (series, fields)", str(sum(int(p.split()[1]) for pr in probs.values()
                                                                      for p in pr if p.startswith("nan"))), "0",
                not any(p.startswith("nan") for pr in probs.values() for p in pr)))
    ends = [sm["reason"] for sm in SM.values()]
    out.append(("runs", "end states: no ice, end time 2100.75; film members with yearly fields",
                f"{ends.count('no ice')}, {ends.count('end time')}; {sum(sm['film'] for sm in SM.values())}", "sum 10062; 3",
                ends.count("no ice") + ends.count("end time") == 10062))
    vr = np.array([sm["vratio"] for sm in SM.values()])
    top = max(SM, key=lambda n: SM[n]["vratio"])
    out.append(("runs", "runaway: largest V / V_2026 (member), members with V / V_2026 > 1 at any save, largest A1 km2, "
                "largest thk m, largest surface speed m a-1",
                f"{vr.max():.2f} ({top}, {M[top]['level']} {M[top]['station']} {M[top]['ssp']}); {(vr > 1).sum()}; "
                f"{max(sm['amax'] for sm in SM.values()):.4f}; {max(sm['thmax'] for sm in SM.values()):.1f}; "
                f"{max(sm['vmax'] for sm in SM.values()):.1f}", "reported", ""))
    with np.load(RES / P6 / f"icefree_{P6}_proj.npz") as L:
        LI = L["last_ice"]                                       # read once (an NpzFile decompresses per access)
        hh = {str(n): hashlib.sha1(LI[i].astype(np.int16).tobytes()).hexdigest() for i, n in enumerate(L["name"])}
    dif = [n for n in SM if hh.get(n) != SM[n]["h"]]
    out.append(("runs", f"ice-free grids of icefree_{P6}_proj.npz identical to last_ice of the run folders",
                f"{len(SM) - len(dif)} of {len(SM)} ({len(hh)} grids in the file)", "10062", not dif and len(hh) == 10062))
    P = {r["name"]: r for r in rcsv(RES / f"{P6}_runs_{P6}_proj.csv")}
    dd = [n for n in SM if (str(SM[n]["d001"] or ""), str(SM[n]["d01"] or "")) != (P[n]["dis_001"], P[n]["dis_01"])
          or P[n]["end"] != SM[n]["reason"]]
    out.append(("runs", f"disappearance years 0.01 and 0.1 km2 and end reason against {P6}_runs_{P6}_proj.csv",
                f"{len(SM) - len(dd)} of {len(SM)} agree" + (f" (first {dd[0]})" if dd else ""), "all", not dd))
    # reruns
    rows = []
    for (st, n), f in R.items():
        if st == f"{P6}_rerun":
            s, F, _, _ = read_run(f)
            rows.append((n, n[3:], (s, F)))
    V = stream(IGM / f"runs_{P6}_verify.tar.gz", {f"{P6}_verify"})[0] if (IGM / f"runs_{P6}_verify.tar.gz").exists() else {}
    for (st, n), f in V.items():
        s, F, end, ov = read_run(f)
        pr, sm = check_run(n, s, F, end, ov, M[n[3:]], post, None)
        rows.append((n, n[3:], (s, F)))
        if pr:
            probs[n] = pr
    for n, o, new in sorted(rows):
        ser, fld = same(new, data[o]) if data.get(o) else (None, None)
        Yn = np.array([int(r["year"]) for r in new[0]])
        An = np.array([fnum(r["A1_km2"]) for r in new[0]])
        out.append(("reruns", f"{n} against {o}: disappearance year 0.01 | 0.1 km2 (first, rerun); series identical "
                    "(all columns but wall_s); every fields.npz array identical",
                    f"{SM[o]['d001']}, {dis(Yn, An, 0.01)} | {SM[o]['d01']}, {dis(Yn, An, 0.1)}; {ser}; {fld}",
                    "same; True; True", SM[o]["d001"] == dis(Yn, An, 0.01) and ser is True and fld is True))
    return SM


WANT = {line.split()[0][3:] for line in open(STAGES / f"{P6}_rerun.txt") if line.startswith("rr_")}   # rerun members


def noflow(out, M, SM, post):
    S = np.load(INP / STATE5)
    thk8 = {int(d): S["thk_dec8"][i] for i, d in enumerate(S["draw"])}
    probs, NF, keepmiss, back = {}, {}, 0, []
    K = 0
    while (RES / P6 / f"noflow_{K}.csv").exists():
        batch = rcsv(STAGES / f"{P6}_noflow_{K}.csv")
        per = {}
        for r in rcsv(RES / P6 / f"noflow_{K}.csv"):
            per.setdefault(int(r["member"]), []).append(r)
        L = np.load(RES / P6 / f"noflow_{K}_last_ice.npz")
        keep, LI = L["keep"], L["last_ice"]
        for i, b in enumerate(batch):
            n, s, li = b["member"], per[i], LI[i]
            m, pr = M[n], []
            p = post[int(m["draw"])]
            if any(abs(float(b[k]) - float(p[k])) > 0 for k in ("Fm", "c_prec", "dT", "dg", "b")) or \
                    abs(float(b["u"]) - float(m["u"])) > 0 or \
                    any(b[k] != m[k] for k in ("station", "chain", "ssp", "level", "seed")) or b["restart"] != m["start"]:
                pr.append("parameters")
            Y = np.array([int(r["date"][:4]) for r in s])               # 1 October Y: the save at Y.75
            V, A = (np.array([fnum(r[c]) for r in s]) for c in ("V_km3", "A1_km2"))
            if not np.array_equal(Y, np.arange(2026, Y[-1] + 1)) or any(r["date"][4:] != "-10-01" for r in s):
                pr.append("years")
            if (Y[-1] < 2100 and A[-1] != 0) or (A[1:-1] == 0).any():   # stop rule: ends at its first save without ice
                pr.append("stop rule")
            nan = int((~np.isfinite(V)).sum() + (~np.isfinite(A)).sum()) + sum(   # Ba needs ice at the year's start
                not np.isfinite(fnum(s[j]["Ba"])) for j in range(1, len(s)) if A[j - 1] > 0)
            if nan:
                pr.append(f"nan {nan}")
            ig = SM[n]
            if abs(V[0] - ig["V"][0]) > 0 or abs(A[0] - ig["A"][0]) > 0:
                pr.append("start")
            if ((li != -1) & ((li < 2026) | (li > Y[-1]))).any():
                pr.append("last_ice range")
            cnt = np.array([(li >= y).sum() for y in Y])
            if (cnt + 0.5 < np.round(A / CELL)).any() or (li == Y[-1]).sum() != round(A[-1] / CELL):
                pr.append("last_ice vs series")
            z = np.flatnonzero(A == 0)
            if len(z) and (A[z[0]:] >= 0.01).any():
                back.append(n)
            NF[n] = dict(Y=Y, V=V, A=A, Ba1=fnum(s[1]["Ba"]), d001=dis(Y, A, 0.01), d01=dis(Y, A, 0.1))
            if pr:
                probs[n] = pr
        for d in {int(M[b["member"]]["draw"]) for b in batch}:     # start ice inside the file's computed cells
            keepmiss += int(((thk8[d] >= 1) & ~keep).sum())
        K += 1
    reg = [n for n in M if M[n]["map_u0"] == "0"]
    out.append(("noflow", "no-flow members from 10 files, members 2026-2100 without a problem (parameters, years, NaN, "
                "start equal to IGM, ice-free grid)", f"{len(NF)}; {len(NF) - len(probs)}"
                + (f" ({'; '.join(f'{n}: {p}' for n, p in list(probs.items())[:3])})" if probs else ""),
                "9984; 9984", len(NF) == 9984 and not probs and set(NF) == set(reg)))
    out.append(("noflow", "start ice cells outside the computed cells of their file (summed over the files)", str(keepmiss), "0",
                keepmiss == 0))
    out.append(("noflow", "no-flow members whose A1 falls to 0 and later returns to >= 0.01 km2 (cut by the stop rule)",
                f"{len(back)}" + (f" (e.g. {back[0]})" if back else ""), "0", not back))
    Nr = {r["member"]: r for r in rcsv(RES / f"{P6}_noflow.csv")}
    dd = [n for n in NF if (str(NF[n]["d001"] or ""), str(NF[n]["d01"] or "")) != (Nr[n]["dis_001"], Nr[n]["dis_01"])]
    out.append(("noflow", f"disappearance years against {P6}_noflow.csv", f"{len(NF) - len(dd)} of {len(NF)} agree",
                "all", not dd))
    dba = np.array([abs(SM[n]["Ba1"] - NF[n]["Ba1"]) for n in NF])
    out.append(("noflow", "IGM minus no-flow, first year (2027) Ba over the same start ice: largest |difference| m w.e., "
                "members above 1e-4", f"{dba.max():.2e}; {(dba > 1e-4).sum()}", "print rounding (1e-5)",
                dba.max() <= 2e-5))
    D = np.array([(SM[n]["d001"] or 2101) - (NF[n]["d001"] or 2101) for n in NF])
    out.append(("noflow", "IGM minus no-flow disappearance year 0.01 km2 (none = 2101), years: 5/50/95 %, min, max; "
                "members IGM later, equal, earlier",
                f"{np.percentile(D, 5):.0f} / {np.median(D):.0f} / {np.percentile(D, 95):.0f}, {D.min()}, {D.max()}; "
                f"{(D > 0).sum()}, {(D == 0).sum()}, {(D < 0).sum()}", "reported", ""))
    keep2100 = sum(NF[n]["A"][-1] > 0 for n in NF)
    out.append(("noflow", "no-flow members never below 0.01 km2 for good (no disappearance year); IGM",
                f"{sum(NF[n]['d001'] is None for n in NF)}; {sum(SM[n]['d001'] is None for n in NF)}", "reported", ""))
    out.append(("noflow", "no-flow members with ice (>= 1 m) in 2100; IGM members (all 10062) running to 2100",
                f"{keep2100}; {sum(sm['reason'] == 'end time' for sm in SM.values())}", "reported", ""))


def main():
    out = []
    M, post = members(out)
    SM = runs(out, M, post)
    noflow(out, M, SM, post)
    with open(RES / f"{P6}_verify.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["part", "quantity", "value", "threshold", "pass"])
        w.writerows(out)
    for r in out:
        print(*r, sep=" | ")


if __name__ == "__main__":
    main()
