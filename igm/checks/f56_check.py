"""Independent check of the fix-round steps F5 (spin-up, hindcast, 2026 states) and F6 (projections) of DESIGN.md.
Own code: imports none of hindcast.py, projection.py, noflow.py, track.py, step5_verify.py, step6_verify.py. Reads the
banked tars and raw outputs, rebuilds the values, and compares them with the reduced files.

  tar      (s02, agl env, igm/): streams runs_f5.tar.gz, runs_f6.tar.gz, nets_f6.tar.gz and the F3 2021-2026 series
           of runs_f3.tar.gz; reads inputs/igm_static.npz, step5_targets.npz, f5_state2026.npz, results/f6/
           icefree_f6_proj.npz, results/f6/noflow_K.csv with stages/f6_noflow_K.csv.
           F5 spin-up: per run of f5_spin1, f5_spin2 the means of V and of the area inside SGI1850 over the model years
             1791-1820 and 1821-1850 of series.csv, steady (|V_last / V_prev - 1| <= 1 %), in tolerance
             (|A_in - 4.008| <= 0.25 km2); acceptance rebuilt (first steady run in tolerance, rounds in order); the
             restart of each round-2 run and of each hindcast run checked against it.
           F5 hindcast: from fields.npz the ice area (thk >= 1 m, 1e-4 km2 per cell) at the four inventory dates, the
             volume at the eleven survey dates, the mean surface (topg + thk) on the observed cells, the tongue (lowest
             surface of the largest 4-connected body) of thk_y2000, y2010, y2020, the ice on the outermost row or
             column and outside posbal at every kept field; tongue and length rows from series.csv; the run's
             parameters against posterior_f3_128.csv (A = 78 f, tau_ref = 0.2154 f^(-1/3)).
           2026 states: thk_dec8 of f5_state2026.npz against inputs/f6_start2026/dNNN.npz (nets_f6.tar.gz) and the
             hindcast thk_end against thk_hind; per draw V, A1, largest-body area, and the overlap with the Sentinel-2
             2026-09-27 outlines (s2_2026 all polygons, s2_2026_body main body).
           F6: per run of f6_proj the end reason, years contiguous, NaN, start V and A1 against its start state,
             parameters against f6_members.csv, disappearance (A1 < 0.01 and < 0.1 km2 with no later return: the year
             after the last save at or above), half-volume years (first year with V <= V_2026 / 2; V <= V_2021 / 2
             from 2022 on through the draw's 2021-2026 run), last_ice (-1 or start..end year, every start ice cell
             kept, = end year on the end ice cells, cells with last_ice >= Y at least A1(Y)) and its identity with the
             collected grid; reruns rr_ against their members (series without wall_s, every field); no-flow twins:
             per member the same years from the raw noflow_K.csv, stop rule (no save after the first without ice).
           -> results/f56_spin.csv, f56_hind.csv, f56_start.csv, f56_proj.csv, f56_noflow.csv, f56_misc.csv
  compare  (Mac, igm/): the validation table rebuilt (observed areas from the SGI rasters, observed volume changes and
           sigma = |dV| sigma_B / |B_geod| from ../data/glamos_basodino_volumechange.csv; 5/50/95 % with linear
           interpolation over the 128 draws, band, z) against f5_validation.csv; the spin-up against f5_spinup.csv;
           the F6 gate against f6_runs_f6_proj.csv and f6_noflow.csv; the year quantiles per 2026 level and SSP for
           F6 and the first ensemble (step6_runs_s6_proj.csv) and the published projections (PUBLISHED below,
           copied from PUBLISHED_PROJECTIONS.md).
           -> results/f56_check.csv (part, quantity, this check, reported, difference, pass), results/f56_years.csv
usage:  s02:  nrun python checks/f56_check.py tar        Mac:  python checks/f56_check.py compare
"""
import csv
import io
import re
import sys
import tarfile
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
RES, INP = IGM / "results", IGM / "inputs"
CELL, VCELL = 1e-4, 1e-7                       # km2 per 10 m cell; km3 per m of ice on one cell
TARGET, TOL = 4.008, 0.25
SURVEYS = ["1907-10-01", "1929-08-13", "1949-10-07", "1961-09-19", "1971-09-03", "1985-09-10", "1991-09-18",
           "2002-09-16", "2008-08-29", "2013-08-21", "2018-09-01", "2024-08-23"]
GLAMOS_DATE = {"1907-10-01": "19079999", "1929-08-13": "19299999", "2018-09-01": "20189999"}
AREAS = {"SGI1931": ("1929-08-13", "sgi1931"), "SGI1973": ("1973-10-01", "sgi1973"),
         "SGI2010": ("2010-10-01", "sgi2010"), "SGI2016": ("2018-10-01", "sgi2016")}
SURF = {"surface 1991": "1991-09-18", "surface 2012": "2012-09-01", "surface 2021": "2021-09-30"}
NONE = 2101                                    # disappearance or half-volume not reached by 2100
# PUBLISHED_PROJECTIONS.md (compiled 2026-10-08): per scenario the median [min, max] over GCMs; None = '> 2100'
PUBLISHED = [
    ("Rounce 2023", "SSP1-2.6", "half_2021", 2033), ("Rounce 2023", "SSP2-4.5", "half_2021", 2030),
    ("Rounce 2023", "SSP5-8.5", "half_2021", 2032), ("Rounce 2023", "SSP1-2.6", "half_2026", 2037),
    ("Rounce 2023", "SSP2-4.5", "half_2026", 2033), ("Rounce 2023", "SSP5-8.5", "half_2026", 2034),
    ("OGGM 2023", "SSP1-2.6", "half_2021", 2039), ("OGGM 2023", "SSP2-4.5", "half_2021", 2033),
    ("OGGM 2023", "SSP5-8.5", "half_2021", 2035), ("OGGM 2023", "SSP1-2.6", "half_2026", 2042),
    ("OGGM 2023", "SSP2-4.5", "half_2026", 2040), ("OGGM 2023", "SSP5-8.5", "half_2026", 2038),
    ("OGGM 2023", "SSP1-2.6", "dis_001", None), ("OGGM 2023", "SSP2-4.5", "dis_001", 2097),
    ("OGGM 2023", "SSP5-8.5", "dis_001", 2058),
    ("Zekollari 2019 (RCP)", "SSP1-2.6", "half_2021", 2039), ("Zekollari 2019 (RCP)", "SSP2-4.5", "half_2021", 2039),
    ("Zekollari 2019 (RCP)", "SSP5-8.5", "half_2021", 2038), ("Zekollari 2019 (RCP)", "SSP1-2.6", "half_2026", 2045),
    ("Zekollari 2019 (RCP)", "SSP2-4.5", "half_2026", 2042), ("Zekollari 2019 (RCP)", "SSP5-8.5", "half_2026", 2041),
    ("Van Tricht 2025 (+1.5 K)", "SSP1-2.6", "dis_001", 2069), ("Van Tricht 2025 (+2.7 K)", "SSP2-4.5", "dis_001", 2061),
    ("Van Tricht 2025 (+4.0 K)", "SSP5-8.5", "dis_001", 2055)]


def read_csv(p):
    return list(csv.DictReader(open(p) if isinstance(p, Path) else io.StringIO(p)))


def write_csv(p, head, rows):
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)


def key(d):
    return "thk_" + d.replace("-", "")


def overrides(text):
    return dict(l[2:].split("=", 1) for l in text.splitlines() if l.startswith("- ") and "=" in l)


def largest(ice):
    from scipy.ndimage import label
    lab, n = label(ice)                         # 4-connected
    if not n:
        return ice
    c = np.bincount(lab.ravel())
    c[0] = 0
    return lab == np.argmax(c)


def series(text):
    """{column: float array} of a series.csv (empty -> nan)."""
    rows = read_csv(text)
    return {c: np.array([float(r[c]) if r[c] not in ("", None) else np.nan for r in rows]) for c in rows[0]}


def after_last_at_or_above(years, A, a):
    """Disappearance year: the year after the last save with A >= a; NONE if the last save is still at or above."""
    k = np.flatnonzero(A >= a)
    if not len(k):
        return int(years[0])
    return NONE if k[-1] == len(A) - 1 else int(years[k[-1] + 1])


def first_le(years, V, v):
    k = np.flatnonzero(V <= v)
    return int(years[k[0]]) if len(k) else NONE


# ------------------------------------------------------------------------------------------------------------ tar
def members_of(tf, pattern):
    """{run: {file: bytes}} for the regular files of the tar whose path matches pattern (groups: run, file)."""
    out = {}
    for m in tf:
        if not m.isfile():
            continue
        g = re.fullmatch(pattern, m.name)
        if g:
            out.setdefault(g.group(1), {})[g.group(2)] = tf.extractfile(m).read()
    return out


def tar():
    G = dict(np.load(INP / "igm_static.npz"))
    T5 = dict(np.load(INP / "step5_targets.npz"))
    S5 = dict(np.load(INP / "f5_state2026.npz"))
    post = read_csv(RES / "posterior_f3_128.csv")
    edge = np.zeros(G["posbal"].shape, bool)
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    misc = []
    misc += [("observed area " + n, G[m].sum() * CELL) for n, (_, m) in AREAS.items()]
    misc += [("observed area s2_2026 all", G["s2_2026"].sum() * CELL),
             ("observed area s2_2026 body", G["s2_2026_body"].sum() * CELL)]
    obs_cells = {"surface 1991": (G["mask_1994"] == 1, G["usurf_1994"]), "surface 2012": (T5["has12"], T5["s12"]),
                 "surface 2021": (G["mask_2021"] == 1, G["usurf_2021"])}
    misc += [("observed " + n, float(z[c].mean())) for n, (c, z) in obs_cells.items()]

    # F5 -------------------------------------------------------------------------------------------------------
    with tarfile.open(IGM / "runs_f5.tar.gz", "r|gz") as tf:
        R = members_of(tf, r"runs/(f5_spin\d/d\d{3}_r\d|f5_hind/h\d{3})/(series\.csv|fields\.npz|track_end\.txt|"
                           r"\.hydra/overrides\.yaml)")
    spin, chain = [], {}
    for run in sorted(k for k in R if k.startswith("f5_spin")):
        f = R[run]
        s = series(f["series.csv"].decode())
        o = overrides(f[".hydra/overrides.yaml"].decode())
        y = s["year"]
        last, prev = (y >= 1821) & (y <= 1850), (y >= 1791) & (y <= 1820)
        Vl, Vp = s["V_km3"][last].mean(), s["V_km3"][prev].mean()
        Ain = s["A1_in1850_km2"][last].mean()
        steady = abs(Vl / Vp - 1) <= 0.01
        ok = steady and abs(Ain - TARGET) <= TOL and int(y[-1]) == 1850 and last.sum() == 30 and prev.sum() == 30
        d, r = int(run.split("/")[1][1:4]), int(run[-1])
        with np.load(io.BytesIO(f["fields.npz"])) as F:
            Vend = float(F["thk_end"].sum()) * VCELL
            out_basin = int(((F["thk_end"] >= 1) & (G["posbal"] != 1)).sum())
        spin.append([d, r, run, float(o["processes.clim_otl.offset"]), o.get("inputs.geometry.restart", ""),
                     int(y[0]), int(y[-1]), Vl, Vp, Vl / Vp - 1, Ain, s["A1_out1850_km2"][last].mean(), steady, ok,
                     Vend, out_basin, f["track_end.txt"].decode().split()[0]])
        chain.setdefault(d, []).append((r, run, ok, o.get("inputs.geometry.restart", "")))
    acc = {}
    for d, runs in chain.items():
        runs.sort()
        acc[d] = next((run for r, run, ok, _ in runs if ok), "")
        for r, run, ok, rs in runs:                            # round R > 1 continues from round R - 1
            if r > 1:
                assert rs == f"runs/{runs[r - 2][1]}/fields.npz", (run, rs)
    write_csv(RES / "f56_spin.csv", ["draw", "round", "run", "dT_1850", "restart", "first_year", "last_year",
              "V_last_cycle", "V_prev_cycle", "V_change", "Ain_last_cycle", "Aout_last_cycle", "steady", "accepted_ok",
              "V_end", "end_cells_outside_basin", "end"], spin)

    hind = []
    for d in range(128):
        f = R[f"f5_hind/h{d:03d}"]
        s = series(f["series.csv"].decode())
        o = overrides(f[".hydra/overrides.yaml"].decode())
        p = post[d]
        fpar = float(p["f"])
        par_bad = sum([abs(float(o["inputs.geometry.u"]) - float(p["u"])) > 1e-6,
                       abs(float(o["processes.clim_otl.dT"]) - float(p["dT"])) > 1e-6,
                       abs(float(o["processes.clim_otl.dT_pre"]) - float(p["dT_pre"])) > 1e-6,
                       abs(float(o["processes.smb_ti.Fm"]) / float(p["Fm"]) - 1) > 1e-6,
                       abs(float(o["processes.clim_otl.c_prec"]) / float(p["c_prec"]) - 1) > 1e-6,
                       abs(float(o["processes.iceflow.physics.viscosity.arrhenius"]) / (78 * fpar) - 1) > 1e-3,
                       abs(float(o["processes.iceflow.physics.sliding.tau_ref"]) / (0.2154 * fpar ** (-1 / 3)) - 1)
                       > 1e-3])
        yr = {int(y): i for i, y in enumerate(s["year"])}
        row = {"draw": d, "sample": p["sample"], "f": fpar, "u": float(p["u"]), "par_bad": par_bad,
               "restart_ok": int(o["inputs.geometry.restart"] == f"runs/{acc[d]}/fields.npz"),
               "years_ok": int(list(s["year"]) == list(range(1850, 2027))),
               "end": f["track_end.txt"].decode().strip()[:20],
               "V_1850_series": s["V_km3"][0],
               "spin_V_end": next(x[14] for x in spin if x[2] == acc[d]),
               "nan_V_A": int(np.isnan(s["V_km3"]).sum() + np.isnan(s["A1_km2"]).sum())}
        with np.load(io.BytesIO(f["fields.npz"])) as F:
            for n, (dt, _) in AREAS.items():
                row[n] = (F[key(dt)] >= 1).sum() * CELL
            V = {dt: float(F[key(dt)].sum()) * VCELL for dt in SURVEYS}
            for a, b in zip(SURVEYS[:-1], SURVEYS[1:]):
                row[f"{a[:4]}-{b[:4]}"] = V[b] - V[a]
            for n, dt in SURF.items():
                c, zo = obs_cells[n]
                z = F["topg"] + F[key(dt)]
                row[n] = float(z[c].mean())
                row[n + " RMS"] = float(np.sqrt(((z - zo)[c] ** 2).mean()))
            for y in (2000, 2010, 2020):
                b = largest(F[f"thk_y{y}"] >= 1)
                row[f"tongue {y} own"] = float((F["topg"] + F[f"thk_y{y}"])[b].min()) if b.any() else np.nan
                row[f"tongue {y} series"] = s["tongue_m"][yr[y]]
            thk = [k for k in F.files if k.startswith("thk_")]
            row["edge cells max"] = max(int((F[k][edge] >= 1).sum()) for k in thk)
            row["outside basin max, saves"] = max(int(((F[k] >= 1) & (G["posbal"] != 1)).sum())
                                                  for k in thk if k.startswith("thk_y") or k == "thk_end")
            row["outside basin max, outline dates"] = max(int(((F[k] >= 1) & (G["posbal"] != 1)).sum())
                                                          for k in thk if k[4].isdigit())
            row["outside basin 1929-08-13"] = int(((F[key("1929-08-13")] >= 1) & (G["posbal"] != 1)).sum())
            row["V 2021-09-30"] = float(F[key("2021-09-30")].sum()) * VCELL
            row["A 2026-09-30"] = (F["thk_end"] >= 1).sum() * CELL
            row["body 2026-09-27"] = largest(F[key("2026-09-27")] >= 1).sum() * CELL
            row["V 1850 fields"] = float(F["thk_y1850"].sum()) * VCELL
            row["thk_hind equal"] = int(np.array_equal(F["thk_end"], S5["thk_hind"][d]))
        for y in range(2000, 2025):
            if y in yr:
                row[f"tongue {y}"] = s["tongue_m"][yr[y]]
        for a in (1929, 1961, 1986, 2024):
            row[f"length 1899-{a}"] = s["length_m"][yr[a]] - s["length_m"][yr[1899]]
        row["basin melt 1850-2026"] = float(np.nansum(s["basin_melt_km3"]))
        row["A_in 1850"], row["A_out 1850"] = s["A1_in1850_km2"][0], s["A1_out1850_km2"][0]
        hind.append(row)
    head = list(hind[0])
    for r in hind:
        head += [k for k in r if k not in head]
    write_csv(RES / "f56_hind.csv", head, [[r.get(k, "") for k in head] for r in hind])
    del R

    # F6 -------------------------------------------------------------------------------------------------------
    starts = {}
    with tarfile.open(IGM / "nets_f6.tar.gz", "r|gz") as tf:
        for m in tf:
            g = re.fullmatch(r"inputs/f6_start2026/(d\d{3}|map_u0)\.npz", m.name)
            if g and m.isfile():
                with np.load(io.BytesIO(tf.extractfile(m).read())) as F:
                    starts[g.group(1)] = (F["thk_end"].copy(), F["topg"].copy())
    body, alls = G["s2_2026_body"] > 0, G["s2_2026"] > 0
    st = []
    for d in range(128):
        t8 = S5["thk_dec8"][d]
        ice = t8 >= 1
        b = largest(ice)
        th, tg = starts[f"d{d:03d}"]
        st.append([d, post[d]["sample"], float(post[d]["u"]), float(post[d]["f"]), float(t8.sum()) * VCELL,
                   float(post[d]["V_2026"]), ice.sum() * CELL, b.sum() * CELL, (b & body).sum() / body.sum(),
                   (b & body).sum() / max(b.sum(), 1), (b & body).sum() / (b | body).sum(),
                   (ice & alls).sum() / alls.sum(), (ice & alls).sum() / max(ice.sum(), 1),
                   (ice & ~alls).sum() * CELL, (alls & ~ice).sum() * CELL,
                   int(np.array_equal(th, t8)), int(np.array_equal(tg, S5["topg_dec8"][d])),
                   float(S5["thk_hind"][d].sum()) * VCELL, (S5["thk_hind"][d] >= 1).sum() * CELL])
    write_csv(RES / "f56_start.csv", ["draw", "sample", "u", "f", "V_dec8", "V_2026_posterior", "A1_dec8",
              "body_dec8", "body_share_of_obs_body_covered", "body_share_inside_obs_body", "body_iou",
              "all_share_of_obs_all_covered", "all_share_inside_obs_all", "model_ice_outside_obs_km2",
              "obs_not_covered_km2", "start_file_thk_equal", "start_file_topg_equal", "V_hind", "A1_hind"], st)

    pre = {}                                               # sample -> (years, V) of its 2021-2026 run of F3
    with tarfile.open(IGM / "runs_f3.tar.gz", "r|gz") as tf:
        for m in tf:
            g = re.fullmatch(r"runs/(f3_r1fix|f3_r1|f3_r2)/(f3r\d_\d{4})_21/series\.csv", m.name)
            if g and m.isfile():
                s = series(tf.extractfile(m).read().decode())
                pre[(g.group(1), g.group(2))] = (s["year"], s["V_km3"])
    mem = {m["member"]: m for m in read_csv(RES / "f6_members.csv")}
    rnd = {p["sample"]: p["round"] for p in post}

    def pre_of(m, P):
        if m["map_u0"] == "1":
            return P
        smp = m["sample"]
        return pre[("f3_r1fix", smp)] if ("f3_r1fix", smp) in pre else pre[(f"f3_r{rnd[smp]}", smp)]

    collected = np.load(RES / "f6" / "icefree_f6_proj.npz")
    cidx = {n: i for i, n in enumerate(collected["name"])}
    cgrid = collected["last_ice"]
    proj, rr, orig, u0pre, defer = [], {}, {}, None, []
    with tarfile.open(IGM / "runs_f6.tar.gz", "r|gz") as tf:
        cur, files = None, {}
        def flush():
            nonlocal u0pre
            if cur is None:
                return
            stage, name = cur
            if stage == "f6_prep":
                if name == "map_u0_21" and "series.csv" in files:
                    s = series(files["series.csv"].decode())
                    u0pre = (s["year"], s["V_km3"])
                return
            if stage == "f6_rerun":
                rr[name[3:]] = files.copy()
                return
            if stage != "f6_proj":
                return
            if name in ("d103_c06_2", "d010_c01_5", "d093_c22_2", "d107_c03_2"):
                orig[name] = files.copy()
            if mem[name]["map_u0"] == "1":                 # needs f6_prep/map_u0_21, which may come later
                defer.append((name, files.copy()))
            else:
                proj.append(member_row(name, files))
        def member_row(name, f):
            m = mem[name]
            s = series(f["series.csv"].decode())
            o = overrides(f[".hydra/overrides.yaml"].decode())
            y, V, A = s["year"], s["V_km3"], s["A1_km2"]
            end = f["track_end.txt"].decode().strip()
            t_end = float(re.search(r"t=([\d.]+)", end).group(1))
            kind = "no ice" if end.startswith("no ice") else ("2100" if end.startswith("end time") else end)
            num = [c for c in s if c not in ("tongue_m", "v_med", "v_p90", "v_max", "its_median", "length_m")]
            nan = sum(int(np.isnan(s[c][1 if c in ("Ba", "Bw", "Bs") else 0:]).sum()) for c in num if c != "length_m")
            nan += sum(int(np.isnan(s[c][A > 0]).sum()) for c in ("tongue_m", "v_med", "v_p90", "v_max"))
            sk = "map_u0" if m["map_u0"] == "1" else f"d{int(m['draw']):03d}"
            th0, _ = starts[sk]
            fpar = float(m["f"])
            par_bad = sum([o["inputs.geometry.restart"] != m["start"], o["processes.warmstart.net"] != m["net"],
                           abs(float(o["inputs.geometry.u"]) - float(m["u"])) > 1e-6,
                           o["processes.clim_otl.scenario.chain"] != m["chain"],
                           o["processes.clim_otl.scenario.ssp"] != m["ssp"],
                           o["processes.clim_otl.scenario.level"] != m["level"],
                           o["processes.clim_otl.scenario.station"] != m["station"],
                           o["processes.clim_otl.scenario.seed"] != m["seed"],
                           o["processes.warmstart.nbit"] != "0",
                           abs(float(o["processes.iceflow.physics.viscosity.arrhenius"]) / (78 * fpar) - 1) > 1e-3])
            P = pre_of(m, u0pre)
            Vpre = np.r_[P[1][1:], V[1:]]
            Ypre = np.r_[P[0][1:], y[1:]]
            pre_ok = int(P[0][-1] == y[0] and abs(P[1][-1] - V[0]) < 1e-6)
            with np.load(io.BytesIO(f["fields.npz"])) as F:
                L, te = F["last_ice"], F["thk_end"]
                endy = int(y[-1])
                ok_vals = np.all((L == -1) | ((L >= 2026) & (L <= endy)))
                ok_start = np.all(L[th0 >= 1] >= 2026)
                ok_end = np.array_equal(L == endy, te >= 1) if A[-1] > 0 else not (L == endy).any()
                cnt = np.array([(L >= yy).sum() for yy in y.astype(int)])
                ok_cover = np.all(cnt * CELL >= A - 1e-6)
                same = int(np.array_equal(L, cgrid[cidx[name]]))
                fnan = int(np.isnan(te).sum())
            return [name, m["draw"], m["level"], m["ssp"], m["map_u0"], kind, t_end, int(y[0]), endy,
                    int(list(y) == list(range(int(y[0]), endy + 1))), nan + fnan, par_bad,
                    abs(V[0] - float(th0.sum()) * VCELL), abs(A[0] - (th0 >= 1).sum() * CELL), V[0], A[0], V[-1], A[-1],
                    int(A[-1] == 0) if kind == "no ice" else int(endy == 2100), int(kind != "no ice" or A[-2] > 0),
                    after_last_at_or_above(y, A, 0.01), after_last_at_or_above(y, A, 0.1),
                    first_le(y[1:], V[1:], V[0] / 2), first_le(Ypre, Vpre, P[1][0] / 2), P[1][0], pre_ok,
                    float(np.max(V) / V[0]), int(ok_vals and ok_start and ok_end and ok_cover), same,
                    float(np.nansum(s["basin_melt_km3"]))]
        for m in tf:
            g = re.fullmatch(r"runs/(f6_proj|f6_rerun|f6_prep)/([^/]+)/(series\.csv|fields\.npz|track_end\.txt|"
                             r"\.hydra/overrides\.yaml)", m.name)
            if not g or not m.isfile():
                continue
            if (g.group(1), g.group(2)) != cur:
                flush()
                cur, files = (g.group(1), g.group(2)), {}
            files[g.group(3)] = tf.extractfile(m).read()
        flush()
    proj += [member_row(n, f) for n, f in defer]
    write_csv(RES / "f56_proj.csv", ["name", "draw", "level", "ssp", "map_u0", "end", "t_end", "first_year",
              "last_year", "years_contiguous", "nan", "par_bad", "dV_start", "dA_start", "V_2026", "A_2026", "V_end",
              "A_end", "end_consistent", "ice_before_stop", "dis_001", "dis_01", "half_2026", "half_2021", "V_2021",
              "pre_link_ok", "max_V_ratio", "icefree_ok", "icefree_same_as_collected", "basin_melt_km3"], proj)
    for n in sorted(rr):
        a, b = orig[n], rr[n]
        sa, sb = read_csv(a["series.csv"].decode()), read_csv(b["series.csv"].decode())
        ser = len(sa) == len(sb) and all({k: v for k, v in x.items() if k != "wall_s"} ==
                                         {k: v for k, v in z.items() if k != "wall_s"} for x, z in zip(sa, sb))
        with np.load(io.BytesIO(a["fields.npz"])) as A_, np.load(io.BytesIO(b["fields.npz"])) as B_:
            fld = set(A_.files) == set(B_.files) and all(np.array_equal(A_[k], B_[k]) for k in A_.files)
        misc.append((f"rerun {n}: series (without wall_s) and fields identical", int(ser and fld)))
        misc.append((f"rerun {n}: dis_001 rerun", after_last_at_or_above(
            np.array([float(r["year"]) for r in sb]), np.array([float(r["A1_km2"]) for r in sb]), 0.01)))
    nf = []
    for k in range(10):
        names = [r["member"] for r in read_csv(IGM / "stages" / f"f6_noflow_{k}.csv")]
        rows = read_csv(RES / "f6" / f"noflow_{k}.csv")
        by = {}
        for r in rows:
            by.setdefault(int(r["member"]), []).append((int(r["date"][:4]), float(r["V_km3"]), float(r["A1_km2"]),
                                                        r["Ba"]))
        for i, n in enumerate(names):
            z = sorted(by[i])
            y = np.array([a[0] for a in z])
            V, A = np.array([a[1] for a in z]), np.array([a[2] for a in z])
            nanv = int(np.isnan(V).sum() + np.isnan(A).sum())
            zero = np.flatnonzero(A == 0)
            after_stop = int(len(zero) > 0 and zero[0] < len(A) - 1)
            ended = int((len(zero) > 0 and zero[0] == len(A) - 1) or y[-1] == 2100)
            nf.append([n, int(y[0]), int(y[-1]), ended, after_stop, nanv, V[0], A[0],
                       int(list(y) == list(range(int(y[0]), int(y[-1]) + 1))),
                       after_last_at_or_above(y, A, 0.01), after_last_at_or_above(y, A, 0.1),
                       first_le(y[1:], V[1:], V[0] / 2), z[1][3] if len(z) > 1 else ""])
    write_csv(RES / "f56_noflow.csv", ["name", "first_year", "last_year", "ended", "save_after_stop", "nan", "V_2026",
              "A_2026", "years_contiguous", "dis_001", "dis_01", "half_2026", "Ba_first"], nf)
    write_csv(RES / "f56_misc.csv", ["quantity", "value"], misc)
    print("written results/f56_spin.csv, f56_hind.csv, f56_start.csv, f56_proj.csv, f56_noflow.csv, f56_misc.csv")


# -------------------------------------------------------------------------------------------------------- compare
def q(x, p=(5, 50, 95)):
    return [float(v) for v in np.percentile(np.asarray(x, float), p)]


def yfmt(v):
    return "none" if v >= NONE else (f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}")


def compare():
    out = []
    add = lambda part, qty, mine, rep, diff, ok: out.append([part, qty, mine, rep, diff, ok])
    misc = {r["quantity"]: r["value"] for r in read_csv(RES / "f56_misc.csv")}
    # observed targets
    T = {r["name"]: r for r in read_csv(RES / "f5_targets.csv")}
    gl = [l for l in open(IGM.parent / "data" / "glamos_basodino_volumechange.csv", encoding="utf-8", errors="replace")
          if l.startswith("C14-10,")]
    gl = {(x[3], x[4]): x for x in (l.split(",") for l in gl)}
    for a, b in zip(SURVEYS[:-1], SURVEYS[1:]):
        n = f"{a[:4]}-{b[:4]}"
        x = gl[(GLAMOS_DATE.get(a, a.replace("-", "")), GLAMOS_DATE.get(b, b.replace("-", "")))]
        dv, sig = float(x[9]), abs(float(x[9])) * float(x[12]) / abs(float(x[11]))
        o, s = float(T[n]["observed"]), float(T[n]["sigma"])
        add("F5 observed", f"dV {n} (km3), sigma", f"{dv:.6f}, {sig:.6f}", f"{o:.6f}, {s:.6f}",
            f"{dv - o:.1e}, {sig - s:.1e}", abs(dv - o) < 1e-6 and abs(sig - s) < 1e-5)
    for n in AREAS:
        mine, rep = float(misc["observed area " + n]), float(T[n]["observed"])
        add("F5 observed", f"area {n} on the lattice (km2)", f"{mine:.4f}", T[n]["observed"], f"{mine - rep:+.4f}",
            abs(mine - rep) < 1e-3)
    for n in SURF:
        mine, rep = float(misc["observed " + n]), float(T[n]["observed"])
        add("F5 observed", f"mean {n} (m)", f"{mine:.3f}", T[n]["observed"], f"{mine - rep:+.3f}", abs(mine - rep) < 0.01)
    # spin-up
    S = read_csv(RES / "f56_spin.csv")
    rep = {int(r["draw"]): r for r in read_csv(RES / "f5_spinup.csv")}
    acc = {}
    for r in sorted(S, key=lambda r: (int(r["draw"]), int(r["round"]))):
        if r["accepted_ok"] == "True" and int(r["draw"]) not in acc:
            acc[int(r["draw"])] = r
    Ain = [float(acc[d]["Ain_last_cycle"]) for d in acc]
    add("F5 spin-up", "draws with an accepted run (steady and within 4.008 +- 0.25)", len(acc), len(rep), "",
        len(acc) == 128)
    add("F5 spin-up", "accepted in round 1 | round 2", f"{sum(acc[d]['round'] == '1' for d in acc)} | "
        f"{sum(acc[d]['round'] == '2' for d in acc)}", f"{sum(rep[d]['runs'] == '1' for d in rep)} | "
        f"{sum(rep[d]['runs'] == '2' for d in rep)}", "", True)
    dmax = max(abs(float(acc[d]["Ain_last_cycle"]) - float(rep[d]["Ain_km2"])) for d in acc)
    ddT = max(abs(float(acc[d]["dT_1850"]) - float(rep[d]["dT_1850"])) for d in acc)
    add("F5 spin-up", "A_in last cycle 5/50/95 %, min (km2); largest difference per draw; dT_1850 difference",
        " / ".join(f"{v:.3f}" for v in q(Ain)) + f", {min(Ain):.3f}", "3.863 / 3.975 / 3.995, 3.763",
        f"{dmax:.1e}; {ddT:.1e}", dmax < 1e-4 and ddT < 1e-6)
    ch = [100 * float(acc[d]["V_change"]) for d in acc]
    add("F5 spin-up", "last-cycle volume change min, max (%)", f"{min(ch):+.2f}, {max(ch):+.2f}", "-0.1 to +0.8", "",
        max(abs(c) for c in ch) <= 1)
    add("F5 spin-up", "ice cells outside the basin at the accepted spin-up end, max",
        max(int(acc[d]["end_cells_outside_basin"]) for d in acc), "0", "", True)
    H = read_csv(RES / "f56_hind.csv")
    add("F5 hindcast", "restart = accepted spin-up | parameters as posterior_f3_128 | years 1850-2026 | 1850 V = "
        "spin-up end V", f"{sum(int(h['restart_ok']) for h in H)} | {sum(h['par_bad'] == '0' for h in H)} | "
        f"{sum(int(h['years_ok']) for h in H)} | "
        f"{sum(abs(float(h['V_1850_series']) - float(h['spin_V_end'])) < 2e-6 for h in H)}", "128 each", "",
        all(int(h["restart_ok"]) and h["par_bad"] == "0" and int(h["years_ok"]) for h in H))
    # validation table
    V5 = {r["name"]: r for r in read_csv(RES / "f5_validation.csv")}
    rows = list(AREAS) + [f"{a[:4]}-{b[:4]}" for a, b in zip(SURVEYS[:-1], SURVEYS[1:])] + \
        [f"tongue {y}" for y in range(2000, 2025) if f"tongue {y}" in V5] + \
        [f"length 1899-{a}" for a in (1929, 1961, 1986, 2024)] + list(SURF)
    val = []
    for n in rows:
        m = np.array([float(h[n]) for h in H if h.get(n, "") not in ("", "nan")])
        m = m[np.isfinite(m)]
        p5, p50, p95 = q(m)
        r = V5[n]
        o = float(r["observed"])
        sig = float(r["sigma"]) if r["sigma"] else None
        band = "inside" if p5 <= o <= p95 else "outside"
        z = (p50 - o) / sig if sig else None
        rp = [float(r[c]) for c in ("p5", "p50", "p95")]
        d = max(abs(a - b) for a, b in zip((p5, p50, p95), rp))
        tol = (lambda b: 1e-6) if r["kind"] == "volume" else (lambda b: 1e-4 * abs(b))   # f5_runs: 6 decimals km3;
        close = all(abs(a - b) <= tol(b) for a, b in zip((p5, p50, p95), rp))           # f5_validation: 5 digits
        ok = close and band == r["band"] and len(m) == int(r["n"])
        val.append([r["kind"], n, o, sig if sig else "", f"{p5:.5g}", f"{p50:.5g}", f"{p95:.5g}", band,
                    "" if z is None else f"{z:+.2f}", "" if z is None else ("yes" if abs(z) <= 2 else "no"),
                    r["expected"], r["first_p50"], r["first_band"], len(m), ok])
        add("F5 validation", f"{n}: 5/50/95 %, band, z", f"{p5:.5g} / {p50:.5g} / {p95:.5g}, {band}"
            + ("" if z is None else f", {z:+.2f}"), f"{r['p5']} / {r['p50']} / {r['p95']}, {r['band']}"
            + (f", {r['z_median']}" if r["z_median"] else ""), f"{d:.1e}", ok)
    write_csv(RES / "f56_validation.csv", ["kind", "name", "observed", "sigma", "p5", "p50", "p95", "band",
              "z_median", "within_2sigma", "expected", "first_p50", "first_band", "n", "matches_f5_validation"], val)
    for a, b in (("1907", "1991"), ("1991", "2024")):
        ks = [f"{x[:4]}-{y[:4]}" for x, y in zip(SURVEYS[:-1], SURVEYS[1:]) if a <= x[:4] and y[:4] <= b]
        tot = [sum(float(h[k]) for k in ks) for h in H]
        obs = sum(float(T[k]["observed"]) for k in ks)
        add("F5 validation", f"sum of volume changes {a}-{b}, 5/50/95 % against observed (km3)",
            " / ".join(f"{v:.3f}" for v in q(tot)) + f" against {obs:.3f}",
            "-0.050 / -0.014 / -0.005 against -0.164" if a == "1907" else "-0.057 / -0.041 / -0.029 against -0.056",
            "", True)
    for k, rep_k in (("tongue 2000", "tongue 2000 series"), ("tongue 2010", "tongue 2010 series"),
                     ("tongue 2020", "tongue 2020 series")):
        own = np.array([float(h[k + " own"]) for h in H])
        ser = np.array([float(h[rep_k]) for h in H])
        add("F5 hindcast", f"{k}: lowest surface of the largest body against series.csv, largest difference (m)",
            f"{np.nanmax(np.abs(own - ser)):.2f}", "0", "", np.nanmax(np.abs(own - ser)) < 0.05)
    add("F5 hindcast", "ice cells on the domain edge at any kept field, max over draws",
        max(int(h["edge cells max"]) for h in H), "0", "", max(int(h["edge cells max"]) for h in H) == 0)
    add("F5 hindcast", "ice cells outside the basin: yearly saves | outline dates | 1929-08-13 (max over draws)",
        f"{max(int(h['outside basin max, saves']) for h in H)} | "
        f"{max(int(h['outside basin max, outline dates']) for h in H)} | "
        f"{max(int(h['outside basin 1929-08-13']) for h in H)}", "0 | 1-7 | 1-7", "",
        max(int(h["outside basin max, saves"]) for h in H) == 0)
    bm = [float(h["basin melt 1850-2026"]) for h in H]
    add("F5 hindcast", "basin melt 1850-2026, 5/50/95 % (km3)", " / ".join(f"{v:.3f}" for v in q(bm)),
        "0.014 / 0.022 / 0.039", "", True)
    add("F5 hindcast", "thk_end equal to f5_state2026 thk_hind", sum(int(h["thk_hind equal"]) for h in H), "128", "",
        all(int(h["thk_hind equal"]) for h in H))
    f = np.log([float(h["f"]) for h in H])
    for n in ("SGI1931", "SGI2016", "V 2021-09-30", "V_1850_series"):
        r = np.corrcoef(f, [float(h[n]) for h in H])[0, 1]
        add("F5 driver", f"correlation of ln f with {n}", f"{r:+.2f}", {"SGI1931": "-0.91", "SGI2016": "-0.87",
            "V 2021-09-30": "-0.96", "V_1850_series": "-0.98"}[n], "", True)
    # start states
    St = read_csv(RES / "f56_start.csv")
    col = lambda k: np.array([float(r[k]) for r in St])
    add("F5/F6 start", "start files equal to f5_state2026 thk_dec8, topg_dec8",
        f"{sum(int(r['start_file_thk_equal']) for r in St)}, {sum(int(r['start_file_topg_equal']) for r in St)}",
        "128, 128", "", all(int(r["start_file_thk_equal"]) and int(r["start_file_topg_equal"]) for r in St))
    add("F5/F6 start", "V_dec8 against posterior_f3_128 V_2026, largest difference (km3)",
        f"{np.max(np.abs(col('V_dec8') - col('V_2026_posterior'))):.1e}", "5e-7", "",
        np.max(np.abs(col("V_dec8") - col("V_2026_posterior"))) < 1e-6)
    obs_b, obs_a = float(misc["observed area s2_2026 body"]), float(misc["observed area s2_2026 all"])
    for k, lab, o in (("A1_dec8", "all ice >= 1 m", obs_a), ("body_dec8", "largest body", obs_b)):
        p = q(col(k))
        add("F5/F6 start", f"2026 {lab} 5/50/95 % against observed (km2)",
            " / ".join(f"{v:.3f}" for v in p) + f" against {o:.3f} ({'inside' if p[0] <= o <= p[2] else 'outside'})",
            "A1 0.842 / 0.980 / 1.112 against body 1.021" if k == "A1_dec8" else "", "", p[0] <= o <= p[2])
    for k, lab in (("body_iou", "intersection over union, largest body and observed main body"),
                   ("body_share_of_obs_body_covered", "share of the observed main body covered by the model body"),
                   ("body_share_inside_obs_body", "share of the model body inside the observed main body"),
                   ("all_share_of_obs_all_covered", "share of all observed 2026 ice covered by model ice >= 1 m"),
                   ("model_ice_outside_obs_km2", "model ice >= 1 m outside the observed 2026 polygons (km2)"),
                   ("obs_not_covered_km2", "observed 2026 polygons without model ice >= 1 m (km2)")):
        add("F5/F6 start", lab + ", 5/50/95 %", " / ".join(f"{v:.3f}" for v in q(col(k))), "not reported", "", True)
    add("F5/F6 start", "V 2026 decision 8 | hindcast, 5/50/95 % (km3)",
        " / ".join(f"{v:.4f}" for v in q(col("V_dec8"))) + " | " + " / ".join(f"{v:.4f}" for v in q(col("V_hind"))),
        "0.0053 / 0.0073 / 0.0103 | 0.009 / 0.034 / 0.084", "", True)
    # F6 gate
    Pj = read_csv(RES / "f56_proj.csv")
    Rp = {r["name"]: r for r in read_csv(RES / "f6_runs_f6_proj.csv")}
    n_end = sum(r["end"] in ("no ice", "2100") and r["end_consistent"] == "1" and r["ice_before_stop"] == "1"
                for r in Pj)
    add("F6 gate", "runs ended (no ice with A1 = 0 at the last save and ice before, or 2100) | no ice | 2100",
        f"{n_end} | {sum(r['end'] == 'no ice' for r in Pj)} | {sum(r['end'] == '2100' for r in Pj)}",
        "10062 | 9004 | 1058", "", n_end == 10062 and len(Pj) == 10062)
    add("F6 gate", "years contiguous | NaN | parameter mismatches | start V, A1 off the start state (> 1e-6, 1e-3)",
        f"{sum(r['years_contiguous'] == '1' for r in Pj)} | {sum(int(r['nan']) for r in Pj)} | "
        f"{sum(int(r['par_bad']) for r in Pj)} | {sum(float(r['dV_start']) > 1e-6 or float(r['dA_start']) > 1e-3 for r in Pj)}",
        "10062 | 0 | 0 | 0", "", all(r["years_contiguous"] == "1" for r in Pj) and
        sum(int(r["nan"]) + int(r["par_bad"]) for r in Pj) == 0)
    add("F6 gate", "ice-free grids consistent | identical to icefree_f6_proj.npz",
        f"{sum(r['icefree_ok'] == '1' for r in Pj)} | {sum(r['icefree_same_as_collected'] == '1' for r in Pj)}",
        "10062 | 10062", "", all(r["icefree_ok"] == "1" and r["icefree_same_as_collected"] == "1" for r in Pj))
    add("F6 gate", "2021-2026 path ends at the start state (year, V)", sum(r["pre_link_ok"] == "1" for r in Pj),
        "10062", "", all(r["pre_link_ok"] == "1" for r in Pj))
    for c in ("dis_001", "dis_01", "half_2026", "half_2021"):
        bad = [r["name"] for r in Pj if int(r[c]) != int(Rp[r["name"]][c] or NONE)]
        add("F6 gate", f"{c} per member against f6_runs_f6_proj.csv (mismatches)", len(bad), "0",
            ", ".join(bad[:5]), not bad)
    rr = {k: v for k, v in misc.items() if k.startswith("rerun")}
    add("F6 gate", "4 reruns: series and fields identical; disappearance year", "; ".join(
        f"{k.split()[1][:-1]} {v}" for k, v in rr.items()), "identical, same year", "",
        all(v == "1" for k, v in rr.items() if "identical" in k) and all(
            int(v) == int(Rp[k.split()[1][:-1]]["dis_001"] or NONE) for k, v in rr.items() if "dis_001" in k))
    bm = [float(r["basin_melt_km3"]) for r in Pj]
    add("F6 gate", "runs with basin melt > 0 | largest (km3)", f"{sum(b > 0 for b in bm)} | {max(bm):.1e}",
        "8 | 3e-6", "", True)
    mx = max(Pj, key=lambda r: float(r["max_V_ratio"]))
    add("F6 gate", "members exceeding V_2026 at some save | largest ratio",
        f"{sum(float(r['max_V_ratio']) > 1 for r in Pj)} | {float(mx['max_V_ratio']):.2f} ({mx['name']})",
        "1511 | 2.36 (d061_c20_5)", "", True)
    NF = {r["name"]: r for r in read_csv(RES / "f56_noflow.csv")}
    Rn = {r["member"]: r for r in read_csv(RES / "f6_noflow.csv")}
    add("F6 no-flow", "twins | ended | saves after a stop | NaN | years contiguous",
        f"{len(NF)} | {sum(r['ended'] == '1' for r in NF.values())} | "
        f"{sum(int(r['save_after_stop']) for r in NF.values())} | {sum(int(r['nan']) for r in NF.values())} | "
        f"{sum(r['years_contiguous'] == '1' for r in NF.values())}", "9984 | 9984 | 0 | 0 | 9984", "",
        len(NF) == 9984 and all(r["ended"] == "1" and r["save_after_stop"] == "0" for r in NF.values()))
    bad = [n for n in NF if int(NF[n]["dis_001"]) != int(Rn[n]["dis_001"] or NONE)]
    add("F6 no-flow", "dis_001 per twin against f6_noflow.csv (mismatches)", len(bad), "0", ", ".join(bad[:5]), not bad)
    PR = {r["name"]: r for r in Pj}
    dstart = sum(abs(float(NF[n]["V_2026"]) - float(PR[n]["V_2026"])) > 1e-6 for n in NF)
    diff = np.array([int(PR[n]["dis_001"]) - int(NF[n]["dis_001"]) for n in NF])
    add("F6 no-flow", "start V equal to the IGM member | IGM minus no-flow disappearance 5/50/95 %, min, max | later, "
        "equal, earlier", f"{len(NF) - dstart} | " + " / ".join(f"{v:.0f}" for v in q(diff)) +
        f", {diff.min()}, {diff.max()} | {(diff > 0).sum()}, {(diff == 0).sum()}, {(diff < 0).sum()}",
        "9984 | 0 / 0 / 0, -47, +54 | 119, 9599, 266", "", dstart == 0)
    write_csv(RES / "f56_check.csv", ["part", "quantity", "this check", "reported", "difference", "pass"], out)
    years(Pj)
    for o in out:
        print(*o, sep=" | ")


def years(Pj):
    """results/f56_years.csv: 5/50/95 % per 2026 level and SSP of the F6 years (this check), of the first ensemble
    (step6_runs_s6_proj.csv with step6_members.csv), and the published medians."""
    Rp = {r["name"]: r for r in read_csv(RES / "f6_runs_f6_proj.csv")}
    M1 = {m["member"]: m for m in read_csv(RES / "step6_members.csv")}
    R1 = read_csv(RES / "step6_runs_s6_proj.csv")
    rows = []
    for c in ("dis_001", "dis_01", "half_2026", "half_2021"):
        for lev in ("anchored", "native"):
            for ssp in ("SSP1-2.6", "SSP2-4.5", "SSP5-8.5"):
                a = [int(r[c]) for r in Pj if r["map_u0"] == "0" and r["level"] == lev and r["ssp"] == ssp]
                b = [int(r[c] or NONE) for r in R1 if M1[r["name"]]["map_u0"] == "0" and M1[r["name"]]["level"] == lev
                     and M1[r["name"]]["ssp"] == ssp]
                pa, pb = q(a), q(b)
                pub = "; ".join(f"{s} {('> 2100' if v is None else v)}" for s, sp, cc, v in PUBLISHED
                                if sp == ssp and cc == c)
                rows.append([c, lev, ssp, len(a), *[yfmt(v) for v in pa], f"{100 * np.mean(np.array(a) >= NONE):.1f}",
                             len(b), *[yfmt(v) for v in pb], f"{100 * np.mean(np.array(b) >= NONE):.1f}",
                             f"{pa[1] - pb[1]:+.1f}", pub])
    write_csv(RES / "f56_years.csv", ["quantity", "level", "ssp", "n", "p5", "p50", "p95", "pct_not_by_2100",
              "first_n", "first_p5", "first_p50", "first_p95", "first_pct_not_by_2100", "median_shift",
              "published_median"], rows)


if __name__ == "__main__":
    {"tar": tar, "compare": compare}[sys.argv[1]]()
