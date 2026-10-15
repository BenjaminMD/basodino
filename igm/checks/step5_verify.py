"""Independent check of DESIGN.md step 5: rebuilds what hindcast.py reduces and validates; imports neither hindcast.py
nor track.py. Reads the banked run tar, not the reduced CSVs, and compares with them at the end.

  tar (s02, agl env)  streams runs_s5.tar.gz once:
         spin-up: every run of s5_spin1-3, dT_1850 from its .hydra/overrides.yaml, cycle means (series years 1791-1820,
         1821-1850) of V and of the area inside SGI1850, steady (|V_last / V_prev - 1| <= 1 %), within 4.008 +- 0.25
         km2; acceptance rebuilt (first steady run in tolerance per draw, rounds in order) and the restart path of each
         hindcast run checked against it.
         hindcast: from fields.npz the areas (thk >= 1 m, 10 m cells) at the four inventory dates, the volume changes
         between survey dates, the mean surface (topg + thk) over the target cells; tongue and length from the run's
         own series.csv; tongue also recomputed from thk_y2000, thk_y2010, thk_y2020 (lowest surface of the largest
         4-connected body); edge contact from the decadal saves.
         observed areas from the SGI rasters of inputs/igm_static.npz.
         2026 states: inputs/step5_state2026.npz against thk_end of each hindcast run; volume and largest-body area of
         both states per draw against the Sentinel-2 2026-09-27 main body (s2_2026_body).
         -> results/step5_verify.csv (quantity, check, this check, reduced value, difference)
  dec8 (s03, env.sh)  thk_dec8 of step5_state2026.npz against runs/s4_rR/SAMPLE_21/fields.npz thk_end per draw.
         -> printed
usage:  s02:  python checks/step5_verify.py tar      s03:  python checks/step5_verify.py dec8
"""
import csv
import io
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
from scipy.ndimage import label

IGM = Path(__file__).resolve().parents[1]
RES, INP = IGM / "results", IGM / "inputs"
CELL = 1e-4                       # km2 per 10 m cell
VOL = 1e-7                        # km3 per m of thickness on one cell
TARGET, TOL = 4.008, 0.25
DATES = {"SGI1931": ("1929-08-13", "sgi1931"), "SGI1973": ("1973-10-01", "sgi1973"),
         "SGI2010": ("2010-10-01", "sgi2010"), "SGI2016": ("2018-10-01", "sgi2016")}
DV = [("1907-1929", "1907-10-01", "1929-08-13"), ("1929-1949", "1929-08-13", "1949-10-07"),
      ("1949-1961", "1949-10-07", "1961-09-19"), ("1961-1971", "1961-09-19", "1971-09-03"),
      ("1971-1985", "1971-09-03", "1985-09-10"), ("1985-1991", "1985-09-10", "1991-09-18")]


def rows(path_or_text):
    f = io.StringIO(path_or_text) if isinstance(path_or_text, str) else open(path_or_text)
    return list(csv.DictReader(f))


def largest(ice):
    lab, n = label(ice)              # 4-connected (default structure)
    if not n:
        return ice
    c = np.bincount(lab.ravel())
    c[0] = 0
    return lab == np.argmax(c)


def k(d):
    return "thk_" + d.replace("-", "")


def tar():
    g = np.load(INP / "igm_static.npz")
    t = np.load(INP / "step5_targets.npz")
    st = np.load(INP / "step5_state2026.npz")
    in1850 = g["sgi1850"] > 0
    spin, hind, restart = {}, {}, {}
    with tarfile.open(IGM / "runs_s5.tar.gz", "r:gz") as tf:
        for m in tf:
            p = m.name.split("/")
            if not m.isfile() or len(p) < 4:
                continue
            stage, run, f = p[1], p[2], "/".join(p[3:])
            if stage.startswith("s5_spin"):
                d = spin.setdefault((int(stage[-1]), run), {})
                if f == "series.csv":
                    d["s"] = rows(tf.extractfile(m).read().decode())
                elif f == ".hydra/overrides.yaml":
                    d["dT"] = float(re.search(r"offset=([-+.\d]+)", tf.extractfile(m).read().decode()).group(1))
            elif stage == "s5_hind":
                d = hind.setdefault(run, {})
                if f == "series.csv":
                    d["s"] = rows(tf.extractfile(m).read().decode())
                elif f == ".hydra/overrides.yaml":
                    restart[run] = re.search(r"restart=(\S+)", tf.extractfile(m).read().decode()).group(1)
                elif f == "fields.npz":
                    F = np.load(io.BytesIO(tf.extractfile(m).read()))
                    d.update(hind_fields(F, g, t, in1850, st, int(run[1:])))
    out = []                                    # (quantity, check, this, reduced)

    # spin-up
    per = {}
    for (R, run), d in sorted(spin.items()):
        s = d["s"]
        yr = {int(r["year"]): r for r in s}
        last = [yr[y] for y in range(1821, 1851) if y in yr]
        prev = [yr[y] for y in range(1791, 1821) if y in yr]
        ok = len(last) == 30 and len(prev) == 30
        if ok:
            Vl, Vp = (np.mean([float(r["V_km3"]) for r in x]) for x in (last, prev))
            Al = np.mean([float(r["A1_in1850_km2"]) for r in last])
            Ao = np.mean([float(r["A1_out1850_km2"]) for r in last])
            steady = abs(Vl / Vp - 1) <= 0.01
        per.setdefault(int(run[1:4]), []).append(
            dict(R=R, run=run, dT=d["dT"], ok=ok, Vl=Vl if ok else np.nan, dV=(Vl / Vp - 1) if ok else np.nan,
                 Ain=Al if ok else np.nan, Aout=Ao if ok else np.nan, steady=ok and steady,
                 tol=ok and abs(Al - TARGET) <= TOL))
    red = {int(r["draw"]): r for r in rows(RES / "step5_spinup.csv")}
    acc = {}
    for dr, runs in per.items():
        runs.sort(key=lambda x: x["R"])
        a = next((x for x in runs if x["steady"] and x["tol"]), None)
        acc[dr] = a
    A = [acc[d] for d in sorted(acc)]
    n_ok = sum(a is not None for a in A)
    out.append(("spin-up draws", "accepted (rebuilt)", n_ok, len(red)))
    out.append(("spin-up runs", "total", sum(len(v) for v in per.values()), ""))
    out.append(("spin-up runs per draw", "max", max(len(v) for v in per.values()), ""))
    mism = [d for d in sorted(acc) if acc[d] is None or acc[d]["run"] != red[d]["run"]]
    out.append(("accepted run", "draws differing from step5_spinup.csv", len(mism), str(mism[:10])))
    dv = np.array([a["dV"] for a in A])
    ain = np.array([a["Ain"] for a in A])
    aout = np.array([a["Aout"] for a in A])
    out.append(("last-cycle volume change", "min / p50 / max %", f"{dv.min()*100:+.2f} / {np.median(dv)*100:+.2f} / "
                f"{dv.max()*100:+.2f}", ""))
    out.append(("steady", "all accepted |dV| <= 1 %", bool((np.abs(dv) <= 0.01).all()), ""))
    out.append(("A_in 1821-1850", "min / p5 / p50 / p95 / max km2",
                " / ".join(f"{v:.3f}" for v in [ain.min(), *np.percentile(ain, [5, 50, 95]), ain.max()]), ""))
    out.append(("A_in", "all within 4.008 +- 0.25", bool((np.abs(ain - TARGET) <= TOL).all()), ""))
    out.append(("A_out 1821-1850", "min / p50 / max km2", f"{aout.min():.3f} / {np.median(aout):.3f} / "
                f"{aout.max():.3f}", ""))
    dmax = max(abs(a["Ain"] - float(red[d]["Ain_km2"])) for d, a in acc.items())
    out.append(("A_in", "max |rebuilt - step5_spinup.csv| km2", f"{dmax:.2e}", ""))
    dTs = np.array([a["dT"] for a in A])
    vals, cnt = np.unique(np.round(dTs, 4), return_counts=True)
    out.append(("dT_1850", "values (count)", "; ".join(f"{v:+.4f} ({c})" for v, c in zip(vals, cnt)), ""))
    rbad = [r for r in restart if f"s5_spin{acc[int(r[1:])]['R']}/{acc[int(r[1:])]['run']}/" not in restart[r]]
    out.append(("hindcast restart", "runs not starting from the rebuilt accepted spin-up", len(rbad), str(rbad[:5])))

    # hindcast against the validation table
    V = {r["name"]: r for r in rows(RES / "step5_validation.csv")}
    H = [hind[f"h{d:03d}"] for d in range(len(hind))]
    for d in H:
        yr = {int(r["year"]): r for r in d["s"]}
        for y in [2000 + i for i in range(25)]:
            if y in yr:
                d[f"tongue {y}"] = float(yr[y]["tongue_m"])
        for name, y0, y1 in (("length 1899-1929", 1899, 1929), ("length 1899-1961", 1899, 1961),
                             ("length 1899-1986", 1899, 1986), ("length 1899-2024", 1899, 2024)):
            a, b = yr[y0]["length_m"], yr[y1]["length_m"]
            d[name] = float(b) - float(a) if a and b else np.nan
    obs = {n: (g[r] > 0).sum() * CELL for n, (_, r) in DATES.items()}
    names = list(DATES) + [x[0] for x in DV] + [n for n in V if V[n]["kind"] in ("tongue", "length", "surface")] + \
        ["area 2026-09-27 body", "A 2026-09-30", "V 2026-09-30", "edge saves", "edge last year"]
    for n in names:
        m = np.array([d.get(n, np.nan) for d in H], float)
        m = m[np.isfinite(m)]
        r = V[n]
        p5, p50, p95 = np.percentile(m, [5, 50, 95])
        ob = obs.get(n, float(r["observed"]) if r["observed"] else np.nan)
        band = "" if np.isnan(ob) else ("inside" if p5 <= ob <= p95 else "outside")
        same = all(abs(a - float(b)) <= 1e-3 * max(1, abs(a)) for a, b in zip((p5, p50, p95), (r["p5"], r["p50"], r["p95"])))
        out.append((n, f"p5/p50/p95 n={len(m)} obs={ob:.5g} {band}", f"{p5:.5g} / {p50:.5g} / {p95:.5g}",
                    f"{r['p5']} / {r['p50']} / {r['p95']} {r['band']}", "same" if same and band == r["band"] else "DIFF"))
    for n, (_, r) in DATES.items():
        out.append((f"observed {n}", "raster area km2", f"{obs[n]:.4f}", V[n]["observed"]))
    for y in (2000, 2010, 2020):
        m = np.array([d[f"tongue_y{y}"] for d in H])
        s = np.array([d.get(f"tongue {y}", np.nan) for d in H])
        out.append((f"tongue {y} from thk_y{y}", "p50; max |fields - series| m", f"{np.nanmedian(m):.1f}",
                    f"{np.nanmax(np.abs(m - s)):.2f}"))

    # 2026 states
    out.append(("state2026", "draws", len(st["draw"]), ""))
    dmax = max(float(np.abs(st["thk_hind"][d] - H[d]["thk_end"]).max()) for d in range(len(H)))
    out.append(("state2026 thk_hind", "max |state - hindcast thk_end| m", f"{dmax:.2e}", ""))
    ob = (g["s2_2026_body"] > 0).sum() * CELL
    for w in ("hind", "dec8"):
        thk = st[f"thk_{w}"]
        v = thk.sum(axis=(1, 2)) * VOL
        a = np.array([largest(x >= 1.0).sum() * CELL for x in thk])
        p = np.percentile(a, [5, 50, 95])
        out.append((f"state2026 {w}", "V km3 p5/p50/p95", " / ".join(f"{x:.5f}" for x in np.percentile(v, [5, 50, 95])), ""))
        out.append((f"state2026 {w}", f"largest-body area km2 p5/p50/p95 vs observed {ob:.4f}",
                    " / ".join(f"{x:.4f}" for x in p), "inside" if p[0] <= ob <= p[2] else
                    f"outside, p95 short by {ob - p[2]:.4f} km2 ({(ob - p[2]) / ob * 100:.0f} %)"))
        out.append((f"state2026 {w}", "draws with no ice", int((v == 0).sum()), ""))
    with open(RES / "step5_verify.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "check", "this_check", "reduced", "difference"])
        w.writerows(out)
    for o in out:
        print(*o, sep=" | ")


def hind_fields(F, g, t, in1850, st, draw):
    d = {"thk_end": F["thk_end"]}
    vol = lambda x: float(F[k(x)].sum()) * VOL
    for n, (date, _) in DATES.items():
        d[n] = (F[k(date)] >= 1.0).sum() * CELL
    for n, a, b in DV:
        d[n] = vol(b) - vol(a)
    for n, date, cells in (("surface 1991", "1991-09-18", g["mask_1994"] == 1),
                           ("surface 2012", "2012-09-01", t["has12"].astype(bool)),
                           ("surface 2021", "2021-09-30", g["mask_2021"] == 1)):
        d[n] = float((F["topg"] + F[k(date)])[cells].mean())
    for y in (2000, 2010, 2020):
        ice = F[f"thk_y{y}"] >= 1.0
        d[f"tongue_y{y}"] = float((F["topg"] + F[f"thk_y{y}"])[largest(ice)].min()) if ice.any() else np.nan
    d["area 2026-09-27 body"] = largest(F[k("2026-09-27")] >= 1.0).sum() * CELL
    d["A 2026-09-30"] = (F["thk_end"] >= 1.0).sum() * CELL
    d["V 2026-09-30"] = float(F["thk_end"].sum()) * VOL
    e = np.zeros(F["topg"].shape, bool)
    e[[0, -1], :] = True
    e[:, [0, -1]] = True
    dec = [x for x in F.files if re.fullmatch(r"thk_y\d{4}", x)]
    touch = [int(x[5:]) for x in dec if (F[x][e] >= 1.0).any()]
    d["edge saves"], d["edge last year"] = float(len(touch)), float(max(touch, default=0))
    return d


def dec8():
    st = np.load(INP / "step5_state2026.npz")
    D = rows(RES / "posterior_128.csv")
    dmax, miss = 0.0, []
    for i, r in enumerate(D):
        p = IGM / "runs" / f"s4_r{r['round']}" / f"{r['sample']}_21" / "fields.npz"
        if not p.exists():
            miss.append(i)
            continue
        with np.load(p) as F:
            dmax = max(dmax, float(np.abs(F["thk_end"] - st["thk_dec8"][i]).max()))
    print(f"dec8: {len(D) - len(miss)} of {len(D)} draws compared, max |state - s4 thk_end| {dmax:.2e} m, missing {miss}")


if __name__ == "__main__":
    {"tar": tar, "dec8": dec8}[sys.argv[1]]()
