"""Independent check of F3 (DESIGN.md, fix round): model values, likelihood, importance weights, effective sample size
and the area gate re-derived from the run outputs with code that does not import fit_f3.py or weights.py.

  extract  (s03, env.sh) every run of runs/f3_r1, f3_r1fix, f3_r2 (a rerun replaces its round-1 run):
           _21  areas thk >= 0.1, 1, 2 m on 2024-08-23 (all cells), 2025-09-07 and 2026-09-27 (largest 4-connected
                body, own flood fill), mean bed (fields topg) per radar profile, the largest start thickness
                (usurf_2021 - topg on the 2021 mask) on each stake disc, V_2026 (series), basin melt (series)
           _12  dh 2012-2021 per 50 m band: topg + thk_20210930 minus the start surface (max(s12, topg) where s12 > 0
                inside SGI2010 or the 2021 mask, topg elsewhere); ITS_LIVE speed = median of series its_median
                over 2014-2021
           observations: radar bedrock mean per profile (inputs/radar_2006.csv, above_2021 = 0); tlm2024 and
           s2_2026_body cell areas
           -> results/verify_f3_runs.csv, results/verify_f3_obs.csv
  weigh    (Mac) likelihood per term with the term's model-error SD s = S_REF x log-uniform [0.1, 10] integrated on
           2000 nodes (Gaussian terms normalised; one-sided bounds as the unnormalised kernel of the design), prior
           uniform in (u, ln f), proposal = balance-heuristic mixture of round 1 (prior) and round 2 (0.9 truncated
           Gaussian of results/f3_proposal_r2.npz, its box mass recomputed, + 0.1 prior), Kish effective sample size,
           weighted 5 / 50 / 95 % of the gate areas; variant speed_strict (speed bound with sigma only)
           -> results/verify_f3.csv (quantity, own, reported)
usage:  s03: python checks/verify_f3.py extract ;  Mac: python3 checks/verify_f3.py weigh
"""
import csv
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
RES, INP = IGM / "results", IGM / "inputs"
STAGES = ("f3_r1", "f3_r1fix", "f3_r2")
DATES = (("2024", "20240823", False), ("2025", "20250907", True), ("2026", "20260927", True))
U_BOX, F_BOX = (-1.0, 2.0), (0.05, 20.0)
S_REF = {"area": 0.1, "dh2": 2.93239, "radar": 5.3, "stake": 1.0, "speed": 0.5}


def body(ice):
    """Largest 4-connected body of the boolean grid ice (own flood fill)."""
    lab = np.zeros(ice.shape, int)
    best, n = 0, 0
    for start in zip(*np.nonzero(ice)):
        if lab[start]:
            continue
        n += 1
        lab[start], stack, size = n, [start], 0
        while stack:
            r, c = stack.pop()
            size += 1
            for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= rr < ice.shape[0] and 0 <= cc < ice.shape[1] and ice[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = n
                    stack.append((rr, cc))
        if size > best:
            best, keep = size, n
    return lab == keep if n else ice


def extract():
    g = np.load(INP / "igm_static.npz")
    t = np.load(INP / "step4_targets.npz")
    s12 = np.load(INP / "step5_targets.npz")["s12"].astype(np.float64)
    ny, nx = g["usurf_2021"].shape
    on12 = (s12 > 0) & ((g["sgi2010"] == 1) | (g["mask_2021"] == 1))
    m21 = g["mask_2021"] == 1
    rad = [r for r in csv.DictReader(open(INP / "radar_2006.csv")) if r["above_2021"] == "0"]
    rcell = np.array([(ny - 1 - int(r["row"])) * nx + int(r["col"]) for r in rad])
    rprof = np.array([int(r["profile"]) for r in rad])
    with open(RES / "verify_f3_obs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value"])
        for p in range(1, 7):
            w.writerow([f"radar bedrock mean profile {p} m", f"{np.mean([float(r['bedrock']) for r, q in zip(rad, rprof) if q == p]):.4f}"])
        w.writerow(["tlm2024 cells x 1e-4 km2", f"{(g['tlm2024'] == 1).sum() * 1e-4:.4f}"])
        w.writerow(["s2_2026_body cells x 1e-4 km2", f"{(g['s2_2026_body'] == 1).sum() * 1e-4:.4f}"])
    band = t["dh2_band"]
    nb = len(t["dh2_obs"])
    runs = {}
    for st in STAGES:
        d = IGM / "runs" / st
        if d.exists():
            for p in sorted(d.iterdir()):
                runs[p.name] = p                         # a later stage (the rerun) replaces the earlier
    rows = []
    for name in sorted(n[:-3] for n in runs if n.endswith("_21")):
        row = {"sample": name}
        p21, p12 = runs.get(name + "_21"), runs.get(name + "_12")
        try:
            F = np.load(p21 / "fields.npz")
            s = list(csv.DictReader(open(p21 / "series.csv")))
            for d_, key, main in DATES:
                for h in (0.1, 1.0, 2.0):
                    ice = F[f"thk_{key}"] >= h
                    row[f"A{d_}_{h}"] = f"{(body(ice) if main else ice).sum() * 1e-4:.4f}"
            b = F["topg"].ravel()[rcell]
            for p in range(1, 7):
                row[f"radar_{p}"] = f"{b[rprof == p].mean():.4f}"
            thk0 = np.where(m21, np.maximum(g["usurf_2021"] - F["topg"], 0.0), 0.0).ravel()
            for k, c in enumerate(t["stake_cells"]):
                row[f"stake_{k}"] = f"{thk0[c[c >= 0]].max():.3f}"
            row["V_2026"], row["end21"] = s[-1]["V_km3"], s[-1]["t"]
            row["basin_melt_21"] = f"{sum(float(r['basin_melt_km3'] or 0) for r in s[1:]):.6f}"
            F = np.load(p12 / "fields.npz")
            s = list(csv.DictReader(open(p12 / "series.csv")))
            z0 = np.where(on12, np.maximum(s12, F["topg"]), F["topg"])
            dz = F["topg"] + F["thk_20210930"] - z0
            for k in range(nb):
                row[f"dh2_{k}"] = f"{dz[band == k].mean():.4f}"
            row["speed"] = f"{np.median([float(r['its_median']) for r in s if 2014 <= int(r['year']) <= 2021]):.4f}"
            row["end12"] = s[-1]["t"]
        except (KeyError, FileNotFoundError, TypeError) as e:
            row["error"] = repr(e)[:80]
        rows.append(row)
    head = ["sample"] + [f"A{d}_{h}" for d, _, _ in DATES for h in (0.1, 1.0, 2.0)] + \
        [f"radar_{p}" for p in range(1, 7)] + [f"stake_{k}" for k in range(len(t["stake_cells"]))] + \
        [f"dh2_{k}" for k in range(nb)] + ["speed", "V_2026", "basin_melt_21", "end21", "end12", "error"]
    with open(RES / "verify_f3_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows([[r.get(k, "") for k in head] for r in rows])
    print(f"{len(rows)} samples, {sum('error' in r for r in rows)} with errors: results/verify_f3_runs.csv")


def loglik(r, sig, sref, one_sided, nodes):
    """log of the mean over s nodes of prod_j kernel(r_j; sig_j^2 + s^2); r (samples, values)."""
    s = sref * nodes
    var = sig[None, None, :] ** 2 + s[:, None, None] ** 2           # (node, 1, value)
    ll = -0.5 * r[None] ** 2 / var
    if not one_sided:
        ll = ll - 0.5 * np.log(2 * np.pi * var)
    ll = ll.sum(-1)
    m = ll.max(0)
    return m + np.log(np.exp(ll - m).mean(0))


def wq(x, w, q):
    o = np.argsort(x)
    c = (np.cumsum(w[o]) - 0.5 * w[o]) / w.sum()
    return np.interp(q, c, x[o])


def weigh():
    from scipy.stats import multivariate_normal
    t = np.load(RES / "step4_targets.npz")
    obs = {r["quantity"]: float(r["value"]) for r in csv.DictReader(open(RES / "verify_f3_obs.csv"))}
    runs = {r["sample"]: r for r in csv.DictReader(open(RES / "verify_f3_runs.csv"))}
    S = [r for rnd in (1, 2) for r in csv.DictReader(open(RES / f"f3_samples_r{rnd}.csv"))]
    ok = [s for s in S if s["name"] in runs and not runs[s["name"]]["error"]
          and abs(float(runs[s["name"]]["end21"]) - 2026.75) < 0.01 and abs(float(runs[s["name"]]["end12"]) - 2021.75) < 0.01]
    R = [runs[s["name"]] for s in ok]
    rep = {r["name"]: r for r in csv.DictReader(open(RES / "f3_weights.csv"))}
    summ = {r["quantity"]: r["value"] for r in csv.DictReader(open(RES / "f3_summary.csv"))}
    out = []
    add = lambda q, own, r="": out.append((q, own, r))
    add("samples with both segments ended, of 2 x 1024", f"{len(ok)}", summ["samples used (both segments ended), rounds"])
    for rnd in (1, 2):
        b = [int(s["balance_draw"]) for s in S if s["round"] == str(rnd)]
        add(f"round {rnd}: balance draws used once each (permutation of 1024)", f"{sorted(b) == list(range(1024))}")
    col = lambda ks: np.array([[float(r[k]) for k in ks] for r in R])
    nodes = np.exp(np.log(0.1) + (np.arange(2000) + 0.5) / 2000 * np.log(100.0))
    sel = t["stake_with2026"]
    radar_obs = np.array([obs[f"radar bedrock mean profile {p} m"] for p in range(1, 7)])

    def terms(h=1.0, strict=()):
        A = col([f"A{d}_{h}" for d in ("2024", "2025", "2026")])
        spec = {"area": (A - t["area_obs"][1:], t["area_sigma"][1:], False),
                "dh2": (col([f"dh2_{k}" for k in range(14)]) - t["dh2_obs"], np.full(14, S_REF["dh2"]), False),
                "radar": (col([f"radar_{p}" for p in range(1, 7)]) - radar_obs, np.full(6, 5.3), False),
                "stake": (np.maximum(t["stake_bound"][sel] - col([f"stake_{k}" for k in np.flatnonzero(sel)]), 0),
                          np.full(sel.sum(), 1.0), True),
                "speed": (np.maximum(col(["speed"]) - float(t["speed_bound"]), 0), np.atleast_1d(float(t["speed_sigma"])),
                          True)}
        return {k: loglik(r, sg, S_REF[k], one, np.array([1e-9]) if k in strict else nodes)
                for k, (r, sg, one) in spec.items()}

    x = np.array([[float(s["u"]), np.log(float(s["f"]))] for s in ok])
    box = (x[:, 0] >= U_BOX[0]) & (x[:, 0] <= U_BOX[1]) & (x[:, 1] >= np.log(F_BOX[0])) & (x[:, 1] <= np.log(F_BOX[1]))
    prior = 1.0 / ((U_BOX[1] - U_BOX[0]) * np.log(F_BOX[1] / F_BOX[0]))
    P = np.load(RES / "f3_proposal_r2.npz")
    mvn = multivariate_normal(P["m"], P["c"])
    lo, hi = np.array([U_BOX[0], np.log(F_BOX[0])]), np.array([U_BOX[1], np.log(F_BOX[1])])
    Z = mvn.cdf(hi) - mvn.cdf([lo[0], hi[1]]) - mvn.cdf([hi[0], lo[1]]) + mvn.cdf(lo)
    add("round-2 proposal mass inside the box (own, Gaussian CDF; saved Monte Carlo)", f"{Z:.4f}", f"{float(P['z']):.4f}")
    q2 = (0.9 * np.where(box, mvn.pdf(x), 0) + 0.1 * prior) / (0.9 * Z + 0.1)
    n1 = sum(s["round"] == "1" for s in ok)
    n2 = len(ok) - n1
    q = (n1 * prior + n2 * q2) / (n1 + n2)
    lq_rep = np.array([float(rep[s["name"]]["log_q"]) for s in ok])
    add("log proposal density: largest |own - reported|", f"{np.abs(np.log(q) - lq_rep).max():.2e}")
    r1 = np.array([s["round"] == "1" for s in ok])
    res = {}
    for v, kw in (("main", {}), ("speed_strict", {"strict": ("speed",)}), ("h0.1", {"h": 0.1}), ("h2", {"h": 2.0})):
        L = terms(**kw)
        tot = sum(L.values()) + np.log(prior) - np.log(q)
        w = np.where(box, np.exp(tot - tot.max()), 0)
        res[v] = (w / w.sum(), L)
        e = 1 / ((w / w.sum()) ** 2).sum()
        add(f"effective sample size, {v} (pooled rounds 1 and 2)", f"{e:.1f}",
            summ.get(f"effective sample size, {v} (pooled rounds, Kish)", ""))
    w, L = res["main"]
    rw = np.array([float(rep[s["name"]]["w_main"]) for s in ok])
    add("weights main: largest |own - reported| normalised weight; correlation", f"{np.abs(w - rw).max():.2e}; "
        f"{np.corrcoef(w, rw)[0, 1]:.6f}")
    for k in S_REF:
        lr = np.array([float(rep[s["name"]][f"ll_{k}"]) for s in ok])
        add(f"term {k}: largest |own - reported| log-likelihood (2000 against 64 s nodes)", f"{np.abs(L[k] - lr).max():.3f}")
    l1 = sum(L.values())[r1] + np.log(prior) - np.log(prior)
    w1 = np.exp(l1 - l1.max())
    add("effective sample size, round 1 alone", f"{w1.sum() ** 2 / (w1 ** 2).sum():.1f}",
        summ.get("effective sample size, rounds 1 to 1 pooled", ""))
    add("largest normalised weight (main)", f"{w.max():.4f}", summ["largest normalised weight (main)"])
    gate = (("2026", "20260927 main body", 3), ("2025", "20250907 main body", 2), ("2024", "20240823 all", 1))
    for d, lab, j in gate:
        for v in ("main", "speed_strict", "h0.1", "h2"):
            h = {"h0.1": 0.1, "h2": 2.0}.get(v, 1.0)
            a = col([f"A{d}_{h}"])[:, 0]
            qq = wq(a, res[v][0], [0.05, 0.5, 0.95])
            o = float(t["area_obs"][j])
            key = [k for k in summ if k.startswith(f"gate: area {d}")]
            add(f"area {lab} km2 ({v}, thk >= {h} m): weighted 5 / 50 / 95 %; observed; inside; outside by",
                " ".join(f"{z:.4f}" for z in qq) + f"; {o:.4f}; {qq[0] <= o <= qq[2]}; "
                f"{max(qq[0] - o, o - qq[2], 0):.4f}", summ[key[0]] if key and v == "main" else "")
    rr = {(s["name"]): r for s, r in zip(ok, R)}
    rc = {r["name"]: r for st in ("f3_r1", "f3_r1fix", "f3_r2")
          for r in csv.DictReader(open(RES / f"f3_runs_{st}.csv")) if r["rc"] == "0" and r["end"]}
    for key, own in (("A2026_1.0", "A2026_1.0"), ("A2024_1.0", "A2024_1.0"), ("speed", "speed"), ("dh2_5", "dh2_5"),
                     ("radar_3", "radar_3"), ("stake_15", "stake_15")):
        seg = "_12" if key in ("speed", "dh2_5") else "_21"
        d = [abs(float(rr[n][own]) - float(rc[n + seg][key])) for n in rr if n + seg in rc]
        add(f"model value {key}: largest |own - reduce csv| over {len(d)} samples", f"{max(d):.4g}")
    sp = col(["speed"])[:, 0]
    add("speed: weighted share above 1 m a-1; weighted mean excess (main)",
        f"{w[sp > 1].sum():.3f}; {w @ np.maximum(sp - 1, 0):.3f}")
    a26, a24 = col(["A2026_1.0"])[:, 0], col(["A2024_1.0"])[:, 0]
    near = np.abs(a26 - float(t["area_obs"][3])) <= float(t["area_sigma"][3])
    add("samples within one sigma of the 2026 area: count; largest 2024 area", f"{near.sum()}; {a24[near].max():.4f}")
    add("observed areas from the static masks: tlm2024; s2_2026_body (km2, cells x 1e-4)",
        f"{obs['tlm2024 cells x 1e-4 km2']:.4f}; {obs['s2_2026_body cells x 1e-4 km2']:.4f}",
        f"{t['area_obs'][1]:.4f}; {t['area_obs'][3]:.4f}")
    with open(RES / "verify_f3.csv", "w", newline="") as f:
        csv.writer(f).writerows([("quantity", "own", "reported"), *out])
    for r in out:
        print(*r, sep=" | ")


if __name__ == "__main__":
    {"extract": extract, "weigh": weigh}[sys.argv[1]]()
