"""Independent check of DESIGN.md step 4: rebuilds what weights.py computes; imports neither weights.py nor geometry.py.

  fields (s03, env.sh)  model side from the raw run outputs (runs/STAGE/NAME/fields.npz, series.csv) for the MAP sample
         and 5 random distinct samples of results/posterior_128.csv (seed 7): areas (thk >= 1 m; 2021 and 2024 all cells,
         2025 and 2026 the largest 4-connected body), radar profile means (mean of the 2005-10-01 and 2006-10-01
         thickness at the kept points), speed (median of series its_median 2014-2021), stakes (largest 2021 start
         thickness over each disc, start state of decision 8 rebuilt here from inputs/igm_static.npz), compared with
         results/step4_runs_STAGE.csv. Also: the taper of igm_static.npz against min(1, d / 50 m), d the Euclidean
         distance from a mask_2021 cell centre to the nearest cell centre outside the mask.
         -> results/step4_verify_fields.csv
  weights (Mac)  likelihood of every sample from results/step4_runs_s4_r*.csv and the targets, written out again
         (64 log-spaced nodes of s; also 4096 nodes to show the quadrature error), proposal densities rebuilt from
         the stated round-2 Gaussian, importance weights pooled by the balance heuristic, Kish effective sample size,
         edge piling of u and ln f; compared with results/step4_weights.csv; the round-2 proposal both as weights.py
         stated it before its correction of 2026-10-08 (0.9 g / Z + 0.1 p) and as the sampler draws it ((0.9 g 1_box + 0.1 p) /
         (0.9 Z + 0.1), rejected Gaussian draws are redrawn). -> results/step4_verify.csv
  d94 (Mac)  the decision-9 check rebuilt: every draw of posterior_128.csv matched by draw number to its October-1994
         run (results/step4_runs_s4_d94.csv; s4_d94fix replaces the draws it reran), the 1991-2021 terms (area 2021,
         dh1, dh2, radar, speed) replaced, the 2021-2026 terms kept; ratio weights L_1994 / L_1991, their Kish size;
         u, f and V_2021 under both datings; the run-to-run spread of the copies (same sample, same inputs) and of
         their total log-likelihood. -> results/step4_verify_d94.csv
usage:  s03:  python checks/step4_verify.py fields      Mac:  python checks/step4_verify.py weights | d94
"""
import csv
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
RES = IGM / "results"
U_BOX, LNF_BOX = (-1.0, 2.0), (np.log(0.05), np.log(4.0))
SREF = {"area": 0.1, "dh1": 2.93239, "dh2": 2.93239, "radar": 5.3, "stake": 1.0, "speed": 0.5}


def rows(path):
    return list(csv.DictReader(open(path)))


def write(path, out):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        w.writerows(out)
    for q, v in out:
        print(f"{q}: {v}")


def pick():
    post = rows(RES / "posterior_128.csv")
    w = {}
    for p in post:
        w[p["sample"]] = float(p["weight"])
    names = sorted(w)
    imap = max(names, key=w.get)
    rest = [n for n in names if n != imap]
    return [imap] + list(np.random.default_rng(7).choice(rest, 5, replace=False))


# --- fields (s03) ----------------------------------------------------------------------------------------------------
def largest_body(ice):
    """Largest 4-connected component, by flood fill (no scipy)."""
    seen = np.zeros(ice.shape, bool)
    best = np.zeros(ice.shape, bool)
    for i0, j0 in zip(*np.nonzero(ice)):
        if seen[i0, j0]:
            continue
        stack, comp = [(i0, j0)], []
        seen[i0, j0] = True
        while stack:
            i, j = stack.pop()
            comp.append((i, j))
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < ice.shape[0] and 0 <= b < ice.shape[1] and ice[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    stack.append((a, b))
        if len(comp) > best.sum():
            best[:] = False
            best[tuple(np.array(comp).T)] = True
    return best


def start_thk_2021(g, u):
    sig = g["sigma_plus"] if u > 0 else g["sigma_minus"]
    b = g["bed_grab"] - u * sig * g["taper"]
    s = g["usurf_2021"].astype(np.float64)
    on = g["mask_2021"] == 1
    b = np.where(np.isnan(b), s - 1.0, np.minimum(b, s - 1.0))
    return np.where(on, np.maximum(s - b, 0.0), 0.0)


def fields():
    from scipy.ndimage import distance_transform_edt
    g = dict(np.load(IGM / "inputs/igm_static.npz"))
    t = dict(np.load(IGM / "inputs/step4_targets.npz"))
    ref = {}
    for st in ("s4_r1", "s4_r1fix", "s4_r2"):
        for r in rows(RES / f"step4_runs_{st}.csv"):
            if r["rc"] == "0" and r["end"]:
                ref[r["name"]] = (st, r)
    samp = {s["name"]: s for k in (1, 2) for s in rows(RES / f"step4_samples_r{k}.csv")}
    on = g["mask_2021"] == 1
    d = distance_transform_edt(on) * 10.0
    tap = np.minimum(1.0, d / 50.0)
    out = [("taper: max |igm_static taper - min(1, d / 50 m)| on mask_2021; cells differing by > 0.05",
            f"{np.abs(g['taper'][on] - tap[on]).max():.3f}; {(np.abs(g['taper'][on] - tap[on]) > 0.05).sum()}")]
    worst = 0.0
    for name in pick():
        u = float(samp[name]["u"])
        st91, r91 = ref[name + "_91"]
        st21, r21 = ref[name + "_21"]
        F91 = np.load(IGM / "runs" / st91 / (name + "_91") / "fields.npz")
        F21 = np.load(IGM / "runs" / st21 / (name + "_21") / "fields.npz")
        ser = rows(IGM / "runs" / st91 / (name + "_91") / "series.csv")
        mine, theirs = {}, {}
        mine["A2021"] = (F91["thk_20210930"] >= 1.0).sum() * 1e-4
        mine["A2024"] = (F21["thk_20240823"] >= 1.0).sum() * 1e-4
        mine["A2025"] = largest_body(F21["thk_20250907"] >= 1.0).sum() * 1e-4
        mine["A2026"] = largest_body(F21["thk_20260927"] >= 1.0).sum() * 1e-4
        h06 = ((F91["thk_20051001"].astype(np.float64) + F91["thk_20061001"]) / 2).ravel()[t["radar_cell"]]
        for p in range(1, 7):
            mine[f"radar_{p}"] = h06[t["radar_profile"] == p].mean()
        mine["speed"] = np.median([float(r["its_median"]) for r in ser if 2014 <= int(r["year"]) <= 2021])
        th0 = start_thk_2021(g, u).ravel()
        for k, c in enumerate(t["stake_cells"]):
            mine[f"stake_{k}"] = th0[c[c >= 0]].max()
        for k in mine:
            col = {"A2021": "A2021_1.0", "A2024": "A2024_1.0", "A2025": "A2025_1.0", "A2026": "A2026_1.0"}.get(k, k)
            theirs[k] = float((r91 if col in r91 and r91[col] != "" else r21)[col])
        dev = {k: abs(mine[k] - theirs[k]) for k in mine}
        k = max(dev, key=dev.get)
        worst = max(worst, dev[k] / max(abs(theirs[k]), 1e-9))
        out.append((f"{name} (u {u:+.4f}, f {float(samp[name]['f']):.4f}): areas 2021, 2024, 2025, 2026 km2 here | "
                    "reduce", " ".join(f"{mine[x]:.4f}" for x in ("A2021", "A2024", "A2025", "A2026")) + " | " +
                    " ".join(f"{theirs[x]:.4f}" for x in ("A2021", "A2024", "A2025", "A2026"))))
        out.append((f"{name}: radar 1-6 m here | reduce", " ".join(f"{mine[f'radar_{p}']:.3f}" for p in range(1, 7)) +
                    " | " + " ".join(f"{theirs[f'radar_{p}']:.3f}" for p in range(1, 7))))
        out.append((f"{name}: speed m a-1 here | reduce; stakes 0-18 m here", f"{mine['speed']:.4f} | "
                    f"{theirs['speed']:.4f}; " + " ".join(f"{mine[f'stake_{k}']:.2f}" for k in range(19))))
        out.append((f"{name}: largest |here - reduce| (value)", f"{dev[k]:.2e} ({k})"))
    out.append(("largest relative difference here - reduce over the 6 samples", f"{worst:.2e}"))
    write(RES / "step4_verify_fields.csv", out)


# --- weights (Mac) ---------------------------------------------------------------------------------------------------
def loglik_term(r, sig, sref, one_sided, n_nodes=64, top=10.0):
    """Mean over log-uniform s in [0.1, top] x sref (midpoint nodes in ln s) of the likelihood of residuals r."""
    lo, hi = np.log(0.1 * sref), np.log(top * sref)
    s = np.exp(lo + (np.arange(n_nodes) + 0.5) / n_nodes * (hi - lo))
    v = sig[None, :, None] ** 2 + s[None, None, :] ** 2               # (1, values, nodes)
    ll = (-0.5 * r[:, :, None] ** 2 / v).sum(1)
    if not one_sided:
        ll = ll + (-0.5 * np.log(2 * np.pi * v)).sum(1)
    m = ll.max(1, keepdims=True)
    return (m + np.log(np.exp(ll - m).mean(1, keepdims=True)))[:, 0]


def terms(R, t, h="1.0", stakes="with2026", **kw):
    c = lambda names: np.array([[float(r[n]) for n in names] for r in R])
    sel = t[f"stake_{stakes}"]
    nd1, nd2 = len(t["dh1_obs"]), len(t["dh2_obs"])
    T = {"area": (c([f"A{y}_{h}" for y in (2021, 2024, 2025, 2026)]) - t["area_obs"], t["area_sigma"], False),
         "dh1": (c([f"dh1_{k}" for k in range(nd1)]) - t["dh1_obs"], np.full(nd1, SREF["dh1"]), False),
         "dh2": (c([f"dh2_{k}" for k in range(nd2)]) - t["dh2_obs"], np.full(nd2, SREF["dh2"]), False),
         "radar": (c([f"radar_{p}" for p in range(1, 7)]) - t["radar_profile_obs"], np.full(6, 5.3), False),
         "stake": (np.maximum(0, t["stake_bound"][sel] - c([f"stake_{k}" for k in np.flatnonzero(sel)])),
                   np.ones(sel.sum()), True),
         "speed": (np.maximum(0, c(["speed"]) - float(t["speed_bound"])), np.array([float(t["speed_sigma"])]), True)}
    return {k: loglik_term(r, s, SREF[k], one, **kw) for k, (r, s, one) in T.items()}


def gauss2(x, m, c):
    d = x - m
    ci = np.linalg.inv(c)
    return np.exp(-0.5 * (d @ ci * d).sum(1)) / (2 * np.pi * np.sqrt(np.linalg.det(c)))


def inbox(x):
    return (x[:, 0] >= U_BOX[0]) & (x[:, 0] <= U_BOX[1]) & (x[:, 1] >= LNF_BOX[0]) & (x[:, 1] <= LNF_BOX[1])


def kish(w):
    return w.sum() ** 2 / (w ** 2).sum()


def wquant(x, w, q):
    i = np.argsort(x)
    return np.interp(q, np.cumsum(w[i]) / w.sum(), x[i])


def weights():
    t = dict(np.load(RES / "step4_targets.npz"))
    runs = {}
    for f in ("s4_r1", "s4_r1fix", "s4_r2"):                        # r1fix replaces the three r1 runs it reran
        for r in rows(RES / f"step4_runs_{f}.csv"):
            if r["rc"] == "0" and r["end"]:
                runs[r["name"]] = r
    S, R = [], []
    for k in (1, 2):
        for s in rows(RES / f"step4_samples_r{k}.csv"):
            a, b = runs.get(s["name"] + "_91"), runs.get(s["name"] + "_21")
            if a and b:
                S.append(s)
                R.append({**{x: v for x, v in a.items() if v}, **{x: v for x, v in b.items() if v}})
    x = np.array([[float(s["u"]), np.log(float(s["f"]))] for s in S])
    rnd = np.array([int(s["round"]) for s in S])
    n1, n2 = (rnd == 1).sum(), (rnd == 2).sum()
    p = 1.0 / ((U_BOX[1] - U_BOX[0]) * (LNF_BOX[1] - LNF_BOX[0]))
    # round-2 Gaussian as stated in DESIGN.md and step4_summary (mean, SD, correlation; Z by Monte Carlo here)
    m = np.array([0.12032976, 1.22068233])
    sd = np.array([0.31170659, 0.54921965])
    rho = -0.22144655455817613
    c = np.array([[sd[0] ** 2, rho * sd[0] * sd[1]], [rho * sd[0] * sd[1], sd[1] ** 2]])
    Z = inbox(np.random.default_rng(3).multivariate_normal(m, c, 2_000_000)).mean()
    g = gauss2(x, m, c) * inbox(x)
    q_stated = 0.9 * g / Z + 0.1 * p
    q_drawn = (0.9 * g + 0.1 * p) / (0.9 * Z + 0.1)
    lq_file = np.array([float(s["log_q"]) for s in S])
    out = [("samples used, rounds 1 | 2", f"{n1} | {n2}"),
            ("round-2 Gaussian mass in the box Z (2e6 draws here)", f"{Z:.4f}"),
            ("round-2 samples: max |log q stated here - log_q of step4_samples_r2.csv|",
             f"{np.abs(np.log(q_stated) - lq_file)[rnd == 2].max():.1e}"),
            ("round-2 samples: q drawn / q stated, min / max", f"{(q_drawn / q_stated)[rnd == 2].min():.3f} / "
             f"{(q_drawn / q_stated)[rnd == 2].max():.3f}")]
    L = terms(R, t)
    W = rows(RES / "step4_weights.csv")
    wname = [w["name"] for w in W]
    assert wname == [s["name"] for s in S], "sample order differs from step4_weights.csv"
    for k in L:
        dif = np.abs(L[k] - np.array([float(w[f"ll_{k}"]) for w in W]))
        out.append((f"term {k}: max |log-likelihood here - step4_weights.csv|", f"{dif.max():.1e}"))
    L4096 = terms(R, t, n_nodes=4096)
    tot, tot4096 = sum(L.values()), sum(L4096.values())
    out.append(("total log-likelihood: max |64 nodes - 4096 nodes| over all samples, at the MAP",
                f"{np.abs(tot - tot4096).max():.3f}, {np.abs(tot - tot4096)[np.argmax(tot)]:.3f}"))
    res = {}
    for lab, q in (("stated", q_stated), ("drawn", q_drawn)):
        den = (n1 * p + n2 * q) / (n1 + n2)
        for nodes, tt in (("64", tot), ("4096", tot4096)):
            lw = tt + np.log(p) - np.log(den)
            w = np.exp(lw - lw.max())
            res[(lab, nodes)] = w / w.sum()
    w1 = np.where(rnd == 1, np.exp(tot - tot[rnd == 1].max()), 0.0)
    out.append(("ESS round 1 alone", f"{kish(w1):.2f}"))
    wfile = np.array([float(w["w_main"]) for w in W])
    for key, w in res.items():
        out.append((f"pooled weights, proposal {key[0]}, {key[1]} s nodes: ESS; largest weight; max |w - w_main of "
                    "step4_weights.csv|", f"{kish(w):.2f}; {w.max():.4f}; {np.abs(w - wfile).max():.2e}"))
    u, lf = x[:, 0], x[:, 1]
    for key in (("stated", "64"), ("drawn", "64")):
        w = res[key]
        out.append((f"u 5/50/95 %, proposal {key[0]}", " ".join(f"{v:+.3f}" for v in wquant(u, w, [.05, .5, .95]))))
        out.append((f"f 5/50/95 %, proposal {key[0]}",
                    " ".join(f"{v:.3f}" for v in np.exp(wquant(lf, w, [.05, .5, .95])))))
        for nm, xx, box in (("u", u, U_BOX), ("ln f", lf, LNF_BOX)):
            h = np.histogram(xx, 20, box, weights=w)[0]
            out.append((f"{nm}, proposal {key[0]}: weight in the lowest | highest of 20 bins; bin holding the most "
                        "(0-19)", f"{h[0]:.3f} | {h[-1]:.3f}; {h.argmax()}"))
        out.append((f"proposal {key[0]}: weight with f > 3.5 | u > 0.5 | u < -0.5",
                    f"{w[lf > np.log(3.5)].sum():.3f} | {w[u > 0.5].sum():.3f} | {w[u < -0.5].sum():.3f}"))
    # samples nearest the f edge: the prior box has no samples beyond ln 4, so the weight piles in the top bin
    out.append(("largest f sampled (all 2048); samples with f > 3.8, their total weight (drawn proposal)",
                f"{np.exp(lf.max()):.4f}; {(lf > np.log(3.8)).sum()}, {res[('drawn', '64')][lf > np.log(3.8)].sum():.3f}"))
    w = res[("drawn", "64")]
    i = int(np.argmax(w))
    out.append(("MAP (drawn proposal): name, u, f, per-term log-likelihood area dh1 dh2 radar stake speed",
                f"{S[i]['name']} {u[i]:+.4f} {np.exp(lf[i]):.4f} " + " ".join(f"{L[k][i]:.2f}" for k in L)))
    write(RES / "step4_verify.csv", out)


def d94():
    t = dict(np.load(RES / "step4_targets.npz"))
    runs = {}
    for f in ("s4_r1", "s4_r1fix", "s4_r2"):
        for r in rows(RES / f"step4_runs_{f}.csv"):
            if r["rc"] == "0" and r["end"]:
                runs[r["name"]] = r
    new = {}
    for f in sorted(RES.glob("step4_runs_s4_d94*.csv")):              # s4_d94fix replaces the draws it reran
        new.update({int(r["name"][4:]): r for r in rows(f) if r["rc"] == "0" and r["end"]})
    post = [p for p in rows(RES / "posterior_128.csv") if int(p["draw"]) in new]
    old = [{**{x: v for x, v in runs[p["sample"] + "_91"].items() if v},
            **{x: v for x, v in runs[p["sample"] + "_21"].items() if v}} for p in post]
    k91 = [k for k in new[0] if k not in ("name", "rc", "end") and new[0][k] != ""]
    upd = [{**o, **{k: new[int(p["draw"])][k] for k in k91}} for p, o in zip(post, old)]
    lo, ln = terms(old, t), terms(upd, t)
    lr = sum(ln.values()) - sum(lo.values())
    r = np.exp(lr - lr.max())
    one = np.ones(len(post))
    u = np.array([float(p["u"]) for p in post])
    f = np.array([float(p["f"]) for p in post])
    v0 = np.array([float(o["V_2021"]) for o in old])
    v1 = np.array([float(o["V_2021"]) for o in upd])
    out = [("draws matched (of 128); distinct samples", f"{len(post)}; {len({p['sample'] for p in post})}"),
           ("ratio weights L_1994 / L_1991: Kish size; largest normalised", f"{kish(r):.1f}; {r.max() / r.sum():.4f}")]
    for nm, x in (("u", u), ("f", f), ("V_2021 km3 (end of the 1991 | 1994 run)", None)):
        if x is None:
            out.append((nm + ": 5/50/95 % of the draws", " ".join(f"{v:.5f}" for v in wquant(v0, one, [.05, .5, .95]))
                        + " | " + " ".join(f"{v:.5f}" for v in wquant(v1, one, [.05, .5, .95]))))
        else:
            out.append((f"{nm} 5/50/95 %: 1991 dating | October 1994 dating (ratio weights)",
                        " ".join(f"{v:.3f}" for v in wquant(x, one, [.05, .5, .95])) + " | " +
                        " ".join(f"{v:.3f}" for v in wquant(x, r, [.05, .5, .95]))))
    for k in lo:
        out.append((f"term {k}: mean log-likelihood over the draws, 1991 | 1994 dating",
                    f"{lo[k].mean():.2f} | {ln[k].mean():.2f}"))
    ref = {q: v for q, v in (tuple(x.values()) for x in rows(RES / "step4_check94.csv"))}
    out.append(("step4_check94.csv for comparison: ratio-weight size; u", ref[
        "effective sample size of the ratio weights L_1994 / L_1991 over the draws"] + "; " +
        ref["u 5 / 50 / 95 %: 1991 dating | October 1994 dating"]))
    # copies: the same sample run again from 1994 with identical inputs
    by = {}
    for p in post:
        by.setdefault(p["sample"], []).append(new[int(p["draw"])])
    cols = {"area 2021 km2": ["A2021_1.0"], "speed m a-1": ["speed"], "radar m": [f"radar_{i}" for i in range(1, 7)],
            "dh1 m": [k for k in k91 if k.startswith("dh1_")], "V_2021 km3": ["V_2021"]}
    for nm, cs in cols.items():
        sd = [np.std([[float(x[c]) for c in cs] for x in v], 0, ddof=1).mean() for v in by.values() if len(v) > 1]
        out.append((f"copies ({sum(len(v) > 1 for v in by.values())} samples run more than once): {nm}, "
                    "median over samples of the SD between copies", f"{np.median(sd):.4g}"))
    tot = sum(ln.values())
    sdl = [np.std([tot[j] for j, p in enumerate(post) if p["sample"] == smp], ddof=1) for smp, v in by.items()
           if len(v) > 1]
    out.append(("copies: total log-likelihood (1994 dating), SD between copies, median | max over samples",
                f"{np.median(sdl):.2f} | {max(sdl):.2f}"))
    write(RES / "step4_verify_d94.csv", out)


if __name__ == "__main__":
    {"fields": fields, "weights": weights, "d94": d94}[sys.argv[1]]()
