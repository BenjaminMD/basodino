"""Start state and flow, one joint fit (decision 9 as amended, DESIGN.md step 4): samples, run lists, the model side
of every term, likelihood, importance weights, the 128 posterior draws.

Samples (round 1): 1024 of (balance draw, u, f): balance draw = a random permutation of the 1024 kept draws of the
accepted step-2 posterior (results/calib_draws.csv; r_ice = 0.012 Fm), (u, ln f) a scrambled Sobol sequence over the
prior box u uniform in [-1, 2], f log-uniform in [0.05, 4]; A = 78 f MPa-3 a-1, tau_ref = 0.2154 f^(-1/3) MPa.
Round 2 (DESIGN.md step 4 fallback, only when the round-1 effective sample size is below 128): 1024 more from a
proposal fitted to the weighted round 1: in (u, ln f) a Gaussian with the weighted mean and twice the weighted
covariance (weights tempered to an effective sample size of at least 8 for this fit only), truncated to the box,
mixed 0.9 / 0.1 per try with the prior (defensive mixture; density in log_q; the log_q column of
step4_samples_r2.csv was written with the uncorrected density and is not read by weigh); balance draws again a random permutation (not adapted). Weights of the pooled rounds by the balance heuristic: w = L p / ((n1 p + n2 q) / (n1 + n2)).
Each sample runs two segments (experiment fit, cold online solver, 3000 initial iterations, observed weather):
  _91  dem 1994 (the GLAMOS survey of 1991-09-18, answer 4), 1991.75-2021.75
  _21  dem 2021, 2021.75-2026.75
Check of decision 9 (stage d94): the 128 posterior draws rerun with the 1994 DEM dated October 1994 (1994.75-2021.75,
27 years) and re-weighted: the 1991-2021 terms from these runs, the 2021-2026 terms unchanged.

Model side (reduce, per run, from series.csv and fields.npz; observations and cells: inputs/step4_targets.npz,
targets_step4.py):
  areas  thk >= h (h = 1 m; 0.1 and 2 m kept for the variants): 2021-09-30 all cells (_91); 2024-08-23 all cells,
         2025-09-07 and 2026-09-27 the largest 4-connected body (_21)
  dh1    per band, mean over the band's cells of the modelled surface on 2012-09-01 minus the start surface (_91)
  dh2    per band, modelled surface 2021-09-30 minus 2012-09-01 (_91)
  radar  per profile, mean over the kept points of the modelled thickness, mean of the 1 October 2005 and 2006 states
         (the survey day of 2006 is not documented) (_91)
  speed  median over the saves of 2014-2021 (1 October) of track.py its_median (_91)
  stakes per bound, the largest start thickness (2021, geometry.start of the run's u) over its disc (_21)
Likelihood: per term, independent Gaussian values with variance sigma_obs^2 + s^2 (areas, dh1, dh2, radar), or a
one-sided kernel exp(-0.5 v^2 / (sigma_obs^2 + s^2)) of the violation v = max(0, bound - model) (stakes) or max(0,
model - bound) (speed); s the term's model-error SD, log-uniform in [0.1, 10] x S_REF (64 nodes), integrated out per
sample. Main variant: h = 1 m, stake bounds with the provisional 2026 values; variants h = 0.1, 2 m and the release
stake bounds (weights only, same runs); diagnostics (not used for the draws): s prior up to 3 x S_REF (s_top3) and
without the two dh terms (no_dh).
Outputs (results/): step4_samples_r{R}.csv, stages/step4_r{R}.txt (samples); step4_runs_{STAGE}.csv (reduce);
step4_weights.csv (per sample: terms, log-likelihood, weights of every variant, log weights of the rounds), step4_terms.csv (per value: observed,
sigma, MAP model, weighted 5/50/95 % of the model, residual of the median in sigma), step4_summary.csv, and
posterior_128.csv (128 draws by systematic resampling of the pooled weights: balance draw, u, f, weight, copies,
effective sample size) (weigh); stages/step4_d94.txt (d94); step4_check94.csv (check94).
usage:  Mac:  python weights.py samples 1 | samples 2 ; weigh ; d94 (run list) ; check94 (after its runs)
        s03 (env.sh):  python weights.py reduce STAGE      (runs/STAGE/* -> results/step4_runs_STAGE.csv)
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parent
RES, INP = IGM / "results", IGM / "inputs"
N, SEED, N_POST, ESS_MIN, ESS_FIT = 1024, 2026, 128, 128, 8
U_BOX, LNF_BOX = (-1.0, 2.0), (np.log(0.05), np.log(4.0))
A0, TAU0 = 78.0, 0.2153697218796329
S_REF = {"area": 0.1, "dh1": 2.93239, "dh2": 2.93239, "radar": 5.3, "stake": 1.0, "speed": 0.5}
def s_nodes(top=10.0):
    """64 log-spaced nodes of the model-error SD from 0.1 to top, x S_REF."""
    return np.exp(np.log(0.1) + (np.arange(64) + 0.5) / 64 * np.log(top / 0.1))


S_NODES = s_nodes()
TERMS = ("area", "dh1", "dh2", "radar", "stake", "speed")
H = (0.1, 1.0, 2.0)
DATES91 = ("20051001", "20061001", "20120901", "20210930")
PAR = ("Fm", "c_prec", "dT", "dg", "b")                  # clim_otl / smb_ti units: dg K m-1, b m-1


def log_prior():
    return -np.log((U_BOX[1] - U_BOX[0]) * (LNF_BOX[1] - LNF_BOX[0]))


def read_csv(path):
    return list(csv.DictReader(open(path)))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# --- samples and run lists -------------------------------------------------------------------------------------------
def balance_draws():
    rows = read_csv(RES / "calib_draws.csv")
    return np.array([[float(r["Fm_m_ice_per_day_degC"]), float(r["c_prec"]), float(r["dT_degC"]),
                      float(r["dg_K_per_km"]) / 1e3, float(r["b_percent_per_100m"]) / 1e4] for r in rows])


def proposal(rnd):
    """(mean, cov, Z, beta) of the round-rnd Gaussian in (u, ln f): moments of the pooled weights of rounds < rnd,
    tempered (w^beta, beta by bisection) so that their effective sample size is at least ESS_FIT (a shape from a
    handful of samples is not a shape); covariance doubled; Z its mass inside the box (Monte Carlo, 1e6)."""
    w = [r for r in read_csv(RES / "step4_weights.csv") if int(r["round"]) < rnd]
    x = np.array([[float(r["u"]), np.log(float(r["f"]))] for r in w])
    lw = np.array([float(r[f"lw_round{rnd - 1}"]) for r in w])
    lo, hi = 0.0, 1.0
    if ess(np.exp(lw - lw.max())) < ESS_FIT:
        for _ in range(60):
            b = (lo + hi) / 2
            lo, hi = (lo, b) if ess(np.exp(b * (lw - lw.max()))) < ESS_FIT else (b, hi)
    beta = lo if ess(np.exp(lw - lw.max())) < ESS_FIT else 1.0
    p = np.exp(beta * (lw - lw.max()))
    p /= p.sum()
    m = p @ x
    c = 2.0 * ((x - m).T * p) @ (x - m)
    z = np.random.default_rng(SEED + 99).multivariate_normal(m, c, 1_000_000)
    return m, c, inside(z).mean(), beta


def inside(x):
    return (x[:, 0] >= U_BOX[0]) & (x[:, 0] <= U_BOX[1]) & (x[:, 1] >= LNF_BOX[0]) & (x[:, 1] <= LNF_BOX[1])


def log_q(x, m, c, z):
    """Log density in (u, ln f) of what samples() draws: each try is the prior (0.1) or the Gaussian (0.9), a Gaussian
    draw outside the box is dropped and tried again, so q = (0.9 g 1_box + 0.1 p) / (0.9 Z + 0.1) (prior share of the
    kept samples 0.1 / (0.9 Z + 0.1), not 0.1; corrected 2026-10-08 by checks/step4_verify.py)."""
    d = x - m
    g = np.exp(-0.5 * np.einsum("ij,jk,ik->i", d, np.linalg.inv(c), d)) / (2 * np.pi * np.sqrt(np.linalg.det(c)))
    return np.log((0.9 * np.where(inside(x), g, 0.0) + 0.1 * np.exp(log_prior())) / (0.9 * z + 0.1))


def samples(rnd):
    from scipy.stats import qmc
    rng = np.random.default_rng(SEED + rnd)
    draws = balance_draws()
    perm = rng.permutation(len(draws))
    if rnd == 1:
        s = qmc.Sobol(2, scramble=True, seed=SEED).random(N)
        x = np.c_[U_BOX[0] + s[:, 0] * (U_BOX[1] - U_BOX[0]), LNF_BOX[0] + s[:, 1] * (LNF_BOX[1] - LNF_BOX[0])]
        lq = np.full(N, log_prior())
    else:
        m, c, z, beta = proposal(rnd)
        x = np.empty((0, 2))
        while len(x) < N:
            if rng.random() < 0.1:
                y = np.c_[rng.uniform(*U_BOX, 1), rng.uniform(*LNF_BOX, 1)]
            else:
                y = rng.multivariate_normal(m, c, 1)
                y = y[inside(y)]
            x = np.r_[x, y]
        lq = log_q(x, m, c, z)
        print(f"round {rnd} proposal: mean u {m[0]:.3f}, ln f {m[1]:.3f}; sd {np.sqrt(c[0, 0]):.3f}, "
              f"{np.sqrt(c[1, 1]):.3f}; correlation {c[0, 1] / np.sqrt(c[0, 0] * c[1, 1]):+.3f}; mass in box {z:.3f}; "
              f"tempering exponent of the round-{rnd - 1} weights {beta:.4f}")
    rows, lines = [], [f"# NAME EXPERIMENT overrides (run_stage.sh); DESIGN.md step 4, round {rnd}: weights.py samples "
                       f"{rnd}; per sample _91 (dem 1994, 1991.75-2021.75) and _21 (dem 2021, 2021.75-2026.75), cold"]
    for i in range(N):
        k = perm[i]
        u, f = x[i, 0], np.exp(x[i, 1])
        a, tau = A0 * f, TAU0 * f ** (-1 / 3)
        name = f"r{rnd}_{i:04d}"
        rows.append([name, rnd, k, *(f"{v:.7g}" for v in draws[k]), f"{u:.6f}", f"{f:.6f}", f"{a:.6g}", f"{tau:.7g}",
                     f"{lq[i]:.6f}"])
        p = draws[k]
        common = (f"inputs.geometry.u={u:.6f} processes.clim_otl.dT={p[2]:.7g} processes.clim_otl.dg={p[3]:.7g} "
                  f"processes.clim_otl.b={p[4]:.7g} processes.clim_otl.c_prec={p[1]:.7g} processes.smb_ti.Fm={p[0]:.7g} "
                  f"processes.iceflow.physics.viscosity.arrhenius={a:.6g} "
                  f"processes.iceflow.physics.sliding.tau_ref={tau:.7g}")
        lines.append(f"{name}_91 fit inputs.geometry.dem=1994 {common}")
        lines.append(f"{name}_21 fit inputs.geometry.dem=2021 processes.time.start=2021.75 processes.time.end=2026.75 "
                     f"{common}")
    write_csv(RES / f"step4_samples_r{rnd}.csv", ["name", "round", "balance_draw", *PAR, "u", "f", "arrhenius",
                                                  "tau_ref", "log_q"], rows)
    (IGM / f"stages/step4_r{rnd}.txt").write_text("\n".join(lines) + "\n")
    print(f"{N} samples, {2 * N} runs: results/step4_samples_r{rnd}.csv, stages/step4_r{rnd}.txt")


# --- model side of the terms (s03) -----------------------------------------------------------------------------------
def geometry_module():
    spec = importlib.util.spec_from_file_location("geometry", IGM / "user/code/inputs/geometry/geometry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def body(ice):
    from scipy.ndimage import label
    lab, n = label(ice)
    return (lab == np.argmax(np.bincount(lab[ice]))) if n else ice


def columns(t):
    c91 = ([f"A2021_{h}" for h in H] + [f"dh1_{k}" for k in range(len(t["dh1_obs"]))] +
           [f"dh2_{k}" for k in range(len(t["dh2_obs"]))] + [f"radar_{p}" for p in range(1, 7)] +
           ["speed", "V_start91", "V_2021", "v_med_1991", "v_p90_1991"])
    c21 = ([f"A{d}_{h}" for d in ("2024", "2025", "2026") for h in H] +
           [f"stake_{k}" for k in range(len(t["stake_bound"]))] + ["V_start21", "V_2026", "v_med_2021", "v_p90_2021"])
    return c91, c21


def reduce(stage):
    geo = geometry_module()
    g = dict(np.load(INP / "igm_static.npz"))
    t = dict(np.load(INP / "step4_targets.npz"))
    c91, c21 = columns(t)
    runs = {r["name"]: r for r in read_csv(IGM / "logs" / stage / "runs.csv")}
    over = {}
    for line in open(IGM / "logs" / stage / "queue.txt"):
        w = line.split()
        over[w[0]] = dict(x.split("=", 1) for x in w[2:])
    rows = []
    for name in sorted(runs):
        o, d = over[name], IGM / "runs" / stage / name
        row = {"name": name, "rc": runs[name]["rc"], "end": ""}
        if (d / "track_end.txt").exists():
            row["end"] = (d / "track_end.txt").read_text().split()[0]
        if runs[name]["rc"] != "0" or not (d / "fields.npz").exists():
            rows.append(row)
            continue
        F = np.load(d / "fields.npz")
        s = read_csv(d / "series.csv")
        dem, u = o["inputs.geometry.dem"], float(o["inputs.geometry.u"])
        topg0, thk0 = geo.start(g, dem, u)
        need = DATES91 if dem == "1994" else ("20240823", "20250907", "20260927")
        if any(f"thk_{k}" not in F for k in need):           # a run that did not reach its dates
            row["end"] = ""
            rows.append(row)
            continue
        if dem == "1994":
            th = {k: F[f"thk_{k}"] for k in DATES91}
            for h in H:
                row[f"A2021_{h}"] = f"{(th['20210930'] >= h).sum() * 1e-4:.4f}"
            s91 = topg0 + thk0
            s12, s21 = F["topg"] + th["20120901"], F["topg"] + th["20210930"]
            for k in range(len(t["dh1_obs"])):
                m = t["dh1_band"] == k
                row[f"dh1_{k}"] = f"{(s12 - s91)[m].mean():.4f}"
            for k in range(len(t["dh2_obs"])):
                m = t["dh2_band"] == k
                row[f"dh2_{k}"] = f"{(s21 - s12)[m].mean():.4f}"
            h06 = 0.5 * (th["20051001"] + th["20061001"]).ravel()[t["radar_cell"]]
            for p in range(1, 7):
                row[f"radar_{p}"] = f"{h06[t['radar_profile'] == p].mean():.4f}"
            row["speed"] = f"{np.median([float(r['its_median']) for r in s if 2014 <= int(r['year']) <= 2021]):.4f}"
            row["V_start91"], row["V_2021"] = s[0]["V_km3"], f"{th['20210930'].sum() * 1e-7:.6f}"
            row["v_med_1991"], row["v_p90_1991"] = s[0]["v_med"], s[0]["v_p90"]
        else:
            for d_, key, whole in (("2024", "20240823", True), ("2025", "20250907", False),
                                   ("2026", "20260927", False)):
                for h in H:
                    ice = F[f"thk_{key}"] >= h
                    row[f"A{d_}_{h}"] = f"{(ice if whole else body(ice)).sum() * 1e-4:.4f}"
            flat = thk0.ravel()
            for k, c in enumerate(t["stake_cells"]):
                row[f"stake_{k}"] = f"{flat[c[c >= 0]].max():.3f}"
            row["V_start21"], row["V_2026"] = s[0]["V_km3"], s[-1]["V_km3"]
            row["v_med_2021"], row["v_p90_2021"] = s[0]["v_med"], s[0]["v_p90"]
        rows.append(row)
    head = ["name", "rc", "end"] + c91 + c21
    write_csv(RES / f"step4_runs_{stage}.csv", head, [[r.get(k, "") for k in head] for r in rows])
    bad = [r["name"] for r in rows if r["rc"] != "0" or r["end"] == ""]
    print(f"{len(rows)} runs reduced, {len(bad)} failed or unfinished {bad[:10]}: results/step4_runs_{stage}.csv")


# --- likelihood and weights ------------------------------------------------------------------------------------------
def term_loglik(r, sig, sref, one_sided=False, nodes=S_NODES):
    """log of the s-integrated likelihood, r (samples, values) residuals (or violations), sig (values,)."""
    s = (nodes * sref)[:, None, None]
    var = sig[None, None, :] ** 2 + s ** 2
    ll = -0.5 * r[None] ** 2 / var
    if not one_sided:
        ll = ll - 0.5 * np.log(2 * np.pi * var)
    ll = ll.sum(-1)                                                # (nodes, samples)
    mx = ll.max(0)
    return mx + np.log(np.exp(ll - mx).mean(0)), ll


def model_values(R, t, h=1.0, stakes="with2026"):
    """Per term: (model, observed, sigma) arrays, model (samples, values)."""
    col = lambda names: np.array([[float(r[n]) for n in names] for r in R])
    sel = t[f"stake_{stakes}"]
    return {
        "area": (col([f"A2021_{h}", f"A2024_{h}", f"A2025_{h}", f"A2026_{h}"]), t["area_obs"], t["area_sigma"]),
        "dh1": (col([f"dh1_{k}" for k in range(len(t["dh1_obs"]))]), t["dh1_obs"], np.full(len(t["dh1_obs"]),
                                                                                         S_REF["dh1"])),
        "dh2": (col([f"dh2_{k}" for k in range(len(t["dh2_obs"]))]), t["dh2_obs"], np.full(len(t["dh2_obs"]),
                                                                                         S_REF["dh2"])),
        "radar": (col([f"radar_{p}" for p in range(1, 7)]), t["radar_profile_obs"], np.full(6, 5.3)),
        "stake": (col([f"stake_{k}" for k in np.flatnonzero(sel)]), t["stake_bound"][sel],
                  np.full(sel.sum(), S_REF["stake"])),
        "speed": (np.nan_to_num(col(["speed"])), np.atleast_1d(t["speed_bound"]), np.atleast_1d(t["speed_sigma"])),
    }


def loglik(R, t, s_top=10.0, drop=(), **variant):
    """Per term log-likelihood (terms in drop count 0) and E[s | sample]; s_top: upper end of the s prior, x S_REF."""
    out, cond = {}, {}
    nodes = s_nodes(s_top)
    for k, (m, o, sg) in model_values(R, t, **variant).items():
        if k == "stake":
            r, one = np.maximum(0.0, o[None] - m), True
        elif k == "speed":
            r, one = np.maximum(0.0, m - o[None]), True
        else:
            r, one = m - o[None], False
        out[k], ll = term_loglik(r, sg, S_REF[k], one, nodes)
        p = np.exp(ll - ll.max(0))
        cond[k] = (p * (nodes * S_REF[k])[:, None]).sum(0) / p.sum(0)     # E[s | sample]
        if k in drop:
            out[k] = np.zeros(len(R))
    return out, cond


def ess(w):
    return w.sum() ** 2 / (w ** 2).sum()


def wq(x, w, q):
    i = np.argsort(x)
    c = np.cumsum(w[i]) / w.sum()
    return np.interp(q, c, x[i])


def load(rounds):
    S, R = [], []
    for rnd in rounds:
        runs = {}
        for f in sorted(RES.glob(f"step4_runs_s4_r{rnd}*.csv")):
            for r in read_csv(f):
                if r["rc"] == "0" and r["end"]:
                    runs[r["name"]] = r
        for s in read_csv(RES / f"step4_samples_r{rnd}.csv"):
            a, b = runs.get(s["name"] + "_91"), runs.get(s["name"] + "_21")
            if a and b:
                S.append(s)
                R.append({**{k: v for k, v in a.items() if v != ""}, **{k: v for k, v in b.items() if v != ""}})
    return S, R


def log_q_all(S, rounds):
    """Balance-heuristic log denominator per sample: log((sum_r n_r q_r(x)) / sum_r n_r)."""
    x = np.array([[float(s["u"]), np.log(float(s["f"]))] for s in S])
    n = {r: sum(int(s["round"]) == r for s in S) for r in rounds}
    dens = np.zeros(len(S))
    for r in rounds:
        if r == 1:
            dens += n[r] * np.exp(log_prior())
        else:
            m, c, z, _ = proposal(r)
            dens += n[r] * np.exp(log_q(x, m, c, z))
    return np.log(dens / sum(n.values()))


def weigh():
    t = dict(np.load(RES / "step4_targets.npz"))   # copy of inputs/step4_targets.npz
    rounds = sorted(int(p.stem.split("_r")[-1]) for p in RES.glob("step4_samples_r*.csv"))
    rounds = [r for r in rounds if any(RES.glob(f"step4_runs_s4_r{r}*.csv"))]
    S, R = load(rounds)
    variants = {"main": {}, "h0.1": {"h": 0.1}, "h2": {"h": 2.0}, "release": {"stakes": "release"},
                "s_top3": {"s_top": 3.0}, "no_dh": {"drop": ("dh1", "dh2")}}       # the last two: diagnostics
    lq = log_q_all(S, rounds)
    LL, W, LW, cond = {}, {}, {}, None
    for v, kw in variants.items():
        terms, c = loglik(R, t, **kw)
        if v == "main":
            LL, cond = terms, c
        tot = sum(terms.values()) + log_prior() - lq
        W[v] = np.exp(tot - tot.max())
    for rnd in rounds:                       # rounds 1..rnd pooled (round 1: prior); the next proposal uses these
        sel = np.array([int(s["round"]) <= rnd for s in S])
        lqr = log_q_all([s for s, k in zip(S, sel) if k], range(1, rnd + 1))
        tot = sum(v[sel] for v in LL.values()) + log_prior() - lqr
        W[f"round{rnd}"], LW[f"round{rnd}"] = np.zeros(len(S)), np.full(len(S), -np.inf)
        W[f"round{rnd}"][sel] = np.exp(tot - tot.max())
        LW[f"round{rnd}"][sel] = tot - tot.max()
    total = sum(LL.values())
    w = W["main"]
    u = np.array([float(s["u"]) for s in S])
    f = np.array([float(s["f"]) for s in S])
    P = np.array([[float(s[k]) for k in PAR] for s in S])
    head = ["name", "round", "balance_draw", *PAR, "u", "f", *(f"ll_{k}" for k in TERMS), "ll_total", "log_q",
            *(f"s_{k}" for k in TERMS), *(f"w_{v}" for v in variants), *(f"w_round{r}" for r in rounds), *(f"lw_round{r}" for r in rounds)]
    rows = [[s["name"], s["round"], s["balance_draw"], *(s[k] for k in PAR), s["u"], s["f"],
             *(f"{LL[k][i]:.4f}" for k in TERMS), f"{total[i]:.4f}", f"{lq[i]:.6f}",
             *(f"{cond[k][i]:.4g}" for k in TERMS), *(f"{W[v][i] / W[v].sum():.6e}" for v in variants),
             *(f"{W[f'round{r}'][i] / max(W[f'round{r}'].sum(), 1e-300):.6e}" for r in rounds),
             *(f"{LW[f'round{r}'][i]:.4f}" for r in rounds)]
            for i, s in enumerate(S)]
    write_csv(RES / "step4_weights.csv", head, rows)

    # fit of every term: observed, MAP, weighted model 5/50/95 %, residual of the median in sigma
    imap = int(np.argmax(w))
    mv = model_values(R, t)
    labels = {"area": [f"area {d} ({k})" for d, k in zip(t["area_date"], t["area_kind"])],
              "dh1": [f"dh 1991-2012 band {e:.0f}-{e + 50:.0f} m" for e in t["dh1_edges"]],
              "dh2": [f"dh 2012-2021 band {e:.0f}-{e + 50:.0f} m" for e in t["dh2_edges"]],
              "radar": [f"radar 2006 profile {p}" for p in range(1, 7)],
              "stake": [f"stake {n} (lower bound)" for n in t["stake_name"][t["stake_with2026"]]],
              "speed": ["ITS_LIVE median 2014-2021 (upper bound)"]}
    trows = []
    for k in TERMS:
        m, o, sg = mv[k]
        for j in range(m.shape[1]):
            q = wq(m[:, j], w, [0.05, 0.5, 0.95])
            p0 = np.percentile(m[:, j], [5, 50, 95])
            trows.append([k, labels[k][j], f"{o[j]:.4f}", f"{sg[j]:.4f}", f"{m[imap, j]:.4f}", *(f"{v:.4f}" for v in q),
                          f"{(q[1] - o[j]) / sg[j]:+.2f}", *(f"{v:.4f}" for v in p0)])
    write_csv(RES / "step4_terms.csv", ["term", "value", "observed", "sigma", "model_MAP", "model_p05", "model_p50",
                                        "model_p95", "median_minus_observed_in_sigma", "prior_model_p05",
                                        "prior_model_p50", "prior_model_p95"], trows)

    # summary
    sm = []
    add = lambda q, v: sm.append((q, v))
    add("runs used (samples with both segments ended), rounds", f"{len(S)}, {rounds}")
    for v in variants:
        add(f"effective sample size, {v} (pooled rounds, Kish)", f"{ess(W[v]):.1f}")
    for r in rounds:
        add(f"effective sample size, rounds 1 to {r} pooled", f"{ess(W[f'round{r}']):.1f}")
    add("largest normalised weight (main)", f"{w.max() / w.sum():.4f}")
    add("gate: effective sample size >= 128 (main)", f"{ess(w):.1f} {'pass' if ess(w) >= ESS_MIN else 'FAIL'}")
    for v in variants:
        q = wq(u, W[v], [0.05, 0.5, 0.95])
        add(f"u 5 / 50 / 95 %, {v}", " ".join(f"{x:+.3f}" for x in q))
    for v in variants:
        q = wq(f, W[v], [0.05, 0.5, 0.95])
        add(f"f 5 / 50 / 95 %, {v}", " ".join(f"{x:.3f}" for x in q))
    for name, x, box in (("u", u, U_BOX), ("ln f", np.log(f), LNF_BOX)):
        hist = np.histogram(x, 20, box, weights=w)[0] / w.sum()
        add(f"{name}: weighted share in the 20 bins of the prior range (low to high)",
            " ".join(f"{v:.3f}" for v in hist))
        add(f"{name}: piled at the lower / upper edge (edge bin holds the most weight)",
            f"{'yes' if hist.argmax() == 0 else 'no'} / {'yes' if hist.argmax() == 19 else 'no'}")
    lf = np.log(f)
    grid = np.linspace(*LNF_BOX, 201)
    cdf = np.array([w[lf <= g].sum() for g in grid]) / w.sum()
    add("ln f: largest |posterior CDF - prior CDF| (0 = flat, as the prior)",
        f"{np.abs(cdf - (grid - LNF_BOX[0]) / (LNF_BOX[1] - LNF_BOX[0])).max():.3f}")
    X = np.c_[u, lf, P]
    names = ["u", "ln f", *PAR]
    m = w @ X / w.sum()
    C = ((X - m).T * w) @ (X - m) / w.sum()
    D = np.sqrt(np.diag(C))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if i < 2:
                add(f"weighted correlation {names[i]} - {names[j]}", f"{C[i, j] / D[i] / D[j]:+.3f}")
    sp = mv["speed"][0][:, 0]
    add("speed (ITS_LIVE median 2014-2021) m a-1: weighted 5 / 50 / 95 %; bound",
        " ".join(f"{x:.3f}" for x in wq(sp, w, [0.05, 0.5, 0.95])) + f"; {float(t['speed_bound']):.1f}")
    add("speed: weighted share above the bound; share of all samples above", f"{w[sp > 1].sum() / w.sum():.3f}; "
        f"{(sp > 1).mean():.3f}")
    for k in TERMS:
        add(f"term {k}: log-likelihood at MAP, weighted mean; model-error SD E[s] weighted mean",
            f"{LL[k][imap]:.2f}, {w @ LL[k] / w.sum():.2f}; {w @ cond[k] / w.sum():.4g}")
    add("MAP sample: name, balance draw, u, f", f"{S[imap]['name']} {S[imap]['balance_draw']} {u[imap]:+.4f} "
        f"{f[imap]:.4f}")
    for key in ("V_2021", "V_2026"):
        x = np.array([float(r[key]) for r in R])
        add(f"{key} km3 (1991 run for 2021; 2021 run for 2026): weighted 5 / 50 / 95 %",
            " ".join(f"{v:.5f}" for v in wq(x, w, [0.05, 0.5, 0.95])))
    x = np.array([float(r["V_start21"]) for r in R])
    add("V_2021 at the start of the 2021 run (geometry of u) km3: weighted 5 / 50 / 95 %",
        " ".join(f"{v:.5f}" for v in wq(x, w, [0.05, 0.5, 0.95])))
    write_csv(RES / "step4_summary.csv", ["quantity", "value"], sm)
    for q, v in sm:
        print(f"{q}: {v}")

    # 128 posterior draws: systematic resampling of the pooled weights
    p = w / w.sum()
    pos = (np.random.default_rng(SEED).random() + np.arange(N_POST)) / N_POST
    idx = np.minimum(np.searchsorted(np.cumsum(p), pos), len(p) - 1)
    copies = np.bincount(idx, minlength=len(p))
    e = ess(w)
    prow = []
    for j, i in enumerate(idx):
        s = S[i]
        fi = float(s["f"])
        prow.append([j, s["name"], s["round"], s["balance_draw"], *(s[k] for k in PAR), f"{0.012 * float(s['Fm']):.7g}",
                     s["u"], s["f"], f"{A0 * fi:.6g}", f"{TAU0 * fi ** (-1 / 3):.7g}", f"{p[i]:.6e}", copies[i],
                     f"{e:.1f}"])
    write_csv(RES / "posterior_128.csv", ["draw", "sample", "round", "balance_draw", *PAR, "r_ice", "u", "f",
                                          "arrhenius", "tau_ref", "weight", "copies", "ess"], prow)
    print(f"posterior_128.csv: {len(set(idx))} unique samples of {len(S)}, effective sample size {e:.1f}")


# --- check of decision 9: the 1994 DEM dated October 1994 -------------------------------------------------------------
def d94_list():
    """stages/step4_d94.txt: every draw of posterior_128.csv (copies of a sample rerun too: the online solver is not
    bit-reproducible, so the copies also show the solver's run-to-run spread), dem 1994 from 1994.75 to 2021.75."""
    post = read_csv(RES / "posterior_128.csv")
    lines = ["# NAME EXPERIMENT overrides (run_stage.sh); DESIGN.md step 4 check: the 128 draws of results/posterior_128.csv",
             "# rerun with the 1994 DEM dated October 1994 (1994.75-2021.75), cold; weights.py d94"]
    for s in post:
        f = float(s["f"])
        lines.append(f"d94_{int(s['draw']):03d} fit inputs.geometry.dem=1994 processes.time.start=1994.75 "
                     f"processes.time.end=2021.75 inputs.geometry.u={s['u']} processes.clim_otl.dT={s['dT']} "
                     f"processes.clim_otl.dg={s['dg']} processes.clim_otl.b={s['b']} "
                     f"processes.clim_otl.c_prec={s['c_prec']} processes.smb_ti.Fm={s['Fm']} "
                     f"processes.iceflow.physics.viscosity.arrhenius={A0 * f:.6g} "
                     f"processes.iceflow.physics.sliding.tau_ref={TAU0 * f ** (-1 / 3):.7g}")
    (IGM / "stages/step4_d94.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(lines) - 2} runs: stages/step4_d94.txt")


def check94():
    """Re-weight the 128 draws: 1991-2021 terms from the October-1994 runs, 2021-2026 terms unchanged; ratio weights
    L_new / L_old over the 128 (they are a posterior sample under L_old). -> results/step4_check94.csv"""
    t = dict(np.load(RES / "step4_targets.npz"))
    post = read_csv(RES / "posterior_128.csv")
    S, R = load(sorted({int(r["round"]) for r in post}))
    by = {s["name"]: r for s, r in zip(S, R)}
    new = {}                                # s4_d94, then s4_d94fix (draws whose sample changed) replaces its draws
    for f in sorted(RES.glob("step4_runs_s4_d94*.csv")):
        new.update({int(r["name"][4:]): r for r in read_csv(f) if r["rc"] == "0" and r["end"]})
    keep = [p for p in post if int(p["draw"]) in new]
    old = [by[p["sample"]] for p in keep]
    c91 = columns(t)[0]
    upd = [{**o, **{k: new[int(p["draw"])][k] for k in c91}} for p, o in zip(keep, old)]
    lo, _ = loglik(old, t)
    ln, _ = loglik(upd, t)
    r = np.exp(sum(ln.values()) - sum(lo.values()) - (sum(ln.values()) - sum(lo.values())).max())
    u = np.array([float(p["u"]) for p in keep])
    f = np.array([float(p["f"]) for p in keep])
    one = np.ones(len(keep))
    rows = [("draws rerun and ended (of 128), unique samples among them", f"{len(keep)}, {len({p['sample'] for p in keep})}"),
            ("effective sample size of the ratio weights L_1994 / L_1991 over the draws", f"{ess(r):.1f}"),
            ("largest normalised ratio weight", f"{r.max() / r.sum():.4f}")]
    for name, x in (("u", u), ("f", f)):
        rows.append((f"{name} 5 / 50 / 95 %: 1991 dating | October 1994 dating",
                     " ".join(f"{v:.3f}" for v in wq(x, one, [0.05, 0.5, 0.95])) + " | " +
                     " ".join(f"{v:.3f}" for v in wq(x, r, [0.05, 0.5, 0.95]))))
    for k in ("area", "dh1", "dh2", "radar", "speed"):
        rows.append((f"term {k}: mean log-likelihood over the draws, 1991 | 1994 dating",
                     f"{lo[k].mean():.2f} | {ln[k].mean():.2f}"))
    mo, mn = model_values(old, t), model_values(upd, t)
    for k, labels in (("area", ["area 2021-09-30"]), ("speed", ["ITS_LIVE median 2014-2021"]),
                      ("radar", [f"radar profile {p}" for p in range(1, 7)])):
        for j, lab in enumerate(labels):
            rows.append((f"{lab}: observed; model median of the draws, 1991 | 1994 dating",
                         f"{mo[k][1][j]:.3f}; {np.median(mo[k][0][:, j]):.3f} | {np.median(mn[k][0][:, j]):.3f}"))
    for k, lab in (("dh1", "dh 1991-2012"), ("dh2", "dh 2012-2021")):
        d_o = (np.median(mo[k][0], 0) - mo[k][1])
        d_n = (np.median(mn[k][0], 0) - mn[k][1])
        rows.append((f"{lab}: root mean square over the bands of (model median - observed) m, 1991 | 1994 dating",
                     f"{np.sqrt((d_o ** 2).mean()):.2f} | {np.sqrt((d_n ** 2).mean()):.2f}"))
    write_csv(RES / "step4_check94.csv", ["quantity", "value"], rows)
    for q, v in rows:
        print(f"{q}: {v}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "samples":
        samples(int(sys.argv[2]))
    elif cmd == "reduce":
        reduce(sys.argv[2])
    elif cmd == "weigh":
        weigh()
    elif cmd == "d94":
        d94_list()
    elif cmd == "check94":
        check94()
