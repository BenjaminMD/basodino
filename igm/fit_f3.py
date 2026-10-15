"""F3 of the fix round (DESIGN.md): the start state and flow refitted on 2021-2026 from the 2021 surface, with the F1
balance posterior: samples, run lists, the model side of every term, likelihood, importance weights, the 128 draws.

Samples: (balance draw, u, f). Balance draw = one of the 1024 kept draws of results/calib_f1_draws.csv (Fm, c_prec,
dT, dg, b, dT_pre; r_ice = 0.012 Fm), a random permutation per round; u uniform in [-1, 2] (decision 8, 50 m edge
taper); f log-uniform in [0.05, 20]; A = 78 f MPa-3 a-1, tau_ref = 0.2154 f^(-1/3) MPa at 100 m a-1.
Round 1: (u, ln f) a scrambled Sobol sequence over the box. Round R > 1: a Gaussian in (u, ln f) with the weighted
mean and twice the weighted covariance of the pooled rounds before R (weights tempered to an effective sample size
of at least ESS_FIT for this fit only), truncated to the box, mixed 0.9 / 0.1 per try with the prior (density
log_q = log((0.9 g 1_box + 0.1 p) / (0.9 Z + 0.1)), as weights.py after its 2026-10-08 correction).
Pooled weights by the balance heuristic: w = L p / (sum_r n_r q_r / sum_r n_r).
Each sample runs two segments (experiment fit with the basin process of F2, observed weather, warm start from a
network trained cold on the segment's start state at u = 0, f = 1 plus WARM_NBIT iterations):
  _12  dem 2012 (the 2012 surface of Grab et al. 2021 on SGI2010 and the 2021 mask, geometry.py), 2012.75-2021.75
  _21  dem 2021, 2021.75-2026.75
Terms (observations of inputs/step4_targets.npz, targets_step4.py, unless noted; each value Gaussian with variance
sigma_obs^2 + s^2, or one-sided with a kernel exp(-0.5 v^2 / (sigma_obs^2 + s^2)) of the violation v, s the term's
model-error SD, log-uniform in [0.1, 10] x S_REF, integrated out per sample (weights.term_loglik); variants, weights
only: ice threshold h = 0.1 and 2 m, stake release values, without dh2, without speed, speed_strict = the speed bound
with sigma_obs only):
  area   _21: thk >= 1 m on 2024-08-23 (all cells; swissTLM3D 1.3801 +- 0.112 km2), 2025-09-07 and 2026-09-27 (the
         largest 4-connected body; Sentinel-2 1.2311 +- 0.0841, 1.0210 +- 0.0925 km2)
  dh2    _12: per 50 m band of the 2012 surface, mean over the band's cells of the modelled surface on 2021-09-30
         minus the start surface (2012-10-01; the observed change runs from 2012-09-01: the September 2012 balance is
         left out, see DESIGN.md), sigma 2.93 m per band
  radar  bed(u) against the 2006 radar bedrock (inputs/radar_2006.csv, the 936 points whose bedrock lies below
         usurf_2021): mean per profile, sigma 5.3 m (the step-4 radar term compared the 2006 thickness from a 1991
         run; with the start in 2021 the bed itself is compared)
  stake  one-sided lower bounds: the largest 2021 start thickness over each stake's disc at least the melt since
         September 2021 (with the provisional 2026 values), sigma 1 m
  speed  one-sided upper bound: _12, median over the 1 October saves of 2014-2021 of the median over the 96 ITS_LIVE
         pixels of the surface speed (track its_median) at most 1 m a-1, sigma 0.5 m a-1
Outputs (results/): f3_samples_r{R}.csv and stages/f3_r{R}.txt (samples); f3_runs_{STAGE}.csv (reduce);
f3_weights.csv, f3_terms.csv, f3_summary.csv, posterior_f3_128.csv (weigh: 128 draws by systematic resampling of the
pooled weights, with each draw's modelled areas, volumes and speed).
usage:  Mac:  python fit_f3.py samples R ; weigh ; warm (run list) ; warmcompare (after its runs)
        s03 (env.sh):  python fit_f3.py reduce STAGE      (runs/STAGE/* -> results/f3_runs_STAGE.csv)
"""
import csv
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parent
sys.path.insert(0, str(IGM))
from weights import body, ess, geometry_module, term_loglik, wq  # noqa: E402

RES, INP = IGM / "results", IGM / "inputs"
N, SEED, N_POST, ESS_MIN, ESS_FIT = 1024, 3026, 128, 128, 16
U_BOX, LNF_BOX = (-1.0, 2.0), (np.log(0.05), np.log(20.0))
A0, TAU0, RATIO = 78.0, 0.2153697218796329, 0.012
WARM_NBIT = 300
NETS = {"12": "nets/f3_2012_u0_f1_n3000.npz", "21": "nets/fit2021_u0_f1_n3000.npz"}
S_REF = {"area": 0.1, "dh2": 2.93239, "radar": 5.3, "stake": 1.0, "speed": 0.5}
TERMS = tuple(S_REF)
PAR = ("Fm", "c_prec", "dT", "dg", "b", "dT_pre")           # clim_otl / smb_ti units: dg K m-1, b m-1
AREAS = (("2024", "20240823", True), ("2025", "20250907", False), ("2026", "20260927", False))


def read_csv(path):
    return list(csv.DictReader(open(path)))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def log_prior():
    return -np.log((U_BOX[1] - U_BOX[0]) * (LNF_BOX[1] - LNF_BOX[0]))


def inside(x):
    return (x[:, 0] >= U_BOX[0]) & (x[:, 0] <= U_BOX[1]) & (x[:, 1] >= LNF_BOX[0]) & (x[:, 1] <= LNF_BOX[1])


# --- samples and run lists -------------------------------------------------------------------------------------------
def balance_draws():
    rows = read_csv(RES / "calib_f1_draws.csv")
    return np.array([[float(r["Fm_m_ice_per_day_degC"]), float(r["c_prec"]), float(r["dT_degC"]),
                      float(r["dg_K_per_km"]) / 1e3, float(r["b_percent_per_100m"]) / 1e4,
                      float(r["dT_pre_degC"])] for r in rows])


def proposal(rnd):
    """(mean, cov, Z, beta) of the round-rnd Gaussian in (u, ln f) from the pooled weights of rounds < rnd
    (f3_weights.csv column lw_round{rnd-1}), tempered (w^beta) to an effective sample size >= ESS_FIT, covariance
    doubled; Z its mass inside the box (Monte Carlo, 1e6)."""
    w = [r for r in read_csv(RES / "f3_weights.csv") if int(r["round"]) < rnd]
    x = np.array([[float(r["u"]), np.log(float(r["f"]))] for r in w])
    lw = np.array([float(r[f"lw_round{rnd - 1}"]) for r in w])
    lw = lw - lw.max()
    beta = 1.0
    if ess(np.exp(lw)) < ESS_FIT:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            b = (lo + hi) / 2
            lo, hi = (lo, b) if ess(np.exp(b * lw)) < ESS_FIT else (b, hi)
        beta = lo
    p = np.exp(beta * lw)
    p /= p.sum()
    m = p @ x
    c = 2.0 * ((x - m).T * p) @ (x - m)
    z = np.random.default_rng(SEED + 99 + rnd).multivariate_normal(m, c, 1_000_000)
    return m, c, inside(z).mean(), beta


def log_q(x, m, c, z):
    d = x - m
    g = np.exp(-0.5 * np.einsum("ij,jk,ik->i", d, np.linalg.inv(c), d)) / (2 * np.pi * np.sqrt(np.linalg.det(c)))
    return np.log((0.9 * np.where(inside(x), g, 0.0) + 0.1 * np.exp(log_prior())) / (0.9 * z + 0.1))


def overrides(p, u, f):
    return (f"inputs.geometry.u={u:.6f} processes.clim_otl.dT={p[2]:.7g} processes.clim_otl.dg={p[3]:.7g} "
            f"processes.clim_otl.b={p[4]:.7g} processes.clim_otl.c_prec={p[1]:.7g} "
            f"processes.clim_otl.dT_pre={p[5]:.7g} processes.smb_ti.Fm={p[0]:.7g} "
            f"processes.iceflow.physics.viscosity.arrhenius={A0 * f:.6g} "
            f"processes.iceflow.physics.sliding.tau_ref={TAU0 * f ** (-1 / 3):.7g}")


def segment(name, seg, common, warm=True):
    t = {"12": "inputs.geometry.dem=2012 processes.time.start=2012.75 processes.time.end=2021.75",
         "21": "inputs.geometry.dem=2021 processes.time.start=2021.75 processes.time.end=2026.75"}[seg]
    w = (f" processes.iceflow.unified.nbit_init=0 processes.warmstart.net={NETS[seg]} "
         f"processes.warmstart.nbit={WARM_NBIT}") if warm else ""
    return f"{name}_{seg} fit {t} {common}{w}"


def samples(rnd):
    from scipy.stats import qmc
    rng = np.random.default_rng(SEED + rnd)
    draws = balance_draws()
    perm = rng.permutation(len(draws))
    if rnd == 1:
        s = qmc.Sobol(2, scramble=True, seed=SEED).random(N)
        x = np.c_[U_BOX[0] + s[:, 0] * (U_BOX[1] - U_BOX[0]), LNF_BOX[0] + s[:, 1] * (LNF_BOX[1] - LNF_BOX[0])]
        lq = np.full(N, log_prior())
        note = "prior (Sobol)"
    else:
        m, c, z, beta = proposal_saved(rnd)
        x = np.empty((0, 2))
        while len(x) < N:
            if rng.random() < 0.1:
                y = np.c_[rng.uniform(*U_BOX, 1), rng.uniform(*LNF_BOX, 1)]
            else:
                y = rng.multivariate_normal(m, c, 1)
                y = y[inside(y)]
            x = np.r_[x, y]
        lq = log_q(x, m, c, z)
        note = (f"proposal mean u {m[0]:.3f}, ln f {m[1]:.3f}; sd {np.sqrt(c[0, 0]):.3f}, {np.sqrt(c[1, 1]):.3f}; "
                f"correlation {c[0, 1] / np.sqrt(c[0, 0] * c[1, 1]):+.3f}; mass in box {z:.3f}; tempering {beta:.4f}")
    rows, lines = [], [f"# NAME EXPERIMENT overrides (run_stage.sh); fix round F3, round {rnd} (fit_f3.py samples "
                       f"{rnd}): {note}; per sample _12 (dem 2012, 2012.75-2021.75) and _21 (dem 2021, "
                       f"2021.75-2026.75), warm start {WARM_NBIT} iterations"]
    for i in range(N):
        k = perm[i]
        u, f = x[i, 0], np.exp(x[i, 1])
        name = f"f3r{rnd}_{i:04d}"
        rows.append([name, rnd, k, *(f"{v:.7g}" for v in draws[k]), f"{u:.6f}", f"{f:.6f}", f"{A0 * f:.6g}",
                     f"{TAU0 * f ** (-1 / 3):.7g}", f"{lq[i]:.6f}"])
        common = overrides(draws[k], u, f)
        lines += [segment(name, "12", common), segment(name, "21", common)]
    write_csv(RES / f"f3_samples_r{rnd}.csv", ["name", "round", "balance_draw", *PAR, "u", "f", "arrhenius", "tau_ref",
                                               "log_q"], rows)
    (IGM / f"stages/f3_r{rnd}.txt").write_text("\n".join(lines) + "\n")
    print(f"{N} samples, {2 * N} runs: results/f3_samples_r{rnd}.csv, stages/f3_r{rnd}.txt; {note}")


# --- model side (s03) ------------------------------------------------------------------------------------------------
def radar_targets():
    """(cells, profile, bedrock) of the kept radar points (bedrock below usurf_2021), lattice rows south to north."""
    rows = [r for r in read_csv(INP / "radar_2006.csv") if r["above_2021"] == "0"]
    g = np.load(INP / "igm_static.npz")
    ny, nx = g["usurf_2021"].shape
    cell = np.array([(ny - 1 - int(r["row"])) * nx + int(r["col"]) for r in rows])
    return cell, np.array([int(r["profile"]) for r in rows]), np.array([float(r["bedrock"]) for r in rows])


def columns(t):
    c12 = [f"dh2_{k}" for k in range(len(t["dh2_obs"]))] + ["speed", "V_start12", "A_start12", "V_2021_from12",
                                                            "A_2021_from12", "v_med_2012", "v_p90_2012"]
    c21 = ([f"A{d}_{h}" for d, _, _ in AREAS for h in (0.1, 1.0, 2.0)] + [f"radar_{p}" for p in range(1, 7)] +
           [f"stake_{k}" for k in range(len(t["stake_bound"]))] +
           ["V_start21", "V_2026", "A_2026_all", "v_med_2021", "v_p90_2021", "v_max_2021", "speed_2021_its",
            "basin_melt_km3"])
    return c12, c21


def reduce(stage):
    geo = geometry_module()
    g = dict(np.load(INP / "igm_static.npz"))
    t = dict(np.load(INP / "step4_targets.npz"))
    s12 = np.load(INP / "step5_targets.npz")["s12"].astype(np.float64)
    rc, rp, rb = radar_targets()
    write_csv(RES / "f3_radar.csv", ["profile", "points", "bedrock_mean"],
              [[p, int((rp == p).sum()), f"{rb[rp == p].mean():.4f}"] for p in range(1, 7)])
    c12, c21 = columns(t)
    runs = {r["name"]: r for r in read_csv(IGM / "logs" / stage / "runs.csv")}
    over = {}
    for line in open(IGM / "logs" / stage / "queue.txt"):
        w = line.split()
        over[w[0]] = dict(x.split("=", 1) for x in w[2:])
    rows = []
    for name in sorted(runs):
        o, d = over[name], IGM / "runs" / stage / name
        row = {"name": name, "rc": runs[name]["rc"], "end": "",
               "wall_s": int(runs[name]["end"]) - int(runs[name]["start"])}
        if (d / "track_end.txt").exists():
            row["end"] = (d / "track_end.txt").read_text().split()[0]
        if runs[name]["rc"] != "0" or not (d / "fields.npz").exists():
            rows.append(row)
            continue
        F = np.load(d / "fields.npz")
        s = read_csv(d / "series.csv")
        dem, u = o["inputs.geometry.dem"], float(o["inputs.geometry.u"])
        if dem == "2012":
            if "thk_20210930" not in F:
                row["end"] = ""
                rows.append(row)
                continue
            topg0, thk0 = geo.start(g, "2012", u, s12=s12)
            z0, z1 = topg0 + thk0, F["topg"] + F["thk_20210930"]
            for k in range(len(t["dh2_obs"])):
                row[f"dh2_{k}"] = f"{(z1 - z0)[t['dh2_band'] == k].mean():.4f}"
            sp = [float(r["its_median"]) for r in s if 2014 <= int(r["year"]) <= 2021]
            row["speed"] = f"{np.median(sp):.4f}"
            row["V_start12"], row["A_start12"] = s[0]["V_km3"], s[0]["A1_km2"]
            row["V_2021_from12"] = f"{F['thk_20210930'].sum() * 1e-7:.6f}"
            row["A_2021_from12"] = f"{(F['thk_20210930'] >= 1.0).sum() * 1e-4:.4f}"
            row["v_med_2012"], row["v_p90_2012"] = s[0]["v_med"], s[0]["v_p90"]
        else:
            if any(f"thk_{k}" not in F for _, k, _ in AREAS):
                row["end"] = ""
                rows.append(row)
                continue
            for d_, key, whole in AREAS:
                for h in (0.1, 1.0, 2.0):
                    ice = F[f"thk_{key}"] >= h
                    row[f"A{d_}_{h}"] = f"{(ice if whole else body(ice)).sum() * 1e-4:.4f}"
            topg0, thk0 = geo.start(g, "2021", u)
            b = geo.bed(u, g).ravel()[rc]
            for p in range(1, 7):
                row[f"radar_{p}"] = f"{b[rp == p].mean():.4f}"
            flat = thk0.ravel()
            for k, c in enumerate(t["stake_cells"]):
                row[f"stake_{k}"] = f"{flat[c[c >= 0]].max():.3f}"
            row["V_start21"], row["V_2026"], row["A_2026_all"] = s[0]["V_km3"], s[-1]["V_km3"], s[-1]["A1_km2"]
            row["v_med_2021"], row["v_p90_2021"], row["v_max_2021"] = s[0]["v_med"], s[0]["v_p90"], s[0]["v_max"]
            row["speed_2021_its"] = s[0]["its_median"]
            row["basin_melt_km3"] = f"{sum(float(r['basin_melt_km3'] or 0) for r in s[1:]):.6f}"
        rows.append(row)
    head = ["name", "rc", "end", "wall_s"] + c12 + c21
    write_csv(RES / f"f3_runs_{stage}.csv", head, [[r.get(k, "") for k in head] for r in rows])
    bad = [r["name"] for r in rows if r["rc"] != "0" or r["end"] == ""]
    print(f"{len(rows)} runs reduced, {len(bad)} failed or unfinished {bad[:10]}: results/f3_runs_{stage}.csv")


# --- likelihood and weights (Mac) --------------------------------------------------------------------------------------
def radar_obs():
    """Observed mean radar bedrock per profile (results/f3_radar.csv, written by reduce from inputs/radar_2006.csv)."""
    rows = read_csv(RES / "f3_radar.csv")
    return np.array([float(r["bedrock_mean"]) for r in rows])


def model_values(R, t, h=1.0, stakes="with2026"):
    col = lambda names: np.array([[float(r[n]) for n in names] for r in R])
    sel = t[f"stake_{stakes}"]
    return {
        "area": (col([f"A{d}_{h}" for d, _, _ in AREAS]), t["area_obs"][1:], t["area_sigma"][1:]),
        "dh2": (col([f"dh2_{k}" for k in range(len(t["dh2_obs"]))]), t["dh2_obs"], np.full(len(t["dh2_obs"]),
                                                                                         S_REF["dh2"])),
        "radar": (col([f"radar_{p}" for p in range(1, 7)]), radar_obs(), np.full(6, 5.3)),
        "stake": (col([f"stake_{k}" for k in np.flatnonzero(sel)]), t["stake_bound"][sel],
                  np.full(sel.sum(), S_REF["stake"])),
        "speed": (np.nan_to_num(col(["speed"])), np.atleast_1d(t["speed_bound"]), np.atleast_1d(t["speed_sigma"])),
    }


def loglik(R, t, drop=(), strict=(), **variant):
    """Per term log-likelihood (terms in drop count 0; terms in strict without a model-error SD, sigma_obs only)
    and E[s | sample]."""
    out, cond = {}, {}
    for k, (m, o, sg) in model_values(R, t, **variant).items():
        nodes = np.array([1e-9]) if k in strict else np.exp(np.log(0.1) + (np.arange(64) + 0.5) / 64 * np.log(100.0))
        if k == "stake":
            r, one = np.maximum(0.0, o[None] - m), True
        elif k == "speed":
            r, one = np.maximum(0.0, m - o[None]), True
        else:
            r, one = m - o[None], False
        out[k], ll = term_loglik(r, sg, S_REF[k], one, nodes)
        p = np.exp(ll - ll.max(0))
        cond[k] = (p * (nodes * S_REF[k])[:, None]).sum(0) / p.sum(0)
        if k in drop:
            out[k] = np.zeros(len(R))
    return out, cond


def load(rounds):
    S, R = [], []
    for rnd in rounds:
        runs = {}
        for f in sorted(RES.glob(f"f3_runs_f3_r{rnd}*.csv")):
            for r in read_csv(f):
                if r["rc"] == "0" and r["end"]:
                    runs[r["name"]] = r
        for s in read_csv(RES / f"f3_samples_r{rnd}.csv"):
            a, b = runs.get(s["name"] + "_12"), runs.get(s["name"] + "_21")
            if a and b:
                S.append(s)
                R.append({**{k: v for k, v in a.items() if v != ""}, **{k: v for k, v in b.items() if v != ""}})
    return S, R


def log_q_all(S, rounds):
    x = np.array([[float(s["u"]), np.log(float(s["f"]))] for s in S])
    n = {r: sum(int(s["round"]) == r for s in S) for r in rounds}
    dens = np.zeros(len(S))
    for r in rounds:
        if n[r] == 0:
            continue
        if r == 1:
            dens += n[r] * np.exp(log_prior())
        else:
            m, c, z, _ = proposal_saved(r)
            dens += n[r] * np.exp(log_q(x, m, c, z))
    return np.log(dens / sum(n.values()))


def proposal_saved(r):
    """The round-r proposal as written when its samples were drawn (results/f3_proposal_r{r}.npz)."""
    with np.load(RES / f"f3_proposal_r{r}.npz") as p:
        return p["m"], p["c"], float(p["z"]), float(p["beta"])


def weigh():
    t = dict(np.load(RES / "step4_targets.npz"))
    rounds = sorted(int(p.stem.split("_r")[-1]) for p in RES.glob("f3_samples_r*.csv"))
    rounds = [r for r in rounds if any(RES.glob(f"f3_runs_f3_r{r}*.csv"))]
    S, R = load(rounds)
    variants = {"main": {}, "h0.1": {"h": 0.1}, "h2": {"h": 2.0}, "release": {"stakes": "release"},
                "no_dh2": {"drop": ("dh2",)}, "no_speed": {"drop": ("speed",)}, "speed_strict": {"strict": ("speed",)}}
    lq = log_q_all(S, rounds)
    LL, W, LW, cond = {}, {}, {}, None
    for v, kw in variants.items():
        terms, c = loglik(R, t, **kw)
        if v == "main":
            LL, cond = terms, c
        tot = sum(terms.values()) + log_prior() - lq
        W[v] = np.exp(tot - tot.max())
    for rnd in rounds:
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
            *(f"s_{k}" for k in TERMS), *(f"w_{v}" for v in variants), *(f"w_round{r}" for r in rounds),
            *(f"lw_round{r}" for r in rounds)]
    rows = [[s["name"], s["round"], s["balance_draw"], *(s[k] for k in PAR), s["u"], s["f"],
             *(f"{LL[k][i]:.4f}" for k in TERMS), f"{total[i]:.4f}", f"{lq[i]:.6f}",
             *(f"{cond[k][i]:.4g}" for k in TERMS), *(f"{W[v][i] / W[v].sum():.6e}" for v in variants),
             *(f"{W[f'round{r}'][i] / max(W[f'round{r}'].sum(), 1e-300):.6e}" for r in rounds),
             *(f"{LW[f'round{r}'][i]:.4f}" for r in rounds)] for i, s in enumerate(S)]
    write_csv(RES / "f3_weights.csv", head, rows)

    imap = int(np.argmax(w))
    mv = model_values(R, t)
    labels = {"area": [f"area {d} ({k})" for d, k in zip(t["area_date"][1:], t["area_kind"][1:])],
              "dh2": [f"dh 2012-2021 band {e:.0f}-{e + 50:.0f} m" for e in t["dh2_edges"]],
              "radar": [f"radar bedrock profile {p}" for p in range(1, 7)],
              "stake": [f"stake {n} (lower bound)" for n in t["stake_name"][t["stake_with2026"]]],
              "speed": ["ITS_LIVE median 2014-2021 (upper bound)"]}
    trows = []
    for k in TERMS:
        m, o, sg = mv[k]
        for j in range(m.shape[1]):
            q = wq(m[:, j], w, [0.05, 0.5, 0.95])
            p0 = np.percentile(m[:, j], [5, 50, 95])
            inside_ = {"stake": q[1] >= o[j], "speed": q[1] <= o[j]}.get(k, q[0] <= o[j] <= q[2])
            trows.append([k, labels[k][j], f"{o[j]:.4f}", f"{sg[j]:.4f}", f"{m[imap, j]:.4f}",
                          *(f"{v:.4f}" for v in q), f"{(q[1] - o[j]) / sg[j]:+.2f}", inside_,
                          *(f"{v:.4f}" for v in p0)])
    write_csv(RES / "f3_terms.csv", ["term", "value", "observed", "sigma", "model_MAP", "model_p05", "model_p50",
                                     "model_p95", "median_minus_observed_in_sigma", "inside_p05_p95_or_bound_met_by_median",
                                     "samples_model_p05", "samples_model_p50", "samples_model_p95"], trows)

    sm = []
    add = lambda q, v: sm.append((q, v))
    add("samples used (both segments ended), rounds", f"{len(S)}, {rounds}")
    for v in variants:
        add(f"effective sample size, {v} (pooled rounds, Kish)", f"{ess(W[v]):.1f}")
    for r in rounds:
        add(f"effective sample size, rounds 1 to {r} pooled", f"{ess(W[f'round{r}']):.1f}")
    add("largest normalised weight (main)", f"{w.max() / w.sum():.4f}")
    add("gate: effective sample size >= 128 (main)", f"{ess(w):.1f} {'pass' if ess(w) >= ESS_MIN else 'FAIL'}")
    for name, key, obs in (("2026-09-27 main body", "A2026_1.0", float(t["area_obs"][3])),
                           ("2025-09-07 main body", "A2025_1.0", float(t["area_obs"][2])),
                           ("2024-08-23 all polygons", "A2024_1.0", float(t["area_obs"][1]))):
        x = np.array([float(r[key]) for r in R])
        q = wq(x, w, [0.05, 0.5, 0.95])
        add(f"gate: area {name} km2, posterior 5 / 50 / 95 %; observed; inside",
            f"{q[0]:.4f} {q[1]:.4f} {q[2]:.4f}; {obs:.4f}; {'pass' if q[0] <= obs <= q[2] else 'FAIL'}")
    for v in variants:
        add(f"u 5 / 50 / 95 %, {v}", " ".join(f"{x:+.3f}" for x in wq(u, W[v], [0.05, 0.5, 0.95])))
    for v in variants:
        add(f"f 5 / 50 / 95 %, {v}", " ".join(f"{x:.3f}" for x in wq(f, W[v], [0.05, 0.5, 0.95])))
    for name, x, box in (("u", u, U_BOX), ("ln f", np.log(f), LNF_BOX)):
        hist = np.histogram(x, 20, box, weights=w)[0] / w.sum()
        add(f"{name}: weighted share in the 20 bins of the prior range (low to high)", " ".join(f"{v:.3f}" for v in hist))
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
    for i in range(2):
        for j in range(i + 1, len(names)):
            add(f"weighted correlation {names[i]} - {names[j]}", f"{C[i, j] / D[i] / D[j]:+.3f}")
    bal = np.array([int(s["balance_draw"]) for s in S])
    wb = np.bincount(bal, weights=w, minlength=1024) / w.sum()
    add("balance draws: distinct among the samples; effective number of balance draws (1 / sum of squared shares)",
        f"{len(set(bal))}; {1 / (wb ** 2).sum():.1f}")
    for k in ("Fm", "c_prec", "dT", "dg", "b"):
        i = PAR.index(k)
        add(f"{k} 5 / 50 / 95 %: F1 posterior | F3 weighted", " ".join(f"{v:.5g}" for v in np.percentile(P[:, i], [5, 50, 95]))
            + " | " + " ".join(f"{v:.5g}" for v in wq(P[:, i], w, [0.05, 0.5, 0.95])))
    for v in ("speed_strict",):
        for key, lab in (("A2026_1.0", "area 2026-09-27 main body"), ("A2024_1.0", "area 2024-08-23 all polygons")):
            x = np.array([float(r[key]) for r in R])
            add(f"{lab} km2 5 / 50 / 95 %, {v}", " ".join(f"{q:.4f}" for q in wq(x, W[v], [0.05, 0.5, 0.95])))
    sp = mv["speed"][0][:, 0]
    add("speed (ITS_LIVE median 2014-2021, _12 run) m a-1: weighted 5 / 50 / 95 %; bound",
        " ".join(f"{x:.3f}" for x in wq(sp, w, [0.05, 0.5, 0.95])) + f"; {float(t['speed_bound']):.1f}")
    add("speed: weighted share above the bound; share of all samples above; weighted mean excess above the bound m a-1",
        f"{w[sp > 1].sum() / w.sum():.3f}; {(sp > 1).mean():.3f}; {w @ np.maximum(sp - 1, 0) / w.sum():.3f}")
    ws = W["speed_strict"]
    add("speed, variant speed_strict (bound with sigma 0.5 m a-1 only): weighted 5 / 50 / 95 %; share above the bound",
        " ".join(f"{x:.3f}" for x in wq(sp, ws, [0.05, 0.5, 0.95])) + f"; {ws[sp > 1].sum() / ws.sum():.3f}")
    for k in TERMS:
        add(f"term {k}: log-likelihood at MAP, weighted mean; model-error SD E[s] weighted mean",
            f"{LL[k][imap]:.2f}, {w @ LL[k] / w.sum():.2f}; {w @ cond[k] / w.sum():.4g}")
    add("MAP sample: name, balance draw, u, f", f"{S[imap]['name']} {S[imap]['balance_draw']} {u[imap]:+.4f} {f[imap]:.4f}")
    for key in ("V_start21", "V_2026", "V_2021_from12", "A_2021_from12", "basin_melt_km3"):
        x = np.array([float(r[key]) for r in R])
        add(f"{key}: weighted 5 / 50 / 95 %", " ".join(f"{v:.5f}" for v in wq(x, w, [0.05, 0.5, 0.95])))
    write_csv(RES / "f3_summary.csv", ["quantity", "value"], sm)
    for q, v in sm:
        print(f"{q}: {v}")

    p = w / w.sum()
    pos = (np.random.default_rng(SEED).random() + np.arange(N_POST)) / N_POST
    idx = np.minimum(np.searchsorted(np.cumsum(p), pos), len(p) - 1)
    copies = np.bincount(idx, minlength=len(p))
    e = ess(w)
    extra = ("A2024_1.0", "A2025_1.0", "A2026_1.0", "V_start21", "V_2026", "speed", "v_p90_2021")
    prow = []
    for j, i in enumerate(idx):
        s, fi = S[i], float(S[i]["f"])
        prow.append([j, s["name"], s["round"], s["balance_draw"], *(s[k] for k in PAR), f"{RATIO * float(s['Fm']):.7g}",
                     s["u"], s["f"], f"{A0 * fi:.6g}", f"{TAU0 * fi ** (-1 / 3):.7g}", f"{p[i]:.6e}", copies[i],
                     f"{e:.1f}", *(R[i][k] for k in extra)])
    write_csv(RES / "posterior_f3_128.csv", ["draw", "sample", "round", "balance_draw", *PAR, "r_ice", "u", "f",
                                             "arrhenius", "tau_ref", "weight", "copies", "ess", *extra], prow)
    print(f"posterior_f3_128.csv: {len(set(idx))} unique samples of {len(S)}, effective sample size {e:.1f}")


WARM_CHECK = ((-1.0, 0.05), (-1.0, 20.0), (0.5, 3.0), (2.0, 0.05), (2.0, 20.0))


def warm_list():
    """stages/f3_warm.txt: the F1 MAP balance at five corners and the middle of the (u, f) box, both segments, warm
    (the run lists' setting) and cold (3000 iterations): is the warm start good enough over the whole box?"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("f2", IGM / "checks/f2_spinup.py")
    f2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f2)
    p = f2.f1_map()
    par = [p["Fm"], p["c_prec"], p["dT"], p["dg"], p["b"], p["dT_pre"]]
    lines = ["# NAME EXPERIMENT overrides (run_stage.sh); fix round F3: warm start against cold over the (u, f) box, "
             "F1 MAP balance (fit_f3.py warm)"]
    for k, (u, f) in enumerate(WARM_CHECK):
        common = overrides(par, u, f)
        for seg in ("12", "21"):
            lines += [segment(f"warm{k}", seg, common, True), segment(f"cold{k}", seg, common, False)]
    (IGM / "stages/f3_warm.txt").write_text("\n".join(lines) + "\n")
    print(f"stages/f3_warm.txt: {len(lines) - 1} runs")


def warm_compare():
    """results/f3_warm.csv: per (u, f) and value, warm and cold model values and their difference."""
    R = {r["name"]: r for r in read_csv(RES / "f3_runs_f3_warm.csv")}
    keys = (["A2024_1.0", "A2025_1.0", "A2026_1.0", "V_2026", "v_med_2021", "v_p90_2021", "speed_2021_its"],
            ["speed", "V_2021_from12", "A_2021_from12", "v_med_2012", "v_p90_2012"])
    rows = []
    for k, (u, f) in enumerate(WARM_CHECK):
        for seg, ks in (("21", keys[0]), ("12", keys[1])):
            w, c = R[f"warm{k}_{seg}"], R[f"cold{k}_{seg}"]
            for q in ks:
                a, b = float(w[q]), float(c[q])
                rows.append([u, f, seg, q, f"{a:.5g}", f"{b:.5g}", f"{a - b:+.4g}",
                             f"{(a - b) / b:+.4f}" if b else "", w["wall_s"], c["wall_s"]])
            if seg == "12":
                n = len([q for q in w if q.startswith("dh2_")])
                d = np.array([float(w[f"dh2_{i}"]) - float(c[f"dh2_{i}"]) for i in range(n)])
                rows.append([u, f, seg, "dh2 bands: root mean square of warm - cold, m", "", "",
                             f"{np.sqrt((d ** 2).mean()):.4f}", "", w["wall_s"], c["wall_s"]])
    write_csv(RES / "f3_warm.csv", ["u", "f", "segment", "value", "warm", "cold", "warm_minus_cold", "relative",
                                    "wall_s_warm", "wall_s_cold"], rows)
    for r in rows:
        print(*r, sep=" | ")


def save_proposal(rnd):
    """Before samples(rnd > 1): fix the round's proposal in results/f3_proposal_r{rnd}.npz (weigh reads it back, so
    later re-weighing cannot change the density the samples were drawn from)."""
    m, c, z, beta = proposal(rnd)
    np.savez(RES / f"f3_proposal_r{rnd}.npz", m=m, c=c, z=z, beta=beta)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "samples":
        R = int(sys.argv[2])
        if R > 1:
            save_proposal(R)
        samples(R)
    elif cmd == "reduce":
        reduce(sys.argv[2])
    elif cmd == "weigh":
        weigh()
    elif cmd == "warm":
        warm_list()
    elif cmd == "warmcompare":
        warm_compare()
