"""Independent check of step 2 (DESIGN.md): the gate numbers re-derived from the calibration outputs, the target
definitions read back from the prepared inputs, and one diagnostic of the winter-balance definition of term (a).

1. Gate, from igm/results/calib/round*.npz, calib_draws.csv and calib_bands.csv: weights recomputed from the
   stored log-likelihood terms, Kish effective sample size per round; the kept draws traced to the last round by Sobol
   index; r_ice / Fm of the kept draws (fixed at calib_smb.RATIO);
   share of each target set inside its 90 % predictive band (bands.csv columns); MAP draw: year of its lowest
   glacier-wide annual balance, its 2001 value; share of kept draws within 2 % of each edge of the last box and of the
   prior (free parameters).
2. Definitions, from igm/inputs/calib_stride1.npz and raw/glamos: each target's dates against the GLAMOS columns
   (date_start, date_end_winter, date_end; stake date0/date1; geodetic dates); sigma of terms c and d against the
   release values; values shown only (fixed dates, glacier-wide) absent from every term matrix; the five geodetic
   periods contiguous and not overlapping.
3. Diagnostic (numpy float64, smb.step, stride 2): the MAP draw with its melt factor Fm times 1, 1.5, 2, 2.5, 3 (r_ice
   = RATIO Fm, other parameters at the MAP): glacier-wide mean annual balance of the observation periods, the five
   volume changes, and the band winter balance two ways, (i) the balance summed from date_start to date_end_winter
   (term a of the first calibration) and (ii) the snow store at date_end_winter (snow accumulated since 1 October
   minus its melt; term a as fitted since 2026-10-07), each as mean and root mean square
   of model minus observed, and the mean residual of the lowest and highest band.
4. MAP draw rerun in numpy float64 on every cell (stride 1), an own loop over smb.step that does not use
   calib_smb.simulate: every target value against the bands.csv MAP column (band Ba and stakes as fitted, minus
   their period / year mean), largest |difference| per kind; the gate values (lowest year, 2001, mean glacier-wide Ba
   1992-2025, the five volume changes against their predictive band) from the float64 values. The winter balances
   (calib_smb.STORE) read three ways, the store at the start of the day before, of, and after date_end_winter.
5. Terms: the MAP draw's four marginal log-likelihoods recomputed in float64 on the stride-2 lattice (written here:
   residual of each term matrix, variance sigma_obs^2 + (s scale)^2, mean of the Gaussian likelihood over the 64
   log-spaced s nodes) against the stored loglik row of the last round.
6. Snow-store definition, MAP draw, stride 1, every lattice cell and observation year: the store at date_end_winter
   against the balance summed from the last day the store was empty (the first lasting snow) to date_end_winter;
   the days in that span that melt at the ice rate (store empty, no accumulation, T > 0); hydrological year of
   date_end_winter against that of date_end; the glacier-mean model store on date_start where date_start is after
   1 October (snow counted that fell before the autumn survey); where date_start is before 1 October, the snowfall
   between date_start and 1 October (not in the store, the reset drops it) and the store left on 30 September;
   every year 1992-2025: the glacier-mean snow store left on 30 September and the share of the year's ice mask with
   snow left (the model's accumulation-area ratio), against the GLAMOS AAR of the observation periods.
7. Station transfer against DESIGN.md (step 2, third attempt): constants and priors; g_m negative; smb.transfer against
   the formula written out here on every column and day (MAP); signs of dg and b and the pivot at 367 m; the MAP's
   warming and precipitation factor at 2500 and 3100 m.
8. Second attempt (2par, dT; dg = 0, b = B_PREC): its MAP through the new transfer, float64 stride 1, against its own
   calib_bands map column and the lines of its earlier calib_step2_check (identical digits = reproduced).
9. Every kept draw in float64 on every cell (values(), member axis, no calib_smb.simulate): model 5/50/95 % against
   calib_bands.csv; per term the share inside the 90 % predictive band from the exact predictive CDF at the
   observation (mixture over kept draws of Gaussians with sigma_obs and the draw's s), against inside_90; the five
   volume changes; glacier-wide mean Ba 1992-2025; the 16 largest-weight draws of the last round rerun at stride 2 and
   the effective sample size with their float64 log-likelihoods.
10. Cost of each attempt's level, per term from its calib_bands / calib_draws files (this run and the earlier
   attempts grad, grad_dT): MAP minus observed as fitted, mean and root mean square; share inside the 90 % predictive
   band; median of the kept model-error SD.
A run made with "bind" (term d without model-error SD, stored as bind in the round files) gets term d with s = 0 in
sections 5 and 9.
The forcing is smb.transfer with the run's dg and b (theta of the earlier attempts, 4 columns: dg = 0, b = B_PREC).
Output: igm/results/calib_step2_check{_TAG}.csv (quantity, value).
usage (s02, agl env, from basodino/):  python igm/checks/calib_step2_check.py [TAG]
       no TAG: the final run (results/calib/, calib_draws.csv, calib_bands.csv);
       TAG 2par, dT, ...: an earlier run (results/calib_TAG/, calib_draws_TAG.csv, calib_bands_TAG.csv).
"""
import csv
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IGM))
import calib_smb as c  # noqa: E402
from smb import step, transfer  # noqa: E402

RES = IGM / "results"
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
SFX = f"_{TAG}" if TAG else ""
RUN = RES / f"calib{SFX}"
out = []


def put(k, v):
    out.append((k, v))
    print(k, "\t", v, flush=True)


# 1. gate
rounds = sorted(int(p.stem[5:]) for p in RUN.glob("round*.npz"))
for r in rounds:
    z = np.load(RUN / f"round{r}.npz")
    lt = z["loglik"].sum(1)
    w = np.exp(lt - lt.max())
    w /= w.sum()
    put(f"round {r}: effective sample size recomputed; largest |stored - recomputed weight|",
        f"{1 / (w ** 2).sum():.1f} {np.abs(w - z['weight']).max():.2e}")
box, free = z["box"], z["free"]
BIND = bool(z["bind"]) if "bind" in z else False
put("term d binding (no model-error SD)", BIND)
theta = np.zeros((len(z["theta"]), len(c.NAMES)))
theta[:, 5] = c.B_PREC
theta[:, :z["theta"].shape[1]] = z["theta"]
draws = list(csv.DictReader(open(RES / f"calib_draws{SFX}.csv")))
j = np.array([int(d["sobol_index"]) for d in draws])
fm = np.array([float(d["Fm_m_ice_per_day_degC"]) for d in draws])
put(f"kept draws: round, largest |Fm in draws.csv - round-{rounds[-1]} theta| relative",
    f"{set(d['round'] for d in draws)} {np.abs(fm / theta[j, 0] - 1).max():.1e}")
imap = int(np.argmax(w))
put(f"MAP: Sobol index (largest weight of round {rounds[-1]}), draws.csv is_map index",
    f"{imap} {[int(d['sobol_index']) for d in draws if d['is_map'] == '1']}")
kept = theta[j]
put("kept draws: r_ice / Fm smallest, largest", f"{(kept[:, 1] / kept[:, 0]).min():.6g} {(kept[:, 1] / kept[:, 0]).max():.6g}")
for b_i, i in enumerate(free):
    nm = c.NAMES[i]
    for lab, (lo, hi) in (("last box", box[b_i]), ("prior", c.PRIOR[i])):
        e = 0.02 * (hi - lo)
        put(f"{nm}: share of kept draws within 2 % of the {lab} lower / upper edge ({lo:.4g}, {hi:.4g})",
            f"{(kept[:, i] < lo + e).mean():.3f} {(kept[:, i] > hi - e).mean():.3f}")
    h, _ = np.histogram(kept[:, i], bins=10, range=tuple(box[b_i]))
    put(f"{nm}: kept draws in 10 equal bins of the last box", " ".join(map(str, h)))

B = list(csv.DictReader(open(RES / f"calib_bands{SFX}.csv")))
for kinds in (["bw_band"], ["ba_band"], ["bw_band", "ba_band"], ["stake"], ["dv"]):
    rows = [b for b in B if b["kind"] in kinds and b["fitted_as"] != "no model cells"]
    ins = [float(b["predictive_p5"]) <= float(b["observed"]) <= float(b["predictive_p95"]) for b in rows]
    put(f"inside the 90 % predictive band, recomputed: {' + '.join(kinds)}", f"{sum(ins)} of {len(ins)} = {np.mean(ins):.3f}")
gl = {int(b["label"]): float(b["map"]) for b in B if b["kind"] == "ba_glacier"}
low = min(gl, key=gl.get)
put("MAP glacier-wide annual balance: lowest year and value; 2001", f"{low} {gl[low]:+.3f}; {gl[2001]:+.3f}")
put("glacier-wide annual balance 1992-2025, mean observed and MAP model",
    f"{np.mean([float(b['observed']) for b in B if b['kind'] == 'ba_glacier']):+.3f} {np.mean(list(gl.values())):+.3f}")

# 2. definitions
X = dict(np.load(IGM / "inputs/calib_stride1.npz"))
kind, d0, d1 = X["kind"], X["d0"], X["d1"]
obs = {(r["date_start"], r["date_end"]): r for r in c.glamos("massbalance_observation_elevationbins.csv", 1)}
bad = 0
for k, a, b in zip(kind, d0, d1):
    if k in ("bw_band", "ba_band"):
        ok = any(c.iso(r["date_start"]).isoformat() == a and
                 c.iso(r["date_end_winter" if k == "bw_band" else "date_end"]).isoformat() == b for r in obs.values())
        bad += not ok
put("band targets whose dates are not a GLAMOS (date_start, date_end_winter / date_end) pair", bad)
st = [s.split() for s in open(IGM.parent / "raw/glamos/point_2025/annual/basodino_annual.dat") if s.strip() and s[0] != "#"]
ic = X["idx_c"]
put("stakes: dates differ from date0/date1; largest |sigma - mb_error| m w.e.",
    f"{sum(c.iso(s[1]).isoformat() != d0[i] or c.iso(s[3]).isoformat() != d1[i] for s, i in zip(st, ic))} "
    f"{np.abs(X['sig_c'] - np.array([float(s[17]) / 1e3 for s in st])).max():.1e}")
idd = X["idx_d"]
put("volume changes: periods, contiguous", f"{[(d0[i], d1[i]) for i in idd]} "
    f"{all(d1[a] == d0[b] for a, b in zip(idd[:-1], idd[1:]))}")
put("volume changes: sigma_obs km3 (GLAMOS sigma x years x mean area / 0.85)", " ".join(f"{v:.4f}" for v in X["sig_d"]))
shown = np.isin(kind, ["bw_glacier", "ba_glacier", "bw_fixdate", "ba_fixdate"])
put("values shown only: largest |entry| in the term matrices a, b, c, d",
    " ".join(f"{np.abs(X[f'M{k}'][shown & (kind != 'ba_glacier')]).max():.0f}" for k in c.TERMS)
    + f"; glacier-wide Ba used only in b as the period mean: {np.abs(X['Ma'][kind == 'ba_glacier']).max() + np.abs(X['Mc'][kind == 'ba_glacier']).max() + np.abs(X['Md'][kind == 'ba_glacier']).max():.0f}")

# 3. diagnostic of the winter-balance definition
X = dict(np.load(IGM / "inputs/calib_stride2.npz"))
mp = theta[imap]
mult = np.array([1, 1.5, 2, 2.5, 3])
Fm = (mp[0] * mult)[:, None]
r = c.RATIO * Fm
D = mp[2] * X["D"][None, :]
nT, pc, G = len(X["obs"]), X["pieces"], X["groups"].astype(np.float64)
val = np.zeros((len(mult), nT))
snow = np.zeros((len(mult), nT))
end = {}
for p in pc:                                        # last piece of each target: year and end day
    if p[0] not in end or p[2] > end[p[0]][0]:
        end[p[0]] = (p[2], p[4], p[1])
for i in range(len(X["years"])):
    T, P = transfer(X["t_otl"][i][:, None], X["p_otl"][i][:, None], X["gm"][:, None], X["z"][i][None, :] - c.Z_OTL,
                    mp[4], mp[5])
    T = T + mp[3]
    S, C, Sh = np.zeros((len(mult), T.shape[1])), [np.zeros((len(mult), T.shape[1]))], [None]
    for k in range(365):
        S, b = step(S, T[k], P[k], X["ipot"][k], D, Fm, r, 0.75 * r)
        C.append(C[-1] + b)
        Sh.append(S)
    for t, g, y, a, e in pc[pc[:, 2] == i]:
        val[:, t] += (C[e] - C[a]) @ G[g]
    for t, (y, e, g) in end.items():
        if y == i:
            snow[:, t] = Sh[e] @ G[g]
ob = X["obs"]
bw = np.nonzero((kind_s := X["kind"]) == "bw_band")[0]
bw = bw[X["ok"][bw]]
band = np.array([int(lab.split()[1]) for lab in X["label"][bw]])
gl = kind_s == "ba_glacier"
dv = X["idx_d"]
for m in range(len(mult)):
    put(f"Fm x {mult[m]} ({Fm[m, 0]:.5f}): glacier-wide mean Ba, observed {ob[gl].mean():+.3f}", f"{val[m, gl].mean():+.3f}")
    put(f"Fm x {mult[m]}: volume changes km3 (observed {' '.join(f'{v:+.4f}' for v in ob[dv])})",
        " ".join(f"{v:+.4f}" for v in val[m, dv]))
    for nm, v in (("(i) summed from date_start", val[m, bw]), ("(ii) snow store at date_end_winter", snow[m, bw])):
        res = v - ob[bw]
        put(f"Fm x {mult[m]}: band Bw {nm}, model minus observed: mean, rms, mean at {band.min()} m and {band.max()} m",
            f"{res.mean():+.3f} {np.sqrt((res ** 2).mean()):.3f} {res[band == band.min()].mean():+.3f} "
            f"{res[band == band.max()].mean():+.3f}")


# 4-6. the MAP draw in numpy float64, own loop over smb.step (no calib_smb.simulate)
def rerun(X, th, define=False):
    """Every target value of the parameter row th (Fm, r_ice, c_prec, dT, dg, b) in float64; the STORE targets also read on
    the day before and after (shift, rows -1, 0, +1). With define: the snow-store definition statistics of section 6."""
    kind, pc, G = X["kind"], X["pieces"], X["groups"].astype(np.float64)
    store = np.isin(kind, c.STORE)
    first, last = {}, {}
    for j, g, y, a, e in pc:
        if j not in last or y > last[j][0]:
            last[j] = (y, g, e)
        if j not in first or y < first[j][0]:
            first[j] = (y, g, a)
    Fm, r = np.full((1, 1), th[0]), np.full((1, 1), th[1])
    D = th[2] * X["D"][None, :]
    nc = int(X["ncell"])
    val, shift = np.zeros(len(X["obs"])), np.zeros((3, len(X["obs"])))
    st = dict(diff=0.0, ice=0, empty=[], after=[], before=[], firn=[], sep30=[], snowshare=[])
    for i in range(len(X["years"])):
        T, P = transfer(X["t_otl"][i][:, None], X["p_otl"][i][:, None], X["gm"][:, None],
                        X["z"][i][None, :] - c.Z_OTL, th[4], th[5])
        T = T + th[3]
        S = np.zeros((1, T.shape[1]))
        Sd, C = np.zeros((366, T.shape[1])), np.zeros((366, T.shape[1]))      # store and summed balance at day starts
        for k in range(365):
            S, b = step(S, T[k], P[k], X["ipot"][k], D, Fm, r, 0.75 * r)
            Sd[k + 1], C[k + 1] = S[0], C[k] + b[0]
        for j, g, y, a, e in pc[pc[:, 2] == i]:
            if not store[j]:
                val[j] += (C[e] - C[a]) @ G[g]
        for j, (y, g, e) in last.items():
            if store[j] and y == i:
                val[j] = Sd[e] @ G[g]
                for m, d in enumerate((-1, 0, 1)):
                    shift[m, j] = Sd[min(max(e + d, 0), 365)] @ G[g]
        if not define:
            continue
        for j in np.nonzero(kind == "ba_fixdate")[0]:                  # 1 October to 1 October: year i, end day 365
            if last[j][0] == i:
                g = G[last[j][1]]
                st["sep30"].append(Sd[365] @ g)
                st["snowshare"].append((Sd[365] > 0) @ g)                 # g: the year's mask / its cells
        acc = np.where(T <= 0.5, P, np.where(T >= 2.5, 0.0, P * (2.5 - T) / 2.0)) * D     # accumulation (day, col)
        ice = (Sd[:365] == 0) & (acc == 0) & (T > 0)                                      # melt at the ice rate
        for e in sorted({last[j][2] for j in last if kind[j] == "bw_glacier" and last[j][0] == i}):
            empty = Sd[:e + 1, :nc] == 0
            L = e - np.argmax(empty[::-1], axis=0)                       # last day start with an empty store
            span = C[e, :nc] - C[L, np.arange(nc)]
            st["diff"] = max(st["diff"], np.abs(Sd[e, :nc] - span).max())
            days = np.arange(e)[:, None]
            st["ice"] += int((ice[:e, :nc] & (days >= L[None, :])).sum())
            st["empty"].append((Sd[e, :nc] == 0).mean())
        for j in np.nonzero(kind == "bw_glacier")[0]:
            y0, g0, a0 = first[j]
            if y0 == i and last[j][0] == i:
                st["after"].append(Sd[a0] @ G[last[j][1]])             # store on date_start, counted in Bw
            elif y0 == i:                                              # snowfall date_start - 30 September, dropped
                st["before"].append(acc[a0:].sum(0) @ G[g0])          # by the reset (upper bound: before its melt)
                st["firn"].append(Sd[365] @ G[g0])                     # whole store on 30 September, dropped
    return val, shift, st, last


X1 = dict(np.load(IGM / "inputs/calib_stride1.npz"))
mp = theta[imap]
put(f"MAP draw rerun ({', '.join(c.NAMES)}; dg K m-1, b m-1)", " ".join(f"{v:.5g}" for v in mp))
v1, sh1, st1, last1 = rerun(X1, mp, define=True)
k1 = X1["kind"]
vf, of = v1.copy(), X1["obs"].astype(float).copy()
for k in ("b", "c"):
    vf[X1[f"idx_{k}"]] = v1 @ X1[f"M{k}"]
    of[X1[f"idx_{k}"]] = X1["obs"] @ X1[f"M{k}"]
mapcol = np.array([float(b["map"]) for b in B])
obscol = np.array([float(b["observed"]) for b in B])
put("bands.csv observed column against the GLAMOS values as fitted (stride-1 inputs): largest |difference|",
    f"{np.abs(obscol - of).max():.1e}")
for k in dict.fromkeys(k1):
    m = k1 == k
    put(f"MAP float64 stride 1 against bands.csv map column, {k} ({m.sum()} values): largest |difference|, largest |value|",
        f"{np.abs(vf[m] - mapcol[m]).max():.2e} {np.abs(vf[m]).max():.3f}")
for k in c.STORE:
    m = k1 == k
    put(f"{k}: store read at the start of the day before / of / after date_end_winter, largest |difference| to bands.csv map",
        " ".join(f"{np.abs(sh1[i, m] - mapcol[m]).max():.2e}" for i in range(3)))
gidx = np.nonzero(k1 == "ba_glacier")[0]
gmap = {int(X1["label"][j]): v1[j] for j in gidx}
low = min(gmap, key=gmap.get)
put("MAP float64: lowest glacier-wide Ba year and value; 2001; mean 1992-2025 (observed mean)",
    f"{low} {gmap[low]:+.3f}; {gmap[2001]:+.3f}; {np.mean(list(gmap.values())):+.3f} ({X1['obs'][gidx].mean():+.3f})")
dvi = X1["idx_d"]
p5 = np.array([float(B[j]["predictive_p5"]) for j in dvi])
p95 = np.array([float(B[j]["predictive_p95"]) for j in dvi])
put("volume changes km3: observed / predictive 5 % / 95 % / MAP float64",
    " | ".join(f"{X1['obs'][j]:+.4f} {a:+.4f} {b:+.4f} {v1[j]:+.4f}" for j, a, b in zip(dvi, p5, p95)))
put("volume changes inside the predictive band (recomputed from the observed and band columns)",
    f"{int(((X1['obs'][dvi] >= p5) & (X1['obs'][dvi] <= p95)).sum())} of 5")

def own_ll(X, V):
    """Marginal log-likelihood of terms a-d for value rows V (member, target), float64, written here (section 5)."""
    nodes = np.exp(np.log(0.05) + (np.arange(64) + 0.5) / 64 * np.log(1 / 0.05))
    out = np.zeros((len(V), len(c.TERMS)))
    for n, k in enumerate(c.TERMS):
        nd = np.zeros(1) if BIND and k == "d" else nodes
        res = (V - X["obs"]) @ X[f"M{k}"]                                          # (member, value)
        var = X[f"sig_{k}"][:, None] ** 2 + (nd[None, :] * X[f"scl_{k}"][:, None]) ** 2
        ll = (-0.5 * (res[:, :, None] ** 2 / var[None] + np.log(2 * np.pi * var[None]))).sum(1)
        mx = ll.max(1, keepdims=True)
        out[:, n] = (mx + np.log(np.mean(np.exp(ll - mx), axis=1, keepdims=True)))[:, 0]
    return out


X2 = dict(np.load(IGM / "inputs/calib_stride2.npz"))
v2 = rerun(X2, mp)[0]
own = own_ll(X2, v2[None])[0]
Lmap = z["loglik"][imap]
put("MAP terms a, b, c, d: log-likelihood float64 stride 2 recomputed / stored (round "
    f"{rounds[-1]}) / difference", " | ".join(f"{a:.3f} {b:.3f} {a - b:+.3f}" for a, b in zip(own, Lmap)))

put("store definition (MAP, stride 1, all lattice cells, 34 years): largest |store at date_end_winter - balance "
    "summed from the last empty-store day|", f"{st1['diff']:.1e}")
put("store definition: days melting at the ice rate inside those spans", st1["ice"])
put("store definition: share of lattice cells with an empty store at date_end_winter, per year min / median / max",
    " ".join(f"{v:.3f}" for v in np.percentile(st1["empty"], [0, 50, 100])))
bwg = np.nonzero(k1 == "bw_glacier")[0]
put("date_end_winter: months; years whose date_end lies in a later hydrological year (date_end after 30 September), "
    "years whose store year differs from the hydrological year of date_end_winter",
    f"{sorted(set(int(d[5:7]) for d in X1['d1'][bwg]))}; {sum(last1[j][0] != last1[j - 1][0] for j in bwg)} "
    f"{sum(int(X1['d1'][j - 1][5:7]) >= 10 for j in bwg)}; "
    f"{sum(last1[j][0] != int(X1['d1'][j][:4]) - int(X1['years'][0]) for j in bwg)}")
put("date_start on or after 1 October: years, glacier-mean model store on date_start mean / max m w.e. (in Bw)",
    f"{len(st1['after'])} " + (f"{np.mean(st1['after']):.3f} {np.max(st1['after']):.3f}" if st1["after"] else "-"))
put("date_start before 1 October: years, glacier-mean snowfall date_start to 30 September mean / max m w.e. (not in "
    "the store), glacier-mean store left on 30 September mean / max (reset)",
    f"{len(st1['before'])} " + (f"{np.mean(st1['before']):.3f} {np.max(st1['before']):.3f}; "
                                f"{np.mean(st1['firn']):.3f} {np.max(st1['firn']):.3f}" if st1["before"] else "-"))

aar = [float(r["AAR"]) / 100 for r in c.glamos("massbalance_observation.csv", 1)]
put(f"snow left on 30 September, MAP, every year ({len(st1['sep30'])}): glacier-mean store m w.e. mean / min / max; "
    "share of the ice mask with snow left mean / min / max (GLAMOS AAR of the observation periods mean / min / max)",
    f"{np.mean(st1['sep30']):.3f} {np.min(st1['sep30']):.3f} {np.max(st1['sep30']):.3f}; "
    f"{np.mean(st1['snowshare']):.3f} {np.min(st1['snowshare']):.3f} {np.max(st1['snowshare']):.3f} "
    f"({np.mean(aar):.3f} {np.min(aar):.3f} {np.max(aar):.3f})")

# 7. the station transfer against DESIGN.md (step-2 settings, third attempt), written out here
put("transfer: Z_OTL m, B_PREC % per 100 m; prior dg K per km, prior b % per 100 m (DESIGN.md: 367; 2; -3..+3; -2..+10)",
    f"{c.Z_OTL:g}; {c.B_PREC * 1e4:g}; {c.PRIOR[4, 0] * 1e3:g}..{c.PRIOR[4, 1] * 1e3:+g}; "
    f"{c.PRIOR[5, 0] * 1e4:g}..{c.PRIOR[5, 1] * 1e4:+g}")
gmv = np.unique(X1["gm"])
put("g_m (Robiei - Locarno, per day of the year): distinct values, range K per 100 m, all negative",
    f"{len(gmv)} {gmv.min() * 100:.3f}..{gmv.max() * 100:.3f} {bool((gmv < 0).all())}")
dT_, dP_ = 0.0, 0.0
for i in range(len(X1["years"])):
    dz = X1["z"][i] - 367.0
    T, P = transfer(X1["t_otl"][i][:, None], X1["p_otl"][i][:, None], X1["gm"][:, None], dz[None, :], mp[4], mp[5])
    th_T = X1["t_otl"][i][:, None] + X1["gm"][:, None] * dz[None, :] + mp[4] * dz[None, :]
    th_P = X1["p_otl"][i][:, None] + X1["p_otl"][i][:, None] * mp[5] * dz[None, :]
    dT_, dP_ = max(dT_, np.abs(T - th_T).max()), max(dP_, np.abs(P - th_P).max())
put("MAP: smb.transfer against T_OTL + g_m dz + dg dz and P_OTL + P_OTL b dz, dz = z - 367 m, every column and day "
    "1991-2025: largest |difference| degC, m w.e. d-1", f"{dT_:.1e} {dP_:.1e}")
zc = X1["z"][-1]
up = transfer(0.0, 1.0, 0.0, zc - c.Z_OTL, 1e-3, 1e-4)
put("signs: dg = +1 K per km, b = +1 % per 100 m, OTL T = 0, P = 1: T and P at the columns min / max (z "
    f"{zc.min():.0f}-{zc.max():.0f} m); at z = 367 m", f"{up[0].min():+.3f} {up[0].max():+.3f}, {up[1].min():.4f} "
    f"{up[1].max():.4f}; {transfer(0.0, 1.0, 0.0, 0.0, 1e-3, 1e-4)}")
for h in (2500, 3100):
    put(f"MAP at {h} m: warming from dg (K), precipitation factor c_prec (1 + b dz) and (1 + b dz) alone; "
        "decision 5 (1 + 2 % per 100 m dz)",
        f"{mp[4] * (h - c.Z_OTL):+.3f} {mp[2] * (1 + mp[5] * (h - c.Z_OTL)):.3f} {1 + mp[5] * (h - c.Z_OTL):.3f}; "
        f"{1 + c.B_PREC * (h - c.Z_OTL):.3f}")


# 8. dg = 0, b = 2 % per 100 m: the second attempt (2par, dT) through the new transfer, against its own outputs
def as_fitted(X, v):
    """Band Ba minus its period mean, stakes minus their year mean (terms b, c); v (..., target)."""
    vf = v.copy()
    for k in ("b", "c"):
        vf[..., X[f"idx_{k}"]] = v @ X[f"M{k}"]
    return vf


for tg in ("2par", "dT"):
    if TAG == tg or not (RES / f"calib_{tg}").is_dir():
        continue
    zt = np.load(sorted((RES / f"calib_{tg}").glob("round*.npz"))[-1])
    th = np.zeros(len(c.NAMES))
    th[5] = c.B_PREC
    th[:zt["theta"].shape[1]] = zt["theta"][np.argmax(zt["loglik"].sum(1))]
    vt = rerun(X1, th)[0]
    mt = np.array([float(b["map"]) for b in csv.DictReader(open(RES / f"calib_bands_{tg}.csv"))])
    old = dict(csv.reader(open(RES / f"calib_step2_check_{tg}.csv")))
    same, vf_t = [], as_fitted(X1, vt)
    for k in dict.fromkeys(k1):
        m = k1 == k
        key = (f"MAP float64 stride 1 against bands.csv map column, {k} ({m.sum()} values): largest |difference|, "
               "largest |value|")
        same.append(f"{np.abs(vf_t[m] - mt[m]).max():.2e} {np.abs(vf_t[m]).max():.3f}" == old.get(key))
    gt = {int(X1["label"][q]): vt[q] for q in gidx}
    lt_ = min(gt, key=gt.get)
    gate = f"{lt_} {gt[lt_]:+.3f}; {gt[2001]:+.3f}; {np.mean(list(gt.values())):+.3f} ({X1['obs'][gidx].mean():+.3f})"
    put(f"{tg} (second attempt) MAP {' '.join(f'{v:.5g}' for v in th)} through smb.transfer (dg 0, b B_PREC): "
        f"largest |difference| to calib_bands_{tg}.csv map; per-kind lines identical to calib_step2_check_{tg}.csv; "
        "lowest year, 2001, mean Ba (identical to the earlier check)",
        f"{np.abs(vf_t - mt)[X1['ok']].max():.2e}; {sum(same)} of {len(same)}; {gate} "
        f"({gate == old.get('MAP float64: lowest glacier-wide Ba year and value; 2001; mean 1992-2025 (observed mean)')})")


# 9. every kept draw in float64, member axis (no calib_smb.simulate): model bands, exact predictive coverage
def values(X, TH, nb=512):
    """Every target value of the parameter rows TH (member, 6) in float64: smb.transfer and smb.step per day, the
    summed balance and the store read at the piece boundaries; columns in blocks of nb (cache), summed over blocks."""
    kind, pc, G = X["kind"], X["pieces"], X["groups"].astype(np.float64)
    store = np.isin(kind, c.STORE)
    last = {}
    for q, g, y, a, e in pc:
        if q not in last or y > last[q][0]:
            last[q] = (y, g, e)
    Fm, r, dT, dg, bp = (TH[:, k:k + 1] for k in (0, 1, 3, 4, 5))
    D = TH[:, 2:3] * X["D"][None, :]
    val = np.zeros((len(TH), len(X["obs"])))
    for i in range(len(X["years"])):
        at = {}
        for q, g, y, a, e in pc[pc[:, 2] == i]:
            if not store[q]:
                at.setdefault(e, []).append((q, g, 1.0))
                at.setdefault(a, []).append((q, g, -1.0))
        for q, (y, g, e) in last.items():
            if store[q] and y == i:
                at.setdefault(e, []).append((q, g, None))
        for c0 in range(0, G.shape[1], nb):
            cs = slice(c0, c0 + nb)
            dz, Db, Ib, Gb = (X["z"][i, cs] - c.Z_OTL)[None, :], D[:, cs], X["ipot"][:, cs], G[:, cs]
            S, C = np.zeros((len(TH), dz.size)), np.zeros((len(TH), dz.size))
            for k in range(366):
                for q, g, s in at.get(k, ()):
                    val[:, q] += (S @ Gb[g]) if s is None else s * (C @ Gb[g])
                if k < 365:
                    T, P = transfer(X["t_otl"][i, k], X["p_otl"][i, k], X["gm"][k], dz, dg, bp)
                    S, b = step(S, T + dT, P, Ib[k], Db, Fm, r, 0.75 * r)
                    C += b
    return val


from scipy.special import ndtr  # noqa: E402

jk = np.array([int(d["sobol_index"]) for d in draws])
uq, inv = np.unique(jk, return_inverse=True)
vk = values(X1, theta[uq])[inv]
put("check of values(): MAP row against the section-4 rerun, largest |difference|",
    f"{np.abs(values(X1, mp[None])[0] - v1).max():.1e}")
vkf = as_fitted(X1, vk)
okm = X1["ok"]
mq = np.percentile(vkf, [5, 50, 95], axis=0)
bq = np.array([[float(b[f"model_p{p}"]) for b in B] for p in (5, 50, 95)])
put(f"kept draws ({len(uq)} unique) float64 stride 1: model 5 / 50 / 95 % against calib_bands.csv, largest |difference|",
    " ".join(f"{np.abs(mq[i] - bq[i])[okm].max():.2e}" for i in range(3)))
sd = np.zeros((len(vk), len(X1["obs"])))
for k in c.TERMS:
    idx = X1[f"idx_{k}"]
    s = np.array([float(d[f"s_{k}"]) for d in draws])
    sd[:, idx] = np.sqrt(X1[f"sig_{k}"][None, :] ** 2 + (s[:, None] * X1[f"scl_{k}"][None, :]) ** 2)
F = ndtr((of[None, :] - vkf) / np.where(sd > 0, sd, 1)).mean(0)        # predictive CDF at the observation (mixture)
ins = (F >= 0.05) & (F <= 0.95)
inb = np.array([b["inside_90"] == "1" for b in B])
for k, kinds in (("a", ["bw_band"]), ("b", ["ba_band"]), ("a+b", ["bw_band", "ba_band"]), ("c", ["stake"]),
                 ("d", ["dv"])):
    m = np.isin(k1, kinds) & okm
    put(f"term {k}: inside the 90 % predictive band, exact mixture CDF of the kept draws (calib_bands.csv inside_90; "
        "values where the two differ)", f"{ins[m].sum()} of {m.sum()} = {ins[m].mean():.3f} ({inb[m].sum()}; "
        f"{(ins[m] != inb[m]).sum()})")
rng = np.random.default_rng(7)
pdv = np.concatenate([vkf[:, dvi] + rng.standard_normal((len(vk), 5)) * sd[:, dvi] for _ in range(64)])
pq = np.percentile(pdv, [5, 95], axis=0)
put("volume changes km3: observed, predictive CDF at the observation, predictive 5 % / 95 % (64 x 1024 noise draws), "
    "model median", " | ".join(f"{of[q]:+.4f} {F[q]:.3f} {a:+.4f} {b:+.4f} {mq[1, q]:+.4f}"
                               for q, a, b in zip(dvi, pq[0], pq[1])))
gk = vk[:, gidx].mean(1)
put("glacier-wide annual balance 1992-2025, mean of the 34 years m w.e.: observed; kept draws float64 5 / 50 / 95 %",
    f"{X1['obs'][gidx].mean():+.3f}; " + " ".join(f"{v:+.3f}" for v in np.percentile(gk, [5, 50, 95])))
top = np.argsort(z["loglik"].sum(1))[-16:]
L16 = own_ll(X2, values(X2, theta[top]))
lt = z["loglik"].sum(1).copy()
lt[top] = L16.sum(1)
w = np.exp(lt - lt.max())
w /= w.sum()
put(f"the 16 largest-weight draws of round {rounds[-1]}: total log-likelihood float64 stride 2 minus stored, largest "
    "|difference|; effective sample size with these replaced",
    f"{np.abs(L16.sum(1) - z['loglik'][top].sum(1)).max():.3f}; {1 / (w ** 2).sum():.1f}")

# 10. cost of the level per term, from each attempt's own output files
for tg in dict.fromkeys([TAG, "grad", "grad_dT"]):
    sf = f"_{tg}" if tg else ""
    if not (RES / f"calib_bands{sf}.csv").exists():
        continue
    Bt = list(csv.DictReader(open(RES / f"calib_bands{sf}.csv")))
    Dt = list(csv.DictReader(open(RES / f"calib_draws{sf}.csv")))
    for k, kinds in (("a", "bw_band"), ("b", "ba_band"), ("c", "stake"), ("d", "dv")):
        rows = [b for b in Bt if b["kind"] == kinds and b["fitted_as"] != "no model cells"]
        r = np.array([float(b["map"]) - float(b["observed"]) for b in rows])
        put(f"{tg or 'this run'}: term {k} ({kinds}, {len(rows)} values): MAP minus observed as fitted mean / rms; share "
            "inside the 90 % predictive band; kept model-error SD median",
            f"{r.mean():+.4f} {np.sqrt((r ** 2).mean()):.4f}; {np.mean([b['inside_90'] == '1' for b in rows]):.3f}; "
            f"{np.median([float(d[f's_{k}']) for d in Dt]):.3g}")

with open(RES / f"calib_step2_check{SFX}.csv", "w", newline="") as f:
    csv.writer(f).writerows([("quantity", "value"), *out])
