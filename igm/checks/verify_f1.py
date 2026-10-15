"""Independent check of F1 (DESIGN.md, fix round): the gate re-derived from the inputs and the round files with code
that does not import calib_smb.py (only the physical constants of smb.py: T_SNOW, T_RAIN, ICE, Z_OTL).

  obs      the 11 volume changes from raw/glamos/volumechange.csv (C14-10, the six pre-1991 pairs and the five of
           1991-2024; dV km3, sigma_dV = |dV| sigma / |B_geod|; survey days as given, 9999 days by design: 1907 ->
           1907-10-01, 1929 -> 1929-08-13, 2018 -> 2018-09-01) and the 34 glacier-wide annual balances of the
           observation periods 1992-2025 (raw/glamos/massbalance_observation.csv)
  ess      per round file (results/STEM/round{r}.npz): proposal log density recomputed from theta and the saved
           mean and covariance (uniform in round 1), weight = likelihood / proposal inside the prior box, Kish
           effective sample size; the kept draws of results/STEM_draws.csv checked against the weights
  forward  own day-by-day balance model on every cell of the lattice that is in any year's mask (stride 1), for the
           unique kept draws of calib_f1 and its MAP draw: per draw the 11 volume changes (each hydrological year
           on its own mask, km3 of ice at 850 kg m-3) and the 34 glacier-wide observation-period balances (mask of
           the year holding the end); geometry rebuilt from igm/inputs/*.tif (nearest outline from 1992; SGI1931
           up to 1929-08-13, then the signed-distance blend SGI1931 -> SGI1973 -> mask_1994, linear in time;
           surface usurf_1994 minus the GLAMOS mean elevation change to 1991-09-18, then linear 1991 -> 2012
           (Grab et al. surface, where it has a value) -> 2021, held after)
  report   predictive band of every volume change as the mixture of the kept draws' model values with Gaussian
           noise sigma_dV (term d is binding: no model-error SD), solved for its 5 and 95 % points; inside or
           outside, distance outside (km3), model median minus observed in sigma; mean Ba 1992-2025; MAP lowest
           year and 2001; each against results/calib_f1_bands.csv and calib_f1_summary.csv
Outputs: results/verify_f1_forward.npz (forward), results/verify_f1.csv (report: quantity, own value, reported value).
usage (s02, agl env, from the basodino folder):
  python igm/checks/verify_f1.py forward NPROC ;  python igm/checks/verify_f1.py report
"""
import csv
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
ROOT, INP, RES = IGM.parent, IGM / "inputs", IGM / "results"
sys.path.insert(0, str(IGM))
from smb import ICE, T_RAIN, T_SNOW, Z_OTL  # noqa: E402  (physical constants only)

RHO, CELL_KM2 = 0.85, 1e-4
FREE = {"calib_f1": [0, 2, 3, 4, 5, 6], "calib_f1_stationary": [0, 2, 4, 5]}   # theta columns that are free
PRIOR = {0: (0.0, 6e-3), 2: (0.5, 3.0), 3: (-3.0, 3.0), 4: (-3e-3, 3e-3), 5: (-2e-4, 1e-3), 6: (-3.0, 3.0)}
DAY = {"19079999": date(1907, 10, 1), "19299999": date(1929, 8, 13), "20189999": date(2018, 9, 1)}
PAIRS = [("19079999", "19299999"), ("19299999", "19491007"), ("19491007", "19610919"), ("19610919", "19710903"),
         ("19710903", "19850910"), ("19850910", "19910918"), ("19910918", "20020916"), ("20020916", "20080829"),
         ("20080829", "20130821"), ("20130821", "20189999"), ("20189999", "20240823")]
LATE = (("mask_1994", date(1991, 9, 18)), ("sgi2010", date(2010, 1, 1)), ("sgi2016", date(2018, 9, 1)),
        ("mask_2021", date(2021, 9, 30)), ("tlm2024", date(2024, 8, 23)))
EARLY = (("sgi1931", date(1929, 8, 13)), ("sgi1973", date(1973, 9, 1)), ("mask_1994", date(1991, 9, 18)))
S91, S12, S21 = date(1991, 9, 18), date(2012, 9, 1), date(2021, 9, 30)
Y0, Y1, PRE_END = 1908, 2025, 1991


def day(s):
    s = s.replace("-", "")
    if s in DAY:
        return DAY[s]
    y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
    return date(y, m, 28 if (m, d) == (2, 29) else d)


def glamos_rows(name):
    lines = open(ROOT / "raw/glamos" / name, encoding="utf-8").read().splitlines()
    return [r for r in csv.reader(lines) if len(r) > 2 and "C14-10" in (r[0].strip(), r[1].strip())]


def obs():
    """(volume changes: list of (d0, d1, dV, sigma)), (glacier-wide Ba periods 1992-2025: list of (d0, d1, Ba))."""
    lines = open(ROOT / "raw/glamos/volumechange.csv", encoding="utf-8").read().splitlines()
    head = next(r for r in csv.reader(lines) if r[:1] == ["SGI-ID"])
    vc = {(r[3], r[4]): dict(zip(head, r)) for r in csv.reader(lines) if r[:1] == ["C14-10"]}
    dv = []
    for a, b in PAIRS:
        r = vc[a, b]
        v, bg, sg = float(r["dV"]), float(r["Bgeod"]), float(r["sigma"])
        dv.append((day(a), day(b), v, abs(v) * sg / abs(bg)))
    lines = open(ROOT / "raw/glamos/massbalance_observation.csv", encoding="utf-8").read().splitlines()
    k = next(i for i, s in enumerate(lines) if s.startswith("glacier name"))
    head = [h.strip() for h in next(csv.reader([lines[k + 1]]))]
    ba = []
    for r in csv.reader(lines[k + 2:]):
        if len(r) > 1 and r[1].strip() == "C14-10":
            d = dict(zip(head, (x.strip() for x in r)))
            ba.append((day(d["date_start"]), day(d["date_end"]), float(d["Ba"]) / 1e3))
    assert len(ba) == 34 and ba[0][1].year == 1992 and ba[-1][1].year == 2025, (len(ba), ba[0], ba[-1])
    return dv, ba


def hydro(d):
    return d.year + (d.month >= 10)


# --- importance weights -------------------------------------------------------------------------------------------
def ess(stem):
    free = FREE[stem]
    lo = np.array([PRIOR[i][0] for i in free])
    hi = np.array([PRIOR[i][1] for i in free])
    out, r = [], 1
    while (RES / stem / f"round{r}.npz").exists():
        R = np.load(RES / stem / f"round{r}.npz")
        th, L = R["theta"], R["loglik"]
        free = [int(i) for i in R["free"]]               # free columns as saved (the stationary set has dT free)
        lo = np.array([PRIOR[i][0] for i in free])
        hi = np.array([PRIOR[i][1] for i in free])
        f = th[:, free]
        box = ((f >= lo) & (f <= hi)).all(1)
        if "mean" not in R.files or R["mean"].size == 0:          # round 1: uniform over the box
            lq = np.zeros(len(f))
        else:
            d = f - R["mean"]
            lq = -0.5 * np.einsum("ij,ij->i", d @ np.linalg.inv(R["cov"]), d)
        lp = np.where(box, L.sum(1), -np.inf)
        assert np.isneginf(L[~box]).all() and np.isfinite(L[box]).all(), r
        lt = lp - lq
        w = np.exp(lt - lt[box].max())
        w /= w.sum()
        dq = (lq - (R["logq"] if "logq" in R.files else 0.0))[box]
        out.append(dict(round=r, n=len(f), inside=box.mean(), ess=1 / (w ** 2).sum(), wmax=w.max(),
                        logq_spread=float(dq.max() - dq.min()), w=w, theta=th, lp=lp))
        r += 1
    return out


# --- geometry, forcing --------------------------------------------------------------------------------------------
def grid(name):
    import rasterio
    with rasterio.open(INP / f"{name}.tif") as r:
        z = r.read(1).astype(np.float64)
        if r.nodata is not None:
            z[z == r.nodata] = np.nan
        return z, r.transform


def z2012(transform, shape):
    """Grab et al. 2012 surface pasted onto the lattice by its own transform (cell edges aligned)."""
    import rasterio
    with rasterio.open(ROOT / "derived/thickness/C14-10_SwissALTI3D_r2019.tif") as r:
        z = r.read(1).astype(np.float64)
        z[z == r.nodata] = np.nan
        assert r.res == (10.0, 10.0)
        c0, r0 = round((r.bounds.left - transform.c) / 10), round((transform.f - r.bounds.top) / 10)
    out = np.full(shape, np.nan)
    rs, cs = slice(max(r0, 0), min(r0 + z.shape[0], shape[0])), slice(max(c0, 0), min(c0 + z.shape[1], shape[1]))
    out[rs, cs] = z[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0]
    return out


def geometry():
    """Per hydrological year Y0..Y1: mask (year, cell) and surface (year, cell) on the union cells; ipot, D there."""
    from scipy.ndimage import distance_transform_edt as edt
    import rasterio
    m = {n: grid(n)[0] == 1 for n, _ in LATE + EARLY}
    z91, tr = grid("usurf_1994")
    z21 = grid("usurf_2021")[0]
    z12 = z2012(tr, z91.shape)
    sd = {n: edt(~m[n]) - edt(m[n]) for n, _ in EARLY}
    lines = open(ROOT / "raw/glamos/volumechange.csv", encoding="utf-8").read().splitlines()
    off = {r[3]: -float(r[10]) for r in csv.reader(lines) if r[:1] == ["C14-10"] and r[4] == "19910918"}
    t_off = [day(a).toordinal() for a, _ in PAIRS[:6]] + [S91.toordinal()]
    z_off = [off[a] for a, _ in PAIRS[:6]] + [0.0]
    years = np.arange(Y0, Y1 + 1)
    masks, surf, names = [], [], []
    for y in years:
        t = date(int(y), 4, 1)
        if y > PRE_END:
            n = min(LATE, key=lambda o: abs((t - o[1]).days))[0]
            masks.append(m[n])
            names.append(n)
        elif t <= EARLY[0][1]:
            masks.append(m[EARLY[0][0]])
            names.append(EARLY[0][0])
        else:
            k = next(i for i in range(1, 3) if t <= EARLY[i][1])
            a = (t - EARLY[k - 1][1]).days / (EARLY[k][1] - EARLY[k - 1][1]).days
            masks.append((1 - a) * sd[EARLY[k - 1][0]] + a * sd[EARLY[k][0]] < 0)
            names.append(f"{EARLY[k - 1][0]}-{EARLY[k][0]} {a:.3f}")
        if y <= PRE_END:
            surf.append(z91 + np.interp(t.toordinal(), t_off, z_off))
        else:
            fr = lambda a, b: min(max((t - a).days / (b - a).days, 0.0), 1.0)
            two = z91 + (z21 - z91) * fr(S91, S21)
            three = np.where(t < S12, z91 + (z12 - z91) * fr(S91, S12), z12 + (z21 - z12) * fr(S12, S21))
            surf.append(np.where(np.isfinite(z12), three, two))
    union = np.logical_or.reduce(masks)
    r, c = np.nonzero(union)
    with rasterio.open(INP / "ipot.tif") as f:
        ip = f.read()[:, r, c].astype(np.float32)
    D = grid("D")[0][r, c]
    M = np.array([mk[r, c] for mk in masks])
    Z = np.array([s[r, c] for s in surf])
    assert np.isfinite(Z).all() and np.isfinite(D).all() and np.isfinite(ip).all()
    return years, M, Z, ip, D.astype(np.float32), names


def forcing(years):
    rows = [r for r in csv.DictReader(open(INP / "weather_otl_daily.csv")) if Y0 <= int(r["hydro_year"]) <= Y1]
    assert len(rows) == 365 * len(years)
    dates = [date.fromisoformat(r["date"]) for r in rows]
    T = np.array([float(r["T_degC"]) for r in rows]).reshape(len(years), 365)
    P = np.array([float(r["P_mm"]) for r in rows]).reshape(len(years), 365) / 1e3
    fit = dict(csv.reader(open(RES / "redist_fit.csv")))
    g12 = np.array(fit["g_m Oct..Sep, K per 100 m"].split(), float) / 100
    gm = g12[[(d.month - 10) % 12 for d in dates[:365]]]
    index = {d: i for i, d in enumerate(dates)}
    return T, P, gm, index


# --- forward model ------------------------------------------------------------------------------------------------
G = {}


def init():
    if not G:
        dv, ba = obs()
        years, M, Z, ip, D, names = geometry()
        T, P, gm, index = forcing(years)
        G.update(dv=dv, ba=ba, years=years, M=M, Z=Z, ip=ip, D=D, T=T, P=P, gm=gm, index=index, names=names)
    return G


def idx(d):
    """Day index of date d in the 365-day calendar (29 February read as 28 February; 1 October after the last
    year = end)."""
    g = init()
    if d == date(Y1, 10, 1):
        return 365 * len(g["years"])
    return g["index"][date(d.year, d.month, 28) if (d.month, d.day) == (2, 29) else d]


def forward(th):
    """th (n, 7): Fm, r_ice, c_prec, dT, dg, b, dT_pre -> (dv (n, 11) km3 ice, ba (n, 34) m w.e.)."""
    g = init()
    n, ny = len(th), len(g["years"])
    Fm, ri, cp, dT, dg, b, dTp = (th[:, j:j + 1].astype(np.float32) for j in range(7))
    rs = np.float32(0.75) * ri
    CD = cp * g["D"][None]
    # daily sums: over the year's own mask (km3 ice), and the means over the masks of this year and the next two
    # (an observation period ending after 1 October starts two hydrological years before the one holding its end)
    own = np.zeros((n, ny * 365))
    mean = np.zeros((3, n, ny * 365))
    for i in range(ny):
        dz = (g["Z"][i] - Z_OTL).astype(np.float32)[None]
        nxt = [g["M"][min(i + k, ny - 1)] for k in range(3)]
        W = np.stack([g["M"][i] * CELL_KM2 * 1e-3 / RHO] + [m / m.sum() for m in nxt], 1).astype(np.float32)
        off = dT + (dTp if g["years"][i] <= PRE_END else 0)
        S = np.zeros((n, dz.shape[1]), np.float32)
        out = np.empty((365, n, 4), np.float32)
        for k in range(365):
            t = np.float32(g["T"][i, k]) + (np.float32(g["gm"][k]) + dg) * dz + off
            p = np.float32(g["P"][i, k]) * (1 + b * dz)
            snow = np.where(t <= T_SNOW, p, np.where(t >= T_RAIN, 0, p * (T_RAIN - t) / (T_RAIN - T_SNOW))) * CD
            S += snow
            melt = np.maximum(t, 0) * (Fm + np.where(S == 0, ri, rs) * g["ip"][k][None]) * ICE
            S = np.maximum(S - melt, 0)
            out[k] = (snow - melt) @ W
        own[:, 365 * i:365 * (i + 1)] = out[:, :, 0].T
        for k in range(3):
            mean[k, :, 365 * i:365 * (i + 1)] = out[:, :, k + 1].T
    dv = np.stack([own[:, idx(a):idx(b)].sum(1) for a, b, _, _ in g["dv"]], 1)
    ba = []
    for a, b, _ in g["ba"]:
        i0, i1 = idx(a), idx(b)
        ref = (i1 - 1) // 365                                  # year holding the end: its mask for every day
        assert ref - i0 // 365 <= 2
        ba.append(sum(mean[ref - j][:, max(i0, 365 * j):min(i1, 365 * (j + 1))].sum(1)
                      for j in range(i0 // 365, ref + 1)))
    return dv, np.stack(ba, 1)


def run_forward(nproc):
    rounds = ess("calib_f1")
    last = rounds[-1]
    rows = list(csv.DictReader(open(RES / "calib_f1_draws.csv")))
    keep = np.array([int(r["sobol_index"]) for r in rows])
    assert all(int(r["round"]) == last["round"] for r in rows)
    imap = int(np.nanargmax(last["lp"]))
    th = last["theta"]
    uniq = np.unique(np.append(keep, imap))
    chunks = np.array_split(uniq, nproc)
    init()
    with Pool(nproc) as pool:
        res = pool.map(forward, [th[c] for c in chunks])
    dv = np.concatenate([r[0] for r in res])
    ba = np.concatenate([r[1] for r in res])
    np.savez_compressed(RES / "verify_f1_forward.npz", sobol_index=uniq, keep=keep, imap=imap, theta=th[uniq], dv=dv,
                        ba=ba, names=np.array(G["names"]), area_km2=G["M"].sum(1) * CELL_KM2)
    print(f"{len(uniq)} draws: results/verify_f1_forward.npz")


# --- report -------------------------------------------------------------------------------------------------------
def band(m, s, q):
    """q-point of the mixture of N(m_i, s^2), i over the draws."""
    from scipy.optimize import brentq
    from scipy.stats import norm
    F = lambda x: norm.cdf((x - m) / s).mean() - q
    return brentq(F, m.min() - 10 * s, m.max() + 10 * s, xtol=1e-9)


def report():
    dvo, bao = obs()
    rows = []
    add = lambda q, own, rep="": rows.append((q, own, rep))
    bands = [r for r in csv.DictReader(open(RES / "calib_f1_bands.csv")) if r["kind"] == "dv"]
    summ = {r["quantity"]: r["value"] for r in csv.DictReader(open(RES / "calib_f1_summary.csv"))}
    for (a, b, v, s), r in zip(dvo, bands):
        add(f"observed volume change {a} {b} km3; sigma", f"{v:.4f}; {s:.4f}",
            f"{r['date0']} {r['date1']}: {r['observed']}; {r['sigma_obs']}")
    for stem in ("calib_f1", "calib_f1_stationary"):
        for R in ess(stem):
            add(f"{stem} round {R['round']}: effective sample size; largest weight; share in the box; "
                "spread of (own - saved) log proposal density inside the box",
                f"{R['ess']:.1f}; {R['wmax']:.4g}; {R['inside']:.3f}; {R['logq_spread']:.2e}",
                summ.get(f"round {R['round']} effective sample size", "") if stem == "calib_f1" else "")
    # kept draws against the weights
    last = ess("calib_f1")[-1]
    F = np.load(RES / "verify_f1_forward.npz")
    keep, th, uniq = F["keep"], last["theta"], F["sobol_index"]
    w = last["w"]
    for j, nm in ((0, "Fm"), (3, "dT"), (6, "dT_pre")):
        o = np.argsort(th[:, j])
        c = np.cumsum(w[o])
        wq = np.interp([0.05, 0.5, 0.95], c, th[o, j])
        add(f"{nm} 5 / 50 / 95 %: weighted (own) | kept draws", " ".join(f"{x:.5g}" for x in wq) + " | "
            + " ".join(f"{x:.5g}" for x in np.percentile(th[keep, j], [5, 50, 95])))
    pos = {s: i for i, s in enumerate(uniq)}
    kv = np.array([pos[s] for s in keep])
    dv, ba = F["dv"][kv], F["ba"][kv]
    dmap, bmap = F["dv"][pos[int(F["imap"])]], F["ba"][pos[int(F["imap"])]]
    n_in, miss = 0, []
    for j, ((a, b, v, s), r) in enumerate(zip(dvo, bands)):
        m = dv[:, j]
        p5, p95 = band(m, s, 0.05), band(m, s, 0.95)
        ins = p5 <= v <= p95
        n_in += ins
        dist = max(p5 - v, v - p95, 0.0)
        z = (np.median(m) - v) / s
        if not ins:
            miss.append(f"{a.year}-{b.year} {dist:.4f} ({z:+.1f})")
        add(f"volume change {a} {b}: model 5 / 50 / 95 %; predictive 5 / 95 %; inside; distance km3; median minus "
            "observed in sigma; MAP",
            " ".join(f"{x:+.4f}" for x in np.percentile(m, [5, 50, 95])) + f"; {p5:+.4f} {p95:+.4f}; {int(ins)}; "
            f"{dist:.4f}; {z:+.2f}; {dmap[j]:+.4f}",
            f"{r['model_p5']} {r['model_p50']} {r['model_p95']}; {r['predictive_p5']} {r['predictive_p95']}; "
            f"{r['inside_90']}; ; ; {r['map']}")
    add("volume changes inside the 90 % predictive band, of 11; misses (km3 outside, sigma)",
        f"{n_in}; " + ", ".join(miss), summ["gate: volume changes inside the 90 % predictive band, of 11"])
    add("volume change 1907-1991, sum of the kept-draw medians; observed km3",
        f"{np.median(dv[:, :6], 0).sum():+.4f}; {sum(x[2] for x in dvo[:6]):+.4f}")
    ob = np.mean([x[2] for x in bao])
    mb = ba.mean(1)
    key = [k for k in summ if k.startswith("glacier-wide annual balance, mean of 34")][0]
    add("mean glacier-wide Ba 1992-2025 m w.e.: observed; MAP; kept 5 / 50 / 95 %",
        f"{ob:+.3f}; {bmap.mean():+.3f}; " + " ".join(f"{x:+.3f}" for x in np.percentile(mb, [5, 50, 95])), summ[key])
    yrs = [x[1].year for x in bao]
    lo = int(np.argmin(bmap))
    add("MAP lowest year, value; 2001", f"{yrs[lo]} {bmap[lo]:+.3f}; {bmap[yrs.index(2001)]:+.3f}",
        summ["gate: MAP draw, year of its lowest glacier-wide annual balance, value m w.e."] + "; "
        + summ["gate: MAP draw, glacier-wide annual balance 2001 m w.e."])
    geo = {r["hydro_year"]: r["area_km2"] for r in csv.DictReader(open(RES / "calib_f1_geometry.csv"))}
    for y in (1908, 1950, 1961, 1971, 1985, 1991, 1992, 2010, 2025):
        add(f"mask area hydrological year {y} km2 ({F['names'][y - Y0]})", f"{F['area_km2'][y - Y0]:.4f}", geo[str(y)])
    with open(RES / "verify_f1.csv", "w", newline="") as f:
        csv.writer(f).writerows([("quantity", "own", "reported"), *rows])
    for r in rows:
        print(*r, sep=" | ")


if __name__ == "__main__":
    if sys.argv[1] == "forward":
        run_forward(int(sys.argv[2]))
    elif sys.argv[1] == "report":
        report()
    elif sys.argv[1] == "time":                 # timing of a few draws
        import time
        R = ess("calib_f1")[-1]
        t0 = time.time()
        dv, ba = forward(R["theta"][np.argsort(-np.nan_to_num(R["lp"], nan=-np.inf))[:int(sys.argv[2])]])
        print(f"{time.time() - t0:.0f} s for {len(dv)} draws", dv[0], ba[0].mean())
