"""Balance calibration of decision 7 (igm/DESIGN.md, step 2, third and fourth attempt): sampling-importance-resampling of
(Fm, c_prec, dg, b), with r_ice fixed at the Aletsch reference ratio r_ice / Fm = 0.012 m2 W-1 (decision 7 fallback:
the first calibration of 2026-10-07 piled r_ice at its prior edge 0, results/calib_step2_check_3par.csv) and the two
station gradients of decision 5 free (author's answer 2026-10-07), as set in DESIGN.md, step-2 settings. Fourth
attempt (switch "bind", author's answer 2026-10-07 'Volume changes binding'): term d without a model-error SD, the
five volume changes enter with their GLAMOS sigma only; everything else as in the third attempt.
Fix round F1 (DESIGN.md 'Fix round', 2026-10-09): term d also holds the six GLAMOS volume changes 1907-1929,
1929-1949, 1949-1961, 1961-1971, 1971-1985, 1985-1991 (11 values; pairs and survey days of targets_step5.PAIRS, the
all-pairs file, dV in km3 only), so the run starts on 1907-10-01. Every value of term d: sigma_obs = |dV| sigma_B /
|B_geod| (GLAMOS's own area normalisation inverted, so the 1946 outline of 1929-1949 and 1949-1961 cancels; no B_geod
value is used). Geometry of the hydrological years up to 1991 (mid-year 1 April): ice mask SGI1931 up to 1929
(GLAMOS length change: the front of 1907 lies within 10 m of 1929, and GLAMOS uses the 1929 outline for 1907), then
the signed-distance blend, linear in time, of SGI1931 (1929-08-13), SGI1973 (1973-09-01) and mask_1994 (1991-09-18);
surface = usurf_1994 minus the GLAMOS mean elevation change from the survey to 1991-09-18 (all-pairs rows ending
19910918), linear in time between the survey days, the same on every cell. From hydrological year 1992 everything is
as before. Switch "pre": a temperature offset dT_pre added to every day of the hydrological years up to 1991, uniform
-3 to +3 degC (the fallback of F1 when one stationary set cannot fit both periods). Outputs carry the stem calib_f1
(with pre) or calib_f1_stationary (without); the geometry of every year in results/calib_f1_geometry.csv.

Model: smb.py on columns of the lattice of inputs.py plus one column per stake, forcing of weather.py with the transfer
smb.transfer (T = T_OTL + (g_m + dg) (z - 367 m), g_m per month from results/redist_fit.csv; P = c_prec P_OTL (1 + b
(z - 367 m))), I_pot of ipot.py (2021 surface), accumulation factor c_prec D (redist.py). Hydrological years 1991
to 2025, snow store reset on 1 October (decision 3); every value is the sum of the daily balance over the days d with
date0 <= d < date1 of its survey dates (29 February read as 28 February), except the winter balances of the
observation periods (band and glacier-wide): those are the modelled snow store at the start of date1 =
date_end_winter (snow since 1 October minus its melt, in the hydrological year holding date1), what the probings
measure.
Geometry per hydrological year Y, at its middle (1 April of Y): surface linear in time between the 1991 survey
(usurf_1994, 1991-09-18), the 2012 surface of Grab et al. (2021) (derived/thickness/C14-10_SwissALTI3D_r2019.tif,
taken as 2012-09-01; where it has no value, linear 1991-2021) and usurf_2021 (2021-09-30), held after 2021; ice
mask = the outline nearest in time of mask_1994 (1991-09-18), sgi2010 (2010-01-01), sgi2016 (2018-09-01), mask_2021
(2021-09-30), tlm2024 (2024-08-23). Lattice columns: the cells of the union of these masks on every STRIDE-th row and
column, each standing for STRIDE^2 cells. A value over a period uses the geometry of the hydrological year holding
its end (band and mask fixed over the period); a volume change uses each year's own mask.
Stakes: GLAMOS point release 2025 (raw/glamos/point_2025/annual/basodino_annual.dat, 344 values, LV03 -> LV95 with
pyproj), at the stake's own height z_pos, I_pot and D of the cell holding it.
Priors: uniform Fm 0-6e-3 m ice d-1 degC-1, c_prec 0.5-3, dg -3 to +3 K per km (added to the monthly Robiei-Locarno
gradients g_m), b -2 to +10 % per 100 m; r_ice = 0.012 Fm m ice m2 W-1 d-1 degC-1 (r_snow = 0.75 r_ice); per term a model-error SD s, log-uniform 0.05-1 m w.e. (m w.e. a-1 for the volume term), integrated out per
draw on 64 log-spaced nodes (the posterior of the free balance parameters is the same as with s among the Sobol
dimensions; s is drawn from its conditional posterior for the kept draws). With "bind" term d has the single node s = 0.
Likelihood, Gaussian, independent values, variance sigma_obs^2 + (s scale)^2:
  a  band winter balance of the observation period as the snow store at date_end_winter, 100 m bands 2500-3200 m
     (232 values; sigma_obs 0)
  b  band annual balance of the observation period minus the mean glacier-wide annual balance of the observation
     years of its geodetic period (232; sigma_obs 0); a year belongs to the period holding its end date, 2025 to the
     last one
  c  stake balance minus the mean of the stakes of its hydrological year (344; sigma_obs = mb_error)
  d  geodetic volume change in km3 (850 kg m-3), the six F1 periods 1907-1991 and 1991-09-18, 2002-09-16,
     2008-08-29, 2013-08-21, 2018 (taken as 2018-09-01), 2024-08-23 (11 values; sigma_obs = |dV| sigma_B / |B_geod|,
     scale |dV / B_geod| = GLAMOS years x mean area / 0.85 for s)
Shown, not fitted: glacier-wide winter and annual balance of the observation periods and of the fixed dates
(1 October to 30 April, to 30 September).
Temperature offset (fallback of DESIGN.md step 2, only if the run without it fails its gate; "python calib_smb.py
run 2 dT"): dT added to every day and cell, uniform -3 to +3 degC, a fifth free parameter.
Rounds: round 1 = 2^17 scrambled Sobol draws over the prior box; while the Kish effective sample size is below 1000,
one more round of 2^17 scrambled Sobol draws mapped through the normal quantile to a Gaussian with the mean and
covariance of the previous round's live draws (likelihood above 1e-6 of the largest, at least the 50 x (free
parameters) best; unweighted, so wider than the posterior; covariance plus (2 % of the prior range)^2 on the diagonal), or, once the previous round reached an
effective sample size of 20, its importance-weighted mean and covariance times 1.5^2; weight = likelihood / proposal
density inside the prior box, 0 outside (uniform prior), at most MAX_ROUNDS = 10 (F1: the shrinking box of the accepted step-2 run,
in git, stalled at an effective sample size of 4.5-21.5 after two rounds on the dT-dg ridge). 1024 draws kept by
systematic resampling of the last round. The kept draws (and the MAP draw, the largest posterior density) are rerun on every cell (STRIDE 1) for the predictive bands: model value, and model
value plus Gaussian noise with sigma_obs and s drawn from its conditional posterior per kept draw.
Inputs (s02): igm/inputs/{usurf_1994,usurf_2021,sgi1931,sgi1973,mask_1994,sgi2010,sgi2016,mask_2021,tlm2024,D,ipot}.tif,
igm/inputs/weather_otl_daily.csv, igm/results/redist_fit.csv, raw/glamos/{massbalance_observation,
massbalance_observation_elevationbins,massbalance_fixdate,volumechange}.csv (rows of C14-10),
raw/glamos/point_2025/annual/basodino_annual.dat, derived/thickness/C14-10_SwissALTI3D_r2019.tif.
Outputs (STEM = calib_f1 or calib_f1_stationary): igm/inputs/calib_f1_stride{S}.npz (prepare);
igm/results/STEM/round{r}.npz (Sobol draws, log-likelihood terms), igm/results/STEM_draws.csv (1024 kept draws),
igm/results/STEM_bands.csv (per value: observation, model and predictive 5/50/95 %, inside the 90 % band, MAP
value), igm/results/STEM_summary.csv (rounds, gate, posterior
percentiles and correlations of the free parameters, band annual balance minus the period mean per band, timing).
The summary also gives per term the MAP draw's misfit as fitted (mean and root mean square of model minus observed,
the volume changes also in units of their sigma) and the band annual-balance gradient (least-squares slope over the
band means of the years).
Earlier attempts: calib_*_3par (r_ice free), calib_*_2par (Fm, c_prec), calib_*_dT (Fm, c_prec, dT), calib_*_grad
(Fm, c_prec, dg, b), calib_*_grad_dT (Fm, c_prec, dT, dg, b), same columns; the round files in results/calib_<tag>/;
the accepted step-2 posterior (1991-2025 only) is results/calib_{draws,bands,summary}.csv (code before F1 in git).
usage:  s02 (agl env):    python igm/calib_smb.py prepare 2 ; python igm/calib_smb.py prepare 1
        s03 (env.sh, one GPU, nrun):  python calib_smb.py run 2 [dT] [bind] [pre]  (rounds on stride 2, bands on
                                      stride 1; dT = with the temperature offset; bind = term d without model-error
                                      SD; pre = with dT_pre; start=R: rounds before R read from
                                      their files; about 11 min per round)
"""
import csv
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from smb import B_PREC, Z_OTL

IGM = Path(__file__).resolve().parent
INP, RES = IGM / "inputs", IGM / "results"
PRIOR = np.array([[0.0, 6e-3], [0.0, 6e-5], [0.5, 3.0], [-3.0, 3.0], [-3e-3, 3e-3], [-2e-4, 1e-3], [-3.0, 3.0]])
NAMES = ("Fm", "r_ice", "c_prec", "dT", "dg", "b", "dT_pre")    # theta columns; dg K m-1, b m-1
PRE_END = 1991                         # dT_pre acts on the hydrological years up to this one
PRE_OUTLINES = (("sgi1931", date(1929, 8, 13)), ("sgi1973", date(1973, 9, 1)), ("mask_1994", date(1991, 9, 18)))
SHOW = {"dg": (1e3, "K per km"), "b": (1e4, "% per 100 m")}     # reporting units
RATIO = 0.012                          # r_ice / Fm, m2 W-1, fixed (Aletsch reference ratio)
STORE = ("bw_band", "bw_glacier")      # values read as the snow store at date1, not as a sum
S_GRID = np.exp(np.log(0.05) + (np.arange(64) + 0.5) / 64 * np.log(1 / 0.05))
TERMS = ("a", "b", "c", "d")
S_NODES = {k: S_GRID for k in TERMS}            # nodes of each term's model-error SD; run(bind=True) sets d to [0]
N_DRAWS, N_KEEP, ESS_MIN, MAX_ROUNDS, SEED = 2 ** 17, 1024, 1000, 10, 1991
ESS_FIT, INFLATE = 20, 1.5             # proposal from the weighted moments once a round reaches ESS_FIT
RHO = 0.85
GPU_BYTES = 4e9                 # chunk: about 12 float32 arrays of (member, column) on the GPU
BANDS = np.arange(2500, 3201, 100)
OUTLINES = (("mask_1994", date(1991, 9, 18)), ("sgi2010", date(2010, 1, 1)), ("sgi2016", date(2018, 9, 1)),
            ("mask_2021", date(2021, 9, 30)), ("tlm2024", date(2024, 8, 23)))
SURFACES = (date(1991, 9, 18), date(2012, 9, 1), date(2021, 9, 30))
GEODETIC = ("19910918", "20020916", "20080829", "20130821", "20189999", "20240823")


def decimal(d):
    return d.year + (d.toordinal() - date(d.year, 1, 1).toordinal()) / 365.25


def glamos(name, id_col):
    """Rows of C14-10 from a GLAMOS release csv (raw/glamos), as dicts keyed by the short column names (the line
    after the long names in the mass-balance files, the name line itself in the volume-change file)."""
    lines = open(IGM.parent / "raw/glamos" / name, encoding="utf-8").read().splitlines()
    k = next(i for i, s in enumerate(lines) if s.startswith(("glacier name", "SGI-ID")))
    k += lines[k].startswith("glacier name")
    head = [h.strip() for h in next(csv.reader([lines[k]]))]
    return [dict(zip(head, r)) for r in csv.reader(lines[k + 1:]) if len(r) > id_col and r[id_col] == "C14-10"]


def iso(s):
    s = s.replace("-", "")
    y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
    return date(2018, 9, 1) if s == "20189999" else date(y, m, 28 if (m, d) == (2, 29) else d)


def prepare(stride):
    """s02, agl env: columns, forcing, targets and term matrices -> igm/inputs/calib_f1_stride{stride}.npz; the
    geometry of every year -> igm/results/calib_f1_geometry.csv."""
    import rasterio
    from pyproj import Transformer
    from scipy.ndimage import distance_transform_edt
    from inputs import DER, DX, E0, N1, read, tif_grid
    from targets_step5 import PAIRS

    # targets: kind, label, date0, date1, observation, sigma_obs, scale, band, period / stake index
    T = []
    bins = glamos("massbalance_observation_elevationbins.csv", 1)
    obs = glamos("massbalance_observation.csv", 1)
    fix = glamos("massbalance_fixdate.csv", 1)
    assert len(bins) == 232 and len(obs) == len(fix) == 34
    geo = [(iso(a), iso(b)) for a, b in zip(GEODETIC[:-1], GEODETIC[1:])]

    def period(d1):
        return next((i for i, (a, b) in enumerate(geo) if a < d1 <= b), len(geo) - 1)

    for r in obs:
        d0, dw, d1 = iso(r["date_start"]), iso(r["date_end_winter"]), iso(r["date_end"])
        T.append(dict(kind="ba_glacier", label=d1.year, d0=d0, d1=d1, obs=float(r["Ba"]) / 1e3, period=period(d1)))
        T.append(dict(kind="bw_glacier", label=d1.year, d0=d0, d1=dw, obs=float(r["Bw"]) / 1e3, ref=d1))
    for r in bins:
        d0, dw, d1 = iso(r["date_start"]), iso(r["date_end_winter"]), iso(r["date_end"])
        b = int((float(r["h_min"]) - 2500) // 100)
        T.append(dict(kind="bw_band", label=f"{d1.year} {r['h_min']}", d0=d0, d1=dw, obs=float(r["Bw"]) / 1e3,
                      band=b, ref=d1))
        T.append(dict(kind="ba_band", label=f"{d1.year} {r['h_min']}", d0=d0, d1=d1, obs=float(r["Ba"]) / 1e3,
                      band=b, period=period(d1)))
    for r in fix:
        y = int(r["date_end"][:4])
        T.append(dict(kind="bw_fixdate", label=y, d0=date(y - 1, 10, 1), d1=date(y, 5, 1), obs=float(r["Bw"]) / 1e3))
        T.append(dict(kind="ba_fixdate", label=y, d0=date(y - 1, 10, 1), d1=date(y, 10, 1), obs=float(r["Ba"]) / 1e3))
    allp = {(r["date_start"], r["date_end"]): r for r in glamos("volumechange.csv", 0)}
    pre = [(a, b, date.fromisoformat(da), date.fromisoformat(db)) for a, b, da, db, _ in PAIRS]
    for a, b, da, db in pre + [(a, b, da, db) for (a, b), (da, db) in zip(zip(GEODETIC[:-1], GEODETIC[1:]), geo)]:
        v = allp[a, b]
        dv, bg = float(v["dV"]), float(v["Bgeod"])
        scale = abs(dv / bg)                 # km3 per m w.e. a-1 in GLAMOS's own normalisation (years x mean area / 0.85)
        T.append(dict(kind="dv", label=f"{da.isoformat()} {db.isoformat()}", d0=da, d1=db, obs=dv,
                      sigma=float(v["sigma"]) * scale, scale=scale))
    stakes = []
    with open(IGM.parent / "raw/glamos/point_2025/annual/basodino_annual.dat") as f:
        for s in f:
            if s.startswith("#") or not s.strip():
                continue
            p = s.split()
            stakes.append((p[0], iso(p[1]), iso(p[3]), float(p[7]), float(p[8]), float(p[9]), float(p[14]) / 1e3,
                           float(p[17]) / 1e3))
    assert len(stakes) == 344, len(stakes)
    E, N = Transformer.from_crs(21781, 2056, always_xy=True).transform(*np.array([s[3:5] for s in stakes]).T)
    for k, s in enumerate(stakes):
        T.append(dict(kind="stake", label=f"{s[2].year} {s[0]}", d0=s[1], d1=s[2], obs=s[6], sigma=s[7], stake=k))

    # forcing and calendar
    rows = list(csv.DictReader(open(INP / "weather_otl_daily.csv")))
    y0 = min(t["d0"] for t in T).year + (min(t["d0"] for t in T).month >= 10)
    last = date.fromordinal(max(t["d1"] for t in T).toordinal() - 1)
    y1 = last.year + (last.month >= 10)
    rows = [r for r in rows if y0 <= int(r["hydro_year"]) <= y1]
    years = np.arange(y0, y1 + 1)
    index = {r["date"]: i for i, r in enumerate(rows)}
    index[date(y1, 10, 1).isoformat()] = len(rows)
    for t in T:
        t["t0"], t["t1"] = index[t["d0"].isoformat()], index[t["d1"].isoformat()]
        assert t["t1"] > t["t0"], t
    t_otl = np.array([float(r["T_degC"]) for r in rows]).reshape(len(years), 365)
    p_otl = np.array([float(r["P_mm"]) for r in rows]).reshape(len(years), 365) / 1e3
    month = np.array([int(r["date"][5:7]) for r in rows[:365]])
    fit = dict(csv.reader(open(RES / "redist_fit.csv")))
    gm = np.array(fit["g_m Oct..Sep, K per 100 m"].split(), float)[(month - 10) % 12] / 100     # K m-1 per day

    # ice mask per year on the full lattice: from 1992 the outline nearest in time; up to 1991 SGI1931, then the
    # signed-distance blend (negative inside, cells) of PRE_OUTLINES, linear in time
    mid = [decimal(date(int(y), 4, 1)) for y in years]
    masks = {n: read(n) == 1 for n, _ in OUTLINES + PRE_OUTLINES}
    sd = {n: distance_transform_edt(~masks[n]) - distance_transform_edt(masks[n]) for n, _ in PRE_OUTLINES}
    od, po = [decimal(d) for _, d in OUTLINES], [decimal(d) for _, d in PRE_OUTLINES]
    full, near = [], []
    for y, t in zip(years, mid):
        if y > PRE_END:
            n = OUTLINES[int(np.argmin([abs(t - o) for o in od]))][0]
            full.append(masks[n])
        elif t <= po[0]:
            n = PRE_OUTLINES[0][0]
            full.append(masks[n])
        else:
            k = int(np.searchsorted(po, t))
            a = (t - po[k - 1]) / (po[k] - po[k - 1])
            n = f"{PRE_OUTLINES[k - 1][0]}-{PRE_OUTLINES[k][0]} {a:.3f}"
            full.append((1 - a) * sd[PRE_OUTLINES[k - 1][0]] + a * sd[PRE_OUTLINES[k][0]] < 0)
        near.append(n)
    # surface offset up to 1991: minus the GLAMOS mean elevation change from each survey to 1991-09-18
    allp_end = {r["date_start"]: float(r["dh_mean"]) for r in glamos("volumechange.csv", 0) if r["date_end"] == "19910918"}
    t_off = [decimal(date.fromisoformat(da)) for _, _, da, _, _ in PAIRS] + [decimal(date(1991, 9, 18))]
    z_off = [-allp_end[a] for a, _, _, _, _ in PAIRS] + [0.0]

    # columns: lattice cells on the stride, then stakes
    union = np.logical_or.reduce(full)
    sub = np.zeros_like(union)
    sub[::stride, ::stride] = True
    rr, cc = np.nonzero(union & sub)
    sr, sc = ((N1 - N) // DX).astype(int), ((E - E0) // DX).astype(int)
    z91, z21 = read("usurf_1994"), read("usurf_2021")
    z12 = tif_grid(DER / "thickness/C14-10_SwissALTI3D_r2019.tif")
    with rasterio.open(INP / "ipot.tif") as r:
        ip = r.read().astype(np.float64)
    D = read("D").astype(np.float64)
    allr, allc = np.concatenate([rr, sr]), np.concatenate([cc, sc])
    ncell, ncol = len(rr), len(rr) + len(stakes)
    assert np.isfinite(D[allr, allc]).all() and np.isfinite(ip[:, allr, allc]).all()
    ts = [decimal(d) for d in SURFACES]
    z = np.empty((len(years), ncol))
    a91, a12, a21 = z91[rr, cc], z12[rr, cc], z21[rr, cc]
    has12 = np.isfinite(a12)
    for i, t in enumerate(mid):
        two = a91 + (a21 - a91) * np.clip((t - ts[0]) / (ts[2] - ts[0]), 0, 1)
        three = np.where(t < ts[1], a91 + (a12 - a91) * np.clip((t - ts[0]) / (ts[1] - ts[0]), 0, 1),
                         a12 + (a21 - a12) * np.clip((t - ts[1]) / (ts[2] - ts[1]), 0, 1))
        z[i, :ncell] = np.where(has12, three, two) + (np.interp(t, t_off, z_off) if years[i] <= PRE_END else 0.0)
        z[i, ncell:] = [s[5] for s in stakes]
    mask = np.zeros((len(years), ncol), bool)
    for i, m in enumerate(full):
        mask[i, :ncell] = m[rr, cc]
    with open(RES / "calib_f1_geometry.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["hydro_year", "ice_mask", "area_km2", "surface_offset_m", "mean_surface_m"])
        for i, (y, t) in enumerate(zip(years, mid)):
            off = np.interp(t, t_off, z_off) if y <= PRE_END else 0.0
            wr.writerow([y, near[i], f"{full[i].sum() * DX ** 2 / 1e6:.4f}", f"{off:.2f}",
                         f"{z[i, :ncell][mask[i, :ncell]].mean():.1f}"])
    band = np.clip(((z - 2500) // 100).astype(int), 0, 6)

    # groups (column weights) and pieces (target, group, year, day0, day1)
    groups, gkey, pieces = [], {}, []

    def group(key, w):
        if key not in gkey:
            gkey[key] = len(groups)
            groups.append(w.astype(np.float32))
        return gkey[key]

    cell_km2 = stride ** 2 * DX ** 2 / 1e6
    for j, t in enumerate(T):
        if t["kind"] == "dv":
            spans = range(t["t0"] // 365, (t["t1"] - 1) // 365 + 1)
            for i in spans:
                w = np.where(mask[i], cell_km2 * 1e-3 / RHO, 0.0)          # m w.e. -> km3 ice
                g = group(("area", i), w)
                pieces.append((j, g, i, max(t["t0"] - 365 * i, 0), min(t["t1"] - 365 * i, 365)))
            continue
        ref = (t["t1"] - 1) // 365                                   # year holding the end
        if t["kind"] == "stake":
            w = np.zeros(ncol)
            w[ncell + t["stake"]] = 1
            g = group(("stake", t["stake"]), w)
        elif t["kind"] in ("bw_band", "ba_band"):
            on = mask[ref] & (band[ref] == t["band"])
            if not on.any():
                t["empty"] = 1
                continue
            g = group(("band", ref, t["band"]), on / on.sum())
        else:
            g = group(("glacier", ref), mask[ref] / mask[ref].sum())
        for i in range(t["t0"] // 365, (t["t1"] - 1) // 365 + 1):
            pieces.append((j, g, i, max(t["t0"] - 365 * i, 0), min(t["t1"] - 365 * i, 365)))

    # term matrices: value_k = vals @ M_k, observation_k = obs @ M_k
    n = len(T)
    kind = np.array([t["kind"] for t in T])
    ok = np.array([not t.get("empty") for t in T])
    obsv = np.array([t["obs"] for t in T])

    def select(idx):
        m = np.zeros((n, len(idx)))
        m[idx, np.arange(len(idx))] = 1
        return m

    Ma = select(np.nonzero((kind == "bw_band") & ok)[0])
    ib = np.nonzero((kind == "ba_band") & ok)[0]
    gl = np.nonzero(kind == "ba_glacier")[0]
    Mb = select(ib)
    for col, j in enumerate(ib):
        same = [g for g in gl if T[g]["period"] == T[j]["period"]]
        Mb[same, col] -= 1 / len(same)
    ic = np.nonzero(kind == "stake")[0]
    Mc = select(ic)
    yr = np.array([(T[j]["t1"] - 1) // 365 for j in ic])
    for col in range(len(ic)):
        Mc[ic[yr == yr[col]], col] -= 1 / (yr == yr[col]).sum()
    idd = np.nonzero(kind == "dv")[0]
    Md = select(idd)
    sig = [np.zeros(Ma.shape[1]), np.zeros(Mb.shape[1]), np.array([T[j]["sigma"] for j in ic]),
           np.array([T[j]["sigma"] for j in idd])]
    scl = [np.ones(Ma.shape[1]), np.ones(Mb.shape[1]), np.ones(len(ic)), np.array([T[j]["scale"] for j in idd])]

    pc = np.array(pieces)
    np.savez_compressed(
        INP / f"calib_f1_stride{stride}.npz", stride=stride, years=years, t_otl=t_otl, p_otl=p_otl, gm=gm, z=z,
        ipot=ip[:, allr, allc], D=D[allr, allc], groups=np.array(groups), pieces=pc, obs=obsv,
        kind=kind, label=np.array([str(t["label"]) for t in T]), d0=np.array([t["d0"].isoformat() for t in T]),
        d1=np.array([t["d1"].isoformat() for t in T]), Ma=Ma, Mb=Mb, Mc=Mc, Md=Md,
        sig_a=sig[0], sig_b=sig[1], sig_c=sig[2], sig_d=sig[3], scl_a=scl[0], scl_b=scl[1], scl_c=scl[2],
        scl_d=scl[3], near=np.array(near), ncell=ncell, ok=ok, idx_a=Ma.argmax(0), idx_b=ib, idx_c=ic, idx_d=idd)
    print(f"stride {stride}: {ncell} cells + {len(stakes)} stakes, years {y0}-{y1}, {n} values "
          f"({', '.join(f'{k} {(kind == k).sum()}' for k in dict.fromkeys(kind))}), empty bands "
          f"{int((~ok).sum())}, {len(groups)} groups, {len(pieces)} pieces; outlines by year: "
          + " ".join(f"{y}:{m}" for y, m in zip(years, near)))


def simulate(X, theta, chunk):
    """Values of every target for the parameter rows theta (Fm, r_ice, c_prec[, dT[, dg[, b[, dT_pre]]]]; missing
    columns dT = 0, dg = 0, b = B_PREC, dT_pre = 0; dT_pre adds to dT in the hydrological years up to PRE_END): smb.transfer and smb.step (the day of smb.balance) in an XLA-compiled loop over the 365
    days of a year, TensorFlow on the GPU, the transfer in float64 (so dg = 0, b = B_PREC give the forcing of decision
    5 as before), the balance in float32; per day the balance and the snow store are summed over the groups (column
    weights), the sums over the pieces accumulated in float64 on the host. A STORE target takes the group's snow store
    at its end day instead of the balance summed over its pieces (only its piece in the year holding the end; the store
    restarts at 0 on 1 October)."""
    import tensorflow as tf
    from smb import step, transfer
    years, pc, nT = X["years"], X["pieces"], len(X["obs"])
    store = np.isin(X["kind"], STORE)
    endyr = np.zeros(nT, int)
    np.maximum.at(endyr, pc[:, 0], pc[:, 2])
    pc = pc[~store[pc[:, 0]] | (pc[:, 2] == endyr[pc[:, 0]])]
    ipot, D = tf.constant(X["ipot"], tf.float32), X["D"]
    ng = max(len(np.unique(pc[pc[:, 2] == i, 1])) for i in range(len(years)))
    per_year = []
    for i in range(len(years)):
        p = pc[pc[:, 2] == i]
        gs, gi = np.unique(p[:, 1], return_inverse=True)
        W = np.zeros((X["groups"].shape[1], ng), np.float32)
        W[:, :len(gs)] = X["groups"][gs].T
        S = np.zeros((len(p), nT))
        S[np.arange(len(p)), p[:, 0]] = 1
        per_year.append((tf.constant(X["t_otl"][i], tf.float64), tf.constant(X["p_otl"][i], tf.float64),
                         tf.constant(X["z"][i] - Z_OTL, tf.float64), tf.constant(W),
                         tf.constant(np.stack([p[:, 4], gi], 1)), tf.constant(np.stack([p[:, 3], gi], 1)), S,
                         store[p[:, 0]][None, :], years[i] <= PRE_END))

    gm = tf.constant(X["gm"], tf.float64)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def year(t_otl, p_otl, dz, W, Dm, Fm, r_ice, r_snow, dT, dg, bp):
        S = tf.zeros_like(Dm)
        G = tf.TensorArray(tf.float32, size=365)
        H = tf.TensorArray(tf.float32, size=365)
        for k in tf.range(365):
            T, P = transfer(t_otl[k], p_otl[k], gm[k], dz, dg, bp)          # float64 (member, column)
            S, b = step(S, tf.cast(T, tf.float32) + dT, tf.cast(P, tf.float32), ipot[k], Dm, Fm, r_ice, r_snow, tf)
            G = G.write(k, tf.matmul(b, W))
            H = H.write(k, tf.matmul(S, W))                            # store after day k
        G, H = G.stack(), H.stack()                                    # (day, member, group)
        z = tf.zeros_like(G[:1])
        return tf.concat([z, tf.cumsum(G, axis=0)], axis=0), tf.concat([z, H], axis=0)

    out = np.zeros((len(theta), nT))
    full = np.zeros((len(theta), len(NAMES)))
    full[:, 5] = B_PREC
    full[:, :theta.shape[1]] = theta
    for k in range(0, len(theta), chunk):
        th = full[k:k + chunk]
        Dm = tf.constant(th[:, 2:3] * D[None, :], tf.float32)
        Fm, r, dT = (tf.constant(th[:, j:j + 1], tf.float32) for j in (0, 1, 3))
        dT_pre = tf.constant(th[:, 3:4] + th[:, 6:7], tf.float32)
        dg, bp = (tf.constant(th[:, j:j + 1], tf.float64) for j in (4, 5))
        for t_otl, p_otl, dz, W, end, start, S, st, is_pre in per_year:
            C, H = (tf.transpose(a, (0, 2, 1))                            # (day + 1, group, member)
                    for a in year(t_otl, p_otl, dz, W, Dm, Fm, r, 0.75 * r, dT_pre if is_pre else dT, dg, bp))
            v = tf.transpose(tf.gather_nd(C, end) - tf.gather_nd(C, start)).numpy()
            h = tf.transpose(tf.gather_nd(H, end)).numpy()
            out[k:k + chunk] += np.where(st, h, v).astype(np.float64) @ S
    return out


def term_grid(X, vals, k):
    """log-likelihood of term k on the s grid, (draw, node)."""
    M, sig, scl = X[f"M{k}"], X[f"sig_{k}"], X[f"scl_{k}"]
    r = vals @ M - X["obs"] @ M
    var = sig[None, :, None] ** 2 + (S_NODES[k][None, None, :] * scl[None, :, None]) ** 2
    return -0.5 * (r[:, :, None] ** 2 / var + np.log(2 * np.pi * var)).sum(axis=1)


def logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def loglik(X, vals):
    """Marginal log-likelihood per term (draw, 4), s integrated out on S_GRID."""
    return np.concatenate([np.stack([logsumexp(term_grid(X, v, k), 1) - np.log(len(S_NODES[k])) for k in TERMS], 1)
                           for v in np.array_split(vals, max(1, len(vals) // 8192))])


def run(stride, with_dt=False, bind=False, pre=False, start=1):
    """Rounds, kept draws, predictive bands and gate. Free parameters: Fm, c_prec, dg, b (and dT, dT_pre); theta rows
    hold all seven (Fm, r_ice = RATIO Fm, c_prec, dT, dg, b, dT_pre, zero when not free). bind: term d with sigma_obs
    only. Outputs under the stem calib_f1 (pre) or calib_f1_stationary. start: rounds before it are read from their
    saved round files (same seeds and box rule, so the same rounds), not rerun."""
    from scipy.stats import norm, qmc
    S_NODES["d"] = np.zeros(1) if bind else S_GRID
    stem = "calib_f1" if pre else "calib_f1_stationary"
    X = dict(np.load(INP / f"calib_f1_stride{stride}.npz"))
    X1 = dict(np.load(INP / "calib_f1_stride1.npz"))
    (RES / stem).mkdir(parents=True, exist_ok=True)
    ncol = X["z"].shape[1]
    chunk = int(2 ** np.floor(np.log2(GPU_BYTES / (12 * ncol * 4))))
    free = ([0, 2, 3, 4, 5] if with_dt else [0, 2, 4, 5]) + ([6] if pre else [])
    npar, fnames = len(free), [NAMES[i] for i in free]

    def full(f):
        th = np.zeros((len(f), len(NAMES)))
        th[:, free] = f
        th[:, 1] = RATIO * th[:, 0]
        return th

    prior = PRIOR[free]
    lo, hi = prior[:, 0], prior[:, 1]
    mean = cov = None                                   # round 1: uniform over the prior box
    summary, t_start = [], time.time()
    for rnd in range(1, MAX_ROUNDS + 1):
        t0 = time.time()
        if rnd < start:
            R = np.load(RES / f"{stem}/round{rnd}.npz")
            assert list(R["free"]) == free and bool(R["bind"]) == bind, rnd
            assert np.allclose(R["box"], prior) if mean is None else np.allclose(R["mean"], mean), rnd
            theta, L = R["theta"], R["loglik"]
            logq = R["logq"] if "logq" in R.files else np.zeros(len(theta))
        else:
            u = qmc.Sobol(npar, scramble=True, seed=SEED + rnd).random_base2(int(np.log2(N_DRAWS)))
            if mean is None:
                f, logq = lo + u * (hi - lo), np.zeros(len(u))
            else:
                z = norm.ppf(u)
                f, logq = mean + z @ np.linalg.cholesky(cov).T, -0.5 * (z ** 2).sum(1)    # log density + const
            theta = full(f)
            ok = ((f >= lo) & (f <= hi)).all(1)          # uniform prior: zero outside the box
            L = np.full((len(f), len(TERMS)), -np.inf)
            L[ok] = loglik(X, simulate(X, theta[ok], chunk))
            np.savez_compressed(RES / f"{stem}/round{rnd}.npz", box=prior, free=free, theta=theta, loglik=L,
                                logq=logq, bind=bind, mean=np.zeros(0) if mean is None else mean,
                                cov=np.zeros(0) if cov is None else cov)
        lp = L.sum(1)                                   # log posterior + const (uniform prior)
        lt = lp - logq
        w = np.exp(lt - lt.max())
        w /= w.sum()
        ess = 1 / (w ** 2).sum()
        summary.append((f"round {rnd} proposal, {', '.join(fnames)} (dg K m-1, b m-1)",
                        "uniform over the prior box " + " ".join(f"{a:.4g}-{b:.4g}" for a, b in prior) if mean is None
                        else "Gaussian, mean " + " ".join(f"{v:.4g}" for v in mean) + "; SD "
                        + " ".join(f"{v:.4g}" for v in np.sqrt(np.diag(cov)))
                        + f"; share inside the prior box {np.isfinite(lp).mean():.3f}"))
        summary.append((f"round {rnd} effective sample size", f"{ess:.1f}"))
        summary.append((f"round {rnd} largest normalised weight", f"{w.max():.4g}"))
        summary.append((f"round {rnd} wall time s (chunk {chunk}, {ncol} columns)",
                        f"{time.time() - t0:.0f}" + (" (read from its round file)" if rnd < start else "")))
        print(*summary[-4:], sep="\n", flush=True)
        if ess >= ESS_MIN or rnd == MAX_ROUNDS:
            break
        if ess >= ESS_FIT:                              # weighted moments, inflated
            mean, cov = (w[:, None] * theta[:, free]).sum(0), INFLATE ** 2 * np.cov(theta[:, free].T, aweights=w)
            how = f"importance-weighted mean and covariance x {INFLATE} ** 2"
        else:
            n_live = max(int((lp > lp.max() + np.log(1e-6)).sum()), 50 * npar)
            live = theta[np.argsort(-lp)[:n_live]][:, free]
            mean = live.mean(0)
            cov = np.cov(live.T) + np.diag((0.02 * (hi - lo)) ** 2)
            how = (f"{n_live} live draws (likelihood above 1e-6 of the largest, at least the {50 * npar} best), "
                   "unweighted")
        summary.append((f"round {rnd + 1} proposal from round {rnd}", how))

    # keep 1024 by systematic resampling; MAP = largest weight
    rng = np.random.default_rng(SEED)
    keep = np.searchsorted(np.cumsum(w), (rng.random() + np.arange(N_KEEP)) / N_KEEP)
    keep = np.minimum(keep, len(w) - 1)
    imap = int(np.argmax(lp))                            # MAP: largest posterior density of the last round
    uniq, inv = np.unique(np.append(keep, imap), return_inverse=True)
    t1 = time.time()
    V = simulate(X1, theta[uniq], int(2 ** np.floor(np.log2(GPU_BYTES / (12 * X1["z"].shape[1] * 4)))))
    summary.append(("rerun of the kept draws on every cell: unique draws, wall time s", f"{len(uniq)} {time.time() - t1:.0f}"))
    vk, vmap = V[inv[:-1]], V[inv[-1]]
    L1 = loglik(X1, V)
    summary.append(("kept draws, log-likelihood stride 1 minus stride " + str(stride) + ", median and largest |difference|",
                    f"{np.median(np.abs(L1.sum(1) - L[uniq].sum(1))):.3g} {np.abs(L1.sum(1) - L[uniq].sum(1)).max():.3g}"))

    # s from its conditional posterior per kept draw, then predictive values for every target
    s_kept = {}
    for k in TERMS:
        g = term_grid(X1, vk, k)
        p = np.exp(g - g.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        s_kept[k] = S_NODES[k][[rng.choice(len(S_NODES[k]), p=q) for q in p]]
    kind, obs = X1["kind"], X1["obs"]
    sig_all, scl_all, s_all = np.zeros(len(obs)), np.ones(len(obs)), np.full((N_KEEP, len(obs)), np.nan)
    for k in TERMS:
        idx = X1[f"idx_{k}"]
        sig_all[idx], scl_all[idx] = X1[f"sig_{k}"], X1[f"scl_{k}"]
        s_all[:, idx] = s_kept[k][:, None]
    # values outside the likelihood (glacier-wide, fixed dates) get no noise
    noise = rng.standard_normal(vk.shape) * np.sqrt(sig_all ** 2 + np.nan_to_num(s_all) ** 2 * scl_all ** 2)
    pred = vk + noise
    # band Ba and stake values are compared as fitted: band minus period mean, stake minus year mean
    fitted = {k: (X1[f"M{k}"], X1[f"idx_{k}"]) for k in ("b", "c")}
    model_f, pred_f, obs_f = vk.copy(), pred.copy(), obs.copy().astype(float)
    for k, (M, idx) in fitted.items():
        model_f[:, idx] = vk @ M
        sd = np.sqrt(X1[f"sig_{k}"] ** 2 + s_kept[k][:, None] ** 2)
        pred_f[:, idx] = vk @ M + rng.standard_normal((N_KEEP, len(idx))) * sd
        obs_f[idx] = obs @ M
    vmap_f = vmap.copy()
    for k, (M, idx) in fitted.items():
        vmap_f[idx] = vmap @ M
    q = lambda a: np.percentile(a, [5, 50, 95], axis=0)
    mq, pq = q(model_f), q(pred_f)
    inside = (obs_f >= pq[0]) & (obs_f <= pq[2]) & X1["ok"]
    with open(RES / f"{stem}_bands.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["kind", "label", "date0", "date1", "observed", "sigma_obs", "model_p5", "model_p50",
                     "model_p95", "predictive_p5", "predictive_p50", "predictive_p95", "inside_90", "map",
                     "fitted_as"])
        how = {"bw_band": "value", "ba_band": "minus period glacier-wide mean", "stake": "minus year stake mean",
               "dv": "value, km3"}
        for j in range(len(obs)):
            wr.writerow([kind[j], X1["label"][j], X1["d0"][j], X1["d1"][j], f"{obs_f[j]:.4f}", f"{sig_all[j]:.4f}",
                         *(f"{v:.4f}" for v in mq[:, j]), *(f"{v:.4f}" for v in pq[:, j]), int(inside[j]),
                         f"{vmap_f[j]:.4f}", how.get(kind[j], "shown, not fitted") if X1["ok"][j] else "no model cells"])

    # gate
    cov = lambda kinds: inside[np.isin(kind, kinds) & X1["ok"]].mean()
    gl = np.nonzero(kind == "ba_glacier")[0]
    bmap = {int(X1["label"][j]): vmap[j] for j in gl}
    lowest = min(bmap, key=bmap.get)
    ba_obs, ba_kept = obs[gl].mean(), vk[:, gl].mean(1)
    ib = X1["idx_b"]
    hb = np.array([int(lab.split()[1]) for lab in X1["label"][ib]])
    unit = lambda nm: SHOW.get(nm, (1, ""))
    hu = np.unique(hb)
    slope = lambda v: np.polyfit(hu / 100, [v[..., ib[hb == h]].mean(-1) for h in hu], 1)[0]   # m w.e. per 100 m
    mis = []
    for k in TERMS:
        M = X1[f"M{k}"]
        r = (vmap - obs) @ M
        line = f"{r.mean():+.4f} {np.sqrt((r ** 2).mean()):.4f}"
        if k == "d":
            line += "; in sigma_obs " + " ".join(f"{v:+.2f}" for v in r / X1["sig_d"])
        mis.append((f"MAP misfit term {k} ({M.shape[1]} values, as fitted{', km3' if k == 'd' else ', m w.e.'}): mean, "
                    "root mean square of model minus observed" + ("; each in units of sigma_obs" if k == "d" else "")
                    + f"; MAP log-likelihood stride 1", f"{line}; {L1[inv[-1], TERMS.index(k)]:.2f}"))
    kt = theta[keep][:, free]
    cor = np.corrcoef(kt.T)
    summary += [
        ("gate: effective sample size of the last round", f"{ess:.1f}"),
        ("gate: share of band winter balances inside the 90 % predictive band", f"{cov(['bw_band']):.3f}"),
        ("gate: share of band annual balances (minus period mean) inside", f"{cov(['ba_band']):.3f}"),
        ("gate: share of band winter and annual balances together inside", f"{cov(['bw_band', 'ba_band']):.3f}"),
        (f"gate: volume changes inside the 90 % predictive band, of {(kind == 'dv').sum()}",
         f"{int(inside[kind == 'dv'].sum())}"),
        ("volume changes: distance of the observation outside the 90 % predictive band, km3 (0 inside), "
         + " | ".join(X1["label"][X1["idx_d"]]),
         " | ".join(f"{max(pq[0, j] - obs[j], obs[j] - pq[2, j], 0):.4f}" for j in X1["idx_d"])),
        ("stakes (minus year mean) inside", f"{cov(['stake']):.3f}"),
        ("glacier-wide observation-period annual balance inside the model 90 % band (not fitted)",
         f"{((obs[gl] >= mq[0, gl]) & (obs[gl] <= mq[2, gl])).mean():.3f}"),
        ("gate: MAP draw, year of its lowest glacier-wide annual balance, value m w.e.", f"{lowest} {bmap[lowest]:+.3f}"),
        ("gate: MAP draw, glacier-wide annual balance 2001 m w.e.", f"{bmap[2001]:+.3f}"),
        (f"glacier-wide annual balance, mean of {len(gl)} observation periods {min(bmap)}-{max(bmap)} m w.e.: observed, "
         "MAP, kept draws 5 / 50 / 95 % (not fitted)",
         f"{ba_obs:+.3f} {np.mean(list(bmap.values())):+.3f} "
         + " ".join(f"{v:+.3f}" for v in np.percentile(ba_kept, [5, 50, 95]))),
        *[(f"band annual balance minus the period glacier-wide mean, mean of the years, band {h} m ({(hb == h).sum()} "
           "years): observed, MAP, kept draws 5 / 50 / 95 %",
           f"{obs_f[ib[hb == h]].mean():+.3f} {vmap_f[ib[hb == h]].mean():+.3f} "
           + " ".join(f"{v:+.3f}" for v in np.percentile(model_f[:, ib[hb == h]].mean(1), [5, 50, 95])))
          for h in np.unique(hb)],
        ("band annual balance gradient, least-squares slope over the band means above, m w.e. per 100 m: observed, MAP, "
         "kept draws 5 / 50 / 95 %", f"{slope(obs_f):+.4f} {slope(vmap_f):+.4f} "
         + " ".join(f"{v:+.4f}" for v in np.percentile([slope(m) for m in model_f], [5, 50, 95]))),
        ("volume changes km3, " + " | ".join(X1["label"][X1["idx_d"]]) + ": observed, MAP, kept-draw model median",
         " | ".join(f"{obs[j]:+.4f} {vmap[j]:+.4f} {np.median(vk[:, j]):+.4f}" for j in X1["idx_d"])),
        *mis,
        (f"MAP draw {', '.join(NAMES)} (r_ice = {RATIO} Fm{'' if with_dt else ', dT not free'}{'' if pre else ', dT_pre not free'}; dg K m-1, b m-1)",
         " ".join(f"{v:.5g}" for v in theta[imap])),
        ("kept draws, unique", f"{len(np.unique(keep))}"),
        *[(f"kept {nm} 5 / 50 / 95 % {unit(nm)[1]}".rstrip(),
           " ".join(f"{v * unit(nm)[0]:.5g}" for v in np.percentile(theta[keep, i], [5, 50, 95])))
          for i, nm in zip(free, fnames)],
        *[(f"kept draws, correlation {fnames[a]} - {fnames[b]}", f"{cor[a, b]:+.3f}")
          for a in range(npar) for b in range(a + 1, npar)],
        *[(f"kept model-error SD term {k} 5 / 50 / 95 %" + (" (binding: none, sigma_obs only)" if bind and k == "d" else ""),
           " ".join(f"{v:.3g}" for v in np.percentile(s_kept[k], [5, 50, 95]))) for k in TERMS],
        ("total wall time s", f"{time.time() - t_start:.0f}")]
    with open(RES / f"{stem}_summary.csv", "w", newline="") as f:
        csv.writer(f).writerows([("quantity", "value"), *summary])
    with open(RES / f"{stem}_draws.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["draw", "round", "sobol_index", "Fm_m_ice_per_day_degC", "r_ice", "r_snow", "c_prec",
                     "dT_degC", "dg_K_per_km", "b_percent_per_100m", "dT_pre_degC", "r_ice_over_Fm", "weight",
                     *[f"loglik_{k}" for k in TERMS], *[f"s_{k}" for k in TERMS], "is_map"])
        for i, j in enumerate(keep):
            wr.writerow([i, rnd, j, f"{theta[j, 0]:.6e}", f"{theta[j, 1]:.6e}", f"{0.75 * theta[j, 1]:.6e}",
                         f"{theta[j, 2]:.5f}", f"{theta[j, 3]:.4f}", f"{theta[j, 4] * 1e3:.4f}",
                         f"{theta[j, 5] * 1e4:.4f}", f"{theta[j, 6]:.4f}",
                         f"{theta[j, 1] / theta[j, 0]:.5f}", f"{w[j]:.4e}",
                         *(f"{v:.3f}" for v in L[j]), *(f"{s_kept[k][i]:.4f}" for k in TERMS), int(j == imap)])
    for r in summary:
        print(*r, sep="\t")


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare(int(sys.argv[2]))
    else:
        a = sys.argv[3:]
        start = int(next((x[6:] for x in a if x.startswith("start=")), 1))
        assert set(a) - {f"start={start}"} <= {"dT", "bind", "pre"}, sys.argv
        run(int(sys.argv[2]), "dT" in a, "bind" in a, "pre" in a, start)
