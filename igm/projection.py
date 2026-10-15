"""Projections 2026-2100 (DESIGN.md step 6, decisions 12-14): scenario weather, members, run lists, start states,
reduction of the runs and the step-6 gate.
Fix round (DESIGN.md 'F6'): draws of results/posterior_f3_128.csv (F3, 128 distinct samples), start states from
inputs/f5_state2026.npz (thk_dec8, topg_dec8), file prefix f6 (stages f6_*, results/f6_*.csv, results/f6/,
inputs/f6_start2026/, nets/f6/). The first ensemble used prefix step6 (stages s6_*), posterior_128.csv,
step5_state2026.npz, inputs/start2026/ and nets/step6/ (git history of this file); its result files stay as they are.

Scenario weather of a member (decision 12), hydrological years Y = 2027-2100:
  T(Y, day) = T_det(y_Y, day) + dT(station, chain, SSP, level, Y, month of the day)      degC at OTL
  P(Y, day) = P(y_Y, day) x P_ratio(station, chain, SSP, Y, month of the day)              m w.e. d-1 at OTL
  y_Y        the observed hydrological year lent to Y: the year order of the member, independent permutations of
             1991-2026 (36 years) laid end to end, the first 74 taken; numpy default_rng(seed)
  T_det      the observed OTL days of forcing_obs.npz (weather.py) detrended to the 1991-2020 mean: per season of the
             day (DJF, MAM, JJA, SON of the hydrological year), the least-squares line through the 36 season means
             1991-2026 (the line of the anchor in scenarios.py), T_det = T - slope (Y - 2005.5); P not detrended
  dT, P_ratio inputs/scenarios_monthly.csv (scenarios.py); the station's changes (OTL or ROE) applied to the OTL days
The station transfer (dT, dg, b, c_prec of the draw) is clim_otl's, unchanged; dT_pre (F1) acts only up to 1991.
Members (decision 13): 128 posterior draws (POSTERIOR) x 26 chains x 3 SSPs; per draw one 2026 level (native,
anchored) and one station (OTL, ROE): the draws sorted by (sample, draw) take the four combinations in turn, the turn
starting so that the MAP draw holds (anchored, OTL); each combination holds 32 draws. Seed of member k (0-9983, order
draw, chain, SSP): SEED + k. MAP draw: the draw of the largest weight (f6: one draw per sample); its 78 forcings
also run with u = 0 (same seeds; start state: the MAP sample's 2021-2026 run of F3 rerun with u = 0, stage P6_prep, the
F3 line with u = 0). Film members: the MAP draw under each SSP with the chain whose 2081-2100 annual mean dT (the
draw's station, anchored) lies closest to the median of the 26 chains; they keep thk and the surface velocity every
year (track yearly). Film member of the films (DESIGN.md Outputs, step 7; 'film' below): of the anchored SSP2-4.5
members (u of the draw) those with the median disappearance year (0.01 km2, lower middle), the one whose A1 series
2026-2100 lies closest (root mean square) to the group median; its hindcast (F5 line), its 2021-2026 path (F3 line)
and its projection rerun with yearly fields (stage P6_film).
Start (decision 8): the 2026-09-30 state of the draw's decision-8 path (STATE5 thk_dec8, topg_dec8), per draw
START_DIR/dNNN.npz (topg, thk_end); one cold network per start state (stage P6_nets, 3000 iterations, saved by track
save_net), the members warm-start from it with 0 iterations (step-3 settings).
Experiment projection: 2026.75 to 2100.75, basin on (F2), stop at no ice (decision 14).
Reduction per run: end reason and time, years, V and A1 (thk >= 1 m) at the start, disappearance years (first year with
A1 < 0.01 km2, and < 0.1 km2, with no later return; empty: not by 2100), half-volume years (first year with
V <= V_2026 / 2, and <= V_2021 / 2 from 2022 on through the draw's 2021-2026 run, V_2021 its start), NaN count, ice-free grid checks
(last_ice of track: -1 or 2026..end year; >= 2026 on every start ice cell; = end year exactly on the end ice cells),
the ice the basin removed 2026-2100 (km3).
No-flow twins (noflow.py, same start, weather and balance; the IGM stop rule: a member stops at the first 1 October
after the start without a cell >= 1 m).
Outputs: results/step6_forcing_check.csv (pack); results/P6_members.csv, stages/P6_{prep,nets,proj,rerun,flowoff}.txt,
stages/P6_noflow_K.csv (members); s03 START_DIR/*.npz (starts); results/P6_runs_STAGE.csv, s03
results/P6/series_STAGE.csv.gz and icefree_STAGE.npz (reduce); results/P6_noflow.csv, s03
results/P6/noflow_series.csv.gz, noflow_icefree.npz (noflow); results/P6_gates.csv (gate); stages/P6_film.txt,
results/P6_film.csv (film); results/P6_film_check.csv (filmcheck).
usage:  s02 (agl env):  python igm/projection.py pack
        Mac:            python projection.py members | film
        s03 (env.sh):   python projection.py starts | reduce STAGE | noflow | gate | filmcheck
        no-flow members (s03, env.sh; one GPU, the thread of slot 5, beside P6_proj):
          nrun bash -c 'for k in 0 1 2 3 4 5 6 7 8 9; do CUDA_VISIBLE_DEVICES=1 taskset -c 5 nice -n 10 python
            noflow.py stages/f6_noflow_$k.csv restart 2026 2100 results/f6/noflow_$k.csv || exit 1; done'
"""
import csv
import gzip
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parent
RES, STAGES, INP = IGM / "results", IGM / "stages", IGM / "inputs"
YEARS, OBS_YEARS, BASE_MID = range(2027, 2101), range(1991, 2027), 2005.5
STATIONS, SSPS, LEVELS = ("OTL", "ROE"), ("SSP1-2.6", "SSP2-4.5", "SSP5-8.5"), ("native", "anchored")
COMBOS = (("anchored", "OTL"), ("anchored", "ROE"), ("native", "OTL"), ("native", "ROE"))
HMONTHS = (10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9)
HDAYS = (31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30)
SEASON = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}     # DJF MAM JJA SON
SEED, START, END = 2100000, 2026.75, 2100.75
SCEN = "inputs/forcing_scen.npz"
NOFLOW_BATCH = 1024
P6, POSTERIOR, STATE5 = "f6", "posterior_f3_128.csv", "f5_state2026.npz"
START_DIR, NET_DIR = "inputs/f6_start2026", "nets/f6"
OUT = RES / P6                                                   # s03: series, ice-free grids, no-flow outputs


def read_csv(path):
    return list(csv.DictReader(open(path)))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# --- scenario weather -------------------------------------------------------------------------------------------------

def pack():
    """inputs/forcing_scen.npz and results/step6_forcing_check.csv (s02, from forcing_obs.npz, scenarios_monthly.csv)."""
    f = np.load(INP / "forcing_obs.npz")
    k = [list(f["hydro_year"]).index(y) for y in OBS_YEARS]
    t, p = f["t"][k], f["p"][k]
    hmonth = np.repeat(np.arange(12), HDAYS)                      # day -> month index, hydrological order
    season = np.array([SEASON[HMONTHS[m]] for m in hmonth])
    Y = np.array(OBS_YEARS, float)
    slope, rows = np.zeros(4), []
    for s, name in enumerate(("DJF", "MAM", "JJA", "SON")):
        x = t[:, season == s].mean(1)
        slope[s], icpt = np.polyfit(Y, x, 1)
        base = x[:30].mean()
        rows.append(("trend 1991-2026 at 2026 minus 1991-2020 mean", name, f"{slope[s] * 2026 + icpt - base:.4f}", "degC"))
        rows.append(("slope 1991-2026", name, f"{slope[s] * 10:.4f}", "degC per decade"))
    t_det = t - slope[season][None] * (Y[:, None] - BASE_MID)
    yr = lambda a: (a.mean(1), np.polyfit(Y, a.mean(1), 1)[0])
    (m0, s0), (m1, s1) = yr(t), yr(t_det)
    rows += [("annual slope before, after detrending", "year", f"{s0 * 10:.4f} {s1 * 10:.4f}", "degC per decade"),
             ("annual 1991-2020 mean before, after detrending", "year", f"{m0[:30].mean():.4f} {m1[:30].mean():.4f}",
              "degC"),
             ("annual trend at 2026 minus 1991-2020 mean (hydrological years)", "year",
              f"{np.polyval(np.polyfit(Y, m0, 1), 2026) - m0[:30].mean():.4f}", "degC")]
    S = read_csv(INP / "scenarios_monthly.csv")
    chains = sorted({r["chain"] for r in S})
    dT = np.full((2, len(chains), 3, 2, len(YEARS), 12), np.nan)
    Pr = np.full((2, len(chains), 3, 2, len(YEARS), 12), np.nan)
    gw = np.full((3, len(YEARS)), np.nan)
    for r in S:
        i = (STATIONS.index(r["station"]), chains.index(r["chain"]), SSPS.index(r["ssp"]), LEVELS.index(r["level"]),
             int(r["hydro_year"]) - YEARS[0], HMONTHS.index(int(r["month"])))
        dT[i], Pr[i] = float(r["dT_degC"]), float(r["P_ratio"])
        gw[i[2], i[4]] = float(r["global_warming_degC"])
    assert np.isfinite(dT).all() and np.isfinite(Pr).all() and (Pr > 0).all()
    assert np.array_equal(Pr[:, :, :, 0], Pr[:, :, :, 1]), "P ratio differs between levels"
    np.savez(INP / "forcing_scen.npz", years_obs=np.array(OBS_YEARS), t_det=t_det, p=p, hmonth=hmonth,
             dT=dT, Pr=Pr[:, :, :, 0], gw=gw, chains=np.array(chains), slope=slope)
    # the member weather of one draw rebuilt by hand against weather()
    F = dict(np.load(INP / "forcing_scen.npz"))
    o = year_order(SEED + 7)
    tt, pp = weather(F, 1, 3, 2, 1, o, 2050)
    k50 = o[2050 - YEARS[0]]
    d = 200                                                       # 19 April
    ref_t = t[k50, d] - slope[1] * (OBS_YEARS[k50] - BASE_MID) + dT[1, 3, 2, 1, 2050 - YEARS[0], 6]
    rows.append(("member weather rebuilt by hand (ROE, chain 4, SSP5-8.5, anchored, 2050, day 200): |t|, |p| diff",
                 "", f"{abs(tt[d] - ref_t):.2e} {abs(pp[d] - p[k50, d] * Pr[1, 3, 2, 0, 23, 6]):.2e}", ""))
    c = np.bincount(o[:72], minlength=36)
    rows.append(("year order of one member: uses per observed year in 2027-2098, min max", "", f"{c.min()} {c.max()}",
                 ""))
    for ssp in range(3):
        for lev in range(2):
            for st in range(2):
                ann = (F["dT"][st, :, ssp, lev] * np.array(HDAYS)).sum(-1) / 365
                rows.append((f"annual dT, median of 26 chains, 2027 | 2050 | 2100, {STATIONS[st]}", f"{SSPS[ssp]} "
                             f"{LEVELS[lev]}", " ".join(f"{np.median(ann[:, y - YEARS[0]]):.3f}"
                                                        for y in (2027, 2050, 2100)), "degC"))
    write_csv(RES / "step6_forcing_check.csv", ["check", "season", "value", "unit"], rows)
    for r in rows:
        print(" | ".join(r))


def year_order(seed):
    """Index (into 1991-2026) of the observed year lent to each of 2027-2100."""
    rng = np.random.default_rng(seed)
    n = len(OBS_YEARS)
    return np.concatenate([rng.permutation(n) for _ in range(-(-len(YEARS) // n))])[:len(YEARS)]


def weather(F, st, ch, ss, lv, order, Y):
    """t, p (..., 365) float64 at OTL of hydrological year Y; st, ch, ss, lv indices (scalars or member arrays),
    order (..., 74) year orders."""
    i = Y - YEARS[0]
    k = np.asarray(order)[..., i]
    t = F["t_det"][k] + F["dT"][st, ch, ss, lv, i][..., F["hmonth"]]
    p = F["p"][k] * F["Pr"][st, ch, ss, i][..., F["hmonth"]]
    return t, p


def member_forcing(path, c):
    """(clim_index, t (74, 365), p) of one member for clim_otl; c the clim_otl scenario config."""
    F = dict(np.load(path))
    chains = list(F["chains"])
    ix = (STATIONS.index(str(c.station)), chains.index(str(c.chain)), SSPS.index(str(c.ssp)), LEVELS.index(str(c.level)))
    order = year_order(int(c.seed))
    tp = [weather(F, *ix, order, Y) for Y in YEARS]
    return {Y: i for i, Y in enumerate(YEARS)}, np.array([a for a, _ in tp]), np.array([b for _, b in tp])


# --- members and run lists --------------------------------------------------------------------------------------------

def overrides(r):
    from hindcast import overrides as o
    return o(r)


def scen_overrides(m):
    return (f"processes.clim_otl.scenario.station={m['station']} processes.clim_otl.scenario.chain={m['chain']} "
            f"processes.clim_otl.scenario.ssp={m['ssp']} processes.clim_otl.scenario.level={m['level']} "
            f"processes.clim_otl.scenario.seed={m['seed']}")


def f3_line(r):
    """The words of the F3 2021-2026 line (fit_f3.py) of a draw's sample: f3_r1fix where it was rerun, else its round."""
    for st in ("f3_r1fix", f"f3_r{r['round']}"):
        for line in open(STAGES / f"{st}.txt"):
            if line.startswith(r["sample"] + "_21 "):
                return line.split()
    raise KeyError(r["sample"])


def members():
    D = read_csv(RES / POSTERIOR)
    F = dict(np.load(INP / "forcing_scen.npz"))                  # rsynced from s02 (pack)
    chains = list(F["chains"])
    mapd = int(max(D, key=lambda r: float(r["weight"]))["draw"])
    order = [int(r["draw"]) for r in sorted(D, key=lambda r: (r["sample"], int(r["draw"])))]
    i0 = order.index(mapd)
    combo = {d: COMBOS[(i - i0) % 4] for i, d in enumerate(order)}
    Dd = {int(r["draw"]): r for r in D}
    top = Dd[mapd]["sample"]
    st = STATIONS.index(combo[mapd][1])
    film = {}
    for s, ssp in enumerate(SSPS):
        ann = (F["dT"][st, :, s, 1, 2081 - YEARS[0]:] * np.array(HDAYS)).sum(-1).mean(-1) / 365
        film[ssp] = int(np.argmin(np.abs(ann - np.median(ann))))
    M, k = [], 0
    for r in D:
        d = int(r["draw"])
        lev, stn = combo[d]
        for c, ch in enumerate(chains):
            for ssp in SSPS:
                M.append(dict(member=f"d{d:03d}_c{c:02d}_{ssp[3]}", draw=d, sample=r["sample"], u=r["u"], f=r["f"],
                              chain=ch, chain_index=c, ssp=ssp, level=lev, station=stn, seed=SEED + k,
                              start=f"{START_DIR}/d{d:03d}.npz", net=f"{NET_DIR}/d{d:03d}.npz", V_2021=r["V_start21"],
                              film=int(d == mapd and film[ssp] == c), map_u0=0))
                k += 1
    for m in [m for m in M if m["draw"] == mapd]:
        M.append(dict(m, member="u0" + m["member"][4:], u="0.0", start=f"{START_DIR}/map_u0.npz",
                      net=f"{NET_DIR}/map_u0.npz", V_2021="", film=0, map_u0=1))
    cols = list(M[0])
    write_csv(RES / f"{P6}_members.csv", cols, [[m[c] for c in cols] for m in M])
    par = lambda m: overrides(dict(Dd[m["draw"]], u=m["u"]))
    head = f"# NAME EXPERIMENT overrides (run_stage.sh); DESIGN.md step 6, fix round F6 (projection.py members): "
    prep = ["map_u0_21"] + [("inputs.geometry.u=0.0" if w.startswith("inputs.geometry.u=") else w)
                            for w in f3_line(Dd[mapd])[1:]]
    (STAGES / f"{P6}_prep.txt").write_text(head + f"MAP draw {mapd} ({top}) with u = 0: its F3 2021-2026 line "
                                           "(2021 DEM, observed weather, warm start as F3) with u = 0 (start state "
                                           "of its 78 u = 0 members)\n" + " ".join(prep) + "\n")
    nets = []
    for m in M:
        if m["chain_index"] == 0 and m["ssp"] == "SSP2-4.5":     # the weather of the training step is not used by it
            nets.append(f"n_{Path(m['net']).stem} projection inputs.geometry.restart={m['start']} {par(m)} "
                        f"{scen_overrides(m)} processes.time.end={START} outputs.track.save_net={m['net']}")
    (STAGES / f"{P6}_nets.txt").write_text(head + "one cold network (3000 iterations) per start state, time.end = "
                                           "time.start\n" + "\n".join(nets) + "\n")
    proj = {}
    for m in M:
        proj[m["member"]] = (f"{m['member']} projection inputs.geometry.restart={m['start']} {par(m)} {scen_overrides(m)} "
                             f"processes.iceflow.unified.nbit_init=0 processes.warmstart.net={m['net']} "
                             f"processes.warmstart.nbit=0" + (" outputs.track.yearly=true" if m["film"] else ""))
    (STAGES / f"{P6}_proj.txt").write_text(head + f"{len(proj)} members 2026.75-2100.75, warm start 0 iterations from "
                                           "the start state's own network\n" + "\n".join(proj.values()) + "\n")
    fo = [" ".join(w for w in proj[n].split()[2:] if not w.startswith(("processes.iceflow", "processes.warmstart")))
          for n in ("d000_c00_1", "d000_c00_5")]
    (STAGES / f"{P6}_flowoff.txt").write_text(head + "scenario weather check (checks/step6.py flowoff): IGM without "
                                              "flow on CPU, 2026.75-2032.75\n" + "".join(
        f"{n} flowoff inputs.geometry.dem=restart {w} processes.clim_otl.scenario.file={SCEN} "
        f"processes.time.start={START} processes.time.end=2032.75 outputs.track.dates=[]\n"
        for n, w in zip(("d000_c00_1", "d000_c00_5"), fo)))
    rng = np.random.default_rng(SEED)
    pick = [m["member"] for m in M if m["film"] and m["ssp"] == "SSP2-4.5"]
    pool = [m["member"] for m in M if not m["film"] and not m["map_u0"]]
    pick += [pool[i] for i in rng.choice(len(pool), 3, replace=False)]
    (STAGES / f"{P6}_rerun.txt").write_text(head + "4 members rerun (gate: same disappearance year)\n" + "\n".join(
        "rr_" + proj[p] for p in pick) + "\n")
    cols_nf = ("member", "Fm", "r_ice", "c_prec", "dT", "dg", "b", "u", "offset", "restart", "station", "chain", "ssp",
               "level", "seed")
    flow = [m for m in M if not m["map_u0"]]
    for b in range(0, len(flow), NOFLOW_BATCH):
        write_csv(STAGES / f"{P6}_noflow_{b // NOFLOW_BATCH}.csv", cols_nf,
                  [[m["member"], Dd[m["draw"]]["Fm"], "", *(Dd[m["draw"]][c] for c in ("c_prec", "dT", "dg", "b")),
                    m["u"], "", m["start"], m["station"], m["chain"], m["ssp"], m["level"], m["seed"]]
                   for m in flow[b:b + NOFLOW_BATCH]])
    write_csv(STAGES / f"{P6}_noflow_check.csv", cols_nf,             # the two members of the flowoff check
              [[m["member"], Dd[m["draw"]]["Fm"], "", *(Dd[m["draw"]][c] for c in ("c_prec", "dT", "dg", "b")),
                m["u"], "", m["start"], m["station"], m["chain"], m["ssp"], m["level"], m["seed"]]
               for m in flow if m["member"] in ("d000_c00_1", "d000_c00_5")])
    n = {c: sum(1 for d in combo.values() if d == c) for c in COMBOS}
    print(f"results/{P6}_members.csv: {len(M)} members ({len(flow)} + {len(M) - len(flow)} u = 0); MAP draw {mapd} "
          f"({top}, {combo[mapd]}); film chains {[chains[film[s]] for s in SSPS]}; draws per combination {n}; "
          f"stages: prep 1, nets {len(nets)}, proj {len(proj)}, rerun {pick}, noflow "
          f"{-(-len(flow) // NOFLOW_BATCH)} files")


def line_of(path, name):
    return next(l.split() for l in open(path) if l.split()[:1] == [name])


def film():
    """Film member of the films (docstring): stages/P6_film.txt (hindcast, 2021-2026 path, projection; yearly fields)
    and results/P6_film.csv; after 'reduce P6_proj' (results/P6_runs_P6_proj.csv, s03 results/P6/series_P6_proj.csv.gz
    copied to the Mac)."""
    import gzip as gz
    M = {m["member"]: m for m in read_csv(RES / f"{P6}_members.csv")}
    R = {r["name"]: r for r in read_csv(RES / f"{P6}_runs_{P6}_proj.csv")}
    G = sorted(n for n, m in M.items() if m["level"] == "anchored" and m["ssp"] == "SSP2-4.5" and m["map_u0"] == "0")
    dis = {n: int(R[n]["dis_001"] or 2101) for n in G}
    med = sorted(dis.values())[(len(G) - 1) // 2]
    A = {n: np.zeros(len(YEARS) + 1) for n in G}
    with gz.open(OUT / f"series_{P6}_proj.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            if r["name"] in A:
                A[r["name"]][int(r["year"]) - 2026] = float(r["A1_km2"])
    Amed = np.median(np.array([A[n] for n in G]), 0)
    cand = sorted((float(np.sqrt(np.mean((A[n] - Amed) ** 2))), n) for n in G if dis[n] == med)
    rms, name = cand[0]
    m = M[name]
    d = int(m["draw"])
    r = {x["draw"]: x for x in read_csv(RES / POSTERIOR)}[str(d)]
    hind = line_of(STAGES / "f5_hind.txt", f"h{d:03d}")
    path = f3_line(r)
    pr = line_of(STAGES / f"{P6}_proj.txt", name)
    lines = [["f_" + hind[0]] + hind[1:], ["f_" + path[0]] + path[1:], ["f_" + pr[0]] + pr[1:]]
    for line in lines:
        if "outputs.track.yearly=true" not in line:
            line.append("outputs.track.yearly=true")
    (STAGES / f"{P6}_film.txt").write_text(
        "# NAME EXPERIMENT overrides (run_stage.sh); DESIGN.md step 6, fix round F6 (projection.py film): the film "
        "member's hindcast, 2021-2026 path and projection rerun with yearly fields\n"
        + "\n".join(" ".join(line) for line in lines) + "\n")
    restart = next(w.split("=", 1)[1] for w in hind if w.startswith("inputs.geometry.restart="))
    write_csv(RES / f"{P6}_film.csv", ["quantity", "value"], [
        ("group", "anchored, SSP2-4.5, u of the draw"), ("members in the group", len(G)),
        ("median disappearance year 0.01 km2 (lower middle)", med),
        ("members with that year", len(cand)), ("film member", name), ("draw", d), ("sample", r["sample"]),
        ("chain", m["chain"]), ("station", m["station"]), ("u", m["u"]), ("f", m["f"]), ("seed", m["seed"]),
        ("rms of A1 minus the group median 2026-2100, km2", f"{rms:.4f}"),
        ("next two candidates (rms km2)", "; ".join(f"{n} {x:.4f}" for x, n in cand[1:3])),
        ("disappearance 0.01 | 0.1 km2, half-volume 2026 | 2021", f"{R[name]['dis_001']} | {R[name]['dis_01']} | "
                                                                   f"{R[name]['half_2026']} | {R[name]['half_2021']}"),
        ("group median disappearance 0.1 km2, half-volume 2026",
         f"{sorted(int(R[n]['dis_01'] or 2101) for n in G)[(len(G) - 1) // 2]}, "
         f"{sorted(int(R[n]['half_2026'] or 2101) for n in G)[(len(G) - 1) // 2]}"),
        ("hindcast restart (spin-up end state)", restart),
        ("stage", f"stages/{P6}_film.txt")])
    print(open(RES / f"{P6}_film.csv").read())


def filmcheck():
    """results/P6_film_check.csv (s03): each film rerun against the run it repeats (f5_hind, the F3 _21 run, P6_proj):
    largest |difference| of V and A1 over the common saves, years, and the yearly fields kept (thk_y, us_y, vs_y)."""
    import hindcast
    F = {r["quantity"]: r["value"] for r in read_csv(RES / f"{P6}_film.csv")}
    r = {x["draw"]: x for x in read_csv(RES / POSTERIOR)}[F["draw"]]
    orig = {"hindcast": IGM / "runs/f5_hind" / f"h{int(F['draw']):03d}", "path": hindcast.dec8_run(r),
            "projection": IGM / "runs" / f"{P6}_proj" / F["film member"]}
    new = {"hindcast": f"f_h{int(F['draw']):03d}", "path": f"f_{r['sample']}_21", "projection": f"f_{F['film member']}"}
    rows = []
    for k in ("hindcast", "path", "projection"):
        a = read_csv(orig[k] / "series.csv")
        b = read_csv(IGM / "runs" / f"{P6}_film" / new[k] / "series.csv")
        n = min(len(a), len(b))
        dv = max(abs(float(x["V_km3"]) - float(y["V_km3"])) for x, y in zip(a[:n], b[:n]))
        da = max(abs(float(x["A1_km2"]) - float(y["A1_km2"])) for x, y in zip(a[:n], b[:n]))
        with np.load(IGM / "runs" / f"{P6}_film" / new[k] / "fields.npz") as Z:
            ny = [sum(f.startswith(p) for f in Z.files) for p in ("thk_y", "us_y", "vs_y")]
        rows.append((k, new[k], f"{len(a)} {len(b)}", f"{dv:.2e}", f"{da:.4f}", " ".join(map(str, ny))))
    write_csv(RES / f"{P6}_film_check.csv", ["run", "rerun", "saves first rerun", "max |dV| km3", "max |dA1| km2",
                                             "yearly thk us vs"], rows)
    for x in rows:
        print(*x, sep=" | ")


# --- start states (s03) -----------------------------------------------------------------------------------------------

def starts():
    out = IGM / START_DIR
    out.mkdir(parents=True, exist_ok=True)
    S = np.load(INP / STATE5)
    D = read_csv(RES / POSTERIOR)
    assert [r["sample"] for r in D] == list(S["sample"]), "draw order of STATE5 differs from POSTERIOR"
    basin = np.load(INP / "igm_static.npz")["posbal"] == 1
    for d in S["draw"]:
        np.savez_compressed(out / f"d{d:03d}.npz", topg=S["topg_dec8"][d], thk_end=S["thk_dec8"][d])
    outside = int(((S["thk_dec8"] > 0) & ~basin[None]).sum())
    with np.load(IGM / f"runs/{P6}_prep/map_u0_21/fields.npz") as F:
        np.savez_compressed(out / "map_u0.npz", topg=F["topg"], thk_end=F["thk_end"])
        s = read_csv(IGM / f"runs/{P6}_prep/map_u0_21/series.csv")
        print(f"map_u0: V 2021 {s[0]['V_km3']} km3, 2026 {F['thk_end'].sum() * 1e-7:.6f} km3, A1 2026 "
              f"{(F['thk_end'] >= 1).sum() * 1e-4:.4f} km2; cells with ice outside the basin "
              f"{int(((F['thk_end'] > 0) & ~basin).sum())}")
    print(f"{START_DIR}: {len(S['draw'])} draws + map_u0; cells with ice outside the basin, all draws: {outside}")


# --- reduction --------------------------------------------------------------------------------------------------------

def first_year(years, x, limit):
    """First year with x < limit (or <= for half-volume, limit given as (v, True)) and no later return; '' if none."""
    v, le = limit if isinstance(limit, tuple) else (limit, False)
    below = x <= v if le else x < v
    if not below[-1]:
        return ""
    k = len(below) - 1
    while k > 0 and below[k - 1]:
        k -= 1
    return int(years[k])


def first_at(years, x, v):
    """First year with x <= v ('' if none): half-volume year (decision 14, no return condition)."""
    k = np.flatnonzero(x <= v)
    return int(years[k[0]]) if len(k) else ""


_PRE = {}


def pre_series(mem):
    """(years 2021-2026, V km3) of the 2021-2026 run of observed weather that leads to a member's start state: the
    F3 run SAMPLE_21 of its draw (hindcast.dec8_run), or P6_prep/map_u0_21 for the u = 0 members."""
    if mem["map_u0"] == "1":
        path = IGM / f"runs/{P6}_prep/map_u0_21/series.csv"
    else:
        if not _PRE:
            import hindcast
            _PRE["run"] = {r["sample"]: hindcast.dec8_run(r) for r in read_csv(RES / POSTERIOR)}
        path = _PRE["run"][mem["sample"]] / "series.csv"
    if path not in _PRE:
        s = read_csv(path)
        _PRE[path] = (np.array([int(r["year"]) for r in s]), np.array([float(r["V_km3"]) for r in s]))
    return _PRE[path]


def member_values(years, V, A, pre):
    """Disappearance at 0.01 and 0.1 km2 and half-volume years of one series (years from 2026); pre the 2021-2026
    series (pre_series): V_2021 its first value; the year with V <= V_2021 / 2 searched from 2022 on through both."""
    out = {"dis_001": first_year(years, A, 0.01), "dis_01": first_year(years, A, 0.1),
           "half_2026": first_at(years[1:], V[1:], V[0] / 2), "half_2021": ""}
    if pre is not None:
        assert pre[0][-1] == years[0] and abs(pre[1][-1] - V[0]) < 1e-6, (pre[0][-1], pre[1][-1], V[0])
        out["half_2021"] = first_at(np.r_[pre[0][1:], years[1:]], np.r_[pre[1][1:], V[1:]], pre[1][0] / 2)
    return out


def reduce(stage):
    M = {m["member"]: m for m in read_csv(RES / f"{P6}_members.csv")}
    runs = {r["name"]: r for r in read_csv(IGM / "logs" / stage / "runs.csv")}    # a rerun line replaces the first
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    rows, ser, names, last = [], [], [], []
    head = ["name", "rc", "end", "t_end", "wall_s", "iterations", "years", "V_2026", "A_2026", "V_end", "A_end",
            "dis_001", "dis_01", "half_2026", "half_2021", "nan", "icefree_bad", "icefree_cells", "icefree_first",
            "icefree_last", "basin_melt_km3"]
    for name in sorted(runs):
        d = IGM / "runs" / stage / name
        row = {"name": name, "rc": runs[name]["rc"], "wall_s": int(runs[name]["end"]) - int(runs[name]["start"])}
        if (d / "track_end.txt").exists():
            e = (d / "track_end.txt").read_text().split()
            row["end"], row["t_end"] = " ".join(e[:-2]), e[-2][2:]
        if (d / "series.csv").exists() and (d / "fields.npz").exists():
            s = read_csv(d / "series.csv")
            Y = np.array([int(r["year"]) for r in s])
            num = lambda c: np.array([float(r[c]) if r[c] != "" else np.nan for r in s])
            V, A = num("V_km3"), num("A1_km2")
            bad = sum(int(np.isnan(num(c)).sum()) for c in ("V_km3", "A01_km2", "A1_km2", "A2_km2"))
            bad += int(sum(1 for r in s[1:] if r["Ba"] != "" and not np.isfinite(float(r["Ba"]))))
            with np.load(d / "fields.npz") as F:
                th, li = F["thk_end"], F["last_ice"]
                bad += int((~np.isfinite(th)).sum()) + sum(int((~np.isfinite(F[k])).sum()) for k in F.files
                                                          if k[:4] in ("thk_", "us_y", "vs_y"))
            mem = M.get(name[3:] if name.startswith("rr_") else name)
            start = np.load(IGM / mem["start"])["thk_end"] if mem else None
            ib = int(((li != -1) & ((li < 2026) | (li > Y[-1]))).sum())
            if start is not None:
                ib += int(((start >= 1.0) & (li < 2026)).sum())
            ib += int(((th >= 1.0) != (li == Y[-1])).sum())
            pre = pre_series(mem) if mem else None
            row.update(iterations=s[-1]["it"], years=len(s) - 1, V_2026=s[0]["V_km3"], A_2026=s[0]["A1_km2"],
                       V_end=s[-1]["V_km3"], A_end=s[-1]["A1_km2"], nan=bad, icefree_bad=ib,
                       icefree_cells=int((li >= 0).sum()), icefree_first=int(li[li >= 0].min()) + 1 if (li >= 0).any()
                       else "", icefree_last=int(li.max()) + 1, **member_values(Y, V, A, pre),
                       basin_melt_km3=f"{sum(float(r['basin_melt_km3'] or 0) for r in s[1:]):.6f}")
            ser += [[name] + [r[c] for c in ("year", "V_km3", "A01_km2", "A1_km2", "A2_km2", "Ba", "Bw", "Bs",
                                             "tongue_m", "v_med", "v_p90", "v_max")] for r in s]
            names.append(name)
            last.append(li)
        rows.append(row)
    write_csv(RES / f"{P6}_runs_{stage}.csv", head, [[r.get(k, "") for k in head] for r in rows])
    with gzip.open(out / f"series_{stage}.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "year", "V_km3", "A01_km2", "A1_km2", "A2_km2", "Ba", "Bw", "Bs", "tongue_m", "v_med",
                    "v_p90", "v_max"])
        w.writerows(ser)
    np.savez_compressed(out / f"icefree_{stage}.npz", name=np.array(names), last_ice=np.array(last, np.int16))
    nb = sum(1 for r in rows if r["rc"] != "0" or "end" not in r)
    print(f"{stage}: {len(rows)} runs reduced, {nb} failed or unfinished; results/{P6}_runs_{stage}.csv, "
          f"results/{P6}/series_{stage}.csv.gz, icefree_{stage}.npz")


def noflow():
    """Reduce the no-flow outputs (s03 results/P6/noflow_K.csv, noflow_K_last_ice.npz) to results/P6_noflow.csv
    (per member: years, end year and reason (no ice: the IGM stop rule of noflow.py; end time: 2100), V and A1 at
    the start and end, disappearance and half-volume years, NaN count) and one series and one ice-free file."""
    M = {m["member"]: m for m in read_csv(RES / f"{P6}_members.csv")}
    out = OUT
    rows, names, last, keep = [], [], [], None
    with gzip.open(out / "noflow_series.csv.gz", "wt", newline="") as g:
        w = csv.writer(g)
        w.writerow(["member", "year", "V_km3", "A01_km2", "A1_km2", "A2_km2", "Ba"])
        K = 0
        while (out / f"noflow_{K}.csv").exists():
            batch = read_csv(STAGES / f"{P6}_noflow_{K}.csv")
            per = {}
            for r in read_csv(out / f"noflow_{K}.csv"):
                per.setdefault(int(r["member"]), []).append(r)
            with np.load(out / f"noflow_{K}_last_ice.npz") as L:
                keep = L["keep"]
                li = L["last_ice"]
            for i, b in enumerate(batch):
                s = per[i]
                Y = np.array([int(r["date"][:4]) for r in s])
                V, A = (np.array([float(r[c]) for r in s]) for c in ("V_km3", "A1_km2"))
                nan = int(np.isnan(V).sum() + np.isnan(A).sum()) + sum(
                    1 for r in s[1:] if r["Ba"] != "" and r["Ba"] != "nan" and not np.isfinite(float(r["Ba"])))
                m = M[b["member"]]
                end = "no ice" if A[-1] == 0 and len(s) > 1 else "end time" if Y[-1] == 2100 else "unfinished"
                bad = int(((li[i] != -1) & ((li[i] < 2026) | (li[i] > Y[-1]))).sum())
                rows.append([b["member"], len(s) - 1, Y[-1], end, s[0]["V_km3"], s[0]["A1_km2"], s[-1]["V_km3"],
                             s[-1]["A1_km2"], *member_values(Y, V, A, pre_series(m)).values(), nan, bad])
                w.writerows([[b["member"], y, r["V_km3"], r["A01_km2"], r["A1_km2"], r["A2_km2"], r["Ba"]]
                             for y, r in zip(Y, s)])
                names.append(b["member"])
            last.append(li)
            K += 1
    write_csv(RES / f"{P6}_noflow.csv", ["member", "years", "end_year", "end", "V_2026", "A_2026", "V_end", "A_end",
                                         "dis_001", "dis_01", "half_2026", "half_2021", "nan", "icefree_bad"], rows)
    np.savez_compressed(out / "noflow_icefree.npz", member=np.array(names), keep=keep,
                        last_ice=np.concatenate(last).astype(np.int16))
    print(f"noflow: {len(rows)} members from {K} files: results/{P6}_noflow.csv, results/{P6}/noflow_*")


# --- gate -------------------------------------------------------------------------------------------------------------

def gate():
    M = read_csv(RES / f"{P6}_members.csv")
    rows = []
    add = lambda n, v, thr, ok: rows.append([n, v, thr, ok])
    run_file = lambda st: RES / f"{P6}_runs_{st}.csv"
    nets = {r["name"]: r for st in (f"{P6}_nets", f"{P6}_nets_fix") if run_file(st).exists()
            for r in read_csv(run_file(st))}                     # a fix stage replaces its runs
    nets = list(nets.values())
    okn = [r for r in nets if r["rc"] == "0" and r["end"] == "end time" and abs(float(r["t_end"]) - START) < 1e-3
           and (IGM / NET_DIR / f"{r['name'][2:]}.npz").exists()]
    add("network trainings ended, network saved", f"{len(okn)} of {len(nets)}", "129", len(okn) == len(nets) == 129)
    P = {r["name"]: r for st in (f"{P6}_proj", f"{P6}_proj_fix") if run_file(st).exists() for r in read_csv(run_file(st))}
    P = list(P.values())
    ended = [r for r in P if r["rc"] == "0" and ((r["end"] == "no ice" and float(r["t_end"]) < END + 1e-3)
                                                 or (r["end"] == "end time" and abs(float(r["t_end"]) - END) < 1e-3))]
    add("projection runs ended (rc 0; no ice, or 2100.75)", f"{len(ended)} of {len(P)} (members {len(M)}); no ice "
        f"{sum(r['end'] == 'no ice' for r in ended)}, 2100 {sum(r['end'] == 'end time' for r in ended)}",
        f"{len(M)}", len(ended) == len(P) == len(M))
    nan = sum(int(r["nan"] or 0) for r in P)
    add("NaN values in series and fields", nan, "0", nan == 0)
    ib = sum(int(r["icefree_bad"] or 1) for r in P)
    n_grid = 0
    for st in (f"{P6}_proj", f"{P6}_proj_fix"):
        if (OUT / f"icefree_{st}.npz").exists():
            with np.load(OUT / f"icefree_{st}.npz") as L:
                n_grid += len(L["name"])
    add("ice-free grids: runs with a grid, inconsistent cells (outside 2026..end, start ice without, end ice "
        "mismatch)", f"{n_grid} grids; {ib} cells", f"{len(M)}; 0", n_grid == len(M) and ib == 0)
    bm = np.array([float(r["basin_melt_km3"] or 0) for r in P])
    add("ice removed by the basin 2026-2100 (F2): largest per run, runs above 0", f"{bm.max():.6f} km3; {(bm > 0).sum()}",
        "reported", "")
    R = {r["name"]: r for r in read_csv(run_file(f"{P6}_rerun"))}
    Pd = {r["name"]: r for r in P}
    same = []
    for n, r in R.items():
        a = Pd[n[3:]]
        same.append(f"{n[3:]}: {a['dis_001']}|{r['dis_001']} {a['dis_01']}|{r['dis_01']} V_end {a['V_end']}|"
                    f"{r['V_end']}")
    ok = all(Pd[n[3:]]["dis_001"] == r["dis_001"] for n, r in R.items()) and len(R) == 4
    add("4 members rerun: disappearance year 0.01 | 0.1 km2 (first | rerun)", "; ".join(same), "same", ok)
    v0 = {}                                                      # iteration-0 surface speeds, m a-1
    for st in (f"{P6}_nets", f"{P6}_nets_fix", f"{P6}_proj", f"{P6}_proj_fix"):
        if (OUT / f"series_{st}.csv.gz").exists():
            with gzip.open(OUT / f"series_{st}.csv.gz", "rt") as f:
                v0.update({r["name"]: (r["v_med"], r["v_p90"]) for r in csv.DictReader(f) if r["year"] == "2026"})
    diff = [abs(float(a) - float(b)) for m in M for a, b in zip(v0[m["member"]], v0["n_" + Path(m["net"]).stem])]
    add("warm start: |member - its network's cold run| of the start-state surface speed median and p90, largest",
        f"{max(diff):.3g} m a-1 over {len(M)} members", "0 (gate 3b)", max(diff) < 1e-3)
    # start state: the spread of the disappearance year with the start area across the draws (median over each
    # draw's 78 members); A1 counts every ice cell (thk >= 1 m), the observed 1.021 km2 is the main body (F3)
    Mm = {m["member"]: m for m in M}
    by = {}
    for r in P:
        m = Mm[r["name"]]
        if m["map_u0"] == "0":
            by.setdefault(int(m["draw"]), []).append((float(r["A_2026"]), float(r["dis_001"] or 2101)))
    A = np.array([v[0][0] for v in by.values()])
    Y = np.array([np.median([x[1] for x in v]) for v in by.values()])
    k, c = np.polyfit(A, Y, 1)
    add("start state: draw-median disappearance year (0.01 km2) against the start ice area A1 2026 over the 128 "
        "draws: range (5 / 50 / 95 %) of A, slope, correlation; the line at the observed main body 1.021 km2",
        f"A {A.min():.3f}-{A.max():.3f} ({' / '.join(f'{x:.3f}' for x in np.percentile(A, (5, 50, 95)))}) km2; "
        f"{k:.1f} a per km2; r {np.corrcoef(A, Y)[0, 1]:.2f}; median of draws {np.median(Y):.1f}, line at 1.021 km2 "
        f"{k * 1.021 + c:.1f}", "reported", "")
    if (RES / f"{P6}_noflow.csv").exists():
        N = read_csv(RES / f"{P6}_noflow.csv")
        okf = [r for r in N if (r["end"] == "end time" and int(r["years"]) == 74) or
               (r["end"] == "no ice" and int(r["years"]) == int(r["end_year"]) - 2026)]
        add("no-flow members ended (no ice by the IGM stop rule, or 2100)", f"{len(okf)} of {len(N)}; no ice "
            f"{sum(r['end'] == 'no ice' for r in okf)}, 2100 {sum(r['end'] == 'end time' for r in okf)}", "9984",
            len(okf) == len(N) == 9984)
        add("no-flow NaN, ice-free grid inconsistencies", f"{sum(int(r['nan']) for r in N)}, "
            f"{sum(int(r['icefree_bad']) for r in N)}", "0, 0",
            sum(int(r["nan"]) + int(r["icefree_bad"]) for r in N) == 0)
    write_csv(RES / f"{P6}_gates.csv", ["gate", "value", "threshold", "pass"], rows)
    for r in rows:
        print(" | ".join(map(str, r)))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "reduce":
        reduce(sys.argv[2])
    else:
        {"pack": pack, "members": members, "starts": starts, "noflow": noflow, "gate": gate, "film": film,
         "filmcheck": filmcheck}[cmd]()
