"""Collect the fixed ensemble (DESIGN.md step 7, fix round: F3 start states, F5 hindcast, F6 projections) into the
output files of DESIGN.md (Outputs) and the small tables the figures read. File prefix f7; the first ensemble's version
of this script (prefix step7, inputs step6_*, posterior_128.csv, runs_s4_rounds.tar.gz, runs_s5.tar.gz) is in the git
history, its outputs in results/first_ensemble/ and s02 results/step7/.

collect (s02, agl env with h5netcdf; reads only banked or reduced files):
  results/f6_members.csv, f6_runs_f6_proj.csv, f6_noflow.csv, posterior_f3_128.csv, f5_series_f5_hind.csv,
  f3_weights.csv, f6_film.csv; results/f6/series_f6_proj.csv.gz, noflow_series.csv.gz, icefree_f6_proj.npz,
  noflow_icefree.npz (projection.py); runs_f3.tar.gz (the 2021-2026 run SAMPLE_21 of every F3 sample, the f3_r1fix
  rerun where one exists, as hindcast.dec8_run), runs_f6.tar.gz (f6_prep/map_u0_21), runs_f5.tar.gz (the hindcast
  fields of the 128 draws)
  ->
  results/members.csv (also on the Mac) one row per member: draw, sample, u, f, the balance parameters, chain, SSP,
      2026 level, station, seed, film flags, end reason and year, V_2021, V_2026, A_2026 (A1 = area with thk >= 1 m),
      V and A1 at the end, the half-volume years (against V_2026 and V_2021) and the disappearance years (A1 < 0.01
      and < 0.1 km2 with no later return) with flow, and the same without flow (noflow.py twins, which stop by the IGM
      rule at the first 1 October without a cell >= 1 m). Empty year: not by 2100. The years are recomputed from the
      series and must equal the reduction of projection.py (assert).
  results/f7_half2021_samples.csv  every F3 sample (2 rounds x 1024, prior u -1 to 2, f 0.05 to 20): V_2021 (start of its
      2021 run, geometry of u), V_2026, the first year 2022-2026 with V <= V_2021 / 2 under observed weather (empty: not
      by 2026), its weight (f3_weights.csv w_main)
  s02 results/f7/series.nc  dims member (10062), year (2021-2100), draw (128), hindcast_year (1850-2026):
      IGM yearly V, A1, A01, A2, Ba, Bw, Bs, tongue, v_med, v_p90, v_max per member (2021-2025: the draw's 2021-2026
      run of observed weather, the decision-8 path, the same for all members of a draw); the no-flow V, A1, A01, A2,
      Ba (2026-2100; NaN for the 78 u = 0 members); NaN after a run's no-ice stop; the hindcast series per draw.
  s02 results/f7/icefree.nc  icefree_year (member, y, x) int16 = 1 + the last hydrological year a cell held ice
      (thk >= 1 m) at a 1 October save 2026-2100; 2101: ice in 2100; -1 (fill): no ice at any save; the same without
      flow, icefree_year_noflow (on the cells noflow.py computes, -1 elsewhere and for the u = 0 members).
  s02 results/f7/hindcast_fields.nc  thk (draw, date, y, x) float32 at the validation dates of F5 and on 2026-09-30,
      and topg (draw, y, x) of each draw.
summary (s02, after collect; small files for figures.py, rsynced to the Mac):
  results/f7_series_quantiles.csv  per group (SSP x 2026 level; MAP draw u = 0) and year 2021-2100: 5, 17, 50, 83,
      95 % and mean of V and A1, with and without flow (after a no-ice stop V = A1 = 0).
  results/f7_years.csv  quantile table: per year quantity (half-volume against V_2026 and V_2021, disappearance at
      0.01 and 0.1 km2; with flow, without flow, IGM minus no-flow) and group (SSP x level, level pooled over SSPs,
      all): 5, 17, 50, 83, 95 %, mean, share not by 2100, n; for the differences also the share of members whose IGM
      year is earlier, the same, later, and the smallest and largest difference. 'Not by 2100' counts as 2101.
  results/f7_spread.csv  share of the spread per factor: the 9984 members (u of the draw) are a full crossing of
      draw (128) x chain (26) x SSP (3) with one member per cell, so the sum of squares splits exactly (balanced
      three-way analysis of variance, one observation per cell): SSP, chain, draw, the three two-way interactions and
      the three-way interaction, which also holds the weather year order of each member; the draw part splits further
      into 2026 level, station, their interaction, and the draws within a level and station (balance, u, f; 128
      distinct F3 samples). Per year quantity, over all 9984 and over each level alone (without the level row). The
      linear share of u, ln f and V_2026 alone (squared correlation over the draws) is added for the record.
  results/f7_icefree_summary.csv  per group: area where any member has ice at some save 2026-2100, area whose median
      ice-free year is after 2026, share of the 2026 outline (s2_2026) whose median ice-free year is 2026 or earlier
      (no ice at any save in most members), the median and latest of the median ice-free years, the area whose median
      ice-free year is 2041 or later, and the median and largest 90 % range (95 minus 5 %), years.
  results/f7_icefree_maps.npz  per group (SSP x level, level pooled): the 5, 50, 95 % of the ice-free year per cell
      over the members of the group (no ice at any save counts as 2026, ice in 2100 as 2101), with and without flow,
      and the share of members with ice on the cell at some save 2026-2100 (key ice2026); x, y and the 2021 surface of
      inputs/igm_static.npz for the map background.
usage (s02):  cd /data/befahl/projects/AGL/basodino/igm && source ~/miniconda3/bin/activate agl
              nrun nice -n 10 python collect.py collect && nrun nice -n 10 python collect.py summary
"""
import csv
import gzip
import io
import sys
import tarfile
from pathlib import Path

import h5netcdf
import h5py
import numpy as np

from projection import first_at, first_year

IGM = Path(__file__).resolve().parent
RES, OUT = IGM / "results", IGM / "results/f7"
P7, POSTERIOR = "f7", "posterior_f3_128.csv"
YEARS = np.arange(2021, 2101)
HYEARS = np.arange(1850, 2027)
SER = ("V_km3", "A1_km2", "A01_km2", "A2_km2", "Ba", "Bw", "Bs", "tongue_m", "v_med", "v_p90", "v_max")
NF = ("V_km3", "A1_km2", "A01_km2", "A2_km2", "Ba")
HSER = ("V_km3", "A1_km2", "A1_in1850_km2", "A1_out1850_km2", "Ba", "Bw", "Bs", "tongue_m", "length_m", "v_med",
        "v_p90", "its_median")
NAMES = {"V_km3": "V", "A1_km2": "A1", "A01_km2": "A01", "A2_km2": "A2", "tongue_m": "tongue", "length_m": "length",
         "A1_in1850_km2": "A1_in1850", "A1_out1850_km2": "A1_out1850"}
UNITS = {"V": "km3", "A1": "km2", "A01": "km2", "A2": "km2", "A1_in1850": "km2", "A1_out1850": "km2",
         "Ba": "m w.e.", "Bw": "m w.e.", "Bs": "m w.e.", "tongue": "m", "length": "m", "v_med": "m a-1",
         "v_p90": "m a-1", "v_max": "m a-1", "its_median": "m a-1"}
LONG = {"V": "ice volume", "A1": "ice area, thickness >= 1 m", "A01": "area, thickness >= 0.1 m",
        "A2": "area, thickness >= 2 m", "Ba": "glacier-wide annual balance", "Bw": "glacier-wide winter balance "
        "(1 October to 30 April)", "Bs": "glacier-wide summer balance Ba - Bw", "tongue": "lowest ice surface of the "
        "largest ice body", "length": "length along the flowline", "v_med": "median surface speed over the ice",
        "v_p90": "90th percentile of the surface speed over the ice", "v_max": "largest surface speed over the ice",
        "its_median": "median speed on the 96 ITS_LIVE pixels", "A1_in1850": "ice area inside SGI1850",
        "A1_out1850": "ice area outside SGI1850"}
HIND_DATES = ("19071001", "19290813", "19491007", "19610919", "19710903", "19731001", "19850910", "19910918",
              "20020916", "20080829", "20101001", "20120901", "20130821", "20180901", "20181001", "20210930",
              "20240823", "20250907", "20260927")
QS = (5, 17, 50, 83, 95)
STR = h5py.string_dtype()


def read_csv(path):
    return list(csv.DictReader(open(path)))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def num(v):
    return float(v) if v not in ("", "nan") else np.nan


def tar_members(path, want):
    """{member path: bytes} for the wanted members of a .tar.gz (one pass); want a set of paths or a predicate."""
    out = {}
    with tarfile.open(path, "r:gz") as t:
        for m in t:
            if (want(m.name) if callable(want) else m.name in want):
                out[m.name] = t.extractfile(m).read()
                if not callable(want) and len(out) == len(want):
                    break
    missing = set() if callable(want) else set(want) - set(out)
    assert not missing, f"{path}: missing {sorted(missing)[:5]}"
    return out


def years_of(Y, V, A, pre):
    """Disappearance and half-volume years of one series from 2026 (projection.member_values without its check)."""
    out = {"dis_001": first_year(Y, A, 0.01), "dis_01": first_year(Y, A, 0.1),
           "half_2026": first_at(Y[1:], V[1:], V[0] / 2)}
    out["half_2021"] = first_at(np.r_[pre[0][1:], Y[1:]], np.r_[pre[1][1:], V[1:]], pre[1][0] / 2)
    return out


def read_series(b):
    s = list(csv.DictReader(io.StringIO(b.decode())))
    return {c: np.array([num(r[c]) for r in s]) for c in ("year",) + SER}


# --- collect ----------------------------------------------------------------------------------------------------------

def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    M = read_csv(RES / "f6_members.csv")
    names = [m["member"] for m in M]
    ix = {n: i for i, n in enumerate(names)}
    P = {r["draw"]: r for r in read_csv(RES / POSTERIOR)}
    R = {r["name"]: r for r in read_csv(RES / "f6_runs_f6_proj.csv")}
    N = {r["member"]: r for r in read_csv(RES / "f6_noflow.csv")}
    film = {r["quantity"]: r["value"] for r in read_csv(RES / "f6_film.csv")}["film member"]
    nm, ny = len(M), len(YEARS)

    # the 2021-2026 runs (decision-8 path) of every F3 sample, from the banked tars; the f3_r1fix rerun replaces its run
    raw = tar_members(IGM / "runs_f3.tar.gz", lambda n: n.endswith("_21/series.csv") and n.count("/") == 3
                      and n.split("/")[1] in ("f3_r1", "f3_r2", "f3_r1fix"))
    pre = {}
    for k in sorted(raw, key=lambda k: "f3_r1fix" in k):              # fix runs last, so they replace
        pre[k.split("/")[2]] = read_series(raw[k])
    fixed = sorted(k.split("/")[2] for k in raw if "/f3_r1fix/" in k)
    pre["map_u0_21"] = read_series(tar_members(IGM / "runs_f6.tar.gz", {"runs/f6_prep/map_u0_21/series.csv"})
                                   ["runs/f6_prep/map_u0_21/series.csv"])
    pre_of = lambda m: pre["map_u0_21" if m["map_u0"] == "1" else m["sample"] + "_21"]
    W = {r["name"]: r for r in read_csv(RES / "f3_weights.csv")}
    rows = []
    for n, w in sorted(W.items()):
        p = pre[n + "_21"]
        assert list(p["year"]) == list(range(2021, 2027)), n
        h = first_at(p["year"][1:], p["V_km3"][1:], p["V_km3"][0] / 2)
        rows.append([n, w["round"], w["u"], w["f"], f"{p['V_km3'][0]:.6f}", f"{p['V_km3'][-1]:.6f}", h, w["w_main"]])
    write_csv(RES / f"{P7}_half2021_samples.csv", ["sample", "round", "u", "f", "V_2021", "V_2026",
                                                   "half_2021_by_2026", "weight"], rows)
    print(f"results/{P7}_half2021_samples.csv: {len(rows)} samples (f3_r1fix replaces {', '.join(fixed)})")

    # IGM series
    S = {c: np.full((nm, ny), np.nan) for c in SER}          # float64 here, float32 in the file
    for m in M:
        p, i = pre_of(m), ix[m["member"]]
        assert list(p["year"]) == list(range(2021, 2027))
        for c in SER:
            S[c][i, :5] = p[c][:5]
    with gzip.open(RES / "f6/series_f6_proj.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            i, j = ix[r["name"]], int(r["year"]) - 2021
            for c in SER:
                S[c][i, j] = num(r[c])
    # no-flow series (2026 to the no-ice stop), NaN for the u = 0 members and after the stop
    NS = {c: np.full((nm, ny), np.nan) for c in NF}
    with gzip.open(RES / "f6/noflow_series.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            i, j = ix[r["member"]], int(r["year"]) - 2021
            for c in NF:
                NS[c][i, j] = num(r[c])
    for m in M:
        if m["map_u0"] == "0":
            i = ix[m["member"]]
            for c in NF:
                NS[c][i, :5] = S[c][i, :5]

    # members.csv
    head = ["member", "draw", "sample", "u", "f", "Fm", "c_prec", "dT", "dg", "b", "dT_pre", "chain", "ssp", "level",
            "station", "seed", "film", "film_f6", "map_u0", "end", "end_year", "V_2021", "V_2026", "A_2026", "V_end",
            "A_end", "half_2026", "half_2021", "dis_001", "dis_01", "end_year_noflow", "V_end_noflow", "A_end_noflow",
            "half_2026_noflow", "half_2021_noflow", "dis_001_noflow", "dis_01_noflow"]
    rows, check = [], 0
    for m in M:
        i, r, p = ix[m["member"]], R[m["member"]], P[m["draw"]]
        assert p["sample"] == m["sample"]
        Y = np.arange(2026, int(float(r["t_end"])) + 1)
        V, A = S["V_km3"][i, 5:5 + len(Y)].astype(float), S["A1_km2"][i, 5:5 + len(Y)].astype(float)
        pr = pre_of(m)
        y = years_of(Y, V, A, (pr["year"], pr["V_km3"]))
        check += sum(str(y[k]) != r[k] for k in ("dis_001", "dis_01", "half_2026", "half_2021"))
        row = [m["member"], m["draw"], m["sample"], m["u"], m["f"], p["Fm"], p["c_prec"], p["dT"], p["dg"], p["b"],
               p["dT_pre"], m["chain"], m["ssp"], m["level"], m["station"], m["seed"], int(m["member"] == film),
               m["film"], m["map_u0"], r["end"], Y[-1], f"{pr['V_km3'][0]:.6f}", r["V_2026"], r["A_2026"], r["V_end"],
               r["A_end"], r["half_2026"], r["half_2021"], r["dis_001"], r["dis_01"]]
        if m["map_u0"] == "0":
            n = N[m["member"]]
            Yn = np.arange(2026, int(n["end_year"]) + 1)
            Vn, An = NS["V_km3"][i, 5:5 + len(Yn)].astype(float), NS["A1_km2"][i, 5:5 + len(Yn)].astype(float)
            assert np.isfinite(Vn).all() and np.isnan(NS["V_km3"][i, 5 + len(Yn):]).all()
            yn = years_of(Yn, Vn, An, (pr["year"], pr["V_km3"]))
            check += sum(str(yn[k]) != n[k] for k in ("dis_001", "dis_01", "half_2026", "half_2021"))
            row += [n["end_year"], n["V_end"], n["A_end"], n["half_2026"], n["half_2021"], n["dis_001"], n["dis_01"]]
        else:
            row += [""] * 7
        rows.append(row)
    write_csv(RES / "members.csv", head, rows)
    write_csv(OUT / "members.csv", head, rows)
    print(f"results/members.csv: {len(rows)} members; years recomputed from the series against the reduce: "
          f"{check} mismatches")
    assert check == 0

    # series.nc
    H = {}
    for r in read_csv(RES / "f5_series_f5_hind.csv"):
        H.setdefault(int(r["name"][1:]), []).append(r)
    draws = np.array(sorted(H))
    for d in draws:
        assert [int(r["year"]) for r in H[d]] == list(HYEARS)
    with h5netcdf.File(OUT / "series.nc", "w") as f:
        f.attrs.update(title="Basodino IGM ensemble after the fix round, yearly series (DESIGN.md step 7)",
                       source="igm/collect.py", conventions="values at the 1 October save that ends hydrological year "
                       "`year`; NaN after a run's no-ice stop (DESIGN.md decision 14) and where a member has no such run")
        f.dimensions.update(member=nm, year=ny, draw=len(draws), hindcast_year=len(HYEARS))
        f.create_variable("member", ("member",), data=np.array(names, dtype=object), dtype=STR)
        f.create_variable("year", ("year",), data=YEARS.astype(np.int16))
        f.create_variable("draw", ("draw",), data=draws.astype(np.int16))
        f.create_variable("hindcast_year", ("hindcast_year",), data=HYEARS.astype(np.int16))
        for c in ("draw", "chain", "ssp", "level", "station", "u", "f", "seed", "map_u0"):
            vals = [m[c] for m in M]
            if c in ("chain", "ssp", "level", "station"):
                f.create_variable(f"member_{c}", ("member",), data=np.array(vals, dtype=object), dtype=STR)
            else:
                f.create_variable(f"member_{c}", ("member",), data=np.array(vals, float if c in "uf" else int))
        kw = dict(compression="gzip", compression_opts=4, shuffle=True)
        for c in SER:
            v = f.create_variable(NAMES.get(c, c), ("member", "year"), data=S[c].astype(np.float32),
                                  chunks=(1024, ny), **kw)
            v.attrs.update(units=UNITS[NAMES.get(c, c)], long_name=LONG[NAMES.get(c, c)] + ", with ice flow; "
                           "2021-2025 the draw's 2021-2026 run under observed weather")
        for c in NF:
            v = f.create_variable(NAMES.get(c, c) + "_noflow", ("member", "year"), data=NS[c].astype(np.float32),
                                  chunks=(1024, ny), **kw)
            v.attrs.update(units=UNITS[NAMES.get(c, c)], long_name=LONG[NAMES.get(c, c)] + ", without ice flow "
                           "(noflow.py), to the no-ice stop (IGM rule)")
        for c in HSER:
            z = np.array([[num(r[c]) for r in H[d]] for d in draws], np.float32)
            v = f.create_variable("hindcast_" + NAMES.get(c, c), ("draw", "hindcast_year"), data=z, **kw)
            v.attrs.update(units=UNITS[NAMES.get(c, c)], long_name=LONG[NAMES.get(c, c)] + ", hindcast 1850-2026 "
                           "of the draw (F5)")
    print(f"{OUT}/series.nc")

    # icefree.nc
    st = np.load(IGM / "inputs/igm_static.npz")
    x, yv = st["x"].astype(np.float64), st["y"].astype(np.float64)
    with np.load(RES / "f6/icefree_f6_proj.npz") as L:
        pos = {n: k for k, n in enumerate(L["name"])}
        last = L["last_ice"][[pos[n] for n in names]]
    ice = np.where(last >= 0, last + 1, -1).astype(np.int16)
    with np.load(RES / "f6/noflow_icefree.npz") as L:
        keep, nl, nn = L["keep"], L["last_ice"], list(L["member"])
    icen = np.full(last.shape, -1, np.int16)
    for k, n in enumerate(nn):
        g = np.full(keep.shape, -1, np.int16)
        g[keep] = np.where(nl[k] >= 0, nl[k] + 1, -1)
        icen[ix[n]] = g
    with h5netcdf.File(OUT / "icefree.nc", "w") as f:
        f.attrs.update(title="Basodino IGM ensemble after the fix round, ice-free year per cell (DESIGN.md decision "
                       "14)", source="igm/collect.py", crs="EPSG:2056 (LV95), cell centres, 10 m")
        f.dimensions.update(member=nm, y=len(yv), x=len(x))
        f.create_variable("member", ("member",), data=np.array(names, dtype=object), dtype=STR)
        f.create_variable("x", ("x",), data=x).attrs.update(units="m", long_name="LV95 easting")
        f.create_variable("y", ("y",), data=yv).attrs.update(units="m", long_name="LV95 northing")
        kw = dict(compression="gzip", compression_opts=4, shuffle=True, chunks=(16, len(yv), len(x)), fillvalue=-1)
        f.create_variable("icefree_year", ("member", "y", "x"), data=ice, dtype=np.int16, **kw).attrs.update(
            long_name="1 + last hydrological year with thk >= 1 m at a 1 October save 2026-2100, with ice flow; "
                      "2101: ice in 2100; -1: no ice at any save")
        f.create_variable("icefree_year_noflow", ("member", "y", "x"), data=icen, dtype=np.int16, **kw).attrs.update(
            long_name="the same without ice flow (noflow.py, IGM stop rule), on the cells noflow.py computes; -1 "
                      "elsewhere and for the u = 0 members")
    print(f"{OUT}/icefree.nc: {nm} members, ice-free years {ice[ice > 0].min()}-{ice.max()}")

    # hindcast_fields.nc
    want = {f"runs/f5_hind/h{d:03d}/fields.npz" for d in draws}
    raw = tar_members(IGM / "runs_f5.tar.gz", want)
    T = np.zeros((len(draws), len(HIND_DATES) + 1, len(yv), len(x)), np.float32)
    B = np.zeros((len(draws), len(yv), len(x)), np.float32)
    for k, d in enumerate(draws):
        with np.load(io.BytesIO(raw[f"runs/f5_hind/h{d:03d}/fields.npz"])) as F:
            for j, dt in enumerate(HIND_DATES):
                T[k, j] = F[f"thk_{dt}"]
            T[k, -1] = F["thk_end"]
            B[k] = F["topg"]
    dates = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in HIND_DATES] + ["2026-09-30"]
    with h5netcdf.File(OUT / "hindcast_fields.nc", "w") as f:
        f.attrs.update(title="Basodino IGM hindcast 1850-2026 (F5): ice thickness at the validation dates",
                       source="igm/collect.py from runs_f5.tar.gz", crs="EPSG:2056 (LV95), cell centres, 10 m")
        f.dimensions.update(draw=len(draws), date=len(dates), y=len(yv), x=len(x))
        f.create_variable("draw", ("draw",), data=draws.astype(np.int16))
        f.create_variable("date", ("date",), data=np.array(dates, dtype=object), dtype=STR)
        f.create_variable("x", ("x",), data=x).attrs.update(units="m", long_name="LV95 easting")
        f.create_variable("y", ("y",), data=yv).attrs.update(units="m", long_name="LV95 northing")
        kw = dict(compression="gzip", compression_opts=4, shuffle=True)
        f.create_variable("thk", ("draw", "date", "y", "x"), data=T, chunks=(1, 1, len(yv), len(x)), **kw).attrs.update(
            units="m", long_name="ice thickness (track.py: on 1 October the save, else the year's flow change spread "
                                 "evenly plus the balance since 1 October / 0.91)")
        f.create_variable("topg", ("draw", "y", "x"), data=B, chunks=(1, len(yv), len(x)), **kw).attrs.update(
            units="m", long_name="bed of the draw (decision 8)")
    print(f"{OUT}/hindcast_fields.nc: {len(draws)} draws x {len(dates)} dates")


# --- summary ----------------------------------------------------------------------------------------------------------

def q(z, axis=0):
    return np.percentile(z, QS, axis=axis)


def summary():
    M = read_csv(RES / "members.csv")
    with h5netcdf.File(OUT / "series.nc", "r") as f:
        S = {k: f[k][:] for k in ("V", "A1", "V_noflow", "A1_noflow")}
    groups = [(s, l) for l in ("anchored", "native") for s in ("SSP1-2.6", "SSP2-4.5", "SSP5-8.5")]
    sel = lambda s, l: np.array([m["ssp"] == s and m["level"] == l and m["map_u0"] == "0" for m in M])
    u0 = np.array([m["map_u0"] == "1" for m in M])
    rows = []
    for (s, l), mask in [((s, l), sel(s, l)) for s, l in groups] + [(("all SSPs", "MAP draw u = 0"), u0)]:
        for v in ("V", "A1", "V_noflow", "A1_noflow"):
            z = np.nan_to_num(S[v][mask].astype(float), nan=0.0)
            if v.endswith("noflow") and l == "MAP draw u = 0":
                continue
            qq, mean = q(z), z.mean(0)
            for j, y in enumerate(YEARS):
                rows.append([s, l, v, y] + [f"{x:.6g}" for x in qq[:, j]] + [f"{mean[j]:.6g}", int(mask.sum())])
    write_csv(RES / f"{P7}_series_quantiles.csv", ["ssp", "level", "variable", "year"] + [f"p{p}" for p in QS] +
              ["mean", "n"], rows)

    # quantile table of the years
    Q = ("half_2026", "half_2021", "dis_001", "dis_01")
    yr = lambda m, k: int(m[k]) if m[k] else 2101
    rows = []
    ugroups = [(s, l) for s, l in groups] + [("all SSPs", l) for l in ("anchored", "native", "both")]
    for s, l in ugroups:
        G = [m for m in M if m["map_u0"] == "0" and l in (m["level"], "both") and (s == "all SSPs" or m["ssp"] == s)]
        for k in Q:
            for var, tag in ((k, "with flow"), (k + "_noflow", "without flow")):
                z = np.array([yr(m, var) for m in G])
                rows.append([tag, k, s, l, *q(z), f"{z.mean():.2f}", f"{(z == 2101).mean():.4f}", len(z)] + [""] * 5)
            z = np.array([yr(m, k) - yr(m, k + "_noflow") for m in G])
            rows.append(["IGM minus no-flow", k, s, l, *q(z), f"{z.mean():.2f}", "", len(z), f"{(z < 0).mean():.4f}",
                         f"{(z == 0).mean():.4f}", f"{(z > 0).mean():.4f}", z.min(), z.max()])
    write_csv(RES / f"{P7}_years.csv", ["run", "quantity", "ssp", "level"] + [f"p{p}" for p in QS] +
              ["mean", "share_not_by_2100", "n", "share_earlier", "share_same", "share_later", "min", "max"], rows)

    # share of the spread (balanced three-way crossing draw x chain x SSP)
    rows = []
    D = [m for m in M if m["map_u0"] == "0"]
    draws = sorted({int(m["draw"]) for m in D})
    chains = sorted({m["chain"] for m in D})
    ssps = sorted({m["ssp"] for m in D})
    attr = {int(m["draw"]): (m["level"], m["station"], float(m["u"]), np.log(float(m["f"])), float(m["V_2026"]))
            for m in D}
    for scope in ("all", "anchored", "native"):
        dd = [d for d in draws if scope == "all" or attr[d][0] == scope]
        di = {d: i for i, d in enumerate(dd)}
        for k in Q:
            Y = np.full((len(dd), len(chains), len(ssps)), np.nan)
            for m in D:
                if int(m["draw"]) in di:
                    Y[di[int(m["draw"])], chains.index(m["chain"]), ssps.index(m["ssp"])] = yr(m, k)
            assert np.isfinite(Y).all()
            g = Y.mean()
            a, b, c = Y.mean((1, 2)) - g, Y.mean((0, 2)) - g, Y.mean((0, 1)) - g
            ab = Y.mean(2) - g - a[:, None] - b[None]
            ac = Y.mean(1) - g - a[:, None] - c[None]
            bc = Y.mean(0) - g - b[:, None] - c[None]
            abc = Y - g - a[:, None, None] - b[None, :, None] - c[None, None] - ab[..., None] - ac[:, None] - bc[None]
            n1, n2, n3 = Y.shape
            tot = ((Y - g) ** 2).sum()
            ss = {"SSP": n1 * n2 * (c ** 2).sum(), "chain": n1 * n3 * (b ** 2).sum(), "draw": n2 * n3 * (a ** 2).sum(),
                  "draw x chain": n3 * (ab ** 2).sum(), "draw x SSP": n2 * (ac ** 2).sum(),
                  "chain x SSP": n1 * (bc ** 2).sum(), "draw x chain x SSP and weather order": (abc ** 2).sum()}
            assert abs(sum(ss.values()) - tot) < 1e-6 * tot
            # the draw part: level, station, level x station, draws within (balance, u, f)
            lev = np.array([attr[d][0] for d in dd])
            sta = np.array([attr[d][1] for d in dd])
            sub = {}
            for name, key in (("2026 level", lev), ("station", sta)):
                if len(set(key)) > 1:
                    sub[name] = sum(n2 * n3 * (key == v).sum() * (a[key == v].mean()) ** 2 for v in set(key))
            cell = np.array([f"{x}|{y}" for x, y in zip(lev, sta)])
            cm = {v: a[cell == v].mean() for v in set(cell)}
            between = sum(n2 * n3 * (cell == v).sum() * cm[v] ** 2 for v in cm)
            if "2026 level" in sub:
                sub["2026 level x station"] = between - sum(sub.values())
            sub["draws within a level and station (balance, u, f)"] = ss["draw"] - between
            rec = {name: np.corrcoef([attr[d][j] for d in dd], a)[0, 1] ** 2 * ss["draw"]
                   for j, name in ((2, "u"), (3, "ln f"), (4, "V_2026"))}
            for name, v in list(ss.items()):
                rows.append([scope, k, name, f"{v / tot:.4f}", f"{np.sqrt(v / Y.size):.3f}", f"{Y.std():.3f}",
                             Y.size])
                if name == "draw":
                    for sn, sv in sub.items():
                        rows.append([scope, k, "  of which " + sn, f"{sv / tot:.4f}", f"{np.sqrt(max(sv, 0) / Y.size):.3f}",
                                     f"{Y.std():.3f}", Y.size])
                    for name, v in rec.items():
                        rows.append([scope, k, f"  for the record: {name} alone, linear (squared correlation over the "
                                     "draws)", f"{v / tot:.4f}", "", f"{Y.std():.3f}", Y.size])
    write_csv(RES / f"{P7}_spread.csv", ["members", "quantity", "factor", "share_of_variance", "rms_years",
                                         "sd_years", "n"], rows)

    # ice-free maps
    with h5netcdf.File(OUT / "icefree.nc", "r") as f:
        ice = f["icefree_year"][:]
        icen = f["icefree_year_noflow"][:]
    out = {}
    lg = [(s, l) for s, l in groups] + [("all SSPs", l) for l in ("anchored", "native")]
    for s, l in lg:
        mask = np.array([m["map_u0"] == "0" and m["level"] == l and (s == "all SSPs" or m["ssp"] == s) for m in M])
        key = f"{s}|{l}"
        z = ice[mask].astype(np.float32)
        z[z < 0] = 2026
        out[key + "|flow"] = np.percentile(z, (5, 50, 95), axis=0).astype(np.float32)
        out[key + "|ice2026"] = (ice[mask] >= 2027).mean(0).astype(np.float32)    # ice at some save from 2026
        zn = icen[mask].astype(np.float32)
        zn[zn < 0] = 2026
        out[key + "|noflow"] = np.percentile(zn, (5, 50, 95), axis=0).astype(np.float32)
    st = np.load(IGM / "inputs/igm_static.npz")
    np.savez_compressed(RES / f"{P7}_icefree_maps.npz", x=st["x"], y=st["y"], usurf_2021=st["usurf_2021"], **out)
    obs = st["s2_2026"] > 0                                      # our outline of 27 September 2026, 10 m cells
    rows = []
    for s, l in lg:
        p5, p50, p95 = out[f"{s}|{l}|flow"]
        any26 = out[f"{s}|{l}|ice2026"] > 0
        held = any26 & (p50 >= 2027)
        rows.append([s, l, f"{any26.sum() / 1e4:.4f}", f"{held.sum() / 1e4:.4f}",
                     f"{(obs & ~held).sum() / obs.sum():.3f}", f"{np.median(p50[held]):.0f}" if held.any() else "",
                     f"{p50[held].max():.0f}" if held.any() else "", f"{((p50 >= 2041) & any26).sum() / 1e4:.4f}",
                     f"{np.median((p95 - p5)[any26]):.0f}", f"{(p95 - p5)[any26].max():.0f}"])
    write_csv(RES / f"{P7}_icefree_summary.csv", ["ssp", "level", "area_any_member_ice_2026_2100_km2",
                                                  "area_median_icefree_after_2026_km2",
                                                  "share_of_2026_outline_median_icefree_by_2026",
                                                  "median_icefree_year_median_over_those_cells",
                                                  "latest_median_icefree_year", "area_median_icefree_2041_or_later_km2",
                                                  "range_90_percent_median_years", "range_90_percent_max_years"], rows)
    print(f"results/{P7}_series_quantiles.csv, {P7}_years.csv, {P7}_spread.csv, {P7}_icefree_maps.npz, "
          f"{P7}_icefree_summary.csv")


if __name__ == "__main__":
    {"collect": collect, "summary": summary}[sys.argv[1]]()
