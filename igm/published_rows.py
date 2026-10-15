"""Basodino (RGI60-11.01987) rows of the published per-glacier projections in ../data/published/.

Writes ../data/published/basodino_volume.csv: source, scenario, member, year, volume_km3 (Zekollari et al. 2019:
EURO-CORDEX multi-model mean and its 1-sigma RCM spread; Rounce et al. 2023: one row per GCM, mass / 900 kg m-3,
value at 1 January of the year; OGGM v1.6.1: one row per GCM, 1 January, from oggm2023/basodino_oggm.csv), and
basodino_members.csv: per series V and A at 2021 and 2026, the first year with V <= V_2021/2 and <= V_2026/2, the
first year with V < 1 % of the first value, and (OGGM only) the first year with A < 0.01 km2 and no later return
(empty: not reached in the file). Prints the median and range of these over the GCMs of each scenario.
"""
import csv
from pathlib import Path

import h5py
import numpy as np

PUB = Path(__file__).resolve().parent.parent / "data" / "published"
RGI = "RGI60-11.01987"


def zekollari(kind):
    for rcp in ("rcp26", "rcp45", "rcp85"):
        name = f"ZekollariHussFarinotti_TC2019_volume_{kind}{rcp}.csv"
        path = next(PUB.joinpath("zekollari2019").rglob(name))
        rows = list(csv.reader(path.open()))
        years = [int(y) for y in rows[1][1:]]
        row = next(r for r in rows[2:] if r[0] == "1987")
        yield rcp, years, np.array(row[1:], float)


def rounce():
    for path in sorted(PUB.joinpath("rounce2023").glob("R11_glac_mass_annual_*.nc")):
        scen = path.stem.split("-")[-1]
        with h5py.File(path) as f:
            ids = [s.decode() if isinstance(s, bytes) else s for s in f["RGIId"][:]]
            gcms = [s.decode() if isinstance(s, bytes) else s for s in f["Climate_Model"][:]]
            years = f["year"][:].tolist()
            mass = f["glac_mass_annual"][:, ids.index(RGI), :]
        for gcm, m in zip(gcms, mass):
            if not m[0] > 0:  # GCMs absent from a scenario hold zeros or NaN
                continue
            yield "rounce2023", scen, gcm, years, m / 900 / 1e9, None


def oggm():
    path = PUB / "oggm2023" / "basodino_oggm.csv"
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    for gcm, ssp in dict.fromkeys((r["gcm"], r["ssp"]) for r in rows):
        g = [r for r in rows if (r["gcm"], r["ssp"]) == (gcm, ssp)]
        yield ("oggm2023", ssp, gcm, [int(r["year"]) for r in g], np.array([float(r["volume_km3"]) for r in g]),
               np.array([float(r["area_km2"]) for r in g]))


def first(years, v, cond, after):
    hit = [y for y, x in zip(years, v) if y > after and cond(x)]
    return hit[0] if hit else ""


def gone(years, a):
    """First year with A < 0.01 km2 and no later return."""
    keep = [y for y, x in zip(years, a) if not x < 0.01]
    return "" if keep[-1] == years[-1] else keep[-1] + 1


def num(x):
    return f"{x:.5f}" if isinstance(x, float) else x


series = [("zekollari2019", rcp, "mean", years, v, None) for rcp, years, v in zekollari("")]
series += list(rounce()) + list(oggm())
cols = ["V_2021_km3", "V_2026_km3", "A_2021_km2", "A_2026_km2", "half_of_2021", "half_of_2026",
        "below_1pct_of_first", "area_below_0.01"]
with PUB.joinpath("basodino_members.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "scenario", "member", "first_year", "V_first_km3"] + cols)
    for src, scen, mem, years, v, a in series:
        at, aa = dict(zip(years, v)), dict(zip(years, a)) if a is not None else {}
        w.writerow([src, scen, mem, years[0], num(float(v[0])), num(float(at[2021])), num(float(at[2026])),
                    num(float(aa[2021])) if aa else "", num(float(aa[2026])) if aa else "",
                    first(years, v, lambda x: x <= at[2021] / 2, 2021),
                    first(years, v, lambda x: x <= at[2026] / 2, 2026),
                    first(years, v, lambda x: x < 0.01 * v[0], years[0]),
                    gone(years, a) if aa else ""])

spread = [("zekollari2019", rcp, "sd", years, v, None) for rcp, years, v in zekollari("RCMspread_")]
with PUB.joinpath("basodino_volume.csv").open("w", newline="") as f:
    c = csv.writer(f)
    c.writerow(["source", "scenario", "member", "year", "volume_km3"])
    for src, scen, mem, years, v, a in series + spread:
        c.writerows((src, scen, mem, y, f"{x:.6g}") for y, x in zip(years, v))

rows = list(csv.DictReader(PUB.joinpath("basodino_members.csv").open()))
print("source scenario n |", " | ".join(cols[:2] + cols[4:]))
print("median [min, max] over the GCMs (lower median for even n); a year not reached counts as later (> end)")
for key in dict.fromkeys((r["source"], r["scenario"]) for r in rows):
    g = [r for r in rows if (r["source"], r["scenario"]) == key]
    out = []
    for c in cols[:2] + cols[4:]:
        if not any(r[c] for r in g) and c == "area_below_0.01" and key[0] != "oggm2023":
            continue
        x = np.array([float(r[c]) if r[c] else np.inf for r in g])
        f = (lambda v: f"{v:.4f}") if c.startswith("V") else (lambda v: f"{v:.0f}" if v < np.inf else "> end")
        out.append(f"{f(np.sort(x)[(len(x) - 1) // 2])} [{f(x.min())}, {f(x.max())}]")
    print(*key, len(g), "|", " | ".join(out))
