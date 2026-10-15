"""Independent check of step 1 (igm/DESIGN.md): re-reads the s02 inputs and prints one line per check.

  1. lattice: every igm/inputs/*.tif has the transform, shape and CRS of inputs.py
  2. forcing: weather_otl_daily.csv covers 1850-10-01 to 2026-09-30, 365 days per hydrological year, no 29 February,
     no month with constant daily T or one constant P > 0; monthly T and P against OTL homogenised (ths200m0,
     rhs150m0) from 1882-12
  3. hydrological-year convention: day 1 = 1 October in the forcing; ipot.tif has 365 bands (band 1 = 1 October)
  4. D: mean 1 on the 2021 mask
  5. fit test: smb.py on the 2021 surface and mask, hydrological years 2001 and 2022, forcing of weather.py with the
     transfer of decision 5 (g_m from results/redist_fit.csv, b = 2 % per 100 m), D.tif, ipot.tif, melt factors of
     the Aletsch reference (results/smb_reference.csv), c_prec 1, 1.5, 2; glacier-wide Ba against GLAMOS
     2001 +1.02, 2022 -3.51 m w.e. A plausibility test of the inputs fitting together, not a calibration.
Inputs (s02): igm/inputs/*.tif, igm/inputs/weather_otl_daily.csv, raw/climate/obs/ogd-nbcn_otl_m.csv;
Mac copies of igm/results/redist_fit.csv and smb_reference.csv rsynced with the code.
usage (s02, agl env, about 10 s):  cd igm && python checks/step1_check.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

IGM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IGM))
from inputs import NX, NY, TRANSFORM  # noqa: E402
from smb import balance  # noqa: E402

INP, ROOT = IGM / "inputs", IGM.parent

# 1 lattice
bad = []
for f in sorted(INP.glob("*.tif")):
    with rasterio.open(f) as r:
        if r.transform != TRANSFORM or r.shape != (NY, NX) or r.crs.to_epsg() != 2056:
            bad.append(f.name)
print(f"lattice: {len(list(INP.glob('*.tif')))} grids, off-lattice: {bad or 'none'}")

# 2-3 forcing
w = pd.read_csv(INP / "weather_otl_daily.csv", parse_dates=["date"])
d = w["date"]
full = pd.date_range("1850-10-01", "2026-09-30")
full = full[~((full.month == 2) & (full.day == 29))]
per_year = w.groupby("hydro_year").size()
print(f"forcing: {len(w)} rows, first {d.iloc[0].date()}, last {d.iloc[-1].date()}, dates equal to the expected "
      f"365-day calendar: {bool((d.values == full.values).all()) if len(d) == len(full) else False}, "
      f"years {per_year.index.min()}-{per_year.index.max()} all 365 days: {bool((per_year == 365).all())}, "
      f"29 Feb rows {int(((d.dt.month == 2) & (d.dt.day == 29)).sum())}, NaN {int(w[['T_degC', 'P_mm']].isna().sum().sum())}")
first = w.groupby("hydro_year").first()
print(f"hydro-year convention: day 1 on 1 October in every year: "
      f"{bool(((first['date'].dt.month == 10) & (first['date'].dt.day == 1)).all())}, "
      f"hydro_year = calendar year + 1 from October: "
      f"{bool((w['hydro_year'] == d.dt.year + (d.dt.month >= 10)).all())}")
w["ym"] = d.dt.to_period("M")
g = w.groupby("ym")
const_T = int((g["T_degC"].nunique() == 1).sum())
pos = w[w["P_mm"] > 0].groupby("ym")["P_mm"]
even_P = int(((pos.nunique() == 1) & (pos.size() > 1)).sum())
print("  months with one constant P on several wet days:", ", ".join(f"{k} ({pos.get_group(k).size} days, {pos.get_group(k).iloc[0]} mm)" for k in pos.nunique()[(pos.nunique() == 1) & (pos.size() > 1)].index))
print(f"evenly spread months: constant T {const_T}, one constant P value on several wet days {even_P}, "
      f"fully dry months {int((g['P_mm'].sum() == 0).sum())} of {len(g)}")
m = pd.read_csv(ROOT / "raw/climate/obs/ogd-nbcn_otl_m.csv", sep=";", encoding="latin-1")
m["ym"] = pd.to_datetime(m["reference_timestamp"], format="%d.%m.%Y %H:%M").dt.to_period("M")
m = m.set_index("ym")
j = pd.DataFrame({"T": g["T_degC"].mean(), "P": g["P_mm"].sum()}).join(m[["ths200m0", "rhs150m0"]], how="inner")
jt, jp = j.dropna(subset=["ths200m0"]), j.dropna(subset=["rhs150m0"])
print(f"monthly vs homogenised: T {len(jt)} months from {jt.index.min()}, max |diff| {np.abs(jt['T'] - jt['ths200m0']).max():.4f} degC; "
      f"P {len(jp)} months from {jp.index.min()}, max |diff| {np.abs(jp['P'] - jp['rhs150m0']).max():.4f} mm")


def grid(name, band=1):
    with rasterio.open(INP / f"{name}.tif") as r:
        z = r.read() if band is None else r.read(band)
        return np.where(z == r.nodata, np.nan, z.astype(np.float64)) if r.nodata is not None else z.astype(np.float64)


with rasterio.open(INP / "ipot.tif") as r:
    print(f"ipot.tif bands {r.count}")
ipot = grid("ipot", None)
print(f"ipot day 1 (1 Oct) / day 82 (21 Dec) / day 264 (21 Jun) domain means: {np.nanmean(ipot[0]):.1f} / "
      f"{np.nanmean(ipot[81]):.1f} / {np.nanmean(ipot[263]):.1f} W m-2")

# 4 D
mask = grid("mask_2021") == 1
D = grid("D")
print(f"D on the 2021 mask: mean {np.nanmean(D[mask]):.4f}, min {np.nanmin(D[mask]):.3f}, max {np.nanmax(D[mask]):.3f}, NaN {int(np.isnan(D[mask]).sum())}")

# 5 fit test
z = grid("usurf_2021")[mask]
I = ipot[:, mask]
Dm = D[mask]
fit = dict(csv.reader(open(IGM / "results/redist_fit.csv")))
gm = np.array(fit["g_m Oct..Sep, K per 100 m"].split(), float)          # Oct..Sep
ref = pd.read_csv(IGM / "results/smb_reference.csv").iloc[0]
Fm, r_ice = ref["Fm_m_ice_per_day_degC"], ref["r_ice"]
print(f"fit test: {mask.sum()} cells, z {z.min():.0f}-{z.max():.0f} m, Fm {Fm} m ice d-1 degC-1, r_ice {r_ice}")
for cp in (1.0, 1.5, 2.0):
    out = []
    for Y in (2001, 2022):
        y = w[w["hydro_year"] == Y]
        mi = (y["date"].dt.month.values - 10) % 12                        # 0 = October
        T = y["T_degC"].values[:, None] + gm[mi][:, None] * (z[None, :] - 367.0) / 100
        P = cp * y["P_mm"].values[:, None] / 1000 * (1 + 0.02 * (z[None, :] - 367.0) / 100)
        b = balance(T, P, I, Dm, Fm, r_ice)[0].sum(axis=0)
        out.append(f"{Y} {b.mean():+.2f}")
    print(f"  c_prec {cp}: Ba m w.e. {', '.join(out)}  (GLAMOS 2001 +1.02, 2022 -3.51)")
