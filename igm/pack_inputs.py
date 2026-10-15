"""Pack the inputs of the IGM runs (DESIGN.md step 3) into two plain numpy files, read by the user modules on s03 and
by noflow.py. No new numbers: everything is read from the step-1 outputs.

igm/inputs/igm_static.npz   the 10 m lattice of inputs.py turned to IGM's orientation (rows south to north, y
                            ascending; x, y the cell centres, m LV95), float32 unless noted:
                              usurf_1994, usurf_2021, mask_1994, mask_2021 (uint8), bed_grab, sigma_plus, sigma_minus,
                              taper (decision 8; NaN kept as in inputs.py), posbal (uint8, decision 2), D (redist.py),
                              ipot (365, y, x; day 0 = 1 October, ipot.py), the outlines sgi1850, sgi1931, sgi1973,
                              sgi2010, sgi2016, tlm2024, s2_2026, s2_2026_body (uint8), itslive_pixel (int16, -1 off)
igm/inputs/forcing_obs.npz  the observed OTL weather of weather.py: hydro_year (176,), t (176, 365) degC, p (176, 365)
                            m w.e. d-1 (P_mm / 1000), gm (365,) K m-1 = the monthly Robiei-Locarno gradient of
                            redist_fit.csv on each day of the hydrological year (as calib_smb.py prepare)
usage (s02, agl env):  python igm/pack_inputs.py
"""
import csv

import numpy as np
import rasterio

from inputs import DX, E0, N0, NX, NY, OUT, RES, read

GRIDS = ("usurf_1994", "usurf_2021", "mask_1994", "mask_2021", "bed_grab", "sigma_plus", "sigma_minus", "taper",
         "posbal", "D", "sgi1850", "sgi1931", "sgi1973", "sgi2010", "sgi2016", "tlm2024", "s2_2026", "s2_2026_body",
         "itslive_pixel")


def main():
    g = {}
    for name in GRIDS:
        z = read(name)[::-1]                                     # rows north to south -> y ascending
        g[name] = z.astype(np.float32) if z.dtype.kind == "f" else z
    with rasterio.open(OUT / "ipot.tif") as r:
        g["ipot"] = r.read()[:, ::-1].astype(np.float32)
    assert g["ipot"].shape == (365, NY, NX) and np.isfinite(g["ipot"]).all() and np.isfinite(g["D"]).all()
    g["x"] = (E0 + DX * (np.arange(NX) + 0.5)).astype(np.float32)
    g["y"] = (N0 + DX * (np.arange(NY) + 0.5)).astype(np.float32)
    np.savez(OUT / "igm_static.npz", **g)

    rows = list(csv.DictReader(open(OUT / "weather_otl_daily.csv")))
    hy = np.array([int(r["hydro_year"]) for r in rows]).reshape(-1, 365)[:, 0]
    t = np.array([float(r["T_degC"]) for r in rows]).reshape(-1, 365)
    p = np.array([float(r["P_mm"]) for r in rows]).reshape(-1, 365) / 1e3
    month = np.array([int(r["date"][5:7]) for r in rows[:365]])
    fit = dict(csv.reader(open(RES / "redist_fit.csv")))
    gm = np.array(fit["g_m Oct..Sep, K per 100 m"].split(), float)[(month - 10) % 12] / 100
    np.savez(OUT / "forcing_obs.npz", hydro_year=hy, t=t, p=p, gm=gm)
    print(f"igm_static.npz: {len(g)} arrays, ipot {g['ipot'].nbytes / 1e6:.0f} MB; forcing_obs.npz: hydrological years "
          f"{hy[0]}-{hy[-1]}, gm {gm.min() * 100:.3f} to {gm.max() * 100:.3f} K per 100 m")


if __name__ == "__main__":
    main()
