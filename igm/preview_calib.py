"""Preview of the balance calibration (step 2): one PNG with six panels from the outputs of calib_smb.py.

Panels: glacier-wide winter and annual balance of the observation periods (not fitted) against the 90 % band of the
kept draws; band winter balance (term a), band annual balance minus the period mean (term b) and stake balance minus
the year mean (term c), observed against the predictive median with the 90 % predictive band; geodetic volume change
per period (term d); the kept draws (radiation factor fixed at 0.012 times the melt factor): with the station
gradients free (third attempt), the melt factor against the temperature-gradient addition dg, coloured by the
precipitation gradient b; in the earlier attempts, the melt factor against the precipitation factor, coloured by the
temperature offset when it is a free parameter.
Fix round F1 (stems calib_f1, calib_f1_stationary): the volume changes 1907-2024, and with dT_pre free the kept draws
as the temperature offset of the years up to 1991 against the one of all years, coloured by the melt factor.
Inputs: igm/results/calib_bands{_tag}.csv, calib_draws{_tag}.csv, or STEM_bands.csv, STEM_draws.csv.
Output: igm/results/preview_calib{_tag}.png, or preview_STEM.png.
usage:  python igm/preview_calib.py [tag | STEM]   (tag e.g. 2par, dT: the earlier attempts; STEM calib_f1...)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent / "results"
UNIT = "metres of water equivalent"


def scatter(ax, d, title):
    lo, hi = d.predictive_p5, d.predictive_p95
    inside = d.inside_90 == 1
    for sel, color, name in ((inside, "tab:blue", "inside"), (~inside, "tab:red", "outside")):
        ax.errorbar(d.observed[sel], d.predictive_p50[sel], yerr=[(d.predictive_p50 - lo)[sel], (hi - d.predictive_p50)[sel]],
                    fmt="o", ms=2.5, lw=0.5, color=color, label=f"{name} the 90 % band ({sel.sum()})")
    v = np.r_[d.observed, lo, hi]
    ax.plot([v.min(), v.max()], [v.min(), v.max()], color="0.4", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"Observed ({UNIT})")
    ax.set_ylabel(f"Modelled, predictive ({UNIT})")
    ax.legend(fontsize=8, frameon=False)


def main(tag=""):
    stem = tag[1:] if tag.startswith("_calib_f1") else None
    b = pd.read_csv(RES / (f"{stem}_bands.csv" if stem else f"calib_bands{tag}.csv"))
    k = pd.read_csv(RES / (f"{stem}_draws.csv" if stem else f"calib_draws{tag}.csv"))
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    ax = axs[0, 0]
    for kind, color, name in (("bw_glacier", "tab:blue", "winter"), ("ba_glacier", "tab:red", "annual")):
        d = b[b.kind == kind]
        y = d.label.astype(int)
        ax.fill_between(y, d.model_p5, d.model_p95, color=color, alpha=0.25, lw=0)
        ax.plot(y, d.model_p50, color=color, lw=1, label=f"Modelled {name} balance, median and 90 % band")
        ax.plot(y, d["map"], color=color, lw=0.8, ls="--", label=f"Most probable draw, {name}")
        ax.plot(y, d.observed, "o", ms=3, color=color, mfc="white", label=f"Observed {name} balance")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_title("Glacier-wide balance, observation periods (not fitted)", fontsize=10)
    ax.set_xlabel("Hydrological year")
    ax.set_ylabel(f"Balance ({UNIT})")
    ax.legend(fontsize=7, frameon=False, loc="lower left")

    scatter(axs[0, 1], b[b.kind == "bw_band"], "Winter balance per 100 metre band")
    scatter(axs[0, 2], b[b.kind == "ba_band"], "Annual balance per band minus the mean of its survey period")
    scatter(axs[1, 0], b[b.kind == "stake"], "Stake balance minus the mean of its year")

    ax = axs[1, 1]
    d = b[b.kind == "dv"].reset_index(drop=True)
    x = np.arange(len(d))
    yr = (pd.to_datetime(d.date1) - pd.to_datetime(d.date0)).dt.days.to_numpy() / 365.25    # per year of the period
    p50 = d.predictive_p50 / yr
    ax.errorbar(x - 0.1, d.observed / yr, yerr=d.sigma_obs / yr, fmt="o", color="black",
                label="Observed, one standard deviation")
    ax.errorbar(x + 0.1, p50, yerr=[p50 - d.predictive_p5 / yr, d.predictive_p95 / yr - p50], fmt="s",
                color="tab:blue", label="Modelled, 90 % predictive band")
    ax.set_xticks(x, [f"{s.split()[0][:4]} to {s.split()[1][:4]}" for s in d.label], fontsize=8, rotation=90)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_title("Geodetic volume change\nper year of each survey period", fontsize=10)
    ax.set_ylabel("Volume change per year (cubic kilometres of ice per year)")
    ax.legend(fontsize=8, frameon=False)

    ax = axs[1, 2]
    ax.set_title("Kept draws\n(radiation factor 0.012 square metres per watt times the melt factor)", fontsize=10)
    ax.set_xlabel("Melt factor (millimetres of ice per day and degree Celsius)")
    if "dT_pre_degC" in k and k.dT_pre_degC.abs().max() > 0:
        sc = ax.scatter(k.dT_degC, k.dT_pre_degC, s=6, c=k.Fm_m_ice_per_day_degC * 1e3, cmap="viridis")
        ax.set_title("Kept draws: the two temperature offsets", fontsize=10)
        ax.set_xlabel("Offset, all years (degrees Celsius)")
        ax.set_ylabel("Added offset, years up to 1991 (degrees Celsius)")
        fig.colorbar(sc, ax=ax, label="Melt factor (millimetres of ice per day and degree Celsius)")
    elif "dg_K_per_km" in k:
        sc = ax.scatter(k.Fm_m_ice_per_day_degC * 1e3, k.dg_K_per_km, s=6, c=k.b_percent_per_100m, cmap="viridis")
        ax.set_ylabel("Added temperature gradient (kelvin per kilometre)")
        fig.colorbar(sc, ax=ax, label="Precipitation gradient (per cent per 100 metres)")
    else:
        with_dt = k.dT_degC.abs().max() > 0
        sc = ax.scatter(k.Fm_m_ice_per_day_degC * 1e3, k.c_prec, s=6, c=k.dT_degC if with_dt else "tab:blue",
                        cmap="viridis" if with_dt else None)
        ax.set_ylabel("Precipitation factor")
        if with_dt:
            fig.colorbar(sc, ax=ax, label="Temperature offset (degrees Celsius)")
    fig.savefig(RES / (f"preview_{stem}.png" if stem else f"preview_calib{tag}.png"), dpi=110)


if __name__ == "__main__":
    main(f"_{sys.argv[1]}" if len(sys.argv) > 1 else "")
