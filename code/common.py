"""Shared by the task scripts: paths, the constants of the assignment, the two readers, the figure style.

Chain (each script reads the plain files the one before wrote):
  task1_geodetic.py      course grids               -> task1_geodetic.csv, dh and outline figures
  task2_meteo.py         MeteoSwiss station files   -> task2_stations.csv, task2_meteo.csv, meteo figure
  task3_calibration.py   task2_meteo.csv, grids     -> task3_terms.csv, task3_ddf.csv, calibration figure
  task4_reconstruction.py  task3 files, GLAMOS      -> task4_balance.csv, task4_summary.csv, balance figure
  task5_projection.py    task1, task4 files         -> task5_projection.csv, projection figure
  errors.py              grids, task files, GLAMOS  -> errors.csv, one standard deviation of every headline number
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # basodino/
CODE, FIG = ROOT / "code", ROOT / "tex/figures"
COURSE, METEO = ROOT / "data/course", ROOT / "data/meteoswiss"

# assignment constants (tex/main.tex, tab:constants)
RHO_DV, RHO_W, RHO_ICE = 850.0, 1000.0, 900.0          # kg m-3: volume change, water, mean glacier density
T0, LAPSE, TM, ND = 2.0, 6.5e-3, 0.0, 365 / 12          # degC, degC m-1, degC, days per month
YEARS = range(1935, 2026)                               # hydrological years, year y = 1 Oct y-1 to 30 Sep y
CALIB = range(1995, 2022)                               # October 1994 to October 2021, the DEM dates

STATIONS = ("GRH", "ANT", "OTL")                        # homogenised T and P since 1934 within 40 km
STATION = "OTL"                                         # used in the report: best GLAMOS check, south side (task4)


def read_esri_ascii(path):
    """Esri ASCII grid -> (array with NaN for no data, header dict)."""
    header = {}
    with open(path) as f:
        for line in f:
            key, value = line.split()[:2]
            if not key[0].isalpha():      # first row of data
                break
            header[key.lower()] = float(value)
    z = np.loadtxt(path, skiprows=len(header))
    if "nodata_value" in header:
        z[z == header["nodata_value"]] = np.nan
    return z, header


def read_csv(name):
    """A table written by an earlier task script -> list of dicts."""
    with open(CODE / name) as f:
        return list(csv.DictReader(f))


def read_glamos():
    """GLAMOS glaciological balances of Basòdino, fixed dates (hydrological year 1 October to 30 September, winter to
    30 April) -> {hydrological year: (Bw, Bs, Ba) in m w.e.}. The observation-period file is not used: its periods run
    9.5 to 14 months."""
    out = {}
    with open(ROOT / "data/glamos_basodino_massbalance_fixdate.csv", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) > 7 and r[1] == "C14-10":
                out[int(r[4][:4])] = tuple(float(v) / 1000 for v in r[5:8])
    return out


def write_csv(name, header, rows):
    with open(CODE / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([f"{v:.6g}" if isinstance(v, float) else v for v in r] for r in rows)


# --- figures: drawn at print size, graphics only (tex/SIDECARS.md), report palette (maps/style.json)
INK, GREY, RULE, ICE, GLACIER, ACCENT = "#1a1a1a", "#4d4d4d", "#b3b3b3", "#dcebf4", "#027abe", "#c51415"
PALE_BLUE, PALE_RED = "#9cc3e0", "#f2c4c0"                 # light tones of GLACIER and ACCENT, for areas and bars
NUM, LAB, LET = 6.5, 7.0, 8.0                           # pt: numbers, words, panel letters
MM = 1 / 25.4


def figure_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = ROOT / "data/satellite_archive/sentinel1/scripts/fonts"
    for weight in ("Regular", "Semibold"):
        font_manager.fontManager.addfont(fonts / f"SourceSans3-{weight}.ttf")
    plt.rcParams.update({"font.family": "Source Sans 3", "pdf.fonttype": 42, "font.size": NUM, "text.color": INK,
                         "axes.linewidth": 0.4, "axes.edgecolor": GREY, "axes.labelsize": LAB, "axes.labelcolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "xtick.color": GREY, "ytick.color": GREY, "xtick.labelsize": NUM, "ytick.labelsize": NUM,
                         "xtick.major.width": 0.4, "ytick.major.width": 0.4, "xtick.minor.width": 0.3,
                         "ytick.minor.width": 0.3, "xtick.major.size": 2.8, "ytick.major.size": 2.8,
                         "xtick.minor.size": 1.7, "ytick.minor.size": 1.7, "xtick.major.pad": 1.7,
                         "ytick.major.pad": 1.7, "lines.solid_capstyle": "round", "legend.fontsize": LAB,
                         "legend.frameon": False})
    return plt


def panel_letter(fig, ax, letter, width_mm, x_mm=1.2):
    """Panel letter x_mm from the left edge of the figure, level with the top of the axes."""
    fig.text(x_mm / width_mm, ax.get_position().y1, letter, fontsize=LET, fontweight=600, va="top", ha="left",
             color=INK)


def save(fig, name, creator):
    fig.savefig(FIG / f"{name}.pdf", metadata={"Title": name, "Creator": f"code/{creator}"})
