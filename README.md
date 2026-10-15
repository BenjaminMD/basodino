# Ghiacciaio del Basòdino: mass balance, melt model and volume projection

Code of the report for Applied Glaciology, ETH Zürich, fall semester 2026, assignments part 1.
Benjamin Fahl and Valentin Istomin, in equal parts.

The report is built from `tex/min_si.tex`. Its figures, the data and the PDFs are not in this repository.

## Layout

- `code/`: the chain of the five tasks. Each script reads the plain files the one before wrote and writes plain
  files; `code/common.py` lists the chain. `errors.py` gives one standard deviation of every headline number.
  The other scripts are the satellite record, the 2026 outline, the thickness-map test and the checks in `code/checks/`.
- `igm/`: the ice-flow ensemble with the Instructed Glacier Model 3.2.0: design (`DESIGN.md`), targets, install
  notes, the stage scripts and the IGM user modules, and the results summary (`RESULTS.md`).
- `hero/`, `relief/`: the sky-view factor of the swissALTI3D 0.5 m model for the title pages, on one GPU.
- `fetch/`: downloads of the public data: swisstopo STAC tiles, MeteoSwiss, GLAMOS, the ice-thickness map.
- `tex/`: the LaTeX sources (LuaLaTeX, biber; `latexmk min_si.tex`). `min_si_report.tex` and
  `min_si_supplement.tex` build the two submitted PDFs from the same source.

## Data

Course material (elevation models and masks of 1994 and 2021); MeteoSwiss homogenised monthly series;
GLAMOS mass balances and volume changes; swisstopo swissALTI3D, SWISSIMAGE and swissTLM3D; Copernicus Sentinel-2
through Earth Search; PlanetScope through the Education and Research programme; the Swiss ice-thickness map of
Grab et al. (2021); CH2025 climate scenarios.

## Running the chain

Python 3.12 with numpy, scipy, matplotlib, rasterio and shapely; the data in `data/` as `code/common.py` expects.

    python code/task1_geodetic.py
    python code/task2_meteo.py
    python code/task3_calibration.py
    python code/task4_reconstruction.py
    python code/task5_projection.py
    python code/errors.py
