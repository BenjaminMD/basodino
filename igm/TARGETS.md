# Calibration and validation targets, Ghiacciaio del Basòdino (SGI C14-10)

Every observation a 1850-2100 model of the glacier can be calibrated or tested against: value, date, uncertainty
and file. Compiled 2026-10-06. Paths are relative to `basodino/`; `s02:` means `/data/befahl/projects/AGL/basodino`.
Areas are planimetric in LV95 (EPSG:2056), heights LN02, balances in m w.e. per hydrological year (1 October to
30 September), volume change converted with 850 kg m-3 (GLAMOS and the assignment). Uncertainties are 1 sigma
unless stated. "Online only" means not in `data/` or on the servers.

## 1 Outlines and areas

The "±10 m" column is the area change for a ±10 m shift of the outline (half the area between the +10 m and the
-10 m buffer), the same for every outline. It is our sensitivity number, not a published uncertainty. Elevations
are those of the outline on the swissALTI3D 2022 surface (`data/dem_2m.tif`): ice surface of 2021 on the glacier,
ground of 2019-2020 outside.

| Outline | Date | Area km2 | Uncertainty | ±10 m | Geometry, elevation | File |
|---|---|---|---|---|---|---|
| SGI1850 | 1850 (reconstructed from historical maps and moraines) | 4.008 | not quantified | ±0.091 | 1 polygon; lowest 2069 m (E 2681651, N 1142515) | `data/vectors.gpkg` layer `sgi_1850` |
| SGI1931 | 1929-08-13 (date of the terrestrial image; outline from the first-edition national map, LK50 in Mannerfelt et al. 2022, LK25 in the ReadMe) | 3.202 | map outlines vs reference: median difference 7.9 m (map outlines smaller), NMAD 28 m (Mannerfelt et al. 2022); perimeter 8205 m x 28 m = ±0.23 | ±0.082 | 1 polygon, 1 hole 544 m2; lowest 2321 m | `sgi_1931` (LV03 to LV95 with CHENyx06a, 0.2 m) |
| SGI1973 | 1973 (aerial images) | 2.320 | not quantified in the ReadMe | ±0.103 | 1 polygon, 1 hole 544 m2; lowest 2421 m | `sgi_1973` |
| SGI2010 | 2008-2011 (image year of C14-10 not in the attributes) | 1.897 | inventory total ±2.6 % (944.3 ± 24.1 km2, Fischer et al. 2014) | ±0.068 | 1 polygon; 2556-3179 m | `sgi_2010` |
| SGI2016 | 2018 (`year_acq`) | 1.641 | inventory total ±2.3 % (961 ± 22 km2, round robin, Linsbauer et al. 2021) | ±0.100 | 1 polygon, 3 holes 2749 m2; debris 0.0014 km2; centreline 1.255 km; attributes 2588-3181 m | `sgi_2016`, `sgi_2016_debriscover`, `sgi_2016_centerlines` |
| swissTLM3D 2024 = SGI2023 | 2024 (aerial images of August 2024) | 1.380 (1.368 + 0.008 + 0.003) | SGI2023 publication in preparation, none given | ±0.112 | 3 polygons, 3 holes 5582 m2; debris 0.0036 km2; centreline 1.088 km; attributes 2585-3193 m | `glacier_2024`; SGI2023 (release 2026) online only, same geometry for C14-10 |
| Sentinel-2, ours | 2026-09-27 10:28 UTC (Sentinel-2C) | 1.025 (main body 1.021 + detached blocks 0.003 + 0.001) | pixel x perimeter ±0.093 (92 500 m2); spread of the 5 late-September scenes 0.015 (14 720 m2); debris-covered ice counts as rock; detached blocks kept when ice in 3 of the 5 late-September scenes and at least half inside 2024 (4228 m2; a 490 m2 polygon in cast shadow left out) | ±0.091 | 3 polygons (body, 2 detached blocks), 4 holes 33 155 m2; 2678-3117 m | `data/outline_2026/outline_2026-09-27.gpkg` (.geojson), `code/outline_2026.csv` |
| Course mask | October 1994 | 2.3971 | 10 m cells | ±0.092 | 23 971 cells, cell-edge perimeter 9220 m | `data/course/mask_basodino_1994.asc` |
| Course mask | October 2021 | 1.6065 | 10 m cells | ±0.109 | 16 065 cells, cell-edge perimeter 10 880 m | `data/course/mask_basodino_2021.asc` |

The 2026 values are from the run of 2026-10-06 18:32, after the rule change of steps 5 and 6 of
`code/outline_2026.py` (narrow strips joined by the 10 m bands; holes and detached blocks by the votes of the five
late-September scenes). Commit 10a3d87 had 1.015 km2: 1 polygon, 3 holes 23 660 m2, 5616 m2 of detached patches
left out.

Further area series (km2):

- GLAMOS mass-balance area, 25 m grid, with its elevation range (`data/glamos_basodino_massbalance_fixdate.csv`):
  1992-1996 2.404 (2540-3186 m), 1997-2005 2.205 (2546-3203), 2006-2010 1.961 (2532-3182), 2011-2016 1.842
  (2566-3180), 2017-2019 1.758 (2600-3178), 2020-2022 1.640 (2590-3178), 2023 1.544 (2617-3178), 2024 1.489
  (2627-3178), 2025 1.354 (2634-3168).
- GLAMOS outlines attached to the survey elevation models (`data/glamos_basodino_volumechange.csv`): 1929 3.202,
  1946 0.997, 1961 2.595, 1971 2.329, 1985 2.338, 1987 2.340, 1991 2.336, 2002 1.691, 2008 1.962, 2012 1.705,
  2014 1.646, 2018 1.641, 2019 1.619, 2020 1.590, 2021 1.591, 2024 1.379.
- Our Sentinel-2 method on other dates (`code/outline_2026.csv`; one scene each, so holes and detached blocks of
  1000 m2 or more are kept; main body in brackets): 2021-09-01 1.578 (1.575), 2024-09-10 1.385 (1.366),
  2025-09-07 1.237 (1.231), 2026-08-13 1.082 (1.076), 2026-09-12 1.067 (1.062).
- RGI 6.0, RGI60-11.01987: 2.092 (area in the Hugonnet et al. 2021 table; outline year not checked).

## 2 Surface elevation models

| Surface | Date | Grid | Uncertainty, checks | File |
|---|---|---|---|---|
| Course DEM 1994 | October 1994 (assignment) | 10 m, 260 x 153, lower left 2678950 / 1140310 | source not stated; off the glacier identical to swissALTI3D 2022 (median 0.00 m, NMAD 0.41 m), so the stable-terrain difference to the 2021 grid (0.00 ± 0.09 m, task 1) holds no error information; on its mask 16.0 m (mean, 98 % of the cells) above the 2012 surface | `data/course/dem_basodino_1994.asc` |
| Course DEM 2021 | October 2021 (assignment) | as above | equals swissALTI3D 2022 averaged to 10 m (mean -0.01 m, SD 0.11 m on the 2021 mask), i.e. the aerial survey of 30 September 2021; 6.1 m (mean) below the 2012 surface on its mask (GLAMOS 2012 to 2021: -5.9 m) | `data/course/dem_basodino_2021.asc` |
| swissALTI3D release 2022 | glacier: stereo aerial survey 2021-09-30; elsewhere LiDAR 2019-2020 | 0.5 m, 2 m | 1-3 m above 2000 m for stereo correlation (swisstopo); against swissSURFACE3D 2019: stable terrain 0.00 m (NMAD 0.06 m), glacier -1.34 m (median) | `data/dem_2m.tif`; `s02:derived/dem_0p5m.tif`, `dem_2m.tif` |
| swissALTI3D release 2020 | glacier 2018 (GLAMOS `sa3d-2020`) | 0.5 m, 2 m | not downloaded; on STAC for all glacier tiles | STAC `ch.swisstopo.swissalti3d`, items `swissalti3d_2020_*` |
| swissALTI3D release 2019 | glacier 2012 (reference year: mode and youngest 2012 on every cell, oldest 2009 on 80 %) | 10 m | the surface the Grab et al. thickness refers to | `data/thickness/C14-10_SwissALTI3D_r2019.tif`, `C14-10_ReferenceYear.tif` |
| swissALTI3D release 2025 | not known here | - | the SGI2023 elevation attributes refer to it; not on STAC for these tiles on 2026-10-06 (2020 and 2022 only) | - |
| swissSURFACE3D raster | 2019-09-20 (LiDAR; GLAMOS `ss3d-r2024`) | 0.5 m | read from the public tiles, not stored | URL pattern in `proposal_esa/alti_date.py` |
| DHM25 level 1 | GLAMOS dates the C14-10 surface 1987 | 25 m | 3-8 m in the Alps (swisstopo) | not on s02, s03, s04; 25 m grid by e-mail order (geodata@swisstopo.ch), 200 m grid free |
| Piemonte DTM5 (ICE LiDAR) | 2009-2011 | 5 m | offset +1.913 ± 0.15 m to LN02; covers 0.077 km2 of the SGI2016 outline (border margin) | `data/italy/out/italy_dem_2m_ln02.tif`, `italy_datum_offset.json` |
| Copernicus DEM GLO-30 | 2011-2015 (TanDEM-X) | 30 m surface model, EGM2008 | not checked on the glacier | `data/italy/src/copdem/` |
| GLAMOS survey models from maps and aerial images | 1907, 1929, 1949, 1961, 1971, 1985, 1991, 2002, 2008, 2013, 2014, 2024 | - | only their differences are published (section 3); the other GLAMOS surveys are DHM25 (1987), swissALTI3D (2012, 2018, 2020, 2021) and swissSURFACE3D (2019) | rasters not public |

## 3 Geodetic volume change

GLAMOS release 2025. Sigma is the uncertainty of Bgeod in m w.e. a-1, 1 sigma, from the spatial correlation of
the differences over stable terrain (GLAMOS ReadMe; the unit row of the CSV says m w.e.). Consecutive periods are
the GLAMOS series file (`volumechange_2025_r2025_series.csv` in the release zip, online only); every row except
1949-1961 equals (to rounding) a row of `data/glamos_basodino_volumechange.csv`, which holds all 144 survey pairs.

| Period | Surveys | Area km2, start to end | dV km3 | Mean dh m | Bgeod ± sigma, m w.e. a-1 |
|---|---|---|---|---|---|
| 1907 to 1929 | Siegfried map to Kurvenplan 1:25 000 | 3.202 to 3.202 | -0.1053 | -32.9 | -1.270 ± 0.411 (covered 96 %) |
| 1929 to 1949-10-07 | Kurvenplan 1:25 000 to 1:10 000 (all-pairs file only) | 3.202 to 0.997 (1946 outline) | -0.0522 | -14.2 | -1.057 ± 0.409 |
| 1949-10-07 to 1961-09-19 | Kurvenplan 1:10 000 to aerial b/w | 2.595 to 2.595 | -0.0051 | -2.0 | -0.139 ± 0.201 (all-pairs file, start area 0.997: -0.343 ± 0.143) |
| 1961-09-19 to 1971-09-03 | aerial b/w | 2.595 to 2.329 | +0.0122 | +4.7 | +0.422 ± 0.155 |
| 1971-09-03 to 1985-09-10 | aerial b/w | 2.329 to 2.338 | +0.0052 | +2.2 | +0.136 ± 0.141 |
| 1987 to 1991-09-18 | DHM25 level 1 to aerial b/w | 2.340 to 2.336 | -0.0091 | -3.9 | -0.831 ± 0.214 (1985 to 1991: -0.908 ± 0.153) |
| 1991-09-18 to 2002-09-16 | aerial b/w | 2.336 to 1.691 | -0.0095 | -4.0 | -0.363 ± 0.147 |
| 2002-09-16 to 2008-08-29 | aerial b/w | 1.691 to 1.962 | -0.0195 | -9.8 | -1.510 ± 0.229 |
| 2008-08-29 to 2013-08-21 | aerial b/w to image strip (special flight) | 1.962 to 1.705 | -0.0052 | -2.6 | -0.480 ± 0.110 |
| 2013-08-21 to 2018 | image strip to swissALTI3D 2020 | 1.705 to 1.641 | -0.0093 | -5.4 | -0.942 ± 0.285 |
| 2018 to 2024-08-23 | swissALTI3D 2020 to image strip (standard flight) | 1.641 to 1.379 | -0.0122 | -7.4 | -1.149 ± 0.133 |

Long baselines from the all-pairs file: 1907 to 2024-08-23 -0.2096 km3, -65.4 m, -0.665 ± 0.285; 1929 to 2024
-0.1106 km3, -34.5 m, -0.432 ± 0.269; 1961 to 2024 -0.0636 km3, -24.5 m, -0.432 ± 0.109; 1991-09-18 to 2021
(swissALTI3D 2022) -0.0478 km3, -20.4 m, -0.690 ± 0.213; 2012 to 2021 -0.0102 km3, -5.9 m, -0.584 ± 0.111;
2019-09-20 to 2024 -0.0125 km3, -7.6 m, -1.417 ± 0.143; 2021 to 2024 -0.0104 km3, -6.5 m, -1.989 ± 0.251.

Other estimates:

- Course DEMs, October 1994 to October 2021 (`code/task1_geodetic.csv`): dV -0.0495 km3, mean dh -20.6 m on the
  cells glaciated in 1994 or 2021, -0.779 m w.e. a-1 with the mean area 2.002 km2 (-0.651 with the 1994 area,
  -0.971 with the 2021 area), -21.0 m w.e. in total. No independent error estimate (the grids share the terrain
  off the glacier).
- Hugonnet et al. (2021), RGI60-11.01987, area 2.092 km2: 2000-01-01 to 2020-01-01 dh/dt -0.870 ± 0.209 m a-1,
  -0.740 ± 0.194 m w.e. a-1; 2000-2010 -0.709 ± 0.284; 2010-2020 -0.771 ± 0.351 (errors as given in the table,
  described as 1 sigma, not checked against the dataset ReadMe). Online only:
  `cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/hugonnet_2021_ds_rgi60_pergla_rates_10_20_worldwide.csv`.

## 4 Glaciological mass balance

GLAMOS release 2025, 34 hydrological years 1991/92 to 2024/25 (8-12 annual stake values a year, section below).

- Fixed dates (winter to 30 April, year to 30 September), `data/glamos_basodino_massbalance_fixdate.csv` (read by
  `code/common.py`). Mean and inter-annual SD in m w.e. a-1:

  | Years | Bw | Bs | Ba | Sum Ba |
  |---|---|---|---|---|
  | 1992-2025 | +1.469 (0.460) | -2.330 (0.806) | -0.861 (0.878) | -29.27 |
  | 1995-2021 (October 1994 to September 2021) | +1.466 (0.441) | -2.219 (0.675) | -0.754 (0.701) | -20.36 |
  | 2016-2025 | +1.657 (0.508) | -3.053 (0.647) | -1.397 (0.980) | -13.96 |

  Extremes of Ba: 2022 -3.512 (Bw +0.887, Bs -4.399), 2023 -2.562, 2003 -2.403; positive in 1994, 1995, 2001
  (+1.018), 2009, 2013. ELA 2535-3587 m and AAR 0-99 % per year; the ELA lies above the top of the glacier in 17
  of the 34 years and the AAR is 0 in 11.
- Observation periods (survey dates, periods of 9.5 to 14 months): `data/glamos_basodino_massbalance_observation.csv`.
- Uncertainty (Huss et al. 2015): random error of annual balances ±0.15-0.40 m w.e. a-1, seasonal components of the
  same order or larger; multi-year means limited by the geodetic surveys, ±0.05-0.15 m w.e. a-1. Where glaciological
  and geodetic balance of a survey period differ by more than 0.1 m w.e. a-1, the GLAMOS model is re-calibrated, so
  this series is not independent of section 3.
- Elevation bands, 100 m from 2500 to 3200 m, 232 rows: observation periods in
  `data/glamos_basodino_massbalance_observation_elevationbins.csv`; fixed dates online only
  (`massbalance_fixdate_elevationbins_2025_r2025.csv` in the release zip). Means 1992-2025, fixed dates:

  | Band m | Years | Bw | Bs | Ba | Area km2 |
  |---|---|---|---|---|---|
  | 2500-2600 | 28 (to 2022) | +1.32 | -3.22 | -1.90 | 0.001-0.082 |
  | 2600-2700 | 34 | +1.39 | -3.15 | -1.77 | 0.028-0.346 |
  | 2700-2800 | 34 | +1.43 | -2.83 | -1.40 | 0.236-0.474 |
  | 2800-2900 | 34 | +1.49 | -2.36 | -0.87 | 0.333-0.494 |
  | 2900-3000 | 34 | +1.56 | -1.98 | -0.42 | 0.432-0.546 |
  | 3000-3100 | 34 | +1.45 | -1.82 | -0.37 | 0.276-0.377 |
  | 3100-3200 | 34 | +1.18 | -1.71 | -0.53 | 0.050-0.092 |

- Point balances, GLAMOS point mass balance release 2025 (doi:10.18750/massbalance.point.2025.r2025, 1.5 MB zip,
  online only; positions in LV03): annual stakes 344 values 1992-2025 (8-12 a year, 2586-3052 m, error median
  53 mm w.e., 90th percentile 155); winter probings and snow pits 1127 values (2424-3161 m, error median 209 mm
  w.e.); intermediate readings 110 values 2021-2025.

## 5 Length change

GLAMOS release 2025, `data/glamos_basodino_lengthchange.csv`: 97 intervals from 1899-09-01 to 2024-11-06,
cumulative -813.5 m. No uncertainty in the file; before 2000 the dates are nominal (1 September, flagged `x`, 74
rows). Cumulative change since 1899: 1911 -191 m, 1925 -118 (advance), 1929 -146, 1946 -233, 1961 -526, 1966 -546,
1986 -458 (advance), 1994 -469, 2002 -559, 2010 -659, 2016 -733, 2021 -765, 2024 -814. Tongue elevation since
2000: 2530 m (2000), 2568 (2008), 2605 (2016), 2663 (2021), 2688 (2024). Centreline length: 1.255 km (SGI2016,
2018), 1.088 km (SGI2023, 2024).

The terminus has more than one lobe: the lowest point of the 2024 outline is 2585 m at the east tip
(E 2681435, N 1141073) while GLAMOS gives a tongue at 2688 m; the 2026 outline ends at 2678 m (E 2681031,
N 1141095).

## 6 Ice thickness and bed

`data/thickness/` (fetched 2026-10-06, `data/thickness/README.md`).

- Grid of Grab et al. (2021), mean of GlaTE and ITVEO constrained by radar, 10 m, 16 406 cells = SGI2016 outline:
  thickness for the 2012 surface, mean 13.8 m, maximum 25.2 m, volume 0.02268 km3 (2012), 0.0178 km3 homogenised
  to 2016 (summary table). Per-cell uncertainty -10.0 to -12.1 m (median -11.1) and +10.2 to +15.5 m (median
  +12.1); every cell at its lower or upper bound gives 0.0065 to 0.0426 km3 (`code/thickness.csv`). Files
  `C14-10_IceThickness.tif`, `C14-10_GlacierBed.tif`, `C14-10_IceThicknessUncertaintyMinus.tif`, `...Plus.tif`.
- Radar, ETH Zurich survey of 2006 (day not documented), 1047 points on 6 profiles (4.27 km): thickness 1-32 m,
  mean 17.2 m; min-max half range median 5.3 m; uncertainty type 3.1 (one reflection, average picking error).
  `C14-10_basodino_GlaThiDa.txt`.
- Derived from these and the course grids (`code/thickness.py` -> `code/thickness.csv`, written 2026-10-06 17:45
  by another workflow): 2021 volume 0.0142 km3 (mean 8.8 m on the 2021 mask; ground = bed); 1994 volume
  0.0637 km3 (mean 26.6 m; ground = bed but at most the 2021 surface, and the 2021 surface off the data's outline,
  which covers 68 % of the 1994 mask). The bed lies above the 2021 surface on 0.257 km2 of the data's outline,
  0.213 km2 of it inside the 2021 mask (13 %, median 5.8 m above), at most 24.8 m above; so the 2012 volume is too
  small by at least 0.0019 km3. Ratio data / volume-area scaling (task 5, 0.0652 km3 for 2021): 0.34 (2012),
  0.22 (2021), 0.56 (1994).
  `data/thickness/sections.csv`: bed and the 1994, 2012, 2021 surfaces along two sections (A summit to tongue,
  B across the body along the 2006 radar profiles), 10 m spacing.
- Not fetched: Farinotti et al. (2019) consensus thickness for RGI60-11.01987; GLAMOS ice thickness release 2018
  (radar 1999-2015, may hold the same 2006 profiles).

## 7 Surface velocity

- In situ: none. The GLAMOS flow velocity release 2025 (doi:10.18750/flowvelocity.2025.r2025) has no C14-10 rows.
- ITS_LIVE version 2: datacube `ITS_LIVE_vel_EPSG32632_G0120_X450000_Y5150000.zarr` (87 053 image pairs over its
  100 km square) and annual composites 2013-2025 at 120 m
  (`its-live-data.s3.amazonaws.com/composites/annual/v2-updated-september2025/N40E000/ITS_LIVE_velocity_EPSG32632_120m_X450000_Y5150000.zarr`).
  96 pixel centres inside the 2024 outline. Annual speed, median over these pixels: 0.8-2.0 m a-1 in 2014-2024
  (4.0 in 2013, 7.4 in 2025); on ground 300-1500 m outside the outline 0.7-1.5 m a-1 (3.6 and 8.8). Climatological
  v0: 0.60 m a-1 on the glacier, 0.27 m a-1 outside. Reported v_error 0-2 m a-1 (integers).
- AlpGlacier (Rabatel et al. 2023): annual Sentinel-2 speeds 2015/16 to 2020/21 at 50 m for all Alpine glaciers
  over 0.03 km2, doi:10.57745/XHQ7TL, one GeoTIFF of 300 MB per year for the whole Alps; not downloaded.
- Millan et al. (2022) global speeds at 50 m (2017-2018): online, not checked.
- Feature tracking on our Sentinel-2 archive: 1576 scenes 2015-2026 on `s02:derived/satellite_archive/sentinel2`
  (B02, B03, B04, B08 at 10 m, 20 m bands, the LV95 10 km square; scene list `catalog.csv`). Clear scenes from
  August to October (cloud below 2 % over the 2024 outline): 2015 3, 2016 7, 2017 13, 2018 9, 2019 7, 2020 10,
  2021 14, 2022 10, 2023 13, 2024 7, 2025 15, 2026 15. Nothing tracked yet.
- Scale for comparison: shallow-ice deformation speed at the surface for the mean slope of 19 deg and
  A = 2.4e-24 Pa-3 s-1: 0.03 m a-1 for 13.8 m of ice, 0.37 m a-1 for 25 m, 1.0 m a-1 for 32 m.

## 8 Available, not processed

- Snow and bare-ice fractions over the 2024 outline per Sentinel-2 scene (`catalog.csv` columns `snowice_g2024`,
  `bareice_g2024`): an end-of-summer snow-line proxy, not compared with the GLAMOS AAR.
- Landsat 5 scenes 1985-09-21, 1994-08-04, 1994-08-20 and 2003-08-13 (`data/satellite/`, snow index at 10 and
  30 m): outlines for 1985, 1994 and 2003 possible.
- SWISSIMAGE 2018 and 2021 (STAC, not downloaded) and the swisstopo aerial image archive (STAC
  `ch.swisstopo.lubis-luftbilder_schwarzweiss`, the source of the GLAMOS b/w surveys).
- WGMS Fluctuations of Glaciers (version 2026-02-10, 868 MB): holds what GLAMOS submits; not checked for C14-10.

## 9 Consistency notes

- One year, several areas: 2021 course mask 1.6065, GLAMOS survey outline 1.591, Sentinel-2 (1 September) 1.575;
  2024 swissTLM3D 1.380, GLAMOS survey outline 1.379, GLAMOS mass-balance area 1.489, Sentinel-2 (10 September)
  1.360.
- GLAMOS survey outlines: 2002 (1.691) is smaller than 2008 (1.962); the 1946 outline (0.997) is attached to the
  1949 survey in the all-pairs file.
- The course DEM of 2021 is swissALTI3D 2022; the course DEM of 1994 carries the modern terrain off the glacier.
- The 2021 surface lies below the Grab bed on 13 % of the 2021 mask (0.213 km2; 0.257 km2 of the data's outline,
  `code/thickness.csv`).
- GLAMOS seasonal balances and geodetic volume changes are linked by the re-calibration (section 4).
- `code/task1_geodetic.csv` gives `glamos_sigma` the unit m w.e.; GLAMOS defines sigma in m w.e. a-1.

## Sources

- GLAMOS (2025) mass balance, doi:10.18750/massbalance.2025.r2025; point mass balance,
  doi:10.18750/massbalance.point.2025.r2025; length change, doi:10.18750/lengthchange.2025.r2025; volume change,
  doi:10.18750/volumechange.2025.r2025; flow velocity, doi:10.18750/flowvelocity.2025.r2025.
- Swiss Glacier Inventories: SGI1850 doi:10.18750/inventory.sgi1850.r1992; SGI1931 doi:10.18750/inventory.sgi1931.r2022
  (Mannerfelt et al. 2022, The Cryosphere 16, 3249-3268); SGI1973 doi:10.18750/inventory.sgi1973.r1976; SGI2010
  doi:10.18750/inventory.sgi2010.r2010 (Fischer et al. 2014, AAAR 46, 933-945); SGI2016 doi:10.18750/inventory.2016.r2020
  (Linsbauer et al. 2021, Front. Earth Sci. 9, 704189); SGI2023 doi:10.18750/inventory.2023.r2026.
- Huss, Dhulst and Bauder (2015), J. Glaciol. 61(227), 551-562, doi:10.3189/2015JoG15J015.
- Grab et al. (2021), J. Glaciol. 67(266), 1074-1092, doi:10.1017/jog.2021.55; data doi:10.3929/ethz-b-000434697;
  radar doi:10.18750/icethickness.2020.r2020.
- Hugonnet et al. (2021), Nature 592, 726-731, doi:10.1038/s41586-021-03436-z.
- ITS_LIVE version 2 (NASA MEaSUREs), its-live.jpl.nasa.gov; Rabatel et al. (2023), Data 8(4), 66,
  doi:10.3390/data8040066.
- swisstopo: swissALTI3D, swissSURFACE3D, DHM25 product pages (accuracies quoted above).
