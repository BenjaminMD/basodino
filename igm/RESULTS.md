# Basòdino ice-flow ensemble (IGM): results of the fixed ensemble

Step 7 of `DESIGN.md` after the fix round (F1 balance over 1907-2025, F2 basin, F3 start state on 2021-2026, F5
spin-up and hindcast, F6 projections), 10 October 2026. The first ensemble (9 October) is archived:
`results/first_ensemble/RESULTS.md`, its tables in `results/first_ensemble/`, its figures in
`results/first_ensemble/figures/`, its films in `tex/animation/first_ensemble/`. Every number below is followed by
the file it comes from (paths relative to `igm/`; `s02:` is `/data/befahl/projects/AGL/basodino/igm/`). Years are
hydrological years (state on 1 October at the end of the year). Ice = thickness of at least 1 m; A1 = area of such
ice; disappearance = first year with A1 < 0.01 km2 (or 0.1 km2) and no later return; half-volume year = first year
with V <= V_2026 / 2 (or V_2021 / 2); "not by 2100" counts as 2101 in quantiles (decision 14).

## 0. Quality bar

| Item | Status | Numbers |
|---|---|---|
| (a) 2026 start states reproduce the observed main body (1.02 km2) | met | largest body 0.849 / 0.980 / 1.111 km2 (5 / 50 / 95 %) against 1.024 on the 10 m lattice; intersection over union 0.69 / 0.74 / 0.78 (`results/f56_start.csv`, `f56_check.csv`) |
| (b) hindcast retreats through the inventories, matches 1907-1991 volume changes | not met, every miss quantified (section 8) | inventory areas 2 of 4 inside the band (2010 z +7.0, 2016 z +12.3); volume changes 5 of 11 within 2 sigma; 1907-1991 sum -0.014 against -0.164 km3; volume on 2021-09-30 0.057 km3 (median) against 0.022 from the measured thickness (`results/f5_validation.csv`) |
| (c) ice inside the domain | met | 0 ice cells on the domain edge or outside the basin at any yearly save of spin-up, hindcast and projection; up to 11 cells (0.0011 km2) outside the basin in outline-date reconstructions (`results/f56_check.csv`) |
| (d) ITS_LIVE speed bound | quantified | 2014-2021 median speed 0.124 / 0.409 / 2.006 m a-1 against the 1 m a-1 bound; 21.7 % of the F3 posterior weight above it, mean excess 0.14 m a-1; 23.4 % of the 128 draws (`results/f3_summary.csv`, `f56_misc.csv`) |
| (e) effective sample size | met | F3 pooled effective sample size 721.6 (gate 128); the 128 draws are 128 distinct samples (`results/f3_summary.csv`, `posterior_f3_128.csv`); first ensemble 8.6, 34 distinct |

## 1. The ensemble

- 10 062 projection runs: 9 984 members (128 posterior draws x 26 chains x 3 SSPs) + 78 runs of the MAP draw (103)
  with u = 0; all ended (9 004 with no ice left, 1 058 at 2100.75 with ice), 0 NaN, 10 062 consistent ice-free grids,
  129 network trainings, 4 reruns identical (`results/f6_gates.csv`). 9 984 no-flow twins with the IGM stop rule,
  0 NaN (`results/f6_gates.csv`). Independent check of F5 and F6 from the banked tars: every row passes
  (`results/f56_check.csv`).
- Collected (`collect.py`, s02, 3 min): `results/members.csv` (one row per member; the four years recomputed from the
  series agree with the reduction of `projection.py` in all 10 062 + 9 984 rows, `s02:results/f7/collect.log`),
  `s02:results/f7/series.nc` (yearly series 2021-2100 per member, hindcast 1850-2026 per draw),
  `s02:results/f7/icefree.nc` (ice-free year per cell and member, with and without flow),
  `s02:results/f7/hindcast_fields.nc` (thickness of the 128 hindcasts at the 19 validation dates and on 2026-09-30,
  and the bed of each draw). Summary tables: `results/f7_series_quantiles.csv`, `f7_years.csv`, `f7_spread.csv`,
  `f7_icefree_summary.csv`, `f7_icefree_maps.npz`, `f7_half2021_samples.csv`.

## 2. Half-volume and disappearance years per SSP

Quantiles over the 1 664 members of each SSP and 2026 level (`results/f7_years.csv`, run "with flow"; percentiles
interpolated between years):

| Level | SSP | Half of V_2026: 5 / 17 / 50 / 83 / 95 % | Below 0.1 km2 | Below 0.01 km2 | Not below 0.01 km2 by 2100 |
|---|---|---|---|---|---|
| anchored | SSP1-2.6 | 2028 / 2028 / 2029 / 2031 / 2032 | 2031 / 2033 / 2035 / 2041 / 2049 | 2035 / 2037 / 2041 / 2046 / 2055 | 0.7 % |
| anchored | SSP2-4.5 | 2028 / 2028 / 2029 / 2031 / 2032 | 2032 / 2033 / 2035 / 2038 / 2043 | 2035 / 2037 / 2039.5 / 2043 / 2048 | 0 % |
| anchored | SSP5-8.5 | 2028 / 2028 / 2029 / 2031 / 2032 | 2032 / 2033 / 2034 / 2037 / 2042 | 2035 / 2036 / 2039 / 2042 / 2046 | 0 % |
| native | SSP1-2.6 | 2028 / 2029 / 2031 / 2034 / 2036 | 2032 / 2035 / 2077 / 2096 / after 2100 | 2037 / 2041 / 2086 / after 2100 / after 2100 | 26.0 % |
| native | SSP2-4.5 | 2028 / 2029 / 2030 / 2033 / 2036 | 2032 / 2035 / 2042 / 2056 / 2082 | 2036 / 2040 / 2048 / 2066 / 2090 | 0.9 % |
| native | SSP5-8.5 | 2028 / 2029 / 2030 / 2032 / 2034 | 2032 / 2034 / 2037 / 2043 / 2051 | 2035 / 2038 / 2042 / 2049 / 2058 | 0 % |

- The anchored level leads (answer 2). Its three SSPs differ by at most 1.5 years in the median disappearance year and
  not at all in the half-volume year; under the native level SSP1-2.6 keeps 0.01 km2 past 2100 in 26 % of members.
- Pooled over the SSPs (`results/f7_years.csv`, "all SSPs"): anchored disappearance 2035 / 2037 / 2039 / 2044 / 2049,
  half of V_2026 2028 / 2028 / 2029 / 2031 / 2032; native 2036 / 2039 / 2047 / 2090 / after 2100.
- Half-volume year against V_2021: 2025 / 2026 / 2026 / 2026 / 2026 for every SSP and level; 9 204 members in 2026,
  780 in 2025 (`results/members.csv`). It falls inside the observed 2021-2026 weather (section 5).
- MAP draw 103 with u = 0 (78 members): median disappearance 2038 / 2036 / 2035 (SSP1-2.6 / 2-4.5 / 5-8.5) against
  2041 / 2038.5 / 2039 at its own u = 0.431 (`results/f6_gates.csv`, DESIGN.md F6 result).
- Figure `tex/figures/igm_years.pdf` (a).

## 3. Share of the spread per factor

The 9 984 members are a full crossing of draw x chain x SSP (one member per cell), so the sum of squares splits exactly
(`results/f7_spread.csv`, "all"; the draw part split into 2026 level, station and the draws within; members without
disappearance by 2100 enter as 2101, so the native SSP1-2.6 tail is censored).

| Factor | Disappearance 0.01 km2 (SD 18.2 years) | Half of V_2026 (SD 2.1 years) |
|---|---|---|
| SSP | 15.0 % | 0.5 % |
| chain | 19.0 % | 10.9 % |
| 2026 level | 19.4 % | 10.8 % |
| station (+ level x station) | 1.1 % + 0.3 % | 0.6 % + 0.4 % |
| draws within a level and station (balance, u, f) | 2.0 % | 5.1 % |
| draw x chain | 14.6 % | 28.1 % |
| draw x SSP | 10.2 % | 2.2 % |
| chain x SSP | 6.9 % | 0.5 % |
| three-way and weather order | 11.6 % | 41.0 % |

- Within the anchored level (SD 6.8 years) the disappearance year splits into SSP 4.7 %, chain 24.0 %, draws within a
  level and station 9.1 %, draw x chain 23.2 %, three-way and weather order 29.4 %; within the native level (22.2
  years) SSP 32.7 %, chain 37.1 %, chain x SSP 15.9 %, draws within 1.9 % (`results/f7_spread.csv`).
- Linear share of single draw parameters (for the record): u 0.06 %, ln f 0.2 %, V_2026 0.3 % of the total
  disappearance-year variance; within the anchored level u 1.0 %, ln f 0.5 %, V_2026 2.1 % (`results/f7_spread.csv`).
- Figure `igm_years.pdf` (b).

## 4. With and without ice flow

- The no-flow twins now stop by the IGM rule (first 1 October without a cell >= 1 m), so with and without flow are
  compared as run (`results/f7_years.csv`, "IGM minus no-flow"): disappearance at 0.01 km2, all 9 984 members, 5 / 17
  / 50 / 83 / 95 % = 0 years; 96.1 % the same year, 2.7 % earlier with flow, 1.2 % later; smallest -47, largest +54
  years; mean -0.16. At 0.1 km2: 98.9 % the same. Half of V_2026: 99.95 % the same (5 members 1 year earlier).
- Under the anchored level 97.3 % have the same disappearance year (1.7 % earlier, 1.1 % later); under the native
  level 95.0 % (3.7 % earlier, 1.3 % later).
- First ensemble: as reduced -16.6 years mean, 0.0 with the stop rule applied afterwards (4 340 regrowing twins);
  that question is closed by the stop rule in the twins (DESIGN.md F6).
- Figure `igm_years.pdf` (c).

## 5. Half-volume year against V_2021 as a function of u

- V_2021 rises with u from 0.0034 km3 (u = -1) through 0.0144 (u = 0) and 0.0216 (u = 0.44, the posterior median) to
  0.0310 (u = 1) and 0.0481 (u = 2) (`results/f7_half2021_samples.csv`, the 2 048 F3 samples, start of their 2021
  run).
- Under the observed weather of 2021-2026, V <= V_2021 / 2 is reached in 2022 for u from -1.00 to -0.55 (166 samples),
  2023 for -0.59 to 0.05 (212), 2024 for -0.08 to 0.13 (22), 2025 for -0.02 to 0.48 (251), 2026 for 0.12 to 1.17
  (1 019), and later than 2026 for 0.91 to 2.00 (378) (`results/f7_half2021_samples.csv`).
- The 128 ensemble draws have u from 0.218 to 0.769 (`results/posterior_f3_128.csv`); their half-volume year against
  V_2021 is 2025 or 2026 (`results/members.csv`). V_2026 / V_2021 per draw 0.27 / 0.34 / 0.41 (`results/f56_misc.csv`).
- Figure `igm_half_volume_u.pdf`.

## 6. Volume and area 2021-2100 and the 2026 start

- Anchored, SSP2-4.5, 5 / 50 / 95 % (`results/f7_series_quantiles.csv`): V_2021 0.0190 / 0.0216 / 0.0254 km3; V_2026
  0.0052 / 0.0073 / 0.0109 km3; V_2030 0.0008 / 0.0027 / 0.0059 km3; A1 in 2026 0.842 / 0.981 / 1.137 km2, in 2030
  0.229 / 0.534 / 0.894, in 2035 0.004 / 0.059 / 0.398, in 2040 0 / 0.004 / 0.078 km2. Native, SSP2-4.5: A1 in 2030
  0.310 / 0.648 / 1.073, in 2040 0 / 0.063 / 0.523, in 2050 0 / 0.003 / 0.158 km2. 2021-2026 is the same for all SSPs
  (observed weather).
- Start against the observed 2026 extent (`results/f56_start.csv`): model ice >= 1 m 0.850 / 0.980 / 1.111 km2;
  share of the observed main body covered 0.75 / 0.83 / 0.91; model ice outside the 2026 polygons 0.085 / 0.126 /
  0.175 km2; observed ice without model ice 0.092 / 0.174 / 0.263 km2. Its effect on the result: over the 128 draws the
  draw-median disappearance year rises by 12.0 years per km2 of start area (r 0.20); at the observed 1.021 km2 the
  line gives 2043.8 against a draw median of 2042.0 (`results/f6_gates.csv`).
- Misfit carried into the projections: the 2024 area (all polygons) 1.211 / 1.280 / 1.334 km2 against 1.380 (outside
  by 0.046 km2, -0.90 standard deviations): the model loses too much area 2021-2024 and too little 2024-2026
  (`results/f3_terms.csv`, DESIGN.md F3 result).
- Some members regrow for a while: 1 511 exceed their V_2026 at some save (largest 2.36 times; native 1 047, anchored
  464), from single positive years of 1991-2026 lent by the year order (DESIGN.md F6 result).
- Figure `igm_volume_area.pdf`; each SSP's time axis ends 3 years after the latest 95th-percentile disappearance year
  of its two levels (2100, 2093, 2061).

## 7. Year each cell becomes ice-free

Anchored, three SSPs pooled (`results/f7_icefree_summary.csv`): where the median member still has ice after 2026
(1.029 km2) the median ice-free year is 2031, the latest 2044; 0.0038 km2 has a median of 2041 or later; on 16.7 % of
the mapped 2026 outline the median member has no ice at any save from 2026 on; the 90 % range is 5 years at the median
cell, 29 at most; some member has ice at some save on 1.859 km2 (regrowth outside the 2026 extent). Native: 1.157 km2,
2034, latest 2051, 0.114 km2 with 2041 or later, 14.3 %, range 26 and 75 years. Per SSP: same file; maps
`results/f7_icefree_maps.npz`. Figure `igm_icefree_map.pdf`.

## 8. Hindcast 1850-2026 against observations (validation, not fitted)

`results/f5_validation.csv` (128 draws, band = 5-95 % of the draws, z = (median - observed) / sigma); expected misses
written before the runs (`results/f5_targets.csv`); first ensemble's median in brackets:

- Areas (km2): SGI1931 2.20 / 2.66 / 3.30 against 3.20, inside, z -2.4 (5.49); SGI1973 2.19 / 2.50 / 2.84 against
  2.32, inside, z +1.8 (5.19); SGI2010 1.90 / 2.24 / 2.66 against 1.90, outside, z +7.0 (2.71); SGI2016 1.72 / 2.11 /
  2.55 against 1.64, outside, z +12.3 (1.92). 1850: 4.04 / 4.18 / 4.23 in all, 3.98 inside SGI1850.
- Volume changes (km3, median against observed, z): 1907-1929 +0.0090 / -0.1053 (+3.4, expected too little loss:
  met); 1929-1949 -0.0085 / -0.0522 (+2.2); 1949-1961 +0.0039 / -0.0087 (+3.5, expected: met); 1961-1971 -0.0041 /
  +0.0122 (-3.6, expected too much loss: met); 1971-1985 +0.0012 / +0.0052 (-0.7); 1985-1991 -0.0163 / -0.0150 (-0.5,
  inside); 1991-2002 +0.0037 / -0.0095 (+3.4, expected: met); 2002-2008 -0.0146 / -0.0195 (+1.7, outside the band,
  expected: met); 2008-2013 -0.0036 / -0.0052 (+1.4, inside); 2013-2018 -0.0082 / -0.0093 (+0.4, inside); 2018-2024
  -0.0183 / -0.0122 (-4.3, too much loss, not expected). 5 of 11 within 2 sigma, 3 inside the band (first ensemble: 6
  of 6 outside). Sums of the per-draw changes: 1907-1991 -0.050 / -0.014 / -0.005 against -0.164 km3; 1991-2024 -0.057
  / -0.041 / -0.029 against -0.056 km3.
- Tongue elevation: inside the band 2000-2021, outside 2022-2024 (2024: 2619, band 2520-2670, against 2688); first
  ensemble outside 2000-2017.
- Length along the flowline: 1899-2024 -956 m (-1312 to -610) against -814, inside (first: -2222); 1899-1929 -426
  against -146, outside (too much retreat before 1929); 1899-1961 and 1899-1986 inside. The figure draws the band to
  2023, the last year in which every draw has ice on the flowline.
- Surfaces: mean over the observed cells inside the band in 1991, 2012, 2021 (+12 m in 2012 and 2021 at the median),
  root mean square 17 m (first ensemble 6-9 m).
- Volume and area at the end (rows "extra" of `results/f5_validation.csv`): the hindcast holds 0.025 / 0.057 / 0.112
  km3 on 2021-09-30 against 0.019 / 0.022 / 0.025 km3 of the 2021 start from the measured thickness
  (`results/f7_series_quantiles.csv`), 2.6 times at the medians, and 0.009 / 0.034 / 0.084 km3 on 2026-09-30 against
  the start state's 0.0073 (4.7 times); area 2021 2.03 (1.65-2.49) against the 2021 mask 1.607 km2, 2024 1.90
  (1.44-2.40) against 1.380 km2, both outside the band. The hindcast 1850-2021 thus ends with too much ice; the
  projections do not inherit it (last point of this section).
- Basin edge: ice that flows beyond the posbal cells (F2) is removed at every step, so the hindcast ice ends there in a
  face: in the film member faces over 20 m on 626 cells (6.3 km of margin) in 1850, 253 in 1900, 201 in 1990, 140
  (1.4 km) in 2020, up to 182 m; elsewhere on the margin at most 67 m; none from the 2021 restart on
  (`results/f7_basin_wall.csv`, `checks/basin_wall.py`). Removed volume 1850-2026 0.014 / 0.022 / 0.039 km3 over the
  draws (`results/f56_check.csv`).
- Driver of the misses: ln f correlates with every misfit (r -0.91 with SGI1931, -0.87 with SGI2016, -0.96 with
  V_2021, -0.98 with V_1850; `results/f56_check.csv`). f >= 1 fits 1931 badly (2.20 km2), f < 0.1 fits 2016 badly
  (2.40 km2); no f fits both. The balance level before 1991 (F1 misses 1907-1929 by +3.1 sigma) leaves the 1907-1991
  loss short by about 0.15 km3.
- The hindcast is a validation only: projections start from the decision-8 path (measured thickness 2021, observed
  weather to 2026), not from the hindcast (hindcast 2026 area 1.12 / 1.76 / 2.30 km2).
- Figure `igm_hindcast.pdf`.

## 9. Calibration and start-state fit, accepted misfits

- Balance (F1, `results/calib_f1_summary.csv`, `calib_f1_bands.csv`): effective sample size 11 854; the temperature
  offset before 1991 dT_pre +1.20 K (5-95 %: 1.03-1.38); 6 of 11 volume changes inside the 90 % predictive band
  (misses 1907-1929 +3.1, 1949-1961 +2.3, 1961-1971 -3.9, 1991-2002 +2.9, 2002-2008 +2.4 sigma); mean glacier-wide
  annual balance 1992-2025 -0.56 (kept 5-95 %: -0.66 to -0.47) against -0.844 m water equivalent (gate not met); bands
  and stakes inside the band: winter 91.8 %, annual 91.4 %, stakes 89.8 %; MAP lowest year 2022, 2001 positive.
  Prior edges: dT below -2.8 K in 13 % and c_prec above 2.9 in 14 % of the kept draws. The stationary set without
  dT_pre cannot fit both periods (`results/calib_f1_stationary_summary.csv`).
- Start state and flow (F3, `results/f3_summary.csv`, `f3_terms.csv`): effective sample size 721.6; u 0.264 / 0.441 /
  0.637; f 0.057 / 0.230 / 1.211 (flat in ln f from the prior edge 0.05 to about 1, cut above by the speed bound);
  2026 main body 0.843 / 0.987 / 1.102 km2 against 1.021 (inside); 2025 1.096 / 1.205 / 1.279 against 1.231 (inside);
  2024 all polygons 1.211 / 1.280 / 1.334 against 1.380 (outside); dh 2012-2021 bands within -0.21 to +0.57 standard
  deviations; radar bedrock 4-5 m above bed(u) on all six profiles (-0.79 to -1.03); stakes: 10 of 12 bounds met,
  stake 05 -0.26, stake 09 -5.1 standard deviations.
- Figure `igm_calibration.pdf`.

## 10. Against published projections

`PUBLISHED_PROJECTIONS.md`; figure `igm_published.pdf`. The global datasets start from the RGI 6.0 outline (2.09 km2)
with 0.05-0.10 km3 in 2021, 2.3 to 4.7 times this ensemble's central V_2021 of 0.0216 km3; their years are 1 January
states, ours 30 September (`data/published/basodino_members.csv`; ours `results/f7_series_quantiles.csv`).

| Quantity | This ensemble (median, anchored; native) | Published |
|---|---|---|
| Half of V_2021 | 2026 (5 %: 2025); 2026 | Rounce 2023 2030-2033, OGGM 2033-2039, Zekollari 2019 2038-2039 (SSP or RCP medians) |
| Half of V_2026 | 2029; 2030-2031 | Rounce 2033-2037, OGGM 2038-2042, Zekollari 2041-2045 |
| V_2026 / V_2021 | 0.34 (draws 0.27-0.41) | Rounce 0.66-0.77, OGGM 0.76-0.81, Zekollari 0.83-0.85 (medians per scenario) |
| Disappearance, SSP1-2.6 | 2041; 2086 | OGGM (A < 0.01 km2) after 2100; Rounce (V < 1 %) 2059; Zekollari (V < 1 %, RCP2.6) after 2100 |
| Disappearance, SSP2-4.5 | 2039.5; 2048 | OGGM 2097; Rounce 2053; Zekollari (RCP4.5) after 2100; Van Tricht 2025 (+2.7 K) 2061 |
| Disappearance, SSP5-8.5 | 2039; 2042 | OGGM 2058; Rounce 2048; Zekollari (RCP8.5) 2076; Van Tricht 2025 (+4.0 K) 2055 |
| Statements | anchored 5-95 %: 2035-2055 | Huss 2026: 2036-2046 (inside); canton 2023: about 2043; Kappenberger 2025: about 2030 (before our 5 %) |

- Anchored medians against the published medians of the same scenario: half of V_2026 4-16 years earlier (Rounce
  4-8, OGGM 9-13, Zekollari 12-16); disappearance 9-58 years earlier, more than 59 for OGGM under SSP1-2.6 and for
  Zekollari under RCP2.6 and RCP4.5 (Rounce, volume below 1 %: 9-18; OGGM, area below 0.01 km2: 19 to more than 59; Zekollari, volume below 1 %, RCP
  scenarios: 37 under RCP8.5, more than 59 under RCP2.6 and RCP4.5; Van Tricht 2025 by warming level: 16-22). The
  anchored disappearance median lies inside the 2036-2046 of Huss 2026; the anchored 5 % (2035) lies after
  Kappenberger's about 2030.
- The published V_2026 / V_2021 spans the summers 2021-2025 (1 January to 1 January); ours spans 2022-2026 (30
  September to 30 September, observed weather, model loss 2021-2026 66 % of V_2021). Not separated here: how much of
  the gap is the start volume, the published models' 2021-2025 climate, and the 2026 summer.
- Volume under SSP2-4.5 (figure a, medians): 2030 ours 0.0027 km3 against Rounce 0.025, OGGM 0.037, Zekollari
  0.074 (9-27 times ours); 2035 0.00024 against 0.017, 0.029, 0.059 (70-250 times) (`data/published/basodino_volume.csv`).

## 11. What changed against the first ensemble

| | First ensemble (9 October) | Fixed ensemble (10 October) | Files |
|---|---|---|---|
| Balance calibration | 1991-2025 terms; mean Ba -0.55; dT at its prior edge -3 K | 1907-2025 terms (11 volume changes) and dT_pre +1.20 K before 1991; mean Ba -0.56 | `results/first_ensemble/RESULTS.md` 9; `calib_f1_summary.csv` |
| Basin | none: 1.1-1.4 km2 of 1850 ice outside SGI1850, ice on the domain edge until about 1990-2020 | ice only inside SGI1850 + 100 m; 0 edge cells; basin melt 1850-2026 0.014 / 0.022 / 0.039 km3 | `f56_check.csv` |
| Start state fit | 1991-2026 joint fit, effective sample size 8.6, 34 distinct samples, f at its upper edge, all speeds above the bound | 2012-2026 fit, effective sample size 721.6, 128 distinct, f 0.06-1.2, 21.7 % above the bound | `step4_summary.csv`; `f3_summary.csv` |
| 2026 start, A1 | 0.338 / 0.455 / 0.720 km2 against 1.02 mapped | 0.842 / 0.980 / 1.112 km2 | `f6_gates.csv` |
| V_2021, V_2026 (km3) | 0.0142, 0.0020 (medians) | 0.0216, 0.0073 | `f7_series_quantiles.csv` |
| Hindcast areas 1931 / 1973 / 2010 / 2016 (km2, medians) | 5.49 / 5.19 / 2.71 / 1.92 | 2.66 / 2.50 / 2.24 / 2.11 (mapped 3.20 / 2.32 / 1.90 / 1.64) | `f5_validation.csv` |
| Hindcast volume changes | 6 of 6 outside the band; 1907-1991 +0.02 against -0.164 km3 | 5 of 11 within 2 sigma; 1907-1991 -0.014 against -0.164 km3 | `f5_validation.csv` |
| Disappearance 0.01 km2, anchored medians | 2034 / 2034 / 2034 | 2041 / 2039.5 / 2039 (+5.5 to +7) | `f7_years.csv`; `first_ensemble/step7_years.csv` |
| Disappearance, native medians | 2048 / 2041 / 2037 | 2086 / 2048 / 2042 | same |
| Half of V_2026, medians | 2028 anchored, 2028-2029 native | 2029 anchored, 2030-2031 native | same |
| Half of V_2021 | 2023 | 2026 | same |
| No-flow twins | run to 2100, 4 340 regrew | IGM stop rule, 0 regrow; with = without flow in 96 % | `f6_gates.csv`, `f7_years.csv` |
| Ice-free map, anchored | 57.2 % of the 2026 outline without ice from 2026; median year 2029 | 16.7 %; median year 2031, latest 2044 | `f7_icefree_summary.csv` |
| Films | member d056_c07_2 (f 3.83), 1850-2100 | member d047_c04_2 (f 0.243), 1850-2042 (author's rule: disappearance + 3) | section 13 |

- What got worse: the SGI2016 area (z +7.4 in the first ensemble, +12.3 now), the surface root mean square (6-9 m,
  now 17 m), and the 1899-1929 length change (-426 against -146 m): the stiffer ice (smaller f) holds more ice late in
  the hindcast and retreats too far early.

## 12. Figures (tex/figures/, drawn at 150 mm by figures.py; captions are for the author to set)

| File | Shows |
|---|---|
| `igm_hindcast.pdf` | a ice area 1850-2026 (median and 5-95 % of all ice; median inside the 1850 outline) with the inventories and the spin-up target; b the 11 volume changes 1907-2024 with the expected misses (filled: met); c lowest ice surface against GLAMOS; d length change since 1899 (band to 2023) |
| `igm_calibration.pdf` | a glacier-wide winter and annual balance (not fitted); b the two temperature offsets (station offset, offset before 1991) with the ensemble draws and the prior limits; c the 11 geodetic volume changes (binding), predictive band; d ice area at the three target dates, posterior and all samples; e elevation change per 50 m band 2012-2021; f median 2014-2021 surface speed on the ITS_LIVE pixels against the 1 m a-1 bound; g standardised misfit of every start-state term (stakes and speed are bounds: positive stake values and negative speed values meet them; arrows: beyond 6) |
| `igm_volume_area.pdf` | volume and ice area from 2021 per SSP: anchored bands (17-83, 5-95 %) and median, native median and 95 %; observed areas 2024-2026 and the 2021 start; time axis at 0.3 of its scale after 2050, ends 3 years after the latest 95th-percentile disappearance year |
| `igm_years.pdf` | a the three years per SSP and level (5-17-50-83-95 %); b share of the variance per factor; c share of members whose year differs without flow (earlier or later with flow) |
| `igm_half_volume_u.pdf` | a V_2021 against u (all F3 samples, ensemble draws); b half-volume year against V_2021 against u |
| `igm_icefree_map.pdf` | a median ice-free year per cell, b its 90 % range, anchored, three SSPs pooled, on the cells with ice in any member, outlines 1850, 1929, 1973, 2010, 2018, 2024, 2026 |
| `igm_published.pdf` | a volume 2000-2100 under SSP2-4.5 (RCP4.5 for Zekollari 2019): this ensemble against the three published models, logarithmic; b disappearance year per scenario (this ensemble 5-17-50-83-95 %; published median and range over the climate models, triangle: beyond 2100; Van Tricht 2025 median and quartiles by warming level; statements); c half of the 2026 volume per scenario |

## 13. Films

Film member (`results/f6_film.csv`, `projection.py film`): of the 1 664 anchored SSP2-4.5 members the 205 with the
median disappearance year 2039; the one closest to the group's median area series (root mean square 0.0085 km2):
d047_c04_2, draw 47 (sample f3r2_0285, u 0.451, f 0.243), chain CLMCOM-CCLM4-MPIESM, station OTL; its years 2039 (0.01
km2), 2034 (0.1 km2), half of V_2026 2030, half of V_2021 2026; no-ice stop 2043. Its hindcast, 2021-2026 path and
projection were rerun with yearly fields (stage f6_film; path and projection identical to the ensemble runs, the
cold-started hindcast within 2.5e-4 km3 and 0.007 km2, `results/f6_film_check.csv`). Both films end in 2042, the
disappearance year plus 3 (author's rule). Film states (`animate.py states`, `s02:results/f7/film_states.npz`):
hindcast 1850 V 0.256 km3, A1 4.218 km2; hindcast 2021 0.057 km3, 2.142 km2; start 2021 from the measured thickness
0.0218 km3, 1.602 km2; 2026 0.0073 km3, 0.983 km2; 2042 1e-6 km3, 0.0002 km2.

- (a) `tex/animation/igm_map_1850_2042.mp4` (`animate.py`; `s02:results/f7/igm_map_1850_2042.mp4`): 1920 x 1080, 60
  frames per second, 5 148 frames, 85.8 s, H.264 CRF 16, yuv420p, BT.709 (ffprobe). Thickness and surface-velocity
  arrows (every 100 m on the ice) over a hillshade of the modelled surface, 24 frames per model year; the year 2020-2021
  runs to the hindcast's own 2021 state, held 0.5 s, cross-faded in 1 s into the 2021 start from the measured thickness
  (labelled in red) and held 0.5 s; status lines for the hindcast, observed weather to 2026, the scenario, and 'Ice
  area below 0.01 km² from 2039'; the latest mapped outline as a red line; 1 s still, 5 s end card with the member's
  own hindcast areas against the inventories (1929 2.81, 1973 2.69, 2010 2.41, 2018 2.23 km2 against 3.20, 2.32, 1.90,
  1.64) and its 2026 area (0.98 against 1.02 km2). New choice (measured): the restarted ice moves at 0.03-0.4 m a-1
  (median 0.2 m a-1 in 2021, 0.03 in 2026) against 1.5-25 m a-1 in the hindcast, so the arrows are 20 times longer
  from the 2021 cross-fade on (6 m per m a-1 before, 120 after; the key changes from '20 m per year' to '1 m per year'
  at the same length and a line states the factor). Frame steps (`checks/film_steps.py`, `results/f7_film_steps.csv`):
  median 0.026 grey levels at 480 x 270, the largest 1.7-2.0 in the 0.5 s fade to white before the end card; every
  frame decoded.
- (b) `tex/animation/igm_3d_1850_2042.mp4` (`film3d.py`, Blender 5.2.2 Cycles with the terrain pipeline of
  `data/verification/drone/terrain/`; `s02:results/f7/igm_3d_1850_2042.mp4`): 1920 x 1080, 60 frames per second, 4 095
  frames, 68.3 s (3 375 rendered frames, 56.2 s, then a 1 s still and the 11 s end card), H.264 CRF 16, yuv420p,
  BT.709 (ffprobe, full decode). 15 rendered frames per model year to 2021 (the year 2020-2021 runs to the hindcast's
  own 2021 state), the hindcast 2021 state held, an 80-frame morph into the start from the measured thickness, held,
  then 30 frames per year to 2042 and a 1 s hold. Camera as the first film (one arc north of the glacier, 2968-3036 m
  from the look-at point, at least 612 m above the terrain, at most 59 m/s and 1.15 degrees per second of pan;
  `results/f7_film3d_path.txt`); 96 samples per pixel, OpenImageDenoise, fixed seed; grade exposure 0.0906, sky 0.403
  (`results/f7_film3d_measure.csv`). New drawing of the ice (DESIGN.md F7): thick ice on the model's own surface and an
  8 m smoothing, because the basin edge of F2 drew white spikes and a jagged wall in the first test renders. The wall
  itself is the model state and stays to 2020: faces over 20 m along 6.3 km of margin in 1850 and 1.4 km in 2020, up
  to 182 m (section 8, `results/f7_basin_wall.csv`); since the review of 10 October both end cards say so (map: 'Model
  basin: ice that flows beyond the 1850 outline plus 100 m is removed, so until 2021 the hindcast ice ends there in a
  step of up to 180 m.'; 3D: the same with 'a wall of up to 180 m: a feature of the model, not of the glacier'). All
  3 375 render frames present and readable; mean brightness change 0.011 grey levels per frame (median; in the film,
  with the camera moving, the mean absolute frame difference has median 0.42 grey levels, `results/f7_film_steps.csv`),
  the largest 3.38 at render
  frames 1894-1895 (`results/f7_film3d_check.csv`). Largest steps in the film at 480 x 270 (`results/f7_film_steps.csv`):
  3.18 into film frames 1280 and 1895 (the hindcast years 1936 and 1977, when a positive balance year adds about 0.5
  km2 of firn of 1 m or more, the area spikes of `igm_hindcast.pdf` a) and 3.18 into 2660 (the end of the 2021 morph,
  when the thinning hindcast band crosses the 0.3 m drawing depth on rough ground within two frames). Status line:
  hindcast, the 2021 restart, observed weather, scenario, 'Ice area below 0.01 km² from 2039'; timeline 1850-2042; end
  card with model, member, scenario, the member's hindcast areas against the inventories and its 2026 area. Render: s02
  RTX A6000 1 068 frames, s03 H200 0 and 1 1 155 and 1 152 frames, 4.5 s each (median), 03:09-04:54 on 10 October;
  compose on s02, 25 min. The end-card attribution keeps the licence wording verbatim (it contains 'DEM' and 'e.V.').

## 14. Banking and housekeeping

- s02 `results/f7/` holds members.csv, series.nc, icefree.nc, hindcast_fields.nc, film_states.npz, both films and the
  logs (collect.log, summary.log, film_map.log, film3d_logs.tar.gz); the run tars of F3, F5 and F6 were banked before
  this step (DESIGN.md). No new IGM runs in this step.
- Render work copies: s02 `/scratch/befahl/agl_igm3d/igm/film3d_frames_f7` (3 375 frames, 8.3 GB; the film can be
  recomposed from them) and s03 `~/dev/AGL/basodino/igm3d` (code, inputs and terrain, 1.3 GB; its frames were merged
  into the s02 folder and deleted). The map-film frames (3.8 GB) were deleted after the encode check; `animate.py
  frames` rebuilds them in 7 min. s02 `work/f6_film` holds the three film runs extracted from runs_f6.tar.gz (63 MB).
- Review of 10 October: map film re-made with the basin line on its end card (`animate.py frames 16`, `encode`, s02,
  7 min; frames deleted after the decode check) and the 3D film recomposed from the kept render frames (`film3d.py
  compose`, s02, 26 min); both banked to `s02:results/f7/` with `film_map_review.log`, `film3d_compose_review.log`.
  Frame counts, durations and frame steps unchanged (`results/f7_film_steps.csv`). Review sheets (`checks/film_sheets.py`):
  `results/f7_review/`.
- s03 run folders kept (author: keep for now).
- Superseded copies: s02 `results/first_ensemble/` (the first ensemble's small tables, moved from `results/`),
  s02 `results/step7/` (its large files and films), unchanged.

## 15. Open items

- The hindcast quality bar (b) is not met; the misses are those of section 8. The F5 and F6 open decisions stand:
  whether the hindcast should constrain f; whether the balance level before 1991 should be split by period; whether
  native SSP1-2.6 (26 % to 2100) is shown as a tail or cut.
- The comparison of section 10: which part of the 9-58-year gap to the global models is the start volume, which the
  observed 2021-2026 path and which the 2026 summer is not separated.
