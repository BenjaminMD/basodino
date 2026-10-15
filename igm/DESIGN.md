# Basòdino ice-flow ensemble (IGM): design

Approved 2026-10-07 with the answers at the end; drafted 2026-10-06, revised after review. Paths are relative to `basodino/igm/` (Mac, git) unless
prefixed: `s03:` is `~/dev/AGL/basodino/igm/`, `s02:` is `/data/befahl/projects/AGL/basodino/`. Target values and
files: `TARGETS.md`; software, versions and the benchmark: `INSTALL.md`.

## Goal

About 10 000 IGM runs of Ghiacciaio del Basòdino that give the distributions of the half-volume year and of the year
of disappearance, maps of the year each cell becomes ice-free, and one animation. Start from the measured thickness,
calibrate on 1991-2026 data, validate with the 1850-2026 hindcast, force with every CH2025 chain for Locarno-Monti
(OTL) to 2100. Every member also runs without ice flow, so the effect of flow is measured.

## Decisions

1. Solver: IGM 3.2.0 (s03 env), online DahuNet solver, one run per process, cfl 0.3 at percentile 100. Steps 4 and 6
   warm-start from a network trained once per starting geometry (per DEM, per posterior draw) and retrain 5 iterations
   every 5 steps; spin-ups start cold.
2. Grid: 10 m, one domain, E 2678700-2682100, N 1140000-1142900 (340 x 290 cells) on the course-grid lattice.
   Positive balance only inside SGI1850 and on the cells of its 100 m buffer flatter than 30° and outside the 1850
   outline of C14-11.
3. Mass balance, `smb.py` (one function with a member axis) and IGM module `smb_ti`: the equations of the reference
   `smb_accmelt` (Hock 1999; Huss et al. 2009): melt = (Fm + r I_pot) T+, r_snow = 0.75 r_ice, snow below 0.5 °C,
   rain above 2.5 °C, snow store reset on 1 October, no refreezing; accumulation times D (4); no accumulation or
   ablation weights. Timing differs from the reference: `clim_otl` and `smb_ti` first update at the start (last update
   = start - 1, so the +1.10 m placeholder never acts); arrays are hydrological years from `weather.py` (shift 0); at
   t = Y.75 hydrological year Y+1 is loaded. Each year is computed on 1 October on the current surface and applied as a
   rate; areas at the outline dates use the daily balance accumulated since 1 October instead.
4. Snow redistribution: D = exp(k κ) s, κ the curvature of the 2 m surface smoothed over 50 m, s = 1 below 30° slope
   and linear to 0 at 60°; k from a least-squares fit of log(probed / modelled winter balance, D = 1) on κ over the
   1127 GLAMOS winter probings, one intercept per year; D scaled to mean 1 on the 2021 glacier. IGM's avalanche
   process stays off (one MAP spin-up with it, as a check).
5. Station transfer, measured once and fixed: T(z) = T_OTL + g_m (z - 367 m), g_m monthly from ROE - OTL daily means
   1991-2025; P(z) = c_prec P_OTL (1 + b (z - 367 m)), b = 2.0 % per 100 m from ROE/OTL 1991-2020.
6. Observed weather, hydrological years 1851-2026: OTL homogenised daily T from 1882-12 (measured 2025-26 shifted per
   month to the homogenised monthly means), measured daily P from 1900-10 scaled to the homogenised monthly totals.
   Earlier: monthly anomalies regressed on OTL over 1883-1930, T from Lugano (from 1864-01) and Grand St-Bernard
   (1850-1863), P from Lugano (before 1864 the months of a random year 1865-1894). A month without measured daily
   values takes the daily anomalies of the same month in a random year 1901-1930 (T added, P scaled). No 29 February.
7. Balance calibration: sampling-importance-resampling, 2^17 Sobol draws, uniform priors Fm 0-6 mm ice d-1 °C-1,
   r_ice 0-0.06 mm ice m2 W-1 d-1 °C-1, c_prec 0.5-3, log-uniform model-error SDs 0.05-1 m w.e. per term; Gaussian
   likelihood, each value summed day by day between its survey dates: (a) GLAMOS observation-period band Bw (100 m);
   (b) band Ba minus the glacier-wide mean Ba of its geodetic period; (c) the 344 stake balances minus their year's
   mean (point release 2025); (d) geodetic volume change in km3 between 1991-09-18, 2002-09-16, 2008-08-29,
   2013-08-21, 2018 and 2024-08-23 (GLAMOS sigma times area). Geometry from the surfaces 1991, 2012, 2021 (linear
   between) and the nearest outline; 1024 draws kept. If r_ice/Fm stays as wide as its prior, it is fixed at the
   Aletsch reference ratio (0.012 m2 W-1) and the report says so. Fixed-date values are shown, not fitted.
8. Start thickness: bed(u) = Grab bed - u σ w, σ the magnitude of the plus (u > 0) or minus (u < 0) uncertainty grid,
   w = min(1, d / 50 m), d the distance inside the 2021 outline (no change at the outline). On the 2021 mask the bed
   lies at least 1 m below the 2021 surface; off it the bed is the ground. Projections start on 2021-10-01 from the
   2021 DEM.
9. Start state and flow, one joint fit: 1024 samples of (posterior balance draw, u uniform in [-1, 1], f log-uniform
   in [0.05, 4]); A = 78 f MPa-3 a-1, Weertman tau_ref = 0.2154 f^(-1/3) MPa at 100 m a-1 (speeds scale with f).
   Each sample runs 1991-10 to 2021-09 from the course DEM labelled 1994 (volume change, offset to 2012 and mask match
   the GLAMOS survey of 1991-09-18) and 2021-10 to 2026-09 from the 2021 DEM, observed weather. Likelihood, each term
   with its sigma plus a log-uniform model-error SD integrated out per sample: areas by each target's definition at its
   date (2021 mask; 2024-08-23 all polygons; 2025-09-07 and 2026-09-27 main body; sigma the ±10 m sensitivity); dh
   per 50 m band 1991-2012 and 2012-2021; the 2006 radar as six profile means (sigma 5.3 m each, independent; points
   whose radar bed lies above the 2021 surface left out); ITS_LIVE as a one-sided bound (median 2014-2021 surface speed
   on the 96 glacier pixels of 120 m ≤ 1 m a-1, sigma 0.5 m a-1 above). Ice threshold 1 m; u's posterior also for
   0.1 and 2 m from the saved fields. Check: the 128 posterior draws rerun from 1994-10 and are re-weighted.
10. Spin-up: from ice-free to steady state under the hydrological years 1865-1894 of (6) in a fixed cycle plus
    dT_1850; steady when the mean volume of the last 30-yr cycle is within 1 % of the one before; secant on dT
    (≤ 5 runs of ≤ 300 yr) to the cycle-mean ice area inside SGI1850, 4.008 ± 0.25 km2 (perimeter 9.06 km x 28 m,
    the map-outline NMAD of Mannerfelt et al. 2022; the inventory gives none); ice outside it reported. The hindcast
    then runs the forcing from 1850-10.
11. Hindcast 1850-10 to 2026-09 for the 128 posterior draws, validation only: areas 1929, 1973, 2010, 2018; volume
    change in km3 1907-1929, 1929-1949, 1949-1961, 1961-1971, 1971-1985, 1985-1991 (survey dates); GLAMOS tongue
    elevation 2000-2024; length change 1899-2024 along the SGI2016 centreline continued down the 1850 flowline;
    surfaces 1991, 2012, 2021. Stated before the runs: with constant melt factors the model should lose too little in
    1929-1949 (1940s radiation high) and too much in 1949-1985 (dimming) (Huss et al. 2009).
12. Scenarios: CH2025 DAILY-LOCAL, 26 chains (24 at GWL3.0); monthly dT (added) and P ratio of each warming-level
    window against the chain's own 1991-2020, for OTL and for ROE; global warming per SSP piecewise linear through
    +0.9 °C in 1991-2020 and the AR6 SPM.1 best estimates for 2021-40, 2041-60, 2081-2100 (SSP1-2.6 1.5, 1.7, 1.8;
    SSP2-4.5 1.5, 2.0, 2.7; SSP5-8.5 1.6, 2.4, 4.4 °C); linear in warming between windows, beyond 3.0 extrapolated
    from 2.5-3.0 (2.0-2.5 for the two chains without GWL3.0); applied from 2026-10 to the OTL days of 1991-2026 (T
    detrended to the 1991-2020 mean), year order drawn per member. Two 2026 levels: native (JJA +0.87 °C over
    1991-2020, +1.02 under SSP5-8.5) and anchored (constant offset per chain and season that puts 2026 on the observed
    1991-2026 OTL trend, JJA +1.48, annual +1.00 °C).
13. Ensemble: 128 posterior draws (balance, u, f) x 26 chains x 3 SSPs = 9 984 members with their own weather draws;
    2026 level (native, anchored) and station changes (OTL, ROE) assigned by draw, balanced (32 draws, 2 496 members
    each); the same draws make the hindcast. All members also run without flow (`noflow.py`: per-cell balance and
    thinning). The MAP draw's 78 forcings also run with u = 0.
14. Definitions: ice = thk ≥ 1 m on 30 September; half-volume year = first year with V ≤ V_2026/2, and with
    V ≤ V_2021/2 (assignment 5.2; set by u and the observed 2021-2026 weather, so shown against u, not per SSP);
    disappearance = first year with ice area < 0.01 km2 and no later return, also at 0.1 km2; ice-free year of a cell
    = 1 + its last year with ice; tongue = lowest ice cell of the main body; mass with 850 kg m-3. A run stops when no
    cell holds ice, at the latest on 2100-09-30.

## Inputs

| Input | Path |
|---|---|
| Course DEMs and masks "1994" (survey 1991-09-18, see 9), 2021 | `../data/course/` |
| Thickness, bed, uncertainty grids, 2012 surface (Grab et al. 2021); radar 2006 | `../data/thickness/` |
| Outlines SGI1850 (C14-10, C14-11) to SGI2016 with centreline, swissTLM3D 2024; Sentinel-2 2025, 2026 | `../data/vectors.gpkg`; `../code/outline_2026.csv` |
| swissALTI3D 2 m; Copernicus GLO-30 for the Italian gaps (horizon only) | `s02:derived/dem_2m.tif`; `../data/italy/src/copdem/` |
| GLAMOS observation-period and band balances, volume change (all pairs), length change, tongue elevation | `../data/glamos_basodino_*.csv` |
| GLAMOS point balances 1992-2025 (stakes, probings; LV03) | fetch (1.5 MB) to `s02:raw/glamos/` |
| OTL and ROE daily, OTL monthly homogenised; ITS_LIVE bound | `s02:raw/climate/obs/`; `TARGETS.md` section 7 |
| Lugano (T, P from 1864-01), Grand St-Bernard (T from 1818-01), monthly homogenised | fetch ogd-nbcn `lug`, `gsb` to `s02:raw/climate/obs/` |
| CH2025 DAILY-LOCAL tas, pr for OTL and ROE | `s02:raw/climate/ch2025/` |
| Reference `smb_accmelt`, `clim_aletsch`; reference L-BFGS solver | `s03:igm-examples/aletsch/user/`; `bench/experiment/reference.yaml` |

## Layout

- Mac, git (the author commits): `inputs.py`, `ipot.py`, `redist.py`, `weather.py`, `scenarios.py`, `smb.py`,
  `calib_smb.py`, `noflow.py`, `user/` (`clim_otl`, `smb_ti`, `track`), `experiment/` (`fit`, `spinup`, `hindcast`,
  `projection`), `members.py`, `weights.py`, `run_stage.sh` (pinning, thread limits, nice 10, nrun), `collect.py`,
  `figures.py`, `animate.py`; small results in `results/`.
- s02: inputs built there (agl env), master `igm/inputs/`; results `igm/results/<stage>/`; `igm/runs_<stage>.tar.gz`.
- s03: env, `inputs/`, code by rsync from the Mac (never edited there), `nets/`, `runs/<stage>/<member>/`, `logs/`;
  run folders deleted once banked on s02; `tmp/` emptied after each stage.

## Steps and gates

1. Inputs, forcing, D, `smb.py` (Mac, s02). Gate: `smb.py` given the Aletsch inputs (snow_redistribution, weights
   1.84 and 2.01, calendar alignment) reproduces `smb_accmelt` to 1e-5 m (arithmetic only); I_pot of a flat cell within
   1 % of the analytic clear-sky value; forcing complete 1850-10 to 2026-09, no evenly spread month; Lugano and
   St-Bernard fits per month reported (slope, r, RMSE of the rebuilt OTL 1883-1930); scenario minus observed OTL for
   2016-2025 and 2021-2025, annual and JJA, reported for both levels, anchored 2026 within 0.05 °C of the trend; chain
   changes reproduce `s02:derived/climate_summary.csv`; k and the probing variance D explains reported.
2. Balance calibration. Gate: effective sample size ≥ 1000 (else a second round inside the first posterior's range);
   the 90 % predictive band holds ≥ 80 % of the band Bw and Ba values and all five volume changes; the MAP draw has
   its lowest Ba in 2022 and a positive one in 2001. Else a temperature offset as a fourth parameter, repeat.
3. IGM modules and checks (s03). First, before any other batch: the Zn ph.x iteration rate sampled every 30 s
   (read-only, as `Zn/mesh48g/campaign_rate.sh`), 20 min alone, 10 min each beside 6 and 8 benchmark runs per GPU,
   20 min after; the slowdown goes to question 1. Gates: (a) the MAP draw 1991-2026 in IGM gives `smb.py`'s yearly Ba
   to 1e-4 m w.e., 2022 lowest, 2001 positive, no thickness gain on ice-free cells in year one; (b) online against the
   reference L-BFGS solve at 10 m on the MAP spin-up end state, 1991 and 2021 at u_MAP: median and p90 surface speed
   within 10 % (else more training iterations, re-timed), warm start within 10 % of cold; (c) timing at 6 runs per GPU,
   pinned, with one 300-yr spin-up from ice-free: within 1.5 times the table, ≤ 3 GB per run; (d) IGM area 2021-2026
   within 0.02 km2 of `noflow.py`; clean stop.
4. Start state and flow (9): 2 x 1024 runs, then 128 from 1994-10. Gate: effective sample size ≥ 128, else a second
   round of 1024 from a proposal fitted to the weighted first round (weights corrected), else the 128 draws resampled
   from all weighted samples with the ESS stated; u not piled at ±1 (else ±2, flagged), f not piled at 0.05; a flat f
   posterior stays sampled and its share of the spread is reported; each term's fit reported.
5. Spin-up and hindcast, 128 draws. Gate: every spin-up steady and within the 1850 tolerance; validation table (inside
   or outside the 5-95 % band, expected misses marked) reported as it comes.
6. Projections: 128 network trainings, 9 984 + 78 runs, 9 984 no-flow runs. Gate: every run ended (no ice, or 2100),
   no NaN; ice-free grids complete; 4 members rerun give the same disappearance year.
7. `collect.py`, figures, animation; bank to s02; delete the s03 run folders.

## Cost

Measured 2026-10-06 under the Zn QE load (`bench/results.csv`): online solver, 10 m, 8 runs per H200, 0.058 s per
step and run, 53 s cold start-up; daily balance 0.37-0.55 s per simulated year and run. Planned per run and year:
1.1 s in spin-up and hindcast, 1.0 s in 1991-2021, 0.7 s after 2021 (cfl 0.3 at 10 m allows a 1-yr step only below
3 m a-1; at f = 1 the 2021 maximum is 7-13 m a-1 for u ≥ 0.5, so up to 4 steps a year); 25 s warm start-up (assumed,
measured in 3c). 16 runs at once, 8 per H200 (answer 1).

| Step | Runs | Years per run | Process-h | Wall | Card-h, 2 shared H200 |
|---|---|---|---|---|---|
| 2 Balance calibration | 2^17 draws, no IGM | 34 | 0.5 | 0.5 h, one card | 0.5 |
| 3 Checks, timing, Zn measurement | about 40 | 5-300 | 10 | 2 h | 4 |
| 4 Start state and flow | 2 x 1 024 + 128 (fallback + 2 x 1 024) | 30 + 5; 27 | 26 (+ 24) | 2 h (+ 2 h) | 4 (+ 4) |
| 5 Spin-up and hindcast | 128 x ≤ 6 | ≤ 1 500 + 176 | 77 | 6.5 h | 13 |
| 6 Projections, trainings, reruns | 10 194 | 5 + up to 74 | 121 (stop at no ice, about 25 yr) to 226 (all to 2100) | 10-19 h | 20-38 |
| 6 No-flow runs | 9 984, batched | 79 | 1 | 1 h, one card | 1 |
| Total | about 13 200 | | 235-340 (+ 24) | 22-31 h (+ 2 h) | 43-61 (+ 4) |

Share: at most 12 IGM processes, 6 per H200 (≤ 3 GB each, ≤ 18 GB of the 105 GB free per card), ordinary CUDA
contexts time-sliced with the Zn MPS server, never on its pipe; pinned with taskset to the 12 hardware threads no
ph.x mask covers (0-5 and 64-69, 6 cores, on 6 Oct; re-checked before each stage); TF_NUM_INTRAOP_THREADS,
TF_NUM_INTEROP_THREADS, OMP_NUM_THREADS = 1; nice 10. Pinning protects the CPU side only; step 3 prices the GPU share.
Disk: s02 ≤ 5 GB; s03 run folders deleted after banking. This revision used no GPU.

## Schedule

| Date | Step |
|---|---|
| Tue 6 Oct | design, review, answers to questions 2-5 |
| Wed 7 | Zn measurement (1 h), answer to question 1; 1 inputs, forcing 1850-2026, D, `smb.py` |
| Thu 8 | 2 balance calibration; 3 modules and checks |
| Fri 9 | 4 start state and flow (2 h; a fallback round overnight) |
| Sat 10 | 5 spin-up and hindcast (6.5 h); 6 projections start in the evening |
| Sun 11 | 6 projections end (10-19 h), no-flow runs, checks |
| Mon 12 | 7 collect, figures, animation, banking |
| Tue 13 | results to the author |
| Wed 14 to Sat 17 | buffer: one projection rerun fits in a night and a day |
| Wed 21, 08:00 | report deadline |

## Outputs

- `members.csv` (draw, u, f, chain, SSP, 2026 level, station changes, seed, V_2021, V_2026, both half-volume years,
  disappearance at 0.01 and 0.1 km2, the same without flow); `series.nc` (yearly V, A, tongue, length, Bw, Bs, Ba);
  `icefree.nc` (ice-free year per cell and member, int16); `hindcast_fields.nc` (thk at the target dates).
- Figures (style of `code/common.py`, PDF to `tex/figures/`): hindcast against observations, expected misses marked;
  calibration (predictive bands, posteriors, speed against the bound); V and A 2021-2100 per SSP and 2026 level; both
  years per SSP with a quantile table, the share of their spread per factor and IGM minus no-flow; half-volume year
  against V_2021 as a function of u; median ice-free year per cell and its 90 % range with the 1850-2026 outlines.
- Animation 1850-2100 (MP4): the draw with the median disappearance year under SSP2-4.5 (2026 level per question 2).

## Risks

- u sets the headline numbers (V_2021 0.0023-0.0323 km3 over u = -1 to 1). With w zero at the outline, slow area
  loss from causes without a parameter (shading and avalanches within a band, thin or dead ice and snow in the mapped
  area) stays a misfit, carried by the area model-error SD and reported.
- u, f and the balance gradient trade off (speed scales with f H^3 to H^4; dh per band also absorbs gradient errors);
  the joint fit, the bands and the stakes limit it; posterior correlations reported.
- Climate: observed OTL warming is about 2.3 times global, the chains about 1.5 times; native 2026 lies 0.46-0.61 °C
  (JJA) below the observed trend. Before 2040 the SSPs differ by ≤ 0.5 °C in JJA, so u and the 2026 level should
  dominate the spread. Not propagated: g_m and b; constant melt factors 1850-2100 (the Aletsch reference changes Fm
  within 1.67-2.31e-3 between periods); I_pot on the 2021 surface.
- Hindcast: 1850 taken as steady; before 1864 the forcing rests on Grand St-Bernard T and resampled precipitation;
  ±0.25 km2 is a floor for the SGI1850 error.
- Solver: online against reference on three states only (3b); at cfl percentile 100 the fastest cell sets dt, and
  7.6 % of SGI1850 + 100 m is steeper than 45°: the main cost risk of the spin-up. s03 on 6 Oct, 19:29: load 106 of
  128, both GPUs at 100 %, swap full.
- Per-cell map, the weakest output: redistribution from one curvature term and a slope cut, debris (0.004 km2) not
  modelled; hence disappearance also at 0.1 km2.

## Answers of the author (2026-10-07)

1. s03 share: 8 IGM processes per H200, 16 at once. Pinned to the 12 hardware threads no ph.x mask covers (0-5,
   64-69, re-checked before each stage) plus 4 more (6-7, 70-71), which the Zn jobs also use; the author accepts the
   larger Zn slowdown. No dedicated measurement batches: the Zn ph.x iteration rate is sampled read-only every 30 s
   (as `Zn/mesh48g/campaign_rate.sh`) from 20 min before the first IGM batch through every stage, and the slowdown
   is reported per stage.
2. The anchored 2026 level leads the report; native is computed and shown beside it (64 draws each).
3. The half-volume year against V_2026, per SSP, leads; the one against V_2021 is shown as a function of u.
4. The report keeps the course date October 1994 (Task 1, 27 years); the ensemble treats that DEM as the GLAMOS
   survey of 1991-09-18 (decision 9).
5. CH2018 request: not sent (not needed by the design).

## Technical settings fixed after step 1 (2026-10-07, consistency, not new design)

- smb gate: 1e-5 m on ice-mask cells in float64, the S == 0 knife-edge switches listed; the reference's own float32
  rounding is out of reach by construction.
- Annual anchor of decision 12 in hydrological years 1991-2026 (+1.12 °C), like the forcing and the JJA anchor (+1.48).
- P before 1864: the 14 years drawn without replacement from 1865-1894 (rule fixed before seeing the result).
- Homogenised daily T 1882-12 to 2024-12 shifted per month onto the homogenised monthly means, as for 2025-2026.
- s02 scripts/fetch_climate.sh gets a home in the Mac repo (igm/fetch_climate.sh), rsynced to s02; the .part download stays.
- The start-thickness prior (decision 8, u in [-1, 1] with the 50 m taper) waits for the mass-conservation test
  (code/area_clock.csv: the observed 2025-2026 areas need u = 0.41-0.50, 2024 up to 1.0 with the taper); decided by
  the author before step 4.
- Step 2 (after the first calibration failed its gate, 2026-10-07): term (a) compares the GLAMOS winter balance with the
  modelled snow store at date_end_winter (snow since the first lasting snow), which is what the probings measure; the
  earlier sum of all balance from date_start included autumn ice melt the probes never see and pulled the melt down.
  r_ice piled at its prior edge 0 counts as unconstrained: r_ice/Fm fixed at the Aletsch ratio 0.012 (decision 7
  fallback). Up to three importance rounds. Temperature offset only if the three-parameter run still fails, prior
  uniform -3 to +3 K. smb.py float64 TensorFlow path repaired, checks/smb_reference.py rerun.
- Scenarios: beyond GWL3.0 the precipitation ratio is extrapolated in log space (stays positive); SSP1-2.6 and
  SSP2-4.5 are unchanged by this (all their ratios lie inside the published levels or are positive already).
- Step 2, third attempt (author's answer 2026-10-07: 'Free both gradients'): decision 5's gradients become calibration
  parameters. Temperature: T(z) = T_OTL + (g_m + dg)(z - 367 m), dg uniform -3 to +3 K per km added to the monthly
  Robiei-Locarno gradients. Precipitation: b uniform -2 to +10 % per 100 m. Free: Fm, c_prec, dg, b (r_ice = 0.012 Fm);
  the temperature offset dT only if this still fails. Same terms, gate and rounds as before.
- Author's answers 2026-10-07 (afternoon): (1) 'Pause Mg GPU ph.x': before the first IGM batch on the H200s, run
  igm/pause_mg.sh stop on s03 (SIGSTOP of the 48 GPU ph.x of the Mg mesh run only; CPU ph.x, Zn drivers and the MPS
  server untouched); igm/pause_mg.sh cont after the last IGM GPU batch of the day or when the ensemble ends; check
  'status' shows T while paused and R/S after. The IGM share stays 8 runs per H200. The Zn rate sampler of answer 1
  samples whatever campaign holds the GPUs (Mg now) and records the pause. (2) 'Widen and add the areas': start
  thickness u uniform in -1 to 2, the 50 m edge taper kept; the start-state likelihood (decision 9) gets, besides the
  areas it already has (2024-08-23, 2025-09-07, 2026-09-27), the GLAMOS stake bounds of code/stake_clock.csv as
  one-sided terms (modelled 2021 thickness within 30 m at least the melt since September 2021 at sites still in ice;
  the 2025/26 values provisional).
- Step 2, fourth attempt (author's answer 2026-10-07: 'Volume changes binding'): the five geodetic volume changes
  enter with their GLAMOS sigma only (no free model-error term for term d); the free gradients dg and b stay, with Fm
  and c_prec (r_ice = 0.012 Fm); dT only if needed. Gate as before; the resulting band, stake and winter misfit is
  reported, not hidden. Results of the third attempt: results/calib_*_grad.csv (Fm, c_prec, dg, b) and
  calib_*_grad_dT.csv (with dT); run as `calib_smb.py run 2 bind` (with dT: `run 2 dT bind`).
- Author's answer 2026-10-08: 'Accept, go to step 3'. The posterior of the fourth attempt with dT (results/calib_*.csv:
  Fm, c_prec, dT, dg, b; term d binding) is used for the ensemble although the step-2 gate is not met: 3 of 5 volume
  changes inside (1991-2002 and 2002-2008 lose too little), glacier-wide mean Ba -0.55 against -0.84 m w.e. a-1, snow
  left on 74 per cent of the glacier on 30 September against a GLAMOS AAR of 21 per cent, dT near its -3 K edge. The
  report states this misfit. Mg GPU ph.x resumed ('Resume until IGM starts'); pause_mg.sh stop right before the first
  IGM batch.
- Step 3 (2026-10-08), modules and runs: `pack_inputs.py` (s02) packs the step-1 grids and the observed forcing into
  `inputs/igm_static.npz` and `inputs/forcing_obs.npz`; IGM user modules `geometry` (input: start state of decision
  8, dem 1994 | 2021 | none | restart), `clim_otl`, `smb_ti`, `warmstart`, `track` (output, at the save hook);
  `noflow.py`; `run_stage.sh` (16 slots, slot s on thread 0-5, 64-69, 6-7, 70-71 and GPU s % 2, nice 10, one TF and
  OpenMP thread); `rate_sampler.sh`; experiments `fit`, `spinup`, `reference`, `flowoff`; run lists in `stages/`;
  gates by `checks/step3.py` (results/step3_*.csv). The smb update is due when the hydrological year of t differs
  from the one loaded last (equal to t - t_last >= 1 on Y.75; cannot miss a year by float32 rounding). The transfer
  runs as one fused XLA kernel (no float64 day x cell arrays; peak GPU memory 2724 instead of 4772 MiB). Areas at an
  outline date: thk(1 October) + the year's flow change spread evenly + the balance since 1 October / 0.91.
- Step 3, solver (gate 3b, 'else more training iterations'): cold start 3000 initial iterations (300 left p90 surface
  speed 20 % and 14 % below the reference on 1991 and 2021); warm start = weights of a cold run (input normaliser
  recomputed for the run) plus 300 iterations for another (u, f), 0 for the network's own start state (5 iterations
  with a fresh Adam state moved speeds by 40-77 %). IGM's own pretrained path is not usable for this (frozen
  normaliser; its artifact wrapper rejects a network trained online). Gate 3b used u = 0, f = 1 (u_MAP comes with
  step 4).
- Correction 2026-10-08: the 48 GPU ph.x ranks on s03 are the author's Zn mesh48g GPU stream, which runs the ph.x
  binary compiled under /scratch/befahl/Mg/dfpt_qe/q-e-qe-7.5; there is no Mg job. Every 'Mg GPU ph.x' above means
  this Zn GPU stream. The stream (driver, nrun, mpirun, ranks) was stopped from 2026-10-07 07:35 before any IGM work;
  a CONT of only the ranks (08:22 and 09:18 on 8 October) left them spinning without their stopped parents and slowed
  the Zn CPU campaign, so they were stopped again at 09:37. Rule from now on: the ranks stay stopped during the
  ensemble; resume the whole stream only when the author says so. Script renamed igm/pause_zn_gpu.sh.
- Step 4 (2026-10-08), `weights.py`, `checks/step4.py`: round 1 (s4_r1, 2048 runs; 3 reruns s4_r1fix after the smb
  year-switch fix, control rerun within solver noise) gave an effective sample size of 1.6, so round 2 ran (s4_r2,
  2048 runs, 0 failed; proposal in (u, ln f): mean 0.120, 1.221, SD 0.312, 0.549, correlation -0.22, round-1 weights
  tempered by 0.381, mixed 0.9 / 0.1 with the prior; balance draws a new permutation, not adapted). Pooled (balance
  heuristic) effective sample size 8.6 < 128: last fallback, `results/posterior_128.csv` resampled from all 2048
  weighted samples (34 distinct samples, largest weight 0.28), effective sample size stated there. Gate outcome,
  per-term fit and variants: `results/step4_summary.csv`, `step4_terms.csv`, `step4_weights.csv`.
- Check of decision 9 (stage s4_d94): every one of the 128 draws rerun from 1994-10 (copies too, so the 128 runs also
  give the online solver's run-to-run spread); re-weighting in `results/step4_check94.csv`.
- Independent check of step 4 (2026-10-08 evening, `checks/step4_verify.py`, results/step4_verify*.csv): weights,
  effective sample size and the per-term fit reproduce; model values of the MAP and 5 random kept samples rebuilt from
  their fields.npz agree with the reduce to 5e-4 m. Defect fixed: the round-2 sampler redraws Gaussian tries that fall
  outside the box, so its density is (0.9 g 1_box + 0.1 p) / (0.9 Z + 0.1), not 0.9 g / Z + 0.1 p (prior share 0.152
  of the kept samples, not 0.1); `weights.py` log_q corrected and weigh rerun (effective sample size 8.6 unchanged,
  largest weight 0.2817; draw 1 of posterior_128.csv moved from r1_0145 to r1_0173). Its October-1994 run is stage
  s4_d94fix (stages/step4_d94fix.txt; banked runs_s4_d94fix.tar.gz); check94 reads s4_d94 and s4_d94fix.
- Step 5 (2026-10-08 evening), `hindcast.py`, `targets_step5.py` (s02), experiments `spinup` (per draw) and
  `hindcast`; `track` gains the ice area inside and outside SGI1850, the length along a flowline (SGI2016 centreline
  of C14-10 continued by steepest descent on the 30 m-smoothed 2021 ground inside SGI1850 to its lowest cell, 3037 m),
  thk every 10 years, and `stop_no_ice` (false for spin-ups: the first scan stopped every run above +0.4 K after
  one year because one year of snow stayed below 1 m). Scan of the largest-weight sample (s5_scan2, -1.4 to +1.6 K):
  the cycle-mean ice area inside SGI1850 cannot exceed the outline's 4.008 km2 and saturates (3.76 at 0.0 K, 3.96 at
  -0.2 K, 4.005 at -1.4 K) while ice outside grows (1.0, 1.5, 2.7 km2, mostly across the north-western divide);
  so the secant aims at 4.008 and accepts a draw's first steady run within 0.25 km2 (round 1 at the scan's
  A_in = 3.883 km2, -0.123 K; secant steps of at most 1 K), and the ice outside is reported with every result.
  Hindcast cold (3000 iterations) from the accepted spin-up end state, observed weather without offset.
  Validation dates: unknown survey days 1907 -> 1907-10-01, 1929 -> 1929-08-13 (SGI1931 image date); inventory years
  1973, 2010 (2008-2011), 2018 -> the 1 October save; tongue and length at the 1 October save nearest the
  observation; 1949-1961 volume change from the all-pairs file (-0.0087 km3; the online series file gives -0.0051).
  Expected misses written to results/step5_targets.csv before the first hindcast run.
  Outputs: results/step5_spinup.csv, step5_secant.csv, step5_validation.csv, step5_series_s5_hind.csv (every save of
  every draw); per draw thk at the validation dates, every 10 years and at the end in the run folders, banked as
  s02 igm/runs_s5.tar.gz (deleted on s03); inputs/step5_state2026.npz (s02 and s03): per draw the 2026-09-30 state of
  the hindcast and of the decision-8 path (the step-4 run of the draw's sample from the 2021 DEM), which step 6 uses
  per decision 8.
- Step 6 (2026-10-08 night), `projection.py` (pack on s02; members on the Mac; starts, reduce, noflow, gate on s03),
  experiment `projection`, `checks/step6.py`. Scenario weather (decision 12): the OTL days of 1991-2026 detrended per
  season (DJF, MAM, JJA, SON of the hydrological year) by the least-squares line through the 36 season means, the line
  of the anchor in scenarios.py (T_det = T - slope (Y - 2005.5); 1991-2020 mean unchanged; JJA line +1.474 °C at 2026,
  annual +1.117); P not detrended; year order per member: permutations of 1991-2026 laid end to end, first 74
  (numpy default_rng(2100000 + member index)); dT added and P ratio multiplied per month of the day, from the draw's
  station (OTL or ROE) applied to the OTL days (inputs/forcing_scen.npz, results/step6_forcing_check.csv). clim_otl
  takes it with `scenario.file` (station, chain, ssp, level, seed); noflow.py with the same columns and `restart`.
  Assignment: draws sorted by (sample, draw) take (anchored, OTL), (anchored, ROE), (native, OTL), (native, ROE) in
  turn (32 draws each; the copies of a sample spread over the four). Start: per draw the 2026-09-30 state of the
  decision-8 path (inputs/start2026/dNNN.npz from step5_state2026.npz). MAP draw: draw 80 (sample r2_0936, the largest
  weight; the lowest-numbered copy with the anchored level, OTL); its u = 0 start: stage s6_prep (that sample's
  2021-2026 run with u = 0). Networks: stage s6_nets, one cold training (3000 iterations) per start state (129);
  members warm-start with 0 iterations from their start state's network. Film members (track yearly): draw 80 under
  each SSP with the chain whose 2081-2100 annual dT lies closest to the 26-chain median (SSP1-2.6 MPICSC-REMO2-MPIESM,
  SSP2-4.5 CNRM-ALADIN-HADGEM, SSP5-8.5 MOHC-HADREM-CNRM). Projection fields.npz without topg (track topg: false;
  the start file holds it). run_stage.sh gains TIMEOUT (s): training n_d028 hung asleep for 25 min (futex wait,
  CPU time constant, GPU idle), was killed and rerun (stage s6_nets_fix); s6_proj runs with TIMEOUT=1200. Rerun gate:
  stage s6_rerun (the SSP2-4.5 film member and three random members).
- Step 7 (2026-10-09), `collect.py` (s02; h5netcdf 1.8.1 added to the agl env for the netCDF-4 files), `figures.py`
  (Mac), `animate.py`, `film3d.py`, `checks/step7_bank.py`; results in `RESULTS.md`. Collect reads only banked tars
  and step-6 reductions (the 2021-2026 runs of the samples from runs_s4_rounds.tar.gz, the hindcast fields from
  runs_s5.tar.gz). No-flow years are reported as reduced and with the IGM stop rule applied to the no-flow series
  (the step-6 check found 4340 regrowing twins). Spread per factor: exact three-way decomposition of the balanced
  draw x chain x SSP crossing, the draw part split into 2026 level, station and the draws within. Film member
  (Outputs: 'the draw with the median disappearance year under SSP2-4.5', anchored): of the 1664 anchored SSP2-4.5
  members those with the median year (2034), the one closest to the group's median area series (d056_c07_2; the
  step-6 yearly members are the MAP draw, 2032 under SSP2-4.5); its hindcast, 2021 run and projection rerun with
  yearly fields (stage s7_film; the cold-started two agree with the banked runs to solver noise,
  results/step7_film_check.csv). Map film 1850-2100 at 24 frames per model year; 3D film in Blender Cycles on the
  s02 A6000 and both s01 GV100 (s03 not used), modelled ice as a mesh over the bed (2021 surface minus the member's
  2021 thickness), no image under today's ice of the glacier, fixed grade from render_views.expose on 8 frames.
- Step 7 review (2026-10-09): in both films the last hindcast year (2020-2021) now runs to the hindcast's own 2021
  state (it had run to the start state from the measured thickness and then jumped back for the 2021 hold); map film
  re-rendered, 3D render frames 2551-2564 redone (the replaced ones kept in s02 scratch film3d_frames_superseded/),
  3D film recomposed; the 3D status line says 'Hindcast' while the hindcast's 2021 state is held; the map end card
  fades in from the map; both end cards state the hindcast area misfit as measured (above every inventory 1929-2018).
  Hindcast figure d draws the length band only while every draw has ice on the flowline (to 2022). Frame steps of
  both films: `checks/film_steps.py` -> results/step7_film_steps.csv.

## Fix round (author's answer 2026-10-09: 'Full fix incl. hindcast')

The first full ensemble (results/RESULTS.md, 9 October) failed two ways: (1) the 2026 start states are far too small
(0.27-0.44 km2 against 1.02 km2 mapped), because step 4 traded the areas and stakes against the dh bands and radar
(ESS 8.6, f at its upper edge, speeds above the ITS_LIVE bound); (2) the hindcast does not retreat before about 1990
(5.0-5.7 km2 against 3.20 km2 in 1929 and 2.32 km2 in 1973; 1907-1991 volume change +0.02 against -0.164 km3; ice
reaching the domain edge), because the balance before 1990 is too positive and nothing keeps ice in the basin.
The fix, in order, each step gated as before:
- F1 Balance over the whole record: the calibration adds the GLAMOS geodetic volume changes 1907-1929, 1929-1949,
  1949-1961, 1961-1971, 1971-1985 and 1985-1991 (in km3, survey dates, geometry from the inventory outlines and the
  nearest surface) to the 1991-2025 terms. If one stationary parameter set cannot fit both periods, a temperature
  offset for the years before 1991 (the station transfer from Lugano and Grand St-Bernard) is added and reported.
- F2 Basin: positive balance and ice only inside the SGI1850 outline of C14-10 plus 100 m (decision 2); cells
  outside it melt any ice that flows in; C14-11 excluded. The spin-up must keep its ice inside the domain.
- F3 Start state: fitted on 2021-2026 from the 2021 surface (areas 2024, 2025, 2026 and the stake bounds; the dh
  bands and radar enter with their model-error SD, not as hard constraints); f log-uniform 0.05-20 with the ITS_LIVE
  bound kept; the posterior must reproduce the observed 2026 main body (1.02 km2) within its spread.
- Then steps 5-7 again (spin-up and hindcast with F1 and F2, projections from the F3 states, figures, films, review).
- Author's rule for every film and time-axis figure (2026-10-09): stop 2-3 years after the glacier is gone (end the
  time axis or the film at the disappearance year of the shown member, or of the latest scenario median shown, plus
  3 years); no empty years to 2100. Ensembles whose members last longer show their own spread up to their latest
  95th-percentile disappearance year plus 3 years.
- F1 settings and result (2026-10-09; `calib_smb.py`, `preview_calib.py`, `checks/f1_station_trend.py`):
  - Targets: term d holds 11 GLAMOS volume changes in km3, the six pairs of `targets_step5.PAIRS` (all-pairs file;
    1907 -> 1907-10-01, 1929 -> 1929-08-13; 1949-1961 the all-pairs -0.0087 km3 as in step 5, the series file's -0.0051
    lies 1.0 sigma away) and the five of 1991-2024; sigma_obs = |dV| sigma_B / |B_geod| for every value (GLAMOS's own
    area normalisation inverted, so the 1946 outline of 1929-1949 and 1949-1961 cancels; no B_geod value enters);
    binding (sigma only) as in step 2. Hydrological years 1908-2025.
  - Geometry up to 1991: SGI1931 for 1908-1929 (GLAMOS length change: the front of 1907 lies 9 m from that of 1929,
    and GLAMOS uses the 1929 outline for the 1907 survey); then the signed-distance blend, linear in time, of SGI1931
    (1929-08-13), SGI1973 (1973-09-01) and mask_1994 (1991-09-18): 2.576, 2.350, 2.366 km2 at the 1961, 1971, 1985
    surveys (GLAMOS 2.595, 2.329, 2.338). Surface: usurf_1994 minus the GLAMOS mean elevation change from each survey
    to 1991-09-18, linear in time, the same on every cell (+45.6 m in 1908, +16.0 in 1929, +1.7 in 1949, -0.6 in
    1961, +4.0 in 1971, +6.3 in 1985). SGI1850 not used (no target before 1907). `results/calib_f1_geometry.csv`.
  - dT_pre: added to every day of the hydrological years up to 1991, uniform -3 to +3 K. Both sets ran:
    `results/calib_f1_*` (with dT_pre) and `results/calib_f1_stationary_*`; round files banked in s02
    `igm/results/calib_f1/`, `calib_f1_stationary/`.
  - Sampler (the shrinking box stalled at an effective sample size of 4.5-21.5 after two rounds on the dT-dg ridge):
    rounds 2+ draw 2^17 Sobol points from a Gaussian, first from the live draws of the previous round (likelihood
    above 1e-6 of the largest, at least the 50 x npar best, unweighted), once a round reaches an effective sample
    size of 20 from its importance-weighted mean and covariance x 1.5^2; weight = likelihood / proposal density, 0
    outside the prior box; at most 10 rounds; `start=R` reuses saved rounds; MAP = largest posterior density.
  - Station check (`results/f1_station_trend.csv`), 1991-2025: Robiei minus Locarno-Monti temperature trend +0.05 ±
    0.06 K per decade (summer +0.02 ± 0.06), precipitation ratio trend -1.8 ± 3.1 % per decade in winter, +0.05 ±
    2.3 over the year: the transfer carries no trend error.
  - Stationary set (`calib_f1_stationary_summary.csv`, effective sample size 23 919): cannot fit both periods; it
    gains ice before 1991 (1907-1929 model median +0.046 against -0.105 km3; misses of +4.5, +3.2, +7.8, +2.8, +3.0
    sigma for 1907-1929, 1929-1949, 1949-1961, 1971-1985, 1985-1991) while 1991-2018 are inside and 2018-2024 loses
    too much (-4.4 sigma); mean Ba 1992-2025 -0.92 (kept 5-95 %: -1.01 to -0.83) against -0.844.
  - With dT_pre (`calib_f1_summary.csv`, `calib_f1_bands.csv`, `calib_f1_draws.csv`, `preview_calib_f1.png`): effective
    sample size 11 854 (gate met); dT_pre +1.20 K (5-95 %: 1.03-1.38); 6 of 11 volume changes inside the 90 %
    predictive band; misses (observation outside the band, km3; model median minus observed in sigma): 1907-1929
    0.046 (+3.1), 1949-1961 0.0014 (+2.3), 1961-1971 0.0097 (-3.9), 1991-2002 0.0044 (+2.9), 2002-2008 0.0022 (+2.4);
    1907-1991 together -0.045 km3 modelled (sum of medians) against -0.164. Mean Ba 1992-2025 -0.56 (kept 5-95 %:
    -0.66 to -0.47) against -0.844: gate not met. MAP lowest year 2022 (-3.02), 2001 +1.06: met. Bands and stakes
    inside: winter 0.918, annual 0.914, stakes 0.898. The 1991-2024 volume changes are fitted equally badly at the
    stationary level (-0.92) and at this one (-0.56) (half the sum of the squared sigma misfits of the MAP draw: 10.9
    and 10.3): the model's loss grows too fast from 1991-2002 to 2018-2024, and the winter term holds the level (MAP
    winter misfit -0.22 m w.e. at the stationary level, -0.14 here; log-likelihood -147.5 against -130.3); with the
    transfer free of trend errors this misfit has no parameter here.
  - Prior edges: dT below -2.8 K in 13 % and c_prec above 2.9 in 14 % of the kept draws (dT-dg correlation -0.96);
    the temperature change at 2800 m, dT + dg x 2433 m, is narrow: +1.43 K (5-95 %: 1.21-1.67).
  - Not done here (needed before the F1 hindcast): `clim_otl` adds dT_pre to the hydrological years up to 1991 (the
    spin-up cycle 1865-1894 included, dT_1850 on top); the ensemble still reads `results/calib_draws.csv`.
- F2 settings and result (2026-10-09; process `user/code/processes/basin`, `track` column basin_melt_km3, `clim_otl`
  dT_pre, `checks/f2_spinup.py`, `quicklook_f2.py`):
  - Basin: a process after thk in the experiments fit, spinup, hindcast and projection sets thk = 0 outside the posbal
    cells (decision 2: SGI1850 of C14-10, its 100 m buffer on cells flatter than 30 degrees, without the 1850 outline
    of C14-11) at every time step, so ice that flows out melts in the step it arrives; the volume removed per year is
    series.csv basin_melt_km3, and track's outline-date thickness counts the removal in the year's flow change F.
    posbal does not touch the domain edge (rows 34-268 of 290, columns 21-321 of 340). The 2021 mask holds 41 cells
    (0.0041 km2, 0.000012 km3 at u = 0.5) outside the basin, the 2012 start of F3 45 cells: they melt in the first step.
  - dT_pre in `clim_otl`: added in the model's hydrological years up to 1991 (calib_smb.py PRE_END), so in a spin-up in
    every year with dT_1850 (offset) on top. For the spin-up only the sum dT_pre + dT_1850 enters, so F1's open
    decision 4 (dT_pre in the cycle with dT_1850 on top, or folded into dT_1850) does not change any spin-up state; both
    numbers are reported.
  - Test (`results/f2_spinup.csv`, `f2_runs_f2_scan.csv`, `f2_runs_f2_sec1.csv`, `quicklook_f2.png`): the F1 MAP draw
    (Fm 1.0438e-3, c_prec 2.7335, dT -2.2545 K, dg 1.5002 K per km, b -1.3432 % per 100 m, dT_pre +1.1805 K; r_ice
    = 0.012 Fm), u = 0, f = 1, 300-year spin-ups from ice-free, cold. Scan of dT_1850 from -2.6 to +0.4 K (stage
    f2_scan, 16 runs), then the secant of hindcast.py (round 1 at the scan's dT_1850 for 3.883 km2: -1.1493 K, stage
    f2_sec1): accepted in round 1, steady (+0.68 % volume change over the last cycle), cycle-mean ice area inside
    SGI1850 3.939 km2 (target 4.008 +- 0.25), 0.172 km2 in the buffer, 4.110 km2 in all, V 0.1638 km3, tongue 2052 m
    on 1850-10-01, dT_pre + dT_1850 = +0.031 K. Ice outside the basin and on the outermost row or column of the domain:
    0 cells at each of the 32 checked saves (every 10 years and the end) of every run; ice melted outside the basin
    0.00077 km3 a-1 (accepted run, last cycle), 0.0052 km3 a-1 at -2.6 K. The area inside SGI1850 saturates at
    3.99-4.00 km2 from -1.4 K down (the step-5 scan without the basin: 1.0-2.7 km2 outside SGI1850, mostly across the
    north-western divide, and ice on the domain edge).
- F3 settings and result (2026-10-09; `fit_f3.py`, `geometry` dem 2012, `checks/f3_sept2012.py`, `quicklook_f3.py`):
  - Samples: (balance draw, u, f), balance draw a random permutation of the 1024 kept draws of
    `results/calib_f1_draws.csv` per round (Fm, c_prec, dT, dg, b, dT_pre; r_ice = 0.012 Fm; F1 with dT_pre, as the
    F1 MAP of F2); u uniform in [-1, 2] with the 50 m edge taper; f log-uniform in [0.05, 20]. Round 1 Sobol over the
    box; round 2 the Gaussian of weights.py (weighted mean, twice the weighted covariance, tempered to an effective
    sample size of at least 16, truncated, 0.9 / 0.1 with the prior; fixed in `results/f3_proposal_r2.npz` before
    drawing); pooled by the balance heuristic.
  - Two segments per sample, experiment fit (basin on), observed weather: _12 from the 2012 surface of Grab et al.
    (inputs/step5_targets.npz s12 = C14-10_SwissALTI3D_r2019.tif + 0.556 m) 2012.75-2021.75 (start state: on the
    cells with an s12 value inside SGI2010 or the 2021 mask, topg = bed(u), thk = max(s12 - bed(u), 0), so off the 2021
    mask the ice is s12 minus the 2021 ground; elsewhere ice-free on usurf_2021), and _21 from the 2021 DEM
    2021.75-2026.75. The 1991-2021 segment of step 4 is dropped; the 2012 start replaces it for the two terms that
    need a run before 2021 (dh 2012-2021, ITS_LIVE 2014-2021).
  - Warm start (decision 1): every run starts from a network trained cold (3000 iterations) at u = 0, f = 1 on its
    segment's start state (nets/f3_2012_u0_f1_n3000.npz, stage f3_nets; nets/fit2021_u0_f1_n3000.npz of step 3) plus
    300 iterations. Checked over the box (`results/f3_warm.csv`, stage f3_warm: the F1 MAP balance at (u, f) = (-1,
    0.05), (-1, 20), (0.5, 3), (2, 0.05), (2, 20), warm against cold): areas 2024-2026 within 0.016 km2 (0.001 km2
    at (0.5, 3)), V_2026 within 0.4 %, dh 2012-2021 bands within 0.6 m root mean square (0.16 m at (0.5, 3)), 2014-2021
    speed within 3 % where it exceeds 0.5 m a-1 (10 % at (-1, 20), 0.996 against 1.110 m a-1), start speeds within
    8 % where above 0.3 m a-1; wall 44-77 s warm against 64-107 s cold.
  - Terms (`fit_f3.py` docstring): areas 2024-08-23 (all polygons), 2025-09-07 and 2026-09-27 (main body) with their
    sigma; dh 2012-2021 per 50 m band (14 bands, sigma 2.93 m); radar as the 2006 bedrock against bed(u) per profile
    (sigma 5.3 m; the start in 2021 compares the bed itself, results/f3_radar.csv); the 12 stake bounds with the
    provisional 2026 values (sigma 1 m); the ITS_LIVE bound on the 2014-2021 median speed of the _12 run (1 m a-1,
    sigma 0.5); every term with its model-error SD integrated out (log-uniform 0.1-10 x the term's reference SD), as
    step 4. The 2021-09-30 area is the start (not a term). The September 2012 balance is not in the _12 runs (start
    on 1 October; the observed dh starts on 2012-09-01): `results/f3_sept2012.csv`, F1 MAP -0.03 to -0.46 m ice per
    band (0.01-0.16 dh sigma, the model loses that much too little), not corrected.
  - Guard (clim_otl, smb_ti): the year that begins at time.end is not loaded; two runs of round 1 (f3r1_0674_21,
    f3r1_0921_21) ended a hair below 2026.75 and asked for hydrological year 2027 (KeyError); rerun after the guard
    (stage f3_r1fix, both ended).
  - Result (`results/f3_summary.csv`, `f3_terms.csv`, `f3_weights.csv`, `posterior_f3_128.csv`, `quicklook_f3.png`;
    runs and logs banked as s02 igm/runs_f3.tar.gz, s03 run folders kept): round 1 (stage f3_r1, 2048 runs, 71 min,
    19.0 process-h) effective sample size 70.2 < 128, so round 2 (proposal mean u 0.439, ln f -1.328, SD 0.169, 1.369,
    correlation -0.085, mass in the box 0.888; stage f3_r2, 2048 runs, 100 min, 26.5 process-h, 0 failed): pooled
    effective sample size 721.6 (largest weight 0.0037; 616 effective balance draws), gate met. posterior_f3_128.csv
    holds 128 distinct samples (systematic resampling, no copies).
  - Gate: 2026-09-27 main body 0.843 / 0.987 / 1.102 km2 (5 / 50 / 95 %) against 1.021: inside; 2025-09-07 main body
    1.096 / 1.205 / 1.279 against 1.231: inside; 2024-08-23 all polygons 1.211 / 1.280 / 1.334 against 1.380: outside
    by 0.046 km2 (median -0.90 sigma). No sample within one sigma of the 2026 area reaches the 2024 area at 1 m (their
    largest 1.338): the model loses too much area 2021-2024 (-0.32 against -0.23 km2) and too little 2024-2026 (-0.29,
    5-95 % 0.23-0.37, against -0.36). The ice threshold brackets the observation: at 0.1 m the same posterior gives
    1.429 / 1.487 / 1.548 km2 in 2024 (and 1.03 in 2026), at 2 m 1.242 in 2024.
  - Parameters: u +0.264 / +0.441 / +0.637 (not at an edge); f 0.057 / 0.230 / 1.211, flat in ln f from the prior
    edge 0.05 up to about 1 and cut above by the speed bound (largest |posterior - prior CDF| 0.42): f is bounded above
    only. The balance posterior stays the F1 posterior (Fm, c_prec, dT, dg, b medians within 1 % of F1). Correlations
    of u and ln f with every other parameter within +-0.10.
  - Speed (ITS_LIVE median 2014-2021 of the _12 runs): 0.124 / 0.409 / 2.006 m a-1 against the 1 m a-1 bound; 21.7 % of
    the weight above it, mean excess 0.14 m a-1. With the bound at its sigma only (variant speed_strict): 0.123 / 0.374
    / 1.432, 15.1 % above, f 0.057 / 0.202 / 0.843; the 2026 area 0.843 / 0.987 / 1.101 km2, unchanged.
  - Other terms at the posterior median: dh 2012-2021 bands within -0.21 to +0.57 sigma (7 of 14 observed values
    inside the 5-95 % band; the lowest band holds no 2012 ice in the model); radar bedrock 4-5 m above bed(u) on all
    six profiles (-0.79 to -1.03 sigma: the areas want a deeper bed than the radar); stakes: 10 of 12 bounds met,
    stake 05 -0.26 sigma, stake 09 -5.1 sigma (largest modelled 2021 thickness within 30 m 10.4 m against 15.5 m of
    melt since September 2021).
  - Consistency, not fitted: the _12 runs reach 2021-09-30 with 1.600 / 1.632 / 1.662 km2 (2021 mask 1.6065) and
    0.0195 / 0.0226 / 0.0259 km3 (start state of the same samples 0.0187 / 0.0216 / 0.0249). V_2026 0.0052 / 0.0073 /
    0.0100 km3. No ice flowed out of the basin in any _21 run (basin melt 0 in all 2048).
  - Not done here: hindcast.py, projection.py and noflow.py still read posterior_128.csv and pass no dT_pre; noflow.py
    has no basin rule (41 start cells outside the basin keep their ice there).
- F5 settings (2026-10-09, step 5 again with F1, F2 and F3; `hindcast.py` with P = f5, written before any f5 run):
  - Draws: the 128 of `results/posterior_f3_128.csv` (equal weights), each with its balance draw (Fm, c_prec, dT, dg, b,
    dT_pre; r_ice = 0.012 Fm), u and f; dT_pre passed to clim_otl, so in the spin-up dT_1850 = offset acts on top of
    dT_pre (every spin-up model year is <= 1991) and the hindcast has dT_pre in the hydrological years up to 1991.
    Basin process on (F2) in spin-up and hindcast.
  - Spin-up and secant as decision 10 and step 5 (300 years from ice-free, steady at 1 %, T = 4.008 km2 +- 0.25 inside
    SGI1850, at most 5 runs, steps <= 1 K). Round 1 from a new scan of the largest-weight F3 draw (f3r2_0798, u 0.431,
    f 0.054; stage f5_scan, dT_1850 -2.6 to +0.4 K in 0.2 K steps, as the F2 scan): every draw at the scan's dT_1850
    for A_in = 3.883 km2. Stages f5_scan, f5_spin1..5, f5_hind; results/f5_*.csv; inputs/f5_state2026.npz.
  - Hindcast 1850.75-2026.75 as step 5 (cold, observed weather, no offset). Track dates gain the geodetic surveys
    2002-09-16, 2008-08-29, 2013-08-21, 2018-09-01 (experiment/hindcast.yaml).
  - Validation table (`results/f5_targets.csv`, written before the hindcast): the step-5 rows (areas SGI1931, SGI1973,
    SGI2010, SGI2016; tongue 2000-2024; length 1899-1929/1961/1986/2024; surfaces 1991, 2012, 2021) and the 11 geodetic
    volume changes of F1 1907-2024 (GLAMOS dV and sigma = |dV| sigma_B / |B_geod|). Band = 5-95 % of the 128 draws
    (model spread, no observation noise), plus z = (median - observed) / sigma and whether |z| <= 2.
  - Expected misses, stated before the runs: the volume changes F1 itself misses on the prescribed geometry (observed
    outside F1's 90 % predictive band, direction of F1's model median): 1907-1929 too little loss, 1949-1961 too little
    loss, 1961-1971 too much loss, 1991-2002 too little loss, 2002-2008 too little loss. Decision 11's statement (Huss
    et al. 2009: 1929-1949 too little, 1949-1961, 1961-1971, 1971-1985 too much) is kept as a second column
    (expected_d11); F1's dT_pre now carries the mean pre-1991 level, so only the decadal pattern can still show.
    Not marked but likely from F3: the 2021-2024 area loss too large.
  - 2026 states for step 6 (decision 8): per draw the 2026-09-30 state of the hindcast and of the decision-8 path, the
    draw's F3 2021-2026 run (runs/f3_r{1,2}/SAMPLE_21, the f3_r1fix rerun where one exists).
  - Comparison with the first ensemble: every validation row carries the first ensemble's 5/50/95 % and band
    (results/step5_validation.csv).
  - GPUs: the Zn mesh48g GPU ranks were found running at the start of F5 (the rate sampler shows R48 from 9 October
    13:05; the last entry of logs/pause_mg.log is a STOP on 8 October 16:44, so they were not resumed by
    pause_zn_gpu.sh). Not touched: F5 runs beside them, 8 IGM processes per H200 as in answer 1.
  - Scan result (`results/f5_runs_f5_scan.csv`, `f5_runs_f5_scanfix.csv`; f3r2_0798, f = 0.054): A_in 4.000 km2 at
    -2.6 K, 3.995 at -1.6, 3.904 at -1.2, 3.605 at -1.0, 2.976 at -0.6, 0.793 at +0.4; V 0.441 km3 at -2.6, 0.324 at
    -1.2, 0.277 at -1.0 (F2 at f = 1: 0.165, 0.143). Only the six coldest runs (-2.6 to -1.4 K, saturated area) are
    steady; from -1.2 K up the volume still grows 1.6-9 % per 30-year cycle after 300 years (stiff ice, long response
    time), so near T a stiff draw cannot pass the 1 % test in one 300-year run from ice-free. Round 1: every draw at
    dT_1850 = -1.1862 K (the scan's A_in = 3.883 km2).
  - New choice (measured above): secant rounds 2-5 continue from the end state of the draw's previous run (dem
    restart: topg, thk_end) instead of ice-free, so each draw's spin-up is one chain of up to 5 x 300 years with the
    steady test unchanged (last 30-year cycle within 1 % of the one before). `hindcast.py` RESTART.
  - Slots: 12 (threads 0-5, 64-69). Slots 14 and 15 (threads 70, 71, inside the mask of the running Zn GPU ranks) got
    about 1 s of CPU per minute: scan_12 and scan_15 were stopped after 18 min and rerun on free slots (stage
    f5_scanfix, 97 and 101 s). TIMEOUT 3600 s per run.
  - Spin-up result (`results/f5_spinup.csv`, `f5_secant.csv`, `f5_runs_f5_spin1.csv`, `f5_runs_f5_spin2.csv`): round 1
    (128 runs at -1.1862 K, 114-384 s each on 12 slots) accepted 102 draws; the other 26 (23 not steady, mostly f <
    0.1; 3 outside the band) passed in round 2 from their round-1 end states (dT_1850 -1.20 to -1.40 K). 128 of 128
    steady and within 4.008 +- 0.25: A_in 3.863 / 3.975 / 3.995 km2 (5 / 50 / 95 %; min 3.763), ice in the buffer
    outside SGI1850 0.159 / 0.198 / 0.230 km2, V_1850 0.156 / 0.237 / 0.358 km3 (first ensemble 0.138-0.154), last
    cycle volume change -0.1 % to +0.8 %, tongue on 1850-10-01 2033 / 2065 / 2138 m (lowest SGI1850 point 2069 m),
    dT_pre + dT_1850 -0.17 / -0.01 / +0.12 K.
  - Hindcast result (`results/f5_validation.csv`, `f5_runs_f5_hind.csv`, `f5_series_f5_hind.csv`, `quicklook_f5.png`;
    stage f5_hind, 128 runs, 0 failed; band = 5-95 % of the 128 draws, z = (median - observed) / sigma; first ensemble's
    median in brackets):
    - Inventory areas (km2): SGI1931 2.20 / 2.66 / 3.30 against 3.20, inside, z -2.4 (5.49); SGI1973 2.19 / 2.50 / 2.84
      against 2.32, inside, z +1.8 (5.19); SGI2010 1.90 / 2.24 / 2.66 against 1.90, outside, z +7.0 (2.71); SGI2016
      1.72 / 2.11 / 2.55 against 1.64, outside, z +12.3 (1.92). 1850: 4.17 in all, 3.98 inside SGI1850.
    - Volume changes (km3, median against observed, z): 1907-1929 +0.0090 / -0.1053 (+3.4, expected too little: met);
      1929-1949 -0.0085 / -0.0522 (+2.2; d11 too little: met); 1949-1961 +0.0039 / -0.0087 (+3.5, expected too little:
      met; d11 too much: not met); 1961-1971 -0.0041 / +0.0122 (-3.6, expected too much: met); 1971-1985 +0.0012 /
      +0.0052 (-0.7, within 2 sigma); 1985-1991 -0.0163 / -0.0150 (-0.5, inside); 1991-2002 +0.0037 / -0.0095 (+3.4,
      expected: met); 2002-2008 -0.0146 / -0.0195 (+1.7, outside the band, expected: met); 2008-2013 -0.0036 / -0.0052
      (+1.4, inside); 2013-2018 -0.0082 / -0.0093 (+0.4, inside); 2018-2024 -0.0183 / -0.0122 (-4.3, too much loss, not
      expected). Sums of the per-draw changes: 1907-1991 -0.050 / -0.014 / -0.005 against -0.164 (F1 on the prescribed
      geometry -0.045; first ensemble sum of medians -0.009); 1991-2024 -0.057 / -0.041 / -0.029 against -0.056.
    - Tongue: inside the band 2000-2021 (2000: 2530 against 2530; 2010: 2552 against 2587; 2021: 2574 against 2663),
      outside 2022-2024 (2024: 2619, band 2520-2670, against 2688); first ensemble outside 2000-2017.
    - Length along the flowline: 1899-2024 -956 (-1312 to -610) against -814, inside (first: -2222); 1899-1929 -426
      against -146, outside (too much retreat before 1929); 1899-1961 and 1899-1986 inside.
    - Surfaces: mean over the observed cells inside the band in 1991, 2012, 2021 (+12 m in 2012 and 2021 at the median),
      RMS 17 m (first ensemble 6-9 m).
    - Confinement: no ice on the domain edge at any kept field (first ensemble: up to 1990-2020); no ice outside the
      basin at any save; the outline-date reconstruction of 1929-08-13 holds 1-7 cells of 1.0-1.23 m outside the basin
      (balance since 1 October added to a zero state, 0.0007 km2 at most); basin melt 1850-2026 0.014 / 0.022 / 0.039 km3.
    - 2021-2026 (compared, not validated): V_2021 0.025 / 0.057 / 0.112 km3 (start state from the measured thickness
      0.019-0.025); area 2026-09-27 main body 1.06 / 1.74 / 2.28 against 1.02 (first: 0.28).
    - Driver: ln f correlates with every misfit (r = -0.98 with V_1850, -0.91 with SGI1931, -0.87 with SGI2016, -0.96
      with V_2021). f >= 1 (9 draws): SGI1931 2.20, SGI2016 1.77, 2026 body 0.86 km2, V_2021 0.023 km3; f < 0.1 (34
      draws): 3.16, 2.40, 2.13 km2, 0.100 km3. No f fits 1931 and 2016 at once: the F3 posterior of f (flat 0.05-1,
      bounded above by the ITS_LIVE speed) is wider than the hindcast allows, and the balance level before 1991 (F1
      misses 1907-1929 by +3.1 sigma) leaves the 1907-1991 loss short by about 0.15 km3.
  - 2026 states (`inputs/f5_state2026.npz`, s03 and s02; draw order of posterior_f3_128.csv): decision-8 path V 0.0053 /
    0.0073 / 0.0103 km3, area 0.85 / 0.98 / 1.11 km2 (matches posterior_f3_128.csv V_2026 to 5e-7 km3); hindcast
    V 0.009 / 0.034 / 0.084 km3, area 1.12 / 1.76 / 2.30 km2. Step 6 starts from thk_dec8 (decision 8).
  - Banked: s02 igm/runs_f5.tar.gz (298 fields.npz, logs, stages); s03 run folders kept (author: keep for now).
- F6 settings (2026-10-09, step 6 again from the F3 2026 states; `projection.py` and `noflow.py` with prefix f6, written
  before any f6 run; first-ensemble files stay as they are):
  - Draws: the 128 of `results/posterior_f3_128.csv` (one draw per sample), each with its balance draw, u and f; dT_pre
    acts only in hydrological years up to 1991, so not in the projections. Start (decision 8): per draw the 2026-09-30
    state of its decision-8 path (`inputs/f5_state2026.npz` thk_dec8, topg_dec8) as s03 `inputs/f6_start2026/dNNN.npz`.
  - Members as decision 13: 128 x 26 chains x 3 SSPs = 9 984, seeds 2100000 + k, year order as step 6; 2026 level and
    station by draw, balanced (32 draws each); the turn through (anchored, OTL), (anchored, ROE), (native, OTL),
    (native, ROE) over the draws sorted by sample starts so that the MAP draw holds (anchored, OTL) (first ensemble:
    the lowest-numbered anchored copy of the MAP sample; F3 has no copies). `results/f6_members.csv`.
  - MAP draw: the largest F3 weight, draw 103 (f3r2_0798, u 0.431, f 0.054, weight 0.0037). Its 78 forcings also run
    with u = 0 from the 2026 state of its F3 2021-2026 line rerun with u = 0 (stage f6_prep, warm start as F3; the first
    ensemble reran the step-4 line cold): V 0.0144 km3 in 2021, 0.0027 km3 and A1 0.534 km2 on 2026-09-30.
  - Networks: stage f6_nets, one cold training (3000 iterations) per start state (129, nets/f6/); members warm-start with
    0 iterations from their own start state's network, as step 6.
  - Film members with yearly thk and surface velocity (track yearly): draw 103 with the chains of step 6 (2081-2100
    annual dT closest to the 26-chain median; the same three chains: SSP1-2.6 MPICSC-REMO2-MPIESM, SSP2-4.5
    CNRM-ALADIN-HADGEM, SSP5-8.5 MOHC-HADREM-CNRM). The member of the films (Outputs: the anchored SSP2-4.5 member
    with the median disappearance year, the one closest to its group's median area series) is chosen from the f6
    result (`projection.py film`) and its hindcast (f5 line), 2021-2026 path (F3 line) and projection rerun with yearly
    fields (stage f6_film, `results/f6_film.csv`, `f6_film_check.csv`).
  - No-flow twins (`noflow.py`): same start, weather and balance as their IGM member; start ice outside the basin set to
    0 (the basin process melts it in IGM's first step); the IGM stop rule (decision 14): a twin stops at the first
    1 October without a cell >= 1 m (the first ensemble's twins ran to 2100 and 4340 regrew). 10 files of <= 1024
    members on GPU 1, thread 5.
  - Checks: flowoff (IGM without flow against noflow.py, members d000_c00_1 and d000_c00_5, 2026-2032; stage f6_flowoff);
    rerun gate stage f6_rerun (the SSP2-4.5 film member d103_c06_2 and three random members), run after f6_proj.
  - Slots: 12 (threads 0-5, 64-69, 8 IGM processes per H200 at most), nice 10, nrun, TIMEOUT 1800 s per run. The Zn
    mesh48g GPU ranks were running (R48 in logs/rate_sampler.log) when f6 started; not touched.
- F6 result (2026-10-10; `results/f6_gates.csv`, `f6_verify.csv`, `f6_runs_f6_proj.csv`, `f6_noflow.csv`, `f6_flowoff.csv`,
  `f6_stage_f6_proj.csv`, `f6_rate_f6_proj.csv`, `f6_film.csv`, `f6_film_check.csv`):
  - Stages (s03, 12 slots, 0 failed): f6_prep 1 run, f6_nets 129 (makespan 830 s), f6_flowoff 2, f6_proj 10 062 (9 984
    + 78 u = 0; 2026-10-09 18:11 to 2026-10-10 02:18, makespan 29 183 s, 97.2 process-h; wall per run median 32 s,
    max 869 s; peak GPU memory 2 724 MiB), f6_rerun 4, f6_film 3. No-flow twins: 10 files on GPU 1, 19:06-19:14.
  - Gate (step 6) met: every run ended (no ice 9 004, 2100.75 1 058), no NaN, 10 062 ice-free grids with 0 inconsistent
    cells, the 4 reruns give the same disappearance years (0.01 and 0.1 km2) and identical series and fields; 129 of 129
    networks; warm start 0 m a-1 from the network's own cold run on every member. No-flow twins: 9 984 ended (no ice
    8 877, 2100 1 107), 0 NaN, 0 grid inconsistencies, 0 regrowing after a stop (first ensemble 4 340). Flowoff: IGM
    without flow equals noflow.py (area, volume 0; Ba within print rounding).
  - Independent check (`checks/step6_verify.py` on s02 from runs_f6.tar.gz): every row passes.
  - Start (quality bar a): A1 2026 (all cells >= 1 m) 0.842 / 0.980 / 1.112 km2 against the mapped main body 1.021 (first
    ensemble 0.338 / 0.455 / 0.720), V_2026 0.0052 / 0.0073 / 0.0103 km3 (first 0.0012 / 0.0020 / 0.0042). Basin (c): 8
    runs removed ice, at most 3e-6 km3 each.
  - Disappearance year 0.01 km2 (5 / 50 / 95 %, 1 664 members each; first ensemble's median in brackets): anchored
    SSP1-2.6 2035 / 2041 / 2055 (2034), SSP2-4.5 2035 / 2039.5 / 2048 (2034), SSP5-8.5 2035 / 2039 / 2046 (2034); native
    SSP1-2.6 2037 / 2086 / none (2048; 26.0 % keep ice to 2100), SSP2-4.5 2036 / 2048 / 2090 (2041), SSP5-8.5 2035 /
    2042 / 2058 (2037). Half-volume year against V_2026, median: 2029 anchored, 2030-2031 native (first 2028-2029).
    1 511 members exceed their V_2026 at some save (largest 2.36 times, d061_c20_5, native OTL SSP5-8.5; native 1 047,
    anchored 464): the year order lends single positive years of 1991-2026, and the native level lies about 0.5 K (JJA)
    below the observed trend, so native members regrow more often.
  - Flow (IGM minus no-flow disappearance year, paired): 0 / 0 / 0 years (5 / 50 / 95 %), min -47, max +54; 119 members
    later, 9 599 equal, 266 earlier. Over the 64 anchored draws the draw-median year correlates with V_2026 (r 0.65),
    u (0.49), Fm (-0.30), dT (0.29), ln f (-0.12); the start area alone r 0.20 (12 years per km2).
  - MAP draw 103 with u = 0 (78 members): median disappearance 2038 / 2036 / 2035 (SSP1-2.6 / 2-4.5 / 5-8.5) against
    2041 / 2038.5 / 2039 at its own u.
  - Film member (`results/f6_film.csv`): of the 1 664 anchored SSP2-4.5 members the 205 with the median year 2039, the
    one closest to the group's median area series: d047_c04_2 (draw 47, f3r2_0285, u 0.451, f 0.243, CLMCOM-CCLM4-MPIESM,
    OTL; root mean square 0.0085 km2; disappearance 2039 at 0.01 km2, 2034 at 0.1 km2; half-volume 2030). Stage f6_film:
    hindcast, 2021-2026 path and projection with yearly thk, us, vs (177, 6, 18 saves); path and projection identical
    to the ensemble runs, the cold-started hindcast within 2.5e-4 km3 and 0.007 km2 of f5_hind (solver noise).
  - Zn GPU stream: R48 in logs/rate_sampler.log until 2026-10-09 20:32:54, none since (ended outside this work; not
    touched). Zn CPU campaign rank-0 rate 1.002 cpu s per wall s before and 1.001 during f6_proj.
  - Banked: s02 igm/runs_f6.tar.gz (81 674 files: runs f6_prep, nets, flowoff, proj, rerun, film; logs; stages),
    igm/nets_f6.tar.gz (nets/f6, inputs/f6_start2026); s02 igm/results/f6/ (series, ice-free grids, raw no-flow
    outputs) and results/f6_*.csv. s03 run folders kept (author: keep for now).
- Independent check of F5 and F6 (2026-10-10; `checks/f56_check.py`, own code, from runs_f5/f6/f3.tar.gz and
  nets_f6.tar.gz on s02; `results/f56_check.csv`, `f56_validation.csv`, `f56_years.csv`, per-run `f56_spin.csv`,
  `f56_hind.csv`, `f56_start.csv`, `f56_proj.csv`, `f56_noflow.csv`): every compared row matches.
  - F5: observed dV and sigma rebuilt from the GLAMOS file, inventory areas from the SGI rasters; acceptance rebuilt
    (102 + 26, restart chains and hindcast restarts as reported); the validation table agrees with f5_validation.csv
    to print rounding on all 45 rows (band, z, n identical). Two summary ranges are 5-95 %, not extremes: last-cycle
    volume change -0.44 % to +0.96 % (all within the 1 % test), outline-date cells outside the basin up to 11 (0.0011
    km2; 0 at every yearly save and on the domain edge).
  - 2026 start (thk_dec8, = inputs/f6_start2026 on all 128) against Sentinel-2 2026-09-27 on the lattice (main body
    1.024, all polygons 1.028 km2): largest body 0.849 / 0.980 / 1.111 km2; intersection over union 0.69 / 0.74 /
    0.78; 83 % (75-91 %) of the observed body covered; model ice outside the polygons 0.13 km2, observed ice without
    model ice >= 1 m 0.17 km2 (5-95 %: 0.09-0.26).
  - F6: 10 062 runs ended, 0 NaN, 0 parameter or start mismatches, ice-free grids consistent and identical to the
    collected file, disappearance and both half-volume years identical per member, the 4 reruns identical; no-flow
    twins 9 984 ended, 0 saves after a stop, paired flow effect as reported.
  - Against the first ensemble (medians): disappearance +5 to +7 years except native SSP1-2.6 (+38); half-volume
    against V_2026 +1 to +2; against V_2021 2026 instead of 2023 (V_2026 / V_2021 0.27 / 0.34 / 0.41). Against
    PUBLISHED_PROJECTIONS.md: half of 2026 in 2029-2031 against Rounce 2033-2037, OGGM 2038-2042, Zekollari
    2041-2045; anchored disappearance 2039-2041 against OGGM 2058 to > 2100 and Van Tricht 2055-2069, inside Huss
    2036-2046, after Kappenberger's about 2030 (anchored 5 % 2035).
- F7 settings (2026-10-10, step 7 again for the fixed ensemble; `collect.py`, `figures.py`, `animate.py`,
  `film3d.py`, `checks/film_steps.py`, written for f7 before their runs; first-ensemble versions in the git history):
  - First ensemble kept for comparison: `results/first_ensemble/` (members.csv, step7_* tables, its RESULTS.md),
    `results/first_ensemble/figures/` (the six igm_*.pdf), `tex/animation/first_ensemble/` (igm_map_1850_2100,
    igm_3d_1850_2100 and the igm_iceflow and igm_particles films); s02 `results/first_ensemble/` (same small files),
    s02 `results/step7/` unchanged. Fixed ensemble: prefix f7, `results/members.csv`, `results/f7_*`, s02 `results/f7/`.
  - collect: inputs f6_* and posterior_f3_128.csv; the 2021-2026 path of each draw from runs_f3.tar.gz (f3_r1fix
    replaces its two runs), hindcast fields from runs_f5.tar.gz (19 dates + 2026-09-30). The no-flow twins stop by the
    IGM rule since F6, so the first ensemble's 'stop rule applied afterwards' columns are dropped; the years table adds
    the share of members earlier, the same, later with flow; the spread table adds the linear shares of ln f and V_2026.
  - Figures: seven PDFs, the six of Outputs plus `igm_published.pdf` (volume 2000-2100 under SSP2-4.5, logarithmic;
    disappearance and half of V_2026 per scenario against OGGM 2023, Rounce 2023, Zekollari 2019, Van Tricht 2025 and
    the statements of PUBLISHED_PROJECTIONS.md; published values are the member medians and ranges, lower middle for an
    even count, as that file). Time axes of projections end 3 years after the latest 95th-percentile disappearance year
    of the groups shown (author's rule): SSP1-2.6 2100 (native 95 % after 2100), SSP2-4.5 2093, SSP5-8.5 2061.
    Calibration figure now F1 and F3 (two temperature offsets, 11 volume changes, speed against the bound); years panel c
    is the share of members whose year differs without flow (the intervals are all 0).
  - Films: member d047_c04_2 of f6_film.csv; END = disappearance year + 3 = 2042 (stop 2043 at no ice). Map film:
    speeds measured on the film states (`animate.py states`): hindcast median 1.5-15 m a-1, p99 up to 24; after the 2021
    restart median 0.2 m a-1 (2021), 0.03 (2026); new choice: arrows 6 m per m a-1 in the hindcast and 20 times longer
    from the 2021 cross-fade on, key and a line in the film state it. 3D film: test renders (s03, frames 1000 and 2850)
    showed white spikes and a jagged wall where thick hindcast ice meets the basin edge of F2 (the 10 m staircase,
    with ice stacked on 2 m ridges); new choice: thick ice is drawn on the model's own surface (bilinear topg +
    thickness) instead of the lattice bed minus its 12 m Gaussian detail, blended into the ground by smoothstep(t / 5
    m) as before, and thickness and surface smoothed by a Gaussian of 8 m instead of 2 m; the spikes are gone in the
    repeat test, the thin 2026 margin unchanged. Grade re-measured on the f7 film (8 frames, before the smoothing
    change): exposure 0.0906, sky 0.403 (0.076-0.097, 0.356-0.479; `results/f7_film3d_measure.csv`).
  - GPUs for the 3D film: s02 RTX A6000 and s03 2 x H200 (idle, the Zn GPU stream stopped and not touched; 58 Mg CPU
    ph.x ranks on threads 6-63 and 70-127, so Blender pinned to 0-5 and 64-69, nice 10, nrun); s01 not used (64 ph.x
    ranks of the author running, load 64, and the crash of 8 October). Shares by frame hash, weights 1, 1.08, 1.08.
- F7 result (2026-10-10; RESULTS.md, rewritten for the fixed ensemble with a section on what changed against the first):
  - collect and summary on s02 (3 min): 10 062 members, the four years recomputed from the series equal the reduction
    in all 10 062 + 9 984 rows; results/members.csv, f7_*.csv, f7_icefree_maps.npz; s02 results/f7/*.nc.
  - Figures: tex/figures/igm_hindcast, igm_calibration, igm_volume_area, igm_years, igm_half_volume_u,
    igm_icefree_map, igm_published (.pdf).
  - Map film tex/animation/igm_map_1850_2042.mp4: 5 148 frames, 85.8 s, 60 frames per second; largest frame steps in
    the fade before the end card (results/f7_film_steps.csv).
  - 3D film tex/animation/igm_3d_1850_2042.mp4: 3 375 render frames (s02 A6000 1 068, s03 H200 1 155 + 1 152, 4.5 s
    each, 03:09-04:54), 4 095 film frames, 68.3 s; all frames readable; largest steps 3.2 grey levels at the hindcast
    firn years 1936 and 1977 and at the end of the 2021 morph (results/f7_film3d_check.csv, f7_film_steps.csv).
  - Quality bar: (a) met, (b) not met with every miss quantified, (c) met, (d) quantified (21.7 % of the weight above
    the bound, mean excess 0.14 m a-1), (e) met (effective sample size 721.6, 128 distinct draws).
- F7 review (2026-10-10, every figure and film frame looked at; no new runs):
  - Measured: the hindcast ice ends at the F2 basin edge in faces over 20 m along 6.3 km of margin in 1850 and 1.4 km
    in 2020, up to 182 m, elsewhere at most 67 m, none after the 2021 restart (`checks/basin_wall.py`,
    results/f7_basin_wall.csv); RESULTS.md had 'until about 1990'. Both end cards now state the basin rule and the
    step (map film re-made, 7 min; 3D film recomposed from the kept render frames, no re-render).
  - Hindcast volume at the end, now in RESULTS.md section 8 and quality bar (b): 0.057 km3 (median) on 2021-09-30
    against 0.022 from the measured thickness, 0.034 against 0.0073 in 2026 (f5_validation.csv rows "extra").
  - Figures: igm_hindcast panel d letter clear of the axis label; igm_volume_area time axis labelled 'Hydrological
    year'. RESULTS.md section 10 adds Zekollari 2019's disappearance years (after 2100 under RCP2.6 and RCP4.5, 2076
    under RCP8.5), as the figure already showed; PUBLISHED_PROJECTIONS.md notes that its 'ours' is the 8 October value.
  - Review sheets (contact sheets, the 1977 firn frames, both end cards): checks/film_sheets.py -> results/f7_review/.
