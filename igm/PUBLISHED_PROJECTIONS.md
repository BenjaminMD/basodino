# Published projections of Ghiacciaio del Basòdino (C14-10, RGI60-11.01987)

Compiled 2026-10-08 for comparison with the IGM ensemble (`DESIGN.md`). Data, provenance and checksums:
`../data/published/README.md`; Basòdino rows: `../data/published/basodino_members.csv` (one line per GCM or
model mean), `basodino_volume.csv` (yearly volumes); extraction `published_rows.py`, `published_oggm.py`.

Note of 10 October: "ours" below is the 2021 volume at u = 0 (0.0142 km3) as of 8 October; the fixed ensemble's
central 2021 volume is 0.0216 km3 (posterior median u = 0.44), so the global datasets hold 2.3 to 4.7 times ours, not
3.5 to 7 times. The comparison with the fixed ensemble is `RESULTS.md` section 10 and `tex/figures/igm_published.pdf`.

## What the numbers are

- Our reference: measured 2021 area 1.591-1.607 km2 (GLAMOS outline, course mask); 2021 volume 0.0142 km3 from the
  Grab et al. (2021) bed (0.0023-0.0323 km3 over u = -1 to 1, `DESIGN.md` Risks); 2024 area 1.379 km2.
- All three global datasets start from the RGI 6.0 outline (2.092 km2; for Switzerland the 2003 Landsat
  inventory of Paul et al. 2011) and a modelled thickness, not from measured ice. Their Basòdino of 2021 holds
  0.05-0.10 km3 (OGGM: on 2.05 km2), 3.5 to 7 times our central volume. The RGI area is also larger than
  GLAMOS's 2002 outline (1.691 km2, `TARGETS.md` section 1); not checked whether it includes C14-11 or snowfields.
- "Half of 2021" and "half of 2026": first year with V <= V(2021)/2 or V(2026)/2 of the same series, which is our
  definition (DESIGN 14), computed here from the published series. Rounce and OGGM values are states on
  1 January of the year; Zekollari columns are labelled by year (2017-2100), timing within the year not stated.
- "V < 1 %": first year with volume below 1 % of the series' first value (2000 or 2017), the volume half of the
  Van Tricht et al. extinction rule. "A < 0.01 km2": first year with area below 0.01 km2 and no later return, our
  disappearance definition (only OGGM publishes area per glacier openly; the Rounce area files need an Earthdata
  login and were not fetched).
- Ranges: median [min, max] over the GCMs of a scenario (lower median for an even count); "> 2100" means the
  threshold is not reached in the file, and counts as later than 2100 in the median.

## Table

| Source | Model | Scenario (runs) | Start volume (year) | V 2021, 2026 km3 | Half of 2021 | Half of 2026 | Disappearance | Link |
|---|---|---|---|---|---|---|---|---|
| Zekollari, Huss, Farinotti 2019, The Cryosphere 13, 1125 | GloGEMflow (GloGEM mass balance, flowline ice flow), calibrated per glacier on the WGMS geodetic balance nearest to 1981-2010 (a neighbour's if none), Huss and Farinotti (2012) thickness, E-OBS to 2017 | RCP2.6, EURO-CORDEX mean (11 RCM runs) | 0.1092 km3 (2017) | 0.0971, 0.0805 | 2039 | 2045 | V < 1 %: > 2100 (V 2100 = 0.0098 ± 0.0086 km3, 9 % of 2017) | [paper](https://doi.org/10.5194/tc-13-1125-2019), [supplement](https://tc.copernicus.org/articles/13/1125/2019/tc-13-1125-2019-supplement.zip) |
| same | same | RCP4.5 (19) | 0.1098 km3 (2017) | 0.1022, 0.0873 | 2039 | 2042 | V < 1 %: > 2100 (V 2100 = 0.0012 ± 0.0018 km3, 1.1 %) | same |
| same | same | RCP8.5 (21) | 0.1098 km3 (2017) | 0.0977, 0.0817 | 2038 | 2041 | V < 1 %: 2076 (V 2100 = 0) | same |
| Rounce et al. 2023, Science 379, 78 | PyGEM mass balance + OGGM flowline, MCMC calibration on Hugonnet et al. (2021) 2000-2019 per glacier, OGGM ice-thickness inversion, median of 50 parameter sets per GCM | RCP2.6 (10 CMIP5 GCMs) | 0.087-0.091 km3 (1 Jan 2000, per GCM) | 0.0510, 0.0399 | 2032 [2029, 2043] | 2038 [2032, 2044] | V < 1 %: 2076 [2047, > 2100] | [paper](https://doi.org/10.1126/science.abo1324), [data, NSIDC](https://doi.org/10.5067/P8BN9VO9N5C7), [R11 copies](https://github.com/OGGM/alps_future/tree/master/data/Rounce_et_al_2023) |
| same | same | RCP4.5 (10) | same | 0.0517, 0.0379 | 2032 [2027, 2039] | 2036 [2030, 2042] | V < 1 %: 2055 [2038, 2095] | same |
| same | same | RCP8.5 (10) | same | 0.0502, 0.0373 | 2030 [2028, 2038] | 2035 [2033, 2040] | V < 1 %: 2050 [2044, 2066] | same |
| same | same | SSP1-1.9 (4 CMIP6 GCMs) | same | 0.0511, 0.0308 | 2029 [2027, 2038] | 2035 [2032, 2040] | V < 1 %: 2069 [2042, > 2100] | same |
| same | same | SSP1-2.6 (12) | same | 0.0502, 0.0370 | 2033 [2029, 2040] | 2037 [2033, 2045] | V < 1 %: 2059 [2048, 2084] | same |
| same | same | SSP2-4.5 (12) | same | 0.0515, 0.0339 | 2030 [2028, 2033] | 2033 [2031, 2038] | V < 1 %: 2053 [2042, 2082] | same |
| same | same | SSP3-7.0 (12) | same | 0.0512, 0.0386 | 2033 [2030, 2038] | 2035 [2033, 2041] | V < 1 %: 2053 [2044, 2067] | same |
| same | same | SSP5-8.5 (12) | same | 0.0527, 0.0379 | 2032 [2029, 2041] | 2034 [2030, 2045] | V < 1 %: 2048 [2041, 2066] | same |
| OGGM v1.6.1 standard projections (Schuster et al. 2023, data release) | OGGM elevation-band flowline, dynamic spin-up, per-glacier geodetic calibration (Hugonnet 2000-2019), W5E5 2000-2019 then GCM | SSP1-1.9 (4 CMIP6 GCMs) | 0.0985 km3 (1 Jan 2000) | 0.0605, 0.0418 | 2030 [2029, > 2100] | 2036 [2034, > 2100] | A < 0.01 km2: > 2100 in all 4 | [data](https://doi.org/10.5281/zenodo.8286065), files in `../data/published/oggm2023/source_urls.txt` |
| same | same | SSP1-2.6 (17) | same | 0.0625, 0.0504 | 2039 [2031, 2077] | 2042 [2034, 2086] | A < 0.01 km2: > 2100 [2090, > 2100] (2 of 17 before 2100) | same |
| same | same | SSP2-4.5 (14) | same | 0.0619, 0.0459 | 2033 [2029, 2045] | 2040 [2034, 2051] | A < 0.01 km2: 2097 [2080, > 2100] (10 of 14) | same |
| same | same | SSP3-7.0 (13) | same | 0.0616, 0.0514 | 2037 [2032, 2061] | 2039 [2034, 2062] | A < 0.01 km2: 2071 [2061, 2090] | same |
| same | same | SSP5-8.5 (19) | same | 0.0630, 0.0507 | 2035 [2030, 2057] | 2038 [2033, 2058] | A < 0.01 km2: 2058 [2049, 2086] | same |
| Van Tricht et al. 2025, Nature Climate Change, "Peak glacier extinction in the mid-twenty-first century" | GloGEM (Zekollari et al. 2024), OGGM and PyGEM (Rounce et al. 2023) pooled, RGI 6.0 start about 2000 | +1.5 °C in 2071-2100 vs 1850-1900 (12 runs: 4 GCM-SSP x 3 models) | each run's 2000 value | - | - | - | extinction year 2069, quartiles 2051 to > 2100 | [paper](https://doi.org/10.1038/s41558-025-02513-9), [data](https://doi.org/10.5281/zenodo.17371641), [code](https://github.com/LanderVT/glacier-extinction) |
| same | same | +2.0 °C (30 runs) | same | - | - | - | 2061 [2050, 2081] | same |
| same | same | +2.7 °C (42 runs) | same | - | - | - | 2061 [2050, 2070] | same |
| same | same | +4.0 °C (21 runs) | same | - | - | - | 2055 [2049, 2062] | same |

Van Tricht et al. numbers are the published file (`extinction_perglacier_RGI11.csv`), not recomputed: per run the
extinction year is the first year with the 5-yr moving-mean area < 0.01 km2 or volume < 1 % of the 2000 value,
whichever comes first; runs without extinction count as after 2100; the median and [Q1, Q3] are order statistics
of the pooled runs (`glacier_extinction_main.m`). As coded, the reported year is the index of the first annual value
below the threshold plus 2000 on an axis 2000-2100, i.e. one year after the labelled year of the first value below
it. The warming levels are not SSPs: +1.5 °C is close to SSP1-1.9/1-2.6, +2.7 °C to SSP2-4.5, +4.0 °C to SSP5-8.5.
The quartile 2051 at +1.5 °C means a quarter of those 12 runs lose Basòdino by 2051 even then; the empty Q3 means
at least a quarter keep it past 2100.

## Other projections and statements

Expert judgement and trend extrapolation, no model run published; most authoritative first. Quotes as printed.

| Date | Who | Statement (translation) | What the number is | Link |
|---|---|---|---|---|
| 2026-10-07 | Matthias Huss (GLAMOS, ETH Zurich), interviewed by RSI; read second-hand in Ossolanews (RSI original not found) | "entro dieci o vent'anni il Basòdino sarà completamente scomparso" (within ten or twenty years Basòdino will have disappeared completely); 2026 thickness loss 4 m | expert judgement, 2036-2046 | [Ossolanews](https://www.ossolanews.it/2026/10/07/leggi-notizia/argomenti/ambiente-2/articolo/formazza-il-ghiacciaio-del-basodino-si-sta-disgregando.html) |
| 2022-10-07 | Canton Ticino, Dipartimento del territorio (press release, no author) | "in 10 - 20 anni si verificherebbe la sparizione totale del ghiacciaio al di sotto dei 3'000 metri [...] e, in meno di mezzo secolo, la fusione totale della parte più a monte, attualmente spessa circa 30 metri" (all ice below 3000 m gone in 10-20 years, the upper part, now about 30 m thick, in under half a century); mean thickness about 27 m in 2005, about 15 m in 2022 | extrapolation of the 2007-2022 thinning (70-80 cm a-1) "if the trend continues": below 3000 m 2032-2042, all before about 2072 | [ti.ch](https://www4.ti.ch/tich/area-media/comunicati/dettaglio-comunicato?NEWS_ID=211810&cHash=bfca72d3507929b0643da0a7d5335ad3) |
| PDF of 2023-01-10 | Canton Ticino, fact sheet "Il ghiacciaio del Basòdino" (data VAW/ETH, GLAMOS) | "secondo le proiezioni più attuali, entro 20 anni (se non addirittura 5-10 anni, se dovessero verificarsi ancora delle condizioni come quelle avute nel 2022) sarà ridotto a poche piccole placche sottocresta" (within 20 years, or 5-10 if 2022-like years recur, reduced to a few small patches below the ridge) | unnamed "projections", about 2043 (2028-2033 with 2022-like years); remnants, not zero ice | [ti.ch PDF](https://m4.ti.ch/fileadmin/DT/temi/forestali/ghiacciai/documenti/08_Schede_GHIACCIAI_Basodino.pdf) |
| 2023-11-13 | Canton Ticino, Dipartimento del territorio | "nei prossimi 5-10 anni i ghiacciai su territorio ticinese saranno in buona parte scomparsi" (in 5-10 years Ticino's glaciers will largely be gone) | all Ticino glaciers, not Basòdino | [ti.ch](https://www4.ti.ch/tich/area-media/comunicati/dettaglio-comunicato/?NEWS_ID=231089) |
| 2025-09-08 | Giovanni Kappenberger (GLAMOS observer on Basòdino), tio.ch | "Fra cinque anni il ghiacciaio sarà sparito" (in five years the glacier will be gone); in 10 years only scattered remnants; the ice no longer moves | expert judgement, about 2030 | [tio.ch](https://www.tio.ch/ticino/attualita/1866260/tra-cinque-anni-non-ci-sara-piu-niente) |
| 2025-09-23 | Kappenberger, area | "Fra cinque anni questo ghiacciaio meraviglioso non esisterà più" (in five years this glacier will no longer exist) | same, about 2030 | [areaonline](https://www.areaonline.ch/Sguardo-antropologico-sul-basodino-in-fin-di-vita-cd723900) |
| about 2025-03-21 (date inferred) | Kappenberger, ticinonews | "Fra cinque anni il ghiacciaio del Basodino non esisterà più" | same, about 2030 | [ticinonews](https://www.ticinonews.ch/ticino/fra-cinque-anni-il-ghiacciaio-del-basodino-non-esistera-piu-409481) |
| 2022-11-03 | Kappenberger, RSI | "resteranno placche di ghiaccio sotto roccia già fra 5 o 10 anni" (only ice patches under the rock in 5 or 10 years); revises his earlier about 2040 | remnants by 2027-2032 | [RSI](https://www.rsi.ch/info/ticino-grigioni-e-insubria/Ghiacciai-ticinesi-un-2022-drammatico--1803420.html) |
| 2022-07-05 | Kappenberger, tio.ch | "Dieci, venti al massimo" (ten, twenty [years] at most) | 2032-2042 | [tio.ch](https://www.tio.ch/ticino/attualita/1593153/basodino-ghiacciaio-ticino-kappenberger-marmolada) |
| 2021-09-16 | Kappenberger, laRegione | "sarà completamente scomparso nei prossimi 20 anni" (completely gone in the next 20 years) | about 2041 | [laRegione](https://www.laregione.ch/cantone/locarnese/1535810/ghiacciai-svizzera-basodino-iniziativa-piedi) |
| 2019-10-15 | Kappenberger, climatizzati.ch | "dovrebbe sparire entro i prossimi vent'anni" (should disappear within twenty years) | about 2039 | [climatizzati](https://www.climatizzati.ch/it/articoli/2019-10-15-cosa-significa-la-scomparsa-dei-ghiacciai-per-la-svizzera) |
| 2019-09-19 | Kappenberger, Corriere del Ticino | "Del ghiacciaio del Basodino non ci sarà più traccia nel giro di 10-20 anni" (no trace left in 10-20 years) | 2029-2039 | [CdT](https://www.cdt.ch/news/ghiacciai-unagonia-silenziosa-e-inarrestabile-212247) |
| 2021-08-30 | Alleanza Climatica Svizzera (NGO), laRegione | "è destinato a scomparire entro una ventina d'anni" (bound to disappear in about twenty years) | about 2041 | [laRegione](https://www.laregione.ch/cantone/locarnese/1532026/alleanza-ghiacciai-basodino-svizzera-pianeta) |
| 2022-08-26 | swissinfo (after a 2020 RSI documentary, nobody named) | "By 2030 there will hardly be anything left of the Basòdino glacier" | 2030 | [swissinfo](https://www.swissinfo.ch/eng/climate-solutions/saving-ticino-s-disappearing-glaciers/47844378) |

No year for Basòdino found in GLAMOS, ETH or MeteoSwiss publications, SRF, NZZ, Tages-Anzeiger, French-language
or Italian national media, or the hydropower operators (Ofima). The Van Tricht et al. viewer
(https://www.glacierextinction.com) shows the same Zenodo data per glacier; not read beyond its front page.

## Searched, no Basòdino number

- GloGEM Swiss per-glacier release by Huss and colleagues: none found as an open per-glacier file; the GloGEM runs
  of Zekollari et al. (2024, The Cryosphere 18, 5045) enter Van Tricht et al. only through the 33.6 GB input
  archive of the Zenodo record (not fetched).
- Zekollari et al. (2024) GloGEM CMIP6 release (doi:10.5281/zenodo.10908278, 584 KB): regional series only.
- Huss and Fischer (2016, Frontiers in Earth Science 4:34, doi:10.3389/feart.2016.00034) model only very small
  Swiss glaciers (< 0.5 km2) to 2060; Basòdino is larger.
- Compagno et al. (2021, Swiss glaciers under CH2018): no per-glacier release found.
- Aellen, Kappenberger and Casartelli (1995, Geografia Fisica e Dinamica Quaternaria 18): description, no
  projection.

## Comparability with our ensemble

- Zekollari 2019: same model family as our balance (GloGEM degree-day, Basòdino's own geodetic balance) but CMIP5
  RCMs, a 2003 outline and a 2021 volume of 0.10 km3, seven times ours; its half-volume years (2038-2045) and V(2100)
  are therefore late bounds, comparable with ours only as a fraction of its own 2021 or 2026 volume.
  It also holds only the multi-model mean, so it compares with our scenario medians, not with our spread.
- Rounce 2023: per-GCM CMIP6 SSPs as ours, glacier-specific 2000-2019 calibration, 1 January states, but 0.05 km3
  in 2021 (3.6 times our central volume; its area files were not fetched); compare our half-of-2021 and
  half-of-2026 years per SSP directly with its per-GCM columns, and read its "V < 1 %" as an upper bound on our
  disappearance year.
- OGGM 2023: the only open per-glacier area series, so the only one that gives our own disappearance rule
  (A < 0.01 km2, no return); its 2021 area 2.05 km2 and volume 0.062 km3 (4.4 times ours) start it too large, and
  its 2026 area 1.75-1.83 km2 against the observed 1.067 km2 (2026-09-12) shows how far behind it already is.
  Comparable per SSP and GCM with our flow members; the gap is mostly the start state, which our u axis shows.
- Van Tricht 2025: three models pooled by warming level, year 2000 start, area or 1 % volume rule; comparable with
  our disappearance year only after grouping our 78 forcings by their 2071-2100 warming and only as a statement on
  starting too large: its runs start from the RGI outline (OGGM and PyGEM here: about 2.1 km2 and 0.09-0.10 km3 in 2000), so its years should be later than ours.
- Statements (Huss 2026: 2036-2046; canton 2022-2023: about 2043, below 3000 m 2032-2042; Kappenberger since 2025:
  about 2030): judgements on the measured, thin glacier of today, with no scenario, definition or start volume; they
  compare with our disappearance year under present-like warming only as an order of magnitude, and they agree with
  each other that Basòdino goes decades before the global models say.
- The difference is the start state: the measured thickness (mean 8.8 m on the 2021 mask, canton 15 m in 2022) is
  what our ensemble starts from, so where our years fall between the statements and the models is a result, not a
  check of either.
