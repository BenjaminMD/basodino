#!/usr/bin/env bash
# Climate forcing for Basodino: station-scale climate scenarios and observed daily series into raw/climate/.
#   Climate CH2025 DAILY-LOCAL (MeteoSwiss & ETH Zurich 2025, doi:10.18751/climate/scenarios/ch2025/data/1.0, CC BY 4.0),
#     STAC collection ch.meteoschweiz.ogd-climate-scenarios-ch2025: bias-adjusted (quantile mapping calibrated on
#     1991-2020) 30-year daily series for the reference period (ref91-20) and the global warming levels 1.5, 2.0, 2.5 and
#     3.0 degC; one CSV per station, variable and period, the 26 EURO-CORDEX GCM-RCM chains as columns (24 at GWL3.0,
#     21 for hurs). All CSVs of Locarno / Monti (OTL, the report's station) and Robiei (ROE, 1898 m, 4.2 km from the
#     glacier), and the station and parameter lists. The zips hold the same values as NetCDF (one file per chain) and are
#     not fetched. STAC has no checksums for this collection; the CSV line count (23 header lines, column line,
#     10950 days) is checked instead.
#   CH2018 (NCCS 2018, doi:10.18751/Climate/Scenarios/CH2018/1.0, CC BY 4.0): the transient daily DAILY-LOCAL series
#     1981-2099 (68 simulations) are only available on request (NCCS contact form, FTPS account) and are not here.
#     Open is the web-atlas climate-chart data per station, fetched into raw/climate/ch2018/ for tas and pr at OTL,
#     GRH (Grimsel Hospiz) and CIM (Cimetta): monthly 30-year means, observations 1981-2010 and 2035, 2060, 2085
#     x RCP2.6, 4.5, 8.5 x lower, medium, upper estimate (5, 50, 95 % of the simulations); pr in mm per day.
#     Source: MeteoSwiss product browser API, product climate-ch2018-climate-normals-data. Check: 12 month rows.
#   MeteoSwiss observations (OGD, CC BY 4.0), sha256-checked against STAC: ogd-nbcn homogenised daily temperature and
#     homogenised monthly series (OTL, GRH), ogd-smn measured daily series (OTL, ROE, GRH), parameter and station lists;
#     ogd-nbcn homogenised monthly series of Lugano (LUG, T and P from 1864-01) and Col du Grand St-Bernard (GSB, T from
#     1818-01), which carry the forcing back to 1850 (igm/weather.py).
#   Documentation into raw/climate/docs/: CH2025 and CH2018 localized-data documentation, CH2018 station list, CH2025
#     ensemble sub-selection guide, CH2025 brochure (source of the GWL timing).
set -euo pipefail
cd "$(dirname "$0")/.."
STAC=https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz
C=raw/climate
mkdir -p $C/ch2025 $C/ch2018 $C/obs $C/docs
fetch() {  # stac-url href-regex dir: every matching asset, sha256-checked when STAC has a checksum; downloaded to .part
           # and moved into place, so a rerun never shows a reader a half-written file
  curl -sSf --retry 5 "$1" |
    jq -r --arg re "$2" '.assets[] | select(.href | test($re)) | "\(.href) \((.["file:checksum"] // "....")[4:])"' |
    while read -r u h; do
      f=$3/${u##*/}
      curl -sSf --retry 5 -o "$f.part" "$u"
      [ -z "$h" ] || echo "$h  $f.part" | sha256sum -c --quiet
      mv "$f.part" "$f"
    done
}
fetch "$STAC.ogd-climate-scenarios-ch2025/items/otl" '\.csv$' $C/ch2025
fetch "$STAC.ogd-climate-scenarios-ch2025/items/roe" '\.csv$' $C/ch2025
fetch "$STAC.ogd-climate-scenarios-ch2025" '\.csv$' $C/ch2025
for s in otl grh; do fetch "$STAC.ogd-nbcn/items/$s" '_(d_historical|d_recent|m)\.csv$' $C/obs; done
for s in lug gsb; do fetch "$STAC.ogd-nbcn/items/$s" '_m\.csv$' $C/obs; done
for s in otl roe grh; do fetch "$STAC.ogd-smn/items/$s" '_d_(historical|recent)\.csv$' $C/obs; done
for c in ogd-nbcn ogd-smn; do fetch "$STAC.$c" '_meta_(parameters|stations)\.csv$' $C/obs; done
for f in $C/ch2025/ogd-climate-scenarios-ch2025_???_*.csv; do
  [ "$(wc -l < "$f")" -eq 10974 ] || { echo "unexpected line count: $f"; exit 1; }
done
PB=https://api.productbrowser.app.meteoswiss.ch/pbbackend/api/v1/products/climate-ch2018-climate-normals-data/realizations
for s in OTL GRH CIM; do for v in tas pr; do
  f=$C/ch2018/climate-ch2018-climate-normals-data_${v}_${s}.csv
  curl -sSf --retry 5 -o $f "$PB?location=$s&indicatorScenario=$v&language=en&productName=climate-ch2018-climate-normals-data&mediaType=text%2Fcsv"
  [ "$(grep -c "^\"[A-Z][a-z][a-z]\";" $f)" -eq 12 ] || { echo "unexpected month rows: $f"; exit 1; }
done; done
doc() { [ -s "$C/docs/$1" ] || curl -sSfL --retry 5 -A Mozilla/5.0 -o "$C/docs/$1" "$2"; }
doc CH2025_documentation_localized_v1.1.pdf \
  https://www.meteoschweiz.admin.ch/dam/jcr:f54fae36-9a40-4e45-8abb-0308d0a68a10/CH2025_documentation_localized_v1.1.pdf
doc CH2025_ensemble_subselection_guide_v1.0.pdf \
  https://www.meteoswiss.admin.ch/dam/jcr:c746625d-da94-4bb0-bd8f-f8260c15f8f1/CH2025_ensemble_subselection_guide_v1.0.pdf
doc CH2025_brochure_en.pdf \
  https://res.cloudinary.com/adminch/image/private/s--h160e77H--/v1762347318/Bundespublikationen/862732184.pdf
doc CH2018_documentation_localized_v1.2.pdf \
  https://www.nccs.admin.ch/dam/nccs/de/dokumente/website/klima/CH2018_documentation_localized_v1.2.pdf.download.pdf/CH2018_documentation_localized_v1.2.pdf
doc stations_CH2018_meta.csv \
  https://www.nccs.admin.ch/dam/nccs/de/dokumente/website/klima/stations_CH2018_meta.csv.download.csv/stations_CH2018_meta.csv
echo "ch2025: $(ls $C/ch2025 | wc -l) files, ch2018: $(ls $C/ch2018 | wc -l) files, obs: $(ls $C/obs | wc -l) files, docs: $(ls $C/docs | wc -l) files"
du -sh $C/ch2025 $C/ch2018 $C/obs $C/docs
