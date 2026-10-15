#!/usr/bin/env bash
# DEM mosaics (0.5 m, 2 m), 0.1 m ortho VRT, Swiss data coverage mask. Grid: LV95, top-left 2676000/1147000.
set -euo pipefail
source ~/miniconda3/bin/activate agl   # sets PROJ_DATA/GDAL_DATA
ulimit -n 65536                       # VRT sources are opened once per thread
cd "$(dirname "$0")/../derived"
export GDAL_CACHEMAX=4096 GDAL_NUM_THREADS=16
BB="2676000 1137000 2686000 1147000"
COG_DEM="-of COG -co COMPRESS=DEFLATE -co PREDICTOR=FLOATING_POINT -co LEVEL=9 -co BLOCKSIZE=512 -co NUM_THREADS=16 -co BIGTIFF=IF_SAFER -co OVERVIEWS=IGNORE_EXISTING -co OVERVIEW_RESAMPLING=AVERAGE"
for r in 0.5 2; do
  n=$(echo $r | tr . p)
  gdalbuildvrt -q -te $BB -tr $r $r -srcnodata -9999 -vrtnodata -9999 dem_${n}m.vrt ../raw/swissalti3d/$r/*.tif
  gdal_translate -q $COG_DEM -a_nodata -9999 dem_${n}m.vrt dem_${n}m.tif
  rm dem_${n}m.vrt
done
gdalbuildvrt -q -te $BB -tr 0.1 0.1 -addalpha ortho_0p1m.vrt ../raw/swissimage-dop10/0.1/*.tif
# 1 = swissALTI3D and SWISSIMAGE tile present, 0 = outside Swiss data (2 m grid)
gdal_calc --quiet -A dem_2m.tif --calc "A!=-9999" --type Byte --NoDataValue none --hideNoData \
  --format COG --co COMPRESS=DEFLATE --co OVERVIEW_RESAMPLING=NEAREST --outfile swiss_coverage_mask.tif --overwrite
ls -la
