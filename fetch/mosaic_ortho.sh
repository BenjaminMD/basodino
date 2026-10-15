#!/usr/bin/env bash
# RGB ortho COGs (0.25 m, 1 m) from the 0.1 m SWISSIMAGE VRT. Cubic, full-resolution sources only (-ovr NONE).
# The VRT alpha band (0 where no Swiss tile) becomes an internal mask; JPEG YCbCr quality 90.
set -euo pipefail
source ~/miniconda3/bin/activate agl
ulimit -n 65536
cd "$(dirname "$0")/../derived"
export GDAL_CACHEMAX=4096 CPL_TMPDIR=$PWD
BB="2676000 1137000 2686000 1147000"
for r in 0.25 1; do
  n=$(echo $r | tr . p)
  gdalwarp -q -overwrite -ovr NONE -r cubic -tr $r $r -te $BB -multi -wo NUM_THREADS=16 -wm 4096 \
    -of COG -co COMPRESS=JPEG -co QUALITY=90 -co BLOCKSIZE=512 -co NUM_THREADS=16 -co BIGTIFF=IF_SAFER \
    -co OVERVIEW_RESAMPLING=CUBIC ortho_0p1m.vrt ortho_${n}m.tif
  gdalinfo ortho_${n}m.tif | grep -E "Size is|COMPRESSION|Mask Flags" | sort -u
done
ls -la
