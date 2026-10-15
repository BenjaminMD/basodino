#!/usr/bin/env bash
# Ice thickness and bed of Basodino (SGI2016 id C14-10): download into raw/thickness/, extract the C14-10 files into
# derived/thickness/.
#   SwissGlacierThickness-R2020 (Grab et al. 2021, doi:10.3929/ethz-b-000434697, CC BY 4.0): per-glacier grids,
#     summary table, radar profile locations. Zips checked against the repository MD5 (the PDFs get a cover page on
#     download, so they have no fixed checksum).
#   GLAMOS Swiss Glacier Ice Thickness release 2020 (doi:10.18750/icethickness.2020.r2020, CC BY 4.0): the Basodino
#     radar points (ETH Zurich survey of 2006, via GlaThiDa); the R2020 point files hold no C14 glacier.
set -euo pipefail
cd "$(dirname "$0")/.."
RC=https://www.research-collection.ethz.ch/server/api/core/bitstreams
mkdir -p raw/thickness derived/thickness
get() {  # uuid md5 file
  [ -s "raw/thickness/$3" ] || curl -sSf --retry 5 -o "raw/thickness/$3" "$RC/$1/content"
  [ "$2" = - ] || echo "$2  raw/thickness/$3" | md5sum -c
}
get 8c329721-868e-45a4-942e-dbcea0007409 - "00_ReadMe(updatedversion).pdf"
get 0e4bef7a-9735-41f7-b057-300dd6e0991d - "00_TablesIllustrations(updatedversion).pdf"
get 623ee896-88a8-4f26-b545-a1017d6b640b 32b9a40aad79421b02e9bc0027a4f0f4 01_Summary.zip
get 12e01d7c-0d8e-4f28-95a0-d227cd202a12 b85244ec87ecb0eaed9fccb4588ca2c1 03_GPR_Profiles_SwissAlps.zip
get e46074ea-01aa-4dde-b0bc-c6d9e3aa50e1 dc0748c426f07157a61b1e47053e6269 04_IceThickness_SGI.zip
get 38416da2-c392-40c0-808f-3740f008aa91 007cc444b2a1f78cd15fcfa37ed14687 05_IceThicknessUncertainty_SGI.zip
get 6762934c-a124-4595-b69d-5d5a31390504 45581789408921bc1e4efa9b5b307d94 06_InventoryYears_SGI.zip
get 91676987-d124-44b8-98d0-89b942174252 d887deda439979c6c4d1ac54f5202070 07_GlacierBed_SGI.zip
get c9b84186-3111-44be-85e9-4cb0e7b355c4 6fe5d6afaa99b446fd05bf7f4ca85886 08_SurfaceElevation_SGI.zip
[ -s raw/thickness/icethickness_2020_r2020.zip ] || curl -sSf --retry 5 -o raw/thickness/icethickness_2020_r2020.zip \
  https://doi.glamos.ch/data/icethickness/icethickness_2020_r2020.zip
for z in 04_IceThickness_SGI 05_IceThicknessUncertainty_SGI 06_InventoryYears_SGI 07_GlacierBed_SGI 08_SurfaceElevation_SGI; do
  unzip -o -j -q raw/thickness/$z.zip "$z/C14-10_*" -d derived/thickness
done
unzip -o -q raw/thickness/01_Summary.zip -d raw/thickness
unzip -o -q raw/thickness/03_GPR_Profiles_SwissAlps.zip -d raw/thickness
unzip -o -j -q raw/thickness/icethickness_2020_r2020.zip C14-10_basodino_GlaThiDa.txt -d derived/thickness
ls -la raw/thickness derived/thickness
