#!/usr/bin/env bash
# derived/vectors.gpkg: swissNAMES3D, swissTLM3D, swissBOUNDARIES3D layers (original layer and attribute names,
# 3D geometries kept, Z = LN02 m a.s.l.), ski routes and winter huts, SGI outlines (sgi.py), 2024 glacier outlines
# (glacier_2024.py). Features intersecting the bbox, not cut. Built in a temporary file, then moved into place.
set -euo pipefail
source ~/miniconda3/bin/activate agl
cd "$(dirname "$0")/.."
OUT=derived/vectors_build.gpkg
rm -f $OUT
BB="2676000 1137000 2686000 1147000"
add() {  # source.gpkg layer
  ogr2ogr -f GPKG $([ -f $OUT ] && echo -update) -nomd -a_srs EPSG:2056 -spat $BB -gt 65536 -nln "$2" $OUT "$1" "$2"
  echo "$2 $(ogrinfo -so $OUT "$2" | grep -o "Feature Count: [0-9]*")"
}
N=raw/swissnames3d/swissnames3d_2026_2056.gpkg
T=raw/swisstlm3d/SWISSTLM3D_2026_LV95_LN02.gpkg
B=raw/swissboundaries3d/swissBOUNDARIES3D_1_5_LV95_LN02.gpkg
for l in swissnames3d_pkt swissnames3d_lin swissnames3d_ply; do add $N $l; done
for l in tlm_strassen_strasse tlm_bb_bodenbedeckung tlm_bb_einzelbaum tlm_gewaesser_stehendes_gewaesser \
         tlm_gewaesser_fliessgewaesser tlm_bauten_gebaeude_footprint tlm_bauten_staubaute tlm_bauten_mauer \
         tlm_bauten_verbauung tlm_bauten_leitung tlm_areale_nutzungsareal tlm_areale_verkehrsareal \
         tlm_oev_uebrige_bahn tlm_oev_haltestelle tlm_eo_einzelobjekt; do add $T $l; done
for l in tlm_hoheitsgrenze tlm_landesgebiet tlm_kantonsgebiet tlm_hoheitsgebiet; do add $B $l; done
for l in ski_network_2056 ski_routes_2056; do add raw/skitouren/$l.gpkg $l; done
add raw/unterkuenfte-winter/unterkuenfte-winter_2056.gpkg accomodation_winter_2056
python scripts/sgi.py $OUT
python scripts/glacier_2024.py $OUT
mv $OUT derived/vectors.gpkg
sha256sum derived/vectors.gpkg
