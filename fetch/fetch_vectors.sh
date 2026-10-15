#!/usr/bin/env bash
# Download national vector datasets, ski routes and winter huts (swisstopo STAC), SGI inventories and GLAMOS tables into raw/.
# swisstopo files are checked against the STAC sha2-256 multihash.
set -euo pipefail
cd "$(dirname "$0")/.."
STAC=https://data.geo.admin.ch/api/stac/v0.9/collections
stac() {  # collection item asset outdir
  local url sum
  url=$(curl -s "$STAC/$1/items/$2" | jq -r ".assets[\"$3\"].href")
  sum=$(curl -s "$STAC/$1/items/$2" | jq -r ".assets[\"$3\"][\"checksum:multihash\"]" | cut -c5- | tr A-F a-f)
  mkdir -p "raw/$4"
  curl -sSf --retry 5 -o "raw/$4/$3" "$url"
  echo "$sum  raw/$4/$3" | sha256sum -c
}
stac ch.swisstopo.swissnames3d swissnames3d_2026 swissnames3d_2026_2056.gpkg swissnames3d
stac ch.swisstopo.swissboundaries3d swissboundaries3d_2026-01 swissboundaries3d_2026-01_2056_5728.gpkg.zip swissboundaries3d
stac ch.swisstopo.swisstlm3d swisstlm3d_2026-02 swisstlm3d_2026-02_2056_5728.gpkg.zip swisstlm3d
stac ch.swisstopo-karto.skitouren skitouren skitouren_2056.gpkg.zip skitouren
stac ch.swisstopo-karto.skitouren skitouren additional-files_skitouren_2056.zip skitouren
stac ch.swisstopo.unterkuenfte-winter unterkuenfte-winter unterkuenfte-winter_2056.gpkg unterkuenfte-winter
mkdir -p raw/sgi raw/glamos
for f in inventory_sgi1850_r1992 inventory_sgi1931_r2022 inventory_sgi1973_r1976 inventory_sgi2010_r2010 inventory_sgi2016_r2020; do
  curl -sSf --retry 5 -o raw/sgi/$f.zip https://doi.glamos.ch/data/inventory/$f.zip
done
for f in massbalance/massbalance_observation.csv massbalance/massbalance_observation_elevationbins.csv \
         massbalance/massbalance_fixdate.csv lengthchange/lengthchange.csv volumechange/volumechange.csv; do
  curl -sSf --retry 5 -o raw/glamos/$(basename $f) https://doi.glamos.ch/data/$f
done
(cd raw/swissboundaries3d && unzip -o -q swissboundaries3d_2026-01_2056_5728.gpkg.zip)
(cd raw/swisstlm3d && unzip -o -q swisstlm3d_2026-02_2056_5728.gpkg.zip)
(cd raw/skitouren && for f in *.zip; do unzip -o -q $f; done)
for f in raw/sgi/*.zip; do d=${f%.zip}; mkdir -p $d; unzip -o -q $f -d $d; done
ls -laR raw/swissnames3d raw/swissboundaries3d raw/swisstlm3d raw/skitouren raw/unterkuenfte-winter raw/sgi raw/glamos
