"""Append SGI outlines (all glaciers intersecting the bbox) to a GeoPackage, one layer per inventory year.
Original attributes kept; area_km2_lv95 added (planimetric area of the LV95 polygon).
1850/1973/2010 are LV95 (+LN02) and are only relabelled EPSG:2056; 1931 is LV03 -> LV95 via the CHENyx06a grid."""
import sys
import geopandas as gpd
from shapely.geometry import box

OUT = sys.argv[1]
BBOX = box(2676000, 1137000, 2686000, 1147000)
SRC = "raw/sgi/inventory_sgi{}/{}.shp"
LAYERS = {  # layer: (zip stem, shapefile, source CRS)
    "sgi_1850": ("1850_r1992", "SGI_1850", None),
    "sgi_1931": ("1931_r2022", "SGI_1931", "EPSG:21781"),
    "sgi_1973": ("1973_r1976", "SGI_1973", None),
    "sgi_2010": ("2010_r2010", "SGI_2010", None),
    "sgi_2016": ("2016_r2020", "SGI_2016_glaciers", None),
    "sgi_2016_debriscover": ("2016_r2020", "SGI_2016_debriscover", None),
    "sgi_2016_centerlines": ("2016_r2020", "SGI_2016_centerlines", None),
}
for name, (stem, shp, crs) in LAYERS.items():
    g = gpd.read_file(SRC.format(stem, shp))
    g = g.to_crs("EPSG:2056") if crs else g.set_crs("EPSG:2056", allow_override=True)
    g = g[g.intersects(BBOX)].copy()
    if g.geom_type.iloc[0].endswith("Polygon"):
        g["area_km2_lv95"] = g.area / 1e6
    g.to_file(OUT, layer=name, driver="GPKG", mode="a")
    ids = g["sgi-id"] if "sgi-id" in g else g["SGI"]
    print(name, len(g), "features; ids:", " ".join(sorted(ids.astype(str).unique())))
