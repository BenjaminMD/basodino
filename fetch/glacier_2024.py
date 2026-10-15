"""Append glacier_2024 to a GeoPackage: swissTLM3D glacier polygons (tlm_bb_bodenbedeckung joined to tlm_bb_glamos on
tlm_bodenbedeckung_uuid = uuid) intersecting the bbox, not cut, 3D kept. One feature per polygon.
Attributes: sgi, name (SGI 2016 name of that id, if any), herkunft_jahr (of the polygon), area_km2 (planimetric, LV95)."""
import sys
import geopandas as gpd
import pyogrio

OUT = sys.argv[1]
TLM = "raw/swisstlm3d/SWISSTLM3D_2026_LV95_LN02.gpkg"
BBOX = (2676000, 1137000, 2686000, 1147000)
link = pyogrio.read_dataframe(TLM, layer="tlm_bb_glamos", columns=["tlm_bodenbedeckung_uuid", "sgi"], read_geometry=False)
poly = pyogrio.read_dataframe(TLM, layer="tlm_bb_bodenbedeckung", columns=["uuid", "herkunft_jahr"], bbox=BBOX)
names = gpd.read_file("raw/sgi/inventory_sgi2016_r2020/SGI_2016_glaciers.shp", ignore_geometry=True).set_index("sgi-id")["name"]
g = poly.merge(link, left_on="uuid", right_on="tlm_bodenbedeckung_uuid")
g["name"] = g["sgi"].map(names)
g["area_km2"] = g.area / 1e6
g = g[["sgi", "name", "herkunft_jahr", "area_km2", "geometry"]].sort_values(["sgi", "area_km2"], ascending=[True, False])
g.to_file(OUT, layer="glacier_2024", driver="GPKG", mode="a")
print(g.drop(columns="geometry").to_string(index=False))
b = g[g["sgi"] == "C14-10"]
print(f"glacier_2024 {len(g)} features; C14-10: {len(b)} polygons, {b['area_km2'].sum():.3f} km2")
