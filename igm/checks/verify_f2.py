"""Independent check of F2 (DESIGN.md, fix round): basin confinement and the 1850 area of the spin-up test, re-derived
from the vector outlines and the run outputs without the basin process, inputs.py or hindcast.py.

  mask   (s02, agl env) the basin rebuilt from derived/vectors.gpkg: SGI1850 of C14-10 (cell centre inside, shapely),
         or within 100 m of it with the usurf_2021 slope below 30 degrees and not inside the 1850 outline of C14-11;
         compared cell by cell with igm/inputs/posbal.tif and sgi1850.tif; polygon area of SGI1850 C14-10
         -> results/verify_f2_masks.npz (own basin and SGI1850, rows north to south)
  runs   (s03, env.sh) every run of runs/f2_scan and runs/f2_sec1: at each save (decadal thk_y*, thk_end) the ice
         cells (thk > 0 and thk >= 1 m) outside the own basin, on the outermost row or column, and ice cells
         (>= 1 m) next to a cell outside the basin (where outflow is removed); from series.csv the last-cycle means
         (1821-1850) of the area >= 1 m inside SGI1850, the volume, the basin melt, and the volume change against
         the cycle before; the area inside the own SGI1850 at the decadal saves of the last cycle (1830, 1840, 1850)
         -> results/verify_f2_runs.csv
usage:  s02: python igm/checks/verify_f2.py mask ;  s03: python checks/verify_f2.py runs
"""
import csv
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parents[1]
RES = IGM / "results"
E0, N1, DX, NX, NY = 2678700, 1142900, 10, 340, 290
STAGES = ("f2_scan", "f2_sec1")
CYCLE, END = 30, 1850


def mask():
    import geopandas as gpd
    import rasterio
    import shapely
    der, inp = IGM.parent / "derived", IGM / "inputs"
    E, N = np.meshgrid(E0 + DX * (np.arange(NX) + 0.5), N1 - DX * (np.arange(NY) + 0.5))
    v = gpd.read_file(der / "vectors.gpkg", layer="sgi_1850")
    a = shapely.union_all(v[v.SGI == "C14-10"].geometry.values)
    b = shapely.union_all(v[v.SGI == "C14-11"].geometry.values)
    inside = lambda g: shapely.contains_xy(g, E, N)
    with rasterio.open(inp / "usurf_2021.tif") as r:
        z = r.read(1).astype(float)
    gy, gx = np.gradient(z, DX)
    flat = np.degrees(np.arctan(np.hypot(gx, gy))) < 30
    s1850 = inside(a)
    basin = s1850 | (inside(a.buffer(100.0)) & flat & ~inside(b))
    ref = {}
    for n in ("posbal", "sgi1850"):
        with rasterio.open(inp / f"{n}.tif") as r:
            ref[n] = r.read(1) == 1
    np.savez_compressed(RES / "verify_f2_masks.npz", basin=basin, sgi1850=s1850)
    print(f"SGI1850 C14-10 polygon area {a.area / 1e6:.4f} km2; cells {s1850.sum()} ({s1850.sum() * 1e-4:.4f} km2); "
          f"differ from sgi1850.tif {int((s1850 != ref['sgi1850']).sum())}")
    print(f"basin cells {basin.sum()} ({basin.sum() * 1e-4:.4f} km2), posbal.tif {ref['posbal'].sum()}; differ "
          f"{int((basin != ref['posbal']).sum())}; basin rows {np.nonzero(basin.any(1))[0][[0, -1]]} of {NY}, "
          f"columns {np.nonzero(basin.any(0))[0][[0, -1]]} of {NX}")


def runs():
    M = np.load(RES / "verify_f2_masks.npz")
    g = np.load(IGM / "inputs/igm_static.npz")
    flip = (M["sgi1850"][::-1] != (g["sgi1850"] == 1)).sum() < (M["sgi1850"] != (g["sgi1850"] == 1)).sum()
    basin, s1850 = (M["basin"][::-1], M["sgi1850"][::-1]) if flip else (M["basin"], M["sgi1850"])
    print(f"model rows {'south to north' if flip else 'north to south'}; own basin against igm_static posbal: "
          f"{int((basin != (g['posbal'] == 1)).sum())} cells differ")
    edge = np.zeros_like(basin)
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    pad = np.pad(~basin, 1, constant_values=True)
    rim = basin & (pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:])    # basin cells next to outside
    rows = []
    for st in STAGES:
        for d in sorted((IGM / "runs" / st).iterdir()):
            F = np.load(d / "fields.npz")
            keys = sorted(k for k in F.files if k.startswith("thk_y")) + ["thk_end"]
            out0 = max(int(((F[k] > 0) & ~basin).sum()) for k in keys)
            out1 = max(int(((F[k] >= 1) & ~basin).sum()) for k in keys)
            ed = max(int(((F[k] > 0) & edge).sum()) for k in keys)
            rim1 = int(((F["thk_end"] >= 1) & rim).sum())
            s = list(csv.DictReader(open(d / "series.csv")))
            last = [r for r in s if END - CYCLE < int(r["year"]) <= END]
            prev = [r for r in s if END - 2 * CYCLE < int(r["year"]) <= END - CYCLE]
            mean = lambda rr, c: np.mean([float(r[c]) for r in rr])
            ain_saves = [((F[f"thk_y{y}"] >= 1) & s1850).sum() * 1e-4 for y in (1830, 1840, 1850)]
            rows.append([st, d.name, len(keys), out0, out1, ed, rim1, int(rim.sum()), len(last),
                         f"{mean(last, 'A1_in1850_km2'):.4f}", f"{mean(last, 'A1_km2'):.4f}",
                         f"{mean(last, 'V_km3'):.6f}", f"{mean(last, 'V_km3') / mean(prev, 'V_km3') - 1:+.4f}",
                         f"{mean(last, 'basin_melt_km3'):.6f}",
                         " ".join(f"{a:.4f}" for a in ain_saves),
                         f"{((F['thk_end'] >= 1) & ~s1850).sum() * 1e-4:.4f}"])
    head = ["stage", "run", "saves", "ice_cells_outside_basin_thk_gt0_max", "ice_cells_outside_basin_ge1m_max",
            "ice_cells_on_domain_edge_max", "rim_cells_with_ice_end", "rim_cells", "last_cycle_years",
            "Ain1850_cycle_mean_km2", "A_cycle_mean_km2", "V_cycle_mean_km3", "V_last_over_prev_minus_1",
            "basin_melt_km3_per_a", "Ain1850_own_saves_1830_1840_1850_km2", "A_outside_sgi1850_end_km2"]
    with open(RES / "verify_f2_runs.csv", "w", newline="") as f:
        csv.writer(f).writerows([head, *rows])
    for r in rows:
        print(*r, sep=" | ")


if __name__ == "__main__":
    {"mask": mask, "runs": runs}[sys.argv[1]]()
