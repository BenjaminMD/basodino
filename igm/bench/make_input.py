"""Write the synthetic benchmark inputs bench/data/synth_dx{10,20}.nc.

A Basodino-sized test case, not Basodino: a 3.5 km x 3.5 km domain, a bed
sloping 0.22 towards y = 0 (2450 m to 3218 m) with a parabolic cross-valley
profile, and an elliptical ice body of 1.9 km2 and 90 m maximum thickness.
Variables: x, y (m), topg, thk (m), read by the IGM input module `local`.
"""
from pathlib import Path

import numpy as np
import xarray as xr

L = 3500.0  # domain side (m)
out = Path(__file__).parent / "data"
for dx in (10, 20):
    x = np.arange(0.0, L, dx, dtype="float32")
    X, Y = np.meshgrid(x, x)
    topg = 2450.0 + 0.22 * Y + (X - 1750.0) ** 2 / 12000.0
    r2 = ((X - 1750.0) / 800.0) ** 2 + ((Y - 2450.0) / 750.0) ** 2
    thk = 90.0 * np.sqrt(np.clip(1.0 - r2, 0.0, None))
    ds = xr.Dataset(
        {"topg": (("y", "x"), topg.astype("float32")),
         "thk": (("y", "x"), thk.astype("float32"))},
        coords={"x": x, "y": x},
    )
    ds.to_netcdf(out / f"synth_dx{dx}.nc")
    print(f"dx={dx} m grid={x.size}x{x.size} ice area={(thk > 0).sum() * dx * dx / 1e6:.2f} km2 "
          f"volume={thk.sum() * dx * dx / 1e9:.3f} km3")
