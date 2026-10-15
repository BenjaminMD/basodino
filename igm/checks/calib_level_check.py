"""Diagnostic for step 2 (not the calibration): calib_smb.py run with the model-error SD of the volume term (d)
restricted to 0.05-0.15 m w.e. a-1 instead of 0.05-1, everything else unchanged.

Question it answers: with the glacier-wide level held by the geodetic volume changes, how well do the band and stake
terms (a, b, c) still fit, and does the MAP draw get its lowest balance in 2022 and a positive one in 2001.
Outputs: igm/results/calib_level_check[_dT]/ (round npz, calib_summary.csv, calib_draws.csv, calib_bands.csv); the
first calibration's runs (r_ice free, 2026-10-07) are in calib_level_check_{3,4}par/.
usage (s03, env.sh, one GPU, nrun):  python checks/calib_level_check.py [dT]   (dT = with the temperature offset)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calib_smb as c  # noqa: E402

c.S_NODES["d"] = np.exp(np.log(0.05) + (np.arange(64) + 0.5) / 64 * np.log(0.15 / 0.05))
dt = sys.argv[1:] == ["dT"]
c.RES = c.RES / ("calib_level_check_dT" if dt else "calib_level_check")
c.run(2, dt)
