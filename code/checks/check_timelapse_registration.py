"""Check of the co-registration of code/timelapse.py: shift its reference scene by known sub-cell amounts and recover
them with its estimate.

Method: B04 of the reference scene REF over the registration window, moved by (rows, columns) d through the Fourier
shift theorem (periodic, exact for a band-limited image) to img(x) = ref(x - d), then timelapse.shift(gradient(REF),
gradient(img)), which estimates d with img(x + d) = ref(x). Error = estimate - d, cells.

Inputs: those of code/timelapse.py (imported; run where the scenes are)
Output: code/checks/check_timelapse_registration.csv, one row per shift: rows, columns, the two estimates, the errors
usage, on s02 in the mirror (env agl), under a minute:
  rsync -a code/checks/check_timelapse_registration.py s02:/data/befahl/projects/AGL/basodino/report/code/checks/
  ssh s02 'source ~/miniconda3/bin/activate agl && cd /data/befahl/projects/AGL/basodino/report &&
           python code/checks/check_timelapse_registration.py'
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import timelapse as T  # noqa: E402

SHIFTS = [(0.0, 0.0), (0.25, -0.1), (-0.6, 0.35), (0.9, 0.9), (-1.3, 0.05), (2.45, -3.7)]


def fourier_shift(a, dy, dx):
    """a(x - d), d = (dy, dx) rows and columns, by the shift theorem (NaN as the mean)."""
    a = np.where(np.isfinite(a), a, np.nanmean(a))
    ky, kx = np.fft.fftfreq(a.shape[0])[:, None], np.fft.fftfreq(a.shape[1])[None, :]
    return np.fft.ifft2(np.fft.fft2(a) * np.exp(-2j * np.pi * (ky * dy + kx * dx))).real


rows = []
for dy, dx in SHIFTS:
    ey, ex = T.shift(T.GREF, T.gradient(fourier_shift(T.REF_B04, dy, dx)))
    rows.append([dy, dx, round(float(ey), 3), round(float(ex), 3), round(float(ey - dy), 3), round(float(ex - dx), 3)])
    print(f"shift rows {dy:+.2f} columns {dx:+.2f}: estimate {ey:+.2f} {ex:+.2f}, error {ey - dy:+.2f} {ex - dx:+.2f}")
with open(Path(__file__).with_suffix(".csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rows", "columns", "estimate_rows", "estimate_columns", "error_rows", "error_columns"])
    w.writerows(rows)
