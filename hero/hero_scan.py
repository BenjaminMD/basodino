"""Scan rendered candidates for single-pixel outliers: grey pixels darker (brighter) than all 8 neighbours by more than
t levels. Prints counts per candidate and the strongest cases (image x, y).

usage: python hero_scan.py <png> [<png> ...]"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as nd

RING = np.ones((3, 3), bool)
RING[1, 1] = False


def scan(path):
    g = np.asarray(Image.open(path)).astype(np.int16)
    mx = nd.maximum_filter(g, footprint=RING, mode="nearest")
    mn = nd.minimum_filter(g, footprint=RING, mode="nearest")
    dark, bright = mn - g, g - mx
    c = {t: (int((dark > t).sum()), int((bright > t).sum())) for t in (40, 80, 120)}
    print(path, "dark/bright beyond all 8 neighbours by >40, >80, >120 levels:", c)
    for name, m in (("dark", dark), ("bright", bright)):
        idx = np.argsort(m, axis=None)[::-1][:10]
        print(f"  {name}:", [(int(k % g.shape[1]), int(k // g.shape[1]), int(m.flat[k])) for k in idx])


if __name__ == "__main__":
    for p in sys.argv[1:]:
        scan(p)
