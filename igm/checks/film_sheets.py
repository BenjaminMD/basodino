"""Review sheets of the two films of the fixed ensemble (DESIGN.md F7 review): a 4 x 4 contact sheet of each film
(every 330th map frame, every 260th 3D frame, 640 x 360 each), the 3D frames 1892-1897 (the firn year 1977, largest
steps of results/f7_film_steps.csv) and both end cards at full size.
usage (Mac, from basodino/):  python igm/checks/film_sheets.py   -> igm/results/f7_review/*.png
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILMS, OUT = ROOT / "tex/animation", ROOT / "igm/results/f7_review"
JOBS = (("igm_map_1850_2042.mp4", r"select='not(mod(n\,330))',scale=640:360,tile=4x4", "map_sheet.png"),
        ("igm_3d_1850_2042.mp4", r"select='not(mod(n\,260))',scale=640:360,tile=4x4", "film3d_sheet.png"),
        ("igm_3d_1850_2042.mp4", r"select='between(n\,1892\,1897)',scale=960:540,tile=2x3", "film3d_1892_1897.png"),
        ("igm_map_1850_2042.mp4", r"select=eq(n\,5120)", "map_card.png"),
        ("igm_3d_1850_2042.mp4", r"select=eq(n\,4050)", "film3d_card.png"))
OUT.mkdir(exist_ok=True)
for film, vf, name in JOBS:
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(FILMS / film), "-vf", vf, "-frames:v", "1", "-y",
                    str(OUT / name)], check=True)
    print(OUT / name)
