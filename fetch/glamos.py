"""Extract Ghiacciaio del Basodino (SGI C14-10) rows from the GLAMOS tables, keeping the header block
(citation lines, title, column names, units). usage: glamos.py  -> derived/glamos_basodino_<table>.csv"""
import re, glob, os

SGI_ID = re.compile(r"^[A-E]\d{2}[a-z]?-\d{2}$")
for src in sorted(glob.glob("raw/glamos/*.csv")):
    lines = open(src, encoding="utf-8").read().splitlines()
    col = 0 if os.path.basename(src) == "volumechange.csv" else 1  # position of the SGI id
    first = next(i for i, l in enumerate(lines) if SGI_ID.match((l.split(",") + [""] * 2)[col].strip()))
    rows = [l for l in lines[first:] if l.split(",")[col].strip() == "C14-10"]
    dst = "derived/glamos_basodino_" + os.path.basename(src)
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(lines[:first] + rows) + "\n")
    print(dst, len(rows), "rows;", first, "header lines")
