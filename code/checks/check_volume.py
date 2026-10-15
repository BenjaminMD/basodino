"""Independent check of step 1 (volume_clock.py), from the GLAMOS files and the outline table only.

Re-derives, without reading volume_clock.csv: GLAMOS's rates from dV, areas and whole years; the k values
(geodetic over glaciological, observation periods chained survey year to survey year); the chain 30 Sep 2021 to
29 Sep 2026 with its gap; the loss over the map's 2021 and 2012 volumes; the density choices (900 for balances to
ice, as approved; 850, the geodetic convention k is calibrated on, as the alternative). Also asks whether GLAMOS's
sigma is per year: over the century-long map pairs, sigma x n is the only plausible total.

Inputs: data/glamos_basodino_volumechange.csv, data/glamos_basodino_massbalance_observation.csv,
        code/outline_2026.csv (areas), code/thickness.csv (V_2021, V_2012 of the map)
Output: code/checks/check_volume.csv (quantity, value, note)
usage:  python code/checks/check_volume.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B_2026 = -3.503                                    # m w.e., Fig. A.1, 6 Sep 2025 - 29 Sep 2026 (provisional)


def rows_of(name, gid="C14-10"):
    with open(ROOT / name, encoding="utf-8") as f:
        return [r for r in csv.reader(f) if len(r) > 10 and r[1 if "observation" in name else 0] == gid]


vc = {(r[3], r[4]): r for r in rows_of("data/glamos_basodino_volumechange.csv")}
obs = [(r[2], r[4], float(r[7]) / 1000) for r in rows_of("data/glamos_basodino_massbalance_observation.csv")]
with open(ROOT / "code/outline_2026.csv") as f:
    area = {r["scene"][:10]: float(r["area_m2"]) for r in csv.DictReader(f)}
with open(ROOT / "code/thickness.csv") as f:
    mp = {r["quantity"]: float(r["value"]) for r in csv.DictReader(f)}
out = []


def put(q, v, note=""):
    out.append((q, f"{v:.6g}" if isinstance(v, float) else v, note))
    print(f"{q:40s} {v if isinstance(v, str) else f'{v:.5g}':>12s}  {note}")


def geod(s, e):
    r = vc[s, e]
    dv, a, n = float(r[9]) * 1e9, (float(r[5]) + float(r[7])) / 2 * 1e6, int(e[:4]) - int(s[:4])  # m3, m2
    return dv, a, n, float(r[11]), float(r[12])


for s, e in (("20129999", "20240823"), ("20219999", "20240823"), ("20129999", "20219999")):
    dv, a, n, b, sb = geod(s, e)
    put(f"rate {s[:4]}-{e[:4]}", 0.85 * dv / a / n, f"published {b}, sigma {sb}")
    glac = [o for o in obs if int(s[:4]) <= int(o[0][:4]) and int(o[1][:4]) <= int(e[:4])]
    chained = all(glac[i][1] == glac[i + 1][0] for i in range(len(glac) - 1))
    k = (0.85 * dv / a) / sum(o[2] for o in glac)
    put(f"k {s[:4]}-{e[:4]}", k, f"+- {k * sb * n / abs(0.85 * dv / a):.4f}; {len(glac)} periods "
        f"{glac[0][0]} to {glac[-1][1]}, chained {chained}")

# chain 2021 basis: survey to 2024-08-23, gap, 2024-09-03..2025-09-06 (GLAMOS), 2025-09-06..2026-09-29 (Fig. A.1)
b25 = next(o for o in obs if o[0] == "2024-09-03")
put("2024/25 period", f"{b25[0]} to {b25[1]}", f"Ba {b25[2]}; 2025/26 starts 2025-09-06: joins {b25[1] == '2025-09-06'}")
k = float(out[[q for q, _, _ in out].index("k 2012-2024")][1])
a1, a2 = (area["2024-09-10"] + area["2025-09-07"]) / 2, (area["2025-09-07"] + area["2026-09-27"]) / 2
for rho, label in ((900, "ice 900 (approved)"), (850, "850, geodetic convention of k")):
    dv21 = -geod("20219999", "20240823")[0]
    loss = dv21 - k * (b25[2] * a1 + B_2026 * a2) / (rho / 1000)
    put(f"loss 2021-2026 [{label}]", loss / 1e9, f"km3; ratio to map {loss / mp['V_2021']:.4f}")
    dv12 = -geod("20129999", "20240823")[0]
    loss12 = dv12 - k * (b25[2] * a1 + B_2026 * a2) / (rho / 1000)
    put(f"loss 2012-2026 [{label}]", loss12 / 1e9, f"km3; ratio to map 2012 {loss12 / mp['V_2012']:.4f}")
put("ratio 2012-2024 (national row)", -geod("20129999", "20240823")[0] / mp["V_2012"])

# is sigma per year? read as a whole-period error, the map pairs would be known to 0.3 m w.e. over a century
for st in ("19079999", "19299999"):
    dv, a, n, b, sb = geod(st, "20129999")
    put(f"sigma x n, {st[:4]}-2012", sb * n, f"m w.e.: the per-year reading; a whole-period reading gives {sb} m w.e. "
        f"for a {n}-year map pair, implausible")
with open(ROOT / "code/checks/check_volume.csv", "w", newline="") as f:
    csv.writer(f).writerows([("quantity", "value", "note"), *out])
