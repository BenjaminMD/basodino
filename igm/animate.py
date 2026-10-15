"""Map film of the fixed ensemble's film member, 1850 to 3 years after its disappearance (DESIGN.md step 7, Outputs:
'the draw with the median disappearance year under SSP2-4.5', 2026 level anchored per answer 2; author's rule of
2026-10-09: the film ends at the member's disappearance year plus 3 years).

Film member and runs: projection.py film (results/f6_film.csv): of the 1664 anchored SSP2-4.5 members (u of the draw)
those with the median disappearance year at 0.01 km2, the one closest to the group's median area series; its three
runs were rerun with yearly thk, us, vs (stage f6_film, results/f6_film_check.csv): f_hDDD the draw's hindcast
1850.75-2026.75 (F5 line), f_SAMPLE_21 the draw's decision-8 path 2021.75-2026.75 from the 2021 DEM (F3 line), f_MEMBER
the projection 2026.75 to the no-ice stop (F6 line). The path and projection reruns equal the ensemble runs; the
cold-started hindcast agrees with f5_hind to solver noise. The first ensemble's version of this script (pick, check;
stage s7_film, results/first_ensemble/step7_film.csv) is in the git history.
states (s02, agl env): per model year the 1 October state: 1850-2020 the hindcast, 2021-2025 the 2021 run, 2026 the
ensemble's start state (inputs/f5_state2026.npz thk_dec8, = the 2021 run's end), 2027 on the projection; after the
run's no-ice stop the thickness is 0; to the end year END = disappearance year (0.01 km2) + 3.
Film, 1920 x 1080 at 60 frames per second: 24 frames per model year from 1850 to the stop year or END (thickness and
surface velocity linear in time between yearly states; the year 2020-2021 runs to the hindcast's own 2021 state), 6 per
year after the stop; at 2021 the hindcast state is held 0.5 s, cross-faded in 1 s into the 2021 start state from the
measured thickness and held 0.5 s (labelled); 1 s still at the start and at the end, 5 s end card (the map fades to
white in 0.5 s, then the text fades in). Map: ice thickness of the cells with at least 1 m (thinner ice pale), arrows
of the surface velocity every 100 m on the ice, over a hillshade of the modelled surface (bed + thickness); the mapped
outline of the latest inventory before the frame's year as a thin red line. Side panel: name, year, volume and area of
the frame, a status line, the member's volume curve 1850-END, colour bar and arrow key. H.264 CRF 16, BT.709, constant
60 frames per second.
usage (s02, agl env, from igm/; the f6_film runs extracted from runs_f6.tar.gz to work/f6_film/runs/f6_film):
  python animate.py states                 -> results/f7/film_states.npz
  nrun python animate.py frames [N]        -> work/film_map_f7/f_NNNNN.png (N processes, default 16)
  nrun python animate.py encode            -> results/f7/igm_map_1850_END.mp4 (then to tex/animation/)
"""
import csv
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parent
RES = IGM / "results"
FILM = RES / "f6_film.csv"
FILM_RUNS = IGM / "work/f6_film/runs/f6_film"
STATES = IGM / "results/f7/film_states.npz"
FRAMES = IGM / "work/film_map_f7"
FPS, PER_YEAR, PER_YEAR_AFTER = 60, 24, 6
ARROW, GAIN = 6.0, 20.0             # map metres of arrow per m a-1 of surface speed in the hindcast; GAIN times
                                    # longer from the 2021 restart on (speeds 0.03-0.4 m a-1 against 1.5-25 before)
OUTLINES = ((1850, "sgi1850", "Swiss Glacier Inventory 1850"), (1929, "sgi1931", "Swiss Glacier Inventory 1931"),
            (1973, "sgi1973", "Swiss Glacier Inventory 1973"), (2010, "sgi2010", "Swiss Glacier Inventory 2010"),
            (2018, "sgi2016", "Swiss Glacier Inventory 2016"), (2024, "tlm2024", "swissTLM3D 2024"),
            (2026, "s2_2026", "Our outline, Sentinel-2, 27 September 2026"))


def film_info():
    return {r["quantity"]: r["value"] for r in csv.DictReader(open(FILM))}


def mp4():
    return IGM / f"results/f7/igm_map_1850_{int(np.load(STATES)['end'])}.mp4"


def states():
    F = film_info()
    d, name, sample = int(F["draw"]), F["film member"], F["sample"]
    dis = int(F["disappearance 0.01 | 0.1 km2, half-volume 2026 | 2021"].split("|")[0])
    end = dis + 3                                               # author's rule (DESIGN.md, fix round)
    H = np.load(FILM_RUNS / f"f_h{d:03d}/fields.npz")
    P21 = np.load(FILM_RUNS / f"f_{sample}_21/fields.npz")
    PR = np.load(FILM_RUNS / f"f_{name}/fields.npz")
    S = np.load(IGM / "inputs/f5_state2026.npz")
    stop = max(int(k[5:]) for k in PR.files if k.startswith("thk_y"))     # the save without ice (no-ice stop)
    years = np.arange(1850, end + 1)
    shape = H["topg"].shape
    thk, us, vs = (np.zeros((len(years),) + shape, np.float32) for _ in range(3))
    src = []
    for i, Y in enumerate(years):
        Z = H if Y <= 2020 else P21 if Y <= 2025 else PR if Y <= stop else None
        if Z is not None:
            thk[i], us[i], vs[i] = Z[f"thk_y{Y}"], Z[f"us_y{Y}"], Z[f"vs_y{Y}"]
        if Y == 2026:
            assert np.array_equal(S["thk_dec8"][d], PR["thk_y2026"])
            assert np.array_equal(S["thk_dec8"][d], P21["thk_y2026"])
            thk[i] = S["thk_dec8"][d]
        src.append("hindcast" if Y <= 2020 else "2021 run" if Y <= 2025 else "projection" if Y <= stop else "none")
    assert np.array_equal(H["topg"], S["topg_dec8"][d]) and np.array_equal(P21["topg"], S["topg_dec8"][d])
    np.savez_compressed(STATES, years=years, thk=thk, us=us, vs=vs, topg=H["topg"], stop=stop, dis=dis, end=end,
                        thk_hind_2021=H["thk_y2021"], us_hind_2021=H["us_y2021"], vs_hind_2021=H["vs_y2021"],
                        source=np.array(src))
    V = thk.sum((1, 2)) * 100 / 1e9
    A = (thk >= 1).sum((1, 2)) / 1e4
    print(f"{STATES}: {name}, years 1850-{end}, disappearance {dis}, no-ice stop {stop}; V 1850 {V[0]:.4f} km3, A1 "
          f"{A[0]:.3f} km2; 2021 hindcast V {H['thk_y2021'].sum() * 1e-7:.4f} km3, A1 "
          f"{(H['thk_y2021'] >= 1).sum() / 1e4:.3f} km2; 2021 start V {V[171]:.4f} km3, A1 {A[171]:.3f} km2; 2026 V "
          f"{V[176]:.5f} km3, A1 {A[176]:.3f} km2; {end} V {V[-1]:.6f} km3, A1 {A[-1]:.4f} km2")


def timeline(stop, end):
    """[(kind, a, b, s, label year)] per frame: state a to state b at fraction s (indices into years 1850-end)."""
    last = end - 1850
    out = [("state", 0, 0, 0.0, 1850)] * FPS
    for i in range(0, 171):                                     # 1850 -> 2021 (hindcast)
        out += [("state", i, i + 1, k / PER_YEAR, 1850 + i) for k in range(PER_YEAR)]
    out += [("hind2021", 0, 0, 0.0, 2021)] * (FPS // 2)
    out += [("switch", 0, 0, k / FPS, 2021) for k in range(FPS)]
    out += [("state", 171, 171, 0.0, 2021)] * (FPS // 2)
    for i in range(171, last):                                  # 2021 -> end
        n = PER_YEAR if 1850 + i < stop else PER_YEAR_AFTER
        out += [("state", i, i + 1, k / n, 1850 + i) for k in range(n)]
    out += [("state", last, last, 0.0, end)] * FPS
    out += [("card", 0, 0, k / (5 * FPS), end) for k in range(5 * FPS)]
    return out


def _font_dir():
    for p in (IGM.parent / "data/satellite_archive/sentinel1/scripts/fonts",
              IGM.parent / "derived/satellite_archive/sentinel1/scripts/fonts"):
        if p.exists():
            return p
    raise FileNotFoundError("Source Sans 3")


_W = {}


def _init():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for w in ("Regular", "Semibold"):
        font_manager.fontManager.addfont(_font_dir() / f"SourceSans3-{w}.ttf")
    plt.rcParams.update({"font.family": "Source Sans 3", "text.color": "#1a1a1a", "axes.edgecolor": "#4d4d4d",
                         "xtick.color": "#4d4d4d", "ytick.color": "#4d4d4d"})
    Z = np.load(STATES)
    _W.update({k: Z[k] for k in Z.files})
    st = np.load(IGM / "inputs/igm_static.npz")
    _W["x"], _W["y"] = st["x"].astype(float), st["y"].astype(float)
    _W["masks"] = {k: st[k] for _, k, _ in OUTLINES}
    _W["V"] = _W["thk"].sum((1, 2)) * 100 / 1e9
    _W["plt"] = plt


def _frame(job):
    k, (kind, a, b, s, year) = job
    plt = _W["plt"]
    from matplotlib.colors import LightSource, ListedColormap, PowerNorm
    x, y, years = _W["x"], _W["y"], _W["years"]
    if kind == "state":
        # the last hindcast year (2020 -> 2021) ends on the hindcast's own 2021 state, not on the start state
        end = {k: _W[f"{k}_hind_2021"] if (a, b) == (170, 171) else _W[k][b] for k in ("thk", "us", "vs")}
        thk = (1 - s) * _W["thk"][a] + s * end["thk"]
        us = (1 - s) * _W["us"][a] + s * end["us"]
        vs = (1 - s) * _W["vs"][a] + s * end["vs"]
        t = years[a] + 0.75 + s * (years[b] - years[a])
    elif kind == "hind2021":
        thk, us, vs, t = _W["thk_hind_2021"], _W["us_hind_2021"], _W["vs_hind_2021"], 2021.75
    elif kind == "switch":
        thk = (1 - s) * _W["thk_hind_2021"] + s * _W["thk"][171]
        us = (1 - s) * _W["us_hind_2021"] + s * _W["us"][171]
        vs = (1 - s) * _W["vs_hind_2021"] + s * _W["vs"][171]
        t = 2021.75
    else:
        thk, us, vs, t = _W["thk"][-1], _W["us"][-1], _W["vs"][-1], int(_W["end"]) + 0.75
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    white = kind == "card" and s * 5 < 0.5                        # first 0.5 s of the card: the last map fades to white
    if kind == "card" and not white:
        a_in = min(1.0, (s * 5 - 0.5) / 0.8)
        lines = _card_lines()
        fig.text(0.08, 0.86, lines[0], fontsize=34, fontweight=600, alpha=a_in, va="top")
        for i, l in enumerate(lines[1:]):
            fig.text(0.08, 0.76 - i * 0.052, l, fontsize=19, alpha=a_in, va="top", color="#1a1a1a")
        fig.savefig(FRAMES / f"f_{k:05d}.png", dpi=100)
        plt.close(fig)
        return
    e0, e1, n0, n1 = x[0] - 5, x[-1] + 5, y[0] - 5, y[-1] + 5          # the whole model domain
    mw = 0.96 * 10.8 * (e1 - e0) / (n1 - n0) / 19.2
    ax = fig.add_axes([0.012, 0.02, mw, 0.96])
    surf = _W["topg"] + thk
    hs = LightSource(azdeg=315, altdeg=40).hillshade(surf[::-1], dx=10, dy=10)
    ext = (x[0] - 5, x[-1] + 5, y[0] - 5, y[-1] + 5)
    ax.imshow(hs, extent=ext, cmap="gray", vmin=-0.4, vmax=1.25, interpolation="bilinear")
    thin = np.where((thk > 0.05) & (thk < 1.0), 1.0, np.nan)
    ax.imshow(thin[::-1], extent=ext, cmap="Blues", vmin=0, vmax=8, alpha=0.55, interpolation="bilinear")
    ice = np.where(thk >= 1.0, thk, np.nan)
    cmap = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.22, 1.0, 256)))   # 1 m already visibly blue
    im = ax.imshow(ice[::-1], extent=ext, cmap=cmap, norm=PowerNorm(0.6, vmin=0, vmax=50), alpha=0.92,
                   interpolation="bilinear")
    cal = int(np.floor(t))
    oy, okey, oname = max((o for o in OUTLINES if o[0] <= cal), key=lambda o: o[0])
    ax.contour(x, y, _W["masks"][okey].astype(float), levels=[0.5], colors=["#c51415"], linewidths=1.2)
    new = kind == "switch" or (kind == "state" and a >= 171)               # arrows of the restarted ice
    scale = ARROW * GAIN ** s if kind == "switch" else ARROW * GAIN if new else ARROW
    sub = (slice(5, None, 10), slice(5, None, 10))
    X, Y = np.meshgrid(x, y)
    on = thk[sub] >= 1.0
    if on.any():
        ax.quiver(X[sub][on], Y[sub][on], us[sub][on], vs[sub][on], angles="xy", scale_units="xy", scale=1 / scale,
                  width=0.0018, headwidth=4, headlength=4.5, color="#1a1a1a", alpha=0.85)
    ax.set_xlim(e0, e1), ax.set_ylim(n0, n1)
    ax.set_axis_off()
    ax.plot([e1 - 620, e1 - 120], [n0 + 90, n0 + 90], color="#1a1a1a", lw=3, solid_capstyle="butt")
    ax.text(e1 - 370, n0 + 120, "500 m", ha="center", va="bottom", fontsize=16)
    ax.annotate("", xy=(e0 + 150, n1 - 120), xytext=(e0 + 150, n1 - 330),
                arrowprops=dict(arrowstyle="-|>", color="#1a1a1a", lw=2))
    ax.text(e0 + 150, n1 - 100, "North", ha="center", va="bottom", fontsize=15)
    # side panel
    fig.text(0.66, 0.93, "Ghiacciaio del Basòdino", fontsize=30, fontweight=600, va="top")
    fig.text(0.66, 0.875, "Ice flow model, one ensemble member", fontsize=18, va="top", color="#4d4d4d")
    fig.text(0.66, 0.80, f"{cal}", fontsize=64, fontweight=600, va="top")
    vol = thk.sum() * 100 / 1e9
    area = (thk >= 1.0).sum() * 100 / 1e6
    fig.text(0.66, 0.685, f"Ice volume {vol * 1000:.2f} million m³", fontsize=20, va="top")
    fig.text(0.66, 0.645, f"Ice area {area:.3f} km²", fontsize=20, va="top")
    note = ("Hindcast from a steady 1850 state" if t < 2021.75 or kind in ("hind2021",) else
            "2021: start from the measured ice thickness" if kind == "switch" or (kind == "state" and a == 171 and b
                                                                                    == 171) else
            "Observed weather to 2026" if t <= 2026.75 else
            "Scenario SSP2-4.5, 2026 warming anchored to observation" if t < _W["dis"] + 0.75 else
            f"Ice area below 0.01 km² from {int(_W['dis'])}" if t < _W["stop"] + 0.75 else
            f"No ice of 1 m or more from {int(_W['stop'])}")
    fig.text(0.66, 0.595, note, fontsize=17, va="top", color="#c51415" if "measured" in note else "#4d4d4d")
    fig.text(0.66, 0.555, f"Red line: {oname}", fontsize=15, va="top", color="#4d4d4d")
    cax = fig.add_axes([0.66, 0.455, 0.30, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([1, 5, 10, 20, 30, 40, 50])
    cb.ax.tick_params(labelsize=14)
    cb.set_label("Ice thickness (m)", fontsize=16)
    fig.text(0.66, 0.375, "Surface speed", fontsize=16, va="center", color="#1a1a1a")
    kx = fig.add_axes([0.745, 0.36, 0.16, 0.03])
    kx.set_axis_off()
    kx.set_xlim(0, 1), kx.set_ylim(0, 1)
    late = scale > ARROW * GAIN ** 0.5
    L = 20 * ARROW / (e1 - e0) * mw / 0.16                    # 20 m a-1 at the hindcast scale, in key-axes units
    kx.annotate("", xy=(L, 0.5), xytext=(0, 0.5), arrowprops=dict(arrowstyle="-|>", color="#1a1a1a", lw=1.4))
    kx.text(L + 0.04, 0.5, f"{20 / GAIN if late else 20:g} m per year", va="center", fontsize=15, color="#4d4d4d")
    if late:
        fig.text(0.66, 0.325, f"From 2021 the arrows are {GAIN:g} times longer", fontsize=14, va="center",
                 color="#4d4d4d")
    gx = fig.add_axes([0.70, 0.08, 0.27, 0.22])
    gx.plot(years + 0.75, _W["V"] * 1000, color="#b3b3b3", lw=1.2)
    done = years + 0.75 <= t
    gx.plot((years + 0.75)[done], _W["V"][done] * 1000, color="#027abe", lw=2)
    gx.plot([t], [vol * 1000], "o", color="#027abe", ms=8)
    gx.set_xlim(1850, int(_W["end"]) + 1), gx.set_ylim(0, _W["V"].max() * 1100)
    gx.set_ylabel("Ice volume (million m³)", fontsize=15)
    gx.tick_params(labelsize=13)
    gx.spines[["top", "right"]].set_visible(False)
    if white:
        fig.patches.append(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color="white", alpha=s * 5 / 0.5,
                                         zorder=1000))
    fig.savefig(FRAMES / f"f_{k:05d}.png", dpi=100)
    plt.close(fig)


def basin_step():
    """Highest ice face at the basin edge of F2 in the hindcast years (results/f7_basin_wall.csv, checks/basin_wall.py),
    rounded down to 10 m."""
    r = [float(x["largest_face_at_basin_edge_m"]) for x in csv.DictReader(open(RES / "f7_basin_wall.csv"))
         if int(x["year"]) <= 2020]
    return 10 * (max(r) // 10)


def _card_lines():
    F = film_info()
    years = list(_W["years"])
    A = lambda Y: (_W["thk"][years.index(Y)] >= 1.0).sum() / 1e4
    A21 = (_W["thk_hind_2021"] >= 1.0).sum() / 1e4
    return [f"Ghiacciaio del Basòdino 1850–{int(_W['end'])}, modelled",
            "Instructed Glacier Model 3.2.0, 10 m grid, one member of the 9984-member ensemble:",
            f"posterior draw {F['draw']}, climate model chain {F['chain']} (CH2025), scenario SSP2-4.5,",
            "2026 warming anchored to the observed trend at Locarno-Monti; the member with the median",
            f"disappearance year ({F['median disappearance year 0.01 km2 (lower middle)']}, ice area below 0.01 km²) "
            f"of its {F['members in the group']} scenario members; the film ends 3 years later.",
            f"1850–2021: hindcast from a steady 1850 state. Its ice area on 1 October: 1929 {A(1929):.2f}, "
            f"1973 {A(1973):.2f}, 2010 {A(2010):.2f},",
            f"2018 {A(2018):.2f} km² (mapped 3.20, 2.32, 1.90, 1.64 km²); 2021 {A21:.2f} km².",
            f"2021: restart from the measured ice thickness ({A(2021):.2f} km²); modelled ice area in 2026 "
            f"{A(2026):.2f} km², mapped 1.02 km².",
            "Ice: thickness of 1 m or more on 1 October, linear between the yearly states. Model basin: ice that flows",
            f"beyond the 1850 outline plus 100 m is removed, so until 2021 the hindcast ice ends there in a step of up to "
            f"{basin_step():.0f} m.",
            "Data: swisstopo, Glacier Monitoring Switzerland, MeteoSwiss, Swiss Glacier Inventory, Copernicus Sentinel-2."]


def frames(n=16):
    from multiprocessing import Pool
    Z = np.load(STATES)
    jobs = list(enumerate(timeline(int(Z["stop"]), int(Z["end"]))))
    FRAMES.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (FRAMES / f"f_{j[0]:05d}.png").exists()]
    with Pool(n, initializer=_init) as p:
        for i, _ in enumerate(p.imap_unordered(_frame, todo, chunksize=8)):
            if i % 500 == 0:
                print(f"{i} of {len(todo)}", flush=True)
    print(f"{FRAMES}: {len(jobs)} frames ({len(jobs) / FPS:.1f} s)")


def encode():
    import subprocess
    n = len(list(FRAMES.glob("f_*.png")))
    Z = np.load(STATES)
    assert n == len(timeline(int(Z["stop"]), int(Z["end"]))), n
    MP4 = mp4()
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(FPS), "-i",
                    str(FRAMES / "f_%05d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt",
                    "yuv420p", "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", "-r", str(FPS),
                    "-movflags", "+faststart", str(MP4)], check=True)
    print(f"{MP4}: {n} frames, {n / FPS:.1f} s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "frames":
        frames(int(sys.argv[2]) if len(sys.argv) > 2 else 16)
    else:
        {"states": states, "encode": encode}[cmd]()
