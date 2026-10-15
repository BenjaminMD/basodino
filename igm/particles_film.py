"""Map film of the ice flow drawn as moving particles (DESIGN.md step 7, after review of the arrow film animate.py).

Inputs   results/step7/film_states.npz (animate.py states): the film member of the first ensemble (d056_c07_2,
         results/step7_film.csv), its 1 October thickness thk and surface velocity us, vs (m per year) for every year
         1850-2100 (1850-2020 hindcast, 2021-2025 the 2021 run, 2026 the ensemble start state, then the projection; all
         three runs were rerun with yearly fields, so every year to the no-ice stop has its own velocity), the
         hindcast's own 2021 state and the bed topg; inputs/igm_static.npz (x, y of the 10 m grid);
         results/step7_film.csv (the member's disappearance year, ice area below 0.01 km2); Source Sans 3.
Method   Cut (author's rule, DESIGN.md): the film ends 3 years after the member's disappearance year (2034 + 3 = 2037),
         then the end card; no empty years to 2100.
         Time line as in animate.py: thickness and velocity linear in time between the yearly states; the year
         2020-2021 runs to the hindcast's own 2021 state, which is held 0.5 s, cross-faded in 1 s into the 2021 start
         from the measured thickness and held 0.5 s; a still at the start and at the end, then the end card (the last
         map fades to white in 0.5 s, then the text fades in). Two films:
           full   1850 to the end year, 24 frames per model year (0.4 s), 5 s card
           short  2000 to the end year, 36 frames per model year (0.6 s), 30 s in all
         Particles (track, sequential, fixed random seed): one per DENSITY cells of ice (thickness >= 1 m), so their
         number follows the ice area (6600 in 1850). Each frame they are carried by the surface velocity, interpolated
         bilinearly in space and linearly in time between the frames' fields, by Heun's second-order Runge-Kutta with
         substeps of at most 4 m; the model time between frames is multiplied by the speed-up factor of the era
         (FACTOR: the hindcast, and from the 2021 restart on, when the ice is thin and slow), stated on screen (the
         old factor fades out in the first half of the 2021 switch, the new one in during the second). No motion while the film holds. A particle fades out (0.3 s) when its cell
         has no ice of 1 m, when it leaves the grid, or at the end of its life (2.5-5 s of film while moving); new ones
         fade in on random ice cells while there are fewer than the density asks for.
         Frames (parallel): hillshade of the modelled surface (bed + thickness) under a pale ice tint (ice of 1 m or
         more; thinner ice paler), trails of the last TRAIL frames (tapered, fading, coloured by the speed of the
         segment), heads as dots, colour = surface speed on a logarithmic scale (small colour bar); side panel with
         year, volume, area, status line, time scale and speed-up factor, volume curve; end card with the caveats.
         Encode: H.264 CRF 16, BT.709, constant 60 frames per second, 1920 x 1080.
         Check (s02): from the tracks, the measured speed-up (displacement per frame over model time times the
         interpolated speed), the share of moves downhill on the modelled surface, visible particles on cells without
         ice of 1 m (and whether they are fading out); from the films, ffprobe rate and frame count and the
         frame-to-frame steps (480 x 270, mean absolute luma difference; unchanged frames outside the stills).
Outputs  work/particles_FILM/{x,y,a,s,b}.npy (position, opacity, speed, birth frame; frame x slot) and f_NNNNN.png
         (intermediate); results/step7/igm_particles_1850_2037.mp4 (full), igm_particles_2000_2037.mp4 (short),
         copied to tex/animation/; results/step7_particles_check.csv.
usage    s02 (agl env, from igm/, after animate.py states):
           python particles_film.py track FILM                 -> work/particles_FILM/{x,y,a,s,b}.npy
           nrun python particles_film.py frames FILM [N]       -> work/particles_FILM/f_NNNNN.png (N processes, 16)
           nrun python particles_film.py encode FILM           -> results/step7/igm_particles_FIRST_END.mp4
           python particles_film.py check                      -> results/step7_particles_check.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

IGM = Path(__file__).resolve().parent
RES = IGM / "results"
STATES = RES / "step7/film_states.npz"
FPS = 60
FILMS = {  # first year, frames per year, start still, end still, card (frames), speed-up factors before | from 2021
    "full": dict(first=1850, per_year=24, start=60, end=60, card=300, factor=(3, 100)),
    "short": dict(first=2000, per_year=36, start=60, end=60, card=228, factor=(10, 100)),
}
DENSITY = 8          # ice cells (10 m) per particle
FADE = 18            # frames of fade in and out
LIFE = (150, 300)    # frames of life while moving
TRAIL = 24           # frames of trail
SPEED = (0.03, 60.0)  # colour range of the surface speed, m per year (logarithmic)
I2021 = 171          # index of 2021 in years 1850-2100


def read_csv(path):
    return list(csv.DictReader(open(path)))


def member():
    """results/step7_film.csv as a dict, with the disappearance year (0.01 km2) and the end year (+ 3)."""
    F = {r["quantity"]: r["value"] for r in read_csv(RES / "step7_film.csv")}
    F["dis"] = int(F["disappearance 0.01 | 0.1 km2, half-volume 2026 | 2021"].split("|")[0])
    F["last"] = F["dis"] + 3
    return F


def work(film):
    return IGM / f"work/particles_{film}"


def mp4(film):
    return RES / f"step7/igm_particles_{FILMS[film]['first']}_{member()['last']}.mp4"


def timeline(film, last):
    """[(kind, a, b, s, year)] per frame: state a to state b at fraction s (indices into years 1850-2100)."""
    c = FILMS[film]
    i0, iE, P = c["first"] - 1850, last - 1850, c["per_year"]
    out = [("state", i0, i0, 0.0, c["first"])] * c["start"]
    for i in range(i0, I2021):
        out += [("state", i, i + 1, k / P, 1850 + i) for k in range(P)]
    out += [("hind2021", 0, 0, 0.0, 2021)] * (FPS // 2)
    out += [("switch", 0, 0, k / FPS, 2021) for k in range(FPS)]
    out += [("state", I2021, I2021, 0.0, 2021)] * (FPS // 2)
    for i in range(I2021, iE):
        out += [("state", i, i + 1, k / P, 1850 + i) for k in range(P)]
    out += [("state", iE, iE, 0.0, last)] * c["end"]
    out += [("card", iE, iE, k / c["card"], last) for k in range(c["card"])]
    return out


def field(Z, job):
    """Thickness, us, vs and model time of one frame (as animate.py _frame); the card holds the last state."""
    kind, a, b, s, _ = job
    if kind in ("state", "card"):
        # the last hindcast year (2020 -> 2021) ends on the hindcast's own 2021 state, not on the start state
        end = {q: Z[f"{q}_hind_2021"] if (a, b) == (I2021 - 1, I2021) else Z[q][b] for q in ("thk", "us", "vs")}
        f = [(1 - s) * Z[q][a] + s * end[q] for q in ("thk", "us", "vs")]
        return f + [1850.75 + a + s * (b - a)]
    if kind == "hind2021":
        return [Z["thk_hind_2021"], Z["us_hind_2021"], Z["vs_hind_2021"], 2021.75]
    return [(1 - s) * Z[f"{q}_hind_2021"] + s * Z[q][I2021] for q in ("thk", "us", "vs")] + [2021.75]


def load_states():
    Z = np.load(STATES)
    Z = {k: Z[k] for k in Z.files}
    st = np.load(IGM / "inputs/igm_static.npz")
    Z["x"], Z["y"] = st["x"].astype(float), st["y"].astype(float)
    return Z


# --- particles ---------------------------------------------------------------------------------------------------------

def bilinear(F, fi, fj):
    """F at fractional row fi, column fj (clipped to the grid)."""
    n, m = F.shape
    fi, fj = np.clip(fi, 0, n - 1), np.clip(fj, 0, m - 1)
    i, j = np.minimum(fi.astype(int), n - 2), np.minimum(fj.astype(int), m - 2)
    u, v = fi - i, fj - j
    return ((1 - u) * (1 - v) * F[i, j] + (1 - u) * v * F[i, j + 1] + u * (1 - v) * F[i + 1, j]
            + u * v * F[i + 1, j + 1])


def factor(film, t_prev):
    """Speed-up factor of the frame step that starts at model time t_prev."""
    f1, f2 = FILMS[film]["factor"]
    return f1 if t_prev < 2021.75 else f2


def track(film):
    Z = load_states()
    x0, y0, nx, ny = Z["x"][0], Z["y"][0], len(Z["x"]), len(Z["y"])
    last = member()["last"]
    tl = timeline(film, last)
    n = len(tl)
    cells = np.array([(Z["thk"][i] >= 1).sum() for i in range(FILMS[film]["first"] - 1850, last - 1850 + 1)]
                     + [(Z["thk_hind_2021"] >= 1).sum()])
    N = int(cells.max() / DENSITY * 1.15) + 50
    rng = np.random.default_rng(20261009)
    X, Y, A = np.zeros(N), np.zeros(N), np.zeros(N)
    age, life, birth = np.zeros(N, int), np.zeros(N, int), np.full(N, -1, np.int32)
    alive, dying = np.zeros(N, bool), np.zeros(N, bool)
    out = {q: np.lib.format.open_memmap(work(film) / f"{q}.npy", "w+", dt, (n, N)) for q, dt in
           (("x", np.float32), ("y", np.float32), ("a", np.float16), ("s", np.float16), ("b", np.int32))}

    def spawn(thk, k, alpha):
        ice = np.flatnonzero(thk.ravel() >= 1)
        need = min(int(round(len(ice) / DENSITY)) - int((alive & ~dying).sum()), int((~alive).sum()))
        if need <= 0 or len(ice) == 0:
            return
        slots = np.flatnonzero(~alive)[:need]
        c = rng.choice(ice, need)
        X[slots] = x0 + 10 * (c % nx) + rng.uniform(-5, 5, need)
        Y[slots] = y0 + 10 * (c // nx) + rng.uniform(-5, 5, need)
        A[slots], age[slots], life[slots], birth[slots] = alpha, 0, rng.integers(*LIFE, need), k
        alive[slots], dying[slots] = True, False

    def velocity(fp, fc, tau, px, py):
        fi, fj = (py - y0) / 10, (px - x0) / 10
        return [(1 - tau) * bilinear(fp[q], fi, fj) + tau * bilinear(fc[q], fi, fj) for q in (1, 2)]

    prev = field(Z, tl[0])
    spawn(prev[0], 0, 1.0)
    for k in range(n):
        cur = field(Z, tl[k]) if k else prev
        h = (cur[3] - prev[3]) * factor(film, prev[3])
        idx = np.flatnonzero(alive)
        if h > 0 and len(idx):
            px, py = X[idx], Y[idx]
            vmax = max(np.abs(prev[1]).max(), np.abs(prev[2]).max(), np.abs(cur[1]).max(), np.abs(cur[2]).max())
            m = max(1, int(np.ceil(vmax * h / 4)))
            for q in range(m):                           # Heun: Euler predictor, trapezoidal corrector
                u0, v0 = velocity(prev, cur, q / m, px, py)
                u1, v1 = velocity(prev, cur, (q + 1) / m, px + h / m * u0, py + h / m * v0)
                px, py = px + h / m * (u0 + u1) / 2, py + h / m * (v0 + v1) / 2
            X[idx], Y[idx] = px, py
            age[idx] += 1
        thk = cur[0]
        i, j = np.rint((Y - y0) / 10).astype(int), np.rint((X - x0) / 10).astype(int)
        inside = (i >= 0) & (i < ny) & (j >= 0) & (j < nx)
        on = inside & (thk[np.clip(i, 0, ny - 1), np.clip(j, 0, nx - 1)] >= 1)
        dying |= alive & (~on | (age >= life))
        A[:] = np.where(dying, A - 1 / FADE, np.minimum(1, A + 1 / FADE))
        gone = alive & dying & (A <= 0)
        alive[gone], dying[gone] = False, False
        A[~alive] = 0
        spawn(thk, k, 1 / FADE)
        fi, fj = (Y - y0) / 10, (X - x0) / 10
        S = np.hypot(bilinear(cur[1], fi, fj), bilinear(cur[2], fi, fj))
        out["x"][k], out["y"][k], out["a"][k], out["b"][k] = X, Y, A, np.where(alive, birth, -1)
        out["s"][k] = np.where(alive, S, 0)
        prev = cur
        if k % 1000 == 0:
            print(f"frame {k} of {n}: {tl[k][4]}, {alive.sum()} particles", flush=True)
    for a in out.values():
        a.flush()
    live = (np.asarray(out["a"]) > 0).sum(1)
    print(f"{work(film)}: {n} frames, {N} slots, particles at most {live.max()}, first frame {live[0]}")


# --- frames ------------------------------------------------------------------------------------------------------------

def _font_dir():
    for p in (IGM.parent / "data/satellite_archive/sentinel1/scripts/fonts",
              IGM.parent / "derived/satellite_archive/sentinel1/scripts/fonts"):
        if p.exists():
            return p
    raise FileNotFoundError("Source Sans 3")


_W = {}


def _init(film):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for w in ("Regular", "Semibold"):
        font_manager.fontManager.addfont(_font_dir() / f"SourceSans3-{w}.ttf")
    plt.rcParams.update({"font.family": "Source Sans 3", "text.color": "#1a1a1a", "axes.edgecolor": "#4d4d4d",
                         "xtick.color": "#4d4d4d", "ytick.color": "#4d4d4d"})
    _W.update(load_states())
    _W["V"] = _W["thk"].sum((1, 2)) * 100 / 1e9
    _W["P"] = {q: np.load(work(film) / f"{q}.npy", mmap_mode="r") for q in "xyasb"}
    _W["plt"], _W["film"], _W["F"] = plt, film, member()


def _speed_cmap(plt):
    from matplotlib.colors import ListedColormap
    return ListedColormap(plt.get_cmap("inferno")(np.linspace(0, 0.85, 256)))


def _frame(job):
    k, (kind, a, b, s, year) = job
    plt, film, F = _W["plt"], _W["film"], _W["F"]
    c = FILMS[film]
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LightSource, ListedColormap, LogNorm
    path = work(film) / f"f_{k:05d}.png"
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    tc = s * c["card"] / FPS if kind == "card" else -1.0           # seconds into the card
    if tc >= 0.5:                                                  # the last map has faded to white: the text
        a_in = min(1.0, (tc - 0.5) / 0.8)
        lines = _card_lines(film)
        fig.text(0.08, 0.88, lines[0], fontsize=34, fontweight=600, alpha=a_in, va="top")
        for i, l in enumerate(lines[1:]):
            fig.text(0.08, 0.79 - i * 0.048, l, fontsize=19, alpha=a_in, va="top", color="#1a1a1a")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return
    thk, us, vs, t = field(_W, (kind, a, b, s, year))
    x, y, years = _W["x"], _W["y"], _W["years"]
    e0, e1, n0, n1 = x[0] - 5, x[-1] + 5, y[0] - 5, y[-1] + 5          # the whole model domain
    mw = 0.96 * 10.8 * (e1 - e0) / (n1 - n0) / 19.2
    ax = fig.add_axes([0.012, 0.02, mw, 0.96])
    hs = LightSource(azdeg=315, altdeg=40).hillshade((_W["topg"] + thk)[::-1], dx=10, dy=10)
    ext = (e0, e1, n0, n1)
    ax.imshow(hs, extent=ext, cmap="gray", vmin=-0.4, vmax=1.25, interpolation="bilinear")
    tint = ListedColormap(["#eef4f9"])
    ax.imshow(np.where((thk > 0.05) & (thk < 1.0), 1.0, np.nan)[::-1], extent=ext, cmap=tint, alpha=0.45,
              interpolation="bilinear")
    ax.imshow(np.where(thk >= 1.0, 1.0, np.nan)[::-1], extent=ext, cmap=tint, alpha=0.82, interpolation="bilinear")
    # particles: trails of the last TRAIL frames of the same particle, then the heads
    cmap = _speed_cmap(plt)
    norm = LogNorm(*SPEED)
    col = lambda v: cmap(norm(np.clip(v, *SPEED)))                 # black (slow) to orange (fast)
    P = _W["P"]
    k0 = max(0, k - TRAIL)
    px, py = np.asarray(P["x"][k0:k + 1]), np.asarray(P["y"][k0:k + 1])
    pb, pa = np.asarray(P["b"][k0:k + 1]), np.asarray(P["a"][k], float)
    ps = np.asarray(P["s"][k0:k + 1], float)
    live = pa > 0
    m = len(px) - 1
    if m > 0 and live.any():
        same = (pb[:-1] == pb[-1]) & (pb[1:] == pb[-1]) & live
        r, p = np.nonzero(same)
        w = (r + 1 + TRAIL - m) / TRAIL                           # 1 at the newest segment
        seg = np.stack([np.stack([px[r, p], py[r, p]], -1), np.stack([px[r + 1, p], py[r + 1, p]], -1)], 1)
        rgba = col(ps[r + 1, p])
        rgba[:, 3] = pa[p] * w * 0.8
        ax.add_collection(LineCollection(seg, colors=rgba, linewidths=0.3 + 1.3 * w, capstyle="round", zorder=3))
    if live.any():
        rgba = col(ps[-1][live])
        rgba[:, 3] = pa[live]
        ax.scatter(px[-1][live], py[-1][live], s=6, c=rgba, linewidths=0, zorder=4)
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
    fig.text(0.66, 0.80, f"{int(np.floor(t))}", fontsize=64, fontweight=600, va="top")
    vol = thk.sum() * 100 / 1e9
    area = (thk >= 1.0).sum() * 100 / 1e6
    fig.text(0.66, 0.685, f"Ice volume {vol * 1000:.2f} million m³", fontsize=20, va="top")
    fig.text(0.66, 0.645, f"Ice area {area:.3f} km²", fontsize=20, va="top")
    note = ("Hindcast from a steady 1850 state" if t < 2021.75 or kind == "hind2021" else
            "2021: start from the measured ice thickness" if kind == "switch" or (kind == "state" and a == b == I2021)
            else "Observed weather to 2026" if t <= 2026.75 else
            f"Ice area below 0.01 km² since {F['dis']}" if int(np.floor(t)) >= F["dis"] else
            "Scenario SSP2-4.5, 2026 warming anchored to observation")
    fig.text(0.66, 0.595, note, fontsize=17, va="top", color="#c51415" if "measured" in note else "#4d4d4d")
    fig.text(0.66, 0.545, f"One year passes in {c['per_year'] / FPS:.1f} seconds", fontsize=17, va="top",
             color="#4d4d4d")
    f1, f2 = c["factor"]
    mix = s if kind == "switch" else 0.0 if t < 2021.75 or kind == "hind2021" else 1.0
    for f, al in ((f1, 1 - 2 * mix), (f2, 2 * mix - 1)):         # the old factor fades out, then the new one in
        if al > 0:
            fig.text(0.66, 0.508, f"Particles move {f} times faster than the ice", fontsize=17, va="top",
                     color="#1a1a1a", alpha=al)
    cax = fig.add_axes([0.66, 0.405, 0.20, 0.016])
    sm = plt.cm.ScalarMappable(norm=LogNorm(SPEED[0], SPEED[1]), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0.1, 1, 10])
    cb.set_ticklabels(["0.1", "1", "10"])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=14)
    cb.set_label("Surface speed (m per year)", fontsize=16)
    kx = fig.add_axes([0.88, 0.405, 0.016, 0.016])
    kx.set_xticks([]), kx.set_yticks([])
    kx.set_facecolor("#eef4f9")
    fig.text(0.902, 0.413, "Ice", fontsize=16, va="center")
    gx = fig.add_axes([0.70, 0.08, 0.27, 0.22])
    first = c["first"]
    sel = (years >= first) & (years <= F["last"])
    yy, V = years[sel] + 0.75, _W["V"][sel] * 1000
    gx.plot(yy, V, color="#b3b3b3", lw=1.2)
    done = yy <= t
    gx.plot(yy[done], V[done], color="#027abe", lw=2)
    gx.plot([t], [vol * 1000], "o", color="#027abe", ms=8)
    gx.set_xlim(first, F["last"] + 1), gx.set_ylim(0, V.max() * 1.1)
    gx.set_ylabel("Ice volume (million m³)", fontsize=15)
    gx.tick_params(labelsize=13)
    gx.spines[["top", "right"]].set_visible(False)
    if tc >= 0:
        fig.patches.append(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color="white", alpha=tc / 0.5,
                                         zorder=1000))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _card_lines(film):
    F, c = _W["F"], FILMS[film]
    f1, f2 = c["factor"]
    return [f"Ghiacciaio del Basòdino {c['first']}–{F['last']}, modelled",
            "Instructed Glacier Model 3.2.0, 10 m grid, one member of the first 9984-member ensemble:",
            f"posterior draw {F['draw']}, climate model chain {F['chain']} (CH2025), scenario SSP2-4.5,",
            "2026 warming anchored to the observed trend at Locarno-Monti; the member with the median",
            f"disappearance year ({F['dis']}, ice area below 0.01 km²) of its 1664 scenario members.",
            f"The film ends three years later; no ice of 1 m or more is left in {int(_W['stop'])}.",
            "1850–2021: hindcast from a steady 1850 state; its ice area exceeds every inventory from 1929 to 2018.",
            "2021: restart from the measured ice thickness; the modelled 2026 ice area (0.45 km²) is less than",
            "half of the mapped 1.02 km². This first ensemble starts with too little ice; the fix round is running.",
            "Particles: carried by the modelled surface velocity of 1 October, linear between the yearly states,",
            f"{f1} times faster than the ice before 2021 and {f2} times faster from 2021; colour: surface speed.",
            "Data: swisstopo, GLAMOS, MeteoSwiss, Swiss Glacier Inventory, Copernicus Sentinel-2 (2026 outline)."]


def frames(film, n=16):
    from multiprocessing import Pool
    jobs = list(enumerate(timeline(film, member()["last"])))
    todo = [j for j in jobs if not (work(film) / f"f_{j[0]:05d}.png").exists()]
    with Pool(n, initializer=_init, initargs=(film,)) as p:
        for i, _ in enumerate(p.imap_unordered(_frame, todo, chunksize=4)):
            if i % 500 == 0:
                print(f"{i} of {len(todo)}", flush=True)
    print(f"{work(film)}: {len(jobs)} frames ({len(jobs) / FPS:.1f} s)")


def encode(film):
    import subprocess
    n = len(list(work(film).glob("f_*.png")))
    assert n == len(timeline(film, member()["last"])), n
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(FPS), "-i",
                    str(work(film) / "f_%05d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt",
                    "yuv420p", "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", "-r", str(FPS),
                    "-movflags", "+faststart", str(mp4(film))], check=True)
    print(f"{mp4(film)}: {n} frames, {n / FPS:.1f} s")


# --- check -------------------------------------------------------------------------------------------------------------

def _check_tracks(film, Z):
    """Measured speed-up per era, share of moves downhill, visible particles on cells without ice of 1 m."""
    tl = timeline(film, member()["last"])
    P = {q: np.load(work(film) / f"{q}.npy", mmap_mode="r") for q in "xyasb"}
    x0, y0, ny, nx = Z["x"][0], Z["y"][0], *Z["topg"].shape
    ratio = {0: [], 1: []}
    down, moves, vis, bare, bare_fading = 0, 0, 0, 0, 0
    prev = field(Z, tl[0])
    for k in range(1, len(tl)):
        cur = field(Z, tl[k])
        dt = cur[3] - prev[3]
        a0, a1 = np.asarray(P["a"][k - 1], float), np.asarray(P["a"][k], float)
        b0, b1 = np.asarray(P["b"][k - 1]), np.asarray(P["b"][k])
        X0, Y0 = np.asarray(P["x"][k - 1], float), np.asarray(P["y"][k - 1], float)
        X1, Y1 = np.asarray(P["x"][k], float), np.asarray(P["y"][k], float)
        i, j = np.rint((Y1 - y0) / 10).astype(int), np.rint((X1 - x0) / 10).astype(int)
        inside = (i >= 0) & (i < ny) & (j >= 0) & (j < nx)
        thk = np.where(inside, cur[0][np.clip(i, 0, ny - 1), np.clip(j, 0, nx - 1)], 0)
        v = a1 > 0
        vis += v.sum()
        nb = v & (thk < 1)
        bare += nb.sum()
        bare_fading += (nb & (a1 < a0)).sum()
        same = (a0 > 0) & v & (b0 == b1) & (b0 >= 0)
        if dt > 0 and same.any():
            fi, fj = (Y0[same] - y0) / 10, (X0[same] - x0) / 10
            u = 0.5 * (bilinear(prev[1], fi, fj) + bilinear(cur[1], fi, fj))
            w = 0.5 * (bilinear(prev[2], fi, fj) + bilinear(cur[2], fi, fj))
            sp = np.hypot(u, w)
            dx, dy = X1[same] - X0[same], Y1[same] - Y0[same]
            ok = sp > 0.05
            ratio[int(prev[3] >= 2021.75)].append(np.hypot(dx, dy)[ok] / (dt * sp[ok]))
            S = Z["topg"] + prev[0]
            gy, gx = np.gradient(S, 10.0)
            ii, jj = np.clip(np.rint(fi).astype(int), 0, ny - 1), np.clip(np.rint(fj).astype(int), 0, nx - 1)
            mv = np.hypot(dx, dy) > 1e-3
            down += ((dx * gx[ii, jj] + dy * gy[ii, jj] < 0) & mv).sum()
            moves += mv.sum()
        prev = cur
    f1, f2 = FILMS[film]["factor"]
    q = lambda r: np.percentile(np.concatenate(r), [5, 50, 95])
    return [(film, "measured speed-up before 2021 (5 / 50 / 95 %), stated", f"{' / '.join(f'{v:.2f}' for v in q(ratio[0]))}, {f1}"),
            (film, "measured speed-up from 2021 (5 / 50 / 95 %), stated", f"{' / '.join(f'{v:.2f}' for v in q(ratio[1]))}, {f2}"),
            (film, "moves downhill on the modelled surface (share of particle moves)", f"{down / moves:.4f} of {moves}"),
            (film, "visible particle-frames on cells without ice of 1 m (share), of them fading out",
             f"{bare / vis:.5f} ({bare} of {vis}), {bare_fading}")]


def _check_video(film):
    import json
    import subprocess
    p = mp4(film)
    pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                    "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,pix_fmt,"
                                    "color_space:format=duration", "-of", "json", str(p)],
                                   capture_output=True, text=True, check=True).stdout)
    s = pr["streams"][0]
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vf",
                        "scale=480:270,tblend=all_mode=difference,signalstats,"
                        "metadata=print:key=lavfi.signalstats.YAVG:file=-", "-f", "null", "-"],
                       capture_output=True, text=True, check=True)
    v = np.array([float(l.split("=")[1]) for l in r.stdout.splitlines() if "YAVG" in l])
    tl = timeline(film, member()["last"])
    c = FILMS[film]
    hold = np.zeros(len(tl), bool)                       # frames that may repeat the one before (stills, holds)
    for k, (kind, a, b, sf, _) in enumerate(tl):
        hold[k] = (kind == "hind2021" or (kind == "state" and a == b)
                   or (kind == "card" and sf * c["card"] / FPS >= 0.5 + 0.8 + 1 / FPS))
    still = np.flatnonzero((v < 0.005) & ~hold[1:len(v) + 1]) + 1
    move = ~hold[1:len(v) + 1] & np.array([tl[k][0] != "card" for k in range(1, len(v) + 1)])
    top = np.argsort(np.where(move, v, 0))[::-1][:5]
    return [(film, "file", f"{p.name}: {s['width']} x {s['height']}, {s['pix_fmt']}, {s['color_space']}, "
                           f"r_frame_rate {s['r_frame_rate']}, avg_frame_rate {s['avg_frame_rate']}, "
                           f"{s['nb_read_frames']} frames decoded (time line {len(tl)}), "
                           f"{float(pr['format']['duration']):.2f} s"),
            (film, "frame steps, moving map frames: median, largest (grey levels at 480 x 270)",
             f"{np.median(v[move[:len(v)]]):.3f}; " + "; ".join(f"{v[i]:.2f} into frame {i + 1}" for i in top)),
            (film, "unchanged frames (step < 0.005) outside the stills and holds", f"{len(still)} {still[:10].tolist()}")]


def check():
    Z = load_states()
    rows = []
    for film in FILMS:
        rows += _check_tracks(film, Z) + _check_video(film)
    with open(RES / "step7_particles_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["film", "quantity", "value"])
        w.writerows(rows)
    for r in rows:
        print(*r, sep=" | ")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check()
    elif cmd == "track":
        work(sys.argv[2]).mkdir(parents=True, exist_ok=True)
        track(sys.argv[2])
    elif cmd == "frames":
        frames(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16)
    else:
        encode(sys.argv[2])
