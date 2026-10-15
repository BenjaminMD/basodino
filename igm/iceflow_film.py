"""Map film of the modelled glacier moving like a real one seen from above (DESIGN.md step 7; texture advection).

Inputs   results/step7/film_states.npz (animate.py states): the film member of the first ensemble (d056_c07_2), its
         1 October thickness thk and surface velocity us, vs (m per year) for every year 1850-2100, the hindcast's own
         2021 state and the bed topg; results/step7_film.csv (the member's disappearance year, ice area below 0.01 km2);
         the swisstopo SWISSIMAGE orthophoto at 1 m (tex_D.jpg, open data) and the swissTLM3D 2024 glacier outline
         (glacier2024.png) of the terrain work folder (s02: ../verification_render_work/src, Mac:
         ../data/verification/drone/terrain/src); Source Sans 3.
Method   Cut (author's rule, DESIGN.md): the film ends 3 years after the member's disappearance year (2034 + 3 = 2037),
         then the end card. Time line as animate.py: thickness and velocity linear in time between the yearly states;
         the year 2020-2021 runs to the hindcast's own 2021 state, held 0.5 s, cross-faded in 1 s into the 2021 start
         from the measured thickness and held 0.5 s; 1 s still at the start and at the end. Two films:
           full   1850 to the end year, 24 frames per model year (0.4 s); the ice moves at a quarter of its modelled
                  speed in the hindcast and 5 times faster than modelled from the 2021 restart on (the restarted ice is
                  thin and slow), so the fastest ice creeps about 0.5 m (0.2 film pixels) per frame in both eras
           short  2000 to the end year, 36 frames per model year (0.6 s), factors 1 and 5; 30 s in all
         prep (once): the map is rendered at 2x (1.333 m per pixel; 2.667 m in the film) on a crop that holds all
         modelled ice 1850-2037. Ground: the orthophoto; inside the 2024 outline (where today's ice is) rock instead:
         the median colour of the rock within 20-150 m outside the outline (no snow, no deep shadow), lit by the
         hillshade of the bed (normalised to mean 1 inside the outline), plus the colour detail (image minus its 15 m
         Gaussian mean, soft-clipped at 0.12) of that rock, quilted (random 64 m blocks, 21 m cosine overlaps,
         variance kept). Ice image: inside the 2024 outline, where the orthophoto is bright (luminance above 0.45:
         ice and firn, not rock or shadow), the orthophoto itself (its real crevasses, dirt and firn); elsewhere the
         median colour of today's bare ice (inside the outline, bluer than red) plus its colour detail, quilted the
         same way.
         frames: texture advection with two phases. Each phase carries a texture coordinate field on the 10 m grid,
         moved each frame by semi-Lagrangian back-tracing (midpoint) with the surface velocity of the frame times the
         factor times the model time of the frame, and reset to the identity every PERIOD (24) moving frames, the two
         phases half a period apart, weights sin^2 (which sum to 1), each times exp(-(strain / 0.15)^2) (strain: how
         far the phase's map is from a pure shift); the weight left goes to the still ice image, so nothing is combed
         out. No motion and no phase clock while the film holds. Ice colour: the image as it is, times (0.88 + 0.12 q)
         with q = square root of thickness over 30 m (thin ice a little darker), a cool shift of at most 4 % on thick
         ice, and 0.35 of the relative hillshade of the modelled surface (bed + thickness, fixed sun from azimuth 160
         degrees, 45 degrees high, divided by its value on flat ground). Ice cover: smoothstep of the thickness
         (bilinear on the 10 m grid, Gaussian 8 m, times exp(0.3 x smooth noise)) from 0.1 to 0.5 m, so the margin
         retreats and rock windows open where the modelled ice thins out. Downsampled 2 x 2 to the film.
         Side panel as animate.py (name, year, volume, area, status line, speed-up in words, volume curve).
         Encode: H.264 CRF 16, BT.709, constant 60 frames per second, 1920 x 1080.
         check: video (ffprobe rate and frames, frame-to-frame steps, unchanged frames outside the stills and holds),
         share of the ice whose velocity points downhill on the modelled surface, the shown texture speed measured
         by phase correlation of frames 6 apart against the velocity times the factor, ice drawn on cells without ice.
Outputs  work/iceflow/prep.npz, work/iceflow_FILM/f_NNNNN.png (intermediate); results/step7/igm_iceflow_1850_2037.mp4
         (full) and igm_iceflow_2000_2037.mp4 (short), copied to tex/animation/ with stills; results/step7_iceflow_check.csv.
usage    s02 (agl env, from igm/, after animate.py states):
           python iceflow_film.py prep                      -> work/iceflow/prep.npz
           nrun python iceflow_film.py frames FILM [N]      -> work/iceflow_FILM/f_NNNNN.png (N processes, 16)
           nrun python iceflow_film.py encode FILM          -> results/step7/igm_iceflow_FIRST_END.mp4
           python iceflow_film.py check                     -> results/step7_iceflow_check.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

IGM = Path(__file__).resolve().parent
RES = IGM / "results"
STATES = RES / "step7/film_states.npz"
PREP = IGM / "work/iceflow/prep.npz"
FPS = 60
FILMS = {  # first year, frames per year, start still, end still, card (frames), factors before | from 2021
    "full": dict(first=1850, per_year=24, start=60, end=60, card=300, factor=(0.25, 5)),
    "short": dict(first=2000, per_year=36, start=60, end=60, card=228, factor=(1, 5)),
}
W2, H2, PIX = 2520, 1920, 4 / 3          # map at 2x: pixels and metres per pixel
E0, N1 = 2678700.0, 1142900.0           # map corner (west, north); east 2682060, south 1140340
X0, Y0 = 2678705.0, 1140005.0           # centre of the first model cell
PERIOD = 24                             # moving frames per texture phase
STRAIN = 0.15                           # strain at which a phase gives way to the still texture (weight 1/e)
COOL = 0.04                             # cool shift of thick ice (red down, blue up, at most this fraction)
SUN = (160.0, 45.0)                     # azimuth, altitude (degrees)
I2021 = 171


def read_csv(path):
    return list(csv.DictReader(open(path)))


def member():
    F = {r["quantity"]: r["value"] for r in read_csv(RES / "step7_film.csv")}
    F["dis"] = int(F["disappearance 0.01 | 0.1 km2, half-volume 2026 | 2021"].split("|")[0])
    F["last"] = F["dis"] + 3
    return F


def src_dir():
    for p in (IGM.parent / "verification_render_work/src", IGM.parent / "data/verification/drone/terrain/src"):
        if p.exists():
            return p
    raise FileNotFoundError("tex_D.jpg")


def work(film):
    return IGM / f"work/iceflow_{film}"


def mp4(film):
    return RES / f"step7/igm_iceflow_{FILMS[film]['first']}_{member()['last']}.mp4"


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
        end = {q: Z[f"{q}_hind_2021"] if (a, b) == (I2021 - 1, I2021) else Z[q][b] for q in ("thk", "us", "vs")}
        return [(1 - s) * Z[q][a] + s * end[q] for q in ("thk", "us", "vs")] + [1850.75 + a + s * (b - a)]
    if kind == "hind2021":
        return [Z["thk_hind_2021"], Z["us_hind_2021"], Z["vs_hind_2021"], 2021.75]
    return [(1 - s) * Z[f"{q}_hind_2021"] + s * Z[q][I2021] for q in ("thk", "us", "vs")] + [2021.75]


def clock(film):
    """Per frame: model time, its step from the frame before, speed-up factor of that step, moving-frame count."""
    tl = timeline(film, member()["last"])
    t = np.array([_time(j) for j in tl])
    dt = np.diff(t, prepend=t[0])
    f1, f2 = FILMS[film]["factor"]
    fac = np.where(np.roll(t, 1) < 2021.75, f1, f2)
    return tl, t, dt, fac, np.cumsum(dt > 0)


def speed_words(f):
    return {0.25: "at a quarter of its modelled speed", 1: "at its modelled speed"}.get(f, f"{f} times faster than modelled")


def _time(job):
    kind, a, b, s, _ = job
    return 1850.75 + a + s * (b - a) if kind in ("state", "card") else 2021.75


def pixel_grid():
    """Fractional model-grid row and column of every 2x map pixel centre."""
    e = E0 + (np.arange(W2) + 0.5) * PIX
    n = N1 - (np.arange(H2) + 0.5) * PIX
    return np.meshgrid((n - Y0) / 10, (e - X0) / 10, indexing="ij")


def hillshade(z):
    """Lambert shade of a 10 m grid (rows northward) for the fixed sun, divided by its flat-ground value."""
    gy, gx = np.gradient(z, 10.0)
    az, alt = np.radians(SUN)
    s = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])
    return (-gx * s[0] - gy * s[1] + s[2]) / np.sqrt(1 + gx ** 2 + gy ** 2) / s[2]


# --- prep --------------------------------------------------------------------------------------------------------------

def _ortho(name, order=1):
    """A 1 m image of the terrain folder (first pixel centre 2678000.5, 1144499.5) on the 2x map pixels."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    a = np.asarray(Image.open(src_dir() / name), np.float32)
    a = a / 255.0
    e = E0 + (np.arange(W2) + 0.5) * PIX
    n = N1 - (np.arange(H2) + 0.5) * PIX
    R, C = np.meshgrid(1144499.5 - n, e - 2678000.5, indexing="ij")
    if a.ndim == 2:
        return ndi.map_coordinates(a, [R, C], order=order)
    a = ndi.gaussian_filter(a, (0.35, 0.35, 0))
    return np.stack([ndi.map_coordinates(a[..., c], [R, C], order=order) for c in range(3)], -1)


def _quilt(D, ok, rng, block=48, overlap=16):
    """Detail field over the map from random blocks of D whose pixels are all ok (cosine overlaps, variance kept)."""
    src = ndi.uniform_filter(ok.astype(float), block) > 0.97
    cand = np.argwhere(src[block // 2:-block // 2:4, block // 2:-block // 2:4]) * 4
    ramp = np.ones(block)
    ramp[:overlap] = np.sin(np.linspace(0, np.pi / 2, overlap + 2)[1:-1]) ** 2
    ramp[-overlap:] = ramp[:overlap][::-1]
    w = np.outer(ramp, ramp)
    acc, wsq = np.zeros((H2 + block, W2 + block) + D.shape[2:]), np.zeros((H2 + block, W2 + block))
    for r in range(0, H2, block - overlap):
        for c in range(0, W2, block - overlap):
            i, j = cand[rng.integers(len(cand))]
            acc[r:r + block, c:c + block] += w.reshape(w.shape + (1,) * (D.ndim - 2)) * D[i:i + block, j:j + block]
            wsq[r:r + block, c:c + block] += w ** 2
    wsq = wsq.reshape(wsq.shape + (1,) * (D.ndim - 2))
    return (acc / np.sqrt(np.maximum(wsq, 1e-6)))[:H2, :W2]


def prep():
    Z = np.load(STATES)
    rgb = _ortho("tex_D.jpg")
    mask = _ortho("glacier2024.png", order=0) > 0.5
    L = rgb.mean(-1)
    rng = np.random.default_rng(20261009)
    bare = mask & (rgb[..., 2] - rgb[..., 0] > 0.01) & (L > 0.35)
    bare = ndi.binary_opening(bare, iterations=2)
    dist = ndi.distance_transform_edt(~mask) * PIX
    ring = (dist > 20) & (dist < 150)
    lit = ring & (L > 0.3) & (L < 0.8)                             # rock, no snow, no deep shadow
    Drgb = rgb - ndi.gaussian_filter(rgb, (15 / PIX, 15 / PIX, 0))
    ice_col = np.median(rgb[bare], 0)
    real = ndi.binary_opening(mask & (L > 0.45), iterations=2)     # today's ice and firn, not rock or shadow
    fo = ndi.gaussian_filter(ndi.binary_erosion(real, iterations=4).astype(float), 3)[..., None]
    ice = np.clip(fo * rgb + (1 - fo) * (ice_col + _quilt(Drgb, bare, rng)), 0, 1)
    D_rock = _quilt(np.tanh(Drgb / 0.12) * 0.12, lit, rng)        # colour detail, soft-clipped
    rock_col = np.median(rgb[lit], 0)
    FI, FJ = pixel_grid()
    shade_bed = ndi.map_coordinates(hillshade(Z["topg"].astype(float)), [FI, FJ], order=3)
    sb = np.clip(1 + 0.6 * (shade_bed - 1), 0.5, 1.4)
    inside = rock_col * (sb / sb[mask].mean())[..., None] + D_rock      # shade kept at the rock's mean level
    mf = ndi.gaussian_filter(ndi.binary_dilation(mask, iterations=6).astype(float), 3)[..., None]
    rock = np.clip(mf * inside + (1 - mf) * rgb, 0, 1)
    noise = ndi.gaussian_filter(D_rock.mean(-1), 6)
    noise /= noise.std()
    PREP.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PREP, rock=(rock * 255).round().astype(np.uint8), ice=(ice * 255).round().astype(np.uint8),
             noise=noise.astype(np.float16))
    print(f"{PREP}: {W2} x {H2}; bare ice {bare.sum() * PIX ** 2 / 1e6:.3f} km2 (of the outline "
          f"{mask.sum() * PIX ** 2 / 1e6:.3f}), rock ring {ring.sum() * PIX ** 2 / 1e6:.3f} km2, rock colour "
          f"{np.round(rock_col * 255).astype(int).tolist()}, bare ice colour {np.round(ice_col * 255).astype(int).tolist()}")


# --- frames ------------------------------------------------------------------------------------------------------------

_W = {}


def _font_dir():
    for p in (IGM.parent / "data/satellite_archive/sentinel1/scripts/fonts",
              IGM.parent / "derived/satellite_archive/sentinel1/scripts/fonts"):
        if p.exists():
            return p
    raise FileNotFoundError("Source Sans 3")


def _init(film):
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
    P = np.load(PREP)
    _W.update(rock=P["rock"].astype(np.float32) / 255, ice=P["ice"].astype(np.float32) / 255,
              noise=P["noise"].astype(np.float32))
    _W["FI"], _W["FJ"] = (a.astype(np.float32) for a in pixel_grid())
    _W["V"] = _W["thk"].sum((1, 2)) * 100 / 1e9
    _W["tl"], _W["t"], _W["dt"], _W["fac"], _W["m"] = clock(film)
    _W.update(plt=plt, film=film, F=member())


def _coords(k):
    """The two phases' texture coordinates (model-grid row, column) at frame k, and their weights."""
    tl, dt, fac, m = _W["tl"], _W["dt"], _W["fac"], _W["m"]
    ny, nx = _W["topg"].shape
    I, J = np.mgrid[0:ny, 0:nx].astype(float)
    out = []
    for p in (0, 1):
        age = int((m[k] - p * PERIOD // 2) % PERIOD)
        steps = [f for f in range(max(0, k - 4 * PERIOD - 400), k + 1) if dt[f] > 0 and m[k] - age < m[f] <= m[k]]
        ci, cj = I.copy(), J.copy()
        for f in steps:
            h = fac[f] * dt[f] / 10                        # cells per (m per year)
            _, u0, v0, _ = field(_W, tl[f - 1])
            _, u1, v1, _ = field(_W, tl[f])
            u, v = (u0 + u1) / 2, (v0 + v1) / 2
            mi = I - 0.5 * h * v
            mj = J - 0.5 * h * u
            um = ndi.map_coordinates(u, [mi, mj], order=1, mode="nearest")
            vm = ndi.map_coordinates(v, [mi, mj], order=1, mode="nearest")
            bi, bj = I - h * vm, J - h * um
            ci, cj = (ndi.map_coordinates(ci, [bi, bj], order=1, mode="nearest"),
                      ndi.map_coordinates(cj, [bi, bj], order=1, mode="nearest"))
        gi, gj = np.gradient(ci), np.gradient(cj)                # strain: deviation of the map from a shift
        strain = np.sqrt((gi[0] - 1) ** 2 + gi[1] ** 2 + gj[0] ** 2 + (gj[1] - 1) ** 2)
        out.append((ci, cj, np.sin(np.pi * age / PERIOD) ** 2 * np.exp(-(strain / STRAIN) ** 2)))
    return out


def map_image(k):
    """The map of frame k at 2x (H2, W2, 3), floats 0-1, and the frame's thickness and model time."""
    thk, us, vs, t = field(_W, _W["tl"][k])
    FI, FJ = _W["FI"], _W["FJ"]
    th = ndi.gaussian_filter(ndi.map_coordinates(thk, [FI, FJ], order=1), 6)   # no single-cell holes
    shade = ndi.map_coordinates(hillshade((_W["topg"] + thk).astype(float)), [FI, FJ], order=3)
    T, ws = 0.0, 1.0
    for ci, cj, w in _coords(k):
        w = ndi.map_coordinates(w, [FI, FJ], order=1)[..., None]
        ti = ndi.map_coordinates(ci, [FI, FJ], order=1)
        tj = ndi.map_coordinates(cj, [FI, FJ], order=1)
        r = (N1 - (Y0 + 10 * ti)) / PIX - 0.5            # texture pixel of the coordinate
        c = (X0 + 10 * tj - E0) / PIX - 0.5
        T = T + w * np.stack([ndi.map_coordinates(_W["ice"][..., b], [r, c], order=1, mode="reflect")
                              for b in range(3)], -1)
        ws = ws - w
    T = T + ws * _W["ice"]                                 # the rest of the weight: the still texture
    q = np.clip(np.sqrt(np.maximum(th, 0) / 30), 0, 1)[..., None]
    tone = (0.88 + 0.12 * q) * (1 + COOL * (q - 0.5) * np.array([-1.0, 0.0, 1.0]))
    ice = T * tone * np.clip(1 + 0.35 * (shade - 1), 0.6, 1.4)[..., None]
    x = np.clip((th * np.exp(0.3 * _W["noise"]) - 0.1) / 0.4, 0, 1)
    cover = (x * x * (3 - 2 * x))[..., None]
    return np.clip(_W["rock"] * (1 - cover) + ice * cover, 0, 1), thk, t


def _frame(k):
    plt, film, F, tl = _W["plt"], _W["film"], _W["F"], _W["tl"]
    c = FILMS[film]
    from matplotlib import patheffects
    kind, a, b, s, _ = tl[k]
    path = work(film) / f"f_{k:05d}.png"
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    tc = s * c["card"] / FPS if kind == "card" else -1.0
    if tc >= 0.5:                                                  # the last map has faded to white: the text
        a_in = min(1.0, (tc - 0.5) / 0.8)
        lines = _card_lines(film)
        fig.text(0.08, 0.88, lines[0], fontsize=34, fontweight=600, alpha=a_in, va="top")
        for i, l in enumerate(lines[1:]):
            fig.text(0.08, 0.79 - i * 0.048, l, fontsize=19, alpha=a_in, va="top", color="#1a1a1a")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return
    img, thk, t = map_image(k)
    img = img.reshape(H2 // 2, 2, W2 // 2, 2, 3).mean((1, 3))
    fig.figimage((img * 255).round().astype(np.uint8), xo=60, yo=60, origin="upper")
    ax = fig.add_axes([60 / 1920, 60 / 1080, W2 / 2 / 1920, H2 / 2 / 1080])
    ax.set_xlim(0, W2 * PIX), ax.set_ylim(0, H2 * PIX)
    ax.set_axis_off()
    halo = [patheffects.withStroke(linewidth=3, foreground="#1a1a1a", alpha=0.45)]
    e1, n0 = W2 * PIX, 0
    ax.plot([e1 - 640, e1 - 140], [n0 + 110, n0 + 110], color="white", lw=3, solid_capstyle="butt", path_effects=halo)
    ax.text(e1 - 390, n0 + 150, "500 m", ha="center", va="bottom", fontsize=16, color="white", path_effects=halo)
    ax.annotate("", xy=(150, H2 * PIX - 150), xytext=(150, H2 * PIX - 380),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2))
    ax.text(150, H2 * PIX - 130, "North", ha="center", va="bottom", fontsize=15, color="white", path_effects=halo)
    # side panel
    x = 1370 / 1920
    fig.text(x, 0.93, "Ghiacciaio del Basòdino", fontsize=28, fontweight=600, va="top")
    fig.text(x, 0.877, "Ice flow model, one ensemble member", fontsize=17, va="top", color="#4d4d4d")
    fig.text(x, 0.80, f"{int(np.floor(t))}", fontsize=64, fontweight=600, va="top")
    vol = thk.sum() * 100 / 1e9
    fig.text(x, 0.685, f"Ice volume {vol * 1000:.2f} million m³", fontsize=19, va="top")
    fig.text(x, 0.647, f"Ice area {(thk >= 1.0).sum() * 100 / 1e6:.3f} km²", fontsize=19, va="top")
    note = ("Hindcast from a steady 1850 state" if t < 2021.75 or kind == "hind2021" else
            "2021: start from the measured ice thickness" if kind == "switch" or (kind == "state" and a == b == I2021)
            else "Observed weather to 2026" if t <= 2026.75 else
            f"Ice area below 0.01 km² since {F['dis']}" if int(np.floor(t)) >= F["dis"] else
            "Scenario SSP2-4.5, 2026 warming anchored to observation")
    fig.text(x, 0.598, note, fontsize=15, va="top", color="#c51415" if "measured" in note else "#4d4d4d")
    fig.text(x, 0.548, f"One year passes in {c['per_year'] / FPS:.1f} seconds", fontsize=16, va="top", color="#4d4d4d")
    f1, f2 = c["factor"]
    mix = s if kind == "switch" else 0.0 if t < 2021.75 or kind == "hind2021" else 1.0
    for f, al in ((f1, 1 - 2 * mix), (f2, 2 * mix - 1)):          # the old factor fades out, then the new one in
        if al > 0:
            fig.text(x, 0.512, "The ice moves " + speed_words(f), fontsize=16, va="top", alpha=al)
    gx = fig.add_axes([x + 0.035, 0.09, 0.215, 0.22])
    years, first = _W["years"], c["first"]
    sel = (years >= first) & (years <= F["last"])
    yy, V = years[sel] + 0.75, _W["V"][sel] * 1000
    gx.plot(yy, V, color="#b3b3b3", lw=1.2)
    gx.plot(yy[yy <= t], V[yy <= t], color="#027abe", lw=2)
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
            "Ice surface: the swisstopo orthophoto of August 2024, carried along by the modelled surface velocity",
            f"{speed_words(f1)} before 2021 and {speed_words(f2)} from 2021; cover and shading follow the thickness.",
            "Data: swisstopo (SWISSIMAGE, swissALTI3D, swissTLM3D), GLAMOS, MeteoSwiss, Swiss Glacier Inventory."]


def frames(film, n=16):
    from multiprocessing import Pool
    work(film).mkdir(parents=True, exist_ok=True)
    tl = timeline(film, member()["last"])
    todo = [k for k in range(len(tl)) if not (work(film) / f"f_{k:05d}.png").exists()]
    with Pool(n, initializer=_init, initargs=(film,)) as p:
        for i, _ in enumerate(p.imap_unordered(_frame, todo, chunksize=2)):
            if i % 500 == 0:
                print(f"{i} of {len(todo)}", flush=True)
    print(f"{work(film)}: {len(tl)} frames ({len(tl) / FPS:.1f} s)")


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

def _shift(a, b):
    """Translation of b against a (rows, columns; pixels) by phase correlation with a parabolic peak fit."""
    A, B = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    R = np.real(np.fft.ifft2(B * np.conj(A) / np.maximum(np.abs(B * np.conj(A)), 1e-9)))
    i, j = np.unravel_index(np.argmax(R), R.shape)
    out = []
    for k, n, axis in ((i, R.shape[0], 0), (j, R.shape[1], 1)):
        m, p = (k - 1) % n, (k + 1) % n
        y0, y1, y2 = (R[m, j], R[i, j], R[p, j]) if axis == 0 else (R[i, m], R[i, j], R[i, p])
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        out.append(((k + d + n / 2) % n) - n / 2)
    return np.array(out)


def _check_film(film):
    import json
    import subprocess
    _init(film)
    tl, t, dt, fac, m = _W["tl"], _W["t"], _W["dt"], _W["fac"], _W["m"]
    rows = []
    # downhill: share of the ice (cells of 1 m or more) whose velocity points down the modelled surface, all frames' years
    down, tot = 0, 0
    for i in range(FILMS[film]["first"] - 1850, member()["last"] - 1850 + 1):
        thk, us, vs = _W["thk"][i], _W["us"][i], _W["vs"][i]
        gy, gx = np.gradient(_W["topg"] + thk, 10.0)
        on = (thk >= 1) & (np.hypot(us, vs) > 1e-3)
        down += ((us * gx + vs * gy)[on] < 0).sum()
        tot += on.sum()
    rows.append((film, "ice cells (1 m or more, moving) whose velocity points downhill, all shown years",
                 f"{down / tot:.4f} of {tot}"))
    # speed of the shown texture: frames k and k + 6 rendered without the panel, fast patch, against the prediction
    res = []
    for y in ((1860, 1900, 1950, 1990) if film == "full" else (2003, 2010)) + (2022, 2024):
        k = next(f for f in range(len(tl)) if tl[f][0] == "state" and tl[f][4] == y and tl[f][3] >= 0.2)
        _, thk, _ = map_image(k)
        sp = np.hypot(*field(_W, tl[k])[1:3])
        sp = np.where(thk >= 5, sp, 0)
        i, j = np.unravel_index(np.argmax(ndi.uniform_filter(sp, 9)), sp.shape)
        r = int(np.clip((N1 - (Y0 + 10 * i)) / PIX, 64, H2 - 64))
        c = int(np.clip((X0 + 10 * j - E0) / PIX, 64, W2 - 64))
        win = np.hanning(128)[:, None] * np.hanning(128)[None]
        # one phase only (the other's weight set to 0) so the cross-fade does not blur the measurement
        a = _texture_only(k)[r - 64:r + 64, c - 64:c + 64] * win
        b = _texture_only(k + 6, like=k)[r - 64:r + 64, c - 64:c + 64] * win
        dr, dc = _shift(a, b)
        meas = np.hypot(dr, dc) * PIX
        pred = sum(fac[f] * dt[f] * sp[i, j] for f in range(k + 1, k + 7))
        res.append(f"{y}: {meas:.2f} m against {pred:.2f} m (factor {fac[k + 1]}, {sp[i, j]:.2f} m per year)")
    rows.append((film, "texture shift over 6 frames at the fastest ice, measured against velocity x factor x time",
                 "; ".join(res)))
    # ice drawn (cover >= 0.5) on map pixels whose modelled thickness is below 1 m and below 0.1 m, every 50th frame
    n05, n1, n01 = 0, 0, 0
    for k in range(0, len(tl), 50):
        th = ndi.map_coordinates(field(_W, tl[k])[0], [_W["FI"], _W["FJ"]], order=1)
        drawn = ndi.gaussian_filter(th, 6) * np.exp(0.3 * _W["noise"]) >= 0.3
        n05 += drawn.sum()
        n1 += (drawn & (th < 1)).sum()
        n01 += (drawn & (th < 0.1)).sum()
    rows.append((film, "drawn ice pixels (cover 0.5 or more) on modelled thickness below 1 m | below 0.1 m, share",
                 f"{n1 / n05:.4f} | {n01 / n05:.6f} (every 50th frame)"))
    p = mp4(film)
    pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                    "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,"
                                    "pix_fmt,color_space:format=duration", "-of", "json", str(p)],
                                   capture_output=True, text=True, check=True).stdout)
    s = pr["streams"][0]
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vf",
                        "scale=480:270,tblend=all_mode=difference,signalstats,"
                        "metadata=print:key=lavfi.signalstats.YAVG:file=-", "-f", "null", "-"],
                       capture_output=True, text=True, check=True)
    v = np.array([float(l.split("=")[1]) for l in r.stdout.splitlines() if "YAVG" in l])
    c = FILMS[film]
    hold = np.array([kind == "hind2021" or (kind == "state" and a == b) or
                     (kind == "card" and sf * c["card"] / FPS >= 0.5 + 0.8 + 1 / FPS) for kind, a, b, sf, _ in tl])
    move = ~hold[1:len(v) + 1] & np.array([tl[k][0] != "card" for k in range(1, len(v) + 1)])
    still = np.flatnonzero((v < 0.005) & move) + 1
    top = np.argsort(np.where(move, v, 0))[::-1][:5]
    rows += [(film, "file", f"{p.name}: {s['width']} x {s['height']}, {s['pix_fmt']}, {s['color_space']}, "
                            f"r_frame_rate {s['r_frame_rate']}, avg_frame_rate {s['avg_frame_rate']}, "
                            f"{s['nb_read_frames']} frames decoded (time line {len(tl)}), "
                            f"{float(pr['format']['duration']):.2f} s"),
             (film, "frame steps, map frames outside stills and holds: median, largest (grey levels at 480 x 270)",
              f"{np.median(v[move]):.3f}; " + "; ".join(f"{v[i]:.2f} into frame {i + 1}" for i in top)),
             (film, "unchanged map frames (step < 0.005) outside the stills and holds, their model years",
              f"{len(still)}" + (f", {tl[still[0]][4]}-{tl[still[-1]][4]}, {sum(tl[i][4] < 2030 for i in still)} "
                                 "before 2030" if len(still) else ""))]
    return rows


def _texture_only(k, like=None):
    """Ice detail of frame k from the phase that is heavier at frame `like` (default k) alone, 2x pixels."""
    FI, FJ = _W["FI"], _W["FJ"]
    ph = _coords(k)
    pick = int(np.argmax([w.mean() for *_, w in _coords(like if like is not None else k)]))
    ci, cj, _ = ph[pick]
    ti, tj = ndi.map_coordinates(ci, [FI, FJ], order=1), ndi.map_coordinates(cj, [FI, FJ], order=1)
    r, c = (N1 - (Y0 + 10 * ti)) / PIX - 0.5, (X0 + 10 * tj - E0) / PIX - 0.5
    return ndi.map_coordinates(_W["ice"].mean(-1), [r, c], order=1, mode="reflect")


def check():
    rows = []
    for film in FILMS:
        rows += _check_film(film)
    with open(RES / "step7_iceflow_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["film", "quantity", "value"])
        w.writerows(rows)
    for r in rows:
        print(*r, sep=" | ")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep()
    elif cmd == "check":
        check()
    elif cmd == "frames":
        frames(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16)
    else:
        encode(sys.argv[2])
