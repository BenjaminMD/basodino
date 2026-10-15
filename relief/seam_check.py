"""Checks that the seamless DEM has no step and no seam at the swissALTI3D data border.   usage: python seam_check.py <project dir>
Border segments in the box (swissALTI3D on one side only):
  vertical  E 2678000, N 1137000-1145000   Piemonte to the west
  north     N 1145000, E 2676000-2678000   Piemonte to the south
  south     N 1137000, E 2678000-2679000   the box's south edge: ring cell E 2678-2679 km N 1136-1137 km has no swissALTI3D
                                           tile (Italy), so Piemonte fills the pad there and the ramp lies inside the box
1. Steps along the whole border. For every pixel pair across the border (a on the Piemonte side, b on the swissALTI3D side):
   raw step = zS(b) - zP(b), the jump a hard cut would make, and
   J = (z(b) - z(a)) - ((z(a) - z(a-1)) + (z(b+1) - z(b))) / 2, the height change across the pair minus the mean change of
   the two neighbouring pairs: a step of height h gives J = h, smooth terrain gives J = curvature x pixel^2.
   J of the hard cut (swissALTI3D where it exists, else Piemonte) and of the seamless model at the border, and for scale
   J of the seamless model 50 m inside the Piemonte part, at the inner end of the ramp (25 m) and 50 m inside swissALTI3D.
2. Hillshade seam index: mean absolute difference of the 315/45 hillshade between neighbouring pixel lines parallel to
   the border, against distance from the border. A seam is a spike at distance 0.
3. Five profiles across the border, one per 2 km of border (four on the vertical segment, one on the north segment), each
   where the two models differ most (5 m running mean of |raw step|), with 150 m x 150 m hillshade crops at native 0.5 m.
Outputs in relief_v2/checks/: seam_steps.csv, seam_profiles.png, seam_hillshade.png, seam_index.png and plain
native-resolution hillshades of the whole border corridor, seam_strip_{vertical,north,south}.png (+-50 m)."""
import csv, json, os, sys
import numpy as np
from osgeo import gdal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from seamless_dem import swiss, piemonte, RES, E0, N1
from relief_gpu import gradient, shade

gdal.UseExceptions()
PAD = 200
W = 300                                     # half width of a corridor, pixels (150 m); border between pixels W-1 and W
SEGMENTS = {"vertical": (2678000 - 150, 1145000, 16000, 2 * W), "north": (2676000, 1145000 + 150, 2 * W, 4000),
            "south": (2678000, 1137000 + 150, 2 * W, 2000)}   # west, north, rows, columns of the corridor
SECTIONS = [("vertical", 1143000, 1145000), ("vertical", 1141000, 1143000), ("vertical", 1139000, 1141000),
            ("vertical", 1137000, 1139000), ("north", 2676000, 2678000)]


def orient(a, seg):
    """Rows along the border, columns across it, Piemonte side first (border between columns W-1 and W)."""
    return a if seg == "vertical" else a[::-1].T


def along(seg, n):
    w, north, rows, cols = SEGMENTS[seg]
    return north - (np.arange(n) + 0.5) * RES if seg == "vertical" else w + (np.arange(n) + 0.5) * RES


def jump(z, c):
    return (z[:, c + 1] - z[:, c]) - ((z[:, c] - z[:, c - 1]) + (z[:, c + 2] - z[:, c + 1])) / 2


def hillshade(z):
    ge, gn = gradient(np.pad(z, 1, mode="edge"), RES)
    return shade(ge, gn, 315).get()


def summary(x):
    a = np.abs(x)
    return f"median |.| {np.median(a):.3f} m, p95 {np.percentile(a, 95):.3f} m, p99 {np.percentile(a, 99):.3f} m, max {a.max():.3f} m"


def main(d):
    out = os.path.join(d, "relief_v2/checks")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(d, "derived/italy/italy_datum_offset.json")) as f:
        offset = json.load(f)["offset_m"]
    pad = gdal.Open(os.path.join(d, f"relief_v2/seamless_dem_0p5m_pad{PAD}m.tif"))
    box = gdal.Open(os.path.join(d, "relief_v2/seamless_dem_0p5m.tif")).ReadAsArray()
    p = int(PAD / RES)
    same = np.array_equal(box, pad.ReadAsArray(p, p, box.shape[1], box.shape[0]))
    print(f"box DEM equals the centre of the padded DEM: {same}; nodata in the box: {int(np.sum(box == -9999))}")
    del box
    data, rows_csv = {}, []
    for seg, (w, north, ny, nx) in SEGMENTS.items():
        gt = (w, RES, 0.0, north, 0.0, -RES)
        z = pad.ReadAsArray(int((w - (E0 - PAD)) / RES), int((N1 + PAD - north) / RES), nx, ny).astype(np.float64)
        zs, zp = swiss(d, gt, (ny, nx)).astype(np.float64), piemonte(d, gt, (ny, nx), offset)
        hard = np.where(np.isnan(zs), zp, zs)
        data[seg] = dict(z=z, zs=zs, zp=zp, hard=hard, hs=hillshade(z), hs_hard=hillshade(hard))
        Z, S, P, H = (orient(v, seg) for v in (z, zs, zp, hard))
        assert np.isnan(S[:, :W]).all() and not np.isnan(S[:, W:]).any() and not np.isnan(P).any(), seg
        raw = S[:, W] - P[:, W]
        cols = {"raw step": raw, "J hard cut": jump(H, W - 1), "J seamless": jump(Z, W - 1),
                "J seamless 50 m inside Piemonte": jump(Z, W - 101), "J seamless 25 m inside swissALTI3D": jump(Z, W + 49),
                "J seamless 50 m inside swissALTI3D": jump(Z, W + 99)}
        print(f"\n{seg} segment, {len(raw)} pixel pairs ({len(raw) * RES:.0f} m)")
        m = np.median(raw)
        print(f"  raw step zS - zP:  median {m:+.3f} m, NMAD {1.4826 * np.median(np.abs(raw - m)):.3f} m, {summary(raw)}")
        for k, v in cols.items():
            if k != "raw step":
                print(f"  {k:36s} {summary(v)}")
        for i, a in enumerate(along(seg, len(raw))):
            rows_csv.append([seg, f"{a:.2f}"] + [f"{cols[k][i]:.4f}" for k in cols])
        data[seg]["raw"] = raw
    with open(os.path.join(out, "seam_steps.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["segment", "along (N for vertical, E otherwise, m)"] + [k + " (m)" for k in cols])
        wr.writerows(rows_csv)

    # seam index
    fig = plt.figure(figsize=(13, 3.6))
    gs = fig.add_gridspec(1, 3, wspace=0.25)
    dist = (np.arange(2 * W - 1) + 1 - W) * RES
    for k, seg in enumerate(SEGMENTS):
        ax = fig.add_subplot(gs[0, k])
        for key, lab, st in (("hs_hard", "hard cut", dict(color="0.6", lw=1.2)), ("hs", "seamless model", dict(color="k", lw=1))):
            h = orient(data[seg][key], seg)
            ax.plot(dist, np.mean(np.abs(np.diff(h, axis=1)), axis=0), label=lab, **st)
        ax.axvspan(0, 25, color="tab:orange", alpha=0.12, lw=0, label="blend zone")
        ax.set_xlim(-60, 60)
        ax.set_title(f"{seg.capitalize()} segment", fontsize=10)
        ax.set_xlabel("Distance from the data border (m), Piemonte side negative", fontsize=8)
        if k == 0:
            ax.set_ylabel("Mean absolute hillshade difference\nbetween neighbouring pixel lines", fontsize=8)
            ax.legend(fontsize=8, frameon=False)
    fig.savefig(os.path.join(out, "seam_index.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    for seg in SEGMENTS:
        for key in ("hs", "hs_hard"):
            i0 = np.mean(np.abs(np.diff(orient(data[seg][key], seg), axis=1)), axis=0)
            print(f"seam index {seg:8s} {'seamless' if key == 'hs' else 'hard cut':8s}: largest within 1 m of the border "
                  f"{i0[np.abs(dist) <= 1].max():.4f}; median 1.5-5 m inside Piemonte {np.median(i0[(dist >= -5) & (dist <= -1.5)]):.4f}, "
                  f"inside swissALTI3D {np.median(i0[(dist >= 1.5) & (dist <= 5)]):.4f}")

    # five profiles and hillshade crops
    fig = plt.figure(figsize=(17, 6.4))
    gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.28, height_ratios=[1.4, 1])
    fig2 = plt.figure(figsize=(17, 7.4))
    gs2 = fig2.add_gridspec(2, 7, wspace=0.08, hspace=0.16, width_ratios=[1] * 5 + [0.06, 0.05])
    x = (np.arange(-120, 120) + 0.5) * RES
    print("\nfive profiles (worst place per 2 km of border):")
    for k, (seg, lo, hi) in enumerate(SECTIONS):
        dd = data[seg]
        a = along(seg, len(dd["raw"]))
        sel = np.nonzero((a >= lo) & (a < hi))[0]
        run = np.convolve(np.abs(dd["raw"]), np.ones(11) / 11, mode="same")
        i = sel[np.argmax(run[sel])]
        Z, S, P = (orient(dd[v], seg)[i, W - 120:W + 120] for v in ("z", "zs", "zp"))
        H = orient(dd["hard"], seg)
        where = f"N {a[i]:.1f}" if seg == "vertical" else f"E {a[i]:.1f}"
        dz = np.abs(np.diff(Z))                  # pair j joins profile pixels j and j + 1; the border pair is j = 119
        print(f"  {k + 1}: {seg} segment at {where}: raw step {dd['raw'][i]:+.2f} m (5 m mean |raw| {run[i]:.2f} m), "
              f"J hard cut {jump(H[i:i + 1], W - 1)[0]:+.3f} m, J seamless {jump(orient(dd['z'], seg)[i:i + 1], W - 1)[0]:+.3f} m; "
              f"seamless profile: largest change between neighbouring pixels within 5 m of the border {dz[110:130].max():.3f} m, "
              f"elsewhere within 60 m {np.r_[dz[:110], dz[130:]].max():.3f} m")
        ax = fig.add_subplot(gs[0, k])
        ax.axvspan(0, 25, color="tab:orange", alpha=0.12, lw=0, label="blend zone")
        ax.plot(x, P, color="tab:orange", lw=1, ls="--", label="Piemonte (LN02)")
        ax.plot(x, S, color="tab:blue", lw=1, ls="--", label="swissALTI3D")
        ax.plot(x, Z, color="k", lw=0.9, label="seamless model")
        ax.set_title(f"Profile {k + 1}: {seg} segment, {where}", fontsize=9)
        if k == 0:
            ax.set_ylabel("Elevation (m above sea level)", fontsize=8)
            ax.legend(fontsize=7, frameon=False)
        ax.tick_params(labelsize=7)
        ax = fig.add_subplot(gs[1, k])
        ax.axvspan(0, 25, color="tab:orange", alpha=0.12, lw=0)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.plot(x, S - P, color="tab:blue", lw=1, ls="--", label="swissALTI3D minus Piemonte")
        ax.plot(x, Z - P, color="k", lw=0.9, label="seamless model minus Piemonte")
        ax.set_xlabel("Distance from the data border (m)", fontsize=8)
        if k == 0:
            ax.set_ylabel("Height difference (m)", fontsize=8)
            ax.legend(fontsize=7, frameon=False)
        ax.tick_params(labelsize=7)
        # hillshade crops, geographic orientation, 300 x 300 px around the border point
        for r, (key, lab) in enumerate((("hs_hard", "hard cut"), ("hs", "seamless model"))):
            h, ic = dd[key], min(max(i, 150), len(a) - 150)
            crop = h[ic - 150:ic + 150, W - 150:W + 150] if seg == "vertical" else h[W - 150:W + 150, ic - 150:ic + 150]
            ax = fig2.add_subplot(gs2[r, k])
            im = ax.imshow(crop, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if seg == "vertical":       # ticks outside the frame mark the data border
                ax.set_xticks([149.5], [""]), ax.set_yticks([])
                ax.tick_params(axis="x", top=True, length=7, width=1.2)
            else:
                ax.set_yticks([149.5], [""]), ax.set_xticks([])
                ax.tick_params(axis="y", right=True, length=7, width=1.2)
            if r == 0:
                ax.set_title(f"Profile {k + 1}, {where}", fontsize=9, pad=10)
            if k == 0:
                ax.set_ylabel(lab, fontsize=9)
    cax = fig2.add_subplot(gs2[:, 6])
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + 0.1 * pos.height, pos.width, 0.8 * pos.height])
    fig2.colorbar(im, cax=cax).set_label("Hillshade 315/45 (reflectance)", fontsize=8)
    fig2.suptitle("150 m x 150 m at native 0.5 m around the five profile places, north up; ticks outside the frame mark the "
                  "data border.\nTop: hard cut (swissALTI3D where it exists, else Piemonte). Bottom: seamless model.",
                  fontsize=10, y=0.97)
    fig.savefig(os.path.join(out, "seam_profiles.png"), dpi=300, bbox_inches="tight")
    fig2.savefig(os.path.join(out, "seam_hillshade.png"), dpi=300, bbox_inches="tight")
    plt.close("all")
    for seg in SEGMENTS:   # plain native-resolution hillshade of the seamless model, +-50 m around the border
        h = data[seg]["hs"]
        strip = h[:, W - 100:W + 100] if seg == "vertical" else h[W - 100:W + 100]
        plt.imsave(os.path.join(out, f"seam_strip_{seg}.png"), strip, cmap="gray", vmin=0, vmax=1)
    print("\nfigures and seam_steps.csv in relief_v2/checks")


if __name__ == "__main__":
    main(sys.argv[1])
