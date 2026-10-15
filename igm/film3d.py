"""3D film of the fixed ensemble's film member, 1850 to its disappearance year + 3 (DESIGN.md step 7; author's rule of
2026-10-09): the modelled ice as a surface of its own (bed + thickness)
on the swisstopo terrain, rendered in Blender Cycles with the terrain pipeline of
data/verification/drone/terrain/ (render_views.py: scene, sun, sky, ground material, grade; flyover.py: camera
spline, motion blur), slow camera, 60 frames per second, 1920 x 1080; labels and the end card composed afterwards.

Inputs (repository paths; the work copies on s01 and s02 have the same shape)
  igm/results/f7/film_states.npz      animate.py states: yearly 1 October thickness 1850-END of the film member
                                      (hindcast to 2020, 2021 run to 2025, start state 2026, projection to its stop,
                                      0 after), the hindcast's 2021 state, the member's bed topg (10 m), dis (its
                                      disappearance year), stop (no-ice stop), end (dis + 3)
  igm/inputs/igm_static.npz           x, y (cell centres) and usurf_2021 (the 2021 surface on the 10 m grid)
  igm/results/f6_film.csv             the member (projection.py film)
  data/verification/drone/terrain/    render_views.py, flyover.py, basodino_terrain.glb, basodino_far.glb, src/

Method
  Bed: the terrain heights of render_views.grids (2021 surface over the glacier) minus D = max(usurf_2021 - topg,
    0), the member's 2021 ice thickness on the 10 m grid, bilinear at every node of L0-L3 (D = 0 off the 2021
    glacier, where the bed is the ground); normals recomputed, level seams re-snapped (render_views.snap). Where D >
    0 or SWISSIMAGE 2024 shows the glacier's ice of today (prep) the ground has no image: the film shows the
    modelled ice only, so the image colour becomes BED_RGB (a rock grey of
    luminance 0.33, the median of the rock outside the glaciers, so the rock albedo curve gives the rock median
    albedo) varied by a noise texture of +-BED_NOISE, and the glacier weight of the albedo curve goes to 0 (rock),
    mask src/igm_bed.png (2 m, edge blurred by 3 m).
  Ice: a mesh on the 2 m lattice of L2 over the cells that ever carry ICE_MIN m (grown by 30 m); per frame (set_ice)
    its heights are bed + thickness - ICE_SINK, the thickness bilinear from the 10 m grid and linear in time between
    the yearly states; under thick ice the surface becomes the model's own surface (its 10 m bed topg + thickness,
    bilinear), so the bed relief finer than the model grid is dropped and no ice is stacked on a ridge next to ice
    that the basin rule (F2) cuts off; where the thickness is below ICE_SINK the surface lies under the ground, so the
    margin meets the terrain without a seam;
    smooth shading; material ICE_RGB (luminance 0.56, between the bare ice 0.50 and firn 0.62 of render_views'
    albedo curve), roughness ICE_ROUGH, specular level ICE_SPEC (bare ice and firn on 1 October, no glints that the
    tone curve would clip), AOV ice = 1. No ice is drawn outside the model domain.
  Timeline (render frames): 1850-2021 PER_YEAR_HIND frames per year (the year 2020-2021 runs to the hindcast's own
    2021 state); at 2021 the hindcast state held, the thickness
    morphed in MORPH frames into the 2021 start from the measured thickness, held; 2021 to END PER_YEAR_NOW frames
    per year (PER_YEAR_AFTER per year after a no-ice stop before END) and END_HOLD frames on END.
  Camera: flyover.flight (cubic splines, speed by viewing distance, eased) through EYE and LOOK over all render
    frames: a slow arc north of the glacier from north-west to north, 2.9-3.2 km from the look-at point; LENS mm;
    camera motion blur over half a frame.
  Grade: render_views (albedo curves, physical sky, sun of 27 September 2026, 15:30, tone curve, dithered 8 bit), one
    fixed EXPOSURE and SKY for all frames, from `measure` (render_views.expose on MEASURE_FRAMES; median of the picks).
  Compose: name, year, ice volume and a status line top left, a timeline 1850-END at the bottom (soft dark backing,
    as flyover.py), then a still, and an end card with the model, the member, the scenario and the caveats; H.264 CRF
    16, constant 60 frames per second, BT.709.
usage (from the work copy's igm/; Blender 5.2: tools/blender-5.2.2-linux-x64/blender):
  python film3d.py prep                                     src/igm_bed.png + .wld (once per work copy)
  blender -b --factory-startup -P film3d.py -- render [--frames A-B|a,b] [--share I --weights W0,W1] [--width 1920]
          [--samples 96] [--out film3d_frames] [--measure 1] [--state F]   one GPU per process (CUDA_VISIBLE_DEVICES); existing
          frames skipped; --state F (or --states F1,F2,...): stills with the ice of frame F under the camera of
          each --frames frame (written as <frame>_s<F>.png, for pairs from one viewpoint); --eye E,N,H --look E,N,H
          [--lens mm]: one free camera instead (LV95 and metres; written as view_s<F>.png, no motion blur);
          --desnow 1: the snow of the image outside the modelled ice drawn as rock (add_desnow); existing
          frames skipped; two processes in one copy: start the second after the first has built its sky
  python film3d.py path                                     camera clearance, distance, speed, pan rate
  python film3d.py measured logs/f7_measure.log             -> results/f7_film3d_measure.csv (the grade)
  python film3d.py check [frames]                           -> results/f7_film3d_check.csv
  python film3d.py path > results/f7_film3d_path.txt
  python film3d.py compose [frames] [--check F,F]           -> results/f7/igm_3d_1850_END.mp4
The first ensemble's version (member d056_c07_2, 1850-2100, results/first_ensemble/step7_film3d_*) is in the git
history.
"""
import csv
import os
import subprocess
import sys
import time

import numpy as np

IGM = os.path.dirname(os.path.abspath(__file__))
TERRAIN = os.path.normpath(os.path.join(IGM, '../data/verification/drone/terrain'))
FONTS = os.path.normpath(os.path.join(IGM, '../data/satellite_archive/sentinel1/scripts/fonts'))
STATES, STATIC = f'{IGM}/results/f7/film_states.npz', f'{IGM}/inputs/igm_static.npz'
FILM = f'{IGM}/results/f6_film.csv'
FPS = 60
PER_YEAR_HIND, MORPH_HOLD, MORPH, PER_YEAR_NOW, PER_YEAR_AFTER, END_HOLD = 15, 20, 80, 30, 4, 60
STILL, CARD_IN, CARD = 60, 40, 660                     # compose: still on the last frame, end card fade and hold
ICE_MIN, ICE_SINK, ICE_RGB, ICE_ROUGH, ICE_SPEC = 0.5, 0.3, (0.50, 0.57, 0.62), 0.6, 0.3
SNOW_LUM = (0.52, 0.66)                                # --desnow: image luminance (linear) over which the ground is snow
BED_RGB, BED_NOISE = (0.345, 0.333, 0.318), 0.3
THK_SMOOTH, RELIEF_DEPTH = 8.0, 5.0                     # m
EXPOSURE, SKY = 0.0906, 0.403                          # medians of the picks of render_views.expose on MEASURE_FRAMES
SAMPLES, LENS, SHUTTER = 96, 35.0, 0.5
LOOK_AT = (2680250, 1141250, 2850)
EYE = [(2678150, 1143350, 3480), (2679150, 1143950, 3500), (2680150, 1144200, 3520), (2680950, 1144100, 3520)]
MEASURE_FRAMES = (0, 700, 1300, 2000, 2600, 2700, 2850, 3000)
BACKING = 0.30


# --- timeline ---------------------------------------------------------------------------------------------------------

def timeline(stop, end):
    """Per render frame (kind, a, b, s): the thickness (1 - s) state a + s state b, indices into 1850-end; kind
    'morph': hindcast 2021 to the 2021 start state at s."""
    last = end - 1850
    out = []
    for i in range(171):
        out += [('state', i, i + 1, k / PER_YEAR_HIND) for k in range(PER_YEAR_HIND)]
    out += [('morph', 171, 171, 0.0)] * MORPH_HOLD
    out += [('morph', 171, 171, (k + 0.5) / MORPH) for k in range(MORPH)]
    out += [('state', 171, 171, 0.0)] * MORPH_HOLD
    for i in range(171, last):
        n = PER_YEAR_NOW if 1850 + i < stop else PER_YEAR_AFTER
        out += [('state', i, i + 1, k / n) for k in range(n)]
    out += [('state', last, last, 0.0)] * END_HOLD
    return out


def frames_of(Z):
    return timeline(int(Z['stop']), int(Z['end']))


def thickness(Z, job):
    kind, a, b, s = job
    if kind == 'morph':
        x = s * s * (3 - 2 * s)
        return (1 - x) * Z['thk_hind_2021'] + x * Z['thk'][171]
    if (a, b) == (170, 171):                     # the last hindcast year ends on the hindcast's own 2021 state
        return (1 - s) * Z['thk'][a] + s * Z['thk_hind_2021']
    return (1 - s) * Z['thk'][a] + s * Z['thk'][b]


def year_of(job):
    kind, a, b, s = job
    return 1850 + a + s * (b - a) + 0.75


def camera(frames, n):
    """Eye and look-at points at the (fractional) frames, flyover.flight on EYE with n frames."""
    sys.path.insert(0, TERRAIN)
    import flyover as fo
    fo.EYE, fo.LOOK, fo.FRAMES = EYE, [LOOK_AT] * len(EYE), n
    return fo.flight(frames)


def path():
    import rasterio
    from scipy.ndimage import maximum_filter
    Z = np.load(STATES)
    n = len(frames_of(Z))
    eye, look = camera(np.arange(n, dtype=float), n)
    r = rasterio.open(f'{TERRAIN}/src/dem_F1.tif')
    top = maximum_filter(r.read(1), footprint=np.hypot(*np.mgrid[-20:21, -20:21]) <= 20)
    col, row = ~r.transform * (eye[:, 0], eye[:, 1])
    clear = eye[:, 2] - top[np.round(row - 0.5).astype(int), np.round(col - 0.5).astype(int)]
    v = look - eye
    dist = np.linalg.norm(v, axis=1)
    pan = np.degrees(np.unwrap(np.arctan2(v[:, 0], v[:, 1])))
    speed = np.linalg.norm(np.gradient(eye, axis=0), axis=1) * FPS
    print(f'{n} frames ({n / FPS:.1f} s); clearance {clear.min():.0f} m at least; distance {dist.min():.0f}-'
          f'{dist.max():.0f} m; speed up to {speed.max():.0f} m/s; view azimuth {pan.min() % 360:.0f}-'
          f'{pan.max() % 360:.0f} deg (sun 220 deg); pan rate up to {np.abs(np.gradient(pan)).max() * FPS:.2f} deg/s')


# --- Blender ----------------------------------------------------------------------------------------------------------

def depth_grid():
    """D = max(usurf_2021 - topg, 0) on the 10 m grid (rows north to south) and its first node."""
    Z, st = np.load(STATES), np.load(STATIC)
    D = np.maximum(st['usurf_2021'] - Z['topg'], 0.0)[::-1].astype(np.float64)
    return D, float(st['x'][0]), float(st['y'][-1])


def prep():
    """src/igm_bed.png + .wld (plain Python: numpy, scipy, PIL): 2 m mask of the ground that shows no image: the
    member's bed (D > 0) and every Basodino outline of 2016-2026 on the 10 m grid (igm_static sgi2016, tlm2024,
    s2_2026, mask_2021: where SWISSIMAGE 2024 shows today's ice or snow of the glacier), grown by one 10 m cell,
    bilinear to 2 m, edge blurred by a Gaussian of 3 m."""
    from PIL import Image
    from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates
    D, x0, y0 = depth_grid()
    st = np.load(STATIC)
    today = (st['sgi2016'] > 0) | (st['tlm2024'] > 0) | (st['s2_2026'] > 0) | (st['mask_2021'] > 0)
    m10 = binary_dilation((D > 0) | today[::-1], iterations=1).astype(np.float64)
    e = np.arange(x0 - 5 + 1, x0 - 5 + 3400, 2.0)
    n = np.arange(y0 + 5 - 1, y0 + 5 - 2900, -2.0)
    E, N = np.meshgrid(e, n)
    d = map_coordinates(m10, [(y0 - N) / 10, (E - x0) / 10], order=1, mode='nearest')
    m = np.clip(gaussian_filter((d > 0.5).astype(np.float64), 1.5), 0, 1)
    Image.fromarray(np.round(m * 255).astype(np.uint8)).save(f'{TERRAIN}/src/igm_bed.png')
    open(f'{TERRAIN}/src/igm_bed.wld', 'w').write(f'2.0\n0.0\n0.0\n-2.0\n{e[0]}\n{n[0]}\n')
    print(f'src/igm_bed.png {m.shape}, no-image area {(m > 0.5).sum() * 4 / 1e6:.3f} km2 (bed D > 0: '
          f'{(D > 0).sum() / 1e4:.4f} km2, outlines 2016-2026 {today.sum() / 1e4:.4f} km2)')


def lowered_grids(rv, grids):
    """render_views.grids with the bed: D subtracted at every node of L0-L3, normals recomputed, seams re-snapped."""
    D, x0, y0 = depth_grid()

    def g_bed():
        g, gn = grids()
        for k in ('L0', 'L1', 'L2', 'L3'):
            e, n = rv.nodes(k, g[k].shape)
            g[k] = g[k] - rv.bilinear(D, x0, y0, 10.0, *np.meshgrid(e, n))
            gn[k] = rv.normals(g[k], rv.LEVELS[k][4])
        for fine, coarse in (('L2', 'L3'), ('L1', 'L2'), ('L0', 'L1')):
            rv.snap(g[fine], fine, g[coarse], coarse)
        g_bed.g = g
        return g, gn
    return g_bed


def add_bed(mat, rv):
    """Ground material (render_views.ground_material): image -> BED_RGB times noise where the bed mask is on, glacier
    weight of the albedo curve times (1 - mask)."""
    nt = mat.node_tree
    pos = nt.nodes.new('ShaderNodeNewGeometry').outputs['Position']
    mask = rv.draped(nt, pos, 'igm_bed', 'png')
    mask.image.colorspace_settings.name = 'Non-Color'
    image = next(n for n in nt.nodes if n.type == 'RGBTOBW').inputs['Color'].links[0].from_socket
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 0.08                   # world metres: about 12 m features
    noise.inputs['Detail'].default_value = 6.0
    nt.links.new(pos, noise.inputs['Vector'])
    var = nt.nodes.new('ShaderNodeMapRange')
    var.inputs['To Min'].default_value, var.inputs['To Max'].default_value = 1 - BED_NOISE, 1 + BED_NOISE
    nt.links.new(noise.outputs['Fac'], var.inputs['Value'])
    rock = nt.nodes.new('ShaderNodeMix')
    rock.data_type, rock.blend_type = 'RGBA', 'MULTIPLY'
    rv.socket(rock.inputs, 'Factor_Float').default_value = 1.0
    rv.socket(rock.inputs, 'A_Color').default_value = (*BED_RGB, 1)
    nt.links.new(var.outputs['Result'], rv.socket(rock.inputs, 'B_Color'))
    swap = nt.nodes.new('ShaderNodeMix')
    swap.data_type = 'RGBA'
    nt.links.new(mask.outputs['Color'], rv.socket(swap.inputs, 'Factor_Float'))
    nt.links.new(image, rv.socket(swap.inputs, 'A_Color'))
    nt.links.new(rv.socket(rock.outputs, 'Result_Color'), rv.socket(swap.inputs, 'B_Color'))
    for link in [l for l in nt.links if l.from_socket == image and l.to_node != swap]:
        nt.links.new(rv.socket(swap.outputs, 'Result_Color'), link.to_socket)
    aov = next(n for n in nt.nodes if n.type == 'OUTPUT_AOV' and n.aov_name == 'ice')
    w = aov.inputs['Value'].links[0].from_socket
    keep = nt.nodes.new('ShaderNodeMath')
    keep.operation = 'SUBTRACT'
    keep.inputs[0].default_value = 1.0
    nt.links.new(mask.outputs['Color'], keep.inputs[1])
    off = nt.nodes.new('ShaderNodeMath')
    off.operation = 'MULTIPLY'
    nt.links.new(w, off.inputs[0])
    nt.links.new(keep.outputs[0], off.inputs[1])
    for link in [l for l in nt.links if l.from_socket == w and l.to_node != off]:
        nt.links.new(off.outputs[0], link.to_socket)


def add_desnow(mat, rv):
    """Ground material (near terrain and far field): image pixels brighter than SNOW_LUM, the snow and firn of
    SWISSIMAGE 2024 outside the modelled ice, take the bed colour with its noise (smoothstep between the two
    luminances), and the glacier weight of the albedo curve goes to 0 there. The modelled ice is a surface of its
    own and is not touched."""
    nt = mat.node_tree
    bw = next(n for n in nt.nodes if n.type == 'RGBTOBW')
    image = bw.inputs['Color'].links[0].from_socket
    pos = nt.nodes.new('ShaderNodeNewGeometry').outputs['Position']
    lum = nt.nodes.new('ShaderNodeRGBToBW')
    nt.links.new(image, lum.inputs['Color'])
    fac = nt.nodes.new('ShaderNodeMapRange')
    fac.interpolation_type = 'SMOOTHSTEP'
    fac.inputs['From Min'].default_value, fac.inputs['From Max'].default_value = SNOW_LUM
    nt.links.new(lum.outputs['Val'], fac.inputs['Value'])
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 0.08
    noise.inputs['Detail'].default_value = 6.0
    nt.links.new(pos, noise.inputs['Vector'])
    var = nt.nodes.new('ShaderNodeMapRange')
    var.inputs['To Min'].default_value, var.inputs['To Max'].default_value = 1 - BED_NOISE, 1 + BED_NOISE
    nt.links.new(noise.outputs['Fac'], var.inputs['Value'])
    rock = nt.nodes.new('ShaderNodeMix')
    rock.data_type, rock.blend_type = 'RGBA', 'MULTIPLY'
    rv.socket(rock.inputs, 'Factor_Float').default_value = 1.0
    rv.socket(rock.inputs, 'A_Color').default_value = (*BED_RGB, 1)
    nt.links.new(var.outputs['Result'], rv.socket(rock.inputs, 'B_Color'))
    desnow = nt.nodes.new('ShaderNodeMix')
    desnow.data_type = 'RGBA'
    nt.links.new(fac.outputs['Result'], rv.socket(desnow.inputs, 'Factor_Float'))
    nt.links.new(image, rv.socket(desnow.inputs, 'A_Color'))
    nt.links.new(rv.socket(rock.outputs, 'Result_Color'), rv.socket(desnow.inputs, 'B_Color'))
    for link in [l for l in nt.links if l.from_socket == image and l.to_node not in (desnow, lum)]:
        nt.links.new(rv.socket(desnow.outputs, 'Result_Color'), link.to_socket)
    aov = next(n for n in nt.nodes if n.type == 'OUTPUT_AOV' and n.aov_name == 'ice')
    w = aov.inputs['Value'].links[0].from_socket
    keep = nt.nodes.new('ShaderNodeMath')
    keep.operation = 'SUBTRACT'
    keep.inputs[0].default_value = 1.0
    nt.links.new(fac.outputs['Result'], keep.inputs[1])
    off = nt.nodes.new('ShaderNodeMath')
    off.operation = 'MULTIPLY'
    nt.links.new(w, off.inputs[0])
    nt.links.new(keep.outputs[0], off.inputs[1])
    for link in [l for l in nt.links if l.from_socket == w and l.to_node != off]:
        nt.links.new(off.outputs[0], link.to_socket)


def ice_material(rv):
    import bpy
    mat = bpy.data.materials.new('igm_ice')
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    bsdf.inputs['Base Color'].default_value = (*ICE_RGB, 1)
    bsdf.inputs['Roughness'].default_value = ICE_ROUGH
    bsdf.inputs['Specular IOR Level'].default_value = ICE_SPEC
    aov = nt.nodes.new('ShaderNodeOutputAOV')
    aov.aov_name = 'ice'
    aov.inputs['Value'].default_value = 1.0
    return mat


def ice_object(rv, g, Z):
    """Mesh on the 2 m L2 lattice over the cells that ever carry ICE_MIN m (grown by 30 m); returns the object, the
    bed at its vertices and the bilinear weights from the 10 m grid."""
    import bpy
    st = np.load(STATIC)
    x, y = st['x'].astype(float), st['y'].astype(float)
    ever = (Z['thk'] >= ICE_MIN).any(0) | (Z['thk_hind_2021'] >= ICE_MIN)
    for _ in range(3):                                          # grown by 3 cells (Blender's Python has no scipy)
        p = np.pad(ever, 1)
        ever = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    e = np.arange(x[0] - 5, x[-1] + 5 + 1, 2.0)
    n = np.arange(y[-1] + 5, y[0] - 5 - 1, -2.0)
    E, N = np.meshgrid(e, n)
    ci = np.clip(((E - x[0] + 5) // 10).astype(int), 0, len(x) - 1)
    ri = np.clip(((N - y[0] + 5) // 10).astype(int), 0, len(y) - 1)
    keep = ever[ri, ci]
    bed = np.full(E.shape, np.nan)
    for k in ('L3', 'L2', 'L1', 'L0'):                           # finest level last
        e0, e1, n0, n1, s = rv.LEVELS[k]
        inside = (E >= e0) & (E <= e1) & (N >= n0) & (N <= n1)
        bed[inside] = rv.bilinear(g[k], e0, n1, s, E[inside], N[inside])
    keep &= np.isfinite(bed)
    cell = keep[:-1, :-1] & keep[1:, :-1] & keep[:-1, 1:] & keep[1:, 1:]
    idx = np.full(E.shape, -1, np.int64)
    idx[keep] = np.arange(keep.sum())
    i, j = np.nonzero(cell)
    a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
    tri = np.stack([np.stack([a, c, b], 1), np.stack([b, c, d], 1)], 1).reshape(-1, 3)
    ve, vn, vb = E[keep], N[keep], bed[keep]
    me = bpy.data.meshes.new('igm_ice')
    me.vertices.add(len(ve))
    me.vertices.foreach_set('co', np.stack([ve - rv.E0, vn - rv.N0, vb - 50 - rv.H0], 1).astype(np.float32).ravel())
    me.loops.add(tri.size)
    me.loops.foreach_set('vertex_index', tri.astype(np.int32).ravel())
    me.polygons.add(len(tri))
    me.polygons.foreach_set('loop_start', np.arange(0, tri.size, 3, dtype=np.int32))
    me.update(calc_edges=True)
    me.shade_smooth()
    me.materials.append(ice_material(rv))
    ob = bpy.data.objects.new('igm_ice', me)
    for owner in (ob, getattr(ob, 'cycles', None)):              # no object motion blur (Blender 5 or older API)
        if owner is not None and hasattr(owner, 'use_motion_blur'):
            owner.use_motion_blur = False
    bpy.context.scene.collection.objects.link(ob)
    # bilinear weights from the 10 m cell centres (rows south to north) to every lattice node
    fx = np.clip((E - x[0]) / 10, 0, len(x) - 1.001)
    fy = np.clip((N - y[0]) / 10, 0, len(y) - 1.001)
    jj, ii = fx.astype(int), fy.astype(int)
    u, v = fx - jj, fy - ii
    # the model's own bed (10 m) at the lattice nodes: thick ice is drawn on it (set_ice)
    topg = Z['topg'].astype(np.float64)
    mb = ((topg[ii, jj] * (1 - u) + topg[ii, jj + 1] * u) * (1 - v)
          + (topg[ii + 1, jj] * (1 - u) + topg[ii + 1, jj + 1] * u) * v)[keep]
    d = vb - mb
    print(f'ice mesh: {len(ve)} vertices, {len(tri)} triangles; lattice bed minus model bed: 5-95 % '
          f'{np.percentile(d, 5):.2f} to {np.percentile(d, 95):.2f} m', flush=True)
    return ob, vb, (ii, jj, u, v, keep, topg)


def set_ice(ob, vb, wts, thk, rv):
    """Ice heights: thickness bilinear from the 10 m grid to the 2 m lattice, smoothed by a Gaussian of THK_SMOOTH m
    (no creases at the 10 m cell edges); surface = (1 - w) (lattice bed + thickness) + w (model surface topg +
    thickness, bilinear and smoothed the same way) - ICE_SINK, w = smoothstep(thickness / RELIEF_DEPTH): thin ice
    follows the ground, thick ice is the model's own surface (so no ice is stacked on relief finer than the model
    grid, e.g. on a ridge next to ice that the basin rule cuts off)."""
    ii, jj, u, v, keep, topg = wts
    bil = lambda a: (a[ii, jj] * (1 - u) + a[ii, jj + 1] * u) * (1 - v) + (a[ii + 1, jj] * (1 - u) + a[ii + 1, jj + 1] * u) * v
    t = np.maximum(rv.gauss(bil(thk), THK_SMOOTH / 2), 0)[keep]
    sm = rv.gauss(bil(topg + thk), THK_SMOOTH / 2)[keep]
    w = np.clip(t / RELIEF_DEPTH, 0, 1)
    w = w * w * (3 - 2 * w)
    me = ob.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    co[:, 2] = (1 - w) * (vb + t) + w * sm - ICE_SINK - rv.H0
    me.vertices.foreach_set('co', co.ravel())
    me.update()


def render(args):
    import bpy
    from mathutils import Vector
    sys.path.insert(0, TERRAIN)
    import render_views as rv
    Z = dict(np.load(STATES))
    T = frames_of(Z)
    opt = {'--frames': f'0-{len(T) - 1}', '--share': None, '--weights': None, '--width': '1920',
           '--samples': str(SAMPLES), '--out': 'film3d_frames', '--measure': '0', '--exposure': str(EXPOSURE),
           '--sky': str(SKY), '--state': None, '--states': None, '--eye': None, '--look': None, '--lens': None,
           '--desnow': '0'}
    while args:
        k = args.pop(0)
        opt[k] = args.pop(0)
    frames = sorted({f for part in opt['--frames'].split(',') for f in
                     (range(int(part.split('-')[0]), int(part.split('-')[1]) + 1) if '-' in part else [int(part)])})
    if opt['--share'] is not None:
        w = np.array([float(x) for x in opt['--weights'].split(',')])
        edges = np.cumsum(w) / w.sum()
        frames = [f for f in frames if np.searchsorted(edges, (f * 0.6180339887) % 1, side='right')
                  == int(opt['--share'])]
    width, out = int(opt['--width']), os.path.join(IGM, opt['--out'])
    os.makedirs(out, exist_ok=True)
    os.makedirs(f'{TERRAIN}/renders', exist_ok=True)
    rv.grids = lowered_grids(rv, rv.grids)
    scene, solids, sun_ob, cam, _ = rv.build_scene(width)
    for slot in bpy.data.objects['basodino_terrain'].material_slots:
        add_bed(slot.material, rv)
    if opt['--desnow'] == '1':                              # every ground material: those with the 'ice' AOV
        for mat in bpy.data.materials:
            if mat.use_nodes and any(n.type == 'OUTPUT_AOV' and n.aov_name == 'ice' for n in mat.node_tree.nodes):
                add_desnow(mat, rv)
    ice, vb, wts = ice_object(rv, rv.grids.g, Z)
    r, cy, vl = scene.render, scene.cycles, scene.view_layers[0]
    r.resolution_x, r.resolution_y = width, width * 9 // 16
    r.film_transparent, r.use_persistent_data = True, True
    measure = opt['--measure'] == '1'
    vl.use_pass_diffuse_color = vl.use_pass_diffuse_direct = measure
    cy.samples, cy.seed, cy.use_animated_seed = int(opt['--samples']), 0, False
    cy.use_denoising, cy.denoiser, cy.denoising_input_passes = True, 'OPENIMAGEDENOISE', 'RGB_ALBEDO_NORMAL'
    cy.denoising_use_gpu = True
    cam.data.lens = LENS
    r.use_motion_blur, r.motion_blur_shutter, r.motion_blur_position = True, SHUTTER, 'CENTER'
    horizon = np.array(scene['horizon'], np.float32).reshape(-1, 3)
    offsets = (-SHUTTER / 2, SHUTTER / 2, 0)
    eye, look = camera(np.add.outer(frames, offsets).ravel(), len(T))
    k_exp, s_sky = float(opt['--exposure']), float(opt['--sky'])
    states = ([int(x) for x in opt['--states'].split(',')] if opt['--states'] else
              [None if opt['--state'] is None else int(opt['--state'])])
    fixed = opt['--eye'] is not None                     # one free camera (--eye, --look, --lens): stills, no blur
    if fixed:
        frames = frames[:1]
        E = np.array([float(x) for x in opt['--eye'].split(',')])
        L = np.array([float(x) for x in opt['--look'].split(',')])
        eye, look = np.tile(E, (3 * len(frames), 1)), np.tile(L, (3 * len(frames), 1))
        cam.data.lens = float(opt['--lens'] or LENS)
    for m, f in enumerate(frames):
        for state in states:
            stem = (f'view_s{state:04d}' if fixed else f'{f:04d}' if state is None else f'{f:04d}_s{state:04d}')
            path = f'{out}/{stem}.png'
            if os.path.exists(path) and not measure:
                continue
            for j, o in enumerate(offsets):
                p = Vector(eye[3 * m + j] - (rv.E0, rv.N0, rv.H0))
                q = Vector(look[3 * m + j] - (rv.E0, rv.N0, rv.H0))
                cam.location, cam.rotation_euler = p, (q - p).to_track_quat('-Z', 'Y').to_euler()
                cam.keyframe_insert('location', frame=f + o)
                cam.keyframe_insert('rotation_euler', frame=f + o)
            scene.frame_set(f)
            set_ice(ice, vb, wts, thickness(Z, T[f if state is None else state]), rv)
            r.filepath, t0 = f'{out}/_{stem}.exr', time.time()
            bpy.ops.render.render(write_still=True)
            passes = rv.read_exr(r.filepath)
            os.remove(r.filepath)
            rgb, alpha, env = passes['Combined'][..., :3], passes['Combined'][..., 3], passes['Environment']
            elevation, azimuth = rv.ray_angles(cam, r.resolution_x, r.resolution_y)
            low = (alpha < 0.5) & (elevation < 0)
            env[low] = horizon[((azimuth[low] + 180) / 360 * len(horizon)).astype(int) % len(horizon)]
            if measure:
                k, s, _, stats = rv.expose(passes, elevation, azimuth, scene['level_direct'], horizon)
                print(f'MEASURE {f} exposure {k:.4f} sky {s:.3f} ice_p50 {stats["ice_p50"]:.3f} rock_p50 '
                      f'{stats["rock_p50"]}', flush=True)
                rv.write_png(f'{out}/measure_{stem}.png', rv.tone(rgb + s * env, k))
                continue
            rv.write_png(path, rv.tone(rgb + s_sky * env, k_exp))
            print(f'FRAME {stem} {time.time() - t0:.1f} s', flush=True)


# --- compose (plain Python: numpy, PIL, ffmpeg) -----------------------------------------------------------------------

def compose(args):
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    check = args[args.index('--check') + 1] if '--check' in args else None
    args = [a for a in args if a not in ('--check', check)]
    src = args[0] if args else f'{IGM}/film3d_frames'
    Z = np.load(STATES)
    T = frames_of(Z)
    stop, dis, end = int(Z['stop']), int(Z['dis']), int(Z['end'])
    MP4 = f'{IGM}/results/f7/igm_3d_1850_{end}.mp4'
    F = {r['quantity']: r['value'] for r in csv.DictReader(open(FILM))}
    yrs = list(Z['years'])
    A = lambda Y: (Z['thk'][yrs.index(Y)] >= 1.0).sum() / 1e4
    # highest ice face at the F2 basin edge in the hindcast years, rounded down to 10 m (checks/basin_wall.py)
    step = 10 * (max(float(r['largest_face_at_basin_edge_m']) for r in
                     csv.DictReader(open(f'{IGM}/results/f7_basin_wall.csv')) if int(r['year']) <= 2020) // 10)
    W, H = Image.open(next(os.path.join(src, f) for f in sorted(os.listdir(src)) if f[0].isdigit())).size
    u = H / 1080
    font = lambda w, s: ImageFont.truetype(f'{FONTS}/SourceSans3-{w}.ttf', round(s * u))
    f_name, f_year, f_vol, f_src = font('Regular', 24), font('Semibold', 60), font('Regular', 30), font('Regular', 20)
    x0, y0 = round(56 * u), round(48 * u)
    t0, t1, ty = round(56 * u), W - round(56 * u), H - round(56 * u)
    tx = lambda year: t0 + (year - 1850) / (end + 0.75 - 1850) * (t1 - t0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    corner = np.clip(1 - np.hypot(xx / (820 * u), yy / (330 * u)), 0, 1)
    band = np.clip((yy - (H - 170 * u)) / (130 * u), 0, 1)
    BACK = Image.fromarray(np.round(BACKING * 255 * np.maximum(corner * corner * (3 - 2 * corner),
                                                               band * band * (3 - 2 * band))).astype(np.uint8))

    def status(job, t):
        if job[0] == 'morph' and job[3] == 0.0:                       # the hindcast's 2021 state, held
            return 'Hindcast from a steady 1850 state, observed weather'
        if job[0] == 'morph' or (job[1] == 171 and job[2] == 171):
            return '2021: start from the measured ice thickness'
        if t < 2021.75:
            return 'Hindcast from a steady 1850 state, observed weather'
        if t <= 2026.75:
            return 'Observed weather'
        if t < dis + 0.75:
            return 'Scenario SSP2-4.5, 2026 warming anchored to observation'
        if t < stop + 0.75:
            return f'Ice area below 0.01 km² from {dis}'
        return f'No ice of 1 m or more from {stop}'

    def text_layer(f, fade=1.0):
        job = T[f]
        t = year_of(job)
        thk = thickness(Z, job)
        vol = float(thk.sum()) * 100 / 1e6                    # million m3
        layer = Image.new('RGBA', (W, H))
        d = ImageDraw.Draw(layer)
        white = lambda a: (255, 255, 255, round(a * fade))
        d.text((x0, y0), 'Ghiacciaio del Basòdino, ice flow model', font=f_name, fill=white(225))
        d.text((x0, y0 + 34 * u), str(int(np.floor(t))), font=f_year, fill=white(255))
        d.text((x0, y0 + 112 * u), f'{vol:.2f} million m³ of ice', font=f_vol, fill=white(255))
        d.text((x0, y0 + 152 * u), status(job, t), font=f_src, fill=white(215))
        d.line([(t0, ty), (t1, ty)], fill=white(160), width=max(1, round(1.5 * u)))
        for year in (1850, 1900, 1950, 2000, end):
            r = 3.5 * u
            d.ellipse([tx(year) - r, ty - r, tx(year) + r, ty + r], fill=white(215))
        r = 7 * u
        d.ellipse([tx(t) - r, ty - r, tx(t) + r, ty + r], outline=white(255), width=round(2 * u))
        for year, anchor in ((1850, 'la'), (end, 'ra')):
            d.text((tx(year), ty + 12 * u), str(year), font=f_src, fill=white(215), anchor=anchor)
        return layer

    def labelled(base, layer, fade=1.0):
        dark = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        dark.putalpha(BACK.point(lambda v: v * fade))
        shadow = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        shadow.putalpha(layer.getchannel('A').filter(ImageFilter.GaussianBlur(3 * u)).point(lambda v: v * 0.6))
        out = Image.alpha_composite(base.convert('RGBA'), dark)
        return Image.alpha_composite(Image.alpha_composite(out, shadow), layer)

    probe = ImageDraw.Draw(Image.new('L', (1, 1)))

    def wrap(text, fnt, width):
        words, line, lines = text.split(), '', []
        for word in words:
            trial = f'{line} {word}'.strip()
            if probe.textlength(trial, font=fnt) > width and line:
                lines.append(line)
                line = word
            else:
                line = trial
        return lines + [line]

    card_text = [
        (f'Ghiacciaio del Basòdino 1850–{end}, modelled', 'title'),
        (f"Instructed Glacier Model 3.2.0 on a 10 m grid, one member of a 9984-member ensemble: posterior draw "
         f"{F['draw']}, climate model chain {F['chain']} (CH2025, Locarno-Monti), scenario SSP2-4.5, 2026 warming "
         f"anchored to the observed trend. Of the 1664 members with this scenario and warming level it has the median "
         f"year in which the ice area falls below 0.01 km² ({F['median disappearance year 0.01 km2 (lower middle)']}); "
         f"the film ends 3 years later.", 'body'),
        (f"Ice: the modelled thickness on the modelled bed, linear between the yearly states of 1 October. 1850–2021 "
         f"is a hindcast from a steady 1850 state; its ice area on 1 October 1929, 1973, 2010 and 2018 is {A(1929):.2f}, "
         f"{A(1973):.2f}, {A(2010):.2f} and {A(2018):.2f} km² against the mapped 3.20, 2.32, 1.90 and 1.64 km². In 2021 "
         f"the model restarts from the measured ice thickness ({A(2021):.2f} km²); its 2026 ice area is {A(2026):.2f} "
         f"km², the mapped one 1.02 km². The model removes ice that flows beyond the 1850 outline plus 100 m, so until "
         f"2021 the hindcast ice ends there in a wall of up to {step:.0f} m: a feature of the model, not of the glacier.",
         'body'),
        ('Ground: swissALTI3D and SWISSIMAGE 2024; under today’s ice of the glacier there is no image, the bed is drawn '
         'in a plain rock colour; the neighbouring glaciers are the aerial image of 2024, not modelled. Light of '
         '27 September, 15:30, for every year.', 'body'),
        ('swisstopo; Glacier Monitoring Switzerland (GLAMOS); MeteoSwiss; Swiss Glacier Inventory. Surroundings: '
         'Copernicus DEM GLO-30, © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under '
         'COPERNICUS by the European Union and ESA; contains modified Copernicus Sentinel data 2024.', 'small')]
    fonts = {'title': font('Semibold', 40), 'body': font('Regular', 27), 'small': font('Regular', 20)}
    cw = W * 0.62
    lines = []
    for text, kind in card_text:
        ls = wrap(text, fonts[kind], cw)
        lines += [(ln, fonts[kind], 0) for ln in ls[:-1]] + [(ls[-1], fonts[kind], 26 if kind == 'title' else 16)]
    last = None

    def card(base, a):
        b = base.filter(ImageFilter.GaussianBlur(10 * a * u)).point(lambda v: v * (1 - 0.6 * a)).convert('RGBA')
        layer = Image.new('RGBA', (W, H))
        d = ImageDraw.Draw(layer)
        y = H * 0.16
        for text, fnt, gap in lines:
            d.text(((W - cw) / 2, y), text, font=fnt, fill=(255, 255, 255, round(255 * a)))
            y += fnt.size * 1.3 + gap * u
        return Image.alpha_composite(b, layer)

    def video_frame(i):
        nonlocal last
        if i < len(T):
            return labelled(Image.open(f'{src}/{i:04d}.png'), text_layer(i)).convert('RGB')
        if last is None:
            last = Image.open(f'{src}/{len(T) - 1:04d}.png').convert('RGB')
        j = i - len(T)
        if j < STILL:
            fade = 1 - min(1.0, max(0.0, (j + 0.5) / STILL))
            return labelled(last, text_layer(len(T) - 1, fade), fade).convert('RGB')
        x = min(1.0, (j - STILL + 0.5) / CARD_IN)
        return card(last, x * x * (3 - 2 * x)).convert('RGB')

    total = len(T) + STILL + CARD
    if check:
        for i in map(int, check.split(',')):
            video_frame(i).save(f'{src}/check_{i:04d}.png')
        return
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                           '-r', str(FPS), '-i', '-', '-vf', 'scale=out_color_matrix=bt709:out_range=tv',
                           '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p',
                           '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                           '-color_range', 'tv', '-fps_mode', 'cfr', '-r', str(FPS), '-movflags', '+faststart', MP4],
                          stdin=subprocess.PIPE)
    for i in range(total):
        ff.stdin.write(video_frame(i).tobytes())
    ff.stdin.close()
    ff.wait()
    print(MP4, total, 'frames', total / FPS, 's')


def check(args):
    """results/f7_film3d_check.csv: the render frames (args[0], default film3d_frames) present, readable, 1920 x
    1080 RGB; mean brightness change from frame to frame (median, largest and where); and the camera path numbers."""
    from PIL import Image
    src = args[0] if args else f'{IGM}/film3d_frames'
    Z = np.load(STATES)
    n = len(frames_of(Z))
    m, bad = [], []
    for f in range(n):
        try:
            a = np.asarray(Image.open(f'{src}/{f:04d}.png').convert('RGB'), np.float32)
            assert a.shape == (1080, 1920, 3)
            m.append(a.mean())
        except Exception:
            bad.append(f)
            m.append(np.nan)
    d = np.abs(np.diff(np.array(m)))
    rows = [('render frames expected', n), ('missing or unreadable or not 1920 x 1080', len(bad)),
            ('mean brightness change per frame, grey levels: median', f'{np.nanmedian(d):.3f}'),
            ('largest, between frames', f'{np.nanmax(d):.2f} at {int(np.nanargmax(d))}-{int(np.nanargmax(d)) + 1}')]
    with open(f'{IGM}/results/f7_film3d_check.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['check', 'value'])
        w.writerows(rows)
    for r in rows:
        print(*r, sep=' | ')


def measured(args):
    """results/f7_film3d_measure.csv from the MEASURE lines of a `render --measure 1` log (args[0]): frame,
    exposure, sky factor, sunlit ice p50 and rock p50 at that exposure; and their medians (the fixed grade)."""
    rows = [l.split() for l in open(args[0]) if l.startswith('MEASURE ')]
    rows = [(r[1], r[3], r[5], r[7], f'{float(r[9]):.3f}') for r in rows]
    k = np.median([float(r[1]) for r in rows])
    s = np.median([float(r[2]) for r in rows])
    with open(f'{IGM}/results/f7_film3d_measure.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'exposure', 'sky', 'sunlit_ice_p50', 'sunlit_rock_p50'])
        w.writerows(rows)
        w.writerow(['median', f'{k:.4f}', f'{s:.3f}', '', ''])
    print(f'{len(rows)} frames: exposure median {k:.4f}, sky median {s:.3f} (film3d.py uses {EXPOSURE}, {SKY})')


if __name__ == '__main__':
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    {'prep': lambda a: prep(), 'path': lambda a: path(), 'render': render, 'compose': compose,
     'measured': measured, 'check': check}[argv[0]](argv[1:])
