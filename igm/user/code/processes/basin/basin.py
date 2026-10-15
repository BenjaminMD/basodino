"""IGM process basin: F2 of the fix round (DESIGN.md): ice only inside the basin, the cells of posbal (decision 2:
SGI1850 of C14-10, plus its 100 m buffer on cells flatter than 30 degrees, without the 1850 outline of C14-11; the
same cells where smb_ti allows a positive balance). Ice that flows onto a cell outside the basin melts in the same
time step: list basin right after thk; thk = 0 outside, usurf = topg + thk.

Kept for track (state): basin_removed (y, x) the thickness removed in this step, m ice; basin_melt the volume
removed since the last save, m3 ice (track writes it as basin_melt_km3 and resets it).
"""
import tensorflow as tf


def initialize(cfg, state):
    state.basin = state.posbal > 0.5
    state.basin_removed = tf.zeros_like(state.thk)
    state.basin_melt = 0.0
    out = float(tf.reduce_sum(tf.where(state.basin, 0.0, state.thk)))
    if out > 0:                                          # start ice outside the basin melts at the first step
        print(f"basin: start ice outside the basin {out * 1e-9 * float(state.dx) ** 2:.6f} km3 on "
              f"{int(tf.reduce_sum(tf.cast((state.thk > 0) & ~state.basin, tf.int32)))} cells")


def update(cfg, state):
    if state.it < 0:
        return
    state.basin_removed = tf.where(state.basin, 0.0, state.thk)
    state.basin_melt += float(tf.reduce_sum(state.basin_removed)) * float(state.dx) ** 2
    state.thk = tf.where(state.basin, state.thk, 0.0)
    state.usurf = state.topg + state.thk


def finalize(cfg, state):
    pass
