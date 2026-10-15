"""Surface mass balance of decision 3 (DESIGN.md): the equations of the reference smb_accmelt of the IGM example
aletsch (Hock 1999; Huss et al. 2009) as one function with a member axis, in numpy or, given TensorFlow tensors, in
TensorFlow on their device. For calib_smb.py, noflow.py and the IGM module user/smb_ti.

One hydrological year, day k = 0 on 1 October (the arrays of weather.py, shift 0; the reference instead rotates a
calendar year by int(365 x 0.75) days inside its loop). The snow store S starts at 0 on 1 October; for each day:
  c = P D                    T <= 0.5 degC          accumulation, m w.e.
  c = P D (2.5 - T) / 2      0.5 < T < 2.5 degC
  c = 0                      T >= 2.5 degC
  S = S + c
  a = max(T, 0) (Fm + r I) 0.91,  r = r_ice where S = 0, else r_snow      melt, m w.e. (0.91 m w.e. per m ice)
  S = max(S - a, 0)          no refreezing
  b = c - a
No accumulation or ablation weights: D carries the snow redistribution of decision 4 (and may carry a per-member
precipitation factor), the melt factors carry the rest. The reference's weights map onto them: D = weight_accumulation
x snow_redistribution; Fm, r_ice, r_snow times weight_ablation (checks/smb_reference.py). IGM's rate is
sum(b) / 0.91 m ice a-1, the reference's 1000 / 910.

Timing in clim_otl and smb_ti (decision 3; the reference differs): the last update starts at start - 1, so the first
update falls on the start time and the reference's placeholder (smb = 1000 / 910 = +1.10 m ice a-1 until its first
update, one year after the start) never acts; an update is due when t - t_last >= 1 (time.save = 1 from a start at
Y.75 lands on these times exactly); at t = Y.75 the hydrological year Y + 1 = hydro_year(t) is loaded, computed on the
current surface and applied as a rate (checks/smb_first_year.py).

Inputs (the member axis leads and broadcasts):
  T   air temperature at the cells, degC                       (day, cell) or (member, day, cell)
  P   precipitation at the cells, m w.e. d-1                   (day, cell) or (member, day, cell)
  I   potential direct radiation, W m-2, same day axis         (day, cell)
  D   accumulation factor                                      (cell,) or (member, cell)
  Fm  m ice d-1 degC-1; r_ice, r_snow m ice m2 W-1 d-1 degC-1, r_snow = 0.75 r_ice unless given; scalar or (member,)
Output: daily balance b, m w.e. d-1, (member, day, cell), in the dtype of T. It holds member x day x cell values:
chunk the member axis.

Station transfer of decision 5 with the gradients free (DESIGN.md, step 2, third attempt), transfer(): from the OTL
day values t, p to the cells at dz = z - 367 m,
  T = t + (g_m + dg) dz      g_m the monthly Robiei-Locarno gradient of the day, dg a per-member addition, K m-1
  P = p (1 + b dz)           b per member, m-1 (decision 5: 2 % per 100 m); c_prec goes into D
dg = 0 and b = B_PREC give decision 5 as fixed before step 2 (same operations, so the same values).
usage:  from smb import balance;  Ba = balance(T, P, I, D, Fm, r_ice).sum(axis=-2)    # (member, cell), m w.e.
        from smb import transfer;  T, P = transfer(t, p, g_m, dz, dg, b)   # dg, b shaped (member, 1)
"""
import numpy as np

T_SNOW, T_RAIN = 0.5, 2.5      # degC: all snow at or below, all rain at or above, linear between
ICE = 0.91                     # m w.e. per m ice in the melt term, as the reference
Z_OTL, B_PREC = 367.0, 0.02 / 100   # m, Locarno-Monti height; m-1, precipitation gradient of decision 5


def transfer(t, p, gm, dz, dg=0.0, b=B_PREC):
    """Air temperature and precipitation at the cells (module docstring): numpy or TensorFlow, broadcasting t, p,
    gm (day or scalar) against dz (cell) and dg, b (member, 1)."""
    return t + (gm + dg) * dz, p * (1 + b * dz)


def balance(T, P, I, D, Fm, r_ice, r_snow=None):
    """Daily balance of one hydrological year (module docstring), m w.e. d-1, shape (member, day, cell)."""
    r_snow = 0.75 * r_ice if r_snow is None else r_snow
    if isinstance(T, np.ndarray):
        xp = np
        P, I, D, Fm, r_ice, r_snow = (np.asarray(v, T.dtype) for v in (P, I, D, Fm, r_ice, r_snow))
    else:
        import tensorflow as xp
        P, I, D, Fm, r_ice, r_snow = (xp.cast(v, T.dtype) for v in (P, I, D, Fm, r_ice, r_snow))
    Fm, r_ice, r_snow = (xp.reshape(v, (-1, 1)) for v in (Fm, r_ice, r_snow))      # (member, 1)
    S, b = 0.0, []
    for k in range(T.shape[-2]):
        S, bk = step(S, T[..., k, :], P[..., k, :], I[k], D, Fm, r_ice, r_snow, xp)
        b.append(bk)
    return xp.stack(b, axis=-2)


def step(S, t, p, I, D, Fm, r_ice, r_snow, xp=np):
    """One day of the module docstring: snow store S and the day's t, p, I at the cells -> (S, b); Fm, r_ice and
    r_snow shaped (member, 1). balance() loops over it; calib_smb.py calls it inside a compiled day loop. The zero
    of the branches is a zero array of t, so TensorFlow's where gets one dtype (float32 or float64), as numpy does."""
    zero = xp.zeros_like(t)
    c = xp.where(t <= T_SNOW, p, xp.where(t >= T_RAIN, zero, p * (T_RAIN - t) / (T_RAIN - T_SNOW))) * D
    S = S + c
    tp = xp.where(t > 0.0, t, zero)
    a = xp.where(S == 0, tp * (Fm + r_ice * I) * ICE, tp * (Fm + r_snow * I) * ICE)
    S = xp.maximum(S - a, 0.0)
    return S, c - a


def hydro_year(t):
    """Hydrological year loaded at model time t (decimal year, 1 October = Y.75; a number or IGM's tf.Variable):
    [Y.75, Y + 1.75) -> Y + 1. Allows 1e-3 a of rounding in IGM's float32 time."""
    return int(np.floor(np.asarray(t, np.float64) + 0.25 + 1e-3))
