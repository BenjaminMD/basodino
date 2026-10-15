"""Start N identical IGM runs together on one GPU, time them, append one row to results.csv.

usage: python bench.py MODE DX N [YEARS] [GPU]
  MODE   offline | online | reference   (experiment/MODE.yaml)
  DX     10 | 20                        (data/synth_dxDX.nc, 350x350 or 175x175)
  N      runs started together (separate processes, niced to 10)
  YEARS  simulated years per run (default 100), GPU (default 1)

Per run: wall time, loop time (sum of the IGM module timers), time steps, peak GPU
memory (nvidia-smi every 0.5 s), final volume and peak surface speed. Per batch:
makespan, throughput in glacier-years per hour, host CPU idle during the batch.
output.nc is deleted after the speed check; logs and .hydra configs are kept.
"""
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent


def gpu_mib():
    """Used GPU memory (MiB) per compute process, all GPUs."""
    q = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    mem = {}
    for line in subprocess.run(q, capture_output=True, text=True).stdout.splitlines():
        pid, mib = (int(v) for v in line.split(","))
        mem[pid] = mem.get(pid, 0) + mib
    return mem


def cpu_times():
    """(idle, total) jiffies from /proc/stat."""
    f = [int(v) for v in open("/proc/stat").readline().split()[1:]]
    return f[3] + f[4], sum(f)


def parse_run(d):
    """Loop time, steps, final volume, peak surface speed of one finished run."""
    stats = (d / "computational-statistics.txt").read_text()
    loop = sum(float(v) for v in re.findall(r"total :\s+([\d.]+)", stats))
    log = (d / "igm.log").read_text(errors="replace")
    steps = int(re.findall(r"=(\d{6}),", log)[-1])
    vol = float(re.findall(r"Volume=([\d.]+)", log)[-1])
    with xr.open_dataset(d / "output.nc") as ds:
        vmax = float(ds["velsurf_mag"].max())
    (d / "output.nc").unlink()
    return loop, steps, vol, vmax


def main():
    mode, dx, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    years = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    gpu = sys.argv[5] if len(sys.argv) > 5 else "1"
    tag = f"{datetime.now():%m%d_%H%M%S}_{mode}_dx{dx}_n{n}_gpu{gpu}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    idle0, tot0 = cpu_times()
    runs = []
    t0 = time.time()
    for i in range(n):
        d = HERE / "runs" / tag / f"r{i}"
        d.mkdir(parents=True)
        cmd = ["nice", "-n", "10", "igm_run", f"+experiment={mode}",
               f"inputs.local.filename=synth_dx{dx}.nc", f"processes.time.end={2000 + years}",
               "core.print_comp=true", f"hydra.run.dir={d}"]
        with open(d / "igm.log", "w") as log:
            p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
        runs.append({"p": p, "d": d, "start": time.time(), "end": None, "mib": 0})
    while any(r["end"] is None for r in runs):
        mem = gpu_mib()
        for r in runs:
            r["mib"] = max(r["mib"], mem.get(r["p"].pid, 0))
            if r["end"] is None and r["p"].poll() is not None:
                r["end"] = time.time()
        time.sleep(0.5)
    makespan = time.time() - t0
    idle1, tot1 = cpu_times()
    ok = [r for r in runs if r["p"].returncode == 0]
    if not ok:
        sys.exit("no run finished: return codes %s" % [r["p"].returncode for r in runs])
    res = [parse_run(r["d"]) for r in ok]
    wall = np.array([r["end"] - r["start"] for r in ok])
    loop, steps, vol, vmax = (np.array(v) for v in zip(*res))
    row = {
        "batch": tag, "gpu": gpu, "mode": mode, "dx_m": dx, "grid": f"{3500 // dx}x{3500 // dx}",
        "years": years, "n": n, "ok": len(ok),
        "wall_mean_s": round(wall.mean(), 1), "makespan_s": round(makespan, 1),
        "startup_mean_s": round((wall - loop).mean(), 1),
        "loop_mean_s": round(np.mean(loop), 1), "steps_mean": round(np.mean(steps)),
        "ms_per_step": round(1000 * np.mean(loop / steps), 1),
        "s_per_yr_loop": round(np.mean(loop) / years, 3),
        "s_per_yr_wall": round(wall.mean() / years, 3),
        "gpu_mib_peak_mean": round(np.mean([r["mib"] for r in ok])),
        "gpu_mib_peak_max": max(r["mib"] for r in ok),
        "glacier_yr_per_h_batch": round(len(ok) * years / makespan * 3600),
        "glacier_yr_per_h_loop": round(len(ok) * 3600 / (np.mean(loop) / years)),
        "final_volume_km3": round(np.mean(vol), 4), "max_speed_m_yr": round(np.max(vmax), 1),
        "host_cpu_idle_pct": round(100 * (idle1 - idle0) / (tot1 - tot0), 1),
    }
    out = HERE / "results.csv"
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
