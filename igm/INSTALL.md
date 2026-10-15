# IGM on s03

Instructed Glacier Model for the Basòdino ensemble, installed 2026-10-06 in
`s03:~/dev/AGL/basodino/igm/`. Environment, caches, temporaries and runs stay in this directory.
Copies of this file, `env.sh` and `bench/` (scripts, configs, `results.csv`) are on the Mac in `basodino/igm/`.

## Versions

| component | version |
|---|---|
| IGM | igm-model 3.2.0 (PyPI, released 2026-06-30, latest on 2026-10-06) plus the DahuNet emulator files of git tag v3.2.0 |
| examples | igm-examples tag v3.2 (924c6d3), in `igm-examples/` |
| Python | 3.11.17 from conda-forge, env `env/` (conda 26.1.1 of `~/miniconda3`) |
| TensorFlow | 2.17.0, pinned by IGM as `tensorflow[and-cuda]==2.17.0`; built for CUDA 12.3, cuDNN 8, sm_60/70/80/89 and compute_90 PTX |
| Keras, numpy | 3.15.1, 1.26.4 |
| CUDA libraries | pip wheels: cuda-runtime 12.3.101, cudnn 8.9.7.29, cublas 12.3.4.1, nvcc/ptxas 12.3.107, nccl 2.19.3 (no system CUDA used) |
| other | hydra-core 1.3.7, omegaconf 2.3.1, oggm 1.6.3, optuna 5.0.0, xarray 2026.9.0, netCDF4 1.7.4, rasterio 1.4.4 |
| system | Ubuntu 24.04.5, kernel 6.8.0-139, driver 595.91.07 (CUDA 13.2), 2 x H200 NVL (143771 MiB, compute capability 9.0) |

All packages: `pip-freeze.txt`. Upstream requirements (igm-model.org, Installation): Python 3.10 to 3.12,
only the NVIDIA driver (>= 525 for CUDA 12), about 5.5 GB on disk (this env: 5.7 GB).
On 2026-10-06 GitHub main is 64 commits ahead of v3.2.0, among them a precipitation-unit fix in
`climate/station` (1200ed93): pin a main commit if the climate modules are used.

## Install

```bash
mkdir -p ~/dev/AGL/basodino/igm/tmp && cd ~/dev/AGL/basodino/igm
~/miniconda3/bin/conda create -y -q -p ~/dev/AGL/basodino/igm/env --override-channels -c conda-forge python=3.11
TMPDIR=$PWD/tmp env/bin/python -m pip install --no-cache-dir igm-model==3.2.0
# the 3.2.0 wheel lacks the DahuNet emulators (upstream issue #55, fixed on main in 6cd10b3d)
SP=$(env/bin/python -c "import igm, os; print(os.path.dirname(igm.__file__))")/processes/iceflow/emulate/emulators
for f in dahunet.keras dahunet_mini.keras dahunet_micro.keras; do
  curl -sSfL -o $SP/$f https://raw.githubusercontent.com/instructed-glacier-model/igm/v3.2.0/igm/processes/iceflow/emulate/emulators/$f
done
git clone -q --depth 1 --branch v3.2 https://github.com/instructed-glacier-model/igm-examples.git
env/bin/python -m pip freeze > pip-freeze.txt
```

sha256 of the emulator files (git blob ids match the tag):

```
91b7f6e5740061bbbf5c13dc17ff868516dc9be7272d3bbc849ed59e5a149c90  dahunet.keras
5ffb0146142a25aefb2e4ebf1ed7c6fc25acc1bbd16b5f288e99e4fdd02b1702  dahunet_mini.keras
15e42b99c4003aac455e495e9dd800b183f01bc11d7e8a10d462c488f7800660  dahunet_micro.keras
```

The env was built 17:17-17:20 (conda command as in `env/conda-meta/history`, logs in `logs/`);
repeating the pip line afterwards was a no-op (92 requirements satisfied).

## Use

```bash
source ~/dev/AGL/basodino/igm/env.sh               # PATH (env, nrun), TMPDIR and Keras/matplotlib/CUDA caches in the project
CUDA_VISIBLE_DEVICES=1 igm_run +experiment=params  # one GPU per process
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"  # lists both H200
```

- igm_run turns on memory growth: a run holds 0.9 to 1.7 GB (below), not the card. It uses one GPU,
  `core.hardware.visible_gpus` (default [0], the first visible card; more than one raises NotImplementedError):
  pick the card with `CUDA_VISIBLE_DEVICES`.
- TensorFlow kernels reach sm_90 through the compute_90 PTX (JIT cache in `cache/nv`); XLA compiles natively with ptxas 12.3.
- TensorFlow leaves about ten `__autograph_generated_file*.py` per run in `tmp/`; empty `tmp/` after a campaign.
- On 2026-10-06 both H200 carried the Zn mesh48g QE campaign: 24 ph.x per GPU through an MPS server
  (pipe `/scratch/befahl/Zn/mesh48g/mpsd/pipe`), 38 GB used, 100 % utilisation, host load 106 to 126 of 128.
  IGM runs do not use that pipe; they run as ordinary CUDA contexts, time-sliced with it. Do not set
  `CUDA_MPS_PIPE_DIRECTORY` to it: a fatal fault in one MPS client can take down every client on the GPU.

## Proof: an IGM example

```bash
cd igm-examples/aletsch
CUDA_VISIBLE_DEVICES=1 igm_run +experiment=params_A_step1 "outputs=[local]" core.print_comp=true hydra.run.dir=$IGM_HOME/runs/example_aletsch_A1
```

Aletsch, 225 x 132 cells of 100 m, 1900-2000, frozen DahuNet emulator, live dashboard off: 555 steps,
final volume 16.82 km3, 41 s wall (18 s time loop), 928 MiB on the GPU. Output in `runs/example_aletsch_A1/`.

## Benchmark

`bench/`: `make_input.py` (synthetic inputs), `experiment/{offline,online,reference}.yaml`, `bench.py`
(starts N runs together, samples nvidia-smi, appends a row to `results.csv`), `ladder.sh` (the batches),
logs and configs of every run in `runs_2026-10-06.tar.gz`.

Case: 3.5 km x 3.5 km, 175 x 175 cells of 20 m or 350 x 350 of 10 m; bed sloping 0.22 (2450 to 3218 m)
with a parabolic cross profile; initial ice an ellipse of 1.88 km2, 90 m thick (0.113 km3); IGM `smb`
method simple, recomputed yearly, ELA 2950 m (2000) to 3150 m (2100); 100 years, cfl 0.3, output every
10 years. The forcing glaciates the upper half of the domain within 10 years (6.3 to 6.8 km2, 0.28 to
0.30 km3, surface speeds up to 45 to 58 m/yr), more ice than Basòdino carried in 1850 (4.0 km2): costs
are on the safe side.

Ice-flow solvers:

- offline: frozen DahuNet emulator (`dahunet_mini.keras`, Nz 2). Trained on 50 to 250 m grids; at 10 to 20 m it extrapolates (IGM does not check the grid spacing).
- online: IGM default, DahuNet trained from scratch at start (300 iterations) and retrained 5 iterations every 5 steps (Nz 4). Valid at any grid spacing.
- reference: identity mapping, velocity solved directly (L-BFGS, 50 iterations per step), as `iceflow_reference` in igm-examples.

All on GPU 1 under the QE load above (no idle H200 was available), runs niced to 10.
Loop = sum of IGM's module timers; start-up = wall minus loop (TensorFlow import, XLA compilation, initial training).

### One run alone

| solver | grid | steps per 100 yr | ms per step | s per simulated yr, loop | start-up s | wall s per 100 yr | GPU MiB |
|---|---|---|---|---|---|---|---|
| offline | 20 m | 381 | 18 | 0.069 | 11 | 18 | 928 |
| offline | 10 m | 732 | 19 | 0.137 | 16 | 30 | 1192 |
| online | 20 m | 393 | 32 | 0.125 | 29 | 41 | 1446 |
| online | 10 m | 758 | 29 | 0.221 | 32 | 54 | 1712 |
| reference | 20 m | 68 per 10 yr | 921 | 6.3 | 34 | 96 per 10 yr | 946 |
| reference | 10 m | stopped after 10 min at year 2.1 (28 steps of 10 to 60 s) | | | | | 1212 |

Halving the grid spacing doubles the steps (CFL); the time per step hardly changes with 4 times the cells,
so a single run is bound by kernel launches and time-slicing, not by arithmetic.

### N runs together on one H200

Per cell: s per simulated year per run (loop); aggregate glacier-years per hour for 100-year runs
including start-up / steady state (loop only).

| solver, grid | N = 1 | N = 4 | N = 8 | N = 16 |
|---|---|---|---|---|
| offline 20 m | 0.069; 19 k / 52 k | 0.097; 53 k / 149 k | 0.121; 87 k / 238 k | 0.129; 152 k / 447 k |
| online 20 m | 0.125; 9 k / 29 k | 0.158; 25 k / 91 k | 0.206; 44 k / 140 k | 0.287; 62 k / 201 k |
| offline 10 m | 0.137; 12 k / 26 k | 0.179; 38 k / 80 k | 0.273; 53 k / 106 k | 0.384; 86 k / 150 k |
| online 10 m | 0.221; 7 k / 16 k | 0.304; 20 k / 47 k | 0.417; 29 k / 69 k | 0.651; 40 k / 89 k |

Start-up grows with N (11 to 32 s alone, 23 to 75 s at N = 16). Host CPU idle during the batches:
15 to 16 % at N = 4, 12 to 15 % at N = 8, 8 to 9 % at N = 16.

### Both H200

Measured, online 10 m, 8 runs on each GPU at once: 0.44 s per simulated year per run, 57 k glacier-years
per hour including start-up, 130 k steady state; 1.98 and 1.88 times the 8-run batch on one GPU (the same
16 runs on one GPU: 1.43 and 1.47 times).

Both GPUs at 8 runs each (twice the N = 8 column), 100-year runs including start-up / steady state:

| solver, grid | glacier-years per hour |
|---|---|
| offline 20 m | 175 k / 475 k |
| online 20 m | 87 k / 279 k |
| offline 10 m | 106 k / 211 k |
| online 10 m | 57 k / 130 k (measured) |

16 runs per GPU would give 1.3 to 1.9 times more on paper but needs about 32 free cores; s03 had 10 to 25 during the test.

### How many runs fit on one H200

- Memory: 0.9 to 1.7 GB per run, so 60 or more fit beside the QE campaign (105 GB free per GPU), 80 or more on an idle card. Memory is not the limit.
- Throughput: offline 20 m still scales almost linearly at N = 16; online 10 m flattens (N = 8 to 16 adds 28 % steady state). Use 8 to 16 separate processes per GPU.
- CPU: every run keeps about one core busy; with the Zn campaigns running, 16 runs in total left 8 to 9 % of s03 idle.
- Batching: IGM 3.2 runs one glacier per process (no ensemble dimension, one GPU per process; Hydra multirun is sequential, the Optuna sweeper starts separate processes). Several members in one larger domain: untested.

### Agreement between the solvers

Volume after 10 years at 20 m: 0.1648 (offline), 0.1649 (online), 0.1649 km3 (reference). After 100
years the frozen emulator carries 5 % (20 m) and 6 % (10 m) more ice than the online solver; no reference
run that long.

### GPU used for this record

GPU 1: about 30 min of wall time in total, 2.9 process-hours summed over all runs, peak 27 GB (16 x 1.7 GB).
GPU 0: 1.7 min, 14 GB (8 x 1.7 GB). No process of ours is left on the GPUs.

### Reproduce

```bash
source ~/dev/AGL/basodino/igm/env.sh && cd $IGM_HOME/bench
python make_input.py
nrun bash ladder.sh singles                 # 5 rows, about 4 min (the 10 m reference line is commented out)
nrun bash ladder.sh concurrency             # 12 rows, about 13 min
nrun bash ladder.sh bothgpus online 10 8    # 2 rows, about 2 min
```
