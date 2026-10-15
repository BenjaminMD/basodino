"""IGM process warmstart: decision 1 warm start of the online solver (list it right after iceflow; run iceflow with
unified.nbit_init = 0).

IGM 3.2 can load a network only as a frozen-normalizer artifact (pretrained: true), and its artifact wrapper does not
take a network that has already been trained online. Here instead: iceflow initialises a fresh network as for a cold
start (its adaptive input normalisation computed on this run's start state) without training; this module then sets
every network weight except the normaliser's from net (a file track.py wrote with save_net after a cold run's
iteration 0) and trains nbit iterations from there with the initial-training settings (IGM's solve_iceflow with
init=True, adam lr_init), then evaluates the velocities. Afterwards the run retrains as usual (nbit every retrain_freq
steps). net empty: nothing happens.
"""
import copy
from pathlib import Path

import numpy as np


def network_weights(model):
    """The network's variables without those of its input normaliser, in a fixed order."""
    norm = {id(v) for v in getattr(model.input_normalizer, "weights", [])}
    return [v for v in model.weights if id(v) not in norm]


def initialize(cfg, state):
    c = cfg.processes.warmstart
    if not c.net:
        return
    from igm.processes.iceflow.unified.evaluator import evaluate_iceflow
    from igm.processes.iceflow.unified.solver import solve_iceflow
    vs = network_weights(state.iceflow_model)
    with np.load(Path(state.original_cwd) / c.net) as f:
        assert list(f["names"]) == [v.path for v in vs], "network layout differs from the saved one"
        for i, v in enumerate(vs):
            v.assign(f[f"w{i}"])
    c2 = copy.deepcopy(cfg)
    c2.processes.iceflow.unified.nbit_init = int(c.nbit)
    solve_iceflow(c2, state, init=True)
    evaluate_iceflow(c2, state)


def update(cfg, state):
    pass


def finalize(cfg, state):
    pass
