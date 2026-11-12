"""Worker-side code for train.py: recording bots and the per-process table runners.

Kept in its own module (not ``train.py``) so that worker processes, which import bots by
their ``python:bots.NLH.nn_training.workers.<Class>`` kind, share module globals with the
task functions that read them.
"""

from __future__ import annotations

import json

import numpy as np

from bots.NLH.neural_net import NeuralNet, features, init_layers, unflatten
from bots.NLH.tag import TightAggressive
from pokerarena import Run

# Labels for behaviour cloning. CHECK means "class 0 or 1" (both map to a check when
# checking is free), so the loss puts no preference between passive and medium there.
FOLD, CALL, RAISE, CHECK = 0, 1, 2, 3

ANCHORS = {
    "random": "python:bots.NLH.random_legal.RandomLegal",
    "station": "python:bots.NLH.calling_station.CallingStation",
    "tag": "python:bots.NLH.tag.TightAggressive",
}

# Per-process buffers (one task runs at a time in a worker).
_samples_x: list = []
_samples_y: list = []
_net = [0, 0]          # [chips won by the candidate's seats, candidate hands played]
_candidate = None      # [(W, b), ...] the ES candidate under evaluation


class Teacher(TightAggressive):
    """The anchored TAG bot, recording (features, abstract action) for every decision."""

    NAME = "teacher"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self._eq = NeuralNet(seed ^ 0x5EED, layers=init_layers(0))  # only for its equity()

    def act(self, obs):
        action = super().act(obs)
        kind = action.kind
        if kind == "fold":
            label = FOLD
        elif kind == "check":
            label = CHECK
        elif kind == "call":
            label = CALL
        else:  # bet_to / raise_to / all_in
            label = RAISE
        _samples_x.append(features(obs, self._eq.equity(obs)))
        _samples_y.append(label)
        return action


class Candidate(NeuralNet):
    """NeuralNet with the current ES candidate's weights; tallies its chips won."""

    NAME = "candidate"

    def __init__(self, seed: int = 0):
        super().__init__(seed, layers=_candidate if _candidate is not None else init_layers(0))
        self._seat = 0

    def new_hand(self, info):
        self._seat = int(info["seat"])

    def end_hand(self, result):
        _net[0] += int(result["net"][self._seat])
        _net[1] += 1


def table_config(lineup, *, seed, hands_per_table, n_tables, seats=6, duplicate=True):
    """JSON config text for one round of `n_tables` tables over `lineup` [(name, kind)]."""
    return json.dumps({
        "run": {"rounds": 1, "seed": seed},
        "format": {"variant": "nlhe", "max_seats": seats},
        "table": {
            "hands_per_table": hands_per_table,
            "starting_stack_bb": 100,
            "blinds": {"sb": 1, "bb": 2},
            "stack_policy": "reset_each_hand",
            "duplicate_mode": duplicate,
        },
        "matchmaking": {"min_appearances": 1, "min_tables": n_tables,
                        "allow_duplicate_bot_at_table": True},
        "bots": [{"name": n, "kind": k} for n, k in lineup],
    })


def _play(text, seed):
    run = Run(text, seed=seed, rounds=1, load_leaderboard=False)
    run.run_tables(0, list(range(len(run.schedule(0)))), threads=1)


def collect(args):
    """Plays teacher tables and returns the recorded (X, y)."""
    seed, hands_per_table, n_tables = args
    lineup = [("teacher", "python:bots.NLH.nn_training.workers.Teacher"),
              ("random", ANCHORS["random"]), ("station", ANCHORS["station"])]
    _samples_x.clear()
    _samples_y.clear()
    _play(table_config(lineup, seed=seed, hands_per_table=hands_per_table,
                       n_tables=n_tables, duplicate=False), seed)
    return np.array(_samples_x, dtype=np.float32), np.array(_samples_y, dtype=np.int8)


def evaluate(args):
    """Big blinds per hand won by a candidate (flat weights) against the anchored bots."""
    global _candidate
    flat, seed, hands_per_table, n_tables = args
    _candidate = unflatten(flat)
    _net[0] = _net[1] = 0
    lineup = [("candidate", "python:bots.NLH.nn_training.workers.Candidate"),
              *ANCHORS.items()]
    _play(table_config(lineup, seed=seed, hands_per_table=hands_per_table,
                       n_tables=n_tables), seed)
    return _net[0] / 2.0 / max(_net[1], 1)  # bb = 2 chips
