"""Small neural-network policy: 3 hidden layers of 10 neurons, 3 action classes.

The network reads 18 hand-crafted features of the current decision (equity bucket, pot odds,
SPR, previous-street aggressor, previous action, position, ...) and outputs a softmax over
three abstract actions, which are then mapped to a concrete legal action:

* ``0`` passive: fold, or check when checking is free;
* ``1`` medium: call, or check when checking is free;
* ``2`` aggressive: bet or raise (fixed sizes below); call if raising is not allowed.

Weights are read from ``nn_training/weights.npz`` next to this file (see
``nn_training/train.py`` for how to produce them). Without that file the bot plays with a
fixed random initialisation and prints a warning once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from pokerarena import Action, BaseBot, clamp_raise, equity, pot_fraction_to

PREFLOP = 0
LIVE = (1, 2)  # seat status: active or all-in
VOLUNTARY = ("check", "call", "bet", "raise")
AGGRESSIVE = ("bet", "raise")

WEIGHTS_FILE = Path(__file__).resolve().parent / "nn_training" / "weights.npz"

# ----------------------------------------------------------------------------- features

FEATURES = (
    "equity_bucket",          # Monte Carlo equity vs the live opponents, bucketed (centre)
    "pot_odds",               # to_call / (pot + to_call)
    "spr",                    # log1p(effective stack / pot) / log1p(50), clipped to [0, 1]
    "facing_size",            # to_call / pot, clipped to [0, 2], halved
    "street_preflop", "street_flop", "street_turn", "street_river",
    "prev_street_aggressor",  # +1 I made the last bet/raise on the previous street, -1 a
                              # villain did, 0 nobody (or preflop)
    "raises_this_street",     # voluntary bets + raises this street, min(n, 4) / 4
    "prev_action_none",       # last voluntary action this street (folds ignored):
    "prev_action_passive",    #   nobody yet / check or call /
    "prev_action_aggressive", #   bet or raise
    "my_action_this_street",  # my last action this street: 0 none, 0.5 check/call, 1 bet/raise
    "position",               # 0 = first to act this street ... 1 = last
    "opponents",              # live opponents / (seats - 1)
    "can_check",
    "can_raise",
)
N_IN = len(FEATURES)
HIDDEN = (10, 10, 10)
N_OUT = 3
EQUITY_BUCKETS = 10


def layer_shapes(n_in: int = N_IN, hidden=HIDDEN, n_out: int = N_OUT):
    """[(fan_in, fan_out), ...] for every dense layer."""
    sizes = (n_in, *hidden, n_out)
    return list(zip(sizes[:-1], sizes[1:]))


def n_params() -> int:
    return sum(i * o + o for i, o in layer_shapes())


def unflatten(flat):
    """Flat parameter vector -> [(W, b), ...]."""
    flat = np.asarray(flat, dtype=np.float64)
    out, k = [], 0
    for i, o in layer_shapes():
        w = flat[k:k + i * o].reshape(i, o)
        k += i * o
        b = flat[k:k + o]
        k += o
        out.append((w, b))
    assert k == flat.size, f"expected {k} parameters, got {flat.size}"
    return out


def flatten(layers) -> np.ndarray:
    return np.concatenate([np.concatenate([w.ravel(), b.ravel()]) for w, b in layers])


def init_layers(seed: int = 0):
    """Xavier-uniform initialisation (tanh hidden units)."""
    rng = np.random.default_rng(seed)
    out = []
    for i, o in layer_shapes():
        lim = np.sqrt(6.0 / (i + o))
        out.append((rng.uniform(-lim, lim, size=(i, o)), np.zeros(o)))
    return out


def forward(layers, x):
    """Softmax action probabilities for a feature vector (or a batch of them)."""
    h = x
    for w, b in layers[:-1]:
        h = np.tanh(h @ w + b)
    w, b = layers[-1]
    z = h @ w + b
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def load_weights(path=WEIGHTS_FILE):
    """[(W, b), ...] from an ``.npz`` written by the trainer (keys W0, b0, W1, ...)."""
    with np.load(path) as z:
        layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(len(HIDDEN) + 1)]
    for (w, b), (i, o) in zip(layers, layer_shapes()):
        if w.shape != (i, o) or b.shape != (o,):
            raise ValueError(f"{path}: layer shape {w.shape} does not match {(i, o)}")
    return layers


def save_weights(layers, path=WEIGHTS_FILE, **meta) -> None:
    arrays = {}
    for i, (w, b) in enumerate(layers):
        arrays[f"W{i}"], arrays[f"b{i}"] = w, b
    for k, v in meta.items():
        arrays[f"meta_{k}"] = np.asarray(v)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def live_opponents(obs) -> int:
    return sum(1 for s in range(obs.n_seats) if s != obs.seat and obs.status[s] in LIVE)


def equity_bucket(e: float) -> float:
    b = min(int(e * EQUITY_BUCKETS), EQUITY_BUCKETS - 1)
    return (b + 0.5) / EQUITY_BUCKETS


def features(obs, eq: float) -> np.ndarray:
    """The network input for ``obs``, given the (unbucketed) equity estimate ``eq``."""
    la = obs.legal
    n, me, street = obs.n_seats, obs.seat, obs.street
    status, stacks = obs.status, obs.stacks
    pot = max(int(obs.pot), 1)
    to_call = int(la.call_amount)
    history = obs.history

    my_stack = int(stacks[me])
    villain_stacks = [int(stacks[s]) for s in range(n) if s != me and status[s] in LIVE]
    effective = min(my_stack, max(villain_stacks)) if villain_stacks else 0
    spr = min(np.log1p(effective / pot) / np.log1p(50.0), 1.0)

    prev_aggr = 0.0
    raises = 0
    prev_action = 0          # 0 none, 1 passive, 2 aggressive
    my_action = 0.0
    bb_seat = obs.button
    for seat, st, kind, _amount, _to, _allin in history:
        if kind == "bb":
            bb_seat = seat
        if st == street - 1 and kind in AGGRESSIVE:
            prev_aggr = 1.0 if seat == me else -1.0
        elif st == street and kind in VOLUNTARY:
            aggressive = kind in AGGRESSIVE
            raises += aggressive
            prev_action = 2 if aggressive else 1
            if seat == me:
                my_action = 1.0 if aggressive else 0.5

    # Position among live players in this street's action order (preflop: left of the
    # big blind first; postflop: left of the button first).
    start = (bb_seat if street == PREFLOP else obs.button) + 1
    order = [(start + k) % n for k in range(n)]
    order = [s for s in order if status[s] in LIVE]
    position = order.index(me) / (len(order) - 1) if len(order) > 1 else 1.0

    x = np.zeros(N_IN)
    x[0] = equity_bucket(eq)
    x[1] = to_call / (pot + to_call)
    x[2] = spr
    x[3] = min(to_call / pot, 2.0) / 2.0
    x[4 + min(street, 3)] = 1.0
    x[8] = prev_aggr
    x[9] = min(raises, 4) / 4.0
    x[10 + prev_action] = 1.0
    x[13] = my_action
    x[14] = position
    x[15] = len(villain_stacks) / max(n - 1, 1)
    x[16] = float(la.can_check)
    x[17] = float(la.can_raise)
    return x


# ----------------------------------------------------------------------------- the bot

_cached_layers = None


def _default_layers():
    global _cached_layers
    if _cached_layers is None:
        try:
            _cached_layers = load_weights()
        except FileNotFoundError:
            print(f"neural_net: {WEIGHTS_FILE} not found; playing with untrained weights",
                  file=sys.stderr)
            _cached_layers = init_layers(0)
    return _cached_layers


class NeuralNet(BaseBot):
    """3x10 tanh MLP over equity, pot odds, SPR and action features; fold/call/raise.

    Picks passive / medium / aggressive from a softmax over 18 features and maps it to
    fold-or-check / call-or-check / bet-or-raise with fixed sizes.
    """

    NAME = "nn"

    SAMPLES = 200             # Monte Carlo equity samples per decision
    SAMPLE_POLICY = True      # sample from the softmax (False: always take the argmax)
    TEMPERATURE = 1.0         # softmax temperature when sampling
    OPEN_BB = 3               # preflop open: raise to OPEN_BB big blinds (+1 per limper)
    RERAISE_MULT = 3.0        # facing a raise: raise to RERAISE_MULT x the current bet
    BET_POT_FRAC = 0.66       # postflop bet when nobody has bet: fraction of the pot

    def __init__(self, seed: int = 0, layers=None):
        super().__init__(seed)
        self.rng = np.random.default_rng(seed)
        self.layers = layers if layers is not None else _default_layers()

    # -- decision -------------------------------------------------------------------

    def equity(self, obs) -> float:
        opponents = live_opponents(obs)
        if opponents == 0:
            return 1.0
        return float(equity(obs.hole, obs.board, opponents, self.SAMPLES,
                            seed=int(self.rng.integers(1 << 62)), variant=obs.variant))

    def choose(self, probs) -> int:
        if not self.SAMPLE_POLICY:
            return int(np.argmax(probs))
        if self.TEMPERATURE != 1.0:
            logits = np.log(np.maximum(probs, 1e-12)) / self.TEMPERATURE
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
        return int(self.rng.choice(N_OUT, p=probs))

    def act(self, obs):
        x = features(obs, self.equity(obs))
        return self.to_action(obs, self.choose(forward(self.layers, x)))

    # -- abstract action -> concrete legal action -----------------------------------------

    def to_action(self, obs, cls: int):
        la = obs.legal
        if cls == 0:
            return Action.check() if la.can_check else Action.fold()
        if cls == 1 or not la.can_raise:
            return Action.check() if la.can_check else Action.call()
        return clamp_raise(obs, self.raise_to(obs))

    def raise_to(self, obs) -> int:
        bb = obs.bb
        if obs.street == PREFLOP:
            if obs.current_bet <= bb:  # unopened (limpers at most)
                limpers = sum(1 for r in obs.history if r[1] == PREFLOP and r[2] == "call")
                return bb * (self.OPEN_BB + limpers)
            return int(obs.current_bet * self.RERAISE_MULT)
        if obs.current_bet == 0:
            return pot_fraction_to(obs, self.BET_POT_FRAC)
        return int(obs.current_bet * self.RERAISE_MULT)
