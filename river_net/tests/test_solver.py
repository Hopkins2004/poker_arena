"""River solver: closed-form toy game, naive-CFR brute force, zero sum, exploitability,
determinism. (O(n) terminal vs O(n²) checks live in crates/solver/tests/solver.rs.)"""

import numpy as np
import pytest

from river_net import cards
from river_net.solver import Spot, solve, solve_abstract
from river_net.tree import build_tree

TOY = dict(oop_bets=[1.0], ip_bets=[], raises=[], allin=False, max_bets=1, allin_threshold=0.0)


# ------------------------------------------------------------------ closed-form toy game

@pytest.mark.parametrize("p", [0.2, 0.4, 0.6, 0.8])
def test_toy_game_matches_theory(p):
    # Hands: 0 = air, 1 = bluff-catcher, 2 = nuts. P1 holds nuts (p) or air; P2 the catcher.
    # Pot-size bet only; P2 may only check behind. No card removal.
    r = solve_abstract([0, 1, 2], [1 - p, 0, p], [0, 1, 0], 10.0, abstraction=TOY,
                       iters=3000)
    t = r.tree
    root = r.strategy_at(0)
    assert t.edge[t.first_child[0]] == "check"
    bet = t.find("bet")
    call = r.strategy_at(bet)[1, 1]  # P2's catcher: [fold, call]
    air_bets, nuts_bets = root[1, 0], root[1, 2]
    assert nuts_bets > 0.99
    if p <= 2 / 3:
        # Bluffs are 1/3 of the betting range: (1-p) x / (p + (1-p) x) = 1/3.
        assert air_bets == pytest.approx(p / (2 * (1 - p)), abs=0.01)
        assert call == pytest.approx(0.5, abs=0.01)
        share = (1 - p) * air_bets / (p * nuts_bets + (1 - p) * air_bets)
        assert share == pytest.approx(1 / 3, abs=0.01)
    else:
        assert air_bets > 0.99 and call < 0.01  # too few bluffs to call
    assert r.expl < 1e-3


# ------------------------------------------------------------------ naive CFR brute force

def naive_dcfr(tree, strength, cards_, r, iters, alpha=1.5, beta=0.0, gamma=2.0):
    """Per-deal recursive DCFR with the same update order as the vectorized solver."""
    n = len(strength)
    A = tree.max_actions

    def compat(h, g):
        if cards_ is None:
            return True
        return not set(cards_[h]) & set(cards_[g])

    def payoff(i, p, h, g):
        c = tree.contrib[i]
        if tree.kind[i] == "fold":
            f = int(tree.player[i])
            return -0.5 - c[f] if f == p else 0.5 + c[f]
        s = np.sign(strength[h] - strength[g])
        return s * (0.5 + c[p])

    regret = [np.zeros((tree.n_slots[p], A, n)) for p in range(2)]
    ssum = [np.zeros((tree.n_slots[p], A, n)) for p in range(2)]

    def rm(p):
        out = np.zeros_like(regret[p])
        for i in range(len(tree)):
            if tree.kind[i] != "decision" or tree.player[i] != p:
                continue
            s, k = tree.slot[i], tree.n_children[i]
            pos = np.maximum(regret[p][s, :k], 0)
            tot = pos.sum(axis=0)
            out[s, :k] = np.where(tot > 0, pos / np.where(tot > 0, tot, 1), 1 / k)
        return out

    cur = [rm(0), rm(1)]

    def value(i, p, h, g, w, inst):
        """u_p of deal (h for p, g for opponent) from node i; accumulates regrets."""
        if tree.kind[i] != "decision":
            return payoff(i, p, h, g)
        kids = list(tree.children(i))
        s = tree.slot[i]
        if tree.player[i] == p:
            vals = [value(c, p, h, g, w, inst) for c in kids]
            v = sum(cur[p][s, a, h] * vals[a] for a in range(len(kids)))
            for a in range(len(kids)):
                inst[s, a, h] += w * (vals[a] - v)
            return v
        o = 1 - p
        return sum(cur[o][s, a, g] * value(c, p, h, g, w * cur[o][s, a, g], inst)
                   for a, c in enumerate(kids))

    def own_reach(p):
        reach = np.zeros((len(tree), n))
        reach[0] = 1
        for i in range(len(tree)):
            if tree.kind[i] != "decision":
                continue
            for a, c in enumerate(tree.children(i)):
                reach[c] = reach[i] * (cur[p][tree.slot[i], a] if tree.player[i] == p else 1)
        return reach

    for t in range(1, iters + 1):
        dpos, dneg = t ** alpha / (t ** alpha + 1), t ** beta / (t ** beta + 1)
        dstrat = (t / (t + 1)) ** gamma
        for p in range(2):
            o = 1 - p
            inst = np.zeros_like(regret[p])
            for h in range(n):
                for g in range(n):
                    if compat(h, g) and r[o][g] > 0:
                        value(0, p, h, g, r[o][g], inst)
            reach = own_reach(p)
            regret[p] += inst
            regret[p] *= np.where(regret[p] > 0, dpos, dneg)
            for i in range(len(tree)):
                if tree.kind[i] == "decision" and tree.player[i] == p:
                    s, k = tree.slot[i], tree.n_children[i]
                    ssum[p][s, :k] = (ssum[p][s, :k] + reach[i] * cur[p][s, :k]) * dstrat
            cur[p] = rm(p)

    avg = []
    for p in range(2):
        a = np.zeros_like(ssum[p])
        for i in range(len(tree)):
            if tree.kind[i] == "decision" and tree.player[i] == p:
                s, k = tree.slot[i], tree.n_children[i]
                tot = ssum[p][s, :k].sum(axis=0)
                a[s, :k] = np.where(tot > 0, ssum[p][s, :k] / np.where(tot > 0, tot, 1), 1 / k)
        avg.append(a)

    def avg_value(i, p, h, g):
        if tree.kind[i] != "decision":
            return payoff(i, p, h, g)
        q = h if tree.player[i] == p else g
        st = avg[int(tree.player[i])][tree.slot[i]]
        return sum(st[a, q] * avg_value(c, p, h, g) for a, c in enumerate(tree.children(i)))

    vals = []
    for p in range(2):
        o = 1 - p
        v = np.zeros(n)
        for h in range(n):
            m = sum(r[o][g] for g in range(n) if compat(h, g))
            cf = sum(r[o][g] * avg_value(0, p, h, g) for g in range(n) if compat(h, g))
            v[h] = cf / m if m > 0 else 0.0
        vals.append(v)
    return avg, vals


def random_abstract_game(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 7))
    strength = rng.integers(0, 4, size=n).tolist()  # ties included
    cards_ = None
    if seed % 2 == 1:  # card pairs from a 6-card deck, so many hands collide
        pairs = [(a, b) for a in range(6) for b in range(a)]
        cards_ = [pairs[i] for i in rng.choice(len(pairs), size=n, replace=False)]
    r = [rng.random(n) * (rng.random(n) > 0.2) + 1e-3 for _ in range(2)]
    stack = float(rng.choice([0.4, 1.0, 2.5]))
    return strength, cards_, r, stack


@pytest.mark.parametrize("seed", range(12))
def test_vectorized_solver_matches_naive_cfr(seed):
    # Same algorithm, same update order: iterates agree up to float32 rounding. (Exact ties
    # make some regrets exactly 0, so after 15-20 iterations float32 vs float64 rounding can
    # flip a pure regret-matching choice and the trajectories fork; hence 10 iterations
    # here, and the converged game value below.)
    strength, cards_, r, stack = random_abstract_game(seed)
    iters = 10
    res = solve_abstract(strength, r[0], r[1], stack, cards=cards_, iters=iters)
    avg, vals = naive_dcfr(res.tree, strength, cards_, r, iters)
    for p, s in enumerate((res.strategy1, res.strategy2)):
        np.testing.assert_allclose(s, avg[p], atol=1e-5, err_msg=f"strategy p{p + 1}")
    np.testing.assert_allclose(res.v1, vals[0], atol=1e-5)
    np.testing.assert_allclose(res.v2, vals[1], atol=1e-5)


@pytest.mark.parametrize("seed", range(4))
def test_converged_game_value_matches_naive_cfr(seed):
    strength, cards_, r, stack = random_abstract_game(seed)
    res = solve_abstract(strength, r[0], r[1], stack, cards=cards_, iters=1000)
    _, vals = naive_dcfr(res.tree, strength, cards_, r, 200)
    m1 = np.array(res.m1)
    game = float((r[0] * m1 * res.v1).sum())
    naive = float((r[0] * m1 * vals[0]).sum())
    assert res.expl < 1e-3
    assert game == pytest.approx(naive, abs=3e-3)


# ------------------------------------------------------------------ real river spots

def random_spot(seed, spr=None):
    rng = np.random.default_rng(seed)
    b = cards.random_board(rng)
    v = cards.valid_mask(b)
    r1 = rng.random(1326) * v * (rng.random(1326) > 0.3)
    r2 = rng.random(1326) * v * (rng.random(1326) > 0.3)
    spr = spr if spr is not None else float(rng.choice([0.5, 1, 2, 4, 10]))
    return Spot(b, 10.0, 10.0 * spr, r1 / r1.sum(), r2 / r2.sum())


@pytest.mark.parametrize("seed", range(3))
def test_zero_sum_identity_and_values(seed):
    s = random_spot(seed)
    res = solve(s)
    v = cards.valid_mask(s.board)
    zs = (s.r1 * res.cfv1).sum() + (s.r2 * res.cfv2).sum()
    assert abs(zs) < 1e-5
    np.testing.assert_allclose(res.cfv1, res.m1 * res.v1, atol=1e-6)
    assert np.all(res.v1[~v] == 0) and np.all(res.m1[~v] == 0)
    # Values are defined for hands outside the range too, and bounded by the payoffs.
    assert np.all(np.abs(res.v1[v]) <= 0.5 + s.spr + 1e-4)
    out_of_range = v & (s.r1 == 0)
    assert out_of_range.any() and np.any(res.v1[out_of_range] != 0)


def test_exploitability_decreases_and_reaches_target():
    for seed in range(3):
        res = solve(random_spot(100 + seed), iters=500, target_expl=0.0)
        h = [e for _, e in res.history]
        assert all(b <= a * 1.2 + 1e-4 for a, b in zip(h, h[1:])), h
        assert h[-1] < 0.002 and h[-1] < h[0]
    early = solve(random_spot(7))
    assert early.expl < 0.002 and early.iters <= 500 and early.iters % 50 == 0


def test_determinism():
    s = random_spot(3)
    a, b = solve(s, strategy=True), solve(s, strategy=True)
    for k in ("v1", "v2", "m1", "m2", "strategy1", "strategy2"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert a.expl == b.expl and a.iters == b.iters


def test_tree_shapes():
    for spr, nodes in [(0.0, 3), (0.5, 9), (2.0, 33), (10.0, 57)]:
        t = build_tree(spr)
        assert len(t) == nodes
        assert t.kind[0] == "decision" and int(t.player[0]) == 0
        sd = [i for i in range(len(t)) if t.kind[i] == "showdown"]
        assert all(abs(t.contrib[i][0] - t.contrib[i][1]) < 1e-6 for i in sd)
        assert t.contrib.max() <= spr + 1e-6
