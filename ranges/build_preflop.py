"""Builds ranges/nlhe_6max_100bb.json from the vendored Leakproof source.

    python3 ranges/build_preflop.py          # rewrite the database
    python3 ranges/build_preflop.py --check  # fail if the committed file is stale

Pure standard library (no numpy) so it runs with any Python 3.9+. What the conversion does
(the "changes" required by CC BY 4.0 attribution):

* links the decisions into a tree (each action points at the next decision, or is a terminal);
* stores frequencies as 169-entry arrays in a fixed hand-class order (see ``classes``), with
  the source's 6-decimal frequencies snapped to their exact steps (1/6 for pairs, 1/4 for
  other hands, as documented by the source);
* adds every terminal that reaches the flop or an all-in showdown as a "spot", with both
  players' ranges (class weights) and the pot and stacks in big blinds.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source" / "leakproof" / "6max-100bb-norake" / "1.0.0" / "nodes.json"
OUT = HERE / "nlhe_6max_100bb.json"

RANKS = "23456789TJQKA"
POSITIONS = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]  # preflop action order
POSTFLOP_ORDER = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
STACK_BB = 100.0
BLINDS = {"SB": 0.5, "BB": 1.0}


def class_names() -> list[str]:
    """The 169 hand classes in 13x13 grid order: row = first rank, column = second rank,
    from A down to 2; diagonal pairs, suited above the diagonal, offsuit below."""
    out = []
    for i in range(12, -1, -1):
        for j in range(12, -1, -1):
            if i == j:
                out.append(RANKS[i] * 2)
            elif i > j:
                out.append(RANKS[i] + RANKS[j] + "s")
            else:
                out.append(RANKS[j] + RANKS[i] + "o")
    return out


CLASSES = class_names()
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


def snap(v: float, cls: str) -> Fraction:
    step = 6 if len(cls) == 2 else 4
    f = Fraction(round(v * step), step)
    if abs(float(f) - v) > 1e-5:
        raise ValueError(f"{cls}: frequency {v} is not a multiple of 1/{step}")
    return f


def parse_action(label: str) -> dict:
    if label in ("fold", "call", "allin"):
        return {"kind": label, "label": label}
    kind, size = label.split()
    assert kind == "raise" and size.endswith("bb"), label
    return {"kind": "raise", "to_bb": float(size[:-2]), "label": label}


def build(src: list) -> dict:
    by_path = {tuple(n["history"]): i for i, n in enumerate(src)}
    nodes = []
    for i, n in enumerate(src):
        path = tuple(n["history"])
        actions = [parse_action(a) for a in n["actions"]]
        reach, strat = [], [[] for _ in actions]
        for cls in CLASSES:
            v = n["classes"].get(cls)
            if v is None:
                reach.append(0.0)
                for s in strat:
                    s.append(0.0)
                continue
            fr = [snap(x, cls) for x in v["strategy"]]
            if sum(fr) != 1:
                raise ValueError(f"node {i} {cls}: frequencies sum to {sum(fr)}")
            reach.append(float(v["reach"]))
            for s, f in zip(strat, fr):
                s.append(float(f))
        nodes.append({
            "id": i,
            "path": list(path),
            "actor": n["actor"],
            "line": n["line"],
            "raises": n["raises"],
            "pot_bb": n["pot_bb"],
            "to_call_bb": n["to_call_bb"],
            "actions": actions,
            "children": [None] * len(actions),
            "reach": reach,
            "strategy": strat,
        })

    # Children: the next decision is the node whose path extends this one by the action
    # index, possibly after implied folds (index 0) of seats that have no decision node.
    for node in nodes:
        for a in range(len(node["actions"])):
            p = (*node["path"], a)
            for _ in range(len(POSITIONS)):
                if p in by_path:
                    node["children"][a] = by_path[p]
                    break
                p = (*p, 0)

    check_reach(nodes, by_path)
    return {
        "schema": "pokerarena.preflop/1",
        "game": {
            "variant": "nlhe", "players": 6, "stack_bb": STACK_BB, "sb_bb": 0.5,
            "bb_bb": 1.0, "ante_bb": 0.0, "rake": "none",
        },
        "source": {
            "name": "Leakproof preflop ranges",
            "game": "6max-100bb-norake",
            "version": "1.0.0",
            "url": "https://github.com/a-hanamoto/leakproof-preflop-ranges",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "changes": "Converted to a linked tree with 169-entry arrays; frequencies snapped "
                       "to exact 1/6 and 1/4 steps; flop and all-in spots with both ranges "
                       "added. Strategy unchanged.",
        },
        "positions": POSITIONS,
        "postflop_order": POSTFLOP_ORDER,
        "classes": CLASSES,
        "nodes": nodes,
        "spots": build_spots(nodes),
    }


def walk(nodes, node_id, upto_action):
    """Yields (node, action index) along the path from the root to ``node_id`` and then the
    final action, skipping implied folds."""
    target = nodes[node_id]
    path = target["path"] + [upto_action]
    by_path = {tuple(n["path"]): n for n in nodes}
    for k, a in enumerate(path):
        n = by_path.get(tuple(path[:k]))
        if n is not None:
            yield n, a


def ranges_and_money(nodes, node_id, action):
    """Class weights of every player still in, and chips committed, after the line."""
    weights: dict[str, list[float]] = {}
    contrib = dict(BLINDS)
    folded: set[str] = set()
    current = 1.0
    for n, a in walk(nodes, node_id, action):
        who = n["actor"]
        w = weights.setdefault(who, [1.0] * 169)
        col = n["strategy"][a]
        weights[who] = [x * y for x, y in zip(w, col)]
        act = n["actions"][a]
        if act["kind"] == "fold":
            folded.add(who)
        elif act["kind"] == "call":
            contrib[who] = current
        elif act["kind"] == "raise":
            contrib[who] = current = act["to_bb"]
        else:  # allin
            contrib[who] = STACK_BB
            current = max(current, STACK_BB)
    live = [p for p in weights if p not in folded]
    return weights, contrib, live


def check_reach(nodes, by_path):
    """The source's ``reach`` must equal the product of the actor's own earlier strategy
    along the path: validates the tree links and the class order."""
    for node in nodes:
        w = [1.0] * 169
        path = node["path"]
        for k, a in enumerate(path):
            n = by_path.get(tuple(path[:k]))
            if n is not None and nodes[n]["actor"] == node["actor"]:
                w = [x * y for x, y in zip(w, nodes[n]["strategy"][a])]
        for c, (mine, theirs) in enumerate(zip(w, node["reach"])):
            if abs(mine - theirs) > 1e-5:
                raise ValueError(f"node {node['id']} {CLASSES[c]}: reach {theirs} "
                                 f"but the path gives {mine}")


def build_spots(nodes):
    spots = []
    for node in nodes:
        for a, act in enumerate(node["actions"]):
            if node["children"][a] is not None or act["kind"] not in ("call", "allin"):
                continue
            weights, contrib, live = ranges_and_money(nodes, node["id"], a)
            assert len(live) == 2, (node["line"], live)
            if act["kind"] == "allin" and any(contrib[p] < STACK_BB for p in live):
                continue  # a shove nobody has answered (no hand reaches the reply)
            pot = round(sum(contrib.values()), 6)
            assert abs(pot - (node["pot_bb"] + node["to_call_bb"])) < 1e-9, node["line"]
            behind = STACK_BB - max(contrib[p] for p in live)
            players = sorted(live, key=POSTFLOP_ORDER.index)
            spots.append({
                "line": (node["line"] + ", " if node["line"] else "")
                        + f"{node['actor']} {act['label']}",
                "node": node["id"],
                "action": a,
                "players": players,  # postflop order: first = out of position
                "pot_bb": pot,
                "stack_bb": behind,
                "allin": behind <= 0,
                "ranges": {p: weights[p] for p in players},
            })
    return spots


def render(db: dict) -> str:
    # One node / spot per line keeps diffs readable while staying compact.
    def dumps(x):
        return json.dumps(x, separators=(",", ":"))

    head = {k: v for k, v in db.items() if k not in ("nodes", "spots")}
    lines = ["{"]
    for k, v in head.items():
        lines.append(f"{json.dumps(k)}:{dumps(v)},")
    for key in ("nodes", "spots"):
        lines.append(f'"{key}":[')
        items = [dumps(x) for x in db[key]]
        lines.append(",\n".join(items))
        lines.append("]" + ("," if key == "nodes" else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the preflop range database.")
    p.add_argument("--check", action="store_true", help="verify the output is up to date")
    a = p.parse_args(argv)
    text = render(build(json.loads(SOURCE.read_text())))
    json.loads(text)  # well-formed
    if a.check:
        if OUT.read_text() != text:
            print(f"{OUT} is stale; run {Path(__file__).name}", file=sys.stderr)
            return 1
        print("up to date")
        return 0
    OUT.write_text(text)
    db = json.loads(text)
    print(f"wrote {OUT} ({len(text) // 1024} KiB): {len(db['nodes'])} decisions, "
          f"{len(db['spots'])} spots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
