# Preflop ranges

`nlhe_6max_100bb.json` is a preflop strategy for 6-max No-Limit Hold'em, 100bb deep, no rake
and no ante. For every decision it gives a frequency per action for each of the 169 hand
classes, and from those it derives each player's range in every spot that reaches the flop.
Bots read it through `pokerarena.preflop`. The planned river solver uses the flop spots'
ranges as its starting ranges.

## Source and license

> Preflop ranges: Leakproof 6max-100bb-norake 1.0.0
> (https://github.com/a-hanamoto/leakproof-preflop-ranges), CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/). **Changed:** converted to a linked tree with
> 169-entry arrays, frequencies snapped to their exact 1/6 and 1/4 steps, and flop and all-in
> spots with both ranges added. The strategy itself is unchanged.

The original file is kept unchanged in `source/leakproof/6max-100bb-norake/1.0.0/nodes.json`
(SHA-256 `f6197db6…a27a266`), together with its `CITATION.cff`. The data is a computed
approximation and comes with no warranty. Its README has the method, the rounding (pairs in
steps of 1/6, other hands in steps of 1/4, so every step is a whole number of combos) and the
published range sizes, which `python/tests/test_preflop.py` checks.

## The game tree

| | |
|---|---|
| Opens | UTG and HJ 2bb, CO and BTN 2.5bb, SB 3bb. No limps, no open shoves |
| Calling an open | BTN and BB only; other seats 3-bet or fold |
| 3-bets | 3x the open in position, 4.5x from the SB, 5.5x from the BB |
| Calling a 3-bet | only the opener (no cold calls) |
| 4-bets | 2.1x the 3-bet in position, 2.65x out of position |
| 5-bets | all-in only; an all-in is an option at every decision facing a raise |
| Multiway | not modelled: once two players are in, everyone else folds (an *implied fold*) |

It has 78 decision points and 66 spots: 38 that reach the flop, and 28 that end in a preflop
all-in. Sizes are in big blinds. With the arena's 1/2 blinds, 2.5bb is a raise to 5 chips.

## Python

```python
from pokerarena import preflop

db = preflop.load()                        # loaded once per process, read-only

# Strategy at a decision: line = voluntary actions so far (folds left out), then the actor.
db.node("", "CO").freqs("KTo")             # {'fold': 0.0, 'raise 2.5bb': 1.0}
db.node("CO raise 2.5bb", "BB").freqs("AJs")
# -> {'fold': 0.0, 'call': 0.5, 'raise 14bb': 0.5, 'allin': 0.0}

# Ranges as 169 class weights (np.float64), in preflop.CLASSES order.
node = db.node("BTN raise 2.5bb", "BB")
node.range(node.action_index("call"))      # BB's calling range
preflop.range_percent(_)                   # 37.6 (% of the 1326 combos)

# Spots: both ranges, pot and stacks where the hand reaches the flop.
spot = db.spot("BTN raise 2.5bb, BB call")
spot.players, spot.pot_bb, spot.stack_bb   # ('BB', 'BTN'), 5.5, 97.5  (first = out of position)
spot.ranges["BTN"]                         # class weights
spot.combos("BB", dead=board_cards)        # 1326 combo weights, card removal applied
db.flop_spots()                            # all 38

# Hand classes and combos.
preflop.class_name(51, 47)                 # 'AKs' (cards are rank * 4 + suit)
preflop.COMBOS, preflop.COMBO_CLASS        # (1326, 2) card pairs and their class index
```

### Playing it in the arena

`db.locate(obs)` places a preflop observation in the tree. It maps seats to positions from the
button, then replays the hand's actions. It returns `None` when the tree doesn't apply: not
NLHE, not preflop, not exactly six occupied seats, or a line the tree doesn't model (a limp, a
cold call of a 3-bet). Otherwise it returns a `Decision`:

```python
import random
from pokerarena import BaseBot, preflop

DB = preflop.load()

class ChartBot(BaseBot):
    def __init__(self, seed=0):
        super().__init__(seed)
        self.rng = random.Random(seed)

    def act(self, obs):
        if obs.street == 0:
            d = DB.locate(obs)
            if d is not None:
                action = d.sample(obs, self.rng)   # None if this hand can't be here
                if action is not None:
                    return action
        ...                                        # postflop / off-tree logic
```

- `d.node` is the current decision. It is `None` for an implied fold, and then `d.sample`
  folds (or checks).
- `d.freqs(obs)` gives the frequencies for the bot's own hand.
- `d.position` is the bot's position, and `d.positions` maps every seat to its position.
- `d.approx` is `True` when an earlier raise size was off the tree and was mapped to the
  tree's size. The tree assumes 100bb stacks and no ante, and `locate` doesn't check either.

## File format (`schema: "pokerarena.preflop/1"`)

The file is one JSON object, with one node or spot per line so diffs stay readable.

| Key | Content |
|---|---|
| `game` | variant, players, stack, blinds and ante in bb, rake |
| `source` | name, game, version, url, license and the changes made |
| `positions` | preflop action order `UTG HJ CO BTN SB BB` |
| `postflop_order` | `SB BB UTG HJ CO BTN` |
| `classes` | the 169 class names; every array below is indexed in this order |
| `nodes[]` | `id`, `path` (each decision's action index, folds and implied folds included), `actor`, `line`, `raises`, `pot_bb`, `to_call_bb`, `actions[]` (`kind` fold/call/raise/allin, `label`, `to_bb` for raises), `children[]` (next node id or `null`), `reach[169]`, `strategy[action][169]` |
| `spots[]` | `line`, `node` and `action` (the closing call), `players` (postflop order), `pot_bb`, `stack_bb`, `allin`, `ranges{position: [169]}` |

- `reach[c]` is how often the actor's own earlier actions bring class `c` to this node.
- `strategy[a][c]` is the frequency of action `a`. Classes that can't reach the node have
  `reach` 0 and all-zero strategy rows.
- A spot's range is the product of the player's frequencies along the line.

Regenerate the file after changing the source or the converter with
`python3 ranges/build_preflop.py`. `--check` reports a stale file, and the test suite runs it.
