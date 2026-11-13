# River and turn solvers: technical reference

This is the reference for the solver stack behind the `range_solver` and `range_solver_turn` bots. It covers:

- the conventions and data types;
- the exact river solver;
- the training-spot generator and data pipeline;
- the river value network;
- the depth-limited turn solver, with its leaf evaluators and speed options;
- the pre-river (chance-averaged) network;
- the bots that use all of the above.

All paths are relative to the repository root.

```
preflop chart (ranges/, pokerarena.preflop)
        │  scenarios, 169-class weights
        ▼
range generator (river_net/ranges.py) ──► river spots ──► exact river solver (crates/solver, Rust)
        │                                                      │ v1, v2 per hand
        │                                                      ▼
        │                                         river data (data/v1, river_net/datagen.py)
        │                                                      │ buckets (K = 64)
        │                                                      ▼
        │                                         river value network  river_net/models/river_k64.pt
        │                                                      │ teacher (all 48 rivers)
        └──► turn-end spots ─────────────────────────────────► pre-river data (data/preriver_v1)
                                                               │ turn buckets (K = 256)
                                                               ▼
                                                  pre-river network   river_net/models/preriver_k256.pt
turn spot ──► turn solver (river_net/turn.py): DCFR on the turn tree, leaves valued by either network
bots: range_tag ─► range_solver (exact river) ─► range_solver_turn (turn solve + exact river)
```

---

## 1. Conventions and types

### Cards, combos and ranges

| Thing | Representation |
|---|---|
| Card | `u8` / `int` in `0..52`, `rank * 4 + suit`, with rank 0 = deuce … 12 = ace (`engine::Card`) |
| Board | `np.ndarray[uint8]` of 5 (river) or 4 (turn) cards; `river_net.cards.parse_board` also accepts strings like `"AsKd7h2c"` |
| Combo (private hand) | index `0..1326`, `hi * (hi - 1) / 2 + lo` for cards `hi > lo` (`solver::combo_index`, `pokerarena.preflop.COMBOS`, `river_net.cards.HAND_CARDS[1326, 2]`) |
| Range | `np.ndarray[float32 or float64]` of shape `[1326]`: non-negative, zero on combos blocked by the board, normalised to sum 1 where the API says so |
| 169-class weights | `np.ndarray[169]` of action frequencies in `[0, 1]` (preflop chart); `preflop.expand(w, dead)` maps them to 1326 combos |
| Blocked combo | shares a card with the board (or, on a river card loop, with that card) |

### Players and positions

- Solver "player 1" (index 0) is **out of position (OOP)** and acts first postflop; "player 2" (index 1) is in position.
- In bots, `players = [OOP seat, IP seat]`. OOP is the first live seat after the button.

### River value convention

A river spot is `(board[5], pot P, stack S, r1[1326], r2[1326])`. Amounts are in units of the river pot (`P = 1`); the solver only sees `S / P` (the SPR). For a deal `(h1, h2)` ending at a terminal:

```
u1 = (chips player 1 collects from the final pot − chips player 1 put in on the river) / P − ½
u2 = −u1
```

- **Fold** by player `f` after putting in `c_f` (pot units): `u_f = −½ − c_f`, and the other player gets `½ + c_f`.
- **Showdown** with equal contributions `c`: winner `½ + c`, loser `−½ − c`, tie `0`.

Per-hand quantities, with card compatibility `c(h, h') = 1` if the combos share no card:

| Symbol | Meaning |
|---|---|
| `m_i(h) = Σ_h' c(h,h') r_{−i}(h')` | compatible opponent mass |
| `ṽ_i(h)` | counterfactual value: the opponent-reach-weighted, unnormalised expected `u_i` |
| `v_i(h) = ṽ_i(h) / m_i(h)` | **conditional value**: the network target, defined for every valid hand (also outside the range) |
| zero-sum identity | `Σ r1·ṽ1 + Σ r2·ṽ2 = 0` |
| exploitability | `½ (Σ r1·BR̃1 + Σ r2·BR̃2)`, in pot units (× 100 = % of pot) |

### Turn value convention

A turn spot is `(board[4], P, S, r1, r2)` with `P` the turn pot. It uses the same formula with "turn and river" in place of "river":

```
u1 = (collected − put in on the turn and river) / P − ½
```

At a river leaf where each player put `c` turn pots in on the turn, the river pot is `1 + 2c` turn pots. River values convert as `u_turn = (1 + 2c) · u_river`, and the river SPR there is `(S − c) / (1 + 2c)`.

---

## 2. Betting abstraction and trees

The river and turn trees share one abstraction: `river_net.tree.DEFAULT_ABSTRACTION` (Rust: `solver::Abstraction::default()`).

| Key | Default | Meaning |
|---|---|---|
| `oop_bets`, `ip_bets` | `[0.5, 1.0]` | bet sizes as fractions of the current pot, per player |
| `raises` | `[1.0]` | pot-size raise: call amount + pot after calling |
| `allin` | `True` | all-in is always available |
| `max_bets` | `3` | bets + raises per street |
| `allin_threshold` | `1/3` | a size leaving less than ⅓ of the pre-action stack behind becomes all-in; duplicate sizes merge |

- **Build.** `Tree::river(stack, abstraction)` builds a flat breadth-first public tree: node kinds Decision / Fold / Showdown (Chance is reserved).
- **Node arrays:** `player`, `first_child`, `n_children`, `contrib[node][2]` (pot units), `edge` (labels such as `check`, `bet 0.5`, `raise 1`, `allin`, `call`, `fold`), and `slot` (the decision index per player).
- **Python view:** `river_net.tree.build_tree(spr, abstraction)` returns a `Tree` dataclass with `children(node)`, `path(node)` and `find(...)`.
- **Node counts** by SPR: 0.5 → 9, 1 → 21, 2 → 33, 4 → 45, 10 → 57 (63 for deep turn trees).

---

## 3. Exact river solver (`crates/solver`, `river_net/solver.py`)

### Algorithm

- **Vectorized public-tree DCFR** with (α, β, γ) = (1.5, 0, 2) and alternating updates. Positive regrets are discounted by `t^α/(t^α+1)` and negative ones by ½; the average strategy is weighted by `(t/(t+1))^γ`.
- **Average strategy** is weighted by the player's own sequence reach, with root weight 1 for every hand. Hands outside the range therefore still get a defined average strategy, and so a defined `v`.
- **Hands are sorted by strength**, so tie groups are contiguous.
- **Terminals cost O(n)** through a gather index (`TermIndex`): per-card prefix sums handle card removal, and sibling terminals share their prepared sums.
- **Exploitability** is checked every `check_every` iterations with a best-response pass. The solve stops below `target_expl` or at `max_iters`.
- **Final pass:** computes `ṽ`, `m` and `v` under the average strategies.

### Python API

```python
from river_net.solver import Spot, solve
spot = Spot(board: np.ndarray[uint8, 5], pot: float, stack: float,
            r1: np.ndarray[float32, 1326], r2: np.ndarray[float32, 1326], meta: dict = {})
res = solve(spot, iters=500, target_expl=0.002, check_every=50,
            abstraction=None, strategy=False)  # -> SolveResult
```

| `SolveResult` field | Type | Meaning |
|---|---|---|
| `v1, v2` | `float32[1326]` | conditional values (0 on blocked combos) |
| `m1, m2` | `float32[1326]` | compatible opponent masses |
| `cfv1, cfv2` | `float32[1326]` | counterfactual values `m · v` |
| `expl` | `float` | final exploitability (pot units) |
| `iters` | `int` | iterations run |
| `history` | `list[(iter, expl)]` | exploitability checks |
| `tree` | `Tree` | only when `strategy=True` |
| `strategy1, strategy2` | `float32[slots, max_actions, 1326]` | average strategies; `strategy_at(node)` returns `[n_actions, 1326]` |

- **Abstract games:** `solve_abstract(strength, r1, r2, stack, cards=None, ...)` (tests, toy games).

### Native bindings (`pokerarena._native`)

- `solver_river_solve`, `solver_abstract_solve`, `solver_build_tree` and `solver_river_strengths(board) -> uint16[1326]`.
- `solver_range_equity(board[3|4], opp[1326], samples, seed) -> float32[1326]`: Monte Carlo on the flop, exact on the turn.
- `solver_river_equity(board[5], opp) -> (equity, mass)`.

### Performance

- **Single core, 500 iterations:** median 46 ms, p90 125 ms.
- **To reach 0.2% of the pot:** median 100 iterations, p90 200.

---

## 4. Training spots: the range generator (`river_net/ranges.py`)

`RangeGenerator(seed, GeneratorConfig()).generate(index, source=None, street=...) -> Spot` is deterministic in `(seed, index, street)`.

| `street` | Board | Lines played | Seed words | Used for |
|---|---|---|---|---|
| `"river"` | 5 cards | flop + turn | `[seed, index]` | river solver data, river network |
| `"turn"` | 4 cards | flop | `[seed, index, 4]` | turn spots (benchmarks, tests) |
| `"turn_end"` | 4 cards | flop + turn | `[seed, index, 44]` | pre-river network data |

### Preflop scenarios

Scenarios come from the chart's 38 heads-up flop spots: 8 single-raised, 15 3-bet and 15 4-bet pots. There are no limped pots and stacks are 100bb.

A scenario is drawn by the following procedure:
1. The opener is uniform over UTG, HJ, CO, BTN and SB; earlier seats fold.
2. Each later seat calls, 3-bets or folds with its chart range's combo-weighted frequencies, and so on through 4-bets.
3. Lines that end preflop are redrawn.

Resulting probabilities: single-raised pots 0.795, 3-bet pots 0.164, 4-bet pots 0.041.

### Sources (mix `A 0.7 / B 0.2 / C 0.1`)

**A. Simulated lines**
1. Take the scenario's chart ranges and apply `jitter_classes`:
   - log-normal noise with σ ~ U(0.05, 0.25);
   - a power transform with γ ~ e^U(ln 0.7, ln 1.4);
   - half the time, some weight on empty classes next to the range.
2. Draw a random board.
3. For each street, draw a line from `cc 0.30, bc 0.25, cbc 0.20, brc 0.10, crc 0.15`. Lines the stack can't afford are dropped. The bet size is drawn from `{0.33, 0.5, 0.75, 1}·pot`, and raises go to 3× the bet.
4. Each action multiplies the actor's range by a soft filter on its equity percentile `q` against the opponent's current range (`percentiles`, weighted mid-rank). The filters use `σ(x) = 1/(1+e^−x)` with temperature `T ~ U(0.03, 0.10)`:

| Action | Filter | Parameter ranges |
|---|---|---|
| bet | `σ((q−t_hi)/T) + β·σ((t_lo−q)/T)` | t_hi U(0.45, 0.80), t_lo U(0.05, 0.30), β U(0.15, 0.8) |
| raise | same | t_hi U(0.70, 0.92), t_lo U(0.03, 0.20) |
| call | `σ((q−t_c)/T) · (1 − ρ·σ((q−t_top)/T))` | t_c U(0.15, 0.55); the second factor applies only when raising was possible: ρ U(0.3, 0.8), t_top U(0.85, 0.97) |
| check | `1 − κ·σ((q−t_cap)/T)` | κ U(0.5, 0.95), t_cap U(0.6, 0.85) |

Equity comes from `solver_range_equity`: 32 seeded Monte Carlo runouts on the flop, exact over the rivers on the turn.

**B. DeepStack recursive split.** A random board and scenario money path. Each player's mass is split recursively at uniform fractions between the lower and upper halves of the strength-sorted tie groups. Half the time it is mixed 50/50 with uniform. Strength is river strength, or turn equity against a uniform range on 4-card boards.

**C. Noisy baselines.** The scenario's chart ranges, filtered by the board only, times Gamma(α, 1) noise with α ~ U(0.5, 2) per combo (Dirichlet-like).

**Rejection** (the spot is redrawn, up to 50 attempts):
- either range has fewer than 20 effective combos (`1/Σr²`);
- or the SPR is outside `[0.1, 20]`.

---

## 5. River data pipeline

### Generation

`python -m river_net.datagen --n 100000 --workers 12 --out data/v1 --seed 0`

| Option | Default |
|---|---|
| `--shard-size` | 2000 |
| `--max-iters` | 500 |
| `--target-expl` | 0.002 |
| `--check-every` | 50 |
| chunk per worker task | 50 samples |

- Each sample is `generate(i)` followed by `solve(...)`.
- Throughput is about 800k samples/hour on 12 cores.
- The shipped `data/v1` (100k samples, 3.0 GB) has exploitability median 0.138% and p95 0.193% of the pot.

### Shard format

`data/<name>/shard_00000.npz` holds samples `[k·shard_size, (k+1)·shard_size)`, sorted by index.

| Array | dtype | Shape | Content |
|---|---|---|---|
| `index` | int64 | `[N]` | sample index (= generator index) |
| `board` | uint8 | `[N, 5]` | board (`[N, 4]` for pre-river data) |
| `pot`, `stack` | float32 | `[N]` | big blinds |
| `spr` | float32 | `[N]` | `stack / pot` |
| `r1, r2` | float32 | `[N, 1326]` | normalised ranges (OOP, IP) |
| `v1, v2` | float32 | `[N, 1326]` | conditional values (pot units) |
| `m1, m2` | float32 | `[N, 1326]` | compatible opponent masses |
| `expl` | float32 | `[N]` | solve exploitability (pot units; 0 for pre-river data) |
| `iters` | int16 | `[N]` | iterations (0 for pre-river data) |
| `source` | str | `[N]` | `A`, `B` or `C` |
| `scenario` | str | `[N]` | preflop line, e.g. `CO-open/BTN-call` |
| `line` | str | `[N]` | postflop line, e.g. `bc0.5/cc` |
| `seconds` | float32 | `[N]` | generation time |

### Manifest and resume

`manifest.json` holds `{config, shards: [{shard, file, start, count, expl_median, seconds}], n}`.
- A rerun with the same config resumes: complete shards are skipped, and a partial last shard is regenerated.
- A different config is refused.

### Split and bucketing

- **Board split** (`river_net/dataset.py`): the sorted board is hashed with blake2b into `0..99`. Below 90 is train, below 95 validation, the rest test. Test boards are never trained on.
- **River buckets** (`river_net/buckets.py`):
  - The valid hands of a board are sorted by river strength and cut into `K` equal-combo buckets.
  - A tie group is never split; it goes to the bucket of its midpoint.
  - The bucket map is `int16[1326]`, with −1 for blocked combos.
- **Bucketed cache:** `bucketed(data_dir, K)` writes `<data>/bucketed_K<K>.npz` with:
  - `rb1, rb2`: bucket range masses;
  - `vb1, vb2`: `m`-weighted bucket means of `v`, with a plain mean where all weights are 0;
  - `w1, w2`: `Σ_{h∈k} r·m`;
  - `nonempty`: `[N, K]` bool;
  - `split`, plus the metadata columns.

---

## 6. River value network (`river_net/model.py`, `river_net/train.py`)

### Architecture

`RiverNet(K, hidden, depth, board_features=False)`:

```
x = [r̄1 (K), r̄2 (K), SPR, log(1+SPR)]                       float32[2K+2]
body: depth × (Linear(hidden) → LayerNorm → GELU)
head: Linear(2K) → [v̂1 (K), v̂2 (K)]                          pot units, conditional
zero-sum projection with w = [w1, w2], w_i(k) = Σ_{h∈k} r_i(h) m_i(h):
    δ = Σ w·ŷ / Σ w ;  y = ŷ − δ
```

### Shipped model

`river_net/models/river_k64.pt`, configuration in `river_k64.config.json`:

| Setting | Value |
|---|---|
| K | 64 |
| Layers | 3 × 512 |
| Training samples | 90k |
| Epochs | 100 |
| Training time | 86 s |

### Training

| Setting | Value |
|---|---|
| Loss | Huber (δ = 0.1) per bucket, weighted by `λ + (1−λ)·K·r̄` with λ = 0.5; empty buckets weight 0 |
| Optimizer | AdamW, lr 1e-3, cosine decay to 0, weight decay 1e-4 |
| Batch | 512 |
| Early stopping | patience 6, on validation loss |

### Test-board results

Full report: `results/river_net_poc.md`.

| | MAE unweighted | MAE range-weighted | Root value error |
|---|---|---|---|
| network | 0.0350 | 0.0352 | 0.0074 |
| bucketing floor (K = 64) | 0.0126 | 0.0208 | 0.005 |
| check-down (equity − ½) | 0.1564 | 0.1288 | 0.050 |

- Inference runs at about 383k spots/s at batch 1024 on one thread.

### Interface

```python
net = RiverValueNet.load("river_net/models/river_k64.pt")
v1, v2 = net.predict(board[5], pot, stack, r1, r2)          # float64[1326] each
V1, V2 = net.predict_batch(boards[B,5], sprs[B], R1[B,1326], R2[B,1326])
y = net.predict_buckets(x[B, 2K+2], w[B, 2K])                # float32[B, 2K]
```

---

## 7. Turn solver (`river_net/turn.py`)

### Game

- The turn tree is the river abstraction built at the turn SPR.
- The **same DCFR** is used, with **simultaneous** updates: both players' regrets are updated from the same iteration's values, so each iteration needs one leaf query.

Terminals:

| Terminal | Value for player `p` (counterfactual, turn-pot units) |
|---|---|
| Fold by `f` | `u · m_p(h)` with `u = −½ − c_f` (folder) or `½ + c_f` |
| All-in (both all-in on the turn) | `(½ + S)/44 · Σ_c net_c(h)`: exact showdown over the river cards, `net_c` = weaker − stronger compatible opponent reach on `board + c`; each deal sees 44 river cards |
| River leaf (betting closed with chips behind) | `(1 + 2c)/44 · Σ_c m_c(h) · v_c(h)`: the leaf model's conditional river values, weighted by opponent mass compatible with `h` and `c` |

### Exploitability

- `exploitability(strat=None)` returns `½ (Σ r1 BR̃1 + Σ r2 BR̃2)` in turn-pot units. It is computed in the solver's own leaf model, the standard depth-limited measure.
- `river_net.turn_check` and `river_net.turn_bench` also score a turn strategy **with exact river play**: every leaf × river card is re-solved with the exact river solver at the ranges the strategy induces, and those values are held fixed (`FixedLeaf`). This is the measure used to compare leaf models and speed options.

### Leaf evaluators

All share `bind(boards: TurnBoards, stack)` and `values(R1[L,1326], R2[L,1326], c[L]) -> (cfv1[L,1326], cfv2[L,1326])`. The network leaves also expose `query` and `expand`, used for caching.

| Class | What it does | Cost per iteration |
|---|---|---|
| `NetworkLeaf(RiverValueNet)` | river network on every leaf × river card (48 rows per leaf). Bucketing, masses and broadcast run in the native `LeafIndex` kernels | L × 48 network rows + O(L·48·1326) kernels |
| `PreRiverLeaf(PreRiverNet)` | pre-river network, one row per leaf: `cfv_p = (1+2c) · m_p(h) · v̄_p(h)` | L network rows |
| `CheckdownLeaf()` | showdown without river betting (exact, zero-sum; tests and baseline) | L showdown sums |
| `FixedLeaf(V)` | fixed per-card conditional values `V[p][L, 48, 1326]` (exact re-solves) | — |

- **Empty reach.** A leaf a player never reaches is bucketed with that player's root range. Its counterfactual values are 0 anyway, because their masses use the true reach.

### `TurnSolver` API

```python
ts = TurnSolver(spot, leaf, abstraction=None, alpha=1.5, beta=0.0, gamma=2.0,
                refresh=None, refresh_warmup=10, river_samples=None, seed=0)
res = ts.solve(iters=300, target_expl=0.005, check_every=25)   # -> TurnResult
ts.set_leaf(other_leaf); ts.solve(iters=50, ...)                # continue (warm start)
R1, R2 = ts.leaf_ranges()                                       # [L, 1326] reach at leaves
ts.exploitability(strat=None)                                   # own leaf model
```

| Option | Effect |
|---|---|
| `refresh=k` | recompute leaf and all-in values only every `k` iterations, after `refresh_warmup` iterations that refresh every time. In between, cached *conditional* values are rescaled by the current opponent mass |
| `river_samples=m` | value the network leaves on `m` random river cards per iteration, scaled by 48/m (unbiased chance sampling). All-in nodes stay exact; checks and final values use all 48 cards |
| `check_every=0` | no exploitability checks (`expl` is NaN): the play setting |
| repeated `solve` / `set_leaf` | continues the same regrets and averages; DCFR weights continue from the iteration count |

`TurnResult` has the same fields as `SolveResult`, plus `seconds` and `leaf_queries` (network rows evaluated). Its strategies are `float64[slots, max_actions, 1326]`.

### Native helpers (`pokerarena._native`)

**`TurnBoards(board[4])`**

| Member | Returns |
|---|---|
| `.cards` | the 48 river cards (the order of every per-card output) |
| `.strengths()` | `uint16[48·1326]` |
| `.masses(q[L·1326])` | `float32[L·48·1326]` |
| `.showdown_net_sum(q, cards=None)` | `float32[1326]` |
| `.showdown_matrix()` | `float32[1326·1326]` matrix `W` with `showdown_net_sum(q) = W @ q`. It values all all-in nodes for both players in one product; about 35 ms to build per board |

**`LeafIndex(cards, bmap[48·1326] int32, K)`**

| Method | Returns |
|---|---|
| `.inputs(r1, r2, q1, q2, cards=None)` | `(x, w)`, each `float32[L·cards·2K]`: per-card normalised bucket ranges and projection weights. `r` gives bucket ranges, `q` the reach the masses come from |
| `.cfv(y, r1, r2, scale[L], cards=None)` | `(cfv1, cfv2)`, each `float64[L·1326]` |

- **Masses** use inclusion–exclusion over card sums, O(1) per (card, hand): `m(c,h) = T − S[a] − S[b] − S[c] + q(ab) + q(ac) + q(bc)`.

---

## 8. Pre-river network (`river_net/preriver.py`)

This is DeepStack's auxiliary-network trick: one network row per turn leaf instead of 48.

### Target

For a turn-end spot `(board[4], P = river pot, S, r1, r2)`:

```
v̄_p(h) = Σ_c m_c(h) · f_c(h)  /  Σ_c m_c(h),        Σ_c m_c(h) = 44 · m(h)
```

- `f_c` is the river network's conditional value on `board + c`.
- `m_c(h)` is the opponent mass compatible with `h` and `c`.
- `m(h)` is the turn-board compatible mass.

In other words, the target is the river value averaged over the river cards `h` survives, weighted by how much opponent range remains. It is 0 where `m(h) = 0`. The turn solver then uses `cfv = (1 + 2c) · m · v̄`, which matches `NetworkLeaf` exactly when `g` matches its teacher.

### Data

- **Command:** `python -m river_net.datagen --kind preriver --teacher river_net/models/river_k64.pt --n 500000 --out data/preriver_v1`.
- **Spots:** `generate(i, street="turn_end")`, labelled by `Teacher` (river network over all 48 rivers, native kernels). No solves are involved.
- **Throughput:** about 1.9M samples/hour on 12 cores, 22 ms per sample. 500k samples take 15 GB.
- **Format:** same shard layout as river data, with `board uint8[N, 4]`, `spr` = river SPR, `m` = turn-board masses, and `expl = iters = 0`.
- **Teacher check:** targets are zero-sum to 6e-8.

### Turn buckets

`turn_bucket_map(board, K, scheme)` uses per-hand features against a uniform random opponent:
- `EHS = mean_c HS_c(h)`;
- `E[HS²]`;
- where `HS_c` is the river equity vs uniform on `board + c`.

| Scheme | Definition |
|---|---|
| `ehs` | K equal-combo buckets of EHS |
| `ehs2:<K2>` | potential-aware: K/K2 equal-combo EHS groups, each split into K2 equal-combo sub-buckets by E[HS²]. Identical features never straddle a boundary |

Bucketing floor (range-weighted / unweighted MAE vs per-hand targets, 600 spots):

| Scheme, K | Floor |
|---|---|
| `ehs`, 64 | 0.0381 / 0.0266 |
| `ehs`, 128 | 0.0239 / 0.0157 |
| `ehs`, 256 | 0.0106 / 0.0064 |
| `ehs2:4`, 64 | 0.0324 / 0.0200 |
| `ehs2:8`, 128 | 0.0186 / 0.0107 |
| **`ehs2:8`, 256 (shipped)** | **0.0083 / 0.0046** |
| `ehs2:16`, 256 | 0.0082 / 0.0042 |

### Model and training

| Setting | Value |
|---|---|
| Class | the `RiverNet` class with turn buckets: K = 256, input `2K + 2 = 514`, 3 × 512 GELU/LayerNorm, output `2K` plus zero-sum projection |
| Data | 450k train / 25k validation / 25k test samples (board split) |
| Recipe | same loss and optimizer as the river network |
| Command | `python -m river_net.train --data data/preriver_v1 --K 256 --scheme ehs2:8 --hidden 512 --epochs 60 --out runs/preriver_k256 --save-model river_net/models/preriver_k256.pt` |
| Result | 60 epochs (still improving slowly), 261 s, best validation loss 0.534 |

- **File:** `river_net/models/preriver_k256.pt`; its training config, including the scheme, is stored in the checkpoint.

### Evaluation

`python -m river_net.preriver --n 3000`: per-hand error against the teacher on 3,000 test-board turn-end spots, river-pot units.

| | MAE range-weighted | MAE unweighted |
|---|---|---|
| pre-river network | 0.0363 | 0.0379 |
| bucketing floor | 0.0080 | 0.0044 |
| check-down (turn equity − ½) | 0.0924 | 0.1241 |

- **By source** (range-weighted): A 0.036, B 0.042, C 0.028.
- **Error stacking:** this error adds to the river network's own error against exact solves (0.035). The turn benchmark below measures the combined effect on strategy quality.

### Interface

```python
pre = PreRiverNet.load("river_net/models/preriver_k256.pt")
v1, v2 = pre.predict(board[4], pot, stack, r1, r2)           # chance-averaged, float[1326]
leaf = PreRiverLeaf(pre)                                     # for TurnSolver
```

---

## 9. Turn solver speed

### Where the time went

Profile of a 200-iteration solve on a 33-node, 5-leaf spot, before the speed work: **2.34 s** in total.

| Part | Time |
|---|---|
| dense one-hot bucketing / broadcast around the network (numpy matmuls) | 1.16 s |
| exact all-in showdown sums (6 all-in nodes × 2 players per iteration) | 0.46 s |
| per-card masses | 0.26 s |
| network forward passes | 0.16 s |
| CFR bookkeeping | about 0.3 s |

The network forward was only about 7%; the gather/scatter around it dominated.

### Exact speedups (no change in results)

1. **Native leaf kernels** (`LeafIndex`) replace the dense one-hot matmuls and the separate mass computation. They match the dense reference to 1e-8.
2. **All-in nodes** are valued in one product with the board's showdown matrix instead of per-river sweeps: 0.13 ms vs 4.4 ms for 20 evaluations.
3. **Fold-node masses** are computed in one batched product per backward pass instead of one per fold node.

Together, these take 200-iteration full-card solves (4 checks) of the first four benchmark spots from 3.27 / 2.31 / 2.32 / 3.24 s to 1.25 / 0.67 / 0.65 / 1.26 s, with the same exploitability up to float rounding (0.0029 / 0.0010 / 0.0106→0.0103 / 0.0030). Everything else below is a speed/accuracy trade measured on the benchmark.

### Benchmark

`python -m river_net.turn_bench` runs a fixed set of 20 turn spots: `RangeGenerator(7)`, indices 0–19, `street="turn"`, the same spots as `turn_check`.

- **Timing:** single-threaded torch, with no exploitability checks inside the timed solve.
- **Scoring:** **expl@exact**, the exploitability with every river leaf × card re-solved exactly at the induced ranges, in % of the turn pot. This is the quality that matters. "own" is the exploitability in the configuration's own leaf model.
- **Config names:**
  - `<leaf>-<iters>`: `net` is the river network on every card, `pre` the pre-river network, `cd` check-down;
  - `-s<m>`: m sampled river cards per iteration;
  - `-r<k>[-w<n>]`: refresh leaf values every k iterations after n warm-up iterations;
  - `a+b`: run `a`, then continue the same solve with `b` (warm start).
- **Results:** `results/turn_speed/bench.json` (round 2, final code) and `bench_round1.json`.

**Round 2** (final code: native kernels, showdown matrix, batched fold masses; the sampled `-s` configurations were re-run after all-in nodes became always-exact, see the `note` in `bench.json`):

| Config | Time median | Time mean | expl@exact median | expl@exact mean | Own median |
|---|---|---|---|---|---|
| cd-300 (check-down baseline) | 1.464 s | 1.353 s | 17.74% | 20.53% | 0.49% |
| net-300 (all cards, 300 iterations) | 1.573 s | 1.470 s | 2.09% | 2.24% | 0.26% |
| net-100 | 0.560 s | 0.522 s | 1.71% | 1.80% | 1.19% |
| net-200-s8 | 0.469 s | 0.436 s | 1.49% | 1.69% | 0.65% |
| **net-150-s8 (bot default)** | **0.363 s** | **0.338 s** | **1.44%** | **1.59%** | 0.79% |
| net-100-s8 | 0.255 s | 0.238 s | 1.73% | 1.84% | 1.24% |
| net-100-s4 | 0.230 s | 0.211 s | 2.02% | 2.10% | 1.80% |
| net-60-s8 | 0.169 s | 0.160 s | 3.00% | 3.30% | 2.63% |
| pre-100 | 0.227 s | 0.212 s | 3.54% | 3.52% | 1.22% |
| pre-100+net-50 | 0.501 s | 0.457 s | 1.98% | 2.13% | 1.24% |
| pre-60+net-40-s8 | 0.249 s | 0.234 s | 2.11% | 2.22% | 1.69% |
| pre-100+net-50-s8 | 0.343 s | 0.318 s | 1.99% | 2.01% | 1.38% |
| pre-50+net-30-s8 | 0.206 s | 0.194 s | 2.60% | 2.81% | 2.23% |

**Round 1** (before batching fold masses, so the times are about 5% higher; same quality):

| Config | Time median | expl@exact median | expl@exact mean |
|---|---|---|---|
| net-200 | 1.127 s | 1.67% | 1.89% |
| net-200-r2 (refresh every 2) | 0.923 s | 1.65% | 1.72% |
| net-200-r5-w100 (100 fresh, then every 5) | 0.961 s | 5.17% | 8.35% |
| net-200-s16 (all-ins also sampled) | 0.836 s | 1.50% | 1.64% |
| net-300-s16 (all-ins also sampled) | 1.234 s | 1.61% | 1.87% |
| pre-300 | 0.678 s | 4.16% | 4.40% |
| pre-200 | 0.468 s | 3.68% | 3.79% |
| pre-200+net-30 | 0.633 s | 2.87% | 2.90% |

Before any speed work, the bot's solve (`net-200` with checks, old code) took about 2.2 s single-threaded (2.3 s profiled).

**Findings**

1. **Leaf error sets the quality floor, not iterations.** With the river network, expl@exact stays around 1.6–2.1% from 100 to 300 iterations, while own-model exploitability keeps falling (1.19% → 0.26%). Iterations past about 100–150 buy nothing measurable.
2. **River sampling of the network leaves is the best trade.** Eight of 48 cards per iteration is unbiased, and the noise averages out: `net-150-s8` beats every full-card configuration at 0.36 s, and `net-100-s8` matches the full-card `net-100` in under half the time. Four cards no longer saves time (fixed costs dominate) and is slightly worse. All-in nodes are not sampled: the exact showdown-matrix product is cheaper than 8 per-river sweeps. Sampling them as well, as the first sampled runs did, cost about 0.07 s per solve and some accuracy.
3. **Stale leaf values hurt.** The current strategy of simultaneous-update DCFR moves too much between iterations for cached conditional values. Refreshing every 2 iterations is neutral; every 5 is 3× worse, and saves little since the kernels are cheap.
4. **The pre-river network on its own is fast but not good enough here.** It is 2.5× faster than `net-100`, but at 3.5–4.2% it is twice as exploitable. Its 0.036 error against the teacher stacks on the river network's own error. As a warm start (`pre-100+net-50`, `pre-60+net-40-s8`) it recovers about 2%, but `net-100-s8` does better (1.73%) in about the same time (0.26 s). With the native kernels, the 48-card leaf query stopped being the bottleneck the pre-river network was meant to remove.
5. **Warm start in general:** solves are resumable (`set_leaf` + `solve`), so a warm start from an earlier solve of the same tree is supported. The bot solves the turn once per hand, so it has no "previous decision" solve to warm-start from yet.

**Remaining cost** of a `net-150-s8` solve (cProfile, 20 spots, before the all-in change, with sampled sweeps removed from the total):
- leaf kernels (`LeafIndex.inputs` / `cfv`): about 25%;
- CFR bookkeeping in Python (backward pass, regret matching, reaches, solve loop, fold masses): about 45%;
- network forward: about 12%;
- showdown-matrix build: about 30 ms per solve (9%).

The next step would be moving the turn CFR loop into Rust, as was done for the river.

---

## 10. Bots

All three bots are unanchored, live in `bots/NLH/`, and play preflop from the chart.

### `range_tag.py`

- **Preflop:** the 6-max 100bb chart via `pokerarena.preflop` (Leakproof ranges, CC BY 4.0).
- **Off-chart lines** are answered as the last aggressive action. Unopened pots with limpers are played as an open, one raise as facing an open, two as facing a 3-bet, and so on, with cold callers ignored.
- **Postflop:** the anchored TAG bot's Monte Carlo equity logic.

### `range_solver.py`

`range_tag`, plus a heads-up river solve.

**River range estimation** (`_river_ranges(obs, players, board, streets)`):
1. Take the preflop chart's 169-class weights along the hand's preflop line (`preflop_ranges`). Use uniform weights if the hand left the chart.
2. Expand them to combos without board cards.
3. Apply each flop and turn action (`streets`) as a deterministic soft filter on the actor's range, by equity percentile against the other range. Equity uses `solver_range_equity`: 32 Monte Carlo samples on the flop, exact on the turn.

| Action | Filter (T = 0.06) |
|---|---|
| check | `1 − 0.7·σ((q−0.72)/T)` |
| bet | `σ((q−0.62)/T) + 0.45·σ((0.17−q)/T)` |
| raise | `σ((q−0.81)/T) + 0.45·σ((0.11−q)/T)` |
| call | `σ((q−0.35)/T)`, times `(1 − 0.5·σ((q−0.91)/T))` when facing a bet |

**River play:**
- **Solve:** once per hand at the bot's first river decision, with `solve(spot, iters=500, target_expl=0.002, strategy=True)`. The pot is the pot at the start of the river; the stack is the smaller stack behind.
- **Navigation:** the river actions so far are followed down the tree. Fold, check and call match by name; sized actions go to the tree action with the nearest contribution, or to all-in.
- **Action:** sampled from the average strategy of the bot's own combo, then converted back to chips (`clamp_raise`).
- **Fallback:** anything multiway, off-tree or erroring falls back to the TAG logic.

### `range_solver_turn.py`

`range_solver`, plus a heads-up turn solve.

- **Turn ranges:** the same estimate, filtered by the flop actions only (`streets=((1, 3),)`).
- **Turn solve:** once per hand at the bot's first turn decision with `TurnSolver(spot, leaf)`. The bot plays its combo's average strategy at the current node, mapping opponent sizes the same way.
- **River:** the exact river solve of `range_solver`. When the turn was solved and the turn line stayed on the tree, the river ranges are the turn solve's root ranges multiplied by both players' turn strategies along the line played (`_ranges_after_turn`), restricted to the river card. Otherwise the soft-filter estimate is used.
- **Parameters:** `TURN_PLAN = (("net", 150, 8),)` (river network leaves, 150 DCFR iterations, 8 sampled river cards per iteration, no exploitability checks) and `TORCH_THREADS = 1`.

  - **Measured cost in the arena:** median 0.43 s (p90 0.46 s) per turn solve, including range estimation and setup, against 2.24 s before the speed work.
  - **Plan stages:** the plan is a tuple of stages `(leaf, iterations, samples)` run on one solver. For example, `(("net", 100, 8),)` is faster (0.26 s, 1.73% on the benchmark), and `(("pre", 60, None), ("net", 40, 8))` is a pre-river warm start (0.25 s, 2.11%).

**Arena** (heads-up, duplicate, 4,800 hands per bot, before the speed work):

| Bot | bb/100 |
|---|---|
| `range_solver_turn` | +15.7 ± 18.8 |
| `range_solver` | −11.8 ± 19.0 |
| tag | −3.9 ± 13.9 |

---

## 11. Commands

```
# build and tests
cargo test -p solver && cd python && maturin develop --release && cd ..
python/.venv/bin/python -m pytest river_net/tests

# river: data -> network -> report
python -m river_net.datagen --n 100000 --workers 12 --out data/v1
python -m river_net.train --data data/v1 --K 64 --hidden 512 --epochs 100 --out runs/k64 \
    --save-model river_net/models/river_k64.pt
python -m river_net.evaluate --data data/v1 --model river_net/models/river_k64.pt

# pre-river: data -> network -> evaluation
python -m river_net.datagen --kind preriver --n 500000 --out data/preriver_v1
python -m river_net.train --data data/preriver_v1 --K 256 --scheme ehs2:8 --hidden 512 \
    --epochs 60 --out runs/preriver_k256 --save-model river_net/models/preriver_k256.pt
python -m river_net.preriver --n 3000

# turn: quality check and speed benchmark (exact river re-solves)
python -m river_net.turn_check --spots 20
python -m river_net.turn_bench                 # writes results/turn_speed/bench.json
```

Run Python commands from the repository root with `PYTHONPATH=.`, or use the venv with the package installed.

---

## 12. Limitations

- **No exact turn baseline.** Every turn number is relative to exact *river* play at fixed ranges. There is no full turn + river exact solve.
- **Approximations stack.** Pre-river leaves are an approximation of an approximation (pre-river network ≈ river network ≈ exact). Their errors add.
- **Range estimates in the bots are heuristic.** They come from the chart and fixed soft filters, not from an opponent model.
- **Bet abstraction.** A single abstraction is used (½ pot, pot, all-in; pot raise). Opponent sizes are mapped to the nearest tree action, with no re-solving after off-tree actions.
- **Training distribution.** Both networks are trained on generator ranges, not on the ranges turn solves actually produce at their leaves (network leaf error rises from 0.035 to 0.040 there).
