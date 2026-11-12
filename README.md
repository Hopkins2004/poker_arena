# Poker Arena

A high-throughput engine for running poker bots against each other: 6-max No-Limit Hold'em
first, plus Pot-Limit Omaha and Fixed-Limit Hold'em. Tables run in parallel, matchmaking is
configurable, and results feed a pairwise multiplayer Elo leaderboard. It is an evaluation
harness for poker AI research. The priorities are rules and chip-accounting correctness
first, speed second.

```
crates/engine   cards, deck, hand evaluator, game rules, hand state machine (no I/O, no threads)
crates/arena    Bot trait, table runner, stats, matchmaking, ratings, leaderboard, parallel runner, config
crates/bots     baseline bots: random-legal, calling station, tight-aggressive; MC equity helper
crates/cli      the `arena` binary: run, bench, leaderboard
python/         PyO3 bindings (crate arena_py) + Python package `pokerarena` + example bots
configs/        example TOML configs
```

## Build, test, run

```sh
cargo build --release
cargo test --workspace
cargo test --workspace --release -- --ignored     # exhaustive C(52,7) evaluator enumeration (~1 s)
cargo bench -p engine                             # evaluator + hand throughput (criterion)
cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check

cargo run --release -p cli -- run --config configs/6max_nlh.toml
cargo run --release -p cli -- bench               # hands/s and evals/s at 1 thread and all threads
cargo run --release -p cli -- leaderboard

cd python && python3 -m venv .venv && .venv/bin/pip install maturin numpy pytest
.venv/bin/maturin develop --release && .venv/bin/python -m pytest
```

`arena run --config <file> [--seed N] [--threads N] [--rounds N] [--hands N] [--fresh]
[--quiet] [--leaderboard F] [--results-dir D]` prints the sorted leaderboard and writes
`results/<run_id>/`, which holds `config.toml`, `schedule.json`, `tables.jsonl`,
`leaderboard.json`, `leaderboard.txt` and, if enabled, `hand_histories/`. It also updates
`leaderboard.json`. Each run starts from the stored ratings unless you pass `--fresh`.

## Performance

Apple M4 Pro (8 performance + 4 efficiency cores), release build:

| | 1 thread | 12 threads |
|---|---|---|
| 7-card evaluator, random hands | 480 M evals/s | 4.2 G evals/s (8.7×) |
| 6-max NLHE, random-legal bots, full runner | 1.14 M hands/s | 8.4 M hands/s (7.3×; 6.4× / 80 % on the 8 P-cores) |
| 6-max PLO, random-legal bots | 0.95 M hands/s | 7.1 M hands/s |
| 6-max NLHE with hand histories on | 0.32 M hands/s | 2.2 M hands/s |

The bare engine loop (no `dyn Bot`, no event broadcast) runs 1.9 M 6-max hands/s on one core.

**Rust vs Python bots.** `python/examples/baselines.py` holds line-by-line Python ports of
`tag`, `random` and `calling_station`. The same three-bot arena (`configs/trio_rust.toml`
vs `configs/trio_py.toml`: 6-max, 100k hands per table, 5 rounds, 1.5 M hands, 3 tables per
round) took 6.2 s with Rust bots and 21.3 s with the Python ports, about 3.5× in both wall
and CPU time. Results agree within their CIs. A Python decision costs about 1.1 µs for a
trivial bot, against 0.01 µs in Rust. The tight-aggressive bot costs 5.5 µs against 2.6 µs,
since its equity sampling runs in Rust in both cases.

## Writing a bot in Rust

Implement `arena::Bot`. Only `act` is required:

```rust
use arena::bot::{Action, Bot, Observation, PublicEvent};

pub struct MinRaiser { name: String }

impl Bot for MinRaiser {
    fn name(&self) -> &str { &self.name }
    fn act(&mut self, obs: &Observation) -> Action {
        let la = obs.legal;                       // precomputed: never re-derive the rules
        if la.can_raise { Action::RaiseTo(la.min_raise_to) }
        else if la.can_check { Action::Check } else { Action::Call }
    }
    // Optional: new_match(&MatchInfo), new_hand(&HandStart), observe(&PublicEvent),
    // end_hand(&HandResult)
}
```

Hand it to the arena as a `BotFactory`, i.e. `Arc<dyn Fn(u64 /*seed*/) -> Box<dyn Bot> + Send + Sync>`.
Every table seat gets a fresh instance, seeded deterministically, so seed any randomness
from that seed. Read-only shared data such as a strategy table belongs in an `Arc` captured
by the factory. To use the bot from configs, add a kind to `bots::factory`
(`crates/bots/src/lib.rs`). Its params arrive as JSON (`serde_json::Value`).

What a bot sees:

- **`Observation`** is built by the engine for the acting seat only. It contains `seat`, `n_seats`,
  `button`, `format`, `street`, `hole` (own cards only), `board`, and per-seat `stacks`,
  `street_contrib`, `total_contrib` and `status`. It also has `pot`, `current_bet`,
  `sb`/`bb`/`ante`, `legal` and `history` (every public action, blinds and antes included).
  Other seats' hole cards never appear. They are revealed only through
  `PublicEvent::Showdown`, and only for players who reach showdown.
- **`Action`** is `Fold | Check | Call | BetTo(x) | RaiseTo(x) | AllIn`. Amounts are the
  *total street contribution* (raise-to). `BetTo` and `RaiseTo` are interchangeable. `Call`
  with nothing to call counts as a check. `AllIn` is legal when committing the whole stack is
  a call or an allowed raise. In pot-limit and fixed-limit, a deep stack therefore can't
  always shove.
- **Illegal actions** are replaced by check or fold and counted in stats under the default
  `illegal_action_policy = "fold_or_check"`. They abort the table under `"error"`.
- **Duplicate mode**: bots must not carry hand-specific state across hands. They should take
  their seat from `HandStart::seat` or `obs.seat`, because seats rotate between replays.

## Writing a bot in Python

```python
import random
from pokerarena import Action, BaseBot, equity

class MyBot(BaseBot):
    def __init__(self, seed: int = 0, samples: int = 200):   # constructed as MyBot(seed=..., **params)
        self.rng = random.Random(seed)                         # seed randomness => reproducible runs
        self.samples = samples

    def act(self, obs):                                        # the only required method
        la = obs.legal
        live = sum(1 for s, st in enumerate(obs.status) if s != obs.seat and st in (1, 2))
        e = equity(obs.hole, obs.board, live, self.samples, seed=self.rng.getrandbits(63),
                   variant=obs.variant)
        if la.can_raise and e > 0.7:
            return Action.raise_to(la.min_raise_to)
        if la.can_check:
            return Action.check()
        return Action.call() if e > la.call_amount / (obs.pot + la.call_amount) else Action.fold()
```

Reference it from a config as `kind = "python:mymodule.MyBot"`. The module must be importable,
and `params` are passed as keyword arguments. `pokerarena.Bot` is the structural protocol.
`BaseBot` supplies no-op hooks: `new_match(info)`, `new_hand(info)`, `observe(event)` and
`end_hand(result)`, each receiving a dict (see `python/pokerarena/bot.py`).

- **Cards** are ints `0..52` (`rank*4 + suit`, deuce = 0, suits c d h s). Helpers:
  `card_to_str`, `parse_cards`, `evaluate`, `hand_category`.
- **Observation arrays**: `hole` and `board` are `np.uint8`; `stacks`, `street_contrib` and
  `total_contrib` are `np.uint64`. `status` is `np.uint8` (0 empty, 1 active, 2 all-in,
  3 folded). `history` is a list of `(seat, street, kind, amount, to, all_in)`.
- **Return value** from `act`: an `Action`, a kind string (`"call"`), or a tuple
  (`("raise_to", 40)`). An exception in `act` counts as an illegal action. The first
  traceback per bot instance is printed.
- **Running**: `pokerarena.run("configs/x.toml", processes=None, threaded=False, seed=None,
  rounds=None, threads=None, fresh=False, save=True)` returns `RunResult(rows, ratings,
  table, results_dir)`. Rust-only tables run inside Rust on a thread pool with the GIL
  released. Tables that seat a Python bot are spread over a `multiprocessing` pool. Each
  worker runs its tables through Rust and returns serialized results, and the parent merges
  them in table-id order before updating ratings. On free-threaded CPython (3.13t+, GIL
  disabled), `threaded=True` runs Python tables on threads instead. On GIL builds it warns
  and falls back to processes. Results are identical for any process or thread count.
- Hooks a bot doesn't override (including `BaseBot`'s no-ops) are never called, so they cost
  nothing.
- The `arena` CLI runs Rust bots only. Configs containing `python:` kinds go through
  `pokerarena.run`.

## Config format

```toml
[run]
rounds = 5                    # R rounds; each round is re-scheduled
seed = 2026                   # master seed
# threads = 8                 # default: all cores
results_dir = "results"
leaderboard = "leaderboard.json"

[format]
variant = "nlhe"              # nlhe | plo | flhe
max_seats = 6                 # 2..=10
# hole_cards = 4              # plo only (4..=6)
# small_bet = 2, big_bet = 4, cap = 4   # flhe only (defaults: bb, 2*small_bet, 4)

[table]
hands_per_table = 1000        # max hands in tournaments; hands (not deals) in duplicate mode
starting_stack_bb = 100
mode = "cash"                 # cash | tournament
blinds = { sb = 1, bb = 2, ante = 0 }                       # cash
# levels = [{ sb = 1, bb = 2, ante = 0, hands = 40 }, ...] # tournament (last level lasts)
stack_policy = "reset_each_hand"   # reset_each_hand | rebuy_when_busted | top_up_to_start
duplicate_mode = false        # replay each deal with bots rotated through all seats
illegal_action_policy = "fold_or_check"   # fold_or_check | error
record_timing = false         # per-decision wall-clock (not deterministic)
hand_history = false          # PokerStars-like text per table, in results/<run_id>/hand_histories/

[matchmaking]
# seats_per_table = 6         # default: format.max_seats
min_appearances = 4
allow_duplicate_bot_at_table = true
min_tables = 0
balance_pairs = false
balance_steps = 500

[rating]
initial = 1500
k = 16
score_mode = "binary"         # binary | margin
margin_c = 30

[[bots]]
name = "tag"
version = "1"                 # optional; leaderboard key becomes "tag@1"
kind = "tag"                  # random | calling_station | tag | python:module.ClassName
params = { samples = 300 }    # passed to the factory / constructor
```

Unknown keys are rejected. Duplicate mode requires `reset_each_hand` and is not available
for tournaments. Example configs: `configs/6max_nlh.toml` (the 7-bot demo),
`configs/hu_nlh.toml` (duplicate mode, margin Elo), `configs/6max_plo.toml` and
`configs/tournament_6max.toml`.

Built-in bots:

- `random` takes params `fold`, `call`, `raise` (weights) and `raise_sizes` (pot fractions).
- `calling_station` takes no params.
- `tag` takes params `samples`, `tightness` and `call_margin`. Preflop it uses Chen-formula
  hand classes; postflop (and preflop in PLO) it uses Monte Carlo equity against the live
  opponents.

## Matchmaking

Inputs are `k` bots, a seats-per-table target `s` (default `max_seats`), `min_appearances` `m`,
`allow_duplicate_bot_at_table` and a seed (derived from the master seed and the round).

1. **Effective seats**: `s' = min(s, k)` without duplicates, otherwise `s' = s`. When `k < s`,
   tables are filled with independent instances of the same bots. Their seats are
   aggregated into that bot's stats and pooled per table for rating.
2. **Tables**: `T = max(k, ⌈m·k/s'⌉, min_tables)`.
3. **Fill**: concatenate independent random permutations of the bots and cut the first `T·s'`
   entries into tables. Every bot then appears `⌊T·s'/k⌋` or `⌈T·s'/k⌉` times, which is at
   least `m`.
4. **Repair**: a table holding a bot more than `cap = ⌈s'/k⌉` times swaps that slot with a
   slot in another table where the swap creates no new violation. `cap` is 1, meaning no
   duplicates, whenever `s' ≤ k`. If no swap exists the run fails loudly.
5. **`balance_pairs`** (optional): swap-based local search that accepts only swaps lowering
   the variance of the pair co-occurrence counts.
6. Seat order within each table is shuffled.

Tests check, over a grid plus 400 random cases (including `k < s`, `k = s`, heads-up and
large `k`), that appearances are at least `m` and differ by at most 1, that duplicate caps
hold, that `T ≥ k`, and that schedules are deterministic.

## Ratings: pairwise multiplayer Elo

**Pooling.** When a bot holds several seats at a table (duplicates allowed and `k < s`), its
seats are pooled into one result for that table. In cash games that's its bb/100 over all
its seats (total net / total hands). In tournaments it's the mean finishing place of its
instances. Rating then works on the `n` distinct bots at the table. For each pair of
distinct bots `(i, j)`:

- Expected score: `E_ij = 1 / (1 + 10^((R_j − R_i)/400))`, using ratings frozen at round start.
- Actual score `S_ij`:
  - **Cash** compares pooled **bb/100** at that table. `binary` gives 1, ½ or 0. `margin`
    gives `½ + ½·tanh((w_i − w_j)/c)`, with `c = 30` by default.
  - **Tournaments** compare (mean) **finishing places** (1, ½ or 0). Players busted in the
    same hand are ordered by their stack at the start of that hand. `margin` falls back to
    places.
- Update: `Δ_i = K/(n−1) · Σ_j (S_ij − E_ij)`.

Deltas accumulate over the whole round and are applied once at round end, so results don't
depend on table order. Each pair's update is symmetric, so total rating is conserved.
Defaults: initial 1500, K = 16. The `RatingSystem` trait (`begin_round`, `add_table`, `end_round`, `ratings`) leaves room for
TrueSkill or Plackett–Luce.

The leaderboard reports Elo and its change, hands, hand-weighted bb/100, and a 95% CI of
`±1.96·sd/√n`, computed over the per-table bb/100 samples (the bot's seats at a table pooled). It also shows VPIP, PFR,
showdown rate, illegal actions, mean decision time and (tournaments) mean place. In
tournaments bb/100 uses the first level's big blind.

## Determinism

Given (config, master seed, bot set), a run is bit-for-bit reproducible at any thread or
process count. The schedule of round `r` is seeded `mix(master, r, SCHEDULE)`. Table `t` is
seeded `mix(master, r, t)`, and that seed derives the deck RNG and every bot instance's seed.
Tables share no mutable state, results merge in table-id order, and ratings update
sequentially after each round. Tests assert identical serialized output at 1, 3 and 8
threads (Rust), and at 1 and 4 processes for mixed Rust+Python runs. Decision timing is the
one non-deterministic field, and it is off by default.

## Rules and conventions

- Chips are `u64` integers. Chip conservation is asserted after every action in debug builds
  and property-tested.
- Heads-up, the button posts the small blind, acts first preflop and last postflop.
- Minimum raise: the increment must be at least the largest full bet or raise on the street.
  Preflop the big blind counts as the opening bet. A short all-in below a full raise does not
  reopen betting for players who already acted, unless what they now face since their last
  action totals a full raise (TDA cumulative rule). In fixed-limit, short all-ins also don't
  count toward the cap.
- Side pots form from all-in contribution levels and are awarded among eligible non-folded
  players. Odd chips go to the first winner left of the button. Uncalled bets are returned
  before showdown. When at most one player can still bet, the board runs out with no further
  action.
- A player who can't cover the big blind posts what they have, and others still call the
  full blind. Antes are dead money.
- PLO: exactly two hole cards plus three board cards. Pot-limit maximum raise-to is current
  bet + (pot after calling).
- Dealing: hole cards go round-robin starting left of the button, then the five board cards.
  There are no burn cards, which keeps pre-arranged test decks readable.
- At showdown every remaining player's cards are revealed. Folded hands and uncontested
  winners are never shown.
- Tournaments use a simplified moving button (next seat with chips).
