# Poker Arena

A high-throughput engine for running poker bots against each other: 6-max No-Limit Hold'em
first, plus Pot-Limit Omaha and Fixed-Limit Hold'em. Tables run in parallel, matchmaking is
configurable, and results feed a pairwise multiplayer Elo leaderboard anchored by three
baseline bots. It is an evaluation harness for poker AI research. The priorities are rules
and chip-accounting correctness first, speed second.

The engine is Rust. **Bots are Python**: you write a class with an `act(obs)` method, and
the Rust engine deals, enforces the rules, runs tables in parallel processes and rates the
results. **[BOTS.md](BOTS.md) is the full bot-author guide.**

```
crates/engine   cards, deck, hand evaluator, equity, hand state machine (no I/O, no threads)
                rules/: one file per format and per betting structure (see "Adding a format")
crates/arena    Bot trait, table runner, stats, matchmaking, ratings, leaderboard, replays, config
crates/cli      the `arena` binary: bench, leaderboard
python/         PyO3 bindings (crate arena_py) + Python package `pokerarena` (runner, GUI, catalog)
bots/NLH/       the bots, one file each (the three anchored baselines live here)
macos/          sources of the double-click app (Poker Arena.app)
configs/        example TOML configs
```

## Desktop app

Double-click **`Poker Arena.app`** at the repository root (macOS) to open the Poker Arena
window. No terminal is needed once the Python environment is built:

```sh
cd python && python3 -m venv .venv
.venv/bin/pip install maturin numpy pytest PySide6-Essentials
.venv/bin/maturin develop --release
```

The first time, macOS asks whether *Poker Arena* may access the folder holding the
project (for example Documents). Click **Allow**. If you clicked Don't Allow, enable it in
System Settings › Privacy & Security › Files & Folders.

- **Settings:** variant, seats, cash or tournament (blind levels are generated from your
  blinds), stack, blinds and ante, stack policy, duplicate mode, hands per table, rounds,
  players per table, minimum tables per round, Elo mode and K, seed, worker processes,
  decision timing, saving, and hand replays.
- **Players:** one row per bot, with an editable name, the bot (a file in `bots/NLH`), its
  **Anchor Elo** (declared by the bot file) and a read-only **Start Elo**. Start Elo is
  where the bot begins: its anchor, its saved rating, or the initial 1500. **Add bot**
  offers every bot in `bots/NLH`, so drop a new `.py` file there to make it available. A
  second copy of a bot gets a numbered name (`tag_2`). **Reset to anchored bots** restores
  the default lineup. Bot parameters are hardcoded in the bot files, so the app has nothing
  to tune per bot. A live preview shows tables per round and total hands.
- **Save results & leaderboard:** bots start from their Elo in `leaderboard.json` when they
  have one (anchors always win). At the end the app writes `results/<run_id>/` and updates
  the leaderboard.
- **Save 100 random hands per bot:** after the run, writes
  `hand_replays/<run_id>/<bot>.txt`. Each file holds 100 hands drawn at random from all the
  hands that bot played, as readable hand histories with every player's cards shown. Only
  the sampled hands are recorded, so this costs no speed. It works independently of saving
  results, and `hand_replays/` is not committed to git.
- **Run** shows a hand-level progress bar with round, tables, hands/s and ETA, then the
  leaderboard, with anchored bots marked.
- **Stop** is a hard stop. Worker processes are killed (even a bot stuck in an infinite
  loop), and nothing is written. Closing the window mid-run does the same. Tests check that
  a stopped run leaves no worker processes, threads, file descriptors or semaphores behind,
  and that a crashed worker fails the run instead of hanging it.

The app is a small AppleScript applet that runs `Contents/Resources/launch.sh`.
`macos/build_app.sh` rebuilds it, and the log goes to `.poker_arena.log`. Without the app:
`python/.venv/bin/python -c "from pokerarena.gui import main; main()"`.

## Running from Python

```python
import pokerarena as pa

if __name__ == "__main__":          # required: worker processes re-import the main script
    res = pa.run("configs/6max_nlh.toml", save=True)
    print(res.table)
```

`pokerarena.run(config, processes=None, threaded=False, seed=None, rounds=None, fresh=False,
save=True, leaderboard=None, results_dir=None, replays_dir=None)` returns
`RunResult(rows, ratings, table, results_dir, replays_dir, hands, tables, elapsed)`.
`config` is a TOML path, TOML or JSON text, or a dict. With `save=True` bots start from
their stored ratings (unless `fresh=True`), and at the end the run writes
`results/<run_id>/` (`config.toml`, `schedule.json`, `tables.jsonl`, `leaderboard.json`,
`leaderboard.txt`) and updates `leaderboard.json`.

`pokerarena.ArenaJob` is the controllable form: `start()` runs in the background,
`progress()` returns a snapshot, `stop()` is the hard stop, and `wait()` waits for cleanup.

Tables run in a `multiprocessing` pool of worker processes (`spawn`), one table per task.
Each worker plays its tables through the Rust engine and reports hand-level progress, and
the parent merges results in table-id order before updating ratings. On free-threaded
CPython (3.13t+, GIL disabled), `threaded=True` runs tables on threads instead. Results
are identical for any process or thread count.

## Build and test

```sh
cargo build --release
cargo test --workspace
cargo test --workspace --release -- --ignored     # exhaustive C(52,7) evaluator enumeration (~1 s)
cargo bench -p engine                             # evaluator + hand throughput (criterion)
cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check
cargo run --release -p cli -- bench               # engine/runner throughput, 1 thread vs all
cargo run --release -p cli -- leaderboard         # print leaderboard.json

cd python && .venv/bin/maturin develop --release && .venv/bin/python -m pytest
```

The Rust tests drive the engine and runner with small test-only bots
(`crates/arena/tests/common`). They're fixtures, not baselines. `arena bench` likewise uses
a built-in random policy that exists only to measure throughput.

## Performance

Apple M4 Pro (8 performance + 4 efficiency cores), release build:

| | 1 thread | 12 threads |
|---|---|---|
| 7-card evaluator, random hands | 460–480 M evals/s | 4.0–4.2 G evals/s (8.7×) |
| Engine + runner, 6-max NLHE, built-in random policy (`arena bench`) | 1.07–1.14 M hands/s | 7.2–8.4 M hands/s |

The bare engine loop (no bot dispatch, no event broadcast) runs 2.0 M 6-max hands/s on one
core. With real Python bots, the bots dominate. The default lineup (the 3 anchored bots,
6-max NLHE) runs at about **82k hands/s** with the default schedule. That schedule has only
3 tables per round, so only 3 cores work. Raising **Min tables per round** to 12 uses every
core: about **210k hands/s**. A Python decision
costs about 1.1 µs for a trivial bot and 3–5 µs for the tight-aggressive baseline, whose
equity sampling runs in Rust.

## Bots folder, baselines and anchors

Bots live in **`bots/<FORMAT>/`**, one file per bot. `bots/NLH` holds the three anchored
baselines:

| File | Class | Name | Anchor Elo | Play |
|---|---|---|---|---|
| `bots/NLH/random_legal.py` | `RandomLegal` | `random` | 850 | fold / call / raise at random (1 : 2 : 1), 0.5 or 1 pot |
| `bots/NLH/calling_station.py` | `CallingStation` | `station` | 400 | always checks or calls |
| `bots/NLH/tag.py` | `TightAggressive` | `tag` | 1300 | Chen-formula preflop, Monte Carlo equity postflop (300 samples) |

A bot file defines one class with an `act` method. Its parameters are class constants in the
file, and two optional class attributes describe it: `NAME` (the default player name) and
`ANCHOR` (a fixed Elo). `pokerarena.discover()` lists the bots in a folder, and the app's
**Add bot** uses it. Config kinds are `python:bots.NLH.<file>.<Class>`. Importing
`pokerarena` puts the repository root on `sys.path`, so `bots` is importable, worker
processes included. BOTS.md covers writing one.

An anchored bot still enters every expected-score computation, but its own rating never
changes, so the anchors pin the scale and ratings stay comparable across runs and
leaderboards. A bot that reliably beats `tag` ends up above 1300. Unanchored bots start at
`[rating] initial` (1500 by default) or at their saved rating.

## Config format

```toml
[run]
rounds = 5                    # R rounds; each round is re-scheduled
seed = 2026                   # master seed
# threads = 8                 # threads for in-process tables (free-threaded Python only)
results_dir = "results"
leaderboard = "leaderboard.json"
replay_hands_per_bot = 0      # > 0: save that many random hands per bot (hand replays)
replays_dir = "hand_replays"

[format]
variant = "nlhe"              # a registered format: nlhe | plo | flhe
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
hand_history = false          # every hand of every table, in results/<run_id>/hand_histories/

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
# version = "1"               # optional; leaderboard key becomes "tag@1"
kind = "python:bots.NLH.tag.TightAggressive"   # python:module.ClassName
anchor = 1300                 # optional fixed Elo (the bot file's ANCHOR)
# params = { ... }            # optional constructor keyword arguments (custom bots only;
                              # the bots in bots/NLH hardcode their parameters)
```

Unknown keys are rejected, and every bot must be a `python:` kind. Duplicate mode requires
`reset_each_hand` and is not available for tournaments. Example configs, all with the three
anchored bots:

- `configs/6max_nlh.toml`: the app's default settings.
- `configs/hu_nlh.toml`: heads-up with duplicate mode and margin Elo.
- `configs/6max_plo.toml`
- `configs/tournament_6max.toml`

## Adding a format

Everything that differs between poker formats lives in `crates/engine/src/rules/`. The
hand state machine (`state.rs`) is format-independent and asks these modules:

```
rules/mod.rs          FormatRules trait + registry (rules(variant), by_name("plo"), ALL)
rules/formats/nlhe.rs  No-Limit Hold'em: 2 hole cards, Hold'em showdown, no-limit betting
rules/formats/plo.rs   Pot-Limit Omaha: 4 hole cards, exactly-2-plus-3 showdown, pot-limit
rules/formats/flhe.rs  Fixed-Limit Hold'em: 2 hole cards, Hold'em showdown, fixed-limit
rules/betting/         one file per betting structure (BettingRules: opening bet, legal
                       raise range, cap): no_limit.rs, pot_limit.rs, fixed_limit.rs
rules/actions.rs       structure-independent action logic: legal_actions() computes what the
                       player to act may do; resolve() validates the action a bot returned
```

A format implements `FormatRules`: its name and title, its standard `GameFormat` (hole
cards and betting structure), validation, showdown strength and Monte Carlo equity. The
last two pass a statically dispatched evaluator to shared generic code. To add one:

1. Write `rules/formats/<name>.rs` implementing `FormatRules` (copy `nlhe.rs`).
2. Add a variant to `Variant` and one line each to `rules()` and `ALL` in `rules/mod.rs`.
3. If it needs a new betting structure, add `rules/betting/<name>.rs` implementing
   `BettingRules` and a `Betting` variant.

Configs (`variant = "<name>"`), hand histories, the Python bindings and `equity(...,
variant=...)` all look formats up through the registry, so nothing else changes. Add
scripted hands for the new rules in `crates/engine/tests/scripted.rs`.

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
depend on table order. **Anchored** bots (see above) keep their rating, so with anchors
total rating is not conserved. Defaults: initial 1500, K = 16.

**Choosing K.** K is a step size. It changes how fast and how noisily ratings move, not what
a rating difference means: the 400-point logistic scale is what makes ratings comparable to
chess (400 points ≈ 91% expected score). Each table splits K across the other n−1 bots, so
the chess-equivalent K per pairwise comparison is K/(n−1), about K/5 at 6-max. Chess uses
10–40 per game. These figures were measured on a 7-bot lineup (the three baseline
strategies plus four variants; 60 rounds of 1,000 hands per table, no anchors):

| K | rating noise (SD) | median move per round | rounds where the order changed |
|---|---|---|---|
| 16 (default) | 19 | 4 | 0–3 of 29 |
| 100 (≈ chess K 20 per pair) | 40 | 23 | 15–21 of 29 |
| 500 | 220 | 200 | 28–29 of 29 |

Small K converges more slowly (about 35 rounds to 90% of the final top-to-bottom gap at
K = 16, against about 20 at K = 100). Large K makes the leaderboard mostly reflect the last
round. When one bot beats another at every table, binary Elo gaps keep widening without
bound, at a speed set by K, so compare gaps only between runs with the same K and round
count.

The `RatingSystem` trait (`begin_round`, `add_table`, `end_round`, `ratings`) leaves room for
TrueSkill or Plackett–Luce.

The leaderboard reports Elo and its change, hands, hand-weighted bb/100, and a 95% CI of
`±1.96·sd/√n`, computed over the per-table bb/100 samples (the bot's seats at a table pooled).
It also shows VPIP, PFR, showdown rate, illegal actions, mean decision time and, for
tournaments, mean place. In tournaments bb/100 uses the first level's big blind.

## Hand replays

With `replay_hands_per_bot = N` (the app uses 100), each bot gets `N` hands drawn uniformly
without replacement from all the hands it played in the run. The sample is drawn up front
from the master seed and the schedules, so every worker process agrees on it, and only
those hands' histories are recorded. Results are identical with and without replays. Files
go to `hand_replays/<run_id>/<bot>.txt` after a completed run (never after Stop). Each hand
is headed `### Round r · table t · hand h` and followed by a PokerStars-style history
showing all hole cards. In tournaments a sampled hand may never be played, if the table
finished earlier, so a file can hold fewer than `N` hands.

## Determinism

Given (config, master seed, bot set), a run is bit-for-bit reproducible at any thread or
process count. The schedule of round `r` is seeded `mix(master, r, SCHEDULE)`. Table `t` is
seeded `mix(master, r, t)`, and that seed derives the deck RNG and every bot instance's seed.
Tables share no mutable state, results merge in table-id order, and ratings update
sequentially after each round. Tests assert identical serialized output at 1, 3 and 8
threads, and at 1 and 4 worker processes for Python runs. Decision timing is the one
non-deterministic field, and it is off by default.

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
