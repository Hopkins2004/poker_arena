# CLAUDE.md — Poker Arena

High-throughput engine for running poker bots against each other (6-max NLH first, other variants later), with parallel table execution, configurable matchmaking, and an Elo leaderboard. It is an evaluation harness for poker AI research, so **correctness of the rules and of chip accounting comes first, raw speed second, everything else third.**

## Stack

- **Rust (stable, edition 2021)** for everything performance-critical: cards, hand evaluation, game state machine, table runner, matchmaking, ratings.
- **rayon** for CPU parallelism (one table = one task; tables never share mutable state).
- **PyO3 + maturin** for Python bindings. **All bots are Python** (NumPy/PyTorch friendly); the Rust engine deals, enforces rules, runs tables in worker processes and rates results. Bots live in `bots/<FORMAT>/` (one file each, parameters hardcoded); the anchored baselines are `bots/NLH/` random (850), calling station (400) and tight-aggressive (1300). The desktop app's Add bot lists that folder.
- **serde + toml** for config, **serde_json** for results and leaderboards, **clap** for the CLI.
- Testing: built-in `#[test]`, **proptest** for invariants, **criterion** for benchmarks.
- RNG: `rand_xoshiro` (Xoshiro256++) or `rand_chacha`, always seeded explicitly. Never use `thread_rng()` in engine code.

## Workspace layout

```
crates/
  engine/    # cards, deck, hand evaluator, equity, state machine; rules/ = one file per format and per betting structure. No I/O, no threads.
  arena/     # bot trait, table runner, matchmaking, ratings (with anchors), stats, replays, config, parallel scheduler
  cli/       # `arena` binary: bench, leaderboard
  solver/    # exact river solver: vectorized public-tree DCFR over 1326 combos, range equity (engine only)
python/      # PyO3 bindings (crate `arena_py`) + Python package `pokerarena` (baselines, runner, desktop GUI) + example bots
bots/NLH/    # bots, one file each (anchored baselines)
river_net/   # river / pre-river / turn value networks, turn and flop solvers: solver wrapper, range generator, datagen, models (Python, uses _native.solver_*); reference: bots/NLH/SOLVERS.md
ranges/      # preflop range database (6-max 100bb, CC BY 4.0 source vendored) + converter; read via pokerarena.preflop
configs/     # example TOML configs
```

Dependency direction: `engine` ← `arena` ← `cli`, `python`; `engine` ← `solver` ← `python`. `engine` must not depend on anything heavier than `rand` core traits.

## Core design rules

**Chips are integers.** Use `u64` chip units (blinds are integers, e.g. SB = 1, BB = 2). Never use floats for money. bb/100 and Elo are floats only in reporting.

**Chip conservation is a hard invariant.** The sum of stacks plus the pot is constant within a hand. Assert it in debug builds after every action and property-test it.

**Information hiding is enforced by the engine.** A bot receives only an `Observation` built for its own seat: its hole cards, the board, the public action history, stacks and pot. Showdown cards are revealed only per the showdown rules. There must be no code path that hands a bot another seat's hole cards or the deck.

**Bots never mutate game state.** They return an `Action`, and the engine validates it. Illegal actions follow `illegal_action_policy` (`fold_or_check` default, `error` in strict mode) and are counted in stats.

**Determinism.** Given (config, master seed, bot set), a run is bit-for-bit reproducible regardless of thread count:
- Each table's RNG is derived from `hash(master_seed, round, table_id)`.
- Results are merged in table-id order.
- Rating updates are applied in a deterministic order after each round, never concurrently.

**Hot path hygiene.**
- No heap allocation per action or per hand: reuse buffers and preallocate action-history vectors.
- No `dyn` dispatch inside the evaluator. `dyn Bot` at the bot boundary is fine.
- No logging and no hand-history formatting unless enabled. Logging is compiled-in but gated by a config flag that is checked once per hand, not per action.
- Cards are `u8` (0..52), and hands/boards are `u64` bitmasks where convenient.

## Rules pitfalls (must be handled and tested)

- **Heads-up:** the button posts the SB and acts first preflop and last postflop.
- **Minimum raise:** the raise increment must be ≥ the largest previous full bet/raise increment on that street, with the BB counting as the opening bet preflop.
- **Short all-ins:** an all-in raise smaller than a full raise does **not** reopen betting for players who have already acted on the current full bet.
- **Side pots:** build them from all-in contribution levels. Each pot is awarded only among the eligible non-folded players.
- **Split pots:** odd chips go to the first eligible seat left of the button.
- **Uncalled bets:** the excess is returned to the bettor before showdown.
- **Everyone all-in:** run out the remaining board with no further actions.
- **PLO:** the best hand must use exactly 2 hole cards and 3 board cards.
- **Pot-limit:** max raise to = call amount + (pot after calling).
- **Busted players** in cash mode are reset or topped up per `stack_policy`. In tournament mode they are eliminated and the table shrinks.

## Bot interface

The Rust trait in `arena::bot` (keep Python mirrors in sync):

```rust
pub trait Bot: Send {
    fn name(&self) -> &str;
    fn new_match(&mut self, info: &MatchInfo) {}        // format, seats, blinds, seat index
    fn new_hand(&mut self, info: &HandStart) {}         // hole cards, button, stacks, blinds
    fn act(&mut self, obs: &Observation) -> Action;     // the only required method
    fn observe(&mut self, event: &PublicEvent) {}       // every public action / street / showdown
    fn end_hand(&mut self, result: &HandResult) {}
}
pub type BotFactory = Arc<dyn Fn(u64 /*seed*/) -> Box<dyn Bot> + Send + Sync>;
```

- `Action` is `Fold | Check | Call | BetTo(u64) | RaiseTo(u64) | AllIn`, with amounts as **total street contribution** (the "raise-to" convention).
- `Observation` includes a precomputed `LegalActions { can_check, call_amount, min_raise_to, max_raise_to }`, so bots never reimplement the rules.
- Each table gets **fresh bot instances** from the factory, so there is no shared state between tables. A bot that wants shared read-only data (e.g. a strategy table) holds an `Arc` captured in its factory.

## Commands

```
cargo build --release
cargo test --workspace
cargo test --workspace --release -- --ignored        # slow exhaustive tests (evaluator enumeration)
cargo bench -p engine                                # evaluator + hand-throughput benchmarks
cargo run --release -p cli -- bench                 # runs with bots: Poker Arena.app or pokerarena.run
cargo clippy --workspace -- -D warnings && cargo fmt --check
cd python && maturin develop --release && pytest
python/.venv/bin/python -m pytest river_net/tests     # from the repo root
python/.venv/bin/python -m river_net.turn_bench      # turn-solver speed/quality benchmark (bots/NLH/SOLVERS.md)
python/.venv/bin/python -m river_net.flop_check      # flop-solver quality check (turn-solver continuation)
cargo run --release -p solver --example bench_solve   # river solve times
```

Always benchmark in `--release`. Set `RUSTFLAGS="-C target-cpu=native"` for local benchmarking only, never in committed build config.

## Working conventions

- Write tests alongside each feature. Rules features need at least one scripted-hand test, i.e. a fixed deck plus a fixed action list with asserted final stacks.
- Before claiming a performance improvement, run `cargo bench` before and after and report the numbers.
- Keep `engine` free of matchmaking and rating concepts. Keep `arena` free of rules logic.
- Format-specific logic belongs in `engine/src/rules/` (a `formats/<name>.rs` per format, a `betting/<name>.rs` per betting structure); the state machine and everything above it look formats up through the registry. Never match on `Variant` outside `rules/`.
- When changing the `Bot` trait or `Observation`, update the Python bindings, the Python protocol class and the example bots in the same change.
- Public types get doc comments. Non-obvious rules code gets a comment citing the rule it implements.
- Don't add dependencies beyond those listed without stating why in the PR/commit message.
