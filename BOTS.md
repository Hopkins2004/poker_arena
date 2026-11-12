# Writing a bot for Poker Arena

This guide covers everything a bot author needs: what to implement, when each method is
called, exactly what information arrives and in what form, what you may return, and the
rules of the road (determinism, duplicate mode, tournaments, performance). It applies to
Rust bots and Python bots alike. The two interfaces mirror each other, and differences are
called out where they exist.

The complete example bots in §8 are real, tested code:
`crates/bots/tests/guide_example.rs` (Rust) and `python/examples/pot_odds_bot.py` (Python).

**Contents**

1. [The short version](#1-the-short-version)
2. [Lifecycle: what is called, and when](#2-lifecycle-what-is-called-and-when)
3. [What your bot sees](#3-what-your-bot-sees)
4. [What your bot returns](#4-what-your-bot-returns)
5. [Rules your bot should know](#5-rules-your-bot-should-know)
6. [Helpers: equity and hand strength](#6-helpers-equity-and-hand-strength)
7. [Rules of the road](#7-rules-of-the-road)
8. [Complete examples](#8-complete-examples)
9. [Testing and evaluating your bot](#9-testing-and-evaluating-your-bot)
10. [Checklist](#10-checklist)

---

## 1. The short version

A bot is an object with one required method, **`act(observation) -> action`**. The engine
calls it whenever it is your turn and validates whatever you return. You never change game
state yourself. Four optional hooks tell you about the match and the hand as it unfolds.

**Rust**: implement `arena::Bot`:

```rust
use arena::bot::{Action, Bot, Observation};

pub struct MyBot { name: String }

impl Bot for MyBot {
    fn name(&self) -> &str { &self.name }
    fn act(&mut self, obs: &Observation) -> Action {
        if obs.legal.can_check { Action::Check } else { Action::Call }
    }
    // optional: new_match, new_hand, observe, end_hand
}
```

**Python**: any class with `act(self, obs)`. Subclassing `pokerarena.BaseBot` gives you
no-op hooks for free:

```python
from pokerarena import Action, BaseBot

class MyBot(BaseBot):
    def __init__(self, seed: int = 0, **params):
        super().__init__(seed)

    def act(self, obs):
        return Action.check() if obs.legal.can_check else Action.call()
```

Use it from a config (TOML file, the desktop app's player list, or `pokerarena.run`):

```toml
[[bots]]
name = "mine"
kind = "python:mybots.MyBot"      # Rust bots use a registered kind, e.g. "tag"
params = { aggression = 0.7 }     # passed to the constructor / factory
```

## 2. Lifecycle: what is called, and when

### 2.1 Creation

The arena never reuses a bot object across tables. **Every seat at every table gets a fresh
instance**, created with a seed derived from the master seed, the round and the table:

- **Rust:** you provide a `BotFactory = Arc<dyn Fn(u64 /*seed*/) -> Box<dyn Bot> + Send + Sync>`.
  It is called once per seat when a table starts.
- **Python:** your class is constructed as **`ClassName(seed=<int>, **params)`**, where
  `params` is the `params` table from the config. A Python bot is also constructed once
  with `seed=0` when the run is set up (to fail fast on errors), and the desktop app does
  the same whenever you edit the settings. Keep constructors cheap and free of side effects.

A bot's lifetime is therefore one table, which is typically 1,000 to 100,000 hands
(`hands_per_table`). State you keep in the object persists across those hands. Nothing
persists across tables or rounds. To share read-only data across instances (a strategy
table, a neural network), load it once: in Rust, capture an `Arc` in the factory; in
Python, use a module-level cache (one per worker process).

### 2.2 Calls during a table

```
new_match(info)                     once, before the first hand
for each hand you are dealt into:
    new_hand(info)                  your hole cards, the button, stacks, blinds
    observe(event) ...              antes and blinds, as action events
    loop until the hand ends:
        act(obs) -> action          only when it's your turn
        observe(event) ...          every action (yours too), every new street
    observe(event) ...              uncalled bet returned, showdown reveals, pot awards
    end_hand(result)                net result, final stacks, final board
```

`observe` receives every public event of the hand, including your own actions and events
that happen after you fold. You are only called for hands you are dealt into: in
tournaments, an eliminated bot receives nothing further.

### 2.3 A real hand, as one bot saw it

This is a verbatim capture (reformatted) of a heads-up hand from the point of view of a
Python bot in seat 0, the button, playing a calling station in seat 1. Blinds are 1/2 and
stacks are 200 chips (100 bb).

```text
new_match {variant: nlhe, max_seats: 2, n_seats: 2, seat: 0, sb: 1, bb: 2, ante: 0,
           starting_stack: 200, hands: 1, tournament: False, duplicate: False}
new_hand  {hand_id: 0, seat: 0, hole: [4h 5s], button: 0, stacks: [200, 200], sb: 1, bb: 2, ante: 0}
  observe {type: action, seat: 0, street: 0, kind: sb, amount: 1, to: 1, all_in: False}
  observe {type: action, seat: 1, street: 0, kind: bb, amount: 2, to: 2, all_in: False}
  act     street=0 hole=[4h 5s] board=[] pot=3 current_bet=2
          stacks=[199, 198] street_contrib=[1, 2] total_contrib=[1, 2] status=[1, 1]
          legal: can_check=False call_amount=1 can_raise=True min_raise_to=4 max_raise_to=200
          history: (0,0,'sb',1,1,False) (1,0,'bb',2,2,False)
          -> Action.raise_to(6)
  observe {type: action, seat: 0, street: 0, kind: raise, amount: 5, to: 6, all_in: False}
  observe {type: action, seat: 1, street: 0, kind: call,  amount: 4, to: 6, all_in: False}
  observe {type: street, street: 1, board: [Th 3c Kc]}
  observe {type: action, seat: 1, street: 1, kind: check, amount: 0, to: 0, all_in: False}
  act     street=1 board=[Th 3c Kc] pot=12 current_bet=0 stacks=[194, 194]
          street_contrib=[0, 0] total_contrib=[6, 6]
          legal: can_check=True call_amount=0 can_raise=True min_raise_to=2 max_raise_to=194
          -> Action.check()
  ... turn [3h] and river [6d] are checked down ...
  observe {type: showdown, seat: 0, hole: [4h 5s], strength: 8556}    (one pair)
  observe {type: showdown, seat: 1, hole: [Ks 4s], strength: 12911}   (two pair)
  observe {type: pot_awarded, seat: 1, amount: 12}
end_hand  {hand_id: 0, net: [-6, 6], stacks: [194, 206], board: [Th 3c Kc 3h 6d], showdown: True}
```

Things to notice:

- **Heads-up, the button posts the small blind and acts first preflop, but last after the flop.**
  Seat 1 checked first on the flop.
- **`pot` includes bets on the current street.** Preflop it's 3 (the blinds), and on the
  flop it's 12.
- **Amounts are raise-to totals.** The raise to 6 moved 5 chips (`amount`) because 1 was
  already in (`to` = 6). The call moved 4 chips (`amount`), bringing seat 1 to 6.
- **`street_contrib` resets every street**, while **`total_contrib` accumulates** across the hand.
- **`min_raise_to` preflop is 4:** the big blind of 2 plus a minimum raise increment of 2.
  On the flop the minimum bet is one big blind (2). `max_raise_to` is your whole stack plus
  what you already put in this street.
- **Hole cards are revealed only at showdown**, and only for players still in the hand.

## 3. What your bot sees

### 3.1 Cards, seats and positions

**Cards** are integers `0..52`: `card = rank * 4 + suit`.

- rank: `0` = 2, `1` = 3, … `8` = T, `9` = J, `10` = Q, `11` = K, `12` = A
- suit: `0` = clubs, `1` = diamonds, `2` = hearts, `3` = spades

So `"As"` = 51, `"2c"` = 0, `"Td"` = 33. Rust: `Card(u8)` with `.rank()`, `.suit()` and
`Display`/`FromStr` (`"As".parse()`). Python: `card_to_str(51) == "As"`,
`parse_cards("AsKd") == [51, 45]`. In NumPy, `obs.hole >> 2` gives ranks and `obs.hole & 3`
gives suits.

**Seats** are numbered `0..n_seats` clockwise, and action proceeds to increasing seat
numbers (wrapping around). `button` is the dealer seat.

- With 3+ players, the small blind is the first occupied seat after the button and the big
  blind the next one. Preflop, action starts left of the big blind. After the flop, it
  starts with the first active seat left of the button.
- Heads-up, the button is the small blind, acts first preflop and acts last after the flop.
- In tournaments, busted players' seats stay in the arrays with status `Empty` (0) and stack
  0. Always skip them when you count players or work out positions.

To find your position, count occupied seats clockwise from the button:

```python
order = [(obs.button + k) % obs.n_seats for k in range(1, obs.n_seats + 1)]
order = [s for s in order if obs.status[s] != 0]        # skip empty seats
position = order.index(obs.seat)                          # 0 = small blind, ..., last = button
```

You can find the big blind seat from the history: the record with kind `bb` (Python) or
`ActionKind::PostBigBlind` (Rust).

### 3.2 The Observation

`act` receives an Observation built by the engine **for your seat only**. Arrays are
indexed by seat. In Python, `stacks`, `street_contrib` and `total_contrib` are `np.uint64`
arrays, `hole`, `board` and `status` are `np.uint8` arrays, and `history` is a list of
tuples.

| Field | Rust type | Meaning |
|---|---|---|
| `seat` | `usize` | Your seat. |
| `n_seats` | `usize` | Seats at the table, including empty ones. |
| `button` | `usize` | Button seat. |
| `format` (Rust) / `variant`, `max_seats` (Python) | `GameFormat` | Variant (`NLHE`/`PLO`/`FLHE`, in Python `"nlhe"`/`"plo"`/`"flhe"`), max seats, hole-card count, betting structure. |
| `street` | `Street` / `int` | 0 preflop, 1 flop, 2 turn, 3 river. |
| `hole` | `&[Card]` | **Your** hole cards (2, or 4 in PLO). |
| `board` | `&[Card]` | Board cards dealt so far (0, 3, 4 or 5). |
| `stacks` | `&[u64]` | Chips **behind** for every seat, i.e. not yet in the pot. |
| `street_contrib` | `&[u64]` | Chips each seat has put in **on this street**. |
| `total_contrib` | `&[u64]` | Chips each seat has put in **this hand** (antes and blinds included). |
| `status` | `&[SeatStatus]` | `Empty` 0, `Active` 1 (can still act), `AllIn` 2, `Folded` 3. |
| `pot` | `u64` | Total chips in the pot, including current-street bets. Equals `sum(total_contrib)`. |
| `current_bet` | `u64` | Street total you must match to call. Preflop it starts at the big blind. |
| `sb`, `bb`, `ante` | `u64` | Blinds and ante of this hand (they rise between tournament levels). |
| `legal` | `LegalActions` | What you may do now (§3.3). |
| `history` | `&[ActionRecord]` | Every public action this hand, in order, blinds and antes included (§3.4). |

Useful derived quantities:

```python
to_call      = obs.legal.call_amount                              # chips needed to call
pot_odds     = to_call / (obs.pot + to_call)                      # equity needed to call
my_stack     = int(obs.stacks[obs.seat])
live         = [s for s in range(obs.n_seats) if obs.status[s] in (1, 2) and s != obs.seat]
effective    = min(my_stack, max(int(obs.stacks[s]) for s in live)) if live else 0
spr          = effective / obs.pot                                  # stack-to-pot ratio
in_bb        = lambda chips: chips / obs.bb
```

### 3.3 LegalActions

Every Observation carries a precomputed summary of the rules for your current decision, so
you never need to re-derive minimum raises, pot limits or caps:

| Field | Meaning |
|---|---|
| `can_check` | `True` when there is nothing to call. |
| `call_amount` | Chips needed to call, capped at your stack (a short call is an all-in). `0` when `can_check`. |
| `can_raise` | `True` when you may bet or raise. |
| `min_raise_to` | Smallest legal bet/raise, as a **street total**. If your stack is smaller than a full raise, this equals your all-in amount. `0` when `!can_raise`. |
| `max_raise_to` | Largest legal bet/raise, as a street total: your all-in in no-limit, the pot limit in PLO (capped by your stack), the fixed size in limit. `0` when `!can_raise`. |

`can_raise` is `False` when:

- **Betting was not reopened to you.** You already acted this street, and the only raises
  since were short all-ins totalling less than a full raise (§5). You may still call or fold.
- **The fixed-limit cap is reached** (bet plus 3 raises by default).
- **Nobody could respond.** Every other remaining player is all-in.
- **You have no chips beyond the call.**

Examples (blinds 1/2):

| Situation | `call_amount` | `min_raise_to` | `max_raise_to` |
|---|---|---|---|
| NLHE, first to act preflop, 200 behind | 2 | 4 | 200 |
| NLHE, facing an open to 6 after posting nothing | 6 | 10 | 200 |
| NLHE, flop, no bet yet, 194 behind | 0 (`can_check`) | 2 | 194 |
| PLO, first to act preflop (pot 3) | 2 | 4 | 7 (= 2 + 3 + 2) |
| PLO, facing a bet of 51 into 51 on the flop, 176 behind | 51 | 102 | 176 (pot limit 204, capped by the stack) |
| FLHE 2/4, first to act preflop | 2 | 4 | 4 |

### 3.4 Action history records

`obs.history` (and every `action` event) uses the same record:

| Field | Rust | Python tuple index | Meaning |
|---|---|---|---|
| seat | `seat: u8` | 0 | Acting seat. |
| street | `street: Street` | 1 | 0–3. |
| kind | `kind: ActionKind` | 2 | `PostAnte`/`ante`, `PostSmallBlind`/`sb`, `PostBigBlind`/`bb`, `Fold`/`fold`, `Check`/`check`, `Call`/`call`, `Bet`/`bet`, `Raise`/`raise`. |
| amount | `amount: u64` | 3 | Chips moved from the stack into the pot by this action. |
| to | `to: u64` | 4 | The seat's street contribution **after** the action (the raise-to total). Antes don't count toward street contributions, so for them `to` is 0. |
| all_in | `all_in: bool` | 5 | The action put the player all-in. |

`Bet` means the first bet on a street. Preflop every voluntary increase is a `Raise`,
because the big blind counts as the opening bet. An all-in is recorded as the call, bet or
raise it amounts to, with `all_in` set.

### 3.5 Events (`observe`)

Rust: `PublicEvent`; Python: a dict with a `"type"` key.

| Rust | Python dict | When |
|---|---|---|
| `Action(ActionRecord)` | `{"type": "action", seat, street, kind, amount, to, all_in}` | Every action, including antes and blinds, and including your own. |
| `Street { street, board }` | `{"type": "street", street, board}` | A new street is dealt. `board` is the full visible board. On an all-in runout the remaining streets arrive back to back, with no actions between them. |
| `UncalledReturned { seat, amount }` | `{"type": "uncalled_returned", seat, amount}` | The part of a bet nobody matched goes back to the bettor, before showdown. |
| `Showdown { seat, hole, strength }` | `{"type": "showdown", seat, hole, strength}` | Hole cards of every player still in at showdown, with their hand strength (§6). |
| `PotAwarded { seat, amount }` | `{"type": "pot_awarded", seat, amount}` | Chips a seat collected (one event per winning seat, summed over main and side pots). |

Order within a hand: blinds and antes, then actions and streets interleaved as they happen,
then `uncalled_returned` (if any), all `showdown` events (if any), and `pot_awarded`.

### 3.6 Hook payloads

**`new_match(info)`**, once per table:

| Field | Meaning |
|---|---|
| `format` (Rust) / `variant`, `max_seats` (Python) | Game format. |
| `n_seats` | Seats at this table. |
| `seat` | Your **initial** seat. In duplicate mode seats rotate, so use `new_hand`/`obs.seat`. |
| `sb`, `bb`, `ante` | Blinds at the start (the first level in tournaments). |
| `starting_stack` | Chips. |
| `hands` | Planned hands at this table (an upper bound in tournaments). |
| `tournament`, `duplicate` | Mode flags. |

**`new_hand(info)`**, each hand you are dealt into: `hand_id` (counts from 0 within the
table), `seat`, `hole` (your cards), `button`, `stacks` (every seat's stack **before**
blinds and antes, 0 for empty seats), `sb`, `bb`, `ante`.

**`end_hand(result)`**: `hand_id`, `net` (chips won or lost by every seat this hand),
`stacks` (after the hand), `board` (the final visible board, empty if the hand ended
preflop), `showdown` (bool).

### 3.7 What you never see

The engine enforces information hiding. The table runner only ever hands a bot its own seat's
observation, and a test inspects every observation and event over many hands (NLHE, PLO
and duplicate mode) for leaked cards:

- Other players' hole cards, except in `showdown` events for players who reached showdown.
  Folded hands and uncontested winners are never shown.
- The deck, or board cards before they are dealt.
- Other bots' internal state, or anything from other tables.

## 4. What your bot returns

### 4.1 Actions

| Rust | Python | Meaning |
|---|---|---|
| `Action::Fold` | `Action.fold()` | Give up the hand. Always legal (even when you could check, though that throws away equity). |
| `Action::Check` | `Action.check()` | Legal only when `legal.can_check`. |
| `Action::Call` | `Action.call()` | Match `current_bet` (all-in if short). When there is nothing to call it counts as a check. |
| `Action::BetTo(x)` | `Action.bet_to(x)` | Bet so that your street total becomes `x`. |
| `Action::RaiseTo(x)` | `Action.raise_to(x)` | Raise so that your street total becomes `x`. `BetTo` and `RaiseTo` are interchangeable. |
| `Action::AllIn` | `Action.all_in()` | Put in your whole stack. Legal when that is a call, or a raise the structure allows. In pot-limit or fixed-limit a deep stack can't always shove. |

**Amounts are raise-to totals for the street, not increments.** To raise by 10 over a bet
of 6, return `RaiseTo(16)`, not `RaiseTo(10)`. The safe pattern is to compute a target and
clamp it:

```python
to = obs.current_bet + round(0.75 * (obs.pot + obs.legal.call_amount))   # 3/4 pot raise
return Action.raise_to(min(max(to, obs.legal.min_raise_to), obs.legal.max_raise_to))
```

A bet/raise is legal when `legal.can_raise and min_raise_to <= x <= max_raise_to`.

Python bots may also return a kind string (`"call"`, `"check"`, `"fold"`, `"all_in"`) or
a `(kind, amount)` tuple such as `("raise_to", 40)`.

### 4.2 Illegal actions and errors

The engine validates every action. With the default `illegal_action_policy =
"fold_or_check"`, an illegal action is replaced by a check (if possible) or a fold, and
counted in the leaderboard's **Illegal** column. With `"error"` the whole table aborts.
A good bot has 0 illegal actions; always check yours.

In Python, an exception raised in `act` counts as an illegal action. An exception in a hook
is ignored. In both cases the first traceback per bot instance goes to stderr, or to
`.poker_arena.log` when you use the desktop app.

## 5. Rules your bot should know

- **Blinds and dealing.** Antes first, then the blinds. Heads-up, the button posts the small
  blind. A player who can't cover a blind posts what they have and is all-in, and others
  still owe the full big blind to call.
- **Minimum raise.** The raise increment must be at least the largest full bet/raise
  increment so far on the street. Preflop the big blind counts as the opening bet, so the
  first raise must be to at least 2 BB. After the flop the minimum bet is 1 BB.
- **Short all-ins don't reopen betting.** An all-in smaller than a full raise doesn't let
  players who already acted re-raise, unless the total they now face since their last
  action is at least a full raise. You'll see this as `can_raise == False` with a call still
  available.
- **Pot limit (PLO).** Maximum raise-to = current bet + pot after you call. The best hand
  must use **exactly two** hole cards and three board cards.
- **Fixed limit (FLHE).** Bets and raises are one small bet preflop and on the flop, and one
  big bet on the turn and river. The cap is the bet plus three raises.
- **All-in runouts.** When at most one player can still bet, the remaining board is dealt
  without further action. You get the `street` events, but no `act` calls.
- **Uncalled bets** are returned before showdown. **Side pots** are formed from all-in
  levels, and each is won among the eligible players still in. **Ties** split, with odd
  chips going to the first winner left of the button.
- **No burn cards.** This has no strategic effect.

## 6. Helpers: equity and hand strength

**Monte Carlo equity** is your share of the pot against random opponent hands, with ties
split:

```python
from pokerarena import equity
e = equity(obs.hole, obs.board, n_opponents, n_samples=1000, seed=0, variant=obs.variant)
# cards may also be strings: equity("AsAh", "", 1, 20000)  ~ 0.852
```

```rust
use bots::equity::equity;
let e = equity(Evaluator::get(), obs.format.variant, obs.hole, obs.board, n_opponents, 1000, &mut rng);
```

It's deterministic for a given seed, allocation-free, and runs in Rust. In Python it also
releases the GIL. Measured cost with 200 samples: about 4 µs preflop heads-up, 2.5 µs on
the flop against one opponent, and 8 µs against five. It scales linearly with samples. The
first call in a process also builds the evaluator tables, a one-time cost of about 0.1 s.
Pass a seed from your own RNG (`self.rng.getrandbits(63)`) so runs stay reproducible.

**Hand strength** is a `u16` where higher is better, `(category + 1) << 12 | rank_within_category`.
Categories go 0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush, 6 full house,
7 quads, 8 straight flush. In the hand above, 8556 >> 12 = 2, so category 1 (one pair), and
12911 >> 12 = 3, so category 2 (two pair).

- **Python:** `evaluate("AsKsQsJsTs")` or `evaluate([51, 47, ...])` (5–7 Hold'em cards);
  `hand_category(strength)` returns, for example, `"Flush"`.
- **Rust:** `Evaluator::get().eval(&cards)` (5–7 cards, about 480M evals/s);
  `Evaluator::get().eval_omaha(&hole, &board)`; `engine::eval::category(s)`. For many
  evaluations against a fixed board, build a `HandAcc` from the board once and
  `.combine()` hole cards into it.

## 7. Rules of the road

**Determinism.** A run is reproducible bit for bit from (config, seed, bot set). Keep it
that way: take all randomness from the seed you are given (`Xoshiro256PlusPlus::seed_from_u64(seed)`,
`random.Random(seed)`). Never use `thread_rng()`, `random.random()` at module level,
wall-clock time or `os.urandom`. Don't depend on dict/hash iteration order that varies
between processes.

**No hand-specific state across hands.** Reset per-hand fields in `new_hand`. Learning
across hands (opponent statistics, adapting bet sizes) is fine in normal mode.

**Duplicate mode is stricter.** Each deal is replayed once per seat, with bots rotated
through the seats and the same cards. Two consequences:

- Your seat changes from hand to hand: read it from `new_hand`/`obs.seat`.
- Anything remembered from an earlier replay of the same deal reveals cards you shouldn't
  know: the board, other players' showdown hands, how the hand played out. The engine
  can't detect this. A bot that learns across hands must check `info["duplicate"]` (Python)
  or `info.duplicate` (Rust) in `new_match`, and in duplicate mode it must not use anything
  learned from previous hands.

**Tournaments.** Blinds rise by hand count, so always read `obs.sb`/`obs.bb`/`obs.ante`
rather than the values from `new_match`. Busted seats become `Empty`. Your stack carries
over between hands. Elo is based on finishing place.

**Cash stack policies.** With `reset_each_hand` (the default) every hand starts at the
starting stack, so you can't build a stack. With `rebuy_when_busted` or `top_up_to_start`,
stacks carry over: read `new_hand.stacks`.

**Performance.** There is no hard time limit, but your decision time is the arena's
throughput. Turn on `record_timing` and watch the **µs/decision** column. For reference:
Rust baselines take 0.01–2.6 µs per decision; a trivial Python bot about 1.1 µs, and the
Python tight-aggressive port about 5.5 µs. Python-specific tips:

- **Don't override hooks you don't need.** Every overridden hook is an extra interpreter
  call per event per seat. Hooks inherited from `BaseBot` are skipped entirely.
- `obs.history` is rebuilt on each `act`, so read what you need from it rather than
  copying it.
- Equity and evaluation calls are cheap, because they run in Rust.

**Where Python bots run.** Tables with any Python bot run in separate worker processes (a
`spawn` multiprocessing pool, one table per task). Consequences:

- **Your module must be importable by name** from the run's working directory. The desktop
  app runs from the repository root, so a file `mybots.py` there is `python:mybots.MyBot`.
  `python/examples/foo.py` is `python:examples.foo.Foo`, or install it as a package.
- **Module-level code runs once per worker.** Keep imports light.
- **Scripts that start runs** (`pokerarena.run`, `ArenaJob`) must use
  `if __name__ == "__main__":`.
- **Stop or crash.** The Stop button kills worker processes outright, so don't rely on
  cleanup code at the end of a table. If a bot kills its worker (a segfault, `os._exit`),
  the run fails with a clear error instead of hanging.

**Rust bots** must be `Send`, and they run on a thread pool, one table per task. They are
called sequentially within a table, so a bot needs no locks of its own.

## 8. Complete examples

Both bots implement the same strategy. They call when Monte Carlo equity beats the pot
odds, value-raise strong hands to two-thirds of the pot, and tighten up when the hand has
already been raised twice. They use `new_hand` and `observe` to track raises, which is
correct hand-scoped state.

### 8.1 Python (`python/examples/pot_odds_bot.py`)

```python
import random

from pokerarena import Action, BaseBot, equity


class PotOdds(BaseBot):
    def __init__(self, seed: int = 0, samples: int = 200):
        super().__init__(seed)
        self.rng = random.Random(seed)  # all randomness from the seed => reproducible runs
        self.samples = samples
        self.raises_this_hand = 0

    def new_hand(self, info):
        self.raises_this_hand = 0  # never carry hand state across hands

    def observe(self, event):
        if event["type"] == "action" and event["kind"] == "raise":
            self.raises_this_hand += 1

    def act(self, obs):
        la = obs.legal
        # Opponents still in the hand: status 1 (active) or 2 (all-in), excluding me.
        opponents = sum(1 for s, st in enumerate(obs.status) if s != obs.seat and st in (1, 2))
        e = equity(obs.hole, obs.board, opponents, self.samples,
                   seed=self.rng.getrandbits(63), variant=obs.variant)
        threshold = 0.85 if self.raises_this_hand >= 2 else 0.70
        if la.can_raise and e >= threshold:
            # Raise-to = current bet + 2/3 of the pot after calling, clamped to the legal range.
            to = obs.current_bet + (obs.pot + la.call_amount) * 2 // 3
            return Action.raise_to(min(max(to, la.min_raise_to), la.max_raise_to))
        if la.can_check:
            return Action.check()
        pot_odds = la.call_amount / (obs.pot + la.call_amount)
        return Action.call() if e >= pot_odds else Action.fold()
```

Use it:

```toml
[[bots]]
name = "pot_odds"
kind = "python:examples.pot_odds_bot.PotOdds"
params = { samples = 200 }
```

In the desktop app, add a player, choose **Custom Python class…**, and enter
`examples.pot_odds_bot.PotOdds`.

### 8.2 Rust (`crates/bots/tests/guide_example.rs`)

```rust
use std::sync::Arc;

use arena::bot::{Action, ActionKind, Bot, BotFactory, HandStart, Observation, PublicEvent};
use bots::equity::equity;
use engine::eval::Evaluator;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub struct PotOdds {
    name: String,
    samples: u32,
    rng: Xoshiro256PlusPlus,
    ev: Evaluator,
    raises_this_hand: u32, // reset in new_hand: never carry hand state across hands
}

impl PotOdds {
    pub fn new(name: &str, samples: u32, seed: u64) -> Self {
        PotOdds {
            name: name.to_string(),
            samples,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed), // all randomness from the seed
            ev: Evaluator::get(),
            raises_this_hand: 0,
        }
    }

    /// A factory: the arena calls it once per seat per table with a derived seed.
    pub fn factory(name: &str, samples: u32) -> BotFactory {
        let name = name.to_string();
        Arc::new(move |seed| Box::new(PotOdds::new(&name, samples, seed)))
    }
}

impl Bot for PotOdds {
    fn name(&self) -> &str {
        &self.name
    }

    fn new_hand(&mut self, _info: &HandStart) {
        self.raises_this_hand = 0;
    }

    fn observe(&mut self, event: &PublicEvent) {
        if let PublicEvent::Action(rec) = event {
            if rec.kind == ActionKind::Raise {
                self.raises_this_hand += 1;
            }
        }
    }

    fn act(&mut self, obs: &Observation) -> Action {
        let la = obs.legal;
        // Opponents still in the hand (status Active or AllIn, excluding me).
        let opponents = obs
            .status
            .iter()
            .enumerate()
            .filter(|&(s, st)| s != obs.seat && st.is_live())
            .count();
        let e = equity(self.ev, obs.format.variant, obs.hole, obs.board, opponents,
                       self.samples, &mut self.rng);
        let threshold = if self.raises_this_hand >= 2 { 0.85 } else { 0.70 };
        if la.can_raise && e >= threshold {
            // Raise-to = current bet + 2/3 of the pot after calling, clamped to the legal range.
            let to = obs.current_bet + (obs.pot + la.call_amount) * 2 / 3;
            return Action::RaiseTo(to.clamp(la.min_raise_to, la.max_raise_to));
        }
        if la.can_check {
            return Action::Check;
        }
        let pot_odds = la.call_amount as f64 / (obs.pot + la.call_amount) as f64;
        if e >= pot_odds { Action::Call } else { Action::Fold }
    }
}
```

**Run it from your own program** with `arena::run`:

```rust
let bots = vec![
    BotEntry { name: "pot_odds".into(), version: None, factory: PotOdds::factory("pot_odds", 200) },
    BotEntry { name: "tag".into(), version: None,
               factory: bots::factory("tag", "tag", &serde_json::Value::Null).unwrap() },
];
let spec = RunSpec { format: GameFormat::nlhe(6), table: TableConfig::cash(1000, 1, 2, 100),
                     matchmaking: Default::default(), rating: Default::default(),
                     rounds: 5, seed: 1, threads: None };
let out = arena::run(&spec, &bots, vec![1500.0; bots.len()], false)?;
print!("{}", arena::leaderboard::render(&out.state.rows()));
```

**Or register it** so configs, the `arena` CLI and Python can use `kind = "pot_odds"`. Put
the bot in `crates/bots/src/`, then add an arm to `bots::factory` and its name to
`bots::KINDS` in `crates/bots/src/lib.rs`:

```rust
"pot_odds" => {
    check_keys(params, &["samples"])?;
    let samples = get_f64(params, "samples", 200.0)? as u32;
    Ok(PotOdds::factory(name, samples))
}
```

To list it in the desktop app's **Bot** menu, add it to `BOT_CHOICES` (and
`DEFAULT_PARAMS`) in `python/pokerarena/gui.py`, then rebuild the extension
(`maturin develop --release`).

## 9. Testing and evaluating your bot

**Smoke-test it against the baselines and check the Illegal column is 0:**

```python
import pokerarena as pa

if __name__ == "__main__":
    res = pa.run({
        "run": {"rounds": 3, "seed": 1},
        "format": {"variant": "nlhe", "max_seats": 6},
        "table": {"hands_per_table": 2000, "blinds": {"sb": 1, "bb": 2}, "record_timing": True},
        "bots": [
            {"name": "mine", "kind": "python:mybots.MyBot"},
            {"name": "tag", "kind": "tag"},
            {"name": "random", "kind": "random"},
            {"name": "station", "kind": "calling_station"},
        ],
    }, save=False)
    print(res.table)
```

Or use the desktop app: load a preset, add your bot, and press Run.

**Unit-test single decisions (Rust)** on a hand you set up yourself:

```rust
use engine::{DeckBuilder, GameFormat, HandSetup, HandState, MAX_SEATS};

let fmt = GameFormat::nlhe(6);
let mut stacks = [0u64; MAX_SEATS];
stacks[..2].fill(200); // two players, 100 bb each
let setup = HandSetup { n_seats: 2, button: 0, stacks, sb: 1, bb: 2, ante: 0 };
// Deal order starts left of the button: seat 1 then seat 0.
let deck = DeckBuilder::new().hole(0, "AsAh").hole(1, "7c2d").build(&[1, 0], 2);
let mut st = HandState::new();
st.start(&fmt, &setup, &deck).unwrap();

let mut bot = PotOdds::new("pot_odds", 500, 42);
let seat = st.to_act().unwrap(); // heads-up, the button (seat 0) acts first preflop
let action = bot.act(&st.observation(seat));
assert!(st.validate(action).is_ok(), "bot chose an illegal action: {action:?}");
```

You can keep calling `st.apply(action)` and `st.observation(st.to_act().unwrap())` to
script a whole hand.

**Evaluate it properly.** One run of a few thousand hands is noise: poker variance is large.
Use the bb/100 confidence interval in the leaderboard. Run more hands per table and more
rounds until the CI is smaller than the effect you care about, and use **duplicate mode**
(cash, `reset_each_hand`) to cancel much of the card luck. Check determinism by running the
same config twice; results must be identical.

## 10. Checklist

- [ ] `act` returns only actions that `obs.legal` allows. Bets and raises use raise-to
      totals, clamped to `[min_raise_to, max_raise_to]`.
- [ ] Illegal column is 0 after a few thousand hands in each variant you support
      (NLHE/PLO/FLHE).
- [ ] All randomness comes from the given seed, and two identical runs give identical
      results.
- [ ] Per-hand state is reset in `new_hand`. Your seat comes from `new_hand`/`obs.seat`,
      not `new_match`.
- [ ] In duplicate mode (`new_match` `duplicate`), nothing learned from previous hands is
      used.
- [ ] Blinds are read per hand (`obs.bb`), empty seats (status 0) are skipped, and stacks
      are read in bb.
- [ ] Constructor (Python) or factory (Rust) is cheap and free of side effects. Shared data
      is loaded once.
- [ ] Python: only the hooks you need are overridden, and the module is importable from the
      run's working directory.
- [ ] Decision time checked with `record_timing`.
