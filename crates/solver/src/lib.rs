//! Exact river solver: vectorized public-tree DCFR over all private hands.
//!
//! # Value convention
//!
//! Amounts are in units of the pot at the start of the river (`P = 1`); `stack` is the
//! effective stack behind in the same units (= SPR). Player 0 ("player 1" in the docs of
//! `river_net`) is out of position and acts first. For a deal `(h0, h1)` ending at a terminal
//! where player `p` put `c_p` in on the river:
//!
//! * fold by `f`: `u_f = −½ − c_f`, the other player gets `½ + c_f`;
//! * showdown (`c0 = c1 = c`): winner `½ + c`, loser `−½ − c`, tie `0`.
//!
//! i.e. `u_p = (chips collected from the final pot − chips put in on the river) − ½`, and
//! `u0 + u1 = 0` at every terminal. With card compatibility `c(h, h')` and ranges `r_p`:
//!
//! * `m_p(h) = Σ_{h'} c(h,h') r_{1−p}(h')` — compatible opponent mass;
//! * `ṽ_p(h) = Σ_{h'} c(h,h') r_{1−p}(h') E[u_p | h, h']` — counterfactual value;
//! * `v_p(h) = ṽ_p(h) / m_p(h)` — conditional value (the network's target), defined for
//!   every valid hand, including hands outside the range;
//! * zero-sum identity: `Σ_h r0(h) ṽ0(h) + Σ_h r1(h) ṽ1(h) = 0`.
//!
//! # Layout
//!
//! [`hands`] (sorted hand sets, river adapter), [`tree`] (bet abstraction, flat public
//! tree), [`terminal`] (O(n) fold / showdown evaluation), [`dcfr`] (the solver),
//! [`equity`] (range-vs-range equity used by the range generator).

#![warn(missing_docs)]

pub mod dcfr;
pub mod equity;
pub mod hands;
pub mod terminal;
pub mod tree;

pub use dcfr::{Params, Solution, Solver};
pub use hands::{combo_cards, combo_index, HandSet, RiverHands, N_COMBOS};
pub use tree::{Abstraction, Edge, NodeKind, Tree};

use engine::Card;

/// A solved river spot: the hand set, tree and solution (vectors in sorted hand order; use
/// [`RiverHands::scatter`] for 1326-combo vectors).
pub struct RiverSolve {
    /// Hands of the board.
    pub hands: RiverHands,
    /// The betting tree.
    pub tree: Tree,
    /// Strategies and values.
    pub solution: Solution,
}

/// Errors in a river spot.
#[derive(Debug, Clone, PartialEq)]
pub struct SpotError(pub String);

impl std::fmt::Display for SpotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid river spot: {}", self.0)
    }
}
impl std::error::Error for SpotError {}

/// Solves a river spot. Ranges are 1326-combo weights; blocked combos are ignored and each
/// range is renormalised to sum to 1 over the valid combos.
pub fn solve_river(
    board: &[Card; 5],
    stack: f32,
    ranges: [&[f32]; 2],
    abs: &Abstraction,
    params: &Params,
) -> Result<RiverSolve, SpotError> {
    if !(stack.is_finite() && stack >= 0.0) {
        return Err(SpotError(format!("stack {stack}")));
    }
    for r in ranges {
        if r.len() != N_COMBOS {
            return Err(SpotError(format!("range of length {}", r.len())));
        }
        if r.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(SpotError("negative or non-finite range weight".into()));
        }
    }
    let mut seen = 0u64;
    for c in board {
        if c.0 >= 52 || seen & (1 << c.0) != 0 {
            return Err(SpotError("board cards must be distinct cards 0..52".into()));
        }
        seen |= 1 << c.0;
    }
    let hands = RiverHands::new(board);
    let mut r = [hands.gather(ranges[0]), hands.gather(ranges[1])];
    for (p, v) in r.iter_mut().enumerate() {
        let s: f64 = v.iter().map(|&x| x as f64).sum();
        if s <= 0.0 {
            return Err(SpotError(format!("range {p} has no mass on valid combos")));
        }
        for x in v.iter_mut() {
            *x = (*x as f64 / s) as f32;
        }
    }
    let tree = Tree::river(stack, abs);
    let solution = Solver::new(&tree, &hands.set, [&r[0], &r[1]]).solve(params);
    Ok(RiverSolve {
        hands,
        tree,
        solution,
    })
}
