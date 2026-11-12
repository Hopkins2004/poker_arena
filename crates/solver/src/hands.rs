//! Hand sets: the private hands a solve runs over.
//!
//! A [`HandSet`] is a list of hands **sorted by strength, ascending**, so tie groups are
//! contiguous index ranges and showdown sweeps are linear passes. Each hand optionally has a
//! card pair, used for card removal (two hands are compatible iff they share no card).
//! Without cards, every pair of hands is compatible (abstract toy games).
//!
//! The river adapter [`RiverHands`] builds the hand set of a 5-card board: the valid combos
//! (those not using a board card), with the map back to the global 1326-combo index
//! ([`combo_index`], the same order as `pokerarena.preflop.COMBOS`).

use crate::terminal::TermIndex;
use engine::{Card, Evaluator, HandAcc};

/// Number of two-card combos.
pub const N_COMBOS: usize = 1326;

/// Global index of the combo `{a, b}` (cards `0..52`, `a != b`): `hi*(hi-1)/2 + lo`.
#[inline]
pub fn combo_index(a: u8, b: u8) -> usize {
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    debug_assert!(hi != lo && hi < 52);
    hi as usize * (hi as usize - 1) / 2 + lo as usize
}

/// Cards `(hi, lo)` of every combo, by global index.
pub fn combo_cards() -> &'static [[u8; 2]; N_COMBOS] {
    static TABLE: std::sync::OnceLock<[[u8; 2]; N_COMBOS]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [[0u8; 2]; N_COMBOS];
        for hi in 1..52u8 {
            for lo in 0..hi {
                t[combo_index(hi, lo)] = [hi, lo];
            }
        }
        t
    })
}

/// A set of private hands sorted by strength (ascending).
#[derive(Clone, Debug, PartialEq)]
pub struct HandSet {
    /// Strength per hand (non-decreasing). Only the order and equality matter.
    pub strength: Vec<u32>,
    /// Card pair per hand, or `None` to switch card removal off.
    pub cards: Option<Vec<[u8; 2]>>,
    /// Tie groups: `[start, end)` index ranges of equal strength, in ascending order.
    pub groups: Vec<(u32, u32)>,
    /// Precomputed gather indices for O(n) terminal evaluation.
    pub index: TermIndex,
}

impl HandSet {
    /// Builds a hand set from hands in any order. Returns the set and `order`, where
    /// `order[i]` is the input index of sorted hand `i` (stable for equal strengths).
    ///
    /// With cards, card pairs must be distinct (each combo at most once) and use cards
    /// `0..52`.
    pub fn new(strength: &[u32], cards: Option<&[[u8; 2]]>) -> (HandSet, Vec<usize>) {
        let n = strength.len();
        if let Some(c) = cards {
            assert_eq!(c.len(), n, "one card pair per hand");
            let mut seen = std::collections::HashSet::new();
            for p in c {
                assert!(
                    p[0] < 52 && p[1] < 52 && p[0] != p[1],
                    "bad card pair {p:?}"
                );
                assert!(
                    seen.insert(combo_index(p[0], p[1])),
                    "duplicate card pair {p:?}"
                );
            }
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| strength[i]);
        let strength: Vec<u32> = order.iter().map(|&i| strength[i]).collect();
        let cards = cards.map(|c| order.iter().map(|&i| c[i]).collect());
        let mut groups = Vec::new();
        let mut start = 0usize;
        for i in 1..=n {
            if i == n || strength[i] != strength[start] {
                groups.push((start as u32, i as u32));
                start = i;
            }
        }
        let index = TermIndex::new(n, &groups, cards.as_deref());
        (
            HandSet {
                strength,
                cards,
                groups,
                index,
            },
            order,
        )
    }

    /// Number of hands.
    #[inline]
    pub fn len(&self) -> usize {
        self.strength.len()
    }

    /// True if there are no hands.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.strength.is_empty()
    }

    /// True if hands `i` and `j` share no card (always true without card removal).
    #[inline]
    pub fn compatible(&self, i: usize, j: usize) -> bool {
        match &self.cards {
            None => true,
            Some(c) => {
                let (a, b) = (c[i], c[j]);
                a[0] != b[0] && a[0] != b[1] && a[1] != b[0] && a[1] != b[1]
            }
        }
    }
}

/// The hand set of a river board, plus the map to global combo indices.
#[derive(Clone, Debug)]
pub struct RiverHands {
    /// Valid combos, sorted by strength.
    pub set: HandSet,
    /// Global combo index of each sorted hand.
    pub combo: Vec<u16>,
    /// Strength of every global combo (0 for combos blocked by the board).
    pub strength_1326: Vec<u16>,
}

impl RiverHands {
    /// Hands for a 5-card board (cards must be distinct).
    pub fn new(board: &[Card; 5]) -> RiverHands {
        let ev = Evaluator::get();
        let acc = HandAcc::from_cards(board);
        let mut dead = 0u64;
        for c in board {
            assert!(dead & (1 << c.0) == 0, "duplicate board card");
            dead |= 1 << c.0;
        }
        let table = combo_cards();
        let mut strength_1326 = vec![0u16; N_COMBOS];
        let mut idx = Vec::with_capacity(N_COMBOS);
        let mut st = Vec::with_capacity(N_COMBOS);
        let mut cards = Vec::with_capacity(N_COMBOS);
        for (i, &[a, b]) in table.iter().enumerate() {
            if dead & (1 << a) != 0 || dead & (1 << b) != 0 {
                continue;
            }
            let s = ev.eval_acc(acc.with(Card(a)).with(Card(b)));
            strength_1326[i] = s;
            idx.push(i as u16);
            st.push(s as u32);
            cards.push([a, b]);
        }
        let (set, order) = HandSet::new(&st, Some(&cards));
        let combo = order.iter().map(|&k| idx[k]).collect();
        RiverHands {
            set,
            combo,
            strength_1326,
        }
    }

    /// Gathers a 1326-combo vector into sorted hand order.
    pub fn gather(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), N_COMBOS);
        self.combo.iter().map(|&c| v[c as usize]).collect()
    }

    /// Scatters a hand-order vector back to 1326 combos (0 on blocked combos).
    pub fn scatter(&self, v: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; N_COMBOS];
        for (&c, &x) in self.combo.iter().zip(v) {
            out[c as usize] = x;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combo_index_matches_python_order() {
        // pokerarena.preflop: [(hi, lo) for hi in range(52) for lo in range(hi)]
        let mut k = 0;
        for hi in 0..52u8 {
            for lo in 0..hi {
                assert_eq!(combo_index(hi, lo), k);
                assert_eq!(combo_index(lo, hi), k);
                assert_eq!(combo_cards()[k], [hi, lo]);
                k += 1;
            }
        }
        assert_eq!(k, N_COMBOS);
    }

    #[test]
    fn river_hands_are_sorted_and_grouped() {
        let board = engine::card::parse_cards("AsKsQsJsTs").unwrap();
        let rh = RiverHands::new(&board.try_into().unwrap());
        assert_eq!(rh.set.len(), 1081); // C(47, 2)
        assert!(rh.set.strength.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(
            rh.set.groups.len(),
            1,
            "royal flush on board: everyone ties"
        );
        let gathered = rh.gather(&vec![1.0; N_COMBOS]);
        assert_eq!(rh.scatter(&gathered).iter().sum::<f32>(), 1081.0);
    }
}
