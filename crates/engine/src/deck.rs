//! A 52-card deck with seeded Fisher–Yates shuffling and support for pre-arranged decks.
//!
//! **Deal convention** (used by [`crate::state::HandState`] and by [`DeckBuilder`]): hole cards
//! are dealt one at a time, round-robin, starting with the first seat in the hand to the left of
//! the button; then the five board cards in order (flop, turn, river). No burn cards are used.

use crate::card::{Card, CardMask, ParseCardError};
use crate::rng::below;
use rand_core::Rng;

/// A deck of 52 cards. The order of `cards` is the deal order.
#[derive(Clone, Debug)]
pub struct Deck {
    cards: [Card; 52],
}

impl Default for Deck {
    fn default() -> Self {
        Self::new()
    }
}

impl Deck {
    /// A deck in canonical order (2c, 2d, 2h, 2s, 3c, ...).
    pub fn new() -> Deck {
        let mut cards = [Card(0); 52];
        for (i, c) in cards.iter_mut().enumerate() {
            *c = Card(i as u8);
        }
        Deck { cards }
    }

    /// Full Fisher–Yates shuffle.
    pub fn shuffle<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        self.shuffle_prefix(rng, 52);
    }

    /// Shuffles only the first `m` positions: they become a uniformly random ordered sample
    /// of the deck, which is all a hand needs. Cheaper than a full shuffle.
    #[inline]
    pub fn shuffle_prefix<R: Rng + ?Sized>(&mut self, rng: &mut R, m: usize) {
        let m = m.min(51);
        for i in 0..m {
            let j = i + below(rng, (52 - i) as u32) as usize;
            self.cards.swap(i, j);
        }
    }

    /// Builds a deck whose first cards are `prefix` (in deal order); the remaining cards follow
    /// in canonical order. Panics if `prefix` has duplicates.
    pub fn from_order(prefix: &[Card]) -> Deck {
        let mut used = CardMask::EMPTY;
        let mut cards = [Card(0); 52];
        for (i, &c) in prefix.iter().enumerate() {
            assert!(!used.contains(c), "duplicate card {c} in pre-arranged deck");
            used.insert(c);
            cards[i] = c;
        }
        let mut k = prefix.len();
        for i in 0..52u8 {
            if !used.contains(Card(i)) {
                cards[k] = Card(i);
                k += 1;
            }
        }
        Deck { cards }
    }

    /// The card at deal position `i`.
    #[inline]
    pub fn at(&self, i: usize) -> Card {
        self.cards[i]
    }

    /// All cards in deal order.
    pub fn cards(&self) -> &[Card; 52] {
        &self.cards
    }
}

/// Builds a pre-arranged deck from per-seat hole cards and a board, following the deal
/// convention described in the module docs. Seats without specified cards get unused cards.
#[derive(Clone, Debug, Default)]
pub struct DeckBuilder {
    holes: Vec<(usize, Vec<Card>)>,
    board: Vec<Card>,
}

impl DeckBuilder {
    /// Empty builder.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the hole cards of `seat`, e.g. `.hole(0, "AsKs")`.
    pub fn hole(mut self, seat: usize, cards: &str) -> Self {
        self.holes.push((
            seat,
            crate::card::parse_cards(cards).expect("bad hole cards"),
        ));
        self
    }
    /// Sets the board (up to five cards), e.g. `.board("2c7d9hTsJs")`.
    pub fn board(mut self, cards: &str) -> Self {
        self.board = crate::card::parse_cards(cards).expect("bad board");
        self
    }

    /// Fallible version of [`DeckBuilder::hole`].
    pub fn try_hole(mut self, seat: usize, cards: &str) -> Result<Self, ParseCardError> {
        self.holes.push((seat, crate::card::parse_cards(cards)?));
        Ok(self)
    }

    /// Produces the deck for a hand with `hole_cards` cards per player where the seats in the
    /// hand, in deal order (starting left of the button), are `deal_order`.
    pub fn build(&self, deal_order: &[usize], hole_cards: usize) -> Deck {
        let n = deal_order.len();
        let total = n * hole_cards + 5;
        let mut slots: Vec<Option<Card>> = vec![None; total];
        for (seat, cards) in &self.holes {
            let pos = deal_order
                .iter()
                .position(|s| s == seat)
                .unwrap_or_else(|| panic!("seat {seat} not in hand"));
            assert!(cards.len() <= hole_cards);
            for (k, &c) in cards.iter().enumerate() {
                slots[k * n + pos] = Some(c);
            }
        }
        for (k, &c) in self.board.iter().enumerate() {
            slots[n * hole_cards + k] = Some(c);
        }
        let mut used = CardMask::EMPTY;
        for c in slots.iter().flatten() {
            assert!(!used.contains(*c), "duplicate card {c}");
            used.insert(*c);
        }
        let mut filler = (0..52u8).map(Card).filter(|c| !used.contains(*c));
        let prefix: Vec<Card> = slots
            .into_iter()
            .map(|s| s.unwrap_or_else(|| filler.next().unwrap()))
            .collect();
        Deck::from_order(&prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn shuffle_is_permutation_and_deterministic() {
        let mut a = Deck::new();
        let mut b = Deck::new();
        a.shuffle(&mut Xoshiro256PlusPlus::seed_from_u64(42));
        b.shuffle(&mut Xoshiro256PlusPlus::seed_from_u64(42));
        assert_eq!(a.cards(), b.cards());
        let m = CardMask::from_cards(a.cards());
        assert_eq!(m.len(), 52);
        assert_ne!(a.cards(), Deck::new().cards());
    }

    #[test]
    fn from_order_prefix() {
        let p = crate::card::parse_cards("AsKsQs").unwrap();
        let d = Deck::from_order(&p);
        assert_eq!(&d.cards()[..3], &p[..]);
        assert_eq!(CardMask::from_cards(d.cards()).len(), 52);
    }

    #[test]
    fn builder_layout() {
        let d = DeckBuilder::new()
            .hole(2, "AsKs")
            .hole(0, "2c2d")
            .board("3h4h5h6h7h")
            .build(&[2, 0], 2);
        let s = crate::card::format_cards(&d.cards()[..9]);
        assert_eq!(s, "As2cKs2d3h4h5h6h7h");
    }
}
