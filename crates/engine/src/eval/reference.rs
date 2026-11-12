//! Slow, obviously-correct reference evaluator.
//!
//! Used to build the fast lookup tables and to cross-check them in tests. Values are `u32`:
//! `category << 20 | r0 << 16 | r1 << 12 | r2 << 8 | r3 << 4 | r4`, where `r0..r4` are the
//! tie-break ranks in significance order. Larger is better.

use crate::card::Card;

/// Hand categories, weakest first. The discriminant is used in [`super::Strength`] encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum HandCategory {
    /// No pair.
    HighCard = 0,
    /// One pair.
    OnePair = 1,
    /// Two pair.
    TwoPair = 2,
    /// Three of a kind.
    ThreeOfAKind = 3,
    /// Five consecutive ranks (A-2-3-4-5 is the lowest).
    Straight = 4,
    /// Five cards of one suit.
    Flush = 5,
    /// Three of a kind plus a pair.
    FullHouse = 6,
    /// Four of a kind.
    FourOfAKind = 7,
    /// Straight flush, including the royal flush.
    StraightFlush = 8,
}

impl HandCategory {
    /// All categories, weakest first.
    pub const ALL: [HandCategory; 9] = [
        HandCategory::HighCard,
        HandCategory::OnePair,
        HandCategory::TwoPair,
        HandCategory::ThreeOfAKind,
        HandCategory::Straight,
        HandCategory::Flush,
        HandCategory::FullHouse,
        HandCategory::FourOfAKind,
        HandCategory::StraightFlush,
    ];
    /// Category from its discriminant.
    pub fn from_u8(v: u8) -> HandCategory {
        Self::ALL[v as usize]
    }
}

/// Evaluates exactly five ranks (with multiplicity) plus a flag saying whether all five cards
/// share a suit. Ranks are `0..13` (deuce = 0).
pub fn eval5_ranks(ranks: [u8; 5], flush: bool) -> u32 {
    let mut counts = [0u8; 13];
    for &r in &ranks {
        counts[r as usize] += 1;
    }
    // Tie-break ranks ordered by (count desc, rank desc); `g0`/`g1` are the two largest
    // group sizes.
    let mut kick = [0u8; 5];
    let mut nk = 0;
    let (mut g0, mut g1) = (0u8, 0u8);
    for c in (1..=4u8).rev() {
        for r in (0..13u8).rev() {
            if counts[r as usize] == c {
                kick[nk] = r;
                if nk == 0 {
                    g0 = c;
                } else if nk == 1 {
                    g1 = c;
                }
                nk += 1;
            }
        }
    }

    let mut straight_high: Option<u8> = None;
    if nk == 5 {
        let (hi, lo) = (kick[0], kick[4]);
        if hi - lo == 4 {
            straight_high = Some(hi);
        } else if hi == 12 && kick[1] == 3 && lo == 0 {
            // Wheel: A-5-4-3-2 plays as a five-high straight.
            straight_high = Some(3);
        }
    }

    let pack = |cat: HandCategory, rs: &[u8]| -> u32 {
        let mut v = (cat as u32) << 20;
        for (i, &r) in rs.iter().enumerate() {
            v |= (r as u32) << (16 - 4 * i);
        }
        v
    };
    let kick = &kick[..nk];

    if let Some(h) = straight_high {
        let cat = if flush {
            HandCategory::StraightFlush
        } else {
            HandCategory::Straight
        };
        return pack(cat, &[h]);
    }
    if flush {
        return pack(HandCategory::Flush, kick);
    }
    let cat = match (g0, g1) {
        (4, _) => HandCategory::FourOfAKind,
        (3, 2) => HandCategory::FullHouse,
        (3, _) => HandCategory::ThreeOfAKind,
        (2, 2) => HandCategory::TwoPair,
        (2, _) => HandCategory::OnePair,
        _ => HandCategory::HighCard,
    };
    pack(cat, kick)
}

/// Evaluates five cards.
pub fn eval5(cards: [Card; 5]) -> u32 {
    let flush = cards.iter().all(|c| c.suit() == cards[0].suit());
    eval5_ranks(cards.map(|c| c.rank()), flush)
}

/// Best five-card value among all 5-subsets of `cards` (5 to 7 cards).
pub fn eval_best(cards: &[Card]) -> u32 {
    let n = cards.len();
    assert!((5..=7).contains(&n));
    let mut best = 0;
    for_each_5_subset(n, |idx| {
        best = best.max(eval5(idx.map(|i| cards[i])));
    });
    best
}

/// Best Omaha value: exactly two of `hole` and exactly three of `board`.
pub fn eval_omaha_best(hole: &[Card], board: &[Card]) -> u32 {
    let mut best = 0;
    for a in 0..hole.len() {
        for b in a + 1..hole.len() {
            for x in 0..board.len() {
                for y in x + 1..board.len() {
                    for z in y + 1..board.len() {
                        best = best.max(eval5([hole[a], hole[b], board[x], board[y], board[z]]));
                    }
                }
            }
        }
    }
    best
}

/// Calls `f` with every 5-element index subset of `0..n`.
pub fn for_each_5_subset(n: usize, mut f: impl FnMut([usize; 5])) {
    for a in 0..n {
        for b in a + 1..n {
            for c in b + 1..n {
                for d in c + 1..n {
                    for e in d + 1..n {
                        f([a, b, c, d, e]);
                    }
                }
            }
        }
    }
}

/// Category of a reference value.
pub fn category_of(v: u32) -> HandCategory {
    HandCategory::from_u8((v >> 20) as u8)
}
