//! Cards, ranks, suits and card bitmasks.
//!
//! A [`Card`] is a `u8` in `0..52` encoded as `rank * 4 + suit`, where rank `0` is a deuce and
//! rank `12` is an ace, and suits are ordered clubs, diamonds, hearts, spades.

use std::fmt;
use std::str::FromStr;

/// Rank characters indexed by rank (`0` = deuce ... `12` = ace).
pub const RANK_CHARS: [char; 13] = [
    '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
];
/// Suit characters indexed by suit (`c`, `d`, `h`, `s`).
pub const SUIT_CHARS: [char; 4] = ['c', 'd', 'h', 's'];

/// A playing card, `rank * 4 + suit`, always in `0..52`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Card(pub u8);

impl Card {
    /// Builds a card from a rank (`0..13`, deuce = 0) and a suit (`0..4`).
    #[inline]
    pub const fn new(rank: u8, suit: u8) -> Card {
        debug_assert!(rank < 13 && suit < 4);
        Card(rank * 4 + suit)
    }
    /// Rank in `0..13` (deuce = 0, ace = 12).
    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 >> 2
    }
    /// Suit in `0..4` (c, d, h, s).
    #[inline]
    pub const fn suit(self) -> u8 {
        self.0 & 3
    }
    /// Index in `0..52`.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
    /// Single-bit mask for this card.
    #[inline]
    pub const fn mask(self) -> u64 {
        1u64 << self.0
    }
}

/// Error returned when parsing a card or a card list fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseCardError(pub String);

impl fmt::Display for ParseCardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid card: {:?}", self.0)
    }
}
impl std::error::Error for ParseCardError {}

impl FromStr for Card {
    type Err = ParseCardError;
    fn from_str(s: &str) -> Result<Card, ParseCardError> {
        let mut it = s.chars();
        let (Some(r), Some(su), None) = (it.next(), it.next(), it.next()) else {
            return Err(ParseCardError(s.to_string()));
        };
        let r = r.to_ascii_uppercase();
        let su = su.to_ascii_lowercase();
        let rank = RANK_CHARS.iter().position(|&c| c == r);
        let suit = SUIT_CHARS.iter().position(|&c| c == su);
        match (rank, suit) {
            (Some(r), Some(s)) => Ok(Card::new(r as u8, s as u8)),
            _ => Err(ParseCardError(s.to_string())),
        }
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            RANK_CHARS[self.rank() as usize],
            SUIT_CHARS[self.suit() as usize]
        )
    }
}

impl fmt::Debug for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Parses a whitespace-optional card list such as `"AsKd"` or `"As Kd 7c"`.
pub fn parse_cards(s: &str) -> Result<Vec<Card>, ParseCardError> {
    let compact: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if !compact.len().is_multiple_of(2) {
        return Err(ParseCardError(s.to_string()));
    }
    compact
        .as_bytes()
        .chunks(2)
        .map(|ch| std::str::from_utf8(ch).unwrap().parse())
        .collect()
}

/// Formats a card slice as `"AsKd7c"`.
pub fn format_cards(cards: &[Card]) -> String {
    cards.iter().map(|c| c.to_string()).collect()
}

/// A set of cards as a 52-bit mask (bit `i` = card `i`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub struct CardMask(pub u64);

impl CardMask {
    /// Empty set.
    pub const EMPTY: CardMask = CardMask(0);
    /// Inserts a card.
    #[inline]
    pub fn insert(&mut self, c: Card) {
        self.0 |= c.mask();
    }
    /// True if the card is in the set.
    #[inline]
    pub fn contains(self, c: Card) -> bool {
        self.0 & c.mask() != 0
    }
    /// Number of cards in the set.
    #[inline]
    pub fn len(self) -> u32 {
        self.0.count_ones()
    }
    /// True if empty.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    /// Builds a mask from a slice.
    pub fn from_cards(cards: &[Card]) -> CardMask {
        CardMask(cards.iter().fold(0, |m, c| m | c.mask()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_all_cards() {
        for i in 0..52u8 {
            let c = Card(i);
            let s = c.to_string();
            assert_eq!(s.parse::<Card>().unwrap(), c);
        }
        assert_eq!("As".parse::<Card>().unwrap(), Card::new(12, 3));
        assert_eq!("Td".parse::<Card>().unwrap(), Card::new(8, 1));
        assert_eq!("2c".parse::<Card>().unwrap(), Card(0));
        assert_eq!("ah".parse::<Card>().unwrap(), Card::new(12, 2));
    }

    #[test]
    fn parse_errors() {
        assert!("Xx".parse::<Card>().is_err());
        assert!("A".parse::<Card>().is_err());
        assert!("Asd".parse::<Card>().is_err());
        assert!(parse_cards("AsK").is_err());
    }

    #[test]
    fn parse_list() {
        let v = parse_cards("As Kd 7c").unwrap();
        assert_eq!(format_cards(&v), "AsKd7c");
        assert_eq!(CardMask::from_cards(&v).len(), 3);
    }
}
