//! Lookup tables for the fast evaluator, built once at first use.
//!
//! Scheme (after OMPEval): each card contributes an additive key. The low 32 bits are a rank
//! key chosen so that the sum over any multiset of at most 7 ranks (≤ 4 per rank) is unique;
//! the high bits are four 4-bit suit counters. Starting each counter at 3, a counter reaching
//! 8 (bit 3 set) means five or more cards of that suit, i.e. a flush. Separately, a 64-bit
//! mask with 16 bits per suit records which ranks are present in each suit.
//!
//! * Non-flush: the rank-key sum is mapped through a displacement perfect hash into a `u16`
//!   table of strengths.
//! * Flush: the 13-bit rank mask of the flush suit indexes an 8192-entry table. With at most
//!   7 cards a flush always beats anything else that the same cards could make (quads or a
//!   full house plus a five-card flush would need at least 8 cards), so the flush table alone
//!   decides those hands.

use super::reference::{self, HandCategory};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Per-rank additive keys (deuce first). All sums of ≤ 7 ranks with ≤ 4 copies are distinct;
/// verified by `rank_keys_unique` below.
pub const RANK_KEYS: [u32; 13] = [
    0x2000, 0x8001, 0x11000, 0x3a000, 0x91000, 0x176005, 0x366000, 0x41a013, 0x47802e, 0x479068,
    0x48c0e4, 0x48f211, 0x494493,
];

/// Bit offset of the suit counters inside a key.
pub const SUIT_SHIFT: u32 = 32;
/// Initial bias of each 4-bit suit counter (3, so that 5 cards reach 8).
pub const SUIT_BIAS: u64 = 0x3333 << SUIT_SHIFT;
/// Flag bits: bit 3 of each suit counter.
pub const FLUSH_FLAGS: u64 = 0x8888 << SUIT_SHIFT;

/// Additive key of each card (`rank key | suit counter increment`).
pub const CARD_KEYS: [u64; 52] = {
    let mut k = [0u64; 52];
    let mut i = 0;
    while i < 52 {
        let rank = i / 4;
        let suit = i % 4;
        k[i] = RANK_KEYS[rank] as u64 | (1u64 << (SUIT_SHIFT + 4 * suit as u32));
        i += 1;
    }
    k
};

/// Per-suit rank mask bit of each card (`1 << (16 * suit + rank)`).
pub const CARD_MASKS: [u64; 52] = {
    let mut m = [0u64; 52];
    let mut i = 0;
    while i < 52 {
        m[i] = 1u64 << (16 * (i % 4) + i / 4);
        i += 1;
    }
    m
};

/// Number of low key bits that form the column of the displacement hash.
pub const HASH_SHIFT: u32 = 11;
const COL_MASK: u32 = (1 << HASH_SHIFT) - 1;

/// The evaluator's lookup tables.
pub struct Tables {
    /// Row displacements of the perfect hash, indexed by `key >> HASH_SHIFT`.
    pub offsets: Vec<u32>,
    /// Non-flush strengths indexed by the perfect hash.
    pub rank_table: Vec<u16>,
    /// Flush strengths indexed by the 13-bit rank mask of the flush suit.
    pub flush_table: Vec<u16>,
}

impl Tables {
    /// Perfect hash of a rank-key sum.
    #[inline(always)]
    pub fn hash(&self, key: u32) -> usize {
        // SAFETY-free: `offsets` covers every row reachable by a valid 5..7 card key; for
        // invalid inputs the index is still bounds-checked by the caller's slice access.
        ((key & COL_MASK) + self.offsets[(key >> HASH_SHIFT) as usize]) as usize
    }
}

static TABLES: OnceLock<Tables> = OnceLock::new();

/// Returns the global tables, building them on first use (a few milliseconds in release).
pub fn tables() -> &'static Tables {
    TABLES.get_or_init(build)
}

/// Converts a reference value into the compact `u16` strength:
/// `(category + 1) << 12 | index within category`. Zero is never a valid strength.
struct Compactor {
    map: HashMap<u32, u16>,
}

impl Compactor {
    fn new() -> Compactor {
        // Enumerate every distinct 5-card hand class: all rank multisets (non-flush) and all
        // 5-distinct-rank sets as flushes.
        let mut vals: Vec<u32> = Vec::new();
        for_each_multiset(5, 5, |counts| {
            let r = expand(counts);
            vals.push(reference::eval5_ranks(r[..5].try_into().unwrap(), false));
            if counts.iter().all(|&c| c <= 1) {
                vals.push(reference::eval5_ranks(r[..5].try_into().unwrap(), true));
            }
        });
        vals.sort_unstable();
        vals.dedup();
        debug_assert_eq!(vals.len(), 7462);
        let mut map = HashMap::with_capacity(vals.len());
        let mut last_cat = u32::MAX;
        let mut idx = 0u16;
        for v in vals {
            let cat = v >> 20;
            if cat != last_cat {
                last_cat = cat;
                idx = 0;
            }
            map.insert(v, ((cat as u16 + 1) << 12) | idx);
            idx += 1;
        }
        Compactor { map }
    }
    fn get(&self, v: u32) -> u16 {
        self.map[&v]
    }
}

/// Calls `f` with every rank count vector (≤ 4 per rank) whose total is in `min..=max`.
fn for_each_multiset(min: u8, max: u8, mut f: impl FnMut(&[u8; 13])) {
    fn rec(
        i: usize,
        left: u8,
        min_total: u8,
        total: u8,
        c: &mut [u8; 13],
        f: &mut dyn FnMut(&[u8; 13]),
    ) {
        if i == 13 {
            if total >= min_total {
                f(c);
            }
            return;
        }
        for k in 0..=left.min(4) {
            c[i] = k;
            rec(i + 1, left - k, min_total, total + k, c, f);
        }
        c[i] = 0;
    }
    let mut c = [0u8; 13];
    rec(0, max, min, 0, &mut c, &mut f);
}

/// Expands counts into a rank list (padded with zeros to 7).
fn expand(counts: &[u8; 13]) -> [u8; 7] {
    let mut out = [0u8; 7];
    let mut n = 0;
    for (r, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            out[n] = r as u8;
            n += 1;
        }
    }
    out
}

fn build() -> Tables {
    let comp = Compactor::new();

    // Non-flush table: every rank multiset of 5..=7 cards -> best non-flush 5-card value.
    let mut entries: Vec<(u32, u16)> = Vec::with_capacity(60_000);
    for_each_multiset(5, 7, |counts| {
        let n: usize = counts.iter().map(|&c| c as usize).sum();
        let ranks = expand(counts);
        let mut best = 0u32;
        reference::for_each_5_subset(n, |idx| {
            best = best.max(reference::eval5_ranks(idx.map(|i| ranks[i]), false));
        });
        let key: u32 = counts
            .iter()
            .enumerate()
            .map(|(r, &c)| RANK_KEYS[r] * c as u32)
            .sum();
        entries.push((key, comp.get(best)));
    });

    // Displacement perfect hash: place rows (by descending population) at the first offset
    // where all their columns land in free slots.
    let max_key = entries.iter().map(|e| e.0).max().unwrap();
    let n_rows = (max_key >> HASH_SHIFT) as usize + 1;
    let mut rows: Vec<Vec<(u32, u16)>> = vec![Vec::new(); n_rows];
    for &(k, v) in &entries {
        rows[(k >> HASH_SHIFT) as usize].push((k & COL_MASK, v));
    }
    let mut order: Vec<usize> = (0..n_rows).collect();
    order.sort_by_key(|&r| std::cmp::Reverse(rows[r].len()));
    let mut offsets = vec![0u32; n_rows];
    let mut used = BitSet::default();
    let mut rank_table: Vec<u16> = Vec::new();
    // Any fitting offset yields a valid perfect hash; starting the search where the previous
    // row of the same size landed keeps the build fast at a small cost in table size.
    let mut hint = vec![0usize; 1 + rows.iter().map(|r| r.len()).max().unwrap_or(0)];
    for &r in &order {
        let row = &mut rows[r];
        if row.is_empty() {
            continue;
        }
        row.sort_unstable_by_key(|e| e.0);
        let c0 = row[0].0 as usize;
        // Candidate offsets put the row's first column on a free slot; jump between free
        // slots instead of probing every offset.
        let mut f = used.next_free(c0 + hint[row.len()]);
        let off = loop {
            let off = f - c0;
            if row[1..].iter().all(|&(c, _)| !used.get(off + c as usize)) {
                break off;
            }
            f = used.next_free(f + 1);
        };
        for &(c, v) in row.iter() {
            let i = off + c as usize;
            used.set(i);
            if i >= rank_table.len() {
                rank_table.resize(i + 1, 0);
            }
            rank_table[i] = v;
        }
        offsets[r] = off as u32;
        hint[row.len()] = off;
    }
    // Pad so that any column of any row stays in bounds.
    let needed = offsets.iter().map(|&o| o as usize).max().unwrap_or(0) + COL_MASK as usize + 1;
    rank_table.resize(needed.max(rank_table.len()), 0);

    // Flush table: best flush (or straight flush) for each 13-bit rank mask with ≥ 5 ranks.
    let mut flush_table = vec![0u16; 8192];
    for (mask, slot) in flush_table.iter_mut().enumerate() {
        if (mask as u32).count_ones() < 5 {
            continue;
        }
        let ranks: Vec<u8> = (0..13u8).filter(|r| mask & (1 << r) != 0).collect();
        let mut best = 0u32;
        reference::for_each_5_subset(ranks.len(), |idx| {
            best = best.max(reference::eval5_ranks(idx.map(|i| ranks[i]), true));
        });
        *slot = comp.get(best);
    }

    Tables {
        offsets,
        rank_table,
        flush_table,
    }
}

/// Growable bitset used while placing hash rows.
#[derive(Default)]
struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    fn get(&self, i: usize) -> bool {
        self.words
            .get(i / 64)
            .is_some_and(|w| w >> (i % 64) & 1 == 1)
    }
    fn set(&mut self, i: usize) {
        if i / 64 >= self.words.len() {
            self.words.resize(i / 64 + 1, 0);
        }
        self.words[i / 64] |= 1 << (i % 64);
    }
    /// Smallest free index `>= from`.
    fn next_free(&self, from: usize) -> usize {
        let mut w = from / 64;
        let mut bits = match self.words.get(w) {
            Some(&x) => x | ((1u64 << (from % 64)) - 1),
            None => return from,
        };
        loop {
            if bits != u64::MAX {
                return w * 64 + (!bits).trailing_zeros() as usize;
            }
            w += 1;
            match self.words.get(w) {
                Some(&x) => bits = x,
                None => return w * 64,
            }
        }
    }
}

/// Category of a compact strength.
#[inline]
pub fn category(strength: u16) -> HandCategory {
    HandCategory::from_u8(((strength >> 12) - 1) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn rank_keys_unique() {
        let mut seen = HashSet::new();
        for_each_multiset(0, 7, |c| {
            let k: u32 = c
                .iter()
                .enumerate()
                .map(|(r, &n)| RANK_KEYS[r] * n as u32)
                .sum();
            assert!(seen.insert(k), "collision for {c:?}");
        });
        assert_eq!(seen.len(), 76_155);
    }

    #[test]
    fn table_sizes() {
        let t = tables();
        assert!(t.rank_table.len() < 200_000, "{}", t.rank_table.len());
    }
}
