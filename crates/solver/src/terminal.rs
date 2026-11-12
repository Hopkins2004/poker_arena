//! O(n) terminal evaluation against an opponent reach vector `q` (one entry per hand).
//!
//! * [`compat_mass`]: `out[h] = Σ_{h' compatible with h} q[h']`. With card removal this is
//!   `T − C[a] − C[b] + q[h]` (inclusion–exclusion over the two cards of `h = {a, b}`; `h`
//!   itself was subtracted twice). Fold terminals are a constant payoff times this mass.
//! * [`showdown_net`]: `out[h] = (compatible mass weaker than h) − (compatible mass
//!   stronger)`. Hands are sorted by strength, so one ascending and one descending sweep
//!   over tie groups, carrying the running total `T` and per-card totals `C[k]`, give both
//!   masses; equal strengths count as neither. Showdown value = `(½ + c) · net`.
//!
//! Running sums are kept in f64 so the result does not depend on summation length.

use crate::hands::HandSet;

/// Gather indices that turn both terminal kinds into independent per-hand lookups (the hot
/// path of the solver; [`compat_mass`] / [`showdown_net`] are the straightforward sweeps).
///
/// For each card `k`, `list` holds the hands containing `k` in ascending (strength) order;
/// `P_k[j]` is the opponent mass of the first `j` of them (prefix sums, one extra entry per
/// card, flattened with offset `k + list_off[k]`). With `S[i]` the prefix mass over all hands
/// and `[gs, ge)` the hand's tie group:
///
/// * weaker(h)   = S[gs] − P_a[lo_a] − P_b[lo_b]
/// * stronger(h) = (T − S[ge]) − (C_a − P_a[hi_a]) − (C_b − P_b[hi_b])
/// * compat(h)   = T − C_a − C_b + q[h]
///
/// where `lo` / `hi` count the card's hands in groups before / up to the hand's group.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TermIndex {
    gs: Vec<u32>,
    ge: Vec<u32>,
    /// Hands containing each card, flattened.
    list: Vec<u32>,
    list_off: Vec<u32>,
    /// Per hand: its two cards and flattened prefix indices (lo, hi) for each card.
    card: Vec<[u8; 2]>,
    lo: Vec<[u32; 2]>,
    hi: Vec<[u32; 2]>,
}

/// Prefix sums of one opponent reach vector (reusable scratch).
#[derive(Clone, Debug, Default)]
pub struct TermScratch {
    s: Vec<f64>,
    p: Vec<f64>,
    c: Vec<f64>,
    total: f64,
}

impl TermIndex {
    /// Builds the index for `n` sorted hands with tie `groups` and optional cards.
    pub fn new(n: usize, groups: &[(u32, u32)], cards: Option<&[[u8; 2]]>) -> TermIndex {
        let mut gs = vec![0u32; n];
        let mut ge = vec![0u32; n];
        for &(s, e) in groups {
            for h in s..e {
                gs[h as usize] = s;
                ge[h as usize] = e;
            }
        }
        let Some(cards) = cards else {
            return TermIndex {
                gs,
                ge,
                ..Default::default()
            };
        };
        let mut per: Vec<Vec<u32>> = vec![vec![]; 52];
        for (h, c) in cards.iter().enumerate() {
            per[c[0] as usize].push(h as u32);
            per[c[1] as usize].push(h as u32);
        }
        let mut list_off = vec![0u32; 53];
        for k in 0..52 {
            list_off[k + 1] = list_off[k] + per[k].len() as u32;
        }
        let list: Vec<u32> = per.iter().flatten().copied().collect();
        let mut lo = vec![[0u32; 2]; n];
        let mut hi = vec![[0u32; 2]; n];
        for (h, c) in cards.iter().enumerate() {
            for s in 0..2 {
                let k = c[s] as usize;
                let l = &per[k];
                // Card k's hands strictly before / up to this hand's group.
                let before = l.partition_point(|&x| x < gs[h]);
                let upto = l.partition_point(|&x| x < ge[h]);
                let base = k as u32 + list_off[k];
                lo[h][s] = base + before as u32;
                hi[h][s] = base + upto as u32;
            }
        }
        TermIndex {
            gs,
            ge,
            list,
            list_off,
            card: cards.to_vec(),
            lo,
            hi,
        }
    }

    fn has_cards(&self) -> bool {
        !self.list_off.is_empty()
    }

    /// Fills `sc` with the prefix sums of `q`.
    pub fn prepare(&self, q: &[f32], sc: &mut TermScratch) {
        let n = q.len();
        sc.s.resize(n + 1, 0.0);
        let mut acc = 0f64;
        sc.s[0] = 0.0;
        for (i, &x) in q.iter().enumerate() {
            acc += x as f64;
            sc.s[i + 1] = acc;
        }
        sc.total = acc;
        if !self.has_cards() || acc == 0.0 {
            return;
        }
        sc.p.resize(self.list.len() + 52, 0.0);
        sc.c.resize(52, 0.0);
        for k in 0..52 {
            let (a, b) = (self.list_off[k] as usize, self.list_off[k + 1] as usize);
            let base = k + a;
            let mut acc = 0f64;
            sc.p[base] = 0.0;
            for (j, &h) in self.list[a..b].iter().enumerate() {
                acc += q[h as usize] as f64;
                sc.p[base + j + 1] = acc;
            }
            sc.c[k] = acc;
        }
    }

    /// Compatible mass (fold terminals) from prepared sums.
    pub fn compat(&self, q: &[f32], sc: &TermScratch, out: &mut [f32]) {
        if sc.total == 0.0 {
            out.fill(0.0);
            return;
        }
        if !self.has_cards() {
            out.fill(sc.total as f32);
            return;
        }
        for (h, o) in out.iter_mut().enumerate() {
            let [a, b] = self.card[h];
            *o = (sc.total - sc.c[a as usize] - sc.c[b as usize] + q[h] as f64) as f32;
        }
    }

    /// Weaker-minus-stronger mass (showdown terminals) from prepared sums.
    pub fn net(&self, sc: &TermScratch, out: &mut [f32]) {
        let (s, t) = (&sc.s, sc.total);
        if t == 0.0 {
            out.fill(0.0);
            return;
        }
        if !self.has_cards() {
            for (h, o) in out.iter_mut().enumerate() {
                let weaker = s[self.gs[h] as usize];
                let stronger = t - s[self.ge[h] as usize];
                *o = (weaker - stronger) as f32;
            }
            return;
        }
        let p = &sc.p;
        for (h, o) in out.iter_mut().enumerate() {
            let [a, b] = self.card[h];
            let [la, lb] = self.lo[h];
            let [ha, hb] = self.hi[h];
            let weaker = s[self.gs[h] as usize] - p[la as usize] - p[lb as usize];
            let stronger = (t - s[self.ge[h] as usize])
                - (sc.c[a as usize] - p[ha as usize])
                - (sc.c[b as usize] - p[hb as usize]);
            *o = (weaker - stronger) as f32;
        }
    }
}

/// Compatible opponent mass of every hand.
pub fn compat_mass(hs: &HandSet, q: &[f32], out: &mut [f32]) {
    let n = hs.len();
    debug_assert!(q.len() == n && out.len() == n);
    let total: f64 = q.iter().map(|&x| x as f64).sum();
    match &hs.cards {
        None => out.fill(total as f32),
        Some(cards) => {
            let mut c = [0f64; 52];
            for (h, &x) in q.iter().enumerate() {
                let [a, b] = cards[h];
                c[a as usize] += x as f64;
                c[b as usize] += x as f64;
            }
            for h in 0..n {
                let [a, b] = cards[h];
                out[h] = (total - c[a as usize] - c[b as usize] + q[h] as f64) as f32;
            }
        }
    }
}

/// Weaker-minus-stronger compatible opponent mass of every hand.
pub fn showdown_net(hs: &HandSet, q: &[f32], out: &mut [f32]) {
    let n = hs.len();
    debug_assert!(q.len() == n && out.len() == n);
    match &hs.cards {
        None => {
            // Ascending: weaker mass.
            let mut t = 0f64;
            for &(s, e) in &hs.groups {
                let (s, e) = (s as usize, e as usize);
                for o in &mut out[s..e] {
                    *o = t as f32;
                }
                t += q[s..e].iter().map(|&x| x as f64).sum::<f64>();
            }
            // Descending: subtract stronger mass.
            let mut t = 0f64;
            for &(s, e) in hs.groups.iter().rev() {
                let (s, e) = (s as usize, e as usize);
                for o in &mut out[s..e] {
                    *o = (*o as f64 - t) as f32;
                }
                t += q[s..e].iter().map(|&x| x as f64).sum::<f64>();
            }
        }
        Some(cards) => {
            let mut t = 0f64;
            let mut c = [0f64; 52];
            for &(s, e) in &hs.groups {
                let (s, e) = (s as usize, e as usize);
                for h in s..e {
                    let [a, b] = cards[h];
                    out[h] = (t - c[a as usize] - c[b as usize]) as f32;
                }
                for h in s..e {
                    let [a, b] = cards[h];
                    let x = q[h] as f64;
                    t += x;
                    c[a as usize] += x;
                    c[b as usize] += x;
                }
            }
            let mut t = 0f64;
            let mut c = [0f64; 52];
            for &(s, e) in hs.groups.iter().rev() {
                let (s, e) = (s as usize, e as usize);
                for h in s..e {
                    let [a, b] = cards[h];
                    out[h] = (out[h] as f64 - (t - c[a as usize] - c[b as usize])) as f32;
                }
                for h in s..e {
                    let [a, b] = cards[h];
                    let x = q[h] as f64;
                    t += x;
                    c[a as usize] += x;
                    c[b as usize] += x;
                }
            }
        }
    }
}

/// O(n²) reference of [`compat_mass`] (tests).
pub fn compat_mass_naive(hs: &HandSet, q: &[f32]) -> Vec<f64> {
    (0..hs.len())
        .map(|h| {
            (0..hs.len())
                .filter(|&j| hs.compatible(h, j))
                .map(|j| q[j] as f64)
                .sum()
        })
        .collect()
}

/// O(n²) reference of [`showdown_net`] (tests).
pub fn showdown_net_naive(hs: &HandSet, q: &[f32]) -> Vec<f64> {
    (0..hs.len())
        .map(|h| {
            (0..hs.len())
                .filter(|&j| hs.compatible(h, j))
                .map(|j| {
                    let s = (hs.strength[h] as i64 - hs.strength[j] as i64).signum();
                    s as f64 * q[j] as f64
                })
                .sum()
        })
        .collect()
}
