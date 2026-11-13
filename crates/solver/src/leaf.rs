//! Batched river-leaf kernels for depth-limited turn solving.
//!
//! A turn leaf (turn betting closed with chips behind) is valued by a bucketed river value
//! network, once per river card. Around each network call the turn solver needs, for every
//! leaf and river card `c`:
//!
//! * the bucketed, normalised ranges `r̄_p(c, k) = Σ_{h∈k} r_p(h) / Σ_h r_p(h)` (hands valid
//!   on `board + c`);
//! * the projection weights `w_p(c, k) = Σ_{h∈k} r_p(h) m_p(c, h) / (tot_p · tot_o)`, with
//!   `m_p(c, h)` the opponent reach compatible with `h` and `c`;
//! * and afterwards the broadcast `cfv_p(h) = scale · Σ_c m_p(c, h) · y_p(c, bucket(c, h))`.
//!
//! `m_p(c, h)` comes from inclusion–exclusion over card sums, `O(1)` per `(c, h)`: with
//! opponent reach `q`, `T = Σ q`, `S[x] = Σ_{h'∋x} q(h')` and `h = {a, b}`,
//! `m(c, h) = T − S[a] − S[b] − S[c] + q(ab) + q(ac) + q(bc)` (no hand holds three cards).
//! Everything is `O(L · cards · hands)` with no dense one-hot products.

use crate::hands::{combo_cards, combo_index, N_COMBOS};

/// Per-card bucket assignment of one turn board's valid hands (see the module docs).
pub struct LeafIndex {
    cards: Vec<u8>,
    k: usize,
    off: Vec<usize>,
    hand: Vec<u16>,
    bucket: Vec<u16>,
    ca: Vec<u8>,
    cb: Vec<u8>,
    pac: Vec<u16>,
    pbc: Vec<u16>,
}

fn card_sums(q: &[f64]) -> (f64, [f64; 52]) {
    let cc = combo_cards();
    let mut s = [0f64; 52];
    let mut t = 0f64;
    for (h, &x) in q.iter().enumerate() {
        if x != 0.0 {
            t += x;
            s[cc[h][0] as usize] += x;
            s[cc[h][1] as usize] += x;
        }
    }
    (t, s)
}

impl LeafIndex {
    /// `cards`: the river cards; `bmap`: `[cards × 1326]` bucket of every combo on each river
    /// board, negative for combos blocked by that board; `k`: buckets per board.
    pub fn new(cards: &[u8], bmap: &[i32], k: usize) -> Result<Self, String> {
        if bmap.len() != cards.len() * N_COMBOS {
            return Err("bmap must be [cards * 1326]".into());
        }
        let cc = combo_cards();
        let mut ix = LeafIndex {
            cards: cards.to_vec(),
            k,
            off: vec![0],
            hand: vec![],
            bucket: vec![],
            ca: vec![],
            cb: vec![],
            pac: vec![],
            pbc: vec![],
        };
        for (ci, &c) in cards.iter().enumerate() {
            if c >= 52 {
                return Err("river cards must be 0..52".into());
            }
            for h in 0..N_COMBOS {
                let b = bmap[ci * N_COMBOS + h];
                if b < 0 {
                    continue;
                }
                if b as usize >= k {
                    return Err(format!("bucket {b} out of range for K = {k}"));
                }
                let [a, bb] = cc[h];
                if a == c || bb == c {
                    return Err("a combo holding the river card has a bucket".into());
                }
                ix.hand.push(h as u16);
                ix.bucket.push(b as u16);
                ix.ca.push(a);
                ix.cb.push(bb);
                ix.pac.push(combo_index(a, c) as u16);
                ix.pbc.push(combo_index(bb, c) as u16);
            }
            ix.off.push(ix.hand.len());
        }
        Ok(ix)
    }

    /// Number of river cards.
    pub fn n_cards(&self) -> usize {
        self.cards.len()
    }

    fn select<'a>(&'a self, sel: Option<&'a [usize]>) -> Box<dyn Iterator<Item = usize> + 'a> {
        match sel {
            Some(s) => Box::new(s.iter().copied()),
            None => Box::new(0..self.cards.len()),
        }
    }

    /// Network inputs for one leaf: `x` and `w`, both `[cards × 2K]` (player 0's K buckets
    /// then player 1's), over the river cards `sel` (indices into the card list; `None` = all,
    /// and the outputs then have `sel.len()` card rows). `r`: both players' reach at the leaf (`[1326]` each, 0 on combos
    /// blocked by the turn board); `q`: the reach the opponent masses `m_p` are computed
    /// from (normally `r`; differs when an empty reach was replaced by a fallback range).
    pub fn inputs(
        &self,
        sel: Option<&[usize]>,
        r: [&[f64]; 2],
        q: [&[f64]; 2],
        x: &mut [f32],
        w: &mut [f32],
    ) {
        let k = self.k;
        let (t0, s0) = card_sums(r[0]);
        let (t1, s1) = card_sums(r[1]);
        let (t, s) = ([t0, t1], [s0, s1]);
        let (u0, z0) = card_sums(q[0]);
        let (u1, z1) = card_sums(q[1]);
        let (u, z) = ([u0, u1], [z0, z1]);
        let mut xb = vec![0f64; 2 * k];
        let mut wb = vec![0f64; 2 * k];
        for (row, ci) in self.select(sel).enumerate() {
            let c = self.cards[ci] as usize;
            xb.iter_mut().for_each(|v| *v = 0.0);
            wb.iter_mut().for_each(|v| *v = 0.0);
            let tot = [t[0] - s[0][c], t[1] - s[1][c]];
            for e in self.off[ci]..self.off[ci + 1] {
                let (h, b) = (self.hand[e] as usize, self.bucket[e] as usize);
                let (a, bb) = (self.ca[e] as usize, self.cb[e] as usize);
                let (pac, pbc) = (self.pac[e] as usize, self.pbc[e] as usize);
                for p in 0..2 {
                    let rp = r[p][h];
                    if rp == 0.0 {
                        continue;
                    }
                    let o = 1 - p;
                    let qo = q[o];
                    let m = u[o] - z[o][a] - z[o][bb] - z[o][c] + qo[h] + qo[pac] + qo[pbc];
                    xb[p * k + b] += rp;
                    wb[p * k + b] += rp * m;
                }
            }
            let safe = tot.map(|v| if v > 0.0 { v } else { 1.0 });
            let base = row * 2 * k;
            for p in 0..2 {
                for j in 0..k {
                    x[base + p * k + j] = (xb[p * k + j] / safe[p]) as f32;
                    w[base + p * k + j] = (wb[p * k + j] / (safe[p] * safe[1 - p])) as f32;
                }
            }
        }
    }

    /// Counterfactual leaf values of one leaf from per-card bucket values `y`
    /// (`[cards × 2K]`, conditional, river-pot units): `out_p(h) = scale · Σ_c m_p(c, h) ·
    /// y_p(c, bucket(c, h))`, the sum over the river cards `sel` (rows of `y`; `None` = all).
    pub fn cfv(
        &self,
        sel: Option<&[usize]>,
        y: &[f32],
        r: [&[f64]; 2],
        scale: f64,
        out: [&mut [f64]; 2],
    ) {
        let k = self.k;
        let (t0, s0) = card_sums(r[0]);
        let (t1, s1) = card_sums(r[1]);
        let (t, s) = ([t0, t1], [s0, s1]);
        let [o0, o1] = out;
        o0.iter_mut().for_each(|v| *v = 0.0);
        o1.iter_mut().for_each(|v| *v = 0.0);
        for (row, ci) in self.select(sel).enumerate() {
            let c = self.cards[ci] as usize;
            let yc = &y[row * 2 * k..(row + 1) * 2 * k];
            for e in self.off[ci]..self.off[ci + 1] {
                let (h, b) = (self.hand[e] as usize, self.bucket[e] as usize);
                let (a, bb) = (self.ca[e] as usize, self.cb[e] as usize);
                let (pac, pbc) = (self.pac[e] as usize, self.pbc[e] as usize);
                // m_0 uses player 1's reach and vice versa.
                let m0 = t[1] - s[1][a] - s[1][bb] - s[1][c] + r[1][h] + r[1][pac] + r[1][pbc];
                let m1 = t[0] - s[0][a] - s[0][bb] - s[0][c] + r[0][h] + r[0][pac] + r[0][pbc];
                o0[h] += m0 * yc[b] as f64;
                o1[h] += m1 * yc[k + b] as f64;
            }
        }
        o0.iter_mut().for_each(|v| *v *= scale);
        o1.iter_mut().for_each(|v| *v *= scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force m(c, h) and the kernels on a random board with random buckets.
    #[test]
    fn kernels_match_brute_force() {
        let board = [3u8, 17, 30, 44];
        // A subset of the river cards keeps the O(cards · 1326²) brute force fast.
        let cards: Vec<u8> = (0..52).filter(|c| !board.contains(c)).step_by(8).collect();
        let cc = combo_cards();
        let k = 5;
        let mut seed = 12345u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 10_000) as f64 / 10_000.0
        };
        let turn_valid = |h: usize| !board.contains(&cc[h][0]) && !board.contains(&cc[h][1]);
        let mut bmap = vec![-1i32; cards.len() * N_COMBOS];
        for (ci, &c) in cards.iter().enumerate() {
            for h in 0..N_COMBOS {
                if turn_valid(h) && cc[h][0] != c && cc[h][1] != c {
                    bmap[ci * N_COMBOS + h] = (h * 7 + ci) as i32 % k as i32;
                }
            }
        }
        let r: Vec<Vec<f64>> = (0..2)
            .map(|_| {
                (0..N_COMBOS)
                    .map(|h| {
                        if turn_valid(h) && rnd() < 0.6 {
                            rnd()
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        let ix = LeafIndex::new(&cards, &bmap, k).unwrap();
        let n = cards.len() * 2 * k;
        let (mut x, mut w) = (vec![0f32; n], vec![0f32; n]);
        ix.inputs(None, [&r[0], &r[1]], [&r[0], &r[1]], &mut x, &mut w);
        let y: Vec<f32> = (0..n).map(|_| rnd() as f32 - 0.5).collect();
        let (mut o0, mut o1) = (vec![0f64; N_COMBOS], vec![0f64; N_COMBOS]);
        ix.cfv(None, &y, [&r[0], &r[1]], 0.3, [&mut o0, &mut o1]);

        let disjoint = |h: usize, g: usize| {
            let (x, y) = (cc[h], cc[g]);
            x[0] != y[0] && x[0] != y[1] && x[1] != y[0] && x[1] != y[1]
        };
        let mut eo = [vec![0f64; N_COMBOS], vec![0f64; N_COMBOS]];
        for ci in 0..cards.len() {
            let alive = |g: usize| bmap[ci * N_COMBOS + g] >= 0;
            let tot: Vec<f64> = (0..2)
                .map(|p| (0..N_COMBOS).filter(|&g| alive(g)).map(|g| r[p][g]).sum())
                .collect();
            let mut ex = vec![0f64; 2 * k];
            let mut ew = vec![0f64; 2 * k];
            for h in (0..N_COMBOS).filter(|&h| alive(h)) {
                let b = bmap[ci * N_COMBOS + h] as usize;
                for p in 0..2 {
                    let o = 1 - p;
                    let m: f64 = (0..N_COMBOS)
                        .filter(|&g| alive(g) && disjoint(h, g))
                        .map(|g| r[o][g])
                        .sum();
                    ex[p * k + b] += r[p][h] / tot[p];
                    ew[p * k + b] += r[p][h] * m / (tot[p] * tot[o]);
                    eo[p][h] += 0.3 * m * y[ci * 2 * k + p * k + b] as f64;
                }
            }
            for j in 0..2 * k {
                let (a, b) = (x[ci * 2 * k + j] as f64, ex[j]);
                assert!((a - b).abs() < 1e-5, "x card {ci} slot {j}: {a} vs {b}");
                let (a, b) = (w[ci * 2 * k + j] as f64, ew[j]);
                assert!((a - b).abs() < 1e-5, "w card {ci} slot {j}: {a} vs {b}");
            }
        }
        for h in 0..N_COMBOS {
            assert!((o0[h] - eo[0][h]).abs() < 1e-6 * (1.0 + eo[0][h].abs()));
            assert!((o1[h] - eo[1][h]).abs() < 1e-6 * (1.0 + eo[1][h].abs()));
        }
    }
}
