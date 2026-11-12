//! Vectorized public-tree DCFR with alternating updates.
//!
//! Arrays are flat with the hand index innermost:
//! * per player `p`: `regret`, `strat_sum`, `cur` of shape `[slot, max_actions, n]`;
//! * per traversal: `reach_opp`, `reach_own` and `cfv` of shape `[node, n]`.
//!
//! Iteration `t` (1-based), for traverser `p = 0` then `p = 1`:
//! 1. forward pass: opponent reach under the opponent's current strategy (root = its
//!    range) and `p`'s own *sequence* reach under its current strategy (root = 1);
//! 2. backward pass: counterfactual values `cfv` of `p` (terminals in O(n), see
//!    [`crate::terminal`]); at `p`'s decisions `regret[a] += cfv[child a] − cfv[node]`;
//! 3. DCFR discount (α, β): positive regrets × `t^α/(t^α+1)`, negative × `t^β/(t^β+1)`;
//! 4. `strat_sum += reach_own · cur`, then `strat_sum ×= (t/(t+1))^γ`;
//! 5. regret matching gives `p`'s new current strategy.
//!
//! The average strategy is weighted by the own *sequence* reach (root weight 1 for every
//! hand). A per-hand root factor would cancel when normalising, so hands in the range are
//! unaffected, and hands outside the range still get a well-defined average strategy (their
//! regrets are counterfactual), hence well-defined values.

use crate::hands::HandSet;
use crate::terminal::{compat_mass, TermScratch};
use crate::tree::{NodeKind, Tree};

/// Solver parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Maximum iterations.
    pub max_iters: u32,
    /// Stop when exploitability (pot units) drops below this.
    pub target_expl: f32,
    /// Exploitability is measured every this many iterations (and at the end).
    pub check_every: u32,
    /// DCFR positive-regret discount exponent.
    pub alpha: f64,
    /// DCFR negative-regret discount exponent.
    pub beta: f64,
    /// DCFR average-strategy exponent.
    pub gamma: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            max_iters: 500,
            target_expl: 0.002,
            check_every: 50,
            alpha: 1.5,
            beta: 0.0,
            gamma: 2.0,
        }
    }
}

/// Result of a solve (vectors in the hand set's sorted order).
#[derive(Clone, Debug)]
pub struct Solution {
    /// Average strategy per player, `[slot, max_actions, n]`, normalised per hand.
    pub avg: [Vec<f32>; 2],
    /// Counterfactual values at the root under the average strategies: `ṽ_p(h)`.
    pub cfv: [Vec<f32>; 2],
    /// Compatible opponent mass `m_p(h)`.
    pub mass: [Vec<f32>; 2],
    /// Conditional values `v_p(h) = ṽ_p(h) / m_p(h)` (0 where the mass is 0).
    pub value: [Vec<f32>; 2],
    /// Final exploitability (pot units).
    pub expl: f32,
    /// Iterations run.
    pub iters: u32,
    /// `(iteration, exploitability)` at every check.
    pub history: Vec<(u32, f32)>,
}

/// What a backward pass computes at the traverser's own decisions.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Values under a given strategy.
    Follow,
    /// Best response: max over actions.
    BestResponse,
}

/// Solver state for one (tree, hand set, ranges).
pub struct Solver<'a> {
    tree: &'a Tree,
    hs: &'a HandSet,
    range: [Vec<f32>; 2],
    n: usize,
    a: usize,
    regret: [Vec<f32>; 2],
    strat_sum: [Vec<f32>; 2],
    cur: [Vec<f32>; 2],
    reach_opp: Vec<f32>,
    reach_own: Vec<f32>,
    cfv: Vec<f32>,
    ones: Vec<f32>,
    /// Per traverser: the node whose opponent reach equals this node's (reach is unchanged
    /// along the traverser's own edges), so sibling terminals share prepared sums.
    qsrc: [Vec<u32>; 2],
    scratch: TermScratch,
    t: u32,
}

impl<'a> Solver<'a> {
    /// New solver. `range[p]` has one entry per hand (sorted order); ranges are used as
    /// given (normalise them first for pot-unit values).
    pub fn new(tree: &'a Tree, hs: &'a HandSet, range: [&[f32]; 2]) -> Solver<'a> {
        let n = hs.len();
        assert!(range[0].len() == n && range[1].len() == n);
        let a = tree.max_actions;
        let size = |p: usize| tree.n_slots[p] * a * n;
        let nodes = tree.len() * n;
        let mut s = Solver {
            tree,
            hs,
            range: [range[0].to_vec(), range[1].to_vec()],
            n,
            a,
            regret: [vec![0.0; size(0)], vec![0.0; size(1)]],
            strat_sum: [vec![0.0; size(0)], vec![0.0; size(1)]],
            cur: [vec![0.0; size(0)], vec![0.0; size(1)]],
            reach_opp: vec![0.0; nodes],
            reach_own: vec![0.0; nodes],
            cfv: vec![0.0; nodes],
            ones: vec![1.0; n],
            qsrc: [reach_sources(tree, 0), reach_sources(tree, 1)],
            scratch: TermScratch::default(),
            t: 0,
        };
        for p in 0..2 {
            s.regret_match(p);
        }
        s
    }

    /// Runs DCFR until the target exploitability or `max_iters`, then the final value pass.
    pub fn solve(mut self, params: &Params) -> Solution {
        let mut history = Vec::new();
        let mut expl = f32::NAN;
        let every = params.check_every.max(1);
        while self.t < params.max_iters {
            self.iterate(params);
            if self.t.is_multiple_of(every) || self.t == params.max_iters {
                expl = self.exploitability();
                history.push((self.t, expl));
                if expl < params.target_expl {
                    break;
                }
            }
        }
        if self.t == 0 {
            expl = self.exploitability();
        }
        let avg = [self.average(0), self.average(1)];
        let mut cfv = [vec![0.0; self.n], vec![0.0; self.n]];
        let mut mass = [vec![0.0; self.n], vec![0.0; self.n]];
        let mut value = [vec![0.0; self.n], vec![0.0; self.n]];
        for p in 0..2 {
            let o = 1 - p;
            cfv[p] = self.values(p, Some(&avg[p]), &avg[o], Mode::Follow);
            compat_mass(self.hs, &self.range[o], &mut mass[p]);
            for h in 0..self.n {
                value[p][h] = if mass[p][h] > 0.0 {
                    cfv[p][h] / mass[p][h]
                } else {
                    0.0
                };
            }
        }
        Solution {
            avg,
            cfv,
            mass,
            value,
            expl,
            iters: self.t,
            history,
        }
    }

    /// One alternating DCFR iteration.
    pub fn iterate(&mut self, params: &Params) {
        self.t += 1;
        let t = self.t as f64;
        let ta = t.powf(params.alpha);
        let tb = t.powf(params.beta);
        let dpos = (ta / (ta + 1.0)) as f32;
        let dneg = (tb / (tb + 1.0)) as f32;
        let dstrat = (t / (t + 1.0)).powf(params.gamma) as f32;
        for p in 0..2 {
            let o = 1 - p;
            let (n, tree) = (self.n, self.tree);
            push_reach(
                tree,
                n,
                self.a,
                &self.cur[o],
                o,
                &self.range[o],
                &mut self.reach_opp,
            );
            push_reach(
                tree,
                n,
                self.a,
                &self.cur[p],
                p,
                &self.ones,
                &mut self.reach_own,
            );
            self.backward_cfr(p);
            // Discount regrets, accumulate the average strategy.
            for r in self.regret[p].iter_mut() {
                *r *= if *r > 0.0 { dpos } else { dneg };
            }
            let a = self.a;
            for i in 0..tree.len() {
                if tree.kind[i] != NodeKind::Decision || tree.player[i] as usize != p {
                    continue;
                }
                let base = tree.slot[i] as usize * a * n;
                let own = &self.reach_own[i * n..(i + 1) * n];
                for k in 0..tree.n_children[i] as usize {
                    let off = base + k * n;
                    let ss = &mut self.strat_sum[p][off..off + n];
                    let cu = &self.cur[p][off..off + n];
                    for h in 0..n {
                        ss[h] = (ss[h] + own[h] * cu[h]) * dstrat;
                    }
                }
            }
            self.regret_match(p);
        }
    }

    /// Regret matching for all of `p`'s decisions.
    fn regret_match(&mut self, p: usize) {
        let (tree, n, a) = (self.tree, self.n, self.a);
        let mut sum = vec![0f32; n];
        for i in 0..tree.len() {
            if tree.kind[i] != NodeKind::Decision || tree.player[i] as usize != p {
                continue;
            }
            let base = tree.slot[i] as usize * a * n;
            let k = tree.n_children[i] as usize;
            sum.fill(0.0);
            for j in 0..k {
                let r = &self.regret[p][base + j * n..base + (j + 1) * n];
                for h in 0..n {
                    sum[h] += r[h].max(0.0);
                }
            }
            let uni = 1.0 / k as f32;
            for x in sum.iter_mut() {
                // Store 1/sum (0 marks "uniform") so the per-action loop is a multiply.
                *x = if *x > 0.0 { 1.0 / *x } else { 0.0 };
            }
            for j in 0..k {
                let off = base + j * n;
                let r = &self.regret[p][off..off + n];
                let c = &mut self.cur[p][off..off + n];
                for h in 0..n {
                    c[h] = if sum[h] > 0.0 {
                        r[h].max(0.0) * sum[h]
                    } else {
                        uni
                    };
                }
            }
        }
    }

    /// Backward pass for traverser `p` under the current strategy, updating its regrets.
    fn backward_cfr(&mut self, p: usize) {
        let (tree, n, a) = (self.tree, self.n, self.a);
        let mut cached = u32::MAX;
        for i in (0..tree.len()).rev() {
            match tree.kind[i] {
                NodeKind::Decision => {
                    let (head, tail) = self.cfv.split_at_mut((i + 1) * n);
                    let node = &mut head[i * n..];
                    let first = tree.first_child[i] as usize;
                    let k = tree.n_children[i] as usize;
                    let child = |j: usize| {
                        let s = (first + j - i - 1) * n;
                        &tail[s..s + n]
                    };
                    node.fill(0.0);
                    if tree.player[i] as usize == p {
                        let base = tree.slot[i] as usize * a * n;
                        for j in 0..k {
                            let cu = &self.cur[p][base + j * n..base + (j + 1) * n];
                            let cv = child(j);
                            for h in 0..n {
                                node[h] += cu[h] * cv[h];
                            }
                        }
                        for j in 0..k {
                            let r = &mut self.regret[p][base + j * n..base + (j + 1) * n];
                            let cv = child(j);
                            for h in 0..n {
                                r[h] += cv[h] - node[h];
                            }
                        }
                    } else {
                        for j in 0..k {
                            let cv = child(j);
                            for h in 0..n {
                                node[h] += cv[h];
                            }
                        }
                    }
                }
                NodeKind::Chance => unimplemented!("chance nodes (turn solving)"),
                _ => {
                    let src = self.qsrc[p][i];
                    let q = &self.reach_opp[i * n..(i + 1) * n];
                    if cached != src {
                        self.hs.index.prepare(q, &mut self.scratch);
                        cached = src;
                    }
                    terminal_value(
                        tree,
                        self.hs,
                        i,
                        p,
                        q,
                        &self.scratch,
                        &mut self.cfv[i * n..(i + 1) * n],
                    )
                }
            }
        }
    }

    /// Root counterfactual values of `p` against the opponent strategy `opp`, following
    /// `own` at `p`'s decisions (Follow) or best-responding (BestResponse).
    fn values(&mut self, p: usize, own: Option<&[f32]>, opp: &[f32], mode: Mode) -> Vec<f32> {
        let (tree, n, a) = (self.tree, self.n, self.a);
        let o = 1 - p;
        push_reach(tree, n, a, opp, o, &self.range[o], &mut self.reach_opp);
        let mut cached = u32::MAX;
        for i in (0..tree.len()).rev() {
            match tree.kind[i] {
                NodeKind::Decision => {
                    let (head, tail) = self.cfv.split_at_mut((i + 1) * n);
                    let node = &mut head[i * n..];
                    let first = tree.first_child[i] as usize;
                    let k = tree.n_children[i] as usize;
                    let child = |j: usize| {
                        let s = (first + j - i - 1) * n;
                        &tail[s..s + n]
                    };
                    if tree.player[i] as usize == p {
                        if mode == Mode::BestResponse {
                            node.copy_from_slice(child(0));
                            for j in 1..k {
                                let cv = child(j);
                                for h in 0..n {
                                    node[h] = node[h].max(cv[h]);
                                }
                            }
                        } else {
                            let own = own.expect("own strategy");
                            let base = tree.slot[i] as usize * a * n;
                            node.fill(0.0);
                            for j in 0..k {
                                let st = &own[base + j * n..base + (j + 1) * n];
                                let cv = child(j);
                                for h in 0..n {
                                    node[h] += st[h] * cv[h];
                                }
                            }
                        }
                    } else {
                        node.fill(0.0);
                        for j in 0..k {
                            let cv = child(j);
                            for h in 0..n {
                                node[h] += cv[h];
                            }
                        }
                    }
                }
                NodeKind::Chance => unimplemented!("chance nodes (turn solving)"),
                _ => {
                    let src = self.qsrc[p][i];
                    let q = &self.reach_opp[i * n..(i + 1) * n];
                    if cached != src {
                        self.hs.index.prepare(q, &mut self.scratch);
                        cached = src;
                    }
                    terminal_value(
                        tree,
                        self.hs,
                        i,
                        p,
                        q,
                        &self.scratch,
                        &mut self.cfv[i * n..(i + 1) * n],
                    )
                }
            }
        }
        self.cfv[..n].to_vec()
    }

    /// Normalised average strategy of `p` (uniform where the accumulated sum is 0).
    pub fn average(&self, p: usize) -> Vec<f32> {
        let (tree, n, a) = (self.tree, self.n, self.a);
        let mut out = vec![0f32; self.strat_sum[p].len()];
        let mut sum = vec![0f32; n];
        for i in 0..tree.len() {
            if tree.kind[i] != NodeKind::Decision || tree.player[i] as usize != p {
                continue;
            }
            let base = tree.slot[i] as usize * a * n;
            let k = tree.n_children[i] as usize;
            sum.fill(0.0);
            for j in 0..k {
                let s = &self.strat_sum[p][base + j * n..base + (j + 1) * n];
                for h in 0..n {
                    sum[h] += s[h];
                }
            }
            for j in 0..k {
                let off = base + j * n;
                for h in 0..n {
                    out[off + h] = if sum[h] > 0.0 {
                        self.strat_sum[p][off + h] / sum[h]
                    } else {
                        1.0 / k as f32
                    };
                }
            }
        }
        out
    }

    /// Exploitability of the average strategies: `½ (Σ r0·BR̃0 + Σ r1·BR̃1)`.
    pub fn exploitability(&mut self) -> f32 {
        let avg = [self.average(0), self.average(1)];
        let mut total = 0f64;
        for p in 0..2 {
            let br = self.values(p, None, &avg[1 - p], Mode::BestResponse);
            total += br
                .iter()
                .zip(&self.range[p])
                .map(|(&v, &r)| v as f64 * r as f64)
                .sum::<f64>();
        }
        (0.5 * total) as f32
    }

    /// Iterations run so far.
    pub fn iterations(&self) -> u32 {
        self.t
    }
}

/// Reach of `player` at every node: root = `root`, multiplied by `strat` (that player's
/// strategy) on the edges leaving its own decisions.
fn push_reach(
    tree: &Tree,
    n: usize,
    a: usize,
    strat: &[f32],
    player: usize,
    root: &[f32],
    out: &mut [f32],
) {
    out[..n].copy_from_slice(root);
    for i in 0..tree.len() {
        if tree.kind[i] != NodeKind::Decision {
            continue;
        }
        let first = tree.first_child[i] as usize;
        let k = tree.n_children[i] as usize;
        let (head, tail) = out.split_at_mut((i + 1) * n);
        let me = &head[i * n..];
        for j in 0..k {
            let c = first + j;
            let dst = &mut tail[(c - i - 1) * n..(c - i) * n];
            if tree.player[i] as usize == player {
                let off = (tree.slot[i] as usize * a + j) * n;
                let st = &strat[off..off + n];
                for h in 0..n {
                    dst[h] = me[h] * st[h];
                }
            } else {
                dst.copy_from_slice(me);
            }
        }
    }
}

/// For traverser `p`: the node whose opponent reach each node shares (the nearest ancestor
/// reached through an opponent edge, or itself).
fn reach_sources(tree: &Tree, p: usize) -> Vec<u32> {
    let mut src: Vec<u32> = (0..tree.len() as u32).collect();
    for i in 0..tree.len() {
        if tree.kind[i] == NodeKind::Decision && tree.player[i] as usize == p {
            for c in tree.children(i) {
                src[c] = src[i];
            }
        }
    }
    src
}

/// Counterfactual value of traverser `p` at terminal `i` given opponent reach `q` and its
/// prepared prefix sums.
fn terminal_value(
    tree: &Tree,
    hs: &HandSet,
    i: usize,
    p: usize,
    q: &[f32],
    sc: &TermScratch,
    out: &mut [f32],
) {
    let c = tree.contrib[i];
    let u = match tree.kind[i] {
        NodeKind::Fold => {
            let folder = tree.player[i] as usize;
            hs.index.compat(q, sc, out);
            // Folder: −½ − c_f. Winner: ½ + c_f (the folder's chips; see crate docs).
            if folder == p {
                -0.5 - c[folder]
            } else {
                0.5 + c[folder]
            }
        }
        NodeKind::Showdown => {
            hs.index.net(sc, out);
            0.5 + c[p]
        }
        _ => unreachable!(),
    };
    for x in out.iter_mut() {
        *x *= u;
    }
}
