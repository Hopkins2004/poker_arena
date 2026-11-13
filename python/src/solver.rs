//! Bindings for the river solver crate (`solver`): exposed as `pokerarena._native.solver_*`
//! and wrapped by the `river_net` package. All solves release the GIL.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use solver::{Abstraction, Edge, HandSet, NodeKind, Params, Solver, Tree, N_COMBOS};

type ArrayPair<'py> = (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>);
type ArrayPair64<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

fn verr<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn abstraction(d: Option<&Bound<'_, PyDict>>) -> PyResult<Abstraction> {
    let mut a = Abstraction::default();
    let Some(d) = d else { return Ok(a) };
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        match key.as_str() {
            "oop_bets" => a.oop_bets = v.extract()?,
            "ip_bets" => a.ip_bets = v.extract()?,
            "raises" => a.raises = v.extract()?,
            "allin" => a.allin = v.extract()?,
            "max_bets" => a.max_bets = v.extract()?,
            "allin_threshold" => a.allin_threshold = v.extract()?,
            _ => return Err(verr(format!("unknown abstraction key {key:?}"))),
        }
    }
    Ok(a)
}

fn params(max_iters: u32, target_expl: f32, check_every: u32) -> Params {
    Params {
        max_iters,
        target_expl,
        check_every,
        ..Params::default()
    }
}

fn slice<'a>(a: &'a PyReadonlyArray1<'_, f32>, n: usize, what: &str) -> PyResult<&'a [f32]> {
    let s = a.as_slice().map_err(verr)?;
    if s.len() != n {
        return Err(verr(format!(
            "{what}: expected {n} entries, got {}",
            s.len()
        )));
    }
    Ok(s)
}

fn edge_label(e: Edge) -> String {
    match e {
        Edge::Root => "root".into(),
        Edge::Check => "check".into(),
        Edge::Call => "call".into(),
        Edge::Fold => "fold".into(),
        Edge::Bet(x) => format!("bet {x:.4}"),
        Edge::Raise(x) => format!("raise {x:.4}"),
        Edge::AllIn(x) => format!("allin {x:.4}"),
    }
}

fn tree_dict<'py>(py: Python<'py>, t: &Tree) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let kind: Vec<&str> = t
        .kind
        .iter()
        .map(|k| match k {
            NodeKind::Decision => "decision",
            NodeKind::Fold => "fold",
            NodeKind::Showdown => "showdown",
            NodeKind::Chance => "chance",
        })
        .collect();
    d.set_item("kind", PyList::new(py, kind)?)?;
    d.set_item("player", PyArray1::from_slice(py, &t.player))?;
    d.set_item("first_child", PyArray1::from_slice(py, &t.first_child))?;
    d.set_item("n_children", PyArray1::from_slice(py, &t.n_children))?;
    let contrib: Vec<f32> = t.contrib.iter().flatten().copied().collect();
    d.set_item("contrib", PyArray1::from_vec(py, contrib))?;
    d.set_item("parent", PyArray1::from_slice(py, &t.parent))?;
    d.set_item("slot", PyArray1::from_slice(py, &t.slot))?;
    d.set_item(
        "edge",
        PyList::new(py, t.edge.iter().map(|&e| edge_label(e)))?,
    )?;
    d.set_item("n_slots", (t.n_slots[0], t.n_slots[1]))?;
    d.set_item("max_actions", t.max_actions)?;
    d.set_item("stack", t.stack)?;
    Ok(d)
}

/// Public tree of a river street with effective stack `stack` (pot units), as arrays.
#[pyfunction]
#[pyo3(signature = (stack, abstraction=None))]
pub fn solver_build_tree<'py>(
    py: Python<'py>,
    stack: f32,
    abstraction: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let a = self::abstraction(abstraction)?;
    if !(stack.is_finite() && stack >= 0.0) {
        return Err(verr(format!("bad stack {stack}")));
    }
    tree_dict(py, &Tree::river(stack, &a))
}

/// Strength of every combo on a 5-card river board (0 for combos blocked by the board).
#[pyfunction]
pub fn solver_river_strengths<'py>(
    py: Python<'py>,
    board: Vec<u8>,
) -> PyResult<Bound<'py, PyArray1<u16>>> {
    let b = board5(&board)?;
    let rh = solver::RiverHands::new(&b);
    Ok(PyArray1::from_vec(py, rh.strength_1326))
}

fn board5(board: &[u8]) -> PyResult<[engine::Card; 5]> {
    if board.len() != 5 {
        return Err(verr("a river board has 5 cards"));
    }
    let mut seen = 0u64;
    for &c in board {
        if c >= 52 || seen & (1 << c) != 0 {
            return Err(verr("board cards must be distinct cards 0..52"));
        }
        seen |= 1 << c;
    }
    Ok([0, 1, 2, 3, 4].map(|i| engine::Card(board[i])))
}

fn solution_dict<'py>(
    py: Python<'py>,
    sol: &solver::Solution,
    tree: &Tree,
    unmap: impl Fn(&[f32]) -> Vec<f32>,
    out_n: usize,
    strategy: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for p in 0..2 {
        let k = p + 1;
        d.set_item(
            format!("v{k}"),
            PyArray1::from_vec(py, unmap(&sol.value[p])),
        )?;
        d.set_item(format!("m{k}"), PyArray1::from_vec(py, unmap(&sol.mass[p])))?;
        d.set_item(
            format!("cfv{k}"),
            PyArray1::from_vec(py, unmap(&sol.cfv[p])),
        )?;
        if strategy {
            let n = sol.value[p].len();
            let mut flat = Vec::with_capacity(sol.avg[p].len() / n.max(1) * out_n);
            for chunk in sol.avg[p].chunks(n.max(1)) {
                flat.extend(unmap(chunk));
            }
            d.set_item(format!("strategy{k}"), PyArray1::from_vec(py, flat))?;
        }
    }
    d.set_item("expl", sol.expl)?;
    d.set_item("iters", sol.iters)?;
    d.set_item("history", sol.history.clone())?;
    if strategy {
        d.set_item("tree", tree_dict(py, tree)?)?;
    }
    Ok(d)
}

/// Solves a river spot (see `river_net.solver.solve`). Ranges are 1326 float32 weights.
#[pyfunction]
#[pyo3(signature = (board, stack, r1, r2, abstraction=None, max_iters=500, target_expl=0.002,
                    check_every=50, strategy=false))]
#[allow(clippy::too_many_arguments)]
pub fn solver_river_solve<'py>(
    py: Python<'py>,
    board: Vec<u8>,
    stack: f32,
    r1: PyReadonlyArray1<'py, f32>,
    r2: PyReadonlyArray1<'py, f32>,
    abstraction: Option<&Bound<'py, PyDict>>,
    max_iters: u32,
    target_expl: f32,
    check_every: u32,
    strategy: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let b = board5(&board)?;
    let a = self::abstraction(abstraction)?;
    let (r1, r2) = (slice(&r1, N_COMBOS, "r1")?, slice(&r2, N_COMBOS, "r2")?);
    let p = params(max_iters, target_expl, check_every);
    let res = py
        .detach(|| solver::solve_river(&b, stack, [r1, r2], &a, &p))
        .map_err(verr)?;
    solution_dict(
        py,
        &res.solution,
        &res.tree,
        |v| res.hands.scatter(v),
        N_COMBOS,
        strategy,
    )
}

/// Solves an abstract game: hands with strengths (higher wins) and optional card pairs
/// (`cards=None` switches card removal off). Vectors are in the input hand order. Ranges
/// are used as given (not renormalised).
#[pyfunction]
#[pyo3(signature = (strength, r1, r2, stack, cards=None, abstraction=None, max_iters=500,
                    target_expl=0.0, check_every=50, strategy=true))]
#[allow(clippy::too_many_arguments)]
pub fn solver_abstract_solve<'py>(
    py: Python<'py>,
    strength: Vec<u32>,
    r1: PyReadonlyArray1<'py, f32>,
    r2: PyReadonlyArray1<'py, f32>,
    stack: f32,
    cards: Option<Vec<[u8; 2]>>,
    abstraction: Option<&Bound<'py, PyDict>>,
    max_iters: u32,
    target_expl: f32,
    check_every: u32,
    strategy: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let n = strength.len();
    let a = self::abstraction(abstraction)?;
    let (r1, r2) = (slice(&r1, n, "r1")?, slice(&r2, n, "r2")?);
    if let Some(c) = &cards {
        if c.len() != n {
            return Err(verr("one card pair per hand"));
        }
        let mut seen = std::collections::HashSet::new();
        for p in c {
            if p[0] >= 52
                || p[1] >= 52
                || p[0] == p[1]
                || !seen.insert((p[0].max(p[1]), p[0].min(p[1])))
            {
                return Err(verr(
                    "card pairs must be distinct pairs of distinct cards 0..52",
                ));
            }
        }
    }
    let p = params(max_iters, target_expl, check_every);
    let (sol, tree, order) = py.detach(|| {
        let (hs, order) = HandSet::new(&strength, cards.as_deref());
        let sorted = |r: &[f32]| order.iter().map(|&i| r[i]).collect::<Vec<f32>>();
        let (s1, s2) = (sorted(r1), sorted(r2));
        let tree = Tree::river(stack, &a);
        let sol = Solver::new(&tree, &hs, [&s1, &s2]).solve(&p);
        (sol, tree, order)
    });
    let unmap = |v: &[f32]| {
        let mut out = vec![0f32; n];
        for (s, &i) in order.iter().enumerate() {
            out[i] = v[s];
        }
        out
    };
    solution_dict(py, &sol, &tree, unmap, n, strategy)
}

/// Equity of every combo against the opponent range `opp` (1326 weights) on a flop (MC over
/// `samples` runouts) or turn (exact over rivers). Blocked combos get 0.
#[pyfunction]
#[pyo3(signature = (board, opp, samples=32, seed=0))]
pub fn solver_range_equity<'py>(
    py: Python<'py>,
    board: Vec<u8>,
    opp: PyReadonlyArray1<'py, f32>,
    samples: u32,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if board.len() != 3 && board.len() != 4 {
        return Err(verr("range_equity needs a flop or turn board"));
    }
    let mut seen = 0u64;
    for &c in &board {
        if c >= 52 || seen & (1 << c) != 0 {
            return Err(verr("board cards must be distinct cards 0..52"));
        }
        seen |= 1 << c;
    }
    let opp = slice(&opp, N_COMBOS, "opp")?;
    let cards: Vec<engine::Card> = board.iter().map(|&c| engine::Card(c)).collect();
    let eq = py.detach(|| solver::equity::range_equity(&cards, opp, samples, seed));
    Ok(PyArray1::from_vec(py, eq))
}

/// Showdown equity of every combo against the opponent range `opp` (1326 weights) on a
/// river board: (win + tie/2) / compatible mass. Blocked combos, or combos with no
/// compatible opponent mass, get 0. Returns `(equity, mass)`.
#[pyfunction]
pub fn solver_river_equity<'py>(
    py: Python<'py>,
    board: Vec<u8>,
    opp: PyReadonlyArray1<'py, f32>,
) -> PyResult<ArrayPair<'py>> {
    let b = board5(&board)?;
    let opp = slice(&opp, N_COMBOS, "opp")?;
    let (eq, mass) = py.detach(|| {
        let rh = solver::RiverHands::new(&b);
        let q = rh.gather(opp);
        let n = rh.set.len();
        let (mut net, mut mass) = (vec![0f32; n], vec![0f32; n]);
        solver::terminal::showdown_net(&rh.set, &q, &mut net);
        solver::terminal::compat_mass(&rh.set, &q, &mut mass);
        let eq: Vec<f32> = (0..n)
            .map(|h| {
                if mass[h] > 0.0 {
                    0.5 * (mass[h] + net[h]) / mass[h]
                } else {
                    0.0
                }
            })
            .collect();
        (rh.scatter(&eq), rh.scatter(&mass))
    });
    Ok((PyArray1::from_vec(py, eq), PyArray1::from_vec(py, mass)))
}

/// Flop all-in showdown matrix: flat `[1326 * 1326]` `W` with `W[h, h'] = Σ sign(str(h) −
/// str(h'))` over the unordered turn + river runouts both hands survive (0 unless disjoint
/// and valid). `W @ q` is the showdown net of every combo against opponent reach `q`
/// summed over runouts; each deal sees `C(45, 2) = 990` runouts.
#[pyfunction]
pub fn solver_flop_showdown_matrix<'py>(
    py: Python<'py>,
    board: Vec<u8>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if board.len() != 3 {
        return Err(verr("a flop board has 3 cards"));
    }
    let mut seen = 0u64;
    for &c in &board {
        if c >= 52 || seen & (1 << c) != 0 {
            return Err(verr("board cards must be distinct cards 0..52"));
        }
        seen |= 1 << c;
    }
    let out = py.detach(|| {
        let cc = solver::combo_cards();
        let mask: Vec<u64> = cc.iter().map(|[a, b]| (1u64 << a) | (1u64 << b)).collect();
        let rest: Vec<u8> = (0..52u8).filter(|&c| seen & (1 << c) == 0).collect();
        // |W| <= 990 runouts, so i16 accumulators suffice (half the memory traffic).
        let mut w = vec![0i16; N_COMBOS * N_COMBOS];
        let (mut idx, mut st, mut mk) = (Vec::new(), Vec::new(), Vec::new());
        for (i, &t) in rest.iter().enumerate() {
            for &r in &rest[i + 1..] {
                let b = [board[0], board[1], board[2], t, r].map(engine::Card);
                let rh = solver::RiverHands::new(&b);
                idx.clear();
                st.clear();
                mk.clear();
                for (h, &s) in rh.strength_1326.iter().enumerate() {
                    if s > 0 {
                        idx.push(h);
                        st.push(s as i32);
                        mk.push(mask[h]);
                    }
                }
                // Upper triangle only (row h < column g in combo order): sequential writes.
                for a in 0..idx.len() {
                    let (sa, ma) = (st[a], mk[a]);
                    let row = &mut w[idx[a] * N_COMBOS..(idx[a] + 1) * N_COMBOS];
                    for ((&g, &sg), &mg) in idx[a + 1..].iter().zip(&st[a + 1..]).zip(&mk[a + 1..])
                    {
                        let d = ((sa > sg) as i16 - (sa < sg) as i16) & -((ma & mg == 0) as i16);
                        row[g] += d;
                    }
                }
            }
        }
        let mut out = vec![0f32; N_COMBOS * N_COMBOS];
        for h in 0..N_COMBOS {
            for g in h + 1..N_COMBOS {
                let x = w[h * N_COMBOS + g] as f32;
                out[h * N_COMBOS + g] = x;
                out[g * N_COMBOS + h] = -x;
            }
        }
        out
    });
    Ok(PyArray1::from_vec(py, out))
}

/// The 48 river boards of a turn board, with cached hand sets: per-card opponent masses and
/// showdown sums for turn solving (`river_net.turn`). Vectors are 1326-combo f32 arrays.
#[pyclass(module = "pokerarena._native", name = "TurnBoards")]
pub struct TurnBoards {
    cards: Vec<u8>,
    rivers: Vec<solver::RiverHands>,
}

#[pymethods]
impl TurnBoards {
    #[new]
    fn new(py: Python<'_>, board: Vec<u8>) -> PyResult<Self> {
        if board.len() != 4 {
            return Err(verr("a turn board has 4 cards"));
        }
        let mut seen = 0u64;
        for &c in &board {
            if c >= 52 || seen & (1 << c) != 0 {
                return Err(verr("board cards must be distinct cards 0..52"));
            }
            seen |= 1 << c;
        }
        let cards: Vec<u8> = (0..52u8).filter(|&c| seen & (1 << c) == 0).collect();
        let rivers = py.detach(|| {
            cards
                .iter()
                .map(|&c| {
                    let b = [board[0], board[1], board[2], board[3], c].map(engine::Card);
                    solver::RiverHands::new(&b)
                })
                .collect()
        });
        Ok(TurnBoards { cards, rivers })
    }

    /// River cards, in the order used by every per-card output.
    #[getter]
    fn cards(&self) -> Vec<u8> {
        self.cards.clone()
    }

    /// Strength of every combo on each river board, flat `[cards, 1326]` (0 = blocked).
    fn strengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u16>> {
        let flat: Vec<u16> = self
            .rivers
            .iter()
            .flat_map(|r| r.strength_1326.iter().copied())
            .collect();
        PyArray1::from_vec(py, flat)
    }

    /// Compatible mass of opponent reach `q[l]` for every combo on every river card:
    /// flat `[L, cards, 1326]` (0 for combos blocked by the river board). `q` is `[L*1326]`.
    fn masses<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let q = q.as_slice().map_err(verr)?;
        if q.len() % N_COMBOS != 0 {
            return Err(verr("q must be [L * 1326]"));
        }
        let l = q.len() / N_COMBOS;
        let nc = self.cards.len();
        let out = py.detach(|| {
            let mut out = vec![0f32; l * nc * N_COMBOS];
            let mut sc = solver::terminal::TermScratch::default();
            for li in 0..l {
                let ql = &q[li * N_COMBOS..(li + 1) * N_COMBOS];
                for (ci, rh) in self.rivers.iter().enumerate() {
                    let g = rh.gather(ql);
                    let mut m = vec![0f32; g.len()];
                    rh.set.index.prepare(&g, &mut sc);
                    rh.set.index.compat(&g, &sc, &mut m);
                    let base = (li * nc + ci) * N_COMBOS;
                    for (&c, &x) in rh.combo.iter().zip(&m) {
                        out[base + c as usize] = x;
                    }
                }
            }
            out
        });
        Ok(PyArray1::from_vec(py, out))
    }

    /// Turn-bucket features of every combo: `(EHS, E[HS²])` with `HS_c(h)` the river equity
    /// against a uniform random hand on `board + c`, averaged over the river cards `h`
    /// doesn't hold (0 for blocked combos). Same numbers as the Python reference
    /// `river_net.preriver.turn_features`.
    fn hs_features<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let (e1, e2) = py.detach(|| {
            let (mut s1, mut s2, mut n) = (
                vec![0f64; N_COMBOS],
                vec![0f64; N_COMBOS],
                vec![0u32; N_COMBOS],
            );
            for rh in &self.rivers {
                let m = rh.set.len();
                let u = 1.0f32 / m as f32;
                let q = vec![u; m];
                let (mut net, mut mass) = (vec![0f32; m], vec![0f32; m]);
                solver::terminal::showdown_net(&rh.set, &q, &mut net);
                solver::terminal::compat_mass(&rh.set, &q, &mut mass);
                for (i, &c) in rh.combo.iter().enumerate() {
                    let eq = if mass[i] > 0.0 {
                        0.5 * (mass[i] + net[i]) / mass[i]
                    } else {
                        0.0
                    };
                    let e = eq as f64;
                    s1[c as usize] += e;
                    s2[c as usize] += e * e;
                    n[c as usize] += 1;
                }
            }
            let f = |s: &[f64]| -> Vec<f64> {
                s.iter()
                    .zip(&n)
                    .map(|(&x, &k)| if k > 0 { x / k as f64 } else { 0.0 })
                    .collect()
            };
            (f(&s1), f(&s2))
        });
        (PyArray1::from_vec(py, e1), PyArray1::from_vec(py, e2))
    }

    /// The linear map behind [`showdown_net_sum`](Self::showdown_net_sum): flat
    /// `[1326 * 1326]` matrix `W` with `W[h, h'] = Σ_c sign(str_c(h) − str_c(h'))` over the
    /// river cards `c` both hands survive (0 unless `h`, `h'` are disjoint and valid), so
    /// that `showdown_net_sum(q) = W @ q`. Lets a solver value many all-in nodes with one
    /// matrix product.
    fn showdown_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let out = py.detach(|| {
            let cc = solver::combo_cards();
            let mask: Vec<u64> = cc.iter().map(|[a, b]| (1u64 << a) | (1u64 << b)).collect();
            let mut w = vec![0i32; N_COMBOS * N_COMBOS];
            for rh in &self.rivers {
                let hs: Vec<(usize, u16)> = rh
                    .strength_1326
                    .iter()
                    .enumerate()
                    .filter(|(_, &s)| s > 0)
                    .map(|(h, &s)| (h, s))
                    .collect();
                for (i, &(h, sh)) in hs.iter().enumerate() {
                    for &(g, sg) in &hs[i + 1..] {
                        if mask[h] & mask[g] != 0 || sh == sg {
                            continue;
                        }
                        let s = if sh > sg { 1 } else { -1 };
                        w[h * N_COMBOS + g] += s;
                        w[g * N_COMBOS + h] -= s;
                    }
                }
            }
            w.into_iter().map(|x| x as f32).collect::<Vec<f32>>()
        });
        PyArray1::from_vec(py, out)
    }

    /// Showdown net (weaker − stronger compatible mass of `q`) summed over the river cards,
    /// for every combo: all-in leaves average this over the 44 cards each deal can see.
    /// `cards`: indices of the river cards to sum over (default all).
    #[pyo3(signature = (q, cards=None))]
    fn showdown_net_sum<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f32>,
        cards: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let q = slice(&q, N_COMBOS, "q")?;
        let sel: Vec<usize> = cards.unwrap_or_else(|| (0..self.rivers.len()).collect());
        if sel.iter().any(|&i| i >= self.rivers.len()) {
            return Err(verr("card index out of range"));
        }
        let out = py.detach(|| {
            let mut acc = vec![0f64; N_COMBOS];
            let mut sc = solver::terminal::TermScratch::default();
            for rh in sel.iter().map(|&i| &self.rivers[i]) {
                let g = rh.gather(q);
                let mut net = vec![0f32; g.len()];
                rh.set.index.prepare(&g, &mut sc);
                rh.set.index.net(&sc, &mut net);
                for (&c, &x) in rh.combo.iter().zip(&net) {
                    acc[c as usize] += x as f64;
                }
            }
            acc.into_iter().map(|x| x as f32).collect::<Vec<f32>>()
        });
        Ok(PyArray1::from_vec(py, out))
    }
}

/// Batched river-leaf kernels of one turn board (see `solver::leaf`): network inputs and the
/// broadcast of per-card bucket values back to counterfactual values, for `L` leaves at once.
#[pyclass(module = "pokerarena._native", name = "LeafIndex")]
pub struct LeafIndex {
    ix: solver::leaf::LeafIndex,
    k: usize,
}

fn leaf_ranges<'a>(
    r1: &'a PyReadonlyArray1<'_, f64>,
    r2: &'a PyReadonlyArray1<'_, f64>,
) -> PyResult<(&'a [f64], &'a [f64], usize)> {
    let (a, b) = (r1.as_slice().map_err(verr)?, r2.as_slice().map_err(verr)?);
    if a.len() != b.len() || a.len() % N_COMBOS != 0 {
        return Err(verr("r1 and r2 must both be [L * 1326]"));
    }
    Ok((a, b, a.len() / N_COMBOS))
}

impl LeafIndex {
    fn check_sel(&self, cards: Option<Vec<usize>>) -> PyResult<Option<Vec<usize>>> {
        if let Some(c) = &cards {
            if c.iter().any(|&i| i >= self.ix.n_cards()) {
                return Err(verr("card index out of range"));
            }
        }
        Ok(cards)
    }

    fn rows(&self, sel: &Option<Vec<usize>>) -> usize {
        sel.as_ref().map_or(self.ix.n_cards(), |s| s.len())
    }
}

#[pymethods]
impl LeafIndex {
    /// `cards`: river cards (e.g. `TurnBoards.cards`); `bmap`: `[cards * 1326]` bucket of
    /// every combo on each river board (negative = blocked); `k`: buckets per board.
    #[new]
    fn new(cards: Vec<u8>, bmap: PyReadonlyArray1<'_, i32>, k: usize) -> PyResult<Self> {
        let ix = solver::leaf::LeafIndex::new(&cards, bmap.as_slice().map_err(verr)?, k)
            .map_err(verr)?;
        Ok(LeafIndex { ix, k })
    }

    /// Network inputs `(x, w)` for `L` leaves, each flat `[L * cards * 2K]`: normalised
    /// bucket ranges and projection weights of both players. `r1`, `r2`: leaf reach
    /// `[L * 1326]` (0 on combos blocked by the turn board); `q1`, `q2`: the reach the
    /// opponent masses come from (normally `r1`, `r2`); `cards`: indices of the river cards
    /// to evaluate (default all; outputs then have `len(cards)` card rows).
    #[pyo3(signature = (r1, r2, q1, q2, cards=None))]
    fn inputs<'py>(
        &self,
        py: Python<'py>,
        r1: PyReadonlyArray1<'py, f64>,
        r2: PyReadonlyArray1<'py, f64>,
        q1: PyReadonlyArray1<'py, f64>,
        q2: PyReadonlyArray1<'py, f64>,
        cards: Option<Vec<usize>>,
    ) -> PyResult<ArrayPair<'py>> {
        let sel = self.check_sel(cards)?;
        let (a, b, l) = leaf_ranges(&r1, &r2)?;
        let (qa, qb, lq) = leaf_ranges(&q1, &q2)?;
        if lq != l {
            return Err(verr("q1, q2 must match r1, r2"));
        }
        let n = self.rows(&sel) * 2 * self.k;
        let (x, w) = py.detach(|| {
            let (mut x, mut w) = (vec![0f32; l * n], vec![0f32; l * n]);
            for li in 0..l {
                let rs = li * N_COMBOS..(li + 1) * N_COMBOS;
                self.ix.inputs(
                    sel.as_deref(),
                    [&a[rs.clone()], &b[rs.clone()]],
                    [&qa[rs.clone()], &qb[rs]],
                    &mut x[li * n..(li + 1) * n],
                    &mut w[li * n..(li + 1) * n],
                );
            }
            (x, w)
        });
        Ok((PyArray1::from_vec(py, x), PyArray1::from_vec(py, w)))
    }

    /// Counterfactual leaf values `(cfv1, cfv2)`, each `[L * 1326]`, from per-card bucket
    /// values `y` (`[L * cards * 2K]`, conditional, river-pot units) and `scale[L]`;
    /// `cards`: the river cards `y` covers (default all).
    #[pyo3(signature = (y, r1, r2, scale, cards=None))]
    fn cfv<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f32>,
        r1: PyReadonlyArray1<'py, f64>,
        r2: PyReadonlyArray1<'py, f64>,
        scale: Vec<f64>,
        cards: Option<Vec<usize>>,
    ) -> PyResult<ArrayPair64<'py>> {
        let sel = self.check_sel(cards)?;
        let (a, b, l) = leaf_ranges(&r1, &r2)?;
        let n = self.rows(&sel) * 2 * self.k;
        let y = slice(&y, l * n, "y")?;
        if scale.len() != l {
            return Err(verr("scale must have one entry per leaf"));
        }
        let (o1, o2) = py.detach(|| {
            let (mut o1, mut o2) = (vec![0f64; l * N_COMBOS], vec![0f64; l * N_COMBOS]);
            for li in 0..l {
                let rs = li * N_COMBOS..(li + 1) * N_COMBOS;
                let (p1, p2) = (&mut o1[rs.clone()], &mut o2[rs.clone()]);
                self.ix.cfv(
                    sel.as_deref(),
                    &y[li * n..(li + 1) * n],
                    [&a[rs.clone()], &b[rs]],
                    scale[li],
                    [p1, p2],
                );
            }
            (o1, o2)
        });
        Ok((PyArray1::from_vec(py, o1), PyArray1::from_vec(py, o2)))
    }
}

/// Registers the solver functions on `_native`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solver_build_tree, m)?)?;
    m.add_function(wrap_pyfunction!(solver_river_strengths, m)?)?;
    m.add_function(wrap_pyfunction!(solver_river_solve, m)?)?;
    m.add_function(wrap_pyfunction!(solver_abstract_solve, m)?)?;
    m.add_function(wrap_pyfunction!(solver_range_equity, m)?)?;
    m.add_function(wrap_pyfunction!(solver_river_equity, m)?)?;
    m.add_function(wrap_pyfunction!(solver_flop_showdown_matrix, m)?)?;
    m.add_class::<TurnBoards>()?;
    m.add_class::<LeafIndex>()?;
    Ok(())
}
