//! PyO3 bindings: `pokerarena._native`.
//!
//! * Python bots (any object with an `act(obs)` method; `new_match`, `new_hand`, `observe`,
//!   `end_hand` optional) are wrapped in [`PyBot`], which implements the Rust `Bot` trait
//!   and attaches to the interpreter only for the duration of each call.
//! * [`PyRun`] exposes a run's schedule, table execution (GIL released; Rust-only tables run
//!   fully in parallel) and round merging, so the Python driver can farm Python tables out
//!   to worker processes and merge results in table-id order.

use arena::bot::{Action, Bot, HandResult, HandStart, MatchInfo, Observation, PublicEvent};
use arena::config::ConfigFile;
use arena::leaderboard::{render, utc_timestamp, write_results_dir};
use arena::replays::{write_replays, ReplayPlan};
use arena::runner::{build_pool, table_error, table_seed, RoundRecord, RunError};
use arena::{
    run_table_with_hooks, BotFactory, LeaderboardFile, RunProgress, RunSpec, RunState, TableHooks,
    TableResult,
};
use engine::action::ActionKind;
use engine::eval::{category, Evaluator};
use engine::format::Variant;
use engine::{Card, SeatStatus};
use numpy::PyArray1;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

mod solver;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

// ------------------------------------------------------------------------------- Action

/// A bot action. Amounts use the raise-to convention (total street contribution).
#[pyclass(frozen, eq, module = "pokerarena._native", name = "Action")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyAction {
    /// One of fold, check, call, bet_to, raise_to, all_in.
    #[pyo3(get)]
    kind: String,
    /// Raise-to amount for bet_to / raise_to (0 otherwise).
    #[pyo3(get)]
    amount: u64,
}

const ACTION_KINDS: [&str; 6] = ["fold", "check", "call", "bet_to", "raise_to", "all_in"];

#[pymethods]
impl PyAction {
    #[new]
    #[pyo3(signature = (kind, amount = 0))]
    fn new(kind: &str, amount: u64) -> PyResult<Self> {
        if !ACTION_KINDS.contains(&kind) {
            return Err(PyValueError::new_err(format!(
                "unknown action kind {kind:?} (expected one of {ACTION_KINDS:?})"
            )));
        }
        Ok(PyAction {
            kind: kind.to_string(),
            amount,
        })
    }
    /// Fold.
    #[staticmethod]
    fn fold() -> Self {
        Self::mk("fold", 0)
    }
    /// Check.
    #[staticmethod]
    fn check() -> Self {
        Self::mk("check", 0)
    }
    /// Call (a check when there is nothing to call).
    #[staticmethod]
    fn call() -> Self {
        Self::mk("call", 0)
    }
    /// Bet to a street total.
    #[staticmethod]
    fn bet_to(amount: u64) -> Self {
        Self::mk("bet_to", amount)
    }
    /// Raise to a street total.
    #[staticmethod]
    fn raise_to(amount: u64) -> Self {
        Self::mk("raise_to", amount)
    }
    /// Commit the whole stack.
    #[staticmethod]
    fn all_in() -> Self {
        Self::mk("all_in", 0)
    }
    fn __repr__(&self) -> String {
        match self.kind.as_str() {
            "bet_to" | "raise_to" => format!("Action.{}({})", self.kind, self.amount),
            k => format!("Action.{k}()"),
        }
    }
}

impl PyAction {
    fn mk(kind: &str, amount: u64) -> Self {
        PyAction {
            kind: kind.into(),
            amount,
        }
    }
    fn to_action(&self) -> Action {
        match self.kind.as_str() {
            "fold" => Action::Fold,
            "check" => Action::Check,
            "call" => Action::Call,
            "bet_to" => Action::BetTo(self.amount),
            "raise_to" => Action::RaiseTo(self.amount),
            _ => Action::AllIn,
        }
    }
}

/// Accepts an `Action`, a kind string (`"call"`) or a `(kind, amount)` tuple.
fn parse_action(obj: &Bound<'_, PyAny>) -> PyResult<Action> {
    if let Ok(a) = obj.extract::<PyRef<'_, PyAction>>() {
        return Ok(a.to_action());
    }
    if let Ok(k) = obj.extract::<String>() {
        return Ok(PyAction::new(&k, 0)?.to_action());
    }
    if let Ok((k, amt)) = obj.extract::<(String, u64)>() {
        return Ok(PyAction::new(&k, amt)?.to_action());
    }
    Err(PyValueError::new_err(format!(
        "act() must return an Action, a kind string or a (kind, amount) tuple, got {}",
        obj.repr()?
    )))
}

// ------------------------------------------------------------------------ Observations

/// What the player to act may do.
#[pyclass(frozen, get_all, module = "pokerarena._native", name = "LegalActions")]
#[derive(Clone, Debug)]
pub struct PyLegalActions {
    can_check: bool,
    call_amount: u64,
    can_raise: bool,
    min_raise_to: u64,
    max_raise_to: u64,
}

#[pymethods]
impl PyLegalActions {
    fn __repr__(&self) -> String {
        format!(
            "LegalActions(can_check={}, call_amount={}, can_raise={}, min_raise_to={}, max_raise_to={})",
            self.can_check, self.call_amount, self.can_raise, self.min_raise_to, self.max_raise_to
        )
    }
}

/// A player's view of the hand: own hole cards and public information only.
///
/// Cards are ints `0..52` (`rank*4 + suit`, rank 0 = deuce, suits c d h s); use
/// `card_to_str`. Array fields are NumPy arrays indexed by seat.
#[pyclass(frozen, get_all, module = "pokerarena._native", name = "Observation")]
pub struct PyObservation {
    seat: usize,
    n_seats: usize,
    button: usize,
    /// 0 preflop, 1 flop, 2 turn, 3 river.
    street: u8,
    variant: &'static str,
    max_seats: u8,
    hole: Py<PyArray1<u8>>,
    board: Py<PyArray1<u8>>,
    stacks: Py<PyArray1<u64>>,
    street_contrib: Py<PyArray1<u64>>,
    total_contrib: Py<PyArray1<u64>>,
    /// 0 empty, 1 active, 2 all-in, 3 folded.
    status: Py<PyArray1<u8>>,
    pot: u64,
    current_bet: u64,
    sb: u64,
    bb: u64,
    ante: u64,
    legal: Py<PyLegalActions>,
    /// List of `(seat, street, kind, amount, to, all_in)`.
    history: Py<PyList>,
}

#[pymethods]
impl PyObservation {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Observation(seat={}, street={}, hole={}, board={}, pot={}, legal={})",
            self.seat,
            self.street,
            self.hole.bind(py).repr()?,
            self.board.bind(py).repr()?,
            self.pot,
            self.legal.bind(py).repr()?
        ))
    }
}

fn cards_u8(c: &[Card]) -> Vec<u8> {
    c.iter().map(|c| c.0).collect()
}

fn kind_str(k: ActionKind) -> &'static str {
    match k {
        ActionKind::PostAnte => "ante",
        ActionKind::PostSmallBlind => "sb",
        ActionKind::PostBigBlind => "bb",
        ActionKind::Fold => "fold",
        ActionKind::Check => "check",
        ActionKind::Call => "call",
        ActionKind::Bet => "bet",
        ActionKind::Raise => "raise",
    }
}

fn status_u8(s: SeatStatus) -> u8 {
    match s {
        SeatStatus::Empty => 0,
        SeatStatus::Active => 1,
        SeatStatus::AllIn => 2,
        SeatStatus::Folded => 3,
    }
}

fn variant_str(v: Variant) -> &'static str {
    engine::rules::rules(v).name()
}

fn build_observation(py: Python<'_>, o: &Observation) -> PyResult<PyObservation> {
    let history = PyList::empty(py);
    for r in o.history {
        history.append((
            r.seat,
            r.street as u8,
            kind_str(r.kind),
            r.amount,
            r.to,
            r.all_in,
        ))?;
    }
    let la = o.legal;
    Ok(PyObservation {
        seat: o.seat,
        n_seats: o.n_seats,
        button: o.button,
        street: o.street as u8,
        variant: variant_str(o.format.variant),
        max_seats: o.format.max_seats,
        hole: PyArray1::from_vec(py, cards_u8(o.hole)).unbind(),
        board: PyArray1::from_vec(py, cards_u8(o.board)).unbind(),
        stacks: PyArray1::from_slice(py, o.stacks).unbind(),
        street_contrib: PyArray1::from_slice(py, o.street_contrib).unbind(),
        total_contrib: PyArray1::from_slice(py, o.total_contrib).unbind(),
        status: PyArray1::from_vec(py, o.status.iter().map(|&s| status_u8(s)).collect()).unbind(),
        pot: o.pot,
        current_bet: o.current_bet,
        sb: o.sb,
        bb: o.bb,
        ante: o.ante,
        legal: Py::new(
            py,
            PyLegalActions {
                can_check: la.can_check,
                call_amount: la.call_amount,
                can_raise: la.can_raise,
                min_raise_to: la.min_raise_to,
                max_raise_to: la.max_raise_to,
            },
        )?,
        history: history.unbind(),
    })
}

// ------------------------------------------------------------------------------ PyBot

/// Wraps a Python bot object as a Rust [`Bot`].
struct PyBot {
    name: String,
    obj: Py<PyAny>,
    has_new_match: bool,
    has_new_hand: bool,
    has_observe: bool,
    has_end_hand: bool,
    reported: bool,
}

impl PyBot {
    fn new(py: Python<'_>, name: String, obj: Py<PyAny>) -> PyBot {
        let b = obj.bind(py);
        let cls = b.get_type();
        let base = py
            .import("pokerarena.bot")
            .and_then(|m| m.getattr("BaseBot"))
            .ok();
        // A hook counts only if the bot defines it and it is not BaseBot's no-op: calling
        // inherited no-ops would cost a dict build and an interpreter call per event.
        let has = |m: &str| {
            if !b.hasattr(m).unwrap_or(false) {
                return false;
            }
            match (&base, cls.getattr(m)) {
                (Some(base), Ok(f)) => base.getattr(m).map_or(true, |bf| !f.is(&bf)),
                _ => true,
            }
        };
        PyBot {
            name,
            has_new_match: has("new_match"),
            has_new_hand: has("new_hand"),
            has_observe: has("observe"),
            has_end_hand: has("end_hand"),
            obj,
            reported: false,
        }
    }

    /// Reports the first exception raised by this bot (later ones are only counted as
    /// illegal actions / ignored hooks).
    fn report(&mut self, py: Python<'_>, hook: &str, e: PyErr) {
        if !self.reported {
            self.reported = true;
            eprintln!("pokerarena: bot `{}` raised in {hook}():", self.name);
            e.print(py);
        }
    }

    fn call_hook(
        &mut self,
        hook: &str,
        build: impl FnOnce(Python<'_>) -> PyResult<Bound<'_, PyDict>>,
    ) {
        Python::attach(|py| {
            let r = build(py).and_then(|d| self.obj.call_method1(py, hook, (d,)));
            if let Err(e) = r {
                self.report(py, hook, e);
            }
        });
    }
}

impl Bot for PyBot {
    fn name(&self) -> &str {
        &self.name
    }

    fn new_match(&mut self, info: &MatchInfo) {
        if !self.has_new_match {
            return;
        }
        let info = *info;
        self.call_hook("new_match", |py| {
            let d = PyDict::new(py);
            d.set_item("variant", variant_str(info.format.variant))?;
            d.set_item("max_seats", info.format.max_seats)?;
            d.set_item("n_seats", info.n_seats)?;
            d.set_item("seat", info.seat)?;
            d.set_item("sb", info.sb)?;
            d.set_item("bb", info.bb)?;
            d.set_item("ante", info.ante)?;
            d.set_item("starting_stack", info.starting_stack)?;
            d.set_item("hands", info.hands)?;
            d.set_item("tournament", info.tournament)?;
            d.set_item("duplicate", info.duplicate)?;
            Ok(d)
        });
    }

    fn new_hand(&mut self, info: &HandStart) {
        if !self.has_new_hand {
            return;
        }
        self.call_hook("new_hand", |py| {
            let d = PyDict::new(py);
            d.set_item("hand_id", info.hand_id)?;
            d.set_item("seat", info.seat)?;
            d.set_item("hole", PyArray1::from_vec(py, cards_u8(info.hole)))?;
            d.set_item("button", info.button)?;
            d.set_item("stacks", PyArray1::from_slice(py, info.stacks))?;
            d.set_item("sb", info.sb)?;
            d.set_item("bb", info.bb)?;
            d.set_item("ante", info.ante)?;
            Ok(d)
        });
    }

    fn act(&mut self, obs: &Observation) -> Action {
        Python::attach(|py| {
            let r = build_observation(py, obs)
                .and_then(|o| self.obj.call_method1(py, "act", (o,)))
                .and_then(|a| parse_action(a.bind(py)));
            match r {
                Ok(a) => a,
                Err(e) => {
                    self.report(py, "act", e);
                    // Always illegal: handled by the table's illegal-action policy.
                    Action::BetTo(0)
                }
            }
        })
    }

    fn observe(&mut self, ev: &PublicEvent) {
        if !self.has_observe {
            return;
        }
        let ev = *ev;
        self.call_hook("observe", |py| {
            let d = PyDict::new(py);
            match ev {
                PublicEvent::Action(r) => {
                    d.set_item("type", "action")?;
                    d.set_item("seat", r.seat)?;
                    d.set_item("street", r.street as u8)?;
                    d.set_item("kind", kind_str(r.kind))?;
                    d.set_item("amount", r.amount)?;
                    d.set_item("to", r.to)?;
                    d.set_item("all_in", r.all_in)?;
                }
                PublicEvent::Street { street, board } => {
                    d.set_item("type", "street")?;
                    d.set_item("street", street as u8)?;
                    d.set_item("board", PyArray1::from_vec(py, cards_u8(board)))?;
                }
                PublicEvent::Showdown {
                    seat,
                    hole,
                    strength,
                } => {
                    d.set_item("type", "showdown")?;
                    d.set_item("seat", seat)?;
                    d.set_item("hole", PyArray1::from_vec(py, cards_u8(hole)))?;
                    d.set_item("strength", strength)?;
                }
                PublicEvent::PotAwarded { seat, amount } => {
                    d.set_item("type", "pot_awarded")?;
                    d.set_item("seat", seat)?;
                    d.set_item("amount", amount)?;
                }
                PublicEvent::UncalledReturned { seat, amount } => {
                    d.set_item("type", "uncalled_returned")?;
                    d.set_item("seat", seat)?;
                    d.set_item("amount", amount)?;
                }
            }
            Ok(d)
        });
    }

    fn end_hand(&mut self, r: &HandResult) {
        if !self.has_end_hand {
            return;
        }
        self.call_hook("end_hand", |py| {
            let d = PyDict::new(py);
            d.set_item("hand_id", r.hand_id)?;
            d.set_item("net", PyArray1::from_slice(py, r.net))?;
            d.set_item("stacks", PyArray1::from_slice(py, r.stacks))?;
            d.set_item("board", PyArray1::from_vec(py, cards_u8(r.board)))?;
            d.set_item("showdown", r.showdown)?;
            Ok(d)
        });
    }
}

/// Builds a factory for `module.ClassName`. The class is constructed as
/// `ClassName(seed=<int>, **params)` for every table seat.
fn python_factory(
    py: Python<'_>,
    name: &str,
    target: &str,
    params: &serde_json::Value,
) -> PyResult<BotFactory> {
    let (module, class) = target.rsplit_once('.').ok_or_else(|| {
        PyValueError::new_err(format!(
            "python bot kind must be python:module.ClassName, got {target:?}"
        ))
    })?;
    let cls: Py<PyAny> = py.import(module)?.getattr(class)?.unbind();
    let params_json = match params {
        serde_json::Value::Null => "{}".to_string(),
        serde_json::Value::Object(_) => params.to_string(),
        _ => return Err(PyValueError::new_err("params must be a table")),
    };
    let make = move |py: Python<'_>, seed: u64| -> PyResult<Py<PyAny>> {
        let kwargs = py
            .import("json")?
            .call_method1("loads", (params_json.as_str(),))?
            .cast_into::<PyDict>()?;
        kwargs.set_item("seed", seed)?;
        Ok(cls.bind(py).call((), Some(&kwargs))?.unbind())
    };
    // Validate eagerly so configuration errors surface before any table runs.
    make(py, 0).map_err(|e| {
        PyRuntimeError::new_err(format!(
            "cannot construct python bot `{name}` ({target}): {e}"
        ))
    })?;
    let name = name.to_string();
    Ok(Arc::new(move |seed| {
        Python::attach(|py| {
            let obj = make(py, seed).unwrap_or_else(|e| {
                e.print(py);
                panic!("python bot `{name}` constructor failed")
            });
            Box::new(PyBot::new(py, name.clone(), obj)) as Box<dyn Bot>
        })
    }))
}

// -------------------------------------------------------------------------------- Run

/// A configured run: schedules, table execution and round merging.
#[pyclass(module = "pokerarena._native", name = "Run")]
pub struct PyRun {
    cfg_text: String,
    cfg: ConfigFile,
    spec: RunSpec,
    factories: Vec<BotFactory>,
    is_python: Vec<bool>,
    state: RunState,
    rounds: Vec<RoundRecord>,
    lb_path: PathBuf,
    lb: LeaderboardFile,
    progress: RunProgress,
    replay_plan: ReplayPlan,
    initial_ratings: Vec<f64>,
    run_id: String,
}

create_exception!(
    _native,
    Cancelled,
    pyo3::exceptions::PyException,
    "Raised by Run.run_tables when the run was cancelled with Run.cancel()."
);

fn run_err(e: RunError) -> PyErr {
    match e {
        RunError::Cancelled => Cancelled::new_err("run cancelled"),
        e => PyRuntimeError::new_err(e.to_string()),
    }
}

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pymethods]
impl PyRun {
    /// Parses a TOML config (text), applies overrides and builds every bot factory.
    /// Ratings start from `leaderboard` (default: the config's) unless `fresh` or
    /// `load_leaderboard=False`.
    #[new]
    #[pyo3(signature = (config_text, *, seed=None, rounds=None, threads=None, leaderboard=None, fresh=false, load_leaderboard=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        config_text: String,
        seed: Option<u64>,
        rounds: Option<usize>,
        threads: Option<usize>,
        leaderboard: Option<PathBuf>,
        fresh: bool,
        load_leaderboard: bool,
    ) -> PyResult<Self> {
        // TOML text, or a JSON object with the same structure (as built by the GUI).
        let (mut cfg, config_text) = if config_text.trim_start().starts_with('{') {
            let mut cfg: ConfigFile = serde_json::from_str(&config_text).map_err(err)?;
            for b in &mut cfg.bots {
                if b.params.is_null() {
                    b.params = serde_json::json!({});
                }
            }
            let text = toml::to_string(&cfg).map_err(err)?;
            (cfg, text)
        } else {
            (ConfigFile::parse(&config_text).map_err(err)?, config_text)
        };
        if let Some(s) = seed {
            cfg.run.seed = s;
        }
        if let Some(r) = rounds {
            cfg.run.rounds = r;
        }
        if threads.is_some() {
            cfg.run.threads = threads;
        }
        let spec = cfg.run_spec().map_err(err)?;
        let mut factories = Vec::new();
        let mut is_python = Vec::new();
        for b in &cfg.bots {
            // All bots are Python bots (the Rust baseline bots were removed; their Python
            // ports live in pokerarena.baselines).
            let Some(t) = b.python_target() else {
                return Err(err(format!(
                    "bot `{}`: kind `{}` is not a Python bot. Use python:module.ClassName, e.g. \
                     python:pokerarena.baselines.TightAggressive / RandomLegal / CallingStation",
                    b.name, b.kind
                )));
            };
            factories.push(python_factory(py, &b.name, t, &b.params)?);
            is_python.push(true);
        }
        let keys: Vec<String> = cfg.bots.iter().map(|b| b.key()).collect();
        let lb_path = leaderboard.unwrap_or_else(|| PathBuf::from(&cfg.run.leaderboard));
        let mut lb = if load_leaderboard {
            LeaderboardFile::load(&lb_path).map_err(err)?
        } else {
            LeaderboardFile::default()
        };
        if fresh {
            for k in &keys {
                lb.entries.remove(k);
            }
        }
        let initial = lb.initial_ratings(&keys, spec.rating.initial);
        let mut state = RunState::new(spec.clone(), keys, initial).map_err(err)?;
        let anchors: Vec<Option<f64>> = cfg.bots.iter().map(|b| b.anchor).collect();
        state.set_anchors(&anchors).map_err(err)?;
        let replay_plan = state.replay_plan().map_err(err)?;
        let initial_ratings = state.ratings().to_vec();
        let run_id = format!("{}-s{}", utc_timestamp(), spec.seed);
        Ok(PyRun {
            cfg_text: config_text,
            cfg,
            spec,
            factories,
            is_python,
            state,
            rounds: Vec::new(),
            lb_path,
            lb,
            progress: RunProgress::hidden(),
            replay_plan,
            initial_ratings,
            run_id,
        })
    }

    /// Number of rounds.
    #[getter]
    fn n_rounds(&self) -> usize {
        self.spec.rounds
    }
    /// Master seed.
    #[getter]
    fn seed(&self) -> u64 {
        self.spec.seed
    }
    /// Worker threads from the config (None = all cores).
    #[getter]
    fn threads(&self) -> Option<usize> {
        self.spec.threads
    }
    /// Bot keys in config order.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.state.keys().to_vec()
    }
    /// Current ratings in config order.
    #[getter]
    fn ratings(&self) -> Vec<f64> {
        self.state.ratings().to_vec()
    }
    /// Starting rating of each bot (config order): its anchor, else its stored leaderboard
    /// rating (when loaded), else the configured initial rating.
    #[getter]
    fn initial_ratings(&self) -> Vec<f64> {
        self.initial_ratings.clone()
    }
    /// Anchor of each bot (config order), or None.
    #[getter]
    fn anchors(&self) -> Vec<Option<f64>> {
        self.cfg.bots.iter().map(|b| b.anchor).collect()
    }
    /// Id of this run (used for results/ and hand_replays/ folders).
    #[getter]
    fn run_id(&self) -> String {
        self.run_id.clone()
    }
    /// Hands per bot sampled for hand replays (0 = off).
    #[getter]
    fn replay_hands_per_bot(&self) -> usize {
        self.spec.replay_hands_per_bot
    }
    /// Writes `<replays_dir>/<run_id>/<bot>.txt` (sampled hands of every bot) after the
    /// last round. Returns the folder, or None when replays are off.
    #[pyo3(signature = (replays_dir=None))]
    fn write_replays(&self, replays_dir: Option<PathBuf>) -> PyResult<Option<String>> {
        if self.spec.replay_hands_per_bot == 0 {
            return Ok(None);
        }
        let base = replays_dir.unwrap_or_else(|| PathBuf::from(&self.cfg.run.replays_dir));
        let dir = write_replays(
            &base,
            &self.run_id,
            self.state.keys(),
            &self.replay_plan,
            &self.rounds,
        )
        .map_err(err)?;
        Ok(Some(dir.display().to_string()))
    }
    /// True for tournament runs.
    #[getter]
    fn is_tournament(&self) -> bool {
        self.spec.table.is_tournament()
    }
    /// Hands per table (an upper bound for tournaments).
    #[getter]
    fn hands_per_table(&self) -> u64 {
        self.spec.table.hands
    }
    /// The effective config as TOML text (after overrides are applied on the Rust side).
    #[getter]
    fn config_toml(&self) -> String {
        self.cfg_text.clone()
    }
    /// Hands finished by `run_tables` calls on this object so far.
    #[getter]
    fn hands_done(&self) -> u64 {
        self.progress.hands_done.load(Ordering::Relaxed)
    }
    /// True once `cancel()` was called.
    #[getter]
    fn cancelled(&self) -> bool {
        self.progress.is_cancelled()
    }
    /// Stops every table running in `run_tables` before its next hand (they raise
    /// `Cancelled`); later `run_tables` calls raise immediately. Safe to call from any
    /// thread at any time; irreversible.
    fn cancel(&self) {
        self.progress.request_cancel();
    }

    /// Which bots are Python bots (config order).
    #[getter]
    fn python_bots(&self) -> Vec<bool> {
        self.is_python.clone()
    }

    /// The schedule of `round`: a list of tables, each a list of bot indices by seat.
    fn schedule(&self, round: usize) -> PyResult<Vec<Vec<usize>>> {
        Ok(self.state.schedule(round).map_err(err)?.tables)
    }

    /// Ids of the tables of `round` that seat at least one Python bot.
    fn python_tables(&self, round: usize) -> PyResult<Vec<usize>> {
        let s = self.state.schedule(round).map_err(err)?;
        Ok((0..s.tables.len())
            .filter(|&t| s.tables[t].iter().any(|&b| self.is_python[b]))
            .collect())
    }

    /// Runs tables `table_ids` of `round` and returns their results as JSON strings (same
    /// order as `table_ids`). Raises `Cancelled` if `cancel()` is called meanwhile.
    ///
    /// Tables seating only Rust bots run on a thread pool (`threads`, default from the
    /// config) with the GIL released. With `threads=1` and Python bots present (e.g. in a
    /// worker process) tables run on the calling thread with the interpreter attached, so
    /// Python bot calls re-enter cheaply; `on_progress(hands_done)` is then called every
    /// `every` hands.
    #[pyo3(signature = (round, table_ids, threads=None, on_progress=None, every=256))]
    fn run_tables(
        &self,
        py: Python<'_>,
        round: usize,
        table_ids: Vec<usize>,
        threads: Option<usize>,
        on_progress: Option<Py<PyAny>>,
        every: u64,
    ) -> PyResult<Vec<String>> {
        if self.progress.is_cancelled() {
            return Err(Cancelled::new_err("run cancelled"));
        }
        let schedule = self.state.schedule(round).map_err(err)?;
        if let Some(&t) = table_ids.iter().find(|&&t| t >= schedule.tables.len()) {
            return Err(err(format!("no table {t} in round {round}")));
        }
        let threads = threads.or(self.spec.threads);
        let spec = &self.spec;
        let factories = &self.factories;
        let progress = &self.progress;
        let replay_plan = &self.replay_plan;
        let has_python = table_ids
            .iter()
            .any(|&t| schedule.tables[t].iter().any(|&b| self.is_python[b]));
        let every = every.max(1);
        let callback = on_progress.map(|cb| {
            move || {
                let n = progress.hands_done.load(Ordering::Relaxed);
                if n.is_multiple_of(every) {
                    Python::attach(|py| {
                        if let Err(e) = cb.call1(py, (n,)) {
                            e.print(py);
                        }
                    });
                }
            }
        });
        let run_one = |t: usize, on_hand: Option<&(dyn Fn() + Sync)>| {
            let hooks = TableHooks {
                hands_done: Some(&progress.hands_done),
                cancel: Some(&progress.cancel),
                on_hand,
                record: (!replay_plan.is_empty()).then(|| replay_plan.record_for(round, t)),
            };
            run_table_with_hooks(
                &spec.format,
                &schedule.tables[t],
                factories,
                &spec.table,
                table_seed(spec.seed, round, t),
                hooks,
            )
            .map(|r| serde_json::to_string(&r).unwrap())
            .map_err(|e| table_error(round, t, e))
        };
        if threads == Some(1) && has_python {
            // Calling thread, interpreter attached: Python bots re-enter cheaply instead of
            // creating a fresh thread state on a pool thread for every call. Other Python
            // threads still get the GIL whenever bot code runs, so cancel() stays reachable.
            let cb = callback.as_ref().map(|f| f as &(dyn Fn() + Sync));
            return table_ids
                .iter()
                .map(|&t| run_one(t, cb))
                .collect::<Result<Vec<_>, _>>()
                .map_err(run_err);
        }
        // Rust-only (or multi-threaded) tables: release the GIL for the whole run so other
        // Python threads (a GUI, a stop button) stay responsive.
        let pool = build_pool(threads).map_err(err)?;
        let res: Result<Vec<String>, RunError> = py
            .detach(|| pool.install(|| table_ids.par_iter().map(|&t| run_one(t, None)).collect()));
        drop(pool);
        res.map_err(run_err)
    }

    /// Merges a round: `results` are JSON TableResults in table-id order.
    fn apply_round(&mut self, round: usize, results: Vec<String>) -> PyResult<()> {
        if round != self.state.rounds_applied() {
            return Err(err(format!(
                "expected round {}, got {round}",
                self.state.rounds_applied()
            )));
        }
        let schedule = self.state.schedule(round).map_err(err)?;
        if results.len() != schedule.tables.len() {
            return Err(err(format!(
                "round {round} has {} tables, got {} results",
                schedule.tables.len(),
                results.len()
            )));
        }
        let parsed: Vec<TableResult> = results
            .iter()
            .map(|s| serde_json::from_str(s).map_err(err))
            .collect::<PyResult<_>>()?;
        for (t, r) in parsed.iter().enumerate() {
            let bots: Vec<usize> = r.seats.iter().map(|s| s.bot).collect();
            if bots != schedule.tables[t] {
                return Err(err(format!("result {t} does not match the schedule")));
            }
        }
        self.state.apply_round(&parsed);
        self.rounds.push(RoundRecord {
            round,
            schedule,
            results: parsed,
        });
        Ok(())
    }

    /// Leaderboard rows (best first) as dicts.
    fn rows(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let s = serde_json::to_string(&self.state.rows()).unwrap();
        Ok(py.import("json")?.call_method1("loads", (s,))?.unbind())
    }

    /// The leaderboard as a text table.
    fn render(&self) -> String {
        render(&self.state.rows())
    }

    /// Writes `results/<run_id>/` and updates the persistent leaderboard. Returns the
    /// results directory.
    #[pyo3(signature = (results_dir=None, save_leaderboard=true))]
    fn finish(&mut self, results_dir: Option<PathBuf>, save_leaderboard: bool) -> PyResult<String> {
        let rows = self.state.rows();
        let run_id = self.run_id.clone();
        let base = results_dir.unwrap_or_else(|| PathBuf::from(&self.cfg.run.results_dir));
        let dir = write_results_dir(
            &base,
            &run_id,
            &self.cfg_text,
            self.state.keys(),
            &self.rounds,
            &rows,
        )
        .map_err(err)?;
        if save_leaderboard {
            self.lb
                .update(&rows, self.spec.table.reference_bb(), &run_id);
            self.lb.save(&self.lb_path).map_err(err)?;
        }
        Ok(dir.display().to_string())
    }
}

// ---------------------------------------------------------------------------- helpers

fn to_cards(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Card>> {
    if let Ok(s) = obj.extract::<String>() {
        return engine::card::parse_cards(&s).map_err(err);
    }
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        let item = item?;
        if let Ok(i) = item.extract::<u8>() {
            if i >= 52 {
                return Err(err(format!("card index {i} out of range")));
            }
            out.push(Card(i));
        } else {
            let s: String = item.extract()?;
            out.push(s.parse().map_err(err)?);
        }
    }
    Ok(out)
}

/// Monte Carlo equity of `hole` with `board` against `n_opponents` random hands.
/// Cards may be ints, strings ("As") or a single string ("AsKd"). Deterministic for a
/// given `seed`. Releases the GIL while computing.
#[pyfunction]
#[pyo3(signature = (hole, board, n_opponents, n_samples=1000, seed=0, variant="nlhe"))]
fn equity(
    py: Python<'_>,
    hole: &Bound<'_, PyAny>,
    board: &Bound<'_, PyAny>,
    n_opponents: usize,
    n_samples: u32,
    seed: u64,
    variant: &str,
) -> PyResult<f64> {
    let hole = to_cards(hole)?;
    let board = to_cards(board)?;
    let rules = engine::rules::by_name(variant)
        .ok_or_else(|| err(format!("unknown variant {variant:?}")))?;
    let v = rules.variant();
    let needed = rules.default_format(2).hole_cards as usize;
    if hole.len() < needed || board.len() > 5 {
        return Err(err(format!(
            "need {needed} hole cards for {} and at most 5 board cards",
            rules.name()
        )));
    }
    let mut seen = engine::CardMask::EMPTY;
    for c in hole.iter().chain(&board) {
        if seen.contains(*c) {
            return Err(err(format!("duplicate card {c}")));
        }
        seen.insert(*c);
    }
    if 52 - (seen.len() as usize) < (5 - board.len()) + n_opponents * hole.len() {
        return Err(err("not enough cards for that many opponents"));
    }
    Ok(py.detach(|| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        engine::equity::equity(
            Evaluator::get(),
            v,
            &hole,
            &board,
            n_opponents,
            n_samples,
            &mut rng,
        )
    }))
}

/// "As" for card index 51, etc.
#[pyfunction]
fn card_to_str(card: u8) -> PyResult<String> {
    if card >= 52 {
        return Err(err("card index out of range"));
    }
    Ok(Card(card).to_string())
}

/// Parses "AsKd" / "As Kd" into card indices.
#[pyfunction]
fn parse_cards(s: &str) -> PyResult<Vec<u32>> {
    // u32 (not u8) so PyO3 returns a list of ints rather than `bytes`.
    Ok(engine::card::parse_cards(s)
        .map_err(err)?
        .iter()
        .map(|c| c.0 as u32)
        .collect())
}

/// Strength (higher is better) of 5–7 Hold'em cards.
#[pyfunction]
fn evaluate(cards: &Bound<'_, PyAny>) -> PyResult<u16> {
    let c = to_cards(cards)?;
    if !(5..=7).contains(&c.len()) {
        return Err(err("need 5 to 7 cards"));
    }
    Ok(Evaluator::get().eval(&c))
}

/// Category name of a strength, e.g. "Flush".
#[pyfunction]
fn hand_category(strength: u16) -> PyResult<String> {
    if strength >> 12 == 0 || strength >> 12 > 9 {
        return Err(err("invalid strength"));
    }
    Ok(format!("{:?}", category(strength)))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAction>()?;
    m.add_class::<PyLegalActions>()?;
    m.add_class::<PyObservation>()?;
    m.add_class::<PyRun>()?;
    m.add("Cancelled", m.py().get_type::<Cancelled>())?;
    m.add_function(wrap_pyfunction!(equity, m)?)?;
    m.add_function(wrap_pyfunction!(card_to_str, m)?)?;
    m.add_function(wrap_pyfunction!(parse_cards, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(hand_category, m)?)?;
    solver::register(m)?;
    m.add(
        "STREETS",
        PyTuple::new(m.py(), ["preflop", "flop", "turn", "river"])?,
    )?;
    Ok(())
}
