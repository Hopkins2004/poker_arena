"""Poker Arena desktop window (PySide6).

Open it by double-clicking ``Poker Arena.app`` at the repository root, or with
``python -c "from pokerarena.gui import main; main()"``.

Set up a run (game, tables, rating, and the players: Rust bots, the Python ports in
:mod:`pokerarena.baselines`, or any ``module.ClassName``), press **Run**, watch the progress
bar, and read the leaderboard at the end. **Stop** is a hard stop. In-process Rust tables
stop before their next hand, Python worker processes are killed (even a bot stuck in an
infinite loop), and nothing is written to disk. See :mod:`pokerarena.job`.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tomllib
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ._native import Run
from .job import ArenaJob

REPO_ROOT = Path(__file__).resolve().parents[2]

# (label, kind) shown in each player's "Bot" menu. CUSTOM enables the class column.
CUSTOM = "Custom Python class…"
BOT_CHOICES = [
    ("Tight-aggressive (Rust)", "tag"),
    ("Random (Rust)", "random"),
    ("Calling station (Rust)", "calling_station"),
    ("Tight-aggressive (Python)", "python:pokerarena.baselines.TightAggressive"),
    ("Random (Python)", "python:pokerarena.baselines.RandomLegal"),
    ("Calling station (Python)", "python:pokerarena.baselines.CallingStation"),
    (CUSTOM, None),
]
KIND_TO_LABEL = {k: l for l, k in BOT_CHOICES if k}
DEFAULT_PARAMS = {
    "tag": {"samples": 300, "tightness": 0.0, "call_margin": 0.05},
    "random": {"fold": 1.0, "call": 2.0, "raise": 1.0, "raise_sizes": [0.5, 1.0]},
    "calling_station": {},
}
DEFAULT_PARAMS["python:pokerarena.baselines.TightAggressive"] = DEFAULT_PARAMS["tag"]
DEFAULT_PARAMS["python:pokerarena.baselines.RandomLegal"] = DEFAULT_PARAMS["random"]
DEFAULT_PARAMS["python:pokerarena.baselines.CallingStation"] = {}

PY = "python:pokerarena.baselines."
_TAG_LOOSE = {"samples": 200, "tightness": -3.0, "call_margin": 0.0}
_TAG = {"samples": 300}
_TAG_TIGHT = {"samples": 300, "tightness": 2.0}
PRESETS = {
    "Rust demo lineup (7 bots)": None,  # loaded from configs/6max_nlh.toml
    "Python champions (3 bots)": [
        ("py_tag_loose", PY + "TightAggressive", _TAG_LOOSE),
        ("py_tag", PY + "TightAggressive", _TAG),
        ("py_tag_tight", PY + "TightAggressive", _TAG_TIGHT),
    ],
    "Rust vs Python champions (6 bots)": [
        ("tag_loose", "tag", _TAG_LOOSE),
        ("tag", "tag", _TAG),
        ("tag_tight", "tag", _TAG_TIGHT),
        ("py_tag_loose", PY + "TightAggressive", _TAG_LOOSE),
        ("py_tag", PY + "TightAggressive", _TAG),
        ("py_tag_tight", PY + "TightAggressive", _TAG_TIGHT),
    ],
    "Python trio (3 bots)": [
        ("py_tag_loose", PY + "TightAggressive", _TAG_LOOSE),
        ("py_random_passive", PY + "RandomLegal",
         {"fold": 2.0, "call": 3.0, "raise": 0.2, "raise_sizes": [0.5]}),
        ("py_station", PY + "CallingStation", {}),
    ],
}

FALLBACK_CONFIG = {
    "run": {"rounds": 5, "seed": 2026, "results_dir": "results", "leaderboard": "leaderboard.json"},
    "format": {"variant": "nlhe", "max_seats": 6},
    "table": {"hands_per_table": 1000, "starting_stack_bb": 100, "mode": "cash",
              "blinds": {"sb": 1, "bb": 2, "ante": 0}, "stack_policy": "reset_each_hand",
              "duplicate_mode": False, "record_timing": True},
    "matchmaking": {"min_appearances": 4, "allow_duplicate_bot_at_table": True, "min_tables": 0,
                    "balance_pairs": True},
    "rating": {"k": 16.0, "score_mode": "binary"},
    "bots": [{"name": "tag", "kind": "tag", "params": {"samples": 300}},
             {"name": "random", "kind": "random", "params": {}},
             {"name": "station", "kind": "calling_station", "params": {}}],
}


def load_default_config() -> dict:
    path = REPO_ROOT / "configs" / "6max_nlh.toml"
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except OSError:
        return copy.deepcopy(FALLBACK_CONFIG)


def blind_levels(sb: int, bb: int, ante: int, hands: int) -> list:
    return [{"sb": sb * m, "bb": bb * m, "ante": ante * m, "hands": hands}
            for m in (1, 2, 3, 5, 10, 20, 40, 80, 150, 300, 600)]


def _spin(lo, hi, value=0, special=None, suffix=""):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    s.setGroupSeparatorShown(True)
    if special:
        s.setSpecialValueText(special)
    if suffix:
        s.setSuffix(suffix)
    s.setMinimumWidth(110)
    return s


def _combo(items):
    c = QComboBox()
    for label, data in items:
        c.addItem(label, data)
    return c


def _set_combo(c: QComboBox, data) -> None:
    i = c.findData(data)
    if i >= 0:
        c.setCurrentIndex(i)


def _fmt_time(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 60}:{s % 60:02d}"


def _fmt_rate(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{x / 1e3:.1f}k"
    return f"{x:.0f}"


class PlayersTable(QTableWidget):
    """Editable list of bots: name, bot type, custom Python class, JSON params."""

    COLS = ["Name", "Bot", "Python class (custom)", "Params (JSON)", ""]

    def __init__(self, on_change):
        super().__init__(0, len(self.COLS))
        self.on_change = on_change
        self.setHorizontalHeaderLabels(self.COLS)
        h = self.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.setColumnWidth(0, 150)
        self.setColumnWidth(2, 210)
        self.verticalHeader().setVisible(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.itemChanged.connect(lambda *_: self.on_change())

    def add_bot(self, name: str, kind: str, params: dict, version=None) -> None:
        r = self.rowCount()
        self.blockSignals(True)
        self.insertRow(r)
        name_item = QTableWidgetItem(name)
        name_item.setData(Qt.ItemDataRole.UserRole, version)
        self.setItem(r, 0, name_item)
        combo = _combo([(label, kind_ or "") for label, kind_ in BOT_CHOICES])
        custom = QLineEdit()
        if kind in KIND_TO_LABEL:
            _set_combo(combo, kind)
            custom.setEnabled(False)
        else:
            _set_combo(combo, "")
            custom.setText(kind.removeprefix("python:"))
            custom.setPlaceholderText("module.ClassName")
        self.setCellWidget(r, 1, combo)
        self.setCellWidget(r, 2, custom)
        self.setItem(r, 3, QTableWidgetItem(json.dumps(params or {})))
        remove = QPushButton("✕")
        remove.setToolTip("Remove this bot")
        remove.setFixedWidth(34)
        self.setCellWidget(r, 4, remove)
        self.blockSignals(False)

        def kind_changed(_i, combo=combo, custom=custom):
            k = combo.currentData()
            custom.setEnabled(not k)
            custom.setPlaceholderText("" if k else "module.ClassName")
            row = self.indexAt(combo.pos()).row()
            if k and row >= 0:
                self.item(row, 3).setText(json.dumps(DEFAULT_PARAMS.get(k, {})))
            self.on_change()

        combo.currentIndexChanged.connect(kind_changed)
        custom.textChanged.connect(lambda *_: self.on_change())
        remove.clicked.connect(lambda *_, b=remove: self._remove(b))
        self.on_change()

    def _remove(self, button) -> None:
        row = self.indexAt(button.pos()).row()
        if row >= 0:
            self.removeRow(row)
            self.on_change()

    def set_bots(self, bots) -> None:
        self.setRowCount(0)
        for b in bots:
            self.add_bot(b["name"], b["kind"], b.get("params") or {}, b.get("version"))

    def bots(self) -> list:
        """Reads the table; raises ValueError with a readable message on bad input."""
        out = []
        for r in range(self.rowCount()):
            name = (self.item(r, 0).text() if self.item(r, 0) else "").strip()
            if not name:
                raise ValueError(f"player {r + 1} needs a name")
            kind = self.cellWidget(r, 1).currentData()
            if not kind:
                cls = self.cellWidget(r, 2).text().strip()
                if "." not in cls:
                    raise ValueError(f"{name}: enter a Python class as module.ClassName")
                kind = "python:" + cls
            text = self.item(r, 3).text().strip() if self.item(r, 3) else ""
            try:
                params = json.loads(text) if text else {}
            except json.JSONDecodeError as e:
                raise ValueError(f"{name}: params are not valid JSON ({e.msg})") from None
            if not isinstance(params, dict):
                raise ValueError(f"{name}: params must be a JSON object, e.g. {{}}")
            bot = {"name": name, "kind": kind, "params": params}
            version = self.item(r, 0).data(Qt.ItemDataRole.UserRole)
            if version:
                bot["version"] = str(version)
            out.append(bot)
        return out


class ArenaWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Poker Arena")
        self.default_cfg = load_default_config()
        self.base_cfg = copy.deepcopy(self.default_cfg)
        self.job: ArenaJob | None = None
        self._kinds_by_key: dict[str, str] = {}

        self.plan_timer = QTimer(self, singleShot=True, interval=250)
        self.plan_timer.timeout.connect(self.update_plan)
        self.poll_timer = QTimer(self, interval=100)
        self.poll_timer.timeout.connect(self.poll)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- settings
        settings = QHBoxLayout()
        self.variant = _combo([("No-Limit Hold'em", "nlhe"), ("Pot-Limit Omaha", "plo"),
                               ("Fixed-Limit Hold'em", "flhe")])
        self.max_seats = _spin(2, 10, 6)
        self.mode = _combo([("Cash", "cash"), ("Tournament", "tournament")])
        self.stack = _spin(1, 100_000, 100, suffix=" bb")
        self.sb = _spin(0, 1_000_000, 1)
        self.bb = _spin(1, 1_000_000, 2)
        self.ante = _spin(0, 1_000_000, 0)
        self.stack_policy = _combo([("Reset each hand", "reset_each_hand"),
                                    ("Rebuy when busted", "rebuy_when_busted"),
                                    ("Top up to start", "top_up_to_start")])
        self.duplicate = QCheckBox()
        self.hands_per_level = _spin(1, 1_000_000, 40)
        game = QFormLayout()
        game.addRow("Variant", self.variant)
        game.addRow("Max seats", self.max_seats)
        game.addRow("Mode", self.mode)
        game.addRow("Starting stack", self.stack)
        blinds = QHBoxLayout()
        blinds.addWidget(self.sb)
        blinds.addWidget(QLabel("/"))
        blinds.addWidget(self.bb)
        game.addRow("Small / big blind", blinds)
        game.addRow("Ante", self.ante)
        game.addRow("Stack policy", self.stack_policy)
        game.addRow("Duplicate mode", self.duplicate)
        game.addRow("Hands per blind level", self.hands_per_level)
        self.game_form = game
        settings.addWidget(self._group("Game", game))

        self.hands = _spin(1, 100_000_000, 1000)
        self.rounds = _spin(1, 100_000, 5)
        self.seats_per_table = _spin(1, 10, 1, special="= max seats")
        self.min_tables = _spin(0, 100_000, 0, special="auto")
        self.min_appearances = _spin(1, 1000, 4)
        self.allow_dup = QCheckBox()
        self.balance = QCheckBox()
        tables = QFormLayout()
        self.hands_label = QLabel("Hands per table")
        tables.addRow(self.hands_label, self.hands)
        tables.addRow("Rounds", self.rounds)
        tables.addRow("Players per table", self.seats_per_table)
        tables.addRow("Min tables per round", self.min_tables)
        tables.addRow("Min appearances per bot", self.min_appearances)
        tables.addRow("Same bot twice at a table", self.allow_dup)
        tables.addRow("Balance pairings", self.balance)
        settings.addWidget(self._group("Tables", tables))

        self.seed = _spin(0, 2_000_000_000, 0)
        self.threads = _spin(0, 1024, 0, special="all cores")
        self.processes = _spin(0, 1024, 0, special="all cores")
        self.score_mode = _combo([("Binary", "binary"), ("Margin (tanh)", "margin")])
        self.k = QDoubleSpinBox()
        self.k.setRange(0.1, 1000)
        self.k.setValue(16)
        self.timing = QCheckBox()
        self.save = QCheckBox()
        self.save.setToolTip(f"Writes results/<run_id>/ and updates leaderboard.json in {os.getcwd()}")
        runf = QFormLayout()
        runf.addRow("Seed", self.seed)
        runf.addRow("Rust threads", self.threads)
        runf.addRow("Python processes", self.processes)
        runf.addRow("Elo score", self.score_mode)
        runf.addRow("Elo K-factor", self.k)
        runf.addRow("Record decision time", self.timing)
        runf.addRow("Save results && leaderboard", self.save)
        settings.addWidget(self._group("Run", runf))
        root.addLayout(settings)

        # --- players
        players_box = QGroupBox("Players")
        pv = QVBoxLayout(players_box)
        bar = QHBoxLayout()
        self.preset = QComboBox()
        self.preset.addItems(list(PRESETS))
        load_preset = QPushButton("Load preset")
        load_preset.clicked.connect(self.load_preset)
        add = QPushButton("+ Add bot")
        add.clicked.connect(self.add_bot)
        self.nbots = QLabel()
        bar.addWidget(QLabel("Preset:"))
        bar.addWidget(self.preset)
        bar.addWidget(load_preset)
        bar.addStretch(1)
        bar.addWidget(self.nbots)
        bar.addWidget(add)
        pv.addLayout(bar)
        self.players = PlayersTable(self.changed)
        pv.addWidget(self.players)

        # --- run controls + leaderboard
        run_box = QWidget()
        rv = QVBoxLayout(run_box)
        rv.setContentsMargins(0, 0, 0, 0)
        ctl = QHBoxLayout()
        self.plan = QLabel()
        self.plan.setTextFormat(Qt.TextFormat.RichText)
        self.plan.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setDefault(True)
        self.run_btn.setMinimumWidth(110)
        self.run_btn.clicked.connect(self.start_run)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setMinimumWidth(110)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Hard stop: halts all tables, kills Python workers, saves nothing")
        self.stop_btn.setStyleSheet(
            "QPushButton:enabled { background: #c62828; color: white; font-weight: 600;"
            " border-radius: 5px; padding: 5px 12px; }")
        self.stop_btn.clicked.connect(self.stop_run)
        ctl.addWidget(self.plan, 1)
        ctl.addWidget(self.run_btn)
        ctl.addWidget(self.stop_btn)
        rv.addLayout(ctl)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        rv.addWidget(self.progress)
        self.stats = QLabel(" ")
        self.stats.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rv.addWidget(self.stats)
        self.board = QTableWidget(0, 0)
        self.board.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.board.verticalHeader().setVisible(False)
        self.board.setAlternatingRowColors(True)
        rv.addWidget(self._group("Leaderboard", self._wrap(self.board)))
        self.saved = QLabel(" ")
        self.saved.setWordWrap(True)
        self.saved.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rv.addWidget(self.saved)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(players_box)
        split.addWidget(run_box)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([330, 420])
        root.addWidget(split, 1)

        for w in (self.variant, self.mode, self.stack_policy, self.score_mode):
            w.currentIndexChanged.connect(self.changed)
        for w in (self.max_seats, self.stack, self.sb, self.bb, self.ante, self.hands_per_level,
                  self.hands, self.rounds, self.seats_per_table, self.min_tables,
                  self.min_appearances, self.seed, self.threads, self.processes):
            w.valueChanged.connect(self.changed)
        self.k.valueChanged.connect(self.changed)
        for w in (self.duplicate, self.allow_dup, self.balance, self.timing):
            w.toggled.connect(self.changed)

        self.load_config(self.default_cfg)
        self.resize(1180, 860)

    # ----------------------------------------------------------------------- helpers

    @staticmethod
    def _group(title, layout_or_widget):
        g = QGroupBox(title)
        if isinstance(layout_or_widget, QWidget):
            v = QVBoxLayout(g)
            v.addWidget(layout_or_widget)
        else:
            g.setLayout(layout_or_widget)
        return g

    @staticmethod
    def _wrap(widget):
        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(0, 0, 0, 0)
        g.addWidget(widget)
        return w

    def load_config(self, cfg: dict) -> None:
        self.base_cfg = copy.deepcopy(cfg)
        t, m, r, f, run = (cfg.get(k, {}) for k in ("table", "matchmaking", "rating", "format", "run"))
        widgets = [self.variant, self.max_seats, self.mode, self.stack, self.sb, self.bb]
        for w in widgets:
            w.blockSignals(True)
        _set_combo(self.variant, f.get("variant", "nlhe"))
        self.max_seats.setValue(int(f.get("max_seats", 6)))
        _set_combo(self.mode, t.get("mode", "cash"))
        self.stack.setValue(int(t.get("starting_stack_bb", 100)))
        bl = t.get("blinds") or (t.get("levels") or [{"sb": 1, "bb": 2, "ante": 0}])[0]
        self.sb.setValue(int(bl.get("sb", 1)))
        self.bb.setValue(int(bl.get("bb", 2)))
        self.ante.setValue(int(bl.get("ante", 0)))
        _set_combo(self.stack_policy, t.get("stack_policy", "reset_each_hand"))
        self.duplicate.setChecked(bool(t.get("duplicate_mode", False)))
        self.hands.setValue(int(t.get("hands_per_table", 1000)))
        self.rounds.setValue(int(run.get("rounds", 5)))
        self.seats_per_table.setValue(int(m.get("seats_per_table") or 1))
        self.min_tables.setValue(int(m.get("min_tables", 0)))
        self.min_appearances.setValue(int(m.get("min_appearances", 4)))
        self.allow_dup.setChecked(bool(m.get("allow_duplicate_bot_at_table", True)))
        self.balance.setChecked(bool(m.get("balance_pairs", False)))
        self.seed.setValue(int(run.get("seed", 0)))
        self.threads.setValue(int(run.get("threads") or 0))
        _set_combo(self.score_mode, r.get("score_mode", "binary"))
        self.k.setValue(float(r.get("k", 16)))
        self.timing.setChecked(bool(t.get("record_timing", False)))
        for w in widgets:
            w.blockSignals(False)
        self.players.set_bots(cfg.get("bots", []))
        self.changed()

    def load_preset(self) -> None:
        bots = PRESETS[self.preset.currentText()]
        if bots is None:
            self.players.set_bots(self.default_cfg.get("bots", []))
        else:
            self.players.set_bots([{"name": n, "kind": k, "params": dict(p)} for n, k, p in bots])

    def add_bot(self) -> None:
        names = {b for b in self._names()}
        i = self.players.rowCount() + 1
        while f"bot{i}" in names:
            i += 1
        self.players.add_bot(f"bot{i}", "random", DEFAULT_PARAMS["random"])

    def _names(self):
        for r in range(self.players.rowCount()):
            it = self.players.item(r, 0)
            if it:
                yield it.text().strip()

    def build_config(self) -> dict:
        cfg = copy.deepcopy(self.base_cfg)
        mode = self.mode.currentData()
        sb, bb, ante = self.sb.value(), self.bb.value(), self.ante.value()
        cfg["format"] = {"variant": self.variant.currentData(), "max_seats": self.max_seats.value()}
        t = cfg.setdefault("table", {})
        t.update({
            "hands_per_table": self.hands.value(),
            "starting_stack_bb": self.stack.value(),
            "mode": mode,
            "stack_policy": self.stack_policy.currentData(),
            "duplicate_mode": mode == "cash" and self.duplicate.isChecked(),
            "record_timing": self.timing.isChecked(),
        })
        if mode == "cash":
            t["blinds"] = {"sb": sb, "bb": bb, "ante": ante}
            t["levels"] = []
        else:
            t.pop("blinds", None)
            t["levels"] = blind_levels(sb, bb, ante, self.hands_per_level.value())
        m = cfg.setdefault("matchmaking", {})
        spt = self.seats_per_table.value()
        m.update({
            "seats_per_table": None if spt < 2 else spt,
            "min_tables": self.min_tables.value(),
            "min_appearances": self.min_appearances.value(),
            "allow_duplicate_bot_at_table": self.allow_dup.isChecked(),
            "balance_pairs": self.balance.isChecked(),
        })
        if m["seats_per_table"] is None:
            m.pop("seats_per_table")
        r = cfg.setdefault("rating", {})
        r.update({"score_mode": self.score_mode.currentData(), "k": self.k.value()})
        run = cfg.setdefault("run", {})
        run.update({"rounds": self.rounds.value(), "seed": self.seed.value()})
        if self.threads.value():
            run["threads"] = self.threads.value()
        else:
            run.pop("threads", None)
        cfg["bots"] = self.players.bots()
        self._kinds_by_key = {b["name"]: b["kind"] for b in cfg["bots"]}
        return cfg

    # ----------------------------------------------------------------------- reactions

    def changed(self, *_):
        tour = self.mode.currentData() == "tournament"
        for w, show in ((self.stack_policy, not tour), (self.duplicate, not tour),
                        (self.hands_per_level, tour)):
            w.setVisible(show)
            label = self.game_form.labelForField(w)
            if label:
                label.setVisible(show)
        self.hands_label.setText("Max hands per table" if tour else "Hands per table")
        self.nbots.setText(f"{self.players.rowCount()} bots")
        if self.job is None or self.job.state not in ("running", "stopping"):
            self.plan_timer.start()

    def update_plan(self) -> None:
        try:
            cfg = self.build_config()
            r = Run(json.dumps(cfg), load_leaderboard=False)
            per_round = len(r.schedule(0))
            seats = len(r.schedule(0)[0]) if per_round else 0
            total = per_round * r.n_rounds
            hands = total * r.hands_per_table
            n_py = sum(r.python_bots)
            self.plan.setText(
                f"<b>{len(cfg['bots'])}</b> bots ({n_py} Python) · <b>{seats}</b> players/table · "
                f"<b>{per_round}</b> tables/round · <b>{total:,}</b> tables · "
                f"{'up to ' if r.is_tournament else ''}<b>{hands:,}</b> hands")
            self.run_btn.setEnabled(self.job is None or self.job.state not in ("running", "stopping"))
        except Exception as e:  # noqa: BLE001 - shown to the user
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            self.plan.setText(f"<span style='color:#d32f2f'>{msg}</span>")
            self.run_btn.setEnabled(False)

    def set_editing(self, enabled: bool) -> None:
        for box in self.findChildren(QGroupBox):
            if box.title() != "Leaderboard":
                box.setEnabled(enabled)

    def start_run(self) -> None:
        try:
            cfg = self.build_config()
            job = ArenaJob(cfg, processes=self.processes.value() or None,
                           save=self.save.isChecked())
        except Exception as e:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Cannot start", str(e))
            return
        self.job = job
        self.board.setRowCount(0)
        self.saved.setText(" ")
        self.progress.setValue(0)
        self.set_editing(False)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stop_btn.setText("■  Stop")
        job.start()
        self.poll_timer.start()
        self.poll()

    def stop_run(self) -> None:
        if self.job is not None:
            self.job.stop()
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stopping…")

    def poll(self) -> None:
        job = self.job
        if job is None:
            return
        p = job.progress()
        frac = p.fraction
        self.progress.setValue(int(frac * 1000))
        self.progress.setFormat(f"{frac * 100:.1f}%")
        eta = ""
        if p.state == "running" and 0 < frac < 1:
            eta = f" · ETA {_fmt_time(p.elapsed * (1 - frac) / frac)}"
        rnd = min(p.rounds_done + (1 if p.state == "running" else 0), p.rounds_total)
        self.stats.setText(
            f"{p.state.capitalize()} · round {rnd}/{p.rounds_total} · "
            f"{p.tables_done:,}/{p.tables_total:,} tables · {p.hands_done:,} hands · "
            f"{_fmt_rate(p.hands_per_sec)} hands/s · {_fmt_time(p.elapsed)} elapsed{eta}")
        if p.state in ("done", "stopped", "error") and job.wait(0):
            self.poll_timer.stop()
            self.finish(job)

    def finish(self, job: ArenaJob) -> None:
        self.set_editing(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("■  Stop")
        self.job = None
        if job.state == "done" and job.result is not None:
            self.show_result(job.result)
        elif job.state == "stopped":
            self.progress.setFormat("Stopped")
            self.saved.setText("Run stopped. Nothing was saved.")
        else:
            self.progress.setFormat("Error")
            msg = f"{type(job.error).__name__}: {job.error}"
            self.saved.setText(msg)
            QMessageBox.critical(self, "Run failed", msg)
        self.update_plan()

    def show_result(self, res) -> None:
        tour = res.tournament
        cols = ["#", "Bot", "Type", "Elo", "ΔElo", "Hands", "bb/100", "±95% CI", "VPIP", "PFR",
                "Showdown", "Illegal", "µs/decision"] + (["Avg place"] if tour else [])
        self.board.setColumnCount(len(cols))
        self.board.setHorizontalHeaderLabels(cols)
        self.board.setRowCount(len(res.rows))
        green, red = QBrush(QColor("#2e7d32")), QBrush(QColor("#c62828"))
        bold = QFont()
        bold.setBold(True)
        for i, x in enumerate(res.rows):
            kind = self._kinds_by_key.get(x["key"].split("@")[0], "")
            ci = x.get("ci95")
            vals = [
                str(i + 1), x["key"], "Python" if kind.startswith("python:") else "Rust",
                f"{x['rating']:.1f}", f"{x['rating_change']:+.1f}", f"{x['hands']:,}",
                f"{x['bb_per_100']:+.1f}", "–" if ci is None else f"±{ci:.1f}",
                f"{100 * x['vpip']:.1f}%", f"{100 * x['pfr']:.1f}%", f"{100 * x['showdown']:.1f}%",
                str(x["illegal"]), f"{x['mean_decision_us']:.2f}",
            ] + ([f"{x['avg_place']:.2f}"] if tour else [])
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j >= 3:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if j in (4, 6):
                    val = x["rating_change"] if j == 4 else x["bb_per_100"]
                    if val:
                        it.setForeground(green if val > 0 else red)
                if i == 0:
                    it.setFont(bold)
                self.board.setItem(i, j, it)
        self.board.resizeColumnsToContents()
        self.board.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.progress.setFormat("Done")
        line = (f"{res.hands:,} hands on {res.tables:,} tables in {res.elapsed:.2f} s "
                f"({_fmt_rate(res.hands / max(res.elapsed, 1e-9))} hands/s).")
        if res.results_dir:
            line += f" Saved to {res.results_dir}; leaderboard updated."
        self.saved.setText(line)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        job = self.job
        if job is not None and job.state in ("running", "stopping"):
            job.stop()
            self.stats.setText("Stopping before closing…")
            QApplication.processEvents()
            job.wait(15)
        self.poll_timer.stop()
        event.accept()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Poker Arena")
    win = ArenaWindow()
    win.show()
    win.raise_()
    win.activateWindow()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
