"""Poker Arena desktop window (PySide6).

Open it by double-clicking ``Poker Arena.app`` at the repository root, or with
``python -c "from pokerarena.gui import main; main()"``.

Set up a run (game, tables, rating, and the players: bots from the ``bots/NLH`` folder, whose
parameters and anchor Elo are hardcoded in their files),
press **Run**, watch the progress bar, and read the leaderboard at the end. With "Save
results & leaderboard" on, bots start from their saved Elo. "Save 100 random hands per bot"
writes hand replays to ``hand_replays/<run_id>/``. **Stop** is a hard stop: worker processes
are killed (even a bot stuck in an infinite loop) and nothing is written to disk. See
:mod:`pokerarena.job`.
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

from . import catalog
from ._native import Run
from .job import ArenaJob

REPO_ROOT = Path(__file__).resolve().parents[2]

REPLAY_HANDS_PER_BOT = 100

FALLBACK_CONFIG = {
    "run": {"rounds": 5, "seed": 2026, "results_dir": "results", "leaderboard": "leaderboard.json"},
    "format": {"variant": "nlhe", "max_seats": 6},
    "table": {"hands_per_table": 1000, "starting_stack_bb": 100, "mode": "cash",
              "blinds": {"sb": 1, "bb": 2, "ante": 0}, "stack_policy": "reset_each_hand",
              "duplicate_mode": False, "record_timing": True},
    "matchmaking": {"min_appearances": 4, "allow_duplicate_bot_at_table": True, "min_tables": 0,
                    "balance_pairs": True},
    "rating": {"k": 16.0, "score_mode": "binary"},
    "bots": [],  # filled with the anchored bots from bots/NLH
}


def load_default_config() -> dict:
    """Settings from configs/6max_nlh.toml (or built-in defaults); the lineup is always the
    anchored bots found in bots/NLH."""
    path = REPO_ROOT / "configs" / "6max_nlh.toml"
    try:
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        cfg = copy.deepcopy(FALLBACK_CONFIG)
    cfg["bots"] = catalog.anchored_lineup()
    return cfg


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
    """The run's players: name (editable), bot (a file in bots/NLH), anchor Elo (declared by
    the bot file) and the Elo the bot will start the run with. Bot parameters are hardcoded
    in the bot files; there is nothing else to edit here."""

    COLS = ["Name", "Bot", "Anchor Elo", "Start Elo", ""]
    NAME, BOT, ANCHOR, START, REMOVE = range(5)
    KIND_ROLE = Qt.ItemDataRole.UserRole
    VERSION_ROLE = Qt.ItemDataRole.UserRole + 1
    ANCHOR_ROLE = Qt.ItemDataRole.UserRole + 2

    def __init__(self, on_change, labels: dict):
        super().__init__(0, len(self.COLS))
        self.on_change = on_change
        self.labels = labels  # kind -> human label
        self.setHorizontalHeaderLabels(self.COLS)
        h = self.horizontalHeader()
        h.setSectionResizeMode(self.NAME, QHeaderView.ResizeMode.Interactive)
        h.setSectionResizeMode(self.BOT, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(self.ANCHOR, QHeaderView.ResizeMode.Interactive)
        h.setSectionResizeMode(self.START, QHeaderView.ResizeMode.Interactive)
        h.setSectionResizeMode(self.REMOVE, QHeaderView.ResizeMode.ResizeToContents)
        self.setColumnWidth(self.NAME, 160)
        self.setColumnWidth(self.ANCHOR, 90)
        self.setColumnWidth(self.START, 130)
        self.horizontalHeaderItem(self.ANCHOR).setToolTip(
            "Fixed Elo declared by the bot file (ANCHOR). An anchored bot's rating never "
            "changes; it pins the scale.")
        self.horizontalHeaderItem(self.START).setToolTip(
            "Elo the bot starts this run with: its anchor, else its saved rating (when "
            "'Save results & leaderboard' is on and it has one), else the initial rating.")
        self.verticalHeader().setVisible(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.itemChanged.connect(lambda *_: self.on_change())

    @staticmethod
    def _readonly(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        return item

    def add_bot(self, name: str, kind: str, anchor=None, version=None) -> None:
        r = self.rowCount()
        self.blockSignals(True)
        self.insertRow(r)
        name_item = QTableWidgetItem(name)
        name_item.setData(self.KIND_ROLE, kind)
        name_item.setData(self.VERSION_ROLE, version)
        name_item.setData(self.ANCHOR_ROLE, None if anchor is None else float(anchor))
        self.setItem(r, self.NAME, name_item)
        bot = self._readonly(self.labels.get(kind, kind.removeprefix("python:")))
        bot.setToolTip(kind)
        self.setItem(r, self.BOT, bot)
        self.setItem(r, self.ANCHOR, self._readonly("—" if anchor is None else f"{float(anchor):g}"))
        self.setItem(r, self.START, self._readonly(""))
        remove = QPushButton("✕")
        remove.setToolTip("Remove this bot")
        remove.setFixedWidth(34)
        remove.clicked.connect(lambda *_, b=remove: self._remove(b))
        self.setCellWidget(r, self.REMOVE, remove)
        self.blockSignals(False)
        self.on_change()

    def _remove(self, button) -> None:
        row = self.indexAt(button.pos()).row()
        if row >= 0:
            self.removeRow(row)
            self.on_change()

    def set_bots(self, bots) -> None:
        self.setRowCount(0)
        for b in bots:
            self.add_bot(b["name"], b["kind"], b.get("anchor"), b.get("version"))
        self.on_change()

    def names(self) -> list:
        return [(self.item(r, self.NAME).text().strip() if self.item(r, self.NAME) else "")
                for r in range(self.rowCount())]

    def bots(self) -> list:
        """Reads the table; raises ValueError with a readable message on bad input."""
        out, seen = [], set()
        for r in range(self.rowCount()):
            item = self.item(r, self.NAME)
            name = item.text().strip()
            if not name:
                raise ValueError(f"player {r + 1} needs a name")
            if name in seen:
                raise ValueError(f"two players are named {name!r}; names must be unique")
            seen.add(name)
            bot = {"name": name, "kind": item.data(self.KIND_ROLE)}
            if item.data(self.ANCHOR_ROLE) is not None:
                bot["anchor"] = float(item.data(self.ANCHOR_ROLE))
            if item.data(self.VERSION_ROLE):
                bot["version"] = str(item.data(self.VERSION_ROLE))
            out.append(bot)
        return out

    def set_start_elos(self, values) -> None:
        """values: one (rating, source) per row, or None to clear."""
        self.blockSignals(True)
        for r in range(self.rowCount()):
            item = self.item(r, self.START)
            if item is None:
                continue
            if values is None or r >= len(values):
                item.setText("")
            else:
                rating, source = values[r]
                item.setText(f"{rating:.0f}  ({source})")
        self.blockSignals(False)


class ArenaWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Poker Arena")
        self.catalog = catalog.discover()
        self.default_cfg = load_default_config()
        self.base_cfg = copy.deepcopy(self.default_cfg)
        self.job: ArenaJob | None = None

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
        self.allow_dup = QCheckBox()
        self.balance = QCheckBox()
        tables = QFormLayout()
        self.hands_label = QLabel("Hands per table")
        tables.addRow(self.hands_label, self.hands)
        tables.addRow("Rounds", self.rounds)
        tables.addRow("Players per table", self.seats_per_table)
        tables.addRow("Min tables per round", self.min_tables)
        tables.addRow("Same bot twice at a table", self.allow_dup)
        tables.addRow("Balance pairings", self.balance)
        settings.addWidget(self._group("Tables", tables))

        self.seed = _spin(0, 2_000_000_000, 0)
        self.processes = _spin(0, 1024, 0, special="all cores")
        self.score_mode = _combo([("Binary", "binary"), ("Margin (tanh)", "margin")])
        self.k = QDoubleSpinBox()
        self.k.setRange(0.1, 1000)
        self.k.setValue(16)
        self.timing = QCheckBox()
        self.save = QCheckBox()
        self.save.setToolTip(
            f"Starts bots from their saved Elo in leaderboard.json (anchors excepted), then "
            f"writes results/<run_id>/ and updates leaderboard.json in {os.getcwd()}")
        self.replays = QCheckBox()
        self.replays.setToolTip(
            f"Saves {REPLAY_HANDS_PER_BOT} random hands per bot from the run as text hand "
            f"histories in hand_replays/<run_id>/<bot>.txt (in {os.getcwd()})")
        runf = QFormLayout()
        runf.addRow("Seed", self.seed)
        runf.addRow("Python processes", self.processes)
        runf.addRow("Elo score", self.score_mode)
        runf.addRow("Elo K-factor", self.k)
        runf.addRow("Record decision time", self.timing)
        runf.addRow("Save results && leaderboard", self.save)
        runf.addRow(f"Save {REPLAY_HANDS_PER_BOT} random hands per bot", self.replays)
        settings.addWidget(self._group("Run", runf))
        root.addLayout(settings)

        # --- players: bots come from the files in bots/NLH
        players_box = QGroupBox("Players")
        pv = QVBoxLayout(players_box)
        bar = QHBoxLayout()
        self.catalog_box = QComboBox()
        for b in self.catalog:
            anchor = "" if b.anchor is None else f"  ·  anchor {b.anchor:g}"
            self.catalog_box.addItem(f"{b.name}{anchor}  —  {b.label}", b.kind)
            self.catalog_box.setItemData(self.catalog_box.count() - 1, b.path,
                                         Qt.ItemDataRole.ToolTipRole)
        self.catalog_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        add = QPushButton("+ Add bot")
        add.setToolTip(f"Adds the selected bot from {catalog.BOTS_DIR / catalog.DEFAULT_FORMAT_DIR}")
        add.clicked.connect(self.add_bot)
        self.add_btn = add
        reset = QPushButton("Reset to anchored bots")
        reset.clicked.connect(self.reset_lineup)
        self.nbots = QLabel()
        bar.addWidget(QLabel("Bot:"))
        bar.addWidget(self.catalog_box)
        bar.addWidget(add)
        bar.addStretch(1)
        bar.addWidget(self.nbots)
        bar.addWidget(reset)
        pv.addLayout(bar)
        self.players = PlayersTable(self.changed, {b.kind: f"{b.name} — {b.label}" for b in self.catalog})
        pv.addWidget(self.players)
        folder = QLabel(f"Bots folder: {catalog.BOTS_DIR / catalog.DEFAULT_FORMAT_DIR}  "
                        "(add a .py file there to make a new bot available)")
        folder.setStyleSheet("color: gray;")
        pv.addWidget(folder)

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
                  self.seed, self.processes):
            w.valueChanged.connect(self.changed)
        self.k.valueChanged.connect(self.changed)
        for w in (self.duplicate, self.allow_dup, self.balance, self.timing, self.save,
                  self.replays):
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
        self.allow_dup.setChecked(bool(m.get("allow_duplicate_bot_at_table", True)))
        self.balance.setChecked(bool(m.get("balance_pairs", False)))
        self.seed.setValue(int(run.get("seed", 0)))
        self.replays.setChecked(int(run.get("replay_hands_per_bot") or 0) > 0)
        _set_combo(self.score_mode, r.get("score_mode", "binary"))
        self.k.setValue(float(r.get("k", 16)))
        self.timing.setChecked(bool(t.get("record_timing", False)))
        for w in widgets:
            w.blockSignals(False)
        self.players.set_bots(cfg.get("bots", []))
        self.changed()

    def reset_lineup(self) -> None:
        """Loads every anchored bot from the bots folder."""
        self.players.set_bots(catalog.anchored_lineup())

    def add_bot(self) -> None:
        kind = self.catalog_box.currentData()
        info = next((b for b in self.catalog if b.kind == kind), None)
        if info is None:
            return
        names = set(self.players.names())
        name, i = info.name, 2
        while name in names:
            name, i = f"{info.name}_{i}", i + 1
        self.players.add_bot(name, info.kind, info.anchor)

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
            "allow_duplicate_bot_at_table": self.allow_dup.isChecked(),
            "balance_pairs": self.balance.isChecked(),
        })
        if m["seats_per_table"] is None:
            m.pop("seats_per_table")
        r = cfg.setdefault("rating", {})
        r.update({"score_mode": self.score_mode.currentData(), "k": self.k.value()})
        run = cfg.setdefault("run", {})
        run.update({"rounds": self.rounds.value(), "seed": self.seed.value()})
        run["replay_hands_per_bot"] = REPLAY_HANDS_PER_BOT if self.replays.isChecked() else 0
        cfg["bots"] = self.players.bots()
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

    def saved_ratings(self) -> dict:
        """Ratings stored in the leaderboard file (empty if none or unreadable)."""
        path = Path(self.base_cfg.get("run", {}).get("leaderboard", "leaderboard.json"))
        try:
            with open(path) as f:
                return {k: v["rating"] for k, v in json.load(f).get("entries", {}).items()}
        except (OSError, ValueError, KeyError, TypeError):
            return {}

    def update_plan(self) -> None:
        try:
            cfg = self.build_config()
            save = self.save.isChecked()
            r = Run(json.dumps(cfg), load_leaderboard=save)
            per_round = len(r.schedule(0))
            seats = len(r.schedule(0)[0]) if per_round else 0
            total = per_round * r.n_rounds
            hands = total * r.hands_per_table
            saved = self.saved_ratings() if save else {}
            starts, n_saved = [], 0
            for key, rating, anchor in zip(r.keys, r.initial_ratings, r.anchors):
                if anchor is not None:
                    starts.append((rating, "anchor"))
                elif key in saved:
                    starts.append((rating, "saved"))
                    n_saved += 1
                else:
                    starts.append((rating, "initial"))
            self.players.set_start_elos(starts)
            n_anchor = sum(a is not None for a in r.anchors)
            extra = []
            if n_anchor:
                extra.append(f"{n_anchor} anchored")
            if save:
                extra.append(f"{n_saved} continue from saved Elo")
            self.plan.setText(
                f"<b>{len(cfg['bots'])}</b> bots{' (' + ', '.join(extra) + ')' if extra else ''} · "
                f"<b>{seats}</b> players/table · <b>{per_round}</b> tables/round · "
                f"<b>{total:,}</b> tables · {'up to ' if r.is_tournament else ''}<b>{hands:,}</b> hands")
            self.run_btn.setEnabled(self.job is None or self.job.state not in ("running", "stopping"))
        except Exception as e:  # noqa: BLE001 - shown to the user
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            self.plan.setText(f"<span style='color:#d32f2f'>{msg}</span>")
            self.players.set_start_elos(None)
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
        cols = ["#", "Bot", "Elo", "ΔElo", "Hands", "bb/100", "±95% CI", "VPIP", "PFR",
                "Showdown", "Illegal", "µs/decision"] + (["Avg place"] if tour else [])
        self.board.setColumnCount(len(cols))
        self.board.setHorizontalHeaderLabels(cols)
        self.board.setRowCount(len(res.rows))
        green, red = QBrush(QColor("#2e7d32")), QBrush(QColor("#c62828"))
        muted = QBrush(QColor("#7a7a7a"))
        bold = QFont()
        bold.setBold(True)
        for i, x in enumerate(res.rows):
            ci = x.get("ci95")
            anchored = bool(x.get("anchored"))
            vals = [
                str(i + 1), x["key"] + ("  (anchor)" if anchored else ""),
                f"{x['rating']:.1f}", "—" if anchored else f"{x['rating_change']:+.1f}",
                f"{x['hands']:,}",
                f"{x['bb_per_100']:+.1f}", "–" if ci is None else f"±{ci:.1f}",
                f"{100 * x['vpip']:.1f}%", f"{100 * x['pfr']:.1f}%", f"{100 * x['showdown']:.1f}%",
                str(x["illegal"]), f"{x['mean_decision_us']:.2f}",
            ] + ([f"{x['avg_place']:.2f}"] if tour else [])
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j >= 2:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if j == 3 and not anchored and x["rating_change"]:
                    it.setForeground(green if x["rating_change"] > 0 else red)
                if j == 5 and x["bb_per_100"]:
                    it.setForeground(green if x["bb_per_100"] > 0 else red)
                if anchored and j in (1, 2):
                    it.setForeground(muted)
                    it.setToolTip("Anchored: this Elo is fixed and pins the rating scale.")
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
        if res.replays_dir:
            line += f" Hand replays: {res.replays_dir}."
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
