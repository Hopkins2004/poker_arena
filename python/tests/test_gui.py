"""Offscreen tests of the desktop window: presets, a full run, the hard stop, closing."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from pokerarena.gui import ArenaWindow, PRESETS  # noqa: E402
from tests.test_job import assert_no_leak, baseline  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def pump(app, until, timeout=60.0):
    t0 = time.time()
    while not until():
        app.processEvents()
        time.sleep(0.01)
        assert time.time() - t0 < timeout, "timed out"
    app.processEvents()


def test_defaults_and_plan(app):
    w = ArenaWindow()
    w.update_plan()
    assert "7</b> bots (3 anchored)" in w.plan.text() and "35</b> tables" in w.plan.text()
    assert not hasattr(w, "min_appearances") and not hasattr(w, "threads")
    starts = {w.players.item(r, 0).text(): w.players.item(r, w.players.START).text()
              for r in range(w.players.rowCount())}
    assert starts["random"] == "850  (anchor)" and starts["station"] == "400  (anchor)"
    assert starts["tag"] == "1300  (anchor)" and starts["tag_loose"] == "1500  (initial)"
    assert w.run_btn.isEnabled() and not w.stop_btn.isEnabled()
    # A broken params cell disables Run with a readable message.
    w.players.item(0, 3).setText("{oops")
    w.update_plan()
    assert "not valid JSON" in w.plan.text() and not w.run_btn.isEnabled()
    w.close()


def test_champions_full_run_with_replays(app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # replays land in ./hand_replays
    w = ArenaWindow()
    w.preset.setCurrentText("Champions: tag variants (3 bots)")
    w.load_preset()
    assert w.players.rowCount() == 3
    w.hands.setValue(300)
    w.rounds.setValue(2)
    w.processes.setValue(2)
    w.replays.setChecked(True)
    w.start_run()
    assert w.stop_btn.isEnabled() and not w.run_btn.isEnabled()
    pump(app, lambda: w.job is None)
    assert w.progress.format() == "Done"
    assert w.board.rowCount() == 3
    cols = [w.board.horizontalHeaderItem(c).text() for c in range(w.board.columnCount())]
    assert "Type" not in cols
    ill = cols.index("Illegal")
    assert all(w.board.item(r, ill).text() == "0" for r in range(3)), "no illegal actions"
    tag_row = next(r for r in range(3) if w.board.item(r, 1).text().startswith("tag  (anchor)"))
    assert w.board.item(tag_row, 2).text() == "1300.0" and w.board.item(tag_row, 3).text() == "—"
    replay_dirs = list((tmp_path / "hand_replays").iterdir())
    assert len(replay_dirs) == 1
    assert sorted(p.name for p in replay_dirs[0].iterdir()) == ["tag.txt", "tag_loose.txt", "tag_tight.txt"]
    assert "Hand replays" in w.saved.text()
    assert w.run_btn.isEnabled()
    w.close()


def test_save_toggle_reuses_saved_elo(app, tmp_path):
    lb = tmp_path / "lb.json"
    lb.write_text('{"entries": {"tag_loose": {"rating": 1612.5, "hands": 1, "net_bb": 0, '
                  '"tables": 1, "runs": 1, "last_run": "x"}, "tag": {"rating": 1777, "hands": 1, '
                  '"net_bb": 0, "tables": 1, "runs": 1, "last_run": "x"}}}')
    w = ArenaWindow()
    w.base_cfg.setdefault("run", {})["leaderboard"] = str(lb)
    w.update_plan()
    col = w.players.START
    start = lambda name: next(w.players.item(r, col).text() for r in range(w.players.rowCount())
                              if w.players.item(r, 0).text() == name)
    assert start("tag_loose") == "1500  (initial)"  # save off: fresh ratings
    w.save.setChecked(True)
    w.update_plan()
    assert start("tag_loose") == "1612  (saved)"  # rendered without decimals
    assert start("tag") == "1300  (anchor)"  # anchors win over saved ratings
    assert "1 continue from saved Elo" in w.plan.text()
    w.close()


def test_stop_button_nukes_mixed_run(app):
    before = baseline()
    w = ArenaWindow()
    w.preset.setCurrentText("Demo lineup (7 bots)")
    w.load_preset()
    w.hands.setValue(5_000_000)
    w.processes.setValue(3)
    w.start_run()
    job = w.job
    pump(app, lambda: job.progress().hands_done > 20_000)
    t0 = time.time()
    w.stop_btn.click()
    pump(app, lambda: w.job is None, timeout=15)
    assert time.time() - t0 < 5
    assert job.state == "stopped"
    assert w.progress.format() == "Stopped"
    assert w.board.rowCount() == 0
    assert w.run_btn.isEnabled() and not w.stop_btn.isEnabled()
    w.close()
    assert_no_leak(before)


def test_closing_window_during_run_stops_it(app):
    before = baseline()
    w = ArenaWindow()
    w.preset.setCurrentText("Trio (3 bots)")
    w.load_preset()
    w.hands.setValue(5_000_000)
    w.start_run()
    job = w.job
    pump(app, lambda: job.progress().hands_done > 1000)
    w.close()
    assert job.state == "stopped" and job.wait(0)
    assert_no_leak(before)


def test_all_presets_valid(app):
    w = ArenaWindow()
    for name in PRESETS:
        w.preset.setCurrentText(name)
        w.load_preset()
        w.update_plan()
        assert "bots" in w.plan.text() and w.run_btn.isEnabled(), (name, w.plan.text())
    w.close()
