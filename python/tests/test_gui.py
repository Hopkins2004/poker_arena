"""Offscreen tests of the desktop window: lineup, Add bot, a full run, the hard stop, closing."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from pokerarena.gui import ArenaWindow  # noqa: E402
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


def cell(w, name, col):
    return next(w.players.item(r, col).text() for r in range(w.players.rowCount())
                if w.players.item(r, 0).text() == name)


def test_default_lineup_is_the_anchored_bots(app):
    w = ArenaWindow()
    w.update_plan()
    assert w.players.names() == ["random", "station", "tag"]
    assert "3</b> bots (3 anchored)" in w.plan.text()
    assert not hasattr(w, "min_appearances") and not hasattr(w, "preset")
    headers = [w.players.horizontalHeaderItem(c).text() for c in range(w.players.columnCount())]
    assert headers == ["Name", "Bot", "Anchor Elo", "Start Elo", ""]
    assert not any("Params" in h or "JSON" in h for h in headers)
    assert cell(w, "tag", w.players.ANCHOR) == "1300"
    assert cell(w, "station", w.players.START) == "400  (anchor)"
    w.close()


def test_add_bot_button_adds_from_the_bots_folder(app):
    w = ArenaWindow()
    choices = [w.catalog_box.itemData(i) for i in range(w.catalog_box.count())]
    assert choices == [b.kind for b in w.catalog] and len(choices) == 3
    w.catalog_box.setCurrentIndex(choices.index("python:bots.NLH.tag.TightAggressive"))
    w.add_btn.click()
    w.add_btn.click()
    assert w.players.names() == ["random", "station", "tag", "tag_2", "tag_3"]
    assert cell(w, "tag_2", w.players.ANCHOR) == "1300"
    w.update_plan()
    assert "5</b> bots" in w.plan.text() and w.run_btn.isEnabled()
    # Duplicate names are rejected with a readable message.
    w.players.item(4, 0).setText("tag")
    w.update_plan()
    assert "names must be unique" in w.plan.text() and not w.run_btn.isEnabled()
    w.reset_lineup()
    assert w.players.names() == ["random", "station", "tag"]
    w.close()


def test_full_run_with_replays(app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # replays land in ./hand_replays
    w = ArenaWindow()
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
    ill = cols.index("Illegal")
    assert all(w.board.item(r, ill).text() == "0" for r in range(3)), "no illegal actions"
    elos = sorted(w.board.item(r, 2).text() for r in range(3))
    assert elos == ["1300.0", "400.0", "850.0"]
    replay_dirs = list((tmp_path / "hand_replays").iterdir())
    assert len(replay_dirs) == 1
    assert sorted(p.name for p in replay_dirs[0].iterdir()) == ["random.txt", "station.txt", "tag.txt"]
    assert "Hand replays" in w.saved.text()
    assert w.run_btn.isEnabled()
    w.close()


def test_save_toggle_reuses_saved_elo(app, tmp_path):
    lb = tmp_path / "lb.json"
    lb.write_text('{"entries": {"newbie": {"rating": 1612.5, "hands": 1, "net_bb": 0, '
                  '"tables": 1, "runs": 1, "last_run": "x"}, "tag": {"rating": 1777, "hands": 1, '
                  '"net_bb": 0, "tables": 1, "runs": 1, "last_run": "x"}}}')
    w = ArenaWindow()
    w.base_cfg.setdefault("run", {})["leaderboard"] = str(lb)
    # An unanchored player (renamed copy of a bot) to show saved-rating reuse.
    w.add_btn.click()
    row = w.players.rowCount() - 1
    w.players.item(row, 0).setText("newbie")
    w.players.item(row, 0).setData(w.players.ANCHOR_ROLE, None)
    w.update_plan()
    assert cell(w, "newbie", w.players.START) == "1500  (initial)"  # save off: fresh ratings
    w.save.setChecked(True)
    w.update_plan()
    assert cell(w, "newbie", w.players.START) == "1612  (saved)"
    assert cell(w, "tag", w.players.START) == "1300  (anchor)"  # anchors win over saved ratings
    assert "1 continue from saved Elo" in w.plan.text()
    w.close()


def test_stop_button_nukes_the_run(app):
    before = baseline()
    w = ArenaWindow()
    w.hands.setValue(5_000_000)
    w.processes.setValue(3)
    w.start_run()
    job = w.job
    pump(app, lambda: job.progress().hands_done > 5_000)
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
    w.hands.setValue(5_000_000)
    w.start_run()
    job = w.job
    pump(app, lambda: job.progress().hands_done > 1000)
    w.close()
    assert job.state == "stopped" and job.wait(0)
    assert_no_leak(before)
