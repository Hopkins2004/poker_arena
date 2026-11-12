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
    assert "7</b> bots" in w.plan.text() and "35</b> tables" in w.plan.text()
    assert w.run_btn.isEnabled() and not w.stop_btn.isEnabled()
    # A broken params cell disables Run with a readable message.
    w.players.item(0, 3).setText("{oops")
    w.update_plan()
    assert "not valid JSON" in w.plan.text() and not w.run_btn.isEnabled()
    w.close()


def test_python_champions_full_run(app):
    w = ArenaWindow()
    w.preset.setCurrentText("Python champions (3 bots)")
    w.load_preset()
    assert w.players.rowCount() == 3
    w.hands.setValue(300)
    w.rounds.setValue(2)
    w.processes.setValue(2)
    w.start_run()
    assert w.stop_btn.isEnabled() and not w.run_btn.isEnabled()
    pump(app, lambda: w.job is None)
    assert w.progress.format() == "Done"
    assert w.board.rowCount() == 3
    types = {w.board.item(r, 2).text() for r in range(3)}
    assert types == {"Python"}
    assert all(w.board.item(r, 11).text() == "0" for r in range(3)), "no illegal actions"
    assert w.run_btn.isEnabled()
    w.close()


def test_stop_button_nukes_mixed_run(app):
    before = baseline()
    w = ArenaWindow()
    w.preset.setCurrentText("Rust vs Python champions (6 bots)")
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
    w.preset.setCurrentText("Python trio (3 bots)")
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
