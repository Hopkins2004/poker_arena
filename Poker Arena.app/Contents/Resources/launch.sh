#!/bin/bash
# Started by the "Poker Arena" applet. Opens the Poker Arena window (pokerarena.gui) for the
# repository that contains the app; results/ and leaderboard.json are written there.
# Output is appended to <repo>/.poker_arena.log.
APP="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="$(dirname "$APP")"
PY="$ROOT/python/.venv/bin/python"

alert() {
  /usr/bin/osascript - "$1" "$2" >/dev/null 2>&1 <<'OSA'
on run argv
	display alert (item 1 of argv) message (item 2 of argv) as critical
end run
OSA
}

if ! cd "$ROOT" 2>/dev/null || ! ls python >/dev/null 2>&1; then
  alert "Poker Arena cannot open its folder" "macOS did not allow access to $ROOT.

Open System Settings > Privacy & Security > Files & Folders, allow \"Poker Arena\" to access the Documents folder, then double-click Poker Arena again."
  exit 1
fi
LOG="$ROOT/.poker_arena.log"
{ echo; echo "=== $(date) launching Poker Arena"; } >>"$LOG"
if [ ! -x "$PY" ] || ! "$PY" -c "import pokerarena._native, PySide6.QtWidgets" >>"$LOG" 2>&1; then
  alert "Poker Arena is not set up yet" "Build it once from Terminal in $ROOT/python:

python3 -m venv .venv
.venv/bin/pip install maturin numpy pytest PySide6-Essentials
.venv/bin/maturin develop --release

Details: $LOG"
  exit 1
fi
# -c (not -m): workers use the "spawn" start method and re-import the main module; with -c
# there is none, so worker processes never load the GUI toolkit.
exec "$PY" -c "from pokerarena.gui import main; raise SystemExit(main())" >>"$LOG" 2>&1
