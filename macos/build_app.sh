#!/bin/bash
# Builds "Poker Arena.app" (double-click launcher) at the repository root.
# Usage: macos/build_app.sh [output.app]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/../Poker Arena.app}"
rm -rf "$OUT"
/usr/bin/osacompile -o "$OUT" "$HERE/PokerArena.applescript"
cp "$HERE/launch.sh" "$OUT/Contents/Resources/launch.sh"
chmod +x "$OUT/Contents/Resources/launch.sh"
/usr/bin/plutil -replace CFBundleName -string "Poker Arena" "$OUT/Contents/Info.plist"
/usr/bin/plutil -replace CFBundleIdentifier -string "local.poker-arena.launcher" "$OUT/Contents/Info.plist"
# Re-sign (ad hoc) after adding the launcher so the bundle signature stays valid.
/usr/bin/codesign --force --deep --sign - "$OUT" >/dev/null 2>&1 || true
echo "built $OUT"
