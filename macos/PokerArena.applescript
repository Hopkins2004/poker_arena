-- Poker Arena: double-click launcher (an AppleScript applet, built by macos/build_app.sh).
-- It starts Contents/Resources/launch.sh detached, which opens the PySide6 window.
-- Being a signed applet, macOS shows its standard one-time "access your Documents folder"
-- prompt when the project lives in ~/Documents.
on run
	set launcher to (POSIX path of (path to me)) & "Contents/Resources/launch.sh"
	try
		do shell script "/bin/bash " & quoted form of launcher & " >/dev/null 2>&1 &"
	on error errMsg
		display alert "Poker Arena could not start" message errMsg as critical
	end try
end run
