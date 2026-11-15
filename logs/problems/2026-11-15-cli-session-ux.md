# CLI session UX

Problems and changes planned:
- Ctrl+O leaves viewed logs in the primary terminal: use a temporary alternate-screen log viewer, restore on collapse and interruption.
- Missing `flexsds clear modal`: remove current local Modal credentials, preserve other profiles, prevent inherited tokens reconnecting automatically.
- Verified endpoint lacks next actions: display endpoint details and copyable chat command; offer watch logs or streaming chat while development serving remains alive.

Validation: targeted terminal, credential and serving-session tests; no cloud deployment required.

Implemented on `develop`:
- Ctrl+O uses a temporary alternate screen, with scrolling and restoration on collapse, close and interruption. Full logs remain in the private log file.
- `flexsds clear modal` removes stored FlexSDS tokens and only the selected native Modal profile's token pair. Other profiles/providers remain. A local logout marker blocks stale inherited shell tokens; fresh FlexSDS login or native `modal setup` restores access.
- Verified serving prints endpoint details and a copyable chat command, then offers Watch log / Realtime chat. Chat releases the progress keyboard listener and `/exit` returns to the menu without terminating development serving. Persistent deployment watching follows `modal app logs --follow` through the same temporary viewer.
- Updated the runbook and isolated wizard tests from real user calibration profiles.

Validation: 115 targeted tests passed, including POSIX PTY Ctrl+O/terminal restoration, local native Modal SDK configuration cleanup using temporary files, stale-token/relogin checks, streaming-chat handoff and CLI help. `git diff --check` passed. No real credentials cleared and no GPU/cloud app launched during this change. Persistent cloud log following was checked against installed Modal 1.6.3 CLI help, without a live deployment test.
