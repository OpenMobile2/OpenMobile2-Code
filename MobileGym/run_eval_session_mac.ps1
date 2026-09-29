#!/usr/bin/env bash
# Mac launcher. This machine does not have pwsh; the real script is bash.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_eval_session_mac.sh" "$@"
