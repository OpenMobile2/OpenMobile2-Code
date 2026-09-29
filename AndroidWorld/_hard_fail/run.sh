#!/usr/bin/env bash
# Self-contained hard-fail eval. Delete this whole directory to undo.
# Swap groups.json to change the 全做错 / 易做错 split.
#
# Usage:
#   ./run.sh [never|lt50|hard] --runtime qwen35_session --checkpoint_dir runs/foo ...
#   ./run.sh summarize runs/foo/result_*.csv
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$DIR/run_subset.py" "$@"
