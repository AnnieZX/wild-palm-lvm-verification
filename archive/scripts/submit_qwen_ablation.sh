#!/usr/bin/env bash
# Deprecated: use jobs/submit_ablation.sh
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/jobs/submit_ablation.sh" "$@"
